use salvo::http::{HeaderValue, Method, StatusCode, header};
use salvo::prelude::*;
use tracing::{debug, info};

/// Check if header is safe
fn is_safe_header(header: &str) -> bool {
    let header_lower = header.trim().to_lowercase();

    // Exclude empty strings
    if header_lower.is_empty() {
        return false;
    }

    // Blacklist: explicit malicious headers
    if matches!(header_lower.as_str(), "cookie" | "set-cookie") {
        return false;
    }

    // Whitelist: allowed header patterns
    // 1. X- prefixed custom headers (like X-Stainless-*)
    // 2. Standard HTTP headers
    header_lower.starts_with("x-")
        || matches!(
            header_lower.as_str(),
            "accept"
                | "accept-encoding"
                | "accept-language"
                | "authorization"
                | "cache-control"
                | "connection"
                | "content-type"
                | "user-agent"
                | "referer"
                | "origin"
                | "pragma"
                | "sec-fetch-dest"
                | "sec-fetch-mode"
                | "sec-fetch-site"
        )
}

/// Parse client request headers and perform security filtering
fn parse_requested_headers(req: &Request) -> Vec<String> {
    req.headers()
        .get(header::ACCESS_CONTROL_REQUEST_HEADERS)
        .and_then(|h| h.to_str().ok())
        .map(|headers_str| {
            headers_str
                .split(',')
                .map(|h| h.trim().to_string())
                .filter(|h| !h.is_empty() && is_safe_header(h))
                .collect()
        })
        .unwrap_or_default()
}

#[handler]
pub async fn cors_middleware(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
    ctrl: &mut FlowCtrl,
) {
    // Get Origin header from request
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("null");

    // Record request origin for debugging
    debug!("📡 Received request from Origin: {}", origin);

    // Set CORS headers
    match HeaderValue::from_str(origin) {
        Ok(origin_value) => {
            res.headers_mut()
                .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin_value);
        }
        Err(e) => {
            debug!("⚠️ Invalid Origin header: {}, Error: {}", origin, e);
            res.headers_mut().insert(
                header::ACCESS_CONTROL_ALLOW_ORIGIN,
                HeaderValue::from_static("null"),
            );
        }
    }

    res.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
        HeaderValue::from_static("true"),
    );

    // Add Vary header to all responses to indicate response varies by Origin header
    res.headers_mut()
        .insert(header::VARY, HeaderValue::from_static("Origin"));

    // If it's an OPTIONS request, handle it and stop further processing
    if req.method() == Method::OPTIONS {
        handle_preflight_request(req, res);
        ctrl.skip_rest();
    } else {
        // Not an OPTIONS request, continue normal processing
        ctrl.call_next(req, depot, res).await;
    }
}

/// Special handler for CORS preflight requests
fn handle_preflight_request(req: &Request, res: &mut Response) {
    info!("🔍 Processing OPTIONS preflight request: {}", req.uri());

    // Set standard CORS preflight response headers
    res.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS, PUT, DELETE, PATCH, HEAD"),
    );

    // Base hardcoded headers (maintain backward compatibility)
    let base_headers = vec![
        "Authorization",
        "Content-Type",
        "User-Agent",
        "Accept",
        "Origin",
        "X-Requested-With",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "Accept-Encoding",
        "Accept-Language",
        "Cache-Control",
        "Connection",
        "Referer",
        "Sec-Fetch-Dest",
        "Sec-Fetch-Mode",
        "Sec-Fetch-Site",
        "Pragma",
        "X-Api-Key",
    ];

    // Parse client request dynamic headers
    let dynamic_headers = parse_requested_headers(req);

    // Merge base headers and dynamic headers
    let mut all_headers = base_headers.clone();
    for header in &dynamic_headers {
        if !all_headers
            .iter()
            .any(|h| h.to_lowercase() == header.to_lowercase())
        {
            all_headers.push(header);
        }
    }

    // Build final headers string
    let headers_str = all_headers.join(", ");

    // Record debug information
    if !dynamic_headers.is_empty() {
        info!("➕ Dynamic added headers: {:?}", dynamic_headers);
    }
    info!("📋 Final allowed headers: {}", headers_str);

    // Set Access-Control-Allow-Headers
    match HeaderValue::from_str(&headers_str) {
        Ok(headers_value) => {
            res.headers_mut()
                .insert(header::ACCESS_CONTROL_ALLOW_HEADERS, headers_value);
        }
        Err(e) => {
            // Fallback: use base headers if dynamic headers fail
            debug!("⚠️ Dynamic headers setup failed: {}, using base headers", e);
            res.headers_mut().insert(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                HeaderValue::from_static(
                    "Authorization, Content-Type, User-Agent, Accept, Origin, \
                    X-Requested-With, Access-Control-Request-Method, \
                    Access-Control-Request-Headers, Accept-Encoding, Accept-Language, \
                    Cache-Control, Connection, Referer, Sec-Fetch-Dest, Sec-Fetch-Mode, \
                    Sec-Fetch-Site, Pragma, X-Api-Key",
                ),
            );
        }
    }

    res.headers_mut().insert(
        header::ACCESS_CONTROL_MAX_AGE,
        HeaderValue::from_static("3600"),
    );

    // Add Vary header to indicate response varies by these request headers
    res.headers_mut().insert(
        header::VARY,
        HeaderValue::from_static("Access-Control-Request-Method, Access-Control-Request-Headers"),
    );

    // Set correct status code: 204 No Content
    res.status_code(StatusCode::NO_CONTENT);
}
