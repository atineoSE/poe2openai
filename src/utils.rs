use crate::poe_client::PoeClientWrapper;
use crate::types::{Config, ImageUrlContent, Message, OpenAiContent, OpenAiContentItem};
use crate::types::{OpenAIError, OpenAIErrorResponse};
use base64::prelude::*;
use nanoid::nanoid;
use poe_api_process::FileUploadRequest;
use salvo::http::StatusCode;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use tiktoken_rs::o200k_base;
use tracing::{debug, error, info, warn};

// Process files/images in messages
pub async fn process_message_images(
    poe_client: &PoeClientWrapper,
    messages: &mut [Message],
) -> Result<(), Box<dyn std::error::Error>> {
    // Collect URLs needing processing
    let mut external_urls = Vec::new();
    let mut data_urls = Vec::new();
    let mut url_indices = Vec::new();
    let mut data_url_indices = Vec::new();
    let mut temp_files: Vec<PathBuf> = Vec::new();

    // Collect all URLs needing processing in messages
    for (msg_idx, message) in messages.iter().enumerate() {
        if let OpenAiContent::Multi(items) = &message.content {
            for (item_idx, item) in items.iter().enumerate() {
                if let OpenAiContentItem::ImageUrl { image_url } = item {
                    if image_url.url.starts_with("data:") {
                        // Process data URL
                        debug!("🔍 Data URL detected");
                        data_urls.push(image_url.url.clone());
                        data_url_indices.push((msg_idx, item_idx));
                    } else if !is_poe_cdn_url(&image_url.url) {
                        // Process external URLs needing upload
                        debug!("🔍 External URL needing upload detected: {}", image_url.url);
                        external_urls.push(image_url.url.clone());
                        url_indices.push((msg_idx, item_idx));
                    }
                }
            }
        }
    }

    // Process external URLs
    if !external_urls.is_empty() {
        debug!("🔄 Preparing to process {} external URLs", external_urls.len());

        // Split external URLs into cache hits and misses
        let mut urls_to_upload = Vec::new();
        let mut urls_indices_to_upload = Vec::new();

        for (idx, (msg_idx, item_idx)) in url_indices.iter().enumerate() {
            let url = &external_urls[idx];

            // Check cache
            if let Some((poe_url, _)) = crate::cache::get_cached_url(url) {
                debug!("✅ URL cache hit: {} -> {}", url, poe_url);

                if let OpenAiContent::Multi(items) = &mut messages[*msg_idx].content {
                    if let OpenAiContentItem::ImageUrl { image_url } = &mut items[*item_idx] {
                        debug!("🔄 Replacing URL from cache: {}", poe_url);
                        image_url.url = poe_url;
                    }
                }
            } else {
                // Cache miss, need upload
                debug!("❌ URL cache miss: {}", url);
                urls_to_upload.push(url.clone());
                urls_indices_to_upload.push((*msg_idx, *item_idx));
            }
        }

        // Upload uncached URLs
        if !urls_to_upload.is_empty() {
            debug!("🔄 Uploading {} uncached URLs", urls_to_upload.len());

            let upload_requests: Vec<FileUploadRequest> = urls_to_upload
                .iter()
                .map(|url| FileUploadRequest::RemoteFile {
                    download_url: url.clone(),
                })
                .collect();

            match poe_client.client.upload_files_batch(upload_requests).await {
                Ok(responses) => {
                    debug!("✅ Successfully uploaded {} external URLs", responses.len());

                    // Update cache and save URL mapping
                    for (idx, ((msg_idx, item_idx), response)) in urls_indices_to_upload
                        .iter()
                        .zip(responses.iter())
                        .enumerate()
                    {
                        let original_url = &urls_to_upload[idx];

                        // Estimate size (default 1MB, can optimize later)
                        let size_bytes = 1024 * 1024;

                        // Add to cache
                        crate::cache::cache_url(original_url, &response.attachment_url, size_bytes);

                        if let OpenAiContent::Multi(items) = &mut messages[*msg_idx].content {
                            if let OpenAiContentItem::ImageUrl { image_url } = &mut items[*item_idx]
                            {
                                debug!(
                                    "🔄 URL replaced | Original: {} | Poe: {}",
                                    image_url.url, response.attachment_url
                                );
                                image_url.url = response.attachment_url.clone();
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("❌ Failed to upload external URL: {}", e);
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to upload external URL: {}", e),
                    )));
                }
            }
        }
    }

    // Process data URLs
    if !data_urls.is_empty() {
        debug!("🔄 Preparing to process {} data URLs", data_urls.len());

        // Split into cache hits and misses
        let mut data_to_upload = Vec::new();
        let mut data_indices_to_upload = Vec::new();
        let mut data_hashes = Vec::new();

        for (idx, (msg_idx, item_idx)) in data_url_indices.iter().enumerate() {
            let data_url = &data_urls[idx];
            let hash = hash_base64_content(data_url);

            debug!("🔍 Calculating data URL hash | Hash prefix: {}...", &hash[..8]);

            // Check cache
            if let Some((poe_url, _)) = crate::cache::get_cached_base64(&hash) {
                debug!("✅ Base64 cache hit | Hash: {}... -> {}", &hash[..8], poe_url);

                if let OpenAiContent::Multi(items) = &mut messages[*msg_idx].content {
                    if let OpenAiContentItem::ImageUrl { image_url } = &mut items[*item_idx] {
                        debug!("🔄 Replacing base64 from cache | URL: {}", poe_url);
                        image_url.url = poe_url;
                    }
                }
            } else {
                // Cache miss, need upload
                debug!("❌ Base64 cache miss | Hash: {}...", &hash[..8]);
                data_to_upload.push(data_url.clone());
                data_indices_to_upload.push((idx, (*msg_idx, *item_idx)));
                data_hashes.push(hash);
            }
        }

        // Upload uncached data URLs
        if !data_to_upload.is_empty() {
            let mut upload_requests = Vec::new();

            // Convert data URL to temporary file
            for data_url in data_to_upload.iter() {
                // Extract MIME type from data URL
                let mime_type = if data_url.starts_with("data:") {
                    let parts: Vec<&str> = data_url.split(";base64,").collect();
                    if !parts.is_empty() {
                        let mime_part = parts[0].trim_start_matches("data:");
                        debug!("🔍 Extracted MIME type: {}", mime_part);
                        Some(mime_part.to_string())
                    } else {
                        None
                    }
                } else {
                    None
                };

                match handle_data_url_to_temp_file(data_url) {
                    Ok(file_path) => {
                        debug!("📄 Temporary file created successfully: {}", file_path.display());
                        upload_requests.push(FileUploadRequest::LocalFile {
                            file: file_path.to_string_lossy().to_string(),
                            mime_type,
                        });
                        temp_files.push(file_path);
                    }
                    Err(e) => {
                        error!("❌ Failed to process data URL: {}", e);
                        // Clean up created temporary file
                        for path in &temp_files {
                            if let Err(e) = fs::remove_file(path) {
                                warn!("⚠️ Failed to delete temporary file {}: {}", path.display(), e);
                            }
                        }
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Failed to process data URL: {}", e),
                        )));
                    }
                }
            }

            // Upload temporary file
            if !upload_requests.is_empty() {
                match poe_client.client.upload_files_batch(upload_requests).await {
                    Ok(responses) => {
                        debug!("✅ Successfully uploaded {} temporary files", responses.len());

                        // Update cache and save URL mapping
                        for (idx, response) in responses.iter().enumerate() {
                            let (_, (msg_idx, item_idx)) = data_indices_to_upload[idx];
                            let hash = &data_hashes[idx];
                            let data_url = &data_to_upload[idx];

                            // Estimate size
                            let size = crate::cache::estimate_base64_size(data_url);

                            // Add to cache
                            crate::cache::cache_base64(hash, &response.attachment_url, size);

                            debug!(
                                "🔄 Mapping base64 hash to Poe URL | Hash: {}... -> {}",
                                &hash[..8],
                                response.attachment_url
                            );

                            if let OpenAiContent::Multi(items) = &mut messages[msg_idx].content {
                                if let OpenAiContentItem::ImageUrl { image_url } =
                                    &mut items[item_idx]
                                {
                                    debug!("🔄 Replacing data URL | Poe: {}", response.attachment_url);
                                    image_url.url = response.attachment_url.clone();
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("❌ Failed to upload temporary file: {}", e);
                        // Clean up temporary file
                        for path in &temp_files {
                            if let Err(e) = fs::remove_file(path) {
                                warn!("⚠️ Failed to delete temporary file {}: {}", path.display(), e);
                            }
                        }
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to upload temporary file: {}", e),
                        )));
                    }
                }
            }

            // Clean up temporary files
            for path in &temp_files {
                if let Err(e) = fs::remove_file(path) {
                    warn!("⚠️ Failed to delete temporary file {}: {}", path.display(), e);
                } else {
                    debug!("🗑️ Temporary file deleted: {}", path.display());
                }
            }
        }
    }

    // Process Poe CDN links in AI response, add to user message image_url
    if messages.len() >= 2 {
        // Find last AI response and user message
        let last_bot_idx = messages
            .iter()
            .enumerate()
            .filter(|(_, msg)| msg.role == "assistant")
            .last()
            .map(|(i, _)| i);
        let last_user_idx = messages
            .iter()
            .enumerate()
            .filter(|(_, msg)| msg.role == "user")
            .last()
            .map(|(i, _)| i);

        if let (Some(bot_idx), Some(user_idx)) = (last_bot_idx, last_user_idx) {
            // Extract Poe CDN links from AI response
            let poe_cdn_urls = extract_poe_cdn_urls_from_message(&messages[bot_idx]);
            if !poe_cdn_urls.is_empty() {
                debug!(
                    "🔄 Extracted {} Poe CDN links from AI response, adding to user message",
                    poe_cdn_urls.len()
                );
                // Add these links to user message image_url
                let user_msg = &mut messages[user_idx];
                match &mut user_msg.content {
                    OpenAiContent::Text(text) => {
                        // Convert text message to multipart message with image
                        let mut items = Vec::new();
                        items.push(OpenAiContentItem::Text { text: text.clone() });
                        for url in poe_cdn_urls {
                            items.push(OpenAiContentItem::ImageUrl {
                                image_url: ImageUrlContent { url },
                            });
                        }
                        user_msg.content = OpenAiContent::Multi(items);
                    }
                    OpenAiContent::Multi(items) => {
                        // Already multipart message, directly add image
                        for url in poe_cdn_urls {
                            items.push(OpenAiContentItem::ImageUrl {
                                image_url: ImageUrlContent { url },
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// Get plain text content from OpenAIContent
pub fn get_text_from_openai_content(content: &OpenAiContent) -> String {
    match content {
        OpenAiContent::Text(s) => s.clone(),
        OpenAiContent::Multi(items) => {
            let mut text_parts = Vec::new();
            for item in items {
                if let OpenAiContentItem::Text { text } = item {
                    // Use serde_json::to_string to handle special characters
                    match serde_json::to_string(text) {
                        Ok(processed_text) => {
                            // Remove quotes added by serde_json::to_string
                            let processed_text = processed_text.trim_matches('"').to_string();
                            // Replace JSON-escaped quotes (\") with normal quotes (")
                            let processed_text = processed_text.replace("\\\"", "\"");
                            text_parts.push(processed_text);
                        }
                        Err(_) => {
                            // If serialization failed, use original text
                            text_parts.push(text.clone());
                        }
                    }
                }
            }
            text_parts.join("\n")
        }
    }
}

// Check if URL is Poe CDN link
pub fn is_poe_cdn_url(url: &str) -> bool {
    url.starts_with("https://pfst.cf2.poecdn.net")
}

// Extract Poe CDN links from message
pub fn extract_poe_cdn_urls_from_message(message: &Message) -> Vec<String> {
    let mut urls = Vec::new();
    match &message.content {
        OpenAiContent::Multi(items) => {
            for item in items {
                if let OpenAiContentItem::ImageUrl { image_url } = item {
                    if is_poe_cdn_url(&image_url.url) {
                        urls.push(image_url.url.clone());
                    }
                } else if let OpenAiContentItem::Text { text } = item {
                    // Extract Poe CDN URL from text
                    extract_urls_from_markdown(text, &mut urls);
                }
            }
        }
        OpenAiContent::Text(text) => {
            // Extract Poe CDN URL from plain text message
            extract_urls_from_markdown(text, &mut urls);
        }
    }
    urls
}

// Helper function to extract Poe CDN URLs from Markdown text
fn extract_urls_from_markdown(text: &str, urls: &mut Vec<String>) {
    // Extract Markdown image URLs: ![alt](url)
    let re_md_img = regex::Regex::new(r"!\[.*?\]\((https?://[^\s)]+)\)").unwrap();
    for cap in re_md_img.captures_iter(text) {
        if let Some(url) = cap.get(1) {
            let url_str = url.as_str();
            if is_poe_cdn_url(url_str) {
                urls.push(url_str.to_string());
            }
        }
    }
    // Process URLs appearing directly
    for word in text.split_whitespace() {
        if is_poe_cdn_url(word) {
            urls.push(word.to_string());
        }
    }
}

// Process base64 data URLs, convert to temporary file
pub fn handle_data_url_to_temp_file(data_url: &str) -> Result<PathBuf, String> {
    // 1. Validate data URL format
    if !data_url.starts_with("data:") {
        return Err("Invalid data URL format".to_string());
    }
    // 2. Split MIME type and base64 data
    let parts: Vec<&str> = data_url.split(";base64,").collect();
    if parts.len() != 2 {
        return Err("Invalid data URL format: missing base64 separator".to_string());
    }
    // 3. Extract MIME type
    let mime_type = parts[0].strip_prefix("data:").unwrap_or(parts[0]);
    debug!("🔍 Extracted MIME type: {}", mime_type);
    // 4. Determine file extension from MIME type
    let file_ext = mime_type_to_extension(mime_type).unwrap_or("bin");
    debug!("📄 Using file extension: {}", file_ext);
    // 5. Decode base64 data (using BASE64_STANDARD only)
    let base64_data = parts[1];
    debug!("🔢 Base64 data length: {}", base64_data.len());
    let decoded = match BASE64_STANDARD.decode(base64_data) {
        Ok(data) => {
            debug!("✅ Base64 decoding successful | Data size: {} bytes", data.len());
            data
        }
        Err(e) => {
            error!("❌ Base64 decoding failed: {}", e);
            return Err(format!("Base64 decoding failed: {}", e));
        }
    };
    // 6. Create temporary file
    let temp_dir = std::env::temp_dir();
    let file_name = format!("poe2openai_{}.{}", nanoid!(16), file_ext);
    let file_path = temp_dir.join(&file_name);
    // 7. Write data to temporary file
    match fs::write(&file_path, &decoded) {
        Ok(_) => {
            debug!("✅ Successfully wrote to temporary file: {}", file_path.display());
            Ok(file_path)
        }
        Err(e) => {
            error!("❌ Failed to write to temporary file: {}", e);
            Err(format!("Failed to write to temporary file: {}", e))
        }
    }
}

// Get file extension from MIME type
fn mime_type_to_extension(mime_type: &str) -> Option<&str> {
    match mime_type {
        "image/jpeg" | "image/jpg" => Some("jpeg"),
        "image/png" => Some("png"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        "image/svg+xml" => Some("svg"),
        "image/bmp" => Some("bmp"),
        "image/tiff" => Some("tiff"),
        "application/pdf" => Some("pdf"),
        "text/plain" => Some("txt"),
        "text/csv" => Some("csv"),
        "application/json" => Some("json"),
        "application/xml" | "text/xml" => Some("xml"),
        "application/zip" => Some("zip"),
        "application/x-tar" => Some("tar"),
        "application/x-gzip" => Some("gz"),
        "audio/mpeg" => Some("mp3"),
        "audio/wav" => Some("wav"),
        "audio/ogg" => Some("ogg"),
        "video/mp4" => Some("mp4"),
        "video/mpeg" => Some("mpeg"),
        "video/quicktime" => Some("mov"),
        _ => None,
    }
}

pub fn convert_poe_error_to_openai(
    error_text: &str,
    allow_retry: bool,
) -> (StatusCode, OpenAIErrorResponse) {
    debug!(
        "🔄 Conversion error response | Error text: {}, Retry allowed: {}",
        error_text, allow_retry
    );
    let (status, error_type, code) = if error_text.contains("Internal server error") {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "internal_error",
        )
    } else if error_text.contains("rate limit") {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_exceeded",
            "rate_limit_exceeded",
        )
    } else if error_text.contains("Invalid token") || error_text.contains("Unauthorized") {
        (StatusCode::UNAUTHORIZED, "invalid_auth", "invalid_api_key")
    } else if error_text.contains("Bot does not exist") {
        (StatusCode::NOT_FOUND, "model_not_found", "model_not_found")
    } else {
        (StatusCode::BAD_REQUEST, "invalid_request", "bad_request")
    };
    debug!(
        "📋 Error conversion result | Status code: {} | Error type: {}",
        status.as_u16(),
        error_type
    );
    (
        status,
        OpenAIErrorResponse {
            error: OpenAIError {
                message: error_text.to_string(),
                r#type: error_type.to_string(),
                code: code.to_string(),
                param: None,
            },
        },
    )
}

pub fn format_bytes_length(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

pub fn get_config_path(filename: &str) -> PathBuf {
    let config_dir = std::env::var("CONFIG_DIR").unwrap_or_else(|_| "./".to_string());
    let mut path = PathBuf::from(config_dir);
    path.push(filename);
    path
}

pub fn load_config_from_yaml() -> Result<Config, String> {
    let path_str = "models.yaml";
    let path = get_config_path(path_str);
    if path.exists() {
        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_yaml::from_str::<Config>(&contents) {
                Ok(config) => {
                    info!("✅ Successfully read and parsed {}", path_str);
                    Ok(config)
                }
                Err(e) => {
                    error!("❌ Failed to parse {}: {}", path_str, e);
                    Err(format!("Failed to parse {}: {}", path_str, e))
                }
            },
            Err(e) => {
                error!("❌ Failed to read {}: {}", path_str, e);
                Err(format!("Failed to read {}: {}", path_str, e))
            }
        }
    } else {
        debug!("⚠️  {} does not exist, using default empty config", path_str);
        // Return a default Config indicating file doesn't exist or can't be read
        Ok(Config {
            enable: Some(false),
            models: std::collections::HashMap::new(),
            custom_models: None,
        })
    }
}

/// Calculate token count for text
pub fn count_tokens(text: &str) -> u32 {
    let bpe = match o200k_base() {
        Ok(bpe) => bpe,
        Err(e) => {
            error!("❌ Failed to initialize BPE encoder: {}", e);
            return 0;
        }
    };
    let tokens = bpe.encode_with_special_tokens(text);
    tokens.len() as u32
}

/// Calculate token count for message list
pub fn count_message_tokens(messages: &[Message]) -> u32 {
    let mut total_tokens = 0;
    for message in messages {
        // Base token count per message (role tokens etc.)
        total_tokens += 4; // Base overhead per message
        // Calculate content token count
        let content_text = get_text_from_openai_content(&message.content);
        total_tokens += count_tokens(&content_text);
    }
    // Add message format overhead tokens
    total_tokens += 2; // Start and end tokens for message format
    total_tokens
}

/// Calculate token count for completion content
pub fn count_completion_tokens(completion: &str) -> u32 {
    count_tokens(completion)
}

/// Calculate SHA256 hash of base64 string
pub fn hash_base64_content(base64_str: &str) -> String {
    // Extract pure base64 part, remove MIME type prefix
    let base64_data = match base64_str.split(";base64,").nth(1) {
        Some(data) => data,
        None => base64_str, // If no separator, use whole string
    };

    let start = &base64_data[..base64_data.len().min(1024)];
    let end = if base64_data.len() > 2048 {
        // Ensure sufficient length
        &base64_data[base64_data.len() - 1024..]
    } else if base64_data.len() > 1024 {
        &base64_data[1024..] // If length between 1024-2048, use remaining part
    } else {
        "" // If less than 1024, use start only
    };

    // Combine head and tail data
    let combined = format!("{}{}", start, end);

    // Calculate SHA256 hash
    let mut hasher = Sha256::new();
    hasher.update(combined.as_bytes());
    let result = hasher.finalize();

    // Record hash calculation info for debugging
    let hash = format!("{:x}", result);
    debug!(
        "🔢 Calculating base64 hash | Data length: {} | Calculated length: {} | Hash prefix: {}...",
        base64_data.len(),
        start.len() + end.len(),
        &hash[..8]
    );

    hash
}
