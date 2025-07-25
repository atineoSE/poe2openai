use crate::{cache::get_cached_config, types::*, utils::get_text_from_openai_content};
use futures_util::Stream;
use poe_api_process::types::Attachment;
use poe_api_process::{ChatMessage, ChatRequest, ChatResponse, PoeClient, PoeError};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info};

pub struct PoeClientWrapper {
    pub client: PoeClient, // Made public for external access
    _model: String,
}

impl PoeClientWrapper {
    pub fn new(model: &str, access_key: &str) -> Self {
        info!("🔑 Initializing POE client | Model: {}", model);
        Self {
            client: PoeClient::new(model, access_key),
            _model: model.to_string(),
        }
    }
    pub async fn stream_request(
        &self,
        chat_request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponse, PoeError>> + Send>>, PoeError> {
        let start_time = Instant::now();
        debug!(
            "📤 Sending streaming request | Message count: {} | Temperature: {:?}",
            chat_request.query.len(),
            chat_request.temperature
        );
        let result = self.client.stream_request(chat_request).await;
        match &result {
            Ok(_) => {
                let duration = start_time.elapsed();
                info!(
                    "✅ Streaming request established successfully | Duration: {}",
                    crate::utils::format_duration(duration)
                );
            }
            Err(e) => {
                let duration = start_time.elapsed();
                error!(
                    "❌ Streaming request failed | Error: {} | Duration: {}",
                    e,
                    crate::utils::format_duration(duration)
                );
            }
        }
        result
    }
}

// Function to convert OpenAI message format to Poe message format
fn openai_message_to_poe(msg: &Message, role_override: Option<String>) -> ChatMessage {
    let mut attachments: Vec<Attachment> = vec![];
    let mut texts: Vec<String> = vec![];

    match &msg.content {
        OpenAiContent::Text(s) => {
            texts.push(s.clone());
        }
        OpenAiContent::Multi(arr) => {
            for item in arr {
                match item {
                    OpenAiContentItem::Text { text } => texts.push(text.clone()),
                    OpenAiContentItem::ImageUrl { image_url } => {
                        debug!("🖼️  Processing image URL: {}", image_url.url);
                        attachments.push(Attachment {
                            url: image_url.url.clone(),
                            content_type: None,
                        });
                    }
                }
            }
        }
    }

    let role = role_override.unwrap_or_else(|| msg.role.clone());
    ChatMessage {
        role,
        content: texts.join("\n"),
        attachments: if !attachments.is_empty() {
            debug!("📎 Added {} attachments to message", attachments.len());
            Some(attachments)
        } else {
            None
        },
        content_type: "text/markdown".to_string(),
    }
}

pub async fn create_chat_request(
    model: &str,
    messages: Vec<Message>,
    temperature: Option<f32>,
    tools: Option<Vec<poe_api_process::types::ChatTool>>,
    logit_bias: Option<HashMap<String, f32>>,
    stop: Option<Vec<String>>,
) -> ChatRequest {
    debug!(
        "📝 Creating chat request | Model: {} | Message count: {} | Temperature: {:?} | Tool count: {:?}",
        model,
        messages.len(),
        temperature,
        tools.as_ref().map(|t| t.len())
    );
    // Get models.yaml config from cache
    let config: Arc<Config> = get_cached_config().await;
    // Check if model needs replace_response processing
    let should_replace_response = if let Some(model_config) = config.models.get(model) {
        // Use cached config
        model_config.replace_response.unwrap_or(false)
    } else {
        false
    };
    debug!(
        "🔍 Model {} replace_response setting: {}",
        model, should_replace_response
    );
    let query = messages
        .iter()
        .map(|msg| {
            let original_role = &msg.role;
            let role_override = match original_role.as_str() {
                // Always convert assistant to bot
                "assistant" => Some("bot".to_string()),
                // Always convert developer to user
                "developer" => Some("user".to_string()),
                // Only convert system to user if replace_response is true
                "system" if should_replace_response => Some("user".to_string()),
                // Keep original otherwise
                _ => None,
            };
            // Convert OpenAI message to Poe message
            let poe_message = openai_message_to_poe(msg, role_override);
            // Record conversion result
            debug!(
                "🔄 Processing message | Original role: {} | Converted role: {} | Content length: {} | Attachments: {}",
                original_role,
                poe_message.role,
                crate::utils::format_bytes_length(poe_message.content.len()),
                poe_message.attachments.as_ref().map_or(0, |a| a.len())
            );
            poe_message
        })
        .collect();
    // Process tool result messages
    let mut tool_results = None;
    // Check for tool role messages and convert to ToolResult
    if messages.iter().any(|msg| msg.role == "tool") {
        let mut results = Vec::new();
        for msg in &messages {
            if msg.role == "tool" {
                // Extract text content
                let content_text = get_text_from_openai_content(&msg.content);
                if let Some(tool_call_id) = extract_tool_call_id(&content_text) {
                    debug!("🔧 Processing tool result | tool_call_id: {}", tool_call_id);
                    results.push(poe_api_process::types::ChatToolResult {
                        role: "tool".to_string(),
                        tool_call_id,
                        name: "unknown".to_string(),
                        content: content_text,
                    });
                } else {
                    debug!("⚠️ Failed to extract tool_call_id from tool message");
                }
            }
        }
        if !results.is_empty() {
            tool_results = Some(results);
            debug!(
                "🔧 Created {} tool results",
                tool_results.as_ref().unwrap().len()
            );
        }
    }
    ChatRequest {
        version: "1.1".to_string(),
        r#type: "query".to_string(),
        query,
        temperature,
        user_id: "".to_string(),
        conversation_id: "".to_string(),
        message_id: "".to_string(),
        tools,
        tool_calls: None,
        tool_results,
        logit_bias,
        stop_sequences: stop,
    }
}

// Extract tool_call_id from tool message
fn extract_tool_call_id(content: &str) -> Option<String> {
    // Try to parse JSON formatted content
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
        if let Some(tool_call_id) = json.get("tool_call_id").and_then(|v| v.as_str()) {
            return Some(tool_call_id.to_string());
        }
    }
    // Try simple text parsing
    if let Some(start) = content.find("tool_call_id") {
        if let Some(id_start) = content[start..].find('"') {
            if let Some(id_end) = content[start + id_start + 1..].find('"') {
                return Some(
                    content[start + id_start + 1..start + id_start + 1 + id_end].to_string(),
                );
            }
        }
    }
    None
}
