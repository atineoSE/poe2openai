use crate::types::*;
use crate::utils::{convert_poe_error_to_openai, format_bytes_length};
use poe_api_process::{ChatEventType, ChatResponse, ChatResponseData};
use salvo::prelude::*;
use std::collections::HashMap;
use tracing::{debug, error};

// Event accumulation context for collecting state during event processing
#[derive(Debug, Clone, Default)]
pub struct EventContext {
    pub content: String,
    pub replace_buffer: Option<String>,
    pub file_refs: HashMap<String, poe_api_process::types::FileData>,
    pub tool_calls: Vec<poe_api_process::types::ChatToolCall>,
    is_replace_mode: bool,
    pub error: Option<(StatusCode, OpenAIErrorResponse)>,
    pub done: bool,
    pub completion_tokens: u32,
    first_text_processed: bool,
    pub role_chunk_sent: bool,
    has_new_file_refs: bool,
    pub image_urls_sent: bool,
}

// Event handler trait
trait EventHandler {
    fn handle(&self, event: &ChatResponse, ctx: &mut EventContext) -> Option<String>;
}

// Text event handler
#[derive(Clone)]
struct TextEventHandler;
impl EventHandler for TextEventHandler {
    fn handle(&self, event: &ChatResponse, ctx: &mut EventContext) -> Option<String> {
        if let Some(ChatResponseData::Text { text }) = &event.data {
            debug!(
                "📝 Processing text event | Length: {} | is_replace_mode: {} | first_text_processed: {}",
                format_bytes_length(text.len()),
                ctx.is_replace_mode,
                ctx.first_text_processed
            );

            // If in replace mode and first text not processed, need to merge replace buffer with new text
            if ctx.is_replace_mode && !ctx.first_text_processed {
                debug!("📝 Merging first Text event with ReplaceResponse");
                if let Some(replace_content) = &mut ctx.replace_buffer {
                    replace_content.push_str(text);
                    ctx.first_text_processed = true;
                    // Return merged content to send fragment
                    return Some(replace_content.clone());
                } else {
                    // No replace_buffer, directly add to content
                    ctx.content.push_str(text);
                    return Some(text.clone());
                }
            }
            // If in replace mode and first text processed, reset to normal mode
            else if ctx.is_replace_mode && ctx.first_text_processed {
                debug!("🔄 Resetting replace mode to normal text mode");
                ctx.is_replace_mode = false;
                ctx.first_text_processed = false;

                // Move replace_buffer content to content
                if let Some(replace_content) = ctx.replace_buffer.take() {
                    ctx.content = replace_content;
                }
                // Directly add new text to content
                ctx.content.push_str(text);
                return Some(text.clone());
            } else {
                // Non-replace mode, directly accumulate and return text
                ctx.content.push_str(text);
                return Some(text.clone());
            }
        }
        None
    }
}

// File event handler
#[derive(Clone)]
struct FileEventHandler;
impl EventHandler for FileEventHandler {
    fn handle(&self, event: &ChatResponse, ctx: &mut EventContext) -> Option<String> {
        if let Some(ChatResponseData::File(file_data)) = &event.data {
            debug!(
                "🖼️  Processing file event | Name: {} | URL: {}",
                file_data.name, file_data.url
            );
            ctx.file_refs
                .insert(file_data.inline_ref.clone(), file_data.clone());
            ctx.has_new_file_refs = true;

            // If replace_buffer exists, process and send it
            if !ctx.image_urls_sent && ctx.replace_buffer.is_some() {
                // Only process if not sent yet
                let content = ctx.replace_buffer.as_ref().unwrap();
                if content.contains(&format!("[{}]", file_data.inline_ref)) {
                    debug!(
                        "🖼️ Detected ReplaceResponse with image reference [{}], processing immediately",
                        file_data.inline_ref
                    );
                    // Process image reference in text
                    let mut processed = content.clone();
                    let img_marker = format!("[{}]", file_data.inline_ref);
                    let replacement = format!("({})", file_data.url);
                    processed = processed.replace(&img_marker, &replacement);
                    ctx.image_urls_sent = true; // Mark as sent
                    return Some(processed);
                }
            }
        }
        None
    }
}

// ReplaceResponse event handler
#[derive(Clone)]
struct ReplaceResponseEventHandler;
impl EventHandler for ReplaceResponseEventHandler {
    fn handle(&self, event: &ChatResponse, ctx: &mut EventContext) -> Option<String> {
        if let Some(ChatResponseData::Text { text }) = &event.data {
            debug!(
                "🔄 Processing ReplaceResponse event | Length: {}",
                format_bytes_length(text.len())
            );
            ctx.is_replace_mode = true;
            ctx.replace_buffer = Some(text.clone());
            ctx.first_text_processed = false;

            // Check if file reference needs processing
            if !ctx.file_refs.is_empty() && text.contains('[') {
                debug!("🔄 ReplaceResponse may contain image reference, checking and processing");
                // Process image reference in text
                let mut processed = text.clone();
                let mut has_refs = false;

                for (ref_id, file_data) in &ctx.file_refs {
                    let img_marker = format!("[{}]", ref_id);
                    if processed.contains(&img_marker) {
                        let replacement = format!("({})", file_data.url);
                        processed = processed.replace(&img_marker, &replacement);
                        has_refs = true;
                        debug!("🖼️  Replacing image reference | ID: {} | URL: {}", ref_id, file_data.url);
                    }
                }

                if has_refs {
                    // If indeed contains image reference, send processed content immediately
                    debug!("✅ ReplaceResponse contains image reference, sending processed content immediately");
                    ctx.image_urls_sent = true; // Mark as sent
                    return Some(processed);
                }
            }

            // Defer ReplaceResponse output, wait for subsequent Text events
            debug!("🔄 Deferring ReplaceResponse output, waiting for Text events");
        }
        None // Don't send directly, wait to merge with Text
    }
}

// Json event handler (for Tool Calls)
#[derive(Clone)]
struct JsonEventHandler;
impl EventHandler for JsonEventHandler {
    fn handle(&self, event: &ChatResponse, ctx: &mut EventContext) -> Option<String> {
        debug!("📝 Processing JSON event");
        if let Some(ChatResponseData::ToolCalls(tool_calls)) = &event.data {
            debug!("🔧 Processing tool calls, count: {}", tool_calls.len());
            ctx.tool_calls.extend(tool_calls.clone());
            // Return Some to indicate tool calls need to be sent
            return Some("tool_calls".to_string());
        }
        None
    }
}

// Error event handler
#[derive(Clone)]
struct ErrorEventHandler;
impl EventHandler for ErrorEventHandler {
    fn handle(&self, event: &ChatResponse, ctx: &mut EventContext) -> Option<String> {
        if let Some(ChatResponseData::Error { text, allow_retry }) = &event.data {
            error!("❌ Processing error event: {}", text);
            let (status, error_response) = convert_poe_error_to_openai(text, *allow_retry);
            ctx.error = Some((status, error_response));
            return Some("error".to_string());
        }
        None
    }
}

// Done event handler
#[derive(Clone)]
struct DoneEventHandler;
impl EventHandler for DoneEventHandler {
    fn handle(&self, _event: &ChatResponse, ctx: &mut EventContext) -> Option<String> {
        debug!("✅ Processing Done event");
        ctx.done = true;

        // Only process if image URLs not sent yet
        if !ctx.image_urls_sent && ctx.replace_buffer.is_some() && !ctx.file_refs.is_empty() {
            let content = ctx.replace_buffer.as_ref().unwrap();
                debug!("🔍 Checking for unprocessed image references on completion");
            let mut processed = content.clone();
            let mut has_refs = false;

            for (ref_id, file_data) in &ctx.file_refs {
                let img_marker = format!("[{}]", ref_id);
                if processed.contains(&img_marker) {
                    let replacement = format!("({})", file_data.url);
                    processed = processed.replace(&img_marker, &replacement);
                    has_refs = true;
                    debug!(
                        "🖼️ Replacing image reference before completion | ID: {} | URL: {}",
                        ref_id, file_data.url
                    );
                }
            }

            if has_refs {
                debug!("✅ Processed image references before completion");
                ctx.image_urls_sent = true; // Mark as sent
                return Some(processed);
            }
        }

        Some("done".to_string())
    }
}

// Event handler manager
#[derive(Clone)]
pub struct EventHandlerManager {
    text_handler: TextEventHandler,
    file_handler: FileEventHandler,
    replace_handler: ReplaceResponseEventHandler,
    json_handler: JsonEventHandler,
    error_handler: ErrorEventHandler,
    done_handler: DoneEventHandler,
}

impl EventHandlerManager {
    pub fn new() -> Self {
        Self {
            text_handler: TextEventHandler,
            file_handler: FileEventHandler,
            replace_handler: ReplaceResponseEventHandler,
            json_handler: JsonEventHandler,
            error_handler: ErrorEventHandler,
            done_handler: DoneEventHandler,
        }
    }

    pub fn handle(&self, event: &ChatResponse, ctx: &mut EventContext) -> Option<String> {
        match event.event {
            ChatEventType::Text => self.text_handler.handle(event, ctx),
            ChatEventType::File => self.file_handler.handle(event, ctx),
            ChatEventType::ReplaceResponse => self.replace_handler.handle(event, ctx),
            ChatEventType::Json => self.json_handler.handle(event, ctx),
            ChatEventType::Error => self.error_handler.handle(event, ctx),
            ChatEventType::Done => self.done_handler.handle(event, ctx),
        }
    }
}
