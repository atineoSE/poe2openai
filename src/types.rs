use poe_api_process::types::{ChatTool, ChatToolCall};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    pub stream_options: Option<StreamOptions>,
}

#[derive(Deserialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

// Define enum for OpenAI content format (String or array)
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum OpenAiContent {
    Text(String),
    Multi(Vec<OpenAiContentItem>),
}

// Define item type for OpenAI content array
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum OpenAiContentItem {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
}

// Define structure for image_url content
#[derive(Debug, Deserialize, Clone)]
pub struct ImageUrlContent {
    pub url: String,
    // Can extend other fields like detail
}

// Update Message struct to use new OpenAiContent
#[derive(Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: OpenAiContent,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub message: CompletionMessage,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct CompletionMessage {
    pub role: String,
    pub content: String,
    pub refusal: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub refusal: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Serialize, Clone, Debug)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

#[derive(Serialize, Clone, Debug)]
pub struct OpenAIError {
    pub message: String,
    pub r#type: String,
    pub code: String,
    pub param: Option<String>,
}

#[derive(Default, Serialize, Deserialize)]
pub(crate) struct Config {
    pub(crate) enable: Option<bool>,
    pub(crate) models: std::collections::HashMap<String, ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) custom_models: Option<Vec<CustomModel>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct CustomModel {
    pub(crate) id: String,
    pub(crate) created: Option<i64>,
    pub(crate) owned_by: Option<String>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub(crate) struct ModelConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mapping: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) replace_response: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) enable: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedUrl {
    pub poe_url: String,
    pub size_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct UrlCache {
    pub external_urls: HashMap<String, CachedUrl>,
    pub base64_hashes: HashMap<String, CachedUrl>,
    pub total_size_bytes: usize,
}
