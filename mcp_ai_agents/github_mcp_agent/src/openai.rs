use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

pub struct OpenAIClient {
    http: Client,
    api_key: String,
    model: String,
}

impl OpenAIClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            http: Client::new(),
            api_key: api_key.to_string(),
            model: "gpt-4o-mini".to_string(),
        }
    }

    pub async fn chat(&self, messages: &[Message], tools: &[Value]) -> Result<Message> {
        let body = if tools.is_empty() {
            json!({ "model": self.model, "messages": messages })
        } else {
            json!({
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"
            })
        };

        let resp = self
            .http
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .context("Failed to reach OpenAI API")?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            anyhow::bail!("OpenAI API error {status}: {text}");
        }

        let data: Value = serde_json::from_str(&text)?;
        let choice = &data["choices"][0]["message"];
        let msg: Message =
            serde_json::from_value(choice.clone()).context("Failed to parse OpenAI message")?;
        Ok(msg)
    }
}
