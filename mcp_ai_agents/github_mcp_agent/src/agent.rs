use anyhow::Result;
use serde_json::{json, Value};

use crate::mcp::{McpClient, Tool};
use crate::openai::{Message, OpenAIClient};

fn tool_to_openai(tool: &Tool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema
        }
    })
}

pub async fn run(query: &str, github_token: &str, openai_key: &str) -> Result<String> {
    println!("🔌 Connecting to GitHub MCP server via Docker…");
    let mut mcp = McpClient::new(github_token).await?;

    println!("📋 Loading available tools…");
    let tools = mcp.list_tools().await?;
    println!("✅ {} tools loaded", tools.len());

    let openai_tools: Vec<Value> = tools.iter().map(tool_to_openai).collect();
    let openai = OpenAIClient::new(openai_key);

    let mut messages = vec![
        Message {
            role: "system".into(),
            content: Some(
                "You are a GitHub assistant. Help users explore repositories and their activity.\n\
                 - Provide organized, concise insights about the repository\n\
                 - Focus on facts and data from the GitHub API\n\
                 - Use markdown formatting for better readability\n\
                 - Present numerical data in tables when appropriate\n\
                 - Include links to relevant GitHub pages when helpful"
                    .into(),
            ),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".into(),
            content: Some(query.into()),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    // Agentic loop: call LLM → execute tools → repeat until no tool calls
    loop {
        println!("🤔 Thinking…");
        let response = openai.chat(&messages, &openai_tools).await?;

        match response.tool_calls.as_deref() {
            Some(calls) if !calls.is_empty() => {
                let n = calls.len();
                messages.push(response.clone());

                for (i, tc) in calls.iter().enumerate() {
                    println!("🔧 [{}/{}] {}", i + 1, n, tc.function.name);
                    let args: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                    let result = mcp
                        .call_tool(&tc.function.name, args)
                        .await
                        .unwrap_or_else(|e| format!("Tool error: {e}"));

                    messages.push(Message {
                        role: "tool".into(),
                        content: Some(result),
                        tool_calls: None,
                        tool_call_id: Some(tc.id.clone()),
                    });
                }
            }
            _ => return Ok(response.content.unwrap_or_default()),
        }
    }
}
