use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    lines: tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

impl McpClient {
    pub async fn new(github_token: &str) -> Result<Self> {
        let mut child = Command::new("docker")
            .args([
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "-e",
                "GITHUB_TOOLSETS",
                "ghcr.io/github/github-mcp-server",
            ])
            .env("GITHUB_PERSONAL_ACCESS_TOKEN", github_token)
            .env("GITHUB_TOOLSETS", "repos,issues,pull_requests")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn Docker. Is Docker installed and running?")?;

        let stdin = child.stdin.take().context("Failed to get child stdin")?;
        let stdout = child.stdout.take().context("Failed to get child stdout")?;
        let lines = BufReader::new(stdout).lines();

        let mut client = Self { child, stdin, lines };
        client.initialize().await?;
        Ok(client)
    }

    async fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);
        let request = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        self.write_line(&request).await?;

        // Read until we get the response matching our id (skip notifications)
        loop {
            let line = self
                .lines
                .next_line()
                .await?
                .context("MCP server closed the connection unexpectedly")?;

            if line.trim().is_empty() {
                continue;
            }

            let resp: Value = serde_json::from_str(&line)
                .context("Failed to parse MCP server response")?;

            if resp.get("id").is_none() {
                // notification — skip
                continue;
            }

            if resp["id"] == id {
                if let Some(err) = resp.get("error") {
                    bail!("MCP error: {err}");
                }
                return Ok(resp["result"].clone());
            }
        }
    }

    async fn send_notification(&mut self, method: &str) -> Result<()> {
        let notif = json!({ "jsonrpc": "2.0", "method": method });
        self.write_line(&notif).await
    }

    async fn write_line(&mut self, value: &Value) -> Result<()> {
        let mut line = serde_json::to_string(value)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn initialize(&mut self) -> Result<()> {
        self.send_request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "github-mcp-agent-rs", "version": "0.1.0" }
            }),
        )
        .await?;
        self.send_notification("notifications/initialized").await
    }

    pub async fn list_tools(&mut self) -> Result<Vec<Tool>> {
        let result = self.send_request("tools/list", json!({})).await?;
        let tools: Vec<Tool> = serde_json::from_value(result["tools"].clone())
            .context("Failed to parse tools list from MCP server")?;
        Ok(tools)
    }

    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<String> {
        let result = self
            .send_request("tools/call", json!({ "name": name, "arguments": arguments }))
            .await?;

        // Extract text content blocks from the response
        if let Some(content) = result["content"].as_array() {
            let text = content
                .iter()
                .filter_map(|item| {
                    if item["type"] == "text" {
                        item["text"].as_str().map(str::to_owned)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(text)
        } else {
            Ok(serde_json::to_string_pretty(&result)?)
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}
