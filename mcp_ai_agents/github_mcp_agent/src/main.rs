use anyhow::{bail, Context, Result};
use clap::Parser;
use std::{env, io::Write};

mod agent;
mod docker;
mod mcp;
mod openai;

use openai::{
    LlmClient, GITHUB_MODELS_BASE_URL, GITHUB_MODELS_DEFAULT_MODEL,
    OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL, OPENAI_BASE_URL, OPENAI_DEFAULT_MODEL,
};

#[derive(Parser)]
#[command(name = "github-mcp-agent")]
#[command(about = "🐙 GitHub MCP Agent — explore GitHub repos with natural language")]
struct Cli {
    /// Repository to analyze (format: owner/repo)
    #[arg(short, long, default_value = "Shubhamsaboo/awesome-llm-apps")]
    repo: String,

    /// Query to run against the repository
    #[arg(short, long)]
    query: Option<String>,

    /// List available MCP tools and exit (no LLM required)
    #[arg(long)]
    list_tools: bool,

    /// GitHub Personal Access Token (or set GITHUB_TOKEN env var)
    #[arg(long)]
    github_token: Option<String>,

    /// OpenAI API Key — optional; omit to use Ollama or a key-free endpoint
    /// (or set OPENAI_API_KEY env var)
    #[arg(long)]
    openai_key: Option<String>,

    /// LLM base URL (default: OpenAI if key present, Ollama otherwise)
    /// Examples: https://api.openai.com/v1  http://localhost:11434/v1
    /// (or set LLM_BASE_URL env var)
    #[arg(long)]
    llm_url: Option<String>,

    /// Model name to use (default: gpt-4o-mini for OpenAI, llama3.2 for Ollama)
    /// (or set LLM_MODEL env var)
    #[arg(long)]
    model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let github_token = cli
        .github_token
        .or_else(|| env::var("GITHUB_TOKEN").ok())
        .context("GitHub token required. Set GITHUB_TOKEN env var or use --github-token")?;

    println!("🐙 GitHub MCP Agent");

    // --list-tools: connect to MCP and print tool catalogue, no LLM needed
    if cli.list_tools {
        return list_tools(&github_token).await;
    }

    // Resolve LLM backend (deferred — only needed for queries)
    let openai_key = cli.openai_key.or_else(|| env::var("OPENAI_API_KEY").ok());
    let llm_url    = cli.llm_url.or_else(|| env::var("LLM_BASE_URL").ok());
    let model      = cli.model.or_else(|| env::var("LLM_MODEL").ok());

    let llm = resolve_llm(openai_key, llm_url, model, &github_token).await?;

    let query = match cli.query {
        Some(q) => q,
        None => {
            print!("Query: ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            input.trim().to_string()
        }
    };

    let full_query = if query.contains(&cli.repo) {
        query
    } else {
        format!("{} in {}", query, cli.repo)
    };

    println!("Repository : {}", cli.repo);
    println!("Query      : {}", full_query);
    println!("{}", "─".repeat(60));

    let result = agent::run(&full_query, &github_token, llm).await?;
    println!("\n### Results\n{}", result);

    Ok(())
}

async fn list_tools(github_token: &str) -> Result<()> {
    println!("🔌 Connecting to GitHub MCP server via Docker…");
    let mut mcp = mcp::McpClient::new(github_token).await?;
    let tools = mcp.list_tools().await?;
    println!("📋 {} tools available:\n", tools.len());
    for t in &tools {
        let desc = t.description.as_deref().unwrap_or("(no description)");
        println!("  {:40} {}", t.name, desc);
    }
    Ok(())
}

/// Determine which LLM backend to use:
///   1. Explicit --llm-url              → use as-is (key optional)
///   2. OPENAI_API_KEY / --openai-key   → OpenAI
///   3. Ollama reachable at localhost   → Ollama (no key needed)
///   4. GitHub token present            → GitHub Models (free, rate-limited)
///   5. None of the above              → error with instructions
async fn resolve_llm(
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    github_token: &str,
) -> Result<LlmClient> {
    if let Some(url) = base_url {
        let m = model.unwrap_or_else(|| OPENAI_DEFAULT_MODEL.into());
        println!("🔗 Using custom LLM endpoint: {url} ({m})");
        return Ok(LlmClient::new(url, api_key, m));
    }

    if let Some(key) = api_key {
        let m = model.unwrap_or_else(|| OPENAI_DEFAULT_MODEL.into());
        println!("🤖 Using OpenAI ({m})");
        return Ok(LlmClient::new(OPENAI_BASE_URL, Some(key), m));
    }

    // No explicit key — probe Ollama first (local, free)
    let ollama_model = model.clone().unwrap_or_else(|| OLLAMA_DEFAULT_MODEL.into());
    let ollama = LlmClient::new(OLLAMA_BASE_URL, None, &ollama_model);
    if ollama.probe().await {
        println!("🦙 Using Ollama at {OLLAMA_BASE_URL} ({ollama_model})");
        return Ok(ollama);
    }

    // Fall back to GitHub Models — authenticated with the GitHub token already in hand.
    // Probe first so we give a clear error if the token lacks the 'models' permission.
    let m = model.unwrap_or_else(|| GITHUB_MODELS_DEFAULT_MODEL.into());
    let gh = LlmClient::new(GITHUB_MODELS_BASE_URL, Some(github_token.to_string()), &m);
    match gh.probe_result().await {
        Ok(()) => {
            println!("🐙 Using GitHub Models ({m})");
            Ok(gh)
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("401") || msg.contains("unauthorized") || msg.contains("models") {
                bail!(
                    "GitHub Models requires a fine-grained PAT with 'Models: Read' permission.\n  \
                     Your current token is missing that scope.\n\n  \
                     Fix: create a new token at https://github.com/settings/personal-access-tokens/new\n  \
                         → Permissions → Account permissions → Models → Read\n  \
                     Then rerun with: --github-token <new-token>\n\n  \
                     Alternatives:\n  \
                     • Set OPENAI_API_KEY (or --openai-key) to use OpenAI\n  \
                     • Start Ollama locally: https://ollama.com  then: ollama pull {m}\n  \
                     • Use --list-tools to browse available MCP tools with no LLM"
                );
            }
            bail!("GitHub Models unavailable: {e}");
        }
    }
}
