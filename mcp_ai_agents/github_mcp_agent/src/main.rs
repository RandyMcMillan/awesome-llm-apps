use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use std::{env, io::Write};

mod agent;
mod docker;
mod mcp;
mod openai;

use agent::ToolFilter;
use openai::{
    LlmClient, GITHUB_MODELS_BASE_URL, GITHUB_MODELS_DEFAULT_MODEL,
    OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL, OPENAI_BASE_URL, OPENAI_DEFAULT_MODEL,
};

#[derive(Parser)]
#[command(name = "github-mcp-agent")]
#[command(about = "🐙 GitHub MCP Agent — explore GitHub repos with natural language")]
#[command(arg_required_else_help = true)]
struct Cli {
    /// GitHub Personal Access Token (or set GITHUB_TOKEN env var)
    #[arg(long, global = true)]
    github_token: Option<String>,

    /// OpenAI API Key — optional (or set OPENAI_API_KEY env var)
    #[arg(long, global = true)]
    openai_key: Option<String>,

    /// LLM base URL — e.g. http://localhost:11434/v1 (or set LLM_BASE_URL env var)
    #[arg(long, global = true)]
    llm_url: Option<String>,

    /// Model name override (or set LLM_MODEL env var)
    #[arg(long, global = true)]
    model: Option<String>,

    /// List all available MCP tools grouped by category (no LLM required)
    #[arg(long, global = true)]
    list_tools: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

/// Shared args for all query subcommands
#[derive(Args)]
struct QueryArgs {
    /// Repository to analyze (format: owner/repo)
    #[arg(short, long, default_value = "gnostr-org/gnostr")]
    repo: String,

    /// Natural language query
    #[arg(short, long)]
    query: Option<String>,

    /// List tools available for this subcommand (no LLM required)
    #[arg(long)]
    list_tools: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Query and explore GitHub issues
    Issues(QueryArgs),

    /// Query and explore pull requests
    #[command(name = "prs")]
    PullRequests(QueryArgs),

    /// Query repository info: files, branches, commits, releases, tags
    #[command(name = "repo")]
    Repository(QueryArgs),

    /// Search code, commits, issues, or repositories
    Search(QueryArgs),

    /// List all available MCP tools grouped by category (no token or LLM required)
    Tools,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let github_token = cli
        .github_token
        .or_else(|| env::var("GITHUB_TOKEN").ok())
        .unwrap_or_default();

    println!("🐙 GitHub MCP Agent");

    // Global --list-tools or `tools` subcommand → show all tools
    let is_tools_cmd = matches!(cli.command, Some(Command::Tools));
    if cli.list_tools || is_tools_cmd {
        return list_tools(&github_token, None).await;
    }

    let Some(cmd) = cli.command else {
        bail!("A subcommand is required. Use --help for usage.");
    };

    // Subcommand --list-tools → show filtered tools for that category
    let sub_list = match &cmd {
        Command::Issues(a) | Command::PullRequests(a)
        | Command::Repository(a) | Command::Search(a) => a.list_tools,
        _ => false,
    };

    let filter = match &cmd {
        Command::Issues(_)        => ToolFilter::Issues,
        Command::PullRequests(_)  => ToolFilter::PullRequests,
        Command::Repository(_)    => ToolFilter::Repository,
        Command::Search(_)        => ToolFilter::Search,
        Command::Tools            => unreachable!(),
    };

    if sub_list {
        return list_tools(&github_token, Some(&filter)).await;
    }

    if github_token.is_empty() {
        bail!("GitHub token required. Set GITHUB_TOKEN env var or use --github-token");
    }

    let openai_key = cli.openai_key.or_else(|| env::var("OPENAI_API_KEY").ok());
    let llm_url    = cli.llm_url.or_else(|| env::var("LLM_BASE_URL").ok());
    let model      = cli.model.or_else(|| env::var("LLM_MODEL").ok());
    let llm        = resolve_llm(openai_key, llm_url, model, &github_token).await?;

    let args = match cmd {
        Command::Issues(a) | Command::PullRequests(a)
        | Command::Repository(a) | Command::Search(a) => a,
        Command::Tools => unreachable!(),
    };

    let query = match args.query {
        Some(q) => q,
        None => {
            print!("Query: ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            input.trim().to_string()
        }
    };

    let full_query = if query.contains(&args.repo) {
        query
    } else {
        format!("{} in {}", query, args.repo)
    };

    println!("Repository : {}", args.repo);
    println!("Query      : {}", full_query);
    println!("{}", "─".repeat(60));

    let result = agent::run(&full_query, &github_token, llm, filter).await?;
    println!("\n### Results\n{}", result);

    Ok(())
}

/// Print tools. If `filter` is Some, show only tools matching that category.
async fn list_tools(github_token: &str, filter: Option<&ToolFilter>) -> Result<()> {
    let toolsets = filter.map(|f| f.toolsets()).unwrap_or("repos,issues,pull_requests");
    let label = filter.map(|f| match f {
        ToolFilter::Issues       => "Issues",
        ToolFilter::PullRequests => "Pull Requests",
        ToolFilter::Repository   => "Repository",
        ToolFilter::Search       => "Search",
    });

    println!("🔌 Connecting to GitHub MCP server via Docker…");
    let mut mcp = mcp::McpClient::new(github_token, toolsets).await?;
    let all_tools = mcp.list_tools().await?;

    let tools: Vec<_> = match filter {
        Some(f) => all_tools.iter().filter(|t| f.matches(&t.name)).collect(),
        None    => all_tools.iter().collect(),
    };

    let sep = "─".repeat(72);

    if let Some(cat) = label {
        // Single-category listing
        println!("\n📋 {} tools — {cat}\n{sep}", tools.len());
        for t in &tools {
            let raw = t.description.as_deref().unwrap_or("").lines().next().unwrap_or("");
            let desc = if raw.len() > 57 { format!("{}…", &raw[..57]) } else { raw.to_string() };
            println!("  {:<38} {}", t.name, desc);
        }
        println!("{sep}");
        return Ok(());
    }

    // All-tools grouped listing
    let categories: &[(&str, &[&str])] = &[
        ("Issues",        &["issue", "sub_issue"]),
        ("Pull Requests", &["pull_request", "add_comment_to_pending", "add_reply"]),
        ("Repository",    &["create_repo", "fork", "list_branch", "list_commit",
                             "list_release", "list_tag", "list_repository",
                             "get_commit", "get_file", "get_label", "get_latest",
                             "get_release", "get_tag", "create_branch",
                             "create_or_update", "delete_file", "push_files"]),
        ("Search",        &["search_"]),
    ];

    let mut assigned: std::collections::HashSet<&str> = Default::default();
    println!("\n📋 {} tools\n{sep}", tools.len());

    for (cat_label, prefixes) in categories {
        let group: Vec<_> = tools
            .iter()
            .filter(|t| {
                !assigned.contains(t.name.as_str())
                    && prefixes.iter().any(|p| t.name.contains(p))
            })
            .collect();
        if group.is_empty() { continue; }
        for t in &group { assigned.insert(t.name.as_str()); }

        println!("  {cat_label}");
        println!("  {}", "─".repeat(68));
        for t in group {
            let raw = t.description.as_deref().unwrap_or("").lines().next().unwrap_or("");
            let desc = if raw.len() > 57 { format!("{}…", &raw[..57]) } else { raw.to_string() };
            println!("  {:<38} {}", t.name, desc);
        }
        println!();
    }

    let rest: Vec<_> = tools.iter().filter(|t| !assigned.contains(t.name.as_str())).collect();
    if !rest.is_empty() {
        println!("  Other");
        println!("  {}", "─".repeat(68));
        for t in rest {
            let raw = t.description.as_deref().unwrap_or("").lines().next().unwrap_or("");
            let desc = if raw.len() > 57 { format!("{}…", &raw[..57]) } else { raw.to_string() };
            println!("  {:<38} {}", t.name, desc);
        }
        println!();
    }

    println!("{sep}");
    Ok(())
}

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

    let ollama_model = model.clone().unwrap_or_else(|| OLLAMA_DEFAULT_MODEL.into());
    let ollama = LlmClient::new(OLLAMA_BASE_URL, None, &ollama_model);
    if ollama.probe().await {
        println!("🦙 Using Ollama at {OLLAMA_BASE_URL} ({ollama_model})");
        return Ok(ollama);
    }

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
                     Fix: https://github.com/settings/personal-access-tokens/new\n  \
                         → Account permissions → Models → Read\n\n  \
                     Alternatives:\n  \
                     • Set OPENAI_API_KEY to use OpenAI\n  \
                     • Start Ollama: https://ollama.com  then: ollama pull {m}"
                );
            }
            bail!("GitHub Models unavailable: {e}");
        }
    }
}
