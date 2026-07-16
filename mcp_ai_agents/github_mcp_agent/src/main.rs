use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use std::{env, io::Write};

use github_mcp_agent::agent::ToolFilter;
use github_mcp_agent::openai::{
    LlmClient, GITHUB_MODELS_BASE_URL, GITHUB_MODELS_DEFAULT_MODEL, OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL, OPENAI_BASE_URL, OPENAI_DEFAULT_MODEL,
};
use github_mcp_agent::{agent, mcp};

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
    #[arg(long)]
    list_tools: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

/// Shared args for all query subcommands
#[derive(Args)]
struct QueryArgs {
    /// Repository (owner/repo) — required for most toolsets
    #[arg(short, long, default_value = "gnostr-org/gnostr")]
    repo: String,

    /// Natural language query
    #[arg(short, long)]
    query: Option<String>,

    /// List tools available for this subcommand (no LLM required)
    #[arg(long)]
    list_tools: bool,

    /// Restrict to specific tools (comma-separated, validated against --list-tools)
    /// Pass without a value to list available tools for this subcommand.
    #[arg(long, num_args(0..), value_delimiter = ',')]
    tools: Option<Vec<String>>,
}

#[derive(Subcommand)]
enum Command {
    /// GitHub Actions workflows and CI/CD operations
    Actions(QueryArgs),
    /// Code quality tools
    #[command(name = "code-quality")]
    CodeQuality(QueryArgs),
    /// Code security — GitHub Code Scanning
    #[command(name = "code-security")]
    CodeSecurity(QueryArgs),
    /// Current user and GitHub context (strongly recommended for context)
    Context(QueryArgs),
    /// Copilot related tools
    Copilot(QueryArgs),
    /// Dependabot tools
    Dependabot(QueryArgs),
    /// GitHub Discussions
    Discussions(QueryArgs),
    /// GitHub Gists
    Gists(QueryArgs),
    /// Low-level Git API operations
    Git(QueryArgs),
    /// GitHub Issues
    Issues(QueryArgs),
    /// GitHub Labels
    Labels(QueryArgs),
    /// GitHub Notifications
    Notifications(QueryArgs),
    /// GitHub Organizations
    Orgs(QueryArgs),
    /// GitHub Projects
    Projects(QueryArgs),
    /// GitHub Pull Requests
    #[command(name = "prs")]
    PullRequests(QueryArgs),
    /// Repository files, branches, commits, releases, tags
    #[command(name = "repo")]
    Repository(QueryArgs),
    /// Secret protection — GitHub Secret Scanning
    #[command(name = "secret-protection")]
    SecretProtection(QueryArgs),
    /// Security advisories
    #[command(name = "security-advisories")]
    SecurityAdvisories(QueryArgs),
    /// Search code, commits, issues, PRs, or repositories
    Search(QueryArgs),
    /// GitHub Stargazers
    Stargazers(QueryArgs),
    /// GitHub Users
    Users(QueryArgs),
    /// List all available MCP tools grouped by category (no token or LLM required)
    Tools,
}

impl Command {
    fn filter(&self) -> ToolFilter {
        match self {
            Self::Actions(_) => ToolFilter::Actions,
            Self::CodeQuality(_) => ToolFilter::CodeQuality,
            Self::CodeSecurity(_) => ToolFilter::CodeSecurity,
            Self::Context(_) => ToolFilter::Context,
            Self::Copilot(_) => ToolFilter::Copilot,
            Self::Dependabot(_) => ToolFilter::Dependabot,
            Self::Discussions(_) => ToolFilter::Discussions,
            Self::Gists(_) => ToolFilter::Gists,
            Self::Git(_) => ToolFilter::Git,
            Self::Issues(_) => ToolFilter::Issues,
            Self::Labels(_) => ToolFilter::Labels,
            Self::Notifications(_) => ToolFilter::Notifications,
            Self::Orgs(_) => ToolFilter::Orgs,
            Self::Projects(_) => ToolFilter::Projects,
            Self::PullRequests(_) => ToolFilter::PullRequests,
            Self::Repository(_) => ToolFilter::Repository,
            Self::SecretProtection(_) => ToolFilter::SecretProtection,
            Self::SecurityAdvisories(_) => ToolFilter::SecurityAdvisories,
            Self::Search(_) => ToolFilter::Search,
            Self::Stargazers(_) => ToolFilter::Stargazers,
            Self::Users(_) => ToolFilter::Users,
            Self::Tools => unreachable!(),
        }
    }

    fn args(self) -> QueryArgs {
        match self {
            Self::Actions(a)
            | Self::CodeQuality(a)
            | Self::CodeSecurity(a)
            | Self::Context(a)
            | Self::Copilot(a)
            | Self::Dependabot(a)
            | Self::Discussions(a)
            | Self::Gists(a)
            | Self::Git(a)
            | Self::Issues(a)
            | Self::Labels(a)
            | Self::Notifications(a)
            | Self::Orgs(a)
            | Self::Projects(a)
            | Self::PullRequests(a)
            | Self::Repository(a)
            | Self::SecretProtection(a)
            | Self::SecurityAdvisories(a)
            | Self::Search(a)
            | Self::Stargazers(a)
            | Self::Users(a) => a,
            Self::Tools => unreachable!(),
        }
    }

    fn tools_flag(&self) -> Option<&Vec<String>> {
        match self {
            Self::Actions(a) | Self::CodeQuality(a) | Self::CodeSecurity(a)
            | Self::Context(a) | Self::Copilot(a) | Self::Dependabot(a)
            | Self::Discussions(a) | Self::Gists(a) | Self::Git(a)
            | Self::Issues(a) | Self::Labels(a) | Self::Notifications(a)
            | Self::Orgs(a) | Self::Projects(a) | Self::PullRequests(a)
            | Self::Repository(a) | Self::SecretProtection(a)
            | Self::SecurityAdvisories(a) | Self::Search(a)
            | Self::Stargazers(a) | Self::Users(a) => a.tools.as_ref(),
            Self::Tools => None,
        }
    }

    fn list_tools_flag(&self) -> bool {
        match self {
            Self::Actions(a)
            | Self::CodeQuality(a)
            | Self::CodeSecurity(a)
            | Self::Context(a)
            | Self::Copilot(a)
            | Self::Dependabot(a)
            | Self::Discussions(a)
            | Self::Gists(a)
            | Self::Git(a)
            | Self::Issues(a)
            | Self::Labels(a)
            | Self::Notifications(a)
            | Self::Orgs(a)
            | Self::Projects(a)
            | Self::PullRequests(a)
            | Self::Repository(a)
            | Self::SecretProtection(a)
            | Self::SecurityAdvisories(a)
            | Self::Search(a)
            | Self::Stargazers(a)
            | Self::Users(a) => a.list_tools,
            Self::Tools => false,
        }
    }
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

    // Subcommand --list-tools OR bare --tools (no values) → show filtered tools
    if cmd.list_tools_flag() || matches!(cmd.tools_flag(), Some(t) if t.is_empty()) {
        let filter = cmd.filter();
        return list_tools(&github_token, Some(&filter)).await;
    }

    if github_token.is_empty() {
        bail!("GitHub token required. Set GITHUB_TOKEN env var or use --github-token");
    }

    let openai_key = cli.openai_key.or_else(|| env::var("OPENAI_API_KEY").ok());
    let llm_url = cli.llm_url.or_else(|| env::var("LLM_BASE_URL").ok());
    let model = cli.model.or_else(|| env::var("LLM_MODEL").ok());
    let llm = resolve_llm(openai_key, llm_url, model, &github_token).await?;

    let filter = cmd.filter();
    let args = cmd.args();

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

    let explicit_tools = args.tools.as_deref().unwrap_or(&[]);
    let result = agent::run(&full_query, &github_token, llm, filter, explicit_tools).await?;
    println!("\n### Results\n{}", result);

    Ok(())
}

/// Print tools. Pass `filter` to show only tools matching that category.
async fn list_tools(github_token: &str, filter: Option<&ToolFilter>) -> Result<()> {
    let toolsets = filter.map(|f| f.toolsets()).unwrap_or("all");

    let label = filter.map(|f| match f {
        ToolFilter::Actions => "Actions",
        ToolFilter::CodeQuality => "Code Quality",
        ToolFilter::CodeSecurity => "Code Security",
        ToolFilter::Context => "Context",
        ToolFilter::Copilot => "Copilot",
        ToolFilter::Dependabot => "Dependabot",
        ToolFilter::Discussions => "Discussions",
        ToolFilter::Gists => "Gists",
        ToolFilter::Git => "Git",
        ToolFilter::Issues => "Issues",
        ToolFilter::Labels => "Labels",
        ToolFilter::Notifications => "Notifications",
        ToolFilter::Orgs => "Organizations",
        ToolFilter::Projects => "Projects",
        ToolFilter::PullRequests => "Pull Requests",
        ToolFilter::Repository => "Repository",
        ToolFilter::SecretProtection => "Secret Protection",
        ToolFilter::SecurityAdvisories => "Security Advisories",
        ToolFilter::Search => "Search",
        ToolFilter::Stargazers => "Stargazers",
        ToolFilter::Users => "Users",
    });

    println!("🔌 Connecting to GitHub MCP server via Docker…");
    let mut mcp = mcp::McpClient::new(github_token, toolsets).await?;
    let all_tools = mcp.list_tools().await?;

    let tools: Vec<_> = match filter {
        Some(f) => all_tools.iter().filter(|t| f.matches(&t.name)).collect(),
        None => all_tools.iter().collect(),
    };

    let sep = "─".repeat(72);

    if let Some(cat) = label {
        println!("\n📋 {} tools\n{sep}", tools.len());
        println!("  {cat}");
        println!("  {}", "─".repeat(68));
        for t in &tools {
            let raw = t
                .description
                .as_deref()
                .unwrap_or("")
                .lines()
                .next()
                .unwrap_or("");
            let desc = if raw.len() > 57 {
                format!("{}…", &raw[..57])
            } else {
                raw.to_string()
            };
            println!("  {:<42} {}", t.name, desc);
        }
        println!("\n{sep}");
        return Ok(());
    }

    // All-tools grouped listing (GITHUB_TOOLSETS=all)
    println!("\n📋 {} tools\n{sep}", tools.len());
    // Group by common name patterns
    let groups: &[(&str, &[&str])] = &[
        (
            "Actions",
            &[
                "workflow", "run", "job", "artifact", "runner", "secret", "variable", "cache",
            ],
        ),
        ("Code Quality", &["code_quality", "autofix"]),
        ("Code Security", &["code_scanning", "alert"]),
        ("Context", &["get_me", "get_github_context"]),
        ("Copilot", &["copilot"]),
        ("Dependabot", &["dependabot"]),
        ("Discussions", &["discussion"]),
        ("Gists", &["gist"]),
        ("Git", &["git_"]),
        ("Issues", &["issue", "sub_issue"]),
        ("Labels", &["label"]),
        ("Notifications", &["notification", "thread"]),
        ("Organizations", &["org", "team", "member"]),
        ("Projects", &["project"]),
        (
            "Pull Requests",
            &["pull_request", "pending_review", "add_reply"],
        ),
        (
            "Repository",
            &[
                "repo",
                "branch",
                "commit",
                "release",
                "tag",
                "file",
                "fork",
                "push_files",
            ],
        ),
        ("Search", &["search_"]),
        ("Secret Protection", &["secret_scanning"]),
        ("Security Advisories", &["advisory", "cve"]),
        ("Stargazers", &["star"]),
        ("Users", &["user", "follow", "block"]),
    ];

    let mut assigned: std::collections::HashSet<&str> = Default::default();
    for (cat_label, prefixes) in groups {
        let group: Vec<_> = tools
            .iter()
            .filter(|t| {
                !assigned.contains(t.name.as_str()) && prefixes.iter().any(|p| t.name.contains(p))
            })
            .collect();
        if group.is_empty() {
            continue;
        }
        for t in &group {
            assigned.insert(t.name.as_str());
        }

        println!("  {cat_label}");
        println!("  {}", "─".repeat(68));
        for t in group {
            let raw = t
                .description
                .as_deref()
                .unwrap_or("")
                .lines()
                .next()
                .unwrap_or("");
            let desc = if raw.len() > 57 {
                format!("{}…", &raw[..57])
            } else {
                raw.to_string()
            };
            println!("  {:<42} {}", t.name, desc);
        }
        println!();
    }

    let rest: Vec<_> = tools
        .iter()
        .filter(|t| !assigned.contains(t.name.as_str()))
        .collect();
    if !rest.is_empty() {
        println!("  Other");
        println!("  {}", "─".repeat(68));
        for t in rest {
            let raw = t
                .description
                .as_deref()
                .unwrap_or("")
                .lines()
                .next()
                .unwrap_or("");
            let desc = if raw.len() > 57 {
                format!("{}…", &raw[..57])
            } else {
                raw.to_string()
            };
            println!("  {:<42} {}", t.name, desc);
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
