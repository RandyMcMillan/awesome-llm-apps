use anyhow::{Context, Result};
use clap::Parser;
use std::{env, io::Write};

mod agent;
mod mcp;
mod openai;

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

    /// GitHub Personal Access Token (or set GITHUB_TOKEN env var)
    #[arg(long)]
    github_token: Option<String>,

    /// OpenAI API Key (or set OPENAI_API_KEY env var)
    #[arg(long)]
    openai_key: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let github_token = cli
        .github_token
        .or_else(|| env::var("GITHUB_TOKEN").ok())
        .context("GitHub token required. Set GITHUB_TOKEN env var or use --github-token")?;

    let openai_key = cli
        .openai_key
        .or_else(|| env::var("OPENAI_API_KEY").ok())
        .context("OpenAI API key required. Set OPENAI_API_KEY env var or use --openai-key")?;

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

    println!("🐙 GitHub MCP Agent");
    println!("Repository : {}", cli.repo);
    println!("Query      : {}", full_query);
    println!("{}", "─".repeat(60));

    let result = agent::run(&full_query, &github_token, &openai_key).await?;
    println!("\n### Results\n{}", result);

    Ok(())
}
