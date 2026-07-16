use anyhow::{bail, Context, Result};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
};
use tokio::process::Command;

pub const IMAGE: &str = "ghcr.io/github/github-mcp-server";

/// Locate the `docker` binary cross-platform (PATH + known fallback locations).
pub fn find_docker() -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        "docker.exe"
    } else {
        "docker"
    };

    // Search PATH first
    if let Ok(path_var) = std::env::var("PATH") {
        let sep = if cfg!(windows) { ';' } else { ':' };
        for dir in path_var.split(sep) {
            let p = Path::new(dir).join(exe);
            if p.is_file() {
                return Some(p);
            }
        }
    }

    // Platform-specific fallbacks
    #[cfg(target_os = "macos")]
    let extras: &[&str] = &[
        "/usr/local/bin/docker",
        "/opt/homebrew/bin/docker",
        "/Applications/Docker.app/Contents/Resources/bin/docker",
    ];
    #[cfg(target_os = "linux")]
    let extras: &[&str] = &[
        "/usr/bin/docker",
        "/usr/local/bin/docker",
        "/snap/bin/docker",
    ];
    #[cfg(windows)]
    let extras: &[&str] = &[
        r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
        r"C:\ProgramData\DockerDesktop\version-bin\docker.exe",
    ];

    extras
        .iter()
        .map(Path::new)
        .find(|p| p.is_file())
        .map(PathBuf::from)
}

/// Find docker, verify the daemon is up, pull the MCP image if absent.
/// Returns the resolved path to the docker binary.
pub async fn ensure_ready() -> Result<PathBuf> {
    let docker = find_docker().with_context(|| {
        format!(
            "Docker binary not found.\n  \
             Install Docker Desktop: https://www.docker.com/get-started\n  \
             Then make sure `docker` is on your PATH."
        )
    })?;

    println!("🐳 Docker: {}", docker.display());
    check_daemon(&docker).await?;
    ensure_image(&docker).await?;
    Ok(docker)
}

async fn check_daemon(docker: &Path) -> Result<()> {
    let ok = Command::new(docker)
        .args(["info", "--format", "{{.ServerVersion}}"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);

    if ok {
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    bail!(
        "Docker daemon is not running.\n  \
         Start it with: open -a Docker\n  \
         Or launch Docker Desktop from your Applications folder."
    );
    #[cfg(target_os = "windows")]
    bail!(
        "Docker daemon is not running.\n  \
         Start Docker Desktop from the Start menu or system tray."
    );
    #[cfg(not(any(target_os = "macos", windows)))]
    bail!(
        "Docker daemon is not running.\n  \
         Try:  sudo systemctl start docker\n  \
         Or:   sudo service docker start"
    );
}

async fn ensure_image(docker: &Path) -> Result<()> {
    let present = Command::new(docker)
        .args(["image", "inspect", IMAGE, "--format", "{{.Id}}"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);

    if present {
        println!("✅ Image {IMAGE} already present");
        return Ok(());
    }

    println!("📦 Pulling {IMAGE} …");
    let status = Command::new(docker)
        .args(["pull", IMAGE])
        .status()
        .await
        .context("Failed to run docker pull")?;

    if !status.success() {
        bail!(
            "Failed to pull {IMAGE}.\n  \
             Check your internet connection, then try:\n  \
             docker pull {IMAGE}"
        );
    }
    println!("✅ Image ready");
    Ok(())
}
