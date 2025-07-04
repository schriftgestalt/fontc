//! CLI args

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};

// this env var can be set by the runner in order to reuse git checkouts
// between runs.
static GIT_CACHE_DIR_VAR: &str = "CRATER_GIT_CACHE";
pub static DEFAULT_CACHE_DIR: &'static str = "~/.fontc_crater_cache";

#[derive(Debug, PartialEq, Parser)]
#[command(about = "compile multiple fonts and report the results")]
pub(super) struct Args {
    #[command(subcommand)]
    pub(super) command: Commands,
}

#[derive(Debug, Subcommand, PartialEq)]
pub(super) enum Commands {
    Ci(CiArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
pub(super) enum RunMode {
    /// Run the default tools (`fontc` and `fontmake` directly), plus `fontc`
    /// and `fontmake` via `gftools`.
    GfTools,
    
    /// Run only the default tools (reduces target count when running locally).
    Default,
    
    /// Run only in Glyphs app version comparison mode.
    GlyphsApp,
}

#[derive(Debug, PartialEq, clap::Args)]
pub(super) struct CiArgs {
    /// Path to a json list of repos + revs to run.
    pub(super) to_run: PathBuf,

    /// Directory to store font sources and the `google/fonts` repo.
    ///
    /// Reusing this directory saves us having to clone all the repos on each run.
    ///
    /// This can also be set via the CRATER_GIT_CACHE environment variable,
    /// although the CLI argument takes precedence.
    ///
    /// If no argument is provided, defaults to `~/.fontc_crater_cache`.
    #[arg(short, long = "cache")]
    cache_dir: Option<PathBuf>,

    /// Directory where results are written.
    ///
    /// This should be consistent between runs.
    ///
    /// When using Glyphs app mode, it is recommended to specify a separate
    /// directory from the one used for other results.
    #[arg(short = 'o', long = "out")]
    pub(super) out_dir: PathBuf,

    /// Specify which tools to run.
    #[arg(long, value_enum, default_value_t = RunMode::GfTools)]
    pub(super) mode: RunMode,

    /// Only generate html (for the provided out_dir).
    #[arg(long)]
    pub(super) html_only: bool,
}

impl CiArgs {
    /// Determine the directory to use for caching git checkouts.
    ///
    /// This may be passed at the command line or via an environment variable.
    pub(crate) fn cache_dir(&self) -> PathBuf {
        let cache_dir = self
            .cache_dir
            .clone()
            .or_else(|| std::env::var_os(GIT_CACHE_DIR_VAR).map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR));

        if cache_dir.components().any(|comp| comp.as_os_str() == "~") {
            resolve_home(&cache_dir)
        } else {
            cache_dir
        }
    }
}

#[allow(deprecated)]
fn resolve_home(path: &Path) -> PathBuf {
    let Some(home_dir) = std::env::home_dir() else {
        log::warn!("No known home directory, ~ will not be resolved");
        return path.to_path_buf();
    };
    let mut result = PathBuf::new();
    for c in path.components() {
        if c.as_os_str() == "~" {
            result.push(home_dir.clone());
        } else {
            result.push(c);
        }
    }
    result
}
