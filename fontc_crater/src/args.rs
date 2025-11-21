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

/// A flat representation of the valid tool type + tool management combinations,
/// used in CLI args.
#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
pub(super) enum ToolTypeCli {
    #[value(name = "fontc")]
    StandaloneFontc,

    #[value(name = "fontmake")]
    StandaloneFontmake,

    #[value(name = "fontc_gftools")]
    GfToolsFontc,

    #[value(name = "fontmake_gftools")]
    GfToolsFontmake,

    #[value(name = "glyphsapp")]
    GlyphsApp,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
pub(super) enum Preset {
    /// Run the default tools (`fontmake` and `fontc` directly), plus `fontmake`
    /// and `fontc` via `gftools`.
    GfTools,

    /// Run only the default tools (reduces target count when running locally).
    Default,

    /// Run two Glyphs app versions. Requires `--tool-1-path` and `--tool-2-path`
    /// to be set.
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
    ///
    /// The `ttx_diff.py` script now also supports a cache flag, so custom cache
    /// locations can be shared.
    #[arg(short, long = "cache")]
    cache_dir: Option<PathBuf>,

    /// Directory where results are written.
    ///
    /// This should be consistent between runs with the same preset or directly
    /// specified combination of tools.
    #[arg(short = 'o', long = "out")]
    pub(super) out_dir: PathBuf,

    /// Preset indicating which tools to run:
    ///
    /// `default` sets tool 1 to `fontmake` and tool 2 to `fontc`.
    ///
    /// `gftools` is like `default` but adds a second set of tools,
    /// `fontmake_gftools` and `fontc_gftools`.
    ///
    /// `glyphsapp` sets both tools to an instance of the Glyphs app, requiring
    /// their app bundles to be specified by the `--tool-x-path` options.
    ///
    /// Use `--tool-1-type` and `--tool-2-type` instead of a preset to specify other
    /// combinations of two tools, such as `fontc` and `glyphsapp`.
    #[arg(long, value_enum, alias = "mode")]
    pub(super) preset: Option<Preset>,

    /// The type of the first tool which should be used to build fonts to
    /// compare.
    #[arg(long, value_enum)]
    pub(super) tool_1_type: Option<ToolTypeCli>,

    /// For `glyphsapp`: Required path to the application bundle of the specific
    /// Glyphs app version to be used as tool 1.
    #[arg(long)]
    pub(super) tool_1_path: Option<PathBuf>,

    /// The type of the second tool which should be used to build fonts to
    /// compare.
    #[arg(long, value_enum)]
    pub(super) tool_2_type: Option<ToolTypeCli>,

    /// For `glyphsapp`: Required path to the application bundle of the specific
    /// Glyphs app version to be used as tool 2.
    #[arg(long)]
    pub(super) tool_2_path: Option<PathBuf>,

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
