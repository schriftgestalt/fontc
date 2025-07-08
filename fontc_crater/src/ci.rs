//! Running in CI.
//!
//! This handles the whole CI workflow.
//!
//! Unlike a normal run, this is expecting to have preexisting results, and to
//! generate a fuller report that includes comparison with past runs.

use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write,
    path::{Path, PathBuf},
    process::Command,
};

use chrono::{DateTime, TimeZone, Utc};
use google_fonts_sources::{Config, FontSource, SourceSet};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::de::DeserializeOwned;

use crate::{
    args::{CiArgs, RunMode},
    error::Error,
    ttx_diff_runner::{DiffError, DiffOutput},
    BuildType, Results, Target,
};

mod html;
mod results_cache;

pub(crate) use results_cache::ResultsCache;

static SUMMARY_FILE: &str = "summary.json";
static SOURCES_FILE: &str = "sources.json";
static FAILED_REPOS_FILE: &str = "failed_repos.json";

type DiffResults = Results<DiffOutput, DiffError>;

/// A summary of a single CI run
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RunSummary {
    began: DateTime<Utc>,
    finished: DateTime<Utc>,
    fontc_rev: String,
    #[serde(default)]
    pip_freeze_sha: String,
    results_file: PathBuf,
    // the name of the file listing targets used by this run.
    // it is intended that when this list is updated, the filename is changed.
    #[serde(alias = "input_file")]
    input_file_sha: String,
    stats: super::ttx_diff_runner::Summary,
}

impl RunSummary {
    fn try_load_results(&self, target_dir: &Path) -> Option<Result<DiffResults, Error>> {
        let report_path = target_dir.join(&self.results_file);
        if !report_path.exists() {
            return None;
        }
        Some(super::try_read_json(&report_path))
    }
}

pub(super) fn run_ci(args: &CiArgs) -> Result<(), Error> {
    if !args.html_only {
        super::ttx_diff_runner::assert_can_run_script();
        run_crater_and_save_results(args)?;
    }
    html::generate(&args.out_dir, &args.cache_dir(), args.mode)?;
    // now we want to generate an html report, based on this info.
    Ok(())
}

fn load_json_if_exists_else_default<T: DeserializeOwned + Default>(
    path: &Path,
) -> Result<T, Error> {
    if path.exists() {
        super::try_read_json(path)
    } else {
        Ok(Default::default())
    }
}

fn run_crater_and_save_results(args: &CiArgs) -> Result<(), Error> {
    if !args.out_dir.exists() {
        super::try_create_dir(&args.out_dir)?;
    }

    log_if_auth_or_not();
    // do this now so we error if the input file doesn't exist
    let inputs: SourceSet = super::try_read_json(&args.to_run)?;

    let summary_file = args.out_dir.join(SUMMARY_FILE);
    let mut prev_runs: Vec<RunSummary> = load_json_if_exists_else_default(&summary_file)?;
    // todo: fontc_repo should be checked out by us, and have a known path
    let fontc_rev = super::get_git_rev(None).unwrap();
    let pip_freeze_sha = super::pip_freeze_sha();
    let input_file_sha = super::get_input_sha(&args.to_run);

    let cache_dir = args.cache_dir();
    log::info!("using cache dir {}", cache_dir.display());
    let results_cache = ResultsCache::in_dir(&cache_dir);

    let mode = args.mode;
    if let Some(last_run) = prev_runs.last() {
        if mode == RunMode::GlyphsApp {
            log::info!("running in Glyphs app mode, continuing even if no changes since last run");
        }
        else if last_run.fontc_rev == fontc_rev
            && input_file_sha == last_run.input_file_sha
            && pip_freeze_sha == last_run.pip_freeze_sha
        {
            log::info!("no changes since last run, skipping");
            return Ok(());
        }
        if pip_freeze_sha != last_run.pip_freeze_sha || ttx_diff_has_changes(&last_run.fontc_rev) {
            log::info!("python deps or ttx_diff have changed, clearing cached results");
            results_cache.delete_all();
        }
    }

    let out_file = result_path_for_current_date();
    let out_path = args.out_dir.join(&out_file);
    // we want to build fontc & normalizer once, and then move them out of the
    // build directory so that they aren't accidentally rebuilt or deleted
    // while we're running
    let temp_bin_dir = tempfile::tempdir().expect("couldn't create tempdir");
    let (fontc_path, normalizer_path) = precompile_rust_binaries(temp_bin_dir.path());

    log::info!("compiled fontc to {}", fontc_path.display());
    log::info!("compiled otl-normalizeer to {}", normalizer_path.display());

    let ResolvedTargets {
        mut targets,
        source_repos,
        failures,
    } = make_targets(&cache_dir, &inputs.sources, mode);

    // If not in `GfTools` mode, keep only targets with `Default` build type.
    // If in `GlyphsApp` mode, keep only targets with `GlyphsApp` build type.
    match mode {
        RunMode::GfTools => (),
        RunMode::Default => targets.retain(|t| t.build == BuildType::Default),
        RunMode::GlyphsApp => targets.retain(|t| t.build == BuildType::GlyphsApp),
    }

    let n_targets = targets.len();

    let context = super::ttx_diff_runner::TtxContext {
        fontc_path,
        normalizer_path,
        source_cache: cache_dir,
        results_cache,
    };

    let began = Utc::now();
    let results = super::run_all(targets, &context, super::ttx_diff_runner::run_ttx_diff)?
        .into_iter()
        .collect();
    let finished = Utc::now();

    let elapsed = format_elapsed_time(&began, &finished);
    log::info!("completed {n_targets} targets in {elapsed}");

    let summary = super::ttx_diff_runner::Summary::new(&results);
    // if nothing has changed we still want to report it, but we don't need to
    // write a new big results file; we can reuse the previous one
    let (results_file, reuse_last_result) = match prev_runs.last() {
        Some(prev) if prev.stats == summary => (prev.results_file.clone(), true),
        _ => (out_file.into(), false),
    };

    let summary = RunSummary {
        began,
        finished,
        fontc_rev,
        pip_freeze_sha,
        results_file,
        input_file_sha,
        stats: summary,
    };

    prev_runs.push(summary);
    super::try_write_json(&prev_runs, &summary_file)?;
    if reuse_last_result {
        // we don't need to rewrite any of these other files if nothing
        // changed.
        return Ok(());
    }

    super::try_write_json(&results, &out_path)?;
    // we write the map of target -> source repo to a separate file because
    // otherwise we're basically duplicating it for each run.
    let sources_file = args.out_dir.join(SOURCES_FILE);
    super::try_write_json(&source_repos, &sources_file)?;
    let failures_file = args.out_dir.join(FAILED_REPOS_FILE);
    super::try_write_json(&failures, &failures_file)
}

fn result_path_for_current_date() -> String {
    let now = chrono::Utc::now();
    let timestamp = now.format("%Y-%m-%d-%H%M%S");
    format!("{timestamp}.json")
}

fn ttx_diff_has_changes(last_run_sha: &str) -> bool {
    let output = std::process::Command::new("git")
        .args(["diff", "--stat"])
        .arg(last_run_sha)
        .output()
        .unwrap();
    std::str::from_utf8(&output.stdout)
        .unwrap()
        .contains("ttx_diff.py")
}

#[derive(Debug, Default)]
struct ResolvedTargets {
    targets: Vec<Target>,
    // map of local path -> repo URL
    source_repos: BTreeMap<PathBuf, String>,
    // repos where we expected to find targets but didn't
    // map of URL -> error message
    failures: BTreeMap<String, String>,
}

impl ResolvedTargets {
    // it's possible for multiple config files to reference the same source.
    //
    // This might make sense for a gftools build, but for a default build it's
    // going to just be duplicate work, so we can remove these.
    fn remove_duplicate_default_targets(&mut self) {
        let mut seen = HashSet::new();
        let empty = Path::new("");
        self.targets.retain(|targ| {
            targ.build != BuildType::Default || seen.insert(targ.source_path(empty))
        });
    }
}

fn make_targets(cache_dir: &Path, repos: &[FontSource], mode: RunMode) -> ResolvedTargets {
    // first instantiate every repo in parallel:
    preflight_all_repos(cache_dir, repos);
    let mut result = ResolvedTargets::default();
    for repo in repos {
        let config_path = match repo.config_path(cache_dir) {
            Ok(config) => config,
            Err(e) => {
                result.failures.insert(repo.repo_url.clone(), e.to_string());
                continue;
            }
        };
        let config = match Config::load(&config_path) {
            Ok(x) => x,
            Err(e) => {
                result.failures.insert(repo.repo_url.clone(), e.to_string());
                continue;
            }
        };
        let relative_config_path = config_path
            .strip_prefix(cache_dir)
            .expect("config always in cache dir");
        let sources_dir = config_path
            .parent()
            .expect("config path always in sources dir");
        // config is always in sources, sources is always in org/repo
        let repo_dir = relative_config_path.parent().unwrap().parent().unwrap();
        result
            .source_repos
            .insert(repo_dir.to_owned(), repo.repo_url.clone());
        for source in &config.sources {
            let src_path = sources_dir.join(source);
            if !src_path.exists() {
                result
                    .failures
                    .insert(repo.repo_url.clone(), format!("missing source '{source}'"));
                continue;
            }
            let src_path = src_path
                .strip_prefix(cache_dir)
                .expect("source is always in cache dir");
            result.targets.extend(targets_for_source(
                repo,
                src_path,
                relative_config_path,
                &config,
                mode
            ))
        }
    }

    result.remove_duplicate_default_targets();
    result
}

// make sure all repos are fetched and up to date; do this in parallel
fn preflight_all_repos(cache_dir: &Path, sources: &[FontSource]) {
    let mut seen = HashSet::new();
    let has_unique_repo_and_sha = sources
        .iter()
        .filter(|src| seen.insert((src.repo_url.as_str(), src.git_rev())))
        .collect::<Vec<_>>();

    has_unique_repo_and_sha.par_iter().for_each(|src| {
        // we will handle errors later
        let _ignore = src.instantiate(cache_dir);
    });
}

fn targets_for_source(
    source: &FontSource,
    src_path: &Path,
    config_path: &Path,
    config: &Config,
    mode: RunMode
) -> Vec<Target> {
    let sha = source.git_rev();

    // `GlyphsApp` builds are mutually exclusive with the others.
    let in_glyphs_app_mode = mode == RunMode::GlyphsApp;
    let mut targets = Vec::with_capacity(if in_glyphs_app_mode { 1 } else { 2 });

    if in_glyphs_app_mode {
        if should_build_in_glyphs_app_mode(src_path) {
            match Target::new(
                src_path.to_owned(),
                config_path.to_owned(),
                sha.to_owned(),
                BuildType::GlyphsApp,
            ) {
                Ok(t)  => targets.push(t),
                Err(e) => log::warn!("failed to generate `GlyphsApp` target: {e}"),
            }
        }
    } else {
        match Target::new(
            src_path.to_owned(),
            config_path.to_owned(),
            sha.to_owned(),
            BuildType::Default,
        ) {
            Ok(t)  => targets.push(t),
            Err(e) => log::warn!("failed to generate `Default` target: {e}"),
        }
    
        if should_build_in_gftools_mode(src_path, config) {
            match Target::new(
                src_path.to_owned(),
                config_path.to_owned(),
                sha.to_owned(),
                BuildType::GfTools,
            ) {
                Ok(t)  => targets.push(t),
                Err(e) => log::warn!("failed to generate `GfTools` target: {e}"),
            }
        }
    }

    targets
}

fn should_build_in_gftools_mode(src_path: &Path, config: &Config) -> bool {
    let file_stem = src_path
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default();
    // skip noto, which have an implicitly different recipe provider
    //https://github.com/googlefonts/oxidize/blob/main/text/2024-06-26-fixes-and-nonstandard-builds.md#noto
    if file_stem.to_lowercase().starts_with("noto") {
        return false;
    }

    // skip Google Sans, which isn't built with gftools (but other fonts with
    // names that begin with 'GoogleSans' might be!)
    if ["GoogleSans", "GoogleSans-Italic"].contains(&file_stem.as_ref()) {
        return false;
    }

    // if there is a recipe provider other than googlefonts, we skip, because
    // it could be doing anything; see above
    config
        .recipe_provider
        .as_ref()
        .filter(|provider| *provider != "googlefonts")
        .is_none()
}

fn should_build_in_glyphs_app_mode(src_path: &Path) -> bool {
    let extension = src_path
        .extension()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default();

    // Accept only `.glyphs` and `.glyphspackage` files.
    extension.to_lowercase().starts_with("glyphs")
}

fn format_elapsed_time<Tmz: TimeZone>(start: &DateTime<Tmz>, end: &DateTime<Tmz>) -> String {
    let delta = end.clone().signed_duration_since(start);
    let mut out = String::new();
    let hours = delta.num_hours();
    let mins = delta.num_minutes() - hours * 60;
    let secs = delta.num_seconds() - (hours * 60 + mins) * 60;
    assert!(!hours.is_negative() | mins.is_negative() | secs.is_negative());
    if delta.num_hours() > 0 {
        write!(&mut out, "{hours}h").unwrap();
    }
    write!(&mut out, "{mins}m").unwrap();
    write!(&mut out, "{secs}s").unwrap();
    out
}

fn precompile_rust_binaries(temp_dir: &Path) -> (PathBuf, PathBuf) {
    let fontc = compile_crate_or_die("fontc");
    let normalizer = compile_crate_or_die("otl-normalizer");

    (
        copy_file_into_dir(&fontc, temp_dir),
        copy_file_into_dir(&normalizer, temp_dir),
    )
}

fn copy_file_into_dir(file_path: &Path, dir_path: &Path) -> PathBuf {
    let new_file_path = dir_path.join(file_path.file_name().unwrap());
    std::fs::copy(file_path, &new_file_path).expect("failed to copy binary to tempdir");
    new_file_path
}

// if we can't compile fontc / otl-normalizer there's nothing we can do?
fn compile_crate_or_die(name: &str) -> PathBuf {
    let status = Command::new("cargo")
        .args(["build", "-p", name, "--release"])
        .status()
        .expect("failed to run cargo build");
    if !status.success() {
        panic!("cargo build '{name}' failed");
    }

    expect_binary_target(name)
}

fn expect_binary_target(name: &str) -> PathBuf {
    let cwd = std::env::current_dir().expect("cwd exists and is readable");
    let target_dir = cwd.join("target/release");
    let target = target_dir.join(name).canonicalize().unwrap();
    assert!(
        target.is_file(),
        "missing target for '{name}' ({} is not a file)",
        target.display()
    );
    target
}

fn log_if_auth_or_not() {
    match std::env::var("GITHUB_TOKEN") {
        Ok(token) => log::info!(
            "authenticated with token ending '{}'",
            &token[token.len() - 10..]
        ),
        Err(_) => log::warn!("no auth token set, private repos will be skipped"),
    }
}
