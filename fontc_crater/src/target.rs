//! targets of a compilation

use std::{
    ffi::OsStr,
    fmt::{Display, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

use serde::{Deserialize, Serialize};

use crate::{
    args::DEFAULT_CACHE_DIR,
    tool::{
        Tool, 
        ToolType,
        ToolManagement,
        ToolPair,
    },
};

static VIRTUAL_CONFIG_DIR: &str = "sources";

fn default_cache_dir() -> PathBuf {
    return Path::new(DEFAULT_CACHE_DIR).to_path_buf();
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Target {
    /// path to the source repo, relative to the cache dir root
    repo_dir: PathBuf,
    sha: String,
    /// Path to the config file.
    ///
    /// - relative to the cache root if it is virtual;
    /// - relative to the repo root otherwise
    pub(crate) config: PathBuf,
    is_virtual: bool,
    /// Path to source file, relative to the source_dir
    source: PathBuf,

    pub(crate) tool_pair: ToolPair,
}

impl Target {
    pub(crate) fn new(
        repo_dir: impl Into<PathBuf>,
        sha: impl Into<String>,
        config: impl Into<PathBuf>,
        is_virtual: bool,
        source: impl Into<PathBuf>,
        tool_pair: ToolPair,
    ) -> Self {
        let mut sha = sha.into();
        let config = config.into();
        let source = source.into();
        sha.truncate(10);
        let config_dir = config.parent().unwrap();
        // if source is a sibling of config we can trim that bit.
        let source = source
            .strip_prefix(config_dir)
            .map(PathBuf::from)
            .unwrap_or(source);
        Self {
            repo_dir: repo_dir.into(),
            sha,
            config,
            is_virtual,
            source,
            tool_pair,
        }
    }

    pub(crate) fn tool_1(&self) -> &Tool {
        &self.tool_pair.tool_1
    }

    pub(crate) fn tool_2(&self) -> &Tool {
        &self.tool_pair.tool_2
    }

    /// Invariant: the source path is always in a directory
    ///
    /// If the config is virtual, then the source dir is '$REPO/sources'
    /// Otherwise, it is the parent directory of the config file.
    fn source_dir(&self) -> PathBuf {
        if self.is_virtual {
            self.repo_dir.clone()
        } else {
            self.repo_dir.join(self.config.parent().unwrap())
        }
    }

    /// The org/repo part of the path, used for looking up repo urls
    pub(crate) fn repo_path(&self) -> &Path {
        &self.repo_dir
    }

    pub(crate) fn source_path(&self, git_cache: &Path) -> PathBuf {
        let mut out = git_cache.join(self.source_dir());
        out.push(&self.source);
        out
    }

    pub(crate) fn config_path(&self, git_cache: &Path) -> PathBuf {
        if self.is_virtual {
            git_cache.join(&self.config)
        } else {
            git_cache.join(&self.repo_dir).join(&self.config)
        }
    }

    // if a target was built in a directory with a sha, the repro command
    // does not need to include that part of the directory, so remove it.
    fn config_path_stripping_disambiguating_sha_if_necessary(
        &self,
        cache_dir: &Path,
    ) -> String {
        let mut path = self
            .config_path(cache_dir)
            .display()
            .to_string();
        // NOTE: this relies on the fact that we always trim the sha to 10 chars,
        // both when we create a target and in google-fonts-sources when we
        // create the disambiguated checkout directory.
        if let Some(ix) = (!self.sha.is_empty())
            .then(|| path.find(&self.sha))
            .flatten()
        {
            path.replace_range(ix - 1..ix + self.sha.len(), "");
        }
        path
    }

    /// Return the path where we should cache the results of running this target.
    ///
    /// This is unique for each target, and is in the form,
    ///
    /// {BASE}{source_dir}/{config_stem}/{file_stem}/{tool_1_name}_{tool_2_name}
    ///
    /// where {source_dir} is the path to the sources/Sources directory of this
    /// target, relative to the root git cache, and the tool names include a
    /// version and build number, if applicable (currently Glyphs app only).
    pub(crate) fn cache_dir(&self, in_dir: &Path) -> PathBuf {
        let config = self.config.file_stem().unwrap_or(OsStr::new("config"));
        let mut result = in_dir.join(self.source_dir());
        result.push(config);
        result.push(self.source.file_stem().unwrap());
        result.push(format!(
            "{}_{}",
            self.tool_1().versioned_name(),
            self.tool_2().versioned_name()
        ));
        result
    }

    pub(crate) fn repro_command(&self, repo_url: &str, cache_dir: &Path) -> String {
        let repo_url = repo_url.trim();
        let source_path = self.source_path(Path::new(""));
        let rel_source_path = source_path
            .strip_prefix(&self.repo_dir)
            .expect("source always in repo");
        let sha_part = if !self.sha.is_empty() {
            format!("?{}", self.sha)
        } else {
            Default::default()
        };
        let tool_1 = self.tool_1();
        let tool_2 = self.tool_2();

        let mut cmd = format!(
            "python3 resources/scripts/ttx_diff.py '{repo_url}{sha_part}#{}'",
            rel_source_path.display()
        );
        write!(&mut cmd, " --tool_1_type {}", tool_1.unversioned_name()).unwrap();
        if tool_1.tool_type() == ToolType::GlyphsApp {
            if let Some(tool_path) = &tool_1.bundle_path() {
                write!(&mut cmd, " --tool_1_path {}", tool_path.display()).unwrap();
            }
        }
        write!(&mut cmd, " --tool_2_type {}", tool_2.unversioned_name()).unwrap();
        if tool_2.tool_type() == ToolType::GlyphsApp {
            if let Some(tool_path) = &tool_2.bundle_path() {
                write!(&mut cmd, " --tool_2_path {}", tool_path.display()).unwrap();
            }
        }
        if tool_1.tool_management() == ToolManagement::ManagedByGfTools ||
            tool_2.tool_management() == ToolManagement::ManagedByGfTools {
            let config = self.config_path_stripping_disambiguating_sha_if_necessary(cache_dir);
            write!(&mut cmd, " --config {config}").unwrap();
        }
        if cache_dir != default_cache_dir() {
            write!(&mut cmd, " --cache_path {0}", cache_dir.display()).unwrap();
        }
        cmd
    }
}

impl Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let config_path = if self.is_virtual {
            self.repo_dir.join("$VIRTUAL").join(&self.config)
        } else {
            self.repo_dir.join(&self.config)
        };

        write!(
            f,
            "{} {}?{} ({} + {})",
            config_path.display(),
            self.source.display(),
            self.sha,
            self.tool_1().versioned_name(),
            self.tool_2().versioned_name()
        )
    }
}

impl Serialize for Target {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Target {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s: &str = Deserialize::deserialize(deserializer)?;
        FromStr::from_str(s).map_err(serde::de::Error::custom)
    }
}

/// in the format,
///
/// $ORG/$REPO/$CONFIG_PATH?$SHA $SRC_PATH ($TOOL_1 + $TOOL_2)
///
/// where a virtual config's $CONFIG_PATH starts with the literal path element
/// '$VIRTUAL'.
impl FromStr for Target {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let (head, tools_part) = s
            .rsplit_once('(')
            .ok_or_else(|| "missing opening paren".to_string())?;

        let head = head.trim();
        let (config_part, source_part) = head
            .split_once(' ')
            .ok_or_else(|| "missing a space?".to_string())?;
        let (source, sha) = source_part
            .trim()
            .split_once('?')
            .ok_or_else(|| "missing '?'".to_string())?;
        let (split_at, _) = config_part
            .match_indices('/')
            .nth(1)
            .ok_or("missing second '/'")?;

        let (org_repo, config_part) = config_part.split_at(split_at);
        let config_part = config_part.trim_start_matches('/').trim();

        let (is_virtual, config_path) = if let Some(path) = config_part.strip_prefix("$VIRTUAL/") {
            (true, path)
        } else {
            (false, config_part)
        };

        let tools_part = tools_part.trim_end_matches(')').trim();
        let (tool_1_str, tool_2_str) = tools_part
            .split_once(" + ")
            .ok_or_else(|| "missing ' + ' separator for tools".to_string())?;

        let tool_pair = ToolPair {
            tool_1: Tool::from_str(tool_1_str.trim())?,
            tool_2: Tool::from_str(tool_2_str.trim())?,
        };

        Ok(Self::new(
            org_repo,
            sha,
            config_path,
            is_virtual,
            source,
            tool_pair,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_repr_simple() {
        let target = Target::new(
            "googlefonts/derp",
            "deadbeef",
            "sources/config.yaml",
            false,
            "sources/derp.glyphs",
            ToolPair {
                tool_1: Tool::Fontc(ToolManagement::Standalone),
                tool_2: Tool::Fontmake(ToolManagement::Standalone),
            },
        );    

        let asstr = target.to_string();

        assert_eq!(
            asstr,
            "googlefonts/derp/sources/config.yaml derp.glyphs?deadbeef (fontc + fontmake)"
        );

        let der = Target::from_str(&asstr).unwrap();
        assert_eq!(target, der)
    }

    #[test]
    fn string_repr_virtual() {
        let target = Target::new(
            "googlefonts/derp",
            "deadbeef",
            "ofl/derp/config.yaml",
            true,
            "derp.glyphs",
            ToolPair {
                tool_1: Tool::Fontc(ToolManagement::Standalone),
                tool_2: Tool::Fontmake(ToolManagement::Standalone),
            },
        );

        let asstr = target.to_string();

        assert_eq!(
            asstr,
            "googlefonts/derp/$VIRTUAL/ofl/derp/config.yaml derp.glyphs?deadbeef (fontc + fontmake)"
        );

        let der = Target::from_str(&asstr).unwrap();
        assert_eq!(target, der)
    }

    #[test]
    fn target_for_disambiguated_source() {
        let target = Target::new(
            "org/repo_123456789a",
            "123456789a",
            "Sources/hmm.yaml",
            false,
            "hello.glyphs",
            ToolPair {
                tool_1: Tool::Fontc(ToolManagement::Standalone),
                tool_2: Tool::Fontmake(ToolManagement::Standalone),
            },
        );

        let cache_dir = default_cache_dir();
        let hmm = target.config_path_stripping_disambiguating_sha_if_necessary(&cache_dir);
        assert_eq!(hmm, format!("{DEFAULT_CACHE_DIR}/org/repo/Sources/hmm.yaml"))
    }

    #[test]
    fn repro_command_with_sha() {
        let target = Target::new(
            "org/repo",
            "123456789a",
            "sources/config.yaml",
            false,
            "sources/hi.glyphs",
            ToolPair {
                tool_1: Tool::Fontc(ToolManagement::Standalone),
                tool_2: Tool::Fontmake(ToolManagement::Standalone),
            },
        );

        let cache_dir = default_cache_dir();
        let hmm = target.repro_command("example.com", &cache_dir);
        assert_eq!(
            hmm,
            "python3 resources/scripts/ttx_diff.py 'example.com?123456789a#sources/hi.glyphs' --tool_1_type fontc --tool_2_type fontmake"
        );
    }
}
