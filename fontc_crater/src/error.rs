use crate::run_conf::RunConfigurationError;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub(super) enum Error {
    #[error(transparent)]
    SourceDiscovery(#[from] google_fonts_sources::Error),
    #[error("Failed to load file '{path}': {error}")]
    ReadFile {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error("Failed to write file '{path}': {error}")]
    WriteFile {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error("Failed to parse input json: '{path}': {error}")]
    ParseJson {
        path: PathBuf,
        #[source]
        error: serde_json::Error,
    },
    #[error("Failed to write json to '{path}': '{error}'")]
    WriteJson {
        path: PathBuf,
        #[source]
        error: serde_json::Error,
    },
    #[error("Failed to create directory '{path}' : {error}")]
    CreateDir {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },

    #[error("Failed to tidy html: '{0}")]
    TidyHtml(#[from] tidier::Error),

    #[error("Invalid run configuration: '{0}")]
    RunConfiguration(#[from] RunConfigurationError),
}
