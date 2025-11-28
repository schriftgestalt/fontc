use std::{fmt::Display, fs::File, path::Path, path::PathBuf, str::FromStr};

use plist::Value;
use serde::{Deserialize, Serialize};

use crate::args::ToolTypeCli;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToolType {
    Fontc,
    Fontmake,
    GlyphsApp,
}

impl ToolType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Fontc => "fontc",
            Self::Fontmake => "fontmake",
            Self::GlyphsApp => "glyphsapp",
        }
    }

    pub fn url(&self) -> &'static str {
        match self {
            Self::Fontc => "https://github.com/googlefonts/fontc",
            Self::Fontmake => "https://github.com/googlefonts/fontmake",
            Self::GlyphsApp => "https://glyphsapp.com/",
        }
    }
}

impl Display for ToolType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ToolManagement {
    Standalone,
    ManagedByGfTools,
}

impl ToolManagement {
    pub fn name_suffix(&self) -> &'static str {
        match self {
            Self::Standalone => "",
            Self::ManagedByGfTools => "_gftools",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Standalone => "standalone",
            Self::ManagedByGfTools => "Google Font Tools",
        }
    }
}

/// Tool version information is currently used for Glyphs app tools only.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ToolVersion {
    pub version_number: String,
    pub build_number: String,
}

#[derive(thiserror::Error, Debug)]
pub enum ToolVersionError {
    #[error("Info.plist could not be found: {0}")]
    InfoPlistNotFound(PathBuf),

    #[error("Info.plist could not be opened")]
    InfoPlistNotOpened(#[from] std::io::Error),

    #[error("Info.plist could not be parsed")]
    InfoPlistNotParsed(#[from] plist::Error),

    #[error("Version or build number could not be found in Info.plist: {0}")]
    VersionNotFound(PathBuf),
}

/// Error when converting a tool type from CLI to internal representation.
#[derive(thiserror::Error, Debug)]
pub enum ToolConversionError {
    #[error("Tool path is required for type {0}")]
    ToolPathRequired(ToolType),

    #[error("Tool version could not be determined")]
    ToolVersionNotDetermined(#[source] ToolVersionError),
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Tool {
    Fontc(ToolManagement),
    Fontmake(ToolManagement),
    GlyphsApp(PathBuf, ToolVersion),
}

impl Display for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.versioned_name())
    }
}

impl FromStr for Tool {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(s) = s.strip_suffix("_gftools") {
            match s {
                "fontc" => Ok(Tool::Fontc(ToolManagement::ManagedByGfTools)),
                "fontmake" => Ok(Tool::Fontmake(ToolManagement::ManagedByGfTools)),
                _ => Err(format!("Unknown gftools tool: {s}")),
            }
        } else if s == "fontc" {
            Ok(Tool::Fontc(ToolManagement::Standalone))
        } else if s == "fontmake" {
            Ok(Tool::Fontmake(ToolManagement::Standalone))
        } else if let Some(rest) = s.strip_prefix("glyphsapp_") {
            let parts: Vec<_> = rest.split('_').collect();
            if parts.len() == 2 {
                let version_number = parts[0].to_string();
                let build_number = parts[1].to_string();
                // When deserializing from string, we don't have the path.
                // We'll use an empty path as a placeholder.
                Ok(Tool::GlyphsApp(
                    PathBuf::new(),
                    ToolVersion {
                        version_number,
                        build_number,
                    },
                ))
            } else {
                Err(format!("Invalid GlyphsApp version string: {rest}"))
            }
        } else {
            Err(format!("Unknown tool string: {s}"))
        }
    }
}

impl Serialize for Tool {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s: &str = Deserialize::deserialize(deserializer)?;
        FromStr::from_str(s).map_err(serde::de::Error::custom)
    }
}

impl Tool {
    pub fn from_cli_args(
        cli_tool_type: ToolTypeCli,
        tool_path: Option<PathBuf>,
    ) -> Result<Self, ToolConversionError> {
        match cli_tool_type {
            // Simple conversions:
            ToolTypeCli::StandaloneFontc => Ok(Tool::Fontc(ToolManagement::Standalone)),
            ToolTypeCli::StandaloneFontmake => Ok(Tool::Fontmake(ToolManagement::Standalone)),
            ToolTypeCli::GfToolsFontc => Ok(Tool::Fontc(ToolManagement::ManagedByGfTools)),
            ToolTypeCli::GfToolsFontmake => Ok(Tool::Fontmake(ToolManagement::ManagedByGfTools)),

            // Glyphs app tool requires additional properties:
            ToolTypeCli::GlyphsApp => {
                let bundle_path =
                    tool_path.ok_or(ToolConversionError::ToolPathRequired(ToolType::GlyphsApp))?;
                let tool_version = macos_app_bundle_version(&bundle_path)
                    .map_err(ToolConversionError::ToolVersionNotDetermined)?;

                Ok(Tool::GlyphsApp(bundle_path, tool_version))
            }
        }
    }

    /// Named `tool_type` to avoid conflict with `type` keyword.
    pub fn tool_type(&self) -> ToolType {
        match self {
            Self::Fontc(_) => ToolType::Fontc,
            Self::Fontmake(_) => ToolType::Fontmake,
            Self::GlyphsApp(_, _) => ToolType::GlyphsApp,
        }
    }

    pub fn tool_management(&self) -> ToolManagement {
        match self {
            Self::Fontc(management) => *management,
            Self::Fontmake(management) => *management,

            // Glyphs app tool does not have a management type:
            Self::GlyphsApp(_, _) => ToolManagement::Standalone,
        }
    }

    pub fn bundle_path(&self) -> Option<PathBuf> {
        match self {
            Self::GlyphsApp(bundle_path, _) => Some(bundle_path.clone()),
            _ => None,
        }
    }

    pub fn version(&self) -> Option<&ToolVersion> {
        match self {
            Self::GlyphsApp(_, tool_version) => Some(tool_version),
            _ => None,
        }
    }

    /// Corresponds to the tool names used as flags to the `ttx_diff.py` script.
    pub fn unversioned_name(&self) -> String {
        match self {
            Self::GlyphsApp(_, _) => self.tool_type().name().to_string(),
            _ => format!(
                "{}{}",
                self.tool_type().name(),
                self.tool_management().name_suffix()
            ),
        }
    }

    /// The default implementation returns the tool type name.
    /// For default tools managed by Google Font Tools, appends the management type.
    /// For Glyphs app tools, returns a name that includes the version and build number.
    /// This is used for naming output files and directories.
    pub fn versioned_name(&self) -> String {
        match self {
            Self::GlyphsApp(_, tool_version) => {
                format!(
                    "{}_{}_{}",
                    self.tool_type().name(),
                    tool_version.version_number,
                    tool_version.build_number
                )
            }
            _ => self.unversioned_name(),
        }
    }

    /// Suitable for display purposes.
    /// For Glyphs app tools, returns a name that includes the version and build number.
    pub fn pretty_name(&self) -> String {
        match self {
            Self::GlyphsApp(_, tool_version) => {
                format!(
                    "Glyphs {} ({})",
                    tool_version.version_number, tool_version.build_number
                )
            }
            _ => format!(
                "{} ({})",
                self.tool_type().name(),
                self.tool_management().description()
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ToolPair {
    pub tool_1: Tool,
    pub tool_2: Tool,
}

impl ToolPair {
    pub fn has_gftools(&self) -> bool {
        self.tool_1.tool_management() == ToolManagement::ManagedByGfTools
            || self.tool_2.tool_management() == ToolManagement::ManagedByGfTools
    }

    pub fn has_glyphs_app(&self) -> bool {
        self.tool_1.tool_type() == ToolType::GlyphsApp
            || self.tool_2.tool_type() == ToolType::GlyphsApp
    }
}

pub fn macos_app_bundle_version<P: AsRef<Path>>(
    bundle_path: P,
) -> Result<ToolVersion, ToolVersionError> {
    let info_plist = bundle_path.as_ref().join("Contents").join("Info.plist");

    if !info_plist.exists() {
        return Err(ToolVersionError::InfoPlistNotFound(info_plist));
    }

    // Error will be converted automatically to `ToolVersionError::InfoPlistNotOpened`.
    let file = File::open(&info_plist)?;

    // Error will be converted automatically to `ToolVersionError::InfoPlistNotParsed`.
    let plist = Value::from_reader(file)?;

    fn get_string(plist: &Value, key: &str) -> Option<String> {
        plist
            .as_dictionary()
            .and_then(|dict| dict.get(key))
            .and_then(|val| val.as_string().map(|s| s.to_owned()))
    }

    let version_number = get_string(&plist, "CFBundleShortVersionString");
    let build_number = get_string(&plist, "CFBundleVersion");

    match (version_number, build_number) {
        (Some(version_number), Some(build_number)) => Ok(ToolVersion {
            version_number,
            build_number,
        }),
        _ => Err(ToolVersionError::VersionNotFound(info_plist)),
    }
}
