//! We support two principal modes of operation:
//! 1. Running tools specified by a preset (previously called build type).
//! 2. Running tools specified by tool types (tool 1 and tool 2).
//!
//! Adding tools beyond `fontc` and `fontmake` as well as the ability to run
//! (mostly) arbitrary tool pairs while maintaining the existing functionality
//! creates several complexities:
//!
//! The `GfTools` preset runs `fontc` and `fontmake`, just like `Default`,
//! but it does so _twice_: once standalone and once managed by `gftools`.
//! While the HTML report differentiates between the two runs in the individual
//! results list, it does combine the results for each tool type in the summary
//! and the results list sections. This means that we have to separate the
//! concepts of tool type, tool management and tool instance.
//!
//! Running two Glyphs app versions is also a special case because it requires
//! two separate tool instances but only one tool type. We use the `tool_1_path`
//! and `tool_2_path` arguments to specify the paths to the two Glyphs app
//! instances. In order to make the tool names unique, we read the version and
//! build number from the app bundles and use them in the tool names.

use crate::{
    args::{
        CiArgs, 
        Preset, 
        ToolTypeCli
    },
    tool::{
        Tool, 
        ToolConversionError, 
        ToolManagement, 
        ToolType
    },
};

#[derive(Clone, Debug, PartialEq)]
pub struct RunConfiguration {
    pub tool_pairs: Vec<ToolPair>,
}

impl RunConfiguration {
    pub fn from_cli_args(args: &CiArgs) -> Result<Self, RunConfigurationError> {
        // Note: Because of the complex relationship between the arguments for presets,
        // tool types and tool paths, we cannot use the `ArgGroup` feature of `clap`.
        // Instead, we match on all five optional arguments.
        match (args.preset, args.tool_1_type, args.tool_1_path.as_ref(), args.tool_2_type, args.tool_2_path.as_ref()) {

            // Default preset. Ignore tool types and paths.
            (Some(Preset::Default), _, _, _, _) => {
                Ok(RunConfiguration {
                    tool_pairs: vec![
                        ToolPair {
                            tool_1: Tool::from_cli_args(ToolTypeCli::StandaloneFontc, None)?,
                            tool_2: Tool::from_cli_args(ToolTypeCli::StandaloneFontmake, None)?,
                        },
                    ],
                })
            },

            // GFTools preset. Ignore tool types and paths.
            (Some(Preset::GfTools), _, _, _, _) => {
                Ok(RunConfiguration {
                    tool_pairs: vec![
                        ToolPair {
                            tool_1: Tool::from_cli_args(ToolTypeCli::StandaloneFontc, None)?,
                            tool_2: Tool::from_cli_args(ToolTypeCli::StandaloneFontmake, None)?,
                        },
                        ToolPair {
                            tool_1: Tool::from_cli_args(ToolTypeCli::GfToolsFontc, None)?,
                            tool_2: Tool::from_cli_args(ToolTypeCli::GfToolsFontmake, None)?,
                        },
                    ],
                })
            },

            // Glyphs app preset. Ignore tool types.
            (Some(Preset::GlyphsApp), _, Some(tool_1_path), _, Some(tool_2_path)) => {
                Ok(RunConfiguration {
                    tool_pairs: vec![
                        ToolPair {
                            tool_1: Tool::from_cli_args(ToolTypeCli::GlyphsApp, Some(tool_1_path.into()))?,
                            tool_2: Tool::from_cli_args(ToolTypeCli::GlyphsApp, Some(tool_2_path.into()))?,
                        },
                    ],
                })
            },

            // Two tool types specified, no preset.
            (None, Some(tool_1_type), tool_1_path, Some(tool_2_type), tool_2_path) => {
                let tool_1 = Tool::from_cli_args(tool_1_type, tool_1_path.map(|p| p.into()))?;
                let tool_2 = Tool::from_cli_args(tool_2_type, tool_2_path.map(|p| p.into()))?;

                // (The following logic is based on the `ttx_diff.py` script.)
                // Currently, having two tools of the same type is supported for Glyphs app
                // instances only. These must have at least different build numbers.
                if tool_1.tool_type() == tool_2.tool_type() {
                    if tool_1.tool_type() != ToolType::GlyphsApp {
                        return Err(RunConfigurationError::InvalidToolCombination(
                            tool_1.tool_type(),
                            tool_2.tool_type(),
                        ));
                    }

                    // Both tools are Glyphs apps.
                    if tool_1.version() == tool_2.version() {
                        return Err(RunConfigurationError::SameToolBuilds);
                    }
                }

                Ok(RunConfiguration {
                    tool_pairs: vec![
                        ToolPair {
                            tool_1: tool_1,
                            tool_2: tool_2,
                        },
                    ],
                })
            },

            _ => {
                return Err(RunConfigurationError::InvalidCliArguments);
            },
        }
    }

    pub fn has_glyphs_app(&self) -> bool {
        self.tool_pairs.iter().any(|pair| pair.has_glyphs_app())
    }

    fn tool_pair_for_category(&self) -> &ToolPair {
        &self.tool_pairs[0]
    }

    fn tool_1_for_category(&self) -> &Tool {
        &self.tool_pair_for_category().tool_1
    }

    fn tool_2_for_category(&self) -> &Tool {
        &self.tool_pair_for_category().tool_2
    }

    pub fn tool_1_category_type(&self) -> ToolType {
        self.tool_1_for_category().tool_type()
    }

    pub fn tool_2_category_type(&self) -> ToolType {
        self.tool_2_for_category().tool_type()
    }

    /// Use this as the category of the first tool in the HTML report.
    pub fn tool_1_category_name(&self) -> String {
        self.tool_1_for_category().pretty_name()
    }

    /// Use this as the category of the second tool in the HTML report.
    pub fn tool_2_category_name(&self) -> String {
        self.tool_2_for_category().pretty_name()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum RunConfigurationError {
    #[error("Invalid CLI arguments: Specify either a preset or two tool types, including paths if required")]
    InvalidCliArguments,

    #[error("Invalid combination of tool types: {0}, {1}")]
    InvalidToolCombination(ToolType, ToolType),

    #[error("Must specify two different tool builds")]
    SameToolBuilds,

    #[error("Tool could not be created from CLI arguments")]
    ToolConversion(#[from] ToolConversionError),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolPair {
    pub tool_1: Tool,
    pub tool_2: Tool,
}

impl ToolPair {
    pub fn has_gftools(&self) -> bool {
        self.tool_1.tool_management() == ToolManagement::ManagedByGfTools ||
        self.tool_2.tool_management() == ToolManagement::ManagedByGfTools
    }

    pub fn has_glyphs_app(&self) -> bool {
        self.tool_1.tool_type() == ToolType::GlyphsApp || 
        self.tool_2.tool_type() == ToolType::GlyphsApp
    }
}
