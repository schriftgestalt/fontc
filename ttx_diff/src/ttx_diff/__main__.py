"""A tool for comparing font compiler outputs (`fontc`, `fontmake`, Glyphs.app)."""

from absl import app, flags

from ttx_diff import core

_COMPARE_DEFAULTS = "default"
_COMPARE_GFTOOLS = "gftools"

# Flag definitions (moved from core.py so they appear in --help)
flags.DEFINE_boolean(
    "version", False, "Show application version and exit.", short_name="V"
)
flags.DEFINE_enum_class(
    "tool_1_type",
    default=None,
    enum_class=core.ToolType,
    help="Choose the type of the first tool which should be used to build fonts to compare. Note that as of 2023-05-21, we still set flags for `fontmake` to match `fontc` behavior.",
)
flags.DEFINE_string(
    "tool_1_path",
    default=None,
    help="For `fontc`: Optional path to precompiled `fontc` binary to be used as tool 1.\nFor `glyphsapp`: Required path to the application bundle of the specific Glyphs app version to be used as tool 1.\nPlease note that if a different instance of the Glyphs app with the same major version is already running, that version may be used instead.",
)
flags.DEFINE_enum_class(
    "tool_2_type",
    default=None,
    enum_class=core.ToolType,
    help="Choose the type of the second tool which should be used to build fonts to compare. Note that as of 2023-05-21, we still set flags for `fontmake` to match `fontc` behavior.",
)
flags.DEFINE_string(
    "tool_2_path",
    default=None,
    help="For `fontc`: Optional path to precompiled `fontc` binary to be used as tool 2.\nFor `glyphsapp`: Required path to the application bundle of the specific Glyphs app version to be used as tool 2.\nPlease note that if a different instance of the Glyphs app with the same major version is already running, that version may be used instead.",
)
flags.DEFINE_string(
    "config",
    default=None,
    help="config.yaml to be passed to gftools in gftools mode",
)
flags.DEFINE_string(
    "normalizer_path",
    default=None,
    help="Optional path to precompiled otl-normalizer binary",
)
flags.DEFINE_string(
    "cache_path",
    default="~/.fontc_crater_cache",
    help="Optional path to custom cache location for font repositories.",
)
flags.DEFINE_enum(
    "rebuild",
    default="both",
    enum_values=["both", core.TOOL_1_NAME, core.TOOL_2_NAME, "none"],
    help="Which compilers to rebuild with if the output appears to already exist. `none` is handy when playing with `ttx_diff.py` itself.",
)
flags.DEFINE_float(
    "off_by_one_budget",
    default=0.1,
    help="The percentage of point (glyf) or delta (gvar) values allowed to differ by one without counting as a diff.",
)
flags.DEFINE_bool(
    "json", default=False, help="Print results in machine-readable JSON format."
)
flags.DEFINE_string("outdir", default=None, help="directory to store generated files")
flags.DEFINE_bool(
    "production_names",
    default=True,
    help="Rename glyphs to AGL-compliant names (uniXXXX, etc.) suitable for production. Disable to see the original glyph names.",
)

# fontmake - and so gftools' - static builds perform overlaps removal, but fontc
# can't do that yet, and so we default to disabling the filter to make the diff
# less noisy.
# TODO: Change the default if/when fontc gains the ability to remove overlaps.
# https://github.com/googlefonts/fontc/issues/975
flags.DEFINE_bool(
    "keep_overlaps",
    default=True,
    help="Keep overlaps when building static fonts. Disable to compare with simplified outlines.",
)
flags.DEFINE_bool(
    "keep_direction",
    default=False,
    help="Preserve contour winding direction from source.",
)
flags.DEFINE_string(
    "font_1",
    default=None,
    help="Optional path to precompiled font 1. Must be used with --font_2.",
)
flags.DEFINE_string(
    "font_2",
    default=None,
    help="Optional path to precompiled font 2. Must be used with --font_1.",
)


def main():
    """Entry point for running ttx-diff as a module."""
    app.run(core.main)


if __name__ == "__main__":
    main()
