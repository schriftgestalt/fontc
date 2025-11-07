#!/usr/bin/env python3
"""Helper for comparing `fontc` (Rust) vs `fontmake` (Python) or Glyphs 3 vs 
Glyphs 4 binaries.

Turns each into `ttx`, eliminates expected sources of difference and prints
a brief summary of the result.

`fontmake` should be installed in an active virtual environment. The easiest
way to do this is probably with `pyenv` and its `pyenv-virtualenv` plugin (see
https://github.com/pyenv/pyenv-virtualenv).

Usage:
    # Rebuild with `fontc` and `fontmake` and compare.
    python resources/scripts/ttx_diff.py --tool_1_type fontc --tool_2_type fontmake ../OswaldFont/sources/Oswald.glyphs

    # Rebuild the `fontc` copy, reuse the prior `fontmake` copy (if present), 
    # and compare. Useful if you are making changes to `fontc` meant to narrow 
    # the diff.
    python resources/scripts/ttx_diff.py --tool_1_type fontc --tool_2_type fontmake --rebuild fontc ../OswaldFont/sources/Oswald.glyphs

    # Rebuild with `fontc` and `fontmake` (managed by `gftools`) and compare.
    # This requires a config file.
    python resources/scripts/ttx_diff.py --tool_1_type fontc_gftools --tool_2_type fontmake_gftools --config ../OswaldFont/sources/config.yaml ../OswaldFont/sources/Oswald.glyphs

    # Rebuild with Glyphs 3 and Glyphs 4 and compare. The paths to the app
    # bundles have to be specified (see below for an exception to this).
    python resources/scripts/ttx_diff.py --tool_1_type glyphsapp --tool_1_path '/Applications/Glyphs 3.3.1 (3343).app' --tool_2_type glyphsapp --tool_2_path '/Applications/Glyphs 4.0a (3837).app' ../OswaldFont/sources/Oswald.glyphs
    
    # For debugging of Glyphs app only: Rebuild with Glyphs 3 and Glyphs 4 and compare.
    # If tool path 1 is missing, tool 1 is automatically set to Glyphs 3, and likewise
    # tool 2 to Glyphs 4. Running instances are opened if possible.
    # The instances are _not_ terminated after the builds. Please note that this conflicts 
    # with `fontc_crater` builds, which expect to be able to launch multiple instances 
    # of the same Glyphs app build and thus require the instances to be terminated.
    python resources/scripts/ttx_diff.py --tool_1_type glyphsapp --tool_2_type glyphsapp ../OswaldFont/sources/Oswald.glyphs
    
    # Rebuild with precompiled `fontc` binary and Glyphs 4 and compare.
    python resources/scripts/ttx_diff.py --tool_1_type fontc --tool_1_path ~/fontc/target/release/fontc --tool_2_type glyphsapp --tool_2_path '/Applications/Glyphs 4.0a (3837).app' ../OswaldFont/sources/Oswald.glyphs

JSON:
    If the `--json` flag is passed, this tool will output JSON.

    If both compilers ran successfully, this dictionary will have a single key,
    "success", which will contain a dictionary, where keys are the tags of tables
    (or another identifier) and the value is either a float representing the
    'difference ratio' (where 1.0 means identical and 0.0 means maximally
    dissimilar) or, if only one compiler produced that table, the name of that
    compiler as a string ("fontc", "glyphsapp_4" etc.).
    For example, when run in `default` or `gftools` modes, the output 
    `{"success": { "GPOS": 0.99, "vmxt": "fontmake" }}` means that the "GPOS"
    table was 99%% similar, and only `fontmake` produced the "vmtx" table (and 
    all other tables were identical).

    If one or both of the compilers fail to exit successfully, we will return a
    dictionary with the single key, "error", where the payload is a dictionary
    where keys are the name of the compiler that failed, and the body is a
    dictionary with "command" and "stderr" fields, where the "command" field
    is the command that was used to run that compiler.
    
Caching:
    It is possible to specify a custom cache directory for the font
    repositories. This is especially useful when a custom cache location is used
    for `fontc_crater`, as the same cache can be shared.
"""

import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
import uuid
import yaml

from abc import ABC, abstractmethod
from absl import app, flags
from cdifflib import CSequenceMatcher as SequenceMatcher
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from fontTools.designspaceLib import DesignSpaceDocument
from fontTools.misc.fixedTools import otRound
from fontTools.ttLib import TTFont
from fontTools.varLib.iup import iup_delta
from Foundation import NSConnection, NSDistantObject, NSObject, NSURL
from functools import cache
from glyphsLib import GSFont
from lxml import etree
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Since `absl` flags do not support overriding the generated flag names, the
# string values should match the lowercase versions of the constants (i. e.
# include the same underscores).
#
# Note:  This is currently a flat list like the `ToolTypeCli` enum in
# `fontc_crater/src/args.rs`. It does not separate the concepts of tool type,
# tool management and tool instance.
class ToolType(StrEnum):
    # Standalone versions
    FONTC = "fontc"
    FONTMAKE = "fontmake"

    # Managed by GFTools
    FONTC_GFTOOLS = "fontc_gftools"
    FONTMAKE_GFTOOLS = "fontmake_gftools"

    GLYPHSAPP = "glyphsapp"

FONTC_NAME = "fontc"
FONTMAKE_NAME = "fontmake"
TOOL_1_NAME = "tool_1"
TOOL_2_NAME = "tool_2"
TTX_NAME = "ttx"

FLAGS = flags.FLAGS

# used instead of a tag for the normalized mark/kern output
MARK_KERN_NAME = "(mark/kern)"

LIG_CARET_NAME = "ligcaret"

# maximum chars of stderr to include when reporting errors; prevents
# too much bloat when run in CI
MAX_ERR_LEN = 1000

# fontc and fontmake's builds may be off by a second or two in the
# head.created/modified; setting this makes them the same
if "SOURCE_DATE_EPOCH" not in os.environ:
    os.environ["SOURCE_DATE_EPOCH"] = str(int(time.time()))


# print to stderr
def eprint(*objects):
    print(*objects, file=sys.stderr)


flags.DEFINE_enum_class(
    "tool_1_type",
    default = None,
    enum_class = ToolType,
    help = "Choose the type of the first tool which should be used to build fonts to compare. Note that as of 2023-05-21, we still set flags for `fontmake` to match `fontc` behavior.",
)
flags.DEFINE_string(
    "tool_1_path",
    default = None,
    help = "For `fontc`: Optional path to precompiled `fontc` binary to be used as tool 1.\nFor `glyphsapp`: Required path to the application bundle of the specific Glyphs app version to be used as tool 1.\nPlease note that if a different instance of the Glyphs app with the same major version is already running, that version may be used instead.",
)
flags.DEFINE_enum_class(
    "tool_2_type",
    default = None,
    enum_class = ToolType,
    help = "Choose the type of the second tool which should be used to build fonts to compare. Note that as of 2023-05-21, we still set flags for `fontmake` to match `fontc` behavior.",
)
flags.DEFINE_string(
    "tool_2_path",
    default = None,
    help = "For `fontc`: Optional path to precompiled `fontc` binary to be used as tool 2.\nFor `glyphsapp`: Required path to the application bundle of the specific Glyphs app version to be used as tool 2.\nPlease note that if a different instance of the Glyphs app with the same major version is already running, that version may be used instead.",
)
flags.DEFINE_string(
    "config",
    default = None,
    help = "`config.yaml` to be passed to `gftools` when using a tool managed by `gftools`",
)
flags.DEFINE_string(
    "normalizer_path",
    default = None,
    help = "Optional path to precompiled `otl-normalizer` binary",
)
flags.DEFINE_string(
    "cache_path",
    default = "~/.fontc_crater_cache",
    help = "Optional path to custom cache location for font repositories.",
)
flags.DEFINE_enum(
    "rebuild",
    default = "both",
    enum_values = ["both", TOOL_1_NAME, TOOL_2_NAME, "none"],
    help = "Which compilers to rebuild with if the output appears to already exist. `none` is handy when playing with `ttx_diff.py` itself.",
)
flags.DEFINE_float(
    "off_by_one_budget",
    default = 0.1,
    help = "The percentage of point (glyf) or delta (gvar) values allowed to differ by one without counting as a diff.",
)
flags.DEFINE_bool(
    "json",
    default = False,
    help = "Print results in machine-readable JSON format."
)
flags.DEFINE_string(
    "outdir",
    default = None,
    help = "Directory to store generated files."
)
flags.DEFINE_bool(
    "production_names",
    default = True,
    help = "Rename glyphs to AGL-compliant names (uniXXXX, etc.) suitable for production. Disable to see the original glyph names.",
)

# fontmake - and so gftools' - static builds perform overlaps removal, but fontc
# can't do that yet, and so we default to disabling the filter to make the diff
# less noisy.
# TODO: Change the default if/when fontc gains the ability to remove overlaps.
# https://github.com/googlefonts/fontc/issues/975
flags.DEFINE_bool(
    "keep_overlaps",
    default = True,
    help = "Keep overlaps when building static fonts. Disable to compare with simplified outlines.",
)
flags.DEFINE_bool(
    "keep_direction", False, "Preserve contour winding direction from source."
)


def to_xml_string(e) -> str:
    xml = etree.tostring(e)
    # some table diffs were mismatched because of inconsistency in ending newline
    xml = xml.strip()
    return xml


@cache
def home_str() -> str:
    return str(Path("~").expanduser())


def rel_user(fragment: Any) -> str:
    """If an absolute path is in the home directory convert to ~ form for display.

    Makes cli output closer to what you'd actually type, e.g. ~/something
    Instead of /usr/blah/blah/blah/something"""
    fragment = str(fragment)
    home = home_str()
    if fragment.startswith(home):
        fragment = "~" + fragment[len(home) :]
    return fragment


#===============================================================================
# Running commands
#===============================================================================


# execute a command after logging it to stderr.
# All additional kwargs are passed to subprocess.run
def log_and_run(cmd: Sequence, cwd=None, **kwargs):
    # Convert to ~ format because it's annoying to see really long usr paths
    log_cmd = " ".join(rel_user(c) for c in cmd)
    if cwd is not None:
        eprint(f"  (cd {rel_user(cwd)} && {log_cmd})")
    else:
        eprint(f"  ({log_cmd})")
    return subprocess.run(
        cmd,
        text=True,
        cwd=cwd,
        capture_output=True,
        **kwargs,
    )


def run_ttx(font_file: Path):
    ttx_file = font_file.with_suffix(".ttx")
    # if this exists we're allowed to reuse it
    if ttx_file.is_file():
        eprint(f"reusing {rel_user(ttx_file)}")
        return ttx_file

    cmd = [
        TTX_NAME,
        "-o",
        ttx_file.name,
        font_file.name,
    ]
    log_and_run(cmd, font_file.parent, check=True)
    return ttx_file


# generate a simple text repr for gpos for this font, with retry
def run_normalizer(normalizer_bin: Path, font_file: Path, table: str):
    if table == "gpos":
        out_path = font_file.with_suffix(".markkern.txt")
    elif table == "gdef":
        out_path = font_file.with_suffix(f".{LIG_CARET_NAME}.txt")
    else:
        raise ValueError(f"unknown table for normalizer: '{table}'")

    if out_path.exists():
        eprint(f"reusing {rel_user(out_path)}")
    NUM_RETRIES = 5
    for i in range(NUM_RETRIES + 1):
        try:
            return try_normalizer(normalizer_bin, font_file, out_path, table)
        except subprocess.CalledProcessError as e:
            time.sleep(0.1)
            if i >= NUM_RETRIES:
                raise e
            eprint(f"normalizer failed with code '{e.returncode}'', retrying")


# we had a bug where this would sometimes hang in mysterious ways, so we may
# call it multiple times if it fails
def try_normalizer(normalizer_bin: Path, font_file: Path, out_path: Path, table: str):
    NORMALIZER_TIMEOUT = 60 * 10  # ten minutes
    if not out_path.is_file():
        cmd = [
            normalizer_bin.absolute(),
            font_file.name,
            "-o",
            out_path.name,
            "--table",
            table,
        ]
        log_and_run(cmd, font_file.parent, check=True, timeout=NORMALIZER_TIMEOUT)
        # if we finished running and there's no file then there's no output:
        if not out_path.is_file():
            return ""
    with open(out_path) as f:
        return f.read()


#===============================================================================
# Building fonts with `fontc` and `fontmake`
#===============================================================================


class BuildFail(Exception):
    """An exception raised if a compiler fails."""

    def __init__(self, cmd: Sequence, msg: str):
        self.command = list(str(c) for c in cmd)
        self.msg = msg


# run a font compiler
def build(cmd: Sequence, build_dir: Optional[Path], **kwargs):
    output = log_and_run(cmd, build_dir, **kwargs)
    if output.returncode != 0:
        raise BuildFail(cmd, output.stderr or output.stdout)


# Standalone `fontc` (not managed by `gftools`)
def build_fontc(source: Path, fontc_bin: Path, build_dir: Path, font_file_name: str):
    out_file = build_dir / font_file_name
    if out_file.exists():
        eprint(f"reusing {rel_user(out_file)}")
        return
    cmd = [
        fontc_bin,
        "--build-dir",
        ".",
        "-o",
        out_file.name,
        source,
        "--emit-debug",
    ]
    if FLAGS.keep_direction:
        cmd.append("--keep-direction")
    if not FLAGS.production_names:
        cmd.append("--no-production-names")
    build(cmd, build_dir)


# Standalone `fontmake` (not managed by `gftools`)
def build_fontmake(source: Path, build_dir: Path, font_file_name: str):
    out_file = build_dir / font_file_name
    if out_file.exists():
        eprint(f"reusing {rel_user(out_file)}")
        return
    variable = source_is_variable(source)
    buildtype = "variable"
    if not variable:
        buildtype = "ttf"
    cmd = [
        FONTMAKE_NAME,
        "-o",
        buildtype,
        "--output-path",
        out_file.name,
        "--drop-implied-oncurves",
        # helpful for troubleshooting
        "--debug-feature-file",
        "debug.fea",
    ]
    if FLAGS.keep_direction:
        cmd.append("--keep-direction")
    if not FLAGS.production_names:
        cmd.append("--no-production-names")
    if FLAGS.keep_overlaps and not variable:
        cmd.append("--keep-overlaps")
    cmd.append(str(source))

    build(cmd, build_dir)


@contextmanager
def modified_gftools_config(
    maybe_path_to_move_config: Optional[Path],
    cmdline: List[str],
    extra_args: Sequence[str],
) -> Generator[None, None, None]:
    """Modify the gftools config file to add extra arguments.

    A temporary config file is created with the extra args added to the
    `extraFontmakeArgs` key, and replaces the original config file in the
    command line arguments' list, which is modified in-place.
    The temporary file is deleted after the context manager exits.
    If the extra_args list is empty, the context manager does nothing.

    Args:
        cmdline: The command line arguments passed to gftools. This must include
            the path to a config.yaml file.
        extra_args: Extra arguments to add to the config file. Can be empty.
    """
    if extra_args:
        try:
            config_idx, config_path = next(
                (
                    (i, arg)
                    for i, arg in enumerate(cmdline)
                    if arg.endswith((".yaml", ".yml"))
                )
            )
        except StopIteration:
            raise ValueError(
                "No config file found in command line arguments. "
                "Please provide a config.yaml file."
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["extraFontmakeArgs"] = " ".join(
            config.get("extraFontmakeArgs", "").split(" ") + extra_args
        )

        config_dir = maybe_path_to_move_config or Path(config_path).parent

        with NamedTemporaryFile(
            mode="w",
            prefix="config_",
            suffix=".yaml",
            delete=False,
            dir=config_dir,
        ) as f:
            yaml.dump(config, f)
        temp_path = Path(f.name)

        cmdline[config_idx] = temp_path

    # if the build later fails for any reason, we still want to delete the temp file
    try:
        yield
    finally:
        if extra_args:
            temp_path.unlink()


def run_gftools(
    source: Path, config_path: Path, build_dir: Path, font_file_name: str, fontc_bin: Optional[Path] = None
):
    out_file = build_dir / font_file_name
    if out_file.exists():
        eprint(f"reusing {rel_user(out_file)}")
        return
    out_dir = build_dir / "gftools_temp_dir"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        "gftools",
        "builder",
        config_path,
        "--experimental-simple-output",
        out_dir,
        "--experimental-single-source",
        source.name,
    ]
    if fontc_bin is not None:
        cmd += ["--experimental-fontc", fontc_bin]

    extra_args = []
    if FLAGS.keep_overlaps:
        # (we only want this for the statics but it's a noop for variables)
        extra_args.append("--keep-overlaps")
    if not FLAGS.production_names:
        extra_args.append("--no-production-names")

    maybe_new_config_path = path_to_move_external_config(source, config_path)
    with modified_gftools_config(maybe_new_config_path, cmd, extra_args):
        build(cmd, None)

    # return a concise error if gftools produces != one output
    contents = list(out_dir.iterdir()) if out_dir.exists() else list()
    if not contents:
        raise BuildFail(cmd, "gftools produced no output")
    elif len(contents) != 1:
        contents = [p.name for p in contents]
        raise BuildFail(cmd, f"gftools produced multiple outputs: {contents}")
    copy(contents[0], out_file)

    if out_dir.exists():
        shutil.rmtree(out_dir)


def path_to_move_external_config(source: Path, config: Path) -> Optional[Path]:
    source_repo = find_repo_root(source)
    config_repo = find_repo_root(config)

    if source_repo != config_repo:
        return source_repo


def find_repo_root(path: Path) -> Optional[Path]:
    if path.is_dir() and path.joinpath(".git").exists():
        return path
    elif path.parent == path:
        return None
    else:
        return find_repo_root(path.parent)


def source_is_variable(path: Path) -> bool:
    if path.suffix == ".ufo":
        return False
    if path.suffix == ".designspace":
        dspace = DesignSpaceDocument.fromfile(path)
        return len(dspace.sources) > 1
    if path.suffix == ".glyphs" or path.suffix == ".glyphspackage":
        font = GSFont(path)
        return len(font.masters) > 1
    # fallback to variable, the existing default, but we should never get here?
    return True


def copy(old, new):
    shutil.copyfile(old, new)
    return new


#===============================================================================
# Building fonts with Glyphs.app
#===============================================================================


# Initial number of tries to establish a JSTalk connection to an already running
# instance of Glyphs.app.
INITIAL_CONNECTION_TRIES = 3


# Maximum number of tries to establish a JSTalk connection to a Glyphs.app
# instance that we just launched.
MAX_CONNECTION_TRIES = 10


BASE_BUNDLE_IDENTIFIER = "com.GeorgSeifert.Glyphs"


# `fontc_crater/ttx_diff_runner.rs` expects nothing but JSON on `stdout`,
# so we log status messages only if we are not in JSON output mode.
def print_if_not_json(message: str):
    if not FLAGS.json:
        print(message)


# This function has been adapted from `Glyphs remote scripts/Glyphs.py`
# in the `https://github.com/schriftgestalt/GlyphsSDK` repository.
def application_with_version(major_version: int) -> (NSDistantObject, str, str):
    bundle_identifier = f"{BASE_BUNDLE_IDENTIFIER}{major_version}"

    # First, try to establish a connection to a running app matching the
    # bundle identifier. We need only a low number of maximum tries because we
    # would not have to wait for the app to launch.
    proxy = application_proxy(bundle_identifier, INITIAL_CONNECTION_TRIES)
    if proxy is not None:
        version = proxy.versionString()
        build_number = proxy.buildNumber()
        if version is None or build_number is None:
            print(f"Cannot determine version or build number for application with bundle identifier {bundle_identifier}")
            return None, None, None

        print(f"Found running instance of Glyphs {version} (build {build_number})")
        return proxy, version, build_number

    # There is no running app, or another error occurred.
    # Try to launch the app via the bundle identifier.
    # `-F` opens the application “fresh” (without restoring windows). This is important 
    #  when relaunching the app after a crash.
    # `-g` opens the application in the background.
    # `-j` opens the application hidden (its windows are not visible).
    # We are _not_ launching a new instance in this case.
    print_if_not_json(f"Opening application for bundle identifier {bundle_identifier}")
    os.system(f"open -b {bundle_identifier} -F -g -j")

    # Try to establish a connection to the running app.
    proxy = application_proxy(bundle_identifier, MAX_CONNECTION_TRIES)
    if proxy is None:
        return None, None, None

    version = proxy.versionString()
    build_number = proxy.buildNumber()
    if version is None or build_number is None:
        print(f"Cannot determine version or build number for application with bundle identifier {bundle_identifier}")
        return None, None, None

    print(f"Launched instance of Glyphs {version} (build {build_number})")
    return proxy, version, build_number


# This function has been adapted from `Glyphs remote scripts/Glyphs.py`
# in the `https://github.com/schriftgestalt/GlyphsSDK` repository.
def application_at_path(bundle_path: str) -> (NSDistantObject, str, str):
    version, build_number = app_bundle_version(bundle_path)
    if version is None or build_number is None:
        print(f"Cannot determine version or build number for application bundle at {bundle_path}")
        return None, None, None

    major_version = version.split('.', 1)[0]

    # The registered name of a Glyphs app instance can include the build number.
    # We add a UUID to support multiple instances of the same build running simultaneously.
    # The UUID is passed to the app when launching it with the `open` command below.
    proxy_uuid = uuid.uuid4()
    registered_name = f"{BASE_BUNDLE_IDENTIFIER}{major_version}.{build_number}.{proxy_uuid}"

    # First, try to establish a connection to a running app matching the
    # registered name. We need only a low number of maximum tries because we
    # would not have to wait for the app to launch.
    proxy = application_proxy(registered_name, INITIAL_CONNECTION_TRIES)
    if proxy is not None:
        print(f"Found running instance of Glyphs {version} (build {build_number})")
        return proxy, version, build_number

    # There is no running app, or another error occurred. Try to launch the app
    # via the provided bundle path.
    # `-F` opens the application “fresh” (without restoring windows). This is important 
    #  when relaunching the app after a crash.
    # `-g` opens the application in the background.
    # `-j` opens the application hidden (its windows are not visible).
    # `-n` opens a new instance of the application, enabling multiple simultaneous exports.
    print_if_not_json(f"Opening application at {bundle_path}")
    os.system(f"open -a '{bundle_path}' -F -g -j -n --args -GSProxyPortNameUUID {proxy_uuid}")

    # Try to establish a connection to the running app.
    proxy = application_proxy(registered_name, MAX_CONNECTION_TRIES)
    if proxy is None:
        return None, None, None

    print(f"Launched instance of Glyphs {version} (build {build_number})")
    return proxy, version, build_number


# Returns the version and build numbers of the application at the given bundle
# path. Returns `None` for values that cannot be found.
def app_bundle_version(bundle_path) -> (Optional[str], Optional[str]):
    info_plist_path = os.path.join(bundle_path, "Contents", "Info.plist")
    if not os.path.exists(info_plist_path):
        raise FileNotFoundError(f"Info.plist not found at: {info_plist_path}")

    with open(info_plist_path, 'rb') as f:
        plist = plistlib.load(f)

    # CFBundleShortVersionString is the user-facing version
    # CFBundleVersion is typically a build number
    version = plist.get("CFBundleShortVersionString", None)
    build_number = plist.get("CFBundleVersion", None)
    return version, build_number


# This function has been adapted from `Glyphs remote scripts/Glyphs.py`
# in the `https://github.com/schriftgestalt/GlyphsSDK` repository.
#
# The registered name should have the following format:
#
#     com.GeorgSeifert.Glyphs(V)[.BBBB][.UUID]
#
# `V` is the major version (currently 3 or 4).
# `BBBB` is an optional build number (e. g. 3343 for Glyphs 3.3.1).
# `UUID` is an optional UUID that identifies one instance of a given build.
# This is required when running multiple instances of the same build to speed up `fontc_crater`.
#
# Returns `None` if the connection could not be established.
def application_proxy(registered_name: str, max_tries: int) -> NSDistantObject:
    conn = None
    tries = 1

    print_if_not_json(f"Looking for a JSTalk connection to {registered_name}")
    while ((conn is None) and (tries < max_tries)):
        conn = NSConnection.connectionWithRegisteredName_host_(registered_name, None)
        tries = tries + 1

        if (not conn):
            print_if_not_json(f"Could not find connection, trying again ({tries} of {max_tries})")
            time.sleep(1)

    if (not conn):
        print_if_not_json(f"Failed to find a JSTalk connection to {registered_name}")
        return None

    return conn.rootProxy()


# This function has been adapted from `Glyphs remote scripts/testExternal.py`
# in the `https://github.com/schriftgestalt/GlyphsSDK` repository.
def exportFirstInstance(
    proxy: NSDistantObject, 
    source: Path, 
    build_dir: Path, 
    out_file_name: str,
    should_terminate: bool
):
    out_file = build_dir / out_file_name
    if out_file.exists():
        print_if_not_json(f"reusing {out_file}")
        return

    source_path = source.as_posix()
    build_dir_path = build_dir.as_posix()

    doc = proxy.openDocumentWithContentsOfFile_display_(source_path, False)
    print_if_not_json(f"Exporting {doc.displayName()} to {build_dir_path}")

    font = doc.font()
    instance = font.instances()[0]

    arguments = {
        'ExportFormat': "OTF",
        'Destination': NSURL.fileURLWithPath_(build_dir_path)
    }
    if not FLAGS.production_names:
        # Glyphs defaults to True.
        arguments['useProductionNames'] = False
    if FLAGS.keep_overlaps:
        # Glyphs defaults to True for non-variable fonts.
        arguments['removeOverlap'] = False

    try:
        result = instance.generate_(arguments)
    except Exception as e:
        # Catch Glyphs app crash and report as error.
        cmd = []
        cmd.append(str(source))
        cmd.append(str(out_file_name))

        if should_terminate:
            # Terminate app just in case.
            terminate_macos_app(proxy)

        raise BuildFail(cmd, str(e))

    doc.close()
    
    if result[0] != "Success":
        # Even though we do not have an actual command for Glyphs exports,
        # we construct one which can be passed to `BuildFail`.
        cmd = []
        for k, v in arguments.items():
            cmd.append(k)
            cmd.append(v)

        cmd.append(str(source))
        cmd.append(str(out_file_name))

        # If not successful, `result` is a list of `NSError` objects.
        errors_string = "; ".join(error.localizedDescription() for error in result)

        if should_terminate:
            # Terminate app before exiting the function.
            terminate_macos_app(proxy)

        raise BuildFail(cmd, errors_string)
    
    # Rename file to our standard name.
    temp_file = Path(result[1]).resolve()
    temp_file_name = temp_file.name
    file_to_rename = build_dir / temp_file_name
    file_to_rename.rename(out_file)

    if should_terminate:
        # Terminate app exiting the function.
        terminate_macos_app(proxy)


def terminate_macos_app(
    proxy: NSDistantObject
):
    try:
        proxy.terminate_(None)
    except Exception as e:
        if 'NSInvalidReceivePortException' in str(e):
            print_if_not_json("Application is terminating. Ignoring expected exception.")
        else:
            cmd = []
            raise BuildFail(cmd, str(e))


#===============================================================================
# Normalizing tables
#===============================================================================


def get_name_to_id_map(ttx: etree.ElementTree):
    return {
        el.attrib["nameID"]: el.text
        for el in ttx.xpath(
            "//name/namerecord[@platformID='3' and @platEncID='1' and @langID='0x409']"
        )
    }


def name_id_to_name(ttx, xpath, attr):
    id_to_name = get_name_to_id_map(ttx)
    for el in ttx.xpath(xpath):
        if attr is None:
            if el.text is None:
                continue
            name_id = el.text
            # names <= 255 have specific assigned slots, names > 255 not
            if int(name_id) <= 255:
                continue
            name = id_to_name.get(name_id, f"NonExistingNameID {name_id}").strip()
            el.text = name
        else:
            if attr not in el.attrib:
                continue
            # names <= 255 have specific assigned slots, names > 255 not
            name_id = el.attrib[attr]
            if int(name_id) <= 255:
                continue
            name = id_to_name.get(name_id, f"NonExistingNameID {name_id}").strip()
            el.attrib[attr] = name


def normalize_name_ids(ttx: etree.ElementTree):
    name = ttx.find("name")
    if name is None:
        return

    records = name.xpath(".//namerecord")
    for record in records:
        name.remove(record)
        # User-defined name IDs get replaced by there value where they are
        # used, so the ID itself is not interesting and we replace them with a
        # fixed value
        if int(record.attrib["nameID"]) > 255:
            record.attrib["nameID"] = "USER_ID"

    records = sorted(
        records,
        key=lambda x: (
            x.attrib["platformID"],
            x.attrib["platEncID"],
            x.attrib["langID"],
            x.attrib["nameID"],
            x.text,
        ),
    )

    for record in records:
        name.append(record)

    # items keep their indentation when we reorder them, so reindent everything
    etree.indent(name)


def find_table(ttx, tag):
    return select_one(ttx, f"/ttFont/{tag}")


def select_one(container, xpath):
    el = container.xpath(xpath)
    if len(el) != 1:
        raise IndexError(f"Wanted 1 name element, got {len(el)}")
    return el[0]


def drop_weird_names(ttx):
    drops = []
    d = ttx.xpath(
        "//name/namerecord[@platformID='1' and @platEncID='0' and @langID='0x0']"
    )
    if d:
        drops.extend(d)
    
    d = ttx.xpath(
        "//name/namerecord[@nameID='5']"
    )
    if d:
        drops.extend(d)
    for drop in drops:
        drop.getparent().remove(drop)


def erase_checksum_and_dates(ttx):
    el = select_one(ttx, "//head/checkSumAdjustment")
    del el.attrib["value"]
    el = select_one(ttx, "//head/created")
    del el.attrib["value"]
    el = select_one(ttx, "//head/modified")
    del el.attrib["value"]


def stat_like_fontmake(ttx):
    try:
        el = find_table(ttx, "STAT")
    except IndexError:
        return
    ver = select_one(el, "Version")
    if ver.attrib["value"] != "0x00010002":
        # nop
        return

    # fontc likes to write STAT 1.2, fontmake prefers 1.1
    # Version 1.2 adds support for the format 4 axis value table
    # So until such time as we start writing format 4 axis value tables it doesn't matter
    ver.attrib["value"] = "0x00010001"


def normalize_all_offcurve_starting_point(contour):
    """Rotate an all-offcurve contour to start at point closest to origin.

    When multiple points are equidistant from the origin, the first one encountered
    (lowest index) is selected as the starting point.

    This is to address differences in starting points of all-offcurve TrueType
    quadratic contours between fontc and fontmake.

    fontmake preserves the original off-curve starting point from the UFO source,
    while fontc creates a synthetic on-curve point at the midpoint between first
    and last off-curve points due to kurbo::BezPath needing some on-curve point
    to move_to. When contours are reversed (default behavior without --keep-direction
    flag), this results in different starting points in the final TrueType output.

    See: <https://github.com/googlefonts/fontc/issues/1653>
    """
    pts = contour.xpath("./pt")
    if not pts or not all(pt.get("on") == "0" for pt in pts):
        return  # Not an all-offcurve contour

    # Find point closest to origin
    min_idx = min(
        range(len(pts)),
        key=lambda i: int(pts[i].get("x", 0)) ** 2 + int(pts[i].get("y", 0)) ** 2,
    )

    # Rotate if needed
    if min_idx != 0:
        contour[:] = pts[min_idx:] + pts[:min_idx]
        # Fix indentation after rotation
        etree.indent(contour, level=3)


# https://github.com/googlefonts/fontc/issues/1107
def normalize_glyf_contours(
    fontc_ttx: etree.ElementTree, fontmake_ttx: etree.ElementTree
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Reorders contours when they are identical between fontc and fontmake.

    If contours differ (e.g., different starting points), leaves
    them in their original order to avoid misleading diffs.

    For all-offcurve contours, normalizes the starting point to be the point
    closest to the origin before comparison.

    Returns a tuple of two dicts, one for fontc and one for fontmake, containing
    the new order of prior point indices for each glyph, later used for sorting
    gvar contours.
    """
    fontc_point_orders: dict[str, list[int]] = {}
    fontmake_point_orders: dict[str, list[int]] = {}

    # Get glyphs from both TTX trees
    fontc_glyphs = {g.attrib["name"]: g for g in fontc_ttx.xpath("//glyf/TTGlyph")}
    fontmake_glyphs = {
        g.attrib["name"]: g for g in fontmake_ttx.xpath("//glyf/TTGlyph")
    }

    # Only process glyphs that exist in both outputs
    for glyph_name in fontc_glyphs.keys() & fontmake_glyphs.keys():
        fontc_glyph = fontc_glyphs[glyph_name]
        fontmake_glyph = fontmake_glyphs[glyph_name]

        fontc_contours = fontc_glyph.xpath("./contour")
        fontmake_contours = fontmake_glyph.xpath("./contour")

        # Skip glyphs with mismatched contour counts
        if len(fontc_contours) != len(fontmake_contours):
            continue

        # Normalize all-offcurve contours to start at point closest to origin
        for contour in fontc_contours + fontmake_contours:
            normalize_all_offcurve_starting_point(contour)

        # Compare contours as sets to see if they're identical (ignoring order)
        fontc_strings = {to_xml_string(c) for c in fontc_contours}
        fontmake_strings = {to_xml_string(c) for c in fontmake_contours}

        if fontc_strings == fontmake_strings:
            # Contours are identical, just in different order - normalize both
            _normalize_single_glyph(fontc_glyph, fontc_contours, fontc_point_orders)
            _normalize_single_glyph(
                fontmake_glyph, fontmake_contours, fontmake_point_orders
            )
        # If sets don't match, skip normalization - leave original order

    return fontc_point_orders, fontmake_point_orders


def _normalize_single_glyph(
    glyph: etree.Element,
    contours: list[etree.Element],
    point_orders: dict[str, list[int]],
):
    """Helper function to normalize contour order within a single glyph.

    Contours are sorted alphabetically by xml string representation.

    The `point_orders` dict is used to store the new order of prior point
    indices for this glyph.
    """
    # annotate each contour with the range of point indices it covers
    with_range: list[tuple[range, etree.Element]] = []
    points_seen = 0
    for contour in contours:
        points_here = len(contour.xpath("./pt"))
        with_range.append((range(points_seen, points_seen + points_here), contour))
        points_seen += points_here
    annotated = sorted(with_range, key=lambda a: to_xml_string(a[1]))

    # sort by string representation, and skip if nothing has changed
    normalized = [contour for _, contour in annotated]
    if normalized == contours:
        return

    # normalized contours should be inserted before any other TTGlyph's
    # subelements (e.g. instructions)
    for contour in contours:
        glyph.remove(contour)
    non_contours = list(glyph)
    for el in non_contours:
        glyph.remove(el)
    glyph.extend(normalized + non_contours)

    # store new indices order
    name = glyph.attrib["name"]
    point_orders[name] = [idx for indices, _ in annotated for idx in indices]


def normalize_gvar_contours(ttx: etree.ElementTree, point_orders: dict[str, list[int]]):
    """Reorder gvar points to match normalised glyf order."""

    for glyph in ttx.xpath("//gvar/glyphVariations"):
        name = glyph.attrib["glyph"]
        order = point_orders.get(name)

        # skip glyph if glyf normalisation did not change its point order
        if order is None:
            continue

        # apply the same order to every tuple
        for tup in glyph.xpath("./tuple"):
            deltas = tup.xpath("./delta")
            assert len(order) + 4 == len(deltas), "gvar is not dense"

            # reorder and change index to match new position
            reordered = []
            for new_idx, old_idx in enumerate(order):
                delta = deltas[old_idx]  # always present as gvars are densified
                delta.attrib["pt"] = str(new_idx)
                reordered.append(delta)
            reordered += deltas[-4:]  # phantom points

            # normalized points should be inserted after any other tuple
            # subelements (e.g. coordinates)
            for delta in deltas:
                tup.remove(delta)
            non_deltas = list(tup)
            for el in non_deltas:
                tup.remove(el)
            tup.extend(non_deltas + reordered)


def normalize_null_tags(ttx: etree.ElementTree, xpath: str, attr):
    """replace the tag 'NONE' with the tag '    '"""
    for el in ttx.xpath(xpath):
        if attr:
            if el.attrib.get(attr, "") == "NONE":
                el.attrib[attr] = "    "
        else:
            if el.text == "NONE":
                el.text = "    "


# https://github.com/googlefonts/fontc/issues/1173
def erase_type_from_stranded_points(ttx):
    for contour in ttx.xpath("//glyf/TTGlyph/contour"):
        points = contour.xpath("./pt")
        if len(points) == 1:
            points[0].attrib["on"] = "irrelevent"


# only fontc emits name 25 currently, don't treat that as an error
def allow_fontc_only_variations_postscript_prefix(tool_1, tool_2):
    xpath_to_name_25 = "/ttFont/name/namerecord[@nameID='25']"
    tool_1_name25 = tool_1.xpath(xpath_to_name_25)
    tool_2_name25 = tool_2.xpath(xpath_to_name_25)
    if tool_1_name25 and not tool_2_name25:
        for n in tool_1_name25:
            n.getparent().remove(n)
    elif tool_2_name25 and not tool_1_name25:
        for n in tool_2_name25:
            n.getparent().remove(n)


# This method is one of the few where it matters which tool is `fontc` and
# which is `fontmake`.
def allow_some_off_by_ones(fontc, fontmake, container, name_attr, coord_holder):
    fontmake_num_coords = len(fontmake.xpath(f"//{container}/{coord_holder}"))
    off_by_one_budget = int(FLAGS.off_by_one_budget / 100.0 * fontmake_num_coords)
    spent = 0
    if off_by_one_budget == 0:
        return

    coord_tag = coord_holder.rpartition("/")[-1]
    # put all the containers into a dict to make querying more efficient:

    fontc_items = {x.attrib[name_attr]: x for x in fontc.xpath(f"//{container}")}
    for fontmake_container in fontmake.xpath(f"//{container}"):
        name = fontmake_container.attrib[name_attr]
        fontc_container = fontc_items.get(name)
        if fontc_container is None:
            eprint(f"no item where {name_attr}='{name}' in {container}")
            continue

        fontc_els = [el for el in fontc_container.iter() if el.tag == coord_tag]
        fontmake_els = [el for el in fontmake_container.iter() if el.tag == coord_tag]

        if len(fontc_els) != len(fontmake_els):
            eprint(
                f"length of {container} '{name}' differs ({len(fontc_els)}/{len(fontmake_els)}), skipping"
            )
            continue

        for fontmake_el, fontc_el in zip(fontc_els, fontmake_els):
            for attr in ("x", "y"):
                delta_x = abs(
                    float(fontmake_el.attrib[attr]) - float(fontc_el.attrib[attr])
                )
                if 0.0 < delta_x <= 1.0:
                    fontc_el.attrib["diff_adjusted"] = "1"
                    fontmake_el.attrib["diff_adjusted"] = "1"
                    fontc_el.attrib[attr] = fontmake_el.attrib[attr]
                    spent += 1
                if spent >= off_by_one_budget:
                    eprint(
                        f"WARN: ran out of budget ({off_by_one_budget}) to fix off-by-ones in {container}"
                    )
                    return

    if spent > 0:
        eprint(
            f"INFO fixed {spent} off-by-ones in {container} (budget {off_by_one_budget})"
        )


# In various cases we have a list of indices where the order doesn't matter;
# often fontc sorts these and fontmake doesn't, so this lets us sort them there.
def sort_indices(ttx, table_tag: str, container_xpath: str, el_name: str):
    table = ttx.find(table_tag)
    if table is None:
        return
    containers = table.xpath(f"{container_xpath}")
    for container in containers:
        indices = [el for el in container.iter() if el.tag == el_name]
        values = sorted(int(v.attrib["value"]) for v in indices)
        for i, index in enumerate(indices):
            index.attrib["value"] = str(values[i])


# the same sets can be assigned different ids, but normalizer
# will resolve them to the actual glyphs and here we can just sort
def sort_gdef_mark_filter_sets(ttx: etree.ElementTree):
    markglyphs = ttx.xpath("//GDEF//MarkGlyphSetsDef")
    if markglyphs is None or len(markglyphs) == 0:
        return

    assert len(markglyphs) == 1
    markglyphs = markglyphs[0]

    coverages = sorted(
        markglyphs.findall("Coverage"), key=lambda cov: [g.attrib["value"] for g in cov]
    )
    for c in coverages:
        markglyphs.remove(c)

    id_map = {c.attrib["index"]: str(i) for (i, c) in enumerate(coverages)}
    for i, c in enumerate(coverages):
        c.attrib["index"] = str(i)
        markglyphs.append(c)

    # remap any MarkFilteringSet nodes that might not get normalized (in contextual
    # pos lookups e.g., or in GSUB)

    for mark_set in ttx.xpath("//MarkFilteringSet"):
        mark_set.attrib["value"] = id_map.get(mark_set.attrib["value"])

    # items keep their indentation when we reorder them, so reindent everything
    etree.indent(markglyphs, level=3)


LOOKUPS_TO_SKIP = set([2, 3, 4, 5, 6])  # pairpos, cursive, markbase, marklig, markmark


def remove_mark_and_kern_and_curs_lookups(ttx):
    gpos = ttx.find("GPOS")
    if gpos is None:
        return
    # './/Lookup' xpath selects all the 'Lookup' elements that are descendants of
    # the current 'GPOS' node - no matter where they are under it.
    # Most importantly, this _excludes_ GSUB lookups, which shouldn't be pruned.
    for lookup in gpos.xpath(".//Lookup"):
        lookup_type_el = lookup.find("LookupType")
        lookup_type = int(lookup_type_el.attrib["value"])
        is_extension = lookup_type == 9
        if is_extension:
            # For extension lookups, take the lookup type of the wrapped subtables;
            # all extension subtables share the same lookup type so checking only
            # the first is enough.
            ext_subtable = lookup.find("ExtensionPos")
            lookup_type = int(ext_subtable.find("ExtensionLookupType").attrib["value"])
        if lookup_type not in LOOKUPS_TO_SKIP:
            continue
        # remove all the elements but the type:
        to_remove = [child for child in lookup if child.tag != "LookupType"]
        for child in to_remove:
            lookup.remove(child)
        if is_extension:
            lookup_type_el.attrib["value"] = str(lookup_type)


# this all gets handled by otl-normalizer
def remove_gdef_lig_caret_and_var_store(ttx: etree.ElementTree):
    gdef = ttx.find("GDEF")
    if gdef is None:
        return

    for ident in ["LigCaretList", "VarStore"]:
        subtable = gdef.find(ident)
        if subtable is not None:
            gdef.remove(subtable)


# reassign class ids within a ClassDef, matching the fontc behaviour.
# returns a map of new -> old ids, which can be used to reorder elements that
# used the class ids as indices
def remap_class_def_ids_like_fontc(
    class_def: etree.ElementTree, glyph_map: Dict[str, int]
) -> Dict[int, int]:
    current_classes = defaultdict(list)
    for glyph in class_def.xpath(".//ClassDef"):
        cls = glyph.attrib["class"]
        current_classes[cls].append(glyph.attrib["glyph"])

    # match the sorting used in write-fonts by using the min GID as the tiebreaker
    # https://github.com/googlefonts/fontations/blob/3fcc52e/write-fonts/src/tables/layout/builders.rs#L183-L189
    new_order = sorted(
        current_classes.values(), key=lambda s: (-len(s), min(glyph_map[g] for g in s))
    )
    new_order_map = {name: i + 1 for (i, cls) in enumerate(new_order) for name in cls}
    result = dict()
    for glyph in class_def.xpath(".//ClassDef"):
        cls = glyph.attrib["class"]
        new = new_order_map.get(glyph.attrib["glyph"])
        glyph.attrib["class"] = str(new)
        result[new] = int(cls)
    return result


def reorder_rules(lookup: etree.ElementTree, new_order: Dict[int, int], rule_name: str):
    # the rules can exist as siblings of other non-rule elements, so we can't just
    # clear all children and set them in the same order.
    # instead we remove them and then append them back in order, using 'addnext'
    # to ensure that we're inserting them at the right location.
    orig_order = [el for el in lookup.iterchildren(tag=rule_name)]
    if len(orig_order) == 0:
        return
    prev_el = orig_order[0].getprevious()
    for el in orig_order:
        lookup.remove(el)

    for ix in range(len(orig_order)):
        prev_ix = new_order.get(ix, ix)
        el = orig_order[prev_ix]
        el.set("index", str(ix))
        prev_el.addnext(el)
        prev_el = el

    # there was a funny issue where if we moved the last element elsewhere
    # in the ordering it would end up having incorrect indentation, so just
    # reindent everything to be safe.
    etree.indent(lookup, level=4)


# for each named child in container, remap the 'value' attribute using the new ordering
def remap_values(
    container: etree.ElementTree, new_order: Dict[int, int], child_name: str
):
    # for the original use we need to map from new to old, but here we need the reverse
    rev_map = {v: k for (k, v) in new_order.items()}
    for el in container.iterchildren(child_name):
        old = int(el.attrib["value"])
        el.attrib["value"] = str(rev_map[old])


# fontmake and fontc assign glyph classes differently for class-based tables;
# fontc uses GIDs but fontmake uses glyph names, so we reorder them to be consistent.
def reorder_contextual_class_based_rules(
    ttx: etree.ElementTree, tag: str, glyph_map: Dict[str, int]
):
    if tag == "GSUB":
        chain_name = "ChainContextSubst"
        class_set_name = "ChainSubClassSet"
        class_rule_name = "ChainSubClassRule"

    elif tag == "GPOS":
        chain_name = "ChainContextPos"
        class_set_name = "ChainPosClassSet"
        class_rule_name = "ChainPosClassRule"
    else:
        raise ValueError("must be one of 'GPOS' or 'GSUB'")

    table = ttx.find(tag)
    if table is None:
        return
    for lookup in table.xpath(".//Lookup"):
        for chain_ctx in lookup.findall(chain_name):
            if chain_ctx is None or int(chain_ctx.attrib["Format"]) != 2:
                continue
            input_class_order = remap_class_def_ids_like_fontc(
                chain_ctx.find("InputClassDef"), glyph_map
            )
            reorder_rules(chain_ctx, input_class_order, class_set_name)
            backtrack_class_order = remap_class_def_ids_like_fontc(
                chain_ctx.find("BacktrackClassDef"), glyph_map
            )
            lookahead_class_order = remap_class_def_ids_like_fontc(
                chain_ctx.find("LookAheadClassDef"), glyph_map
            )
            for class_set in chain_ctx.findall(class_set_name):
                for class_rule in class_set.findall(class_rule_name):
                    remap_values(class_rule, input_class_order, "Input")
                    remap_values(class_rule, backtrack_class_order, "Backtrack")
                    remap_values(class_rule, lookahead_class_order, "LookAhead")


def fill_in_gvar_deltas(
    tree_1: etree.ElementTree,
    ttf_1: Path,
    tree_2: etree.ElementTree,
    ttf_2: Path,
):
    font_1 = TTFont(ttf_1)
    font_2 = TTFont(ttf_2)
    dense_count_1 = densify_gvar(font_1, tree_1)
    dense_count_2 = densify_gvar(font_2, tree_2)

    if dense_count_1 + dense_count_2 > 0:
        eprint(
            f"densified {dense_count_1} glyphVariations in tool 1, {dense_count_2} in tool 2"
        )


def densify_gvar(font: TTFont, ttx: etree.ElementTree):
    gvar = ttx.find("gvar")
    if gvar is None:
        return 0
    glyf = font["glyf"]
    hMetrics = font["hmtx"].metrics
    vMetrics = getattr(font.get("vmtx"), "metrics", None)

    total_deltas_filled = 0
    for variations in gvar.xpath(".//glyphVariations"):
        coords, g = glyf._getCoordinatesAndControls(
            variations.attrib["glyph"], hMetrics, vMetrics
        )
        total_deltas_filled += int(densify_one_glyph(coords, g.endPts, variations))

    return total_deltas_filled


def densify_one_glyph(coords, ends, variations: etree.ElementTree):
    did_work = False
    for tuple_ in variations.findall("tuple"):
        deltas = [None] * len(coords)
        for delta in tuple_.findall("delta"):
            idx = int(delta.attrib["pt"])
            if idx >= len(deltas):
                continue
            deltas[idx] = (int(delta.attrib["x"]), int(delta.attrib["y"]))

        if any(d is None for d in deltas):
            did_work = True
            filled_deltas = iup_delta(deltas, coords, ends)
            for delta in tuple_.findall("delta"):
                tuple_.remove(delta)

            new_deltas = [
                {"pt": str(i), "x": str(otRound(x)), "y": str(otRound(y))}
                for (i, (x, y)) in enumerate(filled_deltas)
            ]
            for attrs in new_deltas:
                new_delta = etree.Element("delta", attrs)
                tuple_.append(new_delta)

            etree.indent(tuple_, level=3)

    return did_work


# Reduce noise specific to `fontmake`.
# Run this before the general `reduce_diff_noise` function.
def reduce_diff_noise_fontmake(fontmake: etree.ElementTree):
    fontmake_glyph_map = {
        el.attrib["name"]: int(el.attrib["id"])
        for el in fontmake.xpath("//GlyphOrder/GlyphID")
    }

    sort_indices(fontmake, "GPOS", "//Feature", "LookupListIndex")
    sort_indices(fontmake, "GSUB", "//LangSys", "FeatureIndex")
    sort_indices(fontmake, "GSUB", "//DefaultLangSys", "FeatureIndex")
    reorder_contextual_class_based_rules(fontmake, "GSUB", fontmake_glyph_map)
    reorder_contextual_class_based_rules(fontmake, "GPOS", fontmake_glyph_map)

def reduce_diff_noise(tool_1: etree.ElementTree, tool_2: etree.ElementTree):
    for ttx in (tool_1, tool_2):
        # different name ids with the same value is fine
        name_id_to_name(ttx, "//NamedInstance", "subfamilyNameID")
        name_id_to_name(ttx, "//NamedInstance", "postscriptNameID")
        name_id_to_name(ttx, "//AxisNameID", "value")
        name_id_to_name(ttx, "//UINameID", "value")
        name_id_to_name(ttx, "//AxisNameID", None)
        name_id_to_name(ttx, "//ValueNameID", "value")
        name_id_to_name(ttx, "//ElidedFallbackNameID", "value")
        normalize_null_tags(ttx, "//OS_2/achVendID", "value")

        # deal with https://github.com/googlefonts/fontmake/issues/1003
        drop_weird_names(ttx)

        # for matching purposes checksum is just noise
        erase_checksum_and_dates(ttx)

        stat_like_fontmake(ttx)
        remove_mark_and_kern_and_curs_lookups(ttx)

        erase_type_from_stranded_points(ttx)
        remove_gdef_lig_caret_and_var_store(ttx)
        sort_gdef_mark_filter_sets(ttx)

        # sort names within the name table (do this at the end, so ids are correct
        # for earlier steps)
        normalize_name_ids(ttx)

    # Normalize glyf contour order but only when contours are identical
    tool_1_point_orders, tool_2_point_orders = normalize_glyf_contours(tool_1, tool_2)
    normalize_gvar_contours(tool_1, tool_1_point_orders)
    normalize_gvar_contours(tool_2, tool_2_point_orders)

    allow_fontc_only_variations_postscript_prefix(
        tool_1, tool_2
    )

    allow_some_off_by_ones(
        tool_1, tool_2, "glyf/TTGlyph", "name", "/contour/pt"
    )
    allow_some_off_by_ones(
        tool_1, tool_2, "gvar/glyphVariations", "glyph", "/tuple/delta"
    )


#===============================================================================
# Generating output
#===============================================================================


# given a font file, return a dictionary of tags -> size in bytes
def get_table_sizes(fontfile: Path) -> dict[str, int]:
    cmd = [TTX_NAME, "-l", str(fontfile)]
    stdout = log_and_run(cmd, check=True).stdout
    result = dict()

    for line in stdout.strip().splitlines()[3:]:
        split = line.split()
        result[split[0]] = int(split[2])

    return result


# return a dict of table tag -> size difference
# only when size difference exceeds some threshold
def check_sizes(font_file_1: Path, font_file_2: Path):
    THRESHOLD = 1 / 10
    sizes_1 = get_table_sizes(font_file_1)
    sizes_2 = get_table_sizes(font_file_2)

    output = dict()
    shared_keys = set(sizes_1.keys() & sizes_2.keys())

    for key in shared_keys:
        len_1 = sizes_1[key]
        len_2 = sizes_2[key]
        len_ratio = min(len_1, len_2) / max(len_1, len_2)
        if (1 - len_ratio) > THRESHOLD:
            if len_2 > len_1:
                rel_len = len_2 - len_1
                eprint(f"{key} {len_1} {len_2} {len_ratio:.3} {rel_len}")
                output[key] = rel_len
            elif len_1 > len_2:
                rel_len = len_1 - len_2
                eprint(f"{key} {len_2} {len_1} {len_ratio:.3} {rel_len}")
                output[key] = rel_len

    return output


# returns a dictionary of {"compiler_name":  {"tag": "xml_text"}}
def generate_output(
    build_dir: Path,
    otl_norm_bin: Path,
    tool_1_type: ToolType,
    tool_1_name: str,
    font_file_1: Path,
    tool_2_type: ToolType,
    tool_2_name: str,
    font_file_2: Path
):
    ttx_1 = run_ttx(font_file_1)
    ttx_2 = run_ttx(font_file_2)
    gpos_1 = run_normalizer(otl_norm_bin, font_file_1, "gpos")
    gpos_2 = run_normalizer(otl_norm_bin, font_file_2, "gpos")
    gdef_1 = run_normalizer(otl_norm_bin, font_file_1, "gdef")
    gdef_2 = run_normalizer(otl_norm_bin, font_file_2, "gdef")

    tree_1 = etree.parse(ttx_1)
    tree_2 = etree.parse(ttx_2)
    fill_in_gvar_deltas(tree_1, font_file_1, tree_2, font_file_2)

    fontmake_tree = None
    if tool_1_type == ToolType.FONTMAKE or tool_1_type == ToolType.FONTMAKE_GFTOOLS:
        fontmake_tree = tree_1
    elif tool_2_type == ToolType.FONTMAKE or tool_2_type == ToolType.FONTMAKE_GFTOOLS:
        fontmake_tree = tree_2
    if fontmake_tree:
        reduce_diff_noise_fontmake(fontmake=fontmake_tree)

    reduce_diff_noise(tree_1, tree_2)

    map_1 = extract_comparables(tree_1, build_dir, tool_1_name)
    map_2 = extract_comparables(tree_2, build_dir, tool_2_name)
    
    if tool_1_type == ToolType.GLYPHSAPP and tool_2_type == ToolType.GLYPHSAPP:
        # Remove BASE table when comparing two Glyphs app versions. We know that
        # it is written only by Glyphs 4+, so we can ignore it in the comparison.
        if ("BASE" in map_1) ^ ("BASE" in map_2):
            map_1.pop("BASE", None)
            map_2.pop("BASE", None)

    size_diffs = check_sizes(font_file_1, font_file_2)
    map_1[MARK_KERN_NAME] = gpos_1
    map_2[MARK_KERN_NAME] = gpos_2
    if len(gdef_1):
        map_1[LIG_CARET_NAME] = gdef_1
    if len(gdef_2):
        map_2[LIG_CARET_NAME] = gdef_2
    
    # Use generic tool names as JSON keys.
    result = {TOOL_1_NAME: map_1, TOOL_2_NAME: map_2}
    if len(size_diffs) > 0:
        result["sizes"] = size_diffs

    return result


def print_output(
    build_dir: Path,
    output: dict[str, dict[str, Any]],
    tool_1_name: str,
    tool_2_name: str
):
    map_1 = output[TOOL_1_NAME]
    map_2 = output[TOOL_2_NAME]
    print("COMPARISON")
    t1 = set(map_1.keys())
    t2 = set(map_2.keys())
    if t1 != t2:
        if t1 - t2:
            tags = ", ".join(f"'{t}'" for t in sorted(t1 - t2))
            print(f"  Only {tool_1_name} produced {tags}")

        if t2 - t1:
            tags = ", ".join(f"'{t}'" for t in sorted(t2 - t1))
            print(f"  Only {tool_2_name} produced {tags}")

    for tag in sorted(t1 & t2):
        t1s = map_1[tag]
        t2s = map_2[tag]
        if t1s == t2s:
            print(f"  Identical '{tag}'")
        else:
            difference = diff_ratio(t1s, t2s)
            p1 = build_dir / path_for_output_item(tag, tool_1_name)
            p2 = build_dir / path_for_output_item(tag, tool_2_name)
            print(f"  DIFF '{tag}', {rel_user(p1)} {rel_user(p2)} ({difference:.3%})")
    if output.get("sizes"):
        print("SIZE DIFFERENCES")
    for tag, diff in output.get("sizes", {}).items():
        print(f"SIZE DIFFERENCE: '{tag}': {diff}B")


def jsonify_output(
    output: dict[str, dict[str, Any]],
):
    map_1 = output[TOOL_1_NAME]
    map_2 = output[TOOL_2_NAME]
    sizes = output.get("sizes", {})
    all_tags = set(map_1.keys()) | set(map_2.keys())
    out = dict()
    same_lines = 0
    different_lines = 0
    for tag in all_tags:
        if tag not in map_1:
            different_lines += len(map_2[tag])
            out[tag] = TOOL_2_NAME
        elif tag not in map_2:
            different_lines += len(map_1[tag])
            out[tag] = TOOL_1_NAME
        else:
            s1 = map_1[tag]
            s2 = map_2[tag]
            if s1 != s2:
                ratio = diff_ratio(s1, s2)
                n_lines = max(len(s1), len(s2))
                same_lines += int(n_lines * ratio)
                different_lines += int(n_lines * (1 - ratio))
                out[tag] = ratio
            else:
                same_lines += len(s1)

    # then also add in size differences, if any
    for tag, size_diff in sizes.items():
        out[f"sizeof({tag})"] = size_diff
        # hacky: we don't want to be perfect if we have a size diff,
        # so let's pretend that whatever our size diff is, it corresponds
        # to some fictional table 100 lines liong
        different_lines += 100

    overall_diff_ratio = same_lines / (same_lines + different_lines)
    out["total"] = overall_diff_ratio
    return {"success": out}


def print_json(output):
    as_json = json.dumps(output, indent=2, sort_keys=True)
    print(as_json)


# given the ttx for a font, return a map of tags -> xml text for each root table.
# also writes the xml to individual files
def extract_comparables(font_xml, build_dir: Path, compiler: str) -> dict[str, str]:
    comparables = dict()
    tables = {e.tag: e for e in font_xml.getroot()}
    for tag in sorted(e.tag for e in font_xml.getroot()):
        table_str = to_xml_string(tables[tag])
        path = build_dir / f"{compiler}.{tag}.ttx"
        path.write_bytes(table_str)
        comparables[tag] = table_str

    return comparables


# the line-wise ratio of difference, i.e. the fraction of lines that are the same
def diff_ratio(text1: str, text2: str) -> float:
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    m = SequenceMatcher(None, lines1, lines2)
    return m.quick_ratio()


def path_for_output_item(tag_or_normalizer_name: str, compiler: str) -> str:
    if tag_or_normalizer_name == MARK_KERN_NAME:
        return f"{compiler}.markkern.txt"
    else:
        return f"{compiler}.{tag_or_normalizer_name}.ttx"


# log or print as json any compilation failures (and exit if there were any)
def report_errors_and_exit_if_there_were_any(errors: dict):
    if len(errors) == 0:
        return
    for error in errors.values():
        cmd = error["command"]
        stderr = error["stderr"]
        eprint(f"command '{cmd}' failed: '{stderr}'")

    if FLAGS.json:
        print_json({"error": errors})
    sys.exit(2)


# for reproducing crater results we have a syntax that lets you specify a repo
# url as the source. in this scheme we pass the path to the particular source
# (relative the repo root) as a url fragment
def resolve_source(source: str, cache_dir: Path) -> Path:
    if source.startswith("git@") or source.startswith("https://"):
        source_url = urlparse(source)
        repo_path = source_url.fragment
        org_name = source_url.path.split("/")[-2]
        repo_name = source_url.path.split("/")[-1]
        sha = source_url.query
        local_repo = (
            cache_dir / org_name / repo_name
        ).resolve()
        if not local_repo.parent.is_dir():
            local_repo.parent.mkdir(parents=True)
        if not local_repo.is_dir():
            cmd = ("git", "clone", source_url._replace(fragment="", query="").geturl())
            print("Running", " ".join(cmd), "in", local_repo.parent)
            subprocess.run(cmd, cwd=local_repo.parent, check=True)
        else:
            print(f"Reusing existing {rel_user(local_repo)}")

        if len(sha) > 0:
            log_and_run(("git", "checkout", sha), cwd=local_repo, check=True)
        source = local_repo / repo_path
    else:
        source = Path(source)
    if not source.exists():
        sys.exit(f"No such source: {source}")
    return source


def delete_things_we_must_rebuild(
    rebuild: str, 
    tool_1_name: str, 
    font_file_1: Path, 
    tool_2_name: str, 
    font_file_2: Path
):
    # Replace generic rebuild flag with specific tool name.
    if rebuild == TOOL_1_NAME:
        rebuild = tool_1_name
    elif rebuild == TOOL_2_NAME:
        rebuild = tool_2_name

    for tool_name, font_file in [(tool_1_name, font_file_1), (tool_2_name, font_file_2)]:
        must_rebuild = rebuild in [tool_name, "both"]
        if must_rebuild:
            for path in [
                font_file,
                font_file.with_suffix(".ttx"),
                font_file.with_suffix(".markkern.txt"),
                font_file.with_suffix(".ligcaret.txt"),
            ]:
                if path.exists():
                    os.remove(path)


# returns the path to the compiled binary
def build_crate(manifest_path: Path):
    cmd = ["cargo", "build", "--release", "--manifest-path", str(manifest_path)]
    log_and_run(cmd, cwd=None, check=True)


def get_fontc_and_normalizer_binary_paths(root_dir: Path) -> Tuple[Path, Path]:
    fontc_path = FLAGS.fontc_path
    norm_path = FLAGS.normalizer_path
    if fontc_path is None:
        fontc_manifest_path = root_dir / "fontc" / "Cargo.toml"
        fontc_path = root_dir / "target" / "release" / "fontc"
        build_crate(fontc_manifest_path)
        assert fontc_path.is_file(), "failed to build fontc?"
    else:
        fontc_path = Path(fontc_path)
        assert fontc_path.is_file(), f"fontc path '{fontc_path}' does not exist"
    if norm_path is None:
        otl_norm_manifest_path = root_dir / "otl-normalizer" / "Cargo.toml"
        norm_path = root_dir / "target" / "release" / "otl-normalizer"
        build_crate(otl_norm_manifest_path)
        assert norm_path.is_file(), "failed to build otl-normalizer?"
    else:
        norm_path = Path(norm_path)
        assert norm_path.is_file(), f"normalizer path '{norm_path}' does not exist"

    manifest_path = root_dir / crate_name / "Cargo.toml"
    bin_path = root_dir / "target" / "release" / crate_name
    build_crate(manifest_path)
    return bin_path
    return (fontc_path, norm_path)


def get_crate_path(
    bin_path: Optional[str], root_dir: Optional[Path], crate_name: str
) -> Path:
    """Get path to a crate binary, building it if in fontc repo, or finding in PATH.

    Args:
        bin_path: Path provided via CLI flag
        root_dir: Path to fontc repository root (if we're in one)
        crate_name: Name of the crate (e.g., "fontc" or "otl-normalizer")

    Returns:
        Path to the binary

    Raises:
        SystemExit: If binary cannot be found or built
    """
    if bin_path:
        path = Path(bin_path)
        if not path.is_file():
            sys.exit(f"Specified {crate_name} path '{path}' does not exist")
        return path

    # If we're in the fontc repo, try to build it
    if root_dir is not None:
        manifest_path = root_dir / crate_name / "Cargo.toml"
        if manifest_path.is_file():
            built_path = root_dir / "target" / "release" / crate_name
            build_crate(manifest_path)
            if built_path.is_file():
                return built_path

    # Try to find in PATH
    which_result = shutil.which(crate_name)
    if which_result:
        return Path(which_result)

    # Give helpful error message
    sys.exit(
        f"Could not find '{crate_name}' binary. Please either:\n"
        f"  1. Specify the path with --{crate_name}_path flag\n"
        f"  2. Install {crate_name} and ensure it's in your PATH\n"
        f"  3. Run from the fontc repository root to build it automatically"
    )


#===============================================================================
# Tool definitions
#===============================================================================


@dataclass
class Tool(ABC):
    source: Path

    # Subclasses should return their specific tool type.
    @property
    @abstractmethod
    def type(self) -> ToolType:
        pass

    # Subclasses may override the tool name, e. g. by appending a version.
    @property
    def name(self) -> str:
        return self.type

    # Subclasses may override the font file extension.
    @property
    def font_file_extension(self) -> str:
        return "ttf"

    # Subclasses should return their specific build action as a closure.
    @abstractmethod
    def build_action(self, build_dir: Path, font_file_name: str = None) -> Callable[[], None]:
        pass

    # Central method for building and exception handling (internal).
    def _build(self, build_action: Callable[[], None]) -> Optional[dict]:
        try:
            build_action()
        except BuildFail as e:
            return {
                "command": " ".join(e.command),
                "stderr": e.msg[-MAX_ERR_LEN:],
            }
        
        return None
    
    # Call this method to generate the specific build action and start a build.
    def run(self, build_dir: Path, font_file_name: str = None) -> Optional[dict]:
        action = self.build_action(build_dir, font_file_name)
        return self._build(action)
        

@dataclass
class FontcTool(Tool):
    root: Path
    fontc_path: str = None

    _fontc_bin: Path = field(init=False, repr=False)

    def __post_init__(self):
        self._fontc_bin = get_crate_path(self.fontc_path, self.root, FONTC_NAME)
        assert self._fontc_bin.is_file(), f"fontc path '{self._fontc_bin}' does not exist"

    # Created once, read-only.
    @property
    def fontc_bin(self):
        return self._fontc_bin


@dataclass(unsafe_hash=True)
class StandaloneFontcTool(FontcTool):

    @property
    def type(self) -> ToolType:
        return ToolType.FONTC

    def build_action(self, build_dir: Path, font_file_name: str = None) -> Callable[[], None]:
        def action():
            build_fontc(self.source, self.fontc_bin, build_dir, font_file_name)
        return action


@dataclass(unsafe_hash=True)
class GfToolsFontcTool(FontcTool):

    @property
    def type(self) -> ToolType:
        return ToolType.FONTC_GFTOOLS

    def build_action(self, build_dir: Path, font_file_name: str = None) -> Callable[[], None]:
        def action():
            run_gftools(self.source, FLAGS.config, build_dir, font_file_name, self.fontc_bin)
        return action


@dataclass(unsafe_hash=True)
class StandaloneFontmakeTool(Tool):

    @property
    def type(self) -> ToolType:
        return ToolType.FONTMAKE

    def build_action(self, build_dir: Path, font_file_name: str = None) -> Callable[[], None]:
        def action():
            build_fontmake(self.source, build_dir, font_file_name)
        return action


@dataclass(unsafe_hash=True)
class GfToolsFontmakeTool(Tool):

    @property
    def type(self) -> ToolType:
        return ToolType.FONTMAKE_GFTOOLS

    def build_action(self, build_dir: Path, font_file_name: str = None) -> Callable[[], None]:
        def action():
            run_gftools(self.source, FLAGS.config, build_dir, font_file_name)
        return action


@dataclass(unsafe_hash=True)
class GlyphsAppTool(Tool):
    # On initialization, set either the bundle path or the major version.
    bundle_path: str = None
    major_version: int = None

    # Created once, internal.
    _glyphsapp_proxy: NSDistantObject = field(init=False, repr=False)
    
    _version: str = None
    _build_number: str = None

    def __post_init__(self):
        if self.bundle_path is not None and self.major_version is None:
            self._glyphsapp_proxy, self._version, self._build_number = application_at_path(self.bundle_path)
            if self._glyphsapp_proxy is None:
                sys.exit(f"No JSTalk connection to Glyphs app at {self.bundle_path}")

        elif self.bundle_path is None and self.major_version is not None:
            self._glyphsapp_proxy, self._version, self._build_number = application_with_version(self.major_version)
            if self._glyphsapp_proxy is None:
                sys.exit(f"No JSTalk connection to Glyphs {self.major_version} app")
        
        else:
            sys.exit("Must specify either bundle path or major version for Glyphs app")

    @property
    def type(self) -> ToolType:
        return ToolType.GLYPHSAPP

    @property
    def name(self) -> str:
        underscored_version = self._version.replace(".", "_")
        return f"{self.type}_{underscored_version}_{self._build_number}"

    @property
    def font_file_extension(self) -> str:
        return "otf"

    def build_action(self, build_dir: Path, font_file_name: str = None) -> Callable[[], None]:
        should_terminate = self.bundle_path is not None and self.major_version is None
        def action():
            exportFirstInstance(self._glyphsapp_proxy, self.source, build_dir, font_file_name, should_terminate)
            self._glyphsapp_proxy = None
        return action


#===============================================================================
# Main method
#===============================================================================


def main(argv):
    if len(argv) != 2:
        sys.exit("Only one argument, a source file, is expected. Pass the `--help` flag to display usage information and available options.")

    cache_dir = Path(FLAGS.cache_path).expanduser().resolve()
    source = resolve_source(argv[1], cache_dir).resolve()

    # Check if we're in the fontc repository (optional - allows building binaries)
    cwd = Path(".").resolve()
    fontc_repo_root = None
    if (cwd / "fontc" / "Cargo.toml").is_file():
        fontc_repo_root = cwd
        eprint(f"Detected fontc repository at {rel_user(fontc_repo_root)}")

    # Get binary paths - will look in PATH or build if in repo
    fontc_bin_path = get_crate_path(FLAGS.fontc_path, fontc_repo_root, "fontc")
    otl_bin_path = get_crate_path(
        FLAGS.normalizer_path, fontc_repo_root, "otl-normalizer"
    )

    if shutil.which(FONTMAKE_NAME) is None:
        sys.exit("No fontmake")
    if shutil.which(TTX_NAME) is None:
        sys.exit("No ttx")

    if FLAGS.outdir is not None:
        out_dir = Path(FLAGS.outdir).expanduser().resolve()
        if not out_dir.exists():
            sys.exit(f"Specified output directory {out_dir} does not exist")
    elif fontc_repo_root is not None:
        # If in fontc repo, use repo's build directory
        out_dir = fontc_repo_root / "build"
    else:
        # Otherwise use current directory
        out_dir = cwd / "ttx_diff_output"
        eprint(f"No --outdir specified, using {rel_user(out_dir)}")

    # Configure tools.
    tool_1_type = FLAGS.tool_1_type
    tool_2_type = FLAGS.tool_2_type
    if tool_1_type is None or tool_2_type is None:
        sys.exit("Must specify two tools")

    # If any of the tools are managed by GFTools, we need to have a config file.
    gftools_tool_types = (ToolType.FONTC_GFTOOLS, ToolType.FONTMAKE_GFTOOLS)
    if tool_1_type in gftools_tool_types or tool_2_type in gftools_tool_types:
        if FLAGS.config is None:
            sys.exit("Must specify a config file for GFTools")

    # If Glyphs app bundle paths are not set, use default registered names.
    # This is intended for launching running apps, e. g. when debugging.
    # Please note that in this case, the Glyphs app instances are not terminated.
    # This conflicts with `fontc_crater` builds, which expect to be able
    # to launch multiple instance of the same Glyphs app build and thus
    # require the instances to be terminated after building.
    tool_1_glyphs_version = None
    tool_2_glyphs_version = None
    if tool_1_type == ToolType.GLYPHSAPP and FLAGS.tool_1_path is None:
        tool_1_glyphs_version = 3
    if tool_2_type == ToolType.GLYPHSAPP and FLAGS.tool_2_path is None:
        tool_2_glyphs_version = 4

    # Currently, having two tools of the same type is supported for Glyphs app
    # instances only. These must have at least different build numbers.
    if tool_1_type == tool_2_type:
        if tool_1_type != ToolType.GLYPHSAPP:
            sys.exit("Must specify two different tools")

        # Both tools are Glyphs apps.
        # Test only if both paths are set.
        if FLAGS.tool_1_path is not None and FLAGS.tool_2_path is not None:
            version_1, build_number_1 = app_bundle_version(FLAGS.tool_1_path)
            version_2, build_number_2 = app_bundle_version(FLAGS.tool_2_path)
            if version_1 == version_2 and build_number_1 == build_number_2:
                sys.exit("Must specify two different Glyphs app builds")

    tool_1 = None;
    if tool_1_type == ToolType.FONTC:
        tool_1 = StandaloneFontcTool(source, root, FLAGS.tool_1_path)
    elif tool_1_type == ToolType.FONTMAKE:
        tool_1 = StandaloneFontmakeTool(source)
    elif tool_1_type == ToolType.FONTC_GFTOOLS:
        tool_1 = GfToolsFontcTool(source, root, FLAGS.tool_1_path)
    elif tool_1_type == ToolType.FONTMAKE_GFTOOLS:
        tool_1 = GfToolsFontmakeTool(source)
    elif tool_1_type == ToolType.GLYPHSAPP:
        tool_1 = GlyphsAppTool(source, FLAGS.tool_1_path, tool_1_glyphs_version)

    tool_2 = None;
    if tool_2_type == ToolType.FONTC:
        tool_2 = StandaloneFontcTool(source, root, FLAGS.tool_2_path)
    elif tool_2_type == ToolType.FONTMAKE:
        tool_2 = StandaloneFontmakeTool(source)
    elif tool_2_type == ToolType.FONTC_GFTOOLS:
        tool_2 = GfToolsFontcTool(source, root, FLAGS.tool_2_path)
    elif tool_2_type == ToolType.FONTMAKE_GFTOOLS:
        tool_2 = GfToolsFontmakeTool(source)
    elif tool_2_type == ToolType.GLYPHSAPP:
        tool_2 = GlyphsAppTool(source, FLAGS.tool_2_path, tool_2_glyphs_version)

    if tool_1 is None or tool_2 is None:
        sys.exit("Failed to configure one or both tools")

    # Concrete tool names are used for file names and in log messages.
    tool_1_name = tool_1.name
    tool_2_name = tool_2.name

    build_dir = out_dir / f"{tool_1_name}_{tool_2_name}"
    build_dir.mkdir(parents=True, exist_ok=True)
    eprint(f"Comparing {tool_1_name} and {tool_2_name} in {rel_user(build_dir)}")

    font_file_name_1 = f"{tool_1_name}.{tool_1.font_file_extension}"
    font_file_name_2 = f"{tool_2_name}.{tool_2.font_file_extension}"
    font_file_1 = build_dir / font_file_name_1
    font_file_2 = build_dir / font_file_name_2

    # We delete all resources that we have to rebuild. The rest of the script
    # will assume it can reuse anything that still exists.
    delete_things_we_must_rebuild(
        FLAGS.rebuild,
        tool_1_name,
        font_file_1,
        tool_2_name,
        font_file_2
    )

    failures = {}

    failure_1 = tool_1.run(build_dir, font_file_name_1)
    failure_2 = tool_2.run(build_dir, font_file_name_2)
    
    if failure_1 is not None:
        failures[TOOL_1_NAME] = failure_1
    if failure_2 is not None:
        failures[TOOL_2_NAME] = failure_2

    report_errors_and_exit_if_there_were_any(failures)

    # if compilation completed, these exist
    assert font_file_1.is_file(), font_file_1
    assert font_file_2.is_file(), font_file_2

    output = generate_output(
        build_dir,
        otl_bin,
        tool_1_type,
        tool_1_name,
        font_file_1,
        tool_2_type,
        tool_2_name,
        font_file_2
    )

    diffs = False
    if output[TOOL_1_NAME] == output[TOOL_2_NAME]:
        eprint("output is identical")
    else:
        diffs = True
        if not FLAGS.json:
            print_output(build_dir, output, tool_1_name, tool_2_name)
        else:
            output = jsonify_output(output)
            print_json(output)

    sys.exit(diffs * 2)  # 0 or 2
