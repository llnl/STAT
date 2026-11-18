"""Build the full user guide from the Markdown sources."""

import sys
from pathlib import Path

# Load the shared settings into the namespace Sphinx reads.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sphinx_config import *  # noqa: F403

root_doc = "stat_userguide"
# Sphinx discovers only chapter pages. They include the remaining sources.
include_patterns = [
    "stat_userguide.md",
    "stat_auspice.md",
    "stat_introduction.md",
    "stat_overview.md",
    "stat_changelog.md",
    "stat_dependencies.md",
    "stat_description.md",
    "statview_description.md",
    "statgui_description.md",
    "stat_preference_files.md",
    "dysectapi_overview.md",
    "stat_at_scale.md",
    "statbench_description.md",
    "stat_script_description.md",
    "stat_troubleshoot.md",
    "stat_bibliography.md",
]

project = "STAT User Guide"
html_title = project
author = (
    "Gregory L. Lee, Dorian C. Arnold, Dong H. Ahn, Bronis R. de Supinski, "
    "Nicklas B. Jensen, Sven Karlsson, Matthew P. LeGendre, Barton P. Miller, "
    "Niklas Q. Nielsen, Martin Schulz"
)
today = "2020-09-15"

latex_documents = [
    (
        root_doc,
        "stat_userguide.tex",
        "STAT: Stack Trace Analysis Tool",
        author.replace(", ", r"\\"),
        "manual",
    ),
]
