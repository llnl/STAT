"""Sphinx settings for the STAT User Guide."""

from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
LATEX_DIR = DOC_DIR / "_latex"

copyright = "2007-2020, Lawrence Livermore National Security, LLC"

# Markdown parsing.
extensions = ["myst_parser"]
myst_enable_extensions = ["deflist"]
smartquotes = False
numfig = True

# HTML presentation.
html_additional_pages = {"index": "guide-index.html"}
html_theme = "alabaster"
html_theme_options = {
    "page_width": "75rem",
    "sidebar_width": "var(--sidebar-size)",
    "font_family": '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    "font_size": "1rem",
    "body_text": "#263548",
    "link": "#195b85",
    "footer_text": "#64748b",
    "sidebar_link": "#263548",
    "base_text": "#263548",
    "code_text": "#244864",
    "code_font_size": "0.88em",
    "table_border": "#e2e8f0",
    "code_bg": "#f5f8fb",
    "pre_bg": "#eef1f4",
    "narrow_sidebar_bg": "white",
    "narrow_sidebar_link": "#263548",
}
html_static_path = [str(DOC_DIR / "_static")]
html_css_files = ["guide.css"]
templates_path = [str(DOC_DIR / "_templates")]

# PDF output and cover layout.
latex_engine = "xelatex"
latex_logo = str(DOC_DIR / "src/figs/stat_logo.png")
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "extraclassoptions": "oneside,openany",
    "fncychap": "",
    "sphinxsetup": (
        "hmargin=25mm,vmargin=20mm,"
        "TitleColor={rgb}{0.09,0.15,0.23},"
        "InnerLinkColor={rgb}{0.10,0.32,0.48},"
        "VerbatimColor={RGB}{238,241,244},"
        "VerbatimBorderColor={rgb}{0.68,0.75,0.82}"
    ),
    "maketitle": (LATEX_DIR / "cover.tex").read_text(encoding="utf-8"),
    "preamble": (LATEX_DIR / "preamble.tex").read_text(encoding="utf-8"),
}
