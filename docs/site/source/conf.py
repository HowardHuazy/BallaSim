"""Build the documentation without importing or building BallaSim."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
project = "BallaSim"
release = re.search(
    r'^__version__ = "([^"]+)"',
    (ROOT / "python/ballasim/__init__.py").read_text(),
    re.MULTILINE,
).group(1)
version = release
language = "en"

extensions = ["myst_parser", "sphinx.ext.githubpages", "sphinx.ext.mathjax"]
source_suffix = {".md": "markdown"}
myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath"]
root_doc = "index"
exclude_patterns = []
templates_path = ["_templates"]
nitpicky = True

html_theme = "furo"
html_title = project
html_short_title = project
html_static_path = ["_static"]
html_css_files = ["css/ballasim.css"]
html_favicon = "_static/images/favicon.png"
html_copy_source = False
html_show_sourcelink = False
html_use_index = False
html_domain_indices = False
html_show_copyright = False
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}
html_theme_options = {
    "light_logo": "images/dog-light.png",
    "dark_logo": "images/dog-dark.png",
    "top_of_page_buttons": [],
    "light_css_variables": {
        "font-stack": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        "color-brand-primary": "#146ac4",
        "color-brand-content": "#146ac4",
        "color-foreground-primary": "#252b36",
        "color-foreground-secondary": "#697385",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f7f8fa",
        "color-background-border": "#e4e7ec",
        "color-sidebar-background": "#f7f8fa",
        "color-sidebar-item-background--current": "#e7f0fc",
        "color-sidebar-item-background--hover": "#edf0f4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#71b5ff",
        "color-brand-content": "#71b5ff",
        "color-foreground-primary": "#e4e7ee",
        "color-foreground-secondary": "#a0a9b9",
        "color-background-primary": "#13151a",
        "color-background-secondary": "#191b21",
        "color-background-border": "#2b2e37",
        "color-sidebar-background": "#191b21",
        "color-sidebar-item-background--current": "#223146",
        "color-sidebar-item-background--hover": "#252932",
    },
}
