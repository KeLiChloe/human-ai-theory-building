"""Render self-contained interactive semantic network HTML demos."""

from __future__ import annotations

import json
import re
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("template.html")
INDEX_TEMPLATE_PATH = Path(__file__).with_name("index_template.html")
SIDEBAR_CSS_PATH = Path(__file__).with_name("sidebar.css")
SIDEBAR_JS_PATH = Path(__file__).with_name("sidebar.js")


def _sidebar_assets() -> tuple[str, str]:
    return (
        SIDEBAR_CSS_PATH.read_text(encoding="utf-8"),
        SIDEBAR_JS_PATH.read_text(encoding="utf-8"),
    )


def render_network_interactive_html(
    payload: dict,
    outpath: Path,
    *,
    nav_entries: list[dict] | None = None,
) -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    sidebar_css, sidebar_js = _sidebar_assets()
    data_json = json.dumps(payload, ensure_ascii=False)
    nav_json = json.dumps(nav_entries or [], ensure_ascii=False)
    html = (
        template.replace("__NETWORK_DATA_JSON__", data_json)
        .replace("__NAV_ENTRIES_JSON__", nav_json)
        .replace("__SIDEBAR_CSS__", sidebar_css)
        .replace("__SIDEBAR_JS__", sidebar_js)
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(html, encoding="utf-8")


def render_network_index_html(entries: list[dict], outpath: Path) -> None:
    template = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(entries, ensure_ascii=False)
    html = template
    html = re.sub(
        r'(<script id="index-data" type="application/json">)(.*?)(</script>)',
        lambda m: m.group(1) + data_json + m.group(3),
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = (
        html.replace('href="sidebar.css"', 'href="network_interactive/sidebar.css"')
        .replace('src="sidebar.js"', 'src="network_interactive/sidebar.js"')
        .replace(
            'src="sidebar.js?v=20260718b"',
            'src="network_interactive/sidebar.js?v=20260718b"',
        )
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(html, encoding="utf-8")
