"""Keep the repository's local documentation navigation usable."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
DATED_REPORT = re.compile(r"\d{4}-\d{2}-\d{2}")


def test_local_documentation_links_resolve() -> None:
    pages = [*ROOT.glob("*.md"), *DOCS.rglob("*.md"), ROOT / "examples/README.md"]
    missing: list[str] = []
    for page in pages:
        for destination in MARKDOWN_LINK.findall(page.read_text(encoding="utf-8")):
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = page.parent / unquote(parsed.path)
            if not target.exists():
                missing.append(f"{page.relative_to(ROOT)} -> {destination}")
    assert not missing, "Broken local Markdown links:\n" + "\n".join(missing)


def test_public_docs_root_has_no_dated_experiment_logs() -> None:
    dated = sorted(path.name for path in DOCS.glob("*.md") if DATED_REPORT.search(path.name))
    assert not dated, f"Move experiment logs out of maintained documentation: {dated}"


def test_editable_diagrams_have_both_previews() -> None:
    for source in (DOCS / "diagrams").glob("*.excalidraw"):
        assert source.with_suffix(".png").is_file()
        assert source.with_suffix(".svg").is_file()
