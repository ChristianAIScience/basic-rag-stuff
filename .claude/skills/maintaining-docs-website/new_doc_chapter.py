"""Scaffold a new documentation chapter (English + German) and print the start-page snippet.

Dependencies:  Python standard library only.

Usage (from the project root):
    python .claude/skills/maintaining-docs-website/new_doc_chapter.py 13 caching "Caching" "Caching"
           number  file-key  English title  German title

Creates (never overwrites):
    docs/13_caching.md
    website/i18n/de/docusaurus-plugin-content-docs/current/13_caching.md
Then follow the "Add a chapter" checklist in SKILL.md (card, write-translations, build, README).
"""

import sys
from pathlib import Path

SECTIONS_EN = ["What", "Why", "The code", "How it connects", "Run", "Expected output"]
SECTIONS_DE = ["Was", "Warum", "Der Code", "Wie es zusammenhängt", "Ausführen", "Erwartete Ausgabe"]
GERMAN_DOCS = Path("website/i18n/de/docusaurus-plugin-content-docs/current")


def chapter_markdown(number: int, title: str, sections: list[str]) -> str:
    body = "\n\n".join(f"## {section}\n\nTODO" for section in sections)
    return f"# {number}. {title}\n\n{body}\n"


def card_snippet(key: str, number: int, title: str) -> str:
    """Entry for getChapters() in website/src/pages/index.js - ids must be literal for write-translations."""
    return (
        "    {\n"
        f"      path: '/docs/{key}',\n"
        f"      title: translate({{id: 'home.chapter.{key}.title', message: '{number}. {title}'}}),\n"
        f"      text: translate({{id: 'home.chapter.{key}.text', message: 'TODO one-line summary'}}),\n"
        "    },"
    )


def scaffold(root: Path, number: int, key: str, title_en: str, title_de: str) -> list[Path]:
    file_name = f"{number:02d}_{key}.md"
    targets = [
        (root / "docs" / file_name, chapter_markdown(number, title_en, SECTIONS_EN)),
        (root / GERMAN_DOCS / file_name, chapter_markdown(number, title_de, SECTIONS_DE)),
    ]
    for path, _ in targets:
        if path.exists():
            raise FileExistsError(f"{path} already exists - not overwritten")
    for path, content in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return [path for path, _ in targets]


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    number, key, title_en, title_de = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    for created in scaffold(Path("."), number, key, title_en, title_de):
        print(f"created {created.as_posix()}")
    print("\nAdd to getChapters() in website/src/pages/index.js:\n")
    print(card_snippet(key, number, title_en))
    print("\nThen: cd website && npm run write-translations -- --locale de && npm run build")
