---
name: maintaining-docs-website
description: Maintains the Docusaurus documentation website in website/ that renders docs/*.md in English and German. Use when adding or editing a documentation chapter or its German translation, changing the start page, or building and previewing the site.
---

# Maintaining the docs website (Docusaurus)

Reference implementation: `website/docusaurus.config.js`, `website/sidebars.js`, `website/src/pages/index.js`, `website/i18n/de/`.
The site reads `../docs/*.md` directly: the Markdown files in `docs/` are the single source of truth for English.
Versions: see `website/package.json`.

## Add a chapter

Copy this checklist:

```
- [ ] 1. Write docs/NN_name.md (prefix = sidebar order; URL drops it: /docs/name)
- [ ] 2. German copy, same file name, in website/i18n/de/docusaurus-plugin-content-docs/current/
- [ ] 3. Card in getChapters() in website/src/pages/index.js with literal ids home.chapter.KEY.title / .text
- [ ] 4. cd website && npm run write-translations -- --locale de, then fill the new messages in i18n/de/code.json
- [ ] 5. npm run build -> fix every reported broken link and repeat until SUCCESS for en and de
- [ ] 6. Link the chapter in README.md ("Learning path")
```

- Step 1: start with `# N. Title`; sections: what, why, the code, how it connects, run, expected output.
- Step 2: translate prose; keep code blocks and program output unchanged (the programs print English).
- Step 5 is the feedback loop: `onBrokenLinks` and `markdown.hooks.onBrokenMarkdownLinks` are `throw`, so a green build means every link resolves. Then check `/docs/name` and `/de/docs/name` with `npm run serve`.

Editing an existing chapter: change `docs/` **and** the German copy in the same step; a missing German file silently falls back to English.

## Reference

- `markdown.format: 'detect'`: `.md` files are parsed as CommonMark, not MDX - braces and `<` need no escaping.
- Mermaid blocks render client-side via `@docusaurus/theme-mermaid`; verify them in the JS bundle (`n.mermaid`), not in the HTML.
- `write-translations` extracts only `translate()` calls with **literal** ids; template-string ids are skipped silently.
- `npm start` serves one locale (`npm start -- --locale de`); the language switch works only in `npm run build` + `npm run serve`.
- `website/node_modules/`, `website/build/`, `website/.docusaurus/` are git-ignored.
