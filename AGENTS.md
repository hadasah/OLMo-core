# Repo Instructions

When interacting with marimo notebooks in this repo in any way, always load and follow the `marimo-notebook` skill first.
When working inside a running marimo session, also load and follow the `marimo-pair` skill.

For marimo notebooks in this repo, treat column layout as structural:
- Inspect and preserve the notebook's `app = marimo.App(width=...)` setting and existing `@app.cell(column=...)` anchors before editing.
- If a notebook already uses columns, preserve that layout unless the user explicitly asks for a redesign.
- Remember that `column` markers are sparse boundary markers: cells without an explicit `column` inherit the previous cell's column, so reordering cells can silently change layout.
- When adding content inside an existing column, usually let it inherit; when starting a new visual column, make the first cell a Python cell with explicit `column=N`.
- Preserve spacer cells, hidden markdown cells, and summary cells when they appear to be serving layout or presentation roles.

For marimo notebooks in this repo:
- Preserve a script-mode save path. Each notebook that renders a plot must include a cell that saves the plot to file when `mo.app_meta().mode == "script"`.
- Verification must include `uvx marimo check <notebook.py>` for each edited marimo notebook.
- Verification must include `uv run <notebook.py>` for each edited marimo notebook.
- Verification is not complete until the generated plot files have been inspected carefully.
- Look for semantic or visual regressions including jagged or suspicious shapes, unexpectedly sparse or smelly plots, overlapping titles, overlapping legends, clipped labels, broken facet layouts, and similar presentation issues.
