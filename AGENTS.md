# Repository Guidelines for Figures & Assets

## Skills & Workflow

- **Paper Figures**: Always use the **`cs-paper-figure`** skill (`/cs-paper-figure`, located at `~/.gemini/config/skills/cs-paper-figure/SKILL.md`) when designing, drawing, critiquing, or redesigning any figure in this paper.
- Follow the `cs-paper-figure` workflow: use `paperkit.Canvas` in `scripts/`, enforce `assert not cv.overlaps() and not cv.text_overlaps() and not cv.wire_crossings()`, and verify with `paperkit.py fill` and `paperkit.py thumb` before finalizing.
- **No PDF Generation**: When using skills (including `cs-paper-figure`), **do not generate PDF files** (export PNG/SVG only, and do not compile LaTeX to PDF unless explicitly requested).

## Directory Structure & Asset Rules

1. **Final Figures (`img/`)**:
   - `img/` must contain **only** final deliverable figures referenced by the paper (e.g., `fig1_overview_v2.png`, `fig1_overview_v2.svg`, `fig2_*.jpg`, etc.).
   - **Do not generate PDF files** for figures (do not export or keep `.pdf` in `img/`).
   - Never leave intermediate renders, thumbnails, comparison sheets, or temporary crops in `img/`.

2. **Intermediate Assets (`assets/`)**:
   - All intermediate artifacts must be placed under `assets/` in the repository root:
     - **`assets/shared/`**: Shared resources reused across multiple figures, such as URDF/CAD models, Blender/mesh rendering scripts (`assets/shared/urdf_render/`), and common palettes or templates.
     - **`assets/fig<N>/`** (e.g., `assets/fig1/`, `assets/fig2/`, ...): All figure-specific intermediate files for Figure *N*, including rendered 3D model PNGs, synthetic heatmaps/masks, raw crops, comparison sheets (`fig<N>_compare.png`), and thumbnail checks (`fig<N>_thumb.png`).

3. **Figure Scripts (`scripts/`)**:
   - Reproducible figure generation scripts live in `scripts/` (e.g., `scripts/gen_fig1.py`).
   - Scripts should read/write figure-specific intermediate assets from `assets/fig<N>/` and write final outputs to `img/`.
   - When embedding transparent RGBA renders (e.g., URDF/Blender renders) inside tinted or grey-ground panels, preserve the RGBA alpha channel directly rather than compositing over pure white.
   - **Record User Guidance in Comments**: Whenever modifying a figure-generation Python script, briefly document the user's visual/design instructions as concise inline comments in the script so subsequent edits preserve those decisions.
