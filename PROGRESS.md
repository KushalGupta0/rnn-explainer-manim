# RNN Explainer Video Project Progress

## Phase 1: Environment, Grounding Tools, and Generic Skill
- **Status**: Completed
- **What is done**:
  - Environment verified: Python 3.14.7, Manim Community v0.21.0, MiKTeX pdfTeX 4.23, ffmpeg 9.0.2.
  - Pinned `manim==0.21.0` in `requirements.txt`.
  - Created `RULES.md` in workspace root.
  - Implemented grounding tools `manim-grounding/tools/api.py` and `manim-grounding/tools/check.py`.
  - Verified tools against `good_scene.py` (0 errors) and `bad_scene.py` (caught 8 errors, 2 warnings).
  - Authored `manim-grounding/SKILL.md`, `pitfalls.md`, `audit.md`, and `components.md`.
- **Known problems**:
  - `ManimColor` raises `TypeError` on `==` comparison with non-color objects (resolved in `check.py` by using identity check `is`).

## Phase 2: Reusable Library (`lib/`)
- **Status**: Completed
- **What is done**:
  - Configured MiKTeX automatic package installation via `initexmf --enable-installer`; verified with MathTex test scene.
  - Audited typography: confirmed Pango handles proportional fonts (Arial, Segoe UI) with correct kerning, refuting legacy skill myth. Recorded in `audit.md`.
  - Created single style source `lib/palette.py` with topic-neutral semantic colors, standard font sizes, run times, and wait constants.
  - Implemented `lib/video_scene.py` base scene with dark background, `show_caption()` / `clear_caption()`, and opt-in layout self-check (`ENABLE_LAYOUT_CHECK`) detecting frame overflow and text overlaps.
  - Implemented 8 verified components in `lib/components.py`: `LabeledBox`, `Node`, `NodeLayer`, `Connector`, `BoxChain`, `MatrixGrid`, `CaptionBar`, and `flow_along`.
  - Built boundary calculation `_get_mobject_edge_point()` ensuring `Connector` and `BoxChain` arrows attach cleanly to mobject perimeters without intersecting contents.
  - Created 8 gallery scenes in `lib/gallery/` and rendered all with `-ql` (or `-ql -s`).
  - Visually inspected rendered still frames and video via `view_file` to verify clean layouts.
  - Updated `manim-grounding/components.md` with all 8 components marked `verified`.
- **What is next (Phase 3)**:
  - Phase 3: Project architecture, storyboard, and scene scripts in `project/` implementing the RNN explainer scenes.
- **Known problems**:
  - Manim CLI sets `sys.path[0]` to the script's folder; gallery scripts require workspace root insertion to import `lib`.
