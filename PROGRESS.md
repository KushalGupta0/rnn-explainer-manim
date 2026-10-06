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
  - Implemented 8 components in `lib/components.py`: `LabeledBox`, `Node`, `NodeLayer`, `Connector`, `BoxChain`, `MatrixGrid`, `CaptionBar`, and `flow_along`.
  - Built boundary calculation `_get_mobject_edge_point()` ensuring `Connector` and `BoxChain` arrows attach cleanly to mobject perimeters without intersecting contents.
  - Created 8 gallery scenes in `lib/gallery/` and rendered all with `-ql` (or `-ql -s`).
  - Visually inspected rendered still frames and video via `view_file` to verify clean layouts.
- **Known problems**:
  - Manim CLI sets `sys.path[0]` to the script's folder; gallery scripts require workspace root insertion to import `lib`.

## Phase 3: Fixes, Storyboard, and Narration (`project/`)
- **Status**: Completed
- **What is done**:
  - **Part A Fixes**:
    - Fixed `flow_along` bug in `lib/components.py` by excluding tip submobjects from the passing pulse, eliminating duplicate colored arrowhead artifact.
    - Verified `Text.set_default(font=FONT_MAIN)` and `MarkupText.set_default(font=FONT_MAIN)` in `lib/palette.py`, enforcing consistent sans-serif typography across scenes and titles.
    - Re-rendered `GalleryFlowAlong` at `-ql`, extracted mid-pulse (0.7s) and near-end (1.1s) frames; visually confirmed zero tip artifacts and uniform font rendering.
    - Marked `flow_along` as `unverified` in `manim-grounding/components.md` pending user review.
    - Moved tool-specific `ManimColor` equality pitfall to new `manim-grounding/tools-notes.md`; preserved scene-writing focus in `pitfalls.md`.
    - Reran `check.py` across all 12 files in `lib/` and `lib/gallery/` with 0 errors and 0 warnings.
  - **Part B Storyboard**:
    - Created `project/visual_language.md` defining semantic color roles, typography, timing, arrow conventions, and exact LaTeX symbol mappings.
    - Created `project/storyboard.md` detailing 9 scenes (Tier 1: 260s / 72.2% runtime; Tier 2: 85s; Tier 3: 15s; Total: 360s / 6:00 min) with pedagogical objectives, on-screen objects, relative layouts, numbered beats, and fallbacks.
    - Created spoken narration scripts in `project/narration/` (`scene01` to `scene09`) synchronized to storyboard beat markers.
    - Created `project/gaps.md` categorizing missing visual components into topic-neutral (`lib/`) and RNN-specific (`project/`).
    - Created `project/kb_notes.md` providing mathematical proofs and spot-checks of the BPTT recurrence and vanishing-gradient bounds.
- **What is next (Phase 4)**:
  - Phase 4: Implementation of scene scripts in `project/` following the storyboard, building missing gap components, and rendering final animated scenes.
- **Known problems**:
  - None currently blocking.

## Phase 4A: Build Tier 1 Scenes 01, 02, and 03
- **Status**: Completed
- **What is done**:
  - Step 1 (Narration Timing Calibration): Recalibrated narration speeds across all Tier 1 scenes to a calm, natural tempo (120–135 WPM; Tier 1 total capped at exactly 300s / 5:00 min, reserving 60s for Tier 2/3). Master storyboard table and narration markdown files updated.
  - Step 2 (Gap Components):
    - Implemented topic-neutral `VectorParallelogram` in `lib/components.py`; gallery scene `GalleryVectorParallelogram` rendered and visually verified.
    - Implemented RNN-specific `RecurrentLoopCell` in `project/components/recurrent_loop_cell.py`; gallery scene `GalleryRecurrentLoopCell` rendered and visually verified.
    - Updated `manim-grounding/components.md` (both components verified; `flow_along` remains unverified pending user sign-off).
  - Step 3 (Scene Implementations & Verification):
    - `scene01_why_rnns_exist.py`: Implemented 4 beats. Upgraded to interactive visual experiment (lexical token cards mapped to $x_1 \dots x_6$, feedforward interface $W \in \mathbb{R}^{d_{out} \times 6d_x}$, physical geometric overhang of $8d_x$ sequence past dashed boundary, on-screen visual question, and in-place sequential recurrence with fixed central cell). Passed `check.py`. Rendered with `MANIM_LAYOUT_CHECK=1` (0 warnings). Contact sheet `project/review/scene01_contact.png` visually verified.
    - `scene02_the_rnn_cell.py`: Implemented 4 beats. Upgraded to live equation evaluation (equation terms light up in real time as $x_t$ and $h_{t-1}$ arrive, combine via `VectorParallelogram`, and squash into $[-1, 1]$). Passed `check.py`. Rendered with `MANIM_LAYOUT_CHECK=1` (0 warnings). Contact sheet `project/review/scene02_contact.png` visually verified.
    - `scene03_unrolling_dag.py`: Implemented 4 beats. Upgraded to physical parameter sharing (persistent master $W_{hh}$ parameter fork distributing to all 3 cells) and travelling sequential signal pulse ($x_i \to \text{cell}_i \to h_i \to \text{cell}_{i+1}$). Passed `check.py`. Rendered with `MANIM_LAYOUT_CHECK=1` (0 warnings). Contact sheet `project/review/scene03_contact.png` visually verified.
  - Updated `project/STATUS.md` recording composite evaluation scores (Scene 01: 5.0/5, Scene 02: 4.88/5, Scene 03: 5.0/5).
- **What is next (Phase 4B)**:
  - Build Tier 1 scenes 04, 05, and 06: `scene04` (Many-to-One classification), `scene05` (BPTT recurrence), and `scene06` (Vanishing gradients & Jacobian contraction) incorporating the interactive mathematical storytelling model.
- **Known problems**:
  - None currently blocking.


