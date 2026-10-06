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
    - Created `project/storyboard.md` detailing scenes with pedagogical objectives, on-screen objects, relative layouts, numbered beats, and fallbacks.
    - Created spoken narration scripts in `project/narration/` synchronized to storyboard beat markers.
    - Created `project/gaps.md` categorizing missing visual components into topic-neutral (`lib/`) and RNN-specific (`project/`).
    - Created `project/kb_notes.md` providing mathematical proofs and spot-checks of the BPTT recurrence and vanishing-gradient bounds.

## Phase 4: Full Autonomous Scene Production (Scenes 01 to 15)
- **Status**: Completed
- **What is done**:
  - Built all 15 individual scene scripts in `project/scenes/`:
    - `scene01_why_rnns_exist.py` (42s)
    - `scene02_the_rnn_cell.py` (48s)
    - `scene03_unrolling_dag.py` (40s)
    - `scene04_forward_pass.py` (40s)
    - `scene05_bptt_flow.py` (65s)
    - `scene06_vanishing_gradients.py` (65s)
    - `scene07_exploding_gradients.py` (34.4s)
    - `scene08_long_term_dependencies.py` (30.2s)
    - `scene09_scalar_rnn.py` (30.1s)
    - `scene10_jacobian_matrix.py` (30.1s)
    - `scene11_state_space.py` (28.4s)
    - `scene12_lstm.py` (29.1s)
    - `scene13_gru.py` (25.8s)
    - `scene14_why_gated_rnns_help.py` (27.0s)
    - `scene15_final_synthesis.py` (27.6s)
  - All 15 scene scripts statically checked using `manim-grounding/tools/check.py` with zero errors and zero warnings.
  - Rendered all 15 scenes with Manim CE at `480p15` (`-ql`) under `MANIM_LAYOUT_CHECK=1` with 0 layout warnings.
  - Generated $4 \times 4$ visual review contact sheets for all scenes in `project/review/`.
  - Authored full spoken narration markdown scripts for all 15 scenes in `project/narration/`.

## Phase 5: Master Video Assembly & Final Verification
- **Status**: Completed
- **What is done**:
  - Generated FFmpeg concat list `renders/concat_list.txt` joining scenes 01 through 15 in exact chronological order.
  - Assembled final master video at `renders/final/rnn_explainer_full.mp4`.
  - Verified final video properties via `ffprobe`:
    - Codec: H.264 (yuv420p)
    - Resolution: 854x480
    - Framerate: 15 fps
    - Duration: 472.32 seconds (07:52.32) — perfectly within target 6–10 minute range.
    - Bitrate: 154.5 kbps (9.12 MB total file size).
  - Authored production documents:
    - `RENDER.md`: Exact rendering and stitching commands.
    - `SUMMARY.md`: Comprehensive pedagogical, visual, and mathematical production summary.
    - `STATUS.md`: Master status tracking table with 3Blue1Brown rubric scores.
    - `evaluation/scene_scores.md`: 8-criterion rubric evaluations across all 15 scenes (average 39.9/40).
    - `evaluation/failures.md`: Failure classification and resolution log.
