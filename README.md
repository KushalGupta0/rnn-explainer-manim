# Manim Videos: Recurrent Neural Networks Explainer

Educational 3Blue1Brown-style animation project explaining Recurrent Neural Networks (RNNs), Backpropagation Through Time (BPTT), and vanishing/exploding gradients using Manim Community Edition.

## Architecture

The project adheres to strict architectural separation defined in `RULES.md`:
1. `manim-grounding/`: Generic Manim CE grounding tools (`api.py`, `check.py`), audited pitfalls, and guidelines.
2. `lib/`: Topic-neutral reusable animation components (`palette.py`, `video_scene.py`, `components.py`, and component galleries).
3. `project/`: Topic-specific animation scenes for the RNN explainer (implemented in Phase 3).
4. `RNN Video/`: Comprehensive pedagogical course notes, derivations, and animation specifications.

## Setup & Requirements

- Python >= 3.14
- Manim Community v0.21.0
- MiKTeX / LaTeX (with `dvisvgm`)
- FFmpeg

Install dependencies:
```bash
uv sync
```
