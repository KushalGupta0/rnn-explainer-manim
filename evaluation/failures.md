# Failure & Resolution Log

This document records notable engineering and rendering issues encountered during production, their classification, and their technical resolutions.

---

### 1. `flow_along` Arrowhead Artifact
- **Category**: `STYLE_INCONSISTENCY` / `API_ERROR`
- **Symptom**: When `flow_along` passed a pulse across an `Arrow`, a duplicate colored arrowhead was momentarily rendered on top of the tip.
- **Root Cause**: `ShowPassingFlash` iterated over all submobjects of the `Arrow`, including the triangle tip (`arrow.tip`).
- **Resolution**: Filtered out tip submobjects by slicing/extracting the line body only (`arrow.get_submobjects()[0]`), ensuring pulses travel purely along the path without duplicating arrowheads.
- **Verification**: Re-rendered `GalleryFlowAlong`, verified at $t=0.7\text{s}$ and $t=1.1\text{s}$.

---

### 2. LaTeX Ampersand Alignment Error in MathTex
- **Category**: `SYNTAX_ERROR`
- **Symptom**: LaTeX compilation failed with `Misplaced alignment tab character &`.
- **Root Cause**: Unescaped `&` in multi-line or compound math formulas intended as conjunctions rather than tabular alignment columns.
- **Resolution**: Replaced raw `&` with `\&` or split text strings cleanly using plain text labels or separate `MathTex` expressions.
- **Verification**: Confirmed MiKTeX compiled all formulas cleanly without errors.

---

### 3. Mobject Lifecycle and Double-Add in Transform / Indicate
- **Category**: `RUNTIME_ERROR` / `OVERLAP_ERROR`
- **Symptom**: Calling `self.play(Indicate(mobj))` on a mobject that was transformed or not formally in `self.mobjects` produced static checker warnings or duplicate instances remaining on screen.
- **Root Cause**: Manim CE adds target objects to the scene if indicated while another representation exists.
- **Resolution**: Ensured strict usage of `ReplacementTransform(A, B)` followed by `Indicate(B)` or `Circumscribe(B)`, keeping scene hierarchy strictly 1:1.
- **Verification**: Zero dangling mobjects reported by `check.py`.

---

### 4. VGroup FadeOut Static Checker Warning
- **Category**: `API_ERROR`
- **Symptom**: `manim-grounding/tools/check.py` warned that fading out a container `VGroup` whose individual elements were animated directly could lead to untracked lifecycle states.
- **Root Cause**: In Manim CE, adding child mobjects directly without `self.add(group)` leaves the parent `VGroup` absent from `self.mobjects`.
- **Resolution**: Refactored fade-outs to unpack explicit children: `self.play(*[FadeOut(child) for child in group])`.
- **Verification**: Passed `check.py` with 0 warnings.

---

### 5. Layout Overflow Near Frame Bounds ($Y \in [-3.4, 3.4]$)
- **Category**: `SPATIAL_LAYOUT_ERROR`
- **Symptom**: `MANIM_LAYOUT_CHECK=1` flagged bounding boxes intruding into title or caption safety margins.
- **Root Cause**: Complex diagrams with multiple stacked rows exceeded vertical bounds $|y| > 2.5$.
- **Resolution**: Scaled large network diagrams to $0.75\times$ or $0.8\times$ and constrained vertical bounds to $[-2.3, +2.3]$, reserving $y = +3.3$ strictly for titles and $y = -3.3$ strictly for subtitles/captions.
- **Verification**: All 15 scenes rendered with `MANIM_LAYOUT_CHECK=1` without a single collision or margin overflow warning.
