---
name: manim-grounding
description: Generic Manim Community Edition grounding tools, layout rules, and static checks for verified animations.
---

# Manim Community Edition Grounding Skill

## The Grounding Rule
The installed Manim package is the sole source of truth. Never write a Manim API call from memory.
Before writing any class, animation, or method call, inspect it with:
```bash
python manim-grounding/tools/api.py <Name>
python manim-grounding/tools/api.py --methods <ClassName>
```
If a symbol or keyword argument does not exist in the installed package, do not use it.

## 6-Step Scene Workflow
1. **Plan**: Define visual objective, persistent elements, entering/exiting elements, and timing.
2. **Ground**: Look up exact signatures and defaults using `api.py`.
3. **Draft**: Write one standalone file per scene in `project/scenes/`.
4. **Lint**: Run static sanity check:
   ```bash
   python manim-grounding/tools/check.py project/scenes/scene_name.py
   ```
5. **Fast Preview**: Render a low-quality still frame (`-ql -s`) to verify layout, scale, and alignment:
   ```bash
   manim -ql -s project/scenes/scene_name.py SceneName
   ```
6. **Final Render**: Fix layout flaws, verify against check.py, then render full animation:
   ```bash
   manim -qm project/scenes/scene_name.py SceneName
   ```

## Layout Discipline
- **Relative Positioning**: Always prefer `next_to`, `arrange`, `to_edge`, `to_corner`, and `align_to` over hard-coded coordinate vectors.
- **Safe Margins**: Keep all critical text and equations at least `buff=0.5` away from screen edges (`to_edge(UP, buff=0.5)`).
- **Zero Accidental Overlaps**: Before placing new elements, move or fade existing elements out, or use `next_to` with explicit `buff`.
- **Cognitive Load Limits**: Maximum 5–7 visual elements active simultaneously. Use opacity layering (primary 1.0, secondary 0.4, structural 0.15).
- **Pacing**: Always include `self.wait(1.0)` to `self.wait(2.0)` after reveals to give viewers breathing room.

## Error-Triage Protocol
When an error or visual defect occurs:
1. **Read Traceback**: Identify the exact line number, exception type, and failing call.
2. **Classify**:
   - *AttributeError/TypeError*: Mobject method or argument mismatch.
   - *LaTeX Error*: Missing raw string prefix `r"..."`, unescaped braces, or missing package.
   - *Visual Bug*: Overlapping coordinates, element animated before being added, missing background color.
3. **Verify with `api.py`**: Query the exact method signature on the installed package.
4. **Fix & Record**: Fix the bug in code, run `tools/check.py`, confirm with a `-ql -s` render, and add a confirmed entry to `manim-grounding/pitfalls.md`.
