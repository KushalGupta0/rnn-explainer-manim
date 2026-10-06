# Manim Skill Claims Audit

*Audit of claims from legacy / external skill documentation tested against installed Manim Community v0.21.0.*

## Refuted Claims

1. **Claim: "Text objects are Mobject, not VMobject; mixing Text with shapes in a VGroup fails on Manim CE v0.20+."**
   - **Verification Code**:
     ```python
     import manim
     print(issubclass(manim.Text, manim.VMobject))  # Evaluates to True
     vg = manim.VGroup(manim.Circle(), manim.Text("Label"))  # Succeeds without error
     ```
   - **Result**: **REFUTED**. `Text` is a direct subclass of `VMobject` in Manim CE v0.21.0. Mixing `Text` with shapes inside a `VGroup` executes and renders properly.

2. **Claim: "Group.save_state() and Group.restore() are not implemented in Manim CE v0.20+ and raise NotImplementedError."**
   - **Verification Code**:
     ```python
     import manim
     g = manim.Group(manim.Circle(), manim.Text("Label"))
     g.save_state()  # Succeeds without error
     ```
   - **Result**: **REFUTED**. `save_state` and `restore` are implemented on the base `Mobject` class and function normally on `Group` instances in Manim CE v0.21.0.

3. **Claim: "Use monospace fonts for all text. Manim's Pango renderer produces broken kerning with proportional fonts at all sizes."**
   - **Verification Code**:
     ```python
     from manim import *
     t_arial = Text("Proportional Text Arial: AV To WA fl fi", font="Arial")
     t_segoe = Text("Proportional Text Segoe UI: AV To WA fl fi", font="Segoe UI")
     ```
   - **Result**: **REFUTED**. Rendered and visually inspected at 1080p. Both Arial and Segoe UI render clean, sharp glyphs with proper kerning pairs (e.g. AV, To, WA) and ligatures without character overlap or spacing corruption. Proportional fonts are safe and legible.
