# Confirmed Manim CE Pitfalls

*Entries below are strictly verified by running executable test cases against installed Manim Community v0.21.0.*

1. **LaTeX Strings Require Raw Prefix (`r"..."`)**: Using standard strings like `MathTex("\\frac{1}{2}")` treats `\f` as ASCII Form Feed (0x0C), corrupting the TeX payload and causing LaTeX compilation errors. Always use `MathTex(r"\frac{1}{2}")`.
2. **`Text` Rejects `letter_spacing` Argument**: `Text()` uses Pango and raises `TypeError: Mobject.__init__() got an unexpected keyword argument 'letter_spacing'`. Use `MarkupText(r'<span letter_spacing="6000">TEXT</span>')` for custom letter kerning.
3. **`Transform(A, B)` Leaves `B` Unregistered in Scene**: Running `self.play(Transform(A, B))` modifies `A`'s geometry to match `B`; `A` remains in `self.mobjects` while `B` is never added. Subsequent animations on `B` have zero visual effect. Use `ReplacementTransform(A, B)` when you want `B` to take `A`'s place in the scene.
4. **Animation Classes Must Not Be Called as Methods**: Writing `mobject.Create()` or `mobject.FadeIn()` raises `AttributeError` or fails. Animations are classes instantiated and passed to `self.play(Create(mobject))`.
5. **Removed Legacy APIs Raise `NameError`**: Pre-CE and early Manim APIs (`ShowCreation`, `ApplyMethod`, `TextMobject`, `TexMobject`, `SmallDot`, `FadeInFrom`) do not exist in Manim CE v0.21.0. Use `Create`, `mobject.animate`, `Text`, `MathTex`, `Dot`, and `FadeIn(shift=...)`.
6. **Pop-in Artifact from Animating Unadded Mobjects**: Animating an unadded mobject with `.animate` (e.g. `self.play(mob.animate.shift(UP))`) forcefully adds it to the scene mid-animation without an entrance effect, causing a jarring pop-in. Always introduce elements deliberately via `self.add(mob)` or entrance animations (`Create`, `FadeIn`, `Write`).
