# Grounding Tools Development Notes

*Notes and edge cases specifically relevant to tool/utility development (such as `api.py` and `check.py`), separate from scene-writing guidelines.*

1. **`ManimColor` Equality Comparison Raises `TypeError` with Sentinels**:
   In Manim CE v0.21.0, comparing a `ManimColor` object using `==` against a non-color sentinel (e.g. `inspect.Parameter.empty` when introspecting default arguments in reflection tools) raises:
   `TypeError: Cannot compare ManimColor with ...`
   *Resolution*: Always use identity comparison (`is sentinel`) when checking against default sentinels in signature inspection tools like `api.py`.
