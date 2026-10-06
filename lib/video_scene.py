"""
Base VideoScene with Theme Setup, Captions, and Layout Verification
Topic-neutral scene base for all project animations.
"""

import os
from typing import Optional, Any, List

from manim import (
    Scene,
    config,
    FadeIn,
    FadeOut,
    ReplacementTransform,
    Text,
    Tex,
    MathTex,
    VMobject,
    UP,
    DOWN,
)

from lib.palette import BACKGROUND, RUN_TIME_FAST
from lib.components import CaptionBar


class VideoScene(Scene):
    """
    Standard base scene class providing:
    - Automatic dark theme background initialization.
    - Synchronized caption bar presentation (show_caption / clear_caption).
    - Post-construct layout boundary and text-overlap validation.
    """

    ENABLE_LAYOUT_CHECK: bool = False

    def setup(self) -> None:
        super().setup()
        self.camera.background_color = BACKGROUND
        self._caption_bar: Optional[CaptionBar] = None

        # Check if environment variable turns on layout checks
        if os.getenv("MANIM_LAYOUT_CHECK", "0").lower() in ("1", "true", "yes"):
            self.ENABLE_LAYOUT_CHECK = True

    def show_caption(self, text: str, run_time: float = RUN_TIME_FAST) -> None:
        """Displays or transitions a lower-third explanatory caption."""
        new_caption = CaptionBar(text)
        new_caption.to_edge(DOWN, buff=0.55)

        if self._caption_bar is None or self._caption_bar not in self.mobjects:
            self._caption_bar = new_caption
            self.play(FadeIn(self._caption_bar, shift=UP * 0.2), run_time=run_time)
        else:
            old_caption = self._caption_bar
            self._caption_bar = new_caption
            self.play(ReplacementTransform(old_caption, new_caption), run_time=run_time)

    def clear_caption(self, run_time: float = RUN_TIME_FAST) -> None:
        """Removes the active caption bar if visible."""
        if self._caption_bar is not None and self._caption_bar in self.mobjects:
            self.play(FadeOut(self._caption_bar, shift=DOWN * 0.2), run_time=run_time)
            self._caption_bar = None

    def tear_down(self) -> None:
        super().tear_down()
        if self.ENABLE_LAYOUT_CHECK:
            self._run_layout_check()

    def _run_layout_check(self) -> None:
        """Inspects all active mobjects for camera frame overflow and text overlaps."""
        half_w = config.frame_width / 2.0
        half_h = config.frame_height / 2.0
        margin = 0.05

        warnings: List[str] = []

        # 1. Check frame boundary overflow
        for mob in self.mobjects:
            if not isinstance(mob, VMobject):
                continue
            # Skip full-frame background rectangles
            if hasattr(mob, "width") and hasattr(mob, "height"):
                if mob.width >= config.frame_width * 0.98 and mob.height >= config.frame_height * 0.98:
                    continue

            left, right = mob.get_left()[0], mob.get_right()[0]
            bottom, top = mob.get_bottom()[1], mob.get_top()[1]

            if left < -half_w - margin or right > half_w + margin:
                warnings.append(
                    f"Mobject '{mob.__class__.__name__}' horizontally overflows frame (X range: [{left:.2f}, {right:.2f}], limit: ±{half_w:.2f})"
                )
            if bottom < -half_h - margin or top > half_h + margin:
                warnings.append(
                    f"Mobject '{mob.__class__.__name__}' vertically overflows frame (Y range: [{bottom:.2f}, {top:.2f}], limit: ±{half_h:.2f})"
                )

        # 2. Check text overlaps
        text_mobs = self._extract_text_mobjects(self.mobjects)
        for i in range(len(text_mobs)):
            for j in range(i + 1, len(text_mobs)):
                t1, t2 = text_mobs[i], text_mobs[j]
                if self._check_aabb_overlap(t1, t2):
                    t1_txt = getattr(t1, "text", t1.__class__.__name__)
                    t2_txt = getattr(t2, "text", t2.__class__.__name__)
                    warnings.append(f"Text overlap detected between '{t1_txt}' and '{t2_txt}'")

        if warnings:
            print(f"\n[Layout Check Warning in {self.__class__.__name__}] ({len(warnings)} issues):")
            for w in warnings:
                print(f"  • {w}")
        else:
            print(f"\n[Layout Check in {self.__class__.__name__}] All elements safely positioned within frame boundaries.")

    def _extract_text_mobjects(self, mobjects: List[Any]) -> List[VMobject]:
        result: List[VMobject] = []
        for mob in mobjects:
            if isinstance(mob, (Text, Tex, MathTex)):
                result.append(mob)
            elif hasattr(mob, "submobjects") and mob.submobjects:
                result.extend(self._extract_text_mobjects(mob.submobjects))
        return result

    def _check_aabb_overlap(self, m1: VMobject, m2: VMobject) -> bool:
        # Check 2D bounding box intersection with a slight forgiveness threshold
        l1, r1 = m1.get_left()[0], m1.get_right()[0]
        b1, t1 = m1.get_bottom()[1], m1.get_top()[1]

        l2, r2 = m2.get_left()[0], m2.get_right()[0]
        b2, t2 = m2.get_bottom()[1], m2.get_top()[1]

        # No overlap if separated along either axis
        if r1 < l2 + 0.05 or l1 > r2 - 0.05:
            return False
        if t1 < b2 + 0.05 or b1 > t2 - 0.05:
            return False
        return True
