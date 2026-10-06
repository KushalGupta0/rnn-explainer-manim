"""
RNN-Specific Visual Component: RecurrentLoopCell
Represents the classical compact recurrent cell with circular temporal feedback loop.
"""

from typing import Optional, Union, Any
import numpy as np

from manim import (
    VGroup,
    CurvedArrow,
    Arrow,
    MathTex,
    Mobject,
    UP,
    DOWN,
    RIGHT,
    LEFT,
    TAU,
)

from lib.palette import (
    PRIMARY,
    PARAM,
    STATE,
    INPUT,
    FONT_SIZE_LABEL,
    FONT_SIZE_BODY,
)
from lib.components import LabeledBox


class RecurrentLoopCell(VGroup):
    r"""
    Compact single-cell RNN representation featuring a central processing box,
    an upward input arrow, an upward readout arrow, and a circular feedback loop.

    Example:
        cell = RecurrentLoopCell(label="A", loop_label=r"\mathbf{W}_{hh}")
    """

    def __init__(
        self,
        label: Union[str, Mobject] = "A",
        box_width: float = 1.6,
        box_height: float = 1.4,
        loop_label: Optional[Union[str, Mobject]] = r"\mathbf{W}_{hh}",
        input_label: Optional[Union[str, Mobject]] = r"\mathbf{x}_t",
        output_label: Optional[Union[str, Mobject]] = r"\mathbf{h}_t",
        stroke_color: Any = PRIMARY,
        loop_color: Any = PARAM,
        input_color: Any = INPUT,
        output_color: Any = STATE,
        font_size: float = FONT_SIZE_LABEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Central processing box
        self.box = LabeledBox(
            label=label,
            width=box_width,
            height=box_height,
            stroke_color=stroke_color,
            font_size=FONT_SIZE_BODY,
        )
        self.add(self.box)

        # Circular feedback loop curving out of the right side
        p_start = self.box.get_right() + UP * 0.35
        p_end = self.box.get_right() + DOWN * 0.35
        self.loop_arrow = CurvedArrow(
            p_start,
            p_end,
            angle=-TAU * 0.75,
            color=loop_color,
            stroke_width=2.5,
        )
        self.add(self.loop_arrow)

        # Loop label
        self.loop_label_mob: Optional[Mobject] = None
        if loop_label is not None:
            self.loop_label_mob = (
                MathTex(loop_label, font_size=font_size, color=loop_color)
                if isinstance(loop_label, str)
                else loop_label
            )
            self.loop_label_mob.next_to(self.loop_arrow, RIGHT, buff=0.15)
            self.add(self.loop_label_mob)

        # Input arrow entering from bottom
        self.input_arrow = Arrow(
            self.box.get_bottom() + DOWN * 1.2,
            self.box.get_bottom(),
            buff=0.08,
            color=input_color,
            stroke_width=2.5,
        )
        self.add(self.input_arrow)

        self.input_label_mob: Optional[Mobject] = None
        if input_label is not None:
            self.input_label_mob = (
                MathTex(input_label, font_size=font_size, color=input_color)
                if isinstance(input_label, str)
                else input_label
            )
            self.input_label_mob.next_to(self.input_arrow, LEFT, buff=0.15)
            self.add(self.input_label_mob)

        # Output arrow exiting to top
        self.output_arrow = Arrow(
            self.box.get_top(),
            self.box.get_top() + UP * 1.2,
            buff=0.08,
            color=output_color,
            stroke_width=2.5,
        )
        self.add(self.output_arrow)

        self.output_label_mob: Optional[Mobject] = None
        if output_label is not None:
            self.output_label_mob = (
                MathTex(output_label, font_size=font_size, color=output_color)
                if isinstance(output_label, str)
                else output_label
            )
            self.output_label_mob.next_to(self.output_arrow, LEFT, buff=0.15)
            self.add(self.output_label_mob)
