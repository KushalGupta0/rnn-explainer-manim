"""
Scene 13: The Gated Recurrent Unit (GRU) — Streamlined Gating
Presents the architectural simplification of GRU: single state pathway, reset gate filtering,
convex combination update gate interpolation, and design tradeoffs with LSTM.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import (
    Text,
    MathTex,
    VGroup,
    Rectangle,
    Arrow,
    FadeIn,
    FadeOut,
    Transform,
    ReplacementTransform,
    Create,
    Indicate,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
)

from lib.video_scene import VideoScene
from lib.components import LabeledBox, Node
from lib.palette import (
    PRIMARY,
    SECONDARY,
    ACCENT,
    MUTED,
    PARAM,
    STATE,
    INPUT,
    OUTPUT,
    LOSS,
    WARNING,
    FONT_SIZE_TITLE,
    FONT_SIZE_BODY,
    FONT_SIZE_LABEL,
)


class Scene13GRU(VideoScene):
    def construct(self) -> None:
        title = Text("The Gated Recurrent Unit: Streamlined Gating", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:11 | 11s): Architectural Simplification
        # -------------------------------------------------------------
        self.show_caption("Cho et al. (2014) simplified gating by merging the cell state into a single hidden vector.")

        # GRU Outer Cell boundary
        cell_rect = Rectangle(
            width=8.6,
            height=3.4,
            stroke_color=SECONDARY,
            stroke_width=2.0,
            fill_color=SECONDARY,
            fill_opacity=0.08,
        ).shift(DOWN * 0.45)

        # Single State Pipeline: h_{t-1} -> h_t
        h_y = 0.4
        h_in_tag = MathTex(r"\mathbf{h}_{t-1}", font_size=FONT_SIZE_LABEL, color=PRIMARY).move_to([-5.1, h_y, 0])
        h_out_tag = MathTex(r"\mathbf{h}_t", font_size=FONT_SIZE_LABEL, color=PRIMARY).move_to([5.1, h_y, 0])
        h_pipe = Arrow(start=[-4.5, h_y, 0], end=[4.5, h_y, 0], buff=0, color=PRIMARY, stroke_width=4.0)
        pipe_label = Text("Single Unified Hidden State Pathway", font_size=FONT_SIZE_LABEL - 4, color=PRIMARY).next_to(h_pipe, UP, buff=0.1)

        # Input token x_t from below
        x_in_tag = MathTex(r"\mathbf{x}_t", font_size=FONT_SIZE_LABEL, color=INPUT).move_to([-2.5, -2.5, 0])
        x_arrow = Arrow(start=[-2.5, -2.25, 0], end=[-2.5, -1.7, 0], buff=0, color=INPUT, stroke_width=3.0)

        self.play(Create(cell_rect), run_time=0.9)
        self.play(FadeIn(h_in_tag), FadeIn(h_out_tag), Create(h_pipe), FadeIn(pipe_label), run_time=1.0)
        self.play(FadeIn(x_in_tag), Create(x_arrow), run_time=0.6)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 2 (0:11 - 0:23 | 12s): The Reset Gate & Candidate Memory
        # -------------------------------------------------------------
        self.show_caption("The reset gate decides how much past context is exposed when proposing new candidates.")

        r_box = LabeledBox(
            r"$\mathbf{r}_t = \sigma(\cdot)$",
            width=1.6,
            height=0.55,
            stroke_color=ACCENT,
            fill_color=ACCENT,
            fill_opacity=0.25,
            font_size=FONT_SIZE_LABEL - 4,
        ).move_to([-2.0, -0.6, 0])

        cand_box = LabeledBox(
            r"$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$",
            width=3.6,
            height=0.55,
            stroke_color=SECONDARY,
            fill_color=SECONDARY,
            fill_opacity=0.25,
            font_size=FONT_SIZE_LABEL - 4,
        ).move_to([1.2, -0.6, 0])

        r_to_cand = Arrow(start=r_box.get_right(), end=cand_box.get_left(), buff=0.1, color=ACCENT, stroke_width=2.5)

        self.play(FadeIn(r_box), FadeIn(cand_box), Create(r_to_cand), run_time=1.0)

        r_note = MathTex(
            r"\mathbf{r}_t \to 0 \implies \text{Forget past context completely (start fresh phrase)}",
            font_size=FONT_SIZE_LABEL - 2,
            color=WARNING,
        ).shift(UP * 2.3)

        self.play(FadeIn(r_note), run_time=0.8)
        self.play(Indicate(r_box, color=ACCENT), run_time=0.7)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 3 (0:23 - 0:34 | 11s): Convex Update Interpolation
        # -------------------------------------------------------------
        self.show_caption("The update gate balances past retention and candidate injection via convex combination.")

        self.play(FadeOut(r_note), run_time=0.5)

        z_box = LabeledBox(
            r"$\mathbf{z}_t = \sigma(\cdot)$",
            width=1.6,
            height=0.55,
            stroke_color=WARNING,
            fill_color=WARNING,
            fill_opacity=0.25,
            font_size=FONT_SIZE_LABEL - 4,
        ).move_to([-2.0, 0.4, 0])

        convex_formula = MathTex(
            r"\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t",
            font_size=FONT_SIZE_BODY + 2,
            color=WARNING,
        ).shift(UP * 2.3)

        convex_cases = MathTex(
            r"\mathbf{z}_t \approx 0 \implies \mathbf{h}_t \approx \mathbf{h}_{t-1} \quad\Bigg|\quad \mathbf{z}_t \approx 1 \implies \mathbf{h}_t \approx \tilde{\mathbf{h}}_t",
            font_size=FONT_SIZE_LABEL - 2,
            color=WHITE,
        ).next_to(convex_formula, DOWN, buff=0.15)

        self.play(FadeIn(z_box), FadeIn(convex_formula), FadeIn(convex_cases), run_time=1.0)
        self.play(Indicate(convex_formula, color=WARNING, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:34 - 0:45 | 11s): Design Tradeoffs: LSTM vs GRU
        # -------------------------------------------------------------
        self.show_caption("Both architectures solve vanishing gradients by embedding an additive linear shortcut.")

        self.play(
            FadeOut(cell_rect),
            FadeOut(h_in_tag),
            FadeOut(h_out_tag),
            FadeOut(h_pipe),
            FadeOut(pipe_label),
            FadeOut(x_in_tag),
            FadeOut(x_arrow),
            FadeOut(r_box),
            FadeOut(cand_box),
            FadeOut(r_to_cand),
            FadeOut(z_box),
            FadeOut(convex_formula),
            FadeOut(convex_cases),
            run_time=0.6,
        )

        tradeoff_title = Text("Architectural Comparison: LSTM vs. GRU", font_size=FONT_SIZE_BODY, color=PRIMARY).shift(UP * 1.5)

        tradeoff_card = VGroup(
            MathTex(r"\mathbf{1.\ \text{Parameter Count: GRU has 3 weight matrices vs. 4 in LSTM (25\% reduction).}}", font_size=FONT_SIZE_LABEL, color=WHITE),
            MathTex(r"\mathbf{2.\ \text{Computational Efficiency: GRU trains faster and converges with less data.}}", font_size=FONT_SIZE_LABEL, color=SECONDARY),
            MathTex(r"\mathbf{3.\ \text{Core Principle Shared: Both establish } \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} \approx \mathbf{I} \text{ additive bypass highways.}}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(tradeoff_title, DOWN, buff=0.4)

        self.play(FadeIn(tradeoff_title), FadeIn(tradeoff_card), run_time=1.0)
        self.play(Indicate(tradeoff_card[2], color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)
