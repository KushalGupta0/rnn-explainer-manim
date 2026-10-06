"""
Scene 09: The Scalar RNN — Under the Mathematical Microscope
Reduces the multidimensional RNN to a 1D scalar model, traces explicit numerical forward steps,
derives the elementary temporal derivative, and provides the conceptual bridge to Jacobian matrices.
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
from lib.components import LabeledBox
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


class Scene09ScalarRNN(VideoScene):
    def construct(self) -> None:
        title = Text("The Scalar RNN: Under the Microscope", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:11 | 11s): Vector to Scalar Reduction
        # -------------------------------------------------------------
        self.show_caption("Stripping away vector spaces reveals the fundamental engine of gradient dynamics.")

        vector_eq = MathTex(
            r"\text{Vector RNN: } \mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b})",
            font_size=FONT_SIZE_BODY,
            color=MUTED,
        ).shift(UP * 1.5)

        scalar_eq = MathTex(
            r"\text{Scalar RNN (1D): } h_t = \tanh(w \cdot h_{t-1} + u \cdot x_t + b)",
            font_size=FONT_SIZE_BODY + 2,
            color=PRIMARY,
        ).next_to(vector_eq, DOWN, buff=0.45)

        reduction_badge = LabeledBox(
            r"$w, u, b, h_t, x_t \in \mathbb{R} \implies \text{Elementary Single-Variable Calculus}$",
            width=7.5,
            height=0.55,
            stroke_color=PRIMARY,
            fill_color=PRIMARY,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL - 2,
        ).next_to(scalar_eq, DOWN, buff=0.45)

        self.play(FadeIn(vector_eq), run_time=0.8)
        self.play(FadeIn(scalar_eq, shift=DOWN * 0.1), run_time=1.0)
        self.play(FadeIn(reduction_badge), run_time=0.8)
        self.play(Indicate(scalar_eq, color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 2 (0:11 - 0:23 | 12s): Concrete Numerical Forward Trace
        # -------------------------------------------------------------
        self.show_caption("Evaluating concrete numbers demonstrates how states evolve across timesteps.")

        self.play(
            FadeOut(vector_eq),
            FadeOut(scalar_eq),
            FadeOut(reduction_badge),
            run_time=0.6,
        )

        params_card = MathTex(
            r"\text{Parameters: } w = 0.8, \quad u = 0.5, \quad b = 0, \quad h_0 = 0.50, \quad x_t = 0.20",
            font_size=FONT_SIZE_LABEL,
            color=PARAM,
        ).to_edge(UP, buff=1.2)

        # Step 1 box
        step1_box = VGroup(
            MathTex(r"\mathbf{\text{Timestep 1:}}", font_size=FONT_SIZE_LABEL, color=ACCENT),
            MathTex(r"a_1 = 0.8(0.50) + 0.5(0.20) = 0.40 + 0.10 = 0.50", font_size=FONT_SIZE_LABEL - 2, color=WHITE),
            MathTex(r"h_1 = \tanh(0.50) \approx \mathbf{0.462}", font_size=FONT_SIZE_LABEL, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).shift(LEFT * 3.0 + DOWN * 0.3)

        # Step 2 box
        step2_box = VGroup(
            MathTex(r"\mathbf{\text{Timestep 2:}}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
            MathTex(r"a_2 = 0.8(0.462) + 0.5(0.20) = 0.370 + 0.10 = 0.470", font_size=FONT_SIZE_LABEL - 2, color=WHITE),
            MathTex(r"h_2 = \tanh(0.470) \approx \mathbf{0.438}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).shift(RIGHT * 3.0 + DOWN * 0.3)

        self.play(FadeIn(params_card), run_time=0.8)
        self.play(FadeIn(step1_box), run_time=1.0)
        self.play(FadeIn(step2_box), run_time=1.0)
        self.play(Indicate(step1_box[2], color=ACCENT), Indicate(step2_box[2], color=PRIMARY), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 3 (0:23 - 0:34 | 11s): Scalar Derivative Derivation
        # -------------------------------------------------------------
        self.show_caption("The chain rule expresses temporal sensitivity as repeated scalar multiplication.")

        self.play(
            FadeOut(params_card),
            FadeOut(step1_box),
            FadeOut(step2_box),
            run_time=0.6,
        )

        deriv_eq1 = MathTex(
            r"\frac{\partial h_t}{\partial h_{t-1}} = \frac{dh_t}{da_t} \cdot \frac{\partial a_t}{\partial h_{t-1}} = \phi'(a_t) \cdot w",
            font_size=FONT_SIZE_BODY,
            color=WHITE,
        ).to_edge(UP, buff=1.2)

        deriv_eval = MathTex(
            r"\text{At Step 1: } \phi'(a_1) = 1 - (0.462)^2 \approx 0.787 \implies \frac{\partial h_1}{\partial h_0} = 0.787 \times 0.8 = \mathbf{0.630}",
            font_size=FONT_SIZE_LABEL,
            color=WARNING,
        ).next_to(deriv_eq1, DOWN, buff=0.35)

        deriv_chain = MathTex(
            r"\text{Multi-step Sensitivity: } \frac{\partial h_T}{\partial h_t} = w^{T-t} \prod_{k=t+1}^T \phi'(a_k)",
            font_size=FONT_SIZE_BODY,
            color=LOSS,
        ).next_to(deriv_eval, DOWN, buff=0.45)

        self.play(FadeIn(deriv_eq1), run_time=0.8)
        self.play(FadeIn(deriv_eval), run_time=0.9)
        self.play(FadeIn(deriv_chain), run_time=0.9)
        self.play(Indicate(deriv_chain, color=LOSS, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:34 - 0:45 | 11s): The Bridge to Jacobians
        # -------------------------------------------------------------
        self.show_caption("In higher dimensions, scalar weights become matrices and derivatives become Jacobians.")

        self.play(
            FadeOut(deriv_eq1),
            FadeOut(deriv_eval),
            FadeOut(deriv_chain),
            run_time=0.6,
        )

        bridge_title = Text("The Bridge from 1D to Multi-Dimensional RNNs", font_size=FONT_SIZE_BODY, color=PRIMARY).shift(UP * 1.5)

        comparison_table = VGroup(
            MathTex(r"\text{1D Scalar World} \quad \longleftrightarrow \quad \text{N-Dimensional Vector World}", font_size=FONT_SIZE_LABEL + 2, color=WARNING),
            MathTex(r"\text{Scalar weight } w \quad \longleftrightarrow \quad \text{Recurrent matrix } \mathbf{W}_{hh}", font_size=FONT_SIZE_LABEL, color=WHITE),
            MathTex(r"\text{Derivative } w \cdot \phi'(a) \quad \longleftrightarrow \quad \text{Jacobian Matrix } \mathbf{J}_t = \mathbf{W}_{hh}^\top \operatorname{diag}(\phi'(\mathbf{a}))", font_size=FONT_SIZE_LABEL, color=WHITE),
            MathTex(r"\text{Scalar power } w^{T-t} \quad \longleftrightarrow \quad \text{Matrix Product } \prod_{k} \mathbf{J}_k", font_size=FONT_SIZE_LABEL, color=PRIMARY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(bridge_title, DOWN, buff=0.4)

        self.play(FadeIn(bridge_title), FadeIn(comparison_table), run_time=1.0)
        self.play(Indicate(comparison_table[3], color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)
