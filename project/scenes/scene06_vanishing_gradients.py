"""
Scene 06: The Vanishing Gradient — Exponential Collapse
Interactive mathematical exploration: Demonstrates the scalar recurrence chain rule,
repeated factor multiplication (0.5^n), physical gradient bar shrinkage, and horizon collapse.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import (
    Text,
    MathTex,
    VGroup,
    Rectangle,
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


class Scene06VanishingGradients(VideoScene):
    def construct(self) -> None:
        title = Text("The Vanishing Gradient: Exponential Collapse", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:13 | 13s): The Temporal Chain Rule Product
        # -------------------------------------------------------------
        self.show_caption("The chain rule backpropagates gradients through a product of Jacobians.")

        chain_rule_eq = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=t+1}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}}",
            font_size=FONT_SIZE_BODY,
            color=WHITE,
        ).shift(UP * 1.5)

        jacobian_def = MathTex(
            r"\frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \mathbf{W}_{hh}^\top \operatorname{diag}(\phi'(\mathbf{a}_k))",
            font_size=FONT_SIZE_LABEL + 2,
            color=PARAM,
        ).next_to(chain_rule_eq, DOWN, buff=0.45)

        self.play(FadeIn(chain_rule_eq, shift=DOWN * 0.15), run_time=1.2)
        self.play(FadeIn(jacobian_def), run_time=1.0)
        self.play(Indicate(chain_rule_eq, color=WARNING, scale_factor=1.05), run_time=1.0)
        self.wait(5.0)

        # -------------------------------------------------------------
        # BEAT 2 (0:13 - 0:28 | 15s): Scalar Simplification & Repeated Multiplication
        # -------------------------------------------------------------
        self.show_caption("In a scalar RNN with tanh, each factor is bounded: c = w * phi'(a) <= 0.5.")

        self.play(
            FadeOut(chain_rule_eq),
            FadeOut(jacobian_def),
            run_time=0.8,
        )

        scalar_banner = LabeledBox(
            r"$\text{Controlled Scalar Example: } c = 0.5$",
            width=5.0,
            height=0.6,
            stroke_color=WARNING,
            fill_color=WARNING,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL,
        ).to_edge(UP, buff=1.2)

        # Visual gradient bar representing physical magnitude
        # Starting with magnitude 1.0 (height 2.0)
        bar_box = Rectangle(
            width=0.8,
            height=2.0,
            fill_color=WARNING,
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=2.0,
        ).shift(LEFT * 2.2 + DOWN * 0.4)

        bar_label = Text("Gradient: 1.000", font_size=FONT_SIZE_LABEL, color=WARNING).next_to(bar_box, UP, buff=0.2)

        # Cascade equations on the right
        step0 = MathTex(r"\text{Step } T: \quad \delta = 1.000", font_size=FONT_SIZE_LABEL, color=WHITE).shift(RIGHT * 1.4 + UP * 0.8)
        step1 = MathTex(r"\text{Step } T-1: \quad 1.0 \times 0.5 = 0.500", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(step0, DOWN, buff=0.3)
        step2 = MathTex(r"\text{Step } T-2: \quad 0.5 \times 0.5 = 0.250", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(step1, DOWN, buff=0.3)
        step3 = MathTex(r"\text{Step } T-3: \quad 0.25 \times 0.5 = 0.125", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(step2, DOWN, buff=0.3)

        self.play(FadeIn(scalar_banner), FadeIn(bar_box), FadeIn(bar_label), FadeIn(step0), run_time=1.2)

        # Contraction 1: x 0.5
        bar_box_1 = Rectangle(width=0.8, height=1.0, fill_color=WARNING, fill_opacity=0.8, stroke_color=WHITE, stroke_width=2.0)
        bar_box_1.align_to(bar_box, DOWN).align_to(bar_box, LEFT)
        bar_lbl_1 = Text("Gradient: 0.500", font_size=FONT_SIZE_LABEL, color=WARNING).next_to(bar_box_1, UP, buff=0.2)
        self.play(Transform(bar_box, bar_box_1), Transform(bar_label, bar_lbl_1), FadeIn(step1), run_time=1.0)

        # Contraction 2: x 0.5
        bar_box_2 = Rectangle(width=0.8, height=0.5, fill_color=WARNING, fill_opacity=0.8, stroke_color=WHITE, stroke_width=2.0)
        bar_box_2.align_to(bar_box, DOWN).align_to(bar_box, LEFT)
        bar_lbl_2 = Text("Gradient: 0.250", font_size=FONT_SIZE_LABEL, color=WARNING).next_to(bar_box_2, UP, buff=0.2)
        self.play(Transform(bar_box, bar_box_2), Transform(bar_label, bar_lbl_2), FadeIn(step2), run_time=1.0)

        # Contraction 3: x 0.5
        bar_box_3 = Rectangle(width=0.8, height=0.25, fill_color=WARNING, fill_opacity=0.8, stroke_color=WHITE, stroke_width=2.0)
        bar_box_3.align_to(bar_box, DOWN).align_to(bar_box, LEFT)
        bar_lbl_3 = Text("Gradient: 0.125", font_size=FONT_SIZE_LABEL, color=WARNING).next_to(bar_box_3, UP, buff=0.2)
        self.play(Transform(bar_box, bar_box_3), Transform(bar_label, bar_lbl_3), FadeIn(step3), run_time=1.0)
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 3 (0:28 - 0:45 | 17s): Horizon Decay (0.5^n Table)
        # -------------------------------------------------------------
        self.show_caption("Across long sequences, exponential decay completely extinguishes the learning signal.")

        self.play(
            FadeOut(bar_box),
            FadeOut(bar_label),
            FadeOut(step0),
            FadeOut(step1),
            FadeOut(step2),
            FadeOut(step3),
            FadeOut(scalar_banner),
            run_time=0.8,
        )

        table_title = MathTex(r"\text{Exponential Collapse: } \delta_t \propto c^{T-t} = (0.5)^{T-t}", font_size=FONT_SIZE_BODY, color=WARNING).to_edge(UP, buff=1.2)

        # Horizon readout cards
        r_1 = MathTex(r"T - t = 1 \implies 0.5^1 = 0.500", font_size=FONT_SIZE_LABEL, color=WHITE).shift(UP * 0.7)
        r_5 = MathTex(r"T - t = 5 \implies 0.5^5 = 0.031", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(r_1, DOWN, buff=0.25)
        r_10 = MathTex(r"T - t = 10 \implies 0.5^{10} \approx 0.00098", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(r_5, DOWN, buff=0.25)
        r_50 = MathTex(r"T - t = 50 \implies 0.5^{50} \approx 8.88 \times 10^{-16} \to 0", font_size=FONT_SIZE_LABEL, color=LOSS).next_to(r_10, DOWN, buff=0.25)

        self.play(FadeIn(table_title), run_time=1.0)
        self.play(FadeIn(r_1), run_time=0.8)
        self.play(FadeIn(r_5), run_time=0.8)
        self.play(FadeIn(r_10), run_time=0.8)
        self.play(FadeIn(r_50), run_time=0.8)
        self.play(Indicate(r_50, color=LOSS, scale_factor=1.08), run_time=1.0)
        self.wait(5.0)

        # -------------------------------------------------------------
        # BEAT 4 (0:45 - 0:55 | 10s): The Architectural Diagnosis
        # -------------------------------------------------------------
        self.show_caption("The network is blinded to early events: weights receive zero gradient from distant history.")

        diagnosis = LabeledBox(
            r"$\lim_{T - t \to \infty} c^{T-t} = 0 \implies \text{Early timesteps receive zero weight update}$",
            width=7.8,
            height=0.65,
            stroke_color=LOSS,
            fill_color=LOSS,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL,
        ).next_to(r_50, DOWN, buff=0.45)

        self.play(FadeIn(diagnosis, shift=DOWN * 0.15), run_time=1.2)
        self.play(Indicate(diagnosis, color=LOSS), run_time=1.0)
        self.wait(4.0)

        self.clear_caption()
        self.wait(0.5)
