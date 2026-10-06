"""
Scene 07: Exploding Gradients & Norm Clipping Geometry
Interactive mathematical exploration: Contrasts vanishing gradients with exponential explosion (c = 1.5^n),
demonstrates numerical overflow / NaN catastrophe, and presents Pascanu's geometric gradient norm clipping.
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
    Circle,
    Arrow,
    DashedLine,
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


class Scene07ExplodingGradients(VideoScene):
    def construct(self) -> None:
        title = Text("Exploding Gradients & Norm Clipping", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:11 | 11s): The Exponential Detonation
        # -------------------------------------------------------------
        self.show_caption("When transition Jacobians exceed unity, backpropagated gradients explode exponentially.")

        explosion_eq = MathTex(
            r"\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \right\| \ge \rho^{T-t} \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \right\|, \quad \text{where } \rho = \|\mathbf{W}_{hh}^\top \mathbf{D}\| > 1",
            font_size=FONT_SIZE_BODY,
            color=WHITE,
        ).shift(UP * 1.5)

        diagnosis_note = MathTex(
            r"\rho < 1 \implies \text{Vanishing} \to 0 \quad\Bigg|\quad \rho > 1 \implies \text{Exploding} \to \infty",
            font_size=FONT_SIZE_LABEL + 2,
            color=WARNING,
        ).next_to(explosion_eq, DOWN, buff=0.45)

        self.play(FadeIn(explosion_eq, shift=DOWN * 0.15), run_time=1.0)
        self.play(FadeIn(diagnosis_note), run_time=1.0)
        self.play(Indicate(diagnosis_note, color=LOSS, scale_factor=1.05), run_time=1.0)
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 2 (0:11 - 0:23 | 12s): Scalar Cascade & Numerical Overflow
        # -------------------------------------------------------------
        self.show_caption("At factor c = 1.5, each backward step inflates the gradient beyond floating-point limits.")

        self.play(
            FadeOut(explosion_eq),
            FadeOut(diagnosis_note),
            run_time=0.6,
        )

        scalar_banner = LabeledBox(
            r"$\text{Controlled Scalar Cascade: } c = 1.5$",
            width=5.0,
            height=0.6,
            stroke_color=LOSS,
            fill_color=LOSS,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL,
        ).to_edge(UP, buff=1.2)

        # Base frame for bar
        base_line = DashedLine(start=LEFT * 3.2 + DOWN * 1.5, end=LEFT * 1.2 + DOWN * 1.5, color=MUTED)
        ceiling_line = DashedLine(start=LEFT * 3.2 + UP * 1.5, end=LEFT * 1.2 + UP * 1.5, color=LOSS)
        ceiling_lbl = Text("Max Capacity", font_size=FONT_SIZE_LABEL - 4, color=LOSS).next_to(ceiling_line, UP, buff=0.1)

        # Gradient bar starting at 0.6 height (representing 1.0)
        bar_box = Rectangle(
            width=0.8,
            height=0.6,
            fill_color=LOSS,
            fill_opacity=0.85,
            stroke_color=WHITE,
            stroke_width=2.0,
        ).move_to(LEFT * 2.2 + DOWN * 1.2)

        bar_label = Text("Norm: 1.000", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(bar_box, UP, buff=0.15)

        # Cascade equations on the right
        step0 = MathTex(r"\text{Step } T: \quad \delta = 1.000", font_size=FONT_SIZE_LABEL, color=WHITE).shift(RIGHT * 1.5 + UP * 0.7)
        step1 = MathTex(r"\text{Step } T-1: \quad 1.0 \times 1.5 = 1.500", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(step0, DOWN, buff=0.25)
        step2 = MathTex(r"\text{Step } T-2: \quad 1.5 \times 1.5 = 2.250", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(step1, DOWN, buff=0.25)
        step3 = MathTex(r"\text{Step } T-3: \quad 2.25 \times 1.5 = 3.375", font_size=FONT_SIZE_LABEL, color=LOSS).next_to(step2, DOWN, buff=0.25)

        self.play(FadeIn(scalar_banner), Create(base_line), Create(ceiling_line), FadeIn(ceiling_lbl), FadeIn(bar_box), FadeIn(bar_label), FadeIn(step0), run_time=1.0)

        # Expansion 1: 1.5
        bar_box_1 = Rectangle(width=0.8, height=0.9, fill_color=LOSS, fill_opacity=0.85, stroke_color=WHITE, stroke_width=2.0)
        bar_box_1.align_to(bar_box, DOWN).align_to(bar_box, LEFT)
        bar_lbl_1 = Text("Norm: 1.500", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(bar_box_1, UP, buff=0.15)
        self.play(ReplacementTransform(bar_box, bar_box_1), ReplacementTransform(bar_label, bar_lbl_1), FadeIn(step1), run_time=0.8)

        # Expansion 2: 2.25
        bar_box_2 = Rectangle(width=0.8, height=1.35, fill_color=LOSS, fill_opacity=0.85, stroke_color=WHITE, stroke_width=2.0)
        bar_box_2.align_to(bar_box_1, DOWN).align_to(bar_box_1, LEFT)
        bar_lbl_2 = Text("Norm: 2.250", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(bar_box_2, UP, buff=0.15)
        self.play(ReplacementTransform(bar_box_1, bar_box_2), ReplacementTransform(bar_lbl_1, bar_lbl_2), FadeIn(step2), run_time=0.8)

        # Expansion 3: 3.375 (bursting through ceiling!)
        bar_box_3 = Rectangle(width=0.8, height=3.1, fill_color=LOSS, fill_opacity=0.9, stroke_color=WHITE, stroke_width=2.0)
        bar_box_3.align_to(bar_box_2, DOWN).align_to(bar_box_2, LEFT)
        bar_lbl_3 = Text("OVERFLOW (NaN)", font_size=FONT_SIZE_LABEL, color=LOSS).next_to(bar_box_3, UP, buff=0.15)
        self.play(ReplacementTransform(bar_box_2, bar_box_3), ReplacementTransform(bar_lbl_2, bar_lbl_3), FadeIn(step3), run_time=0.9)
        self.play(Indicate(bar_lbl_3, color=LOSS, scale_factor=1.1), run_time=0.7)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 3 (0:23 - 0:34 | 11s): Horizon Detonation Table
        # -------------------------------------------------------------
        self.show_caption("Over 50 timesteps, gradients reach hundreds of millions, corrupting model parameters.")

        self.play(
            FadeOut(bar_box_3),
            FadeOut(bar_lbl_3),
            FadeOut(base_line),
            FadeOut(ceiling_line),
            FadeOut(ceiling_lbl),
            FadeOut(step0),
            FadeOut(step1),
            FadeOut(step2),
            FadeOut(step3),
            FadeOut(scalar_banner),
            run_time=0.6,
        )

        table_title = MathTex(r"\text{Exponential Growth: } \|\delta_t\| \propto (1.5)^{T-t}", font_size=FONT_SIZE_BODY, color=LOSS).to_edge(UP, buff=1.2)

        r_1 = MathTex(r"T - t = 1 \implies 1.5^1 = 1.50", font_size=FONT_SIZE_LABEL, color=WHITE).shift(UP * 0.7)
        r_5 = MathTex(r"T - t = 5 \implies 1.5^5 \approx 7.59", font_size=FONT_SIZE_LABEL, color=WHITE).next_to(r_1, DOWN, buff=0.25)
        r_10 = MathTex(r"T - t = 10 \implies 1.5^{10} \approx 57.67", font_size=FONT_SIZE_LABEL, color=WARNING).next_to(r_5, DOWN, buff=0.25)
        r_50 = MathTex(r"T - t = 50 \implies 1.5^{50} \approx 6.38 \times 10^8 \to \mathbf{NaN}", font_size=FONT_SIZE_LABEL, color=LOSS).next_to(r_10, DOWN, buff=0.25)

        self.play(FadeIn(table_title), run_time=0.8)
        self.play(FadeIn(r_1), run_time=0.6)
        self.play(FadeIn(r_5), run_time=0.6)
        self.play(FadeIn(r_10), run_time=0.6)
        self.play(FadeIn(r_50), run_time=0.6)
        self.play(Indicate(r_50, color=LOSS, scale_factor=1.08), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:34 - 0:46 | 12s): Geometric Norm Clipping
        # -------------------------------------------------------------
        self.show_caption("Global norm clipping rescales the magnitude to threshold tau while preserving exact direction.")

        self.play(
            FadeOut(table_title),
            FadeOut(r_1),
            FadeOut(r_5),
            FadeOut(r_10),
            FadeOut(r_50),
            run_time=0.6,
        )

        clip_formula = MathTex(
            r"\mathbf{g} \leftarrow \tau \frac{\mathbf{g}}{\|\mathbf{g}\|_2} \quad \text{if } \|\mathbf{g}\|_2 > \tau",
            font_size=FONT_SIZE_BODY,
            color=PRIMARY,
        ).to_edge(UP, buff=1.2)

        # Vector space on the left: center at (-2.5, -0.4)
        center = LEFT * 2.5 + DOWN * 0.4
        tau_radius = 1.6
        thresh_circle = Circle(radius=tau_radius, color=PRIMARY, stroke_width=2.5).move_to(center)
        thresh_label = MathTex(r"\text{Threshold } \tau", font_size=FONT_SIZE_LABEL - 2, color=PRIMARY).next_to(thresh_circle, UP, buff=0.15)

        # Giant unclipped gradient vector reaching outside
        angle = np.deg2rad(35)
        unclipped_len = 3.2
        target_pt = center + np.array([unclipped_len * np.cos(angle), unclipped_len * np.sin(angle), 0])
        clipped_pt = center + np.array([tau_radius * np.cos(angle), tau_radius * np.sin(angle), 0])

        unclipped_arrow = Arrow(
            start=center,
            end=target_pt,
            buff=0,
            color=LOSS,
            stroke_width=4.0,
            max_tip_length_to_length_ratio=0.15,
        )
        unclipped_tag = MathTex(r"\mathbf{g}_{\text{raw}}", font_size=FONT_SIZE_LABEL, color=LOSS).next_to(target_pt, RIGHT, buff=0.15)

        # Explanatory card on right
        geom_card = VGroup(
            MathTex(r"\text{1. Direction Preserved: } \hat{\mathbf{g}} = \frac{\mathbf{g}}{\|\mathbf{g}\|}", font_size=FONT_SIZE_LABEL, color=WHITE),
            MathTex(r"\text{2. Step Bounded: } \|\Delta \mathbf{W}\| \le \eta \tau", font_size=FONT_SIZE_LABEL, color=WHITE),
            MathTex(r"\text{3. Prevents NaN and Loss Cliff Catapult}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(RIGHT * 2.4 + DOWN * 0.4)

        self.play(FadeIn(clip_formula), Create(thresh_circle), FadeIn(thresh_label), run_time=1.0)
        self.play(Create(unclipped_arrow), FadeIn(unclipped_tag), run_time=1.0)
        self.play(FadeIn(geom_card), run_time=0.8)

        # Projection back to sphere boundary
        clipped_arrow = Arrow(
            start=center,
            end=clipped_pt,
            buff=0,
            color=PRIMARY,
            stroke_width=4.0,
            max_tip_length_to_length_ratio=0.2,
        )
        clipped_tag = MathTex(r"\mathbf{g}_{\text{clipped}}", font_size=FONT_SIZE_LABEL, color=PRIMARY).next_to(clipped_pt, UP + RIGHT, buff=0.1)

        self.play(
            ReplacementTransform(unclipped_arrow, clipped_arrow),
            ReplacementTransform(unclipped_tag, clipped_tag),
            run_time=1.2,
        )
        self.play(Indicate(clipped_arrow, color=PRIMARY, scale_factor=1.08), run_time=0.8)
        self.wait(3.5)
