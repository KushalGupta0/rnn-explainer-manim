"""
Scene 10: The Jacobian Matrix View & Spectral Norm Bounds
Generalizes scalar recurrence to high-dimensional state spaces, defines the transition Jacobian J_t,
visualizes geometric volume squashing via SVD, and establishes the spectral norm upper bound.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import (
    Text,
    MathTex,
    VGroup,
    Circle,
    Ellipse,
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


class Scene10JacobianMatrix(VideoScene):
    def construct(self) -> None:
        title = Text("The Jacobian Matrix: Spectral Dynamics", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:12 | 12s): Multidimensional Jacobian Definition
        # -------------------------------------------------------------
        self.show_caption("In vector spaces, the single-step derivative is a Jacobian matrix: J = D * W_hh.")

        state_eq = MathTex(
            r"\mathbf{h}_t = \phi(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})",
            font_size=FONT_SIZE_BODY,
            color=WHITE,
        ).shift(UP * 1.5)

        jacobian_def = MathTex(
            r"\mathbf{J}_t \equiv \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \mathbf{D}_t \mathbf{W}_{hh}^\top, \quad \text{where } \mathbf{D}_t = \operatorname{diag}(\phi'(\mathbf{a}_t))",
            font_size=FONT_SIZE_LABEL + 2,
            color=PARAM,
        ).next_to(state_eq, DOWN, buff=0.35)

        chain_eq = MathTex(
            r"\text{Multi-Step Matrix Product: } \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^T \mathbf{J}_k = \mathbf{J}_T \mathbf{J}_{T-1} \cdots \mathbf{J}_{t+1}",
            font_size=FONT_SIZE_LABEL + 2,
            color=PRIMARY,
        ).next_to(jacobian_def, DOWN, buff=0.35)

        self.play(FadeIn(state_eq), run_time=0.8)
        self.play(FadeIn(jacobian_def), run_time=0.9)
        self.play(FadeIn(chain_eq), run_time=0.9)
        self.play(Indicate(chain_eq, color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 2 (0:12 - 0:25 | 13s): SVD Volume Squashing (Circle to Ellipse)
        # -------------------------------------------------------------
        self.show_caption("Chained Jacobians repeatedly squash the unit sphere into an ultra-thin ellipsoid.")

        self.play(
            FadeOut(state_eq),
            FadeOut(jacobian_def),
            FadeOut(chain_eq),
            run_time=0.6,
        )

        geom_title = Text("Geometric Action: SVD Singular Value Collapse", font_size=FONT_SIZE_LABEL + 2, color=WARNING).to_edge(UP, buff=1.2)

        # Center for geometric transformation
        center = LEFT * 3.0 + DOWN * 0.3

        # Initial unit sphere (Circle)
        shape_0 = Circle(radius=1.3, color=ACCENT, stroke_width=2.5).move_to(center)
        shape_lbl = Text("Unit Sphere (t = 0)", font_size=FONT_SIZE_LABEL - 4, color=ACCENT).next_to(shape_0, DOWN, buff=0.2)

        # Right side: SVD table
        svd_title = Text("Singular Values (Semi-Axes):", font_size=FONT_SIZE_LABEL, color=WHITE).shift(RIGHT * 2.8 + UP * 0.8)
        r0 = MathTex(r"\text{Step } 0: \quad \sigma_1 = 1.00, \quad \sigma_2 = 1.00", font_size=FONT_SIZE_LABEL - 2, color=ACCENT).next_to(svd_title, DOWN, buff=0.25)
        r1 = MathTex(r"\text{Step } 1: \quad \sigma_1 = 0.85, \quad \sigma_2 = 0.42", font_size=FONT_SIZE_LABEL - 2, color=PRIMARY).next_to(r0, DOWN, buff=0.2)
        r2 = MathTex(r"\text{Step } 2: \quad \sigma_1 = 0.58, \quad \sigma_2 = 0.15", font_size=FONT_SIZE_LABEL - 2, color=WARNING).next_to(r1, DOWN, buff=0.2)
        r3 = MathTex(r"\text{Step } 3: \quad \sigma_1 = 0.27, \quad \sigma_2 \approx 0.03 \to 0", font_size=FONT_SIZE_LABEL - 2, color=LOSS).next_to(r2, DOWN, buff=0.2)

        self.play(FadeIn(geom_title), Create(shape_0), FadeIn(shape_lbl), FadeIn(svd_title), FadeIn(r0), run_time=1.0)

        # Transformation 1: Squashed into rotated ellipse
        shape_1 = Ellipse(width=2.2, height=1.1, color=PRIMARY, stroke_width=2.5).rotate(np.deg2rad(30)).move_to(center)
        lbl_1 = Text("1 Step: J₁ applied", font_size=FONT_SIZE_LABEL - 4, color=PRIMARY).next_to(shape_1, DOWN, buff=0.2)
        self.play(ReplacementTransform(shape_0, shape_1), ReplacementTransform(shape_lbl, lbl_1), FadeIn(r1), run_time=0.8)

        # Transformation 2: More squashed
        shape_2 = Ellipse(width=1.5, height=0.4, color=WARNING, stroke_width=2.5).rotate(np.deg2rad(45)).move_to(center)
        lbl_2 = Text("2 Steps: J₂ J₁ applied", font_size=FONT_SIZE_LABEL - 4, color=WARNING).next_to(shape_2, DOWN, buff=0.2)
        self.play(ReplacementTransform(shape_1, shape_2), ReplacementTransform(lbl_1, lbl_2), FadeIn(r2), run_time=0.8)

        # Transformation 3: Extremely flat (pancake line)
        shape_3 = Ellipse(width=0.7, height=0.08, color=LOSS, stroke_width=2.5).rotate(np.deg2rad(45)).move_to(center)
        lbl_3 = Text("Volume → 0 (Vanishing)", font_size=FONT_SIZE_LABEL - 4, color=LOSS).next_to(shape_3, DOWN, buff=0.2)
        self.play(ReplacementTransform(shape_2, shape_3), ReplacementTransform(lbl_2, lbl_3), FadeIn(r3), run_time=0.8)
        self.play(Indicate(r3, color=LOSS), run_time=0.7)
        self.wait(3.0)

        # -------------------------------------------------------------
        # BEAT 3 (0:25 - 0:38 | 13s): The Spectral Norm Upper Bound
        # -------------------------------------------------------------
        self.show_caption("The maximum singular value of W_hh provides a rigorous upper bound on gradient norms.")

        self.play(
            FadeOut(geom_title),
            FadeOut(shape_3),
            FadeOut(lbl_3),
            FadeOut(svd_title),
            FadeOut(r0),
            FadeOut(r1),
            FadeOut(r2),
            FadeOut(r3),
            run_time=0.6,
        )

        bound_title = MathTex(
            r"\text{Rigorous Spectral Upper Bound:}",
            font_size=FONT_SIZE_BODY,
            color=PRIMARY,
        ).to_edge(UP, buff=1.2)

        norm_inequality = MathTex(
            r"\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\|_2 \le \prod_{k=t+1}^T \|\mathbf{D}_k\|_2 \|\mathbf{W}_{hh}\|_2 \le \big(\sigma_{\max}(\mathbf{W}_{hh})\big)^{T-t}",
            font_size=FONT_SIZE_BODY,
            color=WHITE,
        ).next_to(bound_title, DOWN, buff=0.35)

        norm_note = MathTex(
            r"\text{Since } |\phi'(a)| \le 1.0 \text{ for tanh}, \quad \|\mathbf{D}_k\|_2 \le 1.0",
            font_size=FONT_SIZE_LABEL,
            color=PARAM,
        ).next_to(norm_inequality, DOWN, buff=0.35)

        self.play(FadeIn(bound_title), run_time=0.8)
        self.play(FadeIn(norm_inequality), run_time=0.9)
        self.play(FadeIn(norm_note), run_time=0.8)
        self.play(Indicate(norm_inequality, color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:38 - 0:50 | 12s): The Spectral Criterion Summary
        # -------------------------------------------------------------
        self.show_caption("If sigma_max is below one, gradients vanish everywhere; if above one, directions may explode.")

        self.play(
            FadeOut(bound_title),
            FadeOut(norm_inequality),
            FadeOut(norm_note),
            run_time=0.6,
        )

        summary_title = Text("The Spectral Norm Criterion", font_size=FONT_SIZE_BODY, color=WARNING).shift(UP * 1.5)

        criteria_card = VGroup(
            MathTex(r"\mathbf{1.\ \sigma_{\max}(\mathbf{W}_{hh}) < 1.0 \implies \text{Strictly Contractive} \implies \text{Vanishing Gradients}}", font_size=FONT_SIZE_LABEL, color=MUTED),
            MathTex(r"\mathbf{2.\ \sigma_{\max}(\mathbf{W}_{hh}) > 1.0 \implies \text{Expansion along Principal Axes} \implies \text{Exploding Risk}}", font_size=FONT_SIZE_LABEL, color=LOSS),
            MathTex(r"\mathbf{3.\ \text{Eigenvalues are not enough: Non-normal matrices cause transient explosion!}}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(summary_title, DOWN, buff=0.45)

        self.play(FadeIn(summary_title), FadeIn(criteria_card), run_time=1.0)
        self.play(Indicate(criteria_card[2], color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)
