"""
Scene 15: Final Synthesis — Learning Across Time
Synthesizes the complete narrative arc: Sequence Problem -> RNN Cell -> Unrolling ->
BPTT -> Vanishing/Exploding -> Gated Additive Highways -> Grand Mathematical Coda.
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


class Scene15FinalSynthesis(VideoScene):
    def construct(self) -> None:
        title = Text("Recurrent Dynamics: The Complete Synthesis", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:12 | 12s): The Conceptual Lineage
        # -------------------------------------------------------------
        self.show_caption("Recurrence solves variable sequences by reusing a stationary transformation across time.")

        core_eq = MathTex(
            r"\mathbf{h}_t = \phi(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})",
            font_size=FONT_SIZE_BODY + 2,
            color=PRIMARY,
        ).shift(UP * 1.6)

        c1 = LabeledBox(r"$\text{1. Variable Input Stream}$", width=3.4, height=0.55, stroke_color=MUTED, fill_color=MUTED, fill_opacity=0.2, font_size=FONT_SIZE_LABEL - 3).move_to([-4.0, 0.4, 0])
        c2 = LabeledBox(r"$\text{2. Parameter Sharing}$", width=3.4, height=0.55, stroke_color=PARAM, fill_color=PARAM, fill_opacity=0.2, font_size=FONT_SIZE_LABEL - 3).move_to([0.0, 0.4, 0])
        c3 = LabeledBox(r"$\text{3. Forward State Caching}$", width=3.4, height=0.55, stroke_color=ACCENT, fill_color=ACCENT, fill_opacity=0.2, font_size=FONT_SIZE_LABEL - 3).move_to([4.0, 0.4, 0])

        arr1 = Arrow(start=c1.get_right(), end=c2.get_left(), buff=0.1, color=WHITE, stroke_width=2.5)
        arr2 = Arrow(start=c2.get_right(), end=c3.get_left(), buff=0.1, color=WHITE, stroke_width=2.5)

        self.play(FadeIn(core_eq), run_time=0.9)
        self.play(FadeIn(c1), Create(arr1), FadeIn(c2), Create(arr2), FadeIn(c3), run_time=1.2)
        self.play(Indicate(core_eq, color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 2 (0:12 - 0:24 | 12s): The Optimization Barrier
        # -------------------------------------------------------------
        self.show_caption("Backpropagation reveals the fragility of temporal training: chained Jacobian multiplication.")

        bptt_chain = MathTex(
            r"\text{BPTT Gradient: } \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=t+1}^T \mathbf{J}_k, \quad \mathbf{J}_k = \mathbf{D}_k \mathbf{W}_{hh}^\top",
            font_size=FONT_SIZE_LABEL + 2,
            color=LOSS,
        ).next_to(c2, DOWN, buff=0.55)

        barrier_card = LabeledBox(
            r"$\sigma_{\max}(\mathbf{W}_{hh}) < 1 \implies \text{Vanishing Gradients} \quad\Bigg|\quad \sigma_{\max} > 1 \implies \text{Exploding Gradients}$",
            width=8.8,
            height=0.55,
            stroke_color=LOSS,
            fill_color=LOSS,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL - 3,
        ).next_to(bptt_chain, DOWN, buff=0.35)

        self.play(FadeIn(bptt_chain), FadeIn(barrier_card), run_time=1.2)
        self.play(Indicate(barrier_card, color=LOSS), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 3 (0:24 - 0:35 | 11s): The Architectural Resolution
        # -------------------------------------------------------------
        self.show_caption("Norm clipping bounds explosion; additive gated highways cure vanishing gradients.")

        self.play(
            FadeOut(core_eq),
            FadeOut(c1),
            FadeOut(c2),
            FadeOut(c3),
            FadeOut(arr1),
            FadeOut(arr2),
            FadeOut(bptt_chain),
            FadeOut(barrier_card),
            run_time=0.6,
        )

        res_title = Text("The Architectural Solutions", font_size=FONT_SIZE_BODY, color=WARNING).shift(UP * 1.5)

        sol_box = VGroup(
            MathTex(r"\mathbf{\text{1. Gradient Norm Clipping: } \mathbf{g} \leftarrow \tau \frac{\mathbf{g}}{\|\mathbf{g}\|} \implies \text{Stabilizes exploding cliffs.}}", font_size=FONT_SIZE_LABEL, color=SECONDARY),
            MathTex(r"\mathbf{\text{2. LSTM Additive Highway: } \mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \implies \text{Constant Error Carousel.}}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
            MathTex(r"\mathbf{\text{3. GRU Update Gate: } \mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \implies \text{Convex streamlined bypass.}}", font_size=FONT_SIZE_LABEL, color=WARNING),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(res_title, DOWN, buff=0.4)

        self.play(FadeIn(res_title), FadeIn(sol_box), run_time=1.0)
        self.play(Indicate(sol_box[1], color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:35 - 0:48 | 13s): Grand Mathematical Coda
        # -------------------------------------------------------------
        self.show_caption("Learning across time requires gradients to survive repeated transformations.")

        self.play(
            FadeOut(res_title),
            FadeOut(sol_box),
            run_time=0.6,
        )

        # Master Coda Box
        coda_frame = Rectangle(
            width=8.8,
            height=3.6,
            stroke_color=PRIMARY,
            stroke_width=2.5,
            fill_color=PRIMARY,
            fill_opacity=0.08,
        ).shift(DOWN * 0.2)

        master_recurrence = MathTex(
            r"\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1})",
            font_size=FONT_SIZE_TITLE + 4,
            color=PRIMARY,
        ).move_to(coda_frame.get_center()).shift(UP * 0.7)

        thesis_statement = Text(
            "Learning through time requires gradients to survive repeated transformations.",
            font_size=FONT_SIZE_LABEL - 2,
            color=WHITE,
        ).next_to(master_recurrence, DOWN, buff=0.35)

        footer_note = MathTex(
            r"\text{Vanilla Recurrence} \quad\longrightarrow\quad \text{Gated Linear Highways} \quad\longrightarrow\quad \text{Modern Attention}",
            font_size=FONT_SIZE_LABEL - 3,
            color=PARAM,
        ).next_to(thesis_statement, DOWN, buff=0.35)

        self.play(Create(coda_frame), FadeIn(master_recurrence), run_time=1.2)
        self.play(FadeIn(thesis_statement), FadeIn(footer_note), run_time=1.0)
        self.play(Indicate(master_recurrence, color=PRIMARY, scale_factor=1.08), run_time=0.9)
        self.wait(4.0)
