"""
Scene 14: Why Gated RNNs Help — Gradient Highways Compared
Provides a direct side-by-side comparison between the multiplicative vanilla RNN and
the additive gated LSTM highway, showing simultaneous backward gradient propagation.
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
    Line,
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


class Scene14WhyGatedRNNsHelp(VideoScene):
    def construct(self) -> None:
        title = Text("Why Gated RNNs Help: Gradient Flow Compared", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:12 | 12s): Returning to the Long-Term Problem
        # -------------------------------------------------------------
        self.show_caption("Bridging long temporal gaps requires error gradients to survive dozens of steps.")

        challenge_card = LabeledBox(
            r"$\text{Long-Term Goal: } x_1: \text{clouds} \longleftrightarrow y_T: \text{rained} \quad (\Delta t = 20 \text{ steps})$",
            width=8.5,
            height=0.6,
            stroke_color=WARNING,
            fill_color=WARNING,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL,
        ).shift(UP * 1.5)

        core_question = MathTex(
            r"\text{Why does an additive gated highway succeed where vanilla RNNs fail completely?}",
            font_size=FONT_SIZE_BODY,
            color=WHITE,
        ).next_to(challenge_card, DOWN, buff=0.4)

        self.play(FadeIn(challenge_card), run_time=1.0)
        self.play(FadeIn(core_question), run_time=1.0)
        self.play(Indicate(challenge_card, color=WARNING, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 2 (0:12 - 0:24 | 12s): Side-by-Side Unrolled Networks
        # -------------------------------------------------------------
        self.show_caption("Side-by-side: Vanilla RNN squashes through matrix products; LSTM flows along a linear belt.")

        self.play(
            FadeOut(challenge_card),
            FadeOut(core_question),
            run_time=0.6,
        )

        x_coords = [-4.0, -2.0, 0.0, 2.0, 4.0]

        # Top system: Vanilla RNN (Y = 1.0)
        vanilla_title = Text("Vanilla RNN (Multiplicative Jacobian Bottleneck)", font_size=FONT_SIZE_LABEL - 4, color=LOSS).move_to([-1.5, 2.0, 0])
        vanilla_nodes = VGroup()
        for x in x_coords:
            n = Node(r"$h$", radius=0.3, stroke_color=MUTED, fill_color=MUTED, fill_opacity=0.25).move_to([x, 1.2, 0])
            vanilla_nodes.add(n)

        vanilla_fwd = VGroup()
        for i in range(len(x_coords) - 1):
            arr = Arrow(start=vanilla_nodes[i].get_right(), end=vanilla_nodes[i + 1].get_left(), buff=0.08, color=MUTED, stroke_width=2.5)
            vanilla_fwd.add(arr)

        # Bottom system: Gated LSTM (Y = -1.0)
        gated_title = Text("Gated LSTM (Additive Constant Error Carousel)", font_size=FONT_SIZE_LABEL - 4, color=PRIMARY).move_to([-1.5, -0.2, 0])
        gated_nodes = VGroup()
        for x in x_coords:
            n = Node(r"$C$", radius=0.3, stroke_color=WARNING, fill_color=WARNING, fill_opacity=0.25).move_to([x, -1.0, 0])
            gated_nodes.add(n)

        gated_fwd = VGroup()
        for i in range(len(x_coords) - 1):
            arr = Arrow(start=gated_nodes[i].get_right(), end=gated_nodes[i + 1].get_left(), buff=0.08, color=WARNING, stroke_width=3.5)
            gated_fwd.add(arr)

        self.play(FadeIn(vanilla_title), FadeIn(vanilla_nodes), Create(vanilla_fwd), run_time=1.0)
        self.play(FadeIn(gated_title), FadeIn(gated_nodes), Create(gated_fwd), run_time=1.0)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 3 (0:24 - 0:38 | 14s): Backward Gradient Flow Comparison
        # -------------------------------------------------------------
        self.show_caption("In vanilla RNNs, gradients extinguish exponentially; in LSTMs, they survive intact.")

        # Vanilla backward pulses (collapsing)
        vanilla_bwd = VGroup()
        v_widths = [4.0, 2.2, 1.0, 0.3]
        v_colors = [LOSS, WARNING, MUTED, "#334155"]
        for i in range(3, -1, -1):
            arr = Arrow(
                start=vanilla_nodes[i + 1].get_left(),
                end=vanilla_nodes[i].get_right(),
                buff=0.08,
                color=v_colors[3 - i],
                stroke_width=v_widths[3 - i],
            ).shift(DOWN * 0.35)
            vanilla_bwd.add(arr)

        v_result = MathTex(r"\text{Grad at } t=1: \mathbf{0.000} \text{ (Vanished)}", font_size=FONT_SIZE_LABEL - 4, color=LOSS).next_to(vanilla_nodes[0], DOWN, buff=0.25)

        # Gated backward pulses (preserved along highway)
        gated_bwd = VGroup()
        for i in range(3, -1, -1):
            arr = Arrow(
                start=gated_nodes[i + 1].get_left(),
                end=gated_nodes[i].get_right(),
                buff=0.08,
                color=PRIMARY,
                stroke_width=3.5,
            ).shift(DOWN * 0.35)
            gated_bwd.add(arr)

        g_result = MathTex(r"\text{Grad at } t=1: \mathbf{0.900} \text{ (Intact!)}", font_size=FONT_SIZE_LABEL - 4, color=PRIMARY).next_to(gated_nodes[0], DOWN, buff=0.25)

        for va, ga in zip(vanilla_bwd, gated_bwd):
            self.play(Create(va), Create(ga), run_time=0.4)

        self.play(FadeIn(v_result), FadeIn(g_result), run_time=0.8)
        self.play(Indicate(v_result, color=LOSS), Indicate(g_result, color=PRIMARY), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:38 - 0:50 | 12s): Mathematical Summary
        # -------------------------------------------------------------
        self.show_caption("Additive gating unlocks scalable deep sequence learning across time.")

        self.play(
            FadeOut(vanilla_title),
            FadeOut(vanilla_nodes),
            FadeOut(vanilla_fwd),
            *[FadeOut(arr) for arr in vanilla_bwd],
            FadeOut(v_result),
            FadeOut(gated_title),
            FadeOut(gated_nodes),
            FadeOut(gated_fwd),
            *[FadeOut(arr) for arr in gated_bwd],
            FadeOut(g_result),
            run_time=0.6,
        )

        comparison_summary = Text("The Mathematical Breakthrough", font_size=FONT_SIZE_BODY, color=WARNING).shift(UP * 1.5)

        summary_box = VGroup(
            MathTex(r"\mathbf{\text{Vanilla RNN: } \prod_{k=t+1}^T (\mathbf{W}_{hh}^\top \mathbf{D}_k) \implies \text{Exponential Collapse } (c^{T-t} \to 0)}", font_size=FONT_SIZE_LABEL, color=LOSS),
            MathTex(r"\mathbf{\text{Gated LSTM: } \prod_{k=t+1}^T \operatorname{diag}(\mathbf{f}_k) \approx \mathbf{I} \implies \text{Constant Error Carousel (CEC)}}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
            MathTex(r"\mathbf{\text{Result: Solves Bengio's Dilemma, enabling training across 100+ steps.}}", font_size=FONT_SIZE_LABEL, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(comparison_summary, DOWN, buff=0.4)

        self.play(FadeIn(comparison_summary), FadeIn(summary_box), run_time=1.0)
        self.play(Indicate(summary_box[1], color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)
