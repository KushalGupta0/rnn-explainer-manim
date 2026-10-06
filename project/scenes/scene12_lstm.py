"""
Scene 12: Long Short-Term Memory (LSTM) — The Linear Additive Highway
Constructs the LSTM from first principles: dual-channel topology (C_t and h_t),
the forget gate valve, additive candidate memory injection, and the Constant Error Carousel (CEC).
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


class Scene12LSTM(VideoScene):
    def construct(self) -> None:
        title = Text("LSTM: The Linear Additive Highway", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:12 | 12s): Dual Channel Topology
        # -------------------------------------------------------------
        self.show_caption("Hochreiter and Schmidhuber (1997) decoupled memory into a protected linear conveyor belt.")

        # LSTM Outer Cell boundary shifted to DOWN * 0.45
        cell_rect = Rectangle(
            width=8.6,
            height=3.5,
            stroke_color=PARAM,
            stroke_width=2.0,
            fill_color=PARAM,
            fill_opacity=0.08,
        ).shift(DOWN * 0.45)

        # Highway 1: Cell State C_t (Upper golden linear pipeline)
        c_y = 0.55
        c_in_tag = MathTex(r"\mathbf{C}_{t-1}", font_size=FONT_SIZE_LABEL, color=WARNING).move_to([-5.1, c_y, 0])
        c_out_tag = MathTex(r"\mathbf{C}_t", font_size=FONT_SIZE_LABEL, color=WARNING).move_to([5.1, c_y, 0])
        c_highway = Arrow(start=[-4.5, c_y, 0], end=[4.5, c_y, 0], buff=0, color=WARNING, stroke_width=4.0)
        c_label = Text("Cell State Highway (Linear)", font_size=FONT_SIZE_LABEL - 5, color=WARNING).next_to(c_highway, UP, buff=0.1)

        # Highway 2: Hidden State h_t (Lower working state)
        h_y = -1.55
        h_in_tag = MathTex(r"\mathbf{h}_{t-1}", font_size=FONT_SIZE_LABEL, color=PRIMARY).move_to([-5.1, h_y, 0])
        h_out_tag = MathTex(r"\mathbf{h}_t", font_size=FONT_SIZE_LABEL, color=PRIMARY).move_to([5.1, h_y, 0])
        h_channel = Arrow(start=[-4.5, h_y, 0], end=[4.5, h_y, 0], buff=0, color=PRIMARY, stroke_width=3.0)
        h_label = Text("Hidden State (Working Memory)", font_size=FONT_SIZE_LABEL - 5, color=PRIMARY).next_to(h_channel, DOWN, buff=0.1)

        # Input token x_t from below
        x_in_tag = MathTex(r"\mathbf{x}_t", font_size=FONT_SIZE_LABEL, color=INPUT).move_to([-2.5, -2.5, 0])
        x_arrow = Arrow(start=[-2.5, -2.25, 0], end=[-2.5, -1.75, 0], buff=0, color=INPUT, stroke_width=3.0)

        self.play(Create(cell_rect), run_time=0.9)
        self.play(FadeIn(c_in_tag), FadeIn(c_out_tag), Create(c_highway), FadeIn(c_label), run_time=1.0)
        self.play(FadeIn(h_in_tag), FadeIn(h_out_tag), Create(h_channel), FadeIn(h_label), run_time=1.0)
        self.play(FadeIn(x_in_tag), Create(x_arrow), run_time=0.6)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 2 (0:12 - 0:25 | 13s): The Forget Gate Valve
        # -------------------------------------------------------------
        self.show_caption("The forget gate computes a scaling vector in (0, 1) to regulate memory retention.")

        # Forget gate box and multiply operator
        f_box = LabeledBox(
            r"$\mathbf{f}_t = \sigma(\cdot)$",
            width=1.7,
            height=0.6,
            stroke_color=LOSS,
            fill_color=LOSS,
            fill_opacity=0.25,
            font_size=FONT_SIZE_LABEL - 4,
        ).move_to([-2.2, -0.5, 0])

        mult_node = Node(r"$\odot$", radius=0.25, stroke_color=WARNING, fill_color=WARNING, fill_opacity=0.3).move_to([-2.2, c_y, 0])
        f_to_mult = Arrow(start=f_box.get_top(), end=mult_node.get_bottom(), buff=0.08, color=LOSS, stroke_width=2.5)

        self.play(FadeIn(f_box), Create(mult_node), Create(f_to_mult), run_time=1.0)

        f_formula = MathTex(
            r"\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \in (0, 1)^d",
            font_size=FONT_SIZE_LABEL,
            color=LOSS,
        ).shift(UP * 2.3)

        f_behavior = MathTex(
            r"\mathbf{f}_t \approx 1 \implies \text{Remember forever} \quad\Bigg|\quad \mathbf{f}_t \approx 0 \implies \text{Erase memory}",
            font_size=FONT_SIZE_LABEL - 2,
            color=WHITE,
        ).next_to(f_formula, DOWN, buff=0.15)

        self.play(FadeIn(f_formula), FadeIn(f_behavior), run_time=1.0)
        self.play(Indicate(mult_node, color=WARNING, scale_factor=1.2), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 3 (0:25 - 0:38 | 13s): Additive Writing: Input Gate & Candidate
        # -------------------------------------------------------------
        self.show_caption("New candidate facts are scaled by an input gate and ADDED directly into the cell state.")

        self.play(FadeOut(f_formula), FadeOut(f_behavior), run_time=0.5)

        # Input gate and Candidate boxes
        i_box = LabeledBox(
            r"$\mathbf{i}_t = \sigma(\cdot)$",
            width=1.5,
            height=0.6,
            stroke_color=ACCENT,
            fill_color=ACCENT,
            fill_opacity=0.25,
            font_size=FONT_SIZE_LABEL - 4,
        ).move_to([0.2, -0.5, 0])

        c_tilde_box = LabeledBox(
            r"$\tilde{\mathbf{C}}_t = \tanh(\cdot)$",
            width=1.9,
            height=0.6,
            stroke_color=SECONDARY,
            fill_color=SECONDARY,
            fill_opacity=0.25,
            font_size=FONT_SIZE_LABEL - 4,
        ).move_to([2.1, -0.5, 0])

        add_node = Node(r"$+$", radius=0.25, stroke_color=WARNING, fill_color=WARNING, fill_opacity=0.3).move_to([1.1, c_y, 0])
        write_arrow = Arrow(start=[1.1, -0.2, 0], end=add_node.get_bottom(), buff=0.08, color=ACCENT, stroke_width=2.5)

        self.play(FadeIn(i_box), FadeIn(c_tilde_box), Create(add_node), Create(write_arrow), run_time=1.0)

        update_eq = MathTex(
            r"\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t",
            font_size=FONT_SIZE_BODY + 2,
            color=WARNING,
        ).shift(UP * 2.3)

        additive_badge = LabeledBox(
            r"$\mathbf{\text{Pure Additive Update: No repeated matrix squashing!}}$",
            width=6.6,
            height=0.5,
            stroke_color=WARNING,
            fill_color=WARNING,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL - 3,
        ).next_to(update_eq, DOWN, buff=0.15)

        self.play(FadeIn(update_eq), FadeIn(additive_badge), run_time=1.0)
        self.play(Indicate(add_node, color=WARNING, scale_factor=1.2), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:38 - 0:54 | 16s): Output Gate & Constant Error Carousel
        # -------------------------------------------------------------
        self.show_caption("The Constant Error Carousel provides an unattenuated gradient highway across time.")

        self.play(
            FadeOut(cell_rect),
            FadeOut(c_in_tag),
            FadeOut(c_out_tag),
            FadeOut(c_highway),
            FadeOut(c_label),
            FadeOut(h_in_tag),
            FadeOut(h_out_tag),
            FadeOut(h_channel),
            FadeOut(h_label),
            FadeOut(x_in_tag),
            FadeOut(x_arrow),
            FadeOut(f_box),
            FadeOut(mult_node),
            FadeOut(f_to_mult),
            FadeOut(i_box),
            FadeOut(c_tilde_box),
            FadeOut(add_node),
            FadeOut(write_arrow),
            FadeOut(update_eq),
            FadeOut(additive_badge),
            run_time=0.6,
        )

        cec_title = Text("The Constant Error Carousel (CEC)", font_size=FONT_SIZE_BODY, color=PRIMARY).shift(UP * 1.5)

        cec_deriv = MathTex(
            r"\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} \approx \operatorname{diag}(\mathbf{f}_t)",
            font_size=FONT_SIZE_BODY + 2,
            color=WARNING,
        ).next_to(cec_title, DOWN, buff=0.35)

        cec_card = VGroup(
            MathTex(r"\mathbf{1.\ \text{Multi-step gradient: } \frac{\partial \mathbf{C}_T}{\partial \mathbf{C}_t} \approx \prod_{k=t+1}^T \operatorname{diag}(\mathbf{f}_k)}", font_size=FONT_SIZE_LABEL, color=WHITE),
            MathTex(r"\mathbf{2.\ \text{When } \mathbf{f}_k \approx \mathbf{1}, \text{ the Jacobian is the Identity: } \frac{\partial \mathbf{C}_T}{\partial \mathbf{C}_t} \approx \mathbf{I}}", font_size=FONT_SIZE_LABEL, color=ACCENT),
            MathTex(r"\mathbf{3.\ \text{Gradients survive 100+ timesteps with ZERO exponential decay.}}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(cec_deriv, DOWN, buff=0.4)

        self.play(FadeIn(cec_title), FadeIn(cec_deriv), run_time=1.0)
        self.play(FadeIn(cec_card), run_time=1.0)
        self.play(Indicate(cec_card[1], color=ACCENT, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)
