"""
Scene 02: The RNN Cell — First Principles & Dimensions
Interactive mathematical exploration: Step-by-step evaluation of the recurrent equation,
synchronizing vector transformations with live equation term highlighting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import (
    Text,
    MathTex,
    VGroup,
    FadeIn,
    FadeOut,
    Create,
    ReplacementTransform,
    Indicate,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
)

from lib.video_scene import VideoScene
from lib.components import LabeledBox, Node, Connector, VectorParallelogram
from lib.palette import (
    PRIMARY,
    SECONDARY,
    ACCENT,
    MUTED,
    PARAM,
    STATE,
    INPUT,
    OUTPUT,
    FONT_SIZE_TITLE,
    FONT_SIZE_BODY,
    FONT_SIZE_LABEL,
)


class Scene02TheRNNCell(VideoScene):
    def construct(self) -> None:
        title = Text("The RNN Cell: First Principles & Dimensions", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.5)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:12 | 12s): The Two Inflows
        # -------------------------------------------------------------
        self.show_caption("At each moment, the cell receives fresh input and past memory.")

        # Master Pre-activation equation at top
        eq_sum = MathTex(
            r"\mathbf{a}_t",
            r"=",
            r"\mathbf{W}_{xh}\mathbf{x}_t",
            r"+",
            r"\mathbf{W}_{hh}\mathbf{h}_{t-1}",
            r"+",
            r"\mathbf{b}_h",
            font_size=FONT_SIZE_BODY,
        ).to_edge(UP, buff=1.25)
        # Initially display in muted color before computation
        eq_sum.set_color(MUTED)

        # Central Cell Box
        cell_box = LabeledBox("Recurrent Cell", width=2.4, height=1.5, stroke_color=PRIMARY)
        cell_box.shift(DOWN * 0.25)

        # Input Node (bottom) and Past State Node (left)
        node_x = Node(r"$\mathbf{x}_t$", radius=0.42, stroke_color=INPUT).next_to(cell_box, DOWN, buff=1.1)
        node_h_prev = Node(r"$\mathbf{h}_{t-1}$", radius=0.42, stroke_color=STATE).next_to(cell_box, LEFT, buff=1.3)

        # Connectors representing parameter projections
        conn_x = Connector(node_x, cell_box, label=r"$\mathbf{W}_{xh}$", color=PARAM, label_direction=RIGHT)
        conn_h = Connector(node_h_prev, cell_box, label=r"$\mathbf{W}_{hh}$", color=PARAM, label_direction=UP)

        self.play(FadeIn(eq_sum), FadeIn(cell_box), run_time=1.2)
        self.play(FadeIn(node_x), FadeIn(node_h_prev), Create(conn_x), Create(conn_h), run_time=1.4)

        # Live evaluation 1: W_xh x_t activates
        self.play(
            Indicate(conn_x, color=INPUT, scale_factor=1.1),
            eq_sum[2].animate.set_color(INPUT),
            run_time=1.0,
        )
        # Live evaluation 2: W_hh h_{t-1} activates
        self.play(
            Indicate(conn_h, color=STATE, scale_factor=1.1),
            eq_sum[4].animate.set_color(STATE),
            run_time=1.0,
        )
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 2 (0:12 - 0:25 | 13s): Linear Superposition
        # -------------------------------------------------------------
        self.show_caption("Transformed vectors combine via linear superposition with a bias.")

        self.play(
            FadeOut(cell_box),
            FadeOut(node_x),
            FadeOut(node_h_prev),
            FadeOut(conn_x),
            FadeOut(conn_h),
            run_time=0.8,
        )

        # 2D Vector Parallelogram showing head-to-tail linear combination
        v_diag = VectorParallelogram(
            u=[2.2, 0.4, 0],
            v=[0.7, 1.6, 0],
            origin=[-1.4, -1.2, 0],
            u_color=INPUT,
            v_color=STATE,
            sum_color=ACCENT,
            u_label=r"\mathbf{W}_{xh}\mathbf{x}_t",
            v_label=r"\mathbf{W}_{hh}\mathbf{h}_{t-1}",
            sum_label=r"\mathbf{a}_t",
        )

        # Complete equation term illumination: a_t, +, b_h
        self.play(
            Create(v_diag),
            eq_sum[0].animate.set_color(ACCENT),
            eq_sum[1].animate.set_color(WHITE),
            eq_sum[3].animate.set_color(WHITE),
            eq_sum[5].animate.set_color(WHITE),
            eq_sum[6].animate.set_color(PARAM),
            run_time=2.0,
        )
        self.wait(7.0)

        # -------------------------------------------------------------
        # BEAT 3 (0:25 - 0:37 | 12s): The Hyperbolic Squashing
        # -------------------------------------------------------------
        self.show_caption("Hyperbolic tangent squashes pre-activations into [-1, 1].")

        eq_squash = MathTex(
            r"\mathbf{h}_t",
            r"=",
            r"\tanh(",
            r"\mathbf{a}_t",
            r")",
            font_size=FONT_SIZE_BODY,
        ).next_to(eq_sum, DOWN, buff=0.4)
        eq_squash[0].set_color(STATE)
        eq_squash[3].set_color(ACCENT)

        bound_note = Text("Coordinates strictly bounded in [-1, 1]", font_size=FONT_SIZE_LABEL, color=SECONDARY)
        bound_note.next_to(eq_squash, DOWN, buff=0.3)

        self.play(FadeIn(eq_squash, shift=DOWN * 0.1), FadeIn(bound_note), run_time=1.4)
        self.play(Indicate(bound_note, color=SECONDARY, scale_factor=1.05), run_time=1.0)
        self.wait(6.0)

        # -------------------------------------------------------------
        # BEAT 4 (0:37 - 0:48 | 11s): Readout & State Emission
        # -------------------------------------------------------------
        self.show_caption("The updated state emits a prediction and passes to the next step.")

        self.play(
            FadeOut(v_diag),
            FadeOut(eq_sum),
            FadeOut(eq_squash),
            FadeOut(bound_note),
            run_time=1.0,
        )

        # Re-introduce central cell with updated state label
        cell_active = LabeledBox(r"$\mathbf{h}_t = \tanh(\mathbf{a}_t)$", width=3.0, height=1.4, stroke_color=PRIMARY)
        cell_active.shift(DOWN * 0.1)

        node_y = Node(r"$\mathbf{\hat{y}}_t$", radius=0.45, stroke_color=OUTPUT).next_to(cell_active, UP, buff=1.2)
        node_next = Node(r"$\mathbf{h}_t$", radius=0.45, stroke_color=STATE).next_to(cell_active, RIGHT, buff=1.4)

        conn_y = Connector(cell_active, node_y, label=r"$\mathbf{W}_{hy}$", color=PARAM, label_direction=RIGHT)
        conn_next = Connector(cell_active, node_next, color=STATE)

        readout_eq = MathTex(
            r"\mathbf{\hat{y}}_t = g(\mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y)",
            font_size=FONT_SIZE_BODY,
            color=OUTPUT,
        ).next_to(cell_active, DOWN, buff=0.9)

        self.play(FadeIn(cell_active), run_time=1.0)
        self.play(FadeIn(node_y), FadeIn(node_next), Create(conn_y), Create(conn_next), run_time=1.4)
        self.play(FadeIn(readout_eq), run_time=1.0)
        self.wait(5.0)

        self.clear_caption()
        self.wait(0.5)
