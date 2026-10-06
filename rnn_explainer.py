"""
Recurrent Neural Networks (RNN) Explainer Video - Full Master Script
Combines all 15 scenes into a unified, standalone production animation.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure local project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manim import (
    config,
    Arrow,
    Axes,
    Brace,
    Circle,
    Create,
    DashedLine,
    Dot,
    DoubleArrow,
    DOWN,
    Ellipse,
    FadeIn,
    FadeOut,
    Indicate,
    LEFT,
    Line,
    MathTex,
    ORIGIN,
    Rectangle,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Text,
    Transform,
    UP,
    UR,
    VGroup,
    WHITE,
)

# Prevent cache trimming during multi-scene full rendering
config.max_files_cached = 10000

from lib.video_scene import VideoScene
from lib.components import (
    BoxChain,
    CaptionBar,
    Connector,
    LabeledBox,
    Node,
    VectorParallelogram,
    RecurrentLoopCell,
)
from lib.palette import (
    ACCENT,
    BACKGROUND,
    FONT_SIZE_BODY,
    FONT_SIZE_LABEL,
    FONT_SIZE_TITLE,
    INPUT,
    LOSS,
    MUTED,
    OUTPUT,
    PARAM,
    PRIMARY,
    SECONDARY,
    STATE,
    WARNING,
)



# ============================================================================
# scene01_why_rnns_exist.py
# ============================================================================

class Scene01WhyRNNsExist(VideoScene):
    def construct(self) -> None:
        title = Text("Why RNNs Exist: The Sequence Challenge", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:10 | 10s): The Fixed 6-Token Interface
        # -------------------------------------------------------------
        self.show_caption("Feedforward networks require fixed-dimension inputs.")

        words_6 = ["The", "cat", "sat", "on", "the", "mat"]
        tokens_6 = VGroup(*[
            VGroup(
                LabeledBox(w, width=1.1, height=0.48, stroke_color=INPUT, font_size=FONT_SIZE_LABEL - 4),
                MathTex(rf"\mathbf{{x}}_{i+1}", font_size=FONT_SIZE_LABEL, color=INPUT),
            ).arrange(DOWN, buff=0.08)
            for i, w in enumerate(words_6)
        ]).arrange(RIGHT, buff=0.18).shift(UP * 1.15)

        brace_6 = Brace(tokens_6, UP, buff=0.12)
        label_6 = brace_6.get_tex(r"6d_x \text{ input vector}")

        ff_interface = LabeledBox(
            r"$\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times 6d_x}$",
            width=7.5,
            height=0.75,
            stroke_color=PRIMARY,
            font_size=FONT_SIZE_BODY,
        ).shift(DOWN * 0.45)

        conns_6 = VGroup(*[
            Connector(tokens_6[i], ff_interface, color=MUTED, buff=0.08)
            for i in range(6)
        ])

        self.play(FadeIn(tokens_6, shift=DOWN * 0.15), run_time=1.2)
        self.play(Create(brace_6), FadeIn(label_6), Create(ff_interface), run_time=1.2)
        self.play(Create(conns_6), run_time=1.0)
        self.wait(5.0)

        # -------------------------------------------------------------
        # BEAT 2 (0:10 - 0:21 | 11s): The Geometric Dimensional Mismatch
        # -------------------------------------------------------------
        self.show_caption("Extend to eight tokens, and the geometry breaks down: 8d_x cannot fit 6d_x.")

        # 8-token sentence: "The cat sat on the mat yesterday afternoon"
        words_8 = ["The", "cat", "sat", "on", "the", "mat", "yes-", "terday"]
        tokens_8 = VGroup(*[
            VGroup(
                LabeledBox(
                    w,
                    width=1.1,
                    height=0.48,
                    stroke_color=WARNING if i >= 6 else INPUT,
                    font_size=FONT_SIZE_LABEL - 4,
                ),
                MathTex(
                    rf"\mathbf{{x}}_{i+1}",
                    font_size=FONT_SIZE_LABEL,
                    color=WARNING if i >= 6 else INPUT,
                ),
            ).arrange(DOWN, buff=0.08)
            for i, w in enumerate(words_8)
        ]).arrange(RIGHT, buff=0.18)

        # Align left edge with tokens_6 so tokens 7 & 8 physically overhang the 6d_x interface
        tokens_8.align_to(tokens_6, LEFT).align_to(tokens_6, UP)

        brace_8 = Brace(tokens_8, UP, buff=0.12)
        label_8 = brace_8.get_tex(r"8d_x \text{ input sequence}")

        # Vertical demarcation line marking the 6d_x boundary
        interface_edge_x = ff_interface.get_right()[0]
        divider = DashedLine(
            start=[interface_edge_x, 2.2, 0],
            end=[interface_edge_x, -0.95, 0],
            color=WARNING,
            stroke_width=2.5,
        )

        mismatch_note = MathTex(
            r"8d_x > 6d_x \implies \text{Overhangs Matrix Interface!}",
            font_size=FONT_SIZE_BODY,
            color=WARNING,
        ).next_to(ff_interface, DOWN, buff=0.45)

        self.play(
            FadeOut(conns_6),
            FadeOut(brace_6),
            FadeOut(label_6),
            ReplacementTransform(tokens_6, tokens_8),
            run_time=1.2,
        )
        self.play(Create(brace_8), FadeIn(label_8), Create(divider), run_time=1.2)
        self.play(FadeIn(mismatch_note), Indicate(tokens_8[6:], color=WARNING, scale_factor=1.1), run_time=1.2)
        self.wait(5.0)

        # -------------------------------------------------------------
        # BEAT 3 (0:21 - 0:32 | 11s): Visual Question & Recurrent Compression
        # -------------------------------------------------------------
        self.show_caption("Instead of expanding matrices, we process sequentially while carrying state.")

        question = Text(
            "How can one fixed model process sequences of any length?",
            font_size=FONT_SIZE_BODY,
            color=SECONDARY,
        ).shift(UP * 1.5)

        recurrent_eq = MathTex(
            r"\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1})",
            font_size=36,
            color=WHITE,
        ).to_edge(UP, buff=1.2)

        rnn_cell = LabeledBox(
            r"$\text{RNN}$",
            width=2.0,
            height=1.3,
            stroke_color=PRIMARY,
            font_size=FONT_SIZE_BODY,
        ).move_to(ORIGIN)

        state_node = Node(r"$\mathbf{h}_0$", radius=0.42, stroke_color=STATE).next_to(rnn_cell, RIGHT, buff=1.5)
        conn_state = Connector(rnn_cell, state_node, color=STATE)

        self.play(
            FadeOut(tokens_8),
            FadeOut(brace_8),
            FadeOut(label_8),
            FadeOut(ff_interface),
            FadeOut(divider),
            FadeOut(mismatch_note),
            FadeIn(question, shift=DOWN * 0.2),
            run_time=1.2,
        )
        self.wait(2.0)

        self.play(
            ReplacementTransform(question, recurrent_eq),
            FadeIn(rnn_cell),
            FadeIn(state_node),
            Create(conn_state),
            run_time=1.5,
        )
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 4 (0:32 - 0:42 | 10s): In-Place Sequential Ingestion
        # -------------------------------------------------------------
        self.show_caption("The exact same cell remains in place, reusing the same computation across time.")

        node_x = Node(r"$\mathbf{x}_1$", radius=0.38, stroke_color=INPUT).next_to(rnn_cell, LEFT, buff=1.5)
        conn_in = Connector(node_x, rnn_cell, color=INPUT)

        # Step 1: x_1 arrives -> state h_1
        self.play(FadeIn(node_x), Create(conn_in), run_time=0.6)
        self.play(Indicate(rnn_cell, color=PRIMARY, scale_factor=1.08), run_time=0.5)
        state_h1 = Node(r"$\mathbf{h}_1$", radius=0.42, stroke_color=STATE).move_to(state_node.get_center())
        self.play(Transform(state_node, state_h1), run_time=0.5)

        # Step 2: x_2 arrives -> state h_2
        node_x2 = Node(r"$\mathbf{x}_2$", radius=0.38, stroke_color=INPUT).move_to(node_x.get_center())
        self.play(Transform(node_x, node_x2), run_time=0.5)
        self.play(Indicate(rnn_cell, color=PRIMARY, scale_factor=1.08), run_time=0.5)
        state_h2 = Node(r"$\mathbf{h}_2$", radius=0.42, stroke_color=STATE).move_to(state_node.get_center())
        self.play(Transform(state_node, state_h2), run_time=0.5)

        # Step 3: x_3 arrives -> state h_3
        node_x3 = Node(r"$\mathbf{x}_3$", radius=0.38, stroke_color=INPUT).move_to(node_x.get_center())
        self.play(Transform(node_x, node_x3), run_time=0.5)
        self.play(Indicate(rnn_cell, color=PRIMARY, scale_factor=1.08), run_time=0.5)
        state_h3 = Node(r"$\mathbf{h}_3$", radius=0.42, stroke_color=STATE).move_to(state_node.get_center())
        self.play(Transform(state_node, state_h3), run_time=0.5)
        self.wait(2.5)

        self.clear_caption()
        self.wait(0.5)


# ============================================================================
# scene02_the_rnn_cell.py
# ============================================================================

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
        cell_box.shift(UP * 0.45)

        # Input Node (bottom) and Past State Node (left)
        node_x = Node(r"$\mathbf{x}_t$", radius=0.42, stroke_color=INPUT).next_to(cell_box, DOWN, buff=0.85)
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


# ============================================================================
# scene03_unrolling_dag.py
# ============================================================================

class Scene03UnrollingDAG(VideoScene):
    def construct(self) -> None:
        title = Text("Unrolling Through Time: Loop vs. DAG", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:10 | 10s): The Compact Feedback Loop
        # -------------------------------------------------------------
        self.show_caption("Recurrent networks are often drawn with a recursive feedback loop.")

        loop_cell = RecurrentLoopCell(
            label="A",
            loop_label=r"\mathbf{W}_{hh}",
            input_label=r"\mathbf{x}_t",
            output_label=r"\mathbf{h}_t",
        ).shift(DOWN * 0.45)

        self.play(FadeIn(loop_cell), run_time=1.5)
        self.wait(7.0)

        # -------------------------------------------------------------
        # BEAT 2 (0:10 - 0:21 | 11s): Snapping Open the Chain
        # -------------------------------------------------------------
        self.show_caption("Unspooling across time reveals a directed acyclic graph.")

        # Unrolled 3-step chain
        chain = BoxChain(
            items=["t = 1", "t = 2", "t = 3"],
            spacing=1.1,
            box_width=1.5,
            box_height=0.8,
            stroke_color=PRIMARY,
            connector_color=PARAM,
            connector_labels=[r"$\mathbf{W}_{hh}$", r"$\mathbf{W}_{hh}$"],
        ).shift(DOWN * 0.15)

        # Hidden state nodes below each timestep
        outputs = VGroup(*[
            Node(rf"$\mathbf{{h}}_{i}$", radius=0.35, stroke_color=STATE).next_to(chain.boxes[i - 1], DOWN, buff=0.75)
            for i in range(1, 4)
        ])
        out_conns = VGroup(*[
            Connector(outputs[i], chain.boxes[i], color=STATE)
            for i in range(3)
        ])

        self.play(
            ReplacementTransform(loop_cell, chain),
            run_time=1.8,
        )
        self.play(
            FadeIn(outputs),
            Create(out_conns),
            run_time=1.5,
        )
        self.wait(6.0)

        # -------------------------------------------------------------
        # BEAT 3 (0:21 - 0:31 | 10s): Physical Parameter Sharing
        # -------------------------------------------------------------
        self.show_caption("Crucially, one physical parameter matrix is shared across all timesteps.")

        # Master persistent parameter badge positioned at top
        w_master = LabeledBox(
            r"$\mathbf{W}_{hh} \text{ (Shared Parameter)}$",
            width=3.6,
            height=0.55,
            stroke_color=PARAM,
            fill_color=PARAM,
            fill_opacity=0.22,
            font_size=FONT_SIZE_LABEL,
        ).shift(UP * 1.8)

        # Distribution arrows feeding from master W_hh directly into each timestep cell's top edge
        w_dist_conns = VGroup(*[
            Connector(w_master, chain.boxes[i], color=PARAM, stroke_width=2.0)
            for i in range(3)
        ])

        self.play(FadeIn(w_master, shift=DOWN * 0.15), run_time=1.0)
        self.play(Create(w_dist_conns), run_time=1.5)
        self.play(Indicate(w_master, color=PARAM, scale_factor=1.08), run_time=0.8)
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 4 (0:31 - 0:40 | 9s): Travelling Forward Signal
        # -------------------------------------------------------------
        self.show_caption("Information travels sequentially: each state becomes the context for the next.")

        # Clean distribution lines to emphasize temporal flow
        self.play(FadeOut(w_dist_conns), run_time=0.6)

        # Step 1 execution: h_1 -> Box 1 -> Box 2
        self.play(Indicate(outputs[0], color=STATE), Indicate(chain.boxes[0], color=PRIMARY), run_time=0.6)
        self.play(Indicate(chain.connectors[0], color=PARAM), run_time=0.6)

        # Step 2 execution: h_2 -> Box 2 -> Box 3
        self.play(Indicate(outputs[1], color=STATE), Indicate(chain.boxes[1], color=PRIMARY), run_time=0.6)
        self.play(Indicate(chain.connectors[1], color=PARAM), run_time=0.6)

        # Step 3 execution: h_3 -> Box 3
        self.play(Indicate(outputs[2], color=STATE), Indicate(chain.boxes[2], color=PRIMARY), run_time=0.6)
        self.wait(3.0)

        self.clear_caption()
        self.wait(0.5)


# ============================================================================
# scene04_forward_pass.py
# ============================================================================

class Scene04ForwardPass(VideoScene):
    def construct(self) -> None:
        title = Text("The Forward Pass: Sequential Computation", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:11 | 11s): Initial State and Timestep 1
        # -------------------------------------------------------------
        self.show_caption("Information travels sequentially: Timestep 1 computes the first state.")

        # 3-step unrolled chain
        chain = BoxChain(
            items=["t = 1", "t = 2", "t = 3"],
            spacing=1.3,
            box_width=1.5,
            box_height=0.8,
            stroke_color=MUTED,
            connector_color=MUTED,
            connector_labels=[r"$\mathbf{W}_{hh}$", r"$\mathbf{W}_{hh}$"],
        ).shift(DOWN * 0.65)

        # Input nodes below
        inputs = VGroup(*[
            Node(rf"$\mathbf{{x}}_{i}$", radius=0.3, stroke_color=MUTED).next_to(chain.boxes[i - 1], DOWN, buff=0.55)
            for i in range(1, 4)
        ])
        in_conns = VGroup(*[
            Connector(inputs[i], chain.boxes[i], color=MUTED)
            for i in range(3)
        ])

        # Prediction output nodes above
        outputs = VGroup(*[
            Node(rf"$\mathbf{{\hat{{y}}}}_{i}$", radius=0.3, stroke_color=MUTED).next_to(chain.boxes[i - 1], UP, buff=0.55)
            for i in range(1, 4)
        ])
        out_conns = VGroup(*[
            Connector(chain.boxes[i], outputs[i], color=MUTED)
            for i in range(3)
        ])

        self.play(FadeIn(chain), FadeIn(inputs), Create(in_conns), FadeIn(outputs), Create(out_conns), run_time=1.4)

        # Timestep 1 activates
        self.play(
            inputs[0].circle.animate.set_stroke(INPUT, width=3).set_fill(INPUT, opacity=0.85),
            inputs[0].label.animate.set_color(WHITE),
            in_conns[0].animate.set_color(INPUT),
            chain.boxes[0].box.animate.set_stroke(PRIMARY, width=3).set_fill(PRIMARY, opacity=0.35),
            chain.boxes[0].label.animate.set_color(WHITE),
            run_time=0.8,
        )
        self.play(Indicate(chain.boxes[0], color=PRIMARY, scale_factor=1.08), run_time=0.5)
        self.play(
            out_conns[0].animate.set_color(OUTPUT),
            outputs[0].circle.animate.set_stroke(OUTPUT, width=3).set_fill(OUTPUT, opacity=0.85),
            outputs[0].label.animate.set_color(WHITE),
            run_time=0.8,
        )
        # State h_1 propagates forward along the connector
        self.play(
            chain.connectors[0].animate.set_color(STATE),
            run_time=0.8,
        )
        self.wait(3.0)

        # -------------------------------------------------------------
        # BEAT 2 (0:11 - 0:22 | 11s): Timestep 2 Combines History and Input
        # -------------------------------------------------------------
        self.show_caption("At Timestep 2, fresh input merges with past context h_1.")

        self.play(
            inputs[1].circle.animate.set_stroke(INPUT, width=3).set_fill(INPUT, opacity=0.85),
            inputs[1].label.animate.set_color(WHITE),
            in_conns[1].animate.set_color(INPUT),
            chain.boxes[1].box.animate.set_stroke(PRIMARY, width=3).set_fill(PRIMARY, opacity=0.35),
            chain.boxes[1].label.animate.set_color(WHITE),
            run_time=0.8,
        )
        self.play(Indicate(chain.boxes[1], color=PRIMARY, scale_factor=1.08), run_time=0.5)
        self.play(
            out_conns[1].animate.set_color(OUTPUT),
            outputs[1].circle.animate.set_stroke(OUTPUT, width=3).set_fill(OUTPUT, opacity=0.85),
            outputs[1].label.animate.set_color(WHITE),
            run_time=0.8,
        )
        # State h_2 propagates forward
        self.play(
            chain.connectors[1].animate.set_color(STATE),
            run_time=0.8,
        )
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 3 (0:22 - 0:33 | 11s): Timestep 3 and State Caching
        # -------------------------------------------------------------
        self.show_caption("At Timestep 3, the final prediction emits, and all states are cached.")

        self.play(
            inputs[2].circle.animate.set_stroke(INPUT, width=3).set_fill(INPUT, opacity=0.85),
            inputs[2].label.animate.set_color(WHITE),
            in_conns[2].animate.set_color(INPUT),
            chain.boxes[2].box.animate.set_stroke(PRIMARY, width=3).set_fill(PRIMARY, opacity=0.35),
            chain.boxes[2].label.animate.set_color(WHITE),
            run_time=0.8,
        )
        self.play(Indicate(chain.boxes[2], color=PRIMARY, scale_factor=1.08), run_time=0.5)
        self.play(
            out_conns[2].animate.set_color(OUTPUT),
            outputs[2].circle.animate.set_stroke(OUTPUT, width=3).set_fill(OUTPUT, opacity=0.85),
            outputs[2].label.animate.set_color(WHITE),
            run_time=0.8,
        )

        # Forward cache banner above outputs (high contrast)
        cache_badge = LabeledBox(
            r"$\text{Forward Cache: } \{\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_3\} \text{ preserved in memory}$",
            width=5.8,
            height=0.6,
            stroke_color=STATE,
            stroke_width=2.5,
            fill_color=BACKGROUND,
            fill_opacity=0.95,
            text_color=WHITE,
            font_size=FONT_SIZE_LABEL,
        ).next_to(outputs, UP, buff=0.45)

        self.play(FadeIn(cache_badge, shift=DOWN * 0.15), run_time=1.0)
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 4 (0:33 - 0:43 | 10s): Loss Accumulation Across Time
        # -------------------------------------------------------------
        self.show_caption("Predictions are compared against targets, accumulating total loss.")

        # Loss nodes above each prediction (crisp white label on loss color)
        loss_nodes = VGroup(*[
            Node(
                rf"$\mathcal{{L}}_{i}$",
                radius=0.28,
                stroke_color=LOSS,
                fill_color=LOSS,
                fill_opacity=0.85,
                text_color=WHITE,
            ).next_to(outputs[i - 1], UP, buff=0.45)
            for i in range(1, 4)
        ])
        loss_conns = VGroup(*[
            Connector(outputs[i], loss_nodes[i], color=LOSS)
            for i in range(3)
        ])

        total_loss_eq = MathTex(
            r"\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} \mathcal{L}_t(\mathbf{\hat{y}}_t, \mathbf{y}_t)",
            font_size=FONT_SIZE_LABEL + 2,
            color=WHITE,
        ).next_to(loss_nodes, UP, buff=0.25)

        self.play(
            FadeOut(cache_badge),
            FadeIn(loss_nodes),
            Create(loss_conns),
            FadeIn(total_loss_eq, shift=DOWN * 0.15),
            run_time=1.5,
        )
        self.wait(4.0)

        self.clear_caption()
        self.wait(0.5)


# ============================================================================
# scene05_bptt_flow.py
# ============================================================================

class Scene05BPTTFlow(VideoScene):
    def construct(self) -> None:
        title = Text("Backpropagation Through Time: Gradient Flow", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:12 | 12s): The Forward Graph & Loss Injection
        # -------------------------------------------------------------
        self.show_caption("Training reverses the flow: Loss at step 3 injects an error gradient.")

        # The identical unrolled forward graph from Scene 04
        chain = BoxChain(
            items=["t = 1", "t = 2", "t = 3"],
            spacing=1.3,
            box_width=1.5,
            box_height=0.8,
            stroke_color=PRIMARY,
            connector_color=STATE,
            connector_labels=[r"$\mathbf{W}_{hh}$", r"$\mathbf{W}_{hh}$"],
        ).shift(DOWN * 0.65)

        inputs = VGroup(*[
            Node(rf"$\mathbf{{x}}_{i}$", radius=0.3, stroke_color=INPUT).next_to(chain.boxes[i - 1], DOWN, buff=0.55)
            for i in range(1, 4)
        ])
        in_conns = VGroup(*[
            Connector(inputs[i], chain.boxes[i], color=INPUT)
            for i in range(3)
        ])

        outputs = VGroup(*[
            Node(rf"$\mathbf{{\hat{{y}}}}_{i}$", radius=0.3, stroke_color=OUTPUT).next_to(chain.boxes[i - 1], UP, buff=0.55)
            for i in range(1, 4)
        ])
        out_conns = VGroup(*[
            Connector(chain.boxes[i], outputs[i], color=OUTPUT)
            for i in range(3)
        ])

        # Loss node at output 3
        loss_node = Node(r"$\mathcal{L}$", radius=0.32, stroke_color=LOSS, text_color=LOSS).next_to(outputs[2], UP, buff=0.5)
        loss_conn = Connector(outputs[2], loss_node, color=LOSS)

        self.play(
            FadeIn(chain),
            FadeIn(inputs),
            Create(in_conns),
            FadeIn(outputs),
            Create(out_conns),
            FadeIn(loss_node),
            Create(loss_conn),
            run_time=1.5,
        )

        # Initial gradient injection at h_3
        grad_3_label = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial \mathbf{h}_3}",
            font_size=FONT_SIZE_BODY,
            color=WARNING,
        ).next_to(chain.boxes[2], UP, buff=0.95).shift(RIGHT * 0.9)

        self.play(Indicate(loss_node, color=LOSS), run_time=0.6)
        self.play(
            Indicate(chain.boxes[2], color=WARNING),
            FadeIn(grad_3_label, shift=DOWN * 0.1),
            run_time=1.0,
        )
        self.wait(5.0)

        # -------------------------------------------------------------
        # BEAT 2 (0:12 - 0:24 | 12s): Reverse Traversal Across Transition
        # -------------------------------------------------------------
        self.show_caption("The gradient flows backward into step 2, transformed by W_hh transpose.")

        # Backward gradient arrow between Box 3 and Box 2
        back_arrow_32 = Connector(
            chain.boxes[2],
            chain.boxes[1],
            label=r"$\mathbf{W}_{hh}^\top$",
            color=WARNING,
            label_direction=DOWN,
        )

        grad_2_label = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial \mathbf{h}_2} = \mathbf{W}_{hh}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{h}_3}",
            font_size=FONT_SIZE_LABEL + 2,
            color=WARNING,
        ).to_edge(UP, buff=1.2)

        self.play(Create(back_arrow_32), run_time=1.2)
        self.play(Indicate(chain.boxes[1], color=WARNING), FadeIn(grad_2_label), run_time=1.0)
        self.wait(5.0)

        # -------------------------------------------------------------
        # BEAT 3 (0:24 - 0:35 | 11s): Gradient Reaches Earliest Context
        # -------------------------------------------------------------
        self.show_caption("Traveling back to step 1, the gradient multiplies by W_hh transpose again.")

        back_arrow_21 = Connector(
            chain.boxes[1],
            chain.boxes[0],
            label=r"$\mathbf{W}_{hh}^\top$",
            color=WARNING,
            label_direction=DOWN,
        )

        grad_1_label = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = (\mathbf{W}_{hh}^\top)^2 \frac{\partial \mathcal{L}}{\partial \mathbf{h}_3}",
            font_size=FONT_SIZE_LABEL + 2,
            color=WARNING,
        ).next_to(grad_2_label, DOWN, buff=0.3)

        self.play(Create(back_arrow_21), run_time=1.2)
        self.play(Indicate(chain.boxes[0], color=WARNING), FadeIn(grad_1_label), run_time=1.0)
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 4 (0:35 - 0:45 | 10s): Parameter Gradient Accumulation
        # -------------------------------------------------------------
        self.show_caption("Weight sharing implies summing contributions across all backward steps.")

        accum_eq = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top",
            font_size=FONT_SIZE_BODY,
            color=WHITE,
        ).to_edge(UP, buff=1.25)

        self.play(
            FadeOut(grad_2_label),
            FadeOut(grad_1_label),
            FadeOut(grad_3_label),
            FadeIn(accum_eq, shift=DOWN * 0.15),
            run_time=1.4,
        )
        self.play(Indicate(accum_eq, color=PARAM, scale_factor=1.05), run_time=1.0)
        self.wait(4.0)

        self.clear_caption()
        self.wait(0.5)


# ============================================================================
# scene06_vanishing_gradients.py
# ============================================================================

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


# ============================================================================
# scene07_exploding_gradients.py
# ============================================================================

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
        ceiling_lbl = Text("Max Capacity", font_size=FONT_SIZE_LABEL - 4, color=LOSS).next_to(ceiling_line, LEFT, buff=0.15)

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


# ============================================================================
# scene08_long_term_dependencies.py
# ============================================================================

class Scene08LongTermDependencies(VideoScene):
    def construct(self) -> None:
        title = Text("Long-Term Dependencies: The Optimization Barrier", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:12 | 12s): The Long Sequence Experiment
        # -------------------------------------------------------------
        self.show_caption("Predicting future tokens often requires information from distant past timesteps.")

        # 5 timesteps representing a sequence
        x_positions = [-4.4, -2.2, 0.0, 2.2, 4.4]
        words = ["clouds", "were", "dark", "because", "rained"]
        colors = [ACCENT, MUTED, MUTED, MUTED, PRIMARY]

        tokens = VGroup()
        for i, (x_pos, word, col) in enumerate(zip(x_positions, words, colors)):
            label_text = f'"{word}"' if i != 4 else r"$\mathbf{y}^*:\text{rained}$"
            box = LabeledBox(
                label_text,
                width=1.7,
                height=0.55,
                stroke_color=col,
                fill_color=col,
                fill_opacity=0.2,
                font_size=FONT_SIZE_LABEL - 4,
            ).move_to([x_pos, 1.2, 0])
            tokens.add(box)

        # Gap indicator
        gap_brace = Brace(tokens, DOWN, buff=0.15)
        gap_label = MathTex(
            r"\text{Long Temporal Distance: } \Delta t = T - 1",
            font_size=FONT_SIZE_LABEL,
            color=WARNING,
        ).next_to(gap_brace, DOWN, buff=0.12)

        self.play(FadeIn(tokens, shift=DOWN * 0.15), run_time=1.2)
        self.play(Create(gap_brace), FadeIn(gap_label), run_time=1.0)
        self.play(Indicate(tokens[0], color=ACCENT, scale_factor=1.05), Indicate(tokens[4], color=PRIMARY, scale_factor=1.05), run_time=1.0)
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 2 (0:12 - 0:24 | 12s): Forward Information Dilution
        # -------------------------------------------------------------
        self.show_caption("With each transition, new evidence compresses and dilutes early memory.")

        self.play(
            FadeOut(gap_brace),
            FadeOut(gap_label),
            tokens.animate.shift(UP * 0.6),
            run_time=0.8,
        )

        # Unrolled hidden state cells
        state_nodes = VGroup()
        state_labels = [r"$\mathbf{h}_1$", r"$\mathbf{h}_2$", r"$\mathbf{h}_3$", r"$\mathbf{h}_4$", r"$\mathbf{h}_T$"]
        # Progressive color fading: rich cyan (ACCENT) fades into MUTED gray
        fade_colors = [ACCENT, "#38bdf8", "#0284c7", "#475569", MUTED]

        for x_pos, lbl, col in zip(x_positions, state_labels, fade_colors):
            node = Node(lbl, radius=0.42, stroke_color=col, fill_color=col, fill_opacity=0.25).move_to([x_pos, 0.1, 0])
            state_nodes.add(node)

        # Forward transition arrows
        fwd_arrows = VGroup()
        for i in range(len(x_positions) - 1):
            arr = Arrow(
                start=state_nodes[i].get_right(),
                end=state_nodes[i + 1].get_left(),
                buff=0.1,
                color=fade_colors[i],
                stroke_width=3.0,
                max_tip_length_to_length_ratio=0.25,
            )
            fwd_arrows.add(arr)

        # Input arrows dropping from words into state nodes
        in_arrows = VGroup()
        for i in range(len(x_positions)):
            arr = Arrow(
                start=tokens[i].get_bottom(),
                end=state_nodes[i].get_top(),
                buff=0.1,
                color=MUTED,
                stroke_width=2.0,
                max_tip_length_to_length_ratio=0.25,
            )
            in_arrows.add(arr)

        self.play(FadeIn(state_nodes), Create(fwd_arrows), Create(in_arrows), run_time=1.2)

        dilution_banner = MathTex(
            r"\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t) \implies \text{Information drift and dilution}",
            font_size=FONT_SIZE_LABEL,
            color=WARNING,
        ).shift(DOWN * 1.3)

        self.play(FadeIn(dilution_banner), run_time=0.8)
        self.play(Indicate(state_nodes[4], color=MUTED, scale_factor=1.1), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 3 (0:24 - 0:35 | 11s): The Severed Backward Gradient
        # -------------------------------------------------------------
        self.show_caption("Error at step T cannot send corrective gradients across 20 Jacobians.")

        self.play(FadeOut(dilution_banner), run_time=0.5)

        # Loss node above step T (connecting cleanly to prediction target y*)
        loss_node = Node(r"$\mathcal{L}_T$", radius=0.35, stroke_color=LOSS, fill_color=LOSS, fill_opacity=0.3).move_to([x_positions[4], 2.65, 0])
        loss_arrow = Arrow(
            start=loss_node.get_bottom(),
            end=tokens[4].get_top(),
            buff=0.08,
            color=LOSS,
            stroke_width=3.5,
            max_tip_length_to_length_ratio=0.25,
        )

        self.play(FadeIn(loss_node), Create(loss_arrow), in_arrows[4].animate.set_color(LOSS).set_stroke(width=3.5), run_time=0.8)

        # Backward shrinking pulses
        bwd_arrows = VGroup()
        bwd_widths = [4.5, 3.0, 1.5, 0.4]
        bwd_colors = [LOSS, WARNING, WARNING, MUTED]

        for i in range(3, -1, -1):
            arr = Arrow(
                start=state_nodes[i + 1].get_left(),
                end=state_nodes[i].get_right(),
                buff=0.1,
                color=bwd_colors[3 - i],
                stroke_width=bwd_widths[3 - i],
                max_tip_length_to_length_ratio=0.25 if bwd_widths[3 - i] > 1.0 else 0.1,
            ).shift(DOWN * 0.5)
            bwd_arrows.add(arr)

        bwd_lbl = MathTex(
            r"\text{Backward Error Propagation: } \prod_{k=2}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} \to 0",
            font_size=FONT_SIZE_LABEL,
            color=LOSS,
        ).shift(DOWN * 1.3)

        self.play(FadeIn(bwd_lbl), run_time=0.6)
        for arr in bwd_arrows:
            self.play(Create(arr), run_time=0.4)

        severed_tag = LabeledBox(
            r"$\text{Signal Extinguished: } \mathbf{h}_1 \text{ receives ZERO gradient update}$",
            width=6.5,
            height=0.5,
            stroke_color=LOSS,
            fill_color=LOSS,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL - 3,
        ).shift(DOWN * 2.0)

        self.play(FadeIn(severed_tag), Indicate(state_nodes[0], color=LOSS, scale_factor=1.1), run_time=0.9)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:35 - 0:46 | 11s): Bengio's Paradox (1994)
        # -------------------------------------------------------------
        self.show_caption("Bengio's Paradox: The stability required to retain memory makes it untrainable by gradients.")

        self.play(
            FadeOut(tokens),
            FadeOut(state_nodes),
            FadeOut(fwd_arrows),
            FadeOut(in_arrows),
            FadeOut(loss_node),
            FadeOut(loss_arrow),
            *[FadeOut(arr) for arr in bwd_arrows],
            FadeOut(bwd_lbl),
            FadeOut(severed_tag),
            run_time=0.6,
        )

        paradox_title = Text("Bengio's Impossibility Dilemma (1994)", font_size=FONT_SIZE_BODY, color=WARNING).shift(UP * 1.5)

        dilemma_card = VGroup(
            MathTex(r"\text{1. Storing information robustly requires contracting dynamics: } \|\mathbf{J}\| < 1", font_size=FONT_SIZE_LABEL, color=WHITE),
            MathTex(r"\text{2. Contracting dynamics force backward gradients to vanish: } \prod_{k=2}^T \mathbf{J}_k \to 0", font_size=FONT_SIZE_LABEL, color=LOSS),
            MathTex(r"\text{Conclusion: Vanilla RNNs cannot be trained on long-term dependencies.}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(paradox_title, DOWN, buff=0.45)

        self.play(FadeIn(paradox_title), FadeIn(dilemma_card), run_time=1.0)
        self.play(Indicate(dilemma_card[2], color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)


# ============================================================================
# scene09_scalar_rnn.py
# ============================================================================

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


# ============================================================================
# scene10_jacobian_matrix.py
# ============================================================================

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
        shape_lbl = MathTex(r"\text{Unit Sphere } (t = 0)", font_size=FONT_SIZE_LABEL - 2, color=ACCENT).next_to(shape_0, DOWN, buff=0.2)

        # Right side: SVD table
        svd_title = Text("Singular Values (Semi-Axes):", font_size=FONT_SIZE_LABEL, color=WHITE).shift(RIGHT * 2.8 + UP * 0.8)
        r0 = MathTex(r"\text{Step } 0: \quad \sigma_1 = 1.00, \quad \sigma_2 = 1.00", font_size=FONT_SIZE_LABEL - 2, color=ACCENT).next_to(svd_title, DOWN, buff=0.25)
        r1 = MathTex(r"\text{Step } 1: \quad \sigma_1 = 0.85, \quad \sigma_2 = 0.42", font_size=FONT_SIZE_LABEL - 2, color=PRIMARY).next_to(r0, DOWN, buff=0.2)
        r2 = MathTex(r"\text{Step } 2: \quad \sigma_1 = 0.58, \quad \sigma_2 = 0.15", font_size=FONT_SIZE_LABEL - 2, color=WARNING).next_to(r1, DOWN, buff=0.2)
        r3 = MathTex(r"\text{Step } 3: \quad \sigma_1 = 0.27, \quad \sigma_2 \approx 0.03 \to 0", font_size=FONT_SIZE_LABEL - 2, color=LOSS).next_to(r2, DOWN, buff=0.2)

        self.play(FadeIn(geom_title), Create(shape_0), FadeIn(shape_lbl), FadeIn(svd_title), FadeIn(r0), run_time=1.0)

        # Transformation 1: Squashed into rotated ellipse
        shape_1 = Ellipse(width=2.2, height=1.1, color=PRIMARY, stroke_width=2.5).rotate(np.deg2rad(30)).move_to(center)
        lbl_1 = MathTex(r"\text{1 Step: } \mathbf{J}_1 \text{ applied}", font_size=FONT_SIZE_LABEL - 2, color=PRIMARY).next_to(shape_1, DOWN, buff=0.2)
        self.play(ReplacementTransform(shape_0, shape_1), ReplacementTransform(shape_lbl, lbl_1), FadeIn(r1), run_time=0.8)

        # Transformation 2: More squashed
        shape_2 = Ellipse(width=1.5, height=0.4, color=WARNING, stroke_width=2.5).rotate(np.deg2rad(45)).move_to(center)
        lbl_2 = MathTex(r"\text{2 Steps: } \mathbf{J}_2 \mathbf{J}_1 \text{ applied}", font_size=FONT_SIZE_LABEL - 2, color=WARNING).next_to(shape_2, DOWN, buff=0.2)
        self.play(ReplacementTransform(shape_1, shape_2), ReplacementTransform(lbl_1, lbl_2), FadeIn(r2), run_time=0.8)

        # Transformation 3: Extremely flat (pancake line)
        shape_3 = Ellipse(width=0.7, height=0.08, color=LOSS, stroke_width=2.5).rotate(np.deg2rad(45)).move_to(center)
        lbl_3 = MathTex(r"\text{Volume} \to 0 \text{ (Vanishing)}", font_size=FONT_SIZE_LABEL - 2, color=LOSS).next_to(shape_3, DOWN, buff=0.2)
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


# ============================================================================
# scene11_state_space.py
# ============================================================================

class Scene11StateSpace(VideoScene):
    def construct(self) -> None:
        title = Text("State-Space Dynamics: Latent Trajectories", font_size=FONT_SIZE_TITLE)
        title.to_edge(UP, buff=0.45)
        self.add(title)

        # -------------------------------------------------------------
        # BEAT 1 (0:00 - 0:12 | 12s): The Dynamical System Lens
        # -------------------------------------------------------------
        self.show_caption("Viewed geometrically, an RNN is a discrete dynamical system on a compact state space.")

        # Coordinate axes for 2D latent state space
        axes = Axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            x_length=4.2,
            y_length=4.2,
            tips=True,
            axis_config={"color": MUTED, "stroke_width": 2},
        ).move_to(LEFT * 3.2 + DOWN * 0.05)

        # Latent axis labels
        axis_labels = axes.get_axis_labels(
            x_label=MathTex(r"h^{(1)}", font_size=FONT_SIZE_LABEL - 2, color=MUTED),
            y_label=MathTex(r"h^{(2)}", font_size=FONT_SIZE_LABEL - 2, color=MUTED),
        )

        # Bounding domain box [-1, 1]^2
        p_bl = axes.coords_to_point(-1.0, -1.0)
        p_tr = axes.coords_to_point(1.0, 1.0)
        domain_box = Rectangle(
            width=abs(p_tr[0] - p_bl[0]),
            height=abs(p_tr[1] - p_bl[1]),
            stroke_color=PARAM,
            stroke_width=1.5,
            stroke_opacity=0.6,
        ).move_to(axes.coords_to_point(0, 0))

        domain_lbl = MathTex(r"\text{Domain: } [-1, 1]^2", font_size=FONT_SIZE_LABEL - 2, color=PARAM).move_to(axes.coords_to_point(-0.55, 1.15))

        # Info card on the right
        card_title = Text("Discrete Dynamical System", font_size=FONT_SIZE_BODY, color=PRIMARY).shift(RIGHT * 2.8 + UP * 1.2)
        system_card = VGroup(
            MathTex(r"\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}) = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t)", font_size=FONT_SIZE_LABEL - 2, color=WHITE),
            MathTex(r"\text{State } \mathbf{h}_t \in \mathcal{H} \subset [-1, 1]^d", font_size=FONT_SIZE_LABEL - 2, color=PARAM),
            MathTex(r"\text{Inputs } \mathbf{x}_t \text{ act as external driving forces}", font_size=FONT_SIZE_LABEL - 2, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(card_title, DOWN, buff=0.35)

        self.play(Create(axes), FadeIn(axis_labels), Create(domain_box), FadeIn(domain_lbl), run_time=1.0)
        self.play(FadeIn(card_title), FadeIn(system_card), run_time=0.9)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 2 (0:12 - 0:25 | 13s): Tracing the Latent Trajectory
        # -------------------------------------------------------------
        self.show_caption("Each incoming sequence steers the hidden state along a continuous trajectory.")

        # Trajectory points in coords
        coords_list = [
            (0.0, 0.0),
            (0.6, 0.4),
            (0.2, 0.8),
            (-0.5, 0.5),
            (-0.6, -0.3),
        ]
        pts = [axes.coords_to_point(x, y) for x, y in coords_list]
        state_tags = [r"\mathbf{h}_0", r"\mathbf{h}_1", r"\mathbf{h}_2", r"\mathbf{h}_3", r"\mathbf{h}_4"]

        # Dot and trail
        dot = Dot(point=pts[0], radius=0.08, color=ACCENT)
        tag0 = MathTex(state_tags[0], font_size=FONT_SIZE_LABEL - 4, color=ACCENT).next_to(pts[0], DOWN + LEFT, buff=0.1)
        self.play(FadeIn(dot), FadeIn(tag0), run_time=0.6)

        trail_segments = VGroup()
        state_labels_group = VGroup(tag0)

        for i in range(1, len(pts)):
            seg = Line(start=pts[i - 1], end=pts[i], color=ACCENT, stroke_width=3.0)
            tag = MathTex(state_tags[i], font_size=FONT_SIZE_LABEL - 4, color=ACCENT).next_to(pts[i], UP if pts[i][1] >= pts[i - 1][1] else DOWN, buff=0.1)
            trail_segments.add(seg)
            state_labels_group.add(tag)
            self.play(
                Create(seg),
                dot.animate.move_to(pts[i]),
                FadeIn(tag),
                run_time=0.6,
            )

        self.play(Indicate(dot, color=WARNING, scale_factor=1.2), run_time=0.7)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 3 (0:25 - 0:38 | 13s): Contractive Basin & Information Collapse
        # -------------------------------------------------------------
        self.show_caption("Contractive dynamics pull different historical paths into the same attractor point.")

        # Perturbed trajectory B starting from (0.2, -0.4)
        coords_b = [
            (0.2, -0.4),
            (0.5, -0.1),
            (0.0, 0.3),
            (-0.4, 0.1),
            (-0.6, -0.3),  # Merges into h4!
        ]
        pts_b = [axes.coords_to_point(x, y) for x, y in coords_b]

        dot_b = Dot(point=pts_b[0], radius=0.08, color=WARNING)
        tag_b0 = MathTex(r"\mathbf{h}_0'", font_size=FONT_SIZE_LABEL - 4, color=WARNING).next_to(pts_b[0], DOWN + RIGHT, buff=0.1)

        self.play(FadeIn(dot_b), FadeIn(tag_b0), run_time=0.6)

        trail_b = VGroup()
        for i in range(1, len(pts_b)):
            seg = Line(start=pts_b[i - 1], end=pts_b[i], color=WARNING, stroke_width=2.5)
            trail_b.add(seg)
            self.play(Create(seg), dot_b.animate.move_to(pts_b[i]), run_time=0.5)

        # Collision banner
        collision_tag = LabeledBox(
            r"$\text{Trajectory Collision: Distinct histories merge into } \mathbf{h}^*$",
            width=5.8,
            height=0.5,
            stroke_color=LOSS,
            fill_color=LOSS,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL - 4,
        ).next_to(domain_box, DOWN, buff=0.25).align_to(domain_box, LEFT)

        self.play(FadeIn(collision_tag), run_time=0.8)
        self.play(Indicate(dot, color=LOSS, scale_factor=1.3), Indicate(dot_b, color=LOSS, scale_factor=1.3), run_time=0.8)
        self.wait(3.5)

        # -------------------------------------------------------------
        # BEAT 4 (0:38 - 0:50 | 12s): The Need for Architectural Gating
        # -------------------------------------------------------------
        self.show_caption("To prevent information collapse and chaotic explosion, we need gated linear highways.")

        self.play(
            FadeOut(axes),
            FadeOut(axis_labels),
            FadeOut(domain_box),
            FadeOut(domain_lbl),
            FadeOut(card_title),
            FadeOut(system_card),
            FadeOut(dot),
            FadeOut(dot_b),
            *[FadeOut(seg) for seg in trail_segments],
            *[FadeOut(seg) for seg in trail_b],
            *[FadeOut(tag) for tag in state_labels_group],
            FadeOut(tag_b0),
            FadeOut(collision_tag),
            run_time=0.6,
        )

        dilemma_title = Text("The Dynamical Dilemma & The Gating Solution", font_size=FONT_SIZE_BODY, color=WARNING).shift(UP * 1.5)

        summary_card = VGroup(
            MathTex(r"\mathbf{1.\ \text{Contracting Dynamics: Trajectories merge } \implies \text{Past information is erased.}}", font_size=FONT_SIZE_LABEL, color=MUTED),
            MathTex(r"\mathbf{2.\ \text{Expanding Dynamics: Trajectories diverge } \implies \text{Chaos and numerical explosion.}}", font_size=FONT_SIZE_LABEL, color=LOSS),
            MathTex(r"\mathbf{3.\ \text{The Cure: Additive Gated Memory Highways (LSTM / GRU).}}", font_size=FONT_SIZE_LABEL, color=PRIMARY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(dilemma_title, DOWN, buff=0.45)

        self.play(FadeIn(dilemma_title), FadeIn(summary_card), run_time=1.0)
        self.play(Indicate(summary_card[2], color=PRIMARY, scale_factor=1.05), run_time=0.8)
        self.wait(3.5)


# ============================================================================
# scene12_lstm.py
# ============================================================================

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


# ============================================================================
# scene13_gru.py
# ============================================================================

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
        h_y = 0.6
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
        ).move_to([-2.0, -0.75, 0])

        cand_box = LabeledBox(
            r"$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$",
            width=3.6,
            height=0.55,
            stroke_color=SECONDARY,
            fill_color=SECONDARY,
            fill_opacity=0.25,
            font_size=FONT_SIZE_LABEL - 4,
        ).move_to([1.2, -0.75, 0])

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
        ).move_to([-2.0, 0.0, 0])

        z_valve = Circle(radius=0.15, stroke_color=WARNING, fill_color=WARNING, fill_opacity=0.35).move_to([-2.0, 0.6, 0])
        z_to_pipe = Arrow(start=z_box.get_top(), end=z_valve.get_bottom(), buff=0.05, color=WARNING, stroke_width=2.5)

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

        self.play(FadeIn(z_box), Create(z_valve), Create(z_to_pipe), FadeIn(convex_formula), FadeIn(convex_cases), run_time=1.0)
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
            FadeOut(z_valve),
            FadeOut(z_to_pipe),
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


# ============================================================================
# scene14_why_gated_rnns_help.py
# ============================================================================

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


# ============================================================================
# scene15_final_synthesis.py
# ============================================================================

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


# ============================================================================
# MASTER FULL VIDEO SCENE (ALL 15 SCENES SEQUENCED)
# ============================================================================

class RNNExplainer(VideoScene):
    """
    Full 15-Scene Recurrent Neural Network (RNN) Explainer Video.
    Sequentially executes each chapter with clean transitions to produce
    one continuous master video file.
    """

    def construct(self) -> None:
        chapters = [
            ("Scene 01: Why RNNs Exist", Scene01WhyRNNsExist),
            ("Scene 02: The RNN Cell", Scene02TheRNNCell),
            ("Scene 03: Unrolling Through Time", Scene03UnrollingDAG),
            ("Scene 04: Forward Pass", Scene04ForwardPass),
            ("Scene 05: Backpropagation Through Time", Scene05BPTTFlow),
            ("Scene 06: Vanishing Gradients", Scene06VanishingGradients),
            ("Scene 07: Exploding Gradients", Scene07ExplodingGradients),
            ("Scene 08: Long-Term Dependencies", Scene08LongTermDependencies),
            ("Scene 09: The Scalar RNN", Scene09ScalarRNN),
            ("Scene 10: The Jacobian Matrix", Scene10JacobianMatrix),
            ("Scene 11: Discrete Dynamical Systems", Scene11StateSpace),
            ("Scene 12: Long Short-Term Memory", Scene12LSTM),
            ("Scene 13: Gated Recurrent Units", Scene13GRU),
            ("Scene 14: Why Gated RNNs Help", Scene14WhyGatedRNNsHelp),
            ("Scene 15: Final Synthesis", Scene15FinalSynthesis),
        ]

        for i, (title, scene_cls) in enumerate(chapters):
            scene_cls.construct(self)
            self.clear_caption()
            if i < len(chapters) - 1:
                # Clean fade-out transition between scenes
                self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
                self.wait(0.4)
            else:
                self.wait(1.0)
