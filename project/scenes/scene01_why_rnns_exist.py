"""
Scene 01: Why RNNs Exist — The Sequence Challenge
Interactive mathematical exploration: Demonstrates the geometric failure of fixed feedforward
architectures on variable-length sequences, followed by in-place recurrent state evolution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import (
    Text,
    MathTex,
    VGroup,
    Brace,
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
from lib.components import LabeledBox, Node, Connector
from lib.palette import (
    PRIMARY,
    SECONDARY,
    ACCENT,
    MUTED,
    INPUT,
    STATE,
    WARNING,
    FONT_SIZE_TITLE,
    FONT_SIZE_BODY,
    FONT_SIZE_LABEL,
)


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
        ]).arrange(RIGHT, buff=0.18).shift(UP * 1.3)

        brace_6 = Brace(tokens_6, DOWN, buff=0.12)
        label_6 = brace_6.get_tex(r"6d_x \text{ input vector}")

        ff_interface = LabeledBox(
            r"$\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times 6d_x}$",
            width=7.5,
            height=0.75,
            stroke_color=PRIMARY,
            font_size=FONT_SIZE_BODY,
        ).next_to(brace_6, DOWN, buff=0.4)

        conns_6 = VGroup(*[
            Connector(tokens_6[i], ff_interface, color=MUTED, buff=0.06)
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
            end=[interface_edge_x, -0.6, 0],
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
