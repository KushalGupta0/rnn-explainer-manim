"""
Scene 03: Unrolling Through Time — Loop vs. DAG
Interactive mathematical exploration: Demonstrates the equivalence between a recurrent feedback loop
and an unrolled DAG with physical parameter sharing and traveling forward computation.
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
from lib.components import LabeledBox, Node, Connector, BoxChain
from project.components.recurrent_loop_cell import RecurrentLoopCell
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
        ).shift(DOWN * 0.45)

        # Input and output nodes for the 3 steps
        inputs = VGroup(*[
            Node(rf"$\mathbf{{x}}_{i}$", radius=0.3, stroke_color=INPUT).next_to(chain.boxes[i - 1], DOWN, buff=0.65)
            for i in range(1, 4)
        ])
        in_conns = VGroup(*[
            Connector(inputs[i], chain.boxes[i], color=INPUT)
            for i in range(3)
        ])

        outputs = VGroup(*[
            Node(rf"$\mathbf{{h}}_{i}$", radius=0.3, stroke_color=STATE).next_to(chain.boxes[i - 1], UP, buff=0.65)
            for i in range(1, 4)
        ])
        out_conns = VGroup(*[
            Connector(chain.boxes[i], outputs[i], color=STATE)
            for i in range(3)
        ])

        self.play(
            ReplacementTransform(loop_cell, chain),
            run_time=1.8,
        )
        self.play(
            FadeIn(inputs),
            Create(in_conns),
            FadeIn(outputs),
            Create(out_conns),
            run_time=1.5,
        )
        self.wait(6.0)

        # -------------------------------------------------------------
        # BEAT 3 (0:21 - 0:31 | 10s): Physical Parameter Sharing
        # -------------------------------------------------------------
        self.show_caption("Crucially, one physical parameter matrix is shared across all timesteps.")

        # Master persistent parameter badge
        w_master = LabeledBox(
            r"$\mathbf{W}_{hh} \text{ (Shared Parameter)}$",
            width=3.6,
            height=0.55,
            stroke_color=PARAM,
            fill_color=PARAM,
            fill_opacity=0.22,
            font_size=FONT_SIZE_LABEL,
        ).next_to(outputs, UP, buff=0.45)

        # Distribution arrows feeding from master W_hh into each timestep cell
        w_dist_conns = VGroup(*[
            Connector(w_master, chain.boxes[i], color=PARAM, stroke_width=1.8)
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

        # Step 1 execution: x_1 -> Box 1 -> h_1 -> Box 2
        self.play(Indicate(inputs[0], color=INPUT), Indicate(chain.boxes[0], color=PRIMARY), run_time=0.6)
        self.play(Indicate(outputs[0], color=STATE), Indicate(chain.connectors[0], color=PARAM), run_time=0.6)

        # Step 2 execution: x_2 -> Box 2 -> h_2 -> Box 3
        self.play(Indicate(inputs[1], color=INPUT), Indicate(chain.boxes[1], color=PRIMARY), run_time=0.6)
        self.play(Indicate(outputs[1], color=STATE), Indicate(chain.connectors[1], color=PARAM), run_time=0.6)

        # Step 3 execution: x_3 -> Box 3 -> h_3
        self.play(Indicate(inputs[2], color=INPUT), Indicate(chain.boxes[2], color=PRIMARY), run_time=0.6)
        self.play(Indicate(outputs[2], color=STATE), run_time=0.6)
        self.wait(3.0)

        self.clear_caption()
        self.wait(0.5)
