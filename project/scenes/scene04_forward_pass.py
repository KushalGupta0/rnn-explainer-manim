"""
Scene 04: The Forward Pass — Sequential Computation
Interactive mathematical exploration: Demonstrates how information physically travels
forward through unrolled timesteps, computing states and caching values for backpropagation.
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
    FONT_SIZE_TITLE,
    FONT_SIZE_BODY,
    FONT_SIZE_LABEL,
)


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
            inputs[0].animate.set_color(INPUT),
            in_conns[0].animate.set_color(INPUT),
            chain.boxes[0].animate.set_color(PRIMARY),
            run_time=0.8,
        )
        self.play(Indicate(chain.boxes[0], color=PRIMARY, scale_factor=1.08), run_time=0.5)
        self.play(
            out_conns[0].animate.set_color(OUTPUT),
            outputs[0].animate.set_color(OUTPUT),
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
            inputs[1].animate.set_color(INPUT),
            in_conns[1].animate.set_color(INPUT),
            chain.boxes[1].animate.set_color(PRIMARY),
            run_time=0.8,
        )
        self.play(Indicate(chain.boxes[1], color=PRIMARY, scale_factor=1.08), run_time=0.5)
        self.play(
            out_conns[1].animate.set_color(OUTPUT),
            outputs[1].animate.set_color(OUTPUT),
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
            inputs[2].animate.set_color(INPUT),
            in_conns[2].animate.set_color(INPUT),
            chain.boxes[2].animate.set_color(PRIMARY),
            run_time=0.8,
        )
        self.play(Indicate(chain.boxes[2], color=PRIMARY, scale_factor=1.08), run_time=0.5)
        self.play(
            out_conns[2].animate.set_color(OUTPUT),
            outputs[2].animate.set_color(OUTPUT),
            run_time=0.8,
        )

        # Forward cache banner above outputs
        cache_badge = LabeledBox(
            r"$\text{Forward Cache: } \{\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_3\} \text{ preserved in memory}$",
            width=5.2,
            height=0.55,
            stroke_color=STATE,
            fill_color=STATE,
            fill_opacity=0.2,
            font_size=FONT_SIZE_LABEL,
        ).next_to(outputs, UP, buff=0.45)

        self.play(FadeIn(cache_badge, shift=DOWN * 0.15), run_time=1.0)
        self.wait(4.0)

        # -------------------------------------------------------------
        # BEAT 4 (0:33 - 0:43 | 10s): Loss Accumulation Across Time
        # -------------------------------------------------------------
        self.show_caption("Predictions are compared against targets, accumulating total loss.")

        # Loss nodes above each prediction
        loss_nodes = VGroup(*[
            Node(rf"$\mathcal{{L}}_{i}$", radius=0.28, stroke_color=LOSS, text_color=LOSS).next_to(outputs[i - 1], UP, buff=0.45)
            for i in range(1, 4)
        ])
        loss_conns = VGroup(*[
            Connector(outputs[i], loss_nodes[i], color=LOSS)
            for i in range(3)
        ])

        total_loss_eq = MathTex(
            r"\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} \mathcal{L}_t(\mathbf{\hat{y}}_t, \mathbf{y}_t)",
            font_size=FONT_SIZE_LABEL + 2,
            color=LOSS,
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
