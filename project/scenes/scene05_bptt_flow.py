"""
Scene 05: Backpropagation Through Time — Gradient Flow
Interactive mathematical exploration: Reverses forward information flow across the exact same
computational graph, demonstrating backward gradient propagation and tributary accumulation.
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
    WARNING,
    FONT_SIZE_TITLE,
    FONT_SIZE_BODY,
    FONT_SIZE_LABEL,
)


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
