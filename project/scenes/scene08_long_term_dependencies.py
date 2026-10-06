"""
Scene 08: Long-Term Dependencies & The Optimization Barrier
Demonstrates sequence-level long-term dependency ("The clouds ... because it rained"),
forward state information dilution, the backward severed gradient telegraph, and Bengio's 1994 Paradox.
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
    DoubleArrow,
    DashedLine,
    FadeIn,
    FadeOut,
    Transform,
    ReplacementTransform,
    Create,
    Indicate,
    Brace,
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

        # Loss node above step T
        loss_node = Node(r"$\mathcal{L}_T$", radius=0.35, stroke_color=LOSS, fill_color=LOSS, fill_opacity=0.3).move_to([x_positions[4], 2.6, 0])
        loss_arrow = Arrow(
            start=loss_node.get_bottom(),
            end=state_nodes[4].get_top(),
            buff=0.1,
            color=LOSS,
            stroke_width=3.5,
            max_tip_length_to_length_ratio=0.25,
        )

        self.play(FadeIn(loss_node), Create(loss_arrow), run_time=0.8)

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
