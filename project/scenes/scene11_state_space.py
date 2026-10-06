"""
Scene 11: State-Space Interpretation — Latent Phase Trajectories
Visualizes the recurrent neural network as a discrete dynamical system on a compact latent space [-1, 1]^2,
traces the forward trajectory of hidden states, and demonstrates contractive basin information collapse.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import (
    Text,
    MathTex,
    VGroup,
    Axes,
    Rectangle,
    Dot,
    Line,
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
        ).move_to(LEFT * 3.2 + DOWN * 0.3)

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

        domain_lbl = Text("Domain: [-1, 1]² (tanh)", font_size=FONT_SIZE_LABEL - 4, color=PARAM).next_to(domain_box, UP, buff=0.15)

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
        ).move_to(axes.coords_to_point(-0.1, -1.35))

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
