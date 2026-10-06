import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import LabeledBox
from lib.palette import PRIMARY, SECONDARY

class GalleryLabeledBox(VideoScene):
    def construct(self):
        # Variation 1: Standard text labeled box
        box1 = LabeledBox("State Vector", width=2.8, height=1.4, stroke_color=PRIMARY)
        box1.shift(LEFT * 2.5)

        # Variation 2: Mathematical / formula labeled box with accent styling
        box2 = LabeledBox(r"$\mathbf{W} \cdot \vec{x} + \vec{b}$", width=3.4, height=1.4, stroke_color=SECONDARY)
        box2.shift(RIGHT * 2.5)

        title = Text("Component Gallery: LabeledBox", font_size=32).to_edge(UP, buff=0.5)
        self.add(title, box1, box2)
