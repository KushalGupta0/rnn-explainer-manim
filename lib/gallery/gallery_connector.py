import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import Node, Connector
from lib.palette import PRIMARY, SECONDARY, ACCENT, FORWARD

class GalleryConnector(VideoScene):
    def construct(self):
        # Two pairs of nodes
        n1 = Node("1", radius=0.45, stroke_color=PRIMARY).shift(LEFT * 4 + UP * 0.5)
        n2 = Node("2", radius=0.45, stroke_color=PRIMARY).shift(LEFT * 1 + UP * 0.5)
        conn1 = Connector(n1, n2, label="Signal", color=FORWARD)

        n3 = Node("A", radius=0.45, stroke_color=SECONDARY).shift(RIGHT * 1 + DOWN * 0.5)
        n4 = Node("B", radius=0.45, stroke_color=SECONDARY).shift(RIGHT * 4 + DOWN * 0.5)
        conn2 = Connector(n3, n4, label=r"$\mathbf{W}$", color=ACCENT)

        title = Text("Component Gallery: Connector", font_size=32).to_edge(UP, buff=0.5)
        self.add(title, n1, n2, conn1, n3, n4, conn2)
