import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import Node
from lib.palette import PRIMARY, SECONDARY, FORWARD

class GalleryNode(VideoScene):
    def construct(self):
        # Variation 1: Empty structural node
        node1 = Node(radius=0.5, stroke_color=PRIMARY)
        node1.shift(LEFT * 3)

        # Variation 2: Node with character label
        node2 = Node("A", radius=0.5, stroke_color=SECONDARY)
        node2.shift(ORIGIN)

        # Variation 3: Math glyph node
        node3 = Node(r"$\vec{h}_t$", radius=0.55, stroke_color=FORWARD)
        node3.shift(RIGHT * 3)

        title = Text("Component Gallery: Node", font_size=32).to_edge(UP, buff=0.5)
        self.add(title, node1, node2, node3)
