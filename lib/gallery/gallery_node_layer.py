import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import NodeLayer
from lib.palette import PRIMARY, SECONDARY

class GalleryNodeLayer(VideoScene):
    def construct(self):
        # Variation 1: 4-node layer with label
        layer1 = NodeLayer(nodes=4, spacing=0.35, node_radius=0.35, node_stroke=PRIMARY, label="Layer In")
        layer1.shift(LEFT * 2.5)

        # Variation 2: 3-node layer with label
        layer2 = NodeLayer(nodes=3, spacing=0.45, node_radius=0.4, node_stroke=SECONDARY, label="Layer Out")
        layer2.shift(RIGHT * 2.5)

        title = Text("Component Gallery: NodeLayer", font_size=32).to_edge(UP, buff=0.5)
        self.add(title, layer1, layer2)
