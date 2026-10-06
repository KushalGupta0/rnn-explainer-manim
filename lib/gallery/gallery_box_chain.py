import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import BoxChain
from lib.palette import PRIMARY, FORWARD

class GalleryBoxChain(VideoScene):
    def construct(self):
        # Chain of 3 boxes with step transitions
        chain = BoxChain(
            items=["Step 1", "Step 2", "Step 3"],
            spacing=0.8,
            box_width=2.0,
            box_height=1.1,
            stroke_color=PRIMARY,
            connector_color=FORWARD,
            connector_labels=[r"$\Delta t_1$", r"$\Delta t_2$"],
        )
        chain.shift(DOWN * 0.3)

        title = Text("Component Gallery: BoxChain", font_size=32).to_edge(UP, buff=0.5)
        self.add(title, chain)
