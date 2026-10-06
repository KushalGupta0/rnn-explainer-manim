import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import CaptionBar

class GalleryCaptionBar(VideoScene):
    def construct(self):
        title = Text("Component Gallery: CaptionBar", font_size=32).to_edge(UP, buff=0.5)

        # Direct instance of caption bar placed at bottom
        cap = CaptionBar("Demonstrating lower-third caption display with safe margins.", width=11.5)
        cap.to_edge(DOWN, buff=0.4)

        self.add(title, cap)
