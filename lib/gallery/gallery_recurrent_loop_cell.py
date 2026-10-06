import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from project.components.recurrent_loop_cell import RecurrentLoopCell

class GalleryRecurrentLoopCell(VideoScene):
    def construct(self):
        title = Text("Component Gallery: RecurrentLoopCell", font_size=32).to_edge(UP, buff=0.5)

        cell = RecurrentLoopCell(
            label="A",
            loop_label=r"\mathbf{W}_{hh}",
            input_label=r"\mathbf{x}_t",
            output_label=r"\mathbf{h}_t",
        ).move_to(ORIGIN)

        self.add(title, cell)
        self.wait(0.2)
