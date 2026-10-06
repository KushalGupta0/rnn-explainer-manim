import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import VectorParallelogram
from lib.palette import PRIMARY, SECONDARY, ACCENT

class GalleryVectorParallelogram(VideoScene):
    def construct(self):
        title = Text("Component Gallery: VectorParallelogram", font_size=32).to_edge(UP, buff=0.5)

        parallelogram = VectorParallelogram(
            u=[2.2, 0.4, 0],
            v=[0.8, 1.8, 0],
            origin=[-1.5, -1.0, 0],
            u_color=PRIMARY,
            v_color=SECONDARY,
            sum_color=ACCENT,
            u_label=r"\mathbf{W}_{xh}\mathbf{x}_t",
            v_label=r"\mathbf{W}_{hh}\mathbf{h}_{t-1}",
            sum_label=r"\mathbf{a}_t",
        )

        self.add(title, parallelogram)
        self.wait(0.2)
