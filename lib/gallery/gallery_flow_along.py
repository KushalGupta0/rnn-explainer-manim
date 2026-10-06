import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import Node, Connector, flow_along
from lib.palette import PRIMARY, SECONDARY, ACCENT, FORWARD

class GalleryFlowAlong(VideoScene):
    def construct(self):
        title = Text("Component Gallery: flow_along", font_size=32).to_edge(UP, buff=0.5)

        # Path between two nodes
        n1 = Node("A", radius=0.5, stroke_color=PRIMARY).shift(LEFT * 3)
        n2 = Node("B", radius=0.5, stroke_color=SECONDARY).shift(RIGHT * 3)
        conn = Connector(n1, n2, label="Channel", color=FORWARD)

        self.add(title, n1, n2, conn)
        self.wait(0.2)
        self.play(flow_along(conn.arrow, color=ACCENT, run_time=1.0))
        self.wait(0.2)
