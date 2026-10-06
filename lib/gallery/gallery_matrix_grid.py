import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from lib.video_scene import VideoScene
from lib.components import MatrixGrid
from lib.palette import PRIMARY, SECONDARY, ACCENT

class GalleryMatrixGrid(VideoScene):
    def construct(self):
        # Variation 1: 2x2 matrix with square brackets
        mat1 = MatrixGrid(
            [["0.8", "0.2"], ["-0.1", "0.9"]],
            bracket_color=PRIMARY,
            font_size=24,
        )
        mat1.shift(LEFT * 2.8 + DOWN * 0.3)
        mat1.highlight_cell(0, 0, ACCENT)

        # Variation 2: 3x1 column vector with parenthesis brackets
        mat2 = MatrixGrid(
            [["1.0"], ["0.5"], ["-0.2"]],
            bracket_type="(",
            bracket_color=SECONDARY,
            font_size=24,
        )
        mat2.shift(RIGHT * 2.8 + DOWN * 0.3)
        mat2.highlight_cell(1, 0, SECONDARY)

        title = Text("Component Gallery: MatrixGrid", font_size=32).to_edge(UP, buff=0.5)
        self.add(title, mat1, mat2)
