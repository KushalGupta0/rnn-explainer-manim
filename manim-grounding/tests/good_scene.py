from manim import *

class GoodScene(Scene):
    def construct(self):
        circle = Circle(radius=1.5, color=BLUE)
        label = Text("Valid Text", font_size=36)
        label.next_to(circle, DOWN, buff=0.5)

        self.play(Create(circle))
        self.play(Write(label))
        self.play(circle.animate.set_fill(BLUE, opacity=0.5))
        self.wait(1)
