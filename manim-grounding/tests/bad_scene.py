from manim import *

class BadScene(Scene):
    def construct(self):
        # 1. Unknown Manim symbol (misspelled)
        c = Circl(radius=1.0)
        
        # 2. Missing required argument (Text requires 'text')
        bad_text = Text()
        
        # 3. Non-existent method call on Circle
        circle = Circle()
        circle.fake_method()
        
        # 4. Animation class used as a method call
        circle.Create()
        
        # 5. Deprecated API call
        self.play(ShowCreation(circle))
        
        # 6. Mobject animated before being added to scene
        square = Square()
        self.play(square.animate.shift(UP))
