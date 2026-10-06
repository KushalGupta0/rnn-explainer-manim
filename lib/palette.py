"""
Design Palette and Visual Constants
Topic-neutral, single source of style truth.
"""

from manim import ManimColor

# --- SEMANTIC COLORS ---
# Base canvas & UI
BACKGROUND = ManimColor("#0E1117")   # Deep matte charcoal
PRIMARY = ManimColor("#58C4DD")      # Vibrant cyan
SECONDARY = ManimColor("#83C167")    # Clean fresh green
ACCENT = ManimColor("#FC6255")       # Vivid coral red
MUTED = ManimColor("#6C7A89")        # Cool neutral slate gray

# Semantic visual roles (cool vs warm, components)
FORWARD = ManimColor("#3498DB")      # Cool cerulean blue (forward flow)
BACKWARD = ManimColor("#E67E22")     # Warm vibrant amber/orange (backward flow)
PARAM = ManimColor("#9B59B6")        # Amethyst purple (stationary operators)
STATE = ManimColor("#1ABC9C")        # Emerald turquoise (latent states)
LOSS = ManimColor("#E74C3C")         # Crimson red (error / loss)
INPUT = ManimColor("#2ECC71")        # Bright mint green (injected stimulus)
OUTPUT = ManimColor("#F39C12")       # Golden amber (predictions / readouts)
WARNING = ManimColor("#E67E22")      # Alert amber

# --- TYPOGRAPHY CONSTANTS ---
FONT_MAIN = "Arial"                  # Clean proportional font (verified kerning)
FONT_MONO = "Consolas"               # Clean monospace font for code/numerical data

FONT_SIZE_TITLE = 44
FONT_SIZE_HEADING = 34
FONT_SIZE_BODY = 26
FONT_SIZE_LABEL = 20
FONT_SIZE_CAPTION = 22

# --- PACING & TIMING CONSTANTS ---
RUN_TIME_FAST = 0.5
RUN_TIME_NORMAL = 1.0
RUN_TIME_SLOW = 1.8

WAIT_BRIEF = 0.5
WAIT_NORMAL = 1.2
WAIT_LONG = 2.0


def apply_palette_defaults() -> None:
    """Configures Manim global defaults for consistent typography across scenes."""
    from manim import Text, MarkupText
    Text.set_default(font=FONT_MAIN)
    MarkupText.set_default(font=FONT_MAIN)


# Apply typography defaults automatically on import
apply_palette_defaults()

