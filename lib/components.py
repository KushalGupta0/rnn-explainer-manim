"""
Reusable Topic-Neutral Manim Visual Primitives
Contains core building blocks: LabeledBox, Node, NodeLayer, Connector,
BoxChain, MatrixGrid, CaptionBar, and flow_along.
"""

from typing import Union, Sequence, Optional, Any, Callable
import numpy as np

from manim import (
    Mobject,
    VMobject,
    VGroup,
    RoundedRectangle,
    Circle,
    Arrow,
    Text,
    MathTex,
    Matrix,
    Polygon,
    DashedLine,
    CurvedArrow,
    ShowPassingFlash,
    Animation,
    RIGHT,
    LEFT,
    UP,
    DOWN,
    UR,
    ORIGIN,
    TAU,
    WHITE,
)

from lib.palette import (
    BACKGROUND,
    PRIMARY,
    SECONDARY,
    ACCENT,
    MUTED,
    PARAM,
    INPUT,
    STATE,
    FONT_MAIN,
    FONT_SIZE_BODY,
    FONT_SIZE_LABEL,
    FONT_SIZE_CAPTION,
    RUN_TIME_NORMAL,
)


class LabeledBox(VGroup):
    """
    Rounded rectangular container with an aligned center label.

    Example:
        box = LabeledBox("State", width=2.5, height=1.5, stroke_color=PRIMARY)
    """

    def __init__(
        self,
        label: Union[str, Mobject],
        width: float = 2.0,
        height: float = 1.2,
        corner_radius: float = 0.15,
        fill_color: Any = BACKGROUND,
        fill_opacity: float = 0.85,
        stroke_color: Any = PRIMARY,
        stroke_width: float = 2.5,
        font_size: float = FONT_SIZE_BODY,
        text_color: Any = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.box = RoundedRectangle(
            corner_radius=corner_radius,
            width=width,
            height=height,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )

        if isinstance(label, str):
            if label.startswith("$") and label.endswith("$"):
                self.label = MathTex(label[1:-1], font_size=font_size, color=text_color)
            else:
                self.label = Text(label, font=FONT_MAIN, font_size=font_size, color=text_color)
        else:
            self.label = label

        self.label.move_to(self.box.get_center())
        self.add(self.box, self.label)


class Node(VGroup):
    """
    Circular structural node with an optional centered glyph or label.

    Example:
        node = Node("x", radius=0.4, stroke_color=SECONDARY)
    """

    def __init__(
        self,
        label: Optional[Union[str, Mobject]] = None,
        radius: float = 0.4,
        fill_color: Any = BACKGROUND,
        fill_opacity: float = 0.9,
        stroke_color: Any = PRIMARY,
        stroke_width: float = 2.5,
        font_size: float = FONT_SIZE_LABEL,
        text_color: Any = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.circle = Circle(
            radius=radius,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        self.add(self.circle)

        self.label: Optional[Mobject] = None
        if label is not None:
            if isinstance(label, str):
                if label.startswith("$") and label.endswith("$"):
                    self.label = MathTex(label[1:-1], font_size=font_size, color=text_color)
                else:
                    self.label = Text(label, font=FONT_MAIN, font_size=font_size, color=text_color)
            else:
                self.label = label

            self.label.move_to(self.circle.get_center())
            self.add(self.label)


class NodeLayer(VGroup):
    """
    Vertical column of aligned Node elements.

    Example:
        layer = NodeLayer(count=4, spacing=0.3, stroke_color=PRIMARY)
    """

    def __init__(
        self,
        nodes: Union[int, Sequence[Node]] = 4,
        spacing: float = 0.35,
        node_radius: float = 0.35,
        node_stroke: Any = PRIMARY,
        label: Optional[Union[str, Mobject]] = None,
        label_buff: float = 0.3,
        font_size: float = FONT_SIZE_LABEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(nodes, int):
            self.nodes = [
                Node(radius=node_radius, stroke_color=node_stroke)
                for _ in range(nodes)
            ]
        else:
            self.nodes = list(nodes)

        self.column = VGroup(*self.nodes)
        self.column.arrange(DOWN, buff=spacing)
        self.add(self.column)

        self.title_label: Optional[Mobject] = None
        if label is not None:
            if isinstance(label, str):
                self.title_label = Text(label, font=FONT_MAIN, font_size=font_size, color=WHITE)
            else:
                self.title_label = label
            self.title_label.next_to(self.column, UP, buff=label_buff)
            self.add(self.title_label)


def _get_mobject_edge_point(
    mob_or_point: Union[Mobject, Sequence[float], np.ndarray],
    direction: np.ndarray,
) -> np.ndarray:
    """Calculate point on boundary of mob along direction vector from its center."""
    if not isinstance(mob_or_point, Mobject):
        return np.asarray(mob_or_point, dtype=float)
    mob = mob_or_point
    c = mob.get_center()
    norm = float(np.linalg.norm(direction[:2]))
    if norm < 1e-7:
        return c
    dx = float(direction[0] / norm)
    dy = float(direction[1] / norm)
    if hasattr(mob, "circle") and hasattr(mob.circle, "radius"):
        r = float(mob.circle.radius)
        return c + r * np.array([dx, dy, 0.0])
    if hasattr(mob, "radius"):
        r = float(getattr(mob, "radius"))
        return c + r * np.array([dx, dy, 0.0])
    w = mob.width
    h = mob.height
    tx = (w / 2.0) / abs(dx) if abs(dx) > 1e-7 else float("inf")
    ty = (h / 2.0) / abs(dy) if abs(dy) > 1e-7 else float("inf")
    t = min(tx, ty)
    return c + t * np.array([dx, dy, 0.0])


class Connector(VGroup):
    """
    Directed arrow connector between two mobjects with optional label.
    Arrow endpoints attach to mobject boundary edges.

    Example:
        conn = Connector(node_a, node_b, label="W", color=MUTED)
    """

    def __init__(
        self,
        start_mobj: Mobject,
        end_mobj: Mobject,
        label: Optional[Union[str, Mobject]] = None,
        color: Any = MUTED,
        stroke_width: float = 2.5,
        buff: float = 0.08,
        label_buff: float = 0.15,
        label_direction: np.ndarray = UP,
        font_size: float = FONT_SIZE_LABEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        diff = end_mobj.get_center() - start_mobj.get_center()
        p_start = _get_mobject_edge_point(start_mobj, diff)
        p_end = _get_mobject_edge_point(end_mobj, -diff)

        self.arrow = Arrow(
            p_start,
            p_end,
            buff=buff,
            color=color,
            stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.25,
        )
        self.add(self.arrow)

        self.label: Optional[Mobject] = None
        if label is not None:
            if isinstance(label, str):
                if label.startswith("$") and label.endswith("$"):
                    self.label = MathTex(label[1:-1], font_size=font_size, color=color)
                else:
                    self.label = Text(label, font=FONT_MAIN, font_size=font_size, color=color)
            else:
                self.label = label

            self.label.next_to(self.arrow, label_direction, buff=label_buff)
            self.add(self.label)


class BoxChain(VGroup):
    """
    Horizontal chain of LabeledBoxes connected by directed arrows.

    Example:
        chain = BoxChain(["A", "B", "C"], spacing=1.0)
    """

    def __init__(
        self,
        items: Sequence[Union[str, LabeledBox]],
        spacing: float = 0.9,
        box_width: float = 1.8,
        box_height: float = 1.1,
        stroke_color: Any = PRIMARY,
        connector_color: Any = MUTED,
        connector_labels: Optional[Sequence[Optional[str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.boxes: list[LabeledBox] = []
        for item in items:
            if isinstance(item, LabeledBox):
                self.boxes.append(item)
            else:
                self.boxes.append(
                    LabeledBox(
                        item,
                        width=box_width,
                        height=box_height,
                        stroke_color=stroke_color,
                    )
                )

        boxes_group = VGroup(*self.boxes)
        boxes_group.arrange(RIGHT, buff=spacing)
        self.add(boxes_group)

        self.connectors: list[Connector] = []
        for i in range(len(self.boxes) - 1):
            conn_lbl = None
            if connector_labels and i < len(connector_labels):
                conn_lbl = connector_labels[i]

            conn = Connector(
                self.boxes[i].box,
                self.boxes[i + 1].box,
                label=conn_lbl,
                color=connector_color,
            )
            self.connectors.append(conn)
            self.add(conn)


class MatrixGrid(VGroup):
    """
    Lightweight styled wrapper over Manim's Matrix class.

    Example:
        mat = MatrixGrid([[1, 2], [3, 4]], bracket_color=PRIMARY)
    """

    def __init__(
        self,
        entries: Sequence[Sequence[Any]],
        h_buff: float = 1.0,
        v_buff: float = 0.7,
        bracket_color: Any = PRIMARY,
        element_color: Any = WHITE,
        font_size: float = FONT_SIZE_LABEL,
        bracket_type: str = "[",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        left_bracket = "[" if bracket_type == "[" else "("
        right_bracket = "]" if bracket_type == "[" else ")"

        # Helper element generator with custom font size
        def entry_to_mob(entry: Any) -> VMobject:
            s = str(entry)
            return MathTex(s, font_size=font_size, color=element_color)

        self.matrix = Matrix(
            entries,
            h_buff=h_buff,
            v_buff=v_buff,
            element_to_mobject=entry_to_mob,
            left_bracket=left_bracket,
            right_bracket=right_bracket,
        )
        self.matrix.get_brackets().set_color(bracket_color)
        self.add(self.matrix)

    def get_cell(self, row: int, col: int) -> VMobject:
        """Returns the entry mobject at the given 0-indexed row and column."""
        return self.matrix.get_entries()[row * len(self.matrix.get_columns()) + col]

    def highlight_cell(self, row: int, col: int, color: Any = ACCENT) -> None:
        """Colors a specific cell in the matrix."""
        self.get_cell(row, col).set_color(color)


class CaptionBar(VGroup):
    """
    Sleek lower-third caption bar for explanatory voiceover text.

    Example:
        cap = CaptionBar("Step 1: Ingesting sequential signal.")
    """

    def __init__(
        self,
        text: str = "",
        width: float = 12.0,
        height: float = 0.75,
        bg_color: Any = BACKGROUND,
        bg_opacity: float = 0.88,
        stroke_color: Any = MUTED,
        stroke_width: float = 1.5,
        font_size: float = FONT_SIZE_CAPTION,
        text_color: Any = WHITE,
        show_box: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.label = Text(
            text,
            font=FONT_MAIN,
            font_size=font_size,
            color=text_color,
        )

        if show_box:
            self.bar = RoundedRectangle(
                corner_radius=0.1,
                width=width,
                height=height,
                fill_color=bg_color,
                fill_opacity=bg_opacity,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
            )
            self.add(self.bar)
            self.label.move_to(self.bar.get_center())
        else:
            self.bar = None

        self.add(self.label)

    def set_text(self, new_text: str) -> None:
        """Updates caption label text and recenters."""
        old_center = self.label.get_center()
        self.remove(self.label)
        self.label = Text(
            new_text,
            font=FONT_MAIN,
            font_size=self.label.font_size,
            color=self.label.color,
        )
        self.label.move_to(old_center)
        self.add(self.label)


def flow_along(
    path: VMobject,
    color: Any = ACCENT,
    run_time: float = RUN_TIME_NORMAL,
    time_width: float = 0.35,
) -> Animation:
    """
    Returns an animation of a glowing signal pulse traveling along a directed path.
    Arrow tips are automatically excluded to prevent duplicate arrowhead artifacts.

    Example:
        self.play(flow_along(connector.arrow, color=ACCENT))
    """
    target = getattr(path, "arrow", path)
    pulse_mob = target.copy()

    # Exclude tip submobjects so the pulse does not draw a duplicate arrowhead
    if hasattr(pulse_mob, "tip") and getattr(pulse_mob, "tip") in pulse_mob.submobjects:
        pulse_mob.remove(getattr(pulse_mob, "tip"))
    pulse_mob.submobjects = [
        sub
        for sub in pulse_mob.submobjects
        if not getattr(sub, "__class__", type).__name__.endswith("Tip")
    ]

    pulse_mob.set_color(color)
    base_width = target.get_stroke_width()
    pulse_mob.set_stroke(width=max(base_width * 2.0, 5.0))
    return ShowPassingFlash(pulse_mob, time_width=time_width, run_time=run_time)


class VectorParallelogram(VGroup):
    r"""
    Topic-neutral 2D vector addition diagram showing two vectors,
    dashed projection edges completing the parallelogram, and the resultant sum.

    Example:
        diag = VectorParallelogram(
            u=[2.0, 0.4, 0],
            v=[0.8, 1.8, 0],
            u_label=r"\mathbf{u}",
            v_label=r"\mathbf{v}",
            sum_label=r"\mathbf{u}+\mathbf{v}",
        )
    """

    def __init__(
        self,
        u: Sequence[float],
        v: Sequence[float],
        origin: Sequence[float] = ORIGIN,
        u_color: Any = PRIMARY,
        v_color: Any = SECONDARY,
        sum_color: Any = ACCENT,
        u_label: Optional[Union[str, Mobject]] = None,
        v_label: Optional[Union[str, Mobject]] = None,
        sum_label: Optional[Union[str, Mobject]] = None,
        stroke_width: float = 3.0,
        dash_length: float = 0.08,
        fill_opacity: float = 0.12,
        font_size: float = FONT_SIZE_LABEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        orig = np.array(origin, dtype=float)
        vec_u = np.array(u, dtype=float)
        vec_v = np.array(v, dtype=float)
        p_u = orig + vec_u
        p_v = orig + vec_v
        p_sum = orig + vec_u + vec_v

        # Shaded interior polygon
        self.polygon = Polygon(
            orig, p_u, p_sum, p_v,
            fill_color=sum_color,
            fill_opacity=fill_opacity,
            stroke_width=0,
        )
        self.add(self.polygon)

        # Base vectors u and v
        self.arrow_u = Arrow(orig, p_u, buff=0, color=u_color, stroke_width=stroke_width)
        self.arrow_v = Arrow(orig, p_v, buff=0, color=v_color, stroke_width=stroke_width)
        self.add(self.arrow_u, self.arrow_v)

        # Dashed projection sides
        self.dash_u_top = DashedLine(p_v, p_sum, color=u_color, dash_length=dash_length, stroke_width=stroke_width * 0.8)
        self.dash_v_right = DashedLine(p_u, p_sum, color=v_color, dash_length=dash_length, stroke_width=stroke_width * 0.8)
        self.add(self.dash_u_top, self.dash_v_right)

        # Resultant sum vector
        self.arrow_sum = Arrow(orig, p_sum, buff=0, color=sum_color, stroke_width=stroke_width * 1.2)
        self.add(self.arrow_sum)

        # Labels
        self.u_label_mob: Optional[Mobject] = None
        if u_label is not None:
            self.u_label_mob = MathTex(u_label, font_size=font_size, color=u_color) if isinstance(u_label, str) else u_label
            self.u_label_mob.next_to(self.arrow_u, DOWN, buff=0.15)
            self.add(self.u_label_mob)

        self.v_label_mob: Optional[Mobject] = None
        if v_label is not None:
            self.v_label_mob = MathTex(v_label, font_size=font_size, color=v_color) if isinstance(v_label, str) else v_label
            self.v_label_mob.next_to(self.arrow_v, LEFT, buff=0.15)
            self.add(self.v_label_mob)

        self.sum_label_mob: Optional[Mobject] = None
        if sum_label is not None:
            self.sum_label_mob = MathTex(sum_label, font_size=font_size, color=sum_color) if isinstance(sum_label, str) else sum_label
            self.sum_label_mob.next_to(p_sum, UR, buff=0.15)
            self.add(self.sum_label_mob)


class RecurrentLoopCell(VGroup):
    r"""
    Compact single-cell RNN representation featuring a central processing box,
    an upward input arrow, an upward readout arrow, and a circular feedback loop.

    Example:
        cell = RecurrentLoopCell(label="A", loop_label=r"\mathbf{W}_{hh}")
    """

    def __init__(
        self,
        label: Union[str, Mobject] = "A",
        box_width: float = 1.6,
        box_height: float = 1.4,
        loop_label: Optional[Union[str, Mobject]] = r"\mathbf{W}_{hh}",
        input_label: Optional[Union[str, Mobject]] = r"\mathbf{x}_t",
        output_label: Optional[Union[str, Mobject]] = r"\mathbf{h}_t",
        stroke_color: Any = PRIMARY,
        loop_color: Any = PARAM,
        input_color: Any = INPUT,
        output_color: Any = STATE,
        font_size: float = FONT_SIZE_LABEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.box = LabeledBox(
            label=label,
            width=box_width,
            height=box_height,
            stroke_color=stroke_color,
            font_size=FONT_SIZE_BODY,
        )
        self.add(self.box)

        # Circular feedback loop curving out of the right side
        p_start = self.box.get_right() + UP * 0.35
        p_end = self.box.get_right() + DOWN * 0.35
        self.loop_arrow = CurvedArrow(
            p_start,
            p_end,
            angle=-TAU * 0.75,
            color=loop_color,
            stroke_width=2.5,
        )
        self.add(self.loop_arrow)

        self.loop_label_mob: Optional[Mobject] = None
        if loop_label is not None:
            self.loop_label_mob = (
                MathTex(loop_label, font_size=font_size, color=loop_color)
                if isinstance(loop_label, str)
                else loop_label
            )
            self.loop_label_mob.next_to(self.loop_arrow, RIGHT, buff=0.15)
            self.add(self.loop_label_mob)

        self.input_arrow = Arrow(
            self.box.get_bottom() + DOWN * 1.2,
            self.box.get_bottom(),
            buff=0.08,
            color=input_color,
            stroke_width=2.5,
        )
        self.add(self.input_arrow)

        self.input_label_mob: Optional[Mobject] = None
        if input_label is not None:
            self.input_label_mob = (
                MathTex(input_label, font_size=font_size, color=input_color)
                if isinstance(input_label, str)
                else input_label
            )
            self.input_label_mob.next_to(self.input_arrow, LEFT, buff=0.15)
            self.add(self.input_label_mob)

        self.output_arrow = Arrow(
            self.box.get_top(),
            self.box.get_top() + UP * 1.2,
            buff=0.08,
            color=output_color,
            stroke_width=2.5,
        )
        self.add(self.output_arrow)

        self.output_label_mob: Optional[Mobject] = None
        if output_label is not None:
            self.output_label_mob = (
                MathTex(output_label, font_size=font_size, color=output_color)
                if isinstance(output_label, str)
                else output_label
            )
            self.output_label_mob.next_to(self.output_arrow, LEFT, buff=0.15)
            self.add(self.output_label_mob)


