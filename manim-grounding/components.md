# Component Registry

| Name | Signature / Description | Status | Gallery Scene |
| :--- | :--- | :--- | :--- |
| `LabeledBox` | `LabeledBox(label: Union[str, Mobject], width=2.0, height=1.0, corner_radius=0.15, fill_color=BACKGROUND, fill_opacity=0.85, stroke_color=PRIMARY, stroke_width=2.5, font_size=FONT_SIZE_BODY, text_color=WHITE, **kwargs)` | verified | `GalleryLabeledBox` |
| `Node` | `Node(label: Optional[Union[str, Mobject]] = None, radius=0.4, fill_color=BACKGROUND, fill_opacity=0.9, stroke_color=PRIMARY, stroke_width=2.5, font_size=FONT_SIZE_LABEL, text_color=WHITE, **kwargs)` | verified | `GalleryNode` |
| `NodeLayer` | `NodeLayer(nodes: Union[int, Sequence[Node]] = 4, spacing=0.35, node_radius=0.35, node_stroke=PRIMARY, label: Optional[Union[str, Mobject]] = None, label_buff=0.3, font_size=FONT_SIZE_LABEL, **kwargs)` | verified | `GalleryNodeLayer` |
| `Connector` | `Connector(start_mobj: Mobject, end_mobj: Mobject, label: Optional[Union[str, Mobject]] = None, color=MUTED, stroke_width=2.5, buff=0.08, label_buff=0.15, label_direction=UP, font_size=FONT_SIZE_LABEL, **kwargs)` | verified | `GalleryConnector` |
| `BoxChain` | `BoxChain(items: Sequence[Union[str, LabeledBox]], spacing=0.9, box_width=1.8, box_height=1.1, stroke_color=PRIMARY, connector_color=MUTED, connector_labels: Optional[Sequence[Optional[str]]] = None, **kwargs)` | verified | `GalleryBoxChain` |
| `MatrixGrid` | `MatrixGrid(matrix: Sequence[Sequence[Any]], bracket_type="bracket", bracket_color=PRIMARY, element_color=WHITE, font_size=FONT_SIZE_BODY, h_buff=0.8, v_buff=0.6, **kwargs)` | verified | `GalleryMatrixGrid` |
| `CaptionBar` | `CaptionBar(text="", width=11.0, height=0.9, bg_color=BACKGROUND, bg_opacity=0.9, stroke_color=MUTED, stroke_width=1.5, font_size=FONT_SIZE_CAPTION, text_color=WHITE, **kwargs)` | verified | `GalleryCaptionBar` |
| `flow_along` | `flow_along(path: VMobject, color=ACCENT, run_time=RUN_TIME_NORMAL, time_width=0.35) -> Animation` | verified | `GalleryFlowAlong` |
