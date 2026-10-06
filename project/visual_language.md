# Visual Language & Design System: RNN Explainer

A single-source visual specification for all animations in the RNN Explainer video project, ensuring 3Blue1Brown-quality aesthetic consistency, mathematical rigor, and semantic clarity across all scenes.

---

## 1. Semantic Color Palette

Every color in the animation corresponds strictly to an invariant physical or mathematical role. Colors must never be used decoratively.

| Color Name | Hex Code | Visual Role | Mathematical Meaning |
| :--- | :--- | :--- | :--- |
| `BACKGROUND` | `#0E1117` | Canvas | Deep matte dark slate canvas background. |
| `PRIMARY` | `#58C4DD` | Structure / Focus | Vibrant cyan for active bounding boxes, primary cell outlines, and key labels. |
| `SECONDARY` | `#83C167` | Complement / Target | Fresh green for target outputs $y_t^*$, confirmed states, and stable regimes. |
| `INPUT` | `#2ECC71` | Input Stimulus | Mint green for incoming input vectors $x_t$ injected at time $t$. |
| `STATE` | `#1ABC9C` | Latent Memory | Emerald turquoise for recurrent hidden states $h_t$ and latent space embeddings. |
| `PARAM` | `#9B59B6` | Stationary Operators | Amethyst purple for static weight matrices ($W_{hh}, W_{xh}, W_{hy}$) and biases. |
| `FORWARD` | `#3498DB` | Forward Dynamics | Cool cerulean blue for forward activation flow and chronological signals. |
| `BACKWARD` | `#E67E22` | Gradient Adjoints | Warm amber/orange for backward error propagation and $\frac{\partial L}{\partial h_t}$ adjoints. |
| `LOSS` | `#E74C3C` | Objective / Error | Crimson red for scalar loss functions $L_t$, prediction errors, and penalty signals. |
| `WARNING` | `#E67E22` | Danger / Explosion | High-alert amber for gradient norm explosion, loss cliffs, and saturation zones. |
| `MUTED` | `#6C7A89` | Inactive / Context | Slate gray for past steps, inactive connections, axes, and subdued backgrounds. |

---

## 2. Typography & Text Scaling

All text elements use clean sans-serif typography (`FONT_MAIN = "Arial"`) configured via `Text.set_default(font="Arial")`. Monospace (`FONT_MONO = "Consolas"`) is reserved for raw numerical tensors and source code.

| Role | Constant | Point Size | Placement & Usage |
| :--- | :--- | :--- | :--- |
| **Scene Title** | `FONT_SIZE_TITLE` | 44 pt | Centered at top edge (`to_edge(UP, buff=0.5)`). |
| **Section Heading** | `FONT_SIZE_HEADING` | 34 pt | Prominent sub-section headers and state callouts. |
| **Body & Formula** | `FONT_SIZE_BODY` | 26 pt | Mathematical equations, matrix entries, and structural explanations. |
| **Component Label** | `FONT_SIZE_LABEL` | 20 pt | Node interiors, connector arrow labels, and axis markers. |
| **Caption Bar** | `FONT_SIZE_CAPTION` | 22 pt | Centered within lower-third `CaptionBar` (`buff=0.4` above bottom). |

---

## 3. Pacing, Timing & Motion Rhythms

Animations strictly follow 3Blue1Brown pedagogical breathing rhythm: rapid entry of familiar objects, deliberate evolution of transformations, and generous pauses after conceptual reveals.

* **Fast Transitions** (`RUN_TIME_FAST = 0.5s`): Label changes, fade-outs, auxiliary coordinate highlights.
* **Normal Transformations** (`RUN_TIME_NORMAL = 1.0s`): Vector additions, node creations, camera moves, forward signal flow.
* **Complex Derivations** (`RUN_TIME_SLOW = 1.8s`): SVD ellipse deformation, unrolling loop unspooling, backward gradient cascade.
* **Cognitive Breathing Pauses**:
  * `WAIT_BRIEF = 0.5s`: Inter-beat cadence.
  * `WAIT_NORMAL = 1.2s`: Standard pause to let the viewer inspect an equation or layout.
  * `WAIT_LONG = 2.0s`: Deep pause after a major theoretical takeaway.

---

## 4. Arrow Styles & Directionality Conventions

* **Connectors (`Connector`)**:
  * Stroke width: `2.5`.
  * Arrowhead tip ratio: `max_tip_length_to_length_ratio = 0.25`.
  * Boundary snapping: Connectors always calculate exact boundary intersections via `_get_mobject_edge_point()` with `buff = 0.08` so arrows never penetrate node interiors or box boundaries.
* **Flow Pulses (`flow_along`)**:
  * Built using `ShowPassingFlash` on a tip-less trajectory copy.
  * Pulse width: `time_width = 0.35`.
  * Pulse stroke width: `6.0` (twice the base connection stroke).
* **Directional Orientation**:
  * **Time & Forward Flow**: Left-to-Right ($t=1 \to t=T$).
  * **Input Ingestion**: Top-to-Bottom ($x_t$ drops vertically down into $h_t$).
  * **Prediction Readout**: Bottom-to-Top ($h_t$ emits upward to $\hat{y}_t$ and $L_t$).
  * **Backward Gradients**: Right-to-Left ($t=T \to t=1$, warm amber/orange pulses).

---

## 5. Mathematical Symbol Table & Exact LaTeX Notation

To prevent notation collisions between modules, every mathematical entity uses identical LaTeX representations:

| Symbol | LaTeX Code | Mathematical Object | Standard Dimension |
| :--- | :--- | :--- | :--- |
| $x_t$ | `\mathbf{x}_t` | Input feature vector at time $t$ | $\mathbb{R}^{d_x}$ (mint green) |
| $h_t$ | `\mathbf{h}_t` | Latent hidden state vector at time $t$ | $\mathbb{R}^{d_h}$ (turquoise) |
| $a_t$ | `\mathbf{a}_t` | Pre-activation vector before nonlinearity | $\mathbb{R}^{d_h}$ |
| $\hat{y}_t$ | `\mathbf{\hat{y}}_t` | Model prediction vector | $\mathbb{R}^{d_y}$ (amber) |
| $y_t^*$ | `\mathbf{y}_t^*` | Ground-truth sequence target | $\mathbb{R}^{d_y}$ (clean green) |
| $L_t$ | `L_t` | Local loss scalar at timestep $t$ | $\mathbb{R}$ (crimson red) |
| $L$ | `L` | Cumulative sequence loss $\sum_{t=1}^T L_t$ | $\mathbb{R}$ |
| $W_{xh}$ | `\mathbf{W}_{xh}` | Input-to-hidden weight matrix | $\mathbb{R}^{d_h \times d_x}$ (purple) |
| $W_{hh}$ | `\mathbf{W}_{hh}` | Recurrent hidden-to-hidden transition matrix | $\mathbb{R}^{d_h \times d_h}$ (purple) |
| $W_{hy}$ | `\mathbf{W}_{hy}` | Hidden-to-output emission matrix | $\mathbb{R}^{d_y \times d_h}$ (purple) |
| $b_h$ | `\mathbf{b}_h` | Hidden layer bias vector | $\mathbb{R}^{d_h}$ |
| $b_y$ | `\mathbf{b}_y` | Output layer bias vector | $\mathbb{R}^{d_y}$ |
| $\phi$ | `\phi` | Activation function (standard: `\tanh`) | $\mathbb{R} \to (-1, 1)$ |
| $D_t$ | `\mathbf{D}_t` | Diagonal Jacobian matrix $\operatorname{diag}(\phi'(\mathbf{a}_t))$ | $\mathbb{R}^{d_h \times d_h}$ |
| $J_t$ | `\mathbf{J}_t` | Single-step transition Jacobian $\mathbf{D}_t \mathbf{W}_{hh}$ | $\mathbb{R}^{d_h \times d_h}$ |
| $\frac{\partial L}{\partial h_t}$ | `\frac{\partial L}{\partial \mathbf{h}_t}` | Adjoint loss gradient with respect to state $h_t$ | $\mathbb{R}^{d_h}$ (orange) |
| $\sigma_{\max}$ | `\sigma_{\max}` | Largest singular value / Operator 2-norm | $\mathbb{R}_{\ge 0}$ |
| $\tau$ | `\tau` | Global gradient clipping threshold | $\mathbb{R}_{> 0}$ |
| $C_t$ | `\mathbf{C}_t` | LSTM additive cell state (memory highway) | $\mathbb{R}^{d_h}$ |
