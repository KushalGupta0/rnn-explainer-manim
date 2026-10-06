# Component Gaps Analysis

*Audit of visual components and helper abstractions required by storyboarded scenes that currently do not exist in `lib/`.*

---

## 1. Summary of Identified Component Needs

| Component Name | Description & Visual Function | Needed by Scene(s) | Category | Planned Location |
| :--- | :--- | :--- | :--- | :--- |
| `VectorParallelogram` | 2D vector addition diagram with chained arrows and dashed projection lines | `scene02` | **(a) Topic-neutral** | `lib/components.py` |
| `ActivationPlot` | 2D coordinate plot of activation function $\phi(z)$ and its derivative $\phi'(z)$ with dynamic tangent slope line | `scene06` | **(a) Topic-neutral** | `lib/components.py` |
| `ThresholdRing` | Circular threshold radius ring in 2D with radial vector projection $\mathbf{g} \to \mathbf{g} \frac{\tau}{\|\mathbf{g}\|}$ | `scene07` | **(a) Topic-neutral** | `lib/components.py` |
| `LinearTransform2D` | Coordinate grid and unit circle morphing into rotated/stretched ellipse with labeled singular axes | `scene09` | **(a) Topic-neutral** | `lib/components.py` |
| `RecurrentLoopCell` | Compact cell node featuring a curved circular feedback loop arrow labeled $\mathbf{W}_{hh}$ | `scene03` | **(b) RNN-specific** | `project/components/` |
| `DualChannelLSTM` | Dual-pipeline visual containing upper linear highway ($\mathbf{C}_t$) and lower gated conduit ($\mathbf{h}_t$) | `scene08` | **(b) RNN-specific** | `project/components/` |

---

## 2. Layer Placement Justification

### Category (a): Topic-Neutral (Target for `lib/`)
* **`VectorParallelogram`**: Represents fundamental linear vector addition ($\mathbf{u} + \mathbf{v}$). Directly applicable to any mathematics or physics animation.
* **`ActivationPlot`**: General function graphing utility with tangent slopes, useful for calculus and general machine learning explainers.
* **`ThresholdRing`**: Geometric vector projection onto a hypersphere/circle; general utility for constrained optimization.
* **`LinearTransform2D`**: Classic 3Blue1Brown-style linear map visualization (eigenvectors, SVD, matrix action on a circle).

### Category (b): RNN-Specific (Target for `project/`)
* **`RecurrentLoopCell`**: Represents the discrete recurrent feedback abstraction specific to sequential and recurrent architectures.
* **`DualChannelLSTM`**: Represents the proprietary multi-gate topology of Long Short-Term Memory units.

*(Note: Per Phase 3 rules, none of these components are built in this phase; this document serves as the implementation specification for Phase 4).*
