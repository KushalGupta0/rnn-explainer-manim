# Module 14: 3D Geometric Representation of Recurrent Dynamics

[Back to Master Index](../index.md) | [Back to Module 13](13_bptt_visual_semantics.md) | [Next to Module 15](15_common_misconceptions.md)

---

## 1. Why 3D? Avoiding Gimmicks to Reveal Geometry

In educational animations, 3D is frequently abused for superficial eye-candy: spinning 3D cubes, floating text, and arbitrary isometric rotations that obscure mathematical relationships.

To be pedagogically defensible, **a 3D coordinate frame must represent a true mathematical space where all three spatial axes encode distinct, quantifiable physical variables**.

We evaluate two competing 3D representations for an unrolled RNN:

```
MODEL A: Spatial Spatio-Temporal Extrusion
X-axis: Discrete Time (t = 1, 2, ..., T)
Y-axis: State Neuron Index (Feature 1, 2, ..., d_h)
Z-axis: Activation Value / Magnitude h_t[i] ∈ [-1, 1]

MODEL B: True Latent Phase-Space Trajectory (Recommended for Geometry)
X-axis: Latent Coordinate h^(1) ∈ [-1, 1]
Y-axis: Latent Coordinate h^(2) ∈ [-1, 1]
Z-axis: Latent Coordinate h^(3) ∈ [-1, 1]
Time (t): Parametric curve progression γ(t) marching through space
```

---

## 2. Critical Evaluation of Both Representations

| Criteria | Model A: Spatio-Temporal Extrusion (X=Time, Y=Neuron, Z=Value) | Model B: Latent Phase Space (X, Y, Z = Hidden Dimensions) |
| :--- | :--- | :--- |
| **Mathematical Meaning** | Represents an unrolled activation surface $z = f(t, i)$. | Represents the true dynamical state space $\mathcal{H} \subset \mathbb{R}^3$. |
| **Strengths** | Clear temporal separation; easy to see timeline and individual neuron activations across time. | Directly visualizes fixed points, attractors, linear matrix transformations, and limit cycles. |
| **Weaknesses** | Arbitrary ordering of neurons along Y axis (neuron 2 is not "closer" to neuron 3 than neuron 10). | Only visualizes 3 latent dimensions directly; higher dimensions require PCA projection. |
| **BPTT Representation** | Gradients flow backward along the X axis like a collapsing mountain wave. | Backward gradients form a 3D vector field attached to the trajectory curve. |

### The Hybrid Pedagogical Recommendation:
1. **Use Model A for Global Network Architecture & BPTT Flow**: When showing the unrolled computational graph, placing Time on the horizontal X-axis and Activation values on the vertical Z-axis provides an intuitive view of forward pulses and backward gradient decay.
2. **Use Model B for Deep Dive into Jacobian Dynamics & Vanishing Gradients**: When explaining why $W_{hh}$ and $\tanh$ cause gradients to vanish or explode, switch to the 3D Phase Space ($h_1, h_2, h_3$) to show how the unit sphere collapses into a razor-thin pancake under repeated Jacobian transformations.

---

## 3. Coherent 3D Scene Design: The Phase-Space Ribbon

### 1. Spatial Axis Definition (Model B):
* Origin $(0, 0, 0)$ is the zero-activation state.
* The bounding box is the compact unit hypercube $[-1, 1]^3$ (the valid domain of $\tanh$).
* The state at timestep $t$ is a point $\mathbf{h_t} = [h_t^{(1)}, h_t^{(2)}, h_t^{(3)}]^\top$.

### 2. Forward Trajectory:
* An initial state $h_0 = \mathbf{0}$ sits at the origin.
* At each step $t$, an incoming input vector $x_t$ applies a force. The state leaps from $h_{t-1} \to h_t$.
* Draw a continuous smooth interpolated ribbon or glowing tubular curve $\gamma(t)$ connecting the discrete states.
* As the state progresses, the ribbon glows in cool cyan.

### 3. Visualizing Vanishing Gradients in 3D:
* A loss error at step $T$ injects an adjoint vector $\nabla_{h_T} L$ (a vibrant red arrow attached to the tip of $h_T$).
* As the camera tracks backward along the ribbon toward $h_1$:
  * At each step, the backward transformation $J_k^\top = W_{hh}^\top D_k$ acts on the arrow.
  * The arrow visibly shrinks in 3D Euclidean length:
    $$\|\nabla_{h_k} L\|_2 \to 0$$
  * By the time the camera reaches $h_1$, the gradient arrow has shrunk to an invisible point at the origin.
  * The visual proves: **The initial state cannot feel the pull of the loss vector at the end of the ribbon.**

### 4. Visualizing Exploding Gradients in 3D:
* If $\|W_{hh}\|_2 > 1$ and states remain unsaturated:
  * As the gradient vector travels backwards along the ribbon, it doubles in length at every node.
  * Within 5 steps, the arrow pierces through the $[-1, 1]^3$ bounding cube, growing so massive that the camera must pull back by a factor of 100 to keep it in view.

---

## 4. Camera Choreography & Movement Script

To ensure visual continuity in Manim 3D (`ThreeDScene`):
1. **Initial Establishing Shot**:
   * Camera angle: `phi = 75°` (elevation), `theta = -45°` (azimuth).
   * Gives a clear perspective view of the 3D axes and the bounding hypercube.
2. **Forward Tracking Shot**:
   * As the forward trajectory is computed from $t=1 \to T$, the camera pans smoothly along the trajectory curve, keeping the active state $h_t$ centered in the frame.
3. **Orbital Rotation for SVD**:
   * When explaining the Jacobian deformation at step $t$, halt time.
   * Slowly rotate the camera azimuth (`theta: -45° → 45°`) while showing the unit sphere deform into a 3D triaxial ellipsoid under $D_t W_{hh}$.
4. **Reverse Dolly for Backward BPTT**:
   * As backpropagation commences, reverse the camera path along the ribbon, pulling back to a wide-angle overview to show the entire gradient wave extinguishing across the sequence.

---

## 📝 Checkpoint 14 Exercises

### Level 1 — Conceptual
Why is ordering neurons along an arbitrary spatial axis (such as the Y-axis in Model A) geometrically arbitrary, and why is this acceptable when visualizing computational graphs?

### Level 2 — Mathematical
Let an RNN state space be 3-dimensional ($d_h = 3$).
Suppose $W_{hh} = \operatorname{diag}(0.9, 0.5, 0.1)$ and activations are in the linear regime ($D_t = I$).
If a gradient vector at step $T$ is initialized as $g_T = [1, 1, 1]^\top$:
1. Compute the components of the gradient vector $g_t$ at lag $\Delta t = 10$.
2. Along which 3D coordinate axis does the gradient vanish fastest? Along which axis does it survive longest?

### Level 3 — Debugging / Misconception
A video animator rotates the 3D camera rapidly while simultaneous forward and backward pulses are traveling along the network.
Why does rapid camera rotation cause cognitive overload and destroy mathematical comprehension in educational videos?

### Level 4 — Visualization (Manim Translation)
Write the conceptual storyboard for a 20-second 3D Manim scene demonstrating how an attractor in phase space traps the hidden state trajectory, causing all subsequent gradients to vanish.

### Level 5 — Advanced (Manifold Hypothesis in 3D)
In real NLP models, $d_h = 512$ or $1024$. Explain how Principal Component Analysis (PCA) or t-SNE can project high-dimensional latent trajectories onto a 3D Manim coordinate frame while preserving the dominant singular vectors of $W_{hh}$.
