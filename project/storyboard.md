# Master Storyboard: Recurrent Neural Networks & Gradient Dynamics

Target Length: **6 minutes (360 seconds)** | Aspect Ratio: **16:9** | Visual Philosophy: **3Blue1Brown Geometric Rigor**

---

## 🎬 Master Scene Table

| Scene ID | Title | Tier | Duration | Risk / Complexity |
| :--- | :--- | :---: | :---: | :--- |
| `scene01` | **Why RNNs Exist: The Sequence Challenge** | Tier 1 | 40s | Low |
| `scene02` | **The RNN Cell: First Principles & Dimensions** | Tier 1 | 45s | Low |
| `scene03` | **Unrolling Through Time: Loop vs. DAG** | Tier 1 | 40s | Medium |
| `scene04` | **The Forward Pass: Numerical State Caching** | Tier 1 | 45s | Low |
| `scene05` | **Backpropagation Through Time: Gradient Flow** | Tier 1 | 45s | Medium |
| `scene06` | **The Vanishing Gradient: Exponential Collapse** | Tier 1 | 45s | High |
| `scene07` | **Exploding Gradients & Norm Clipping Geometry** | Tier 2 | 45s | Medium |
| `scene08` | **Architectural Evolution: The LSTM Linear Highway** | Tier 2 | 40s | Medium |
| `scene09` | **Geometric Phase Space & SVD Contraction** | Tier 3 | 15s | High (Optional/Fallback) |
| **TOTAL**| **Complete Master Curriculum** | — | **360s (6:00)**| — |

*Tier 1 Total: 260s (72.2% of video runtime, guaranteeing the core thesis ships under tight deadlines).*

---

## Scene 01: Why RNNs Exist: The Sequence Challenge
* **ID**: `scene01`
* **Title**: Why RNNs Exist: The Sequence Challenge
* **Tier**: Tier 1 (Must Ship) | **Target Duration**: 40s | **Risk**: Low
* **The ONE Idea**: Feedforward networks fail on sequences because fixed weight matrices cannot handle variable sequence lengths or position-shifted context without exploding parameter counts.
* **Knowledge-Base References**: `RNN Video/modules/01_why_rnns_exist.md`, `deep_dives/variable_length_sequences/README.md`.
* **On-Screen Objects & Components**:
  * `NodeLayer` (MLP input layer of fixed size $N$).
  * `LabeledBox` (Feedforward hidden layer $W$).
  * `Connector` (rigid arrows connecting fixed input slots).
  * `MathTex` formulas for sequence tokens $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T$.
* **Layout**:
  * Fixed MLP architecture positioned horizontally in screen center.
  * Variable-length token sequence positioned directly above, aligning with inputs.
  * Caption bar locked at lower third.
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–10s, 10s)**: Introduce a standard feedforward network processing a fixed 3-word phrase ("clouds in sky"). All weights connect cleanly.
  2. **Beat 2 (10–22s, 12s)**: Present a 6-word sentence ("the dark clouds floating in sky"). Nodes overflow the fixed input grid; question mark highlights the structural mismatch.
  3. **Beat 3 (22–32s, 10s)**: Shift the keyword from the start to the end of the sentence. Highlight that a standard dense layer learns position-dependent weights rather than invariant meaning.
  4. **Beat 4 (32–40s, 8s)**: Dissolve the rigid MLP frame into a sequential input stream $\mathbf{x}_t$ arriving one step at a time into an adaptive recurrent core.
* **Narration Cues**:
  * *"Standard neural networks expect fixed-dimensional vectors. But real-world data—language, audio, sensor streams—unfolds sequentially across variable lengths..."*
* **Fallback / Simplification**: If time is restricted, skip the word-shift example and directly animate the 3-token vs 6-token dimension overflow.

---

## Scene 02: The RNN Cell: First Principles & Dimensions
* **ID**: `scene02`
* **Title**: The RNN Cell: First Principles & Dimensions
* **Tier**: Tier 1 (Must Ship) | **Target Duration**: 45s | **Risk**: Low
* **The ONE Idea**: The hidden state $\mathbf{h}_t$ is an updated summary vector combining current input and past memory through linear superposition followed by bounded activation squashing.
* **Knowledge-Base References**: `RNN Video/modules/02_the_rnn_cell.md`, `spec/rnn_animation_spec.md` (Sec 1).
* **On-Screen Objects & Components**:
  * `LabeledBox` (Recurrent Cell body).
  * `Node` ($\mathbf{x}_t$ input node, $\mathbf{h}_{t-1}$ past memory node, $\mathbf{h}_t$ updated state node, $\mathbf{\hat{y}}_t$ output node).
  * `Connector` (inflow and outflow arrows with weight badges $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$).
  * `MathTex` equation block: $\mathbf{a}_t = \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h$ and $\mathbf{h}_t = \tanh(\mathbf{a}_t)$.
* **Layout**:
  * Main cell box centered at $(0, 0)$.
  * Input $\mathbf{x}_t$ directly below at $(0, -2.5)$; past state $\mathbf{h}_{t-1}$ entering from left at $(-3.5, 0)$.
  * Output state $\mathbf{h}_t$ exiting to right at $(3.5, 0)$; prediction $\mathbf{\hat{y}}_t$ emitted upward to $(0, 2.5)$.
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–12s, 12s)**: Fade in the central cell with recurrent state $\mathbf{h}_{t-1}$ and input $\mathbf{x}_t$. Show matrix transformations $\mathbf{W}_{hh}\mathbf{h}_{t-1}$ and $\mathbf{W}_{xh}\mathbf{x}_t$.
  2. **Beat 2 (12–25s, 13s)**: Display vector superposition inside the cell: the two vectors add together with bias $\mathbf{b}_h$ to form pre-activation $\mathbf{a}_t$.
  3. **Beat 3 (25–35s, 10s)**: Apply $\tanh$ squashing. Show numerical coordinates contracting into the bounded interval $[-1, 1]$.
  4. **Beat 4 (35–45s, 10s)**: Emission of updated state $\mathbf{h}_t$ and readout prediction $\mathbf{\hat{y}}_t = g(\mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y)$.
* **Narration Cues**:
  * *"At each timestep, the recurrent cell receives two distinct signals: fresh external evidence and a compressed summary of the entire past..."*
* **Fallback / Simplification**: Replace the inner vector addition parallelogram with direct scalar formula popups.

---

## Scene 03: Unrolling Through Time: Loop vs. DAG
* **ID**: `scene03`
* **Title**: Unrolling Through Time: Loop vs. DAG
* **Tier**: Tier 1 (Must Ship) | **Target Duration**: 40s | **Risk**: Medium
* **The ONE Idea**: A recurrent loop in space is mathematically equivalent to a deep feedforward network across time, where the identical weight matrices are shared across all layers.
* **Knowledge-Base References**: `RNN Video/modules/03_unrolling_the_rnn.md`, `spec/rnn_animation_spec.md` (Sec 10).
* **On-Screen Objects & Components**:
  * `Node` (Single cyclical RNN cell with circular feedback arrow).
  * `BoxChain` (Horizontal sequence of unrolled cells for $t=1, 2, 3$).
  * `Connector` (Horizontal temporal transitions labeled $\mathbf{W}_{hh}$).
  * Master weight badge $\mathbf{W}_{hh}$ casting identical color highlights onto each unrolled step.
* **Layout**:
  * Start with compact cyclical cell centered at $(0, 0)$.
  * Morph into horizontal `BoxChain` centered at screen middle, spanning $X \in [-4.5, 4.5]$.
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–10s, 10s)**: Focus on the single compact recurrent cell with self-loop arrow spinning as time advances $t=1, 2, 3$.
  2. **Beat 2 (10–22s, 12s)**: The circular feedback loop snaps open and stretches horizontally into an unrolled chain of discrete steps: $t=1 \to t=2 \to t=3$.
  3. **Beat 3 (22–32s, 10s)**: Reveal parameter sharing: illuminate the master parameter badge $\mathbf{W}_{hh}$ above the chain; highlight that every step reuses the identical matrix.
  4. **Beat 4 (32–40s, 8s)**: Contrast with deep feedforward networks: a 100-step RNN is a 100-layer network with stationary, shared weights.
* **Narration Cues**:
  * *"A loop in time can be unspooled into a directed acyclic graph. This reveals the core property of RNNs: parameter sharing..."*
* **Fallback / Simplification**: Instead of dynamic morphing of the circular arrow, fade out the loop and fade in the unrolled `BoxChain`.

---

## Scene 04: The Forward Pass: Numerical State Caching
* **ID**: `scene04`
* **Title**: The Forward Pass: Numerical State Caching
* **Tier**: Tier 1 (Must Ship) | **Target Duration**: 45s | **Risk**: Low
* **The ONE Idea**: Forward execution sequentially computes and caches every latent state vector $\mathbf{h}_t$, producing local predictions and accumulating sequence loss.
* **Knowledge-Base References**: `RNN Video/modules/04_forward_pass_numerical.md`, `modules/05_loss_and_bptt_foundations.md`.
* **On-Screen Objects & Components**:
  * `BoxChain` (3 unrolled timesteps).
  * `flow_along` (Cool blue signal pulses advancing left-to-right).
  * `MatrixGrid` (Concrete state vectors: $\mathbf{h}_0 = [0, 0]^\top, \mathbf{h}_1 = [0.8, 0.2]^\top, \mathbf{h}_2 = [-0.1, 0.9]^\top$).
  * `Node` (Loss beacons $L_1, L_2, L_3$ igniting above each step).
* **Layout**:
  * Horizontal chain in lower-middle ($Y = -0.5$).
  * Numerical state vectors float directly below respective cells.
  * Loss nodes ignite above cells at $Y = 2.0$.
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–12s, 12s)**: Initialize $\mathbf{h}_0 = \mathbf{0}$. Input $\mathbf{x}_1$ triggers `flow_along` into Step 1, producing concrete state $\mathbf{h}_1$.
  2. **Beat 2 (12–24s, 12s)**: State $\mathbf{h}_1$ flows into Step 2 alongside $\mathbf{x}_2$. Compute and cache $\mathbf{h}_2$.
  3. **Beat 3 (24–35s, 11s)**: State $\mathbf{h}_2$ flows into Step 3. Show all states $\{\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_3\}$ frozen in the forward memory cache.
  4. **Beat 4 (35–45s, 10s)**: Predictions $\mathbf{\hat{y}}_t$ compare with targets $\mathbf{y}_t^*$, igniting loss beacons $L_t$ and defining total loss $L = \sum L_t$.
* **Narration Cues**:
  * *"As data flows forward, each step must remember what came before. Crucially, every intermediate state must be retained in memory..."*
* **Fallback / Simplification**: Show 2 timesteps instead of 3 if screen real-estate becomes tight.

---

## Scene 05: Backpropagation Through Time: Gradient Flow
* **ID**: `scene05`
* **Title**: Backpropagation Through Time: Gradient Flow
* **Tier**: Tier 1 (Must Ship) | **Target Duration**: 45s | **Risk**: Medium
* **The ONE Idea**: Gradient signals flow anti-chronologically from future errors backward through the state chain, branching and accumulating at every state nexus.
* **Knowledge-Base References**: `RNN Video/modules/06_bptt_deep_derivation.md`, `spec/rnn_animation_spec.md` (Sec 3, 4).
* **On-Screen Objects & Components**:
  * Unrolled 3-step computational graph from Scene 04.
  * `flow_along` (Warm amber/orange gradient fluid pulses traveling right-to-left).
  * `MathTex` adjoint recurrence equation: $\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L_t}{\partial \mathbf{h}_t} + \mathbf{W}_{hh}^\top \mathbf{D}_{t+1} \frac{\partial L}{\partial \mathbf{h}_{t+1}}$.
  * Forked gradient arrows combining at $\mathbf{h}_t$.
* **Layout**:
  * Unrolled chain across middle screen.
  * Backward pulses originate from $L_3$ at top right, flowing leftward along arrows.
  * Governing BPTT equation displayed prominently at top screen.
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–12s, 12s)**: Loss $L_3$ injects error gradient $\frac{\partial L_3}{\partial \mathbf{h}_3}$. Orange pulse flows backward into $\mathbf{h}_3$.
  2. **Beat 2 (12–24s, 12s)**: Pulse passes through transpose Jacobian $\mathbf{W}_{hh}^\top \mathbf{D}_3$ into $\mathbf{h}_2$. At $\mathbf{h}_2$, the incoming pulse merges with local error from $L_2$.
  3. **Beat 3 (24–35s, 11s)**: Gradient streams converge and advance into $\mathbf{h}_1$. Highlight the two tributary paths at each state nexus.
  4. **Beat 4 (35–45s, 10s)**: Display cumulative weight update $\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \delta_t \mathbf{h}_{t-1}^\top$, summing contributions across all steps.
* **Narration Cues**:
  * *"To train the network, error signals must travel backward through time. At every step, future loss gradients flow into the past..."*
* **Fallback / Simplification**: Animate single-path BPTT from final loss $L_T$ to $\mathbf{h}_1$ before showing intermediate loss sums.

---

## Scene 06: The Vanishing Gradient: Exponential Collapse
* **ID**: `scene06`
* **Title**: The Vanishing Gradient: Exponential Collapse
* **Tier**: Tier 1 (Must Ship) | **Target Duration**: 45s | **Risk**: High
* **The ONE Idea**: Gradients across multiple steps scale as the product of transition Jacobians; when effective spectral norms are less than 1, gradients decay exponentially to zero, blinding the network to early events.
* **Knowledge-Base References**: `RNN Video/modules/07_scalar_rnn_gradient_dynamics.md`, `modules/08_activation_functions.md`, `modules/10_vanishing_gradients.md`.
* **On-Screen Objects & Components**:
  * Long horizontal chain (5–6 steps).
  * Gradient pulse arrow shrinking exponentially: $1.0 \to 0.5 \to 0.25 \to 0.06 \to 0.001$.
  * Tanh curve with derivative graph $\phi'(a) \le 1.0$ highlighting derivative saturation.
  * LaTeX equation: $\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^T (\mathbf{D}_k \mathbf{W}_{hh}) \approx c^{T-t} \to 0$.
* **Layout**:
  * Top half: Long unrolled chain with fading orange pulses.
  * Bottom left: Mathematical power law $c^{T-t}$.
  * Bottom right: Tanh activation curve showing derivative collapse.
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–12s, 12s)**: Inject error at $t=5$. Pulse flows backward toward $t=1$. As it crosses each $\mathbf{W}_{hh}^\top \mathbf{D}_k$, its thickness and brightness shrink by half.
  2. **Beat 2 (12–23s, 11s)**: By $t=1$, the gradient pulse has shrunk to an invisible hair. State $\mathbf{h}_1$ receives zero learning signal.
  3. **Beat 3 (23–34s, 11s)**: Isolate the culprit: show the chain of Jacobians. Show that $\tanh'(a) \le 1.0$, and in saturated regions drops near zero.
  4. **Beat 4 (34–45s, 11s)**: Present Bengio's Dilemma: $0.85^{20} \approx 0.038$. The network suffers from severe recency bias and forgets distant context.
* **Narration Cues**:
  * *"Every step backward multiplies the gradient by another Jacobian matrix. If this effective multiplier is even slightly below one, exponential decay takes over..."*
* **Fallback / Simplification**: Replace the multi-step chain with a clean graph of $f(t) = 0.8^t$ decaying to zero alongside a shrinking 2D arrow.

---

## Scene 07: Exploding Gradients & Norm Clipping Geometry
* **ID**: `scene07`
* **Title**: Exploding Gradients & Norm Clipping Geometry
* **Tier**: Tier 2 | **Target Duration**: 45s | **Risk**: Medium
* **The ONE Idea**: When spectral norms exceed 1, gradients grow exponentially, catapulting parameters across loss cliffs; global norm clipping fixes this by rescaling length while preserving direction.
* **Knowledge-Base References**: `RNN Video/modules/11_exploding_gradients.md`, `deep_dives/gradient_clipping_geometry/README.md`.
* **On-Screen Objects & Components**:
  * Growing red gradient arrow stretching off-screen.
  * Threshold circular ring of radius $\tau$.
  * Clipped vector projection $\mathbf{g}_{\text{clipped}} = \mathbf{g} \frac{\tau}{\|\mathbf{g}\|_2}$.
  * 2D Loss surface contour showing parameter descent path.
* **Layout**:
  * Left: Contour map of steep loss ravine.
  * Right: 2D vector space with threshold boundary circle centered at $(3, 0)$.
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–12s, 12s)**: Set transition norm $\|W_{hh}\|_2 > 1$. Backward pulse grows exponentially: $1 \to 3 \to 9 \to 81$, bursting outside the canvas. Red `NaN` warning flashes.
  2. **Beat 2 (12–22s, 10s)**: Show the catastrophic result on optimization: an unclipped gradient step catapults parameters out of the optimal valley.
  3. **Beat 3 (22–34s, 12s)**: Introduce Pascanu's Norm Clipping: draw a circular threshold boundary of radius $\tau$. When the gradient vector extends beyond $\tau$, project its tip back to the circle.
  4. **Beat 4 (34–45s, 11s)**: Emphasize geometric invariance: the vector's direction is preserved exactly; only its step magnitude is safely constrained.
* **Narration Cues**:
  * *"If the multiplier exceeds one, the opposite disaster occurs: gradients explode exponentially. But unlike vanishing gradients, exploding gradients have an elegant geometric fix..."*
* **Fallback / Simplification**: Omit the 2D contour terrain and illustrate clipping entirely as a vector shrinking to a circle of radius $\tau$.

---

## Scene 08: Architectural Evolution: The LSTM Linear Highway
* **ID**: `scene08`
* **Title**: Architectural Evolution: The LSTM Linear Highway
* **Tier**: Tier 2 | **Target Duration**: 40s | **Risk**: Medium
* **The ONE Idea**: LSTMs prevent vanishing gradients by routing memory through an additive cell state highway $\mathbf{C}_t$, replacing multiplicative Jacobian chains with gated linear additions.
* **Knowledge-Base References**: `RNN Video/modules/16_lstm_and_gru.md`, `spec/rnn_animation_spec.md` (Sec 9).
* **On-Screen Objects & Components**:
  * Dual-channel cell diagram: upper highway ($\mathbf{C}_t$) and lower recurrent channel ($\mathbf{h}_t$).
  * Forget gate valve ($f_t \odot \mathbf{C}_{t-1}$) and input addition ($i_t \odot \tilde{\mathbf{C}}_t$).
  * Clean unhindered backward gradient pulse gliding through the upper pipeline without decay.
* **Layout**:
  * Two horizontal parallel pipelines across center screen: upper cyan pipe ($\mathbf{C}_t$), lower muted pipe ($\mathbf{h}_t$).
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–10s, 10s)**: Revisit the standard RNN: repeated non-linear transformations squash and decay gradients.
  2. **Beat 2 (10–22s, 12s)**: Introduce the LSTM cell state $\mathbf{C}_t$: a dedicated linear highway isolated from recurrent matrix squashing.
  3. **Beat 3 (22–32s, 10s)**: Animate the additive update rule: $\mathbf{C}_t = f_t \odot \mathbf{C}_{t-1} + i_t \odot \tilde{\mathbf{C}}_t$. When forget gate $f_t \approx 1$, the past is preserved untouched.
  4. **Beat 4 (32–40s, 8s)**: Animate backward BPTT through the cell state: gradient flows across 50 steps with zero decay—the Constant Error Carousel.
* **Narration Cues**:
  * *"Instead of forcing memory through repeated matrix squashing, LSTMs construct an unhindered linear highway. When the forget gate is open, gradients flow across dozens of timesteps unchanged..."*
* **Fallback / Simplification**: Focus entirely on the additive formula $\mathbf{C}_t = \mathbf{C}_{t-1} + \Delta \mathbf{C}$ without detailing all 4 gate inner equations.

---

## Scene 09: Geometric Phase Space & SVD Contraction
* **ID**: `scene09`
* **Title**: Geometric Phase Space & SVD Contraction
* **Tier**: Tier 3 (Optional / Fallback) | **Target Duration**: 15s | **Risk**: High
* **The ONE Idea**: Under singular value decomposition, recurrent transformations rotate and compress the unit sphere of latent states along principal axes, visually demonstrating why arbitrary vectors contract into thin manifolds.
* **Knowledge-Base References**: `RNN Video/modules/09_matrix_jacobian_view.md`, `modules/14_3d_rnn_representation.md`.
* **On-Screen Objects & Components**:
  * 2D / 3D unit circle or sphere representing uniform perturbations $\|\Delta \mathbf{h}\| = 1$.
  * Transformation sequence: rotation $\mathbf{V}^\top$, stretching along singular axes $\sigma_1, \sigma_2$, rotation $\mathbf{U}$, and squashing by $\mathbf{D}_t$.
  * Flattened contracting ellipse.
* **Layout**:
  * Centered 2D coordinate grid with unit circle morphing into an elongated ellipse.
* **Animation Sequence (Beats)**:
  1. **Beat 1 (0–7s, 7s)**: Unit circle transforms under $\mathbf{W}_{hh}$ into an ellipse with semi-axes determined by singular values $\sigma_1, \sigma_2$.
  2. **Beat 2 (7–15s, 8s)**: Subsequent application of activation derivative $\mathbf{D}_t$ compresses the ellipse further into a thin needle. Conclude with summary caption.
* **Narration Cues**:
  * *"Geometrically, the recurrent Jacobian reshapes the space of hidden states. Repeated transformations collapse the volume of perturbations, sealing the fate of early information."*
* **Fallback / Simplification**: Use a simple 2D circle-to-ellipse transformation rather than a full 3D camera orbit.
