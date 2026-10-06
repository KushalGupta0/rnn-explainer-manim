# Module 01: Why RNNs Exist — The Problem of Sequential Data

[Back to Master Index](../index.md)

---

## 1. The Fundamental Problem: Sequential Data vs. Feed-Forward Networks

A standard Feed-Forward Neural Network (Multilayer Perceptron / MLP) computes a static map:
$$\hat{y} = \mathcal{F}_\theta(x)$$
where the domain and co-domain vector spaces are fixed: $x \in \mathbb{R}^{D_{\text{in}}}$ and $\hat{y} \in \mathbb{R}^{D_{\text{out}}}$.

In real-world physical and symbolic processes, data does not arrive as isolated, fixed-dimensional vectors. It arrives as **sequences** ordered by an index (often time $t \in \mathbb{N}$ or position in text):
$$x_1, x_2, \dots, x_T$$

### Concrete Sequential Modalities
1. **Natural Language / Text**:
   * Elements: $x_t \in \mathbb{R}^{d_x}$ are word embeddings or one-hot token vectors.
   * Semantic dependency: `"The flight from Paris arrived in France on time."`
     To resolve pronouns, grammar, or sentiment, the meaning of a word at step $t$ is fundamentally dependent on words at $t - 10$ or $t - 50$.
2. **Speech / Audio**:
   * Elements: $x_t \in \mathbb{R}^{d_x}$ are acoustic feature frames (e.g., MFCCs or Mel-spectrogram slices sampled every 10 ms).
   * Temporal dynamics: A phoneme is not a static point; it is a trajectory of acoustic frequencies over time.
3. **Time-Series / Sensor & Financial Dynamics**:
   * Elements: $x_t \in \mathbb{R}^{d_x}$ are multivariate sensor readings or asset prices.
   * Non-Markovian history: Predicting $x_{t+1}$ requires knowing not just the current price $x_t$, but the momentum, moving averages, and volatility regimes established across $x_1, \dots, x_{t-1}$.

---

## 2. Mathematical Failure Modes of Ordinary Feed-Forward Networks

Why can't we simply feed sequential data into an ordinary MLP?

```
Naive MLP approach:
Concatenate: [x₁, x₂, x₃, ..., x_T] → [ W ∈ ℝ^{D_out × (T·d_x)} ] → y
```

### Failure Mode A: The Fixed-Dimension Constraint ($T$ is Variable)
A standard linear transformation $z = W x + b$ requires that the column dimension of $W$ equals the dimension of $x$.
* In sequential tasks, the sequence length $T$ is **variable**: one sentence has 4 words ($T=4$), another has 38 words ($T=38$).
* An MLP cannot natively accept an input vector of fluctuating dimensionality without artificial hacks (zero-padding or truncation).
* Padding to a fixed $T_{\max}$ creates enormous memory waste ($\mathcal{O}(T_{\max} \cdot d_x)$) and **strictly fails to generalize** to any sequence where $T > T_{\max}$.

### Failure Mode B: Lack of Temporal Translation Invariance (Weight Sharing Failure)
In an MLP with flattened input $x_{\text{flat}} = [x_1^\top, x_2^\top, \dots, x_T^\top]^\top$, the weight matrix partitions into $T$ separate column blocks:
$$W = \begin{bmatrix} W_{(1)} & W_{(2)} & \dots & W_{(T)} \end{bmatrix}$$
where $W_{(k)} \in \mathbb{R}^{D_{\text{out}} \times d_x}$ interacts exclusively with the token at position $k$.

* If the phrase `"deep learning"` appears at the start of a sentence ($t=1, 2$), it activates $W_{(1)}$ and $W_{(2)}$.
* If the exact same phrase appears at the end of a sentence ($t=15, 16$), it activates $W_{(15)}$ and $W_{(16)}$.
* **The MLP has zero temporal translation invariance**: it must learn the semantic meaning of `"deep learning"` independently at every conceivable index $t$.
* Parameter count explodes as $\mathcal{O}(T \cdot d_x \cdot D_{\text{out}})$.

### Failure Mode C: Arbitrary Context Horizons & Pooling Degradation
If we try to fix the dimension by averaging tokens ($x_{\text{avg}} = \frac{1}{T}\sum_{t=1}^T x_t$):
* Order is completely lost because vector addition is commutative:
  $$\vec{x}_{\text{man}} + \vec{x}_{\text{bites}} + \vec{x}_{\text{dog}} \equiv \vec{x}_{\text{dog}} + \vec{x}_{\text{bites}} + \vec{x}_{\text{man}}$$
* Permutation destroys the causal syntax of language, music, and time series.

> **Related Deep Dives:**
> * [Variable-Length Sequences & Weight Invariance](../deep_dives/variable_length_sequences/README.md)
> * [Markovian Models vs. Latent Autoregressive Recurrence](../deep_dives/markov_vs_recurrent/README.md)

---

## 3. The Sequential Abstraction

To process sequences properly, we must reformulate our computational paradigm from static batch mapping to **sequential recursive processing**.

Let an input sequence be an ordered tuple of vectors:
$$x_1, x_2, \dots, x_t, \dots, x_T \quad \text{where each } x_t \in \mathbb{R}^{d_x}$$

Processing the sequence **sequentially** means:
1. We do not ingest the entire sequence simultaneously in a single static step.
2. We feed the model one token vector $x_t$ at a time, stepping $t = 1, 2, \dots, T$.
3. At each step $t$, the model must maintain an internal representation of what it has observed so far.

---

## 4. The Hidden State: $h_t = f(x_t, h_{t-1})$

To retain past context without storing the raw history $x_1, \dots, x_{t-1}$ (which grows unboundedly with $t$), we introduce an internal summary vector called the **hidden state**:
$$h_t \in \mathbb{R}^{d_h}$$

The fundamental recurrence relation governing an RNN is:
$$\mathbf{h_t = f(x_t, h_{t-1})}$$

```
                x_t  (New sensory input at time t)
                 │
                 ▼
h_{t-1} ──────► [ f ] ──────► h_t  (Updated latent summary)
(Past summary)
```

### Precise Mathematical Deconstruction of $h_t = f(x_t, h_{t-1})$:
* **$x_t \in \mathbb{R}^{d_x}$**: The external input vector at the current timestep $t$. It provides purely *local, instantaneous* information.
* **$h_{t-1} \in \mathbb{R}^{d_h}$**: The latent state vector accumulated from step $0$ up to step $t-1$. It encodes a lossy, continuous summary of the prefix history $(x_1, \dots, x_{t-1})$.
* **$f: \mathbb{R}^{d_x} \times \mathbb{R}^{d_h} \to \mathbb{R}^{d_h}$**: A parameterized non-linear transition function (governed by weights that are invariant to $t$).
* **$h_t \in \mathbb{R}^{d_h}$**: The updated latent state. It represents the *new synthesis* combining the historical context with the fresh observation.
* **Initialization $h_0 \in \mathbb{R}^{d_h}$**: The base state before any data has arrived, typically set to $\mathbf{0}$ or learned via backpropagation.

> **Related Deep Dive:**
> * [Hidden State Dynamics & Latent State Spaces](../deep_dives/hidden_state_dynamics/README.md)

---

## 5. The Complete Input-State-Output Relationship

How does the network produce predictions?

At any timestep $t$, the hidden state $h_t$ is decoded to produce an output vector $y_t$ (or prediction $\hat{y}_t$):
$$y_t = g(h_t)$$
where $g: \mathbb{R}^{d_h} \to \mathbb{R}^{d_y}$ is an output projection operator (typically a linear layer followed by softmax or identity).

```
                      y_t  (Task Prediction / Output)
                       ▲
                       │ g(·)
                       │
x_t ───────────────► [ h_t ] ───────────────► h_{t+1}
(Current Input)        ▲                      (Passed to future)
                       │
                     h_{t-1}
                  (Past Context)
```

### The Exact Interaction Triangle:
1. **$x_t$ (The Disturbance / Input)**: Injects new information into the latent system.
2. **$h_{t-1}$ (The Memory / Past Context)**: Bridges the entire temporal history from $t=1$ to $t-1$.
3. **$h_t$ (The Latent Pivot)**: Merges past ($h_{t-1}$) and present ($x_t$). It serves two simultaneous roles:
   * **Downstream Output Source**: Feeds into $g(h_t)$ to generate immediate prediction $y_t$.
   * **Temporal Relay**: Passes forward in time to become the past context $h_{(t+1)-1}$ for the next step.
4. **$y_t$ (The Observable Output)**: Derived solely from $h_t$. Notice that $x_t$ does not need to connect directly to $y_t$ (though skip connections exist in variants); $h_t$ acts as the complete informational bridge.

### Canonical Sequence Topologies (The Four Archetypes):
1. **One-to-Many (Generative / Decoding)**:
   * Single static input vector $x$ (or image embedding) initializes or conditions the recurrence; model emits a sequence: $y_1, y_2, \dots, y_T$.
   * *Example*: Image captioning ($x = \text{CNN}(I) \implies y = \text{"A black cat sitting on a mat"}$).
2. **Many-to-One (Sequence Classification / Aggregation)**:
   * Full input sequence $x_1, \dots, x_T$ is processed; intermediate outputs are discarded; only the final latent state $h_T$ produces prediction:
     $$y = g(h_T)$$
   * *Example*: Sentiment analysis, video action recognition, document topic assignment.
3. **Many-to-Many: Synchronized / Frame-by-Frame (Per-Step Sequence Tagging)**:
   * Input length equals output length ($T_x = T_y$). At every timestep $t$, an immediate prediction is produced:
     $$y_t = g(h_t)$$
   * *Example*: Part-of-speech (POS) tagging, named entity recognition (NER), acoustic frame alignment.
4. **Many-to-Many: Asynchronous / Sequence-to-Sequence (Encoder-Decoder)**:
   * Input sequence $x_1, \dots, x_{T_x}$ is encoded into a summary state $h_{T_x}$, which then primes a second decoder RNN producing $y_1, \dots, y_{T_y}$ where $T_x \ne T_y$.
   * *Example*: Machine translation (English sentence to French sentence).

---

## 6. Visual Semantics for Manim: Part I Mental Models

When designing a 3Blue1Brown-style video, how should we translate Part I onto the screen?

| Mathematical Object | Computational Interpretation | Geometric Interpretation | 2D Animation Design | 3D Animation Design |
| :--- | :--- | :--- | :--- | :--- |
| **Sequence $(x_1, \dots, x_T)$** | Stream of incoming data packets. | Discrete vectors in input space $\mathbb{R}^{d_x}$. | Horizontal timeline with sliding pointer highlight. | Helical ribbon marching along the time axis $Z$. |
| **Transition $f(x_t, h_{t-1})$** | Recursive state update function. | Coordinate warping + vector addition. | Two vectors ($h_{t-1}$ and $x_t$) merging into a node and stretching. | Vector $h_{t-1}$ rotated by matrix $W_h$, translated by $W_x x_t$, then bounded to $[-1,1]^3$. |
| **Hidden State $h_t$** | Continuous contextual summary vector. | A point/vector tracing a trajectory in latent space $\mathbb{R}^{d_h}$. | An arrow in a 2D plane whose tip traces a continuous path over time steps. | A glowing particle tracing a 3D phase-space trajectory as steps advance. |
| **Output $y_t = g(h_t)$** | Prediction decoded from latent state. | Projection from latent manifold $\mathbb{R}^{d_h} \to \mathbb{R}^{d_y}$. | Arrow projecting upward from $h_t$ to an output node. | Ray projecting from the 3D latent particle onto a 2D decision plane. |

### Visual Misconceptions to Avoid in Animation:
1. **Misconception**: Visualizing $h_t$ as a growing box or a physical bucket storing token icons.
   * *Correct visual*: $h_t$ is a fixed-size vector whose values shift and mutate. It does not grow larger; its internal coordinates shift.
2. **Misconception**: Treating the time arrow as a physical neural layer.
   * *Correct visual*: Make clear that the recurrent cell is a **single machine** being queried repeatedly across time.

---

## 📝 Checkpoint 01 Exercises

### Level 1 — Conceptual
Explain why an ordinary Multilayer Perceptron cannot natively accept a dataset containing sentences ranging from 3 words to 50 words without modifying its input layer or introducing padding.

### Level 2 — Mathematical
Let $h_0 = \mathbf{0} \in \mathbb{R}^2$. Let the transition function be linear for simplicity:
$$h_t = W_h h_{t-1} + W_x x_t$$
where $W_h = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}$ and $W_x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$.
Given inputs $x_1 = [1]$, $x_2 = [0]$, $x_3 = [2]$:
Compute the exact coordinates of $h_1, h_2,$ and $h_3$.

### Level 3 — Debugging / Misconception
A junior engineer claims:
*"To handle variable sequence length $T$ in an MLP, we can simply average all input vectors: $x_{\text{input}} = \frac{1}{T}\sum_{t=1}^T x_t$. This gives a fixed vector of dimension $d_x$, solving the variable length problem completely without needing an RNN."*
Identify the fundamental mathematical and structural failure in this reasoning.

### Level 4 — Visualization (Manim Translation)
Describe how you would animate the transition between a single recurrent loop (a node feeding back into itself) and an unrolled timeline of states in Manim. What visual cue would you use to show that the parameters are identical across time?

### Level 5 — Advanced (Dynamical Systems)
View $h_t = f(x_t, h_{t-1})$ as a discrete dynamical system. If $x_t = \mathbf{0}$ for all $t > 5$, what mathematical condition on $f$ determines whether the system approaches a fixed point $h^* = f(\mathbf{0}, h^*)$ or diverges?

