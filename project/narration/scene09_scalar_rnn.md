# Spoken Narration: Scene 09 — The Scalar RNN: Under the Mathematical Microscope

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~45 seconds.

---

### [Beat 1: The One-Dimensional Lens] (0:00 - 0:11 | 11s)
To understand what is happening inside complex high-dimensional weight matrices, we place the recurrent cell under a mathematical microscope. We strip away all vector spaces and reduce the RNN to a single scalar dimension: $h_t = \tanh(w h_{t-1} + u x_t + b)$.

### [Beat 2: Concrete Numerical Forward Steps] (0:11 - 0:23 | 12s)
Let's trace this computation with concrete numbers. We set recurrent weight $w = 0.8$, input weight $u = 0.5$, bias $b = 0$, and start from $h_0 = 0.5$. With an input stream of $x_t = 0.2$, each timestep computes a clear scalar state: $h_1 = \tanh(0.50) \approx 0.46$, followed by $h_2 = \tanh(0.47) \approx 0.44$.

### [Beat 3: Elementary Chain Rule Derivation] (0:23 - 0:34 | 11s)
Now, let's take the derivative. By elementary single-variable calculus, the sensitivity of $h_t$ to $h_{t-1}$ is simply: $\frac{\partial h_t}{\partial h_{t-1}} = w \cdot (1 - h_t^2)$. Across $T - t$ temporal steps, the chain rule multiplies these scalar factors: $\frac{\partial h_T}{\partial h_t} = w^{T-t} \prod \phi'(a_k)$.

### [Beat 4: The Bridge to Matrices] (0:34 - 0:45 | 11s)
This scalar product is the exact algebraic foundation of the RNN dilemma. When we generalize back to multi-dimensional vectors, this single number $w$ will become a weight matrix $W_{hh}$, and the scalar derivative will expand into a Jacobian matrix.
