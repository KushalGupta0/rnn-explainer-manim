# Spoken Narration: Scene 05 — Backpropagation Through Time: Gradient Flow

**Pacing**: Calm, deliberate, conversational. Target duration: 65s (142 words | 131.1 WPM).

---

### [Beat 1: The Backward Inversion] (0:00 - 0:15 | 15s)
Once the forward pass reaches the final step, time runs in reverse. To train shared parameters, error signals flow backward through the unrolled computational graph: Backpropagation Through Time.

### [Beat 2: The Temporal Adjoint] (0:15 - 0:32 | 17s)
Watch the error from step three. Its gradient enters state $h_3$, then travels across the horizontal connection into $h_2$. Mathematically, this multiplies the incoming gradient by the transpose weight matrix $W_{hh}^\top$, scaled by the local activation derivative $D_3$.

### [Beat 3: Convergence at the Nexus] (0:32 - 0:49 | 17s)
At state node $h_2$, two error streams meet: the local loss descending vertically from step two, and the accumulated future error arriving horizontally. By the multivariable chain rule, they simply add together.

### [Beat 4: Accumulating Weight Gradients] (0:49 - 1:05 | 16s)
Because matrix $W_{hh}$ was shared across all steps, its final update is the sum of outer products across the entire sequence. Every timestep casts a vote on how weights evolve.
