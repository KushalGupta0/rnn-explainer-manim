# Spoken Narration: Scene 05 — Backpropagation Through Time: Gradient Flow

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~45 seconds.

---

### [Beat 1: The Backward Inversion] (0:00 - 0:12 | 12s)
Once the forward pass reaches the final timestep, time must run in reverse. To compute gradients with respect to our shared parameters, error signals must travel backward through the unrolled computational graph—an algorithm known as Backpropagation Through Time.

### [Beat 2: The Temporal Adjoint] (0:12 - 0:24 | 12s)
Look at the error ignited at step three. Its gradient flows backward into state $h_3$, then passes across the horizontal connection into state $h_2$. Mathematically, this transition multiplies the incoming gradient by the transpose of the recurrent weight matrix, scaled by the local derivative of the activation function.

### [Beat 3: Convergence at the Nexus] (0:24 - 0:35 | 11s)
At state node $h_2$, notice the critical intersection: two separate streams of error converge. One stream descends vertically from the immediate local loss at step two; the other stream arrives horizontally from the future. By the multivariable chain rule, these two gradient vectors simply add together.

### [Beat 4: Accumulating Weight Gradients] (0:35 - 0:45 | 10s)
Because the recurrent matrix $W_{hh}$ was used at every single timestep, its final gradient update is the cumulative sum of outer products across the entire sequence. Every timestep casts its vote on how the weights should evolve.
