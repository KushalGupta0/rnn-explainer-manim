# Spoken Narration: Scene 15 — Final Synthesis: Learning Across Time

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~45 seconds.

---

### [Beat 1: The Arc of Recurrence] (0:00 - 0:12 | 12s)
We began with a fundamental question: how can neural networks process variable sequences with finite parameters? The answer was recurrence: an elegant, stationary transformation reused at every tick of time: $h_t = f(x_t, h_{t-1})$.

### [Beat 2: The Computational Chain] (0:12 - 0:24 | 12s)
Unrolling this recurrence across time revealed its dual nature: a directed acyclic graph where forward signals accumulate context, but where training requires backpropagation through long, fragile chains of transition Jacobians.

### [Beat 3: The Geometric Crucible] (0:24 - 0:35 | 11s)
Repeated multiplication across time is an unforgiving geometric filter. If spectral norms exceed one, gradients detonate; if they fall below one, memory vanishes into numerical silence, blind to distant history.

### [Beat 4: The Core Takeaway] (0:35 - 0:48 | 13s)
From gradient clipping to the additive linear highways of LSTM and GRU, the history of recurrent deep learning is the quest to let gradients survive time. Learning across time is learning through repetition—and memory is an unbroken gradient path.
