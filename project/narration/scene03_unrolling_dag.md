# Spoken Narration: Scene 03 — Unrolling Through Time: Loop vs. DAG

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~40 seconds.

---

### [Beat 1: The Compact Feedback Loop] (0:00 - 0:10 | 10s)
In textbooks, recurrent neural networks are often depicted as a single compact unit with a recursive self-loop. The hidden state leaves the cell, wraps back around, and re-enters itself as time ticks forward.

### [Beat 2: Snapping Open the Chain] (0:10 - 0:22 | 12s)
To understand how information truly flows, we can unspool this loop across time. What appeared to be a recursive circle is mathematically identical to a straight horizontal chain—a directed acyclic graph where each step represents a discrete moment in time.

### [Beat 3: The Power of Parameter Sharing] (0:22 - 0:32 | 10s)
Notice something profound here: this is not a traditional deep network with different weights at every layer. The transition matrix $W_{hh}$ applied at step one is identically the exact same matrix applied at step two, step three, and step one hundred.

### [Beat 4: Stationarity in Action] (0:32 - 0:40 | 8s)
This weight sharing is what gives the network its temporal translation invariance: the rule for updating memory remains constant, no matter how long the sequence stretches.
