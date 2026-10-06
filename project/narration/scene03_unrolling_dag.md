# Spoken Narration: Scene 03 — Unrolling Through Time: Loop vs. DAG

**Pacing**: Calm, deliberate, conversational. Target duration: 40s (92 words | 138.0 WPM).

---

### [Beat 1: The Compact Feedback Loop] (0:00 - 0:10 | 10s)
Recurrent networks are often drawn as a single cell with a recursive loop, feeding its hidden state back into itself as time advances.

### [Beat 2: Snapping Open the Chain] (0:10 - 0:21 | 11s)
Unspooling this loop across time reveals a straight horizontal chain: a directed acyclic graph where each step represents a discrete moment.

### [Beat 3: Parameter Sharing] (0:21 - 0:31 | 10s)
Crucially, this is not a standard deep network with different weights per layer. The transition matrix $W_{hh}$ is identical at every timestep.

### [Beat 4: Stationarity in Action] (0:31 - 0:40 | 9s)
Sharing parameters keeps model size constant and guarantees that the rules of memory do not change over time.
