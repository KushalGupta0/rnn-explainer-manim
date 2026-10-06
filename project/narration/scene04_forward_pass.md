# Spoken Narration: Scene 04 — The Forward Pass: Numerical State Caching

**Pacing**: Calm, deliberate, conversational. Target duration: 40s (92 words | 138.0 WPM).

---

### [Beat 1: Initial State & Step One] (0:00 - 0:10 | 10s)
We begin with an initial memory vector $h_0$ at zero. The first input $x_1$ enters, producing the first hidden state $h_1$.

### [Beat 2: Compounding Context] (0:10 - 0:20 | 10s)
At step two, $h_1$ travels rightward, combining with $x_2$ to create $h_2$, which now compresses information from both inputs.

### [Beat 3: Freezing the Activation Cache] (0:20 - 0:30 | 10s)
During training, every intermediate state must be cached in memory. Discarding them would make backpropagation impossible.

### [Beat 4: Sequence Loss] (0:30 - 0:40 | 10s)
Each step emits a prediction compared against ground truth, accumulating local errors into the total sequence loss $L$.
