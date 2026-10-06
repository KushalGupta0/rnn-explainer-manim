# Spoken Narration: Scene 02 — The RNN Cell: First Principles & Dimensions

**Pacing**: Calm, deliberate, conversational. Target duration: 48s (110 words | 137.5 WPM).

---

### [Beat 1: The Two Inflows] (0:00 - 0:12 | 12s)
Here is the recurrent cell. At each moment, it receives two signals: fresh external input, vector $x_t$, and a summary of past context, hidden state $h_{t-1}$.

### [Beat 2: Linear Superposition] (0:12 - 0:25 | 13s)
Weight matrix $W_{xh}$ projects the input, while $W_{hh}$ transforms past memory. Inside the cell, both transformed vectors add together with a bias to form the pre-activation $a_t$.

### [Beat 3: The Hyperbolic Squashing] (0:25 - 0:37 | 12s)
Next, the hyperbolic tangent squashes every component into the interval minus one to one, preventing forward values from blowing up.

### [Beat 4: Readout & State Emission] (0:37 - 0:48 | 11s)
The resulting vector $h_t$ emits an output prediction $\hat{y}_t$, and passes forward as memory for the next timestep.
