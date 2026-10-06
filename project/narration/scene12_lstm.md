# Spoken Narration: Scene 12 — Long Short-Term Memory (LSTM): The Linear Additive Highway

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~55 seconds.

---

### [Beat 1: The Architectural Dilemma Solved] (0:00 - 0:12 | 12s)
To escape the vanishing gradient trap, Hochreiter and Schmidhuber reimagined recurrent memory in 1997. Instead of forcing memory through repeated matrix multiplications, the Long Short-Term Memory network introduces an unhindered linear highway: the cell state, $C_t$.

### [Beat 2: The Two Channels & The Forget Gate] (0:12 - 0:25 | 13s)
The cell state runs horizontally across time along the top of the cell. Before modifying it, the network computes a forget gate vector $f_t$ using a sigmoid activation between zero and one. When $f_t$ equals one, the past cell state passes through completely intact; when $f_t$ is zero, old memory is wiped clean.

### [Beat 3: Additive Writing: Input Gate & Candidate] (0:25 - 0:38 | 13s)
Next, the network generates new candidate information $\tilde{C}_t$ via tanh, and scales it by an input gate $i_t$. The update to memory is strictly additive: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. No matrix product squashes the stored values—new facts are simply added onto the conveyor belt.

### [Beat 4: Output Gate & The Constant Error Carousel] (0:38 - 0:55 | 17s)
Finally, an output gate $o_t$ filters the squashed cell state to emit the visible hidden state $h_t = o_t \odot \tanh(C_t)$. In backpropagation, gradients travel backward along the cell state highway with Jacobian approximately equal to $f_t$. When the forget gate is open, gradients flow across hundreds of timesteps without decaying a single percent.
