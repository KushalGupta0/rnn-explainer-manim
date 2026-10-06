# Spoken Narration: Scene 01 — Why RNNs Exist: The Sequence Challenge

**Pacing**: Calm, deliberate, conversational. Target duration: 42s (97 words | 138.6 WPM).

---

### [Beat 1: The Fixed-Dimension Interface] (0:00 - 0:10 | 10s)
Feedforward networks require fixed-dimension inputs. A six-token sentence like "The cat sat on the mat" maps to a dedicated weight matrix of size d-out by six d-x.

### [Beat 2: The Dimensional Mismatch] (0:10 - 0:21 | 11s)
Extend the sequence to eight tokens, and the interface breaks down: an eight-d-x vector cannot multiply a matrix expecting six dimensions.

### [Beat 3: Architecture Compression] (0:21 - 0:32 | 11s)
Instead of resizing matrices for every sequence length, we compress the model into a single recurrent operator: h-t equals f of x-t and h-t-minus-one.

### [Beat 4: In-Place Sequential Ingestion] (0:32 - 0:42 | 10s)
The exact same cell remains in place, sequentially ingesting x-one, x-two, and x-three while evolving its internal state across time.
