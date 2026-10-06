# Spoken Narration: Scene 01 — Why RNNs Exist: The Sequence Challenge

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~40 seconds.

---

### [Beat 1: The Rigid Feedforward Grid] (0:00 - 0:10 | 10s)
Standard neural networks are remarkably capable, but they are built with a rigid assumption: their inputs must have a fixed, predetermined size. When we pass a three-word phrase like "clouds in sky", each word maps cleanly to dedicated input neurons.

### [Beat 2: The Variable-Length Crisis] (0:10 - 0:22 | 12s)
But real-world data rarely obeys fixed dimensions. Sentences expand, audio recordings vary in duration, and financial signals unfold over unpredictable intervals. If our input grows to six words, the architecture breaks down: extra tokens overflow the rigid boundary, leaving the model stranded.

### [Beat 3: The Position Fallacy] (0:22 - 0:32 | 10s)
Even if we padded our inputs to an arbitrary length, feedforward networks suffer from another fundamental limitation: they lack translation invariance across time. A critical word appearing at the beginning of a sentence activates completely different weights than the exact same word appearing at the end.

### [Beat 4: The Recurrent Solution] (0:32 - 0:40 | 8s)
To process sequences naturally, we need a model that processes tokens one step at a time, carrying forward a compact mathematical memory of everything it has seen so far.
