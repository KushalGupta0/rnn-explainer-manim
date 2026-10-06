1. The installed Manim version is the source of truth. Never write a Manim API call from memory. Look it up with tools/api.py first. If a name does not exist, do not use it.
2. Keep three layers strictly separate:
   - manim-grounding/ : generic skill and tools, containing NOTHING about RNNs or neural networks
   - lib/ : reusable, topic-neutral components (labeled box, node, layer of nodes, connector arrow, matrix grid, caption bar, palette)
   - project/ : everything specific to this video
   Sorting test: true for any Manim video = skill; reusable code not tied to the topic = lib; about RNNs = project.
3. Never edit the existing skill folder or my knowledge base. Create new folders only.
4. Do not read the whole knowledge base. Read index.md and rnn_animation_spec.md, then only the modules a scene needs (modules 13 and 14 hold the visual translation). Spot-check derivations before animating them, and tell me about any error you find.
5. Never claim something works unless you ran it. Report exact commands and results.
6. Time-box every phase. If something takes too long, simplify it. A finished essential video beats an unfinished complete one.
7. After each phase, give a 5-line status report, and update PROGRESS.md (what is done, what is next, known problems) so a fresh session can continue.
