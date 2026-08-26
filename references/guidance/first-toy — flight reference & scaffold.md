# first-toy — flight reference & scaffold

**How to use this.** It's a map and a set of prompts — _not_ a solution manual. There is no Rust in here on purpose; the code is yours to write with an editor in front of you. Read a section → close it → produce something (a paper derivation, a shape trace, a decision in STATUS.md, a passing test). If a section can be consumed without producing, it's being read wrong.
The later stages are deliberately lighter than the near ones. That's not an omission — abstractions earn their place once you have two concrete instances. Orientation + gate, not detail.

* * *

## 0. Where you are right now (Stage A — refactor for backprop readiness)

You've drafted the modules: `activation` (enum), `layer`, `cache`, `network`. The data shapes are settled:

*   `Layer` = { weights: (out×in), biases: (out,), activation: enum }
    
*   `Entry` = { input, pre_activations } — one per layer
    
*   `ForwardCache` = { entries: Vec }
    
*   `forward` returns **(output, cache)**
    

**What's left to hit the Stage A gate:**

1.  **Return type.** `forward` must return the final _output_ together with the cache — not just the cache. The last `per_layer_input` at the end of the loop _is_ the output; don't let it vanish.
    
2.  **Cache API.** The `cache` fields are private, so `network` can't build them. Decide the surface: constructors + methods, or public fields. It must let the network (a) make an empty cache, (b) record one entry per layer.
    
3.  **Compile fixes.** Borrow the layers (`for layer in &self.layers`, don't move the Vec); initialize the cache via its constructor; push _one_ `Entry` (not two args); match the variable name you actually defined; close the open paren in the RELU arm.
    
4.  **First test.** Pin the forward pass to the hand-computed value (§1).
    
5.  **Thin** `main.rs`**.** Build network → call forward → print. Nothing else.
    
6.  **STATUS.md.** Stage + every decision and _why_.
    

**Gate =** `cargo test` **green on the pinned value.** Not "it compiles."

* * *

## 1. Your two anchor truths (ground truth — derived by hand)

If code ever disagrees with these, **the code is wrong**, not the numbers.

*   **Anchor 1:** input `[0,0,0,1]` [1] → output `[0.5, 0, 0.25, 0.25]`.  
    _Why:_ one-hot at index 3 → z1 = column 3 of W1 = `[0.5, −0.5]` [1] → ReLU → `[0.5, 0]` → z2 = 0.5 × row 0 of W1 = `[0.5, 0, 0.25, 0.25]`. The dead (2nd) hidden neuron contributed nothing.
    
*   **Anchor 2 (the falsifier for "activation by position"):** input `[0,1,0,0]` → 4th output = **−0.5** under identity, but **0** if you wrongly ReLU the output layer. This is _why_ the activation must be per-layer **data**, not a positional `if` [1].
    

* * *

## 2. Stage A — close the loop (sit-with questions)

*   Why does `forward` return (output, cache) rather than storing the cache on `Network`? What goes stale? What would `&mut` buy you that you don't need?
    
*   Why `(input, pre)` per entry and not `(pre, post)`? What _is_ a layer's post, and who consumes it?
    
*   Why is `Layer::forward` pure (self + input in, pair out)? What can you now test in isolation because of it?
    
*   What should the cache expose, and why that shape? (Your "small focused functions" preference is a hint toward the answer.)
    

* * *

## 3. Stage B — backpropagation _(the core — budget multiple sessions)_

**Concept work BEFORE code.** Don't write `backward` until you can do the following on paper.

*   Loss as a scalar function of the parameters. MSE first.
    
*   The chain rule as **local derivative × incoming gradient**. Walk ONE layer on paper.
    
*   **Shape discipline:** the gradient of the loss w.r.t. any tensor has _that tensor's_ shape.
    *   Your code computes `W.dot(x)`, so `W` is (out, in). **Predict the shapes of** `dW`**,** `db`**, and the upstream gradient _before_ writing anything.** In your own convention.
        
*   **Why the cache exists:** for each backprop formula, which cached value does it consume? (Pre-activation → the activation's derivative. The layer's input → the weight gradient.)
    
*   ReLU's derivative at 0 — notice it's a _convention choice_. What do you pick, and does it matter?
    

**Design decisions (you propose, then argue them):**

*   A gradients structure mirroring the parameters (per-layer `dW`, `db`).
    
*   Who owns `backward` — a method on `Network`? Does it take the cache by value or reference? Argue it.
    
*   The parameter update as a separate, dumb step: `p = p − lr·dp`. Keep the optimizer **out** of `backward`.
    

**THE GRADIENT CHECK — non-negotiable. This is your ground truth.**

*   Central difference: perturb ONE parameter by ±ε; numeric gradient = `(L₊ − L₋) / 2ε`.
    
*   Compare by relative error: `|analytic − numeric| / (|analytic| + |numeric| + tiny)`.
    
*   For f32, a passing threshold is _loose_ (~1e-2 to 1e-3) — f32 can't hit tight tolerances. (Numerics question worth sitting with: _why_?)
    
*   Check **every** parameter of a small random network, as a `cargo test`.
    
*   **RULE: no training run is trusted until the gradient check passes. If training "works" but the check fails, the check wins.**
    

**First training tasks:**

1.  The 4→2→4 autoencoder you already have, on one-hot inputs. **Predict before running:** can a 2-dim bottleneck reconstruct 4 one-hot vectors? What's the loss floor, and what does it mean?
    
2.  XOR with a small MLP — the "nonlinearity is necessary" demo. Predict what happens if you remove the activations.
    

Watch for: learning-rate blowups (let one happen, then discuss), loss plateaus from bad init, and yourself trusting a falling loss curve over the gradient check.
**Gate:** gradient check green in `cargo test`; both toy tasks trained with a logged loss curve; you can derive the one-layer backprop equations on paper, unprompted, no code (teach-back).

* * *

## 4. Stage C — batching + training infrastructure

*   **Batching refactor:** activations gain a batch axis. You pick the layout (batch×features vs features×batch) — **defend it**, because the choice ripples through every formula.
    
*   Sit with: why do gradients _average_ over the batch, and what does that do to learning-rate scaling?
    
*   **Re-run the gradient check AFTER the refactor** — that's the whole point of having it.
    
*   An optimizer struct (plain SGD first). A training-loop function: epochs, shuffling with your own RNG, loss logging. Keep it boring and small.
    

**Gate:** batched training reproduces Stage B results (same seeds, comparable curves); gradient check still green.

* * *

## 5. Stage D — sequence models & RNN _(orientation)_

*   Concept first: sequence modeling as **next-token prediction**; autoregressive factorization; cross-entropy loss; softmax (including the numerically stable form).
    
*   **Bridge task** before recurrence: a fixed-context next-character model (bigram / small-window MLP) on a tiny corpus — introduces embeddings, softmax, cross-entropy, and the sample loop with _zero_ new architecture.
    
*   **Then the RNN: this is where a TRAIT earns its place** — two layer kinds now exist (dense, RNN cell). Design what `forward`/`backward` must carry; where the cache/state lives.
    
*   **BPTT = the same cache idea, unrolled over time; gradients sum across steps into shared weights. Derive it — don't receive it.**
    

**Gate:** gradient check on the unrolled loss; generated samples; you can explain why the same `W` receives gradient from every time step.

* * *

## 6. Stage E — transformer _(orientation)_

*   Build and unit-test each component **standalone** before assembly: stable softmax, scaled dot-product attention (`QKᵀ/√d`), causal mask, multi-head split/merge, LayerNorm (the single hardest backward — gradient-check it in isolation), residual connections, the MLP block, learned positional embeddings.
    
*   Sit with: _why does attention need positional embeddings at all?_
    
*   **Task:** tiny char-level model on the same corpus; compare against the RNN on a copy task. Predict that attention wins — and say _why_.
    

**Gate:** per-component gradient checks; end-to-end tiny model trains; teach-back — trace one token's forward pass through one block, naming every tensor shape.

* * *

## 7. Cross-cutting protocol (the things that don't change)

*   **PREDICT-BEFORE-RUN:** before any experiment or test, state what you expect and _why_. The surprise is the teaching moment.
    
*   **STATUS.md every session:** current stage, decisions + why, open questions.
    
*   **Stage gates end in a TEACH-BACK:** explain the mechanism on paper (derivation / shape trace / design rationale) with no code in front of you. Don't advance on "it runs."
    
*   **The gradient check outranks a falling loss curve.** Always.
    
*   **Minimal dependencies:** ndarray only, until a new one is genuinely warranted and discussed.
    
*   Small focused functions; early returns over nesting; short comments that tell the story.
    

* * *

## 8. Sit-with questions (the deep ones — close the doc and think)

1.  **Tied weights.** `W2 = W1ᵀ` as a _copy_ means the two matrices train **independently** [1]. Do you actually _want_ tied weights? If yes, the gradient for `W` must sum both contributions. Which model are you building — and **where does the ReLU sit?** (Your current code: ReLU on the hidden layer, identity on the output [1].)
    
2.  What makes a cache a _tape_? Why is recording `(input, pre)` per layer enough to reconstruct everything `backward` needs?
    
3.  If f32 can't hit a tight gradient-check tolerance, is that a bug or a property of the precision? What would checking in f64 buy you, and cost?
    
4.  When does an enum stop being enough and a trait become right? (You'll hold two concrete layer kinds before you need to answer.)
    

* * *

## 9. The one rule for the whole arc

Understanding is built by _doing_, not by reading. This doc is a map and a set of prompts. Every section should end with you closing it and producing something. The flight is for thinking, not consuming — protect that.