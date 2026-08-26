# Stage B Working Doc — Backprop for the 4→2→4

Purpose: enough structure to write real backprop code offline. Rules of the doc:
math sections are scaffolds to complete, code sections are specs + decisions to make,
shape statements are self-checks to verify against your derivation — never the other way around.

## 0. Preflight (30 min, do this first)

Two owed answers from the last session. Write them here.

1. In `forward_pass` [1], the cache entry is built from `layer_input` BEFORE
   `layer_input` is updated to the layer's post-activation. If that order were
   swapped, what would entry zero's `input` slot hold? And does the pair stored in
   that entry then describe a computation the first layer actually performed?
   (Backward is the consumer that this ordering protects — you'll see it in §2.)
2. Entry zero's `input` slot holds the network input itself [4]. The pre-refactor
   single-file design kept two parallel Vecs and needed a dummy zeros entry at
   index 0 to line them up. In one sentence: what did the Entry design do to that dummy?

## 1. Conventions (read, don't re-derive)

- A layer computes z = W·x + b, where W is shaped (out, in). The forward pass
  returns (pre_activation, post_activation) [2]; activation is per-layer data [5].
- The cache: one Entry per layer holding (input, pre_activation); post is NOT
  stored because a layer's post IS the next layer's input — each boundary value
  is stored once, in the entry that consumes it [4]. The final post is the
  output, returned separately in NetworkOutput [1].
- Forward + derivative arrive as a pair on the Activation enum: one exhaustive
  match, compiler enforces the pairing [5].
- f32 throughout. Expect ~1e-7 scale noise in any comparison; exact equality is
  only safe for values you proved bit-exact.

## 2. Math — derive on paper, fill in here

Network: 4→2→4. W1 (2×4), b1 (2,), W2 (4×2), b2 (4,).
Layer 1: ReLU. Layer 2: use IDENTITY for the derivation — it sidesteps the
kink, and it's what the original program used. (Your pinned test uses ReLU on
layer 2; note why that's a problem for differentiating at the pinned point,
where z2 contains an exact 0.0 [1].)

### 2a. The loss

Write L as a scalar function of (W1, b1, W2, b2) for a chosen input x and
target t. Decisions to make and record:

- MSE is the obvious first candidate. Why — what property must the loss have
  for gradient descent to work at all, and does this network admit the
  alternatives (what does the output layer actually produce — values or
  probabilities)?
- The constant in front of the sum: make it deliberate. What does it buy you
  the moment you differentiate?
- TRAP: the pinned pair (input [0,0,0,1], target [0.5, 0, 0.25, 0.25]) [1]
  gives L = 0 and every gradient zero. Pick a different one-hot input and a
  target where L ≠ 0, and state what in your choice guarantees that.
- Warm-up: compute L by hand for your chosen x, t at the test weights. This
  number is later ground truth.

### 2b. Last layer (output layer), fill in each line

Definitions: z2 = W2·x2 + b2, output y = z2 (identity), x2 = post-activation of
layer 1. You are given the incoming gradient at the output:
dL/dy, with shape (4,) — for MSE that's output − target (up to your constant).
Everything below derives FROM that, in four chain-rule steps.

1. Let δ2 := dL/dz2. Derive δ2 in terms of y and t. (One line.)
2. dL/db2 = ?   (Look at your MSE sum: which term contains b2[i]?)
   Self-check: shape (4,).
3. dL/dW2 = ?   (The j-th row of W2 appears in exactly one component of z2 —
   which one? Differentiate that component.)
   Self-check: shape (4×2) — same shape as W2.
4. The previous layer needs dL/dx2. Derive it from δ2 and W2.
   Self-check: shape (2,) — the shape of x2, i.e. of W2's columns.
   Look at step 3 and step 4 together: the forward pass sends data through W;
   what does the backward pass use to send gradient through the same layer?

### 2c. General layer i, given incoming dL/dz_{i+1} of shape (out_{i+1},)

1. δi = dL/dz_i = ? in terms of the incoming gradient and the layer's own
   activation. This is where the Activation derivative method is consumed [5].
2. dL/dW_i = ?    Self-check: shape = shape of W_i.
3. dL/db_i = ?    Self-check: shape = shape of b_i.
4. dL/dx_i = ?    Self-check: shape = shape of x_i.

Now answer, in one sentence each:
- Which cached value does each of steps 1–4 consume? (Entry.input,
  Entry.pre_activation [4] — say which goes where.)
- Where does the layer's weights come from in this pass? (They're not in the
  cache. Why can't they be?)

### 2d. The ReLU convention

ReLU's derivative at exactly 0 is undefined; your code must pick a value.
Pick one, write it down, and note that your hand-computed tests contain
pre-activations that land on exactly 0.0 [2] — the convention will be
observable there.

## 3. Code — three pieces, in this order (smallest first)

Write the signatures and the bodies. For each piece, write its design
decisions first (from §4) before the body.

Piece 1 — Activation::derivative, in activation.rs [5].
Spec: takes the pre-activation, returns an array of the same shape holding the
derivative applied element-wise. One exhaustive match, adjacent to apply, same
style. The ReLU arm uses your §2d convention.
Test: extend the existing test module — for each activation, derivative of a
vector you choose, expected values by hand. Include a pre-activation with a 0.0
so the convention is pinned.

Piece 2 — a per-layer gradients struct, in layer.rs [2].
Spec: mirrors the parameters a layer owns. Decide: which fields, are they pub,
what derives.

Piece 3 — Network::backward, in network.rs [1].
Spec: consumes what forward produced, walks layers in reverse, returns one
gradients struct per layer. It must take the output and the cache together —
same pairing as forward returns [1].
Open signature question (answer in §4): by value or by reference?

## 4. Design questions — write your answers before writing Piece 3

1. Where does the loss live — a separate fn, or computed inside backward? What
   does main / a future training loop need the scalar itself for?
2. Where does the target t enter the backward call?
3. Cache by value or by reference in backward? Argue it: does backward mutate
   entries, and does the caller need the cache after?
4. The incoming gradient dL/dy — computed inside backward from (output, t), or
   passed in? What does each choice do to where the loss lives (Q1)?
5. forward_pass currently propagates the layer's error type, flagged in your
   own comment [1]. backward will produce the same situation. Settle the
   conversion you already sketched (layer error re-emitted at the network
   boundary) in the same edit.
6. How do you build the outer product (step 2b.3) with what ndarray gives you?
   Find the operation before you need it.

## 5. The numerical check — design it on the plane, run it when landed

This is the acceptance test for all of Stage B: no training result is trusted
until this passes. Design, don't implement:

1. It needs a small RANDOM network with a known seed (your seeded
   new_random [2] exists for this) plus a chosen x and t where L ≠ 0.
2. For ONE chosen weight element: how do you compute its gradient without the
   backward pass? (Perturb that one element by a small ε, run forward, compute
   L. How many forward runs? What combination of those L values estimates the
   slope? Why that combination over the naive one?)
3. How do you compare one analytical element to its numerical estimate so the
   comparison doesn't depend on the scale of the values? (Denominator design.)
4. What's your pass threshold in f32, and why can't it be 1e-9?
5. Pseudocode the whole check as a future test, including: every element of
   every layer, not just one.

## 6. Your references, ranked for this week

1. Parr & Howard, "The Matrix Calculus You Need For Deep Learning" —
   Section 4.5 (the chain rules) and Section 5 (where it turns to a network).
   Read Section 5 in YOUR convention: W (out, in), z = W·x + b. When you reach
   its worked example, cover it and do §2b yourself first. This is the only
   external text you need for backward.
2. Ruder, "An overview of gradient descent optimization algorithms" — NOT for
   this week. It's about what happens to the gradient AFTER you have it
   (SGD variants, momentum, learning rates) — that's the next stage. At most,
   steal the one sentence: the update is a step opposite the gradient, scaled
   by a learning rate.
3. Your own code is the third reference: the loop in forward_pass [1] tells you
   exactly what backward consumes [4]; the hand-computed layer tests [2] are
   free worked examples — hand-computing dL/dW for test 1 of
   test_layer_forward_pass is a legitimate exercise if you finish early.

## 7. Session plan (3–4 h)

- 0:00–0:30  §0 preflight + §1 read-through.
- 0:30–1:15  §2 fully, on paper. Self-checks last, not first.
- 1:15–1:30  §4 design answers, written down.
- 1:30–2:45  §3 pieces 1–3. Piece 1's test runs in your head (no compiler on
            board — write the expected values by hand).
- 2:45–3:15  §5 design.
- If early: the hand-computed dL/dW exercise from §6.3.

## 8. Send back when landed

1. This doc, filled in (every blank, every design answer).
2. A summary of what code you wrote, and any signature you settled.
3. Any point where your derivation disagreed with a self-check, and what you
   concluded.
4. Then we run it: cargo test, the numerical check, and the review of §2
   against what you derived.
