# %% [markdown]
# # 05 — Assembling the Model
#
# Notebooks 01-04 built the pieces of a modern transformer language model one at
# a time: a bag-of-words baseline, token embeddings, attention, and then the
# Llama-style upgrades (RoPE, RMSNorm, SwiGLU). This notebook does **not**
# introduce any new math. Instead it does something almost as important: it
# proves those pieces are actually wired together correctly.
#
# ## Why this matters (80/20)
#
# The real intellectual work of this course happened in notebooks 01-04 — that's
# where ~80% of the *understanding* lives. This notebook is more like a final
# inspection before a flight: we walk through `model.py` (the single file every
# later notebook imports from) piece by piece, count its parameters, and then
# run the one test every ML engineer runs on a brand-new architecture before
# trusting it with real data — **overfitting a single tiny batch**. If a model
# can't even memorize four sentences, it has no business training on the whole
# Pokédex. Get this notebook green, and notebook 06 (the real training run) has
# nothing to worry about on the "is the model built right" front.
#
# This notebook is part of the **core path** (00-08).

# %% [markdown]
# ## Working directory
#
# Jupyter runs notebooks from the `notebooks/` folder. The cell below walks up
# the directory tree until it finds the project root (marked by
# `requirements.txt`) and changes into it, so paths like `assets/...` and
# imports like `from model import ...` resolve correctly no matter how you
# launched Jupyter or Colab.

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "notebooks"))
print("Working directory:", os.getcwd())

# %% [markdown]
# ## Imports
#
# This is the first notebook that imports the model instead of building it.
# `model.py` is the course's **single source of truth** for the architecture:
# every later notebook (06 training, 07 evaluation, 08 tuning) imports `GPT`
# from the same file, so there is never a "which version of the model am I
# using" question. We are not going to redefine `GPT`, `Block`, `RMSNorm`, or
# any other class here — we're going to read the real file and run it.

# %%
import torch
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend so plt.show() can't block a headless run
import matplotlib.pyplot as plt

from model import GPT, NANO_CONFIG, DEFAULT_CONFIG, get_device

torch.manual_seed(1337)
device = get_device()
print("Using device   :", device)
print("NANO_CONFIG    :", NANO_CONFIG)
print("DEFAULT_CONFIG :", DEFAULT_CONFIG)

# %% [markdown]
# `NANO_CONFIG` is a deliberately tiny preset (3 layers, 128-dim embeddings,
# 64-token context, no dropout) meant for fast debugging — exactly what we'll
# use for the overfit test below. `DEFAULT_CONFIG` is the full-size model
# (6 layers, 384-dim embeddings, 256-token context) that notebook 06 will
# actually train on the Pokémon corpus.

# %% [markdown]
# ---
# ## Part 1 — The pre-norm transformer block
#
# A **transformer block** is the repeating unit that gets stacked `n_layer`
# times inside the model — `DEFAULT_CONFIG` stacks 6 of them, `NANO_CONFIG`
# stacks 3. Each block does two things to every token's vector representation:
#
# 1. **Self-attention** — look at earlier tokens and pull in relevant context
#    (this is the mechanism notebook 03 built from scratch).
# 2. **A small MLP (SwiGLU)** — process each token's vector on its own, with no
#    look-back to other tokens. If attention is "gather information from
#    everyone else," the MLP is "now think about what you gathered."
#
# Here is `Block` exactly as it's defined in `model.py`:
#
# ```python
# class Block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attn_norm = RMSNorm(config.n_embd)
#         self.attn = CausalSelfAttention(config)
#         self.mlp_norm = RMSNorm(config.n_embd)
#         self.mlp = SwiGLU(config)
#
#     def forward(self, x, cos, sin, kv_cache=None):
#         a, present = self.attn(self.attn_norm(x), cos, sin, kv_cache)
#         x = x + a                           # residual connection
#         x = x + self.mlp(self.mlp_norm(x))  # residual connection
#         return x, present
# ```
#
# ### Why normalize *before* the sub-layer, not after?
#
# Look closely at the order: `self.attn(self.attn_norm(x), ...)`. The
# normalization (`RMSNorm`, built in notebook 04 — it rescales a vector so its
# values have a consistent typical size) happens to a *copy* of `x` that gets
# fed into attention, while the *original*, un-normalized `x` is what the
# residual connection carries forward. This is called **pre-norm**.
#
# The original 2017 transformer paper did the opposite — normalize *after*
# each sub-layer ("post-norm"). That works for shallow stacks, but as models
# get deeper (dozens or hundreds of blocks), post-norm makes training unstable:
# large, un-normalized activations from early layers keep growing as they pass
# through the residual stream, and gradients during backpropagation swing
# wildly in size. Pre-norm sidesteps this: every sub-layer always sees a freshly
# rescaled input, no matter how large the raw residual stream has grown, so the
# scale of what each layer works with stays predictable at any depth. This is
# why every modern LLM (GPT-3, Llama, Mistral, and this course's model) uses
# pre-norm.
#
# ### Residual (skip) connections: the gradient highway
#
# The `x = x + ...` lines are **residual connections**: instead of replacing
# `x` with the sub-layer's output, the block adds the sub-layer's output *on
# top of* `x`. This single idea (from image-recognition research in 2015) is
# one of the most important tricks in all of deep learning, for two reasons:
#
# - **Training a deep stack of blocks requires gradients to flow backward
#   through all of them.** Without residuals, each block's gradient gets
#   multiplied by that block's local derivatives — and multiplying many small
#   numbers together shrinks toward zero fast (the "vanishing gradient"
#   problem), so early layers barely learn. With a residual connection, the
#   gradient has a direct additive path — `d(x + f(x))/dx = 1 + f'(x)` — so
#   there is always a route back to the beginning where the gradient doesn't
#   get multiplied away to nothing. Think of it as a highway that runs the
#   full depth of the network, with each block able to add a small on-ramp of
#   its own contribution, rather than gradient having to squeeze through every
#   single block's narrow local pathway.
# - **Each block only has to learn a *correction*, not a full transformation.**
#   Right after initialization, a block's attention and MLP outputs start out
#   close to zero (small random weights), so `x = x + a` and `x = x + mlp(...)`
#   both start out very close to `x = x` — the identity function. That gives
#   training a stable, "do nothing yet" starting point, and then each block
#   gradually learns what small adjustment to add. Learning "here's a small
#   nudge to make" is much easier to optimize than learning "compute the
#   entire right answer from scratch."
#
# Every block in `model.py` uses two residual connections: one wrapping
# attention, one wrapping the MLP.

# %% [markdown]
# ---
# ## Part 2 — Weight tying
#
# A language model has two large matrices sitting at its two ends:
#
# - `tok_emb`, the **token embedding table** (shape `vocab_size x n_embd`):
#   converts an input token ID into a dense vector when text *enters* the
#   model (notebook 02's topic).
# - `lm_head`, the **output projection** (shape `n_embd x vocab_size`): converts
#   the model's final hidden vector back into a score per vocabulary token when
#   the model is about to *predict* the next token.
#
# Both matrices are really doing the same job from opposite directions — each
# one associates every vocabulary token with a vector in the same `n_embd`-
# dimensional space. **Weight tying** notices this and uses one matrix for
# both jobs instead of two separate ones:
#
# ```python
# self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
# self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
# self.tok_emb.weight = self.lm_head.weight   # weight tying
# ```
#
# After that last line, `tok_emb.weight` and `lm_head.weight` are not two
# tensors with equal values — they are the *same* tensor in memory. Update one
# during training and the other changes too, automatically, because there was
# only ever one matrix to begin with.
#
# **Why bother?**
# - **Fewer parameters to store and train.** One `vocab_size x n_embd` matrix
#   instead of two. For a small character vocabulary this saves a modest
#   amount; for a large subword vocabulary (tens of thousands of tokens, as in
#   notebook 09's BPE tokenizer) it can save millions of parameters.
# - **It tends to help generalization.** The model is forced to represent each
#   token with a single, consistent vector whether that token is being read in
#   or predicted out, rather than learning two independently-drifting
#   representations of the same symbol.
#
# One subtlety: because the two matrices share storage, naively summing
# `p.numel()` over every parameter would count that shared matrix twice.
# `model.py` corrects for this in `num_params()`:
#
# ```python
# def num_params(self):
#     n = sum(p.numel() for p in self.parameters())
#     return n - self.lm_head.weight.numel()   # subtract the tied duplicate
# ```

# %% [markdown]
# ---
# ## Part 3 — Counting parameters
#
# Let's instantiate both presets and see how big they actually are.

# %%
nano = GPT(NANO_CONFIG)
full = GPT(DEFAULT_CONFIG)

print(f"NANO_CONFIG    params: {nano.num_params():>10,}  ({nano.num_params()/1e6:.3f}M)")
print(f"DEFAULT_CONFIG params: {full.num_params():>10,}  ({full.num_params()/1e6:.3f}M)")

# %% [markdown]
# `DEFAULT_CONFIG` is deliberately sized to be trainable on a laptop CPU in a
# few minutes while still being a genuine multi-million-parameter transformer.
# It's tiny by industry standards (GPT-3 has 175 *billion* parameters — about
# 20,000x more) but every architectural idea — attention, RoPE, RMSNorm,
# SwiGLU, weight tying — is exactly the same idea, just at a scale a laptop can
# handle in one sitting.

# %% [markdown]
# ### Where do the parameters live?
#
# A model isn't one big blob of numbers — it's made of named components, and
# some contribute far more parameters than others. Let's break `DEFAULT_CONFIG`
# down by top-level component.

# %%
def param_table(model):
    totals = {}
    for name, module in model.named_modules():
        if list(module.children()):        # skip non-leaf (container) modules
            continue
        n = sum(p.numel() for p in module.parameters(recurse=False))
        if n == 0:
            continue
        top = name.split(".")[0]
        totals[top] = totals.get(top, 0) + n

    raw_total = sum(totals.values())
    print(f"{'Component':<12} {'Params':>12} {'Share of raw total':>20}")
    print("-" * 46)
    for name, n in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{name:<12} {n:>12,} {n / raw_total * 100:>19.1f}%")
    print("-" * 46)
    print(f"{'raw total':<12} {raw_total:>12,}   (double-counts the tied matrix)")
    print(f"{'model total':<12} {model.num_params():>12,}   (via num_params())")

param_table(full)

# %% [markdown]
# The `blocks` (the 6 stacked transformer layers) dominate the parameter
# count, as they should — that's where nearly all of the model's "thinking"
# happens. `tok_emb` shows up in this breakdown because `named_modules` walks
# the module tree, but its parameter tensor is the very same object counted
# under `lm_head` thanks to weight tying, which is exactly why `num_params()`
# subtracts it back out to get the true count.

# %% [markdown]
# ---
# ## Part 4 — Forward-pass shape check
#
# Before testing anything about *learning*, check that the plumbing is right:
# feed in a batch of fake token IDs and confirm the output shapes match what
# the rest of the codebase expects.
#
# Given input token IDs of shape `(B, T)` — `B` sequences ("batch size"), each
# `T` tokens long — the model should return:
# - `logits` of shape `(B, T, vocab_size)`: one score per possible next token,
#   for every position in every sequence.
# - `loss`: a single scalar number (only computed when `targets` are given).

# %%
torch.manual_seed(42)
B, T = 2, 16
vocab = DEFAULT_CONFIG.vocab_size

x = torch.randint(0, vocab, (B, T))
y = torch.randint(0, vocab, (B, T))
logits, loss, _ = full(x, targets=y)

expected_loss = torch.log(torch.tensor(float(vocab))).item()
print(f"logits shape : {tuple(logits.shape)}  (expected ({B}, {T}, {vocab}))")
print(f"loss          : {loss.item():.4f}  (expected roughly log({vocab}) = {expected_loss:.4f})")

assert logits.shape == (B, T, vocab), f"unexpected logits shape {logits.shape}"
assert loss.ndim == 0, "loss should be a scalar"
print("Shape check PASSED")

# %% [markdown]
# **Why should the loss start near `log(vocab_size)`?** Right after
# initialization the weights are small and random, so the model has no
# learned preference — its predicted probability distribution over the
# vocabulary is roughly *uniform*. The cross-entropy loss of a uniform guess
# over `V` outcomes is exactly `log(V)` — the maximum possible loss for that
# vocabulary size, i.e. maximum uncertainty. If the very first loss you ever
# see from a fresh model is much *lower* than `log(vocab_size)`, that's a red
# flag: something has already leaked information it shouldn't have (for
# example, targets accidentally visible to the input).

# %% [markdown]
# ---
# ## Part 5 — Overfit a single batch
#
# This is the single most important test in this notebook, and arguably one of
# the most important tests in all of deep learning practice.
#
# **The idea:** take one small, fixed batch of data and train on *only that
# batch*, repeatedly, for many steps. A correctly-wired neural network with
# enough capacity should be able to drive the loss on that one batch down to
# nearly zero — it should be able to memorize four short sequences perfectly.
#
# **Why this is the first thing to check on any new architecture:** if the
# model *can't* memorize one tiny batch, there is almost certainly a bug
# somewhere in the wiring — a bad reshape, a detached gradient, targets not
# lining up with inputs, a frozen layer that should be trainable, and so on.
# Real training data is noisy and won't perfectly fit any model, so if you
# skip straight to training on the full Pokémon corpus, a badly-wired model and
# a working-but-undertrained model can look confusingly similar (both have
# "high loss that goes down slowly"). Overfitting one batch removes that
# ambiguity: there's no excuse for a correctly-implemented model not to nail a
# single memorization task, so failing this test points straight at a bug
# rather than "just needs more training."
#
# We use random synthetic token IDs — there's nothing Pokémon-specific about
# this test, and using `NANO_CONFIG` keeps it fast (seconds, not minutes) on a
# CPU.

# %%
torch.manual_seed(1337)
tiny = GPT(NANO_CONFIG).to(device)

xb = torch.randint(0, NANO_CONFIG.vocab_size, (4, 16), device=device)
yb = torch.randint(0, NANO_CONFIG.vocab_size, (4, 16), device=device)

optimizer = torch.optim.AdamW(tiny.parameters(), lr=3e-3)
losses = []

for step in range(300):
    _, loss, _ = tiny(xb, targets=yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"Step 0   loss: {losses[0]:.4f}")
print(f"Step 299 loss: {losses[-1]:.4f}")

assert losses[-1] < 0.5, (
    f"Overfit-one-batch test FAILED: final loss {losses[-1]:.4f} is still >= 0.5. "
    "That would mean something in model.py's wiring is broken -- this should "
    "never happen with the shipped model.py."
)
print("Overfit-one-batch test PASSED -- model wiring is correct.")

# %% [markdown]
# ### Plotting the loss curve
#
# A healthy run drops fast in the first few dozen steps and then flattens out
# near zero. A curve that plateaus early at a high value, or bounces around
# without ever trending down, would point to a bug rather than "needs more
# steps."

# %%
os.makedirs("assets", exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(losses)
ax.axhline(0.5, color="red", linestyle="--", label="pass threshold (0.5)")
ax.set_xlabel("Gradient step")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Overfitting one synthetic batch (NANO_CONFIG, 300 steps)")
ax.legend()
plt.tight_layout()
plt.savefig("assets/05_overfit_curve.png", dpi=120)
plt.show()
print("Saved assets/05_overfit_curve.png")

# %% [markdown]
# ---
# ## Summary
#
# | Check | What it confirms |
# |---|---|
# | Pre-norm `Block` structure | normalize before attention/MLP, not after -> stable deep training |
# | Residual connections | `x = x + sub_layer(norm(x))` gives gradients a direct path back |
# | Weight tying | `tok_emb.weight is lm_head.weight` -- one matrix, fewer params |
# | Parameter counts | NANO and DEFAULT sizes printed and broken down by component |
# | Forward-shape check | logits `(B, T, vocab_size)`, scalar loss near `log(vocab_size)` at init |
# | Overfit-one-batch | loss < 0.5 after 300 steps on synthetic data -> wiring is correct |
#
# We didn't write a single new class in this notebook — every piece already
# existed in `model.py`, built conceptually across notebooks 01-04. What we did
# here was confirm those pieces snap together into a model that actually
# learns. That confirmation is what makes notebook 06 — training this same
# model on real Pokémon text — a reasonable thing to spend several minutes of
# compute on.

# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. In `Block.forward`, why is `RMSNorm` applied to a *copy* of `x` that feeds
#    into attention, while the residual connection uses the original,
#    un-normalized `x`? What would change if the residual line were
#    `x = self.attn_norm(x) + a` instead?
# 2. Suppose someone removes both residual connections from `Block` (so
#    `x = a` and `x = self.mlp(...)` directly, no `+ x`). Would the model
#    still be mathematically capable of learning the same function? Why does
#    removing them still hurt training in practice?
# 3. `model.num_params()` subtracts `self.lm_head.weight.numel()` from the raw
#    parameter sum. Why is that subtraction necessary, and what would go wrong
#    if `num_params()` just returned the raw sum instead?
# 4. You train `GPT(NANO_CONFIG)` on a single batch and the loss gets stuck
#    around 4.1 for all 300 steps, never dropping. Is this more likely a sign
#    that the model needs more training steps, or a sign of a bug? What's your
#    reasoning?
# 5. At initialization, why is the expected loss close to `log(vocab_size)`
#    rather than, say, 0 or some arbitrary large number?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Pre-norm feeds attention a rescaled *view* of `x` so attention always sees
#    inputs of a predictable, consistent scale no matter how large the
#    residual stream has grown at that depth — but the residual connection
#    carries forward the raw, un-normalized `x` so information isn't lost or
#    repeatedly rescaled as it passes through many blocks. If the residual
#    line were `x = self.attn_norm(x) + a` instead, that would be post-norm:
#    the normalization would sit *after* the addition, so the un-normalized,
#    potentially large raw activations would flow directly into the next
#    block's residual stream, which is exactly the instability pre-norm was
#    adopted to avoid in deep stacks.
# 2. Mathematically, yes — a deep-enough stack of attention/MLP layers without
#    residuals can still represent very similar functions in principle,
#    because nothing prevents a layer from learning to approximate an
#    identity mapping plus a change on its own. In practice it hurts badly
#    because gradients would have to flow backward through every
#    layer's full nonlinear transformation with no additive shortcut,
#    so gradients shrink toward zero in deep stacks (vanishing gradients),
#    early layers get very small or noisy learning signal, and there's no
#    longer a stable "do nothing yet" starting point at initialization the
#    way `x = x + small_correction` provides.
# 3. `tok_emb.weight` and `lm_head.weight` are tied — they are literally the
#    same tensor in memory, not two equal-valued tensors. `model.parameters()`
#    yields it twice, once as part of `tok_emb` and once as part of
#    `lm_head`, so summing `p.numel()` over all parameters double-counts that
#    one shared matrix. Without the subtraction, `num_params()` would report
#    a parameter count larger than the model's true, unique parameter count
#    — misleading for anything that cares about model size (memory footprint,
#    comparisons to other models, etc.).
# 4. A bug, not "needs more training." Overfitting one small, fixed batch is
#    something even a small model should nail — that's the entire point of
#    the test. A loss stuck at a high, unmoving value (especially one close
#    to `log(vocab_size)`, meaning it never moved off "uniform random
#    guessing") points to something broken in the wiring: for example,
#    gradients not reaching some parameters, targets not actually lined up
#    with predictions, or a part of the forward pass being detached from the
#    graph. More training steps would not fix a wiring bug.
# 5. Right after initialization the model's weights are small and random, so
#    it has no learned reason to favor any vocabulary token over another —
#    its predicted distribution over the vocabulary is close to uniform. The
#    cross-entropy loss of a uniform guess over `V` possible tokens is
#    exactly `log(V)`, the maximum-entropy (maximum uncertainty) case. Loss
#    can only go down from there as the model learns actual structure; a
#    freshly initialized model reporting a much lower loss would suggest
#    information leaked into the input that shouldn't be there.
# </details>
