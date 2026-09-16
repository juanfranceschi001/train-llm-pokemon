# %% [markdown]
# # 08 — Tuning: Learning-Rate Schedules & Sweeps
#
# **Core path — the last stop before the bonus notebooks.** Notebooks 01-07
# built a working transformer and watched it generate Pokémon-flavored text.
# This notebook asks a different question: how do you actually *choose* good
# training settings instead of guessing? Two ideas, both standard practice in
# every real LLM training run:
#
# 1. **Learning-rate scheduling** — ramping the learning rate up, then back
#    down, instead of holding it fixed, often improves the final result for
#    free.
# 2. **Hyperparameter sweeps** — training several small, cheap variants and
#    comparing them, instead of hoping your first guess was right.
#
# By notebook 00's 80/20 framing: if you stop right after this notebook,
# you have a complete, working, from-scratch LLM, understand every piece of
# it, and know how to tune it. Notebooks 09-10 are bonus depth from here.
#
# > **Speed note:** every training run below is deliberately tiny (a few
# > hundred iterations on the small NANO_CONFIG model over a subset of the
# > corpus) so this whole notebook finishes in roughly a minute. Real sweeps
# > run for hours across many more candidates — the *pattern* is identical,
# > only the scale changes.

# %% [markdown]
# ## Working directory setup (same cell every notebook starts with)

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

# %%
import dataclasses
import math
import time

import torch
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend so plt.show() can't block a headless run
import matplotlib.pyplot as plt

from model import GPT, GPTConfig, NANO_CONFIG, get_device
from _shared import ensure_corpus

torch.manual_seed(1337)
device = get_device()
print("Using device:", device)
os.makedirs("assets", exist_ok=True)

# %% [markdown]
# ## Part 1 — Data (a subset, for speed)
#
# We reuse the same char-level pipeline as notebook 06, but only the first
# 200,000 characters of the corpus — plenty of Pokémon flavor text to see
# real learning-rate effects, without waiting for a full-corpus training run
# on every single sweep candidate.

# %%
corpus_path = ensure_corpus()
with open(corpus_path, encoding="utf-8") as f:
    text = f.read()

SUBSET = 200_000
text = text[:SUBSET]

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"Text subset  : {len(text):,} chars")
print(f"Vocab size   : {vocab_size} unique characters")
print(f"Train tokens : {len(train_data):,}  |  Val tokens: {len(val_data):,}")

BLOCK_SIZE = NANO_CONFIG.block_size
BATCH_SIZE = 32

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print(f"Batch shape  : x={tuple(xb.shape)}, y={tuple(yb.shape)}")

# %% [markdown]
# ## Part 2 — What is a hyperparameter?
#
# A **hyperparameter** is any setting you pick *before* training that isn't
# itself learned by gradient descent — learning rate, batch size, context
# length, model depth/width. This notebook focuses on the single most
# sensitive one: **learning rate**.

# %% [markdown]
# ## Part 3 — Learning-rate scheduling: warmup, then cosine decay
#
# Holding the learning rate fixed has two failure modes:
# - **Too high at the start** — before the model has learned anything, a big
#   step can destabilize training or even diverge.
# - **Too high near the end** — the optimizer keeps overshooting the minimum
#   instead of settling into it.
#
# The fix used by essentially every modern LLM (GPT-3, Llama, ...): start the
# learning rate small, **linearly warm it up** to its peak, then **decay it
# along a cosine curve** back down to a small floor. Cosine decay falls
# slowly at first (letting the model keep exploring), speeds up through the
# middle, then slows again near the end (letting it settle).

# %%
def get_lr(it, warmup_iters, decay_iters, max_lr, min_lr):
    """Linear warmup -> cosine decay -> floor."""
    if it < warmup_iters:                       # phase 1: ramp up
        return max_lr * (it + 1) / warmup_iters
    if it > decay_iters:                         # phase 3: floor
        return min_lr
    ratio = (it - warmup_iters) / (decay_iters - warmup_iters)   # phase 2: cosine
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))               # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)

# Sanity check the three phases before trusting the plot.
warm, decay, hi, lo = 100, 1000, 1e-3, 1e-4
assert get_lr(0, warm, decay, hi, lo) < get_lr(warm, warm, decay, hi, lo), \
    "LR should increase during warmup"
assert abs(get_lr(decay + 50, warm, decay, hi, lo) - lo) < 1e-12, \
    "LR should sit exactly at min_lr past the decay horizon"
assert get_lr(warm, warm, decay, hi, lo) <= hi + 1e-9, \
    "LR should not exceed max_lr"
print("get_lr sanity checks PASSED")

# %%
TOTAL_ITERS, WARMUP_ITERS, DECAY_ITERS = 300, 30, 270
MAX_LR, MIN_LR = 3e-3, 3e-4

iters = list(range(TOTAL_ITERS + 30))
lrs = [get_lr(i, WARMUP_ITERS, DECAY_ITERS, MAX_LR, MIN_LR) for i in iters]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(iters, lrs, color="#2563EB", linewidth=2)
ax.axvline(WARMUP_ITERS, color="#DC2626", linestyle="--", alpha=0.7, label=f"end of warmup (it={WARMUP_ITERS})")
ax.axvline(DECAY_ITERS, color="#16A34A", linestyle="--", alpha=0.7, label=f"end of decay (it={DECAY_ITERS})")
ax.axhline(MAX_LR, color="gray", linestyle=":", alpha=0.5, label=f"max_lr = {MAX_LR:.0e}")
ax.axhline(MIN_LR, color="gray", linestyle="-.", alpha=0.5, label=f"min_lr = {MIN_LR:.0e}")
ax.set_xlabel("Iteration"); ax.set_ylabel("Learning rate")
ax.set_title("Linear warmup + cosine decay schedule")
ax.legend(fontsize=9); ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("assets/08_lr_schedule.png", dpi=120)
plt.show(); plt.close()
print("Saved assets/08_lr_schedule.png")

# %% [markdown]
# ## Part 4 — A learning-rate sweep
#
# A **sweep** means training the same model several times, changing one
# hyperparameter each time, and comparing **validation** loss (never training
# loss — a model can always lower training loss by memorizing, which is
# exactly the overfitting trap notebook 02 ran into). We try four peak
# learning rates spanning two orders of magnitude: `1e-2, 3e-3, 1e-3, 3e-4`.

# %%
@torch.no_grad()
def estimate_val_loss(model, eval_iters=30):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        xb, yb = get_batch("val")
        _, loss, _ = model(xb, targets=yb)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()

def train_nano(lr, iters=300):
    """Train a fresh nano GPT for `iters` steps at peak learning rate `lr`."""
    cfg = dataclasses.replace(NANO_CONFIG, vocab_size=vocab_size)  # copy, don't mutate the shared default
    model = GPT(cfg).to(device)
    warmup = max(1, iters // 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    t0 = time.time()
    for it in range(iters):
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(it, warmup, iters, lr, lr / 10)
        xb, yb = get_batch("train")
        _, loss, _ = model(xb, targets=yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    val_loss = estimate_val_loss(model)
    print(f"  lr={lr:.0e}  |  val_loss={val_loss:.4f}  |  {time.time()-t0:.1f}s")
    return val_loss

# %%
LR_CANDIDATES = [1e-2, 3e-3, 1e-3, 3e-4]
val_losses = []
print("LR sweep (300 iters per run, nano model):")
for lr in LR_CANDIDATES:
    torch.manual_seed(1337)   # same init every run -- an apples-to-apples comparison
    val_losses.append(train_nano(lr))

best_lr = LR_CANDIDATES[val_losses.index(min(val_losses))]
print(f"\nBest lr={best_lr:.0e}  val_loss={min(val_losses):.4f}")

assert min(val_losses) < max(val_losses), \
    "a sweep that gives identical losses everywhere suggests a bug (e.g. shared model state)"
print("Sweep assert PASSED -- learning rate measurably changed the outcome")

# %%
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#DC2626" if v == min(val_losses) else "#93C5FD" for v in val_losses]
bars = ax.bar([f"{lr:.0e}" for lr in LR_CANDIDATES], val_losses, color=colors, edgecolor="white")
for bar, v in zip(bars, val_losses):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Peak learning rate"); ax.set_ylabel("Final validation loss (lower = better)")
ax.set_title("LR sweep -- 300-iter nano GPT on Pokemon text (red = best)")
ax.set_ylim(0, max(val_losses) * 1.15)
plt.tight_layout()
plt.savefig("assets/08_lr_sweep.png", dpi=120)
plt.show(); plt.close()
print("Saved assets/08_lr_sweep.png")

# %% [markdown]
# ### Reading the sweep
#
# - **Very high (`1e-2`)** — steps this large risk instability; loss can be
#   high or erratic.
# - **A sweet spot (often `3e-3` for a model this small)** — converges
#   quickly, finds a good minimum in just 300 steps.
# - **Low (`3e-4`)** — stable, but barely moves in 300 steps: it *underfits*
#   simply from not enough effective gradient signal yet.
#
# The best value isn't universal — it depends on model size, batch size, and
# how many steps you're training for. Bigger models generally want smaller
# learning rates.

# %% [markdown]
# ## Part 5 — Model size is a knob too
#
# Learning rate is the single most sensitive knob, but width matters too.
# Rather than run a second full training loop, compare parameter counts
# directly: halving `n_embd` roughly quarters total parameters (it appears
# squared-ish across attention and MLP projections).

# %%
nano_cfg = dataclasses.replace(NANO_CONFIG, vocab_size=vocab_size)
nano_model = GPT(nano_cfg).to(device)
nano_params = nano_model.num_params()

half_cfg = dataclasses.replace(nano_cfg, n_embd=NANO_CONFIG.n_embd // 2)
half_model = GPT(half_cfg).to(device)
half_params = half_model.num_params()

print(f"Nano  n_embd={nano_cfg.n_embd:3d}  params={nano_params/1e3:.1f}K")
print(f"Half  n_embd={half_cfg.n_embd:3d}  params={half_params/1e3:.1f}K")
print(f"Size ratio: {nano_params / half_params:.1f}x")
del nano_model, half_model

# %% [markdown]
# ### Rough order of impact, for a model this small
#
# 1. **Learning rate** — most sensitive; a 10x change can make or break a run.
# 2. **Training steps** — more compute almost always helps, up to a point.
# 3. **Model size** — more capacity, but slower and more prone to overfitting
#    on small data (notebook 02's whole plot twist, in miniature).
# 4. **Batch size** — bigger batches smooth gradients but usually want a
#    correspondingly higher learning rate.
# 5. **Context length** — helps most when the task actually needs long-range
#    dependencies.
# 6. **Regularization** (dropout, weight decay) — protects against
#    overfitting when data is limited relative to model size.

# %% [markdown]
# ## Part 6 — The core path, wrapped up
#
# That's notebooks 00-08: a Bag-of-Words baseline, embeddings that beat it,
# attention, the modern Llama-style components, a fully assembled transformer,
# a real training run, generation with multiple sampling strategies and a
# measured KV-cache speedup, and now the tools to tune it. Every step beat a
# *measured* number from the step before — this was never "trust us, it's
# better."
#
# **You can stop here.** That's a complete, working, from-scratch LLM, and
# you understand every piece of it. Notebooks 09 (BPE tokenizer) and 10
# (Mixture-of-Experts) are bonus depth for anyone who wants to keep going —
# skip them with no guilt if 08 already felt like a satisfying finish line.

# %%
print("Notebook 08 complete.")
print(f"  Sweep best lr : {best_lr:.0e}  (val_loss={min(val_losses):.4f})")
print(f"  assets/08_lr_schedule.png written: {os.path.exists('assets/08_lr_schedule.png')}")
print(f"  assets/08_lr_sweep.png     written: {os.path.exists('assets/08_lr_sweep.png')}")

# %% [markdown]
# ## ✅ Check your understanding
#
# 1. Why do we compare **validation** loss across sweep candidates instead of
#    training loss?
# 2. In your own words, what problem does linear warmup solve that a fixed
#    learning rate from iteration 0 wouldn't?
# 3. If you ran this exact sweep again with a completely different random
#    seed for each candidate (instead of resetting to the same seed each
#    time), would you trust the comparison as much? Why or why not?
# 4. Suppose the `1e-2` candidate's loss came back as `nan`. What would that
#    tell you, and which failure mode from Part 3 does it match?
# 5. Notebook 00 said you could "stop at notebook 08." What did that notebook
#    argue was in the remaining ~20% covered by notebooks 09-10?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Training loss can always be driven down by memorizing the training
#    batches more precisely — that's overfitting, not learning. Validation
#    loss is measured on data the model never trained on, so it's the honest
#    signal for "did this hyperparameter actually help the model generalize?"
# 2. Without warmup, the very first gradient steps hit a randomly-initialized
#    model with the full peak learning rate, which can produce a large,
#    destabilizing update before the model has learned anything sensible yet.
#    Warmup ramps the step size up gradually so early training is stable.
# 3. Less. Using the same seed for every candidate means the only thing that
#    differs between runs is the learning rate itself — a fair, controlled
#    comparison. Different seeds per candidate would mix in random
#    initialization luck with the effect of the hyperparameter, making it
#    harder to tell whether a candidate won because it was genuinely better or
#    just got a luckier starting point.
# 4. `nan` loss means the training diverged — gradients or weights blew up to
#    infinity (or produced an invalid `0/0`-style computation) instead of
#    converging. That matches the "too high at the start" failure mode from
#    Part 3: a learning rate large enough to destabilize training entirely.
# 5. A from-scratch Byte-Pair Encoding (BPE) tokenizer (notebook 09) as a
#    smarter alternative to char-level tokenization, and Mixture-of-Experts
#    (notebook 10) as a capstone on how production-scale models add capacity
#    without proportionally more compute per token -- both real techniques,
#    both explicitly optional.
# </details>
