# %% [markdown]
# # 06 — Training Loop: Actually Training the Model
#
# Notebook 05 built the full transformer and proved it works by overfitting a
# single tiny batch until its loss dropped near zero — a wiring check, not a
# real result. This notebook is where that wiring pays off: we train the same
# model on the whole Pokémon corpus for real, and save a checkpoint that
# notebook 07 will load to generate text.
#
# **Why this notebook matters (the 80/20 case):** every notebook before this
# one built a *component* — a tokenizer, an embedding, attention, a full
# model. None of them, on their own, produce something you can point at and
# say "this generates Pokédex text." This notebook is where all of that
# machinery gets pointed at real data and actually learns something. If you
# only read one code cell in this entire course, make it the training loop
# below — it's a five-line pattern (forward pass, compute loss, zero old
# gradients, backward pass, step the optimizer) that is, almost verbatim, how
# every large language model in production today is trained. The scale is
# different; the loop is not.

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
import time

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend so plt.show() can't block a headless run
import matplotlib.pyplot as plt
import torch

from model import GPT, DEFAULT_CONFIG, get_device
from _shared import ensure_corpus

torch.manual_seed(1337)
device = get_device()
print("Using device:", device)

# %% [markdown]
# ## Part 1 — Character-level tokenization
#
# Notebook 01 used a *word*-level vocabulary (one token per word, capped at
# 2000 words, with an `<unk>` token for anything rarer). Here we go the
# opposite direction: **character-level** tokenization, where every distinct
# character in the corpus — letters, spaces, punctuation, even the accented
# `é` in "Pokémon" — is its own token.
#
# Why char-level for this notebook?
#
# - **The vocabulary is tiny and fixed.** A word-level vocabulary needs a
#   cutoff (2000 words) and an escape hatch for words it's never seen
#   (`<unk>`). A character vocabulary just *is* every character that appears
#   in the text — usually somewhere between 50 and 100 for English prose.
#   There's no such thing as an "unknown character" the model can't
#   represent, because every character it will ever see during training is
#   already in the vocabulary.
# - **It's the simplest possible complete solution.** The model has to work
#   harder (it must learn to spell "Charizard" one letter at a time instead
#   of getting it as a single token), but nothing about the training loop,
#   loss function, or checkpoint format changes based on that choice. That
#   makes char-level the cleanest way to see the training loop itself
#   clearly, without a subword-tokenizer detour. (Notebook 09, bonus/optional,
#   revisits this with a real BPE subword tokenizer — the kind production
#   models actually use — once you've seen why it helps.)
#
# `ensure_corpus()` guarantees `data/pokemon_corpus.txt` exists, downloading
# and caching it from PokeAPI if this is a fresh machine that has never run
# any notebook in this repo before.

# %%
corpus_path = ensure_corpus()
with open(corpus_path, encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(f"Corpus length: {len(text):,} characters")
print(f"Vocab size   : {vocab_size} unique characters")
print(f"First 120 chars: {text[:120]!r}")

# %% [markdown]
# `stoi` ("string to index") and `itos` ("index to string") are the two
# halves of the tokenizer: `stoi` turns each character into the integer ID
# the model works with, `itos` turns predicted IDs back into text. We save
# both inside the checkpoint later, because a model trained with one mapping
# is meaningless without the *exact same* mapping to decode its output —
# swap in a different `stoi`/`itos` and "logit index 7" would mean a
# different letter entirely.
#
# Encoding the whole corpus turns it into one long 1-D tensor of integers —
# every character replaced by its `stoi` lookup, in order. We then hold out
# the last 10% as a **validation set**: data the model will never train on,
# used purely to check whether it's learning patterns that generalize or
# just memorizing the training text.

# %%
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

config = DEFAULT_CONFIG
config.vocab_size = vocab_size
print(f"\nModel config: {config}")

# %% [markdown]
# ## Part 2 — Batches, iterations, and `get_batch`
#
# Two terms that come up constantly in training:
#
# - An **iteration** is one gradient update: show the model one batch of
#   data, measure how wrong it was, nudge its weights slightly to be less
#   wrong next time. Repeat thousands of times.
# - An **epoch** is one full pass over the entire training set. With a
#   dataset this size and a training budget measured in minutes rather than
#   hours, we'll only ever see a small fraction of one epoch — so from here
#   on we count **iterations**, not epochs.
#
# A **batch** is a small group of training examples processed together (here,
# 32 of them at once) instead of one at a time. This matters for two
# reasons: GPUs (and even CPUs, to a lesser extent) do many parallel
# multiplications at once, so processing 32 examples together is barely
# slower than processing 1; and averaging the gradient over 32 examples gives
# a much less noisy update direction than trusting a single example.
#
# `get_batch()` builds one batch by picking 32 random starting positions in
# the data and slicing out a `block_size`-character window from each:
#
# - **`x`** (the input): characters at positions `[i, i + block_size)`.
# - **`y`** (the target): the *same* window shifted one character to the
#   right — `[i+1, i + block_size + 1)`. So `y[t]` is always "the character
#   that actually came after `x[0..t]`." This is what lets one window teach
#   the model `block_size` separate next-character predictions simultaneously
#   (predict position 1 from position 0, position 2 from positions 0-1, and
#   so on) instead of just one.

# %%
def get_batch(split, batch_size=32):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - config.block_size, (batch_size,))
    x = torch.stack([d[i:i + config.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print(f"Batch shapes: x={tuple(xb.shape)}, y={tuple(yb.shape)}")
assert xb.shape == (32, config.block_size)
assert yb.shape == (32, config.block_size)
print("get_batch shape check PASSED")

# %% [markdown]
# ## Part 3 — Measuring loss reliably: `estimate_loss`
#
# The model's `forward()` (see `model.py`) already returns a **cross-entropy
# loss** whenever you pass it `targets`: a single number measuring how
# "surprised" the model was by the real next character, averaged over every
# position in the batch. Lower is better. At the very start of training,
# before the model has learned anything, it should be close to
# `log(vocab_size)` — the loss of a model that just guesses uniformly at
# random among all characters.
#
# One batch's loss is noisy — it depends entirely on which 32 random windows
# happened to get sampled. Two back-to-back single-batch measurements can
# easily differ by more than any real trend in the loss. `estimate_loss()`
# fixes that by averaging over `eval_iters` (50) independent batches for both
# train and val, giving a stable number worth plotting and comparing.
#
# Two other details matter here:
#
# - **`model.eval()`**: switches off dropout (`model.py`'s `SwiGLU` and
#   attention layers randomly zero out some activations during training as a
#   regularizer — see notebook 04). We don't want that randomness distorting
#   a loss measurement; `model.train()` at the end turns it back on for the
#   next training step.
# - **`@torch.no_grad()`**: tells PyTorch not to bother tracking gradients
#   for anything inside this function. We're only reading the loss here, not
#   updating weights from it, so tracking gradients would just waste memory
#   and time.

# %%
@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss, _ = model(xb, targets=yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# %% [markdown]
# ## Part 4 — Dry run: how long will this actually take?
#
# We're about to commit to a real training run, but "real" needs a time
# budget — nobody wants a notebook cell that might run for an unknown number
# of hours. Instead of guessing, we measure: run 50 real training iterations,
# time them, and use that rate to compute how many iterations fit inside a
# **6-minute wall-clock budget**.
#
# Six minutes is a laptop-friendly target — long enough to see the loss drop
# substantially, short enough that this notebook still finishes in one
# sitting on a CPU-only machine. It is **not** a law of nature: if you have
# more time or a GPU, raising the time budget (or just multiplying
# `max_iters` directly) gives the model more updates to work with and
# produces a lower final loss and noticeably sharper generated text in
# notebook 07. This run is capped short so the notebook renders quickly and
# predictably for everyone following along — not because 6 minutes is
# secretly enough to fully train this model.

# %%
torch.manual_seed(1337)
model = GPT(config).to(device)
print(f"Model parameters: {model.num_params()/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

DRY_ITERS = 50
t0 = time.time()
for _ in range(DRY_ITERS):
    xb, yb = get_batch("train")
    _, loss, _ = model(xb, targets=yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
its_per_sec = DRY_ITERS / (time.time() - t0)
print(f"Throughput: {its_per_sec:.1f} iterations/sec on {device}")

TARGET_SECONDS = 6 * 60
max_iters = max(500, (int(its_per_sec * TARGET_SECONDS) // 500) * 500)
print(f"max_iters : {max_iters}  (~{max_iters/its_per_sec/60:.1f} min at this rate)")
print("(Rounded to a multiple of 500 so evaluation checkpoints land evenly.)")

# %% [markdown]
# Note that the dry run above wasn't wasted work — those 50 iterations
# already updated the model's weights a little (we needed real gradient
# steps to get a realistic timing measurement, not fake no-op ones). The
# training loop below just continues from wherever the model already is.
#
# ## Part 5 — The training loop
#
# This is the core of the whole course. Each iteration does exactly the same
# five things:
#
# 1. **Forward pass**: feed a batch through the model, get back a loss (how
#    wrong its next-character predictions were).
# 2. **`optimizer.zero_grad()`**: clear out gradients left over from the
#    *previous* iteration. PyTorch accumulates gradients by default, so
#    skipping this would silently mix gradients from unrelated batches
#    together.
# 3. **`loss.backward()`**: **backpropagation**. PyTorch walks the
#    computation graph backward from the loss and computes the **gradient**
#    — for every single weight in the model, how much (and in which
#    direction) nudging that weight would increase the loss.
# 4. **`optimizer.step()`**: actually update every weight, moving it a small
#    step *opposite* its gradient (since we want to *decrease* loss, not
#    increase it).
# 5. Occasionally, **evaluate and checkpoint** (every `eval_interval=500`
#    iterations): call `estimate_loss()` and, if val loss just hit a new
#    low, save the model.
#
# **AdamW** is the optimizer doing step 4. Plain gradient descent moves every
# weight by the same step size scaled only by its gradient; Adam-family
# optimizers additionally track a running average of each individual
# weight's recent gradient history and use it to give frequently/steeply
# updated weights smaller effective steps and rarely-updated ones larger
# ones — in practice this converges much faster and more reliably than plain
# gradient descent, which is why it's the default for nearly all transformer
# training. The "W" adds **weight decay** (`weight_decay=0.1`): at every
# step, shrink every weight very slightly toward zero, independent of the
# gradient. This is a regularizer — it discourages any single weight from
# growing huge and overfitting to quirks of the training data.
#
# The **learning rate** (`lr=3e-4`) scales every update step. Set it too
# high and training diverges (loss shoots up or oscillates wildly); set it
# too low and training crawls, barely moving in a reasonable number of
# iterations. `3e-4` is a well-tested default for AdamW on models this size
# — it's even nicknamed "the Karpathy constant" in ML folklore because it
# works surprisingly often without tuning.
#
# We checkpoint the **best** val loss seen so far, not just whatever the
# model looks like at the very last iteration — the last iteration isn't
# guaranteed to be the best one (loss can bounce around), so always saving
# "whichever snapshot had the lowest val loss" is a strictly better choice
# for notebook 07 to load from.

# %%
EVAL_INTERVAL = 500
EVAL_ITERS = 50
BATCH_SIZE = 32

history = {"train": [], "val": [], "iters": []}
best_val_loss = float("inf")
os.makedirs("checkpoints", exist_ok=True)

print(f"Training for {max_iters} iters | eval every {EVAL_INTERVAL} | batch={BATCH_SIZE}")
train_start = time.time()

for it in range(max_iters + 1):
    if it % EVAL_INTERVAL == 0:
        losses = estimate_loss(model, eval_iters=EVAL_ITERS)
        elapsed = time.time() - train_start
        print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.0f}s")
        history["train"].append(losses["train"])
        history["val"].append(losses["val"])
        history["iters"].append(it)

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "stoi": stoi,
                "itos": itos,
            }, "checkpoints/model.pt")

    if it == max_iters:
        break

    xb, yb = get_batch("train", batch_size=BATCH_SIZE)
    _, loss, _ = model(xb, targets=yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

total_time = time.time() - train_start
print(f"\nTraining complete: {total_time/60:.1f} min | best val loss: {best_val_loss:.4f}")

# %% [markdown]
# ## Part 6 — Reading the train/val curves
#
# What to look for in the numbers just printed:
#
# - **Both losses dropping together** is the healthy case — the model is
#   learning patterns (letter frequencies, common words like "Pokémon",
#   typical sentence shapes) that show up in *both* the training text and
#   the held-out validation text, i.e. patterns that generalize.
# - **Train loss dropping while val loss flattens or rises** is
#   **overfitting**: the model is starting to memorize specific quirks of
#   the training sequences rather than learning anything that transfers to
#   text it hasn't seen. It's the same failure mode from notebook 01's
#   Bag-of-Words baseline, just showing up in a fancier model.
# - **Val loss staying slightly above train loss** throughout (not
#   diverging, just a small persistent gap) is completely normal — the
#   model will always do a little better on data it's updating its weights
#   against directly.
#
# For a short, capped run like this one you should mainly see steady
# improvement rather than a full plateau — that's expected, since we
# deliberately limited training to a 6-minute budget rather than training to
# convergence.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history["iters"], history["train"], label="train loss", marker="o", markersize=4)
ax.plot(history["iters"], history["val"], label="val loss", marker="s", markersize=4)
ax.set_xlabel("Training iteration")
ax.set_ylabel("Cross-entropy loss")
ax.set_title(f"Training run — {config.n_layer}L x {config.n_embd}d GPT on Pokémon corpus ({max_iters} iters)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/06_loss_curve.png", dpi=120)
plt.show()

print(f"Initial val loss: {history['val'][0]:.4f}")
print(f"Final   val loss: {history['val'][-1]:.4f}")
print(f"Final   train loss: {history['train'][-1]:.4f}")

# %% [markdown]
# ## Part 7 — Confirming the checkpoint
#
# The checkpoint saved during training holds exactly four things, which is
# everything notebook 07 (and 08) will need to reload this model and
# generate text with it:
#
# - **`model_state`**: every learned weight, via `model.state_dict()`.
# - **`config`**: the `GPTConfig` used to build the model — needed to
#   reconstruct an identical architecture before the weights can be loaded
#   back into it.
# - **`stoi` / `itos`**: the character-to-index mapping this specific model
#   was trained with, so a prompt can be encoded and generated output
#   decoded consistently.
#
# We deliberately did **not** overwrite the checkpoint with the final
# iteration's weights here — it was already saved inside the loop above,
# every time a new best val loss was reached, and the final iteration isn't
# guaranteed to be the best one.

# %%
assert os.path.exists("checkpoints/model.pt"), (
    "checkpoints/model.pt is missing -- the training loop should have saved it "
    "every time val loss improved."
)
print(f"checkpoints/model.pt confirmed on disk (best val loss: {best_val_loss:.4f})")

# %% [markdown]
# ## Part 8 — Sanity checks

# %%
assert history["val"][-1] < history["val"][0], (
    f"Val loss did not improve at all: {history['val'][0]:.4f} -> {history['val'][-1]:.4f}"
)
loaded = torch.load("checkpoints/model.pt", map_location="cpu", weights_only=False)
assert set(loaded.keys()) == {"model_state", "config", "stoi", "itos"}
assert loaded["config"].vocab_size == vocab_size
print("All asserts PASSED")
print(f"  Val loss: {history['val'][0]:.4f} -> {history['val'][-1]:.4f}  "
      f"(drop: {history['val'][0] - history['val'][-1]:.4f})")
print("  Checkpoint keys: model_state, config, stoi, itos (all present)")

# %% [markdown]
# ## Summary
#
# | Metric | Value |
# |---|---|
# | Model | 6-layer, 384-dim GPT (~9.4M params) |
# | Training data | Pokémon corpus, 90% / 10% train/val split |
# | Tokenization | character-level |
# | Optimizer | AdamW (lr=3e-4, weight_decay=0.1) |
# | Batch size | 32 |
# | Iterations | `max_iters` (auto-computed for a ~6 min budget) |
# | Best checkpoint | `checkpoints/model.pt` |
#
# You now have a genuinely trained, from-scratch transformer language model
# saved to disk. **Next:** notebook 07 loads this checkpoint and actually
# generates Pokémon-flavored text with it.

# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. Why does this notebook tokenize by individual *character* instead of by
#    *word*, and what problem does that sidestep that notebook 01's
#    word-level tokenizer had to handle explicitly?
# 2. `get_batch()` builds `y` as the same window as `x` but shifted right by
#    one character. Why does that one-character shift let a single window of
#    length `block_size` train the model on `block_size` separate
#    predictions at once, instead of just one?
# 3. Suppose you doubled `TARGET_SECONDS` from 6 minutes to 12 minutes and
#    reran this notebook on the same machine. What would you expect to
#    happen to `max_iters`, and what would you expect to happen to the final
#    val loss? Would you expect it to happen instantly, or would it require
#    letting the loop actually run longer?
# 4. During training, train loss keeps dropping every single evaluation, but
#    val loss starts climbing back up partway through. What's this called,
#    and what does it tell you about what the model is spending its
#    remaining capacity on?
# 5. Why does the training loop save a checkpoint only when
#    `losses["val"] < best_val_loss`, rather than just saving once at the
#    very end of training?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Character-level tokenization sidesteps the **out-of-vocabulary**
#    problem entirely. Notebook 01's word-level tokenizer had to cap its
#    vocabulary at 2000 words and route everything else through an `<unk>`
#    token, meaning any word outside the top 2000 becomes indistinguishable
#    "unknown" mush. A character vocabulary just contains every character
#    that appears in the corpus — there's no such thing as a character the
#    model can't represent, so no `<unk>` escape hatch is needed.
# 2. Because `y[t]` is defined as "the character that came right after
#    `x[0..t]]` in the original text," every position `t` in the window
#    gives the model an independent (input-so-far, correct-next-token) pair
#    to learn from — position 0 predicts what comes after just the first
#    character, position 1 predicts what comes after the first two, and so
#    on up to `block_size`. One forward pass over one window scores all of
#    those predictions simultaneously via `F.cross_entropy` over every
#    position, instead of needing a separate window per prediction.
# 3. `max_iters` would roughly double too, since it's computed as
#    `its_per_sec * TARGET_SECONDS` (rounded to a multiple of 500) — same
#    hardware, same throughput, twice the time budget. Final val loss should
#    generally improve (go lower) since the model gets roughly twice as many
#    gradient updates, though with diminishing returns as training
#    progresses. And no, this isn't instant: `max_iters` doubling means the
#    loop itself must run for roughly twice as long in wall-clock time — the
#    dry run only measures throughput, it doesn't let you skip the actual
#    training.
# 4. This is **overfitting**. Train loss dropping while val loss rises means
#    the model isn't spending its remaining capacity learning patterns that
#    generalize — it's memorizing specific sequences from the training
#    split (getting better and better at predicting the training data
#    exactly) at the expense of patterns that would also apply to text it
#    hasn't seen. This is the same core problem the Bag-of-Words baseline in
#    notebook 01 could suffer from, just visible here as a diverging curve
#    instead of a single accuracy number.
# 5. Because the loss doesn't monotonically improve every single iteration —
#    it can bounce around due to noise in which random batches get sampled,
#    and especially in a longer run, val loss could rise near the end due
#    to overfitting. Saving only on a new best guarantees the checkpoint on
#    disk is always the single best snapshot seen during the whole run,
#    which is what you'd actually want notebook 07 to load from — not
#    whatever the model happened to look like at the arbitrary final
#    iteration.
# </details>
