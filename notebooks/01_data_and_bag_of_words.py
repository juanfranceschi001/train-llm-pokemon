# %% [markdown]
# # 01 — Data & a Bag-of-Words Baseline
#
# This is the first notebook on the core path (see notebook 00's 80/20 table). Its
# job is small but important: build the *dumbest reasonable* next-word predictor,
# measure exactly how good it is, and write that number down. Every later notebook
# (embeddings in 02, attention in 03, the full transformer in 05-06) exists to beat
# this number — and beat it by understanding precisely *why* this baseline fails.
#
# That's the 80/20 case for a notebook that doesn't build anything impressive: without
# an honest baseline, "our transformer generates good Pokédex text" is just a vibe. With
# one, it's a measured claim — "perplexity dropped from ~150 to under 10" — that you can
# actually defend. This notebook is 100% infrastructure for that claim; expect it to
# feel simple compared to notebook 03.
#
# By the end you will have:
# - Loaded a real dataset (Pokédex, move, and ability text).
# - Converted words into numbers a neural network can use (tokenization).
# - Built a **Bag-of-Words** dataset that deliberately throws away word order.
# - Trained a single linear layer on it and measured its **perplexity**.
# - Seen a concrete proof that this model literally cannot tell `"charizard breathes fire"`
#   from `"fire breathes charizard"` — which motivates everything that follows.

# %% [markdown]
# ## Working directory setup (same cell every notebook starts with)
#
# Jupyter runs a notebook from the folder it lives in (`notebooks/`), not the project
# root, so a path like `data/pokemon_corpus.txt` would otherwise resolve to the wrong
# place. This walks up to the folder containing `requirements.txt` and treats that as
# home base — see notebook 00 for the full explanation.

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
import json
import re
from collections import Counter

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend so plt.show() can't block a headless run
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from _shared import ensure_corpus

torch.manual_seed(1337)

# %% [markdown]
# ## Step 1 — Get the data
#
# Every notebook in this course trains on the same corpus: Pokédex species
# descriptions, move descriptions, and ability descriptions, pulled from the free
# [PokeAPI](https://pokeapi.co). Each line looks like `"Charizard: Spits fire that is
# hot enough to melt boulders."` or `"The move Ember: The target is attacked with a
# small flame."`.
#
# We never hardcode a download here — `ensure_corpus()` (from `_shared.py`) checks
# whether `data/pokemon_corpus.txt` already exists and, if not, builds it from
# PokeAPI in about 5-15 seconds. That's the cold-start-safety rule this whole repo
# follows: a notebook never assumes an earlier notebook already ran on this machine.

# %%
corpus_path = ensure_corpus()
with open(corpus_path, encoding="utf-8") as f:
    text = f.read()

print(f"Corpus: {len(text):,} characters at {corpus_path}")
print("----- sample -----")
print(text[:300])

# %% [markdown]
# ## Step 2 — Turning text into numbers (tokenization)
#
# A neural network only does arithmetic — it has no idea what a "letter" or a "word"
# is. So the first job is always **tokenization**: chopping text into pieces (called
# **tokens**) and assigning each unique piece an integer id. This notebook tokenizes
# at the **word level** — each word, and each punctuation mark, is one token. (Later
# notebooks switch to character-level and then, in the bonus notebook 09, a learned
# subword scheme called BPE. Word-level is the easiest to reason about, which is why
# we start here.)
#
# ### Step 2a — the chopping rule
#
# `tokenize_words` splits a string into a list of tokens using a regular expression:
# - `s.lower()` — lowercase everything first, so `"Pikachu"` and `"pikachu"` aren't
#   treated as two different words.
# - `[a-z]+` — matches "one or more letters in a row", i.e. a whole word like
#   `pikachu` or `electric`.
# - `[^a-z\s]` — matches a single character that's neither a letter nor whitespace,
#   i.e. a lone punctuation mark like `,` or `.`.
# - The two are joined with `|` ("or"), and spaces match neither pattern, so they're
#   silently dropped — they're separators, not tokens.
#
# For example: `tokenize_words("Pikachu, an Electric type!")` gives
# `['pikachu', ',', 'an', 'electric', 'type', '!']` — six tokens, punctuation kept as
# its own token because `!` versus `.` genuinely changes how a sentence reads.

# %%
def tokenize_words(s: str) -> list[str]:
    return re.findall(r"[a-z]+|[^a-z\s]", s.lower())

print(tokenize_words("Pikachu, an Electric type!"))

# %% [markdown]
# ### Step 2b — the word <-> id dictionary: `WordTokenizer`
#
# Now we need to assign every unique token an integer id. The corpus has a few
# thousand distinct words (species names alone number in the hundreds — `pikachu`,
# `charizard`, `metapod`, ...), and many of them are rare — a species name might show
# up in only two or three lines. Training a model with one output slot per rare word
# wastes parameters on things it will barely ever see.
#
# So we cap the vocabulary at **2,000 words**: keep the 1,999 most frequent tokens,
# and introduce one special catch-all token, `<unk>` ("unknown"), at id 0, standing in
# for every word that didn't make the cut. Any rare species name or obscure word gets
# mapped to `<unk>` instead of getting its own slot.
#
# What the class does:
# - `__init__`: tokenize the whole corpus, count occurrences with `Counter`, keep the
#   top `max_vocab - 1` words, and build two lookup tables — `itos` ("int to string",
#   a list where `itos[5]` is the 5th word) and `stoi` ("string to int", the reverse
#   dictionary).
# - `encode`: tokenize the input and look up each token's id, falling back to `<unk>`'s
#   id for anything not in the vocabulary.
# - `decode`: look up each id's word and join them with spaces (a rough approximation —
#   it won't restore exact punctuation spacing, but it's readable).

# %%
VOCAB_SIZE = 2000  # cap: keeps the model small enough to train on a laptop in seconds

class WordTokenizer:
    def __init__(self, corpus: str, max_vocab: int = VOCAB_SIZE):
        counts = Counter(tokenize_words(corpus))
        most_common = [w for w, _ in counts.most_common(max_vocab - 1)]
        self.itos = ["<unk>"] + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, s: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(w, unk) for w in tokenize_words(s)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

tok = WordTokenizer(text)
print("Vocab size:", tok.vocab_size)

# %% [markdown]
# ### Sanity check: round-trip encoding
#
# Before trusting the tokenizer for anything, verify it can encode a sentence and
# decode it back to the same words. We pick a sentence built entirely from common
# words (so nothing falls back to `<unk>`), and use `assert` to crash loudly if the
# round trip doesn't match exactly — much better than silently training on a buggy
# tokenizer for an hour before noticing.

# %%
sample = "the fire type breathes flame"
ids = tok.encode(sample)
assert tok.decode(ids) == sample, tok.decode(ids)
print("round-trip ok:", ids, "->", tok.decode(ids))

# %% [markdown]
# ## Step 3 — Train / validation split
#
# We split the token stream into two pieces:
# - **Train** (first 90%) — what the model actually learns from.
# - **Validation** (last 10%) — held out and *never* trained on, used only to check
#   whether the model generalizes.
#
# This is one of the most important habits in machine learning: **never judge a model
# using data it trained on**. A model can always get better at data it's already
# memorized; the only honest test is data it has never touched.
#
# We take the *last* 10% as a contiguous block rather than shuffling and randomly
# sampling 10% of individual tokens. The corpus is one continuous stream of text
# (Pikachu's line, then Raichu's line, then the next Pikachu line, and so on) — if we
# shuffled first, a validation token could end up sitting right next to a training
# token it came from, letting information leak across the split. A sequential split
# keeps train and validation cleanly separated in time (or in this case, in
# reading order through the corpus).

# %%
ids_all = tok.encode(text)
n = len(ids_all)
split = int(0.9 * n)
train_ids = ids_all[:split]
val_ids = ids_all[split:]
print(f"tokens: {n:,} (train {len(train_ids):,}, val {len(val_ids):,})")

# %% [markdown]
# ## Step 4 — The Bag-of-Words representation
#
# Here's the core idea for this baseline: look at the `CONTEXT` words right before the
# current position, and predict the next word.
#
# The "Bag-of-Words" twist is *how* we represent those `CONTEXT` words. Instead of
# feeding them in as an ordered sequence (word 1, then word 2, ..., then word 8), we
# throw the order away and just count them. The context window becomes one **count
# vector** of length `vocab_size`, where slot `i` says "how many times did word id `i`
# appear somewhere in this window?" — like dumping a handful of Scrabble tiles into a
# bag: you know which letters are in there, not what order you drew them.
#
# Why would we deliberately cripple the model like this? Because it's a *controlled*
# way to fail. Bag-of-words is genuinely useful for order-insensitive tasks (like "does
# this line mention `fire` at all?"), and by using it here for a task that clearly
# *does* care about order (predicting the next word), we get to measure precisely how
# much accuracy that missing order information costs us.
#
# ### `make_bow_dataset`
#
# For each window position `i`:
# - `window = ids[i : i + context]` — the `context` token ids right before the target.
# - `vec = torch.zeros(vocab_size)`, then `vec[w] += 1.0` for every id `w` in the
#   window — the count vector described above.
# - `ys.append(ids[i + context])` — the target is whatever word actually came next.
#
# `step` subsamples the sliding window so we don't build one example per token (that
# would be hundreds of thousands of near-duplicate 8-word windows); instead we take at
# most `limit` evenly spaced examples, which is plenty to train on and much faster to
# build.
#
# The result: `X` has shape `(N, vocab_size)` (N count vectors) and `Y` has shape
# `(N,)` (N next-word ids).

# %%
CONTEXT = 8

def make_bow_dataset(ids: list[int], context: int, vocab_size: int, limit: int):
    xs, ys = [], []
    step = max(1, (len(ids) - context - 1) // limit)
    for i in range(0, len(ids) - context - 1, step):
        window = ids[i:i + context]
        vec = torch.zeros(vocab_size)
        for w in window:
            vec[w] += 1.0
        xs.append(vec)
        ys.append(ids[i + context])
    return torch.stack(xs), torch.tensor(ys)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

Xtr, Ytr = make_bow_dataset(train_ids, CONTEXT, tok.vocab_size, limit=20000)
Xva, Yva = make_bow_dataset(val_ids, CONTEXT, tok.vocab_size, limit=4000)
Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)
print("BoW train X:", tuple(Xtr.shape), "val X:", tuple(Xva.shape))

# %% [markdown]
# This loop is doing real work (thousands of tiny count vectors, built one word at a
# time), so it can take a little while to finish — that's expected, not a bug.

# %% [markdown]
# ## Step 5 — The model: a single linear layer
#
# Now we need something that turns a count vector into a prediction. We use the
# simplest learnable model that exists: **one linear layer**, nothing else.
#
# A linear layer computes `output = input @ W + b`, where `W` (the **weights**) and
# `b` (the **bias**) are numbers the model is allowed to adjust during training. Our
# input is a count vector of length `vocab_size`; the output is *also* length
# `vocab_size` — one raw score per vocabulary word, one score per word.
#
# Those raw scores are called **logits**. They aren't probabilities yet (a logit can
# be negative, or bigger than 1) — think of a logit as "how much the model likes this
# word right now," where a higher number means more likely. A separate step (folded
# into the loss function below, called **softmax**) turns logits into an actual
# probability distribution that sums to 1.
#
# `nn.Linear(vocab_size, vocab_size)` has `vocab_size * vocab_size` weights plus
# `vocab_size` biases. With `vocab_size = 2000` that's about 4 million numbers — a lot
# to look at, but tiny by modern standards, and small enough to train on a laptop CPU
# in seconds.

# %%
bow = nn.Linear(tok.vocab_size, tok.vocab_size).to(device)
print("BoW params:", sum(p.numel() for p in bow.parameters()))

# %% [markdown]
# ## Step 6 — Training
#
# Training means nudging the model's weights so its predictions get better. A few
# terms, defined the moment they're used:
#
# - **Cross-entropy loss** — one number that measures "how wrong is the model right
#   now?" It converts the logits to probabilities (via softmax) and looks at how much
#   probability the model put on the *correct* next word. Confident and correct → low
#   loss. Confident and wrong → high loss. `nn.functional.cross_entropy` does the
#   softmax-plus-comparison in one call — you hand it raw logits and the correct ids.
# - **Gradient** — after computing the loss, `loss.backward()` computes, for every
#   weight, "which direction would reduce the loss, and by how much." Think of it as
#   the model asking "if I nudged this one number up or down, would `Charizard's move
#   text` be predicted better?" for all 4 million numbers at once.
# - **AdamW optimizer** — the piece that actually applies those nudges. It's a
#   well-tuned variant of gradient descent that adapts its step size per-parameter and
#   adds a small penalty (**weight decay**) that discourages weights from growing huge,
#   which helps the model generalize instead of memorizing. `lr=1e-2` is the
#   **learning rate** — how big a step to take each update.
# - **Epoch** — one full pass over the training data. We train for 60 epochs; since
#   our whole training set fits comfortably in memory, each epoch here is just one
#   more forward pass, loss, backward pass, and weight update on the *entire* training
#   set at once (no mini-batches needed at this tiny scale).
#
# `evaluate` measures the loss on a dataset *without* changing any weights:
# `model.eval()` puts the model in evaluation mode (irrelevant for a plain linear
# layer, but good habit for later notebooks that use dropout), and `torch.no_grad()`
# skips gradient bookkeeping since we're not going to call `.backward()`.

# %%
def evaluate(model, X, Y) -> float:
    model.eval()
    with torch.no_grad():
        loss = nn.functional.cross_entropy(model(X), Y)
    model.train()
    return loss.item()

opt = torch.optim.AdamW(bow.parameters(), lr=1e-2)
EPOCHS = 60
losses = []
for epoch in range(EPOCHS):
    logits = bow(Xtr)
    loss = nn.functional.cross_entropy(logits, Ytr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

val_loss = evaluate(bow, Xva, Yva)
bow_val_ppl = float(torch.exp(torch.tensor(val_loss)))
print(f"final train loss {losses[-1]:.3f} | val loss {val_loss:.3f} | val perplexity {bow_val_ppl:.1f}")

# %%
os.makedirs("assets", exist_ok=True)
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("train loss"); plt.title("BoW training loss")
plt.tight_layout()
plt.savefig("assets/01_bow_loss.png", dpi=120)
plt.show()

# %% [markdown]
# ## Step 7 — Making sense of the number: perplexity
#
# A raw cross-entropy loss like "2.3" doesn't mean much on its own. **Perplexity**
# rescales it into something more intuitive: roughly, *how many candidate words is the
# model effectively choosing between* each time it guesses?
#
# Formally, `perplexity = exp(loss)`. A few reference points:
# - A **perfect** model that always assigns 100% probability to the right word has
#   perplexity 1 (zero uncertainty).
# - A **random** model that spreads probability equally over all 2,000 vocabulary
#   words has perplexity 2,000 (maximally uncertain — every word is equally likely).
# - Our trained BoW model lands somewhere well below 2,000 but well above 1 — it has
#   clearly learned *something* about which words tend to follow which contexts (e.g.
#   that `"fire"` is a plausible-ish next word after a run of Charizard-related tokens),
#   but it's still guessing among a large handful of candidates for every word.
#
# Lower is always better. The exact number you got is printed above and saved below —
# that's the number notebook 02's embedding model has to beat.

# %% [markdown]
# ## Step 8 — The punchline: word order is invisible to this model
#
# Time to prove, not just claim, the limitation this whole notebook exists to expose.
#
# Consider `"charizard breathes fire"` and its reversal, `"fire breathes charizard"`.
# They contain exactly the same three words, just in a different order — and to a
# human reader they mean very different things (one is a fire-breathing dragon, the
# other is nonsense). But a Bag-of-Words count vector only tracks *which* words showed
# up and *how many times* — never *where*. So both phrases produce the identical count
# vector. And since our model is just `output = input @ W + b`, identical inputs are
# mathematically guaranteed to produce identical outputs.
#
# This isn't a bug we could fix by training longer or tuning hyperparameters — it's a
# structural property of throwing away order in the input representation. Any model
# built this way has a hard ceiling on how well it can do, no matter how much data or
# compute you give it. That hard ceiling is exactly why notebooks 02 onward move to
# representations that preserve order.

# %%
phrase_a = tok.encode("charizard breathes fire")
phrase_b = list(reversed(phrase_a))

def bow_vec(ids):
    v = torch.zeros(tok.vocab_size, device=device)
    for w in ids:
        v[w] += 1.0
    return v

va, vb = bow_vec(phrase_a), bow_vec(phrase_b)
assert torch.equal(va, vb), "BoW vectors should be identical regardless of word order"

with torch.no_grad():
    pa = bow(va.unsqueeze(0))
    pb = bow(vb.unsqueeze(0))
assert torch.allclose(pa, pb), "predictions should be identical regardless of word order"
print("Confirmed: '", tok.decode(phrase_a), "' and '", tok.decode(phrase_b),
      "' produce identical predictions. Order is invisible to this model.")

# %% [markdown]
# ## Step 9 — Save the baseline
#
# We write the headline numbers to `assets/phase1_metrics.json` so later notebooks can
# load them and compare instead of just taking our word for it. This notebook *owns*
# this file — it recomputes it fresh every time it runs, rather than reusing a cached
# copy, so the number always reflects the current code. (Notebook 02 will instead call
# `ensure_bow_baseline()` from `_shared.py`, which only recomputes this if the file is
# somehow missing — the one real computation lives here.)
#
# Saved fields:
# - `bow_val_perplexity` — the headline number every later notebook tries to beat.
# - `bow_params` — parameter count, so later comparisons are apples-to-apples on
#   "how much model did it take to get this score."
# - `context` and `vocab_size` — the settings used, so later notebooks can match them
#   for a fair comparison.

# %%
os.makedirs("assets", exist_ok=True)
metrics = {
    "bow_val_perplexity": bow_val_ppl,
    "bow_params": sum(p.numel() for p in bow.parameters()),
    "context": CONTEXT,
    "vocab_size": tok.vocab_size,
}
with open("assets/phase1_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved baseline:", metrics)

# %% [markdown]
# ## Where this leaves us
#
# We now have an honest, measured baseline: a Bag-of-Words model that gets *some*
# signal from word frequency but is structurally blind to word order, sitting at the
# perplexity printed above. That number isn't the point — the *reason* for it is. It
# tells us precisely what the next few notebooks need to fix: give the model a way to
# know not just *which* words came before, but *in what order*.
#
# Notebook 02 takes the first step toward that: replacing sparse 2,000-slot count
# vectors with dense **embeddings** — a handful of numbers per word that capture
# meaning far more efficiently than a count ever could (you'll see `charizard` land
# near `fire` and `dragon` in that space). It's still not order-aware — that's
# notebook 03's job — but it's a big step down in parameters and up in quality, and
# you'll watch perplexity drop against the exact number we just saved.

# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. Why does the Bag-of-Words model give the exact same prediction for
#    `"charizard breathes fire"` and `"fire breathes charizard"`? Is this a bug that
#    more training data could fix?
# 2. If you doubled `VOCAB_SIZE` from 2,000 to 4,000, what would happen to `bow_params`
#    (the total parameter count)? Give the rough new number.
# 3. Why do we split train/validation by taking the *last* 10% of the token stream,
#    instead of randomly shuffling all tokens and holding out a random 10%?
# 4. A model that always assigns equal probability to all 2,000 vocabulary words would
#    have what perplexity? What about a model that always predicts the true next word
#    with 100% confidence?
# 5. `<unk>` always gets id 0 in `WordTokenizer`. What happens if `tok.encode(...)` is
#    called on a sentence containing a rare Pokémon name that wasn't in the top 1,999
#    words?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Because the Bag-of-Words representation converts a window of words into a *count
#    vector* that only records how many times each word appeared, never their
#    positions. Both phrases contain the same three words, so they produce identical
#    count vectors, and a linear model (`output = input @ W + b`) always produces
#    identical outputs for identical inputs. No amount of extra training data fixes
#    this — it's a structural limitation of the input representation itself, not
#    something the model could learn its way around.
# 2. `bow_params = vocab_size * vocab_size + vocab_size` (weights plus biases).
#    Doubling `vocab_size` from 2,000 to 4,000 roughly *quadruples* the weight matrix
#    (2000² = 4,000,000 to 4000² = 16,000,000) since it grows with the square of
#    vocab size, so total params would go from about 4.0 million to about 16.0
#    million — not a 2x increase, a 4x increase.
# 3. The corpus is one continuous stream of text read in order. If we shuffled first,
#    a "validation" token could end up right next to a token it trained on, letting
#    information leak across the split and making the validation score overly
#    optimistic. Taking a contiguous trailing block keeps the two sets cleanly
#    separated, so the validation score honestly reflects performance on text the
#    model never saw anything adjacent to during training.
# 4. Equal probability over all 2,000 words gives perplexity 2,000 (maximum
#    uncertainty — the model is "choosing between" all 2,000 candidates equally). A
#    model that's always 100% confident and correct gives perplexity 1 (zero
#    uncertainty). Our trained model should land somewhere between these two extremes,
#    much closer to 1 than to 2,000, but still far from 1.
# 5. It gets mapped to id 0, the `<unk>` token — `WordTokenizer.encode` falls back to
#    `stoi.get(w, unk)`, so any word not in the top 1,999 kept words (including a rare
#    species name) is treated identically to every other out-of-vocabulary word. This
#    is expected: it's exactly what capping the vocabulary at 2,000 was designed to
#    do, at the cost of losing the ability to tell rare words apart from each other.
# </details>
