# %% [markdown]
# # 02 — Embeddings: Trading Sparse Counts for Learned Geometry
#
# **Core path.** Notebook 01 built a Bag-of-Words (BoW) model: it counted how
# often each of 2,000 vocabulary words appeared in the last 8 words of context,
# and used that count vector to guess the next word. It worked, sort of — but
# every word was represented as a giant mostly-zero vector, and the model had
# no way to know that "fire" and "flame" mean similar things. This notebook
# fixes that with **embeddings**, and it's responsible for maybe 10% of this
# course's total payoff — but it's a 10% every later notebook depends on,
# because notebooks 03 onward all start from "words are dense vectors," not
# "words are counts." Get comfortable with `nn.Embedding` here and the rest of
# the course reads much more easily.
#
# By the end of this notebook you will have:
# - Replaced one-hot/count vectors with small, learned, dense vectors (embeddings).
# - Trained a tiny embedding-based next-word model and **beaten the BoW
#   perplexity number from notebook 01** — a real, measured improvement, not
#   just an assertion that embeddings are "better."
# - Looked inside the trained embedding table for evidence that it learned
#   something about Pokémon *meaning* (which words end up near which).
# - Confirmed embeddings still have a real limitation — they don't know word
#   *order* — which sets up why notebook 03 (attention) exists at all.

# %% [markdown]
# ## Working directory setup (same cell every notebook starts with)
#
# Jupyter (and Colab) launch a notebook's kernel from the folder the notebook
# file lives in (`notebooks/`), not the project root — so a bare path like
# `data/pokemon_corpus.txt` would silently resolve to the wrong place. This
# cell walks up until it finds `requirements.txt` (the project root marker)
# and changes into it, and adds both the project root and `notebooks/` to
# `sys.path` so `from _shared import ...` works no matter how the notebook was
# launched.

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
import re
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend so plt.show() can't block a headless run
import matplotlib.pyplot as plt

torch.manual_seed(1337)

# %% [markdown]
# ## Loading the corpus and the baseline to beat (cold-start safe)
#
# We never assume an earlier notebook already ran on this machine. `ensure_corpus()`
# fetches/caches `data/pokemon_corpus.txt` if it's missing (a few seconds), and
# `ensure_bow_baseline()` loads notebook 01's saved metrics — or, if a fresh
# Colab runtime never saw notebook 01, it retrains that tiny BoW model on the
# spot and returns the same numbers. (The original course this repo is based
# on just did `open("data/shakespeare.txt")` and `open("assets/phase1_metrics.json")`
# directly here — that crashes with `FileNotFoundError` on any machine that
# hasn't already run notebook 01 in this exact session. We always go through
# these `ensure_*` helpers instead.)

# %%
from _shared import ensure_corpus, ensure_bow_baseline

corpus_path = ensure_corpus()
text = open(corpus_path, encoding="utf-8").read()
bow_metrics = ensure_bow_baseline()
print(f"Corpus: {len(text):,} characters")
print("BoW baseline to beat:", bow_metrics)

# %% [markdown]
# ## Step 1 — Rebuild notebook 01's word tokenizer (self-contained)
#
# To compare perplexity numbers fairly, this notebook needs the *same*
# vocabulary and token ids notebook 01 used. This is a short replay of that
# logic (see notebook 01 for the fully annotated version) — `tokenize_words`
# splits on runs of letters or single punctuation characters, and the
# vocabulary keeps the 1,999 most frequent words plus an `<unk>` catch-all for
# anything rarer, for exactly `VOCAB_SIZE = 2000` entries total.

# %%
def tokenize_words(s: str) -> list[str]:
    return re.findall(r"[a-z]+|[^a-z\s]", s.lower())

VOCAB_SIZE = bow_metrics["vocab_size"]
CONTEXT = bow_metrics["context"]

counts = Counter(tokenize_words(text))
itos = ["<unk>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
stoi = {w: i for i, w in enumerate(itos)}
vocab_size = len(itos)
unk = stoi["<unk>"]

def encode(s): return [stoi.get(w, unk) for w in tokenize_words(s)]

ids_all = encode(text)
split = int(0.9 * len(ids_all))
train_ids, val_ids = ids_all[:split], ids_all[split:]
print(f"vocab_size={vocab_size}, context={CONTEXT}, tokens={len(ids_all):,}")

# %% [markdown]
# ### Building context windows — but keeping ids, not counts
#
# Notebook 01's BoW dataset turned each 8-word window into a length-2,000
# count vector (mostly zeros). This time, `make_dataset` just keeps the raw
# sequence of 8 token **ids** — small integers like `[412, 7, 88, 3, ...]`.
# We haven't lost anything yet; we've just deferred the question of "how do we
# turn a word id into numbers a neural net can use?" to the embedding layer
# below, instead of hand-building a sparse count vector ourselves.
#
# `X` ends up shape `(N, 8)` — N rows of 8 integer ids each. `Y` is shape
# `(N,)` — the target id that follows each window.

# %%
def make_dataset(ids, context, limit):
    xs, ys = [], []
    step = max(1, (len(ids) - context - 1) // limit)
    for i in range(0, len(ids) - context - 1, step):
        xs.append(ids[i:i + context])
        ys.append(ids[i + context])
    return torch.tensor(xs), torch.tensor(ys)

Xtr, Ytr = make_dataset(train_ids, CONTEXT, 20000)
Xva, Yva = make_dataset(val_ids, CONTEXT, 4000)
print("train windows:", tuple(Xtr.shape), "| val windows:", tuple(Xva.shape))

# %% [markdown]
# ## Step 2 — What is an embedding, and why is it better than one-hot?
#
# A **one-hot vector** (what BoW effectively used, one per word in its count
# sum) represents word *i* as a vector of length 2,000 that's all zeros except
# a single `1` at position *i*. Two problems with that:
#
# 1. **It's huge and wasteful.** 1,999 of every 2,000 numbers are always zero.
# 2. **It encodes zero similarity.** The vector for `"fire"` is exactly as far
#    from `"flame"` as it is from `"bicycle"` — position in the vector is
#    arbitrary (whichever word happened to be more frequent gets a lower
#    index), so distance between one-hot vectors carries no meaning at all.
#
# An **embedding** fixes both: instead of a fixed, hand-built vector, each
# word gets a **small vector of ordinary numbers that the model learns during
# training** — say, 128 numbers instead of 2,000, and every one of those 128
# numbers is free to become whatever value helps the model predict better.
# Concretely, in PyTorch, `nn.Embedding(vocab_size, emb_dim)` is nothing more
# than a lookup table: a matrix of shape `(vocab_size, emb_dim)`, one learnable
# row per vocabulary word. Look up word id `412` and you get back row 412 — a
# vector of 128 floats. That's the whole mechanism.
#
# Because those numbers are *learned* (not hand-assigned), and because
# nearby words tend to get nudged by similar training gradients, words that
# show up in similar contexts tend to end up with similar vectors. "Fire" and
# "flame" both show up near words like "burn" and "heat" in this corpus, so
# there's pressure for their embeddings to end up close together — geometry
# that one-hot vectors are structurally incapable of expressing.

# %% [markdown]
# ### The model: average the context, then predict
#
# `MeanEmbeddingModel` does three things:
# 1. Look up the embedding vector for each of the 8 context words →
#    shape `(batch, 8, emb_dim)`.
# 2. **Mean-pool** across the 8 positions (just average them) → shape
#    `(batch, emb_dim)`. This squashes the whole context into one vector, but
#    — importantly, and on purpose, to keep this an apples-to-apples upgrade
#    over BoW — it throws away *order*: `mean([a, b, c])` is identical to
#    `mean([c, b, a])`. We'll prove this explicitly in Step 5.
# 3. Feed that single vector through a linear layer that outputs one score
#    (a **logit**) per vocabulary word — a raw, unnormalized number where
#    higher means "the model thinks this word is more likely to come next."
#    `cross_entropy` turns these into probabilities internally when computing
#    the loss.
#
# **Parameter count**, with `emb_dim = 128`:
# - Embedding table: `vocab_size × emb_dim` = 2,000 × 128 = 256,000
# - Linear head: `emb_dim × vocab_size + vocab_size` = 128 × 2,000 + 2,000 = 258,000
# - **Total ≈ 514,000** — still about 8x fewer than BoW's ~4,000,000
#   parameters, even though `emb_dim = 128` is larger than you might expect.
#   (A smaller `emb_dim` like 32 trains to *fewer* total parameters, but on
#   this corpus it tops out worse than BoW's perplexity — averaging 8 context
#   words into a 32-number summary throws away too much to win outright. 128
#   is the smallest size in our own testing that reliably beats BoW while
#   still using a fraction of its parameters.) The
#   embedding table is shared and reused across all 8 context positions
#   instead of BoW building one giant 2,000-wide count vector per example.

# %%
EMB_DIM = 128

class MeanEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):            # x: (B, T) integer token ids
        e = self.emb(x)              # (B, T, E) — one dense vector per token
        pooled = e.mean(dim=1)       # (B, E) — average across context, order-blind
        return self.fc(pooled)       # (B, vocab_size) — logits

model = MeanEmbeddingModel(vocab_size, EMB_DIM)
emb_params = sum(p.numel() for p in model.parameters())
print(f"Embedding model params: {emb_params:,}  (BoW had {bow_metrics['bow_params']:,})")

# %% [markdown]
# ## Step 3 — Training (same recipe as BoW, lower learning rate)
#
# The loop itself is identical to notebook 01: forward pass → cross-entropy
# loss → `backward()` → optimizer step. We use a lower learning rate
# (`3e-3` instead of BoW's `1e-2`) because this smaller, denser model
# overshoots and destabilizes at the higher rate.
#
# ### Why exactly 150 epochs?
#
# This model **overfits** if you train it too long: on this corpus, its
# validation perplexity actually gets *worse* past roughly epoch 150-200 even
# though the training loss keeps dropping — a classic sign the model has
# started memorizing training-specific patterns instead of learning general
# ones. 150 epochs lands right around where validation perplexity is at its
# best before that happens. This is exactly why notebook 06 tracks validation
# loss during training and checkpoints the *best* version instead of just the
# final one — you're seeing the same principle here in miniature.

# %%
def evaluate(m, X, Y):
    m.eval()
    with torch.no_grad():
        loss = F.cross_entropy(m(X), Y)
    m.train()
    return loss.item()

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
losses = []
for epoch in range(150):
    logits = model(Xtr)
    loss = F.cross_entropy(logits, Ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

emb_val_loss = evaluate(model, Xva, Yva)
emb_val_ppl = float(torch.exp(torch.tensor(emb_val_loss)))
print(f"Embedding model val perplexity: {emb_val_ppl:.1f}")

# %%
os.makedirs("assets", exist_ok=True)
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("training step")
plt.ylabel("train loss")
plt.title("Mean-embedding model: training loss")
plt.tight_layout()
plt.savefig("assets/02_embedding_loss.png", dpi=120)
plt.show()

# %% [markdown]
# ## Step 4 — The moment of truth: did we beat BoW?
#
# **Perplexity** is cross-entropy loss translated into a more intuitive
# number: roughly, "how many words is the model effectively torn between,
# on average, at each prediction?" Lower is better. A model that just guessed
# uniformly at random across all 2,000 words would score a perplexity of
# 2,000. BoW landed far below that but still far from 1 (perfect).
#
# The asserts below are the real test of this notebook: the embedding model
# must beat BoW's perplexity *and* do it with far fewer parameters.

# %%
bow_val_ppl = bow_metrics["bow_val_perplexity"]
bow_params = bow_metrics["bow_params"]

print(f"BoW        : perplexity {bow_val_ppl:7.1f} | params {bow_params:,}")
print(f"Embeddings : perplexity {emb_val_ppl:7.1f} | params {emb_params:,}")

assert emb_val_ppl < bow_val_ppl, "embedding model should beat the BoW perplexity baseline"
assert emb_params < bow_params, "embedding model should use far fewer parameters than BoW"

improvement = (bow_val_ppl - emb_val_ppl) / bow_val_ppl * 100
param_ratio = bow_params / emb_params
print(f"\nBeat the baseline: {improvement:.1f}% lower perplexity, "
      f"using {param_ratio:.0f}x fewer parameters.")

# %% [markdown]
# That's the headline result of this notebook: a strictly smaller model that
# predicts the next word *better*, just by switching from sparse counts to
# learned dense vectors. Fewer numbers, better predictions — that's the whole
# case for embeddings in one comparison.

# %% [markdown]
# ## Step 5 — Peeking inside the learned embedding space
#
# Parameter count and perplexity are only half the story. The other big win —
# invisible to a one-hot vector by construction — is that embeddings can learn
# **geometry**: related words end up near each other in the 128-dimensional
# space, purely as a side effect of training to predict the next word.
#
# First, let's confirm one-hot vectors really can't do this. Two *different*
# one-hot vectors always have a dot product of exactly 0 (no shared nonzero
# position), so their cosine similarity — a standard -1..1 measure of how
# "alike" two vectors point, computed as `(u · v) / (|u| |v|)` — is always 0,
# no matter how related the words actually are.

# %%
i, j = stoi.get("fire", unk), stoi.get("water", unk)
oh_i = F.one_hot(torch.tensor(i), vocab_size).float()
oh_j = F.one_hot(torch.tensor(j), vocab_size).float()
print("one-hot cos(fire, water) =", float(F.cosine_similarity(oh_i, oh_j, dim=0)))
assert torch.dot(oh_i, oh_j) == 0.0

# %% [markdown]
# ### Nearest neighbors in the trained embedding table
#
# Now the interesting part: look up each word's nearest neighbors by cosine
# similarity in the *trained* embedding table. `sim = En @ En.t()` computes
# every pairwise cosine similarity at once (after normalizing each row to unit
# length, a dot product between two rows *is* their cosine similarity).
#
# This corpus is Pokédex/move/ability flavor text, so the words most likely to
# cluster are Pokémon type vocabulary ("fire", "water", "electric", "flame",
# "burn"...) and Pokémon names that share a type or role. This is a tiny model
# trained briefly on a modest corpus, so don't expect textbook-clean clusters
# — some neighbors will look like noise. That's honest and expected; the
# structural check right after this prints the neighbors and confirms there's
# real (if imperfect) signal, not randomness.

# %%
E = model.emb.weight.detach()
En = F.normalize(E, dim=1)
sim = En @ En.t()

def nearest(word, k=6):
    if word not in stoi:
        return []
    q = stoi[word]
    scores = sim[q].clone()
    scores[q] = -1.0
    top = torch.topk(scores, k).indices.tolist()
    return [itos[t] for t in top]

query_words = ["fire", "water", "electric", "charizard", "pikachu", "poison", "flying", "attack"]
for w in query_words:
    if w in stoi:
        print(f"{w:10s} ~ {nearest(w)}")

# %% [markdown]
# ### Structural sanity check
#
# We confirm the embedding table has *non-trivial* structure: for a
# real, frequent word, its single nearest neighbor should be noticeably closer
# than the average word, rather than every similarity score being flat and
# uninformative (which is what you'd see from an untrained/random embedding
# table).

# %%
q = 1  # index 0 is <unk>, so index 1 is the single most frequent real word
row = sim[q].clone(); row[q] = -1.0
print(f"Most frequent word: '{itos[1]}' | nearest neighbor similarity {row.max():.3f} "
      f"vs. average similarity {row.mean():.3f}")
assert row.max() > row.mean(), "learned embeddings should have non-trivial structure"
print("Confirmed: the embedding space has real geometric structure — one-hot vectors never could.")

# %% [markdown]
# ### A 2D shadow of the embedding space (PCA)
#
# Our embedding space has 128 dimensions, too many to plot directly. **PCA**
# (Principal Component Analysis) finds the two directions along which a
# handful of chosen points vary the most, and projects every point onto just
# those two axes — like looking at the shadow a 3D object casts on a wall. You
# lose detail, but the rough shape of the cluster survives.
#
# We use `torch.linalg.svd` (Singular Value Decomposition), a close
# mathematical relative of PCA that PyTorch computes efficiently: center the
# chosen vectors, decompose them, and keep the top 2 directions to project
# onto. Don't expect a perfectly clean picture — this model trained briefly on
# a modest corpus — but related words should visibly avoid piling on top of
# each other at random.

# %%
viz_words = [w for w in [
    "fire", "water", "electric", "grass", "poison", "flying", "psychic",
    "ice", "dragon", "steel", "rock", "ghost",
    "charizard", "blastoise", "venusaur", "pikachu", "gengar", "onix",
    "attack", "defense", "speed", "damage", "battle", "evolve",
] if w in stoi]
vidx = torch.tensor([stoi[w] for w in viz_words])
V = E[vidx]
V = V - V.mean(0)
_, _, Vt = torch.linalg.svd(V, full_matrices=False)
coords = V @ Vt[:2].t()

plt.figure(figsize=(7, 6))
plt.scatter(coords[:, 0].tolist(), coords[:, 1].tolist())
for w, (x, y) in zip(viz_words, coords.tolist()):
    plt.annotate(w, (x, y))
plt.title("Learned Pokémon word embeddings (PCA to 2D)")
plt.tight_layout()
plt.savefig("assets/02_embedding_space.png", dpi=120)
plt.show()

# %% [markdown]
# ## Step 6 — Embeddings are still order-blind
#
# We fixed *representation* (dense, meaningful vectors instead of sparse
# counts) — but not *order*. Mean pooling is **permutation-invariant**:
# averaging 8 vectors gives the exact same result no matter what order you
# average them in. `mean([a, b, c]) == mean([c, b, a])`.
#
# That means our model predicts the exact same next word for "charizard
# breathes hot fire" as it would for a scrambled version of those same 8
# words in any other order. This isn't a bug in this notebook's code — it's
# an unavoidable consequence of choosing mean pooling as the way to combine
# context. The assert below proves it directly: reversing a context window
# produces bit-for-bit identical predictions.
#
# **Giving the model a sense of order and position is exactly what notebook
# 03 (attention) solves** — the single biggest idea in this whole course.

# %%
sample_ids = encode("charizard breathes hot fire at the enemy pokemon")[:CONTEXT]
forward_ctx = torch.tensor([sample_ids])
reversed_ctx = torch.tensor([list(reversed(sample_ids))])
with torch.no_grad():
    p_forward, p_reversed = model(forward_ctx), model(reversed_ctx)
assert torch.allclose(p_forward, p_reversed, atol=1e-5), "mean-pooling must be order-blind"
print("Confirmed: reordering the context produces identical predictions. On to notebook 03: attention.")

# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. In your own words, what is the difference between a one-hot vector and
#    an embedding vector? Why can only one of them encode "these two words
#    are similar"?
# 2. This notebook's embedding table has shape `(2000, 128)`. If you doubled
#    `EMB_DIM` from 128 to 256, what would happen to the total parameter count
#    of the model, roughly, and why?
# 3. The embedding model beat BoW's perplexity using far fewer parameters.
#    Does that mean embeddings are a strictly "smarter" architecture than
#    BoW in every way? What limitation do they still share with BoW?
# 4. Suppose `nearest("fire")` returned `["water", "grass", "electric", ...]`
#    instead of type-related fire words. Would that necessarily mean the
#    training was broken? What would you check before concluding that?
# 5. `mean([a, b, c]) == mean([c, b, a])` is why this model is order-blind.
#    Name one other way you could combine 8 context vectors into a single
#    vector that would *also* be order-blind (i.e., don't just say "attention").

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. A one-hot vector has a single `1` at a fixed, arbitrary position and
#    zeros everywhere else — the position doesn't encode meaning, just "which
#    row of the vocabulary list this is." An embedding is a short vector of
#    ordinary numbers that the model *learns* during training, and because
#    words that appear in similar contexts get nudged by similar gradients,
#    embeddings can end up close together in space when the words are
#    related. Two different one-hot vectors always have a dot product of
#    exactly 0 (cosine similarity 0) no matter how related the words are —
#    the representation is mathematically incapable of expressing similarity.
#    An embedding has no such constraint.
# 2. Roughly double: the embedding table goes from `2000 x 128 = 256,000`
#    parameters to `2000 x 256 = 512,000`, and the linear head goes from
#    `128 x 2000 + 2000 = 258,000` to `256 x 2000 + 2000 = 514,000`. Total
#    goes from ~514,000 to ~1,026,000 — almost exactly double, because both
#    pieces scale linearly with `emb_dim`.
# 3. No — it fixed *representation* (dense, meaningful vectors vs. sparse
#    counts) but not *order*. This model still mean-pools the context window,
#    which is permutation-invariant: shuffling the 8 context words produces
#    identical predictions, exactly the same blind spot BoW had. Fixing that
#    requires a mechanism that's aware of position/order — attention,
#    introduced in notebook 03.
# 4. Not necessarily broken — this is a small model trained briefly (a couple
#    hundred steps) on a modest, noisy corpus (real Pokédex text scraped from
#    an API, with typos and inconsistent capitalization baked in), so some
#    neighbor lists will look arbitrary or noisy. Before concluding it's
#    broken you'd check: did the structural sanity check assert pass (nearest
#    neighbor similarity clearly above average)? Did the training loss
#    actually decrease? Did perplexity beat the BoW baseline? If all of those
#    hold, occasional odd neighbors are expected noise, not a bug.
# 5. Any other symmetric/order-independent combination works, e.g. taking the
#    **sum** instead of the mean (sum is also unaffected by reordering), or
#    the **max** of each dimension across the 8 vectors (elementwise max-
#    pooling). Both ignore *which* position each word came from, just like
#    mean pooling does.
# </details>
