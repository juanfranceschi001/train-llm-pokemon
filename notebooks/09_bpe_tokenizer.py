# %% [markdown]
# # 09 — BPE Tokenizer From Scratch
#
# **Bonus / advanced — entirely optional.** You already have a complete,
# working, from-scratch LLM after notebook 08, and you understand every piece
# of it. Nothing later in this course depends on this notebook. Read on if
# you're curious how real production models turn text into tokens more
# efficiently than the character-level scheme we've used since notebook 06 —
# skip it with no guilt otherwise.
#
# ## The problem with character-level tokens
#
# Every notebook since 06 has used **char-level tokenization**: one token per
# character, giving a vocabulary of ~80-ish symbols for our Pokémon corpus.
# Simple, and it can never fail to represent a word — but it has two real
# costs:
#
# 1. **Long sequences.** "Charizard" becomes 9 separate tokens. The model has
#    to spend its limited context window (and its limited attention) learning
#    that those 9 characters belong together, over and over, for every word.
# 2. **Wasted capacity.** The model spends effort predicting individual
#    letters instead of learning higher-level patterns.
#
# **Byte-Pair Encoding (BPE)** is the standard fix, used by GPT-2, GPT-3,
# GPT-4, Llama, and nearly every modern LLM. The idea: start from raw bytes
# (so any text can always be represented — never "unknown token"), then
# repeatedly find the most frequent adjacent pair of tokens and merge it into
# one new token. Repeat until you hit your target vocabulary size. Common
# substrings — `"the "`, whole short words, `"ing"` — end up as single
# tokens, so the same text needs fewer tokens to represent.

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
import time

import torch
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend so plt.show() can't block a headless run
import matplotlib.pyplot as plt

from model import GPT, NANO_CONFIG, get_device
from _shared import ensure_corpus

torch.manual_seed(42)
device = get_device()
print("Using device:", device)
os.makedirs("assets", exist_ok=True)

# %% [markdown]
# ## Part 1 — Data: a training slice and a held-out slice
#
# We train BPE on the first 200,000 characters of the Pokémon corpus, and
# hold out a separate 10,000-character chunk that BPE never sees during
# training, to measure compression honestly on text it wasn't fit to.

# %%
corpus_path = ensure_corpus()
with open(corpus_path, encoding="utf-8") as f:
    full_text = f.read()

TRAIN_CHARS = 200_000
HELD_OUT_CHARS = 10_000
train_text = full_text[:TRAIN_CHARS]
held_out = full_text[TRAIN_CHARS:TRAIN_CHARS + HELD_OUT_CHARS]

print(f"Full corpus    : {len(full_text):,} chars")
print(f"BPE training   : {len(train_text):,} chars")
print(f"Held-out chunk : {len(held_out):,} chars (for an honest compression measurement)")

# %% [markdown]
# ## Part 2 — The two building blocks
#
# BPE needs exactly two small functions:
#
# - **`get_stats(ids)`** — count how often every adjacent pair of token ids
#   occurs in a sequence.
# - **`merge(ids, pair, idx)`** — walk the sequence and replace every
#   occurrence of `pair` with one new token id `idx`.
#
# Everything else in BPE is just "call these two functions in a loop."

# %%
def get_stats(ids):
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts

def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

# Tiny worked example so the mechanism is concrete before we run it at scale.
demo_ids = list("pika pika".encode("utf-8"))
demo_stats = get_stats(demo_ids)
best_pair = max(demo_stats, key=demo_stats.get)
print("ids   :", demo_ids)
print("stats :", dict(sorted(demo_stats.items(), key=lambda kv: -kv[1])))
print(f"most frequent pair: {best_pair} = {(chr(best_pair[0]), chr(best_pair[1]))}")
merged = merge(demo_ids, best_pair, 256)
print(f"after merging into token 256: {merged}  (len {len(demo_ids)} -> {len(merged)})")

# %% [markdown]
# ## Part 3 — The BPETokenizer class
#
# `train(text, vocab_size)` starts from the 256 possible byte values (the
# base vocabulary — this is why BPE can never fail to represent a character:
# worst case, it falls back to raw bytes), then merges the most frequent pair
# `vocab_size - 256` times, recording each merge rule.
#
# `encode(text)` re-applies those merge rules in the order they were learned.
# `decode(ids)` reverses it: look up each token's byte sequence and
# concatenate.

# %%
class BPETokenizer:
    def __init__(self):
        self.merges = {}   # (int, int) -> int
        self.vocab = {}    # int -> bytes

    def train(self, text, vocab_size):
        assert vocab_size >= 256, "vocab_size must be at least 256 (one per byte)"
        ids = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        print(f"Trained BPE: {len(self.vocab)} tokens, {len(self.merges)} merges "
              f"| {len(text):,} chars -> {len(ids):,} tokens on the training text")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

# %% [markdown]
# ## Part 4 — Train it on Pokémon flavor text

# %%
VOCAB_SIZE = 512
bpe = BPETokenizer()
t0 = time.time()
bpe.train(train_text, VOCAB_SIZE)
print(f"Training time: {time.time()-t0:.1f}s")

print("\nFirst 15 learned merges:")
for k, (pair, idx) in enumerate(list(bpe.merges.items())[:15]):
    a = bpe.vocab[pair[0]].decode("utf-8", errors="replace")
    b = bpe.vocab[pair[1]].decode("utf-8", errors="replace")
    merged_str = bpe.vocab[idx].decode("utf-8", errors="replace")
    print(f"  merge {k+1:3d}: {a!r} + {b!r} -> {merged_str!r}  (id={idx})")

# %% [markdown]
# ## Part 5 — Round-trip correctness
#
# Every tokenizer must satisfy `decode(encode(text)) == text`. This is the
# single most important property to verify — if it fails, nothing downstream
# can be trusted.

# %%
sample = "Charizard breathes intense flames that can melt almost anything."
encoded = bpe.encode(sample)
decoded = bpe.decode(encoded)
print(f"Original: {sample!r}")
print(f"Tokens  : {len(encoded)} for {len(sample)} chars ({len(sample)/len(encoded):.2f} chars/token)")
assert decoded == sample, f"Round-trip FAILED: got {decoded!r}"
print("Round-trip assert PASSED")

# %% [markdown]
# ## Part 6 — Compression: the actual payoff
#
# **Compression ratio** = char-level token count ÷ BPE token count, measured
# on the held-out chunk BPE never trained on (so this is an honest estimate
# of compression on new text, not text it memorized merge rules from).

# %%
char_tokens = len(held_out)
bpe_tokens = len(bpe.encode(held_out))
compression_ratio = char_tokens / bpe_tokens

print(f"Held-out chunk : {char_tokens:,} chars")
print(f"Char-level     : {char_tokens:,} tokens (1 per char)")
print(f"BPE            : {bpe_tokens:,} tokens")
print(f"Compression    : {compression_ratio:.2f}x  ({compression_ratio:.2f} chars/token)")
print(f"Tokens saved   : {char_tokens - bpe_tokens:,}  ({100*(1 - bpe_tokens/char_tokens):.1f}% reduction)")

assert bpe_tokens < char_tokens, \
    f"BPE should use fewer tokens than char-level, got {bpe_tokens} >= {char_tokens}"
print("Compression assert PASSED")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["Char-level\n(1 token/char)", f"BPE\n(vocab={VOCAB_SIZE})"]
values = [char_tokens, bpe_tokens]
bars = ax.bar(labels, values, color=["#93C5FD", "#2563EB"], edgecolor="white", width=0.5)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 50, f"{v:,}", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Number of tokens (lower = better)")
ax.set_title(f"Compression on a {HELD_OUT_CHARS:,}-char held-out chunk of Pokémon text\n"
             f"BPE compression ratio = {compression_ratio:.2f}x")
ax.set_ylim(0, char_tokens * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("assets/09_compression.png", dpi=120)
plt.show(); plt.close()
print("Saved assets/09_compression.png")

# %% [markdown]
# ## Part 7 — How does this compare to a real production tokenizer?
#
# [tiktoken](https://github.com/openai/tiktoken) is OpenAI's production BPE
# implementation — the exact tokenizer behind GPT-2 through GPT-4. Same
# algorithm we just built, but implemented in Rust for speed and trained on
# an internet-scale corpus with a vocabulary of 50,000-100,000+ tokens
# instead of our 512. This cell is import-guarded: if `tiktoken` isn't
# installed, it prints a skip message and the notebook continues normally.

# %%
try:
    import tiktoken
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    gpt2_heldout = enc_gpt2.encode(held_out)
    print(f"tiktoken GPT-2 vocab size: {enc_gpt2.n_vocab:,}")
    print(f"Held-out chunk ({HELD_OUT_CHARS:,} chars):")
    print(f"  char-level : {char_tokens} tokens")
    print(f"  our BPE    : {bpe_tokens} tokens  (compression={compression_ratio:.2f}x, vocab={VOCAB_SIZE})")
    print(f"  GPT-2 BPE  : {len(gpt2_heldout)} tokens  "
          f"(compression={char_tokens/len(gpt2_heldout):.2f}x, vocab={enc_gpt2.n_vocab:,})")
    print("\nSame algorithm, much bigger vocabulary -> better compression. That's "
          "essentially the whole difference.")
except ImportError:
    print("tiktoken not installed -- skipping reference comparison.")
    print("  Install with: pip install tiktoken>=0.5")

# %% [markdown]
# ## Part 8 — End to end: train the nano model on BPE tokens
#
# Re-tokenize the same 200,000-character slice with our trained BPE, then
# train `NANO_CONFIG` on the resulting (shorter!) token sequence, exactly
# like notebook 06 but with `vocab_size=512` instead of a character count.
# The only real change anywhere in `model.py` or the training loop is that
# one number — everything else about the architecture is tokenization-agnostic.

# %%
print("Encoding text with BPE...")
t0 = time.time()
bpe_ids = bpe.encode(train_text)
print(f"Encoded {len(train_text):,} chars -> {len(bpe_ids):,} BPE tokens "
      f"({time.time()-t0:.1f}s, compression={len(train_text)/len(bpe_ids):.2f}x)")

bpe_data = torch.tensor(bpe_ids, dtype=torch.long)
n_split = int(0.9 * len(bpe_data))
bpe_train, bpe_val = bpe_data[:n_split], bpe_data[n_split:]

BLOCK_SIZE, BATCH_SIZE = NANO_CONFIG.block_size, 32
bpe_cfg = dataclasses.replace(NANO_CONFIG, vocab_size=VOCAB_SIZE)

def get_bpe_batch(split):
    d = bpe_train if split == "train" else bpe_val
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

torch.manual_seed(42)
nano_bpe = GPT(bpe_cfg).to(device)
optimizer = torch.optim.AdamW(nano_bpe.parameters(), lr=3e-3, weight_decay=0.1)

TRAIN_ITERS = 300
losses = []
t0 = time.time()
for it in range(TRAIN_ITERS):
    xb, yb = get_bpe_batch("train")
    _, loss, _ = nano_bpe(xb, targets=yb)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
    if it == 0 or (it + 1) % 50 == 0:
        print(f"  iter {it+1:3d}/{TRAIN_ITERS}  loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")

print(f"\nInitial loss: {losses[0]:.4f}  ->  Final loss: {losses[-1]:.4f}")
assert losses[-1] < losses[0], \
    f"training should reduce loss: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
print("Loss-drop assert PASSED")

# %% [markdown]
# ## Summary
#
# | Built | Result |
# |---|---|
# | `get_stats` / `merge` | the two primitives BPE is built from |
# | `BPETokenizer.train` | learned merge rules from Pokémon flavor text |
# | Round-trip | `decode(encode(text)) == text` — verified |
# | Compression | fewer tokens than char-level on held-out text — verified |
# | Nano-on-BPE | same architecture, same training loop, different tokenizer |
#
# BPE itself didn't require touching `model.py` at all — only `vocab_size`
# changed. That's the real lesson: tokenization and architecture are
# separate, swappable concerns.

# %%
print("Notebook 09 complete.")
print(f"  Vocab size        : {VOCAB_SIZE}")
print(f"  Merges learned    : {len(bpe.merges)}")
print(f"  Compression ratio : {compression_ratio:.2f}x")
print(f"  Nano loss drop    : {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"  assets/09_compression.png written: {os.path.exists('assets/09_compression.png')}")

# %% [markdown]
# ## ✅ Check your understanding
#
# 1. Why does BPE start from raw bytes (0-255) instead of, say, whole words?
# 2. `encode()` re-applies merges "in the order they were learned." Why does
#    that order matter — what could go wrong if you applied them in a
#    different order?
# 3. If you trained BPE with `vocab_size=1024` instead of `512`, would you
#    expect the compression ratio to go up or down? Why?
# 4. This notebook's BPE was trained only on the first 200,000 characters of
#    the corpus. Would you expect it to compress a *completely different*
#    kind of text (say, French poetry) as well as it compresses held-out
#    Pokémon text? Why or why not?
# 5. Notebook 00 called this notebook "bonus." What's the argument that
#    understanding BPE is genuinely optional, given every notebook since 06
#    already worked fine on character-level tokens?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Starting from bytes guarantees the tokenizer can represent *any* text —
#    there's no such thing as an "unknown word." A word-level vocabulary
#    would need an entry for every possible word (and still fail on
#    misspellings, new names, or other languages), forcing an `<unk>` token
#    that throws information away exactly like notebook 01's Bag-of-Words
#    vocabulary cap did.
# 2. Merges must be applied in the order they were learned because later
#    merges were only discovered *after* earlier merges had already
#    collapsed the sequence — a later merge rule might reference a token id
#    that only exists once an earlier merge has already happened. Applying
#    them out of order could miss valid merges or apply a merge to raw bytes
#    that were supposed to already be combined into an earlier token first.
# 3. Up (better compression, i.e. fewer tokens per unit of text). A larger
#    vocabulary means more merge rounds happened, so more and longer common
#    substrings get their own single token, shortening the average encoded
#    sequence -- at the cost of a bigger embedding table.
# 4. Worse. The learned merge rules reflect frequent patterns *in this
#    specific corpus* (English Pokémon flavor text) -- common English
#    substrings, Pokémon name fragments, battle-description phrasing. French
#    text has different frequent letter combinations entirely, so far fewer
#    of our merge rules would apply, and compression would drop toward
#    char-level performance.
# 5. Every notebook from 06 onward already trained, evaluated, and generated
#    text successfully using plain character-level tokens -- BPE is a
#    genuine *efficiency* improvement (shorter sequences, more effective
#    context), not a *correctness* requirement. The architecture in
#    `model.py` doesn't care what the tokens mean, only how many there are,
#    so nothing about "having a working LLM" depends on this notebook.
# </details>
