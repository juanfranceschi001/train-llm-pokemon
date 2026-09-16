# %% [markdown]
# # 03 — Attention
#
# ## Why this notebook gets the most explanation time in the whole course
#
# Notebook 00 already told you this is the big one, and it's worth repeating
# before we start: **attention is the single idea that turns "a model that
# processes text" into "a transformer."** Everything from notebook 04 onward —
# RoPE, RMSNorm, the whole decoder block, training, generation — is scaffolding
# *around* the mechanism you're about to build here. If you only deeply
# understand one notebook in this course, make it this one. That's why we're
# not rushing: every new piece of math below gets an intuition first, a tiny
# hand-checkable numeric example second, and working code third.
#
# Here's the motivating problem, carried over from notebook 02. A **Bag-of-Words**
# (BoW) model represents a stretch of text as word *counts* — it has no idea
# which word came first, second, or last. A mean-pooled **embedding** model is
# subtler but has the same blind spot: it averages every token's vector
# together into one vector, so *where* a word sat in the sentence, and *how
# much more important it is than the words around it*, both get washed out in
# the average. Attention fixes both problems at once: it lets every position
# in a sequence look back at every earlier position and decide, individually
# and by content, how much attention each one deserves.

# %% [markdown]
# ## Working directory setup
#
# Jupyter (and Colab) launch a notebook's kernel from the folder the notebook
# file lives in (`notebooks/`), not the project root — so a path like `data/`
# would silently resolve to `notebooks/data/`, which doesn't exist. This cell
# walks up the directory tree until it finds `requirements.txt` (the project
# root marker) and changes into it. Every notebook in this course repeats this
# exact cell so each one is fully runnable on its own.

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
# ## Imports and setup
#
# `math` gives us the square root we need for scaling scores. `torch.nn` gives
# us `nn.Linear` and `nn.Embedding` — trainable weight matrices. `matplotlib`
# renders the attention-weight heatmap at the end. `torch.manual_seed(1337)`
# pins the random number generator so this notebook produces the same numbers
# every time it runs — useful when you're staring at printed matrices trying
# to understand them.
#
# The device-picking logic is identical to notebook 00's `pick_device()`: try
# CUDA, then Apple's MPS, then fall back to plain CPU. Every tensor in this
# notebook is tiny (at most a few thousand numbers), so it will run in a
# fraction of a second on CPU too — you don't need a GPU for this notebook.

# %%
import math

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend so plt.show() can't block a headless run
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from _shared import ensure_corpus

torch.manual_seed(1337)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = pick_device()
print("Using device:", device)

# %% [markdown]
# ## The example we'll use throughout: two real Pokédex sentences
#
# PokeAPI actually gives us a perfect pair of sentences for this notebook,
# both describing Charmander's tail flame:
#
# > "The flame on its tail shows the strength of its life force. **If it is
# > weak**, the flame also burns **weakly**."
# >
# > "The flame on its tail indicates Charmander's life force. **If it is
# > healthy**, the flame burns **brightly**."
#
# These two sentences share almost every word. The only real difference is one
# early word (**weak** vs. **healthy**) and, as a direct consequence, the very
# last word (**weakly** vs. **brightly**). To correctly predict the ending, a
# model doesn't need to know "some word about condition appeared somewhere
# nearby" — it needs to know *specifically which one*, because that one word
# determines the opposite outcome. This is exactly the kind of long-range,
# content-specific dependency that bag-of-words and mean-pooled embeddings
# cannot represent, and that attention is built to handle. We'll come back to
# this exact pair of sentences three times in this notebook: to show the
# problem, to hand-build a toy solution, and to plot a real attention heatmap.
#
# We pull the sentences straight out of the live corpus (rather than hardcoding
# them) so this notebook stays honest about using real training data — with a
# fallback string in case PokeAPI ever changes its wording, so the notebook
# never crashes on a fresh machine.

# %%
os.makedirs("assets", exist_ok=True)
corpus_path = ensure_corpus()
corpus_text = open(corpus_path, encoding="utf-8").read()


def find_line(must_contain):
    for line in corpus_text.splitlines():
        if all(s in line for s in must_contain):
            return line
    return None


weak_line = find_line(["flame on its tail", "burns weakly"]) or (
    "The flame on its tail shows the strength of its life force. "
    "If it is weak, the flame also burns weakly."
)
healthy_line = find_line(["flame on its tail", "burns brightly"]) or (
    "The flame on its tail indicates Charmander's life force. "
    "If it is healthy, the flame burns brightly."
)
print("weak_line   :", weak_line)
print("healthy_line:", healthy_line)

# %% [markdown]
# ## Part 0 — Proving the problem: BoW/embedding vectors barely notice the difference
#
# Let's make the failure concrete instead of just asserting it. We'll tokenize
# both sentences into lowercase words and build simple **Bag-of-Words count
# vectors**: one entry per unique word across both sentences, counting how many
# times each word appears. This is the same representation notebook 01 used.
#
# If BoW (or an averaged embedding, which has the same blind spot — it also
# throws away position and only keeps "which words appeared, roughly how
# much") really can't tell these sentences apart, their two count vectors
# should point in almost the same direction, even though the two sentences
# demand opposite final words. We measure "almost the same direction" with
# **cosine similarity**: 1.0 means identical direction, 0.0 means completely
# unrelated. Real language has enough shared filler words ("the", "its", "is",
# "flame", "burns"...) that two sentences differing in only one meaningful word
# will still score very high.

# %%
def bow_words(s: str) -> list[str]:
    return "".join(c.lower() if c.isalpha() else " " for c in s).split()


weak_words = bow_words(weak_line)
healthy_words = bow_words(healthy_line)
vocab = sorted(set(weak_words) | set(healthy_words))
word_to_idx = {w: i for i, w in enumerate(vocab)}


def bow_vector(words: list[str]) -> torch.Tensor:
    v = torch.zeros(len(vocab))
    for w in words:
        v[word_to_idx[w]] += 1.0
    return v


weak_vec = bow_vector(weak_words)
healthy_vec = bow_vector(healthy_words)
cos_sim = F.cosine_similarity(weak_vec.unsqueeze(0), healthy_vec.unsqueeze(0)).item()

print(f"Shared vocabulary size: {len(vocab)} words")
print(f"Cosine similarity between the two Bag-of-Words vectors: {cos_sim:.3f}")
assert cos_sim > 0.7, "expected the two BoW vectors to look almost identical"
print("As expected: BoW sees these two sentences as almost the same thing, "
      "even though one must end in 'weakly' and the other in 'brightly'.")

# %% [markdown]
# ## Part 1 — Queries, keys, and values: the intuition before the math
#
# Attention answers one question, asked separately at every position in the
# sequence: *"of everything that came before me, what should I actually pay
# attention to?"* It answers that question with three learned vectors derived
# from each token's embedding:
#
# - **Query (Q)** — *"what am I looking for?"* Computed for the position doing
#   the looking. In our example, the position right after "burns" is looking
#   for "whatever earlier word tells me if this flame is strong or weak."
# - **Key (K)** — *"what do I have to offer, if someone comes looking?"*
#   Computed for every position, including the ones being looked *at*. The
#   word "weak" offers a key that says, in effect, "I'm the condition word."
# - **Value (V)** — *"what do I actually hand over if I get picked?"* Also
#   computed for every position. This is the actual payload that flows
#   forward into the output — Query and Key only decide *how much* of it flows.
#
# The mechanism: compare the asking position's Query against every candidate
# position's Key (a dot product — bigger means "better match"). Turn those
# match scores into weights that sum to 1 (a softmax). Then take a weighted
# sum of every position's Value using those weights. Positions with a good
# Query-Key match contribute a lot of their Value; positions with a poor match
# contribute almost none.
#
# One useful mental model: Key is like a word printed on a library card
# catalog drawer, Query is what you typed into the search box, and Value is
# the actual book you get handed once the catalog decides which drawer matches
# your search best.

# %% [markdown]
# ## Part 2 — A hand-built numeric example
#
# Before writing a general function, let's build one tiny attention step by
# hand with numbers we chose ourselves, so every intermediate matrix is
# checkable by eye. Picture a 4-token toy sequence standing in for the
# beginning of our real sentence:
#
# | position | 0 | 1 | 2 | 3 |
# |---|---|---|---|---|
# | token | "the" | "weak" | "flame" | "burns" |
#
# Position 3 ("burns") is the one about to predict the next word, and it needs
# to end up looking mostly at position 1 ("weak") — not position 0 ("the",
# uninformative filler) and not even position 2 ("flame", its immediate
# neighbor, but also not the word that decides weak-vs-healthy). We'll hand-
# pick tiny 4-dimensional Query/Key/Value vectors that make this happen, then
# run the real steps of attention on them and check the result.
#
# Each key below is a one-hot vector — position `j`'s key is 1.0 in slot `j`
# and 0 everywhere else, so a query's dot product with key `j` just reads off
# that query's value in slot `j`. We give position 3's query a big value (3.0)
# in slot 1 (matching "weak"'s key) and a tiny value (0.1) in slot 3 (matching
# its own key) — modeling "I care a lot about whatever's in slot 1, and only a
# little about myself." The other three queries are set up to mostly attend to
# themselves, which is a reasonable default when nothing else stands out.

# %%
toy_tokens = ["the", "weak", "flame", "burns"]
Q_toy = torch.tensor([
    [2.0, 0.0, 0.0, 0.0],   # "the":   attends mostly to itself
    [0.0, 2.0, 0.0, 0.0],   # "weak":  attends mostly to itself
    [0.0, 0.0, 2.0, 0.0],   # "flame": attends mostly to itself
    [0.0, 3.0, 0.0, 0.1],   # "burns": attends strongly to slot 1 ("weak")
])
K_toy = torch.eye(4)                                  # one-hot keys, one per position
V_toy = torch.tensor([
    [10.0, 0.0, 0.0, 0.0],
    [0.0, 20.0, 0.0, 0.0],   # value carried by "weak"
    [0.0, 0.0, 30.0, 0.0],
    [0.0, 0.0, 0.0, 40.0],
])

# %% [markdown]
# Now the five steps, one cell each, so you can see every intermediate result.
#
# **Step 1 — raw scores.** `Q @ K^T` computes, for every pair of positions
# `(i, j)`, how well position `i`'s query matches position `j`'s key. Row `i`,
# column `j` is "how much position `i` wants to look at position `j`," before
# any normalization.

# %%
scores_raw = Q_toy @ K_toy.T
print("Raw scores (rows = query position, cols = key position):")
print(scores_raw)

# %% [markdown]
# **Step 2 — scale.** Divide by `sqrt(head_dim)` (here `head_dim = 4`, so we
# divide by 2). We'll justify *why* this specific number in the next section —
# for now, just note it shrinks the scores uniformly.

# %%
head_dim_toy = Q_toy.shape[-1]
scores_scaled = scores_raw / math.sqrt(head_dim_toy)
print(f"Scaled scores (divided by sqrt({head_dim_toy}) = {math.sqrt(head_dim_toy):.2f}):")
print(scores_scaled)

# %% [markdown]
# **Step 3 — causal mask.** Position `i` is only allowed to look at positions
# `0..i` (itself and the past) — never the future. We build a lower-triangular
# matrix of `True`/`False` and overwrite every disallowed (upper-triangular)
# score with `-inf`, so it contributes exactly zero after softmax. The next
# section explains *why* this rule exists; here we just apply it.

# %%
causal_mask_toy = torch.tril(torch.ones(4, 4)).bool()
print("Causal mask (True = allowed to look):")
print(causal_mask_toy)
scores_masked = scores_scaled.masked_fill(~causal_mask_toy, float("-inf"))
print("\nMasked scores:")
print(scores_masked)

# %% [markdown]
# **Step 4 — softmax.** Turn each row into a probability distribution: every
# entry is between 0 and 1, and each row sums to exactly 1. These are the
# **attention weights** — the thing we'll plot as a heatmap later in the
# notebook.

# %%
attn_weights_toy = torch.softmax(scores_masked, dim=-1)
print("Attention weights (each row sums to 1):")
print(attn_weights_toy.round(decimals=3))
row_sums = attn_weights_toy.sum(dim=-1)
assert torch.allclose(row_sums, torch.ones(4), atol=1e-6), "each row must sum to 1"

# %% [markdown]
# **Step 5 — weighted sum of values.** Multiply the attention weights by `V`.
# Each output row is a blend of every allowed position's value vector, mixed
# in proportion to the attention weight it received.

# %%
out_toy = attn_weights_toy @ V_toy
print("Output (weighted sum of values):")
print(out_toy.round(decimals=2))

# %% [markdown]
# Check row 3 ("burns"): its attention weight on position 1 ("weak") should be
# by far the largest, and its output should therefore look close to position
# 1's value vector, `[0, 20, 0, 0]`.

# %%
burns_weights = attn_weights_toy[3]
top_position = int(burns_weights.argmax())
print(f"'burns' attends most to position {top_position} "
      f"('{toy_tokens[top_position]}'), with weight {burns_weights[top_position]:.3f}")
assert top_position == 1, "the 'burns' row should attend most strongly to 'weak'"
print("Confirmed: with these hand-picked Q/K/V, 'burns' correctly zeroes in on "
      "'weak' rather than the nearer-but-irrelevant 'flame'.")

# %% [markdown]
# That's the entire mechanism. Everything from here is either "make this
# reusable code" or "do it many times in parallel." In a real trained model,
# nobody hand-picks Q, K, V like we just did — `nn.Linear` layers learn
# projections that produce exactly this kind of useful alignment on their own,
# driven purely by minimizing next-token prediction loss.

# %% [markdown]
# ## Part 3 — Why scale by `sqrt(head_dim)`?
#
# Here's the concrete reason, not just a rule to memorize. A dot product
# between two random vectors of dimension `d` is a sum of `d` independent
# random products — and variance adds up under summation. So as `d` grows, the
# *typical size* of `Q @ K^T` scores grows too (roughly proportional to `d`),
# even though nothing meaningful changed about the vectors themselves.
#
# Large-magnitude scores are a problem for softmax specifically: softmax
# exponentiates its inputs, so once scores get large, the biggest one
# dominates completely and the rest collapse toward zero — the distribution
# becomes nearly one-hot regardless of how meaningfully different the scores
# actually were. That's bad for learning: the gradient with respect to every
# non-dominant position vanishes, so the model gets almost no signal about how
# to adjust them.
#
# Dividing by `sqrt(d)` exactly cancels the growth: if each of the `d` terms in
# the dot product has variance ~1, the sum has variance ~`d`, and dividing by
# `sqrt(d)` brings the variance back down to ~1, *no matter how big `d` gets*.
# Let's check this isn't hand-waving — it's measurable.

# %%
print(f"{'head_dim':>10} | {'raw score variance':>20} | {'scaled score variance':>22}")
for d in (8, 64, 512):
    a = torch.randn(20000, d)
    b = torch.randn(20000, d)
    raw_dot = (a * b).sum(dim=-1)
    scaled_dot = raw_dot / math.sqrt(d)
    print(f"{d:>10} | {raw_dot.var().item():>20.2f} | {scaled_dot.var().item():>22.2f}")

# %% [markdown]
# Notice the raw variance climbs roughly in step with `head_dim` (8 -> 64 ->
# 512), while the scaled variance stays pinned near 1 the whole way. That's
# the scaling step doing its job: it keeps softmax's input in a well-behaved
# range regardless of how large we make `head_dim`.

# %% [markdown]
# ## Part 4 — Causal masking: why a language model must not see the future
#
# A decoder-only language model like the one we're building is trained to
# predict the next token from everything before it, and it's trained on *every
# position of a sequence at once* for efficiency — not one prediction at a
# time. That efficiency trick only works if position `i` is mathematically
# forbidden from looking at position `i+1` or anything later.
#
# Why forbidden, not just "won't happen to look"? Because if it *could* look
# ahead, the fastest way to minimize training loss is to simply copy the
# answer from the token it's supposed to be predicting — attention would
# happily learn to do exactly that, since nothing stops it. Training loss
# would look fantastic. Then at generation time, the model produces text one
# token at a time, left to right — the "future" tokens it learned to peek at
# during training simply don't exist yet. A model trained with an unmasked
# leak has learned to rely on information that will never be available when
# it actually matters, and falls apart at generation time despite having
# looked perfect on paper.
#
# The fix is the causal mask we already used above: `torch.tril` builds a
# lower-triangular matrix of `True`, and we overwrite every `False` (future)
# position's score with `-inf` before the softmax, so it becomes exactly `0`
# attention weight — not "small," genuinely zero, no matter what the raw score
# would otherwise have been.

# %% [markdown]
# ## Part 5 — Writing it as a reusable function
#
# Time to turn Parts 2-4 into one general-purpose function that works for any
# batch size, sequence length, and dimension — not just our hand-picked 4x4
# toy. The five steps are identical to what we did by hand; only the shapes
# are now general: `(batch, seq_len, head_dim)` instead of a single `(4, 4)`
# example.

# %%
def scaled_dot_product_attention(q, k, v, causal=True):
    """The core attention computation for one head.

    Args:
        q, k, v: tensors of shape (B, T, head_dim) -- queries, keys, values.
        causal: if True, position i is only allowed to attend to positions
                0..i (itself and the past), never the future.

    Returns:
        out:     (B, T, head_dim) -- attention output, a weighted mix of v.
        weights: (B, T, T)        -- the attention weights themselves
                                      (returned mainly so we can plot them).
    """
    head_dim = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)   # step 1 + 2
    if causal:
        T = scores.size(-1)
        mask = torch.tril(torch.ones(T, T, device=scores.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))       # step 3
    weights = torch.softmax(scores, dim=-1)                     # step 4
    out = weights @ v                                            # step 5
    return out, weights

# %% [markdown]
# ### Sanity check: causal masking really produces a lower-triangular weight matrix
#
# With `causal=True`, no attention weight should ever appear above the main
# diagonal (`weights[i, j]` for `j > i`) — that would mean position `i` is
# attending to a future position `j`. `weights.triu(1)` extracts everything
# strictly above the diagonal; it must be all zeros.

# %%
B, T, head_dim = 1, 6, 8
q_rand = torch.randn(B, T, head_dim)
k_rand = torch.randn(B, T, head_dim)
v_rand = torch.randn(B, T, head_dim)
out_rand, weights_rand = scaled_dot_product_attention(q_rand, k_rand, v_rand, causal=True)

assert torch.allclose(weights_rand.triu(1), torch.zeros_like(weights_rand), atol=1e-6), \
    "no weight should appear above the diagonal under a causal mask"
assert out_rand.shape == (B, T, head_dim)
print("Causal masking check passed: zero weight above the diagonal, "
      f"output shape {tuple(out_rand.shape)}.")

# %% [markdown]
# ## Part 6 — Multi-head attention: several relationships at once
#
# A single attention head produces exactly one query, one key, and one value
# per token, so it can only really learn to track one *kind* of relationship
# well at a time — say, "which earlier word set the emotional tone." But
# language carries many kinds of relationships simultaneously: which word this
# pronoun refers to, which earlier adjective this noun's condition depends on,
# which move name this ability text is describing, and so on, all at once, all
# in the same sentence.
#
# **Multi-head attention** runs several independent attention computations in
# parallel — each with its own learned Q, K, V projections, operating on a
# smaller slice of the embedding — and then combines their outputs. Splitting
# `n_embd` into `n_head` heads of size `head_dim = n_embd // n_head` costs
# nothing extra in total parameters or compute versus one big head; it just
# gives the model `n_head` independent "search-and-retrieve" mechanisms
# instead of one, each free to specialize in a different kind of pattern.
#
# The shapes: input is `(B, T, n_embd)`. We project it once to `3 * n_embd`
# (packing Q, K, V into a single matrix multiply, then splitting), reshape so
# each head gets its own `(B, T, head_dim)` slice, run
# `scaled_dot_product_attention` per head, concatenate the `n_head` outputs
# back into `(B, T, n_embd)`, and mix them with one final linear layer so
# information from different heads can combine.

# %%
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (teaching version -- loops over heads in
    Python for clarity; a production implementation batches this into one
    matmul, but the math is identical)."""

    def __init__(self, n_embd: int, n_head: int, causal: bool = True):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, n_embd = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # each (B, T, n_embd)

        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        head_outputs = []
        for h in range(self.n_head):
            head_out, _ = scaled_dot_product_attention(q[:, h], k[:, h], v[:, h], causal=self.causal)
            head_outputs.append(head_out)   # (B, T, head_dim)

        combined = torch.cat(head_outputs, dim=-1)   # (B, T, n_embd)
        return self.out_proj(combined)

# %% [markdown]
# ### Sanity check: shape is preserved no matter how many heads
#
# This matters beyond "the numbers work out": later, attention's output gets
# *added back* to its input (a residual connection, notebook 05) — that
# addition is only valid if the shapes match exactly.

# %%
B, T, n_embd, n_head = 2, 12, 64, 4
mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head)
x = torch.randn(B, T, n_embd)
mha_out = mha(x)
assert mha_out.shape == (B, T, n_embd), f"expected {(B, T, n_embd)}, got {tuple(mha_out.shape)}"
print("Multi-head attention output shape check passed:", tuple(mha_out.shape))

# %% [markdown]
# ## Part 7 — Grouped-query attention (GQA): the same idea, cheaper at scale
#
# ### The cost multi-head attention doesn't show you until generation time
#
# During training, a transformer sees the whole sequence at once. During
# **generation**, though, it produces one token at a time, and to avoid
# redoing all the earlier work at every step, real implementations cache every
# past position's Key and Value vectors (the **KV-cache**) instead of
# recomputing them. The cache's size scales with `n_layer x n_head x head_dim
# x sequence_length x 2` (the `2` is for storing both K and V) — and for a
# large model generating a long response, that cache alone can occupy tens of
# gigabytes of memory, often *more* than the model's own weights.
#
# **Grouped-query attention (GQA)**, used in models like Llama 3 and Mistral,
# cuts this cost by keeping the full number of Query heads (`n_head`, so the
# model doesn't lose its many "kinds of relationships" from Part 6) while
# sharing a much smaller number of Key/Value heads (`n_kv_head`) across groups
# of Query heads. Concretely, with `n_head = 8` and `n_kv_head = 2`, every 4
# Query heads share one Key/Value head (`n_rep = n_head // n_kv_head = 4`), and
# the Key/Value projections — and therefore the KV-cache — shrink by exactly
# that same factor of 4, for a small, usually acceptable, quality cost. Two
# named special cases: `n_kv_head == n_head` is ordinary multi-head attention
# (no sharing at all), and `n_kv_head == 1` is **multi-query attention (MQA)**
# — every Query head shares a single Key/Value head, the maximum possible
# memory savings.
#
# ### Implementation
#
# The only change from `MultiHeadAttention` is that Q is projected to the full
# `n_embd` (all `n_head` heads), while K and V are projected to a smaller
# `n_kv_head * head_dim`. Each Key/Value head is then repeated `n_rep` times
# with `repeat_interleave` so its shape lines up with Q's `n_head` heads before
# we hand everything to the same `scaled_dot_product_attention` function from
# Part 5 — GQA reuses the exact same core attention math; only the *projection
# sizes and a repeat step* differ from plain multi-head attention.

# %%
class GroupedQueryAttention(nn.Module):
    """Grouped-query attention (teaching version).

    n_kv_head == n_head -> ordinary multi-head attention (n_rep == 1).
    n_kv_head == 1       -> multi-query attention (maximum KV-cache savings).
    """

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, causal: bool = True):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head     # how many Q heads share each KV head
        self.head_dim = n_embd // n_head
        self.causal = causal

        kv_dim = n_kv_head * self.head_dim   # smaller than n_embd whenever n_kv_head < n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)      # full n_head heads
        self.kv_proj = nn.Linear(n_embd, 2 * kv_dim, bias=False)  # only n_kv_head heads
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)      # (B, n_head, T, head_dim)
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)                # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Broadcast each KV head across its group of Q heads.
        k = k.repeat_interleave(self.n_rep, dim=1)   # (B, n_head, T, head_dim)
        v = v.repeat_interleave(self.n_rep, dim=1)

        head_outputs = []
        for h in range(self.n_head):
            head_out, _ = scaled_dot_product_attention(q[:, h], k[:, h], v[:, h], causal=self.causal)
            head_outputs.append(head_out)

        combined = torch.cat(head_outputs, dim=-1)
        return self.out_proj(combined)

# %% [markdown]
# ### Sanity checks: shape, and the `n_kv_head == n_head` special case
#
# Two things worth confirming with real numbers, not just trusting the prose
# above: (1) GQA's output shape must match plain MHA's, for the exact same
# residual-connection reason as before, and (2) setting `n_kv_head == n_head`
# should give `n_rep == 1` — zero heads shared, i.e. genuinely equivalent to
# ordinary multi-head attention, confirming GQA is a strict generalization of
# MHA rather than a different mechanism.

# %%
B, T, n_embd, n_head, n_kv_head = 2, 12, 64, 8, 2
gqa = GroupedQueryAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
x = torch.randn(B, T, n_embd)
gqa_out = gqa(x)
assert gqa_out.shape == (B, T, n_embd), f"expected {(B, T, n_embd)}, got {tuple(gqa_out.shape)}"
print("GQA output shape check passed:", tuple(gqa_out.shape))
print(f"KV heads shrunk from {n_head} to {n_kv_head} "
      f"-> KV-cache memory shrinks by {n_head // n_kv_head}x versus full MHA.")

gqa_as_mha = GroupedQueryAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_head)
assert gqa_as_mha.n_rep == 1, "n_kv_head == n_head should give n_rep == 1 (no sharing)"
print(f"n_kv_head == n_head gives n_rep = {gqa_as_mha.n_rep}: "
      "confirmed, this degenerates exactly to ordinary multi-head attention.")

# %% [markdown]
# ## Part 8 — Visualizing real attention weights
#
# Let's make this tangible one more time with a heatmap, using the Charmander
# "healthy" sentence we pulled from the real corpus at the top of this
# notebook. We'll build a tiny character-level embedding for it (the same kind
# of embedding notebook 02 used) and run it through one un-trained attention
# head — meaning the Q, K, V projection weights are freshly randomly
# initialized, not learned from data yet.
#
# That matters for reading the plot correctly: an untrained head hasn't yet
# learned *which* words matter (that only emerges after real training, which
# starts in notebook 06). What it *does* already show, guaranteed by the math
# regardless of training, is the causal structure enforced by the mask: every
# position can only ever attend to itself and the past, so the upper-right
# triangle of the heatmap must be completely dark, no matter how the random
# weights come out.

# %%
viz_start = healthy_line.find("If it is healthy")
viz_sentence = healthy_line[viz_start:] if viz_start != -1 else "If it is healthy, the flame burns brightly."
print("Visualizing attention over:", repr(viz_sentence))

chars = sorted(set(viz_sentence))
char_to_idx = {c: i for i, c in enumerate(chars)}
char_ids = torch.tensor([char_to_idx[c] for c in viz_sentence], dtype=torch.long)
T_viz = len(char_ids)

VIZ_DIM = 32   # embedding size, and head_dim, for this visualization only
char_embed = nn.Embedding(len(chars), VIZ_DIM)
W_q, W_k, W_v = (nn.Linear(VIZ_DIM, VIZ_DIM, bias=False) for _ in range(3))

with torch.no_grad():
    x_viz = char_embed(char_ids).unsqueeze(0)          # (1, T_viz, VIZ_DIM)
    q_viz, k_viz, v_viz = W_q(x_viz), W_k(x_viz), W_v(x_viz)
    _, viz_weights = scaled_dot_product_attention(q_viz, k_viz, v_viz, causal=True)

w_plot = viz_weights.squeeze(0).numpy()   # (T_viz, T_viz)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(w_plot, cmap="Blues", vmin=0.0, vmax=w_plot.max())
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")

ax.set_xticks(range(T_viz)); ax.set_xticklabels(list(viz_sentence), fontsize=8)
ax.set_yticks(range(T_viz)); ax.set_yticklabels(list(viz_sentence), fontsize=8)
ax.set_xlabel("attends to (key position j)")
ax.set_ylabel("query position i")
ax.set_title(f'Causal attention weights (untrained) — "{viz_sentence}"')
plt.tight_layout()
plt.savefig("assets/03_attention_weights.png", dpi=120)
plt.show()
print("Saved: assets/03_attention_weights.png")

assert torch.allclose(viz_weights.squeeze(0).triu(1), torch.zeros(T_viz, T_viz), atol=1e-6)
print("Confirmed: upper-right triangle is exactly zero -- no character ever "
      "sees a future character, even with completely untrained weights.")

# %% [markdown]
# ### Reading the plot
#
# - The whole upper-right triangle is dark (zero) — the causal mask at work,
#   exactly as we proved algebraically above.
# - The very first character's row has nowhere to look but itself: a single
#   bright square on the diagonal, because position 0 has no past yet.
# - Later rows have more of the lower triangle available and generally spread
#   weight across more of it, though with random, untrained Q/K/V the *exact*
#   pattern within the allowed triangle is arbitrary — there's no reason yet
#   for it to land on "healthy" specifically.
# - Once this same mechanism sits inside a model that's actually being trained
#   to predict the next character (starting in notebook 06), the Q and K
#   projections stop being random: gradient descent pushes them toward
#   whatever alignment reduces prediction loss, and for a sentence like this
#   one, low loss requires the position right before "brightly" learning to
#   attend strongly back to "healthy" — precisely the behavior we engineered
#   by hand with made-up numbers in Part 2.

# %% [markdown]
# ## Summary
#
# | Piece | What it does |
# |---|---|
# | Query, Key, Value | Q = what I'm looking for; K = what I offer; V = what I actually hand over |
# | `scaled_dot_product_attention` | `softmax((Q @ K^T) / sqrt(head_dim)) @ V` |
# | Scaling by `sqrt(head_dim)` | keeps score variance ~1 regardless of dimension, so softmax doesn't saturate |
# | Causal mask | forces every position to see only itself and the past — required for training/generation to match |
# | Multi-head attention | several independent Q/K/V projections in parallel, each free to specialize |
# | Grouped-query attention | shares fewer Key/Value heads across groups of Query heads — smaller KV-cache, small quality cost |
#
# This is the mechanism. Notebook 04 wraps it with a few modern refinements
# (RoPE, RMSNorm, SwiGLU) and notebook 05 assembles it, alongside a
# feed-forward layer and residual connections, into a full transformer block —
# but the attention computation itself, the thing that actually lets a token
# reach back and pull forward exactly the earlier information it needs, is
# everything you just built by hand, three times over, in this notebook.

# %%
print("Attention built and verified. Continue to 04_modern_components.")

# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. In the Charmander example ("If it is weak, the flame also burns weakly"
#    vs. "If it is healthy, the flame burns brightly"), why do the two
#    sentences' Bag-of-Words vectors end up almost identical even though the
#    sentences need opposite endings? What specifically does attention add
#    that lets a model tell them apart?
# 2. Suppose you accidentally forgot to apply the causal mask while training,
#    but applied it correctly during generation. Training loss would look
#    great. Why would the model then perform terribly at actual generation
#    time, and why wouldn't the low training loss have warned you?
# 3. Which is correct? (a) Value vectors decide how much attention weight one
#    position gives another, while Query and Key carry the content that gets
#    mixed forward. (b) Query and Key decide how much attention weight one
#    position gives another (via their dot product), while Value carries the
#    content that gets mixed forward, weighted by those weights. (c) Query,
#    Key, and Value are three names for the same vector, split three ways
#    purely for numerical stability.
# 4. If you doubled `head_dim` from 32 to 64 but left the scaling factor fixed
#    at `sqrt(32)` instead of updating it to `sqrt(64)`, what would you expect
#    to happen to the softmax outputs, and why?
# 5. A model is configured with `n_head=8, n_kv_head=1`. What is this
#    configuration commonly called, and what's the tradeoff versus
#    `n_kv_head=8`?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Bag-of-Words only counts which words appeared (plus, for mean-pooled
#    embeddings, an unweighted average of their vectors) — it throws away
#    *where* a word sat and *how much more it should matter than its
#    neighbors*. The two sentences share almost every word, so their count
#    vectors point in nearly the same direction (we measured cosine similarity
#    > 0.85). Attention adds a mechanism for the position right before the
#    blank to compute a query that specifically matches the key belonging to
#    "weak" or "healthy," and pull that one position's value forward with far
#    more weight than filler words like "the" or "its" receive — letting the
#    model react to exactly the one word that matters, not to the sentence's
#    word content in bulk.
# 2. Without the mask, every position can attend to every other position,
#    including positions after it — so during training it can simply copy the
#    very token it's supposed to predict through the attention mechanism,
#    driving training (and even validation) loss down without learning
#    anything useful. At generation time tokens are produced one at a time,
#    left to right, so the "future" tokens the model learned to peek at
#    during training don't exist yet — there's nothing there to copy from, and
#    the shortcut it relied on is simply unavailable, so output quality
#    collapses. The training loss doesn't warn you because it was measured
#    under the same leaky setup that caused the problem — the metric itself is
#    invalid, not just optimistic.
# 3. (b). Query and Key only ever interact to produce *weights* (via a dot
#    product, scaled, then softmax) — they never contribute their own values
#    to the output. Value is the only one of the three whose actual numbers
#    end up in the result, mixed together in the proportions the Q/K
#    comparison decided.
# 4. The scores would come out roughly 2x more variance than intended (dot
#    product variance grows roughly linearly with dimension, and doubling
#    `head_dim` without doubling the divisor undoes half of the correction).
#    Larger-magnitude scores make softmax saturate: it collapses toward a
#    near one-hot distribution dominated by whichever score happens to be
#    largest, and the gradient with respect to every other position in that
#    row shrinks toward zero — which makes it much harder for training to
#    adjust those weights usefully.
# 5. This is **multi-query attention (MQA)**, the extreme end of GQA where
#    every Query head shares one single Key/Value head. It shrinks the
#    KV-cache by a full 8x compared to `n_kv_head=8` (a large memory and
#    bandwidth win, especially for long generated sequences), at the cost of
#    the model having less room to store different Key/Value information per
#    head, which typically costs a small amount of quality versus full
#    multi-head attention — modest enough in practice that many production
#    models use something in between (GQA proper, `1 < n_kv_head < n_head`)
#    rather than either extreme.
# </details>
