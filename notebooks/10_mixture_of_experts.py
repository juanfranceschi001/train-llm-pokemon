# %% [markdown]
# # 10 — Mixture-of-Experts: More Brain, Same Bill
#
# **Bonus / advanced / optional.** You already have a complete, working,
# from-scratch LLM after notebook 08 — trained, evaluated, and tuned. Notebook
# 09 gave it a smarter tokenizer. Neither of those was required to say "I
# built a transformer language model from scratch," and neither is this one.
# This notebook is the capstone for anyone who wants to go further: a look at
# **Mixture-of-Experts (MoE)**, the trick that lets real frontier models
# (Mixtral, and reportedly GPT-4 and others) pack in far more parameters than
# they actually use per token. If notebook 08 already felt like a satisfying
# finish line, it's completely fine to stop there — skip this one with no
# guilt.
#
# Here's the whole idea in one sentence: instead of one feed-forward network
# that every token has to share, give the model several smaller ones ("experts")
# and a tiny router that sends each token to only a couple of them. The model
# ends up with a lot more total capacity, but each individual token still only
# costs about the same amount of compute to process — you get more "brain"
# without a bigger compute bill per token.
#
# This notebook builds and trains its **own small model from scratch** — it
# does not reuse notebook 06's saved checkpoint. It does reuse a few
# battle-tested building blocks straight from `model.py` (RMSNorm, the
# attention layer, RoPE) so we're not re-deriving things notebooks 03-05
# already covered; the only genuinely new piece is the feed-forward layer.
#
# What you'll build:
# 1. **Expert** — a small feed-forward network; one of several specialists.
# 2. **Router (gate)** — a tiny linear layer that scores every expert for
#    every token.
# 3. **Top-k routing** — each token only visits its best-scoring `k` experts,
#    out of `N` total — sparse compute, more capacity.
# 4. **Load-balancing auxiliary loss** — why naive routing collapses onto a
#    handful of "popular" experts, and how a second loss term fixes it.
# 5. **MoEGPT** — a compact transformer that uses this MoE layer in every
#    block instead of a single shared feed-forward network.
# 6. A short training run on the Pokémon corpus, plus a bar chart of how
#    evenly the trained router actually spreads tokens across experts.
# 7. **Active vs. total parameters** — the accounting trick that makes MoE
#    such an attractive way to scale up real models.

# %% [markdown]
# ## Working directory setup (same cell every notebook starts with)
#
# Jupyter (and Colab) launch a notebook's kernel from the folder the notebook
# file lives in (`notebooks/`), not the project root — so a bare path like
# `data/pokemon_corpus.txt` would silently resolve to the wrong place. This
# cell walks up until it finds `requirements.txt` (the project root marker)
# and changes into it, so paths and imports behave the same no matter how you
# launched the notebook.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend so plt.show() can't block a headless run
import matplotlib.pyplot as plt

from model import RMSNorm, CausalSelfAttention, build_rope, GPTConfig, get_device

torch.manual_seed(42)
device = get_device()
print("Using device:", device)

# %% [markdown]
# ## Loading the corpus (cold-start safe)
#
# Same rule as every other notebook: never assume a file already exists on
# disk. `ensure_corpus()` fetches and caches `data/pokemon_corpus.txt` from
# PokeAPI if it isn't already there (a few seconds), so this notebook runs
# standalone on a totally empty machine. This is the same corpus every
# notebook trains on, but this notebook builds its **own** character-level
# vocabulary and its **own** small model — it doesn't load anything notebook
# 06 saved.

# %%
from _shared import ensure_corpus

corpus_path = ensure_corpus()
text = open(corpus_path, encoding="utf-8").read()
print(f"Corpus: {len(text):,} characters")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Char-level vocab size: {vocab_size} | train chars: {len(train_data):,} | val chars: {len(val_data):,}")

def get_batch(split, block_size, batch_size):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# %% [markdown]
# ---
# ## Part 1 — Why bother? One generalist vs. several specialists
#
# Every transformer block you've built so far (notebooks 04-08) has exactly
# one feed-forward network (a `SwiGLU` MLP in `model.py`) per block, and
# *every single token* passes through that same MLP, every time. A comma, the
# word "charizard", a digit, a rare move name — all squeezed through one set
# of weights that has to learn to handle all of them at once.
#
# **Mixture-of-Experts** asks: what if, instead, a layer had `N` smaller
# feed-forward networks (the **experts**), and a small learned **router**
# decided — per token — which `k` of those `N` experts are worth consulting?
# Think of it like a Pokémon Center with several specialists on staff instead
# of one overworked generalist nurse: a Fire-type trainer's Charizard doesn't
# need the same specialist as someone's waterlogged Magikarp. You still only
# see one or two specialists per visit, but the Center as a whole covers far
# more ground than any single person could.
#
# | | one shared MLP (what you've built so far) | Mixture-of-Experts |
# |---|---|---|
# | Total feed-forward parameters | 1x | Nx |
# | Feed-forward compute *per token* | 1x | roughly (k/N)x |
# | Effective knowledge capacity | small | large |
#
# The key word is **sparse**: with `N=8` experts and `top_k=2`, the model owns
# 8 experts' worth of parameters, but any given token only ever runs through
# 2 of them. Compute per token stays cheap (close to what a single expert
# would cost), while the model's total capacity to specialize scales with
# `N`. That's the whole MoE bargain, and it's exactly how a model like
# Mixtral-8x7B affords being enormous on paper while staying cheap to run.

# %% [markdown]
# ---
# ## Part 2 — A single expert is nothing new
#
# Each expert here is just a small SwiGLU feed-forward network — the exact
# same shape of MLP used everywhere in `model.py`, just made small enough
# that having several of them is still cheap. Recall from notebook 04:
# `w1` and `w3` project up to a wider hidden dimension, `SiLU` (a smooth
# activation function) gates `w1`'s output, that gets multiplied
# elementwise by `w3`'s output, and `w2` projects back down. Nothing here is
# MoE-specific — a single `Expert` in isolation is indistinguishable from the
# ordinary MLP you already know. The interesting part is what happens once
# there are several of them and a router choosing between them, in Part 3.

# %%
class Expert(nn.Module):
    """One small SwiGLU feed-forward network — architecturally identical to
    the shared MLP in model.py, just one of several specialists instead of
    the only one."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        hidden = 64 * (((int(8 / 3 * n_embd)) + 63) // 64)  # round up to a multiple of 64
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# Quick shape check before building anything bigger on top of it.
_e = Expert(64)
_out = _e(torch.randn(5, 64))
assert _out.shape == (5, 64)
print("Expert output shape:", tuple(_out.shape))

# %% [markdown]
# ---
# ## Part 3 — The router, top-k routing, and why naive routing breaks
#
# ### 3a. The router (gate)
# A single linear layer, `gate`, maps each token's embedding to `n_experts`
# raw scores (**logits**). Softmax turns those into a probability
# distribution over experts — "how much does this token want each expert?"
#
# ### 3b. Top-k selection
# We only keep the `top_k` highest-scoring experts per token and throw the
# rest away, then **renormalize** those `k` kept probabilities so they sum to
# 1 again (a softmax over all `N` experts gives you `k` numbers that sum to
# *less* than 1 once you drop the rest — renormalizing fixes that so the
# weighted combination below is a proper weighted average). The token's
# output is the weighted sum of just those `k` experts' outputs.
#
# ### 3c. Sparse dispatch
# For each chosen expert, we gather only the tokens that picked it and run
# the expert on just those tokens. A token that never picked expert 5 never
# touches expert 5's weights at all — that's the "sparse" in sparse
# activation, and it's *why* MoE compute per token stays cheap regardless of
# how many total experts exist.
#
# ### 3d. Why this collapses without help
# Here's the trap: at the start of training, the router's weights are close
# to random, so it might slightly prefer, say, expert 0. Expert 0 then gets
# most of the training signal (gradient updates only flow through experts
# that actually process tokens), so it gets a little better at its job than
# the others — which makes the router like it even more next time. That
# feedback loop snowballs: within a few hundred steps the router can end up
# routing almost every token to 1-2 "popular" experts while the rest sit
# idle, undertrained, and useless. You'd have paid for `N` experts' worth of
# parameters and be using the compute capacity of maybe 2 of them — the exact
# opposite of what MoE is supposed to buy you.
#
# ### 3e. The fix: a load-balancing auxiliary loss
# The **Switch Transformer** paper (Google, 2022) proposes penalizing
# imbalance directly, by adding a second loss term the optimizer also tries
# to minimize, alongside the usual next-token-prediction loss:
#
# - **Importance** `f_e` — the router's *average* probability for expert `e`
#   across all tokens (how much the router "wants" to use it, on average).
# - **Load** `l_e` — the *fraction* of tokens that actually picked expert `e`
#   as their top-1 choice (how much it's actually used).
# - **Auxiliary loss** `= n_experts * sum_e(f_e * l_e)`.
#
# If routing were perfectly uniform, every `f_e` and every `l_e` would equal
# `1/n_experts`, so each product is `1/n_experts^2`, and summing `n_experts`
# of those and multiplying by `n_experts` gives exactly `1.0` — the
# theoretical minimum. Any expert that's disproportionately popular pushes
# *both* `f_e` and `l_e` above `1/n_experts` for that expert, which makes
# their product — and the total — bigger. So minimizing this loss term
# directly discourages the router from concentrating on a few favorites.
# We'll add `aux_loss` to the training loss with a small weight, so it
# nudges the router toward balance without overpowering the actual
# language-modeling objective.

# %%
class MoEFeedForward(nn.Module):
    """Mixture-of-Experts feed-forward layer: drop-in replacement for a
    single SwiGLU MLP.

    Returns (output, aux_loss) — output has the same shape as the input;
    aux_loss is a scalar that should be added (with a small weight) to the
    training loss to keep routing balanced."""

    def __init__(self, n_embd, n_experts=4, top_k=2, dropout=0.0):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(n_embd, dropout) for _ in range(n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)                              # (n_tok, C) — flatten batch+time

        probs = F.softmax(self.gate(xf), dim=-1)           # (n_tok, n_experts)
        topv, topi = probs.topk(self.top_k, dim=-1)         # each (n_tok, top_k)
        topv = topv / topv.sum(-1, keepdim=True)            # renormalize kept weights to sum to 1

        out = torch.zeros_like(xf)
        for slot in range(self.top_k):
            idx = topi[:, slot]                              # which expert this slot picked, per token
            w = topv[:, slot].unsqueeze(-1)                  # that expert's weight, per token
            for e in range(self.n_experts):
                mask = idx == e                              # tokens that chose expert e in this slot
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](xf[mask])

        # Load-balancing auxiliary loss (Switch-Transformer style).
        importance = probs.mean(0)                                              # (n_experts,)
        top1 = probs.argmax(-1)
        load = torch.bincount(top1, minlength=self.n_experts).float() / xf.size(0)
        aux_loss = self.n_experts * (importance * load).sum()

        return out.reshape(B, T, C), aux_loss

# %% [markdown]
# ### Sanity check: shape and aux loss
#
# Before training anything, confirm the output shape matches the input and
# the aux loss is a real, finite, non-negative number.

# %%
_moe = MoEFeedForward(64, n_experts=4, top_k=2)
_y, _aux = _moe(torch.randn(2, 8, 64))
print("MoE output shape:", tuple(_y.shape), "| aux loss:", round(_aux.item(), 4))
assert _y.shape == (2, 8, 64)
assert torch.isfinite(_aux) and _aux.item() >= 0
print("Shape and aux-loss checks passed.")

# %% [markdown]
# ---
# ## Part 4 — Assembling MoEGPT
#
# Now we plug `MoEFeedForward` into a full transformer, reusing three pieces
# straight from `model.py` instead of re-deriving them:
#
# - `RMSNorm` — the normalization layer from notebook 04.
# - `CausalSelfAttention` — multi-head attention with GQA and RoPE, from
#   notebooks 03-04 (unchanged — MoE only touches the feed-forward sublayer).
# - `build_rope` / `GPTConfig` — RoPE table setup and the hyperparameter
#   dataclass from notebook 04-05.
#
# The residual structure is identical to the `Block` in `model.py`
# (`x = x + Attention(norm(x))`, then `x = x + FeedForward(norm(x))`) — the
# *only* change is that the feed-forward sublayer is now `MoEFeedForward`,
# and each block now also hands back an `aux_loss` scalar that we'll sum
# across every layer.

# %%
class MoEBlock(nn.Module):
    def __init__(self, config, n_experts, top_k):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.moe = MoEFeedForward(config.n_embd, n_experts, top_k, config.dropout)

    def forward(self, x, cos, sin):
        a, _ = self.attn(self.attn_norm(x), cos, sin, kv_cache=None)
        x = x + a
        m, aux = self.moe(self.mlp_norm(x))
        return x + m, aux


class MoEGPT(nn.Module):
    """Same overall shape as model.py's GPT, except every block's SwiGLU MLP
    is replaced by an MoEFeedForward layer. forward() returns
    (logits, ce_loss, aux_loss_total) — aux_loss_total sums every block's
    load-balancing loss."""

    def __init__(self, config, n_experts=4, top_k=2):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [MoEBlock(config, n_experts, top_k) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying, same trick as model.py

        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)

        aux_total = 0.0
        for block in self.blocks:
            x, aux = block(x, cos, sin)
            aux_total = aux_total + aux

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, aux_total

# %% [markdown]
# ## Part 5 — Building a nano MoE model and counting parameters
#
# This notebook's whole point is conceptual, so the model stays deliberately
# tiny — small enough that training finishes in well under a couple of
# minutes on CPU. That's a much shorter run than notebook 06's real training
# loop on purpose: this is a demonstration of *how MoE behaves*, not an
# attempt to train the best possible Pokédex model.
#
# `n_experts=4, top_k=2` means every token consults 2 out of 4 experts —
# half of the expert capacity, per token.

# %%
MOE_CONFIG = GPTConfig(
    vocab_size=vocab_size,
    block_size=64,
    n_layer=3,
    n_head=4,
    n_kv_head=2,
    n_embd=128,
    dropout=0.0,
)
N_EXPERTS = 4
TOP_K = 2

torch.manual_seed(42)
model = MoEGPT(MOE_CONFIG, n_experts=N_EXPERTS, top_k=TOP_K).to(device)

total_params = sum(p.numel() for p in model.parameters())
expert_params_per_block = sum(p.numel() for p in model.blocks[0].moe.experts.parameters())
active_expert_params_per_block = expert_params_per_block * TOP_K // N_EXPERTS
non_expert_per_block = (
    sum(p.numel() for p in model.blocks[0].attn.parameters())
    + sum(p.numel() for p in model.blocks[0].attn_norm.parameters())
    + sum(p.numel() for p in model.blocks[0].mlp_norm.parameters())
    + sum(p.numel() for p in model.blocks[0].moe.gate.parameters())
)
active_params = (
    sum(p.numel() for p in model.tok_emb.parameters())
    + sum(p.numel() for p in model.norm.parameters())
    + MOE_CONFIG.n_layer * (non_expert_per_block + active_expert_params_per_block)
)

print(f"Total parameters (all {N_EXPERTS} experts stored) : {total_params:>9,}")
print(f"Active parameters per token (top-{TOP_K})          : {active_params:>9,}")
print(f"Sparsity ratio (active / total)                : {active_params/total_params:>9.1%}")
print()
print(f"With {N_EXPERTS} experts and top-{TOP_K} routing, every token activates "
      f"only {TOP_K}/{N_EXPERTS} = {TOP_K/N_EXPERTS:.0%} of the model's expert weights, "
      "but all experts are available to be chosen for different tokens.")

# %% [markdown]
# ---
# ## Part 6 — Training the nano MoE
#
# The loop is almost identical to notebook 06's: forward pass, cross-entropy
# loss, backward, optimizer step. The one addition is the auxiliary loss:
#
# ```python
# total_loss = ce_loss + AUX_COEFF * aux_loss
# ```
#
# `AUX_COEFF` is a small weight (`0.01` here) — big enough to meaningfully
# discourage collapse, small enough that the model still prioritizes
# actually predicting the next character correctly. We keep this run short
# (a few hundred steps) — again, the goal is to *see the mechanism working*,
# not to match notebook 06's training budget.

# %%
BATCH_SIZE = 32
MAX_ITERS = 300
LR = 3e-3
AUX_COEFF = 0.01

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

losses = []
initial_loss = None
print(f"Training nano MoE for {MAX_ITERS} iterations...")
t0 = time.time()
for it in range(MAX_ITERS):
    xb, yb = get_batch("train", MOE_CONFIG.block_size, BATCH_SIZE)
    _, ce_loss, aux_loss = model(xb, yb)
    total_loss = ce_loss + AUX_COEFF * aux_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    lv = ce_loss.item()
    losses.append(lv)
    if initial_loss is None:
        initial_loss = lv
    if (it + 1) % 50 == 0:
        print(f"  iter {it+1:4d}/{MAX_ITERS} | ce_loss {lv:.4f} | aux_loss {aux_loss.item():.4f}")

final_loss = losses[-1]
print(f"\nDone in {time.time()-t0:.1f}s | initial ce_loss {initial_loss:.4f} -> final ce_loss {final_loss:.4f}")

# %% [markdown]
# ### Assert: training actually reduced loss
#
# A real check, not a formality — if the loss didn't go down, the model
# didn't learn anything, regardless of how nice the architecture looks.

# %%
assert final_loss < initial_loss, (
    f"Training failed to reduce loss: initial={initial_loss:.4f}, final={final_loss:.4f}"
)
print(f"Training assert passed: ce_loss went {initial_loss:.4f} -> {final_loss:.4f}")

# %% [markdown]
# ---
# ## Part 7 — Did the router actually balance itself?
#
# Time to check whether the load-balancing loss did its job. We run a batch
# of validation text through block 0's router and count how often each
# expert was picked as a token's **top-1** choice (its single
# highest-scoring expert, regardless of what its top-2 partner was). With 4
# experts, perfectly even routing would put 25% of tokens on each one.
#
# Two failure modes the aux loss is specifically meant to prevent:
# - **Dead expert**: an expert gets ~0% of tokens — wasted parameters that
#   never receive gradient signal and never improve.
# - **Dominant expert**: an expert gets nearly 100% of tokens — the model is
#   effectively back to being a dense, single-MLP network wearing an MoE
#   costume, and the other experts are dead weight.

# %%
model.eval()
N_EVAL_BATCHES = 20
counts = torch.zeros(N_EXPERTS, dtype=torch.long)

with torch.no_grad():
    for _ in range(N_EVAL_BATCHES):
        xb, _ = get_batch("val", MOE_CONFIG.block_size, BATCH_SIZE)
        x_emb = model.tok_emb(xb)
        cos = model.rope_cos[:MOE_CONFIG.block_size].to(x_emb.device)
        sin = model.rope_sin[:MOE_CONFIG.block_size].to(x_emb.device)
        a, _ = model.blocks[0].attn(model.blocks[0].attn_norm(x_emb), cos, sin, kv_cache=None)
        x_normed = model.blocks[0].mlp_norm(x_emb + a)
        B, T, C = x_normed.shape
        probs = F.softmax(model.blocks[0].moe.gate(x_normed.reshape(-1, C)), dim=-1)
        top1 = probs.argmax(-1)
        counts += torch.bincount(top1.cpu(), minlength=N_EXPERTS)

utilization = counts.float() / counts.sum()
print("Expert utilization (share of tokens routed here as top-1 choice):")
for e, u in enumerate(utilization):
    bar = "#" * int(u.item() * 40)
    print(f"  Expert {e}: {u.item():5.1%}  {bar}")

# %% [markdown]
# ### Assert: no dead or dominant expert
#
# We don't demand a perfect 25/25/25/25 split — some unevenness is normal —
# but every expert should get a meaningful share (more than 5%) and no
# single expert should hoover up nearly everything (less than 95%).

# %%
assert utilization.min() > 0.05 and utilization.max() < 0.95, (
    f"Load balancing failed: min={utilization.min():.1%}, max={utilization.max():.1%} "
    "-- routing has collapsed onto too few experts"
)
print(f"\nLoad-balancing assert passed: min={utilization.min():.1%}, max={utilization.max():.1%}")

# %%
os.makedirs("assets", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 3.5))
x_pos = range(N_EXPERTS)
ax.bar(x_pos, utilization.numpy() * 100, color="steelblue", edgecolor="white", linewidth=0.5)
ax.axhline(100 / N_EXPERTS, color="tomato", linestyle="--", linewidth=1.5,
           label=f"Ideal ({100/N_EXPERTS:.0f}%)")
ax.set_xticks(list(x_pos))
ax.set_xticklabels([f"Expert {e}" for e in x_pos])
ax.set_ylabel("% of tokens routed here (top-1)")
ax.set_title("Expert utilization — nano MoE, block 0")
ax.set_ylim(0, 60)
ax.legend()
fig.tight_layout()
fig.savefig("assets/10_expert_utilization.png", dpi=150)
plt.show()
print("Saved assets/10_expert_utilization.png")

# %% [markdown]
# ---
# ## Part 8 — Active vs. total parameters: the MoE bargain
#
# This is the property that makes MoE so attractive for scaling up real
# models: you can grow **capacity** (total parameters, i.e. how much the
# model could in principle know) largely independently of **compute**
# (active parameters per token, i.e. what you actually pay to run it).

# %%
print("=" * 58)
print("  Parameter accounting for the nano MoEGPT")
print("=" * 58)
print(f"  Total parameters (all experts stored)  : {total_params:>9,}")
print(f"  Active parameters per token (top-{TOP_K})   : {active_params:>9,}")
print(f"  Sparsity ratio                          : {active_params/total_params:>8.1%}")
print("=" * 58)
print()
print(
    f"A dense model with the same compute cost per token would have roughly "
    f"{active_params:,} parameters. This MoE model has {total_params:,} — "
    f"{total_params/active_params:.2f}x more capacity for about the same FLOPs "
    "per forward pass.\n\n"
    "This is the same trick real frontier models lean on: Mixtral 8x7B has "
    "~47B total parameters split across 8 experts per layer, but only ~13B "
    "of them are active for any given token (top-2 routing) — the inference "
    "cost of a much smaller dense model, with more total knowledge to draw on."
)

# %% [markdown]
# ---
# ## Part 9 — Loss curve

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(losses, color="steelblue", linewidth=0.8, alpha=0.6, label="per-iter loss")
window = 20
if len(losses) >= window:
    smooth = [sum(losses[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(losses))]
    ax.plot(smooth, color="navy", linewidth=2, label=f"smoothed ({window}-iter avg)")
ax.set_xlabel("iteration")
ax.set_ylabel("cross-entropy loss")
ax.set_title("Nano MoE training loss")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Wrapping up
#
# You built a Mixture-of-Experts feed-forward layer from scratch: a router
# that scores experts per token, top-k sparse dispatch, and a load-balancing
# auxiliary loss that stopped the router from collapsing onto one or two
# favorites. You slotted that layer into a transformer whose attention, norm,
# and RoPE machinery you already understood from the core path, trained it
# briefly, and confirmed with real measurements — not just an assertion —
# that routing stayed balanced and total capacity outgrew per-token compute.
#
# That's genuinely the same idea behind why a model like Mixtral can compete
# with much larger dense models while staying cheaper to run: more total
# parameters, but only a slice of them switched on for any one token.
#
# If you're looking for where to go from here, some natural next steps beyond
# this course: instruction tuning / RLHF (turning a raw language model into
# an assistant that follows instructions), quantization (shrinking a trained
# model's memory footprint), and looking at how real serving systems batch
# many users' requests through a model efficiently. None of that changes what
# you now understand about how these models are built — you've seen every
# major structural idea from the ground up.

# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. In your own words, why does sparse top-k routing let a model have more
#    total parameters without a proportional increase in compute per token?
# 2. Suppose you trained this notebook's `MoEFeedForward` with `AUX_COEFF = 0`
#    (no load-balancing loss at all). What would you predict happens to the
#    expert utilization bar chart, and why?
# 3. If you set `top_k = n_experts` (e.g. `top_k=4` with 4 experts), what
#    would the MoE layer effectively become, and would the aux loss still do
#    anything meaningful?
# 4. A learner claims "MoE models are strictly cheaper to train than dense
#    models with the same total parameter count, since only a few experts run
#    per token." What's wrong with that claim? (Hint: think about what "total
#    parameters" costs you even for experts that don't run on a given token.)
# 5. This notebook's `Expert` class is architecturally identical to the
#    `SwiGLU` class in `model.py`. What's the actual difference between "a
#    transformer block with one SwiGLU MLP" and "a transformer block with
#    four `Expert`s behind a router restricted to `top_k=4`"?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Because a token's forward pass only ever runs through `top_k` of the `N`
#    experts, not all of them — the unused experts' weights simply aren't
#    touched for that token, so they cost storage/memory but not compute for
#    it. Adding more experts grows the model's total parameter count (and its
#    capacity to have different specialists learn different things), but the
#    matrix multiplications actually performed per token stay proportional to
#    `top_k`, not `N`.
# 2. Utilization would very likely collapse toward one or two dominant
#    experts (or even a single one), with the rest sitting near 0%. Without
#    the aux loss penalizing concentration, the feedback loop described in
#    Part 3d (an early-favored expert gets more training signal, gets
#    better, gets favored even more) has nothing pushing back against it.
# 3. With `top_k = n_experts`, every token visits *every* expert and their
#    outputs get combined with softmax weights over all of them — that's no
#    longer sparse at all; it's a "soft mixture" that runs full compute
#    through all N experts, closer in spirit to an ensemble than to a normal
#    MoE. The aux loss would still be computable, but it stops being useful:
#    there's no routing collapse to prevent when nothing is being routed
#    sparsely in the first place.
# 4. The claim ignores that all of an MoE model's experts still have to be
#    **stored in memory** (and loaded onto whatever hardware is serving the
#    model) even though only a few run per token — total parameter count
#    still drives memory cost, and during training every expert that
#    receives any tokens across a batch still gets a gradient update. MoE
#    saves *compute* (FLOPs) per token relative to a dense model with the
#    same total parameters — it does not make the memory/storage cost of
#    those parameters disappear.
# 5. With `top_k=4` and 4 experts, every token is routed to all 4 experts —
#    functionally, that's very close to a single wide feed-forward layer,
#    just split into 4 weighted, separately-parameterized pieces combined by
#    softmax weights instead of one set of weights the whole width. The real
#    difference from a single `SwiGLU` MLP only shows up once `top_k < N`:
#    that's what makes routing *sparse* (different tokens see different
#    subsets of experts) rather than just a reparameterized dense layer.
# </details>
