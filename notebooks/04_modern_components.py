# %% [markdown]
# # 04 — Modern Components: RoPE, RMSNorm, SwiGLU
#
# **This is a core-path notebook (00-08).** Notebook 03 built attention itself —
# the mechanism that lets a token look at other tokens. This notebook covers
# three smaller but still important pieces that sit *around* attention in every
# modern transformer (Llama, Mistral, Gemma, and the model you'll build in
# notebook 05): how the model knows word order (RoPE), how it keeps numbers from
# blowing up during training (RMSNorm), and how each token's representation gets
# refined between attention layers (SwiGLU).
#
# **Why this notebook matters, and how much:** none of these three ideas is as
# conceptually big as attention. But all three are load-bearing — swap any one
# of them out for nothing at all and training breaks or the model can't learn
# word order. They're also short: each one is a five-line function or class. The
# 80/20 trade here is that a small amount of study time buys you the ability to
# read the source of *any* modern open-weight LLM release and recognize exactly
# these three names. By the end, you'll have built a small version of each one
# yourself and checked it against `model.py` — the real implementation notebook
# 05 assembles into the full model.

# %% [markdown]
# ## Working directory setup
#
# Jupyter (and Colab) launches a notebook's kernel from wherever the notebook
# file lives (`notebooks/`), not the project root — so a path like `data/`
# would silently resolve to `notebooks/data/`, which doesn't exist. This cell
# walks up until it finds `requirements.txt` (the project root marker) and
# changes into it. Every notebook in this course starts with this exact cell so
# each one runs correctly no matter how it was launched.

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
# We use plain `torch` for tensors and math, `matplotlib` for the plots, and
# `model.py`'s real implementations at the end of each section to check our
# illustrative versions against the ones notebook 05 will actually use.

# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPTConfig, RMSNorm, SwiGLU, apply_rope, build_rope, rotate_half

SEED = 1337
torch.manual_seed(SEED)
os.makedirs("assets", exist_ok=True)
print("Ready.")

# %% [markdown]
# ---
# ## Part 0 — Why a transformer needs *some* way to know word order
#
# Before getting into RoPE specifically, it's worth seeing the problem it
# solves. **Self-attention, on its own, has no idea what order its inputs came
# in.** It's a *set* operation: every token computes a weighted average over
# every other token's value vector, and that weighted average depends only on
# *which* tokens are present, not on *where* they sit in the sequence.
#
# Here's a direct demonstration. We'll build three fake "token embeddings" —
# one standing in for "Pikachu", one for "used", one for "Thunderbolt" — and
# run plain scaled dot-product attention (no position information at all) over
# two different orderings of the same three tokens.

# %%
torch.manual_seed(SEED)
d = 8
tok_vecs = {name: torch.randn(d) for name in ["pikachu", "used", "thunderbolt"]}


def attention_no_position(seq: list[str]) -> torch.Tensor:
    """Plain scaled dot-product attention over a list of token names, with no
    positional information injected anywhere. Returns one output row per token,
    in the order given."""
    x = torch.stack([tok_vecs[name] for name in seq])          # (T, d)
    scores = (x @ x.T) / (d ** 0.5)                             # (T, T)
    weights = F.softmax(scores, dim=-1)
    return weights @ x                                          # (T, d)


order_a = ["pikachu", "used", "thunderbolt"]     # "Pikachu used Thunderbolt"
order_b = ["thunderbolt", "pikachu", "used"]     # same 3 tokens, shuffled

out_a = attention_no_position(order_a)
out_b = attention_no_position(order_b)

# "pikachu" sits at index 0 in order_a and index 1 in order_b. If attention
# were sensitive to order, moving pikachu to a different slot in the sequence
# should change its output. It doesn't:
pika_in_a = out_a[order_a.index("pikachu")]
pika_in_b = out_b[order_b.index("pikachu")]
assert torch.allclose(pika_in_a, pika_in_b, atol=1e-6), \
    "expected pikachu's attention output to be identical regardless of position"
print("pikachu's attention output is identical in both orderings:",
      torch.allclose(pika_in_a, pika_in_b, atol=1e-6))

# %% [markdown]
# That assert passing is the whole point: attention computed *the same output*
# for "pikachu" whether it appeared first or second, because attention only
# ever asks "which other tokens are present and how relevant is each one" —
# never "where am I, and where are they." As far as raw attention is concerned,
# "Pikachu used Thunderbolt" and "Thunderbolt used Pikachu" are the same bag of
# three tokens. Obviously those two sentences mean different things, so the
# model needs *some* explicit signal for order. The rest of this section
# compares two ways to add it.

# %% [markdown]
# ---
# ## Part 1 — RoPE: encoding position by rotation
#
# ### The naive fix, and why it falls short
#
# The original transformer paper's fix was **learned absolute position
# embeddings**: a lookup table, `nn.Embedding(max_seq_len, n_embd)`, where row
# `p` is a learned vector added to whatever token sits at position `p`. GPT-2
# used exactly this.
#
# It works, but it has two real costs:
#
# - **Every position is learned independently.** The vector for position 5 and
#   the vector for position 6 have no built-in relationship — the model has to
#   *discover from data* that they're neighbors. Nothing in the architecture
#   tells it "5 and 6 are one apart" the way it would for, say, two numbers.
# - **It can't go past the length it was trained on.** The table only has rows
#   for `0` to `max_seq_len - 1`. Position 2048 simply doesn't have a row if
#   the model only ever saw sequences up to 2048 long — there's no reasonable
#   way to even look it up, let alone extrapolate to it.
#
# ### RoPE's idea: encode position as a rotation, not an addition
#
# **RoPE (Rotary Position Embedding)** takes a different approach entirely: it
# doesn't touch the token embeddings at all. Instead, right before the
# attention dot product, it **rotates** each query and key vector by an angle
# that depends on its position.
#
# Why rotation specifically? Because of a clean mathematical fact: if you
# rotate one vector by angle `p * theta` and another by angle `q * theta`, the
# angle *between* them after rotation depends only on `p - q` — the relative
# offset — not on `p` and `q` individually. So the dot product between a
# rotated query at position `p` and a rotated key at position `p + k` comes out
# the same no matter what `p` is, as long as the offset `k` stays the same.
# **RoPE bakes relative position directly into the geometry**, instead of
# hoping the model learns it from independently-assigned position vectors —
# and because it's a fixed rotation formula (not a lookup table with a hard
# maximum row), it has no built-in wall at the training length.
#
# ### The mechanics
#
# Split each head's `head_dim` values into pairs. Pair `i` gets rotated by
# angle `position * theta_i`, where lower-numbered pairs rotate slowly (they
# end up encoding long-range position information) and higher-numbered pairs
# rotate quickly (short-range, fine-grained position information):
#
# ```
# theta_i = 1 / base^(2i / head_dim)      base = 10000 by convention
# angle(position, i) = position * theta_i
# ```
#
# Rather than rotating literal neighboring pairs `(x0, x1), (x2, x3), ...`, the
# common implementation (used here and in `model.py`) pairs each element in the
# first half of the vector with the matching element in the second half, and
# rotates both together:
#
# ```
# rotate_half([a, b]) = [-b, a]                 # a, b are each half the vector
# rotated = x * cos(angle) + rotate_half(x) * sin(angle)
# ```

# %%
def my_build_rope(head_dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute cos/sin rotation tables, one row per position, one
    column-pair per (repeated) frequency. Shapes: (seq_len, head_dim)."""
    i = torch.arange(0, head_dim, 2).float()          # pair indices: 0, 2, 4, ...
    inv_freq = 1.0 / (theta ** (i / head_dim))          # one frequency per pair
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, inv_freq)           # (seq_len, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)        # (seq_len, head_dim)
    return angles.cos(), angles.sin()


def my_rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat([-second_half, first_half], dim=-1)


def my_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (..., T, head_dim); cos/sin: (T, head_dim), broadcast over leading dims.
    return x * cos + my_rotate_half(x) * sin


# %% [markdown]
# ### Check: does our version match `model.py`?
#
# `build_rope`/`apply_rope` are pure math — no learned weights, no randomness —
# so if our formula is right, it should match `model.py`'s implementation
# exactly for the same inputs.

# %%
head_dim, seq_len = 16, 64
my_cos, my_sin = my_build_rope(head_dim, seq_len)
real_cos, real_sin = build_rope(head_dim, seq_len)
assert torch.allclose(my_cos, real_cos) and torch.allclose(my_sin, real_sin), \
    "my_build_rope doesn't match model.py's build_rope"

x_probe = torch.randn(2, 3, seq_len, head_dim)   # (B, n_head, T, head_dim)
assert torch.allclose(my_apply_rope(x_probe, my_cos, my_sin),
                       apply_rope(x_probe, real_cos, real_sin)), \
    "my_apply_rope doesn't match model.py's apply_rope"
print("RoPE: illustrative implementation matches model.py exactly.")

# %% [markdown]
# ### The key property: dot products depend only on relative distance
#
# Let's verify the claim directly. Take one fixed query vector and one fixed
# key vector (the *content* never changes). Rotate them at two different pairs
# of absolute positions that share the same offset `k = 5`: `(10, 15)` and
# `(30, 35)`. If RoPE really encodes relative position, the two dot products
# must match even though the absolute positions are completely different.

# %%
torch.manual_seed(SEED)
q_vec = torch.randn(1, 1, 1, head_dim)
k_vec = torch.randn(1, 1, 1, head_dim)
offset = 5

def rope_dot(base_pos: int, offset: int) -> float:
    q_rot = my_apply_rope(q_vec, my_cos[base_pos:base_pos + 1], my_sin[base_pos:base_pos + 1])
    k_rot = my_apply_rope(k_vec, my_cos[base_pos + offset:base_pos + offset + 1],
                           my_sin[base_pos + offset:base_pos + offset + 1])
    return (q_rot * k_rot).sum().item()

dot_10_15 = rope_dot(10, offset)
dot_30_35 = rope_dot(30, offset)
print(f"dot product at positions (10, 15): {dot_10_15:.6f}")
print(f"dot product at positions (30, 35): {dot_30_35:.6f}")
assert abs(dot_10_15 - dot_30_35) < 1e-4, "RoPE relative-position property failed"
print("Same offset -> same dot product, regardless of absolute position. RoPE assert passed.")

# %% [markdown]
# ### Seeing it as a curve: RoPE vs. the naive alternative
#
# Now let's sweep the offset from 0 to 40 starting from two *different*
# absolute positions and plot the resulting dot product. With RoPE, the two
# curves (start=0 and start=20) should lie exactly on top of each other,
# because both only depend on the offset. For contrast, we do the same thing
# with random, independently-drawn "naive absolute" position vectors (standing
# in for a learned absolute position embedding table before it's been trained)
# — there, nothing ties position 0's vector to position 20's vector, so the two
# curves should look unrelated.

# %%
offsets = list(range(0, 41))
rope_curve_0 = [rope_dot(0, k) for k in offsets]
rope_curve_20 = [rope_dot(20, k) for k in offsets]

torch.manual_seed(SEED)
naive_pos_table = torch.randn(100, head_dim)      # stand-in for an untrained
                                                    # learned position embedding
def naive_dot(base_pos, k):
    return (naive_pos_table[base_pos] * naive_pos_table[base_pos + k]).sum().item()

naive_curve_0 = [naive_dot(0, k) for k in offsets]
naive_curve_20 = [naive_dot(20, k) for k in offsets]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
axes[0].plot(offsets, naive_curve_0, label="start position 0")
axes[0].plot(offsets, naive_curve_20, label="start position 20")
axes[0].set_title("Naive per-position vectors\n(no relationship between positions)")
axes[0].set_xlabel("relative offset k")
axes[0].set_ylabel("dot product")
axes[0].legend()

axes[1].plot(offsets, rope_curve_0, label="start position 0")
axes[1].plot(offsets, rope_curve_20, label="start position 20", linestyle="--")
axes[1].set_title("RoPE\n(depends only on offset k)")
axes[1].set_xlabel("relative offset k")
axes[1].legend()

fig.tight_layout()
fig.savefig("assets/04_rope_relative_position.png", dpi=120)
plt.close(fig)
print("Saved assets/04_rope_relative_position.png")
print("Note how the two RoPE curves overlap almost perfectly, while the naive curves don't.")

# %% [markdown]
# ---
# ## Part 2 — RMSNorm: keeping activations under control, more cheaply
#
# ### Why normalize at all
#
# As data flows through a deep network, the numbers (**activations**) at each
# layer can drift — growing larger and larger, or shrinking toward zero, layer
# after layer. Either extreme makes training unstable: gradients explode or
# vanish, and the loss stops improving (or becomes `NaN`). **Normalization**
# rescales activations back to a controlled range at every layer, without
# taking away the model's ability to learn — a small learned scale parameter
# lets it undo the rescaling if that's ever useful.
#
# Here's the growth problem made concrete: repeatedly multiply a random vector
# by a random matrix (standing in for what a stack of un-normalized layers
# does) and watch its magnitude over "layers".

# %%
torch.manual_seed(SEED)
n_fake_layers = 15
dim = 64
x = torch.randn(dim)
weight_matrices = [torch.randn(dim, dim) * 0.15 for _ in range(n_fake_layers)]

norms_unnormalized = [x.norm().item()]
h = x.clone()
for W in weight_matrices:
    h = h @ W
    norms_unnormalized.append(h.norm().item())

print("Vector norm after each un-normalized fake layer:")
print([f"{v:.1f}" for v in norms_unnormalized])
print("Unchecked, the magnitude drifts by orders of magnitude across depth —")
print("exactly the instability normalization exists to prevent.")

# %% [markdown]
# ### LayerNorm: the classic fix
#
# **Layer normalization** (Ba et al., 2016) normalizes each token's vector
# independently, in four steps:
#
# 1. Compute the mean across the feature dimension: `mu = mean(x)`.
# 2. Compute the standard deviation: `sigma = std(x)`.
# 3. **Mean-center and rescale**: `x_hat = (x - mu) / (sigma + eps)`.
# 4. Apply a learned scale `gamma` and shift `beta`: `output = gamma * x_hat + beta`.
#
# Step 3's `x - mu` is called **mean-centering** — it forces the output to have
# zero mean before the learned shift is applied.
#
# ### RMSNorm: keep the rescaling, drop the rest
#
# **RMSNorm** (Zhang & Sennrich, 2019) asks: how much of that actually matters?
# Empirically, mean-centering and the bias term contribute little to model
# quality, but they do cost extra computation on every single layer of every
# single token — and a transformer normalizes constantly. RMSNorm keeps only
# the part that controls the *scale* (the thing actually causing the
# instability above) and throws out the rest:
#
# ```
# RMS(x) = sqrt(mean(x^2))
# output = x / RMS(x) * gamma          # no mean subtraction, no bias
# ```
#
# In PyTorch, `torch.rsqrt(a)` computes `1 / sqrt(a)` in one fused op, so:
# `output = x * rsqrt(mean(x^2) + eps) * gamma`.

# %%
class MyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# %% [markdown]
# ### Check 1: it controls the scale
#
# With `weight` at its initial value of all ones, RMSNorm's output should have
# an RMS of ~1.0 per row, regardless of the input's original scale.

# %%
torch.manual_seed(SEED)
my_norm = MyRMSNorm(dim)
x = torch.randn(8, dim) * 37.0            # deliberately large scale
out = my_norm(x)
row_rms = out.pow(2).mean(-1)
assert torch.allclose(row_rms, torch.ones_like(row_rms), atol=1e-2), \
    f"expected per-row RMS ~= 1, got {row_rms}"
print("Per-row RMS after normalization (input was scaled by 37x):", row_rms[:3].tolist())

# %% [markdown]
# ### Check 2: it does *not* force zero mean (unlike LayerNorm) — and matches model.py
#
# This is the easy misconception to have: "RMSNorm still normalizes, so its
# output must still have zero mean." It doesn't — that was specifically the
# part RMSNorm removes. We can see that directly, and also confirm our version
# lines up exactly with `model.py`'s `RMSNorm` (both use the same `eps` and a
# weight of all ones by default, so there's no randomness to account for).

# %%
layernorm = nn.LayerNorm(dim)
real_norm = RMSNorm(dim)

out_ln = layernorm(x)
out_my_rms = my_norm(x)
out_real_rms = real_norm(x)

print(f"LayerNorm output mean (row 0): {out_ln[0].mean().item():+.6f}  <- forced near 0")
print(f"RMSNorm  output mean (row 0): {out_my_rms[0].mean().item():+.6f}  <- not forced to 0")
assert torch.allclose(out_my_rms, out_real_rms), "MyRMSNorm doesn't match model.py's RMSNorm"
print("RMSNorm: illustrative implementation matches model.py exactly.")
print(f"LayerNorm params: weight {tuple(layernorm.weight.shape)} + bias {tuple(layernorm.bias.shape)}")
print(f"RMSNorm  params: weight {tuple(real_norm.weight.shape)} only (no bias)")

# %% [markdown]
# ---
# ## Part 3 — SwiGLU: a smarter feedforward block
#
# ### What the feedforward block does
#
# Every transformer block has two sublayers: attention (tokens exchange
# information with each other) and a **feedforward network / MLP** (each
# token's vector gets independently transformed — no cross-token mixing here
# at all). The classic GPT-2-style MLP is: expand, apply a nonlinearity,
# contract back down:
#
# ```
# output = W_down( GELU( W_up(x) ) )        # W_up expands by 4x, W_down contracts back
# ```
#
# The nonlinearity (originally ReLU, later GELU) is what gives the network any
# expressive power beyond a plain linear transform — without it, stacking
# linear layers would collapse into a single linear layer.
#
# ### SwiGLU: gate the value instead of just squashing it
#
# **SwiGLU** (Shazeer, 2020) replaces the single up-projection with *two*
# parallel ones of the same size:
#
# - `w1(x)` — the **gate**, passed through the SiLU activation.
# - `w3(x)` — the **value**, left alone.
#
# They're multiplied together element-wise before the down-projection:
#
# ```
# output = W_down( SiLU(W1(x)) * W3(x) )
# ```
#
# Think of the gate as a per-dimension volume knob computed *from the input
# itself*: for each of the hidden dimensions, the gate decides how much of the
# corresponding value dimension to let through, based on what the input
# actually looks like. A fixed activation like GELU applies the exact same
# squashing curve to every dimension regardless of context; a gate lets the
# network learn *content-dependent* filtering instead — which turns out to
# matter more than the extra parameters it costs.
#
# **SiLU** (`x * sigmoid(x)`, also called Swish) is the activation used for the
# gate. Compared to ReLU, it's smooth everywhere (no sharp corner at 0) and
# allows small negative outputs, both of which tend to help gradient flow.

# %%
xs = torch.linspace(-5, 5, 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xs, F.relu(xs), label="ReLU")
ax.plot(xs, F.gelu(xs), label="GELU")
ax.plot(xs, F.silu(xs), label="SiLU (used by SwiGLU)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_title("Activation functions: ReLU vs GELU vs SiLU")
ax.legend()
fig.tight_layout()
fig.savefig("assets/04_activation_functions.png", dpi=120)
plt.close(fig)
print("Saved assets/04_activation_functions.png")
print("Notice SiLU (and GELU) are smooth and dip slightly negative; ReLU is a hard corner at 0.")

# %% [markdown]
# ### Sizing the hidden layer
#
# A plain MLP uses hidden size `4 * n_embd` with 2 weight matrices. SwiGLU has
# 3 weight matrices (`w1`, `w3`, `w2`), so to keep the parameter count roughly
# comparable, its hidden size is shrunk to about `(8/3) * n_embd`, then rounded
# up to a multiple of 64 (hardware likes multiples of 64):
#
# `hidden = 64 * (((int(8/3 * n_embd)) + 63) // 64)`

# %%
class MySwiGLU(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        hidden = 64 * (((int(8 / 3 * n_embd)) + 63) // 64)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)   # gate
        self.w3 = nn.Linear(n_embd, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, n_embd, bias=False)   # down-projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# %% [markdown]
# ### Check: shape, and an exact match against `model.py`
#
# The MLP is a residual sublayer (its output gets *added* back to its input),
# so the output shape must equal the input shape. To confirm our version
# really is the same computation as `model.py`'s `SwiGLU` — not just
# shape-compatible — we build both with the same random seed right before
# construction. Since both create their three `nn.Linear` layers in the same
# order with the same shapes, seeding identically makes every weight come out
# bit-for-bit the same, so the two modules should produce identical output.

# %%
n_embd = 128
config = GPTConfig(n_embd=n_embd, dropout=0.0)

torch.manual_seed(SEED)
my_swiglu = MySwiGLU(n_embd, dropout=0.0).eval()
torch.manual_seed(SEED)
real_swiglu = SwiGLU(config).eval()

B, T = 2, 8
x = torch.randn(B, T, n_embd)
out_my = my_swiglu(x)
out_real = real_swiglu(x)
assert out_my.shape == x.shape, f"SwiGLU shape mismatch: {out_my.shape} != {x.shape}"
assert torch.allclose(out_my, out_real, atol=1e-6), "MySwiGLU doesn't match model.py's SwiGLU"
print(f"SwiGLU output shape: {tuple(out_my.shape)} (matches input)")
print("SwiGLU: illustrative implementation matches model.py exactly.")

gelu_mlp_params = 2 * (n_embd * 4 * n_embd) + 4 * n_embd + n_embd   # 2 linears + biases
swiglu_params = sum(p.numel() for p in real_swiglu.parameters())
print(f"Classic 4x GELU MLP params: ~{gelu_mlp_params:,}")
print(f"SwiGLU params:               {swiglu_params:,}  (hidden={real_swiglu.w1.out_features}, "
      f"3 matrices instead of 2, but each one is smaller)")

# %% [markdown]
# ---
# ## Summary
#
# | Component | Naive predecessor | What changed | Why it's worth it |
# |---|---|---|---|
# | **RoPE** | Learned absolute position embeddings | Rotate Q/K by a position-dependent angle instead of adding a lookup vector | Encodes *relative* distance directly; no capped table; no extra parameters |
# | **RMSNorm** | LayerNorm | Drop mean-centering and the bias term; keep only the rescale-by-magnitude step | Fewer ops per layer, fewer parameters, same practical stability |
# | **SwiGLU** | ReLU/GELU MLP | Two up-projections (gate x value) instead of one fixed nonlinearity | Content-dependent filtering, empirically stronger at similar parameter count |
#
# You built a small, from-scratch version of each of these and checked it
# against the exact function/class `model.py` uses. Notebook 05 imports
# `build_rope`, `apply_rope`, `RMSNorm`, and `SwiGLU` straight from `model.py`
# and assembles them — along with attention from notebook 03 — into the full
# decoder-only transformer.

# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. In Part 0 we showed that "pikachu"'s attention output was identical
#    whether it appeared first or second in the sequence. In one sentence,
#    *why* is that true of plain self-attention, independent of any specific
#    example?
# 2. Suppose you trained a model with learned absolute position embeddings on
#    sequences up to length 512, then tried to run it on a sequence of length
#    600. What breaks, and does the same problem happen with RoPE? Why or why
#    not?
# 3. True or false, and explain: "RMSNorm still forces every output vector to
#    have zero mean, just like LayerNorm, since it's still a normalization."
# 4. If you replaced SiLU in SwiGLU's gate branch with a function that always
#    outputs exactly `1`, no matter the input, what would `SwiGLU`'s forward
#    pass reduce to, and what capability would the block lose?
# 5. Multiple choice: which of these best describes why RoPE's dot-product
#    check in this notebook used *two different* starting positions (10 and
#    30) with the *same* offset (5), rather than just checking one pair of
#    positions?
#    (a) To make the plot look nicer.
#    (b) Because a single matching pair could be a coincidence; matching at
#        two unrelated starting points is evidence the result holds for *any*
#        starting point, i.e. that only the offset matters.
#    (c) Because RoPE only works correctly at position 10 and position 30.

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Self-attention computes each token's output as a weighted average over
#    the *values* of all tokens present, where the weights come from
#    content-based dot products (query . key). Nothing in that computation
#    reads off "which index am I" or "which index are they" — only "what are
#    the actual vectors." Reordering the input just reorders which row holds
#    which token's output; it doesn't change any individual token's result,
#    because the set of tokens being averaged over hasn't changed.
# 2. With learned absolute position embeddings, the embedding table simply has
#    no row for position 512-599 — there's nothing to look up, so the model
#    can't process those positions at all (in practice, you'd get an index
#    error, or you'd have to leave those positions with random/untrained
#    vectors). RoPE doesn't have this problem in the same way: `theta_i` and
#    the rotation formula are defined for any position, computed on the fly,
#    not stored in a table with a hard maximum row. It can technically compute
#    angles for position 600 today. (Quality at much longer lengths than
#    trained on can still degrade, but there's no hard architectural wall.)
# 3. False. Mean-centering (`x - mu`) is exactly the step RMSNorm removes.
#    RMSNorm only divides by the root-mean-square magnitude and applies a
#    learned scale — it controls the *magnitude* of the output but says
#    nothing about its mean. We showed this directly: LayerNorm's output mean
#    came out near 0, while RMSNorm's did not.
# 4. `SwiGLU`'s forward pass is `w2(SiLU(w1(x)) * w3(x))`. If `SiLU(w1(x))`
#    were replaced with a constant `1` everywhere, that becomes
#    `w2(1 * w3(x)) = w2(w3(x))` — the composition of two *linear* layers with
#    no nonlinearity in between, which is mathematically equivalent to a
#    single linear layer. The block would lose all of its nonlinear,
#    content-dependent filtering — exactly the ability that made SwiGLU worth
#    using over a single linear projection in the first place.
# 5. (b). One matching pair of dot products could in principle be a
#    coincidence for those specific numbers. Getting the same value from two
#    *different* starting positions with the same offset is evidence that the
#    result generalizes — that RoPE's dot product genuinely depends only on
#    relative offset, not on where you start counting from.
# </details>
