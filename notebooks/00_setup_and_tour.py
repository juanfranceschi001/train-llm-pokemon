# %% [markdown]
# # 00 — Setup & Tour
#
# Welcome! You're about to build a real language model — the same *kind* of
# architecture behind tools like ChatGPT — completely from scratch in Python
# and PyTorch. Instead of Shakespeare, you'll train it on Pokémon: Pokédex
# entries, move descriptions, and ability text, pulled live from a free public
# API. By the end, your own tiny model will generate text that reads like an
# invented (and slightly weird) Pokédex entry.
#
# No machine-learning background is assumed. If you can read basic Python,
# you're ready — every new idea (math, jargon, all of it) gets explained the
# first time it shows up.
#
# ## The 80/20 rule for this course
#
# Eleven notebooks is a lot. Not all of them matter equally. This course is
# split into two tiers:
#
# - **Notebooks 00-08 are the core path.** They take you from "I can write a
#   for loop" to "I trained a working transformer language model and
#   understand every piece of it." This is where ~80% of the payoff lives, and
#   it's a complete, working thing on its own — you can stop at 08 and you've
#   genuinely built an LLM from scratch.
# - **Notebooks 09-10 are bonus / advanced.** They cover a smarter tokenizer
#   (BPE) and Mixture-of-Experts — real techniques used in production models,
#   but not required to understand how an LLM works at its core. Do them if
#   you're curious or want the extra depth; skip them with no guilt if you'd
#   rather stop after 08.
#
# Within the core path, one notebook matters more than the rest: **03,
# Attention**. It's the single idea that separates a language model from
# every simpler model that came before it, and it gets the most explanation
# time of any notebook in this course.

# %% [markdown]
# ## A note on working directories (and why every notebook repeats this cell)
#
# Jupyter runs a notebook from the folder it lives in (`notebooks/`), not the
# project root. That means paths like `data/` would silently resolve to
# `notebooks/data/` — the wrong place. The cell below walks up the directory
# tree until it finds the project root (identified by `requirements.txt`) and
# changes into it, so every path in the notebook means what you think it
# means, no matter how you launched Jupyter or Colab.
#
# You'll see this exact cell at the top of every notebook in this course. That
# repetition is deliberate: it's what makes each notebook runnable entirely on
# its own, which matters a lot on Google Colab (see the section below).

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():          # reached filesystem root; stop
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "notebooks"))
print("Working directory:", os.getcwd())

# %%
import platform
import torch

print("Python :", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch :", torch.__version__)

# %% [markdown]
# ## Running this course on Google Colab (read this if that's you)
#
# Colab is a great free way to run these notebooks, especially if your laptop
# doesn't have a GPU. But there's one Colab-specific gotcha worth knowing
# about up front, because it's the exact bug this repo was rewritten to fix.
#
# **Each Colab notebook you open can land on a brand-new, empty virtual
# machine — even if you already ran a different notebook from this repo five
# minutes ago.** If notebook 06 saves a trained model to `checkpoints/`, and
# then you open notebook 07 in a *different* Colab session, that file won't be
# there. A course that assumes it will be there just crashes.
#
# This course handles that for you two ways:
#
# 1. **Every notebook can rebuild what it needs on its own.** If a file a
#    notebook depends on is missing, it regenerates it automatically (the
#    corpus re-downloads in seconds; a missing trained checkpoint retrains a
#    quick version instead of crashing). You'll see print statements when this
#    happens — it's expected, not an error.
# 2. **If you want real persistence across Colab sessions** (so you don't
#    retrain things you already trained), mount Google Drive and point this
#    repo's `data/`, `checkpoints/`, and `assets/` folders there:
#
# ```python
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# os.makedirs('/content/drive/MyDrive/train-llm-pokemon', exist_ok=True)
# # then symlink or just cd into that Drive folder before running the notebooks
# ```
#
# Either approach works. Self-healing means you're never *stuck*; mounting
# Drive means you're never *waiting on a re-download or re-train* you didn't
# need to do.

# %% [markdown]
# ## Picking a device
#
# Neural networks do a huge number of arithmetic operations — millions of
# multiplications per second. Modern hardware has chips that do this
# arithmetic in parallel, much faster than an ordinary CPU.
#
# In PyTorch, **device** means the piece of hardware a computation runs on:
#
# - **CUDA** — NVIDIA GPUs (Windows/Linux, and Colab's free GPU runtime).
#   NVIDIA's parallel-computing platform; dramatically faster than CPU.
# - **MPS** — the GPU built into Apple Silicon Macs (M1/M2/M3/M4). Apple's
#   name for the interface that lets PyTorch use that GPU.
# - **CPU** — your computer's main processor. Always available, always
#   slower for this kind of work, but perfectly fine for every notebook in
#   this course — nothing here needs a GPU to finish in a reasonable time.
#
# `pick_device()` checks for the best option automatically, in that order, so
# you never have to edit a device line yourself. Every later notebook reuses
# this exact function.

# %%
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print("Using device:", device)

# %% [markdown]
# ## Sanity check — a tiny matrix multiplication
#
# Before trusting a device with real training, run a quick smoke test.
#
# A **tensor** is PyTorch's core data structure: a multi-dimensional array of
# numbers. `torch.randn(3, 3)` makes a 3x3 tensor of random numbers. `@` is
# **matrix multiplication** — the single most-repeated operation in a neural
# network (every layer is essentially a big matmul). If it works here, every
# later notebook's matmuls will work too.

# %%
x = torch.randn(3, 3, device=device)
y = x @ x
assert y.shape == (3, 3)
print("Matmul on", device, "ok. Sum =", float(y.sum()))

# %% [markdown]
# ## Meet the data: a live preview of the Pokémon corpus
#
# Every notebook from here on trains on the same text: Pokédex species
# descriptions plus move and ability text, fetched from
# [PokeAPI](https://pokeapi.co) (free, no login, no scraping — it's a public
# read-only API built for exactly this kind of use). The cell below fetches
# and caches it now, so later notebooks start instantly.

# %%
from _shared import ensure_corpus

corpus_path = ensure_corpus()
with open(corpus_path, encoding="utf-8") as f:
    text = f.read()
print(f"Corpus ready: {len(text):,} characters at {corpus_path}")
print("----- sample -----")
print(text[:400])

# %% [markdown]
# ## The journey ahead
#
# | # | Notebook | Tier | Idea |
# |---|----------|------|------|
# | 00 | Setup & tour | core | you are here |
# | 01 | Data & Bag-of-Words | core | a runnable baseline that ignores word order |
# | 02 | Embeddings | core | dense vectors beat BoW with far fewer parameters |
# | 03 | Attention | core (biggest idea) | finally, order-aware |
# | 04 | Modern components | core | RoPE, RMSNorm, SwiGLU |
# | 05 | Assembling the model | core | the full decoder-only transformer |
# | 06 | Training loop | core | actually train it |
# | 07 | Evaluation & generation | core | sample Pokédex text; KV-cache |
# | 08 | Tuning | core | learning-rate schedules, sweeps |
# | 09 | BPE tokenizer | bonus | subword tokenization from scratch |
# | 10 | Mixture-of-Experts | bonus | the capstone |
#
# Each step beats a *measured* number from the step before — you won't just be
# told a newer technique is better, you'll watch the metric improve. On to
# notebook 01!

# %%
print("Environment looks good. Continue to 01_data_and_bag_of_words.")

# %% [markdown]
# ## ✅ Check your understanding
#
# 1. Why does every notebook in this course repeat the same "walk up to the
#    project root" cell at the top, instead of just assuming you're already in
#    the right folder?
# 2. You open notebook 07 on a fresh Colab runtime that has never seen this
#    repo before, without running any earlier notebook first. What happens,
#    and why doesn't it just crash?
# 3. Suppose your laptop has no NVIDIA or Apple Silicon GPU. Will any notebook
#    in this course fail to run? Why or why not?
# 4. According to the 80/20 framing above, which single notebook is claimed to
#    matter most, and what's the argument for why?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. Because Jupyter (and Colab) can launch a notebook's kernel with its
#    working directory set to wherever the notebook file lives — usually
#    `notebooks/` — not the project root. Without that cell, `data/...` would
#    quietly resolve to `notebooks/data/...`, a folder that doesn't exist,
#    and every relative path in the notebook would break in a confusing way
#    that has nothing to do with the lesson you're trying to learn.
# 2. It won't crash: `ensure_baseline_checkpoint()` (used in notebook 07)
#    notices `checkpoints/model.pt` is missing and retrains a quick baseline
#    model on the spot (a couple of minutes on CPU) instead of raising
#    `FileNotFoundError`. You'll see a printed message explaining that it's
#    doing this.
# 3. No — every notebook is designed to finish in a reasonable time on CPU
#    alone (that's explicitly one of the constraints of this course). A GPU
#    makes training faster, but `pick_device()` falls back to CPU
#    automatically and nothing requires a GPU to complete.
# 4. Notebook 03 (Attention). The argument: attention is the one idea that
#    turns an order-blind model (like the Bag-of-Words baseline in notebook
#    01) into one that actually understands sequence and context — it's the
#    core mechanism that makes "transformer" a transformer, so the course
#    invests the most explanation time there.
# </details>
