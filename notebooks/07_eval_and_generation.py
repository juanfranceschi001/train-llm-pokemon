# %% [markdown]
# # 07 — Evaluation & Generation: Watching Your Model Talk
#
# This is the payoff notebook. Notebooks 01-06 (all core path — see notebook
# 00's 80/20 table) spent their time on *infrastructure*: tokenizers,
# embeddings, attention, the modern transformer components, and finally a
# training loop that produced a saved model. None of that mattered yet in a
# way you could actually see. This notebook is where you finally point the
# trained model at a blank page and watch it write something.
#
# Three things happen here, and together they're worth as much attention as
# any single earlier notebook because this is where "did all that work pay
# off?" gets answered concretely:
#
# 1. **Perplexity** — one number that says how well the model predicts text
#    it has never seen, so "it works" becomes a measured claim.
# 2. **Sampling strategies** — greedy, temperature, top-k, top-p — the knobs
#    that control *how* the model turns its predictions into actual text, and
#    the real (somewhat weird, since this is a small model) Pokédex-style
#    text each one produces.
# 3. **KV-cache** — the one optimization that makes generating long text fast
#    instead of painfully slow, explained and benchmarked with a real
#    measured speedup.
#
# Notebook 08 (tuning) is still core path and comes after this one, but this
# is the notebook where the model's output stops being an abstraction.

# %% [markdown]
# ## Working directory setup (same cell every notebook starts with)
#
# Jupyter runs a notebook from the folder it lives in (`notebooks/`), not the
# project root, so a path like `checkpoints/model.pt` would otherwise resolve
# to the wrong place. This walks up to the folder containing `requirements.txt`
# and treats that as home base — see notebook 00 for the full explanation.

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

from model import GPT, get_device
from _shared import ensure_baseline_checkpoint

torch.manual_seed(42)
device = get_device()
print("Using device:", device)

# %% [markdown]
# ---
# ## Part 1 — Loading the trained model (safely, even on a fresh machine)
#
# Notebook 06 trains the model and saves the checkpoint with the best
# validation loss it saw during training to `checkpoints/model.pt`. The naive
# way to load it here would be:
#
# ```python
# checkpoint = torch.load("checkpoints/model.pt", ...)   # crashes if missing!
# ```
#
# That's exactly the bug this whole repo was rewritten to avoid. On a brand
# new machine — a fresh Google Colab runtime, or simply opening this notebook
# without having run notebook 06 first in this session — `checkpoints/model.pt`
# just isn't there yet, and that line would crash with `FileNotFoundError`
# before you'd seen a single generated sentence.
#
# Instead we call `ensure_baseline_checkpoint()` from `_shared.py`. It:
# - Loads `checkpoints/model.pt` instantly if it's already there (the normal
#   case if you just ran notebook 06), or
# - If it's missing, **prints an explanation and retrains a quick baseline
#   model on the spot** (about 2 minutes on a CPU — the same architecture as
#   notebook 06, just capped to a shorter training run) instead of crashing.
#
# So don't be alarmed if this next cell takes a couple of minutes and prints
# training progress instead of loading instantly — that's the self-healing
# behavior working as designed, not an error. Either way you end up with the
# same kind of checkpoint dict, with four keys:
#
# | Key | What's inside |
# |---|---|
# | `model_state` | the trained weights (a PyTorch `state_dict`) |
# | `config` | a `GPTConfig` — every architectural hyperparameter (layer count, embedding size, etc.) needed to rebuild the exact same model shape |
# | `stoi` | "string to int" — maps each character in the corpus to its id |
# | `itos` | "int to string" — the reverse mapping, used to turn generated ids back into text |
#
# We rebuild an untrained `GPT` from `config` (so its architecture matches
# exactly), then load the saved weights into it with `load_state_dict`.

# %%
checkpoint = ensure_baseline_checkpoint()
config = checkpoint["config"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model = GPT(config).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

print(f"Config     : {config}")
print(f"Params     : {model.num_params()/1e6:.2f}M")
print(f"Vocab size : {config.vocab_size}  (this is a character-level model — "
      f"each of the {config.vocab_size} distinct characters in the corpus is one token)")
print(f"Block size : {config.block_size}  (max tokens of context the model can look at)")

# %% [markdown]
# ---
# ## Part 2 — Perplexity: putting a number on "how good is it?"
#
# **Perplexity (PPL)** answers a concrete question: *on average, how many
# characters is the model seriously considering as "what comes next" at each
# position?* It's computed as `exp(average cross-entropy loss)` — the same
# loss notebook 06 minimized during training, just rescaled so it's easier to
# reason about than a raw loss number like "1.4".
#
# Two reference points, since our vocabulary here is characters, not words:
# - A model that has learned *nothing* and spreads probability equally across
#   all `config.vocab_size` characters has perplexity equal to `vocab_size`
#   (with 85 characters, that's PPL = 85 — totally lost).
# - A perfect model that's always 100% certain of the correct next character
#   has perplexity 1 (never happens in practice, but it's the floor).
#
# Our trained model should land well below `vocab_size` but well above 1 —
# it has learned real structure (common letter sequences, where a Pokémon's
# name ends and its description begins, and so on) without memorizing the
# validation text it never trained on.
#
# We rebuild the validation split exactly the way notebook 06 built it: same
# corpus, same 90/10 split by position (not shuffled — see notebook 01's
# reasoning for why a sequential split is the honest one), using the
# checkpoint's own `stoi` so token ids match what the model was trained on.

# %%
from _shared import ensure_corpus

corpus_path = ensure_corpus()
with open(corpus_path, encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
val_data = data[n:]
print(f"Val characters: {len(val_data):,}")

@torch.no_grad()
def estimate_val_loss(model, val_data, block_size, batch_size=32, eval_iters=100):
    """Average cross-entropy loss over random windows of the validation set."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x = torch.stack([val_data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix]).to(device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    return sum(losses) / len(losses)

val_loss = estimate_val_loss(model, val_data, config.block_size)
val_ppl = torch.exp(torch.tensor(val_loss)).item()

print(f"\nVal loss      : {val_loss:.4f}")
print(f"Val perplexity: {val_ppl:.2f}")
print(f"(out of {config.vocab_size} possible characters, the model is effectively "
      f"choosing among ~{val_ppl:.1f} plausible next characters at each position)")

assert 1.0 <= val_ppl <= config.vocab_size, \
    "perplexity should always land between 1 (perfect) and vocab_size (random guessing)"

# %% [markdown]
# ---
# ## Part 3 — Sampling strategies: turning predictions into text
#
# At every step the model outputs **logits**: one raw score per character in
# the vocabulary, saying how much it "likes" each candidate as the next
# character. Turning those scores into an actual chosen character is a
# separate decision, and there's more than one reasonable way to make it.
# Each strategy below is a different setting on `model.generate()` (see
# `sample()` in `model.py`) — think of them as different dials on the same
# machine, each trading *quality* (does it look like real Pokédex text?)
# against *diversity* (does it avoid saying the same safe thing every time?).
#
# - **Greedy (`temperature=0`)** — always pick the single most likely next
#   character. Fully deterministic: the same prompt always produces the same
#   output. Tends to get stuck in loops ("the the the...") because it never
#   risks a lower-scoring but more interesting option.
# - **Temperature scaling** — divide every logit by `T` before turning them
#   into probabilities. `T < 1` sharpens the distribution (the model leans
#   even harder into its favorite characters — more repetitive, more
#   coherent); `T > 1` flattens it (more randomness, more chances of
#   nonsense); `T = 1` uses the model's raw, un-adjusted probabilities.
# - **Top-k** — before sampling, throw away every character except the `k`
#   most likely ones, then sample from just those. This puts a hard ceiling
#   on how weird a choice can be, no matter how flat the distribution gets.
# - **Top-p (nucleus)** — instead of a fixed count, keep the *smallest* set of
#   top characters whose probabilities add up to at least `p`. When the model
#   is very confident, that set might be just one or two characters; when
#   it's unsure, the set grows automatically. This adapts per-step in a way
#   a fixed top-k can't.
#
# We'll generate from the same prompt with each strategy so the comparison is
# fair. Since our model is small (a few million parameters, trained briefly)
# and character-level, don't expect flawless Pokédex prose — expect
# recognizably Pokémon-flavored but somewhat garbled text: real words mixed
# with invented ones, occasional made-up creature names, and sentences that
# trail off strangely. That's an honest reflection of model size and training
# time, not a bug — raising `max_iters` back in notebook 06 (training longer)
# is the single biggest lever for sharpening these samples.

# %%
PROMPT = "Charizard:"
MAX_NEW = 200

prompt_ids = encode(PROMPT)
print(f"Prompt    : {PROMPT!r} ({len(prompt_ids)} tokens)")
print(f"Max total : {len(prompt_ids) + MAX_NEW} tokens (block_size = {config.block_size})")
assert len(prompt_ids) + MAX_NEW <= config.block_size, \
    "prompt + new tokens must fit inside block_size for cached generation"

def make_seed():
    return torch.tensor([prompt_ids], dtype=torch.long, device=device)

def generate_and_print(label, temperature, top_k=None, top_p=None):
    torch.manual_seed(42)
    out = model.generate(make_seed(), MAX_NEW, temperature=temperature,
                         top_k=top_k, top_p=top_p, use_cache=True)
    generated = decode(out[0].tolist())
    print(f"\n{'='*60}\n[{label}]\n{'='*60}")
    print(generated)

# %%
generate_and_print("Greedy (temperature=0)", temperature=0.0)
generate_and_print("Temperature=0.8", temperature=0.8)
generate_and_print("Top-k=40 (temperature=1.0)", temperature=1.0, top_k=40)
generate_and_print("Top-p=0.9, nucleus (temperature=1.0)", temperature=1.0, top_p=0.9)

# %% [markdown]
# A few things worth noticing in your own output above:
#
# - **Greedy** is the most repetitive-looking — it's mechanically choosing the
#   "safest" character every time, which for a small model often means
#   falling into a loop.
# - **Temperature=0.8** usually looks a little more varied than greedy while
#   staying reasonably close to real words.
# - **Top-k=40** and **top-p=0.9** tend to produce the most natural-looking
#   variety, since both cut off the very unlikely tail of characters while
#   still letting the model take some risks.
#
# None of these will read like a real Pokédex entry end to end — this model
# trained for a couple of minutes on a laptop CPU, not hours on a datacenter
# GPU. What matters here is the *mechanism*: the same four knobs (temperature,
# top-k, top-p, and a deterministic greedy baseline) are exactly what
# production LLM APIs expose to you as `temperature`, `top_k`, and `top_p`
# parameters.

# %% [markdown]
# ---
# ## Part 4 — KV-cache: same output, much less recomputation
#
# ### What problem does it solve?
#
# `model.generate()` produces text one character at a time: predict the next
# character, append it, repeat. The *naive* way to predict character `t+1` is
# to feed the model the entire sequence so far (characters `1..t`) and read
# off its prediction for position `t`. But inside attention, the keys and
# values for characters `1..t-1` are recomputed from scratch every single
# step, even though they never change — token 3's key vector at step 50 is
# identical to token 3's key vector at step 10. That's a lot of repeated work
# for no new information.
#
# ### The fix
#
# A **KV-cache** stores each layer's key and value tensors after computing
# them once, and reuses them on every later step. Look at
# `CausalSelfAttention.forward` in `model.py`: when a cache is passed in, the
# new token's key/value get concatenated onto the cached ones (`torch.cat`)
# instead of recomputing the whole sequence, and only the *new* token's query
# needs to run through attention. `GPT.generate()` (also in `model.py`) does
# exactly this when `use_cache=True`: one full "prefill" pass over the prompt
# to build the initial cache, then one cheap single-token step per new
# character afterward.
#
# The payoff: without a cache, generating token `t` costs work proportional
# to the whole sequence length so far, and that cost grows as generation
# continues — generating `N` tokens costs roughly `O(N²)` total work. With a
# cache, each new token costs a small, roughly constant amount of extra work,
# so generating `N` tokens costs roughly `O(N)` total — genuinely less work,
# not just less code.
#
# ### Correctness first
#
# A cache is purely a speed optimization — it must never change *what* the
# model generates. We check that directly: greedy-decode the same prompt with
# and without the cache and assert the outputs are byte-for-byte identical.

# %%
seed = make_seed()
NEW_TOKENS = 200  # prompt + NEW_TOKENS must stay <= block_size, checked above

out_cached = model.generate(seed.clone(), NEW_TOKENS, temperature=0.0, use_cache=True)
out_plain = model.generate(seed.clone(), NEW_TOKENS, temperature=0.0, use_cache=False)

assert torch.equal(out_cached, out_plain), \
    "KV-cache changed the output -- it should be a pure speed optimization"
print("Correctness check passed: cached and non-cached greedy generation are identical.")

# %% [markdown]
# ### Now the speed comparison
#
# We time several repetitions of the same generation with the cache on and
# off and report the real measured speedup on this machine. (The exact number
# depends on your hardware, model size, and sequence length — a bigger model
# or longer generation will generally show a bigger gain, since there's more
# redundant recomputation being avoided.)

# %%
REPS = 5

t0 = time.perf_counter()
for _ in range(REPS):
    model.generate(seed.clone(), NEW_TOKENS, temperature=0.0, use_cache=True)
t_cached = (time.perf_counter() - t0) / REPS

t0 = time.perf_counter()
for _ in range(REPS):
    model.generate(seed.clone(), NEW_TOKENS, temperature=0.0, use_cache=False)
t_plain = (time.perf_counter() - t0) / REPS

speedup = t_plain / t_cached
print(f"With KV-cache   : {t_cached*1000:.1f} ms/generation")
print(f"Without KV-cache: {t_plain*1000:.1f} ms/generation")
print(f"Measured speedup: {speedup:.2f}x")

# %% [markdown]
# That speedup is the whole reason production chat models feel responsive:
# every token you see streamed back from a hosted LLM is generated one at a
# time, exactly like our loop above, and none of them would be usable in real
# time without a KV-cache doing this same trick at a much larger scale.

# %% [markdown]
# ---
# ## Where this leaves us
#
# Zoom out for a second: notebook 01 built a Bag-of-Words baseline and
# measured its perplexity. Notebook 02 beat it with embeddings. Notebooks
# 03-05 built up a real transformer piece by piece. Notebook 06 trained it.
# This notebook is where all of that turns into something you can actually
# read — real (if imperfect) generated Pokédex text, a measured perplexity
# number, and an understanding of exactly how a production LLM turns
# predictions into a fast, streaming response.
#
# That's the complete core path: notebooks 00-08 take you from nothing to a
# working, understood, from-scratch language model. Notebook 08 (tuning)
# covers learning-rate schedules and sweeps to squeeze out more quality;
# notebooks 09-10 are optional bonus depth (a smarter BPE tokenizer, and
# Mixture-of-Experts) if you want to go further.

# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. You open this notebook on a brand new machine that has never run any
#    part of this repo before, and you run every cell top to bottom without
#    running notebook 06 first. What happens, and why doesn't it crash?
# 2. If you set `temperature=2.0` with no `top_k` or `top_p`, would you expect
#    the output to look *more* or *less* like coherent Pokédex text than
#    `temperature=1.0`? Why?
# 3. Why must a KV-cache produce byte-for-byte identical output to
#    non-cached generation, rather than just "close enough"? What would it
#    mean if the assert in Part 4 failed?
# 4. Roughly how does the *total* cost of generating `N` tokens scale with
#    `N`, with and without a KV-cache? What's actually being saved?
# 5. A model that always assigns equal probability to all `config.vocab_size`
#    characters would have what perplexity? Why does that number make sense
#    as the "worst case"?

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. It won't crash: `ensure_baseline_checkpoint()` checks for
#    `checkpoints/model.pt`, notices it's missing, prints an explanation, and
#    retrains a quick baseline model on the spot (a couple of minutes on CPU)
#    instead of raising `FileNotFoundError`. Every later cell then runs
#    against that freshly trained checkpoint exactly as if notebook 06 had
#    been run first — the only difference is a short wait and some extra
#    printed output.
# 2. Less coherent. `temperature` divides the logits before converting them
#    to probabilities; a higher temperature flattens the distribution, so
#    characters the model considers unlikely become much more likely to get
#    picked. At `T=2.0` the model is taking far more risks per character than
#    at `T=1.0`, which usually shows up as more garbled, less word-like
#    output — more "creative" in the sense of unpredictable, not in the sense
#    of better.
# 3. Because the KV-cache is purely a computational shortcut for something
#    the model would compute anyway — it must not change the *math*, only
#    how much redundant work is done to get there. If the assert failed, it
#    would mean the cached code path has a bug (e.g. concatenating the wrong
#    keys/values, or getting positions wrong), and every notebook or product
#    relying on cached generation for speed would silently be generating
#    different text than "true" greedy decoding, without any error being
#    raised.
# 4. Without a cache, generating `N` tokens costs roughly `O(N²)` total work,
#    because producing token `t` re-processes all `t-1` previous tokens
#    through every layer, and that per-step cost grows as generation
#    continues. With a cache, each new token only computes attention for
#    *itself* against the already-cached keys/values of everything before
#    it, so each step costs a small, roughly constant amount, and generating
#    `N` tokens costs roughly `O(N)` total. What's saved is the repeated
#    recomputation of key/value vectors for tokens that were already
#    processed in an earlier step and haven't changed.
# 5. Perplexity would equal `config.vocab_size` (85 for this corpus) — every
#    character is equally likely, so the model is effectively choosing
#    uniformly at random among all of them, with zero useful signal. That's
#    the ceiling / "no better than random guessing" case; a trained model's
#    perplexity should sit well below it, which is exactly the assert we
#    checked in Part 2.
# </details>
