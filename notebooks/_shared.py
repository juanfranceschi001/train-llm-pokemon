"""Cold-start-safe helpers shared by the notebooks.

Why this file exists
---------------------
On Google Colab, opening a notebook can hand you a brand-new virtual machine
with an empty disk -- even if you already ran an earlier notebook in the same
browser tab five minutes ago. In the original course this repo is based on,
several notebooks just did `open("data/shakespeare.txt")` and crashed with
FileNotFoundError if that file wasn't already sitting on disk, and a couple of
notebooks assumed a checkpoint from an earlier notebook was already saved.

The fix isn't "remember to run notebooks in order" (that's exactly the trap
that caused the bug) -- it's making every notebook able to rebuild anything it
needs, on its own, on a completely empty machine. Each `ensure_*` function
below does that: check for the artifact, and if it's missing, regenerate it
using the *exact same procedure* the "owning" notebook uses, so the result is
identical either way.

Every notebook should start with:
    import sys, os
    while not os.path.exists("requirements.txt"):
        os.chdir("..")
    sys.path.insert(0, os.path.join(os.getcwd(), "notebooks"))
    from _shared import ensure_corpus, ...
"""
import json
import os
import re
import time
from collections import Counter

import torch
import torch.nn as nn

CORPUS_PATH = "data/pokemon_corpus.txt"
BOW_METRICS_PATH = "assets/phase1_metrics.json"
CHECKPOINT_PATH = "checkpoints/model.pt"


def ensure_corpus():
    """Guarantee data/pokemon_corpus.txt exists, fetching it from PokeAPI if not.
    Cheap either way: an existing file returns instantly, a missing one takes
    ~5-15 seconds to rebuild. Used by every notebook that reads the corpus."""
    from data.pokemon_fetch import build_corpus
    return build_corpus()


# ---------------------------------------------------------------------------
# Notebook 01 <-> 02: the Bag-of-Words baseline metrics
# ---------------------------------------------------------------------------

def _tokenize_words(s: str) -> list[str]:
    return re.findall(r"[a-z]+|[^a-z\s]", s.lower())


class _WordTokenizer:
    """A stripped-down copy of notebook 01's WordTokenizer, used only to
    regenerate the baseline if assets/phase1_metrics.json is missing. See
    notebook 01 for the annotated, from-scratch version you're meant to read."""

    def __init__(self, corpus: str, max_vocab: int = 2000):
        counts = Counter(_tokenize_words(corpus))
        most_common = [w for w, _ in counts.most_common(max_vocab - 1)]
        self.itos = ["<unk>"] + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, s: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(w, unk) for w in _tokenize_words(s)]


def compute_bow_baseline(text: str, context: int = 8, vocab_size: int = 2000):
    """Trains the tiny Bag-of-Words baseline from notebook 01 and returns its
    metrics dict. This is the one real implementation -- notebook 01 calls it
    to produce the file the first time, and ensure_bow_baseline() below calls
    it again only if that file ever goes missing."""
    tok = _WordTokenizer(text, max_vocab=vocab_size)
    ids = [tok.stoi.get(w, 0) for w in _tokenize_words(text)]
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]

    def make_dataset(ids, limit):
        xs, ys = [], []
        step = max(1, (len(ids) - context - 1) // limit)
        for i in range(0, len(ids) - context - 1, step):
            vec = torch.zeros(tok.vocab_size)
            for w in ids[i:i + context]:
                vec[w] += 1.0
            xs.append(vec)
            ys.append(ids[i + context])
        return torch.stack(xs), torch.tensor(ys)

    Xtr, Ytr = make_dataset(train_ids, limit=20000)
    Xva, Yva = make_dataset(val_ids, limit=4000)

    bow = nn.Linear(tok.vocab_size, tok.vocab_size)
    opt = torch.optim.AdamW(bow.parameters(), lr=1e-2)
    for _ in range(60):
        loss = nn.functional.cross_entropy(bow(Xtr), Ytr)
        opt.zero_grad(); loss.backward(); opt.step()

    bow.eval()
    with torch.no_grad():
        val_loss = nn.functional.cross_entropy(bow(Xva), Yva).item()
    val_ppl = float(torch.exp(torch.tensor(val_loss)))

    metrics = {
        "bow_val_perplexity": val_ppl,
        "bow_params": sum(p.numel() for p in bow.parameters()),
        "context": context,
        "vocab_size": tok.vocab_size,
    }
    os.makedirs("assets", exist_ok=True)
    with open(BOW_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def ensure_bow_baseline():
    """Load assets/phase1_metrics.json, or recompute it (~a few seconds) if a
    fresh runtime doesn't have it -- e.g. you opened notebook 02 without
    notebook 01 having run first in this session."""
    if os.path.exists(BOW_METRICS_PATH):
        with open(BOW_METRICS_PATH) as f:
            return json.load(f)
    print("assets/phase1_metrics.json missing -- recomputing the notebook 01 "
          "baseline now (takes a few seconds)...")
    text = open(CORPUS_PATH, encoding="utf-8").read()
    return compute_bow_baseline(text)


# ---------------------------------------------------------------------------
# Notebook 06 <-> 07/08: the trained checkpoint
# ---------------------------------------------------------------------------

def train_baseline_model(target_seconds=360, verbose=True):
    """Trains the main char-level transformer from notebook 06 and saves it
    to checkpoints/model.pt. This is the one real implementation -- notebook
    06 calls it for the "real" run, and ensure_baseline_checkpoint() below
    calls it again only as a fallback if that checkpoint ever goes missing."""
    from model import GPT, DEFAULT_CONFIG, get_device

    device = get_device()
    text = open(CORPUS_PATH, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = DEFAULT_CONFIG
    config.vocab_size = len(chars)

    def get_batch(split, batch_size=32):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - config.block_size, (batch_size,))
        x = torch.stack([d[i:i + config.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + config.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

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

    torch.manual_seed(1337)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Dry run to measure throughput, then pick max_iters to hit target_seconds.
    dry_iters = 50
    t0 = time.time()
    for _ in range(dry_iters):
        xb, yb = get_batch("train")
        _, loss, _ = model(xb, targets=yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    its_per_sec = dry_iters / (time.time() - t0)
    max_iters = max(500, (int(its_per_sec * target_seconds) // 500) * 500)

    if verbose:
        print(f"Training {config.n_layer}L x {config.n_embd}d GPT for {max_iters} "
              f"iters (~{max_iters/its_per_sec/60:.1f} min on this hardware)")

    eval_interval, eval_iters = 500, 50
    history = {"train": [], "val": [], "iters": []}
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model, eval_iters)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            history["iters"].append(it)
            if verbose:
                print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "stoi": stoi,
                    "itos": itos,
                }, CHECKPOINT_PATH)
        if it == max_iters:
            break
        xb, yb = get_batch("train")
        _, loss, _ = model(xb, targets=yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    return history, best_val_loss


def ensure_baseline_checkpoint():
    """Load checkpoints/model.pt, or retrain a quick version (~2 min on CPU)
    if a fresh runtime doesn't have it -- e.g. you opened notebook 07 without
    notebook 06 having run first in this session."""
    ensure_corpus()
    if not os.path.exists(CHECKPOINT_PATH):
        print("checkpoints/model.pt missing -- retraining a quick baseline now "
              "(this mirrors notebook 06, but capped shorter; run notebook 06 "
              "for the full training run)...")
        train_baseline_model(target_seconds=120)
    from model import GPT
    return torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
