# House style for these notebooks (internal — not part of the published course)

This file is the shared spec every notebook in this repo follows. It exists so
each notebook reads consistently even though they were drafted independently.

## Format

- Every notebook is a **jupytext "percent format" `.py` file** in `notebooks/`.
  Markdown cells: `# %% [markdown]` followed by lines starting with `#`. Code
  cells: `# %%` followed by plain Python. This is plain, valid, executable
  Python top to bottom — running `python notebooks/0X_name.py` must work with
  no errors, exactly like running the notebook cell-by-cell.
- After writing, generate the paired notebook with:
  `jupytext --to notebook notebooks/0X_name.py`
- **You must actually run `python notebooks/0X_name.py` and get a clean exit
  (no traceback) before you're done.** If it fails, fix the notebook and
  re-run — don't hand back code you haven't executed. If the notebook trains
  anything, let it actually finish; don't assume it works.

## Opening cell (copy verbatim into every notebook)

```python
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
```

## Cold-start safety (the whole reason this repo exists)

The original course this is based on would crash on a fresh Google Colab
runtime because some notebooks assumed a file an earlier notebook produced
was already sitting on disk. **Every notebook you write must be runnable on
its own, from an empty `data/`, `checkpoints/`, `assets/` state, with no
manual "run notebook X first" step.**

- Need the training corpus? Call `from _shared import ensure_corpus;
  ensure_corpus()` (returns the path to `data/pokemon_corpus.txt`, fetching it
  from PokeAPI in ~5-15s if it isn't already cached). Never assume the file is
  already there, and never re-implement the download yourself.
- Notebook 02 needs notebook 01's baseline numbers: call
  `from _shared import ensure_bow_baseline; metrics = ensure_bow_baseline()`.
- Notebooks 07/08 need notebook 06's trained checkpoint: call
  `from _shared import ensure_baseline_checkpoint; ckpt =
  ensure_baseline_checkpoint()`.
- Read `notebooks/_shared.py` before you start — it already implements all of
  the above. Don't duplicate its logic; import it.
- If you introduce a NEW cross-notebook artifact dependency, add an `ensure_*`
  function to `_shared.py` following the same pattern (check for the file,
  regenerate it via the same procedure the owning notebook uses if it's
  missing) instead of assuming the file exists.

## Content: what to change from the original, what to keep

You're rewriting a specific notebook from https://github.com/orbek/train-llm
(Apache-2.0). Fetch it yourself for structural reference:
```
gh api repos/orbek/train-llm/contents/notebooks/<ORIGINAL_FILENAME>.py --jq '.content' | base64 -d
```
Keep: the overall teaching arc and "beat the previous baseline" narrative,
the use of `assert`-based sanity checks tied to real math, inline plots.

Change:
- **Dataset**: Shakespeare -> Pokémon. The corpus is Pokédex/move/ability
  flavor text, one entry per line, format `"<Name>: <description>"` or
  `"The move <Name>: <description>"` / `"The ability <Name>: <description>"`.
  It's plain English prose, so anywhere the original says "Shakespeare" or
  gives a Shakespeare-specific example, swap in a Pokémon-flavored one (e.g.
  nearest-neighbor embedding examples like "charizard" landing near "fire" and
  "dragon" are more fun and legible to a young learner than Shakespeare word
  geometry). When you show sample generated text in markdown, keep it
  plausible for a small/undertrained model (garbled but recognizably
  Pokémon-flavored), not perfect prose.
- **Tone**: rewrite explanations from scratch in your own words for a true
  beginner — assume less background than the original. Define every term the
  first time it's used (e.g. don't say "logits" without one sentence on what
  that means). Prefer short paragraphs and concrete numeric examples over
  abstract description.
- **80/20 framing**: at the top of the notebook, add 2-3 sentences on *why
  this concept matters* and *how much of the final result it's responsible
  for* before diving into implementation. Notebooks 00-08 are the **core
  path** — say so explicitly. Notebooks 09 and 10 are **bonus / advanced /
  optional** — their opening markdown cell must say this plainly ("You can
  stop at notebook 08 and have a complete, working, from-scratch LLM. This
  notebook is optional extra depth.") so a learner doesn't feel obligated.

## End-of-notebook comprehension quiz (new — the original doesn't have this)

Every notebook ends with two new cells, in this order:

```python
# %% [markdown]
# ## ✅ Check your understanding
#
# Answer these before moving on — no peeking at the answers below first.
#
# 1. <a conceptual question about the notebook's core idea>
# 2. <a "predict what happens if..." question>
# 3. <a question that would catch a common misconception>
# (3-5 questions total; mix multiple-choice and short-answer/predict-the-output)

# %% [markdown]
# <details>
# <summary>Answers (click to expand)</summary>
#
# 1. ...
# 2. ...
# 3. ...
# </details>
```

Use a real `<details>`/`<summary>` HTML block (renders as a collapsible
spoiler in Jupyter/GitHub) so the learner can self-test honestly before
revealing answers. Questions should test understanding of *why*, not just
recall of a term — e.g. "if you doubled `block_size`, what would happen to
the model's memory usage and why?" rather than "what does `block_size` mean?"

## Assets

Save any plots to `assets/0X_description.png` (same naming convention as the
original) — these get committed to the repo (see `.gitignore`: `data/` and
`checkpoints/` are ignored, `assets/` is not).
