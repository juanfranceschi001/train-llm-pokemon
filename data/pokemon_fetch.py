"""Builds the training corpus for this course from PokeAPI (https://pokeapi.co).

PokeAPI is free, requires no API key, and its data (species descriptions, move
and ability text) is released for reuse -- no scraping or copyright concerns,
unlike lyrics or show scripts.

This is written to be *cold-start safe*: run it from any notebook, on any
machine (including a fresh Google Colab VM that has never seen this repo
before), and it will (re)build data/pokemon_corpus.txt from scratch. If the
file already exists and looks complete, it does nothing -- so re-running it
is always cheap.

Run directly to build the corpus:
    python data/pokemon_fetch.py
"""

import concurrent.futures
import os
import re
import time

import requests

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "pokemon_corpus.txt")
API_ROOT = "https://pokeapi.co/api/v2"

# Roughly Gen 1-3 (Kanto/Johto/Hoenn) -- the Pokemon most people already know,
# and enough requests to build a corpus comparable in size to TinyShakespeare
# (~1MB) without hammering the API. Raise these if you want more variety.
NUM_SPECIES = 400
NUM_MOVES = 350
NUM_ABILITIES = 150

MIN_CORPUS_CHARS = 200_000  # sanity floor: below this, treat the file as incomplete
_WORKERS = 12
_SESSION = requests.Session()


def _get_json(url, retries=3):
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def _clean_flavor_text(raw):
    # PokeAPI flavor text is riddled with literal newlines and form-feed
    # characters left over from the original Game Boy text boxes.
    text = raw.replace("\n", " ").replace("\x0c", " ").replace("\r", " ")
    text = text.replace("­", "")  # soft hyphen left over from GB/GBA line wrapping
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _english_flavor_entries(data, key="flavor_text_entries", text_field="flavor_text"):
    seen = set()
    lines = []
    for entry in data.get(key, []):
        if entry["language"]["name"] != "en":
            continue
        cleaned = _clean_flavor_text(entry[text_field])
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            lines.append(cleaned)
    return lines


def _fetch_species_lines(species_id):
    data = _get_json(f"{API_ROOT}/pokemon-species/{species_id}")
    if not data:
        return []
    name = data["name"].capitalize()
    lines = _english_flavor_entries(data)
    return [f"{name}: {line}" for line in lines]


def _fetch_move_lines(move_id):
    data = _get_json(f"{API_ROOT}/move/{move_id}")
    if not data:
        return []
    name = data["name"].replace("-", " ").capitalize()
    lines = _english_flavor_entries(data)
    return [f"The move {name}: {line}" for line in lines]


def _fetch_ability_lines(ability_id):
    data = _get_json(f"{API_ROOT}/ability/{ability_id}")
    if not data:
        return []
    name = data["name"].replace("-", " ").capitalize()
    lines = _english_flavor_entries(data)
    return [f"The ability {name}: {line}" for line in lines]


def _fetch_many(fetch_one, ids, label):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(fetch_one, i): i for i in ids}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
            done += 1
            if done % 100 == 0 or done == len(ids):
                print(f"  {label}: {done}/{len(ids)}")
    return results


def build_corpus(force=False):
    """Fetch the Pokemon text corpus and cache it to disk. Idempotent."""
    if not force and os.path.exists(CORPUS_PATH):
        size = os.path.getsize(CORPUS_PATH)
        if size >= MIN_CORPUS_CHARS:
            print(f"Using cached corpus at {CORPUS_PATH} ({size:,} bytes)")
            return CORPUS_PATH
        print(f"Cached corpus looks incomplete ({size:,} bytes) -- rebuilding.")

    print("Building the Pokemon corpus from PokeAPI (one-time; cached after this)...")
    all_lines = []
    all_lines += _fetch_many(_fetch_species_lines, range(1, NUM_SPECIES + 1), "species")
    all_lines += _fetch_many(_fetch_move_lines, range(1, NUM_MOVES + 1), "moves")
    all_lines += _fetch_many(_fetch_ability_lines, range(1, NUM_ABILITIES + 1), "abilities")

    text = "\n".join(all_lines) + "\n"
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote {len(text):,} characters ({len(all_lines):,} lines) to {CORPUS_PATH}")
    return CORPUS_PATH


if __name__ == "__main__":
    build_corpus()
