"""Reason-string mapping layer (BUILD_PLAN Stage 1) — the clinical-text
normalization core. Maps a raw injury notes string -> structured label.

A single classify() call returns:
    category    : one of the modeled categories, or None
    time_loss   : True for injuries; False for rest/load-management
    ambiguous   : True if nothing injury-like matched (set aside honestly)
    severe_tail : True for the spotlighted Achilles tail

Design notes:
  - Patterns live in reason_map.yaml so the dictionary is iterable without code
    changes (the build plan expects this to grow as new strings appear).
  - Priority order is the YAML order: load-management checked first (to strip
    non-injuries), then severe tail, then categories top-to-bottom; first match
    wins. This is why achilles sits above generic lower-limb.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_MAP_PATH = Path(__file__).resolve().parent / "reason_map.yaml"


@dataclass(frozen=True)
class Label:
    category: str | None
    time_loss: bool
    ambiguous: bool
    severe_tail: bool
    raw: str


@lru_cache(maxsize=1)
def _load_map() -> dict:
    with _MAP_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _compile(patterns: list[str]) -> list[re.Pattern]:
    # Treat each pattern as a regex (most are plain substrings, which are valid
    # regexes). Lowercase, search anywhere in the string.
    return [re.compile(p, re.IGNORECASE) for p in patterns]


@lru_cache(maxsize=1)
def _compiled():
    cfg = _load_map()
    lm = _compile(cfg["load_management"]["patterns"])
    cats = []
    for c in cfg["categories"]:
        cats.append((c["name"], c.get("severe_tail", False), _compile(c["patterns"])))
    return lm, cats


def classify(notes: str) -> Label:
    """Classify one raw reason string into a structured Label."""
    raw = (notes or "").strip()
    text = raw.lower()
    lm, cats = _compiled()

    if not text:
        return Label(None, False, True, False, raw)

    # 1) Strip load-management / non-injury rows first.
    for pat in lm:
        if pat.search(text):
            return Label(None, time_loss=False, ambiguous=False,
                         severe_tail=False, raw=raw)

    # 2) Injury categories in priority order; first match wins.
    for name, severe, pats in cats:
        for pat in pats:
            if pat.search(text):
                return Label(name, time_loss=True, ambiguous=False,
                             severe_tail=severe, raw=raw)

    # 3) Nothing matched -> ambiguous, set aside honestly.
    return Label(None, time_loss=False, ambiguous=True,
                 severe_tail=False, raw=raw)


def classify_many(notes_iter) -> list[Label]:
    return [classify(n) for n in notes_iter]
