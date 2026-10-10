"""Small deterministic inflector for the controlled realizer vocabulary."""

from __future__ import annotations

_IRREGULAR = {
    "be": {
        "base": "be",
        "present_3s": "is",
        "present_pl": "are",
        "past": "was",
        "past_pl": "were",
        "part": "been",
        "prog": "being",
    },
    "have": {
        "base": "have",
        "present_3s": "has",
        "present_pl": "have",
        "past": "had",
        "past_pl": "had",
        "part": "had",
        "prog": "having",
    },
    "do": {
        "base": "do",
        "present_3s": "does",
        "present_pl": "do",
        "past": "did",
        "past_pl": "did",
        "part": "done",
        "prog": "doing",
    },
    "pass": {
        "base": "pass",
        "present_3s": "passes",
        "present_pl": "pass",
        "past": "passed",
        "past_pl": "passed",
        "part": "passed",
        "prog": "passing",
    },
    "undergo": {
        "base": "undergo",
        "present_3s": "undergoes",
        "present_pl": "undergo",
        "past": "underwent",
        "past_pl": "underwent",
        "part": "undergone",
        "prog": "undergoing",
    },
}


def _regular_present_3s(lemma: str) -> str:
    if lemma.endswith(("s", "x", "z", "ch", "sh")):
        return lemma + "es"
    if len(lemma) > 1 and lemma.endswith("y") and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    return lemma + "s"


def _regular_past(lemma: str) -> str:
    if lemma.endswith("e"):
        return lemma + "d"
    if len(lemma) > 1 and lemma.endswith("y") and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ied"
    return lemma + "ed"


def _regular_progressive(lemma: str) -> str:
    if lemma.endswith("ie"):
        return lemma[:-2] + "ying"
    if lemma.endswith("e") and not lemma.endswith("ee"):
        return lemma[:-1] + "ing"
    return lemma + "ing"


def forms(lemma: str) -> dict[str, str]:
    key = lemma.casefold()
    if key in _IRREGULAR:
        return dict(_IRREGULAR[key])
    past = _regular_past(key)
    return {
        "base": key,
        "present_3s": _regular_present_3s(key),
        "present_pl": key,
        "past": past,
        "past_pl": past,
        "part": past,
        "prog": _regular_progressive(key),
    }


def surface_tokens(lemma: str) -> set[str]:
    return {value.casefold() for value in forms(lemma).values()}
