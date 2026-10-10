"""Section 4. Round-trip check of one delivered sentence against its proposition.

The check is lexical: polarity on the clause head, modality, passive voice,
every condition threshold, the subject lemma, and an inflection of the
predicate lemma. A "not" inside the condition is not proposition polarity.
The check does not load a parser, and a string match is not open-domain entailment.
"""

from __future__ import annotations

import re

from cpu_llm.ir.schema import EIRProposition, LogicalExpressionNode, Modality, Polarity, QuantifierType
from cpu_llm.realizer.inflect import surface_tokens

_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?")
_BE = {"be", "is", "are", "was", "were", "been", "being"}


def _tokens(sentence: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN.finditer(sentence)]


def _threshold_text(node: LogicalExpressionNode) -> list[str]:
    found: list[str] = []
    if node.operands:
        for child in node.operands:
            found.extend(_threshold_text(child))
    elif node.threshold_value is not None:
        found.append(str(node.threshold_value))
    return found


class EpistemicEntailmentAuditor:
    def audit_delivered_output(
        self,
        delivered_map: dict[str, str],
        propositions: list[EIRProposition],
    ) -> bool:
        by_id = {item.proposition_id: item for item in propositions}
        if len(delivered_map) != len(propositions):
            return False
        for proposition_id, sentence in delivered_map.items():
            proposition = by_id.get(proposition_id)
            if proposition is None or not _sentence_matches(sentence, proposition):
                return False
        return True


def _sentence_matches(sentence: str, proposition: EIRProposition) -> bool:
    lowered = sentence.casefold()
    tokens = _tokens(sentence)
    token_set = set(tokens)
    head = lowered.partition(" provided that ")[0]
    head_tokens = set(_tokens(head))
    quantifier = proposition.quantifier.qtype
    if quantifier is QuantifierType.NONE:
        if not lowered.startswith("no "):
            return False
    else:
        expected_negative = proposition.polarity is Polarity.NEGATIVE
        has_negation = "not" in head_tokens or "never" in head_tokens
        if has_negation != expected_negative:
            return False
    if proposition.modality is Modality.OBLIGATIVE and token_set.isdisjoint({"must", "shall", "require"}):
        return False
    if proposition.modality is Modality.PERMISSIVE and token_set.isdisjoint({"may", "can"}):
        return False
    if proposition.predicate.voice == "PASSIVE" and token_set.isdisjoint(_BE):
        return False
    if proposition.condition_tree is not None:
        for threshold in _threshold_text(proposition.condition_tree):
            if threshold not in lowered:
                return False
    subject = proposition.subject.lemma.casefold()
    if subject not in lowered and "it" not in token_set and "they" not in token_set:
        return False
    if token_set.isdisjoint(surface_tokens(proposition.predicate.lemma)):
        return False
    return True
