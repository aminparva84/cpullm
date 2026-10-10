"""Section 4. Controlled-English surface realizer.

Each proposition becomes one clause, terminated with a period. Clause order is
a topological sort of the supplied graph. One edge is removed from each cycle
until that sort covers every proposition. "Consequently" is inserted only when
the immediately previous clause is the source of a causal edge.

The clause is not passed through a grammar rewriter. A rewriter can change
polarity, and the auditor would then be checking a different sentence.
"""

from __future__ import annotations

import re

from cpu_llm.ir.schema import (
    EdgeType,
    EIRProposition,
    LogicalExpressionNode,
    LogicalOperator,
    Modality,
    Polarity,
    QuantifierType,
    TypedGraphEdge,
)
from cpu_llm.realizer.inflect import forms

_ANCHOR = re.compile(r"<!--anchor:([\w-]+)-->\s*([^<]+)")


def _render_condition(node: LogicalExpressionNode) -> str:
    children = node.operands or []
    if node.operator is LogicalOperator.NOT and children:
        inner = " AND ".join(_render_condition(child) for child in children)
        return f"not ({inner})"
    if children and node.operator in {LogicalOperator.AND, LogicalOperator.OR}:
        parts = [_render_condition(child) for child in children]
        return f" {node.operator.value} ".join(parts)
    words = {
        LogicalOperator.GT: "is greater than",
        LogicalOperator.LT: "is less than",
        LogicalOperator.GTE: "is at least",
        LogicalOperator.LTE: "is at most",
        LogicalOperator.EQ: "equals",
    }
    relation = words.get(node.operator, "satisfies")
    unit = f" {node.unit}" if node.unit else ""
    return f"{node.metric_name} {relation} {node.threshold_value}{unit}"


def _ordered_ids(propositions: list[EIRProposition], edges: list[TypedGraphEdge]) -> list[str]:
    present: dict[str, None] = {}
    for item in propositions:
        present.setdefault(item.proposition_id, None)
    incoming = {item: 0 for item in present}
    children: dict[str, list[str]] = {item: [] for item in present}
    for edge in edges:
        if edge.source_id not in present or edge.target_id not in present:
            continue
        if edge.source_id == edge.target_id:
            continue
        children[edge.source_id].append(edge.target_id)
        incoming[edge.target_id] += 1
    while True:
        ordered = _kahn(incoming, children)
        if len(ordered) == len(present):
            return ordered
        if not _remove_cycle_edge(children, incoming):
            raise ValueError("proposition graph still has a cycle after removing a back edge")


def _kahn(incoming: dict[str, int], children: dict[str, list[str]]) -> list[str]:
    pending = dict(incoming)
    ready = [node for node, count in pending.items() if count == 0]
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for child in children.get(node, []):
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    return ordered


def _remove_cycle_edge(children: dict[str, list[str]], incoming: dict[str, int]) -> bool:
    cycle = _find_cycle(children)
    if not cycle:
        return False
    source, target = cycle[-1], cycle[0]
    if target not in children[source]:
        return False
    children[source].remove(target)
    incoming[target] -= 1
    return True


def _find_cycle(children: dict[str, list[str]]) -> list[str] | None:
    color = {node: 0 for node in children}
    stack: list[str] = []

    def walk(node: str) -> list[str] | None:
        color[node] = 1
        stack.append(node)
        for nxt in children[node]:
            if color[nxt] == 1:
                return stack[stack.index(nxt) :]
            if color[nxt] == 0:
                found = walk(nxt)
                if found:
                    return found
        stack.pop()
        color[node] = 2
        return None

    for node in children:
        if color[node] == 0:
            found = walk(node)
            if found:
                return found
    return None


def _clause(proposition: EIRProposition, seen_subjects: dict[str, int]) -> str:
    quantifier = proposition.quantifier.qtype
    prefix = {
        QuantifierType.NONE: "No ",
        QuantifierType.SOME: "Some ",
        QuantifierType.ALL: "All ",
    }.get(quantifier, "")
    lemma = proposition.subject.lemma.casefold()
    seen = seen_subjects.get(lemma, 0)
    seen_subjects[lemma] = seen + 1
    if seen > 0 and quantifier is QuantifierType.UNSPECIFIED:
        subject = "they" if proposition.subject.is_plural else "it"
    else:
        qualifiers = " ".join(proposition.subject.qualifiers)
        qualifier_text = f"{qualifiers} " if qualifiers else ""
        subject = f"{prefix}{qualifier_text}{proposition.subject.raw_text}"
    plural = proposition.subject.is_plural or subject.casefold().startswith(("all ", "some "))
    negative = proposition.polarity is Polarity.NEGATIVE and quantifier is not QuantifierType.NONE
    verb = _verb_phrase(proposition, plural=plural, negative=negative)
    obj = f" {proposition.object_target.raw_text}" if proposition.object_target else ""
    clause = " ".join(f"{subject} {verb}{obj}".split())
    if proposition.condition_tree is not None:
        clause += f" provided that {_render_condition(proposition.condition_tree)}"
    return clause


def _verb_phrase(proposition: EIRProposition, *, plural: bool, negative: bool) -> str:
    """Active clauses use present agreement. Only passive voice reads tense."""
    lemma = proposition.predicate.lemma
    table = forms(lemma)
    if proposition.predicate.voice == "PASSIVE":
        participle = table["part"]
        if proposition.modality is Modality.OBLIGATIVE:
            return f"must not be {participle}" if negative else f"must be {participle}"
        if proposition.modality is Modality.PERMISSIVE:
            return f"may not be {participle}" if negative else f"may be {participle}"
        if proposition.predicate.tense == "PAST":
            be = "were" if plural else "was"
        else:
            be = "are" if plural else "is"
        return f"{be} not {participle}" if negative else f"{be} {participle}"
    if proposition.modality is Modality.OBLIGATIVE:
        return f"must not {lemma}" if negative else f"must {lemma}"
    if proposition.modality is Modality.PERMISSIVE:
        return f"may not {lemma}" if negative else f"may {lemma}"
    if negative:
        auxiliary = "do not" if plural else "does not"
        return f"{auxiliary} {lemma}"
    return table["present_pl"] if plural else table["present_3s"]


class SemanticSurfaceRealizer:
    def synthesize(
        self,
        propositions: list[EIRProposition],
        graph_edges: list[TypedGraphEdge],
    ) -> tuple[str, dict[str, str]]:
        if not propositions:
            return "No verified epistemic evidence available.", {}
        by_id = {item.proposition_id: item for item in propositions}
        ordered = _ordered_ids(propositions, graph_edges)
        causal = {
            (edge.source_id, edge.target_id)
            for edge in graph_edges
            if edge.edge_type is EdgeType.CAUSAL_RELATION
        }
        seen: dict[str, int] = {}
        clauses: list[str] = []
        for index, proposition_id in enumerate(ordered):
            clause = _clause(by_id[proposition_id], seen)
            if index > 0 and (ordered[index - 1], proposition_id) in causal:
                clause = f"Consequently, {clause[0].lower() + clause[1:]}"
            clauses.append(f"<!--anchor:{proposition_id}--> {clause}.")
        passage = " ".join(clauses)
        delivered: dict[str, str] = {}
        for match in _ANCHOR.finditer(passage):
            delivered[match.group(1)] = match.group(2).strip()
        visible = re.sub(r"<!--anchor:[^>]+-->\s*", "", passage).strip()
        return visible, delivered
