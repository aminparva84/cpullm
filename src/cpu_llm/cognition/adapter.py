"""Phase 1. Deliberation over EIR propositions.

Contradiction uses the modality matrix and temporal overlap. Ruin is 1 when
that pair is a contradiction, and 0 otherwise. There is no random bilinear
map and no SVD.

A rule condition is true, false, or unknown against metrics stated about the
instance's subject. Unknown keeps the condition. False emits no derivation.

When both evidence tensors are inside the Poincaré ball, the instance must
lie in the rule's entailment cone. A tensor that is not a point in the ball
is not treated as a failed cone test.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from cpu_llm.cpu_guard import assert_cpu_only
from cpu_llm.geometry.poincare import is_in_entailment_cone
from cpu_llm.ir.schema import (
    ConditionEval,
    DerivationTrace,
    EIRProposition,
    EdgeType,
    EpistemicStatus,
    LogicalExpressionNode,
    Modality,
    Polarity,
    Quantifier,
    QuantifierType,
    TypedEntity,
    TypedGraphEdge,
    TypedPredicate,
    evaluate_condition,
)

_COPULA = {"be", "qualify", "classify", "act"}
_TENSION_DECAY = 0.5
_HARD_CONTRADICTION = {
    frozenset({Modality.DECLARATIVE}),
    frozenset({Modality.OBLIGATIVE}),
    frozenset({Modality.OBLIGATIVE, Modality.PERMISSIVE}),
}


class DeliberationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    target_vector: list[float]
    grounded_propositions: list[EIRProposition]
    graph_edges: list[TypedGraphEdge]
    abstention_triggered: bool
    ruin_distance: float
    audit_trace_id: str


def poincare_slice(tensor: np.ndarray) -> Optional[np.ndarray]:
    """Return a 128-d in-ball coordinate, or None when the tensor is not one.

    A 1024-d evidence tensor contributes the slice [400:528], matching the
    hyperbolic block in the architecture diagram. Any other width is used
    only when it is already 128-d.
    """
    point = np.asarray(tensor, dtype=np.float64).reshape(-1)
    if point.size >= 528:
        point = point[400:528]
    elif point.size != 128:
        return None
    norm = float(np.linalg.norm(point))
    if norm < 1e-6 or norm >= 1.0:
        return None
    return point


def modal_relation(left: EIRProposition, right: EIRProposition) -> str:
    """CONTRADICTION, TENSION, or COMPATIBLE for one pair of opposite claims."""
    if left.polarity is right.polarity:
        return "COMPATIBLE"
    pair = frozenset({left.modality, right.modality})
    if pair in _HARD_CONTRADICTION:
        return "CONTRADICTION"
    if pair == frozenset({Modality.PROBABILISTIC, Modality.DECLARATIVE}):
        return "TENSION"
    return "COMPATIBLE"


def _same_atom(left: EIRProposition, right: EIRProposition) -> bool:
    if left.subject.lemma.casefold() != right.subject.lemma.casefold():
        return False
    if left.predicate.lemma.casefold() != right.predicate.lemma.casefold():
        return False
    left_object = None if left.object_target is None else left.object_target.lemma.casefold()
    right_object = None if right.object_target is None else right.object_target.lemma.casefold()
    return left_object == right_object


def _time_overlaps(left: EIRProposition, right: EIRProposition) -> bool:
    if left.temporal_interval is not None and right.temporal_interval is not None:
        return left.temporal_interval.overlaps(right.temporal_interval)
    return True


def metrics_for_subject(
    subject_lemma: str,
    evidence: list[EIRProposition],
) -> tuple[dict[str, float], dict[str, Optional[str]], set[str]]:
    """Metrics stated about this subject. Disagreeing values or units are conflicts."""
    found: dict[str, float] = {}
    units: dict[str, Optional[str]] = {}
    conflicts: set[str] = set()
    key = subject_lemma.casefold()
    for item in evidence:
        if item.metric_name is None or item.metric_value is None:
            continue
        if item.subject.lemma.casefold() != key:
            continue
        name = item.metric_name
        if name in found and (found[name] != item.metric_value or units.get(name) != item.metric_unit):
            conflicts.add(name)
            continue
        if name not in found:
            found[name] = item.metric_value
            units[name] = item.metric_unit
    return found, units, conflicts


def _trace_id(tenant_id: str, count: int) -> str:
    digest = hashlib.sha256(f"{tenant_id}:{count}".encode("utf-8")).hexdigest()[:16]
    return f"TRACE_{digest}"


def _abstention(trace_id: str, query_vector: np.ndarray) -> DeliberationResult:
    proposition = EIRProposition(
        proposition_id=f"abstain_{hashlib.sha256(b'ABSTAIN_VACUUM').hexdigest()[:12]}",
        polarity=Polarity.NEGATIVE,
        modality=Modality.DECLARATIVE,
        subject=TypedEntity(raw_text="system", lemma="system", is_plural=False),
        predicate=TypedPredicate(lemma="possess", voice="ACTIVE"),
        object_target=TypedEntity(raw_text="sufficient verified evidence", lemma="evidence", is_plural=False),
        epistemic_status=EpistemicStatus.ABSTENTION,
        epistemic_weight=0.0,
    )
    return DeliberationResult(
        target_vector=np.asarray(query_vector, dtype=np.float64).reshape(-1).tolist(),
        grounded_propositions=[proposition],
        graph_edges=[],
        abstention_triggered=True,
        ruin_distance=0.0,
        audit_trace_id=trace_id,
    )


def _conflict(trace_id: str, query_vector: np.ndarray, edges: list[TypedGraphEdge]) -> DeliberationResult:
    proposition = EIRProposition(
        proposition_id=f"conflict_{hashlib.sha256(trace_id.encode('utf-8')).hexdigest()[:12]}",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.DECLARATIVE,
        subject=TypedEntity(raw_text="evidence", lemma="evidence", is_plural=False),
        predicate=TypedPredicate(lemma="exhibit", voice="ACTIVE"),
        object_target=TypedEntity(raw_text="unresolved mutual contradiction", lemma="contradiction", is_plural=False),
        epistemic_status=EpistemicStatus.UNRESOLVED_CONTRADICTION,
        epistemic_weight=1.0,
        derivation=DerivationTrace(
            inference_rule="MODAL_TEMPORAL_CONTRADICTION",
            singular_values=[1.0],
            ruin_distance=1.0,
        ),
    )
    return DeliberationResult(
        target_vector=np.asarray(query_vector, dtype=np.float64).reshape(-1).tolist(),
        grounded_propositions=[proposition],
        graph_edges=edges,
        abstention_triggered=False,
        ruin_distance=1.0,
        audit_trace_id=trace_id,
    )


def _cone_allows(rule_tensor: np.ndarray, instance_tensor: np.ndarray) -> bool:
    parent = poincare_slice(rule_tensor)
    child = poincare_slice(instance_tensor)
    if parent is None or child is None:
        return True
    return is_in_entailment_cone(parent, child)


class PrismCognitionSubspaceAdapter:
    def __init__(self, ruin_threshold: float = 0.85) -> None:
        assert_cpu_only()
        if not 0.0 <= ruin_threshold <= 1.0:
            raise ValueError("ruin_threshold must lie in [0, 1]")
        self.ruin_threshold = ruin_threshold

    def deliberate(
        self,
        query_vector: np.ndarray,
        evidence_propositions: list[EIRProposition],
        evidence_tensors: list[np.ndarray],
        tenant_id: str,
    ) -> DeliberationResult:
        trace_id = _trace_id(tenant_id, len(evidence_propositions))
        query = np.asarray(query_vector, dtype=np.float64).reshape(-1)
        if not evidence_propositions:
            return _abstention(trace_id, query)
        if len(evidence_tensors) != len(evidence_propositions):
            raise ValueError("each evidence proposition needs one tensor")

        edges: list[TypedGraphEdge] = []
        tension_ids: set[str] = set()
        contradicted = False
        count = len(evidence_propositions)
        for left_index in range(count):
            for right_index in range(left_index + 1, count):
                left = evidence_propositions[left_index]
                right = evidence_propositions[right_index]
                if not _same_atom(left, right) or not _time_overlaps(left, right):
                    continue
                relation = modal_relation(left, right)
                if relation == "CONTRADICTION":
                    contradicted = True
                    edges.append(
                        TypedGraphEdge(
                            source_id=left.proposition_id,
                            target_id=right.proposition_id,
                            edge_type=EdgeType.CONTRADICTION,
                        )
                    )
                elif relation == "TENSION":
                    tension_ids.add(left.proposition_id)
                    tension_ids.add(right.proposition_id)

        ruin = 1.0 if contradicted else 0.0
        if ruin > self.ruin_threshold:
            return _conflict(trace_id, query, edges)

        def weighted(proposition: EIRProposition) -> float:
            value = proposition.epistemic_weight
            if proposition.proposition_id in tension_ids:
                value = max(0.0, min(1.0, value * _TENSION_DECAY))
            return value

        derived: list[EIRProposition] = [
            item.model_copy(update={"epistemic_weight": weighted(item)}) for item in evidence_propositions
        ]
        for rule_index, rule in enumerate(evidence_propositions):
            if rule.quantifier.qtype is not QuantifierType.ALL or rule.polarity is not Polarity.AFFIRMATIVE:
                continue
            for instance_index, instance in enumerate(evidence_propositions):
                if rule_index == instance_index:
                    continue
                if not _instantiates(rule, instance):
                    continue
                if not _cone_allows(evidence_tensors[rule_index], evidence_tensors[instance_index]):
                    continue
                status, kept = _condition_status(rule.condition_tree, instance, evidence_propositions)
                if status is None:
                    continue
                derived_id = "derived_" + hashlib.sha256(
                    f"{rule.proposition_id}:{instance.proposition_id}".encode()
                ).hexdigest()[:12]
                derived.append(
                    EIRProposition(
                        proposition_id=derived_id,
                        polarity=rule.polarity,
                        modality=rule.modality,
                        quantifier=Quantifier(),
                        subject=instance.subject,
                        predicate=rule.predicate,
                        object_target=rule.object_target,
                        condition_tree=kept,
                        epistemic_status=status,
                        epistemic_weight=min(weighted(rule), weighted(instance)),
                        derivation=DerivationTrace(
                            inference_rule="SOUND_UNIVERSAL_MODUS_PONENS",
                            premise_ids=[rule.proposition_id, instance.proposition_id],
                            singular_values=[ruin],
                            ruin_distance=ruin,
                        ),
                    )
                )
                edges.append(
                    TypedGraphEdge(
                        source_id=instance.proposition_id,
                        target_id=derived_id,
                        edge_type=EdgeType.INFERENCE_DEPENDENCY,
                    )
                )
                edges.append(
                    TypedGraphEdge(
                        source_id=rule.proposition_id,
                        target_id=derived_id,
                        edge_type=EdgeType.INFERENCE_DEPENDENCY,
                    )
                )

        context = np.mean(np.stack([np.asarray(item, dtype=np.float64).reshape(-1) for item in evidence_tensors]), axis=0)
        if context.shape != query.shape:
            target = query
        else:
            target = query * 0.6 + context * 0.4
        target = target / (float(np.linalg.norm(target)) + 1e-8)
        return DeliberationResult(
            target_vector=target.tolist(),
            grounded_propositions=derived,
            graph_edges=edges,
            abstention_triggered=False,
            ruin_distance=ruin,
            audit_trace_id=trace_id,
        )


def _instantiates(rule: EIRProposition, instance: EIRProposition) -> bool:
    if instance.polarity is not Polarity.AFFIRMATIVE or instance.modality is not Modality.DECLARATIVE:
        return False
    if instance.predicate.lemma.casefold() not in _COPULA or instance.object_target is None:
        return False
    if instance.object_target.lemma.casefold() != rule.subject.lemma.casefold():
        return False
    required = {item.casefold() for item in rule.subject.qualifiers}
    present = {item.casefold() for item in instance.object_target.qualifiers}
    return required.issubset(present)


def _condition_status(
    tree: Optional[LogicalExpressionNode],
    instance: EIRProposition,
    evidence: list[EIRProposition],
) -> tuple[Optional[EpistemicStatus], Optional[LogicalExpressionNode]]:
    if tree is None:
        return EpistemicStatus.ACTIVATED_CONCLUSION, None
    facts, units, conflicts = metrics_for_subject(instance.subject.lemma, evidence)
    verdict = evaluate_condition(tree, facts, units=units, conflicts=conflicts)
    if verdict is ConditionEval.EVAL_FALSE:
        return None, None
    if verdict is ConditionEval.EVAL_UNKNOWN:
        return EpistemicStatus.CONDITIONAL_DERIVATION, tree
    return EpistemicStatus.ACTIVATED_CONCLUSION, None
