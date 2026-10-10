"""EIR records from the v1.6.1 schema.

Metric observations are optional fields on the proposition that states them.
A condition is not evaluated against a process-global number.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Polarity(str, Enum):
    AFFIRMATIVE = "AFFIRMATIVE"
    NEGATIVE = "NEGATIVE"


class NegationScope(str, Enum):
    NONE = "NONE"
    PREDICATE = "PREDICATE"
    ENTIRE_PROPOSITION = "ENTIRE_PROPOSITION"
    OBJECT_ONLY = "OBJECT_ONLY"


class Modality(str, Enum):
    DECLARATIVE = "DECLARATIVE"
    OBLIGATIVE = "OBLIGATIVE"
    PERMISSIVE = "PERMISSIVE"
    PROBABILISTIC = "PROBABILISTIC"
    COUNTERFACTUAL = "COUNTERFACTUAL"


class QuantifierType(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    ALL = "ALL"
    SOME = "SOME"
    NONE = "NONE"
    EXACTLY = "EXACTLY"
    AT_LEAST = "AT_LEAST"
    AT_MOST = "AT_MOST"


class Quantifier(BaseModel):
    model_config = ConfigDict(frozen=True)
    qtype: QuantifierType = QuantifierType.UNSPECIFIED
    value: Optional[float] = None


class LogicalOperator(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    GT = "GT"
    LT = "LT"
    GTE = "GTE"
    LTE = "LTE"
    EQ = "EQ"


class ConditionEval(str, Enum):
    EVAL_TRUE = "EVAL_TRUE"
    EVAL_FALSE = "EVAL_FALSE"
    EVAL_UNKNOWN = "EVAL_UNKNOWN"


class LogicalExpressionNode(BaseModel):
    model_config = ConfigDict(frozen=True)
    operator: LogicalOperator
    operands: Optional[list["LogicalExpressionNode"]] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[Union[float, str]] = None
    unit: Optional[str] = None


class EpistemicStatus(str, Enum):
    RETRIEVED_FACT = "RETRIEVED_FACT"
    ACTIVATED_CONCLUSION = "ACTIVATED_CONCLUSION"
    CONDITIONAL_DERIVATION = "CONDITIONAL_DERIVATION"
    HYPOTHESIS = "HYPOTHESIS"
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"
    ABSTENTION = "ABSTENTION"


class EdgeType(str, Enum):
    INFERENCE_DEPENDENCY = "INFERENCE_DEPENDENCY"
    CAUSAL_RELATION = "CAUSAL_RELATION"
    TEMPORAL_PRECEDENCE = "TEMPORAL_PRECEDENCE"
    CONTRADICTION = "CONTRADICTION"


class TypedGraphEdge(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_id: str
    target_id: str
    edge_type: EdgeType


class SourceSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_uri: str
    start_char: int
    end_char: int
    literal_text: str


class TypedEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    raw_text: str
    lemma: str
    is_plural: bool
    qualifiers: list[str] = Field(default_factory=list)


class TypedPredicate(BaseModel):
    model_config = ConfigDict(frozen=True)
    lemma: str
    tense: Literal["PAST", "PRESENT", "FUTURE"] = "PRESENT"
    aspect: Literal["SIMPLE", "PERFECT", "PROGRESSIVE"] = "SIMPLE"
    voice: Literal["ACTIVE", "PASSIVE"] = "ACTIVE"


class TemporalInterval(BaseModel):
    model_config = ConfigDict(frozen=True)
    valid_from: Optional[int] = None
    valid_to: Optional[int] = None

    def overlaps(self, other: Optional["TemporalInterval"]) -> bool:
        if other is None:
            return True
        self_start = self.valid_from if self.valid_from is not None else float("-inf")
        self_end = self.valid_to if self.valid_to is not None else float("inf")
        other_start = other.valid_from if other.valid_from is not None else float("-inf")
        other_end = other.valid_to if other.valid_to is not None else float("inf")
        return max(self_start, other_start) <= min(self_end, other_end)


class DerivationTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    inference_rule: str
    premise_ids: list[str] = Field(default_factory=list)
    singular_values: list[float] = Field(default_factory=list)
    ruin_distance: float = 0.0


class EIRProposition(BaseModel):
    model_config = ConfigDict(frozen=True)
    proposition_id: str
    source_span: Optional[SourceSpan] = None
    polarity: Polarity = Polarity.AFFIRMATIVE
    negation_scope: NegationScope = NegationScope.NONE
    modality: Modality = Modality.DECLARATIVE
    quantifier: Quantifier = Field(default_factory=Quantifier)
    subject: TypedEntity
    predicate: TypedPredicate
    object_target: Optional[TypedEntity] = None
    condition_tree: Optional[LogicalExpressionNode] = None
    temporal_interval: Optional[TemporalInterval] = None
    epistemic_status: EpistemicStatus = EpistemicStatus.RETRIEVED_FACT
    epistemic_weight: float = Field(ge=0.0, le=1.0)
    derivation: Optional[DerivationTrace] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = None


def _threshold(node: LogicalExpressionNode) -> float:
    if isinstance(node.threshold_value, (int, float)):
        return float(node.threshold_value)
    if node.threshold_value is None:
        raise ValueError(f"{node.metric_name} comparison has no threshold")
    return float(node.threshold_value)


def evaluate_condition(
    node: Optional[LogicalExpressionNode],
    facts: Optional[dict[str, float]],
    *,
    units: Optional[dict[str, Optional[str]]] = None,
    conflicts: Optional[set[str]] = None,
) -> ConditionEval:
    """True, false, or unknown for one condition tree.

    Unknown means the metric is absent, the unit differs, or two observations
    of it disagreed. A missing fact dict is unknown, not false.
    """
    if node is None:
        return ConditionEval.EVAL_TRUE
    operator = node.operator
    children = node.operands or []
    if operator is LogicalOperator.AND:
        verdicts = [evaluate_condition(child, facts, units=units, conflicts=conflicts) for child in children]
        if any(item is ConditionEval.EVAL_FALSE for item in verdicts):
            return ConditionEval.EVAL_FALSE
        if any(item is ConditionEval.EVAL_UNKNOWN for item in verdicts):
            return ConditionEval.EVAL_UNKNOWN
        return ConditionEval.EVAL_TRUE
    if operator is LogicalOperator.OR:
        verdicts = [evaluate_condition(child, facts, units=units, conflicts=conflicts) for child in children]
        if any(item is ConditionEval.EVAL_TRUE for item in verdicts):
            return ConditionEval.EVAL_TRUE
        if any(item is ConditionEval.EVAL_UNKNOWN for item in verdicts):
            return ConditionEval.EVAL_UNKNOWN
        return ConditionEval.EVAL_FALSE
    if operator is LogicalOperator.NOT:
        if len(children) != 1:
            return ConditionEval.EVAL_UNKNOWN
        inner = evaluate_condition(children[0], facts, units=units, conflicts=conflicts)
        if inner is ConditionEval.EVAL_UNKNOWN:
            return ConditionEval.EVAL_UNKNOWN
        if inner is ConditionEval.EVAL_TRUE:
            return ConditionEval.EVAL_FALSE
        return ConditionEval.EVAL_TRUE
    if facts is None or node.metric_name is None or node.metric_name not in facts:
        return ConditionEval.EVAL_UNKNOWN
    if conflicts is not None and node.metric_name in conflicts:
        return ConditionEval.EVAL_UNKNOWN
    if node.unit is not None and (units or {}).get(node.metric_name) != node.unit:
        return ConditionEval.EVAL_UNKNOWN
    value = facts[node.metric_name]
    threshold = _threshold(node)
    if operator is LogicalOperator.GT:
        passed = value > threshold
    elif operator is LogicalOperator.GTE:
        passed = value >= threshold
    elif operator is LogicalOperator.LT:
        passed = value < threshold
    elif operator is LogicalOperator.LTE:
        passed = value <= threshold
    elif operator is LogicalOperator.EQ:
        passed = math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-9)
    else:
        return ConditionEval.EVAL_UNKNOWN
    return ConditionEval.EVAL_TRUE if passed else ConditionEval.EVAL_FALSE


LogicalExpressionNode.model_rebuild()
EIRProposition.model_rebuild()
