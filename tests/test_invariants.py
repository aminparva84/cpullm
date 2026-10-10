"""Acceptance checks from the v1.6.1 specification, plus the cone and tri-state corrections."""

import numpy as np

from cpu_llm.audit.entailment import EpistemicEntailmentAuditor
from cpu_llm.cognition.adapter import PrismCognitionSubspaceAdapter, poincare_slice
from cpu_llm.geometry.poincare import is_in_entailment_cone
from cpu_llm.ir.schema import (
    ConditionEval,
    EdgeType,
    EIRProposition,
    EpistemicStatus,
    LogicalExpressionNode,
    LogicalOperator,
    Modality,
    Polarity,
    Quantifier,
    QuantifierType,
    TemporalInterval,
    TypedEntity,
    TypedGraphEdge,
    TypedPredicate,
    evaluate_condition,
)
from cpu_llm.realizer.surface import SemanticSurfaceRealizer


def _entity(raw: str, lemma: str, *, plural: bool = False, qualifiers: list[str] | None = None) -> TypedEntity:
    return TypedEntity(raw_text=raw, lemma=lemma, is_plural=plural, qualifiers=qualifiers or [])


def _predicate(lemma: str, *, tense: str = "PRESENT", voice: str = "ACTIVE") -> TypedPredicate:
    return TypedPredicate(lemma=lemma, tense=tense, voice=voice)


def test_sound_deductive_inference_and_condition_separation():
    adapter = PrismCognitionSubspaceAdapter()
    rule = EIRProposition(
        proposition_id="rule_cond",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.OBLIGATIVE,
        quantifier=Quantifier(qtype=QuantifierType.ALL),
        subject=_entity("approved suppliers", "supplier", plural=True, qualifiers=["approved"]),
        predicate=_predicate("undergo"),
        object_target=_entity("an audit", "audit"),
        condition_tree=LogicalExpressionNode(
            operator=LogicalOperator.GT,
            metric_name="revenue",
            threshold_value=10.0,
            unit="M",
        ),
        epistemic_weight=1.0,
    )
    fact = EIRProposition(
        proposition_id="fact_supp_x",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.DECLARATIVE,
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("be"),
        object_target=_entity("an approved supplier", "supplier", qualifiers=["approved"]),
        epistemic_weight=1.0,
    )
    dummy = [np.ones(1024, dtype=np.float32) * 0.1, np.ones(1024, dtype=np.float32) * 0.1]
    result = adapter.deliberate(np.zeros(1024, dtype=np.float32), [rule, fact], dummy, "tenant_test")
    derived = [
        item
        for item in result.grounded_propositions
        if item.epistemic_status is EpistemicStatus.CONDITIONAL_DERIVATION
    ]
    assert len(derived) == 1
    assert derived[0].condition_tree is not None
    assert poincare_slice(dummy[0]) is None


def test_condition_true_activates_and_false_emits_nothing():
    adapter = PrismCognitionSubspaceAdapter()
    rule = EIRProposition(
        proposition_id="rule_cond",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.OBLIGATIVE,
        quantifier=Quantifier(qtype=QuantifierType.ALL),
        subject=_entity("approved suppliers", "supplier", plural=True, qualifiers=["approved"]),
        predicate=_predicate("pass"),
        object_target=_entity("audit", "audit"),
        condition_tree=LogicalExpressionNode(
            operator=LogicalOperator.GT,
            metric_name="revenue",
            threshold_value=10.0,
            unit="M",
        ),
        epistemic_weight=1.0,
    )
    fact = EIRProposition(
        proposition_id="fact_42",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.DECLARATIVE,
        subject=_entity("Supplier_42", "Supplier_42"),
        predicate=_predicate("be"),
        object_target=_entity("approved supplier", "supplier", qualifiers=["approved"]),
        epistemic_weight=1.0,
    )

    def revenue(value: float) -> EIRProposition:
        return EIRProposition(
            proposition_id=f"rev_{value}",
            subject=_entity("Supplier_42", "Supplier_42"),
            predicate=_predicate("report"),
            metric_name="revenue",
            metric_value=value,
            metric_unit="M",
            epistemic_weight=1.0,
        )

    zeros = [np.zeros(128), np.zeros(128), np.zeros(128)]
    activated = adapter.deliberate(np.zeros(128), [rule, fact, revenue(15.0)], zeros, "tenant_test")
    conclusions = [
        item
        for item in activated.grounded_propositions
        if item.epistemic_status is EpistemicStatus.ACTIVATED_CONCLUSION
    ]
    assert len(conclusions) == 1
    assert conclusions[0].condition_tree is None
    blocked = adapter.deliberate(np.zeros(128), [rule, fact, revenue(3.0)], zeros, "tenant_test")
    assert not any(
        item.epistemic_status
        in {EpistemicStatus.ACTIVATED_CONCLUSION, EpistemicStatus.CONDITIONAL_DERIVATION}
        for item in blocked.grounded_propositions
    )
    node = rule.condition_tree
    assert node is not None
    assert evaluate_condition(node, {"revenue": 15.0}, units={"revenue": "M"}) is ConditionEval.EVAL_TRUE
    assert evaluate_condition(node, {"revenue": 3.0}, units={"revenue": "M"}) is ConditionEval.EVAL_FALSE
    assert evaluate_condition(node, {}, units={}) is ConditionEval.EVAL_UNKNOWN


def test_scope_aware_temporal_contradiction_isolation():
    adapter = PrismCognitionSubspaceAdapter()
    yes = EIRProposition(
        proposition_id="prop_2024",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.DECLARATIVE,
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("approve", tense="PAST", voice="PASSIVE"),
        temporal_interval=TemporalInterval(valid_from=2024, valid_to=2024),
        epistemic_weight=1.0,
    )
    no = EIRProposition(
        proposition_id="prop_2026",
        polarity=Polarity.NEGATIVE,
        modality=Modality.DECLARATIVE,
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("approve", tense="PRESENT", voice="PASSIVE"),
        temporal_interval=TemporalInterval(valid_from=2026, valid_to=2026),
        epistemic_weight=1.0,
    )
    dummy = [np.ones(1024, dtype=np.float32) * 0.1, np.ones(1024, dtype=np.float32) * 0.1]
    result = adapter.deliberate(np.zeros(1024, dtype=np.float32), [yes, no], dummy, "tenant_test")
    assert result.ruin_distance == 0.0
    assert not any(
        item.epistemic_status is EpistemicStatus.UNRESOLVED_CONTRADICTION
        for item in result.grounded_propositions
    )


def test_overlapping_declaratives_ruin_and_permission_does_not():
    adapter = PrismCognitionSubspaceAdapter()
    subject = _entity("Supplier X", "Supplier X")
    predicate = _predicate("approve")
    yes = EIRProposition(
        proposition_id="yes",
        subject=subject,
        predicate=predicate,
        epistemic_weight=1.0,
    )
    no = EIRProposition(
        proposition_id="no",
        polarity=Polarity.NEGATIVE,
        subject=subject,
        predicate=predicate,
        epistemic_weight=1.0,
    )
    result = adapter.deliberate(np.zeros(4), [yes, no], [np.zeros(4), np.zeros(4)], "tenant_test")
    assert result.ruin_distance == 1.0
    assert result.grounded_propositions[0].epistemic_status is EpistemicStatus.UNRESOLVED_CONTRADICTION
    permitted = no.model_copy(update={"modality": Modality.PERMISSIVE, "proposition_id": "may"})
    compatible = adapter.deliberate(
        np.zeros(4),
        [yes, permitted],
        [np.zeros(4), np.zeros(4)],
        "tenant_test",
    )
    assert compatible.ruin_distance == 0.0
    likely = yes.model_copy(update={"modality": Modality.PROBABILISTIC, "proposition_id": "likely"})
    tension = adapter.deliberate(np.zeros(4), [likely, no], [np.zeros(4), np.zeros(4)], "tenant_test")
    assert tension.ruin_distance == 0.0
    weights = {item.proposition_id: item.epistemic_weight for item in tension.grounded_propositions}
    assert weights["likely"] == 0.5
    assert weights["no"] == 0.5


def test_cone_keeps_an_inside_child_and_drops_an_outside_one():
    parent = np.zeros(128)
    parent[0] = 0.2
    child = np.zeros(128)
    child[0] = 0.55
    outside = np.zeros(128)
    outside[3] = 0.55
    assert is_in_entailment_cone(parent, child)
    assert not is_in_entailment_cone(child, parent)
    assert not is_in_entailment_cone(parent, outside)
    adapter = PrismCognitionSubspaceAdapter()
    rule = EIRProposition(
        proposition_id="rule",
        modality=Modality.OBLIGATIVE,
        quantifier=Quantifier(qtype=QuantifierType.ALL),
        subject=_entity("suppliers", "supplier", qualifiers=["approved"]),
        predicate=_predicate("pass"),
        object_target=_entity("audit", "audit"),
        epistemic_weight=1.0,
    )
    fact = EIRProposition(
        proposition_id="fact",
        subject=_entity("Supplier_42", "Supplier_42"),
        predicate=_predicate("be"),
        object_target=_entity("approved supplier", "supplier", qualifiers=["approved"]),
        epistemic_weight=1.0,
    )

    def derived(coord: np.ndarray) -> list[EIRProposition]:
        result = adapter.deliberate(np.zeros(128), [rule, fact], [parent, coord], "tenant_test")
        return [item for item in result.grounded_propositions if item.derivation is not None]

    assert len(derived(child)) == 1
    assert derived(outside) == []


def test_generalized_anchor_regex_and_terminal_punctuation():
    realizer = SemanticSurfaceRealizer()
    auditor = EpistemicEntailmentAuditor()
    proposition = EIRProposition(
        proposition_id="prop_none_quant",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.PERMISSIVE,
        quantifier=Quantifier(qtype=QuantifierType.NONE),
        subject=_entity("patient", "patient"),
        predicate=_predicate("discontinue"),
        object_target=_entity("therapy", "therapy"),
        epistemic_weight=1.0,
    )
    text, delivered = realizer.synthesize([proposition], [])
    assert text.endswith(".")
    assert "prop_none_quant" in delivered
    assert auditor.audit_delivered_output(delivered, [proposition]) is True
    assert text == "No patient may discontinue therapy."


def test_phase2_empty_passage_is_the_specified_sentence():
    text, delivered = SemanticSurfaceRealizer().synthesize([], [])
    assert text == "No verified epistemic evidence available."
    assert delivered == {}


def test_phase2_causal_order_pronoun_and_inference_without_consequently():
    realizer = SemanticSurfaceRealizer()
    auditor = EpistemicEntailmentAuditor()
    earlier = EIRProposition(
        proposition_id="fact-a",
        subject=_entity("Supplier X", "supplier"),
        predicate=_predicate("be"),
        object_target=_entity("a supplier", "supplier"),
        epistemic_weight=1.0,
    )
    later = EIRProposition(
        proposition_id="duty-b",
        modality=Modality.OBLIGATIVE,
        subject=_entity("Supplier X", "supplier"),
        predicate=_predicate("pass"),
        object_target=_entity("audit", "audit"),
        epistemic_weight=1.0,
    )
    causal = [
        TypedGraphEdge(source_id="fact-a", target_id="duty-b", edge_type=EdgeType.CAUSAL_RELATION)
    ]
    text, delivered = realizer.synthesize([later, earlier], causal)
    assert list(delivered) == ["fact-a", "duty-b"]
    assert text == "Supplier X is a supplier. Consequently, it must pass audit."
    assert auditor.audit_delivered_output(delivered, [earlier, later]) is True
    inference = [
        TypedGraphEdge(
            source_id="fact-a",
            target_id="duty-b",
            edge_type=EdgeType.INFERENCE_DEPENDENCY,
        )
    ]
    plain, _ = realizer.synthesize([later, earlier], inference)
    assert plain == "Supplier X is a supplier. it must pass audit."
    assert "Consequently" not in plain


def test_phase2_quantifiers_polarity_passive_and_conditions():
    realizer = SemanticSurfaceRealizer()
    auditor = EpistemicEntailmentAuditor()
    universal = EIRProposition(
        proposition_id="all-suppliers",
        modality=Modality.DECLARATIVE,
        quantifier=Quantifier(qtype=QuantifierType.ALL),
        subject=_entity("suppliers", "supplier", plural=True, qualifiers=["approved"]),
        predicate=_predicate("pass"),
        object_target=_entity("audit", "audit"),
        epistemic_weight=1.0,
    )
    partial = EIRProposition(
        proposition_id="some-suppliers",
        polarity=Polarity.NEGATIVE,
        quantifier=Quantifier(qtype=QuantifierType.SOME),
        subject=_entity("suppliers", "supplier", plural=True),
        predicate=_predicate("pass"),
        object_target=_entity("audit", "audit"),
        epistemic_weight=1.0,
    )
    past = EIRProposition(
        proposition_id="past-approval",
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("approve", tense="PAST", voice="PASSIVE"),
        epistemic_weight=1.0,
    )
    obliged = EIRProposition(
        proposition_id="must-be",
        polarity=Polarity.NEGATIVE,
        modality=Modality.OBLIGATIVE,
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("approve", voice="PASSIVE"),
        epistemic_weight=1.0,
    )
    conditional = EIRProposition(
        proposition_id="revenue-rule",
        modality=Modality.OBLIGATIVE,
        quantifier=Quantifier(qtype=QuantifierType.ALL),
        subject=_entity("suppliers", "supplier", plural=True),
        predicate=_predicate("undergo"),
        object_target=_entity("an audit", "audit"),
        condition_tree=LogicalExpressionNode(
            operator=LogicalOperator.AND,
            operands=[
                LogicalExpressionNode(
                    operator=LogicalOperator.GT,
                    metric_name="revenue",
                    threshold_value=10.0,
                    unit="M",
                ),
                LogicalExpressionNode(
                    operator=LogicalOperator.NOT,
                    operands=[
                        LogicalExpressionNode(
                            operator=LogicalOperator.LT,
                            metric_name="margin",
                            threshold_value=1.5,
                        )
                    ],
                ),
            ],
        ),
        epistemic_weight=1.0,
    )
    text, delivered = realizer.synthesize([universal, partial, past, obliged, conditional], [])
    assert "All approved suppliers pass audit." in text
    assert "Some suppliers do not pass audit." in text
    assert "Supplier X was approved." in text
    assert "it must not be approved." in text
    assert (
        "All suppliers must undergo an audit provided that revenue is greater than 10.0 M "
        "AND not (margin is less than 1.5)."
    ) in text
    assert auditor.audit_delivered_output(delivered, [universal, partial, past, obliged, conditional]) is True
    hidden = dict(delivered)
    hidden["revenue-rule"] = hidden["revenue-rule"].replace("10.0", "9")
    assert auditor.audit_delivered_output(hidden, [universal, partial, past, obliged, conditional]) is False
    dropped_voice = dict(delivered)
    dropped_voice["past-approval"] = "Supplier X approved."
    assert auditor.audit_delivered_output(dropped_voice, [universal, partial, past, obliged, conditional]) is False


def test_phase2_auditor_rejects_flips_and_a_cycle_still_realizes():
    realizer = SemanticSurfaceRealizer()
    auditor = EpistemicEntailmentAuditor()
    denied = EIRProposition(
        proposition_id="denied",
        polarity=Polarity.NEGATIVE,
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("approve"),
        object_target=_entity("the filing", "filing"),
        epistemic_weight=1.0,
    )
    text, delivered = realizer.synthesize([denied], [])
    assert text == "Supplier X does not approve the filing."
    assert auditor.audit_delivered_output(delivered, [denied]) is True
    flipped = {"denied": "Supplier X approves the filing."}
    assert auditor.audit_delivered_output(flipped, [denied]) is False
    wrong_verb = {"denied": "Supplier X does not reject the filing."}
    assert auditor.audit_delivered_output(wrong_verb, [denied]) is False
    missing = {"denied": "Supplier X does not approve the filing.", "extra": "Unused."}
    assert auditor.audit_delivered_output(missing, [denied]) is False
    left = EIRProposition(
        proposition_id="left",
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("approve", voice="PASSIVE"),
        epistemic_weight=1.0,
    )
    right = EIRProposition(
        proposition_id="right-node",
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("pass"),
        object_target=_entity("audit", "audit"),
        epistemic_weight=1.0,
    )
    cycle = [
        TypedGraphEdge(source_id="left", target_id="right-node", edge_type=EdgeType.CAUSAL_RELATION),
        TypedGraphEdge(source_id="right-node", target_id="left", edge_type=EdgeType.CAUSAL_RELATION),
        TypedGraphEdge(source_id="left", target_id="left", edge_type=EdgeType.CAUSAL_RELATION),
    ]
    cycled, cycled_map = realizer.synthesize([left, right], cycle)
    assert cycled.startswith("Supplier X is approved.")
    assert "Consequently, it passes audit." in cycled
    assert auditor.audit_delivered_output(cycled_map, [left, right]) is True


def test_phase2_realizes_the_conditional_derivation():
    adapter = PrismCognitionSubspaceAdapter()
    rule = EIRProposition(
        proposition_id="rule_cond",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.OBLIGATIVE,
        quantifier=Quantifier(qtype=QuantifierType.ALL),
        subject=_entity("approved suppliers", "supplier", plural=True, qualifiers=["approved"]),
        predicate=_predicate("undergo"),
        object_target=_entity("an audit", "audit"),
        condition_tree=LogicalExpressionNode(
            operator=LogicalOperator.GT,
            metric_name="revenue",
            threshold_value=10.0,
            unit="M",
        ),
        epistemic_weight=1.0,
    )
    fact = EIRProposition(
        proposition_id="fact_supp_x",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.DECLARATIVE,
        subject=_entity("Supplier X", "Supplier X"),
        predicate=_predicate("be"),
        object_target=_entity("an approved supplier", "supplier", qualifiers=["approved"]),
        epistemic_weight=1.0,
    )
    dummy = [np.ones(1024, dtype=np.float32) * 0.1, np.ones(1024, dtype=np.float32) * 0.1]
    result = adapter.deliberate(np.zeros(1024, dtype=np.float32), [rule, fact], dummy, "tenant_test")
    text, delivered = SemanticSurfaceRealizer().synthesize(result.grounded_propositions, result.graph_edges)
    assert "Consequently" not in text
    assert "it must undergo an audit provided that revenue is greater than 10.0 M." in text
    assert EpistemicEntailmentAuditor().audit_delivered_output(delivered, result.grounded_propositions) is True
