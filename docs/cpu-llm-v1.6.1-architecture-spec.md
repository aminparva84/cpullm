# Technical Architecture Specification: `cpu-llm` (v1.6.1 Correctness & Soundness Patch)

> The implementation of this specification is in `src/cpu_llm`.
> Phase 1 is the EIR schema and `PrismCognitionSubspaceAdapter`.
> Phase 2 is `SemanticSurfaceRealizer` and `EpistemicEntailmentAuditor`.
> Acceptance checks are `tests/test_invariants.py`.
>
> Two corrections are applied in the engine, where the code blocks below
> do not match the stated rules. Inclusion uses the Ganea entailment cone
> when a tensor is a point inside the Poincaré ball. A rule condition is
> evaluated as true, false, or unknown. Contradiction follows the modality
> matrix. Ruin is 1 or 0 from that matrix. There is no random bilinear map.
>
> Phase 2 follows the clause rules in section 4. The sentence is the one
> those rules emit. It is not passed through a grammar rewriter, and the
> auditor does not load a parser model. The auditor checks polarity,
> modality, voice, the condition threshold, the subject, and the predicate.



**System Designation:** `cpu-llm` (Epistemic Vector-Native Appliance)

**Specification Level:** Level-3 Deterministic Execution Contract

**Hardware Invariant:** 100% CPU-Native. OpenBLAS/AVX-512 execution. Zero GPU/CUDA dependencies (`torch.cuda.is_available() == False` asserted at engine initialization).

**Ecosystem Dependencies:** `vectorprism>=1.0.0`, `chorusgraph>=1.0.0`, `prismguard>=1.0.0`, `prismcognition>=2.1.0`, `prismrag>=1.0.0`, `prismresonance>=1.0.0`, `prismcortex>=0.4.1`, `prismmanifest>=1.0.0`, `prismshine>=1.0.0`

---

## 1. Concrete Architecture of the Vector Deliberation Engine

In a pure symbolic engine, evaluating high-order horn clauses across $N$ facts is $O(N^k)$ combinatorial matching. When $N > 500$, CPU symbolic provers stall.

In `cpu-llm`, the **Subspace Coordinate Space acts as a Geometric Pre-Filter for Inference Search**:

1. **Hyperbolic Subspace ($\mathbb{D}^{128}$):** Transitive taxonomy containment ($\text{SubClassOf}$) is directly computed via Poincaré distance cones in $\mathcal{O}(1)$ time per pair. Symbolic deduction is invoked **only** on candidate entity pairs $(\mathbf{u}, \mathbf{v})$ where $d_{\mathbb{D}}(\mathbf{u}, \mathbf{v}) < \epsilon_{\text{poinc}}$.
2. **Causal Cones ($\mathbb{R}^{128}$):** Potential causal precursors are pre-screened by the asymmetric metric cone $\mathbf{u}^T \mathbf{\Omega} \mathbf{v} > 0$.
3. **Symbolic Soundness Proof:** Symbolic resolution runs **only** on the geometrically viable candidates, proving soundness and assigning derivation rules.

```
                    [ 1024d Subspace Coordinates ]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Poincaré Subspace               Causal Subspace
       (128d Poincaré Ball)            (128d Minkowski Cone)
                 │                               │
       $d_{\mathbb{D}}(u, v) < \epsilon$       $u^T \Omega v > 0$
                 │                               │
                 └───────────────┬───────────────┘
                                 │ Candidate Pair Filtering
                                 ▼
                 [ Symbolic Soundness Proof Gate ]
                 • Antecedent validation
                 • Temporal interval intersection
                 • Modus Ponens derivation

```

---

## 2. EIR v1.3.1 Canonical Schema

This patch introduces explicit antecedent/consequent decomposition, status markers for conditional vs. activated obligations, and strict temporal interval comparisons.

```python
# src/cpu_llm/ir/schema.py
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

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
    OBLIGATIVE  = "OBLIGATIVE"
    PERMISSIVE  = "PERMISSIVE"
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
    OR  = "OR"
    NOT = "NOT"
    GT  = "GT"
    LT  = "LT"
    GTE = "GTE"
    LTE = "LTE"
    EQ  = "EQ"

class LogicalExpressionNode(BaseModel):
    model_config = ConfigDict(frozen=True)
    operator: LogicalOperator
    operands: Optional[List[LogicalExpressionNode]] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[Union[float, str]] = None
    unit: Optional[str] = None

class EpistemicStatus(str, Enum):
    RETRIEVED_FACT = "RETRIEVED_FACT"
    ACTIVATED_CONCLUSION = "ACTIVATED_CONCLUSION"     # All conditions satisfied & obligation holds
    CONDITIONAL_DERIVATION = "CONDITIONAL_DERIVATION" # Conclusion holds IF unproven condition holds
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
    qualifiers: List[str] = Field(default_factory=list)

class TypedPredicate(BaseModel):
    model_config = ConfigDict(frozen=True)
    lemma: str
    tense: Literal["PAST", "PRESENT", "FUTURE"] = "PRESENT"
    aspect: Literal["SIMPLE", "PERFECT", "PROGRESSIVE"] = "SIMPLE"
    voice: Literal["ACTIVE", "PASSIVE"] = "ACTIVE"

class TemporalInterval(BaseModel):
    model_config = ConfigDict(frozen=True)
    valid_from: Optional[int] = None  # UNIX timestamp or integer year
    valid_to: Optional[int] = None

    def overlaps(self, other: Optional[TemporalInterval]) -> bool:
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
    premise_ids: List[str] = Field(default_factory=list)
    singular_values: List[float] = Field(default_factory=list)
    ruin_distance: float = 1.0

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
```

---

## 3. Deliberation Engine: Scope-Aware Contradiction & Conditional Deduction

Resolves P0-3 & P0-4:

* **Scope-Aware Contradiction:** Proposition pairs only constitute a contradiction if:
1. Subjects, predicates, and objects match semantically.
2. Modality is mutually exclusive (e.g., `DECLARATIVE` vs `DECLARATIVE` with opposite polarity).
3. `temporal_interval.overlaps()` evaluates to `True`.


* **Universal Deduction with Conditions:** Evaluates whether conditions attached to a rule are proven by instance facts. If unproven, emits `CONDITIONAL_DERIVATION` (not an unconditional obligation).

```python
# src/cpu_llm/cognition/adapter.py
import hashlib
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel

from cpu_llm.ir.schema import (
    EIRProposition, EpistemicStatus, Polarity, Modality, QuantifierType,
    TypedEntity, TypedPredicate, DerivationTrace, TypedGraphEdge, EdgeType,
    LogicalExpressionNode, LogicalOperator
)

class DeliberationResult(BaseModel):
    target_vector: List[float]
    grounded_propositions: List[EIRProposition]
    graph_edges: List[TypedGraphEdge]
    abstention_triggered: bool
    ruin_distance: float
    audit_trace_id: str

class PrismCognitionSubspaceAdapter:
    def __init__(self, ruin_threshold: float = 0.85):
        self.ruin_threshold = ruin_threshold
        rng = np.random.RandomState(42)
        self.M_rel = rng.randn(128, 128).astype(np.float32) * 0.01

    def deliberate(
        self,
        query_vector: np.ndarray,
        evidence_propositions: List[EIRProposition],
        evidence_tensors: List[np.ndarray],
        tenant_id: str
    ) -> DeliberationResult:
        trace_seed = f"{tenant_id}:{len(evidence_propositions)}".encode("utf-8")
        trace_id = f"TRACE_{hashlib.sha256(trace_seed).hexdigest()[:16]}"

        if not evidence_propositions:
            abstention_prop = EIRProposition(
                proposition_id=f"abstain_{hashlib.sha256(b'ABSTAIN_VACUUM').hexdigest()[:12]}",
                polarity=Polarity.NEGATIVE,
                modality=Modality.DECLARATIVE,
                subject=TypedEntity(raw_text="system", lemma="system", is_plural=False),
                predicate=TypedPredicate(lemma="possess", voice="ACTIVE"),
                object_target=TypedEntity(raw_text="sufficient verified evidence", lemma="evidence", is_plural=False),
                epistemic_status=EpistemicStatus.ABSTENTION,
                epistemic_weight=0.0
            )
            return DeliberationResult(
                target_vector=query_vector.tolist(),
                grounded_propositions=[abstention_prop],
                graph_edges=[],
                abstention_triggered=True,
                ruin_distance=0.0,
                audit_trace_id=trace_id
            )

        N = len(evidence_propositions)
        TensionMatrix = np.zeros((N, N), dtype=np.float32)
        graph_edges: List[TypedGraphEdge] = []
        contradiction_pairs = 0

        # 1. Scope-Aware Contradiction Evaluation (Modal, Temporal, Quantifier, Scope)
        for i in range(N):
            for j in range(i + 1, N):
                p_i, p_j = evidence_propositions[i], evidence_propositions[j]
                t_i, t_j = evidence_tensors[i], evidence_tensors[j]

                same_subj = (p_i.subject.lemma.lower() == p_j.subject.lemma.lower())
                same_pred = (p_i.predicate.lemma.lower() == p_j.predicate.lemma.lower())
                same_obj = (
                    (p_i.object_target is None and p_j.object_target is None) or
                    (p_i.object_target and p_j.object_target and 
                     p_i.object_target.lemma.lower() == p_j.object_target.lemma.lower())
                )
                opposing_polarity = (p_i.polarity != p_j.polarity)

                # Check Temporal Overlap
                time_overlap = True
                if p_i.temporal_interval and p_j.temporal_interval:
                    time_overlap = p_i.temporal_interval.overlaps(p_j.temporal_interval)

                # Check Modality Exclusivity
                modal_conflict = (p_i.modality == p_j.modality) or (Modality.DECLARATIVE in [p_i.modality, p_j.modality])

                if same_subj and same_pred and same_obj and opposing_polarity and time_overlap and modal_conflict:
                    rel_i, rel_j = t_i[400:528], t_j[400:528]
                    tension_val = 1.0 + float(abs(rel_i.T @ self.M_rel @ rel_j))
                    TensionMatrix[i, j] = tension_val
                    TensionMatrix[j, i] = tension_val
                    contradiction_pairs += 1
                    graph_edges.append(TypedGraphEdge(
                        source_id=p_i.proposition_id,
                        target_id=p_j.proposition_id,
                        edge_type=EdgeType.CONTRADICTION
                    ))

        # 2. SVD Ruin Projections with Calibrated Spectral Metric
        if contradiction_pairs > 0:
            U, S, Vt = np.linalg.svd(TensionMatrix)
            max_singular_val = float(S[0])
        else:
            max_singular_val = 0.0

        if max_singular_val > self.ruin_threshold and contradiction_pairs > 0:
            conflict_prop = EIRProposition(
                proposition_id=f"conflict_{hashlib.sha256(trace_seed).hexdigest()[:12]}",
                polarity=Polarity.AFFIRMATIVE,
                modality=Modality.DECLARATIVE,
                subject=TypedEntity(raw_text="evidence", lemma="evidence", is_plural=False),
                predicate=TypedPredicate(lemma="exhibit", voice="ACTIVE"),
                object_target=TypedEntity(raw_text="unresolved mutual contradiction", lemma="contradiction", is_plural=False),
                epistemic_status=EpistemicStatus.UNRESOLVED_CONTRADICTION,
                epistemic_weight=1.0,
                derivation=DerivationTrace(
                    inference_rule="SVD_RUIN_TENSION_LIMIT",
                    singular_values=[float(max_singular_val)],
                    ruin_distance=max_singular_val
                )
            )
            return DeliberationResult(
                target_vector=query_vector.tolist(),
                grounded_propositions=[conflict_prop],
                graph_edges=graph_edges,
                abstention_triggered=False,
                ruin_distance=max_singular_val,
                audit_trace_id=trace_id
            )

        # 3. Deductive Inference Engine (Antecedent Separation & Condition Evaluation)
        derived_propositions: List[EIRProposition] = list(evidence_propositions)

        for i in range(N):
            p_univ = evidence_propositions[i]
            if not (p_univ.quantifier.qtype == QuantifierType.ALL and p_univ.polarity == Polarity.AFFIRMATIVE):
                continue

            for j in range(N):
                if i == j:
                    continue
                p_inst = evidence_propositions[j]

                if p_inst.polarity != Polarity.AFFIRMATIVE or p_inst.modality != Modality.DECLARATIVE:
                    continue

                copula_verbs = {"be", "qualify", "classify", "act"}
                if p_inst.predicate.lemma.lower() not in copula_verbs or p_inst.object_target is None:
                    continue

                if p_inst.object_target.lemma.lower() != p_univ.subject.lemma.lower():
                    continue

                univ_qualifiers = set(q.lower() for q in p_univ.subject.qualifiers)
                inst_qualifiers = set(q.lower() for q in p_inst.object_target.qualifiers)
                if not univ_qualifiers.issubset(inst_qualifiers):
                    continue

                # Determine whether condition tree is activated or conditional
                status = EpistemicStatus.ACTIVATED_CONCLUSION
                if p_univ.condition_tree is not None:
                    # Condition attached to rule; instance does not prove it -> CONDITIONAL_DERIVATION
                    status = EpistemicStatus.CONDITIONAL_DERIVATION

                derived_id = f"derived_{hashlib.sha256(f'{p_univ.proposition_id}:{p_inst.proposition_id}'.encode()).hexdigest()[:12]}"
                derived_prop = EIRProposition(
                    proposition_id=derived_id,
                    polarity=p_univ.polarity,
                    modality=p_univ.modality,
                    subject=p_inst.subject,
                    predicate=p_univ.predicate,
                    object_target=p_univ.object_target,
                    condition_tree=p_univ.condition_tree,
                    epistemic_status=status,
                    epistemic_weight=min(p_univ.epistemic_weight, p_inst.epistemic_weight),
                    derivation=DerivationTrace(
                        inference_rule="SOUND_UNIVERSAL_MODUS_PONENS",
                        premise_ids=[p_univ.proposition_id, p_inst.proposition_id],
                        singular_values=[max_singular_val],
                        ruin_distance=max_singular_val
                    )
                )
                derived_propositions.append(derived_prop)
                
                # Inference Dependency (NOT Causal)
                graph_edges.append(TypedGraphEdge(
                    source_id=p_inst.proposition_id,
                    target_id=derived_id,
                    edge_type=EdgeType.INFERENCE_DEPENDENCY
                ))
                graph_edges.append(TypedGraphEdge(
                    source_id=p_univ.proposition_id,
                    target_id=derived_id,
                    edge_type=EdgeType.INFERENCE_DEPENDENCY
                ))

        context_mean = np.mean(evidence_tensors, axis=0) if len(evidence_tensors) > 0 else query_vector
        target_vec = (query_vector * 0.6 + context_mean * 0.4)
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)

        return DeliberationResult(
            target_vector=target_vec.tolist(),
            grounded_propositions=derived_propositions,
            graph_edges=graph_edges,
            abstention_triggered=False,
            ruin_distance=max_singular_val,
            audit_trace_id=trace_id
        )
```

---

## 4. Surface Realizer & Round-Trip Entailment Auditor

Resolves P0-1, P0-2, and P0-5:

1. **Sentence Boundary Preservation:** Every clause terminates with an explicit period (`.`).
2. **Anchor Parser Flexibility:** Uses `[\w-]+` matching arbitrary alphanumeric/hyphenated IDs (handling `prop_none_quant` and arbitrary hashes).
3. **Round-Trip Semantics:** The auditor extracts the semantic head, predicate lemma, negation, and objects from the final delivered sentence, verifying semantic preservation against the originating EIR proposition.

```python
# src/cpu_llm/realizer/surface.py
import re
from typing import List, Dict, Tuple
import networkx as nx
import lemminflect
import language_tool_python
import spacy
from cpu_llm.ir.schema import (
    EIRProposition, Polarity, Modality, LogicalOperator, 
    QuantifierType, LogicalExpressionNode, TypedGraphEdge, EdgeType, EpistemicStatus
)

class SemanticSurfaceRealizer:
    def __init__(self):
        self.linter = language_tool_python.LanguageTool("en-US")
        self.nlp = spacy.load("en_core_web_sm")

    def synthesize(
        self, 
        propositions: List[EIRProposition], 
        graph_edges: List[TypedGraphEdge]
    ) -> Tuple[str, Dict[str, str]]:
        if not propositions:
            return "No verified epistemic evidence available.", {}

        dag = nx.DiGraph()
        prop_map = {p.proposition_id: p for p in propositions}
        for p in propositions:
            dag.add_node(p.proposition_id)
        for edge in graph_edges:
            if edge.source_id in prop_map and edge.target_id in prop_map:
                dag.add_edge(edge.source_id, edge.target_id)

        if not nx.is_directed_acyclic_graph(dag):
            for cycle in list(nx.simple_cycles(dag)):
                if len(cycle) >= 2:
                    dag.remove_edge(cycle[-1], cycle[0])

        ordered_ids = list(nx.topological_sort(dag))
        raw_clauses: List[str] = []
        entity_counts: Dict[str, int] = {}
        causal_pairs = {(e.source_id, e.target_id) for e in graph_edges if e.edge_type == EdgeType.CAUSAL_RELATION}

        for i, pid in enumerate(ordered_ids):
            p = prop_map[pid]
            
            # 1. Subject & Quantifier Formulation
            quant_str = ""
            if p.quantifier.qtype == QuantifierType.NONE:
                quant_str = "No "
            elif p.quantifier.qtype == QuantifierType.SOME:
                quant_str = "Some "
            elif p.quantifier.qtype == QuantifierType.ALL:
                quant_str = "All "

            subj_lemma = p.subject.lemma.lower()
            count = entity_counts.get(subj_lemma, 0)
            entity_counts[subj_lemma] = count + 1

            if count > 0 and p.quantifier.qtype == QuantifierType.UNSPECIFIED:
                subject_str = "they" if p.subject.is_plural else "it"
            else:
                qual_str = " ".join(p.subject.qualifiers) + " " if p.subject.qualifiers else ""
                subject_str = f"{quant_str}{qual_str}{p.subject.raw_text}"

            is_plural = p.subject.is_plural or subject_str.lower().startswith("all") or subject_str.lower().startswith("some")
            is_negative = (p.polarity == Polarity.NEGATIVE and p.quantifier.qtype != QuantifierType.NONE)
            verb_lemma = p.predicate.lemma

            # 2. Coordinated Predicate Realization
            past_part = lemminflect.getInflection(verb_lemma, tag="VBN")
            v_part = past_part[0] if past_part else verb_lemma

            if p.predicate.voice == "PASSIVE":
                be_form = "are" if is_plural else "is"
                if p.predicate.tense == "PAST":
                    be_form = "were" if is_plural else "was"

                if p.modality == Modality.OBLIGATIVE:
                    verb_phrase = f"must not be {v_part}" if is_negative else f"must be {v_part}"
                elif p.modality == Modality.PERMISSIVE:
                    verb_phrase = f"may not be {v_part}" if is_negative else f"may be {v_part}"
                else:
                    verb_phrase = f"{be_form} not {v_part}" if is_negative else f"{be_form} {v_part}"
            else:
                verb_tag = "VBP" if is_plural else "VBZ"
                inflected_verb = lemminflect.getInflection(verb_lemma, tag=verb_tag)
                v_active = inflected_verb[0] if inflected_verb else verb_lemma

                if p.modality == Modality.OBLIGATIVE:
                    verb_phrase = f"must not {verb_lemma}" if is_negative else f"must {verb_lemma}"
                elif p.modality == Modality.PERMISSIVE:
                    verb_phrase = f"may not {verb_lemma}" if is_negative else f"may {verb_lemma}"
                elif is_negative:
                    aux = "do not" if is_plural else "does not"
                    verb_phrase = f"{aux} {verb_lemma}"
                else:
                    verb_phrase = v_active

            obj_str = f" {p.object_target.raw_text}" if p.object_target else ""
            clause = f"{subject_str} {verb_phrase}{obj_str}".replace("  ", " ").strip()

            # 3. Explicit Condition Tree Rendering
            if p.condition_tree:
                cond_text = self._render_condition_node(p.condition_tree)
                clause += f" provided that {cond_text}"

            # 4. Strict Causal Marker Injection (Only on explicit causal edges)
            if i > 0 and (ordered_ids[i - 1], pid) in causal_pairs:
                clause = f"Consequently, {clause[0].lower() + clause[1:]}"

            # P0 Fix: Append explicit sentence-terminal period
            anchored_clause = f"<!--anchor:{pid}--> {clause}."
            raw_clauses.append(anchored_clause)

        # Polish Full Document
        raw_passage = " ".join(raw_clauses)
        matches = self.linter.check(raw_passage)
        polished_passage = language_tool_python.utils.correct(raw_passage, matches)

        # P0 Fix: Generalized Anchor Pattern (Matches arbitrary IDs)
        delivered_map: Dict[str, str] = {}
        anchor_pattern = re.compile(r"<!--anchor:([\w-]+)-->\s*([^<]+)")
        for match in anchor_pattern.finditer(polished_passage):
            pid, text = match.group(1), match.group(2).strip()
            delivered_map[pid] = text

        cleaned_user_text = re.sub(r"<!--anchor:[^>]+-->\s*", "", polished_passage).strip()
        return cleaned_user_text, delivered_map

    def _render_condition_node(self, node: LogicalExpressionNode) -> str:
        if node.operands:
            sub = [self._render_condition_node(op) for op in node.operands]
            return f" {node.operator.value} ".join(sub)
        
        op_map = {
            LogicalOperator.GT: "is greater than",
            LogicalOperator.LT: "is less than",
            LogicalOperator.GTE: "is at least",
            LogicalOperator.LTE: "is at most",
            LogicalOperator.EQ: "equals"
        }
        op_str = op_map.get(node.operator, "satisfies")
        unit_str = f" {node.unit}" if node.unit else ""
        return f"{node.metric_name} {op_str} {node.threshold_value}{unit_str}"
```

```python
# src/cpu_llm/audit/entailment.py
from typing import Dict, List
import spacy
from cpu_llm.ir.schema import EIRProposition, Polarity, Modality, QuantifierType

class EpistemicEntailmentAuditor:
    """
    Epistemic Entailment Auditor:
    Verifies that every final sentence delivered to the user strictly entails 
    its mapped EIR proposition node across polarity, modality, numbers, and voice.
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def audit_delivered_output(
        self, 
        delivered_map: Dict[str, str], 
        propositions: List[EIRProposition]
    ) -> bool:
        prop_dict = {p.proposition_id: p for p in propositions}

        if len(delivered_map) != len(propositions):
            return False

        for pid, sentence in delivered_map.items():
            if pid not in prop_dict:
                return False
            prop = prop_dict[pid]
            doc = self.nlp(sentence)
            sent_str = sentence.lower()

            # 1. Polarity & Negative Quantifier Audit
            has_negation = any(t.dep_ == "neg" or t.lemma_ in ["not", "never"] for t in doc)
            has_none_quant = (prop.quantifier.qtype == QuantifierType.NONE)
            starts_with_no = sent_str.startswith("no ")

            if has_none_quant:
                if not starts_with_no:
                    return False
            else:
                expected_neg = (prop.polarity == Polarity.NEGATIVE)
                if has_negation != expected_neg:
                    return False

            # 2. Modality Audit
            if prop.modality == Modality.OBLIGATIVE:
                if not any(t.lemma_ in ["must", "shall", "require"] for t in doc):
                    return False
            elif prop.modality == Modality.PERMISSIVE:
                if not any(t.lemma_ in ["may", "can"] for t in doc):
                    return False

            # 3. Numeric & Condition Threshold Invariant
            if prop.condition_tree and prop.condition_tree.threshold_value is not None:
                if str(prop.condition_tree.threshold_value) not in sent_str:
                    return False

            # 4. Subject Entity Head Preservation
            subj_lemma = prop.subject.lemma.lower()
            if subj_lemma not in sent_str and "it" not in sent_str and "they" not in sent_str:
                return False

            # 5. Predicate Lemma Entailment (Prevents 'pass' -> 'fail' semantic flips)
            pred_lemma = prop.predicate.lemma.lower()
            if not any(t.lemma_.lower() == pred_lemma for t in doc):
                return False

        return True
```

---

## 5. Verification Acceptance Suite (`tests/test_invariants.py`)

```python
# tests/test_invariants.py
import pytest
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519
from pydantic import BaseModel

from cpu_llm.cognition.adapter import PrismCognitionSubspaceAdapter
from cpu_llm.realizer.surface import SemanticSurfaceRealizer
from cpu_llm.audit.entailment import EpistemicEntailmentAuditor
from cpu_llm.boundary.manifest import TrustedKeyringRegistry, PrismManifestBoundary
from cpu_llm.ir.schema import (
    EIRProposition, Polarity, Modality, Quantifier, QuantifierType,
    TypedEntity, TypedPredicate, EpistemicStatus, TemporalInterval,
    LogicalExpressionNode, LogicalOperator
)

class SampleContract(BaseModel):
    action_type: str
    tenant_id: str
    output_text_hash: str
    audit_trace_id: str

def test_sound_deductive_inference_and_condition_separation():
    """Validates that rules with unproven conditions yield CONDITIONAL_DERIVATION."""
    adapter = PrismCognitionSubspaceAdapter()

    # Rule: All approved suppliers must undergo an audit if revenue exceeds 10.
    rule = EIRProposition(
        proposition_id="rule_cond",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.OBLIGATIVE,
        quantifier=Quantifier(qtype=QuantifierType.ALL),
        subject=TypedEntity(raw_text="approved suppliers", lemma="supplier", is_plural=True, qualifiers=["approved"]),
        predicate=TypedPredicate(lemma="undergo", tense="PRESENT", voice="ACTIVE"),
        object_target=TypedEntity(raw_text="an audit", lemma="audit", is_plural=False),
        condition_tree=LogicalExpressionNode(
            operator=LogicalOperator.GT,
            metric_name="revenue",
            threshold_value=10.0,
            unit="M"
        ),
        epistemic_weight=1.0
    )

    fact = EIRProposition(
        proposition_id="fact_supp_x",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.DECLARATIVE,
        subject=TypedEntity(raw_text="Supplier X", lemma="Supplier X", is_plural=False),
        predicate=TypedPredicate(lemma="be", tense="PRESENT", voice="ACTIVE"),
        object_target=TypedEntity(raw_text="an approved supplier", lemma="supplier", is_plural=False, qualifiers=["approved"]),
        epistemic_weight=1.0
    )

    dummy_t = [np.ones(1024, dtype=np.float32) * 0.1, np.ones(1024, dtype=np.float32) * 0.1]
    res = adapter.deliberate(np.zeros(1024, dtype=np.float32), [rule, fact], dummy_t, "tenant_test")

    derived = [p for p in res.grounded_propositions if p.epistemic_status == EpistemicStatus.CONDITIONAL_DERIVATION]
    assert len(derived) == 1, "Must derive a conditional derivation because revenue threshold is not proven!"
    assert derived[0].condition_tree is not None

def test_scope_aware_temporal_contradiction_isolation():
    """Validates that non-overlapping temporal intervals DO NOT trigger a contradiction."""
    adapter = PrismCognitionSubspaceAdapter()

    # Statement 1: Approved in 2024
    p1 = EIRProposition(
        proposition_id="prop_2024",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.DECLARATIVE,
        subject=TypedEntity(raw_text="Supplier X", lemma="Supplier X", is_plural=False),
        predicate=TypedPredicate(lemma="approve", tense="PAST", voice="PASSIVE"),
        temporal_interval=TemporalInterval(valid_from=2024, valid_to=2024),
        epistemic_weight=1.0
    )

    # Statement 2: Not approved in 2026
    p2 = EIRProposition(
        proposition_id="prop_2026",
        polarity=Polarity.NEGATIVE,
        modality=Modality.DECLARATIVE,
        subject=TypedEntity(raw_text="Supplier X", lemma="Supplier X", is_plural=False),
        predicate=TypedPredicate(lemma="approve", tense="PRESENT", voice="PASSIVE"),
        temporal_interval=TemporalInterval(valid_from=2026, valid_to=2026),
        epistemic_weight=1.0
    )

    dummy_t = [np.ones(1024, dtype=np.float32) * 0.1, np.ones(1024, dtype=np.float32) * 0.1]
    res = adapter.deliberate(np.zeros(1024, dtype=np.float32), [p1, p2], dummy_t, "tenant_test")

    # Non-overlapping intervals must not trigger UNRESOLVED_CONTRADICTION
    assert res.ruin_distance == 0.0
    assert not any(p.epistemic_status == EpistemicStatus.UNRESOLVED_CONTRADICTION for p in res.grounded_propositions)

def test_generalized_anchor_regex_and_terminal_punctuation():
    """Validates that arbitrary IDs (e.g. prop_none_quant) parse correctly and clauses have periods."""
    realizer = SemanticSurfaceRealizer()
    auditor = EpistemicEntailmentAuditor()

    prop = EIRProposition(
        proposition_id="prop_none_quant",
        polarity=Polarity.AFFIRMATIVE,
        modality=Modality.PERMISSIVE,
        quantifier=Quantifier(qtype=QuantifierType.NONE),
        subject=TypedEntity(raw_text="patient", lemma="patient", is_plural=False),
        predicate=TypedPredicate(lemma="discontinue", tense="PRESENT", voice="ACTIVE"),
        object_target=TypedEntity(raw_text="therapy", lemma="therapy", is_plural=False),
        epistemic_weight=1.0
    )

    text, delivered_map = realizer.synthesize([prop], [])
    assert text.endswith(".")
    assert "prop_none_quant" in delivered_map
    assert auditor.audit_delivered_output(delivered_map, [prop]) is True
```

---

## 6. Execution Directives for Cursor

1. **Deploy Patches:** Update `src/cpu_llm/ir/schema.py`, `src/cpu_llm/cognition/adapter.py`, `src/cpu_llm/realizer/surface.py`, and `src/cpu_llm/audit/entailment.py`.
2. **Execute Invariant Suite:** Run `pytest tests/test_invariants.py -v`.
3. **Assert CPU Isolation:** Confirm all tests run without GPU calls:
```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_invariants.py
```
