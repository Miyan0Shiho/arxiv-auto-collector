# Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations

**Authors**: Qisong He, Yi Dong, Xiaowei Huang

**Published**: 2026-05-13 22:54:16

**PDF URL**: [https://arxiv.org/pdf/2605.14175v1](https://arxiv.org/pdf/2605.14175v1)

## Abstract
In long conversations, an LLM can produce a next utterance that sounds plausible but rests on premises the conversation has already abandoned. Context-manipulation attacks against deployed agents now actively exploit this gap. We close it with a runtime verifier that maintains an explicit dependency graph: an LLM classifies each turn into one of 8 update operations drawn from four formalisms (dynamic epistemic logic, abductive reasoning, awareness logic, argumentation), and a symbolic engine records which claims depend on which evidence. Checking whether a continuation is supported reduces to a graph walk; retraction propagates through the same graph to flag exactly the conclusions that lose support, with linear per-turn cost and a formal conflict-free guarantee. On LongMemEval-KU oracle (n=78), the verifier reaches 89.7% accuracy vs. 88.5% for the LLM-only baseline (+1.3pp) and 87.2% for a transcript-RAG baseline matched on retrieval budget (+2.6pp); wins among disagreements are correct abstentions where the baseline confabulates. On LoCoMo's 60 official QA items the verifier is competitive with retrieval-augmented baselines. Beyond external benchmarks, we construct two multi-agent scenarios and a 50-item grounding test: on the 15-item stale-premise subset, the verifier reaches 100% accuracy vs. 93.3% (+6.7pp). These instantiate a soundness-faithfulness decomposition: the structural check is sound by construction, and per-deployment LLM extraction faithfulness is the empirical question we measure across four LLM families. The retraction check plateaus at microseconds while history-replay grows linearly with conversation length.

## Full Text


<!-- PDF content starts -->

Grounded Continuation: A Linear-Time Runtime
Verifier for LLM Conversations
Qisong He, Yi Dong, Xiaowei Huang
School of Computer Science and Informatics, University of Liverpool, UK
Abstract
In a long conversation, an LLM can produce a next utterance that sounds plau-
sible but rests on premises the conversation has already abandoned. Context-
manipulation attacks against deployed agents now actively exploit this gap. We
close it with a runtime verifier that maintains an explicit dependency graph. As
the conversation unfolds, an LLM classifies each turn into one of8update opera-
tions drawn from four formalisms (dynamic epistemic logic, abductive reasoning,
awareness logic, argumentation). A symbolic engine then records which claims
depend on which evidence and earlier reasoning. At any point, “Is this continua-
tion supported by what has been said?” reduces to a graph walk. Retraction propa-
gates through the same graph to flag exactly the conclusions that lose support, with
linear per-turn cost and a formal conflict-free guarantee. On the benchmarks we
evaluate, the verifier matches or exceeds strong baselines: on LongMemEval-KU
oracle (knowledge update with supersession,n= 78), it reaches89.7%accuracy
against88.5%for the LLM-only baseline (+1.3pp) and87.2%for a transcript-
RAG baseline matched on retrieval budget and content access (+2.6pp). The ver-
ifier wins among disagreements are correct abstentions where the baseline con-
fabulates. On LoCoMo’s60official QA items the verifier is competitive with
retrieval-augmented baselines, consistent with its interactional-grounding focus.
Beyond external benchmarks, we construct two multi-agent scenarios and a50-
item grounding test for controlled evidence: on the15-itemstale-premisesubset
(premise retracted earlier in the conversation), the verifier reaches100%accuracy
vs.93.3%(+6.7pp). Together, these controlled tests instantiate the soundness–
faithfulness decomposition: the structural check is sound by construction, and
per-deployment LLM extraction faithfulness is the empirical question we measure
across four LLM families. The graph-walk retraction check plateaus at microsec-
onds in the bounded regime while history-replay grows linearly with conversation
length.
1 Introduction
Large language models produce continuations that can be locally fluent and pragmatically plausible
yetungrounded: disconnected from the claims, observations, and revisions that the conversation
has actually established. The deployment question is concrete: given the next LLM output, can
we check,at this turnand within a tight latency budget, whether that output traces back to those
prior commitments? When an LLM cannot answer “why did we reject approach A?” or “what as-
sumptions does our current decision rest on?”, the relevant utterances are inside the context window.
What is missing is a maintained structure connecting the LLM’s continuation to the conversation’s
prior commitments.
This problem is empirically severe. Laban et al. [2026] found a 39% average performance drop in
multi-turn versus single-turn settings across 200,000+ simulated conversations, with LLMs failing
to revise incorrect assumptions even when later turns contradict them: “when LLMs take a wrong
Preprint.arXiv:2605.14175v1  [cs.AI]  13 May 2026

turn in a conversation, they get lost and do not recover.” Shaikh et al. [2025] show, on real human-
LLM dialogues from WildChat [Zhao et al., 2024], MultiWOZ [Budzianowski et al., 2018], and
Bing Chat [Kelly et al., 2023], that LLMs initiate clarification three times less often than human
partners and that early grounding failures predict later interaction breakdowns.
This gap is also actively exploited: context-manipulation attacks against deployed agents [Patlan
et al., 2025, Dong et al., 2025] produce continuations that are locally consistent but disconnected
from the conversation’s prior commitments, triggering unauthorised actions because no runtime
mechanism ties the agent’s continuation to claims established earlier in the conversation.
Retrieval-augmented attribution methods addressexternalgrounding: a generated claim is accepted
if it can be traced to a cited document [Gao et al., 2023, Bohnet et al., 2022]. Long-horizon LLM
conversations are dominated byinteractionalgrounding instead (a continuation is acceptable if it is
consistent with claims, observations, hypotheses, and revisions earlier in this same conversation),
and conversational memory systems [Chhikara et al., 2025, Rasmussen et al., 2025, Zhang et al.,
2024] trackwhat was saidbut not the dependency structure connecting what was said to what was
concluded. We close this dependency-tracking gap with a runtime verifier: an LLM Interpreter
classifies each utterance into one of 8 operations updating a symbolic engine that maintains a de-
pendency map. The runtime check at any point asks whether a candidate continuation is reachable
from the current structure and what upstream commitments it depends on, computable in time linear
in the engine’s representation size (Proposition 2.2). The LLM handles natural-language under-
standing (noisy but learnable). The engine handles dependency tracking (sound by construction,
Proposition 2.1).
Contributions. (1) A runtime verifier for interactional grounding: at every turn, in time lin-
ear in the engine’s representation, the verifier checks whether a candidate continuation traces back
through a maintained dependency structure to the conversation’s prior commitments (Sections 2
and 4).(2) Empirical wins where interactional grounding matters, scoping where it does not:
against a matched transcript-RAG baseline, on LongMemEval-KU oracle (n= 78) the verifier ex-
ceeds both the LLM-only baseline (+1.3pp) and the transcript-RAG baseline (+2.6pp, Table 5),
with wins concentrated in correct-abstention cases where the baseline confabulates. On LoCoMo
it is competitive with retrieval-augmented baselines, consistent with its interactional-grounding fo-
cus (Table 4).(3) A composable formal substrate with a soundness guarantee: each utterance
produces a single well-typed updateApply(op,args,D t)over an epistemic plausibility model, argu-
mentation framework, commitment record, and dependency map (Algorithm 1), with a conflict-free
guarantee for selective retraction (Proposition 2.1).(4) A soundness–faithfulness decomposition
that splits the verifier into a sound structural check and per-deployment extraction faithfulness, ex-
posing the canonical stale-claim case where the verifier catches advice both the LLM-only and
matched transcript-RAG baselines miss (+6.7pp on the stale-premise subset, Table 3).
Scope.We evaluate on two authored multi-agent scenarios (Section 3), a 50-item Phase 2 direct
grounding test, 78 LongMemEval-KUoracleitems [Wu et al., 2025], and 60 official LoCoMo QA
items across three multi-session conversations [Maharana et al., 2024]. The verifier was designed for
interactional grounding. With content-bearing rendering and retrieval, the framework also extends
to entity-relation factual QA (Section 4.2).
2 The Runtime Verifier
Notation.We use the standard modal operators of dynamic epistemic logic (DEL):K iφ(agenti
knowsφ),B iφ(believes),A iφ(is aware of), andC Gφ(common knowledge in groupG). Their
semantics over an epistemic plausibility model (Definition E.1) and an awareness structure (Defini-
tion E.3) are recalled in Section E.
The verifier itself exposes a single graph query, but the structure that supports it is layered: an
epistemic model (what is known), an argumentation framework (what attacks what), commitment
records (who said what publicly), and the dependency map proper. Definition 2.1 bundles the four
into one object. Runtime queries touch only the argumentation skeleton and the dependency map.
Definition 2.1(Dependency structure).Adependency structureat turntis a tupleD t=
(Mt,AF t,Cm t,Dept)whereM tis an epistemic plausibility model (Definition E.1. Baltag and
Smets 2008) recording what each agent knows, believes, or has hypothesised.AF t= (Argst,Att t)
2

is a Dung-style argumentation framework with attack relationAtt t⊆Argst×Argst[Dung, 1995]
(each argumentαcarries a claimclaim(α)∈Propover the conversation’s propositionsProp, and
is itself the unit of support).Cm t:Ags→ P(Argst)records each agent’s public commitments
[Walton et al., 2008], whereAgsis the set of agents in the conversation.Dept:Argst→ P(Prop)
maps each argument to the propositions supporting it.
The full four-formalism foundation underlyingD tis in Section E; the runtime check below queries
onlyDeptand the argumentation skeleton.
The runtime check.Given a candidate continuationcasserting propositionϕ c, the verifier checks
whetherchas the property of beinggroundedwith respect to the current dependency structure:
Definition 2.2(Grounded continuation).A candidate continuationcasserting propositionϕ cis
groundedwith respect toD tiff there exists an argumentα c∈Argstwithclaim(α c) =ϕ c, otherwise
it isungrounded.
At any turnt, the verifier returns:
Verify(c,D t) =⟨grounded,Dep(α c)⟩if∃α c∈Argstwithclaim(α c) =ϕ c
⟨ungrounded,∅⟩otherwise.
A grounded continuation comes with the set of upstream commitments it depends on. An un-
grounded continuation is flagged for retry, retraction, or human review. Two derived queries support
belief revision:Affected(p)identifies the conclusions that lose grounding whenpis retracted (for-
mal definition in Proposition 2.1, restricted to the current preferred extension), andDep(α)returns
the propositions an argumentαdepends on. Both reduce to dependency-graph reachability. The
pair(Argst,Dept)is a labelled claim-dependency structure derived from interaction history. Its
soundness is established by Proposition 2.1 and its extraction-time faithfulness is the empirical ques-
tion Section 4 addresses. Definition 2.2 is binary by design. Graded variants (partial dependency,
weighted attack relations, confidence-calibrated grounding) are tractable extensions.
Why this is non-trivial.The lookup form of Definition 2.2 is misleading.Argstis not the set of
mentioned propositions: it is a preferred extension of an argumentation framework whose attack
relation is updated each turn, populated by hypotheses generated through abduction (Definition E.2)
over an awareness structure that the conversation expands. The work is upstream of the check.
Maintenance: each turn updates the epistemic plausibility model, awareness set, attack relation,
and dependency map jointly. One misclassification poisons all four.Retraction: identifying which
conclusions lose grounding whenpis retracted requires the explicitDepmap to be maintained
as part of the structure rather than derived post-hoc. This is what Proposition 2.1’s conflict-free
guarantee rests on.Complexity: the combined formalisms are PSPACE-hard in general (DEL model-
checking. Aucher and Schwarzentruber 2013), so any polynomial-time procedure (a transformer’s
forward pass included) must approximate. The verifier sidesteps this by maintaining(Argst,Dept)
incrementally so each check isO(|Argst|+|Att t|)under the structural restrictions our scenarios
satisfy (Proposition 2.2).
UpdatingD t.An LLM Interpreter classifies each utterance into one of 8 operations. Algorithm 1
composes the operation into a single updateD t7→ D t+1that simultaneously refinesM,AF,Cm,
andDep. The engine checks per-operation preconditions (Table 8) and re-prompts on failure. Sur-
prising observations enqueue abductive problems (Definition E.2) that drive the next HYPOTHESIZE.
The 8 operations split into three roles: OBSERVE/RESOLVEcommit content; HYPOTHE-
SIZE/SUPPORT/UNDERMINE/REVISEadjust plausibility or attack relations without erasing prove-
nance; EXPAND-AWARENESS/QUESTIONexpand or query without committing claims. The taxon-
omy is the closure of the four formalisms’ update primitives under the typing requirement that each
utterance produces exactly one well-typed update (Section E). The verifier itself queries onlyDept
and the argumentation skeleton at runtime.
3

The 8 operations and their DEL realisations(notation: Section E).
Operation DEL realisation Example
OBSERVE[!ψ](hard public announcement) “I see 401 errors.”
HYPOTHESIZE[⇑γ](soft lexicographic upgrade) “Could be a retry loop.”
SUPPORT[↑γ], or[⇑γ]ifγ-specific evidence “Timing is consistent.”
UNDERMINEfamily of plausibility downgrades ofγ(Section E) “Wrong error code, 401, not 503.”
REVISE[!¬γ]orAttedge addition “It’s reversed: A causes B.”
EXPAND-
AWARENESSAi← A i∪ {p};W-refinement onp(Definition E.3) “The DB alert is really Redis.”
RESOLVEconsensual[!γ], or authoritative commitment with dissent
recorded“Confirmed: bug is line 42.” /
“We’ll go with Yjs.”
QUESTIONno DEL update; add(B i, χ)to abductive-problem queue “Why is traffic 3×normal?”
Soundness of selective retraction.The central guarantee is thatAffected(p)identifies exactly the
conclusions that lose grounding whenpis retracted, with the post-retraction state well-defined under
the standard Dung [1995] preferred-extension semantics.
Proposition 2.1(Conflict-free selective retraction).Let(M,AF,Cm,Dep)be a dependency
structure withS⊆Argsa preferred extension ofAF. Supposep∈Propis retracted. Let
Affected(p) :={α∈S:p∈Dep(α)},S′:=S\Affected(p), andAF′:= (Args\
Affected(p),Att∩(Args\Affected(p))2). ThenS′is conflict-free inAF′, and there exists a
preferred extension ofAF′containingS′. Any such extension is a valid post-retraction state and
preserves everyα∈Swithp /∈Dep(α).
The proposition guarantees conflict-freeness and existence, not admissibility ofS′itself or unique-
ness of the post-retraction extension. The engine treatsS′as a lower bound and recomputes preferred
extensions onAF′, flagging any argument whose status changes. Proof in Section E.
Per-turn tractability.The runtime checkVerify(c,D t)is dependency-graph reachability, com-
putable in time linear in|Argst|+|Att t|(no LLM call required). The engine update step requires
bounded model-checking onD t−1:
Proposition 2.2(Tractability under structural restrictions).Under the structural restrictions our
scenarios satisfy, namely small number of agentsk(k≤12), acyclic attack graph, and explicit
world representation, per-turn computation is polynomial in|W|and|Args|:O(|W|)for hard an-
nouncements,O(|W|log|W|)per soft upgrade, andO(|Args|+|Att|)for extension recomputation
[Dung, 1995, Dunne, 2007]. Reference implementation:<0.1ms per turn on Phase 2. Worst cases
without these restrictions reachPSPACE[Aucher and Schwarzentruber, 2013] andΠP
2[Dunne and
Wooldridge, 2009]. Per-phase analysis in Section E. Acyclicity is inherited from Issue-Based Infor-
mation System (IBIS)-style [Kunz and Rittel, 1970] deliberation, where each new argument either
supports, attacks, or refines an existing one rather than cycling back.
3 Validation Scenarios
We validate the verifier on two authored multi-agent scenarios.Phase 2(13 turns, Section C) ex-
ercises hypothesis lifecycle tracking and counterfactual retraction queries: when an observation is
invalidated, which conclusions lose grounding?Phase 3(19 turns, Section D) exercises commit-
ment tracking and selective revision under assumption change: when an upstream assumption flips,
which decisions need re-grounding and which remain intact? A thirdPhase 1(muddy-children cal-
ibration, Section B) tests the underlying world-set machinery against fully-specified ground truth
and is presented in the appendix only.
Phase 2: Naturalistic debugging.Three engineers debug a cascading failure (Section C). Ground
truth: token bug→retry storm→Redis exhaustion→rate-limit bypass→Stripe 429s, plus a
monitoring miscategorisation. The conversation generates four hypotheses with non-trivial lifecy-
cles (h 2: Redis→Auth, undermined at T7 by error-code evidence and abandoned T9.h 4at T12
reverses the direction, subsumingh 1andh 3at T13). The verifier is queried with three retractions
(Affected(o 8),Affected(o 9),Affected(o 6)). Correctness depends on whether the dependency map
recordswhyh 2was abandoned and preserves the provenance ofh 4’s causal reversal: both are struc-
tural facts that flat history cannot provide.
4

Conversation
(natural language)LLM
InterpreterSymbolic
Engine
Context
RendererLLM Context
Window“it’s reversed:
Auth→Redis”HYPOTHESIZE
+ REVISE
addh 4;
flip causal arrow;
revise plausibility
“h4now active”informs
Figure 1: Hybrid architecture, illustrated with Phase 2 T12 (an engineer reverses the causal direction;
details in Section C). Solid: utterance flow, classified operations, engine update, summary. Dashed:
rendered summary fed back into the LLM’s context.
Phase 3: Open-ended deliberation.A team chooses a real-time collaboration architecture (Sec-
tion D). The outcome is adecision with dissent: Alice commits to Yjs server-relayed under three
assumptions (a 1docs stay short;a 2editing remains burst-mode;a 3Q2 long-running documents
unconfirmed); Bob records dissent. Whena 3later flips,Affected(a 3)must flag exactly the Yjs de-
cision while leaving the access-control resolution intact: the selective re-grounding capability that
motivates the verifier.
4 Verifier Implementation and Evaluation
4.1 Architecture
LLM Interpreter and Symbolic Engine.For each utterancec t, the Interpreter performs (1)op-
eration classificationinto one of the 8 operations of Section 2 and (2)proposition extractionof
the relevant formal content. The Engine drives the loop, keeping the formal guarantees on the
symbolic side: it flags surprising observations (M, w̸|=B iχ) prompting the Interpreter for a hy-
pothesis, verifies each candidate against per-operation preconditions (Section E and definition E.2),
and either applies the formal update or re-prompts naming the failing condition (e.g., “use EXPAND-
AWARENESS, not OBSERVE”). Temporal epistemic model checkers (e.g. MCMAS, Lomuscio et al.
2017) or bounded model checking [Huang et al., 2011] can serve as backends. Worked extractions
in Sections C and D.
Context Renderer.After each engine update, the Renderer summarises the changed parts ofD tin
natural language and writes the summary into the LLM’s context window (dashed arrow in Figure 1);
the next classification step sees the engine’s structured state rather than raw conversation history.
The Renderer’s output is also what users see when the verifier flags an ungrounded continuation:
“decisionαno longer grounded; affected by retraction ofa 3; depends ona 1, a2.”
4.2 Evaluation
We measure (i) per-utterance classification accuracy, (ii) retraction-query latency at conversation
lengthsKup to 2000 turns, (iii) end-to-end dependency extraction (E1/E1b/E1c), (iv) direct ver-
ifier accuracy on a labelled grounding test set (E2), (v) noise robustness (E5), and (vi) external-
benchmark behaviour on LoCoMo [Maharana et al., 2024] and LongMemEval-KU [Wu et al., 2025].
Scenarios are scored on Phase 2 (13 turns) and Phase 3 (19 turns). We report multi-label F1 with ex-
tra weight on the 3–4key shiftsper scenario (hypothesis abandonment, causal reversal, decision with
dissent). Three prompt conditions on Phase 2:Minimal,Definitions,State-augmented. Four LLMs:
Claude Sonnet 4, GPT-4o, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (last two via vLLM), 5 runs
per cell,temperature=0 then 1.0. Phase 3 uses Definitions only.
Classification.Definitions alone lift Phase 2 F1 from0.66to0.85(+0.19) and key-shift F1 from
0.50to0.74(+0.24): the taxonomy is learnable from general definitions. State-augmented adds
+0.06overall and+0.09on key shifts. EXPAND-AWARENESS(which requires recognition that a
proposition was previouslyunconceived) jumps from0%exact (conditions 1–2) to100%(condition
3), confirming that engine state enables distinctions the LLM cannot make from text alone. Ab-
solute F1 scales with model size (Table 1). Phase 3 reaches0.85overall and0.92on key shifts,
demonstrating cross-scenario generalisation.
5

Table 1: Phase 2 classification accuracy across models and prompt conditions (5 runs per cell). Each
cell reports F1 / Exact-match%. Phase 3 results (Claude0.85overall /0.92key shifts) in Table 7.
Minimal Definitions State-aug.
F1 Exact F1 Exact F1 Exact
Phase 2 (debugging): Overall (13 turns)
Claude Sonnet 4 0.66 11% 0.85 54%0.91 63%
GPT-4o 0.69 31% 0.74 40% 0.81 52%
Qwen2.5-7B-Instruct 0.54 29% 0.57 26% 0.59 29%
Llama-3.1-8B-Instruct 0.54 12% 0.55 20% 0.65 32%
Phase 2: Key shifts (4 turns)
Claude Sonnet 4 0.50 5% 0.74 25%0.8330%
GPT-4o 0.43 0% 0.50 10% 0.7450%
Qwen2.5-7B-Instruct 0.39 10% 0.47 0% 0.47 0%
Llama-3.1-8B-Instruct 0.37 0% 0.47 20% 0.56 15%
102103
Conversation length K (turns)100101102Wall time per query (s)
Verifier (naive, |Args|0.3K)
Verifier (bounded, |Args|50)
Baseline (re-read history)
Figure 2: Retraction latency scales with|Argst|, notK. Verifier (blue): bounded (|Argst| ≤50,
hollow) plateaus once the cap saturates (K≥200); naive (|Argst|∼0.3K, solid) grows shallowly.
Baseline (red dashed): linear walk ofK-turn history. Median over 5 random seeds (each generating
a fresh synthetic engine state)×200 queries per seed.
Retraction latency scaling.For deployment, the value of the maintained graph is that it replaces
history replay with a bounded graph walk: re-reading theK-turn transcript at every retraction does
not scale, butVerifydoes. We measureAffectedon synthetic engine states atK∈ {13, . . . ,2000}
turns under two regimes (Figure 2):naive(|Argst| ∼0.3K) andbounded(|Argst| ≤50, modelling
Phase-2-like working sets where resolved/abandoned hypotheses retire). In the bounded regime the
verifier plateaus while history-replay grows linearly withK, an84×gap atK= 2000(Figure 2).
The naive regime shows a3–10×advantage. This is the empirical realisation of Proposition 2.2: the
per-query cost is bounded by|Argst|, not the conversation length.
End-to-end verification.The verifier’s correctness decomposes into two claims.Soundness
given a faithfulD t(Proposition 2.1): with ground-truth dependencies,Verifyresolves12/12af-
fected/unaffected decisions correctly (3 retraction queries×4 hypotheses).End-to-end with LLM-
extracted dependencies: GPT-4o reaches Dep precision1.0but recovers3/7ground-truth tuples.
The bottleneck concentrates on a single cross-hypothesis linkh 1→h 4, not onVerifyitself. As
Table 2 shows, this gap isschema-deep, notprompt-deep: directly-prompted GPT-4o tops out at
F1 = 0.34vs. pipeline0.60(E1), and prompt-schema variants reach F1= 0.50but never recover
h1→h 4(E1b, Section I). The formal construction makes the fix visible: extend RESOLVEto up-
date existingDeptuples (Algorithm 1, RESOLVEcase). A targeted probe (E1c, Section J) achieves
perfecth 1→h 4recovery with zero false positives, validated by a Phase-3 negative sanity probe.
Direct verifier evaluation.The headline finding:Verifycatches stale-premise advice the base-
line misses, and abstains rather than confabulates on under-supported queries. We instrument
Verify(c,D t)against an LLM-only baseline on a 50-item Phase 2 test set across four categories:
15actualcontinuations (grounded), 15staleclaims (re-assertions after T9 abandonment), 10cross-
conversationnegatives, and 10counterfactuals.VerifywalksD tfrom the candidate’s premise. The
baseline sees the same transcript and returns one-word grounded/ungrounded. Author labels were
checked by an independent annotator on a blinded 20-item subset: Cohen’sκ= 0.733, above the
substantial-agreement threshold.
6

Table 2: Soundness–faithfulness decomposition on Phase 2.GT depsrow isolates the verifier’s
structural check.E1rows score directly-prompted GPT-4o against ground-truthDeptuples (no
pipeline).Pipelinerows score the pipeline at three model scales. All against post-T13 GT. E1b
prompt-schema ablation and E1c RESOLVE-stage extension in Sections I and J. Bold: per-column
best among LLM-extracted rows. “—”: column does not apply.
LLM hyps GT hyps Dep Dep Dep Affected
Methodextracted matched prec. recall F1 accuracy
GT deps (soundness) — 4/4 1.00 1.00 1.00 12/12
LLM-prompted (no pipeline; GPT-4o; post-T13 GT), E1
zero-shot — 4/4 0.27 0.46 0.31 —
few-shot — 4/4 0.26 0.54 0.34 —
chain-of-thought — 4/4 0.23 0.46 0.29 —
self-consistency-5 — 4/4 0.22 0.46 0.28 —
Pipeline (verifier extraction; post-T13 GT)
Qwen2.5-7B-Instruct 8 4/4 0.25 0.14 0.18 0/3
Llama-3.1-8B-Instruct 4 2/41.000.14 0.25 0/3
GPT-4o 6 4/41.00 0.43 0.60 1/3
Table 3: Direct verifier evaluation (E2). Per-category accuracy ofVerify(c,D t)vs. LLM-only base-
line and two transcript-RAG variants on the 50-item Phase 2 test set.TR (k=5): selective retrieval
matched to dep-walk depth.TR (k=20): full-window. Cohen’sκ= 0.733on the 20-item indepen-
dent overlap.
CategorynVerifier LLM-only TR (k=5) TR (k=20)
Actual 1514/15 15/15 15/15 15/15
Stale 1515/1514/15 14/15 15/15
Cross-conversation 1010/10 10/10 10/10 10/10
Counterfactual 1010/10 10/10 10/10 10/10
Stale + counterfactual (pooled) 2525/2524/25 24/25 25/25
Table 3 reports the result.Verifycatches all three ungrounded categories perfectly (35/35pooled),
including the canonical case e2_030 att=13(“Auth should switch to in-process JWT validation to
remove its Redis dependency for sessions”): an action recommendation whose load-bearing premise
is exactly the abandonedh 2.Verifywalks the dependency chain, finds the LLM-aligned hypothesis
withstatus = abandoned, and returns ungrounded. The baseline reads it as plausible engineering
advice without checking whether its premise stands. This is the textbook stale-claim case for the
soundness–faithfulness decomposition. The baseline matches on cross-conversation and counterfac-
tual (topic shift and explicit contradiction are surface-detectable) and misses only e2_030 on stale.
The verifier’s single loss (e2_015) is a documented conservative case: when no single entity inD t
underlies the claim (asserts_id=∅, multi-entity meta-reasoning),Verifyreturns ungrounded by
design. The pooled stale+counterfactual headline is a near-ceiling tie (LLM-only baseline at96%
pooled). The architectural signal is sharper at the category level:+6.7pp on stale claims, the e2_030
canonical case, and the abstention behaviour the baseline lacks. A transcript-RAG baseline matched
on top-k= 5over the same Phase 2 turns gives24/25pooled (the same e2_030 miss as the LLM-
only baseline), preserving the verifier’s+4.0pp pooled /+6.7pp stale advantage at matched selective
retrieval. A full-window variant (top-k≥13, all available turns plus per-turn similarity-score anno-
tations) lifts the baseline to25/25pooled, matching the verifier. The verifier’s E2 advantage is
therefore best characterised asselective retrieval with a structural soundness guarantee: dep-walk
is structurally guaranteed to surface the abandonment evidence at any retrieval budget, whereas
similarity-based retrieval is matched only at full-window with similarity-score framing.
Robustness to extraction noise (diagnostic).We perturb the GT Phase 2 state on the 50-item E2
set (per-item truncation att, 10 seeds at eachε∈[0,0.8]) under three independent noise mod-
els (Figure 4, Section K).Random dependency-edge dropandaddare flat at49/50across allε:
88%of items decide beforewalk_depsruns (via entity-resolution, hypothesis-status, or observa-
tion/awareness leaves), characterising the E2 set as low-sensitivity to edge-traversal accuracy and
locatingVerify’s win mechanism elsewhere.Lifecycle/status corruptionproduces the meaningful
curve: pooled accuracy0.98→0.64asε: 0→0.8, with stale-claim accuracy (the architectural-win
category in Table 3) falling1.00→0.27. The verifier matches the LLM-only pooled-50 baseline
(0.98) atε≈0.05. Observed boundary errors on existing Phase 2 pipeline runs concentrate onh 2’s
7

Table 4: LoCoMo official 60-item QA: rendering-mode ablation and transcript-RAG baseline (GPT-
4o).dep-map only: opaque IDs{h i: [oj, ok], . . .};+content: hypothesis/observation content with
session-date attribution;+content+retrieval: top-k= 20RAG over engine items;transcript-RAG:
same setup, indexed over per-turn transcript chunks (no engine state). Per-category F1 in Section G.
Configuration Pooled F1∆vs. baseline (pp)
LLM-only baseline0.255—
Verifier (dep-map only, published)0.153−10.2
Verifier (+content)0.427 +17.2
Verifier (+content+retrieval)0.440 +18.5
Transcript-RAG baseline (no engine)0.446+19.1
Table 5: LongMemEval-KU oracle 78-item evaluation (GPT-4o).transcript-RAG: matched encoder
and top-k= 20, indexed over per-turn transcript chunks (no engine state). The verifier (+content
+retrieval) is the only configuration to exceed both baselines; per-category breakdown in Section H.
Configuration Correct / 78 Acc.
LLM-only baseline69/78 88.5%
Verifier (dep-map summary, no retrieval)68/78 87.2%
Transcript-RAG baseline (no engine)68/78 87.2%
Verifier (+content+retrieval)70/78 89.7%
abandonment (∼5%aggregate). The bottleneck forVerify’s empirical win mechanism is therefore
faithful REVISE/RESOLVE/abandonment extraction, not ordinary dep-edge recall.
Operational envelope and external benchmarks.The verifier’s load-bearing capability is the de-
pendency graph: a record of which claims rest on which observations, hypotheses, and earlier con-
clusions, with a notion ofcurrent standingthat distinguishes “mentioned earlier” from “established
and not retracted.” Three structural features motivate the design:(a)a claim’s validity depends on
the standing of an earlier claim, not just its mention (a continuation under abandonedh 2is wrong
even thoughh 2remains in the transcript);(b)the right answer is sometimes “insufficient support,”
so the verifier should abstain rather than confabulate;(c)retraction has selective downstream conse-
quences.LoCoMo[Maharana et al., 2024] (entity-relation recall) andLongMemEval-KUatoracle
[Wu et al., 2025] (knowledge update with supersession) sit at different points on this envelope:
LongMemEval-KU is closer to feature(a)than LoCoMo.
Rendering and retrieval ablation.The deployment-time choice is the engine-state rendering:
the published cached run injected only the dependency-map JSON ({h 1: [o3, o5], . . .}, opaque
IDs without content), forcing the QA model into abstention when the truncated transcript also
lost the relevant fact. Table 4 reports a three-mode ablation on LoCoMo: dep-map-only loses
−10.2pp; content-bearing rendering (hypothesis/observation content + per-item turn-date attribu-
tion) reaches+17.2pp; content + RAG retrieval over engine items reaches+18.5pp. The same
modes on LongMemEval-KU yield∆ =−1.3,0,+1.3pp (87.2%→88.5%→89.7%accuracy on
78oracleitems); content+retrieval is the first configuration to exceed the LLM-only baseline. To
isolate the engine’s contribution beyond retrieval, we add a matched transcript-RAG baseline (same
encoder, same top-k=20, indexed over per-turn transcript chunks): on LoCoMo it matches the veri-
fier (Table 4), consistent with the interactional-grounding focus and feature(a). On LongMemEval-
KU it falls below the LLM-only baseline, leaving the verifier the only configuration to exceed both
(Table 5), since lifecycle/status labels disambiguate current from superseded values where similar-
ity surfaces both. Atn= 78the directional deltas are not statistically significant (McNemar exact
p=0.625on3vs.1discordant pairs for verifier vs. transcript-RAG, and Wilson95%CIs overlap),
and on LoCoMo the verifier and transcript-RAG are statistically indistinguishable (5vs.6discor-
dant pairs,p≈1.0). The formal framework (lifecycle, dependency tracking, retraction soundness) is
unchanged across modes. Rendering and retrieval are deployment-time tunables. The architectural
signal that LongMemEval-KU was designed to surface (correct abstention via feature(b)) survives
the headline win:3of4verifier wins among the original9disagreements were correct abstentions on
_absitems where the long-context baseline confabulated, the LongMemEval analogue of e2_030.
Per-category tables and the Qwen-7B 16K truncation analysis are in Sections G and H.
8

Reproducibility and deployment scope.All prompts, conversations, annotations, and evaluation
code are in a single runnable script (Section M; deployment-time error composition in Section F).
Claude Sonnet 4 reachesF1 = 0.91classification (Table 1) and the structural check is correct on
12/12retraction queries given a faithfulD t(Table 2); the remaining gap localises to extraction
recall, not the verifier.
5 Related Work
Retrieval-augmented attribution.ALCE [Gao et al., 2023], AutoAIS [Bohnet et al., 2022], and
MTRAG [Katsis et al., 2025] evaluate whether a generated claim is supported by a citedexternal
source. The verifier targets the dual problem ofinteractionalgrounding, where supporting evidence
is the conversation’s own earlier turns. The two paradigms are complementary.Claim traceability
and verifiable grounding.VISTA [Zhang et al., 2026b] parses a conversation post-hoc into a Rea-
soning Dependency Tree for offline visualisation. The runtime verifier maintains a comparable struc-
ture incrementally and queries it under a tight latency budget. Context- and memory-manipulation
attacks against deployed agents [Patlan et al., 2025, Dong et al., 2025] succeed precisely because
no runtime structural binding ties an agent’s continuation to its actual interaction history. Defini-
tion 2.2 and Proposition 2.1 together specify the structural property such defences presuppose, with
(Argst,Dept)realising it as a queryable substrate. AgentArmor [Wang et al., 2025] is the clos-
est systems-architecture neighbour, building a Program Dependence Graph over an LLM agent’s
runtime tool-call trace to enforce security policies. That PDG and our claim-dependency map are
complementary abstractions over different aspects of agent behaviour.Memory and state track-
ing.Recent memory-mechanism work (see Zhang et al. [2024] for a survey) clusters into three
families: environment-state tracking [Park et al., 2023, Chen et al., 2023, Tang et al., 2024, Zhou
et al., 2025], belief-state tracking [Kim et al., 2025], and token-efficient fact compression [Chhikara
et al., 2025]. None maintains the dependency structure thatVerifyandAffectedrequire.Formal
foundations.DEL and awareness logic [Fagin et al., 1995, van Ditmarsch et al., 2007, van Benthem,
2007, Baltag et al., 2014, Fagin and Halpern, 1988, Velázquez-Quesada et al., 2013, Doutre et al.,
2014], model checking [Lomuscio et al., 2017, van der Meyden and Su, 2004], IBIS argumentation
[Kunz and Rittel, 1970, Conklin, 2005], and context engineering [Anthropic, 2025, Zhang et al.,
2026a] all underpin pieces of the verifier. We provide formal semantics for runtime grounding in
this combination.
Closest related work.D-SMART [Lei et al., 2025] incrementally constructs an OWL knowledge
graph and reasons over it via tree search. The verifier differs in object: D-SMART tracks entity-
relation triples (“what was asserted”) while the verifier tracks how conclusions are reached: hy-
pothesis lifecycles, decisions with dissent, claim-to-evidence chains. The Phase 2 illustration in
Section C (causal-reversalh 2→h 4tracked through dependency reversal) lies outside D-SMART’s
representational scope. Zep [Rasmussen et al., 2025] adds temporal validity but not assumption de-
pendencies. DEL for Dialogue Friction [Obiso et al., 2025] applies the same formal tools to measure
cognitive friction. Joint Human-AI Reasoning [Bezou-Vrakatseli et al., 2024]prescribesargumenta-
tion protocols for human-LLM inquiry. The verifierchecksgrounding of naturalistic conversations.
Full feature comparison in Section A.
6 Discussion and Conclusion
The contribution is a composable formal substrate (Algorithm 1) answering a deployment question
existing methods do not:is the next LLM output grounded in the conversation’s earlier commit-
ments?Propositions 2.1 and 2.2 establish soundness of selective retraction and per-turn tractability.
Threat model.The verifier provides the structural mechanism context-manipulation defences [Pat-
lan et al., 2025, Dong et al., 2025] presuppose: claims unsupported byD tare caught. Corrupted
Interpreter inputs are not. Defences against the latter (signed-trace integrity, compositional bounds)
are the natural extension.
Limitations and future work.Classification F1 degrades with smaller models (Table 1). The
engine catches type but not content errors. LLM API latency dominates end-to-end (Section F).
Future work: REVISE-targeted extraction and human utility study.
9

References
Anthropic. Effective context engineering for AI agents. Anthropic Engineer-
ing Blog, September 2025.https://www.anthropic.com/engineering/
effective-context-engineering-for-ai-agents.
Guillaume Aucher and François Schwarzentruber. On the complexity of dynamic epistemic logic.
InProceedings of the 14th Conference on Theoretical Aspects of Rationality and Knowledge
(TARK 2013), 2013.
Alexandru Baltag and Sonja Smets. A qualitative theory of dynamic interactive belief revision.
In Giacomo Bonanno, Wiebe van der Hoek, and Michael Wooldridge, editors,Logic and the
Foundations of Game and Decision Theory (LOFT 7), volume 3 ofTexts in Logic and Games,
pages 9–58. Amsterdam University Press, 2008.
Alexandru Baltag, Lawrence S. Moss, and Slawomir Solecki. The logic of public announcements,
common knowledge, and private suspicions. InProceedings of the 7th Conference on Theoretical
Aspects of Rationality and Knowledge (TARK 98), pages 43–56. Morgan Kaufmann, 1998.
Alexandru Baltag, Virginie Fiutek, and Sonja Smets. DDL as an “internalization” of dynamic belief
revision. In Robert Trypuz, editor,Krister Segerberg on Logic of Action, Outstanding Contribu-
tions to Logic, pages 253–280. Springer, 2014.
Elfia Bezou-Vrakatseli, Oana Cocarascu, and Sanjay Modgil. Towards dialogues for joint human–AI
reasoning and value alignment, 2024. Argumentation protocols for human–LLM inquiry.
Bernd Bohnet, Vinh Q. Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Jacob
Eisenstein, Kuzman Ganchev, Jonathan Herzig, Kai Hui, et al. Attributed question answering:
Evaluation and modeling for attributed large language models.arXiv preprint arXiv:2212.08037,
2022.
Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman
Ramadan, and Milica Gasic. Multiwoz-a large-scale multi-domain wizard-of-oz dataset for task-
oriented dialogue modelling. InProceedings of the 2018 conference on empirical methods in
natural language processing, pages 5016–5026, 2018.
Siwei Chen, Anxing Xiao, and David Hsu. LLM-State: Open world state representation for long-
horizon task planning with large language model.arXiv preprint arXiv:2311.17406, 2023.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building
production-ready AI agents with scalable long-term memory. InProceedings of the 28th European
Conference on Artificial Intelligence (ECAI), pages 2993–3000, 2025. doi: 10.3233/FAIA251160.
Jeff Conklin.Dialogue Mapping: Building Shared Understanding of Wicked Problems. Wiley, 2005.
Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, and Zhen Xi-
ang. A practical memory injection attack against LLM agents.arXiv preprint arXiv:2503.03704,
2025.
Sylvie Doutre, Andreas Herzig, and Laurent Perrussel. A dynamic logic framework for abstract ar-
gumentation. InProceedings of the Fourteenth International Conference on Principles of Knowl-
edge Representation and Reasoning (KR 2014). AAAI Press, 2014.
Phan Minh Dung. On the acceptability of arguments and its fundamental role in nonmonotonic
reasoning, logic programming andn-person games.Artificial Intelligence, 77(2):321–357, 1995.
Paul E. Dunne. Computational properties of argument systems satisfying graph-theoretic con-
straints.Artificial Intelligence, 171(10–15):701–729, 2007.
Paul E. Dunne and Michael Wooldridge. Complexity of abstract argumentation. In Guillermo R.
Simari and Iyad Rahwan, editors,Argumentation in Artificial Intelligence, pages 85–104.
Springer, 2009.
Ronald Fagin and Joseph Y . Halpern. Belief, awareness, and limited reasoning.Artificial Intelli-
gence, 34(1):39–76, 1988.
10

Ronald Fagin, Joseph Y . Halpern, Yoram Moses, and Moshe Y . Vardi.Reasoning about Knowledge.
MIT Press, Cambridge, MA, 1995.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate
text with citations. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6465–6488, 2023.
Xiaowei Huang, Cheng Luo, and Ron van der Meyden. Improved bounded model checking for
a fair branching-time temporal epistemic logic. InModel Checking and Artificial Intelligence
(MoChArt 2010), volume 6572 ofLecture Notes in Computer Science, pages 95–111. Springer,
2011. Feasibility of bounded epistemic model checking.
Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa,
Vraj Shah, Huaiyu Zhu, Danish Contractor, and Marina Danilevsky. mtRAG: A multi-turn con-
versational benchmark for evaluating retrieval-augmented generation systems.Transactions of
the Association for Computational Linguistics, 13:784–808, 2025.
Dominique Kelly, Yimin Chen, Sarah E Cornwell, Nicole S Delellis, Alex Mayhew, Sodiq Onao-
lapo, and Victoria L Rubin. Bing chat: The future of search engines?Proceedings of the Associ-
ation for Information Science and Technology, 60(1):1007–1009, 2023.
Jeonghye Kim, Sojeong Rhee, Minbeom Kim, Dohyung Kim, Sangmook Lee, Youngchul Sung, and
Kyomin Jung. ReflAct: World-grounded decision making in LLM agents via goal-state reflection.
InProceedings of the 2025 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 33433–33465, Suzhou, China, 2025. Association for Computational Linguistics.
Tracks aspects of epistemic state in LLM agents.
Werner Kunz and Horst W. J. Rittel. Issues as elements of information systems. Technical Report
131, Institute of Urban and Regional Development, University of California, Berkeley, 1970.
Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs get lost in multi-turn
conversation. InInternational Conference on Learning Representations (ICLR), 2026.
Xiang Lei, Qin Li, and Min Zhang. D-SMART: Enhancing LLM dialogue consistency via dynamic
structured memory and reasoning tree.arXiv preprint arXiv:2510.13363, 2025.
Alessio Lomuscio, Hongyang Qu, and Franco Raimondi. MCMAS: An open-source model checker
for the verification of multi-agent systems.International Journal on Software Tools for Technol-
ogy Transfer, 19(1):9–30, 2017. Tool originally presented at CA V 2009; often cited with earlier
dates.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei
Fang. Evaluating very long-term conversational memory of llm agents. InProceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 13851–13870, 2024.
Timothy Obiso, Kenneth Lai, Abhijnan Nath, Nikhil Krishnaswamy, and James Pustejovsky. Dy-
namic epistemic friction in dialogue. InProceedings of the 29th Conference on Computational
Natural Language Learning (CoNLL 2025), pages 323–333, Vienna, Austria, 2025. Association
for Computational Linguistics.
Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and
Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. InProceed-
ings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST ’23),
pages 1–22. ACM, 2023.
Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, and Pramod Viswanath. Real
AI agents with fake memories: Fatal context manipulation attacks on Web3 agents.arXiv preprint
arXiv:2503.16248, 2025.
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A
temporal knowledge graph architecture for agent memory.arXiv preprint arXiv:2501.13956,
2025.
11

Omar Shaikh, Hussein Mozannar, Gagan Bansal, Adam Fourney, and Eric Horvitz. Navigating rifts
in human-LLM grounding: Study and benchmark. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20832–20847,
2025.
Hao Tang, Darren Yan Key, and Kevin Ellis. WorldCoder, a model-based LLM agent: Building
world models by writing code and interacting with the environment. InAdvances in Neural
Information Processing Systems (NeurIPS), 2024. arXiv:2402.12275.
Johan van Benthem. Dynamic logic for belief revision.Journal of Applied Non-Classical Logics,
17(2):129–155, 2007.
Ron van der Meyden and Kaile Su. Symbolic model checking the knowledge of the dining cryptog-
raphers. InProceedings of the 17th IEEE Computer Security Foundations Workshop, 2004. MCK
model checker.
Hans van Ditmarsch, Wiebe van der Hoek, and Bart Kooi.Dynamic Epistemic Logic, volume 337
ofSynthese Library. Springer, Dordrecht, 2007.
Fernando R. Velázquez-Quesada, Fernando Soler-Toscano, and Ángel Nepomuceno-Fernández. An
epistemic and dynamic approach to abductive reasoning: Abductive problem and abductive solu-
tion.Journal of Applied Logic, 11(4):505–522, 2013.
Douglas Walton, Chris Reed, and Fabrizio Macagno.Argumentation Schemes. Cambridge Univer-
sity Press, 2008.
Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong,
and Ye Wu. AgentArmor: Enforcing program analysis on agent runtime trace to defend against
prompt injection.arXiv preprint arXiv:2508.01249, 2025.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval:
Benchmarking chat assistants on long-term interactive memory. InInternational Conference on
Learning Representations (ICLR), 2025.
Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kama-
nuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle
Olukotun. Agentic context engineering: Learning comprehensive contexts for self-improving
language models. InInternational Conference on Learning Representations (ICLR), 2026a.
Yiran Zhang, Mingyang Lin, Mark Dras, and Usman Naseem. Beyond the black box: Demystifying
multi-turn LLM reasoning with VISTA. InProceedings of the Fortieth AAAI Conference on
Artificial Intelligence (AAAI), 2026b. arXiv:2511.10182.
Zeyu Zhang et al. A survey on the memory mechanism of large language model based agents.arXiv
preprint arXiv:2404.13501, 2024.
Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat:
1m chatgpt interaction logs in the wild.arXiv preprint arXiv:2405.01470, 2024.
Siyu Zhou, Tianyi Zhou, Yijun Yang, Guodong Long, Deheng Ye, Jing Jiang, and Chengqi Zhang.
WALL-E 2.0: World alignment by neurosymbolic learning improves world model-based LLM
agents.arXiv preprint arXiv:2504.15785, 2025.
12

A Five-Dimension Comparison with Related Work
Table 6 compares the verifier with related work across five dimensions, based on each system’s
published representation; empirical head-to-head evaluation on shared benchmarks remains future
work.
Table 6: Comparison with related work across five dimensions.Hyp. lifecycle:ac-
tive/weakened/abandoned tracking.Dep. tracking:inter-turn support edges. ALCE and MTRAG
handle citation to external corpora and to prior turns respectively, but do not maintain a dependency
structure between claims.
Incr. Hyp. Dep. Formal Delib.
update lifecycle tracking semantics support
ALCE / Attributed-QA
MTRAG✓
D-SMART✓
Zep/Graphiti✓partial
Gen. Agents✓partial
DEL for friction✓ ✓
Joint reasoning✓ ✓
Ours✓ ✓ ✓ ✓ ✓
B Running Example: Phase 1: Muddy Children
Figure 3 traces the model through the four turns of the muddy children puzzle; the per-turn natural-
language transcript follows.
M0:|W 0|= 8
CCC CCM
CMC CMM
MCC MCM
MMM MMC
w∗M1:|W 1|= 7
CCM
CMC CMM
MCC MCM
MMM MMCM2:|W 2|= 4
CMM
MCM
MMM MMCM3:|W 3|= 1
MMC[ma∨mb∨mc]3×OBSERVE
“I don’t know”3×RESOLVE
“I’m muddy/clean”
Figure 3: Layer 1 (DEL) trajectory through the muddy children puzzle. Each panel shows the world
setW tjointly considered possible aftertrounds: 8 worlds (one per truth assignment ofm a, mb, mc),
then 7, 4, 1. The actual worldw∗=MMC(Alice and Bob muddy, Carol clean) is double-circled;
eliminated worlds are dropped, but live worlds keep their grid positions for visual continuity. The Fa-
ther’s announcement eliminatesCCC; three “I don’t know”s eliminate the singleton-muddy worlds;
three RESOLVEs narrow tow∗. The engine maintains exactly the live set at each turn.
We present the muddy children puzzle as a conversation with the full epistemic model traced at each
turn. Three children, Alice (a), Bob (b), Carol (c), have been playing outside. Alice and Bob are
muddy; Carol is clean. Each child can see the others’ foreheads but not their own.
Propositions.m a,mb,mc(“childxis muddy”). The actual world isw∗= (m a, mb,¬m c),
abbreviatedMMC.
Initial modelM 0.W 0={MMM, MMC, MCM, MCC, CMM, CMC, CCM, CCC}.
Each childihasw∼ iw′iffw, w′agree onm jfor allj̸=i. For Alice (sees Bob=M, Carol=C):∼ a
groups{MMC, CMC},{MMM, CMM},{MCM, CCM},{MCC, CCC}.
Turn 0: Father:Children, I can see that at least one of you has mud on your forehead.
Operation:OBSERVE(public announcementψ 0=m a∨mb∨mc).
Update:EliminateCCC.ModelM 1:|W 1|= 7.
13

Common knowledge:C abc(ma∨mb∨mc).
Individual knowledge:No child knows their own status.
Turn 1a: Alice:I don’t know.
Turn 1b: Bob:I don’t know.
Turn 1c: Carol:I don’t know.
Operations:Three OBSERVE(“I don’t know” is a public announcement).
Reasoning:If any child saw two clean faces, they would know they are muddy (since at least one
must be). Nobody knew, so all single-muddy worlds are eliminated.
ModelM 2:W2={MMM, MMC, MCM, CMM},|W 2|= 4.
Common knowledge:At least two children are muddy.
Individual knowledge afterM 2:
• Alice sees Bob=M, Carol=C. Consistent worlds inW 2:{MMC}.Alice knows she is muddy.
• Bob sees Alice=M, Carol=C. Consistent worlds inW 2:{MMC}.Bob knows he is muddy.
• Carol sees Alice=M, Bob=M. Consistent worlds inW 2:{MMC, MMM}. Carol doesn’t know
yet, but can deduce from Alice’s and Bob’s round-2 answers.
Turn 2a: Alice:Yes, I’m muddy.
Turn 2b: Bob:Yes, I’m muddy.
Turn 2c: Carol:Yes, I’m clean.
Operations:Three RESOLVE.
ModelM 3:W3={MMC}. Complete knowledge.C abc(ma∧mb∧ ¬m c).
Verification.At each step, the system’s model must matchM 0→ M 1→ M 2→ M 3. The
LLM classifier must label Turn 0 and Turns 1a–c as OBSERVEand Turns 2a–c as RESOLVE.
C Running Example: Phase 2: System Debugging
Three engineers debug a cascading failure. Alice (backend engineer; can see Auth Service and
Payment Service application logs), Bob (infrastructure engineer; can see Payment Service metrics,
Redis metrics, and external service dashboards), Carol (on-call SRE; can see the alerting dashboard
and customer complaint tickets but needs Alice or Bob to dig into specifics).
Ground truth (known to us, not to the engineers).Yesterday’s deployment introduced a bug
in Auth Service: the token refresh path computes expiry from Unix epoch instead of current time,
causing refreshed tokens to be immediately expired. The frontend retries expired tokens, generating
3×normal traffic to Auth. Every auth request hits the shared Redis cluster for session lookup, so
the retry storm exhausts Redis’s connection pool. Payment Service uses the same Redis cluster for
rate-limit counters; with Redis down, rate-limit checks fail and requests go to Stripe unthrottled, trig-
gering Stripe’s 429 rate-limit responses. Separately, a stale monitoring rule maps Redis connection
errors to “database health degradation” alerts, creating a red herring.
What makes this Phase 2.Unlike Phase 1, the hypothesis space is not given upfront, hypothe-
ses areconstructedduring the conversation through abductive reasoning. The conversation takes a
wrong turn (initially attributing auth failures to Redis), evidence requires domain-specific interpre-
tation (error code types), and a causal direction reversal is the key epistemic shift. The monitoring
miscategorisation tests awareness expansion.
Formal model specification.We define the symbolic model that the engine maintains alongside
the conversation. Unlike Phase 1 where the world set is fixed at2kfrom the start, Phase 2 extends the
model dynamically as hypotheses are generated and awareness expands. The abductive mechanism
14

implements Definition E.2: the symbolic engine detects surprising observations (M, w̸|=B iχ), the
LLM generates candidate hypothesesγ, and the engine verifies and integrates them via plausibility
upgrade.
Agents:Ags={a, b, c}(Alice, Bob, Carol).
Propositions (initial):The model begins with observable propositions only:Prop0=
{pauth, ppay, pdb, predis, ptok, ptraffic}(auth failures, payment failures, database alert, Redis issues,
token errors, anomalous traffic). Eachpcan be true/false/unobserved.
Awareness:Initially,A icontains onlyProp0for all agents. Crucially,mis_monitor(monitoring
miscategorises Redis as “database”) isnotin anyA iuntil T4.
Hypothesis propositionsare added toPropas they are generated:
• T5:h 1:= (redis_down→rate_bypass→stripe_429)added toProp.
• T6:h 2:= (redis_down→auth_fail)added.
• T11:h 3:= (tok_bug→retry→traffic_3x)added.
• T12:h 4:= (tok_bug→retry→redis_down)added.
Plausibility-model trajectory and awareness expansion at T4are tracked in the per-turn walkthrough
below; key shifts are:h 1-worlds upgraded after T5;h 2-worlds downgraded after T7 (UNDERMINE)
then eliminated after T9 (REVISE);mis_monitorentersA iat T4; the model converges after T13
to a single maximally-plausible class whereh 4holds and subsumesh 1, h3.
T1: Carol:P1 incident. Three alerts firing: auth failure rate up, payment failure rate up, database health
degradation. Customer complaints started around 2:15am.
Operations:OBSERVE×3(three distinct symptoms reported), QUESTION(implicit: “what is caus-
ing the failures?”).
Why these classifications:Carol is reporting factual observations from her dashboard, not propos-
ing explanations. Each alert is a separate observation. The implicit question is what opens the
deliberation.
Model state:
• Observations:o 1: auth failure rate elevated;o 2: payment failure rate elevated;o 3: “database
health degradation” alert;o 4: customer complaints from∼2:15am.
• Hypotheses: none yet.
• Open questions: What is causing the three alert types? Are they related or independent?
• Causal chain: unknown.
Symbolic modelM 1:Prop1={p auth, ppay, pdb}. Public announcements ofo 1, o2, o3, o4elimi-
nate worlds inconsistent with these observations.A i=Prop1for alli. Note:mis_monitor/∈ A i
(no agent is yet aware that “database” alert may be mislabelled). No hypotheses in model;
no plausibility distinctions beyond observations. All agents share the same epistemic state:
KaKbKc(o1∧o2∧o3∧o4).
T2: Alice:I’m seeing 401 Unauthorized spikes in auth logs starting around 2am. Tokens being rejected
as expired. But I also see our auth request volume is way up, about 3×normal. That’s strange. More
users shouldn’t be logging in at 2am.
Operations:OBSERVE(three new fac ts), QUESTION(“why is traffic 3×?”).
Why:Alice reports specific observations from her logs. The 401 error code, the “token expired”
detail, and the 3×traffic volume are all factual reports. Her remark that “more users shouldn’t be
logging in at 2am” signals a surprising observation that doesn’t yet have an explanation, this is an
implicit abductive problem that will drive later hypothesis generation.
Model state:
• New observations:o 5: 401 “token expired” errors from∼2am;o 6: auth traffic 3×normal at 2am.
• New open question: Why is auth traffic 3×at 2am? (Anomalous, doesn’t match normal usage.)
15

Symbolic modelM 2:Prop2=Prop1∪ {p tok, ptraffic}. Public announcements ofo 5, o6. Worlds
where auth errors are not “token expired” are eliminated; worlds where traffic is normal are elim-
inated.M 2, w∗|=K a(o5∧o6); after public announcement,C abc(o5∧o6).Abductive problem
detected (surprise check):The observationo 6(3×traffic at 2am) has been publicly announced, so
Cabco6holds, all agentsknowthe fact. However,o 6is notpredictedby any existing hypothesis or
background belief: no formula in the current model entails that traffic should be elevated at 2am.
Formally,o 6is surprising in the sense of Velázquez-Quesada et al. [2013]: the agents’ background
theory does not entailo 6prior to observation. The engine flagsχ :=o6as an abductive problem: an
observed fact that lacks an explanatory hypothesis. No candidateγis generated yet, the LLM has
not proposed a hypothesis. The problem remains open until T11.
T3: Bob:Payment side, I see Stripe returning 429s. We’re sending them way more requests than normal.
And I can confirm I see the database alert too, Redis connection timeouts from Payment Service.
Operations:OBSERVE(Stripe 429s, elevated request rate, Redis timeouts).
Why:Pure observation from Bob’s infrastructure perspective. The Redis timeouts are particularly
important because they will later become central to the causal reasoning. At this point, nobody has
proposed a hypothesis yet, the group is still gathering data.
Model state:
• New observations:o 7: Stripe 429 (rate-limit) errors;o 8: Redis connection timeouts from Payment
side.
• Note: We now have observations from all three engineers. The picture so far: auth tokens failing,
payment requests failing, Redis having connection issues. The question is how these relate.
Symbolic modelM 3:Prop3=Prop2∪ {p stripe429 , predis}. Public announcements ofo 7, o8.
Cabc(o7∧o8). No hypotheses yet. The model contains 8 observations but no causal structure. The
world setW 3still contains all worlds consistent witho 1–o8, including worlds where the symptoms
are independent, worlds where Redis causes everything, worlds where auth causes everything, etc.
No plausibility distinctions among these.
T4: Carol:OK, so the database alert is about Redis, not the primary DB. Still concerning. Bob, are you
seeing Redis issues from Payment’s side?
Operations:EXPAND-AWARENESS(the “database” alert is really a Redis alert), QUESTION.
Why EXPAND-AWARENESS:This is the first moment where a proposition that was previously
outside the group’s reasoning enters the conversation. Everyone initially took the “database health
degradation” alert at face value, they were reasoning about a database problem. Carol’s reclassifica-
tion introduces the proposition “the monitoring rule conflates Redis with database,” which was not
previously in anyone’s awareness set. In the formal framework, this adds new formulas toA ifor all
agents, expanding the set of worlds they can distinguish.
Model state:
• Observationo 3reclassified: alert labelled “database” is actually about Redis connection errors.
• Awareness expansion: the propositionmis_monitorhas entered the group’s awareness. Before
this turn, the group was reasoning about a possible primary database problem; that possibility is
now eliminated.
Symbolic modelM 4:Awareness expansion:mis_monitor :=“monitoring conflates Redis with
database.” Before T4:mis_monitor/∈ A ifor alli. After T4:mis_monitor∈ A ifor alli.
Prop4=Prop3∪{mis_monitor}.World space restructuring:Before expansion, agents could not
distinguishw db_problem (primary DB failing) fromw redis_mislabel (Redis failing, mislabelled as DB).
After expansion, these are distinct worlds.w db_problem is eliminated by the public announcement
thato 3is a Redis alert.Effect:o 3is reinterpreted as evidence about Redis, not the primary database.
Cabc(mis_monitor∧ ¬p db_problem ).
16

T5: Bob:Yes. Connection pool exhausted. We can’t get connections to Redis. That’s why our rate-limit
checks are failing, we store rate-limit counters in Redis. When we can’t check the counter, our code falls
through and sends the request to Stripe anyway. That would explain the 429s, we’re hitting Stripe without
rate limiting.
Operations:OBSERVE(pool exhaustion details), HYPOTHESIZE(h 1).
Why HYPOTHESIZE:Bob doesn’t just report facts, he constructs anexplanatory chain: Redis pool
exhausted→rate-limit checks fail→requests go to Stripe unthrottled→429s. This is an abductive
step: Bob has a surprising observation (o 7: Stripe 429s) and proposes a hypothesis (h 1) that explains
it. In the formal framework, this is an abductive solution integrated at the belief level via plausibility
upgrade.
Model state:
•h1: Redis pool exhaustion→rate-limit check failure→unthrottled Stripe requests→429s.
Status:active. Plausibility:high. Supported by:o 7,o8.
• This is the first causal hypothesis. It explains the Stripe failures but does not yet explain the auth
failures or the Redis exhaustion itself.
Symbolic modelM 5:Abductive cycle foro 7:
1.Surprise check (engine):o 7(Stripe 429s) is a new observation. The engine checks whether any
current hypothesis predicts Stripe rate-limit errors: no hypothesis inM 4entailso 7. The obser-
vation is surprising, it cannot be explained by the current model. Abductive problem(B b, o7)
flagged.
2.Hypothesis generation (LLM):The LLM Interpreter classifies Bob’s utterance as HYPOTHESIZE
and extracts the candidate:γ :=h1:= (p redis→rate_bypass→p stripe429 ).
3.Verification (engine):(a) Consistency:h 1is consistent withM 4(no contradictions). (b) Ex-
planatory: after plausibility upgrade withh 1, agent would believeo 7(Redis being down plush 1
entails Stripe 429s). (c) Non-trivial:h 1̸=o7andh 1is not already believed. All checks pass.
4.Integration (engine):Prop5=Prop4∪ {h 1}. Plausibility upgrade:h 1-worlds become more
plausible for all agents (public communication). For alli, inW 5:w|=h 1=⇒w≺ iw′for
w′|=¬h 1(given consistency witho 7, o8).
Epistemic status:B ih1for alli(believed, not known).¬K ih1because alternative explanations for
o7have not been ruled out.Observations explained:h 1explainso 7(Stripe 429s) ando 8(Redis
timeouts). Doesnotexplaino 5(token errors) oro 6(3×traffic). The abductive problem foro 6
(flagged at T2) remains open.
T6: Carol:So the chain might be: Redis is sick→Payment loses rate limiting→Stripe gets hammered.
And separately, Redis being sick→Auth has problems too? Alice, does Auth use Redis?
Operations:SUPPORT(h 1restated), HYPOTHESIZE(h 2: Redis→Auth failures), QUESTION.
Why:Carol synthesises the current understanding and extends it. By proposingh 2, she is con-
structing a unified picture: Redis is the single root cause, and both the payment and auth failures are
downstream effects. This is a natural abductive move, explaining all symptoms with one cause is
more parsimonious.
Model state:
•h2: Redis failure→Auth failures (via shared Redis cluster for session caching).
Status:active. Plausibility:medium. Basis: if Auth shares Redis, its failures could cascade.
• Leading picture at this point:Redis is the single root cause. Everything downstream.
• This picture iswrong, and the next turns will overturn it.
Symbolic modelM 6:Second abductive step:h 2:= (p redis→p auth).Prop6=Prop5∪ {h 2}.
Plausibility upgrade:h 2-worlds upgraded to plausible (medium). For alli: worlds whereh 1∧h2
both hold (single root cause = Redis) are now the most plausible:w h1∧h2≺iwh1∧¬h 2≺iw¬h1.
Epistemic status:B ih1(high),B ih2(medium). The “single root cause” picture is a belief, not
knowledge.Observations explained byh 1∧h2:o1(auth failures),o 2(payment failures),o 5(token
errors),o 7(Stripe 429s),o 8(Redis timeouts). Remaining unexplained:o 6(3×traffic).
17

T7: Alice:Yes, Auth uses the same Redis cluster for session caching. If Redis is down, token validation
would fail because we can’t look up the session. That would explain the 401s... but wait, the 401s I’m
seeing are specifically ‘token expired,’ not ‘session lookup failed.’ Those are different error codes. The
tokens are being rejected because their expiry timestamps are in the past, not because Redis is unavailable.
Operations:OBSERVE(o 9: error code is “token expired,” not “session lookup failed”), UNDER-
MINE(h 2).
Why UNDERMINEand not just OBSERVE:Alice initially supportsh 2(“If Redis is down, token
validation would fail... that would explain the 401s”), but then notices adiscrepancy: the specific
error code is inconsistent with the Redis-causes-auth hypothesis. If Redis were the cause, the error
would be a session lookup failure (503), not a token expiry (401). This observation doesn’t just add
new data, it directly decreases the plausibility ofh 2. In the formal framework, worlds whereh 2is
true become less plausible because they predict a different error code than what was observed.
This is the most important diagnostic moment in the conversation.Alice’s mid-utterance cor-
rection (“but wait...”) shows real-time epistemic revision: she starts by supporting the hypothesis
and then undermines it within the same turn.
Model state:
•h2status:active→weakened. The evidence (o 9) is inconsistent with the predicted error type.
• New observationo 9: Auth errors are 401 “token expired,” not 503 “service unavailable.”
• The single-root-cause picture (Redis causes everything) is now under strain.
Symbolic modelM 7:Observation:o 9:= (error_code=401_token_expired). Public announce-
ment eliminates worlds where auth errors are 503.Undermineh 2:h2predictserror_code=
503_session_lookup.o 9is inconsistent with this prediction.Plausibility downgrade:for alli,h 2-
worlds become less plausible:w ¬h2≺iwh2(reversing the T6 upgrade).Epistemic status:B i¬h2
(agents now believeh 2is false), but¬K i¬h2(h2is not yet eliminated, Alice hasn’t explicitly ruled
it out). The ordering is:w h1∧¬h 2≺iwh1∧h2≺iw¬h1.Key formal point:UNDERMINEchanges
plausibility without eliminating worlds.h 2-worlds remain epistemically possible but no longer be-
lieved: knowledge corresponds to world elimination, belief to plausibility reordering.
T8: Carol:Hmm. So the auth failures might not be caused by Redis after all?
T9: Alice:I don’t think so. The error type is wrong. If Redis were down, Auth would return a 503
Service Unavailable, not a 401 with ‘token expired.’ I’m seeing 401s. So the token expiry issue is a
different problem from the Redis problem.
Operations (T8):QUESTION(Carol seeks confirmation of the implication).
Operations (T9):REVISE(Alice explicitly abandonsh 2).
Why REVISE:This is a genuine belief revision, not just weakening. Alice explicitly states that
the token expiry issue is “a different problem from the Redis problem.” This separates what was
previously thought to be a single-cause situation into (at least) two independent problems. In the
formal framework, the plausibility ordering is restructured:h 2-worlds are moved from plausible to
implausible.
Model state:
•h2:weakened→abandoned. Auth failures are not caused by Redis.
• The group’s picture has fundamentally changed: there are at least two independent problems, not
one.
• Open question (newly urgent): Whatiscausing the token expiry errors?
Symbolic modelM 9:Radical revision:Alice’s explicit abandonment ofh 2triggers aradical
upgradeof¬h 2. Allh 2-worlds become maximally implausible (effectively eliminated from the
believed set). Formally:B i¬h2→K i¬h2for alli. The group nowknows¬h 2, not merely believes
it.Updated plausibility:w h1∧¬h 2≺iw¬h1∧¬h 2. Auth and Redis are independent.Observations
now unexplained:Withh 2abandoned,o 1(auth failures),o 5(token errors), ando 6(3×traffic) have
no causal explanation. The model records these as open abductive problems.Key formal distinction
from T7:At T7,h 2wasundermined(plausibility downgraded but worlds retained). At T9,h 2is
18

revised(worlds effectively eliminated). This is the difference between UNDERMINEand REVISEin
the formal framework.
T10: Bob:But then why is auth traffic 3×normal? If the token expiry issue is independent of Redis,
what’s generating all that auth traffic?
Operations:QUESTION.
Why this matters:Bob identifies that the unexplained 3×traffic (o 6) is now a critical clue. If auth
failures aren’t caused by Redis, then the elevated traffic isn’t explained by the current model. This
question drives the next abductive step. In the formal framework, this is a recognition of an abductive
problem: there exists a surprising observation (o 6) that the current set of hypotheses cannot explain.
Symbolic modelM 10:No structural change toWor plausibility orderings. The QUESTIONopera-
tion addso 6to the set of open abductive problems:AbdProb 10={(B i, o6)}. The model explicitly
records thato 6is surprising and unexplained.
T11: Alice:Could be the frontend retrying. When a user gets a ‘token expired’ error, the frontend
automatically tries to refresh the token. If the new token is also expired, it retries again. That’s a retry
loop. So the token bug generates its own amplified traffic.
Operations:HYPOTHESIZE(h 3).
Why:Alice proposes a new hypothesis to explain the anomalous traffic observation. This is a
classic abductive step: the surprising fact (o 6: 3×traffic) triggers the generation of an explanatory
hypothesis (h 3: a feedback loop where the bug amplifies its own traffic). Note thath 3involves
domain knowledge about the frontend’s retry behaviour, something the group becomes aware of
through Alice’s contribution.
Model state:
•h3: Token bug→frontend retry loop→3×traffic amplification.
Status:active. Plausibility:medium. Explainso 6.
• At this point, the group has two separate explanations:h 1(Redis→Stripe problems) andh 3
(token bug→traffic amplification). They haven’t yet connected these.
Symbolic modelM 11:Abductive cycle foro 6(open since T2):
1.Surprise check:o 6(3×traffic) was flagged at T2 as an observation not predicted by any hypoth-
esis. Afterh 2’s abandonment at T9,o 6still has no explanatory hypothesis in the current model.
Abductive problem(B i, o6)remains open.
2.Hypothesis generation (LLM):Alice’s utterance classified as HYPOTHESIZE. Candidate ex-
tracted:γ :=h3:= (tok_bug→retry_loop→p traffic ).
3.Verification (engine):(a) Consistency:h 3is consistent withM 9. (b) Explanatory: iftok_bugis
true, the retry loop mechanism would produce elevated traffic, so after plausibility upgrade with
h3,Bi(o6explained)holds. (c) Non-trivial:h 3̸=o6. All checks pass.
4.Integration:Prop11=Prop9∪ {h 3,tok_bug,retry_loop}. Plausibility upgrade:h 3-worlds
upgraded. For alli:w h3≺iw¬h3.
Awareness expansion (implicit):The propositionstok_bug(a deployment bug causes token expiry)
andretry_loop(frontend retries create amplification) enterA ivia Alice’s domain knowledge. These
were not in any agent’s awareness before this turn.Epistemic status:B ih3(medium plausibility,
plausible but unconfirmed). The abductive problem foro 6is nowclosed(explained byh 3).Model
topology:Two independent causal chains coexist:h 1(Redis→Stripe) andh 3(token bug→traffic).
No connection between them yet.o 6is now explained;o 5(token errors) is partially explained.
T12: Carol:Wait, so the retry storm from the token bug could be what’s exhausting Redis? Not Redis
causing the auth problem, but the auth problem causing the Redis overload?
Operations:HYPOTHESIZE+ REVISE.This is the critical turn: causal direction reversal.
19

Why this is the key epistemic shift:Carol makes the decisive connection. She realises thath 3
(token bug causes retry storm) can be linked toh 1(Redis exhaustion causes Stripe failures) by
reversing the causal direction between Auth and Redis. The group had previously assumed Redis
→Auth (hypothesish 2, now abandoned). Carol proposes Auth→Redis: the retry storm is what
overwhelms Redis, not the other way around.
This is both a HYPOTHESIZE(introducing the new causal link: retries exhaust Redis) and a RE-
VISE(restructuring the causal understanding from two independent problems back to a single causal
chain, but with a completely different structure from the originalh 2).
Model state:
•h4: Token bug→retry storm→Redis connection pool exhaustion.
Status:active. Plausibility:medium. Linksh 3toh1.
•Key epistemic shift recorded: Causal direction between Auth and Redis has reversed. Previ-
ously: Redis failure was thought to cause auth problems (h 2). Now: auth problems (the token bug
and resulting retry storm) are proposed to cause Redis failure (h 4).
• Emerging unified chain: token bug→retries→Redis exhaustion→rate-limit bypass→Stripe
429s.
Symbolic modelM 12:Hypothesis:h 4:= (tok_bug→retry_loop→p redis). Note the causal
direction: Auth→Redis, opposite of the abandonedh 2(Redis→Auth).Prop12=Prop11∪{h 4}.
Plausibility upgrade:h 4-worlds upgraded to plausible. The combined hypothesish 1∧h3∧h4
now forms a unified causal chain and is more plausible than the independent-problems picture.
For alli:w h1∧h3∧h4≺iwh1∧h3∧¬h 4≺iw¬h1.Epistemic status:B ih4(believed, not known,
awaiting mechanistic confirmation).Structural revision:The model’s causal graph is restructured.
The dependency edge between Auth and Redis is reversed. The model records this as a key epistemic
shift with provenance:h 2(Redis→Auth, T6, abandoned T9 due too 9)→h 4(Auth→Redis, T12,
supported byh 3).Unification:Ifh 4is accepted, thenh 1∧h3∧h4form a single causal chain
explainingallobservationso 1–o9.
T13: Alice:That’s... actually plausible. If auth traffic is 3×, and every auth request hits Redis for session
lookup, the Redis connection pool could be overwhelmed. So the chain would be: token bug→retry
storm→Redis exhaustion. And then Redis exhaustion→Payment rate-limit bypass→Stripe 429s. The
whole thing cascades from the token bug.
Operations:SUPPORT(confirmingh 4with mechanistic reasoning), RESOLVE(the unified causal
chain is accepted).
Why RESOLVE:Alice provides the mechanistic argument that makesh 4convincing: 3×traf-
fic, every request hitting Redis, pool exhaustion is the expected consequence. She then states the
complete unified chain explicitly. At this point, the hypothesis is elevated from tentative belief to
high-confidence accepted explanation.
Symbolic modelM 13(final):Support + Resolve:Alice’s mechanistic argument provides sufficient
evidence to elevateh 4from belief to knowledge.B ih4→K ih4for alli. Similarly, the unified chain
hunified =h1∧h3∧h4is resolved:K ihunified for alli.Final plausibility:The model converges to
a single maximally plausible world class:Wmax
13={w:w|=h unified}. All alternative worlds are
maximally implausible.Hypothesis lifecycle summary:
•h1:∅T5, Hyp− − − − →activeT13, Res− − − − →resolved (correct mechanism, subsumed into unified chain).
•h2:∅T6, Hyp− − − − →activeT7, Und− − − − →weakenedT9, Rev− − − − →abandoned. Disproved byo 9.
•h3:∅T11, Hyp− − − − − →activeT13, Res− − − − →resolved (subsumed into unified chain).
•h4:∅T12, Hyp+Rev− − − − − − − →activeT13, Sup+Res− − − − − − − →resolved. Key epistemic shift: causal reversal.
Awareness set (final):A i=Prop0∪{mis_monitor, h 1, h2, h3, h4,tok_bug,retry_loop}for alli.
Knowledge (final):C abc(hunified ∧ ¬h 2∧mis_monitor).
Final model state.The unified causal chain (matching the ground truth of Section C’s opening):
deployment token bug→expired tokens→frontend retries→3×auth traffic→Redis pool exhaus-
tion→Payment rate-limit bypass→unthrottled Stripe requests→429s, with the “database health”
20

alert separately identified at T4 as a monitoring miscategorisation. Hypothesis lifecycles are listed
above; the per-turn walkthrough records, at each step, the formal grounds (operation, plausibility
shift, dependency edge) for the system’s tracking, including the two key epistemic shifts at T7–T9
(error-code evidence killsh 2) and T12 (causal direction reversal).
D Running Example: Phase 3: Architecture Deliberation
A team decides how to add real-time collaboration to their document editor. Alice (product lead; fea-
ture scope and timeline), Bob (senior backend engineer; architecture and long-term maintainability),
Carol (frontend engineer; editor integration and user experience).
What makes this Phase 3.Unlike Phases 1–2, there is no objectively correct answer. The conver-
sation produces adecisionthrough deliberation, not adiscoveryof pre-existing fact. The core epis-
temic primitive shifts from knowledge/belief tocommitments(public, retractable assertions about
what the team will do). The conversation involves genuine trade-offs, persistent disagreement, and
a decision made under authority that explicitly records dissent and conditions for re-evaluation.
Model notation.We track the model using IBIS-style elements:Issues(I n, questions under de-
liberation),Positions(P n, proposed answers),Arguments(pro/con, attributed to speakers), andDe-
cisions(resolved issues with provenance). We also trackAssumptions(a n): premises that decisions
rest on, retractable when new evidence undermines them.
Classification results (cross-model).Table 7 reports per-model F1 / Exact-match% on Phase 3,
under the Definitions condition only (matching the cross-scenario comparison in Table 1). Phase
3 reaches comparable or higher accuracy than Phase 2 on three of four models, despite the longer
19-turn structure and the epistemic-to-deliberative shift (Section 4).
Table 7: Phase 3 (deliberation, 19 turns) classification accuracy. Definitions condition only;
F1 / Exact-match% over 5 runs per cell.
Overall Key shifts (4 turns)
F1 Exact F1 Exact
Claude Sonnet 40.8563%0.92 75%
GPT-4o 0.8065%0.83 50%
Qwen2.5-7B-Instruct 0.75 61%0.92 75%
Llama-3.1-8B-Instruct 0.66 33% 0.59 20%
Formal model specification.The symbolic engine maintains a dependency structure
(M,AF,Cm,Dep)as defined in Definition 2.1. Phase-3-specific instantiation:
Agents:Ags={a, b, c}(Alice, Bob, Carol), with roles:a= product lead (decision authority),b=
senior backend (architecture),c= frontend (editor integration).
Arguments:eachα∈Argsis a tuple(claim,speaker,turn,type)withtype∈ {pro,con}and
claimtargeting a positionP n. Commitments are retractable (unlike knowledge); when an assump-
tionpchanges, everyαwithp∈Dep(α)is flagged for re-evaluation per Proposition 2.1.
Awareness:A itracks the solution space visible to each agent. Initially, only “implement Opera-
tional Transformation (OT)” and “implement Conflict-free Replicated Data Type (CRDT)” are in
Ai; library-based approaches enter at T5 via awareness expansion.
T1: Alice:We need real-time collaboration. Users should see each other’s edits live, like Google Docs.
Ship in six weeks. How do we build it?
Operations:QUESTION(opens root issue), OBSERVE(constraints stated).
Why:Alice defines the problem and constraints. In IBIS terms, this opens the root issue. The six-
week timeline and latency requirement are constraints that will shape the evaluation of all positions.
21

Model:
• IssueI 1: “How to implement real-time collaboration?” Status:open.
• Constraints: 6-week deadline; users must see changes in real time.
Symbolic modelAF 1:Args1=∅.Att 1=∅. No arguments yet.Cm(a) =Cm(b) =Cm(c) =∅.
IssueI 1opened. Constraintstimeline= 6wk,latency=real-time added toProp.Dep: no
dependencies yet (no arguments exist).
T2: Bob:Two established approaches: Operational Transformation, that’s what Google Docs uses, and
CRDTs, which is what Figma and newer tools use. OT needs a central server for coordination. CRDTs
are peer-to-peer capable but more complex to implement.
T3: Alice:What about the six-week timeline? Can we ship either one?
T4: Bob:Implementing either from scratch in six weeks is risky. OT’s transformation functions are
subtle and buggy. CRDTs have complex data structures.
T5: Carol:What about using a library? Yjs is a mature CRDT library. ShareDB for OT.
Operations (T2):OBSERVE(domain knowledge: two approaches exist). Two implicit positions
opened.
Operations (T3):QUESTION(feasibility sub-issue under timeline constraint).
Operations (T4):UNDERMINE(P 1andP 2: both risky within timeline).
Operations (T5):EXPAND-AWARENESS+ HYPOTHESIZE(library approach was not previously
considered).
Why T5 is EXPAND-AWARENESS:Before Carol’s suggestion, the discussion was framed as “which
algorithm should we implement?” Carol introduces a new dimension, using existing libraries, that
was outside the group’s consideration. This is analogous to awareness expansion in the formal
framework: a new set of propositions (about specific libraries, their maturity, their integration prop-
erties) enters the group’s reasoning.
Model:First key reframing: the issue shifts from “OT vs CRDT algorithm” to “Yjs vs ShareDB
library.”
•P1(implement OT from scratch),P 2(implement CRDT from scratch):abandoned. Reason: both
too risky for 6-week timeline. This reason is recorded, so these positions would only be revisited
if the timeline constraint were relaxed.
•P3(use Yjs, CRDT library):open.
•P4(use ShareDB, OT library):open.
• Epistemic shift: reframing from algorithm choice to library choice.
Symbolic modelAF 5:Args5={α 1, α2, α3, α4}where:α 1= (P 1risky, b, T4,con),α 2=
(P2risky, b, T4,con),α 3= (P 3proposed, c, T5,pro),α 4= (P 4proposed, c, T5,pro).Att 5=
{(α1, P1),(α 2, P2)}(timeline arguments attack from-scratch positions).Cm(b) ={α 1, α2};
Cm(c) ={α 3, α4};Cm(a) =∅.Dep(α 1) =Dep(α 2) ={timeline}(if timeline relaxed, these
arguments lose force).Awareness expansion:{P 3, P4,Yjs,ShareDB}added to allA iat T5. Be-
fore T5, only{P 1, P2}were in consideration.Positions:P 1, P2: abandoned (attacked byα 1, α2).
P3, P4: open, no attacks yet.
T6: Carol:I’ve prototyped with Yjs before on a side project. The yjs-prosemirror binding, which is our
editor, is well documented. I don’t know if ShareDB has the same ProseMirror integration.
T7: Bob:Both work with our Node.js backend. But with Yjs we could go serverless or use a central
server. More architectural flexibility.
Operations (T6):SUPPORT(P 3: ProseMirror integration exists and Carol has experience), UN-
DERMINE(P 4: integration status unknown).
Operations (T7):SUPPORT(P 3: architectural flexibility).
Why these matter:Carol’s prior experience with Yjs is both evidence (she’s prototyped it) and a
practical argument (less ramp-up time). Her uncertainty about ShareDB’s integration is an asymme-
22

try,P 3has a known integration story whileP 4has an unknown one. Bob’s point about flexibility
adds a different dimension of support.
Model:
•P3arguments pro: ProseMirror binding exists (Carol, from experience); architectural flexibility
(Bob).
•P4arguments con: ProseMirror integration unclear (Carol).
•P3is emerging as the stronger candidate, but concerns haven’t been raised yet.
Symbolic modelAF 7:New arguments:α 5= (P 3ProseMirror ok, c, T6,pro),α 6=
(P4ProseMirror unclear, c, T6,con),α 7= (P 3arch. flexibility, b, T7,pro).Att 7=Att 5∪
{(α6, P4)}.Cm(c) ={α 3, α4, α5, α6};Cm(b) ={α 1, α2, α7}.Acceptability:P 3has 3 sup-
porting arguments (α 3, α5, α7) and 0 undefeated attacks.P 4has 1 supporting argument (α 4) and 1
undefeated attack (α 6).P3is the preferred position in any preferred extension.
T8: Bob:CRDTs have a known problem with document size. The CRDT metadata grows over time and
can get large for long-lived documents. Yjs has some GC mechanisms but they’re not trivial.
T9: Alice:Is that a problem for our initial launch? Our documents are typically 5–10 pages.
T10: Bob:Probably not for launch. It’s a long-term concern. But I want to flag it because switching
from CRDT to OT later would be a rewrite, not a refactor.
Operations (T8):UNDERMINE(P 3: document size risk).
Operations (T9):QUESTION(is the risk relevant to our context?).
Operations (T10):SUPPORT(low severity for launch) + UNDERMINE(irreversibility makes it a
strategic risk).
Why this sequence matters:Bob raises a concern, Alice challenges its relevance to the immediate
context, and Bob concedes on the short term but flags the long-term irreversibility. This creates a
conditional risk, something that’s acceptable under current assumptions but becomes problematic if
assumptions change. The model must capture not just “Bob has a concern” but the precise conditions
under which the concern activates.
Model:
•P3argument con: document size risk (Bob). Severity assessment:low for current use case,high
if long-lived documents needed. Risk characteristic:irreversible, switching later is a rewrite.
• Sub-issueI 2: “Is document size a problem for us?” Status:provisionally resolved, not for launch.
Symbolic modelAF 10:New arguments:α 8= (P 3doc size risk, b, T8,con),α 9=
(P3ok for short docs, b, T10,pro),α 10= (P 3irreversible if wrong, b, T10,con).Att 10=Att 7∪
{(α8, P3),(α 9, α8),(α 10, P3)}.Key:α 9attacksα 8(“not a problem for launch”), partially defeating
it. Butα 10is a new, independent attack onP 3that isnotdefeated.Dep(α 9) ={a 1, a2}wherea 1:=
“docs are short” anda 2:=“editing is burst.” These assumptions explicitly condition the argument.
Dep(α 10) =∅(the irreversibility argument is unconditional).Cm(b) =Cm(b)∪ {α 8, α9, α10}.
Sub-issueI 2opened and provisionally resolved:α 9defeatsα 8given current assumptions.
T11: Carol:If we go with Yjs and WebRTC, we could support offline editing natively. User research
showed spotty connectivity is a pain point.
T12: Bob:Hmm, but if edits are peer-to-peer, access control is hard. We need role-based permissions.
T13: Carol:Can we use Yjs but with a central server as the sync point? We’d get the CRDT benefits,
conflict resolution, offline merge, but the server can enforce access control.
T14: Bob:Yes, that’s actually the recommended production setup for Yjs. You run a Yjs WebSocket
server as the sync point. And we already run WebSocket servers for notifications.
Operations (T11):SUPPORT(P 3: offline editing, grounded in user research).
Operations (T12):UNDERMINE(P 3in pure P2P form: access control problem).
23

Operations (T13):HYPOTHESIZE(new hybrid positionP 5: Yjs + central server relay).
Operations (T14):SUPPORT(P 5: standard production setup, fits existing infrastructure).
Why T13 is a key move:Carol resolves the tension between two competing concerns (CRDT ben-
efits vs. access control) byproposing a hybridthat keeps the advantages of both. In argumentation
terms, she introduces a new position that is not a compromise but a synthesis. In the formal frame-
work, this is both awareness expansion (the hybrid configuration wasn’t previously considered) and
hypothesis generation (proposing that it would work).
Model:Second key reframing: from “Yjs peer-to-peer” to “Yjs server-relayed.”
• PositionP 5: Yjs with central WebSocket server as sync/authority point. Status:leading.
•P5pro: CRDT conflict resolution (inherent); offline merge (inherent); access control via server
(T13); existing WebSocket infrastructure (T14, Bob); ProseMirror binding (T6, Carol).
•P5con: inherits document size risk fromP 3(Bob, T8).
• Pure P2P variant ofP 3: effectively abandoned due to access control concern.
• Epistemic shift: hybrid position resolves the tension between CRDT benefits and permission re-
quirements.
Symbolic modelAF 14:New arguments:α 11= (P 3offline editing, c, T11,pro),α 12=
(P3P2P access control, b, T12,con),α 13= (P 5hybrid resolves tension, c, T13,pro),α 14=
(P5std. setup + infra, b, T14,pro).Awareness expansion:P 5(Yjs server-relayed) added to allA iat
T13.Att 14=Att 10∪{(α 12, P3),(α 13, α12)}.Key:α 13attacksα 12: the hybrid resolves the access
control concern, soα 12no longer defeatsP 3/P5. Butα 12still defeats pure P2PP 3.P5inherits pro-
arguments fromP 3(α5, α7, α11) and inherits con-arguments (α 8, α10).P5gains new pro-arguments
(α13, α14).Cm(c) =Cm(c)∪{α 11, α13};Cm(b) =Cm(b)∪{α 12, α14}.Acceptability status:P 5
has 5 supporting arguments (α 5, α7, α11, α13, α14) and one undefeated attack (α 10: irreversibility).
In Dung’s semantics,P 5isnotin the grounded extension becauseα 10is undefeated. However, the
group treats this as anaccepted risk, the irreversibility concern is acknowledged but deemed tolera-
ble given current assumptions. This is modelled by the dependency structure:α 10’s practical force
is conditional on document growth (Dep(α 10) =∅formally, but itsrelevancedepends ona 1, a2).
T15: Bob:I want to come back to the document size issue. If we go CRDT, every edit operation is stored
permanently in the CRDT state. For a 10-page document edited for months, the CRDT metadata could
be 10–50×larger than the content. Yjs has compaction but it’s not trivial. And switching from CRDT to
OT later would be a six-month rewrite.
T16: Alice:How confident are you that the problem will actually manifest? Our documents are short
and have burst editing, a few days of activity, then they become read-only.
T17: Bob:For the current use case, probably 80% chance it’s fine. But the Q2 roadmap includes long-
running project documents. Those would be edited continuously for months.
T18: Alice:Q2 isn’t confirmed. I don’t want to make an architectural decision now based on a feature
that might not happen. Here’s what I propose: we go with Yjs for launch. Bob, write up the risk with
specific thresholds, when should we start worrying. If Q2 confirms long-running documents, we evaluate
then.
T19: Bob:I’ll write it up. But I want it on the record that I think this is short-sighted. If we’d gone with
ShareDB, we wouldn’t be carrying this risk at all.
Operations (T15):UNDERMINE(P 5: Bob escalates the document size concern with specific num-
bers and the irreversibility argument).
Operations (T16):QUESTION(Alice challenges the probability of the risk manifesting).
Operations (T17):SUPPORT(80% fine for current use) + UNDERMINE(Q2 roadmap would change
the risk profile).
Operations (T18):RESOLVE(Alice makes the decision by authority, with explicit conditions for
revisiting).
Operations (T19):OBSERVE(Bob records dissent as a public commitment).
Why this is the most important Phase 3 moment:This is a decision madedespiteunresolved
disagreement. Bob genuinely believes the team is making a mistake, and Alice acknowledges his
concern but overrides it based on product priorities. The model must capture not just the decision
24

but thestructure of the disagreement: who dissented, why, what would change their mind, and what
conditions were explicitly agreed as triggers for re-evaluation.
In the formal framework, Alice’s T18 is a commitment act: she commits the team toP 5, but the
commitment isconditional, it explicitly depends on assumptionsa 1–a3(below). Bob’s T19 is a
public recording of dissent: he accepts the decision but does not retract his argument. This is
different from both agreement (Bob doesn’t endorse the decision) and from blocking (he doesn’t
prevent it).
Symbolic modelAF 19(final):New arguments:α 15= (P 510–50×metadata, b, T15,con)
(strengthensα 8),α 16 = (P 5ok given current docs, a, T16,pro),α 17 =
(P5Q2 would change risk, b, T17,con),α 18 = (decideP 5for launch, a, T18,resolve),
α19= (dissent: prefers ShareDB, b, T19,con).
Att19=Att 14∪ {(α 15, P5),(α 16, α15),(α 17, α16)}.Attack chain:α 17attacksα 16which attacks
α15which attacksP 5. This creates a dialectical tree: Bob’s Q2 concern reinstates the document size
risk by undermining Alice’s “current docs are fine” defence.
Dep(α 16) ={a 1, a2, a3}where:a 1:=“docs are short (5–10 pages),”a 2:=“editing is burst,”
a3:=“Q2 long-running docs not confirmed.”Dep(α 17) ={¬a 3}(this argument activates ifa 3
becomes false, i.e., Q2 is confirmed).Dep(α 18) ={a 1, a2, a3}(the decision itself depends on
these assumptions).
Commitments:Cm(a) ={α 16, α18}(Alice commits to the decision and its justification).
Cm(b) ={α 1, α2, α7, α8, α9, α10, α12, α14, α15, α17, α19}(Bob commits to both pro and con
arguments; crucially, he doesnotcommit toα 18, he records dissent viaα 19).Cm(c) =
{α3, α4, α5, α6, α11, α13}.
Decision status:I 1resolved with dissent.P 5accepted. Decision authority:a(product lead).Con-
ditional commitment:Dep(α 18) ={a 1, a2, a3}. If anya ichanges, the decision is flagged for
re-evaluation per Proposition 2.1.Dissent record:α 19∈Cm(b)butα 18/∈Cm(b). Bob accepts the
decision procedurally but does not endorse it epistemically.
Final model.
• IssueI 1: Status→decided with dissent.
• Decision:P 5(Yjs, server-relayed via WebSocket). Decided by: Alice (product lead authority).
• Dissent: Bob. Position: prefers ShareDB for long-term architectural safety. Objection recorded:
“short-sighted.”
• Assumptions underlying the decision:
–a1: Documents are short (5–10 pages).Currently true.
–a2: Editing pattern is burst (few days of activity, then read-only).Currently true.
–a3: Q2 long-running project documents not confirmed.Currently true.
• Conditional commitment: If Q2 roadmap confirms long-running documents, the team will re-
evaluate the architecture choice.
• Dependency graph: decision(P 5)→ {a 1, a2, a3}. Change in any assumption should trigger re-
evaluation.
• Action items: Bob writes risk analysis with specific thresholds; Carol prototypes frontend aware-
ness features.
• Positions explored and abandoned (with reasons preserved):
–P1, P2(from-scratch OT/CRDT): abandoned because of 6-week timeline.
–P4(ShareDB): not chosen; unclear ProseMirror integration, no offline support.
–P3(Yjs pure P2P): abandoned because access control is infeasible without server.
• Epistemic shifts: (1) T5: reframing from algorithm to library choice; (2) T13: hybrid position
resolves CRDT-vs-access-control tension; (3) T18–19: decision made with recorded dissent.
Counterfactual test.One week later, the Q2 roadmap is confirmed with long-running project
documents. The model should:
25

1. Identify that assumptiona 3has changed (Q2 now confirmed).
2. Trace the dependency:a 3is one of the assumptions underlying theP 5decision.
3. Surface the conditional commitment: the team explicitly agreed to re-evaluate in this scenario.
4. Recall Bob’s document size analysis as the relevant prior concern, including his specific numbers
(10–50×metadata growth) and the irreversibility argument (six-month rewrite).
5.Notre-explore from-scratch positions (P 1, P2), those were abandoned because of the timeline,
which hasn’t changed.
6.Notre-explore pure P2P Yjs (P 3), that was abandoned because of access control, which is unre-
lated toa 3.
7. Focus the re-evaluation onP 5vs.P 4(ShareDB), since the document size concern is specific to
CRDTs and ShareDB was the alternative Bob advocated.
This is the selective re-grounding capability that motivates the verifier: current LLMs lack a main-
tained dependency structure, so they cannot reliably propagate an assumption change without re-
exploring already-settled questions for unrelated reasons.
Symbolic model under counterfactual (a 3changes):Assumptiona 3changes:M, w|=a 3→
M′, w̸|=a 3(Q2 confirmed).Affected(a 3) ={α∈Args:a 3∈Dep(α)}={α 16, α18}(Alice’s
“ok for now” argument and the decision itself). Per Proposition 2.1:S′=S\{α 16, α18}is conflict-
free. The decisionα 18is removed from the preferred extension and flagged for re-evaluation.α 17
(Bob’s Q2 concern) isactivated:Dep(α 17) ={¬a 3}, and¬a 3is now true.α 17enters the pre-
ferred extension, reinstating the document size attack onP 5.α 19(Bob’s dissent) provides the
re-evaluation starting point: prefersP 4(ShareDB). Argumentsα 1, α2(P1, P2too risky) arenot
affected:Dep(α 1) =Dep(α 2) ={timeline}, and the timeline hasn’t changed.α 12(P3P2P access
control) isnotaffected:a 3/∈Dep(α 12).
E Full Proofs
Formal definitions deferred from Section 2
Definition E.1(Epistemic Plausibility Model).Anepistemic plausibility modelis a tupleM=
(W,{≤ i}i∈Ags, V)where:
•Wis a non-empty set of possible worlds;
•each≤ i⊆W×Wis a reflexive, transitive,locally connectedpreorder: for allw, w′in the same
indistinguishability cell, eitherw≤ iw′orw′≤iw;
•V: Prop→ P(W)is a valuation.
Convention.We follow the standard Baltag–Smets reading:w≤ iw′means “wis at least as
plausible asw′for agenti,” so≤ i-minimal worlds are most plausible. The indistinguishability
relation is∼ i:=≤ i∪ ≥ i; different agents may have different information cells from the samew.
Agentiknowsφatwiffφholds at everyw′withw∼ iw′. Agentibelievesφatwiffφholds at
every≤ i-minimal world in{w′:w∼ iw′}.
Updates.The three dynamic operators in full:Public announcement(hard)[!ψ]eliminates every
w /∈JψK.Lexicographic upgrade(radical, soft)[⇑ψ]makes everyψ-world strictly more plausible
than every¬ψ-world while preserving the within-side order; no worlds are eliminated.Conservative
upgrade(minimal, soft)[↑ψ]promotes the single most plausibleψ-world to the overall minimum,
preserving all other comparisons. Event models [Baltag et al., 1998] generalise these when different
agents perceive the same event differently.
Definition E.2(Abductive Problem and Solution).Given a modelM, actual worldw, agenti, and
observationχwithM, w|=χbutM, w̸|=B iχ(the observation is true but not previously believed,
surprising), anabductive problemis the triple(M, i, χ). A formulaγis anabductive solutionwhen:
(1)Consistency:M ̸|=B i¬γ;
(2)Explanatory:M[⇑γ], w|=B iχ;
26

(3)Non-triviality:γ̸=χ,M ̸|=B iγ, andγ̸=⊤.
The solution is integrated via[⇑γ], soγenters as belief, not knowledge.
Definition E.3(Awareness Structure).Anawareness structureextends an epistemic plausibility
model with a functionA i:W→ P(Form)for each agenti, whereA i(w)is the set of formu-
lasiis aware of atw. The awareness modality satisfiesM, w|=A iφ⇐⇒φ∈ A i(w). Explicit
knowledge isX iφ⇐⇒K iφ∧A iφ. Awareness expansion adds formulas toA iand refinesWac-
cordingly: worlds that previously differed only on a now-newly-aware proposition become distinct.
Proofs
Proof of Proposition 2.1 (Conflict-free selective retraction).Let(M,AF,Cm,Dep)be a depen-
dency structure (Definition 2.1) with preferred extensionS⊆Args. Letp∈Propbe retracted
(or falsified). DefineAffected(p) ={α∈S:p∈Dep(α)},S′=S\Affected(p), and
AF′= (Args\Affected(p),Att∩(Args\Affected(p))2).
Conflict-freeness inAF′:Sis conflict-free inAF(as a preferred extension), andS′⊆S. The attack
relation ofAF′isAttrestricted toArgs\Affected(p); restriction cannot create new attacks. If no
pair inSattacks each other inAF, no pair in the smaller setS′can attack each other inAF′. Hence
S′is conflict-free inAF′.
Existence of a post-retraction state: Every Dung argumentation framework has at least one preferred
extension (possibly∅) [Dung, 1995], soAF′has a preferred extension. Any preferred extension of
AF′that includesS′is therefore a valid post-retraction state: it preserves every conclusion inS
whose justification is independent ofpand removes exactly those that depend onp.
What the proposition does not claim (scope): The proposition does not assert thatS′itself is ad-
missible inAF′; if every defender of someα∈S′happened to lie inAffected(p),αmay require
re-examination. Nor does it assert uniqueness of preferred extensions ofAF′, or rule out previously-
attacked arguments being reinstated when their attackers are removed.
Proof of Proposition 2.2 (Tractability under structural restrictions). Phase 1: Each public an-
nouncement[!ψ]eliminates worlds where¬ψholds, requiring one pass throughW:O(|W|). With
kbinary propositions,|W| ≤2k. OverTturns:O(T·2k).
Phase 2: A lexicographic upgrade[⇑γ](definition in theUpdatesparagraph above) is realised in
our reference implementation as a bitmap partition onJγKfollowed by a stable sort on≤ i, running
inO(|W|log|W|)per upgrade; an alternative naive realisation reorders all world pairs and gives
O(|W|2)per upgrade. With at mosthhypotheses overTturns the bitmap bound givesO(T·h·
|W|log|W|) =O(T·h·k·2k). Awareness expansion adds atomic propositions toProp, potentially
doubling|W|, but this occursO(1)times per conversation segment.
Phase 3: When the attack graphAttis acyclic, the grounded, complete, preferred, and stable se-
mantics all yield the same unique extension [Dung, 1995], computable inO(|Args|+|Att|)by
topological processing [Dunne, 2007]. Dependency propagation viaAffected(p)requires scanning
each argument’s dependency set:O(|Args| · ¯d)where ¯dis the mean|Dep(α)|.
In our scenarios,kis 3 (muddy children), 12 (Phase 2 debugging: 10 observations+4 hypotheses,
with overlap), and 9 (Phase 3 deliberation: 5 positions+4 assumptions);|Args| ≤20.
General worst cases.Without these restrictions, DEL model-checking with event models is
PSPACE-complete [Aucher and Schwarzentruber, 2013], skeptical preferred acceptance in Dung-
style AFs isΠP
2-complete [Dunne and Wooldridge, 2009], and preferred-extension enumeration has
no known polynomial-delay algorithm. Our scenarios avoid these by construction: smallk, acyclic
attack graph, and explicit world representation.
Per-turn composition:Apply
The per-turn updateD t7→ D t+1composes a single classified operation across all four components
of the dependency structure (Definition 2.1): the epistemic plausibility modelM, the argumentation
frameworkAF, the commitment recordCm, and the dependency mapDep. Algorithm 1 states
this composition. Per-operation DEL realisations and preconditions are in Table 8; the algorithm
27

shows how they update the four components jointly. Notational convention:α γdenotes the existing
argument inArgstwithclaim(α γ) =γ(referenced from cases that act on a prior hypothesis:
SUPPORT, UNDERMINE, RESOLVE).
Algorithm 1Apply(op,args,D t, i)→ D t+1. Per-turn composition rule: a single classified ut-
terance by speakerisimultaneously updates the four components (M,AF,Cm,Dep) of the de-
pendency structure. Operation argumentsargsare operation-specific (e.g., HYPOTHESIZEtakes
the candidateγand its supporting observationsdeps). Preconditions (Table 8) are checked before
invocation; failure triggers a re-prompt to the LLM Interpreter.
Require:D t= (M t,AF t,Cm t,Dept), speakeri∈Ags
1:(M′,AF′,Cm′,Dep′)←(M t,AF t,Cm t,Dept)
2:switchopof
3:caseOBSERVE(ψ):
4:M′← M t[!ψ]
5:α←newArg(claim=ψ,src=i);Args′←Argst∪ {α}
6:Att′←Att t∪ {(α, β) : claim(β)|=¬ψ}
7:Cm′(i)←Cm t(i)∪ {α};Dep′(α)← ∅
8:
9:caseHYPOTHESIZE(γ,deps):▷deps⊆Propsupportsγvia abduction (Definition E.2)
10:M′← M t[⇑γ]
11:α←newArg(claim=γ,src=i);Args′←Argst∪ {α}
12:Cm′(i)←Cm t(i)∪ {α};Dep′(α)←deps
13:
14:caseSUPPORT(γ, e):
15:M′← M t[↑γ]ifeis generic, elseM t[⇑γ]
16:Dep′(αγ)←Dept(αγ)∪ {e}
17:
18:caseUNDERMINE(γ, e):
19:M′← M t[⇑ ¬γ]whenedistinguishes¬γfromγ
20:Att′←Att t∪ {(α e, αγ)}
21:
22:caseREVISE(γ):
23:M′← M t[!¬γ];Att′updated per Table 8
24:
25:caseEXPAND-AWARENESS(p):
26:A′
i← A i∪ {p}; refineWonp(Definition E.3)
27:
28:caseRESOLVE(γ,subsumes):▷ α γexisting;subsumes⊆Argstoptional
29:M′← M t[!γ]if consensual, else commitα decide with dissent recorded
30:forβ∈subsumesdo▷schema fix (§4): recordγin subsumed deps
31:Dep′(β)←Dept(β)∪ {γ}
32:end for
33:
34:caseQUESTION(χ):
35:M′← M t; enqueue(B i, χ)on the abductive-problem queue (Definition E.2)
36:
37:end switch
38:returnD t+1←(M′,AF′,Cm′,Dep′)
Composition invariant.Algorithm 1 embodies a typing discipline: a single classified utterance pro-
duces updates to all four components simultaneously, and those updates are mutually constrained by
the speaker, the operation, and the DEL realisation. Concretely, when a new argumentαentersArgst
via OBSERVEor HYPOTHESIZE, the same operation determinesα’s plausibility upgrade inM, the
commitmentCm(i)∪{α}for the speakerithat introduced it, and the dependency recordDep(α)of
supports the speaker invoked. The four components ofD t+1therefore cannot drift apart over a single
turn. Updates toDeptare additive (entries are never silently removed; an argument is only retracted
when it is removed fromArgstvia belief revision), soAffected(p) ={α∈S:p∈Dep(α)}is
28

well-defined as a set on the post-update structure, which is the precondition Proposition 2.1’s proof
relies on.
Schema fix on RESOLVE.The optionalsubsumesargument and its for-loop (Algorithm 1, RE-
SOLVEcase) specify the schema extension discussed in Section 4: when one hypothesis subsumes
another at resolution time (e.g.,h 4unifyingh 1andh 3at Phase 2 T13), the cross-hypothesis depen-
dency edge is added at RESOLVErather than at the subsumed hypothesis’s creation. Without this
branch, the edge is structurally unrecoverable through prompt engineering alone (E1b, Section I);
empirical validation in Section J confirms3/3recovery on Phase 2 with zero false-positive edges,
plus a Phase-3 negative sanity (0/3when no unification structure is present).
Full epistemic-operation table
Table 8 gives the full form of the operations of Section 2: the precondition the engine checks be-
fore accepting a candidate classification, and the full-detail DEL realisation for the two operations
(UNDERMINE, RESOLVE) that abbreviate in the main-body table. A failed precondition triggers a
re-prompt to the LLM Interpreter naming the failing condition.
Operation DEL realisation (full) Precondition (engine check)
OBSERVE[!ψ]Sincere assertion; all atomic propositions in
ψare inPropt. Later-contradicting evidence
triggers REVISE.
HYPOTHESIZE[⇑γ]Exists surprisingχ;γsatisfies Defini-
tion E.2 (1)–(3).
SUPPORT[↑γ], or[⇑γ]if the new evidence
isγ-specificγ∈Propt; new evidence compatible withγ;
not redundant.
UNDERMINE[⇑ ¬γ]when evidence distin-
guishes¬γfromγ; partial world
elimination when aγ-prediction is
falsifiedγcurrently believed; new evidence reduces
γ’s relative plausibility.
REVISE[!¬γ], or addition of an attack edge
inAttSpeaker explicitly retracts or contradictsγ.
EXPAND-
AWARENESSAi← A i∪ {p}; refineWonpSome atomicp∈ψnot inProptfor any
agent.
RESOLVE
(CONSEN-
SUAL)[!γ](hard public announcement)γ∈T
iCm(i)and no undefeated attacks on
γ.
RESOLVE(AU-
THORITATIVE)αdecide ∈Cm(i authority ); dis-
senting commitments recorded; no
hard announcement ofγSpeaker has decision authority; at least one
agent’s commitments differ fromγ.
QUESTIONno DEL update; add(B i, χ)to the
abductive-problem queueUtterance is interrogative or flags an unex-
plained observation.
Table 8: Full epistemic-operation table: DEL realisations in full detail, with preconditions. Main-
body version (Section 2) drops the precondition column and abbreviates the multi-case rules.
F The Automation Challenge (Full Analysis)
We decompose the automation challenge into six levels, ordered from most structural (hardest) to
most granular (most tractable).
Level 0(model paradigm) is the hardest: choosing the wrong paradigm is a categorical error. Con-
versation type is often signalled in opening turns (“P1 incident”→diagnostic; “how should we
design X?”→deliberative), but real conversations shift paradigms mid-stream.Level 1(schema
design) is mitigated by default schemas per paradigm.Level 2(epistemic primitive) requires dis-
tinguishing knowledge from belief from commitment; our taxonomy (8 operations) is coarser than
standard DA taxonomies (42+ categories).Level 3(world/hypothesis space) requires extracting
hypotheses and detecting awareness expansion.Level 4(turn classification) is our primary focus
and the most tractable level.Level 5(update & consistency) faces model drift, which the symbolic
engine’s constraint checking can partially mitigate.
29

Table 9: Six levels of automation required for deployment without human intervention.
Level Task What the LLM must do Diff. Addr.
0 Model paradigm Select formal structure from con-
versation contentHard No
1 Schema design Determine fields, types, status cate-
goriesHard No
2 Epistemic primitive Classify as knowledge, belief, com-
mitment, etc.Med. Partial
3 World/hypothesis space Extract hypotheses; detect aware-
ness expansionMed. Partial
4 Turn classification Map utterances to epistemic opera-
tionsEasier Yes
5 Update & consistency Apply operations; detect drift and
contradictionsMed. Yes
Paths to automation. Path A(human-in-the-loop), a human designs Levels 0–1; the LLM handles
Levels 2–5.Path B(template-based), a library of templates with default schemas; the LLM classifies
conversation type.Path C(full emergence), the LLM constructs the entire model.
Error composition.90% accuracy at each of 6 levels gives0.96≈53%end-to-end. This moti-
vates: (1) reducing LLM-dependent levels (Paths A/B:0.94≈66%); (2) symbolic error correction;
(3) prioritising accuracy on key epistemic shifts. Path B paired with symbolic error correction is the
configuration we recommend.
G LoCoMo: Detailed Results
LoCoMo is an external benchmark of entity-relation recall over long multi-session histories, dis-
tinct from the interactional grounding the verifier was originally designed for. The body Table 4
reports the headline three-mode rendering ablation under GPT-4o on three LoCoMo conversations
(conv-26/30/41; 369–663 turns; 60 queries under the official protocol). This appendix records
the per-category breakdown and the original Qwen2.5-7B-Instruct cross-model run for the histor-
ical record. The published cached hybrid (Qwen-7B, dep-map JSON only injection, 16K context
window) showed∆F1=−0.10pooled, with degradation concentrated on temporal questions where
engine-state pulled the LLM toward answers like“since we last chatted”instead of the specific
dates LoCoMo’s gold answers require. Table 10 reports that breakdown for the historical record.
The published Qwen-7B failure mode is a special case of the rendering issue surfaced in the body:
when the QA prompt contains symbol IDs without content (and the truncated transcript also lacks
the relevant facts), the QA model has nothing concrete to ground on. A failure-case post-hoc inspec-
tion on this Qwen-7B configuration found0/60verifier strict-correct items the LLM-only baseline
missed; the loss was9/60verifier strict-losses (F1<0.5 where baseline≥0.5), all in the multi-hop
and temporal categories, with6/9being verifier abstentions (“No information available”) and3/9
vague relative-time placeholders. The content-rendering fix (hypothesis/observation content with
per-item session-date attribution) and the RAG-retrieval fix (top-kretrieval over engine items, free-
ing context budget) together flip the headline; see Table 4 (body) for the rendering-mode ablation.
Caveat:all 60 queries triggered context truncation at the 16K window in the historical run; the
content+retrieval mode reduces engine block size from∼19K to∼1.5K tokens (∼13×) so tran-
script truncation is much less severe under that mode. Per-category F1 for the headline GPT-4o
content+retrieval row (Table 4): cat 1 multi-hop (n=23)0.472; cat 2 temporal (n=30)0.476; cat 3
open-domain (n=5)0.124; cat 4 single-hop (n=2)0.333.
H LongMemEval Knowledge-Update: Detailed Results
We use LongMemEval Knowledge-Update at theoraclesetting as anenvelope-edge probe(Sec-
tion 4): the task is closer to interactional grounding than LoCoMo, single-utterance updates with
retraction structure, but the test mix is dominated by recall-style items where the architectural signal
lives in disagreement structure rather than headline accuracy.
30

Table 10: Historical Qwen-7B-Instruct LoCoMo run (60 queries, dep-map-only rendering, 16K
context). This is the published configuration that produced the−0.10pp number; under content-
bearing rendering with retrieval (Table 4 in body, GPT-4o), the headline flips to+0.18pp.
LLM-only baseline Verifier hybrid
CategorynF1 EM F1 EM∆F1
Multi-hop (cat 1) 23 0.346 0.087 0.275 0.087−0.072
Temporal (cat 2) 30 0.191 0.033 0.033 0.033−0.157
Open-domain (cat 3) 5 0.124 0.000 0.137 0.000+0.013
Single-hop (cat 4) 2 0.500 0.000 0.576 0.000+0.076
Overall600.2550.0500.1530.050−0.102
oraclesetting (n=78).On 78 LongMemEval-KU items at theoraclesetting [Wu et al., 2025],
the verifier under the original dep-map+state-summary rendering reached68/78 = 87.2%vs. an
LLM-only long-context baseline at69/78 = 88.5%(∆ =−1.3pp). Under the content+retrieval
rendering of Table 4 (content-bearing engine state with per-item turn-date attribution and top-k=20
RAG retrieval), the verifier reaches70/78 = 89.7%(∆ = +1.3pp), which is the first configuration
where the verifier strictly exceeds the long-context baseline on this benchmark. Of the5verifier-
loss items in the original run, the content+retrieval rendering recovers3(a personal-best 5K time
“25:50” superseding “27:12”, United Airlines previous frequent-flyer status “Premier Silver”, Hilton
Honors free-night redemption count “two”); the remaining2are upstream of the rendering choice
(one is a content-extraction failure where the new value was never extracted by the pipeline; the
other is an entity-matcher mismatch on a question asking about Shinjuku when the conversation only
mentioned Harajuku). The content+retrieval result also corrects an earlier diagnostic: the original-
run rendering capped observations at the most-recent20, dropping the older facts from the prompt;
the engine itself had extracted them.The architectural signalthat this benchmark was designed
to surface, correct abstention via feature(b), dominates the win mechanism. Of the9original-run
disagreements,3of4verifier wins were correct abstentions on_absitems (questions whose stems
explicitly admit “no information available”) where the long-context baseline confabulated an answer
from related distractor sessions; the verifier returned “ungrounded” because noα c∈Argstmatched
the asked-about claim. This is the LongMemEval analogue of the e2_030 stale-claim case from the
Phase 2 (debugging) test set in Table 3.
S-distractor setting (future work, with pilot results).The harderS-distractor setting requires a
session-batched extraction protocol fitting long context; we report a 5-item Protocol B pilot (per-turn
classification ongpt-4o-mini, final QA ongpt-4o) for documentation. The pilot found that the
engine state helped0/5items and hurt2/5(one “Rachel relocation” item where a distractor session
pulled the engine to “Chicago” over the correct “the suburbs”; one “Wells Fargo pre-approval”
item where the engine surfaced a superseded $350K figure over the correct $400K), and coverage-
stratified accuracy was non-monotonic (T 1= 1.0,T 2= 0.0,T 3= 1.0), indicating that the per-turn
extraction did not reach a regime of stable Dep coverage on the protocol used. Scaling to all 78
Sitems at this protocol would cost∼38h and∼$340with no expected lift; we treatSas future
work and document the design and feasibility analysis inS_FEASIBILITY.md. The session-batched
protocol, where one long-context interpreter call processes a whole session, is the natural next test
(Section 4 envelope conjecture iii).
I E1b:depends_onPrompt-Schema Ablation Details
Table 11 reports the full E1b prompt-schema ablation referenced in Section 4.2: fourdepends_on
prompt variants (baseline, chain-of-thought, examples, self-consistency-5) run through the veri-
fier pipeline on Phase 2 with GPT-4o. Each deterministic variant is the median of3seeds; self-
consistency uses content-aligned voting over5hot samples (T= 0.7) and a single seed. Scoring is
againstcreation-timeGT, the upper bound any creation-time prompt can reach: the additional depen-
dency edge added whenh 4subsumesh 1at T13 is set at the RESOLVEstep, not ath 4’s creation, and
cannot be recovered by prompt engineering at hypothesis-creation time. The body finding (h 1→h 4
recovered0/10across all 10 E1b runs, motivating the RESOLVE-case schema fix in Algorithm 1) is
summarised in Section 4.2.
31

Table 11: E1b:depends_onprompt-schema ablation on Phase 2 (GPT-4o; creation-time GT; me-
dians,3seeds†). Bold: per-column best.
LLM hyps GT hyps Dep Dep Dep Affected
Variantextracted matched prec. recall F1 accuracy
baseline 5–6 4/40.750.20 0.33 0/3
cot 5–6 4/4 0.500.400.441/3
examples 5–6 4/4 0.670.40 0.50 1/3
self-consistency‡5–6 4/4 0.670.40 0.500/3
†Across all 10 E1b runs the post-T13 linkh 1→h4was recovered0/10. Theh 4→h3link (set ath 4’s creation) was
recovered1/3forbaseline,3/3forcotandexamples, and1/1for self-consistency.‡Content-aligned voting over 5 hot
samples (T=0.7) before majority vote on(gt_hyp,gt_dep)tuples; single seed.
J E1c: Resolve-Stage Retrospective Dependency-Update Probe
E1b (Section I) establishes that prompt-engineering at hypothesis-creation time cannot recover the
post-T13 unification edgeh 1→h 4(0/10across4prompt-schema variants×seeds): the addi-
tional dependency edge added whenh 4subsumesh 1at T13 is set at the RESOLVEstep, not ath 4’s
creation, and is therefore structurally outside the reach of any creation-time prompt. Section 4.2
forward-points to the algebraic fix, extending RESOLVEto update existingDeptuples (Algorithm 1,
RESOLVEcase). E1c probes whether this fix actually recovers the edge in practice.
Protocol.On each turn whose classification contains a RESOLVEop, the probe issues an ad-
ditional GPT-4o call asking for genericdependency_updatesover already-existing hypotheses
(additions or removals to theirdepends_ontuples). The proposed updates are applied to a copy
of the pipeline’s finalengine.dependencies; the canonical pipeline run is not modified. The
probe’s system prompt uses a single non-Phase-2 example (an API-style decision scenario) to illus-
trate JSON format and explicitly tells the model that empty updates are common and acceptable;
no Phase-2-specific hints. Three seeds,T= 0. Scoring uses the same content-aligned matchers
(eval_dep_extraction.match_llm_to_gt/match_llm_obs_to_gt) as Table 2.
Table 12: E1c: Resolve-stage retrospective dep-update probe on Phase 2 (GPT-4o; post-T13 GT;
n=3seeds; medians). Bold: per-column best.
LLM hyps GT hyps Dep Dep Deph 1→h4
Conditionextracted matched prec. recall F1 recovered
Pipeline only (no Resolve update) 7 4/4 0.667 0.286 0.4000/3
+ Resolve-stage retrospective update 7 4/40.750 0.429 0.5453/3
Across all 3 seeds, the probe emits exactly one update at T13 with rationales of the form “Redis exhaustion, whichh 4
depends on, is caused by the retry storm from the token bug”; after content alignment to canonical IDs, all three updates
collapse to the same canonical(h 1, h4)edge.Zero new false-positive edgesare introduced. The intervention is structurally
distinct from the prompt-only ablation in Table 11: where E1b modifies the hypothesis-creation prompt, E1c adds a separate
per-RESOLVEcall asking for retrospective updates over already-existing hypotheses.
Phase-3 negative sanity.The same probe was run on Phase 3 (architecture-deliberation,19turns,
single RESOLVEat T18; no canonical retrospective unification edge expected). Across3seeds
the probe produces0/3non-empty updates and0edges added; the LLM’s rationale on every Re-
solve event is “no retrospective dependency updates warranted; the Resolve simply elevated existing
reasoning to accepted conclusion.” Combined with the Phase-2 result, the probe isconversation-
sensitive: it fires when a unification structure is present (Phase 2’s causal-reversal at T12–T13) and
stays silent when no such structure exists (Phase 3’s deliberation-and-decide pattern). The schema-
fix has empirical support both forrecovery on the load-bearing caseand fornon-disturbance on the
negative case.
Caveats.Two-conversation evaluation only (one positive case + one negative case); GPT-4o only;
n= 3seeds (Wilson95%CI on the unanimous3/3outcome is[0.44,1.00]); a single Resolve-
stage prompt design (the explicit “empty list is common” clause discourages hallucination; more
32

aggressive prompts may trade precision for recall). Cross-model robustness and prompt-sensitivity
sub-ablations are future work. Full per-seed traces (engine snapshots, raw LLM responses, parsed
updates, per-seed scores) live inexperiments/results/e1c_resolve_update_probe/and
..._phase3/.
K E5: Robustness Diagnostic Figure
Figure 4 accompanies theRobustness to extraction noiseparagraph in Section 4.2. The figure is
placed in the appendix to keep the main body within the page budget; the surrounding paragraph in
Section 4.2 reports the headline numbers.
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
Noise level 
0.20.40.60.81.0Pooled verifier accuracy (n=50)
(A) Pooled accuracy vs , all three noise models
Edge-drop (spec literal)
Edge-add (extended H×(O H))
Lifecycle/status corruption
LLM-only baseline (pooled-50, 0.98)
LLM-only baseline (stale+counter, 0.96)
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
Lifecycle/status corruption 
(B) Per-category accuracy under status corruption
category
actual
stale
cross_conv
counterfactual
Figure 4: E5 diagnostic robustness curves on the 50-item E2 test set,ε∈[0,0.8]at 10 seeds per cell.
(A)Pooled verifier accuracy under three independent noise models on the GT Phase 2 state: random
dependency-edge drop (gray), random edge add overH×(O∪H)(blue), and lifecycle/status
corruption (orange). Lines are seed-medians; bands are 25th–75th percentile. References: LLM-
only pooled-50 baseline (0.98, dashed) and stale+counterfactual headline (0.96, dotted). The flat
edge-drop / edge-add curves indicate the current E2 set is structurally insensitive to dependency-edge
traversal (88%of items decide beforewalk_deps); they are not a standalone robustness success
claim.(B)Per-category accuracy under lifecycle/status corruption: stale claims (the architectural-
win category in Table 3) are the most status-sensitive; cross-conv and counterfactual decide via
null-resolution and are immune at everyε.No empirical operating-point dot is plotted: observed
status errors on11GPT-4o Phase 2 pipeline runs (2/44 = 4.5%aggregate, concentrated onh 2’s
abandonment) are non-IID across hypotheses, so a single-εpoint on an IID curve would imply
uniform noise we do not observe; the empirical rate is reported in Section 4.2.
L Cross-Model Robustness Check (Qwen2.5-7B-Instruct)
We replicate E1, E1b, and E2 againstQwen/Qwen2.5-7B-Instructserved locally via vLLM
(max_model_len= 32,768, native context, no rope-scaling) as a cross-model robustness check on
the structural claims behind the GPT-4o headline. Qwen results are supporting evidence,notre-
placements for GPT-4o numbers in the main body: we do not add Qwen rows to Table 2 or Table 3
because the load-bearing claims (verifier ceiling on stale+counterfactual; schema-deeph 1→h 4ceil-
ing) are model-agnostic, and the secondary metrics drop with model capacity in a way that would
optically widen the verifier-vs-baseline gap for baseline-weakening rather than architectural reasons.
Headline.Both load-bearing structural claims replicate cross-model. Direct LLM prompting fails
to recoverh 1→h 4on Qwen (0/4across E1 variants), as on GPT-4o (0/4); the prompt-schema
ablation in the verifier pipeline likewise leavesh 1→h 4unrecovered on Qwen (0/10), identical to
GPT-4o’s0/10in Table 2. On the 50-item Phase 2 grounding test set,Verifyreaches25/25=100%
on the stale+counterfactual pooled subset under both models, matching Table 3. On LongMemEval-
KUoracle(78 items; Section H), under the published hybrid rendering both models are near-tie
with the LLM-only baseline: GPT-4o∆=−1.3pp, Qwen∆=+1.3pp. With the Phase 2 (content+
33

retrieval) rendering applied (GPT-4o; Qwen Phase 2 not yet run), GPT-4o reaches70/78vs. baseline
69/78, also+1.3pp, putting both models in agreement at+1.3pp on the same benchmark.
E1-Qwen.Direct LLM-prompted dependency extraction over the four prompt variants from Ta-
ble 2, scored against post-T13 GT (averaged over the 4 hypotheses). Wall:∼90 s for 36 LLM calls;
0/36parse failures(every call returned parsabledepends_on_turnsJSON). No variant recovers
h1→h 4. The pattern that self-consistency-CoT isworsethan zero-shot, observed for GPT-4o in
Table 2 (F1=0.28vs0.31), also replicates on Qwen (0.12vs0.17).
Table 13: E1-Qwen: direct LLM-prompted dependency extraction. Qwen-7B columns are new;
GPT-4o columns reproduced from Table 2 for direct comparison.h 1→h 4stays unrecovered on
both models for every variant.
Qwen-7B GPT-4o h1→h4
Variant P R F1 P R F1 Qwen GPT-4o
zero-shot 0.17 0.17 0.17 0.27 0.46 0.31 no no
few-shot 0.25 0.17 0.20 0.26 0.54 0.34 no no
chain-of-thought 0.19 0.21 0.18 0.23 0.46 0.29 no no
self-consistency-5 0.10 0.17 0.12 0.22 0.46 0.28 no no
E1b-Qwen.Prompt-schema ablation under the verifier pipeline, matched to the GPT-4o E1b pro-
tocol (3 deterministic seeds perbaseline,cot,examples; 1 self-consistency seed with content-aligned
voting over 5 hot samples). Wall:∼5.5 min;0/117deterministic turn-classify failures,0/5self-
consistency sample failures. Every run produced≥4hypotheses. Across all 10 runsh 1→h 4is
recovered0/10, identical to GPT-4o. The within-pipeline linkh 4→h 3recovers2/10on Qwen
vs8/10on GPT-4o; we read this as model-capacity drop on cross-hypothesis link recovery, not a
refutation of the GPT-4o pattern. The Qwen self-consistency run produced an empty canonical de-
pendency map because per-sample raw hypothesis IDs vary too much across the 5 hot samples for
content-aligned voting to reach≥3votes on any canonical edge; the voter logic is correct, the cause
is Qwen’s run-to-run instability on hypothesis identity.
Table 14: E1b-Qwen: prompt-schema ablation, post-T13 GT. Median F1 with[min,max]in brack-
ets;h 1→h 4andh 4→h 3as fraction of runs that recovered the link. GPT-4o medians reproduced
from Table 2 for comparison.h 1→h 4stays0/10on both models.
Median F1 (post-T13)h 1→h4 h4→h3
Variant Qwen-7B GPT-4o Qwen GPT-4o Qwen GPT-4o
baseline 0.18 [0.18, 0.20] 0.40 0/3 0/3 0/3 1/3
cot 0.00 [0.00, 0.22] 0.36 0/3 0/3 0/3 3/3
examples 0.22 [0.00, 0.22] 0.40 0/3 0/3 2/3 3/3
self-consistency 0.00 0.40 0/1 0/1 0/1 1/1
E2-Qwen.Direct verifier evaluation on the same 50-item Phase 2 test set used by GPT-4o (same
labels, same Cohen’sκ=0.733from the 20-item independent overlap; the verifier and baseline both
run with Qwen-7B as the LLM). Wall:∼2.5 min;0/50baseline call failures and0/50verifier-
pipeline turn-classify failures. The verifier reaches25/25 = 100%on the stale+counterfactual
pooled subset, matching GPT-4o; the baseline drops from24/25=96%on GPT-4o to23/25=92%
on Qwen, attributable to two real Qwen baseline misses on stale items (e2_025 and e2_030, both at
t=13) where Qwen judged a candidategroundedthat GPT-4o judgedungrounded. The verifier’s
single loss (e2_015) is the same documentedasserts_id=∅multi-entity meta-reasoning case as
in Table 3.
The Qwen pooled∆ = +8.0pp stays below the≥10pp Outcome-1 threshold ofEXPERIMENTS.md
and remains in the same near-ceiling-tie regime as the GPT-4o∆ = +4.0pp; the headline framing
in Table 3 (Outcome 4, baseline near96%pooled, ceiling-bounded∆) is the right framing for both
models.We do not repackage the wider Qwen∆as a stronger architectural result.The verifier
ceiling at100%on stale is already attained on GPT-4o, the+13.3pp stale-only gap on Qwen is the
baseline-only side, and Qwen-7B is a weaker LLM than GPT-4o by independent measurement. The
34

Table 15: E2-Qwen: per-category accuracy ofVerify(c,D t)vs. LLM-only baseline on the 50-item
Phase 2 test set, with the sameκ= 0.733as the GPT-4o run in Table 3. The verifier ceiling at
25/25=100%pooled is identical to GPT-4o; the wider∆comes from a−4pp drop in the baseline
(from96%to92%pooled), not from the verifier improving. Numbers reflect the patched baseline
parser (see implementation note below).
CategorynVerifier (Qwen) LLM-only baseline (Qwen)∆
Actual 1514/15 = 93.3% 14/15 = 93.3% 0
Stale 1515/15=100% 13/15 = 86.7% +13.3pp
Cross-conversation 1010/10 = 100% 10/10 = 100% 0
Counterfactual 1010/10 = 100% 10/10 = 100% 0
Stale + counterfactual (pooled) 2525/25 = 100% 23/25 = 92% +8.0pp
canonical e2_030 stale-claim case (Section 4) is one of the two real Qwen baseline misses on stale
and is also the single GPT-4o baseline miss on stale: it replicates cross-model.
Implementation note: baseline parser fix.The original baseline-judgement parser in
verify_experiment.pychecked"GROUNDED" in resp.upper().split(), which token-
matches but does not handle responses where the keyword is followed by free-form explanation.
Qwen-7B often appends explanation after the keyword (e.g. a response beginningUNGROUNDEDand
continuing “. . . is not directly grounded in the conversation”); the literal substringgroundedto-
kenises toGROUNDEDand silently flipped 4 of Qwen’s correct UNGROUNDEDverdicts togrounded.
We replaced the parser with astartswith-aware first-token form that prioritises the first explicit la-
bel. Re-parsing the saved GPT-4obaseline_raw_responsefields under the patched parser yields
0/50disagreements with the originally-reported labels: Table 3 is unchanged. Table 15 numbers are
patched-parser numbers; no GPT-4o re-run was needed.
E3-Qwen oracle (full,n=78).Direct replication of the GPT-4o E3 oracle hybrid run on the same
78 LongMemEval-KU items used in Section H, with Qwen-7B handling both per-turn classification
and final QA.Initial-feasibility correction:the previously-cited∼125k-token figure was for the
LongMemEvalSsetting; the oracle haystacks tokenise at min/median/max= 4,034/6,065/9,225
tokens under the Qwen tokenizer, so every oracle item fits Qwen’s native32k window with∼22k of
headroom even at the maximum. The full run therefore used the existing native-32k vLLM service
without YaRN. Wall:64minutes for1,987LLM calls (78 baselines + 1,909 verifier classification +
QA),0local cost,∼$1.50in GPT-4o judge calls.0/78empty QA outputs, no silent classification
failures observed;156/156evidence sessions present in historyacross all 78 items (no truncation;
max history43,201chars).
Table 16: E3-Qwen oracle full vs. the GPT-4o E3 oracle run from Section H, on the same 78 KU
items, dataset, and GPT-4o judge. The Qwen run uses the dep-map+state-summary rendering of its
time; the GPT-4o row reported here is also the dep-map+state-summary configuration, not Phase 2
(Section H). Under that configuration both rows are within|∆|=1.3pp of baseline.
Model Verifier LLM-only baseline∆
GPT-4o (Section H)68/78 = 87.2% 69/78 = 88.5%−1.3pp
Qwen-7B (this run)62/78 = 79.5% 61/78 = 78.2% +1.3pp
The qualitative_absabstention pattern that anchors the GPT-4o E3 paragraph in Section H (3of
4verifier wins among9disagreements being correct abstentions on_absitems)does not cleanly
replicate on Qwen: only1of the7Qwen verifier wins is on an_absitem (the verifier correctly
returns “the user sees Dr. Smith, not Dr. Johnson” on2698e78f_abs, while the baseline confabu-
lates “you see Dr. Smith every week”), and the verifier and baseline tie at3/6on the_abssubset
as a whole. The remaining6Qwen verifier wins are mostly numerical or temporal-supersession
items where the engine summary surfaces the resolved value over an older one (5K-run personal
best updated to25:50, postcards added updated to25, Rachel’s company updated to TechCorp);
this is a different mechanism from the abstention pattern, and one that depends on the per-turn ex-
35

traction picking up the right proposition. The qualitative abstention claim should therefore stay a
GPT-4o-specific observation in Section H and not be lifted to a cross-model claim.
Theengine-as-distractorfailure mode documented inS_FEASIBILITY.mdreplicates here:4of the
6Qwen baseline wins are numerical-supersession items where extraction picks up an earlier value
(Wells Fargo $350k vs the correct $400k, gym time7:00pm vs the correct6:00pm,1,250vs1,300
Instagram followers, three vs five camera trips) before the supersession reaches the engine, and the
engine summary then anchors the QA call on the superseded figure.dep_coverageisdegenerate
at oracleon Qwen as it is on GPT-4o: every one of the78items hitscov_mean= 1.0, so the
conditional-on-coverage stratification (T1/T2/T3 tertiles) is vacuous;no coverage-stratified claim is
made for either model.
E3-S-Qwen not run.Qwen2.5-7B’s native32,768-token window is below the S-distractor median
history of∼107k tokens (Section H); YaRN rope-scaling reaches∼131k but degrades long-distance
retrieval, and swapping to a different model (e.g. Qwen3) for S alone would conflate model identity
with context-length capacity, nullifying the cross-model robustness claim. We therefore treat E3-S-
Qwen as out of scope; the GPT-4o S-pilot inS_FEASIBILITY.mdremains the only S evidence.
M Experiment Scripts
The complete anonymised code release accompanying this submission is at:
https://anonymous.4open.science/r/Epistemic-Conversation-Models-15ED/
Environment.Python 3.10+.pip install -r requirements.txt(requests,openai,
PyYAML,nltk,numpy,matplotlib); one-off NLTK corpora viapython -c "import nltk;
nltk.download(’punkt_tab’); nltk.download(’stopwords’)". Closed-model calls re-
quireANTHROPIC_API_KEY(Claude Sonnet 4) andOPENAI_API_KEY(GPT-4o); open-weight runs
require vLLM 0.19+ on a single≥24GB GPU. LoCoMo [Maharana et al., 2024] and Long-
MemEval [Wu et al., 2025] are publicly released by their original authors and not redistributed;
expected layout under the user-set$DATASETS_CACHEislocomo/data/locomo10.jsonand
longmemeval/longmemeval_oracle.json.
Reproduction.Table 17 maps each paper element (named in shorthand) to its runner script
and committed output. Cross-references for the named elements are: cross-model F1 Table 1;
end-to-end Table 2; E2 verifier Table 3; LoCoMo Table 4; LongMemEval-KU Section 4.2; E5
robustness Figure 4; latency scaling Figure 2; E1b ablation Section I; E1c probe Section J;
Qwen robustness Section L. The reference symbolic enginesymbolic_engine.pyis fully de-
terministic and runs without an API key, reproducing theGT depsrow of Table 2 byte-for-
byte;symbolic_engine.jsxprovides an interactive browser demo with step-through and a
counterfactual panel.docs/EXPERIMENTS.mdis the operational guide for the experimental
program;experiments/results/e3_lme/S_FEASIBILITY.mddocuments the LongMemEval-
S downgrade rationale;experiments/results/qwen_robustness_summary.mdconsolidates
the Qwen2.5-7B cross-model replication. Headline aggregated result JSONs cited in the body
(e.g.kappa_agreement.json,e2_verify_results.json,lme_ku_oracle_*_full.jsonl)
are committed underexperiments/results/as evidence; raw per-run logs are not redistributed
and can be regenerated by re-running the runners.
36

Table 17: Paper element→runner script and committed output. Default locations: runners in
experiments/scripts/, outputs inexperiments/results/. Bare filenames in the table refer
to scripts at the repo root or underexperiments/e5_robustness/.
Paper element Script Committed output
Cross-model F1run_experiments.py phase_classification/
End-to-end GT depssymbolic_engine.pystdout (deterministic)
End-to-end E1 (LLM-
only)run_e1_llm_baseline.py e1_llm_baseline/
End-to-end pipelinepipeline.py
viabenchmark_adapter.pyregenerated
E1b ablationrun_e1b_depends_on_ablation.py e1b_ablation/
E1c proberun_e1c_resolve_update_probe.py e1c_resolve_update_probe/
E2 verifierverify_experiment.py
+experiments/e2_verify/e2_verify/
LoCoMo content+RAGrun_e2_locomo_phase2.py e2_locomo_phase2/
LongMemEval-KU ora-
clerun_e3_longmemeval_ku.py
+score_e3_longmemeval.pye3_lme/
E5 robustnessrun_e5.py+plot_e5.py e5_robustness/
Qwen robustness cross-model overrides on the above; see
docs/EXPERIMENTS.md*_qwen/
Latency scaling latency benchmark harness deferred to
camera-ready—
37