# Policy-as-Skill: Governed LLM Decision Support with Evidence, Deterministic Control, and Audit

**Authors**: Kabeh Mohsenzadegan, Vahid Tavakkoli, Kyandoghere Kyamakya

**Published**: 2026-09-22 21:39:06

**PDF URL**: [https://arxiv.org/pdf/2609.27087v1](https://arxiv.org/pdf/2609.27087v1)

## Abstract
Organizations increasingly use LLMs for policy, compliance, risk, and operational decision support, requiring evidence validation, review routing, version control, and auditability. We introduce Policy-as-Skill (PaS), a modular runtime that packages these functions as executable, versioned policy capabilities. Thirteen methods are evaluated with a fixed Gemma4 backend on 600 development tasks. PaS+Audit achieves 53.8% exact accuracy, macro-F1 0.346, review F1 0.854, citation precision 1.000, policy-reference recall 0.984, and audit completeness 1.000, outperforming LLM+RAG on most governance and review metrics. Deterministic control raises aggregate accuracy to 61.2% but is strongly task dependent, supporting selective rather than universal rule-based intervention.

## Full Text


<!-- PDF content starts -->

Policy-as-Skill: Governed LLM Decision Support
with Evidence, Deterministic Control, and Audit
Kabeh Mohsenzadegan
Institute for Smart Systems Technologies
University of Klagenfurt
Klagenfurt, Austria
kabeh.mohsenzadegan@aau.atVahid Tavakkoli
Institute for Smart Systems Technologies
University of Klagenfurt
Klagenfurt, Austria
vahid.tavakkoli@aau.atKyandoghere Kyamakya
University of Klagenfurt / Inst. f. Smart
Systems Technologies, Austria
& Facult ´e Polytechnique
Universit ´e de Kinshasa, DR-Congo
kyandoghere.kyamakya@aau.at
Abstract—Organizations increasingly use LLMs for policy,
compliance, risk, and operational decision support, requiring
evidence validation, review routing, version control, and au-
ditability. We introduce Policy-as-Skill (PaS), a modular runtime
that packages these functions as executable, versioned policy ca-
pabilities. Thirteen methods are evaluated with a fixed Gemma4
backend on 600 development tasks. PaS+Audit achieves 53.8%
exact accuracy, macro-F1 0.346, review F1 0.854, citation preci-
sion 1.000, policy-reference recall 0.984, and audit completeness
1.000, outperforming LLM+RAG on most governance and review
metrics. Deterministic control raises aggregate accuracy to 61.2%
but is strongly task dependent, supporting selective rather than
universal rule-based intervention.
Index Terms—AI governance, policy-as-skill, retrieval-
augmented generation, policy-as-prompt, auditability, human-
in-the-loop, deterministic control, compliance decision support.
I. INTRODUCTION
Organizations increasingly use large language models for
policy question answering, compliance triage, risk classifi-
cation, and operational decision support. In such settings, a
fluent answer is insufficient. A useful system must identify
applicable policy sources, distinguish mandatory requirements
from optional guidance, route uncertain or high-impact cases
to accountable humans, record the policy and model versions
used, and preserve enough evidence to reconstruct the de-
cision. These requirements align with broader AI-risk and
management-system guidance that emphasizes traceability,
oversight, accountability, and lifecycle controls [1]–[3].
Retrieval-augmented generation (RAG) improves grounding
by providing external documents at inference time [4]. Policy-
as-Prompt (PaP) further treats natural-language policy as a
direct model instruction [5], [6]. However, prompt content and
retrieval alone do not define how policy versions are selected,
when missing evidence becomes a failure, when a human must
review the case, or which fields must exist in an auditable
record. Recent prompt-governance work questions whether
natural-language instructions can serve as stable governance
mechanisms across changing technical and institutional con-
texts [7], [8].
We therefore studyPolicy-as-Skill(PaS), an architectural
abstraction in which policy is packaged as a reusable, ver-
sioned, executable capability rather than only as text. The coreidea is intentionally modular: skill-scoped retrieval can be used
without deterministic decision override; audit/validation can be
added without changing the model decision; and a controller
can be enabled only where policy semantics are sufficiently
explicit. This decomposition is important because a governed
system can appear stronger under a composite metric simply
by emitting more metadata, while a deterministic controller
can improve some task types and degrade others.
This paper makes five contributions. First, it defines a
formal policy-skill contract and maps every component to
an executable runtime artifact. Second, it provides a repro-
ducible implementation that separates skill retrieval, deter-
ministic control, and audit validation into clean ablations.
Third, it evaluates thirteen methods over 7,800 task-method
runs using a fixed model backend and primary metrics that
do not reward PaS-specific metadata. Fourth, it performs
controller-intervention and composite-weight sensitivity anal-
yses, showing that the controller’s aggregate gain is highly
task dependent. Fifth, it explicitly distinguishesmodel-level
unseendata fromsystem-level held-outevaluation, avoiding
the inaccurate claim that benchmark instances were used to
train or fine-tune the LLM.
The open implementation and experiment artifacts are avail-
able at https://github.com/vtavakkoli/policy-as-skill.
II. RELATEDWORK
RAG augments generation with non-parametric evidence
and is a standard mechanism for knowledge-intensive NLP
[4]. Policy-facing applications extend this idea from generic
knowledge retrieval to interpretation of normative documents.
Recent studies use prompting and LLMs for privacy-policy
analysis, explainability, and policy understanding [9]–[11].
These works demonstrate that LLMs can extract and reason
over policy text, but evidence retrieval alone does not define
runtime precedence, failure behavior, review routing, or the
audit record required to reconstruct a governed decision.
PaP makes the governance policy itself part of the model
instruction. Palla et al. study policy-as-prompt for content
moderation, while Kholkar and Ahuja frame governance rules
as prompt-level agent guardrails [5], [6]. Related scholarship
examines the institutional politics of prompting, prompt gov-
ernance, and collective prompting as a governance mecha-
arXiv:2609.27087v1  [cs.AI]  22 Sep 2026

nism [7], [8], [12]. Prompt behavior can also be sensitive to
language and framing, including in ethical-reasoning settings
[13]. Together, this literature motivates our decision not to
equate a natural-language system prompt with an enforceable
policy control. PaS instead treats the prompt as one component
of a versioned executable contract that also specifies evidence,
validation, escalation, failure, and audit requirements.
A second neighboring literature studies prompts as mech-
anisms for reasoning and action. Chain-of-thought prompting
elicits intermediate reasoning traces [14]; ReAct interleaves
reasoning and acting [15]; autonomous multi-agent prompting
extends these patterns to coordinated agents [16]; and code-
as-policies work uses language-model outputs to compose
robotic manipulation skills [17]. These approaches inform the
decomposition of generation, action, and control, but they
generally optimize task execution rather than organizational-
policy accountability.
Prompt optimization has likewise been formulated as a
learned decision problem. Examples include reinforce-learned
clarification questions [18], policy-gradient prompt learning
[19], policy-gradient discrete prompt generation [20], multi-
objective reinforcement learning for prompt optimization [21],
and reinforcement-learning-based automatic prompt tuning
[22]. A related set of works uses the wordpolicyin the
control or recommendation sense, including MCTS dialogue-
policy planning [23], zero-shot policy learning [24], hierar-
chical prompt decision transformers [25], contrastive prompt
ensembles for embodied policy adaptation [26], and prompt-
as-policy over knowledge graphs [27]. We distinguish this
action-policy meaning from the governance-policy meaning in
PaS: our policy object represents organizational or regulatory
constraints to be retrieved, validated, audited, and selectively
enforced at runtime.
Finally, AI governance frameworks emphasize human over-
sight, traceability, accountability, risk management, and lifecy-
cle change control [1]–[3]. PaS operationalizes these concerns
at the application layer by binding policy scope, evidence
requirements, review triggers, failure rules, and audit fields to
each policy capability rather than relying on the model prompt
alone.
III. POLICY-AS-SKILLARCHITECTURE
We define a policy skill as
S=⟨n, v, R, E, D, H, A, F, P, C⟩,(1)
wherenis skill identity,vis version,Ris retrieval scope,Eis
required evidence,Dis the decision schema,His the human-
review trigger set,Ais the audit-field set,Fis failure behavior,
Pis the structured prompt template, andCis the contextual
contract. The context can include data category, user role,
processing location, responsible owner, or other deployment-
specific fields.
The current registry implements skills for compliance
checking, risk classification, conflict detection, evidence-
grounded recommendation, human-review routing, and policy-
version adaptation. The research runner validates that all tenTABLE I
FORMAL SKILL TUPLE AND EXECUTABLE REALIZATION.
Term Meaning Runtime realization
nskill identity registry key and selected-skill field
vversion version recorded in the governed trace
Rretrieval scope allowed policy tags/sources and top-kscope
Eevidence contract required tags, citations, policy references
Ddecision schema constrained labels and schema validation
Hreview triggers policy-based routing/escalation rules
Aaudit fields evidence IDs, hashes, versions, timestamps
Ffailure policy unknown/review behavior on missing evidence
Pprompt template structured task/evidence/output instruction
Ccontext contract data, role, location, owner and boundaries
tuple components are materialized in PaS traces. Importantly,
the tuple does not imply that every control must override the
LLM. Figure 1 shows the modular runtime: an LLM produces
a candidate answer and structured fields; validation/audit can
run independently; and a deterministic controller is optional.
This modularity yields four research variants:PaS Retrieval
uses skill selection and scoped retrieval;PaS+Controlleraddi-
tionally applies the generic deterministic controller;PaS+Audit
applies citation/schema/audit validation without the controller;
andPaS Fullcombines both. The research path records that
the historical benchmark-specific phrase-refinement controller
is not used. The deterministic controller instead inspects
retrieved normative statements and task context; it does not
read task IDs, expected labels, or expected answers.
IV. EVALUATIONDESIGN
A. Benchmark, Corpus, and Provenance
The benchmark contains 600 instances, balanced across four
task families: 150 policy question-answering, 150 compliance-
checking, 150 risk-classification, and 150 policy-conflict-
detection cases. Reference decisions are distributed across 173
not allowed, 155conditional, and 272needs reviewlabels; 453
of 600 cases require human review according to the reference
metadata. The conflict-detection subset is intentionally an
escalation-oriented task: all 150 reference decisions areneeds
review. This makes conflict detection useful for testing conflict
recognition/routing but less discriminative as a multi-class
decision benchmark.
The policy corpus contains eleven short research fix-
tures spanning data protection, cloud procurement, cybersecu-
rity/access control, workplace AI, model governance, public-
sector guidance, and policy-version changes. They are syn-
thetic, illustrative, or real-world-inspired rather than authori-
tative legal or enterprise policies. The EU AI Act sample is a
simplified educational artifact, not the regulation itself. Refer-
ence labels are single-expert-curated and are not presented as
an independently adjudicated legal gold standard.
The reported run uses seed 7, 1,000 bootstrap iterations,
Python 3.12.12 in Docker/WSL2, and the fixed Ollama iden-
tifiergemma4:e2b. Ollama was available for the run. No
benchmark data were used for model parameter training or
fine-tuning. At the same time, earlier trace-level analysis of

Fig. 1. Policy-as-Skill runtime. Skill-scoped retrieval and audit validation are separable from the optional deterministic decision controller.
the 600-task benchmark informed system/controller devel-
opment. The correct provenance statement is therefore: the
instances aremodel-level unseenbut this experiment is a
system-development-informedevaluation, not a frozen system-
level held-out test. The executable benchmark, configuration,
metrics, traces, and evaluation utilities are released with the
research artifact.
B. Methods and Research Questions
We compare thirteen rows: Direct LLM, generic LLM,
Keyword Search, Standard RAG, Hybrid RAG, Hybrid RAG
with a deterministic reranker, LLM+RAG, PaP, Structured PaP,
and the four PaS variants described above. All model-based
methods use the same configured backend.
We ask four questions.RQ1: How do the methods com-
pare on exact decision correctness and human-review routing
when PaS-specific audit fields receive no credit?RQ2: What
evidence-grounding and native audit properties are added by
PaS?RQ3: What does the controller change relative to the
audit-only PaS variant?RQ4: Are conclusions based on the
composite readiness score stable to alternative engineering
weights?
C. Metrics and Fairness
Primary decision metrics are exact categorical accuracy
and macro-F1 over the reference decision labels. Human-
review routing is reported with precision, recall, and F1.
These metrics depend only on the substantive decision/review
fields and do not reward audit metadata. Citation precision
measures whether emitted citations refer to retrieved evidence,
and policy-reference recall measures recovery of expected pol-
icy references. The automatic evidence-faithfulness metric is
retained as a diagnostic but is not treated as human validation
because the manual annotation file is empty in this run.To avoid a structural advantage from richer native output
schemas, the new runner also applies acommon trace envelope
that every method can emit. Common trace completeness is
1.0 for all methods in the reported run. We reportnative audit
completenessseparately as an architecture capability; unlike
task accuracy, it intentionally asks whether a method natively
emits the fields needed for reconstruction.
The repository retains a weighted governance-readiness
index for engineering diagnostics. Because its weights are
design choices, it is secondary. We evaluate 286 nonnegative
weight combinations on a 0.1 simplex over decision, evidence,
governance, and answer-similarity components rather than
selecting one weighting as a neutral correctness metric.
V. RESULTS
A. Primary Decision and Review Performance
Table II reports the primary results. PaS+Audit has the
strongest balanced PaS profile without deterministic decision
override: 0.538 exact accuracy (95% bootstrap CI 0.497–
0.577), macro-F1 0.346, and review F1 0.854. PaS Retrieval is
similar (0.533, 0.342, and 0.855). LLM+RAG is the strongest
conventional baseline by exact accuracy at 0.498 (0.458–
0.537), macro-F1 0.323, and review F1 0.695. Thus, skill-
scoped retrieval/audit substantially improves review routing
while exact decision gains over LLM+RAG are more modest.
PaS Full reaches the highest aggregate exact accuracy,
0.612 (0.573–0.652), but its macro-F1 is lower (0.306) than
PaS+Audit because the deterministic controller changes the
class distribution. Figure 2 visualizes this tradeoff. A paired di-
agnostic against LLM+RAG gives an aggregate exact-accuracy
difference of +11.3 percentage points (bootstrap CI +6.3 to
+16.3; exact McNemarp= 1.24×10−5). However, this ag-
gregate comparison must be interpreted together with the task-

Fig. 2. Exact decision accuracy and human-review F1 for representative
methods. PaS audit/retrieval variants provide the strongest review routing,
while the controller increases aggregate exact accuracy.
Fig. 3. Task-specific effect of adding the deterministic controller to the audit-
only PaS variant. The direction changes across task types.
type analysis below because all conflict-detection reference
labels areneeds review.
B. Controller Intervention Analysis
The clean ablation makes it possible to separate determinis-
tic control from audit validation. Relative to PaS+Audit, PaS
Full changes the model decision on 401 of 600 tasks (66.8%).
Of those interventions, 181 convert an incorrect decision into
the reference class, 137 convert a correct decision into an
incorrect class, and 83 change one incorrect class to another.
The net improvement is therefore 44 tasks, or 7.3 percentage
points overall.
That aggregate gain is not uniform. Table III and Fig. 3
show the paired effect by task type. The controller yields +59.3
points on conflict detection because it strongly encodes conflict
escalation, but it reduces risk-classification accuracy by 23.3
points and policy-QA accuracy by 8.7 points. Compliance
changes only +2.0 points. The risk-classification degradation
is statistically clear in this development set (paired exact test
p= 1.27×10−4), while the QA and compliance changes are
not significant at 0.05.
A useful diagnostic is therefore to exclude the escalation-
oriented conflict subset. Over the remaining 450 tasks,
PaS Full achieves 48.2% exact accuracy versus 50.4% for
LLM+RAG (difference -2.2 points, 95% CI -7.8 to +3.3),
whereas PaS+Audit reaches 58.2% (difference +7.8 points
versus LLM+RAG, CI +2.4 to +12.9). This post-hoc diagnostic
Fig. 4. Exact decision accuracy by task family. The full controller is strong
on the conflict-routing task but degrades risk classification relative to PaS
Retrieval and PaS+Audit.
reinforces the central design conclusion: the controller should
be applied selectively where its policy semantics are well
specified rather than treated as a universal correctness layer.
C. Evidence, Audit, and Composite Sensitivity
PaS+Audit and PaS Full both obtain citation precision 1.000
and policy-reference recall 0.984. LLM+RAG also has high
citation precision (0.994) and recall (0.965), showing that
citation validity alone does not explain the PaS review-routing
gains. Keyword Search reaches extremely high automatic
evidence faithfulness because it is extractive, yet its exact
decision accuracy is only 0.285. This illustrates why evidence
overlap and decision correctness must be reported separately.
The common trace envelope is complete for all methods
(1.000), so the primary comparison does not penalize baselines
for missing PaS-only fields. Native audit completeness, by
contrast, is an explicit architecture capability: PaS+Audit and
PaS Full score 1.000, RAG/PaP variants are approximately
0.812, and closed-book LLM variants 0.625. We interpret
these values as differences in reconstructability, not as decision
accuracy.
The secondary governance-readiness index is 0.714 for PaS
Full, 0.707 for PaS+Audit, 0.681 for PaS+Controller, 0.665
for PaS Retrieval, and 0.636 for LLM+RAG. However, the
286-point weight sensitivity analysis shows that the top-ranked
method changes materially with the engineering weights: PaS
Full ranks first for 35.1% of tested weightings, PaS+Audit
for 30.8%, and Keyword Search for 33.9%; nearly all other
methods are never first. This result directly demonstrates why
the composite should not be the paper’s primary correctness
claim.
Observed mean end-to-end latency is 9.75 s for PaS Full,
9.72 s for PaS+Audit, 9.86 s for PaS+Controller, and 16.65
s for LLM+RAG. These timings are descriptive only: the
run used Docker/WSL2 and an external host Ollama service
using CPU, not a controlled physical edge-device benchmark,
and run order/caching was not designed for performance
attribution.

TABLE II
RESULTS ON THE600-INSTANCE DEVELOPMENT BENCHMARK. EXACT ACCURACY,MACRO-F1,AND REVIEWF1ARE PRIMARY SUBSTANTIVE METRICS.
CITATION PRECISION(CIT.P),POLICY-REFERENCE RECALL(REF.R),AND NATIVE AUDIT COMPLETENESS(AUDIT)ARE SEPARATE CAPABILITIES.
Method Exact Macro-F1 Review F1 Cit.P Ref.R Audit
Direct LLM 0.335 0.239 0.425 0.000 0.034 0.625
LLM 0.295 0.211 0.490 0.000 0.026 0.625
Keyword Search 0.285 0.186 0.689 1.000 0.965 0.812
Standard RAG 0.380 0.247 0.621 0.949 0.959 0.811
Hybrid RAG 0.375 0.243 0.609 0.979 0.961 0.812
Hybrid RAG + RR 0.367 0.231 0.647 0.983 0.965 0.812
LLM + RAG 0.498 0.323 0.695 0.994 0.965 0.812
Policy-as-Prompt 0.400 0.254 0.603 0.974 0.959 0.812
Structured PaP 0.393 0.247 0.491 0.987 0.961 0.812
PaS Retrieval 0.533 0.3420.8550.9900.9840.874
PaS + Controller0.6120.306 0.820 0.9910.9840.875
PaS + Audit 0.5380.3460.8541.000 0.984 1.000
PaS Full0.6120.306 0.8201.000 0.984 1.000
TABLE III
CONTROLLER EFFECT: PAS FULL MINUSPAS+AUDIT EXACT ACCURACY.
CIS ARE PAIRED BOOTSTRAP95%INTERVALS.
Task Audit Full∆95% CI
Policy QA 0.407 0.320 -0.087 [-0.200, 0.027]
Compliance 0.633 0.653 +0.020 [-0.087, 0.127]
Risk class. 0.707 0.473 -0.233 [-0.347, -0.120]
Conflict det. 0.407 1.000 +0.593 [0.513, 0.673]
VI. DISCUSSION
The new experiment changes the interpretation of Policy-
as-Skill in three important ways. First, the architectural value
of PaS is clearest when decision, evidence, and governance
properties are kept separate. The strongest conventional base-
line already retrieves and cites policy evidence well; PaS con-
tributes a more explicit skill contract, stronger review routing,
and complete native audit reconstruction. These properties
are meaningful even when the underlying model decision is
unchanged.
Second, deterministic control is not synonymous with better
reasoning. The controller is highly effective for a narrowly
specified escalation task, but its broad use degrades two other
task families. The result argues forpolicy-selective controllers:
deterministic logic should be used where a policy rule can be
represented with high semantic precision (for example, explicit
prohibition, missing mandatory evidence, or required escala-
tion). Ambiguous classification and interpretive QA should
preserve model uncertainty and rely more heavily on evidence,
validation, and human review.
Third, benchmark provenance must be described at the
correct system layer. The LLM was not trained or fine-tuned
on the benchmark, so calling the data “seen by the model”
would be inaccurate. Nevertheless, system-level error analysis
influenced controller development, so the 600 tasks are not a
pristine held-out test of the final system. This distinction mat-
ters for governed hybrid systems in which important behavior
resides outside model weights.A. Operational Implications for Decision Support
From a decision-support perspective, PaS shifts governance
from an unstructured prompt fragment to a lifecycle-managed
capability. A policy skill binds retrieval scope, evidence re-
quirements, decision schema, review triggers, audit fields, fail-
ure behavior, and contextual boundaries to an explicit version.
This separates the model’s role in interpreting evidence from
the organization’s role in defining decision authority. When
policy changes, a new skill version can be introduced while
historical traces retain the version, evidence, and routing logic
that governed the original decision. This makes policy change
observable and supports replay or regression analysis without
silently changing the basis of earlier decisions.
The results also supportdifferentiated automation. PaS Re-
trieval, PaS+Audit, and PaS Full correspond to increasingly in-
terventionist operating profiles: evidence-grounded assistance,
governed assistance with validation and reconstruction, and
assistance with deterministic decision intervention. Their task-
level behavior shows that no single profile should be assumed
universally preferable. Explicit prohibitions, mandatory ev-
idence requirements, and unambiguous escalation rules are
suitable candidates for deterministic enforcement; interpretive
QA and classification-heavy policies should preserve uncer-
tainty and rely more on validated evidence and accountable
review. In this sense, controller eligibility should be treated
as a property of the policy skill rather than a global system
switch.
Human review is therefore a first-class governed outcome
rather than an error state. Review triggersHand failure
behaviorFallow missing evidence, unresolved conflicts, or
high-impact contexts to produce a visible escalation instead
of forcing a model decision. The repository additionally sep-
arates development evaluation from stronger future validation
through benchmark provenance, frozen-test manifests, overlap
checks, human-annotation support, and physical-device mea-
surement scripts. These mechanisms do not make the present
benchmark held out, but they make subsequent validation and
change management explicit and reproducible.
A related management implication concerns measurement.
Exact decision quality, evidence grounding, review routing,

and audit completeness answer different operational questions
and should not be collapsed into a single success indicator. The
sensitivity analysis illustrates this directly: changing composite
weights changes the preferred method. For governance report-
ing, organizations should therefore retain a small metric port-
folio and define acceptable trade-offs per use case, rather than
interpreting a single readiness score as a universal measure of
correctness or deployment fitness. This also makes deployment
trade-offs easier to revisit when organizational risk tolerance
or review capacity changes.
VII. CONCLUSION
This paper presented Policy-as-Skill as a modular architec-
ture for governed LLM decision support. The contribution is
not a claim that a policy skill makes the base LLM intrinsically
more intelligent. Rather, PaS makes policy scope, evidence
requirements, review triggers, failure behavior, validation, and
audit reconstruction explicit and executable.
The revised evaluation supports a more nuanced conclusion
than a single aggregate score. Skill-scoped retrieval and audit
validation provide strong review routing, citation validity,
and reconstructable records without necessarily overriding
the model decision. A deterministic controller can improve
narrowly specified escalation behavior, but the current ablation
shows that blanket controller use can reduce accuracy on risk
classification and policy QA. The most defensible design is
therefore modular: use deterministic controls selectively, pre-
serve evidence and uncertainty, and keep accountable humans
in the loop for ambiguous or high-impact cases. A frozen,
independently validated held-out evaluation remains the next
step before making system-generalization or legal-correctness
claims.
REFERENCES
[1] E. Tabassi, “Artificial Intelligence Risk Management Framework (AI
RMF 1.0),” National Institute of Standards and Technology, Gaithers-
burg, MD, USA, NIST AI 100-1, Jan. 2023, doi: 10.6028/NIST.AI.100-
1.
[2] ISO/IEC 42001:2023,Information Technology—Artificial Intelligence—
Management System, 1st ed., International Organization for Standard-
ization/International Electrotechnical Commission, Dec. 2023.
[3] European Parliament and Council of the European Union, “Regulation
(EU) 2024/1689 of 13 June 2024 laying down harmonised rules on arti-
ficial intelligence and amending Regulations (EC) No. 300/2008, (EU)
No. 167/2013, (EU) No. 168/2013, (EU) 2018/858, (EU) 2018/1139
and (EU) 2019/2144 and Directives 2014/90/EU, (EU) 2016/797 and
(EU) 2020/1828 (Artificial Intelligence Act),”Official Journal of the
European Union, L 2024/1689, PE/24/2024/REV/1, Jul. 12, 2024.
[Online]. Available: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
[4] P. Lewiset al., “Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks,” inAdvances in Neural Information Processing
Systems, vol. 33, pp. 9459–9474, 2020.
[5] K. Palla, J. L. R. Garc ´ıa, C. Hauff, F. Fabbri, A. Damianou, H.
Lindstr ¨om, D. R. Taber, and M. Lalmas, “Policy-as-Prompt: Re-
thinking Content Moderation in the Age of Large Language Mod-
els,” inProceedings of the 2025 ACM Conference on Fairness, Ac-
countability, and Transparency (FAccT ’25), pp. 840–854, 2025, doi:
10.1145/3715275.3732054.
[6] G. Kholkar and R. Ahuja, “Policy-as-Prompt: Turning AI Governance
Rules into Guardrails for AI Agents,” arXiv:2509.23994, 2025.
[7] A. Neumann, H. Sargeant, and J. Singh, “Prompt Governance? On
Governing Technologies Governed by Natural Language,” inProc. ACM
Conf. Fairness, Accountability, and Transparency (FAccT), pp. 6466–
6509, 2026, doi: 10.1145/3805689.3806763.[8] L. Amoore, S. J. Bennett, A. Campolo, B. Jacobsen, and L. Rella,
“Politics of the Prompt: Government in the Age of Generative AI,”
Economy and Society, vol. 54, no. 3, pp. 573–596, 2025.
[9] A. Goknil, F. B. Gelderblom, S. Tverdal, S. Tokas, and H. Song,
“Privacy Policy Analysis through Prompt Engineering for LLMs,”
arXiv:2409.14879, 2024.
[10] Y . Chen, P. Tang, W. Qiu, and S. Li, “Using LLMs for Automated
Privacy Policy Analysis: Prompt Engineering, Fine-Tuning and Explain-
ability,” arXiv:2503.16516, 2025.
[11] H. Chen, K. S. Cherukuri, X. Zhu, and S. Yang, “Are Prompts All You
Need?: Chatting with ChatGPT on Disinformation Policy Understand-
ing,”Proc. Assoc. Inf. Sci. Technol., vol. 61, no. 1, pp. 488–492, 2024.
[12] R. Mushkani, “Prompt Commons: Collective Prompting as Governance
for Urban AI,” arXiv:2509.12415, 2025.
[13] U. Agarwal, K. Tanmay, A. Khandelwal, and M. Choudhury, “Ethical
Reasoning and Moral Value Alignment of LLMs Depend on the Lan-
guage We Prompt Them In,” inProc. LREC-COLING, pp. 6330–6340,
2024.
[14] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q.
V . Le, and D. Zhou, “Chain-of-Thought Prompting Elicits Reasoning in
Large Language Models,” inAdvances in Neural Information Processing
Systems, vol. 35, pp. 24824–24837, 2022.
[15] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“ReAct: Synergizing Reasoning and Acting in Language Models,” in
International Conference on Learning Representations (ICLR), 2023.
[16] Q. Wang, T. Wang, Z. Tang, Q. Li, N. Chen, J. Liang, and B. He, “All
It Takes Is One Prompt: An Autonomous LLM-MA System,” inICLR
Workshop on Foundation Models in the Wild, 2025.
[17] M. G. Arenaset al., “How to Prompt Your Robot: A PromptBook for
Manipulation Skills with Code as Policies,” in2024 IEEE International
Conference on Robotics and Automation (ICRA), pp. 4340–4348, 2024,
doi: 10.1109/ICRA57147.2024.10610784.
[18] X. Yan, Y . Song, X. Cui, F. Christianos, H. Zhang, D. H. Mguni, and J.
Wang, “Ask More, Know Better: Reinforce-Learned Prompt Questions
for Decision Making with Large Language Models,” arXiv:2310.18127,
2023.
[19] P. Lu, L. Qiu, K.-W. Chang, Y . N. Wu, S.-C. Zhu, T. Rajpurohit, P.
Clark, and A. Kalyan, “Dynamic Prompt Learning via Policy Gradient
for Semi-Structured Mathematical Reasoning,” arXiv:2209.14610, 2022.
[20] C. Li, X. Liu, Y . Wang, D. Li, Y . Lan, and C. Shen, “Dialogue for
Prompting: A Policy-Gradient-Based Discrete Prompt Generation for
Few-Shot Learning,” inProc. AAAI Conf. Artificial Intelligence, vol.
38, no. 16, pp. 18481–18489, 2024.
[21] Y . Jafari, D. Mekala, R. Yu, and T. Berg-Kirkpatrick, “MORL-Prompt:
An Empirical Analysis of Multi-Objective Reinforcement Learning for
Discrete Prompt Optimization,” inFindings of ACL: EMNLP, pp. 9878–
9889, 2024.
[22] M. Kwon, G. Kim, J. Kim, H. Lee, and J. Kim, “StablePrompt:
Automatic Prompt Tuning Using Reinforcement Learning for Large Lan-
guage Model,” inProc. Conf. Empirical Methods in Natural Language
Processing (EMNLP), pp. 9868–9884, 2024.
[23] X. Yu, M. Chen, and Z. Yu, “Prompt-Based Monte-Carlo Tree Search
for Goal-Oriented Dialogue Policy Planning,” inProc. Conf. Empirical
Methods in Natural Language Processing (EMNLP), pp. 7101–7125,
2023.
[24] M. Song, X. Wang, T. Biradar, Y . Qin, and M. Chandraker, “A Mini-
malist Prompt for Zero-Shot Policy Learning,” arXiv:2405.06063, 2024.
[25] Z. Wang, H. Wang, and Y . Qi, “Hierarchical Prompt Decision Trans-
former: Improving Few-Shot Policy Generalization with Global and
Adaptive Guidance,” inCompanion Proceedings of the ACM Web
Conference 2025 (WWW ’25 Companion), pp. 520–529, 2025, doi:
10.1145/3701716.3715233.
[26] W. K. Kim, S. Kim, and H. Woo, “Efficient Policy Adaptation with
Contrastive Prompt Ensemble for Embodied Agents,” inAdvances in
Neural Information Processing Systems, vol. 36, pp. 55442–55453, 2023.
[27] J. Wang, L. Zhang, Y . Cui, T. Zhang, Z. Shen, Y . Liu, X. Ma, and J. Jin,
“Do We Really Need SFT? Prompt-as-Policy over Knowledge Graphs
for Cold-Start Next POI Recommendation,” arXiv:2510.08012, 2025.