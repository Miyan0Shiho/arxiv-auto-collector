# Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents

**Authors**: Chunming Wu, Dafei Qiu, Congde Yuan, Charles Quan, Jun Wu, Suipeng Li, Mo Wu, Gavin Xie, Hope Chen, Max Yao

**Published**: 2026-07-20 15:07:58

**PDF URL**: [https://arxiv.org/pdf/2607.18039v1](https://arxiv.org/pdf/2607.18039v1)

## Abstract
Production customer-service bots must improve answer quality across iterative releases, yet large language models must not bypass evidence boundaries, policy rules, or human-handoff safeguards. We present an \textbf{Evidence-Grounded Customer-Service Agent Workflow} deployed in a real-world customer-service setting. BM25 recall, issue-title-vector recall, issue-description-vector recall, weighted RRF fusion, and cross-encoder reranking construct grounded FAQ evidence for controlled LLM decisions. Policy-guided orchestration then combines this RAG evidence with scenario-specific rule evidence, conversation memory, and clarification state inside a fixed LangGraph DAG~\cite{langgraph2024}. The paper contributes three reusable deployment patterns: \textbf{hybrid RAG evidence construction}, where multi-channel retrieval and reranking produce auditable FAQ candidates; \textbf{evidence-grounded issue/action decision}, where an Evidence-Grounded Decision Module selects an issue/action from typed FAQ evidence and scenario-specific rule evidence; and \textbf{trace-driven RAG and reranker improvement}, where traces diagnose whether failures come from recall, ranking, final candidate selection, clarification, rule-derived evidence, or action policy, and where reranker fine-tuning is evaluated not only for in-domain gain but also for forgetting risk.

## Full Text


<!-- PDF content starts -->

Evidence-in-the-Loop: Trace-Driven Optimization for
Customer-Service LLM Agents
Chunming Wu
mingo.wu@flsdex.com
Shenzhen, ChinaDafei Qiu
david.qiu@flsdex.com
Shenzhen, ChinaCharles Quan
charles.quan@flsdex.com
Shenzhen, ChinaJun Wu∗
john.wu@flsdex.com
Shenzhen, China
Suipeng Li
patrick.s.li@flsdex.com
Shenzhen, ChinaMo Wu
morgan.mo@flsdex.com
Shenzhen, ChinaGavin Xie
gavin.xie@flsdex.com
Shenzhen, ChinaHope Chen
hope.chen@flsdex.com
Shenzhen, China
Max Yao
max.yao@flsdex.com
Shenzhen, ChinaCongde Yuan
congdeyuan@gmail.com
Shenzhen, China
Abstract
Production customer-service bots must improve answer quality
across iterative releases, yet large language models must not bypass
evidence boundaries, policy rules, or human-handoff safeguards. We
present anEvidence-Grounded Customer-Service Agent Work-
flowdeployed in a real-world customer-service setting. BM25 recall,
issue-title-vector recall, issue-description-vector recall, weighted
RRF fusion, and cross-encoder reranking construct grounded FAQ
evidence for controlled LLM decisions. Policy-guided orchestration
then combines this RAG evidence with scenario-specific rule evi-
dence, conversation memory, and clarification state inside a fixed
LangGraph DAG [ 10]. The paper contributes three reusable de-
ployment patterns:hybrid RAG evidence construction, where
multi-channel retrieval and reranking produce auditable FAQ can-
didates;evidence-grounded issue/action decision, where an
Evidence-Grounded Decision Module selects an issue/action from
typed FAQ evidence and scenario-specific rule evidence; andtrace-
driven RAG and reranker improvement, where traces diag-
nose whether failures come from recall, ranking, final candidate
selection, clarification, rule-derived evidence, or action policy, and
where reranker fine-tuning is evaluated not only for in-domain
gain but also for forgetting risk. The evaluation is organized as a
Diagnosis→Optimization→Validation story rather than a sin-
gle leaderboard number. In E1, a 309-query non-test diagnostic
subset from the curated historical T-Set sample shows that hybrid
recall provides high top-50 coverage (96.76%), but the base BGE
reranker reaches only 56.31% Hit@1; under this diagnostic pipeline,
replacing Qwen3.5-27B with GPT-4o changes final issue-election
accuracy by less than 1pp, suggesting that backbone scaling alone
is not the dominant bottleneck. In E2, the curated historical T-Set
sample (605 anonymized conversation cases / 1,427 turns) provides
a cleaned FAQ subset: 581 turn-level queries labeled with gold
FAQ issue IDs, split into 492 train, 52 development, and 37 held-
out test queries. The 544 non-test queries show that RRF fusion
reaches 99.3% Hit@50 and the base reranker reaches 95.2% Hit@10,
while teacher-score distillation improves held-out reranker Hit@1
from 56.76% to 75.68% and improves the six-task C-MTEB average
from 66.09 to 66.58. In E3, on a sealed 200-session B-Set, the base
∗Corresponding author.and distilled rerankers share the same RRF candidate pool, while
the distilled reranker improves issue-turn Rerank@1 from 69.4% to
79.2% and raises KB-grounded session accuracy from 86.5% to 88.5%.
In E4, with the base reranker fixed, a DPO-trained final-decision
LLM raises KB-grounded B-Set session accuracy to 90.5%, isolating
decision-stage post-training from reranker changes. A post-launch
operational review on two equally sized production samples reports
89.52% reviewed accuracy for the Agent workflow and 79.00% for
the legacy RAG-only workflow under the same session-level rubric.
The main takeaway is that production customer-service agents
should first diagnose where the decision pipeline fails—retrieval,
reranking, or final selection—and then apply the right update to
the evidence layer or the final decision layer, rather than relying
solely on a larger or more autonomous backbone model.
Keywords
large language models, customer service, retrieval-augmented gen-
eration, evidence-grounded decision making, reranking, direct pref-
erence optimization, production systems
1 Introduction
Production customer-service agents face two closely related chal-
lenges. First, user intents, product policies, and support knowledge
evolve continuously, while the base LLM and existing FAQ corpus
may lag behind these changes. Second, when evidence is incomplete,
relying on the LLM alone can lead to hallucinated, over-generalized,
or policy-inconsistent responses. In customer-service workflows,
such deviations may mislead users, increase complaint risk, and
create operational or compliance exposure. We therefore avoid
treating the LLM as a free-form answer generator. Instead, the
system supplies the LLM with current FAQ evidence, rule-derived
scenario evidence, and replay-verified feedback, so that responses
are grounded in auditable evidence and can be continuously re-
viewed and improved.
Customer-service assistants in regulated, high-stakes domains
combine heterogeneous evidence sources: unstructured FAQ arti-
cles, permissioned context fields, and deterministic business rules.
Existing systems typically adopt one of two extremes:
1
arXiv:2607.18039v1  [cs.IR]  20 Jul 2026

Chunming Wu, Dafei Qiu, Charles Quan, Jun Wu, Suipeng Li, Mo Wu, Gavin Xie, Hope Chen, Max Yao, and Congde Yuan
Table 1: Reusable workflow patterns in the Evidence-
Grounded Customer-Service Agent Workflow.
Pattern Role Production support
Hybrid RAG evidence
constructionBuild auditable FAQ
candidates from lexical and
semantic retrieval.BM25 + vector recall;
weighted RRF;
cross-encoder reranking
Evidence-grounded
issue/action decisionSelect answer, clarification,
or handoff from fused
evidence.Fixed LangGraph DAG;
rule-derived evidence;
clarification state
Trace-driven
improvementTurn replay failures into
targeted updates.Replay traces; Human+AI
review; hard negatives and
preference pairs; sealed
B-Set validation
•Pure RAG pipelines[ 7,11] retrieve semantically similar FAQ
entries and generate answers with an LLM. They handle para-
phrase and long-tail wording well, but may ignore scenario-
specific context fields and strict policy constraints when struc-
tured evidence is available.
•Hard rule-first pipelines[ 8,29] invoke backend services and
return rule-matched answers directly. They enforce policy pre-
cisely when triggers fire, but over-trigger on noisy context fields
and cannot reconcile conflicting or incomplete evidence.
We study a deployedEvidence-Grounded Customer-Service Agent
Workflow. It lets the LLM interpret intent and select an answer, clar-
ification, or handoff from normalized FAQ evidence, rule-derived
scenario evidence, conversation memory, and clarification state,
while external calls, rule execution, and serving boundaries remain
under workflow control. We useEvidence-in-the-Loopfor the sur-
rounding trace-driven process in which the same evidence localizes
failures and guides KB, reranker, prompt/rule, or final-decision-
model updates.
The implementation uses a LangGraph [ 10]StateGraph orga-
nized as a fixed DAG with session analysis, evidence availability
checks, intent routing, evidence retrieval, and final issue/action
selection. Unlike an open-ended tool loop [ 21,28], retrieval, the
MCP context adapter, and rules only provide evidence; Python code
enforces hard boundaries such as evidence isolation, clarification
limits, and human-handoff rules; and the final LLM selects from
the fused evidence under these constraints.
Contributions.Our contributions to applied customer-service
LLM systems are:
•Hybrid RAG evidence construction.We report a deployed
retrieval pipeline that combines lexical and semantic FAQ recall,
RRF fusion, and cross-encoder reranking to produce auditable
FAQ evidence for customer-service decisions.
•Evidence-grounded issue/action decision.We show how
FAQ evidence, rule-derived scenario evidence, memory, and clar-
ification state can be fused before the final LLM chooses whether
to answer, clarify, or hand off, while deterministic code enforces
serving boundaries.
•Trace-driven improvement.We introduce a replay-based re-
view process that identifies whether a failure comes from recall,
reranking, final selection, clarification, or policy, and turns re-
viewed failures into KB updates, reranker hard negatives, deci-
sion preference pairs, or rule/prompt fixes.Table 2: Capability positioning against common customer-
service architectures. The table summarizes typical architec-
tural support rather than benchmark performance.
Capability RAG-only Rule-first Ours
Intent-aware routing△✓ ✓
Hybrid FAQ + rule-evidence fusion△–✓
Rule evidence in LLM decision –△✓
Bounded issue/action decision△✓ ✓
Stateful clarify / handoff△✓ ✓
Replay-based improvement△ △✓
✓: native / typical support;△: possible with add-on integration; –: not typical.
Figure 1 summarizes the replay artifacts, review outputs, and
update paths used by this process.
What we evaluate now vs. what the system supports.Section 6
evaluates evidence recall, reranking, final selection, and session-
level outcomes. It does not claim complete coverage of MCP/rule
evidence, action behavior, or downstream business outcomes.
2 Related Work
RAG grounds LLM answers in retrieved documents [ 7,11]; adap-
tive variants decide when to retrieve or critique passages [ 2,9,27],
and customer-service RAG has been deployed with structured
knowledge such as ticket knowledge graphs [ 26]. Hybrid lexical–
semantic retrieval and reranking are standard ways to improve
candidate quality [ 3,4,15,18,25]; Azure AI Search exposes a simi-
lar BM25/vector/RRF/semantic-ranker stack [ 12–14]. Our system
uses retrieval and reranking as one evidence channel rather than
the whole control surface.
Tool-augmented agents such as ReAct, Toolformer, and ToolLLM
interleave reasoning with API calls [ 16,19,28], while task-oriented
dialogue and constrained routers separate state, slots, and action
policy [ 5,22,24,29]. We adopt the production constraint that the
LLM may route intent, ask, and select an issue, but policy-guided or-
chestration governs MCP/rule-evidence access, clarification bounds,
and human handoff. This also aligns with work arguing that practi-
cal LLM systems improve through explicit memory, protocols, and
feedback harnesses rather than model weights alone [20, 30, 31].
Table 2 summarizes how the Evidence-Grounded Customer-
Service Agent Workflow combines retrieval, rule-derived evidence,
bounded decisions, and replay-based improvement.
3 Problem Setting
Given a user request 𝑞, conversation history 𝐻, and lightweight
session context 𝑢(such as evidence availability and dialogue state),
the system predicts a response action 𝑎and response content 𝑦.
The deployed system stores conversation history across turns. Serv-
ing prompts use a bounded history window, while the retained
history supports context recovery, audit, and offline trace-driven
improvement.
Actions follow task-oriented dialogue conventions [24, 29]:
•Answer: return a grounded response from a selected FAQ issue,
•Clarify: ask for necessary context when evidence is insufficient,
•Human Handoff: hand off to a human agent.
2

Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents
Figure 1: Trace-driven review and improvement loop. Batch replay surfaces failures with retrieval evidence; Human+AI review
converts them into corrected labels, preference pairs, and hard negatives; KB, reranker, decision-prompt, or rule updates are
validated on held-out replay validation splits before integration.
Non-service and policy-blocked inputs are handled by intent-routing
guardrails before FAQ selection; they are not the focus of the FAQ
issue/action evaluation.
The system must jointly satisfy:
•Factual correctnessgrounded in knowledge-base or backend
evidence,
•Business-policy compliancevia rule grounding without over-
triggering,
•Multi-turn robustnesswhen user context is incomplete,
•Evidence robustnesswhen only part of the evidence set is
available.
4 Method
4.1 Notation and Objectives
Let𝑞𝑡denote the latest user utterance at turn 𝑡,𝐻the trimmed dia-
logue history, and 𝑢lightweight user/session state such as evidence
availability and dialogue state. The intent-routing LLM outputs a
structured tuple
Γ(𝑞𝑡,𝐻,𝑢)= 𝑠𝑡, 𝜂𝑡,˜𝑞𝑡, 𝑎route,𝑡,(1)
where𝑠𝑡∈S is an intent/scene label ( |S|=7in production), 𝜂𝑡
denotes context needs used by the rule-derived evidence branch,
˜𝑞𝑡a rewritten retrieval query, and 𝑎route,𝑡 a routing signal such as a
policy guardrail or a state-qualified handoff cue. The decision-stage
LLM maps a fused evidence poolCto
Δ(C,𝑞 𝑡,𝐻)= ˆ𝜄,ˆ𝑎,(2)
with ˆ𝜄an elected FAQ issue ID and ˆ𝑎a dialogue action. The final
response text is produced by an audited response stage conditioned
on the selected issue/action and supporting evidence. Our goal is
to maximize grounded issue/action accuracy while preserving hard
guardrails (evidence-boundary isolation, clarification limits, and
human handoff) independent ofΔ.4.2 Evidence-Grounded Workflow Design
The framework is an evidence-grounded workflow implemented
as a LangGraph [ 10]StateGraph . Each non-terminal customer-
service turn invokes the LLM only through bounded workflow
stages. The logical workflow is:
Session analysis→Intent-routing LLM→{rule-derived evidence path || RAG
evidence path}→evidence fusion→final issue/action selection
•Intent-routing LLM: intent and scene routing over a deployment-
defined label set, context extraction, and routing-signal judg-
ment.
•Guardrail routing and parallel evidence paths: intent rout-
ing surfaces guardrail routes before FAQ selection. Policy-blocked
inputs can be resolved immediately; handoff remains state-qualified
and is governed by dialogue state. Otherwise, the workflow uses
context needs to access permissioned context fields through
MCP and construct rule-derived evidence in parallel with RAG
evidence from query rewriting, hybrid FAQ retrieval, weighted
RRF fusion, and reranking.
•Decision-stage LLM: issue/action selection over typed evidence
that combines general FAQ candidates with scenario-specific
rule evidence, yielding an issue ID,Clarify, orHuman Handoff.
An optional formatting stage adds a collapsible KB source to the
final answer.
In the base workflow, both stages use a shared LLM through customer-
service-specific prompts and structured output contracts; Section 6.6
separately evaluates a DPO-trained final-decision LLM as a post-
training variant. Prompts are versioned and served through the
production prompt infrastructure. Non-LLM modules—hybrid re-
trieval, MCP context access, and rule matching—supply evidence
rather than dictating the final outcome.
Figure 2 situates the workflow modules across the production ar-
chitecture, including the conversation-memory store that preserves
long-horizon interaction history outside the prompt window.
Figure 3 illustrates the presentation layer of the same evidence
pipeline: the final answer is shown as natural language, while neigh-
boring FAQ candidates can be rendered as related questions for
adjacent self-service paths.
3

Chunming Wu, Dafei Qiu, Charles Quan, Jun Wu, Suipeng Li, Mo Wu, Gavin Xie, Hope Chen, Max Yao, and Congde Yuan
Client layer Customer-Service Agent AI InfraChat UI
Session context
Workﬂow orchestrator
evidence · routing · state
Intent stage
scene · context needsEvidence layer
FAQ evidence · rule evidenceDecision stage
issue/action selection
FAQ KBMemory
historyLLM inference Rule engine MCP  service
Figure 2: Three-layer logical architecture: client channel,
Evidence-Grounded Customer-Service Agent Workflow, and
shared AI infrastructure with persistent conversation mem-
ory.
Customer-Service Agent Chat
User
I cannot withdraw my funds. What should I do?
Agent
Your withdrawal request may require additional verification or review .
Please follow the verification steps shown in the interface. After
completing the requested checks, return to the relevant page to review
the latest status and any additional instructions. If the status remains
unchanged, follow the available support guidance.
You may also want to know
• Why is my withdrawal temporarily disabled?
• What should I do if my withdrawal is under review?
• How can I check my withdrawal security review status?
• What should I do if withdrawal access is restricted?
Type your question... Send
Figure 3: User-facing customer-service chat surface. The user
sees a grounded answer and related question suggestions in
a bot-style dialog; internal FAQ issue IDs, rule evidence, and
retrieval scores are not exposed.
4.3 Workflow Overview
The session-analysis stage reads persisted session memory and
trims dialogue history to configured bounds ( 𝐻maxturns and𝐿max
characters). The turn first enters the intent-routing LLM, which
produces intent, context needs, rewritten-query, and routing sig-
nals. Depending on those signals, the workflow either resolves a
guarded path under deterministic serving conditions or constructs
two kinds of evidence in parallel: rule-derived evidence from MCP-
accessed context fields and rule-engine matches, and general FAQ
evidence from hybrid retrieval and reranking. For ordinary FAQ
Guardrail
policy / handof f1. Context
turn + memory
2. Intent routing
intent + rewrite
3a. Rule evidence
context + rules3b. RAG evidence
FAQ retrieval
4. Fusion
rule + F AQ
5. Decision
issue / clarify / handof f
6. Response
grounded answer / clarify / handof fFigure 4: Conceptual workflow from context construction
through intent routing, evidence construction, evidence-
grounded decision making, and audited response. Guardrail
outcomes may bypass issue election; otherwise rule and RAG
evidence are built in parallel, fused, and audited by the deci-
sion stage.
turns, scenario-specific rule evidence and general FAQ candidates
converge on the final issue/action selection stage.
Figure 4 summarizes the workflow node sequence: after intent
routing, state-qualified guardrail paths may be resolved before
FAQ issue selection, while rule-derived evidence and general FAQ
evidence are constructed in parallel and then converge before the
final decision LLM. The internal RAG path is expanded separately
in Figure 5. The DAG enables per-node logging and partial-evidence
preservation.
4.4 Evidence-Conditioned Workflow Routing
The intent-routing node implements Equation (1). LetAguard de-
note routing signals that may be resolved before FAQ evidence
construction, such as policy-blocked inputs or state-qualified hand-
off. The routing function is deterministic in evidence availability
and𝑎 route,𝑡 :
next(𝑢,𝑎 route,𝑡)=(
End, 𝑎 route,𝑡∈A guard,
Parallel,otherwise.(3)
Paralleldenotes the shared evidence-fusion path for ordinary customer-
service turns. FAQ recall runs in the RAG path, while the rule-
derived evidence path reads context fields through MCP and passes
matched fields to the rule engine. If no rule condition applies, the
rule-derived evidence set is empty while FAQ recall still runs. Pro-
grammatic guardrails complement LLM outputs. When context evi-
dence indicates that necessary information is missing, that evidence
is carried forward as clarification evidence rather than exposed as a
separate user-visible flow. Prior-turn intent-routing metadata is fed
back as assistant content so relevant context accumulates across
turns; scene changes reset accumulated state to avoid cross-scene
contamination.
4

Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents
4.5 Dual-Path Evidence Construction with
Rule-Derived Evidence
In the parallel evidence path, general FAQ retrieval and rule-derived
evidence construction are launched together. The latter is a scenario-
conditioned evidence-construction path: it reads context fields via
the Model Context Protocol (MCP) [ 1] and feeds them to the rule
engine:
 Efaq
𝑡,Erule
𝑡=ParallelGather HybridRetrieve( ˜𝑞𝑡),
RuleEvidence(MCP(𝑠 𝑡,𝜂𝑡)),(4)
ParallelGather returns general FAQ evidence and rule-derived
evidence under the same candidate-evidence contract. In our de-
ployment, MCP is therefore not an answer source or an autonomous
decision path: it is the context-access layer that supplies permis-
sioned fields to the rule engine.
Symbolic rule matching.Let 𝜙𝑡denote the normalized context-
field record returned through the MCP context layer at turn 𝑡. Each
policy rule𝑟defines a predicate over𝜙 𝑡:
𝑃𝑟(𝜙𝑡)=Û
𝑘EvalSub 𝑟,𝑘 𝜙𝑡,(5)
where EvalSub implements field resolvers and operators. The matched
rule set is
R𝑡={𝑟:𝑃 𝑟(𝜙𝑡)=1}.(6)
Each𝑟∈R 𝑡is converted to a KB-shaped rule-derived evidence
record and prepended as auxiliary scenario evidence:
C𝑡=RulesToEvidence(R 𝑡)∥Efaq
𝑡,(7)
where∥denotes list concatenation with rule-evidence rows first.
Rule matches are exposed as scenario evidence rather than returned
directly, so even|R𝑡|=1flows to Δin(2)rather than forcing an
immediate answer. In other words, MCP-accessed context fields are
translated into rule-derived evidence records, then judged together
with FAQ candidates by the final issue/action selection stage. A
small class of state-qualified guardrail outcomes can bypass final
FAQ election when evidence construction cannot safely resolve the
turn; these outcomes are governed by the deterministic state policy
in Section 4.7.
Unified candidate contract.The novelty is not the existence of
multiple retrievers, but the evidence contract that all candidates
must satisfy before the decision LLM can use them. Each item in C
carries: (i) a stable FAQ issue ID when it is a FAQ candidate, or a
rule-evidence identifier when it is scenario evidence, (ii) a category
such as FAQ or rule evidence, (iii) provenance fields identifying the
retrieval channel, rule ID, or scenario, (iv) rank signals such as RRF
score, reranker score, and candidate position when available, and (v)
an allowed action surface (Answer,Clarify,Human Handoff) rather
than an unconstrained response. This contract has three practical
effects. First, it makes heterogeneous evidence comparable without
forcing backend JSON into a free-form prompt. Second, it supports
partial evidence: when only a subset of retrieval, rerank, or MCP
evidence is available, the remaining candidates still satisfy the same
interface. Third, it makes the decision trace auditable because the
final issue ID and action can be tied back to an explicit candidate
record. The decision prompt therefore asks the LLM to resolve
Retrieval query
rewritten from context
BM25
lexicalIssue-title
vectorDescription
vector
Weighted RRF fusion
lexical + semantic merge
Cross-encoder reranking
base / ﬁne-tuned BGE
FAQ candidate pool
for ﬁnal decisionFigure 5: Three-channel hybrid FAQ recall with weighted
RRF and optional reranking.
ambiguity among grounded candidates, not to discover hidden
evidence sources or infer business policy from raw logs.
Unified clarification action.Clarification can be triggered by mul-
tiple evidence gaps, including missing scenario-specific context
fields or ambiguity among FAQ candidates. These evidence gaps
are carried into the final decision prompt, which decides whether to
answer with grounded evidence, ask a necessaryClarify, or follow
the human-handoff policy. Counters bound repeated clarification
turns, so the model can ask, while workflow state decides when
asking stops.
4.6 Hybrid Retrieval and Ranking
FAQ retrieval executes three channels, with BM25 and embedding
fetches started in parallel:
•BM25 over the issue title and issue description fields (fetch 150),
•KNN over issue-title embeddings (fetch 10),
•KNN over issue-description embeddings (fetch 10).
Available hits feed RRF fusion. For channel 𝑐∈{BM25,text,desc} ,
let𝐿𝑐=(𝑑 𝑐,1,𝑑𝑐,2,...) be the ranked hit list returned by OpenSearch.
BM25 uses weighted multi-field matching over the issue title and
issue description fields, with the issue title field boosted by2 ×;
KNN channels querye (˜𝑞)against issue-title and issue-description
embeddings indexed with Qwen3-Embedding-0.6B.
Figure 5 expands the RAG evidence branch in Figure 4: the rewrit-
ten query fans out to lexical and semantic channels, RRF merges
their evidence, and the cross-encoder reranker produces the FAQ
candidate pool consumed by the final issue/action selection stage.
After RRF fusion, each document 𝑑(keyed by stable FAQ ID)
receives a weighted reciprocal rank fusion score with per-channel
deduplication (a channel contributes at most once per ID):
𝑆RRF(𝑑)=∑︁
𝑐:𝑑∈𝐿 𝑐𝑤𝑐
𝐾+rank 𝑐(𝑑),(8)
5

Chunming Wu, Dafei Qiu, Charles Quan, Jun Wu, Suipeng Li, Mo Wu, Gavin Xie, Hope Chen, Max Yao, and Congde Yuan
with a fixed production configuration 𝐾=40,𝑤BM25=1.05,𝑤text=0.85,
and𝑤desc=1.0[3,4,6]. Candidates are sorted by 𝑆RRFand truncated
to the pre-rerank pool size𝑁 RRF=50.
Cross-encoder reranking.Let 𝜋RRFbe the RRF ordering. When
reranking is enabled, the reranker scores pairs (˜𝑞,doc(𝑑)) with
cross-encoder score function𝑠 CEand induces
𝜋out=argsort
𝑑∈𝜋 RRF𝑠CE ˜𝑞,doc(𝑑).(9)
Here argsort orders candidates by descending cross-encoder score.
When reranking is disabled, 𝜋out=𝜋 RRF. Final FAQ hits are Cfaq=
𝜋out[:𝑘]. Offline evaluation (E2, Section 6.5) shows that the base
BGE reranker [ 17,25] has weak top-1 ranking on business FAQ
candidates, whereas the distilled fine-tuned reranker improves
held-out business Hit@1 and MRR without changing the serving
architecture—motivating a distilled fine-tuned checkpoint in pro-
duction.
Related-question presentation.The ranked FAQ pool is also reused
for user-facing related question suggestions. After the final answer
is generated, selected neighboring candidates from the same re-
trieved and reranked evidence pool can be rendered below the
response as related questions. Internal issue IDs, issue descriptions,
replies, rule evidence, and retrieval scores are not exposed to users.
This reuse keeps the suggestion layer aligned with the same evi-
dence used by the customer-service agent while offering adjacent
self-service paths when the user’s problem is broader than the
selected answer.
4.7 State-Constrained LLM Decision Layer
The decision stage implements (2):(ˆ𝜄,ˆ𝑎)=Δ(C,𝑞 𝑡,𝐻)whereC
follows (7); when no rule-derived candidate is available, Csimply
contains the general FAQ candidates. E1 (Section 6.4) evaluates this
mapping in isolation: with the base BGE reranker, Hit@1 rises from
≈57% to≈81% when the LLM reads retrieved FAQ candidates and
rule-evidence blocks—evidence that Δshould not be replaced by
𝜋out[1]alone.
Programmatic state and human handoff.Turn metadata M𝑡aug-
ments the LLM with persistent counters:
M𝑡+1=F M𝑡, 𝑎route,𝑡,ˆ𝑎, 𝑠 𝑡,(10)
whereFupdates clarification and handoff-request state using
the current scene state 𝑠𝑡and merges accumulated context; scene
changes reset state to avoid cross-scene leakage. Human handoff is
therefore a state-qualified outcome rather than a single-turn key-
word decision: routing signals propose the need for escalation, while
deterministic state policy decides whether escalation is appropriate.
These policy thresholds are serving configuration, not experimental
claims. The session-analysis stage enforces |𝐻′|≤𝐻 maxturns and
|chars(𝐻′)|≤𝐿 maxbefore any LLM call. OnAnswer, the system
resolves reply text from the elected issue ID, optionally wraps it
with an LLM summary and collapsible KB source, and may persist
QA pairs when the audit configuration enables it.
5 System Deployment
The workflow is deployed as an online API integrated with plat-
form risk controls and observability infrastructure. The deploymentis not a prompt wrapper around an LLM: every production turn
passes through policy-guided checks for MCP/rule-evidence access,
context sufficiency, evidence construction, action selection, and
logging. We emphasize deployment lessons that complement the
offline FAQ metrics:
Channel isolation.The MCP context branch and OpenSearch
retrieval run as isolated evidence branches. When only one branch
contributes evidence, the decision stage still receives typed candi-
dates rather than a malformed prompt.
Configuration-controlled serving.Feature flags control OpenSearch
recall, KB-service routing, reranker enablement, and answer for-
matting modes. The production reranker component can use a fine-
tuned BGE checkpoint rather than the base BGE checkpoint [ 25],
based on held-out business ranking checks, public reranking bench-
marks, and B-Set replay.
Auditability.Each request emits a chain log tracing intent-routing
JSON, MCP status, rule hit list, FAQ recall scores, and final deci-
sion rationale—supporting post-hoc review in regulated customer-
service settings.
Trace-driven improvement loop.Workflow traces localize failures
to intent routing, clarification, rule-evidence coverage, rule triggers,
recall, reranking, final candidate selection, or action policy instead
of treating wrong answers as generic LLM mistakes. This attribu-
tion determines the update target: KB content and query rewriting
for recall misses, hard negatives or model choice for reranker mis-
matches, rule edits for policy mismatches, clarification-prompt or
context-handling edits for clarification failures, and action-policy
examples for handoff mistakes. E2 is one concrete instance: logs
exposed reranker head quality as the dominant bottleneck, leading
to in-domain reranker adaptation rather than backbone scaling.
The loop is intentionally conservative. Production traces do not
directly rewrite rules, prompts, or KB entries at serving time; they
produce review queues and replay sets. This is closer to a customer-
support data flywheel [ 30] than to autonomous self-modification,
preserving auditability and rollback.
Support-agent-assisted evidence annotation.After deployment,
unresolved conversations are routed to human support. We embed
evidence annotation into this support workflow rather than treat-
ing it as a separate offline labeling task. When a handoff occurs, the
support workspace shows the current user query, necessary con-
versation context, the agent’s attempted response, and an evidence
panel containing both retrieved FAQ candidates and rule-derived ev-
idence. FAQ candidates are expanded from RRF@50 and displayed
with their issue ID, issue description, and reply, so support agents
can judge whether a candidate directly resolves the user’s problem.
The annotation output is structured by update target. If a correct
FAQ is present but ranked poorly, the selected issue becomes the
positive label and the top-ranked remaining candidates are used as
hard negatives, producing reranker supervision with one positive
and seven hard negatives. If rule-derived evidence is misleading,
the trace is routed to evidence-condition supervision for updating
evidence conditions or mappings. If the evidence is available but
the final LLM decision is wrong, the trace becomes decision-level
supervision: the corrected issue ID orClarifydecision can serve as
6

Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents
the chosen response, while the wrong issue ID or unnecessary clari-
fication becomes the rejected response for future preference-tuning
studies. Human handoff is treated more conservatively because it is
primarily governed by policy and conversation state. If no evidence
covers the query, the trace is marked as a recall miss or knowledge
gap and is routed to retrieval improvement or FAQ maintenance.
All updates are validated through replay before integration.
Safety gates.Input and output guards, URL risk filtering, and im-
age moderation run at API boundaries independent of the retrieval
pipeline.
6 Experiments
6.1 Evaluation Scope and Claims
This paper is anApplied Data Scienceaccount of a deployed
Evidence-Grounded Customer-Service Agent Workflow. The method
sections describe the full DAG, while four evaluation blocks isolate
the components that can be labeled reliably: pipeline diagnosis (E1),
reranker optimization (E2), B-Set session validation (E3), and final-
decision preference optimization (E4). Table 3 maps each block to
its protocol and claim. Clarification precision, rule-trigger precision,
MCP/rule-evidence coverage, and full action accuracy are not yet
comprehensively labeled in the offline set.
Human-AI evaluation workflow.The evaluation process follows a
human-AI collaborative workflow: automated replay jobs generate
candidate lists, rerank traces, and preliminary error attribution,
while human reviewers define valid labels, inspect boundary cases,
and approve the final B-Set review rubric. Thus the paper reports
not only model scores, but also the trace evidence that explains
which subsystem should be changed next.
6.2 Evaluation Blocks and Headline Metrics
Table 3 separateswhich evidence supports which claim, so
diagnostic ablations, public benchmark checks, B-Set review, and
final-decision LLM post-training are not collapsed into a single
score. Detailed Hit@ 𝐾curves are reported in later tables; this table
is a roadmap for the evaluation blocks.
6.3 Shared Dataset and Protocol
Data roles.Table 4 summarizes the data used in the evaluation.
For offline development, we curate a historical labeled-trace sample
containing 605 anonymized dialogue sessions and 1,427 turns. Af-
ter cleaning, deduplication, and FAQ-issue filtering, its T-Set FAQ
subset contains 581 turn-level FAQ queries: 492/52/37 are used for
reranker train/development/held-out test, the 544 non-test queries
support module-level ablations, and a 309-query non-test diagnostic
subset supports early controlled pipeline diagnosis. The 309-query
diagnostic subset is used only for early pipeline diagnosis and is
not treated as an independent held-out evaluation set. B-Set is a
sealed multi-turn holdout for KB-grounded session review.
Query construction.User utterances are machine-translated
to English. Forretrieval, all user turns in a session are concate-
nated into a single query string (chronological order) and passed
to three-channel hybrid recall, which returns up to 50 FAQ candi-
dates. Fordecision, the decision-stage LLM receives only thelastuser turn as the active query, together with the recalled candi-
date list—mirroring production behavior where the latest utterance
disambiguates intent while prior turns inform recall breadth.
Decision models in E1.The diagnostic comparison uses Qwen3.5-
27B and Azure GPT-4o under the same pipeline setting. Both emit
structured JSON containing the selected issue ID, action, and ratio-
nale. Reranking uses the off-the-shelf BGE reranker checkpoint and
reorders the recall top-50 before LLM candidate selection overtop-
20candidates. A top-10 diagnostic run is used only to understand
candidate-window sensitivity.
Reranker protocol in E2.Hybrid recall returns top-50 candi-
dates, which are rescored by a cross-encoder reranker over FAQ is-
sue, description, and reply text. We compare the base BGE reranker
against supervised and teacher-student distilled fine-tuned check-
points trained on query–FAQ pairs: each training query has one
positive FAQ labeled by its gold issue ID and negatives sampled
from the recall pool. Generalization is checked on the cleaned held-
out business split, public C-MTEB tasks, and B-Set review. Addi-
tional cleaned-data variants are kept in the experiment archive and
reported only when they clarify forgetting risk.
Metric.We reportHit@ 𝐾: the fraction of cases whose gold issue
ID appears within the top- 𝐾ranked candidates at a given pipeline
stage. Hit@1 at the LLM stage isend-to-end issue ID accuracy—
the model’s elected issue ID matches the label. At recall, Hit@50
indicates whether the gold issue appears anywhere in the retrieved
pool (recall ceiling before reranking/LLM).
B-Set profile.Because the deployed system predicts both FAQ
issue IDs and dialogue actions, the B-Set is not a pure single-label
FAQ-matching benchmark. Table 5 reports the dataset-level profile
used for end-to-end validation. Ranking denominators are reported
separately in Table 12, because ranking metrics require turns with
FAQ issue targets and available retrieval/rerank evidence.
Early fixed-evidence prompt replay.Table 6 reports the early loop
used to settle the final-decision prompt before the main B-Set eval-
uation; it is an auxiliary development check, not a B-Set headline
result. The scoped loop follows the same engineering pattern used
elsewhere in our replay workflow: define the fixed-evidence deci-
sion rubric, set the goal of improving final-decision accuracy with-
out regressions, replay the base prompt, attribute old-fail/new-pass
and old-pass/new-fail cases, apply small human-approved prompt
patches across rounds, rerun the same subset, and retain the best
approved prompt. The replay subset is sampled from the curated
historical T-Set data and freezes query context, rewritten query,
top-10 FAQ candidates, expanded FAQ text, early-exit decisions,
and the grader; only the final-decision prompt changes.
6.4 E1: Pipeline Diagnosis
Purpose.E1 answers the first diagnostic question:is low accuracy
caused mainly by the decision LLM, or by the evidence it receives?We
combine three controlled views: an early T-Set non-test diagnostic
subset, T-Set retrieval/reranking ablation, and a decision-model
comparison. The goal is to localize the bottleneck before changing
the reranker, prompt, or generator.
Table 7 checks the retrieval/ranking stack on the 544 non-test
T-Set FAQ queries, excluding the held-out Test37 split. It shows
7

Chunming Wu, Dafei Qiu, Charles Quan, Jun Wu, Suipeng Li, Mo Wu, Gavin Xie, Hope Chen, Max Yao, and Congde Yuan
Support Annotation W orkspaceReview queue evidence annotation
Conversation
User
I tried to withdraw an asset yesterday .
Agent
Please confirm whether the withdrawal is
pending or rejected.
User
It says reviewing after I passed the
checks.
Agent
Ask whether the user means a pending
withdrawal, a failed withdrawal, or an
account restriction.
User
I completed verification, but my withdrawal
is still under review . What should I do?
Type a reply ... SendEvidence Candidates
Expanded F AQ candidates from RRF@50 Select the best positive label
Rank FAQ issue description reply Use
1 Issue-A Candidate answer temporary restriction restore-access
guidance□
2 Issue-B Candidate answer pending review after checks review status guidance ✓
3 Issue-C Nearby candidate on-chain pending state partial evidence only □
4 Issue-D Nearby candidate identity check state partial evidence only □
Rule-derived evidence
context fields permissioned review state / relevant context
matched rule review-status condition as evidenceReview Output
Reviewed outcome
● FAQ answer
○ clarify
○ human handof f
Reranker training sample
positive: Issue-B
negative: top unselected F AQ candidates
Decision preference pair
✓ chosen
Your withdrawal is under review . Follow the
in-app review notice and wait for the result.
✕ rejected
Please clarify whether the withdrawal is
pending, failed, or restricted.
Reviewer reason
The query already contains enough context;
direct F AQ answer is valid.
Save review Replay check
Figure 6: Anonymized support-agent workspace with evidence annotation. Support agents review the masked user query,
conversation context, the agent’s attempted response, RRF@50 FAQ candidates, and rule-derived evidence. The same workspace
produces reranker supervision, evidence-condition fixes, decision-level preference pairs, or recall/FAQ update signals, all
validated by replay before integration.
Table 3: Canonical evidence map for the stepwise evaluation. Detailed Hit@ 𝐾tables are reported in the corresponding E1/E2/E3
subsections.
Block Question Dataset / protocol Key result Use in paper
E1 Pipeline diagnosis Is the baseline bottleneck retrieval,
reranking, or the generator?T-Set non-test diagnostic subset
(𝑁=309) and module ablation
(𝑁=544).Recall top-50 is high; base
reranker top-1 accuracy is weak;
diagnostic issue-election differs
by<1pp across Qwen/GPT-4o.Locate evidence loss
before changing models.
E2 Reranker
optimizationDoes distilled reranker tuning
improve evidence ordering without
obvious benchmark regression?T-Set 492/52/37 split and selected
six-task C-MTEB proxy.Test37 Hit@1 improves
56.76→75.68; C-MTEB Avg.
reaches 66.58.Select the distilled
reranker and monitor
forgetting risk.
E3 B-Set validation Does better evidence improve
user-facing sessions?B-Set, 200 multi-turn sessions. Base reranker 86.5%; distilled
reranker 88.5% KB-grounded
session accuracy.Headline session-level
validation.
E4 Decision-stage DPO Can final-decision post-training
improve issue/clarification election
under a fixed evidence path?B-Set with intent routing,
retrieval/RRF, candidate format, and
review rubric fixed.Isolated base-reranker + DPO
reaches 90.5%; best combined
distilled-reranker + DPO reaches
92.5%.Isolate final-decision
LLM optimization; report
best system
configuration.
why the production path uses hybrid recall, RRF fusion, and a cross-
encoder reranker. Table 8 then reports Hit@ 𝐾curves on the 309-
query T-Set non-test diagnostic subset and compares issue Hit@1
across decision backbones: under this diagnostic pipeline, the <1pp
gap between Qwen3.5-27B and GPT-4o suggests that raising end-to-
end accuracy on this benchmark requires better evidence ordering,
not a larger generator alone.RRF and reranker configuration.We set RRF to 𝐾=40with chan-
nel weights(𝑤BM25,𝑤text,𝑤desc)=( 1.05,0.85,1.00)and fetch sizes
150/10/10for BM25, issue-title vector, and issue-description vector
recall. Query and indexed FAQ vectors use Qwen3-Embedding-0.6B.
This setting reaches the highest T-Set top-50 coverage among the
representative configurations we tested; BM25-only, dense-only,
smaller candidate budgets, and over-expanded dense fetch all reduce
8

Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents
Table 4: Dataset roles used throughout the paper. T-Set FAQ rows are turn-level issue queries; B-Set is session-level validation
and also contains action labels.
Set Sessions / cases Turns / queries FAQ / issue subset Use / caveat
T-Set curated sample 605 1,427 581 Curated historical labeled traces; the FAQ subset is used for reranker
work.
T-Set non-test diagnostic subset — 309 309 Early FAQ-issue diagnosis; non-test development subset.
T-Set non-test ablation subset — 544 544 Module-level recall/ranking ablations; excludes the held-out Test37
split.
T-train — 492 492 Reranker training; issue-only, not an action-evaluation set.
T-dev — 52 52 Reranker development and checkpoint selection.
T-test — 37 37 Held-out business FAQ reranker test.
B-Set 200 632 409 ranking-evaluable issue
turnsSealed session-level validation; evaluates FAQ, clarification, and hand-
off behavior.
Table 5: B-Set dataset profile. B-Set is used for KB-grounded
session review; issue-ranking denominators are reported sep-
arately because ranking metrics apply only to turns with FAQ
issue targets and retrieval/rerank evidence.
Set Sessions Turns Avg. turns Validation role
B-Set 200 632 3.16 Sealed multi-turn holdout with FAQ,
clarification, and handoff labels; session
accuracy requires all turns in a session to
pass.
Table 6: Early fixed-evidence prompt replay on a T-Set de-
velopment subset (200sessions /578turns). Only the final-
decision prompt changes; results are auxiliary rather than
B-Set headline metrics.
Round / applied patch Turn Acc. (%) Session Acc. (%)
round0 / base prompt 89.8 79.0
round1 / direct matching 92.4 83.0
round2 / context and parent FAQ 92.483.5
round3 / unsupported-claim guardrail92.6 83.5
Table 7: T-Set module-level recall and ranking ablation
(𝑁=544non-test turn-level FAQ queries labeled with gold FAQ
issue IDs; Test37 excluded). Dense recall uses issue-title and
issue-description vectors; RRF fuses BM25, issue-title-vector,
and issue-description-vector channels before cross-encoder
reranking.
Variant @1 @5 @10 @20 @50 MRR
BM25 lexical recall 41.9 70.0 82.2 88.4 95.0 0.5434
Dense recall 30.3 61.9 76.5 88.8 88.8 0.4492
RRF fusion 41.2 74.1 86.8 94.199.30.5532
Base BGE reranker59.0 89.2 95.2 97.6 99.3 0.7181
top-50 coverage or add cost without improving the downstream
candidate pool. For the cross-encoder base, we select BGE-reranker-
large: with full FAQ payloads and a shared 500-token budget, it
reaches 95.2% Hit@10 and 97.6% Hit@20 on T-Set, while improv-
ing MRR from the RRF-only 0.5532 to 0.7181. BGE-reranker-v2-m3
improves Hit@20 only slightly (98.3%) and shows higher latency
risk when allowed to use its longer native context window.
Stage-wise interpretation.Table 8 decomposes where error re-
mains under abase BGEreranker:Table 8: Issue ID Hit@ 𝐾by pipeline stage on the 𝑁=309T-Set
non-test diagnostic subset. Recall: concatenated multi-turn
user queries. LLM: last user turn only.
Setting Stage @1 @5 @10 @15 @20 @50
Base BGE/Qwen Hybrid recall 42.07 74.43 84.47 88.35 90.29 96.76
Base BGE/Qwen Rerank (top-50) 56.31 84.14 91.59 94.50 95.15 —
Base BGE/Qwen LLM (Qwen3.5-27B)81.23— — — — —
Base BGE/GPT-4o Hybrid recall 42.53 74.03 83.77 87.66 89.94 96.75
Base BGE/GPT-4o Rerank (top-50) 56.82 83.77 91.56 94.48 95.13 —
Base BGE/GPT-4o LLM (Azure GPT-4o)81.88— — — — —
•Recall (≈42% Hit@1,≈97% Hit@50):hybrid fusion finds the
gold FAQ in the top-50 pool for most cases but rarely at rank 1—
the system should not terminate at recall.
•Reranking (≈57% Hit@1):generic cross-encoder reranking
helps (+15pp) yet≈44% of cases still have a wrong top-1 candi-
date.
•Candidate election ( ≈81% Hit@1):the decision-stage LLM
reads candidate summaries and the latest user turn, recovering
+25pp over reranking—primary offline evidence that the decision
flow should not stop at retrieve-then-rank when top-1 ranking
is weak.
Under the diagnostic pipeline, replacing Qwen3.5-27B with GPT-4o
changes final issue-election accuracy by less than 1pp, suggesting
that backbone scaling alone is not the dominant bottleneck in this
setting.
Candidate window.The early T-Set non-test diagnostic subset
used top-20 evidence to test whether LLM candidate selection can
recover from weak top-1 ranking. A top-10 diagnostic run un-
der the same base reranker lowers decision Hit@1 from 81.23%
to 78.96%, showing that ranks 11–20 can compensate for a weak
generic reranker. After reranker adaptation, the compact top-10
window is preferred for serving because the T-Set and Test37 re-
sults show that the gold FAQ is already usually inside the top-10
evidence window.
6.5 E2: Teacher-Student Reranker Fine-Tuning
Purpose.E2 is the intervention suggested by the funnel diagnosis.
Since decision LLMs cannot fully compensate for weak top-1 ranks
when≈44% of top-1 rerank results are wrong, we ask whether
a reranker fine-tuned on in-domain FAQ pairs can improve top-
rank ordering on held-out business supervision—withoutinvolving
9

Chunming Wu, Dafei Qiu, Charles Quan, Jun Wu, Suipeng Li, Mo Wu, Gavin Xie, Hope Chen, Max Yao, and Congde Yuan
Table 9: Internal held-out reranker test on the same 37 busi-
ness FAQ groups. The saturated Hit@10 supports using top-
10 as the compact FAQ evidence window; Hit@1 and MRR
show whether the gold FAQ is ranked early enough to reduce
downstream ambiguity.
Reranker @1 @5 @10 @20 MRR
Base BGE 56.76 86.49 97.30 97.30 0.6991
Supervised fine-tuned reranker 64.86 86.49 94.59100.000.7391
Distilled fine-tuned reranker75.68 91.8997.30 97.300.8215
the LLM—and at what cost to public reranking benchmarks. Af-
ter training, we evaluate each checkpoint on (a) public reranking
benchmarks and (b) the cleaned 492/52/37 train/development/test
business split.
Training-data cleaning.Each reranker supervision group con-
tains one positive FAQ and seven hard negatives sampled from
the recall pool. The teacher-student run starts from 506 candidate
training groups and removes 14 groups before training, leaving
492 training groups with the same 52/37 development/test split.
Removed groups are cases where the query and positive FAQ are
visibly mismatched, the teacher margin is too low to provide a
stable preference, multiple unrelated intents are forced into one
positive, or the label comes from a weak human-correction path
rather than a high-confidence FAQ target. We do not promote the
14 removed groups to a test set because their labels are judged
unreliable; uncleaned runs are retained in the experiment archive
as sensitivity controls.
Teacher-score distillation.The distilled reranker is initialized
from the same BGE checkpoint as the supervised run. For each
query and recalled candidate set, the teacher provides soft rele-
vance scores; the student is optimized with the supervised pos-
itive/negative ranking objective plus a teacher-score distillation
term, so that in-domain supervision improves FAQ ordering with-
out discarding the teacher’s broader ranking preferences.
Internal held-out rerank test.Table 9 evaluates the base BGE,
supervised fine-tuned, and distilled fine-tuned rerankers on the
same held-out Test37 split. Although this FAQ test split is small, it
is strictly separated from training and development; we use it as
a controlled internal ranking check, while B-Set provides session-
level validation. We use top-10 as the production-style FAQ evidence
window because the base BGE and distilled fine-tuned rerankers
both reach 97.30% Hit@10, indicating that a wider window is un-
likely to add much candidate coverage on this split. The remaining
improvement should therefore come fromtop-rank ordering: the
distilled fine-tuned reranker preserves the same Hit@10 while im-
proving Hit@1 from 56.76% to 75.68% and MRR from 0.6991 to
0.8215.
General benchmark sanity check.We also checked six C-MTEB
reranking tasks to avoid choosing a checkpoint that only memo-
rizes the business set. The distilled fine-tuned reranker is trained
on the cleaned 492 groups; the development split is used only for
model selection, and the held-out business test result is reported in
Table 9. Table 10 keeps only the comparison needed for the paper:Table 10: Selected six-task C-MTEB reranking
MAP/main_score comparison. Older and partial checkpoints
are omitted from the paper table and retained in logs.
Model T2 Zh2En En2Zh MMarco CMed1 CMed2 Avg.
Official BGE reranker 67.60 64.03 61.44 37.16 82.15 84.18 66.09
Supervised fine-tuned reranker 67.55 64.70 62.93 33.94 81.11 83.51 65.62
Distilled fine-tuned reranker67.55 64.89 63.37 38.27 81.80 83.59 66.58
official/base scores, the deployed supervised fine-tuned reranker,
and a distilled fine-tuned reranker that best preserves public rank-
ing ability. The supervised fine-tuned reranker slightly lowers the
six-task average relative to the public base model, while the distilled
fine-tuned reranker recovers and exceeds the base average.
We use these public tasks as forgetting probes, not target-domain
objectives. Teacher-score distillation improves the C-MTEB average
from 66.09 to 66.58, suggesting that it preserves general rerank-
ing quality under this selected six-task check while retaining the
business-domain top-ranking gains.
Implications.Teacher-score distillation reduces observed forget-
ting risk on the selected public reranking proxy: after cleaning the
training supervision from 506 to 492 groups, the distilled fine-tuned
reranker improves the six-task C-MTEB average from 66.09 to 66.58
while keeping the business test-set Hit@10 at 97.3%. We therefore
treat distillation as the preferred checkpoint recipe, while using
B-Set review to decide which checkpoint is best for end-to-end
customer-service quality.
6.6 E3: End-to-End B-Set Evaluation
Purpose.E3 validates whether reranker and evidence-quality changes
improve user-facing session quality. Ranking metrics alone are in-
sufficient for production acceptance because a customer-service
agent may need to answer with a FAQ, ask a necessary clarification,
or follow an action policy depending on the dialogue context. We
therefore use aB-Setholdout of 200 multi-turn sessions that was
not used for reranker training, and report both issue-turn ranking
quality and KB-grounded session-level review.
Review protocol.The B-Set gold labels are produced by a KB-
grounded review process over current user query, preceding user
turns, maintained FAQ issue title, issue description, reply text, and
action policy. A turn is correct if the final output selects a directly
applicable FAQ, asks a necessary clarification when information is
insufficient, or follows the appropriate dialogue action. A clarifica-
tion is rejected when the KB already provides a precise answer; an
FAQ answer is rejected when it is off-topic, over-specific, or ignores
required session context. Session accuracy requires every turn in
the session to pass under the fixed review rubric.
Issue-turn ranking quality.Table 12 isolates the ranking-evaluable
FAQ issue turns inside B-Set, namely turns with gold FAQ issue
targets and available retrieval/rerank evidence. Base and distilled
configurations use the same RRF candidate pool, so differences after
RRF are attributable to reranking.
Rerank-only contrast.Table 13 gives inspected examples behind
the aggregate B-Set ranking gains. On this shared candidate pool,
10

Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents
Table 11: B-Set multi-turn evaluation. The headline metric is
KB-grounded session accuracy on 𝑁=200sessions; the same
review rubric is applied to all systems. Values are point esti-
mates, and we do not claim statistical significance. The base-
reranker + DPO row isolates final-decision post-training,
while the distilled-reranker + DPO row reports the best com-
bined system configuration.
System Reranker KB-grounded
session acc.
Base reranker + base decision LLM Base BGE reranker 86.5
Distilled FT reranker + base decision
LLMDistilled FT
reranker88.5
Base reranker + DPO final-decision
LLMBase BGE reranker 90.5
Distilled FT reranker + DPO
final-decision LLMDistilled FT
reranker92.5
B-Set session accuracy by optimization axis
8688909294
86.588.590.592.5
+2.0 pp+2.0 pp
Base decision LLM DPO decision LLMBase reranker
Distilled reranker
Same B-Set review rubric; values are point estimates.
Figure 7: B-Set session accuracy under a two-axis optimiza-
tion design. The grouped bars separate the reranker axis from
the final-decision LLM axis under the same KB-grounded
session review rubric over 200 sessions; issue-turn ranking
metrics are reported separately in Table 12.
Table 12: B-Set issue-turn ranking quality under the same
RRF candidate pool. Values are percentages over409ranking-
evaluable gold issue turns.
Reranker RRF@50 RRF@10 Rerank@1 Rerank@5 Rerank@10
Base 98.3 93.4 69.4 91.9 96.6
Distilled FT 98.3 93.4 79.2 95.4 97.8
the distilled fine-tuned reranker moves five inspected FAQ targets
from outside the base reranker Top-10 into Top-10, with no reverse
Top-10 regression in the inspected differences. Three gains translate
into corrected final FAQ decisions; the remaining gains improve the
evidence set, while the final output LLM still chooses clarification
under the dialogue policy. This qualitative inspection illustrates
where the ranking gain helps without turning the reranker table
into an end-to-end decision metric.
6.7 E4: Decision-Stage Preference Optimization
We further evaluate whether replay-derived preference pairs can
improve the final issue/action selection step. This run keeps the
B-Set file, intent-routing stage, context construction, retrieval/RRF
pipeline, base reranker, candidate format, review rubric, and servingTable 13: Inspected B-Set examples where the distilled
reranker improves FAQ evidence ordering. Queries are lightly
paraphrased for readability; the table is qualitative support
for Table 12, not a separate headline metric.
Ex. Representative user query Base Distilled Downstream effect
1What is the fee for futures
trading on a specific pair?miss@10 rank2 Final FAQ decision be-
comes correct
2My withdrawal is still under
review; can I cancel it?miss@10 rank2 Final FAQ decision be-
comes correct
3How can I get an income-tax
report?miss@10 rank1 Final FAQ decision be-
comes correct
4My account verification shows
an unusual status.miss@10 rank9 Evidence recovered; policy
still clarifies
5Can my KYC/KYB review be
extended?miss@10 rank10 Evidence recovered; policy
still clarifies
Table 14: Anonymized B-Set turns where the DPO final-
decision LLM corrects the baseline decision. Queries are
lightly paraphrased; the table illustrates decision-boundary
types rather than adding a separate headline metric.
Boundary type Anonymized user query Baseline behavior DPO behavior
Wrong issue→
clarifyMy asset was in the trading account,
then I transferred it to funding, and
when I checked again it disappeared.Over-specific
transfer / asset FAQClarify
Wrong clarify→
issueUnfreeze my withdrawal. Generic
clarificationWithdrawal
restriction FAQ
Wrong issue→
better issueIt says the required credential uses a
different format, but I see another code
in my account. Should I use that?Anti-phishing-code
FAQTrading-
password FAQ
policy fixed; the only changed component is the final-decision
LLM used by the Evidence-Grounded Decision Module. The DPO
data contains 117 high-confidence preference pairs mined from
development replay failures, not from the sealed B-Set validation
set: 53wrong-issue-to-clarify, 38wrong-issue-to-issue, and 26wrong-
clarify-to-issuepairs. No B-Set turn or session is used for DPO
preference pairs or checkpoint selection. Thus the preference signal
is not a one-sided push toward answering with more issue IDs. It
teaches three decision boundaries: reject an FAQ when it is too
narrow and ask a useful clarification, avoid unnecessary generic
clarification when a suitable FAQ candidate exists, and replace
a mismatched FAQ with a more applicable issue. Table 14 gives
anonymized examples of these boundary types.
Under the same B-Set review rubric, the resulting base-reranker
+ DPO final-decision configuration reaches 612/632 turn hits and
181/200 session hits (90.5%). Combining the distilled reranker with
the DPO final-decision LLM yields the best B-Set session accuracy
of 92.5%, but this combined row is reported as a system configura-
tion rather than as the isolated DPO effect. This isolates a second
optimization axis: after evidence quality is stable, decision-stage
post-training can further calibrate the boundary between grounded
FAQ answers and necessary clarifications.
Interpretation.The distilled fine-tuned reranker is therefore eval-
uated as an evidence-ordering change, not as a new answer gen-
erator. It improves top-rank ordering substantially on issue turns,
while the final output LLM may still choose clarification when the
session context is insufficient. The deployable claim is:under the
same B-Set file and the same KB-grounded review rubric, the distilled
11

Chunming Wu, Dafei Qiu, Charles Quan, Jun Wu, Suipeng Li, Mo Wu, Gavin Xie, Hope Chen, Max Yao, and Congde Yuan
Table 15: Stepwise accuracy-improvement chain. Each row
states the question asked, the evidence used, and the resulting
design decision.
Step Question / evidence Design decision
Diagnose baseline E1: recall has high top-50
coverage, base BGE reranker
is weak at rank 1, and LLM
candidate selection recovers
to≈81%;
BM25/vector/RRF/rerank
ablations identify where
evidence is lost.Keep controlled candidate
selection; do not rely on
reranker top-1 or generator
scaling alone.
Improve evidence E2: the distilled fine-tuned
reranker trained on cleaned
supervision improves
held-out business
Hit@1/MRR, reaches 97.3%
test-set Hit@10, and
improves C-MTEB Avg. to
66.58.Move the main optimization
upstream into the RAG evidence
layer while monitoring
forgetting.
Improve decision
promptEarly fixed-evidence prompt
replay: turn accuracy
improves from 89.8% to
92.6%, and session accuracy
from 79.0% to 83.5%.Treat prompt edits as
replay-verified development
patches, not headline B-Set
results.
Review user
outcomeE3: B-Set shows base
reranker 86.5% and distilled
fine-tuned reranker 88.5%
KB-grounded session
accuracy.Report KB-grounded session
review as the headline
end-to-end score; use reviewed
traces to update labels, KB rules,
and guardrails.
Optimize final
decisionE4: with the base reranker
fixed, DPO final-decision
LLM reaches 90.5% B-Set
session accuracy.Use replay-derived preference
pairs to tune
issue-vs-clarification boundaries
after the evidence path is stable.
Remaining gap Comprehensive context,
rule-evidence, and action
accuracy remains planned.Extend the same trace-driven
method to action, context, and
rule metrics after launch.
fine-tuned reranker improves user-facing multi-turn session accuracy
while preserving workflow guardrails against over-answering.
6.8 Post-Launch Operational Check
We additionally examine reviewed accuracy after deployment. We
draw two equally sized operational samples, each containing 1,508
dialogue sessions, from the Agent and legacy RAG-only entry paths,
respectively. Each consultation is counted as one independent ses-
sion, and both samples are evaluated under the same session-level
operational-review rubric. The Agent workflow achieves 89.52% re-
viewed accuracy, compared with 79.00% for the legacy workflow, an
observed difference of 10.52 percentage points. This result comple-
ments rather than replaces the controlled B-Set evaluation. Because
the operational samples come from separate production entry paths
rather than randomized assignment or matched sessions, we treat
the comparison as descriptive operational evidence.
6.9 Synthesis Across Evaluation Blocks
Table 15 summarizes the controlled experiments as a sequence of
diagnostic questions and design decisions: diagnose evidence loss
(E1), improve evidence ordering (E2), validate session quality (E3),
and optimize the final-decision LLM under a fixed evidence path
(E4).7 Discussion
The main novelty is not a new retriever, a larger chat model, or
an unconstrained agent loop. It is a deployment pattern in which
LLMs participate in customer-service decisions while deterministic
workflow code retains authority over evidence access and policy
boundaries. Hybrid RAG evidence construction keeps execution
boundaries in code; evidence-grounded decision making turns re-
trieval results, scenario-specific rule evidence, clarification, and
human handoff into auditable evidence/actions; trace-driven im-
provement turns traces into concrete update targets rather than a
generic “bad answer” bucket. This explains why backbone replace-
ment changes less than 1pp under the E1 diagnostic pipeline, while
reranker adaptation is evaluated as an evidence-quality change
whose user-facing value must be confirmed by B-Set review.
The trade-off is reduced autonomy. The benefit is operational
control: traces reveal whether a failure came from evidence, ranking,
rules, clarification, or handoff policy before an update is selected. E3
also shows why customer-service evaluation needs KB-grounded
session review: valid outcomes may include answering with a FAQ,
asking a necessary clarification, or following the dialogue policy.
8 Limitations and Future Work
Experiments focus on one financial customer-service domain and
use curated internal labels rather than web-scale retrieval suites
such as BEIR [ 23]. The B-Set adds a 200-session end-to-end holdout
with KB-grounded human review, but it is still single-domain and
manually reviewed. Because user utterances are machine-translated
to English before retrieval and evaluation, translation errors may af-
fect recall and decision quality; multilingual end-to-end evaluation
is left for future work. Scenario-specific rule evidence, clarification,
and dialogue-action labels are deployed workflow behavior but not
yet comprehensively labeled in the offline set. Controlled headline
metrics follow E1/E2/E3/E4 (Table 3); reranker generalization is
supported by the cleaned teacher-student split, public C-MTEB
checks, and the B-Set review [ 25]. The separate post-launch com-
parison uses equally sized samples from different production entry
paths under the same rubric; it is descriptive rather than a random-
ized causal estimate. We do not claim autonomous “agent” bench-
mark performance [ 21]; the contribution is an Evidence-Grounded
Customer-Service Agent Workflow [ 5,10,31] with explicit eval-
uation boundaries. Generalization beyond financial-platform cus-
tomer service remains to be tested. Future work includes broader
replay evaluation, clarification/action/rule-evidence metrics, cross-
domain transfer, and formal analysis of candidate fusion under
noisy rule triggers.
Ethics Statement
Customer-service automation should include explicit human-handoff
safeguards, policy-compliance checks, and logging for post-hoc au-
ditability. Sensitive user attributes should be minimized in retrieval
features and logs. Automated decisions with high user impact re-
quire human-in-the-loop review paths.
References
[1] Anthropic PBC. 2024. Model Context Protocol. https://modelcontextprotocol.io/
docs/getting-started/intro. Accessed June 2026.
12

Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents
[2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InInternational Conference on Learning Representations. https://openreview.net/
forum?id=hSyW5go0v8
[3] Sebastian Bruch, Siyu Gai, and Amir Ingber. 2024. An Analysis of Fusion Func-
tions for Hybrid Retrieval.ACM Transactions on Information Systems42, 1, Article
20 (2024), 20:1–20:35 pages. doi:10.1145/3596512
[4] Tao Chen, Mingyang Zhang, Jing Lu, Michael Bendersky, and Marc Najork. 2022.
Out-of-Domain Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models. In
Advances in Information Retrieval: 44th European Conference on IR Research, ECIR
2022 (Lecture Notes in Computer Science, Vol. 13185). 95–110. doi:10.1007/978-3-
030-99736-6_7
[5] Willy Chung, Samuel Cahyawijaya, Bryan Wilie, Holy Lovenia, and Pascale Fung.
2023. InstructTODS: Large Language Models for End-to-End Task-Oriented
Dialogue Systems. InProceedings of the Second Workshop on Natural Language
Interfaces. 1–21. doi:10.18653/v1/2023.nlint-1.1
[6] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher. 2009. Reciprocal
Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.
InProceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval. 758–759. doi:10.1145/1571941.1572114
[7] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Gen-
eration for Large Language Models: A Survey.arXiv preprint arXiv:2312.10997
(2024). doi:10.48550/arXiv.2312.10997
[8] Artur d’Avila Garcez and Luis C. Lamb. 2023. Neurosymbolic AI: The 3rd Wave.
Artificial Intelligence Review56, 11 (2023), 12387–12406. doi:10.1007/s10462-023-
10448-w
[9]Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers). 7036–7050. doi:10.18653/v1/2024.
naacl-long.389
[10] LangChain AI. 2024. LangGraph: Build Stateful, Multi-Actor Applications with
LLMs. https://langchain-ai.github.io/langgraph/. Accessed June 2026.
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Infor-
mation Processing Systems, Vol. 33. 9459–9474. https://proceedings.neurips.cc/
paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[12] Microsoft. 2026. Azure AI Search: Hybrid Search. https://learn.microsoft.com/en-
us/azure/search/hybrid-search-overview. Accessed 2026-06-16.
[13] Microsoft. 2026. Azure AI Search: Reciprocal Rank Fusion. https://learn.microsoft.
com/en-us/azure/search/hybrid-search-ranking. Accessed 2026-06-16.
[14] Microsoft. 2026. Azure AI Search: Semantic Ranking. https://learn.microsoft.
com/en-us/azure/search/semantic-search-overview. Accessed 2026-06-16.
[15] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT.
arXiv preprint arXiv:1901.04085(2019). doi:10.48550/arXiv.1901.04085
[16] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin,
Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian,
Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong
Sun. 2024. ToolLLM: Facilitating Large Language Models to Master 16000+
Real-world APIs. InInternational Conference on Learning Representations. https:
//openreview.net/forum?id=dHng2O0Jjr
[17] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP). 3982–3992. doi:10.18653/v1/
D19-1410
[18] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends in Information Retrieval3, 4
(2009), 333–389. doi:10.1561/1500000019
[19] Timo Schick, Jane Dwivedi-Yu, Roberto Dessí, Roberta Raileanu, Maria Lomeli,
Eric Hambro, Luke Zettlemoyer, Nicolas Cancedda, and Thomas Scialom. 2023.
Toolformer: Language Models Can Teach Themselves to Use Tools. InAdvances in
Neural Information Processing Systems, Vol. 36. https://arxiv.org/abs/2302.04761
[20] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik
Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal
Reinforcement Learning.arXiv preprint arXiv:2303.11366(2023). doi:10.48550/
arXiv.2303.11366
[21] Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei, and Athanasios V.
Vasilakos. 2025. Agentic Retrieval-Augmented Generation: A Survey on Agentic
RAG.arXiv preprint arXiv:2501.09136(2025). doi:10.48550/arXiv.2501.09136
[22] Yiyou Sun, Junjie Hu, Wei Cheng, and Haifeng Chen. 2024. DFA-RAG: Con-
versational Semantic Router for Large Language Model with Definite Finite
Automaton. InProceedings of the 41st International Conference on Machine Learn-
ing (Proceedings of Machine Learning Research, Vol. 235). PMLR, 47033–47055.https://proceedings.mlr.press/v235/sun24e.html
[23] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
Information Retrieval Models. InProceedings of the Neural Information Processing
Systems Track on Datasets and Benchmarks, Vol. 1. https://openreview.net/forum?
id=wCu6T5xFjeJ
[24] Tsung-Hsien Wen, David Vandyke, Nikola Mrkšić, Milica Gašić, Lina M. Rojas-
Barahona, Pei-Hao Su, Stefan Ultes, and Steve Young. 2017. A Network-based
End-to-End Trainable Task-Oriented Dialogue System. InProceedings of the
15th Conference of the European Chapter of the Association for Computational
Linguistics: Volume 1, Long Papers. 438–449. https://aclanthology.org/E17-1042/
[25] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and
Jian-Yun Nie. 2024. C-Pack: Packed Resources For General Chinese Embeddings.
InProceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval. 641–649. doi:10.1145/3626772.3657878
Accessed June 2026.
[26] Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Deshpande,
Xiaofeng Wang, and Zheng Li. 2024. Retrieval-Augmented Generation with
Knowledge Graphs for Customer Service Question Answering. InProceedings
of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval. 2905–2909. doi:10.1145/3626772.3661370
[27] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective Retrieval
Augmented Generation.arXiv preprint arXiv:2401.15884(2024). doi:10.48550/
arXiv.2401.15884
[28] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Mod-
els. InInternational Conference on Learning Representations. https://openreview.
net/forum?id=WE_vluYUL-X
[29] Steve Young, Milica Gašić, Blaise Thomson, and Jason D. Williams. 2013. POMDP-
Based Statistical Spoken Dialog Systems: A Review.Proc. IEEE101, 5 (2013),
1160–1179. doi:10.1109/JPROC.2012.2225812
[30] Cen Zhao, Tiantian Zhang, Hanchen Su, Yufeng Zhang, Shaowei Su, Mingzhi Xu,
Yu Liu, Wei Han, Jeremy Werner, Claire Na Cheng, and Yashar Mehdad. 2025.
Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based
Customer Support. InProceedings of the 2025 Conference on Empirical Methods
in Natural Language Processing: Industry Track. Association for Computational
Linguistics, Suzhou, China, 1919–1930. doi:10.18653/v1/2025.emnlp-industry.135
[31] Chenyu Zhou, Huacan Chai, Wenteng Chen, Zihan Guo, Rong Shan, Yuanyi Song,
Tianyi Xu, Yingxuan Yang, Aofan Yu, Weiming Zhang, Congming Zheng, Jiachen
Zhu, Zeyu Zheng, Zhuosheng Zhang, Xingyu Lou, Changwang Zhang, Zhihui Fu,
Jun Wang, Weiwen Liu, Jianghao Lin, and Weinan Zhang. 2026. Externalization
in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness
Engineering.arXiv preprint arXiv:2604.08224(2026). doi:10.48550/arXiv.2604.
08224
13