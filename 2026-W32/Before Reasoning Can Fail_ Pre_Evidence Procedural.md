# Before Reasoning Can Fail: Pre-Evidence Procedural Failures in Agentic RAG

**Authors**: Daeyoung Roh, Donghee Han

**Published**: 2026-08-03 10:09:19

**PDF URL**: [https://arxiv.org/pdf/2608.02011v2](https://arxiv.org/pdf/2608.02011v2)

## Abstract
Agentic retrieval-augmented generation (RAG) systems can fail before evidence-conditioned reasoning is tested: an agent may retrieve candidate snippets but finalize without inspecting them. We study this failure mode as a procedural property of the agent trajectory, decomposing wrong answers into pre-evidence discipline failures and post-gold-read failures using saved tool-call traces, retrieved evidence, read passages, and final answers. Across 12,000 paired trajectories on HotpotQA, 2WikiMultiHopQA, and MuSiQue, the two failure types are largely non-redundant: the both-trigger rate is in [11.2%, 13.1%] across regex and spaCy entity extractors. We then evaluate Read-Gate, a minimal runtime invariant requiring an agent to read after search and before finalization. Forced reading improves LLM-Acc by 14.9-19.9 points on trajectories that would otherwise skip reading and by 3.2-9.4 points on full minimal-reasoning cells. Additional diagnostics show that larger hidden thinking budgets do not necessarily increase evidence inspection. Together, these results indicate that evidence-gathering should be evaluated as a trajectory-level control problem, separately from answer-side reasoning.

## Full Text


<!-- PDF content starts -->

Before Reasoning Can Fail: Pre-Evidence Procedural Failures in Agentic
RAG*
Daeyoung Roh
Independent Researcher
dybroh@gmail.comDonghee Han
KAIST
handonghee@kaist.ac.kr
Abstract
Agentic retrieval-augmented generation (RAG)
systems can fail before evidence-conditioned
reasoning is tested: an agent may retrieve can-
didate snippets but finalize without inspecting
them. We study this failure mode as a proce-
dural property of the agent trajectory, decom-
posing wrong answers into pre-evidence disci-
pline failures and post-gold-read failures using
saved tool-call traces, retrieved evidence, read
passages, and final answers. Across 12,000
paired trajectories on HotpotQA, 2WikiMulti-
HopQA, and MuSiQue, the two failure types
are largely non-redundant: the both-trigger rate
is in [11.2%,13.1%] across regex and spaCy
entity extractors. We then evaluate READ-
GATE, a minimal runtime invariant requiring
an agent to read after search and before final-
ization. Forced reading improves LLM-Acc
by 14.9–19.9 points on trajectories that would
otherwise skip reading and by 3.2–9.4 points
on full minimal-reasoning cells. Additional
diagnostics show that larger hidden thinking
budgets do not necessarily increase evidence
inspection. Together, these results indicate
that evidence-gathering should be evaluated as
a trajectory-level control problem, separately
from answer-side reasoning.
1 Introduction
Agentic retrieval-augmented generation (RAG) sys-
tems can fail before evidence-conditioned reason-
ing ever begins. In multi-hop question answer-
ing, we observe agents that issue a search, receive
plausible retrieved snippets, and finalize an answer
without reading any retrieved passage (Figure 1).
Such trajectories are not ordinary reasoning fail-
ures: the model has not yet inspected the evidence
on which its answer is supposed to depend.
This failure mode is specific to agentic RAG. Un-
like fixed-context RAG systems, agentic systems
*Code is available at https://github.com/Noverse0/
before-reasoning-fails
Question Search Read
skippedAnswer
retrieved chunks (never read)
pre-evidence finalizationAgentic RAG: the agent searches, then answers without readingFigure 1: Pre-evidence finalization in agentic RAG. The
agent retrieves candidate chunks with snippets but final-
izes without executing a read action, so the final answer
is produced before evidence inspection.
decide when to search, which retrieved results to
inspect, which passages to read, and when to stop
and answer (Nakano et al., 2021; Yao et al., 2023;
Schick et al., 2023; Qin et al., 2024). This flexi-
bility is useful for multi-hop questions, where the
required evidence is often distributed across pas-
sages (Yang et al., 2018; Ho et al., 2020; Trivedi
et al., 2022), but it also introduces a procedural risk.
An agent may retrieve relevant evidence yet answer
from a snippet, read only part of the evidence chain,
or terminate without reading at all. Final-answer
accuracy alone does not reveal whether the agent
actually executed the evidence-gathering procedure
before answering (Liu et al., 2024).
We argue that these failures should be separated
from post-gold-read failures. If a model reads the
necessary evidence and still answers incorrectly,
the system may need stronger reasoning, better
synthesis, or better supervision. If the model never
reads the evidence, the problem is different: the
execution procedure allowed the agent to stop too
early. Treating both cases as generic answer errors
hides an actionable source of failure in agentic
RAG.
To make this distinction measurable, we intro-
duce a trajectory-level decomposition of wrong
answers into failures of evidence inspection and
failures after evidence has been read. The de-
arXiv:2608.02011v2  [cs.AI]  4 Aug 2026

composition is operational: it is computed from
tool calls, read counts, retrieved snippets, read pas-
sages, gold evidence annotations when available,
and final-answer correctness. We call wrong tra-
jectories with insufficient evidence inspectiondis-
cipline failures, where “discipline” refers to pro-
cedural compliance with the evidence-inspection
protocol rather than answer-side reasoning. We
call wrong trajectories after reading at least some
gold-supporting evidencepost-gold-read failures.
These labels are not claims about latent model cog-
nition; they are deterministic measurements over
saved agent traces.
This framing suggests a simple intervention. If
some errors occur because the agent answers before
reading, then increasing reasoning effort is not the
only relevant fix. A runtime constraint may recover
failures that a stronger or more deliberative model
would not necessarily prevent. We therefore eval-
uate READ-GATE, a minimal environment-level
invariant: after search, the agent must read before
it can answer. If the model attempts to finalize
without reading, the environment rejects the action
and returns a corrective observation prompting the
agent to read a retrieved chunk. The intervention
does not change the model, retriever, index, judge,
or reasoning budget.
We evaluate this hypothesis on HotpotQA,
2WikiMultiHopQA, and MuSiQue using 12,000
paired OpenAI-controller trajectories. The tra-
jectory decomposition shows that discipline fail-
ures and post-gold-read failures are largely non-
redundant: under multi-label reclassification, the
both-trigger rate is in [11.2%,13.1%] across regex
and spaCy entity extractors. READ-GATEdirectly
targets the pre-evidence regime. On examples
with zero-read finalization under the voluntary
minimal-reasoning policy, forced reading improves
LLM-Acc by 14.9–19.9 points; on full minimal-
reasoning cells, READ-GATEimproves LLM-Acc
by 3.2–9.4 points. Additional diagnostics show
that evidence inspection is not simply a byprod-
uct of larger hidden thinking budgets. A context-
injection control further shows that delivering the
same chunk text without a read action recovers only
a minority of READ-GATE’s gain.
Our contributions are threefold:
•We operationalize pre-evidence procedural
failures in agentic RAG through a trajectory-
level decomposition that separates discipline
failures from post-gold-read failures.•We introduce READ-GATE, a minimal run-
time invariant that prevents finalization before
reading retrieved evidence, without changing
the model, retriever, judge, or reasoning bud-
get.
•Across three multi-hop QA benchmarks, we
show that the two failure types are largely non-
redundant, that READ-GATErecovers zero-
read failures when residual discipline error
is high, and that evidence inspection is a dis-
tinct control axis from answer-side reasoning,
not guaranteed by increased reasoning effort
alone.
2 Related Work
Agentic RAG and retrieval control.RAG
combines parametric language models with non-
parametric evidence retrieval for knowledge-
intensive generation (Lewis et al., 2020). Agentic
systems such as WebGPT, ReAct, Toolformer, and
ToolLLM interleave model reasoning with external
tool actions (Nakano et al., 2021; Yao et al., 2023;
Schick et al., 2023; Qin et al., 2024); retrieval-
oriented systems such as IRCoT, FLARE, and A-
RAG study how agents search, retrieve, and read
evidence during multi-step QA (Trivedi et al., 2023;
Jiang et al., 2023; Du et al., 2026). These works
make retrieval and tool use controllable, but they do
not isolate the execution-time case where retrieval
is invoked and then bypassed before evidence in-
spection.
Policy learning versus runtime constraints.
Self-RAG, CRAG, and recent RL-based search
agents train or adapt retrieval policies through re-
flection tokens, learned evaluators, or reinforce-
ment learning (Asai et al., 2024; Yan et al., 2024;
Jin et al., 2025; Chen et al., 2025; Song et al., 2025;
Wang et al., 2025). READ-GATEinstead imposes
a runtime invariant on the environment: after re-
trieval, finalization can be rejected until a read
action occurs. Related calibration and retrieval-
triggering methods estimate when model knowl-
edge is insufficient from confidence, uncertainty,
or answer-level signals (Kadavath et al., 2022; Lin
et al., 2022; Mallen et al., 2023; Press et al., 2023;
Kuhn et al., 2023; Yoran et al., 2024). Our trigger
is different: it is an observable trajectory violation
in which the agent has retrieved candidates but has
not inspected them before answering.

Evaluation, grounding, and reasoning effort.
Attribution, faithfulness, RAG evaluation, and
LLM-as-judge frameworks assess whether final
answers are correct or supported by evidence
(Rashkin et al., 2023; Gao et al., 2023; Liu et al.,
2023a; Es et al., 2024; Saad-Falcon et al., 2024;
Zheng et al., 2023; Liu et al., 2023b), while agent
benchmarks show that final outcomes can mask
tool-use failures (Liu et al., 2024). Reasoning meth-
ods such as chain-of-thought, self-consistency, pro-
cess supervision, and verification improve or super-
vise answer-side reasoning (Wei et al., 2022; Wang
et al., 2023; Kojima et al., 2022; Lightman et al.,
2023; Dhuliawala et al., 2024). We instead evalu-
ate the external action sequence itself: whether the
agent executes evidence inspection before produc-
ing the final answer.
3 Framework: Pre-Evidence Failures in
Agentic RAG
We now make the preceding distinction operational.
An agent trajectory consists of a sequence of tool
actions, including search ,read , and final , to-
gether with retrieved snippets, read passages, gold
evidence annotations when available, and final-
answer correctness. Our goal is not to infer la-
tent model intent, but to label observable fail-
ure modes in the saved trajectory. The frame-
work applies to agents that expose such discrete
search/read/finaltool actions; agents that inter-
leave retrieval and generation implicitly, with no
observable action boundary, are outside its scope
(see Limitations).
We define two primary error axes for agentic
RAG (Figure 2). The first isdiscipline: whether
the agent follows the evidence-inspection proce-
dure before answering. The second ispost-gold-
read error: whether the agent still answers incor-
rectly after reading at least some gold-supporting
evidence. This distinction separates failures that
occur before evidence-conditioned reasoning can
be tested from failures that remain after evidence
has been inspected.
3.1 Trajectory-Level Error Decomposition
For mutually exclusive accounting, we assign each
wrong trajectory to the first matching bucket under
a fixed priority order:
Ewrong =Edisc˙∪ Epost˙∪ Eretr˙∪ Eamb.
The disjointness is therefore operational rather than
semantic: a trajectory may satisfy multiple failure
Wrong answer trajectory
discipline violation ?
no-read / snippet-only /
low-evidence finalyesdiscDiscipline
no
gold-supporting
chunk read ?yespostPost-gold-read
no
gold in retrieved top-k ?noretrRetrieval
otherwiseambAmbiguityRead-Gate targets
this branch
wrong=disċ∪posṫ∪retṙ∪amb   (priority order)Figure 2: Trajectory-level decomposition of wrong an-
swers. Each wrong trajectory is assigned to a discipline,
post-gold-read, retrieval, or residual ambiguity branch.
READ-GATEtargets the discipline branch by blocking
finalization before evidence inspection.
indicators, but the priority rule assigns it to exactly
one accounting bucket. Discipline checks fire first,
followed by post-gold-read, retrieval, and residual
ambiguity. Section 3.4 separately analyzes inde-
pendent multi-label indicators.
3.2 Operational Failure Indicators
Discipline failure. Ediscfires when the answer
is wrong and the trajectory violates a procedural
evidence-inspection condition. We use “discipline”
in the procedural sense: whether the agent follows
the required evidence-inspection protocol before
answering. We use three observable subtypes:
•No-read final: the agent emits a final answer
withread count=0.
•Snippet-only final: an answer entity appears
only in retrieved search snippets, never in any
read chunk.
•Low-evidence final: named entities from the
question are covered in read chunks at less
than an 80% rate.
Entity coverage is computed with a deterministic
named-entity matching heuristic; implementation
details are given in Appendix G. Because snippet-
only and low-evidence subtypes rely on surface
matching heuristics, we also report a strict no-read-
only lower bound and threshold sweeps in the ro-
bustness analyses. Pooled over the main cells, no-
read final is the majority subtype and the only one

that requires no entity-matching heuristic at all; we
therefore treat it as the primary discipline signal
and the other two as secondary diagnostics (Ap-
pendix H).
Post-gold-read failure. Epostfires when retrieval
succeeded, the agent read at least one gold-
supporting chunk, and the final answer is still
wrong. This category is narrower than complete
evidence-conditioned answering: in multi-hop QA,
reading one gold chunk may still leave the sup-
porting chain incomplete. Because this category
depends on gold supporting evidence annotations,
datasets without complete gold evidence fields sup-
port only limited Epostanalysis (see §4.1 for the
dataset-specific caveat).
Retrieval and ambiguity. Eretrcaptures cases
where gold evidence does not appear in any top-
kretrieval result. Eambis the residual bucket for
cases that do not cleanly fit the operational defini-
tions. In robustness analysis, residual ambiguity
remains below 15% of wrong cases.
3.3 READ-GATE: A Runtime
Evidence-Inspection Constraint
The decomposition suggests a direct diagnostic in-
tervention. If a wrong trajectory occurs because
the agent finalizes before inspecting evidence, then
the environment can enforce a minimal evidence-
inspection invariant: after search, the agent must
read before it can answer.
READ-GATEis a deterministic environment rule
that enforces a singleread-before-finalinvariant:
if the model emits final with read count =0, the
environment rejects the action and returns a correc-
tive observation instructing the agent to call read
on a promising chunk before answering.
READ-GATEoperates at inference time. It does
not change model weights, retrieval, the index, the
judge, decoding temperature, or reasoning bud-
get. A stricter checkpoint that additionally rejects a
non-read action issued immediately after search
(read-after-search) is implemented within the same
corrective-hint framework but is not part of the
main READ-GATEcondition used throughout the
paper; Appendix F explores a family of stronger
procedural gates in this spirit. Operational details
for the main READ-GATEinvariant (trigger pred-
icate, corrective hint, rollout protocol) are in Ap-
pendix B.3.4 Priority Accounting and Multi-Label
Indicators
Priority assignment is useful for mutually exclu-
sive accounting, but it can mask co-occurring labels.
We therefore also compute independent binary in-
dicators for discipline and post-gold-read failure.
We write PdiscandPpostfor the corresponding
trajectory-level event probabilities, estimated as
cell-level rates in aggregate analyses and modeled
with question-clustered logistic regressions in Sec-
tion 5.5. This multi-label version is the basis for the
both-trigger overlap test in Section 5.1. Robustness
checks further report a strict no-read-only lower
bound and vary the low-evidence threshold from
0.6 to 0.9.
3.5 Predictions
The framework yields three falsifiable predictions.
(H1)Non-redundancy: discipline and post-gold-
read labels should not collapse to a single failure
axis. (H2)Conditional benefit: READ-GATE’s ac-
curacy gain should scale with residual discipline
error rather than with reasoning level alone. (H3)
Differential response: READ-GATEand reasoning
effort should produce distinguishable signatures
over(Pdisc, Ppost), indicating distinct control axes.
We test these predictions through multi-label re-
classification, cell-level correlation, and question-
clustered logistic models with trajectory-level label
permutation.
H1’s overlap test requires per-chunk gold evi-
dence and is therefore established on HotpotQA
and 2WikiMultiHopQA, the two datasets with such
annotations; MuSiQue, which lacks gold chunk
IDs, instead tests H2 alongside the other two
datasets, with its discipline failures corroborated
by a structural, annotation-free signal (90.4% strict
no-read on the most severe cell; Appendix H).
4 Experimental Setup
4.1 Datasets
We evaluate on three Wikipedia-style multi-hop QA
datasets: HotpotQA (Yang et al., 2018), 2WikiMul-
tiHopQA (Ho et al., 2020), and MuSiQue (Trivedi
et al., 2022). Unless otherwise noted, all main
table cells use n= 1,000 examples per dataset–
condition pair, paired by question ID across con-
ditions; this includes the OpenAI controllers, the
Gemini 2.5 Flash thinking-budget diagnostic, and
the prompt-only and ctx-inject mechanism controls.
Boundary ablations, including medium-reasoning

READ-GATEchecks and the gate-family probe
(Appendix F), use matched n= 100 samples and
are interpreted only as scope checks.
One measurement caveat affects the decomposi-
tion. The post-gold-read bucket requiresper-chunk
gold supporting-evidence annotations, but the pro-
cessed MuSiQue export does not contain complete
per-chunk gold-evidence fields. We therefore use
MuSiQue for aggregate accuracy, discipline fail-
ures, and READ-GATEeffects, but restrict analyses
requiring Epostor chunk-level gold-read quantities
to HotpotQA and 2WikiMultiHopQA.
4.2 Agent Interface and Retrieval Tools
The main experiments use a two-tool agent inter-
face with hybrid search and chunk read. The hybrid
search tool combines BM25 and dense retrieval us-
ing reciprocal-rank fusion with k= 60 and returns
the top five chunk IDs with short snippets. The
read tool expands a selected chunk ID to the full
chunk text. Cross-search chunk-ID deduplication is
enabled, so repeated chunk IDs are not repeatedly
charged as new evidence.
4.3 Controller Models and Reasoning Levels
The main experiments use OpenAI gpt-4o-mini
and gpt-5-mini with minimal and medium reason-
ing effort, which separates backbone choice from
reasoning-effort choice within gpt-5-mini. We use
OpenAI controllers for the main within-provider
analysis because they provide structured tool-call
traces and multiple reasoning-effort settings for
the same model. The 12,000 OpenAI trajectories
come from three datasets and four conditions per
dataset (1,000 questions each): (i) gpt-4o-mini, (ii)
gpt-5-mini minimal, (iii) gpt-5-mini medium, and
(iv) gpt-5-mini minimal plus READ-GATE. Gemini
2.5 Flash is used only as an external diagnostic for
hidden thinking-budget behavior.
4.4 Evaluation Protocol
All main cells use the same retrieval and loop con-
figuration unless otherwise stated: a 0.6B dense
embedding model specified in Appendix A, hybrid
top-k= 5 , maximum loop budget 10, maximum
token budget 128k, and temperature 0.0. The Gem-
ini thinking-budget runs use the same search/read
API and temperature 0.0; the thinking-budget com-
parison holds the agent loop fixed and varies only
the model’s hidden thinking budget.
We report LLM-Acc as the primary outcome us-
ing a fixed gpt-5-mini judge at temperature 0.0; thejudge sees only the question, gold answer, and pre-
diction and returns a binary semantic-equivalence
label. Contain-Acc is reported as a secondary
metric where short-form gold answers are avail-
able. A cross-family Gemini 2.5 Pro re-judge of
a stratified n= 450 sample agrees at κ= 0.924
with direction-preserving READ-GATEgains on all
three datasets (Appendix L). Full prompts and tool
descriptions are in Appendix A.
Paired analyses use shared question IDs be-
tween conditions. We report exact McNemar tests
for paired correctness changes and use question-
clustered standard errors in logistic models to ac-
count for repeated evaluation of the same question
under different conditions.
Reproducibility.At the URL in the title footnote,
we release the full agent-loop and READ-GATE
implementation, the reproduction scripts used to
generate every table and figure in this paper, and a
corpus of 33,950 raw agent trajectories (49 files),
including the 12,000-trajectory paired corpus with
per-question discipline and post-gold-read labels;
the trajectories back every per-question table ex-
cept two externally-sourced A-RAG comparison
rows, documented in the release. The release also
includes cached analysis outputs that reproduce
every reported number without further API access.
5 Experiments
We evaluate the three predictions from Section 3.5.
Section 5.1 tests non-redundancy; Sections 5.2–
5.3 evaluate when READ-GATEhelps; and Sec-
tions 5.5–5.6 compare runtime constraints with
reasoning-effort changes.
5.1 Discipline and Post-Gold-Read Failures
Are Non-Redundant
We first test whether discipline and post-gold-read
failures identify distinct trajectory regimes rather
than two names for the same wrong answers. Fig-
ure 3 shows a three-regime pattern: discipline fail-
ures are lower under gpt-4o-mini (HotpotQA 7.6%,
2WikiMultiHopQA 12.8%), rise under gpt-5-mini
minimal (13.3%, 22.1%, and 57.0% on HotpotQA,
2WikiMultiHopQA, and MuSiQue), and decrease
under medium reasoning on HotpotQA and 2Wiki-
MultiHopQA.
Post-gold-read errors follow a different curve
on datasets with gold-evidence annotations: they
generally decrease with reasoning effort, while dis-
cipline failures peak in the low-effort regime where

Figure 3: Error indicators across agent regimes. Disci-
pline failures peak at gpt-5-mini minimal; post-gold-
read errors follow a different curve. The x-axis is
regime-level, not a scaling curve.
Table 1:Forced reading recovers skipped-read cases.
LLM-Acc on the self-selected zero-read subset; paired
McNemar p <10−4. This estimates a rescue effect, not
the population-level marginal effect reported in Table 3.
DatasetnZero-read Forced read∆
HotpotQA 215 58.1 73.0 +14.9
2Wiki 261 42.1 62.1 +19.9
MuSiQue 617 22.5 37.4 +14.9
the model can use tools but does not reliably exe-
cute the evidence-gathering procedure.
The strongest evidence for non-redundancy
comes from a multi-label reclassification that al-
lows discipline and post-gold-read indicators to
co-occur. Among 3,807 wrong cases across the
12,000 OpenAI-family trajectories, the both-trigger
rateP(disc∧post|wrong) stays in a narrow
[11.2%,13.1%] band across two independent en-
tity extractors (regex / spaCy encore websm;
the binary discipline indicator agrees at Cohen’s
κ= 0.628 , observed agreement 82.1% ). Under
both extractors the overlap sits far below the 60%
level at which the two axes would collapse into
one, and the discipline-only share dominates the
post-gold-read-only share by more than two to
one (regex: 46.5% vs 21.4%; spaCy: 50.2% vs
19.5%; Appendix D). The small, extractor-stable
overlap is the key result: many wrong answers oc-
cur before evidence-conditioned reasoning is tested,
while a separate set occurs after the agent has in-
spected gold-supporting evidence. This full-trace
accounting uses the operational indicators as imple-
mented; post-gold-read interpretation relies on the
HotpotQA and 2WikiMultiHopQA portions where
complete gold evidence is available.
A strict no-read-only lower bound preserves the
same minimal-reasoning discipline hump: Hot-potQA 9.3%, 2WikiMultiHopQA 15.4%, and
MuSiQue 51.5%. Thus, the main discipline signal
is not driven only by snippet-only or low-evidence
heuristics. This satisfies prediction H1.
5.2 READ-GATERecovers Zero-Read
Failures
We next use READ-GATEas a targeted interven-
tion for trajectories that terminate before evidence
inspection. The goal is not to make READ-GATE
a universally optimal decoding policy, but to test
whether a measurable subset of errors occurs be-
cause the agent finalizes before reading. If the
model emits a final answer before reading, the en-
vironment rejects the action and continues the loop.
Retrieval, model weights, the judge, and reasoning
effort remain unchanged.
The most direct diagnostic test is the paired
induced-read contrast. We identify examples with
zero-read finalization under the voluntary minimal-
reasoning policy, then rerun the same questions
under a forced-read environment. Table 1 reports
a within-subset paired contrast: on the zero-read
subset, forced reading raises LLM-Acc by 14.9–
19.9 points across datasets, with paired McNemar
tests below 10−4. This subset is self-selected by
the voluntary policy and is not a random popula-
tion; the contrast measures the rescue effect on
cases the agent would otherwise skip rather than
the marginal effect of forced reading at the pop-
ulation level (which is reported separately as the
+3.2–9.4 full-cell READ-GATEgain in §5.3).
A prompt-only control across the three minimal-
reasoning cells reduces zero-read behavior but does
not replicate READ-GATE’s accuracy gain. In con-
trast, READ-GATEboth nearly eliminates zero-read
finalization and improves LLM-Acc on all three
datasets (Appendix K). Thus, instruction alone can
reduce the surface behavior, but it does not recover
the accuracy gain of an execution-level constraint.
A rank analysis shows that READ-GATEprimar-
ily enforces that a read happens, not which chunk
is read: forced reads remain rank-1 dominated, and
94% of forced reads on gold-annotated datasets
land on gold-supporting chunks (Appendix B.2).
On the full minimal-reasoning cells, READ-
GATEimproves LLM-Acc by 3.2 points on Hot-
potQA, 5.3 points on 2WikiMultiHopQA, and 9.4
points on MuSiQue. Appendix B.1 reports the cor-
responding overhead profile, including reads, loop
count, corrective interventions, retrieved tokens,
and LLM-Acc. Appendix B.3 reports Contain-Acc

Table 2:Three-arm mechanism ablation.gpt-5-mini minimal, paired n= 1,000 per dataset. Deltas are LLM-Acc
changes relative to no-READ-GATE. READ-GATEimproves accuracy on all three datasets, whereas ctx-inject does
not replicate the gain and is net-negative on 2Wiki.
Dataset no-READ-GATEREAD-GATE(∆) ctx-inj (∆) READ-GATEvs ctxp
HotpotQA 79.6 82.8(+3.2) 79.5 (−0.1)1.4×10−4
2Wiki 64.4 69.7(+5.3) 57.0 (−7.4)<10−22
MuSiQue 34.2 43.6(+9.4) 38.1 (+3.9)1.0×10−5
Table 3:READ-GATE’s gain tracks residual disci-
pline error.Baseline and Ediscare no-READ-GATE
values; ∆READ-GATEis the paired LLM-Acc change.
Minimal-reasoning rows recover accuracy (green);
medium-reasoning rows do not (red), consistent with
H2.†Medium rows use the same n= 100 matched-
question subsample as the paired READ-GATEablation
(Appendix B.1); the same no-READ-GATEsystem’s full
n= 1,000 cell mean is 90.8/88.8/67.1 (Appendix M) —
the two are consistent measurements on different ques-
tion subsamples, not two values for different systems.
Effort DatasetE disc(%) Baseline∆READ-GATE
MinimalHotpotQA13.379.6 +3.2
2Wiki22.164.4 +5.3
MuSiQue57.034.2 +9.4
Medium†HotpotQA 4.5 95.0 +0.0
2Wiki 5.7 90.0 −7.0
MuSiQue 21.7 74.0 −4.0
for the same cells; its direction matches LLM-Acc
on all three datasets.
5.3 READ-GATEHelps When Residual
Discipline Error Is High
READ-GATEis conditional: it helps most when
residual discipline error is high. We compare disci-
pline error before READ-GATEagainst the READ-
GATEchange in LLM-Acc across full READ-GATE
cells (Table 3; visualization in Appendix B.4).
Minimal-reasoning cells show a monotonic pat-
tern: larger residual discipline error corresponds to
larger READ-GATEgains. Medium-reasoning cells
have lower residual discipline error and show zero
or negative gains. This is expected boundary be-
havior rather than a contradiction. When the agent
already reads reliably, the gate has little headroom
and can introduce unnecessary interventions.
This satisfies prediction H2: READ-GATE’s ben-
efit tracks residual discipline error rather than rea-
soning level alone. Dataset difficulty is unlikely
to confound this pattern: a gold-chunk retrieval-
success proxy has near-zero cross-cell variance and
cannot explain the spread in ∆READ-GATE; con-
trolling instead for retrieval effort (total retrievedtokens), the correlation survives and strengthens
(partial r= +0.881 ,n= 6 ), holding direction-
ally at +0.56 pooled with n= 13 cross-family
cells. Given the small cell counts we report this
as a robustness check rather than a confirmatory
test. The cross-family cells are Qwen2.5-Instruct
(3B/7B/14B) runs, where the same dose-response
direction reproduces at r= +0.629 (vs.+0.728
for the OpenAI family; Appendix D.1).
Under medium reasoning, READ-GATEdoes
not merely fail to help; it increases mean loop
count on all three datasets while accuracy does
not improve. Matched-question-ID paired means
(n= 100 /dataset) rise from 3.92 to 4.25 on Hot-
potQA, 5.08 to 6.10 on 2WikiMultiHopQA, and
6.90 to 7.44 on MuSiQue, so additional trajectory
steps do not explain the minimal-regime gains re-
ported above. On 2Wiki-medium, where the effect
is significant ( −7.0 , Table 3), the max-loop rate
rises from 15% to 35% under READ-GATE, and
all 8 paired cases where READ-GATEbreaks a pre-
viously correct answer terminate at the loop cap
with an abstention or partial final answer — forced
reads exhaust the budget of an already-competent
agent rather than giving it more useful steps.
5.4 Mechanism: Not Merely Context Injection
An alternative interpretation of READ-GATE’s gain
is that it merely pushes additional chunk text into
the agent’s context window. We test this with a
three-arm paired ablation across all three datasets
(gpt-5-mini minimal; paired by question ID; all
arms share retrieval, backbone, loop budget, and
judge). Arm (i) is no-READ-GATE; (ii) is READ-
GATE; (iii) (ctx-inject) uses the same trigger as
(ii) but silently appends the rank-1 chunk text as a
user -role observation without issuing a read tool
call (Table 2).
Across all three datasets, ctx-inject fails to repli-
cate READ-GATE’s gain and is sometimes net-
negative. On HotpotQA, ctx-inject changes LLM-
Acc by −0.1 pp while READ-GATEimproves it by
+3.2 pp; on 2WikiMultiHopQA, ctx-inject reduces

accuracy by −7.4 pp while READ-GATEimproves
it by+5.3 pp; on MuSiQue, ctx-inject improves ac-
curacy by +3.9 pp but still recovers less than half of
READ-GATE’s +9.4 pp gain. On every dataset, the
paired READ-GATEvs. ctx-inject contrast is highly
significant (McNemar p≤10−4). Delivering the
same chunk text without the read action therefore
recovers at most a minority of READ-GATE’s gain
and can be net-negative, so the gain is not explained
by additional context alone. A follow-up four-arm
probe on MuSiQue ( n= 500 , gpt-5-mini mini-
mal) decomposes this further by separating delivery
channel from the self-issued read action: relative
to no-READ-GATE, READ-GATEgives +8.2 pp, a
fabricated tool -role injection gives +5.2 pp, and
auser -role injection gives +4.4 pp, so the chan-
nel difference ( +0.8 pp) is not significant while
committing to the read action itself adds a further
+3.0 pp over tool-channel injection (Appendix C.1).
This decomposes, rather than contradicts, the three-
arm result above: across these controls, the residual
gain tracks the agent’s self-issued read action rather
than the channel through which the text is deliv-
ered. We treat this action-commitment account as
the interpretation best supported by the controls,
not as a conclusively isolated mechanism.
When to enable READ-GATEin practice.A
practical trigger for enabling READ-GATEdoes
not require gold labels. The most direct proxy is
the zero-read finalization rate in unlabeled logs:
if an agent often searches and then finalizes with
read count= 0 , it is a natural candidate for the
gate. Gold annotations are needed to evaluate accu-
racy and post-gold-read failures, but not to detect
the procedural violation itself.
5.5 Runtime Constraints and Reasoning
Effort Have Different Signatures
We next ask whether READ-GATEmerely mimics
additional reasoning effort, or whether it changes
a different part of the agent trajectory. The results
support the second view: procedural constraints
and reasoning effort produce different signatures
over(Pdisc, Ppost)(Table 4; see Appendix J for a
visualization).
We fit question-clustered logistic models over
the OpenAI trajectories; Epost terms restrict
to HotpotQA and 2WikiMultiHopQA, which
carry per-chunk gold annotations. READ-GATE
strongly reduces discipline failure ( Pdisc OR
0.38 [0.35,0.41] ), as expected for an interventionTable 4:READ-GATEand reasoning effort target dif-
ferent indicators.The marginal READ-GATE →P post
OR is exposure-induced re-classification; the within-
strata row ( n= 1,863 , read-exposure matched) is the be-
havioral estimate. Correction details are in Appendix D.
Intervention Spec. Indicator OR 95% CI / test
READ-GATEfullP disc 0.38[0.35,0.41]
marginalP post 1.46[1.36,1.57]
within-strataP post 1.00[0.84,1.19]
Effortmed vs minP disc 0.22[0.20,0.26]
med vs minP post 0.51[0.43,0.61]
Interaction effort×axis stacked2.02p <10−4
Perm. test —T obs 18.6 ppp <10−4
that blocks finalization before reading. Its ef-
fect on post-gold-read failure requires more care.
Marginally, READ-GATEappears to increase Ppost
(OR1.46 [1.36,1.57] ), but this is a re-classification
effect: the gate converts some no-read wrong an-
swers into with-read wrong answers, making them
eligible for the post-gold-read label. Indeed, all
106 paired Ppostlabel flips are 0→1 cases driven
byread countchanging from 0 to positive.
To estimate the behavioral effect after read ex-
posure is held fixed, we compare trajectories with
read count>0 in both conditions ( n= 1,863 ).
In this within-strata comparison, READ-GATE’s
OR for Ppostis1.00 [0.84,1.19] ; the stricter gold-
read-both stratum (HotpotQA+2Wiki, n= 1,471 )
gives the same OR, 1.00 [0.83,1.20] . Thus, READ-
GATEchanges whether evidence is inspected, but
does not measurably change post-read answering
once read exposure is matched.
Reasoning effort has a different signature. In-
creasing gpt-5-mini from minimal to medium ef-
fort with READ-GATEoff lowers both indica-
tors:PdiscOR0.22 [0.20,0.26] andPpostOR0.51
[0.43,0.61] , with a significant effort-by-axis in-
teraction (OR 2.02,p <10−4). In other words,
READ-GATEmainly repairs whether evidence is
inspected, whereas additional reasoning effort af-
fects both inspection behavior and post-read an-
swering. A within-cell-stratified label-permutation
test further rejects exchangeability of Pdiscand
Ppostunder READ-GATE( Tobs= 18.6 pp, null
99th percentile 3.5pp,p <10−4; Appendix D).
This satisfies prediction H3: READ-GATEand rea-
soning effort produce distinguishable signatures
over(Pdisc, Ppost)even after read-exposure is held
fixed.

Table 5:Gemini 2.5 Flash thinking budget increases
zero-read finalization on all three datasets.Paired
contrast between thinking budget off ( b= 0 ) and on
(b= 1024 ), with no READ-GATEand matched n=
1,000question IDs per dataset.
Dataset ZR∆Net∆McNemarp
HotpotQA+5.7pp −440.0025
2Wiki +24.8pp −670.0001
MuSiQue +42.6pp −73 3.1×10−5
5.6 Hidden Thinking Does Not Guarantee
Evidence Inspection
We use Gemini 2.5 Flash as an external diag-
nostic and repeat the thinking-budget contrast on
three datasets. With the same two-tool loop and
no READ-GATE, enabling a 1024-token think-
ing budget increases zero-read finalization on all
three datasets and reduces paired correctness (Ta-
ble 5). On HotpotQA, zero-read rises by +5.7 pp
with a net loss of 44correct answers (McNemar
p= 0.0025 ); on 2WikiMultiHopQA, zero-read
rises by +24.8 pp with a net loss of 67correct
answers (McNemar p= 0.0001 ); on MuSiQue,
zero-read rises by +42.6 pp with a net loss of 73
correct answers (McNemar p= 3.1×10−5). The
zero-read shift is largest on MuSiQue and small-
est on HotpotQA, suggesting that the effect may
be more pronounced when answers require longer
evidence chains. Appendix E provides additional
break-type analysis.
The replication does not imply that reasoning ef-
fort is harmful in general. In this diagnostic setting,
the result supports the broader claim that hidden
deliberation and evidence inspection can vary in-
dependently: a model can spend more compute
while still failing to execute the external evidence-
gathering procedure. The larger shift on MuSiQue
suggests that this failure mode may be more pro-
nounced when evidence chains are harder to re-
cover from internal memory alone.
Cost–accuracy implication.Adding READ-
GATEto gpt-5-mini at minimal reasoning closes 20–
24% of the LLM-Acc gap to gpt-5-mini at medium
reasoning, with modest overhead in reads and loop
count (Appendix B.5). This suggests READ-GATE
can be a partial substitute for higher reasoning ef-
fort when residual discipline error is high, but not
a replacement for stronger reasoning. Appendix M
reports an additional interface-granularity compari-
son as supporting context.Robustness and scope.Robustness checks sup-
port the same qualitative conclusions: the disci-
pline signal is stable under low-evidence thresh-
old sweeps and a strict no-read-only lower bound;
paired bootstrap intervals exclude zero for the head-
line READ-GATEgains and induced-read contrast;
and the label-permutation result remains signifi-
cant under a within-cell-stratified null. Full de-
tails are in Appendix D. Our claims are scoped
to Wikipedia-style multi-hop QA, where snippet-
grounded shortcutting and explicit evidence chains
make discipline failures visible.
6 Conclusion
Agentic RAG systems can fail before evidence-
conditioned reasoning begins. In Wikipedia-style
multi-hop QA, we show that a substantial subset
of wrong answers arises when the agent searches,
observes plausible surface evidence, and finalizes
without sufficient reading. These failures are not or-
dinary post-evidence reasoning errors: the system
has not yet reached the state in which its answer is
grounded in inspected evidence.
We make this bottleneck measurable through a
trajectory-level decomposition that separates disci-
pline failures from post-gold-read failures. Across
paired trajectories, these indicators identify largely
non-redundant error regimes and respond differ-
ently to runtime constraints and reasoning-effort
changes. READ-GATErecovers zero-read failures
when residual discipline error is high, but has little
headroom when the agent already reads reliably.
This supports a diagnostic view of READ-GATE:
it is not a universally optimal control policy, but a
minimal intervention for testing whether evidence
inspection itself is the bottleneck.
Future work.Future work should test this de-
composition across larger controllers, additional
model families, non-English settings, and non-
Wikipedia RAG domains, while studying whether
the self-issued read action’s advantage over pas-
sively delivered context generalizes across back-
bones and domains.
Final-answer accuracy remains necessary, but
it is incomplete without trajectory-level measure-
ments of evidence inspection.
Limitations
Scope.Our main experiments focus on cost-
sensitive agent controllers (gpt-4o-mini and gpt-
5-mini), with Gemini 2.5 Flash used as an exter-

nal diagnostic. This operating regime is aligned
with READ-GATE’s intended use: the gate has
the most headroom when residual discipline er-
ror is high (§5.3). Larger frontier controllers, ad-
ditional model families, and non-English settings
remain important directions for future work. The
context-injection mechanism check (§5.4) is run
on three multi-hop datasets at gpt-5-mini mini-
mal, and the channel-vs-action decomposition (Ap-
pendix C.1) is run on a single dataset and backbone;
whether the self-issued-action effect generalizes
identically across other backbones and datasets
is open. The framework also presumes discrete
search /read /final tool actions (§3); agents that
interleave retrieval and generation implicitly ex-
pose no read boundary to observe or gate on.
Domain dependence.All datasets are English
Wikipedia-style multi-hop QA benchmarks, where
snippet-grounded shortcuts and explicit evidence
chaining make pre-evidence failures visible. Other
RAG domains have different snippet semantics and
retrieval requirements. The trajectory-level decom-
position may transfer, but entity-coverage thresh-
olds and gate behavior should be re-calibrated per
domain.
Retrieval and safety.READ-GATEis not a
retrieval-quality fix: it assumes that search returns
useful candidates often enough that forcing a read
can expose relevant evidence. It also modestly in-
creases reads-per-question and retrieved tokens (Ta-
ble 6). Production systems should therefore com-
bine forced-read mechanisms with retrieval-quality
monitoring, filtering, and answer-side verification,
especially in sensitive domains.
Ethics Statement
This work uses public QA benchmarks and does not
involve human subjects or private user data. The
main risks are deployment risks common to RAG
systems. A procedural gate can improve evidence
inspection, but it does not guarantee factual correct-
ness. Systems using such gates should still retain
answer verification, logging, and domain-specific
safety checks, especially in high-stakes settings.
The results should not be used to justify deploying
low-effort agents without downstream validation.
Use of Scientific Artifacts.Our experiments use
open-source tools including PyTorch (Paszke et al.,
2019), with the Qwen3-Embedding-0.6B dense re-
triever (Zhang et al., 2025) obtained via the Hug-gingFace (Wolf et al., 2020) library. For agent
controllers and LLM-as-judge scoring we use Ope-
nAI’s API (gpt-4o-mini, gpt-5-mini) under their
sharing and publication policy, and Google’s Gem-
ini API (Gemini 2.5 Flash as an external thinking-
budget diagnostic and Gemini 2.5 Pro for cross-
judge robustness) under their respective terms of
service. All datasets (HotpotQA, 2WikiMulti-
HopQA, MuSiQue) are publicly released by their
authors under their original licenses and used as
intended for research evaluation.
Use of AI Assistants.We used Google Gemini to
refine wording and polish prose. Some experimen-
tal code was written with assistance from Anthropic
Claude. All final scientific claims, experimental
designs, and analyses are the authors’ own; AI as-
sistants were used only as a writing and coding
aid, and no AI-generated content was incorporated
without author review and verification.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions.
Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze
Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang,
Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan
Yang, Zenan Zhou, and Weipeng Chen. 2025.
ReSearch: Learning to reason with search for
LLMs via reinforcement learning.arXiv preprint
arXiv:2503.19470.
Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu,
Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Ja-
son Weston. 2024. Chain-of-verification reduces hal-
lucination in large language models. InFindings of
the Association for Computational Linguistics: ACL
2024.
Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan
Wang, Pengyu Wang, Xiaorui Wang, and Zhen-
dong Mao. 2026. A-RAG: Scaling agentic retrieval-
augmented generation via hierarchical retrieval inter-
faces.Preprint, arXiv:2602.03442.
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2024. RAGAS: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate

text with citations. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6465–6488.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-R1: Training LLMs to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, Scott Johnston, Sheer El-Showk,
Andy Jones, Nelson Elhage, Tristan Hume, Anna
Chen, Yuntao Bai, Sam Bowman, Stanislav Fort,
Deep Ganguli, Danny Hernandez, Josh Jacobson,
Jackson Kernion, Shauna Kravec, Liane Lovitt, Ka-
mal Ndousse, Catherine Olsson, Sam Ringer, Dario
Amodei, Tom Brown, Jack Clark, Nicholas Joseph,
Ben Mann, Sam McCandlish, Chris Olah, and Jared
Kaplan. 2022. Language models (mostly) know what
they know.arXiv preprint arXiv:2207.05221.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yu-
taka Matsuo, and Yusuke Iwasawa. 2022. Large lan-
guage models are zero-shot reasoners. InAdvances
in Neural Information Processing Systems.
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023.
Semantic uncertainty: Linguistic invariances for un-
certainty estimation in natural language generation.
InInternational Conference on Learning Representa-
tions.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih,
Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. InAdvances in Neu-
ral Information Processing Systems.
Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri
Edwards, Bowen Baker, Teddy Lee, Jan Leike,
John Schulman, Ilya Sutskever, and Karl Cobbe.
2023. Let’s verify step by step.arXiv preprint
arXiv:2305.20050.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
Teaching models to express their uncertainty in
words.Transactions on Machine Learning Research.Nelson F. Liu, Tianyi Zhang, and Percy Liang. 2023a.
Evaluating verifiability in generative search engines.
InFindings of the Association for Computational
Linguistics: EMNLP 2023.
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu
Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen
Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Ao-
han Zeng, Zhengxiao Du, Chenhui Zhang, Sheng
Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie
Huang, Yuxiao Dong, and Jie Tang. 2024. Agent-
Bench: Evaluating LLMs as agents. InInternational
Conference on Learning Representations.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023b. G-Eval:
NLG evaluation using GPT-4 with better human
alignment. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 2511–2522.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics, pages
9802–9822.
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff
Wu, Long Ouyang, Christina Kim, Christopher
Hesse, Shantanu Jain, Vineet Kosaraju, William
Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou,
Gretchen Krueger, Kevin Button, Matthew Knight,
Benjamin Chess, and John Schulman. 2021. We-
bGPT: Browser-assisted question-answering with hu-
man feedback.arXiv preprint arXiv:2112.09332.
Adam Paszke, Sam Gross, Francisco Massa, Adam
Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca
Antiga, Alban Desmaison, Andreas K ¨opf, Edward
Yang, Zachary DeVito, Martin Raison, Alykhan Te-
jani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang,
Junjie Bai, and Soumith Chintala. 2019. PyTorch:
An imperative style, high-performance deep learning
library. InAdvances in Neural Information Process-
ing Systems.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 5687–5711.
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan
Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang,
Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian,
Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li,
Zhiyuan Liu, and Maosong Sun. 2024. ToolLLM:
Facilitating large language models to master 16000+
real-world APIs. InInternational Conference on
Learning Representations.
Qwen Team. 2024. Qwen2.5 technical report.arXiv
preprint arXiv:2412.15115.

Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm,
Lora Aroyo, Michael Collins, Dipanjan Das, Slav
Petrov, Gaurav Singh Tomar, Iulia Turc, and David
Reitter. 2023. Measuring attribution in natural lan-
guage generation models.Computational Linguistics,
49(4):777–840.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics.
Timo Schick, Jane Dwivedi-Yu, Roberto Dess `ı, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves
to use tools. InAdvances in Neural Information Pro-
cessing Systems.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-Searcher: Incentivizing the
search capability in LLMs via reinforcement learning.
arXiv preprint arXiv:2503.05592.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics, pages 10014–10037.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le,
Ed H. Chi, Sharan Narang, Aakanksha Chowdhery,
and Denny Zhou. 2023. Self-consistency improves
chain of thought reasoning in language models. In
International Conference on Learning Representa-
tions.
Ziliang Wang, Kang An, Xuhui Zheng, Faqiang
Qian, Weikun Zhang, Cijun Ouyang, Jialu Cai,
Yuhang Wang, and Yichao Wu. 2025. Erase to im-
prove: Erasable reinforcement learning for search-
augmented LLMs.arXiv preprint arXiv:2510.00861.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V . Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, R ´emi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven
Le Scao, Sylvain Gugger, Mariama Drame, QuentinLhoest, and Alexander M. Rush. 2020. Transform-
ers: State-of-the-art natural language processing. In
Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing: System
Demonstrations, pages 38–45.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation. In
First Conference on Language Modeling.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
pages 2369–2380.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in language
models. InInternational Conference on Learning
Representations.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2024. Making retrieval-augmented language
models robust to irrelevant context. InInternational
Conference on Learning Representations.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-bench and chatbot arena.
InAdvances in Neural Information Processing Sys-
tems, Datasets and Benchmarks Track.

A Prompts and Tool Definitions
For reproducibility, this appendix records the verba-
tim agent system prompt, the READ-GATEcorrec-
tive hints, the tool schemas, and the judge prompt
used in all experiments. Question wording and
gold annotations are inherited from each dataset’s
release.
A.1 Agent System Prompt
The following prompt is used for both the no-
READ-GATEand the READ-GATEconditions on
all main cells:
You are a question-answering assistant with
access to a corpus through hybrid retrieval
tools.
## Available Tools
-search(query, k): Hybrid (BM25 + dense)
retrieval. Returns top-k chunk IDs with the
5 most query-relevant sentences as snippets.
Snippets are NOT the full chunk --- use ‘read’
for full context. Snippet overlap across calls
signals the index has converged on this query
intent; reword or pivot.
-read(evidence id): Read the full content of
a specific chunk.
## Strategy
Work iteratively: search→read→evaluate
→search→read→. . .→answer.
Never re-issue a near-duplicate query.If
a search returned poor or repeated results,
do ONE of: (a) reword with different
surface forms, (b) add a discriminating
entity/date/number from prior results, (c)
decompose into a narrower sub-question, (d)
stop searching and answer from what you have.
Multi-hop discipline.If the question chains
two facts, do NOT search the full composed
question. Use a hop-1 search to identify
the bridge entity, read, extract entity,
then hop-2 search using the bridge entity.
Single-hop questions should be answered in one
search+read whenever possible.
## When Answering
- Before answering, you MUST ‘read’ at least
one chunk whose snippet directly supports
the answer. Snippets alone are insufficient
evidence. Exception: if the snippet contains
the literal extractive answer span and matches
the question entity exactly, you may answer
without ‘read’. - Ground your response in the
retrieved chunks. Cite the specific chunks
that support your answer. -Stop when:you
have read evidence that uniquely answers the
question, OR two reformulations in a row
produced no new useful evidence.
The strict-prompt control replaces the snippet
exception with:“Before answering, you MUST
‘read’ at least one chunk whose text directly sup-
ports the answer. Snippets alone are never suffi-
cient evidence. Do not answer until at least one
read(evidence id) call has succeeded. ”A.2 Read-Gate Corrective Hints
The main READ-GATEcondition used throughout
the paper triggers a single environment-level check:
the agent emits a final action with read count =0.
The offending action is rejected, the read-before-
final hint below is appended to the next observa-
tion, and the loop resumes without modifying any
other prompt component. We additionally imple-
ment a stricter hint that triggers when the agent
issues an action other than read immediately af-
ter a search call without inspecting any returned
chunk (read-after-search); this hint is not part of the
main READ-GATEcondition and is not separately
ablated with its own reported numbers here, but it
is available within the same corrective-hint frame-
work that motivates the broader family of stronger
procedural gates evaluated in Appendix F.
Read-before-final hint (main READ-GATE).
You attempted to answer before reading any
evidence. You MUST call read(evidence id) on a
promising chunk before giving the final answer.
Read-after-search hint (implemented, not used
in main READ-GATE).
Your last search returned candidate chunks but
you have not read any of them yet. You MUST call
read(evidence id) on at least one of the most
recently returned chunk IDs before answering
or issuing another search.
A.3 Tool Schemas
The two-tool agent uses the following JSON tool
schemas. search performs reciprocal-rank fusion
over BM25 and a 0.6B dense retriever with top-
k= 5 returned to the agent and an internal fusion
k= 60.readreturns the full chunk text by ID.
{"name": "search", "parameters":{"query":
string, "k": integer (default 5)} }
{"name": "read", "parameters":{
"evidence id": string} }
A.4 Judge Prompt
We score answers with a fixed gpt-5-mini judge
at temperature 0.0. The judge receives the ques-
tion, gold reference, and predicted answer, and
returns a single JSON object {"correct": bool,
"rationale": str}. The judge instruction is:
You are a strict evaluator of
question-answering systems. Given a question,
a system-predicted answer, and a reference
answer, decide whether the prediction
conveys the same factual answer as the

reference. Reply with a single JSON object:
{"correct": true|false, "rationale": "..."}.
Be conservative: numeric/extractive answers
require exact factual match; abstractive
answers require all key facts to be present.
Minor wording differences are OK; missing
facts or different facts are NOT OK.
The judge runs on the question + gold +
predicted-answer triple only and does not see re-
trieved chunks, system identity, or run metadata.
B Read-Gate Operational Details
B.1 Read-Gate Overhead
Table 6 reports the observable overhead of READ-
GATEon the full minimal-reasoning cells. Correc-
tions/Q counts rejected actions that consumed a
loop without an executed tool call, after subtracting
the final answer turn; this inferred count is zero for
all matched no-gate cells.
B.2 Read-Chunk Rank Distribution Under
READ-GATE
Figure 4 reports the rank distribution of read events
under the READ-GATE(V5) condition and under
the voluntary policy (V4) on the same gpt-5-mini
minimal backbone. Both conditions are dominated
by rank-1 reads, supporting the §5.2 claim that
READ-GATEenforcesthata read happens rather
thanwhichchunk gets read.
Quantitatively, when READ-GATEforces a read
after blocking a finalization attempt, the agent picks
the rank-1 retrieval result in 50.0% of terminal-read
events and a top-2 chunk in roughly 64%;94% of
forced reads on gold-annotated datasets (HotpotQA
+ 2WikiMultiHopQA) land on a gold-supporting
chunk. Without READ-GATE, the rank-1 share
is60.0% on the same backbone — comparable,
slightly higher. The agent’s chunk choice is there-
fore already rank-1-dominated and is not changed
materially by the gate.
Figure 4: Distribution of read-chunk rank under READ-
GATE(V5) and voluntary policy (V4), gpt-5-mini mini-
mal backbone. Rank-1 dominance is largely backbone
behavior, not gate-induced.B.3 Contain-Acc Check
Table 7 reports Contain-Acc alongside LLM-Acc
for the main minimal-reasoning cells. Contain-Acc
is a string-containment secondary metric and is less
semantically flexible than the LLM judge, but it
moves in the same direction as LLM-Acc under
READ-GATEon all three datasets.
Table 7: LLM-Acc and Contain-Acc for the main gpt-5-
mini minimal cells; READ-GATErows bolded.
Dataset Condition LLM-Acc Contain-Acc
HotpotQA No gate 79.6 76.8
HotpotQAREAD-GATE82.8 80.1
2Wiki No gate 64.4 72.3
2WikiREAD-GATE69.7 75.9
MuSiQue No gate 34.2 32.8
MuSiQueREAD-GATE43.6 41.3
B.4 Conditional Read-Gate Benefit
Figure 5 visualizes the conditional benefit of
READ-GATEcorresponding to Table 3 in the main
text: across datasets, READ-GATE’s accuracy gain
grows with residual discipline error.
10 20 30 40 50
residual discipline error before RG (%)7.5
5.0
2.5
0.02.55.07.510.0RG gain in LLM-Acc (points)HotpotQA2WikiMultiHopQAMuSiQue
minimal
medium
Figure 5: Conditional benefit of READ-GATE. Each
dataset contributes minimal- and medium-reasoning
cells connected by a dashed arrow; gains grow with
residual discipline error. Medium cells use n= 100
matched ablations (boundary evidence only).
B.5 Cost–Accuracy Trade-off
Table 8 reports the cost–accuracy trade-off summa-
rized in §5.6: adding READ-GATEto gpt-5-mini at
minimal reasoning closes 20–24% of the LLM-Acc
gap to gpt-5-mini at medium reasoning, without
changing model weights or the reasoning budget.
The corresponding read/loop overhead profile is in
Table 6.

Table 6: Deployment-relevant overhead for the main gpt-5-mini minimal cells. Ret. toks/Q counts retrieved evidence
tokens rather than total API tokens. Corrections/Q is inferred from loop turns that correspond to rejected actions
rather than executed tools.
Dataset Condition Reads/Q Loops/Q Corrections/Q Ret. toks/Q LLM-Acc
HotpotQA No gate 0.81 3.07 0.00 8,227 79.6
HotpotQA READ-GATE1.03 3.63 0.25 9,011 82.8
2Wiki No gate 0.77 3.54 0.00 12,721 64.4
2Wiki READ-GATE1.06 4.32 0.29 14,783 69.7
MuSiQue No gate 0.35 2.90 0.00 9,711 34.2
MuSiQue READ-GATE1.00 4.97 0.89 14,192 43.6
Table 8:READ-GATEvs. medium reasoning within
the two-tool architecture(within-architecture trade-
off; not vs. best system overall). Adding READ-GATEat
minimal reasoning (shaded row) closes 20–24% of the
LLM-Acc gap.†Medium-reasoning rows are n= 100
matched ablations paired with the READ-GATEmedium
ablation; others n= 1,000 . The same no-READ-GATE
system’s full n= 1,000 cell mean is 90.8/88.8/67.1
(Appendix M), consistent with this n= 100 subsample.
Configuration HotpotQA 2Wiki MuSiQue
5-mini-min 79.6 64.4 34.2
5-mini-min + READ-GATE 82.8 69.7 43.6
gap-closing % 20.8 20.7 23.6
5-mini-med (target)†95.0 90.0 74.0
C Mechanism Ablation Details
Table 9 reports the per-arm overhead of the
three-arm mechanism ablation summarized in
§5.4, across all three datasets (seed=42). All
three arms use gpt-5-mini at minimal reason-
ing effort, gpt-5-mini judge, max loops=10, and
max token budget=128k. The error rate (records
with anerrorfield) is0.0%in all nine cells.
Table 9: Per-arm overhead for the three-arm mechanism
ablation across three datasets. The ctx-inject arm per-
forms the same trigger detection as READ-GATEbut
appends the rank-1 chunk text as a user -role observa-
tion instead of forcing areadtool call.
Dataset Arm LLM-Acc Reads/Q CtxInj/Q Loops/Q Cost/Q
HotpotQA(i) no-READ-GATE79.6 0.81 — 3.07 $0.0052
(ii) READ-GATE82.81.03 — 3.63 $0.0020
(iii) ctx-inject 79.5 0.85 0.41 3.47 $0.0019
2Wiki(i) no-READ-GATE64.4 0.77 — 3.54 $0.0082
(ii) READ-GATE69.71.06 — 4.32 $0.0054
(iii) ctx-inject 57.0 0.74 0.76 4.00 $0.0045
MuSiQue(i) no-READ-GATE34.2 0.35 — 2.90 $0.0057
(ii) READ-GATE43.61.00 — 4.97 $0.0088
(iii) ctx-inject 38.1 0.47 0.96 3.87 $0.0042
Per-arm McNemar tests and effect sizes.
Across the three paired samples, the READ-GATE–
vs–no-READ-GATEcontrast (arm (ii) vs (i)) is
highly significant on every dataset: +3.2 pp
on HotpotQA (paired bootstrap [+2.0,+4.5] ;exact two-sided McNemar p= 1.9×10−7),
+5.3 pp on 2WikiMultiHopQA ( [+3.7,+6.9] ;
p= 3.2×10−11), and +9.4 pp on MuSiQue
([+7.2,+11.6] ;p= 2.3×10−21). The ctx-inject–
vs–no-READ-GATEcontrast (arm (iii) vs (i)) is
−0.1 pp on HotpotQA ( [−1.9,+1.7] ;p= 1.0 ),
−7.4 pp on 2Wiki ( [−9.5,−5.4] ;p= 8.4×10−9),
and+3.9 pp on MuSiQue ( [+1.7,+6.1] ;p=
0.003 ). The direct (iii) vs (ii) contrast favors
READ-GATEon every dataset (HotpotQA b=
20ctx-wins, c= 53 READ-GATE-wins, exact
p= 0.000142 ; 2Wiki b= 27 ,c= 154 ,p=
8.5×10−23; MuSiQue b= 49 ,c= 104 ,p=
1.0×10−5). Delivering the same chunk text with-
out the read action therefore recovers at most a
minority of READ-GATE’s gain (point estimate
+3.9 pp on MuSiQue vs READ-GATE’s +9.4 pp),
and on 2Wiki it is substantially net-negative (ctx-
inject is7.4ppworsethan no-READ-GATE).
Disagreement composition.On the paired
HotpotQA sample, arm (iii) and arm (ii) disagree
on73trajectories: arm (ii) is correct and arm (iii) is
wrong on 53(73%), and arm (iii) is correct and arm
(ii) is wrong on 20(27%). The deflection-pattern
rate (final answers containing conversational
hedges such as“thanks”,“do you want me to
re-evaluate”,“please specify”) is 21/53 = 40% in
arm-(iii)-wrong responses on these disagreement
cases, and 0/53 = 0% in the corresponding
arm-(ii)-correct responses. A representative
case is HotpotQA 5a84e0a45542991dd0999e11
(“Which is farther north, Steel Venom or Wicked
Twister?”): under READ-GATE, the agent reads
chunk 29 and answers“Wicked Twister”correctly;
under ctx-inject, the agent receives the identical
chunk-29 text in its message history and replies
“Thanks — I’ve noted that additional context for
chunk 29. Do you want me to re-evaluate . . . ?”,
never committing to an answer. A second case
(5a7c8a3b55429935c91b5204 ) shows the oppo-

site direction — ctx-inject succeeds on a question
READ-GATEgot wrong — and contributes to the
20ctx-wins. Full case study and analysis script in
revision analysis/r6 context injection.md .
C.1 Channel-vs-Action Decomposition
(Four-Arm Probe)
The three-arm ablation above delivers ctx-inject
text as a user -role observation, leaving open
whether READ-GATE’s advantage reflects the
tool -role delivery channel or the self-issued read
action itself. We add a fourth arm on MuSiQue
(n= 500 , gpt-5-mini minimal, same paired ques-
tion IDs, seed 42) that injects the identical rank-
1 chunk text as a fabricated tool -role response
(a synthetic read tool-call/result pair) without the
agent issuing the corresponding action (Table 10).
Table 10: Four-arm channel-vs-action probe (MuSiQue,
gpt-5-mini minimal, paired n= 500 , seed 42). ∆is
LLM-Acc relative to no-READ-GATE(34.8).
Arm LLM-Acc∆95% CI
no-READ-GATE34.8 — —
ctx-inject (user-role) 39.2+4.4 [+0.8,+7.8]
ctx-inject (tool-role) 40.0+5.2 [+2.4,+8.8]
READ-GATE 43.0 +8.2 [+4.6,+11.6]
Channel alone (tool-role vs. user-role injection,
neither self-issued) accounts for +5.2−4.4 =
+0.8 pp, with heavily overlapping CIs — not signif-
icant. The self-issued read action (READ-GATEvs.
tool-inject, same channel) adds a further +3.0 pp.
READ-GATE’s advantage is therefore most consis-
tent with the agent committing to a self-issued read
action rather than with the tool -role delivery chan-
nel — an action-commitment interpretation that
these controls support but do not conclusively iso-
late — refining, rather than contradicting, the three-
arm result above: forced context in either chan-
nel recovers at most about half of READ-GATE’s
gain, and channel choice is not the active ingredi-
ent. This probe uses a single dataset and backbone
(MuSiQue was chosen for its large READ-GATE
headroom); the three-arm result across all three
datasets (Table 2) remains the primary mechanism
evidence.
D Robustness Details
This appendix expands the robustness checks sum-
marized in §5.6.Threshold and priority.Varying the low-
evidence entity-coverage threshold from 0.6 to
0.9 barely changes minimal-reasoning discipline
rates: HotpotQA ranges from 12.4–13.3%, 2Wiki-
MultiHopQA from 21.9–22.1%, and MuSiQue
from 56.9–57.0%. Changing the priority order
changes absolute subtype counts, but a strict no-
read-only lower bound preserves the same minimal-
reasoning hump (HotpotQA 9.3%, 2WikiMulti-
HopQA 15.4%, MuSiQue 51.5%). Because the
fine-grained subtype boundaries depend on the
entity-coverage threshold, we treat them as approx-
imate and base the H1 overlap result on the coarser
binary discipline indicator, whose cross-extractor
agreement we quantify next.
Extractor robustness for the discipline/post-
gold-read overlap.Table 11 gives the 2 ×4
bucket shares over the 3,807 wrong cases under
regex and spaCy encore websmentity extractors.
The two extractors agree on the binary discipline in-
dicator at Cohen’s κ= 0.628 (observed agreement
82.1% ; expected 52.0% ). The H1 conclusion is not
the precise value of any single share but the qualita-
tive two-axis structure, and that structure is invari-
ant to extractor choice: under both extractors the
both-trigger overlap stays in a tight [11.2%,13.1%]
band—far below the 60% level at which the two
axes would collapse into one—and the discipline-
only share exceeds the post-gold-read-only share
by more than two to one. The residual extractor
disagreement is concentrated in borderline cover-
age cases near the 0.8threshold; because spaCy
and regex shift such cases in thesamedirection
(spaCy raises both the discipline-only and both-
trigger shares), the rank order and the sub-threshold
overlap are preserved rather than reshuffled.
Table 11: Multi-label buckets over the 3,807
wrong cases, by entity extractor (regex / spaCy
encore websm). Cohen’sκ= 0.628.
Extractor disc-only post-only both neither
regex 46.5% 21.4% 11.2% 20.9%
spaCy 50.2% 19.5% 13.1% 17.2%
Bootstrap intervals.We compute paired boot-
strap 95% CIs over question IDs (1,000 itera-
tions, seed 42). The minimal-reasoning READ-
GATEaccuracy gains exclude zero on all three
datasets: HotpotQA +3.2 pp[+2.0,+4.5] , 2Wiki-
MultiHopQA +5.3 pp[+3.8,+6.9] , MuSiQue
+9.4 pp[+7.5,+11.5] . Bootstrap estimates for

the induced-read contrast on the zero-read sub-
set are consistent with the point estimates in Ta-
ble 1: HotpotQA +14.7 pp[+9.2,+20.2] , 2Wiki-
MultiHopQA +20.1 pp[+14.4,+25.4] , MuSiQue
+14.4 pp[+11.5,+17.3] . The Gemini thinking-
budget paired effect is−4.4pp[−7.3,−1.3].
Stratified label permutation.The label-
permutation test reported in §5.5 uses a within-
cell-stratified null draw over all 12,000 trajectories;
this preserves cell-level marginals while ran-
domizing the discipline / post-gold-read label
assignment. The observed statistic exceeds every
one of the 10,000 permutation samples, giving
empirical p <10−4that is robust to cell-marginal
differences.
Multiple-testing adjustment.We apply
Benjamini–Hochberg (BH) FDR and Bonferroni
corrections across the 11 tests supporting Ta-
ble 4: three READ-GATEmain effects ( Pdisc,Ppost
marginal, correctness), six effort contrasts (min and
med vs. gpt-4o-mini, on Pdisc/Ppost/correctness),
the effort ×axis interaction reported in Table 4,
and the label-permutation exchangeability test.
Nine GLM-based tests have original p <10−10
(BHqand Bonferroni pboth <10−10); effort
(min vs. 4o) →P post hasp <10−4(adjusted
similarly); the permutation test reaches its 10,000-
sample resolution limit ( p <10−4; Bonferroni
p= 0.0011 ). All 11 survive both corrections at
α= 0.05 . Surviving FDR is necessary but not
sufficient: the marginal READ-GATE →P postOR
is significant after correction yet mechanically
exposure-confounded, so the within-strata row
in Table 4 is the appropriate behavioral estimate
(R-2).
D.1 Cross-Family Generalization (Qwen2.5)
The main H2 result (§5.3) is established on the Ope-
nAI controller family. As a generalization check,
we run the same two-tool interface, retrieval index,
top-k, and loop budget with open-weight Qwen2.5-
Instruct controllers (Qwen Team, 2024) (3B and 7B
on all three datasets, plus 14B on HotpotQA; gpt-
5-mini judge; matched n= 200 per cell, READ-
GATEon/off). Table 12 reports Ediscand the paired
READ-GATE∆for each cell.Table 12: Qwen2.5 cross-family cells (matched n=
200, gpt-5-mini judge). Ediscis the no-READ-GATE
discipline-failure rate; ∆READ-GATEis the paired
LLM-Acc change. Only musique-3B is individually
significant at this sample size.
Model DatasetE disc(%)∆READ-GATE95% CI McNemarp
3B HotpotQA 51.0−2.0 [−8.0,+4.0]0.627
7B HotpotQA 10.5−1.0 [−4.5,+2.5]0.774
14B HotpotQA 31.5−2.5 [−8.0,+2.5]0.442
3B 2Wiki 78.0+1.5 [−4.0,+7.5]0.743
7B 2Wiki 17.5−2.0 [−8.0,+3.5]0.597
3B MuSiQue74.0 +6.0[+1.0,+11.5]0.043
7B MuSiQue 19.5+1.0 [−4.0,+6.0]0.845
Across the 7 cells, Ediscand∆READ-GATE
correlate at Pearson r= +0.629 (Spearman
ρ= 0.393 ), close to the OpenAI-family value of
+0.728 reported in §5.3: high- Edisccells (musique-
3B, 2wiki-3B) get the largest READ-GATEgains,
and low- Edisccells cluster near zero or negative,
reproducing the same qualitative dose-response
shape on an independent model family.
This is a direction, not a power, result. At
n= 200 /cell most individual cells are not indepen-
dently significant; only musique-3B is ( +6.0 pp
[+1.0,+11.5] ,p= 0.043 ), and all three Hot-
potQA cells are flat or slightly negative regardless
of model size. The claim this table supports is
narrow: theframework’s structural relationship—
READ-GATE’s gain scales with residual discipline
error — generalizes across model families. It is
not a claim that READ-GATEhelps uniformly; on
this family, as on the OpenAI family’s medium-
reasoning cells, low- Ediscoperating points show no
benefit or a small cost.
E Gemini Representative Trace and
Break Decomposition
Break decomposition.The break decomposition
is consistent across both datasets: roughly half
of the breaks are direct thinking-caused zero-read
transitions (HotpotQA 41.9% , 2Wiki 50.5% ), and
roughly half retain the same read count but produce
a different wrong answer (HotpotQA 50.0% , 2Wiki
45.5% ). The 2Wiki effect is in fact stronger than
on HotpotQA, with a four-fold larger zero-read
shift, suggesting the failure mode is robust rather
than dataset-specific within Wikipedia-style multi-
hop QA. Headline zero-read shifts: HotpotQA
52.3%→58.0% (+5.7 pp); 2WikiMultiHopQA
40.8%→65.6%(+24.8pp).

Truncation artifact.A subset of the same-
read breaks on 2Wiki are mid-sentence trun-
cations driven by a thinking budget×
maxoutput tokens interaction at the API surface,
rather than evidence-skipping; the headline
zero-read transition rate ( 50.5% of breaks) is
unaffected by this artifact.
Representative trace.A representative Gemini
break is the question“What is the genus of the viral
disease that has symptoms such as fever, chills, loss
of appetite, nausea, muscle pains, and headaches,
and has a chance of causing liver damage?”(gold
answer:Flavivirus). With thinking disabled, the
agent searches, expands a chunk that links yellow-
fever symptoms to the genus, and answersFla-
viviruscorrectly (one read, three loops). With a
1024-token thinking budget, the agent emits 1024
internal thinking tokens, searches once, and final-
izes without any read action; the answer text re-
states the symptoms but never names the genus.
The trajectory is wrong despite spending more com-
pute.
F Procedural Gate-Family Probe
READ-GATEis the minimal procedural gate used
in the main experiments. We also ran a small ap-
pendix probe on broader deterministic gates to
measure the trade-off from adding stronger run-
time invariants (Table 13). The variants reuse the
same saved detector checks: noread blocks fi-
nal answers before any read; snippet additionally
blocks answers whose entities appear only in search
snippets; lowev additionally blocks final answers
when read text has low question-entity coverage;
full adds a format-mismatch check. These vari-
ants use rules-only corrective hints and no learned
controller.
The result supports the conservative main de-
sign choice. As Figure 6 makes visible, a snippet-
only extension is relatively stable, but low-evidence
and format gates introduce a stronger precision–
intervention trade-off. We therefore use READ-
GATEas the main intervention: it is the weak-
est runtime invariant that directly targets pre-
evidence termination without adding heuristic cov-
erage thresholds or a learned controller.
G Entity-Coverage Labeling
The discipline subtypes snippet-only final
and low-evidence final require detecting
whether question-side named entities appear inretrieved snippets and read chunks (§3.2). We
use a deterministic matching pipeline. When
spaCy is available, we run the encore websm
pipeline on the question, retrieved snippets,
and read chunks, and treat its named-entity
spans as coverage candidates. When spaCy is
unavailable or returns no spans, we fall back to a
regex that captures capitalized multi-word phrases
([A-Z][A-Za-z0-9.-]+(?:\s[A-Z][A-Za-z0-9.-]+)* )
and standalone numeric tokens, which together
cover proper-noun answer entities and date or
count answers common in HotpotQA / 2WikiMul-
tiHopQA / MuSiQue. Matching is case-insensitive
over the union of all question entities; an entity is
covered if any of its surface forms appears verbatim
in the candidate text. Coverage rate is the fraction
of question entities matched in read chunks;
a trajectory is labeled low-evidence final
when this rate falls below 0.8. Because subtype
assignment depends on this coverage threshold,
the fine-grained snippet-only /low-evidence
split is approximate; the H1 overlap analysis in
Appendix D therefore rests on the coarser binary
discipline indicator, on which the two extractor
paths agree at Cohen’sκ= 0.628.
G.1 Entity-Matcher Validation
We validate the regex entity matcher against an
author-verified, hand-labeled sample ( n= 60 : 30
questions and 30 gold reference answers, stratified
across HotpotQA and 2WikiMultiHopQA; every
label reviewed by an author). Table 14 reports pre-
cision, recall, and F1 under contains-match, the
same matching rule the discipline detector uses
operationally. Precision is 1.000 (no false posi-
tives among 69 predicted entities; 95% Wilson CI
[0.947,1.000] ); the recall gap ( 0.734 overall) is
concentrated almost entirely in single-token enti-
ties (recall 0.148 ), which the regex cannot match
by construction (it requires ≥2 consecutive cap-
italized tokens), versus 0.966 recall on the multi-
word entities it targets. Because the matcher pro-
duced no spurious entity on this sample, the re-
ported discipline-failure rates are a conservative
lower bound rather than an inflated estimate.

Read-Gate +snippet +lowev full020406080Accuracy (%)82 828581HotpotQA
Read-Gate +snippet +lowev full020406080Accuracy (%)646765 642WikiMultiHopQA
0.00.51.01.52.02.53.0
Corrections/Q
0.000.462.072.13
0.00.51.01.52.02.53.0
Corrections/Q
0.000.842.182.63Figure 6: Precision–intervention trade-off across gate variants. Bars: LLM-Acc; lines: Corrections/Q. +snippet
stays below 1 Corr./Q with neutral-to-positive accuracy; +lowev andfull exceed 2 Corr./Q for accuracy within ±4
pts of READ-GATE. Matchedn= 100.
Table 13: Gate-family probe on matched n= 100 samples (smaller than main-cell tables 6–7). Corr./Q is mean
interventions per question; Net vs. READ-GATEis paired fixes minus breaks. Broader gates help on some cells but
are intervention-heavy and dataset-sensitive.
Dataset Variant Acc Zero-read Corr./Q Net vs. READ-GATE
HotpotQA READ-GATE(n= 100) 82.0 0.0 0.00 –
HotpotQA no-read + snippet 82.0 0.0 0.460
HotpotQA no-read + lowev 85.0 0.0 2.07+3
HotpotQA full 81.0 0.0 2.13−1
2Wiki READ-GATE(n= 100) 64.0 0.0 0.00 –
2Wiki no-read + snippet 67.0 0.0 0.84+3
2Wiki no-read + lowev 65.0 0.0 2.18+1
2Wiki full 64.0 0.0 2.630
Table 14: Entity-matcher validation against an author-
verified hand-labeled sample ( n= 60 ), contains-match.
Precision Recall F1
Overall 1.000 0.734 0.847
Single-token entities — 0.148 —
Multi-word entities — 0.966 —
H Discipline-Failure Subtype
Decomposition
Section 3.2 defines three discipline-failure sub-
types: no-read final, snippet-only final, and low-
evidence final. Table 15 reports their relative share,
as a percentage of Edisc(the discipline failures in
that cell), across the nine main dataset ×backbone
cells.Table 15: Discipline-failure subtype shares (% of Edisc
in that cell) across the nine main cells.
Dataset Backbonen Edisc No-read Snippet-only Low-ev.
HotpotQA 4o-mini 76 3.9 9.2 86.8
HotpotQA 5mini-min 133 69.9 1.5 28.6
HotpotQA 5mini-med 45 37.8 6.7 55.6
2Wiki 4o-mini 128 7.8 11.7 80.5
2Wiki 5mini-min 221 69.7 3.6 26.7
2Wiki 5mini-med 57 38.6 26.3 35.1
MuSiQue 4o-mini 278 15.5 17.6 66.9
MuSiQue 5mini-min 570 90.4 2.1 7.5
MuSiQue 5mini-med 217 56.7 22.6 20.7
Pooled (1,725 discipline failures) 56.8 9.3 33.9
Pooled over all nine cells, 56.8% of discipline
failures are strict no-read finals, 9.3% take the
prompt-permitted snippet-only path, and 33.9%
are low-evidence. Because it requires no entity-
matching heuristic, no-read is the primary, non-
heuristic discipline signal; the two NER-based
subtypes are secondary diagnostics whose relative
weight is backbone-dependent (under gpt-4o-mini,
which rarely skips reads outright, low-evidence
dominates instead). Notably, 43.2% of discipline
failures ( 745/1,725 ) occur despite read count≥1

(snippet-only plus low-evidence): READ-GATEis
framed as removing the zero-read floor, not as guar-
anteeing adequate post-read inspection.
I Zero-Read Provenance: Leakage vs.
Unsupported Guesses
Using the exact chunk IDs each search call re-
turned (already logged per trajectory), we check
whether a zero-read answer’s predicted entity is
visible anywhere in the retrieved-but-unread search
results, separating snippet leakage from genuinely
unsupported guesses.
Wrong zero-read answers.Pooled across the
nine non-READ-GATEmain cells ( n= 980 wrong
zero-read trajectories, 300-character snippet def-
inition): 23.7% (232) show snippet leakage (the
predicted entity appears in a retrieved-but-unread
chunk), 58.5% (573) show no visible evidence (no
trace of the guessed entity in anything retrieved),
5.1% (50) never called search at all, and the re-
maining 12.8% (125) yield no informative entity in
the predicted answer to match against (e.g., yes/no
or clarification-style answers). The four categories
are mutually exclusive and sum to 100% . Leakage
is a real but minority contributor; most zero-read
errors are unsupported guesses rather than surfaced-
but-ignored answers.
Correct zero-read answers.Restricted to the
HotpotQA+2WikiMultiHopQA gold-annotated
subset ( n= 282 classifiable correct zero-read
cases), 19.5% (55) show clean snippet leakage
of the correct entity, 68.4% (193) retrieved the
gold-supporting chunk but without a verbatim
entity match (a candidate for genuine cross-snippet
synthesis without a full read), and 12.1% (34)
show no retrieved connection to gold evidence
at all, consistent with parametric knowledge.
Leakage is again a minority explanation, on both
the wrong and correct sides.
Critically, leakage is not the same question as
whether a read action occurred: even when the an-
swer surfaces in a snippet, finalizing without read-
ing still skips the verification step our framework
measures.
Snippet-length sensitivity.Because “snippet” is
an operational choice, Table 16 sweeps the char-
prefix definition from 100 characters to the full
chunk. Leakage rises with a more generous defi-
nition, from 12.2% to60.3% on wrong zero-read
answers; we report the conservative 300-characterdefinition ( 23.7% ) as the headline, with this sweep
reported as a bounded robustness range rather than
a point estimate.
Table 16: Snippet-length sensitivity sweep (char-prefix
definition; 9 non-READ-GATEmain cells).
Snippet definition Wrong-ZR leak % Correct-ZR leak %
100 chars/chunk 12.2 8.7
300 chars/chunk (default) 23.7 21.1
600 chars/chunk 30.9 32.6
1200 chars/chunk 40.8 44.4
Full chunk 60.3 73.0
J Differential Intervention Visualization
Figure 7 visualizes the intervention signatures in
the(Pdisc, Ppost)plane. The corresponding odds
ratios and tests are reported in Table 4 in the main
text.
K Prompt-Only Control
Table 17 reports the prompt-only baseline summa-
rized in §5.2, extended to all three datasets (gpt-5-
mini minimal, paired by question ID). The strict-
prompt variant removes the snippet exception from
the system prompt and explicitly forbids answering
before a successful read, but uses no environment-
level gate. Across the three datasets, the strict
prompt reduces zero-read trajectories, but its ac-
curacy effect is unreliable: +0.0 pp on HotpotQA,
−2.8 pp on 2WikiMultiHopQA, and +3.2 pp on
MuSiQue. READ-GATE, by contrast, recovers
+3.2 /+5.3 /+9.4 pp on the same cells and reduces
zero-read to ≤3.8% everywhere. On every dataset,
the paired (RG vs strict-prompt) contrast is signifi-
cant (McNemarp≤0.005).
Table 17: Prompt-only control across three datasets, gpt-
5-mini minimal. Prompting reduces zero-read behavior
on all datasets but does not match READ-GATE’s accu-
racy gain on any dataset, and is net-negative on 2Wiki.
McNemar pis the paired READ-GATEvs strict-prompt
LLM-Acc test.
Dataset Condition Zero-read LLM-Acc vs strictp
HotpotQAOriginal prompt 21.8% 79.6 —
Strict prompt only 12.4% 79.6 —
READ-GATE 0.3% 82.8 0.0048
2WikiOriginal prompt 26.4% 64.4 —
Strict prompt only 20.7% 61.6 —
READ-GATE 0.3% 69.7 2.5×10−7
MuSiQueOriginal prompt 65.5% 34.2 —
Strict prompt only 48.1% 37.4 —
READ-GATE 3.8% 43.6 1.5×10−5

4 6 8 10 12 14
Pdisc  discipline failure (%)
681012141618Ppost  post-gold-read error (%)
HotpotQA
5 10 15 20 25
Pdisc  discipline failure (%)
7.510.012.515.017.520.022.525.02WikiMultiHopQA
20 30 40 50 60
Pdisc  discipline failure (%)
n/aPpost not measurable
(no gold evidence annotations)MuSiQue
4o-mini 5mini-min 5mini-min + Read-Gate 5mini-med Model upgrade (4o-mini  5mini-min)
Figure 7: Differential intervention signatures in the (Pdisc, Ppost)plane. READ-GATEmainly reduces pre-evidence
failures; higher effort moves both indicators with a different slope. Markers:4o=gpt-4o-mini;min/med=gpt-5-mini
minimal/medium;min+READ-GATE=min with READ-GATE. MuSiQue’sP post=0(no gold).
L Cross-Judge Robustness Details
This appendix expands the cross-judge robust-
ness paragraph in §4.4. We re-judged n=
450 stored (question, gold, prediction) triples
(75 per cell across HotpotQA, 2WikiMulti-
HopQA, and MuSiQue ×no-READ-GATE/ READ-
GATE) under Gemini 2.5 Pro (temperature 0.0,
thinking budget = 256 ), using the verbatim
judge prompt in Appendix A.4. The sample is
stratified by the gpt-5-mini-judge agreement land-
scape between no-READ-GATEand READ-GATE
on the same question (50% disagreement, 30%
both-correct, 20% both-incorrect) so that the κ
statistic is informed by the cases most likely to
flip across judges. Stratum-wise raw agreement
is0.952 on the disagreement stratum (228 judg-
ments), 0.970 on both-correct (132), and 0.978 on
both-incorrect (90).
Per-cell κand reweighted LLM-Acc.Ta-
ble 18 reports Cohen’s κalongside the stratum-
reweighted Gemini LLM-Acc and its gap to the
paper headline. All six cells fall in the “almost
perfect” band ( κ≥0.80 ); the reweighted Gemini
estimates are within ±3.7 pp of the paper head-
line on every cell. The largest gaps are uniform
shifts across conditions that cancel in ∆(Gem-
ini HotpotQA ∆ = +3.2 , identical to the pa-
per; Gemini 2Wiki ∆ = +7.4 vs the paper’s
+5.3 ; Gemini MuSiQue ∆ = +7.3 vs the pa-
per’s +9.4 ). The family-bias hypothesis would
predict ∆Gemini≪∆ gpt5mini ; we observe direction-
preserving and similar-magnitude ∆across all
three datasets.
Disagreement case study.gpt-5-mini and Gem-
ini 2.5 Pro disagree on 17/450 cases (3.8%). The
pattern is bidirectional: a small share where GeminiTable 18: Cross-judge agreement and stratum-
reweighted LLM-Acc on the n= 450 sample. “Paper”
is gpt-5-mini on the full n= 1000 cell; “Gemini-rw”
is the Horvitz–Thompson-reweighted Gemini 2.5 Pro
estimate on the 75-per-cell sample.
CellκPaper Gemini-rw Gap
HotpotQA, no-READ-GATE0.940 79.6 75.9−3.7
HotpotQA, READ-GATE0.932 82.8 79.1−3.7
2Wiki, no-READ-GATE0.886 64.4 63.6−0.8
2Wiki, READ-GATE0.871 69.7 71.0+1.3
MuSiQue, no-READ-GATE0.970 34.2 34.20.0
MuSiQue, READ-GATE0.860 43.6 41.5−2.1
Pooled (all 6)0.924— — —
∆HotpotQA —+3.2 +3.2 0.0
∆2Wiki —+5.3 +7.4 +2.1
∆MuSiQue —+9.4 +7.3−2.1
catches a subject-mismatch, internal contradiction,
or logic inversion that gpt-5-mini missed; a similar
share where gpt-5-mini correctly accepts a para-
phrase or partial answer that Gemini penalizes on
format strictness; the rest are genuinely borderline.
A representative Gemini-catches case is the
2Wiki question“Where was the place of death
of Bess Taffel’s husband?”, where the prediction
confuses subject (gives Bess Taffel’s place of death,
not her husband’s) but contains the token-matching
“Los Angeles”; gpt-5-mini marks it correct, Gem-
ini marks it incorrect. Such cases reflect a slight
strictness difference, not a family-bias artifact: the
disagreements are small in number, bidirectional,
and do not concentrate on gpt-5-mini-family gener-
ations. Full case study and reweighting derivation
inrevision analysis/r1b cross judge.md.
M Architecture ×Backbone Comparison
The runtime findings in the main text fix the agent
interface. We now vary it, comparing the two-tool
design against A-RAG’s three-tool hierarchical in-

Table 19:Backbone-conditional interface effect.Two-
tool interface vs. A-RAG’s three-tool hierarchical inter-
face; ∆is 2-tool minus A-RAG in LLM-Acc points;∗
marks paired McNemar p <0.05 . Same retrieval in-
dex, top- k, loop budget, and judge throughout.†marks
cells cited from Du et al. (2026); all other A-RAG cells
(4o-mini, 5mini-min rows) are re-measured under our
harness.
Backbone Dataset A-RAG 2-tool∆
4o-mini HotpotQA 77.6 81.6 +4.0∗
4o-mini 2Wiki 53.8 70.1 +16.3∗
4o-mini MuSiQue 41.4 46.6 +5.2∗
5mini-min HotpotQA 73.6 79.6 +6.0∗
5mini-min 2Wiki 67.5 64.4 −3.1
5mini-min MuSiQue 40.7 34.2 −6.5∗
5mini-med HotpotQA 94.5†90.8 −3.7
5mini-med 2Wiki 89.7†88.8 −0.9
5mini-med MuSiQue 74.1†67.1 −7.0∗
terface (keyword search, semantic search, read)
under the same index, top- k, loop budget, judge,
and reasoning effort (Table 19). A-RAG numbers at
gpt-5-mini medium are taken from Du et al. (2026);
A-RAG at gpt-4o-mini and at gpt-5-mini minimal
are our re-measurements under the same retrieval
index, top-k, loop budget, judge, and protocol.
The 2-tool 5mini-med row reports the full
n= 1,000 no-READ-GATEcell mean for this
backbone. §5.3 additionally reports a matched
n= 100 subsample of the same system,
paired question-by-question with the READ-GATE
medium ablation, giving 95.0/90.0/74.0 on Hot-
potQA/2Wiki/MuSiQue (Table 3) — a consistent
measurement of the same no-READ-GATEsystem
on a different, smaller question subsample used for
the paired READ-GATEcomparison, not a different
system or a labeling error.
The ordering is monotone in backbone strength.
On gpt-4o-mini, the two-tool interface wins on
all three datasets ( +4.0 ,+16.3 ,+5.2 , all paired-
significant). On gpt-5-mini at minimal reasoning,
the advantage holds on HotpotQA but reverses on
2WikiMultiHopQA and MuSiQue. On gpt-5-mini
at medium reasoning, the hierarchical interface
wins on all three datasets ( −3.7 ,−0.9 ,−7.0 ). Nei-
ther architecture is universally better.
A natural reading is that interface granularity
trades decision burden against controllability: the
two-tool interface folds the lexical-vs.-semantic
choice into the environment via reciprocal-rank fu-
sion, removing one decision axis from the model.
A weaker controller benefits from the reduced bur-den; a stronger controller with more reasoning ef-
fort can exploit the extra control axis when it can
reliably route each subquery to a retrieval mode.
This is complementary to the READ-GATEresult:
both findings say the right agentic-RAG design
is regime-dependent — simplified interface plus
runtime gate at the low-cost operating point, hier-
archical interface at the high-reasoning operating
point — without changes to model weights or the
retriever.