# Self-evolving Agentic Customer Support System at LinkedIn

**Authors**: Chih Hui Wang, Mengdie Tu, Qianyun Zhang, Wei Wu, Lili Zhou, Mingqi Shen, Changshuai Wei

**Published**: 2026-08-10 20:51:18

**PDF URL**: [https://arxiv.org/pdf/2608.10224v1](https://arxiv.org/pdf/2608.10224v1)

## Abstract
Enterprise support agents operate in rapidly changing environments where policies, product capabilities, and knowledge bases evolve continuously, making static assistants brittle and costly to maintain. We present LinkedIn's self-evolving agentic support system, which integrates retrieval-augmented generation with evolutionary auto-prompting and a modular, production-aligned evaluation framework to enable safe, continuous improvement without retraining foundation models. The system treats prompts, retrieval, and evaluation as a closed-loop, versioned workflow with operational guardrails. Offline simulations and ablations show clear quality gains over vanilla RAG and baseline agents, including reduced hallucinations and improved response completeness. In a two-week user-randomized A/B test on LinkedIn's production support traffic, the integrated self-evolved workflow increased QA self-serve by 9.0 percentage points, cancellation self-serve by 4.8 points, and routing accuracy by 30.6 points. These results demonstrate a practical path to scalable, self-evolving AI agents in real-world enterprise settings.

## Full Text


<!-- PDF content starts -->

Self-evolving Agentic Customer Support System at LinkedIn
Chih Hui Wang*, Mengdie Tu*, Qianyun Zhang*, Wei Wu*,
Lili Zhou, Mingqi Shen, Changshuai Wei†
LinkedIn
{ciwang, metu, qazhang, wwu1, lizhou, minshen, chawei}@linkedin.com
Abstract
Enterprise support agents operate in rapidly
changing environments where policies, prod-
uct capabilities, and knowledge bases evolve
continuously, making static assistants brittle
and costly to maintain. We present LinkedIn’s
self-evolving agentic support system, which in-
tegrates retrieval-augmented generation with
evolutionary auto-prompting and a modular,
production-aligned evaluation framework to
enable safe, continuous improvement without
retraining foundation models. The system
treats prompts, retrieval, and evaluation as a
closed-loop, versioned workflow with opera-
tional guardrails. Offline simulations and abla-
tions show clear quality gains over vanilla RAG
and baseline agents, including reduced halluci-
nations and improved response completeness.
In a two-week user-randomized A/B test on
LinkedIn’s production support traffic, the in-
tegrated self-evolved workflow increased QA
self-serve by 9.0 percentage points, cancella-
tion self-serve by 4.8 points, and routing accu-
racy by 30.6 points. These results demonstrate
a practical path to scalable, self-evolving AI
agents in real-world enterprise settings.
1 Introduction
Enterprise customer support is increasingly medi-
ated by AI agents, yet real-world deployments face
a reliability problem qualitatively different from
standard chatbot benchmarks. At LinkedIn, sup-
port spans many product surfaces, multiple lines of
business, and dozens of languages, while the un-
derlying ecosystem changes continuously: product
launches reshape the knowledge base, content is
published or deprecated, retrieval indices are re-
freshed, prompts evolve, and underlying LLMs and
tool APIs change over time. Even small upstream
changes can induce regressions—hallucinations,
stale guidance, broken tool flows, or inconsistent
*Equal contribution
†Corresponding authormultilingual tone—making a “set-and-forget” assis-
tant brittle and expensive to maintain. Traditional
production support agents—built on a handcrafted
prompt, a fixed retrieval pipeline, and periodic hu-
man QA—do not scale with this velocity, and lack
a principled feedback loop that can detect regres-
sions, localize failures to specific components, and
adapt without heavy human intervention.
Three lines of recent work motivate the design
we present.Prompt optimizationtreats prompts
as programs, with score-based search (APE (Zhou
et al., 2023)), LLM-as-optimizer (OPRO (Yang
et al., 2024)), evolutionary search (PromptBreeder,
EvoPrompt (Fernando et al., 2023; Guo et al.,
2024)), and declarative compilation (DSPy (Khat-
tab et al., 2023)) all showing that prompts can be
evolved offline against task metrics and shipped
as versioned artifacts.LLM-as-a-judgeevaluation
enables scalable assessment of multi-dimensional
quality (Zheng et al., 2023; Liu et al., 2023),
with documented biases that can be mitigated
through structured rubrics, length control (Dubois
et al., 2024), and open evaluator models (Kim
et al., 2024).Retrieval-augmented generation
grounds responses in proprietary knowledge (Lewis
et al., 2020b; Karpukhin et al., 2020; Gao et al.,
2023a), and recent self-reflective variants (Self-
RAG, CRAG (Asai et al., 2024; Yan et al., 2024))
treat retrieval as an explicit agent action rather than
a preprocessing step, aligning with reasoning-and-
acting paradigms (Yao et al., 2023).
For enterprise support, the operational implica-
tion of these advances is that prompts, retrievers,
and evaluators all becomeversioned artifactsthat
can change frequently—and often function like pol-
icy changes affecting routing, escalation thresholds,
safety posture, and brand tone across multiple prod-
ucts and languages. A self-evolving system there-
fore requires not just optimization, but the engi-
neering discipline to deploy these changes safely:
regression testing, staged rollout, gated promotion,
1
arXiv:2608.10224v1  [cs.AI]  10 Aug 2026

and rollback. Our evaluation layer must in turn be
reliable enough to drive these decisions at scale,
robust to knowledge drift, and decomposed into
interpretable signals so failures can be attributed
and acted on.
We co-evolve the three layers under strict version
control, deliberately avoiding fine-tuning because
support knowledge changes too quickly for re-
training cycles and because retrieval-based knowl-
edge injection often outperforms unsupervised fine-
tuning for factual updates (Ovadia et al., 2024;
Soudani et al., 2024). Our evaluation layer is struc-
tured as specialized evaluator agents (Wu et al.,
2023), and our operational “memory” is imple-
mented as bounded, auditable, versioned artifacts
rather than unconstrained long-horizon memory
(Packer et al., 2023; Shinn et al., 2023).
Contributions.We (1) describe an end-to-end
self-evolving support agent that remains reliable
under continual upstream change via closed-loop
evaluation and optimization; (2) present an auto-
matic prompt evolution engine suitable for enter-
prise constraints (tone, policy, multilingual); (3)
propose a modular, multi-signal evaluation frame-
work actionable for debugging at scale; and (4)
detail a version-controlled RAG-and-tools layer
supporting reproducibility, safe rollout, and roll-
back.
2 Methodology
2.1 Overall system
Figure 1 shows the Support AI Agent as a closed-
loop, self-evolving system in which prompting, re-
trieval, and evaluation form an explicit feedback
cycle. At inference time the agent is conditioned
by a system prompt from the Automatic Prompt
Engine that encodes task instructions, tone, and
implicit policies for reasoning depth and tool use.
RAG is exposed as an explicit action rather than a
preprocessing step, letting the agent decide when to
retrieve and how to integrate evidence, consistent
with reasoning-and-acting paradigms (Yao et al.,
2023; Schick et al., 2023). Outputs are scored by
a modular evaluation framework that decomposes
quality into grounding, intent alignment, multilin-
gual fidelity, and stylistic compliance using rule-
based diagnostics and LLM-based judges (Zheng
et al., 2023); aggregate fitness signals feed back
into the prompt engine, which refines the system
prompt via evolutionary search. The architecture
exposes two coupled loops—an inner inferenceloop (agent ↔RAG↔content lake) and an outer
optimization loop (auto-prompt →agent→eval-
uator→auto-prompt)—enabling scalable self-
improvement and rapid adaptation across products
and languages without modifying the base model
or retrieval corpus.
2.2 Auto-Prompt
Manual prompt development is a major bottleneck
in production: in our deployment, a single line-
of-business iteration can take weeks. We replace
this with an Automatic Prompt Engineering Engine
(“Auto-Prompt”) that formulates prompt optimiza-
tion as evolutionary search inspired by genetic al-
gorithms (Holland, 1992; Whitley, 1994). Domain-
specific business rules guide an LLM-driven initial-
ization stage, after which a population of candidate
prompts evolves through selection, crossover, and
mutation, evaluated each generation against task-
specific metrics; only the strongest survive into
the next iteration. The full pseudocode is given in
Algorithm 1.
Evolutionary search fits this setting be-
cause prompt optimization is black-box, non-
differentiable search over discrete text, with
fitness supplied by LLM and rule-based evalu-
ators. Unlike single-candidate hill climbing, a
population preserves diverse combinations of tone,
tool-use policy, reasoning depth, and response
structure; crossover recombines complementary
instructions discovered in different lineages, while
mutation provides local exploration. APE and
OPRO optimize prompts through score-guided
generation (Zhou et al., 2023; Yang et al., 2024),
and DSPy compiles declarative programs against
metrics (Khattab et al., 2023); our use of con-
strained evolution emphasizes population diversity,
recombination, and immutable enterprise-policy
filters. The ablation in Table 2 tests whether both
evolutionary operators are necessary.
To prevent evolutionary drift from violating
safety or policy requirements, business rules are
treated as immutable hard constraints rather than
optimization objectives: violating prompts are fil-
tered or assigned zero fitness prior to selection.
Crossover and mutation therefore operate within a
constrained prompt space that preserves required
behaviors by construction, ensuring evolutionary
gains target accuracy without eroding compliance
or product invariants.
2

Support AI Agent
(reason + act)Auto-Prompt
Engine
(evolutionary search)Modular Evaluation
Framework
(scores & diagnostics)
RAG Tool
(retrieve evidence)
Versioned Content Lake
(help/learn/docs)User Query
+ Chat ContextAgent Response
(+ citations)
optimized
system prompt responses + traces
fitness signals (multi-metric)tool invocation
(as action)grounding docs
+ provenance
retrieve top-k
evidencerequestquality report
& failure typesClosed-loop self-evolution
What evolves:prompt policies (tool use, tone, structure)
What stays fixed per iteration:base LLM1+ content snapshot
Figure 1: Self-evolving Support AI Agent architecture.
2.3 Retrieval Augmented Generation (RAG)
Support knowledge is proprietary and changes con-
stantly, so we ground responses in retrieved content
rather than parametric model knowledge (Lewis
et al., 2020b; Guu et al., 2020). We depart from
fixed retrieval pipelines and expose RAG as atool
the agent invokes within its reasoning loop (Yao
et al., 2023; Schick et al., 2023), letting the model
decide when to retrieve and how to query, mirror-
ing recent agentic RAG systems (Press et al., 2023;
Shinn et al., 2023).
Our unified knowledge base consolidates three
content sources: help articles, learning content,
and product documentation microsites. They are
refreshed by an ETL (Extract–Transform–Load)
pipeline that chunks, embeds, facets, and re-
tires content before loading it into the versioned
content lake. Retirement is enforced through
a snapshot pointer, so each refresh scopes re-
trieval to the latest snapshot and supersedes prior
content without deleting it. Each document is
tagged with three facets—product(line of busi-
ness),locale(7+ languages), andsnapshot date—
enabling scoped, reproducible retrieval (Lin et al.,
2021; Gao et al., 2023b). At query time, hybrid
dense–sparse retrieval is followed by semantic re-
ranking (Berntson, 2023; Nogueira and Cho, 2019),
returning ranked grounding documents with prove-nance (URL, locale, source) for citation and audit.3
2.4 Evaluation Framework
Operating across 36+ languages and many prod-
uct lines with a constantly drifting knowledge
base, we cannot rely on fixed gold sets or static
benchmarks (Lewis et al., 2020a; Longpre et al.,
2021). We instead use a modular, agent-aware
framework that decomposes quality into inter-
pretable dimensions—grounding, intent under-
standing, tool-execution correctness, and multilin-
gual fidelity—each scored by specialized evalua-
tors combining rule-based checks with LLM judg-
ments. Multilingual quality is assessed by dedi-
cated modules that detect untranslated spans, ver-
ify terminology, and measure fluency via XGLM
perplexity (Lin et al., 2022), while an LLM evalu-
ator scores semantic fidelity under realistic dialog
conditions. Production monitoring spans all sup-
ported locales, while the controlled benchmark in
Section 4.4 focuses on English ↔Chinese as a chal-
lenging distant-language pair.
Unlike static benchmarks, evaluation is condi-
tioned on the context retrieved at inference time,
addressing knowledge drift directly (Zheng et al.,
2023). Signals are aggregated through a consen-
sus layer calibrated against periodic human anno-
tations, stored in a versioned replayable format,
3GPT-4o-mini (via Azure OpenAI service) was used as the
base LLM in production at the time of writing.
3

Algorithm 1Genetic Prompt Optimization
Require: Business rules R, evaluation dataset D,
population size N, elite size K, maximum gen-
erationsG, scoring functionS(·)
Ensure:Optimized promptp∗
1:Initializeprompt population P0←
LLMGENERATE(R, N)
2:Initialize evaluation cacheC ← ∅
3:forg= 1toGdo
4:for eachpromptp∈P g−1do
5:ifp /∈ Cthen
6:Run LLM inference onDusingp
7:Compute fitnessf(p)← S(p,D)
8:Store(p, f(p))in cacheC
9:else
10:Retrievef(p)fromC
11:end if
12:end for
13: Select top- Kelite prompts Eg⊂P g−1by
fitness
14:Initialize new populationP g←E g
15:while|P g|< Ndo
16:Sample parents(p i, pj)fromE g
17:p′←SEMANTICBLEND(p i, pj)1
18:p′′←MUTATE(p′)2
19:Addp′′toPg
20:end while
21:end for
22:returnp∗←arg max p∈PGf(p)
and consumed by upstream optimization (auto-
prompt and RAG iteration), closing the loop be-
tween agent behavior, measurement, and controlled
self-evolution.
3 Engineering Architecture and
Deployment
Continuous evolution of prompts, retrieval, and
evaluation expands the failure surface relative to
static deployments. The production architecture
(Figure 2) is structured around four principles.
Reliability as a prerequisite for self-evolution.
A lightweight orchestration and fallback layer me-
diates user traffic with deterministic routing, error
handling, and graceful degradation. This isolates
end users from rapid iteration in the agent layer and
enables fallback to human-assisted support when
downstream AI components degrade or fail.Runtime as a controlled mutation boundary.
Rather than embedding behavior in code, prompts
are assembled dynamically, context-construction
policies are swappable, and tool usage is declara-
tively configured. Prompt evolution, retrieval tun-
ing, and workflow updates therefore ship through
configuration and artifact updates rather than code
deploys, allowing behavioral improvements to
propagate without redeploying the runtime itself.
Memory as versioned artifacts, not latent state.
The runtime carries no cross-session latent state be-
yond short-horizon, task-scoped context required
for conversational continuity. Longer-lived knowl-
edge is promoted only through explicitly versioned
and evaluated artifacts, enabling auditability, re-
producibility, controlled rollback, and principled
promotion of improvements while preventing un-
controlled memory drift.
Telemetry as an evaluation substrate.Execu-
tion telemetry serves both monitoring and auto-
mated evaluation. Structured event streams feed
nearline quality checks and offline evaluators mea-
suring groundedness, relevance, completeness, and
policy compliance, forming the control plane that
drives prompt evolution, retrieval tuning, and gated
deployment without direct model retraining.
Optimization cadence.The outer optimization
loop typically runs weekly and can also be trig-
gered by content refreshes or detected regressions.
One end-to-end evolution cycle takes hours to days
depending on dataset size and fitness-evaluation
budget. Candidate prompts and retrieval configu-
rations are promoted only after offline regression
checks and staged rollout; the execution plane can
roll back to the prior versioned artifact if guardrails
fail.
Deployment learnings.In production, three pat-
terns proved load-bearing.Centralized execu-
tionthrough configuration, modular prompts, and
declarative tool selection—rather than per-LoB
code forks, or separately maintained code paths
for individual lines of business—was critical to
avoiding behavioral divergence and ensuring im-
provements generalized across workflows.Lay-
ered fail-safessuch as targeted query redirection
allowed traffic to route to a known-safe content
pool when a polluted vector index required hours-
long reindexing, cutting recovery from hours to
minutes and limiting manual intervention. Finally,
4

User Chat Interfaces
Orchestration &
Fallback
Agent Runtime
Prompt assembly
Context / RAG
Tool invocationVersioned External Capabilities
LLM gateway
Tool APIs
Knowledge store
Moderation
Telemetry & Event Stream
Traces
Events
ETLModular Evaluation Agents
Groundedness
Relevance
Completeness
TranslationVersioned Artifacts Registry
Scores
Datasets
Regressions
Optimization & Safe Rollout
Prompt evolution
Retrieval tuning
Gated deploy / rollbackevaluation driven evolution
scores + regressionscandidate artifactsgated deploy / rollbackExecution Plane
Control Plane (Self-Evolution)
Figure 2: Production architecture for a self-evolving support agent. Optimization and Safe Rollout consumes
evaluator scores and regression signals together with candidate versioned artifacts, then emits gated deployment or
rollback decisions to the execution plane.
staged rolloutof new execution patterns in green-
field workflows—new, low-traffic workflows used
as controlled proving grounds—before extension to
mature traffic let evolution mechanisms and evalua-
tion signals mature before broader deployment. To-
gether these practices sustain continuous improve-
ment under rapidly changing models, policies, and
knowledge.
4 Simulation and Ablation Studies
We evaluate the agent through controlled simula-
tions that replay anonymized historical support in-
teractions and synthetic edge cases across multi-
lingual, product-diverse, and evolving knowledge
settings. Each simulation acts as a system-level
ablation, comparing configurations with different
agent capabilities enabled under identical evalua-
tion criteria.
4.1 RAG Simulation
We compare four response-generation configura-
tions using an LLM-based evaluator (GPT-4.1 via
Azure OpenAI service): Vanilla RAG, RAG-only
agents (ReAct-style and OpenAI tool invocation4),
and the full Support AI Agent. The production
generator is GPT-4o-mini while the offline judge is
GPT-4.1, so they are different models. We measure
overall quality, hallucination rate, and complete-
ness with respect to retrieved support articles; a
response is hallucinated if it contains any factual
4OpenAI tool agent is accessed via Azure OpenAI serviceclaim not attributable to the retrieved content. Ta-
ble 3 calibrates the judge against human labels.
As shown in Table 1, vanilla and agentic RAG
configurations hallucinate at 4.8–6.2%, while the
full Support AI Agent reaches <0.1% with the
highest overall score (2.78) and completeness
(87.8%). The gain comes from explicit agentic
constraints that prioritize synthesis from retrieved
authoritative content and suppress reliance on para-
metric knowledge when grounded sources exist.
4.2 Auto-Prompt Simulation
On the routing intent detection dataset, we iter-
atively optimize LLM-generated initial prompts
across generations using the genetic algorithm
pipeline (Algorithm 1). Table 2 shows that prompt
accuracy improves consistently only when both
crossover and mutation are enabled—increasing av-
erage accuracy from 62.6% to 68.0% after two gen-
erations, with the best prompt reaching 73.3%. Re-
moving either operator slows improvement and sat-
urates early, indicating that promptevolution, not
random sampling, is what drives sustained gains.
4.3 Evaluator-Signal Ablation
Beyond the system-level simulations above, we
ablate individual evaluator signals to isolate their
contribution, replaying each configuration on an
anonymized validation set and comparing against
a human-labeled reference. As shown in Table 3,
response groundedness is the dominant driver of
evaluator reliability—removing it causes the largest
5

Table 1: RAG Simulation Results
Setting1Overall Score2Hallucination3Completeness4
Vanilla RAG 2.59 5.3% 78.7%
RAG Only (ReAct Agent) 2.56 4.8% 79.0%
RAG Only (OpenAI Tool Agent) 2.48 6.2% 75.0%
Support AI Agent2.78 <0.1% 87.8%
1The simulation uses an anonymized, human-validated set of 100 support interactions with multiple
inference runs per interaction.
2Overall score is on a 0–3 scale to measure the overall quality of the response.
3A response is labeled as hallucinated if it contains any factual claim that cannot be attributed to the
retrieved support articles provided to the model at inference time.
4A response is labeled as complete if it fully addresses all aspects of the user inquiry.
Table 2: Automatic Prompt Optimization Simulation
Generation Crossover + Mutation No Mutation No Crossover
0 (Baseline) 62.6 / 66.7 62.6 / 66.7 62.6 / 66.7
1 66.7 / 73.3 63.3 / 66.7 62.6 / 66.7
2 68.0 / 73.3 64.0 / 66.7 63.3 / 66.7
The simulation uses a curated, human-validated intent-detection dataset (Search Knowl-
edge Base vs Direct to Consultant; N= 30 ). Metrics are average prompt accuracy /
best prompt accuracy per generation.
Table 3: Ablation Study of RAG Auto-Evaluation
Dimensions
Evaluation SettingAlignment1
w/ Human Labels2
Full RAG Evaluation (Baseline)87%
A1: No Groundedness 76% (-11)
A2: No Content Relevance 82% (-5)
A3: No Completeness 86% (-1)
A4: Overall Score Only 67% (-20)
1Alignment is measured against human-labeled
ground truth on a randomly sampled set of 100
premium-tier subscription support chats, with multi-
ple runs per interaction.
2Three trained reviewers, blind to system condition,
followed standardized guidelines. Raw agreement
was 92%; disagreements were resolved by majority
vote.
drop in alignment with human labels, while remov-
ing completeness barely changes it—and collaps-
ing all signals into a single overall score performs
worst, obscuring critical failure modes. These find-
ings justify retaining groundedness, relevance, and
completeness as first-class, independently reported
dimensions (Qiao et al., 2025; Park et al., 2025; Es
et al., 2023).
4.4 Multilingual Translation Evaluation
This evaluator-design ablation tests the modular
multi-agent evaluator that produces the fitness sig-Table 4: Translation Quality Evaluation
Result1
Evaluation Paradigm Accuracy (%)
Model-based eval (COMET) 49.7
Pure LLM single-agent eval 76.5
Modular multi-agent eval84.8
1Evaluation is performed on an anonymized sup-
port chat dataset (N=300).
nals driving Auto-Prompt and RAG iteration. We
compare it on the En–Zh slice with COMET (Rei
et al., 2020) and a single-agent LLM judge. As
summarized in Table 4, accuracy rises from 49.7%
(COMET) to 76.5% (single-agent LLM judge) to
84.8% (modular multi-agent evaluator): decompos-
ing translation quality into specialized dimensions
captures the domain-specific, context-sensitive de-
mands of support interactions that metric-based
scoring misses.
5 Online Experiment
We ran a two-week A/B test on continuous live
member and customer support traffic. Users were
randomized once with fixed 50/50 assignment and
remained in one arm for the experiment, prevent-
ing repeated-user contamination across conditions.
The control was the original production agent—a
handcrafted prompt, fixed retrieval pipeline, and pe-
6

riodic human QA. The treatment was the integrated
self-evolved workflow, which bundles (i) evolved
Auto-Prompt, (ii) agent-invoked RAG, and (iii)
closed-loop evaluator-driven iteration with gated
rollout. The three metrics use separate randomized
subpopulations.
The experiment was planned for four weeks
with sequential monitoring and concluded after
two weeks, when the effects were stable, power
exceeded 99.9%, and significance remained un-
der alpha-spending corrections. We compare each
proportion with a two-sided two-proportion z-test
(equivalently, a χ2test of independence) and ap-
ply Holm correction across the three primary out-
comes. The user is the inference unit; average
conversations per user are approximately 1.26 for
QA and 1.53 for cancellation. The corresponding
cluster adjustment is modest (design effect ≤1.5 ),
and user-clustered GEE/CR2 standard errors pre-
serve significance for all outcomes; for example,
the cancellation statistic changes from z= 10.0
toz= 8.1 . Table 5 reports the unclustered point
estimates and confidence intervals.
QA self-serve.This is the share of product-
question or technical conversations resolved with-
out human escalation. It increased from 33.7% to
42.7%, an absolute lift of 9.0 percentage points
(95% CI [8.4, 9.6];z= 27.6).
Cancellation self-serve.This is the share of
cancellation-intent conversations completed end-
to-end without handoff. It increased from 61.9%
to 66.6%, an absolute lift of 4.8 percentage points
(95% CI [3.8, 5.7];z= 10.0).
Routing accuracy.This is the share routed to the
correct human queue against the labeled target. It
increased from 38.2% to 68.8%, an absolute lift
of 30.6 percentage points (95% CI [23.6, 37.6];
z= 8.2).
All three outcomes have p≪10−4and remain
significant after Holm correction. We also mon-
itored escalation rate, thumbs-up/down feedback,
latency, customer-satisfaction score, and modera-
tion incidents; none regressed under treatment. The
online study reports the end-to-end system effect
of the integrated self-evolved workflow, while Ta-
bles 1–3 isolate the components offline.
Beyond aggregate gains, these results show that
the integrated self-evolved workflow improved the
measured outcomes over the two-week deploy-
ment window, consistent with calls for holistic,production-oriented evaluation (Liang et al., 2023;
Es et al., 2023; Qiao et al., 2025).
Limitations
Evaluator dependence.The closed loop relies
on LLM-as-judge evaluators (GPT-4.1) for fitness
signals. Our ablations measure alignment with
human labels (Table 3); larger samples and cross-
checks with open evaluators would further charac-
terize variance and model-specific judge sensitiv-
ity (Zheng et al., 2023; Dubois et al., 2024).
Cost and latency of evolution.Genetic prompt
search incurs additional inference cost per gener-
ation, and end-to-end evolution cycles take hours
to days depending on dataset size and fitness bud-
get. Future work could study tighter adaptation
windows for rapidly changing knowledge.
Single-tenant evaluation.The online study cov-
ers two weeks on one deployment surface at a sin-
gle enterprise. Longer observation windows and
deployments in other domains would test durabil-
ity and transfer to different tool inventories and
operating environments.
Online component attribution.The online treat-
ment bundles Auto-Prompt, agent-invoked RAG,
and evaluator-driven iteration as the integrated self-
evolved workflow. Factorial deployments could
quantify the marginal contribution and interaction
of each component.
Retrieval ceiling.Response quality is bounded
by retrieval coverage. When the underlying content
lake lacks an authoritative document, the agent’s
grounding constraints correctly suppress hallucina-
tion but cannot synthesize a correct answer; further
gains require improvements to retrieval recall and
authoritative-content coverage that are orthogonal
to the self-evolution loop.
Limited multilingual stress test.The controlled
translation study focuses on English ↔Chinese
(Section 4.4). Systematic evaluation of low-
resource languages and code-mixed input remains
an important extension.
Closed-source dependencies.The production
system depends on closed-source foundation mod-
els (GPT-4o-mini, GPT-4.1) and a proprietary
search backend. Reproduction with open-source
equivalents (e.g., Llama 3, BGE retrievers) is fea-
sible in principle but has not been verified end-to-
end.
7

Table 5: Online Experiment Results
MetricControl
original production
agentTreatment
integrated self-evolved
workflowAbsolute Lift
(95% CI)z
QA self-serve133.7% 42.7% +9.0 pp [8.4, 9.6] 27.6
Cancellation self-serve261.9% 66.6% +4.8 pp [3.8, 5.7] 10.0
Routing accuracy338.2% 68.8% +30.6 pp [23.6, 37.6] 8.2
1QA: 42,982 / 45,669 conversations and 35,867 / 34,334 users (control / treatment).
2Cancellation: 20,468 / 20,456 conversations and 13,442 / 13,308 users (control / treatment).
3Routing: 356 / 356 decisions (control / treatment) on a fixed labeled evaluation set.
Two-sided two-proportionz-tests; allp≪10−4and significant after Holm correction.
Acknowledgments
We thank Artem Grigoryan, Tony Huynh, Zhentao
Lin, Chris Korbel, Umang Lahoti and Ajay Vish-
wanathan for their support and collaboration on the
agent application and A/B test.
Use of AI assistance.In preparing this
manuscript, the authors used Anthropic’s Claude
(via Claude Code) for prose polishing, LaTeX
formatting and table layout, restructuring of
section ordering, and drafting assistance on
the Limitations section. All technical claims,
experimental design, results, figures, and citations
were authored, reviewed, and verified by the
human authors, who take full responsibility for the
content of the paper.
References
Akari Asai and 1 others. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InICLR.
Alec Berntson. 2023. Azure ai search: Outperforming
vector search with hybrid retrieval and reranking.
Yann Dubois and 1 others. 2024. Length-controlled
alpacaeval: A simple way to debias automatic evalu-
ators.arXiv:2404.04475.
Shahul Es and 1 others. 2023. Ragas: Auto-
mated evaluation of retrieval augmented generation.
arXiv:2309.15217.
Chrisantha Fernando and 1 others. 2023. Prompt-
breeder: Self-referential self-improvement via
prompt evolution. ArXiv:2309.16797.
Luyu Gao and 1 others. 2023a. Precise zero-shot dense
retrieval without relevance labels. InACL.
Yunfan Gao and 1 others. 2023b. Retrieval-augmented
generation for large language models: A survey.
arXiv:2312.10997.Qingyan Guo and 1 others. 2024. Connecting large
language models with evolutionary algorithms yields
powerful prompt optimizers. InICLR.
Kelvin Guu and 1 others. 2020. Realm: Retrieval-
augmented language model pre-training.ICML.
John H. Holland. 1992.Adaptation in Natural and
Artificial Systems. MIT Press.
Vladimir Karpukhin and 1 others. 2020. Dense passage
retrieval for open-domain question answering. In
EMNLP.
Omar Khattab and 1 others. 2023. Dspy: Compiling
declarative language model calls into self-improving
pipelines. ArXiv:2310.03714.
Seungone Kim and 1 others. 2024. Prometheus 2: An
open source language model specialized in evaluating
other language models. InEMNLP.
Patrick Lewis and 1 others. 2020a. Mlqa: Evaluat-
ing cross-lingual extractive question answering. In
Proceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics.
Patrick Lewis and 1 others. 2020b. Retrieval-augmented
generation for knowledge-intensive nlp tasks. In
NeurIPS.
Percy Liang and 1 others. 2023. Holistic evaluation of
language models.Transactions on Machine Learning
Research (TMLR). ArXiv:2211.09110.
Jimmy Lin and 1 others. 2021. Pretrained transformers
for text ranking: Bert and beyond.Synthesis Lectures
on Human Language Technologies.
Xi Victoria Lin and 1 others. 2022. Few-shot learning
with multilingual generative language models. In
Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 9019–9052.
Yang Liu and 1 others. 2023. G-eval: Nlg evaluation
using gpt-4 with better human alignment. InEMNLP.
Shayne Longpre and 1 others. 2021. Mkqa: A linguis-
tically diverse benchmark for multilingual open do-
main question answering.Transactions of the Asso-
ciation for Computational Linguistics, 9:1389–1406.
8

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage
re-ranking with bert.arXiv:1901.04085.
Oded Ovadia and 1 others. 2024. Fine-tuning or re-
trieval? comparing knowledge injection in llms.
arXiv:2312.05934.
Charles Packer and 1 others. 2023. Memgpt: Towards
llms as operating systems.arXiv:2310.08560.
Chanhee Park and 1 others. 2025. Mirage: A
metric-intensive benchmark for retrieval-augmented
generation evaluation.Findings of NAACL.
ArXiv:2504.17137.
Ofir Press and 1 others. 2023. Measuring and narrowing
the compositionality gap in language models. In
Findings of EMNLP. ArXiv:2210.03350.
Shuofei Qiao and 1 others. 2025. Benchmarking agentic
workflow generation. InICLR. ArXiv:2410.07869.
Ricardo Rei and 1 others. 2020. Comet: A neu-
ral framework for mt evaluation.arXiv preprint
arXiv:2009.09025.
Timo Schick and 1 others. 2023. Toolformer: Language
models can teach themselves to use tools.NeurIPS.
Noah Shinn and 1 others. 2023. Reflexion: Lan-
guage agents with verbal reinforcement learning.
arXiv:2303.11366.
Heydar Soudani and 1 others. 2024. Fine tuning vs. re-
trieval augmented generation for less popular knowl-
edge.arXiv:2403.01432.
Darrell Whitley. 1994. A genetic algorithm tutorial.
Statistics and Computing.
Qingyun Wu and 1 others. 2023. Autogen: Enabling
next-gen llm applications via multi-agent conversa-
tion.arXiv:2308.08155.
Shi-Qi Yan and 1 others. 2024. Corrective retrieval
augmented generation.arXiv:2401.15884.
Chengrun Yang and 1 others. 2024. Large language
models as optimizers. InICLR.
Shunyu Yao and 1 others. 2023. React: Synergizing
reasoning and acting in language models. InICLR.
Lianmin Zheng and 1 others. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena.
arXiv:2306.05685.
Yongchao Zhou and 1 others. 2023. Large language
models are human-level prompt engineers. InICLR.
9