# To Isolate or to Score? Model-Adaptive Assessment for Cost-Efficient Multi-Agent RAG

**Authors**: Jungseob Lee, Chanjun Park, Heuiseok Lim

**Published**: 2026-06-23 21:34:23

**PDF URL**: [https://arxiv.org/pdf/2606.25191v1](https://arxiv.org/pdf/2606.25191v1)

## Abstract
Multi-agent document assessment for retrieval-augmented generation is computationally expensive, driving practitioners toward smaller, deployable models whose assessment mechanisms remain poorly understood. We conduct a controlled study of training-free interventions on 7B-9B instruction-tuned models across diverse QA benchmarks, revealing a sharp dichotomy in how models benefit from assessment. For weaker baselines, the dominant mechanism is per-document isolation. Astoundingly, assessment-free isolation matches full multi-agent assessment, demonstrating that resolving multi-document context confusion, rather than scoring quality, drives outsized gains of up to 50 percentage points. Conversely, for strong baselines where scoring quality matters, we introduce Reasoning-Score Coupling, a label-free perturbation probe that classifies scoring behavior. Integrating these findings, we propose MADARA, a model-adaptive routing architecture. Crucially, MADARA's diagnostic thresholds derived from a single pilot model generalize zero-shot to four unseen model families, providing a robust, lightweight pipeline to eliminate computational overhead.

## Full Text


<!-- PDF content starts -->

To Isolate or to Score? Model-Adaptive Assessment
for Cost-Efficient Multi-Agent RAG
Jungseob Lee1Chanjun Park2*Heuiseok Lim1*
1Korea University2Soongsil University
{omanma1928, limhseok}@korea.ac.kr
chanjun.park@ssu.ac.kr
Abstract
Multi-agent document assessment for
retrieval-augmented generation is compu-
tationally expensive, driving practitioners
toward smaller, deployable models whose
assessment mechanisms remain poorly un-
derstood. We conduct a controlled study
of training-free interventions on 7B–9B
instruction-tuned models across diverse QA
benchmarks, revealing a sharp dichotomy in
how models benefit from assessment. For
weaker baselines, the dominant mechanism
is per-document isolation. Astoundingly,
assessment-free isolation matches full
multi-agent assessment, demonstrating that
resolving multi-document context confusion,
rather than scoring quality, drives outsized
gains of up to 50 percentage points. Con-
versely, for strong baselines where scoring
quality matters, we introduce Reasoning-
Score Coupling, a label-free perturbation
probe that classifies scoring behavior. Inte-
grating these findings, we propose MADARA,
a model-adaptive routing architecture. Cru-
cially, MADARA’s diagnostic thresholds
derived from a single pilot model generalize
zero-shot to four unseen model families,
providing a robust, lightweight pipeline to
eliminate computational overhead.
1 Introduction
Multi-agent document assessment is increasingly
used to improve retrieval-augmented generation
(RAG) by deploying specialized agents to evalu-
ate, filter, or debate retrieved documents (Lewis
et al., 2020; Chang et al., 2025; Wang et al.,
2025c; Hu et al., 2025). Despite its popular-
ity, this paradigm multiplies inference calls by
O(T rounds×N agents×N docs)compared to stan-
dard RAG, especially when iterative consensus
or multi-perspective scoring is required. In pro-
duction environments, applying this combinato-
rial multiplier to massive (30B+) language models
results in prohibitive latency and financial costs.
*Corresponding authors.To circumvent this overhead, practitioners are in-
creasingly turning to smaller, more cost-effective
models (e.g., 7B–9B parameters) (Wang et al.,
2025a; Lu et al., 2024; Cai et al., 2026; Prieto and
Abad, 2025).
However, this pragmatic shift creates a criti-
cal mechanistic misalignment: do these smaller,
deployable models actually possess the sophisti-
cated reasoning capabilities required to conduct
meaningful multi-agent assessment? If improve-
ments primarily stem from structural side-effects
(e.g., bypassing long-context confusion) rather
than from the agents’ substantive reasoning, the
community may be paying massive compute costs
for theoretically redundant processing (Liu et al.,
2024).
Indeed, our initial probing of these deploy-
able models reveals a highly inconsistent land-
scape. We observe that the exact same assessment
pipeline can boost exact-match accuracy by more
than ten percentage points for one model while ac-
tively degrading it for another (Luo et al., 2025;
Gao et al., 2026; Du et al., 2025). This stark con-
trast raises a fundamental mechanistic question:
what drives the gains when they occur, and why
do they fail to transfer across models?
To bridge this gap, we address this directly
through controlled ablations of a standard three-
agent RAG pipeline (Chang et al., 2025). Us-
ing only training-free interventions (prompting,
aggregation, and generation strategies), we an-
alyze the behavior of instruction-tuned models
across diverse QA benchmarks. Building on
these mechanistic insights, we propose Model-
Adaptive Document Assessment Routing Archi-
tecture (MADARA), a dynamic pipeline that auto-
matically routes model–task pairs to their optimal,
most cost-effective assessment strategy.
Our central claim is thatassessment value is
gated by an intrinsic capability divide that de-
pends on the interaction of model capacity with
task structure. We operationalise this claim as a
Diagnose→Treatpipeline: RSC together with the
No-Filter baseline diagnoses the model–task pair,
1arXiv:2606.25191v1  [cs.AI]  23 Jun 2026

and the appropriate treatment among PDE, SDA,
CoT, and ATF follows directly from that diagno-
sis. The two findings below are the diagnosis and
treatment halves of this single claim, not indepen-
dent contributions.
1. Diagnosis: a sharp isolation–scoring asym-
metry governs which assessment regime ap-
plies.We identify the asymmetry empirically.
For weaker models,per-document isolationdrives
outsized gains, boosting performance by 25 to 36
percentage points on adversarial conflicts and by
up to 50 percentage points on standard QA. As-
toundingly, even random, assessment-free isola-
tion matches full multi-agent variants. This proves
that resolving multi-document context confusion,
rather than scoring quality, is the actual bottle-
neck, rendering heavy multi-agent compute redun-
dant and reducing inference calls by roughly4×.
Conversely, strong-baseline models show no ben-
efit from isolation. Therefore, Per-Document Ex-
traction (PDE) can retain peak performance while
entirely eliminating assessment overhead in the
weak-baseline regime.
2. Treatment: RSC-driven MADARA rout-
ing transfers zero-shot.For strong models,
scoring quality remains critical. We introduce
Reasoning-Score Coupling(RSC), a perturbation-
based probe (Lanham et al., 2023; Paul et al.,
2024) that classifies model–task pairs asquality-
orderedorstochasticusing only 100 unlabeled
queries; RSC is the diagnostic arm. MADARA in-
tegrates RSC with the No-Filter baseline to route
each model–task pair to its optimal treatment;
MADARA is the treatment arm. Crucially, rout-
ing thresholds derived from a single pilot model
transfer zero-shot to four unseen model families.
This confirms the capability divide is intrinsic
rather than an overfitted artifact, providing a ro-
bust, lightweight pipeline to eliminate computa-
tional waste. Furthermore, our mechanistic find-
ings regarding the necessity of isolation persist
even when upgrading from sparse to state-of-the-
art dense retrieval and generative reranking.
2 Background and Related Work
Multi-Agent Debate and Assessment.Multi-
agent debate for improving LLM factuality (Du
et al., 2024) has expanded into broader agen-
tic RAG architectures (Singh et al., 2025).
Subsequent works address sycophantic conver-
gence (Liang et al., 2024; Jain et al., 2025; Pitreet al., 2025; Zhu et al., 2026), majority vot-
ing (Smit et al., 2023), consensus-free alterna-
tives (Cui et al., 2025), and mental-set diversifi-
cation (Liu et al., 2025b). In RAG, MADAM-
RAG (Wang et al., 2025c) debates answers us-
ing 70B+ agents, while DRAG (Hu et al., 2025),
MA-RAG (Nguyen et al., 2025), and MAIN-
RAG (Chang et al., 2025) explore reranking
and filtering. Astute-RAG (Wang et al., 2025b)
consolidates knowledge internally within a sin-
gle model; architecturally, our setting differs by
studying howmultiple assessment agentsinter-
act with a downstream generator, though our per-
document extraction (PDE) finding could com-
plement Astute-RAG when internal consolidation
is insufficient. Crucially, while existing litera-
ture largely treats multi-agent assessment as a
black-box performance enhancer, our controlled
PDE-Random ablation (Table 1) specifically iso-
lates this mechanism. This provides amechanistic
proofthat the entire gain for weak models comes
strictly from isolation, rendering assessment com-
pute irrelevant, a diagnostic claim fundamentally
addressing the gap in mechanistic understanding.
Adaptive and Self-Correcting RAG.Adaptive
systems route queries based on per-query signals:
learned reflection tokens (Asai et al., 2024), cor-
rective search (Yan et al., 2024), complexity clas-
sifiers (Jeong et al., 2024), or preference data (Ong
et al., 2024). Concurrent works further explore
cooperative RL optimization (Chen et al., 2025),
knowledge-graph conflict resolution (Liu et al.,
2025a), search conflict detection (Cattan et al.,
2025), information-gain reranking (Wang et al.,
2025d), and conflict taxonomies (Xu et al., 2024).
In contrast, our RSC-based routing operates at
themodel–domain level(a one-time probe applied
uniformly to all queries), requiring no training and
only 100 queries.
Score Calibration and CoT Faithfulness.
LLM score calibration is widely studied in the
judge setting (Jung et al., 2024; Jain et al., 2025;
Li et al., 2025b; Pitre et al., 2025; Li et al., 2025a).
More directly relevant is the CoT faithfulness
literature, showing LLMs often produce unfaith-
ful explanations (Turpin et al., 2023; Lanham
et al., 2023). Subsequent work quantifies this via
causal mediation (Paul et al., 2024), examines
reasoning–answer correlations (Jiang et al.,
2025b), unlearning (Tutek et al., 2025), and fun-
damental limits (Lyu et al., 2023; Tanneru et al.,
2

2024). Concurrently, MATCHA (Jiang et al.,
2025a) probes whether CoT answers decouple
from reasoning under perturbation. Our RSC
diagnostic adapts this to target multi-agentscor-
ing. By testing whether numerical scores degrade
monotonically as reasoning quality declines, RSC
provides a fine-grained diagnostic to measure
reasoning–score coupling strength, allowing
us to prescribe targeted remedies like CoT de-
polarization for uncalibrated models. (Extended
related work is provided in Appendix A).
3 Reasoning-Score Coupling
The diagnostic arm of Diagnose→Treat.RSC
is thediagnostichalf of the pipeline introduced
in §1: it identifies, without gold labels, whether
a model’s scoring behaviour on a given task
is reasoning-driven or stochastic, and the corre-
sponding treatment (§4) follows directly from the
diagnosis. We introduceReasoning-Score Cou-
pling(RSC), a task-specific diagnostic detect-
ing whether a model’s document scores degrade
monotonically under systematic reasoning pertur-
bation. If scores track reasoning quality, degrading
reasoning monotonically decreases score reliabil-
ity; a lack of this pattern indicates CoT interven-
tions are unlikely to help. RSC providescorre-
lationalevidence of a reasoning–scoring associa-
tion; establishing causality requires intervention-
based methods (Paul et al., 2024). The probe mea-
suresscoring sensitivity to reasoning perturba-
tion, distinct from standard CoT faithfulness (Ap-
pendix F).
3.1 Perturbation Protocol
RSC compares a model’s document scores un-
der CoT reasoning against those obtained after
three levels of increasing perturbation:(1) Shuf-
fled(reasoning steps randomly reordered; dis-
rupts logical flow but preserves semantics),(2)
Contradicted(steps semantically negated; dis-
rupts both coherence and directional cues), and(3)
Random(reasoning from a completely different
query-document pair; entirely irrelevant).
For each levelk∈ {1,2,3}, we compute
the Spearman rank correlationˆρ kbetween nor-
mal (s normal∈[0,5]m) and perturbed scores (s Pk)
across all calibration documents:
ˆρk=Spearman 
snormal,sPk
(1)
The severity ordering is motivateda priori, repre-senting monotonically decreasing preserved infor-
mation (empirical confirmation in Appendix F).
3.2 Formal Definition of RSC
Terminology.We use “perturbation-level correla-
tionsˆρ k” for Equation 1 and “trend coefficientρ∗”
for the final monotonicity statistic.
Definition 1(Reasoning-Score Coupling, RSC).
LetMbe a language model andDa calibration set
ofnexamples each withd iretrieved documents.
TheRSC trend coefficientis:
ρ∗=Spearman 
[1,2,3],[ˆρ 1,ˆρ2,ˆρ3]
(2)
ModelMon domainDis classified as:
quality-ordered ifρ∗=−1.0
stochastic (non-monotonic) otherwise
Sinceρ∗takes only five discrete values, this re-
duces to a deterministic check for perfect mono-
tonic degradation (ˆρ 1>ˆρ 2>ˆρ 3). The metric’s
reliability derives from the highly significant per-
levelˆρ kvalues (p <0.001) and bootstrap resam-
pling (Appendix F.3, Table 13).
Beyond the binary classification,ˆρ 1measures
thestrengthof baseline coupling, defining three
states:stochastic(ρ∗>−1.0),weakly coupled
(ρ∗=−1.0,ˆρ 1<0.5), andstrongly coupled
(ρ∗=−1.0,ˆρ 1≥0.5). The protocol requires no
gold labels: the quality oracle is the model’s own
generated reasoning. Beyond this binary trend, we
additionally report a continuous, magnitude-aware
companion signal¯ρ=1
3(ˆρ1+ ˆρ 2+ ˆρ 3), used as an
explicit robustness check (Appendix F.4). Further-
more, RSC captures process-level coupling rather
than distributional spread, making it a distinct and
more robust routing signal than standard score en-
tropy (see Appendix I).
4 Candidate Treatment Strategies
The treatment arm of Diagnose→Treat.
MADARA is thetreatmenthalf of the pipeline.
The four candidate strategies below address
distinct failure modes diagnosed by RSC (§3) and
the No-Filter (NF) baseline; the routing decisions
are operational consequences of the capability
divide identified by these two diagnostic signals,
not independent design choices. We evaluate the
four strategies on real and synthetic failure modes,
and route a model–task pair to its corresponding
treatment based on two diagnostics: RSC (scoring
3

Calibration Data
(n=100)Target LLM
One-Time
RSC
Probe  & EvalScoring
TrendPhase 1: Model Profiling
(Offline)Phase 2: MADARA Router
(Decision Logic)Phase 3: Treatment Execution (Online Processing)
Path A
   (Weak Multi -Doc Capacity)
Path B
   (Strong Multi -Doc Capacity)PDE (Per-Document Extraction) SDA  (Score Distribution Alignment)
CoT  (De-polarization)
Thinking  /  Reasoning
ATF  (Adaptive Threshold Filtering)
Generator
   (LLM)Stochastic Scoring Prompt Score
Weakly Coupled
Quality -Ordered
Strongly Coupled
Quality -Ordered
Coupling
Strength
(ොp1)(𝑝∗)
Percentile Rank(𝐸𝑀𝑁𝐹)Context
Capacity
𝐸𝑀𝑁𝐹<30%
𝐸𝑀𝑁𝐹≥30%
Rank 3 (60%)Rank 2 (80%)Rank 1 (90%)
Rank 4 (40%)
Figure 1: The MADARA Model-Adaptive Routing Architecture. (Left) A one-time RSC probe evaluates the target
LLM’s context capacity (EM NF) and scoring behavior (ρ∗,ˆρ1). (Middle) The router identifies a capability phase-
transition: weak-baseline models are strictly routed to bypass multi-document evaluation. (Right) PDE (Isolation)
structurally separates documents to cure context confusion, while scoring-only treatments (SDA, CoT, ATF) refine
document assessment for strong models. Full pseudocode is provided in Appendix E.
behaviour) and NF accuracy (context-handling
capacity). All treatments operate within a three-
agent assessment framework (agent design and
prompts in Appendix N).
4.1 CoT De-Polarization
For quality-ordered models (ρ∗=−1.0), the base-
line failure mode ispolarization: without reason-
ing guidance, agents assign extreme scores (0 or
5) to≈80%of documents. CoT de-polarization
requires agents to generate explicit reasoning be-
fore scoring, mitigating this direct-to-extreme pat-
tern. For example, extreme scores for Mistral-7B
on CONFLICTS drop from80.8%to3.5%, yield-
ing a+4.7pp EM gain over NF (Table 2).
4.2 Score Distribution Alignment
For stochastic models (ρ∗>−1.0), reasoning
quality does not drive scores, rendering CoT de-
polarization ineffective.Score Distribution Align-
ment(SDA) bypasses reasoning entirely. It con-
verts each agent’s raw scores to percentile ranks,
then aggregates them via weighted averaging to
produce a calibrated ranking (Algorithm 1, Ap-
pendix B).
4.3 Adaptive Threshold Filtering
For strongly coupled models (ρ∗=−1.0,ˆρ 1≥
0.5), baselines already produce effective rankings.
ATF leverages these scores tofilterrather than
rerank:
τ=µ(s)−κ·σ(s)(3)(withκ= 0.5). It retains only documents with
si≥τ(minimum 2, maximumk), requiring no
additional LLM calls.
4.4 Per-Document Answer Extraction
For models struggling with multi-document con-
text (low NF accuracy),Per-Document Extraction
(PDE) structurally decomposes generation: (1) 3-
agent scoring evaluates documents; (2) the model
generates an answer from each top-kdocument in-
dividually; (3) candidates are grouped by normal-
ized string match, and the group with the highest
cumulative score is selected. Component ablations
(§ 6) reveal that this isolation, rather than scoring
quality, drives PDE’s outsized gains.
4.5 MADARA Routing Protocol
The MADARA pipeline (Figure 1) operates in two
phases. Phase 1 runs a one-time RSC probe to
classify the model-task pair. Phase 2 processes
each query using the selected optimal treatment:
SDA for stochastic models; PDE for quality-
ordered models with weak NF baselines; and CoT
de-polarization (ˆρ 1<0.5) or ATF (ˆρ 1≥0.5) for
quality-ordered models with strong NF baselines.
The NF threshold is estimated from the RSC cali-
bration set (pseudocode in Appendix E).
5 Experimental Setup
5.1 Models
To investigate the target regime of cost-effective,
deployable LLMs, we evaluate five open-weight,
4

instruction-tuned 7B–9B models:Llama-3.1-
8B-Instruct(Grattafiori et al., 2024),Mistral-
7B-Instruct-v0.3(Jiang et al., 2023),Qwen3-
8B(Yang et al., 2025),Qwen2.5-7B-Instruct
(Qwen et al., 2025), andGemma-2-9B-IT(Team
et al., 2024). All models are served in
bfloat16via vLLM v0.8.5 (Kwon et al.,
2023) withmax_seq_len=4096 on NVIDIA
A100/RTX8000 GPUs at temperature0.6.
5.2 Benchmarks
To manage the massiveO(T×N×D)multi-
agent inference overhead, we evaluate on sampled
subsets (∼1K queries) of three diverse datasets:
(1) CONFLICTS (CFL)(Xie et al., 2023): an
adversarial QA benchmark with inter-document
contradictions via entity substitution, filtered for
strict exact-match fidelity.(2) FEVER (FVR)
(Thorne et al., 2018): a binary fact-verification
task with BM25 retrieval, representing a high-
baseline regime.(3) TriviaQA (TQA)(Joshi
et al., 2017): a standard factoid QA benchmark
with BM25 retrieval, serving as our held-out set to
replicate the isolation and RSC findings (§ 6).
5.3 Methods Compared
We compare six strategies across two categories.
Scoring treatments:(1) NF: all documents to
generator (standard RAG);(2) 3-Agent Baseline:
weighted score aggregation (0.4, 0.3, 0.3);(3)
CoT De-Polarization: explicit reasoning before
scoring (§ 4.1);(4) SDA: percentile-rank nor-
malization (SDA outperforms standard RRF by
adapting to heterogeneous distributions; see Ap-
pendix C).Generation strategies:(5) ATF: adap-
tive threshold (µ−0.5σ) filtering (§ 4.3);(6) PDE:
per-document extraction with score-weighted ma-
jority voting (§ 4.4). The MADARA router dy-
namically selects among these treatments.
We exclude MADAM-RAG (Wang et al.,
2025c) (which debates generated answers using
70B+ models) and FiD (Izacard and Grave, 2021)
(which modifies cross-attention) to strictly isolate
training-free, document-level assessment behav-
iors in the 7B–9B regime. A detailed discussion
on baseline scope is provided in Appendix L.
5.4 Evaluation Metrics
We report Exact Match (EM; strictly matched after
whitespace normalization and lowercasing) and
Token F1 (harmonic mean of token-level precisionand recall). For the binary FEVER task, EM and
F1 are equivalent.
5.5 Implementation Details
Agents share the same base model per experiment,
reranking top-k=5from 10 retrieved documents.
The RSC probe usesn=100label-free calibra-
tion queries. SDA uses uniform agent weights
w= (1/3,1/3,1/3). Crucially, routing thresh-
olds (ρ∗=−1.0,ˆρ 1= 0.5,τ NF= 30%) were
derived strictly from a single pilot model (Mistral-
7B) to prevent overfitting and applied zero-shot to
all others (sensitivity analyses in Appendices D.1
and D.2).
6 Results
6.1 Isolation vs. Scoring
Isolation dominates for weak models.PDE
provides outsized gains exclusively for weak-
baseline models (Table 2), yielding+36.3pp for
Llama and+25.4pp for Mistral on CONFLICTS
(p <0.001). In contrast, scoring-only treatments
provide at most+5.5pp across all models. Cru-
cially, this isolation finding replicates on a held-
out factoid benchmark (TriviaQA, Table 1), where
Llama gains+49.8pp. This provides strong con-
firmation that the outsized PDE gain is a fun-
damental mechanism, not an artifact of CON-
FLICTS’ adversarial document structure.
Assessment quality is redundant for weak base-
lines.A component ablation (Table 1) confirms
that structural isolation, rather than assessment
quality, drives these gains. PDE-Random (ran-
dom document selection with uniform voting)
completely bypasses multi-agent assessment yet
matches the full PDE pipeline for Llama on both
CONFLICTS (50.6%vs.50.2%) and TriviaQA
(79.6%vs.79.6%). For Mistral, assessment-
guided selection adds+19pp beyond random iso-
lation on CONFLICTS. By bypassing multi-agent
evaluation, PDE-Random reduces inference calls
by roughly4×. This proves that resolving con-
text confusion renders heavy assessment compute
largely redundant for weak baselines. Token F1
scores further verify that these improvements are
not mere formatting artifacts (see Appendix M).
The capability divide.This isolation and scor-
ing asymmetry reflects a sharp capability phase-
transition based on intrinsic context-handling ca-
pacity. Weak-baseline models (EM NF<30%
5

Base PDE Component Ablation
Model Task NF Rand. Unif. Full
(1×cost) (1×cost) (≈4×) (≈4×)
Llama-3.1 Conflicts 13.950.6 51.1 50.2
TriviaQA 29.879.6 80.1 79.6
Mistral-v0.3 Conflicts 18.1 24.5 38.843.5
TriviaQA 67.6 71.573.7 73.7
Table 1:PDE component ablation (EM%) re-
veals that multi-agent assessment is redundant for
weak baselines.For Llama, completely assessment-
free isolation (Rand.) yields identical outsized gains
as the computationally heavyFullpipeline, proving
that resolving context confusion drives the improve-
ment. (Cost multipliersindicate relative inference calls
vs. standard RAG. CFL=Conflicts, TQA=TriviaQA;
NF=No Filter; Unif.=assessment-guided + uniform
vote; Full=score-weighted vote.)
on a given task) gain+25to+36pp from forced
isolation. Conversely, strong-baseline models
(EM NF≥60%) show no benefit from PDE (e.g.,
Qwen3 CFL drops from60.8%to58.6%); instead,
they selectively respond to scoring-only interven-
tions like SDA. Crucially, extended scaling ex-
periments (up to 32B parameters; Appendix H)
prove that this divide is strictly governed by intrin-
sic baseline capacity, not mere parameter count.
Even at larger scales, strong context-handling ren-
ders forced isolation unnecessary but harmless,
confirming that the critical need for isolation in
weak models is a fundamental architectural prop-
erty rather than a small-model artifact.
6.2 RSC Diagnostic Results
Scoring behavior is a model-task interaction.
Table 4 demonstrates that scoring behavior is not
a fixed model property but a dynamic model-task
interaction. On the held-out TriviaQA benchmark,
RSC classifications perfectly replicate those ob-
served on CONFLICTS: Mistral remains Quality-
Ordered (ρ∗=−1.0), while Qwen3, Qwen2.5,
and Gemma-2 remain Stochastic (ρ∗=−0.5).
Llama’s TQA scoring is entirely degenerate (98%
of scores at the absolute floor, mean0.17/5.0),
making perturbation uninformative; this extreme
baseline failure independently necessitates per-
document isolation. Across all benchmarks,
strong-baseline models lose quality-ordered scor-
ing on complex adversarial and factoid-QA tasks
but maintain it on the simpler binary FEVER task.
This pattern replicates across three model families
and three distinct task types, proving the interac-tion reflects an intrinsic capability threshold rather
than benchmark-specific artifacts.
6.3 Superiority of RSC over Score Entropy
A natural baseline heuristic for evaluating multi-
agent assessment isscore entropy, based on the
premise that high score entropy correlates with re-
trieval noise and scoring uncertainty. Since our
RSC diagnostic also uses score perturbation, a
critical question arises: does RSC provide routing
decisions that are meaningfully different and more
accurate than a standard entropy-based heuristic?
To address this, we evaluated routing accuracy
across 10 model–benchmark pairs using a simpli-
fied binary setup for fair comparison (routing to
CoT vs. SDA based on RSC classification versus
high/low entropy). As summarized in Table 3,
RSC and entropy disagree on the optimal treat-
ment in 4 out of 10 cases, proving they capture
fundamentally distinct properties.
Crucially, RSC significantly outperforms score
entropy. It successfully matches the oracle (opti-
mal) treatment 3 times more frequently than en-
tropy (3/10 vs. 1/10) and more consistently sur-
passes both the NF and 3-Agent baselines.
Beyond this binary performance gap, RSC pro-
vides a vital mechanistic advantage: the per-
level coupling value (ˆρ 1). Entropy strictly mea-
sures variance and cannot quantify thecoupling
strengthbetween a model’s reasoning and its
scores. Therefore, entropy alone cannot motivate
the precise distinction between applying CoT (for
weakly coupled models) versus Adaptive Thresh-
old Filtering (ATF, for strongly coupled models).
RSC’s ability to measure this coupling is what
enables the full four-treatment MADARA archi-
tecture, reducing the mean oracle gap on CON-
FLICTS to≤0.4pp.
6.4 Robustness Across Retrieval Quality
A critical question is whether the isolation and
scoring asymmetry, along with the resulting need
for MADARA routing, are merely artifacts of
sparse retrieval (BM25) noise. To test this,
we evaluate our pipelines using a dense re-
triever (Izacard et al., 2021) and a robust gener-
ative reranker (Qwen3-0.6B).
As shown in Table 5 (with extended five-model
results detailed in Appendix H.1), upgrading to
high-precision dense retrieval natively reduces
multi-document context confusion. This causes
the marginal benefit of PDE for weak models to
6

Diagnostics Component Methods EM (%) MADARA (Ours)
Model Task RSCˆρ 1 NF 3-Agent CoT SDA Strategy EM∆ NF
Llama-3.1-8B Conflicts Quality-Ordered 0.76 13.9 24.1 16.5 19.4 PDE 50.2∗∗∗+36.3
FEVER Quality-Ordered 0.65 88.7 87.8 88.4 88.2 ATF 90.9∗+2.2
Mistral-7B-v0.3 Conflicts Quality-Ordered 0.35 18.1 18.6 22.8 20.3 PDE 43.5∗∗∗+25.4
FEVER Quality-Ordered 0.28 90.7 90.992.690.7 CoT 92.6+1.9
Qwen3-8B Conflicts Stochastic 0.39 60.8 62.4 62.463.0 SDA 63.0+2.2
FEVER Quality-Ordered 0.63 89.6 89.8 89.5 90.0 ATF 90.9+1.3
Qwen2.5-7B Conflicts Stochastic 0.46 59.9 65.0 62.465.4 SDA 65.4+5.5
FEVER Quality-Ordered 0.62 87.1 88.9 87.590.5∗∗ATF†88.2+1.1
Gemma-2-9B Conflicts Stochastic 0.79 60.864.663.3 63.3 SDA 63.3+2.5
FEVER Quality-Ordered 0.58 92.2 92.492.592.5 ATF 93.6+1.4
Average Conflicts42.7 46.9 45.5 46.3 –57.1+14.4
Average FEVER89.7 90.0 90.1 90.4 –91.2+1.5
Table 2:MADARA dynamically routes models to optimal assessment strategies, maximizing Exact Match
(EM%).The Strategy is determined zero-shot via RSC and No-Filter (NF) baselines. Significance vs. NF (McNe-
mar’s test with Holm-Bonferroni):∗p<0.05,∗∗p<0.01,∗∗∗p<0.001.†Routed to ATF due to crossing the baseline
coupling threshold (ˆρ 1= 0.62≥0.5), narrowly missing the optimal SDA.
Routing Accuracy Metric Score Entropy RSC (Ours)
Beats NF Baseline 7/108/10
Beats 3-Agent Baseline 4/105/10
Matches Oracle (Optimal)1/103/10
Table 3:RSC vs. Entropy-based Routing.Compari-
son across 10 model–benchmark pairs using a simpli-
fied binary routing setup. RSC captures mechanistic
coupling rather than mere variance, tripling the success
rate of identifying the optimal assessment strategy. Full
breakdown is in Appendix I.
shrink (e.g., Llama drops from+49.8pp under
BM25 to+20.8pp under Contriever). However,
crucially, the weak model still exhibits a massive
>+20pp gain. Furthermore, even when a power-
ful generative reranker is applied to order the doc-
uments, the weak model suffers from severe con-
text confusion and yields a staggering+50.4pp
gain from PDE. This confirms our mechanistic
hypothesis: multi-document confusion is a fun-
damental model deficit, and structural isolation
(PDE) remains a mandatory architectural interven-
tion regardless of how clean the retrieved context
is.
Conversely, for strong models, high-precision
retrieval shifts PDE from being a slightly harm-
ful filter bypass under BM25 to a modest ensem-
bling mechanism, yielding up to+3.6pp under
Contriever, though it offers no meaningful bene-
fit under generative reranking.
It is worth noting a limitation of rigid thresh-
olding in these upgraded contexts. Under dense
retrieval, Llama’s No-Filter baseline on TriviaQACorrelation (ˆρ k) Trend RSC
TaskShuffled Contra. Random (ρ∗)Class
Llama-3.1-8B
Conflicts .76 .43 .08−1.0Quality-Ordered
FEVER .65 .49 .10−1.0Quality-Ordered
TriviaQADegenerate scoring (mean=0.17/5.0)‡
Mistral-7B-v0.3
Conflicts .35 .22 .18−1.0Quality-Ordered
FEVER .28 .09 .01−1.0Quality-Ordered
TriviaQA .47 .05 .02−1.0Quality-Ordered
Qwen3-8B
Conflicts .39 .04 .14−0.5Stochastic
FEVER .63 .51 .14−1.0Quality-Ordered
TriviaQA .64 .45 .45−0.5Stochastic
Qwen2.5-7B
Conflicts .46−.26.08−0.5Stochastic
FEVER .62 .21 .06−1.0Quality-Ordered
TriviaQA .55−.28.15−0.5Stochastic
Gemma-2-9B
Conflicts .79 .40 .40−0.5Stochastic†
FEVER .58 .41 .18−1.0Quality-Ordered
TriviaQA .83 .19 .48−0.5Stochastic
Table 4:RSC diagnostic results reveal that scoring
behavior is a model-task interaction.Spearman cor-
relations (ˆρ k) are shown under three increasing pertur-
bation levels. Perfect monotonic degradation yields a
trend coefficient ofρ∗=−1.0, classifying the model-
task pair as Quality-Ordered; otherwise, it is Stochas-
tic.†Aggregate vs. per-query disagreement.‡Scores
degenerate at the absolute floor, making perturbation
uninformative.
marginally crosses the strictτ NF= 30%thresh-
old (34.6%). A strict application of the MADARA
router would bypass PDE and miss the+20.8pp
isolation gain. This highlights that while a zero-
shot threshold derived from sparse retrieval acts
as a highly effective general heuristic, dynamic
thresholding adaptive to retriever quality repre-
sents an important direction for future refinement.
Nevertheless, systematically forcing PDE in these
dense and reranked scenarios empirically proves
7

PDE Gain (∆EM vs. NF Baseline)
ModelBM25
(Sparse)Contriever
(Dense)Qwen3-0.6B
(Reranker)
Weak-Baseline Capacity
Llama-3.1-8B+49.8 +20.8 +50.4
Strong-Baseline Capacity
Qwen2.5-7B−0.5 +3.6−1.0
Gemma-2-9B−0.8 +2.4 +0.8
Table 5:Impact of Context Quality Upgrades
on TriviaQA.Upgrading retrieval quality (Dense/R-
eranker) reduces the isolation benefit for weak models,
yet PDE remains mandatory to cure severe context con-
fusion (e.g.,+50.4pp for Llama). Conversely, strong
models (Qwen, Gemma) possess intrinsic context ca-
pacity, rendering PDE unnecessary.
our core mechanistic claim: multi-document con-
fusion persists, and isolation remains the definitive
cure.
This dynamic model-environment interaction
definitively proves that a “one-size-fits-all” RAG
pipeline is suboptimal. The fact that PDE is a life-
saver for weak models but acts completely differ-
ently for strong ones serves as the ultimate justi-
fication for MADARA: RAG architectures must
dynamically route treatments based on intrinsic
model capacity and scoring behavior.
6.5 Multi-Hop Generalization (MuSiQue)
Result.On MuSiQue, the same diagnostic prin-
ciple predicts an inversion: scoring treatments be-
come optimal, while isolation alone fails. We eval-
uate Qwen2.5-7B on 100 MuSiQue (Trivedi et al.,
2022) queries with 20 mixed supporting/distractor
paragraphs per query.
Method EM (%) F1 (%)∆EM
NF (No-Filter) 14.0 21.3 —
3-Agent Baseline30.0 39.6+16.0
CoT De-Polarization 21.0 30.9+7.0
SDA (rank)30.038.7+16.0
PDE 24.0 34.4+10.0
PDE-Random 12.0 19.6−2.0
Table 6:MuSiQue results (Qwen2.5-7B-Instruct,
100 queries).Bootstrap CI for the+16.0pp 3-Agent
gain over NF: 95% CI[+5.0,+27.0]pp,p= 0.003
(10,000 resamples).
The RSC probe yields a strongly coupled
Quality-Ordered profile (ˆρ 1= 0.83,ˆρ 2= 0.54,
ˆρ3= 0.03,ρ∗=−1.0; per-levelp <0.001).
Consistent with this diagnosis, Table 6 shows that
3-Agent and SDA both improve EM by+16.0pp,while PDE-Random falls below NF (−2.0pp).
PDE remains useful (+10.0pp) because scoring
surfaces relevant single-hop pivots, but it lags the
best scoring treatments by 6pp because isolated
per-document voting cannot reconstruct missing
chain steps. A chain-coverage analysis in Ap-
pendix J verifies that this gap is largest when the
selected top-5 omits one or more supporting para-
graphs. Thus, the same Diagnose→Treat principle
extends to multi-hop: when task structure shifts
the bottleneck from context confusion to chain
decomposition, the optimal treatment shifts from
isolation to scoring.
7 Discussion and Conclusion
One claim, three demonstrations.Our central
claim is a single mechanistic statement:assess-
ment value is gated by an intrinsic capabil-
ity divide that depends on model capacity and
task structure. CONFLICTS, FEVER, and Triv-
iaQA show the weak/strong single-hop asymme-
try; zero-shot transfer to four unseen model fami-
lies shows that the diagnostic boundary is not over-
fit; and MuSiQue shows that the bottleneck shifts
from context confusion to chain decomposition in
multi-hop reasoning. RSC is the label-free probe
that makes this diagnosis possible, and MADARA
is the operational consequence of acting on it.
Practical implication.Multi-agent document
assessment is not a universally beneficial black
box. Weak single-hop baselines need struc-
tural isolation, often without costly assessment,
while stronger or multi-hop settings require
scoring treatments. Practitioners should there-
fore diagnose the model–task pair before pay-
ing for multi-agent scoring or forcing isolation.
This Diagnose→Treat view explains why PDE-
Random can match full PDE for weak single-hop
models, why RSC-based routing improves strong
baselines, and why MuSiQue reverses the pre-
ferred treatment.
Limitations
Multi-hop as a Predicted Boundary, Quan-
tified.The Per-Document Extraction (PDE)
mechanism cannot perform cross-document syn-
thesis: with documents processed in strict isola-
tion, PDE is structurally unable to recover chains
that require combining information across mul-
tiple documents. This is apredicted boundary
rather than a hidden risk, and §6.5 quantifies it on
8

MuSiQue. A second-model check with Mistral-
7B shows the same pattern (Appendix J): isolation
without scoring is harmful, while scoring treat-
ments are the dominant lever. Because a vast ma-
jority of production RAG queries are single-hop
factual retrievals, identifying the redundancy of
multi-agent scoring in that regime remains highly
impactful; the multi-hop setting cleanly demar-
cates where the single-hop cure ceases to apply.
Scope of Document-Intensive Benchmarks.
We deliberately bound the study to cost-effective,
deployable 7B–9B instruction-tuned models on
document-level retrieval-augmented QA, where
multi-agent assessment overhead is most punish-
ing. Document-intensive deep-search benchmarks
(BrowseComp, GAIA, xBench) and deep-research
benchmarks (DeepResearch Bench) lie outside
this scope: they require live browser tool-use,
multi-turn agent control, or long-form planning,
all of which would mix in confounds (tool-use
error, planner quality) that obscure the isolation–
vs.–scoring mechanism we isolate. Extending
the framework to these settings is a natural di-
rection for future work; the diagnostic princi-
ple (capability-divide-driven routing) should gen-
eralise, but the candidate treatments will need to
be re-derived for the new failure modes those set-
tings introduce.
Heuristic Thresholding vs. Dynamic Adap-
tation.A structural limitation of the current
MADARA routing implementation is its reliance
on a static baseline threshold (τ NF= 30%).
Two pieces of evidence circumscribe this limita-
tion. First, the static rule is empirically robust:
a sensitivity sweep overτ NF∈[15%,45%]flips
the routing decision on only1/12of the model–
task cells it routes (Appendix D.2). Second, the
multi-hop pilot (§6.5) reveals a deployment cost
the static rule alone cannot absorb: a strongly-
coupled Quality-Ordered model with NF≪τ NF
is routed to PDE, but the empirically optimal treat-
ment on multi-hop is scoring (+16pp via 3-Agent
/ SDA versus+10pp via PDE), leaving a 6pp
deployment gap. This indicates that the capa-
bility phase-transition is relative to the interac-
tion of model, retriever precision, and task struc-
ture, rather than an absolute constant. There-
fore, while the static heuristic successfully val-
idates the isolation–scoring asymmetry and re-
duces compute overhead in this study, deploy-
ing adaptive RAG across highly heterogeneous re-trieval pipelines and task structures will require
dynamic thresholding. Developing methods to
learn this boundary directly from a calibration
set’s distribution — and to detect outlier-low NF
that signals a task-structure change — remains an
important engineering direction for future work.
Ethics Statement
While our work significantly reduces the compu-
tational overhead of multi-agent RAG pipelines,
we acknowledge several potential risks associated
with the deployment of our proposed MADARA
architecture and Per-Document Extraction (PDE)
mechanism.
Misinformation and Bias Propagation.By
structurally isolating documents to bypass context
confusion, PDE may inadvertently reduce the op-
portunity for agents to organically cross-examine
and debate conflicting sources. If the underlying
retrieval corpus is poisoned or heavily biased, the
system risks surfacing and amplifying this misin-
formation without the friction of full multi-agent
scrutiny.
Over-reliance in High-Stakes Domains.The
ability to achieve highly competitive exact-match
performance using easily deployable 7B–9B mod-
els may encourage practitioners to deploy these
pipelines in high-stakes domains, such as medi-
cal or legal QA. Because these smaller models still
possess an intrinsic capability floor, over-reliance
on them could lead to critical factual errors that
users might mistakenly trust due to the seemingly
rigorous multi-agent assessment pipeline.
Dual-Use and Malicious Application.The pri-
mary contribution of our work is making sophis-
ticated document assessment highly cost-efficient.
Consequently, this lowers the barrier to entry for
deploying large-scale automated generation sys-
tems. Malicious actors could leverage these op-
timized pipelines to generate highly contextual-
ized deceptive content, spam, or disinformation at
a fraction of the traditional computational cost.
Use of AI AssistantsWe used AI assistants
solely for proofreading, formatting tables, and re-
fining the clarity of the English text. The core
research, experimental design, and data analysis
were conducted entirely by the human authors.
9

References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning
to retrieve, generate, and critique through self-
reflection. InInternational conference on learning
representations, volume 2024, pages 9112–9141.
Guanyu Cai, Ruiming Tian, Lang Yang, Yunzhe Jia,
Lingkun Li, and Jiliang Wang. 2026. Efficient infer-
ence for edge large language models: A survey.Ts-
inghua Science and Technology, 31(3):1365–1380.
Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig,
Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan
Szpektor, and Avi Caciularu. 2025. Dragged
into conflicts: Detecting and addressing conflicting
sources in search-augmented llms.arXiv preprint
arXiv:2506.08500.
Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh,
Menghai Pan, Chin-Chia Michael Yeh, Guanchu
Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Ma-
hashweta Das, and 1 others. 2025. Main-rag: Multi-
agent filtering retrieval-augmented generation. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 2607–2622.
Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu
Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin,
Yiming Yang, and Jiaxin Mao. 2025. Improv-
ing retrieval-augmented generation through multi-
agent reinforcement learning.arXiv preprint
arXiv:2501.15228.
Gordon V Cormack, Charles LA Clarke, and Stefan
Buettcher. 2009. Reciprocal rank fusion outper-
forms condorcet and individual rank learning meth-
ods. InProceedings of the 32nd international ACM
SIGIR conference on Research and development in
information retrieval, pages 758–759.
Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, and
Cong Zuo. 2025. Free-mad: Consensus-free multi-
agent debate.arXiv preprint arXiv:2509.11035.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B
Tenenbaum, and Igor Mordatch. 2024. Improving
factuality and reasoning in language models through
multiagent debate. InForty-first international con-
ference on machine learning.
Yufeng Du, Minyang Tian, Srikanth Ronanki,
Subendhu Rongali, Sravan Bodapati, Aram Gal-
styan, Azton Wells, Roy Schwartz, Eliu A Huerta,
and Hao Peng. 2025. Context length alone hurts
llm performance despite perfect retrieval.arXiv
preprint arXiv:2510.05381.
Yunfan Gao, Yun Xiong, Wenlong Wu, Bohan Li, Yijie
Zhong, and Haofen Wang. 2026. U-niah: Unified
rag and llm evaluation for long context needle-in-
a-haystack.ACM Transactions on Information Sys-
tems, 44(3):1–30.Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, and 1 others. 2024. The llama 3
herd of models.arXiv preprint arXiv:2407.21783.
Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Ja-
son Zhang, Xiaoyong Wei, and Li Qing. 2025. Re-
moval of hallucination on hallucination: Debate-
augmented rag. InProceedings of the 63rd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 15839–
15853.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open
domain question answering. InProceedings of the
16th conference of the european chapter of the asso-
ciation for computational linguistics: main volume,
pages 874–880.
Suryaansh Jain, Umair Z Ahmed, Shubham Sahai, and
Ben Leong. 2025. Beyond consensus: Mitigat-
ing the agreeableness bias in llm judge evaluations.
arXiv preprint arXiv:2510.11822.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024. Adaptive-rag:
Learning to adapt retrieval-augmented large lan-
guage models through question complexity. InPro-
ceedings of the 2024 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies (Vol-
ume 1: Long Papers), pages 7036–7050.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Re-
nard Lavaud, Marie-Anne Lachaux, Pierre Stock,
Teven Le Scao, Thibaut Lavril, Thomas Wang, Tim-
othée Lacroix, and William El Sayed. 2023. Mistral
7b.Preprint, arXiv:2310.06825.
Enyi Jiang, Changming Xu, Nischay Singh, and
Gagandeep Singh. 2025a. Robust answers, frag-
ile logic: Probing the decoupling hypothesis in llm
reasoning.Transactions on Machine Learning Re-
search.
Gangwei Jiang, Yahui Liu, Zhaoyi Li, Wei Bi, Fuzheng
Zhang, Linqi Song, Ying Wei, and Defu Lian.
2025b. What makes a good reasoning chain? un-
covering structural patterns in long chain-of-thought
reasoning. InProceedings of the 2025 Conference
on Empirical Methods in Natural Language Pro-
cessing, pages 6501–6525.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
10

supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611.
Jaehun Jung, Faeze Brahman, and Yejin Choi. 2024.
Trust or escalate: Llm judges with provable guar-
antees for human agreement.arXiv preprint
arXiv:2407.18370.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. InProceedings of the
29th symposium on operating systems principles,
pages 611–626.
Tamera Lanham, Anna Chen, Ansh Radhakrishnan,
Benoit Steiner, Carson Denison, Danny Hernandez,
Dustin Li, Esin Durmus, Evan Hubinger, Jackson
Kernion, and 1 others. 2023. Measuring faithful-
ness in chain-of-thought reasoning.arXiv preprint
arXiv:2307.13702.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in neural information processing systems,
33:9459–9474.
Dawei Li, Bohan Jiang, Liangjie Huang, Alimoham-
mad Beigi, Chengshuai Zhao, Zhen Tan, Amrita
Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tian-
hao Wu, and 1 others. 2025a. From generation to
judgment: Opportunities and challenges of llm-as-
a-judge. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Process-
ing, pages 2757–2791.
Feiyang Li, Peng Fang, Zhan Shi, Arijit Khan, Fang
Wang, Dan Feng, Weihao Wang, Xin Zhang, and
Yongjian Cui. 2025b. Cot-rag: Integrating chain of
thought and retrieval-augmented generation to en-
hance reasoning in large language models.arXiv
preprint arXiv:2504.13534, page 22.
Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang,
Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi,
and Zhaopeng Tu. 2024. Encouraging divergent
thinking in large language models through multi-
agent debate. InProceedings of the 2024 conference
on empirical methods in natural language process-
ing, pages 17889–17904.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language
models use long contexts.Transactions of the asso-
ciation for computational linguistics, 12:157–173.
Shuyi Liu, Yuming Shang, and Xi Zhang. 2025a.
Truthfulrag: Resolving factual-level conflicts in
retrieval-augmented generation with knowledge
graphs.arXiv preprint arXiv:2511.10375.Yexiang Liu, Jie Cao, Zekun Li, Ran He, and Tieniu
Tan. 2025b. Breaking mental set to improve reason-
ing through diverse multi-agent debate. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
Zhenyan Lu, Xiang Li, Dongqi Cai, Rongjie Yi, Fang-
ming Liu, Xiwen Zhang, Nicholas D Lane, and
Mengwei Xu. 2024. Small language models: Sur-
vey, measurements, and insights.arXiv preprint
arXiv:2409.15790.
Qi Luo, Xiaonan Li, Junqi Dai, Shuang Cheng, and
Xipeng Qiu. 2025. Zero-rag: Towards retrieval-
augmented generation with zero redundant knowl-
edge.arXiv preprint arXiv:2511.00505.
Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang,
Delip Rao, Eric Wong, Marianna Apidianaki, and
Chris Callison-Burch. 2023. Faithful chain-of-
thought reasoning. InProceedings of the 13th In-
ternational Joint Conference on Natural Language
Processing and the 3rd Conference of the Asia-
Pacific Chapter of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pages
305–329.
Thang Nguyen, Peter Chin, and Yu-Wing Tai. 2025.
Ma-rag: Multi-agent retrieval-augmented genera-
tion via collaborative chain-of-thought reasoning.
arXiv preprint arXiv:2505.20096.
Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin
Chiang, Tianhao Wu, Joseph E Gonzalez, M Waleed
Kadous, and Ion Stoica. 2024. Routellm: Learning
to route llms with preference data.arXiv preprint
arXiv:2406.18665.
Debjit Paul, Robert West, Antoine Bosselut, and Boi
Faltings. 2024. Making reasoning matter: Measur-
ing and improving faithfulness of chain-of-thought
reasoning. InFindings of the Association for Com-
putational Linguistics: EMNLP 2024, pages 15012–
15032.
Priya Pitre, Naren Ramakrishnan, and Xuan Wang.
2025. Consensagent: Towards efficient and ef-
fective consensus in multi-agent llm interactions
through sycophancy mitigation. InFindings of the
Association for Computational Linguistics: ACL
2025, pages 22112–22133.
Pablo Prieto and Pablo Abad. 2025. Edge deployment
of small language models, a comprehensive compar-
ison of cpu, gpu and npu backends.arXiv preprint
arXiv:2511.22334.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
11

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane
Suhr. 2023. Quantifying language models’ sensitiv-
ity to spurious features in prompt design or: How i
learned to start worrying about prompt formatting.
arXiv preprint arXiv:2310.11324.
Aditi Singh, Abul Ehtesham, Saket Kumar, and
Tala Talaei Khoei. 2025. Agentic retrieval-
augmented generation: A survey on agentic rag.
arXiv preprint arXiv:2501.09136.
Andries Smit, Paul Duckworth, Nathan Grinsztajn,
Thomas D Barrett, and Arnu Pretorius. 2023.
Should we be going mad? a look at multi-
agent debate strategies for llms.arXiv preprint
arXiv:2311.17371.
Sree Harsha Tanneru, Dan Ley, Chirag Agarwal, and
Himabindu Lakkaraju. 2024. On the hardness
of faithful chain-of-thought reasoning in large lan-
guage models.arXiv preprint arXiv:2406.10625.
Gemma Team, Morgane Riviere, Shreya Pathak,
Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhu-
patiraju, Léonard Hussenot, Thomas Mesnard,
Bobak Shahriari, Alexandre Ramé, and 1 others.
2024. Gemma 2: Improving open language models
at a practical size.arXiv preprint arXiv:2408.00118.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
Fever: a large-scale dataset for fact extraction and
verification. InProceedings of the 2018 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages
809–819.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022. MuSiQue: Mul-
tihop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Miles Turpin, Julian Michael, Ethan Perez, and Samuel
Bowman. 2023. Language models don’t always say
what they think: Unfaithful explanations in chain-
of-thought prompting.Advances in Neural Informa-
tion Processing Systems, 36:74952–74965.
Martin Tutek, Fateme Hashemi Chaleshtori, Ana
Marasovi ´c, and Yonatan Belinkov. 2025. Measuring
chain of thought faithfulness by unlearning reason-
ing steps. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Process-
ing, pages 9946–9971.
Fali Wang, Zhiwei Zhang, Xianren Zhang, Zongyu Wu,
Tzuhao Mo, Qiuhao Lu, Wanjing Wang, Rui Li,
Junjie Xu, Xianfeng Tang, and 1 others. 2025a. A
comprehensive survey of small language models in
the era of large language models: Techniques, en-
hancements, applications, collaboration with llms,
and trustworthiness.ACM Transactions on Intelli-
gent Systems and Technology, 16(6):1–87.Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen,
and Sercan O Arik. 2025b. Astute rag: Overcom-
ing imperfect retrieval augmentation and knowledge
conflicts for large language models. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 30553–30571.
Han Wang, Archiki Prasad, Elias Stengel-Eskin, and
Mohit Bansal. 2025c. Retrieval-augmented gen-
eration with conflicting evidence.arXiv preprint
arXiv:2504.13079.
Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma,
Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei,
Yuqing Ding, and Han Li. 2025d. Infogain-rag:
Boosting retrieval-augmented generation through
document information gain-based reranking and fil-
tering. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Process-
ing, pages 7201–7215.
Albert Webson and Ellie Pavlick. 2022. Do prompt-
based models really understand the meaning of their
prompts? InProceedings of the 2022 conference
of the north american chapter of the association for
computational linguistics: Human language tech-
nologies, pages 2300–2344.
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and
Yu Su. 2023. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in
knowledge conflicts. InThe Twelfth International
Conference on Learning Representations.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,
Hongru Wang, Yue Zhang, and Wei Xu. 2024.
Knowledge conflicts for llms: A survey. InProceed-
ings of the 2024 Conference on Empirical Methods
in Natural Language Processing, pages 8541–8565.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua
Ling. 2024. Corrective retrieval augmented gener-
ation.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Xiaochen Zhu, Caiqi Zhang, Yizhou Chi, Tom
Stafford, Nigel Collier, and Andreas Vlachos.
2026. Demystifying multi-agent debate: The
role of confidence and diversity.arXiv preprint
arXiv:2601.19921.
A Extended Related Work
Foundational multi-agent debate.Multi-agent
debate improves LLM factuality and reasoning
(Du et al., 2024; Liang et al., 2024), though major-
ity voting accounts for much of the benefit (Smit
et al., 2023). Sycophancy and conformity bias re-
main key challenges (Cui et al., 2025; Jain et al.,
2025; Pitre et al., 2025).
12

Adaptive RAG systems.Prior adaptive RAG
systems use reflection tokens (Asai et al., 2024),
corrective web search (Yan et al., 2024), com-
plexity classifiers (Jeong et al., 2024), model rout-
ing (Ong et al., 2024), information gain reranking
(Wang et al., 2025d), cooperative RL (Chen et al.,
2025), and knowledge graphs (Liu et al., 2025a;
Xu et al., 2024). RSC-based routing operates at
the model–domain level (one-time probe) rather
than per-query.
Score calibration and CoT faithfulness.Prior
work addresses judge verdict escalation (Jung
et al., 2024), CoT-RAG integration (Li et al.,
2025b), conformity mitigation (Pitre et al., 2025),
and CoT faithfulness (Turpin et al., 2023; Lan-
ham et al., 2023; Tutek et al., 2025; Lyu et al.,
2023; Tanneru et al., 2024; Paul et al., 2024).
RSC adapts perturbation-based probing to multi-
agentscoring, detecting monotonic score degrada-
tion rather than reasoning–answer coupling. For a
broader perspective, see Li et al. (2025a).
B Method Details
Extended RSC characterization.Whileρ∗
classifies thetypeof scoring behavior,ˆρ 1provides
critical complementary information: it measures
thestrengthof baseline reasoning–score coupling.
The thresholdˆρ 1= 0.5separates weakly from
strongly coupled models, reflecting the natural gap
between Mistral’s weak coupling (ˆρ 1≤0.35) and
the next-nearest model (0.46); this value is robust
across[0.4,0.6](Appendix D.1).
Relationship to prompt sensitivity.RSC’s per-
turbations are a form of prompt variation, a well-
documented LLM phenomenon (Sclar et al., 2023;
Webson and Pavlick, 2022). A general prompt
sensitivity metric would detect that scoreschange,
but not whether the changetracks reasoning qual-
ity, the directional, ordinal prediction thatρ∗cap-
tures.
SDA algorithm.SDA’s novelty lies not in per-
centile normalization itself (a standard tech-
nique (Cormack et al., 2009)) but in identifying
whenit should be applied: specifically, for models
where RSC diagnoses reasoning–score decoupling
and CoT intervention is futile.
ATF filtering behavior.We analyze ATF’s doc-
ument filtering behavior using Qwen2.5-7B on
CONFLICTS (the model–benchmark pair whereAlgorithm 1Score Distribution Alignment (SDA)
Require:Score matrixS∈[0,5]m×A(m≥2documents,
Aagents)
Require:Agent weightsw∈∆A−1(default: uniform)
Ensure:Calibrated ranking ˆr∈[0,5]m
// Step 1: Percentile conversion per agent
1:forj= 1, . . . , Ado
2:s j←S :,j ▷Scores for agentj
3:r j←rank(s j)/(m−1)▷Percentile ranks∈[0,1]
4:end for
// Step 2: Weighted aggregation
5:ˆr←PA
j=1wj·rj·5▷Map to[0,5]
// Step 3: Select top-kdocuments
6:returnargsort(− ˆr)[:k]
ATF produces the strongest gain: +5.5pp over
NF). ATF uses an adaptive thresholdµ−0.5σon
the aggregated document scores, removing docu-
ments that score below this threshold.
On CONFLICTS (Qwen2.5-7B, 458 queries),
ATF retains all top-5 documents for 83.6% of
queries (mean thresholdµ=2.38, mean 4.8/10
docs selected). The +5.5pp gain comes primarily
from the 16.4% of queries where filtering is active,
removing 1–3 documents scoring−1.2 points be-
lowµ−0.5σ.
C SDA vs. RRF Ablation
To validate our choice of percentile-rank aggre-
gation over Reciprocal Rank Fusion (RRF) (Cor-
mack et al., 2009), we compare SDA with RRF
on Qwen3-8B (one of three stochastic models on
CONFLICTS). RRF aggregates agent rankings via
RRF(d) =PA
j=11/(k+rank j(d))withk= 60
(standard default).
Method EM (%) Token F1
Existing Methods
3-Agent (baseline) 62.40.695
RRF (k= 60) 61.6 0.678
Proposed Alignment
SDA (rank) 63.0 0.682
SDA+CoT (rank)63.9 0.695
Table 7:SDA vs. RRF aggregation on Qwen3-8B
×CONFLICTS.SDA uses percentile-rank normaliza-
tion; RRF usesk= 60. Both use the exact same un-
derlying 3-agent scores.
Table 7 shows that SDA outperforms RRF
by +1.4% EM and +0.004 F1, while SDA+CoT
achieves the best result (+2.3% EM over RRF).
RRF’s fixedkparameter does not adapt to the
13

heterogeneous score distributions across agents,
whereas SDA’s percentile-rank normalization is
scale-invariant by construction.
D RSC Threshold Sensitivity
All thresholds in[−1.0,−0.7]yield identical rout-
ing decisions due to the bimodal distribution
ofρ∗(−1.00for 7/10 pairs vs.−0.50for 3
pairs). This robustness suggests that the quality-
ordered/stochastic distinction reflects a robust cat-
egorical distinction rather than a continuous spec-
trum, at least for the 7–9B models tested.
D.1 Two-Dimensional Routing:ˆρ 1Threshold
Sensitivity
The two-dimensional routing criterion (defined in
§3.2 and applied in §4.5) usesˆρ 1= 0.5to dis-
tinguish weakly coupled (CoT-responsive) from
strongly coupled (baseline-preferred) quality-
ordered models. Table 8 reports threshold sensi-
tivity acrossτvalues.
Threshold(τ)Changed†Oracle Matches(↑)Mean Gap(↓)
0.31/10 6/10 0.50pp
Optimal Robust Plateau (τ∈[0.4,0.6])
0.42/10 8/10 0.07pp
0.52/10 8/10 0.07pp
0.63/109/10 0.06pp
Degradation Phase
0.76/10 8/10 0.18pp
0.87/10 7/10 0.94pp
Table 8:ˆρ 1threshold sensitivity on CFL + FVR
(10 pairs).The routing is highly robust across
τ∈[0.4,0.6], achieving minimal performance gap (≤
0.07pp). The chosenτ= 0.5sits safely in the center of
this plateau. Atτ= 0.8, performance degrades signifi-
cantly as strongly coupled models (e.g., Llama-CFL)
are misrouted to CoT.†Number of Quality-Ordered
models routed to CoT (vs. default baseline).
D.2τ NFThreshold Sensitivity and Per-Model
Calibration
The staticτ NF= 30%is derived from a Mistral-7B
pilot and applied zero-shot. Two analyses circum-
scribe its robustness using only the measurements
already in the paper.
Sensitivity sweep.We sweepτ NF across
{15,20,25,30,35,40,45}% over the12model–
task cells the router routes. The router’s choice
changes on only1/12cells: Mistral-7B / CON-
FLICTS atτ= 15%selects CoT instead of
PDE, and on those data CoT scores22.8%whilePDE scores43.5%— so the original choice is
also empirically optimal. The staticτ= 30%
therefore sits on a robust plateau within the
paper’s single-hop sparse-retrieval regime.
Calibration grounding ofτ NF.The robustness
of the static threshold established by the sweep
above is grounded in the per-model calibration NF
distribution itself (Table 9). On the routing grid
(Table 2) these scores are sharply separated: every
cell the policy routes to PDE has NF≤18.1%,
while every cell it routes to a scoring treatment
has NF≥59.9%, leaving a gap of more than
40points with no intervening routing observation.
The static30%lies inside this gap, so the pub-
lished routing is invariant to the exact threshold
across the entire interval(18.1%,59.9%): the cal-
ibration distribution, through this wide separation
rather than any precise cut, is what fixes the rout-
ing. A per-model re-estimation of the threshold
is therefore not required in the single-hop sparse-
retrieval regime studied here, becoming relevant
only once the task structure itself shifts, as in the
multi-hop regime (§6.5).
Model Calibration NF (CFL/FVR/TQA)τ static
Llama-3.1-8B 13.9 / 88.7 / 29.8 30.0
Mistral-7B-v0.3 18.1 / 90.7 / 67.6 30.0
Qwen3-8B 60.8 / 89.6 / — 30.0
Qwen2.5-7B 59.9 / 87.1 / — 30.0
Gemma-2-9B 60.8 / 92.2 / — 30.0
Table 9:Per-model calibration NF vs. the static rout-
ing threshold.Each model’s No-Filter (NF) exact-
match on the routing tasks (CONFLICTS, FEVER) and
held-out TriviaQA, againstτ static= 30%. On the rout-
ing grid, every PDE-routed cell has NF≤18.1%and
every scoring-routed cell has NF≥59.9%— a>40pp
gap containingτ static, so any threshold in the gap repro-
duces the published routing. Dashes (—) mark strong-
baseline TriviaQA NF, which the routing rule does not
use.
Together, the sweep plateau (1/12cells), the
zero-shot transfer of the pilot-calibrated threshold
to four unseen families, and this calibration gap es-
tablish that the published routing is fixed by a ro-
bust capacity separation rather than by the precise
threshold value; the routing algorithm of Table 2
stands unchanged.
E MADARA Routing Algorithm
Algorithm 2 presents the complete pseudocode for
the MADARA routing protocol. The architecture
14

Algorithm 2The MADARA routing protocol.
Require:ModelM, calibration setC, test queriesQ
Ensure:Answer sequence ˆA
// Phase 1: One-time RSC probe & Calibration
1:ρ∗,ˆρ1←RSCTrendProbe(M,C)
2: EM NF←EvalNF(M,C)
3:ifEM NF< τ NFthen
4:mode←PDE
5:else ifρ∗>−1.0then
6:mode←SDA
7:else ifˆρ 1<0.5then
8:mode←CoT
9:else
10:mode←ATF
11:end if
// Phase 2: Per-query adaptive assessment
12:foreach queryq∈ Qdo
13:D←Retrieve(q)
14:S←ThreeAgentScore(M, q, D,mode)
15:ifmode=SDAthen
16:D′←SDA(S)
17:ˆa q←Generate(M, q, D′)
18:else ifmode=CoTthen
19:D′←RerankedTopK(S)
20:ˆa q←Generate(M, q, D′)
21:else ifmode=ATFthen
22:D′←ThresholdFilter(S, µ−0.5σ)
23:ˆa q←Generate(M, q, D′)
24:else▷PDE mode
25:D k←TopK(S)
26:{c i}k
i=1←PerDocExtract(M, q, D k)
27:ˆa q←ScoreWeightedV ote({c i},S)
28:end if
29:end for
30:return ˆA={ˆa q}q∈Q
operates in two distinct phases. Phase 1 performs
an offline model profiling using the calibration set
Cto evaluate both the scoring behavior via the
RSC diagnostic (ρ∗,ˆρ1) and the baseline context
capacity (EM NF). Phase 2 executes the dynami-
cally selected treatment strategy for each incom-
ing queryq∈ Q.
Crucially, regarding the calibration setC, while
the RSC probe itself (RSCTrendProbe) is strictly
label-free, the baseline evaluation (EvalNF) re-
quires standard QA answer pairs to compute Ex-
act Match. However, no expensive document-level
relevance annotations are required at any point in
the pipeline.
F RSC Probe Calibration Details
F.1 Perturbation Implementation
Level 1 (Shuffled).Given the model’s original
reasoning chainR= [r 1, r2, . . . , r n](split by sen-
tence boundaries), we generateR′=permute(R)
using a fixed random seed. The shuffled chain ispresented as-is to the scoring prompt.
Level 2 (Contradicted).We apply antonym
substitution to the reasoning chain: positive
indicators (“relevant”, “consistent”, “supports”,
“accurate”, “reliable”) are replaced with their
antonyms (“irrelevant”, “inconsistent”, “contra-
dicts”, “inaccurate”, “unreliable”), and vice versa.
Numerical quality indicators are inverted: “high
quality”→“low quality”, “score of 4”→“score
of 1”.
Level 3 (Random).The reasoning chain for
documentd iin queryq jis replaced with the rea-
soning chain produced for documentd kin query
qlwhere(j, i)̸= (l, k), selected uniformly at ran-
dom. This makes the reasoning entirely irrelevant
to the document being scored.
Discrete nature ofρ∗.With 3 perturba-
tion levels,ρ∗takes five discrete values:
{−1.0,−0.5,0,0.5,1.0}. We classify based
on whetherρ∗=−1.0(perfect monotonic
degradation), which requires strict ordering
ˆρ1>ˆρ 2>ˆρ 3. The wide gap between observed
classes (−1.0vs.−0.5) suggests a categorical
property.
F.2 Per-Query Correlation Distributions
The aggregate correlationsˆρ kreported in the
main text (Table 4) are computed by pooling all
document–score pairs across queries within each
perturbation level. To assess whether these ag-
gregate metrics obscure important per-query vari-
ance, we compute Spearmanρindependently
for each query (correlating normal vs. perturbed
scores across the 10 documents within that query).
We then analyze the distributional statistics across
the 100-query causal ablation probes for all five
models on the CONFLICTS benchmark.
Table 10 shows the mean±standard deviation
of per-queryρfor each model, and Figure 2 visu-
alizes the full distributions via violin plots. Our
per-query analysis reveals two distinct behavioral
patterns that strongly align with our RSC classifi-
cations:
Quality-Ordered Models.As shown in Fig-
ure 2 (green boxes), Llama-3.1-8B and Mistral-
7B-v0.3 exhibit consistent monotonic degradation
across the vast majority of individual queries.
Their per-query means (Table 10) strictly decrease
across the three perturbation levels. This con-
firms that theρ∗=−1.0trend is a robust, query-
15

agnostic property rather than an artifact of aggre-
gate pooling.
Stochastic Models.Conversely, stochastic
models exhibit non-monotonic or highly volatile
distributions (red boxes in Figure 2). For Qwen3
and Qwen2.5, the degradation trend breaks
entirely at the Contradicted level; Qwen3 shows
massive variance (ρ 2≈0.03±0.51), explaining
the lack of a monotonic trend, while Qwen2.5
actively inverts the score ordering with a negative
mean correlation (ρ=−0.20). Finally, although
Gemma-2 shows monotonic means at the per-
query level, its aggregate pooled correlation
(Table 4) remains non-monotonic, which drives
its Stochastic classification in the MADARA
pipeline.
Shuffled Contradicted Random1.0
0.5
0.00.51.0Per-Query Spearman 
+0.29±0.42 +0.18±0.45 +0.08±0.34Trend =-1.00, p<0.001
Mistral-7B
Shuffled Contradicted Random1.0
0.5
0.00.51.0Per-Query Spearman 
+0.72±0.24 +0.38±0.42 -0.01±0.36Trend =-1.00, p<0.001
Llama-3.1-8B
Shuffled Contradicted Random1.0
0.5
0.00.51.0Per-Query Spearman 
+0.30±0.37 +0.03±0.51 +0.16±0.36Trend =-0.50, p=0.667
Qwen3-8B
Figure 2:Per-query Spearmanρdistributions on
CONFLICTS.Violin widths show density, and hor-
izontal lines mark medians. Numbers below violins
indicate meanρvalues. Green and red boxes denote
significant (ρ∗=−1.0) and non-significant monotonic
trends, respectively.
Per-Query Correlation (ρ±std)
Model Shuffled Contradicted Random
Quality-Ordered Models (Monotonic Decrease)
Llama-3.1-8B+0.72±0.24 +0.38±0.42−0.01±0.36
Mistral-7B-v0.3+0.29±0.42 +0.18±0.45 +0.08±0.34
Stochastic Models (Non-Monotonic or High Variance)
Qwen3-8B+0.30±0.37 +0.03±0.51 +0.16±0.36
Qwen2.5-7B+0.54±0.33−0.20±0.49 +0.09±0.36
Gemma-2-9B†+0.73±0.26 +0.44±0.42 +0.30±0.33
Table 10:Per-query Spearmanρdistributions on
CONFLICTS.Mean±standard deviation ofρcom-
puted independently for each of the 100 queries.†For
Gemma-2, while per-query means strictly decrease,
its aggregate pooled correlation (Table 4) is non-
monotonic, resulting in a Stochastic classification.
Findings.(1) Per-queryρdistributions have
substantial variance (std. dev. = 0.24–0.51), but
this doesnotcontradict the aggregate trends:
Llama’s per-queryρdecreases from+0.72(shuf-
fled) to−0.01(random), confirming monotonic
degradation at both aggregate and per-query lev-
els. (2) Qwen3’s stochastic classification is drivenby themiddleperturbation level: contradictedρ
has high variance (std. = 0.51) and mean near
zero (+0.03), indicating that Qwen3’s sensitiv-
ity to reasoning degradation is query-dependent
rather than systematically ordered. (3) The trend
test operates onmeansof per-queryρ, not individ-
ual queries, so variance within levels does not af-
fect classification; only theorderingof the three
means matters. Llama’s means are strictly or-
dered (0.72>0.38>−0.01, yieldingρ∗=
−1.0), while Qwen3’s means violate monotonic-
ity (0.30>0.03but0.03<0.16, yieldingρ∗=
−0.5).
F.3 Split-Half Stability
To test whether RSC classifications depend on
which examples are used for probing, we split
each 100-query causal ablation dataset into the
first 50 and last 50 queries, recompute per-level
ˆρkon each half, and check whetherρ∗yields the
same classification.
Table 11 summarizes split-half and 30-query
subset stability for each model–benchmark pair.
Five of six model–benchmark pairs yield identical
classifications across splits, confirming that RSC
is robust to sample composition for all models ex-
cept the already-identified borderline case (Mis-
tral). 30-query stability analysis shows Llama and
Qwen3-FEVER classifications are perfectly sta-
ble, while Mistral exhibits moderate instability on
CONFLICTS (62% Quality-Ordered). This val-
idates that borderline models benefit from larger
probe samples (≥100 queries), while strongly
classified models (Llama, Qwen3) are stable even
atn=50.
Table 12 compares routing recommendations
from RSC and entropy against oracle outcomes.
F.4 Continuous¯ρas a Robustness Check
The binary trend coefficientρ∗(Definition 3.2)
captures themonotonicityof degradation across
perturbation levels but is insensitive to magnitude.
We report a continuous companion signal com-
puted from the same per-level correlations:
¯ρ(M,D) =1
3 
ˆρ1+ ˆρ 2+ ˆρ 3
.(4)
¯ρmeasures average coupling strength regardless
of monotonic ordering, so it is complementary to
ρ∗rather than redundant:ρ∗flags whether the
degradation is ordered, and¯ρflags how strong
the underlying coupling is. Where the two sig-
16

Split-Half (n= 50) 30-Query Subsets (%)
Model Task First 50 Last 50 Quality-Ordered (%) Stochastic (%)
Llama-3.1-8B Conflicts Quality-Ordered Quality-Ordered1000
FEVER Quality-Ordered Quality-Ordered1000
Mistral-7B-v0.3 Conflicts Quality-Ordered Quality-Ordered6238
FEVER Quality-Ordered Stochastic8119
Qwen3-8B Conflicts Stochastic Stochastic 1585
FEVER Quality-Ordered Quality-Ordered964
Table 11:RSC stability across split-half and 30-query subsets.Split-half compares the first vs. last 50 queries.
The 30-query columns show the percentage of 100 random 30-query subsets yielding each classification. Bold
numbers indicate the classification that matches the full 100-query ground truth.
Diagnostics Recommendation Exact Match (EM %)
Model RSC Class Entropy RSC Entropy RSC Entropy Oracle (Best)
CONFLICTS
Llama-3.1-8B Quality-Ordered 0.75 CoT SDA 16.5 19.4 3-Agent (24.1)
Mistral-7B-v0.3 Quality-Ordered 0.91 CoT SDA22.820.3 CoT (22.8)
Qwen3-8B Stochastic 0.84 SDA SDA 63.0 63.0 SDA+CoT (63.9)
Qwen2.5-7B Stochastic 0.82 SDA SDA65.4 65.4SDA (65.4)
Gemma-2-9B Stochastic 0.85 SDA SDA 63.3 63.3 3-Agent (64.6)
FEVER
Llama-3.1-8B Quality-Ordered 0.89 CoT SDA 88.4 88.2 SDA+CoT (88.5)
Mistral-7B-v0.3 Quality-Ordered 0.80 CoT SDA92.690.7 CoT (92.6)
Qwen3-8B Quality-Ordered 0.58 CoT CoT 89.5 89.5 SDA (90.0)
Qwen2.5-7B Quality-Ordered 0.44 CoT CoT 87.5 87.5 SDA (90.5)
Gemma-2-9B Quality-Ordered 0.61 CoT CoT 92.5 92.5 SDA+CoT (92.7)
Overall Performance vs. Baselines & Oracle
Beats NF baseline 8/10 7/10 —
Beats 3-Agent baseline 5/10 4/10 —
Matches oracle (Optimal routing) 3/101/10 —
Agreement (same recommendation)6/10 —
Table 12:RSC vs. entropy-based routing accuracy.For each model–benchmark pair, we report the diagnostic
values, the recommended treatment, and the resulting exact match.Boldindicates that the routing heuristic suc-
cessfully selected the exact oracle (best) method.Notes:Entropy is normalized score entropy. This table uses
simplified binary routing (Quality-Ordered→CoT, Stochastic→SDA) for a fair comparison with entropy, which
distinguishes only two treatments. Full four-treatment routing including ATF and PDE is reported in Table 2.
nals agree — high baseline NF with highˆρ 1rout-
ing to ATF; low baseline with a clear monotonic
trend routing to PDE — the routing-relevant de-
cision is unambiguous. The disagreement re-
gion (e.g., monotonic but small magnitudes for
Mistral-7B on FEVER, or large magnitudes but
non-monotonic for Gemma-2 on CONFLICTS)
is precisely the band where the per-query Spear-
manρdistributions in §F.2 expose the case-by-
case structure for inspection. The bootstrap CI
table (Table 13) confirms the per-levelˆρ kval-
ues used to compute either signal are statistically
stable (1,000 resamples; classification stability≥
79%per cell,100%on most strongly-classified
cells). The deployment-ready binary signal there-
fore stands on a continuous robustness check:ρ∗is the binary classifier used for routing, and¯ρis
the magnitude-aware sanity check that motivates
the per-query inspection appendix.
G Statistical Tests and Supplementary
Results
G.1 Token F1 Results
Table G.1 reports Token F1 scores for all model–
benchmark–method combinations, complement-
ing the Exact Match results in Table 2. For
FEVER (binary classification), Token F1 equals
EM. The EM–F1 discrepancy for PDE on CON-
FLICTS (high EM, lower F1 than baseline) is
driven by answer length: PDE’s per-document
voting produces focused 2–3 word answers that
maximize exact match probability, while base-
17

line’s combined-context generation occasionally
produces longer responses with higher token over-
lap but lower exact match rates.
G.2 Bootstrap CIs on Per-Levelˆρ k
Table 13 reports bootstrap 95% confidence in-
tervals (1,000 resamples with replacement over
queries) for each per-level correlationˆρ kand the
trend coefficientρ∗.
G.3 McNemar’s Test Details
Table 14 reports McNemar’s exact test results for
method-vs.-NF comparisons across each model
and benchmark. Here,bandcare discordant-
pair counts: method correct with NF wrong, and
NF correct with method wrong, respectively. Note
that Scoring-only and PDE tests are corrected sep-
arately because they address distinct hypotheses
(incremental scoring improvement vs. isolation
mechanism); joint correction across all 14 tests
would not change PDE significance (p <10−12).
Effect sizes.For PDE, the discordant-pair odds
ratios areb/c= 97/11 = 8.8(Llama) and69/9 =
7.7(Mistral), indicating large effects. For scoring-
only methods, odds ratios range from 1.0 to 2.3,
consistent with small effects.
H Extended Model Scale and Retrieval
Experiments
Table 15 extends the CONFLICTS evaluation
to eight models by adding three larger variants
ranging from 14B to 32B parameters (Qwen2.5-
14B, Gemma-2-27B, and Qwen2.5-32B). This
comprehensively supports our core mechanistic
claim: strong-baseline models possess intrinsic
multi-document handling capacity and are largely
unharmed by forced isolation (PDE). The re-
sults confirm that the isolation-scoring asymme-
try is governed by the model’s baseline capabil-
ity (EM NF≥30%) rather than mere parameter
scale.
The Absolute Capability Floor.It is worth
noting that while PDE provides outsized gains
for weak models, it requires a basic founda-
tional level of instruction-following and single-
document reading comprehension. In our prelim-
inary tests with older generation models lacking
advanced instruction-tuning (e.g., Llama-2-13b),
the baseline performance was catastrophically low
(EM NF= 2.1%). Applying PDE only yielded amarginal increase to7.6%. This establishes a clear
boundary condition: structural isolation cures con-
text confusion, but it cannot synthesize reasoning
capabilities that are fundamentally absent from the
base model.
H.1 Dense Retrieval Generalization
Table 16 evaluates PDE with dense retrieval (Con-
triever top-10) on TriviaQA. We observe a mono-
tonic trend wherein a lower No-Filter (NF) base-
line score correlates with a larger PDE gain
(Spearmanρ=−0.90), demonstrating that the
isolation benefit is not an artifact of BM25’s lower
retrieval precision.
I RSC vs. Score Entropy as Routing
Heuristic
A natural baseline heuristic for evaluating multi-
agent assessment isscore entropy, based on the
premise that high score entropy correlates with re-
trieval noise and scoring uncertainty. Since our
RSC diagnostic also uses score perturbation, a
critical question arises: does RSC provide rout-
ing decisions that aremeaningfully differentfrom
standard entropy-based routing?
We address this question empirically by com-
paring routing accuracy on the 10 model-
benchmark pairs where we have both RSC probes
and full experimental results (5 models×2 bench-
marks: CONFLICTS and FEVER). For each pair,
we determine:
1.RSC recommendation: Simplified binary
routing for fair comparison with entropy. We
select CoT for quality-ordered models (ρ∗=
−1.0) and SDA for stochastic models (ρ∗̸=
−1.0).
2.Entropy recommendation: Following the
entropy-based heuristic, we map normalized
score entropy to treatment selection. High
entropy (>0.7) suggests noisy scoring that
requires robust aggregation (we select SDA),
while low entropy (≤0.7) suggests cleaner
scoring where simpler methods suffice (we
select CoT).
3.Oracle: The best-performing method among
all five strategies (NF, 3-Agent, CoT, SDA,
SDA+CoT) for that pair.
Key findings.(1) RSC and entropydisagreeon
treatment recommendation in 4/10 cases, demon-
strating that the two heuristics capture partially
18

Correlation (ˆρ k) with 95% Bootstrap CI
Model Shuffled (ˆρ 1) Contradicted (ˆρ 2) Random (ˆρ 3) Stability
CONFLICTS
Llama-3.1-8B 0.76[0.71,0.80]0.43[0.35,0.51]0.08[0.01,0.14]100%
Mistral-7B-v0.3 0.35[0.27,0.43]0.22[0.13,0.30]0.18[0.12,0.24]79%
Qwen3-8B 0.39[0.30,0.48]0.04[−0.07,0.15]0.14[0.03,0.23]95%†
Qwen2.5-7B 0.54[0.47,0.61]−0.20 [−0.30,−0.10]0.09[0.02,0.16]100%†
Gemma-2-9B 0.73[0.68,0.79]0.44[0.36,0.53]0.30[0.23,0.36]100%†‡
FEVER
Llama-3.1-8B 0.65[0.56,0.75]0.49[0.36,0.60]0.10[0.00,0.19]100%
Mistral-7B-v0.3 0.28[0.19,0.37]0.09[−0.01,0.19]0.01[−0.08,0.11]93%
Qwen3-8B 0.63[0.54,0.70]0.51[0.41,0.61]0.14[0.04,0.23]100%
Qwen2.5-7B 0.64[0.57,0.71]0.16[0.03,0.29]0.07[−0.04,0.18]92%
Gemma-2-9B 0.58[0.40,0.60]0.36[0.24,0.49]0.09[−0.03,0.21]100%
Table 13:Bootstrap 95% CIs on per-levelˆρ kvalues and RSC classification stability(1,000 bootstrap resamples
over queries). Stab. = percentage of resamples yielding the exact same classification as the full-sample result.
†Stability for stochastic classification.‡Gemma-2×CFL: aggregateˆρ 2≈ˆρ 3yieldsρ∗=−0.50(stochastic). Per-
query meanˆρ 1values may differ from the aggregate-pooled values in Table 4, which pool all document–score
pairs across queries and are the canonical values used for routing decisions.
distinct properties of model scoring behavior.
The 6/10 agreement rate (up from 3/10 prior to
the 100-query RSC re-probing) reflects the up-
dated stochastic classifications for Qwen2.5 and
Gemma-2 on CONFLICTS, which now align
RSC’s SDA recommendations with entropy’s.
(2) RSC beats the NF baseline in 8/10 cases (vs.
7/10 for entropy) and outperforms the 3-Agent
baseline in 5/10 cases (vs. 4/10 for entropy).
(3) RSC matches the oracle in 3/10 cases (vs. 1/10
for entropy), demonstrating better peak routing ac-
curacy.
Interpretation.While both heuristics are com-
petitive for binary CoT/SDA routing, RSC pro-
vides a critical additional advantage: the per-
level coupling valueˆρ 1enables expanded four-
treatment routing (CoT/SDA/PDE/ATF), which
reduces the mean oracle gap on CONFLICTS
to≤0.4pp. Entropy alone cannot motivate the
CoT vs. ATF/PDE distinction because it does not
measurecoupling strengthbetween reasoning and
scores. Combining both signals could yield an
even more robust routing criterion.
J Additional MuSiQue Analysis
Table 17 stratifies Qwen2.5-7B performance by
whether PDE’s selected top-5 passages contain the
full annotated reasoning chain. The gap between
3-Agent and PDE is largest when at least one sup-
porting paragraph is missing, confirming that per-document voting cannot reconstruct chain steps
that retrieval or scoring failed to surface.
Finer stratification by the number of supporting
paragraphs in PDE’s top-5 yields the same mech-
anism: 0 supporting paragraphs→PDE 0%, 3-
Agent 0%; 1→PDE 21.4%, 3-Agent 28.6%; 2
→PDE 40.0%, 3-Agent 48.6%. As hop count in-
creases from 2-hop to 4-hop, the fraction of sup-
porting paragraphs in the selected top-5 falls from
0.57 to 0.33, making the chain-broken regime
dominant.
A second-model check with Mistral-7B-
Instruct-v0.3 further supports the boundary. On
the same 100-query MuSiQue setup, Mistral
has a strongly coupled Quality-Ordered RSC
profile (ˆρ 1= 0.81,ˆρ 2= 0.56,ˆρ 3= 0.10,
ρ∗=−1.0) and an NF baseline of 14.0%. Under
the staticτ NF= 30%rule it would be routed to
PDE, but chain decomposition makes isolation
harmful (PDE 9.0%, PDE-Random 5.0%). Scor-
ing treatments are stronger: 3-Agent and CoT
both reach 20.0%. This mirrors Qwen2.5 and
indicates that multi-hop reasoning requires either
task-aware routing or a treatment designed for
chain assembly.
K Cost–Accuracy Analysis
Table 18 compares cross-encoder reranking (ms-
marco-MiniLM-L-12-v2, 33M parameters;∼40×
cheaper than 3-Agent assessment) against NF.
19

Model Method vs. Baseline Method EM Base EMb c p-value Sig.
Scoring-Only Methods
CONFLICTS
Llama-3.1-8B 3-Agent vs. NF 24.1% 13.9% 36 12 0.001∗∗∗
Mistral-7B-v0.3 CoT vs. NF 22.8% 18.1% 14 11 0.689
Qwen3-8B SDA+CoT vs. NF 63.9% 60.8% 20 15 0.499
Qwen2.5-7B SDA vs. NF 65.4% 59.9% 23 10 0.037
Gemma-2-9B 3-Agent vs. NF 64.6% 60.8% 20 11 0.151
FEVER
Mistral-7B-v0.3 CoT vs. NF 92.6% 90.7% 14 9 0.405
Qwen3-8B SDA vs. NF 90.0% 89.6% 14 12 0.845
Qwen2.5-7B SDA vs. NF 90.5% 87.1% 30 110.005∗∗
Gemma-2-9B SDA+CoT vs. NF 92.7% 92.2% 14 11 0.689
PDE Methods on CONFLICTS
Llama-3.1-8B PDE vs. NF (13.9%) 50.2% 13.9% 97 112.4×10−18∗∗∗
PDE vs. 3-Agent (BL) 50.2% 24.1% 73 119.3×10−12∗∗∗
Mistral-7B-v0.3 PDE vs. NF (18.1%) 43.5% 18.1% 69 91.4×10−12∗∗∗
PDE vs. 3-Agent (BL) 43.5% 18.6% 64 77.7×10−12∗∗∗
Table 14:McNemar’s exact test with Holm-Bonferroni correction (α= 0.05).Best scoring-only methods (top
section) and PDE (bottom section).∗∗p <0.01,∗∗∗p <0.001(after correction).bandcdenote discordant pairs
(method-correct/base-wrong vs. base-correct/method-wrong).
Exact Match (%)
Model Params NF PDE Gain (∆)
Weak-Baseline Category (NF<30%)
Llama-3.1-8B 8B 13.950.2+36.3∗∗∗
Mistral-7B-v0.3 7B 18.143.5+25.4∗∗∗
Avg (Weak) – 16.0 46.9 +30.9
Strong-Baseline Category (NF≥30%)
Qwen2.5-7B 7B 59.9 58.6−1.3
Qwen3-8B 8B60.858.6−2.2
Gemma-2-9B 9B60.859.9−0.9
Qwen2.5-14B†14B 58.261.2+3.0
Gemma-2-27B†27B 62.9 62.0−0.9
Qwen2.5-32B†32B 61.262.0+0.8
Avg (Strong) – 60.6 60.4 −0.3
Table 15:CONFLICTS Exact Match (%) across
model scales.Grouping models by their baseline ca-
pacity clearly reveals the capability divide: PDE con-
sistently benefits weak-baseline models while leaving
strong models unharmed. This pattern persists and gen-
eralizes even as the parameter scale increases up to
32B.†Extended models.∗∗∗p <0.001(McNemar test
with Holm-Bonferroni correction).
Cross-encoder rerankinghurtsweak models on
CONFLICTS (Mistral:−2.1pp; Llama:−4.6pp)
while providing modest gains for strong models
(+2.5–4.2pp).
Table 19 compares NF-only routing (PDE if NF
<30%, BL otherwise) against RSC-based four-
treatment routing. RSC routing reduces the mean
oracle gap from 0.79pp to 0.45pp overall, with
the primary advantage on FEVER where RSC se-
lects appropriate scoring treatments (1.40pp→Model NF PDE Gain (∆)
Llama-3.1-8B 34.655.4+20.8
Mistral-7B-v0.3 38.441.8+3.4
Qwen3-8B 38.844.4+5.6
Qwen2.5-7B 44.047.6+3.6
Gemma-2-9B 55.057.4+2.4
Average 42.949.1+6.2
Table 16:Dense retrieval generalization (TriviaQA
EM % with Contriever top-10).PDE benefits extend
beyond sparse to dense retrieval. Crucially, a mono-
tonic trend emerges: a lower baseline (NF) capacity
strongly correlates with a larger PDE gain (Spearman
ρ=−0.90), with the weakest model (Llama) gaining
+20.8pp.N= 500queries. All models score above
the strictτ NF= 30%threshold, though Llama remains
borderline (34.6%).
0.46pp). On CONFLICTS, both routing strate-
gies achieve the same significant gains via PDE
for weak models.
L Discussion on Excluded Baselines
In evaluating our training-free document assess-
ment pipelines, we intentionally do not com-
pare against MADAM-RAG (Wang et al., 2025c)
for two primary reasons. First, MADAM-RAG
operates at theanswerlevel (debating already-
generated answers), whereas our focus is entirely
at thedocument assessmentlevel (scoring docu-
ments prior to generation). Architecturally, this
makes it incomparable to the early-stage scoring
20

Subset n NF 3-Agent PDE
All queries 100 14.0 30.0 24.0
Chain-complete†23 13.0 39.1 34.8
Chain-broken‡77 14.3 27.3 20.8
Table 17:Chain-coverage analysis on MuSiQue (EM
%).†PDE’s top-5 contains all gold-supporting para-
graphs.‡Top-5 misses at least one supporting para-
graph.
CONFLICTS FEVER
ModelNF CE∆NF CE∆
Mistral-7B-v0.3 18.1 16.0−2.1 90.7 91.3 +0.5
Llama-3.1-8B 13.9 9.3−4.6 88.7 88.9 +0.2
Qwen3-8B 60.8 65.0 +4.2 89.6 90.2 +0.6
Qwen2.5-7B 59.9 64.1 +4.2 87.1 89.8 +2.7
Gemma-2-9B 60.8 63.3 +2.5 92.2 93.3 +1.1
Avg. 42.7 43.5 +0.9 89.7 90.7 +1.0
Table 18: Cross-encoder reranking vs. NF base-
line (EM%) and LLM call costs. CE = ms-marco-
MiniLM-L-12-v2 (33M params,∼40×cheaper than
3-Agent).Notes:LLM calls/query: NF = 1 gen.; 3-
Agent/CoT/SDA≈19; ATF≈19 + 1 gen.; PDE≈19
+dgen. (d=10); PDE-Random =dgen. (no assess-
ment).
pipelines we analyze. Second, MADAM-RAG
demonstrates its gains using massive 70B+ mod-
els. Because our core motivation is resolving
the combinatorial compute overhead specifically
in the highly deployable 7B–9B regime, testing
MADAM-RAG in our setting would conflate ar-
chitectural differences with pure scale effects.
Furthermore, while PDE isolates context to pre-
vent "lost-in-the-middle" syndrome, it fundamen-
tally differs from Fusion-in-Decoder (FiD) (Izac-
ard and Grave, 2021). PDE uses explicit answer
extraction and score-weighted voting, requiring
absolutely no architectural modification or train-
ing. Conversely, FiD relies on internal cross-
attention fusion. Thus, a direct comparison is in-
feasible within our strictly training-free, decoder-
only setting.
M Mechanistic Verification via Token F1
One might suspect that PDE’s massive EM gains
partly reflect an answer formatting artifact, as per-
document prompting inherently elicits concise an-
swers that easily match gold annotations. How-
ever, Token F1 scores (Table G.1), which are
highly robust to verbosity, also show massive im-
provements for weak models under PDE (e.g.,
Llama’s F1 jumps from 0.256 to 0.560). This con-Oracle Gap (pp,↓better)
Model Task NF-Only RSC-Based
Llama-3.1-8B Conflicts0.0 0.0
FEVER 3.10.0
Mistral-7B-v0.3 Conflicts0.0 0.0
FEVER 0.70.0
Qwen3-8B Conflicts0.50.9
FEVER 1.10.0
Qwen2.5-7B Conflicts 0.40.0
FEVER1.62.3
Gemma-2-9B Conflicts0.01.3
FEVER 0.50.0
Mean Gap: Conflicts0.180.44
Mean Gap: FEVER1.400.46
Mean Gap: Overall0.790.45
Table 19:NF-only routing vs. RSC-based routing:
oracle gap (pp).The gap measures performance loss
compared to the oracle best treatment (lower is bet-
ter). While a simplistic NF-only heuristic (PDE if NF
<30%, baseline otherwise) performs well on CON-
FLICTS by successfully isolating weak models, it fails
to optimize scoring treatments for strong models on
FEVER. The RSC-based four-treatment routing cuts
the overall mean oracle gap nearly in half (0.79→
0.45pp).
firms the gain is not a formatting artifact.
Furthermore, applying standard self-
consistency generation from the combined
full context activelyhurtsperformance (−8.6pp
to−12.7pp on CONFLICTS). This confirms that
the root failure mode is the inability to locate
information within long multi-document contexts
(the “lost-in-the-middle” bottleneck). Isolation
structurally bypasses this bottleneck, serving as
the true mechanism behind the performance leap.
21

N Agent System Prompts
The 3-Agent baseline employs three structurally distinct system prompts, one per agent (Relevance As-
sessor, Consistency Verifier, Conflict Detector). CoT De-Polarization replaces each base prompt with a
CoT variant that prepends an explicit step-by-step reasoning template before the JSON output. We list
the system messages verbatim below; the user input concatenates the question and the document(s) to
evaluate.
Relevance Assessor (base)
You are a Relevance Assessor. Your task is to evaluate whether a document contains information that can directly contribute
to answering the given question. Focusonlyon relevance — does this document provide useful evidence for the answer?
Evaluation criteria: (1) Does the document contain key entities or relationships mentioned in the question? (2) Does the
document provide direct evidence that could be used to derive an answer? (3) Distinguish between surface-level keyword
overlap and genuine informational relevance.
Response format:
{"score": <integer 0-5>, "evidence": "<specific quote or paraphrase>",
"reasoning": "<1-2 sentences>"}
Scoring guide:
5 = directly answers with strong evidence; 4 = strong supporting evidence; 3 = partial or indirect evidence; 2 = surface-level
keyword overlap only; 1 = tangentially related; 0 = completely irrelevant.
Consistency Verifier (base)
You are a Consistency Verifier. Your task is to evaluate the internal consistency of a document and its consistency with
other retrieved documents. Focusonlyon consistency — is this document internally coherent and does it agree with other
sources? You will receive: (1) the question, (2) the document to evaluate, (3) a summary of claims from other retrieved
documents. Evaluation criteria: (a) Does the document contain any internal contradictions? (b) Do the document’s claims
agree with the majority of other retrieved documents? (c) Are the claims specific and verifiable, or vague and unsupported?
Response format:
{"score": <0-5>, "internal_consistency": ..., "cross_consistency": ...,
"reasoning": ...}
Scoring guide:
5 = fully consistent internally and with other sources; 3 = some inconsistencies but generally reliable; 0 = internally
contradictory or completely at odds with all other sources.
Conflict Detector (base)
You are a Conflict Detector. Your task is to identify contradictions and conflicts among claims from multiple retrieved
documents. Focusonlyon conflicts — do the documents disagree with each other? You receive: (1) the question, (2) a
list of claim summaries from each retrieved document with document IDs. Evaluation criteria: (a) Do any documents
provide contradictory answers? (b) Are there temporal differences (outdated vs. current information)? (c) Which position
is supported by the majority?
Response format:
{"has_conflict": <bool>, "conflicting_pairs": [{"doc_a": <id>, "doc_b": <id>,
"description": ...}], "majority_position": ..., "doc_adjustments": {<doc_id>:
<-2..+2>}}
Adjustment guide:
+2strongly supported by majority and most recent;+1supported by majority;0no conflict or neutral;−1contradicts
majority;−2contradicts majority and appears outdated.
CoT De-Polarization variants (Relevance / Consistency / Conflict).Each base prompt is rewritten
to require five explicit reasoning steps before the JSON output (e.g., for Relevance: identify key entities
→check document mentions→assess direct evidence→consider temporal relevance→assign score).
The JSON output gains areasoning_stepsarray (one string per step) and aconfidencefield
(1–5). The scoring guide and structural format are otherwise unchanged from the base prompts. The
intent is to disrupt the direct-to-extreme polarization observed in baseline prompts (≈80%of scores at
the floor or ceiling for Mistral-7B on CONFLICTS, dropping to 3.5% under CoT).
Generator prompts.After document selection, the generator produces the final answer using one of
two task-specific prompts.
22

Generator — Factoid QA (TriviaQA, NQ-open, CONFLICTS, MuSiQue)
You are a helpful assistant. Answer the question based ONLY on the provided documents. IMPORTANT: Give ONLY the
answer itself — a name, number, date, or short phrase. Do NOT write a full sentence. Do NOT explain. Examples of good
answers: ‘Paris’, ‘1969’, ‘Albert Einstein’.
Generator — Binary Fact Verification (FEVER)
You are a fact verification assistant. Based ONLY on the provided documents, determine whether the evidence SUPPORTS
or REFUTES the following claim. IMPORTANT: Answer with exactly one word: SUPPORTS or REFUTES. Do NOT
explain.
PDE invokes the factoid generator once per top-kdocument and aggregates predictions via score-
weighted majority voting (Algorithm 2, §E).
23