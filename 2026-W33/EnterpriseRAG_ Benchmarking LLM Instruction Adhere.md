# EnterpriseRAG: Benchmarking LLM Instruction Adherence and Robustness under Non-Ideal Enterprise Retrieval

**Authors**: Huiqi Miao, Xinbao Sun, Bo Wang, Fanyu Meng, Lijun Mei, Na Wu, Di Jin, Chao Deng, Junlan Feng

**Published**: 2026-08-12 02:51:04

**PDF URL**: [https://arxiv.org/pdf/2608.11584v1](https://arxiv.org/pdf/2608.11584v1)

## Abstract
Enterprise RAG deployments face a critical reliability gap: while LLMs satisfy 80% of individual constraints, only 26.8% of responses meet all requirements simultaneously, revealing a 57-point orchestration gap. Existing benchmarks assume clean retrieval with simple queries, failing to capture production conditions where noisy documents and multi-dimensional constraints coexist. We introduce EnterpriseRAG, a benchmark of 983 expert-validated samples across six domains that systematically simulates three failure modes absent from prior work: retrieval noise, knowledge gaps, and factual conflicts, coupled with complex instructions. Evaluation of 13 state-of-the-art LLMs reveals a severe instruction adherence collapse, where high per-constraint satisfaction masks low holistic compliance. Critical findings expose deep barriers under knowledge gaps and factual conflicts, even with reasoning-enhanced inference, indicating production RAG requires explicit context-aware protocols and calibrated judgment. EnterpriseRAG provides a reproducible foundation for measuring and closing these gaps, directly informing deployment decisions for enterprise-scale RAG systems. We will release the benchmark and evaluation framework upon publication.

## Full Text


<!-- PDF content starts -->

EnterpriseRAG: Benchmarking LLM Instruction Adherence and
Robustness under Non-Ideal Enterprise Retrieval
Huiqi Miao, Xinbao Sun, Bo Wang, Fanyu Meng, Lijun Mei, Na Wu, Di Jin, Chao Deng, Junlan Feng
Jiutian Research, China Mobile, Beijing, China
{miaohuiqi,sunxinbao,wangbo}@cmjt.chinamobile.com
Abstract
Enterprise RAG deployments face a critical re-
liability gap: while LLMs satisfy individual
constraints at rates up to 84%, only 27% of re-
sponses meet all requirements simultaneously,
revealing a 57-point orchestration gap. Exist-
ing benchmarks assume clean retrieval with
simple queries, failing to capture production
conditions where noisy documents and multi-
dimensional constraints coexist. We introduce
EnterpriseRAG, a benchmark of 983 expert-
validated samples across six domains that sys-
tematically simulates three failure modes ab-
sent from prior work: retrieval noise, knowl-
edge gaps, and factual conflicts, coupled with
complex instructions. Evaluation of 13 state-
of-the-art LLMs reveals a severe instruction
adherence collapse, where high per-constraint
satisfaction masks low holistic compliance.
Critical findings expose deep barriers under
knowledge gaps and factual conflicts, even with
reasoning-enhanced inference, indicating pro-
duction RAG requires explicit context-aware
protocols and calibrated judgment. Enterpris-
eRAG provides a reproducible foundation for
measuring and closing these gaps, directly in-
forming deployment decisions for enterprise-
scale RAG systems. We will release the bench-
mark and evaluation framework upon publica-
tion.
Keywords:RAG benchmark, instruction follow-
ing, LLM robustness, enterprise retrieval, knowl-
edge gaps, factual conflicts, retrieval noise
1 Introduction
Enterprise RAG systems face complex queries like
"Summarize Q3 revenue by region in markdown ta-
bles. If data is incomplete, state ’Data Unavailable’
rather than estimating. If audit and management re-
ports conflict, cite both explicitly."requiring simul-
taneous factual extraction, formatting compliance,
and protocol adherence.These challenges stem not from inadequate fac-
tual grounding, but from a fundamental evaluation
gap. Current benchmarks assess RAG systems on
clean retrieval scenarios with simple queries (Gao
et al., 2023; Es et al., 2025), while production de-
ployments face three compounding challenges ab-
sent from existing evaluations:(1)complex multi-
constraint instructions integrating formatting rules
with context-aware protocols for evidence adju-
dication;(2)high retrieval noise from latency-
constrained systems that surface 10–20 documents
with substantial irrelevant content;(3)frequent
knowledge failures including coverage gaps and
factual conflicts driven by temporal drift or source
fallibility.
While recent work advances robustness testing
(Zeng et al., 2025) and instruction following (Dong
et al., 2024), these efforts evaluate constraints in
isolation with synthetic noise, missing the com-
pounding complexity of real enterprise workflows
where multiple dimensions interact.
We introduceEnterpriseRAG, a benchmark
grounded in real-world enterprise deployments,
comprising 983 expert-validated samples across six
vertical domains. Unlike prior benchmarks over-
laying synthetic instructions onto standard datasets,
EnterpriseRAG reflects authentic multi-domain sce-
narios derived from real operational queries. We
preserve original user intents while systematically
scaling up constraint complexity through an expert-
informed synthesis protocol, and construct three
orthogonal non-ideal retrieval modes (irrelevant
noise, knowledge gaps, and factual conflicts) vali-
dated through LLM-assisted generation and human
verification. Our evaluation framework extends
traditional RAG metrics withStrict IAS(holistic
compliance) versusLoose IAS(per-constraint sat-
isfaction) to expose compositional adherence fail-
ures, plus robustness indicators for safety-critical
scenarios. Testing 13 state-of-the-art LLMs reveals
that while models handle structural formatting ad-
1
arXiv:2608.11584v1  [cs.AI]  12 Aug 2026

Table 1: Comparison with prior RAG benchmarks across Source, Complexity, and Robustness.
Benchmark Dataset SourceTask
ComplexityRobustness
Human-
CuratedVertical
DomainNatural User
QueriesComplex
ConstraintsNegative
RejectionConflict
CRUD-RAG(Lyu et al., 2024)✗ ✓ ✗ ✗ ✗ ✗
CRAG(Yang et al., 2024)✓ ✓ ✗ ✗ ✓ ✗
RAGBench(Friel et al., 2025)✗ ✓ ✓ ✗ ✗ ✗
RAGEval(Zhu et al., 2025)✗ ✓ ✗ ✗ ✗ ✗
FollowRAG(Dong et al., 2024)✓ ✗ ✗ ✓ ✗ ✗
EKRAG(Yu et al., 2025)✓ ✓ ✓ ✗ ✗ ✗
RARE(Zeng et al., 2025)✗ ✓ ✗ ✗ ✓ ✓
GaRaGe(Sorodoc et al., 2025)✓ ✓ ✗ ✗ ✓ ✗
EnterpriseRAG (Ours)✓ ✓ ✓ ✓ ✓ ✓
equately, they systematically fail on behavioral
protocols, particularly judgment under uncertainty,
where even reasoning models achieve insufficient
reliability for production deployment.
Contributions.Our contributions are threefold:
•Enterprise-grade RAG benchmark: We in-
troduceEnterpriseRAG, 983 expert-validated
instances across six domains, pairing complex
multi-constraint instructions with three con-
trolled non-ideal retrieval settings, plus a repro-
ducible construction pipeline.
•Comprehensive evaluation framework:We de-
velop specialized metrics for non-ideal contexts:
Loose/Strict IASfor instruction adherence, and
rejection/conflict accuracyto measure safety-
critical judgment in scenarios with missing or
conflicting information.
•Experimental Insights: Across 13 LLMs, we
find an orchestration gap of up to 57pp (83.8%
Loose vs. 26.8% Strict), with best rejection accu-
racy only 42.7% and conflict recognition <45%,
indicating behavioral judgment under uncertainty
as the main bottleneck for enterprise RAG.
2 Related Work
RAG evaluation has evolved from factual correct-
ness toward robustness and instruction compliance
(Zhao et al., 2024; Gupta et al., 2024). Table 1 posi-
tions EnterpriseRAG among representative bench-
marks. A recurring pattern across prior work is that
retrieval quality and instruction adherence are stud-
ied separately—leaving open whether models can
satisfy both under realistic enterprise conditions.
RAG Evaluation Paradigms.Early benchmarks
focused on factual accuracy and grounding (ALCE
(Gao et al., 2023), RAGAS (Es et al., 2025))
or multi-hop reasoning (Tang and Yang, 2024).While domain-specific benchmarks exist ((Jin et al.,
2019);(Pipitone and Alami, 2024);(Chen et al.,
2023b)), they often rely on curated sources like
Wikipedia (Yang et al., 2018). Recent frameworks
like RAGBench (Friel et al., 2025), RAGEval (Zhu
et al., 2025) and EKRAG (Yu et al., 2025) offer
multi-dimensional metrics and human-curated en-
terprise samples but lack systematic assessment of
complex instruction adherence.
Non-Ideal Retrieval Contexts.Benchmarks like
CRAG (Yang et al., 2024), CRUD-RAG (Lyu et al.,
2024) and GaRaGe (Sorodoc et al., 2025) address
dynamic KBs and calibration. Others such as RGB
(Chen et al., 2023a), RARE (Zeng et al., 2025), and
Magic Mushroom (Zhang et al., 2025) introduce
noise and unanswerability. While conflict (Lee
et al., 2025; Choi et al., 2025) and gap management
(Guo et al., 2025) exist, they are typically evaluated
in isolation, decoupled from uncertainty protocols.
Instruction Following in RAG.General bench-
marks (Zhou et al., 2023; Jiang et al., 2024; Wen
et al., 2024; Zou et al., 2025; Qin et al., 2024)
highlight the need for strict verification of format-
ting and negative constraints. In RAG contexts,
MT-RAG (Katsis et al., 2025) and CRP-RAG (Xu
et al., 2025) overlook strict protocol adherence,
while FollowRAG (Dong et al., 2024) remains
limited by synthetic injection and clean contexts.
EnterpriseRAG targets the intersection: complex
multi-constraint instructions grounded in opera-
tional workflows, evaluated under systematic re-
trieval noise, knowledge gaps, and factual conflicts.
3 EnterpriseRAG Benchmark
Construction
We construct EnterpriseRAG from production
RAG logs, yielding 983 expert-validated instances
2

Data SourcesReal User queriesEnergy / Medical / Legal / Finance...Domain Knowledge BasesTraining manuals / Financial reports...
Non
-
Ideal ContextsRetrieval NoiseTests: attention allocationKnowledge GapsTests: rejection protocalFactual ConflictsTests: conflict resolution
Benchmark Construction PipelineInstruction SynthesisAtomic Library (50-60/task):• Persona definition• Operational goals• Output constraints• Knowledge protocols ★Constraint FusionSample constraints↓Conflict detection↓Query-aware refinementContext InjectionNon-ideal simulation:• 40-60% irrelevant docs• Query outside KB •Contradictory evidence  Human VerificationEnterpriseRAG: 983 samples in 6 domains
Benchmark EvaluationRAG Metrics
FaithfulnessAnswer Coverage
Rule & LLM judgeLoose/Strict IAS
Rejection AccConflict Rec. AccInstruction AdherenceRobustnessFigure 1: EnterpriseRAG Overview. The top section presents the pipeline for complex instruction schema de-
sign, while the bottom section shows non-ideal context simulation and evaluation metrics, respectively. All are
automatically generated by LLM and quality verified by humans.
spanning six domains (Energy, Medical, Legal, Fi-
nancial, Party Building and Web Search). All data
are desensitized to remove PII; raw logs cannot
be released, but we will release the desensitized
benchmark and a reproducible generation pipeline
(details in Appendix A.1).
Instance format.Each instance is a triple
⟨q,I,D⟩ : user query q, a fused multi-constraint
instruction set I, and a retrieved document bun-
dleD. Starting from 491 authentic queries, we
construct 983 instances by pairing queries with
controlled non-ideal retrieval scenarios (overview
in Figure 1; subset composition in Table 6). Figure
15 illustrates a Legal domain case study.
Construction pipeline.Our pipeline operational-
izes two principles:realistic constraint complex-
ity(from enterprise prompts) andcontrolled non-
ideal retrieval(noise/gap/conflict). We: (1) col-
lect and filter queries; (2) synthesize Iby fusing
atomic constraints and removing internal contradic-
tions; (3) retrieve documents via hybrid retrieval
(BM25+dense) and assign non-ideal modes; (4)
conduct expert verification for instruction–query
consistency and context validity (Appendix A.2).
Prompt templates and quality control recipes are
documented in Appendix E.1.
15.0 17.5 20.0 22.5 25.0 27.5
Strict IAS (Filtered) (%)510152025303540Rejection Accuracy (%)
Qwen3-8bQwen3-14b
Qwen3-32bQwen3-235B-A22B-Thinking-2507
Qwen3-30B-A3B-Instruct-2507Qwen3-235B-A22B-Instruct-2507DeepSeek-R1-0528
DeepSeek-V3.1GLM-4.5Gemini-2.5-Pro
Claude-Sonnet-4
GPT-4.1Claude-Opus-4.5
Model Type
Reasoning
StandardFigure 2: Instruction Adherence and Robustness com-
parison on the knowledge gap subset. Reasoning-
enhanced models generally demonstrate superior ca-
pability in both protocol adherence and refusal of unan-
swerable queries.
3.1 Complex Instruction Schema
We organize constraints into three orthogonal
dimensions:Persona Definition,Output Con-
straints, andKnowledge Interaction Protocols.
This schema captures enterprise-critical behavioral
requirements (e.g., citation, gap identification, con-
flict handling) in addition to structural formatting
rules. Definitions and distributions are provided in
Appendix A.3–A.4 (Table 4, Table 5, Figure 7).
3

Table 2: Performance of various LLMs on noisy subset. The average scores across the dataset are reported as
percentages. The best and second-best scores are marked inboldand underlined , respectively.
ModelInference
ParadigmRAG Quality IAS Loose IAS
FaithfulnessAnswer
CoverageLoose StrictPersona
DefinitionOutput
ConstraintsKnowledge
Interaction
Protocol
open-source models
Qwen3-8b Reasoning 64.8 56.4 75.5 12.3 70.1 82.0 70.3
Qwen3-14b Reasoning 66.8 59.4 77.0 14.3 74.2 81.5 72.4
Qwen3-32b Reasoning 64.9 59.5 77.2 15.9 75.4 80.1 75.1
Qwen3-235B-A22B-Thinking-2507 Reasoning 67.1 64.183.8 26.882.286.282.5
Qwen3-30B-A3B-Instruct-2507 Standard 63.9 65.6 76.4 13.2 79.0 81.5 68.9
Qwen3-235B-A22B-Instruct-2507 Standard 67.4 67.1 80.6 20.8 82.3 83.6 76.3
DeepSeek-R1-0528 Reasoning 68.9 66.4 83.1 21.9 81.3 84.382.9
DeepSeek-V3.1 Standard 69.9 60.4 82.2 22.1 79.9 85.9 78.8
GLM-4.5 Reasoning 76.6 64.3 81.6 21.5 75.4 85.7 78.7
closed-source models
Gemini-2.5-Pro Reasoning 73.5 61.6 83.7 26.5 86.685.6 80.3
GPT-4.1 Standard 69.8 65.8 80.0 19.5 79.4 82.1 77.8
Claude-Opus-4.5 Reasoning 76.468.583.3 25.3 84.7 84.1 82.4
Claude-Sonnet-4 Standard76.866.7 79.8 19.5 77.1 81.5 79.1
15 20 25 30 35 40 45 50
Conflict Recognition Rate (%)4850525456586062Answer Coverage (%)
ρ = +0.90
ρ = -0.50
Qwen3-8B Qwen3-14BQwen3-32BQwen3-235B-TQwen3-30B-I
Qwen3-235B-IDS-R1
DS-V3GLM-4.5
Gemini-2.5
Sonnet-4GPT-4.1
Opus-4.5Reasoning
Standard
Figure 3: Correlation between Conflict Recognition
Rate and Answer Coverage across 13 models on the
factual conflict subset (309 cases). Each point repre-
sents one model. Reasoning-enhanced models exhibit a
strong positive correlation ( ρ= +0.90), while standard
models show no significant relationship (ρ= -0.50).
3.2 Non-Ideal Retrieval Scenarios
We construct three non-ideal retrieval scenarios
that stress the generator under realistic enterprise
failure conditions:Noisy Retrieval(topically simi-
lar but contextually irrelevant documents),Knowl-
edge Gaps(topically related but insufficient evi-
dence in retrieved contexts), andFactual Conflicts
(contradictory statements in retrieved passages).
Because gaps and conflicts are sparse in natural
logs, we augment them with controlled procedures
while preserving domain coherence; we further
validate that synthetic conflicts match natural dif-
ficulty on core robustness signals (Appendix A.5and Appendix B.2).
3.3 Evaluation Metrics
Given the open-ended nature of enterprise queries,
we report RAG quality and instruction adher-
ence signals without requiring gold reference an-
swers. Faithfulness and Answer Coverage follow
a RAGAS-style claim-based evaluation (Es et al.,
2025).
Faithfulness ( F).We calculate faithfulness as
F=|C sup|/|C total|, where Ctotal denotes all
claims extracted from the response, and Csupde-
notes those supported by the retrieved context.
Answer Coverage (C).
C=α|Cans∩R|
|Cans|+ (1−α)|Sans∩R|
|Sans|(1)
where CansandSansdenote core and supplemen-
tary claims from contexts, respectively, Rrepre-
sents the response content, and α= 0.7 weights
core claims higher.
Instruction Adherence Score (IAS).We report:
Loose IASas the proportion of satisfied constraints,
andStrict IASas a binary score indicating whether
all constraints are satisfied.
Robustness metrics.For non-ideal subsets, we
compute:Rejection Accuracyon knowledge gaps,
andConflict Recognition Accuracyon factual con-
flicts.
4

Model CR Model CR
DeepSeek-R1†44.3 DeepSeek-V3.1 29.8
Gemini-2.5-Pro†42.3 Qwen3-30B-I 28.5
Qwen3-235B-T†41.4 Qwen3-14B-T†27.2
GLM-4.5†40.1 Claude-Opus†26.9
Qwen3-235B-I 37.5 Claude-Sonnet 25.9
Qwen3-32B-T†36.6 Qwen3-8B-T†23.9
GPT-4.1 18.5
†Reasoning-enhanced. CR: Conflict Recog. (%).
Table 3: Conflict recognition rates (CR) across 13 mod-
els.
4 Experiments
4.1 Experimental Setup
Models.We evaluate 13 LLMs spanning
open/closed-source and standard/reasoning-
enhanced variants (Yang et al., 2025; DeepSeek-AI
et al., 2025; OpenAI, 2025; Team et al., 2025a;
Comanici et al., 2025; Anthropic, 2025), full list
in Table 2. For consistency, we adopt simplified
names after the first mention: “Thinking” models
are denoted as-Thinking(abbrev.-T) and standard
instruction-tuned counterparts as-Instruct(abbrev.
-I). We omit version suffixes (e.g.,-2507) unless
needed for disambiguation.
Evaluation protocol.Results are reported on
three non-ideal subsets: Noisy Retrieval ( n=447 ),
Knowledge Gaps ( n=227 ), and Factual Conflicts
(n=309 ). IAS evaluation uses rule-based checks
for structural constraints (Zhou et al., 2023) and
LLM-as-a-judge (Kimi-k2-thinking (Team et al.,
2025b)) for behavioral protocols. Not all in-
stances include explicit Knowledge Interaction Pro-
tocol constraints; we therefore compare naturally
protocol-present vs. protocol-absent cases for ro-
bustness analyses. Evaluation prompt templates
are in Appendix E.
Evaluator reliability.Cross-judge comparison
across three LLM evaluators shows stable scores
and consistent model rankings (Section B.1). On
150 human-annotated samples, experts achieve
strong agreement ( κ=0.85 ), and the LLM evalu-
ator (Kimi-k2-thinking) aligns well with human-
annotated gold labels ( κ=0.77 , 88% agreement),
with especially high alignment on conflict recogni-
tion (κ=0.93; Section B.3).
4.2 Main Results
Finding 1: Orchestration under noisy retrieval.
Table 2 reveals a severeadherence collapse: Loose
IAS achieves up to 83.8%, yet Strict IAS reaches
0 2 4 6 8 10 12 14 16
Error Percentage Distribution (%)Output Constraints: OrderingOutput Constraints: /glyph1197egativeOutput Constraints: StyleOutput Constraints: ContentOutput Constraints: FormatKnowledge Protocol: ConflictKnowledge Protocol: UncertaintyKnowledge Protocol: FilteringKnowledge Protocol: Gap IDKnowledge Protocol: CitationPersona Definition: RolePersona Definition: AudienceError Sub-Dimension Distribution Comparison
Models:
Qwen3-235b-think
Qwen3-235b-instructFigure 4: Error distribution by constraint category.
Knowledge Interaction Protocols exhibit the highest
failure rates, confirming that behavioral judgment, not
formatting, is the core bottleneck.
Noisy Retrieval
Strict IASKnowledge Gaps
Strict IASFactual Conflicts
Strict IASKnowledge Gaps
Reject AccFactual Conflicts
Conflict Recog Acc010203040506070Accuracy (%)Reasoning vs Standard Models across RAG Scenarios
+6.1pp**-0.2pp
+10.6pp***
+2.2pp+7.2pp**
+1.0pp+18.1pp***+11.1pp***+3.6pp+14.4pp***
Standrd Instruct Model Reasoning Gain (Thinking) Qwen3 DeepSeek
Figure 5: Reasoning vs. standard model accuracy across
RAG scenarios. Blue segments denote reasoning gains
over standard baselines (yellow). Error bars: 95% CI.
∗p < .05,∗∗p < .01,∗∗∗p < .001(McNemar’s test).
only 26.8% (Qwen3-235B-Thinking). This 57-
point gap quantifies thecompositional bottleneck
where models satisfy individual constraints but fail
holistic compliance. Reasoning-enhanced models
consistently outperform standard variants, with the
largest gains in Knowledge Interaction Protocols.
Finding 2: Rejection under knowledge gaps.
In production, hallucinating on unanswerable
queries is often more harmful than being unhelp-
ful. Figure 2 exposes a pervasivehelpfulness bias:
Qwen3-30B-Instruct achieves only 6.6% rejection
accuracy, hallucinating in 93.4% of unanswerable
cases. Reasoning-enhanced models improve sub-
stantially (Claude-Opus-4.5: 42.7%), yet remain
far from production-grade reliability.
5

Qw3-8BQw3-32BQw3-30BGPT-4.1Qw3-14BClaude-O4Claude-S4Qw3-235B-IGLM-4.5GeminiDS-V3Qw3-235B-TDS-R1
***A. Rejection Accuracy
-20 0 20 40 60Qw3-8BQw3-32BQw3-30BGPT-4.1Qw3-14BClaude-O4Claude-S4Qw3-235B-IGLM-4.5GeminiDS-V3Qw3-235B-TDS-R1
***************************************B. Conflict Recognition Accuracy
Improved (  > 0)
 No gain (   0)
 * p<.05  ** p<.01  *** p<.001Figure 6: Effect of knowledge interaction protocol
on(A)rejection accuracy and(B)conflict recognition.
Points show accuracy differences (with/without proto-
col) with 95% CIs. The protocol significantly improves
conflict recognition in all 13 models, with more modest
effects on rejection accuracy (2/13 significant). Inde-
pendent samples (with/without: 157/69 for A, 113/193
for B);p-values fromχ2test with Yates’ correction.
Finding 3: Conflict recognition.Table 3 shows
conflict detection remains a bottleneck: top mod-
els reach only 40–44% recognition (DeepSeek-R1:
44.3%), while GPT-4.1 detects merely 18.5%. Fig-
ure 3 shows reasoning-enhanced models achieve
a strong positive correlation between recognition
and coverage ( ρ= + 0.90 ,p<0.01 ), while standard
models show no consistent relationship ( ρ=−0.50 )
with high variance. This suggests inference-time
computation may resolve the traditional safety-
informativeness dilemma. Synthetic and natural
conflicts show equivalent difficulty on core metrics
(Section B.2).
4.3 Analysis
Protocol bottleneck.Figure 4 decomposes IAS
failures by constraint category. Knowledge Inter-
action Protocols exhibit the highest error rates and
variance, particularly for citation and gap identifi-
cation. Comparing Qwen3-235B-Thinking to its
Instruct counterpart, the largest reasoning gains
occur precisely in these protocol dimensions, con-
firming thatjudgment under uncertaintyis the
core enterprise bottleneck.
Scaling.Within Qwen3-Thinking, Strict IAS
scales non-linearly (12.3% at 8B →26.8% at 235B)
while Faithfulness saturates (64.8% →67.1%), in-dicating orchestration is an emergent capability
requiring substantial scale.
Reasoning vs. standard instruction-tuned vari-
ants.Figure 5 compares matched reasoning
vs. standard variants (Qwen3-235B-Thinking
vs. Qwen3-235B-Instruct; DeepSeek-R1 vs.
DeepSeek-V3.1). Across scenarios, reasoning vari-
ants exhibit substantial robustness gains: Qwen3-
235B-Thinking boosts rejection accuracy by
18.1pp and DeepSeek-R1 improves conflict recog-
nition by 14.4pp, consistent with reduced helpful-
ness bias. For Strict IAS, Qwen3-235B-Thinking
shows consistent gains (+6.1pp to +10.6pp; p<.01 ),
while DeepSeek-R1 shows minimal improvement,
suggesting architecture-dependent benefits.
Effect of explicit protocols.Figure 6 compares
instances with explicit Knowledge Interaction Pro-
tocol constraints to those without such constraints
under the same retrieval failure mode. Overall, ex-
plicit protocols yield a large and consistent gain in
conflict recognition across all 13 models, but only
modest improvements in rejection under knowl-
edge gaps. This asymmetry suggests that protocols
help most when the failure is explicit in-context
(contradictions), whereas proper refusal requires a
harder judgment of evidence sufficiency and sep-
arating parametric knowledge from retrieved evi-
dence. Notably, Claude-Opus-4.5 shows a small,
non-significant decrease, indicating potential inter-
action with model-specific safety behaviors. Be-
cause this is an observational comparison (protocol
presence is not randomized), we report domain-
level breakdowns in Section D.1.
5 Conclusion
EnterpriseRAG combines complex multi-constraint
instructions with three non-ideal retrieval modes
across 983 expert-validated instances. Across
13 LLMs, we find a persistent orchestration col-
lapse: even the best model reaches 83.8% per-
constraint adherence (Loose IAS) but only 26.8%
holistic compliance (Strict IAS), leaving a 57-point
gap. Robustness failures concentrate in knowledge-
interaction protocols: under knowledge gaps, mod-
els frequently over-answer despite explicit refusal
requirements (with Claude-Opus-4.5 peaking at
42.7%); under factual conflicts, even the strongest
systems recognize contradictions in fewer than half
of cases (led by DeepSeek-R1 at 44.3%).
Practically, our results suggest that enterprise-
6

ready RAG requires (i) training and evaluation tar-
geted at protocol-level judgment (evidence suffi-
ciency, calibrated refusal, and conflict-aware re-
porting), not just formatting or factuality; and (ii)
explicit operational protocols in prompts, which
reliably improve conflict handling but are insuffi-
cient to solve evidence-gap refusal. EnterpriseRAG
provides a realistic and reproducible foundation to
measure and close these gaps.
6 Limitations
While EnterpriseRAG encompasses six diverse do-
mains, the current scope is limited to text-based
RAG. Multimodal contexts, such as those involving
charts or images within PDFs, are not yet included.
Additionally, our reliance on a reasoning-enhanced
LLM as an evaluator, while effective, may intro-
duce bias compared to human evaluation, although
our sampling checks indicate high alignment. Fi-
nally, the strict adherence metric is binary and strin-
gent; future metrics could explore more nuanced
semantic gradations of constraint satisfaction.
References
Anthropic. 2025. Introducing Claude Opus 4.5. Blog
post.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2023a. Benchmarking large language mod-
els in retrieval-augmented generation. Preprint ,
arXiv:2309.01431.
Wei Chen, Qiushi Wang, Zefei Long, Xianyin Zhang,
Zhongtian Lu, Bingxuan Li, Siyuan Wang, Jiarong
Xu, Xiang Bai, Xuanjing Huang, and Zhongyu Wei.
2023b. DISC-FinLLM: A Chinese Financial Large
Language Model based on Multiple Experts Fine-
tuning.
Eunseong Choi, June Park, Hyeri Lee, and Jong-
wuk Lee. 2025. Conflict-aware soft prompting
for retrieval-augmented generation. In Proceedings
ofthe2025 Conference onEmpirical Methods in
Natural Language Processing , pages 26969–26983,
Suzhou, China. Association for Computational Lin-
guistics.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, and 1 others. 2025. Gemini 2.5:
Pushing the frontier with advanced reasoning, multi-
modality, long context, and next generation agentic
capabilities. Preprint, arxiv:2507.06261.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhi-
hong Shao, Zhuoshu Li, Ziyi Gao, and 181 others.
2025. DeepSeek-r1: Incentivizing reasoning capa-
bility in LLMs via reinforcement learning. Preprint ,
arxiv:2501.12948 [cs].
Guanting Dong, Xiaoshuai Song, Yutao Zhu, Runqi
Qiao, Zhicheng Dou, and Ji-Rong Wen. 2024. To-
ward General Instruction-Following Alignment for
Retrieval-Augmented Generation. arXiv preprint .
ArXiv:2410.09584 [cs].
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2025. Ragas: Automated Eval-
uation of Retrieval Augmented Generation. arXiv
preprint. ArXiv:2309.15217 [cs].
Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2025.
RAGBench: Explainable Benchmark for Retrieval-
Augmented Generation Systems. arXiv preprint .
ArXiv:2407.11005 [cs].
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi
Chen. 2023. Enabling Large Language Models
to Generate Text with Citations. arXiv preprint .
ArXiv:2305.14627 [cs].
Xiaofan Guo, Yaxuan Luan, Yue Kang, Xiangchen
Song, and Jinxu Guo. 2025. Llm-centric rag with
multi-granular indexing and confidence constraints.
Preprint, arXiv:2510.27054.
Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A comprehensive survey of
retrieval-augmented generation (rag): Evolution,
current landscape and future directions. Preprint ,
arXiv:2410.12837.
Xinguang Jiang, Sihan Hu, Dingfu Yu, Yuhao Zhang,
Zhongliang Yang, Yu Li, Linna Zhou, and Valuesim-
plex AI Lab. 2023. FinLongEval. https://github.
com/valuesimplex/FinLongEval.
Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun
Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin
Jiang, Qun Liu, and Wei Wang. 2024. Follow-
Bench: A multi-level fine-grained constraints fol-
lowing benchmark for large language models. In
Proceedings ofthe62nd Annual Meeting ofthe
Association forComputational Linguistics (V olume
1:Long Papers) , pages 4667–4688, Bangkok, Thai-
land. Association for Computational Linguistics.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. PubMedQA: A
Dataset for Biomedical Research Question Answer-
ing. In Proceedings ofthe2019 Conference on
Empirical Methods inNatural Language Processing
and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP) ,
pages 2567–2577, Hong Kong, China. Association
for Computational Linguistics.
Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chu-
laka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj
7

Shah, Huaiyu Zhu, Danish Contractor, and Ma-
rina Danilevsky. 2025. MTRAG: A Multi-Turn
Conversational Benchmark for Evaluating Retrieval-
Augmented Generation Systems. arXiv preprint .
ArXiv:2501.03468 [cs].
Jungyeon Lee, Kangmin Lee, and Taeuk Kim. 2025.
Magic: A multi-hop and graph-based benchmark for
inter-context conflicts in retrieval-augmented genera-
tion. Preprint, arXiv:2507.21544.
Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong,
Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu,
Tong Xu, and Enhong Chen. 2024. CRUD-RAG: A
Comprehensive Chinese Benchmark for Retrieval-
Augmented Generation of Large Language Models.
arXiv preprint. ArXiv:2401.17043 [cs].
OpenAI. 2025. Introducing GPT-4.1 in the API. Blog
post.
Nicholas Pipitone and Ghita Houir Alami. 2024.
LegalBench-RAG: A Benchmark for Retrieval-
Augmented Generation in the Legal Domain. arXiv
preprint. ArXiv:2408.10343 [cs].
Yanzhao Qin, Tao Zhang, Tao Zhang, Yanjun Shen,
Wenjing Luo, Haoze Sun, Yan Zhang, Yujing Qiao,
Weipeng Chen, Zenan Zhou, Wentao Zhang, and Bin
Cui. 2024. SysBench: Can large language models fol-
low system messages? Preprint , arxiv:2408.10943
[cs]. Version: 1.
Ionut-Teodor Sorodoc, Leonardo F. R. Ribeiro, Rex-
hina Blloshmi, Christopher Davis, and Adrià de Gis-
pert. 2025. GaRAGe: A Benchmark with Grounding
Annotations for RAG Evaluation. arXiv preprint .
ArXiv:2506.07671 [cs].
Yixuan Tang and Yi Yang. 2024. MultiHop-
RAG: Benchmarking Retrieval-Augmented Gen-
eration for Multi-Hop Queries. arXiv preprint .
ArXiv:2401.15391 [cs].
GLM-4 5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng,
Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang
Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong
Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin
Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei,
and 152 others. 2025a. GLM-4.5: Agentic, reason-
ing, and coding (ARC) foundation models. Preprint ,
arxiv:2508.06471 [cs].
Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jia-
hao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen,
Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei
Cui, Hao Ding, Mengnan Dong, Angang Du, Chen-
zhuang Du, Dikang Du, Yulun Du, Yu Fan, and 150
others. 2025b. Kimi k2: Open agentic intelligence.
Preprint, arxiv:2507.20534 [cs].
Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao
Huang, Jinfeng Zhou, Wenchuang Li, Binxin
Hu, Wendy Gao, Jiaxin Xu, Yiming Liu, Jie
Tang, Hongning Wang, and Minlie Huang. 2024.
Benchmarking Complex Instruction-Following with
Multiple Constraints Composition.Kehan Xu, Kun Zhang, Jingyuan Li, Wei Huang, and
Yuanzhuo Wang. 2025. CRP-RAG: A retrieval-
augmented generation framework for supporting
complex logical reasoning and knowledge planning.
Electronics, 14(1):47.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41
others. 2025. Qwen3 technical report. Preprint ,
arxiv:2505.09388 [cs].
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze Daniel
Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong,
Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan,
Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang,
and 8 others. 2024. CRAG – Comprehensive RAG
Benchmark. arXiv preprint . ArXiv:2406.04744 [cs].
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. Preprint, arXiv:1809.09600.
Tan Yu, Wenfei Zhou, Leiyang Leiyang, Aaditya
Shukla, Mmadugula Mmadugula, Pritam Gundecha,
Nicholas Burnett, Anbang Xu, Viseth Viseth, Tbar
Tbar, Rama Akkiraju, and Vivienne Zhang. 2025.
EKRAG: Benchmark RAG for Enterprise Knowl-
edge Question Answering. In Proceedings ofthe4th
International Workshop onKnowledge-Augmented
Methods forNatural Language Processing , pages
152–159, Albuquerque, New Mexico, USA. Asso-
ciation for Computational Linguistics.
Yixiao Zeng, Tianyu Cao, Danqing Wang, Xinran Zhao,
Zimeng Qiu, Morteza Ziyadi, Tongshuang Wu, and
Lei Li. 2025. RARE: Retrieval-aware robustness
evaluation for retrieval-augmented generation sys-
tems. Preprint, arXiv:2506.00789.
Yuxin Zhang, Yan Wang, Yongrui Chen, Shenyu Zhang,
Xinbang Dai, Sheng Bi, and Guilin Qi. 2025. Magic
mushroom: A customizable benchmark for fine-
grained analysis of retrieval noise erosion in rag sys-
tems. Preprint, arXiv:2506.03901.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhen-
gren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024.
Retrieval-augmented generation for ai-generated con-
tent: A survey. Preprint, arXiv:2402.19473.
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha
Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and
Le Hou. 2023. Instruction-following evaluation for
large language models. Preprint , arxiv:2311.07911
[cs].
Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan,
Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang,
Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and
8

Maosong Sun. 2025. RAGEval: Scenario Spe-
cific RAG Evaluation Dataset Generation Framework.
arXiv preprint. ArXiv:2408.01262 [cs].
Tao Zou, Xinghua Zhang, Haiyang Yu, Minzheng Wang,
Fei Huang, and Yongbin Li. 2025. EIFBENCH: Ex-
tremely complex instruction following benchmark
for large language models. In Proceedings ofthe
2025 Conference onEmpirical Methods inNatural
Language Processing , pages 20941–20964. Associa-
tion for Computational Linguistics.
A Dataset Details & Statistics
A.1 Data Source, Privacy, and Domains
Our benchmark is constructed from real-world en-
terprise operational logs in China, collected under
internal data usage agreements. The native lan-
guage of EnterpriseRAG is Chinese. To ensure
privacy, all raw logs undergo a strict multi-stage
desensitization pipeline, including rule-based re-
moval of sensitive fields and manual expert review.
Consequently, no personally identifiable informa-
tion (PII), proprietary identifiers, or confidential
business content is included.
Due to privacy and compliance constraints, the
raw data cannot be publicly released. However,
we release the desensitized benchmark dataset,
along with a reproducible synthetic data genera-
tion pipeline and a representative sample subset,
to enable independent verification and follow-up
research.
Domain Specifications.Based on the aforemen-
tioned data sources, we select six vertical domains
constructed according to three complementary cri-
teria: (1)Industrial Prevalence—domains with sub-
stantial enterprise RAG deployments; (2)Data Ac-
cessibility—availability of production logs and do-
main expertise; and (3)Task Diversity—coverage
of distinct knowledge processing paradigms.
•Energy: Procedural QA with version drift (e.g.,
equipment maintenance protocols across ERP
system transitions).
•Medical: Clinical abstraction from fragmented
dialogues, including discharge summaries and
treatment synthesis.
•Legal: Multi-hop statute-case correlation with
jurisdictional hierarchies.
•Financial: Investment advisory and policy
interpretation tasks, partially leveraging Fin-
LongEval (Jiang et al., 2023).
•Party Building: Regulatory knowledge manage-
ment and policy interpretation in organizational
contexts.•Web Search: Real-time queries emphasizing in-
formation freshness and source credibility.
A.2 Data construction process
Our data construction follows a four-stage pipeline:
(1)Query Collection: We gather 491 authentic
user queries from operational logs across six do-
mains, ensuring diversity in information needs and
complexity levels. (2)Instruction Synthesis: For
each query, we apply the constraint fusion pro-
tocol (Section 3.3.2) to generate complex multi-
dimensional instructions averaging 8 constraints
per sample. (3)Context Preparation: We retrieve
relevant documents using hybrid retrieval (BM25
+ dense retrieval) and apply non-ideal simulation
(Section 3.4) to create noise, knowledge gaps, and
factual conflicts. (4)Human Verification: Instead
of generating reference answers, the constructed
samples undergo a rigorous two-stage expert veri-
fication process. Experts verify Instruction-Query
Consistency and Context Validity (confirming topi-
cal relevance and the presence/absence of necessary
information for non-ideal scenarios), filtering out
low-quality samples to ensure ecological validity.
A.3 Constraints Taxonomy
Table 4 and figure 7 outlines the taxonomy across
three dimensions. Persona Definition establishes
the model’s virtual identity and audience context,
while Output Constraints dictate the structural for-
mat and stylistic boundaries. Crucially, the Knowl-
edge Interaction Protocol enforces strict behavioral
rules for evidence handling, such as citation and
conflict resolution. This dimension moves beyond
simple formatting to ensure the rigorous reliability
required in enterprise environments.
A.4 Domain Statistics
Table 5 shows that the Medical and Financial do-
mains exhibit the highest complexity, with 10.21
and 8.74 average constraints respectively. Legal
tasks feature the highest density of Knowledge Pro-
tocol constraints (3.45). In contrast, Web Search
peaks in Output Constraints (4.73) with lower pro-
tocol requirements (1.80), while Energy shows the
lowest Persona usage (0.85).
Figure 7 illustrates the overall constraint distri-
bution. Output Constraints constitute the majority
(53.1%), followed by Knowledge Interaction Pro-
tocols (29.2%) and Persona Definitions. Within
protocols, Citation and Conflict Handling represent
the most frequent sub-dimensions.
9

Table 4: Detailed taxonomy and definitions of sub-dimension constraints. The table describes the specific require-
ments for Persona Definition, Output Constraints, and Knowledge Interaction Protocols used in the benchmark.
Category Sub-Dimension Description
Persona
DefinitionRole Defines the user’s virtual identity, such as a software engineer, medical expert, or customer
service representative.
Audience Specifies the target readers of generated content, influencing the level of detail, terminology,
and professionalism.
Output
ConstraintsFormat Specifies the output structure, such as Markdown, JSON, or requirements like the number
of sections to include.
Content Defines content requirements including word count limits, prefix/suffix specifications, and
keyword frequency constraints.
Negative Explicitly specifies prohibited elements in the output, testing the model’s fine-grained
control capability.
Ordering Specifies the arrangement of output content, e.g., chronological order, alphabetical sorting,
or frequency-based ranking.
Style Defines the response tone, ranging from rigorous and formal to relaxed and conversational.
Knowledge
Interaction
ProtocolConflict
HandlingDefines how the model should respond when contradictory information exists within the
knowledge base.
Knowledge Gap Requires the model to identify and explicitly report when necessary information is missing
or unavailable.
Uncertainty Expr. Specifies how the model should express uncertainty when information is ambiguous or
evidence is insufficient.
Source Filtering Instructs the model to selectively trust or ignore specific types of information sources.
Citation Requires key conclusions to be accompanied by directly cited original text excerpts as
supporting evidence.
Table 5: Statistics of queries and constraint density. The table shows the number of samples and the average number
of constraints per query across the three orthogonal dimensions for each of the six vertical domains.
Domain #Data #ConstraintsPersona
Def.Output
Const.Knowledge
Protocol
Financial100 8.74 1.94 5.08 1.72
Energy100 7.01 0.85 3.51 2.65
Legal44 8.61 1.59 3.57 3.45
Medical100 10.21 1.46 5.02 3.16
Party Building48 7.79 1.21 4.08 2.50
Web Search99 8.41 1.87 4.73 1.80
All491 8.40 1.50 4.46 2.45
Note:#Data = Number of Data Samples; #Const. = Avg. Constraints per query.
A.5 Non-Ideal Context Composition
Table 6 reports the distribution of non-ideal context
types in the final 983 instances. Knowledge gaps
and factual conflicts are augmented due to sparsity
in naturally occurring logs; we therefore explicitly
report the natural vs. augmented breakdown for
transparency.
B Evaluation Reliability Analysis
B.1 Cross-Judge Consistency
We validate the stability of our evaluation proto-
col by comparing three LLM judges: Kimi-k2-
thinking, Qwen3-235B-Thinking, and GPT-4o. As
shown in Table 7, the raw evaluation scores (upperTable 6: Distribution of non-ideal contexts in Enterpris-
eRAG.
Context Count Ratio Natural Occ. Augment.
Noisy 447 45.5% 447 (100%) –
Knowledge Gaps 227 23.1% 12 (5.3%) 215 (94.7%)
Factual Conflicts 309 31.4% 27 (8.7%) 282 (91.3%)
Note:Natural Occ.= Natural Occurrence;Augment.= Augmentation. Percent-
ages in parentheses indicate the proportion before augmentation. We augmented
underrepresented failure modes to reflect realistic deployment distributions.
section) exhibit minimal variance across judges;
for instance, Loose IAS scores for DeepSeek-V3.1
differ by less than 0.5% between Kimi and Qwen3.
This consistency is rigorously confirmed by sta-
tistical reliability metrics (lower section). We ob-
serve "Good" to "Excellent" reliability across all di-
mensions, with Intraclass Correlation Coefficients
10

Citation
Conflict
Handling
Knowledge GapIdentification
Source
Filtering
Uncertainty
Expression
Content
Format NegativeOrderingStyleAudienceRole
Knowledge
OutputPersonaFigure 7: Hierarchical distribution of constraints in En-
terpriseRAG. The inner ring represents the three primary
categories, while the outer ring details the specific sub-
dimensions.
(ICC) exceeding 0.80 for Loose IAS and approach-
ing 1.0 for robustness metrics. The high Spear-
man’s ρfurther indicates that different judges pre-
serve the same relative model rankings, ensuring
that our reported performance gaps are robust to
the choice of evaluator.
B.2 Synthetic vs. Real-world Data
Performance
Figure 8 and Table 8 contrast model performance
on naturally occurring (n=27) versus synthetic
(n=282) conflicts. Across five models, synthetic
samples show slightly lower Faithfulness ( ∆= -
0.143) and Answer Coverage ( ∆= -0.020) due
to their engineered contradictions. Critically, we
observe statistical equivalence in the core robust-
ness metrics of Conflict Recognition Accuracy and
Strict IAS ( ∆= -0.038, p = .120; ∆= -0.004, p =
.826), with 95% confidence intervals within ±0.2.
This confirms that our synthesis pipeline effectively
replicates the difficulty of real-world scenarios, en-
suring the ecological validity of the augmentation
strategy.
B.3 Human–LLM Judge Alignment Study
To ensure rigorous evaluation, we employed a two-
stage annotation protocol on a stratified sample of
150 instances. First, two experts independently la-
beled the data, achieving robust Inter-Annotator
Agreement (IAA) (avg. κ=0.85, see Table 9),
which validates the clarity of our instruction taxon-omy. Disagreements were adjudicated to establish
a Gold Standard.
Against this baseline, the LLM judge demon-
strates substantial reliability with an average κof
0.77. Conflict Recognition achieves the highest
alignment ( κ=0.93) due to the objective nature
of contradiction detection, while Strict Adherence
(κ=0.68) exhibits minor divergence on borderline
formatting nuances.
C Statistical Analysis Details
C.1 Figure 3
Data and MethodEach point: model-level ag-
gregate over n=309 conflict instances (X: conflict
recognition rate; Y: mean answer coverage). We
computeSpearman’s ρseparately for reasoning-
enhanced ( n=8) and standard ( n=5) models due
to their distinct architectures.
LimitationsSmall sample sizes limit power.
Models within families (e.g., Qwen3) may not be
fully independent; sensitivity analysis yields con-
sistent patterns. Model-level correlations may not
reflect instance-level relationships.
C.2 Figure 5
Method: McNemar’s test (one-sided) for paired
binary outcomes. Each instance evaluated by both
reasoning-enhanced and standard models within
the same family.
Samples: Noisy Retrieval ( n=447 ), Knowledge
Gaps ( n=227 ), Factual Conflicts ( n=309 ). Exact
binomial test when discordant pairs <25 ; other-
wise z-test with continuity correction.
Confidence Intervals: Percentile bootstrap
(10,000 resamples) preserving pairing.
Multiple Testing: Uncorrected p-values for
10 planned comparisons; Bonferroni correction
(α=.005) does not change conclusions.
C.3 Figure 6
We compare accuracy between independent groups
(samples with vs. without the protocol) using stan-
dard methods for comparing two proportions.
For each model, we construct a 2×2 contingency
table and apply:Pearson’s χ2test(with Yates’
correction) when expected counts ≥5,Fisher’s
exact testotherwise.
We report the accuracy difference ∆ = ˆp with−
ˆpwithout with 95% Wald confidence intervals.
Sample sizes: rejection (157 with / 69 without),
conflict recognition (113 / 193). Analysis used
11

Table 7: Consistency and Reliability Analysis of Evaluator LLMs. The upper section compares the raw evaluation
scores of Kimi, Qwen3, and GPT-4o across three metrics. The lower section reports the statistical inter-annotator
agreement metrics, validating the stability of the evaluation protocol. ICC values are reported with 95% confidence
intervals.
Model / MetricLoose IAS Reject Acc Conflict Acc
Kimi Qwen3 GPT-4o Kimi Qwen3 GPT-4o Kimi Qwen3 GPT-4o
Raw Evaluation Scores (%)
Qwen3-235B-Thinking 89.6 89.0 91.9 27.8 27.9 28.8 37.8 39.1 39.1
DeepSeek-V3.1 86.0 86.2 90.4 18.9 19.4 19.4 28.3 28.9 31.0
Gemini-2.5-Pro 84.9 89.1 88.3 33.6 35.8 34.5 37.2 39.4 36.9
GPT-4.1 82.7 83.8 85.8 10.1 9.3 9.3 16.6 16.6 17.6
Inter-Judge Reliability Statistics
ICC (2,k)0.809[0.11–0.99]0.999[0.99–1.00]0.996[0.98–1.00]
Avg. Spearman’sρ0.600 1.000 0.867
Reliability Verdict Good Excellent Excellent
−0.4 −0.2 0.0 0.2 0.4 0.6
Mean Difference (Natural − Synthetic)FaithfulnessAnswer CoverageLoose IASStrict IASConflict Recognition
|Δ|=0.143  ▲ By design†|Δ|=0.020  △ Marginal|Δ|=0.026  △ Minor diff.|Δ|=0.004  ✓ Equivalent|Δ|=0.038  ✓ Equivalentδ=0.2 δ=−0.295% Confidence Intervals for Mean Differences (Natural − Synthetic)
✓ Equivalent (p≥.10) △ Minor diff./Marginal ▲ By design†Synthetic vs. Natural Conflict Data: Core Metric Comparison
Figure 8: Equivalence testing results comparing synthetic and natural conflict data across five evaluation metrics.
Points represent mean differences (Natural - Synthetic) with 95% confidence intervals; dashed lines mark equivalence
bounds (±0.2). Core robustness metrics (Conflict Recognition and Strict IAS) achieve statistical equivalence,
validating the synthetic data generation approach.
SciPy. Of 26 tests, 25 used χ2and 1 used Fisher’s
exact. Uncorrected p-values are reported; Bonfer-
roni correction ( α=.002 ) does not alter substan-
tive conclusions.
C.4 Figure 8
We useTOST equivalence testingto validate
that synthetic conflicts ( n=282 ) replicate natural
conflict difficulty ( n=27 ). Equivalence margin:
δ=±0.2 (20pp, based on RAG benchmark reli-
ability thresholds). Decision rule: 95% CI for
difference (Natural −Synthetic) must fall within
[−0.2,0.2].
Aggregate mean accuracy across 5 models for
each dataset; bootstrap 95% CI (10,000 resamples).LimitationsSmall natural sample ( n=27 );
model-level aggregation; families share architec-
tures.
C.5 Table 8
We usepaired-samples t-testto compare 13 mod-
els’ performance on natural ( n= 27 ) vs. synthetic
(n= 282 ) conflicts, with Cohen’s dfor effect sizes.
LimitationsModels within families may not be
fully independent; unequal natural/synthetic sam-
ple sizes affect precision but not paired comparison
validity.
12

Table 8: Natural vs. Synthetic Data Comparison.n.s.
denotes no significant difference ( p≥0.05 ), indicating
successful replication of difficulty on core metrics.
Metric Natural Synth. Diff.p-val
Core Robustness(Target: Equivalent)
Conflict Recog. 0.360 0.322 -0.038 .120(n.s.)
Strict IAS 0.160 0.156 -0.004 .826(n.s.)
General Quality
Loose IAS 0.793 0.767 -0.026 <.001∗∗∗
Answer Cov. 0.551 0.532 -0.020 .054
Faithfulness 0.827 0.684 -0.143 <.001∗∗∗
Table 9:Human-LLM Judge Alignment Study.Anal-
ysis based on a stratified sample of 150 instances (50
per dimension).Data Quality: Inter-Annotator Agree-
ment (IAA) between two human experts.LLM Judge
Reliability: Primary evaluator (Kimi-k2-thinking) vs.
adjudicated gold standard.
Evaluation DimensionData Quality LLM Judge Reliability
Human IAA Cohen’sκAgreement
(κ) Rate (%)
Strict IAS 0.73 0.68 86
Proper Rej. 0.87 0.71 87
Conflict Recog. 0.96 0.93 91
Overall avg. 0.85 0.77 88
D Additional Experiments
D.1 Fine-grained Analysis
Figure 9 illustrates the performance of five
representative LLMs—including both reasoning-
enhanced and standard instruction-tuned vari-
ants—across the six vertical domains of Enterpris-
eRAG under noisy retrieval conditions. Across all
domains, models generally maintain high Faithful-
ness and Loose IAS , but experience a sharp "or-
chestration collapse" in Strict IAS, confirming that
simultaneously satisfying multiple domain-specific
constraints remains a primary bottleneck. Specifi-
cally, domains with higher complexity and stricter
behavioral requirements, such as Medical and Le-
gal, exhibit lower absolute Strict IAS scores com-
pared to Web Search. This trend aligns with the do-
main statistics in Table 5, which show that Medical
and Legal tasks feature the highest average num-
ber of constraints and the densest concentration
of Knowledge Interaction Protocols. Furthermore,
reasoning-enhanced models (e.g., Qwen3-235B-
Thinking and DeepSeek-R1) consistently outper-
form their counterparts across all metrics and do-
mains, particularly in Strict IAS and Answer Cov-
erage. These results suggest that the difficulty of
instruction adherence is inherently tied to domain-
specific constraint density, and that robust perfor-mance in complex enterprise scenarios is an emer-
gent capability heavily dependent on inference-
time reasoning rather than simple pattern matching.
E Complete Prompts Repository
Note: All prompts and examples presented in this
paper have been translated from the original Chi-
nese for readership clarity. The actual evaluation
was performed using the Chinese versions.
E.1 The Prompts for Data Construction
E.2 The Prompts for Evaluation
Figure 15 presents an example from theLegal
domain of EnterpriseRAG. This case illustrates the
extreme difficulty of satisfying12 simultaneous
constraintswhile processing a noisy context con-
taining20 retrieved documents, many of which
are domain-adjacent (e.g., Litigation Law vs. Re-
consideration Law) but factually insufficient.
13

financial
search
law
medicalindustryparty
0.480.650.82Faithfulness
financial
search
law
medicalindustryparty
0.480.650.82Answer Coverage
financial
search
law
medicalindustryparty
0.480.650.82Loose IAS
financial
search
law
medicalindustryparty
0.120.250.38Strict IAS
Qwen3-235B-A22B-Thinking-2507
Qwen3-235B-A22B-Instruct-2507
DeepSeek-R1-0528
Gemini-2.5-Pro
Claude-Opus-4.5Figure 9: Model performance on noisy retrieval contexts across six vertical domains. Radar charts illustrate the
trade-offs between Faithfulness, Answer Coverage, and Instruction Adherence (Strict/Loose) for representative
models.
14

D.1.1 Initial Atomic Libraries Generation
[System]
You are an expert in constructing the EnterpriseRAG benchmark. Your goal is to generate realistic, complex atomic
instructions for retrieval-augmented generation systems in specific domains.
[Context Input]
Definition of TaskRAG Dimensions:
•P (Persona):Role, Audience.
•C (Constraints):Format, Content, Negative (forbidden), Ordering, Style.
•K (Knowledge Protocol):Conflict Handling, Gap ID, Uncertainty, Source Filtering, Evidence Chaining.
[Task Description]
Domain: <scenario>. Details: <detailed_scenario_description>.
Requirements:
1. Generate 5+ atomic instructions per sub-dimension (except Operational Goal).
2. Assign probability (0.0-1.0) and verifyGeneralization(vs context-specific).
3. DetermineEvaluability(Rule vs LLM) and provideJudge Logic(ifeval/prompt).
4. Refer toInstruct_Follow_Evaluationfor constraint types (word count, bullets, etc.).
[Response Examples]
Output strictly in the following JSON format:
{
"Persona␣&␣Scenario": {
"Role": [
{
"instruction": "Do not include names of medical staff other than the attending physician.",
"probability": 0.4,
"generalization": "no",
"judge_method": "LLM",
"judge_detail": "Prompt: Determine if report contains other names... Format: {judge_format}..."
},
{
"instruction": "Do not use exclamation marks.",
"probability": 0.5,
"generalization": "yes",
"judge_method": "rule",
"judge_detail": { "instruction_id_list": ["forbidden_words"], "kwargs": [{ "forbidden_words": ["!"] }] }
}
]
}
}
Figure 10: The prompt template used for generating EnterpriseRAG Atomic Libraries.
D.1.2 Check Combined Constraints Conflict
Please evaluate whether there are logical conflicts or contradictions within the atomic instruction list below. If conflicts
exist, prioritize retaining content related to[Operational Execution].
Atomic Instruction List:<constraints_list>
Please strictly return the following JSON:
{
"has_conflict": false,
"reason": "Brief explanation",
"unconflict_list": "If conflicts exist and can be resolved by removing specific atomic instructions, provide the list of
indices after removal, e.g., [1, 3, 5]; otherwise, provide the original list of atomic instruction indices."
}
Figure 11: The prompt template used for detecting conflict in combined constraints.
15

D.1.3 Prompt Templates for Instruction Refinement
Prompt 1: Filter Irrelevant Instructions
Based on the given question, please remove atomic instructions that cannot be triggered by the current data:
Original Atomic Instruction List:
<constraints_list>
Question:question
Context:context
Please analyze which atomic instructions cannot be triggered in the current question/context and remove them. Please
return in the following JSON format:
{
"removed_atoms": ["List of indices of removed atomic instructions, e.g., [1, 3]"],
"reason": "Reason for removal"
}
If no atomic instructions need to be removed, please return an empty list.
Augment Constraints Based on the Query
Based on the given question and context, please add specific constraints to the current instruction list:
Current Atomic Instruction List:
<constraints_list>
Question:questionContext:context
Requirements are as follows:
1. Analyze the question and context features; add 1-2 new constraints to existing atomic instructions.
2. Must not repeat or overlap with existing atomic instructions.
3. Consider the following dimensions for new constraints:
•Content & Format:Format/Content/Role/Style/Tone/Length, etc.
•Positive & Negative:Must include/Must not include/Avoid including, etc.
•Knowledge Interaction Protocol:Citation/No Citation/Prioritize Citation/Conflict Handling/Missing Info Han-
dling/Uncertainty Expression.
4.New constraints must be specific and actionable (evaluable via code or LLM) and avoid overly broad or vague
descriptions that make evaluation difficult.
5.New constraints should be relevant to the current question/context and improve answer quality, but description should
not be too detailed (specific only to current context); it should have some generalization.
6. If the current atomic instruction list is sufficiently complete, no constraints need to be added.
7. Please return in the following JSON format:
{
"additional_constraints": [
{
"category": "Additional Constraints",
"dimension": "Additional Constraints",
"instruction": "Constraint instruction text",
"judge_method": "llm",
"judge_detail": "LLM-as-judge prompt for evaluating this constraint"
}
],
"reason": "Reason for adding these constraints"
}
If no constraints need to be added, please return an empty list.
8. Reference Example:
{
"instruction": "Do not include names of medical staff other than the attending physician in the report.",
"probability": 0.4, "generalization": "no", "judge_method": "LLM",
"judge_detail": "Please determine whether the report below only contains the attending physician’s name... Output format
:\n{judge_format}\n\nReport:\n{response}\n\nOriginal medical record:\n{context}"
}
9.judge_detail must be complete and accurate. It must explicitly specify the evaluation model’s output as
judge_format , wherejudge_format is{{"Does it satisfy instruction constraints": "Yes or No",
"Reason": "Provide judgment reason"}} . The evaluation can cite context ,question , andresponse fields.
The instruction andjudge_detailmust maintain logical consistency.
16

D.1.4 Quality Assessment of Complex Instructions
You are an expert in benchmarking <task_name> tasks. Your task is to evaluate the quality of the following complex
instruction, composed of multiple randomly combined atomic instructions, to determine if it is suitable as a valid evaluation
case.
Background:
This instruction is used to evaluate a <task_name> large model oriented towards the domain domain. The model needs to
answer questions based on the given context while strictly adhering to the instructions.
Atomic Instructions to be Evaluated:
<instruction_text>
Please evaluate the above combined instruction based on the following five dimensions and output your analysis
results in JSON format:
1.Logical Consistency: Evaluate whether there are internal conflicts or incoordination among the parts of the instruction
(role, goal, format, constraints). Score (1-5, where 1 is severely inconsistent, 5 is completely consistent).
2.Feasibility: Evaluate whether it is theoretically possible for a top-tier <task_name> model to satisfy all these
requirements simultaneously. Check for absolute contradictions (e.g., requiring a list while prohibiting lists). Score
(1-5, where 1 is completely unexecutable, 5 is completely executable).
3.Realism: Evaluate whether this instruction combination simulates a real, reasonable work scenario likely to occur in a
<task_name> task within the <domain> domain. Score (1-5, where 1 is completely unrealistic, 5 is very realistic).
4.Clarity of Evaluation: Evaluate whether we still have clear, actionable methods (whether via rules or LLM-as-Judge)
to judge if the model followed every instruction after combination. Score (1-5, where 1 is very vague evaluation criteria,
5 is very clear).
5.Appropriate Complexity: Evaluate whether the overall difficulty is too simple, moderate, or too complex to be
practical. Score (1-5, where 1 is too simple/complex making it ineffective, 5 is moderate complexity with good
discrimination).
Output Format:
Please strictly return your evaluation results following the JSON structure below.
{
"instruction_id": "[Assign a unique ID for this instruction combination]",
"evaluation_summary": {
"logical_consistency": {
"score": <Score int>,
"reasoning": "<Your analysis reasoning>"
},
"feasibility": {
"score": <Score int>,
"reasoning": "<Your analysis reasoning; explicitly point out contradictions if any>"
},
"realism": {
"score": <Score int>,
"reasoning": "<Your analysis reasoning>"
},
"evaluation_clarity": {
"score": <Score int>,
"reasoning": "<Your analysis reasoning>"
},
"complexity": {
"score": <Score int>,
"reasoning": "<Your analysis reasoning>"
}
},
"overall_judgment": {
"average_score": <Composite Score float>,
"recommendation": "<’Recommended’ | ’Use with Caution’ | ’Not Recommended’>",
"final_remarks": "<Final summary and modification suggestions for this instruction>"
}
}
Figure 12: The prompt template for evaluating and scoring the refined complex instruction.
17

D.1.5 Document Relevance Assessment
[System Instruction]
# Task Description
Please strictly evaluate whether the retrieved document below effectively supports answering the user’s question. Analyze
the relevance between the document and the question step-by-step and output structured results.
# Evaluation Steps
1.Understand Question Core
•Extract keywords and core requirements of the user question, clarifying the type of information needed for the
answer (e.g., data, reasons, steps, etc.).
2.Document Content Analysis
•Check the document sentence by sentence, marking content directly related to the question (e.g., data, definitions,
causal explanations).
• Identify potentially indirectly supporting information (e.g., background knowledge, analogous cases).
3.Relevance Judgment
•Determine if the document containskey evidenceneeded to answer the question (e.g., "Yes/No", must specify
concretely).
• Check if the information is complete and reliable (e.g., source of data, existence of contradictions).
4.Support Confirmation
•Confirm whether the document content can fully or partially answer the question, rather than being unable to answer
or completely irrelevant to the question content.
# Output Format
{
"supports_question": "Yes/No",
"confidence": "Percentage (0-100%)",
"reasoning": "1-2 sentences explaining the basis, e.g., missing information or completely irrelevant to question content",
"key_evidence_citations": ["Original text fragments from the document"]
}
[User Instruction]
**Data for Analysis**
User Question:0
Retrieved Document:1
Figure 13: The prompt used to verify the relevance of retrieved documents and query.
18

D.1.6 Conflict Management
Conflict Detection
[System]
You are an expert in information consistency and time-sensitivity analysis. Please judge whether conflicts/contradictions
exist in the given context segments: two or more segments provide opposite or mutually exclusive conclusions regarding
the same fact. Please list: 1) Whether the above issue exists (Yes/No); 2) The specific context segments involved in the
conflict, labeled as Index 1 and Index 2 (indices start from 0); 3) Summary of the conflict point; 4) Whether it affects the
answer (Yes/No) and the reason.
[User]
User Question:query
Context:
[0] Context_0
[1] Context_1...
Please output in JSON: {"has_conflict": "Yes/No", "details": [{"index1":[], "index2":[],
"summary":"...", "affects_answer": "Yes/No"}]}
Conflict document synthesis
[System]You are a data construction expert.
[User]Please carefully read and follow the instructions below to complete a conflict context construction task.
# Task Objective
Your task is to act as a data fabrication expert. Based on the user’s "Question" and a series of "Correct Context" segments,
you need to construct numnew, deceptive context segments. These new segments must contain information that directly
conflicts or contradicts specific facts in the "Correct Context".
# Core Requirements
1.Modify Based on Facts: Do not fabricate information out of thin air that is irrelevant to the original context. You must
select one or more key fact points (e.g., numbers, dates, names, conclusions, status) from the "Correct Context" and
modify them to create contradictions.
2.Maintain Context Relevance: The constructed conflict segments must be highly relevant to the original question and
context in terms of topic and phrasing. They should read naturally and credibly, not appearing obviously fake.
3.Explicitly Identify Conflict Points: After construction, clearly indicate which original segment(s) the new segment
conflicts with and concisely summarize the core content of the conflict.
# Input Format
•Question: User’s original question.
•Correct Context: One or more segments, each prefixed with-[Index], e.g.,-[0].
# Output Format
Strictly output in the following JSON format without additional explanation:
{
"generated_contexts": [
{
"source_indices": [0],
"conflict_id": "c0",
"text": "Your first constructed conflict segment here. It should look credible but contain information contradicting
paragraph ‘-[0]‘.",
"conflict_summary": "E.g.: Changed the release year 2023 in the original context to 2022."
},
{
"source_indices": [1, 2],
"conflict_id": "c1",
"text": "Your second constructed conflict segment...",
"conflict_summary": "E.g.: Replaced the main contributor ’John Doe’ with ’Jane Doe’."
}
]
}
Usage Example:
Question:queryCorrect Context:contextOutput:
Figure 14: The two-stage process for conflict management: detection of existing contradictions and generation of
synthetic conflicts to test model robustness.
19

D.2.1 Answer Coverage Evaluation
Prompt 1: Claim Extraction
You are a highly precise information extraction expert. Your task is to analyze a user’s question and a provided context,
then extract all relevant factual claims. You must classify each claim as either[core]or[supplementary].
•[core]: The essential, direct answer to the question.
•[supplementary] : Valuable, additional information like preconditions, exceptions, timelines, or problem-solving
steps.
Follow the output format exactly as shown in the examples. Each claim must be on a new line. If no relevant information
exists, you MUST respond with the single word: "None".
— Example 1 —
Question:How do I reset my password?
Context:To reset your password, click the ’Forgot Password’ link on the login page. You will receive an email with
instructions. Please note that the reset link is only valid for 10 minutes.
Key Information Points:
[core]Users can reset their password by clicking the ’Forgot Password’ link.
[supplementary]After clicking the link, an email with instructions will be sent.
[supplementary]The password reset link is valid for only 10 minutes.
— Example 2 —
Question:What happens if the ’Pay without getting out’ button doesn’t respond in the app?
Context:In the ’Pay without getting out’ section of the Alipay mini-program, you need to swipe up to reveal the fuel
pump selection screen. If that doesn’t work, ensure your network connection is stable. For persistent issues, contact
support at 400-123-4567.
Key Information Points:
[core]The user needs to swipe up on the screen to show the fuel pump selection page.
[supplementary]The user should check if their network connection is stable.
[supplementary]For persistent issues, users can contact support at 400-123-4567.
— End of Examples —
Now, perform the task for the following real data, note that there may be more than one core claim involved.
Question:query
Context:context
Key Information Points:
Prompt 2: Coverage Verification
You are a meticulous verifier. For the given "Generated Answer", determine if each "Key Information Statement" from the
list is semantically covered.
Generated Answer:
answer
Key Information Statements:
claims
Respond ONLY with a JSON object where keys are the exact claim index number(start from 1) and values are a boolean
(true for covered, false for not covered). Example:{"1": true, "2": false}
JSON Output:
20

D.2.2 Prompt Template for instruction Adherence Evaluation
[System Instruction]
You are a fair, objective, and inclusive expert in instruction-following evaluation. Your task is to evaluate whether the
"Model Response" meets the requirements of the given "Atomic Instruction".
In conducting the evaluation, please strictly adhere to the following principles:
1. Substance Over Form:
• Focus on whether the model captured the core intent of the instruction.
• Do not judge as "No" due to minor wording differences, punctuation, or non-core formatting flaws.
• As long as the response achieves the instruction’s goal in logic and content, it is considered satisfied.
2. Avoid Over-Interpretation:
• For negative constraints (e.g., "Do not include..."), judge as "No" only when there is a clear violation.
•For style/persona instructions (e.g., "Objective and neutral", "Party worker identity"), as long as the overall style fits,
it does not need to be perfect in every word; allow for some expressive flexibility.
3. Precondition Check:
•If the instruction contains a conditional clause (e.g., "If the knowledge base contains contradictions..."), but the
condition is not triggered in the context (i.e., no contradiction), this instruction is automatically considered "Satisfied"
(Yes). Do not force the model to fabricate contradictions when none exist.
4. Independence:
•Judge only against the current specific "Atomic Instruction". Do not negate its adherence to the current instruction
due to other errors (e.g., typos, hallucinations) in the response.
5. Judgment Criteria:
•Yes: Completely satisfied, or satisfied in core elements with only minor flaws.
•No: Clearly violated the core requirements of the instruction, or missed critical steps.
Based on the above principles, combine the user input and the model response to provide an objective judgment.
[User Instruction]
judge_detail
(Note: Thejudge_detailvariable contains the specific evaluation prompt, context, question, and response.)
Output Format Requirement:
Please strictly output in the following JSON format:
{
" Does i t s a t i s f y i n s t r u c t i o n c o n s t r a i n t s " : " Yes or No " ,
" Reason " : " P r o v i d e judgment r e a s o n "
}
21

D.2.3 Rejection Accuracy Evaluation
[System Instruction]
You are an expert in evaluating the robustness of Large Language Models in RAG (Retrieval-Augmented Generation)
scenarios. Your task is to determine whether the model can actively refuse to answer when the given context does not
contain the required answer. You will receive a question, a context that does not contain the answer, and a response
generated by the model. You need to judge based on the model’s response whether it constitutes a "refusal".
[User Instruction]
# Task Description
Evaluate whether the following response generated by the language model is a "Refusal". "Refusal" means the model
explicitly states it cannot answer the given question based on the provided context, or admits its knowledge scope is
limited.
Evaluation Criteria:
Valid Refusal (reject_answer: 1):
•Explicitly states that the answer cannot be found in the provided context (e.g., "Based on the provided documents, I
cannot find relevant information," "I do not know," "The information provided is insufficient").
• Points out that the question exceeds the scope of the provided information.
• Any form of response indicating inability to answer the question.
Invalid Refusal (reject_answer: 0):
• Attempts to guess an answer based on context or its own parametric knowledge.
• Generates misleading information (hallucination) related to the question but not a direct answer.
• Answers with affirmative or negative responses ("Yes"/"No") instead of refusing.
• Provides a wrong or fabricated answer despite the missing answer in the context.
# Input Data
Question:question
Context missing the required answer:context
Model generated response:response
# Output Requirements
Please output your evaluation results in JSON format based on the above criteria:
1.reject_answer: A binary indicator. If theModel generated responseis a valid refusal, it is1; otherwise,0.
2.reasoning: A brief explanation for your judgment.
# Examples
Example 1:
Question:"When was Steve Jobs born?"Context:"Apple is a multinational technology company..." (No date
mentioned).
Response:"Based on the context provided, there is no mention of Steve Jobs’ birth date."
Output:
{
"reject_answer": 1,
"reasoning": "The model explicitly states that the answer cannot be found in the provided context, which is a valid
refusal."
}
Example 2:
Question:"What is the capital of France?"Context:"The Eiffel Tower is a famous landmark in Paris, France."
Response:"The capital of France is Paris."
Output:
{
"reject_answer": 0,
"reasoning": "The model answered the question correctly, but the answer did not come from the provided context (though
factual). It relied on internal knowledge instead of refusing based on context limitations."
}
Example 3:
Question:"Detailed description of Mars canals?"Context:"Mars is the fourth planet from the Sun..."
Response:"Mars canals are complex irrigation systems built by ancient Martian civilizations..."
Output:
{
"reject_answer": 0,
"reasoning": "In the absence of relevant information in the context, the model fabricated a detailed description of Mars
canals. This is a typical hallucination, not a refusal."
}
———-Start Evaluation———–
Please output your evaluation result:
22

D.2.4 Conflict Recognition Accuracy Evaluation
[System Instruction]
You are a rigorous and precise AI model evaluator. Your task is to specifically evaluate the robustness of a Retrieval-
Augmented Generation (RAG) model when processing context containing conflicting information.
You will receive three pieces of information: 1.[Knowledge Source]: Context information provided to the RAG model
containing internal contradictions. 2.[User Question]: The question the RAG model needs to answer. 3.[Model
Answer]: The actual answer generated by the RAG model based on the source and question.
Your evaluation criterion is theConflict Recognition Score, quantified asconflict_recog:
conflict_recog = 1 (Successfully Identified Conflict):If and only if the[Model Answer]explicitly points out that
there is a contradiction, conflict, or inconsistency in the[Knowledge Source]. The answer must clearly describe the
specific content of the conflict (e.g., "Source A says X, while Source B says Y").
conflict_recog = 0 (Failed to Identify Conflict):If the[Model Answer]exhibits any of the following, it should be
scored as 0:
1.Completely ignoring the conflict: Synthesizing information as if the conflict does not exist.
2.Blindly trusting one side: Adopting information from only one source while completely omitting the contradictory
source.
3.Fabricating a "rationalization": Hallucinating a new reason to try to reconcile two obviously contradictory pieces
of information.
4.Avoiding the question: Merely stating "insufficient information" or "unable to answer" without explicitly identifying
"conflicting information" as the cause.
Your output must strictly follow the JSON format requested by the user.
[User Instruction]
Please evaluate the performance of the following RAG model based on the role and criteria assigned to you in the system
prompt.
[Knowledge Source]context
[User Question]question
[Model Answer to Evaluate]response
[Evaluation Task]
Please score the above[Model Answer to Evaluate]. Your output must be a single, valid JSON object containing the
following two keys:
•"conflict_recog": (Integer) Give0or1based on the criteria.
•"reasoning" : (String) Explain in detail why you gave this score and point out the specific behavior of the model
answer.
Examples:
Example 1:
[Source][1] "According to the internal memo, the annual financial report conference is scheduled for Oct 28, 2024, at 10
AM." [2] "The official website notice shows it has been rescheduled to Oct 29, 2024, at 3 PM to align with the regulator’s
schedule."
[Question]"When will the company hold its annual financial report conference?"
[Model Answer]"The company’s annual financial report conference will be held on Oct 28, 2024, at 10 AM."
Evaluation Output:
{
"conflict_recog": 0,
"reasoning": "The model failed to identify and report the conflict between knowledge sources, so the score is 0. The
answer ’will be held on Oct 28...’ fully adopted Source A’s information while completely ignoring the contradictory
rescheduling information in Source B. This falls under ’Blindly trusting one side’."
}
Example 2:
[Source](Same as above)[Question](Same as above)
[Model Answer]"There is a conflict regarding the date. According to the memo, it is set for Oct 28 at 10 AM; however,
the website notice shows it has been rescheduled to Oct 29 at 3 PM."
Evaluation Output:
{
"conflict_recog": 1,
"reasoning": "The model successfully identified and reported the conflict, so the score is 1. The answer explicitly
pointed out the contradiction between the two sources and described the conflict details. This meets the ’
Successfully Identified Conflict’ criterion."
}
23

E Data Case
Complex Instruction Set (12 Constraints)
[Role & Scene]
1. You are a rigorous legal researcher writing an academic analysis report.
2. The answer will serve as a draft for legal documents; ensure precise and formal language.
[Operational Goal]
3. Answer the user’s question basedonlyon the provided query and legal knowledge base.
[Format & Content Constraints]
4. Use a numbered list for the detailed response.
5.Citation Requirement:You must cite at least2 differentlaws/regulations.
6.Negative Constraint:Do not use any emojis.
7.Structure:State theCore Conclusionfirst, then expand on details point-by-point.
8.Style:Maintain a rigorous, objective, and neutral legal professional style.
[Knowledge Interaction Protocols]
9.Conflict Handling:If KB information conflicts, do not judge correctness; present both and cite sources.
10. Gap Identification:If the KB cannot provide direct information to answer the question, explicitly state: "Based on existing materials, I cannot
directly answer your question."
11.Uncertainty:Do not use overly affirmative words like "definitely", "must", "inevitably".
12.Citation Format:Must cite the full statute name and specific article number (e.g., "Civil Code of the PRC, Article 188").
Retrieved Context (Noisy & Incomplete: 20 Docs)
Query:"I stole items worth <500 RMB. First offense. Detained for 5 days. Can the penalty be lightened via Administrative Reconsideration?"
[1] Procedural Provisions for Public Security Organs (PPSOA), Art. 222:Discusses applying forsuspensionof detention during review.
[2] PPSOA, Art. 225:Fines are not suspended during detention suspension.
[3] PPSOA, Art. 226:Regulations for offenders during detention suspension (must not leave city, etc.).
[4] PPSOA, Art. 175:Penalty decisions must state facts, evidence, andlighter/mitigating circumstances.
[5] PPSOA, Art. 218:Measures for failure to pay fines (auctioning seized property).
[6] Administrative Litigation Law, Art. 59:Courts can fine/detain for hindering litigation. (Irrelevant: Judicial procedure).
[7] PPSOA, Art. 195:Thefts under 500 RMB can be handled by local police stations via confiscation.
[8] PPSOA, Art. 232:Refund of security deposit upon revocation of detention.
[9] Administrative Litigation Law, Art. 77: People’s Courtcan change a penalty if it is "obviously unfair."(Note: Applies to Litigation, not Reconsideration).
[10] Civil Procedure Law, Art. 119:Detention/fines must be approved by the Court President.
[11] PPSOA, Art. 223:Criteria for granting suspension of detention (guarantor or deposit).
[12] PPSOA, Art. 229:Obligations of the guarantor.
[13] PPSOA, Art. 224:Conditions for denying suspension (risk of escape).
[14] PPSOA, Art. 163:Time in custody offsets administrative detention time.
[15] Criminal Law, Art. 37:Exemption from criminal punishment for minor crimes. (Irrelevant: Administrative case).
[16] PPSOA, Art. 217:Installment payments for fines in case of economic hardship.
[17] PPSOA, Art. 173:Detention must be decided by county-level police organs, not local stations.
[18] Criminal Procedure Law, Art. 199:Disrupting court order.
[19] Mine Safety Law, Art. 45:Reconsideration for mining penalties. (Irrelevant: Domain mismatch).
[20] Civil Procedure Law, Art. 118:Fines for individuals/units.
Standard Model Response (Qwen3-235B-A22B-Instruct-2507)
Core Conclusion:Based on current legal norms... applying for administrative reconsiderationdoes not necessarilylead to a lighter penalty, but you can
request a change.[Fail: Should have Rejected]
1. **Administrative Reconsideration can serve as a basis for changing penalties** According toAdministrative Litigation Law, Article 77[Fail:
Misapplied Law]: "If an administrative penalty is obviously improper... the People’s Court may judge to change it." Although this applies to litigation, its
spirit is reflected in the reconsideration system...[Fail: Inference forbidden]
2. **Discretion based on Mitigating Circumstances**Procedural Provisions for Public Security Organs, Article 175states that penalty decisions must
record "mitigating circumstances." Since the value is under 500 RMB...
Strict Adherence Analysis: FAILED.
•Format:Passed (Numbered list, Conclusion first).
•Citation:Passed (Cited >2 laws).
•Protocol (Gap ID): FAILED. The KB lacks theAdministrative Reconsideration Law. The model forced an answer by misapplying theLitigation Law
(Court) to aReconsideration(Police/Gov) question.
•Protocol (Uncertainty):Passed (Used "does not necessarily").
Reasoning Model Response (Qwen3-235B-A22B-Thinking-2507)
[Internal Thought Process: User asks about "Administrative Reconsideration" reducing a penalty. I need to check the KB. - Doc 1-5, 7-8, 11-14, 16-17:
PPSOA (Procedural rules for police). - Doc 6, 9: Administrative Litigation Law (Courts, not Reconsideration). - Doc 19: Mine Safety Law (Irrelevant).
- Missing: "Administrative Reconsideration Law". Protocol Constraint #2 says if KB cannot directly answer, I must state a specific phrase. Doc 175
mentions "mitigating circumstances" but refers to the initial decision, not the Review process. Doc 77 allows change but it is for Courts. Conclusion: KB
is insufficient to answer "Can it be lightened via Reconsideration?".]
Based on existing materials, I cannot directly answer your question.
Strict Adherence Analysis: PASSED.
•The model prioritized theKnowledge Gap Protocolover the user’s desire for an answer, correctly identifying that the specific law governing
Reconsideration outcomes was missing from the noisy 20-document context.
Figure 15: Detailed Selective Adherence Failure. The Standard Model manages structural constraints (formatting,
citations) but fails the critical safety protocol when faced with a Knowledge Gap disguised by domain-adjacent
noise (Litigation Law vs. Reconsideration Law). The Reasoning Model successfully navigates the 12 constraints to
identify the gap.24