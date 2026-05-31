# FAB-Bench: A Framework for Adaptive RAG Benchmarking in Semiconductor Manufacturing

**Authors**: Jingbin Qian, Congwen Yi, Min Xia, Wen Wu, Jun Zhu, Jian Guan

**Published**: 2026-05-26 02:32:51

**PDF URL**: [https://arxiv.org/pdf/2605.26476v1](https://arxiv.org/pdf/2605.26476v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become critical for knowledge-intensive applications, yet evaluating its performance in vertical domains remains difficult due to domain complexity, diverse context scales, and heavy reliance on expert assessments that are costly, inconsistent, and non-scalable. We introduce FAB-Bench, an end-to-end framework for adaptive benchmarking of RAG systems in semiconductor manufacturing. FAB-Bench defines six diagnostic metrics measuring factual accuracy, contextual utilization, completeness, retrieval relevance, technical depth, and reasoning consistency. The framework couples retriever diagnostics with generator-level reasoning analysis across context windows of 4K-32K tokens, quantifying how retrieval precision and generative fidelity co-evolve as contextual scope expands. From over 1,300 generated candidates, we curated a high-quality benchmark of 200 query-answer pairs spanning three synthesis strategies: needle-in-haystack, intra-document multi-topic, and cross-document multi-hop. Systematic evaluation across four LLMs and four RAG frameworks reveals three distinct context-scaling behaviors: logarithmic growth, early saturation, and cold-start dynamics, and identifies attention dilution as the primary mechanism behind performance degradation at extreme context lengths. Cross-framework validation on three additional production RAG systems confirms evaluation portability.

## Full Text


<!-- PDF content starts -->

FAB-BENCH: A FRAMEWORK FORADAPTIVERAG
BENCHMARKING INSEMICONDUCTORMANUFACTURING
A PREPRINT
Jingbin Qian Congwen Yi Min Xia Wen Wu Jun Zhu Jian Guan*
FutureFab.AI
*andrewg@futurefab.ai
ABSTRACT
Retrieval-Augmented Generation (RAG) has become critical for knowledge-intensive applications,
yet evaluating its performance in vertical domains remains difficult due to domain complexity, diverse
context scales, and heavy reliance on expert assessments that are costly, inconsistent, and non-scalable.
We introduceFAB-Bench, an end-to-endframework foradaptivebenchmarking of RAG systems in
semiconductor manufacturing. FAB-Bench defines six diagnostic metrics measuring factual accuracy,
contextual utilization, completeness, retrieval relevance, technical depth, and reasoning consistency.
The framework couples retriever diagnostics with generator-level reasoning analysis across context
windows of 4K–32K tokens, quantifying how retrieval precision and generative fidelity co-evolve
as contextual scope expands. From over 1,300 generated candidates, we curated a high-quality
benchmark of 200 query–answer pairs spanning three synthesis strategies: needle-in-haystack, intra-
document multi-topic, and cross-document multi-hop. Systematic evaluation across four LLMs and
four RAG frameworks reveals three distinct context-scaling behaviors—logarithmic growth, early
saturation, and cold-start dynamics—and identifies attention dilution as the primary mechanism
behind performance degradation at extreme context lengths. Cross-framework validation on three
additional production RAG systems confirms evaluation portability.
KeywordsRAG Evaluation ·Vertical Domain Benchmark ·LLM-as-Judge ·Context Window Scaling ·Semiconductor
Manufacturing
1 Introduction
Large language models (LLMs) have demonstrated remarkable capabilities in various tasks [Achiam et al., 2023, Brown
et al., 2020], motivating the development of diverse evaluation benchmarks. Early benchmarks such as GLUE [Wang
et al., 2018] and SuperGLUE [Wang et al., 2019] focused on natural language understanding tasks, including sentiment
analysis, textual understanding, and question answering.MMLU[Hendrycks et al., 2021a] expands assessment
to broad knowledge coverage across 57 subjects using 15,908 multiple-choice questions spanning from elementary
to professional levels, with an emphasis on zero-shot and few-shot evaluation of pre-trained knowledge. However,
rapid benchmark saturation has significantly reduced its discriminative power: while GPT-3 achieved only 43.9%
accuracy [Brown et al., 2020], GPT-4 exceeded 86% [OpenAI, 2023]. To improve linguistic coverage,C-Eval[Huang
et al., 2023] extends this paradigm to Chinese with 13,948 questions across 52 subjects. Despite their value, these
benchmarks share fundamental limitations: reliance on staticpublicknowledge and the absence of retrieval contexts,
making them poorly suited for evaluating RAG systems in vertical domains.
Beyond generic language benchmarks, more task- or capability-specific evaluations have been proposed.ARC[Clark
et al., 2018] evaluates scientific reasoning using graduate-school–level questions;TruthfulQA[Lin et al., 2022]
measures propensity for factual reliability;GSM8K[Cobbe et al., 2021] andMATH[Hendrycks et al., 2021b] assess
Benchmark dataset available at:https://github.com/FuturefabAI/FAB-BencharXiv:2605.26476v1  [cs.CL]  26 May 2026

FAB-Bench A PREPRINT
mathematical reasoning from graduate-school to competition-level; andHumanEval[Chen et al., 2021] evaluates code
generation through 164 programming tasks.
Domain-specific benchmarks further address specialized requirements. In healthcare,MedQA[Jin et al., 2021],
MedMCQA[Pal et al., 2022], andMultiMedQA[Singhal et al., 2023a] evaluate medical reasoning under safety-
critical constraints, with Med-PaLM 2 achieving 85% accuracy on USMLE-style questions [Singhal et al., 2023b].
Legal benchmarks such asLegalBench[Guha et al., 2023] andLawBench[Fei et al., 2023] assess regulatory reasoning
across jurisdictions, while financial benchmarks includingFinanceBench[Islam et al., 2023],FinBen[Xie et al.,
2024], andBloombergGPT[Wu et al., 2023] evaluate financial analysis and decision-making. In semiconductor
design, ChipNeMo [Liu et al., 2023a] adapts LLMs for chip design but focuses on model training rather than RAG
evaluation. Although these domain benchmarks improve specialization, they largely rely on public datasets and expert
curation, and provide limited visibility into how effectively RAG systems retrieve, integrate, and reason over proprietary,
multi-document corpora.
RAG has become the dominant paradigm for deploying LLMs in knowledge-intensive applications, particularly in
enterprise and industrial settings [Lewis et al., 2020]—where models must reason and answer questions over proprietary
documents not seen during training. Despite the progress of domain-specific benchmarks, many remain poorly suited
for enterprise RAG use cases: they typically evaluate overpublicknowledge sources (e.g., medical licensing exams,
legal case law, published financial reports) rather than the proprietary documentation enterprises actually deploy [Chen
et al., 2024a]. Moreover, heavy dependence on manual expert curation fundamentally limits scalability [Zheng et al.,
2024]. Existing evaluation metrics are often accuracy-based, offering little diagnostic insight into whether failures
arise from incomplete retrieval, faulty reasoning, or inadequate multi-document synthesis. They also lack mechanisms
for systematic deployment optimization: they cannot evaluate RAG effectiveness on proprietary corpora, quantify
knowledge augmentation value, or guide configuration decisions such as context window allocation. Thus, these
benchmarks emphasize knowledge recall or task-specific reasoning while providing limited assessment of workflow-
level integration or nuanced judgment in real-world settings [Budler et al., 2025, Gao et al., 2023]. While recent work
has begun to address RAG-specific evaluation, important gaps remain: RGB [Chen et al., 2024b] and RECALL [Liu
et al., 2023b] focus on general-domain QA rather than specialized knowledge, ARES [Saad-Falcon et al., 2024] requires
substantial human calibration, and RAGAS [Es et al., 2024] lacks domain-specific customization. As a result, enterprises
still lack quantitative guidance for deployment-critical decisions such as model selection and context window allocation,
relying instead on ad-hoc qualitative feedback.
1.1 Contributions
In this work, we introduceFAB-Bench, an end-to-end evaluation methodology for vertical-domain RAG for realistic
enterprise reasoning. Our main contributions are summarized as follows:
•Methodology for evaluating vertical-domain RAG via cross-document synthesis under adaptive bench-
mark generation.We formulate vertical-domain RAG evaluation as evidence-based synthesis over long
and heterogeneous private corpora, and design benchmarks that require explicit multi-document integration,
including needle-in-haystack grounding, intra-document multi-topic reasoning, and cross-document multi-hop
composition. To improve robustness and coverage of synthesized queries, we employ an adaptive genera-
tion mechanism with temperature modification, adjusting sampling temperature in response to quality and
consistency signals to obtain diverse and stable benchmark instances.
•A diagnostic measurement protocol that attributes failures across retrieval and generation.We introduce a
six-dimensional evaluation rubric—Completeness, Factuality, Context Utilization, Technical Depth, Relevance,
and Support Quality—that separates missing evidence, irrelevant retrieval, shallow synthesis, and unsupported
generation, enabling fine-grained localization of performance bottlenecks.
•An empirical characterization of context-window scaling regimes for vertical RAG.By measuring perfor-
mance from 4K to 32K tokens across four LLMs, we identify three distinct scaling behaviors and characterize
attention dilution through metric-level decomposition, offering actionable guidance for configuration decisions.
2 Related Work
2.1 LLM Evaluation Benchmarks
General-purpose benchmarks have evolved from task-specific assessments like GLUE [Wang et al., 2018] and Super-
GLUE [Wang et al., 2019] to broad knowledge evaluations. MMLU [Hendrycks et al., 2021a] covers 57 subjects with
15,908 questions, though rapid saturation (GPT-3: 43.9% [Brown et al., 2020] to GPT-4: 86%+ [OpenAI, 2023]) reduces

FAB-Bench A PREPRINT
its discriminative power. C-Eval [Huang et al., 2023] extends coverage to Chinese. Capability-specific benchmarks
target scientific reasoning (ARC [Clark et al., 2018]), factual reliability (TruthfulQA [Lin et al., 2022]), mathematical
reasoning (GSM8K [Cobbe et al., 2021], MATH [Hendrycks et al., 2021b]), and code generation (HumanEval [Chen
et al., 2021]). These benchmarks share a fundamental limitation: reliance on static, public knowledge without retrieval
contexts, making them poorly suited for RAG evaluation.
2.2 Domain-Specific Evaluation
Domain benchmarks address specialized requirements but inherit similar limitations. In healthcare, MedQA [Jin et al.,
2021], MedMCQA [Pal et al., 2022], and MultiMedQA [Singhal et al., 2023a] evaluate medical reasoning, with Med-
PaLM 2 reaching 85% on USMLE-style questions [Singhal et al., 2023b]. Legal benchmarks (LegalBench [Guha et al.,
2023], LawBench [Fei et al., 2023]) assess regulatory reasoning, while financial benchmarks (FinanceBench [Islam
et al., 2023], FinBen [Xie et al., 2024], BloombergGPT [Wu et al., 2023]) evaluate financial analysis. In semiconductor
design, ChipNeMo [Liu et al., 2023a] adapts LLMs for chip design but focuses on model training rather than RAG
evaluation. These benchmarks primarily evaluate parametric knowledge over public datasets, providing limited visibility
into how RAG systems retrieve, integrate, and reason over proprietary multi-document corpora [Budler et al., 2025].
2.3 RAG Evaluation Frameworks
RAG-specific evaluation has received increasing attention. RAGAS [Es et al., 2024] provides multi-dimensional
metrics (faithfulness, relevance, context precision/recall) and supports test generation from user-provided corpora,
but does not address context-window scaling or domain-specific metric customization. ARES [Saad-Falcon et al.,
2024] automates RAG evaluation through prediction-powered inference with multi-dimensional scoring, but requires
∼150 human-annotated samples for calibration. RGB [Chen et al., 2024b] evaluates four RAG robustness abilities
including information integration across documents, but uses a fixed general-domain dataset. RECALL [Liu et al.,
2023b] evaluates robustness against counterfactual knowledge. CRAG [Yang et al., 2024] provides a comprehensive
benchmark with multi-hop and aggregation questions requiring cross-document synthesis, but operates on a fixed dataset
without vertical-domain customization. MultiHop-RAG [Tang and Yi, 2024] specifically targets multi-hop reasoning
with evidence distributed across 2–4 documents, but is limited to a fixed English news corpus. SCARF [Rengo et al.,
2025] proposes a system-level assessment framework but does not include benchmark generation.
FAB-Bench complements these efforts by addressing two gaps that none of the above frameworks cover simultaneously:
(1)systematic context-window scaling analysisthat characterizes how RAG performance evolves from 4K to 32K
tokens, and (2)domain-specific evaluation with a structured knowledge base(431 semiconductor terms across 7
weighted categories) that enables precision-aware benchmark generation and domain-grounded scoring.
2.4 LLM-as-Judge Methodology
Using LLMs as evaluation judges has become widespread following MT-Bench [Zheng et al., 2024], which demonstrated
strong correlation between LLM judgments and human preferences. G-Eval [Liu et al., 2023c] formalizes this through
chain-of-thought prompting with probability-weighted scoring. However, LLM judges exhibit known biases including
position bias, verbosity bias, and self-enhancement bias [Zheng et al., 2024, Wang et al., 2023]. Recent work on
calibrating LLM judges [Liu et al., 2024a] suggests that structured rubrics with explicit scoring criteria mitigate these
biases. Our approach addresses reliability through: (1) structured rubrics with separate objective/subjective variants per
metric; (2) chain-of-thought reasoning via G-Eval; and (3) empirical validation of metric independence (Section 5.5).
3 FAB-Bench Framework
Evaluating RAG systems in vertical domains requires benchmarks satisfying four criteria:authenticity(questions re-
flecting real-world complexity),contamination resistance(avoiding train-test overlap),diagnostic granularity(isolating
specific failure modes), anddiscriminative power(meaningfully differentiating models). FAB-Bench addresses these
through two design principles: evaluation mirrors deployment by generating questions from user-provided corpora, and
multi-dimensional assessment enables precise failure attribution (Figure 1).
3.1 Knowledge Corpus and Domain Encoding
Our benchmark draws from three source types: academic literature comprising 150+ peer-reviewed papers from IEDM,
ISSCC, and VLSI symposia; patent documents with 70+ filings containing proprietary fabrication details; and industry
standards from SEMI specifications. The corpus totals approximately 347 million tokens across 188 topics.

FAB-Bench A PREPRINT
Research Papers
150+ documents
Patents
70+ documentsBenchmark Generation
Context Synthesis
Diversity-optimized chunks
Adaptive Temperature
τ = τ_prog + Δτ_content + Δτ_fail
LLM Generation
Enhanced promptsQuality Control
• Format validation
• Uniqueness check
• Diversity score
Statistics
• Technical density
• Category dist.
• MetadataQA Benchmark
1500 QA  pairs
• Question
• Gold answer
• Gold context
• Test type
• Precision level
• Technical category
200 curated
Multi-Dimensional Evaluation
RAG Frameworks
AnythingLLM
MaxKB
RAGFlowMetaso
Mode:
With/Without KBRAG Response
• Actual answer
• Retrieved sources
• Confidence
• MetadataEvaluatorGPT
6 Metrics: Completeness | Factuality
Context Util. | Tech Depth | Relevance | Support
G-Eval: GPT  + CoT  + Token Prob.
Scoring: [0,1] normalized per dimensionEvaluation Report
• Per-question scores (6 dims)
• Aggregate statistics
• Dimensional breakdown
• Context scaling curves
• Cross-model comparison
• Framework ranking
• Diagnostic insightsRetry on failureText Extraction
Image Recognition and Processing
Table Parsing
CategorizationDocument Parsing
Gold context
+ answerIndustrial Guidelines
Figure 1: FAB-Bench system overview. The framework comprises two components: an adaptive benchmark generation
system that produces domain-specific QA pairs from proprietary corpora, and a multi-dimensional evaluation platform
that scores RAG system responses across six diagnostic metrics.
To enable domain-aware processing, we constructed a hierarchical knowledge base Kencoding 431 technical terms
organized into seven semantic categories (Table 1), each with precision weights wireflecting technical rigor requirements.
The knowledge base serves three functions: (1) computing technical density ρ(d) for adaptive generation control,
(2) classifying document precision levels for temperature scheduling, and (3) weighting domain terminology in similarity
computations.
Table 1: Hierarchical knowledge base: seven semantic categories with term counts and precision weights. Higher
weights indicate categories requiring greater quantitative precision.
Category Example Terms Terms Weight
Performance ParametersV th, leakage current,f T, DIBL 80 1.7
Process Nodes EUV , FinFET, HKMG, SADP, TSV 65 1.6
Manufacturing Processes ALD, CMP, RIE, PECVD, epitaxy 60 1.5
Device Physics GAAFET, MRAM, nanosheet, CFET 62 1.4
Materials Science HfO 2, low-k, SiGe, ruthenium 51 1.3
Testing Methodologies NBTI, TDDB, CD-SEM, electromigration 51 1.2
Applications AI chip, NPU, ADAS, 5G transceiver 62 1.1
Total 431
Questions span two balanced formats. Objective questions (50%) assess factual accuracy through mathematical
calculations, fill-in-blank, true/false, and multiple choice items. Subjective questions (50%) evaluate reasoning through
mechanism explanations, causal reasoning, comparative analysis, and problem diagnosis. Both formats require
context-dependent answers that cannot be resolved from parametric knowledge alone.
3.2 Cross-Document Context Synthesis
A defining RAG capability is cross-document synthesis—integrating information across multiple sources. Single-
document benchmarks cannot distinguish retrieval-based reasoning from parametric knowledge reliance. We design
three synthesis strategies targeting complementary objectives.
The corpus is segmented into chunks C=Sm
i=1Chunk(D i)via sliding window (512 tokens, 128-token overlap).
TF-IDF vectors⃗ v(c)enable semantic comparison between chunks.
Strategy 1: Needle-in-Haystack.Critical information is embedded within topically dissimilar distractors. Given a tar-
get chunk ctarget, we select distractor chunks {cd}that minimize cosine similarity: cd= arg min c∈Ccos(⃗ v(c target),⃗ v(c)) .
This tests precise fact location amid irrelevant content.

FAB-Bench A PREPRINT
Strategy 2: Intra-Document Multi-Topic.Chunks from different topic clusters within the same document are
combined, requiring integration of dispersed information. We cluster chunks within each document using TF-IDF
cosine similarity and select chunks from at least two distinct clusters (cos(⃗ v(c i),⃗ v(c j))<0.3for cluster separation).
Strategy 3: Cross-Document Multi-Hop.The most challenging strategy constructs contexts requiring cross-source
reasoning:
1. Select a seed chunkc seedfrom documentD i.
2. Identify a semantically related chunk from a different document:c link= arg max c∈D j,j̸=icos(⃗ v(c seed),⃗ v(c)).
3. Validate connection strength:cos(⃗ v(c seed),⃗ v(c link))> θ link, whereθ link= 0.1.
4. Construct the final context by combining seed, link, and distractor chunks.
Questions are generatedaftercontext construction, guaranteeing that correct answers depend on synthesizing linked
information from multiple documents.
3.3 Adaptive Generation Control
Generating high-quality test cases requires balancing precision and diversity. We adapt generation parameters to content
characteristics through two mechanisms.
Technical Density and Precision Classification.We compute technical density ρ(d) as the ratio of domain term
occurrences (weighted by category) to total words. Documents are classified into precision levels:
p(d) =

high ifρ(d)>0.20orω h(d)>8
medium ifρ(d)>0.12orω h(d)>4
low otherwise(1)
whereω h(d) =P
ci∈{parameters, processes}wi· |{t∈T ci:n(t, d)>0}|counts high-weight category term occurrences.
Adaptive Temperature.Temperatureτis computed as:
τ=clip(τ min(p) +τ prog(k) + ∆τ c(c∗) + ∆τ fail(a, s),0.1,1.0)(2)
where base ranges [τmin, τmax]are[0.4,0.8] for high-precision, [0.5,0.9] for medium, and [0.6,1.0] for low; τprog(k) =
(τmax−τmin)×min(k/20,0.8) increases diversity as successful generations kaccumulate; ∆τcadjusts for dominant
content category ( −0.10 for parameters, +0.05 for applications); and ∆τfailboosts temperature after generation failures.
These mechanisms follow established temperature-diversity trade-offs [Holtzman et al., 2020, Renze, 2024].
Adaptive Similarity Threshold.To prevent duplicate generation while allowing necessary terminology overlap in
technical content, the similarity threshold relaxes progressively:
θsim(r) = max (0.50, θ base(p)−0.05×r)(3)
where θbaseis 0.70 for high-precision, 0.75 for medium, and 0.80 for low-precision content, and rcounts retry attempts.
Similarity combines weighted Jaccard overlap of all tokens and domain terms with TF-IDF cosine similarity for corpora
exceeding five questions.
Generation Pipeline.Algorithm 1 integrates these components into a fully automated pipeline (Figure 2).

FAB-Bench A PREPRINT
Load Original data
json format
IntelligentTokenManager
Calculate context window
Set rate limit (TPM/RPM)
DiversityOptimizedProcessor
Analyze Content Distribution
Calculate Optimized Chunk
analyze_technical_content
Calculate technical density
Recognize main category
Conﬁrm precision level (high/medium/low)
Call OpenAI API
Rate Limit Check
Token Usage Tracking
Price Tracking
Save Test Cases (QA Pairs)
Allocate test_id
Add metadataSelect Context
Poll Diverse Context
Update Tracker
Add to generated QA sets
Update progress
Record Cost
Generation Statistic Analysis
Save Benchmark Summary Diversity Score
Technical Density DistributionInitialization Step
Preprocessing Step
Generation Step
Postprocessing StepSemiconductorEnhancedUniquenessTrackerInitialize OpenAI Client
Set Model, API KeysLoad Semi Knowledge Base
100+ Professional Words
SemiconductorAdaptiveTemperatureController
diversity_mode -> global_temp_oﬀset
Adjust precision level's temperature range
Create Diverse Context
According to test types
robustness: needle_haystack
multi_hop: multi_topic
generation: analytical
Calculate Adaptive Temperature
Based on precision level, progress, retry 
times, similarity failureGenerate Enhanced Prompts
Load Subjective/ Objective Template
+ Question Archetype
+ Diversity Requirements
+ Sub/Objective force balanced
Extract JSON Results
Clean Dirty Characters
Structured Output
Uniqueness/Similarity Check
Basic Words Overlap
Technical Words Weighting
Semantic CheckPass?
No
Retry 5 timesNo
After 5 times retry
YesFailed to extractPrevents duplicate questions
Weighs technical terms heavily
Balances question formats
Manages test IDsCore System Components
Load Prompt Conﬁg for 
Diﬀerent Test Types
Loop QA Generation for 
Each Scenarios
Loop
Reached Required QA PairsCostTracker
Track tokens used
Calculate $ costs
Monitor spending
Objective Format Validation
Check proper preﬁxes
Check QA Formatobjective
Pass? Yes
Figure 2: Benchmark generation workflow. The pipeline iterates through synthesized contexts, adaptively adjusting
generation parameters based on precision classification and failure feedback.
Algorithm 1Adaptive QA Pair Generation
Require:CorpusD, test typet, target countN
Ensure:QA pairsQ={(q i, di, ai)}N
i=1
1:C←CHUNK(D); compute TF-IDF vectors⃗ v(c)for allc∈C
2:D ←SYNTHESIZECONTEXTS(C, t)// Strategy 1, 2, or 3
3:Q ← ∅
4:foreach contextd∈ Dwhile|Q|< Ndo
5: Classify precisionp(d)via Eq. 1; compute densityρ(d)
6:τ←ADAPTIVETEMPERATURE(p,|Q|,failures)// Eq. 2
7:θ←ADAPTIVETHRESHOLD(p,retries)// Eq. 3
8:forretryr= 1toR maxdo
9:(q, a)←LLMGENERATE(d, t, τ)
10:ifVALID(q, a)andUNIQUE(q,Q, θ)then
11:Q ← Q ∪ {(q, d, a)};break
12:end if
13:end for
14:end for
15:returnQ
Expert Validation.A controlled ablation study validated the generation mechanism. Domain experts rated QA pairs
(18 per condition, spanning ROB/MULTI/GEN types equally) on four dimensions (1–5 scale): accuracy, relevance,
difficulty, and diversity. The full system combining adaptive parameters with enhanced prompts achieved the highest
scores across all dimensions (Accuracy: 5.00, Relevance: 5.00, Difficulty: 4.60, Diversity: 4.53) with zero retries, while
adaptive parameters alone yielded mixed results (Accuracy: 4.33, Difficulty: 3.28), indicating that prompt engineering
is essential for quality generation (Appendix D).

FAB-Bench A PREPRINT
Per-question-type analysis reveals an appropriate difficulty gradient in the generated benchmark (Table 2), confirming
that the three synthesis strategies produce questions of distinct and meaningful complexity.
Table 2: Expert-rated quality by question type (averaged across all ablation conditions). The difficulty gradient (ROB <
MULTI<GEN) confirms the three synthesis strategies produce appropriately graded complexity.
Question Type Accuracy Relevance Difficulty Diversity
Robustness (needle-in-haystack) 4.58 4.62 2.38 2.92
Multi-Hop (cross-document) 4.58 4.83 4.12 3.33
Generation (deep reasoning) 4.71 4.75 4.79 3.88
3.4 Six-Dimensional Evaluation Metrics
We define six diagnostic metrics, each evaluated on a 10-point scale with separate rubrics for objective and subjective
questions (Table 3). Full rubric definitions are provided in Appendix E.
Table 3: Six-dimensional evaluation metrics with failure mode attribution.
Metric Evaluates Failure Attribution
Completeness Coverage of required elements Missing retrieval or shallow generation
Technical Depth Sophistication of analysis Generation: insufficient reasoning
Factuality Alignment with sources Generation: hallucination
Relevance Focus on query topic Both: off-topic retrieval or generation
Context Utilization Use of retrieved context Retrieval failure or context neglect
Support Quality Citation specificity/accuracy Generation: unsupported claims
These metrics are designed to capture orthogonal failure modes. A response may exhibit high relevance but low
factuality (on-topic hallucination), or strong technical depth but poor context utilization (ignoring retrieved documents
in favor of parametric knowledge). This design enables diagnostic attribution: retrieval failures manifest as low context
utilization despite high factuality, while generation failures appear as uniformly low scores.
We empirically assess metric independence in Section 5.5 through correlation analysis across all experimental configu-
rations.
3.5 Evaluation Platform Architecture
Our platform implements a three-layer architecture inspired by SCARF [Rengo et al., 2025]:
•Orchestration Layer:Manages test distribution, parallel execution across models and context configurations, and
result aggregation.
•Adapter Layer:Normalizes heterogeneous RAG framework interfaces (AnythingLLM, RAGFlow, MaxKB, Metaso)
into unified query-response protocols, enabling fair cross-framework comparison. For frameworks without source
exposure (e.g., MaxKB), the adapter employs heuristic context detection using domain-specific indicators.
•Evaluation Layer:Applies the six-dimensional metrics via G-Eval [Liu et al., 2023c] through DeepEval [Confident
AI, 2025]. G-Eval employs LLMs as judges with chain-of-thought reasoning: each metric is defined through
evaluation objectives, reasoning procedures, and scoring rubrics. The evaluator extracts token-level probabilities over
score candidates and computes normalized weighted scores.
The platform supports two complementary evaluation paradigms:cross-model comparison(multiple LLMs within a
fixed RAG framework, isolating model capabilities) andcross-framework comparison(multiple RAG architectures with
a fixed model, isolating retrieval and system design choices).

FAB-Bench A PREPRINT
4 Experiments: Context Window Scaling Analysis
4 12 20 28
Context Window (K)0.50.60.70.80.9Average Score(a)Gemini-2.5-Flash
(Cold Start)
4 12 20 28
Context Window (K)(b)Qwen-Plus
(Early Saturation)
4 12 20 28
Context Window (K)(c)DeepSeek-v3.2-Exp
(Logarithmic Growth)
Figure 3: Aggregate performance trajectories across context windows (4K–32K). Three distinct scaling behaviors
emerge: logarithmic growth (DeepSeek), early saturation (Qwen-Plus), and cold-start dynamics (Gemini).
4.1 Experimental Configuration
We conducted context-scaling analysis across four LLMs: DeepSeek-v3.2-Exp (2025-09-29), Qwen-Plus (2025-09-11),
Gemini-2.5-Flash, and Qwen-2.5-72B-Instruct. All experiments used AnythingLLM as the unified RAG framework
with a fixed, pre-processed segmented JSON corpus to minimize confounding variables.
Context Window and Output Settings.Context window size was varied via AnythingLLM’s workspace configura-
tion through OpenAI-compatible APIs. Output token limits scaled with context: 1K for 4K context, 2K for 8K–10K,
and 4K for ≥12K. We evaluated 11 configurations from (4K, 1K) to (32K, 4K), with finer granularity in the 10K–20K
range where preliminary results indicated performance transitions. For Qwen-2.5-72B, we additionally tested extended
contexts at 64K and 128K tokens.
Evaluation Protocol.Each model–context configuration was evaluated on the full 200-question benchmark. The
benchmark comprises 59 robustness questions (needle-in-haystack), 90 multi-hop reasoning questions (cross-document
synthesis), and 51 generation quality questions (intra-document multi-topic). Responses were scored using GPT-4.1-mini
via DeepEval’s G-Eval implementation.
Statistical Scale.The evaluation encompasses 200 questions ×4 models ×11+ configurations ×6 metrics = over
52,800 individual metric evaluations, providing sufficient statistical power for the reported comparisons.
4.2 Overall Scaling Trends
Figure 3 presents aggregate performance trajectories for the three primary models.
Convergent Performance at Scale.At 4K context, model differences are pronounced: Gemini achieves only 0.474,
while Qwen-Plus leads at 0.689 and DeepSeek reaches 0.619—a 45% gap between best and worst. This spread narrows
substantially as context increases: by 20K, all models cluster between 0.80–0.85, and at 28K, performance converges
within a 1% band (0.868–0.876).
Divergent Saturation Patterns.Beyond 28K, models exhibit divergent behaviors. Gemini and Qwen-Plus both
decline: Gemini drops from 0.876 to 0.836 ( −4.6%), and Qwen-Plus from 0.874 to 0.853 ( −2.4%). In contrast,
DeepSeek continues improving from 0.868 to 0.883 (+1.7%), demonstrating robust noise tolerance.
Architecture-Specific Inflection Points.Gemini shows minimal gains from 4K–8K before accelerating rapidly in the
12K–20K range, indicating a critical context threshold of ∼12K tokens. Qwen-Plus reaches 90% of peak performance
at approximately 16K tokens. DeepSeek requires 20K tokens for equivalent relative performance but continues scaling
where others plateau.

FAB-Bench A PREPRINT
4.3 Three Scaling Behaviors
Our analysis reveals three distinct scaling behaviors, presented in order of increasing context efficiency.
Cold-Start Dynamics.Gemini exhibits S-curve behavior: poor initial performance (0.474 at 4K) indicates limited
parametric knowledge for this domain, with a critical mass requirement of ∼12K tokens before effective reasoning
activates. Performance peaks at 28K (0.876) then declines at 32K (0.836).
Early Saturation.Qwen-Plus achieves the highest initial performance (0.689 at 4K), indicating strong parametric
knowledge. However, performance peaks at 28K (0.874) and declines at 32K—extended sequences lead to attention
dispersion where the model struggles to maintain focus amid noise. Optimal for short-to-medium context (4K–20K).
Logarithmic Growth.DeepSeek exhibits consistent logarithmic scaling ( R2≈0.91 ), improving from 0.619 (4K) to
0.883 (32K)—the highest final performance with no decline at extended lengths. This indicates robust attention filtering
suited for complex multi-document reasoning.
Extended Context Validation.Qwen-2.5-72B-Instruct extends our analysis to 128K tokens. Performance improves
from 0.594 (4K) to 0.802 (32K), then plateaus at 64K (0.795) and 128K (0.805), confirming that marginal returns
diminish beyond 32K. Complete results for all models are provided in Appendix G.
5 Diagnostic Analysis: Metric-Level Attribution
The scaling curves revealwhathappens; our six-dimensional metrics diagnosewhy.
5.1 4K Context Window
Gemini-2.5-Flash
(4K, 1K)Qwen-Plus
(4K, 1K)DeepSeek-v3.2
(4K, 1K)0.00.20.40.60.8Score4K Context Window
Factuality Depth Completeness Relevance Context Util. Support Quality
Figure 4: Metric breakdown at 4K context window—the most resource-constrained scenario.
Figure 4 reveals dimension-level performance at 4K:
Gemini(0.474) shows catastrophically low Depth (0.374), Completeness (0.399), and Support Quality (0.350), with
only Context Utilization (0.669) approaching acceptable levels—confirming inability to generate domain-specific
content without extensive grounding material.

FAB-Bench A PREPRINT
Qwen-Plus(0.689) leads through balanced scores: Factuality (0.663), Depth (0.662), Completeness (0.686), and
notably high Context Utilization (0.805), indicating strong parametric knowledge compensating for limited retrieval.
DeepSeek(0.619) shows moderate performance with high Context Utilization (0.783) but lower Depth (0.554) and
Support Quality (0.534), suggesting conservative evidence extraction without speculation.
5.2 28K Context Window
0.750.800.850.900.951.0028K Context Window
Gemini-2.5-Flash
(28K, 4K)Qwen-Plus
(28K, 4K)DeepSeek-v3.2
(28K, 4K)0Score
Factuality Depth Completeness Relevance Context Util. Support Quality
Figure 5: Metric breakdown at 28K context window (peak performance for most models).
At 28K (Figure 5), the performance landscape transforms:
Gemini(0.876) achieves remarkable recovery, with Factuality (0.901) and Completeness (0.886) leading all models—
confirming that its short-context weakness stems from grounding dependency rather than fundamental incapability.
Qwen-Plus(0.874) reaches its peak with Context Utilization (0.916) and Completeness (0.894) as strongest dimensions.
DeepSeek(0.868) demonstrates balanced excellence with all dimensions exceeding 0.81 and no discernible weak
points.
5.3 Mechanism Attribution
Three mechanisms explain the observed behaviors:
•Noise Tolerance (DeepSeek):Monotonic improvement through 32K indicates effective attention filtering—the
model benefits from additional context without information overload.
•Parametric Compensation (Qwen-Plus):Strong 4K performance reflects internal knowledge compensating for
limited retrieval. The 32K decline suggests an optimal context threshold beyond which noise degrades performance.
•Critical Mass Activation (Gemini):S-curve behavior with 4K–8K stagnation followed by 12K–28K acceleration
indicates a∼12K token threshold for reasoning activation in this domain.

FAB-Bench A PREPRINT
5.4 Attention Dilution at Extreme Context (32K)
Factuality Depth Completeness Relevance Context Util. Support Qual.DeepSeek-v3.2
Qwen-Plus
Gemini-2.5+0.012 +0.027 +0.019 +0.007 -0.004 +0.030
-0.011 -0.020 -0.027 -0.012 -0.027 -0.026
-0.032 -0.043 -0.042 -0.054 -0.018 -0.051Metric-Level Performance Change: 28K  32K Context Window
-0.06-0.04-0.020.000.020.04Performance Change ( )
Figure 6: Metric-level performance changes from 28K to 32K context window. Green: improvement; red: decline.
Gemini shows severe attention dilution with Relevance (−5.4%) and Support Quality (−5.1%) most affected.
Figure 6 presents metric-level changes from 28K to 32K, revealing a consistentattention dilution signaturein declining
models.
Attention Dilution Signature.For Gemini, the largest drops occur inRelevance( −6.1%) andSupport Quality
(−6.0%), indicating failure to identify query-relevant information. TheDepthdecline ( −5.3%) suggests scattered
attention prevents deep analysis. Critically,Context Utilizationshows relatively smaller decline (Gemini −2.0%,
Qwen-Plus −2.9%), indicating these models stillattemptto leverage extended context but extract less value per token.
This distinguishes attention dilution from context truncation.
Contrast with Noise-Tolerant Architecture.DeepSeek continues improving at 32K (+1.7%), with Support Quality
improvement (+3.6%) paired with slight Context Utilization decline ( −0.4%)—indicatingselectiveutilization where
the model processes more context but references only high-quality evidence.
5.5 Metric Independence Analysis
To validate that our six metrics capture distinct failure modes rather than collapsing into a single quality factor,
we analyze inter-metric correlations across all experimental configurations ( ∼33 model-context combinations ×6
dimensions).
Table 4: Inter-metric Pearson correlation coefficients computed across all model-context configurations. Values below
0.70 indicate meaningful independence; values above 0.85 suggest potential redundancy.
Fact. Depth Comp. Rel. Ctx.U. Supp.
Factuality 1.00
Depth 0.96 1.00
Completeness 0.97 0.99 1.00
Relevance 0.98 0.99 0.99 1.00
Context Util. 0.90 0.93 0.93 0.92 1.00
Support Quality 0.98 0.99 0.99 0.99 0.91 1.00
Table 4 reveals high aggregate correlations ( >0.90) across all metric pairs when computed over mean scores per
model-context configuration. This is expected: as context increases,allmetrics improve together because models

FAB-Bench A PREPRINT
receive more relevant information. However, this aggregate correlation masks the diagnostic value that emerges at
specific operating points.
Diagnostic Independence at Fixed Operating Points.The metrics’ diagnostic utility manifests when comparing
models at identical context windows, where inter-metricprofilesdiverge meaningfully:
•At 4K, Gemini’s Context Utilization (0.669) is 79% higher than its Support Quality (0.350)—a 2.3 ×ratio indicating
the modelreadscontext but cannotciteit accurately.
•At 28K, Gemini achieves the highest Factuality (0.901) but lower Depth (0.818), while DeepSeek shows the opposite
pattern (Factuality: 0.892, but highest balanced profile). These cross-metric divergences would be invisible to a single
composite score.
•In the case study (Section 7), Gemini-2.5-Flash achieves high Context Utilization (0.90) but catastrophic Factuality
(0.17)—a failure mode uniquely identifiable through multi-dimensional evaluation.
Distinct Sensitivity to Context Scaling.Different metrics respond differently to context expansion: Context Uti-
lization saturates earliest (reaching >0.80 by 8K for all models), while Technical Depth and Support Quality show
the steepest improvement trajectories and the largest model-specific variation. The 28K-to-32K attention dilution
(Section 5.4) disproportionately affects Relevance and Support Quality while largely preserving Context Utilization—a
diagnostic pattern only visible through multi-dimensional evaluation.
We acknowledge that the high aggregate correlations limit the metrics’ discriminative power for ranking models at a
single context point. The primary diagnostic value lies inprofile analysis—comparing metric patterns across models,
context windows, and failure cases—rather than individual metric rankings.
6 Cross-Framework Evaluation
To validate evaluation portability, we deployed our benchmark on three additional production RAG frameworks:
RAGFlow, MaxKB, and Metaso.
0.450.500.550.600.650.700.750.80Cross-Framework RAG Performance
MaxKB
(DeepSeek-v3.2)Metaso
(General)RAGFlow
(DeepSeek-v3.2)0Score
Factuality Depth Completeness Relevance Context Util. Support Quality
Figure 7: Performance comparison across three external RAG frameworks.

FAB-Bench A PREPRINT
Table 5: Cross-framework performance breakdown. All frameworks evaluated on the same 200-question benchmark.
Retrieval strategies differ: RAGFlow uses visual document parsing with hybrid search; MaxKB uses DeepSeek-V3.2
(non-thinking mode) with chunk-based retrieval; Metaso uses a proprietary model with web-augmented retrieval.
Metric RAGFlow MaxKB Metaso
Factuality 0.610 0.560 0.650
Depth 0.485 0.531 0.624
Completeness 0.550 0.566 0.692
Relevance 0.577 0.602 0.684
Context Util. 0.549 0.662 0.711
Support Quality 0.498 0.520 0.637
Average0.545 0.574 0.666
Framework-Specific Failure Modes.Each framework exhibits distinct metric profiles reflecting different architec-
tural choices.Metaso(0.67 avg) leads in Context Utilization (0.711) and Completeness (0.692), suggesting effective
information synthesis.MaxKB(0.57 avg) shows a notable gap between Context Utilization (0.662) and Depth
(0.531), indicating surface-level extraction without deep reasoning—consistent with its use of DeepSeek-V3.2 in
non-thinking mode.RAGFlow(0.55 avg) scores lowest in Depth (0.485) and Support Quality (0.498), suggesting
chunk fragmentation issues despite its advanced visual parsing capabilities.
Consistent Difficulty Ordering.Depth and Support Quality consistently score lower than Factuality across all
frameworks, confirming the benchmark captures intrinsic task difficulty independent of platform architecture.
Deployment Portability.Adapting our benchmark to each new framework required minimal engineering: API
integration for query submission, response parsing, and source citation extraction. The core evaluation pipeline
remained unchanged, suggesting our methodology can serve as a reusable evaluation layer for heterogeneous enterprise
RAG deployments.
7 Case Study: Pulsed Atomic Layer Etching
To illustrate how multi-dimensional evaluation reveals failure modes invisible to single-score metrics, we present a
representative case from technical parameter extraction.
7.1 Task Description
Test case MULTI_069 requires extracting precise process parameters from patent documentation describing pulsed
atomic layer etching (ALE) for ruthenium removal—a critical BEOL process in advanced semiconductor manufacturing.
The fill-in-the-blank question requires five specific values: optimal bias voltage range, comparison direction, etch rate
comparison, and synergy comparison relative to continuous ALE.
The ground truth specifies:600V–1200Vbias window,higherbias voltages (vs. 60–100V for continuous ALE),higher
etch rates (5–6 Å/cycle vs. 2–3 Å/cycle), andhighersynergy.
7.2 Model Responses and Error Analysis
Table 6: Model responses for pulsed ALE parameter extraction (MULTI_069, 18K context).
Model Voltage Range Direction Etch Rate Synergy Factuality
Ground Truth 600–1200V higher faster increased —
DeepSeek-V3.2 600–1200V higher faster increased0.99
Qwen-Plus 50–150V lower higher enhanced 0.11
Gemini-2.5-Flash 10–50V lower higher higher 0.17
Both Qwen-Plus and Gemini exhibitprocess variant confusion: the retrieved context interleaves continuous ALE
(60–100V) and pulsed ALE (600–1200V) specifications. Qwen-Plus extracts values near the continuous range (50–

FAB-Bench A PREPRINT
150V), while Gemini hallucinates 10–50V . DeepSeek correctly disambiguates by reasoning about the pulsed duty cycle
mechanism.
7.3 Diagnostic Attribution via Multi-Dimensional Metrics
0.20.40.60.81.0Completeness
Depth
Factuality
RelevanceContext Util.Support Quality
Gemini-2.5-Flash
Qwen-Plus
DeepSeek-v3.2
Figure 8: Six-dimensional comparison for MULTI_069 (18K context). The divergence between Context Utilization
and Factuality for Gemini (0.90 vs. 0.17) reveals confident misattribution—a failure mode invisible to single-score
evaluation.
The radar chart (Figure 8) reveals a critical diagnostic pattern: Gemini achieves high Context Utilization (0.90) but
catastrophic Factuality (0.17). This combination indicates the modelconfidently cites incorrect portionsof retrieved
context—confusing continuous and pulsed ALE parameters. A single composite score would obscure this specific
failure mode, which has direct implications for safety-critical semiconductor process specification.
This case exemplifiesdisambiguation under structural similarity: when retrieved contexts contain multiple process
variants with similar descriptive patterns but different quantitative specifications, models must parse document structure,
maintain attention to qualifying terms across long spans, and cross-validate extracted values.
8 Discussion
8.1 Interpreting Scaling Behaviors
We connect our three scaling behaviors to prior findings. Qwen-Plus’s strong 4K performance and early saturation
aligns with Mallen et al. [2023], who found models with stronger parametric knowledge show diminishing returns
from retrieval. DeepSeek’s sustained 32K improvement contrasts with the “lost in the middle” phenomenon [Liu et al.,
2024b], suggesting effective attention distribution. Gemini’s S-curve resembles multi-document QA patterns requiring
sufficient context density for coherent reasoning [Liu et al., 2024b]. These interpretations remain hypotheses; definitive
mechanistic attribution would require access to model internals unavailable through commercial APIs.
8.2 Dynamic Routing Strategy
The crossover point at∼15K–16K tokens motivates context-aware model routing:

FAB-Bench A PREPRINT
•Short contexts (<14K):Route to Qwen-Plus for maximum efficiency.
•Complex reasoning (>16K):Route to DeepSeek for stable multi-document synthesis.
•Batch summarization (20K–28K):Deploy Gemini for massive context ingestion, capped at 28K.
8.3 Limitations and Validity Threats
LLM-as-Judge Reliability.Our evaluation relies on GPT-4.1-mini as the judge model via G-Eval with structured
rubrics [Liu et al., 2023c, Zheng et al., 2024]. While the QA generation pipeline has been validated through expert
evaluation (Appendix D), full human–judge correlation studies for the six downstream evaluation metrics—using
calibrated annotator pools with inter-annotator agreement metrics—remain an important direction for future work. The
structured G-Eval rubrics with separate objective/subjective scoring criteria are designed to mitigate known LLM judge
biases, and the high consistency of our results across four LLMs and four RAG frameworks suggests the evaluation
signal is robust.
Benchmark Scale.Our 200 curated questions (from 1,300+ generated) span three synthesis strategies across 188
topics. While sufficient for identifying significant performance differences (52,800+ metric evaluations across all
configurations), larger benchmarks would enable finer-grained edge-case analysis.
Retrieval Configuration.Experiments use AnythingLLM’s default retrieval (max context snippets = 4, similarity
threshold = 0.25). The cross-framework evaluation (Section 6) provides implicit retrieval ablation—RAGFlow,
MaxKB, and Metaso employ different retrieval strategies (visual parsing, chunk-based, web-augmented) yet our metrics
consistently diagnose framework-specific failure modes. Explicit retriever ablations (e.g., TF-IDF vs. dense retrieval)
remain future work.
Domain Specificity.Semiconductor manufacturing represents an extreme case of specialized knowledge. The
generation pipeline is domain-agnostic (requiring only a corpus and knowledge base), but optimal context thresholds
may require recalibration for other domains.
Model Selection and Reproducibility.We evaluate three primary models and one extended model, pinning exact
version strings and dates. The rapidly evolving LLM landscape means specific findings may not transfer to newer
releases, though the methodology and benchmark remain applicable.
9 Conclusion
We present FAB-Bench, an automated framework for evaluating RAG systems on proprietary domain knowledge.
Through systematic experiments on semiconductor manufacturing documentation (150+ papers, 70+ patents), we make
three contributions:
First, we establish a reproducible methodology for private-domain RAG evaluation with adaptive benchmark generation
and six diagnostic metrics enabling precise failure attribution. The curated 200-question benchmark dataset is released
athttps://github.com/FuturefabAI/FAB-Bench and is directly applicable to other vertical domains, requiring
only a domain-specific corpus and knowledge base.
Second, context-window scaling analysis (4K–32K) reveals three distinct behaviors—logarithmic growth,early
saturation, andcold-start dynamics—with metric-level decomposition identifyingattention dilutionas the mechanism
behind performance degradation at extreme context lengths.
Third, cross-framework validation on four production RAG systems confirms evaluation portability and demonstrates
consistent diagnostic capability across heterogeneous architectures.
Future Work.Priority extensions include: (1) human correlation studies validating LLM judge reliability with inter-
annotator agreement metrics; (2) explicit retriever ablations comparing sparse, dense, and hybrid retrieval strategies;
(3) per-category scaling analysis disaggregating performance across robustness, multi-hop, and generation quality
question types; and (4) extension to additional vertical domains with domain-specific knowledge bases.

FAB-Bench A PREPRINT
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, et al. Gpt-4
technical report.arXiv preprint arXiv:2303.08774, 2023.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, et al. Language
models are few-shot learners.Advances in Neural Information Processing Systems, 33:1877–1901, 2020.
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-
task benchmark and analysis platform for natural language understanding. InProceedings of EMNLP Workshop
BlackboxNLP, 2018.
Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel
Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. InAdvances in
Neural Information Processing Systems, 2019.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring
massive multitask language understanding.Proceedings of ICLR, 2021a.
OpenAI. Gpt-4 technical report.arXiv preprint arXiv:2303.08774, 2023.
Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, et al. C-eval: A multi-level
multi-discipline chinese evaluation suite for foundation models.Advances in Neural Information Processing Systems,
2023.
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
Think you have solved question answering? try arc, the ai2 reasoning challenge. InProceedings of NeurIPS, 2018.
Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods.
Proceedings of ACL, 2022.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, et al. Training verifiers
to solve math word problems.arXiv preprint arXiv:2110.14168, 2021.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob
Steinhardt. Measuring mathematical problem solving with the math dataset.Proceedings of NeurIPS, 2021b.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, et al. Evaluating
large language models trained on code.arXiv preprint arXiv:2107.03374, 2021.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient
have? a large-scale open domain question answering dataset from medical exams.Applied Sciences, 2021.
Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-
choice dataset for medical domain question answering.Proceedings of CHIL, 2022.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, et al. Large language models
encode clinical knowledge.Nature, 2023a.
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Le Hou, et al. Towards expert-level medical
question answering with large language models.arXiv preprint arXiv:2305.09617, 2023b.
Neel Guha et al. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models.
arXiv preprint arXiv:2308.11462, 2023.
Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Songyang Zhang, Kai Chen, Zongwen Shen, and
Jidong Ge. Lawbench: Benchmarking legal knowledge of large language models.arXiv preprint arXiv:2309.16289,
2023.
Pranab Islam et al. Financebench: A new benchmark for financial question answering.arXiv preprint, 2023.
Qianqian Xie, Weiguang Han, Zhengyu Zhang, et al. Finben: A holistic financial benchmark for large language models.
arXiv preprint arXiv:2402.12659, 2024.
Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David
Rosenberg, and Gideon Mann. Bloomberggpt: A large language model for finance.arXiv preprint arXiv:2303.17564,
2023.
Mingjie Liu, Teodor-Dumitru Ene, Robert Kirber, Chris Cheng, Nathaniel Tiber, Tiffany Greaves, et al. Chipnemo:
Domain-adapted llms for chip design.arXiv preprint arXiv:2311.00176, 2023a.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, et al. Retrieval-
augmented generation for knowledge-intensive nlp tasks.Advances in Neural Information Processing Systems, 33:
9459–9474, 2020.

FAB-Bench A PREPRINT
Rubing Chen, Jiaxin Wu, Jian Wang, et al. Rethinking domain-specific llm benchmark construction: A
comprehensiveness-compactness approach.arXiv preprint arXiv:2508.07353, 2024a.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, et al. Judging
llm-as-a-judge with mt-bench and chatbot arena.Advances in Neural Information Processing Systems, 2024.
Luka Budler, Luka Gosak, and Gregor Štiglic. A brief review on benchmarking for large language models evaluation in
healthcare.Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2025.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang.
Retrieval-augmented generation for large language models: A survey.arXiv preprint arXiv:2312.10997, 2023.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented
generation. InProceedings of AAAI, volume 38, pages 17754–17762, 2024b.
Yi Liu, Lianzhe Huang, Shicheng Li, Sishuo Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Recall: A
benchmark for llms robustness against external counterfactual knowledge.arXiv preprint arXiv:2311.08147, 2023b.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. Ares: An automated evaluation framework for
retrieval-augmented generation systems. InProceedings of NAACL, 2024.
Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval
augmented generation. InProceedings of EACL: System Demonstrations, pages 150–158, 2024.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, et al. Crag – comprehensive rag benchmark.
arXiv preprint arXiv:2406.04744, 2024.
Yixuan Tang and Yang Yi. Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries.arXiv
preprint arXiv:2401.15391, 2024.
Mattia Rengo, Senad Beadini, Domenico Alfano, and Roberto Abbruzzese. A system for comprehensive assessment of
rag frameworks.arXiv preprint arXiv:2504.07803, 2025.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: NLG evaluation using
GPT-4 with better human alignment. InProceedings of EMNLP, pages 2511–2522, 2023c.
Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang
Sui. Large language models are not fair evaluators.arXiv preprint arXiv:2305.17926, 2023.
Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vulic, Anna Korhonen, and Nigel Collier. Aligning
with human judgement: The role of pairwise preference in large language model evaluators.arXiv preprint
arXiv:2403.16950, 2024a.
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In
International Conference on Learning Representations, 2020.
Matthew Renze. The effect of sampling temperature on problem solving in large language models. InFindings of
EMNLP, pages 7346–7356, 2024.
Confident AI. DeepEval: Open-source llm evaluation framework. https://docs.confident-ai.com , 2025. Version
3.2.1.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust
language models: Investigating effectiveness of parametric and non-parametric memories. InProceedings of ACL,
pages 9802–9822, 2023.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in
the middle: How language models use long contexts.Transactions of the Association for Computational Linguistics,
12:157–173, 2024b.

FAB-Bench A PREPRINT
A Semiconductor Domain Knowledge Base
This appendix provides the complete listing of 431 technical terms in our semiconductor domain knowledge base,
organized into seven categories with associated precision weights.
A.1 Manufacturing Processes (60 terms, weight = 1.5)
Core fabrication techniques:wafer, lithography, etching, CVD, PVD, ALD, ion implantation, annealing, CMP,
cleaning, inspection, metrology, mask, exposure, develop, strip, deposition, diffusion, oxidation, epitaxy, packaging,
wire bonding, flip chip, WLP, SiP, BGA, CSP
Advanced lithography:immersion lithography, multi-patterning, self-aligned, spacer, hard mask, anti-reflective
coating, photoresist, reticle
Etching techniques:reactive ion etching, RIE, plasma etching, dry etching, wet etching, isotropic etching, anisotropic
etching
Deposition methods:PECVD, LPCVD, MOCVD, MBE, electroplating, sputtering, atomic layer deposition, physical
vapor deposition
CMP and cleaning:chemical mechanical planarization, post-CMP cleaning, megasonic cleaning, RCA clean, HF dip
Metrology:overlay metrology, critical dimension measurement, ellipsometry, optical CD, scatterometry
A.2 Materials Science (51 terms, weight = 1.3)
Substrate materials:silicon, GaAs, GaN, SiC, InP, germanium, polysilicon, monocrystalline, amorphous
Dielectric materials:high-k, low-k, SiO2, Si3N4, Al2O3, HfO2, silicon dioxide, silicon nitride, hafnium oxide,
aluminum oxide, zirconium oxide, tantalum pentoxide, ultra-low-k, porous silica, carbon-doped oxide
Metal materials:copper, aluminum, tungsten, titanium, tantalum, indium, silver, gold, platinum, nickel, cobalt,
ruthenium, molybdenum
Compound semiconductors:gallium arsenide, gallium nitride, silicon carbide, indium phosphide, aluminum nitride,
zinc oxide
2D materials:graphene, MoS2, transition metal dichalcogenides
Gate stack materials:metal gate, work function metal, barrier layer, capping layer, etch stop layer
A.3 Device Physics (62 terms, weight = 1.4)
Transistor types:MOSFET, FinFET, GAAFET, CMOS, BJT, JFET, IGBT, diode, transistor
Advanced transistors:gate-all-around, nanosheet, nanowire, CFET, complementary FET, vertical transistor, TFET,
tunnel FET, FDSOI, fully depleted SOI
Memory devices:memory, DRAM, SRAM, NAND, NOR, FRAM, ReRAM, MRAM, flash memory, 3D NAND, PCM,
phase change memory, RRAM, STT-MRAM, emerging memory
Passive components:resistor, capacitor, inductor
Integrated circuits:IC, integrated circuit, ASIC, SoC, system on chip, logic circuit, analog circuit, mixed-signal
Sensors and MEMS:MEMS, sensor, actuator, image sensor, CMOS sensor, pressure sensor, accelerometer, gyroscope
Specialty devices:RF, power device, optoelectronic, LED, laser diode, photodiode, solar cell, TFT, thin film transistor
A.4 Process Nodes (65 terms, weight = 1.6)
Process nodes:14nm, 10nm, 7nm, 5nm, 3nm, 2nm, 1nm, 28nm, 22nm, 16nm, 12nm, 8nm, 6nm, 4nm, process node,
technology node
Lithography technologies:EUV , DUV , 193nm, 248nm, 365nm, ArF, KrF, i-line, g-line, extreme ultraviolet, deep
ultraviolet, 13.5nm, high-NA EUV

FAB-Bench A PREPRINT
Process integration:FinFET process, SOI, FD-SOI, bulk, bulk silicon, strained silicon, strain engineering, SiGe,
silicon germanium, HEMT, high electron mobility
Process stages:FEOL, BEOL, MEOL, front end of line, back end of line, middle of line
Advanced techniques:gate-first, gate-last, replacement gate, high-k metal gate, HKMG, multiple patterning, SADP,
SAQP, self-aligned double patterning, self-aligned quadruple patterning, LELE, LFLE
Interconnect:damascene, dual damascene, copper interconnect, via, through silicon via, TSV , backside power delivery
A.5 Testing Methodologies (51 terms, weight = 1.2)
Reliability testing:reliability, yield, failure analysis, electrical test, functional test, burn-in, temperature cycling,
thermal shock, humidity test
Failure mechanisms:electromigration, hot carrier, NBTI, TDDB, negative bias temperature instability, time dependent
dielectric breakdown, stress migration, void formation
Quality control:SPC, statistical process control, parameter drift, defect density, process capability, Cpk, design of
experiments, DOE
Inspection techniques:critical dimension, CD-SEM, AFM, SEM, TEM, X-ray, atomic force microscopy, scanning
electron microscopy, transmission electron microscopy, optical inspection, e-beam inspection
Electrical characterization:I-V curve, C-V measurement, DLTS, deep level transient spectroscopy, Hall effect,
mobility measurement, sheet resistance, four-point probe, contact resistance
Wafer-level testing:probe test, wafer sort, parametric test, functional test, final test, package test
A.6 Applications (62 terms, weight = 1.1)
Computing processors:AI chip, GPU, CPU, NPU, TPU, FPGA, ASIC, graphics processing unit, neural processing
unit, tensor processing unit, application specific IC, DSP, digital signal processor, MCU, microcontroller
Computing systems:edge computing, cloud computing, data center, server, supercomputer, quantum computing,
neuromorphic computing, in-memory computing
Consumer electronics:smartphone, tablet, laptop, wearable, smartwatch, AR glasses, VR headset
Automotive:automotive electronics, ADAS, autonomous driving, LiDAR, radar, infotainment, powertrain
Communication:5G, 6G, IoT, internet of things, wireless, RF transceiver, baseband, modem
AI/ML:neural network, machine learning, deep learning, inference, training, transformer, large language model
Emerging applications:blockchain, cryptocurrency, mining, virtual reality, VR, augmented reality, AR, metaverse,
HPC, high performance computing
A.7 Performance Parameters (80 terms, weight = 1.7)
Voltage parameters:threshold voltage, Vth, supply voltage, VDD, VSS, voltage, operating voltage, breakdown voltage,
junction temperature
Current parameters:leakage current, drive current, Ion, Ioff, current, saturation current, subthreshold current, gate
leakage, junction leakage
Power parameters:power consumption, static power, dynamic power, power dissipation, TDP, thermal design power,
power density, energy efficiency
Timing parameters:switching speed, delay, propagation delay, rise time, fall time, setup time, hold time, clock skew,
clock frequency, access time
Frequency parameters:frequency, bandwidth, cutoff frequency, fT, fmax, operating frequency, clock speed
Resistance and capacitance:resistance, capacitance, Ron, on-resistance, gate capacitance, parasitic capacitance, RC
delay, interconnect resistance
Temperature parameters:operating temperature, junction temperature, thermal resistance, temperature coefficient

FAB-Bench A PREPRINT
Noise and linearity:noise, noise figure, SNR, signal to noise ratio, linearity, THD, total harmonic distortion
RF parameters:gain, phase, S-parameters, impedance matching, insertion loss, return loss
Performance metrics:DIBL, drain induced barrier lowering, subthreshold swing, SS, transconductance, gm, output
conductance, GIDL, gate induced drain leakage, mobility, carrier mobility, saturation velocity
A.8 Knowledge Base Design Rationale
Coverage breadth:The 431 terms span the full semiconductor technology stack from materials science to system
applications.
Precision weighting:Category weights ( wi∈[1.1,1.7] ) reflect technical rigor requirements, with parameters (1.7)
and process nodes (1.6) weighted highest due to quantitative precision requirements.
Term granularity:Includes both high-level concepts (e.g., “transistor”) and specific implementations (e.g., “FinFET”,
“GAAFET”, “nanosheet”).
Temporal coverage:Terms encompass mature technologies, current leading-edge, and emerging concepts to ensure
benchmark relevance as technology advances.
B Generation System Implementation
B.1 Precision Classification
Documents are classified based on technical density ρ(d) and weighted high-precision category presence (Eq. 1). The
disjunctive logic ensures documents containing critical parameters are classified as high-precision even with low overall
density.
B.2 Adaptive Temperature Control
Base Temperature Ranges.Temperature ranges vary by precision level to balance accuracy with diversity:
[τmin(p), τ max(p)] =

[0.4,0.8]p=high
[0.5,0.9]p=medium
[0.6,1.0]p=low(4)
Progressive Temperature.Base temperature progresses with successful generations to encourage diversity:
τprogress (k) = (τ max−τmin)×mink
20,0.8
(5)
wherekis the count of successfully generated questions.
Category-Specific Adjustments.Fine-grained adjustments based on dominant content category:
∆τc(c∗) =

−0.10c∗=parameters
−0.08c∗=processes
−0.05c∗=devices
−0.03c∗=testing
0.00c∗=materials
+0.02c∗=manufacturing
+0.05c∗=applications(6)
Failure Recovery Boosts.Temperature increases after generation failures:
∆τattempt(a) = min(0.25,0.08×a)(7)
∆τsimilarity (s) = min(0.15,0.05×s)(8)
whereacounts all failed attempts andstracks consecutive similarity failures.

FAB-Bench A PREPRINT
Final Temperature Computation.
τ=clip(τ min+τ progress + ∆τ c+ ∆τ attempt + ∆τ similarity ,0.1,1.0)(9)
Complementary Nucleus Sampling.Adaptive top-pmaintains coherence during high-temperature exploration:
pnucleus (τ) =

0.95τ≤0.4
0.90 0.4< τ≤0.7
0.85τ >0.7(10)
B.3 Adaptive Similarity Thresholds
Base Thresholds by Precision Level.
θbase(p) =

0.70p=high
0.75p=medium
0.80p=low(11)
Lower thresholds for high-precision content acknowledge that questions about specialized topics necessarily share
technical terminology while remaining substantively different.
Progressive Relaxation.
θsim(r) = max(0.50, θ base(p)−0.05×r)(12)
Weighted Similarity Computation.
sim weighted (q, q′) =Jaccard(W q, Wq′) +α·Jaccard(T q, Tq′)(13)
whereWdenotes all word tokens,T⊆Wdenotes technical terms, andα= 0.05.
For corpora with≥5existing questions, TF-IDF semantic similarity provides additional validation:
sim semantic (q, q′) = cos(⃗ v TF-IDF (q),⃗ v TF-IDF (q′))(14)
Questions are rejected ifmax q′∈Qsim semantic (q, q′)> θ sim+ 0.05.
B.4 Document Chunking
Documents are segmented using sliding window approach: chunk size 512 tokens, overlap 128 tokens (25%). Cross-
document linking threshold:θ link= 0.1.
C Dynamic Prompt Templates
This section describes the core logic of our dynamic prompt engineering system.
C.1 Prompt Architecture
All prompts consist of four modular components that combine based on generation context:
1.Base Template: Defines role (AI Benchmark Scientist), injects context sections, specifies JSON output format
2.Test-Type Module: Scenario-specific instructions based on evaluation target
3.Format Specification: Question type requirements with balance tracking
4.Diversity Directive: Activated after repeated similarity failures
C.2 Test-Type Configurations
Three test types target different RAG capabilities:

FAB-Bench A PREPRINT
Table 7: Test-type configurations
Test Type Instruction Focus Primary Secondary
Robustness Find rare/specific facts that are easily overlooked (needle-in-
haystack); construct unambiguous ground truthFactuality Depth, Completeness
Multi-Hop Identify logical chains requiring ≥2 sections; force cross-section
synthesisCompleteness Depth, Factuality
Generation Assume perfect context; require deep reasoning (analysis, compari-
son, causation)Depth Completeness, Factuality
C.3 Question Format Balance
We maintain target distribution of 40% objective and 60% subjective questions. The system tracks:
robj=nobj
nobj+n subj(15)
Whenr obj<0.5, the system forces objective question generation.
Objective Questions (40% target).Four subtypes with specific requirements:
•Mathematical Calculation (30%): Must extract ALL parameters from context; use context-specific formulas; show
step-by-step with units
•Fill-in-Blank (25%): Extract EXACT values with conditions and units as stated
•True/False (25%): Statements about specific mechanisms requiring multi-step understanding; create both true and
false statements
•Multiple Choice (20%): All options in one line; only context provides distinguishing details
Subjective Questions (60% target).Five archetypes rotate via(question_count) mod 5:
1. Definition/Specification: specific definitions or specification values
2. Process Explanation: process sequences or event chains
3. Causal Reasoning: reasons or purposes behind specifications
4. Comparative Analysis: compare/contrast related concepts
5. Problem Identification: potential problems, defects, or limitations
C.4 Diversity Enhancement
When similarity failures exceed threshold (s >2), a lightweight directive is injected:
DIVERSITY FOCUS: Previoussuniqueness attempts suggest trying a new approach
RECOMMENDED STYLE: Focus on “[current archetype from rotation]”
This provides guidance to explore unexplored patterns without over-constraining generation.
D Expert Validation of Generated Benchmarks
To assess generation quality, domain experts rated QA pairs from a multi-stage ablation study on four dimensions (1–5
scale): accuracy (factual correctness of answer), relevance (alignment with context), difficulty (cognitive challenge
level), and diversity (novelty relative to other questions). Each condition generated 18 QA pairs (6 per question type:
ROB, MULTI, GEN), with precision levels (high/medium/low) annotated per question based on the domain knowledge
base classification.

FAB-Bench A PREPRINT
D.1 Ablation Results
As shown in Table 8, the full adaptive system with enhanced prompts (Group E) produces the highest-quality outputs
across all evaluated dimensions, with zero retry failures.
Table 8: Expert evaluation of QA generation quality across ablation conditions (18 QA pairs per group, 1–5 scale).
Group Configuration Acc. Rel. Diff. Div. Retries Time
A (Baseline) Fixedτ= 0.5, Fixedθ= 0.654.67 4.72 4.00 3.67 15 18m 33s
B (+Adaptiveτ) Adaptiveτ, Fixedθ= 0.654.89 4.83 4.17 3.39 16 30m 37s
C (+Adaptiveθ) Fixedτ= 0.5, Adaptiveθ4.61 4.61 3.61 3.28 47 36m 55s
D (Adaptiveτ+θ) Adaptiveτ, Adaptiveθ4.33 4.78 3.28 3.17 12 32m 19s
E (Full System) Adaptiveτ+θ+ enhanced prompts5.00 5.00 4.60 4.53 025m 15s
Notably, adaptive temperature and threshold alone (Group D) do not outperform the baseline in all dimensions—
Difficulty (3.28 vs. 4.00) and Diversity (3.17 vs. 3.67) decrease, suggesting that parameter adaptation without prompt
enhancement leads to conservative generation. The full system (Group E) combines adaptive parameters with enhanced
prompt templates (Appendix C), achieving consistent improvements across all dimensions.
D.2 Per-Question-Type Quality Analysis
Table 9 breaks down expert ratings by question type, revealing systematic quality patterns across synthesis strategies.
Table 9: Expert-rated quality by question type (averaged across all ablation groups).
Type Accuracy Relevance Difficulty Diversity
Robustness (ROB) 4.58 4.62 2.38 2.92
Multi-Hop (MULTI) 4.58 4.83 4.12 3.33
Generation (GEN) 4.71 4.75 4.79 3.88
The difficulty gradient (ROB: 2.38 <MULTI: 4.12 <GEN: 4.79) confirms the three synthesis strategies produce
appropriately graded complexity: needle-in-haystack questions are factual and straightforward, cross-document multi-
hop questions require moderate reasoning, and generation quality questions demand deep analytical reasoning. Accuracy
remains high (>4.5) across all types, indicating factual correctness is maintained regardless of difficulty level.
E Evaluation Rubrics
All metrics use GPT-4.1-mini via DeepEval’s G-Eval with chain-of-thought reasoning, scoring 0–10 (internally
normalized to 0–1). Rubrics adapt between objective and subjective questions.
E.1 Completeness
Table 10: Completeness rubric
Score Objective Subjective
9–10 Correct answer with complete explanation and clear rea-
soningAddresses all key aspects thoroughly
7–8 Correct answer with good explanation, minor gaps Covers all major aspects with good detail
5–6 Answer with basic explanation, some missing details Addresses most important aspects reasonably
3–4 Partial answer or incomplete explanation Covers key aspects with varying depth
1–2 Minimal answer, lacks proper explanation Addresses some but misses important points
0 No clear answer or off-topic Fails to address key aspects

FAB-Bench A PREPRINT
E.2 Technical Depth
Table 11: Technical depth rubric
Score Objective Subjective
9–10 Sophisticated understanding with detailed methodology Sophisticated understanding with nuanced analysis
7–8 Strong analytical thinking with clear solution steps Strong analytical thinking with good conceptual depth
5–6 Reasonable technical detail with some analysis Reasonable analysis with adequate technical detail
3–4 Basic approach but relatively shallow analysis Surface-level analysis with limited sophistication
1–2 Minimal analytical content, superficial approach Superficial treatment with minimal content
0 No meaningful analysis demonstrated No meaningful analysis or understanding
E.3 Factuality
Table 12: Factuality rubric
Score With Retrieval Context Without Context
9–10 All claims match context exactly, no contradictions Claims match expected output, no contradictions
7–8 Minor imprecision in non-critical details Most claims consistent, no significant errors
5–6 Generally accurate with minor factual gaps Some inconsistencies but generally reasonable
3–4 Noticeable inaccuracies but core info correct Notable contradictions or implausible info
1–2 Several factual errors or contradictions Major contradictions or obviously incorrect
0 Major errors or predominantly false information —
E.4 Relevance
Table 13: Relevance rubric
Score Objective Subjective
9–10 Directly addresses specific question format Every sentence directly addresses the question
7–8 Stays focused with minimal deviation Strongly focused with minimal deviation
5–6 Generally addresses question, some unnecessary details Mostly focused with minor tangential elements
3–4 Partially addresses with irrelevant information Generally on-topic but includes unnecessary info
1–2 Somewhat related with significant drift Partially addresses with significant drift
0 Fails to address specific requirements Minimal connection to actual question
E.5 Context Utilization
Table 14: Context utilization rubric
Score With Retrieval Context Manual Setup
9–10 Seamlessly weaves multiple sources together Sophisticated domain expertise with advanced terminol-
ogy
7–8 Effectively uses most relevant context Clear evidence of domain-specific knowledge
5–6 Incorporates key context adequately Some evidence of specialized knowledge
3–4 Uses some context, integration could improve Minimal evidence of specialized knowledge
1–2 Minimal effective use of available context Generic response with limited domain content
0 Completely ignores or contradicts context No domain-specific content

FAB-Bench A PREPRINT
E.6 Support Quality
Table 15: Support quality rubric
Score Objective Subjective
9–10 Exceptional evidence with specific calculations/formulas Exceptional evidence with specific, concrete details
7–8 Strong evidence with good specificity Strong evidence with good specificity
5–6 Adequate details with reasonable explanation Adequate details with reasonable examples
3–4 Some details but could be more specific Some details but could be more specific
1–2 Minimal evidence or poorly explained Minimal evidence or poorly chosen examples
0 Lacks evidence or contains misleading explanations Lacks evidence or contains misleading examples
F Evaluation Platform Details
F.1 Framework Capabilities
Table 16: RAG framework integration capabilities.
Framework API Upload Sources Exposed Model Config Setup
AnythingLLM✓ ✓ ✓Recommended
RAGFlow✓ ✓ ✓Recommended
MaxKB – –✓Required
Metaso✓Partial – Required
Key Differences. AnythingLLMprovides full retrieval transparency through its sources field, returning document
chunks with relevance scores.
RAGFlowuses an OpenAI-compatible API endpoint, returning responses in standard format. Source attribution
requires inference from response content.
MaxKBdoes not expose retrieval context through its API—sources are visible only in the web interface. Our adapter
implements heuristic detection using domain-specific indicators (e.g., “according to”, “standard specifies”, “SEMI”)
and professional terminology presence.
Metasoreturns structured references but uses a proprietary model that cannot be reconfigured, limiting participation to
cross-framework evaluation.
F.2 Adaptive Evaluation Strategy
The evaluation layer adapts metric computation based on framework capabilities. For frameworks without source
exposure, Context Utilization switches from direct source comparison to heuristic inference based on response
characteristics, ensuring fair evaluation across frameworks with different transparency levels.
F.3 Dual-Mode Assessment
•Mode A (with_kb): Standard RAG query through framework’s native retrieval pipeline
•Mode B (without_kb): Gold context injected into prompt, bypassing retrieval to isolate generation capabilities
Cross-mode score differences ( ∆ =Mode B−Mode A ) enable failure attribution: large positive ∆indicates retrieval
failures; consistently low scores in both modes indicate generation-stage weaknesses.

FAB-Bench A PREPRINT
G Complete Experimental Results
Table 17: DeepSeek-v3.2-Exp: complete results across all context configurations.
Config Fact. Depth Comp. Rel. Ctx.U. Supp. Avg.
(4K, 1K) 0.626 0.554 0.592 0.627 0.783 0.534 0.619
(8K, 2K) 0.632 0.602 0.640 0.661 0.804 0.594 0.656
(10K, 4K) 0.663 0.606 0.651 0.669 0.814 0.609 0.669
(12K, 4K) 0.690 0.640 0.688 0.701 0.824 0.644 0.698
(14K, 4K) 0.744 0.693 0.752 0.740 0.845 0.709 0.747
(16K, 4K) 0.792 0.718 0.783 0.781 0.868 0.735 0.780
(18K, 4K) 0.837 0.755 0.820 0.813 0.884 0.776 0.814
(20K, 4K) 0.865 0.796 0.860 0.850 0.901 0.823 0.849
(24K, 4K) 0.868 0.812 0.880 0.857 0.915 0.839 0.862
(28K, 4K) 0.892 0.811 0.879 0.868 0.916 0.842 0.868
(32K, 4K)0.904 0.838 0.898 0.8750.9120.872 0.883
Table 18: Qwen-Plus: complete results across all context configurations.
Config Fact. Depth Comp. Rel. Ctx.U. Supp. Avg.
(4K, 1K) 0.663 0.662 0.686 0.677 0.805 0.641 0.689
(8K, 2K) 0.665 0.710 0.724 0.704 0.801 0.696 0.717
(10K, 4K) 0.680 0.736 0.764 0.736 0.828 0.724 0.745
(12K, 4K) 0.697 0.749 0.765 0.722 0.823 0.747 0.751
(14K, 4K) 0.723 0.772 0.804 0.760 0.834 0.767 0.777
(16K, 4K) 0.750 0.789 0.809 0.768 0.848 0.770 0.789
(18K, 4K) 0.732 0.783 0.813 0.774 0.844 0.783 0.788
(20K, 4K) 0.778 0.810 0.830 0.798 0.852 0.801 0.812
(24K, 4K) 0.833 0.829 0.861 0.827 0.882 0.835 0.845
(28K, 4K)0.866 0.849 0.894 0.859 0.916 0.857 0.874
(32K, 4K) 0.855 0.829 0.867 0.847 0.889 0.831 0.853
Table 19: Gemini-2.5-Flash: complete results across all context configurations.
Config Fact. Depth Comp. Rel. Ctx.U. Supp. Avg.
(4K, 1K) 0.600 0.374 0.399 0.449 0.669 0.350 0.474
(8K, 2K) 0.643 0.536 0.563 0.583 0.757 0.523 0.601
(10K, 2K) 0.666 0.550 0.595 0.622 0.758 0.622 0.636
(12K, 4K) 0.701 0.641 0.692 0.708 0.819 0.651 0.702
(14K, 4K) 0.743 0.655 0.709 0.726 0.839 0.656 0.721
(16K, 4K) 0.760 0.669 0.733 0.731 0.816 0.676 0.731
(18K, 4K) 0.776 0.710 0.771 0.784 0.847 0.728 0.769
(20K, 4K) 0.833 0.738 0.804 0.808 0.878 0.752 0.802
(24K, 4K) 0.854 0.765 0.827 0.837 0.882 0.790 0.826
(28K, 4K)0.901 0.818 0.886 0.890 0.907 0.852 0.876
(32K, 4K) 0.869 0.775 0.844 0.836 0.889 0.801 0.836
Table 20: Qwen-2.5-72B-Instruct: results with extended context windows up to 128K.
Config Fact. Depth Comp. Rel. Ctx.U. Supp. Avg.
(4K, 1K) 0.511 0.568 0.603 0.589 0.758 0.536 0.594
(8K, 2K) 0.544 0.590 0.629 0.605 0.785 0.560 0.619
(16K, 4K) 0.643 0.661 0.713 0.692 0.798 0.630 0.690
(32K, 4K) 0.794 0.762 0.827 0.812 0.852 0.764 0.802
(64K, 6K) 0.801 0.759 0.820 0.796 0.856 0.738 0.795
(128K, 8K) 0.797 0.766 0.829 0.8120.8630.762 0.805

FAB-Bench A PREPRINT
Table 21: Cross-model comparison at key context configurations.
Model 4K 16K 28K 32K
DeepSeek-v3.2-Exp 0.619 0.780 0.8680.883
Qwen-Plus0.689 0.789 0.8740.853
Gemini-2.5-Flash 0.474 0.731 0.876 0.836
Qwen-2.5-72B 0.594 0.690 — 0.802