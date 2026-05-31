# PennySynth: RAG-Driven Data Synthesis for Automated Quantum Code Generation

**Authors**: Minghao Shao, Nouhaila Innan, Hariharan Janardhanan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique

**Published**: 2026-05-25 08:26:43

**PDF URL**: [https://arxiv.org/pdf/2605.25572v1](https://arxiv.org/pdf/2605.25572v1)

## Abstract
The growing complexity of quantum programming frameworks has exposed a critical limitation in existing large language model (LLM)-based code assistants: general-purpose models hallucinate PennyLane-specific gate names, misplace device configurations, and produce structurally invalid circuits when faced with specialized quantum coding challenges. We present PennySynth, a retrieval-augmented generation framework that addresses this gap by conditioning LLM inference on a curated knowledge base of 13,389 PennyLane instruction-code pairs, built via a three-stage extraction, verification, and deduplication pipeline over official PennyLane repositories, community GitHub sources, and QHack competition archives. PennySynth introduces a code-aware embedding strategy using st-codesearch-distilroberta-base, trained for natural-language-to-code retrieval, increasing average retrieval cosine similarity from 0.45 to 0.726 compared to a general-purpose baseline. Evaluated across 74 challenges spanning three years of the QHack competition (2022, 2023, 2024), PennySynth achieves 64%, 68%, and 52% pass@5 on QHack 2022, 2023, and 2024, respectively, improving over Claude Sonnet 4.6 without retrieval by +28, +25, and +28 percentage points. We further introduce a quantum-adapted CodeBLEU metric that upweights qml.* token patterns and show that structural code similarity and functional correctness capture distinct aspects of quantum code quality. Controlled ablations reveal that code-aware embeddings are the primary driver of retrieval performance, while dataset expansion and source composition provide additional gains when retrieval quality is sufficiently precise.

## Full Text


<!-- PDF content starts -->

PennySynth: RAG-Driven Data Synthesis for
Automated Quantum Code Generation
Minghao Shao1,3, Nouhaila Innan1,2, Hariharan Janardhanan1,2, Muhammad Kashif1,2,
Alberto Marchisio1,2, and Muhammad Shafique1,2
1eBRAIN Lab, Division of Engineering, New York University Abu Dhabi (NYUAD), Abu Dhabi, UAE
2Center for Quantum and Topological Systems (CQTS), NYUAD Research Institute, NYUAD, Abu Dhabi, UAE
3Department of Computer Science and Engineering, NYU Tandon School of Engineering, New York, USA
{shao.minghao, nouhaila.innan, hj2342, muhammadkashif, alberto.marchisio, muhammad.shafique}@nyu.edu
Abstract—The growing complexity of quantum programming
frameworks has exposed a critical limitation in existing large
language model (LLM)-based code assistants: general-purpose
models hallucinate PennyLane-specific gate names, misplace
device configurations, and produce structurally invalid circuits
when faced with specialized quantum coding challenges. We
present PennySynth, a retrieval-augmented generation framework
that addresses this gap by conditioning LLM inference on
a curated knowledge base of 13,389 PennyLane instruction-
code pairs, built via a three-stage extraction, verification, and
deduplication pipeline over official PennyLane repositories,
community GitHub sources, and QHack competition archives.
PennySynth introduces a code-aware embedding strategy using
st-codesearch-distilroberta-base , trained for natural-
language-to-code retrieval, increasing average retrieval cosine
similarity from 0.45 to 0.726 compared to a general-purpose
baseline. Evaluated across 74 challenges spanning three years of
the QHack competition (2022, 2023, 2024), PennySynth achieves
64%, 68%, and 52% pass@5 on QHack 2022, 2023, and 2024,
respectively, improving over Claude Sonnet 4.6 without retrieval
by +28, +25, and +28 percentage points. We further introduce a
quantum-adapted CodeBLEU metric that upweights qml. *token
patterns and show that structural code similarity and functional
correctness capture distinct aspects of quantum code quality.
Controlled ablations reveal that code-aware embeddings are the
primary driver of retrieval performance, while dataset expansion
and source composition provide additional gains when retrieval
quality is sufficiently precise.
Index Terms—Quantum Computing, PennyLane, Retrieval-
Augmented Generation, Code Generation, QHack, Quantum
Circuit Synthesis, Large Language Models
I. INTRODUCTION
Quantum computing is progressing from theoretical foun-
dations toward increasingly practical software and hardware
ecosystems. Recent hardware advances have expanded the scale
of available quantum processors, while software frameworks
such as PennyLane [1], Qiskit [2], and related quantum
programming toolkits have made quantum and hybrid quantum-
classical programming more accessible to researchers and
developers. At the same time, quantum software development
remains difficult because programs must correctly combine cir-
cuit construction, device configuration, differentiable execution,
measurement design, and version-specific API behavior [3].
These challenges create a growing need for intelligent coding
tools that can assist developers in writing correct, efficient, and
framework-compatible quantum programs.Large language models (LLMs) have substantially improved
general-purpose code generation [4], [5], and recent work has
begun to extend these capabilities to quantum programming.
For Qiskit, domain-specific models and tools, including IBM’s
Qiskit Code Assistant, have been developed using curated
quantum code corpora [6], while Qiskit HumanEval [7]
provides a dedicated benchmark for executable Qiskit code
generation. Beyond Qiskit, QuanBench [8] evaluates LLM-
based quantum code generation across a broader set of
quantum programming tasks. For PennyLane, PennyLang [9],
PennyCoder [10], and QHackBench [11] establish datasets,
models, and competition-grade evaluation settings for LLM-
assisted PennyLane programming. The AWS Braket RAG
system [12] further shows that retrieval-augmented generation
can support quantum program development. Despite this
progress, generating correct quantum code remains challenging,
particularly when outputs are evaluated through execution rather
than textual similarity.
While recent efforts show that LLMs can support quantum
code generation, reliable execution remains difficult, and two
main bottlenecks continue to limit progress. First, quantum
programming is a low-resource domain: PennyLane-specific
code represents only a small fraction of the training data
used by general-purpose LLMs, which limits their coverage of
framework-specific operations, device declarations, templates,
differentiable workflows, and measurement patterns. Second,
PennyLane evolves rapidly, with API interfaces, gate naming
conventions, device usage rules, and supported operations
changing across releases. As a result, code samples that were
valid under earlier versions may become outdated, increasing
the risk of unsupported operations, incorrect device usage,
missing imports, or structurally invalid circuits [11]. These
limitations suggest that model scale alone is not sufficient;
LLMs also need access to verified, up-to-date, and framework-
specific code examples at inference time.
To address these limitations, we proposePennySynth,
a data synthesis and retrieval framework for LLM-assisted
PennyLane code generation. PennySynth operates in two
stages. In the offline stage, it automatically extracts, verifies,
and deduplicates PennyLane code from GitHub repositories,
official documentation, and competition archives using AST-
based function isolation, multi-layer quality verification, andarXiv:2605.25572v1  [cs.CL]  25 May 2026

MinHash-based deduplication. In the online stage, it retrieves
task-relevant examples at inference time using a code-aware
embedding model trained for natural-language-to-code retrieval.
The resulting dataset,PennySynth-13K, contains 13,389
verified instruction-code pairs. Evaluated on QHack challenges,
PennySynth achieves up to 68% pass@5 on QHack 2023 with
Claude Sonnet 4.6 as the base generator, corresponding to
a +25 percentage point improvement over the same model
without retrieval. Fig. 1 illustrates the full pipeline.
QHack ChallengesDescription + template74 tasks · 2022 / 23 / 24QHackBenchPennySynthKnowledge BasePennySynth-13K13,389 instruction–code pairsGitHub corpus + Foundational LayerRoBERTa embed · sim = 0.726ChromaDB vector indexRAG Inference Pipeline①  Query expansion (LLM)②  Top-5 retrieval · τ = 0.60③  RAG / base prompt select④  LLM generation (Claude 4.6)⑤  Execute + auto-fix (T = 2)
✓Evaluated Outputpass@1 · partial creditCodeBLEU(quantum-adapted)ROUGE-L64% QHack 202268% QHack 202352% QHack 2024
Fig. 1: Overview of the PennySynth framework for automated Penny-
Lane code generation. The system constructs a verified instruction-
code knowledge base, retrieves task-relevant examples for each
challenge, and uses retrieval-augmented generation with iterative
execution-based repair to produce final solutions.
Our main contributions are:
1)PennySynth, an automated data synthesis framework
for PennyLane quantum code that extracts, verifies, and
deduplicates instruction-code pairs from heterogeneous
sources, andPennySynth-13K, the resulting dataset of
13,389 AST-verified pairs released to support future
research.
2)Code-aware RAG pipelinewith query expansion,
relevance threshold, and selective context instructions,
improving pass@5 over Claude Sonnet 4.6 without
retrieval by up to +28.2 percentage points.
3)Quantum-adapted CodeBLEU: upweighting qml. *to-
kens; first systematic CodeBLEU evaluation on generated
PennyLane code vs. official QHack reference templates.
4)Systematic multi-model comparisonacross seven state-
of-the-art LLMs (Claude Sonnet 4.6, GPT-5.5, Gemini
2.5 Pro, Qwen3-235B-A22B, GLM-5.1, DeepSeek-V3,
LLaMA 3.1-8B), with ablations isolating the effects of
the embedding model, dataset size, and source composi-
tion.
II. BACKGROUND ANDRELATEDWORK
A. LLMs for Quantum Code Generation
LLM-based quantum code assistance has gained increasing
attention across several quantum programming frameworks.For Qiskit, a domain-specific model based on the Granite
architecture was fine-tuned on a curated corpus filtered to
include post-2023 sources, reducing the risk of deprecated API
usage [6]. Qiskit HumanEval [7] further provides more than
100 hand-curated quantum programming tasks across eight
categories, ranging from basic circuit construction to more
complex algorithm implementation. These efforts establish
a foundation for standardized evaluation of LLM-generated
Qiskit code.
For PennyLane, PennyLang [9] provides 3,347 PennyLane-
specific code samples collected from textbooks, documentation,
and open-source repositories. When used within a RAG
pipeline, PennyLang increased Qwen 7B’s success rate from
8.7% to 41.7%, although stronger commercial models showed
smaller gains from retrieval augmentation. PennyCoder [10]
fine-tunes LLaMA 3.1-8B with LoRA for local deployment
without external API dependencies, achieving 44.3% accuracy
on a 264-task internal benchmark and outperforming its RAG-
only baseline at 40.2%. However, its evaluation remains limited
to in-domain tasks drawn from the PennyLang distribution.
QHackBench [11] addresses this limitation by introducing a
reproducible benchmark based on real QHack competition
challenges and incorporating iterative refinement to correct
generated solutions. Our work builds on this direction by
introducing a larger verified PennyLane corpus and a code-
aware retrieval pipeline, and evaluating multiple recent LLMs
across QHack 2022, 2023, and 2024.
B. Retrieval-Augmented Generation for Code
Retrieval-Augmented Generation (RAG) conditions language
model outputs on external examples or documents retrieved
at inference time [13]. This approach reduces reliance on
parametric memory alone and can provide models with task-
specific context that may be absent or under represented
in pretraining. RAG has also been extended through query
expansion, where an LLM generates enriched search queries
or pseudo-documents to improve retrieval quality [14], and
through reranking methods that rescore retrieved candidates
for finer-grained relevance estimation [15].
For code generation, retrieving semantically related code
snippets or documentation fragments has been shown to
improve generation accuracy compared with standalone
LLMs [16]. In quantum computing, the AWS Braket RAG sys-
tem [12] demonstrated that retrieved documentation can support
quantum program development. However, this system focuses
on Braket-oriented workflows and does not target PennyLane
competition-style challenges. More broadly, Graph RAG [17]
organizes retrieved knowledge through graph structures, which
can capture relations among entities and concepts. Such graph-
based retrieval may be useful for quantum programming, where
gates, templates, devices, measurements, and differentiation
methods are strongly interconnected. In this work, we focus on
a flat code-aware vector retrieval pipeline and leave PennyLane-
specific graph retrieval as a future extension.

①Offline: knowledge base construction
Official PL docs
Community GitHub
Archives & docs
Stage 1: ExtractionAST function isolation76.9% ret. · 11,643
Stage 2: Verification4-layer quality check97.4% ret. · 11,344Stage 3: DeduplicationMinHash LSH 70% thr.~10K unique pairs
Prompt generationGemini 2.0 Flash API95% Top-1 accuracy
Code-aware embeddistilroberta 768-dimavg sim → 0.726
ChromaDB index13,389 instr-code pairsCosine sim · persistentPennySynth-13kknowledge baseOfficial PL repos: 14.4% · Unofficial GitHub: 61.6% · Archives: 24.0%Total: 13,389 verified instruction-code pairs②Online: inference & evaluation (per QHack challenge)QHackchallenge
①Query expansionLLM rewrites desc asPennyLane query qʹ
②RetrievalEmbed qʹ (768-dim)Top-5 by cosine sim
③Threshold checkmax sim ≥ 0.60 → RAG promptmax sim < 0.60 → Base prompt
④LLM generationClaude Sonnet 4.6temp=0.7 · max_tokens=3000
⑤ExecuteRun QHack test casesCompare stdout vs expected
⑥Auto-fix looperror + code → LLM≤ 2 retries
⑦Metricspass@1 · pass@5 CodeBLEU· ROUGE-L halluc. rate↻Iterative
PASSpass@1 = 1.0
PARTIALcredit > 0
FAIL / ERRORpass@1 = 0.0vector lookupFig. 2: PennySynth full system architecture.Offline:three-stage knowledge base construction from heterogeneous PennyLane sources, followed
by code-aware embedding into ChromaDB.Online:per-challenge inference via query expansion, top-5 retrieval, similarity threshold check,
LLM generation, and iterative auto-fix feedback loop.
C. Evaluation Metrics
BLEU [18] measures n-gram precision with a brevity
penalty and remains widely used in code generation evaluation.
However, it is limited for code because it is sensitive to
surface-level token overlap and does not account well for
variable renaming, control-flow structure, or semantically
equivalent implementations. ROUGE-L [19] measures the
longest common subsequence between generated and reference
outputs, providing an additional view of textual similarity.
CodeBLEU [20] extends BLEU by combining token-level
matching with structural and semantic code information. It
includes token BLEU, weighted BLEU, AST matching, and
dataflow matching, typically combined with equal weights.
However, standard CodeBLEU does not explicitly distinguish
framework-specific operations from generic tokens, even
though API calls such as qml. *operations are central to
the correctness of PennyLane programs. To better reflect
PennyLane code structure, we adapt CodeBLEU in two ways:
first,qml. *tokens are upweighted by 3 ×in the weighted
BLEU component; second, the dataflow component is replaced
with a quantum-oriented Jaccard match over extracted qml. *
gate names, device declarations, and measurement return types.
This provides a more domain-aware similarity measure for
comparing generated PennyLane code with reference solutions.
III. METHODOLOGY
Fig. 2 presents the overall PennySynth pipeline, which
consists of two main stages: anofflineknowledge-base con-
struction stage and anonlineinference-and-repair stage. In
the offline stage, PennySynth builds a verified repository
of PennyLane instruction-code pairs collected from official
sources, community repositories, and QHack archives. In
the online stage, the system retrieves task-relevant examples
for each challenge, adds them to the generation prompt
when retrieval confidence is sufficiently high, and applies anexecution-based repair loop when the generated code fails the
test cases.
A. PennySynth-13K Dataset
1) Data Collection:The quality of retrieval-augmented
code generation depends on the coverage, correctness, and
relevance of the retrieval corpus. To construct a knowledge
base that captures both canonical PennyLane usage and realistic
community coding patterns, we aggregate source code from
three types of sources.
1)Official PennyLane sources:We collect code from offi-
cial PennyLane repositories and documentation examples,
which provide trusted reference implementations of core
PennyLane functionality.
2)Community GitHub repositories:We gather a large
community corpus from GitHub by searching for
PennyLane-related keywords such as pennylane and
qml, yielding more than 285,000 candidate records
before filtering.
3)QHack competition archives:We include public chal-
lenge descriptions and template code from the QHack
2022, 2023, and 2024 archives. These entries are valuable
because they are aligned with the competition-style
PennyLane programming tasks used in our evaluation.
Since user-submitted solutions may appear in public
repositories, we exclude solution-like files and files
matching benchmark answer patterns from the knowl-
edge base. This prevents direct answer retrieval during
evaluation and ensures that performance gains come from
task-relevant retrieved examples rather than memorized
solutions.
This source selection is designed to balance correctness,
diversity, and benchmark relevance. Official sources provide
trusted API usage, community repositories expand coverage
of real-world PennyLane coding patterns, and QHack archives

provide examples aligned with competition-style synthesis
tasks.
2) Dataset Profiling:Table I summarizes the composition of
PennySynth-13K across three source groups: official PennyLane
repositories, unofficial GitHub repositories, and QHack/tutorial
archives. The final dataset contains 13,389 instruction-code
pairs. Unofficial GitHub repositories contribute the largest share,
with 8,245 pairs, corresponding to 61.6% of the corpus. Official
PennyLane repositories contribute 1,934 pairs, corresponding
to 14.4%. These two GitHub-derived sources form a 10,179-
pair corpus before archive augmentation. The remaining 3,210
pairs, corresponding to 24.0%, come from QHack and tutorial
archives.
TABLE I: PennySynth-13K Dataset Composition by Source
Category Source Samples %
Official Official PennyLane repos 1,934 14.4%
Community Unofficial GitHub repos 8,245 61.6%
Archives QHack & Tutorial archives 3,210 24.0%
Total (after deduplication + archives) 13,389 100%
This composition reflects the role of each source group in
the final corpus. Unofficial GitHub repositories provide broad
coverage of real-world PennyLane coding patterns, official
repositories provide trusted framework-level examples, and
QHack/tutorial archives add benchmark-aligned prompts and
template structures. This breakdown also motivates the dataset
ablations reported later, where we compare the 10,179-pair
GitHub-derived corpus against the full 13,389-pair PennySynth-
13K corpus and further analyze the contribution of each source
category.
3) Three-Stage Dataset Construction Pipeline:We transform
raw source files into verified instruction-code pairs through a
three-stage pipeline: extraction, verification, and deduplication.
The pipeline is designed to preserve executable structure and
PennyLane-specific operations while removing unrelated helper
code, invalid transformations, and near-duplicate entries.
Stage 1: Extraction.We first retrieve complete Python
source files using direct GitHub URL access. Preliminary
trials with pre-chunked CSV-based sources led to severe
fragmentation and retained only 0.6% of usable code, mainly
because function boundaries, imports, and surrounding execu-
tion context were often lost. We therefore operate directly on
full source files and perform structural parsing.
Function-level extraction is performed using Python AST
analysis. Each file is parsed into an abstract syntax tree, and
candidate functions are retained when they contain PennyLane-
specific operations or quantum-programming structure. This
allows the pipeline to preserve syntactic scope, argument lists,
return statements, and internal control flow while filtering out
unrelated helper functions. After this stage, 11,643 functions
are retained, corresponding to a 76.9% extraction retention
rate.
Stage 2: Verification.Extracted functions are passed through
a four-layer verification pipeline to ensure that the resulting
code remains syntactically valid, structurally coherent, and con-sistent with the original quantum operations. The verification
layers are:
1)Syntax validation:each candidate is checked using
Python compilation.
2)Import validation:required libraries and quantum-
programming dependencies must remain present.
3)Quantum-operation preservation:gate count changes
must remain within 20%, measurements preserved, and
qml. *call changes limited to at most 50%.
4)Semantic structure preservation:the function must
preserve its return structure and intended output path.
We adopt a conservative fallback policy: if any verification
step fails, the original extracted code is retained instead of
the transformed version. After verification, 11,344 functions
remain, corresponding to a 97.4% verification retention rate.
Stage 3: Deduplication.We apply MinHash LSH [21]
with 128 permutations and a 70% similarity threshold to
remove near-duplicate samples from the verified GitHub-
derived corpus. This stage produces 10,179 deduplicated
GitHub-derived instruction-code pairs. We then append 3,210
QHack and tutorial archive entries, resulting in the final
PennySynth-13Kcorpus of 13,389 verified instruction-code
pairs.
4) Prompt Generation:Each code sample in PennySynth-
13K is paired with a natural-language instruction to support
natural-language-to-code retrieval. Since many extracted func-
tions do not contain an explicit task description, we generate
instructions automatically using the Gemini API under a
constrained template. The template requires a short imperative
description of 20-40 words and asks the model to mention the
main PennyLane operations, device or wire information when
available, and the expected return type.
To assess whether the generated instructions are suitable for
retrieval, we perform an internal retrieval consistency check.
Each generated instruction is used as a query against the
constructed corpus, and the retrieval is considered correct
when the original code sample appears among the retrieved
candidates. Under this evaluation, the generated instructions
achieve 95% Top-1 retrieval accuracy and 100% Top-5 retrieval
accuracy, indicating that the instructions preserve enough
task-specific information to recover their corresponding code
samples.
B. Embedding Strategy
The retrieval stage requires an embedding model that
can align natural-language task descriptions with Pen-
nyLane code examples. We evaluate two configurations
in the ablation study. TheMiniLM configurationuses
all-MiniLM-L6-v2 , a 384-dimensional general-purpose
sentence embedding model, serving as a lightweight se-
mantic retrieval baseline. ThePennySynth configuration
uses st-codesearch-distilroberta-base , a 768-
dimensional model trained on CodeSearchNet for natural-
language-to-code retrieval, making it better suited for matching
QHack problem descriptions to instruction-code pairs.

Given an expanded query q′and a candidate document di,
retrieval similarity is computed using cosine similarity between
their embeddings:
s(q′, di) =eq′·ei
∥eq′∥ · ∥e i∥,(1)
where eq′represents the embedding of the expanded challenge
query, and eiis the embedding of the i-th instruction-code
pair in the knowledge base.
In our retrieval consistency evaluation, MiniLM yields an
average cosine similarity of approximately ¯s≈0.45 , while the
PennySynth configuration increases this value to ¯s= 0.726 .
We therefore use the PennySynth configuration as the default
retrieval backbone in the main experiments.
C. Retrieval and Prompt Design
Given a challenge description q, the system first performs
query expansion to produce a more focused PennyLane-
oriented query q′. The expanded query is embedded using the
selected retrieval backbone and compared against the embedded
instruction-code pairs in the ChromaDB knowledge base using
Eq. (1).
The system retrieves the top- k= 5 most similar examples. If
the maximum similarity score among the retrieved candidates
falls below the threshold τ= 0.60 , the system falls back
to base generation to preventcontext contamination, i.e., the
inclusion of irrelevant or weakly related examples that may
mislead the model. When the threshold is met, the prompt is
augmented with the retrieved examples and an explicit selective-
context instruction:“If retrieved examples are not relevant to
this challenge, ignore them and rely on your own PennyLane
knowledge. ”
After generation, the produced code is executed against the
challenge test cases. If execution fails, the model enters an auto-
fix loop that provides the previous code, execution feedback,
and the original challenge to a repair prompt, with at most
T= 2 repair attempts. Algorithm 1 summarizes this procedure.
D. Quantum-Adapted CodeBLEU
We build on CodeBLEU to evaluate generated PennyLane
code while accounting for quantum-specific API usage. Stan-
dard CodeBLEU combines token-level similarity, weighted
n-gram matching, AST matching, and dataflow matching, but
it does not explicitly distinguish PennyLane operations from
generic Python tokens. This distinction is important because
qml. *operations, device declarations, and measurement state-
ments directly affect circuit behavior and execution correctness.
We introduce two quantum-specific modifications. First,
token upweighting: all qml. *tokens are repeated 3×before
computing the weighted n-gram component, giving greater
emphasis to PennyLane-specific operations. Second, we replace
the standard dataflow component with aquantum dataflow
matchbased on Jaccard similarity over the set K(c) extracted
from a code sample c, where K(c) contains qml. *gate names,
device types, and measurement return types:
DF(h, r) =|K(h)∩K(r)|
|K(h)∪K(r)|,(2)Algorithm 1PennySynth Inference Pipeline
Require: Challenge q, knowledge base D, threshold τ, max
fixesT
Ensure:Solutionc∗, resultr∗
1:q′←LLM expand (q)
2:R ←Top-k-retrieve(Enc(q′),D)
3:p←BasePrompt(q) ifmax d∈Rs < τ else
RAGPrompt(q,R)
4:c0←ExtractCode(LLM(p));r 0←Execute(c 0)
5:fort= 1, . . . , Tdo
6:ifr t−1.passthen break
7:end if
8:c t←ExtractCode(LLM fix(ct−1, rt−1, q,R))
9:r t←Execute(c t)
10:end for
11:returnc t, rt
where his the code generated by the LLM, also referred to as
the hypothesis, and ris the QHack challenge template used as
the reference. Thus, DF(h, r) measures the overlap between the
quantum-relevant operations, device types, and measurement
structures extracted from the generated code and the reference
template.
IV. EXPERIMENTSETTINGS
A. Benchmark
We evaluate PennySynth on QHack, a PennyLane coding
competition covering three benchmark years: 25 challenges
from 2022, 28 from 2023, and 21 from 2024, for a total of 74
challenges. Following QHackBench [11], we use these tasks
as a standardized benchmark for executable PennyLane code
generation. Each challenge provides a problem statement, a
code template to complete, input-output specifications, and test
functions for validating submitted solutions. Generated code is
considered correct only when it passes the corresponding tests
under this execution-based evaluation protocol.
B. Systems
We evaluate six recent LLMs as base generators without
retrieval to establish strong baselines:Claude Sonnet 4.6[22],
GPT-5.5[23],Gemini 2.5 Pro[24],Qwen3-235B-A22B[25],
GLM-5.1[26], andDeepSeek-V3[27].
C. Metrics
We report four evaluation metrics.pass@5measures whether
at least one of five independently generated solutions passes
all test cases for a given challenge. We also reportpass@1for
single-generation evaluation of the RAG systems.Partial credit
measures the fraction of test cases passed within each challenge.
CodeBLEUandROUGE-Lmeasure code similarity against
the available QHack 2022 reference templates, with CodeBLEU
evaluated using the quantum-adapted variant described in
Section III-D .Hallucination ratemeasures the percentage

of generated solutions containing qml. *calls absent from a
curated PennyLane operation whitelist.
D. Implementation
All generation experiments use a temperature of 0.7 and
a maximum generation length of 3,000 tokens. Retrieval is
performed with top- k=5, while a relevance threshold of τ=0.60
is used to exclude weakly related examples that may introduce
noise into the prompt. The auto-fix loop permits up to T=2
retry attempts.
V. RESULTS ANDDISCUSSION
A. Multi-Model Baseline Comparison
We first evaluate PennySynth on QHack challenges from
2022, 2023, and 2024 and compare it against six recent base
LLMs without retrieval. This evaluation establishes the main
performance gap between general-purpose code generation and
retrieval-augmented PennyLane code generation. As shown in
Fig. 3, PennySynth achieves the strongest performance across
all three benchmark years, reaching 64.0%, 68.0%, and 52.0%
pass@5 on QHack 2022, 2023, and 2024, respectively.
(%)
0%20%40%60%
36%
32%
24% 24%
20%
8%64%
43%
32%39%
29% 29%
4%68%
24% 24% 24%
10%14%52%PennySynth (RAG)
Base Claude
Sonnet 4.6GPT-5.5Qwen3
235BGLM -5.1DeepSeek
V3Gemini
2.5 ProPennySynth
(RAG+Claude )QHack 2022 QHack 2023 QHack 2024
Fig. 3: pass@5 across six base LLMs and PennySynth (RAG+Claude)
on QHack 2022, 2023, and 2024. PennySynth substantially outper-
forms all base LLMs.
Among the base models, Claude Sonnet 4.6 obtains the best
or tied-best results across the three years, with pass@5 scores
of 36.0%, 42.9%, and 23.8%. GPT-5.5 is competitive on QHack
2022, while Qwen3-235B-A22B approaches Claude on QHack
2023. However, all base models remain far below PennySynth.
Relative to Claude Sonnet 4.6 without retrieval, PennySynth
improves pass@5 by +28.0, +25.1, and +28.2 percentage points
on QHack 2022, 2023, and 2024, respectively. These gains
indicate that retrieval from PennySynth-13K provides task-
specific PennyLane patterns that are not consistently recovered
from the model’s internal knowledge alone.
To position PennySynth against the closest prior PennyLane-
oriented code-generation baseline, we compare it with QHack-
Bench, which also reports retrieval-augmented results on QHack
challenges. On the overlapping QHack years, PennySynth
achieves 68.0% pass@5 on QHack 2023 and 52.0% on QHack
2024, compared with 60.7% and 33.3% reported by QHack-
Bench RAG [11]. This corresponds to gains of +7.3 and +18.7
percentage points, respectively. These improvements indicate
that the proposed PennySynth-13K dataset and retrieval pipelineprovide stronger support for competition-style PennyLane code
generation than the prior QHackBench RAG setting.
B. PennySynth RAG Performance
We next evaluate PennySynth-13K as a retrieval component
across different generator models. Table II reports pass@1 and
pass@5 results for seven LLMs on QHack 2022-2024 using
the same retrieval pipeline. This analysis examines whether
retrieved PennyLane examples improve generation across model
families and motivates the choice of Claude Sonnet 4.6 as the
main generator for the remaining experiments.
TABLE II: Performance of RAG with PennySynth-13k across 7 cutting-
edge models (pass@1 / pass@5) on QHack challenges from 2022 to
2024.
Model2022 2023 2024
1 5 1 5 1 5
Claude Sonnet 4.6 56% 64% 61% 68% 43%52%
GPT-5.5 64% 68%50% 57% 29% 43%
Qwen3-235B 52% 64% 50% 61% 43% 53%
GLM-5.1 20% 48% 11% 32% 5% 14%
DeepSeek-V3 32% 48% 36% 36% 19% 33%
Gemini 2.5 Pro 8% 8% 4% 4% 0% 0%
LLaMA 3.1-8B 12% 20% 11% 32% 0% 0%
The results indicate that PennySynth-13K is most effective
when paired with strong generator models. Claude Sonnet
4.6 provides the most stable performance across the three
benchmark years, achieving 64%, 68%, and 52% pass@5 on
QHack 2022, 2023, and 2024, respectively. GPT-5.5 obtains
the best result on QHack 2022, while Qwen3-235B achieves
the highest pass@5 on QHack 2024. These findings show that
the retrieval pipeline is not tied to a single model, although
its effectiveness depends on the generator’s ability to adapt
retrieved examples to the target challenge.
The results also reveal a clear difference between strong
and weaker generators. GLM-5.1 and DeepSeek-V3 obtain
moderate but less stable performance, while Gemini 2.5 Pro and
LLaMA 3.1-8B remain weak across most settings. This suggests
that retrieval alone is not sufficient for reliable PennyLane
code generation. The model must still follow strict output
formats, use the PennyLane API correctly, and modify retrieved
examples instead of copying incompatible code. Based on its
stable performance, Claude Sonnet 4.6 is used as the main
PennySynth generator in the following analyses.
C. Code Quality Analysis
Beyond functional correctness, we evaluate whether gener-
ated solutions preserve the expected structure of the official
QHack reference code. Although official reference code is
available for all benchmark years, we focus this code-level
analysis on QHack 2022 to provide a controlled comparison
over a fixed set of challenges. In this setting, the reference
templates define the required structure, while the model must
complete the missing functions correctly. Table III reports

CodeBLEU, ROUGE-L, AST match, and quantum dataflow
match on the selected QHack 2022 challenges.
TABLE III: Code quality against official QHack reference templates on
11 selected QHack 2022 challenges. PennySynth and RAG-MiniLM
use Claude Sonnet 4.6 as the generator, while Base Gemini uses
Gemini 2.0 Flash. CB = CodeBLEU, RL = ROUGE-L, AST = AST
match, and DF = quantum dataflow match.
System (Generator Model) CB↑RL↑AST↑DF↑
Gemini 2.0 Flash (no RAG) 0.531 0.569 0.685 0.548
RAG-MiniLM (Claude Sonnet 4.6) 0.495 0.473 0.688 0.573
Claude Sonnet 4.6 (no RAG) 0.480 0.448 0.696 0.575
PennySynth (Claude Sonnet 4.6) 0.522 0.5290.688 0.570
The results show that similarity to the official reference
template does not always align with functional correctness.
Gemini 2.0 Flash obtains the highest CodeBLEU score, but its
functional pass rate remains low, indicating that a generated
solution can preserve much of the expected template structure
while still completing the missing functions incorrectly or
producing incorrect numerical outputs. In contrast, PennySynth
improves over Claude Sonnet 4.6 without retrieval in both
CodeBLEU and ROUGE-L, increasing CodeBLEU from 0.480
to 0.522 and ROUGE-L from 0.448 to 0.529. These gains
suggest that retrieval helps the generator use more appropriate
PennyLane tokens, function patterns, and API-level expressions
within the required template.
However, the AST and quantum dataflow scores remain
close across Claude-based systems. This indicates that retrieval
mainly improves local code formulation and API alignment
rather than substantially changing the high-level circuit structure
constrained by the template. Therefore, code-similarity metrics
should be interpreted as supporting evidence rather than
substitutes for functional evaluation. For this reason, pass@k
remains the primary metric in our benchmark.
D. Ablation Studies
We conduct ablation experiments on QHack 2022 to identify
which components of PennySynth-13K contribute most to
retrieval performance. The analysis first separates the effect
of the embedding model from the effect of dataset size, then
examines the contribution of each source category in the full
corpus.
1) Embedding Model and Dataset Composition:To disen-
tangle the effects of the code-aware embedding model and the
expanded dataset, we conduct a controlled 2 ×2 ablation on
QHack 2022 while keeping all other hyperparameters fixed.
Table IV compares four configurations across two embedding
models and two corpus sizes.
Three findings emerge. First, the code-aware embedding is
the primary performance driver: switching from MiniLM to the
PennySynth embedding model delivers +4 pp at the smaller
dataset size and +8 pp at the full dataset, confirming that cross-
modal code retrieval quality gates all downstream gains. Second,
dataset scale is a conditional contributor: expanding from 10K
to 13K improves pass@1 only under the code-aware embedding
(+4 pp), while providing no benefit under the general-purposeTABLE IV: Ablation study on QHack 2022 (25 challenges).
MiniLM = all-MiniLM-L6-v2 (384-dim); PennySynth emb.
=st-codesearch-distilroberta-base (768-dim). Dataset
sizes: Community = 10,179 pairs (unofficial + official GitHub); Full =
13,389 pairs (+ QHack archives). See Table V for source breakdown.
Embedding Dataset pass@1 Data effect
MiniLM (384-dim) Community (10K) 48%+0 ppMiniLM (384-dim) Full (13K) 48%
PennySynth emb. (768-dim) Community (10K) 52%+4 ppPennySynth emb. (768-dim) Full (13K) 56%
Embedding effect (10K) — +4 pp
Embedding effect (13K) — +8 pp
embedding. This occurs because MiniLM cannot distinguish
between the additional competition-style entries and existing
similar entries, introducing retrieval noise that cancels coverage
gains. Third, the two factors are synergistic: the combined
effect (+8 pp) exceeds the sum of individual contributions, as
the code-aware embedding is better positioned to exploit the
structural diversity introduced by the larger corpus.
Challenge-level analysis further reveals that although
MiniLM achieves identical pass@1 at both dataset sizes, the
two configurations solve different challenges: 7 are solved
by both, 3 only by the smaller dataset (algorithm challenges
favoring focused retrieval), and 4 only by the larger dataset
(chemistry and QML challenges benefiting from broader
coverage). The union pass rate of 14/25 (56%) matches the
PennySynth result, confirming that dataset expansion under
MiniLM trades precision for coverage without net gain.
2) Dataset Profiling:To assess the contribution of each
source category in PennySynth-13K, we evaluate individual
and cumulative corpus configurations using the PennySynth
embedding model. Table V reports the contribution of unofficial
GitHub code, official PennyLane repositories, and QHack/tuto-
rial archives.
TABLE V: Dataset profiling ablation on QHack 2022 (25 challenges)
using PennySynth embedding. Individual source rows show each
corpus in isolation; cumulative rows show incremental gains. Gain is
relative to the previous cumulative row.
Corpus Configuration Size pass@1 Gain
Individual sources (isolated)
Unofficial GitHub only 8,245 40% —
Official PennyLane only 1,934 48% —
QHack archives only 3,210 48% —
Cumulative (incremental addition)
Unofficial GitHub only 8,245 40% —
+ Official PennyLane repos 10,179 52% +12 pp
+ QHack archives 13,389 56% +4 pp
This presents both individual and cumulative source contri-
butions. When evaluated in isolation, unofficial GitHub (40%),
official PennyLane repos (48%), and QHack archives (48%)
each provide moderate standalone performance, confirming
that no single source alone is sufficient. The cumulative
analysis reveals a clear incremental structure: unofficial GitHub
alone achieves 40%, adding official PennyLane repositories
(+1,934 entries) produces the largest single gain of +12 pp

(40% →52%), and adding QHack and tutorial archives
contributes a further +4 pp (52%→56%).
The disproportionate +12 pp gain from official repositories,
despite their smaller size (1,934 vs 8,245 unofficial entries),
confirms that source quality and canonical API correctness
matter more than raw corpus scale. Official repositories provide
version-consistent, idiomatic PennyLane implementations that
directly address the hallucination and API misuse failure modes
identified in Table VI. The synergistic effect, where the full
13K corpus outperforms any individual source, validates the
deliberate multi-source design of PennySynth-13K.
TABLE VI: Error profile on QHack 2024 as percentage of each
system’s total failures. Base Claude: n= 11 failures, PennySynth:
n= 12failures.
Category Base PennySynth∆
formatting_failure45% 42%−3%
hallucination27% 33% +6%
reasoning_error18% 8%−10%
api_misuse9% 17% +8%
timeout0% 0% 0%
E. Failure Analysis
After the QHack 2022 code-quality and ablation analyses,
we examine the remaining failures on QHack 2024 to better
understand where retrieval still falls short. Table VI reports
the error distribution for Claude Sonnet 4.6 without retrieval
and PennySynth. Errors are grouped into formatting failures,
hallucinations, reasoning errors, API misuse, and timeouts.
Formatting failures dominate both profiles at 45% and
42%, caused by non-compliance with QHack’s strict template
structure. The key shift introduced by retrieval augmentation
is a trade-off between reasoning errors and context-induced
errors: PennySynth reduces reasoning errors from 18% to 8%
while hallucination increases from 27% to 33% and API misuse
rises from 9% to 17%. Retrieved context occasionally surfaces
gate names from older PennyLane versions or incompatible
function signatures, introducing the same mechanism that
supplies correct domain-specific patterns also surfaces incorrect
ones at moderate similarity.
This indicates that retrieved context can introduce outdated
gate names, incompatible PennyLane syntax, or function
signatures that do not match the target challenge. Thus, the
same mechanism that provides useful domain-specific patterns
can also introduce errors when the retrieved example is only
partially aligned with the task.
No timeout failures are observed for either system, sug-
gesting that the main limitations are not execution cost but
formatting, API compatibility, and challenge-specific reasoning.
F . Dataset Pipeline Case 1: Ground Transformation
We trace Entry #32 from PennySynth-13K through every
stage of the pipeline, showing the actual prompts, LLM
responses, and transformation decisions at each step. All outputs
were captured by running the pipeline scripts on the actual
dataset entry.Stage 0: Raw GitHub File:The GitHub Search API
returns a CSV of PennyLane-related Python files. The pipeline
fetches raw file content via GitHub URL. For Entry #32, the
source is quantum_classifier.py , a 36-line community
repository file containing 7 functions:
✗5 non-quantum helpers: normalize_data ,
plot_loss,accuracy,train_epoch,forward
✓1 quantum function:quantum_circuit
Stage 1: AST Extraction (stage1.py) : The AST parser
scans all 7 functions and classifies each using the lenient
retention strategy. Functions containing any qml. *call are
classified as direct ; the remaining 5 helpers are rejected.
The extracted function:
Listing 1: Entry #32 Extracted quantum function (pre-modernization)
defquantum_circuit(inputs, weights):
qml.templates.AngleEmbedding(inputs,
wires=range(n_qubits))
qml.templates.
StronglyEntanglingLayers(weights,
wires=range(n_qubits))
return[qml.expval(qml.PauliZ(i))
foriinrange(n_qubits)]
Extraction result:2 of 7 functions retained (71% rejection of
non-quantum code).
Stage 2: LLM Transformation (Stage2_Verified.py) :
The extracted code is sent to claude-3.5-haiku via
OpenRouter. The system prompt instructs the model to fix
deprecated APIs while preserving all quantum operations:
♂robotSYSTEM
You are a PennyLane code quality assistant. Fix deprecated APIs
only. Return clean Python code with no markdown fences.
/userUSER
Fix this PennyLane code by replacing deprecated
qml.templates. *calls with their modern equivalents.
Rules:nosep, leftmargin=1.2em
1)Replace qml.templates.AngleEmbedding→
qml.AngleEmbedding
2)Replaceqml.templates.Strongly-
EntanglingLayers→
qml.StronglyEntanglingLayers
3)Keep ALL other code identical
4)Add missing imports at the top
5)Return ONLY the fixed Python code
Code:[function body]
Claude Haiku responds with the modernized code:
Listing 2: Entry #32 Modernized output from Claude Haiku
importpennylaneasqml
defquantum_circuit(inputs, weights):
qml.AngleEmbedding(inputs, wires=←-
range(n_qubits))

qml.StronglyEntanglingLayers(weights,
wires=range(n_qubits))
return[qml.expval(qml.PauliZ(i))
foriinrange(n_qubits)]
Operation preservation check:
qml.templates. * 2→0✓ Modern
API0→2✓qml. *total4→4✓Measurements1→1✓
Verdict: transformed_valid
Stage 3: MinHash Deduplication (Stage3_Dedup.py) :
The modernized code is tokenized into 3-shingles using
datasketch MinHash (128 permutations). Entry #265 a
structurally similar classifier from a different repository is
compared:
Deduplication check (Entry #32 vs. #265):Shingles:
48vs.89| Intersection/Union: 43/94| Jaccard: 0.457<
0.70⇒ UNIQUE⇒RETAIN
Both entries are retained since their implementation differences
(device declaration style, variable scoping) produce sufficient
token divergence despite semantic similarity.
Stage 4: Instruction Generation
(PromptGen_Hybrid.py) : The final code is sent
togemini-2.5-flash-lite with the instruction-style
prompt:
/userUSER
Based on this PennyLane code, write a clear instruction prompt
(20–40 words).
Format:Start with an action verb. Include: key quantum
operations, number of qubits/wires, return type.
Return ONLY the instruction prompt.
Generated instruction:
/commen◎ASSISTANT
Implement a PennyLane quantum circuit with n_qubits using
AngleEmbedding and StronglyEntanglingLayers ,
returning expectation values of PauliZ on each wire.
Quality Improvement Summary:The pipeline transformed
a raw community file containing 5 irrelevant helper func-
tions and 2 deprecated API calls into a single ver-
ified, modernized, instruction-annotated entry. The key
transformation — qml.templates.AngleEmbedding→
qml.AngleEmbedding directly reduces the API misuse
failure mode identified in Table VI, since retrieved code using
the old namespace causes AttributeError at inference
time under PennyLane≥v0.40.
G. Dataset Pipeline Case 2: Precise Deduplication
This case demonstrates that the deduplication stage is
precision-preserving: structurally similar entries implementing
meaningfully different circuits are correctly retained as distinct
knowledge base entries.Entry #265 Candidate Near-Duplicate of Entry #32:
During Stage 3 deduplication, Entry #265 is compared against
Entry #32:
Listing 3: Entry #265 Candidate near-duplicate with dynamic device
construction
@qml.qnode(create_quantum_device(n_qubits←-
),
interface=’torch’)
defqnode(inputs, weights):
qml.AngleEmbedding(inputs,
wires=range(n_qubits))
qml.StronglyEntanglingLayers(weights,
wires=range(n_qubits))
return[qml.expval(qml.PauliZ(i))
foriinrange(n_qubits)]
Pipeline Stages for Entry #265:Stage 1 (Extraction):The
AST parser extracts this function from a different community
repository. Classified as direct (3qml. *calls). No dep-
recated API is present, Entry #265 already uses the modern
qml.AngleEmbeddingnamespace.
Stage 2 (Verification):All four verification layers pass. No
transformation required. Verdict: original_valid
Stage 3 Deduplication Decision:
MinHash LSH (128 permutations, k=3 shingles,
threshold=0.70) computes the following:
Deduplication check (Entry #32 vs. #265):Shingles:
48vs.89| Intersection/Union: 43/94| Jaccard: 0.457<
0.70⇒ UNIQUE⇒RETAIN
Despite both entries implementing AngleEmbedding +
StronglyEntanglingLayers with PauliZ measurements,
they differ structurally:
Despite both entries implementing the same gate set, they differ
structurally: Entry #32 declares the device globally ( dev =
qml.device(...) ), uses 4 fixed wires, and names the func-
tionquantum_circuit ; Entry #265 wraps device creation
in a helper ( create_quantum_device(n_qubits) ),
uses a variable wire count, names the function qnode , and
specifies interface=’torch’ in the decorator. These
differences produce sufficient token divergence to fall below
the 0.70 threshold.
Stage 4 Instruction Generation:
/userUSER
Based on this PennyLane code, write a clear instruction prompt
(20–40 words).
Format:Start with an action verb. Include: key quantum
operations, number of qubits/wires, return type.
Return ONLY the instruction prompt.
Code:[function body]
Generated instruction:

/commen◎ASSISTANT
Implement a PennyLane QNode using a dynamically created
default.qubit device with n_qubits wires, applying
AngleEmbedding and StronglyEntanglingLayers ,
and returning PauliZ expectation values.
Value of Retaining Both Entries:At retrieval time, Entry #32
and Entry #265 serve different challenge patterns: leftmar-
gin=1.5em
•Entry #32matches challenges withfixed-wire device
declarations(common in QHack 2022 templates).
•Entry #265matches challenges requiringdynamic device
constructionwith variable qubit counts.
Retaining both increases retrieval coverage across circuit
topology variants, precisely the complementary coverage that
the profiling ablation in Section V-D2 confirms is valuable.
Over-aggressive deduplication (e.g., threshold = 0.40 ) would
have incorrectly removed Entry #265, reducing coverage for
dynamic-device challenges.
H. Retrieval Case Studies
a) Successful retrieval:
three_shipping_companies :This challenge requires
a neutral-atom gate set using RY(π/2) and CZ gates, absent
from standard PennyLane tutorials. PennySynth retrieved
a near-identical QHack 2023 challenge at similarity 0.850,
enabling the model to adapt the correct gate set. Cross-system
CodeBLEU of 0.237 confirms fundamentally different code
was produced compared to the base model, with only the
retrieval-augmented version succeeding.
b) Retrieval interference: GHZ_inn :Base Claude solves
this correctly while PennySynth fails despite a cross-system
CodeBLEU of 0.752. Retrieved context introduced a subtly
different circuit topology that overrode the model’s correct
internal reasoning, a failure mode when retrieved similarity is
above threshold but the example targets a different problem
variant.
c) Coverage gap: mojito_hhlime_twist :Both sys-
tems fail with cross-system CodeBLEU of 0.016, indicating
entirely different solution strategies. Neither parametric knowl-
edge nor retrieved examples contained sufficient information,
representing a genuine capability gap motivating Graph RAG
as future work.
VI. CONCLUSION
We presented PennySynth, a retrieval-augmented generation
system for automated PennyLane quantum code synthesis
built around an automated three-stage data synthesis pipeline
producing PennySynth-13K, a corpus of 13,389 AST-verified
instruction-code pairs. By adopting a code-aware embedding
model and expanding the knowledge base with competition-
style QHack archives, PennySynth achieves up to 68% pass@5
on QHack challenges, a +25 pp improvement over a strong
base LLM. A controlled 2 ×2 ablation demonstrates that
code-aware embeddings are the primary performance driver,
with dataset expansion providing synergistic gains only underprecise retrieval. Multi-model evaluation across six state-of-
the-art LLMs establishes the first comprehensive baseline for
PennyLane competition-level code generation.
Future work will address: (1) a custom PennyLane
knowledge graph for Graph RAG; (2) the combined Pen-
nySynth + fine-tuned system; (3) hybrid dense-sparse retrieval
(BM25 + CodeBERT); and (4) extending the multi-model
evaluation to additional LLMs and newer QHack benchmark
years.
ACKNOWLEDGMENT
This work was supported in part by the NYUAD Center
for Quantum and Topological Systems (CQTS), funded by
Tamkeen under the NYUAD Research Institute grant CG008,
and Center for CyberSecurity (CCS), funded by Tamkeen under
the NYUAD Research Institute Award G1104. This research
was carried out on the High Performance Computing resources
at New York University Abu Dhabi.
REFERENCES
[1]V . Bergholm, J. Izaac, M. Schuld, C. Gogolin, and N. Killoran,
“PennyLane: Automatic differentiation of hybrid quantum-classical
computations,”arXiv preprint arXiv:1811.04968, 2018.
[2]A. Javadi-Abhari, M. Treinish, K. Krsulich, C. J. Wood, J. Lishman,
J. Gacon, S. Martiel, P. D. Nation, L. S. Bishop, A. W. Crosset al.,
“Quantum computing with qiskit,”arXiv preprint arXiv:2405.08810, 2024.
[3]K. Zaman, A. Marchisio, M. A. Hanif, and M. Shafique, “A survey on
quantum machine learning: Current trends, challenges, opportunities, and
the road ahead,”arXiv preprint arXiv:2310.10315, 2023.
[4]M. Chen, J. Tworek, H. Jun, Q. Yuanet al., “Evaluating large language
models trained on code,”arXiv preprint arXiv:2107.03374, 2021.
[5]A. Lozhkov, R. Li, L. B. Allalet al., “StarCoder 2 and the stack v2:
The next generation,”arXiv preprint arXiv:2402.19173, 2024.
[6]N. Dupuis, L. Buratti, S. Vishwakarmaet al., “Qiskit code assistant:
Training LLMs for generating quantum computing code,” IBM Quantum
Computing Blog, 2024.
[7]S. Vishwakarma, F. Harkins, S. Golechaet al., “Qiskit HumanEval:
An evaluation benchmark for quantum code generative models,”arXiv
preprint arXiv:2406.14712, 2024.
[8]X. Guo, M. Wang, and J. Zhao, “Quanbench: Benchmarking quan-
tum code generation with large language models,”arXiv preprint
arXiv:2510.16779, 2025.
[9]A. Basit, N. Innan, M. H. Asif, M. Shao, M. Kashif, A. Marchisio, and
M. Shafique, “A pennylane-centric dataset to enhance llm-based quantum
code generation using rag,”arXiv preprint arXiv:2503.02497, 2025.
[10] A. Basit, M. Shao, M. H. Asif, N. Innan, M. Kashif, A. Marchisio, and
M. Shafique, “PennyCoder: Efficient domain-specific llms for pennylane-
based quantum code generation,” in2025 IEEE International Conference
on Quantum Computing and Engineering (QCE), vol. 2. IEEE, 2025,
pp. 229–234.
[11] A. Basit, M. Shao, M. H. Asif, N. Innan, M. Kashif, A. Marchisio,
and M. Shafique, “QHackBench: Benchmarking large language models
for quantum code generation using pennylane hackathon challenges,” in
2025 IEEE International Conference on Quantum Artificial Intelligence
(QAI). IEEE, 2025, pp. 316–322.
[12] Y . Kharkov, Z. Mohammad, M. Beach, and E. Kessler, “Accelerate
quantum software development on Amazon Braket with Claude-3,” AWS
Quantum Technologies Blog, 2024.
[13] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive NLP tasks,” inAdvances
in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp.
9459–9474.
[14] L. Wang, N. Yang, and F. Wei, “Query2doc: Query expansion with large
language models,” inProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, 2023, pp. 9414–9423.
[15] R. Nogueira and K. Cho, “Passage re-ranking with BERT,”arXiv preprint
arXiv:1901.04085, 2019.

[16] M. R. Parvez, W. U. Ahmad, S. Chakraborty, B. Ray, and K.-W. Chang,
“Retrieval augmented code generation and summarization,”arXiv preprint
arXiv:2108.11601, 2021.
[17] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and
J. Larson, “From local to global: A graph RAG approach to query-focused
summarization,”arXiv preprint arXiv:2404.16130, 2024.
[18] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: A method for
automatic evaluation of machine translation,” inProceedings of the 40th
Annual Meeting of the Association for Computational Linguistics (ACL),
2002, pp. 311–318.
[19] C.-Y . Lin, “ROUGE: A package for automatic evaluation of summaries,”
inText Summarization Branches Out, 2004, pp. 74–81.
[20] S. Ren, D. Guo, S. Lu, L. Zhou, S. Liu, D. Tang, N. Sundaresan, M. Zhou,
A. Blanco, and S. Ma, “CodeBLEU: A method for automatic evaluation
of code synthesis,”arXiv preprint arXiv:2009.10297, 2020.
[21] J. Leskovec, A. Rajaraman, and J. D. Ullman,Mining of massive data
sets. Cambridge university press, 2020.
[22] Anthropic, “Claude Sonnet 4.6 model card,” Anthropic Technical Report,
2024.
[23] OpenAI, “GPT-5.5 system card,” OpenAI Technical Report, 2024.
[24] Google DeepMind, “Gemini 2.5 Pro technical report,” Google DeepMind
Technical Report, 2024.
[25] Qwen Team, “Qwen3 technical report,” arXiv preprint arXiv:2505.09388,
2025.
[26] Zhipu AI, “GLM-5.1: General language model technical report,” Zhipu
AI Technical Report, 2024.
[27] DeepSeek AI, “DeepSeek-V3 technical report,” arXiv preprint
arXiv:2412.19437, 2024.