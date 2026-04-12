# Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering

**Authors**: Byeolhee Kim, Min-Kyung Kim, Young-Hak Kim, Tae-Joon Jeon

**Published**: 2026-04-06 11:13:57

**PDF URL**: [https://arxiv.org/pdf/2604.04593v1](https://arxiv.org/pdf/2604.04593v1)

## Abstract
Retrieval-augmented generation (RAG) grounds large language models in external medical knowledge, yet standard retrievers frequently surface hard negatives that are semantically close to the query but describe clinically distinct conditions. While existing query-expansion methods improve query representation to mitigate ambiguity, they typically focus on enriching target-relevant semantics without an explicit mechanism to selectively suppress specific, clinically plausible hard negatives. This leaves the system prone to retrieving plausible mimics that overshadow the actual diagnosis, particularly when such mimics are dominant within the corpus. We propose Contrastive Hypothesis Retrieval (CHR), a framework inspired by the process of clinical differential diagnosis. CHR generates a target hypothesis $H^+$ for the likely correct answer and a mimic hypothesis $H^-$ for the most plausible incorrect alternative, then scores documents by promoting $H^+$-aligned evidence while penalizing $H^-$-aligned content. Across three medical QA benchmarks and three answer generators, CHR outperforms all five baselines in every configuration, with improvements of up to 10.4 percentage points over the next-best method. On the $n=587$ pooled cases where CHR answers correctly while embedded hypothetical-document query expansion does not, 85.2\% have no shared documents between the top-5 retrieval lists of CHR and of that baseline, consistent with substantive retrieval redirection rather than light re-ranking of the same candidates. By explicitly modeling what to avoid alongside what to find, CHR bridges clinical reasoning with retrieval mechanism design and offers a practical path to reducing hard-negative contamination in medical RAG systems.

## Full Text


<!-- PDF content starts -->

Ruling Out to Rule In: Contrastive Hypothesis Retrieval for
Medical Question Answering
Byeolhee Kim
Department of Information Medicine
Asan Medical Center
Seoul, Republic of KoreaMin-Kyung Kim
Department of Computer Science
Yonsei University
Seoul, Republic of Korea
INMED DATA
Seoul, Republic of Korea
Young-Hak Kim
Division of Cardiology, Department of Information
Medicine
Asan Medical Center, University of Ulsan College of
Medicine
Seoul, Republic of KoreaTae-Joon Jeon∗
Department of Information Medicine
Asan Medical Center, University of Ulsan College of
Medicine
Seoul, Republic of Korea
Abstract
Retrieval-augmented generation (RAG) grounds large language
models in external medical knowledge, yet standard retrievers fre-
quently surface hard negatives that are semantically close to the
query but describe clinically distinct conditions. While existing
query-expansion methods improve query representation to miti-
gate ambiguity, they typically focus on enriching target-relevant
semantics without an explicit mechanism to selectively suppress
specific, clinically plausible hard negatives. This leaves the system
prone to retrieving plausible mimics that overshadow the actual
diagnosis, particularly when such mimics are dominant within
the corpus. We propose Contrastive Hypothesis Retrieval (CHR),
a framework inspired by the process of clinical differential diag-
nosis. CHR generates a target hypothesis 𝐻+for the likely correct
answer and a mimic hypothesis 𝐻−for the most plausible incorrect
alternative, then scores documents by promoting 𝐻+-aligned evi-
dence while penalizing 𝐻−-aligned content. Across three medical
QA benchmarks and three answer generators, CHR outperforms
all five baselines in every configuration, with improvements of
up to 10.4 percentage points over the next-best method. On the
𝑛=587pooled cases where CHR answers correctly while embed-
ded hypothetical-document query expansion does not, 85.2% have
no shared documents between the top-5 retrieval lists of CHR and
of that baseline, consistent with substantive retrieval redirection
rather than light re-ranking of the same candidates. By explicitly
modeling what to avoid alongside what to find, CHR bridges clinical
reasoning with retrieval mechanism design and offers a practical
path to reducing hard-negative contamination in medical RAG
systems.
CCS Concepts
•Information systems →Question answering;Information
retrieval;•Computing methodologies→Natural language pro-
cessing.
∗Corresponding author.Keywords
medical question answering, retrieval-augmented generation, con-
trastive retrieval, hard negatives, differential diagnosis
1 Introduction
Retrieval-augmented generation (RAG) has become the dominant
paradigm for grounding large language models (LLMs) in external
knowledge, particularly in high-stakes medical domains where fac-
tual accuracy is critical [ 3,5,11]. However, standard RAG pipelines
are systematically vulnerable to hard negatives—documents that
share surface-level similarity with the query but describe clini-
cally distinct conditions—in medical question answering. Previous
work has documented retrieval relevance as low as 22% on med-
ical benchmarks [ 17], indicating that the main bottleneck lies in
retrieval rather than generation.
Query-expansion methods such as HyDE [ 4], Query2Doc [ 16],
CSQE [ 9], and ThinkQE [ 10] bridge the query-document semantic
gap by generating hypothetical or augmented content. While these
methods enrich the query representation, they share a fundamental
limitation. They model onlywhat to findwithout any mechanism
to explicitly suppresswhat to avoid. When the generated content
converges toward a plausible but incorrect interpretation, which
is common in medical differential diagnosis, the expanded query
remains biased toward the wrong semantic region.
We propose Contrastive Hypothesis Retrieval (CHR), a frame-
work inspired by clinical differential diagnosis that operationalizes
this principle at inference time. CHR produces a contrastive pair
consisting of a target hypothesis ( 𝐻+) describing the likely correct
answer and a mimic hypothesis ( 𝐻−) describing the most plausible
incorrect alternative. The retriever scores documents by promoting
𝐻+-aligned content while penalizing 𝐻−-aligned content, steer-
ing the query away from the hard-negative region and surfacing
documents that conventional methods fail to reach.
In three medical QA benchmarks with three answer generators,
CHR consistently outperforms all five baselines. Retrieval shift anal-
ysis pools all instances where CHR answers correctly but HyDE
does not (𝑛= 587). In 85.2% of these cases, the top-5 documentsarXiv:2604.04593v1  [cs.IR]  6 Apr 2026

Kim et al.
retrieved by CHR share no overlap with HyDE’s top-5, which sup-
ports the view that the contrastive mechanism redirects retrieval to
different evidence rather than incrementally re-ranking the same
candidate pool.
Our contributions are as follows.
•We identify a systematic failure mode shared by existing query-
expansion methods for medical QA. They model only what to
find, leaving the retriever vulnerable to hard negatives when the
dominant corpus pattern favors a clinically plausible mimic.
•We propose Contrastive Hypothesis Retrieval (CHR), which gen-
erates a contrastive hypothesis pair ( 𝐻+and𝐻−) and retrieves
documents via explicit contrast, mirroring clinical differential
diagnosis reasoning.
•We demonstrate consistent improvements over five baselines
across three medical QA benchmarks and three generators, with
retrieval shift analysis confirming that CHR redirects retrieval
toward fundamentally different semantic regions.
2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-augmented generation (RAG) conditions language model
outputs on documents retrieved from an external corpus, reducing
hallucination and enabling access to specialized knowledge beyond
the model’s parametric memory [ 5,11]. In medicine, RAG has been
widely adopted to ground clinical question answering in biomed-
ical literature [ 3]. However, Xiong et al . [17] show that retrieval
relevance on medical benchmarks can be as low as 22%, indicating
that the primary bottleneck lies not in generation but in retrieval
quality. Domain-specific retrievers such as MedCPT [ 8], pretrained
on PubMed search logs, improve biomedical retrieval but do not
explicitly address the hard-negative problem where clinically dis-
tinct conditions share similar surface features. A complementary
line of work focuses on post-retrieval refinement or iterative feed-
back loops to improve document quality. These approaches range
from rationale-based filtering of medical snippets [ 14] and iterative
self-reflection for clinical QA [ 13] to critique-gated document selec-
tion in adaptive RAG systems [ 1]. While such techniques enhance
downstream grounding, they operate on a pre-defined candidate
pool and cannot rectify fundamentally misdirected searches. In
contrast, CHR addresses this bottleneck at the query construction
stage, reshaping the initial retrieval vector to preemptively steer
away from anticipated hard negatives.
2.2 Query Expansion for Retrieval
Query expansion methods bridge the semantic gap between short
queries and long documents by augmenting the original query
with LLM-generated content. HyDE [ 4] generates multiple hypo-
thetical answer documents and averages their embeddings as the
retrieval query, relying on noise dilution to filter spurious directions.
Query2Doc [ 16] appends a generated pseudo-document directly to
the query for sparse or dense retrieval. CSQE [ 9] combines corpus-
originated pivotal sentences with LLM-generated expansions via
a two-stage pipeline, while ThinkQE [ 10] introduces an iterative
thinking-based process to refine queries using retrieval feedback.
Figure 1: Overview of Contrastive Hypothesis Retrieval
(CHR). Given a clinical question, CHR generates a contrastive
hypothesis pair consisting of a target hypothesis ( 𝐻+) and
a mimic hypothesis ( 𝐻−), retrieves documents using con-
trastive scoring that promotes target-aligned content while
penalizing mimic-aligned content, and generates the final
answer from the retrieved evidence.
Although these methods effectively enrich query semantics, they
share a fundamental limitation. They lack a dedicated mechanism
to counteract misleading retrieval directions. This absence of nega-
tive constraints leaves the retriever vulnerable to hard negatives,
particularly when an expanded query converges toward a plausible
but incorrect interpretation—a frequent challenge in medical differ-
ential diagnosis. This necessitates a retrieval framework that can
explicitly counteract these biases at the query construction stage.
3 Methodology
Given a clinical query 𝑄and candidate options 𝑂
={𝑜 1,𝑜2,...,𝑜𝑛}, the objective is to identify the correct answer.
As illustrated in Figure 1, CHR operates in three stages.(1) Con-
trastive Hypothesis Generation.Given 𝑄and𝑂, an LLM pro-
duces a target hypothesis 𝐻+describing the likely correct an-
swer and a mimic hypothesis 𝐻−describing the most plausible
incorrect alternative, both in a single structured call.(2) Con-
trastive Document Scoring.Each candidate document is scored
by𝑆(𝑑)=Sim(𝑑,𝐻+)−𝜆·Sim(𝑑,𝐻−), promoting target-aligned
evidence while suppressing mimic-aligned hard negatives. The top-
𝐾documents are returned.(3) Answer Generation.The retrieved
documents are provided to an answer generator along with the
original question to produce the final answer.

Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering
3.1 Contrastive Hypothesis Generation
The core idea of CHR is that suppressing hard negatives requires
explicit knowledge of what makes them misleading. We instantiate
this principle through a contrastive hypothesis pair that captures
both the target and its closest mimic.
The target hypothesis 𝐻+describes the clinical reasoning and
distinguishing features of the likely correct diagnosis. It captures
the pathophysiology, characteristic symptoms, and gold-standard
treatment that differentiate the target condition from alternatives.
This hypothesis guides the retriever toward documents containing
target-relevant evidence.
The mimic hypothesis 𝐻−describes the most plausible incorrect
alternative that shares surface-level similarities with the correct
answer. It articulates why a clinician might mistakenly consider this
condition while specifying the subtle features that rule it out. By
explicitly modeling the mimic, CHR enables the retriever to identify
and down-weight documents that would otherwise rank highly due
to superficial similarity. This mirrors clinical differential diagnosis,
where physicians consider and exclude similar conditions before
confirming a diagnosis.
Both hypotheses are generated in a single LLM call using a struc-
tured prompt that requests a JSON-formatted output containing
𝐻+and𝐻−. The prompt instructs the model to analyze the clinical
scenario and generate two conflicting hypotheses for retrieval, as
shown in Figure 2.
3.2 Contrastive Document Scoring
Given the hypothesis pair (𝐻+,𝐻−), we score each candidate docu-
ment𝑑using a contrastive scoring function:
𝑆(𝑑)=Sim(𝑑,𝐻+)−𝜆·Sim(𝑑,𝐻−)(1)
where Sim(·,·) denotes cosine similarity between document and
hypothesis embeddings, and 𝜆≥ 0controls the penalty weight
for mimic-aligned documents. The first term promotes documents
relevant to the target diagnosis, while the second term penalizes
documents that align with the mimic. Documents that are similar
to𝐻+but dissimilar to 𝐻−receive the highest scores, effectively
filtering out hard negatives that would mislead the answer genera-
tor. We set𝜆=1.0for all experiments. A sensitivity analysis over 𝜆
is provided in the supplementary material.
This scoring function admits a geometric interpretation. Since
cosine similarity is equivalent to a dot product for normalized
vectors, Equation 1 can be rewritten as Equation 2.
𝑆(𝑑)=®𝑑·(®𝐻+−𝜆®𝐻−)(2)
This reveals that contrastive scoring is equivalent to retrieving with
a shifted query vector (®𝐻+−𝜆®𝐻−). Subtracting the mimic direction
steers the query away from the semantic region dominated by
hard negatives and toward the region containing target-relevant
documents. Documents are ranked by 𝑆(𝑑), and the top- 𝐾results
are returned.
3.3 Answer Generation
The retrieved documents are concatenated into a context string and
provided to the answer generator along with the original question
and options. The generator produces the final answer conditionedSystem Prompt
You are a medical specialist assisting with complex clinical decision-
making. Your goal is to generate precise diagnostic hypotheses to
guide an evidence-based search engine. Ensure all outputs are in strict
JSON format.
User Prompt
Analyze the clinical scenario below and generate two conflicting hy-
potheses for retrieval:
1. H_plus (Target Hypothesis):Describe the pathophysiology, dis-
tinct symptoms, or gold-standard treatment for the CORRECT diagno-
sis. Focus on specific details that differentiate it from other conditions.
2. H_minus (Mimic Hypothesis):Describe the primary differen-
tial diagnosis or closest mimic that is INCORRECT. Explain why a
clinician might mistakenly consider this condition due to overlapping
symptoms, but specify the subtle features that rule it out.
Question:{question}
Options:{options}
Output Requirement:
Return ONLY a JSON object with keys “H_plus” and “H_minus”.
{“H_plus”: “...”, “H_minus”: “...”}
Figure 2: Prompt template for contrastive hypothesis gener-
ation. The system prompt establishes the role of a medical
specialist, and the user prompt instructs the model to gener-
ate a target hypothesis ( 𝐻+) for the likely correct diagnosis
and a mimic hypothesis ( 𝐻−) for the most plausible incorrect
alternative.
on this evidence. This step follows standard RAG practice and is
agnostic to the specific generator architecture, allowing CHR to be
combined with any LLM.
4 Experiments
We evaluate Contrastive Hypothesis Retrieval (CHR) on three med-
ical question answering benchmarks and compare it against five
baselines spanning standard retrieval and recent query-expansion
methods.
4.1 Evaluation Datasets
We use three representative datasets to assess different aspects of
medical knowledge. MMLU-Med [ 6] consists of six medical sub-
sets (1,089 questions) from the Massive Multitask Language Under-
standing benchmark, covering clinical knowledge, medical genet-
ics, anatomy, professional medicine, college biology, and college
medicine. MedQA [ 7] provides 1,273 United States Medical Licens-
ing Examination (USMLE) style questions that require multi-step
clinical reasoning. BioASQ [ 15] includes 618 yes/no questions de-
rived from biomedical literature, testing factual knowledge retrieval.
All datasets are evaluated using the standardized test splits and
multiple-choice formatting provided by the MedRAG benchmark
[17].

Kim et al.
4.2 Implementation Details
For the retrieval component, we use MedCPT [ 8] as the dense re-
triever, which is pre-trained on PubMed search logs. The retrieval
corpus is MedCorp [ 17], a comprehensive medical database ag-
gregating approximately 5.8 million text chunks from PubMed ab-
stracts, StatPearls, medical textbooks, and Wikipedia health articles.
To ensure a rigorous and fair comparison, we fix the LLM used
for all query-expansion methods (including CHR) to Qwen2.5-72B-
Instruct [ 18]. For answer generation, we evaluate three diverse
models: Llama-3-8B-Instruct [ 2], Qwen2.5-7B-Instruct [ 18], and
Gemma-2-9B-It [ 12]. Across all experiments, we retrieve 𝐾= 5
documents. Standard RAG serves as the primary baseline, retriev-
ing evidence using the original query without any expansion or
hypothetical content.
4.3 Overall Performance
We compare CHR against five baselines: Standard RAG (original
query retrieval), HyDE [ 4] (hypothetic document embedding aver-
aging), Query2Doc [ 16] (pseudo-document augmentation), CSQE
[9] (corpus-steered query expansion), and ThinkQE [ 10] (thinking-
based iterative expansion). Table 1 presents the end-to-end QA
accuracy across nine independent settings (three datasets ×three
generators). Following the MedRAG evaluation protocol [ 17], we
adopt end-to-end QA accuracy as the primary metric. However, by
fixing the retriever, corpus, and answer generator as constants, we
ensure that any systematic difference in accuracy is directly attrib-
utable to the quality of the retrieved evidence. The consistency of
improvements across various architectures reinforces end-to-end
accuracy as a reliable proxy for retrieval quality in this controlled
setup.
CHR consistently achieves the highest accuracy in all 18 head-to-
head comparisons. The performance gain is most pronounced on
BioASQ, where CHR achieves 75.6% accuracy with Qwen2.5-7B, out-
performing the next-best method (Query2Doc) by 10.4 percentage
points. On MedQA, which requires multi-step clinical reasoning,
CHR with Llama-3-8B reaches 51.5%, leading all baselines by at
least 5.0 percentage points.
A critical observation is that several query-expansion baselines
occasionally underperform Standard RAG. For instance, on BioASQ
with Qwen2.5-7B, HyDE drops to 42.2% and ThinkQE to 43.4%,
whereas Standard RAG maintains 60.5%. This confirms that aug-
menting a query without an explicit mechanism to suppress mis-
leading directions can actively harm performance when generated
content converges toward a hard negative. CHR effectively avoids
this failure mode by using the mimic hypothesis 𝐻−to steer re-
trieval away from such regions, ensuring robust improvements
regardless of the dataset or generator architecture.
4.4 Retrieval Shift Analysis
To understand how CHR improves upon prior methods, we analyze
the𝑛= 587cases where CHR answers correctly but HyDE fails.
We specifically compare CHR against HyDE because both methods
rely on LLM-generated hypotheses, yet they differ in how they
operationalize them: HyDE focuses solely on the target-aligned
direction, whereas CHR incorporates a contrastive mimic. For eachcase, we measure the overlap between the top-5 documents re-
trieved by CHR and those retrieved by HyDE. If CHR’s gains were
derived from marginal re-ranking—promoting existing candidates
to higher positions—one would expect substantial document over-
lap. Conversely, low overlap would indicate that CHR surfaces
fundamentally different evidence from distinct semantic regions.
Table 2 summarizes this comparative analysis. In these cases,
there is zero document overlap between the top-5 results of CHR
and HyDE in 85.2% of the instances, and the mean overlap ratio is
only 0.18. This near-total lack of overlap confirms that CHR does
not merely re-rank the same candidate pool. Instead, it redirects
the retrieval process to an entirely different set of evidence. The
contrastive subtraction of ®𝐻−effectively shifts the query vector
away from the mimic-dominated region, enabling the retriever to
access informative documents that hypothesis averaging systemati-
cally overlooks. These results provide direct empirical evidence that
CHR’s accuracy gains stem from retrieving qualitatively superior
evidence rather than incremental ranking improvements.
4.5 Efficiency Analysis
Beyond accuracy, CHR offers a practical efficiency advantage in
terms of hypothesis generation cost. Table 3 compares the query-
expansion cost per sample across all evaluated methods. HyDE
requires approximately eight separate LLM calls to generate hy-
pothetical documents for embedding averaging, while ThinkQE
performs multiple iterative expansion–retrieval rounds. Query2Doc
and CSQE each require only one LLM call, they typically generate
significantly longer pseudo-documents. In contrast, CHR produces
both𝐻+and𝐻−in a single structured LLM call, yielding a compact
JSON object containing two targeted hypotheses. This efficiency
stems from CHR’s design principle, which explicitly models the
contrastive direction with only two focused hypotheses rather than
averaging over multiple noisy ones. As a result, CHR achieves a
9.7×reduction in output tokens relative to HyDE, making it the
most cost-effective strategy among all query-expansion baselines.
4.6 Mimic Hypothesis Quality Analysis
Earlier sections demonstrate that CHR effectively redirects retrieval
and improves accuracy relative to strong baselines. We next strengthen
the plausibility of the contrastive mechanism by analyzing the
quality of the generated hypotheses. Hard negatives in biomed-
ical retrieval often share significant surface-level cues with the
target answer. Therefore, the penalty term should be most effective
when𝐻−accurately approximates the clinical mimic that typically
pollutes retrieval. We investigate whether end-to-end CHR perfor-
mance co-varies with the clinical validity of 𝐻−, specifically its
ability to capture the intended confounder.
To this end, a board-certified physician reviewed 50 randomly
sampled CHR outputs from MedQA and MMLU-Med, with items
stratified to represent all three answer generators equally. Each
item was assigned an 𝐻−rating ofExcellent,Good, orPoor
based on the following criteria.Excellent: 𝐻−identifies the most
clinically plausible alternative for the stem—the primary differential
diagnosis a clinician must rule out.Good: 𝐻−identifies a plausible
mimic but not the most challenging one for the given scenario.
Poor:𝐻−either (i) collapses into the same semantic frame as 𝐻+

Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering
Generator Method MMLU-Med MedQA-US BioASQ-Y/N Avg.
Llama-3-8B-InstructStandard RAG 36.8 36.1 41.8 38.2
HyDE (𝐻+only) 47.2 44.6 51.3 47.7
Query2doc 49.1 46.2 46.9 47.4
CSQE 49.7 46.5 56.5 50.9
ThinkQE 41.3 45.6 52.1 46.3
CHR (Ours) 50.4 51.5 63.1 55.0
Qwen2.5-7B-InstructStandard RAG 58.8 39.7 60.5 53.0
HyDE (𝐻+only) 56.0 46.0 42.2 48.1
Query2doc 52.8 41.5 65.2 53.2
CSQE 54.9 42.2 44.8 47.3
ThinkQE 55.7 43.0 43.4 47.4
CHR (Ours) 59.4 49.2 75.6 61.4
Gemma-2-9B-ItStandard RAG 62.4 46.0 74.3 60.9
HyDE (𝐻+only) 64.2 51.6 48.4 54.7
Query2doc 61.2 46.9 58.6 55.6
CSQE 69.7 49.6 72.2 63.8
ThinkQE 60.5 52.2 44.5 52.4
CHR (Ours) 70.3 53.1 80.7 68.1
Table 1: Accuracy (%) across three medical QA benchmarks. All query-expansion baselines use the same hypothesis generator
(Qwen2.5-72B-Instruct), retriever (MedCPT), and corpus (MedCorp). CHR consistently achieves the highest accuracy across all
settings.
through paraphrasing, or (ii) fails as a mimic by focusing on content
irrelevant to the provided options. Three cases were excluded due
to clinically invalid 𝐻+outputs, leaving a total of 47 evaluated
samples.
Table 4 aligns with the expectation that high-quality mimic hy-
potheses track CHR success. Accuracy increases fromPoor(37.5%)
toGood(40.0%) and reaches 66.7% forExcellentratings. Given the
small sample sizes in theExcellentandPoortiers, we treat this
trend as qualitative support rather than a precise estimate of the
performance gap. When 𝐻−describes the same misleading clinical
narrative emphasized by incorrect passages, the penalty effectively
pushes retrieval away from that region. Conversely, when 𝐻−re-
peats the logic of 𝐻+, the contrast fades and the system’s behavior
converges toward using𝐻+alone.
Excellent-rated mimic (MRKH versus AIS)..The stem describes a
patient with primary amenorrhea and a 46,XX karyotype, but with
normal breast development and absent uterus. 𝐻+correctly iden-
tifies Müllerian agenesis (MRKH), emphasizing the typical female
testosterone levels. 𝐻−identifies Androgen Insensitivity Syndrome
(AIS), which presents with nearly identical surface features but typ-
ically involves elevated testosterone. Because MRKH and AIS share
significant literature (e.g., “primary amenorrhea absent uterus”),
they frequently surface as hard negatives for one another. By sub-
tracting the AIS-aligned direction, CHR surfaces MRKH-specific
evidence, leading to the correct diagnosis.
Poor-rated collapse (Starling forces).In a question regarding Star-
ling forces and efferent arteriole constriction, 𝐻+correctly identifies
increased glomerular hydrostatic pressure. The rated-Poor 𝐻−fails
to provide a distinct alternative; instead, it replicates the mechanism
of𝐻+while merely adding a warning about vessel misidentification.
Because their texts overlap almost entirely, their embeddings arenearly parallel, making the difference vectorH+−𝜆H−negligible.
Consequently, the retriever fails to isolate discriminative evidence,
returning broad renal physiology content instead of specific evi-
dence for the keyed choice.
In summary, tier-wise accuracies and these vignettes provide
qualitative support for the necessity of a confounder-specific mimic.
The observed patterns confirm that the contrastive penalty is most
informative when 𝐻−targets the misleading document neighbor-
hood rather than merely paraphrasing the target hypothesis.
4.7 Case Study: The Discriminative Role of the
Mimic Hypothesis
To investigate the impact of 𝐻−on retrieval dynamics, we conduct
a qualitative comparison between 𝐻+-only retrieval and 𝐻+−𝜆𝐻−
retrieval using an MMLU-Med case study (Figure 3). Since 𝐻+-only
retrieval omits the contrastive subtraction of 𝐻−, this comparison
serves as a qualitative ablation that isolates the contribution of the
mimic hypothesis.
The question describes a patient with parkinsonian symptoms
whose condition improves after receiving an antiviral drug. The
correct answer is “amantadine,” a drug originally developed as an
antiviral but repurposed for Parkinson’s disease due to its dopamin-
ergic mechanisms. The reasoning challenge lies in resolving a cross-
domain link between an antiviral agent and a neurodegenerative
disorder.
𝐻+frames idiopathic Parkinson’s disease and correctly steers
retrieval toward anti-parkinsonian pharmacotherapy. However, it
fails to capture the stem’s distinguishing detail regarding the an-
tiviral class, nor does it foreground amantadine’s dual role. While
𝐻+encodes the correct clinical axis (treating parkinsonism), it is
insufficient to bridge the gap to the specific evidence required. This

Kim et al.
Dataset n Zero Overlap Mean Overlap
MMLU-Med 172 78.5% 0.25
MedQA 249 80.3% 0.25
BioASQ 166 99.4% 0.01
Combined 587 85.2% 0.18
Table 2: Retrieval divergence analysis for cases where CHR succeeds but HyDE fails ( 𝑛= 587). Zero Overlap indicates the
percentage of cases with no shared documents in top-5 results. Mean Overlap is the average overlap ratio of top-5 results
between CHR and HyDE.
Method LLM Calls Output Tokens Token Reduction
Standard RAG 0 0 N/A
Query2Doc 1∼512 5.7×
CSQE 1 + retrieval∼480 6.1×
ThinkQE≥3∼1,500 2.0×
HyDE∼8∼2,926 1.0×
CHR (Ours) 1∼303 9.7×
Table 3: Query-expansion cost per sample, averaged across
MedQA and MMLU-Med. Output token counts are estimated
from generated text length. Token Reduction is relative to
HyDE (most expensive). CHR achieves the lowest cost while
consistently achieving the highest accuracy.
𝐻−Quality𝑛Correct Accuracy
Excellent 9 6 66.7%
Good 30 12 40.0%
Poor 8 3 37.5%
Table 4: CHR accuracy stratified by physician-rated mimic
hypothesis quality. Samples where 𝐻+was factually incorrect
are excluded (𝑛=47).
gap is closed once 𝐻−down-weights the misleading infection-to-
neurological-complication narrative, allowing re-ranked passages
to explicitly link amantadine to both roles.
Retrieving with 𝐻+alone yields documents where “antiviral”
and “parkinsonism” co-occur in the wrong causal direction. In the
medical corpus, these terms appear far more frequently in the con-
text of drug-induced parkinsonism side effects than in the context
of treatment. All three top-ranked documents for 𝐻+-only retrieval
describe cases where medications caused parkinsonian symptoms
rather than alleviating them (Figure 3, left). Consequently, the gen-
erator defaults to ribavirin, the most prototypical antiviral among
the provided choices, which leads to an incorrect answer.
The mimic hypothesis 𝐻−models the anticipated misinterpreta-
tion where viral encephalitis presents with parkinsonian features.
Subtracting 𝜆H−fromH+suppresses documents aligned with this
infection-centric narrative. As shown in the right box of Figure 3,
CHR instead surfaces documents that explicitly describe amanta-
dine’s repurposed role, providing the generator with the direct
evidence necessary for a correct answer.5 Limitations and Discussion
While CHR effectively suppresses hard negatives, its performance
is inherently tied to the separability of the hypothesis embeddings.
When𝐻+and𝐻−overlap extensively in the corpus narrative, a
phenomenon we termsemantic co-occurrence collapse, subtracting
𝜆H−may inadvertently suppress target-relevant evidence. Our er-
ror analysis reveals that such collapse accounts for approximately
23% of CHR’s failure cases, occurring primarily when 𝐻−closely
paraphrases𝐻+instead of identifying a distinct clinical mimic.
In these instances, the nearly parallel embeddings of the two
hypotheses lead to a negligible difference vector, which strips the re-
triever of its discriminative power. To mitigate this, future research
will investigate an adaptive penalty scheme where 𝜆is dynamically
modulated based on the cosine similarity betweenH+andH−. By
reducing the penalty when hypotheses are semantically close, the
system can preserve informative cues while maintaining aggres-
sive contrastive filtering when the hypotheses are well-separated.
Furthermore, evaluating CHR across different retriever architec-
tures and open-ended medical QA formats remains an important
direction to ensure broader clinical utility.
6 Conclusion
We presented Contrastive Hypothesis Retrieval (CHR), a framework
that bridges clinical differential diagnosis with retrieval mechanism
design. By explicitly modeling what to avoid through a mimic hy-
pothesis𝐻−, CHR addresses the systematic vulnerability of existing
query-expansion methods to hard-negative contamination. Exper-
iments across three medical benchmarks demonstrate that CHR
consistently outperforms state-of-the-art baselines, with retrieval
shift analysis confirming that its gains stem from a fundamental
redirection of retrieval toward qualitatively superior evidence. CHR
offers a practical and efficient approach to enhancing the reliabil-
ity of medical RAG systems, providing a robust foundation for
AI-assisted clinical decision-making.
Acknowledgments
This research was supported by a grant of the Korea Health Technol-
ogy R&D Project through the Korea Health Industry Development
Institute (KHIDI), funded by the Ministry of Health & Welfare,
Republic of Korea (grant number: RS-2025-02213061).
This work was supported by the Starting growth Technological
R&D Program (TIPS Program,(No. RS-2024-00508828)) funded by
the Ministry of SMEs and Startups(MSS, Korea) in 2024.

Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering
Clinical Question
A 65-year-old man presents with gradually worsening rigidity of his arms and legs and slowness in performing tasks. He says he has also noticed hand
tremors, which increase at rest and decrease with focused movements. On examination, the patient does not swing his arms while walking and has
a shortened, shuffling gait. An antiviral drug is prescribed which alleviates the patient’s symptoms. Which of the following drugs was most likely
prescribed?
(A) Amantadine (B) Ribavirin (C) Levodopa (D) Zidovudine
𝐻+(Target Hypothesis)
Parkinson’s disease is characterized by bradykinesia, resting tremor, rigidity, and postural instability due to dopaminergic neuron loss in the substantia
nigra pars compacta; the gold-standard treatment is levodopa, which crosses the blood-brain barrier and is converted to dopamine, directly replenishing
depleted neurotransmitter levels and significantly improving motor symptoms such as rigidity, tremor, and gait disturbances.
𝐻−(Mimic Hypothesis)
Viral encephalitis, particularly from herpes simplex or other neurotropic viruses, can present with parkinsonian features such as rigidity and bradykinesia
due to inflammation in basal ganglia regions; however, it is typically accompanied by fever, altered mental status, seizures, or CSF pleocytosis, and is
treated with antivirals like acyclovir or ribavirin, but these do not reliably improve core motor symptoms of Parkinson’s, and the patient’s chronic,
progressive course without systemic signs makes this unlikely despite transient symptom overlap.
𝐻+only→(B) Ribavirin
Retrieved Documents (Top-3):
(1)“To report a case of Parkinson-like symptoms appearing in a patient after in-
troduction of the anxiolytic buspirone to a treatment regimen that included
the protease inhibitor ritonavir. A 54-year-old white man infected with
human immunodeficiency virus was takingritonavir, zidovudine, and
lamivudinewhen buspirone was added for anxiety. Within days, he devel-
opedsevere rigidity, bradykinesia, and tremor resembling idiopathic
Parkinson’s disease...”
(2)“We present the case of a 78-year-old male who, 16 years ago, was diagnosed
with Parkinson’s disease. His initial symptoms included resting tremor,
bradykinesia, and rigidity. He was treated withstandard dopaminergic
therapy including levodopa and dopamine agonistswith good response.
Approximately 5 years ago, the patient reported gradual improvement in
his motor symptoms despite reduction in his medications...”
(3)“Trazodone is prescribed off-label to treat insomnia, especially in older, not
demented patients. We present a case oftrazodone-induced parkinson-
ismin a 68-year-old woman who developedprogressive rigidity, bradyki-
nesia, and resting tremortwo weeks after initiating trazodone 50 mg
at bedtime for insomnia. The parkinsonian symptoms resolved completely
within three weeks of discontinuing trazodone...”𝐻+−𝜆𝐻−→(A) Amantadine✓
Retrieved Documents (Top-3):
(1)“Parkinson’s disease is a disorder of decreased dopamine-secreting neurons
of the substantia nigra. The gold standard treatment is levodopa (L-DOPA),
a dopamine precursor that crosses the blood-brain barrier and is converted
to dopamine in the brain. However, levodopa is also converted to dopamine
in peripheral tissues, causing side effects like nausea and hypotension. Car-
bidopa inhibits peripheral decarboxylase, allowing more levodopa to reach
the brain...”
(2)“L-DOPA has a counterpart with opposite chirality, D-DOPA. As is true
for many molecules, the human body produces and uses only one of these
enantiomers (in this case, the L-form). L-DOPA crosses the blood-brain
barrier, whereas dopamine itself cannot. Once in the brain, it is converted
to dopamine by aromatic L-amino acid decarboxylase.Amantadine is also
used in Parkinson’s disease as it increases dopamine release and
blocks NMDA receptors...”
(3)“Medications: Levodopa is the mainstay of treatment for motor symptoms of
PD. It is combined with carbidopa to prevent peripheral breakdown. Other
medications include dopamine agonists (pramipexole, ropinirole), MAO-B
inhibitors (selegiline, rasagiline), and amantadine.Amantadine was origi-
nally developed as an antiviral medication but was found to improve
motor symptoms in Parkinson’s disease through dopaminergic ef-
fects and NMDA receptor antagonism...”
Figure 3: Case study showing why the negative (mimic) hypothesis 𝐻−is essential for discriminative retrieval. In the left box,
bolded text highlights misleading co-occurrences of antivirals with parkinsonian symptoms (as side effects, not treatments),
which led to the incorrect answer. In the right box, bolded text highlights the key sentences identifying amantadine as an
antiviral repurposed for Parkinson’s disease, directly leading to the correct answer.
References
[1] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InProceedings of the Twelfth International Conference on Learning Representations.
https://openreview.net/forum?id=hSyW5go0v8
[2] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, et al .2024. The Llama 3
Herd of Models.arXiv preprint arXiv:2407.21783(2024).
[3]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A Survey on RAG Meeting LLMs: Towards
Retrieval-Augmented Large Language Models. InProceedings of the 30th ACMSIGKDD Conference on Knowledge Discovery and Data Mining. 4854–4865. doi:10.
1145/3637528.3671470
[4] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot
Dense Retrieval without Relevance Labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
Association for Computational Linguistics, 1762–1777.
[5] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. REALM: Retrieval-Augmented Language Model Pre-Training. InProceed-
ings of the 37th International Conference on Machine Learning (PMLR, Vol. 119).
3929–3938.

Kim et al.
[6] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn
Song, and Jacob Steinhardt. 2021. Measuring Massive Multitask Language Un-
derstanding. InProceedings of the International Conference on Learning Represen-
tations.
[7]Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter
Szolovits. 2021. What Disease Does This Patient Have? A Large-Scale Open
Domain Question Answering Dataset from Medical Exams.Applied Sciences11,
14 (2021), 6421. doi:10.3390/app11146421
[8] Qiao Jin, Won Kim, Qingyu Chen, Donald C. Comeau, Lana Yeganova, W. John
Wilbur, and Zhiyong Lu. 2023. MedCPT: Contrastive Pre-trained Transformers
with Large-Scale PubMed Search Logs for Zero-Shot Biomedical Information
Retrieval.Bioinformatics39, 11 (2023), btad651. doi:10.1093/bioinformatics/
btad651
[9]Yibin Lei, Yu Cao, Tianyi Zhou, Tao Shen, and Andrew Yates. 2024. Corpus-
Steered Query Expansion with Large Language Models. InProceedings of the
18th Conference of the European Chapter of the Association for Computational
Linguistics (Volume 2: Short Papers). Association for Computational Linguistics,
393–401. doi:10.18653/v1/2024.eacl-short.34
[10] Yibin Lei, Tao Shen, and Andrew Yates. 2025. ThinkQE: Query Expansion via
an Evolving Thinking Process. InFindings of the Association for Computational
Linguistics: EMNLP 2025. 17772–17781. doi:10.18653/v1/2025.findings-emnlp.965
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Informa-
tion Processing Systems, Vol. 33. 9459–9474.
[12] Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, et al .2024. Gemma 2: Im-
proving Open Language Models at a Practical Size.arXiv preprint arXiv:2408.00118
(2024).
[13] Jessica Ryan, Alexander I. Gumilang, Robert Wiliam, and Derwin Suhartono. 2026.
Self-MedRAG: A Self-Reflective Hybrid Retrieval-Augmented Generation Frame-
work for Reliable Medical Question Answering.arXiv preprint arXiv:2601.04531
(2026).
[14] Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon Park, Hyeon Hwang, Mu-
jeen Sung, Hyunjae Kim, and Jaewoo Kang. 2025. Rationale-Guided Retrieval
Augmented Generation for Medical Question Answering. InProceedings of the
2025 Conference of the North American Chapter of the Association for Computa-
tional Linguistics. 12739–12753. doi:10.18653/v1/2025.naacl-long.635
[15] George Tsatsaronis, Georgios Balikas, Prodromos Malakasiotis, Ioannis Partalas,
Matthias Zschunke, Michael R. Albers, Dirk Akkermans, Paul Backfried, Sanne
Hage, Anastasia Krithara, Georgios Paliouras, et al .2015. An Overview of the
BIOASQ Large-Scale Biomedical Semantic Indexing and Question Answering
Competition.BMC Bioinformatics16 (2015), 138. doi:10.1186/s12859-015-0564-6
[16] Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query Expansion with
Large Language Models. InProceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing. Association for Computational Linguistics,
9414–9423. doi:10.18653/v1/2023.emnlp-main.585
[17] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. 2024. Benchmarking
Retrieval-Augmented Generation for Medicine. InFindings of the Association for
Computational Linguistics: ACL 2024. doi:10.18653/v1/2024.findings-acl.372
[18] An Yang, Baosong Yang, Beichen Zhang, et al .2024. Qwen2.5 Technical Report.
arXiv preprint arXiv:2412.15115(2024).

Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering
A Additional Case Study
Figure 4 presents an additional case study from MedQA, demonstrat-
ing that CHR’s discriminative mechanism generalizes effectively
across different clinical domains.
The question examines the complications associated with tamox-
ifen therapy in a postmenopausal breast cancer patient. The correct
answer is “deep venous thrombosis,” a risk driven by tamoxifen’s
estrogen-agonist effects in the liver, which increase the synthesis
of clotting factors. The retrieval challenge stems from the fact that
tamoxifen’s most frequently documented complications in medical
literature involve gynecologic bleeding and endometrial hyperpla-
sia, which act as a dominant retrieval attractor toward the incorrect
answer.
As with the amantadine case discussed in Section 4.5,H+does not
identify the correct answer directly. Instead, it incorrectly describes
tamoxifen’s effects on bone density, claiming it reduces mineral
density in postmenopausal women when it actually helps preserve
it. Despite this factual inaccuracy in the target hypothesis, CHR
successfully produces the correct answer.
When retrieving with 𝐻+alone, the broad “tamoxifen in breast
cancer“ semantics dominate the search, and the corpus-prevalent
narrative of uterine bleeding overwhelms the bone-specific con-
tent. As shown in the left box of Figure 4, two of the top three
retrieved documents focus on abnormal uterine bleeding, leading
the generator to incorrectly select “Increased bleeding.“
The mimic hypothesis 𝐻−explicitly models this bleeding-related
misinterpretation. By subtracting this direction, CHR suppresses
the uterine bleeding literature and surfaces documents focused
on systemic SERM pharmacology. As shown in the right box, all
three re-ranked documents explicitly describe tamoxifen’s hepatic
estrogen-agonist effects on coagulation factor production, directly
identifying venous thromboembolism as the primary risk. This case
highlights the robustness of CHR: even when 𝐻+is factually flawed,
𝐻−correctly identifies the misleading retrieval direction, and the
contrastive mechanism steers the retriever toward discriminative
evidence.
B Sensitivity Analysis of𝜆
The contrastive scoring function (Equation 1) introduces a single hy-
perparameter 𝜆that controls the penalty weight for mimic-aligned
documents. We analyze the sensitivity of CHR to varying 𝜆val-
ues on the MedQA dataset, using Gemma-2-9B-It as the answer
generator.
Figure 5 reports the QA accuracy across a range of 𝜆∈
{0.2,0.4,0.6,0.8,1.0,1.2,1.4}. The results exhibit a clear inverted-U
pattern. At low values ( 𝜆≤0.4), the mimic penalty is insufficient
to suppress hard negatives, leaving performance below the HyDE
baseline. Performance peaks at 𝜆=1.0, where the contrastive vec-
tor effectively steers retrieval away from the mimic-dominated
region while preserving alignment with target-relevant documents,
achieving 53.1% accuracy.
Beyond𝜆= 1.2, over-penalization begins to distort the query
vector, causing accuracy to drop sharply to 43.1% at 𝜆= 1.4. No-
tably, CHR maintains accuracy above 50.4% across a broad plateau
(𝜆∈ [ 0.6,1.2]), consistently outperforming both Standard RAGand HyDE. This stability suggests that CHR is robust to moder-
ate variations in 𝜆and does not require extensive dataset-specific
tuning. Based on these findings, we utilize 𝜆=1.0for all primary
experiments reported in this paper.
C Failure Case: Semantic Co-occurrence
Collapse
Table 5 illustrates a specific failure mode where CHR’s contrastive
mechanism becomes counterproductive due to heavy semantic
co-occurrence between the target and mimic concepts within the
corpus.
The question concerns IL-4, the primary cytokine responsible
for IgE class switching in the context of allergic asthma. CHR cor-
rectly identifies IL-13 as the most plausible mimic for 𝐻−, as both
cytokines are central to Th2-mediated inflammation. However, in
the biomedical literature, IL-4 and IL-13 are discussed almost in-
terchangeably or as a single functional axis because they share
signaling components and biological pathways.
Consequently, the contrastive vectorH+−𝜆H−suppresses not
only the IL-13-specific documents but also many of the high-quality
documents that discuss IL-4 in conjunction with IL-13. This over-
suppression leaves the generator with ambiguous or diluted evi-
dence, whereas a positive-only approach like HyDE succeeds by
focusing strictly on the target keyword without regard for the
mimic-aligned noise. This failure case underscores the challenge
of applying contrastive retrieval when the target and mimic are
fundamentally inseparable within the existing corpus narrative.

Kim et al.
Clinical Question
A 55-year-old woman is found to have an abnormal mass on routine mammography. The mass is biopsied and cytology results are diagnostic for
invasive ductal adenocarcinoma that is estrogen receptor positive. The patient is started on chemotherapy and ultimately has the mass resected. She is
taking tamoxifen and has tolerated the treatment well. Which complication is she now at increased risk for?
(A) Deep venous thrombosis✓(B) Seborrheic keratoses (C) Increased bleeding (D) Osteoporosis
𝐻+(Target Hypothesis)
Tamoxifen, a selective estrogen receptor modulator (SERM) used in estrogen receptor-positive breast cancer, acts as an estrogen antagonist in breast
tissue but as a partial agonist in bone, leading to reduced bone mineral density over time; this increases the risk of osteoporosis and fragility fractures,
particularly in postmenopausal women who already have low estrogen levels, as in this patient who has undergone chemotherapy and may have
treatment-induced menopause.
𝐻−(Mimic Hypothesis)
Tamoxifen is associated with an increased risk of endometrial hyperplasia and endometrial cancer due to its estrogen-agonist effects on the uterus.
However, this patient had a previously treated uterine leiomyoma and no current gynecologic symptoms or abnormal uterine bleeding, making
endometrial malignancy an unlikely complication here, and while bleeding could theoretically occur, it is not a direct complication of tamoxifen therapy
in this context.
𝐻+only→(C) Increased bleeding
Retrieved Documents (Top-3):
(1)“We present a case of adenomatous polyp with degenerative changes caus-
ingabnormal uterine bleeding in a 62-year-old woman undergoing
tamoxifen therapyfor breast cancer. The patient developed vaginal bleed-
ing 18 months after initiating tamoxifen 20 mg daily. Endometrial biopsy
revealed an adenomatous polyp with hemorrhagic degeneration. This case
highlights thattamoxifen’s estrogenic effects on the endometrium can
lead to polyp formation and subsequent bleeding...”
(2)“Benign metastasizing leiomyoma (BML) is a rare disease that occurs in
middle-aged women with a history of uterine leiomyoma. We report a case
of a 54-year-old woman with a history of hysterectomy for uterine fibroids
who presented with multiple pulmonary nodules. The nodules were found to
be histologically identical to her previous uterine leiomyomas. The patient
was treated with aromatase inhibitor therapy given the hormone-responsive
nature of the lesions...”
(3)“We report a case of a 58-year-old woman with a history of invasive lobular
carcinoma of the breast who presented withpostmenopausal vaginal
bleeding. Endometrial curettage revealed metastatic lobular carcinoma.
This case illustrates that inbreast cancer patients on tamoxifen who
present with abnormal bleeding, metastatic disease should be considered
in the differential diagnosis...”𝐻+−𝜆𝐻−→(A) Deep venous thrombosis✓
Retrieved Documents (Top-3):
(1)“Recent data indicate that women with breast cancer receiving aromatase in-
hibitor (AI) therapy experience accelerated bone loss and increased fracture
risk compared to those on selective estrogen receptor modulators (SERMs)
like tamoxifen. However, tamoxifen itself has complex skeletal effects: while
it acts as an estrogen agonist in postmenopausal women, preserving bone
density, it alsoincreases the risk of venous thromboembolism (VTE)
including deep vein thrombosis and pulmonary embolism through
pro-coagulant effects...”
(2)“Selective estrogen receptor modulators (SERMs) represent a class with
a growing number of available agents for osteoporosis prevention and
treatment. These drugs have tissue-selective estrogen agonist or antagonist
effects. In bone, SERMs like raloxifene and tamoxifen act as estrogen agonists,
reducing bone resorption and maintaining bone mineral density. However,
their estrogen-agonist effects in the liver lead to increased production
of clotting factors, elevating the risk of venous thromboembolism,
particularly in the first two years of treatment...”
(3)“Prevention of osteopenia/osteoporosis in postmenopausal patients can re-
duce fracture risk and improve quality of life. SERMs like tamoxifen and
raloxifene have been shown to maintain bone density in postmenopausal
women.The primary concern with SERM therapy is the increased
risk of thromboembolic events, including deep vein thrombosis and
pulmonary embolism, which occurs due to hepatic estrogen-agonist ef-
fects that alter coagulation factor synthesis and increase fibrinogen levels...”
Figure 4: Additional case study from MedQA (oncology domain). In the left box, bolded text highlights how retrieved documents
focus on tamoxifen-related gynecologic bleeding, the dominant complication in the tamoxifen safety literature, which led to
the incorrect answer. In the right box, bolded text highlights evidence describing tamoxifen’s pro-coagulant effects via hepatic
estrogen agonism, directly supporting deep venous thrombosis as the correct answer.

Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering
0.2 0.4 0.6 0.8 1 1.2 1.4404244464850525456
53.1
𝜆Accuracy (%)CHR
HyDE
Std. RAG
Figure 5: Sensitivity of CHR accuracy to the contrastive weight 𝜆on MedQA (Gemma-2-9B-It). The dashed red lines indicate the
robust plateau region𝜆∈[0.6,1.2]where CHR consistently outperforms both baselines. Performance peaks at𝜆=1.0.
Question (MedQA #16):A 7-year-old boy with seasonal allergic asthma is being considered for an experimental therapy targeting a mediator of
antibody class switching. Which mediator is described?
Options:(A) IL-2 (B) IL-10 (C) IL-13(D) IL-4✓
𝐻+:IL-4 is the key cytokine driving Th2 differentiation andB-cell class switching to IgE.
𝐻−:IL-13 overlaps functionally with IL-4 butdoes not directly drive B-cell class switching to IgE.
Prediction: CHR→(C) IL-13(Incorrect);HyDE→(D) IL-4(Correct)
CHR top evidence:
(1) Co-occurringIL-4 and IL-13with IgE switching.
(2) “IL-13, another switch factor for IgE. . . ”
(3) “BothIL13 and IL4are capable of inducing class switching. . . ”HyDE top evidence:
(1) “B cells areswitched to IgE by IL-4. . . ”
(2) Th2-focused evidence emphasizingIL-4-targeted interventions.
(3) Receptor-level discussion (IL-4/IL-13 axis) with clearer IL-4 role.
Analysis:Strong IL-4/IL-13 co-occurrence causes partial overlap between target and mimic directions; subtracting 𝜆H−can therefore suppress
IL-4-relevant evidence and make CHR’s context more ambiguous than HyDE’s.
Table 5: Failure case of semantic co-occurrence collapse. CHR correctly identifies IL-13 as the mimic for IL-4, but heavy corpus
co-occurrence of these cytokines in the same Th2 pathway causes the contrastive subtraction to suppress some target-relevant
evidence.