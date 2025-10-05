# MRAG-Suite: A Diagnostic Evaluation Platform for Visual Retrieval-Augmented Generation

**Authors**: Yuelyu Ji

**Published**: 2025-09-29 03:55:28

**PDF URL**: [http://arxiv.org/pdf/2509.24253v1](http://arxiv.org/pdf/2509.24253v1)

## Abstract
Multimodal Retrieval-Augmented Generation (Visual RAG) significantly advances
question answering by integrating visual and textual evidence. Yet, current
evaluations fail to systematically account for query difficulty and ambiguity.
We propose MRAG-Suite, a diagnostic evaluation platform integrating diverse
multimodal benchmarks (WebQA, Chart-RAG, Visual-RAG, MRAG-Bench). We introduce
difficulty-based and ambiguity-aware filtering strategies, alongside
MM-RAGChecker, a claim-level diagnostic tool. Our results demonstrate
substantial accuracy reductions under difficult and ambiguous queries,
highlighting prevalent hallucinations. MM-RAGChecker effectively diagnoses
these issues, guiding future improvements in Visual RAG systems.

## Full Text


<!-- PDF content starts -->

MRAG-Suite: A Diagnostic Evaluation Platform for Visual
Retrieval-Augmented Generation
Yuelyu Ji
University of Pittsburgh
yuj49@pitt.edu
Abstract
Multimodal Retrieval-Augmented Generation
(Visual RAG) significantly advances question
answering by integrating visual and textual ev-
idence. Yet, current evaluations fail to sys-
tematically account for query difficulty and
ambiguity. We propose MRAG-Suite, a diag-
nostic evaluation platform integrating diverse
multimodal benchmarks (WebQA, Chart-RAG,
Visual-RAG, MRAG-Bench). We introduce
difficulty-based and ambiguity-aware filter-
ing strategies, alongside MM-RAGChecker, a
claim-level diagnostic tool. Our results demon-
strate substantial accuracy reductions under
difficult and ambiguous queries, highlighting
prevalent hallucinations. MM-RAGChecker ef-
fectively diagnoses these issues, guiding future
improvements in Visual RAG systems.
1 Introduction
Retrieval-Augmented Generation (RAG) has
emerged as a foundational paradigm for enhanc-
ing the factuality and reliability of large language
models (LLMs) by grounding their outputs in exter-
nal evidence. With recent advancements in multi-
modal capabilities, visual RAG systems—capable
of leveraging textual and visual information si-
multaneously have garnered significant interest,
promising substantial improvements in answering
complex questions grounded in diverse modalities
such as charts, plots, figures, images, and docu-
ments. Despite these advancements, current evalu-
ation methodologies have not kept pace with these
multimodal developments, primarily focusing ei-
ther on purely textual contexts or closed-domain
multimodal setups that provide pre-selected vi-
sual evidence. Consequently, the research com-
munity lacks rigorous evaluation benchmarks ca-
pable of systematically assessing multimodal re-
trieval accuracy, reasoning fidelity, and robust-
ness to ambiguity and distractors. Existing mul-
timodal benchmarks like ScienceQA (Lu et al.,
Question : Can you identify this animal from  A. Prairie dog B. marmot C. Groundhog D.  
Alpine  marmot?  
Ground Truth Answer: Marmots found in mountainous areas . They have 
characteristic rounded bodies ,… Marmots are well-adapted to life in alpine 
environment and high-altitude environments .
Rag System Output:  rounded body , short legs , small ears , and a brownish-tan fur 
coat. … Marmots live in mountainous areas , have stocky bodies …
Retrieve image 
Correct claims Entailed 
images 
marmot has 
rounded bodies 
marmot has 
short but sturdy 
limbs
marmot find/live 
in mountainous 
areas
Missing claim: Entailed 
images 
marmot live in 
high-altitude 
environments Not entail 
any 
images Figure 1: Illustration of our multimodal claim verifi-
cation process. Given a long-form answer generated
by a retrieval-augmented generation (RAG) system, we
extract atomic factual claims and assess whether each is
supported by retrieved visual evidence. Claims are cate-
gorized asEntailedif supported by at least one image.
In this example, the model correctly describes several
physical traits and habitat features of a marmot , but
omits the gold claim about “high-altitude environments”
, which remains unsupported by the retrieved evidence.
2022), MMMU (Yue et al., 2023), and MMMU-
Pro (Yue et al., 2024) typically provide necessary
visual contexts as part of the inputs, neglecting the
retrieval component critical in real-world scenar-
ios. Meanwhile, retrieval-centric evaluations like
Visual Haystacks (Wu et al., 2024) emphasize chal-
lenging retrieval tasks but limit their scope to a
single domain without fine-grained error diagnos-
tics. Recent text-focused diagnostic frameworks
such as RAGChecker and RefChekcer (Ru et al.,
2024; Hu et al., 2024b) offer detailed claim-level
analysis but remain restricted to textual contexts,
unable to handle the additional complexity intro-
duced by visual evidence.
1arXiv:2509.24253v1  [cs.CL]  29 Sep 2025

To address these critical gaps, we propose
MRAG-Suite, a comprehensive diagnostic eval-
uation platform designed explicitly for visual RAG
systems. MRAG-Suite integrates eight heteroge-
neous datasets, spanning a wide range of real-world
modalities, including charts, plots, scanned docu-
ments, slides, and open-web images. Our bench-
mark systematically introduces difficulty-based fil-
tering and ambiguity-aware query variants, cou-
pled with controlled distractor scenarios, enabling
robust assessment of a model’s capability to handle
ambiguity, retrieve precise evidence, and produce
faithful, evidence-grounded answers.
Specifically, MRAG-Suite contributes:
1.Multi-domain multimodal benchmark: We
consolidate and normalize datasets across di-
verse domains, providing unified evaluation
standards and enabling direct comparisons of
model performance across various multimodal
tasks.
2.Difficulty- and ambiguity-aware filtering:
By filtering trivial or easily guessable ques-
tions and introducing ambiguity through care-
fully constructed query variations, MRAG-
Suite rigorously evaluates robustness against
ambiguity and query complexity.
3.Claim-level multimodal diagnostic tool
(MM-RAGChecker): Extending claim-level
diagnostics to multimodal contexts, MM-
RAGChecker systematically verifies each
claim within generated answers against tex-
tual and visual evidence, enabling precise
identification of hallucinations, modality bi-
ases, and retrieval accuracy.
All code, prompts, and diagnostics are available at:
https://anonymous.4open.science/status/
MRAGChecker-B33D
2 Related Work
Retrieval-Augmented Generation Evaluation
Traditional evaluation of RAG systems primarily
relied on metrics like exact match and ROUGE
scores, which are insufficiently fine-grained to di-
agnose specific failures. RAGChecker (Ru et al.,
2024) introduced claim-level diagnostics for textual
RAG, evaluating retrieval accuracy, factual con-
sistency, and generation hallucinations. However,
these tools remain limited to text-only contexts,
lacking the capability to assess visual evidence in-
tegration and multimodal reasoning.Multimodal QA and Reasoning Benchmarks
Benchmarks such as ScienceQA (Lu et al., 2022),
MMMU (Yue et al., 2023), and their advanced vari-
ants MMMU-Pro (Yue et al., 2024) evaluate mod-
els’ abilities to reason jointly over provided image
and text inputs but exclude retrieval challenges. In
contrast, Visual Haystacks (Wu et al., 2024) tar-
gets multimodal retrieval tasks but is constrained
to single-domain visual reasoning, omitting exten-
sive claim-level diagnostics and failing to capture
multimodal reasoning intricacies across diverse do-
mains(Li et al., 2025b,a).
Biases in Multimodal Retrieval-Augmented Sys-
temsRecent studies reveal significant biases in
multimodal RAG systems. For example, Yao et al.
(Yao et al., 2025) demonstrated strong positional
and modality biases, with models disproportion-
ately relying on initial evidence items or textual
over visual modalities. However, existing bench-
marks have not systematically quantified these bi-
ases across multiple domains under controlled ex-
perimental conditions. MRAG-Suite specifically
addresses this shortcoming by incorporating var-
ied distractor and ambiguity scenarios to measure
these biases explicitly(Wang et al., 2025; Liang
et al., 2025; Jiang et al., 2024; Dan et al., 2024; Lu
et al., 2025; Li et al.).
Domain-specific Multimodal QAPrior work on
multimodal QA has predominantly focused on spe-
cific domains or modalities, such as ChartMRAG
(Yang et al., 2025) and VDocRAG (Tanaka et al.,
2025), demonstrating the feasibility and utility of
retrieval-augmented QA systems in targeted con-
texts. Nonetheless, these domain-specific evalua-
tions lack a unified framework for cross-domain
comparisons and fine-grained diagnostics across
modalities, a gap that MRAG-Suite explicitly aims
to fill.
3 MRAG-Suite Benchmark Construction
3.1 Task Definition
We evaluate multimodal Retrieval-Augmented Gen-
eration (RAG) for open-ended QA. Given a query
q, the system retrieves textual ( Etxt) and visual
(Eimg) evidence and must produce (i) a concise
short answer and (ii) a grounded long answer:
(ashort, along) =f 
q, Etxt, Eimg
.
2

3.2 Corpora Overview
MRAG-Suite merges eight heterogeneous sources
spanning charts, natural photos, open-web images,
scanned PDFs, and slides. Each source contributes
distinct modalities and reasoning demands:
•WebQA(Chang et al., 2022): open-web ques-
tions with both images (captions/descriptions)
and short text snippets; often multi-hop and
open-ended.
•Chart-RAG(Yang et al., 2025)(from
ChartQA (Masry et al., 2022)): chart under-
standing (e.g., “When did X peak?”); evidence
includes the chart image plus OCR/metadata;
answers are numeric/categorical values.
•Visual-RAG (Wu et al., 2025): queries
thatrequirevisual knowledge (ob-
ject/attribute recognition), drawn from
Visual7W/FVQA/MRAG-Bench-style
scenarios.
•MRAG-Bench(Hu et al., 2024a): 1,353
vision-centric MCQ converted to open-ended
by taking the correct option as ashort; empha-
sizes fine-grained and multi-view recognition.
•VisRAG(Yu et al., 2024) subsets: Arxiv (Li
et al., 2024) (scholarly PDF pages),
Plot (Methani et al., 2019) (technical plots),
Slide (Tanaka et al., 2023) (slide decks),
and Document (Tito et al., 2022) (scanned
manuals/forms). Each page is treated as a
visual document with OCR text.
Dataset statistics.Table 1 summarizes domains,
filtering counts, image resolutions, and text length
statistics.
3.3 Normalization and Indexing
All samples are normalized to {question,
short_answer, long_answer, evidence_imgs,
evidence_txts} . Images use CLIP (ViT-L/14)
features plus captions; text passages use dense em-
beddings. Visual items are indexed in FAISS; text
is stored in a DPR/ElasticSearch store.
3.4 Two-Step Filtering for Non-triviality
We remove trivial and guessable items via:
1.Retrieval-Independent Filter: questions
solved by a strong closed-book model (Claude
3.5, confidence >0.9 ) or whose answers ap-
pear verbatim in the question.2.Difficulty-Based Filter: rank remaining
items by multi-hop requirement, modality de-
pendency, and baseline success rate; drop the
easiest∼10% per domain.
We release two query variants:Filtered(mini-
mal, disambiguated) andFull(original, potentially
noisy).
Ambiguity-aware evaluation.To isolate the im-
pact of query ambiguity on Visual RAG, we con-
struct anambiguity-awaresubset (200 items) using
a two-stage pipeline: an LLM-based pre-filter with
short rationales, followed by double-annotator ad-
judication (substantial agreement, κ= 0.74 ). We
also provide evidence-grounded answer rewriting
and disambiguated query rewrites for these items.
Full prompts and guidelines are deferred to Ap-
pendix A.1.
3.5 Result of Distractors
To probe robustness, we evaluate three settings:
•gt_only (GO): only gold evidence;
•gt_plus_distractors (GPD): gold mixed with
CLIP-similar distractors;
•distractors_only (DO): only distractors, test-
ing spurious reasoning.
3.6 Output Specification
Systems must return both short and long answers.
We compute EM/Accuracy for ashort, and claim-
level diagnostics plus ROUGE-L for along(MM-
RAGChecker in Sec. 4).
Having standardized data and retrieval settings,
we next introduceMM-RAGChecker, a claim-
level diagnostic tool that evaluates how long-form
answers are grounded in the retrieved multimodal
evidence. It operates in three stages—claim ex-
traction, evidence matching, and verification—and
produces a set of fine-grained metrics for halluci-
nation, faithfulness, and cross-modality behavior.
Evaluation axes (metric families).We evaluate
along two complementary metric families rather
than by answer length: (i)End-task accuracy
on the short answer ( ashort) via EM/Acc; and (ii)
Claim-level diagnosticson the long answer ( along)
using MM-RAGChecker ; Hallucination, Faith-
fulness, Claim Recall, Context Precision, Self-
Knowledge, and cross-modality metrics).
3

Table 1: Overview of the eight sources in MRAG-Suite, covering domains, filtering counts, image resolution, and
question/answer lengths. The suite spans open-web images, charts/plots, scanned documents, slides, and scholarly
PDFs to enable cross-domain visual RAG evaluation.
Dataset Domain Two-way filter Avg. Img (px) Max Q Max A Avg Q Avg A Ambig.#
MRAG-Bench Natural photos
(animals, objects)845/1353 803×658 20 9 8 2 753/1353
Chart-RAG Business / science
charts2720/4738 466×576 74 81 7 20.7 4093/4738
Visual-RAG Fine-grained natu-
ral images298/378 429×430 39 34 19 5.8 309/378
WebQA Open-web images
& captions19480/21382 792×641 70 99 17.5 12.5 9730/21382
VisRAG-Arxiv Scholarly PDF
pages709/816 1790×1278 40 21 22 12 46/816
VisRAG-Plot Technical plots 794/863 1098×684 45 4 15 8 9/863
VisRAG-Slide Slide decks
(mixed media)525/556 1026×734 39 15 14 10 101/556
VisRAG-Doc Scanned manuals /
forms562/591 1845×2244 24 37 20 4 6/591
4 MM-RAGChecker: Multimodal
Claim-Level Diagnosis
4.1 Overview
Stage 1 — Claim extraction.We split the long
answer alonginto minimal verifiable units C={c i}
using simple heuristics plus an LLM splitter for
compound clauses (e.g., numeric comparisons).
Stage 2 — Per-evidence judging (3-
way).Given images {Ik}Kimg
k=1and text
passages {tk}Ktxt
k=1retrieved for the ques-
tion, we directly judge each pair (ci, e),
e∈ {I k} ∪ {t k}, with a three-way decision
{ENTAILMENT,NEUTRAL,CONTRADICTION} .
No image captions are synthesized and no numeric
scores are used. We cap to top Kimg=3and
Ktxt=3by default and set temperature to zero
for determinism. For each claim we store two
multisets of judgments Jimg
i={j(c i, Ik)}and
Jtxt
i={j(c i, tk)}.
Stage 3 — Aggregation to claim labels.
Each claim receives a single final label Li∈
{ENTAILMENT,NEUTRAL,CONTRADICTION}
via a simple precedence rule: (i)
Li=ENTAILMENT if any j∈ Jimg
i∪ Jtxt
i
is ENTAILMENT; (ii) else Li=CONTRADICTION
if any judgment is CONTRADICTION; (iii) else
Li=NEUTRAL . For modality analysis, define
binary flags simg
i=1iff some j∈ Jimg
iis ENTAIL-
MENT, and analogously stxt
ifor text. We also mark
a retrieved item easusedif there exists a claim
ciwithj(ci, e)=ENTAILMENT . All metrics below
are computed on these final labels and flags.4.2 Metrics
Core metrics.Let Cbe the set of extracted
claims, Lithe final label of ci, andEthe set of
retrieved items. Then
Hallucination Rate =#{i:L i=NEUTRAL}
|C|,(1)
Faithfulness =#{i:L i=ENTAILMENT}
|C|.(2)
LetGbe gold salient claims mined from a refer-
ence long answer.
Claim Recall (CR) =#{goldg∈ Gthat appear inalong}
|G|.
(3)
LetEused ={e∈ E:∃i, j(c i, e) =
ENTAILMENT}.
Context Precision (CP) =|Eused|
|E|.(4)
Self-knowledge measures how many en-
tailed claims did not rely on retrieval:
Self-Knowledge =#{ENTAILMENTc iwith noj(c i,e)=ENTAILMENT∀e∈E}
#{ENTAILMENTc i}
Cross-modality metrics.Let Kimg/Ktxtbe
the numbers of retrieved images/texts. Using
simg
i, stxt
i∈ {0,1}:
∆CR = CR img−CR txt,(5)
∆CP = CP img−CP txt.(6)
Coverage of each modality:
VIS−Hit@k=#retrieved images that
entail at least one claim	
Kimg,(7)
TXT−MissRate = 1−#retrieved text chunks that
entail any claim	
Ktxt.(8)
4

Relevant image Irrelevant but used image Correct 
Wrong 
Missing 
Irrelevant but used doc 
Relevant doc 
Visual Recovery Rate Ground Truth Answer Model Response 
Entailment 
Retrieved Chunks 
Ambiguous query flag 
Modality Claim-Recall Gain Modality Context-Precision Gain Cross-Mod Agreement 
∩ 
∪ 
Context Precision Claim Recall 
Hallucination 
Self-knowledge Faithfulness 
Doc Recovery Rate 
 Context Utilization Generation Metrics Retriever Metrics 
Not Entailment 
Figure 2: MM-RAGChecker overview: claim extraction, multimodal evidence matching, and verification with
per-claim labels and diagnostic metrics.
Agreement and attribution:
CMA =P
i1[simg
i= 1∧stxt
i= 1]P
i1[simg
i= 1∨stxt
i= 1],(9)
V−HR =#{i:simg
i= 1∧L i=NEUTRAL}
#{i:L i=NEUTRAL},(10)
D−HR =#{i:stxt
i= 1∧L i=NEUTRAL}
#{i:L i=NEUTRAL}.(11)
4.3 Retrieval Modes and Distractors
We compare distractors_only (DO),
gt_plus_distractors (GPD), and gt_only
(GO). Table 5 shows that DO degrades perfor-
mance substantially; GPD lies between DO and
GO. The GPD–GO gap quantifies susceptibility to
distractors.
5 Experiments and Results
We evaluate MRAG-Suite along three axes: (1)
end-task accuracy on short answers, (2) factuality
and grounding quality of long answers via MM-
RAGChecker, and (3) robustness to query difficulty,
ambiguity, and distractors.
5.1 Experimental Setup
Models.Our main baseline isLLaVa-v1.6-34B
(vision-language model fine-tuned from LLaMA-
2). Retrieval uses a CLIP-based dual encoder for
images and a DPR retriever for text. At test timewe fetch up to 5 passages and 5 images per query
(when available) and serialize them (image fea-
tures/captions + text) before generation. The model
is prompted to output a short answer followed by a
long answer. For comparison we also runClaude
3.5(API) on a subset to estimate the upper bound
of current proprietary VLMs.
Metrics.For short answers we report
EM/Accuracy. For long answers we com-
pute ROUGE-L against our reference explanations
and all claim-level diagnostics using MM-
RAGChecker (Hallucination Rate, Faithfulness,
Claim Recall, Context Precision, Self-Knowledge,
Modality Bias). We further report the fraction of
retrieved items actually used (evidence utilization).
Retrieval robustness is probed under three modes:
gt_only (GO), gt_plus_distractors (GPD),
anddistractors_only(DO).
5.2 End-task Accuracy (short-answer
EM/Acc)
Table 13 summarizes EM/Accuracy across datasets.
On theFullsplit, LLaV A-2 reaches an overall ac-
curacy of62.5%. After applying our two-step fil-
tering, accuracy drops to47.8%(–14.7 points),
confirming that the retained questions are substan-
tially harder and more retrieval-dependent. Perfor-
mance varies widely by domain: WebQA remains
5

DatasetQwen2.5-VL-72B LLaVa-v1.6-34B Claude 3.5 Phi-4 Pixtral-12B-2409 InternVL3-8B
Filt Full Filt Full Filt Full Filt Full Filt Full Filt Full
Chart-MRAG 22.4% 19.1% 15.3% 25.7% 10.8% 11.9% 11.2% 13.5% 14.7% 29.4% 19.2% 27.6%
MRAG-Bench 32.6% 33.8% 19.5% 24.6% 28.3% 26.1% 32.4% 30.7% 38.8% 37.5% 21.9% 21.4%
VisRAG-ArXiv 55.3% 48.7% 34.6% 34.5% 35.5% 35.1% 38.6% 36.9% 43.1% 43.7% 63.9% 55.4%
VisRAG-Doc 31.1% 35.4% 20.2% 21.9% 28.5% 29.2% 27.6% 28.2% 26.9% 26.6% 39.7% 34.6%
VisRAG-Plot 36.7% 45.8% 12.3% 13.6% 35.9% 36.8% 6.4% 9.2% 51.4% 56.5% 33.5% 36.1%
VisRAG-Slide 18.5% 25.3% 8.7% 10.4% 17.6% 22.1% 24.3% 22.6% 18.2% 15.5% 20.8% 22.9%
Visual-RAG 13.8% 13.1% 9.6% 11.5% 11.7% 10.6% 8.5% 9.2% 12.1% 13.3% 11.2% 12.6%
WebQA 16.2% 14.9% 12.7% 13.4% 14.8% 17.3% 20.6% 22.1% 26.4% 27.9% 17.8% 28.7%
Table 2: EM/Accuracy performance of each model on theFiltered(retained valid samples) andFull(all samples)
splits.
Dataset ModelRecall Precision F1 Halluc. Faith. Self-know. Claim R. Ctx.Prec.
(img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt)
Chart-MRAGQwen2.5-VL-72B 12.1/12.9 7.9/8.1 6.3/7.2 57.3/58.9 54.9/7.8 8.3/7.7 8.9/14.7 85.2/31.8
LLaVa-v1.6-34B 7.6/10.1 0.6/18.9 1.1/13.2 57.4/66.8 50.5/8.8 3.6/10.6 2.0/12.8 86.1/22.3
Claude 3.5 13.3/13.2 0.8/2.1 1.4/3.4 24.9/87.3 65.1/10.6 0.0/2.1 75/10 90/15
MRAG-BenchQwen2.5-VL-72B 16.3 1.2 2.2 45.2 62.6 12.4 11.3 72.1
LLaVa-v1.6-34B 10.9 1.0 1.8 57.8 54.8 3.6 8.5 87.4
Claude 3.5 53.3 4.1 7.6 48.2 66.3 5.7 11.0 79.6
VisRAG-ArXivQwen2.5-VL-72B 23.7 1.9 3.5 30.3 67.2 17.2 23.2 61.0
LLaVa-v1.6-34B 19.7 1.7 3.1 37.0 65.5 12.3 19.4 67.4
Claude 3.5 17.0 1.6 2.9 41.4/0 64.4 9.1 16.9 71.6
VisRAG-DocQwen2.5-VL-72B 16.8 1.5 2.8 45.0 57.5 8.0 11.1 74.3
LLaVa-v1.6-34B 11.0 0.9 1.6 54.3 56.3 4.2 5.6 82.3
Claude 3.5 12.1 1.2 2.2 51.6 57.7 4.7 7.3 80.4
VisRAG-PlotQwen2.5-VL-72B 17.7 1.9 3.4 42.1 65.5 9.6 16.4 75.1
LLaVa-v1.6-34B 9.5 0.8 1.5 60.9 50.1 3.2 1.9 88.6
Claude 3.5 17.1 1.5 2.8 46.7 62.8 8.3 13.5 75.9
VisRAG-SlideQwen2.5-VL-72B 8.7 0.6 1.1 52.7 47.7 4.4 3.2 85.9
LLaVa-v1.6-34B 6.3 0.5 0.9 65.8 46.3 1.2 1.3 92.2
Claude 3.5 8.2 0.5 0.9 55.8 51.3 2.1 1.0 89.6
Visual-RAGQwen2.5-VL-72B 6.5 0.5 1.0 74.0 42.3 2.7 5.2 11.0
LLaVa-v1.6-34B 5.3 0.4 0.8 76.8 44.1 3.2 4.5 12.0
Claude 3.5 6.0 0.6 1.1 85.1 39.5 1.8 6.1 14.5
WebQAQwen2.5-VL-72B 8.8/8.1 0.8/16.1 1.5/10.8 62.8/70.4 52.5/7.9 6.5/3.6 5.9/13.5 89.7/30.4
LLaVa-v1.6-34B 6.5/7.1 0.7/15.8 1.3/9.8 64.5/70.8 50.2/6.2 2.5/4.1 1.1/12.2 91.4/28.6
Claude 3.5 9.0/7.9 1.1/1.8 1.7/3.7 26.6/98.8 62.7/10.7 0.3/0.6 72.0/8.8 87.4/13.6
Table 3: Diagnostic metrics on theFilteredsplit (image/text pairs where applicable).
the easiest for the baseline (52% on the filtered set),
while Chart-RAG is the hardest (low 30%s). Visual-
RAG sits in between ( 45%), indicating persistent
difficulty with fine-grained visual recognition. A
no-retrieval ablation reduces accuracy to25%on
the filtered split, showing that at least half of the
questions truly require external evidence. Manual
inspection of 50 incorrect cases reveals two major
categories: (a)Visual under-utilization(40%): the
model retrieves the correct image/chart but mis-
reads or ignores it; (b)Multi-hop failures(30%):
the model retrieves partial evidence but fails to in-
tegrate it. The rest are simple reasoning mistakes
or retriever misses.5.3 Claim-level Diagnostics
(MM-RAGChecker)
Using MM-RAGChecker, Table 3 summarizes
claim-level diagnostics on theFilteredsplit. For
LLaV A-2,68%of claims are supported (hallu-
cination32%); ROUGE-L is0.61. Faithfulness
varies by domain, with WebQA higher ( ∼72%)
than Visual-RAG (∼60%).
Hallucinations in Long Answers.MM-
RAGChecker shows that about one-third of answer
statements are unsupported. Most areintrinsic
hallucinations—plausible but unevidenced details
(e.g., adding a material or breed not visible in the
image)—whileextrinsic hallucinations(off-topic
filler) are rarer. For instance, in a Visual-RAG
sample the model correctly states that a dog is
catching a frisbee, but hallucinates its color and
6

Dataset Model∆CR∆CP VIS-Hit@k TXT-MissRate CMA V-HR D-HR
Chart-MRAG Phi-4 (GPD) 54.54 59.50 81.38 83.00 13.02 66.87 45.2
WebQA Phi-4 (GPD) 6.59 1.50 14.79 87.00 10.86 12.41 0.33
WebQA Claude 2.5 (GPD) 3.09 -0.35 13.00 83.33 6.52 32.44 3.70
Table 4: Cross-modality diagnostics: differences in claim recall/precision between images and text ( ∆CR/CP), visual
hit rate (VIS-Hit@ k), text miss rate, cross-modal agreement (CMA), and hallucination attribution (V-HR/D-HR).
Higher CMA indicates stronger agreement between modalities.
DatasetQwen2.5-VL-72B LLaVa-v1.6-34B Claude 3.5 Phi-4 InternVL3-8B Pixtral-12B-2409
DO GPD GO DO GPD GO DO GPD GO DO GPD GO DO GPD GO DO GPD GO
Chart-MRAG 22.7 25.4 26.8 10.3 18.1 13.9 8.2 15.5 9.7 15.9 19.8 17.4 23.3 19.5 21.7 14.8 14.1 16.6
MRAG-Bench 30.9 29.6 29.8 17.8 16.4 20.7 26.9 24.1 29.7 2.1 2.4 27.9 26.2 21.3 28.5 33.6 29.8 28.2
VisRAG-ArXiv 4.4 4.1 4.3 4.3 11.7 8.5 7.3 4.1 5.0 1.7 4.4 5.5 6.8 5.1 5.2 3.2 1.4 1.8
VisRAG-Doc 1.2 29.1 31.3 12.9 20.5 20.2 2.4 26.6 25.9 2.3 21.6 28.3 4.7 38.4 38.1 0.6 26.5 27.1
VisRAG-Plot 5.9 33.8 41.4 8.4 12.9 12.6 12.7 25.7 29.3 6.2 6.7 32.1 4.9 28.3 28.4 0.8 51.2 52.0
VisRAG-Slide 1.6 18.7 18.1 1.9 8.4 8.2 4.3 18.9 18.8 1.2 24.6 15.3 0.9 17.5 15.7 0.6 15.3 15.6
Visual-RAG 14.8 26.9 27.3 13.6 18.7 18.1 15.5 25.4 26.5 15.1 21.4 20.2 17.6 23.1 23.2 16.5 23.6 18.9
WebQA 17.2 15.8 15.4 12.3 12.6 13.4 9.9 14.3 16.4 17.5 7.3 12.8 20.1 20.8 17.4 25.6 25.3 0.9
Table 5: Scores (EM preferred; Accuracy used when EM is unavailable) under the three retrieval modes:DO=
distractors_only,GPD=gt_plus_distractors, andGO=gt_only.
breed; our checker flags only those clauses.
Cross-modality behavior.Beyond per-claim
labels, we analyze modality interplay in Ta-
ble 4, using ∆CR/∆CP (image vs. text claim
recall/precision gaps), VIS-Hit@ kand TXT-
MissRate (evidence coverage), CMA (cross-
modality agreement), and V-HR/D-HR (where hal-
lucinations are mistakenly “entailment”). Large
positive ∆CR/∆CP indicates animage-leaning
model, while high TXT-MissRate reveals wasted
textual context. Lower CMA suggests the model
tends to commit to a single modality instead of
reconciling both.
Evidence Utilization and Modality Bias.On
average, the model uses ∼21% of retrieved
pieces—suggesting poor integration. We observe
a bias to stick to either text or image evidence in-
stead of combining both, echoing prior findings
that VLMs over-rely on text. Providing only gold
evidence (GO mode) increases the Supported frac-
tion to ∼90%, indicating distractors significantly
hurt grounding.
Evidence Utilization.Across datasets, LLaV A-
2 uses on average2.1out of 10 retrieved items
(∼21% utilization), indicating under-use of avail-
able context. When provided only gold evidence
(GO mode), the supported claim ratio climbs to
about 90%, implying distractors directly harm
grounding.5.4 Effect of Retrieval Modes
Table 5 shows EM/Accuracy under DO,GPD, and GO.
DOseverely degrades performance (as expected),
while GPDlies between DOandGO. The gap between
GPDandGOquantifies the model’s susceptibility to
distractors; Chart-RAG and VisRAG-Doc exhibit
the largest drops, revealing brittle selection of evi-
dence.
5.5 Prompt Sensitivity Study
We sweep 12 prompt designs (init style, few-shot
exemplar count, reasoning pattern). Table 6 lists
the best-performing prompt per dataset; Figure 3
visualizes the direct vs. retrieve-then-reason trend.
Table 6 reports, for each dataset, the prompt that
achieves the highest mean Accuracy/EM across six
models (filtered split). We observe clear patterns:
Reasoning matters: retrieve_then_reason
dominates on text-heavy retrieval tasks (MRAG,
WebQA), whereas direct is consistently better
on VisRAG-* visual QA splits where context is al-
ready focused.Few-shot count is task-dependent:
Using three exemplars ( ex3) suffices—and some-
times outperforms ex6—for purely visual/numeric
QA (plots/slides). In contrast, ex6 helps more
on document-style and open-web questions that
benefit from pattern coverage.Init style is sec-
ondary:Switching between plain andexpert
seldom changes the rank unless combined with the
right reasoning pattern and example count.
If a single prompt must be used across all
datasets,expert-ex3-directprovides the best over-
7

Dataset Best Prompt Mean Acc. (%)
MRAG-Bench expert3 retrieve & reason 31.5
VisRAG-ArXiv expert6 direct 5.0
VisRAG-Doc plain6 direct 34.3
VisRAG-Plot expert3 direct 36.5
VisRAG-Slide plain3 direct 35.2
WebQA expert6 retrieve&reason 23.0
Table 6: Best-performing prompt (by mean EM/Acc
across models) per dataset on the filtered split. The
patterns highlight when “retrieve-then-reason” helps
(text-heavy tasks) versus when direct prompting is suffi-
cient (focused visual tasks).
Figure 3
all trade-off (top-1 on 3/6 datasets and close second
elsewhere).
5.6 Impact of Distractors on Diagnostic
Metrics
Distractors hurt not only end-task EM/Acc (Ta-
ble 5) but also grounding quality (Table 3). From
GO→GPD /DO, we observe: (i)higher hallucina-
tion(unsupported clauses rise), (ii)lower context
precision(more retrieved items unused or mis-
used), and (iii)stronger modality skew(larger
|∆CR|/|∆CP|and lower CMA), indicating a ten-
dency to commit to one modality under conflicting
evidence. Overall, distractors degrade bothwhatis
answered andhowit is justified.
5.7 Effect of Filtering
Filtering not only reduces accuracy but also
changes answer style. Average long-answer length
increases from45to60words after filtering, yet
factuality does not improve—MM-RAGChecker
flags many of the additional sentences as unsup-
ported. This suggests that harder questions trigger
verbosity rather than better grounding.5.8 Takeaways
(1) Filtering and distractors expose large drops in
both accuracy and grounding. (2) Current VLMs
hallucinate frequently even when retrieval is suc-
cessful; claim-level diagnostics are necessary. (3)
Ambiguity is a major failure source: without ex-
plicit clarification, models either ignore alternative
readings or fuse them incorrectly. (4) Improving
evidence selection and utilization—possibly via
tighter retrieval–generation coupling or in-the-loop
verification—is a promising direction.
6 Conclusion
We presented MRAG-Suite, a comprehensive suite
for evaluating and diagnosing visual retrieval-
augmented generation in multimodal question an-
swering. MRAG-Suite introduces a unified bench-
mark that stresses truly multimodal, knowledge-
intensive QA, combining diverse sources like web
images, textual articles, and informational charts.
To focus evaluation on the intended challenges,
we devised a two-step filtering method to re-
move trivial cases and emphasize multi-hop, vi-
sion+language reasoning. We also contributed an
automated way to generate long-form answers with
evidence, enabling novel evaluation of explana-
tions in addition to factual answers. Crucially, we
developed MM-RAGChecker, a claim-level multi-
modal verifier, to assess the factual consistency of
generated answers.
Limitations
This work targets visual RAG for images, charts,
slides, scanned pages, and scholarly PDFs; results
may not transfer to videos, embodied settings, or
low-resource languages. Hybrid retrieval depends
on CLIP-like encoders and OCR text, so known
dataset/model biases and OCR noise can affect
measured recall/precision and inflate hallucination
labels. Our diagnostics rely on LLM/VLM compo-
nents for claim extraction and entailment; despite
prompt controls and spot-checks, such judges can
diverge from expert assessments in domain-specific
cases. Scores are also sensitive to prompt format,
retrieval k, and distractor sampling; we report set-
tings but do not exhaust all combinations. Some
baselines use third-party APIs, so provider updates
and rate limits may introduce temporal variance
despite fixed prompts and released scripts.
8

References
Yingshan Chang, Mridu Narang, Hisami Suzuki, Gui-
hong Cao, Jianfeng Gao, and Yonatan Bisk. 2022.
Webqa: Multihop and multimodal qa. InProceed-
ings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 16495–16504.
Han-Cheng Dan, Bingjie Lu, and Mengyu Li. 2024.
Evaluation of asphalt pavement texture using multi-
view stereo reconstruction based on deep learning.
Construction and Building Materials, 412:134837.
Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz,
Pan Lu, Kai-Wei Chang, and Nanyun Peng. 2024a.
Mrag-bench: Vision-centric evaluation for retrieval-
augmented multimodal models.arXiv preprint
arXiv:2410.08182.
Xiangkun Hu, Dongyu Ru, Lin Qiu, Qipeng Guo,
Tianhang Zhang, Yang Xu, Yun Luo, Pengfei Liu,
Yue Zhang, and Zheng Zhang. 2024b. Refchecker:
Reference-based fine-grained hallucination checker
and benchmark for large language models.ArXiv,
abs/2405.14486.
Tanqiu Jiang, Zian Wang, Jiacheng Liang, Changjiang
Li, Yuhui Wang, and Ting Wang. 2024. Ro-
bustkv: Defending large language models against
jailbreak attacks via kv eviction.arXiv preprint
arXiv:2410.19937.
Lei Li, Sen Jia, Jianhao Wang, Zhaochong An, Jiaang
Li, Jenq-Neng Hwang, and Serge Belongie. 2025a.
Chatmotion: A multimodal multi-agent for human
motion analysis.arXiv preprint arXiv:2502.18180.
Lei Li, Sen Jia, Jianhao Wang, Zhongyu Jiang, Feng
Zhou, Ju Dai, Tianfang Zhang, Zongkai Wu, and
Jenq-Neng Hwang. 2025b. Human motion instruc-
tion tuning. InProceedings of the Computer Vision
and Pattern Recognition Conference, pages 17582–
17591.
Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong
Feng, Lingpeng Kong, and Qi Liu. 2024. Multimodal
arxiv: A dataset for improving scientific comprehen-
sion of large vision-language models. InAnnual
Meeting of the Association for Computational Lin-
guistics.
Yuqi Li, Chuangang Yang, Hansheng Zeng, Zeyu Dong,
Zhulin An, Yongjun Xu, Yingli Tian, and Hao
Wu. Frequency-aligned knowledge distillation for
lightweight spatiotemporal forecasting.
Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi
Zhu, Tanqiu Jiang, Neil Gong, and Ting Wang.
2025. Graphrag under fire.arXiv preprint
arXiv:2501.14050.
Bingjie Lu, Zhengyang Lu, Yijiashun Qi, Hanzhe Guo,
Tianyao Sun, and Zunduo Zhao. 2025. Predicting
asphalt pavement friction by using a texture-based
image indicator.Lubricants, 13(8):341.Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei
Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark,
and A. Kalyan. 2022. Learn to explain: Multimodal
reasoning via thought chains for science question
answering.ArXiv, abs/2209.09513.
Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty,
and Enamul Hoque. 2022. Chartqa: A benchmark
for question answering about charts with visual and
logical reasoning.arXiv preprint arXiv:2203.10244.
Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and
Pratyush Kumar. 2019. Plotqa: Reasoning over scien-
tific plots.2020 IEEE Winter Conference on Applica-
tions of Computer Vision (WACV), pages 1516–1525.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang,
Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunx-
iang Wang, Shichao Sun, Huanyu Li, and 1 others.
2024. Ragchecker: A fine-grained framework for di-
agnosing retrieval-augmented generation.Advances
in Neural Information Processing Systems, 37:21999–
22027.
Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke
Nishida, Kuniko Saito, and Jun Suzuki. 2025.
Vdocrag: Retrieval-augmented generation over
visually-rich documents.ArXiv, abs/2504.09795.
Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku
Hasegawa, Itsumi Saito, and Kuniko Saito. 2023.
Slidevqa: a dataset for document visual question
answering on multiple images. InProceedings of
the Thirty-Seventh AAAI Conference on Artificial
Intelligence and Thirty-Fifth Conference on Inno-
vative Applications of Artificial Intelligence and
Thirteenth Symposium on Educational Advances in
Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23.
AAAI Press.
Rubèn Pérez Tito, Dimosthenis Karatzas, and Ernest
Valveny. 2022. Hierarchical multimodal transformers
for multi-page docvqa.ArXiv, abs/2212.05935.
Yuhui Wang, Rongyi Zhu, and Ting Wang. 2025.
Self-destructive language model.Preprint,
arXiv:2505.12186.
Tsung-Han Wu, Giscard Biamby, Jerome Quenum,
Ritwik Gupta, Joseph Gonzalez, Trevor Darrell, and
David M. Chan. 2024. Visual haystacks: A vision-
centric needle-in-a-haystack benchmark. InInterna-
tional Conference on Learning Representations.
Yin Wu, Quanyu Long, Jing Li, Jianfei Yu, and Wenya
Wang. 2025. Visual-rag: Benchmarking text-to-
image retrieval augmented generation for visual
knowledge intensive queries.ArXiv, abs/2502.16636.
Yuming Yang, Jiang Zhong, Li Jin, Jingwang Huang,
Jingpeng Gao, Qing Liu, Yang Bai, Jingyuan Zhang,
Rui Jiang, and Kaiwen Wei. 2025. Benchmark-
ing multimodal rag through a chart-based document
question-answering generation framework.arXiv
preprint arXiv:2502.14864.
9

Jiayu Yao, Shenghua Liu, Yiwei Wang, Lingrui Mei,
Baolong Bi, Yuyao Ge, Zhecheng Li, and Xueqi
Cheng. 2025. Who is in the spotlight: The hidden
bias undermining multimodal retrieval-augmented
generation.ArXiv, abs/2506.11063.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2024. Vis-
rag: Vision-based retrieval-augmented generation on
multi-modality documents.ArXiv, abs/2410.10594.
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng,
Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu
Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao
Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan
Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and
3 others. 2023. Mmmu: A massive multi-discipline
multimodal understanding and reasoning benchmark
for expert agi.2024 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages
9556–9567.
Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang,
Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming
Yin, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu
Chen, and Graham Neubig. 2024. Mmmu-pro: A
more robust multi-discipline multimodal understand-
ing benchmark. InAnnual Meeting of the Association
for Computational Linguistics.
A Ambiguity Process
A.1 Ambiguity Definition and Annotation
We curate a 200-question subset to study ambiguity
in multimodal RAG.
Definition.A question isAMBIGUOUSif a rea-
sonable reader cannot determine asinglecorrect an-
swer without additional constraints (time, location,
unit, object, visibility in the image, etc.); otherwise
it isCLEAR. For image-conditioned queries, we
additionally mark as ambiguous when key visual
evidence is not visible or is blurred.
Detection Pipeline.We adopt a two-stage proce-
dure:
1.LLM-based pre-filter.We prompt an asses-
sor model to output CLEAR orAMBIGUOUS to-
gether with a short rationale (30 words). We
use separate prompts for text-only and image-
conditioned questions (see Appendix A.3).
2.Human adjudication.Two annotators inde-
pendently review the flagged items and re-
solve disagreements. We achieve substantial
agreement (Cohen’sκ= 0.74).Rewriting and Answer Normalization.For
each ambiguous item we:
•Evidence-grounded answer rewrite: given re-
trieved evidence and the original answer, an
LLM produces a concise answer or returns
“Evidence inconclusive.” (Appendix A.5).
•Query disambiguation: another prompt
rewrites the question into a precise version
that can be answered solely from the evidence
(Appendix A.6).
The final ambiguity split (200 queries) and a
matched clear set (200 queries) are used in Ap-
pendix. A.7 to isolate the impact of ambiguity on
accuracy, claim recall, and hallucination rate.
A.2 Error Patterns under Ambiguity
On the 200-query ambiguity subset, two dominant
patterns emerge:
1.Single-interpretation bias: the model com-
mits to one reasonable sense, lowering Claim
Recall and short-answer accuracy.
2.Interpretation conflation: retrieval returns
evidence for multiple senses; the generator
merges them, producing composite unsup-
ported claims and higher hallucination rates.
These results motivate anambiguity detection &
clarificationstep before final answer generation.
A.3 Ambiguity Detection (Image-conditioned)
You are an expert question-quality assessor.Your goal is to decide from the question alone whether it unambiguously identifies exactly one correct answer.Ignore any answer or multiple-choice options you see below; pretend you do **not** know whether they are correct.If the question lacks a needed constraint (time, location, unit, object, etc.) →“AMBIGUOUS”.If reasonable readers would arrive at one and only one answer → “CLEAR”.Return only this JSON object:{"status": "CLEAR" | "AMBIGUOUS","explanation": "<≤30 words why>"}Question: "{q_esc}"# The following answer is provided for reference only.# Do NOT rely on it when deciding ambiguity.Answer: "{a_esc}"DETECT AMBIGUOUS QUERY WITHOUT IMAGEYou are an expert multimodal question-quality assessor.Your goal is to decide **from the question plus the attached image**(which may be blurred or obscured) whether it unambiguously identifies exactly one correct answer.Ignore any answer or multiple-choice options you see below; pretend you do **not** know whether they are correct.Special considerations when an image is provided:• If the image is too blurred or key details are not visible → “AMBIGUOUS”.• If the question refers to visual features not visible in the image → “AMBIGUOUS”.• Otherwise, apply the rule that the question lacks a needed constraint (time,location, unit, object, etc.) →“AMBIGUOUS”.If reasonable readers would arrive at one and only one answer → “CLEAR”.Return **only** this JSON object:{"status": "CLEAR" | "AMBIGUOUS", "explanation": "<≤30 words why>"}DETECT AMBIGUOUS QUERY WITH IMAGE 
10

A.4 Ambiguity Detection (Text-only)
You are an expert question-quality assessor.Your goal is to decide from the question alone whether it unambiguously identifies exactly one correct answer.Ignore any answer or multiple-choice options you see below; pretend you do **not** know whether they are correct.If the question lacks a needed constraint (time, location, unit, object, etc.) →“AMBIGUOUS”.If reasonable readers would arrive at one and only one answer → “CLEAR”.Return only this JSON object:{"status": "CLEAR" | "AMBIGUOUS","explanation": "<≤30 words why>"}Question: "{q_esc}"# The following answer is provided for reference only.# Do NOT rely on it when deciding ambiguity.Answer: "{a_esc}"DETECT AMBIGUOUS QUERY WITHOUT IMAGEYou are an expert multimodal question-quality assessor.Your goal is to decide **from the question plus the attached image**(which may be blurred or obscured) whether it unambiguously identifies exactly one correct answer.Ignore any answer or multiple-choice options you see below; pretend you do **not** know whether they are correct.Special considerations when an image is provided:• If the image is too blurred or key details are not visible → “AMBIGUOUS”.• If the question refers to visual features not visible in the image → “AMBIGUOUS”.• Otherwise, apply the rule that the question lacks a needed constraint (time,location, unit, object, etc.) →“AMBIGUOUS”.If reasonable readers would arrive at one and only one answer → “CLEAR”.Return **only** this JSON object:{"status": "CLEAR" | "AMBIGUOUS", "explanation": "<≤30 words why>"}DETECT AMBIGUOUS QUERY WITH IMAGE 
A.5 Evidence-based Answer Rewrite
You are a meticulous answer-generation expert.You will first examine the “Retrieved Evidence” below, then produce a concise,unambiguous answer strictly supported by that evidence.Follow these steps:1. Identify Evidence• Quote or paraphrase the specific sentence(s)/data that directly relate to the question.• If evidence pieces contradict or only partially address the question, label them clearly(e.g., “Conflicting Evidence A / B”) and note the conflict.2. Answer the Question• **If the evidence is clear and consistent**, provide the answer in ≤ 2 sentences.• **If the evidence is related but insufficient or conflicting**, output exactly:Evidence inconclusive.Then add one short sentence (< 20 words) explaining why it is inconclusive(e.g., “figures cover different years”).• **If no relevant evidence exists**, output exactly: Insufficient evidence to answer.3. **Do NOT speculate** beyond the given evidence. Never invent facts.**Question:**{question}**Original Answer (for reference):**{orig_answer}**Retrieved Evidence (bullet-list each item):**{evidence}---Now, produce your evidence-based answer following the instructions above.REWRITE ANSWERA.6 Ambiguous Query Rewrite
You are an expert at rewriting ambiguous queries into precise, unambiguous questionsTHAT CAN BE ANSWERED *SOLELY* FROM THE EVIDENCE BELOW.The evidence includes:- An explanation of why the query was flagged as ambiguous.- All relevant text or image-based evidence needed for answering.EVIDENCE:{evidence}REQUIREMENTS (must all be satisfied):• Embed ANY explicit time-stamp, location, unit and source that the evidence uses.• If evidence gives a percentage, ask for that percentage (not an absolute count) and vice-versa.• For image items, phrase the question so that the answer is inferable from THE PROVIDED IMAGE alone.• Avoid superlatives/comparatives unless the evidence contains a unique maximum/minimum.• Produce ONLY the rewritten question text on one line – no bullets, no “Rewritten Question:” label.1. The new question must NOT contain phrases like “according to the evidence / image / chart / passage”.2. Do NOT refer to “the following image” or “the study” unless you embed the study’s full name, year, and author in your question text.3. If the answer comes from quantitative data, embed the dataset’s year OR the exact figure you expect, so only one answer is possible.4. Avoid vague temporal words (“recent”, “modern”, “in the 2000s”); give exact years.5. The rewritten text must be **just the question**, one sentence if possible.ORIGINAL Q → "{raw_question}"ORIGINAL A → "{raw_answer}"REWRITE:REWRITE QUERY
Table 7: Ambiguity subset construction statistics. We
report candidate counts, items flagged as ambiguous by
the two-stage pipeline, and the final curated set (200).
Candidates Marked Ambiguous Final Set
Text-only 26,120 13,823 184
Image-conditioned 4,557 1,224 16
Total 30,677 15,047 200
A.7 Ambiguity Subset Results
On the 200-query ambiguous subset, short-answer
EM drops by5.9points compared to clear queries,
while hallucination rate increases by9.5points.
Claim Recall and Context Precision also decrease,
indicating single-interpretation bias or interpreta-
tion conflation; see Appendix. A.2 for qualitative
patterns. Table 8 reports the aggregate metrics.
Also, we have the ambiguous set in the code pro-
vided.
Table 8: Results on the 200-query ambiguity subset
versus a matched clear set. We report EM/Accuracy,
Hallucination Rate (lower is better), Claim Recall, and
Context Precision, isolating the effect of query under-
specification on retrieval and grounding.
EM/Acc↑Halluc.↓Claim Recall↑Ctx. Prec.↑
Clear (200 random) 23.5 49.2 12.4 14.1
Ambiguous (200) 17.6 58.7 9.1 12.5
∆(Ambig−Clear)-5.9 +9.5 -3.3 -1.6
B Retrieve system settings
11

Dataset Text Retriever Image Retriever
WebQA SBERT (mpnet) CLIP ViT-B/32
Chart-RAG DPR (ctx-enc) CLIP ViT-B/32
MRAG — CLIP ViT-B/32 (img+txt avg)
VisRAG VisRAG-Ret —
VisualRAG — CLIP ViT-B/32
Table 9: Backbones used in the hybrid retriever for each
sub-corpus. Images are indexed with CLIP variants; text
uses DPR/SBERT; VisRAG documents use a document-
image retriever.
C 12 Prompt Experiment Settings
Nickname init_style ex_style Example template
expert1 expert ex1 Q: {q}
A: {ans}
expert2 expert ex3 Question + brief Expla-
nation + Final Answer
expert3 expert ex6 Dialog style: User: {q}
/ Assistant: {ans}
plain1 plain ex1 Q: {q}
A: {ans}
plain2 plain ex3 Question + brief Expla-
nation + Final Answer
plain3 plain ex6 Dialog style: User: {q}
/ Assistant: {ans}
Table 11: Mapping from prompt nicknames to template
families used in the 12-way prompt sweep. Variations
differ in initialization style and exemplar formatting.
D Distractor results overview
12

Layer Arg name Values in this paper Template / Semantics
System Promptinit_style plain plain : “Please read the following
question and retrieve the relevant doc-
ument(s)/image(s) to produce the best
possible answer.”
expert expert : “You are an expert in <DOMAIN>.
Analyze the question carefully, think step-
by-step about which sources to use, and
present your answer clearly at the end.”
Few-shot Demoexample_style ex1,ex3,ex6 Three fixed 0-shot style exemplars (chosen
from the pool). Content differs in word-
ing/format hints.
max_examples0 or 5 Whether to insert 5 illustrative QA pairs
(kept at 0 in our sweep).
Reasoning Instructiondirect direct : “Question: . . . Answer:” (no man-
dated reasoning steps).
reasoning_process retrieve_then_reason : “First outline re-
trieval/reasoning, then give final answer.”
Answer Formatanswer_format freeFree-form text (no forced JSON or letter).
Context Packagingcontext_order img_first Images precede text in the prompt (fixed in
our runs).
include_doc_ids0 / 1 Whether to show doc/image IDs (0 in our
runs).
Table 10: Prompt dimensions and concrete templates used in our 12-way sweep.
p_reason Reasoning block (excerpt) COT line?
direct Answer directly without explaining your reasoning. No
retrieve_then_reason First list which retrieved snippets/images you will use
(by Doc/Image IDs). Then reason step by step using
them, and finally give the answer.Yes
structured Follow this structure: 1) Question understanding 2)
Relevant sources (IDs or short quotes) 3) Reasoning 4)
Final AnswerYes
plan_execute Outline a short plan first (bullet points). After the plan,
execute it and derive the final answer.Yes
verify Propose an initial answer. Then self-check it briefly.
If it passes, output the final answer prefixed with “FI-
NAL:”.Yes
none (no additional block) No
Table 12: Reasoning patterns injected into prompts. Variants differ in whether to expose chain-of-thought–like
structure and explicit self-verification before the final answer.
DatasetQwen2.5-VL-72B LLaVa-v1.6-34B Claude 3.5 Phi-4 Pixtral-12B-2409 InternVL3-8B
Filt Full Filt Full Filt Full Filt Full Filt Full Filt Full
Chart-MRAG 22 19 15 25 10 11 13 12 14 29 19 27
MRAG-Bench 32 33 19 24 28 26 23 24 38 37 21 21
VisRAG-ArXiv 55 48 34 34 35 35 26 27 43 43 63 55
VisRAG-Doc 31 35 20 21 28 29 24 22 26 26 39 34
VisRAG-Plot 36 45 12 13 35 36 6 9 51 56 33 36
VisRAG-Slide 18 25 8 10 17 22 24 22 18 15 20 22
Visual-RAG 13 13 9 11 11 10 8 9 12 13 11 12
WebQA 16 14 12 13 14 17 10 11 26 27 17 28
Table 13: EM/Accuracy performance (%) of each model on theFilteredandFullsplits.
13

Dataset Model DistractorRecall Precision F1 Halluc. Faith. Self-know. Claim R. Ctx.Prec.
(img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt)
VisRAG-ArXivQwen2.5-VL-72B distractors only 21.8 6.7 7.9 66.2 23.5 5.9 20.4 31.0
gt_plus_distractors 29.6 8.9 10.8 60.1 28.9 7.1 24.1 33.5
gt_only 33.4 10.8 12.7 55.3 32.1 8.2 25.9 35.8
LLaVa-v1.6-34B distractors only 18.4 5.9 6.8 69.4 20.1 4.8 18.0 28.2
gt_plus_distractors 24.7 7.6 9.1 63.5 24.9 5.9 21.6 30.4
gt_only 28.9 9.2 10.9 58.2 29.6 6.8 23.7 33.1
Claude 3.5 distractors only 30.4 9.2 10.1 61.4 28.0 7.6 29.9 40.0
gt_plus_distractors 40.7 10.6 14.3 56.0 37.0 7.0 25.6 33.0
gt_only 35.9 8.6 11.3 57.7 36.5 4.7 27.6 39.0
Phi-4 distractors only 22.5 15.3 13.7 50.8 20.6 12.9 22.2 33.8
gt_plus_distractors 34.4 24.0 23.5 45.3 22.3 16.3 26.0 36.0
gt_only 37.8 27.2 26.6 41.1 24.7 18.5 27.9 37.9
Pixtral-12B-2409 distractors only 17.6 7.1 6.3 70.5 18.7 5.1 16.0 27.6
gt_plus_distractors 25.9 10.4 9.7 63.0 23.3 7.4 18.9 29.5
gt_only 30.5 12.8 12.0 58.9 25.6 8.3 21.5 31.2
InternVL3-8B distractors only 26.5 14.6 12.6 59.3 24.0 10.8 28.8 37.0
gt_plus_distractors 39.2 15.7 16.5 54.4 30.0 10.6 29.0 38.0
gt_only 40.3 14.6 17.6 48.2 27.4 9.5 24.4 35.0
VisRAG-DocQwen2.5-VL-72B distractors only 10.8 4.0 3.6 57.9 13.4 2.6 21.0 23.5
gt_plus_distractors 46.7 31.9 27.2 28.1 22.7 20.4 25.9 27.0
gt_only 52.9 36.2 31.0 24.4 24.5 22.6 24.1 25.1
LLaVa-v1.6-34B distractors only 9.4 3.7 3.2 60.1 12.1 2.3 19.5 22.0
gt_plus_distractors 39.8 26.1 22.0 33.9 20.1 16.4 23.8 25.6
gt_only 44.0 28.9 24.7 30.2 22.4 18.5 22.9 24.2
Claude 3.5 distractors only 12.0 4.6 4.2 38.2 16.9 3.0 24.5 26.0
gt_plus_distractors 75.5 35.2 38.2 41.4 32.2 25.4 27.7 31.0
gt_only 68.8 33.6 33.7 41.5 30.7 23.8 26.5 28.0
Phi-4 distractors only 18.1 12.2 11.0 27.6 7.5 10.5 20.8 22.7
gt_plus_distractors 45.3 37.8 37.0 15.2 4.7 32.3 24.5 26.0
gt_only 50.6 41.9 41.1 13.0 3.9 35.2 23.6 24.8
Pixtral-12B-2409 distractors only 7.8 3.5 3.0 61.3 10.6 2.1 18.8 21.3
gt_plus_distractors 34.4 25.6 20.4 36.8 17.7 15.1 22.1 23.9
gt_only 39.1 29.7 24.1 32.6 19.8 17.2 21.5 22.8
InternVL3-8B distractors only 8.0 4.1 3.4 55.2 12.8 3.1 22.5 25.0
gt_plus_distractors 68.3 65.2 26.5 15.8 25.6 45.5 28.0 26.5
gt_only 60.3 77.8 57.8 17.2 27.3 54.5 21.5 23.0
MRAG-BenchQwen2.5-VL-72B distractors only 22.6 6.2 5.1 79.4 14.3 4.0 7.3 4.5
gt_plus_distractors 28.4 7.9 6.7 75.8 16.2 4.7 8.9 5.2
gt_only 33.7 9.5 8.4 71.1 18.6 5.3 10.4 6.0
LLaVa-v1.6-34B distractors only 21.1 5.3 4.1 82.4 13.2 3.5 8.2 4.2
gt_plus_distractors 27.8 4.8 3.1 77.1 17.6 4.3 10.1 6.0
gt_only 31.9 6.7 4.6 73.4 19.1 4.8 9.8 5.0
Claude 3.5 distractors only 42.2 3.9 4.6 73.5 19.8 3.8 9.2 5.5
gt_plus_distractors 47.2 2.8 3.9 80.5 16.1 2.4 7.7 6.0
gt_only 47.7 4.2 5.6 78.8 17.1 4.1 5.5 3.0
Phi-4 distractors only 14.5 12.6 10.8 36.4 8.8 11.9 7.1 4.6
gt_plus_distractors 18.7 18.3 16.2 31.4 9.3 16.3 8.6 5.5
gt_only 22.6 21.4 18.7 27.8 10.4 18.5 9.5 6.2
Pixtral-12B-2409 distractors only 12.8 6.9 5.7 84.6 11.0 5.6 6.4 3.9
gt_plus_distractors 17.2 8.8 7.2 80.3 12.5 6.4 7.3 4.2
gt_only 21.0 10.6 8.6 76.1 14.3 7.2 8.4 4.9
InternVL3-8B distractors only 35.3 37.6 31.0 54.5 9.3 35.1 5.3 3.0
gt_plus_distractors 29.0 33.1 25.1 46.7 8.2 32.1 5.3 3.5
gt_only 31.8 31.6 25.6 62.0 7.5 28.6 6.2 4.0
VisRAG-PlotQwen2.5-VL-72B distractors only 10.1 3.9 2.6 61.7 12.2 2.7 9.4 10.2
gt_plus_distractors 17.6 7.2 5.8 58.5 11.6 5.4 9.7 9.6
gt_only 23.8 10.7 8.2 56.0 12.9 7.7 8.6 8.8
LLaVa-v1.6-34B distractors only 7.9 3.1 1.9 64.9 10.7 2.1 9.1 10.4
gt_plus_distractors 13.4 6.1 4.0 61.2 11.5 3.9 9.5 9.8
gt_only 19.6 8.3 5.7 58.9 12.1 5.5 8.9 9.3
Claude 3.5 distractors only 8.0 4.1 2.1 55.1 10.1 2.8 10.5 11.0
gt_plus_distractors 19.8 9.4 6.6 77.8 13.4 8.9 9.0 9.0
gt_only 26.5 11.6 9.4 75.2 12.7 11.1 7.5 8.0
Phi-4 distractors only 11.3 4.7 1.8 83.7 10.3 4.0 10.0 10.0
gt_plus_distractors 16.8 8.9 3.7 79.5 11.2 6.8 9.6 9.7
gt_only 22.4 11.5 5.2 76.4 12.0 9.4 8.4 8.7
Pixtral-12B-2409 distractors only 6.5 2.8 1.1 66.0 9.8 1.7 8.8 9.5
gt_plus_distractors 12.9 6.4 2.9 62.1 10.5 3.7 9.2 9.6
gt_only 18.7 8.9 4.6 59.6 11.2 5.2 8.7 9.0
InternVL3-8B distractors only 2.3 1.8 0.4 52.3 11.0 1.7 8.5 9.0
gt_plus_distractors 25.5 15.1 7.9 72.9 9.3 13.8 10.0 10.0
gt_only 21.5 11.2 3.2 76.5 8.2 10.3 11.1 12.0
VisRAG-SlideQwen2.5-VL-72B distractors only 12.6 5.5 4.8 63.3 14.7 3.8 51.1 56.8
gt_plus_distractors 41.3 27.2 26.1 36.5 42.2 21.1 46.9 50.6
gt_only 49.7 34.9 33.5 31.2 48.4 23.2 43.8 48.0
LLaVa-v1.6-34B distractors only 10.9 4.9 4.1 66.0 13.3 3.3 49.6 54.3
gt_plus_distractors 35.6 22.7 21.0 40.2 39.1 18.8 45.1 49.2
gt_only 42.8 29.6 27.4 35.5 43.7 20.6 41.9 46.6
Claude 3.5 distractors only 23.1 12.0 11.2 53.3 36.7 9.1 53.3 59.1
gt_plus_distractors 78.8 30.6 37.3 38.7 45.8 15.4 55.7 61.1
gt_only 64.2 39.1 36.0 30.1 49.2 18.9 49.2 56.0
Phi-4 distractors only 19.9 11.7 10.3 41.2 31.6 8.6 50.1 55.6
gt_plus_distractors 71.8 42.7 44.5 33.7 40.1 23.4 45.9 48.6
gt_only 60.4 47.9 45.0 29.0 37.7 26.1 43.6 46.2
Pixtral-12B-2409 distractors only 9.6 5.1 4.2 67.9 12.0 3.4 48.5 53.7
gt_plus_distractors 29.8 18.9 16.7 45.6 34.7 16.3 44.3 47.5
gt_only 36.7 24.5 22.1 39.8 38.9 17.9 41.1 45.3
InternVL3-8B distractors only 17.2 4.4 5.1 58.7 34.0 3.2 54.1 59.7
gt_plus_distractors 67.1 42.6 41.9 23.4 51.4 22.5 47.3 51.4
gt_only 57.8 41.1 38.7 28.8 44.6 23.8 40.8 45.8
WebQAQwen2.5-VL-72B distractors only 14.0/13.1 6.5/5.8 5.0/4.3 69.1/71.2 9.4/8.6 5.0/5.4 12.8/12.0 14.0/13.2
gt_plus_distractors 18.3/17.2 8.9/8.1 7.3/6.5 66.5/68.7 11.2/10.3 6.3/6.8 12.0/11.3 13.5/12.6
gt_only 22.4/21.0 11.3/10.2 9.1/8.0 62.8/64.9 13.0/12.0 7.4/8.0 11.2/10.5 12.7/11.8
LLaVa-v1.6-34B distractors only 12.6/11.4 5.7/5.0 4.5/3.8 71.6/73.9 8.0/7.2 4.8/5.1 12.1/11.3 13.1/12.1
gt_plus_distractors 16.8/15.6 8.2/7.4 6.6/5.7 68.7/70.8 10.4/9.5 5.9/6.3 11.3/10.6 12.7/11.8
gt_only 21.3/19.9 10.5/9.4 8.4/7.3 65.2/67.3 12.6/11.6 7.0/7.6 10.7/10.0 12.0/11.2
Claude 3.5 distractors only 23.0/21.5 4.7/4.3 5.2/4.4 63.4/86.4 31.9/4.1 3.7/3.5 12.6/4.8 15.0/12.0
gt_plus_distractors 22.4/20.7 5.9/4.3 5.4/4.7 61.1/87.2 30.9/4.4 4.9/3.3 10.1/7.0 13.0/16.5
gt_only 26.8/24.1 6.8/5.0 6.6/5.4 57.6/84.3 33.5/5.2 5.5/3.9 11.2/8.1 14.2/17.8
Phi-4 distractors only 14.2/12.7 22.0/19.1 8.7/7.5 53.4/60.2 9.1/0.7 20.1/18.5 10.9/4.7 13.6/12.3
gt_plus_distractors 10.4/12.5 25.6/22.3 9.5/11.1 49.7/56.5 8.4/0.5 23.9/22.0 11.8/5.2 14.5/13.0
gt_only 16.1/14.8 29.0/24.7 12.1/12.4 46.2/53.0 10.2/0.9 26.7/24.1 12.3/5.8 15.9/13.9
Pixtral-12B-2409 distractors only 10.5/ 9.6 15.3/14.1 6.4/5.7 58.9/65.2 6.9/1.0 13.8/12.6 9.6/3.9 12.0/10.9
gt_plus_distractors 13.4/12.6 19.6/17.4 8.7/7.5 55.2/61.8 8.0/1.2 17.3/15.5 10.5/4.4 13.2/11.9
gt_only 17.4/16.0 23.5/20.7 11.3/9.8 51.7/58.1 9.6/1.6 20.4/18.0 11.8/5.0 14.7/13.0
InternVL3-8B distractors only 27.9/27.3 13.8/14.6 14.7/12.8 68.1/77.2 13.8/8.8 13.1/11.9 13.0/4.5 14.0/10.5
gt_plus_distractors 32.0/31.0 15.4/15.3 17.0/15.1 69.0/74.4 14.2/8.4 14.8/12.2 11.2/6.3 14.0/14.0
gt_only 16.3/16.9 7.0/6.9 5.6/5.1 64.0/63.6 8.9/12.0 6.1/5.3 8.5/6.4 11.5/13.5
Table 14: Full MM-RAGChecker diagnostics across datasets, models, and retrieval modes. For each setting we
report per-modality (image/text) Recall, Precision, F1, Hallucination, Faithfulness, Self-Knowledge, Claim Recall,
and Context Precision.
14

Dataset Model DistractorRecall Precision F1 Halluc. Faith. Self-know. Claim R. Ctx.Prec.
(img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt) (img/txt)
ChartMRAG-BenchQwen2.5-VL-72B distractors only 11.0/10.0 4.2/3.7 2.9/2.4 60.6/62.8 12.8/11.9 2.6/2.9 9.8/9.1 10.6/10.0
gt_plus_distractors 18.4/17.1 7.6/6.9 6.2/5.4 57.1/59.4 12.2/11.1 5.1/5.6 10.1/9.4 10.0/9.7
gt_only 24.6/23.1 11.1/10.3 8.6/7.7 54.6/56.9 13.4/12.6 7.4/8.1 9.0/8.3 9.2/8.9
LLaVa-v1.6-34B distractors only 8.6/7.6 3.4/2.9 2.2/1.7 63.7/65.8 11.2/10.5 2.0/2.3 9.5/8.8 10.7/10.2
gt_plus_distractors 14.2/13.0 6.5/5.8 4.3/3.5 60.1/62.3 12.0/11.0 3.8/4.2 9.9/9.2 10.2/9.8
gt_only 20.7/19.2 8.9/8.1 6.1/5.3 57.8/60.1 12.6/11.8 5.4/6.0 9.2/8.6 9.6/9.1
Claude 3.5 distractors only 8.6/7.5 4.3/3.9 2.3/1.9 54.0/56.0 10.4/9.8 2.7/3.0 10.8/10.2 11.2/10.7
gt_plus_distractors 20.7/18.9 9.8/8.9 6.9/6.2 76.6/78.9 13.8/12.7 9.2/9.8 9.2/8.9 9.2/8.8
gt_only 27.5/25.4 12.2/11.0 9.7/8.9 74.3/76.6 13.1/12.2 11.2/11.9 7.7/7.3 8.2/7.8
Phi-4 distractors only 12.1/10.6 4.9/4.4 2.0/1.7 82.4/84.7 10.9/9.8 3.7/4.0 10.2/9.7 10.2/9.9
gt_plus_distractors 17.6/16.0 9.4/8.5 4.1/3.4 78.3/80.6 11.7/10.8 6.5/7.0 9.9/9.3 9.9/9.5
gt_only 23.3/21.6 12.0/10.9 5.5/4.9 75.1/77.4 12.5/11.6 9.1/9.7 8.7/8.1 8.9/8.5
Pixtral-12B-2409 distractors only 7.2/6.0 3.1/2.6 1.3/1.0 64.8/66.9 10.3/9.4 1.6/1.9 9.1/8.5 9.8/9.3
gt_plus_distractors 13.7/12.2 6.7/5.9 3.1/2.7 60.9/63.1 11.0/10.1 3.6/3.9 9.5/8.9 9.9/9.4
gt_only 19.6/18.0 9.3/8.2 4.9/4.3 58.6/60.8 11.7/10.8 5.1/5.6 9.1/8.5 9.3/8.9
InternVL3-8B distractors only 2.6/2.1 1.9/1.6 0.5/0.4 51.1/53.5 11.3/10.7 1.6/1.8 8.7/8.3 9.2/8.8
gt_plus_distractors 26.5/24.6 15.7/14.4 8.4/7.5 71.8/74.0 9.7/8.9 14.3/15.0 10.1/9.7 10.1/9.8
gt_only 22.3/20.8 11.6/10.7 3.5/2.9 75.4/77.5 8.5/7.8 10.6/11.1 11.4/10.8 12.1/11.4
Visual-RAGQwen2.5-VL-72B distractors only 18.4 6.3 6.8 42.1 27.2 48.3 4.9 18.2
gt_plus_distractors 44.5 14.7 16.8 36.3 34.5 53.9 6.8 25.0
gt_only 41.3 13.2 15.4 37.5 32.8 51.1 6.5 23.3
LLaVa-v1.6-34B distractors only 16.1 5.2 5.7 43.7 25.4 45.0 4.4 16.1
gt_plus_distractors 42.0 12.9 14.7 38.8 30.1 51.6 6.0 22.7
gt_only 38.8 11.8 13.2 39.5 28.0 50.2 5.7 21.4
Claude 3.5 distractors only 23.3 7.0 8.4 40.2 28.6 50.3 6.2 20.0
gt_plus_distractors 52.7 16.5 19.7 35.0 36.7 55.5 7.5 28.8
gt_only 48.1 15.2 17.4 36.1 34.9 53.2 7.1 26.5
Phi-4 distractors only 20.7 6.5 7.3 39.0 26.0 46.4 5.3 18.0
gt_plus_distractors 47.2 15.0 17.6 34.1 32.5 52.9 6.7 25.2
gt_only 43.6 13.9 16.0 35.3 31.0 50.7 6.4 23.7
Pixtral-12B-2409 distractors only 15.3 5.8 6.2 44.9 23.5 43.2 4.1 15.3
gt_plus_distractors 40.8 12.0 13.9 38.2 29.8 49.8 5.8 21.9
gt_only 37.9 11.1 12.8 39.0 27.7 48.1 5.5 20.4
InternVL3-8B distractors only 22.6 8.1 9.2 41.3 30.0 49.0 6.0 19.5
gt_plus_distractors 50.1 17.2 20.3 33.7 37.8 54.4 7.8 27.1
gt_only 45.7 15.5 18.1 34.9 35.3 52.0 7.3 25.0
Table 15
15