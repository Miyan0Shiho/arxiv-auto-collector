# Investigating LLM Capabilities on Long Context Comprehension for Medical Question Answering

**Authors**: Feras AlMannaa, Talia Tseriotou, Jenny Chim, Maria Liakata

**Published**: 2025-10-21 14:50:24

**PDF URL**: [http://arxiv.org/pdf/2510.18691v1](http://arxiv.org/pdf/2510.18691v1)

## Abstract
This study is the first to investigate LLM comprehension capabilities over
long-context (LC) medical QA of clinical relevance. Our comprehensive
assessment spans a range of content-inclusion settings based on their
relevance, LLM models of varying capabilities and datasets across task
formulations, revealing insights on model size effects, limitations, underlying
memorization issues and the benefits of reasoning models. Importantly, we
examine the effect of RAG on medical LC comprehension, uncover best settings in
single versus multi-document reasoning datasets and showcase RAG strategies for
improvements over LC. We shed light into some of the evaluation aspects using a
multi-faceted approach. Our qualitative and error analyses address open
questions on when RAG is beneficial over LC, revealing common failure cases.

## Full Text


<!-- PDF content starts -->

Investigating LLM Capabilities on Long Context Comprehension for
Medical Question Answering
Feras AlMannaa1, Talia Tseriotou2, Jenny Chim2, Maria Liakata2,3
1Istanbul Aydın University
2Queen Mary University of London
3The Alan Turing Institute
{ferasmhdaymanalmanna}@stu.aydin.edu.tr,{t.tseriotou,m.liakata}@qmul.ac.uk
Abstract
This study is the first to investigate LLM
comprehension capabilities over long-context
(LC) medical QA of clinical relevance. Our
comprehensive assessment spans a range of
content-inclusion settings based on their rele-
vance, LLM models of varying capabilities and
datasets across task formulations, revealing in-
sights on model size effects, limitations, un-
derlying memorization issues and the benefits
of reasoning models. Importantly, we exam-
ine the effect of RAG on medical LC compre-
hension, uncover best settings in single versus
multi-document reasoning datasets and show-
case RAG strategies for improvements over LC.
We shed light into some of the evaluation as-
pects using a multi-faceted approach. Our qual-
itative and error analyses address open ques-
tions on when RAG is beneficial over LC, re-
vealing common failure cases.
1 Introduction
Large Language Models (LLMs) (Achiam et al.,
2023; Dubey et al., 2024; Yang et al., 2024) per-
form impressively on medical tasks (Borgeaud et
al., 2021; Nori et al., 2023), achieving superhuman
scores on USMLE-style exam question-answering
(QA) (Chen et al., 2023; Tang et al., 2023; Pal
and Sankarasubbu, 2024; Singhal et al., 2025). Yet
multiple-choice QA (MCQA) results are not indica-
tive of performance in more complex tasks (Arias-
Duart et al., 2025), including open-ended medical
QA (Sandeep Nachane et al., 2024) and long-form
generation. Furthermore, board exam questions
rely primarily on textbook knowledge, likely ex-
posed during pre-training (Chen et al., 2025).
Beyond MCQA, researchers have constructed
datasets based on authentic de-identified electronic
health records (EHRs). While these benchmarks
still face challenges regarding data diversity, eco-
logical, external, and construct validity (Wornow et
al., 2023), they provide crucial signals on how wellmodels can address complexities in real data, in-
cluding domain-specific vocabulary, heterogeneous
document types, multi-document reasoning, data
noise, linguistic diversity, long contexts, and long-
range dependencies.
To date most EHR-based benchmarks involve
tasks that only require single-document contexts
(Pampari et al., 2018; Yue et al., 2020). In practice,
there is no guarantee that the necessary informa-
tion (e.g. to perform a diagnosis) can be found in
a single document known a priori; what is needed
is comprehension and reasoning over largely noisy
and at best complimentary records. EHR-Note-
QA (Kweon et al., 2024) is the first to include
questions requiring information across records in-
cluding both multiple-choice (MC) and open-ended
questions. There is limited work on the effect of
long-range dependencies within biomedical ma-
chine reading comprehension (MRC) (Vatsal and
Singh, 2024) and this tends to focus on shorter texts
(∼1.5-4K tokens) and biomedical articles and span-
based QA rather than complex EHRs. Work that
addresses MRC in other domains has utilised meth-
ods such as vector compression (Chevalier et al.,
2023), memory-based models (Xiao et al., 2024)
and context window chunking or sliding (Xiao et
al., 2023; Han et al., 2023) to process long context.
Retrieval Augmented Generation (RAG) (Lewis
et al., 2021) forms a cost efficient alternative that
mitigates risks around information loss or utiliz-
ing outdated or incomplete knowledge by selecting
relevant context on demand rather than trying to
process the entire context. What is preferable is
subject to debate, with each showing benefits on dif-
ferent tasks and settings (Li et al., 2024): RAG has
been reported to enhance LLM performance com-
pared to long-context prompt compression for QA
(Zhang et al., 2024) outperforming SOTA LLMs for
both 32K and 128K benchmarks (Xu et al., 2024;
Bai et al., 2024). By contrast, others showed that
RAG performance peaks at 32K context length forarXiv:2510.18691v1  [cs.CL]  21 Oct 2025

Figure 1: Overview of our Approach
large LLMs (Leng et al., 2024).
Here we investigate how LLMs handle diverse in-
put patient record combinations, using EHR-based
human-verified datasets in a variety of long context
and RAG settings. Fig. 1 shows a diagram of our
study setup. We make the following contributions:
•We are the first to assess how LLMs of differ-
ent underlying capabilities and sizes perform
on long-context medical QA given longitudinal
EHR patient notes. We cover three task formula-
tions: 1) MC , 2) Extractive and 3) Open-ended
generative QA. (§3.1,§3.2)
•We perform comprehensive experiments evalu-
ating the effect of different levels of granularity
in the input context while also testing for EHR
memorisation (§4.2).
•We examine the effect of RAG across task formu-
lations and context sizes, showing that given the
right setting RAG has superior performance than
LC across tasks (§5).
•We examine the distinct challenges of open-
ended QA when reasoning over a single docu-
ment versus multiple documents (§5).
•Our qualitative analyses shed light on some of the
challenges that LLMs currently face in reasoning
over long EHRs (§5.1).
2 Related Work
2.1 Leveraging Long Context
While the increasing long-context window capa-
bilities of some LLMs, i.e. GPT-4o (Hurst et al.,
2024) (128K), Claude 3 (Anthropic, 2023) (200K),
Gemini 1.5 (Team et al., 2024) (1 million), boosttheir long-context performance they still struggle
in general purpose open-form QA (beyond short
passage retrieval) (Zhang et al., 2024; Chen et al.,
2025). Despite positional embeddings techniques
like RoPE (Su et al., 2024), YaRN (Peng et al.,
2023) and Position Interpolation (Chen et al., 2023)
allowing context extrapolation, QA task perfor-
mance is low even for 32K context (Chen et al.,
2025). Furthermore, the performance gap between
open-source and longer-context close-source mod-
els (Li et al., 2024) poses questions around the po-
tential avenues for patient-centric long-dependency
data where privacy matters.
Benchmarking and assessment of long-context
performance of LLMs (Dong et al., 2023; Bai et al.,
2024; Liu et al., 2023; Zhang et al., 2024; Hsieh et
al., 2024), shows that apart from performance drops
with increasing token length LLMs are particularly
challenged by long-range dependency QA tasks
(Li et al., 2024) that potentially require reasoning.
Fan et al. (2024) show notable LLM performance
deterioration in medical QA tasks with increased
context length up to 200K tokens, with open-source
LLMs struggling to produce output given larger
contexts. Although long and longitudinal context
is prominent in real-world patient and medical data,
there is little work investigating LLM capabilities
in such settings.
2.2 Retrieval-augmented Generation (RAG)
RAG has been leveraged in a prompt-based plug-
in manner to address black-box long-context chal-
lenges (Yu et al., 2023). Current retrieval strategies
are primarily: chunk-based (Izacard et al., 2021),
index-based (Liu, 2022) and summarization-based
(Sarthi et al., 2024). LLM-based large-scale retriev-

Figure 2: RAG Pipeline
ers for chunk-based approaches like E5-Mistral-7b
(Wang et al., 2022) and BGE M3 (Chen et al., 2024)
give notable performance improvement. Further-
more, approaches like Self-RAG (Asai et al., 2023)
introduce on-demand retrieval followed by genera-
tion and self-reflection outperforming ChatGPT in
open-domain QA.
Domain-specific retrievers strongly benefit
biomedical retrieval. MedCPT (Jin et al., 2023),
a first of its kind contrastive biomedical dense se-
mantic retriever and re-ranker, achieved top per-
formance on 6 biomedical tasks surpassing LLM-
based retrievers. BMRETRIEVER (Xu et al.,
2024), a biomedical dense retriever trained both
contrastively and instruction fine-tuned on syn-
thetic data, shows better performance than Med-
CPT on most downstream tasks. The Med-RAG
(Zhao et al., 2025) method provides better diagnos-
tics over patient EHRs, being enhanced by medical
knowledge graph-elicited reasoning. Finally, the
Self-BioRAG (Jeong et al., 2024) is a framework
that retrieves relevant medical documents using
MedCPT and generates through self-reflection on
rationales using a domain-specific LLM. Yet Fan
et al. (2025) underscore the challenges in retrieval
for the medical domain: despite strong medical
retrieval capabilities across large corpora, models
struggle with specialized medical content such as
EHRs. Myers et al. (2025) further demonstrated
variability in retrieval across EHR datasets. There
is need for work that assesses retrieval performance
on patient-oriented QA over long-context.
3 Approach
3.1 Task Selection
To uncover the limitations of LLMs in medical
QA given complex patient-oriented clinical scenar-
ios, we focus oncontent-comprehensionsettingsespecially ones where patient context is not supple-
mentary but crucial to complete a task (e.g. obtain
diagnoses), thus emphasizinglong temporalpa-
tient context spanning across time and documents.
Aiming for a comprehensive approach we eval-
uate datasets coveringthree task formulations:
1) MCQA, 2) Extractive, and 3) Open-ended QA.
We further assess each dataset acrossfour settings
with varying note relevance.
3.2 Dataset Selection
Datasets were included in our study based on the
following criteria (datasets comparison in Table 9):
Going beyond Open-domain MCQA: Widely
used medical MC benchmarks i.e. MedQA (Jin
et al., 2021), MedMCQA (Pal et al., 2022), Pub-
MedQA (Jin et al., 2019), MMLU (Hendrycks
et al., 2021) evaluate option selection instead of
grounded synthesis and risk pretraining leakage
from exams and textbooks, which can inflate scores,
while importantly providing little signal towards
noisy clinical settings.
Long-context Documents: To address LLM capa-
bilities in handling long context that reflects real-
world settings we seek benchmarks that extend
well-beyond 8K. To avoid a simplified needle in
the haystack setting we aim for such content to be
relevant rather than distracting.
Expert QA Annotations: We selected expert
annotated QA pairs grounded on multiple EHR
documents to avoid limitations that hinder clini-
cal reliability such as non-expert curation (Fan et
al., 2024) or synthetic data (Adams et al., 2024).
Patient-centric Longitudinal Content: To study
performance on clinically-relevant QA with multi-
note reasoning, we prioritize EHR datasets derived
from MIMIC-III and MIMIC-IV (Johnson et al.,
2016, 2023). These corpora offer standardized de-

identified notes across document types (e.g., dis-
charge summaries, radiology reports) and enable
consistent retrieval granularity.
3.3 RAG Strategy
RAG strategies can be distinguished intosparse
anddense. Sparse retrievers such as BM25 (Robert-
son et al., 2009) and Splade (Formal et al., 2021)
use overlapping terms to match queries with snip-
pets, while dense retrievers encode them into em-
beddings and match them based on semantic sim-
ilarity. Recently, using LLM-based embeddings
such as E5-Mistral-7B-Instruct (Wang et al., 2023)
and Qwen3 Embedding (Zhang et al., 2025) for
dense retrieval has been standard practice.
Research on biomedical MCQA (Wang et al.,
2024) and biomedical document retrieval (Luo
et al., 2022) has shown that hybrid approaches
of sparse and dense retrieval outperform single
component counterparts. Further, re-ranking of
retrieved snippets also exhibits superior perfor-
mance in biomedical applications (Jin et al., 2023;
Wang et al., 2024; Sohn et al., 2025). Addition-
ally, non-hybrid experiments on different EHR sub-
sets of varying granularity (Fan et al., 2025) show
that there is no clear performance winner between
sparse and dense retrievers. Therefore, we use a
hybrid approach of a sparse and denser retriever
with a re-ranker.
Our hybrid approach is presented in Figure
2. Clinical notes are segmented into 512-token
chunks, which form our document collections D=
{d1, d2, . . . , d n}. The question respectively forms
the queryQ.
Forsparseretrieval we employ a lexical retriever
as a ranking function of each chunk djthat extends
TF-IDF by considering term frequency saturation
and document length1:
Ssparse (Q, d j) =nX
i=1IDF(q i)·fqi,dj(k1+1)
fqi,dj+k1
1−b+b·|dj|
avgdl
(1)
For thedenseretriever we use the same Encoder,
Eto encode the query and each of the chunks and
capture semantic similarity of pairs using the cosine
similarity:
Sdense(Q, d j) =E(Q)·E(d j)
∥E(Q)∥∥E(d j)∥(2)
Topkchunks are retrieved with each retriever
and then combined through Reciprocal Rank Fu-
1Here terms document and chunk are used interchangeably.sion (RRF) that takes into account the rank of each
document:
RRF(d j) =X
s∈S1
kRRF+ rank s(dj)(3)
, wherek RRF is a smoothing constant.
Finally, we employ Late Interaction Reranking, a
measure of how well every query token is semanti-
cally supported by at least one document token, to
produce the final top kmost contextually relevant
set.
SMaxSim (Q, d j) =|Q|X
i=1max
1≤r≤|d j|⟨qi,dj,r⟩(4)
Document Ordering: We leverage an order-
preserving RAG that sorts retrieved chunks tem-
porally rather than by retrieval score, due to its
superior long-context performance (Yu et al., 2024)
regardless of the number of chunks.
Note Inclusion: We evaluated two different strate-
gies: direct chunk inclusion, and parent note inclu-
sion (§4.3).
3.4 Evaluation Methodology
Evaluating medical QA systems has traditionally
relied on automatic metrics and multiple-choice
benchmarks (Jin et al., 2019, 2021; Pal et al.,
2022), which probe memorized knowledge but un-
derrepresent the complexity and factual rigor of
clinical reasoning, long-form answers, and EHR-
grounded context. To provide clinically mean-
ingful assessment, we adopt a multi-dimensional,
reference-based scheme that complements lexical
overlap with embedding-based semantic similarity
and domain-adapted Natural Language Inference
(NLI) for factual consistency and precision. In line
with emerging clinical evaluation practices (e.g.,
MEDIC (Kanithi et al., 2024)) and the rise of LLM-
as-a-judge, we employ a calibrated rubric to cap-
tureCorrectness,Completeness, andFaithfulness
at scale, and we prioritize widely available, repro-
ducible metrics while cross-checking signals for
robustness. Given that gold standard answers are
available, we employ reference-based evaluation:
METEOR(Banerjee and Lavie, 2005) is used to
capturesurface-levelandlexical similarity.
BERTScore(Zhang et al., 2020) computed with
Clinical BioBERT embeddings (Alsentzer et al.,
2019) assesses thesemantic similarity.

Dataset Context Data QA Pairs Patients Mean/Max Context Task Types Answer Location Reasoning Question Generation Answer Annotation
CliniQG4QAMIMIC-III1,287 36 4K / 7KExtractiveClinical Note× ××Experts Clinical experts (3)
RadQA3,509 80 6K / 14K Radiology Report Single-Note Human-generated Physicians (2)
EHR-DS-QAMIMIC-IV478 70 46K / 131K Open-ended Clinical Note/Dis. Summary× ××LLMs Physician-verified (1)
EHRNoteQA962 962 9K / 39K MC,Open-ended Dis. Summaries Multi-Note LLM Clinician-refined (3)
Table 1: EHR QA Dataset Statistics. Mean/Max Context corresponds to theInclude Allsetting.
NLIscores capture the logical relationship between
reference and candidate answers. We use a domain-
adapted NLI model (Deka et al., 2023) with refer-
ence answers as premise and candidate answers as
hypotheses. We then measureFactual Consistency
using the probability of non-contradiction (Song et
al., 2024) and measureFactual Precisionusing the
entailment probability.
LLM-as-a-judgewas employed with a 5-point
scale rubric on three aspects, using QWEN-2.5-
32B-INSTRUCTas the evaluator model2.Correct-
nesscaptures factual consistency with respect to
the gold answer, penalizing contradictions while
allowing compatible additional information.Com-
pletenesscaptures recall by assessing the predicted
answer’s coverage of information present in the
reference.Faithfulnesscaptures precision by as-
sessing whether the predicted answer only contains
information that is supported by or derivable from
the reference.
We reportAccuracyon the MC task.
4 Experiments
4.1 Datasets
Context Sources : We leverage MIMIC-III (John-
son et al., 2016) and MIMIC-IV (Johnson et al.,
2023) comprising de-identified EHRs covering hos-
pital admissions and ICU stays. They support
single- and multi-note settings and long contexts.
Our data includes: clinical notes, discharge sum-
maries and radiology reports per patient over time.
QA Datasets : We test on four EHR QA bench-
marks spanning different tasks given MIMIC-based
underlying context. We only use QA pairs curated
or validated by clinical experts to ensure medical
validity. We provide a brief dataset overview below,
with specifics summarized in Table 1.
CliniQG4QA(Yue et al., 2020): An extractive
span benchmark originally created for domain
adaptation purposes based on MIMIC-III clinical
2We selected Qwen for stability and agreement after
comparing against Selene-8B (Alexandru et al., 2025) and
Prometheus-8x7B v2.0 (Kim et al., 2024) in a small pilot (see
Appendix D.2).notes. While the 8,824-QA pair dev set is machine-
generated (MG), the 1,287 QA test pairs are expert-
annotated using the MG questions for reference.
Answers were either generated or verified by three
clinical experts. Here we only use the expert-
annotated test set.
RadQA(Soni et al., 2022): An extractive QA
dataset comprising 3,074 physician-crafted ques-
tions and 6,148 answer spans from the Findings
and Impressions sections of radiology reports in
MIMIC-III. Many questions correspond to multi-
ple sentences and are shown to require different
types of reasoning to answer. Annotation is carried
out by two human annotators with medical under-
standing. Here we use the train set and filter out
UnanswerableQA pairs resulting in 3,509 pairs.
EHR-DS-QA(Kotschenreuther, 2024): A large
synthetic QA corpus of 156,599 pairs generated
by two Llama-based (Touvron et al., 2023) LLMs
and guided by predefined prompt templates on
discharge summaries and notes from MIMIC-IV .
A subset of 506 pairs were physician-verified of
which we retain 478, marked as correct.
EHRNoteQA(Kweon et al., 2024): Built on
MIMIC-IV discharge summaries, it contains 962
QA pairs authored by GPT-4 and iteratively refined
by three clinicians. Questions span a diverse set of
topics, each corresponding to multiple discharge
summaries ( ∼2.3 per patient). The dataset supports
both open-ended and MC formats.
4.2 Context Formulation
Since most of the datasets are based on a single
note/report (CliniQG4QA, RadQA, EHR-DS-QA)
or up to three discharge summaries (EHRNoteQA),
we identify a large amount of longitudinal patient
supplementary content that remains unused. For
a more realistic clinical setting, we leverage our
context sources, MIMIC-III and MIMIC-IV , to aug-
ment few-note context with each patient’s longitudi-
nal notes. We then investigate how LLMs leverage
long-form context to answer patient-specific ques-
tions. We believe that this setting, (Include All),
presents a more realistic scenario where a model
would have no prior knowledge of the relevant doc-

Model Setting EHRNoteQA EHR-DS-QA RadQA CliniQG4QA
Open-ended MC Open-ended Extractive Extractive
LLM NLI F1 Acc. LLM NLI F1 LLM NLI F1 LLM NLI F1
HuatuoGPT-o1 7B
Exclude All7.85 23.36 68.8251.0826.65 33.8672.21 25.06 27.9 63.78 15.96 23.01 64.44
Qwen2.5 7B9.7318.18 68.41 51.46 24.79 33.0 72.35 39.08 29.24 70.73 27.02 26.3 74.06
Qwen2.5 32B 0.71 11.3 66.56 58.34 20.16 20.2372.673.86 2.88 10.35 0.11 0.19 2.16
QwQ 32B 2.68 12.23 66.9859.2424.16 23.8 71.70 1.51 6.44 15.29 0.93 5.62 14.77
HuatuoGPT-o1 7B
Exclude Related26.62 40.2672.57 58.35 28.49 33.9871.84 24.39 28.28 63.33 15.70 24.13 64.15
Qwen2.5 7B 25.24 28.64 72.72 59.86 29.928.17 73.35 38.86 29.07 70.72 27.01 25.95 74.09
Qwen2.5 32B 25.59 23.67 73.01 58.72 22.16 22.35 73.15 3.77 2.78 10.02 0.13 0.2 2.17
QwQ 32B27.4623.0174.35 60.1527.19 23.9873.511.14 7.11 17.48 0.99 4.84 15.75
HuatuoGPT-o1 7B
Include All72.60 45.45 78.21 78.94 59.76 63.6677.14 63.92 49.41 76.38 67.85 52.45 80.08
Qwen2.5 7B 70.49 38.11 79.95 78.8162.4961.00 79.14 62.56 50.13 76.65 68.89 54.13 81.27
Qwen2.5 32B 67.8855.33 81.50 90.9757.21 61.96 79.59 67.7450.1077.55 80.4959.22 83.8
QwQ 32B75.5247.14 81.02 89.25 57.68 60.54 78.9 67.46 53.2677.47 78.95 66.00 83.94
HuatuoGPT-o1 7B
Include Related70.73 39.30 79.71 76.87 63.46 60.65 77.20 64.37 49.19 76.48 70.47 53.19 79.83
Qwen2.5 7B 74.44 39.15 80.78 80.9 64.76 65.1480.28 63.01 50.25 76.71 69.06 54.87 81.29
Qwen2.5 32B 76.01 61.41 82.48 90.3359.12 64.86 80.7067.76 50.78 77.67 79.8159.21 83.55
QwQ 32B 82.0347.19 82.36 87.57 61.31 61.68 79.70 67.79 55.48 77.6979.74 65.86 84.35
Table 2: Results for Full Context in each setting across datasets.Boldis best and underlined the second best model
in each Setting and Metric. Red highlights the global best across all Settings per Metric.LLMcorresponds to LLM
Correctness,NLIto NLI Entailment andF1to BioBERT F1
ument and would therefore need to process the
entire patient note history3.
We explore four data inclusion scenarios de-
picted in Fig. 1, allowing comprehensive analy-
sis of model sensitivity to context size and noise,
ability to handle supplementary information and
memorization:
1.Exclude All: Exclusion of all clinical notes -
assessing model note memorization.
2.Exclude Relevant: Exclusion of relevant clini-
cal notes - assessing usefulness of supplemen-
tary material.
3.Include All: See above.
4.Include Related: Inclusion of only dataset
relevant clinical note types: discharge sum-
mary(ies), note or radiology report (as marked
in each underlying dataset) - assessing model’s
ability to leverage relevant context information
for patient-centric QA.
Context Segmentation: For theInclude Allsetting
we report context-length-based performance across
four bins: i)Short context: 0–8K, ii)Medium con-
text: 8–16K, iii)Large context: 16–32K, iv)Ex-
tended context: 32–128K tokens.
4.3 Models and Experimental Setup
LLMs: To explore the effect of LLM model size,
domain specialization and reasoning capabilities
in a controlled way we study four open Qwen2.5
models (Team and others, 2024), namely:
3MIMIC preprocessing is described in Appendix C.2.•Qwen2.5-7B-Instruct
•HuatuoGPT-o1-Qwen-7B(Chen et al., 2024):
medical LLM based on Qwen2.5-7B-Instruct
•Qwen2.5-32B-Instruct-128K
•QwQ:32B: reasoning model
Retrieval Settings: We implemented a hybrid re-
trieval pipeline designed to properly handle the
semantic, lexical, and query complexity of medical
QA. Our pipeline involves a combination of three
different retrievers (more details are in §3.3):
•Dense Retrievalbased on the leading open em-
bedding model Qwen3-Embedding-8B (Zhang
et al., 2025).
•Sparse Retrievalperformed using BM25 selected
after ablating two commonly used sparse retriev-
ers (Table 8 in Appendix).
•Late Interaction Rerankingwith Reason-
ModernColBERT, a ColBERT (Khattab and
Zaharia, 2020) model finetuned on the ReasonIR
dataset (Shao et al., 2025) leading to very high
performance on reasoning-intensive retrieval
benchmark (Su et al., 2024). Due to its per-
formance intensive nature, we use it as a final
reranker in our pipeline.
Experimental Setup: We defer inference specifics
to Appendix A and prompt formulation details in
Appendix C.3.
5 Results and Discussion
Full context performance for the different settings
for all datasets and models is shown in Table 2.

For RAG, Table 3 shows different evaluated RAG
settings across 3 selected models on the 2 MIMIC-
IV based datasets. We excluded QwQ:32B due to
time and budget constraints. Similarly, in Table 4
we only evaluate 7B sized models using a single
RAG strategy, due to the task nature and limited
context size of the MIMIC-III based datasets.
Model Size, Context Size and Performance
Consistent with general findings, larger models
consistently perform better across all tasks (Ta-
ble 2). Qwen2.5:32b and QwQ:32b generally out-
performed the 7b parameter models across most
metrics and tasks. The performance gap was par-
ticularly pronounced in tasks requiring reason-
ing or synthesizing of information from multiple
sources, namely EHRNoteQA and RadQA. How-
ever, smaller models performed better in settings
excluding relevant information or completely re-
moving the context. This suggests that they are
more prone to memorization.
The ‘Include Related’ setting outperforms ‘Include
All’ on most datasets and metrics except for the
MC task where no benefits from filtering are seen
for 3 out of the 4 models. Fig. 3 shows results for
Qwen2.5:32B-Instruct over different context sizes.
We observed similar trends between other models
and context size (see Appendix, Fig. 8). Worth
noting that for context range (8-16K) EHR-DS-QA
exhibits a dip in performance across all metrics and
models, attributable to noise (see §5.1).
Figure 3: Metric performance over context size on
‘Include All’ for Qwen2.5:32B-Instruct.
Reasoning and Fine-tuningThe impact of
reasoning-focused models and medical fine-tuning
is mixed.The QwQ:32b general reasoning model
showed improved results on Table 2 over other
models overall on our two reasoning datasets
(RadQA, EHRNoteQA). This suggests that spe-
cialized reasoning models can offer an advantage
in tasks that demand complex inference across mul-
tiple sources of information, but this benefit may beModel Setting EHRNoteQA EHR-DS-QA
Open-ended MC Open-ended
LLM NLI F1 Acc. LLM NLI F1
HuatuoGPT-o1 7BInclude All 70.68 39.52 77.26 75.36 50.25 59.01 75.30
RAG 5 59.94 61.3478.44 75.67 57.57 61.53 78.22
RAG 10 61.03 48.51 77.99 73.4 58.08 62.45 78.61
RAG 15 60.84 52.53 77.92 74.97 61.95 64.8477.68
RAG HIR 3 70.13 56.91 77.99 70.81 50.06 53.82 76.17
RAG HIR 570.9257.51 79.5272.38 48.46 55.21 76.66
RAG HIR 7 70.8 54.61 79.35 80.4655.99 57.12 77.35
Qwen2.5 7BInclude All 66.86 28.36 79.23 75.04 54.26 52.78 77.07
RAG 5 59.54 40.33 78.82 75.67 54.17 56.1179.82
RAG 10 64.03 45.02 79.05 73.4 57.11 62.7679.6
RAG 15 58.07 52.08 79.2 75.659.2560.32 79.65
RAG HIR 3 65.54 46.34 80.37 76.93 50.73 53.12 78.65
RAG HIR 572.68 54.6980.17 79.76 53.59 52.56 77.66
RAG HIR 7 67.55 46.4180.69 84.3152.1 52.39 78.26
Qwen2.5 32BInclude All 62.54 50.90 80.87 89.79 50.27 57.78 78.22
RAG 5 60.35 46.36 80.28 80.46 58.02 58.91 80.29
RAG 10 67.00 54.54 80.47 77.8760.5260.31 80.24
RAG 15 73.4553.94 80.84 81.02 55.68 61.39 80.71
RAG HIR 3 67.4857.17 82.4183.36 55.1363.3280.55
RAG HIR 5 68.86 54.87 81.59 90.74 51.2 60.73 80.12
RAG HIR 7 72.0 47.33 81.84 91.6852.82 57.27 80.21
Table 3: Open-ended and MC QA Results for Context
of 8K+ tokens including LC and RAG Methods.Bold
is best and underlined the second best performance per
Model and Metric. Red highlights the global best
across all models per Metric.
more apparent in larger model sizes where reason-
ing capabilities can be effectively leveraged with-
out compromising other essential skills.
By contrast,the medically fine-tuned reasoning
model, HuatuoGPT-o1-Qwen2.5:7b, did not show
any improvement over its standard instruction-
tuned base model Qwen2.5:7b-instruct, in line with
studies revealing that biomedical LLMs often lead
to reduced performance (Dorfner et al., 2024; Dada
et al., 2024).
RAG PerformanceTable 3 provides insights
into the effectiveness of different RAG strategies
per task and dataset.RAG demonstrated clear
performance improvement for the single-note gen-
erative task of EHR-DS-QA. In this scenario, the
chunk-in-context strategy consistently provided the
best results.For the more complex, multi-note rea-
soning task of EHRNoteQA, the hierarchical RAG
strategy was the bestwith few exceptions. The
results also show similar patterns on MC indicating
clear performance improvement for all models on
the RAG HIR 7 setting. These findings are consis-
tent across different context sizes (Fig. 4).
Worth noting that using RAG, the medi-
cally fine-tuned reasoning model HuatuoGPT-o1-
Qwen2.5:7b was able to score the best results on
multiple metrics across the 2 datasets. This shows
that RAG could benefit smaller models that strug-
gle in FC settings.

Model Setting RadQA CliniQG4QA
LLM NLI F1 LLM NLI F1
HuatuoGPT-o1 7BInclude All 35.60 48.8175.61 43.73 50.44 78.97
RAG 536.6129.5176.61 44.20 54.51 79.54
Qwen2.5 7BInclude All 36.1048.5975.92 44.55 53.26 80.04
RAG 5 37.0928.39 77.10 47.03 58.36 81.73
Table 4: Extractive QA Results for Context of 4K+
tokens.Boldis best performing per Model and Metric
and red highlights the global best per Metric.
While the extractive datasets have shorter con-
texts overall, Table 4 shows that the RAG setting
yields the best performance.
Figure 4: RAG performance across settings with
Qwen2.5:32B-Instruct (min-max normalized).
Metrics InsightsFig. 5 provides insights into
the relation between models, metrics, and datasets
while focusing on the open-ended generative tasks
for the ‘Include All’ and ‘Include Related’ settings
combined.Instruct models lead on single-note non-
reasoning tasks (EHR-DS-QA).Larger models lead
on EHRNoteQA: instruct models are better based
on semantic and NLI metrics, while reasoning ones
score better on LLM as a Judge metrics4.
FC vs RAG in QA TasksWe did a qualitative
analysis on the performance of FC and RAG on
the generative task across both datasets detailed in
Appendix G with examples.
We provide a summary of the main points below:
•FC performs better for questions requiring com-
prehensive understandingof the entire document
or when answers are located in structured sec-
tions, such as summaries, lists, or temporal se-
quences.
•RAG outperforms FC for questions that require
retrieving specific detailsor synthesizing infor-
4We include all-context and long-context correlations be-
tween metrics in Appendix E.1mation scattered across multiple parts of the doc-
ument.
•RAG tends to perform better overall in datasets
with complex or lengthy documents, likely due
to its ability to focus on relevant information and
reduce noise.
•There is mixed evidence suggesting FC might
perform better for tasks involving inferential rea-
soningor identifying the absence of information.
Figure 5: Model performance across metrics for
open-ended QA, normalized with min-max per metric.
5.1 Error Analysis
We analyze failure instances that complement the
quantitative trends and provide diagnostic guidance
for practical deployment. We summarize three re-
current themes and refer to Appendix F for full
cases and diagnostic tables.
Data noise.We observe inconsistencies and am-
biguous gold answers in some pairs across all
datasets (more profoundly in EHR-DS-QA) that
correlate with strange dips in performance at mid-
range context range windows. Fan et al. (2024)
also observed similar performance dips.
Numerical Values.Questions requiring numeric
values (e.g., vitals, dosages, dates) are sensitive to
formatting, rounding, or range expressions.
Metric Insights.We did a qualitative analysis to
understand the reasons behind metrics’ disagree-
ments across six metrics on the two generative
datasets (see details in Appendix E.2).
We found 233 disagreements in EHRNoteQA
and 47 in EHR-DS-QA. These are mainly due to:
1.Correct/Complete but Unfaithful (addition of
unsupported details). Most common in
EHRNoteQA (173/233; 74%) and present in
EHR-DS-QA (17/47; 36%).
2.Correct answer but low surface overlap
(lower Meteor/Bert) appears in both datasets
(EHRNoteQA: 20%; EHR-DS-QA: 26%).

3.Correct answer but low NLI entailment occurs
due to negation/clinical phrasing (EHRNoteQA:
17%; EHR-DS-QA: 21%).
4.EHR-DS-QA shows several short, risk/negation
items where NLI flags contradiction despite clin-
ically aligned answers.
This points to the need for thinking about more
reasoning appropriate metrics.
6 Conclusion
We studied long-context, patient-centric clinical
QA across EHR-grounded datasets, contrasting
Full-Context prompting with Retrieval-augmented
generation (RAG). Our analysis shows that care-
fully scoped context (e.g., Include DS) often out-
performs feeding all notes, and that hybrid RAG
pipelines with reranking are competitive, espe-
cially for specific factual queries and multi-note
synthesis. Larger models generally help, but per-
formance remains sensitive to context length, ev-
idence placement, and task formulation. Future
directions include: (i) scaling beyond single-note
or single-source assumptions toward richer multi-
note, multi-visit reasoning; (ii) advancing tempo-
ral and causal reasoning over longitudinal records,
with explicit timeline grounding; (iii) strengthen-
ing evaluation via clinically faithful, judge-robust
rubrics and cross-metric reliability analyses.
Limitations
Our work focuses on MIMIC-derived datasets,
which represents English-only data in a single U.S.
medical center. As such, performance may not
generalize to other languages, populations, and
healthcare systems. Furthermore, the de-identified
nature of the data limits us from examining the
potential cultural and other biases of LLMs over
long contexts. Importantly, although our work
presents an early step towards benchmarking long-
context LLM performance in longitudinal patient-
centric settings, our findings are intended solely
for research purposes. Results reflect model per-
formance on question answering benchmarks and
should not be interpreted as guarantees of clini-
cal safety, equitable performance, or readiness for
clinical deployment.
Beyond complexities of modeling clinical notes,
real-world records comprise heterogeneous data, in-
cluding data from other sources and in other modal-
ities. Our work poses limitations in the assessment
of long-context by focusing only on the textualmodality, something we aim to address in future
work.
Despite the in depth study of literature and our
initial ablations in selecting a competitive RAG
methodology, our work does not exhaustively ex-
amine the full potential of RAG under different
settings, i.e. different retrievers and chunk sizes, in
long-form medical QA. Finally, while we focus on
Qwen-based models due to their competitive per-
formance, we acknowledge that an evaluation of a
wider range of LLMs could offer a more complete
picture of the LLM landscape in long-form medical
QA. Our analysis has also pointed to the limitations
of current evaluation metrics, both in terms of as-
sessing reasoning as well as lack of transparency in
LLM as a judge metrics and how faithfulness can
be truly assessed currently when additional infor-
mation, not present in the benchmark, is added.
Ethics Statement
This work uses MIMIC-III, MIMIC-IV , and four
derivative datasets (EHRNoteQA, EHR-DS-QA,
RadQA, CliniQG4QA) accessed under the Phys-
ioNet Credentialed Health Data License 1.5.0. All
datasets contain de-identified patient records in ac-
cordance with HIPAA Safe Harbor standards. We
exclusively used open-weight LLMs to ensure no
patient data were transmitted to third-party propri-
etary systems.
References
Ben Abacha, Asma, and Dina Demner-Fushman. A
question-entailment approach to question answering.
BMC bioinformatics, 20(1), p.511, 2019.
OpenAI Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, Red Avila, Igor Babuschkin,
Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim-
ing Bao, Mo Bavarian, Jeff Belgum, Irwan Bello,
Jake Berdine, Gabriel Bernadett-Shapiro, Christo-
pher Berner, Lenny Bogdonoff, Oleg Boiko, Made-
laine Boyd, Anna-Luisa Brakman, Greg Brockman,
Tim Brooks, Miles Brundage, Kevin Button, Trevor
Cai, Rosie Campbell, Andrew Cann, Brittany Carey,
Chelsea Carlson, Rory Carmichael, Brooke Chan,
Che Chang, Fotis Chantzis, Derek Chen, Sully Chen,
Ruby Chen, Jason Chen, Mark Chen, Benjamin
Chess, Chester Cho, Casey Chu, Hyung Won Chung,
Dave Cummings, Jeremiah Currier, Yunxing Dai,
Cory Decareaux, Thomas Degry, Noah Deutsch,
Damien Deville, Arka Dhar, David Dohan, Steve
Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti,
Tyna Eloundou, David Farhi, Liam Fedus, Niko

Felix, Sim’on Posada Fishman, Juston Forte, Is-
abella Fulford, Leo Gao, Elie Georges, Christian
Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh,
Raphael Gontijo-Lopes, Jonathan Gordon, Morgan
Grafstein, Scott Gray, Ryan Greene, Joshua Gross,
Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse
Han, Jeff Harris, Yuchen He, Mike Heaton, Jo-
hannes Heidecke, Chris Hesse, Alan Hickey, Wade
Hickey, Peter Hoeschele, Brandon Houghton, Kenny
Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shan-
tanu Jain, Shawn Jain, Joanne Jang, Angela Jiang,
Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto,
Billie Jonn, Heewoo Jun, Tomer Kaftan, Lukasz
Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish
Shirish Keskar, Tabarak Khan, Logan Kilpatrick,
Jong Wook Kim, Christina Kim, Yongjik Kim, Hen-
drik Kirchner, Jamie Ryan Kiros, Matthew Knight,
Daniel Kokotajlo, Lukasz Kondraciuk, Andrew Kon-
drich, Aris Konstantinidis, Kyle Kosic, Gretchen
Krueger, Vishal Kuo, Michael Lampe, Ikai Lan,
Teddy Lee, Jan Leike, Jade Leung, Daniel Levy,
Chak Li, Rachel Lim, Molly Lin, Stephanie Lin, Ma-
teusz Litwin, Theresa Lopez, Ryan Lowe, Patricia
Lue, Anna Makanju, Kim Malfacini, Sam Manning,
Todor Markov, Yaniv Markovski, Bianca Martin,
Katie Mayer, Andrew Mayne, Bob McGrew, Scott
Mayer McKinney, Christine McLeavey, Paul McMil-
lan, Jake McNeil, David Medina, Aalok Mehta, Jacob
Menick, Luke Metz, An-drey Mishchenko, Pamela
Mishkin, Vinnie Monaco, Evan Morikawa, Daniel
P. Mossing, Tong Mu, Mira Murati, Oleg Murk,
David M’ely, Ashvin Nair, Reiichiro Nakano, Rajeev
Nayak, Arvind Neelakantan, Richard Ngo, Hyeon-
woo Noh, Ouyang Long, Cullen O’Keefe, Jakub W.
Pachocki, Alex Paino, Joe Palermo, Ashley Pantu-
liano, Giambattista Parascandolo, Joel Parish, Emy
Parparita, Alexandre Passos, Mikhail Pavlov, An-
drew Peng, Adam Perelman, Filipe de Avila Belbute
Peres, Michael Petrov, Henrique Pondé de Oliveira
Pinto, Michael Pokorny, Michelle Pokrass, Vitchyr
H. Pong, Tolly Powell, Alethea Power, Boris Power,
Elizabeth Proehl, Raul Puri, Alec Radford, Jack W.
Rae, Aditya Ramesh, Cameron Raymond, Francis
Real, Kendra Rimbach, Carl Ross, Bob Rotsted,
Henri Roussez, Nick Ryder, Mario D. Saltarelli, Ted
Sanders, Shibani Santurkar, Girish Sastry, Heather
Schmidt, David Schnurr, John Schulman, Daniel
Selsam, Kyla Sheppard, Toki Sherbakov, Jessica
Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor,
Eric Sigler, Maddie Simens, Jordan Sitkin, Kata-
rina Slama, Ian Sohl, Benjamin Sokolowsky, Yang
Song, Natalie Staudacher, Felipe Petroski Such, Na-
talie Summers, Ilya Sutskever, Jie Tang, Nikolas
A. Tezak, Madeleine Thompson, Phil Tillet, Amin
Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick
Turley, Jerry Tworek, Juan Felipe Cer’on Uribe, An-
drea Vallone, Arun Vijayvergiya, Chelsea V oss, Car-
roll L. Wainwright, Justin Jay Wang, Alvin Wang,
Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann,
Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian
Weng, Matt Wiethoff, Dave Willner, Clemens Win-
ter, Samuel Wolrich, Hannah Wong, Lauren Work-
man, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao,
Tao Xu, Sarah Yoo, Kevin Yu, Qim-ing Yuan, Woj-ciech Zaremba, Rowan Zellers, Chong Zhang, Mar-
vin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang
Zhuang, William Zhuk, and Barret Zoph. GPT-4
Technical Report. 2023.
Lisa C. Adams, Felix Busch, Tianyu Han, Jean-Baptiste
Excoffier, Matthieu Ortala, Alexander Loser, Hugo J.
W. L. Aerts, Jakob Nikolas Kather, Daniel Truhn, and
Keno Kyrill Bressem. LongHealth: A Question An-
swering Benchmark with Long Clinical Documents.
ArXiv, abs/2401.14490, 2024.
Andrei Alexandru, Antonia Calvi, Henry Broomfield,
Jackson Golden, Kyle Dai, Mathias Leys, Maurice
Burger, Max Bartolo, Roman Engeler, Sashank Pisu-
pati, and others. Atla selene mini: A general purpose
evaluation model. arXiv preprint arXiv, 2025.
Emily Alsentzer, John Murphy, William Boag, Wei-
Hung Weng, Di Jindi, Tristan Naumann, and
Matthew McDermott. Publicly Available Clinical
BERT Embeddings. In Proceedings of the 2nd Clini-
cal Natural Language Processing Workshop, pp. 72–
78, 2019.
Anthropic. Model card and evaluations for claude mod-
els.. com/production/images/Model-Card-Claude-2.
pdf., 2023.
Anna Arias-Duart, Pablo Agustin Martin-Torres, Daniel
Hinjos, Pablo Bernabeu-Perez, Lucia Urcelay Ganza-
bal, Marta Gonzalez Mallo, Ashwin Kumar Gurura-
jan, Enrique Lopez-Cuena, Sergio Alvarez-Napagao,
and Dario Garcia-Gasulla. Automatic Evaluation of
Healthcare {LLM}s Beyond Question-Answering. In
Proceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies
(V olume 2: Short Papers), pp. 108–130, 2025.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil,
and Hannaneh Hajishirzi. Self-RAG: Learning to
Retrieve, Generate, and Critique through Self-
Reflection. ArXiv, abs/2310.11511, 2023.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, and others. LongBench:
A Bilingual, Multitask Benchmark for Long Context
Understanding. In Proceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (V olume 1: Long Papers) (pp. 3119-3137),
2024.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Long-
Bench v2: Towards Deeper Understanding and Rea-
soning on Realistic Long-context Multitasks. ArXiv,
abs/2412.15204, 2024.
Satanjeev Banerjee and Alon Lavie. METEOR: An au-
tomatic metric for MT evaluation with improved cor-
relation with human judgments. In Proceedings of
the acl workshop on intrinsic and extrinsic evaluation

measures for machine translation and/or summariza-
tion, pp. 65–72, 2005. Association for Computational
Linguistics.
Amanda Bertsch, Maor Ivgi, Emily Xiao, Uri Alon,
Jonathan Berant, Matthew R Gormley, and Graham
Neubig. In-context learning with long-context mod-
els: An in-depth exploration. arXiv preprint arXiv,
2024.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George
van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, Aurelia
Guy, Jacob Menick, Roman Ring, T. W. Hennigan,
Saffron Huang, Lorenzo Maggiore, Chris Jones, Al-
bin Cassirer, Andy Brock, Michela Paganini, Geof-
frey Irving, Oriol Vinyals, Simon Osindero, Karen
Simonyan, Jack W. Rae, Erich Elsen, and L. Sifre. Im-
proving language models by retrieving from trillions
of tokens. In International Conference on Machine
Learning, 2021.
Peter G Brodeur, Thomas A Buckley, Zahir Kanjee,
Ethan Goh, Evelyn Bin Ling, Priyank Jain, Stephanie
Cabral, Raja-Elie Abdulnour, Adrian Haimovich, Ja-
son A Freed, and others. Superhuman performance
of a large language model on the reasoning tasks of a
physician. arXiv preprint arXiv, 2024.
Shouyuan Chen, Sherman Wong, Liangjian Chen, and
Yuandong Tian. Extending context window of large
language models via positional interpolation. arXiv
preprint arXiv, 2023.
Zeming Chen, Alejandro Hernández Cano, Angelika
Romanou, Antoine Bonnet, Kyle Matoba, Francesco
Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf,
Amirkeivan Mohtashami, and others. Meditron-70b:
Scaling medical pretraining for large language mod-
els. arXiv preprint arXiv, 2023.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo,
Defu Lian, and Zheng Liu. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
arXiv preprint arXiv, 2024.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. BGE M3-Embedding: Multi-
Lingual, Multi-Functionality, Multi-Granularity Text
Embeddings Through Self-Knowledge Distillation.
In Annual Meeting of the Association for Computa-
tional Linguistics, 2024.
Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wan-
long Liu, Rongsheng Wang, Jianye Hou, and Benyou
Wang. Huatuogpt-o1, towards medical complex rea-
soning with llms. arXiv preprint arXiv, 2024.
Hanjie Chen, Zhouxiang Fang, Yash Singla, and Mark
Dredze. Benchmarking large language models on
answering and explaining challenging medical ques-
tions. In Proceedings of the 2025 Conference of the
Nations of the Americas Chapter of the Association
for Computational Linguistics: Human LanguageTechnologies (V olume 1: Long Papers), pp. 3563–
3599, 2025.
Pei Chen, Hongye Jin, Cheng-Che Lee, Rulin Shao,
Jingfeng Yang, Mingyu Zhao, Zhaoyu Zhang, Qin
Lu, Kaiwen Men, Ning Xie, and others. LongLeader:
A Comprehensive Leaderboard for Large Language
Models in Long-context Scenarios. In Proceedings of
the 2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (V olume 1:
Long Papers), pp. 8734–8750, 2025.
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and
Danqi Chen. Adapting Language Models to Com-
press Contexts. In Proceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, pp. 3829–3846, 2023.
Arman Cohan, Sergey Feldman, Iz Beltagy, Doug
Downey, and Daniel S Weld. Specter: Document-
level representation learning using citation-informed
transformers. arXiv preprint arXiv, 2020.
Amin Dada, Marie Bauer, Amanda Butler Contreras, Os-
man Alperen Kora¸ s, Constantin Marc Seibold, Kaleb
E Smith and Jens Kleesiek. Does biomedical training
lead to better medical performance?. arXiv preprint
arXiv:2404.04067., 2024.
Pritam Deka, Anna Jurek-Loughrey, and Deepak P. Mul-
tiple Evidence Combination for Fact-Checking of
Health-Related Information. In Proceedings of the
22nd Workshop on Biomedical Natural Language
Processing and BioNLP Shared Tasks, pp. 237–247,
2023.
Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Ger-
stein and Arman Cohan Benchmark Probing: Inves-
tigating Data Leakage in Large Language Models.
NeurIPS BUGS Poster, 2024.
Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao,
and Ji-Rong Wen. BAMBOO: A Comprehensive
Benchmark for Evaluating Long Text Modeling Ca-
pacities of Large Language Models. In Proceedings
of the 2024 Joint International Conference on Com-
putational Linguistics, Language Resources and Eval-
uation (pp. 2086-2099), 2024.
Felix J. Dorfner, Amin Dada, Felix Busch, Marcus R.
Makowski, Tianyu Han, Daniel Truhn, Jens Kleesiek,
Madhumita Sushil, Jacqueline Lammert, Lisa C.
Adams and Keno K. Bressem. Biomedical large lan-
guages models seem not to be superior to general-
ist models on unseen medical data. arXiv preprint
arXiv:2408.13833., 2024.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, Anirudh Goyal, Anthony S. Hartshorn, Aobo
Yang, Archi Mitra, Archie Sravankumar, Artem Ko-
renev, Arthur Hinsvark, Arun Rao, Aston Zhang,
Aur’elien Rodriguez, Austen Gregerson, Ava Spataru,

Baptiste Rozier, Bethany Biron, Binh Tang, Bob-
bie Chern, Charlotte Caucheteux, Chaya Nayak,
Chloe Bi, Chris Marra, Chris McConnell, Christian
Keller, Christophe Touret, Chunyang Wu, Corinne
Wong, Cris-tian Cantón Ferrer, Cyrus Nikolaidis,
Damien Allonsius, Daniel Song, Danielle Pintz,
Danny Livshits, David Esiobu, Dhruv Choudhary,
Dhruv Mahajan, Diego Garcia-Olano, Diego Perino,
Dieuwke Hupkes, Egor Lakomkin, Ehab A. Al-
Badawy, Elina Lobanova, Emily Dinan, Eric Michael
Smith, Filip Radenovic, Frank Zhang, Gabriele
Synnaeve, Gabrielle Lee, Georgia Lewis Ander-
son, Graeme Nail, Gréire Mialon, Guanglong Pang,
Guillem Cucurell, Hailey Nguyen, Hannah Kore-
vaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol
Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra,
Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Gef-
fert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet
Shah, Jelmer van der Linde, Jennifer Billock, Jenny
Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu
Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna
Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca,
Joshua Johnstun, Joshua Saxe, Ju-Qing Jia, Kalyan
Vasuden Alwala, K. Upasani, Kate Plawiak, Ke-
qian Li, Ken-591 neth Heafield, Kevin R. Stone,
Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuen-
ley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Lau-
rens van der Maaten, Lawrence Chen, Liang Tan,
Liz Jenkins, Louis Martin, Lovish Madaan, Lubo
Malo, Lukas Blecher, Lukas Landzaat, Luke de
Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Man-
nat Singh, Manohar Paluri, Marcin Kardas, Mathew
Oldham, Mathieu Rita, Maya Pavlova, Melissa Hall
Melanie Kambadur, Mike Lewis, Min Si, Mitesh
Kumar Singh, Mona Hassan, Naman Goyal, Nar-
jes Torabi, Niko-lay Bashlykov, Nikolay Bogoy-
chev, Niladri S. Chatterji, Olivier Duchenne, Onur
cCelebi, Patrick Alrassy, Pengchuan Zhang, Peng-
wei Li, Petar Vasi ´c, Peter Weng, Prajjwal Bhargava,
Pratik Dubal, Praveen Krishnan, Punit Singh Koura,
Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srini-
vasan, Raj Ganapathy, Ramon Calderer, Ricardo Sil-
veira Cabral, Robert Stojnic, Roberta Raileanu, Ro-
hit Girdhar, Rohit Patel, Ro-main Sauvestre, Ron-
nie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan
Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sa-
hana Chennabasappa, Sanjay Singh, Sean Bell, Seo-
hyun Sonia Kim, Sergey Edunov, Shaoliang Nie,
Sharan Narang, Sharath Chandra Raparthy, Sheng
Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Si-
mon Vandenhende, Soumya Batra, Spencer Whitman,
Sten Sootla, Stephane Collot, Suchin Gururangan,
Sydney Borodinsky, Tamar Herman, Tara Fowler,
Tarek Sheasha, Thomas Georgiou, Thomas Scialom,
Tobias Speckbacher, Todor Mihaylov, Tong Xiao,
Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vig-
nesh Ramanathan, Viktor Kerkez, Vincent Gonguet,
Vir-ginie Do, Vish V ogeti, Vladan Petrovic, Weiwei
Chu, Wenhan Xiong, Wenyin Fu, Whit-ney Meers,
Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen
Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle
Goldschlag, Yashesh Gaur, Yasmine Babaei, Yiqian
Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yun-
ing Mao, Zacharie Delpierre Coudert, Zhengxu Yan,Zhengxing Chen, Zoe Papakipos, Aaditya K. Singh,
Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam
Shajnfeld, Adi Gangidi, Adolfo Victoria, Ahuva
Goldstand, Ajay Menon, Ajay Sharma, Alex Boesen-
berg, Alex Vaughan, Alexei Baevski, Allie Feinstein,
Amanda Kallet, Amit Sangani, Anam Yunus, Andrei
Lupu, Andres Alvarado, Andrew Caples, Andrew
Gu, Andrew Ho, Andrew Poulton, Andrew Ryan,
Ankit Ramchandani, Annie Franco, Aparajita Saraf,
Arkabandhu Chowdhury, Ashley Gabriel, Ashwin
Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau
James, Ben Maurer, Benjamin Leonhardi, Po-Yao
(Bernie) Huang, Beth Loyd, Beto de Paola, Bhargavi
Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Han-
cock, Bram Wasti, Brandon Spence, Brani Stojkovic,
Brian Gamido, Britt Montalvo, Carl Parker, Carly
Burton, Catalina Mejia, Changhan Wang, Changkyu
Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu,
Chris Cai, Chris Tindal, Christoph Feichtenhofer,
Damon Civin, Dana Beaty, Daniel Kreymer, Shang-
Wen Li, Danny Wyatt, David Adkins, David Xu,
Davide Testuggine, Delia David, Devi Parikh, Di-
ana Liskovich, Didem Foss, Dingkang Wu, Duc
Le, Dustin Holland, Edward Dowling, Eissa Jamil,
Elaine Montgomery, Eleonora Presani, Emily Hahn,
Emily Wood, Erik Brinkman, Esteban Arcaute, Evan
Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng
Tian, Firat Ozgenel, Francesco Caggioni, Francisco
Guzm’an, Frank J. Kanayet, Frank Seide, Gabriela
Medina Florez, Gabriella Schwarz, Gada Badeer,
Georgia Swee, Gil Halpern, Govind Thattai, Grant
Herman, Grigory G. Sizov, Guangyi Zhang, Guna
Lakshminarayanan, Hamid Shojanazeri, Han Zou,
Hannah Wang, Han Zha, Haroun Habeeb, Harri-
son Rudolph, Helen Suk, Henry Aspegren, Hunter
Goldman, Igor Molybog, Igor Tufanov, Irina-Elena
Veliche, Itai Gat, Jake Weissman, James Geboski,
James Kohli, Japhet Asher, Jean-Baptiste Gaya,
Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen,
Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong,
Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill,
Jon Shepard, Jonathan McPhie, Jonathan Torres,
Josh Ginsburg, Junjie Wang, Kaixing(Kai) Wu,
U KamHou, Karan Saxena, Karthik Prasad, Kar-
tikay Khandelwal, Katayoun Zand, Kathy Matosich,
Kaushik Veeraraghavan, Kelly Michelena, Keqian
Li, Kun Huang, Kunal Chawla, Kushal Lakhotia,
Kyle Huang, Lailin Chen, Lakshya Garg, A Laven-
der, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng
Guo, Licheng Yu, Liron Moshkovich, Luca Wehrst-
edt, Madian Khabsa, Manav Avalani, Manish Bhatt,
Maria Tsimpoukelli, Martynas Mankus, Matan Has-
son, Matthew Lennie, Matthias Reso, Maxim Gro-
shev, Maxim Naumov, Maya Lathi, Meghan Keneally,
Michael L. Seltzer, Michal Valko, Michelle Restrepo,
Mihir Patel, Mik Vyatskov, Mikayel Samvelyan,
Mike Clark, Mike Macey, Mike Wang, Miquel Jubert
Hermoso, Mo Metanat, Mohammad Rastegari, Mu-
nish Bansal, Nandhini Santhanam, Natascha Parks,
Natasha White, Navyata Bawa, Nayan Singhal, Nick
Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev,
Ning Dong, Ning Zhang, Norman Cheng, Oleg
Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem
Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pa-

van Balaji, Pe-dro Rittner, Philip Bontrager, Pierre
Roux, Piotr Dollár, Polina Zvyagina, Prashant Ratan-
chandani, Pritish Yuvraj, Qian Liang, Rachad Alao,
Rachel Rodriguez, Rafi Ayub, Raghotham Murthy,
Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah
Hogan, Robin Battey, Rocky Wang, Rohan Mah-
eswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu,
Samyak Datta, Sara Chugh, Sara Hunt, Sargun
Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma,
Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lind-
say, Sheng Feng, Shenghao Lin, Shengxin Cindy
Zha, Shiva Shankar, Shuqiang Zhang, Sinong Wang,
Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala,
Stephanie Max, Stephen Chen, Steve Kehoe, Steve
Satterfield, Sudarshan Govindaprasad, Sumit Gupta,
Sung-Bae Cho, Sunny Virk, Suraj Subramanian, Sy
Choudhary, Sydney Goldman, Tal Remez, Tamar
Glaser, Tamara Best, Thilo Kohler, Thomas Robin-
son, Tianhe Li, Tianjun Zhang, Tim Matthews, Timo-
thy Chou, Tzook Shaked, Varun V ontimitta, Victoria
Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish
Kumar, Vishal Mangla, Vlad Ionescu, Vlad Andrei
Poenaru, Vlad T. Mihailescu, Vladimir Ivanov, Wei
Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz,
Will Constable, Xia Tang, Xiaofang Wang, Xiao-
jian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo
Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li,
Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam,
Yu Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach
Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen,
Zhenyu Yang, Zhiwei Zhao. The Llama 3 Herd of
Models. ArXiv, abs/2407.21783, 2024.
Yongqi Fan, Hongli Sun, Kui Xue, Xiaofan Zhang,
Shaoting Zhang, and Tong Ruan. MedOdyssey: A
Medical Domain Benchmark for Long Context Evalu-
ation Up to 200K Tokens. arXiv preprint arXiv, 2024.
Yongqi Fan, Nan Wang, Kui Xue, Jingping Liu, and
Tong Ruan. MedEureka: A Medical Domain Bench-
mark for Multi-Granularity and Multi-Data-Type
Embedding-Based Retrieval. In Findings of the Asso-
ciation for Computational Linguistics: NAACL 2025,
pp. 2825–2851, 2025.
S. Fleming, Alejandro Lozano, William J. Haberkorn,
Jenelle A. Jindal, Eduardo Pontes Reis, Rahul Thapa,
Louis Blankemeier, Julian Z. Genkins, Ethan H.
Steinberg, Ashwin Nayak, Birju S. Patel, Chia-Chun
Chiang, Alison Callahan, Zepeng Huo, Sergios Ga-
tidis, Scott J. Adams, Oluseyi Fayanju, Shreya Shah,
Thomas Savage, Ethan Goh, Akshay S. Chaudhari,
Nima Aghaeepour, Christopher D. Sharp, Michael
A. Pfeffer, Percy Liang, Jonathan H. Chen, Keith
E. Morse, Emma Brunskill, Jason Alan Fries, and
Nigam H. Shah. MedAlign: A Clinician-Generated
Dataset for Instruction Following with Electronic
Medical Records. In AAAI Conference on Artificial
Intelligence, 2023.
Thibault Formal, Benjamin Piwowarski, and Stéphane
Clinchant. SPLADE: Sparse lexical and expansion
model for first stage ranking. In Proceedings of the
44th International ACM SIGIR Conference on Re-search and Development in Information Retrieval, pp.
2288–2292, 2021.
Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng
Ji, and Sinong Wang. Lm-infinite: Simple on-the-
fly length generalization for large language models.
2023.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
Measuring Massive Multitask Language Understand-
ing. Proceedings of the International Conference on
Learning Representations (ICLR), 2021.
Pedram Hosseini, Jessica M. Sin, Bing Ren, Bryceton
G. Thomas, Elnaz Nouri, Ali Farahanchi, and Saeed
Hassanpour. A Benchmark for Long-Form Medical
Question Answering. ArXiv, abs/2411.09834, 2024.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shan-
tanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang,
and Boris Ginsburg. RULER: What’s the Real Con-
text Size of Your Long-Context Language Models?.
arXiv preprint arXiv, 2024.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and oth-
ers. Gpt-4o system card. arXiv preprint arXiv, 2024.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. Unsupervised Dense Information
Retrieval with Contrastive Learning. Trans. Mach.
Learn. Res., 2022, 2021.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane A. Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave.
Few-shot Learning with Retrieval Augmented Lan-
guage Models. J. Mach. Learn. Res., 24, 251:1–
251:43, 2022.
Minbyul Jeong, Jiwoong Sohn, Mujeen Sung,
and Jaewoo Kang. Improving medical reasoning
through retrieval and self-reflection with retrieval-
augmented large language models. Bioinformatics,
40(Supplement-1), i119–i129, 2024.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. LLMLingua: Compressing
Prompts for Accelerated Inference of Large Lan-
guage Models. In Proceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, pp. 13358–13376, 2023.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
W Cohen, and Xinghua Lu. Pubmedqa: A dataset
for biomedical research question answering. arXiv
preprint arXiv, 2019.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
W. Cohen, and Xinghua Lu. PubMedQA: A Dataset
for Biomedical Research Question Answering. arXiv
preprint arXiv:1909.06146, 2019.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. What disease does
this patient have? a large-scale open domain ques-
tion answering dataset from medical exams. Applied
Sciences, 11(14), 6421, 2021.
Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau,
Lana Yeganova, W John Wilbur, and Zhiyong Lu.
Medcpt: Contrastive pre-trained transformers with
large-scale pubmed search logs for zero-shot biomed-
ical information retrieval. Bioinformatics, 39(11),
btad651, 2023.
Alistair E.W. Johnson, Tom J. Pollard, Lu Shen, Li-wei
H. Lehman, Melania Feng, Marzyeh Ghassemi, Ben-
jamin Moody, Peter Szolovits, Leo Anthony Celi, and
Roger G. Mark. MIMIC-III, a freely accessible criti-
cal care database. Scientific Data, 3, 160035, 2016.
Alistair Johnson, Lucia Bulgarelli, Tom Pollard, Steve
Horng, Leo A. Celi, and Roger Mark. MIMIC-IV-
Note: Deidentified free-text clinical notes (version
2.2). PhysioNet, 2023.
Praveen K Kanithi, Clément Christophe, Marco AF Pi-
mentel, Tathagata Raha, Nada Saadi, Hamza Javed,
Svetlana Maslenkova, Nasir Hayat, Ronnie Rajan,
and Shadab Khan. Medic: Towards a comprehensive
framework for evaluating llms in clinical applications.
arXiv preprint arXiv, 2024.
Omar Khattab and Matei Zaharia. Colbert: Efficient
and effective passage search via contextualized late
interaction over bert. In Proceedings of the 43rd In-
ternational ACM SIGIR conference on research and
development in Information Retrieval, pp. 39–48,
2020.
Seungone Kim, Juyoung Suk, Shayne Longpre, Bill
Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neu-
big, Moontae Lee, Kyungjae Lee, and Minjoon Seo.
Prometheus 2: An open source language model spe-
cialized in evaluating other language models. arXiv
preprint arXiv, 2024.
Yubin Kim, Hyewon Jeong, Shen Chen, Shuyue Stella
Li, Mingyu Lu, Kumail Alhamoud, Jimin Mun,
Cristina Grau, Minseok Jung, Rodrigo R Gameiro,
and others. Medical Hallucination in Foundation
Models and Their Impact on Healthcare. medRxiv,
2025–02, 2025.
Konstantin Kotschenreuther. EHR-DS-QA: A Synthetic
QA Dataset Derived from Medical Discharge Sum-
maries for Enhanced Medical Information Retrieval
Systems. PhysioNet, 2024.
Anastasia Krithara, Anastasios Nentidis, Konstantinos
Bougiatiotis, and Georgios Paliouras. BioASQ-QA:
A manually curated corpus for Biomedical Question
Answering. Scientific Data, 10(1), 170, 2023.
Ishita Kumar, Snigdha Viswanathan, Sushrita Yerra,
Alireza Salemi, Ryan A. Rossi, Franck Dernoncourt,
Hanieh Deilamsalehy, Xiang Chen, Ruiyi Zhang,Shubham Agarwal, Nedim Lipka, and Hamed Za-
mani. LongLaMP: A Benchmark for Personalized
Long-form Text Generation. ArXiv, abs/2407.11016,
2024.
Sunjun Kweon, Jiyoun Kim, Heeyoung Kwak,
Dongchul Cha, Hangyul Yoon, Kwanghyun Kim,
Jeewon Yang, Seunghyun Won, and Edward Choi.
EHRNoteQA: An LLM Benchmark for Real-World
Clinical Practice Using Discharge Summaries. In
Neural Information Processing Systems, 2024.
Eric Lehman, Vladislav Lialin, Katelyn Y . Legaspi,
Anne Janelle R. Sy, Patricia Therese S. Pile, Nicole
Rose I. Alberto, Richard Raymund R. Ragasa,
Corinna Victoria M. Puyat, Isabelle Rose I. Alberto,
Pia Gabrielle I. Alfonso, Marianne Taliño, Dana
Moukheiber, Byron C. Wallace, Anna Rumshisky,
Jenifer J. Liang, Preethi Raghavan, Leo Anthony
Celi and Peter Szolovits. Learning to Ask Like a
Physician. In Proceedings of the 4th Clinical Natural
Language Processing Workshop, Seattle, pp. 74–86,
2022. Association for Computational Linguistics.
Quinn Leng, Jacob Portes, Sam Havens, Matei Zaharia,
and Michael Carbin. Long context rag performance
of large language models. arXiv preprint arXiv, 2024.
Hui Yi Leong, Yifan Gao, and Shuai Ji. A gen ai frame-
work for medical note generation. In 2024 6th in-
ternational conference on artificial intelligence and
computer applications (ICAICA), pp. 423–429, 2024.
Mosh Levy, Alon Jacoby, and Yoav Goldberg. Same
task, more tokens: the impact of input length on
the reasoning performance of large language models.
arXiv preprint arXiv, 2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. 2021.
Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan
Zhang. LooGLE: Can Long-Context Language Mod-
els Understand Long Contexts?. In Proceedings of
the 62nd Annual Meeting of the Association for Com-
putational Linguistics (V olume 1: Long Papers), pp.
16304–16333, 2024.
Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and
Wenhu Chen. Long-context llms struggle with long
in-context learning. arXiv preprint arXiv, 2024.
Xinze Li, Yixin Cao, Yubo Ma, and Aixin Sun. Long
Context vs. RAG for LLMs: An Evaluation and Re-
visits. 2024.
Valentin Liévin, Christoffer Egeberg Hother, Andreas
Geert Motzfeldt, and Ole Winther. Can large lan-
guage models reason about medical questions?. 2023.
Jerry Liu. LlamaIndex. CoRR, 2022.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. Lost in the middle: How language models use
long contexts. arXiv preprint arXiv, 2023.
Man Luo, Arindam Mitra, Tejas Gokhale, and Chitta
Baral. Improving biomedical information retrieval
with neural retrievers. In proceedings of the AAAI
conference on artificial intelligence, pp. 11038–
11046, 2022.
Zizhan Ma, Wenxuan Wang, Guo Yu, Yiu-Fai Cheung,
Meidan Ding, Jie Liu, Wenting Chen, and Linlin
Shen. Beyond the Leaderboard: Rethinking Medi-
cal Benchmarks for Large Language Models. arXiv
preprint arXiv, 2025.
Nikita Mehandru, Brenda Y Miao, Eduardo Rodriguez
Almaraz, Madhumita Sushil, Atul J Butte, and
Ahmed Alaa. Evaluating large language models as
agents in the clinic. NPJ digital medicine, 7(1), 84,
2024.
Ali Modarressi, Hanieh Deilamsalehy, Franck Dernon-
court, Trung Bui, Ryan A Rossi, Seunghyun Yoon,
and Hinrich Schütze. Nolima: Long-context evalu-
ation beyond literal matching. arXiv preprint arXiv,
2025.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne,
and Nils Reimers. MTEB: Massive Text Embedding
Benchmark. arXiv preprint arXiv, 2022.
Skatje Myers, Timothy A Miller, Yanjun Gao, Matthew
M Churpek, Anoop Mayampurath, Dmitriy Dligach,
and Majid Afshar. Lessons learned on information
retrieval in electronic health records: a comparison
of embedding models and pooling strategies. Journal
of the American Medical Informatics Association,
32(2), 357–364, 2025.
Saeel Sandeep Nachane, Ojas Gramopadhye, Prateek
Chanda, Ganesh Ramakrishnan, Kshitij Sharad Jad-
hav, Yatin Nandwani, Dinesh Raghu, and Sachindra
Joshi. Few shot chain-of-thought driven reasoning
to prompt LLMs for open ended medical question
answering. arXiv e-prints, arXiv–2403, 2024.
Anastasios Nentidis, Georgios Katsimpras, Anastasia
Krithara, Martin Krallinger, Miguel Rodríguez-
Ortega, Eduard Rodriguez-López, Natalia
Loukachevitch, Andrey Sakhovskiy, Elena Tu-
tubalina, Dimitris Dimitriadis, and others. Overview
of BioASQ 2025: The thirteenth BioASQ challenge
on large-scale biomedical semantic indexing and
question answering. In International Conference of
the Cross-Language Evaluation Forum for European
Languages, pp. 173–198, 2025.
Robert Osazuwa Ness, Katie Matton, Hayden S. Helm,
Sheng Zhang, Junaid Bajwa, Carey E. Priebe, and
Eric Horvitz. MedFuzz: Exploring the Robustness
of Large Language Models in Medical Question An-
swering. ArXiv, abs/2406.06573, 2024.Harsha Nori, Nicholas King, Scott Mayer McKinney,
Dean Carignan, and Eric Horvitz. Capabilities of
gpt-4 on medical challenge problems. arXiv preprint
arXiv, 2023.
Ankit Pal, Logesh Kumar Umapathi, and Malaikan-
nan Sankarasubbu. Medmcqa: A large-scale multi-
subject multi-choice dataset for medical domain ques-
tion answering. In Conference on health, inference,
and learning, pp. 248–260, 2022.
Ankit Pal and Malaikannan Sankarasubbu. Gemini Goes
to Med School: Exploring the Capabilities of Mul-
timodal Large Language Models on Medical Chal-
lenge Problems & Hallucinations. In Clinical Natural
Language Processing Workshop, 2024.
Anusri Pampari, Preethi Raghavan, Jennifer J. Liang,
and Jian Peng. emrQA: A Large Corpus for Question
Answering on Electronic Medical Records. In Con-
ference on Empirical Methods in Natural Language
Processing, 2018.
Dimitris Pappas, Petros Stavropoulos, Ion Androut-
sopoulos, and Ryan McDonald. BioMRC: A dataset
for biomedical machine reading comprehension. In
Proceedings of the 19th SIGBioMed workshop on
biomedical language processing, pp. 140–149, 2020.
Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico
Shippole. Yarn: Efficient context window extension
of large language models. arXiv preprint arXiv, 2023.
Stephen Robertson, Hugo Zaragoza, and others. The
probabilistic relevance framework: BM25 and be-
yond. Foundations and Trends in Information Re-
trieval, 3(4), 333–389, 2009.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
RAPTOR: Recursive Abstractive Processing for Tree-
Organized Retrieval. ArXiv, abs/2401.18059, 2024.
Burcu Sayin, Pasquale Minervini, Jacopo Staiano, and
Andrea Passerini. Can LLMs Correct Physicians,
Yet? Investigating Effective Interaction Methods in
the Medical Domain. In Proceedings of the 6th Clini-
cal Natural Language Processing Workshop, pp. 218–
237, 2024.
Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muen-
nighoff, Xi Victoria Lin, Daniela Rus, Bryan Kian
Hsiang Low, Sewon Min, Wen-tau Yih, Pang Wei
Koh, and Luke Zettlemoyer. ReasonIR: Training Re-
trievers for Reasoning Tasks. arXiv preprint arXiv,
2025.
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres,
Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin
Clark, Stephen R Pfohl, Heather Cole-Lewis, and
others. Toward expert-level medical question answer-
ing with large language models. Nature Medicine,
1–8, 2025.

Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon
Park, Hyeon Hwang, Mujeen Sung, Hyunjae Kim,
and Jaewoo Kang. Rationale-Guided Retrieval Aug-
mented Generation for Medical Question Answering.
In Proceedings of the 2025 Conference of the Na-
tions of the Americas Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (V olume 1: Long Papers), pp. 12739–12753,
2025.
Jiayu Song, Jenny Chim, Adam Tsakalidis, Julia Ive,
Dana Atzil-Slonim, and Maria Liakata. Combining
Hierachical V AEs with LLMs for clinically mean-
ingful timeline summarisation in social media. In
Findings of the Association for Computational Lin-
guistics: ACL 2024, pp. 14651–14672, 2024.
Sarvesh Soni, Meghana Gudala, Atieh Pajouhi, and
Kirk Roberts. Radqa: A question answering dataset
to improve comprehension of radiology reports. In
Proceedings of the thirteenth language resources and
evaluation conference, pp. 6250–6259, 2022.
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan
Shi, Zachary S Siegel, Michael Tang, Ruoxi Sun, Jin-
sung Yoon, Sercan O Arik, Danqi Chen, and Tao Yu.
BRIGHT: A Realistic and Challenging Benchmark
for Reasoning-Intensive Retrieval. arXiv preprint
arXiv:2407.12883, 2024.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan,
Wen Bo, and Yunfeng Liu. Roformer: Enhanced
transformer with rotary position embedding. Neu-
rocomputing, 568, 127063, 2024.
Simon Šuster and Walter Daelemans. CliCR: a dataset
of clinical case reports for machine reading compre-
hension. arXiv preprint arXiv, 2018.
Ekaterina Sviridova, Anar Yeginbergen, Ainara Estar-
rona, Elena Cabrio, Serena Villata, and Rodrigo
Agerri. CasiMedicos-Arg: A Medical Question An-
swering Dataset Annotated with Explanatory Argu-
mentative Structures. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing, pp. 18463–18475, 2024.
Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun
Zhao, Xingyao Zhang, Arman Cohan, and Mark
B. Gerstein. MedAgents: Large Language Models
as Collaborators for Zero-shot Medical Reasoning.
ArXiv, abs/2311.10537, 2023.
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan
Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer,
Damien Vincent, Zhufeng Pan, Shibo Wang, and
others. Gemini 1.5: Unlocking multimodal under-
standing across millions of tokens of context. arXiv
preprint arXiv, 2024.
Qwen Team and others. Qwen2 technical report. arXiv
preprint arXiv, 2, 3, 2024.Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, and others. Llama 2: Open foundation and
fine-tuned chat models. arXiv preprint arXiv, 2023.
Shubham Vatsal and Ayush Singh. Can GPT Re-
define Medical Understanding? Evaluating GPT
on Biomedical Machine Reading Comprehension.
ArXiv, abs/2405.18682, 2024.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao,
Linjun Yang, Daxin Jiang, Rangan Majumder, and
Furu Wei. Text Embeddings by Weakly-Supervised
Contrastive Pre-training. arXiv preprint arXiv, 2022.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. Improving text em-
beddings with large language models. arXiv preprint
arXiv, 2023.
Huimin Wang, Yutian Zhao, Xian Wu, and Yefeng
Zheng. imapScore: Medical Fact Evaluation Made
Easy. In Annual Meeting of the Association for Com-
putational Linguistics, 2024.
Yubo Wang, Xueguang Ma, and Wenhu Chen. Augment-
ing Black-box LLMs with Medical Textbooks for
Biomedical Question Answering. In Findings of the
Association for Computational Linguistics: EMNLP
2024, pp. 1754–1770, 2024.
Orion Weller, Michael Boratko, Iftekhar Naim, and
Jinhyuk Lee. On the theoretical limitations of
embedding-based retrieval. arXiv preprint arXiv,
2025.
Michael Wornow, Yizhe Xu, Rahul Thapa, Birju Patel,
Ethan Steinberg, Scott Fleming, Michael A Pfeffer,
Jason Fries, and Nigam H Shah. The shaky founda-
tions of large language models and foundation mod-
els for electronic health records. npj digital medicine,
6(1), 135, 2023.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. Efficient streaming language
models with attention sinks. arXiv preprint arXiv,
2023.
Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan
Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and
Maosong Sun. Infllm: Training-free long-context ex-
trapolation for llms with an efficient context memory.
Advances in Neural Information Processing Systems,
37, pp.119638-119661., 2024.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. Benchmarking retrieval-augmented genera-
tion for medicine. In Findings of the Association
for Computational Linguistics ACL 2024, pp. 6233–
6251, 2024.
Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang,
Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi
Rungta, Karthik Abinav Sankararaman, Barlas Oguz,

and others. Effective Long-Context Scaling of Foun-
dation Models. In Proceedings of the 2024 Confer-
ence of the North American Chapter of the Associ-
ation for Computational Linguistics: Human Lan-
guage Technologies (V olume 1: Long Papers), pp.
4643–4663, 2024.
Peng Xu, Wei Ping, Xianchao Wu, Chejian Xu, Zi-
han Liu, Mohammad Shoeybi, and Bryan Catanzaro.
Chatqa 2: Bridging the gap to proprietary llms in long
context and rag capabilities. arXiv preprint arXiv,
2024.
Ran Xu, Wenqi Shi, Yue Yu, Yuchen Zhuang, Yanqiao
Zhu, May D Wang, Joyce C Ho, Chao Zhang, and
Carl Yang. Bmretriever: Tuning large language mod-
els as better biomedical text retrievers. arXiv preprint
arXiv, 2024.
Xi Yang, Aokun Chen, Nima M. Pournejatian, Hoo-
Chang Shin, Kaleb E. Smith, Christopher Parisien,
Colin B. Compas, Cheryl Martin, Anthony B Costa,
Mona G. Flores, Ying Zhang, Tanja Magoc, Christo-
pher A. Harle, Gloria P. Lipori, Duane A. Mitchell,
William R. Hogan, Elizabeth A. Shenkman, Jiang
Bian, and Yonghui Wu. A large language model for
electronic health records. NPJ Digital Medicine, 5,
2022.
Qwen An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei
Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Jun-
yang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin
Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin
Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang
Su, Yi-Chao Zhang, Yunyang Wan, Yuqi Liu, Zeyu
Cui, Zhenru Zhang, Zihan Qiu, Shanghaoran Quan,
and Zekun Wang. Qwen2.5 Technical Report. ArXiv,
abs/2412.15115, 2024.
W. Yu, Zhihan Zhang, Zhenwen Liang, Meng Jiang,
and Ashish Sabharwal. Improving Language Mod-
els via Plug-and-Play Retrieval Feedback. ArXiv,
abs/2305.14002, 2023.
Tan Yu, Anbang Xu, and Rama Akkiraju. In defense of
rag in the era of long-context language models, 2024.
URL org/abs/2409.01666, 2024.
Xiang Yue, Xinliang Frederick Zhang, Ziyu Yao, Simon
M. Lin, and Huan Sun. CliniQG4QA: Generating Di-
verse Questions for Domain Adaptation of Clinical
Question Answering. 2021 IEEE International Con-
ference on Bioinformatics and Biomedicine (BIBM),
580–587, 2020.
Ximing Yue, Xuefang Zhang, and Hua Sun. Clin-
iQG4QA: Generating Diverse Questions for Domain
Adaptation of Clinical Question Answering. In IEEE
BIBM, 2021.
Tianyi Zhang, Varsha Kishore, Felix Wu*, Kilian Q.
Weinberger, and Yoav Artzi. BERTScore: EvaluatingText Generation with BERT. In International Confer-
ence on Learning Representations, 2020.
Lei Zhang, Yunshui Li, Ziqiang Liu, Jiaxi Yang, Jun-
hao Liu, Longze Chen, Run Luo, and Min Yang.
Marathon: A Race Through the Realm of Long Con-
text with Large Language Models. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (V olume 1: Long Papers),
pp. 5201–5217, 2024.
Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang
Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai,
Shuo Wang, Zhiyuan Liu, and others. ∞Bench: Ex-
tending Long Context Evaluation Beyond 100K To-
kens. In Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (V ol-
ume 1: Long Papers), pp. 15262–15277, 2024.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and others. Qwen3 Em-
bedding: Advancing Text Embedding and Reranking
Through Foundation Models. arXiv preprint arXiv,
2025.
Two
Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chunyan
Miao. Medrag: Enhancing retrieval-augmented gen-
eration with knowledge graph-elicited reasoning for
healthcare copilot. In Proceedings of the ACM on
Web Conference 2025, pp. 4442–4457, 2025.
Ming Zhu, Aman Ahuja, Da-Cheng Juan, Wei Wei, and
Chandan K Reddy. Question answering with long
multiple-span answers. In Findings of the Associa-
tion for Computational Linguistics: EMNLP 2020,
pp. 3840–3849, 2020.
A Inference Setup
•Inference Engine:Nvidia TensorRT.
•Quantization:FP8.
•Hardware:7B parameter models: Deployed
on single Nvidia H100 GPUs with 40GB
VRAM. 32B parameter models with extended
context: Deployed on dual Nvidia H100
GPUs, each equipped with 80GB VRAM. A
total of 260 GPU hours was spent.
•Libraries:Qdrant, DSPy, Litellm, pandas,
Evaluate, Fastembed, Pylate
B Hyperparameters
C Datasets
C.1 Context Data Distributions
Token-length distributions per dataset are shown in
Figures 6 and 7.

Hyperparameter Value
LLMtemperature inst:0, reas:1
freq_penalty0
pres_penalty0
think tokensQwQ:20k
HuatuoGPT:8k
RAGtop_k2×num chunks
kRRF 60
k1 TF saturation
blength norm.
avgdl avg. doc length
Table 5: Hyperparameter Configuration
C.2 Context Data Preparation
We performed minimal cleaning and standardiza-
tion to support retrieval and prompting, including
merging notes by patient and stay, normalizing
timestamps, adding note-type metadata, computing
token counts for context segmentation, and filtering
to human-verified subsets where applicable.
We describe the cleaning and normalization
steps applied prior to indexing and prompting:
•Cleaning and de-duplication of clinical notes;
removal of template boilerplate where appli-
cable.
•Merging notes by patient and hospital stay;
preserving note-type and section headers for
downstream retrieval.
•Datetime standardization and chronological
ordering; normalization where timezones or
partial timestamps occur.
•Metadata augmentation (e.g., note type, en-
counter identifiers) to support Include DS vs
Include All settings.
•Tokenization and token-count computation at
note- and patient-level for context segmenta-
tion.
These details enable reproducibility of sampling
and retrieval segmentation.
C.3 Prompt Formulations
We used a simple one shot prompt structure and
tailored it explicitly for each task formulation:
•Extractive: Request the model to answer by
extracting the most relevant answer from the
context.
Figure 6: MIMIC-III Context Distributions
Figure 7: MIMIC-IV Context Distributions
•Multiple-choice: Standard multiple choice
prompt.
•Open-ended: Focusing on open ended ques-
tion answering favoring short, single sentence
answers.
Below is the sample prompt for extractive tasks:
System message :
Your i n p u t f i e l d s a r e :
1 .`m e d i c a l _ r e c o r d`( l i s t [ s t r ] ) : L i s t of p a t i e n t n o t e s ( c h r o n o l o g i c a l o r d e r ) .
2 .`q u e s t i o n`( s t r ) : A q u e s t i o n a b o u t t h e p a t i e n t ' s r e c o r d .
Your o u t p u t f i e l d s a r e :
1 .`answer`( s t r ) : S h o r t s i n g l e − s e n t e n c e answer t o t h e q u e s t i o n .
A l l i n t e r a c t i o n s w i l l be s t r u c t u r e d i n t h e f o l l o w i n g way , w ith t h e a p p r o p r i a t e v a l u e s f i l l e d i n .
[ [ ## m e d i c a l _ r e c o r d ## ] ]
{ m e d i c a l _ r e c o r d }
[ [ ## q u e s t i o n ## ] ]
{ q u e s t i o n }
[ [ ## answer ## ] ]
{ answer }
[ [ ## completed ## ] ]
In a d h e r i n g t o t h i s s t r u c t u r e , your o b j e c t i v e i s :
Given a p a t i e n t ' s m e d i c a l r e c o r d and a q u e s t i o n , answer t h e q u e s t i o n c o r r e c t l y and s h o r t l y .
C.4 Dataset Comparisons
Aiming to stress-test LLM capabilities under real
patient-centric scenarios, our dataset selection pro-
cess was based on a extensive grid of datasets. Ta-
ble 9 summarizes the QA datasets we analyzed
in our selection process, including the EHR can-
didates. The chosen EHR-based, human-verified
datasets provide diverse but comparable settings
across generative, extractive, and MC formulations.
While some exceed 8K tokens, supporting our long-
context evaluation, they all provide strong augmen-
tation potential towards a multi-note longer eval-
uation setting that allows content extension up to
128K for each patient (§4.2), enabling a realistic
comparison of full-context and RAG pipelines. We

Dataset LLM-as-a-judge QA Pairs
EHR-DS-QASelene-8B 113
Prometheus-8x7B-v2.0 98
Qwen2.5-32B-Instruct 71
EHRNoteQASelene-8B 180
Prometheus-8x7B-v2.0 220
Qwen2.5-32B-Instruct 42
Table 6: Disagreement instances where NLI Med
Contradiction is high (>0.7) and LLM Correctness is
also high (>0.7).
Dataset LLM-as-a-judge1LLM-as-a-judge 2 QA Pairs
EHR-DS-QAPrometheus Qwen2.5-32B-Instruct 520
Prometheus Selene-8B 363
Qwen2.5-32B-Instruct Selene-8B 664
EHRNoteQAPrometheus Qwen2.5-32B-Instruct 690
Prometheus Selene-8B 236
Qwen2.5-32B-Instruct Selene-8B 519
Table 7: Judge-pair disagreements where the absolute
difference in LLM Correctness≥0.5.
Metric Splade BM25
LLM Correctness 67.3069.34
LLM Completeness 65.4167.14
LLM Faithfulness47.6446.23
NLI Med Entailment 52.6854.35
NLI Med Contradiction 16.9816.94
Bio BERTScore F179.7579.67
METEOR43.9343.63
Table 8: Comparison of Different Sparse Retrievers
provide citations of each considered dataset from
Table 9 in Table 10.
D Model Comparisons
D.1 Retrievers
On Table 8 we provide a sparse retriever compari-
son on the EHR-DS-QA dataset using the Qwen2.5-
7B-Instruct-1M LLM and Qwen3-Embedding-8B
as the dense retriever. BM25 has better per-
formance across LLM Correctness, LLM Com-
pleteness and NLI-based metrics while showcas-
ing slightly inferior performance on BioClinical
BERTScore F1, METEOR and LLM Faithfulness.
D.2 LLM-as-a-judge
In Table 6 we examine the number of QA cases
for each dataset that the different Judges cause dis-
agreement with the NLI Med Contradiction met-ric. We observe that Qwen2.5-32B-Instruct5has
far less such disagreement compared to more spe-
cialized judges. Table 7 also provides direct com-
parisons of LLM-as-a-judge model, demonstrat-
ing cases where LLM Correctness between such
pairs is more than 0.5 and therefore high. Selene-
8B6and Prometheus-8x7B-v2.07comparisons
show that these two models are consistently more
in agreement between them while Qwen2.5-32B-
Instruct shows more independent judging abilities.
D.3 Context Size Performance
Figure 8 presents performance for on open-ended
QA across metrics and models.
E Metrics
E.1 Correlations
We examine correlations between evaluation
metrics to identify metric independence. On
EHRNoteQA and EHR-DS-QA across all non-
exclude settings (Figure 9) and long-context high
LLM Correctness non-exclude settings (Figure 10),
LLM evaluation metrics exhibited strong inter-
correlations and METEOR and BERTScore were
highly correlated. Although the above correlations
remain high, they are less pronounced for the long-
context high LLM Correctness setting of Figure 10.
In general all metrics are less correlated for the
longer context setting.
E.2 Analysis
Table 11 showcases a qualitative analysis on what
metrics are capturing vs missing based on analyz-
ing disagreement data.
F Case Studies
Here we show sum sample studies of our error
analysis.
EHR-DS-QA: Case 10083814 (final diagnoses).
Three discharge summaries exist across distinct ad-
missions; each lists diagnoses. The correct target
for “final diagnoses” is the most recent summary.
Earlier summaries contain different diagnoses that
are no longerfinalat discharge, explaining predic-
tion–gold divergence without model hallucination.
5https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
6https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-
3.1-8B
7https://huggingface.co/prometheus-eval/prometheus-
8x7b-v2.0

Figure 8: Model performance over context size.
Figure 9: Metric correlations across all non-exclude settings.
Figure 10: Metric correlations across all non-exclude settings filtered by LLM Correctness>50and Context
>= 16K.

Dataset # QA Pairs Token Len. (max) Task Types Synthetic Annotation Reasoning
Medical Textbooks, Websites, Knowledge Bases
MedQuAD 47,457 < 2K extractive automatic× ××
MashQA 34,808 < 2K extractive (multi-span) experts✓
MedOdyssey(En.KG)100 < 128K extrative (graph) model&human ✓
MedQA(USMLE)12,723 > 1M MC experts✓
Biomedical Literature
BioMRC(LARGE)812,707 < 2K MC, extractive automatic× ××
BioASQ(b)5,729 < 8K extractive, generative experts✓
PubMedQA 1,000 < 2K MC, generative experts✓
Clinical Case Reports
CliCR 104,919 < 8K extractive (cloze) automatic✓
Patient Notes
LongHealth 400 < 8K MC✓experts✓
DiSCQ 2,029 < 2K extractive, generative experts✓
RadQA 6,148 < 16K extractive experts✓
CliniQG4QA 8,824/1,287 < 8K extractive model/experts ×××
EHR-DS-QA 156,599/478 < 8K generative✓ model/experts ×××
EHRNoteQA 962 < 8K MC, generative experts✓
Table 9: Biomedical Datasets Comparison. References for each dataset are in Table 10.
Dataset Source
MedQA Jin et al. (2021)
MedQuAD Abacha and Demner-
Fushman (2019)
MashQA Zhu et al. (2020)
MedOdyssey Fan et al. (2024)
BioMRC Pappas et al. (2020)
BioASQ Nentidis et al. (2025)
PubMedQA Jin et al. (2019)
CliCR Šuster and Daelemans (2018)
LongHealth Adams et al. (2024)
MediNote Leong et al. (2024)
DiSCQ Lehman et al. (2022)
RadQA Soni et al. (2022)
CliniQG4QA Yue et al. (2020)
EHR-DS-QA Kotschenreuther (2024)
EHRNoteQA Kweon et al. (2024)
Table 10: Corresponding Citations for Datasets
considered in Dataset Selection of Table 9
EHRNoteQA: Case 15877599 (cause of AKI).
The note supports a causal chain: gastroenteritis
→increased ostomy output →severe dehydration
(prerenal) →acute kidney injury. The gold cap-
tures the underlying illness; a model response may
describe the immediate mechanism. Both are clini-
cally coherent parts of the same sequence.EHR-DS-QA: Case 10023117 (blood pressure at
discharge).Predictions expressed as exact values
versus ranges produce different behaviors across
metrics. Ranges can be faithful to notes yet fail en-
tailment or completeness thresholds defined against
a single gold value.
G RAG vs FC: Detailed Analysis
We summarize our insights about which RAG or
FC setting tends to be advantageous across datasets
and query types in Table 13. We also provide ex-
amples along each of our main findings below:
FC performs better for questions requiring com-
prehensive understandingof the entire document
or when answers are located in structured sections,
such as summaries, lists, or temporal sequences.
Examples include:
•Questions like “What is the patient’s discharge
condition?” or “What were the patient’s dis-
charge diagnoses?”.
• Temporal sequence questions, such as “What
surgeries has the patient undergone and in
what order?”.
RAG outperforms FC for questions that require re-
trieving specific detailsor synthesizing information
scattered across multiple parts of the document.
Examples include:
•Questions like “What family history does the
patient have?” or “How many tablets of dilau-
did did the patient receive?”.

What they capture well What they miss Typical pattern Examples (dataset #subject)
LLM Correctness
Whether the main claim is right. Unsupported add-ons that don’t
change the core fact.Correctness high; Faithfulness
low (adds extra).EHRNoteQA #10043423 (core claim right, adds recommendation);
EHRDSQA #10094318 (chief complaint right, extra symptom).
LLM Completeness
Coverage of the requested
pieces/elements.Penalizes omissions but not un-
grounded embellishments.Completeness high; Faithful-
ness low (fully covers, then em-
bellishes).EHRNoteQA #10043423 (covers requested items + extras);
EHRDSQA #19796003 (lists largely complete; minor extras else-
where reduce faithfulness/F1).
LLM Faithfulness
Grounding to evidence/gold; flags
hallucinated or embellished con-
tent.Can be over-strict on benign con-
text or plausible but unsupported
expansions.Correct/Complete high; Faith-
fulness low (ungrounded detail).EHRNoteQA #10043423 (extra recommendation not supported);
EHRDSQA #10131388 (adds follow-up/psychiatric context not
explicit).
NLI Med Entailment / Contradiction
Sentence-level logical relation to
gold (entailed vs. contradicted).Sensitive to negation, hedging
(“low risk”vs“no risk”), and
phrasing templates.Correctness high; Entailment
low or Contradiction high (word-
ing/negation mismatch).EHRNoteQA #10494486 (mechanism phrasing hurts entailment);
EHRDSQA #10010655 (“no risk”vs“low acute risk”triggers
contradiction).
Meteor
Surface-form overlap, good for
close paraphrases.Under-rewards long phrases, syn-
onyms, and re-phrasings with low
lexical overlap. Incorrectly re-
wards inaccuracies with lexical
overlap.Correctness high; Meteor low
(semantic match, low surface
match).EHRDSQA #10076958 (concise “diffuse ischemic bowel” vs long
gold narrative); #19796003 correct single word answer but low
score;
EHRNoteQA #10404814 (correct causal link, different word-
ing/structure).
BioClinical F1
Token/span overlap for clinical en-
tities; good checklist signal for
missing/extra items.Under-rewards clinically accept-
able reformulations (class vs. spe-
cific drug),. Incorrectly rewards
inaccuracies with lexical overlap.Correctness high; F1 moder-
ate/low (one entity missing or
formatted differently).EHRNoteQA #10043423 (right ideas, entity list/format differences
lower F1);
EHRDSQA #19796003 (near-exact meds but lower F1 for small
misses).
Table 11: What each metric captures vs. misses, with typical disagreement patterns and representative examples
from EHRNoteQA and EHRDSQA.
Prediction LLM Corr. LLM Compl. LLM Faith. NLI Ent. NLI Contr. BioClin F1 Meteor
The patient’s blood pressure at discharge
was SBP in the 80–90s.1.00 0.25 1.00 0.21 0.79 0.80 0.79
The patient’s blood pressure at discharge
was 107/76.0.00 0.00 0.25 0.00 1.00 0.90 0.82
The patient’s blood pressure at discharge
was 80–100s/55–70s.1.00 0.00 0.75 0.96 0.04 0.83 0.82
The patient’s blood pressure at discharge
was 107/59 to 80–100s/55–70s.1.00 0.00 0.25 0.09 0.91 0.83 0.80
Table 12: Metrics performance on numerical data.
•Synthesis tasks, such as “What were the pa-
tient’s postoperative course details?”.
•Asking about specific dates like “What was
the outcome of the patient’s colonoscopy as
described in the discharge summary from the
stay starting on 2113-09-30?” or “What was
the patient’s diagnosis for the hospital admis-
sion on 2154-01-28. . . ” and other examples.
RAG tends to perform better overall in datasets
with complex or lengthy documents.
For example:
•In the EHRNoteQA dataset, RAG consistently
outperformed FC for questions needing spe-
cific details from notes or summaries, such
as “What was the outcome of the patient’s
colonoscopy?”
There is mixed evidence suggesting FC might per-
form better for tasks involving inferential reasoning
or identifying the absence of information.
For example:•Questions like “Were there any complications
during the procedure?” where RAG retrieves
statements like “No complications,” poten-
tially diminishing FC’s advantage.
•Subtle inference tasks, such as “Does the pa-
tient have any psychological issues?” where
FC occasionally performs better, though in-
consistently.

Insight Favored Explanation Supporting Examples
Specific Fact Re-
trievalRAG RAG excels at extracting pre-
cise, well-defined medical facts
(dates, medications, lab values,
procedures) that are typically
documented in structured sec-
tions of medical records.EHRNoteQA:
Subject 15036658:Colonoscopy outcome from specific date (RAG: 0.632 vs.
FC: 0.399)
Subject 11049732:Medication changes (RAG: 0.829 vs. FC: 0.352)
Subject 17818938:Surgical procedure for erectile dysfunction (RAG: 0.918
vs. FC: 0.479)
EHR-DS-QA:
Subject 10131388:Dilaudid tablet count (RAG: 0.982 vs. FC: 0.778)
Subject 19926045:DVT medication (RAG: 0.901 vs. FC: 0.564)
Subject 10090787:Discharge medications (RAG: 0.425 vs. FC: 0.190)
Temporal Infor-
mation Process-
ingMixed RAG excels at explicit temporal
facts (specific dates, temporal re-
lationships) while FC is better at
temporal reasoning (sequencing,
duration calculation, recogniz-
ing absence of temporal infor-
mation).RAG Advantage:
EHRNoteQA 15036658:Specific date anchoring
EHRNoteQA 18467824:Temporal relationship between admissions
EHR-DS-QA 10264949:Nausea/vomiting timing
FC Advantage:
EHRNoteQA 11552479:Temporal sequencing (FC: 0.696 vs. RAG: 0.236)
EHR-DS-QA 10751849:Duration calculation (FC: 0.541 vs. RAG: 0.320)
EHR-DS-QA 19397212:Absence of temporal information (FC: 0.426 vs.
RAG: 0.180)
Medical Termi-
nology and Tech-
nical ContentRAG RAG performs better with spe-
cialized medical terminology,
complex procedures, and techni-
cal test results due to its ability
to locate and interpret specific
sections containing this informa-
tion.EHRNoteQA:
Subject 17445067:Diagnosis and surgical procedure details (RAG: 0.708 vs.
FC: 0.328)
Subject 18122852:MRI and EMG test findings (RAG: 0.750 vs. FC: 0.454)
Subject 16313269:Brain mass pathological diagnosis (RAG: 0.915 vs. FC:
0.372)
EHR-DS-QA:
Subject 10044189:Necrotic ulcer treatment (RAG: 0.762 vs. FC: 0.465)
Subject 19401508:Treatment for hyponatremia (RAG: 0.479 vs. FC: 0.262)
Discharge Plan-
ning and Instruc-
tionsRAG RAG performs better on Dis-
charge information (instructions,
medications, condition) that is
usually well-structured in spe-
cific sections.EHRNoteQA:
Subject 11690633:Discharge condition and instructions (RAG: 0.762 vs. FC:
0.349)
Subject 11863782:Discharge disposition and medications (RAG: 0.749 vs.
FC: 0.327)
EHR-DS-QA:
Subject 10921250:Discharge condition (RAG: 0.856 vs. FC: 0.466)
Subject 10940920:Discharge instructions (RAG: 0.650 vs. FC: 0.253)
Subject 10064678:Discharge instructions (RAG: 0.352 vs. FC: 0.094)
Cause–Effect
and Relationship
UnderstandingRAG RAG is better at understanding
relationships (symptom–
procedure, medication–
outcome, test result–action).EHRNoteQA:
Subject 13032648:Causes of leg pain and surgical procedure (RAG: 0.730
vs. FC: 0.293)
Subject 15748482:Flomax usage reason and outcome (RAG: 0.819 vs. FC:
0.495)
Subject 17436366:Blood/urine culture results and actions (RAG: 0.718 vs.
FC: 0.445)
EHR-DS-QA:
Subject 19926045:Lovenox symptom management (RAG: 0.545 vs. FC:
0.334)
Subject 19401508:Admission cause and treatment (RAG: 0.372 vs. FC:
0.108)
Holistic Patient
UnderstandingFC FC excels when questions re-
quire synthesis of information
across multiple document sec-
tions to build a complete picture
of the patient’s status, multiple
diagnoses.EHRNoteQA:
Subject 18753609:Therapeutic interventions for leg pain (FC: 0.675 vs.
RAG: 0.450)
EHR-DS-QA:
Subject 10262565:Discharge condition (FC: 0.882 vs. RAG: 0.681)
Subject 10049941:Discharge diagnoses (FC: 0.579 vs. RAG: 0.188)
Subject 10978236:Discharge condition — unusual circumstance (FC: 0.321
vs. RAG: 0.093)
Absence or
Negative In-
formation
RecognitionFC FC is better at recognizing
when information is absent or
when negative findings are doc-
umented, as it can assess the en-
tire document context.EHR-DS-QA:
Subject 19397212:Absence of symptom presentation timing (FC: 0.426 vs.
RAG: 0.180)
Subject 10751849:Absence of major procedures (FC: 0.243 vs. RAG: 0.037)
Subject 10264949:Absence of social factors (FC: 0.497 vs. RAG: 0.149)
Subject 19397212:Absence of age information (FC: 0.725 vs. RAG: 0.452)
Table 13: RAG vs FC Detailed Analysis.