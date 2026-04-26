# Domain-oriented RAG Assessment (DoRA): Synthetic Benchmarking for RAG-based Question Answering on Defense Documents

**Authors**: Bao Gia Doan, Aditya Joshi, Pantelis Elinas, Aarya Bodhankar, Oscar Leslie, Tom Marchant, Flora Salim

**Published**: 2026-04-20 08:22:15

**PDF URL**: [https://arxiv.org/pdf/2604.17943v1](https://arxiv.org/pdf/2604.17943v1)

## Abstract
Open-domain RAG benchmarks over public corpora can overestimate deployment performance due to pretraining overlap and weak attribution requirements. We present DoRA (Domain-oriented RAG Assessment), a domain-grounded benchmark built from defense documents that pairs synthetic, intent-conditioned QA (question answering) with auditable evidence passages for attribution. DoRA covers five question types (find, explain, summarize, generate, provide) and contains 6.5K curated instances. In end-to-end evaluation with a fixed dense retriever, general-purpose Language Models (LMs) perform similarly, while a model trained on DoRA (DoRA SFT) yields large gains over the base model (Llama3.1-8B-Instruct): up to 26% improvement in QA task success, while reducing the hallucination rate by 47% in RAG faithfulness scores, supporting contamination-aware regression testing under domain shift.

## Full Text


<!-- PDF content starts -->

Domain-o riented R AG A ssessment (DoRA): Synthetic Benchmarking for
RAG-based Question Answering on Defense Documents
Bao Gia Doan†, Aditya Joshi†, Pantelis Elinas†, Aarya Bodhankar†,
Oscar Leslie‡, Tom Marchant‡, Flora Salim†
†UNSW Sydney, Australia
‡Cyndr AI, Australia
{bao.doan1,aditya.joshi,p.elinas,a.bodhankar,flora.salim}@unsw.edu.au
{oscar,tom}@cyndr.ai
Abstract
Open-domain RAG benchmarks over pub-
lic corpora can overestimate deployment per-
formance due to pretraining overlap and
weak attribution requirements. We present
DoRA (Domain-oriented RAG Assessment),
a domain-grounded benchmark built from de-
fense documents that pairs synthetic, intent-
conditioned QA (question answering) with
auditable evidence passages for attribution.
DoRA covers five question types (FIND,EX-
PLAIN,SUMMARIZE,GENERATE,PROVIDE)
and contains 6.5K curated instances. In end-
to-end evaluation with a fixed dense retriever,
general-purpose Language Models (LMs) per-
form similarly, while a model trained on DoRA
(DoRA SFT) yields large gains over the base
model (Llama3.1-8B-Instruct): up to 26% im-
provement in QA task success, while reducing
the hallucination rate by 47% in RAG faithful-
ness scores, supporting contamination-aware
regression testing under domain shift.
1 Introduction
Retrieval-augmented generation (RAG) has
rapidly become a popular paradigm for building
knowledge-intensive NLP applications (Lewis
et al., 2020; Gao et al., 2023). Commercial
RAG-based applications are deployed to answer
questions overspecificdocument collections, and
can generate structured and attributable outputs.
Despite that, evaluative RAG benchmarks remain
dominated by open-domain, Wikipedia-style
benchmarks (Kwiatkowski et al., 2019; Yang et al.,
2018; Joshi et al., 2017; Petroni et al., 2021).
However, three recurring failure modes occur
in practice:(i) contamination—public corpora
overlap with pre-training and instruction-tuning
data, which inflates scores and masks brittle gener-
alization (Zhu et al., 2024; Li et al., 2024; Carlini
et al., 2021);(ii) mismatch of user intents—factoid
QA under-represents question types expected inpractice (Filice et al., 2025; Friel et al., 2024); and
(iii) misaligned metrics—overlap-based metrics
(BLEU/ROUGE) reward surface similarity while
under-weighting attribution and faithfulness,
which are central in deployed retrieval pipelines
(Zhu et al., 2025; Lin, 2004). Consequently,
improvements on open-domain leaderboards often
fail to predict behavior on proprietary corpora
in domains such as Defense where the cost of
an ungrounded answer can be operationally
significant. This gap, compounded by the difficulty
of producing high-quality, domain-specific
human annotations, motivates the development of
automated approaches based on scenario-synthetic
datasets (Zhu et al., 2025).
In this paper, we target this gap with DoRA
(Domain-oriented RAG Assessment): an evalu-
ation suite for domain-specific RAG built from
synthetic, type-conditionedquestion–answer pairs
grounded in defense source documents, for our de-
fense technology industry partner. DoRA is in-
tended not as a one-off benchmark, but as a repeat-
able evaluation harness used throughout the life-
cycle of a RAG system—from early prototyping
through to regression testing as corpora, retriev-
ers, and models evolve—while minimizing expo-
sure to public-corpus contamination. In building
DoRA, our design goals are: (1)domain orien-
tation(avoid Wikipedia-style leakage and accom-
modate private corpora), (2)question type cover-
age(capture a broader distribution of real practi-
tioner asks beyond factoids), and (3)deployment-
relevant scoringusing modern RAG evaluation
protocols (Es et al., 2024; Ming et al., 2024; Chen
et al., 2024a; Friel et al., 2024; Zhu et al., 2025).
2 Motivation: The Defense Context
Defense and national security settings act as a stress
test for RAG systems: private corpora, strict access
controls, and decision-critical use cases exposearXiv:2604.17943v1  [cs.CL]  20 Apr 2026

weaknesses that remain hidden in open-domain
benchmarks. Language systems operating in these
environments must search, explain, and synthesize
across heterogeneous internal sources that evolve
continuously and require auditing. For such critical
use cases, an assistant that ismostly correctbut
occasionally hallucinates a number, omits a crucial
caveat, or fails to cite evidence can create unaccept-
able downstream risk. In defense and national se-
curity settings, these failures do not merely reduce
answer quality; they can distort planning assump-
tions, capability assessments, and downstream deci-
sions with material operational and investment con-
sequences. In many defense environments, users
operate under partial visibility and strict access con-
trols, meaning evaluation must assess grounding
and attribution only within authorized subsets of
the corpus. We develop this pipeline to support an
emerging enterprise RAG system for government
and defense stakeholders, where defense-focused
users require auditable, evidence-linked outputs
for operational decision-making. To align with
the user needs, DoRA coversfive question types
—FIND,EXPLAIN,SUMMARIZE,GENERATE, and
PROVIDE—enabling targeted evaluation of retrieval
quality, generation quality, and their interaction un-
der realistic requests. The question types map di-
rectly to intents in defense workflows, such as fact
checking and briefing preparation (FIND), policy
and capability interpretation (EXPLAIN), report di-
gestion and situational awareness (SUMMARIZE),
checklist and option framing (GENERATE), and
quantitative inputs to planning and procurement
(PROVIDE). In providing an enabling infrastruc-
ture to evaluate RAG systems, DoRA makes the
following contributions:
1.A domain-grounded benchmark and refer-
ence harness.We generate synthetic Q&Adi-
rectlyfrom defense source documents and pair
each instance with auditable evidence bundles,
enabling evidence-grounded, task-aware scoring
(e.g., NLI-based support/contradiction signals,
extractive span alignment, and numeric consis-
tency) and a standardized reference harness for
reproducible benchmarking1.
2.Defense Domain: Ours is the first investigation
into an evaluative benchmark for the defense
domain.
3.Benchmark-driven domain adaptation
1Source code will be released after acceptance
Styled QA
GenerationData Preparation
Synthetic Data Generation
DoRA Domain
Documents
indicates stages primarily driven by LLMsChunking Indexing
Evidence
SelectionQuality
Control
SFT RAGEvalDomain-expert AIDomain RAG
faithfulness
DoRA
DatasetFigure 1: DoRA pipeline, from data preparation to
grounded-styled QA generation, and downstream do-
main evaluation and adaptation.
(DoRA SFT).Beyond evaluation, we show how
DoRA can serve as SFT-ready supervision: a
LoRA-adapted open model trained on DoRA
improves both task success and faithfulness di-
agnostics over strong general-purpose baselines
under the same retriever setting, supporting an
industry workflow of iterative, regression-tested
improvement.
DoRA has been integrated into a web-based proto-
type for the defense tech partner who will deploy it
in the appropriate setting.
3 Methodology
3.1 Formulation
LetCdenote a domain corpus seg-
mented into passages with IDs, and let
Sdenote the set of intent styles: S=
{FIND,EXPLAIN,SUMMARIZE,GENERATE,PROVIDE}.
The resultant dataset arising from the DoRA
pipeline is D={(q i, ai, si, Ei)}N
i=1. Each
instance is a quadruple (q, a, s, E) comprising a
question q, reference answer a, a style s∈ S , and a
set of supporting passages E⊂ C . An LLM-based
agent produces a prediction ˆafor each question q.
In RAG mode, we model this as
ˆa=G(q, R k(q)),
where Rkis a retriever that returns the top- kpas-
sages from CandGis a generator that conditions
on both the question and retrieved context (with the
LLM-only setting as the special caseR k(q) =∅).

Style # Avg|q|Avg|a|Refs Qual.
EXPLAIN165 13.67 33.59 2.99 0.78
FIND421 12.65 20.83 1.00 0.88
GENERATE309 13.76 31.42 3.34 0.78
PROVIDE155 14.00 30.45 1.91 0.76
SUMMARIZE209 12.85 29.35 2.65 0.82
Total/Avg1259 13.44 29.53 2.38 0.80
Table 1: DoRA benchmark statistics by style: sample
counts, average question/answer lengths (in tokens),
average references per item, and average construction-
time quality score.
3.2 Synthetic Data Generation
We construct DoRA from 20seed documents(de-
fense documents and reports) by chunking each
document into passages (forming C) and generat-
ing candidate evidence Eper seed. For each style
s, we select a small evidence set Efrom the cur-
rent seed and prompt a style-specific LLM template
(see Section H) to generate (q, a) conditioned on
(s, E) ; we store Efor auditability. Each style has
an evidence budget ks, which shapes evidence se-
lection and downstream quality checks. We model
this synthesis step abstractly as sampling from a
style- and evidence-conditionedgenerator:
(q, a)∼p ϕ(· |s, E),(1)
where ϕdenotes the parameters of the chosen LLM
and prompt configuration. Details of chosen pa-
rameters are in Section G.
Task-conditioned evidence selection.Evidence
selection chooses a small evidence bundle (refer-
ence contexts) Econditioned on the target style
s, which determines both the desired cardinality
ks≜|E| and the selection behavior (e.g., extrac-
tive single-chunk evidence forFINDversus multi-
chunk synthesis for long-form styles). During
dataset construction, the candidate pool is typically
restricted to passages from the current seed docu-
ment,Cseed⊂ C. We first apply a Natural Language
Inference (NLI)-based prefilter to remove clearly ir-
relevant candidates and obtain a reduced set, before
using an LLM to get to the final set eCseed(top-Ks
by NLI relevance). The LLM explicitly rates bun-
dlecompleteness,complementarity(non-redundant
coverage),coherence, andtask fitness; the highest-
scoring bundle is returned. Formally, we select:
E∗= arg max
E⊆eCseed
|E|=k sgs(E),(2)where gs(E)is a style-conditionedbundlescorer
computed from the rubric dimensions above (with
configurable weights and bounded search over can-
didate bundles). Details are in Section G.
Grounding and Quality Control.Each instance
stores an auditable evidence bundle E(passage
references), and we accept a generated item only if
it satisfies style-specific deployability constraints
and meets an evidence-grounded quality threshold:
accept(q, a, s, E)⇐⇒h s(q, a, E) = 1
∧Q s(q, a, E)≥τ s,
where hsencodes style-specific hard checks (e.g.,
non-empty fields;PROVIDEmust contain numeric
content) and Qsis a composite, evidence-grounded
score with thresholdτ s.
We measure grounding against Eusing
evidence-grounded metrics such as document en-
tailment score (DES; sentence-level NLI with con-
tradiction penalties), extractive span alignment
(SQuAD-style span F1) for short answers (Ra-
jpurkar et al., 2016), and numeric/date consistency
when numbers are present; for longer-form in-
tents we also compute attribution-oriented coverage
(context recall/precision against E) and question–
answer relevancy (semantic similarity between q
anda), following common RAG evaluation prac-
tice (e.g., Es et al., 2024). Concretely, Qsaggre-
gates a style-specific metric set Mswith config-
urable weightsw s,m:
Qs(q, a, E) =X
m∈M sws,mm(q, a, E),(3)
followed by deduplication and diversity filtering
to reduce repeated patterns and maintain balanced
coverage across the five styles. At the end of this
process, we obtain our DoRA dataset consisting of
the quadruples defined above.
Utility.We use the synthetically generated QA
dataset for benchmarking, considering multiple set-
tings such as alternative retrievers, supervised fine-
tuning, in-context examples and a comparison with
a manually curated dataset. The rest of the paper
covers the detailed results of our evaluation.
4 Experiment Setup
Dataset.DoRA dataset comprises around 6.5K
domain-grounded Q&A pairs spanning five ques-
tion types. We split aheld-out test setof 1,259

Task success Faithfulness (RAGEval)
Model Tok. F1 ROUGE-L BLEU BLEURT BERTScore F1Comp.↑Hall.↓Irr.↓
GPT-4o 37.89 38.67 14.72 -0.426 83.18 55.85 7.51 36.64
Llama3.1-8B-Instruct (base) 38.13 38.88 14.46 -0.416 83.30 56.23 7.04 36.73
GPT-4o-mini 38.24 38.92 14.61 -0.413 83.34 56.50 7.34 36.15
GPT-3.5-Turbo 38.31 39.07 14.42 -0.412 83.34 56.71 7.21 36.08
Claude-3-Haiku 38.00 38.70 14.77 -0.429 83.19 55.89 6.63 37.48
Mistral-7B-Instruct 37.98 38.65 14.59 -0.420 83.20 55.83 6.65 37.52
Llama3-8B-Instruct 38.27 38.96 14.67 -0.417 83.31 55.83 7.13 37.04
Qwen2.5-7B-Instruct 37.74 38.50 14.65 -0.425 83.20 55.68 7.09 37.23
DoRA SFT (Ours)56.53 56.52 40.56 0.015 89.54 66.59 3.50 29.91
Table 2: End-to-end QA with the GTE retriever: task-success QA metrics (left, percentages except BLEURT) and
RAGEval-style faithfulness diagnostics (Zhu et al., 2025) (right, percentages).
k0.60.8Hit@k
BGE-M3
BM25
IBM Granite
GTE
MiniCPM
135 10 20
k0.20.4Recall@k
Figure 2: Retriever performance across top- kon DoRA.
GTE retriever achieved overall better performance.
instances as the benchmark evaluation set; sum-
mary statistics are shown in Table 1. The rest of
DoRA dataset provides a larger pool suitable for
SFT (5,150 instances split 90/10 into train/eval)
for domain adaptation experiments. Details of this
training set are in Section E.
Experimental design.We evaluate (i) retrieval
performance and (ii) end-to-end RAG answering.
We report two complementary settings. First, to
measure end-to-end RAG performance under re-
alistic retrieval errors, we evaluate task success
(QA) and RAG faithfulness with a fixed retriever
(e.g., IBM Granite (IBM-Research, 2025) or GTE-
multilingual (Alibaba-NLP, 2025)). Second, we
evaluate the same metrics under anoracleretriever
that returns the recorded gold evidence bundle E
for each question; this isolates generation behavior
and provides an auditable reference for attribution
in domain deployments.
Metrics.For retriever evaluation, we report
Hit@ kand Recall@ k(Zhu et al., 2025; Manning
et al., 2008; Thakur et al., 2021) (see Section C for
definitions). For end-to-end QA, we report task-sucess metrics (token overlap and semantic simi-
larity) such as Token F1, ROUGE-L (Lin, 2004),
BLEU (Papineni et al., 2002), BLEURT (Sellam
et al., 2020), and BERTScore-F1 (Zhang et al.,
2019). To assess grounding under retrieval, we ad-
ditionally report RAGEval diagnostics (complete-
ness/hallucination/irrelevance) (Zhu et al., 2025).
We report micro-averages over the full test set and
inspect per-type breakdowns for diagnosis.
5 Results
5.1 Retriever Evaluation
Following recent work of RAGEval (Zhu et al.,
2025), we evaluate four closely related retriev-
ers, along with the emerging IBM Granite re-
triever (IBM-Research, 2025), across various top- k
values. Figure 2 shows that, on defense documents,
BM25 provides strong early hits but lags dense re-
trievers in broader coverage at larger k, motivating
dense/hybrid retrieval in deployment-facing RAG,
which is different from findings on open domains
as in (Zhu et al., 2025).
5.2 QA Evaluation
We evaluate end-to-end QA with a fixed retriever
(GTE-multilingual-Base, top- k=3; justified from
Section 5.1). Results in Table 2 show that under
a fixed retrieval setting, general-purpose LLMs
cluster closely on both task-success and faith-
fulness, suggesting that retrieval and grounding
constraints dominate performance in this private,
defense-domain corpus. Oracle-context results in
Section A.1 (providing gold evidence E) confirm
this finding once the retrieval error is removed.
We Supervise-Fine-Tune (SFT) an open model
(Llama3.1-8B-Instruct)2with LoRA (Hu et al.,
2We choose Llama 3.1–8B for its deployment-friendly

2021) on the DoRA training split (5,150 instances;
90/10 train/eval). Each training example concate-
nates the question with its evidence context ctx(E)
and predicts the reference answer, mirroring the
end-to-end RAG setting. We optimize:
min
θX
(q,a,E)∈D train−logp θ(a|q,ctx(E)),(4)
where θdenotes the (adapter) parameters. As
shown in Table 2, DoRA SFT yields large gains
over the base model (Llama3.1-8B-Instruct: +18.4
Tok. F1, +17.6 ROUGE-L, +26.1 BLEU) and im-
proves RAGEval faithfulness (completeness +10
percentage points, hallucination by halved, and
irrelevance reduced), indicating that even with
lightweight domain adaptation on DoRA can com-
plement retrieval error for improved task success
and faithfulness of domain RAG-based systems.
5.3 Baseline Comparison with In-Context
Learning
In the absence of any other related baselines, we
use In-Context Learning (ICL) with 1-shot, 2-shot
QA curated from the DoRA dataset as alternative
baselines. We pick (experimentally determined)
pairs for each question type, then feed them to
the prompt of two baseline models (GPT-4o and
Llama3.1-8B (our base model)). Results in Fig-
ure 3 demonstrate the superior performance of our
DoRA SFT compared to simple ICL with even a
large model like GPT-4o.
0.40.6Score0.565
0.431 0.4370.392 0.396ROUGE-L
0.80.9Score0.895
0.861 0.8630.842 0.842BERTScore-F1
DoRA SFTGPT-4o 1-shot GPT-4o 2-shotLlama3.1 1-shot Llama3.1 2-shot0Score0.015
-0.184 -0.152
-0.347 -0.351BLEURT
Figure 3: Our DoRA SFT model vs ICL baselines.
footprint and prior adoption in defense and national security
AI contexts (Meta AI, 2024; Kapko, 2024).5.4 Skyline Comparison with Manually
Curated Dataset
Finally, as a skyline where high-precision ex-
pert supervision is available, we curate an expert-
annotated set from the same seed documents. A
domain expert authors 25 seed Q&A pairs (5 per
question type), which we use to bootstrap LLM
generation of an additional 100 type-conditioned
questions paired with retrieved supporting pas-
sages. After automatic filtering with an LLM judge,
the expert answers the remaining valid questions
(86/100)withoutseeing LLM-generated answers,
yielding an expert-answered, evidence-linked set.
Full details and results are in Section B, showing
that DoRA SFT model with expert supervision im-
proves over the base Llama3.1-8B model and GPT-
4o on expert-answered questions, suggesting that
even limited manual supervision can strengthen
domain robustness.
6 Discussion
Tight clustering among general-purpose LMs.
Under the same retriever configuration, metrics
for general-purpose LMs are close (Table 2), sug-
gesting that evaluation in private, domain-specific
settings can be bottlenecked by evidence selection
and grounding rather than by raw model size.
Retrieval vs. generation.On defense documents,
early-rank lexical matching remains strong (BM25
Hit@1), but dense retrieval improves breadth and
stability at larger k, making dense retrieval attrac-
tive for deployment-facing RAG in this domain.
DoRA SFT as a practical complement.DoRA
also enables lightweight domain adaptation: a
LoRA-adapted model trained on DoRA achieves
markedly higher task-success scores and improved
faithfulness diagnostics (Table 2). This supports
an industry workflow where teams combine re-
trieval with small, auditable adaptation steps and
validate each iteration via regression testing on a
fixed benchmark.
Industry implications.DoRA is designed to
cater to constraints prevalent in defense and paral-
lel industry segments: private and evolving corpora,
provenance requirements, and the need for repeat-
able regression tests when information formats, re-
trievers, prompt templates, or models change. By
pairing intent coverage with auditable evidence,
DoRA supports bothmeasurement(question type-
stratified evaluation and diagnostic metrics) and

engineering(a reproducible harness and SFT-ready
data) for real-world deployments.
7 Related Work
QA benchmarks (Kwiatkowski et al., 2019; Ra-
jpurkar et al., 2016; Yang et al., 2018; Joshi et al.,
2017; Petroni et al., 2021) predominantly target
open-domain, Wikipedia-centric sources. Comple-
mentary retrieval evaluations like BEIR (Thakur
et al., 2021) study retriever generalization across
diverse datasets, and multi-hop benchmarks probe
compositional reasoning that is often paired with
RAG (Tang and Yang, 2024). Recent work has in-
troduced automated evaluation for faithfulness and
attribution for RAG pipelines, including RAGAS
(Es et al., 2024), ARES (Saad-Falcon et al., 2024),
FaithEval (Ming et al., 2024), LLM-in-RAG bench-
marking (Chen et al., 2024a), and RAGBench (Friel
et al., 2024). Diagnostic tools such as RAGChecker
(Ru et al., 2024) further decompose failures into an-
swerability, support, and hallucination. DoRA com-
plements this line of work by emphasizingdomain-
groundedevaluation on non-encyclopedic corpora
and by coveringmultiple user intentsthat better
reflect real deployments.
A related stream of research moves beyond
generic QA to scenario-driven or domain-focused
testing: RAGEval (Zhu et al., 2025) derives sce-
nario templates and uses synthetic content for sce-
nario controlled evaluation, while domain datasets
like KIQA (Yuen et al., 2025) synthesize questions
from technical sources. For enterprise settings,
DataMorgana (Filice et al., 2025) generates syn-
thetic Q&A benchmarks over private corpora with
lightweight traffic control via user-defined question
categorizations. These efforts broaden the evalu-
ation landscape, but DoRAexplicitly anchorsan-
swers to the original domain documents and em-
ploys task-aware evaluation metrics that directly
assess grounding and faithfulness to the source ma-
terial.
Synthetic pipelines such as DoRA reduce anno-
tation costs and enable controlled stress tests. Ap-
proaches include zero/few-shot prompting for QA
generation (Ye et al., 2022; Schmidt et al., 2024),
few-shot multi-hop synthesis over Wikipedia pairs
(Chen et al., 2024b), multimodal multi-hop syn-
thesis (Abaskohi et al., 2025), and modality/style-
conditioned question generation (Wu et al., 2024).
Compared to open-domain QA and retrieval bench-
marks (Kwiatkowski et al., 2019; Rajpurkar et al.,Work S E MM Seed Dom
Few-shot Multi-hop QA
(EACL’24) (Chen et al., 2024b)✓ ✗ ✗ ✓Open
FM2DS (EMNLP’25) (Abaskohi
et al., 2025)✓ ✗ ✓ ✓Open
SMMQG (EMNLP’24) (Wu et al.,
2024)✓ ✗ ✓ ✓Open
Prompting-based SDG
(LREC’24) (Schmidt et al., 2024)✓ ✗ ✗ ✗Open
RAGAS (EACL’24) (Es et al.,
2024)✗ ✓ ✗N/A Open
ARES (NAACL’24) (Saad-Falcon
et al., 2024)✗ ✓ ✗N/A Open
RAGChecker (NeurIPS’24) (Ru
et al., 2024)✗ ✓ ✗N/A Open
RAGEval (ACL’25) (Zhu et al.,
2025)✓ ✓ ✗ ✗Scenario
KIQA Gen. (arXiv’25) (Yuen et al.,
2025)✓ ✗ ✗ ✓Domain
DataMorgana (ACL’25 Ind.) (Filice
et al., 2025)✓ ✗ ✗ ✓Domain
DoRA (Ours)✓ ✓ ✗†✓Domain
Table 3: Comparison of synthetic QA generation
and RAG evaluation. S=Synthetic, E=RAG Eval-
uation, MM=Multimodal, Seed=Seed-doc grounded,
Dom=Context scope (Open / Scenario / Domain).
†Multimodality is a configurable stressor rather than
a core assumption.
2016; Yang et al., 2018; Joshi et al., 2017; Petroni
et al., 2021; Thakur et al., 2021) and existing
RAG evaluation frameworks (Es et al., 2024; Saad-
Falcon et al., 2024; Ming et al., 2024; Chen et al.,
2024a; Friel et al., 2024; Ru et al., 2024), DoRA
emphasizes three aspects: (i) grounding in domain
documents; (ii) coverage of five user-style intents
reflecting real-world deployment scenarios; and
(iii) task-aware evaluation metrics and baselines
tailored to private, domain-specific applications
(including defense). A comparison with existing
work is summarized in Tables 3 and 9.
8 Conclusion
DoRA is a domain-grounded evaluation suite and
adaptation for RAG in private, high-stakes settings,
built from defense documents with auditable evi-
dence passages and five question types that reflect
real practitioner requests. Our results show that un-
der a fixed retrieval configuration, general-purpose
LMs cluster closely, while lightweight domain
adaptation using DoRA SFT yields large gains in
both task-success metrics and groundedness diag-
nostics. We release the dataset, prompts, and a re-
producible harness to support contamination-aware
regression testing and iterative improvement of re-
trieval and generation components under domain
shift.

Acknowledgement
This work was funded by Defence Trailblazer
through an academia–industry partnership between
UNSW and Cyndr. Defence Trailblazer is a part of
the Australian Government’s Department of Edu-
cation Trailblazer Universities Program.
Limitations and Ethics
DoRA presently has: (1) a limited number of Q&A;
(2) automatic metrics that may under -approximate
semantic correctness or faithfulness under para-
phrase; and (3) evaluation results that depend on the
chosen retrieval configuration and corpus snapshot.
We are extending the dataset with more items (in-
cluding multi -modal, multi-hop) and adding larger-
scale human evaluation. Since the underlying
defense-domain documents may be sensitive or pro-
prietary, we focus on releasing generation recipes,
prompts, derived benchmark instances with prove-
nance fields, and run the pipeline on open-public
defense documents rather than private classified
defense ones.
By and large, the contribution of this work lies in
the synthetic creation of a QA dataset and its eval-
uation using (a) alternative RAG configurations
and (b) a manually curated dataset. The defense
application described in this paper is restricted to
answering questions based on a fixed set of defense
documents. The system does not make decisions,
take actions, or replace human judgment. All inter-
pretation and downstream use of its outputs remain
the responsibility of human operators and fall out-
side the scope of this work.
References
Amirhossein Abaskohi, Spandana Gella, Giuseppe
Carenini, and Issam H. Laradji. 2025. FM2DS: Few-
shot multimodal multihop data synthesis with knowl-
edge distillation for question answering. InProceed-
ings of the 2025 Conference on Empirical Methods
in Natural Language Processing.
Alibaba-NLP. 2025. mgte: Generalized text em-
bedding models for 75 languages and 8k context
length.https://huggingface.co/Alibaba-NLP/
gte-multilingual-mlm-base . Accessed 2026-01-
23.
Nicholas Carlini, Florian Tramer, Eric Wallace,
Matthew Jagielski, Ariel Herbert-V oss, Katherine
Lee, Adam Roberts, Tom Brown, Dawn Song, Úl-
far Erlingsson, Alina Oprea, and Colin Raffel. 2021.
Extracting training data from large language models.In30th USENIX Security Symposium (USENIX Secu-
rity 21), pages 2633–2650, Vancouver, BC. USENIX
Association.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024a. Benchmarking large language models in
retrieval-augmented generation. InProceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 38, pages 17754–17762.
Mingda Chen, Xilun Chen, and Wen-tau Yih. 2024b.
Few-shot data synthesis for open domain multi-hop
question answering. InProceedings of the 18th Con-
ference of the European Chapter of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 190–208, St. Julian’s, Malta. Association
for Computational Linguistics.
André V . Duarte, João DS Marques, Miguel Graça,
Miguel Freire, Lei Li, and Arlindo L. Oliveira. 2024.
LumberChunker: Long-form narrative document seg-
mentation. InFindings of the Association for Com-
putational Linguistics: EMNLP 2024, pages 6473–
6486, Miami, Florida, USA. Association for Compu-
tational Linguistics.
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2024. Ragas: Automated evalua-
tion of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158, St. Julians,
Malta. Association for Computational Linguistics.
Simone Filice, Guy Horowitz, David Carmel, Zohar
Karnin, Liane Lewin-Eytan, and Yoelle Maarek.
2025. Generating Q&A benchmarks for RAG eval-
uation in enterprise settings. InProceedings of the
63rd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 6: Industry Track), pages
469–484.
Robert Friel, Masha Belyi, and Atindriyo Sanyal.
2024. Ragbench: Explainable benchmark for
retrieval-augmented generation systems.Preprint,
arXiv:2407.11005.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey.CoRR, abs/2312.10997.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. 2021. Lora: Low-rank adap-
tation of large language models.arXiv preprint
arXiv:2106.09685.
IBM-Research. 2025. Granite embedding r2 models.
arXiv preprint arXiv:2508.21085.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of

the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611.
Matt Kapko. 2024. Meta offers llama ai models to u.s.
government for national security.CIO. Accessed
2026-01-22.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in Neural Information Processing Systems,
33:9459–9474.
Yixiao Li, Mathijs van Leeuwen, Gabriel Drevon, Mar-
lies van der Wees, Yixuan Tang, Jannis Bulian, Mattia
Di Gangi, Samuel Cahyawijaya, Md Tahmid Rah-
man Laskar, Jianguo Pan, Wenyi Zhang, Paul Michel,
Graham Neubig, and Orion Weller. 2024. An open-
source data contamination report for large language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2024, pages 5106–5124,
Miami, Florida, USA. Association for Computational
Linguistics.
Chin-Yew Lin. 2004. Rouge: A package for automatic
evaluation of summaries. InText summarization
branches out, pages 74–81.
Christopher D. Manning, Prabhakar Raghavan, and Hin-
rich Schütze. 2008.Introduction to Information Re-
trieval. Cambridge University Press, Cambridge,
UK.
Meta AI. 2024. Introducing llama 3.1: Open founda-
tion models at scale. https://ai.meta.com/blog/
meta-llama-3-1/. Accessed 2026-01-22.
Yifei Ming, Senthil Purushwalkam, Shrey Pandit,
Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong,
and Shafiq Joty. 2024. Faitheval: Can your lan-
guage model stay faithful to context, even if “the
moon is made of marshmallows”.arXiv preprint
arXiv:2410.03727.
OpenBMB. 2025. Minicpm-embedding: A bilingual
and cross-lingual text embedding model. https://
huggingface.co/openbmb/MiniCPM-Embedding .
Accessed 2026-01-23.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002. Bleu: a method for automatic evalu-
ation of machine translation. InProceedings of the
40th annual meeting of the Association for Computa-
tional Linguistics, pages 311–318.Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
Vassilis Plachouras, Tim Rocktäschel, and Sebastian
Riedel. 2021. KILT: a benchmark for knowledge
intensive language tasks. InProceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, pages 2523–2544, Online.
Association for Computational Linguistics.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100,000+ questions
for machine comprehension of text.arXiv preprint
arXiv:1606.05250.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing. Associa-
tion for Computational Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Foundations and Trends in Information Re-
trieval, 3(4):333–389.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang,
Peng Shi, Shuaichen Chang, Jiayang Cheng, Cunxi-
ang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang,
Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang,
Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024.
Ragchecker: A fine-grained framework for diag-
nosing retrieval-augmented generation. InProceed-
ings of the 38th Conference on Neural Information
Processing Systems (NeurIPS 2024), Datasets and
Benchmarks Track.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. Ares: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 338–354,
Mexico City, Mexico. Association for Computational
Linguistics.
Maximilian Schmidt, Andrea Bartezzaghi, and
Ngoc Thang Vu. 2024. Prompting-based synthetic
data generation for few-shot question answering. In
Proceedings of the 2024 Joint International Con-
ference on Computational Linguistics, Language
Resources and Evaluation (LREC-COLING 2024),
pages 13168–13178, Torino, Italia. ELRA and ICCL.
Thibault Sellam, Dipanjan Das, and Ankur Parikh. 2020.
BLEURT: Learning robust metrics for text genera-
tion.arXiv preprint arXiv:2004.04696.
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries.
Nandan Thakur, Nils Reimers, Andreas Rücklë, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. Beir:

A heterogeneous benchmark for zero-shot evalu-
ation of information retrieval models.Preprint,
arXiv:2104.08663.
Ian Wu, Sravan Jayanthi, Vijay Viswanathan, Simon
Rosenberg, Sina Khoshfetrat Pakazad, Tongshuang
Wu, and Graham Neubig. 2024. Synthetic multi-
modal question generation. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2024,
pages 12960–12993, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Pinlong Xiao and 1 others. 2024. M3-embedding:
Multi-linguality, multi-functionality, multi-
granularity text embeddings through self-knowledge
distillation.arXiv preprint arXiv:2402.03216.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380.
Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao
Feng, Zhiyong Wu, Tao Yu, and Lingpeng Kong.
2022. Zerogen: Efficient zero-shot learning via
dataset generation. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 11653–11669, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Sizhe Yuen, Ting Su, Ziyang Wang, Yali Du, and
Adam J. Sobey. 2025. Automatic dataset generation
for knowledge intensive question answering tasks.
arXiv preprint arXiv:2505.14212.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2019. Bertscore: Eval-
uating text generation with BERT.arXiv preprint
arXiv:1904.09675.
Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan,
Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang,
Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and
Maosong Sun. 2025. Rageval: Scenario specific rag
evaluation dataset generation framework. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 8520–8544, Vienna, Austria. Associa-
tion for Computational Linguistics.
Yifeng Zhu, Yiqi Liu, Jiashuo Yang, Mengzhou Jia, Min-
jie Wang, Chao Li, Jia Li, Kam-Fai Wong, and Zitao
Liu. 2024. CLEAN EV AL: Clean evaluation on con-
taminated large language models. InFindings of the
Association for Computational Linguistics: NAACL
2024, pages 791–812, Mexico City, Mexico. Associ-
ation for Computational Linguistics.

Appendix Overview
This appendix provides additional results and implementation details for reproducibility:
•Additional QA Results(Section A): QA with oracle-context (gold evidence) (Section A.1 and ta-
bles 4 and 5), and end-to-end QA with IBM Granite retrieval (Section A.2 and table 6).
•Manually Curated Dataset(Section B): expert-answered set creation and how it is used for mixed-
data SFT and as an expert check set.
•Task-Success Metrics(Section C): retrieval metrics (Hit@k, Recall@k), retriever identifiers (Ta-
ble 8), and the model identifiers used for BLEURT and BERTScore.
•RAGEval Metrics(Section D): definitions and formulas for the RAGEval diagnostics (completeness,
hallucination, irrelevance) used in our end-to-end QA evaluation.
•DoRA Dataset(Section E): dataset composition and generation statistics by style.
•Style Examples(Section F): example questions and answers for each of the five intent styles.
•Detailed Methodology(Section G): model choices (Table 13) and evidence-selection budgets
(Table 14).
•Prompts(Section H): prompt templates used for style-conditioned QA generation and filtering.

A Additional QA Results
A.1 QA with Gold Evidence Context
To isolate generation quality from retrieval errors, we also reportgold evidence contextresults where the
generator is given the recorded gold evidence passages E(oracle context) instead of retrieved passages.
Table 4 reports overall results across models, and Table 5 provides a per-style breakdown for DoRA SFT.
Results in Table 4 show that with golden context, QA results improve across models. In addition, there
is a larger gap between models (GPT-4o vs GPT-4o-mini: 45.60 vs 35.50) compared to using a fixed
retriever (with inherent retrieving errors), demonstrating that with better domain context, larger models
perform better. Nonetheless, with golden context, our DoRA SFT still shows a huge margin with models
like GPT-4o or the base Llama-3.1-8B-Instruct model.
Model / Run Tok. F1 ROUGE-L BLEU BLEURT BERTScore F1
GPT-4o 45.60 46.70 14.20 -0.088 86.92
GPT-4o-mini 35.50 36.40 9.20 -0.196 83.85
GPT-3.5-Turbo 43.80 44.70 15.30 -0.099 87.09
Claude-3-Haiku 29.30 30.30 10.40 -0.381 82.99
Mistral-7B-Instruct 59.20 58.40 38.00 -0.004 90.32
Llama3-8B-Instruct 47.10 49.00 18.00 -0.231 86.88
Llama3.1-8B-Instruct 46.30 48.10 17.00 -0.227 86.63
Qwen2.5-7B-Instruct 30.60 32.00 7.90 -0.311 82.25
DoRA SFT (Ours)70.60 71.50 57.30 0.224 93.20
Table 4: Gold evidence context QA results: the generator receives the gold evidence passages E(oracle context)
rather than retrieved passages. All metrics are percentages except BLEURT.
Style # Tok. F1 ROUGE-L BLEU BLEURT BERTScore F1
FIND421 76.74 79.12 62.39 0.352 94.72
PROVIDE155 63.98 64.02 44.85 0.060 91.17
SUMMARIZE209 70.73 71.77 51.85 0.225 93.26
EXPLAIN165 65.32 66.02 45.76 0.183 92.21
GENERATE309 68.44 67.68 50.36 0.153 92.64
Table 5: Gold evidence context results for DoRA SFT (Ours) by style. All metrics are percentages except BLEURT.
A.2 End-to-End QA with IBM Granite Retriever
We also report end-to-end QA under an alternative fixed retriever, IBM Granite (english-r2), to stress-test
robustness to retrieval variation.
Task success Faithfulness (RAGEval)
Model Tok. F1 ROUGE-L BLEU BLEURT BERTScore F1 Comp.↑Hall.↓Irr.↓
GPT-4o 30.23 30.30 7.61 -0.316 81.99 66.10 5.1128.80
Llama3.1-8B-Instruct (base) 33.73 34.14 9.53 -0.390 82.84 60.75 5.75 33.49
GPT-4o-mini 25.35 25.52 5.78 -0.361 79.94 62.03 6.38 31.60
GPT-3.5-Turbo 32.22 32.36 9.27 -0.273 83.30 62.72 4.25 33.03
Claude-3-Haiku 22.40 22.51 7.05 -0.517 79.75 60.91 6.20 32.89
Mistral-7B-Instruct 46.00 44.52 23.30 -0.223 86.44 57.59 6.22 36.19
Llama3-8B-Instruct 33.59 33.89 9.64 -0.393 82.90 60.52 5.45 34.02
Qwen2.5-7B-Instruct 21.94 22.28 5.08 -0.444 78.84 59.69 6.56 33.74
DoRA SFT (Ours)57.52 57.23 40.57 0.038 89.85 66.46 4.2129.33
Table 6: End-to-end QA with the IBM Granite retriever: task-success QA metrics (left, percentages except BLEURT)
and RAGEval-style faithfulness diagnostics (Zhu et al., 2025) (right, percentages).

In this setting (where IBM Granite is weaker than GTE on retrieval metrics as shown in Figure 2),
DoRA SFT maintains strong performance and widens the margin over representative baselines. Relative
to GPT-4o, DoRA SFT improves Tok. F1 by +27.3 percentage points (57.5 vs. 30.2), ROUGE-L by
+26.9 percentage points (57.2 vs. 30.3), and BLEU by +33.0 percentage points (40.6 vs. 7.6). Relative to
the base open model (Llama3.1-8B-Instruct), DoRA SFT improves Tok. F1 by +23.8 percentage points
(57.5 vs. 33.7) and increases completeness by +5.7 percentage points (66.5 vs. 60.8) while reducing
hallucination by -1.5 percentage points (4.2 vs. 5.8), suggesting that DoRA-based domain adaptation
helpscompensate for retrieval noisein defense-domain corpora.
B Manually Curated Dataset
We created a high-quality QA dataset using a combination of LLM-driven generation, LLM judgment,
and human expert annotation. We followed a five-step process blending human expert feedback with
LLM-generated data.
Bootstrap dataset.We asked a domain expert to generate 25 QA pairs based on and answerable with
the information in our set of domain-specific seed documents. We obtained 5 examples for each question
style:FIND,PROVIDE,GENERATE,SUMMARIZE, andEXPLAIN.
Document chunking.We chunked the seed documents into semantically coherent passages using Lum-
berChunker (Duarte et al., 2024). We used theall-MiniLM-L6-v23model via theSentence Transformers
library (Reimers and Gurevych, 2019) to calculate an embedding vector for each passage. This process
produced 2,187 non-overlapping passages from all seed documents.
Semantic graph construction.We constructed a k-NN graph (for k= 5 ) with nodes corresponding
to text passages and edges between two nodes representing semantic similarity estimated using cosine
distance between the passages’ embedding vectors. We only accepted edges between nodes when the
cosine distance was less than 0.35. The resulting directed graph had 2,187 nodes and 8,121 edges with the
average node degree of 3.7±0.9.
LLM-based dataset generation.We promptedGPT-4oto generate QA pairs for each question style
using context retrieved by querying the k-NN graph. First, we selected uniformly at random a node from
the graph. Then, depending on the question style, we extracted the ego-graph around the selected node.
ForFINDandPROVIDEstyle questions, we only returned the selected node, and for the other styles we
returned the one-hop neighborhood around the node, including the selected node itself. We then prompted
the LLM to generate QA pairs for the given question type and context. Figure 4 shows the prompt used for
this step. The prompt includes two QA examples for In-Context Learning selected uniformly at random
from the bootstrap dataset conditioned on the question style.
The generated question, answer, and context were evaluated by an LLM for quality using the prompt in
Figure 5. We usedGemini-2.5-proas the judge. We asked the LLM to rate the quality of the response
using a score in the range [0,1] where 0 is worse and 1 is best. We accepted generated data that were
judged with a score higher than 0.75. This process resulted in 100 QA pairs with context for 20 examples
corresponding to each question style.
Human annotation.Finally, a domain expert was shown the generated questions and context and asked
to provide an answer. Note that we did not show the generated answer to the annotator. Of the 100
generated examples, the annotator deemed 86 as valid and provided answers derived from the provided
passages. The combination of LLM-generated questions and human-generated answers, along with the
relevant context comprise the final dataset.
B.1 High-precision Supervision for Domain Adaptation
To use the human-annotated set as high-precision supervision during domain adaptation, we construct
a mixed SFT training set that combines synthesized DoRA instances with manually curated examples.
3https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

You are a helpful assistant tasked with creating a single-turn question answering dataset.
Your task is to generate one question-answer (QA) pair based on the provided text passages.
**Instructions:**
- The question must be answerable using only the information in the passages.
- The question type must match the specified type below.
- Use the passage references ([1], [2], etc.) to ground your question and answer.
- Do not use external knowledge or make assumptions beyond the passages.
- Ensure the answer is concise, factually correct, and directly supported by the text.
**Question Type:** {question_type}
**Examples:**
{examples}
**Passages:**
{passages}
**Output Format (JSON):**
{{
{{
"question": "Your generated question here",
"answer": "Your generated answer here"
}}
}}
Figure 4: Prompt template used for generating QA pairs with In-Context Learning.
We split the 86 human-annotated items into 52 for training and 34 for held-out testing to avoid train/test
leakage. Because 52 manual training examples are too few to strongly influence optimization when
mixed with thousands of synthetic instances, we upsample the manual subset via repeated sampling
(i.e., sampling with replacement) to 2,207 effective manual training instances. We then mix these with
the 5,150 synthesized DoRA SFT instances, yielding a 7,357-instance training set, corresponding to an
approximately 70/30 synthesized-to-human ratio. This weighting retains broad synthetic coverage across
intents while substantially increasing exposure to human-quality answers grounded in the same private
seed documents. We then resume from our previous checkpoint of SFT on DoRA synthetic data, and train
for a further 600 steps with the same configuration on this mixed dataset to achieve the DoRA SFT with
human-annotated supervision.
Results in Table 7 indicate that incorporating only a small amount of human-annotated data (52 samples)
into DoRA SFT improves answer quality, aligns responses more closely with domain-expert style, and
surpasses general-purpose LLMs, including GPT-4o. Meanwhile, the synthetic-only DoRA SFT retains
stronger semantic similarity performance (BLEURT and BERTScore-F1). The comparatively lower token-
level scores are attributable to a stylistic shift in the human-annotated data, which contains substantially
longer answers (87 tokens on average vs. 28 for synthetic data).
Model / Run Tok. F1 ROUGE-L BLEU BLEURT BERTScore F1
GPT-4o 25.19 23.61 5.90 -0.793 71.05
Llama3.i1-8B-Instruct (base) 25.27 23.50 6.62 -0.827 70.98
DoRA SFT 22.96 23.07 2.00-0.565 72.26
DoRA SFT with human-annotated supervision31.65 29.28 6.78 -0.400 75.88
Table 7: QA results with GTE-based retrieval on held-outexpert-curatedtest set. All metrics are percentages except
BLEURT.

You are an expert evaluator tasked with assessing the quality of a question-answer (QA) pair
generated from a set of passages. Your evaluation should determine how well the QA pair aligns
with the intended question type and whether it is accurate and contextually grounded.
Please assess the QA pair based on the following criteria:
1. **Question Type Alignment**: Does the question clearly reflect the specified question
type (e.g., Find, Explain, Summarise)?
2. **Contextual Answerability**: Can the question be answered using only the information provided
in the passages?
3. **Answer Accuracy**: Is the answer factually correct and directly supported by the passages?
Provide a brief reasoning for your evaluation and assign a **quality score between 0.0 (poor)
and 1.0 (excellent)**.
**Question Type:** {question_type}
**Passages:**
{passages}
**Question:** {question}
**Answer:** {answer}
**Output Format (JSON):**
{{
{{
"reasoning": "Your reasoning about the evaluation and score",
"score": 0.0 to 1.0
}}
}}
Figure 5: Prompt template used for judging the quality of generated question and answers conditioned on the
question style.
C Task-Success Metrics
For each benchmark instance i, we treat the question qias the retrieval query, and the gold evidence
passages recorded in DoRA as the set of relevant items Gi(one or more chunk IDs). A retriever produces
a ranked list of retrieved passages Rk(qi) = (d i,1, di,2, . . . , d i,k), where di,jis the chunk ID at rank jfor
queryq i, andkis the evaluation cutoff (e.g.,k= 3in our QA study).
Hit@k.LetIbe the set of evaluated (e.g. DoRA test set). We define the instance-level indicator
Hit@k i=1[∃j∈ {1, . . . , k}:d i,j∈G i],
and report the average Hit@k=1
|I|P
i∈IHit@k i. Intuitively, Hit@k measures whether the retriever
returnsanygold evidence chunk in its top-klist forq i.
Recall@k.When multiple evidence passages are marked relevant, we measure coverage of the gold set:
Recall@k i={di,1, . . . , d i,k} ∩G i
|Gi|,
and report Recall@k=1
|I|P
i∈IRecall@k i. This measures how completely the top- kretrieved chunks
cover the gold evidence set (with set intersection preventing repeated retrieval of the same chunk from
inflating the score).
Details of retrievers are shown in Table 8.
Task-success metric models.For reproducibility of learned task-success scores, we compute BLEURT
usingbleurt-base-128and BERTScore F1usingdistilbert-base-uncasedas the encoder.

Retriever Model / Implementation ID
BM25 (Robertson and Zaragoza, 2009) rank-bm25
GTE-multilingual-Base (Alibaba-NLP, 2025) Alibaba-NLP/gte-multilingual-base
MiniCPM-Embedding (OpenBMB, 2025) openbmb/MiniCPM-Embedding
BGE-M3 (Xiao et al., 2024) BAAI/bge-m3
IBM Granite (english-r2) (IBM-Research, 2025) ibm-granite/granite-embedding-english-r2
Table 8: Retriever identifiers for reproducibility.
Work S E MH MM Scn Sty Seed Notes
Few-shot Data Synthesis for
Open-Domain Multi-hop QA
(EACL’24) (Chen et al., 2024b)✓ ✗ ✓ ✗Open✗ ✓ Few-shot pipeline synthesizing multi-hop QA from Wikipedia docu-
ment pairs; improves smaller LMs on Hotpot -style tasks; no dedicated
RAG evaluation metrics.
FM2DS (EMNLP’25) (Abaskohi
et al., 2025)✓ ✗ ✓ ✓Open✗ ✓ Few-shot multimodal, multihop synthesis grounded in multimodal
Wikipedia pages; introduces a multimodal multi-hop benchmark over
long docs.
SMMQG (Findings
EMNLP’24) (Wu et al., 2024)✓ ✗ ✗ ✓Open✓ ✓ Generates modality/style -conditioned multimodal questions grounded
in Wikipedia documents to stress MMRAG; not explicitly multi-hop.
Prompting -based SDG
(LREC -COLING’24) (Schmidt
et al., 2024)✓ ✗ ✗ ✗Open✗ ✗ Prompting -based synthetic QA generation for few -shot settings; not
grounded to specific seed docs.
RAGAS (EACL’24) (Es et al.,
2024)✗ ✓ ✗ ✗Open N/A N/A Reference -free RAG metrics (faithfulness, answer/context relevance);
framework only.
ARES (NAACL’24) (Saad-Falcon
et al., 2024)✗ ✓ ✗ ✗Open N/A N/A Automated evaluation framework with faithfulness/attribution metrics
and diagnostics.
RAGChecker (NeurIPS’24
D&B) (Ru et al., 2024)✗ ✓ ✗ ✗Open N/A N/A Fine-grained diagnostic framework for RAG (answerability, support,
hallucination).
RAGEval (ACL’25) (Zhu et al.,
2025)✓ ✓ ✗ ✗Scn.✗ ✗ Scenario templates derived from seed docs, but QA/eval built over
generated articles/configs rather than directly over the original seed
docs.
Automatic Dataset Generation
for KIQA (arXiv’25) (Yuen et al.,
2025)✓ ✗ ✗ ✗Domain✗ ✓ Generates domain QA from technical sources (e.g., manuals/specs).
Emphasis on creating fine-tuning data rather than defining an evalua-
tion framework.
DataMorgana (ACL’25 Ind.) (Filice
et al., 2025)✓ ✗ ✗ ✗Domain✗ ✓ Enterprise benchmark synthesis over private corpora with user-
defined question categories to control traffic; emphasizes diversity
and question–passage fit, leaving evidence-grounded answer scoring
largely out of scope.
Ours (DoRA)✓ ✓ ✗†✗†Domain✓ ✓ Domain -oriented, contamination -aware evaluation with
style -conditioned synthetic Q&A grounded in domain sources;
task-aware scoring and baselines.
Table 9: Comparison across synthetic data generation, RAG evaluation, realistic context dimensions, and whether
question generation is grounded in seed documents. Abbrev.: S=Synthetic, E=RAG Evaluation, MH=Multi -hop,
MM=Multimodal, Scn=Scenario, Sty=Style, Seed=Seed -doc. N/A marks evaluation frameworks rather than
data-generation methods.†Multi -hop, Multimodal are configurable (planned stressors) rather than a core require-
ment.
D RAGEval Metrics
We compute the end-to-end faithfulness diagnostics using the open-source RAGEval evaluation imple-
mentation4from Zhu et al. (2025), with gpt-4o-mini as the rubric judge (prompt version v1). For each
instance i, RAGEval extracts a set of reference keypoints Ki={k i,1, . . . , k i,Ni}from the gold answer ai
and evaluates a system answerˆa iby classifying each keypoint into one of three categories:
•Ki,comp : keypoints judged ascomplete(covered correctly byˆa i);
•Ki,irr: keypoints judged asirrelevant(missing/unaddressed byˆa i); and
•Ki,hall: keypoints judged ashallucinated/wrong(addressed incorrectly byˆa i).
Instance-level scores.The metrics are defined as normalized keypoint counts:
RAG-Compi=|Ki,comp|
|Ki|,RAG-Hall i=|Ki,hall|
|Ki|,RAG-Irr i=|Ki,irr|
|Ki|.
4https://github.com/OpenBMB/RAGEval/tree/main/rageval/evaluation

Task Samples % Avg Q Len Avg A Len Refs/Sample Duration (s) Docs
EXPLAIN490 9.5% 13.49 37.21 2.70 11.68 20
FIND1730 33.6% 12.76 21.61 1.00 4.76 19
GENERATE1628 31.6% 13.80 33.36 3.18 13.56 20
PROVIDE563 10.9% 14.10 28.87 1.84 12.72 19
SUMMARIZE739 14.3% 13.07 31.25 2.41 9.75 20
Total/Avg 5150100.0% 13.35 28.99 2.14 9.78∼20
Table 10: Train dataset composition and basic statistics by style. “Refs/Sample” counts linked evidence passages.
Task Samples % Avg Q Len Avg A Len Refs/Sample Duration (s) Docs
EXPLAIN165 13.1% 13.67 33.59 2.99 14.06 18
FIND421 33.4% 12.65 20.83 1.00 4.79 20
GENERATE309 24.5% 13.76 31.42 3.34 14.31 20
PROVIDE155 12.3% 14.00 30.45 1.91 12.37 19
SUMMARIZE209 16.6% 12.85 29.35 2.65 11.29 18
Total/Avg 1259100.0% 13.44 29.53 2.38 11.76∼19
Table 11: Dataset composition and basic statistics by style. “Refs/Sample” counts linked evidence passages.
All three scores lie in [0,1] ; higher completeness is better, while lower hallucination and irrelevance are
better.
Dataset-level reporting.We report micro-averages over instances:
RAG-Metric =1
|I|X
i∈IRAG-Metric i,
whereIis the evaluated set (e.g., the 1,259 test split from DoRA dataset).
E DoRA Dataset
Details regarding the train/test set used to benchmark different LLMs in defense domain and its generation
details are shown in Tables 10 and 11 accordingly.
F Example Questions and Answers by Style
We use five intent styles to approximate deployment traffic in defense settings:
•FIND: pinpointing exact data points (e.g., deployment dates, cost figures);
•EXPLAIN: clarifying technical distinctions critical to planning and risk assessment;
•SUMMARIZE: condensing lengthy reports into concise, actionable overviews;
•GENERATE: producing structured lists of capabilities, gaps, or priorities at a glance; and
•PROVIDE: surfacing key quantitative metrics (budgets, procurement data).
Examples for each style are shown in Table 12.
G Detailed Methodology
Core model choices.Table 13 summarizes the default model choices by pipeline stage.
Tokenization and chunking.We tokenize with cl100k_base . Chunks are built by packing paragraphs
to a target budget of 512 tokens (min 100; max 1024). Oversized chunks trigger a sentence-level sliding-
window fallback with ≈25% sentence overlap. Each chunk retains provenance (document id/name/path,
page numbers, optional section header).

Style Example question Example answer
FIND What capabilities does the Research Technology and Operations provide
to Defense IS&T?Research Technology and Operations provides specialist technical and
business capabilities to Defense IS&T.
EXPLAIN How is the concept of ‘asymmetric advantage’ defined in a military
context?In a military context, ‘asymmetric advantage’ refers to military capa-
bilities that pit strength against weakness, often in a non-traditional or
unconventional manner, and that disrupt a potential adversary’s decision
calculus.
SUMMARIZE What initiatives are being continued and enhanced to support the growth
of the defense industry workforce?The Schools Pathways Program, Defense Industry Internship Program,
and Defense Industry Pathways Program will continue and be enhanced
to achieve better outcomes.
GENERATE What measures will be taken to provide the industry with clearer infor-
mation about Defense’s needs?The industry will receive clearer and more detailed information about
Defense’s needs through more classified briefings, the establishment of a
tri-partite defense industry council, and industry forums.
PROVIDEWhat was the total value of contracts awarded by Defense in 2022–23? In 2022–23, Defense awarded over38billion in contracts.
Table 12: Illustrative examples of the five DoRA intent styles. Examples are shown for readability and to convey the
expected output format.
Stage Model / setting
Q&A generation (LLM)openai/gpt-4o(temp 0.3, max 512 new tokens)
Retrieval encoder (construction)sentence-transformers/all-mpnet-base-v2
Retrieval encoder (retrieval-conditioned runs)Alibaba-NLP/gte-multilingual-base
Evidence selection (NLI prefilter)facebook/bart-large-mnli
Evidence selection (local scorer)google/gemma-2-9b-it(8-bit)
Faithfulness: entailment (DES)microsoft/deberta-large-mnli
Faithfulness: extractive span QAdeepset/roberta-base-squad2
Faithfulness: embedding similaritysentence-transformers/all-MiniLM-L6-v2
Table 13: Default model choices by stage.
Retrieval.We use hybrid dense+BM25 retrieval with cosine similarity (L2-normalized embeddings,
inner-product search) and score mixing (0.7 dense / 0.3 BM25). For retrieval-conditioned benchmark runs,
we fix chunk size to 320 tokens with an overlap setting of 64 and use top-k=3.
Evidence selection and budgets.Evidence selection is conditioned on style s∈
{FIND,EXPLAIN,SUMMARIZE,GENERATE,PROVIDE} . We first retrieve candidates, then apply an NLI
prefilter K, and finally score candidatecombinationsfor completeness/complementarity/coherence/task
fit. The per-style evidence budgets and selection hyperparameters are summarized in Table 14; when
multi-chunk synthesis is enabled, we require at least two distinct evidence chunks for long-form styles.
Styles|E|PrefilterKCombos Notes
FIND1 20 6 Extractive
PROVIDE2 30 7 Numeric content required
EXPLAIN4 60 8 Long-form reasoning
SUMMARIZE4 40 8 Coverage-oriented
GENERATE5 50 7 List/category synthesis
Table 14: Evidence-selection budgets. |E|is the target number of evidence chunks in the selected bundle. Prefilter
Kis the NLI top- Kcandidate pool size used before combination scoring. “Combos” is the maximum number
of candidate chunk combinations evaluated per instance. For multi-chunk, we require at least 2 distinct evidence
chunks.
Construction-time filtering.We over-generate (2 ×target size), remove exact duplicate questions
per seed document, and apply task-aware quality control. Hard gates enforce non-empty fields and
numeric content forPROVIDE; for multi-chunk, accepted items must draw from at least two distinct
evidence chunks (the prompt requests explicit chunk citations). Evidence-grounded scoring uses DES,
span alignment, numeric consistency, and semantic relevance/coverage with the models listed above. We
stratify candidates by style and select the highest-quality items under per-style quota.

H Prompts
H.1 Multi-Chunk Synthesis Directives
For multi-chunk synthesis, we append the following directives (with configuration-dependent values, e.g.,
minimum chunks/citations) to each style prompt:
MULTI-CHUNK SYNTHESIS MODE (ENFORCED):
- Integrate evidence from at least 2 distinct [Chunk] sections when they offer relevant details.
- Cite each supporting chunk explicitly by number or by quoting a short anchor
phrase (minimum 2 citations).
- Merge overlapping insights into one cohesive answer rather than separate mini-answers.
- Reiterate key facts at the start of the answer so earlier chunks are not ignored.
- Answers lacking multi-chunk grounding will be rejected by downstream verification.
H.2 Style-Conditioned Q&A Generation Prompts
We generate each candidate instance by prompting an instruction-tuned LLM with an evidence context
and a style-specific template. In all prompts below, {context} is replaced with the selected evidence
passages (concatenated). Our default dataset-construction runs useopenai/gpt-4o.
FIND.
Based on the provided context, generate a factual question-answer pair that asks
for specific information (dates, numbers, names, locations, etc.) that can be
found in the text.
Context:
{context}
Requirements:
- Question should ask for specific factual information
- Answer should be a short, precise factual response directly from the text
- Copy the shortest exact phrase that answers the question (verbatim) from the supporting sentence
- If needed, include a brief quoted clause that contains the fact
- Answer must be declarative and must not be phrased as a question; do not include a question mark
- The answer must exist exactly in the provided context
- Focus on concrete details like dates, amounts, names, locations, or specific facts
- Prefer interrogatives: What/When/Where/Who/Which/How many/How much (avoid standalone'How')
- If multiple values appear, choose the one that most directly answers the question
- Use only information present in the Context; do not add external knowledge
- Do not refer to the document/text/section; produce a self-contained Q&A
IMPORTANT - DO NOT generate questions about:
- Page numbers, indexes, or table of contents
- Who is mentioned/listed in the document
- Where information appears in the document structure
- Document metadata or navigation elements
Format your response as:
Question: [Your question here]
Answer: [Your answer here]
Example:
Question: What is the budget allocation mentioned for defense procurement?
Answer: $2.4 billion

EXPLAIN.
Based on the provided context, generate a question that asks for an explanation
or clarification of a concept, distinction, or technical detail mentioned in
the text.
Context:
{context}
Requirements:
- Question should ask for an explanation, clarification, or distinction
(prefer'How. . .'or'Why. . .'unless a clear'What is the difference. . .'is present)
- Answer must be 2–3 sentences:
•Sentence 1: direct explanation (what/how/why) using terms from the Context.
•Sentence 2–3: tie the explanation to one or two specific supporting clauses
(quote 2–8 words) from the Context.
- Ground the explanation in specific source sentences; avoid generic or speculative claims
- Use only information present in the Context; do not add background not present
- Where helpful, use short phrases from the source for key terms
- Do not refer to the document/text/section; produce a self-contained Q&A
Format your response as:
Question: [Your question here]
Answer: [Your answer here]
Example:
Question: What is the difference between operational capability and strategic capacity?
Answer: Operational capability refers to the immediate ability to conduct
specific military operations, while strategic capacity represents the long-term
potential to sustain and expand operations over time.
SUMMARIZE.
Based on the provided context, generate a question that asks for a summary or
overview of the key points, and provide a concise summary as the answer.
Context:
{context}
Requirements:
- Question should ask for a summary, overview, or key points
- Answer should be a comprehensive but concise summary (2-4 sentences)
- Include the most important information from the context
- Use specific names, figures, and terms where they appear (avoid vague phrasing)
- Structure the summary logically and clearly
- Include at least one concrete figure or named program if present
- Cover what is being done (action), why (rationale), and scale (numbers/timeframes) when present
- Avoid speculation or recommendations; report only what is explicitly stated
- Use only information present in the Context; do not add external knowledge
- Do not refer to the document/text/section; produce a self-contained Q&A
Format your response as:
Question: [Your question here]
Answer: [Your answer here]
Example:
Question: What are the key points of the cybersecurity strategy outlined in this section?
Answer: The cybersecurity strategy focuses on three main areas: strengthening
defensive capabilities through advanced threat detection systems, building
workforce expertise through specialized training programs, and enhancing
international cooperation through information sharing agreements.
The strategy emphasizes a proactive approach to cyber threats with increased
investment in both technology and human resources.

PROVIDE.
Based on the provided context, generate a question that asks for specific
quantitative data (budgets, numbers, percentages, dates) and provide the
precise numerical answer.
Context:
{context}
Requirements:
- Question should ask for specific quantitative information
- Answer should copy the exact numeric phrase from the context (including units)
- If the number appears within a sentence, quote the key clause containing it
- Include units where appropriate (dollars, percentages, dates, quantities)
- The answer must be factually accurate and directly from the text
- NO HEDGING: Do not use words like "likely", "may", "could",
"approximately", "suggests". Provide definitive figures only as stated in
the context.
- NO INFERENCE: Do not infer or estimate numbers; only copy what is explicitly present.
- If a range is shown and the question asks for total/planned/overall, copy
the range exactly (e.g., "$5.9bn – $7.9bn"). If the question asks for
'approved'vs'unapproved', pick the specific column value.
- If no numeric value exists for your question, select a different
quantitative question that can be answered exactly from the Context.
- Use only information present in the Context; do not add external knowledge
- Do not refer to the document/text/section; produce a self-contained Q&A
IMPORTANT - DO NOT generate questions about:
- Page numbers or document navigation
- Index entries or table of contents data
- Which page or section contains information
- Document structure or metadata
Format your response as:
Question: [Your question here]
Answer: [Your answer here]
Example:
Question: What is the total budget allocation for procurement mentioned in the report?
Answer: $1.2 billion

GENERATE.
Based on the provided context, generate a question that asks for a structured
list (capabilities, priorities, requirements, etc.) and provide the answer as a
numbered list.
Context:
{context}
Requirements:
- Question should ask for a list of items (capabilities, priorities, gaps, requirements, etc.)
- Answer should be a numbered list with 3-6 items
- Each item must be grounded in a single supporting sentence from the Context
- Include a short quoted anchor (2–6 words) from that sentence for each item
- Do not merge information from multiple sentences into one item
- Use phrases directly from the source sentence (verbatim or near-verbatim)
- Do not introduce items that are not explicitly stated in the Context
- Items should be substantial noun phrases (avoid weak forms like'there is/are')
- Use only information present in the Context; do not add external knowledge
- Do not refer to the document/text/section; produce a self-contained Q&A
Format your response as:
Question: [Your question here]
Answer: [Your numbered list here]
Example:
Question: What are the key capabilities mentioned for the new defense system?
Answer: 1. Advanced threat detection using AI-powered surveillance systems
2. Rapid response coordination through integrated command networks
3. Multi-domain operational flexibility across land, sea, and air environments
4. Enhanced cybersecurity protection for critical infrastructure