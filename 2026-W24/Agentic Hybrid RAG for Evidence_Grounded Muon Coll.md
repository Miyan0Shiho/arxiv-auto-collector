# Agentic Hybrid RAG for Evidence-Grounded Muon Collider Analysis

**Authors**: Ruobing Jiang, Dawei Fu, Cheng Jiang, Tianyi Yang, Zijian Wang, Youpeng Wu, Yong Ban, Yajun Mao, Qiang Li

**Published**: 2026-06-09 03:42:50

**PDF URL**: [https://arxiv.org/pdf/2606.10381v1](https://arxiv.org/pdf/2606.10381v1)

## Abstract
Muon collider research spans accelerator physics, detector instrumentation, and high-energy phenomenology, with relevant evidence scattered across a rapidly expanding and heterogeneous body of scientific literature. As high-energy physics (HEP) increasingly explores agent-assisted analysis workflows, efficiently locating, integrating, and verifying scientific evidence becomes an essential capability. While retrieval-augmented generation (RAG) offers a promising framework for scientific question answering, integrating agentic reasoning without compromising retrieval precision remains a key challenge. In this work, we present agentic hybrid RAG, an evidence-grounded RAG framework for muon collider research. The framework combines a hybrid retriever, integrating sparse lexical and dense semantic retrieval, with an agentic reasoning module for query decomposition, evidence expansion, and grounded answer generation. To enable systematic evaluation, we construct the first benchmark for retrieval-augmented scientific question answering in the muon collider domain, comprising a curated literature corpus together with dedicated retrieval and answer-generation benchmarks covering major detector and physics research topics. Extensive evaluation shows that hybrid retrieval provides the strongest retrieval backbone, while agentic reasoning is most effective for controlled evidence expansion and answer synthesis. Built on this principle, agentic hybrid RAG consistently outperforms representative retrieval and RAG baselines in retrieval effectiveness, answer quality, evidence coverage, and factual grounding. Together, the benchmark and framework provide a foundation for evidence-grounded scientific question answering and future HEP analysis agents operating over large-scale scientific literature.

## Full Text


<!-- PDF content starts -->

Agentic Hybrid RAG for Evidence-Grounded Muon
Collider Analysis
Ruobing Jiang1, Dawei Fu1∗, Cheng Jiang2, Tianyi Yang1, Zijian Wang1, Youpeng Wu1,
Yong Ban1, Yajun Mao1, Qiang Li1 1
1State Key Laboratory of Nuclear Physics and Technology, Peking University, China
2School of Physics and Astronomy, University of Edinburgh, UK
∗E-mail: fudw@pku.edu.cn
Abstract
Muon collider research spans accelerator physics, detector instrumentation, and high-
energy phenomenology, with relevant evidence scattered across a rapidly expanding and
heterogeneous body of scientific literature. As high-energy physics (HEP) increasingly
explores agent-assisted analysis workflows, efficiently locating, integrating, and verifying
scientific evidence becomes an essential capability. While retrieval-augmented generation
(RAG) offers a promising framework for scientific question answering, integrating agentic
reasoning without compromising retrieval precision remains a key challenge. In this work, we
present agentic hybrid RAG, an evidence-grounded RAG framework for muon collider research.
The framework combines a hybrid retriever, integrating sparse lexical and dense semantic
retrieval, with an agentic reasoning module for query decomposition, evidence expansion,
and grounded answer generation. To enable systematic evaluation, we construct the first
benchmark for retrieval-augmented scientific question answering in the muon collider domain,
comprising a curated literature corpus together with dedicated retrieval and answer-generation
benchmarks covering major detector and physics research topics. Extensive evaluation shows
that hybrid retrieval provides the strongest retrieval backbone, while agentic reasoning is
most effective for controlled evidence expansion and answer synthesis. Built on this principle,
agentic hybrid RAG consistently outperforms representative retrieval and RAG baselines in
retrieval effectiveness, answer quality, evidence coverage, and factual grounding. Together,
the benchmark and framework provide a foundation for evidence-grounded scientific question
answering and future HEP analysis agents operating over large-scale scientific literature.
Keywords:Retrieval-Augmented Generation, Agentic RAG, Hybrid Retrieval, Muon Collider,
Scientific Question Answering
*Corresponding author.
The code and data will be released upon publication.
1arXiv:2606.10381v1  [hep-ex]  9 Jun 2026

Contents
1 Introduction 2
2 Background and Motivation 3
2.1 RAG for HEP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2 Promoting Muon Collider Analysis with RAG . . . . . . . . . . . . . . . . . . . . 4
3 Methodology 6
3.1 Hybrid Retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.2 Agentic Query Decomposition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.3 Evidence-Grounded Answer Generation . . . . . . . . . . . . . . . . . . . . . . . 8
3.4 System Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
4 Experiments 9
4.1 Experimental Setup and Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.2 Hybrid Retriever Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
4.3 Retrieval Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.4 Answer Generation Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
5 Conclusion and Outlook 14
Appendix 18
A Query Decomposition Prompt Templates 18
B Retrieval Metrics Formulation 20
C Answer Metrics Formulation 21
D Reproducibility Commands 22
1 Introduction
As AI agents become increasingly explored for HEP research and analysis workflows [ 1,2], there
is a growing need for systems that can reliably access, retrieve, and utilize scientific literature.
Analysis decisions rarely depend on isolated facts but instead on evidence distributed across a
large and rapidly evolving body of publications, making literature retrieval and interpretation a
central capability of next-generation HEP analysis agents.
Building such agents is non-trivial: they must support reliable evidence extraction and
synthesis over long-tail and rapidly evolving knowledge, where relevant information is often
fragmented across multiple papers and subfields. Large language models (LLMs) have been
widely explored for this purpose but remain limited in producing faithful, evidence-grounded
outputs without explicit external grounding [ 3,4]. Retrieval-augmented generation (RAG)
addresses this by grounding generation in external corpora, reducing reliance on parametric
memory and costly model retraining [4].
Effective retrieval for HEP literature requires integrating complementary signals. Dense
semantic retrieval can effectively capture semantic similarity and match paraphrased queries [ 5,6],
but it may overlook exact acronyms, mathematical symbols, and process names that are prevalent
in HEP. Sparse lexical retrieval methods, exemplified by Okapi Best Matching 25 (BM25) [ 7],
provide robust keyword-level matching and are particularly effective for terminology-sensitive
queries. Hybrid strategies combining these signals via reciprocal rank fusion (RRF) [ 8] consistently
improve robustness across diverse query types. Recent agentic RAG extensions incorporate
2

query decomposition and routing [ 9], but excessive agentic exploration can introduce retrieval
drift, motivating architectures that balance hybrid retrieval with lightweight agentic reasoning.
Muon collider research provides a concrete and challenging testbed for such systems. It
spans accelerator physics, detector instrumentation, and high-energy phenomenology, drawing on
beam dynamics, machine-detector interfaces (MDI), beam-induced backgrounds (BIB), detector
design, and physics analyses. The field is advancing rapidly: following renewed international
interest since the early 2020s, the volume of technical reports, conference proceedings, and
preprints has grown substantially, with contributions distributed across accelerator, detector,
and phenomenology communities [ 10,11]. Information relevant to a single question is often
distributed across multiple papers and subfields: understanding a detector-design choice, for
instance, may require connecting evidence from background studies, shielding reports, and
performance analyses. Effective scientific assistants must therefore go beyond document retrieval
and support evidence-grounded question answering, which motivates RAG systems that integrate
high-quality retrieval, multi-step reasoning, and traceable answer generation.
In this work, we develop and evaluate an adaptive RAG agent over a muon collider scientific
corpus. We employ a hybrid retriever combining dense semantic retrieval [ 5,6], sparse BM25,
and FAISS indexing [ 12], fused via RRF. On top of this retrieval backbone, a lightweight agent
performs query decomposition and follow-up query generation to recover evidence missed by
initial retrieval, while maintaining the fidelity and traceability required for scientific question
answering in muon collider research.
The key contributions of this work are summarized as follows:
•We construct a comprehensive muon collider literature corpus and RAG benchmark, com-
prising a retrieval benchmark with expert-curated relevance annotations and an answer-level
benchmark with reference answers, required key points, and unanswerable questions.
•We propose an agentic hybrid RAG framework combining hybrid retrieval, agentic query
decomposition, evidence expansion, and evidence-grounded answer generation, designed to
maintain traceability between generated claims and supporting literature evidence.
•We conduct comprehensive evaluations across retrieval and answer generation tasks, demon-
strating that agentic hybrid RAG consistently outperforms representative baselines in retrieval
effectiveness, answer quality, evidence coverage, and factual grounding. We find that hybrid
retrieval and controlled evidence expansion are the primary drivers of these gains, validating
the evidence-aware design of the framework.
•We establish an end-to-end workflow spanning corpus construction, benchmark development,
retrieval, reasoning, and answer generation, illustrating how evidence-aware RAG systems
can support future muon collider studies and HEP analysis agents.
2 Background and Motivation
2.1 RAG for HEP
RAG integrates user queries with an external knowledge base by coupling two core components:
a retrieval module and a generation module. The retrieval module selects relevant documents
from the external corpus based on the input query, aiming to surface evidence most pertinent to
the user’s information need. The generation module then conditions on the retrieved content to
produce coherent and contextually grounded responses, leveraging a language model to synthesize
information from the provided evidence.
In canonical RAG formulations, retrieval typically returns a fixed number of documents
ranked by relevance to the query, and the generation process is constrained to rely primarily on
this retrieved context. A widely used instantiation of this paradigm is dense retrieval with text
3

embeddings, where documents are retrieved based on proximity to the query in a vector space,
with similarity serving as a proxy for semantic relevance [ 13]. Although alternative retrieval
strategies exist, this embedding-based formulation is often referred to as naive or standard
vector-based RAG, and serves as a common baseline in many retrieval-augmented systems.
RAG-based question answering systems are increasingly being developed for applications in
nuclear and high-energy physics. Recent work has explored retrieval and question answering
over Electron–Ion Collider (EIC) literature [ 14,15], as well as within large LHC experiments
such as LHCb [ 16] and CMS [ 17]. These efforts are supported by a broader ecosystem of
scientific text-mining and information infrastructure, including INSPIRE, S2ORC, SciBERT,
and GROBID [ 18–20]. Collectively, they demonstrate the feasibility of applying retrieval-
augmented methods to domain-specific scientific corpora, where relevant knowledge is distributed
across large, heterogeneous, and rapidly evolving collections of literature.
Meanwhile, these studies also highlight limitations in current evaluation practices for scientific
question answering. While general-purpose metrics and frameworks such as RAGAS, ROUGE,
and classical information-retrieval measures [ 21–24] provide useful proxies for retrieval and
generation quality, they do not fully capture the requirements of scientific settings. In particular,
scientific question answering requires fine-grained assessment of evidence grounding, chunk-level
retrieval correctness, and the ability to abstain when supporting evidence is absent, all of which
are only partially reflected by standard automatic metrics. To address these requirements, we
construct a dedicated evaluation framework that combines classical retrieval metrics with LLM-
as-a-judge assessment for answer quality. The benchmark design and evaluation are described in
Section 4, with detailed metric definitions provided in Appendix B and C.
2.2 Promoting Muon Collider Analysis with RAG
Muon colliders have re-emerged as a compelling option for future energy-frontier particle physics
because they combine a lepton-collider initial state with access to multi-TeV center-of-mass
energies. Compared with electrons, muons are heavier by a factor of about 200, which strongly
suppresses synchrotron radiation and makes circular lepton-collider concepts feasible at energies
where electron rings become impractical. As a result, the physics program is qualitatively distinct
from both hadron and lower-energy lepton colliders, offering a clean initial state, high partonic
center-of-mass energy, and sensitivity to both direct production of new particles and precision
deviations from the Standard Model [ 25–31]. The physics case spans Higgs, electroweak, top,
flavour, and beyond-the-Standard-Model studies, with particular emphasis on Higgs precision
measurements and high-energy vector-boson fusion and multi-boson processes [32–37].
The same features that enable this physics reach also introduce significant accelerator and
detector challenges. Muon decays along the beam line produce intense beam-induced backgrounds,
generating large fluxes of secondary particles with broad spatial and temporal distributions.
These backgrounds increase detector occupancies, contaminate calorimeter energy deposits,
and degrade tracking and reconstruction performance. Consequently, the machine-detector
interface becomes a central design element rather than a peripheral engineering consideration.
Mitigation strategies include dedicated shielding, optimized interaction-region design, precise
timing, high-granularity detectors, and reconstruction algorithms capable of rejecting out-of-time
activity [28, 38–41].
Beyond its physics motivation and detector challenges, the muon collider domain provides
a useful testbed for retrieval-augmented generation systems in scientific research. Unlike
established high-energy physics subfields, it lacks mature AI-assisted literature navigation tools
and standardized benchmarks for information retrieval and synthesis. At the same time, relevant
knowledge is distributed across accelerator design, beam-induced background studies, detector
performance, and physics analyses, making it difficult to answer focused questions without cross-
document evidence aggregation. This combination makes it well-suited for evaluating whether
RAG systems can perform fine-grained retrieval, multi-source synthesis, and evidence-grounded
4

reasoning under realistic scientific conditions.
To characterize the structure of information needs in this domain, we define representative
application scenarios for our agentic hybrid RAG framework in Table 1. These scenarios span
accelerator design and beam performance, machine–detector interface studies, beam-induced
background mitigation, Higgs and electroweak measurements, beyond-the-Standard-Model
searches, multi-boson processes, and answerability checking.
Table 1: Representative RAG application scenarios and query intents for muon collider analysis.
Query category User intent Retrieval objective
Physics motivation Understand the motivation for
muon colliders in future collider
programs.Retrieve high-level collider com-
parisons and physics case discus-
sions.
Beam cooling and
beam qualityUnderstand cooling requirements,
emittance evolution, and beam-
quality limitations.Retrieve accelerator concepts,
cooling schemes, and beam-
performance studies.
Machine–detector in-
terfaceRelate luminosity goals, beam con-
straints, and interaction-region de-
sign.Retrieve accelerator design princi-
ples and machine–detector inter-
face studies.
Beam-induced back-
groundsAnalyze the origin of beam-
induced backgrounds and evalu-
ate mitigation strategies.Retrieve detector-performance
studies and simulation-based
mitigation results.
Higgs and elec-
troweak physicsSummarize Higgs production
channels and precision elec-
troweak measurements.Integrate phenomenology with de-
tector and collider constraints.
Beyond-Standard-
Model physicsAssess discovery potential and sig-
natures of new physics scenarios.Retrieve BSM phenomenology,
sensitivity studies, and collider
projections.
Multi-boson pro-
cessesDistinguish diboson, triboson,
and EFT-driven anomalous inter-
actions.Retrieve high-energy scattering
and EFT interpretation literature.
Answerability check Verify whether claims are sup-
ported by available evidence.Prefer abstention over unsup-
ported or hallucinated answers.
These characteristics create a practical challenge for literature navigation. Scientific questions
often require connecting accelerator constraints, beam-induced background mitigation, detector
performance limitations, and physics analysis requirements across multiple sources. Correct
answers frequently depend on retrieving specific evidence fragments rather than entire documents.
For instance, a query on timing cuts for beam-induced background rejection requires different
evidence from one on vector-boson fusion sensitivity to anomalous quartic gauge couplings, even
though both arise within the same literature corpus. This motivates a retrieval-augmented system
capable of chunk-level evidence retrieval, source traceability, and grounded answer synthesis for
detector and physics studies.
5

3 Methodology
3.1 Hybrid Retrieval
The retrieval stage combines sparse lexical retrieval and dense semantic retrieval. Sparse retrieval
preserves exact technical terminology commonly used in HEP literature, while dense retrieval
improves robustness to paraphrased scientific descriptions.
Sparse retriever.The sparse component uses BM25 [ 7], which scores a tokenized query q
and document chunk daccording to term frequency, inverse document frequency, and length
normalization:
SBM25(q, d) =X
t∈qIDF(t)f(t, d)(k 1+ 1)
f(t, d) +k 1(1−b+b|d|/ |d|).(1)
BM25 prioritizes exact term overlap, making it particularly effective in HEP literature where
key concepts are often expressed through stable acronyms and technical keywords such as
BIB, MDI, VBS, and aQGC. The parameter b∈[0,1] controls document-length normalization,
interpolating between no normalization ( b= 0) and full normalization ( b= 1); we use b= 0.75
as a standard compromise that mitigates bias toward longer chunks while remaining sensitive to
length variation. Despite its precision on keyword-driven queries, BM25 is fundamentally limited
by its reliance on exact lexical overlap: it cannot capture semantic similarity when relevant
concepts are expressed using different terminology—such as ”beam-induced background” versus
”backgrounds from muon decays”—nor model compositional meaning or contextual relationships,
motivating the complementary use of dense retrieval.
Dense retriever.The dense component embeds both queries and document chunks into a
shared vector space using sentence-transformers/all-MiniLM-L6-v2 . Chunk embeddings
are indexed with FAISS under cosine similarity [13].
Formally, letf(·) denote the encoder that maps a text input into an embedding vector:
eq=f(q),e d=f(d).(2)
Dense retrieval ranks document chunks by cosine similarity:
Sdense(q, d) =eq·ed
∥eq∥∥ed∥.(3)
Equivalently, FAISS performs nearest neighbor search in the embedding space:
d∗= arg max
d∈DSdense(q, d).(4)
Dense retrieval captures conceptual similarity even when surface forms differ significantly—for
example, ”beam-induced background” may be retrieved via ”backgrounds from muon decays,”
and ”power efficiency” queries may match chunks framed as ”luminosity-normalized energy
consumption.” It also generalizes across heterogeneous writing styles across papers, detector
studies, and phenomenology analyses, and improves recall for exploratory queries where users
may not know the exact terminology in the literature. However, embedding-based similarity can
over-generalize on fine-grained technical queries, and dense models may underperform on rare
acronyms or newly introduced terminology not well represented in the embedding space. These
limitations motivate combining dense retrieval with BM25 in a hybrid scheme.
6

Hybrid retriever.The sparse and dense rankings are merged using the weighted reciprocal-
rank fusion (RRF) score:
SRRF(c) =wd
K+r d(c)+ws
K+r s(c),(5)
where rd(c) and rs(c) denote the dense and sparse ranks of chunk c, and wd,wsare the
corresponding weights. We set K= 60 following the original RRF formulation of Cormack
et al. [8], in which this value was empirically shown to yield robust performance across diverse
retrieval benchmarks and has since been widely adopted as the de facto standard in hybrid
retrieval systems [ 42,43]. The constant Kacts as a smoothing term that mitigates the outsized
influence of the single top-ranked candidate and prevents any one retriever from dominating
the fused score when it produces a high-confidence outlier at rank 1. We further constrain
wd+ws= 1, so the fusion reflects a convex combination of dense and sparse signals, improving
interpretability and reducing the effective hyperparameter space to a single degree of freedom.
The default configuration uses wd= 0.9 and ws= 0.1, reflecting the stronger semantic coverage
of dense retrieval, with the sparse retriever serving primarily as a high-precision fallback for
exact terminology, acronyms, and named entities.
3.2 Agentic Query Decomposition
For complex scientific queries that a single retrieval call cannot adequately address, we introduce
an agentic query expansion layer built on top of the hybrid retriever. The core idea is to
decompose the original query into a set of targeted subqueries, each probing a complementary
aspect of the underlying information need, and to aggregate the resulting evidence into an
expanded candidate pool.
Query decomposition.Given an input query q, the system applies three lightweight language-
model prompts in sequence to produce a set of retrieval-oriented subqueries{q 1, q2, . . . , q N}.
First, a domain tagging prompt identifies which physics domains are semantically relevant to
qfrom a controlled vocabulary (higgs, VBS, multiboson, detector, machine, BSM, ...), relying
on semantic inference rather than keyword matching to handle paraphrase and implicit context.
Second, a query classification prompt assigns qto one of three retrieval strategies: precise fact
(a specific number, parameter, or direct claim), broad synthesis (a summary spanning multiple
papers or concepts), or reasoning (causal, comparative, or mechanistic questions).
Third, a subquery generation prompt produces the final expansion conditioned on the original
query, its detected tags, and its query type. Up to five additional subqueries are generated
according to the classification: precise fact queries receive at most two narrow expansions
to preserve retrieval precision, whereas reasoning queries are decomposed along mechanism,
motivation, and limitation dimensions, and broad synthesis queries are split by domain or process
boundary. Each subquery is self-contained and independently retrievable, targeting a specific
facet of qrather than paraphrasing it holistically. The prompt explicitly prohibits inventing
paper titles, numerical values, or unsupported claims, ensuring that subqueries remain grounded
in the original question.
In this work, we use GPT-OSS-120B for all three stages, and the maximum subquery budget
is capped atN max= 5. The full prompt templates are provided in Appendix A.
Subquery retrieval and aggregation.Each generated subquery qiis passed through the
same hybrid retrieval pipeline described in Sec. 3.1, producing a ranked list of candidate
chunks. The resulting Ncandidate lists are merged by taking the union of retrieved chunks and
deduplicating by chunk identifier. The final evidence pool retains the top- Mchunks ranked by
their best RRF score across all subquery retrievals, where Mis the total evidence budget shared
with the answer generator. This design ensures that the agentic expansion layer reuses the same
7

sparse-dense retrieval infrastructure without introducing additional retrieval mechanisms and
that evidence quality is governed by the same hybrid ranking criteria as the retrieval stage.
3.3 Evidence-Grounded Answer Generation
The answer generation module receives the original query together with a consolidated evidence
set retrieved via both the original query and its decomposed subqueries. The generator is
instructed to produce responses strictly grounded in the provided chunks, to cite supporting
evidence, and to abstain when the retrieved material is insufficient to support a reliable answer.
This grounding constraint is particularly important in detector and physics applications, where
unsupported claims regarding background levels, detector performance, or physics reach can
lead to incorrect scientific interpretations.
For each subquery qi, relevant chunks are retrieved using the hybrid retrieval system described
in Sec. 3.1. The resulting candidate sets are then merged into a unified evidence pool through
deduplication at the chunk level. When multiple subqueries retrieve overlapping or redundant
evidence, duplicates are removed while preserving the highest-scoring occurrence according to
the hybrid ranking function. The final evidence set is constrained to a fixed budget of top- M
chunks, ensuring a consistent input size for downstream generation.
The language model then conditions on this evidence set to produce the final answer to the
original query. For benchmark evaluation, the same generation protocol is applied uniformly to
both answerable and unanswerable queries, enabling consistent measurement of groundness and
abstention behavior, as detailed in Sec. 4.
3.4 System Overview
Figure 1: Overview of the proposed agentic hybrid RAG system fo scientific queries.
Figure 1 presents an overview of the proposed system. The pipeline is organized into three
tightly coupled stages: agentic query decomposition, hybrid retrieval, and evidence aggregation,
followed by grounded answer generation.
8

Given a scientific query, the system first applies an agentic query decomposition module that
generates a set of decomposed subqueries. The purpose of this stage is to explicitly expand the
original information need into multiple complementary retrieval perspectives, particularly for
complex scientific questions that involve multiple physical processes, detector effects, or analysis
assumptions. Instead of relying on a single query formulation, this decomposition step increases
coverage over heterogeneous evidence sources and reduces the risk of missing relevant chun ks
due to lexical or semantic mismatch.
In the second stage, each decomposed subquery is independently processed by the hybrid
retrieval module. This module combines BM25-based sparse retrieval, which is effective for
exact matching of technical terminology and acronyms, with dense embedding-based retrieval,
which captures semantic similarity across paraphrased or contextually reformulated scientific
descriptions. The same hybrid retrieval pipeline is also applied to the original query, ensuring
that the system retains a high-precision evidence backbone while simultaneously exploring
expanded query views. Prior to retrieval, the scientific document corpus is segmented into
chunk-level chunks, which are indexed under both the sparse and dense representations. The
result of this stage is a set of ranked chunk lists corresponding to the original query and each
decomposed subquery.
In the third stage, all retrieved chunks are merged into a unified evidence pool. This
aggregation process performs deduplication at the chunk level to remove redundant content
retrieved from multiple query views, while preserving the highest-ranked occurrence of each
chunk according to the hybrid scoring signal. A fixed evidence budget is then enforced through
rank-based selection, which balances relevance and diversity while maintaining a compact and
controllable context size for downstream generation.
Finally, the answer generation module conditions on the consolidated evidence set to produce
the final response. The model is explicitly constrained to ground its output in the retrieved
chunks, refusing to synthesize claims that lack direct support in the evidence pool. This end-to-
end design integrates decomposition-driven query expansion, hybrid sparse-dense retrieval, and
evidence fusion into a unified framework that improves both recall and robustness in scientific
question answering.
4 Experiments
4.1 Experimental Setup and Benchmarks
Corpus and benchmark.The collected corpus contains 215 muon collider publications
and is segmented into 5,813 indexed chunks, which serve as the fundamental retrieval units
throughout this work. To support evidence attribution and source traceability, each chunk
retains its associated metadata. For dense retrieval, all chunks are encoded into 384-dimensional
embeddings and indexed using FAISS.
The benchmark consists of two components: a retrieval benchmark and an answer-generation
benchmark, designed to decouple retrieval quality from end-to-end generation performance. The
following subsections introduce each component in detail.
Retrieval benchmark and metrics.The retrieval benchmark consists of 58 questions,
including 45 retrievable and 13 unretrievable questions, summarized in Table 2. Retrieval
performance is evaluated on the retrievable subset, while the unretrievable questions are reserved
for abstention evaluation and robustness analysis.
9

Table 2: Retrieval benchmark overview.
Component Count Description
Total benchmark quesions 58 Complete evaluation set
Retrievable questions 45 Used for retrieval evaluation
Unretrievable questions 13 Used for abstention evaluation
For each retrievable question, candidate chunks are annotated with graded relevance judg-
ments on a four-level scale. Grade 3 denotes chunks that directly answer the question, grade 2
indicates strong supporting evidence, grade 1 corresponds to relevant contextual information,
and grade 0 denotes non-relevant content.
For retrieval evaluation, we report chunk-level Precision@1, Recall@5, Mean Reciprocal Rank
(MRR), and graded Normalized Discounted Cumulative Gain (gNDCG)@5. The first three
metrics treat relevance as binary, whereas gNDCG@5 leverages the graded relevance judgments
defined above to evaluate ranking quality. The formulations of retrieval metrics are detailed in
Appendix B.
In addition, we report source-level retrieval performance, which evaluates whether at least
one chunk originating from a relevant source document appears within the retrieved top-k results.
This complementary metric reflects the practical utility of retrieval for literature discovery, where
locating the correct paper may be sufficient even when the exact evidence chunk is not ranked
highest.
End-to-end answer generation benchmark and metrics.The answer generation bench-
mark is intentionally distinct from the retrieval benchmark, enabling assessment of the complete
retrieval-to-generation pipeline rather than the retrieval performance in isolation. Retrieval qual-
ity alone does not adequately reflect end-to-end answer generation performance. We construct
a dedicated answer-level benchmark consisting of 40 questions, including 35 answerable and 5
unanswerable cases, summarized in Table 3. Each question is annotated with a reference answer,
required key points, and a list of unsupported claims that should not appear in the response.
The benchmark evaluates answer correctness, evidence coverage, hallucination resistance, and
abstention behavior, complementing retrieval-focused evaluation.
Table 3: Answer generation benchmark overview.
Component Count Description
Total benchmark questions 40 Complete QA evaluation set
Answerable questions 35 Used for correctness and key-point coverage evaluation
Unanswerable questions 5 Used for abstention evaluation
Required key points 128 Reference-answer evaluation rubric
To evaluate the generated answers, a deterministic LLM-as-a-judge prompt compares each
generated answer against the reference answer, required key points, and unsupported-claim
criteria. We report Good Rate, Satisfactory-or-Better Rate, Key-Point Coverage, Hallucination
Rate, and Abstention Accuracy. We also perform qualitative inspection of representative examples
to validate the consistency of automated judgments. The formulations of answer-generation
metrics are detailed in Appendix C.
4.2 Hybrid Retriever Optimization
Figure 2 presents the weight optimization study for the hybrid retriever, where dense semantic
retrieval and sparse lexical retrieval are combined with different relative weights.
10

Figure 2: Dense weight optimization of the hybrid retriever.
The results indicate that retrieval performance is strongly influenced by the dense retriever
component. Increasing the weight of semantic retrieval generally improves ranking quality,
while a small lexical component remains beneficial. The best performance is achieved with a
dense-retrieval weight of 0.9 and a sparse-retrieval (BM25) weight of 0.1, which is therefore
adopted as the default configuration in all subsequent experiments.
This behavior is consistent with the characteristics of the benchmark. Many questions require
semantic matching across heterogeneous descriptions of detector concepts, physics processes,
and analysis methodologies, which favors dense retrieval. At the same time, a modest BM25
component helps preserve sensitivity to specialized high-energy physics terminology, detector
component names, and exact technical expressions. Overall, the hybrid retriever combines the
broad semantic coverage of dense retrieval with the lexical precision of BM25, outperforming
either signal alone.
4.3 Retrieval Evaluation
Table 4: Chunk-level retrieval performance on the retrievable set of questions. Higher values
indicate better performance, and the best score in each column is highlighted inbold.
Retriever Precision@1 Hit@5 MRR gNDCG@5
Sparse (BM25) 0.222 0.467 0.351 0.122
Dense (vector) 0.6890.9560.806 0.484
Hybrid (RRF)0.756 0.956 0.843 0.510
Because the agentic workflow performs query decomposition and retrieves evidence for
multiple sub-queries, it is not directly comparable under fixed single-query retrieval metrics.
Therefore, it is excluded from the chunk-level retrieval evaluation and is instead assessed in the
end-to-end answer-generation setting.
Table 4 and Figure 3 present the chunk-level retrieval performance of the three retrievers.
The hybrid approach achieves the strongest overall performance, outperforming both standard
BM25 and dense vector retrieval. BM25 alone underperforms due to the prevalence of semanti-
11

Figure 3: Chunk-level retrieval performance across different retrievers on the retrievable questions.
cally diverse queries that require conceptual matching beyond exact lexical overlap, whereas
dense retrieval captures most semantic signals. By combining dense and sparse retrievals, the
hybrid configuration provides both strong semantic matching and robustness to exact technical
terminology.
Figure 4: Chunk-level vs. source-level retrieval performance.
Figure 4 compares chunk-level and source-level retrieval performance. Source-level perfor-
mance is consistently higher than chunk-level performance, indicating that the system often
retrieves the correct source document even when the exact annotated chunk is not ranked first.
This reflects the fact that multiple chunks within the same document may provide redundant or
complementary evidence for the same query, even if only one is explicitly annotated as relevant.
The retrieval evaluation establishes the hybrid retriever as the strongest retrieval backbone.
We next evaluate whether agentic query decomposition and evidence aggregation improve answer
12

quality when built on top of this retriever.
4.4 Answer Generation Evaluation
The retrieval evaluation establishes the hybrid retriever as the strongest retrieval backbone. We
next evaluate whether agentic query decomposition and evidence aggregation improve answer
quality when built on top of this retriever.
Table 5 and Figure 5 summarize the answer-generation results on the 40-question benchmark.
We compare BM25, vanilla RAG, hybrid RAG, and the proposed agentic hybrid RAG. GPT-
OSS-120B is used for both answer generation and answer evaluation.
Evaluation is conducted using rubric-based metrics. Good and Satisfactory+ denote two levels
of correctness based on reference answers and required key points. Key-point coverage measures
the fraction of required reference points covered by the generated response. Hallucination
rate denotes the proportion of answers containing unsupported claims, while the abstention
rate measures the fraction of correctly unanswered questions. Higher values indicate better
performance for all metrics except hallucination and abstention rates, where lower values are
preferred.
Table 5: Answer generation performance on the 40-question benchmark. Higher values are better
for all metrics except hallucination rate (lower is better). The best score is highlighted inbold.
Method Good[%] Satisfactory+[%] Key-points[%] Hallucination[%] Abstention[%]
BM25 30.0 40.0 45.9 15.0 40.0
Vanilla RAG 50.0 52.5 55.1 15.060.0
Hybrid RAG 42.5 47.5 50.2 15.060.0
Agentic Hybrid RAG60.0 62.5 79.3 12.5 60.0
Figure 5: Answer-generation performance across four methods on the 40-question benchmark.
Numbers on Agentic Hybrid RAG show percentage-point improvement vs. Vanilla RAG.
Interestingly, improvements in retrieval metrics do not directly translate into answer-
generation quality. Although the hybrid retriever achieves the strongest retrieval performance,
the corresponding Hybrid RAG baseline does not outperform Vanilla RAG on this benchmark.
This discrepancy suggests that retrieval quality alone is insufficient to characterize end-to-end
answer quality, motivating the use of answer-level evaluation in addition to retrieval metrics.
Agentic hybrid RAG achieves the strongest overall answer-generation performance. Compared
with Vanilla RAG, the Good rate increases from 50.0% to 60.0% and the Satisfactory-or-Better
13

rate from 52.5% to 62.5%. The largest gain is observed in key-point coverage, which rises from
55.1% to 79.3%, indicating substantially more complete utilization of retrieved evidence. The
hallucination rate is also reduced from 15.0% to 12.5%, suggesting that additional evidence
aggregation does not increase unsupported claims.
Abstention accuracy remains unchanged at 60.0% across retrieval-based methods, indicat-
ing that retrieval improvements alone are insufficient to fully address unsupported-question
detection.
5 Conclusion and Outlook
This work introduced a benchmark for retrieval-augmented scientific question answering in the
muon collider domain, covering detector and physics literature spanning accelerator concepts,
beam-induced backgrounds, machine-detector interfaces, Higgs studies, multi-boson processes,
vector-boson scattering, and detector-performance research. Building on this benchmark, we
developed and evaluated an agentic hybrid RAG framework designed for evidence-grounded
scientific literature exploration.
A key observation of this study is that agentic retrieval should complement, rather than
replace, a strong hybrid retriever, particularly in answer generation settings. Across the retrieval
benchmark, the hybrid retrieval component of agentic hybrid RAG consistently achieves the
strongest performance among the evaluated retrieval methods. This indicates that agentic
retrieval is most effective when built on a strong underlying retrieval backbone rather than used
as a replacement.
This distinction is particularly important for HEP analysis agents, where retrieved evidence
may directly influence detector studies, background estimation, and physics analysis workflows.
In such settings, the value of an answer depends not only on its linguistic quality but also on
the ability to trace every scientific claim back to supporting evidence.
The answer-level evaluation further demonstrates the value of controlled evidence expansion.
Agentic hybrid RAG outperforms all evaluated baselines in answer quality, evidence coverage, and
factual grounding, while maintaining strong citation fidelity and low hallucination rates. Overall,
agentic reasoning is most effective when applied to evidence organization, contextualization,
and answer synthesis, rather than as an unconstrained replacement for retrieval. More broadly,
the study highlights the importance of balancing retrieval precision with reasoning flexibility in
scientific RAG systems.
Several limitations remain. The benchmark is self-constructed, some degree of terminology
overlap exists between queries and reference evidence, and answer evaluation relies partly on
an LLM-as-a-judge framework. Future work should therefore incorporate community-reviewed
benchmarks, broader domain coverage, and expert auditing of both retrieval and answer quality.
Beyond benchmark evaluation, it will be important to assess the framework in realistic scientific
workflows, including detector-background studies, machine-detector interface design reviews,
detector optimization tasks, and physics-performance analyses. Such studies would provide a
more direct measure of how evidence-grounded retrieval and reasoning can support day-to-day
research activities.
Looking further ahead, practical deployment of HEP analysis agents will require more than
strong retrieval and question answering. Future systems must operate over versioned scientific
corpora, maintain persistent links between generated conclusions and supporting evidence, and
provide transparent mechanisms for citation, verification, and human review. We view agentic
hybrid RAG as a foundational component of this broader vision: an evidence-aware knowledge
layer that enables future HEP analysis agents to retrieve, inspect, connect, and reason over
scientific literature while keeping analysis decisions and scientific claims grounded in their
sources.
14

Acknowledgements
We appreciate fruitful discussions with Sitian Qian and Chen Zhou. This work is supported in
part by the National Natural Science Foundation of China under Grant No. 12325504.
References
[1]Eli Gendreau-Distler, Joshua Ho, Dongwon Kim, Luc Tomas Le Pottier, Haichen Wang,
and Chengxi Yang. Automating high energy physics data analysis with llm-powered agents.
arXiv preprint arXiv:2512.07785, 2025.
[2]Eric A. Moreno, Samuel Bright-Thonney, Andrzej Novak, Dolores Garcia, and Philip Harris.
Ai agents can already autonomously perform experimental high energy physics.arXiv
preprint arXiv:2603.20179, 2026.
[3]Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.
ACM Computing Surveys, 55(12):1–38, 2023.
[4]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K¨ uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, et al. Retrieval-
augmented generation for knowledge-intensive nlp tasks.Advances in Neural Information
Processing Systems, 33:9459–9474, 2020.
[5]Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InProceedings of EMNLP, pages 6769–6781, 2020.
[6]Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese
bert-networks. InProceedings of EMNLP-IJCNLP, pages 3982–3992, 2019.
[7]Stephen Robertson and Hugo Zaragoza. The Probabilistic Relevance Framework: BM25
and beyond.Foundations and Trends®in Information Retrieval, 4(1-2):1–174, 2009. doi:
10.1561/1500000019.
[8]Gordon V. Cormack, Charles L. A. Clarke, and Stefan B¨ uttcher. Reciprocal rank fusion
outperforms Condorcet and individual rank learning methods. InProceedings of the
32nd Annual International ACM SIGIR Conference, pages 758–759. ACM, 2009. doi:
10.1145/1571941.1572114.
[9]Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei, and Athanasios V Vasi-
lakos. Agentic retrieval-augmented generation: A survey on agentic rag.arXiv preprint
arXiv:2501.09136, 2025.
[10] J.-P. Delahaye et al. Muon colliders. Technical report, CERN, 2019. arXiv:1901.06150.
[11]C. Aime et al. Muon collider physics summary. Technical report, International Muon
Collider Collaboration, 2022. arXiv:2203.07256.
[12]Jeff Johnson, Matthijs Douze, and Herv´ e J´ egou. Billion-scale similarity search with gpus.
IEEE Transactions on Big Data, 7(3):535–547, 2019.
[13]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language
models: A survey.arXiv preprint arXiv:2312.10997, 2024.
15

[14]Karthik Suresh, Neeltje Kackar, Luke Schleck, and Cristiano Fanelli. Towards a RAG-
based summarization for the Electron Ion Collider.JINST, 19(07):C07006, 2024. doi:
10.1088/1748-0221/19/07/C07006.
[15]Tina J. Jat, T. Ghosh, and Karthik Suresh. Retrieval-augmented question answering over
scientific literature for the electron-ion collider.arXiv preprint arXiv:2604.02259, 2026.
[16]James McGreivy, Blaise Delaney, Anja Beck, and Mike Williams. Seeing the Forest Through
the Trees: Knowledge Retrieval for Streamlining Particle Physics Analysis. 2025.
[17]Abhishikth Mallampalli and Sridhara Dasu. MITRA: An AI Assistant for Knowledge
Retrieval in Physics Collaborations. In39th Annual Conference on Neural Information
Processing Systems: Includes Machine Learning and the Physical Sciences (ML4PS), 2026.
[18]Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel S. Weld. S2orc: The
semantic scholar open research corpus. InProceedings of ACL, pages 4969–4983, 2020.
[19]Iz Beltagy, Kyle Lo, and Arman Cohan. Scibert: A pretrained language model for scientific
text. InProceedings of EMNLP-IJCNLP, pages 3615–3620, 2019.
[20]Patrice Lopez. Grobid: Combining automatic bibliographic data recognition and term
extraction for scholarship publications. InProceedings of ECDL, pages 473–474, 2009.
[21]Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. Ragas: Automated
evaluation of retrieval augmented generation.arXiv preprint arXiv:2309.15217, 2023.
[22]Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. InText Summa-
rization Branches Out, pages 74–81, 2004.
[23]Ellen M. Voorhees. The trec-8 question answering track report. InProceedings of TREC,
1999.
[24]Kalervo J¨ arvelin and Jaana Kek¨ al¨ ainen. Cumulated gain-based evaluation of ir techniques.
ACM Transactions on Information Systems, 20(4):422–446, 2002.
[25] J. P. Delahaye et al. Muon colliders.arXiv preprint arXiv:1901.06150, 2019.
[26]K. M. Black et al. Muon Collider Forum report.JINST, 19(02):T02015, 2024. doi:
10.1088/1748-0221/19/02/T02015.
[27] Chiara Aime et al. Muon Collider Physics Summary. 2022.
[28]C. Accettura et al. Interim report for the international muon collider collaboration.arXiv
preprint arXiv:2407.12450, 2024.
[29]Sitian Qian, Congqiao Li, Qiang Li, Fanqiang Meng, Jie Xiao, Tianyi Yang, Meng Lu, and
Zhengyun You. Searching for heavy leptoquarks at a muon collider.JHEP, 12:047, 2021.
doi: 10.1007/JHEP12(2021)047.
[30]Ruobing Jiang, Tianyi Yang, Sitian Qian, Yong Ban, Jingshu Li, Zhengyun You, and Qiang
Li. Searching for Majorana neutrinos at a same-sign muon collider.Phys. Rev. D, 109(3):
035020, 2024. doi: 10.1103/PhysRevD.109.035020.
[31]Ruobing Jiang, Chuqiao Jiang, Alim Ruzi, Tianyi Yang, Yong Ban, and Qiang Li. Searches
for multi-Z boson productions and anomalous gauge boson couplings at a muon collider.
Chin. Phys. C, 48(10):103102, 2024. doi: 10.1088/1674-1137/ad5661.
16

[32]Vernon D. Barger, M. S. Berger, J. F. Gunion, and Tao Han. S-channel Higgs boson
production at a muon muon collider.Phys. Rev. Lett., 75:1462–1465, 1995. doi: 10.1103/
PhysRevLett.75.1462.
[33]D. Neuffer, M. Palmer, Y. Alexahin, C. Ankenbrandt, and J. P. Delahaye. A Muon Collider
as a Higgs Factory. 2013.
[34]E. Celada et al. Probing higgs-muon interactions at a multi-tev muon collider.Journal of
High Energy Physics, 2024(8):21, 2024.
[35]Antonio Costantini, Federico De Lillo, Fabio Maltoni, Luca Mantani, Olivier Mattelaer,
Richard Ruiz, and Xiaoran Zhao. Vector boson fusion at multi-TeV muon colliders.JHEP,
09:080, 2020. doi: 10.1007/JHEP09(2020)080.
[36]Tao Han, Da Liu, Ian Low, and Xing Wang. Electroweak couplings of the higgs boson at a
multi-tev muon collider.Physical Review D, 103:013002, 2021.
[37]B. Abbott et al. Anomalous production of massive gauge boson pairs at muon colliders.
Physical Review D, 108:093009, 2023.
[38]N. V. Mokhov et al. Muon collider interaction region and machine-detector interface design.
arXiv preprint arXiv:1202.3979, 2011.
[39]N. Bartosik et al. Detector and physics performance at a muon collider.Journal of
Instrumentation, 15(05):P05001, 2020.
[40]F. Collamati, C. Curatolo, D. Lucchesi, A. Mereghetti, N. Mokhov, M. Palmer, and
P. Sala. Advanced assessment of beam-induced background at a muon collider.Journal of
Instrumentation, 16(11):P11009, 2021.
[41]D. Lucchesi et al. Detector performance studies at a muon collider.PoS EPS-HEP2019,
page 118, 2020.
[42]Ronak Pradeep, Rodrigo Nogueira, and Jimmy Lin. RRF102: Meeting the TREC-COVID
challenge with a 100+ runs ensemble.arXiv preprint arXiv:2010.00200, 2021.
[43]Ramtin Mesbahi et al. Ask-EDA: A design assistant empowered by LLM, hybrid RAG and
abbreviation de-hallucination. 2024.
17

Appendix
A Query Decomposition Prompt Templates
The following prompts are used in the three-stage query decomposition pipeline described in
Section 3.2. All prompts instruct the model to return structured JSON output only, with no
preamble or markdown formatting.
Stage 1: Domain Tag Detection
TAGDETECTION PROMPT
You are a high-energy-physics query analysis assistant.
Given a user query about muon-collider physics, detector
studies, or related literature, identify the relevant domains.
Allowed tags:
- higgs
- multiboson
- vbs
- aqgc
- detector
- machine
- general
Rules:
- Return only tags that are semantically relevant.
- Do not rely only on exact keywords; infer from context
and paraphrase.
- Use "general" only if no specific tag applies.
- Return JSON only.
Output schema:
{
"tags": ["..."],
"reason": "brief explanation"
}
Stage 2: Query Classification
QUERY CLASSIFICATION PROMPT
You are classifying scientific RAG queries.
Classify the query into exactly one query type:
- precise_fact: asks for a specific number, result,
parameter, paper claim, definition,
or direct fact.
- broad_synthesis: asks for a summary across several
papers, topics, or concepts.
- reasoning: asks why, how, compare, connect,
affect, influence, limitation,
18

motivation, or implication.
- paraphrase: asks to rewrite, polish, or rephrase
provided text.
Rules:
- Use semantic intent, not only keywords.
- If the query asks for both a fact and an explanation,
choose reasoning.
- Return JSON only.
Output schema:
{
"query_type": "...",
"reason": "brief explanation"
}
Stage 3: Subquery Generation
QUERY DECOMPOSITION PROMPT
You are a domain-aware query decomposition module for a
scientific RAG system.
Given:
1. the original user query,
2. detected domain tags,
3. query type,
generate retrieval-oriented subqueries.
Rules:
- The first subquery must be the original query verbatim.
- Generate 2 to 6 additional subqueries only if they
genuinely aid retrieval.
- Subqueries should retrieve supporting evidence, not
answer the question directly.
- For precise_fact queries: keep subqueries narrow;
generate at most 2 extra subqueries.
- For reasoning queries: decompose into mechanism,
motivation, limitation, and relevant evidence angles.
- For broad_synthesis queries: decompose by domain
or process boundary.
- Do not invent paper titles, numerical values, or
unsupported claims.
- Avoid generic subqueries that would retrieve too many
unrelated chunks.
- Return JSON only.
Output schema:
{
"subqueries": [
19

"original query",
"subquery 1",
"subquery 2"
],
"reason": "brief explanation"
}
Illustrative Example
Table 6 shows the pipeline output for a representative reasoning query tagged as vbsandaqgc .
Stage Output
Input“Why is a muon collider particularly sensitive to anomalous quartic gauge couplings in
VBS?”
Tagsvbs,aqgc
Typereasoning
Subqueries1. Why is a muon collider particularly sensitive to anomalous quartic gauge couplings in
VBS?
2. Vector boson scattering at a muon collider
3. Vector boson fusion high-energy muon collider
4. VBS anomalous quartic gauge couplings muon collider
5. Sensitivity to aQGC in VBS at high-energy lepton colliders
6. Unitarity and aQGC constraints from VBS
Table 6: Pipeline output for a reasoning query tagged vbs+aqgc . Subquery 1 is the original
query verbatim; subqueries 2–6 target mechanism, phenomenology, and theoretical context
angles respectively.
B Retrieval Metrics Formulation
LetRk={d1, . . . , d k}denote the top- kretrieved chunks for a query, ranked by decreasing
retrieval score. Let rel idenote the graded relevance score of the chunk at ranki.
Precision@k.Precision@k measures the fraction of retrieved chunks within the top- kresults
that are relevant:
Precision@k=1
kkX
i=11(rel i>0),(6)
where1(·) is the indicator function.
Hit@k.Hit@k evaluates whether at least one relevant chunk appears among the top- kretrieved
results:
Hit@k=1 kX
i=11(rel i>0)>0!
.(7)
For a set of queries, Hit@k is averaged across all queries.
20

Mean Reciprocal Rank (MRR).Let rdenote the rank position of the first relevant chunk.
The reciprocal rank (RR) is defined as
RR =1
r.(8)
The Mean Reciprocal Rank over a set ofNqueries is
MRR =1
NNX
q=11
rq,(9)
wherer qis the rank of the first relevant chunk for queryq.
Graded Discounted Cumulative Gain (gDCG).To account for graded relevance, the
Discounted Cumulative Gain at rankkis defined as
gDCG@k=kX
i=12reli−1
log2(i+ 1).(10)
Graded Normalized Discounted Cumulative Gain (gNDCG).The graded Normalized
Discounted Cumulative Gain is obtained by normalizing gDCG with the ideal ranking:
gNDCG@k=gDCG@k
IDCG@k,(11)
where IDCG @kdenotes the maximum achievable gDCG obtained by sorting retrieved chunks
according to decreasing relevance scores.
C Answer Metrics Formulation
For answer generation evaluation, a deterministic judge prompt compares each generated answer
against the reference answer, required key points, and unsupported-claim criteria. Let Ndenote
the total number of evaluated questions.
Good Rate.Good Rate measures the fraction of answers judged asGood:
GoodRate =1
NNX
i=11(yi= Good),(12)
wherey iis the judge label assigned to thei-th answer.
Satisfactory-or-Better Rate.This metric measures the fraction of answers judged as either
GoodorSatisfactory:
Sat + Rate =1
NNX
i=11(yi∈ {Good,Satisfactory}).(13)
Key-Point Coverage.Let Kidenote the set of required key points for question i, and let ˆKi
denote the subset correctly covered by the generated answer. Key-Point Coverage is defined as
KPC =1
NNX
i=1|ˆKi|
|Ki|.(14)
21

Hallucination Rate.Hallucination Rate measures the proportion of answers containing
unsupported factual claims:
HallucinationRate =1
NNX
i=11(hi= 1),(15)
where hi= 1 indicates that the judge identifies at least one unsupported claim in the generated
answer.
Abstention Accuracy.For questions labeled as unanswerable from the available evidence,
Abstention Accuracy measures whether the model correctly refrains from providing unsupported
answers:
AbstentionAcc =1
MMX
i=11(ai= ˆai),(16)
where Mis the number of unanswerable questions, aiis the ground-truth abstention label, and
ˆaiis the model’s abstention decision.
In addition, qualitative inspection of representative examples is performed to verify the
consistency of automated judgments and to identify common failure modes.
D Reproducibility Commands
Primary retrieval benchmark:
python evaluate_agentic.py \
--query_file data/eval/queries_eval_v4_template.json \
--dense_weight 0.9 \
--bm25_weight 0.1 \
--sweep_weights \
--output data/eval/eval_results_agentic_retrieval_FULL.json
Answer-level QA evaluation:
python evaluate_agentic.py \
--query_file data/eval/queries_eval_qa40_publishable.json \
--dense_weight 0.9 \
--bm25_weight 0.1 \
--judge_answers \
--qa_systems hybrid,agentic,oracle \
--resume \
--output data/eval/eval_results_agentic_qa40_FULL.json
Figure and table generation:
python make_qa_paper_outputs.py \
--input data/eval/eval_results_agentic_qa40_FULL.json \
--outdir paper_outputs_qa40_FULL
22