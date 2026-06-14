# Multi-Field Hybrid Retrieval-Augmented Generation for Maritime Accident Root Cause Analysis

**Authors**: Seongjin Kim, Sungil Kim

**Published**: 2026-06-11 12:05:47

**PDF URL**: [https://arxiv.org/pdf/2606.13249v1](https://arxiv.org/pdf/2606.13249v1)

## Abstract
Maritime accident adjudication reports contain critical tribunal findings for root cause analysis (RCA), yet retrieving relevant precedents and drafting consistent reports from decades of records remains labor-intensive. This paper proposes a multi-field hybrid retrieval-augmented generation (RAG) framework for automated maritime RCA, utilizing a comprehensive dataset of 13,329 Korea Maritime Safety Tribunal (KMST) reports (1971-2025). We transform raw adjudications into a structured knowledge base of "incident cards", indexing three distinct fields-Summary, Causes, and Disposition-alongside a hierarchical L1/L2 cause taxonomy. Our retrieval strategy employs a field-aware hybrid approach, fusing sparse and dense rankings via Reciprocal Rank Fusion (RRF). Given the lack of large-scale expert relevance labels, we evaluate retrieval performance using ceiling-normalized recall and nDCG based on a metadata-derived proxy relevance score. Experimental results demonstrate that our proposed retrieval significantly outperforms baseline methods, improving NormRecall@100 from 0.18 to 0.55. Furthermore, grounding the generator on the retrieved precedents enhances RCA generation quality over an LLM-only baseline, increasing the LLM-as-a-judge score from 3.34 to 3.72. These findings suggest that field-aware RAG can substantially streamline maritime safety investigation workflows by enabling faster precedent search and more consistent, evidence-based RCA drafting.

## Full Text


<!-- PDF content starts -->

Multi-Field Hybrid Retrieval-Augmented Generation for Maritime
Accident Root Cause Analysis
Seongjin Kimaand Sungil Kima,†
Abstract— Maritime accident adjudication reports contain
critical tribunal findings for root cause analysis (RCA), yet
retrieving relevant precedents and drafting consistent reports
from decades of records remains labor-intensive. This paper
proposes a multi-field hybrid retrieval-augmented generation
(RAG) framework for automated maritime RCA, utilizing
a comprehensive dataset of 13,329 Korea Maritime Safety
Tribunal (KMST) reports (1971–2025). We transform raw
adjudications into a structured knowledge base of “incident
cards”, indexing three distinct fields—Summary,Causes, and
Disposition—alongside a hierarchical L1/L2 cause taxonomy.
Our retrieval strategy employs a field-aware hybrid approach,
fusing sparse and dense rankings via Reciprocal Rank Fusion
(RRF). Given the lack of large-scale expert relevance labels, we
evaluate retrieval performance using ceiling-normalized recall
and nDCG based on a metadata-derived proxy relevance score.
Experimental results demonstrate that our proposed retrieval
significantly outperforms baseline methods, improving Norm-
Recall@100 from 0.18 to 0.55. Furthermore, grounding the
generator on the retrieved precedents enhances RCA generation
quality over an LLM-only baseline, increasing the LLM-as-
a-judge score from 3.34 to 3.72. These findings suggest that
field-aware RAG can substantially streamline maritime safety
investigation workflows by enabling faster precedent search and
more consistent, evidence-based RCA drafting.
I. INTRODUCTION
Maritime accidents can lead to severe safety consequences
and operational disruptions, including delays and capacity
losses in maritime transport and port logistics [1]–[3]. To
prevent recurrence, systematic Root Cause Analysis (RCA)
is essential for robust safety management [4], [5]. In practice,
RCA often relies on analogical reasoning over past adju-
dications: investigators search for historically similar cases,
examine tribunal cause findings, and justify conclusions with
precedent-based evidence.
Despite its importance, performing precedent-based search
over formal adjudication reports (e.g., Fig. 1) remains a
daunting task. These records span decades and are character-
ized by a dense mixture of domain-specific legal and techni-
cal terminology. Furthermore, the information is fragmented
across multiple sections, such as accident narratives, causal
reasoning, and administrative dispositions. Consequently, an-
alysts spend substantial time manually synthesizing these
disparate parts, which hinders timely organizational learning
and consistent safety intervention.
Recent advances in Large Language Models (LLMs) offer
a potential solution for automating such narrative synthesis.
However, in specialized legal–technical domains, LLMs are
aDepartment of Industrial Engineering, Ulsan National Institute of Sci-
ence and Technology (UNIST), Ulsan 44919, Republic of Korea.
†Corresponding author: sungil.kim@unist.ac.kr
Fig. 1. Example KMST adjudication report (translated into English from
the original Korean, identifiers blurred)
prone tohallucinations—generating plausible but factually
incorrect regulations or causal mechanisms. While Retrieval-
Augmented Generation (RAG) mitigates this by conditioning
outputs on external evidence [6], [7], its application to mar-
itime RCA faces two distinct challenges. First, adjudication
reports aremulti-field documentswith diverse semantics
(Summary,Causes,Disposition); treating them as monolithic
text blocks dilutes the field-specific signals necessary for
precise retrieval [8]. Second,evaluation at scaleis notori-
ously difficult, as expert relevance judgments are expensive
to obtain, yet retrieval quality is the primary determinant of
downstream reliability.
In this work, we formulate maritime RCA assistance as
case-informed structured generation. Given a new accident
scenario, our system retrieves historically similar adjudica-
tions and generates a structured RCA output grounded in
these precedents. We specifically focus on leveraging multi-
field document structures to enhance retrieval precision and
introduce a reproducible evaluation proxy for large-scale
benchmarking.
The main contributions of this study are as follows:
•Curating a Structured Maritime Knowledge Base:arXiv:2606.13249v1  [cs.AI]  11 Jun 2026

We construct a large-scale, structured knowledge base of
13,329 KMST adjudication cases (1971–2025), featur-
ing retrievable document fields and hierarchical L1/L2
cause tags to facilitate evidence-grounded analysis.
•Proposing an Integrated Retrieval and Evaluation
Framework:We develop a multi-field hybrid retrieval
strategy that optimizes case matching and introduce a
reproducible, metadata-driven evaluation protocol that
demonstrates significantly improved structured RCA
generation over LLM-only baselines.
II. RELATED WORK
A. RCA and Precedent-Grounded Investigation
RCA in safety-critical systems seeks to explain how
multiple contributing factors combine to produce adverse
events [4], [5]. In maritime safety, tribunal-style adjudication
reports encode expert RCA judgments—comprising causal
reasoning and administrative dispositions—that investigators
consult as precedents. This motivates treating adjudication
records as an evidence corpus for retrieval-based analysis,
where a system must not onlyfindrelevant precedents but
alsosynthesizeexplanations with traceable support.
B. Specialized RAG and Hybrid Retrieval
RAG reduces hallucination in knowledge-intensive tasks
by grounding LLM outputs in external evidence [6], [7].
For semi-structured technical reports, retrieval relevance is
multifaceted: it involves descriptive similarity in narratives,
causal similarity in reasoning, and outcome similarity in dis-
positions. While sparse lexical methods (e.g., BM25) excel at
matching rare technical terms [9], dense retrieval captures se-
mantic nuances using embeddings likeBAAI/bge-m3[10].
Hybrid retrieval, fused via Reciprocal Rank Fusion (RRF),
combines these strengths [11], [12]. Our work extends this
by utilizingfield-awareindexing [13], preventing the dilution
of distinct causal signals that occurs in monolithic text
blocks [14].
C. Weak Supervision for Domain Taxonomies
In specialized domains, large-scale expert annotation is
often infeasible, necessitating rule-based information extrac-
tion and weak supervision to generate structured signals
[15]. Taxonomy-aligned metadata enables standardized out-
puts and corpus-level analysis. Following these principles,
we employ a rule-based mapping to assign KMST L1/L2
cause tags, providing a consistent label space for both the
structured knowledge base and the generatorG.
D. Maritime Accident Text Mining
Prior research on maritime reports has addressed various
analytical tasks, including accident type prediction [16],
statistical cause analysis [17], [18], causal relation extraction
[19], [20], semantic clustering [21], and knowledge graph
construction [22]. While these studies demonstrate the utility
of NLP for tribunal records, they are primarilydiscriminative
or focused on extracting local patterns from historical data.
Such approaches lack the capacity to generate coherent,evidence-grounded explanations fornewscenarios. In con-
trast, our framework performsgenerativeRCA, synthesizing
a causal narrative and standardized tags while explicitly
surfacing supporting precedents for traceable investigation.
III. METHOD
A. Problem Formulation
We define the task of maritime RCA assistance ascase-
informed structured generation. Given a query accident
scenarioq—typically a free-text description of a vessel
incident—the system retrieves a ranked set ofkprecedent
chunks from a structured knowledge baseD:
Ck=Retriever(q,D) ={c 1,c2,...,c k},(1)
where each chunkc i∈Drepresents a specific semantic
section from a historical adjudication report, such as the
Summary,Causes, orDispositionfield. For instance, a re-
trieved chunkc 1may contain the causal reasoning for a past
collision similar toq.
Conditioned on the inputqand the retrieved evidenceC k,
the generatorGproduces a structured RCA outputY:
Y=G(q,C k) ={n,T},(2)
wherenis a concise cause narrative andTis a set of
taxonomy-aligned cause tags. This formulation ensures that
the generation is not only structurally consistent but also
grounded in verified historical precedents.
B. KMST Adjudication Knowledge Base
The knowledge baseDis constructed from official ad-
judication reports published by the Korea Maritime Safety
Tribunal (KMST) [23]. These records, spanning decades
of maritime history, are semi-structured tribunal documents
characterized by standardized headings but highly variable
phrasing and narrative lengths. As illustrated in Fig. 2,
we converted each raw record into plain text and applied
cleaning and normalization. Each processed record was then
standardized into acardidentified by a unique ID, integrating
the textual content with essential metadata such as accident
type, keywords, tribunal jurisdiction, and incident year. To
preserve semantic specificity, the main text of each card was
partitioned into three distinct retrievable fields:Summary(in-
cident description),Causes(tribunal’s causal reasoning), and
Disposition(administrative outcomes). Each field was stored
as an independently retrievablechunkwhile maintaining a
reference to its parent case ID for traceability. Across 13,329
standardized cards, this process yielded 37,007 retrievable
chunks with an average length of 283 characters (median:
212), providing a dense and structured evidence pool for the
subsequent retrieval stage.
C. Cause Taxonomy and Rule-Based Tagging
To facilitate structured generation, we adopted the KMST
cause taxonomy as a standardized label space, which consists
of high-level groups (L1) and 24 fine-grained factors (L2).
Since raw reports often lack exhaustive ground-truth tags, we

Fig. 2. Data construction pipeline: from raw document extraction to structured card normalization and hierarchical L1/L2 tagging.
employed a rule-based labeling system following the princi-
ples of weak supervision [15], [24]. This system utilizes ap-
proximately 2,400 dictionary entries and 800 regex patterns
to map theKeywordsandCausesfields of each card to the
L1/L2 taxonomy. The labeling process achieved a coverage
of 97.6% (13,012 cases), and the resulting tag distribution
aligns with official maritime statistics [25], ensuring the face
validity of our structured knowledge base.
D. Multi-Field Hybrid Retrieval
Given an input scenarioq, our retriever employs a hybrid
strategy designed to capture both lexical overlaps and se-
mantic nuances across different document sections. For each
field inD, we implement two complementary retrievers:
(1)Sparse Retrievalusing BM25 [9] for exact keyword
matching of technical identifiers, and (2)Dense Retrieval
usingbge-m3[10] for capturing semantic paraphrases.
To aggregate results across the three fields and two
retrieval methods—totaling six independent rankings—we
utilize Reciprocal Rank Fusion (RRF) [11]:
RRFScore(c) = ∑
r∈R1
krrf+rank r(c),(3)
whereRis the set of rankings andk rrfis a smoothing constant
set to 60. This field-aware fusion ensures that the system
recovers relevant evidence even when the queryqshares
vocabulary with different semantic sections of a historical
card.
E. Structured RCA Generation
The final stage of our framework involves generating the
structured outputY={n,T}by conditioning the generator
on the retrieved evidence. As illustrated in Fig. 3, the top-
kchunksC kare first sorted in descending order of their
RRFScore and then provided ascontextto the genera-
torG(Qwen3-Next-80B-A3B-Instruct). To maintain
traceability and allow the model to distinguish between
different types of evidence, each chunk is annotated with
its original case ID and field type (Summary,Causes, or
Disposition).We utilize a temperature of 0.0 to minimize output vari-
ance and enforce a strict JSON schema via system prompt-
ing. The model is specifically tasked with synthesizing the
disparate retrieved precedents into a coherent and concise
cause narrativen, while simultaneously mapping the incident
to the appropriate taxonomy-aligned tagsT. This structured
approach ensures that the resulting RCA report is not only
human-readable but also consistent with the established
KMST cause classification system.
IV. EXPERIMENTS
A. Setup and Baselines
We partitioned the 13,329 cards into train, validation, and
test sets with a 70%/15%/15% ratio. All retrieval indices
were constructed using the training set. Strategy comparisons
and hyperparameter tuning (e.g.,k rrf) were performed on the
validation set, while final performance is reported on the
held-out test set.
Retrieval Baselines.To evaluate the efficacy of our multi-
field hybrid strategy, we compared it against:
•Whole-document indexing:Concatenating theSum-
mary,Causes, andDispositionfields into a single mono-
lithic text block per card.
•Single-field indexing:Indexing only theSummaryor
Causesfields to assess the information gain from field
separation.
For each indexing strategy, we evaluated sparse retrieval
(BM25), dense retrieval (BAAI/bge-m3), and the proposed
hybrid retrieval. Unless otherwise specified, we retrieved
k=100 chunks for evaluation.
Generation Baselines.We compared our RAG framework
against anLLM-only baseline, which utilizes the same
generatorG(Qwen3-Next-80B-A3B-Instruct) and
prompt structure but without any retrieved contextC k. This
comparison isolates the impact of precedent-grounding on
generation quality. For our proposed system,Greceives the
topk=40 chunks as context; given our field-aware indexing,
this typically provides evidence from 10–13 distinct histori-
cal cards.

Fig. 3. Overview of the proposed multi-field hybrid RAG framework.
Tag Normalization.Despite a temperature of 0.0, LLMs
may occasionally produce non-standard surface forms for
tags instead of exact taxonomy terms. To ensure consistent
evaluation ofT, we applied a lightweight normalization
step whereGmaps any non-standard output to the closest
standardized L2 tag. This was applied identically to both the
baseline and the proposed method.
B. Evaluation Protocol
In the absence of large-scale expert relevance labels for
retrieval, we define a proxy relevance metric, RelScore, to
evaluate the alignment between a retrieved candidate cardc
and a query cardq. This score is calculated as a weighted
sum of four domain-specific factors:
RelScore(q,c) = ∑
i∈{kw,type,loc,year}wi·Si(q,c),(4)
whereS kwdenotes the IDF-weighted Jaccard similarity of
keywords,S typeandS locrepresent binary matches for acci-
dent type and tribunal jurisdiction, andS year=1/(1+|∆ year|)
captures temporal proximity. We empiricallly set the weights
(wkw,wtype,wloc,wyear)to(0.5,0.4,0.05,0.05)and define the
pseudo-gold setas candidates exceeding a RelScore of 0.5.
Retrieval performance is assessed usingNormRe-
call@100andnDCG@100. To account for varying
sizes of the gold set across different accident patterns,
NormRecall@kmeasures coverage relative to the maximum
possible recall achievable at a given cutoffk:
NormRecall@k=|Retrieved k∩GoldSet|
min(k,|GoldSet|).(5)
Complementarily, nDCG@100 evaluates the ranking quality
by rewarding the placement of highly relevant candidates
(those with higher RelScore) at earlier ranks. The DCG is
computed as ∑k
i=1Gi
log2(i+1), where the gainG iis defined
by the RelScore of the candidate at ranki, subsequently
normalized by the Ideal DCG (IDCG).
Regarding the generation task, the expert-authoredCauses
field serves as the ground-truth reference for evaluating the
synthesized cause narrativen. We employ a multi-facetedevaluation involving ROUGE-L for lexical overlap, SBERT
cosine similarity for semantic alignment, and an LLM-as-a-
judge score on a 1–5 scale to assess overall coherence and
grounding [26]. For the structured tagsT, we report exact-
match accuracy for the primary L1 group along with Micro-
F1 and Jaccard scores for the multi-label L2 factors.
All retrieval and generation processes are executed on the
original Korean adjudication text to preserve domain-specific
linguistic nuances and technical precision. For the purpose
of international dissemination, all tables and case examples
provided in this paper utilize condensed English translations.
V. RESULTS
A. Retrieval Results
Table I shows that leveraging document fields is critical.
Whole-document indexing mixes accident description, causal
reasoning, and administrative outcomes into one long text
block, which dilutes field-specific signals and increases spu-
rious matches. Field-wise indexing enables the retriever to
match scenario descriptions in Summary while also capturing
technical causal patterns inCausesand decision patterns in
Disposition, improving both coverage and ranking quality.
Impact of field separation.Moving from whole-
document indexing (Dense: 0.1819 NormRecall) to multi-
field indexing over all fields (Dense: 0.4981) yields a 174%
relative gain in coverage. Notably, addingDispositionto
Summary+Causesprovides a large additional boost (Norm-
Recall 0.3060→0.4981; nDCG 0.5336→0.7512), suggest-
ing that administrative outcomes carry discriminative signals
that help disambiguate lexically similar but causally different
cases.
Effectiveness of hybrid retrieval.Hybrid retrieval yields
an additional gain by combining complementary strengths.
BM25 captures exact matches of rare legal/technical tokens,
while dense retrieval captures paraphrases and semantically
equivalent expressions. Fusing both reduces misses caused by
either lexical mismatch or over-reliance on surface keywords,
improving NormRecall from 0.4981 (Dense) and 0.4215
(BM25) to 0.5463 (Hybrid).

TABLE I
RETRIEVAL PERFORMANCE(TOP-k=100).
Strategy Norm. Recall nDCG
Whole-Document (Dense) 0.1819 0.6123
Single-Field Summary (Dense) 0.1785 0.6055
Single-Field Causes (Dense) 0.1450 0.4405
Multi-Field (Summary+Causes; Dense) 0.3060 0.5336
Multi-Field (All Fields; Dense) 0.4981 0.7512
Multi-Field (All Fields; BM25) 0.4215 0.7034
Multi-Field (All Fields; Hybrid) 0.5463 0.7697
TABLE II
GENERATION PERFORMANCE(TOP-k=40).
Metric LLM-only Proposed Improv.
ROUGE-L (F1) 0.192 0.206 +7.3%
Semantic Sim. (SBERT) 0.723 0.724 +0.1%
LLM-as-a-Judge (1–5) 3.340 3.723 +11.5%
L1 Accuracy 0.771 0.774 +0.4%
L2 Micro-F1 0.460 0.490 +6.5%
L2 Jaccard 0.365 0.395 +8.2%
B. Generation Results
Table II reports end-to-end improvements when the gener-
ator is grounded in retrieved precedents (top-k=40 chunks).
The judge score increases substantially (3.340→3.723;
+11.5%), while SBERT similarity changes only marginally.
This pattern is expected: an instruction-tuned LLM can
produce fluent summaries from the scenario alone, but re-
trieval primarily improves factual consistency and key-cause
coverage by injecting concrete precedent evidence.
For tags, improvements are smaller but consistent (e.g., L2
micro-F1 0.460→0.490), indicating that retrieval provides
useful signals for taxonomy alignment even without super-
vised tag training. In practice, the largest qualitative benefit is
that outputs becomeprecedent-grounded: the generator can
cite retrieved causes/dispositions as support, which is not
possible for an LLM-only baseline.
While traditional supervised classifiers can perform iso-
lated tag prediction, they are structurally incapable of gener-
ating the cohesive causal narratives or providing the traceable
precedent evidence required for this task. Therefore, to
evaluate the end-to-end framework, we compare the proposed
method against an LLM-only baseline, explicitly isolating the
impact of retrieval grounding on structured generation.
C. Case Study
Table III highlights two representative cases demonstrat-
ing how multi-field retrieval grounding mitigates halluci-
nated causal links. In the LLM-only setting, the model
frequently relies on spurious correlations derived from tan-
gential scenario details. For instance, in the stranding case,
the LLM-only baseline infers vague structural defects from
the “sudden impact” description, completely missing the
geographical hazard. In the explosion case, the presence of an
intoxicated crew triggers an erroneous logical leap to alcohol
vapors as the ignition source.
Conversely, the proposed framework leverages retrieved
precedents—specifically utilizing the technical and legalcontext preserved in the independent Causes and Disposition
fields—to ground the generation process. This contextual in-
jection shifts the model’s reasoning from speculative deduc-
tion to precedent-aligned diagnostic inference. By accessing
historically and situationally similar records, the framework
correctly identifies specific technical mechanisms, such as
charting failures and propane gas ignition, thereby producing
an analysis highly consistent with expert judgments.
VI. DISCUSSION AND LIMITATIONS
The experimental results validate that structuring RAG
architectures around domain-specific document fields sig-
nificantly enhances downstream generation. By isolating
descriptive narratives (Summary) from technical and legal
reasoning (CausesandDisposition), our multi-field hybrid
retrieval strategy effectively reconstructs the diagnostic rea-
soning pathway utilized by expert investigators. This struc-
tural alignment ensures that the generator is grounded not
merely in lexically similar texts, but in historically and
causally analogous precedents. Furthermore, the fusion of
dense and sparse retrieval signals proves essential for mar-
itime safety workflows, providing robustness against vocab-
ulary mismatches while preserving the precise retrieval of
critical regulatory and technical terminology.
Despite these gains, certain limitations remain within
the evaluation framework. Due to the absence of large-
scale expert relevance labels, retrieval performance was
measured using a metadata-based proxy score. Additionally,
the hierarchical cause tags were assigned via rule-based
extraction rather than manual annotation, and the LLM-
as-a-judge evaluation carries the inherent risk of model-
dependent bias. However, formulating this fixed, heuristic
proxy directly addresses the practical lack of a universal gold
standard in specialized administrative domains. It provides
a reproducible, systematic baseline for large-scale strategy
comparisons under realistic data constraints. Ultimately, the
consistent end-to-end generation improvements confirm that
optimizing for this retrieval proxy translates into more accu-
rate and traceable RCA outputs, while highlighting future
directions for enhancing the uncertainty-awareness of the
evaluation framework [27].
VII. CONCLUSION
This paper proposed a multi-field hybrid RAG frame-
work for maritime accident RCA, utilizing a comprehensive
dataset of KMST adjudication reports. By structuring his-
torical records into field-aware “incident cards” and fusing
sparse and dense retrieval signals via RRF, the proposed
method significantly enhances the quality of precedent re-
trieval. To address the lack of human-annotated gold stan-
dards in specialized domains, we introduced a reproducible,
metadata-based evaluation proxy that enables a systematic
and objective comparison of retrieval strategies. End-to-
end experiments demonstrate that these retrieval enhance-
ments directly translate into more accurate, structured, and
precedent-grounded RCA generation compared to an LLM-
only baseline. Our framework establishes a practical foun-

TABLE III
CASESTUDYEXAMPLES: COMPARISON OFCAUSALREASONING(CONDENSED)
Scenario (Excerpt) Expert CausesGenerated Analysis
LLM-only Baseline Proposed Framework
[STRANDING]
Occurred shortly after passing a
lighthouse in calm weather and
good visibility; a sudden severe
hull impact and shaking were re-
ported.Approached a reef too
closely without position
verification; improper
navigation near the hazard.Impact was likely due to structural de-
fects in the hull or improper equipment
handling; stability may have been weak-
ened despite the clear weather.Vessel stranded on underwater rocks after
navigating too close to the lighthouse; the
core cause was failure to verify position
and recognize reefs on the chart.
[FIRE/EXPLOSION]
Occurred shortly after a crew
member entered a cabin with
matches and cigarettes; a sudden
explosion and fire were reported,
causing casualties.Negligent handling of
propane gas; gas leaked and
accumulated in the cabin
and was ignited when a
match was struck.Explosion was caused by alcohol vapors
accumulated from drinking in the en-
closed cabin; poor supervision of intoxi-
cated crew was the primary factor.Explosion was caused by ignition of
leaked propane gas accumulated in the
cabin due to poor maintenance; the match
acted as the ignition source.
* All content examples are translated from Korean.
dation for automated maritime investigation support. Future
work will explore the integration of formal legal regulations,
such as COLREGs, and graph-based reasoning to further
expand the system’s diagnostic depth and interpretability.
ACKNOWLEDGMENT
This work was supported by the National Research
Foundation of Korea(NRF) grant funded by the Korea
government(MSIT)(No.RS-2023-00218913).
REFERENCES
[1] J. Park and S. Kim, “Maritime anomaly detection based on vae-cusum
monitoring system,”Journal of the Korean Institute of Industrial
Engineers, vol. 46, no. 4, pp. 432–442, 2020.
[2] Y . Oh and S. Kim, “Grid-based bayesian bootstrap approach for real-
time detection of abnormal vessel behaviors from ais data in maritime
logistics,”IEEE Transactions on Automation Science and Engineering,
vol. 21, no. 4, pp. 6680–6692, 2023.
[3] Y . Oh, K. Yoon, J. Park, and S. Kim, “Comparative evaluation of
vae-based monitoring statistics for real-time anomaly detection in ais
data,”Maritime Policy & Management, vol. 52, no. 4, pp. 609–626,
2025.
[4] J. Reason,Human Error. Cambridge University Press, 1990.
[5] S. Fanet al., “A review of human error in maritime accidents: Ap-
proaches and applications,”Ocean Engineering, vol. 216, p. 107906,
2020.
[6] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, R. Nogueira,
H. Paskov, P. Stenetorp, T. Rocktäschel, and S. Riedel, “Retrieval-
augmented generation for knowledge-intensive NLP tasks,” inAd-
vances in Neural Information Processing Systems (NeurIPS), vol. 33,
2020, pp. 9459–974.
[7] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2023.
[8] W. Soh, H. Kim, and B.-J. Yum, “Application of kernel principal com-
ponent analysis to multi-characteristic parameter design problems,”
Annals of Operations research, vol. 263, no. 1, pp. 69–91, 2018.
[9] S. Robertson and H. Zaragoza, “The probabilistic relevance frame-
work: BM25 and beyond,”Foundations and Trends in Information
Retrieval, vol. 3, no. 4, pp. 333–389, 2009.
[10] Y . Chen, S. Xiaoet al., “Bge-M3 embedding: A multilingual, multi-
function, multi-granularity text embedding model,” arXiv preprint
arXiv:2409.10216, 2024.
[11] G. V . Cormack, C. L. A. Clarke, and S. Buettcher, “Reciprocal rank
fusion outperforms condorcet and individual rank learning methods,”
inProceedings of the 32nd International ACM SIGIR Conference on
Research and Development in Information Retrieval, 2009, pp. 758–
759.
[12] S. Kim and S. Kim, “A hybrid prompt agent for maritime ais
data analysis: performance optimization through query classification
and dynamic routing,”Journal of the Korean Institute of Industrial
Engineers-Vol, vol. 52, no. 2, pp. 163–175, 2026.[13] S. MacAvaney, A. Yates, A. Cohan, and N. Goharian, “Structured
document retrieval for NLU,” inProceedings of the 43rd International
ACM SIGIR Conference on Research and Development in Information
Retrieval, 2020.
[14] H. Kim and H. Kim, “Contextual anomaly detection for high-
dimensional data using dirichlet process variational autoencoder,”IISE
Transactions, vol. 55, no. 5, pp. 433–444, 2023.
[15] A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Ré,
“Snorkel: Rapid training data creation with weak supervision,” in
Proceedings of the VLDB Endowment, vol. 11, no. 3, 2017, pp. 269–
282.
[16] H.-M. Park, M.-A. Cheon, and J.-H. Kim, “Developing a model for
predicting of ships accident using multi-task learning,” inProceedings
of the 32nd Annual Conference on Human and Cognitive Language
Technology, 2020, pp. 179–183.
[17] J. Y . Choi, “A study on the causes of marine accidents and prevention
of marine accidents in vessels,”Cultural Interaction Studies of Sea
Port Cities, vol. 25, pp. 337–359, 2021.
[18] J. Lee, J. Kwak, Y . Oh, and S. Kim, “Quantifying incident impacts and
identifying influential features in urban traffic networks,”Transport-
metrica B: Transport Dynamics, vol. 11, no. 1, pp. 279–300, 2023.
[19] X. Yan, J. Zhang, S. Fan, Z. Yang, and W. Wang, “Causation analysis
of marine traffic accidents using deep learning approaches: A case
study from china’s coasts,”Ocean Engineering, vol. 278, p. 114353,
2023.
[20] S. Moon, J. Kim, and H. Park, “Sequence labeling-based multi-
ple causal relations extraction using pre-trained language model for
maritime accident prevention,” inProceedings of the 35th Annual
Conference on Human and Cognitive Language Technology. HCLT,
2023, pp. 101–108, korean Maritime Accident Reports corpus, multi-
causal extraction.
[21] B. Yoon, S. Park, H. Bae, and S. Sim, “Maritime safety tribunal
ruling analysis using SentenceBERT,”Journal of the Korean Society
of Marine Environment & Safety, vol. 29, no. 7, pp. 843–856, 2023.
[22] J. Zhang, X. Yan, S. Fan, Z. Yang, and W. Wang, “Makg: A
maritime accident knowledge graph for intelligent accident analysis
and management,”Ocean Engineering, vol. 295, p. 119280, 2024.
[23] Korea Maritime Safety Tribunal, “Korea maritime safety tribunal
(kmst) website,” https://www.kmst.go.kr, data source of adjudication
documents; portal landing site.
[24] K. Lee, A. Gray, and H. Kim, “Dependence maps, a dimensionality
reduction with dependence distance for high-dimensional data,”Data
Mining and Knowledge Discovery, vol. 26, no. 3, pp. 512–532, 2013.
[25] Korea Maritime Safety Tribunal, “Statistics of marine accidents,” https:
//kmst.go.kr/eng/page.do?menuIdx=227, accessed: 2025-09-01.
[26] L. Zheng, W.-L. Chiang, Y . Sheng, S. Zhuanget al., “Judging
LLM-as-a-judge with MT-bench and chatbot arena,”arXiv preprint
arXiv:2306.05685, 2023.
[27] T. Yoon and H. Kim, “Uncertainty estimation by density aware
evidential deep learning,”arXiv preprint arXiv:2409.08754, 2024.