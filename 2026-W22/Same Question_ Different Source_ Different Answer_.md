# Same Question, Different Source, Different Answer: Auditing Source-Dependence in Medical Multi-Source RAG

**Authors**: Yubo Li, Rema Padman, Ramayya Krishnan

**Published**: 2026-05-27 20:38:05

**PDF URL**: [https://arxiv.org/pdf/2605.29084v1](https://arxiv.org/pdf/2605.29084v1)

## Abstract
A retrieval-augmented generation (RAG) system deployed over a multi-author institutional corpus can give a different answer to the same question depending on which source it retrieves -- a failure mode the dominant single-gold-answer paradigm cannot diagnose. We argue that source-dependence is a missing axis of NLP evaluation, and that auditing it means shifting the unit of evaluation from answer correctness to the inter-source relationship. We make this concrete in transplant patient education, where institutional sources demonstrably disagree, releasing three artefacts: TransplantQA, a benchmark of real patient questions, each answered by grounding generation in multiple institutional handbooks as candidate sources; HERO-QA, a hierarchical retrieval strategy that grounds and audits each answer; and a structured-output judge that scores inter-source relationships on a validated 5-label taxonomy. At scale, better retrieval reveals far more disagreement than prior estimates suggested -- understating its prevalence, not its intensity. The framework is domain-agnostic and transfers to legal and educational RAG: measuring source-dependence is a responsibility for deployed multi-source NLP generally.

## Full Text


<!-- PDF content starts -->

Same Question, Different Source, Different Answer:
Auditing Source-Dependence in Medical Multi-Source RAG
Yubo Li, Rema Padman, Ramayya Krishnan
Carnegie Mellon University
{yubol, rpadman, rk2x}@andrew.cmu.edu
Abstract
A retrieval-augmented generation (RAG) sys-
tem deployed over a multi-author institutional
corpus can give a different answer to the same
question depending on which source it retrieves
— a failure mode the dominant single-gold-
answer paradigm cannot diagnose. We argue
thatsource-dependenceis a missing axis of
NLP evaluation, and that auditing it means
shifting the unit of evaluation from answer cor-
rectness to theinter-source relationship. We
make this concrete in transplant patient edu-
cation, where institutional sources demonstra-
bly disagree, releasing three artefacts:Trans-
plantQA, a benchmark of real patient ques-
tions, each answered by grounding generation
in multiple institutional handbooks as candidate
sources;HERO-QA, a hierarchical retrieval
strategy that grounds and audits each answer;
and a structured-output judge that scores inter-
source relationships on a validated 5-label tax-
onomy. At scale, better retrieval reveals far
more disagreement than prior estimates sug-
gested — understating itsprevalence, not its
intensity. The framework is domain-agnostic
and transfers to legal and educational RAG:
measuring source-dependence is a responsibil-
ity for deployed multi-source NLP generally.
1 Introduction
A patient three months past a heart transplant
types a question into an institutional Q&A system:
“When can I travel internationally again?”1Behind
the system, an RAG pipeline retrieves passages
from the patient-education handbook of the insti-
tution that performed the surgery. The answer is
grounded, cited, and confidently delivered. Had the
same query been grounded in a peer institution’s
handbook, the recommended waiting period might
have been three, six, or twelve months — with
identical confidence and fluency, and no indication
1Adapted from a real patient post on a transplant forum
included in our benchmark.that the guidance is institution-specific rather than
universal.
This kind ofinter-source heterogeneityis en-
demic to medical RAG. Patient-facing institutional
documents reflect local protocols, editorial choices,
and decades of accumulated risk-management cau-
tion; they are not interchangeable. Yet the domi-
nant benchmarks for medical question answering
— MedQA (Jin et al., 2021), MedMCQA (Pal et al.,
2022), PubMedQA (Jin et al., 2019), BioASQ
(Tsatsaronis et al., 2015) — assume one correct
answer per question and cannot diagnose whether
the answer a patient sees is contingent on which
document the retriever happened to return.
We argue this exposes a missing axis of NLP
evaluation. As RAG becomes deployed infrastruc-
ture over multi-author institutional corpora — in
medicine, but equally in law and education — the
field needs to measuresource-dependence: whether
the answer a user receives is contingent on which
source the retriever happened to return. We frame
this as a new mission for evaluation research, and
operationalise it by shifting the unit of analysis
from single-answer correctness tointer-source re-
lationship: given the same question, what is the
structured relationship between the answer a gen-
erator produces when grounded in document A
versus document B? This paper makes four contri-
butions toward that shift, using transplant patient
education as a case study in which institutional
sources demonstrably disagree.
1.An evaluation-paradigm argument(§1, §7):
the single-gold-answer paradigm cannot diag-
nose source-dependence, the dominant failure
mode of deployed multi-source RAG; closing
the gap requires evaluating the inter-source
relationship, not refining single-gold bench-
marks.
2.TransplantQA(§3): a benchmark opera-
tionalising this shift — 1,115 real patient ques-
1arXiv:2605.29084v1  [cs.CL]  27 May 2026

tions, each answered by grounding generation
in 102 transplant patient-education handbooks
(the candidate sources) from 23 U.S. centers
across five organ types, partitioned into agen-
eralsubset (answered by every handbook) and
anorgan-specificsubset, enabling both full-
corpus and stratified inter-source comparison.
3.HERO-QA(§4.2): a hierarchical evi-
dence retrieval and orchestration strategy for
handbook-grounded clinical QA, using full-
document context for short handbooks (elim-
inating retrieval-miss failures) and section-
aware hierarchical retrieval with reranking for
longer ones, with explicit retrieval metadata
for grounding audit.
4.Empirical characterization at scale(§6):
the full output of a production run over
the benchmark (48,056 grounded answers,
5,730,465 pairwise comparisons), released for
reuse. The inter-source relationship is mea-
sured by a structured-output judge (the eval-
uation instrument; §4.3) validated against hu-
man annotators atκ= 0.842(§5).
Our characterization also yields a methodolog-
ical observation: comparing the reference run
against an earlier 14B run with a lower-capacity
retriever, the average handbook absence rate drops
13.6 pp while per-pair divergence is essentially un-
changed (§6.4) — prior estimates understated the
prevalenceof disagreement, not itsintensity. Cru-
cially, the framework is not medicine-specific: le-
gal RAG (retrieving over federal/state/circuit prece-
dent) and educational RAG (retrieving over state-
stratified curriculum standards) deploy over the
same kind of multi-source corpora and inherit
the same blind spot, and the three components —
multi-source benchmark, inter-source taxonomy,
structured-output judge — transfer directly to both
(§7). Measuring source-dependence is thus a mis-
sion for deployed multi-source NLP broadly, not a
medical-domain convenience.
2 Related Work
Medical QA benchmarks.Medical-QA eval-
uation treats QA as single-best-answer predic-
tion: MedQA (Jin et al., 2021), MedMCQA (Pal
et al., 2022), PubMedQA (Jin et al., 2019), and
BioASQ (Tsatsaronis et al., 2015) score against cu-
rated gold answers, and patient-facing extensions
(Ben Abacha et al., 2017; Zeng et al., 2020; Sing-
hal et al., 2023) retain the single-gold assumption.TransplantQA instead makes therelationshipbe-
tween answers grounded in different documents the
unit of analysis; to our knowledge no prior medical
QA benchmark tests inter-source heterogeneity at
this scale.
LLM-as-judge and cross-document incon-
sistency.LLM-as-judge protocols (Zheng
et al., 2023; Zhu et al., 2025; Kim et al.,
2024; Liu et al., 2023) typically return a single
scalar or label; our judge instead co-emits
narrative metadata ( divergence_topic ,
clinical_significance ), enabling the
taxonomy and severity analyses of §6 at es-
sentially unchanged per-pair cost. Separately,
contradiction detection via NLI (Schuster et al.,
2022), factuality decomposition (Min et al., 2023),
and RAG-hallucination evaluation (Niu et al.,
2024) target a binary signal against a reference;
we instead treat each answer as faithful to its
source and ask whether two sourcesthemselves
agree, with a 5-label taxonomy that surfaces
COMPLEMENTARY/DIVERGENTvariation a
binary lens misses.
Institutional variation in medicine.Wennberg
and Gittelsohn (1973) documented small-area vari-
ation in clinical practice unexplained by patient
characteristics, launching a long literature on
clinical-practice variation. Patient-facing educa-
tional material is the visible boundary of this insti-
tutional variation; TransplantQA provides an NLP-
tractable instrument for measuring it.
3 The TransplantQA Benchmark
TransplantQA pairs a corpus of patient-education
handbooks from U.S. transplant centers with a
question set drawn from real patient information-
seeking behavior, so that an RAG system’s answer
to any benchmark question can be grounded in
(and evaluated against) multiple plausible institu-
tional sources. Unlike single-gold medical QA
benchmarks, the unit of analysis in TransplantQA
is the inter-sourcerelationshipbetween answers
grounded in different documents.
3.1 Handbook Corpus
We collected 102 patient-education handbooks
from 23 major U.S. solid-organ transplant centers,
representing 16 of the 20 largest programs by pro-
cedure volume. The corpus spans five organ types
— heart (26), lung (26), kidney (22), liver (17), and
2

Table 1: TransplantQA handbook corpus by organ.Cen-
tersis the number of distinct contributing institutions.
Heart Kidney Liver Lung Panc.
Handbooks 26 22 17 26 11
Centers 17 14 11 15 8
Pre 10 8 5 10 4
Post 11 10 4 11 3
Combined 5 4 8 5 4
pancreas (11) — and the contributing institutions
are geographically distributed across the United
States, comprising both large academic medical
centers and community-based transplant programs.
All documents were obtained as PDFs from institu-
tional websites and patient education portals.
Centers organize patient education differently:
some provide separate documents for the pre-
transplant phase (evaluation, listing, waiting) and
the post-transplant phase (recovery, medications,
long-term follow-up), while others issue a sin-
gle combined handbook. We treat each phase-
specific document as a distinct unit, yielding 37
pre-transplant, 39 post-transplant, and 26 com-
bined handbooks. Each is assigned an identifier
encoding organ, institution, and care phase (e.g.,
heart_baylor_combined ). Table 1 summa-
rizes the corpus.
3.2 Question Set
We curated 1,115 patient questions to serve as the
evaluation set for cross-center comparison (Fig-
ure 1). Questions wereharvested from real on-
line transplant communities and platforms— pa-
tient forums and social media (e.g., Reddit trans-
plant subreddits, Mayo Clinic Connect, Inspire),
patient-advocacy organizations (National Kidney
Foundation, American Liver Foundation), and in-
stitutional Q&A pages — using transplant- and
symptom-keyword search to surface genuine infor-
mation needs. The 3,000+ harvested candidates
were then (i) de-duplicated (cosine >0.85 plus
manual review), (ii) double-checked for quality
and relevance, and (iii)anonymized and rephrased
to strip user-identifying content and make each
question self-contained, yielding the released 1,115
(mean length 23.6 words). Source breakdown and
inclusion criteria are in Appendix A.
Each question is annotated with: (i) anorgan-
type label— heart, kidney, liver, lung, pancreas, or
general; (ii) one or more clinical topic categories
drawn from a 13-topic taxonomy (Appendix B);and (iii) fine-grained sub-topic tags (43 unique).
Questions are multi-labeled to reflect cross-cutting
concerns.
General vs. organ-specific split.A central de-
sign choice is the partition of the question set into
ageneralsubset (311 questions, 27.9%) and an
organ-specificsubset (804 questions across five
organ types). General questions address topics rel-
evant to all transplant recipients — immunosup-
pressant side effects, reproductive health, mental
health — and are answered byeveryhandbook in
the corpus, producing 102
2
= 5,151 pairwise com-
parisons per question. Organ-specific questions
are answered only by handbooks of the matching
organ type, producing No
2
comparisons where
No∈ {11,17,22,26,26} . The two subsets to-
gether support both full-corpus and stratified inter-
source analyses.
3.3 Anonymization and Release
Because questions are harvested from public fo-
rums and social media, every released question
was anonymized and rephrased to remove any
user-identifying content from the original post
(Appendix A); the released benchmark also uses
anonymized handbook identifiers. Center names
in handbook IDs are retained because transplant
centers are public institutions and the analyses we
enable are explicitly cross-institutional. Release-
location metadata is anonymized for review; the
planned release package includes the benchmark,
the raw handbook-extraction output, the question
annotations, and the full pairwise-comparison out-
puts. Original PDFs are not redistributed but
are listed by URL for independent retrieval. Ap-
pendix C provides a Datasheet-style data card (Ge-
bru et al., 2021).
4 Pipeline Architecture
Our pipeline is a three-stage process that takes the
benchmark question set and the handbook corpus as
input and produces, for every benchmark question,
a structured matrix of pairwise inter-handbook re-
lationships. It runs on open-weight LLMs (Qwen3-
32B for both generation and judging in our refer-
ence run) and is designed for resumable execution
on heterogeneous SLURM clusters. The method-
ological core of this section isHERO-QA, the hier-
archical evidence-retrieval strategy used in Stage 2
(§4.2, Figure 2); the structured pairwise judge in
3

Figure 1: TransplantQA construction. Patient questions are harvested from real online transplant communities and
platforms (patient forums and social media, patient-advocacy organizations, and institutional Q&A) via transplant-
and symptom-keyword search, then de-duplicated, quality/relevance-checked, and anonymized and rephrased to
remove user-identifying information — yielding 1,115 questions (311 general answered by every handbook + 804
organ-specific), paired with 102 patient-education handbooks from 23 U.S. centers across five organ types.
Stage 3 (§4.3) is the measurement instrument that
operationalises the inter-source evaluation.
4.1 Stage 1: Structured Extraction
Raw PDF handbooks are converted to structured
JSON using LlamaParse (LlamaIndex, 2024), pre-
serving section headings, paragraph boundaries,
and page metadata. The per-handbook output con-
tains organ type, institution, care phase, source
path, full text, and a section list with headings,
body text, and page numbers. This structure en-
ables section-aware chunking in Stage 2. Extrac-
tion is idempotent.
4.2 Stage 2: HERO-QA Retrieval-Augmented
Generation
HERO-QA(Hierarchical Evidence Retrieval and
Orchestration for Handbook-grounded clinical QA)
is the retrieval strategy used in Stage 2 (Fig-
ure 2). It is a recall-firstmulti-layerretrieval sys-
tem designed for the institutional-handbook set-
ting, in which a query descends through a length-
routing gate, a hierarchical document model, four
parallel first-stage retrievers, rank fusion, cross-
encoder reranking, and parent-section expansion.
Throughout, HERO-QA exposes retrieval metadata
(which mode produced the context, which sections
were touched) so downstream evaluation can audit
whether an answer was grounded in full-document
or retrieved evidence.Routing and document model (Layers 0–1).
Short handbooks (full text ≤80 k chars) are passed
in full and retrieval is skipped, eliminating retrieval-
miss for short documents. Longer handbooks are
decomposed intoparent sections(preserving head-
ings/pages) and overlappingchild chunks(160
words, 32-word overlap, each prefixed with its par-
ent heading); this document →section →chunk hier-
archy is the substrate for retrieval and expansion.
Four parallel retrievers + fusion + rerank (Lay-
ers 2–4).Against the expanded query, HERO-
QA runs four first-stage retrievers: dense child-
chunk retrieval (FAISS (Douze et al., 2026)
withBAAI/bge-large-en-v1.5 (Xiao et al.,
2024)), sparse child-chunk BM25 (Robertson and
Zaragoza, 2009),section-body navigation(BM25
over section text, hits mapped to child chunks),
andtitle navigation(BM25 over section headings,
catching topic matches when body wording differs).
The four rankings are combined by Reciprocal
Rank Fusion ( kRRF= 60 (Cormack et al., 2009);
navigation signals down-weighted) and reranked
with a MiniLM cross-encoder (Wang et al., 2020).
Parent-section expansion (Layer 5).Top child
chunks are expanded back to their parent sections
plus immediate neighbours, so the generator re-
ceives coherent section-level context; the top-5 ex-
panded passages form the evidence. An evidence-
sufficiency check triggers full-document fallback
4

Figure 2: HERO-QA: a multi-layer retrieval system. A query is routed by handbook length: short handbooks
bypass retrieval and use full-document context (Route A); long handbooks descend through a hierarchical document
model (document →sections →child chunks), four parallel first-stage retrievers (dense FAISS, child BM25,
section-body navigation, title navigation), RRF fusion, cross-encoder reranking, and parent-section expansion. The
top evidence grounds Qwen3-32B generation; retrieval metadata is retained for audit, and a low-evidence signal
triggers full-document fallback.
when retrieved evidence is weak.
Answer generation.For each (question, hand-
book) pair the retrieved passages are supplied to
Qwen3-32B at temperature 0 with a fixed prompt
(Appendix D) instructing the model to (a) rely ex-
clusively on the provided context, (b) return a stan-
dardized NOT ADDRESSED prefix when the hand-
book contains no relevant information rather than
fabricate, and (c) cite the supporting section head-
ing when one exists. The stage produces 48,056
grounded answers in the reference run.
4.3 Stage 3: Structured Pairwise Judgment
Absence pre-screen.Each answer is first
screened for absence: a fast heuristic checks for
the canonical NOT ADDRESSED prefix, and an-
swers that escape the heuristic are passed to a bi-
nary classifier (also Qwen3-32B) using a structured
YES/NO prompt. Absence is cached per (hand-
book, question) pair, so each handbook is screened
once across all comparisons it participates in. Any
pair containing at least one absent answer is imme-
diately assigned the ABSENTlabel, skipping the
comparison call.
Five-label taxonomy.For every pair of non-
absent answers, the judge classifies their relation-
ship into one of five categories with operationaldefinitions (Table 2). The taxonomy is designed to
be (a) clinically interpretable, (b) jointly exhaus-
tive over the relationships we observed during pi-
lot annotation, and (c) ordered along a coverage–
agreement axis from no information (ABSENT)
through full alignment (CONSISTENT), additive
but compatible content (COMPLEMENTARY), sub-
stantive but bounded disagreement (DIVERGENT),
to outright opposition (CONTRADICTORY).
Structured output beyond the label.A standard
LLM-as-judge protocol would return only the clas-
sification. Our judge instead returns a structured
JSON record per pair containing five fields:
1.classification — one of the five labels;
2.reasoning — a 2–3 sentence clinical justi-
fication;
3.divergence_topic — a short noun
phrase naming thelocusof disagreement
(emitted only when classification̸∈
{CONSISTENT,ABSENT});
4.clinical_significance∈
{low,medium,high} — judge-assessed
severity (emitted only for DIVERGENTand
CONTRADICTORY);
5.judge_metadata — input/output token
counts and decoding latency.
5

Table 2: Five-label taxonomy for pairwise comparison of center-specific answers. Examples are drawn from the
released benchmark.
Label Definition Example
ABSENT One or both answers indicate the handbook
does not address the topic.Center A provides dietary guidance; Center B’s hand-
book has no relevant section.
CONSISTENT Same clinical recommendation, no meaning-
ful informational difference.Both centers advise avoiding grapefruit due to tacrolimus
interactions.
COMPLEMENTARY Clinically compatible but differing in detail
or scope.Center A lists side effects; Center B additionally de-
scribes management strategies.
DIVERGENT Substantive, clinically meaningful differ-
ence (different thresholds, timelines, or rec-
ommended actions).Center A recommends exercise at 6 weeks post-
transplant; Center B at 8–12 weeks.
CONTRADICTORYDirectly opposing clinical guidance. Center A allows ABO-incompatible live donors; Cen-
ter B states they cannot proceed.
The two narrative fields are the key methodologi-
cal enabler of the downstream analyses described
in §6. Clustering 34,706 divergence_topic
strings yields a 991-node taxonomy of disagree-
ment themes; the clinical_significance
field permits stakes-adjusted aggregation. The
judge prompt and the full output schema are in
Appendix E; inference is greedy (temperature 0)
for reproducibility.
Comparison matrix.For a question answered
byNhandbooks the N
2
pairwise records and the
integer matrix M∈ {0, . . . ,4}N×Nencoding the
labels are written together as a single per-question
JSON file. Diagonal entries are CONSISTENTby
convention. Per-question artefacts are independent
and idempotent, enabling resume-safe incremental
execution.
4.4 Implementation and Scale
The released pipeline runs over the full benchmark
on a heterogeneous SLURM cluster (PSC Bridges-
2, NVIDIA H100 80 GB) with a sharded executor
that splits the question set into 10generalshards
and 10non-generalshards per pipeline stage; each
shard is resumable at the matrix-file granularity for
comparison and the question-file granularity for
generation. The complete production run produces
48,056 answers(Stage 2) and5,730,465 pairwise
comparisons(Stage 3), of which 4,519,245 pre-
screen as ABSENTand 1,211,220 require an LLM-
judge call. Total wall-time and compute cost are
reported in Appendix F. To our knowledge this
is the largest documented application of LLM-as-
judge to a single medical heterogeneity benchmark.5 Validating the Evaluation Instrument
The structured-output judge is the measurement
instrument through which we read inter-source re-
lationships; its trustworthiness underwrites every
finding in §6. We validate it along two axes: agree-
ment with human clinical annotators (§5.1) and an
ablation against the natural alternative protocol —
a label-only judge followed by a post-hoc extractor
— confirming that the structured single-call design
is required, not a convenience (§5.2).
5.1 Human–judge agreement
We validate the structured-output judge against
human annotators on a stratified sample of 200
pairwise records (40 per non-absent label, plus
40 ABSENTcontrols); CONTRADICTORYis over-
sampled at 46% of all contradictions in the produc-
tion run for power on the rare class. Annotators
see the original question and both handbook an-
swers;the judge’s label, reasoning, divergence
topic, and clinical-significance rating are with-
held. Two annotators rate each pair following the
operational definitions in Table 2; protocol and
rubric in Appendix I.
Results.Both annotators completed all 200
pairs. Inter-annotator agreement isCohen’s κ=
0.655 (raw agreement 73.0% ) — substantial un-
der Landis–Koch. The two annotators agreed on
146/200 pairs; we treat their joint-agreed label
as the human-majority gold. On those 146 pairs
the judge agrees with the majority 87.7% of the
time, yieldingjudge-vs-majority κ= 0.842 (al-
most perfect) and weighted F1 = 0.876 (macro F1
= 0.841 ). Per-label F1: ABSENT 1.00, CONTRA-
DICTORY 0.99, CONSISTENT 0.83, COMPLEMEN-
TARY0.70, DIVERGENT0.69.
6

Failure-mode taxonomy.Of 18 judge errors
against the majority, 14 (78%) cluster on the COM-
PLEMENTARY/DIVERGENTboundary: 8 cases
where the majority calls COMPLEMENTARYbut
the judge calls DIVERGENT, and 6 where the ma-
jority calls COMPLEMENTARYbut the judge calls
CONSISTENT. The judge’s discrimination is ro-
bust at the extremes (presence/absence; flat contra-
dictions) but soft on the middle of the coverage–
agreement axis — consistent with the taxonomy’s
design intent that COMPLEMENTARYsits between
CONSISTENTand DIVERGENT.
Clinical significance.On 49 paired DIVER-
GENT/CONTRADICTORYpairs where all three
(judge, A, B) rated significance, judge-vs-human
κ= 0.385 — fair but not strong. The judge’s
grades are directionally correct (no systematic
low/highflips) but the fine-grained gradations
should be treated as a population-level signal, not
a per-pair adjudication.
5.2 Structured vs. label-only judge: an
ablation
A natural alternative to our structured single-call
judge is a label-only judge followed by a post-
hoc extractor that conditions on (question, answer a,
answer b, label) to recover divergence_topic
andclinical_significance in a second
call. We test the two protocols (Condition A: struc-
tured single-call, ours; Condition B: label-only +
post-hoc) on the same 200-pair sample. Three find-
ings emerge. (i) Categorical agreement is κ=
0.669 , but the disagreement concentrates on the
most consequential class: of A’s 40 DIVERGENT
pairs, B agrees on only 4 and downgrades 31 (78%)
to COMPLEMENTARY. (ii) Clinical significance
is unrecoverable post-hoc: on n= 44 paired DI-
VERGENT/CONTRADICTORYpairs B returnshigh
for all 44 ( κ=0 against A’s mixedhigh/medium).
(iii) Topic strings on agreed-label pairs are semanti-
cally equivalent and cluster identically under the §6
pipeline. Condition B is ≈5–6×faster per pair but
loses the DIVERGENT/COMPLEMENTARYdiscrim-
ination and severity gradation. Structured single-
call output is therefore a design requirement of the
framework, not a convenience.
6 Benchmark Characterization
We apply our pipeline to the full TransplantQA
benchmark using Qwen3-32B as both generator
and judge, reporting global and stratified label dis-Table 3: Per-organ heterogeneity rates from our refer-
ence production run. Per-pair rates are averaged over
non-absent pairs.
rabs Rdiv Rcon pctany div
general 0.778 0.175 0.287 55.6%
heart 0.692 0.146 0.177 45.3%
kidney 0.655 0.157 0.161 49.5%
liver 0.683 0.138 0.180 39.0%
lung 0.596 0.148 0.156 54.2%
pancreas 0.668 0.185 0.202 29.9%
tributions, the per-organ heterogeneity profile, and
a system-level comparison.
6.1 Global Label Distribution
Of the 5,730,465 pairwise comparisons, 4,519,245
(78.9%) pre-screen as ABSENTbecause at least
one handbook returned NOT ADDRESSED . Of the
remaining 1,211,220 LLM-judged pairs, COM-
PLEMENTARYdominates (75.4%), followed by
DIVERGENT(12.9%), CONSISTENT(7.1%), and
CONTRADICTORY( <0.1% ). Explicit contradic-
tion is therefore rare; the dominant mode of dis-
agreement is two centers covering different as-
pects of the same question (COMPLEMENTARY)
or giving substantively different recommendations
(DIVERGENT).
6.2 Per-Organ Heterogeneity
Table 3 reports per-organ rates: the absence rate
rabs, the per-pair divergence rate Rdiv(fraction of
non-absent pairs labelled DIVERGENTor CONTRA-
DICTORY), the per-pair consistency rate Rcon, and
the proportion of questions in each organ for which
at least one pair is divergent (pctany div ).
Absence dominates across all organs (60–78%):
even within the matching-organ subsets, the av-
erage handbook addresses only one third to half
of relevant patient questions. Per-pair divergence
rates cluster between 0.14 and 0.19, with pancreas
and general questions sitting at the top of the range.
The prevalence metric pctany div exhibits broader
spread (30–56%), reflecting that pancreas and liver
questions are more often answered by a small sub-
set of handbooks (so even when divergence exists,
it concentrates within a few questions).
6.3 Per-Handbook Coverage Spread
Per-handbook absence rates span0.45 to 0.99
(mean 0.74), a 2 ×spread between the most-
comprehensive and most-silent handbooks. The
handbook ×question-organ heatmap (Appendix
7

Figure 3) shows the expected block-diagonal pat-
tern but also systematic editorial differences: some
handbooks are broadly comprehensive across all
columns, while others are silent even within their
own organ.
6.4 System-Level Comparison: 14B Earlier
Run vs. 32B Reference Run
A previous run over the same benchmark used a
hybrid-retrieval pipeline with Qwen3-14B as both
generator and judge. Comparing it to the 32B refer-
ence run (per-organ deltas in Appendix H) isolates
the effect of the pipeline upgrade. Three observa-
tions stand out: (i) absence drops 12–19 pp across
every organ (mean ∆rabs=−0.136 ) as better re-
trieval surfaces passages the earlier pipeline missed;
(ii) per-pair divergence rates are roughly unchanged
or modestly lower (mean ∆Rdiv=−0.031 ; the
stronger judge is not more aggressive); (iii) the
proportion of questions showinganydivergence
rises substantially (mean +15.9 pp), driven me-
chanically by the absence drop. The per-pair rate
reported by earlier baselines ( ≈20% ) is thus stable,
but theprevalenceof disagreement was substan-
tially understated because absence was hiding it:
stronger pipelines reveal latent disagreement rather
than manufacturing it.
6.5 Downstream Uses Enabled by Structured
Output
The two narrative fields support analyses that
classifier-only judges cannot. Embedding the
16,113 unique divergence_topic strings and
clustering them yields a 991-theme taxonomy
ofwhatsources disagree about (largest themes:
post-transplant pregnancy timing, blood-test fre-
quency, rejection symptoms, dental-care timing);
theclinical_significance field permits
severity-weighted re-aggregation, which empiri-
cally tracks unweighted disagreement frequency
closely (Spearmanρ >0.99at the question, topic,
and handbook levels) and is most useful for surfac-
ing individual high-stakes pairs. These analyses
are enabled by the structured judge output, not by
the labels alone.
7 Discussion
Generalisation to non-medical deployed RAG.
The framework’s three slots — multi-source bench-
mark, inter-source taxonomy, structured-output
judge — are domain-agnostic.Legal RAG(West-
law AI, Lexis+ AI, Harvey) retrieves over jurisdic-tional layers and firm-specific research, yet single-
gold benchmarks (LegalBench, LexGLUE) can-
not surface whether a query grounded in Cali-
fornia versus Texas precedent diverges in client-
actionable ways.Educational RAGretrieves
over state-stratified standards (Common Core,
NGSS) and publisher-specific expositions, while
ScienceQA/GSM8K cannot surface whether a stu-
dent’s answer depends on which state’s materials
were indexed. Each instantiates the same slots with
a domain-appropriate taxonomy: this paper’s em-
pirical contribution is medical, its methodological
contribution is for deployed RAG generally.
Judge limitations.An LLM judge inherits
known biases (Zheng et al., 2023; Kim et al., 2024):
self-preferencewhen generator and judge share
a family (pair-symmetric framing mitigates but
does not eliminate this; §5 measures κ= 0.842
agreement),length/citation artefacts, andcost(Ap-
pendix F).
8 Conclusion
We introduced TransplantQA, the HERO-QA re-
trieval system, and a structured-output LLM-as-
judge as instruments for measuring inter-source
heterogeneity in deployed medical RAG; all arte-
facts (48,056 answers, 5.73M pairwise compar-
isons, judge–majority κ= 0.842 ) are released.
Empirically, prior estimates understated thepreva-
lenceof disagreement, not its intensity — ab-
sence was hiding it. Methodologically, struc-
tured single-call judging is a requirement, not a
convenience: post-hoc extraction loses the DI-
VERGENT/COMPLEMENTARYdiscrimination and
severity gradation the framework depends on.
8

Limitations
The empirical instantiation is confined to U.S. solid-
organ transplant patient education (English, 2024–
2025 snapshot); legal and educational transferabil-
ity (§7) is conceptual. The judge is an LLM;
the 200-pair validation measures population-level
agreement but cannot detect sub-axis biases (insti-
tution, organ, answer length) (Zheng et al., 2023;
Kim et al., 2024); the released per-pair JSON pre-
serves judge reasoning for individual-decision au-
dit. Apparent inter-source divergence can also be
inflated by retrieval failures rather than true dis-
agreement; the absence pre-screen partially miti-
gates this.
References
Asma Ben Abacha, Eugene Agichtein, Yuval Pinter, and
Dina Demner-Fushman. 2017. Overview of the med-
ical question answering task at TREC 2017 LiveQA.
Gordon V . Cormack, Charles L. A. Clarke, and Stefan
Büttcher. 2009. Reciprocal rank fusion outperforms
Condorcet and individual rank learning methods. In
Proceedings of the 32nd International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, pages 758–759. ACM.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré,
Maria Lomeli, Lucas Hosseini, and Hervé Jégou.
2026. The Faiss library.IEEE Transactions on Big
Data. Early access.
Timnit Gebru, Jamie Morgenstern, Briana Vecchione,
Jennifer Wortman Vaughan, Hanna Wallach, Hal
Daumé III, and Kate Crawford. 2021. Datasheets for
datasets.Communications of the ACM, 64(12):86–
92.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2021. What disease
does this patient have? a large-scale open domain
question answering dataset from medical exams.Ap-
plied Sciences, 11(14):6421.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. PubMedQA: A
dataset for biomedical research question answering.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 2567–
2577, Hong Kong, China. Association for Computa-
tional Linguistics.
Seungone Kim, Juyoung Suk, Shayne Longpre,
Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham
Neubig, Moontae Lee, Kyungjae Lee, and Minjoon
Seo. 2024. Prometheus 2: An open source languagemodel specialized in evaluating other language mod-
els. InProceedings of the 2024 Conference on Empir-
ical Methods in Natural Language Processing, pages
4334–4353, Miami, Florida, USA. Association for
Computational Linguistics.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using GPT-4 with better human
alignment. InProceedings of the 2023 Conference
on Empirical Methods in Natural Language Process-
ing, pages 2511–2522, Singapore. Association for
Computational Linguistics.
LlamaIndex. 2024. LlamaParse: A document pars-
ing service for structured PDF extraction. https:
//www.llamaindex.ai/llamaparse . Ac-
cessed: 2026-05-26.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision
in long form text generation. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 12076–12100, Singa-
pore. Association for Computational Linguistics.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu,
KaShun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. 2024. RAGTruth: A hallucination cor-
pus for developing trustworthy retrieval-augmented
language models. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10862–
10878, Bangkok, Thailand. Association for Compu-
tational Linguistics.
Ankit Pal, Logesh Kumar Umapathi, and Malaikan-
nan Sankarasubbu. 2022. MedMCQA: A large-scale
multi-subject multi-choice dataset for medical do-
main question answering. InProceedings of the Con-
ference on Health, Inference, and Learning, volume
174 ofProceedings of Machine Learning Research,
pages 248–260. PMLR.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond.Foundations and Trends in Information Re-
trieval, 4(1–2):1–174.
Tal Schuster, Sihao Chen, Senaka Buthpitiya, Alex
Fabrikant, and Donald Metzler. 2022. Stretching
sentence-pair NLI models to reason over long doc-
uments and clusters. InFindings of the Association
for Computational Linguistics: EMNLP 2022, pages
394–412, Abu Dhabi, United Arab Emirates. Associ-
ation for Computational Linguistics.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara
Mahdavi, Jason Wei, Hyung Won Chung, Nathan
Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen
Pfohl, Perry Payne, Martin Seneviratne, Paul Gam-
ble, Chris Kelly, Abubakr Babiker, Nathanael Schärli,
Aakanksha Chowdhery, Philip Mansfield, Dina
9

Demner-Fushman, and 13 others. 2023. Large lan-
guage models encode clinical knowledge.Nature,
620(7972):172–180.
George Tsatsaronis, Georgios Balikas, Prodromos
Malakasiotis, Ioannis Partalas, Matthias Zschunke,
Michael R. Alvers, Dirk Weissenborn, Anastasia
Krithara, Sergios Petridis, Dimitris Polychronopou-
los, Yannis Almirantis, John Pavlopoulos, Nico-
las Baskiotis, Patrick Gallinari, Thierry Artiéres,
Axel-Cyrille Ngonga Ngomo, Norman Heino, Eric
Gaussier, Liliana Barrio-Alvers, and 3 others. 2015.
An overview of the BIOASQ large-scale biomedical
semantic indexing and question answering competi-
tion.BMC Bioinformatics, 16(1):138.
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan
Yang, and Ming Zhou. 2020. MiniLM: Deep self-
attention distillation for task-agnostic compression
of pre-trained transformers. InAdvances in Neural
Information Processing Systems, volume 33, pages
5776–5788. Curran Associates, Inc.
John Wennberg and Alan Gittelsohn. 1973. Small
area variations in health care delivery.Science,
182(4117):1102–1108.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, pages 641–649. ACM.
Guangtao Zeng, Wenmian Yang, Zeqian Ju, Yue Yang,
Sicheng Wang, Ruisi Zhang, Meng Zhou, Jiaqi
Zeng, Xiangyu Dong, Ruoyu Zhang, Hongchao Fang,
Penghui Zhu, Shu Chen, and Pengtao Xie. 2020.
MedDialog: Large-scale medical dialogue datasets.
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 9241–9250, Online. Association for Computa-
tional Linguistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judg-
ing LLM-as-a-Judge with MT-Bench and Chatbot
Arena. InAdvances in Neural Information Process-
ing Systems, volume 36, pages 46595–46623. Curran
Associates, Inc.
Lianghui Zhu, Xinggang Wang, and Xinlong Wang.
2025. JudgeLM: Fine-tuned large language models
are scalable judges. InThe Thirteenth International
Conference on Learning Representations.
10

A Question sources and inclusion criteria
The 1,115 released questions were drawn from an initial pool of 3,000+ candidates collected from four
families of public, patient-facing sources. Table 4 reports the top-10 source names in the final benchmark.
Table 4: Top-10 source names for the released question set, by number of contributing questions.
Source N questions
Mayo Clinic patient forums 280
Reddit (transplant subreddits) 268
National Kidney Foundation Q&A 162
American Liver Foundation Q&A 93
Healthy Transplant 29
Inspire (community) 21
HRSA (US Government) 15
RWJBarnabas Health 14
Endocrine Society 13
UCSF Health 12
Source families (final shares): institutional Q&A pages (31.2%), community forums such as Reddit
and Mayo Clinic Connect (25.1%), patient-facing medical organizations (24.9%), and a long tail of
government health agencies and patient advocacy sites (18.8%). 69.9% of questions are geolocated to the
United States.
Collection and inclusion.Candidate questions were harvested from the source platforms above using
transplant- and symptom-keyword search. A candidate was retained if it (a) wasrelevantto transplant
patient education (excluding administrative or off-topic questions) and (b) wasnon-duplicativeof an
earlier-retained question (cosine deduplication at threshold 0.85 followed by manual review of near-
duplicates). Every retained question was thenanonymized and rephrasedto (c) strip personally identifying
information about the asker or named individuals and (d) make the questionself-contained(interpretable
without surrounding conversational context).
B Topic taxonomy
Each question is annotated with one or more of 13 top-level topic categories. Table 5 lists the categories
and their share of the question set (multi-label, percentages can sum to>100%).
Table 5: 13-topic taxonomy for the question set. Multi-label.
Topic Share (%)
Medical Complications 28.6
Reproductive Health 26.1
Lifestyle & Daily Living 20.2
Pre-Transplant 15.8
Medications 9.9
Monitoring & Follow-up 9.7
Mental & Emotional Health 9.6
Surgery & Recovery 7.0
Special Populations & Education 6.5
Transplant Process & Logistics 1.9
Financial & Insurance 0.6
Financial & Administrative 1.7
Support & Community 0.5
A second tier of 43 fine-grained sub-topic tags refines these categories (e.g.,Medications →Tacrolimus
interactions;Reproductive Health →Mycophenolate timing before pregnancy). The sub-topic list is
included in the released annotation file.
C Data card (Datasheet for Datasets)
Following the recommendations of Gebru et al. (2021), we provide a structured data card.
11

Motivation.Created to enable evaluation of medical RAG systems on a corpus with genuine institu-
tional heterogeneity, and to enable analysis of that heterogeneity itself.
Composition.1,115 patient-derived questions; 102 transplant patient-education handbooks from 23
U.S. centres across 5 organ types; 48,056 grounded answers from the reference production run; 5,730,465
pairwise comparisons (1,211,220 LLM-judged, 4,519,245 absence-pre-screened); per-question matrices;
per-shard summaries.
Collection.Questions collected over [date range] from public sources listed in Appendix A. Handbooks
downloaded as PDFs from public institutional websites in 2024–2025. No interaction with patients or
clinicians for data collection.
Preprocessing.Questions lightly paraphrased for anonymisation and self-containment. Handbooks
extracted from PDF via LlamaParse; chunked at section boundaries with 512-token sub-chunking. Answers
and judgments produced by Qwen3-32B at temperature 0.
Uses.Intended for evaluating medical RAG systems’ behaviour under multi-source corpora, for
measuring institutional heterogeneity in patient education, and as a benchmark for new LLM judges.Not
intendedfor ranking individual transplant centres or for direct clinical decision support.
Distribution.Release-location metadata is anonymized for review. Original handbook PDFs are not
redistributed but are listed by URL.
Maintenance.Maintained by the authors, with annual updates planned when new handbook revisions
are detected.
D Answer-generation prompt
The reference production run uses the HERO-QA system prompt (the HERO_QA_SYSTEM_PROMPT
below) paired with the USER_TEMPLATE for evidence framing. The earlier hybrid-retrieval baseline
used a comparable system prompt without the section-citation requirement.
System: You are a clinical information assistant using HERO-QA evidence.
Answer the patient’s question based ONLY on the provided handbook evidence
from this specific transplant center. Follow these rules strictly:
1. If the evidence answers the question, give the answer using only that
evidence.
2. Cite the supporting section heading, and page if provided. If pages are
unknown, cite the section heading only.
3. If the evidence does not answer the question, respond exactly: "NOT
ADDRESSED: This handbook does not contain information on this topic."
4. Do not use outside medical knowledge. Do not fill gaps with general
transplant advice.
User: ## Handbook Context
{context}
## Patient Question
{question}
Generation runs with greedy decoding (temperature 0), max_new_tokens=512 , and
<think>...</think>reasoning blocks stripped before the answer is persisted.
E Judge prompt and output schema
Our judge uses two prompts: a binary absence-detection prompt (used only when the heuristic NOT
ADDRESSEDprefix is not detected) and the main comparison prompt.
Absence-detection prompt.
You are a clinical information assistant. Read the following response that
was generated from a transplant center handbook and determine whether it
effectively states that the handbook does NOT contain information on the
topic.
Response:
{answer}
Does this response indicate the handbook does not address the question?
Answer with exactly one word: YES or NO
12

Comparison prompt.
You are a clinical expert evaluating whether two transplant center handbooks
give consistent guidance on the same patient question.
## Task
Compare Answer A and Answer B and classify their relationship as exactly one
of:
ABSENT / CONSISTENT / COMPLEMENTARY / DIVERGENT / CONTRADICTORY
## Definitions
- ABSENT: One or both answers indicate the handbook does not contain
information on the topic, so no meaningful comparison can be made.
- CONSISTENT: Both answers provide substantive clinical content and give the
same clinical recommendation.
- COMPLEMENTARY: Both answers provide substantive clinical content that is
compatible, but they differ in level of detail.
- DIVERGENT: Both answers provide substantive clinical content but differ
in a clinically meaningful way (different thresholds, timelines, or
recommendations that would lead to different patient behavior).
- CONTRADICTORY: Both answers provide substantive clinical content that
gives directly opposing guidance.
IMPORTANT: If either answer states the handbook does not address the topic,
or provides no substantive clinical content, you MUST classify the pair as
ABSENT.
## Input
Question: {question}
Answer A ({center_a}): {answer_a}
Answer B ({center_b}): {answer_b}
## Output (JSON only, no other text)
{{
"classification": "<label>",
"reasoning": "<2-3 sentence clinical justification>",
"divergence_topic": "<specific sub-topic of divergence, if applicable, else
null>",
"clinical_significance": "<low/medium/high if divergent or contradictory,
else null>"
}}
Output schema and parsing.Judge outputs are parsed as JSON; if parsing fails, a fallback extractor
scans the raw text for a recognised label and assigns the remaining fields to null . Across the 1,211,220
LLM-judged pairs in the reference run, JSON parsing succeeded on>99.5% of calls.
F Compute cost
Production runs used NVIDIA H100 80 GB GPUs on PSC Bridges-2 via a SLURM allocation. Wall-time
figures aggregate the 20 generation shards and 20 comparison shards from the released reference run.
Table 6: Approximate compute cost of the reference production run.
Stage Wall hours GPU-hours (H100-80)
Document extraction (CPU) 12 0
Indexing 8 8
Stage 2: Generation≈130≈130
Stage 3: Comparison≈300≈300
Total≈450≈438
At an indicative H100-80 GB cloud rate of $3–4/hour, the total reference-run cost is approximately
$1.3K–$1.8K. The pipeline is fully resumable: a stalled or pre-empted shard can be re-launched without
recomputing its already-persisted per-question artefacts. Smaller domains (10–20 handbooks) are runnable
on a single H100 in under 24 hours.
13

G Per-handbook coverage heatmap
Figure 3: Handbook ×question-organ absence rate. Rows are the 102 handbooks (grouped and colour-coded by
organ); columns are the six question-organ groups. Red = the handbook is silent on that organ’s questions. The
block-diagonal structure reflects that organ-specific handbooks answer mainly their own-organ and general questions;
rows that are pale across all columns (e.g., several Mayo Clinic, UChicago, Houston Methodist handbooks) are
broadly comprehensive.
14

H System-level delta, 14B vs. 32B
Table 7 reports the per-organ deltas underlying the system-level comparison in §6.4.
Table 7: System delta: 32B reference run (HERO-QA retrieval + 32B judge) −14B earlier run (hybrid retrieval +
14B judge). The pipeline upgrade systematically lowers absence without inflating per-pair divergence; instead, the
prevalenceof divergence rises.
∆rabs ∆R div ∆R con ∆pctany div
general−0.127−0.014 +0.002 +0.270
heart−0.126−0.006−0.010 +0.190
kidney−0.125−0.080 +0.014 +0.097
liver−0.122−0.071 +0.037 +0.085
lung−0.129−0.003 +0.020 +0.124
pancreas−0.188−0.010−0.020 +0.188
mean−0.136−0.031 +0.007 +0.159
I Annotation protocol
The full validation protocol — sample design, annotator-facing rubric with operational tiebreak-
ers, clinical-significance definitions, calibration plan, quality assurance, and scoring metrics —
is provided as supplementary material under drafts/annotation_study/PROTOCOL.md .
The 200-pair stratified sample ( sample_v1/annotation_sample_full.csv ), two shuffled
annotator-facing packets ( packets/annotator_{A,B}.csv ), and the deterministic sampler
(src/analysis/build_annotation_sample.py ) are released alongside the benchmark for
full reproducibility.
15