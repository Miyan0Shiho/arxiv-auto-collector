# How Fine-Grained Should a RAG Benchmark Be? A Hierarchical Framework for Synthetic Question Generation

**Authors**: Chase M. Fensore, Kaustubh Dhole, Jason Fan, Eugene Agichtein, Joyce C. Ho

**Published**: 2026-06-11 01:14:07

**PDF URL**: [https://arxiv.org/pdf/2606.12789v1](https://arxiv.org/pdf/2606.12789v1)

## Abstract
Evaluating retrieval-augmented generation (RAG) systems requires benchmarks that capture diverse question characteristics, yet practitioners lack empirical guidance on which dimensions to vary and at what granularity. We present HieraRAG, a hierarchical framework for studying granularity in RAG benchmark construction, defining optimal granularity as the level that maximizes discriminative power (the standard deviation of generation quality across categories) within a given RAG configuration. As a case study, we generate 5,872 synthetic question-answer (QA) pairs from FineWeb-10BT across 3 dimensions (Question Complexity, Answer Type, Linguistic Variation) at 3 granularity levels (2, 4, and 8 categories). With a BM25+Falcon-3-10B pipeline, optimal granularity varies by dimension: complexity benefits from fine-grained distinctions (discriminative power: 0.053) while answer type and linguistic variation peak at medium granularity. We introduce a Coherence Ratio metric to quantify whether fine-grained splits cleanly subdivide parent categories, revealing structural differences across dimensions (Question Complexity: 0.40 vs. Answer Type: 1.44). Human evaluation of 110 stratified QA pairs confirms synthetic quality. While these specific findings reflect a single configuration, HieraRAG provides a portable procedure and validation metric for practitioners to determine evaluation granularity within their own RAG settings.

## Full Text


<!-- PDF content starts -->

How Fine-Grained Should a RAG Benchmark Be? A Hierarchical
Framework for Synthetic Question Generation
Chase M. Fensore
Department of Computer Science
Emory University
Atlanta, USA
chase.fensore@emory.eduKaustubh Dhole
Department of Computer Science
Emory University
Atlanta, USA
kaustubh.dhole@emory.eduJason Fan
Department of Computer Science
Emory University
Atlanta, USA
jason.fan@emory.edu
Eugene Agichtein
Department of Computer Science
Emory University
Atlanta, USA
eugene.agichtein@emory.eduJoyce C. Ho
Department of Computer Science
Emory University
Atlanta, USA
joyce.c.ho@emory.edu
Abstract
Evaluating retrieval-augmented generation (RAG) systems requires
benchmarks that capture diverse question characteristics, yet prac-
titioners lack empirical guidance on which dimensions to vary and
at what granularity. We presentHieraRAG, a hierarchical frame-
work for studying granularity in RAG benchmark construction,
defining optimal granularity as the level that maximizesdiscrimi-
native power(the standard deviation of generation quality across
categories) within a given RAG configuration. As a case study, we
generate 5,872 synthetic question–answer (QA) pairs from FineWeb-
10BT across 3 dimensions (Question Complexity, Answer Type,
Linguistic Variation) at 3 granularity levels (2, 4, and 8 categories).
With a BM25+Falcon-3-10B pipeline, optimal granularity varies
by dimension: complexity benefits from fine-grained distinctions
(discriminative power: 0.053) while answer type and linguistic varia-
tion peak at medium granularity. We introduce aCoherence Ratio
metric to quantify whether fine-grained splits cleanly subdivide par-
ent categories, revealing structural differences across dimensions
(Question Complexity: 0.40 vs. Answer Type: 1.44). Human evalu-
ation of 110 stratified QA pairs confirms synthetic quality. While
these specific findings reflect a single configuration, HieraRAG pro-
vides a portable procedure and validation metric for practitioners
to determine evaluation granularity within their own RAG settings.
Keywords
retrieval-augmented generation, RAG evaluation, synthetic ques-
tion generation, question answering
1 Introduction
Retrieval-augmented generation (RAG) has become the dominant
approach for question-answering (QA) systems, grounding lan-
guage model outputs in retrieved evidence to improve factual ac-
curacy [ 8]. As organizations deploy RAG over proprietary cor-
pora, from enterprise documentation to scientific literature, eval-
uating system performance requires benchmarks that capture di-
verse question characteristics. Recent work demonstrates that di-
versity matters for RAG: diverse retrieved content improves answer
quality [ 21], and diverse instruction data enhances model capabil-
ities [ 10]. But as synthetic QA generation becomes more popularfor RAG benchmarking [ 4,6], a fundamental question remains: at
whatlevel of granularity should question characteristics be varied?
QA evaluation has evolved from factoid extraction [ 19] toward
multi-hop reasoning [ 22], diverse answer types [ 23], and varied
linguistic formulations [ 1]. This reveals natural dimensions along
which questions vary: complexity (single- vs. multi-hop reasoning),
answer type (factoid vs. abstractive), and linguistic variation (vo-
cabulary alignment, phrasing diversity). Yet existing benchmarks
treat these dimensions with inconsistent granularity. HotpotQA
categorizes by answer type (Person, Date, Yes/No) but provides
limited complexity control [ 22]. KILT spans task formats but lacks
systematic linguistic variation [ 14]. Should we distinguish only
between “simple" and “complex" questions, or use finer categories
like “factoid," “multi-hop," “reasoning," and “comparative"? Finer
distinctions may reveal nuanced performance differences, but they
increase generation cost and risk redundant categories. Without
empirical guidance, designers risk under-sampling failure modes
or diluting evaluation signal with redundant distinctions.
We address this gap throughHieraRAG, a hierarchical frame-
work for synthetic RAG QA benchmark construction that systemat-
ically varies questions along 3 illustrative dimensions: (1) Question
Complexity (QC), (2) Answer Type (AT), and (3) Linguistic Variation
(LV). Each dimension is evaluated at 3 granularity levels: coarse (2
categories), medium (4), and fine (8). Unlike prior work that varies
answer granularity while keeping questions fixed [ 23], we vary
question characteristics themselves to probe how RAG systems re-
spond to realistic query diversity. To validate whether fine-grained
splits are well-structured (i.e., cleanly subdividing parent categories
rather than introducing unrelated constraints), we introduce a co-
herence ratio metric analogous to the clustering silhouette coeffi-
cient [ 16]. We generate 5,872 synthetic questions from the FineWeb-
10BT corpus [ 13], evaluate them on a BM25+Falcon-3-10B pipeline.
To ensure synthetic question quality and validate category assign-
ments, 2 annotators independently evaluate 110 QA pairs for cor-
rectness, answerability, and category alignment. Through our ex-
periments, we investigate:
•RQ1:Which dimensions most differentiate RAG perfor-
mance?
•RQ2:Does finer granularity reveal insights or add noise?
•RQ3:Are fine-grained splits well-structured?
1arXiv:2606.12789v1  [cs.CL]  11 Jun 2026

HieraRAG, March 2026, Atlanta, GA, USA Chase M. Fensore, Kaustubh Dhole, Jason Fan, Eugene Agichtein, Joyce C. Ho
HIERARCHICAL QUESTION CATEGORISATION 3 DIMENSIONS × 3 GRANULARITY LEVELS (2 / 4 / 8)
QC
Question
Complexity
Cognitive demand to
answer, from fact
lookup to multi-hop
reasoning.COARSE · 2Simple Complex
MEDIUM · 4Single-fact Multi-fact local Cross-section synthesis Comparative analysis
FINE · 8Extractive
spanEntity
extractionMulti-span
aggregationParaphrasing
requiredSingle-hop
inferenceMulti-hop
reasoningComparative
synthesis
(entities)Comparative
synthesis
(concepts)
AT
Answer Type
Expected response
format, from direct
extraction to
synthesised generation.COARSE · 2Extractive Abstractive
MEDIUM · 4Short span extraction List or enumerationSummary or
explanationSynthesis or analysis
FINE · 8 Entity
extractionPhrase
extractionUnordered
listOrdered
sequenceCondensed
summarySentence
extractionAnalytical
synthesisExplanatory
synthesis
LV
Linguistic
Variation
Lexical alignment of
question to document,
from verbatim to
paraphrased.COARSE · 2Similar to document Distant from document
MEDIUM · 4Exact terminology Partial overlap Synonym substitution Conceptual rephrase
FINE · 8Verbatim
terminologyHigh
lexical
overlapModerate
lexical
overlapModerate-
low lexical
overlapLow lexical
overlapSynonym-
based
rephraseDomain
shift
terminologyAbstraction
level shift
*"Phrasing" dimension (RQ1) has 4 categories, only at the coarse level (concise-natural, verbose-natural, short-query, long-query), and is not shown in this hierarchy. RQ1
examines dimensions; RQ2 examines granularity; RQ3 examines the LV × QC interaction.WORKED EXAMPLE DOCUMENT →  GENERATED QUESTIONS
DOCUMENT
Pharmaceutical recall notice for losartan
“Thousands of bottles of a prescription medicine for treating high blood pressure
... have been recalled over the detection of trace amounts of a possible
carcinogen.”
CATEGORIES QUESTION
QC
COARSE
SimpleWhat should patients do if they are currently taking
losartan tablets that might be recalled?
QC
FINE
Extractive spanWhat should patients do before stopping their losartan
medication after hearing about the recall?
LV
COARSE
Distant from
documentWhat should people do if they are currently taking blood
pressure medication that might be contaminated?
LV
FINE
Abstraction level
shiftIs it safe to immediately stop taking losartan if it’s part of
the recall?
Figure 1: Hierarchical structure of three question dimensions (QC, AT, LV) across three granularity levels. Each dimension
subdivides from coarse (2 categories) to medium (4) to fine (8). User Expertise set to “Novice” for example QAs shown.
Our findings reveal that optimal granularity varies by dimen-
sion: the complexity dimension benefits from fine-grained distinc-
tions (8 categories,discriminative power—the standard deviation of
generation quality across categories at a given granularity level—
reaches 0.053) while answer type and linguistic variation peak at
medium granularity (4 categories). The Coherence Ratio reveals
structural differences across dimensions (QC: 0.40 vs. AT: 1.44), ex-
plaining why some dimensions benefit from finer splits while others
plateau. Human evaluation confirms high synthetic quality (98%
acceptable) while low consensus on fine-grained categories (29%
agreement) suggests semantic ambiguity. Preliminary correlation
analysis ( 𝑟𝑝𝑏=0.24) indicates that our data-driven Coherence Ratio
aligns with this human perception, offering a potential automated
proxy for separating categories.
We contribute: (1) a hierarchical framework for determining op-
timal evaluation granularity in RAG benchmarks, (2) a Coherence
Ratio metric for validating hierarchical question category struc-
tures, (3) empirical evidence—within a single BM25+Falcon-3-10B
configuration—that different question characteristics may require
different granularity levels for diagnostic evaluation, and (4) a
dataset of 5,872 hierarchically-organized questions with code for
replication.1
2 Methods
2.1 Question Categorization Framework
To systematically evaluate RAG systems across diverse single-turn
question settings with a predefined corpus, we adopted a hierar-
chical categorization framework capturing 3 key dimensions of
question variation (Figure 1):
(1)Question Complexity (QC)measures the cognitive de-
mand required to answer questions, ranging from simple
fact extraction to multi-hop reasoning.
1https://github.com/fensorechase/rag-diverse-benchmarks-synthetic-qa(2)Answer Type (AT)indicates the expected response format,
distinguishing between questions requiring direct extrac-
tion of information versus those requiring synthesis or gen-
eration of new formulations. This dimension builds upon
prior work exploring answer type and granularity [23].
(3)Linguistic Variation (LV)quantifies the lexical alignment
between question phrasing and document content, rang-
ing from verbatim terminology to paraphrased concepts
requiring semantic understanding.
For each dimension, we define categories at 3 granularity levels:
coarse (2 categories), medium (4), and fine (8). Each fine-grained
category specializes its parent medium category, which subdivides
a coarse category. Categories within each level are mutually ex-
clusive for generation, though our coherence analysis reveals se-
mantic overlap in practice. This structure lets us assess whether
finer distinctions reveal additional performance variations, helping
practitioners choose appropriate evaluation granularity.
Single-dimension assignment as a methodological choice.
For RQ1 and RQ2, each generated question is assigned to a single di-
mension during synthesis. We adopt this design to isolate the effect
of one dimension at a time, but we acknowledge two consequences.
First, real questions can be characterized along all three dimensions
simultaneously; the single-dimension assignment is a simplification,
not a property of natural questions. Second, when generating along
one dimension, the other two dimensions become uncontrolled
confounders that can shift retrieval and generation performance
independently of the dimension under study, which could compress
or amplify the discriminative power reported (e.g., a QC-controlled
batch may incidentally contain more distant-vocabulary questions,
lowering its mean MAP). RQ3’s factorial design (§3.3) probes one
such interaction. Post-hoc multi-label tagging via a few-shot classi-
fier is a natural extension.
2

HieraRAG, March 2026, Atlanta, GA, USA Chase M. Fensore, Kaustubh Dhole, Jason Fan, Eugene Agichtein, Joyce C. Ho
2.2 Synthetic Question Generation
We generated 5,872 questions using DataMorgana [ 4], a tool for
creating diverse synthetic QA benchmarks by leveraging Claude
3.5 Sonnet.2Questions were generated from documents randomly
sampled from FineWeb-10BT [ 13], a web corpus with 10B tokens
and∼15M documents. Each question includes a DataMorgana-
generated reference answer and source document ID to evaluate
retrieval and generation quality.
User expertise as a held variable.Across all three research
questions, user expertise was randomly specified at generation
time (50% novice, 50% expert) so that observed performance differ-
ences reflect question characteristics rather than user-level vari-
ation. Studying interactions between user persona and question
dimensions (QC, AT, LV) is a natural extension for future work.
RQ1:We generated 1,600 questions (400 per dimension) using
coarse-level categories from4 dimensions: QC, AT, LV, and Ques-
tion Phrasing (4 coarse categories only). [ 5,20]. Each question
belongs to one dimension, enabling independent evaluation. Our
dimensions were informed by established QA datasets [ 1,7,14,22]
and RAG evaluation frameworks [2].
RQ2:We generated 3,272 questions across 3 dimensions (QC,
AT, LV) at 3 granularity levels (coarse/medium/fine). We focus on
these three because they exhibit hierarchical structure suitable
for multi-level subdivision. Sample sizes per category vary due to
generation stochasticity but provide sufficient power ( 𝑛≥38per
fine category).
RQ3:We generated 1,000 questions in a 2×2 factorial design,
crossing LV (similar/distant) with QC (simple/complex), yielding
∼250 questions per cell.
2.3 RAG System Configuration
We evaluate a standard two-stage RAG pipeline on an NVIDIA H100.
Retrieval:We indexed 512-token chunks of FineWeb-10BT [ 13]
using PyTerrier [ 11] and retrieved 𝑘= 10documents via BM25 .
Generation:We employed Falcon-3-10B-Instruct [17] (temp =
0.6) to generate answers from retrieved contexts, instructing the
model to refuse if information was insufficient.
2.4 Evaluation Metrics
We evaluate the RAG system along two dimensions: retrieval quality
and generation quality [2, 3].
Retrieval:Mean Average Precision (MAP) as the primary metric,
nDCG@10, and Recall@10 measure whether ground-truth docu-
ments appear in top-10 results.
Generation:Cosine similarity (CS) as a primary metric (MiniLM-
L6-v2 embeddings) [ 15], ROUGE-1 [ 9], BLEU [ 12] assess answer
quality against the reference answer.
Discriminative Power:For each granularity level, we compute
the standard deviation of cosine similarity scores across categories.
Higher standard deviation indicates that categories reveal mean-
ingful performance distinctions, justifying finer granularity.
Information Content:We compute normalized mutual infor-
mation (MI) between category assignments and performance bins.
2See our supplemental code, [ 4], for details on synthetic QA generation; we note
our framework may also leverage other pipelines for synthetic QA generation [6].Higher MI indicates category assignments are more informative
about system performance [18].
Hierarchical Calibration:Within a dimension, to validate that
children categories meaningfully subdivide their parents, we intro-
duce a novelCoherence Ratioinspired by the silhouette coefficient
in clustering [ 16]. Asplitis one parent-children grouping in the hier-
archy (e.g., the medium category summary_or_explanation split-
ting into the fine categories condensed_summary andsentence
_extraction ). Each dimension contains 6 splits in total: 2 coarse-
to-medium splits (each coarse parent has 2 medium children) and 4
medium-to-fine splits (each medium parent has 2 fine children). For
a given corpus and set of questions across nested categories, this
metric quantifies whether sibling categories provide non-redundant
answer evaluation signals while remaining vertically consistent,
and is defined as:
𝜌coherence =𝜎sib
𝛿vert+𝜖,(1)
where 𝜎sibis the standard deviation of CS across sibling categories
(horizontal discrimination), 𝛿vertis the absolute difference between
parent and mean child CS (vertical consistency).3High 𝜌(𝜌> 2.0)
indicates discriminative but aligned siblings (preferred); low 𝜌(𝜌<
1.0) suggests poor hierarchical structure (Figure 2).
2.5 Human Validation
Two annotators independently evaluated 110 QA pairs stratified
across dimensions and granularity levels. Annotators rated an-
swer correctness (0-3), hallucination severity (0-2), and question
answerability (0-3) for all questions. For a subset of 60 questions, an-
notators additionally validated fine-grained category assignments.
Inter-annotator agreement for answer correctness was moderate
(Fleiss’ 𝜅=0.47). Reference answers proved highly reliable: 98.2%
were acceptable (score≥2) and 90.9% were hallucination-free.
For category validation, annotators achieved 28.6% combined
agreement (exact or hierarchical parent-match), significantly above
the 12.5% random baseline, but consistent with the inherent diffi-
culty of consistently applying fine-grained definitions manually.
Filice et al. [ 4] note similar challenges during DataMorgana valida-
tion and rely primarily on automated faithfulness filtering rather
than fine-grained human category agreement, suggesting that low
inter-annotator consensus on fine categories is a known property
of synthetic QA benchmarks rather than a flaw specific to our
framework. Practically, low agreement implies that fine-grained
per-categoryperformance estimates carry larger effective error bars
than coarse-level estimates, and should be interpreted as relative
orderings rather than absolute values.
3 Experimental Results
3.1 RQ1: Most Discriminative QA Dimensions
To determine which aspects of question variation most strongly
differentiate RAG performance, we evaluated 4 dimensions inde-
pendently using coarse-level categories (Table 1).
LV emerged as the most discriminative dimension (range=0.077)
despite achieving the worst absolute performance (MAP=0.369,
CS=0.695), while AT achieved the best performance (MAP=0.506,
CS=0.711) yet showed lower discrimination (range=0.059).
3𝜖=0.001prevents division by zero.
3

HieraRAG, March 2026, Atlanta, GA, USA Chase M. Fensore, Kaustubh Dhole, Jason Fan, Eugene Agichtein, Joyce C. Ho
Dimension n MAP CS Range CS Rank
LV 400 .369 .695.0771
Phrasing 400 .503 .703 .069 2
AT 400.506 .711.059 3
QC 400 .478 .704 .010 4
Table 1: RQ1: Coarse-level dimension comparison. Each di-
mension uses 2-4 coarse categories. Range =max(CS)−
min(CS) across those categories within a dimension. Note
that Range≠DiscPow in Table 2; CS = Cosine Similarity.
Dimension Gran. n MAP CS NMI DiscPowΔ
QCC (2) 333 .518.712.008 .007 –
M (4) 333.564.709 .027 .0354.8×
F (8) 446 .562 .708.040 .053 1.5×
ATC (2) 333 .491 .692 .028 .024 –
M (4) 333.514 .699.026 .044 1.8×
F (8) 382 .504 .684.035 .037 -16%
LVC (2) 333 .412 .672 .049 .039 –
M (4) 333 .343 .671.054 .047 1.2×
F (8) 446.446 .699.020 .031 -33%
Table 2: RQ2: Discriminative power and performance over
granularity levels. QC=Question Complexity; AT=Answer
Type; LV=Linguistic Variation; C=Coarse (2 cat.); M=Medium
(4 cat.); F=Fine (8 cat.). DiscPow= 𝜎(CS) across categories;
NMI=norm. mutual info.Δ: change from parent level.
3.2 RQ2: Impact of Categorization Granularity
We evaluated whether increasing categorical granularity reveals
additional performance distinctions (Table 2). Discriminative power
(DiscPow) is measured as the standard deviation of cosine similarity
scores across categories within a granularity level; higher DiscPow
indicates that member categories provide greater diagnostic resolu-
tion for identifying system capabilities and limitations.
QCshowed monotonic improvement across all three levels (Dis-
cPow from 0.007 at coarse to 0.053 at fine), with a 4.8 ×gain from
coarse to medium and an additional 1.5 ×gain at fine granular-
ity. This indicates that fine-grained complexity distinctions (e.g.,
single-hop inference vs. multi-hop reasoning) capture meaningful
performance variation beyond coarse simple/complex categoriza-
tion. In contrast,ATandLVpeaked at medium granularity (0.044
and 0.047 respectively) and declined at fine (0.037 and 0.031). This
suggestsdiminishing returns beyond 4 categoriesfor these
dimensions. Here, finer distinctions add noise rather than signal.
The normalized MI analysis confirms these patterns: QC increases
monotonically but LV peaks at medium.
Hierarchical calibration analysisof the 6 splits per dimension
reveals that fine-grained splits are not uniformly well-structured
across dimensions (mean 𝜌coherence =0.73±0.89). AT shows the best-
calibrated split forsummary_or_explanation(coherence=3.31), where
children (condensed summary,sentence extraction) discriminate
strongly while aligning with their parent. Conversely, QC showsDimension Category n MAP CS
QCcomparative_synthesis_concepts 50.566 .766
single_hop_inference 59 .495.755
entity_extraction 62.606.636
extractive_span 55 .523 .619
ATexplanatory_synthesis 48 .395.738
condensed_summary 44 .349.725
phrase_extraction 46.513.647
ordered_sequence 38.547.625
LVabstraction_level_shift 57.451 .729
domain_shift_terminology 60.333 .722
synonym_based_rephrase 59 .293 .683
low_lexical_overlap 61 .225 .624
Table 3: RQ2 Fine-grained: Top-2 and bottom-2 performing
categories per dimension by generation quality (CS).
poor calibration (mean coherence=0.40) despite achieving the high-
est discriminative power at fine granularity (DiscPow=0.053). While
the 8 fine QC categories successfully partition performance space,
they do not cleanly subdivide their medium-level parents.
This also explains why AT’s discriminative powerdecreasesat
fine granularity (from 0.044 to 0.037). Though there is one high-
quality split (summary_or_explanation), the other three AT medium-
to-fine splits have low mean coherence (0.82), creating fine cate-
gories that cluster together rather than expanding performance
coverage. LV’sconceptual_rephrasesplit exhibits the poorest cal-
ibration, with high vertical deviation ( 𝛿vert=0.086) indicating fine
children (domain shift terminology, abstraction level shift) stray
substantially from their parent’s performance profile. These poorly
calibrated splits introduce constraints beyond simple subdivision,
consistent with hierarchy not being strictly taxonomic. Validating
𝜌coherence , we observed a positive correlation ( 𝑟𝑝𝑏=0.24, 𝑝= 0.21)
between a split’s Coherence Ratio and human agreement, suggest-
ing that𝜌 coherence is a promising proxy for semantic separation.
3.2.1 Which Categories Are Hardest?Within fine granularity, we
observe substantial performance variation (Table 3). For example
in QC, extractive span questions (CS=0.619) underperform compar-
ative synthesis (concepts) (CS=0.766) by 24%.
Notably, several "complex" children yield higher RAG CS than
"simple" children (e.g., comparative_synthesis_concepts CS=0.766
vs. extractive_span CS=0.619), and distant-vocabulary categories
achieve high generation scores (CS=0.722–0.729) despite poor re-
trieval (MAP=0.225–0.333).
3.3 RQ3: Interaction Between LV and QC
To assess whether vocabulary mismatch amplifies complexity ef-
fects, we conducted a 2×2 factorial experiment on LV×QC (Table 4).
LV shows a strong main effect ( Δ=0.076, 13% relative), while
complexity shows a weak effect ( Δ=0.018, 3%). The vocabulary gap
is consistent across complexity levels: similar vocabulary helps
complex questions by +0.035 CS, while distant vocabulary shows
no complexity effect (+0.0). This suggestsadditive rather than
interactive effects. Here, LV and QC are largely independent.
4

HieraRAG, March 2026, Atlanta, GA, USA Chase M. Fensore, Kaustubh Dhole, Jason Fan, Eugene Agichtein, Joyce C. Ho
LV QC n MAP CS Gap
Similar Simple 252 .547 .637 –
Similar Complex 265 .566 .672 +.035
Distant Simple 265 .237 .578 –
Distant Complex 218 .164 .578 +.000
Main effects:
Similar (pooled) – 517 .557 .655 –
Distant (pooled) – 483 .201 .578–.077
Table 4: RQ3: LV×QC interaction. Gap shows difference (sim-
ple to complex) within LV level.
WELL-CALIBRATED SPLIT ρ = 3.31
PARENT
(MED.)summary_or_explanation
CS = .705
δ_vert
= .008
CHILDREN
(FINE)
sentence_extraction
CS = .670condensed_summa ry
CS = .725
σ_sib
= .027
POORLY-CALIBRATED SPLIT ρ = 0.04
PARENT
(MED.)conceptual_rephrase
CS = .639
δ_vert
= .086
CHILDREN
(FINE)
σ_sib
= .004
domain_shift_terminology (.722)
abstraction_level_shift (.729)
CS →
.60 .65 .70 .75
Figure 2: Demonstrated Coherence Ratio ( 𝜌) calculation for
two medium to fine-grained splits within AT and LV dimen-
sions. 𝜌> 2.0indicates discriminative-yet-aligned children
(preferred);𝜌<1.0suggests poor hierarchical structure.
The interaction pattern reveals that when vocabulary is mis-
matched, complexity becomes irrelevant because retrieval has al-
ready failed (MAP=0.164–0.237 for distant vocabulary vs. 0.547–0.566
for similar). Complexity distinctions only matter when the system
successfully retrieves relevant documents.
4 Discussion and Conclusion
Our hierarchical framework reveals that optimal evaluation gran-
ularity varies by dimension. QC benefits from fine-grained dis-
tinctions (8 categories) while AT and LV peak at medium (4 cat-
egories). This challenges one-size-fits-all benchmark design anddemonstrates how single-level evaluation can obscure critical pat-
terns, as seen where LV dominated at coarse granularity (RQ1), yet
QC showed strongest discriminative power hierarchically (RQ2).
The additive effects of vocabulary and complexity (RQ3) also sup-
port independent dimension design.
Calibration for Validation.We introduce the Coherence Ratio
to validate whether fine splits are well-structured. High coherence
(e.g., AT’ssummary_or_explanation: 3.31) indicates discriminative
yet aligned children, whereas low coherence (e.g., QC mean: 0.40)
reveals redundant distinctions. The correlation between Coherence
Ratio and human agreement ( 𝑟𝑝𝑏=0.24) suggests that our metric
captures true semantic split boundaries. Practitioners can use𝜌<
1.0as a signal to refine category definitions or reduce granularity
before large-scale QA benchmark generation.
Generalization and Limitations.Several limitations should be
acknowledged during interpretation of our results: first, all experi-
ments use a single retriever ( BM25 ), generator ( Falcon-3-10B ), and
corpus (FineWeb-10BT). The specific granularity findings should
be read as illustrative for this configuration rather than as general
properties of RAG benchmarking; BM25 ’s lexical sensitivity likely
amplifies the LV signal at coarse level (Table 2), and a dense or hy-
brid retriever may shift which dimensions are most discriminative.
The portable contribution is HieraRAG’s procedure (hierarchical
design, discriminative power, Coherence Ratio), not the category-
level outcomes. Second, single-dimension assignment in RQ1/RQ2
leaves the other two dimensions as uncontrolled confounders that
can inflate or deflate estimated discriminative power; this could be
addressed using full-factorial coverage with post-hoc multi-label
tagging via few-shot LLM classifiers. Third, correlation of Coher-
ence Ratio with human agreement is positive but non-significant
(𝑟𝑝𝑏=0.24,𝑝=0.21,𝑛=60); we treat it as a useful diagnostic that re-
quires further validation. Finally, because properties of synthetic
queries differ from human ones, [ 24], future work should evaluate
granularity findings against human-written benchmarks.
Acknowledgments
This work is supported by the National Science Foundation (NSF)
grant IIS-2145411 and CISE Graduate Fellowships under Grant No.
2313998. Any opinions, findings, and conclusions or recommenda-
tions expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.
References
[1]Valeriia Bolotova, Vladislav Blinov, Falk Scholer, W. Bruce Croft, and Mark
Sanderson. 2022. A Non-Factoid Question-Answering Taxonomy. InProceedings
of the 45th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’22). Association for Computing Machinery, New
York, NY, USA, 1196–1207. doi:10.1145/3477495.3531926
[2] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large
language models in retrieval-augmented generation. InProceedings of the AAAI
Conference on Artificial Intelligence, Vol. 38. 17754–17762.
[3] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs:
Automated Evaluation of Retrieval Augmented Generation. InProceedings of
the 18th Conference of the European Chapter of the Association for Computational
Linguistics: System Demonstrations. 150–158.
[4] Simone Filice, Guy Horowitz, David Carmel, Zohar Karnin, Liane Lewin-Eytan,
and Yoelle Maarek. 2025. Generating Diverse Q&A Benchmarks for RAG Evalua-
tion with DataMorgana. arXiv:2501.12789 [cs] doi:10.48550/arXiv.2501.12789
[5] Manish Gupta and Michael Bendersky. 2015. Information Retrieval with Verbose
Queries.Foundations and Trends in Information Retrieval9, 3-4 (2015), 209–354.
5

HieraRAG, March 2026, Atlanta, GA, USA Chase M. Fensore, Kaustubh Dhole, Jason Fan, Eugene Agichtein, Joyce C. Ho
[6] Jeffrey Ip and Kritin Vongthongsri. 2025. deepeval. https://github.com/confident-
ai/deepeval original-date: 2023-08-10T05:35:04Z.
[7] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee,
Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M.
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Transactions of the Association
for Computational Linguistics7 (2019), 452–466. doi:10.1162/tacl_a_00276 Place:
Cambridge, MA Publisher: MIT Press.
[8]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[9] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries.
InText summarization branches out. 74–81.
[10] Wanlong Liu, Junying Chen, Ke Ji, Li Zhou, Wenyu Chen, and Benyou Wang.
2025. Rag-instruct: Boosting llms with diverse retrieval-augmented instructions.
InProceedings of the 2025 Conference on Empirical Methods in Natural Language
Processing. 3865–3888.
[11] Craig Macdonald, Nicola Tonellotto, Sean MacAvaney, and Iadh Ounis. 2021.
PyTerrier: Declarative experimentation in Python from BM25 to dense retrieval.
InProceedings of the 30th acm international conference on information & knowledge
management. 4526–4533.
[12] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a
method for automatic evaluation of machine translation. InProceedings of the
40th annual meeting of the Association for Computational Linguistics. 311–318.
[13] Guilherme Penedo, Hynek Kydlíček, Anton Lozhkov, Margaret Mitchell, Colin A
Raffel, Leandro Von Werra, Thomas Wolf, et al .2024. The fineweb datasets:
Decanting the web for the finest text data at scale.Advances in Neural Information
Processing Systems37 (2024), 30811–30849.
[14] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani,
Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021. KILT: a Bench-
mark for Knowledge Intensive Language Tasks. doi:10.48550/arXiv.2009.02252
arXiv:2009.02252 [cs].
[15] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP). Association for Computational
Linguistics, 3982.
[16] Peter J Rousseeuw. 1987. Silhouettes: a graphical aid to the interpretation and
validation of cluster analysis.Journal of computational and applied mathematics
20 (1987), 53–65.
[17] Falcon-LLM Team. 2024. The Falcon 3 Family of Open Models. https:
//huggingface.co/blog/falcon3
[18] Nguyen Xuan Vinh, Julien Epps, and James Bailey. 2010. Information theoretic
measures for clusterings comparison: Variants, properties, normalization and
correction for chance.Journal of Machine Learning Research11 (2010), 2837–2854.
[19] Ellen M. Voorhees and D. M. Tice. 2000. The TREC-8 Question Answering
Track Evaluation.NIST3 (May 2000). https://www.nist.gov/publications/trec-
8-question-answering-track-evaluation Last Modified: 2017-02-17T13:34-05:00
Publisher: Ellen M. Voorhees, D M. Tice.
[20] QianYing Wang, Clifford Nass, and Jiang Hu. 2005. Natural language query vs.
keyword search: Effects of task complexity on search performance, participant
perceptions, and preferences. InIFIP Conference on Human-Computer Interaction.
Springer, 106–116.
[21] Zhichao Wang, Bin Bi, Yanqi Luo, Sitaram Asur, and Claire Na Cheng. 2025.
Diversity Enhances an LLM’s Performance in RAG and Long-context Task.arXiv
preprint arXiv:2502.09017(2025).
[22] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. doi:10.48550/arXiv.1809.
09600 arXiv:1809.09600 [cs].
[23] Gal Yona, Roee Aharoni, and Mor Geva. 2024. Narrowing the Knowledge Evalu-
ation Gap: Open-Domain Question Answering with Multi-Granularity Answers.
InProceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
6737–6751. doi:10.18653/v1/2024.acl-long.365
[24] Oleg Zendel, Sara Fahad Dawood Al Lawati, Lida Rashidi, Falk Scholer, and Mark
Sanderson. 2025. A Comparative Analysis of Linguistic and Retrieval Diversity
in LLM-Generated Search Queries. InProceedings of the 34th ACM International
Conference on Information and Knowledge Management. 4014–4023.
6