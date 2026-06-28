# MMed-Bench-IR: A Heterogeneous Benchmark for Multilingual Medical Information Retrieval

**Authors**: Junhyeok Lee, Han Jang, Hyeonjin Goh, Kyu Sung Choi

**Published**: 2026-06-23 06:41:13

**PDF URL**: [https://arxiv.org/pdf/2606.24200v1](https://arxiv.org/pdf/2606.24200v1)

## Abstract
Retrieval-augmented generation (RAG) in clinical settings increasingly requires multilingual retrieval against predominantly English evidence corpora. Multilingual medical retrieval demands three capabilities: cross-lingual alignment, concept discrimination, and evidence retrieval. However, existing benchmarks evaluate these only in isolation, leaving the interaction between biomedical expertise and multilingual coverage unmeasured. We introduce MMed-Bench-IR, a benchmark designed to disentangle these axes across 6 languages and three structurally heterogeneous tasks: (1) cross-lingual medical QA retrieval with 6,127 queries grounded in the Unified Medical Language System (UMLS), (2) concept discrimination over 4,975 confusion sets at three difficulty tiers, and (3) multilingual evidence retrieval for RAG with 2,040 quality-assured queries. The three tasks share zero concept and query overlap by design, ensuring that aggregate scores reflect genuine capability breadth. Evaluation of ten systems across six paradigm families reveals severe cross-lingual failure: biomedical encoders that score 0.818 nDCG@10 in English drop to 0.056 in Japanese, a gap that English-only benchmarks cannot detect.

## Full Text


<!-- PDF content starts -->

MMed-Bench-IR: A Heterogeneous Benchmark for Multilingual Medical
Information Retrieval
Junhyeok Lee1,2,†, Han Jang1,3,†, Hyeonjin Goh3, Kyu Sung Choi1,2,3,∗
1Seoul National University2Seoul National University College of Medicine3Seoul National University Hospital
†Equal contribution∗Corresponding author
ent1127@snu.ac.kr
Abstract
Retrieval-augmented generation (RAG) in clin-
ical settings increasingly requires multilingual
retrieval against predominantly English evi-
dence corpora. Multilingual medical retrieval
demands three capabilities: cross-lingual align-
ment, concept discrimination, and evidence re-
trieval. However, existing benchmarks evaluate
these only in isolation, leaving the interaction
between biomedical expertise and multilingual
coverage unmeasured. We introduceMMed-
Bench-IR, a benchmark designed to disentan-
gle these axes across 6 languages and three
structurally heterogeneous tasks: (1) cross-
lingual medical QA retrieval with 6,127 queries
grounded in the Unified Medical Language Sys-
tem (UMLS), (2) concept discrimination over
4,975 confusion sets at three difficulty tiers,
and (3) multilingual evidence retrieval for RAG
with 2,040 quality-assured queries. The three
tasks share zero concept and query overlap by
design, ensuring that aggregate scores reflect
genuine capability breadth. Evaluation of ten
systems across six paradigm families reveals se-
vere cross-lingual failure: biomedical encoders
that score 0.818 nDCG@10 in English drop
to 0.056 in Japanese, a gap that English-only
benchmarks cannot detect.
1 Introduction
Large language models are increasingly adopted
in healthcare (Sahni and Carrus, 2023), and the
quality of their clinical responses depends heav-
ily on the retrieval systems that supply supporting
evidence (Qiu et al., 2024; Wang et al., 2024b).
However, these retrieval systems are predominantly
built for and evaluated in English, creating a grow-
ing gap for non-English-speaking populations. Lan-
guage barriers are associated with reduced access
to care, higher rates of adverse events, and poorer
health outcomes across clinical settings (Joo et al.,
2023), and recent work warns that AI-powered clin-
ical tools risk deepening these inequities by exhibit-
ing significant performance disparities between
(a) English -only Biomedical Retrieval  
(b) Limited Multilingual 
(c) Proposed: MMed -Bench -IR[EN]
Task 1: 
    QA
Task 2: 
    Concept 
Task3: 
    RAG[JA]    心房異所性頻拍 [RU]
Level 1
 synonymLevel 2 
sibling Level 3
Related
[EN]Frequent side effects 
of metformin are 
nausea, abdominal 
discomfort 
and diarrhoea .[EN]Drug -induced 
asthmaDrug -induced 
bronchospasm
[EN] [ES]¿Complicaciones  
de la diabetes?Diabetes mellitus
: a chronic…
[RU]           Ринит ???
[EN]Atrial Tachyscardia  
Ectopic[ES]Taquicardia  
Auricular Ectópica
[EN]Adverse effects 
of metformin
[JA] メトホルミ
ンの副作用EKTOPI
CHESKAIA .. ✓
MedicalMultilingual
Multi Task ✓
✓MedicalMultilingual
Multi Task
MedicalMultilingual
Multi Task✘
✘
✓
△
✘
△Figure 1:Motivation.(a) English-only biomedical
benchmarks miss non-English queries. (b) Existing
multilingual medical benchmarks cover 2–3 languages
and one task. (c) MMed-Bench-IR evaluates all three
axes across 6 languages and 3 writing systems.
well-resourced and digitally under-represented lan-
guages (Anyaegbuna et al., 2026; Ortega et al.,
2025).
A key bottleneck lies in the retrieval stage: em-
bedding models and dense retrievers that power
these systems often lack the multilingual align-
ment needed to handle diverse languages (Yuan
et al., 2022; Liu et al., 2021), yet the field has no
way to measure this shortcoming systematically.
This multilingual medical retrieval problem decom-
poses into three distinct capabilities: cross-lingual
alignment so that the same concept in different
languages maps to similar representations, con-
cept discrimination so that clinically confusable
entities such as Type 1 versus Type 2 diabetes
are correctly separated, and evidence retrieval so
that queries in any language can surface relevant
English-language passages. These three capabili-
ties are individually well studied (Liu et al., 2021;
Remy et al., 2023; Zhang et al., 2023), yet no ex-
isting benchmark evaluates them jointly (ThakurarXiv:2606.24200v1  [cs.CL]  23 Jun 2026

et al., 2021; Muennighoff et al., 2023), leaving the
field unable to measure whether progress on one
dimension comes at the cost of another.
Prior work addresses each axis only partially:
biomedical encoders such as SapBERT (Liu et al.,
2021) and BioLORD-2023 (Remy et al., 2023)
evaluate only in English, while multilingual bench-
marks such as MIRACL (Zhang et al., 2023) lack
medical concept grounding. Recent efforts in
cross-lingual medical retrieval (Athar Sheikh et al.,
2025; Acharya et al., 2025) cover limited language
pairs or target multimodal rather than text-only re-
trieval. The intersection of multilingual coverage
and biomedical specialization remains uncharted
(Figure 1; Section 2).
To address these gaps, we introduceMMed-
Bench-IR, a structured evaluation suite spanning
6 languages (English, Spanish, French, Japanese,
Chinese, Russian) and 3 writing systems (Latin,
Cyrillic, CJK). Inspired by the Benchmarking Infor-
mation Retrieval (BEIR) suite (Thakur et al., 2021),
which showed that heterogeneous evaluation across
diverse tasks and domains yields insights invisible
in any single benchmark, we design MMed-Bench-
IR around the same principle applied to multilin-
gual medical information retrieval.
It comprises three structurally heterogeneous
tasks: cross-lingual medical question answer-
ing (QA) retrieval with 6,127 queries grounded
in the Unified Medical Language System (UMLS)
Metathesaurus ontology (Bodenreider, 2004), con-
cept discrimination over 4,975 confusion sets orga-
nized into three difficulty levels, and multilingual
evidence retrieval for retrieval-augmented gener-
ation (RAG) with 2,040 quality-assured queries
against an 80,049-passage English corpus. By con-
struction, the three tasks share zero annotated con-
cept overlap, zero query overlap, and minimal cor-
pus vocabulary overlap, ensuring that aggregate
scores reflect genuine capability breadth rather than
mastery of a single skill.
Our contributions are as follows:
•To the best of our knowledge, MMed-Bench-
IR is the first benchmark to jointly evaluate
multilingual alignment and biomedical spe-
cialization across 6 languages, 3 writing sys-
tems, and three tasks with zero concept and
query overlap.
•We evaluate ten systems across six paradigm
families and reveal a consistent paradigm-
level hierarchy across all tasks, with conceptdiscrimination emerging as the hardest lexical
bottleneck at a +0.38 gap between BM25 and
the best dense model.
•We expose severe cross-lingual failure modes
invisible in English-only evaluation: biomed-
ical encoders that score 0.818 nDCG@10 in
English collapse to 0.056 in Japanese, with a
fairness gap of 0.76.
2 Related Work and Scope
Multilingual and biomedical retrieval models.
Recent work has made significant progress on mul-
tilingual and biomedical retrieval independently,
but the two axes remain largely disconnected.
MMedC (Qiu et al., 2024) and Apollo (Wang et al.,
2024b) curate multilingual medical corpora for
LLM training but provide no retrieval evaluation.
On the retriever side, BGE-M3 (Chen et al., 2024)
and Multilingual-E5 (Wang et al., 2024a) sup-
port over 100 languages but lack medical concept
grounding, while BioLORD-2023 (Remy et al.,
2023) and MedCPT (Jin et al., 2023) advance
biomedical entity representation but evaluate only
in English. UMLS-based (Bodenreider, 2004) en-
tity linking work such as SapBERT (Liu et al.,
2021) and CODER (Yuan et al., 2022) tests concept
alignment but not passage retrieval. No existing
model jointly addresses both axes.
Retrieval benchmarks.Several benchmarks
have advanced retrieval evaluation along individ-
ual axes, but none jointly targets multilingual and
biomedical capabilities. BEIR (Thakur et al., 2021)
established that in-domain performance does not
predict out-of-domain generalization, motivating
heterogeneous evaluation. MTEB (Muennighoff
et al., 2023) demonstrated that no single embedding
method dominates across diverse tasks, reinforc-
ing the need for multi-task assessment. However,
neither targets the biomedical domain. On the mul-
tilingual side, MIRACL (Zhang et al., 2023) and
Mr. TyDi (Zhang et al., 2021) cover diverse lan-
guages but do not incorporate medical terminology
or ontology structure. Despite these advances, no
existing benchmark can evaluate the interaction
between multilingual coverage and biomedical spe-
cialization.
Cross-lingual medical retrieval.A small but
growing body of work has begun to address
cross-lingual retrieval in the medical domain di-
rectly. The MUCHMORE project (V olk et al.,

Task1
Cross -lingual Medical QA RetrievalTask2
Concept DiscriminationTask3
Multilingual  Evidence Retrieval(RAG)
RetrieverQuery
[EN] 
[JA] Lip and/ or oral cavity cancer
口唇および口腔 内癌… 
[EN] Atrial Tachycardia Ectopic
[ES] EKTOPICHESKAIA 
PREDSERDNAIA ..
TargetLevel 1
Synonym
[EN] Drug -indued asthmaLevel 2
Sibling
[EN] Bowel infarctionLevel 3
Related 
[EN] ADP 
Phosphocreatine 
PhosphotransferaseQuery
[EN] 
[ZH] What is the most common 
EGFR mutation 
in glioblastoma multiforme?
什么是最常见的EGFR 突变
在多形质母细胞瘤 ?EN     Latin
ES     Latin
FR     Latin
RU   Cyrillic
JA      CJK
ZH      CJKC0039234[qid] 
[qid] 
[qid] 
[qid] C0039234C0861542Target Target Target Target
[JA] 薬物誘発性喘息
[RU] Астма, вызванная 
лекарственным  ..
[ES] Asma inducida 
por fármacos[JA] 腸梗塞
[RU] Инфаркт 
кишечника
[ES] Infarto intestinal[JA] CKアイソザイム
[RU] KREATIN
FOSFOKINAZA
[ES] Creatina  
FosfoquinasaIn this study we determined 
whether EGFR amplification 
and expression of the most common 
mutation in GBMs (EGFRvIII) 
is retained at tumor recurrence.
6 Languages
ResultMMed -IR =  𝟏
𝟑 (Task1 score + Task2 score + Task3 score)
fairness gap: 𝒎𝒂𝒙 𝒍𝐧𝐃𝐂𝐆@𝟏𝟎 𝒍−𝒎𝒊𝒏 𝒍𝐧𝐃𝐂𝐆@𝟏𝟎 𝒍per task  ( 𝑙: language )C0861542
𝐓𝐚𝐬𝐤 𝟏𝒔𝒄𝒐𝒓𝒆 =𝐦𝐞𝐚𝐧 𝐧𝐃𝐂𝐆 @𝟏𝟎,𝐑@𝟏     |     𝐓𝐚𝐬𝐤 𝟐𝒔𝒄𝒐𝒓𝒆 =𝐦𝐞𝐚𝐧 (𝐑@𝟏  𝐩𝐞𝐫 𝐭𝐢𝐞𝐫 )     |     𝐓𝐚𝐬𝐤 𝟑𝒔𝒄𝒐𝒓𝒆 =𝐑@𝟓
Figure 2:Overview of MMed-Bench-IR. Three tasks target distinct retrieval capabilities across 6 languages with
zero query and CUI overlap. Scores aggregate into MMed-IR and per-task fairness gaps.
2002) explored UMLS-based cross-language in-
formation retrieval (CLIR) over English and
German medical abstracts, establishing medical
CLIR as a research direction. More recently,
CURE (Athar Sheikh et al., 2025) evaluates point-
of-care passage ranking across English, French-
to-English, and Spanish-to-English conditions but
covers only three languages and a single retrieval
task. M3Retrieve (Acharya et al., 2025) targets
multimodal medical retrieval rather than multi-
lingual text-only information retrieval. MMed-
Bench-IR differs by jointly evaluating concept
alignment, concept discrimination, and evidence
retrieval within a single benchmark spanning 6 lan-
guages and 3 writing systems.
3 Benchmark Design Principles
We design MMed-Bench-IR around the principle
that benchmark value comes from principled het-
erogeneity rather than dataset count, and organize
it along four selection criteria.
Capability heterogeneity.The three tasks isolate
distinct retrieval capabilities: cross-lingual concept
alignment (Task 1), fine-grained semantic discrimi-
nation (Task 2), and cross-lingual passage retrieval
(Task 3). This separation ensures that a model’s
aggregate score cannot be driven by strength in a
single capability.
Linguistic heterogeneity.Six languages span
three language families (Indo-European [en, es, fr,
ru], Sino-Tibetan [zh], Japonic [ja]), three writingsystems (Latin, Cyrillic, CJK), and varying levels
of medical NLP resource availability. This ensures
that benchmark scores reflect genuine multilingual
robustness rather than performance on closely re-
lated languages.
Difficulty heterogeneity.Task 1 spans 500 med-
ical concepts across 5 languages with varying
UMLS coverage, creating natural difficulty gra-
dients across languages. Task 2 includes three dif-
ficulty tiers (synonym, sibling, related-but-distinct)
validated by multi-encoder consensus. Task 3 uses
quality-assured translations with an 83.7% reten-
tion rate. The resulting score distributions span a
5×range across systems, with no task saturated.
Validation heterogeneity.Each task uses a dif-
ferent validation strategy targeting a different er-
ror source: UMLS ontology grounding (Task 1),
multi-encoder majority vote (Task 2), and bilingual
concept-fidelity and back-translation QA (Task 3).
This ensures that no single validation blind spot
propagates across tasks.
4 Benchmark Construction and
Characterization
4.1 Benchmark Statistics
Table 1 and Figure 2 summarize the benchmark.
The three tasks are structurally complementary:
they share zero CUI overlap between Task 1 and
Task 2 concept spaces, and the pairwise corpus vo-
cabulary overlap is low (Jaccard <0.14 between
Tasks 1/2, <0.02 between either and Task 3).

Table 1: MMed-Bench-IR benchmark statistics. The three tasks differ structurally in retrieval setting, corpus scale,
query granularity, and relevance definition, ensuring that aggregate scores reflect capability breadth.
Task 1: QA Task 2: Concept Task 3: RAG
Retrieval setting Q lang→C multi Qterm→C pool Qmulti→C en
Query languages 5 2 (en, zh) 6
Corpus language Multilingual Multilingual English
# Queries 6,127 4,143 2,040
# Corpus docs 2,552 6,340 80,049
Avg. query len.
(words)5.0 4.2 6.7
Avg. doc len. (words) 5.7 5.2 30.1
Avg. pos. per query 1.9 1.5 32.9
Relevance definition CUI alignment Same-CUI synonym BioASQ gold
Validation method UMLS grounding 3-encoder vote Concept fidelity + BT
Primary metric mean(nDCG, R@1) mean(R@1 per tier) R@5
Query granularity ranges from short concept terms
(4 to 5 words) to biomedical questions (6.7 words),
and corpus scale spans two orders of magnitude
(2,552 to 80,049 documents).
4.2 Task Construction
Task 1: Cross-lingual Medical QA Retrieval.
From MMedBench (Qiu et al., 2024) test-split mul-
tilingual medical QA data, we extract question
stems (without answer options) as queries across
5 languages (French is excluded because fewer than
50 queries passed the CUI-tagging pipeline at this
granularity). Each query is tagged with a UMLS
CUI via a cascading tagger that first attempts exact
lexical match against UMLS preferred terms and
synonyms, then falls back to a biomedical linker
for unmatched queries. Queries sharing the same
CUI across languages form cross-lingual positive
groups. The corpus consists of concept definition
texts (1,290 positives + 1,262 hard-negative con-
cepts from non-matching CUIs), yielding 6,127
queries and 2,552 corpus documents across 500
concepts.
Task 2: Concept Discrimination.From the
UMLS 2025AB Metathesaurus concept file (MR-
CONSO.RRF) (Bodenreider, 2004), we parse 1.5M
concept atoms grouped by source vocabulary and
construct confusion sets at three tiers: Tier 1 (syn-
onym) contains the same CUI with different sur-
face forms or languages. Tier 2 (sibling) pairs
different CUIs sharing the same source vocabu-
lary and 3-character code prefix, meaning they are
clinically related but distinct (e.g., ICD-10 codes
under the same category). Tier 3 (related) pairs dif-
ferent CUIs from the same source vocabulary but
different code prefix groups, meaning they belong
to the same medical domain but are clearly dis-tinct. Queries are drawn from English and Chinese
surface forms, and positives are different surface
forms of the same CUI in any language, with the
query’s own form excluded from the corpus. All
6,340 corpus terms across multiple languages form
a shared retrieval pool. Because each confusion set
generates one query per unique CUI it contains, the
4,975 validated sets yield 4,143 evaluation queries.
Task 3: Multilingual Evidence Retrieval for
RAG.340 English biomedical questions from
BioASQ 13b (Tsatsaronis et al., 2015) with 80,049
evidence snippets are translated to 5 target lan-
guages using NLLB-200-3.3B (Costa-Jussà et al.,
2022). Translation quality is assured via two
checks. The first is concept fidelity, where each
translation is verified against a 587K-entry UMLS
bilingual lexicon plus a set of 87 protected medical
abbreviations (e.g., HIV , MRI, HbA1c) that must
be preserved. The second is back-translation con-
sistency, where each translation is back-translated
via NLLB-200 and compared with the original us-
ing token-overlap F1. Queries passing at least one
check form the official evaluation subset (1,708 of
2,040, 83.7%).
4.3 Heterogeneity Analysis
Figure 3 confirms that the three tasks are struc-
turally independent. V ocabulary overlap between
Task 3 (biomedical passages) and either term-level
task is below 0.02 Jaccard, reflecting fundamen-
tally different retrieval regimes. The language dis-
tributions differ by design: Task 1 reflects natural
UMLS coverage (English-dominant), Task 2 cap-
tures the two most distinct scripts (en, zh), and
Task 3 is uniformly balanced. Zero queries over-
lap between any two tasks, and zero annotated
CUIs overlap between Tasks 1 and 2 (Task 3 uses

T ask 1
QAT ask 2
ConceptT ask 3
RAGT ask 1
QA
T ask 2
Concept
T ask 3
RAG
0.140 0.015
0.140
 0.012
0.015 0.012
(a) Inter-task Vocabulary Overlap
(Jaccard Index)
EN ES FR JA RU ZH0.00.20.40.60.81.0Query Proportion(b) Language Distribution
per Task
T ask 1
T ask 2
T ask 3
0.000.050.100.150.20
JaccardFigure 3:Benchmark heterogeneity.(a) Inter-task cor-
pus vocabulary overlap is low (Jaccard ≤0.14), confirm-
ing that the three tasks target distinct lexical domains.
(b) Language distributions vary across tasks: Task 1 is
English-dominant due to UMLS coverage, Task 2 uses
English and Chinese only, and Task 3 is uniformly bal-
anced across all six languages.
BioASQ questions without CUI annotation). These
properties ensure that a model’s aggregate perfor-
mance cannot be inflated by excelling on a single
lexical domain or language.
4.4 Validation
Benchmark validity rests on three controls, each
targeting a different error source.
Task 1.Ontology-grounded positives via UMLS
CUI alignment ensure face validity.
Task 2.Tier definitions are anchored in UMLS
ontology structure (synonym, sibling, related-but-
distinct), providing an encoder-independent cri-
terion. A 3-encoder majority vote using BGE-
M3 (Chen et al., 2024), E5-large (Wang et al.,
2024a), and SapBERT (Liu et al., 2021) opera-
tionalizes acceptance thresholds based on pairwise
cosine similarity (Tier 1: ≥0.70, Tier 2: 0.40–0.75,
Tier 3: 0.15–0.45), requiring agreement from at
least two of three encoders. This process rejects
967 sets (16.3%) and re-tiers 1,597, yielding 4,975
validated sets. To verify that the tier ordering is not
biased toward the validators, we confirm that BM25
and BioLORD-2023 (Remy et al., 2023), neither of
which participated in validation, exhibit the same
monotonic Tier 1, Tier 2, Tier 3 ordering (Table 3),
with tier boundaries confirmed by non-overlapping
confidence intervals (Appendix A).
Task 3.Two-stage translation quality assurance
based on concept fidelity and back-translation con-
sistency retains 83.7% of queries. Full thresholds
and per-language breakdowns are in Appendix A.4.5 Scoring
The primary scoreMMed-IRis the macro average
of three task scores:
MMed-IR=1
3 
Task1 score+Task2 score+Task3 score
(1)
where Task 1 score =mean(nDCG@10, R@1),
Task 2 score =mean(R@1 Tier1, R@1 Tier2,
R@1 Tier3), and Task 3 score =R@5. The fair-
ness gap for each task is max ℓ(nDCG@10 ℓ)−
min ℓ(nDCG@10 ℓ)across languages ℓ, providing
a uniform cross-lingual equity measure.
5 Experimental Setup
5.1 Baselines
We evaluate ten systems spanning six paradigm
families to characterize the benchmark’s discrimi-
native properties:
•Lexical: BM25 (Robertson and Zaragoza,
2009) (Okapi, whitespace tokenization; no
CJK-specific segmentation, which disadvan-
tages BM25 on Japanese/Chinese).
•Biomedical dense: SapBERT (Liu et al.,
2021) and BioLORD-2023 (Remy et al.,
2023), English-centric biomedical encoders.
•Multilingual dense: BGE-M3 (Chen et al.,
2024) and E5-large (Wang et al., 2024a),
general-purpose encoders supporting over
100 languages.
•Late-interaction: ColBERT-XM (Louis et al.,
2025), a multilingual late-interaction retriever
based on X-MOD (Pfeiffer et al., 2022) with
per-token scoring, trained on MS-MARCO.
•Hybrid: BM25 (Robertson and Zaragoza,
2009) and BGE-M3 (Chen et al., 2024) com-
bined via reciprocal rank fusion (RRF, k=60 ).
•Two-stage reranker: BGE-M3 (Chen et al.,
2024) retrieval followed by BGE-reranker-v2-
m3 cross-encoder reranking (top-30).
•Multilingual and medical: MMed-Embed
(Section 5.2) and MMed-Embed with
Reranker, within-distribution references.
MMed-Embed combines both biomedical special-
ization and multilingual coverage, providing an
approximate within-distribution ceiling.

Table 2: MMed-Bench-IR main results with query-level bootstrap 95% confidence intervals ( n=2,000 ). All
paradigm-level differences exceed CI widths. Best inbold, second-best underlined .†Within-distribution.
Model Task 1 Task 2 Task 3 MMed-IR Family
Zero-shot
BM25 .091±.005.076±.006.056±.008.075±.004Lexical
BioLORD .188±.006.213±.012.088±.008.163±.005Bio-dense
SapBERT .212±.006.282±.010.107±.009.201±.005Bio-dense
BM25+BGE-M3 .173±.005.124±.007.191±.012.163±.005Hybrid
ColBERT-XM .268±.007.372±.009.244±.013.295±.006Late-int
BGE-M3 .344±.007.432±.010.264±.013.347±.006Multi-dense
E5-large .335±.008.447 ±.012.262±.013.348±.007Multi-dense
BGE-M3+Reranker .387±.007.433±.010.294 ±.014.371±.006Two-stage
Within-distribution†
MMed-Embed .388±.007.454 ±.011.267±.013.369±.006Multi+Med
MMed-Embed+Rer..404±.007.436±.010.291±.014.377±.006Multi+Med+R
5.2 Reference Model: MMed-Embed
To assess the benchmark’s discriminative range
above existing baselines, we fine-tune BGE-
M3 (Chen et al., 2024) (568M parameters, XLM-
RoBERTa (Conneau et al., 2020) backbone) on
137K training rows from MMedBench (Qiu et al.,
2024) (40K, 6 languages), UMLS (Bodenreider,
2004) cross-lingual pairs (87K, 5 languages), and
xBioASQ (10K). No overlap exists with benchmark
test sets (verified: 0% query overlap on Task 3;
<0.5% trivial term overlap on Tasks 1/2).
Training objective. Lretis temperature-scaled
cross-entropy over cosine similarities between each
query and all in-batch positives plus k=15 hard
negatives, with temperatureτ=0.02.
Training procedure.We fine-tune BGE-M3 on
the full 137K rows with 15 random in-batch nega-
tives per query for 10 epochs (effective batch 1,024,
lr2×10−6, 1,330 steps) using AdamW with linear
warmup (10%) and linear decay, mean pooling with
L2 normalization, max sequence length 256, gra-
dient checkpointing, and mixed precision (bf16)
on 8×B200 GPUs. A 2 ×2 ablation (Appendix B)
shows that neither topology-aware margin loss nor
ANCE-style hard negative mining (Xiong et al.,
2020) yields statistically significant gains over this
baseline recipe; the primary benefit comes from
domain-specific contrastive fine-tuning on multi-
lingual medical data. Full hyperparameters are in
Appendix B.
Results.MMed-Embed achieves 0.369 MMed-
IR, a marginal +0.022 over the zero-shot BGE-
M3 baseline (0.347). Combining MMed-Embed
with cross-encoder reranking yields the best overall
score (0.377) and the lowest fairness gap (0.170),showing that domain-specific first-stage retrieval
and cross-encoder reranking are complementary.
The modest gain from 137K-row fine-tuning sug-
gests that the benchmark’s primary challenge lies
in architecture-level multilingual alignment rather
than in training data or loss design; MMed-Embed
scores should be interpreted as an approximate
within-distribution ceiling rather than a fair zero-
shot comparison.
6 Results and Analysis
6.1 Overall Comparison
Table 2 presents the main results across all ten
systems. The spread from BM25 (0.075) to the
best system (0.377) spans a 5 ×range, confirming
effective discrimination across paradigm families,
with substantial headroom remaining.
6.2 Key Findings
Finding A: Consistent hierarchy across
paradigms and tasks.All three tasks produce
the same paradigm ordering: lexical methods
trail biomedical-only encoders, which in turn
trail multilingual dense retrievers (Table 2). This
ordering holds across all six paradigm families,
including ColBERT-XM (0.295), which falls
between biomedical encoders (SapBERT 0.201)
and multilingual dense retrievers (BGE-M3 0.347),
consistent with its English-centric contrastive
training. Hybrid RRF (0.163) underperforms
standalone dense retrieval, while cross-encoder
reranking (0.371) and MMed-Embed with
Reranker (0.377) occupy the top tier. This
consistency across tasks with zero query and CUI
overlap cannot be attributed to data leakage.

Table 3: Task 2 Recall@1 by difficulty tier with query-
level bootstrap 95% CIs.
Model Tier 1 Tier 2 Tier 3
(Synonym) (Sibling) (Related)
Zero-shot
BM25 .172±.014.053±.009.004±.006
BioLORD .449±.018.166±.017.025±.025
SapBERT .606±.018.233±.020.008±.010
BM25+BGE-M3 .281±.016.087±.013.004±.006
ColBERT-XM .799±.014.309±.021.008±.010
BGE-M3 .892 ±.010.384±.022.019±.017
E5-large .892 ±.010.419 ±.022.029 ±.028
BGE-M3+Rer. .862±.011.406±.020.031±.021
Within-distribution
MMed-Embed.893±.010.435±.021.033±.022
MMed+Rer. .866±.011.419±.021.023±.017
Finding B: Concept discrimination is the hard-
est lexical bottleneck.Task 2 most sharply sep-
arates paradigms: BM25 scores 0.076 versus the
best model at 0.454, a +0.38 gap (Table 2). Hybrid
RRF degrades Task 2 performance (0.124 vs. 0.432
for BGE-M3 alone), confirming that lexical sig-
nal is counterproductive for concept discrimination.
Tier 3 (related-but-distinct concepts) remains a gen-
uine frontier: even the strongest model reaches
only 3.6%, and ColBERT-XM’s token-level scor-
ing yields only 0.8% (Table 3).
Finding C: Biomedical specialization does not
rescue cross-lingual failure.SapBERT achieves
0.818 nDCG@10 in English but collapses to
0.056 in Japanese (Table 4), with a fairness gap
of 0.76 ±0.04 versus 0.24 ±0.05 for BGE-M3, a
3×equity difference with non-overlapping CIs.
ColBERT-XM exhibits the same pattern: its Task 3
fairness gap (0.37) exceeds BGE-M3 (0.24), con-
firming that late-interaction scoring alone does
not rescue cross-lingual retrieval when contrastive
training is English-centric. Cross-encoder rerank-
ing achieves the lowest fairness gap (0.18 ±0.04).
Full per-task gaps are in Appendix C.
6.3 Evaluation Impact
Each finding requires the multi-axis evaluation that
MMed-Bench-IR provides: the consistent hierar-
chy emerges only from multi-task comparison, the
lexical bottleneck only from concept-level retrieval,
and the cross-lingual collapse only from joint mul-
tilingual and biomedical evaluation.7 Benchmark Validity and Bias Analysis
Annotation bias.All three tasks use automated
relevance judgments rather than human annota-
tions, eliminating inter-annotator disagreement
but introducing potential systematic biases. For
Task 2, tier definitions are anchored in encoder-
independent UMLS ontology relations, and both
BM25 and BioLORD-2023 (neither of which par-
ticipated in validation) exhibit the same monotonic
tier ordering (Table 3). A leave-one-validator-out
analysis confirms ≥85.4% tier stability across all
encoder-pair conditions (Appendix A).
Translation bias.Task 3 uses NLLB-200 ma-
chine translation, which may introduce system-
atic quality differences across languages. Model
ranking is perfectly preserved between the official
(1,708 queries) and all-translated (2,040) subsets
(Spearman ρ=1.0 ), confirming that QA filtering
does not introduce ranking artifacts (Appendix A).
LLM-based quality audit.We conduct a 600-
sample quality audit using Claude Opus 4.7 (An-
thropic, 2026) across all 12 task ×language cells
(50 per cell; Table 5). Task 1 CUI alignment accu-
racy is ≥0.98 across all languages; Task 2 Tier-1
synonym accuracy is 1.00. For Task 3 translations,
ES achieves 1.00, FR 0.86, RU 0.92, while JA and
ZH drop to 0.80 and 0.78 respectively, consistent
with limited UMLS coverage for non-Latin scripts
(Section 9). On a 50-query French subset, an author
independently evaluated translation fidelity under
six predefined medical-translation criteria; inter-
rater agreement with the LLM judge was 98% (Co-
hen’s κ=0.92 ; Gwet’s AC1 =0.97 ), with the LLM
judge identifying 7 of 8 human-flagged errors.
8 Release and Societal Considerations
Evaluation scripts, benchmark splits, baseline
results, and construction code are provided as
supplementary material and will be hosted on
HuggingFace Datasets. Reproducing from raw
sources requires UMLS 2025AB (NLM license)
and BioASQ 13b (registration); the released pack-
age includes all derived artifacts needed for evalua-
tion. All data derives from licensed sources with no
patient-identifiable information. The fairness gap
metric operationalizes cross-lingual equity; bench-
mark scores measure system capability, not clinical
safety.

Table 4: Task 3 nDCG@10 by query language with bootstrap 95% CIs and fairness gap. Gap CIs computed via
bootstrap of max−min across languages.
Model EN ES FR JA ZH RU Gap
Zero-shot
BM25 .518±.04.082±.02.122±.03.063±.02.036±.02.110±.03.482±.04
BioLORD .755±.04.358±.04.500±.04.062±.02.081±.02.067±.02.693±.04
SapBERT .818±.03.452±.05.501±.04.056±.02.095±.03.113±.03.762±.04
BM25+BGE-M3 .731±.03.498±.03.533±.03.369±.03.330±.03.453±.03.401±.04
ColBERT-XM .870±.02.795±.03.757±.03.527±.05.502±.04.747±.04.369±.05
BGE-M3 .855±.03.836±.03.823±.03.679±.04.620±.04.798±.03.235±.05
E5-large .893 ±.02.840±.03.827±.03.659±.04.499±.04.814±.03.394±.05
BGE-M3+Reranker .895 ±.02.876 ±.02.876 ±.02.772 ±.04.715 ±.04.860 ±.03.180 ±.04
Within-distribution
MMed-Embed .853±.03.830±.03.821±.03.688±.04.632±.04.787±.03.221±.05
MMed-Embed+Rer..892±.02.867±.03.869±.03.776±.04.722±.04.847±.03.170±.04
Table 5: LLM-based quality audit accuracy per task
and language ( n=50 per cell). Evaluator: Claude Opus
4.7 (Anthropic, 2026).
Task EN ES FR JA ZH RU
T1 (CUI align.) .98 1.00 — .98 1.00 .98
T2 (Tier-1 syn.) 1.00 — — — 1.00 —
T3 (Translation) — 1.00 .86 .80 .78 .92
9 Conclusion
In this work, we introduce MMed-Bench-IR, a het-
erogeneous benchmark that jointly evaluates mul-
tilingual alignment and biomedical specialization
in medical information retrieval across 6 languages
and 3 writing systems. Its three structurally distinct
tasks share zero query and concept overlap, ensur-
ing that aggregate scores reflect genuine capability
breadth rather than mastery of a single skill. Evalu-
ation of ten systems across six paradigm families
reveals consistent paradigm-level rankings, identi-
fies concept discrimination as the hardest lexical
bottleneck, and exposes the collapse of biomed-
ical models on non-Latin scripts, failure modes
detectable only through multi-axis evaluation. We
hope that MMed-Bench-IR serves as a foundation
for developing retrieval systems that work equi-
tably across languages in clinical settings.
Limitations
MMed-Bench-IR provides the first joint evalua-
tion of multilingual alignment and biomedical spe-
cialization in medical retrieval, but several design
choices constrain the current version.
Task 3 results for Japanese and Chinese should
be interpreted as conservative estimates. An LLM-
based audit (Table 5) confirms that 78 to 80%of JA/ZH translations preserve clinical meaning,
while ES achieves 100% and FR achieves 86%,
consistent with limited UMLS coverage for non-
Latin scripts. Future versions could benefit from
ensemble translation or human post-editing for CJK
languages.
The Task 2 corpus of 6,340 terms constrains neg-
ative discrimination difficulty, particularly at Tier 3
where only 325 confusion sets are available. This
limits the statistical power of Tier 3 evaluation, as
reflected in the wide bootstrap confidence intervals
(Table 3). Scaling the confusion set corpus with
additional UMLS source vocabularies is a priority
for future versions.
Six languages exclude Arabic, Hindi, and other
high-population languages with significant health-
care needs, though the construction framework
is designed for community extension. Task 1
excludes French because fewer than 50 French
queries passed the CUI-tagging pipeline at the 500-
concept granularity. French is retained in Task 3
via NLLB-derived translations.
MMed-Embed is the only within-distribution
system evaluated, and the current baselines do not
include recent LLM-based embedding models (e.g.,
GritLM, E5-Mistral). Community submissions
would strengthen evaluation and broaden paradigm
coverage beyond the six families tested here.
All three tasks rely on automated relevance judg-
ments rather than expert human annotation. While
Section 7 demonstrates high agreement between
LLM-based audits and human evaluation ( κ=0.92 ),
systematic biases may still exist for underrepre-
sented languages where reference resources are
sparse. Incorporating expert clinician review across
all six languages remains an important direction
for future iterations.

References
Arkadeep Acharya, Akash Ghosh, Pradeepika Verma,
Kitsuchart Pasupa, Sriparna Saha, and Dr Priti Singh.
2025. M3Retrieve: Benchmarking multimodal re-
trieval for medicine. InProceedings of the 2025 Con-
ference on Empirical Methods in Natural Language
Processing, pages 15263–15276, Suzhou, China. As-
sociation for Computational Linguistics.
Chukwuebuka Anyaegbuna, Natasha Steele,
April Shichu Liang, Stephen P Ma, Ivan Lopez,
Nymisha Chilukuri, Kavita Patel, Kevin Schulman,
and Jonathan H Chen. 2026. Artificial intelligence
translation in healthcare: an urgent call for evidence-
informed policy frameworks.BMJ Health & Care
Informatics, 33(1):e102007.
Nadia Athar Sheikh, Daniel Buades Marcos, Anne-
Laure Jousse, Akintunde Oladipo, Olivier Rousseau,
and Jimmy Lin. 2025. Cure: A dataset for clinical
understanding & retrieval evaluation. InProceedings
of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining V . 2, pages 5270–5277.
Olivier Bodenreider. 2004. The unified medical lan-
guage system (umls): integrating biomedical termi-
nology.Nucleic acids research, 32(suppl_1):D267–
D270.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2024,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Alexis Conneau, Kartikay Khandelwal, Naman Goyal,
Vishrav Chaudhary, Guillaume Wenzek, Francisco
Guzmán, Edouard Grave, Myle Ott, Luke Zettle-
moyer, and Veselin Stoyanov. 2020. Unsupervised
cross-lingual representation learning at scale. InPro-
ceedings of the 58th annual meeting of the associa-
tion for computational linguistics, pages 8440–8451.
Marta R Costa-Jussà, James Cross, Onur Çelebi, Maha
Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe
Kalbassi, Janice Lam, Daniel Licht, Jean Maillard,
and 1 others. 2022. No language left behind: Scaling
human-centered machine translation.arXiv preprint
arXiv:2207.04672.
Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau,
Lana Yeganova, W John Wilbur, and Zhiyong Lu.
2023. Medcpt: Contrastive pre-trained transformers
with large-scale pubmed search logs for zero-shot
biomedical information retrieval.Bioinformatics,
39(11):btad651.
Hyundeok Joo, Alicia Fernandez, Elizabeth C Wick,
Gala Moreno Lepe, and Solmaz P Manuel. 2023.
Association of language barriers with perioperative
and surgical outcomes: a systematic review.JAMA
Network Open, 6(7):e2322743.Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco
Basaldella, and Nigel Collier. 2021. Self-alignment
pretraining for biomedical entity representations. In
Proceedings of the 2021 conference of the North
American chapter of the association for computa-
tional linguistics: human language technologies,
pages 4228–4238.
Antoine Louis, Vageesh Kumar Saxena, Gijs Van Dijck,
and Gerasimos Spanakis. 2025. Colbert-xm: A mod-
ular multi-vector representation model for zero-shot
multilingual information retrieval. InProceedings of
the 31st International Conference on Computational
Linguistics, pages 4370–4383.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and
Nils Reimers. 2023. Mteb: Massive text embedding
benchmark. InProceedings of the 17th Conference
of the European Chapter of the Association for Com-
putational Linguistics, pages 2014–2037.
Pilar Ortega, Alyssia Miller De Rutté, and Mónica Vela.
2025. Language equity in health technology for pa-
tients with non–english language preference.JAMA
Network Open, 8(2):e2457424.
Jonas Pfeiffer, Naman Goyal, Xi Victoria Lin, Xian Li,
James Cross, Sebastian Riedel, and Mikel Artetxe.
2022. Lifting the curse of multilinguality by pre-
training modular transformers. InProceedings of
the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies, pages 3479–3495,
Seattle, United States. Association for Computational
Linguistics.
Pengcheng Qiu, Chaoyi Wu, Xiaoman Zhang, Weixiong
Lin, Haicheng Wang, Ya Zhang, Yanfeng Wang, and
Weidi Xie. 2024. Towards building multilingual lan-
guage model for medicine.Nature Communications,
15(1):8384.
François Remy, Kris Demuynck, and Thomas De-
meester. 2023. Biolord-2023: Semantic textual rep-
resentations fusing llm and clinical knowledge graph
insights.arXiv preprint arXiv:2311.16075.
Stephen Robertson and Hugo Zaragoza. 2009.The prob-
abilistic relevance framework: BM25 and beyond,
volume 4. Now Publishers Inc.
Nikhil R Sahni and Brandon Carrus. 2023. Artificial
intelligence in us health care delivery.New England
Journal of Medicine, 389(4):348–358.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. InThirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 2).
George Tsatsaronis, Georgios Balikas, Prodromos
Malakasiotis, Ioannis Partalas, Matthias Zschunke,
Michael R Alvers, Dirk Weissenborn, Anastasia

Krithara, Sergios Petridis, Dimitris Polychronopou-
los, and 1 others. 2015. An overview of the bioasq
large-scale biomedical semantic indexing and ques-
tion answering competition.BMC bioinformatics,
16(1):138.
Martin V olk, Bärbel Ripplinger, Špela Vintar, Paul
Buitelaar, Diana Raileanu, and Bogdan Sacaleanu.
2002. Semantic annotation for concept-based cross-
language medical information retrieval.Interna-
tional Journal of Medical Informatics, 67(1-3):97–
112.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024a. Multilin-
gual e5 text embeddings: A technical report.arXiv
preprint arXiv:2402.05672.
Xidong Wang, Nuo Chen, Junyin Chen, Yidong Wang,
Guorui Zhen, Chunxian Zhang, Xiangbo Wu, Yan
Hu, Anningzhe Gao, Xiang Wan, and 1 others. 2024b.
Apollo: A lightweight multilingual medical llm to-
wards democratizing medical ai to 6b people.arXiv
preprint arXiv:2403.03640.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang,
Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold
Overwijk. 2020. Approximate nearest neighbor neg-
ative contrastive learning for dense text retrieval.
arXiv preprint arXiv:2007.00808.
Zheng Yuan, Zhengyun Zhao, Haixia Sun, Jiao Li, Fei
Wang, and Sheng Yu. 2022. Coder: Knowledge-
infused cross-lingual medical term embedding for
term normalization.Journal of biomedical informat-
ics, 126:103983.
Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin.
2021. Mr. tydi: A multi-lingual benchmark for dense
retrieval. InProceedings of the 1st workshop on
multilingual representation learning, pages 127–137.
Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo,
Ehsan Kamalloo, David Alfonso-Hermelo, Xi-
aoguang Li, Qun Liu, Mehdi Rezagholizadeh, and
Jimmy Lin. 2023. Miracl: A multilingual retrieval
dataset covering 18 diverse languages.Transactions
of the Association for Computational Linguistics,
11:1114–1131.

A Benchmark Construction Details
A.1 Task 1: Data Statistics
Table A1: Task 1 query distribution by language.
EN ES JA RU ZH
Queries 3,696 1,693 346 303 89
Total corpus: 2,552 documents (1,290 positive +
1,262 hard negatives). All queries grounded with
UMLS CUI tags (100% coverage).
A.2 Task 2: Multi-Encoder Validation
Three independent encoders (BGE-M3, E5-large,
SapBERT) compute pairwise cosine similarities
for each confusion set. A set is accepted only if
≥2/3 encoders agree on the tier assignment based
on cosine similarity thresholds: Tier 1: ≥0.70;
Tier 2: 0.40–0.75; Tier 3: 0.15–0.45.
Table A2: Tier validation statistics. Mean cosine simi-
larity (3-encoder average) confirms monotonic tier or-
dering with non-overlapping confidence intervals.
Tier Mean±Std Min Max N
T1 (Synonym).790±.089.525 1.000 3,005
T2 (Sibling).655±.047.493 .803 1,645
T3 (Related).514±.029.433 .577 325
Rejected — — — 967
Re-tiered — — — 1,597
A.3 Task 2: Leave-One-Validator-Out
Analysis
To assess whether tier assignments depend on any
single validator, we re-run the validation using each
2-encoder pair (requiring 2/2 agreement) and com-
pare with the full 3-encoder result.
Table A3: Leave-one-validator-out sensitivity. Among
sets accepted by each 2-encoder pair, we report the
fraction matching the full 3-encoder tier assignment.
No set shifts to a non-adjacent tier in any condition.
Excluded Accepted Same tier Stability
BGE-M3 1,212 1,212 100.0%
E5-large 3,477 2,968 85.4%
SapBERT 2,879 2,879 100.0%
The stricter 2/2 agreement threshold naturally
reduces the number of accepted sets, but among
those accepted, tier assignments are highly stable.
When E5-large is excluded, 509 sets (14.6%) shift
between adjacent tiers (predominantly Tier 1 →Tier 2), reflecting differences in the encoders’ simi-
larity calibration rather than fundamental tier ambi-
guity.
A.4 Task 3: Translation Quality Assurance
Concept fidelity.A bilingual medical lexicon
built from 587K UMLS English terms mapped to
target-language equivalents verifies that medical
concepts are preserved in translation. Standard
local-language medical terminology is accepted,
not just English surface forms.
Back-translation consistency.Each translation
is back-translated to English via NLLB-200 and
compared with the original via token-overlap F1.
Audit status thresholds. Pass: term preservation
rate≥0.9 AND back-translation F1 ≥0.8.Fail:
term preservation <0.7 OR back-translation F1
<0.6.Flag: all other cases (at least one criterion
met but not all). Queries with status “pass” or “flag”
form the official evaluation subset.
CUI tagging thresholds.Task 1 queries are CUI-
tagged via a cascading pipeline: (1) exact lexi-
cal match against UMLS preferred terms (confi-
dence≥0.9, substring match requires ≥3 charac-
ters); (2) biomedical linker fallback (confidence
≥0.7); (3) null if both fail.
Table A4: Translation QA results by language. This
table covers the 5 translated languages only (English
source excluded). Official subset includes pass +
flag queries. Total official = 1,368 translated + 340
English source = 1,708.
Lang Pass Flag Fail Official Total
ES 82 249 9 331 340
FR 53 271 16 324 340
JA 2 213 125 215 340
ZH 9 209 122 218 340
RU 6 274 60 280 340
Total 152 1,216 332 1,368 1,700
A.5 Train/Test Separation Verification
Table A5: Query text overlap between MMed-Embed
training data (137K rows) and benchmark test sets.
Task Overlap Rate
Task 1 (6,127 queries) 2 0.03%
Task 2 (4,143 queries) 20 0.48%
Task 3 (2,040 queries) 0 0.00%

Table A6: MMed-Embed 2 ×2 factorial ablation with bootstrap 95% CIs. All variants: 137K rows, 10 epochs,
identical hyperparameters.
Loss Negatives Task 1 Task 2 Task 3 MMed-IR
A0 — — .344±.007.432±.010.264±.013.347
A1L ret random.388±.007.454±.011.267±.013.369
A2L ret+L topo random .387±.007.451±.011.265±.013.368
A3L ret ANCE .370±.007.459±.011.265±.013.364
A4L ret+L topo ANCE .370±.007.459±.011.263±.013.364
Task 2 overlaps are short UMLS terms (e.g.,
“ATC”, “airway resistance”) common to both train-
ing vocabulary and confusion set members. No
semantic leakage is present.
B Reference Model: MMed-Embed
Training objective, two-stage procedure, and key
hyperparameters are described in Section 5.2.
Hard negative mining.Hard negatives are
mined using ANCE-style (Xiong et al., 2020) re-
trieval from the Stage-1 model’s embeddings: for
each query, the top- kmost similar non-positive
documents are selected ( k=15 ). To avoid false
negatives, candidates with similarity ≥95% of the
positive score are filtered out (positive-aware filter-
ing).
Ablation study.Table A6 presents a 2 ×2 facto-
rial ablation isolating the contributions of topology-
aware loss ( Ltopo) and ANCE-style hard negative
mining. All variants use identical hyperparame-
ters: 137K training rows, 10 epochs, effective batch
1,024, lr2×10−6, starting from BGE-M3.
The dominant effect is fine-tuning itself: Stage-
1 contrastive training (A1) improves MMed-IR
by +0.022 over the BGE-M3 base. Neither Ltopo
nor hard negative mining produces a statistically
significant additional gain under controlled condi-
tions: A1 vs. A2 ( +L topo, random neg): ∆MMed-
IR=− 0.001; A1 vs. A3 (mined neg, Lretonly):
∆MMed-IR =− 0.005. Hard negative mining
shifts the performance profile, improving Task 2
(+0.005) at the cost of Task 1 ( −0.018), but does
not improve the aggregate. This suggests that for
MMed-Bench-IR, the primary benefit comes from
domain-specific contrastive fine-tuning on multilin-
gual medical data, rather than from loss function de-
sign or negative mining strategy. The marginal gain
of fine-tuning over zero-shot BGE-M3 ( +0.022
MMed-IR) further indicates that the benchmark’s
core challenge is architecture-level multilingual
alignment, not training data availability.Full hyperparameters are in Table A7.
C Full Per-Model Results
All tables below include all ten evaluated systems
across six paradigm families.
Statistical framework.All 95% CIs reported in
this paper use query-level bootstrap resampling
(n=2,000 iterations) applied uniformly across all
tables and headline claims. For each metric (R@1,
R@5, nDCG@10), per-query scores are resampled
with replacement and the 2.5th/97.5th percentiles
form the CI. For composite metrics (Task 1 score
= mean of per-query nDCG@10 and R@1; Task 2
score = mean of three tier R@1s; MMed-IR = mean
of three task scores), we resample per-query scores
at each level and propagate through the composi-
tion. Fairness gap CIs are computed by bootstrap-
ping the max −min of per-language means, captur-
ing the full uncertainty in both the maximum and
minimum languages. All CIs are reported as ±
half-width of the 95% interval.
C.1 Task 1: Per-Language nDCG@10
Table A8 reports per-language nDCG@10 for all
ten systems. Biomedical-only models show near-
zero scores on Japanese, while multilingual mod-
els maintain moderate performance across all lan-
guages.
C.2 Task 2: Per-Tier Recall@5
Table A9 extends the Recall@1 analysis in the main
text to Recall@5. The tier ordering (Synonym,
Sibling, Related) is preserved across all models at
this cutoff.
C.3 Fairness Gaps
Table A10 reports cross-language fairness gaps
across all three tasks. MMed-Embed with Reranker
achieves the lowest gap on all tasks, while
biomedical-only models show the largest gaps, con-
sistent with Finding C.

Table A7: MMed-Embed training hyperparameters.
Parameter Value
Base model BAAI/bge-m3 (568M params)
Pooling Mean pooling + L2 normalization
Training data format FlagEmbedding (query,pos,neg)
Optimizer AdamW (weight decay 0.01)
LR schedule Linear warmup (10%) + linear decay
Temperatureτ0.02
Max query length 256 tokens
Max passage length 256 tokens
Hard negatives per query 15 random
In-batch negatives Cross-GPU gathered (effectiveB−1)
Gradient checkpointing Enabled
Precision Mixed (bf16)
Hardware 8×NVIDIA B200 GPUs
DDP backend NCCL
Seed 42
Training data 137K rows (MMedBench 40K + UMLS 87K + xBioASQ 10K)
Negatives 15 random per query
LossL ret(contrastive)
Per-device batch size 128 (effective 1,024)
Learning rate2×10−6
Epochs 10 (1,330 steps)
Table A8: Task 1 per-language nDCG@10 with bootstrap 95% CIs for all ten systems.
Model Par. EN ES JA RU ZH
BM25 Lex .162±.009.133±.012.004±.005.064±.018.077±.037
BioLORD Bio .367±.009.292±.015.000±.000.058±.020.032±.027
SapBERT Bio .428±.009.315±.015.001±.002.091±.024.011±.012
Hybrid Hyb .323±.009.250±.013.185±.024.276±.028.145±.038
BGE-M3 Multi .590±.010.493±.014.254±.033.407±.032.148±.048
E5-large Multi .570±.009.456±.015.248±.029.332±.028.061±.037
ColBERT-XM Late-int .516±.010.304±.014.151±.024.262±.030.057±.036
Reranker 2-stg .628±.009.566±.014.316±.034.482±.031.269±.051
MMed-Embed†M+Med .638±.009.593±.011.347±.033.539±.027.385±.053
MMed+Rer.†M+R.645±.009.607±.013.392±.035.537±.028.467±.049
Table A9: Task 2 Recall@5 by tier with bootstrap 95% CIs for all ten systems.
Model Par. Synonym Sibling Related
BM25 Lex .265±.016.105±.014.004±.006
BioLORD Bio .613±.017.273±.022.044±.034
SapBERT Bio .729±.016.380±.022.015±.013
Hybrid Hyb .829±.013.439±.023.025±.019
ColBERT-XM Late-int .921±.010.494±.022.023±.018
BGE-M3 Multi .982±.004.631±.021.033±.024
E5-large Multi .977±.005.637±.021.048±.032
Reranker 2-stg .985±.004.676±.020.044±.027
MMed-Embed†M+Med .989±.003.710±.021.048±.026
MMed+Rer.†M+R.988±.004.723±.020.048±.028

Table A10: Cross-language fairness gap (max −min
nDCG@10 across languages) per task for all ten sys-
tems. Lower is more equitable.
Model Par. T1 T2 T3
BM25 Lex .162 .012 .485
BioLORD Bio .367 .353 .693
SapBERT Bio .427 .520 .762
BM25+BGE-M3 Hyb .189 .104 .400
BGE-M3 Multi .441 .308 .235
E5-large Multi .509 .296 .394
ColBERT-XM Late .459 .331 .369
BGE-M3+Rer. 2-stg .358 .208 .180
MMed-Embed†M+M .291 .126 .221
MMed+Rer.†M+R .253 .104 .170
C.4 Task 3: Translation Sensitivity
To verify that the QA filtering does not shape bench-
mark outcomes, we compare model rankings under
three query subsets: pass-only (492 queries, Latin-
script dominated), official pass+flag (1,708), and
all translated (2,040). This analysis uses five zero-
shot baselines (excluding hybrid, reranker, and
MMed-Embed) to isolate the effect of QA filtering
on ranking order.
Table A11: Task 3 weighted nDCG@10 under different
QA subsets. Model ranking is perfectly preserved be-
tween official and all-translated ( ρ=1.0 ) and nearly so
with pass-only (ρ=0.9).
Model Pass-only Official All translated
BM25 0.387 0.172 0.154
BioLORD 0.638 0.344 0.304
SapBERT 0.698 0.383 0.339
BGE-M3 0.843 0.784 0.769
E5-large 0.868 0.778 0.755
D Benchmark Card
Task type.Information retrieval (ranking).
Languages.English, Spanish, French, Japanese,
Chinese, Russian (6 languages, 3 writing systems).
Domains.Biomedical (UMLS-grounded con-
cepts, BioASQ questions).
Data sources.UMLS 2025AB (NLM license re-
quired), BioASQ 13b (registration required).
Annotation method.Automatic construction
with algorithmic validation: UMLS CUI ground-
ing for Task 1; 3-encoder majority vote for Task 2;
UMLS concept-fidelity lexicon + back-translation
consistency for Task 3.Known limitations.See Section 9. Non-Latin-
script translation quality is bounded by UMLS cov-
erage for those languages.
Maintenance plan.Benchmark hosted on Hug-
gingFace Datasets. Human audit results and ver-
sion updates documented in the dataset card.
Intended use.Evaluation of multilingual medi-
cal retrieval systems. Not intended as training data.
Prohibited use.Clinical decision support with-
out further validation.
E Qualitative Examples
Tables A12–A14 present representative examples
from each task, drawn from actual benchmark data.
These examples illustrate the cross-lingual and
concept-level challenges that MMed-Bench-IR is
designed to evaluate.

Table A12:Task 1 qualitative examples.Two cross-lingual QA retrieval examples spanning Latin, CJK, and
Cyrillic scripts. All positives share the same UMLS CUI as the English query.
Example Lang Text Role
Atrial Tachycardia
Ectopic
(C0039234)EN ATRIAL TACHYCARDIA ECTOPIC Query
ES Latidos cardiacos anormalmente rápidos que se originan de uno o más focos
automáticos en el ATRIO CARDÍACO...Positive
JA心房異所性頻拍Positive
RU EKTOPICHESKAIA PREDSERDNAIA TAKHIKARDIIA Positive
Lip/oral cavity
cancer
(C0861542)EN Lip and/or oral cavity cancer stage unspecified Query
ES Cáncer del labio o de la cavidad oral, estadio no especificado Positive
JA口唇および口腔内癌、病期不明Positive
RU Rak guby i (ili) rotovoj polosti neutochnennoj stadii Positive
Table A13:Task 2 qualitative examples.Three difficulty tiers with increasing surface-form divergence. Encoder
cosine similarities (BGE-M3 / E5-large / SapBERT) are shown for Tiers 2 and 3. Tier 3 exhibits near-zero lexical
overlap across languages, rendering BM25 ineffective.
Tier Lang Text CUI Note
Tier 1
(Synonym)
EasyEN Drug-induced asthma C0340067 High cross-lingual
cohesion; lexical
overlap partially
preservedJA薬物誘発性喘息C0340067
RU Astma, vyzvannaia lekarstvennym sredstvom C0340067
ES Asma inducida por fármacos C0340067
Tier 2
(Sibling)
MediumEN Bowel infarction C0241950 BGE-M3: 0.61
E5-large: 0.88
SapBERT: 0.43
Moderate divergenceJA腸梗塞C0241950
RU Infarkt kishechnika C0241950
ES Infarto intestinal C0241950
Tier 3
(Related)
HardEN ADP Phosphocreatine Phosphotransferase C0010287 BGE-M3: 0.39
E5-large: 0.83
SapBERT: 0.43
BM25 fails entirelyJA CKアイソザイムC0010287
RU KREATINFOSFOKINAZA C0010287
ES Creatina Fosfoquinasa C0010287
Table A14:Task 3 qualitative example.EGFR mutation query translated to 6 languages via NLLB-200 with
two-stage translation QA. Audit status reflects translation quality gating: PASS (both checks satisfied), FLAG (one
check satisfied). Latin-script translations pass more reliably than CJK translations, consistent with limited UMLS
coverage for non-Latin scripts.
Role Lang Text Audit
Queries (QID: 67da0c0f)
Source EN What is the most common EGFR mutation in glioblastoma multiforme? —
Translation ES ¿Cuál es la mutación EGFR más común en el glioblastoma multiforme? PASS
Translation FR Quelle est la mutation EGFR la plus fréquente dans le glioblastome
multiforme ?FLAG
Translation JA膠芽腫で最も一般的なEGFR変異は? FLAG
Translation ZH什是最常的EGFR突在多形母胞瘤? FLAG
Translation RU Kakova naibolee rasprostranennaia mutatsiia EGFR v mul’tiformennoj
glioblastome?FLAG
Gold evidence snippets (BioASQ 13b)
Evidence EN The epidermal growth factor receptor variant type III (EGFRvIII) is the
most common mutation of EGFR in glioblastoma multiforme (GBM)
and is found in approximately 25% of all GBMs.—
Evidence EN In this study we determined whether EGFR amplification and expression
of the most common mutation in GBMs (EGFRvIII) is retained at tumor
recurrence.—