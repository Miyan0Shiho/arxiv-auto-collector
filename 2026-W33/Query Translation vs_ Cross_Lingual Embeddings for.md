# Query Translation vs. Cross-Lingual Embeddings for Sinhala-Tamil E-Government Information Retrieval

**Authors**: Dharshi Balasubramaniyam, Tiroshan Madushanka

**Published**: 2026-08-13 04:49:39

**PDF URL**: [https://arxiv.org/pdf/2608.12820v1](https://arxiv.org/pdf/2608.12820v1)

## Abstract
This paper presents a comparative evaluation of cross-lingual information retrieval (CLIR) methods for retrieving English government information using Sinhala and Tamil queries. Two CLIR paradigms are investigated: Query Translation (QT), employing Google Translate, NLLB, and mBART50, and Cross-Lingual Embeddings (CLE), using LaBSE, multilingual E5, and BGE-M3, with monolingual English retrieval as the baseline. Experiments are conducted on a human-verified benchmark comprising 500 Sinhala, Tamil, and English question-answer pairs derived from 1,699 segmented contexts from Sri Lanka's Government Information Center (GIC). Retrieval performance is evaluated using Recall@k (k = 1, 3, 5, 10, 15). Monolingual retrieval performs poorly (Recall@15 <10%), whereas all CLIR approaches substantially improve retrieval accuracy. Among them, BGE-M3 achieves the highest Recall@15, reaching 96.2% for Sinhala-English and 95.6% for Tamil-English, outperforming the best QT approach (Google Translate: 92.4% and 93.0%) while avoiding translation overhead. These results demonstrate that multilingual embedding models provide a more effective and scalable solution for cross-lingual retrieval-augmented generation (RAG) in low-resource government domains.

## Full Text


<!-- PDF content starts -->

Query Translation vs. Cross-Lingual Embeddings
for Sinhala–Tamil E-Government Information
Retrieval
1stDharshi Balasubramaniyam
University of Kelaniya
Sri Lanka
dharshib.8@gmail.com2ndTiroshan Madushanka
University of Kelaniya
Sri Lanka
tiroshanm@kln.ac.lk
Abstract—This paper presents a comparative evaluation of
cross-lingual information retrieval (CLIR) methods for retrieving
English government information using Sinhala and Tamil queries.
Two CLIR paradigms are investigated: Query Translation (QT),
employing Google Translate, NLLB, and mBART50, and Cross-
Lingual Embeddings (CLE), using LaBSE, multilingual E5, and
BGE-M3, with monolingual English retrieval as the baseline.
Experiments are conducted on a human-verified benchmark
comprising 500 Sinhala, Tamil, and English question-answer
pairs derived from 1,699 segmented contexts from Sri Lanka’s
Government Information Center (GIC). Retrieval performance
is evaluated using Recall@k (k = 1, 3, 5, 10, 15). Monolingual
retrieval performs poorly (Recall@15 ¡10%), whereas all CLIR
approaches substantially improve retrieval accuracy. Among
them, BGE-M3 achieves the highest Recall@15, reaching 96.2%
for Sinhala-English and 95.6% for Tamil-English, outperforming
the best QT approach (Google Translate: 92.4% and 93.0%)
while avoiding translation overhead. These results demonstrate
that multilingual embedding models provide a more effective and
scalable solution for cross-lingual retrieval-augmented generation
(RAG) in low-resource government domains.
Index Terms—Cross-Lingual Information Retrieval, Low-
Resource Languages, Sinhala, Tamil, Query Translation, Cross-
Lingual Embeddings, Retrieval-Augmented Generation
I. INTRODUCTION
Retrieval-Augmented Generation (RAG) improves the fac-
tual grounding of large language models by retrieving relevant
external content at inference time [1], [2]. Most RAG systems
assume the query and document languages coincide [1], which
limits their usefulness for speakers of low-resource languages
who wish to query in their native language while retrieving
from a predominantly English-language corpus. This mismatch
is acute in Sri Lanka, where citizens commonly interact
with public digital services in Sinhala or Tamil while official
information remains largely English-centric.
Existing retrieval models are developed and optimized
chiefly for high-resource languages, leaving Sinhala and
Tamil underrepresented due to limited curated datasets, in-
consistent domain-specific translation quality, and weak cross-
lingual embedding alignment. This work systematically eval-
uates practical cross-lingual retrieval pipelines for these two
languages against a real-world, government-domain English
knowledge base.A. Research Questions
•RQ1:How effective is monolingual English retrieval
when queries are issued in Sinhala and Tamil without
cross-lingual handling?
•RQ2:How do query translation-based retrieval ap-
proaches compare against cross-lingual embedding-based
approaches for Sinhala–English and Tamil–English re-
trieval?
•RQ3:Which cross-lingual retrieval strategy is most ef-
fective and robust for low-resource Sri Lankan languages
in a government-domain setting?
B. Contributions
This study contributes: (i) a systematic comparison of
monolingual, query-translation, and cross-lingual-embedding
retrieval pipelines for Sinhala–English and Tamil–English re-
trieval; (ii) a retrieval-ready English knowledge base built
from real Sri Lankan government service pages, segmented
into 1,699 semantically coherent contexts; (iii) empirical ev-
idence that cross-lingual embeddings, particularly BGE-M3,
can outperform translation-based retrieval while removing the
translation step entirely; and (iv) a verified, publicly available
multilingual (English–Sinhala–Tamil) question-answer bench-
mark for future CLIR research.
II. RELATEDWORK
Traditional RAG assumes alignment between query and
document language [1], restricting applicability for low-
resource language speakers [3]. Retrieval must bridge this
linguistic gap by locating passages that are semantically equiv-
alent to the query despite being in a different language [4]; a
weak retrieval component propagates errors into hallucinated
generation [5], [6].
Monolingual embeddings optimized for a single language
leave semantically equivalent cross-lingual pairs far apart
in vector space [3], [7], motivating two established CLIR
strategies. Query translation converts the query into the doc-
ument language before retrieval but is sensitive to translation
ambiguity, especially for short, under-specified queries [5],
[6]. Document translation avoids this ambiguity but introduces
arXiv:2608.12820v1  [cs.IR]  13 Aug 2026

substantial latency and translation noise, making it impractical
at scale [3].
Cross-lingual embeddings instead learn a shared represen-
tation space across languages, making semantic similarity
language-agnostic [7]. Early models such as mBERT [7] and
XLM-R [8] targeted general multilingual alignment, while
LaBSE [7] and multilingual E5 [9] focused on stronger
sentence-level cross-lingual retrieval, and BGE-M3 [10] ex-
tended this with multi-functional dense and hybrid retrieval
training. Translation-based pipelines using NLLB [11] or high-
quality commercial MT remain competitive baselines, but
recent multilingual embeddings are closing, and in some cases
exceeding, this gap [4]. General-purpose multilingual retrieval
benchmarks such as MIRACL [12] and mMARCO [13] cover
a broad set of languages but do not include Sinhala or Tamil
and are not grounded in a single, government-domain corpus,
which limits their ability to characterize retrieval behavior for
these two languages in a realistic, terminology-heavy public-
service setting. Prior work has not systematically benchmarked
government-domain CLIR for Sinhala and Tamil, or directly
compared QT and CLE pipelines for these two typologically
distinct, low-resource South Asian languages, which this study
addresses [14].
III. METHODOLOGY
This section describes the proposed comparative retrieval
architecture used to evaluate cross-lingual information access
for Sinhala–English and Tamil–English queries.
A. Overview of the Proposed Architecture
The proposed solution is a controlled, three-pipeline com-
parative architecture that isolates the effect of cross-lingual
adaptation strategy while holding the underlying document
index, query set, and evaluation protocol fixed. Each pipeline
accepts the same Sinhala or Tamil query and returns a ranked
list of the top-15 English contexts from a common indexed
knowledge base, differing only inhow(or whether) the lin-
guistic gap between the query and the English documents is
bridged before ranking:
1)Baseline (Monolingual English Embeddings, No
Translation): the naive cross-lingual case, i.e., no adap-
tation is applied.
2)Query Translation (QT): an explicit, translation-based
strategy that lexically bridges the language gap using
machine translation prior to embedding.
3)Cross-Lingual Embedding (CLE): an implicit,
embedding-based strategy that achieves direct semantic
alignment without any translation step.
All three pipelines share a common English document index,
built once over the full context collection, and are evaluated
independently under identical retrieval-depth and similarity-
ranking conditions, isolating retrieval-strategy effects from
indexing effects. Fig. 1 illustrates the resulting three-pipeline
architecture.B. Baseline Pipeline
The baseline pipeline represents the naive cross-lingual
case, where no translation or multilingual alignment is applied.
The original Sinhala or Tamil query is embedded directly using
the same monolingual English embedding model (FastEmbed)
that was used to build the document index, and retrieval is
performed by cosine similarity between this mismatched, non-
English query embedding and the English document vectors.
Because the embedding model was never optimized to align
Sinhala/Tamil and English representations, this pipeline serves
as a lower-bound reference against which the benefit of explicit
cross-lingual adaptation can be measured.
C. Query Translation (QT) Pipeline
The QT pipeline models the explicit translation strategy: the
Sinhala or Tamil query is first machine-translated into English,
and the translated query is then embedded using the same
monolingual English embedding model (FastEmbed) used for
the document index, before cosine similarity ranks the English
passages. Three translation models, spanning commercial and
open-source, and modern and earlier-generation systems, are
evaluated within this pipeline:
•Google Translate: a commercial system with mature in-
frastructure and high translation quality for low-resource
languages, included to provide a practical, near upper-
bound QT baseline.
•NLLB (No Language Left Behind): an open-source
neural machine translation model optimized for low-
resource languages, included to assess whether open-
source MT can approach commercial translation quality
for Sinhala and Tamil.
•mBART50: a widely cited multilingual sequence-to-
sequence model, included to represent an academic MT
baseline and to characterize the limitations of earlier
multilingual translation architectures when applied to
retrieval-focused tasks.
D. Cross-Lingual Embedding (CLE) Pipeline
The CLE pipeline models the implicit, semantic-alignment
strategy: the original Sinhala or Tamil query is embedded di-
rectly in a multilingual embedding space, and passage ranking
is computed via cosine similarity between this non-English
query vector and the English document vectors within the
same shared space, without any translation step. Three multi-
lingual bi-encoder models are evaluated within this pipeline:
•LaBSE (Language-Agnostic BERT Sentence Embed-
dings): a widely adopted baseline for cross-lingual sen-
tence similarity and retrieval, included as a stable point
of reference.
•Multilingual E5: an instruction-tuned embedding model
optimized for retrieval tasks, included to examine the
benefit of task-aware, retrieval-oriented multilingual em-
beddings.
•BGE-M3: a state-of-the-art multilingual embedding
model supporting dense and hybrid retrieval, included for

Query<SI/TA>
English Embeddings
<FastEmbed>
Top-15 ContextsBaseline RAG
Query<SI/TA>
Translation Model
<Google Translate /
mBART50 / NLLB>
English Embeddings
<FastEmbed>
Top-15 ContextsQT-based RAG
Query<SI/TA>
Cross-lingual Embeddings
<LaBSE / E5-M /
BGE-M3>
Top-15 ContextsCLE-based RAG
Fig. 1. Proposed three-pipeline retrieval architecture. All pipelines share the same indexed English knowledge base and return the top-15 ranked contexts by
cosine similarity; they differ only in how the Sinhala/Tamil query is bridged into the English embedding space before ranking.
its strong performance on multilingual retrieval bench-
marks and its ability to operate in a fully language-
agnostic manner without reliance on translation.
E. Embedding, Indexing, and Ranking
Document contexts are vectorized once per embedding
model (FastEmbed for the Baseline/QT pipelines; LaBSE,
multilingual E5, and BGE-M3 for the CLE pipeline) and
stored in separate Pinecone vector stores, in conjunction with
LangChain utilities for embedding generation, vector indexing,
and similarity-based retrieval. For every pipeline and query,
ranking is computed using cosine similarity between the query
embedding and the indexed English context embeddings, and
results are collected up to a fixed depth of 15 contexts to
support Recall@k evaluation across multiple retrieval depths.
F . Implementation Details
The proposed pipelines were implemented in Python within
a Jupyter Notebook environment. Structured data was managed
with CSV files and manipulated using pandas and NumPy. GIC
service pages were programmatically scraped using Beauti-
fulSoup4, complemented by LangChain-GenAI for automated
extraction and organization of textual content, and the resulting
embeddings were indexed and queried through Pinecone.
Together, these tools enabled a consistent, reproducible imple-
mentation of the Baseline, QT, and CLE pipelines described
above.
IV. EXPERIMENT
A. Experimental Setup
1) Dataset:The knowledge base was constructed from Sri
Lanka’s Government Information Center (GIC) website1. A
total of 761 web pages spanning ten main service categories
(e.g., Trade, Agriculture, Health, Education, Justice, Banking)
and 70 subcategories were scraped, then segmented via an
1https://gic.gov.lk/LLM-guided prompt into 1,699 semantically coherent con-
texts of 250–400 words each, preserving original wording.
A stratified sample of 500 contexts, proportional to category
distribution, was selected for evaluation (Table I). For each
sampled context, an LLM generated a question–answer pair in
English, Sinhala, and Tamil, which was subsequently human-
verified for linguistic correctness and alignment, yielding 500
verified Sinhala and 500 verified Tamil queries, each mapped
to a single gold English context.
TABLE I
STRATIFIEDCONTEXTSAMPLE BYSERVICECATEGORY
Main Category Original Sampled
Trade, Business & Industry 419 124
Agriculture, Livestock & Fisheries 197 59
Health, Well-being & Social Service 168 49
Justice, Law & Rights 147 43
Banking, Tax & Insurance 113 33
Education & Training 110 32
Employment Information 107 31
Housing, Property & Utilities 92 27
Citizen’s Registrations 89 26
Travel, Tourism & Leisure 88 26
Environment 88 26
Communication & Media 81 24
Total 1699 500
All three pipelines described in Section II (Baseline, QT,
CLE) were evaluated under identical conditions using the
same 1,699 indexed contexts and 500 Sinhala/Tamil query sets,
retrieving up to 15 contexts per query.
2) Evaluation Metric:Because each query maps to exactly
one gold English passage, Recall@k was adopted as the
primary metric, defined for a queryqas 1 if the gold context
appears in the top-kretrieved results and 0 otherwise, averaged
over allNqueries:
R@k=1
NNX
i=1R@k(q i)(1)

Recall was computed atk= 1,3,5,10,and15to capture
both top-rank precision and overall retrieval coverage, which
is directly relevant to downstream RAG generation quality.
B. Results
Table II and Table III report Recall@k for Sinhala–English
and Tamil–English queries, respectively. The baseline pipeline
performed extremely poorly for both languages (Recall@15
of 8.2% for Sinhala and 4.2% for Tamil), confirming that
monolingual English embeddings cannot bridge the linguis-
tic gap (RQ1). All QT and CLE pipelines improved recall
substantially over this baseline.
1) Why Baseline Retrieval Fails:This near-zero recall
stems from a fundamental misalignment between the linguistic
spaces of the query and the document index rather than from
any weakness in the retrieval procedure itself. Because the
baseline pipeline embeds the Sinhala or Tamil query with
a monolingual English embedding model (FastEmbed), it is,
in effect, asking a model that has only ever learned an
English semantic space to place a non-English query within
that space. FastEmbed was optimized for a single language,
so semantically equivalent terms across Sinhala/Tamil and
English are not co-located in the resulting vector space, i.e.,
a query and its correct English passage can be conceptually
identical while being geometrically distant as embeddings.
Cosine similarity ranking is only meaningful when queries
and documents are embedded in a shared, aligned space, and
no such alignment exists for this pipeline. Consequently, the
ranked results returned by the baseline are effectively unrelated
to the true semantic content of the query, which explains why
recall remains below 10% at every retrieval depth and why ex-
plicit translation or cross-lingual embedding alignment, rather
than a larger monolingual index or a deeper retrieval depth, is
required to make Sinhala- and Tamil-language retrieval viable.
TABLE II
RECALL@K FORSINHALA–ENGLISHQUERIES(%)
Approach R@1 R@3 R@5 R@10 R@15
CLE - BGE-M3 59.4 83.4 90.2 94.0 96.2
QT - Google Translate 60.0 79.6 85.4 90.4 92.4
QT - NLLB 53.8 71.4 77.6 85.4 87.0
QT - mBART50 46.2 66.8 73.8 80.0 82.8
CLE - E5 Multilingual 36.4 54.2 59.6 68.4 73.0
CLE - LaBSE 30.4 49.4 59.8 68.0 72.4
Baseline (no adaptation) 1.6 3.8 5.6 7.4 8.2
TABLE III
RECALL@K FORTAMIL–ENGLISHQUERIES(%)
Approach R@1 R@3 R@5 R@10 R@15
CLE - BGE-M3 61.0 81.0 89.6 94.2 95.6
QT - Google Translate 59.4 78.4 85.2 91.2 93.0
QT - NLLB 53.2 74.6 80.0 86.6 89.0
CLE - E5 Multilingual 48.2 71.2 78.6 85.8 88.6
QT - mBART50 48.4 68.0 76.0 82.6 85.2
CLE - LaBSE 29.0 46.2 55.4 66.8 72.4
Baseline (no adaptation) 0.6 1.8 2.0 3.8 4.2
Fig. 2. Recall@15 across all evaluated retrieval approaches for Sinhala–
English and Tamil–English queries.
CLE–BGE-M3 achieved the highest Recall@15 for both
language pairs (96.2% Sinhala, 95.6% Tamil), followed by
QT–Google Translate (92.4%/93.0%). NLLB (87.0%/89.0%)
and mBART50 (82.8%/85.2%) trailed the commercial trans-
lation engine, while CLE–E5 Multilingual (73.0%/88.6%)
and CLE–LaBSE (72.4%/72.4%) trailed BGE-M3 (RQ2), as
summarized in Fig. 2.
C. Ablation Study
To better understandwhythe pipelines in Section IV-B
differ, this subsection breaks the results down along two
axes: the retrieval paradigm (QT vs. CLE) and the individual
embedding/translation model used within each paradigm.
1) Query Translation vs. Cross-Lingual Embeddings:
Google Translate provided a strong QT baseline due to its
mature infrastructure and extensive multilingual training data,
but open-source alternatives NLLB and mBART50 lagged,
likely reflecting weaker handling of the morphological com-
plexity of Sinhala and Tamil and consequent error propagation
into retrieval. CLE pipelines avoid this dependency entirely:
by mapping queries and documents into a shared semantic
space, BGE-M3 is not exposed to lexical translation errors,
and it achieved the highest Recall@1 for both languages
(approximately 60%, Fig. 3), indicating that it also ranks the
correct passage earlier, which is a property that is particularly
valuable in RAG settings where only the top few retrieved
passages are used for generation.
2) Language-Specific and Model-Level Behavior:Model
performance was not uniform across languages. Table IV
quantifies this by reporting the Recall@15 gap (∆= Tamil
−Sinhala) for every model. Multilingual E5 shows by far the
largest gap (+15.6 points: 88.6% Tamil–English vs. 73.0%
Sinhala–English), evidencing that its cross-lingual robustness
depends heavily on per-language training-data coverage. A
plausible explanation, which this study does not verify directly,
is that Tamil is more heavily represented than Sinhala in
the large-scale multilingual web corpora (e.g., mC4, CC-100)
typically used to pretrain models such as multilingual E5,
so its embedding space is likely to be better aligned with
English for Tamil than for Sinhala; confirming this would
require inspecting E5’s pretraining data composition or corpus-
size statistics directly, which falls outside the scope of this

Fig. 3. Recall@k trend for the strongest CLE and QT pipelines at increasing
retrieval depth.
study. LaBSE, in contrast, shows a gap of exactly 0.0 points
(72.4% for both languages), i.e., the most consistent of any
model, although at the lowest overall recall, evidencing its
general-purpose (rather than retrieval- or domain-optimized)
training objective. BGE-M3 shows a near-zero gap (−0.6
points: 96.2% Sinhala vs. 95.6% Tamil) while simultaneously
achieving the highest recall for both languages, evidencing that
its multi-functional dense-and-hybrid retrieval training gener-
alizes across typologically different low-resource languages
more effectively than the other CLE models (RQ3). The QT
pipelines show smaller, comparatively uniform gaps (+0.6
to+2.4 points), consistent with translation engines that are
not language-pair-specific in the way individual embedding
models are.
TABLE IV
RECALL@15 GAPBETWEENTAMIL ANDSINHALA BYMODEL
Approach Sinhala Tamil ∆(Ta−Si)
CLE - BGE-M3 96.2 95.6 −0.6
QT - Google Translate 92.4 93.0 +0.6
QT - NLLB 87.0 89.0 +2.0
QT - mBART50 82.8 85.2 +2.4
CLE - E5 Multilingual 73.0 88.6 +15.6
CLE - LaBSE 72.4 72.4 0.0
V. CONCLUSION
This study proposed and evaluated a controlled, three-
pipeline retrieval architecture, namely a monolingual baseline,
query translation, and cross-lingual embeddings, for Sinhala–
English and Tamil–English information access against a real-
world Sri Lankan government-domain knowledge base. Mono-
lingual retrieval fails almost entirely for both languages, while
all cross-lingual pipelines yield large improvements. BGE-M3
consistently achieved the best and most language-consistent
recall (95–96% at Recall@15) for both languages, outperform-
ing the strongest translation-based pipeline while removing the
need for an explicit translation step, and Google Translate
remained the strongest QT option with NLLB as a viable
open-source alternative. These results position cross-lingualembeddings, particularly BGE-M3, as the more effective and
scalable foundation for cross-lingual RAG in low-resource,
government-domain settings. These conclusions are based on
retrieval-level Recall@k over a single government-services
domain, and the evaluated embedding and translation models
were used in their pre-trained form without fine-tuning; the ob-
served recall differences are also reported as raw percentages
rather than as statistically tested effects. Future work should
therefore evaluate end-to-end generation quality, extend the
evaluation to additional domains, and explore fine-tuning and
hybrid QT/CLE retrieval architectures.
REFERENCES
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive NLP tasks,”
arXiv preprint arXiv:2005.11401, 2020.
[2] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2023.
[3] P. Shi, R. Zhang, H. Bai, and J. Lin, “Cross-lingual training of dense
retrievers for document retrieval,” inProceedings of the 1st Workshop
on Multilingual Representation Learning (MRL), 2021.
[4] R. Goworek, O. Macmillan-Scott, and E. B. ¨Ozyi ˘git, “Bridging language
gaps: Advances in cross-lingual information retrieval with multilingual
LLMs,”arXiv preprint arXiv:2510.00908, 2025.
[5] S. Saleh and P. Pecina, “Document translation vs. query translation for
cross-lingual information retrieval in the medical domain,” inProceed-
ings of the 58th Annual Meeting of the Association for Computational
Linguistics, 2020, pp. 6849–6860.
[6] Z. Huang, H. Bonab, S. M. Sarwar, R. Rahimi, and J. Allan, “Mixed
attention transformer for leveraging word-level knowledge to neural
cross-lingual information retrieval,”arXiv preprint arXiv:2109.02789,
2021.
[7] F. Feng, Y . Yang, D. Cer, N. Arivazhagan, and W. Wang, “Language-
agnostic BERT sentence embedding,”arXiv preprint arXiv:2007.01852,
2020.
[8] A. Conneau, K. Khandelwal, N. Goyal, V . Chaudhary, G. Wenzek,
F. Guzm ´an, E. Grave, M. Ott, L. Zettlemoyer, and V . Stoyanov, “Unsu-
pervised cross-lingual representation learning at scale,”arXiv preprint
arXiv:1911.02116, 2019.
[9] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei,
“Multilingual E5 text embeddings: A technical report,”arXiv preprint
arXiv:2402.05672, 2024.
[10] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, “BGE
M3-embedding: Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation,”arXiv preprint
arXiv:2402.03216, 2024.
[11] NLLB Team, M. R. Costa-juss `a, J. Cross, O. C ¸ elebi, M. Elbayad,
K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard,
A. Sun, S. Wang, G. Wenzek, A. Youngblood, B. Akula, L. Bar-
rault, G. M. Gonzalez, P. Hansanti, and J. Wang, “No language left
behind: Scaling human-centered machine translation,”arXiv preprint
arXiv:2207.04672, 2022.
[12] X. Zhang, N. Thakur, O. Ogundepo, E. Kamalloo, D. Alfonso-Hermelo,
X. Li, Q. Liu, M. Rezagholizadeh, and J. Lin, “MIRACL: A multilingual
retrieval dataset covering 18 diverse languages,”Transactions of the
Association for Computational Linguistics, vol. 11, pp. 1114–1131,
2023.
[13] L. H. Bonifacio, H. Abonizio, M. Fadaee, and R. Nogueira, “mMARCO:
A multilingual version of the MS MARCO passage ranking dataset,”
arXiv preprint arXiv:2108.13897, 2021.
[14] F. Zhang, Z. Zhang, X. Ao, D. Gao, F. Zhuang, Y . Wei, and Q. He, “Mind
the gap: Cross-lingual information retrieval with hierarchical knowledge
enhancement,”arXiv preprint arXiv:2112.13510, 2021.