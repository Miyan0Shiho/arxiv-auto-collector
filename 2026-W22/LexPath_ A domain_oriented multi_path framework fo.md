# LexPath: A domain-oriented multi-path framework for legal article retrieval

**Authors**: Weixuan Liu, Qingfeng Zhuge, Xuyang Chen

**Published**: 2026-05-28 16:43:17

**PDF URL**: [https://arxiv.org/pdf/2605.30205v1](https://arxiv.org/pdf/2605.30205v1)

## Abstract
Legal article retrieval is critical for building traceable and reliable legal AI systems, where conclusions must be grounded in specific legal articles. However, existing open-domain retrieval methods rely heavily on surface-level lexical or semantic similarity, making it difficult for them to distinguish legally relevant articles from those that are textually similar but legally inapplicable or misaligned with the user's underlying intent. To bridge this gap, we propose \textsc{LexPath}, a domain-oriented multi-path framework comprising a multi-path retrieval module and an intent-aware reranking module. The retrieval module combines two complementary legal-specific paths to collect candidate articles: an IRAC-guided sparse path that expands queries with legally informative keywords, and a structure-guided dense path trained with hard negatives derived from legal hierarchy and citation relations. Then, the reranking module further refines the candidate ranking by incorporating the intent consistency score between queries and legal articles. We evaluate \textsc{LexPath} on two publicly available benchmarks focusing on general-public queries and a self-constructed benchmark targeting domain-professional scenarios. Experimental results demonstrate that \textsc{LexPath} consistently outperforms lexical, dense, hybrid, and adaptive retrieval-augmented generation (RAG) baselines. Ablation studies further verify the effectiveness of each component.

## Full Text


<!-- PDF content starts -->

LEXPATH: A Domain-Oriented Multi-Path Framework for Legal Article
Retrieval
Weixuan Liu, Qingfeng Zhuge†, and Xuyang Chen
College of Computer Science and Technology, East China Normal University
qfzhuge@cs.ecnu.edu.cn
Abstract
Legal article retrieval is critical for building
traceable and reliable legal AI systems, where
conclusions must be grounded in specific le-
gal articles. However, existing open-domain
retrieval methods rely heavily on surface-level
lexical or semantic similarity, making it dif-
ficult for them to distinguish legally relevant
articles from those that are textually similar
but legally inapplicable or misaligned with
the user’s underlying intent. To bridge this
gap, we propose LEXPATH, a domain-oriented
multi-path framework comprising a multi-path
retrieval module and an intent-aware rerank-
ing module. The retrieval module combines
two complementary legal-specific paths to col-
lect candidate articles: an IRAC-guided sparse
path that expands queries with legally infor-
mative keywords, and a structure-guided dense
path trained with hard negatives derived from
legal hierarchy and citation relations. Then,
the reranking module further refines the candi-
date ranking by incorporating the intent consis-
tency score between queries and legal articles.
We evaluate LEXPATHon two publicly avail-
able benchmarks focusing on general-public
queries and a self-constructed benchmark tar-
geting domain-professional scenarios. Experi-
mental results demonstrate that LEXPATHcon-
sistently outperforms lexical, dense, hybrid,
and adaptive retrieval-augmented generation
(RAG) baselines. Ablation studies further ver-
ify the effectiveness of each component.1
1 Introduction
Legal article retrieval is crucial for building trace-
able and reliable legal AI systems (Pipitone and
Alami, 2024), where conclusions must be sup-
ported by specific legal articles. This task poses
significant challenges, even to existing state-of-the-
art (SOTA) open-domain solutions, because legal
relevance is not equivalent to lexical or semantic
1The data and code are available at: https://github.
com/lexpath-project/LexPath.
Ground-Truth Article: 
Article 2, Administrative Penalties Law:
An administrative penalty refers to an act by which an 
administrative agency...
 Primary Statute
 DefinitionQuery: The Market Supervision Bureau told me to close down 
my unlicensed street stall. Does this count as an 
administrative penalty?
 Definition
Article 13, Measures for the Investigation and 
Punishment of Unlicensed Business Operations:
Any person engaged in business operations without a 
license shall be penalized by the administrative 
department for industry and commerce in accordance 
with laws...
  Administrative Regulation
 Procedure
Article 7, Regulations on Administrative Penalty 
Procedures for Market Supervision and Administration:
Administrative penalties shall be administered by the 
market supervision and administration department at or 
above the county level...
 Departmental Rule
 Procedure
Article 19, Regulations on Administrative Penalty 
Procedures for Market Supervision and Administration:
Upon verification, a case shall be filed if the following 
conditions are met: (1) There is preliminary evidence ...; 
(2) ...; (3) ...
 Departmental Rule
 ApplicabilityTop-3 Retrieved Articles:Figure 1: An example where top-ranked articles are
textually related but legally inappropriate. Red text
denotes keyword matches, and each article is annotated
with its legal hierarchy level and intent label.
similarity (Li et al., 2025a). Legal articles that ap-
pear similar in wording or topic may involve differ-
ent legal concepts, authorities, scopes, procedural
functions, or conditions of applicability.
Legal article retrieval involves several domain-
specific challenges. First, user queries often de-
scribe concrete situations, or use colloquial expres-
sions, while legal articles are written in abstract and
terminology-intensive language (Su et al., 2024).
Second, legal articles are organized within a legal
hierarchy. Even when articles are semantically sim-
ilar or linked by citations, they may differ in legal
authority and may not be equally applicable to a
given query (Su et al., 2024). Third, open-domain
retrievers may return top-ranked articles that are
topically related to the query but do not match its
1arXiv:2605.30205v1  [cs.IR]  28 May 2026

underlying legal intent (Shao et al., 2023).
An example in Figure 1 shows that legal arti-
cle retrieval requires more than mere lexical or
semantic relevance. The ground-truth article de-
fines an administrative penalty in a primary statute.
Although the retrieved articles mention adminis-
trative penalties, they concern penalty procedures
or specific application conditions, and originate
from regulations or departmental rules with lower
authority than the ground-truth article.
Existing open-domain solutions are far from
perfect for this task. Lexical matching meth-
ods (Robertson and Zaragoza, 2009; Ponte and
Croft, 2017) rely on lexical matching and strug-
gle when users describe concrete scenarios or use
colloquial expressions. Dense embedding meth-
ods (Ming, 2022; Xiao et al., 2023; Li et al., 2023b;
Wang et al., 2024; Chen et al., 2024) excel at model-
ing semantic similarity, but struggle to distinguish
legal applicability. Although hybrid methods com-
bine sparse and dense features, they still fail to
explicitly incorporate legal-specific features and
remain constrained by surface-level textual sim-
ilarity. Adaptive retrieval-augmented generation
(RAG) methods (Trivedi et al., 2023; Yan et al.,
2024; Du et al., 2026) improve retrieval by refor-
mulating queries or iteratively refining retrieved
evidence, but they similarly fail to explicitly model
legal-specific relevance. As a result, existing meth-
ods often rank inapplicable but textually similar
articles above the truly applicable ones.
To address these challenges, we propose LEX-
PATH, a domain-oriented framework consisting of
a multi-path retrieval module and an intent-aware
reranking module. The retrieval module first col-
lects candidate articles through two complementary
paths: an IRAC-guided sparse path and a structure-
guided dense path. In the sparse path, we pro-
pose IRAC-Exp to expand queries with retrieval-
oriented keywords. In the dense path, we propose
Struct-Neg to train the embeddings with hierarchy-
and citation-aware hard negatives. Then the can-
didate ranking is refined by the reranking module
that incorporates an intent consistency score.
We evaluate LEXPATHon two public bench-
marks based on general-public queries, and
STATUTERAG, our newly constructed benchmark
for professional scenarios. Experiments show that
LEXPATHconsistently outperforms lexical, dense,
hybrid, and adaptive RAG baselines. Further abla-
tion studies demonstrate the effectiveness of each
component, and efficiency analysis illustrates thetrade-off between retrieval quality and latency.
In summary, our contributions are as follows:
•We propose LEXPATH, a domain-oriented
multi-path framework for legal article retrieval
that jointly models lexical, structural, and
intent-level relevance between user queries
and legal articles.
•We construct StatuteRAG, a professional-
scenario benchmark for Chinese legal article
retrieval, which complements existing bench-
marks based on general-public queries.
•Experiments on three benchmarks show that
LEXPATHconsistently outperforms lexical,
dense, hybrid, and adaptive RAG baselines,
and ablation studies further verify the effec-
tiveness of each component.
2 Related Work
There is a growing interest in building reliable le-
gal AI systems across different languages (Bhat-
tacharya et al., 2019; Louis and Spanakis, 2022;
Goebel et al., 2023; Pipitone and Alami, 2024;
Hou et al., 2025). Legal applications often require
conclusions to be traceable to specific legal arti-
cles (Pipitone and Alami, 2024), making legal ar-
ticle retrieval a fundamental capability for down-
stream tasks such as legal QA (Fei et al., 2024; Li
et al., 2025b), judgment prediction (Xiao et al.,
2018), and judgment document generation (Su
et al., 2025). Despite its importance, legal arti-
cle retrieval remains less explored than legal case
retrieval, which has been extensively studied to
identify similarities between judicial precedents (Li
et al., 2023a; Shao et al., 2023; Deng et al., 2024;
Kim et al., 2025) using domain-specific features.
Open-domain retrieval methods, including lex-
ical matching (Robertson and Zaragoza, 2009;
Ponte and Croft, 2017), dense embedding (Xiao
et al., 2023; Li et al., 2023b; Wang et al., 2024),
and adaptive RAG (Trivedi et al., 2023; Yan et al.,
2024; Du et al., 2026), are not fully suited to le-
gal article retrieval. These methods usually do not
explicitly model domain-specific factors such as le-
gal terminology, legal hierarchy, and query-article
intent consistency, and therefore capture only part
of the relevance between user queries and legal ar-
ticles (Deng et al., 2024; Li et al., 2025c). These
limitations motivate our work, which focuses on
fine-grained domain-specific relevance modeling
2

Intent-Aware Reranking Module Multi-Path Retrieval Module
A. IRAC-Guided Sparse Path (Online)
B. Structure-Guided Dense Path (Offline)
(i) Article-Side Structures
Constitution
Primary Statute
Administrative
Regulation
Local Regulation
Departmental RuleStatutory Hierarchy
Judicial
InterpretationCitation
GraphIRAC-Exp
···Expanded
QuerySparse
Index
User Query
Issue
Rule
Application
ConclusionIRAC Analysis
Keyword
Extraction
(iii) Training PipelineSame
HierarchyAdjacent
HierarchyCitation
Neighbors
Definition
 Applicability
 Consequence
Procedure
 OthersIntent Taxonomy
LLM-Based Intent Classifier
Query
Intent···
Article
Intent
Article intents 
can be cached.
Fine-Tuned
Embedding ModelDense 
IndexContrastive
LearningAnchor
Positive
Hard
NegativePull
Closer
Push Away
...
Top-k ArticlesWeighted
Fusion 
Candidates
Neural 
Reranker 
ScoreFused 
Retrieval 
ScoreIntent 
Consistency
Score(ii) Struct-Neg
Figure 2: An overview of our framework, LEXPATH, for legal article retrieval.
for legal article retrieval. To the best of our knowl-
edge, LEXPATHis among the first domain-oriented
multi-path frameworks tailored to Chinese legal
article retrieval.
3 Methodology
3.1 Overview
To address the challenges discussed above, we pro-
pose LEXPATH, a legal-specific multi-path retrieval
framework that models lexical, legal structural, and
intent-level relevance in a unified manner.
We define legal article retrieval as a task over
an article corpus Cand a query set Q. For each
query q∈Q , we annotate a ground-truth set of
articles Aq⊆C. The goal is to develop a retrieval
model that, given q, ranks the articles in Cso that
articles in Aqare assigned higher ranks and can
be retrieved in the top- Kresults. As shown in
Figure 2, LEXPATHfirst collects candidate articles
Rq⊆C with|Rq|> K by weighting the scores
from two complementary retrieval paths. Then, the
intent-aware reranker refines the candidate ranking
inRqby incorporating the intent consistency score,
yielding the final top-Kranked articles.
3.2 IRAC-Guided Sparse Path
This component aims to improve sparse retrieval in
legal article retrieval. Lexical matching is impor-
tant in this task because small wording differences
may correspond to different legal concepts, appli-
cable conditions, or legal consequences. However,
the original query may not contain all the keywords
needed to retrieve the relevant article. While aquery may describe concrete scenarios or use col-
loquial expressions, the target article is typically
characterized by abstract legal concepts and dense
terminology. To address this issue, we develop
IRAC-Exp, which uses IRAC analysis to expand
the query with legal keywords for sparse retrieval.
IRAC analysis is a commonly used legal rea-
soning framework that decomposes a legal prob-
lem into Issue, Rule, Application, and Conclusion.
Prior studies have shown that IRAC analysis can
improve the performance of LLMs in complex legal
reasoning tasks (Yu et al., 2022; Servantez et al.,
2024). We use it not for answer generation, but
as an intermediate representation to expose legal
concepts and retrieval-oriented terminology from
original queries.
In IRAC-Exp, we first prompt the LLM to per-
form an IRAC analysis of the user query, and then
extract a set of legally specialized keywords from
the resulting analysis. The extracted keywords are
appended to the original query to form an expanded
query for BM25 (Robertson and Zaragoza, 2009)
retrieval. For the specific prompts used for IRAC
analysis and keyword extraction, please refer to
Figure 4 and Figure 5 in Appendix A, respectively.
3.3 Structure-Guided Dense Path
This component is designed to address the limita-
tion of open-domain embedding models, which of-
ten fail to distinguish between semantically similar
articles that differ in legal authority, applicability,
or scope in a structured legal system. Specifically,
we train the embedding model with contrastive
3

learning and developStruct-Neg, a structure-aware
hard negative mining strategy based on the hierar-
chical and citation relationships among legal arti-
cles. Given a corpus C, where each article a∈C
is associated with a law title t, an article number p,
and article content x, we build hierarchy buckets
and a citation graph for hard negative mining.
Hierarchy Buckets ConstructionTo model the
hierarchical relationships of the legal system, we
assign each article aa discrete hierarchy label
L(a)∈ {0,1, . . . ,6} according to its title t:0
for the Constitution, 1for Primary Statute, 2for
Administrative Regulation, 3for Local Regulation
or Autonomous Regulations, 4for Departmental
Rule or Local Government Rule, 5for Judicial In-
terpretations, and 6for other normative documents.
This taxonomy is presented based on the Legisla-
tion Law (National People’s Congress of the Peo-
ple’s Republic of China, 2023): labels 0–4 form an
ordered hierarchy of upper-level and lower-level
authorities, while label 5 is treated separately.
Given a query q, we partition the retrieved candi-
date set Rinto hierarchy buckets according to the
hierarchy labels:
Bℓ(q) ={a∈R|L(a) =ℓ}.(1)
Citation Graph ConstructionTo capture cita-
tion relationships among legal articles, we con-
struct an article-level citation graph based on the
corpus. We define a directed graph G= (V, E)
with V=C and build an index function Φ :
(t, p)7→a that maps a normalized law title and
article number to the corresponding article node.
For each article ai, we extract citation mentions
from its content xusing pattern matching, includ-
ing both cross-law citations and internal citations.
Each mention is resolved to a target article through
title normalization and article-level lookup via Φ,
where internal citations are grounded to the title of
the source article itself. If a citation mention can
be successfully resolved to an article aj, we add a
directed edge(a i, aj)∈E.
Struct-NegBased on the hierarchy buckets and
the citation graph, the workflow of Struct-Neg is
shown in Algorithm 1. For each query, we first
determine the hierarchy label of its positive article
and retrieve the top- Kmost similar articles from
the corpus as candidates. We then sample negatives
from the same hierarchy, adjacent hierarchies, and
the citation graph to construct multi-source hardAlgorithm 1Workflow of Struct-Neg.
Input: Query q, positive sample p+, ground-truth
article set Aq, article corpus C, citation graph
G, hierarchy mapping L(·), retrieval depth K,
negative budgetM
Output:Negative sample setN q
1:ℓ+←L(p+)
2:R←DenseRetrieve(q, C, K)
3:R←R\A q
4:Partition Rinto hierarchy buckets {Bℓ}ac-
cording toL(·)
5:T← {ℓ+}
6:ifIsValidLevel(ℓ+−1)then
7:T←T∪ {ℓ+−1}
8:end if
9:ifIsValidLevel(ℓ++ 1)then
10:T←T∪ {ℓ++ 1}
11:end if
12:N q←SampleByHierarchy({B ℓ}, ℓ+, T, M)
13:G q←CitationNeighbors(p+, G)\A q
14:N q←N q∪G q
15:returnN q
negatives. In contrastive learning, these negatives
train the embedding model to better differentiate
articles that are semantically similar but possess
distinct hierarchical or citation relations.
3.4 Multi-Path Score Fusion
To effectively integrate lexical and semantic fea-
tures, we combine the scores of the sparse and
dense paths through weighted fusion. Let rsp(q, a)
andrde(q, a) denote the raw sparse and dense rel-
evance scores between query qand article a, re-
spectively. Since the two scores are produced in
different value ranges, we first normalize them as
¯rm(q, a) =1
2+1
πarctan 
rm(q, a)
,
m∈ {sp,de}.(2)
where ¯rm(q, a) denotes the normalized score of
retrieval path m. The final fused score is then com-
puted as
r(q, a) =α¯r sp(q, a) + (1−α) ¯r de(q, a),(3)
where α∈[0,1] controls the relative contribution
of the sparse and dense paths. The fused scores
rank the articles, and a pool of highly ranked arti-
clesR qis used as candidates for reranking.
4

3.5 Intent-Aware Reranker
Although the multi-path retrieval module retrieves
high-quality candidate articles, it does not explic-
itly model intent consistency between user queries
and legal articles. In real-world applications, such
intent priors are usually unavailable due to the
scarcity of domain annotations. To address this,
we leverage few-shot prompting to identify intent
labels and refine the final rankings.
Intent TaxonomyWe propose a task-oriented in-
tent taxonomy for legal article retrieval, drawing
on previous research on legal case retrieval and
the typology of legal rules (Nazarenko et al., 2018;
Shao et al., 2023): (1)Definition, which seeks the
interpretation or clarification of legal concepts and
terms; (2)Applicability, which concerns whether
and under what conditions the legal article applies
to the facts of the case; (3)Consequence, which
concerns legal liabilities, sanctions, or other legal
effects triggered by a factual situation or violation;
(4)Procedure, which concerns procedural require-
ments, execution steps, and related processes; and
(5)Others, a residual category covering articles
that do not fit the above intents, such as legislative
purpose clauses.
Intent-Aware RerankingWe use few-shot
prompting with the LLM to assign both queries
and legal articles to one of the predefined intent
labels. Separate few-shot prompt templates are de-
signed for queries and legal articles, as described
in Figure 6 and Figure 7, respectively. Given a
query qand its candidate article set Rq, we use an
LLM-based intent classifier to assign intent labels
to the query and each candidate articlea∈R q:
ˆyq= LLM intent(q),
ˆya= LLM intent(a),∀a∈R q,(4)
where ˆyqandˆyaare the predicted intent labels. To
verify intent reliability for reranking, we manually
evaluated 100 queries and 100 articles, where the
LLM classifier achieved 81% and 86% agreement
with human labels, respectively.
Given a query qand its candidate article set
Rq={a 1, . . . , a W}, we define the intent consis-
tency score for each candidate articlea∈R qas
si(q, a) =(
1,ˆy q= ˆya,
0,otherwise,(5)
where ˆyqandˆyaare the predicted intent labels of
the query and article.The final reranking score is then computed by
s(q, a) =λ 1sr(q, a) +λ 2sp(a) +λ 3si(q, a),(6)
where sr(q, a) denotes the neural reranker score,
andsp(a)denotes the prior score induced by the ini-
tial retrieval order. The top- Kresults are returned
based ons(q, a).
4 Experiments
4.1 Setup
DatasetsWe evaluate LEXPATHon two publicly
available benchmarks, where queries are from the
general public, and one self-constructed bench-
mark, where queries are from professionals:
•STARD(Su et al., 2024) contains 1,543 real-
world legal consultation queries and 55,348
candidate legal articles, with reference article
annotations but no reference answers.
•LexRAG(Li et al., 2025c) contains 1,013 pub-
lic legal consultation dialogues and 17,228
candidate legal articles, with reference arti-
cles, reference answers, and evaluation key-
words for each query.
•StatuteRAGis our self-constructed bench-
mark adapted from professional training mate-
rials in market supervision and administration.
It contains 1,361 queries, 56,982 candidate le-
gal articles, reference article annotations, and
reference answers. Details are provided in
Appendix B.
BaselinesWe evaluate LEXPATHagainst the
following baselines: (1)Lexical matching:
BM25 (Robertson and Zaragoza, 2009), QL (Ponte
and Croft, 2017); (2)Open-domain embedding:
text2vec-base-chinese (Ming, 2022), bge-large-zh-
v1.5 (Xiao et al., 2023), gte-Qwen2-1.5B (Li et al.,
2023b), bge-m3 (Chen et al., 2024), multilingual-
e5-large (Wang et al., 2024); (3)Legal-domain
embedding: ChatLaw_Text2Vec (Cui et al.,
2023), SAILER_zh (Li et al., 2023a); (4)Hybrid
score fusion: BM25 + bge-large-zh-v1.5, BM25
+ bge-m3; (5)Controlled neural reranking:
BM25 + jina-reranker-v2-base-multilingual (Jina
AI Team, 2025), bge-m3 + jina-reranker-v2-base-
multilingual, BM25 + bge-m3 + jina-reranker-v2-
base-multilingual; and (6)Adaptive RAG: IR-
CoT (Trivedi et al., 2023), CRAG (Yan et al.,
2024), A-RAG (Du et al., 2026). For adaptive
5

Table 1: Main retrieval results on STARD, LexRAG, and StatuteRAG. "R" stands for Recall, and "N" stands for
NDCG. Best and second-best results are shown inboldand underlined , respectively.
Dataset STARD LexRAG StatuteRAG
Model R@1 R@3 R@5 N@1 N@3 N@5 R@1 R@3 R@5 N@1 N@3 N@5 R@1 R@3 R@5 N@1 N@3 N@5
BM25 23.81 33.77 37.54 30.10 31.95 33.63 4.82 10.78 13.69 5.23 8.52 9.73 50.37 67.95 72.34 51.28 61.13 62.95
QL 20.77 31.93 36.49 26.21 29.39 31.33 4.64 9.35 12.54 4.93 7.52 8.85 50.55 65.75 71.25 51.65 59.94 62.17
text2vec-base-chinese 20.71 32.05 36.41 26.21 29.52 31.39 5.26 10.08 12.52 5.52 8.16 9.20 36.81 52.38 57.33 37.36 46.25 48.29
bge-large-zh-v1.5 27.05 39.38 45.95 33.98 36.98 39.80 9.78 17.42 20.73 10.26 14.48 15.90 41.94 61.72 69.78 43.22 54.34 57.70
gte-Qwen2-1.5B 19.90 30.20 32.49 25.89 28.23 29.00 4.98 8.96 11.51 5.13 7.38 8.44 38.10 61.90 71.06 39.19 52.36 56.22
bge-m3 27.31 40.88 47.47 35.92 38.34 41.08 10.54 19.12 23.60 11.05 15.70 17.61 43.04 64.65 72.53 44.32 56.28 59.53
multilingual-e5-large 28.61 38.57 44.72 36.89 37.40 40.01 7.23 14.28 17.95 7.59 11.50 13.09 45.42 66.67 73.26 46.52 58.48 61.22
ChatLaw_Text2Vec 6.39 11.31 14.70 8.09 9.75 11.23 1.97 4.98 7.25 2.17 3.83 4.79 5.31 12.09 15.02 5.49 9.22 10.41
SAILER_zh 15.18 23.37 26.58 18.12 21.41 22.76 2.76 5.62 7.05 2.86 4.43 5.02 31.68 44.51 51.47 31.87 39.48 42.39
BM25 + bge-large-zh-v1.5 28.82 40.04 43.98 36.25 38.12 39.83 8.61 15.17 19.64 9.17 12.64 14.54 45.97 67.40 75.82 46.89 59.06 62.60
BM25 + bge-m3 27.92 41.60 45.68 36.25 38.93 40.65 10.04 17.08 20.51 10.65 14.39 15.83 48.35 72.71 77.84 50.18 63.37 65.45
BM25 + jina-reranker 25.01 39.88 44.15 31.39 36.16 38.08 4.81 12.31 16.33 5.22 9.24 10.90 50.36 70.51 75.27 51.28 62.75 64.76
bge-m3 + jina-reranker 27.31 44.00 48.88 35.92 40.12 42.35 10.53 20.06 23.96 11.04 16.20 17.87 43.40 67.39 73.26 44.68 58.28 60.69
BM25 + bge-m3 + jina-reranker 28.55 44.08 48.77 37.54 40.90 42.97 10.83 19.55 23.22 11.53 16.02 17.59 47.98 67.94 73.44 49.81 60.24 62.50
IRCoT 19.56 27.08 30.98 23.62 25.49 27.16 3.50 8.17 10.49 3.65 6.24 7.24 46.34 61.72 66.67 46.89 55.60 57.66
CRAG 21.05 34.70 39.05 25.57 30.73 32.74 5.52 9.22 12.66 5.72 7.66 9.09 50.37 63.92 71.25 51.28 58.49 61.56
A-RAG 20.88 29.11 30.19 27.51 27.82 28.24 11.36 14.76 15.64 11.98 13.52 13.90 38.64 44.69 45.42 39.56 42.57 42.87
LEXPATH 37.54 51.78 56.31 47.90 49.64 51.68 16.86 26.53 32.45 17.85 22.72 25.23 66.67 87.91 93.04 68.13 79.86 81.98
RAG baselines, we adapt their query reformulation
or retrieval-control modules to legal article retrieval
while keeping the same base retriever and candidate
pool for fairness. Since our task focuses on Chinese
legal article retrieval, models that do not support
Chinese are excluded from the experiments.
MetricsFor evaluation, we use standard rank-
ing metrics, including Recall@ Kand NDCG@ K,
withK∈ {1,3,5} . We report small- Kresults be-
cause they better reflect the practical application
in the legal domain, where the system is expected
to precisely reference legal articles, and a larger
candidate set may increase noise.
Implementation DetailsEach dataset is split
into training, development, and test sets in a
7:1:2 ratio. For StatuteRAG, we split at the seed-
question level to avoid data leakage. Retrieval
experiments are conducted over the full article
corpus, with a candidate pool size of 20. We
use Qwen2.5-7B-Instruct (Qwen Team, 2024) for
query expansion, Qwen3-8B (Qwen Team, 2025)
for few-shot intent classification, jina-reranker-v2-
base-multilingual (Jina AI Team, 2025) for neural
reranking, and Milvus (Wang et al., 2021) for in-
dex management. For the dense path, we fine-tune
bge-large-zh-v1.5 (Xiao et al., 2023) with FlagEm-
bedding for 5 epochs using a batch size of 16 and a
learning rate of 1×10−5, and merge the fine-tuned
checkpoint with the original checkpoint at a 5:5
ratio. We set α= 0.4 and tune λ3on the develop-
ment split of each dataset, reporting the best per-
formance among the candidate settings. Parameter
analysis is provided in Section 4.5. All experimentsare implemented in Python 3.12 with Hugging Face
Transformers (Wolf et al., 2020), vLLM (Kwon
et al., 2023), and the OpenAI Python library (Ope-
nAI, 2026b), and run on a single NVIDIA A6000
GPU with 48GB of memory. Dense-path training
takes less than 0.5 GPU hours.
4.2 Main Results
The main results on STARD, LexRAG, and
StatuteRAG are presented in Table 1. We have
the following observations: (1) LEXPATHcon-
sistently outperforms the strongest baseline, im-
proving Recall@5 by 7.43, 8.85, and 15.24 on the
three datasets, respectively, showing the effective-
ness of multi-level legal relevance modeling. (2)
Legal-domain embeddings do not always outper-
form open-domain embeddings, as document-level
representations may not necessarily be suitable for
article-level retrieval. (3) Both hybrid retrieval
with reranking and adaptive RAG yield limited
and inconsistent performance gains. Relying solely
on general-purpose strategies such as combining
open-domain retrievers, reformulating queries, or
expanding to multi-turn retrieval is inherently in-
sufficient without domain-oriented relevance mod-
eling. (4) All baseline methods underperform on
LexRAG, largely due to its multi-turn consultation
setting. We further analyze these challenges in Ap-
pendix C and explore the impact of retrieval quality
on downstream legal QA in Appendix D.
4.3 Ablation Studies
We conduct ablation studies to show the effective-
ness of each component of LEXPATH, as shown
6

in Table 2, including the following variants: (1)
w/o Reranker: removing the intent-aware reranker;
(2) w/o Intent Consistency: removing the intent
consistency score in the reranker, leaving the neu-
ral reranker score and prior retrieval score used
for reranking; (3) w/o Sparse Path: removing the
sparse path, where only the dense path is used for
retrieving candidates; (4) w/o IRAC-Exp: remov-
ing the IRAC-Exp in the sparse path, where only
BM25 is used for hybrid score fusion; (5) w/o
Dense Path: removing dense path, where only the
sparse path is used for retrieving candidates; (6)
w/o Struct-Neg: removing training on negatives
extracted by Struct-Neg, only the original dense
retriever for hybrid score fusion.
The ablation results show that different compo-
nents contribute to LEXPATHat different stages
of the retrieval pipeline. (1) Removing the dense
path causes the largest performance drop across all
datasets, indicating that the dense path serves as the
main recall backbone of LEXPATH. (2) Removing
IRAC-Exp leads to a larger drop than removing the
entire sparse path, suggesting that an unexpanded
sparse path may introduce noisy lexical matches,
whereas IRAC-Exp makes sparse evidence more
complementary to the dense path. (3) Removing
Struct-Neg results in a moderate drop, showing
that hierarchy- and citation-aware hard negatives
help the dense path distinguish semantically simi-
lar but legally different articles. (4) Removing the
reranker or the Intent Consistency leads to smaller
yet consistent drops, suggesting that intent-aware
reranking is useful for distinguishing legally appli-
cable articles among candidate articles.
Table 2: Ablation results on three datasets measured by
Recall@5.
Method STARD LexRAG StatuteRAG
LEXPATH 56.31 32.45 93.04
w/o Reranker 55.34 31.16 90.84
w/o Intent Consistency 54.69 31.76 92.67
w/o Sparse Path 55.66 30.97 92.67
w/o IRAC-Exp 55.02 29.09 91.94
w/o Dense Path 51.46 22.49 84.98
w/o Struct-Neg 55.02 31.26 92.31
4.4 Design Choice Analysis
To further examine the key design choices in
LEXPATH, we conduct additional experiments on
STARD, focusing on IRAC-Exp in the sparse path
and Struct-Neg in the dense path.Comparison of Query Expansion Strate-
giesWe compare IRAC-Exp with representa-
tive query reformulation methods, including di-
rect query rewriting (QR) (Ma et al., 2023),
Query2Doc (Wang et al., 2023), and HyDE (Gao
et al., 2023). As shown in Table 3, IRAC-Exp
achieves the best performance among all expansion
methods. This suggests that explicitly decompos-
ing a legal query into issue, rule, application, and
conclusion helps expose retrieval-oriented legal ter-
minology that is often missing from the original
query. Compared with generic expansion meth-
ods, IRAC-Exp provides more task-specific lexical
evidence for legal article retrieval.
Table 3: Comparison of query expansion and negative
mining strategies measured by Recall@5.
Method STARD LexRAG StatuteRAG
BM25 37.54 13.71 72.16
+ QR 41.42 18.15 71.79
+ Query2Doc 44.98 21.99 76.56
+ HyDE 45.63 19.33 79.12
+ IRAC-Exp 47.57 24.65 81.68
bge-large-zh-v1.5 45.95 20.71 69.60
+ Random 53.40 29.68 87.55
+ ANN 54.05 28.50 86.45
+ Struct-Neg 55.34 30.18 89.74
Comparison of Hard Negative Mining Strategies
We further compare Struct-Neg with random and
approximate nearest neighbor (ANN)-based hard
negative mining strategies for dense retriever train-
ing. Random negatives provide weak supervision,
while ANN-based negatives mainly capture seman-
tic similarity and may overlook legal hierarchy or
citation relations. In contrast, Struct-Neg achieves
the best performance by sampling negatives from
the same or adjacent hierarchy levels and citation-
related articles. This indicates that structure-aware
negatives provide more informative supervision for
distinguishing semantically similar but legally dif-
ferent articles.
4.5 Parameter Analysis
In this section, we analyze the sensitivity of three
key parameters: the checkpoint-merge weight, the
sparse-path weight α, and the intent-consistency
score weightλ 3. We report Recall@5 in Figure 3.
Across the three datasets, moderate checkpoint-
merge weights generally perform better than us-
ing either checkpoint alone, suggesting that check-
point merging balances general semantic ability
7

2030405060708090100
00.10.20.30.40.50.60.70.80.91Recall@5
Checkpoint-merge WeightSTARD LexRAG StatuteRAG
2030405060708090100
00.10.20.30.40.50.60.70.80.91Recall@5
αSTARD LexRAG StatuteRAG
2030405060708090100
0 0.30.60.91.21.5Recall@5
λSTARD LexRAG StatuteRAGFigure 3: Parameter analysis of LEXPATHby varying
the checkpoint-merge weight, sparse-path weight α, and
intent-consistency score weightλ 3.
and task-specific structural signals. Performance
also remains stable within a moderate range of α,
confirming the complementarity between sparse
and dense paths. When αis close to 1.0, perfor-
mance drops, indicating that sparse matching alone
is insufficient for legal article retrieval. Finally,
LEXPATHis relatively insensitive to λ3, as moder-
ate intent-consistency score weights provide useful
reranking signals, while overly large weights bring
no clear additional gains.
4.6 Efficiency Analysis
Table 4 shows the accuracy-efficiency trade-off of
different LEXPATHconfigurations. The full LEX-
PATHachieves the best recall, but it requires addi-
tional online LLM calls and higher latency, reflect-
ing a trade-off between retrieval quality and infer-
ence cost. The dense path provides a low-latency
option with no online LLM calls while achieving
recall close to the full pipeline. The w/o Sparse
Path variant offers a middle-ground configuration,
retaining strong recall with fewer online LLM calls.
These results indicate that article-side structural fea-
tures can be effectively incorporated offline, while
query-side expansion and intent modeling provide
additional gains at a higher cost. Overall, LEX-
PATHsupports flexible deployment under different
latency budgets, from low-latency dense retrieval
to accuracy-oriented full retrieval.
4.7 Case Study
We present two representative cases to illustrate typ-
ical errors made by open-domain retrievers. Please
refer to Appendix E.Table 4: Efficiency analysis on STARD.
Method R@5 # LLM Call Latency (s)
BM25 37.54 0 0.33
bge-large-zh-v1.5 45.95 0 0.30
LEXPATH 56.31 3 10.12
w/o Reranker 55.34 2 6.79
w/o Sparse Path 55.66 1 3.61
Sparse Path 47.57 2 6.78
Dense Path 55.34 0 0.30
As shown in Figure 8, the first case illustrates a
typical error made by sparse retrievers. The top-3
articles retrieved by the sparse retriever contain
surface keyword matches such as "Food Safety
Law", but they miss the more informative phrase
"serious circumstances", which reflects the query’s
definition-seeking intent. In addition, sparse re-
trieval fails to distinguish articles with different
hierarchy levels. In contrast, LEXPATHranks the
ground-truth article first and assigns higher ranks
to articles from more appropriate legal levels.
As shown in Figure 9, the second case illustrates
a typical error made by dense retrievers. The top-3
articles retrieved by the dense retriever are semanti-
cally close to the query. Still, they give insufficient
attention to the keyword "suspend" and fail to dis-
tinguish it from the related but legally different
term "terminate". They also overlook differences
in legal hierarchy. In contrast, LEXPATHbetter
captures the keyword "suspend", ranks the ground-
truth article first, and promotes articles from more
appropriate legal levels.
5 Conclusion
We proposed LEXPATH, a domain-oriented frame-
work for legal article retrieval that captures rele-
vance through lexical matching, semantic retrieval,
and intent-level consistency. Specifically, LEX-
PATHcombines two complementary paths: the
sparse path enriched by IRAC-Exp with legal ter-
minology, and the dense path enhanced by Struct-
Neg with legal structure-based hard negatives. The
candidate articles are then refined by an intent-
aware reranker. Experiments on three benchmarks
show that LEXPATHconsistently outperforms lex-
ical, dense, hybrid, and adaptive RAG baselines,
with ablation studies confirming the effectiveness
of each component. Future work will explore its
transferability to other legal systems and integra-
tion with downstream tasks.
8

Limitations
There are several limitations in this work that we
plan to investigate and address in future work:
•This study focuses on Chinese legal article re-
trieval. While the proposed framework builds
on general legal characteristics such as termi-
nology, legal relations, and intent, its transfer-
ability to other languages and more complex
legal tasks remains to be explored.
•The proposed framework uses LLMs for query
expansion and intent classification, which in-
evitably adds inference cost. This cost can
be partly mitigated because most article-side
information can be precomputed and cached.
Future work may explore more efficient query-
side modeling.
•This study focuses on retrieval-side optimiza-
tion and uses a task-oriented intent taxonomy.
While the taxonomy captures common func-
tional roles of legal articles, some queries and
articles may involve multiple or implicit in-
tents. Future work may refine the taxonomy
and extend the domain-oriented design to le-
gal systems where retrieval, legal reasoning,
evidence verification, and answer generation
are jointly optimized.
Ethics Statement
Our proposed dataset is adapted from an exist-
ing publicly available professional training manual
with appropriate authorization for non-commercial
research use. We properly cite the source and en-
sure that our use is consistent with both the autho-
rized scope and the original educational purpose
of the manual. The source manual is a published
professional training resource that has undergone
editorial review and provides well-structured ques-
tions, explanations, and legal article references.
The constructed dataset is manually verified by the
authors, who are native Chinese speakers and were
instructed by legal professionals on the annotation
criteria before verification. No external annotators
were involved, and the verification process does
not require reviewing private user data, sensitive
personal information, or harmful content. There-
fore, our verification focuses on the quality of the
conversion process, including query completeness,
answer consistency, and article support. The ver-
ification process takes approximately 16 working
hours per author.References
Anthropic. 2026. Claude Sonnet 4.6. https://www.
anthropic.com/news/claude-sonnet-4-6 . Ac-
cessed: 2026-05-23.
Paheli Bhattacharya, Kripabandhu Ghosh, Saptarshi
Ghosh, Arindam Pal, Parth Mehta, Arnab Bhat-
tacharya, and Prasenjit Majumder. 2019. Fire 2019
aila track: artificial intelligence for legal assistance.
InProceedings of the 11th annual meeting of the fo-
rum for information retrieval evaluation, pages 4–6.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.
Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and
Li Yuan. 2023. Chatlaw. https://github.com/
PKU-YuanGroup/ChatLaw.
Chenlong Deng, Zhicheng Dou, Yujia Zhou, Peitian
Zhang, and Kelong Mao. 2024. An element is worth
a thousand words: Enhancing legal case retrieval by
incorporating legal elements. InFindings of the As-
sociation for Computational Linguistics: ACL 2024,
pages 2354–2365.
Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan
Wang, Pengyu Wang, Xiaorui Wang, and Zhen-
dong Mao. 2026. A-rag: Scaling agentic retrieval-
augmented generation via hierarchical retrieval inter-
faces.arXiv preprint arXiv:2602.03442.
Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou,
Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen,
Zhixin Yin, Zongwen Shen, and 1 others. 2024. Law-
bench: Benchmarking legal knowledge of large lan-
guage models. InProceedings of the 2024 conference
on empirical methods in natural language processing,
pages 7933–7962.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chen-
hui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Han-
lin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai
Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang,
Jing Zhang, Juanzi Li, and 37 others. 2024. Chatglm:
A family of large language models from glm-130b to
glm-4 all tools.Preprint, arXiv:2406.12793.
Randy Goebel, Yoshinobu Kano, Mi-Young Kim, Ju-
liano Rabelo, Ken Satoh, and Masaharu Yoshioka.
2023. Summary of the competition on legal infor-
mation, extraction/entailment (coliee) 2023. InPro-
ceedings of the nineteenth international conference
on artificial intelligence and law, pages 472–480.
9

Abe Bohan Hou, Orion Weller, Guanghui Qin, Eugene
Yang, Dawn Lawrie, Nils Holzenberger, Andrew
Blair-Stanek, and Benjamin Van Durme. 2025. Clerc:
A dataset for us legal case retrieval and retrieval-
augmented analysis generation. InFindings of the
Association for Computational Linguistics: NAACL
2025, pages 7898–7913.
Jina AI Team. 2025. jina-reranker-v2-base-multilingual.
Accessed: 2026-04-24.
Chaeeun Kim, Jinu Lee, and Wonseok Hwang. 2025.
Legalsearchlm: Rethinking legal case retrieval as
legal elements generation. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing, pages 4521–4554.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. InProceedings of the 29th
Symposium on Operating Systems Principles, pages
611–626.
Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Yueyue
Wu, Yiqun Liu, Chong Chen, and Qi Tian. 2023a.
Sailer: Structure-aware pre-trained language model
for legal case retrieval.Preprint, arXiv:2304.11370.
Haitao Li, Qingyao Ai, Xinyan Han, Jia Chen, Qian
Dong, and Yiqun Liu. 2025a. Delta: Pre-train a dis-
criminative encoder for legal case retrieval via struc-
tural word alignment. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 39,
pages 27072–27080.
Haitao Li, Junjie Chen, Jingli Yang, Qingyao Ai, Wei
Jia, Youfeng Liu, Kai Lin, Yueyue Wu, Guozhi Yuan,
Yiran Hu, and 1 others. 2025b. Legalagentbench:
Evaluating llm agents in legal domain. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 2322–2344.
Haitao Li, Yifan Chen, Hu YiRan, Qingyao Ai, Jun-
jie Chen, Xiaoyu Yang, Jianhui Yang, Yueyue Wu,
Zeyang Liu, and Yiqun Liu. 2025c. Lexrag: Bench-
marking retrieval-augmented generation in multi-turn
legal consultation conversation. InProceedings of
the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval,
pages 3606–3615.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023b. Towards
general text embeddings with multi-stage contrastive
learning.arXiv preprint arXiv:2308.03281.
Antoine Louis and Gerasimos Spanakis. 2022. A statu-
tory article retrieval dataset in french. InProceedings
of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 6789–6803.Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. InProceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 5303–5315.
Xu Ming. 2022. text2vec: A tool for text to vector.
National People’s Congress of the People’s Re-
public of China. 2023. Legislation law
of the people’s republic of china. https:
//www.npc.gov.cn/zgrdw/englishnpc/Law/
2007-12/12/content_1383899.htm . Adopted on
March 15, 2000; amended on March 15, 2015;
revised on March 13, 2023.
Adeline Nazarenko, François Levy, and Adam Wyner.
2018. An annotation language for semantic search
of legal sources. InProceedings of the Eleventh
International Conference on Language Resources
and Evaluation (LREC 2018).
OpenAI. 2026a. GPT-5.5. https://platform.
openai.com/docs/models. Accessed: 2026-05-23.
OpenAI. 2026b. Openai python api library. https://
github.com/openai/openai-python . Accessed:
2026-05-13.
Nicholas Pipitone and Ghita Houir Alami. 2024.
Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain.arXiv
preprint arXiv:2408.10343.
Jay M Ponte and W Bruce Croft. 2017. A language
modeling approach to information retrieval. InACM
SIGIR Forum, volume 51, pages 202–208. ACM New
York, NY , USA.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Stephen Robertson and Hugo Zaragoza. 2009.The prob-
abilistic relevance framework: BM25 and beyond,
volume 4. Now Publishers Inc.
Sergio Servantez, Joe Barrow, Kristian Hammond, and
Rajiv Jain. 2024. Chain of logic: Rule-based rea-
soning with large language models.arXiv preprint
arXiv:2402.10400.
Shanghai Municipal Administration for Market Regula-
tion. 2021.Market Supervision and Administration
Practical Training: Analysis of 500 Questions and
Answers. China Industry and Commerce Press, Bei-
jing.
Yunqiu Shao, Haitao Li, Yueyue Wu, Yiqun Liu,
Qingyao Ai, Jiaxin Mao, Yixiao Ma, and Shaoping
Ma. 2023. An intent taxonomy of legal case retrieval.
ACM Transactions on Information Systems, 42(2):1–
27.
10

Weihang Su, Yiran Hu, Anzhe Xie, Qingyao Ai, Quezi
Bing, Ning Zheng, Yun Liu, Weixing Shen, and
Yiqun Liu. 2024. Stard: A chinese statute re-
trieval dataset derived from real-life queries by non-
professionals. InFindings of the Association for
Computational Linguistics: EMNLP 2024, pages
10658–10671.
Weihang Su, Baoqing Yue, Qingyao Ai, Yiran Hu, Jiaqi
Li, Changyue Wang, Kaiyuan Zhang, Yueyue Wu,
and Yiqun Liu. 2025. Judge: Benchmarking judg-
ment document generation for chinese legal system.
InProceedings of the 48th International ACM SI-
GIR Conference on Research and Development in
Information Retrieval, pages 3573–3583.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin,
Peng Xu, Shengjun Li, Xiangyu Wang, Xiangzhou
Guo, Chengming Li, Xiaohai Xu, and 1 others. 2021.
Milvus: A purpose-built vector data management sys-
tem. InProceedings of the 2021 international con-
ference on management of data, pages 2614–2627.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024. Multilin-
gual e5 text embeddings: A technical report.arXiv
preprint arXiv:2402.05672.
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 9414–9423.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
and 1 others. 2020. Transformers: State-of-the-art
natural language processing. InProceedings of the
2020 conference on empirical methods in natural
language processing: system demonstrations, pages
38–45.
Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao
Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xi-
anpei Han, Zhen Hu, Heng Wang, and 1 others. 2018.
Cail2018: A large-scale legal dataset for judgment
prediction.arXiv preprint arXiv:1807.02478.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding.Preprint,
arXiv:2309.07597.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
1
你是一位经验丰富的法律顾问。请针对
下面的法律问题，使用IRAC方法进行分
析并回答，即:
1. 首先明确问题（Issue）
2. 接着说明相关法律规则（Rule）
3. 然后分析事实如何适用这些规则
（Application）
4. 最后给出结论（Conclusion）
问题: {question}
English translation:
You are an experienced legal advisor. 
Please analyze and answer the following 
legal question using the IRAC method: 
1. First, identify the issue; 
2. Next, state the relevant legal rules; 
3. Then, analyze how the facts apply to 
these rules; 
4. Finally, draw a conclusion. 
Question: {question}Prompts for IRAC AnalysisFangyi Yu, Lee Quartey, and Frank Schilder. 2022. Le-
gal prompting: Teaching a language model to think
like a lawyer.arXiv preprint arXiv:2212.01326.
A Prompt for LEXPATH
The prompts used in IRAC-Exp are shown in Fig-
ure 4 for IRAC analysis and Figure 5 for keyword
extraction. The prompts used for few-shot intent
classification are shown in Figure 6 for queries and
Figure 7 for legal articles.
B Construction Process of StatuteRAG
Existing benchmarks of legal article retrieval (Su
et al., 2024; Li et al., 2025c) are collected from
daily legal consultations submitted by the gen-
eral public. To supplement the application scenar-
ios for the legal article retrieval task, we propose
StatuteRAG, in which queries are adapted from
questions used to train domain professionals.
The seed questions to build StatuteRAG are se-
lected from a publicly available manual (Shanghai
Municipal Administration for Market Regulation,
2021) used for training legal professionals. The
manual focuses on market supervision and admin-
istration laws and contains 500 single-choice ques-
tions, together with explanatory analysis and legal
articles reference for each option. The process of
constructing a query dataset is as follows:
11

1
任务：
1. 理解问题的语义并改写为更专业的法
律术语。
2. 执行同义词替换，确保问题的核心法
律含义不变。
3. 提取出用于检索相关法律条文的关键
词。
4. 请输出尽可能多的关键词。
5. 请直接以[关键词1, 关键词2,...]的格式
输出关键词列表，不要有多余的内容。
问题：{question}
关键词列表：
English translation:
Task:
1. Understand the meaning of the question 
and rephrase it using more professional 
legal terminology.
2. Replace words with synonyms while 
ensuring the core legal meaning of the 
question remains unchanged.
3. Extract keywords to be used for searching 
relevant legal provisions.
4. Please provide as many keywords as 
possible.
5. Please output the keyword list directly in 
the format [Keyword 1, Keyword 2, ...] 
without any additional content.
Question: {question}
Keyword list:Prompts for Keyword Extraction•Exclude questions that are overly trivial, am-
biguous, overly dependent, incomplete con-
text, or lack a clear legal article reference, and
use the remaining ones as seed questions.
•For each seed question, use regular expres-
sions to extract the text from the question stem
and each answer choice.
•Combine the question stem with the content
of each answer choice to create a true or false
question, and match it with the corresponding
legal articles.
To ensure data quality, the constructed sam-
ples are manually reviewed. Since StatuteRAG is
adapted from a professional training manual with
existing explanations and legal article references,
the review focuses on verifying the conversion qual-
ity, including query completeness, answer consis-
tency, and article support. Samples with unclear
wording, missing context, or weak article support
are removed.
The resulting dataset contains 1,361 queries an-
notated with relevant legal articles and ground-truth
answers, together with a corpus of 56,982 candi-
date legal articles compiled from 1,295 effective
legal documents at the time of collection. The
statistics of StatuteRAG are shown in Table 5.
Table 5: Statistics of StatuteRAG.
Statistic Value
Number of queries 1,361
Number of candidate legal articles 56,982
Number of referenced article annotations 1,406
Average number of articles referenced per query 1.03
Average query length 72.57
Average legal article length 86.90
C Challenges on LexRAG
LexRAG is challenging because its queries are de-
rived from real multi-turn legal consultations. Com-
pared with STARD and StatuteRAG, the informa-
tion needed for retrieval is not always contained in
a single self-contained query. Relevant facts, legal
issues, and user intents may be distributed across
multiple dialogue turns, which partly explains the
relatively lower recall on LexRAG.
First, LexRAG requires retrievers to handle
dialogue-dependent information. Key facts may
appear in earlier turns rather than in the final user
question alone. In the following example, the last
12

1
任务：请判断下列查询的意图类型，必须从以下五种类型中选择一种：定义类、适用类、后果类、
程序类或其它类。
以下是意图类型的示例：
定义类：解释概念、术语定义 
查询：责令停产停业是否属于行政处罚？ 冒用他人姓名，构成侵害姓名权吗？录音和转账记录是
否可以作为有效证据？
适用类：描述适用条件和范围 
查询：什么情况下可解除劳动合同？ 那他需要什么条件才会发生作用？有哪些情况无须肖像权人
的许可，就可以使用别人的肖像？
后果类：描述法律责任和处罚 
查询：盗窃罪会受到什么处罚？ 货车右转，十字路口，开了转向灯，后面小轿车追尾，责任怎么
付？仓储物毁损、灭失，谁承担责任？
程序类：描述法律手续和流程 
查询：如何申请专利？这个期间如果出现纠纷怎么办？产品侵权找哪个法院起诉？ 
其它类：立法目的、例外规定等 
查询：制定环境保护法的目的是什么？ 那还有其他方式？制定合同法的目的是什么？
查询：{question} 
仅输出"定义类"、"适用类"、"后果类"、"程序类"或"其它类"中的一种。
English translation:
Task: Please identify the intent type of the following queries. You must select one of the following five 
types: Definition, Applicability, Consequence, Procedure, or Other. 
Examples of intent types: 
Definition: Explaining concepts and defining terms. 
Query: Does an order to suspend production or business operations constitute an administrative penalty? 
Does impersonating another person’s name constitute an infringement of the right to one’s name? Can 
audio recordings and transaction records serve as valid evidence? 
Applicability: Describing conditions and scope of application. 
Query: Under what circumstances can an employment contract be terminated? What conditions must be 
met for it to take effect? In what situations can a person’s likeness be used without the likeness holder’s 
permission? 
Consequence: Describes legal liabilities and penalties. 
Query: What penalties apply for the crime of theft? A truck signals to turn right at an intersection, and a car 
behind it rear-ends it—how is liability determined? Who bears liability for the damage or loss of stored 
goods? 
Procedure: Describes legal procedures and processes. 
Query: How do I apply for a patent? What should be done if a dispute arises during this period? Which 
court should one file a lawsuit with for product infringement? 
Other: Legislative purposes, exceptions, etc. 
Query: What is the purpose of enacting the Environmental Protection Law? Are there other methods? What 
is the purpose of enacting the Contract Law? 
Query: {question} 
Output only one of the following categories: Definition, Applicability, Consequence, Procedure, or Other. Prompts for Query Intent Classification13

1
任务：请判断下列法条的意图类型，必须从以下五种类型中选择一种：定义类、适用类、后果类、
程序类或其它类。
以下是意图类型的示例：
定义类：解释概念、术语定义 
法条：知识产权是权利人依法就...享有的专有权利；合同是民事主体之间设立、变更、终止民事法
律关系的协议；自然人的民事权利能力一律平等。
适用类：描述适用条件和范围 
法条：用人单位有下列情形之一的，劳动者可以解除劳动合同...；有下列情形之一的，出租人或者
承租人可以解除融资租赁合同...；有下列情形之一的，监护关系终止...
后果类：描述法律责任和处罚 
法条：盗窃公私财物，数额较大的，处三年以下有期徒刑... ；因承包人的原因致使建设工程在合
理使用期限内造成人身损害和财产损失的，承包人应当承担赔偿责任；代理人不履行或者不完全
履行职责，造成被代理人损害的，应当承担民事责任。
程序类：描述法律手续和流程 
法条：申请发明或者实用新型专利的，应当提交请求书...；当事人协商一致，可以变更合同；对监
护人的确定有争议的，由被监护人住所地的居民委员会...
其它类：立法目的、例外规定等 
法条：为保护和改善环境，防治污染...；为了保护个人信息权益，规范个人信息处理活动...；为保
护文学、艺术和科学作品作者的著作权...
法条：{content} 
仅输出"定义类"、"适用类"、"后果类"、"程序类"或"其它类"中的一种。
English translation:
Task: Please identify the type of article for the following legal articles by selecting one of the following 
five categories: Definition, Applicability, Consequence, Procedure, or Other. 
Examples of intent types: 
Definition: Explains concepts or defines terms. 
Legal articles: Intellectual property refers to the exclusive rights enjoyed by the right holder in accordance 
with the law regarding...; A contract is an agreement between civil entities to establish, modify, or 
terminate civil legal relationships; All natural persons have equal capacity for civil rights. 
Applicability: Describes conditions and scope of application. 
Legal articles: Where an employer falls under any of the following circumstances, an employee may 
terminate the employment contract...; Where any of the following circumstances exist, the lessor or lessee 
may terminate the financial lease contract...; Where any of the following circumstances exist, the 
guardianship relationship shall terminate... 
Consequence: Describes legal liabilities and penalties. 
Legal articles: Whoever steals public or private property of a substantial value shall be sentenced to fixed-
term imprisonment of not more than three years...; If, due to the contractor’s fault, a construction project 
causes personal injury or property damage within its reasonable service life, the contractor shall bear 
liability for compensation; If an agent fails to perform or performs his duties incompletely, thereby causing 
damage to the principal, he shall bear civil liability. 
Procedure: Describes legal procedures and processes. 
Legal articles: To apply for a patent for an invention or a utility model, a request form shall be submitted...; 
Parties may amend a contract by mutual agreement; In the event of a dispute regarding the appointment of 
a guardian, the residents’ committee at the ward’s place of residence shall... 
Other: Legislative purposes, exceptions, etc. 
Legal articles: To protect and improve the environment and prevent pollution...; To protect personal 
information rights and regulate personal information processing activities...; To protect the copyright of 
authors of literary, artistic, and scientific works... 
Legal article: {content} 
Output only one of the following categories: Definition, Applicability, Consequence, Procedure, or Other. Prompts for Legal Article Intent Classification14

turn asks how the user can protect their rights, but
the underlying issue depends on previous turns
about a financed car, insurance requirements, and
vehicle repossession.
Turn 1: If I did not purchase the second-year insurance for
my financed car through the finance company, do they have
the right to repossess the vehicle? I have been making my
monthly payments on time.
Turn 2: Should we specify the method of purchasing insur-
ance in this agreement?
Turn 3: Is it legal for a finance company to repossess a
vehicle?
Turn 4: Is there any other way for this company to hold the
other party accountable?
Turn 5: What can I do to protect my rights if he forcibly
impounds my vehicle?
Second, LexRAG contains many queries from
ordinary users, which increases the gap between
user expressions and legal language. Users often
describe concrete facts in everyday language with-
out explicitly mentioning the legal concepts useful
for retrieval. For example, the following query de-
scribes a lost phone and surveillance footage, but
does not directly state concepts such as evidence
collection, privacy protection, or access to records.
I lost my phone in the classroom this morning, but when
I went to the school to check the surveillance footage, they
refused to let me.
Third, the legal intent of a LexRAG query can be
implicit or underspecified. In the following exam-
ple, the user asks whether parents can give resettle-
ment housing to their grandson without a property
deed, while the reference article concerns the effec-
tiveness of real property rights through registration.
Although the query and the article are legally re-
lated, the connection is indirect and may not be
captured by a single intent label.
Query: Hello, my parents would like to give their resettle-
ment housing to their grandson, but they currently do not have
a property deed.
Reference Article: Article 209 of the Civil Code of the Peo-
ple’s Republic of China: The creation, modification, transfer,
and termination of real property rights take effect upon reg-
istration in accordance with the law; they do not take effect
without such registration, unless otherwise provided by law.
Ownership of natural resources that belong to the State by law
may be exempt from registration.
These examples suggest that LexRAG requires
more context-aware retrieval and more flexible in-
tent modeling. We leave the integration of dialogue
context and multi-intent retrieval as future work.
D Impact on Downstream Legal QA
To assess the impact of retrieval quality on down-
stream legal QA, we evaluate two lightweight mod-els, GLM4-9B (GLM et al., 2024) and Qwen2.5-
7B-Instruct (Qwen Team, 2024), together with
two flagship models, Claude4.6-Sonnet (Anthropic,
2026) and GPT-5.5 (OpenAI, 2026a), under four
retriever settings: zero-shot, BM25, bge-large-zh-
v1.5, and LEXPATH. Following (Li et al., 2025c),
keyword accuracy is adopted as the metric for
short-answer queries in LexRAG. For StatuteRAG,
which consists entirely of true-or-false queries, an-
swer accuracy is reported. STARD is excluded
because it does not provide reference answers.
Experimental results in Table 6 suggest the fol-
lowing findings: (1) Model scale alone does not
guarantee stronger performance in domain-specific
QA, as flagship models do not consistently outper-
form lightweight models in the zero-shot setting.
(2) LEXPATHgenerally brings larger gains than
BM25 and bge-large-zh-v1.5, showing the value of
domain-oriented retrieval design. (3) The improve-
ments are especially clear for lightweight models
and StatuteRAG, indicating that high-quality re-
trieval can benefit resource-constrained legal QA
scenarios. (4) For flagship models on LexRAG,
retrieval-augmented results are less stable, possibly
because limited retrieval recall introduces noisy or
incomplete evidence.
Table 6: Performance of different retriever settings on
downstream legal QA. "BGE" denotes bge-large-zh-
v1.5.
Generator Retriever LexRAG StatuteRAG
GLM4-9BZero-Shot 22.49 62.64
BM25 23.77 77.29
BGE 28.21 75.09
LEXPATH30.18 78.39
Qwen2.5-7BZero-Shot 31.07 76.19
BM25 27.81 76.56
BGE 27.51 75.46
LEXPATH33.53 78.75
Claude4.6-SonnetZero-Shot33.3367.77
BM25 25.25 76.92
BGE 27.32 72.16
LEXPATH29.09 86.81
GPT-5.5Zero-Shot32.2580.59
BM25 23.77 84.25
BGE 27.51 82.78
LEXPATH28.01 87.91
E Case Study
Figure 8 shows a typical failure case of sparse re-
trieval. The query asks whether Zhao’s conduct
constitutes a "serious circumstance" under the Reg-
ulations for the Implementation of the Food Safety
15

Query: Does the fact that Zhao continued to sell food contaminated by packaging materials after receiving criminal punishment for violating the Food Safety 
Law constitute a “serious circumstance” as defined in the Regulations on the Implementation of the Food Safety Law?
Ground-Truth Article: 
Article 67, Regulations for the Implementation of the Food Safety Law:
Any of the following circumstances shall be deemed to constitute “serious circumstances” as defined in Articles 123 through 126 and Article 132 of the Food 
Safety Law, as well as Articles 72 and 73 of these Regulations: (1) The value of the products involved in the violation exceeds 20,000 yuan, or the violation has 
continued for more than three months...
 Administrative Regulation
 Definition
Definition
Article 20, Measures for the Supervision and Administration of Salt Quality and Safety:
Where illegal acts related to the production and operation of table salt are already addressed in laws and regulations such as the Food Safety Law and its 
implementing regulations, such provisions shall apply.
Article 75, Regulations for the Implementation of the Food Safety Law of the People's Republic of China:
Where food production and operation enterprises or other entities commit violations as defined in the Food Safety Law, in addition to penalties imposed 
in accordance with the provisions of the Food Safety Law, if any of the following circumstances apply, the legal representative, principal responsible 
person, ... shall be fined an amount equivalent to ..: (1) intentionally committing a violation; (2) ...; (3) ...
Article 70, Regulations for the Implementation of the Food Safety Law:
Except for the circumstances specified in Article 125, paragraph 1, and Article 126 of the Food Safety Law, if the production and operation activities of 
food producers and operators fail to comply with the provisions of Article 33, ..., penalties shall be imposed in accordance with Article 126, paragraph 1, 
of the Food Safety Law and Article 75 of these Regulations.Top-3 Articles Retrieved by Sparse Retriever: 
Departmental Rule
 Procedure
 Administrative Regulation
 Applicability
 Administrative Regulation
 Procedure
Article 67, Regulations for the Implementation of the Food Safety Law:
Any of the following circumstances shall be deemed to constitute “serious circumstances” as defined in Articles 123 through 126 and Article 132 of the 
Food Safety Law, as well as Articles 72 and 73 of these Regulations: (1) The value of the products involved in the violation exceeds 20,000 yuan, or the 
violation has continued for more than three months...
Article 75, Regulations for the Implementation of the Food Safety Law of the People's Republic of China:
Where food production and operation enterprises or other entities commit violations as defined in the Food Safety Law, in addition to penalties imposed 
in accordance with the provisions of the Food Safety Law, if any of the following circumstances apply, the legal representative, principal responsible 
person, ... shall be fined an amount equivalent to ..: (1) intentionally committing a violation; (2) ...; (3) ...
Article 143, Criminal Law of the People's Republic of China:
Any person who produces or sells food that fails to meet food safety standards... shall be sentenced to fixed-term imprisonment of not more than three 
years or criminal detention, ...; if such acts cause serious harm to human health or involve other aggravating circumstances...; if the consequences are 
particularly severe...Top-3 Articles Retrieved by LexPath: 
Definition
  Administrative Regulation
Applicability
  Administrative Regulation
Consequence
 Primary Statute✓ Ground-truth match✗ Hierarchy mismatch, intent mismatch
✗ Intent mismatch
✗ Intent mismatchFigure 8: Case study of a standalone sparse retrieval failure. The sparse retriever ranks articles with keyword overlap
but mismatched hierarchy or intent, while LEXPATHretrieves the ground-truth definitional article at the top.
16

Law, so the ground-truth article is a definitional
article that directly explains this term. However,
the sparse retriever ranks articles that contain over-
lapping keywords such as "Food Safety Law" and
"food", but these articles either come from a mis-
matched legal hierarchy or focus on applicabil-
ity and procedure rather than definition. In con-
trast, LEXPATHranks the ground-truth article first.
This case shows that keyword overlap alone is not
enough for legal article retrieval, and that hierarchy
and intent information help identify the article that
directly supports the query.
Figure 9 shows a typical dense retrieval error.
The query asks whether the reconsideration pro-
cess should be suspended after the applicant goes
missing. The dense retriever returns articles that
are related to administrative reconsideration or ad-
ministrative procedures, but these articles mainly
concern other situations, such as non-suspension
of enforcement or termination of investigation.
They also come from lower-level departmental
rules. LEXPATHranks the ground-truth article first,
which directly states that reconsideration shall be
suspended when the applicant is missing. This case
shows that legal retrieval needs to distinguish not
only semantic similarity, but also legal hierarchy
and the exact procedural condition.
17

Query: Li filed an application for administrative reconsideration, challenging the administrative penalty decision issued by the market regulation authority. Li 
went missing in an accident. Should the administrative reconsideration proceedings be suspended?
Ground-Truth Article: 
Article 41, Regulations for the Implementation of the Administrative Reconsideration Law:
Administrative reconsideration shall be suspended if any of the following circumstances arise during the administrative reconsideration process and affect the 
adjudication of the case: (1) ...; (2) ...; (3) ...; (4) Where the natural person who is the applicant is missing or has been declared missing; ... 
 Administrative Regulation
 Procedure
Procedure
Article 199, Regulations on Procedures for the Handling of Administrative Cases by Public Security Organs:
Unless otherwise provided by law, the enforcement of an administrative disposition shall not be suspended if the party subject to the disposition files an 
application for administrative reconsideration or initiates administrative litigation against the disposition.
Article 47, Regulations on Administrative Penalty Procedures for Market Supervision and Administration:
If an investigation cannot proceed due to the death of a natural person suspected of violating the law, or the dissolution of a legal person or other 
organization, and there is no successor to assume the rights and obligations, the investigation shall be terminated upon approval by the head of the 
market supervision and administration department.
Article 48, Measures for Patent Administrative Law Enforcement:
If, after the department responsible for patent administration has issued a penalty decision, the party concerned applies for administrative 
reconsideration or files an administrative lawsuit with the people’s court, the enforcement of the decision shall not be suspended during the period of 
administrative reconsideration or litigation.Top-3 Articles Retrieved by Dense Retriever: 
Departmental Rule
 Procedure
Article 41, Regulations for the Implementation of the Administrative Reconsideration Law:
Administrative reconsideration shall be suspended if any of the following circumstances arise during the administrative reconsideration process and 
affect the adjudication of the case: (1) ...; (2) ...; (3) ...; (4) Where the natural person who is the applicant is missing or has been declared missing; ... 
Article 41, Law of the People's Republic of China on Administrative Reconsideration:
During the administrative reconsideration process, the administrative reconsideration authority shall decide to terminate the administrative 
reconsideration if any of the following circumstances arise: (1) the applicant withdraws the application for administrative reconsideration, and the 
administrative reconsideration authority approves the withdrawal; (2) ...
Article 42, Regulations for the Implementation of the Administrative Reconsideration Law:
Administrative reconsideration shall be terminated under any of the following circumstances: (1) where the applicant requests to withdraw the 
application for administrative reconsideration and the administrative reconsideration authority grants such withdrawal; (2) ...Top-3 Articles Retrieved by LexPath: 
Procedure
  Administrative Regulation
Procedure
 Primary Statute
Procedure
 Primary Statute
Departmental Rule
 Procedure
Departmental Rule
 Procedure✗ Hierarchy mismatch
✗ Hierarchy mismatch
✗ Hierarchy mismatch
✓ Ground-truth matchFigure 9: Case study of a standalone dense retrieval failure. The dense retriever retrieves semantically related
procedural articles but fails to distinguish legal hierarchy and the key condition of suspension, while LEXPATH
ranks the ground-truth article first.
18