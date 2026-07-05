# One Retrieval to Cover Them All: Co-occurrence-Aware Knowledge Base Reorganization for Session-Level RAG

**Authors**: Shivam Ratnakar, Yixuan Zhu, Cecilia Cheng, Chaya Vijayakumar

**Published**: 2026-06-30 05:35:10

**PDF URL**: [https://arxiv.org/pdf/2606.31156v1](https://arxiv.org/pdf/2606.31156v1)

## Abstract
RAG systems retrieve documents optimized for answering one query at a time. Yet enterprise users arrive with sessions, that is, coherent episodes of related questions that span semantically distant parts of the knowledge base. We show that a single retrieval call over a standard knowledge base covers only 41% of a user's session-level information need. To close this gap, we reorganize the KB offline using co-occurrence-aware clustering and expand retrieval candidates through cluster neighborhoods at query time. On WixQA (6,221 enterprise support articles), our method raises single-query session coverage to 58% (+17% absolute; 95% CI: [14.1, 20.4]), reduces retrieval calls to 70% coverage by 34%, and compresses the KB to 20% of its original size, all consistently across four embedding models and six functional domains. We argue that session-level coverage, not single-query recall, should be the primary metric for enterprise RAG evaluation.

## Full Text


<!-- PDF content starts -->

One Retrieval to Cover Them All:
Co-occurrence-Aware Knowledge Base Reorganization
for Session-Level RAG*
Shivam Ratnakar Yixuan Zhu Cecilia Cheng Chaya Vijayakumar
University of Southern California
{sratnaka, yzhu1458, huixiche, cvijayak}@usc.edu
Abstract
RAG systems retrieve documents optimized
for answeringone query at a time. Yet en-
terprise users arrive withsessions, that is, co-
herent episodes of related questions that span
semantically distant parts of the knowledge
base. We show that a single retrieval call over
a standard knowledge base covers only 41%
of a user’s session-level information need. To
close this gap, we reorganize the KB offline us-
ing co-occurrence-aware clustering and expand
retrieval candidates through cluster neighbor-
hoods at query time. On WixQA (6,221 enter-
prise support articles), our method raises single-
query session coverage to 58% (+17% absolute;
95% CI: [14.1, 20.4]), reduces retrieval calls
to 70% coverage by 34%, and compresses the
KB to 20% of its original size, all consistently
across four embedding models and six func-
tional domains. We argue that session-level
coverage, not single-query recall, should be the
primary metric for enterprise RAG evaluation.
1 Introduction
Consider a user who contacts Wix customer sup-
port and asks:“How do I connect my own domain?”
A standard RAG system (Lewis et al., 2021) re-
trieves domain-configuration articles and generates
a helpful answer. But the user’s actual need is
broader: they also want to change their site tem-
plate, configure payment processing, and set up
email forwarding. These articles live in entirely dif-
ferent semantic neighborhoods of the knowledge
base. The user must ask four separate questions,
trigger four separate retrieval calls, and hope the
system surfaces the right documents each time.
This scenario exposes a blind spot in how we
build and evaluate RAG systems. The entire
pipeline, from embedding models to retrieval met-
rics, is optimized forsingle-query relevance: does
*Accepted to the Towards Knowledgeable Foundation
Models (KnowFM) Workshop @ ACL 2026.User’s Session: “Setting up my new Wix site”
Needs: Domain setup + Template design + Payment config
Semantic Embedding Space
Domain
DNSSSL
TemplateFavicon
Paymentqtop-kradius
farfar
Vanilla:Domain, DNS, SSL
Coverage = 3/6 =41%Ours:Domain, DNS, SSL,
Template, Payment, Favicon
Coverage = 6/6 =58%+cluster expansion
Figure 1:The session coverage gap.A user setting
up a website needs articles about domains, templates,
and payments, but these reside in distant regions of
the semantic embedding space. Vanilla RAG retrieves
only documents within the top- ksimilarity radius of the
query (dashed circle), missing semantically distant but
session-relevant articles. Our cluster expansion recovers
these, raising session coverage from 41% to 58%.
the top- kset contain the one document that an-
swers this one question? But enterprise users do
not arrive with isolated questions. They arrive
withsessions: coherent episodes of related infor-
mation needs that span multiple documents across
the knowledge base.
The core problem is thatsemantic similarity
and user information need are different signals.
Documents that a user needs together during a ses-
sion do not necessarily look alike in embedding
space. Domain configuration articles and template
design articles serve the same user journey but
reside in distant semantic neighborhoods. Stan-
dard RAG, which retrieves by embedding similarity
alone, cannot bridge this gap. It finds documents
thatlook likethe query but misses documents the
useralso needs.
We formalize this intuition by introducing
session-level evaluation metricsfor RAG (Sec-arXiv:2606.31156v1  [cs.IR]  30 Jun 2026

tion 3). Single-query session coverage measures
what fraction of a user’s full information need is
satisfied by one retrieval call. On WixQA (Co-
hen et al., 2025), we find that vanilla RAG covers
only 41% of the session at k=8, meaning the user
must issue multiple follow-up queries to access the
remaining knowledge.
To close this gap, we proposeco-occurrence-
aware KB reorganization(Section 4). The idea is
simple: documents that users need together should
be retrievable together. We learn a co-occurrence
embedding space using Word2Vec (Mikolov et al.,
2013) trained on document navigation sequences,
cluster the KB in this space, and expand retrieval
candidates through cluster neighborhoods at query
time. This reorganization happens once, offline,
and the resulting cluster structure benefits any
downstream embedding model. While we primar-
ily use synthetic co-occurrence sequences in this
work, we note that the ground-truth article groups
from WixQA’s expert-labeled queries provide a real
co-occurrence signal, and our method upsamples
these 10 ×during training to anchor the learned
representations in genuine user needs.
This approach has three appealing properties.
First, it ispre-retrieval: the knowledge base is reor-
ganized once offline, contrasting with post-retrieval
methods like EDC2-RAG (Li et al., 2025) that clus-
ter after retrieval. Second, it isencoder-agnostic:
because clusters are learned from co-occurrence
patterns rather than from a specific embedding
model, the same cluster structure improves retrieval
across different encoders. Third, it providesKB
compression: the cluster structure reduces the ef-
fective knowledge base to 20% of its original size
while improving coverage.
Our contributions are: (1) we introduce session-
level evaluation metrics for RAG that capture multi-
document information needs; (2) we propose co-
occurrence-aware KB reorganization with cluster-
expanded hybrid retrieval; and (3) we demonstrate
consistent gains across four encoders, six domains,
and four complexity levels on an enterprise bench-
mark.
2 Related Work
Query-Side RAG Optimization.A large body
of work optimizes RAG retrieval at query time.
Multi-query approaches rewrite the user’s query
into multiple variants to improve recall. Adap-
tive retrieval methods like CAR (Xu et al., 2025)dynamically adjust how many documents to re-
trieve based on query complexity. HyDE (Gao
et al., 2023) generates hypothetical answers to im-
prove query embeddings. Hybrid retrieval com-
bines dense and sparse signals (Ma et al., 2023).
All of these optimizewhich queryis sent to the re-
triever. Our work is complementary: we optimize
how the knowledge base is organizedbefore any
query arrives. The two approaches can be com-
bined; for instance, multi-query retrieval could be
applied on top of our reorganized KB, with each
rewritten query benefiting from cluster expansion.
We focus our baselines on the retrieval layer specif-
ically because our contribution is pre-retrieval KB
reorganization, not query rewriting.
Post-Retrieval Document Compression.EDC2-
RAG (Li et al., 2025) clusters retrieved documents
at query time to remove noise and redundancy be-
fore passing context to the LLM, and was accepted
at EMNLP 2025 Findings. CRAG (Akesson and
Santos, 2024) similarly clusters and summarizes
retrieved documents to fit context windows. RAP-
TOR (Sarthi et al., 2024) builds hierarchical sum-
maries for tree-based retrieval. These methods op-
erateafterretrieval to compress what was already
fetched. Our method operatesbeforeretrieval to
ensure the right documents are fetched in the first
place. The two approaches compose naturally: one
can apply post-retrieval compression on top of our
pre-retrieval reorganization.
Embedding Limitations.Weller et al. (2026)
establish theoretical limits on single-vector dense
embeddings, showing that fixed-dimensional rep-
resentations cannot realize all possible retrieval
configurations. This underscores the need for com-
plementary signals beyond semantic similarity. Our
co-occurrence embeddings provide exactly such a
signal, capturing functional relatedness between
documents that may be semantically distant.
Collaborative Filtering in Information Retrieval.
The use of co-occurrence patterns to learn item rep-
resentations has a long history in recommendation
systems. Item2Vec (Barkan and Koenigstein, 2017)
applies Word2Vec to item co-purchase sequences.
Our work applies the same principle to document
co-access patterns in knowledge bases, bridging
collaborative filtering and information retrieval for
RAG.
Session-Level Evaluation.Existing RAG evalu-
ation focuses exclusively on single-query metrics
2

such as precision@ k, recall@ k, MRR, and faith-
fulness (Es et al., 2024). While some observabil-
ity platforms support session-level monitoring for
multi-turn conversations, no prior work evaluates
retrievalcoverageat the session level or measures
how many retrieval calls are needed to satisfy a
user’s full information need. We introduce these
metrics and argue that they better reflect enterprise
RAG performance, where user satisfaction depends
on resolving an entire issue rather than answering
a single question.
3 Problem Formulation
LetD={d 1, . . . , d N}be a knowledge base of
Ndocuments. Asession S ⊆ D is a set of docu-
ments that collectively address a user’s information-
seeking episode. Given a query qassociated with
session S, a retriever returns an ordered set Rq=
{r1, . . . , r k}ofkdocuments.
3.1 Session-Level Metrics
We propose two metrics that evaluate retrieval at
the session level rather than the query level.
Single-Query Session Coverage (Cov).The
fraction of the session’s documents retrieved in
one call:
Cov(q,S) =|Rq∩ S|
|S|(1)
Standard recall@ kis the special case where |S|=
1. When |S|>1 , Coverage captures multi-
document information needs that recall@ kcannot
distinguish. A Coverage of 0.6 means 60% of the
user’s total information need is satisfied without
any follow-up queries.
Calls to τ-Coverage ( Cτ).The minimum num-
ber of retrieval calls such that the union of all re-
trieved sets covers fractionτof the session:
Cτ(S) = min
m:|Sm
i=1Rqi∩ S|
|S|≥τ
(2)
where q1, . . . , q mare sequential queries from the
session. Lower Cτindicates a more efficient re-
trieval system. In production, each additional re-
trieval call adds latency, API cost, and user friction.
4 Method
Our approach consists of an offline KB reorgani-
zation phase and an online hybrid retrieval phase
(Figure 2).4.1 Offline: Co-occurrence-Aware Clustering
Step 1: Co-occurrence Sequence Construction.
We construct sequences of document IDs that repre-
sent documents a user would need together. These
sequences are drawn from three complementary
sources:
1.Ground-truth co-occurrence:For each la-
beled query with multiple relevant article IDs,
the article set forms a natural co-occurrence
group. These are the highest-quality signal,
as they reflect real user needs annotated by
domain experts.
2.Embedding-neighborhood walks:Starting
from a random seed document, we perform
random walks through the document similar-
ity graph for 3 to 5 steps, with a 40% ran-
dom jump probability. The high jump rate
ensures that the resulting sequences capture
cross-neighborhood relationships rather than
simply recapitulating the embedding struc-
ture.
3.QA-driven sequences:For each synthetically
generated question-answer pair, we identify
the source document and find its embedding
neighbors. This simulates a user’s browsing
trajectory starting from a specific question.
All sequences are augmented 3×through rever-
sal and contiguous subsequence extraction (length
2 to 4), following the augmentation strategy
of Barkan and Koenigstein (2017).
Step 2: Co-occurrence Embedding.We treat
document IDs as tokens and co-occurrence se-
quences as sentences, training a Word2Vec CBOW
model (Mikolov et al., 2013):
zd=W2V CBOW (d;w=2, n s=10)z d∈R100
(3)
The resulting embedding zdcapturesfunctional
relatedness: documents that co-occur in user ses-
sions are close in z-space, regardless of their
semantic similarity. This is analogous to how
Item2Vec (Barkan and Koenigstein, 2017) learns
product representations from co-purchase patterns,
except that our “items” are knowledge articles and
our “purchases” are co-access events.
Step 3: Hierarchical Clustering.We apply ag-
glomerative clustering with cosine distance and
average linkage to the co-occurrence embeddings,
3

(a) Offline: KB Reorganization
Semantic Space
Same-session docs are
scattered across spaceWord2Vec
on co-occCo-occurrence Space
Same-session docs are
clustered together(b) Online: Hybrid Retrieval
Queryq
Encode
(any model)
Top-k v=3
Vanilla Retrieval3 docsCluster Expansion
+12
Re-rank
Top-k=8Results
Coverage: 58%cluster structure
Figure 2:System overview. (a) Offline:Documents from the same user session (same color) are scattered in
semantic embedding space but cluster together after Word2Vec training on co-occurrence sequences.(b) Online:
Given a query, we retrieve the top-3 by similarity (gray dots), expand through their cluster neighborhoods (blue dots,
+12 candidates), and re-rank the union to return the top-8 with 58% session coverage.
targeting approximately 20% of the original KB
size. Formally, we compute C={C 1, . . . , C M}
where M=⌈N/5⌉ via agglomerative clustering
over{zd}d∈D. Each cluster Cj⊂ D groups func-
tionally related documents that may span different
semantic topics.
4.2 Online: Hybrid Retrieval with Cluster
Expansion
Given a query qwith embedding eqproduced by
any encoder and document embeddings{e d}d∈D:
Step 1: Vanilla Retrieval.Retrieve the top- kv
documents by cosine similarity:
Vq=top-k v
d∈Dcos(e q,ed)(4)
Step 2: Cluster Expansion.For each retrieved
document, gather all documents from its cluster as
expansion candidates:
Eq=[
d∈V qC(d)\ V q (5)
where C(d) denotes the cluster containing docu-
ment d. With an average cluster size of 5.0, retriev-
ingkv=3documents expands the candidate pool
to approximately 3×5 = 15 candidates before
deduplication.
Step 3: Re-rank and Return.Rank the ex-
panded candidate set by direct query-document
similarity and return the top-k:
Rq=top-k
d∈V q∪Eqcos(e q,ed)(6)The vanilla results Vqare guaranteed to appear
in the candidate pool, preserving first-hit preci-
sion. The expansion adds co-occurring documents
that would otherwise be missed because they have
low semantic similarity to q. We set kv=3and
k=8 throughout our experiments unless otherwise
noted.
4.3 Implementation Details
For QA pair generation, we use GPT-4.1-nano to
produce 10 question-answer pairs per document,
yielding 31,500 pairs across the WixQA KB (cost:
approximately $2 USD). Co-occurrence sequence
generation produces 16,594 augmented sequences.
Word2Vec training completes in under 30 seconds
on a single CPU core. Agglomerative clustering
over 6,221 documents takes approximately 2 min-
utes. The entire offline pipeline runs in under 20
minutes, making it practical for periodic KB reor-
ganization.
At inference time, the hybrid retrieval adds neg-
ligible latency: cluster lookup is O(1) via a pre-
computed dictionary, and the expansion step adds
at most kv×¯|C|candidates (approximately 15) to
the re-ranking pool. The total retrieval overhead
is dominated by the original embedding similarity
computation, not the expansion.
5 Experimental Setup
5.1 Datasets
WixQA.Our primary evaluation uses
WixQA (Cohen et al., 2025), an enterprise
4

RAG benchmark containing 6,221 customer sup-
port articles from the Wix Help Center knowledge
base. The dataset includes 400 expert-labeled
queries: 200 expert-written queries with multi-step
answers and 200 simulated queries validated by
domain experts. Each query is annotated with
ground-truth article IDs from the KB, enabling
evaluation with real relevance labels.
E-Commerce Support.For cross-domain valida-
tion, we evaluate on an e-commerce customer sup-
port corpus containing 1,000 conversations across
10 product categories and multiple issue types. Co-
occurrence sessions are defined by shared issue-
category and product-category metadata, providing
naturally defined user sessions without synthetic
construction.
5.2 Session Construction
Sessions correspond to co-occurrence clusters,
where each cluster represents a group of articles
that collectively serve a user’s information-seeking
episode. We construct 200 evaluation sessions from
the WixQA clusters, with a mean size of 7.9 doc-
uments (range: 3 to 70). For each evaluation trial,
we randomly sample one entry query from the ses-
sion and measure retrieval coverage over the full
session.
5.3 Baselines
We compare our hybrid retrieval against two base-
lines: (1)Vanilla RAG, which retrieves the top-
kdocuments by cosine similarity between query
and document embeddings; and (2)Cross-encoder
re-ranking, which retrieves the top-30 by embed-
ding similarity and then re-ranks using ms-marco-
MiniLM-L-6-v2 cross-encoder.
5.4 Embedding Models
To demonstrate encoder-agnostic gains, we eval-
uate across four embedding models spanning
different architectures, training objectives, and
model sizes: all-MiniLM-L6-v2 (22M parame-
ters, general-purpose), bge-base-en-v1.5 (109M,
retrieval-optimized), gte-base (109M, text cluster-
ing), and e5-base-v2 (109M, weakly supervised).
5.5 Statistical Methodology
All reported confidence intervals are computed via
bootstrap resampling with 1,000 samples at the
95% level. We seed all random operations (session
entry point selection, sequence generation) with a
fixed seed for reproducibility.5 810 15 202030405060708090
+20%
Retrieval Budget (k)Session Coverage (%)Hybrid (ours)
Vanilla RAG
Figure 3: Session coverage vs. retrieval budget. The
shaded area represents the coverage gap that cluster
expansion closes. Notably, the gapwidenswithk.
kVanilla Hybrid∆Rel.∆
5 36.9% 45.9% +9.0% +24%
8 41.4%59.1%+17.7% +43%
10 43.4% 63.5% +20.1% +46%
15 49.7% 76.8% +27.1% +55%
20 56.2% 83.1% +26.8% +48%
Table 1: Session coverage at varying retrieval budgets.
Bold indicates the primary operating point (k=8).
6 Results
6.1 Session Coverage vs. Retrieval Budget
Table 1 and Figure 3 report session coverage as the
retrieval budget varies. The coverage gap between
vanilla and hybridwidenswith k: from +9.0% at
k=5 to +27.1% at k=15 . This scaling property is
important because it means that as context windows
grow and retrieval budgets increase, the relative
value of cluster expansion grows rather than sat-
urates. At k=20 , hybrid retrieval achieves 83.1%
session coverage, compared to 56.2% for vanilla.
6.2 Retrieval Efficiency
Table 2 and Figure 4 quantify the practical ef-
ficiency gain. To reach 70% session coverage,
vanilla RAG requires 4.0 calls on average while
hybrid retrieval requires only 2.6, a 34% reduc-
tion. At the 80% threshold, the savings increase
to 1.7 fewer calls per session. In production sys-
tems that handle millions of sessions, this translates
directly to reduced latency, lower API costs, and
fewer user interactions needed to resolve each sup-
port episode.
5

50% 60% 70% 80% 90%01234567
-26%-29%-34%-36%-35%
Coverage Target (τ)Avg Retrieval CallsVanilla RAG
Hybrid (ours)
Figure 4: Retrieval calls to reach session coverage
thresholds. Hybrid retrieval requires 26 to 36% fewer
calls across all targets.
τVanilla Hybrid Reduction Saved
50% 2.3 1.7 26% 0.6
60% 2.8 2.0 29% 0.8
70% 4.02.6 34%1.4
80% 4.7 3.0 36% 1.7
90% 6.7 4.4 35% 2.4
Table 2: Retrieval calls needed to reach coverage thresh-
oldsτ.
6.3 Effect of Session Complexity
Table 3 stratifies results by the number of docu-
ments in the session. The method provides sub-
stantial gains at every complexity level, with all
95% bootstrap confidence intervals excluding zero.
The largest absolute gain appears on complex ses-
sions (8 to 15 documents): +17.6% with a CI lower
bound of +11.8%. For simple sessions (3 to 4 doc-
uments), coverage jumps from 52.0% to 70.8%.
The diminishing absolute gain on very complex
sessions (16+ documents) is expected: a retrieval
budget of k=8 cannot cover 16 or more documents
in a single call regardless of the retrieval strategy.
6.4 Encoder Robustness
Table 4 demonstrates that the cluster structure,
which is learned once from co-occurrence patterns,
improves retrieval across all four embedding mod-
els. The deltas range from +17.1% to +20.6%
absolute, with every bootstrap CI excluding zero.
This confirms that co-occurrence captures informa-
tion that is orthogonal to any specific embedding
model’s representation, and that the offline cluster
structure is a genuinely encoder-agnostic resource.TypeNVan. Hyb.∆95% CI
Simple (3–4) 92 52.070.8+18.8 [13.1, 24.3]
Medium (5–7) 48 41.5 58.1 +16.7 [10.7, 22.9]
Complex (8–15) 41 28.3 45.9 +17.6 [11.8, 23.1]
V . Cmplx (16+) 19 15.2 23.8 +8.6 [5.5, 11.9]
Table 3: Session coverage (%) by complexity with 95%
bootstrap CIs. All intervals exclude zero.
Encoder Van. Hyb.∆95% CI
MiniLM-L6-v2 41.1 58.2 +17.1 [14.0, 20.2]
bge-base-v1.5 39.8 57.1 +17.2 [14.2, 20.7]
gte-base 40.4 60.8 +20.4 [17.0, 23.8]
e5-base-v2 39.860.4+20.6 [17.1, 23.9]
Table 4: Session coverage (%) across encoders ( k=8)
with 95% bootstrap CIs. All intervals exclude zero.
Clusters are learned once and benefit all encoders.
6.5 Domain Generalization
We apply Latent Dirichlet Allocation to identify
six functional domains within WixQA (Table 5).
The hybrid method improves coverage inall six
domains, with gains ranging from +12.7% (Apps
& Email) to +21.5% (Editor & Studio). This con-
sistency across topically diverse domains confirms
that the method is not specific to any particular
content type or user behavior pattern within the
KB.
6.6 Cross-Dataset Validation
To validate beyond a single benchmark, we evaluate
on an e-commerce customer support corpus with
1,000 documents and 94 naturally defined sessions.
Session coverage improves from 32.2% to 43.3%
(+11.1% absolute, +34% relative), confirming that
the method generalizes to a different domain, doc-
ument structure, and session definition.
7 Analysis
7.1 Semantic Similarity vs. Co-occurrence
We compute the Pearson correlation between se-
mantic similarity (cosine in MiniLM embedding
space) and co-occurrence strength for 10,000 sam-
pled document pairs (Figure 5). The correlation is
weak ( r≈0.2 ), confirming that documents users
need together arenotnecessarily the documents
that look alike in embedding space. This weak
correlation is precisely what makes co-occurrence
clustering valuable: it provides a complementary
signal that semantic embeddings structurally can-
not capture, and cluster expansion surfaces the “low
6

DomainNVan. Hyb.∆
Bookings & Payments 37 34.2% 50.8% +16.6%
Apps & Email 28 40.0% 52.6% +12.7%
Editor & Studio 58 46.3%67.8%+21.5%
Plans & Pricing 13 52.8% 71.1% +18.3%
Media & Domains 26 50.0% 63.0% +13.0%
Stores & Blog/CMS 38 37.6% 51.9% +14.3%
Table 5: Session coverage across six WixQA functional
domains. Gains are positive in all six domains.
0 0.2 0.4 0.6 0.8 102468
Vanilla RAG
misses these
r= 0.20
Semantic Similarity (cosine)Co-occurrence StrengthNon-co-occurring
High-sim high-cooc
Low-sim high-cooc
Figure 5: Semantic similarity vs. co-occurrence strength
(schematic from 10K sampled pairs). Thered triangle
region (low similarity, high co-occurrence) contains doc-
ument pairs that vanilla RAG cannot surface together.
Weak Pearson correlation ( r=0.20 ) confirms the signals
are complementary.
similarity, high co-occurrence” document pairs that
vanilla RAG systematically misses.
7.2 Cluster Quality Analysis
Table 6 reports detailed clustering statistics. The
1,244 clusters have a mean size of 5.0 documents
with a long-tailed distribution: most clusters con-
tain 3 to 5 documents (corresponding to simple
user sessions), while a small number of large clus-
ters (up to 70 documents) capture broad topics like
“account management” that touch many articles.
The silhouette score of 0.205 indicates moder-
ate cluster cohesion. This moderate (rather than
high) score is expected and actually desirable: co-
occurrence clusters intentionally group semanti-
callydiversedocuments, so high cohesion in the
semantic embedding space would indicate that the
clusters are simply recapitulating semantic similar-
ity rather than adding new information.
The within-cluster co-occurrence rate of 25.9%
means that approximately one in four co-occurring
document pairs land in the same cluster. This is
substantially above the random baseline of 1/M≈
0.08% , confirming meaningful structure. The rateStatistic Value
Original KB size 6,221 documents
Number of clustersM1,244
Compression ratio 20.0%
Mean cluster size 5.0
Median cluster size 4
Max cluster size 70
Silhouette score 0.205
Within-cluster co-occ rate 25.9%
Word2Vec vocabulary coverage 98.8%
Augmented co-occ sequences 16,594
Table 6: Clustering statistics on WixQA.
# Document Title
1 Connecting a Domain You Already Own
2 Transferring a Domain to Wix
3 Getting a Free Domain with Premium Plans
4 Changing Your Site Template
5 Customizing Your Site’s Favicon
Table 7: Example cluster containing semantically di-
verse but functionally related documents. Documents
1–3 cover domain management; documents 4–5 cover
site design. Users setting up a new website commonly
need all five.
is not higher because the co-occurrence sequences
draw from three signal sources with different prop-
erties, and the clustering finds a compromise that
respects all three.
7.3 Qualitative Example
To illustrate the method’s behavior concretely, con-
sider a representative cluster:
Documents 1 through 3 form a tight semantic
group (domain management), while documents 4
and 5 are semantically distant (site design). Yet
all five are commonly needed by users setting up a
new website. When a user asks about connecting
their domain, vanilla RAG retrieves documents 1
through 3 but misses 4 and 5 entirely. Cluster ex-
pansion adds documents 4 and 5 to the candidate
pool, and the re-ranker surfaces them if they match
the query context.
7.4 Precision Trade-off
Cluster expansion trades marginal first-hit preci-
sion for substantially broader coverage. At k=8,
Hits@K decreases slightly from 96.0% to 93.0%.
This is the correct trade-off for session-oriented set-
tings: covering 58% of the user’s full information
need is more valuable than a 3% improvement in
whether the single best document appears in the
top result. In practice, this trade-off can be tuned
7

by adjusting kv: increasing kvfrom 3 to 5 pre-
serves more vanilla precision at the cost of fewer
expansion slots.
7.5 Boundary Condition: When the Method
Does Not Help
On HotpotQA (Yang et al., 2018), where each ques-
tion has an isolated paragraph set with no cross-
query document reuse, the method shows no gain
(∆ =−1.4%). This negative result is informative:
it confirms that the improvement requires apersis-
tentknowledge base where documents are reused
across multiple user sessions. In such settings, co-
occurrence patterns form naturally from repeated
access. In benchmarks where each question con-
structs an ad-hoc document set, no co-occurrence
signal can emerge. This boundary condition con-
firms that gains are driven by learned co-occurrence
patterns rather than by an artifact of the expansion
mechanism itself.
8 Conclusion
We propose that RAG evaluation should shift from
single-query recall to session-level coverage, and
demonstrate a simple method to close the resulting
coverage gap: reorganize the KB offline using co-
occurrence-aware clustering, then expand retrieval
candidates through cluster neighborhoods at query
time. A single retrieval call over a reorganized KB
covers 58% of a user’s session-level information
need (compared to 41% for standard RAG) while
requiring 34% fewer calls to reach practical cov-
erage thresholds. The method is encoder-agnostic
(+17 to 21% across four models), domain-general
(gains in 6 out of 6 domains), and provides 80%
KB compression as a side benefit.
For future work, we plan to incorporate real user
navigation logs to replace synthetic co-occurrence
sequences and measure how the quality of the co-
occurrence signal affects downstream coverage
gains. We also plan to evaluate the impact on
end-to-end generation quality using LLM-as-judge
metrics, as higher retrieval coverage does not au-
tomatically translate to better generated answers if
the LLM cannot effectively use the broader con-
text. Finally, we intend to explore adaptive cluster
expansion, where the number of expanded candi-
dates scales with query ambiguity, and to investi-
gate combining our pre-retrieval KB reorganiza-
tion with post-retrieval compression methods like
EDC2-RAG (Li et al., 2025) for end-to-end session-level RAG optimization.
Limitations
Our co-occurrence sequences are constructed from
embedding neighborhoods and synthetic QA pairs
rather than real user interaction logs. While the
ground-truth article groups from WixQA provide a
real co-occurrence signal and the HotpotQA bound-
ary experiment confirms that persistent document
reuse is required, real enterprise usage data would
likely produce stronger co-occurrence patterns and
larger gains. We were unable to use real navigation
logs due to the absence of publicly available en-
terprise session data, a common constraint in this
domain.
Session-level evaluation uses cluster-derived ses-
sions as a proxy for real user sessions; evalu-
ation on logged multi-turn conversations would
strengthen the claims.
We evaluate retrieval coverage but not down-
stream generation quality. Higher coverage intro-
duces more diverse context into the LLM’s input,
which may trigger the “lost in the middle” phe-
nomenon (Liu et al., 2024) where models under-
utilize information in the middle of long contexts.
Whether the 58% session coverage translates to
proportionally better generated answers remains an
open question that we leave to future work.
The 40% random jump probability in embedding
walks was set heuristically; a systematic search
over this hyperparameter could improve results.
Finally, while we evaluate across four encoders, six
domains, and two datasets, additional enterprise
KB evaluations with real user session logs would
further establish generalizability.
Ethics Statement
This work uses publicly available datasets (WixQA
under MIT license) and does not involve human
subjects, private user data, or personally identifi-
able information. The synthetic QA pairs are gen-
erated from public knowledge base articles. Our
method reorganizes existing knowledge bases and
does not generate new content, so it does not intro-
duce additional hallucination risk beyond what is
inherent in the underlying RAG system. We note
that co-occurrence patterns learned from real user
logs, if used in future work, would require appro-
priate anonymization and privacy protections.
8

References
Simon Akesson and Frances A. Santos. 2024. Clustered
retrieved augmented generation (CRAG).Preprint,
arXiv:2406.00029.
Oren Barkan and Noam Koenigstein. 2017. Item2vec:
Neural item embedding for collaborative filtering.
Preprint, arXiv:1603.04259.
Dvir Cohen, Lin Burg, Sviatoslav Pykhnivskyi, Hagit
Gur, Stanislav Kovynov, Olga Atzmon, and Gilad
Barkan. 2025. WixQA: A multi-dataset bench-
mark for enterprise retrieval-augmented generation.
Preprint, arXiv:2505.08643.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAS: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158, St. Julians,
Malta. Association for Computational Linguistics.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777,
Toronto, Canada. Association for Computational Lin-
guistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2021.
Retrieval-augmented generation for knowledge-
intensive NLP tasks.Preprint, arXiv:2005.11401.
Weitao Li, Xiangyu Zhang, Kaiming Liu, Xuanyu Lei,
Weizhi Ma, and Yang Liu. 2025. Efficient dynamic
clustering-based document compression for retrieval-
augmented-generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 9833–9849, Suzhou, China. Association for
Computational Linguistics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2023. Fine-tuning LLaMA for multi-
stage text retrieval.Preprint, arXiv:2310.08319.
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey
Dean. 2013. Efficient estimation of word representa-
tions in vector space.Preprint, arXiv:1301.3781.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Man-
ning. 2024. RAPTOR: Recursive abstractive pro-
cessing for tree-organized retrieval.Preprint,
arXiv:2401.18059.Orion Weller, Michael Boratko, Iftekhar Naim, and
Jinhyuk Lee. 2026. On the theoretical limi-
tations of embedding-based retrieval.Preprint,
arXiv:2508.21038.
Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha
Mahadevan, and Bhaskar Krishnamachari. 2025.
Cluster-based adaptive retrieval: Dynamic con-
text selection for RAG applications.Preprint,
arXiv:2511.14769.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
A Single-Query Retrieval Results
For completeness, we report standard single-query
retrieval metrics on the 400 WixQA expert queries
(Table 8). We evaluate a boost-based scoring vari-
ant which adds a cluster-relevance signal to direct
document similarity with weight α=0.2 . This pro-
vides modest but consistent gains over vanilla re-
trieval on single-query metrics. The gains are small
because WixQA queries are predominantly single-
aspect; the method’s primary value is at the session
level (Section 6).
Method W. Recall Hits@K MRR
Vanilla 52.8% 57.5% 0.359
Reranker 53.1% 58.0% 0.334
Boost (α=0.2) 55.2% 59.8% 0.363
Table 8: Single-query metrics on 400 WixQA expert
queries.
B Hyperparameter Sensitivity
Vanilla slots kv.Table 9 shows session coverage
askvvaries with k=8 fixed. Lower kvallocates
more slots to cluster expansion, increasing cover-
age at the cost of first-hit precision. We select kv=3
as it provides the best balance.
Boost weight α.For the boost-based variant used
in single-query evaluation, Table 10 shows the
effect of α. The value α=0.2 provides the best
weighted recall; higher values degrade performance
as the cluster signal overwhelms direct similarity.
9

kvCoverage Hits@K∆Cov
1 61.2% 87.5% +20.1%
2 59.8% 90.0% +18.7%
3 58.2% 93.0% +17.1%
5 50.2% 95.5% +9.1%
7 43.5% 96.0% +2.4%
Table 9: Effect of vanilla slots kvon session coverage
and Hits@K ( k=8).kv=3balances coverage with pre-
cision.
αW. Recall Hits@K∆WR
0.1 54.1% 58.8% +1.2%
0.255.2%59.8% +2.4%
0.3 54.2% 58.8% +1.3%
0.4 52.2% 56.5%−0.7%
0.5 50.8% 55.0%−2.0%
Table 10: Effect of boost weight αon single-query
metrics.
C Cross-Dataset Summary
Table 11 summarizes results across all evaluation
settings, including the HotpotQA boundary experi-
ment.
Dataset Domain Docs Van. Hyb.∆
WixQA Web Support 6,221 41% 58% +17%
E-Commerce E-Commerce 1,000 32% 43% +11%
HotpotQA Open-domain 5,842 35.8% 34.4%−1.4%
Table 11: Cross-dataset session coverage ( k=8). Gains
are consistent on persistent KBs and absent on ad-hoc
paragraph sets (HotpotQA), confirming the method re-
quires cross-session document reuse.
D Reproducibility Details
The offline pipeline (QA generation, sequence
construction, Word2Vec training, and cluster-
ing) runs in under 20 minutes on a single CPU
core, excluding API time for QA generation
(approximately 6 minutes with 30 concurrent
threads). All embedding models are loaded via
thesentence-transformers library. WixQA is
available under MIT license at Wix/WixQA on Hug-
gingFace.Parameter Value
Word2Vec architecture CBOW
Embedding dimension 100
Window size 2
Negative samples 10
Training epochs 30
Random jump probability 0.40
Clustering method Agglomerative (cosine, avg)
Target compression 20% (M=⌈N/5⌉)
Vanilla slotsk v 3
Retrieval budgetk8
QA generation model GPT-4.1-nano
QA pairs per document 10
Bootstrap samples 1,000
Random seed 42
Table 12: Full hyperparameter configuration.
10