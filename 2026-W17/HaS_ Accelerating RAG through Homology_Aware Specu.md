# HaS: Accelerating RAG through Homology-Aware Speculative Retrieval

**Authors**: Peng Peng, Weiwei Lin, Wentai Wu, Xinyang Wang, Yongheng Liu

**Published**: 2026-04-22 11:15:54

**PDF URL**: [https://arxiv.org/pdf/2604.20452v1](https://arxiv.org/pdf/2604.20452v1)

## Abstract
Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) at inference by retrieving external documents as context. However, retrieval becomes increasingly time-consuming as the knowledge databases grow in size. Existing acceleration strategies either compromise accuracy through approximate retrieval, or achieve marginal gains by reusing results of strictly identical queries. We propose HaS, a homology-aware speculative retrieval framework that performs low-latency speculative retrieval over restricted scopes to obtain candidate documents, followed by validating whether they contain the required knowledge. The validation, grounded in the homology relation between queries, is formulated as a homologous query re-identification task: once a previously observed query is identified as a homologous re-encounter of the incoming query, the draft is deemed acceptable, allowing the system to bypass slow full-database retrieval. Benefiting from the prevalence of homologous queries under real-world popularity patterns, HaS achieves substantial efficiency gains. Extensive experiments demonstrate that HaS reduces retrieval latency by 23.74% and 36.99% across datasets with only a 1-2% marginal accuracy drop. As a plug-and-play solution, HaS also significantly accelerates complex multi-hop queries in modern agentic RAG pipelines. Source code is available at: https://github.com/ErrEqualsNil/HaS.

## Full Text


<!-- PDF content starts -->

HaS: Accelerating RAG through Homology-Aware
Speculative Retrieval
1stPeng Peng
South China University of Technology
Guangzhou, China
Pengcheng Laboratory
Shenzhen, China
pengp@pcl.ac.cn2ndWeiwei Lin
South China University of Technology
Guangzhou, China
Pengcheng Laboratory
Shenzhen, China
linww@scut.edu.cn3rdWentai Wu
Jinan University
Guangzhou, China
wentaiwu@jnu.edu.cn
4thXinyang Wang
Beijing Forestry University
Beijing, China
wxyyuppie@bjfu.edu.cn5thYongheng Liu
Pengcheng Laboratory
Shenzhen, China
yongheng.liu@pcl.ac.cn
Abstract—Retrieval-Augmented Generation (RAG) expands
the knowledge boundary of large language models (LLMs) at
inference by retrieving external documents as context. However,
retrieval becomes increasingly time-consuming as the knowledge
databases grow in size. Existing acceleration strategies either
compromise accuracy through approximate retrieval, or achieve
marginal gains by reusing results of strictly identical queries. We
propose HaS, a homology-aware speculative retrieval framework
that performs low-latency speculative retrieval over restricted
scopes to obtain candidate documents, followed by validating
whether they contain the required knowledge. The validation,
grounded in the homology relation between queries, is formulated
as a homologous query re-identification task: once a previously
observed query is identified as a homologous re-encounter of
the incoming query, the draft is deemed acceptable, allowing
the system to bypass slow full-database retrieval. Benefiting
from the prevalence of homologous queries under real-world
popularity patterns, HaS achieves substantial efficiency gains.
Extensive experiments demonstrate that HaS reduces retrieval
latency by 23.74% and 36.99% across datasets with only a
1-2% marginal accuracy drop. As a plug-and-play solution,
HaS also significantly accelerates complex multi-hop queries in
modern agentic RAG pipelines. Source code is available at:
https://github.com/ErrEqualsNil/HaS.
Index Terms—Retrieval-Augmented Generation, Speculative
Retrieval, Homology Similarity
I. INTRODUCTION
RAG well complements the insufficient knowledge coverage
of LLMs [1]. Instead of relying on parametric memory, RAG
retrieves factual documents from external databases to aug-
ment LLM responses through input prompts. Proved effective
in reducing hallucinations, RAG has already been adopted
Corresponding Author: Weiwei Lin.
This work was supported by the Shandong Provincial Natural Science
Foundation (ZR2024LZH012), the Guangxi Key Research and Development
Project (2024AB02018), the Major Key Project of PCL, China (PCL2025A11
and PCL2025A08), the New Generation Artificial Intelligence National Sci-
ence and Technology Major Project (2025ZD0123605), the National Natural
Science Foundation of China (62402198), and the Basic and Applied Basic
Research Foundation of Guangzhou (2025A04J2212)widely in the industry for both general-purpose and domain-
specific chat applications.
Fig. 1: Retrieval is much slower than generation, as revealed
by the comparison between retrieval latency and the time-to-
first-token for a bare LLM.
Despite extensive efforts to reduce LLM prefilling latency
caused by longer RAG prompts [2]–[4], recent studies have
highlighted that the retrieval, implemented as an exact near-
est neighbor search (ENNS), has emerged as a dominant
latency bottleneck, significantly prolonging the time-to-first-
token (TTFT) and degrading user experience [5]. As global
knowledge repositories continue to scale, this challenge be-
comes increasingly pronounced. As illustrated in Fig. 1, our
profiling reveals that, even on a small knowledge database with
6 million entries only, regular retrieval average incurs 0.62
seconds latency1. While it is far from the reality scale, the
observed latency exceeds the TTFT of bare LLMs by a large
margin2. The rise of agentic RAG pipelines further intensifies
this challenge, as iterative retrieval over decomposed sub-
queries amplifies the overhead [6]–[9].
With respect to retrieval acceleration, existing approaches
generally fall into two paradigms. Approximate nearest neigh-
borhood search (ANNS), such as IVF and ScaNN [10], are
1Tested withFaiss-IndexFlaton our workstation.
2Data source: https://github.com/tenstorrent/tt-metalarXiv:2604.20452v1  [cs.IR]  22 Apr 2026

well-established in industry, as shown in Fig.2a. They par-
tition the database into buckets and retrieve only from the
closest ones. The reduced retrieval scope accelerates retrieval,
but inevitably compromises accuracy [5]. In contrast, reuse-
based methods [11], [12] aim to reuse retrieval results of
recent queries with strictly identical semantics, as illustrated
in Fig.2b. Due to the diversity of the real world, such a query
infrequently exists, leading to marginal efficiency gains.
Inspired by the speculative decoding in LLM, we propose
HaS, aHomology-awareSpeculative Retrieval method. HaS
is designed as a plug-and-play service that intercepts queries
prior to full-database retrieval. Conceptually, the framework
first executes rapid retrieval over a restricted subset to gen-
erate a ”draft” of candidate documents. It then leverages the
homology relation between queries to validate whether this
draft contains the necessary knowledge to be accepted. High-
latency full-database retrieval is thus bypassed whenever the
draft is accepted.
Specifically, as depicted in Fig.2c, HaS first executestwo-
channel fast retrievalto speculatively obtain a draft. Both
channels maintain a small retrieval scope to ensure low latency.
The cache channel comprises documents retrieved from recent
queries, providing initial knowledge coverage while serving as
probes for the subsequent validation stage. The fuzzy channel
maintains an extremely narrow subset of the database, refining
draft quality and enhancing validation reliability by capturing
knowledge that may be absent from the cache.
The core of HaS is thehomology validationmecha-
nism, which determines whether a draft can be accepted.
Our acceptance principle is that the draft contains the right
document(s) to support a factual response. Direct verification
via query-document semantic evaluation is computationally
prohibitive. Instead, we introduce a symmetric query–query
relation termedhomology, which captures entity alignment
between queries. By leveraging this relation, the validation
is reduced to determining whether a previously cached query
constitutes a homologous re-encounter of the incoming query.
A failure to re-identify provides strong evidence for rejecting
the draft, whereas successful re-identification justifies its ac-
ceptance. Given the high frequency of homologous queries in
real-world popularity patterns [13], HaS has huge potential for
accelerating RAG.
Extensive experiments demonstrate that, when integrated
into a standard RAG pipeline, HaS reduces retrieval latency
by 23.74% and 36.99% across datasets, while incurring only
a marginal 1–2% degradation in response accuracy. It delivers
superior performance compared to competitive methods. HaS
can also be seamlessly integrated into modern agentic RAG
pipelines featuring query decomposition and iterative retrieval,
yielding more substantial end-to-end latency reductions.
The contributions of this work are summarized as follows:
1) We design HaS to accelerate retrieval in RAG. To
conditionally bypass slow full-database retrieval, HaS
speculatively retrieves from two channels to quickly
obtain a draft, followed by a validation procedure.
IrrelevantDocumentRetrievewithin aSubset
Miss
LLM(a) Approximate Nearest Neighbor Search
Recent Queries
FindSame Query
Full-DatabaseRetrievalMismatch
LLM
(b) Reuse-based Retrieval
DraftTwo-ChannelFast Retrieval
HomologyValidation
Re-identifyHomologous Query
AcceptDraft
LLMTwo SmallScopes
Full-DatabaseRetrieval
BypassRetrieve
(c) Homology-aware Speculative Retrieval (HaS)
Fig. 2: Illustration of different approaches for accelerating
retrieval in RAG.
TABLE I: Key Notations
Symbol Description
qCurrent user query
dDocument
E(·)Target entity
A(·),A(·)Target attribute(s)
DA draft as set of documents retrieved
Cc,Cf The cache channel and fuzzy channel
P={(q h,Dh)}Cached queries and retrieved documents
Hmax Cache capacity
s(·)Homology score
JInverted index mapping documents to cached queries
kNumber of documents to retrieve
τThreshold for homology validation
G(d, q)Indicator thatdis a golden document forq
H(·,·), HQ(·,·)Indicators of homology and quasi-homology
V(D, q), ˆV(D, q)Draft acceptance hypothesis and its surrogate
2) The charm of HaS lies in its homology validation. By
re-identifying homologous queries from the cache as
evidence for draft acceptance, HaS achieves both high
efficiency and accuracy.
3) Experiments show that HaS cuts retrieval latency by
23.74% and 36.99% across datasets in the standard RAG
pipeline, outperforming SOTA methods. Being plug-
gable to modern RAG pipelines, this benefit becomes
even more significant.
II. THEOVERALLPIPELINE OFHAS
In this section, we present the problem formulation and
provide an overview of HaS. The design rationale is presented
in the next section. Key notations are summarized in Table I
A. Problem Formulation
In a bare RAG pipeline, the system maintains a large-
scale corpusC. Each documentd∈ Ccontains information

Merge & Re-rank
Draft
Document-Query Inverted Indexq1Aq2Bq2……If Any Homologous Query ExistsAccept DraftAB
Full-DatabaseRetrievalqUpdate Inverted IndexUpdate Cache ChannelCountTwo-Channel Fast RetrievalHomology ValidationOtherwise
DownstreamLLMq
Homologous Scoreq113#q2q223#
Threshold-BasedRe-identification<𝜏≥𝜏Retrieve
CacheChannel
ABC
FuzzyChannel
DEFABDDRetrieval ResultADFADFADF
qUser QueryAGolden DocumentDBNormal DocumentCEFig. 3: Framework of HaS. Given a queryq, the two-channel fast retrieval is first performed. Documents A–F are retrieved, and
the Top-3 (A, B, and D) form the draft. For validation, documents in the draft are indexed to cached queries by the document-
query inverted index. Frequencies of queries hit in the cache are used to compute their homology scores for threshold-based
re-identification. If any quasi-homologous query is re-identified, the draft is accepted; otherwise, a full-database retrieval is
performed, and results are used to update the cache channel and the inverted index.
regarding several attributesA(d)of an entityE(d). Given a
queryq, the retrieval stage recalls a set of relevant documents
ˆD, which are prepended to the LLM’s prompt to generate a
responseL(q, ˆD).
We focus on accelerating the retrieval stage. We assume that
queries are well-organized, targeting a specific attributeA(q)
of a target entityE(q). For complex, multi-hop queries, we
can follow the standard practice of performing query decom-
position prior to retrieval [6], [7] (an use case is provided in
Section IV-E).
For each queryq, HaS first performs retrieval over a
restricted scopeC s⊂ Cto obtain ak-document draftD
with an average latency of ¯ℓs. Since retrieval latency scales
with the database size, we have ¯ℓs≪¯ℓ, where ¯ℓis the
average retrieval latency onC. We define a validation function
R(q,D)∈ {0,1}, whereR(q,D) = 1indicates that the draft is
acceptable and vice versa (the validation mechanism is detailed
in Section III). If acceptable,Dis returned; otherwise, full-
database retrieval is invoked to obtainD full. The final result
set is therefore:
ˆD=(
D, R(q,D) = 1,
Dfull, R(q,D) = 0.(1)
By optimizingC sandR, HaS aims to reduce retrieval
latency while maximizing response accuracy. For a query set
Q, the multi-objective optimization problem is formulated as:
min
Cs,R1
|Q|X
q∈Q
R(q,D) ¯ℓs+ (1−R(q,D))( ¯ℓs+¯ℓ)
,
max
Cs,R1
|Q|X
q∈QAcc(L(q, ˆD)),
whereAcc(·)∈ {0,1}measures whether the response matches
the reference answer.B. Overview of HaS
HaS is designed as a pluggable, lightweight component
before the full-database retrieval, as illustrated in Fig.3. Fol-
lowing modern service practice, such as Content Delivery
Networks (CDNs), it can be deployed proximate to the LLM
to enable low-latency access. Incoming queries can be first
processed by HaS in a local network, while those with draft
rejection are redirected to the full database hosted remotely.
The pseudo-code of HaS is provided in Algorithm 1.
HaS maintains two retrieval channels whose union consti-
tutesC s, namely the cache channel and the fuzzy channel.
The combined retrieval scope of these channels is restricted
for fast retrieval. The cache channel is populated by docu-
ments previously retrieved for historical queries. Formally, let
P={(q h,Dh)|0≤h≤H max}represent the system cache,
where each entry consists of a historical queryq hand its
corresponding full-database retrieval result setD h, up to a
maximum capacityH max. The cache channelC c=SHmax
h=0Dh
is defined as the union of all documents within these cached
results3. The fuzzy channel is implemented by ANNS, but
configured aggressively to retrieve only a very narrow subset.
This allows it to return results rapidly, albeit with reduced
accuracy. The benefits of introducing these two channels are
detailed in Section III-D.
For each queryq, HaS first executes two-
channel fast retrieval. Top-kdocuments from both
channels are recalled, denoted asD candD f, where
Di= arg topk
d∈C isim(g(q), g(d)), i∈ {c, f}, and
g(·)is an semantic encoder. These documents are re-
ranked, and the top-kdocuments are selected as the draft
D= arg topk
d∈D c∪Dfsim(g(q), g(d)).
After the draft is obtained, HaS initiates homology valida-
tion by seeking to re-identify a homologous counterpart in the
cache, who share the same target entity of the current query.
3Without ambiguity, the cachePdenotes the query–documents pairs, while
the cache channelC crefers to the collection of all documents contained in
the cache.

This relationship is quantified by the homology score, defined
as the overlap ratio between the retrieval result sets of the two
queries. Queries whose homology score is higher than a preset
threshold are considered homologous. The formal definition of
homology is given in Section III-A, while the mechanism of
draft validation will be elaborated in Sections III-B and III-C.
To efficiently compute homology scores against all cached
queries, HaS maintains a document-query inverted indexJ:
D → Q, which maps each documentdto a set of cached
queries whose retrieval results included, i.e.J(d) ={q h|
(qh,Dh)∈ P, d∈ D h}. During validation,Jindexes each
d∈ Dto its associated cached queries, forming a multiset
M=S
d∈DJ(d). For each cached queryq h, its homology
score is thus computed ass(q h) =f(q h)/k, wheref(q h)
denotes the frequency ofq hinM. If there exists aq hsuch
thats(q h)> τ, the query is considered homologous and the
draft is accepted. Otherwise, HaS falls back to full-database
retrieval, and correspondingly updatesP,C c, andJ.
Algorithm 1HaS
Input:A user queryq.
Output:A set of documents.
1:Perform two-channel fast retrieval from bothC fandC cto
obtain candidate setsD fandD c.
2:Re-rank and merge candidates to form the draftD.
3:Initialize a multisetM=∅.
4:foreachd∈ Ddo
5:Find queries through the inverted indexJ(d).
6:M=M ∪ J(d)
7:end for
8:foreach unique queryq h∈ Mdo
9:Count its occurrence frequencyf(q h).
10:Compute the homology score:s(q h) =f(q h)/k.
11:iff(q h)/k > τthen
12:returnAccept draftD.
13:end if
14:end for
15:Perform full-database retrieval to obtain resultD full.
16:UpdateP,C candJ.
17:returnD full
III. KEYMECHANISMS INHAS
In this section, we introduce the key mechanisms of HaS
by explaining: (1) what homology is, (2) how validation is
formulated as a homologous query re-identification problem,
(3) how homology is quantified for re-identification, and (4)
why two channels are incorporated into the framework.
A. What Homology is
To guarantee factuality, the recalled set of documents in the
retrieval stage must include at least one document to provide
the necessary supporting evidence. In common practice, this
is judged by the alignment of entities and attributes between
the query and the document, which forms the basis for our
definition ofgolden documents:Definition1 (Golden document):A documentdis agolden
documentfor a queryqif it contains the evidence to support a
factual response. Concretely,dshould correspond to the target
entity and cover the specific attribute of interest. We define the
indicator function:
G(d, q)≜I[E(q) =E(d)]∧I[A(q)∈ A(d)],
whereI(·)equals true if the condition holds and false other-
wise.dis a golden document toqifG(d, q) =true.
This definition implies a fundamental relationship between
queries: when two queries align in both target entity and
attributes, they necessarily share an overlapping set of golden
documents. We characterize this relationship asfull homology,
as illustrated in Fig.5a:
Definition2 (Full Homology):Full Homologyexists be-
tween two queriesq 1andq 2, or the two queries are considered
fully homologousif they point to the same entity and cover
the same attributes of interest:
Hf(q1, q2)≜I
E(q 1) =E(q 2)
∧I
A(q 1) =A(q 2)
.
Existing reuse-based methods leverage this golden-
document sharing property by matching fully homologous
cached queries, which typically require identical semantics, to
facilitate result reuse. However, their potential for acceleration
is severely constrained in practice, as the inherent diversity of
real-world user queries makes such strict semantic equivalence
exceptionally rare.
Fig. 4: Estimated proportion of queries that have homologous
counterparts in the real world.
This diversity is primarily driven by varying attribute tar-
gets, while real-world queries nonetheless follow the popular-
ity pattern to cluster around popular entities. Pioneering anal-
ysis of search engine logs reveals that entities are queried 3.97
times on average, with over 83.9% of queries sharing a target
entity with at least one other request. We conduct a similar
analysis using Wikipedia’s TopViews statistics. Specifically,
we curated a dataset of the Top-1000 most-visited Wikipedia
entities, sampled weekly every Wednesday over a three-month
duration4. As illustrated in Fig.4, over 60% of queries have
4Data Source: https://pageviews.wmcloud.org/topviews. View counts are
normalized relative to the least-visited page. Entities with a normalized count
exceeding 2 are identified as having at least one homologous counterpart.

such counterparts, and the proportion rises remarkably during
major or trending events (e.g., “Charlie Kirk” on September
10, 2025). We refer to this special relationship between queries
ashomology, as shown in Fig.5b:
Definition3 (Homology):Homologyexists between two
queriesq 1andq 2, or the two queries are consideredhomolo-
gousif they point to the same entity:
H(q 1, q2)≜I
E(q 1) =E(q 2)
.
In line with full homology, our empirical analysis shows
that homologous queries tend to share a subset of golden
documents. Based on this observation, we first formalize the
weaker form of golden-document sharing asquasi-homology:
Definition4 (Quasi-homology):We definequasi-homology
between two queriesq 1andq 2if they share at least one golden
document:
HQ(q1, q2)≜I
∃d, G(d, q 1) =G(d, q 2) =true
.
We then bridge these two concepts through the following
insight:
Insight1:Given two homologous queriesq 1, q2, they are
empirically quasi-homologous.
This empirical insight is substantiated by the following two
observations:
1) Retrieval Exhibits Strong Entity Alignment.Analysis
of retrieval results reveals that an average of 2.35 out of the
top-5 documents are entity-aligned with the query, with 64.3%
of queries having an entity-aligned document in the top-1
position. This alignment is primarily attributed to an entity-
centric bias within semantic encoders, as positive training
samples often contain the same entity. Consequently, a golden
documentdfor queryq 1inherently satisfies the condition
E(d) =E(q 1) =E(q 2).
2) Documents Provide Multi-Attribute Coverage.Com-
prehensive knowledge sources, such as Wikipedia, typically
describe multiple attributes of an entity within a single doc-
ument. As reported in [13], 5% of documents fulfill 60%
of queries, demonstrating the rich knowledge coverage of
documents to support distinct queries. This implies a high
probability that a documentdsupportingq 1also encompasses
the specific attributeA(q 2)required by queryq 2, i.e.,A(q 2)∈
A(d).
Collectively, these observations imply that two homologous
queries,q 1andq 2, are highly likely to share at least one golden
document, thereby exhibiting quasi-homology.
In the subsequent sections, we leverage this prevalent ho-
mologous relationship, alongside this insight, to develop a
robust and efficient mechanism for draft validation.
B. From Validation to Re-identification
For candidate documents (i.e., a draft) speculatively re-
trieved, we need to validate whether it is acceptable following
What was Albert Einstein's field of work?
What was Einstein’sprofession?
Albert Einsteinwas a physicist…
Albert Einstein was a German-born theoretical physicist…
Share AllGolden Doc.Target Entity AlignedTarget AttributeAligned(a) Full Homology
Where was Einsteinborn?
What was Einstein’sprofession?
Albert Einsteinwas a physicist…
Albert Einstein was a German-born theoretical physicist…
Share SomeGolden Doc.Target Entity Aligned
(b) Homology
Fig. 5: An illustration of (fully) homologous queries.
the rule:a draftDfor queryqis considered acceptable if
it contains at least one golden document. We express it as
the following hypothesis:
V(D, q) :∃d∗∈ Ds.t.G(d∗, q) =true.
Thus,the draft validation process is to verify whether
V(D, q)holds.If it is true, the draft can be accepted.
Since bothqanddare in natural language, directly verifying
V(D, q)requires a reliable evaluator (e.g., an LLM) for
G(d, q), which is very costly. Hence, we instead introduce
a surrogate hypothesis ˆV(D, q). It brings in a reference query
qhin the cacheP, requiring thatd∗is a golden document for
bothqandq hand appears in the cached retrieval resultD h:
ˆV(D, q) :∃(q h,Dh)∈ P,∃d∗∈ D ∩ D h,
s.t.G(d∗, q) =G(d∗, qh) =true.
This surrogate hypothesis is not overly strict. Under our
two-channel design, the recalled documentd∗is likely drawn
from the cache channel, meaning it belongs to someD hof
a cached queryq h. Meanwhile, sinceD his obtained through
full-database retrieval,d∗is likely golden toq h. In this way,
validating ˆV(D, q)equivalently verifiesV(D, q).
The existence of such aq hin the cache provides key
evidence for validating the surrogate hypothesis ˆV(D, q). Re-
call that quasi-homologous queries share at least one golden
document, and that homologous queries are likely to ex-
hibit quasi-homology. Therefore, if the surrogate hypothesis
ˆV(D, q)holds, the presence of the shared golden document
d∗implies thatq handqmust be quasi-homologous, and
consequently homologous. Formally, we have:
ˆV(D, q) =⇒ ∃q hs.t.HQ(q, qh) =⇒ ∃q hs.t.H(q, q h).
More importantly, the contrapositive shows that, if none of
such aq hexists, the hypothesis is directly proved false:
∀qhs.t.¬H(q, q h) =⇒ ¬ ˆV(D, q),

(a)
 (b)
 (c)
Fig. 6: Distributions of semantic similarity scores and homology scores for Easy Positives (Fully homologous), Hard Positives
(Homologous), and Negatives (Other) queries. A larger positive area beyond the threshold indicates that more required queries
can be distinguished from others to support draft validation.
which helps the system reject the draft with confidence.
Drawing an analogy to person re-identification, we formu-
late draft validation as a homologous query re-identification
problem. Specifically, the objective is to determine whether
an incoming query constitutes a homologous re-encounter of
a previously observed query, or equivalently, whether any
cached query is homologous to the current one. If no such
homologous counterpart can be identified within the cache, the
draft is rejected with high confidence; otherwise, it is deemed
acceptable.
C. Re-identification via Homology Score
Efficient and accurate re-identification of homologous
queries presents a challenge. Conventional lexical matching
struggles with entity alignment due to the inherent issues of
linguistic ambiguity and polysemy. While LLMs can enhance
robustness, their inference latency is prohibitive. Furthermore,
the semantic similarity score, which is frequently employed to
identify fully homologous queries, is insufficient for this task.
As demonstrated in Fig.6a and Fig.6b, homologous queries
are nearly indistinguishable from non-homologous ones, since
semantics are meanwhile influenced by attribute variations.
Recalling the entity-alignment property inherent in retrieval,
entity-aligned homologous queries are predisposed to recall
overlapping sets of entity-related documents. Building upon
this observation, we exploit the overlap ratio between retrieval
results as a metric for the homology relation between queries.
We formally define thehomology scoreto quantify the likeli-
hood that a reference query is homologous to a target query:
Definition5 (Homology score):Given a queryq 1and a
reference queryq 2, letD 1andD 2denote their respective
retrieval results containingkdocuments each. Thehomology
scoreis defined as the proportion of overlapping documents
between the two sets:s(q 1, q2) =|D 1∩ D 2|/k.
Fig.6c visualizes the efficacy of the homology score. With
the score, homologous queries can be easily separated from
the others with a noticeably distinct distributional gap. Quanti-
tatively, the recall reaches 53.28%, significantly outperforming
the semantic similarity score.With this metric, a threshold-based re-identification process
can be performed, where we try to find a cached query
that happens to have a high homology score with the given
query. If no cached query satisfies this criterion, the draft
is rejected with confidence. Otherwise, the existence of a
homologous queryq htoqin the cache is established. A
high overlap betweenDandD hindicates a greater likelihood
that their shared golden documentd∗lies inD ∩ D h, thereby
strengthening the surrogate hypothesis ˆV(D, q)and supporting
draft acceptance.
Since we have past queries and their documents cached,
instead of computing the homology score for each query, we
leverage a document-query inverted indexJ(d) :D → Qfor
acceleration. For any draftDto be validated, each retrieved
document acts as a probe that maps back to a set of cached
queries, yielding a multisetM D=S
d∈DJ(d). In this way,
the frequencyf(q h)of queryq hinMreflects the number
of overlapping documents between its retrieval result and the
draft. The homology scores(q, q h)can thus be computed by
f(qh)/k. Ifmax qhs(q, q h)> τ, the draft can be accepted.
D. Why two channels are incorporated
The caching mechanism in HaS is designed to sustain
robust homology validation and preserve adequate knowledge
coverage, while still enabling fast retrieval. To achieve this,
HaS integrates two distinct channels: the cache channel and
the fuzzy channel.
The cache channel, which consists of the retrieved doc-
uments of cached queries, is an indispensable component.
It provides the most important knowledge coverage. Once a
homologous query is re-identified, its associated documents
in the cache channel are likely golden to the current query,
supporting a correct response. These documents also serve as
probes to support homology validation. When some of them
appear in the draft, they are linked back to the corresponding
cached queries for re-identification.
However, using the cache channel alone faces two critical
issues. First, ranking-based retrieval introduces noisy docu-
ments. When only< krelevant documents exist, the Top-k
retrieval result is filled with noisy ones. These noisy docu-
ments inflate the homology scores of non-homologous queries

and potentially lead to incorrect re-identification. Second, the
cache spans only a small portion of the knowledge space
and often lacks fine-grained information specific to the target
attribute. To address these limitations, we introduce the fuzzy
channel as a complement.
The fuzzy channel restricts retrieval to a very small
subset, returning fuzzy results with extremely low latency.
We implement it using ANNS with an aggressive configu-
ration. We partition the database into thousands of buckets,
and retrieve only a few dozen of the closest ones. This yields
fuzzy yet sufficiently informative documents. We deliberately
accept the associated moderate accuracy loss in exchange
for extremely fast retrieval. Retrieved documents from both
channels are merged and re-ranked to form the draft.
Incorporating the fuzzy channel enhances both valida-
tion reliability and draft quality.First, although the fuzzy
channel produces fuzzy results, after re-ranking, these weakly-
relevant documents can edge out noisy documents retrieved
from the cache channel in the draft. This mitigates noise-
induced homology score inflation, thereby avoiding incorrect
re-identification and improving validation reliability. Second,
retrieval from the closest buckets improves alignment with the
attribute-specific intent of queries, enabling the fuzzy channel
to capture fine-grained knowledge that may not exist in the
cache channel. Meanwhile, documents from both channels
compete for inclusion in the draft. Only highly relevant ones
are retained, while noisy candidates are discarded. Thus, the
two-channel design yields a high-quality and factual draft. An
illustrative example is provided in Section IV-E to clarify the
above discussion.
IV. EXPERIMENTS
A. Experimental Settings
Implementation Details.We implement HaS on a work-
station equipped with an I9-13900KF CPU and an RTX-
3090 GPU. We primarily useQwen3 8B[14] as the down-
stream LLM, and additionally evaluateLlama3 8B[15] and
Mixtral 7B[16], all in their AWQ-quantized versions. We
simulate a cloud–edge system for deployment. The full-dataset
retrieval service is deployed on the cloud, with a simulated
network latency of 0.1–0.2 seconds injected. It is implemented
withFaiss-IndexPQ[17]. HaS are simulated on the edge
with a network latency of 0.01-0.05 seconds. The fuzzy
channel is implemented byFaiss-IndexIVFPQto retrieve
64 out of 8192 buckets. The cache channel usesHNSWLib.
A maximum of 5000 queries with their retrieved documents
can be cached. A first-in-first-out (FIFO) policy is applied for
cache replacement. Cold start is always included. In default,
kis set to 10 andτis set to 0.2.
Dataset Selection and Augmentation.We collect the
Wikipedia documents dumpon 2023-11-015, and seg-
ment it into 49.2 million passages of less than 100 words [18].
To simulate realistic query patterns, we first note that most
existing QA datasets are filtered and de-duplicated. For exam-
5https://huggingface.co/datasets/wikimedia/wikipediaple, in the Granola-EQ dataset, most entities appear in only
a single query, as shown in Fig.7a. These datasets overlook
the tendency of real-world queries to concentrate on popular
entities, leading to a deviation from realistic usage patterns.
(a) Granola-EQ (Original)
 (b) Granola-EQ (Augmented)
Fig. 7: Distribution of the number of attributes queried per
entity. After sampling queries from the augmented dataset,
the resulting distribution better reflects real-world popularity
patterns.
To address this limitation, we augment existing datasets by
leveraging Wikidata, a knowledge graph that connects entities
through diverse relations. We primarily use theGranola-EQ
dataset [19], which augments the classical EQ dataset [20]
with entity annotations, and also includePopQA[21]. The
workflow is illustrated in Fig.8. We leverage their annotated
entities to find relations and linked entities. A subset of
representative relations is selected. For each relation, we create
5 templates and fill them to obtain the question–answer pairs.
To better reflect real-world patterns, we sample queries from
the augmented dataset while aligning with the daily TopView
Statistics of Wikipedia, as illustrated in Fig.7b.
Metrics.The primary focus lies in system efficiency and
response accuracy. The following metrics are measured:
•Average Latency (AvgL): The average end-to-end re-
trieval latency in seconds over a batch of queries. It
directly represents the system efficiency.
•Response Accuracy (RA): We follow [22] to evaluate
answers by checking whether they contain the golden
answers. It reflects the retrieval quality.
•Document Hit Rate: It is defined as the proportion of
queries for which at least one retrieved document contains
the golden answer.
•Response Accuracy conditioned on Draft Acceptance
(RA@DA): It reports accuracy restricted to accepted
drafts, allowing us to isolate and assess the effect of HaS.
•Draft Acceptance Rate (DAR): It indicates how often
a homologous query is re-identified and the draft is
accepted.
•Correct Acceptance Rate (CAR): It reflects the preci-
sion of the re-identifications and draft acceptances.
Baseline methods.We compare HaS with three repre-
sentative categories of state-of-the-art retrieval strategies for
acceleration:

Generate Question TemplatesForRelations
Sex or gender ': ['How is ${entity}\'s sex or gender identity defined exactly?','What defines the sex or gender identity of ${entity}clearly?’,…… ]{"questions":["HowisLukeProkopec'ssexorgenderidentitydefinedexactly?","WhatdefinesthesexorgenderidentityofLukeProkopecclearly?",…],"answer_list":["Malegender"]}
Fill in the entities and golden answers
Date of birth ': ['What is the exact year ${entity}was born?','Can you specify which year ${entity} was born exactly?',…… ]{"questions": ["What is the exact year Luka Elsner was born?", "Can you specify which year Luka Elsner was born exactly?", … ], "answer_list": ["1978”]}
Search Entity QID
Original Data{“entity”: “LukeProkopec”, “question”: …}
Q6702280
Sex or gender (P21) : maleCountry of citizenship (P27): AustraliaDate of birth(P569): 23 February 1978
Search Relations
Fig. 8: Dataset augmentation workflow. For entity mentions
in existing datasets, we retrieve their connected relations and
target entities from Wikidata. We then use GPT-4o to generate
templates and construct the augmented QA dataset.
•We compare HaS with mature ANNS methods, including
IVF and ScaNN, under two configurations: (1) using a
similar retrieval scope to replace HaS on the edge, and (2)
using an optimized scope to replace full-database retrieval
on the cloud.
•We consider reuse-based methods, including Proximity
[12], SafeRadius [11], and MinCache [23]. They skip re-
trieval by reusing cached results of semantically identical
queries.
•To illustrate the efficiency of the homology validation,
we replace it with the evaluator from CRAG [24], which
calls the LLM to assess each retrieved document.
B. Main Results
Validation is crucial to draft quality.We first compare
HaS with ANNS methods under a similar retrieval scope ratio,
with ANNS deployed on the edge as an alternative to HaS. As
shown at the top of Table II, ANNS methods under a narrow
retrieval scope have a high risk of missing critical documents.
Without a validation mechanism, they suffer over 10% ac-
curacy degradation, making their latency savings ineffective.
In contrast, HaS can perform validation and fallback to ensure
draft quality. Its better response accuracy highlights the critical
role of validation in speculative retrieval.
ANNS methods and HaS are complementary.In practical
deployments, ANNS methods are typically configured with
a larger, fine-tuned retrieval scope, as a substitute for full-
database retrieval on the cloud. While these methods achieve
strong performance, they are complementary to HaS. Building
on their strengths, HaS can be integrated with them to further
improve efficiency. We evaluate this combination at the bottomof Table II and observe that HaS brings an additional 7%–28%
latency reduction on top of their performance.
TABLE II: Comparison of HaS and ANNS Methods.♠:
Limited retrieval scope to replace HaS.♢: Optimized retrieval
scope to replace full-database retrieval.↑/↓: Higher / lower
is better.⋆: Sampled Queries from the Augmented dataset.
MethodsGranola-EQ⋆PopQA⋆
AvgL(s)↓RA↑AvgL(s)↓RA↑
IVF♠0.0921 0.4283 0.0916 0.2569
ScaNN♠0.0802 0.4353 0.0778 0.2524
HaS 1.0559 0.4829 0.8725 0.2906
IVF♢0.5431 0.4824 0.5432 0.2825
HaS + IVF♢0.4603 0.4786 0.3872 0.2784
-15.24% -0.79% -28.73% -1.48%
ScaNN♢0.3554 0.4824 0.3553 0.2862
HaS + ScaNN♢0.3285 0.4790 0.2904 0.2812
-7.55% -0.70% -18.27% -1.76%
HaS Leverages Homologous Queries to Achieve Higher
Efficiency.Existing reuse-based methods attempt to locate a
semantically identical query for result reuse. However, such
queries are rarely present in a limited cache, limiting their
efficiency. As shown in Table III, while maintaining only a
small drop in response accuracy, they fail to achieve substantial
latency reduction. Fig.9 further examines their performance
across different thresholds. Lowering the threshold relaxes the
matching criteria for acceleration, but at the cost of reduced
response accuracy. HaS consistently demonstrates Pareto su-
periority. By exploiting more prevalent homologous queries
for validation, HaS achieves greater opportunities for draft
acceptance, thus realizing a larger latency reduction.
Fig. 9: Comparison on varying threshold settings. Point size in-
dicates the percentage of relevant queries found (semantically
similar or homologous). MinCache involves two thresholds,
resulting in multiple performance curves.
Homologous Validation is more efficient.To evaluate the
benefit of our homologous validation, we replace it with the
LLM-based evaluator from CRAG. Some additional metrics
are provided in Table IV. While the LLM-based evaluator
more precisely identifies relevant documents, it introduces
approximately 0.7 seconds for LLM inference per query.
Despite accepting 42.2% of drafts, the method still results in a
9.76% increase in average retrieval latency compared to full-
database retrieval. Meanwhile, it exhibits weaker confidence

TABLE III: Performance comparisons. The percentage difference is computed relative to the full-database retrieval. CRAG†:
Replaces homologous validation with the LLM-based evaluator from CRAG.↑/↓: Higher / lower is better.⋆: Sampled Queries
from the Augmented dataset.
MethodGranola-EQ⋆PopQA⋆
AvgL(s)↓Doc. Hit
Rate↑Response Accuracy↑AvgL(s)↓Doc. Hit
Rate↑Response Accuracy↑
Qwen38B Llama 8B Mixtral 7B Qwen38B Llama 8B Mixtral 7B
Full-Database
Retrieval1.3845 0.6457 0.4875 0.4715 0.4806 1.3847 0.4652 0.2970 0.2780 0.2703
Reuse-based Methods
Proximity1.3186 0.6397 0.4824 0.46560.47641.1328 0.4415 0.2802 0.2614 0.2522
-4.76% -0.92% -1.04% -1.25%-0.86%-18.19% -5.09% -5.66% -5.96% -6.70%
MinCache1.3044 0.6329 0.4746 0.4590 0.4679 1.0437 0.4254 0.2676 0.2452 0.2360
-5.78% -1.97% -2.64% -2.65% -2.64% -24.63% -8.55% -9.91% -11.81% -12.68%
SafeRadius1.2870 0.6355 0.4779 0.4603 0.4718 0.9773 0.4245 0.2649 0.2477 0.2338
-7.05% -1.58% -1.97% -2.38% -1.82% -29.42% -8.74% -10.83% -10.90% -13.51%
LLM for Validation
CRAG† 1.5196 0.6287 0.4702 0.4549 0.4625 1.8186 0.4557 0.2885 0.2706 0.2542
9.76% -2.63% -3.54% -3.53% -3.75% 31.33% -2.04% -2.87% -2.65% -5.97%
HaS1.0559 0.6402 0.4829 0.46670.47550.8725 0.4578 0.2906 0.2720 0.2638
-23.74% -0.84% -0.94% -1.03%-1.06%-36.99% -1.60% -2.14% -2.14% -2.41%
when evaluating out-of-distribution data in PopQA. As most
drafts are rejected, the latency overhead escalates by 31.33%.
In contrast, by leveraging lightweight validation with minimal
overhead, HaS achieves substantially lower average retrieval
latency.
TABLE IV: Comparison on additional performance metrics.
L@DA and L@DR denote the per-query retrieval latency (in
seconds) upon draft acceptance and rejection.
DAR↑
(Granola-EQ⋆)DAR↑
(PopQA⋆)L@DA
(s)↓L@DR
(s)↓
CRAG†42.20% 20.82% 0.7006 2.1168
HaS 29.63% 43.15% 0.0555 1.4896
HaS is generalizable across datasets.Table III shows
that HaS exhibits steady performance gain across datasets. To
further assess its generalizability, we additionally evaluate it
on the TriviaQA6[25] and SQuAD [26] datasets, although
these benchmarks deviate from real-world popularity patterns
to contain more scattered queries. As shown in Table V,
such variability is expected to affect all methods that rely
on the query popularity. In this setting, by exploiting the
prevalence of homologous queries, HaS can still deliver better
performance gains. By contrast, reuse-based methods rely
on less frequent semantically identical queries, leading to
significant performance degradation, particularly on SQuAD.
C. Analysis on the Fuzzy Channel
Fuzzy Channel Enhances Validation Precision and Draft
Quality.The fuzzy channel in HaS serves a dual role:
enhancing homologous validation and improving draft qual-
ity. We conduct an ablation study under two settings: (i)
removing it from validation, where only documents from
the cache channel are used for homologous verification7;
and (ii) disabling it for draft enhancement, where accepted
6Filtered to contain only well-organized single-hop queries
7To prevent cold-start failures due to repeated matches, only in this
experiment, we pre-fill the cache with random queries.TABLE V: Performance comparison with reuse-based methods
on two datasets that deviate from real-world query patterns. ‡:
queries sampled from the original dataset.
MethodTriviaQA‡ SQuAD‡
AvgL(s)↓RA↑AvgL(s)↓RA↑
Full-Database
Retrieval1.3843 0.7615 1.3846 0.2787
Proximity1.3460 0.7601 1.4135 0.2779
-2.77% -0.19% 2.09% -0.28%
MinCache1.3438 0.7574 1.4204 0.2781
-2.93% -0.54% 2.59% -0.21%
Safe Radius1.3296 0.7572 1.4918 0.2758
-3.96% -0.56% 7.75% -1.04%
Ours1.0810 0.7603 1.2910 0.2782
-21.91% -0.15% -6.76% -0.19%
drafts exclude documents from the fuzzy channel. Results are
illustrated in Table VI. Excluding the fuzzy channel during
validation abnormally increases DAR but reduces CAR. It
indicates that the system mistakenly matches non-homologous
queries, thus undermining validation reliability and reducing
response accuracy. When removing the fuzzy channel for draft
enhancement, RA@DA drops from 0.4907 to 0.4504. The
degradation is more pronounced under unreliable validation
settings. It underscores its importance in enhancing draft
quality, particularly when documents from the cache channel
are unreliable.
TABLE VI: Impact of incorporating the fuzzy channel for
validation (V) and draft enhancement (E).
V E AvgL(s)↓RA↑DAR↑CAR↑RA@DA↑
× ×0.14110.230892.80% 34.77%0.2076
×✓ 0.4294 0.4216
✓×0.99260.467334.67% 88.77%0.4504
✓ ✓ 0.4813 0.4907
Compressing the Fuzzy Channel Enables Resource-
Constrained Deployment.Although the fuzzy channel per-
forms ANNS to retrieve only a small scope of the database, it
still has to load the entire database, incurring considerable stor-
age overhead. To enable deployment on resource-constrained

edge devices, HaS can compress the fuzzy channel by loading
only a smaller subset of the database. We evaluate the system
performance under different subset proportions to examine this
characteristic. On top of Table VII, with a fixed thresholdτ,
the performance shows a decline as the proportion shrinks.
Since it becomes less likely to include documents relevant
to the current query, its validation and enhancement ability
are weakened. However, this degradation can be mitigated by
adjusting the thresholdτ. The lower half of the table shows the
fine-tuned performances. Even with the proportion reduced to
just 1% of the full database, tuning the threshold toτ= 0.6
enables HaS to maintain comparable performance, incurring
only a 4.4% increase in AvgL and a 2.8% drop in RA.
These findings confirm the robustness of HaS under aggressive
compression, enabling practical and efficient deployment on
lightweight devices.
TABLE VII: Impact of the proportion of the full database
used for fuzzy channel construction and matching threshold
τon system performance. Fine-tuningτenables maintaining
comparable performance as the fuzzy channel compressed.
%τ AvgL(s)↓RA↑DAR↑RA@DA↑
1
0.20.4982 0.3236 67.38% 0.2361
10 0.8223 0.4342 45.07% 0.3643
50 0.9736 0.4696 34.73% 0.4414
100 1.0559 0.4829 29.63% 0.4892
1 0.6 1.0964 0.4698 25.71% 0.4323
10 0.4 1.0573 0.4706 28.70% 0.4440
50 0.3 1.0776 0.4790 27.49% 0.4732
100 0.2 1.0559 0.4829 29.63% 0.4892
D. Parameter Analysis
VaryingkReveals a U-Shaped Performance Impact.
The number of retrieved documentskaffects both homolo-
gous validation and response generation. Since its U-shaped
influence on generation has been discussed in prior work
[27], we focus on its role in homologous validation. Fig.10
shows that, unlike random queries, homologous queries tend
to share retrieved documents at top rank positions. A smaller
kcan better capture these overlaps and yield higher validation
precision. As evidenced by the left side of Fig.11, decreasing
kfilters noise on lower rank positions, thus bringing better
re-identification precision and higher CAR. Combining both
aspects, the influence ofkin HaS follows a U-shaped trend,
as demonstrated by the right side of Fig.11. While a smaller
kimproves validation accuracy, it risks omitting factual doc-
uments.k= 10strikes the best balance, whereas largerk
slightly degrades performances due to increased noise and
weaker validation.
Increasingτtrades latency for accuracy.The thresholdτ
distinguishes homologous queries from others during the re-
identification process. Increasingτenforces stricter homology
validation, directly affecting the latency–accuracy trade-off.
As shown in Table VIII, higherτvalues generally increase
average latency while improving response accuracy, as a
stricter condition is enforced to re-identify homologous queries
and accept drafts. However, excessively highτsuffers from
Fig. 10: Joint distribution of document rankings for two
homologous queries or two random queries. Warm-colored
pixels at the top-left corner indicate that the two queries have
many retrieved documents in common.
Fig. 11: System performance under differentkand varying
thresholds.
diminishing returns, causing unnecessary latency for marginal
accuracy gains. In our setup,τ= 0.2achieves a satisfactory
balance.
HaS is robust to the choice of encoders.In retrieval, the
underlying encoder serves to extract features for ENNS, and
variations across encoders can result in differing retrieval qual-
ity and preferences. In addition to the classicalContriever
[28], we select two recent encoders,BGE-Large-en-v1.5
[29] ande5-Base-v2[30]. Table VIII demonstrates that
HaS maintains robust performance across these encoders.
It consistently reduces average latency compared to full-
database retrieval while preserving comparable response accu-
racy. Moreover, under the sameτ, HaS exhibits similar trade-
offs across different encoders. This further highlights its plug-
and-play nature. Switching encoders does not require complex
parameter tuning.
Larger cache size leads to better system efficiency.To iso-
late the impact of cache size on efficiency, we adjust thresholds
to maintain comparable response accuracy across settings. As
shown in Table IX, enlarging the cache consistently enhances
efficiency by increasing the likelihood of re-identifying homol-
ogous queries and draft acceptance. Meanwhile, the overhead
introduced by a larger cache is minimal. For queries with
draft acceptance, retrieval latency remains nearly constant.
When the draft is rejected, FIFO-based updates incur only an
additional∼12 ms per query, which is negligible compared to
the latency saved. Furthermore, the memory footprint increases

TABLE VIII: Impact of the encoder and thresholdτon system
performance. HaS is robust to different encoders. Higher
values ofτtrade off average latency for improved response
accuracy.
τ AvgL(s)↓RA↑
ContrieverFull-Database
Retrieval- 1.3845 0.4875
HaS0.1 0.8491 0.4709
0.2 1.0496 0.4835
0.3 1.1477 0.4863
BGE-LargeFull-Database
Retrieval- 1.4191 0.4971
HaS0.1 0.9076 0.4788
0.2 1.1528 0.4922
0.3 1.2708 0.4954
e5-BaseFull-Database
Retrieval- 1.4923 0.5003
HaS0.1 0.9256 0.4844
0.2 1.0586 0.4932
0.3 1.2425 0.4965
linearly at a modest rate, approximately 5 MB for every ad-
ditional 1,000 entries, which remains well within the capacity
of modern server infrastructures. These results highlight that
enlarging the cache enhances system effectiveness without
incurring significant resource overhead.
TABLE IX: System efficiency under varying cache sizeH max.
L@DA and L@DR denote the per-query retrieval latency (in
seconds) upon draft acceptance and rejection. Mem represents
the memory footprint (in MB) of the cache.
Hmax AvgL(s)↓DAR↑L@DA(s)↓L@DR(s)↓Mem(MB)
2000 1.2702 14.30% 0.0547 1.4778 15.10
3000 1.1876 20.24% 0.0549 1.4828 20.45
4000 1.1165 25.31% 0.0552 1.4892 26.59
5000 1.0559 29.63% 0.0555 1.4896 31.38
E. Use Case
I. Case Study of HaS on Simple Queries.We present an
illustrative case study in Fig.12. For a query, a two-channel
fast retrieval is first performed, followed by the homology
validation. Due to the entity-alignment preference in retrieval,
although the retrieval results of the cached query focus on the
occupation attribute, some of these documents (Documents 2
and 3) are also included in the draft. This correctly confirms
their homology and leads to the acceptance of the draft.
This facilitates the efficiency and accuracy of our validation
mechanism.
This example also highlights the advantages of the fuzzy
channel. First, the fuzzy channel enhances validation reli-
ability. As illustrated in Fig.12a, when only 2 documents
are relevant, rank-based retrieval on the cache channel re-
turns a noisy Document 4 among the Top-3 results. Without
the fuzzy channel, this document would be included in the
draft, erroneously inflating the homology score of the linked
irrelevant query, thus increasing the risk of incorrect re-
identification. Incorporating the fuzzy channel provides more
relevant documents, such as Document 1, to replace noisy onesfor better validation reliability. Second, the fuzzy channel com-
plements the cache channel by contributing additional valuable
documents. In Fig.12b, assuming the cache channel lacks the
informative Document 2, the fuzzy channel contributes the
supplemental Document 1, safeguarding the system against
accuracy degradation.
CanyouspecifywhichyearDonaldFagenwasbornexactly?
2. Donald Jay Fagen (born January 10, 1948) …
3. Fagen'sfourth album was released in 2012…
4. Donald Adeosun Faison(born June 22, 1974) …
Retrieve From Cache ChannelHomologyQueryScore = 2/ 3
IrrelevantQueryScore = 1/ 3
1. Donald Fagen was born on January 10, 1948 …
Retrieve From Fuzzy Channel
Merge & Re-Rank1
2
3
4
…
Draft
(a) Two-channel fast retrieval.
CanyouspecifywhichyearDonaldFagenwasbornexactly?
Current Query
1. Donald Fagen was born on January 10, 1948…
DraftFrom Fuzzy ChannelFrom Cache ChannelCached Retrieval Results
InwhatgenreorareaisDonaldFagenactive?Cached Query
1948LLM Response3. Fagen's fourth album was released in 2012…
2. Donald Jay Fagen (born January 10, 1948) is an American musician …
6. Fagen's third solo album, Morph the Cat, …
3. Fagen's fourth album was released in 2012…
2. Donald Jay Fagen (born January 10, 1948) is an American musician …
(b) Homology validation.
Fig. 12: An illustrative example for a case study.
II. Plugging HaS into modern RAG pipelines to solve
complex queries.In previous sections, we evaluated HaS
within a basic RAG pipeline for simple queries. This does not
mean that HaS is restrictive. As a pluggable component, HaS
can be seamlessly integrated into any RAG pipelines without
much effort. Practically, one only needs to direct queries
to our HaS for retrieval. No modification to the pipeline
implementation is required. With built-in query decomposition
mechanisms in modern pipelines, complex queries involving
multiple entities and attributes can be solved iteratively.
As a case study, we integrate HaS into Auto-RAG [6], a
Chain-of-Thought-based RAG pipeline. It performs query de-
composition and iterative retrieval to handle complex queries.
We similarly constructed a 2-hop dataset for evaluation. Fig.13
shows that HaS reduces average retrieval latency by 69.4%
with only 3.72% accuracy degradation. The high frequency of
homologous patterns among decomposed sub-queries boosts
the draft acceptance rate, bringing significant acceleration.
Fig.14 presents an example. For a complex query, Auto-RAG
decomposes the query into sub-queries through reasoning. For
the first sub-query, since a homologous query is re-identified,

Fig. 13: System performance with and without HaS in the
Auto-RAG pipeline on complex queries.
the draft is accepted for acceleration. Auto-RAG then proceeds
to the next hop. At this point, the draft is rejected, and the
system falls back to the full-database retrieval. These results
demonstrate that HaS functions as a practical drop-in solution
for accelerating modern RAG systems.
WhatlanguagesdoesthecomposerofAmokTimespeak?
Analysis:IwouldneedtoknowthecomposerofAmokTime.InitialQuery:WhoisthecomposerofAmokTime?
Amok Time" is…, scored by Gerald Fried
HaSDraft Acceptance
Analysis:thecomposerofAmokTimeisGeraldFried.WeneedtofindthelanguagesspokenbyGeraldFried.RefinedQuery:WhatlanguagesdoesGeraldFriedspeak?'HaSDraft RejectionFull-Database Retrieval
Gerald Fried was an Americancomposer …
BasedontheRetrievedDocument,thecomposerofAmokTimeisGeraldFried,andhespeakEnglish.FinalAnswer:English
Fig. 14: An illustrative case study demonstrating how HaS can
be plugged into Auto-RAG to solve multi-hop queries.
V. RELATEDWORKS
A. Empowering LLMs with an Efficient RAG System
RAG-empowered LLM systems have attracted extensive
research effort, particularly on efficiency. A primary direction
focuses on reducing prefilling latency caused by extra context
length. At the lexical level, methods like RECOMP [31] try
to filter out irrelevant content to shorten the token length of
prompts. In [32], documents are distributed across multiple
specialists for parallel accelerated generation, followed by
a generalist who selects the best result. At the embedding
level, works like xRAG [4] compress features of multiple
tokens into a single one as input to the downstream LLM.
Several studies explore caching key-value (KV) features to
bypass prefilling. TurboRAG [33] precomputes KV features
of documents, and directly concatenates retrieved ones to skipprefill. However, the omission of cross-document attention
risks diminishing LLM response accuracy. Considering this
limitation, RAGCache [3] employs a prefix tree structure for
KV caching, matching only the longest prefix. CacheBlend
[2] selectively recomputes part of tokens exhibiting high KV
deviation, thus mitigating accuracy degradation with minimal
computational costs.
B. Accelerating Retrieval over Large-Scale Corpora
Retrieval latency in large-scale RAG systems has become a
bottleneck [5]. Industrial solutions like Faiss typically employ
ANNS methods such as IVF, HNSW [34], and ScaNN [10],
which aim to retrieve within a smaller scope for acceleration.
Despite their efficiency, they inevitably sacrifice accuracy. To
mitigate this issue, some studies introduce evaluators to assess
retrieved documents [24], [35], [36], though the associated
inference latency diminishes the overall efficiency gains. Fed-
erated retrieval adheres to a similar underlying concept. They
partition the large corpus into shards. A router selects relevant
shards for parallel retrieval [37], [38]. Although it brings
retrieval acceleration, the computational cost associated with
the router remains substantial.
Another intuitive optimization is results reuse. A caching
mechanism stores historical queries and their corresponding
results, allowing reuse when sufficient semantic similarity with
current queries is established. Common approaches quantify
semantic similarity through cosine similarity between query
embeddings [12]. A safe radius criterion has been proposed
in [11], defining a hyperball covering the cached query and
retrieved results, with reuse permitted if the current query
lies within this region. A three-tier matching framework is
introduced in [23], featuring a resemblance matching step
that assesses similarity using lexical-level Jaccard similarity
over MinHash signatures. However, in real-world scenarios,
users tend to issue queries about different attributes of the
same entity. This diversity conflicts with the strict semantic
similarity requirement, limiting acceleration gains.
VI. CONCLUSION
We present HaS, a speculative retrieval-based framework
that leverages historical homologous queries for lightweight
draft validation. HaS first performs two-channel fast retrieval
from a cache channel and a fuzzy channel. They jointly
define a much narrower retrieval range than the full database
to provide draft documents with much lower latency. By
trying to re-identify a homologous query from the cache,
HaS determines the draft’s relevance and skips full retrieval
if validated. Experiments show that HaS significantly reduces
average retrieval latency with minimal accuracy degradation,
consistently outperforming other methods. As a modular re-
triever, HaS can also be seamlessly integrated into existing
RAG pipelines, delivering further efficiency gains as modern
pipelines continue to grow in depth and complexity.

AI-GENERATEDCONTENTACKNOWLEDGEMENT
Portions of the text in this paper are refined using Ope-
nAI GPT-5 and Google Gemini-2.5-Flash to improve clarity,
grammar, and readability. The AI systems are used solely for
language polishing. All ideas, figures, analyses, results, and
conclusions are entirely those of the authors.
REFERENCES
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive nlp tasks,” in
Advances in Neural Information Processing Systems, vol. 33, 2020, pp.
9459–9474.
[2] J. Yao, H. Li, Y . Liu, S. Ray, Y . Cheng, Q. Zhang, K. Du, S. Lu, and
J. Jiang, “Cacheblend: Fast large language model serving for rag with
cached knowledge fusion,” inProceedings of the Twentieth European
Conference on Computer Systems, 2025, p. 94–109.
[3] C. Jin, Z. Zhang, X. Jiang, F. Liu, S. Liu, X. Liu, and X. Jin, “RAG-
Cache: Efficient knowledge caching for retrieval-augmented generation,”
ACM Trans. Comput. Syst., vol. 44, no. 1, Nov. 2025.
[4] X. Cheng, X. Wang, X. Zhang, T. Ge, S.-Q. Chen, F. Wei, H. Zhang, and
D. Zhao, “xrag: Extreme context compression for retrieval-augmented
generation with one token,” inAdvances in Neural Information Process-
ing Systems, vol. 37, 2024, pp. 109 487–109 516.
[5] D. Quinn, M. Nouri, N. Patel, J. Salihu, A. Salemi, S. Lee, H. Za-
mani, and M. Alian, “Accelerating retrieval-augmented generation,” in
Proceedings of the 30th ACM International Conference on Architectural
Support for Programming Languages and Operating Systems, Volume 1,
2025, p. 15–32.
[6] T. Yu, S. Zhang, and Y . Feng, “Auto-RAG: Autonomous retrieval-
augmented generation for large language models,” 2024. [Online].
Available: https://arxiv.org/abs/2411.19443
[7] Y . Liu, X. Peng, X. Zhang, W. Liu, J. Yin, J. Cao, and T. Du, “RA-ISF:
Learning to answer and understand from retrieval augmentation via it-
erative self-feedback,” inFindings of the Association for Computational
Linguistics: ACL 2024, 2024, pp. 4730–4749.
[8] D. Yang, J. Rao, K. Chen, X. Guo, Y . Zhang, J. Yang, and Y . Zhang,
“Im-rag: Multi-round retrieval-augmented generation through learning
inner monologues,” inProceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval,
2024, p. 730–740.
[9] Y . Gao, Y . Xiong, M. Wang, and H. Wang, “Modular
RAG: Transforming RAG systems into LEGO-like reconfigurable
frameworks,” 2024. [Online]. Available: https://arxiv.org/abs/2407.21059
[10] R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, and
S. Kumar, “Accelerating large-scale inference with anisotropic vector
quantization,” inProceedings of the 37th International Conference on
Machine Learning, vol. 119, 2020, pp. 3887–3896.
[11] O. Frieder, I. Mele, C. I. Muntean, F. M. Nardini, R. Perego, and
N. Tonellotto, “Caching historical embeddings in conversational search,”
ACM Trans. Web, vol. 18, no. 4, 2024.
[12] S. A. Bergman, Z. Ji, A.-M. Kermarrec, D. Petrescu, R. Pires, M. Randl,
and M. de V os, “Leveraging approximate caching for faster retrieval-
augmented generation,” inProceedings of the 5th Workshop on Machine
Learning and Systems, 2025, p. 66–73.
[13] S. Agarwal, S. Sundaresan, S. Mitra, D. Mahapatra, A. Gupta,
R. Sharma, N. J. Kapu, T. Yu, and S. Saini, “Cache-craft: Managing
chunk-caches for efficient retrieval-augmented generation,”Proc. ACM
Manag. Data, vol. 3, no. 3, 2025.
[14] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao,
C. Huang, C. Lvet al., “Qwen3 technical report,” 2025. [Online].
Available: https://arxiv.org/abs/2505.09388
[15] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian,
A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan
et al., “The Llama 3 herd of models,” 2024. [Online]. Available:
https://arxiv.org/abs/2407.21783
[16] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot,
D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier,
L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang,
T. Lacroix, and W. E. Sayed, “Mistral 7b,” 2023. [Online]. Available:
https://arxiv.org/abs/2310.06825[17] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazar ´e,
M. Lomeli, L. Hosseini, and H. J ´egou, “The faiss library,”IEEE
Transactions on Big Data, 2025.
[18] V . Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen,
and W.-t. Yih, “Dense passage retrieval for open-domain question an-
swering,” inProceedings of the 2020 Conference on Empirical Methods
in Natural Language Processing (EMNLP), 2020, pp. 6769–6781.
[19] G. Yona, R. Aharoni, and M. Geva, “Narrowing the knowledge eval-
uation gap: Open-domain question answering with multi-granularity
answers,” inProceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 6737–
6751.
[20] C. Sciavolino, Z. Zhong, J. Lee, and D. Chen, “Simple entity-centric
questions challenge dense retrievers,” inProceedings of the 2021 Con-
ference on Empirical Methods in Natural Language Processing, 2021,
pp. 6138–6148.
[21] A. Mallen, A. Asai, V . Zhong, R. Das, D. Khashabi, and H. Hajishirzi,
“When not to trust language models: Investigating effectiveness of para-
metric and non-parametric memories,” inProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 2023, pp. 9802–9822.
[22] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning
to retrieve, generate, and critique through self-reflection,” inThe Twelfth
International Conference on Learning Representations, 2023.
[23] K. Haqiq, M. V . Jahan, S. A. Farimani, and S. M. F. Masoom, “Min-
Cache: A hybrid cache system for efficient chatbots with hierarchical
embedding matching and LLM,”Future Generation Computer Systems,
vol. 170, p. 107822, 2025.
[24] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling, “Corrective retrieval
augmented generation,” 2024. [Online]. Available: https://arxiv.org/abs/
2401.15884
[25] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large
scale distantly supervised challenge dataset for reading comprehension,”
2017. [Online]. Available: https://arxiv.org/abs/1705.03551
[26] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+
questions for machine comprehension of text,” inProceedings of the
2016 Conference on Empirical Methods in Natural Language Process-
ing, 2016, pp. 2383–2392.
[27] B. Jin, J. Yoon, J. Han, and S. O. Arik, “Long-context LLMs meet
RAG: Overcoming challenges for long inputs in RAG,” inThe Thirteenth
International Conference on Learning Representations, 2025.
[28] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin,
and E. Grave, “Unsupervised dense information retrieval with contrastive
learning,” 2022. [Online]. Available: https://arxiv.org/abs/2112.09118
[29] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y . Nie,
“C-Pack: Packed resources for general chinese embeddings,” 2024.
[Online]. Available: https://arxiv.org/abs/2309.07597
[30] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder,
and F. Wei, “Text embeddings by weakly-supervised contrastive pre-
training,” 2024. [Online]. Available: https://arxiv.org/abs/2212.03533
[31] F. Xu, W. Shi, and E. Choi, “RECOMP: Improving retrieval-augmented
LMs with context compression and selective augmentation,” inThe
Twelfth International Conference on Learning Representations, 2024.
[32] Z. Wang, Z. Wang, L. Le, S. Zheng, S. Mishra, V . Perot, Y . Zhang,
A. Mattapalli, A. Taly, J. Shang, C.-Y . Lee, and T. Pfister, “Speculative
RAG: Enhancing retrieval augmented generation through drafting,” in
The Thirteenth International Conference on Learning Representations,
2025.
[33] S. Lu, H. Wang, Y . Rong, Z. Chen, and Y . Tang, “TurboRAG: Ac-
celerating retrieval-augmented generation with precomputed KV caches
for chunked text,” inProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, 2025, pp. 6588–6601.
[34] Y . A. Malkov and D. A. Yashunin, “Efficient and robust approxi-
mate nearest neighbor search using hierarchical navigable small world
graphs,”IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, vol. 42, no. 4, pp. 824–836, 2020.
[35] Z. Hei, W. Liu, W. Ou, J. Qiao, J. Jiao, G. Song, T. Tian,
and Y . Lin, “DR-RAG: Applying dynamic document relevance to
retrieval-augmented generation for question-answering,” 2024. [Online].
Available: https://arxiv.org/abs/2406.07348
[36] Y . Zhu, J.-C. Gu, C. Sikora, H. Ko, Y . Liu, C.-C. Lin, L. Shu, L. Luo,
L. Meng, B. Liu, and J. Chen, “Accelerating inference of retrieval-
augmented generation via sparse context selection,” inThe Thirteenth
International Conference on Learning Representations, 2025.

[37] J. Li, C. Xu, L. Jia, F. Wang, C. Zhang, and J. Liu, “EACO-RAG: Edge-
Assisted and Collaborative RAG with Adaptive Knowledge Update,”
Oct. 2024.
[38] P. Shojaee, S. S. Harsha, D. Luo, A. Maharaj, T. Yu, and Y . Li,
“Federated retrieval augmented generation for multi-product question
answering,” inProceedings of the 31st International Conference on
Computational Linguistics: Industry Track, 2025, pp. 387–397.