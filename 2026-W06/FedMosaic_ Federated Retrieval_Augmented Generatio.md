# FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters

**Authors**: Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, Yongxin Tong

**Published**: 2026-02-05 02:52:49

**PDF URL**: [https://arxiv.org/pdf/2602.05235v1](https://arxiv.org/pdf/2602.05235v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge to improve factuality and reduce hallucinations. Yet most deployments assume a centralized corpus, which is infeasible in privacy aware domains where knowledge remains siloed. This motivates federated RAG (FedRAG), where a central LLM server collaborates with distributed silos without sharing raw documents. In context RAG violates this requirement by transmitting verbatim documents, whereas parametric RAG encodes documents into lightweight adapters that merge with a frozen LLM at inference, avoiding raw-text exchange. We adopt the parametric approach but face two unique challenges induced by FedRAG: high storage and communication from per-document adapters, and destructive aggregation caused by indiscriminately merging multiple adapters. We present FedMosaic, the first federated RAG framework built on parametric adapters. FedMosaic clusters semantically related documents into multi-document adapters with document-specific masks to reduce overhead while preserving specificity, and performs selective adapter aggregation to combine only relevance-aligned, nonconflicting adapters. Experiments show that FedMosaic achieves an average 10.9% higher accuracy than state-of-the-art methods in four categories, while lowering storage costs by 78.8% to 86.3% and communication costs by 91.4%, and never sharing raw documents.

## Full Text


<!-- PDF content starts -->

FedMosaic: Federated Retrieval-Augmented Generation via
Parametric Adapters
Zhilin Liang
SKLCCSE Lab
Beihang University
Beijing, China
zlliang@buaa.edu.cnYuxiang Wang
SKLCCSE Lab
Beihang University
Beijing, China
yuxiangwang@buaa.edu.cnZimu Zhou
Department of Data Science
City University of Hong Kong
Hong Kong, China
zimuzhou@cityu.edu.hk
Hainan Zhang
Beijing Advanced Innovation Center
Beihang University
Beijing, China
zhanghainan1990@163.comBoyi Liu
SKLCCSE Lab
Beihang University
Beijing, China
boyliu@buaa.edu.cnYongxin Tong
SKLCCSE Lab
Beihang University
Beijing, China
yxtong@buaa.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language
Models (LLMs) by grounding generation in external knowledge
to improve factuality and reduce hallucinations. Yet most deploy-
ments assume a centralized corpus, which is infeasible in privacy-
aware domains where knowledge remains siloed. This motivates
federated RAG (FedRAG), where a central LLM server collabo-
rates with distributed silos without sharing raw documents. In-
context RAG violates this requirement by transmitting verbatim
documents, whereas parametric RAG encodes documents into light-
weight adapters that merge with a frozen LLM at inference, avoiding
raw-text exchange. We adopt the parametric approach but face two
unique challenges induced by FedRAG: high storage and communi-
cation from per-document adapters, and destructive aggregation
caused by indiscriminately merging multiple adapters. We present
FedMosaic, the first federated RAG framework built on paramet-
ric adapters. FedMosaic clusters semantically related documents
into multi-document adapters with document-specific masks to
reduce overhead while preserving specificity, and performs selec-
tive adapter aggregation to combine only relevance-aligned, non-
conflicting adapters. Experiments show that FedMosaic achieves
an average10 .9%higher accuracy than state-of-the-art methods in
four categories, while lowering storage costs by78 .8%to86.3%and
communication costs by91 .4%, and never sharing raw documents.
CCS Concepts
•Computing methodologies→Learning paradigms.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, July 2017, Washington, DC, USA
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnnKeywords
Retrieval-Augmented Generation; Parametric Adapters; Federated
Computing
ACM Reference Format:
Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, and Yongxin
Tong. 2026. FedMosaic: Federated Retrieval-Augmented Generation via Para-
metric Adapters. In.ACM, New York, NY, USA, 11 pages. https://doi.org/10.
1145/nnnnnnn.nnnnnnn
1 Introduction
Retrieval-Augmented Generation (RAG) [ 12] has emerged a core
technique for enhancing Large Language Models (LLMs) by ground-
ing their responses indynamic external knowledgerather than static
pre-training alone. By retrieving relevant documents and condi-
tioning generation on them, RAG improves factual accuracy and
mitigates hallucination, driving its adoption in applications such as
conversational search [ 43], enterprise knowledge assistants [ 6], and
domain-specific Q&A [ 50]. However, most RAG deployments as-
sume access to acentralizedcorpus (e.g.Wikipedia, Common Crawl,
or enterprise repositories). While feasible in open-domain settings,
this assumption fails in vertical domains such as healthcare and
finance, where critical knowledge remains locked in institutional
silos due to privacy and compliance constraints.
This limitation motivatesFederated RAG(FedRAG), where a
central LLM server collaborates with multiple distributed silos,
each maintaining its own local corpus and serving as a knowl-
edge provider without sharing its raw documents,i.e., thelocality
constraint(see Fig. 1). In rare-disease diagnosis, for instance, a med-
ical LLM could generate well-informed suggestions by leveraging
clinical narratives (e.g.pathology reports, triage notes, genomic
records) from multiple hospitals. No single hospital offers suffi-
cient coverage for reliable decisions [ 3], and regulations such as
HIPAA [ 8] and GDPR [ 33] prohibit centralizing sensitive records
[26]. FedRAG allows the server to integrate knowledge across hos-
pitals without exposing raw documents, supporting robust diag-
nosis while ensuring compliance. More broadly, it extends RAG
to distributed, privacy-conscious environments that increasingly
characterize modern web-of-silos information systems.arXiv:2602.05235v1  [cs.CL]  5 Feb 2026

Conference’17, July 2017, Washington, DC, USA Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, and Yongxin Tong
… Silos
 Silo 1
 Silo 2
 Silo 3
 Silo n
ServerKnowledge Sharing
w/o Raw Documents
LLM
Locality Constraint
Figure 1: Federated RAG with locality constraint.
Despite rapid progress in RAG, existing methods cannot be di-
rectly extended to the federated setting because they violate the
locality constraint. Conventional RAG depends onin-contextin-
tegration, where retrieved passages are inserted into the LLM’s
prompt [ 10,14,19,29,43]. In a federated environment, this re-
quires the server to fetch verbatim documents from silos, which
inherently violates locality. Other efforts pursue privacy-preserving
prompt engineering by injecting noise into raw documents, but our
empirical study shows that these approaches suffer severe accuracy
degradation in FedRAG settings (see Sec. 4.3.3). Noise injected inde-
pendently by silos accumulates at the server, leading to performance
even worse than single-silo retrieval.
Recent advances in parametric RAG [ 30] offer a promising al-
ternative. Instead of raw text, documents are encoded into light-
weight adapters (e.g.LoRA) that can be merged with the frozen base
LLM model at inference. It avoids raw-text exchange and naturally
satisfies the locality constraint, making it an attractive candidate
for FedRAG. In practice, each silo can encode its local documents
into adapters and upload relevant ones to the server for adapter
aggregation and response generation. Yet directly extending para-
metric RAG to federated environments is far from trivial due to
two challenges.(i) Storage and communication overhead. Training
one adapter per document leads to an explosion in adapter count,
imposing unsustainable storage demands at silos and heavy commu-
nication costs. A natural remedy is to encodemultipledocuments
per adapter, but naive grouping of heterogeneous documents causes
intra-silo adapter interference.(ii) Destructive adapter aggregation:
As in federated learning, the standard adapter aggregation strategy
is to average adapters acrossallsilos. While this initially broadens
knowledge coverage,indiscriminateaggregation causesinter-silo
adapter interference. Irrelevant adapters inject noise, and parameter
averaging creates conflicts.
To address these challenges, we propose FedMosaic, an effi-
cient and accurate federated RAG framework built upon parametric
adapters. FedMosaic enforces the locality constraint while reducing
overhead and mitigating interference through two innovations.(i)
Multi-Document Parametric Adapters. It clusterssemantically related
documents into a single adapter while learningdocument-specific
masksto gate parameters. This preserves per-document specificity,
prevents conflicting signals within an adapter, and substantially
reduces storage and communication costs.(ii) Selective Adapter Ag-
gregation. It ensures that only relevant and non-conflicting adapters
are combined across silos. For each query, silos re-rank their localAlgorithm 1:Naive Federated Parametric RAG
Input:user query𝑞, LLM serverG, data silos{S 𝑚}𝑀
𝑚=1
Output:answer𝑎
1// Offline Stage
2foreach siloS 𝑚do
3Train an independent LoRA adapter for each document
4// Online Stage
5foreach siloS 𝑚do
6Retrieve top-𝑘documents based on𝑞and upload the
corresponding averaged adapters to the server
7Server aggregates multiple adapters by parameter
summation and generates answer𝑎
8return𝑎
documents and share only relevance scores and masks. The server
then selects globally relevant adapters with minimal parameter
conflict and aggregates them to generate responses. Analogous
to arranging tiles into a coherent mosaic, FedMosaic composes
only the most relevant knowledge across silos, achieving locality-
preserving, relevance-aware, and efficient federated RAG.
Our main contributions are summarized as follows.
•We introduce the first framework for federated RAG that
enforces the locality constraint, enabling multi-silo retrieval
without transmitting raw documents.
•We propose FedMosaic, a unified design that tackles the
unique efficiency and accuracy challenges in federated para-
metric RAG. It introduces multi-document parametric adapters
with document-specific masks to reduce storage and com-
munication overhead while preserving per-document speci-
ficity, and selective adapter aggregation to mitigate inter-
silo conflicts by aggregating only relevant, non-overlapping
adapters.
•Extensive evaluations on four datasets show that FedMosaic
achieves an average10 .9%higher accuracy than state-of-
the-art methods in four categories: local RAG [ 29,38,43], in-
context FedRAG [ 1,13,28,49], federated fine-tuning [ 37,48],
and parametric RAG [ 30], while enforcing locality constraint
and yielding a reduction of78 .8%to86.3%in storage and
91.4%in communication cost.
2 Preliminaries
2.1 Federated RAG
Problem Setting.Consider a federation with a central LLM server
Gand𝑀data silos{S1,...,S𝑀}(e.g.distinct organizations or geo-
distributed data centers). Each silo S𝑚maintains a local knowledge
corpusD𝑚={𝑑1
𝑚,...,𝑑𝑁𝑚𝑚}, where each 𝑑𝑖
𝑚is a document. The
silos act as distributed external knowledge providers, while the
server is responsible for generating responses.
Given a user query 𝑞, federated retrieval-augmented generation
aims to produce an answer 𝑎using knowledge across {D𝑚}𝑀
𝑚=1
under thelocality constraint: no plaintext document 𝑑𝑖
𝑚may leave
its siloS𝑚. The objective is to maximize answeraccuracywhile
keepingcommunicationoverhead low.

FedMosaic : Federated Retrieval-Augmented Generation via Parametric Adapters Conference’17, July 2017, Washington, DC, USA
Limitations of Prior Arts.Existing federated RAG frameworks
[1,13,28,35,49] rely onin-contextintegration, where the server
fetchesverbatimdocuments from silos and inserts them into the
LLM context, which violates thelocality constraintby nature.
2.2 Parametric Adapters
Parametric RAG [ 30] offers a promising alternative by integrating
external knowledgewithoutin-context documents. It converts doc-
uments intoparametric adaptersthat directly augment the LLM.
The pipeline consists of two steps.
•Document Augmentation.For each document 𝑑𝑖, an LLM gen-
erates𝑛rewrites and 𝑚question-answer (QA) pairs, forming
an augmented set
𝐷𝑖={(𝑑𝑖,𝑘,𝑞𝑖,𝑗,𝑎𝑖,𝑗)|1≤𝑘≤𝑛,1≤𝑗≤𝑚},(1)
where𝑑𝑖,𝑘is a rewrite and(𝑞𝑖,𝑗,𝑎𝑖,𝑗)is a QA pair.
•Parametric Encoding.Using LoRA adapters [ 16], each𝐷𝑖is
encoded into a parametric form by minimizing the next-
token loss:
min
Δ𝜃∑︁
𝑥∈𝐷𝑖𝑇∑︁
𝑡=1−log𝑃𝜃+Δ𝜃(𝑥𝑡|𝑥<𝑡),(2)
where𝜃is the frozen LLM parameters, Δ𝜃is the trainable
LoRA parameters, 𝑥𝑡is the𝑡-th token in sequence 𝑥, and𝑇
is the sequence length.
At inference, relevant adapters are composed with 𝜃rather than
sending raw text. Thisparametricintegration naturally enforces
locality, making it an attractive candidate for federated RAG. Al-
gorithm 1 illustrates the straightforward extension of parametric
RAG to federated scenarios.
2.3 Challenges
Although parametric RAG is attractive, directly extending it to the
federated setting introduces two challenges.
•Storage and Communication Overhead. Training one
adapterper documentcauses an explosion in the number
of adapters, leading to excessive storage requirements at
silos and high silo-server communication costs. A natural
remedy is to encodemultiple documentsinto a single adapter.
Yet experiments show that naive grouping of heterogeneous
documents severely degrades accuracy. As shown in Fig. 2a,
when the number of documents per adapter increases from
1to20, the F1 score of parametric RAG at a single silo drops
sharply1. The degradation persists across training epochs,
indicating that heterogeneous documents within a shared
adapter causeintra-silo adapter interference.
•Destructive Adapter Aggregation. In federated RAG, a
straightforward strategy is to aggregate adapters fromall
silos by averaging their parameters. While this can initially
improve accuracy by incorporatingdiverseknowledge,indis-
criminateaggregation ultimately harms accuracy. As shown
in Fig. 2b, the F1 score improves when a small number of
relevant adapters are averaged, but declines as more are
1Results are measured on 2WikiMultihopQA Bridge dataset. We randomly group a
fixed number of documents to each adapter at different epochs. At inference time, the
response is generated using the adapter most relevant to the query.
1235 10 20
Number of Documents per LoRA Adapter0.380.400.430.450.480.500.530.55F1 Score
Epoch=1
Epoch=5
Epoch=10
Epoch=20(a) Grouped Documents
135810 15 20 30
Number of LoRA Adapter Aggregations0.520.540.560.580.600.62F1 Score
0.5170.5770.5910.628
0.595
0.563 0.563
0.553Parametric RAG (b) Multiple LoRA Aggregation
Figure 2: Accuracy curves of parametric RAG for (a) grouped
documents under a LoRA adapter across different training
epochs, and (b) aggregation of multiple LoRA adapters.
included2. This degradation arises frominter-silo adapter
interference, caused by two factors:(i)not all silo documents
are equally relevant to a given query, so aggregating too
many introduces noise, and(ii)parameter averaging can
create parameter conflicts among adapters [ 17,21,23,44].
Together, these effects lead to destructive aggregation and
reduced answer quality.
Objectives.These challenges motivate the design of a federated
RAG framework that:(i)preserves locality by exchanging only
parametric adapters,(ii)reduces storage and communication costs
by supporting multi-document adapters without intra-silo interfer-
ence, and(iii)mitigates destructive aggregation through selective
adapter combination that avoids inter-silo interference. These ob-
jectives underpin the design of FedMosaic.
3 Method
3.1 FedMosaic Overview
We present FedMosaic, an efficient and accurate federated RAG
framework built upon parametric adapters (see Fig. 3). FedMosaic
addresses the challenges in Sec. 2.3 by(i)encoding multiple seman-
tically related documents into shared adapters, and(ii)selectively
aggregating relevant, non-conflicting adapter parameters across
silos. Analogous to a mosaic that arranges relevant tiles into a coher-
ent image, FedMosaic groups documents into parametric adapters
and assembles only the most relevant ones across silos to support
efficient and accurate federated RAG. To our knowledge, FedMo-
saic is the first federated RAG scheme that ensures thelocality
constraint.
Architecture.FedMosaic consists of two functional modules.
•Multi-Document Parametric Adapters(Sec. 3.2). To re-
duce storage and communication overhead, FedMosaic shares
one adapter across a cluster of semantically coherent docu-
ments. This is feasible because a single document primarily
activates a small subset of adapter parameters. Accordingly,
FedMosaic(i)groups semantically similar documents and
trains one cluster-level adapter per cluster, and(ii)learns
2Results are measured on 2WikiMultihopQA Comparison dataset. Each document
has an independent adapter trained for3epochs. At inference time, we increase the
number of averaged adapters for the same query from1to30.

Conference’17, July 2017, Washington, DC, USA Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, and Yongxin Tong
Multi -Document Parametric Adapters
Clustered Adapter Assignment Masked Adapter Sharing
Documents
Balanced
Similarity
ClusteringCluster
Training
Cluster
Training
Clusters LoRA  Adapters
 Document -Specific Mask
Mask Training
Document -Parametric Representation Pairs
, ,
 ,
 , …
Offline Stage Selective Adapter Aggregation Online Stage
Local Document Re -Ranking
…
Local
Retrieval
Local
Retrieval
Silos
 Top-k DocumentsLocal
Re-Ranking
Relevance  ScoresWith Masks
Server
Document Selection and Adapter Aggregation
Re-Rank Scores
…
Document Masks0.8 0.2
…
Candidates0.8 0.2
0.3 0.9
Select
Selected CandidatesWeighted
Aggregation
Merged Adapter
 LLM
Answer Generation
Figure 3: FedMosaic architecture and workflow.
document-specific masks that gate adapter parameters dur-
ing adapter aggregation and answer generation. This design
preserves per-document specificity and mitigates intra-silo
adapter interference during adapter sharing.
•Selective Adapter Aggregation(Sec. 3.3). To mitigate inter-
silo adapter interference during adapter averaging, FedMo-
saic aggregates only adapters associated with the most rel-
evant documents and least conflicting parameters. This is
enabled by the document-specific masks from the multi-
document adapters. For each query, silos re-rank their local
documents and upload only the relevance scores and corre-
sponding masks. The server then selects the globally most
relevant masks while minimizing parameter conflicts, mea-
sured by overlap among mask positions. The corresponding
adapters are requested, gated by the selected masks, and
averaged using standard parameter aggregation. This yields
a relevance- and conflict-aware aggregation that preserves
locality and improves answer quality.
Workflow.We assume each silo has the same base LLM as the
server and employs a homogeneous re-ranking model M𝑟. These
assumptions are standard in cross-silo federated learning, where
participants typically have sufficient resources to fine-tune large
models locally. Under this setting, FedMosaic operates in two stages.
•Offline Stage. Each silo clusters its local documents, trains
a cluster-level adapter for each cluster, and learns a binary
mask for every document. The resulting adapters and masks
are stored locally as the parametric representation of the
silo’s corpus.
•Online Stage. When a query arrives, the server broadcasts
it to all silos. Each silo performs local retrieval, applies the re-
ranking model to compute relevance scores, and uploads the
scores together with the masks of the retrieved documents.
The server then selects top-ranked documents, requests their
adapters from silos, aggregates the masked adapters, and
composes the result with the base LLM to generate the final
answer.
Algorithm 2 illustrates the overall workflow. In theofflinestage,
each silo first performs balanced clustering on its local document
corpus and trains an adapter for each cluster (line 4-5), then train aAlgorithm 2:FedMosaic
Input:user query𝑞, LLM serverG, data silos{S 𝑚}𝑀
𝑚=1
Output:answer𝑎
1// Offline Stage
2foreach siloS 𝑚do
3// Clustered Adapter Assignment
4Obtain clusters
𝐶1
𝑚,𝐶2
𝑚,...,𝐶𝑡
𝑚	
via Eq. (3)
5Train adapters{(𝐴𝑖
𝑚,𝐵𝑖
𝑚)}𝑡
𝑖=1per cluster via Eq. (2)
6// Masked Adapter Sharing
7Train mask per document via Eq. (7)
8// Online Stage
9// Local Document Re-Ranking
10foreach siloS 𝑚do
11Retrieve top-𝑘documents{𝑑𝑟1𝑚,···,𝑑𝑟𝑘𝑚}based on𝑞
12Compute relevance scores{𝑠𝑟1𝑚,···,𝑠𝑟𝑘𝑚}viaM𝑟
13Upload the relevance scores and the corresponding
document masks{(𝑠𝑟𝑖𝑚,𝑀𝑟𝑖𝑚)}𝑘
𝑖=1to the server
14// Conflict-Aware Document Selection
15Server selects𝑘′documents via Eq. (11) and notifies silos
16Silos upload corresponding LoRA adpters{(𝐵 𝑖,𝐴𝑖)}𝑘′
𝑖=1
17// Masked Adapter Aggregation
18Server aggregates LoRA adaptersΔ𝑊 merge via Eq. (13)
19LLM generates answer𝑎←G(𝑞,Δ𝑊 merge)
20return𝑎
mask for each document (lines 7). This stage is executed only once.
In theonlinestage, the server broadcasts the user query to all silos.
Each silo retrieves top- 𝑘documents and computes their relevance
scores (lines 11-13). The server then selects documents based on the
relevance scores and corresponding masks (lines 15-16), aggregates
the associated LoRA adapters weighted by the scores, and generates
the final answer𝑎(line 18-19).
3.2 Multi-Document Parametric Adapters
To reduce silo-side storage and silo–server communication while
avoidingintra-silo adapter interference, FedMosaic replaces per-
document adapters withmulti-documentadapters. The design has

FedMosaic : Federated Retrieval-Augmented Generation via Parametric Adapters Conference’17, July 2017, Washington, DC, USA
two components:(i) clustered adapter assignment, which groups
semantically similar documents and trains one adapter per cluster
(Sec. 3.2.1); and(ii) masked adapter sharing, which learns document-
specific binary masks to activate distinct parameter subsets within
a shared adapter (Sec. 3.2.2). To our knowledge, this is the first
multi-documentparametric RAG scheme that exploits the sparsity
of LoRA adapters.
3.2.1 Clustered Adapter Assignment.This module partitions a silo’s
corpus into balanced clusters of semantically related documents
and trains one adapter per cluster.
Balanced Document Clustering.Each document is embedded
into a vector representation, and semantic similarity is measured
in the embedding space usinge.g.Euclidean distance.
To controlintra-silo adapter interference, we restrict the maxi-
mum number of documents assigned to each cluster. This is because
grouping semantically related documents improves adapter sharing,
but overly large clusters reintroduce interference.
In FedMosaic, we adopt the constrained 𝑘-means clustering al-
gorithm [ 4] with an empirically chosen maximum cluster size (5to
10, see Sec. 4.4.1). Consequently, each silo S𝑚obtains a balanced
partition of its local corpus
𝐶𝑚={𝐶1
𝑚,𝐶2
𝑚,...,𝐶𝑡
𝑚},(3)
where𝑡is the number of clusters, and 𝐶𝑖
𝑚represents the set of
documents assigned to the𝑖-th cluster.
Cluster-Level Parametric Adapters.After clustering, adapters
are trained at the level of clusters rather than individual documents
as in Sec. 2.2. For each cluster 𝐶𝑖
𝑚in siloS𝑚, we construct an aug-
mented document set 𝐷𝑖
𝑚following Eq. (1). This set is encoded
into a LoRA adapter (𝐴𝑖
𝑚,𝐵𝑖
𝑚)via Eq. (2), where 𝐵𝑖
𝑚∈R𝑑×𝑟and
𝐴𝑖
𝑚∈R𝑟×𝑑define the low-rank update Δ𝑊=𝐵𝑖
𝑚𝐴𝑖
𝑚, with𝑑repre-
senting the LLM layer size and 𝑟≪𝑑 the adapter rank. Thus, each
siloS𝑚obtains𝑡cluster-level adapters {(𝐴1
𝑚,𝐵1
𝑚),...,(𝐴𝑡
𝑚,𝐵𝑡
𝑚)},
reducing the number of adapters compared to per-document train-
ing while retaining semantic coherence within clusters.
3.2.2 Masked Adapter Sharing.This module aims to learndocument-
specific binary masksthat activate distinct subsets of a shared
adapter, thereby reducingintra-silo adapter interference. The de-
sign is motivated by two observations:(i)LoRA adapters contain
redundancy [ 40], and(ii)fine-tuning typically affects only subsets
of model parameters, which differ across tasks [ 11,44]. We hy-
pothesize that document-specific knowledge can also be captured
by distinct subsets of LoRA parameters. Accordingly, FedMosaic
learns binary masks over afrozencluster-level adapter so that each
document activates a tailored subspace. These masks also assist in
selective adapter aggregation (Sec. 3.3).
Adapter Mask Designation.Let (𝐴𝑖
𝑚,𝐵𝑖
𝑚)be the LoRA adapter
for cluster𝐶𝑖
𝑚, which encodes documents {𝑑𝑖1𝑚,...,𝑑𝑖𝑛𝑚}. Applying
a binary mask directly to the full update Δ𝑊=𝐵𝑖
𝑚𝐴𝑖
𝑚∈R𝑑×𝑑
would require 𝑂(𝑑2)mask parameters, which is both storage- and
computation-intensive. Instead, we adopt a lightweightrow-wise
masking scheme on𝐵𝑖
𝑚∈R𝑑×𝑟, requiring only𝑂(𝑑)parameters.
For document 𝑑𝑖𝑗
𝑚, we define a binary vector 𝑀𝑖𝑗
𝑚∈{0,1}𝑑and
apply it row-wise to𝐵𝑖
𝑚:
˜𝐵𝑖𝑗
𝑚=𝑀𝑖𝑗
𝑚◦𝐵𝑖
𝑚,(4)where◦denotes the Hadamard product with broadcasting across
columns. To stabilize the magnitude of masked updates, we intro-
duce a rescale factor:
𝜆𝑖𝑗
𝑚=𝑑
∥𝑀𝑖𝑗
𝑚∥1,(5)
and define the masked low-rank update as
Δ˜𝑊𝑖𝑗
𝑚=𝜆𝑖𝑗
𝑚·˜𝐵𝑖𝑗
𝑚𝐴𝑖
𝑚.(6)
This ensures stable masked updating while allowing each document
to activate only a subset of adapter rows.
Adapter Mask Training.During training, the adapter parame-
ters(𝐴𝑖
𝑚,𝐵𝑖
𝑚)remain frozen and only the masks are optimized.
This preserves shared cluster-level knowledge while enabling each
document to learn a sparse, specialized activation pattern.
To enable gradient-based optimization, the binary mask is re-
laxed via a sigmoid parameterization 𝜎. Let ˆ𝑀𝑖𝑗
𝑚∈R𝑑be trainable
logits. During training we use 𝜎(𝛼 ˆ𝑀𝑖𝑗
𝑚)with sharpening factor
𝛼>0, and imposeℓ 1sparsity, The total loss function is defined as
min
ˆ𝑀𝑖𝑗
𝑚L(ˆ𝑀𝑖𝑗
𝑚;𝜽𝐹)=L next(ˆ𝑀𝑖𝑗
𝑚;𝜽𝐹)+𝜆ℓ1·𝜎
𝛼·ˆ𝑀𝑖𝑗
𝑚
1,(7)
where 𝜽𝐹is the frozen base and adapter parameters, Lnextis the
next-token loss, and𝜆 ℓ1balances task accuracy and mask sparsity.
After training, the masks are binarized via thresholding:
𝑀𝑖𝑗
𝑚=Ih
ˆ𝑀𝑖𝑗
𝑚>0i
,(8)
whereI[·]is the indicator function.
This procedure is applied independently for each document in
the cluster, producing a set of sparse, document-specific masksn
𝑀𝑖𝑗
𝑚o𝑛
𝑗=1that(i)mitigate intra-adapter interference by isolating
document-specific subspaces, and(ii)provide conflict indicators
for selective aggregation across silos.
To further reduce the storage overhead introduced by the binary
mask parameters, we adopt a bit-packing scheme that encodes each
𝑑-dimensional binary mask vector 𝑀𝑖𝑗
𝑚∈{0,1}𝑑into⌈𝑑/8⌉bytes
by packing 8 bits per byte.
b𝑖=8∑︁
𝑗=1𝑚8(𝑖−1)+𝑗·2𝑗−1,for𝑖=1,...,𝑑
8
,(9)
whereb𝑖denotes the 𝑖-th packed byte and 𝑚8(𝑖−1)+𝑗 denotes the
8(𝑖−1)+𝑗-th bit in𝑀𝑖𝑗
𝑚. This reduces storage by 8 ×over naive byte-
aligned uint8 representations and supports encoding and decoding
in𝑂(𝑑)time via simple linear scans of the mask entries.
3.3 Selective Adapter Aggregation
As discussed in Sec. 2.3, naive aggregation of adapters from all silos
introducesinter-silo adapter interference, since irrelevant documents
introduce noise and overlapping parameters cause conflicts. To
overcome this, FedMosaic introducesselective adapter aggregation,
which aggregates only query-relevant documents while suppress-
ing parameter conflicts. The process consists of three steps:(i) local
document re-ranking, each silo re-ranks its local documents for the
query and uploads only their relevance scores and masks (Sec. 3.3.1);
(ii) conflict-aware document selection, the server selects a globally
relevant subset under a conflict-aware criterion (Sec. 3.3.2); and(iii)
masked adapter aggregation, the selected adapters are aggregated

Conference’17, July 2017, Washington, DC, USA Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, and Yongxin Tong
Table 1: Overall accuracy measured by F1 score for FedMosaic and four types of baseline methods.
Type MethodHotpotQA 2WikiMultihopQA PopQA CWQ
Bridge Compare Bridge Compare Inf. Compose Total Total
LocalStandard RAG 0.1474 0.3602 0.2838 0.2837 0.1549 0.0748 0.1366 0.2804
CoTRAG [38] 0.0808 0.2632 0.3821 0.3444 0.1380 0.0377 0.0444 0.2378
React [43] 0.1731 0.3103 0.3108 0.2608 0.1528 0.0590 0.1135 0.2443
Dargin [29] 0.1257 0.3540 0.3271 0.3786 0.1473 0.0766 0.0895 0.3662
FedRAGFRAG [1, 49] 0.1317 0.3390 0.3357 0.2964 0.1527 0.0265 0.1207 0.1946
MKPQA [28] 0.1562 0.2757 0.2831 0.2679 0.2112 0.0510 0.1329 0.2008
RAGRoute [13] 0.1137 0.2825 0.2239 0.2472 0.1351 0.0293 0.0380 0.1767
FedFTFedIT [48] 0.1107 0.4188 0.4250 0.4377 0.1996 0.0692 0.2152 0.3687
FLoRA [37] 0.1030 0.4259 0.3625 0.4038 0.1864 0.0770 0.1917 0.3497
Param. PRAG [30] 0.0983 0.4161 0.3875 0.4805 0.2138 0.0462 0.0759 0.2810
Ours FedMosaic0.1980 0.4547 0.4453 0.5275 0.2474 0.0940 0.2368 0.3841
in a mask-gated, relevance-weighted manner (Sec. 3.3.3). These
designs address the unique challenges on accuracy when adapting
parametric RAG to federated scenarios.
3.3.1 Local Document Re-Ranking.Upon receiving a query, the
server broadcasts it to all silos. Each silo S𝑚retrieves𝑘candi-
date documents{𝑑𝑟1𝑚,...,𝑑𝑟𝑘𝑚}and applies a re-ranking model M𝑟,
which is the same across silos, to assign normalized relevance
scores{𝑠𝑟1𝑚,...,𝑠𝑟𝑘𝑚}. For each candidate, the silo collects its associ-
ated mask𝑀𝑟𝑖𝑚over the cluster-level adapter and uploads only the
pairs{(𝑠𝑟𝑖𝑚,𝑀𝑟𝑖𝑚)}𝑘
𝑖=1, while keeping the raw documents and LoRA
adapters local.
3.3.2 Conflict-Aware Document Selection.The server collects the
relevance scores and masks of all 𝑘𝑀candidates and selects a
subset𝑆of𝑘′documents that maximize relevance while minimizing
parameter conflicts among their masks, given that relevance-based
re-ranking alone is insufficient for RAG [ 18]. We measure conflicts
between two masks𝑀 𝑖and𝑀𝑗as their overlap:
overlap(𝑀 𝑖,𝑀𝑗)=⟨𝑀𝑖,𝑀𝑗⟩
𝑑,(10)
where𝑑is the mask dimension.
The selection objective is formulated as
max
𝑆⊆[𝑛],|𝑆|=𝑘′ ∑︁
𝑖∈𝑆𝑠𝑖
𝑘′−2𝜆 ol·Í
𝑖,𝑗∈𝑆,𝑖<𝑗 overlap(𝑀 𝑖,𝑀𝑗)
𝑘′·(𝑘′−1)!
,(11)
where[𝑛]is the index set of all candidate documents, 𝑠𝑖is the
relevance score of document 𝑖,𝑀𝑖is its mask, and 𝜆ol>0balances
relevance against conflicts. We prove the above selection problem
is NP-hard in Appendix.
We solve Eq. (11) via a greedy algorithm. At each iteration, we
select the candidate with the highest marginal gain
Δ𝑖=𝑠𝑖−𝜆ol·1
|𝑆|∑︁
𝑗∈𝑆overlap(𝑀 𝑖,𝑀𝑗),(12)
until𝑘′documents are chosen. To avoid noisy selections, a score
threshold𝜏is applied. Only candidates with𝑠 𝑖>𝜏are considered,
and the procedure terminates early if none remain.Table 2: Privacy evaluation over target and prefix attack.
Attack MethodTarget
PromptsTarget
Info.Repeat
PromptsRepeat
Length
TargetIC-FedRAG 138.93 102.20 131.33 21.34
FedMosaic0 0 0 0
PrefixIC-FedRAG 52,86 46.66 55.93 23.58
FedMosaic44.00 29.53 22.86 22.22
3.3.3 Masked Adapter Aggregation.After selection, the server re-
quests only the necessary information from silos. For each chosen
document𝑖∈𝑆 , this includes its mask 𝑀𝑖and the associated clus-
ter adapter(𝐵𝑖,𝐴𝑖). Let𝑤𝑖=𝑠𝑖/Í
𝑗∈𝑆𝑠𝑗be normalized relevance
weights. The aggregated LoRA adapter is then computed as
Δ𝑊 merge=∑︁
𝑖∈𝑆𝑤𝑖(𝑀𝑖◦𝐵𝑖)𝐴𝑖,(13)
which is combined with the base LLM Gto generate the final an-
swer. This aggregation strategy ensures query relevance, suppresses
parameter conflicts, and maintains locality of raw documents.
4 Experiments
4.1 Experimental Setup
Baselines.We compare FedMosaic with four categories of meth-
ods: local RAG (Standard RAG, CoTRAG [ 38], ReAct [ 43], Dargin
[29]), in-context FedRAG (FRAG [ 1,49], MKPQA [ 28], RAGRoute
[13]), federated fine-tuning (FedIT [ 48], FLora [ 37]), and parame-
terized RAG (PRAG [ 30]). All baselines are adapted to the federated
setting for fair comparison. Methods based on privacy-preserving
prompt engineering are not included as baselines due to their severe
accuracy degradation (see Sec. 4.3.3).
Datasets and Models.We experiment on HotpotQA (HQA) [ 42],
2WikiMultihopQA (2WQA) [ 15], PopQA (PQA) [ 22], and Com-
plexWebQuestions (CWQ) [ 31] with different question types (e.g.
Bridge).

FedMosaic : Federated Retrieval-Augmented Generation via Parametric Adapters Conference’17, July 2017, Washington, DC, USA
Table 3: LLaMA3-8B compared with competitive baselines.
Type Method2WikiMultiHopQA
Bridge Compare Inf. Compose
Local ReAct 0.3432 0.4022 0.2093 0.0746
FedRAG MKPQA 0.2238 0.4011 0.2797 0.0932
FedFT FedIT 0.4995 0.6105 0.2677 0.1311
FedFT FLoRA 0.4701 0.5714 0.2355 0.1138
Param. PRAG 0.1812 0.2527 0.0829 0.0271
Ours FedMosaic0.5207 0.6237 0.2859 0.1584
By default, each dataset is subsampled to 300 Q&A instances
and a document corpus is constructed following [ 30]. This corpus
is then partitioned into silos based on underlying topics using a
Dirichlet-based allocation strategy with𝛼=0.1.
We also conduct privacy evaluations on Enron Emails and Wiki-
Text datasets following [ 46,47]. LLaMA3.2-1B-Instruct and LLaMA3-
8B-Instruct are used as backbone LLMs.
Metrics.We report four metrics: (1)Accuracy: correctness of server
LLM responses to user queries measured by F1 score; (2)Privacy:
average success protection rate against targeted attack [ 46] and pre-
fix attack [ 5] at server-side inference; (3)Communication Efficiency:
average parameters transmitted per query from silo to server; (4)
Storage Overhead: extra silo-side storage for parametric adapters.
Experiment.We conduct our experiments on a machine equipped
with an Intel Xeon Gold 6230R CPU and four NVIDIA A100 GPUs,
each providing 40 GB of memory for computation.
4.2 Main Results
Table. 1 reports the overall F1 score of FedMosaic and baselines on
four datasets with the LLaMA3.2-1b-instruct backbone. Local RAG
methods show unstable performance, since each silo can only rely
on its own documents without cross-silo knowledge. In-context
FedRAG baselines attempt to integrate cross-silo information but
violate the locality constraint and often introduce noisy documents,
limiting their gains. FedIT and FLoRA achieve relatively high scores
on some subsets (e.g.0 .4250on 2WQA Bridge) and exhibit more
stable performance overall, but require10 ×higher training budgets.
Naive federated parametric RAG performs well on certain subsets
(e.g.0.4805on 2WQA Compare), but suffers severe drops on others
(e.g.0.2810on CWQ), consistent with the challenges in Sec. 2.3.
In contrast, FedMosaic consistently achieves state-of-the-art per-
formance across all datasets, demonstrating its effectiveness in
mitigating both intra- and inter-silo adapter interference. Specifi-
cally, compared to the strongest baseline, FedMosaic improves the
average F1 scores on HQA, 2WQA, and PQA by10 .57%,13.08%,
and10.03%, respectively. These results highlight the robustness of
FedMosaic and its ability to leverage distributed knowledge while
preserving locality constraints.
4.3 Micro-Benchmarks
4.3.1 Privacy Evaluation.The experiment justifies the necessity
of the locality constraint for FedRAG, and further quantifies pri-
vacy benefits of FedMosaic under the locality constraint versus
c=5 c=8 c=10 w/o clust.
Clustering Capacity0.00.51.01.52.02.53.03.5Storage per Document (MB)FedMosaic Mask
FedMosaic LoRA
FedMosaic w/o clust.(a) Per-Document Storage
k=1 k=3 k=5 k=10
T op-k Documents Retrieved by Each Client050100150200Communication per Query (MB)FedMosaic
FedMosaic w/o clust. (b) Per-Query Communication
Figure 4: Storage and communication overhead.
in-context FedRAG. We consider two data extraction attacks:tar-
getedandprefix[ 5,46]. We construct300privacy-sensitive samples
following [ 47], and then apply retrieval-data attacks to in-context
FedRAG and training-data attacks to FedMosaic. The in-context
FedRAG (i.e., IC-FedRAG) adopts a minimal FRAG [ 1,49] design,
capturing key characteristics of MKPQA [28] and RAGRoute [13].
Table. 2 reports the attack success rates (lower is better) of Fed-
Mosaic and in-context FedRAG, showing that FedMosaic provides
stronger resistance to data extraction attacks. Under targeted at-
tacks, FedMosaic is almost immune (0%across all metrics) since
the absence of explicit context makes it difficult to trigger adapter-
encoded knowledge, whereas in-context FedRAG prompts are more
easily repeated by the LLM. Under prefix attacks, FedMosaic also
outperforms in-context FedRAG (e.g.a59%reduction in repeated
prompts and a36 .7%reduction in target information), benefiting
from training on rewritten data (see Eq. (1)), where sensitive content
can be filtered using simple prompts (e.g.Ensure the revision has no
emails). Furthermore, successful attacks require knowledge of the
actual training data, which further increases attack difficulty and
demonstrates the higher degree of privacy achieved by FedMosaic.
4.3.2 Performance with Larger Backbones.To evaluate the scalabil-
ity of FedMosaic, we extend the backbone to LLaMA3-8B-Instruct.
For a comprehensive comparison, we select the most competitive
baselines reported in Sec. 4.2 and conduct experiments on the 2WQA
dataset, as it contains the most diverse set of Q&A types.
Table. 3 reports the F1 scores of FedMosaic and the competitive
baselines. We observe that PRAG exhibits notable performance
degradation due to accumulated noise and parameter conflicts. In
contrast, FedMosaic consistently maintains state-of-the-art perfor-
mance, demonstrating strong scalability to larger model sizes. For
example, on the Bridge and Compose datasets, FedMosaic outper-
forms the strongest baselines by4.2%and20.8%, respectively.
4.3.3 Compared to Privacy-Preserving Prompt Engineering Meth-
ods.Privacy-preserving prompt engineering seeks to protect sen-
sitive information by introducing noise through anonymization
or differential privacy. Representative approaches include differ-
ential privacy-based methods (DP-Prompt [ 34], AUGPR [ 39]) and
anonymization-based methods (Sage [47]).
Table. 4 reports the F1 scores of these methods, together with
their performance drops relative to FedMosaic. All methods show
severe accuracy degradation, especially differential privacy ap-
proaches (e.g.AUGPR), which perturb the global distribution by
introducing noise that obscures document-specific information,

Conference’17, July 2017, Washington, DC, USA Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, and Yongxin Tong
Table 4: Performance of privacy-preserving prompt engineering methods.
MethodHotpotQA 2WikiMultihopQA PopQA CWQ
Bridge Compare Bridge Compare Inf. Compose Total Total
DP_Prompt0.0228 ↓88.5% 0.2179↓52.1% 0.0785↓82.4% 0.1305↓75.3% 0.0385↓84.4% 0.0114↓87.9% 0.0060↓97.5% 0.0778↓79.7%
Sage (Attr)0.0791 ↓60.1% 0.3286↓27.7% 0.1808↓59.4% 0.2218↓58.0% 0.1318↓46.7% 0.0222↓76.4% 0.0676↓71.5% 0.1217↓68.3%
Sage (Agent)0.0923 ↓53.4% 0.3314↓27.1% 0.2898↓34.9% 0.2645↓49.9% 0.1343↓45.7% 0.0231↓75.4% 0.0560↓76.4% 0.1485↓61.3%
AUGPE0.0079 ↓96.0% 0.0514↓88.7% 0.0153↓96.6% 0.0283↓94.6% 0.0225↓90.9% 0.0032↓96.6% 0.0004↓99.8% 0.0247↓93.6%
0 10 20 30 40 50
Training Steps0.51.01.52.0Initial Next-T oken Loss
Next-T oken Loss (Left Y-Axis)
Mask Loss (Right Y-Axis)
8101214161820
(a) Training Loss
0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Sparsity Ratio0.460.470.480.490.500.510.520.53F1 ScoreFedMosaic
FedMosaic w/o Mask
Implicit Sparsity Radio (b) Mask to Accuracy
Figure 5: Impact of document mask.
and, even when applied document-wise (e.g.DP-Prompt), can dam-
age fine-grained knowledge crucial for accurate RAG. Similarly,
anonymization-based methods such as Sage also lose substantial
information, with an average53%degradation, indicating that key
evidence for Q&A is often compromised during anonymizatio. Over-
all, the poor utility of these approaches in RAG tasks makes them
unsuitable as primary baselines in our evaluation.
4.4 Ablation Studies
We conduct a thorough ablation study to isolate and verify the con-
tribution of each component in FedMosaic. Specifically, we analyze
multi-document parametric adaptersin terms of overhead reduc-
tion (Sec. 4.4.1) and mask effectiveness (Sec. 4.4.2), whileselective
adapter aggregationis evaluated for the impact of the selection 𝑘
(Sec. 4.4.3) and the retrieval𝑘(Sec. 4.4.4) on accuracy.
4.4.1 Overhead Reduction by Clustering.Fig. 4 compares the stor-
age and communication overhead of FedMosaic with the w/o clus-
tering variant. We choose 𝐶∈{ 5,8,10}and𝑘∈{ 1,3,5,10}for
silo retrieval in our experiments. Fig. 4a shows that adapter size
inversely proportional decreases with clustering (e.g.12 .67%of no-
clustering at 𝑐=8), while per-document masks stay constant at
∼1.03%. The total storage is reduced to11 .23%of the w/o cluster-
ing variant at 𝑐=10. Fig. 4b shows that per-query communication
increases linearly in the w/o clustering variant, whereas clustering
keeps it stable; at 𝑘=10, the cost is reduced to4 .86%of the variant.
4.4.2 Contributions of Document-Specific Mask.Fig. 5 demonstrates
the role of the document-specific mask in alleviatingintra-silo
adapter interferenceand its effect on model accuracy. We select
the number of mask training epochs from {3,5,10,20}with varying
learning rate. Fig. 5a shows training curves of next-token and mask
losses, where both prediction accuracy and mask sparsity improvesimultaneously, supporting the hypothesis in Sec. 3.2.2. Fig. 5b fur-
ther evaluates accuracy under different sparsity settings, where
FedMosaic consistently outperforms the w/o mask variant. Notably,
setting𝜆ℓ1=0in Eq. (7) drives mask learning solely via next-token
loss, yielding implicit sparsity and indicating that document-specific
knowledge can be captured by distinct subsets of LoRA parameters.
4.4.3 Effectiveness of Selective Aggregation.Table. 5 presents the
effectiveness of selective aggregation on the 2WQA dataset, which
is selected for its diversity, under different values of selection 𝑘.
FedMosaic outperforms its w/o selection variant as 𝑘increases (e.g.
20.7%higher in Inf.), demonstrating its ability to filter irrelevant
documents and mitigate parameter conflicts. Moreover, its perfor-
mance stabilizes with larger𝑘, highlighting strong robustness.
4.4.4 Impact of Top- 𝑘Retrieval.Fig. 6 compares FedMosaic with
the most competitive FedRAG baseline across all types of datasets
under different top- 𝑘retrieval settings. Other methods show fluc-
tuating performance, indicating sensitivity to retrieval noise. With
selective aggregation, FedMosaic achieves consistently higher and
more stable accuracy,e.g.achieving up to a10 .17%average improve-
ment on HQA Compare over the strongest baseline.
5 Related Work
Retrieval-Augmented Generation (RAG).RAG [ 12] enhances
LLMs with external knowledge, reducing hallucinations and com-
pensating for outdated parametric memory. The mainstream isin-
context RAG[ 10,14,19], and it can be improved through retrieval
optimization [ 20,24,25,41], retriever-LLM alignment [ 7,27,36,45],
and multi-round reasoning [ 9,29,43]. However, in-context RAG is
unfit for the federated settings by design, since it requires transmit-
ting verbatim documents to the server, which violates the locality
constraint.
Parametric RAG [ 30] offers an alternative by compiling knowl-
edge into model parameters rather than context tokens through
training. DyPRAG [ 32] further trains a parameter translator model
to convert documents into parametric knowledge.
By encoding knowledge in reusable parameters rather than
prompts, parametric RAG allows documents to remain local while
uploading only adapterss, making it suited to federated scenarios.
Our work is built upon parametric RAG, and focuses on the unique
efficiency and accuracy challenges when adapting it to federated
environments.
Federated Retrieval-Augmented Generation (FedRAG).Fe-
dRAG extends RAG to settings where knowledge is distributed
across silos that cannot share raw documents. Recent efforts have

FedMosaic : Federated Retrieval-Augmented Generation via Parametric Adapters Conference’17, July 2017, Washington, DC, USA
Table 5: Accuracy improvement and robustness of selective aggregation under different selection𝑘.
Dataset Type w/o Sel.𝑘=1𝑘=3𝑘=5𝑘=10𝑘=15𝑘=30
2WQABridge0.4224 0.4552 ↑7.77% 0.4447↑5.28% 0.4492↑6.34% 0.4525↑7.13% 0.4525↑7.13% 0.4525↑7.13%
Compare0.5001 0.4940 ↓1.22% 0.5108↑2.14% 0.5142↑2.82% 0.5142↑2.82% 0.5142↑2.82% 0.5142↑2.82%
Inf.0.2137 0.2316 ↑8.38% 0.2581↑20.7% 0.2455↑14.8% 0.2445↑14.4% 0.2422↑13.3% 0.2422↑13.3%
Compose0.0862 0.0750 ↓12.9% 0.0869↑0.81% 0.0953↑10.5% 0.0953↑10.5% 0.0953↑10.5% 0.0953↑10.5%
Standard RAG ReAct Dragin Mkpqa PRAG FedMosaic
123 5 8 10 12 15
T op-k Retrieval0.280.300.330.350.380.400.430.45F1 Score
(a) HQA Compare
123 5 8 10 12 15
T op-k Retrieval0.250.300.350.400.45F1 Score
 (b) 2WQA Bridge
123 5 8 10 12 15
T op-k Retrieval0.140.160.180.200.220.240.26F1 Score
 (c) 2WQA Inference
123 5 8 10 12 15
T op-k Retrieval0.030.040.050.060.070.080.09F1 Score
 (d) 2WQA Compose
123 5 8 10 12 15
T op-k Retrieval0.050.100.150.200.25F1 Score
 (e) PQA Total
123 5 8 10 12 15
T op-k Retrieval0.200.230.250.280.300.330.350.38F1 Score
 (f) CWQ Total
Figure 6: Performance of federated RAG under varying top-𝑘retrieval settings.
explored various strategies for federated search, yet rely onin-
contextRAG. For example, FeB4RAG [ 35] provides a benchmark
for federated search pipelines, MKPQA [ 28] enables probabilistic
cross-domain search, and RAGRoute [ 13] improves efficiency via
query routing. Other variants target aggregation or search security.
For example, C-FedRAG [ 1] prevent the leakage of raw documents
during aggregation and FRAG [ 49] leverage encrypted similarity
search. Despite these advances, existing frameworks still transmit
raw documents across silos, violating locality constraints.
Other efforts adopt privacy-preserving prompt engineering by
injecting noise into documents. For example, DP-Prompt [34] and
AUGPE [ 39] adopt differential privacy strategies. Sage [ 47] adopt
anonymization strategies to achieve data desensitization. However,
these approaches suffer severe accuracy degradation in FedRAG
settings, limiting their applicability.
Another relevant solution to FedRAG is federated fine-tuning, as
explored in FedIT [ 48] and FLora [ 37]. Compared to FedRAG, fed-
erated fine-tuning is less flexible because new documents demand
costly retraining and it cannot selectively activate relevant silos.
In contrast, our FedMosaic introduces the firstfederated paramet-
ric RAGframework. By encoding documents into local adapters and
composing them across silos at inference, FedMosaic adheres to the
locality constraint while providing high flexibility. Its optimizations
ensure low storage and communication overhead as well as high
accuracy, making it a practical and scalable solution for federated
knowledge-intensive generation.
6 Conclusion
In this work, we explore federated RAG where a central LLM col-
laborates with distributed knowledge silos without sharing their
raw documents. We propose FedMosaic, the first federated RAG
framework built on parametric adapters. By clustering semantically
related documents into multi-document adapters with document-
specific masks, FedMosaic drastically reduces storage and com-
munication overhead while maintaining per-document specificity.
Furthermore, its selective adapter aggregation mechanism ensuresthat only relevance-aligned, non-conflicting adapters are combined,
mitigating destructive aggregation across silos. Extensive experi-
ments on four datasets show that FedMosaic consistently outper-
forms state-of-the-art methods across four categories including
local RAG, in-context FedRAG, federated fine-tuning, and para-
metric RAG, while enforcing locality constraints and significantly
reducing storage and communication overhead. We envision Fed-
Mosaic as a foundation for scalable, privacy-preserving knowledge
integration, and an important step toward deploying RAG systems
in real-world distributed information ecosystems.
Appendix
We show that the conflict-aware document selection optimization
problem is NP-hard by through a reduction fromCLIQUEproblem
[2]. The proof is given in two steps.
First, We can formally cast the conflict-aware document selection
problem as a weighted subgraph selection problem over a complete
graph, defined as follows.
Definition 1 (Weighted Subgraph Selection Problem).Let
𝐺=(𝑉,𝐸) be the complete graph on 𝑛vertices. Each vertex 𝑣∈𝑉 is
associated with a positive vertex weight 𝑎𝑣>0, and each unordered
pair{𝑢,𝑣}⊆𝑉 is assigned an edge weight 𝑏𝑢𝑣∈R. For any subset
𝑆⊆𝑉with cardinality|𝑆|=𝑘, the total weight of𝑆is defined as
𝑊(𝑆)=∑︁
𝑣∈𝑆𝑎𝑣−∑︁
{𝑢,𝑣}⊆𝑆𝑏𝑢𝑣.(14)
The Weighted Subgraph Selection Problem asks: given a weighted com-
plete graph with vertex weights {𝑎𝑣}𝑣∈𝑉, edge weights{𝑏𝑢𝑣}{𝑢,𝑣}⊆𝑉 ,
and an integer 𝑘, find a subset 𝑆⊆𝑉 of size𝑘that maximizes 𝑊(𝑆) .
Then, we reduce from the classicalCLIQUEproblem, which is
known to be NP-complete [ 2]. Given a graph 𝐺′=(𝑉′,𝐸′)and
an integer𝑞, theCLIQUEproblem asks whether there exists a
subset𝐶⊆𝑉′with|𝐶|=𝑞 such that every pair of vertices in 𝐶is
connected by an edge in 𝐸′. From an instance (𝐺′,𝑞)we construct
an instance of the Weighted Subgraph Selection Problem as follows:
•The vertex set of the new instance is𝑉:=𝑉′.

Conference’17, July 2017, Washington, DC, USA Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, and Yongxin Tong
•For each𝑣∈𝑉, set the vertex weight𝑎 𝑣:=1.
•For each unordered pair{𝑢,𝑣}⊆𝑉, define the edge weight
𝑏𝑢𝑣=(
0,{𝑢,𝑣}∈𝐸′,
𝐵,{𝑢,𝑣}∉𝐸′,(15)
where𝐵is a sufficiently large constant (e.g.,𝐵>𝑞).
•The target subset size is set to𝑘:=𝑞.
Consider any subset 𝑆⊆𝑉 with|𝑆|=𝑞 . Its weight is 𝑊(𝑆)=
𝑞−𝐵·𝑡(𝑆) , where𝑡(𝑆) denotes the number of non-edges induced
by𝑆, i.e.,
𝑡(𝑆)={{𝑢,𝑣}⊆𝑆:{𝑢,𝑣}∉𝐸′}.(16)
By construction, if 𝑆is a clique of size 𝑞, then𝑡(𝑆)= 0and𝑊(𝑆)=𝑞 .
Otherwise, 𝑡(𝑆)≥ 1and thus𝑊(𝑆) ≤𝑞−𝐵 . Since𝐵>𝑞 , the
global maximum of 𝑊(𝑆) equals𝑞if and only if 𝐺′contains a clique
of size𝑞. Hence, deciding whether 𝐺′admits a clique of size 𝑞can
be answered by solving the weighted subgraph selection problem
on the constructed instance. This establishes a polynomial-time
reduction
CLIQUE≤ 𝑝Weighted Subgraph Selection.(17)
SinceCLIQUEis NP-complete, it follows immediately that the
weighted subgraph selection problem, and consequently the conflict-
aware document selection problem, is NP-hard.

FedMosaic : Federated Retrieval-Augmented Generation via Parametric Adapters Conference’17, July 2017, Washington, DC, USA
References
[1]Parker Addison, Minh-Tuan H Nguyen, Tomislav Medan, Jinali Shah, Moham-
mad T Manzari, Brendan McElrone, Laksh Lalwani, Aboli More, Smita Sharma,
Holger R Roth, et al .2024. C-fedrag: A confidential federated retrieval-augmented
generation system.arXiv preprint arXiv:2412.13163(2024).
[2]Immanuel M Bomze, Marco Budinich, Panos M Pardalos, and Marcello Pelillo.
1999. The maximum clique problem. InHandbook of Combinatorial Optimization:
Supplement Volume A. Springer, 1–74.
[3]Kym M Boycott and Roberto Giugliani. 2025. The RDI–Lancet Commission on
Rare Diseases: improving visibility to address health-care disparities for 400
million people.The Lancet405, 10479 (2025), 605–607.
[4]Paul S Bradley, Kristin P Bennett, and Ayhan Demiriz. 2000. Constrained k-means
clustering.Microsoft Research, Redmond20 (2000).
[5]Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian
Tramer, and Chiyuan Zhang. 2022. Quantifying memorization across neural
language models. InICLR.
[6]Zhe Chen, Yusheng Liao, Shuyang Jiang, Pingjie Wang, YiQiu Guo, Yanfeng Wang,
and Yu Wang. 2025. Towards omni-RAG: Comprehensive retrieval-augmented
generation for large language models in medical applications. InACL. 15285–
15309.
[7]Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan. 2023.
Lift yourself up: Retrieval-augmented text generation with self-memory.NeurIPS
36 (2023), 43780–43799.
[8]U.S. Congress. 1996. Health Insurance Portability and Accountability Act
of 1996. https://www.govinfo.gov/content/pkg/PLAW-104publ191/pdf/PLAW-
104publ191.pdf. Accessed: 2025-9-19.
[9]Chenxu Cui, Haihui Fan, Jinchao Zhang, Lin Shen, Bo Li, and Weiping Wang.
2025. CIRAG: Retrieval-Augmented Language Model with Collective Intelligence.
InSIGIR. 1316–1326.
[10] Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Ji-Rong Wen, and
Zhicheng Dou. 2025. Understand what LLM needs: Dual preference alignment
for retrieval-augmented generation. InWWW. 4206–4225.
[11] Guodong Du, Zitao Fang, Jing Li, Junlin Li, Runhua Jiang, Shuyang Yu, Yifei Guo,
Yangneng Chen, Sim Kuan Goh, Ho-Kin Tang, Daojing He, Honghai Liu, and
Min Zhang. 2025. Neural Parameter Search for Slimmer Fine-Tuned Models and
Better Transfer. InACL. 32668–32687.
[12] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InSIGKDD. 6491–6501.
[13] Rachid Guerraoui, Anne-Marie Kermarrec, Diana Petrescu, Rafael Pires, Mathis
Randl, and Martijn de Vos. 2025. Efficient federated search for retrieval-
augmented generation. InProceedings of the 5th Workshop on Machine Learning
and Systems. 74–81.
[14] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang.
2020. Retrieval augmented language model pre-training. InICML. 3929–3938.
[15] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning
steps. InCOLING. 6609–6625.
[16] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu
Wang, Weizhu Chen, et al .2022. LoRA: Low-rank adaptation of large language
models. InICLR.
[17] Chenyu Huang, Peng Ye, Tao Chen, Tong He, Xiangyu Yue, and Wanli Ouyang.
2024. Emr-merging: Tuning-free high-performance model merging.NeurIPS37
(2024), 122741–122769.
[18] Dahyun Lee, Yongrae Jo, Haeju Park, and Moontae Lee. 2025. Shifting from
Ranking to Set Selection for Retrieval Augmented Generation. InACL. 17606–
17619.
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .
2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.NeurIPS
33 (2020), 9459–9474.
[20] Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan,
Ruiming Tang, Yong Yu, and Weinan Zhang. 2024. Rella: Retrieval-enhanced
large language models for lifelong sequential behavior comprehension in recom-
mendation. InWWW. 3497–3508.
[21] Zhenyi Lu, Chenghao Fan, Wei Wei, Xiaoye Qu, Dangyang Chen, and Yu Cheng.
2024. Twin-merging: Dynamic integration of modular expertise in model merging.
NeurIPS37 (2024), 78905–78935.
[22] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating
effectiveness of parametric and non-parametric memories. InACL. 9802–9822.
[23] Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. 2023. Task
arithmetic in the tangent space: Improved editing of pre-trained models.NeurIPS
36 (2023), 66727–66754.
[24] Tao Ouyang, Guihang Hong, Kongyange Zhao, Zhi Zhou, Weigang Wu, Zhaobiao
Lv, and Xu Chen. 2025. AdaRAG: Adaptive Optimization for Retrieval Augmented
Generation with Multilevel Retrievers at the Edge. InINFOCOM. IEEE, 1–10.[25] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond.Foundations and Trends®in Information Retrieval
3 (2009), 333–389.
[26] Tianyi Shen, Yuxi Li, Yanlin Cao, Xin Du, Xinru Wang, Yajuan Zhang, and Yi
Zhang. 2025. Rapid deployment of large language model DeepSeek in Chinese
hospitals demands a regulatory response.Nature Medicine(2025), 1–6.
[27] Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, and Han Li.
2025. Retrieval Augmented Generation with Collaborative Filtering for Personal-
ized Text Generation. InSIGIR.
[28] Parshin Shojaee, Sai Sree Harsha, Dan Luo, Akash Maharaj, Tong Yu, and Yunyao
Li. 2025. Federated retrieval augmented generation for multi-product question
answering. InCOLING, Vol. Industry Track. 387–397.
[29] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN:
Dynamic retrieval augmented generation based on the real-time information
needs of large language models. InACL. 12991–13013.
[30] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning
Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented
generation. InSIGIR. 1240–1250.
[31] Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for
answering complex questions. InNAACL. 641–651.
[32] Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. 2025. Dynamic
parametric retrieval augmented generation for test-time knowledge enhancement.
arXiv preprint arXiv:2503.23895(2025).
[33] European Union. 2016. Regulation (EU) 2016/679 of the European Parliament
and of the Council of 27 April 2016 on the protection of natural persons with
regard to the processing of personal data and on the free movement of such
data, and repealing Directive 95/46/EC (General Data Protection Regulation).
https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng. Accessed: 2025-9-19.
[34] Saiteja Utpala, Sara Hooker, and Pin-Yu Chen. 2023. Locally differentially private
document generation using zero shot prompting. InFindings of EMNLP.
[35] Shuai Wang, Ekaterina Khramtsova, Shengyao Zhuang, and Guido Zuccon. 2024.
Feb4rag: Evaluating federated search in the context of retrieval augmented gen-
eration. InSIGIR. 763–773.
[36] Yuhao Wang, Ruiyang Ren, Yucheng Wang, Wayne Xin Zhao, Jing Liu, Hua
Wu, and Haifeng Wang. 2025. Unveiling Knowledge Utilization Mechanisms in
LLM-based Retrieval-Augmented Generation. InSIGIR.
[37] Ziyao Wang, Zheyu Shen, Yexiao He, Guoheng Sun, Hongyi Wang, Lingjuan
Lyu, and Ang Li. 2024. Flora: Federated fine-tuning large language models with
heterogeneous low-rank adaptations.NeurIPS37 (2024), 22513–22533.
[38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning
in large language models.NeurIPS35 (2022), 24824–24837.
[39] Chulin Xie, Zinan Lin, Arturs Backurs, Sivakanth Gopi, Da Yu, Huseyin A Inan,
Harsha Nori, Haotian Jiang, Huishuai Zhang, Yin Tat Lee, et al .2024. Differentially
private synthetic data via foundation model apis 2: Text. InICML.
[40] Jing Xu and Jingzhao Zhang. 2024. Random masking finds winning tickets for
parameter efficient fine-tuning. InICML. 55501–55524.
[41] Shicheng Xu, Liang Pang, Jun Xu, Huawei Shen, and Xueqi Cheng. 2024. List-
aware reranking-truncation joint model for search and retrieval-augmented
generation. InWWW. 1330–1340.
[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering. InEMNLP. 2369–2380.
[43] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, and et al. 2023. React:
Synergizing reasoning and acting in language models. InICLR.
[44] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2024. Language models
are super mario: Absorbing abilities from homologous models as a free lunch. In
ICML.
[45] Zichun Yu, Chenyan Xiong, Shi Yu, and Zhiyuan Liu. 2023. Augmentation-
adapted retriever improves generalization of language models as generic plug-in.
InACL. 2421–2436.
[46] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu, Yue Xing, Han Xu, Jie
Ren, Yi Chang, Shuaiqiang Wang, Dawei Yin, et al .2024. The good and the bad:
Exploring privacy issues in retrieval-augmented generation (RAG). InFindings
of ACL. 4505–4524.
[47] Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren, Tianqi Zheng, Hanqing Lu,
Han Xu, Hui Liu, Yue Xing, and Jiliang Tang. 2024. Mitigating the privacy issues
in retrieval-augmented generation (rag) via pure synthetic data.arXiv preprint
arXiv:2406.14773(2024).
[48] Jianyi Zhang, Saeed Vahidian, Martin Kuo, Chunyuan Li, Ruiyi Zhang, Tong
Yu, Guoyin Wang, and Yiran Chen. 2024. Towards building the federatedgpt:
Federated instruction tuning. InICASSP. 6915–6919.
[49] Dongfang Zhao. 2024. Frag: Toward federated vector database management
for collaborative and secure retrieval-augmented generation.arXiv preprint
arXiv:2410.13272(2024).
[50] Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chunyan Miao. 2025. Medrag: Enhanc-
ing retrieval-augmented generation with knowledge graph-elicited reasoning
for healthcare copilot. InWWW. 4442–4457.