# ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning

**Authors**: Jiawei Zhou, Hang Ding, Haiyun Jiang

**Published**: 2025-11-20 13:05:09

**PDF URL**: [https://arxiv.org/pdf/2511.16326v1](https://arxiv.org/pdf/2511.16326v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful framework for knowledge-intensive tasks, yet its effectiveness in long-context scenarios is often bottlenecked by the retriever's inability to distinguish sparse yet crucial evidence. Standard retrievers, optimized for query-document similarity, frequently fail to align with the downstream goal of generating a precise answer. To bridge this gap, we propose a novel fine-tuning framework that optimizes the retriever for Answer Alignment. Specifically, we first identify high-quality positive chunks by evaluating their sufficiency to generate the correct answer. We then employ a curriculum-based contrastive learning scheme to fine-tune the retriever. This curriculum leverages LLM-constructed Knowledge Graphs (KGs) to generate augmented queries, which in turn mine progressively challenging hard negatives. This process trains the retriever to distinguish the answer-sufficient positive chunks from these nuanced distractors, enhancing its generalization. Extensive experiments on 10 datasets from the Ultradomain and LongBench benchmarks demonstrate that our fine-tuned retriever achieves state-of-the-art performance, improving 14.5% over the base model without substantial architectural modifications and maintaining strong efficiency for long-context RAG. Our work presents a robust and effective methodology for building truly answer-centric retrievers.

## Full Text


<!-- PDF content starts -->

ARK: Answer-Centric Retriever Tuning via
KG-augmented Curriculum Learning
Jiawei Zhou1*, Hang Ding1*, Haiyun Jiang2†
1ACEM, Shanghai Jiao Tong University,2SAIS, Shanghai Jiao Tong University
{davidzjw,dearsloth}@sjtu.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has
emerged as a powerful framework for
knowledge-intensive tasks, yet its effective-
ness in long-context scenarios is often bottle-
necked by the retriever’s inability to distinguish
sparse yet crucial evidence. Standard retriev-
ers, optimized for query-document similarity,
frequently fail to align with the downstream
goal of generating a precise answer. To bridge
this gap, we propose a novel fine-tuning frame-
work that optimizes the retriever for Answer
Alignment. Specifically, we first identify high-
quality positive chunks by evaluating their suf-
ficiency to generate the correct answer. We
then employ a curriculum-based contrastive
learning scheme to fine-tune the retriever. This
curriculum leverages LLM-constructed Knowl-
edge Graphs (KGs) to generate augmented
queries, which in turn mine progressively chal-
lenging hard negatives. This process trains the
retriever to distinguish the answer-sufficient
positive chunks from these nuanced distractors,
enhancing its generalization. Extensive exper-
iments on 10 datasets from the Ultradomain
and LongBench benchmarks demonstrate that
our fine-tuned retriever achieves state-of-the-art
performance, improving 14.5% over the base
model without substantial architectural modi-
fications and maintaining strong efficiency for
long-context RAG. Our work presents a robust
and effective methodology for building truly
answer-centric retrievers.
1 Introduction
Large Language Models (LLMs) have achieved
human-level performance on many NLP tasks
(Achiam et al., 2023; Touvron et al., 2023), but still
struggle with long-term memory, often omitting
or conflating details in scenarios requiring com-
*Equal contribution. Author order is randomized.
†Corresponding Author: Haiyun Jiang. (E-mail:
haiyun2025@sjtu.edu.cn)plex reasoning or extended context (Li et al., 2024;
Lazaridou et al., 2021).
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020) addresses this limitation by connecting
LLMs to external knowledge sources, refreshing
their memory at inference time. Since its introduc-
tion, RAG has rapidly evolved from naive RAG
pipeline, with vector retrieval (Shi et al., 2023;
Borgeaud et al., 2022) to advanced pipelines that
incorporate recursive chunking (Sarthi et al., 2024),
knowledge graphs (KGs) (Edge et al., 2024), and
internal memory modules (Qian et al., 2024), sub-
stantially improving the handling of long-context
input.
While KG-integrated pipelines (Edge et al.,
2024) have shown promising gains for complex
summarization, they suffer from efficiency and ac-
curacy bottlenecks in broader retrieval tasks. The
indexing phase in KG-based RAG (Edge et al.,
2024) and follow-up works (Gutiérrez et al., 2024;
Guo et al., 2025) requires processing extremely
large token volumes with powerful LLMs, result-
ing in high computational cost. In addition, KGs
often struggle with fine-grained entity disambigua-
tion: community-curated clusters, though rich, are
noisy and insufficiently filtered. Consequently, re-
trieval may aggregate irrelevant or even conflicting
content, reducing both the consistency and quality
of generated outputs.
To train a retriever for true answer sufficiency,
we propose ARK (Answer-centric Retriever fine-
tuning via KG-driven curriculum), a framework
that redefines the role of Knowledge Graphs in
RAG. Rather than serving as a direct retrieval
source, the KG powers anAnswer-Centric Cur-
riculum Learningscheme, enabling fine-grained
discrimination, sufficiency-aware retrieval, and im-
proved generalization of the retriever.
At its core, ARKfirst identifies what makes evi-
dence truly useful—whether it suffices to generate
the correct answer. We formalize this with an in-
1arXiv:2511.16326v1  [cs.IR]  20 Nov 2025

context sufficiency metric combining three align-
ment strategies (Forward, Backward, Retriever) to
extract high-qualitypositive chunksas anchors.
Building on these positives, ARKleverages the KG
as ahard-negative generator: It runs Personalized
PageRank(PPR) over the KG to extract answer-
relevant subgraphs, which in turn guide the cre-
ation of augmented queries. These queries are
specifically designed to mine progressively chal-
lenging hard negatives. Concretely, we use PPR on
a query-specific subgraph to expose co-occurrence
neighbors near the gold entities, and inject them
into query augmentations so the retriever is drawn
to false positives—highly related yet insufficient
evidence. This community-aware mining yields
harder, more calibration-relevant negatives than
random or keyword-based baselines. Through con-
trastive training against this curriculum, ARKlearns
to prioritize truly answer-informative segments
while filtering misleading context, mastering both
sufficiency and fine-grained discrimination.
To summarize, our contributions are:
•We propose ARK, a framework that finetunes the
retriever through contrastive learning for scalable
long-context retrieval.
•We devise a synthetic query-generation pipeline
that uses KG subgraphs to produce challenging
hard negatives, and integrate them into an answer-
centric curriculum learning scheme that progres-
sively increases negative difficulty.
•We introduce an in-context answer sufficiency
metric, composed of three complementary align-
ment scores (Forward, Backward, and Retriever),
to identify high-quality positive chunks that serve
as the anchor for contrastive learning.
•Through extensive experiments, we demonstrate
thatARKachieves state-of-the-art retrieval perfor-
mance on 8 out of 10 datasets across the Long-
Bench and Ultradomain benchmarks, with an
average F1-score improvement of 14.5% over
the base model, showcasing its effectiveness and
efficiency.
2 Related work
2.1 Traditional RAG Techniques
RAG systems enhance LLM outputs by combin-
ing retrieval with generation. Early implementa-
tions used fixed retrievers to supply documents to
a reader model: classical methods such as BM25
(Robertson et al., 2009) relied on lexical matching,
while neural retrievers like DPR (Karpukhin et al.,2020) employed dense embeddings for semantic
retrieval. Typically, retrievers were trained on QA
pairs, with readers fine-tuned independently.
Later work emphasized tighter integration. RAG
models (Lewis et al., 2020) enabled end-to-end fine-
tuning by treating retrieved documents as latent
variables, aligning the retriever with a BART-based
generator. This improved recall and consistency
but increased complexity due to non-differentiable
retrieval. To mitigate this, techniques such as hard
negative mining (Robinson et al., 2020) and knowl-
edge distillation from generators into retrievers
(Izacard and Grave, 2021) were proposed, further
enhancing retriever–reader synergy.
2.2 Advanced RAG Techniques
Recent advances in RAG aim to improve retrieval
reasoning and integrate hybrid knowledge sources.
One line of work focuses on query rewriting and
decomposition. RQ-RAG (Chan et al., 2024) en-
hances multi-hop QA by decomposing complex
queries into simpler sub-queries, while HyDE (Gao
et al., 2022) generates hypothetical documents that
serve as refined queries to boost retrieval precision.
Context rewriting techniques have also emerged.
MemoRAG (Qian et al., 2024) further addresses
long-context retrieval by compressing memory and
producing clue phrases to guide retrieval, compos-
ing answers from retrieved snippets.
Another direction integrates symbolic and neu-
ral approaches, often by incorporating KGs into
retrieval pipelines. GraphRAG (Edge et al., 2024)
constructs graphs from documents and summarizes
clusters into “community reports” via community
detection, supporting improved multi-hop reason-
ing. LightRAG (Guo et al., 2025) simplifies this
pipeline by first retrieving low-level nodes and then
following graph links to higher-level concepts, im-
proving both recall and efficiency. HippoRAG
(Gutiérrez et al., 2024) models memory consoli-
dation using KGs and PPR, retrieving subgraphs at
query time to emulate long-term memory access.
3 Methodology
3.1 Framework
Our proposed framework, ARK, follows a two-
stage architecture—Query ConstructionandCon-
trastive Finetuning—as illustrated in Figure 1.
The first stage prepares curriculum components,
while the second performs training.
2

Figure 1:Our RAG Retriever Finetuning Framework ARK, which consists of two major stages: A (Query
Construction): From long documents and their corresponding QA pairs, we extract a query-based subgraph using
an LLM-generated KG. The subgraph is reformulated with knowledge injection to produce enriched queries. B
(Contrastive Finetuning): Using both the original query and injected variants, we identify positive chunks (via
alignment scoring) and hard negatives (that match injected queries but differ semantically from ground truth).
InQuery Construction, we build an LLM-
derived KG from the context, identify entities from
the QA pair, and extract a query-specific subgraph.
This subgraph supports the creation ofinjected
queries, which preserve the semantics of the orig-
inal while adding contextual structure. These en-
riched queries serve as the basis for generating hard
negatives. InContrastive Finetuning, we define
Figure 2:Query Construction Phase.The pipeline
begins withKG Construction, where we extract entities,
relations, and covariates from long documents to con-
struct an LLM-generated KG. Given a corresponding
QA pair, relevant entities are extracted and used tocon-
struct PPR-based subgraphsfrom the KG, with varying
maximum sizes to control difficulty. Finally,Augmented
Queriesare formulated with LLM conditioned on these
candidate subgraphs.
positives using our in-context answer sufficiency
metric (combining three alignment scores) to rankcontext chunks and select the most sufficient ones.
Hard negatives are chunks that score highly for
an injected query but are absent from the positive
set. With these positives and negatives, we finetune
the retriever via contrastive learning, enhancing
its discriminative ability. The resulting retriever
integrates seamlessly into existing RAG pipelines
without architectural changes.
3.2 KG-based Query Construction
During training, to effectively extract high-quality,
answer-guided queries from ultra-long source con-
texts, we develop an innovative pipeline that inte-
grates LLM-assisted KG construction, PPR-based
subgraph construction, and finally query formation.
As illustrated in Figure 2, the data generation pro-
cess consists of three steps.
KG constructionWe adopt a prompt design in-
corporating Chain-of-Thought (CoT) reasoning to
enhance the quality of entity recognition. Different
from the GraphRAG (Edge et al., 2024) approach,
our method focuses solely on extracting entities
and their relationships, removing the need for gen-
erating community-level summaries.
Due to the inherent limitations of LLMs, specif-
ically their focus on localized context, the con-
structed KG may exhibit lower edge density com-
pared to traditional KGs, as only intra-chunk rela-
tionships are identified. To address this sparsity,
we augment the graph by introducing undirected
edges between entities whose embedding similarity
exceeds a predefined threshold τ, thereby enrich-
ing the graph’s connectivity and facilitating more
3

Algorithm 1:Cohesive Subgraph Extrac-
tion
Require: Matched entities Va, LLM-generated KG G(V, E) ,
PPR parameterα, ϵ, Maximum subgraph sizek
Ensure:Answer-related entitiesV comm
1:χ a←Indicator function ofV a
2: Computeaprcomm(α, χ a)withϵthreshold
3:{apri, vi}i∈V←sort(aprcomm,desc=True)
4:{apri, vi}i∈V′←Filter trailing terms of{apri, vi}i∈V
where{i|apri< ϵ}
5:{apri, vi}i∈V′← {−logapri, vi}i∈V′
6:index←∆ arg max i, i<min{k,|V′|}apr
7:V comm←Firstindexentities from{apri, vi}i∈V′
effective downstream retrieval.
Cohesive Subgraph Extraction by Community
SearchTo capture neighborhoods most relevant
to the input answer query, we employ commu-
nity search, which identifies query-dependent sub-
graphs rather than global clusters. This aligns with
our retrieval objective: surfacing context that is se-
mantically close to the query yet hard to distinguish
from true positives, thus serving as high-quality
hard negatives. LLM-derived knowledge graphs
are typically sparse, with edges concentrated within
intra-chunk links, making purely topology-driven
methods less effective. To address this, we adopt
Personalized PageRank (PPR) seeded on query en-
tities to extract semantically coherent communities.
However, due to the sparsity of LLM-derived
KGs, where edges are primarily confined to intra-
chunk relationships, conventional topology-driven
community search methods often fail to perform
effectively. To overcome this limitation, we utilize
PPR to assess and construct communities based on
semantically aligned entities. We begin by extract-
ing the positive query entities Vafrom the answer
and identifying their corresponding entities within
the KG. Using these matched entities, we perform
PPR based on the matched ones, formally defined
as:
pr(α, χ a) =αχ a+ (1−α)Wpr(α, χ a)(1)
Here, pr(α, χ a)denotes the PPR value for the
given entities aon KG G,χadenotes the indi-
cator function of a,αdenotes the teleport probabil-
ity, and W=A⊤D−1represents the normalized
transition matrix based on adjacency matrix Aand
degree matrixDfrom graphG.
Due to scalability challenges in computing PPR
over large graphs, we employ power iteration to ap-
proximate the solution. The approximation vector
Figure 3:Contrastive Finetuning Phase.Our fine-
tuning pipeline comprises two sequential components:
Ranking Alignment, in which for each sample, we com-
bine three alignment scores to select the Top- Mchunks
as positive chunks; followed byCurriculum-based Con-
trastive Learning, which progressively refines the re-
triever through (i) in-batch negative sampling, (ii) hard
negatives T−
hardLmined via query set T−
hardL, and (iii)
more challenging negatives T−
hardSobtained from Qaug
S.
apr(0)(α, χ a)is initialized as χa, and iteratively
updated as follows:
apr(n+1)(α, χ a) =αχ a+(1−α)Wapr(n)(α, χ a)
(2)
Given the absolute approximation error bound:apr(n+1)(α, χ a)−apr(n)(α, χ a)
∞< ϵ The
computational complexity of the iterative method is
O 
|E|log1
ϵ
, as initially proposed by (Haveliwala
and Kamvar, 2003).
Following the approach of (Andersen and Chung,
2007; Zhou et al., 2025), we detect sharp drops in
the approximated PPR scores to delineate topologi-
cally coherent communities, but here we choose the
largest first difference of |logapr| as it naturally
shapes a cohesive subgraph. From the resulting
subgraph, community information is subsequently
used to construct augmented queries to generate
hard negatives T−
hardused for the retriever. The
overall subgraph extraction is outlined in Algo-
rithm 1.
3.2.1 Query Formation
Once we have retrieved community entities Vcomm,
we synthesize a richer query collection Qaugby
perturbing and extending the original QAinstance.
Each new query carries additional context, such
as answer spans, related entities, or semantic re-
lations, and thus provides the retriever with more
4

signals for hard negative chunks. Concretely, for
each original QA pair, we sample subgraph entities
Vcomm⊆V of controllable size and connectivity,
and then apply the following query-transformation
requirements:
Category Example
Comparison Compare or contrast among the
given entities.
Causal Ask for a cause or effect based on
the descriptions.
Quotation Use a quote or a redefined term from
the descriptions.
Perspective Shift Frame the question from a different
perspective.
Table 1:Query Transformation Categories.Examples
are instructions used in generating injected queries.
By instantiating the augmented query genera-
tion process using an LLM over subgraphs of two
different sizes, we construct a diverse pool of aug-
mented queriesQaug={q 1, . . . , q N}, categorized
into large and small variants, Qaug
LandQaug
S, re-
spectively. Each query type is paired with a corre-
sponding set of hard negatives, denoted as T−
hardL
andT−
hardS. This augmentation strategy facilitates
the curriculum learning of the retriever, which en-
hances its capacity to capture fine-grained semantic
distinctions.
3.3 Alignment-Based Fine-Tuning
After extracting answer-relevant chunks, we fine-
tune the retriever using a multi-stage framework.
Specifically, we introduce an answer-centric in-
context scoring approach, carefully designed to
mitigate weight collapse by maintaining balanced
and informative gradients. Building upon this
alignment-based scoring and ranking, we progres-
sively apply a curriculum learning strategy that
incrementally exposes the retriever to increasingly
challenging negative samples, thereby systemati-
cally enhancing its discriminative power and ensur-
ing stable generalization performance.
3.3.1 Alignment-Based Ranking
To retrieve positive chunks, we propose several
alignment-based scoring functions that link a can-
didate chunk not only to the query but also to the
expected answer, which is relevant to the query and
sufficient to support accurate answer generation.
Forward AlignmentGiven a chunk tand the
full question q, we evaluate the likelihood of the
generator LLM (parameterized by θ) reproducing
the reference answer a=⟨a 1, . . . , a |a|⟩. This scorequantifies thesufficiencyof the chunk tto gener-
ate the correct answer. During scoring, the con-
catenated query and chunk prompt [q;t] is fed to
the model. We employ teacher forcing, using the
ground-truth answer tokens as targets to calculate
the mean token log-likelihood without them be-
ing part of the conditioning context. The forward
alignment scoreS fis thus defined as:
Sf(q, t, a) =1
|a||a|X
i=1logp θ(ai|q, t, a <i)(3)
This score is computed in a single forward pass by
aggregating the log-softmax probabilities for each
token in the ground-truth answer.
Backward AlignmentAnalogously, to measure
therelevanceof a chunk in linking the answer back
to the original query, we pair the chunk twith the
answer aand task the model with reconstructing
the question q. The backward score Sbis calculated
using the same teacher-forcing technique:
Sb(a, t, q) =1
|q||q|X
j=1logp θ(qj|a, t, q <j)(4)
This bidirectional scoring mechanism ensures that
selected chunks are strongly correlated with the
reasoning path from question to answer.
Parameter AlignmentTo regularize the fine-
tuning process and mitigate catastrophic forgetting,
we incorporate the original retriever’s similarity
score, Sv, as a form of parameter alignment. This
score uses the cosine similarity to preserve the ge-
ometric structure learned inherent in the original
retriever.
Sv(q, t) = sim(q, t)(5)
The final unified score Sfor a given chunk tand
QAq, ais a weighted combination of these three
components. We did not tune these weights, as
we believe adjustment is unnecessary. Instead, we
adopt intuitive fixed values: equal weights for for-
ward and backward alignment, and a slightly lower
weight for parameter regularization ( λf= 1.0 ,
λb= 0.3,λ v= 1.0).
S(t) =λ fSf+λbSb+λvSv (6)
This unified score serves as the primary criterion
for identifying high-quality positive samples for
the initial stage of our fine-tuning curriculum.
5

3.3.2 Curriculum-Based Contrastive
Finetuning
Illustrated in Figure 3. We structure the fine-tuning
process as a three-stage curriculum, where the dif-
ficulty of the discrimination task increases at each
stage. This approach allows the model to first learn
a robust answer-centric representation and then pro-
gressively refine it by focusing on increasingly sub-
tle and challenging distractors.
Stage 1: Initial Answer-Centric Alignment
The primary goal of this stage is to align the re-
triever with chunks that are highly conducive to
generating the correct answer. For each query q,
we calculate the unified score S(t) for all candi-
date chunks in T. We select the top- Mchunks with
the highest scores as the positive set T+. The re-
triever is then trained using a contrastive objective.
Specifically, for a given positive pair (q, t+)where
t+∈ T+, we use other positive chunks within the
same mini-batch as in-batch negatives. This is an
effective and efficient method for initial training.
The loss for this stage is the InfoNCE loss:
LStage1 =−logexp(sim(q, t+)/τ)P
t′
j∈Batch exp(sim(q, t′
j)/τ)
(7)
where sim(·,·) is the cosine similarity from the
retriever being trained and τis a temperature hy-
perparameter.
Stage 2: Coarse Alignment with Qaug
LAfter the
initial alignment, we further enhance the retriever’s
robustness by incorporating hard negatives. Specif-
ically, we leverage Qaug
Lgenerated from the KG.
Using the retriever fine-tuned in Stage 1, we re-
trieve the top- Kchunks for each complex query
q∈ Qaug
L. From this set, we exclude any chunks
that appear in the ground-truth positive set T+.
The remaining chunks constitute the hard negative
setT−
hardL. Compared to T−
hardS, these negatives
are less challenging due to their greater semantic
diversity. The retriever is then further trained to
distinguish the original positive chunks from these
hard negatives.
LStage2 =−loges(q,t+)
P
t∈C(q) es(q,t),(8)
where s(q, t) = sim(q, t)/τ andC(q) ={t+} ∪
T−
hard.Stage 3: Fine-Grained Alignment with Qaug
SIn
the final stage, we further sharpen the retriever
using the simple augmented queries Qaug
S, which
are minor perturbations of the original query (e.g.,
with only distracting covariates added). We use
the retriever from Stage 2 to retrieve chunks and,
after filtering out the golden positives, obtain a set
of "harder" negative chunks T−
hardS. The training
objective remains the InfoNCE loss, but with a
more challenging target.
4 Experiment
4.1 Datasets
In this paper, we focus on the long-text QA task.
For training, we sample 200 cases each from the
Finance and Legal domains (both from the Ultrado-
main dataset) to generate augmented queries. For
evaluation, we first select five domains from the
Ultradomain benchmark in MemoRAG (Qian et al.,
2024), namely Biology, Fiction, Music, Technol-
ogy, and Philosophy, each represented by a dis-
tinct domain-specific dataset. We also use five
LongBench (Bai et al., 2023) dataset which in-
cludes both single-document QA: NarrativeQA
(Koˇcisk`y et al., 2018), Qasper (Dasigi et al., 2021)
and multi-document QA: HotpotQA (Yang et al.,
2018), 2WikiMQA (Ho et al., 2020), and MuSiQue
(Trivedi et al., 2022).
4.2 Experiment Setup
Our FrameworkWe adoptQwen3-embedding
(Zhang et al., 2025b) as our base model for fine-
tuning, due to its state-of-the-art performance
across a wide range of downstream tasks and its
strong knowledge inheritance from the Qwen LLM.
To emphasize the efficiency and scalability of our
framework, we utilize the smallest variant (0.6B
parameters). For each subgraph, we generate 10
augmented queries and retrieve 10 positive chunks
per query. To construct the negative set, we sample
the top 20 retrieved chunks and exclude any that
overlap with the positives.
Model SelectionWe first employ the latest
Gemini-2.5-Flash(Comanici et al., 2025) as our
foundational model for OpenIE. The construction
of our KG follows the GraphRAG (Edge et al.,
2024), but the community generation is different.
The NetworkX library is utilized to execute approx-
imated PPR when conducting a query-based com-
munity search. We then employGemini-2.5-Pro
(Comanici et al., 2025) for constructing injected
6

Table 2:Main Evaulation results. The evaluation metrics are F1-score / Win Rate (%), with thebestresults
highlighted in bold and the second-best results underlined. The improvement rate ( ↑%, (ARK- Base) / Base) is
calculated based on our base model Qwen3-embedding. Cell shading indicates relative win rates compared to ARK.
METRICSMODELSLongBench UltraDomain
nar qas mus 2wiki hot bio fic music tech phil
F1Full 12.95 22.79 6.74 20.13 26.87 27.47 25.75 25.50 22.68 23.05
Qwen3-embedding 19.58 23.90 14.19 21.24 35.27 32.99 29.41 34.90 38.03 34.04
BGE-M3 18.37 23.3321.1322.86 38.64 32.52 31.72 35.34 39.13 35.97
Stella-v5 20.90 23.39 17.08 22.13 35.45 33.85 32.41 35.02 35.16 34.09
Jina-emb-v3 19.39 20.70 20.58 19.34 39.17 32.88 29.00 33.74 38.74 36.81
GraphRAG 4.21 7.69 2.15 5.52 3.03 18.87 16.92 14.97 21.93 20.01
LightRAG 2.65 3.25 1.95 3.67 2.74 16.06 14.13 15.08 12.19 14.04
HippoRAG 11.51 21.90 13.0930.9628.71 36.13 29.23 32.94 27.15 29.06
MemoRAG 15.49 17.96 8.74 16.57 22.79 31.08 27.87 33.26 39.14 31.98
ARK(Ours) 21.57 24.0420.60 23.41 42.35 36.19 32.59 38.03 40.16 37.86
↑% 10.2 0.6 45.2 22.41 20.1 9.7 10.8 9.0 5.6 11.2
ARK WINRATEFull 83.33 46.03 80.00 64.52 70.97 95.00 100.00 88.89 94.74 100.00
Qwen3-embedding 58.33 52.54 63.46 57.14 68.89 95.00 85.71 94.74 78.57 55.56
BGE-M3 60.00 50.77 56.00 52.54 52.83 70.59 84.62 58.82 73.33 60.00
Stella-v5 65.67 66.67 58.00 50.00 67.39 72.22 62.50 64.71 71.43 89.47
Jina-emb-v3 63.08 54.84 54.90 57.41 43.24 77.78 61.54 66.67 58.82 50.00
GraphRAG 93.62 90.70 78.26 83.75 78.57 100.00 100.00 85.00 95.00 100.00
LightRAG 96.63 96.70 91.46 91.36 96.74 100.00 100.00 95.00 100.00 100.00
HippoRAG 87.34 53.85 58.21 34.67 60.76 77.78 44.44 62.50 70.00 70.00
MemoRAG 80.00 75.00 66.22 57.14 67.65 92.86 94.12 84.21 87.50 88.24
queries with the same ground truth as the original
query. We useGPT-4.1to evaluate win rates via
pairwise comparisons.
BaselinesWe compare our approach against
three categories of baselines: 1)Full: Directly pro-
viding the entire context to an LLM. 2) Dense Re-
trieval model:Qwen3-embedding: The original re-
triever without fine-tuning.BGE-M3(Chen et al.,
2023): A hybrid retrieval model that integrates mul-
tiple strategies to achieve high accuracy and gen-
eralization across benchmarks.Stella-v5(Zhang
et al., 2025a): A top-ranking retriever on the MTEB
leaderboard (Muennighoff et al., 2022).Jina-emb-
v3(Sturua et al., 2024): A powerful and widely-
used multilingual, multi-task embedding model.
3) Advanced RAG methods:GraphRAG(Edge
et al., 2024): Utilizes LLM-generated knowledge
graphs and the Leiden algorithm for hierarchical
retrieval.LightRAG(Guo et al., 2025): Combines
dual-level retrieval with vector and graph struc-
tures.HippoRAG(Gutiérrez et al., 2024): Lever-
ages PPR for community-based retrieval.l.Mem-
oRAG(Qian et al., 2024): Employs a lightweight
long-context LLM to construct global memory and
generate retrieval cues.
For online QA, we adoptMistral-7B-Instruct-
v0.2-32K(Jiang et al., 2023) as the default gener-
ator to avoid potential pretraining contamination.
Effectiveness is evaluated with F1-score (followingeach dataset’s original setup) and pairwise win rate
averaged over five runs for consistency.
4.2.1 Running Environment
ARKis implemented in Python 3.10 with PyTorch
2.7.1. All experiments are conducted on a ma-
chine equipped with 8 NVIDIA H20 GPU. We
utilize NetworkX ,PyTorch ,transformers , and
sentence transformers as the main libraries.
In addition, we use Ollama for inference and
SWIFT (Zhao et al., 2024) to finetune the Qwen3-
embedding.
4.3 Performance Evaluation
Table 7 reports the F1 scores and pairwise win-
rate on the LongBench and UltraDomain bench-
marks. Overall, ARK consistently outperforms
its base model (Qwen3-embedding) across all
datasets and achieves state-of-the-art performance
compared to both KG-based baselines and top-
ranking dense retrievers. In particular, ARKattains
consistently higher pairwise win rates across the
10 evaluated datasets, outperforming both graph-
based approaches (e.g., GraphRAG, LightRAG)
and strong dense encoders (e.g., BGE-M3, Stella,
Jina, Qwen3). Notably, it exceeds a 50% win rate
on the majority of benchmarks.
ARKshows strong generalization beyond its train-
ing domains (Finance and Legal), maintaining ro-
bust performance on unseen datasets, showing
7

that our KG-guided curriculum not only improves
retrieval accuracy but also enhances the ability
to surface contextually meaningful evidence be-
yond the training distribution. Gains are especially
pronounced on reasoning-intensive tasks such as
MuSiQue and HotpotQA, where retrieval must syn-
thesize dispersed evidence. By explicitly optimiz-
ing foranswer sufficiency, the retriever learns to
favor chunks that are both relevant and sufficient for
generating faithful answers. On certain multi-hop
STAGESCORINGmus hot bio phil
Original - 14.19 35.27 32.99 34.04
1stFull 18.44 40.96 32.35 34.89
w/o F.A. 14.31 36.40 31.36 32.95
w/o B.A. 16.73 39.74 33.57 34.12
w/o P.A. 15.29 37.03 33.18 34.73
2ndFull 18.47 42.13 34.74 36.11
3rd(ARK) Full 20.60 42.35 36.19 37.86
Table 3:Ablation study. STAGEdenotes finetuning
stages (Originalis the base model), and SCORINGspec-
ifies alignment types: Forward (F.A.), Backward (B.A.),
and Parameter (P.A.).
tasks like 2Wiki, graph-centric methods such as
HippoRAG remain competitive due to their traver-
sal advantage. Nevertheless, ARKmatches or sur-
passes their performance without requiring costly
graph construction or long-context LLMs, thus of-
fering higher efficiency. Moreover, unlike resource-
intensive systems such as MemoRAG, which de-
mand end-to-end training, our approach fine-tunes
only the retriever. This modular design enables
seamless integration into existing RAG pipelines,
making ARKboth practical and scalable. In sum-
mary, by emphasizinganswer sufficiencywhile pre-
servingquery similarity, ARKconsistently yields
relevant and sufficient evidence for long-context
retrieval without altering the underlying RAG ar-
chitecture.
4.4 Ablation Study
To better understand the role of each design compo-
nent, we conduct ablations over the three alignment
strategies and the curriculum process (Table 3). For-
ward alignment proves most critical, as it directly
measures chunk sufficiency for generating correct
answers; its removal leads to the largest drop in per-
formance. Backward alignment further supports
complex reasoning by enforcing semantic coher-
ence, and removing it causes retrieved passages
to match superficially but lack utility. Parameter
alignment, though less dominant, stabilizes train-METHODmus 2wiki nar tech phil
Llama3.1-8B-Instruct
QWEN39.63 28.2617.7019.91 20.72
ARK 11.48(+)36.00(+)15.04(-)21.47(+)21.17(+)
Qwen2.5-7B-Instruct
QWEN310.74 27.65 16.70 19.46 22.62
ARK 14.61(+)29.47(+)16.77(+)20.09(+)24.64(+)
Table 4: Transferability of the ARK to different genera-
tors.
ing by anchoring the embedding space, reducing
overfitting and collapse in noisier domains.
The curriculum stages also show clear benefits:
Stage 1 (answer-aligned positives with in-batch
negatives) provides a strong initial boost, Stage 2
introduces coarse hard negatives from larger sub-
graphs to expose the model to more diverse distrac-
tors and enhance robustness, and Stage 3 employs
fine-grained negatives from smaller subgraphs for
sharper discrimination. This progressive structure
teaches the retriever not only to capture relevance
but also to identify evidence truly necessary for
accurate answer generation.
4.5 Transferability Across Generators
We further investigate whether the retriever fine-
tuned on QWEN3-EMBEDDINGcan transfer to
other generators without additional adaptation. We
evaluate end-to-end QA by directly plugging ARK
into two instruction-tuned LLMs: LLAMA-3.1-
8B-INSTRUCTand QWEN2.5-7B-INSTRUCT. As
shown in Table 4, ARK consistently improves per-
formance across both generators, indicating the
generalization of out training method. A slight
drop onNARwith Llama-3.1 suggests interaction
between generator decoding preferences and re-
trieval signals, motivating future work on generator-
aware optimization or joint training aligned with
generation loss.
5 Conclusion
We propose ARK, a fine-tuning framework for re-
trievers that uses curriculum learning to integrate
KG knowledge through hard-negative generation.
ARK constructs a compact knowledge subgraph
using LLM-generated KG and PPR-based cohesive
community search; selects positive examples via
three-way alignment while preserving the base en-
coder’s similarity signal; and applies a three-stage
curriculum with augmented query retrieval that in-
crementally incorporates harder negative chunks.
8

The framework requires no architectural modifi-
cations, which fits standard RAG pipelines. Ex-
periments on UltraDomain and LongBench demon-
strate consistent improvements in F1 score and pair-
wise win-rate.
Limitations
While our framework demonstrates strong effec-
tiveness across diverse domains and tasks, it also
has several limitations. First, our evaluation is con-
strained to publicly available benchmarks, which
may not fully capture the diversity of real-world ap-
plications. In addition, while inference is KG-free,
our training pipeline depends on an LLM-derived
KG for hard-negative mining; noise in entity ex-
traction or linking can affect the curriculum quality.
We leave these aspects for future work.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Reid Andersen and Fan Chung. 2007. Detecting sharp
drops in pagerank and a simplified local partitioning
algorithm. InInternational Conference on Theory
and Applications of Models of Computation, pages
1–12. Springer.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, and 1 others. 2023.
Longbench: A bilingual, multitask benchmark
for long context understanding.arXiv preprint
arXiv:2308.14508.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, and 1 others.
2022. Improving language models by retrieving from
trillions of tokens. InInternational conference on
machine learning, pages 2206–2240. PMLR.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. Rq-rag: Learn-
ing to refine queries for retrieval augmented genera-
tion.arXiv preprint arXiv:2404.00610.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2023. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2309.07597.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and1 others. 2025. Gemini 2.5: Pushing the frontier with
advanced reasoning, multimodality, long context, and
next generation agentic capabilities.arXiv preprint
arXiv:2507.06261.
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan,
Noah A Smith, and Matt Gardner. 2021. A dataset of
information-seeking questions and answers anchored
in research papers.arXiv preprint arXiv:2105.03011.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2022. Precise zero-shot dense retrieval without rele-
vance labels.Preprint, arXiv:2212.10496.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. Lightrag: Simple and fast retrieval-
augmented generation.Preprint, arXiv:2410.05779.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. InThe Thirty-eighth Annual Con-
ference on Neural Information Processing Systems.
Taher H Haveliwala and Sepandar D Kamvar. 2003. The
second eigenvalue of the google matrix. Technical
report, Citeseer.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps.arXiv preprint arXiv:2011.01060.
Gautier Izacard and Edouard Grave. 2021. Distilling
knowledge from reader to retriever for question an-
swering. InICLR 2021-9th International Conference
on Learning Representations.
Albert Q Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, and 1 others. 2023.
Mistral 7b.arXiv preprint arXiv:2310.06825.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP, pages 6769–6781.
Tomáš Ko ˇcisk`y, Jonathan Schwarz, Phil Blunsom, Chris
Dyer, Karl Moritz Hermann, Gábor Melis, and Ed-
ward Grefenstette. 2018. The narrativeqa reading
comprehension challenge.Transactions of the Asso-
ciation for Computational Linguistics, 6:317–328.
Angeliki Lazaridou, Adhi Kuncoro, Elena Gribovskaya,
Devang Agrawal, Adam Liska, Tayfun Terzi, Mai
Gimenez, Cyprien de Masson d’Autume, Tomas Ko-
cisky, Sebastian Ruder, and 1 others. 2021. Mind
9

the gap: Assessing temporal generalization in neural
language models.Advances in Neural Information
Processing Systems, 34:29348–29363.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in Neural Information Processing Systems,
33:9459–9474.
Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue,
and Wenhu Chen. 2024. Long-context llms strug-
gle with long in-context learning.arXiv preprint
arXiv:2404.02060.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and
Nils Reimers. 2022. Mteb: Massive text embedding
benchmark.arXiv preprint arXiv:2210.07316.
Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao,
and Zhicheng Dou. 2024. Memorag: Moving to-
wards next-gen rag via memory-inspired knowledge
discovery.arXiv preprint arXiv:2409.05591.
Stephen Robertson, Hugo Zaragoza, and 1 others. 2009.
The probabilistic relevance framework: Bm25 and
beyond.Foundations and Trends® in Information
Retrieval, 3(4):333–389.
Joshua Robinson, Ching-Yao Chuang, Suvrit Sra,
and Stefanie Jegelka. 2020. Contrastive learn-
ing with hard negative samples.arXiv preprint
arXiv:2010.04592.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. RAPTOR: Recursive Abstractive Process-
ing for Tree-Organized Retrieval.arXiv preprint.
ArXiv:2401.18059 [cs].
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2023. Replug: Retrieval-
augmented black-box language models.arXiv
preprint arXiv:2301.12652.
Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram,
Michael Günther, Bo Wang, Markus Krimmel, Feng
Wang, Georgios Mastrapas, Andreas Koukounas, An-
dreas Koukounas, Nan Wang, and Han Xiao. 2024.
jina-embeddings-v3: Multilingual embeddings with
task lora.Preprint, arXiv:2409.10173.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, and 1 others. 2023. Llama 2: Open foun-
dation and fine-tuned chat models.arXiv preprint
arXiv:2307.09288.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.arXiv preprint arXiv:1809.09600.
Dun Zhang, Jiacheng Li, Ziyang Zeng, and Fulong
Wang. 2025a. Jasper and stella: distillation of sota
embedding models.Preprint, arXiv:2412.19048.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025b. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang,
Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu,
Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda
Chen. 2024. Swift:a scalable lightweight infrastruc-
ture for fine-tuning.Preprint, arXiv:2408.05517.
Jiawei Zhou, Kai Wang, Jianwei Wang, Kunpeng Zhang,
and Xuemin Lin. 2025. Comet: An interactive frame-
work for efficient and effective community search via
active learning.INFORMS Journal on Computing.
10

A KG-based Query Generation
The KG serves as a core component of our framework, providing structured semantic representations that
enable both entity-level reasoning and query augmentation. In this section, we detail the KG-construction,
hyperparameters, and prompts used for Entity Extraction, and Query Generation.
A.1 KG construction
Our KG construction pipeline transforms unstructured text into a structured graph of entities and relations.
Using an LLM-based extraction process followed by embedding-driven augmentation, we ensure semantic
consistency and connectivity between related concepts. The algorithm 2 outlines this process.
Algorithm 2:KG Construction
Input:Contextcontext, Chunk sizeBand overlap sizeb, Selected generatorLLM, Embedding
modelEMB, Similarity thresholdτ
Output:Generated KGG(V, E)
// 1. Chunking and Extraction
1chunks←Chunk(context, B, b)▷Using token-based chunking
2Entities, Rels← {},{}
3foreachchunkinchunksdo
4llm_output←LLM(prompt,chunk)
5(extracted_ents,extracted_rels)←ParseLLMOutput(llm_output)
6Entities∪(ent,chunk_id)forentinextracted_ents
7Rels∪(tuple,chunk_id)fortupleinextracted_rels
8end
9G←KG_generation(Entities,Rels)
10▷Construct KG using LLM-generated entities and relations
// 2. Graph Augmentation
11sim_matrix←GetCosSimilarityMatrix(G.nodes)
12▷Using embedding model to calculate cosine similarity
13forifrom0tosim_matrix.rows−1do
14forjfromi+ 1tosim_matrix.cols−1do
15ifsim_matrix[i, j]> τthen
16src←ent[i]
17tgt←ent[j]
18tuple←(src,Rel_aug,tgt)
19G.InsertEdge(tuple,None)
20end
21end
22end
23returnG
A.2 Hyperparmeter
Table 5 summarizes the key hyperparameters used in document processing, KG generation, and PPR re-
trieval. The default settings are empirically chosen to balance computational efficiency and representation
quality.
11

Table 5: Key hyperparameters used in KG generation.
Category Parameter Default
Document ProcessingLLMGemini-2.5 Flash
B 512
b 12
Graph AugmentationEMBBGE-m3
τ0.8
PPRα0.85
ϵ1e-4
k200
A.3 Entity Extraction Prompt
To build a coherent KG, we use a prompt that explicitly instruct the LLM to extract entities, their attributes,
and relationships in a structured format.
Entity Extraction Prompt
Given a text document that is potentially relevant to this activity, your task is to identify all entities
of specific types and the relationships among them.
Steps:
1. Identify Entity Types:
...
2. Extract Entities:
...
Format each entity as:
("entity"<|>entity_name<|>entity_type<|>entity_description>)
3. Identify Relationships:
...
Format each relationship as:
("relationship" <|>source_entity<|>target_entity<|>relationship_description<|>
relationship_strength>)
...
Listing 1: A snippet of the prompt used for entity and relationship extraction. The full prompt provides detailed
instructions and examples to the LLM.
A.4 Query Generation Prompt
Beyond KG construction, we employ a query generation process to expand the dataset and develop the
curriculum learning pipeline. This prompt guides an LLM to craft semantically diverse yet answer-
consistent questions by leveraging entity-level context from the KG.
12

Query Generation Prompt
You are an expert in creating complex and confusing questions for educational purposes. Your task
is to generate 10 distinct and challenging questions based on an original question-answer pair and
a set of related entities with their descriptions.
The goal is to formulate questions that are semantically different from the original but lead to the
exact same answer. These new questions should be confusing by design, incorporating details from
the provided entity descriptions to misdirect or challenge the user’s understanding.
Input:
You will receive the following in JSON format:
- original_question: A straightforward question.
- answer: The correct and sole answer to the original question.
- entities: A list of objects, where each object contains:
- name: The name of the entity.
- type: The category of the entity.
- descriptions: A list of strings, each describing a different aspect of the entity.
Task:
Generate 10 new questions (we’ll call them "confusing questions").
Requirements for Confusing Questions:
1. Same Answer: Every generated question must have the exact same answer as the original
question.
2. Incorporate Entities: Each question should subtly weave in information from the entities and
their descriptions.
3. Variety: The questions should be diverse in their structure and focus. You should not include
exact wording / entities in the original question / answer. For example, you can:
...
4. Clarity and Grammar: Despite being confusing, the questions must be grammatically correct
and coherent.
Output Format:
Produce a single JSON object with one key, confusing_questions, which contains a list of 10 string
questions.
...
Listing 2: A snippet of the prompt used for generating confusing questions. The LLM is instructed to use provided
entities to create challenging reformulations of an original question.
B Alignment-based Finetuning
This section describes the alignment-based finetuning procedure, which enables the model to better
evaluate the quality and relevance of retrieved chunks. The alignment module computes log-likelihood
scores for given question–answer pairs relative to context passages, serving as a signal for measuring
faithfulness and guiding downstream retrieval calibration. Model finetuning is performed using the
MS-Swift library (used primarily for fine-tuning Qwen-series model).
B.1 Fintuning Pipeline
The complete training workflow of our alignment-based retriever optimization is summarized in Algorithm
3, which details the three-stage curriculum described in Section 3.3.
B.2 Alignment Prompt
The alignment prompt is designed to evaluate the faithfulness of model-generated answers relative to the
provided context. In the forward alignment setting, the model is prompted to produce an answer strictly
grounded in a provided context. Conversely, reverse alignment evaluates whether a given answer can be
13

Algorithm 3:Three-Stage Alignment-Based Finetuning (ARK)
Input:QA corpus{(q, a,T)}; pretrained generatorp θ; retriever sim(·,·); weights(λ f, λb, λv); temperatureτ;
augmented query setsQaug
L,Qaug
S.
Output:Fine-tuned retriever parameters.
1Stage 0: Alignment Scoring
2foreach(q, a,T)do
3foreacht∈ Tdo
4S f=1
|a|P
ilogp θ(ai|q, t, a <i)
5S b=1
|q|P
jlogp θ(qj|a, t, q <j)
6S v= sim(q, t)
7S(t) =λ fSf+λbSb+λvSv
8end
9T+= TopMt∈TS(t)
10end
11Stage 1: Initial Answer-Centric Alignment
12foreachmini-batchBdo
13foreach(q, t+)∈ Bdo
14s(q, t) = sim(q, t)/τ
15L Stage1(q) =−loges(q,t+)
P
t′∈Batches(q,t′)
16end
17Update retriever by∇1
|B|PLStage1
18end
19Stage 2: Coarse Alignment withQaug
L
20foreachq′∈ Qaug
Ldo
21Retrieve Top-Kchunks and formT−
hardL(q′)
22end
23foreachmini-batchBdo
24L Stage2(q) =−loges(q,t+)
P
t∈C(q)es(q,t) ,C(q) ={t+} ∪ T−
hardL
25Update retriever
26end
27Stage 3: Fine-Grained Alignment withQaug
S
28foreachq′′∈ Qaug
Sdo
29Retrieve Top-Kchunks and formT−
hardS(q′′)
30end
31foreachmini-batchBdo
32L Stage3(q) =−loges(q,t+)
P
t∈C′(q)es(q,t) ,C′(q) ={t+} ∪ T−
hardS
33Update retriever
34end
35returnFine-tuned retriever
justified by its corresponding context. Together, these two directions enable a bidirectional evaluation of
model reliability through log-likelihood estimation, which is later integrated into the retrieval scoring and
finetuning stages.
Forward Alignment Prompt
You are given a context and a question. Answer the question as concisely as you can, using a single
phrase or sentence if possible. If the question cannot be answered based on the information in the
context, write ünanswerable¨. If the question is a yes/no question, answer ÿes¨,¨no¨, or ünanswerable¨.
Do not provide any explanation.
Context:{chunk}
Question:{question}
Answer:
Listing 3: The forward prompt used to retrieve log-likelihood.
14

B.3 Hyperparameter
Table 6 summarizes the key hyperparameters used during alignment-based finetuning. The process
employsfull-parameter tuningon the embedding model to ensure that the learned representations are
optimally aligned with the retrieval task objectives. The configuration is chosen by default and not through
grid-search since the fine-tuning stage is relatively stable.
Table 6: Training Hyperparameters
Parameter Default
Epochs10
batch size2
gradient accumulation steps8
Learning rate6e-6
Lossinfonce
C Inference
This section describes the inference components of our system. The inference module generates final
answers conditioned on those retrieved ones.
C.1 Inference Prompt
The inference prompt serves as the main instruction template for generating final answers from retrieved
text chunks. It is intentionally concise and task-oriented, ensuring that responses are direct, factual, and
free from unnecessary reasoning chains.
Inference Prompt
You are given a scientific article and a question. Answer the question as concisely as you can,
using a single phrase or sentence if possible. If the question cannot be answered based on the
information in the article, write "unanswerable". If the question is a yes/no question, answer "yes",
"no", or "unanswerable". Do not provide any explanation.
Context:{chunk}
Question:{question}
Answer:
Listing 4: The general prompt used to generate answers from the retrieved context.
D Win-rate Evaluation
This section presents the evaluation protocol and results used to compare the performance of different
retrieval and reasoning models. We adopt an LLM-based evaluator that systematically measures pairwise
model performance through criteria grounded in faithfulness and conciseness.
D.1 Win-rate Prompt
To ensure consistent and interpretable evaluation, we employ a structured prompt that directs an LLM
to act as a neutral expert judge. The evaluator receives a ground truth reference, a question, and two
candidate answers. It then performs a two-stage comparison: first applying a disqualification rule to
detect unsupported answers, and subsequently assessing faithfulness (support from the ground truth),
conciseness (brevity without loss of correctness) and overall winner.
15

Win-rate Prompt
You are an expert evaluator. Your task is to rigorously assess two answers to a specific question,
based on a provided Ground Truth. You will use two criteria: Faithfulness and Conciseness.
...
Evaluation Rules:
Disqualification Rule (Primary Check):
First, check if either answer explicitly states that the Ground Truth document does not contain
enough information or evidence to answer the Question.
...
Evaluation Criteria (Secondary Check):
Faithfulness:The degree to which the answer is exclusively and accurately supported by the
provided Ground Truth document.
Conciseness:The degree to which the answer avoids mentioning excessive entities or relationships
that are not essential for answering the Question.
...
Output Format:
Output your complete evaluation in the following JSON format.
{
"Faithfulness": {"Winner": "[Answer 1, Answer 2, Tie, or None]", "Explanation": ...},
"Conciseness": {"Winner": "[Answer 1, Answer 2, Tie, or None]", "Explanation": ...},
"Overall Winner": {"Winner": "[Answer 1, Answer 2, Tie, or None]", "Explanation": ...}
}
Listing 5: A snippet of the prompt for LLM-based evaluation. The prompt defines a strict set of rules, including a
disqualification rule, and requires a structured JSON output.
D.2 Additional Results
Table 7 extends win-rate comparisons to Faithfullness and Conciseness.
Table 7:Additional win-rate results. winrate= 1/|q|×Σ q[1Answer 1 /(1 Answer 1 + 1 Answer 2 )].Cell shading indicates
relative win rates compared toARK.
CRITERIAMODELSLongBench UltraDomain
nar qas mus 2wiki hot bio fic music tech phil
FAITHFULNESSFull 83.10 58.00 75.00 64.44 71.11 100.00 100.00 94.12 100.00 100.00
Qwen3-embedding 59.57 54.17 63.16 61.11 77.42 100.00 92.31 94.74 78.57 58.82
BGE-M3 62.26 55.10 63.41 58.14 56.76 70.59 91.67 64.71 76.92 63.16
Stella-v5 68.52 84.62 64.29 40.00 61.76 72.22 66.67 70.59 76.92 89.47
Jina-emb-v3 60.78 62.50 55.88 53.85 45.45 77.78 66.67 66.67 60.00 55.56
GraphRAG 89.13 88.75 77.27 73.33 76.74 100.00 100.00 90.00 95.00 100.00
LightRAG 76.40 91.86 76.71 76.32 89.41 100.00 70.00 100.00 100.00 100.00
HippoRAG 69.01 65.00 46.03 40.91 60.87 77.78 37.50 50.00 60.00 70.00
MemoRAG 79.17 78.79 68.33 57.38 65.38 93.33 93.75 84.21 87.50 88.24
CONCISENESSFull 90.24 52.31 81.71 60.94 75.00 94.74 100.00 83.33 94.74 100.00
Qwen3-embedding 55.17 50.00 65.31 57.69 62.79 90.00 78.57 73.68 76.92 44.44
BGE-M3 55.88 46.97 48.94 56.90 51.92 70.59 69.23 38.89 66.67 55.00
Stella-v5 72.46 66.67 61.22 60.00 71.11 72.22 62.50 61.11 71.43 78.95
Jina-emb-v3 62.50 50.00 55.10 61.82 50.00 66.67 53.85 60.00 58.82 40.00
GraphRAG 95.83 96.63 95.74 95.29 97.78 100.00 100.00 90.00 90.00 100.00
LightRAG 98.94 96.70 96.34 94.19 97.85 100.00 100.00 95.00 100.00 100.00
HippoRAG 79.22 45.45 48.57 17.33 48.15 33.33 22.22 50.00 60.00 40.00
MemoRAG 86.75 82.19 72.37 67.09 70.42 86.67 94.12 84.21 81.25 76.47
16