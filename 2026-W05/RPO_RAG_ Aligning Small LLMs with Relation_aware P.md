# RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering

**Authors**: Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, Kyong-Ho Lee

**Published**: 2026-01-27 05:46:32

**PDF URL**: [https://arxiv.org/pdf/2601.19225v2](https://arxiv.org/pdf/2601.19225v2)

## Abstract
Large Language Models (LLMs) have recently demonstrated remarkable reasoning abilities, yet hallucinate on knowledge-intensive tasks. Retrieval-augmented generation (RAG) mitigates this issue by grounding answers in external sources, e.g., knowledge graphs (KGs). However, existing KG-based RAG approaches rely on semantics-unaware path sampling and are weakly aligned with KG reasoning objectives, which limits further accuracy gains. They also feed retrieved paths directly into the reasoner without organizing them into answer-centered reasoning paths, hindering small LLMs' ability to leverage the retrieved knowledge. Furthermore, prior works predominantly rely on large LLMs (e.g., ChatGPT/GPT-4) or assume backbones above 7B parameters, leaving sub-7B models underexplored. We address this gap with RPO-RAG, the first KG-based RAG framework specifically designed for small LLMs, to the best of our knowledge. RPO-RAG introduces three key innovations: (1) a query-path semantic sampling strategy that provides informative supervisory signals; (2) a relation-aware preference optimization that aligns training with intermediate KG reasoning signals (e.g., relation); and (3) an answer-centered prompt design that organizes entities and reasoning paths in an interpretable format. Extensive experiments on two benchmark Knowledge Graph Question Answering (KGQA) datasets, WebQSP and CWQ, demonstrate that RPO-RAG effectively bridges the performance gap between small and large language models. On WebQSP, it improves F1 by up to 8.8%, reflecting enhanced answer precision, while on CWQ it achieves new state-of-the-art results among models under 8B parameters in both Hit and F1. Overall, RPO-RAG substantially improves the reasoning capability of small LLMs, even under 3B parameters-highlighting their potential for resource-efficient and practical on-device KGQA applications.

## Full Text


<!-- PDF content starts -->

RPO-RAG: Aligning Small LLMs with Relation-aware Preference
Optimization for Knowledge Graph Question Answering
Kaehyun Um
Department of Computer Science
Yonsei University
Seoul, Republic of Korea
khyun33@yonsei.ac.krKyuHwan Yeom
Department of Computer Science
Yonsei University
Seoul, Republic of Korea
tomma1121@yonsei.ac.krHaerim Yang
Department of Artificial Intelligence
Yonsei University
Seoul, Republic of Korea
hly1013@yonsei.ac.kr
Minyoung Choi
Department of Computer Science
Yonsei University
Seoul, Republic of Korea
min02choi@yonsei.ac.krHyeongjun Yang
Department of Computer Science
Yonsei University
Seoul, Republic of Korea
edbm95@yonsei.ac.krKyong-Ho Lee∗
Department of Computer Science
Yonsei University
Seoul, Republic of Korea
khlee89@yonsei.ac.kr
Abstract
Large Language Models (LLMs) have recently demonstrated remark-
able reasoning abilities, yet hallucinate on knowledge-intensive
tasks. Retrieval-augmented generation (RAG) mitigates this issue
by grounding answers in external sources, e.g., knowledge graphs
(KGs). However, existing KG-based RAG approaches rely on semantics-
unaware path sampling and are weakly aligned with KG reasoning
objectives, which limits further accuracy gains. They also feed re-
trieved paths directly into the reasoner without organizing them
into answer-centered reasoning paths, hindering small LLMs’ abil-
ity to leverage the retrieved knowledge. Furthermore, prior works
predominantly rely on large LLMs (e.g., ChatGPT/GPT-4) or assume
backbones above 7B parameters, leaving sub-7B models underex-
plored. We address this gap withRPO-RAG, the first KG-based
RAG framework specifically designed for small LLMs, to the best
of our knowledge. RPO-RAG introduces three key innovations: (1)
a query-path semantic sampling strategy that provides informa-
tive supervisory signals; (2) a relation-aware preference optimiza-
tion that aligns training with intermediate KG reasoning signals
(e.g., relation); and (3) an answer-centered prompt design that or-
ganizes entities and reasoning paths in an interpretable format.
Extensive experiments on two benchmark Knowledge Graph Ques-
tion Answering (KGQA) datasets, WebQSP and CWQ, demonstrate
that RPO-RAG effectively bridges the performance gap between
small and large language models. On WebQSP, it improves F1 by
up to 8.8%, reflecting enhanced answer precision, while on CWQ
it achieves new state-of-the-art results among models under 8B
parameters in both Hit and F1. Overall, RPO-RAG substantially
improves the reasoning capability of small LLMs—even under 3B
parameters—highlighting their potential for resource-efficient and
practical on-device KGQA applications.
∗Corresponding author.
This work is licensed under a Creative Commons Attribution-NonCommercial-
NoDerivatives 4.0 International License.
WWW ’26, Dubai, United Arab Emirates
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2307-0/2026/04
https://doi.org/10.1145/3774904.3792730CCS Concepts
•Information systems→Question answering.
Keywords
Knowledge Graph Question Answering, Large Language Models,
Retrieval-Augmented Generation, Preference Optimization
ACM Reference Format:
Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun
Yang, and Kyong-Ho Lee. 2026. RPO-RAG: Aligning Small LLMs with
Relation-aware Preference Optimization for Knowledge Graph Question
Answering. InProceedings of the ACM Web Conference 2026 (WWW ’26),
April 13–17, 2026, Dubai, United Arab Emirates.ACM, New York, NY, USA,
11 pages. https://doi.org/10.1145/3774904.3792730
Resource Availability:
The source code of this paper is publicly available at https://github.com/
KaeHyun/RPO-RAG and archived at https://doi.org/10.5281/zenodo.18322650.
The trained RPO-RAG models are publicly available and archived at https:
//doi.org/10.5281/zenodo.18322931.
1 Introduction
Large language models (LLMs) have achieved impressive perfor-
mance across a wide range of NLP tasks [ 13,22,33] but remain
vulnerable to hallucination [ 9,10] in knowledge-intensive tasks.
Retrieval-augmented generation (RAG) mitigates this issue by ground-
ing generation in external knowledge sources (e.g., documents,
databases) [ 26,34]. Among them, knowledge graphs (KGs) are par-
ticularly appealing due to their structured representation of factual
and relational knowledge, providing a natural foundation for com-
plex reasoning tasks such as question answering [ 4,15,35] and
conversational recommender system [ 7,32]. However, when RAG
leverages KGs—so-called KG-based RAG [ 6,21]—the key challenge
lies in bridging symbolic graph reasoning with the text-based rea-
soning capabilities of LLMs.
Existing KG-based RAG approaches can be broadly divided into
two lines. A first group [ 16,17] leverages LLMs themselves to plan
or retrieve knowledge from KGs by generating candidate path se-
quences given a query, which are then grounded in the KG before
1https://huggingface.co/meta-llama/Llama-3.2-3B-InstructarXiv:2601.19225v2  [cs.CL]  28 Jan 2026

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, and Kyong-Ho Lee
Q. What w as the name of music producer of T he Don Killuminati: T he 
7 Day T heory in Juice ?
Based on the reasoning paths, please answer the given question.
Reasoning Paths: 
(B1)Juice → film.performance.film→ m.0j_81z →…  → Bishop
(B2) Juice → film.film.starring → m.0j_823 →…  → Q
…
(B15)The Don Killuminati: The 7 Day Theory → music.release.producer → T upac Shakur →…  
→ film.performance.character → Bishop
(B16) The Don Killuminati: The 7 Day Theory → music.release.producer → T upac Shakur →…  
→ Omar Epps → …  film.performance.character → Q
Predicted Answer: Q
Based on the Answer-Centered-Paths, please answer the given question. 
The Answer-Centered-Paths helps you to step-by-step reasoning to answer the question. 
…
Answer-Centered-Paths:
< Bishop>
(O1) Juice → film.performance.film → …  → Bishop
(O2) The Don Killuminati: The 7 Day Theory→ music.release.producer → T upac Shakur →…  
→ film.performance.character → Bishop
< Q>
(O3) Juice → film.film.starring → …  → Q
(O4) The Don Killuminati: The 7 Day Theory→ music.release.producer → T upac Shakur →…  
→ Omar Epps → …  film.performance.character → Q
Predicted Answer: BishopQis the character portrayed by Omar Epps 
in Juice , not the music producer of The Don 
Killuminati: The 7 Day Theory .
Bishopis the character portrayed by T upac 
Shakur in Juice , and he is also the music 
producer of The Don Killuminati: The 7 Day 
Theory .
Figure 1: An example of the prompt used in existing works
(top) and our designed prompt (bottom), each shown with a
predicted answer from a small LLM (Llama3.2-3B1). Paths
(B1)–(B16) correspond to reasoning paths in existing works,
while (O1)–(O4) denote our answer-centered reasoning paths.
Angle brackets mark candidate answers (e.g., <Bishop> and
<Q>); (O2) and (O3) group paths under these candidates, re-
spectively.
reasoning. A second group [ 14,18] adopts lightweight retrievers
such as graph neural networks (GNNs) to extract relevant knowl-
edge. These methods are more efficient than LLM–based retrieval
and naturally exploit graph structure, making them less vulnerable
to hallucination and suitable for complex multi-hop reasoning.
However, these approaches still face two fundamental challenges.
First, their path sampling method for training data construction is
typically semantics-unaware, relying on shortest-path heuristics
(e.g., BFS) that often select paths inconsistent with a query’s intent.
These irrelevant paths are then used as supervision signals, causing
models to internalize misaligned reasoning patterns. Consequently,
models tend to prioritize topological proximity along paths rather
than semantic relevance to the query.
Second, there exists a weak alignment between the retrieved
paths and the reasoning objectives of LLMs. While large-scale mod-
els (e.g., GPT-4) can partially offset retrieval noise using their exten-
sive parametric knowledge, small LLMs (1–8B parameters) are far
more sensitive to both retrieval noise and ungrouped path evidence
due to their limited reasoning capacity. As retrieved paths are pre-
sented as flat lists rather than answer-centered reasoning paths,
small models receive little guidance for integrating them into coher-
ent reasoning. Moreover, the training objective focuses solely on
predicting the final answer from consecutive paths, which neglects
explicit supervision of intermediate step-by-step reasoning. Thismismatch weakens supervision and leads to unstable reasoning
performance.
Figure 1 illustrates how unordered retrieval supervision distorts
reasoning behavior—small LLMs fail to compose semantically rele-
vant paths even when the correct evidence exists in the KG. The
query asks for the character inJuicewho is also the music producer
ofThe Don Killuminati: The 7 Day Theoryand the answer isBishop.
To satisfy both constraints, the reasoning process must identify
the person who serves as the bridge between the two domains —
the album (through the producer relation) and the movie (through
theperformance relation) — thereby linking both domains into a
coherent reasoning path. Prior prompts provide retrieved KG paths
as an ungrouped list (e.g., path (B2)), which are topologically close
to the topic entityJuicebut ignore the album constraint. Such un-
organized evidence biases the model toward frequent but incorrect
entities such asQ. In contrast, our approach reconstructs retrieved
paths into answer-centered reasoning paths that preserve semantic
intent (e.g., path (O2)). This representation enables small LLMs to
reason over coherent evidence rather than isolated fragments.
Building on this idea, we proposeRPO-RAG(Relation-aware
weightedPreferenceOptimization for RAG), a framework tailored
to small LLMs. Unlike prior works, RPO-RAG refines supervision
signals across the entire retrieval–reasoning pipeline to better align
with the structure of KGs. First, we introduce a query-path se-
mantic sampling strategy that replaces heuristic path construction
with a similarity-based approach. By selecting reasoning paths
based on semantic consistency with the query, RPO-RAG builds a
high-quality training dataset that provides precise supervision for
both retriever and reasoner. Second, RPO-RAG employs a relation-
aware weighted preference optimization objective that explicitly
supervises intermediate reasoning at the relation level. This allows
small LLMs to learn fine-grained relational preferences and align
their reasoning process with the query intent. Finally, we design
an answer-centered prompt that organizes retrieved entities and
supporting paths into coherent reasoning contexts, enabling small
LLMs to focus on relevant evidence more effectively.
The main contributions of the work are as follows:
•We propose RPO-RAG, a KG-based RAG framework tailored
to small LLMs. It introduces a query-path semantic sampling
strategy that automatically identifies semantic relevance between
queries and reasoning paths, enabling the construction of high-
quality training dataset with refined supervision signals.
•RPO-RAG explicitly models the reasoning process of LLMs at the
relation level. To the best of our knowledge, it is the first frame-
work that incorporates relations into preference optimization for
KG-based RAG.
•Experimental results on KGQA benchmarks show that our frame-
work consistently outperforms existing methods and, in partic-
ular, substantially narrows the performance gap of small-scale
LLMs compared to larger-scale baselines.
2 Related Work
2.1 Preference Optimization
Preference optimization (PO) is a training paradigm that optimizes
a model’s generation with human preferences by comparing pairs

RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for KGQA WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
Q: What countries in EU are located in the Western European Time Zone ?
Semantic Retriever
Input Path
Gradient-based 
Dynamic Clustering 
 = !
Output Cluster"
"Query-Path 
Semantic Sampling
Retrieved Paths
Dual-objective Optimization
Query-aligned Paths
RPO modelSemantic-matching based 
Path Retrieval
䞱"OTXFS $FOUFSFE䞱1SPNQU䞱0QUJNJ[BUJPO
Based on the Answer-Centered-Paths, please answer 
the given question. 
…
Answer-Centered-Paths:
< Spain >
EU → organization.membership_organization.
members → … → Spain 
EU → organization.geographic_scope → … → Spain
Western European Time Zone → … →  Spain 
Question: What countries in EUare located in the 
Western European Time Zone ?
Answer: [ Spain, Portugal ]䞱3FMBUJPO BXBSF 1SFGFSFODF䞱0QUJNJ[BUJPO
Question: 
What countries in EU are located in the Western 
European Time Zone?
Current Path:
organization.organization.geographic_scope
NEXT RELATION
1SFGFSSFE䞱3FMBUJPOT
/PO QSFGFSSFE䞱3FMBUJPOT
Figure 2: Overview of the RPO-RAG framework. (1) Query-Path Semantic Sampling: constructs query-aligned training paths via
dynamic clustering to capture query intent. (2) Semantic-Matching Retriever: retrieves reasoning paths semantically consistent
with the query using a pretrained language model. (3) Dual-Objective Optimization: optimizes relation-level preference and
answer-centered prompt objectives to align small LLMs with structured reasoning.
of responses, typically consisting of a preferred (𝑦+)and a non-
preferred(𝑦−). Early approaches are grounded in Reinforcement
Learning from Human Feedback (RLHF) [ 3,20]. In this setting, a
supervised model is first trained, followed by a separate reward
model learned from human-annotated preferences. While effective,
this two-stage pipeline suffers from instability and inefficiency,
as reinforcement learning is computationally intensive and slow.
To improve efficiency, Direct Preference Optimization (DPO) [ 23]
directly optimizes the model such that the probability of generat-
ing𝑦+exceeds that of 𝑦−, removing the need for a reward model.
However, DPO typically requires a costly reference model to pre-
vent policy collapse. SimPO [ 19] further enhances efficiency by
discarding the reference model and employing length correction to
mitigate response bias.
Despite these advances, existing PO methods [ 12,20,23,27] have
been exclusively applied to high-level text generation tasks such as
dialogue or summarization, relying heavily on human-annotated
data. In contrast, we present an approach that extends preference
optimization to relation-level supervision in KGQA. Specifically,
we optimize the probability of generating the next relation condi-
tioned on partial paths, aligning the model with the query’s intent.
Crucially, we construct preference pairs via a weakly supervised
method based on semantic relevance of sampled paths, avoiding
reliance on costly manual annotations.
2.2 KG-based RAG
Integrating KGs into RAG has emerged as a promising strategy to
mitigate hallucination in knowledge-intensive tasks. Early workssuch as SR [ 37], NSM[ 8], and UniKGQA [ 11] leverage pretrained lan-
guage models (PLMs) or graph neural networks (GNNs) to retrieve
question-relevant subgraphs without relying on LLMs in the rea-
soning phase. These methods achieve fast retrieval and reasoning
efficiency, but struggles with compositional multi-hop reasoning
that requires deeper semantic understanding.
Building on this line, recent KG-based RAG methods can be
categorized by whether LLMs are directly involved in the retrieval
stage. Some works [ 16,17] leverage the generative capacity of
LLMs to plan or retrieve knowledge from KGs. RoG [ 16] employs
a planning-retrieval-reasoning pipeline where the LLM proposes
candidate relation sequences later grounded in the KG. Similarly,
GCR [ 17] introduces KG-Trie as a constraint to enforce relation
ordering during path extraction, thereby reducing hallucinations.
While these methods leverage the reasoning ability of LLMs, they
also incur substantial inference overhead due to the complexity of
LLM-based path generation.
To improve efficiency, other approaches avoid LLM inference
during retrieval. SubgraphRAG [ 14] adopts an MLP-based parallel
triple extraction scheme with distance-based encoding, effectively
capturing entities near the topic entity but struggling with multi-
hop reasoning. GNN-RAG [ 18] instead frames KGQA as a node
classification task, first predicting candidate answers via GNN prop-
agation and then extracting paths to those predicted candidates.
Despite these advances, both lines of work share notable limita-
tions: reliance on shortest-path heuristics that ignore query-path

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, and Kyong-Ho Lee
semantics, and limited modeling of intermediate relational reason-
ing caused by flat, ungrouped prompt design, which is particularly
detrimental for smaller LLMs.
Within this landscape, our framework aligns with efficiency-
oriented approaches by employing a PLM-based semantic retriever.
Its novelty lies in introducing adaptive, semantics-aware path sam-
pling and relation-level optimization.
3 Method
RPO(Relation-awarePreferenceOptimization)-RAGis a novel
framework designed to enhance the reasoning capabilities of small
LLMs in the KG-based RAG paradigm. The framework adopts an
end-to-end retrieval-and-reasoning pipeline that refines learning
signals across the system. It consists of three main components.
(1) Query-Path Semantic Samplingconstructs a high-fidelity
dataset that captures query intent and provides supervisory signals
for both retriever and reasoner.(2) Semantic-Matching Retriever
trained on this sampled dataset employs dynamic beam search to
efficiently extract semantically relevant reasoning paths. Finally,(3)
Dual-Objective Optimizationintegratesrelation-aware relevance-
weighted preference optimizationwithanswer-centered prompt op-
timization, substantially improving the reasoning ability of small
LLMs. The overall architecture is illustrated in Figure 2.
3.1 Query-Path Semantic Sampling
The purpose of query-path semantic sampling is to construct train-
ing data that captures query intent, providing reliable supervision
for both retriever and reasoner. Prior studies that rely on shortest-
path heuristics (e.g., BFS) often include semantically irrelevant paths
and fail to reflect the varying reasoning semantics across queries. In
contrast, our approach dynamically identifies query-aligned paths
by leveraging a gradient-based dynamic clustering, as illustrated in
Figure 3. This process allows the model to automatically adjust to
varying query complexities, rather than depending on pre-defined
sampling rules.
Formalization and Procedure
Given a query 𝑞, we denote the topic entity by 𝑒𝑞and the answer
entity by𝑒𝑎. The set of candidate paths connecting them is defined
as:
𝑃={𝑝|𝑝=(𝑒 𝑞;𝑟1;𝑟2;...;𝑒𝑎)}(1)
Candidate paths are initially obtained by enumerating all possible
shortest paths between 𝑒𝑞and𝑒𝑎in KG. Each query and path is
embedded with a PLM, producing vectors ℎ𝑞=𝑃𝐿𝑀(𝑞) andℎ𝑝=
PLM(𝑝) . The semantic relevance between a query and a path is
measured by cosine similarity:
𝑠(𝑞,𝑝)=𝑠𝑖𝑚(ℎ 𝑞,ℎ𝑝)(2)
To capture varying degrees of relevance, we apply a gradient-based
dynamic clustering algorithm [ 25] that partitions paths according
to their similarity distribution. This method adaptively determines
the optimal number of clusters 𝑁by identifying inflection points
where the inter-cluster similarity differences are maximized. Con-
cretely,𝑁is determined at the point of maximum curvature in the
similarity–cluster curve, effectively detecting semantic boundaries
among reasoning paths. From the resulting clusters, the represen-
tative clusterC∗whose embedding 𝑐𝑘is most similar to the query
Q: Which languages are 
spoken at the location, 
where the film, “Shutter" , 
occurs?
Output Cluster
Input PathGradient-based Dynamic Clustering  film.film.featured_film_locations → 
location.country .lanuages_spoken
sports.fight_song.sports_team
Inflection Point3Figure 3: An example of Query-Path Semantic Sampling.
embedding is selected:
C∗=arg max
𝐶𝑘sim(ℎ𝑞,𝑐𝑘)(3)
The final high-fidelity training set is then defined as:
ˆ𝑃(𝑞,𝑒𝑞,𝑒𝑎)={𝑝∈𝑃|𝑝∈C∗}(4)
Example
Figure 3 illustrates the procedure with an example. Given the query
“Which languages are spoken at the location where the film ‘Shutter’
occurs?”, the topic entity isShutterand the answer entity isThai
language. Initially, all shortest paths between these entities are
enumerated, including both semantically relevant and irrelevant
ones, such as:
•film.film.featured_film_locations→location.country.
languages_spoken(relevant)
•film.film.subjects→sports.fight.song.sports_team
(irrelevant)
The gradient-based clustering then identifies an inflection point at
𝐾=3, grouping paths by semantic similarity. The selected cluster
(rightmost in Figure 3) retains only those paths that align with
the query intent, filtering out irrelevant connections. This process
enables the training data to capture the precise reasoning semantics
of each query and reflects reasoning complexity, providing cleaner
and more consistent supervision for both the retriever and the
reasoner. We further verify the quality improvement of the sampled
data in Section 4.5.
3.2 Semantic-Matching based Path Retrieval
The retriever is designed to identify reasoning paths that align
semantically with the query intent and to deliver them effectively
to the reasoning module. It is trained on the dataset obtained from
query-path semantic sampling (Section 3.1), enabling it to encode
contextual information that strengthens query-to-path matching.
Retrieval Scheme
Following SR [ 37], we pretrain a PLM using weak supervision from
(𝑞,𝑎) pairs. The model learns to predict correct relations along
query-answer paths while minimizing sampled negatives, thereby
capturing semantic alignment between questions and reasoning
paths.
At inference time, the retriever computes semantic similarities
between the query and the candidate paths based on aggregated
scores. The probability of expanding a relation 𝑟𝑖given the current
query𝑞and partial path𝑝 𝑡−1is formalized as [37]:
𝑝(𝑟|𝑞(𝑡))=1
1+exp(𝑠(𝑞(𝑡),𝐸𝑁𝐷)−𝑠(𝑞(𝑡),𝑟)),(5)

RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for KGQA WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
where ENDdenotes a virtual relation representing the termination
of path expansion.
Efficient and Noise-Resistant Path Filtering
Instead of expanding a fixed number of paths as in SR, we employ a
dynamic beam search. The expansion size is dynamically adjusted
by a threshold value that reflects the gap between similarity scores,
where larger gaps trigger pruning of less relevant candidates. This
adaptive design prevents redundant path expansion and ensures
retrieval that is both efficient and semantically faithful.
To filter out paths irrelevant to the query intent, we constrain
the retrieval phase using entity type information from the KG.
Structured KGs like Freebase [ 2] and DBpedia [ 1] provide rich type
schemas, and answer entity type prediction helps to identify paths
that do not meet the query intent. Firstly, for each entity 𝑒∈E ,
we retrieve a set of entire entity types 𝑇𝑢=
𝜏1,𝜏2,...,𝜏|𝑇𝑢|	
from
the schema. Afterwards, retrieved entity types are labeled as 1 if
connected with answer entities, otherwise 0. We train our model
with those labels to predict the answer entity types. Specifically,
we define a relevance score between each question 𝑞and type𝜏as:
ˆ𝑚𝑞,𝜏=𝑅𝑒𝐿𝑈(W 𝑞h𝑞×W𝜏h𝜏),(6)
where W𝑞,W𝜏are projection matrices, and h𝑞,h𝜏are embeddings
of𝑞and𝜏, respectively. Finally, we optimize the following cross-
entropy loss to predict top- 𝐾answer entity types for each question:
L𝑇𝑦𝑝𝑒=−∑︁
𝑞∈𝑄∑︁
𝜏∈𝑇𝑞(𝑚𝑞,𝜏log( ˆ𝑚𝑞,𝜏)+
(1−𝑚𝑞,𝜏)log(1− ˆ𝑚𝑞,𝜏)),(7)
where𝑚𝑞,𝜏is a labeled entity type score. After training, we use the
top-5 prediction result to exclude paths whose terminal entity is
inconsistent with the predicted types.
3.3 Dual-Objective Optimization
Relation-aware Weighted Preference Optimization
A key novelty of our framework is, to our knowledge, the first appli-
cation of preference optimization at the relation level for knowledge
graph reasoning. While prior works mainly focus on optimizing
answers or paths as whole units, we introduce a fine-grained ob-
jective that supervises the inference of intermediate relations. This
design explicitly guides small LLMs to reason over structured re-
lation sequences step by step, rather than only focusing on end
entities. As illustrated in Figure 4, our model learns to prefer re-
lations semantically consistent with the query context. As shown
in the example, these relation-level preference signals encourage
step-by-step, semantically grounded reasoning.
Since the importance of each relation varies across queries, we
assign adaptive confidence weights based on semantic relevance. To
construct preference signals, we first identify positive and negative
relation sets. Relations from the representative cluster C∗obtained
during path sampling are regarded as preferred responses 𝑌+, while
relations from alternative clusters are treated as non-preferred re-
sponses𝑌−. This weakly supervised construction enables prefer-
ence pairs without annotation.
Each relation is weighted according to its semantic proximity to
the cluster centroid. Preferred relations closer to the centroid re-
ceive higher confidence, while non-preferred relations farther away
!:".# $
%:".& '
CandidateRelationsQuestion: 
Whatcountries in EU are located in the Western
European Time Z one?
Current Path:
EU →organization.membership_organization.members
Inputlocation.location.containedby CHOSEN
military .military_combatant.includes_allies REJECT  
Figure 4: Illustration of Relation-aware Weighted Prefer-
ence Optimization. For the same question and current path
(left), candidate next relations (right) are scored by semantic
relevance. Higher-scored relations are treated as preferred
(CHOSEN ), while semantically misaligned ones are treated as
non-preferred (REJECT).
are penalized. Formally, letting 𝑢(𝑦) denote the centroid distance
and𝛼>0a decay rate, we define:
𝑠(𝑦)=(
𝑒−𝛼𝑢(𝑦)if𝑦∈𝑌+(preferred)
1−𝑒−𝛼𝑢(𝑦)if𝑦∈𝑌−(non-preferred)(8)
These confidence scores are transformed into weights (𝑤+,𝑤−)
with a scaling factor𝛽>0:
𝑤=𝛽 1+0.5(𝑠−0.5)(9)
Training then proceeds with a margin-based preference objective:
L(𝜋𝜃)=−Eh
log𝜎 𝑊+log𝜋𝜃(𝑦+|𝑥)−𝑊−log𝜋𝜃(𝑦−|𝑥)−𝛾i
,
(10)
where𝑊+=𝑤+/|𝑦+|and𝑊−=𝑤−/|𝑦−|are normalized by rela-
tion length, and 𝛾is a margin term. By explicitly applying relation-
level preference optimization with relevance-weighted signals, our
approach introduces a new training paradigm for KGQA. This en-
ables small LLMs to internalize structured, step-by-step relational
reasoning and improve their capacity to execute faithful, inter-
pretable inference over KGs.
Answer-Centered Prompt Optimization
The second objective complements relation-level training by align-
ing answer generation with answer-centered reasoning paths.
Prompts are explicitly designed to incorporate multiple reasoning
paths as evidence supporting candidate answers. Unlike conven-
tional prompts that treat each path in isolation and fail to integrate
information from multiple topic entities, our proposed prompt uni-
fies these paths into a coherent representation better suited for small
LLMs. As illustrated in Figure 1, we adopt ananswer-centeredlayout:
paths are grouped by their end entities, and all paths supporting the
same candidate are presented together. For multi-topic questions,
this structure naturally merges reasoning paths originating from
different topics but converging on the same answer, allowing small
LLMs to aggregate consistent evidence more effectively.
Formally, we maximize the likelihood of the correct answer
conditioned on the answer-centered prompt:
Lans=−E(𝑞,𝑎) log𝑃𝜃(𝑒𝑎|𝑥answer-centered)(11)
where𝑥answer-centered denotes the grouped (answer-centered) prompt.
This optimization not only improves answer accuracy but also en-
hances interpretability, as predictions are explicitly tied to trans-
parent reasoning paths.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, and Kyong-Ho Lee
Table 1: Main results on WebQSP and CWQ. RPO-RAG achieves strong improvements among small LLMs ( ≤8B) and significantly
narrows the gap with GPT-based models.†Reproduced using the official implementation released by the original authors.
Type MethodsWebQSP CWQ
Hit F1 Hit F1
GNN-basedGraftNet 66.7 62.4 36.8 32.7
NSM 68.7 62.8 47.6 42.4
SR + NSM 68.9 64.1 50.2 47.1
UniKGQA 77.2 72.2 51.2 49.1
Vanilla LLMLlama2-7B 56.4 36.5 28.4 21.4
Llama3.1-8B 55.5 34.8 28.1 22.4
Llama3.2-3B 62.5 32.2 19.9 14.4
Llama3.2-1B 50.2 28.9 14.3 11.2
KG + LLMRoG (Llama2-7B) 85.7 70.8 62.6 56.2
GNN-RAG (Llama2-7B) 85.7 71.3 66.8 59.4
GNN-RAG (Llama3.1-8B†) 85.6 70.3 66.6 58.0
SubgraphRAG (Llama3.1-8B) 86.6 70.6 57.0 47.2
GCR (Llama3.1-8B†) 87.2 69.1 60.5 50.4
RPO-RAG (Llama2-7B) 88.3 77.8 68.1 59.2
RPO-RAG (Llama3.1-8B)89.9 81.3 72.3 64.5
RPO-RAG (Llama3.2-3B) 87.3 76.4 66.0 57.3
RPO-RAG (Llama3.2-1B) 82.3 69.8 60.3 50.4
4 Experiments
In this section, we presented experimental results and analyses to
evaluate the effectiveness of our proposed RPO-RAG framework.
We aimed to answer the following research questions:
RQ1:How effectively and efficiently does RPO-RAG improve over-
all KGQA reasoning performance?
RQ2:Can small LLMs in RPO-RAG narrow the reasoning gap with
large-scale models (e.g., GPT-4o-mini)?
RQ3:How do the individual components contribute to the overall
performance?
RQ4:Does proposed data sampling method enhances data quality
and retriever supervision?
4.1 Experimental Settings
Datasets.We evaluated RPO-RAG on two widely used KGQA
benchmarks, WebQuestionsSP (WebQSP) [ 36] and Complex We-
bQuestions (CWQ) [ 30], both grounded in Freebase [ 2]. WebQSP
mainly contains relatively simple 1–2 hop questions, whereas CWQ
includes more complex ones requiring up to 4-hop reasoning. De-
tailed dataset statistics are provided in Appendix B.1.
Implementation Details.We utilized Sentence-BERT [ 24] as a
default PLM for path retrieval, trained using query–path semantic
sampling. During inference, dynamic beam search is applied with
adaptive thresholds for WebQSP and CWQ. The reasoner employs
small LLMs—Llama2-7B [ 31], Llama3.1-8B [ 5], Llama3.2-3B1, and
Llama3.2-1B2—fine-tuned with LoRA on 2 ×NVIDIA RTX 4090
GPUs. More implementation details are available in Appendix B.2.
Baselines. We compared RPO-RAG against representative mod-
els from three categories: (1)Graph-based reasoning methods— GraftNet [ 28], NSM [ 8], SR+NSM [ 37], and UniKGQA [ 11]; (2)
LLM-only methods— vanilla small and medium LLMs, including
Llama2–7B [ 31], Llama3.1–8B [ 5], Llama3.2-3B1, and Llama3.2-1B2;
and (3)LLM+KG methods— ToG [ 29], RoG [ 16], GCR [ 17], Sub-
graphRAG [ 14], and GNN-RAG [ 18]. Detailed descriptions for all
baselines are provided in Appendix B.3.
Evaluation Metrics.We adopted Hit and F1 as evaluation metrics.
Hit measures whether at least one correct answer appears in the
predicted set, while F1 balances precision and recall to provide a
comprehensive assessment of answer quality.
4.2 Main Results (RQ1)
Effectiveness
Table 1 presents the overall performance comparison on WebQSP
and CWQ datasets. Our proposed RPO-RAG consistently outper-
formed all small-scale and graph-based baselines, demonstrating
the effectiveness of relation-aware and answer-centered prompt
optimization.
On WebQSP, RPO-RAG (Llama3.1–8B) achieves 89.9 Hit and
81.3 F1, establishing a new state-of-the-art (SOTA) among all mod-
els up to 8B parameters—surpassing the previous best (GCR) by
+2.7% Hit and+10.2% F1. RPO-RAG (Llama2–7B) also delivers the
second-best overall result with 88.3 Hit and 77.8 F1, confirming
the framework’s robustness across architectures. Notably, even the
small-scaled RPO-RAG (Llama3.2–3B) exceeds the performance of
several larger baselines, demonstrating that RPO-RAG effectively
transfers structured reasoning capability to small LLMs.
1https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for KGQA WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
On the more challenging CWQ dataset, which requires multi-
hop reasoning paths, RPO-RAG achieved consistent gains across
both Hit and F1. Specifically, RPO-RAG (Llama2–7B) improved Hit
+1.3% and F1+4.9% over GNN-RAG (Llama2–7B), while RPO-RAG
(Llama3.1-8B) reached the highest Hit and F1 among all≤8B base-
lines. These gains highlight that RPO-RAG not only identifies more
correct answers but also sustains better coverage and compositional
reasoning across hops.
Beyond these strong relative improvements, RPO-RAG also de-
livers substantial absolute gains over vanilla LLMs without any ex-
ternal reasoning mechanism. For instance, RPO-RAG (Llama3.2–3B)
improves Hit by+24.8% on WebQSP and +46.1% on CWQ compared
to the base Llama3.2–3B, while RPO-RAG (Llama3.2–1B) yields
+32.1% and+46% Hit gains over its vanilla counterpart. These
results confirm that our framework effectively injects structured
reasoning capability into small LLMs, even when model capacity
is highly constrained. Overall, RPO-RAG substantially enhances
the reasoning reliability and scalability of small open-weight LLMs
across both benchmarks.
Efficiency
To assess the efficiency of RPO-RAG, we measured average per-
query retrieval and reasoning times on the CWQ, a benchmark
that requires complex multi-hop reasoning. Results in Table 2 were
obtained on a single NVIDIA RTX 3090 GPU.
For the retrieval stage, existing methods adopt varying strate-
gies. GNN-RAG and SubgraphRAG use lightweight neural retriev-
ers—graph propagation with a GNN and an MLP-based embedding
similarity scorer (SBERT embeddings), respectively. GCR, in con-
trast, relies on an LLM to explicitly generate candidate reasoning
paths. RPO-RAG employs a compact PLM trained on semantically
sampled data, which efficiently encodes query–path relevance. On
the reasoning side, GNN-RAG and RPO-RAG use fine-tuned LLMs,
whereas SubgraphRAG and GCR use vanilla LLMs without task-
specific adaptation.
Among the compared methods, RPO-RAG offers the best accu-
racy–latency balance. SubgraphRAG attains the fastest retrieval but
suffers from the slowest end-to-end time and the lowest Hit. GCR
shows the opposite pattern: LLM-based retrieval dominates latency.
GNN-RAG is the fastest overall but falls behind in accuracy. By
contrast, RPO-RAG keeps retrieval lightweight and achieves near-
minimal total latency while delivering the highest Hit, providing
the strongest accuracy–efficiency trade-off on CWQ.
4.3 Competitiveness of Small LLMs (RQ2)
This section evaluates whether the proposed RPO-RAG framework
enables small LLMs to approach the reasoning capability of GPT-
based models. Table 3 summarizes the comparison results across
both datasets. Overall, RPO-RAG substantially boosts the reasoning
performance of small LLMs, closing the gap with large proprietary
LLMs while maintaining a fraction of their parameter scale.
On the relatively simple WebQSP dataset, which mostly in-
volves up to 2-hop reasoning, even the smallest variant, RPO-RAG
(Llama3.2–1B), achieves a Hit score of 82.3, surpassing ToG (Chat-
GPT) by (+6.1%). This demonstrates that relation-aware preference
optimization allows a 1B-parameter model to reach GPT-level accu-
racy despite its limited capacity.Table 2: Efficiency and Hit comparison of different methods
on CWQ. Average per-question wall-clock time (seconds). Hit
(↑) is higher-is-better; Retrieval, Reasoning, and Total ( ↓) are
lower-is-better.
Methods Hit Retrieval Reasoning Total
GNN-RAG (Llama2-7B) 66.8 0.11 0.951.06
SubgraphRAG (Llama3.1-8B) 57.0 0.02 6.12 6.14
GCR (Llama3.1-8B) 60.5 6.84 0.56 7.4
RPO-RAG (Llama2-7B) 68.10.071.36 1.43
RPO-RAG (Llama3.1-8B) 72.31.1 1.17
On the more challenging CWQ dataset, which requires up to 4-
hop reasoning, RPO-RAG (Llama3.1–8B) achieves 72.3 Hit and 64.5
F1, approaching the performance of GCR (ChatGPT) and reducing
the gap to GCR (GPT-4o-mini) to within roughly 3–4 points. These
results collectively confirm that RPO-RAG successfully transfers
reasoning consistency and compositional understanding to smaller
LLMs. By aligning intermediate reasoning through relation-level
preference optimization, our framework enables small, open-weight
models to achieve GPT-comparable reasoning accuracy under real-
istic computational constraints.
Table 3: Comparison of RPO-RAG with GPT-based models
on WebQSP and CWQ.
MethodsWebQSP CWQ
Hit F1 Hit F1
ToG (ChatGPT) 76.2 – 57.6 –
ToG (GPT-4) 82.6 – 68.5 –
GCR (ChatGPT) 92.6 73.2 72.7 60.9
GCR (GPT-4o-mini) 92.2 74.1 75.8 61.7
SubgraphRAG (GPT-4o-mini) 90.1 77.5 62.0 54.1
RPO-RAG (Llama2–7B)88.3 77.8 68.1 59.2
RPO-RAG (Llama3.1–8B)89.9 81.3 72.3 64.5
RPO-RAG (Llama3.2–3B)87.3 76.4 66.0 57.3
RPO-RAG (Llama3.2–1B)82.3 69.8 60.3 50.4
4.4 Ablation Study (RQ3)
Table 4 shows the ablation results of RPO-RAG.w/o Relation-aware
Optimizationremoves the relation-level preference optimization,
training the model only with answer-centered prompt.w/o Answer-
Centered Promptomits the answer-centered prompt design, instead
providing flat lists of retrieved paths.
The results clearly show that both components are essential
to the framework’s overall performance. Removing either compo-
nent consistently degrades both Hit and F1 across datasets and
model sizes. Relation-aware optimization contributes more signifi-
cantly on WebQSP, suggesting its importance for precise reasoning
alignment, while the answer-centered prompt yields larger gains
on CWQ, reflecting its effectiveness for multi-hop compositional
reasoning.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, and Kyong-Ho Lee
Table 4: Ablation study on WebQSP and CWQ. Removing
either component leads to consistent drops in both Hit and
F1.
MethodWebQSP CWQ
Hit F1 Hit F1
RPO-RAG (Llama2-7B) 88.3 77.8 68.1 59.2
w/o Relation-aware Optimization 81.6 66.5 61.1 54.2
w/o Answer-Centered Prompt 80.3 65.7 58.6 47.5
RPO-RAG (Llama3.2-3B) 87.3 76.4 66.0 57.3
w/o Relation-aware Optimization 78.8 63.4 58.1 55.9
w/o Answer-Centered Prompt 78.1 62.7 57.3 46.6
Table 5: Comparison of relation coverage between RoG-cwq
and our dataset.
Dataset Precision Recall F1
RoG-cwq 31.5 35.2 30.8
Ours-cwq39.2 35.3 35.1
4.5 Dataset Quality Analysis (RQ4)
In this section, we evaluated the quality of the dataset constructed
using our semantic query–path sampling strategy (Section 3.1). To
assess its effectiveness, we compared against the dataset released by
RoG [ 16], which has been widely adopted in prior studies [ 16–18].
Our analysis covered three dimensions: relation coverage, semantic
alignment, and improvements in retriever performance.
Relation Coverage Analysis
We defined ground-truth relations from the annotated SPARQL
queries and evaluate how well each dataset reflects the intended
semantics of the questions. Specifically, we computed precision,
recall, and F1 by comparing the overlap between the ground-truth
relation set and the relations contained in the training data. As
shown in Table 5, our dataset maintains comparable recall while
improving precision by +7.7%, leading to a+4.3% increase in F1.
These results demonstrate that our sampling strategy effectively
filters out noisy paths and preserves semantically aligned relations,
thereby providing higher-quality supervision for training.
Semantic Alignment Analysis
To evaluate semantic consistency, we computed cosine similarity
between query and candidate path embeddings using SBERT. For
each query, we reported average similarity scores under thetop-1
andtop-3settings. As shown in Figure 5, whiletop-1results are
comparable between datasets, the gap widens to 0.04 attop-3. More-
over, the similarity drop fromtop-1totop-3is smaller in our dataset
(0.03 vs. 0.05 for the baseline). This demonstrates that our strategy
maintains stronger semantic alignment with questions even as can-
didate paths are expanded, highlighting its robustness.
Retriever Performance Evaluation
Building on the previous data-quality verification, we evaluated
how the sampled dataset enhances retriever performance by provid-
ing stronger supervision signals. As shown in Table 6, our retriever
0.39
0.350.41
0.39
0.340.360.380.40.42
Top-1 Top-3RoG
OursMean Alignment ScoreFigure 5: Comparison of query–path semantic alignment
between datasets under top-1 and top-3 settings.
Table 6: Retriever performance comparison showing accu-
racy and average retrieved paths (ARP).
Model Accuracy (%) ARP
RoG 69.6 27
GNN-RAG 65.0 20
Ours 87.4 116
achieved a substantial accuracy improvement of +22.4%over GNN-
RAG, demonstrating that query-aligned supervision leads to more
faithful path retrieval. Although the average number of retrieved
paths (ARP) increases, this reflects broader yet semantically con-
sistent coverage rather than redundancy. In contrast to shortest-
path-based retrievers, which tend to miss relevant evidence beyond
local neighborhoods, our semantically guided retriever expands
reasoning coverage while maintaining precision. Overall, these
findings indicate that the proposed dataset enables retrievers to
move beyond brittle shortest-path heuristics toward semantically
guided retrieval, delivering higher accuracy.
5 Conclusion
We presentedRPO-RAG, a KG-based RAG framework tailored
to small LLMs (≤8B) that (i) samples query–aligned reasoning
paths via semantic clustering, (ii) applies relation-aware, relevance-
weighted preference optimization to supervise intermediate steps,
and (iii) reconstructs prompts to aggregate evidence along answer-
centered paths. Across WebQSP and CWQ, RPO-RAG consistently
improves Hit and F1 over graph-based and small-LLM baselines,
while markedly narrowing the gap to GPT-based systems. No-
tably, even 1B–3B backbones show substantial gains, indicating
that relation-level supervision and answer-centered prompting ef-
fectively transfer compositional reasoning to compact models.
Our dataset-quality analyses further show that semantics-aware
sampling yields higher-quality supervision and more faithful re-
trieval. Taken together, these findings suggest a practical path to
scalable, resource-efficient KGQA: align retrieval and reasoningat
the relation leveland deliver evidence in a form that small LLMs
can exploit. A more detailed discussion of additional limitations
and future research directions is provided in Appendix A.

RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for KGQA WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
References
[1]Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak,
and Zachary Ives. 2007. Dbpedia: A nucleus for a web of open data. Ininternational
semantic web conference. Springer, 722–735.
[2]Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor.
2008. Freebase: a collaboratively created graph database for structuring human
knowledge. InProceedings of the 2008 ACM SIGMOD international conference on
Management of data. 1247–1250.
[3]Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario
Amodei. 2017. Deep reinforcement learning from human preferences.Advances
in neural information processing systems30 (2017).
[4]Wentao Ding, Jinmao Li, Liangchuan Luo, and Yuzhong Qu. 2024. Enhancing
complex question answering over knowledge graphs through evidence pattern
retrieval. InProceedings of the ACM Web Conference 2024. 2106–2115.
[5]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models.arXiv e-prints(2024), arXiv–2407.
[6]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[7]Luke Friedman, Sameer Ahuja, David Allen, Zhenning Tan, Hakim Sidahmed,
Changbo Long, Jun Xie, Gabriel Schubiner, Ajay Patel, Harsh Lara, et al .2023.
Leveraging large language models in conversational recommender systems.arXiv
preprint arXiv:2305.07961(2023).
[8] Gaole He, Yunshi Lan, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen. 2021. Im-
proving multi-hop knowledge base question answering by learning intermediate
supervision signals. InProceedings of the 14th ACM international conference on
web search and data mining. 553–561.
[9]Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu,
Xinying Song, and Denny Zhou. 2023. Large language models cannot self-correct
reasoning yet.arXiv preprint arXiv:2310.01798(2023).
[10] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in
natural language generation.ACM computing surveys55, 12 (2023), 1–38.
[11] Jinhao Jiang, Kun Zhou, Xin Zhao, and Ji-Rong Wen. 2023. UniKGQA: Unified
Retrieval and Reasoning for Solving Multi-hop Question Answering Over Knowl-
edge Graph. InThe Eleventh International Conference on Learning Representations.
[12] Sungho Ko, Hyunjin Cho, Hyungjoo Chae, Jinyoung Yeo, and Dongha Lee. 2024.
Evidence-Focused Fact Summarization for Knowledge-Augmented Zero-Shot
Question Answering. InProceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing. 10636–10651.
[13] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke
Iwasawa. 2022. Large language models are zero-shot reasoners.Advances in
neural information processing systems35 (2022), 22199–22213.
[14] Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is Effective: The Roles of Graphs
and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented
Generation. InThe Thirteenth International Conference on Learning Representa-
tions.
[15] Ben Liu, Jihai Zhang, Fangquan Lin, Cheng Yang, Min Peng, and Wotao Yin.
2025. Symagent: A neural-symbolic self-learning agent framework for complex
reasoning over knowledge graphs. InProceedings of the ACM on Web Conference
2025. 98–108.
[16] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning
on Graphs: Faithful and Interpretable Large Language Model Reasoning. InThe
Twelfth International Conference on Learning Representations.
[17] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Yuan-Fang Li, Chen Gong, and
Shirui Pan. 2025. Graph-constrained Reasoning: Faithful Reasoning on Knowl-
edge Graphs with Large Language Models. InForty-second International Confer-
ence on Machine Learning.
[18] Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph Neural Retrieval
for Efficient Large Language Model Reasoning on Knowledge Graphs. InFindings
of the Association for Computational Linguistics: ACL 2025. 16682–16699.
[19] Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference
optimization with a reference-free reward.Advances in Neural Information
Processing Systems37 (2024), 124198–124235.
[20] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al .2022.
Training language models to follow instructions with human feedback.Advances
in neural information processing systems35 (2022), 27730–27744.
[21] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu.
2024. Unifying large language models and knowledge graphs: A roadmap.IEEE
Transactions on Knowledge and Data Engineering36, 7 (2024), 3580–3599.
[22] Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng,
Chuanqi Tan, Fei Huang, and Huajun Chen. 2023. Reasoning with Language
Model Prompting: A Survey. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 5368–5393.[23] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano
Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural information processing
systems36 (2023), 53728–53741.
[24] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP). 3982–3992.
[25] Ville Satopaa, Jeannie Albrecht, David Irwin, and Barath Raghavan. 2011. Finding
a" kneedle" in a haystack: Detecting knee points in system behavior. In2011 31st
international conference on distributed computing systems workshops. IEEE, 166–
171.
[26] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021.
Retrieval augmentation reduces hallucination in conversation.arXiv preprint
arXiv:2104.07567(2021).
[27] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea
Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to
summarize with human feedback.Advances in neural information processing
systems33 (2020), 3008–3021.
[28] Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhut-
dinov, and William Cohen. 2018. Open Domain Question Answering Using Early
Fusion of Knowledge Bases and Text. InProceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing. 4231–4242.
[29] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph: Deep
and Responsible Reasoning of Large Language Model on Knowledge Graph. In
The Twelfth International Conference on Learning Representations.
[30] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for
Answering Complex Questions. InProceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long Papers). 641–651.
[31] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas-
mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos-
ale, et al .2023. Llama 2: Open foundation and fine-tuned chat models.arXiv
preprint arXiv:2307.09288(2023).
[32] Xiaolei Wang, Kun Zhou, Xinyu Tang, Wayne Xin Zhao, Fan Pan, Zhao Cao,
and Ji-Rong Wen. 2023. Improving conversational recommendation systems via
counterfactual data simulation. InProceedings of the 29th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining. 2398–2408.
[33] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning
in large language models.Advances in neural information processing systems35
(2022), 24824–24837.
[34] Yike Wu, Nan Hu, Sheng Bi, Guilin Qi, Jie Ren, Anhuan Xie, and Wei Song. 2023.
Retrieve-rewrite-answer: A kg-to-text enhanced llms framework for knowledge
graph question answering.arXiv preprint arXiv:2309.11206(2023).
[35] Derong Xu, Xinhang Li, Ziheng Zhang, Zhenxi Lin, Zhihong Zhu, Zhi Zheng,
Xian Wu, Xiangyu Zhao, Tong Xu, and Enhong Chen. 2025. Harnessing large
language models for knowledge graph question answering via adaptive multi-
aspect retrieval-augmentation. InProceedings of the AAAI Conference on Artificial
Intelligence, Vol. 39. 25570–25578.
[36] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and
Jina Suh. 2016. The value of semantic parse labeling for knowledge base ques-
tion answering. InProceedings of the 54th Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers). 201–206.
[37] Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, and Hong
Chen. 2022. Subgraph Retrieval Enhanced Model for Multi-hop Knowledge Base
Question Answering. InProceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers). 5773–5784.
A Limitations and Future Work
While RPO-RAG substantially narrows the gap to larger systems,
there remains ample room for improvement on compositional
queries that require reasoning over attribute values. As shown
in Section C and Figure 8, our current pipeline does not explicitly
encode or propagate attribute-conditioned evidence, which leads to
a higher error rate on such query types. This limitation is amplified
for smaller backbones ( ≤3B), where the capacity to infer attribute-
dependent relations is further constrained. Future work includes
integrating attribute-aware retrieval and path construction (e.g.,
schema- or property-guided sampling), designing training signals
that supervise relation and attribute consistency, and developing

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, and Kyong-Ho Lee
prompt or adapter mechanisms that let small LLMs represent and
exploit attribute information more reliably.
B Experimental Details
B.1 KGQA benchmarks
We utilized two representative KGQA benchmarks: WebQuestion-
sSP (WebQSP) [ 36]and ComplexWebQuestions (CWQ) [ 30]. Both
datasets are grounded in the Freebase knowledge graph [ 2], and
their statistics are summarized in Table 7.
Table 7: Statistics of datasets used in experiments. The num-
ber of QA pairs for training and testing is reported along
with the maximum reasoning hop.
Dataset Train Test Max Hop
WebQSP 2,848 1,628 2
CWQ 27,639 3,531 4
B.2 Implementation Details
Query-Path Semantic Sampling
For embedding construction, we use theall-MiniLM-L6-v2(22.7M
params) as an encoder. All query and path embeddings are L2-
normalized, and cosine similarity is computed via inner product
between normalized vectors. To identify semantically coherent path
groups, we apply KMeans clustering, switching to MiniBatchK-
Means when the number of candidate paths exceeds 1,500. The
maximum number of clusters is set to 𝐾max=10, and the optimal
cluster count 𝑘is automatically determined using the elbow crite-
rion, selecting the point of largest inertia drop. All experiments are
performed with a fixed random seed of 42 to ensure reproducibility.
Semantic-Matching based Path Retrieval
We fine-tuned Sentence-BERT as a default retriever. For path ex-
pansion, we initialize beam width as 10. For dynamic beam search,
we set threshold as 0.3 for both WebQSP and CWQ. For type pre-
diction, we utilized gte-large-en-v1.5 for encoder. We used top-5
type prediction result for path filtering.
Relation-aware Weighted Preference Optimization
We fine-tune with LoRA on 2 ×NVIDIA RTX 4090 GPUs. The
detailed hyperparameters are listed in Table 8 (dataset-specific
LR/epochs for WebQSP vs. CWQ). For preference-based optimiza-
tion, we construct training data that encode question–path pairs,
along with preferred ( chosen ) and non-preferred ( rejected ) rea-
soning paths. Each example corresponds to a partial traversal in
the knowledge graph, where the model must predict the next re-
lation or terminate the path ( "STOP" ). An example of such data is
illustrated in Figure 6.
Answer-Centered Prompt Optimization
Using the same LoRA configuration, we fine-tune both WebQSP and
CWQ for 3 epochs with AdamW-8bit (lr =2 ×10−4), a cosine sched-
uler with warmup = 0.03. Training is conducted on 2 ×NVIDIA
RTX 4090 GPUs. The corresponding hyperparameters are also sum-
marized in Table 8.
{
 "prompt": "Question: What languag es are spoken in the country where Jamaican English is spoken?
 Start entity: Jamaican English
 Partial Path: languag e.human_languag e.countries_spoken_in
 Predict ONLY the next relation (schema name).
If the path should end now, output 'ST OP' .
Output just the relation string (or 'ST OP')." ,
 "chosen": [
 {
 "content": "location.country .languag es_spoken" ,
 "role": "assistant"
 } ],
 "rejected": [
 {
"content": "location.country .official_languag e" ,
"role": "assistant"
 } ]
}Figure 6: An example train data of Relation-aware Weighted
Preference Optimization
B.3 Baselines
We evaluated RPO-RAG against representative graph-reasoning
and KG-augmented LLM baselines.
Graph-based Methods
•GraftNet [ 28] introduces an early graph-neural approach for
KGQA that retrieves question-specific subgraphs via entity link-
ing and performs reasoning using a convolution-based GNN.
•NSM [ 8] models multi-hop reasoning as a sequential decision
process, predicting relations step-by-step over KGs.
•SR+NSM [ 37] extends NSM by proposing relation-path retrieval,
which selects relation sequences relevant to the question to con-
struct subgraphs for reasoning.
•UniKGQA [ 11] unifies graph retrieval and reasoning into a single
framework, employing language modeling to bridge symbolic
KG reasoning and text understanding.
LLM-only methods
•Llama2–7B [ 31] is a 7B-parameter open-weight model that serves
as a mid-scale backbone for instruction-tuned reasoning.
•Llama3.1–8B [ 5] is a recent 8B variant offering improved align-
ment and reasoning stability.
•Llama3.2–3B1is a compact 3B model balancing efficiency and
compositional reasoning capability.
•Llama3.2–1B2is the smallest model evaluated, used to assess
reasoning scalability under extreme parameter constraints.
KG+LLM Methods
•ToG [ 29] explores multiple reasoning paths over KGs and aggre-
gates evidence from them to generate faithful answers, balancing
structural exploration and LLM reasoning.
•RoG [ 16] introduces a planning–retrieval–reasoning pipeline
where an LLM generates candidate relation sequences that are
later grounded in the KG, guiding faithful reasoning.
•GCR [ 17] constrains relation ordering through a KG-Trie mech-
anism, enforcing structural consistency during path extraction
and reducing hallucinations in multi-hop reasoning.
•SubgraphRAG [ 14] encodes KG triples using a text encoder and
trains an MLP classifier to rank and select top- 𝑘triplets rele-
vant to the question, enabling structured retrieval without LLM
inference.

RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for KGQA WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
Table 8: Detailed hyperparameters for both training stages.
LoRA configs are shared across stages.
Hyperparameter Value
Relation-aware Preference Optimization
lora_r32
lora_alpha64
lora_dropout0.05
optimizerAdamW
warmup_ratio0.10
learning_rate (WebQSP)7.5×10−6
learning_rate (CWQ)1.0×10−6
schedulercosine
max_length2048
epochs (WebQSP)3
epochs (CWQ)1
answer-centered Prompt Optimization
lora_r32
lora_alpha64
lora_dropout0.05
optimizerAdamW
warmup_ratio0.03
learning_rate2.0×10−4
schedulercosine
max_length4096
epochs3
•GNN-RAG [ 18] replaces LLM-based retrieval with a lightweight
graph neural network that retrieves semantically relevant paths
while maintaining efficiency.
B.4 Answer-Centered Prompt Detailed
Figure 7 illustrates the structure of our proposedanswer-centered
prompt. In this design, the retrieved results are grouped according
to the candidate answer entities, allowing the model to focus on
reasoning over entity-specific evidence.
Structured Prompts 
Based on the Answer-Centered-Paths, please answer the given question.      
The Answer-Centered-Paths helps you to step-by-step reasoning to answer the question.
Let's think step by step. Return the most possible answers based on the given paths by listing each answer on a 
separate line.
Please keep the answer as simple as possible and return all the possible answers as a list.
Answer-Centered Paths:
< Candidate Answer Entity >
Retrieved Paths grouped by Candidate Answer Entity 
…
Question: 
Input QuestionAnswer-Centered PromptFigure 7: A template of Answer Centered Prompt
C Error Case
In Figure 8, the example illustrates a failure case where the model
does not account for attribute values. Predicting the correct answer
requires reasoning over the attribute values associated with the can-
didate entities. However, our current approach does not explicitly
incorporate such information, leading to an incorrect prediction.
This limitation becomes more pronounced in smaller LLMs ( ≤3B),
where the capacity to infer attribute-dependent relations is further
constrained.
Structured Prompts 
Question: Which team from the American League West was founded the most recently? 
Answer-Centered Paths:
<Los Angeles Angels of Anaheim>
American League West → baseball.baseball_division.teams→ Los Angeles Angels of Anaheim
American League West → sports.sports_league.teams→ m.0crtd2r → sports.sports_league_participation.team→ Los Angeles Angels of Anaheim
<Seattle Mariners>
American League West → baseball.baseball_division.teams→ Seattle Mariners
American League West → sports.sports_league.teams→ m.0crtd21 → sports.sports_league_participation.team→ Seattle Mariners
…
<Anaheim Angels>
American League West → sports.sports_league.teams→ m.0crtd31 → sports.sports_league_participation.team→ Anaheim Angels
RPO Responses: Seattle Mariners  
s  Ground Truth: Anaheim Angels WebQTrn-2152_52aec0171d93064e46b6e747d8d82010
Figure 8: Representative error case of RPO-RAG. The model
fails to generate the correct answer despite retrieving valid
reasoning paths.
1https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct