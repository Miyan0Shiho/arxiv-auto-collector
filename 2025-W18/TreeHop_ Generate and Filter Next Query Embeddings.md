# TreeHop: Generate and Filter Next Query Embeddings Efficiently for Multi-hop Question Answering

**Authors**: Zhonghao Li, Kunpeng Zhang, Jinghuai Ou, Shuliang Liu, Xuming Hu

**Published**: 2025-04-28 01:56:31

**PDF URL**: [http://arxiv.org/pdf/2504.20114v2](http://arxiv.org/pdf/2504.20114v2)

## Abstract
Retrieval-augmented generation (RAG) systems face significant challenges in
multi-hop question answering (MHQA), where complex queries require synthesizing
information across multiple document chunks. Existing approaches typically rely
on iterative LLM-based query rewriting and routing, resulting in high
computational costs due to repeated LLM invocations and multi-stage processes.
To address these limitations, we propose TreeHop, an embedding-level framework
without the need for LLMs in query refinement. TreeHop dynamically updates
query embeddings by fusing semantic information from prior queries and
retrieved documents, enabling iterative retrieval through embedding-space
operations alone. This method replaces the traditional
"Retrieve-Rewrite-Vectorize-Retrieve" cycle with a streamlined
"Retrieve-Embed-Retrieve" loop, significantly reducing computational overhead.
Moreover, a rule-based stop criterion is introduced to further prune redundant
retrievals, balancing efficiency and recall rate. Experimental results show
that TreeHop rivals advanced RAG methods across three open-domain MHQA
datasets, achieving comparable performance with only 5\%-0.4\% of the model
parameter size and reducing the query latency by approximately 99\% compared to
concurrent approaches. This makes TreeHop a faster and more cost-effective
solution for deployment in a range of knowledge-intensive applications. For
reproducibility purposes, codes and data are available here:
https://github.com/allen-li1231/TreeHop-RAG.

## Full Text


<!-- PDF content starts -->

Preprint. Under review.
TreeHop: Generate and Filter Next Query Embeddings Efficiently
for Multi-hop Question Answering
Zhonghao Li1, Kunpeng Zhang1, Jinghuai Ou, Shuliang Liu2,
Xuming Hu2,
1University of Maryland,
2Hong Kong University of Science and Technology.
al1231@terpmail.umd.edu, kpzhang@umd.edu
Abstract
Retrieval-augmented generation (RAG) systems face significant challenges in multi-
hop question answering (MHQA), where complex queries require synthesizing
information across multiple document chunks. Existing approaches typically rely
on iterative LLM-based query rewriting and routing, resulting in high computational
costs due to repeated LLM invocations and multi-stage processes. To address these
limitations, we propose TreeHop, an embedding-level framework without the need
for LLMs in query refinement. TreeHop dynamically updates query embeddings by
fusing semantic information from prior queries and retrieved documents, enabling
iterative retrieval through embedding-space operations alone. This method replaces
the traditional "Retrieve-Rewrite-Vectorize-Retrieve" cycle with a streamlined
"Retrieve-Embed-Retrieve" loop, significantly reducing computational overhead.
Moreover, a rule-based stop criterion is introduced to further prune redundant
retrievals, balancing efficiency and recall rate. Experimental results show that
TreeHop rivals advanced RAG methods across three open-domain MHQA datasets,
achieving comparable performance with only 5%-0.4% of the model parameter
size and reducing the query latency by approximately 99% compared to concurrent
approaches. This makes TreeHop a faster and more cost-effective solution for
deployment in a range of knowledge-intensive applications. For reproducibility
purposes, codes and data are available here1.
1 Introduction
Recent breakthroughs in Large Language Models (LLMs) (DeepSeek-AI et al., 2025; OpenAI
et al., 2024) have demonstrated their impressive capabilities in understanding queries (Brown et al.,
2020; Ouyang et al., 2022) and generating human-like language texts. Nonetheless, LLMs still
face significant limitations, particularly in domain-specific (Li et al., 2024; Zhang et al., 2024)
or knowledge-intensive (Kandpal et al., 2023) tasks, where they often hallucinate (Zhang et al.,
2023) when dealing with queries that exceed their parametric knowledge (Muhlgay et al., 2024). To
address this issue, Retrieval-augmented generation (RAG) (Lewis et al., 2021) has undergone rapid
development (Gao et al., 2024), leveraging external knowledge bases to retrieve relevant document
chunks and integrate them into LLMs, thereby producing more faithful (Khandelwal et al., 2020) and
generalizable (Kamalloo et al., 2023) answers.
However, the conventional single-retrieval paradigm of RAG falters in multi-hop question answering
(MHQA) scenarios (Yang et al., 2018; Ho et al., 2020; Tang & Yang, 2024; Trivedi et al., 2022), where
answers require synthesizing information from multiple document chunks. For instance, consider the
query " Who is the grandfather of Donald Trump? " A single retrieval might return a chunk stating
"Donald John Trump was born on June 14, 1946..., the fourth child of Fred Trump and Mary Ann
Macleod Trump. ", but resolving the grandfather requires a follow-up query like " Who is the father of
Fred Trump? ". This typical multi-hop scenario reveals the need to dynamically compose new query
based on information in relevant document chunk. Current methods like query-rewriters (Ma et al.,
2023), routers (Zhuang et al., 2024), and iterative loops (Shao et al., 2023) attempt to resolve this by
1https://github.com/allen-li1231/TreeHop-RAG
1arXiv:2504.20114v2  [cs.IR]  30 Apr 2025

Preprint. Under review.
iteratively refining queries with retrieved information, and drop chunks irrelevant to answering to
the query. While these approaches improve retrieval, they introduce computational overhead due to
repeated LLM invocations and multi-stage processes, leading to latency and complexity trade-offs.
To addresses these limitations, we propose TreeHop, a framework enabling iterative retrieval through
embedding-level updates without requiring LLM rewrites. Inspired by the semantic and structural
properties of sentence embeddings (Zhu et al., 2018), TreeHop dynamically generates next-step
query embeddings by fusing prior queries and retrieved content embeddings (Step 3, Figure 1). For
the aforementioned example, the initial information in query "grandfather of Donald Trump" was
substituted with "father of Fred Trump", now encoded directly at the embedding level. This approach
collapses the traditional "Retrieve-Rewrite-Vectorize-Retrieve" cycle into a streamlined "Retrieve-
Embed-Retrieve" loop, significantly reducing computational costs. TreeHop further introduces two
pruning strategies to ensure computational efficiency: redundancy pruning terminates paths where
the retrieved chunks have been seen in previous iterations, while layer-wise top-K pruning retains
only the top-ranked retrieval candidates at each step, curbing exponential branch growth (Step 4,
Figure 1).
TreeHop employs a gated cross-attention mechanism (Vaswani et al., 2023) to effectively focus
on extracting salient information from retrieved chunks, making the model effective while param-
eterizing with only 25 million parameters. Trained with contrastive learning (Chen et al., 2020;
Wu et al., 2022b), TreeHop is capable of achieving a performance comparable to computationally
intensive multi-hop methods across three benchmarks while maintaining significantly lower latency.
Remarkably, TreeHop reduces retrieval latency by 99% compared to LLM-based methods while
sacrificing only 4.8% of the recall rate at maximum, even outperforming some advanced systems by
4.1% in deeper retrieval iterations.
In summary, this work makes three key contributions:
•A novel embedding-updating mechanism that replaces LLM-driven iterative query rewrites
with lightweight neural operations, enabling linear computational complexity for MHQA tasks.
•An efficient rule-based stopping criterion that controls branching factor growth while maintain-
ing performance in retrieval iteration.
•Empirical validation demonstrating superior efficiency-accuracy trade-offs in three MHQA
tasks. Our approach bridges the gap between computational efficiency and retrieval effective-
ness, offering a scalable solution for diverse knowledge-intensive applications.
2 Preliminaries
2.1 Multi-hop Retrieval-Augmented Generation
The Retrieval Augmented Generation (RAG) (Lewis et al., 2021; Gao et al., 2024) fundamentally
enhances the capabilities of LLMs by retrieving pertinent documents from an external knowledge
base, which is made possible through the calculation of semantic similarity between user’s query
and document chunks. Building upon RAG, multi-hop variants have been proposed to tackle more
complex tasks, such as multi-hop question answering (MHQA). Notable approaches include iterative
retrieval methods (Shao et al., 2023), where the knowledge base is repeatedly searched based on
the initial query and generated text, providing a more comprehensive information retrieval. Other
approaches revolve around employing cooperative language models as query-rewriters (Ma et al.,
2023), routers (Manias et al., 2024) or both (Zhuang et al., 2024). These models generate new queries
for document chunk retrieval and filter out irrelevant chunks, ensuring the most relevant information
is retained. It is worth noting that they all mentioned solutions utilize one or multiple transformer
model variants (Vaswani et al., 2023; Reimers & Gurevych, 2019) for enhanced retrieval, which
induces additional computational cost and significantly increases system latency.
2.2 Sentence Representation Learning and Contrastive Learning
Sentence representation learning, a technique for training retrieval model in the realm of RAG,
refers to the task of encoding sentences into fixed-dimensional embeddings. Early approaches
2

Preprint. Under review.
Figure 1: The TreeHop model utilizes query and content chunk embeddings to generate new query
embeddings, which are subsequently filtered with similarity and ranking thresholds, thereby stream-
lines the conventional "Retrieve-Rewrite-Vectorize-Retrieve" into a "Retrieve-Embed-Retrieve" loop.
extended word-level techniques like word2vec (Mikolov et al., 2013) to sentences, such as Skip-
Though (Kiros et al., 2015) and FastSent citephill-etal-2016-learning, which learned unsupervised
sentence embeddings by optimizing sequential or semantic coherence objectives. Subsequent work
leveraged pre-trained language models like BERT (Reimers & Gurevych, 2019), extracting sentence
embeddings via the [CLS] token or mean pooling of contextualized token representations (Reimers
& Gurevych, 2019; Li et al., 2020; Su et al., 2021).
To further improve the performance, contrastive learning emerged as a powerful paradigm for learning
discriminative sentence representations (Zhang et al., 2020; Carlsson et al., 2021; Giorgi et al., 2021;
Yan et al., 2021; Kim et al., 2021). A cornerstone in this space is SimCSE (Gao et al., 2021), which
employs InfoNCE (van den Oord et al., 2019) to maximize agreement between augmented views of
the same sentence. The loss function is defined as:
LinfoNCE =−N
∑
i=1logesim(f(xi),f(xi)′)/τ
∑N
j=1esim(f(xi),f(xj)′)/τ, (1)
where Nis the batch size, τis a temperature hyperparameter and sim(f(xi),f(xj)′) =
f(xi)⊤f(xj)′
∥f(xi)∥·∥f(xj)′∥is the cosine similarity used in this work. f(·)is the sentence representation encoder,
xiandx′
iare a paired semantically related sentences derived from positive set D={(xi,x′
i)}m
i=1.
Additionally, SimCSE applied a dropout as a data augmentation strategy, inspired many following
works. Meanwhile, DiffCSE (Chuang et al., 2022) introduces equivariant transformations to ensure
invariance to input perturbations, while PCL (Wu et al., 2022a) leverages diverse augmentation strate-
gies to reduce bias in negative sampling. InfoCSE (Wu et al., 2022c) learns sentence representations
with the ability to reconstruct the original sentence fragments, RankCSE (Liu et al., 2023) further
introduce a listwise ranking objectives for learning effective sentence representations.
3 The Proposed Method: TreeHop
Our proposed model, TreeHop, is designed to generate the next query embedding by integrating
previous query embeddings and retrieved content embeddings. This approach streamlines the
3

Preprint. Under review.
Figure 2: The model architecture of TreeHop. The UpdateGate , using cross-attention, updates
embeddings via selectively incorporating new information from chunk embeddings. The output is
combined with the difference between the previous query and chunk embeddings to form the next
query embedding.
conventional iterative "Retrieve-Rewrite-Vectorize-Retrieve" process inherent in RAG systems into a
more efficient "Retrieve-Embed-Retrieve" workflow, reducing both system latency and computational
overhead. Furthermore, we have optimized the architecture to achieve high retrieval performance
while ensuring a compact parameter size. In the following sections, we first formally define the
problem, then detail the model architecture and stopping criterion that contribute to its computational
efficiency and effectiveness. Finally, we explain the construction of the training data.
3.1 Problem Formulation
At retrieval iteration r, given query embedding q, a set of the top Kdocument chunk embeddings,
T={ci}K
i=1, is retrieved using the retriever g(q,K). The TreeHop model then generates the
corresponding next query embedding set Q1={qi
r+1}K
i=1for the subsequent hop retrieval.
qi
r+1=TreeHop (qr,ci
r),ci
r∈ Tr (2)
Please refer to Figure 1 for detailed TreeHop inference steps with the stop criterion included in
subsection 3.3.
3.2 Model Architecture
TreeHop’s architecture is tailored to be effective in performance while maintain a small parameter
size. The core of TreeHop’s query update is the UpdateGate (see Figure 2), which modulates how
information from prior queries qand retrieved chunks cis retained or discarded. The intuition behind
is that we only need to remove information presents in both embeddings, and update information yet
to be further retrieved from the retrieved chunks to form a new query embedding.
TreeHop (qr,ci
r) =qr−ci
r+UpdateGate (qr,ci
r) (3)
The term qr−ci
rsuppresses semantic overlap between the current query and chunk embeddings.
This prevents redundant retrieval of information already captured. (e.g., if the chunk confirms "Fred
Trump is Donald’s father," the model avoids re-searching for Fred Trump in subsequent hops). The
UpdateGate (qr,ci
r)selectively incorporates new information from ci
rto form the next query. We
implement cross-attention mechanism (Vaswani et al., 2023) for UpdateGate .
4

Preprint. Under review.
Input: Initial user query text x, embedding model f(·)that generates embeddings for input text, total
number of hops N≥1, retriever g(·)that takes one query embedding and outputs top Kdocument chunks
{ci}K
i=1and respective embeddings {υi}K
i=1, cosine similarity scores {si}K
i=1.
Output: Retrieved document chunk set C.
1:C=∅
Q={f(x)}// Query embedding set
2:forr∈ {1, . . . , N}do
3:S=∅
4: forqinQdo
5: // Retrieve and iterate over top Kdocument chunks, embeddings and similarity scores
6: forc,υ,sing(qi,K)do
7: ifc/∈ C then
8: // Include only distinct chunks.
S ← [q,c,v,s]
9:Q=∅
t=TopSimilarityScore (S,K)// Get layer-wise K-th similarity score tfrom set S.
10: forq,c,υ,sinSdo
11: ifs≥tthen
12: C ← c
Q ← TreeHop (q,v)
Figure 3: Multihop inference steps and rule-based stop criterion for TreeHop.
UpdateGate (q,c) =CrossAttn u(q,c)
=softmax (Qu(q)⊙Ku(c)√
d)⊙Vu(c)(4)
Where dis the number of embedding dimension, Qu,KuandVuare three weight and bias matrices
forUpdateGate . Information to be maintained in the chunk embeddings is selectively extracted
through comparing the information in qrandci
r. This architectural design is based on empirical
experiments for improving the model performance (see subsection 5.1 for ablation details).
3.3 Stopping Criterion
The TreeHop iteratively generates query embedding for every retrieved document chunk, this risks
excessive computational costs if every query proceeds to subsequent hops. Unchecked, this approach
could lead to an exponential increase in retrieved chunks ( O(nr)), degrading efficiency without
proportional gains in accuracy. To address this, we introduce a set of rule-based stop criterion that
dynamically prunes irrelevant or redundant retrieval branches to ensure only promising paths advance.
Redundancy Pruning We terminate branches where the document chunk has been retrieved in the
previous iterations, as depicted in line 8, Figure 3.
Layer-wise Top- KPruning At each retrieval layer, we retain only the top- Kchunks with highest
similarity scores across all generated query embeddings. This reduces the branching factor from
O(nr)toO(nr)by focusing computation on the most promising paths, as shown in line 12, Figure 3.
3.4 Train Data Construction
To train the TreeHop model, we require a dataset that explicitly captures the multi-step knowledge re-
trieval process for MHQA. The 2WikiMultiHop dataset (Ho et al., 2020) provides an ideal foundation
due to its explicit decomposition of complex questions into intermediate steps, with each step linked
to corresponding evidence chunks from Wikipedia. However, not all question types in this dataset are
equally suitable for our purposes, the following processes are implemented to clean the dataset:
Question Type Selection We focus on inference, compositional and bridge comparison questions,
as they strictly require the model to synthesize information across multiple hops (e.g., deriving a
5

Preprint. Under review.
grandfather’s identity by first retrieving a father’s name). In contrast, comparison questions rely more
on direct factual retrieval without requiring iterative information interaction. See Appendix A for
detailed information about question types.
Query Type Integrity Check We filter the dataset to retain only instances where the provided
query decompositions align precisely with the multi-step reasoning required by the question type.
Through this curation process, we obtained 111,239 high-quality training samples.
3.5 Model Training
We utilize BGE-m3 (Chen et al., 2024), a multilingual embedding model that supports more than
100 languages, to generate dense embeddings for the initial query and construct a document chunk
embedding database. This gives our trained TreeHop model the potential to be versatile and applicable
to a wide range of languages and use cases. Note that BGE-m3 remains frozen during training to
ensure the training process focus on the TreeHop model. For detailed prompt templates for generating
embeddings on three datasets, please refer to Table 7 and Table 8 in Appendix C.
Following previous work (van den Oord et al., 2019), we adopt contrastive learning framework
to train TreeHop to generate embeddings that maximizes the similarities with their corresponding
positive chunk embeddings while minimizing similarity with negative ones. Specifically, we employ
theLinfoNCE objective in Equation 1 with temperature τof 0.15 and five negatives sampled from
embedding database. The model is compact enough to be trained on a single Nvidia V100 GPU, with
batch size of 64, AdamW optimizer and learning rate of 6e-5. Inspired by SimCSE (Gao et al., 2021),
we add a dropout layer after the hidden representations for data augmentation.
4 Experiments and Results
To examine TreeHop, experiments are conducted regarding its retrieval performance and efficiency.
Below, we introduce the selective evaluation datasets, evaluate metrics, baselines and concurrent
advanced RAG solutions that involve in the experiments for comparison.
4.1 Datasets
We benchmark TreeHop on three widely used MHQA datasets in the literature: 2WikiMultiHop (Ho
et al., 2020), MuSiQue answerable (Trivedi et al., 2022), and MultiHop RAG (Tang & Yang, 2024).
Some of their questions do not challenge multihop retrieval performance but require LLMs to deduce
from multiple documents. Therefore, to repel extraneous noises, we focus on question types requiring
multi-step retrieval. To be more specific, in 2WikiMultiHop, we filter to inference, compositional and
bridge comparison questions (9,536 records), while for MuSiQue, all 2,417 answerable questions
are used. MultiHop RAG’s inference questions (816 records) are included. This ensures that the
evaluation targets scenarios where multi-hop retrieval is essential. See Appendix A for detailed
introduction to the types of question among the evaluate datasets, and Table 6 for detailed number of
queries for each selective types of question.
Dataset Query Embedding Database
2WikiMultihop 9,536 56,709
MuSiQue 2,417 21,100
Multihop RAG 816 609
Table 1: Descriptive statistics of datasets in terms of the number of queries and sizes of the corre-
sponding embedding database.
4.2 Evaluation Metrics & Benchmarks
We use the standard evaluation metric, the recall rate, to test the retrieval performance, specifically in
the top Kretrieval setting, denoted Recall@K. It measures whether the relevant documents are present
6

Preprint. Under review.
among the top Kretrieved documents. Higher Recall@K values indicate better retrieval performance.
To compare the efficiency among selected RAG solutions, we record the average durations for each
query in seconds on each dataset, denoted latency. The whole evaluation process is conducted on a
single Nvidia A100 GPU and 64 GB of RAM.
Baselines and Advanced RAG We evaluate the performance of TreeHop by comparing it to a
native top retrieval method using the BGE-m3 embedding model as the baseline. In addition, we
include two advanced iterative retrieval-augmented generation (RAG) methods: Iter-RetGen (Shao
et al., 2023) and EfficientRAG (Zhuang et al., 2024) to assess both performance and latency. For
Iter-RetGen, we use the vanilla Meta Llama3 8B Instruct model (AI@Meta, 2024) as the inference
model. Additionally, we test Iter-RetGen and TreeHop under the second and third retrieval iterations,
respectively, to evaluate their performance across different stages of the retrieval process. For
more detailed information on the prompt templates used in Iter-RetGen, please refer to Table 9 in
Appendix C.
4.3 Results
In this section, we present the experimental results of TreeHop and benchmarks on three datasets,
including retrieval efficiency and recall.
Retrieval Efficiency As shown in Table 2, TreeHop significantly reduces computational overhead
while maintaining competitive retrieval performance. It achieves latencies of 0.02 seconds per
query in the second iteration and 0.06 seconds in the third, outperforming the next best solution,
EfficientRAG, by over 2.9 seconds. This significant reduction of 99.2%–99.6% in latency is attributed
to TreeHop’s mechanism, stems from its embedding-level computation, which avoids the recursive
inference loops required by LLM-based methods. This is confirmed by examining the latency, which
is proportional to the number of retrieved document chunks.
2WIKI MUSIQUE MULTIHOP-RAG
Retriever Recall@K K Latency Recall@K K Latency Recall@K K Latency
Baselines
Direct-R@5 49.3 5 0.002 45.4 5 0.002 48.6 5 0.019
Direct-R@10 53.2 10 0.003 53.8 10 0.002 67.8 10 0.019
Advanced RAG
Iter-RetGen @5 iter2 59.2 9.9 4.690 52.8 9.9 4.949 55.0 9.9 4.876
Iter-RetGen @5 iter3 61.9 14.7 7.278 54.1 14.8 7.274 57.0 14.5 7.322
EfficientRAG @5 60.5 3.8 2.846 46.9 6.1 2.907 51.8 4.1 2.855
Ours
TreeHop @5 iter2 61.6 8.6 0.022 48.0 8.1 0.023 57.9 7.0 0.023
TreeHop @5 iter3 65.4 11.8 0.067 48.1 11.0 0.064 61.1 8.4 0.062
TreeHop @10 iter2 57.9 17.2 0.062 55.7 15.3 0.056 72.8 13.1 0.049
Table 2: We report results of baselines, concurrent advanced RAG solutions and TreeHop on three
MHQA datasets. Bold numbers indicate the best performance in the same iteration among retrievers.
Retrieval Performance TreeHop achieves strong performance across datasets while balancing
efficiency and effectiveness. On the 2WikiMultiHop and Multihop dataset, TreeHop surpasses the
second best solution, Iter-RetGen, by 2.4%-2.9% recall in the second iteration and 3.5%-4.1% recall
in the third iteration, with 3.1 less chunks retrieved on average. This demonstrates the effectiveness
of the embedding mechanism in Equation 4. For the MuSiQue dataset, recall is 4.8% lower than
Iter-RetGen, likely due to the dataset’s requirement for synthesizing information from multiple
chunks (e.g., branching and converging paths in Appendix A), which TreeHop’s current architecture
addresses less effectively than iterative LLM-based approaches. The performance degradation aligns
with EfficientRAG solution, which also struggles with this dataset, suggesting a limitation common
to query-chunk-pair strategies.
Average Number of KOverall, our TreeHop’s average number of retrieved document chunks falls
in the middle of the advanced RAG solutions. This is contributed by stop criteria, which drastically
curtails computational overhead. For top-5 retrieval, it reduces the theoretical exponential growth of
7

Preprint. Under review.
chunks, 52=25chunks in second iteration, to 7.1–8.8 chunks, and 53=125chunks to 8.3–12.1
chunks in the third iteration. For top-10 retrieval, this scales linearly to 13.8–17.9 chunks, versus 102
chunks without pruning.
5 Ablation Study
5.1 Effectiveness of Architecture
To evaluate the necessity of each component in Equation (3), we ablated the query term q, chunk term
c, and UpdateGate in isolation. Each variant was trained 10 times with identical hyperparameters
(as in subsection 3.5), and performance was evaluated on the second hop’s recall rate. Results are
summarized in Table 3 and analyzed below.
Architecture 2WIKI (avg.) MUSIQUE (avg.) MULTIHOP RAG (avg.)
Direct-R@5 49.3 45.4 48.6
TreeHop @5 iter2 61.6 48.0 57.9
w/o.c 57.5 (4.1 ↓) 47.1 (0.9 ↓) 51.9 (6.0 ↓)
w/o.q 54.6 ( 6.0↓) 46.3 ( 2.5↓) 50.8 ( 7.1↓)
w/o.UpdateGate 49.3 ( 12.3↓) 45.4 ( 3.4↓) 48.6 ( 9.3↓)
Table 3: Ablation study on TreeHop model architecture. The TreeHop experiences degraded perfor-
mances when core components q,candUpdateGate are removed from its architecture, demonstrating
their functionalities to the model performance.
Impact of Component Removal The impact of structure without cis the minimal, with a decrease
of 0.9%-6.0% of average recall rates across datasets. The UpdateGate mitigated information loss
by selectively retaining critical chunk information, without c, it keeps the information to the extent
that still makes the model effective. However, average training convergence time increased by
approximately 15% on average, as the model struggled to suppress redundant information without
theq−cstructure.
Without q, the model loses critical information from query, thereby experiences significant perfor-
mance degrade on the three datasets, achieves only 0.09%-5.3% improvement of average recall rates
comparing to vanilla top 5 retrieval. It is observable that the model exhibits a lower degrades on
2WikiMultihop dataset, we conclude from the result that this is due to our usage of 2WikiMultihop
training data that make the model overfit to similar questions in evaluate dataset, ultimately leaving
no generalization ability to the other two datasets.
Without UpdateGate , recall dropped to near baseline retrieval performance (within 0.1% of random
retrieval), confirming the gate’s critical role in integrating new information. Without it, the model
degenerated to a simple vector difference, failing to refine queries iteratively.
Dataset-Specific Insights The three datasets exhibit different extents of performance decay when
the same components are removed. The MuSiQue dataset decays the least without c,q, and
UpdateGate , this is due to the inherent deficiency of TreeHop on multihop queries with converging
paths, making it perform inferior to the other datasets. The Multihop RAG dataset experiences the
greatest negative impact without q, possibly due to its complex queries that mention more than three
entities for the retrieval model to gather each piece of information. Without q, TreeHop cannot
navigate the missing information. The 2WikiMultihop dataset influences less than Multihop RAG
without candq, possibly because of its less challenging queries and query decomposition paths.
5.2 Effectiveness of Stop Criterion
The stop criterion serves for filtering query paths to reduce computational cost without sacrificing
too much performance. Below we examine the performance without the presence of Redundancy
Pruning and Layer-wise Top Pruning, illustrated in Table 4.
8

Preprint. Under review.
2WIKI MUSIQUE MULTIHOP RAG
Stop Criterion Recall@K K Recall@K K Recall@K K
TreeHop @5 iter2 61.6 8.6 48.0 8.1 57.9 7.0
w/o. Redundancy Pruning 57.4 10 46.4 10 52.8 10
w/o. Layer-wise Top Pruning 80.2 18.4 53.6 14.5 61.3 9.7
w/o. Both 80.2 30 53.6 30 61.3 30
Table 4: Ablation study on stop criterion, where redundancy pruning and layer-wise top 5 pruning are
removed from post retrieval process, respectively. The results indicate that the recall rate does not
exhibit a considerable enhancement, despite a substantial grow in the number of average K.
Redundancy Pruning Redundancy pruning prevents revisiting previously retrieved chunks. Table 4
shows that removing this pruning increases the average number of retrieved chunks Kto 10, but
reduces recall by 4.2 points (e.g., 61.6 →57.4 on 2WikiMultihop). This occurs because when
cooperate with layer-wise top pruning, redundancy pruning further ensures only unique, informative
paths are pursued, thereby maintaining recall performance. Without redundancy pruning, more
duplicated information take the place, resulting in degraded performance.
Layer-wise Top Pruning This pruning strategy selects top-5 chunks at each layer to control
branching. Removing this strategy results in a maximum increase of 113% in the average number of
retrieved chunks (8.6 →18.4) on the 2WikiMultihop dataset, but yields a 18.6% improvement in
recall (61.6% →80.2%). This suggests that the pruning introduces a trade-off between computational
efficiency and recall performance. In contrast, for the Multihop RAG dataset, the recall improves by
5.6% with a 79% increase in average retrieved chunks, a significantly lower increase compared to
2WikiMultihop. This disparity among datasets is strongly correlated with the size of the corresponding
embedding database. Specifically, in large databases such as 2WikiMultihop, which features 56,709
chunk embeddings, the model benefits from exploring more paths (higher K) due to the large
information pool. Conversely, in smaller databases, e.g., Multihop RAG with 609 chunk embeddings,
iterative retrieval tends to introduce more redundant chunks, making layer-wise top pruning filter out
more useful chunks.
Combined Effects The synergistic effect of combining redundancy pruning and layer-wise top
pruning is critical to achieving TreeHop’s efficiency gains without excessive recall loss. Take
Multihop RAG for example, When both criteria are applied, they achieve a recall of 57.9% with an
average number of retrieved chunks 7.0. When both criteria are disabled, the system’s computational
complexity balloons to 30 chunks in the second iteration, a 329% increase, while yielding only a
marginal 3.4% recall improvement. This demonstrates that layer-wise top pruning is essential for
limiting branching factor, while redundancy pruning prevents recall degradation from redundant
paths. Their combined use ensures that TreeHop avoids the exponential retrieval path explosion
inherent to iterative systems while maintaining competitive performance. This synergy underscores
their necessity for achieving the model’s design goal of efficient multi-hop retrieval.
6 Conclusion
This work presents a novel paradigm for Retrieval-Augmented Generation (RAG), introducing
TreeHop, a novel lightweight query embedding generator that dynamically refines query embeddings
through iterative retrieval without relying on additional LLMs or complex rewrite components,
thereby enhancing the efficiency of RAG system. Its core mechanism, the UpdateGate , employs
cross-attention to selectively integrate information from retrieved chunks while discarding redundant
information, enables a compact model size of 25 million parameters while maintaining competitive
performance on three MHQA benchmarks when integrated with simple downstream rule-based
stop criterion. Future work could explore more effective model architecture, adaptive stop criteria
or extensions to handle lengthy, structural, or multi-modal inputs. Our approach underscores the
potential of embedding-centric strategies to enhance retrieval process for RAG systems, offering a
practical balance between performance and computational efficiency, paving the way for solutions to
real-world multi-hop reasoning challenges in industrial applications.
9

Preprint. Under review.
References
AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/
blob/main/MODEL_CARD.md .
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-V oss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler,
Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott
Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.
Fredrik Carlsson, Amaru Cuba Gyllensten, Evangelia Gogoulou, Erik Ylipää Hellqvist, and Magnus
Sahlgren. Semantic re-tuning with contrastive tension. In International Conference on Learning
Representations , 2021. URL https://openreview.net/forum?id=Ov_sMNau-PF .
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge
distillation, 2024.
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
contrastive learning of visual representations, 2020. URL https://arxiv.org/abs/2002.
05709 .
Yung-Sung Chuang, Rumen Dangovski, Hongyin Luo, Yang Zhang, Shiyu Chang, Marin Sol-
jacic, Shang-Wen Li, Scott Yih, Yoon Kim, and James Glass. DiffCSE: Difference-based con-
trastive learning for sentence embeddings. In Marine Carpuat, Marie-Catherine de Marneffe,
and Ivan Vladimir Meza Ruiz (eds.), Proceedings of the 2022 Conference of the North Amer-
ican Chapter of the Association for Computational Linguistics: Human Language Technolo-
gies, pp. 4207–4218, Seattle, United States, July 2022. Association for Computational Linguis-
tics. doi: 10.18653/v1/2022.naacl-main.311. URL https://aclanthology.org/2022.
naacl-main.311/ .
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu,
Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu,
Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao
Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao,
Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding,
Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang
Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong,
Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao,
Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang,
Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang,
Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L.
Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang,
Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng
Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng
Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan
Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang,
Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen,
Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y . K. Li,
Y . Q. Wang, Y . X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang,
Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan,
Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia
He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y . X. Zhu, Yanhong
Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha,
Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang,
Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li,
Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen
Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
URLhttps://arxiv.org/abs/2501.12948 .
10

Preprint. Under review.
Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple contrastive learning of sentence
embeddings. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau
Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language
Processing , pp. 6894–6910, Online and Punta Cana, Dominican Republic, November 2021.
Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.552. URL https:
//aclanthology.org/2021.emnlp-main.552/ .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey,
2024.
John Giorgi, Osvald Nitski, Bo Wang, and Gary Bader. DeCLUTR: Deep contrastive learning for
unsupervised textual representations. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli
(eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics
and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long
Papers) , pp. 879–895, Online, August 2021. Association for Computational Linguistics. doi: 10.
18653/v1/2021.acl-long.72. URL https://aclanthology.org/2021.acl-long.72/ .
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-
hop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th
International Conference on Computational Linguistics , pp. 6609–6625, Barcelona, Spain (Online),
December 2020. International Committee on Computational Linguistics. URL https://www.
aclweb.org/anthology/2020.coling-main.580 .
Ehsan Kamalloo, Nouha Dziri, Charles L. A. Clarke, and Davood Rafiei. Evaluating open-domain
question answering in the era of large language models, 2023.
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language
models struggle to learn long-tail knowledge, 2023.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization
through memorization: Nearest neighbor language models, 2020.
Taeuk Kim, Kang Min Yoo, and Sang-goo Lee. Self-guided contrastive learning for BERT sentence
representations. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceed-
ings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th
International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , pp.
2528–2540, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/
2021.acl-long.197. URL https://aclanthology.org/2021.acl-long.197/ .
Ryan Kiros, Yukun Zhu, Russ R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba,
and Sanja Fidler. Skip-thought vectors. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and
R. Garnett (eds.), Advances in Neural Information Processing Systems , volume 28. Curran Asso-
ciates, Inc., 2015. URL https://proceedings.neurips.cc/paper_files/paper/
2015/file/f442d33fa06832082290ad8544a8da27-Paper.pdf .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021.
Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. On the sentence
embeddings from pre-trained language models. In Bonnie Webber, Trevor Cohn, Yulan He, and
Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP) , pp. 9119–9130, Online, November 2020. Association for Computational
Linguistics. doi: 10.18653/v1/2020.emnlp-main.733. URL https://aclanthology.org/
2020.emnlp-main.733/ .
Zhonghao Li, Xuming Hu, Aiwei Liu, Kening Zheng, Sirui Huang, and Hui Xiong. Refiner:
Restructure retrieval content efficiently to advance question-answering capabilities, 2024. URL
https://arxiv.org/abs/2406.11357 .
11

Preprint. Under review.
Jiduan Liu, Jiahao Liu, Qifan Wang, Jingang Wang, Wei Wu, Yunsen Xian, Dongyan Zhao, Kai Chen,
and Rui Yan. Rankcse: Unsupervised sentence representations learning via learning to rank, 2023.
URLhttps://arxiv.org/abs/2305.16726 .
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting for retrieval-
augmented large language models, 2023. URL https://arxiv.org/abs/2305.14283 .
Dimitrios Michael Manias, Ali Chouman, and Abdallah Shami. Semantic routing for enhanced
performance of llm-assisted intent-based 5g core network management and orchestration, 2024.
URLhttps://arxiv.org/abs/2404.15869 .
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations
of words and phrases and their compositionality. In C.J. Burges, L. Bottou, M. Welling, Z. Ghahra-
mani, and K.Q. Weinberger (eds.), Advances in Neural Information Processing Systems , volume 26.
Curran Associates, Inc., 2013. URL https://proceedings.neurips.cc/paper_
files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf .
Dor Muhlgay, Ori Ram, Inbal Magar, Yoav Levine, Nir Ratner, Yonatan Belinkov, Omri Abend,
Kevin Leyton-Brown, Amnon Shashua, and Yoav Shoham. Generating benchmarks for factuality
evaluation of language models, 2024.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor
Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian,
Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny
Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks,
Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea
Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen,
Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung,
Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch,
Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty
Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte,
Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel
Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua
Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike
Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon
Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne
Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo
Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar,
Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik
Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich,
Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy
Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie
Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini,
Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne,
Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David
Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie
Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély,
Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo
Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano,
Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng,
Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto,
Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power,
Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis
Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted
Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel
Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon
Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky,
Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang,
Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston
12

Preprint. Under review.
Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya,
Chelsea V oss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason
Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff,
Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu,
Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba,
Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang,
William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024.
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton,
Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and
Ryan Lowe. Training language models to follow instructions with human feedback, 2022.
Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-
networks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the
2019 Conference on Empirical Methods in Natural Language Processing and the 9th International
Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pp. 3982–3992, Hong Kong,
China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1410.
URLhttps://aclanthology.org/D19-1410/ .
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Enhancing
retrieval-augmented large language models with iterative retrieval-generation synergy, 2023. URL
https://arxiv.org/abs/2305.15294 .
Jianlin Su, Jiarun Cao, Weijie Liu, and Yangyiwen Ou. Whitening sentence representations for better
semantics and faster retrieval, 2021. URL https://arxiv.org/abs/2103.15316 .
Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop
queries, 2024. URL https://arxiv.org/abs/2401.15391 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop
questions via single-hop question composition. Transactions of the Association for Computational
Linguistics , 10:539–554, 2022. doi: 10.1162/tacl_a_00475. URL https://aclanthology.
org/2022.tacl-1.31/ .
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive
coding, 2019. URL https://arxiv.org/abs/1807.03748 .
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. Attention is all you need, 2023. URL https://arxiv.org/
abs/1706.03762 .
Qiyu Wu, Chongyang Tao, Tao Shen, Can Xu, Xiubo Geng, and Daxin Jiang. Pcl: Peer-contrastive
learning with diverse augmentations for unsupervised sentence embeddings, 2022a. URL https:
//arxiv.org/abs/2201.12093 .
Xing Wu, Chaochen Gao, Zijia Lin, Jizhong Han, Zhongyuan Wang, and Songlin Hu. InfoCSE:
Information-aggregated contrastive learning of sentence embeddings. In Yoav Goldberg, Zornitsa
Kozareva, and Yue Zhang (eds.), Findings of the Association for Computational Linguistics:
EMNLP 2022 , pp. 3060–3070, Abu Dhabi, United Arab Emirates, December 2022b. Association
for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp.223. URL https://
aclanthology.org/2022.findings-emnlp.223/ .
Xing Wu, Chaochen Gao, Zijia Lin, Jizhong Han, Zhongyuan Wang, and Songlin Hu. Infocse:
Information-aggregated contrastive learning of sentence embeddings, 2022c. URL https:
//arxiv.org/abs/2210.06432 .
Yuanmeng Yan, Rumei Li, Sirui Wang, Fuzheng Zhang, Wei Wu, and Weiran Xu. ConSERT:
A contrastive framework for self-supervised sentence representation transfer. In Chengqing
Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceedings of the 59th Annual Meeting
of the Association for Computational Linguistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long Papers) , pp. 5065–5075, Online, August
2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.393. URL
https://aclanthology.org/2021.acl-long.393/ .
13

Preprint. Under review.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP) , 2018.
Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E.
Gonzalez. Raft: Adapting language model to domain specific rag, 2024.
Yan Zhang, Ruidan He, Zuozhu Liu, Kwan Hui Lim, and Lidong Bing. An unsupervised sen-
tence embedding method by mutual information maximization. In Bonnie Webber, Trevor
Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) , pp. 1601–1610, Online, November 2020.
Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.124. URL
https://aclanthology.org/2020.emnlp-main.124/ .
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao,
Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi.
Siren’s song in the ai ocean: A survey on hallucination in large language models, 2023.
Xunjie Zhu, Tingfeng Li, and Gerard de Melo. Exploring semantic properties of sentence embeddings.
In Iryna Gurevych and Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the
Association for Computational Linguistics (Volume 2: Short Papers) , pp. 632–637, Melbourne,
Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-2100.
URLhttps://aclanthology.org/P18-2100/ .
Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai Yang, Jia Liu, Shujian Huang, Qingwei Lin,
Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. Efficientrag: Efficient retriever for multi-hop
question answering, 2024. URL https://arxiv.org/abs/2408.04259 .
14

Preprint. Under review.
Part I
Appendix
Table of Contents
A Dataset Cards 16
B Details on Evaluate Dataset Question Types 19
C Iter-RetGen Prompt Templates 20
15

Preprint. Under review.
A Dataset Cards
Below illustrates datasets inclusive in our work, the question types for evaluation are selected to ensure
synthesizing information from query and retrieved document chunks are mandatory for multihop
retriever.
Dataset Question Type Require Synthesize
2WikiMultiHopComparison question : Questions requiring direct comparison of at-
tributes between entities within the same category.
Example Question : Who was born first, Albert Einstein or Abraham
Lincoln?
Inference question : Questions requiring derivation of implicit relation-
ships by combining triples from a knowledge graph.
Example Question : Who is the maternal grandfather of Abraham Lin-
coln?✓
Triples : (Abraham Lincoln, mother, Nancy Hanks Lincoln); (Nancy
Hanks Lincoln, father, James Hanks).
Compositional question : Questions requiring multi-step relational rea-
soning across non-explicitly linked triples.
Example Question : Who founded the distributor of La La Land ?✓
Triples : (La La Land, distributor, Summit Entertainment); (Summit
Entertainment, founded by, Bernd Eichinger).
Bridge-comparison question : Questions requiring both bridging to
intermediate entities and comparative reasoning.
Example Question : Which movie has the director born first, La La Land
orTenet ?✓
Steps : 1. Find directors: La La Land →Damien Chazelle; Tenet→
Christopher Nolan.
2. Compare birth years: Damien Chazelle (1985) vs. Christopher Nolan
(1970).
MuSiQueUnanswerable : Questions with potential support paragraphs are partially
removed, making the reasoning infeasible or unable to arrive at the
correct answer.
2-Hop Reasoning (Linear Path) : A single, straightforward logical path
connecting two facts.
Example Question : Who succeeded the first President of Namibia? ✓
steps : 1. Identify the first President of Namibia.
2. Determine who succeeded them.
3-Hop Reasoning (Linear Path) A sequential, three-step logical con-
nection.
Example Question : What currency is used where Billy Giles died? ✓
steps : 1. Find the location of Billy Giles’ death.
2. Locate the region this place belongs to.
3. Identify the currency used in that region.
3-Hop Reasoning (Branching Path) Begins with a single inquiry but
diverges into different, branching sub-questions.
Example Question : When was the first establishment that McDonaldiza-
tion is named after, opened in the country Horndean is located?✓
steps : 1. Determine what McDonaldization refers to.
2. Identify the country where Horndean is located.
3. Find the date the first establishment opened in that country.
16

Preprint. Under review.
Dataset Question Type Require Synthesize
MuSiQue4-Hop Reasoning (Linear Path) A continuous, four-step logical pro-
gression.
Example Question : When did Napoleon occupy the city where the
mother of the woman who brought Louis XVI style to the court died?✓
steps : 1. Identify who introduced Louis XVI style.
2. Find their mother.
3. Determine the city of the mother’s death.
4. Discover when Napoleon occupied that city.
3-Hop Reasoning (Branching Path) Begins with a single inquiry but
diverges into different, branching sub-questions.
Example Question : When was the first establishment that McDonaldiza-
tion is named after, opened in the country Horndean is located?✓
steps : 1. Determine what McDonaldization refers to.
2. Identify the country where Horndean is located.
3. Find the date the first establishment opened in that country.
4-Hop Reasoning (Linear Path) A continuous, four-step logical pro-
gression.
Example Question : When did Napoleon occupy the city where the
mother of the woman who brought Louis XVI style to the court died?✓
steps : 1. Identify who introduced Louis XVI style.
2. Find their mother.
3. Determine the city of the mother’s death.
4. Discover when Napoleon occupied that city.
4-Hop Reasoning (Branching Path) Starts with a single query, splits
into multiple paths, and then converges.
Example Question : How many Germans live in the colonial holding in
Aruba’s continent that was governed by Prazeres’s country?✓
steps : 1. Locate Aruba’s continent.
2. Identify Prazeres’ country.
3. Determine the colonial holding governed by that country in Aruba’s
continent.
4. Find the number of Germans there.
4-Hop Reasoning (Converging Path) : Multiple distinct lines of reason-
ing that eventually converge on the answer.
Example Question : When did the people who captured Malakoff come
to the region where Philipsburg is located?✓
steps : 1. Determine Philipsburg’s location.
2. Identify the terrain feature it belongs to.
3. Find who captured Malakoff.
4. Determine when those people came to that terrain.
MultiHop RAGInference Query : Questions requiring derivation of implicit relation-
ships by combining triples from a knowledge graph.
Example Question : Who is the maternal grandfather of Abraham Lin-
coln?✓
Triples : (Abraham Lincoln, mother, Nancy Hanks Lincoln); (Nancy
Hanks Lincoln, father, James Hanks).
Comparison query : Questions requiring direct comparison of attributes
between entities within the same category.
Example Question : Did Netflix or Google report higher revenue for the
year 2023?
17

Preprint. Under review.
Dataset Question Type Require Synthesize
MultiHop RAGTemporal query : Question that requires an analysis of the temporal
information of the retrieved chunks
Example Question : Did Apple introduce the AirTag tracking device
before or after the launch of the 5th generation iPad Pro?
Null query : Question whose answer cannot be derived from the retrieved
set.This is purposely for testing the issue of hallucination. The LLM
should produce a null response instead of hallucinating an answer.
Example Question : What are the sales of company ABCD as reported in
its 2022 and 2023 annual reports?
Table 5: Dataset Cards.
18

Preprint. Under review.
B Details on Evaluate Dataset Question Types
Below, we provide detailed number of questions for each question type in our evaluate datasets.
Please refer to Appendix A for introduction to the types.
Dataset Question Type Count
2WikiMultiHopCompositional 5,236
Bridge Comparison 2,751
Inference 1,549
MuSiQue2-hop reasoning (linear path) 1,252
3-Hop Reasoning (Linear Path) 568
3-Hop Reasoning (Branching Path) 192
4-Hop Reasoning (Linear Path) 246
4-Hop Reasoning (Branching Path) 64
4-Hop Reasoning (Converging Path) 95
Multihop RAG Inference 816
Table 6: Evaluate data statistics on number of queries and sizes of embedding database.
19

Preprint. Under review.
C Iter-RetGen Prompt Templates
Below we illustrate prompt templates for generating embedding and Iter-RetGen. Templates for
2WikiMultihop and MuSiQue are identical, while for MultiHop-RAG we add source of content as
many of its question decomposition revolve around this. Following previous work (Zhuang et al.,
2024), we adopt the same prompt template on three evaluate datasets for Iter-RetGen.
Document Chunk Prompt Template for 2WikiMultihop and MuSiQue
Title: [doc title]
Context: [doc text]
Table 7: Prompt template for generating embedding using BGE-m3 embedding model on 2Wiki and
MuSiQue train and evaluate datasets.
Document Chunk Prompt Template for MultiHop-RAG
Title: [doc title]
Source: [doc source]
Context: [doc text]
Table 8: Prompt template for generating embedding using BGE-m3 embedding model on MultiHop-
RAG evaluate dataset.
20

Preprint. Under review.
Iter-RetGen Prompt Template for 2WikiMultihop, MuSiQue and MultiHop-RAG
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc>
and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should
be string.
Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: In which year did the publisher of In Cold Blood form?
Let’s think step by step.
<Answer>:
``` json
{{ "thought": "In Cold Blood was first published in book form by Random House. Random House was form in 2001.",
"answer": "2011" }}
```
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
Let’s think step by step.
<Answer>:
``` json
{{ "thought": "The Killing of a Sacred Deer was filmed in Cincinnati. The present Mayor of Cincinnati is John Cranley.
Therefore, John Cranley is in charge of the city.", "answer": "John Cranley" }}
```
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
Let’s think step by step.
<Answer>:
``` json
{{ "thought": "Signal Hill is a hill which overlooks the city of St. John’s. St. John’s is located on the eastern tip of the
Avalon Peninsula.", "answer": "eastern tip" }}
```
Now based on the given doc, answer the question after <Question>.
<doc>
{documents}
</doc>
<Question>: {question}
Let’s think step by step.
<Answer>:
Table 9: Prompt template for Iter-RetGen on 2Wiki, MuSiQue and MultiHop-RAG evaluate datasets.
21