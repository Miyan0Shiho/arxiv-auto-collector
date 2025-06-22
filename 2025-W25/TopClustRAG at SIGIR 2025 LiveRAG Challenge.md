# TopClustRAG at SIGIR 2025 LiveRAG Challenge

**Authors**: Juli Bakagianni, John Pavlopoulos, Aristidis Likas

**Published**: 2025-06-18 08:24:27

**PDF URL**: [http://arxiv.org/pdf/2506.15246v1](http://arxiv.org/pdf/2506.15246v1)

## Abstract
We present TopClustRAG, a retrieval-augmented generation (RAG) system
developed for the LiveRAG Challenge, which evaluates end-to-end question
answering over large-scale web corpora. Our system employs a hybrid retrieval
strategy combining sparse and dense indices, followed by K-Means clustering to
group semantically similar passages. Representative passages from each cluster
are used to construct cluster-specific prompts for a large language model
(LLM), generating intermediate answers that are filtered, reranked, and finally
synthesized into a single, comprehensive response. This multi-stage pipeline
enhances answer diversity, relevance, and faithfulness to retrieved evidence.
Evaluated on the FineWeb Sample-10BT dataset, TopClustRAG ranked 2nd in
faithfulness and 7th in correctness on the official leaderboard, demonstrating
the effectiveness of clustering-based context filtering and prompt aggregation
in large-scale RAG systems.

## Full Text


<!-- PDF content starts -->

arXiv:2506.15246v1  [cs.CL]  18 Jun 2025TopClustRAG at SIGIR 2025 LiveRAG Challenge
Juli Bakagianni∗, John Pavlopoulos∗†, Aristidis Likas‡
∗Athens University of Economics and Business, Greece
Email: {julibak, ipavlopoulos }@aueb.gr
†Archimedes, Athena Research Center, Greece
‡Computer Science and Engineering, University of Ioannina, Greece
Email: arly@cs.uoi.gr
Abstract —We present T OPCLUST RAG, a retrieval-augmented
generation (RAG) system developed for the LiveRAG Challenge,
which evaluates end-to-end question answering over large-scale
web corpora. Our system employs a hybrid retrieval strategy
combining sparse and dense indices, followed by K-Means
clustering to group semantically similar passages. Representa-
tive passages from each cluster are used to construct cluster-
specific prompts for a large language model (LLM), generating
intermediate answers that are filtered, reranked, and finally syn-
thesized into a single, comprehensive response. This multi-stage
pipeline enhances answer diversity, relevance, and faithfulness
to retrieved evidence. Evaluated on the FineWeb Sample-10BT
dataset, T OPCLUST RAG ranked 2nd in faithfulness and 7th
in correctness on the official leaderboard, demonstrating the
effectiveness of clustering-based context filtering and prompt
aggregation in large-scale RAG systems.
I. I NTRODUCTION
Retrieval-Augmented Generation (RAG) has emerged as a
promising paradigm to enhance the factuality and contextual
grounding of large language models (LLMs), particularly in
open-domain question answering. By supplementing genera-
tion with document retrieval, RAG systems aim to produce
responses that are both informative and faithful to source
material. However, designing effective RAG pipelines remains
challenging due to issues such as noisy retrieval results, redun-
dant or semantically overlapping passages, and the difficulty
of aggregating diverse evidence into a coherent response.
To advance research in this domain, the Technology In-
novation Institute (TII) organized the LiveRAG Challenge,1
a leaderboard-based competition that evaluates end-to-end
RAG systems at scale. Participants were tasked with building
retrieval-augmented QA systems over the FineWeb Sample-
10BT corpus Penedo et al. (2024), a 10-billion-token web
dataset, and were evaluated on correctness and faithfulness
using metrics derived from LLM-based automatic assessments.
Systems were required to operate in a low-latency setting,
with strong emphasis on faithfulness to retrieved evidence and
informativeness of generated answers.
In this paper, we present T OPCLUST RAG, our submission
to the LiveRAG Challenge. T OPCLUST RAG is a multi-stage
RAG system designed to improve answer quality through hy-
brid retrieval, clustering-based content selection, and prompt-
based synthesis. Our pipeline clusters semantically similar
passages retrieved via a hybrid sparse-dense index, selects
1https://liverag.tii.ae/representative passages from each cluster, and prompts an
LLM to generate intermediate answers. These responses are
then filtered, re-ranked, and synthesized into a final answer.
Through this design, T OPCLUST RAG balances retrieval di-
versity with answer precision.
Our system ranked 2nd in faithfulness and7th in correct-
ness in the first session of the leaderboard, demonstrating that
cluster-based prompt diversification and re-ranking can yield
highly grounded and accurate responses. We provide a detailed
description of our system components, synthetic evaluation
setup, and leaderboard performance in the following sections.
II. S YSTEM DESCRIPTION
In the following sections, we present an overview of our
system, T OPCLUST RAG, along with the key components of
its pipeline.
A. System Overview
Figure 1 illustrates the architecture of our proposed RAG
system, T OPCLUST RAG. The system employs a multi-stage
pipeline designed to enhance answer quality through hybrid
retrieval, clustering-based context filtering, and prompt aug-
mentation.
The process begins by retrieving the top ppassages for a
given query using a hybrid retriever that combines results from
both sparse (BM25) and dense (embedding-based) indices.
The final retrieval scores are computed using Reciprocal
Rank Fusion (RRF), and the top pranked passages form the
candidate pool.
To reduce redundancy and emphasize relevant content, we
apply K-Means clustering to the TF-IDF representations of
the top- ppassages. The optimal number of clusters kis se-
lected by maximizing the average macro-silhouette coefficient
Pavlopoulos et al. (2024), which is robust to class imbalance.
From each cluster, we select the top rrepresentative pas-
sages based on their original hybrid retrieval scores, reflecting
their relevance to the query. Each set of representative passages
from a cluster is used as context in a prompt to Falcon3-10B-
instruct LLM Team (2024), alongside the original query. The
LLM is asked to generate an answer if the required information
is present in the given context.
The generated candidate answers (one per cluster) are
filtered to retain only those that contain substantive responses.2
2We exclude the “I don’t know.” responses.

🌥
 Query BM25 
E5RRF Reranker 
❌
❌
✅
❌
❌
✅
 ✅
Top-p1
Top-p2Top-p3Top-p3 Clustering Top-r of cluster 0 Query Prompts 
Top-r of cluster 1 
Top-r of cluster 2 
Top-r of cluster 3 
Top-r of cluster 4 
Top-r of cluster k LLM Responses 
Final LLM Response ✅
✅
Context Fig. 1. Overview of the T OPCLUST RAG pipeline.
These are then reranked using the cross-encoder/ms-marco-
MiniLM-L6-v23CrossEncoder reranker. Finally, all reranked
answers are provided to the LLM in an aggregated prompt,
instructing it to synthesize a single, final answer based on the
combined evidence.
B. Synthetic Data Generation
The LiveRAG Challenge is based on the FineWeb Sample-
10BT dataset Penedo et al. (2024), a collection of 14.9 million
documents and 10 billion tokens randomly sampled from
FineWeb. This dataset includes diverse web content, such as
news articles, blogs, academic texts, and product descriptions.
As the challenge organizers did not provide a validation set,
we constructed a synthetic validation dataset using the method
proposed by Filice et al. (2025), applied to the FineWeb
Sample-10BT collection. The resulting dataset comprises 27
entries, covering 22 unique question categories and six distinct
user categories. Notably, three of the question–answer pairs
are supported by two different documents from the dataset,
allowing for multi-source reasoning.
C. System Modules
1) Retrieval System: Our retrieval system builds upon the
pre-constructed indices provided by the challenge organizers:
a sparse index based on BM25 (via OpenSearch)4and a dense
index based on intfloat/e5-base-v2 Wang et al. (2022) hosted
on Pinecone.5In addition to evaluating each retriever individ-
ually, we explored a hybrid retrieval strategy that combines
the outputs of both using Reciprocal Rank Fusion (RRF). For
each query, documents received fused scores based on their
3https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
4https://opensearch.org/
5https://www.pinecone.io/TABLE I
RETRIEVAL PERFORMANCE OF SPARSE ,DENSE ,AND HYBRID SYSTEMS ON
THE SYNTHETIC VALIDATION SET .
System MRR R@1 R@5 R@10 R@50 R@100 R@200 R@1000
Sparse 0.3361 0.2037 0.4074 0.4815 0.5741 0.6481 0.7593 0.8704
Dense 0.0526 0.0000 0.0926 0.1111 0.2963 0.3333 0.3519 0.5556
Hybrid 0.1322 0.0370 0.1111 0.3519 0.6852 0.7778 0.8519 0.8889
reciprocal ranks in the sparse and dense results, and were
subsequently re-ranked to produce a unified top- klist.
To select the most suitable retrieval method, we evaluated
the three systems on the synthetic validation dataset using
standard information retrieval metrics. Specifically, we report
Mean Reciprocal Rank (MRR) Baeza-Yates and Ribeiro-Neto
(1999), which captures the average inverse rank of the first
relevant document, and Recall at rank k(R@1, R@5, R@10,
R@50, R@100, R@200, R@1000), which measures whether
at least one relevant document is retrieved within the top k
results.
As shown in Table I, the sparse retriever outperforms both
the dense and hybrid methods in terms of MRR and R@1
through R@10, indicating superior early precision. However,
the hybrid retriever achieves the highest recall at deeper ranks
(R@50 and beyond), which is critical for our system, as it
benefits from access to relevant passages beyond the top-
10. Based on this trade-off, we selected the hybrid retrieval
approach for the final system deployment.
2) Passage Embedding and Dimensionality Reduction:
To enable clustering over the retrieved passages, we first
represent each passage using TF-IDF vectors, which capture
the importance of terms across the corpus. Given the high
dimensionality of TF-IDF representations, we apply Singular
Value Decomposition (SVD) to reduce each vector to 100

dimensions. This dimensionality reduction step helps preserve
semantic structure while improving computational efficiency.
We opted for TF-IDF embeddings primarily due to their
speed, which was a key consideration in the challenge setting.
For comparison, we also experimented with contextual em-
beddings derived from the billatsectorflow/stella en1.5B v5
model (Zhang et al., 2025), a language model that offers a
favorable trade-off in terms of MTEB performance,6memory
consumption, and token capacity.
Table II shows that T OPCLUST RAG achieves com-
parable—and in terms of BERTScore F1, even supe-
rior—performance using the simpler TF-IDF representations.
This empirical result justifies our choice of TF-IDF as the
default embedding method, balancing efficiency and effective-
ness.
TABLE II
ROUGE-L AND BERTS CORE F1OFTOPCLUST RAG USING
CONTEXTUAL (STELLA 1.5B)VS. TF-IDF REPRESENTATIONS .
Embedding Type ROUGE-L BERTScore F1
contextual embeddings 0.217 0.531
TF-IDF 0.214 0.546
3) Clustering of Retrieved Passages: To group semanti-
cally similar passages, we apply K-Means clustering to the
dimensionality-reduced passage embeddings. The number of
clusters kis not fixed a priori; instead, it is selected dy-
namically for each query by maximizing the macro-averaged
silhouette coefficient, a metric that captures both intra-cluster
cohesion and inter-cluster separation. This approach ensures
robustness to potential class imbalance in the distribution of
relevant content across clusters Pavlopoulos et al. (2024). This
clustering step serves two purposes: it filters out redundant
content, and it enables the generation of diverse candidate
answers by isolating different semantic aspects of the retrieved
content.
4) Selection of Retrieved Passages: We select rrepresen-
tative passages per cluster to serve as context for answer
generation. We compare two strategies for this selection:
(i) ranking passages by their original hybrid retrieval scores
(reflecting query relevance), and (ii) ranking by proximity to
the cluster centroid. Table III shows that selecting passages
based on retrieval scores significantly outperforms centroid-
based selection in Recall. Consequently, we adopt score-based
selection as the default strategy. We determine the optimal
value of rto be 5, because adding more than 5 passages (R@5-
R@10) does not improve Recall.
5) Cluster-Based Prompt Construction: For each cluster,
the selected representative passages are concatenated to form
a single prompt. This results in one prompt per cluster, each
tailored to a specific semantic grouping of the retrieved con-
tent. This approach preserves topical diversity and enhances
relevance by isolating distinct thematic components. Table IV
shows the instruction template used for prompt construction.
6https://huggingface.co/spaces/mteb/leaderboardTABLE III
RECALL @R FOR PASSAGE SELECTION STRATEGIES ,BY QUERY
RELEVANCE SCORE OR BY PROXIMITY TO THE CLUSTER CENTROID (FOR
CLUSTERS CONTAINING AT LEAST ONE GOLD PASSAGE ).
Metric R@1 R@2 R@3 R@4 R@5 R@10
Distance 0.00 0.04 0.19 0.26 0.30 0.48
Score 0.48 0.67 0.70 0.85 0.93 0.93
TABLE IV
INSTRUCTION TEMPLATE FOR RAG.
Answer the question using only the context below. Do not make up any new
information. If no part of the answer is found in the context, respond only
with: ‘‘I don’t know.’’ If only part of the answer is found, include
that part in a complete sentence that uses the phrasing of the question, and
state that the rest is not available in the context. If the full answer is found,
respond with a complete sentence that includes the phrasing of the question.
Context:
<1st text>
<2nd text>
<5th text>
Question: <question>
Answer:
6) Intermediate Response Generation: Each cluster-
specific prompt is independently processed by the Falcon3-
10B-instruct LLM to generate one intermediate response per
cluster. This step enables the model to generate focused
answers grounded in diverse subsets of the retrieved passages,
increasing coverage of relevant content across semantic clus-
ters.
7) Final Response Synthesis: Intermediate responses
are first filtered to remove non-informative outputs,
meaning those containing “I don’t know.” The
remaining responses are then scored using the
cross-encoder/ms-marco-MiniLM-L6-v2 model,
which evaluates their relevance to the original query. The
top-ranked responses are concatenated into a final aggregated
context, which is used as input for a final prompt to the
language model. The instruction template employed is the
same as in the cluster-based prompt construction step,
enabling the model to generate a single, comprehensive
answer based on consolidated evidence.
III. E XPERIMENTAL SETUP
All experiments, including the Live Challenge Day session,
were conducted on Google Colab using CPU resources. The
language model employed was Falcon3-10B-Instruct running
on the AI71 platform,7as provided by the challenge organiz-
ers. Retrieval leveraged the prebuilt indices from OpenSearch
(sparse) and Pinecone (dense). For each question, we retrieved
200 passages from OpenSearch and 200 from Pinecone. These
results were combined using our hybrid retrieval strategy, and
the top 100 passages were retained for downstream processing.
The LiveRAG Challenge Day event lasted two hours and
consisted of 500 questions. To maximize throughput, we
utilized ten parallel requests per query and eight parallel AI71
7https://ai71.ai/

API clients. Our system processed all 500 questions within
approximately one hour.
IV. E VALUATION ON LEADERBOARD
We submitted our system to the leaderboard evaluation,
which ranks participating systems based on two metrics:
Correctness and Faithfulness . The former ranges from -1
to 2 and combines two components: (i) coverage , defined as
the proportion of vital information—identified by a strong
LLM—in the ground truth answer that is present in the
generated response Pradeep et al. (2025); and (ii) relevance ,
which measures the extent to which the generated response
directly addresses the question, regardless of factual correct-
ness. The latter ranges from -1 to 1 and evaluates whether the
generated response is grounded in the retrieved passages Es
et al. (2024). We participated in the first evaluation round,
ranking 7th in correctness with a score of 0.685146, and 2nd
in faithfulness with a score of 0.460062. Out of 500 evaluation
questions, our system responded with “I don’t know.” to
109 questions—receiving a zero score for both metrics on
those instances—while it generated substantive answers for
the remaining queries.
V. E XPERIMENTS
a) Baselines: We compare our T OPCLUST RAG system
three baselines. The first baseline uses the top- kpassages
ranked by the retrieval score, where k∈ {0,1,5}, representing
common values. The second baseline is an ablation of our
system that ignores clustering. It retrieves ten passages accord-
ing to the retrieval score, which are split into five batches of
five passages, then processed following the T OPCLUST RAG
steps II-C5–II-C7–without clustering. The third baseline
extends the second baseline by selecting an optimal kvalue
per query, following the approach T OPCLUST RAG. For each
query, kis defined by the number of clusters (see the clustering
step of §II-C3) multiplied by five (see §II-C4).
b) Evaluation metrics: Evaluation is performed using
ROUGE-L and BERTScore F1 and Table V summarizes the
results for the baselines against our T OPCLUST RAG system.
We observe that when no passages are retrieved (top-0),
ROUGE-L is near zero while BERTScore F1 is relatively
higher. In this setting, 25 out of 27 responses return “I don’t
know”, which means that BERTScore may capture general
fluency or semantic plausibility even in the absence of content-
specific grounding. However, even though we observe that
both ROUGE-L and BERTScore are not ideal for this task—
since answers may be found across multiple documents and
not solely in the gold reference(s)—they offer useful insights
when comparing systems under the same input constraints.
c) Results: The top three rows of Table V show that
TOPCLUST RAG outperforms the first baseline for common k
values. The second baseline (top-10), which uses batching and
employs an LLM, T OPCLUST RAG is overall better, perform-
ing slightly worse in ROUGE-L but better in BERTScore F1.
The third baseline, i.e., the top- 5kablation, is better than top-5
(of the 1st baseline) but worse than top-10 (2nd baseline) andTOPCLUST RAG. Although it uses approximately the same
number of passages as T OPCLUST RAG (without clustering),
performance is worse likely due to diminishing returns (addi-
tional passages contribute noise) and lack of structure (pas-
sages are not organized to ensure diverse topical coverage).
This suggests that clustering contributes meaningfully beyond
simple context size expansion. Specifically, the structured
selection of representative passages from each cluster (rather
than random or purely relevance-based sampling) helps ground
the generation process in more diverse and complementary
information. We note that there may be kvalues for which
the 1st baseline could be the best, but the selection of the best
kvalue is arbitrary and cannot guarantee robustness.
TABLE V
ROUGE-L AND BERTS CORE F1FOR THE THREE BASELINES ,TOP-k
WITH k∈ {0,1,5},TOP-10, AND TOP -5k,COMPARED TO OUR
TOPCLUST RAG. T HE BEST RESULTS ARE SHOWN IN BOLD .
ROUGE-L BERTScore F1
top-0 0.031 0.317
top-1 0.186 0.494
top-5 0.202 0.515
top-10 0.217 0.541
top-5k 0.202 0.532
TopClustRAG 0.214 0.546
VI. C ONCLUSIONS
In this study, we presented T OPCLUST RAG, our RAG
system designed for large-scale question answering over di-
verse web data in the LiveRAG Challenge. By leverag-
ing a hybrid sparse-dense retrieval approach combined with
clustering-based filtering, our system effectively balances
retrieval diversity and relevance. The proposed multi-stage
pipeline—comprising cluster-based prompt construction, in-
termediate answer generation, re-ranking, and final synthe-
sis—enhances the faithfulness and informativeness of the
generated responses.
Our evaluation on the FineWeb Sample-10BT dataset
demonstrated that T OPCLUST RAG achieves competitive
leaderboard performance, ranking 2nd in faithfulness and 7th
in correctness in the first session. These results highlight
the potential of clustering and prompt aggregation techniques
to improve large language model grounding in retrieval-
augmented generation. In the future, we plan to extend cluster-
ing to handle larger numbers of retrieved passages, assessing
whether clustering can help uncover relevant information that
may be hidden within the extensive data.
REFERENCES
Ricardo A. Baeza-Yates and Berthier Ribeiro-Neto. 1999.
Modern Information Retrieval . Addison-Wesley Longman
Publishing Co., Inc., USA.
Shahul Es, Jithin James, Luis Espinosa Anke, and Steven
Schockaert. 2024. Ragas: Automated evaluation of retrieval

augmented generation. In Proceedings of the 18th Con-
ference of the European Chapter of the Association for
Computational Linguistics: System Demonstrations . 150–
158.
Simone Filice, Guy Horowitz, David Carmel, Zohar Karnin,
Liane Lewin-Eytan, and Yoelle Maarek. 2025. Generating
Diverse Q&A Benchmarks for RAG Evaluation with Data-
Morgana. arXiv preprint arXiv:2501.12789 (2025).
John Pavlopoulos, Georgios Vardakas, and Aristidis Likas.
2024. Revisiting Silhouette Aggregation. In International
Conference on Discovery Science . Springer, 354–368.
Guilherme Penedo, Hynek Kydl ´ıˇcek, Loubna Ben allal, Anton
Lozhkov, Margaret Mitchell, Colin Raffel, Leandro V on
Werra, and Thomas Wolf. 2024. The FineWeb Datasets:
Decanting the Web for the Finest Text Data at Scale. In
The Thirty-eight Conference on Neural Information Pro-
cessing Systems Datasets and Benchmarks Track . https:
//openreview.net/forum?id=n6SCkn2QaG
Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel
Campos, Nick Craswell, and Jimmy Lin. 2025. The
Great Nugget Recall: Automating Fact Extraction and RAG
Evaluation with Large Language Models. arXiv preprint
arXiv:2504.15068 (2025).
TII Team. 2024. The Falcon 3 family of Open Models.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun
Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022.
Text Embeddings by Weakly-Supervised Contrastive Pre-
training. arXiv preprint arXiv:2212.03533 (2022).
Dun Zhang, Jiacheng Li, Ziyang Zeng, and Fulong Wang.
2025. Jasper and Stella: distillation of SOTA embed-
ding models. arXiv preprint arXiv:2412.19048 (2025).
doi:10.48550/arXiv.2412.19048