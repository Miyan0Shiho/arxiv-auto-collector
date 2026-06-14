# Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses

**Authors**: Sooho Moon, Jian Kang, Yunyong Ko

**Published**: 2026-06-08 02:00:38

**PDF URL**: [https://arxiv.org/pdf/2606.08921v1](https://arxiv.org/pdf/2606.08921v1)

## Abstract
Knowledge graph completion (KGC) aims to predict missing facts from an observed knowledge graph (KG), playing a crucial role in a wide range of real-world applications such as drug discovery, recommender systems, and retrieval-augmented generation (RAG). Although numerous KGC models have been proposed, the evaluation of KGC remains underexplored, despite its critical role in reliably assessing model performance and selecting appropriate models for real-world applications. In this paper, we introduce two important perspectives for KGC evaluation that are overlooked by existing evaluation metrics, (P1) predictive sharpness and (P2) popularity-bias robustness. To address both perspectives, we propose a generalized evaluation framework, PROBE, which consists of a rank transformer (RT) that estimates the score of each prediction based on a desired level of predictive sharpness and a rank aggregator (RA) that determines the final evaluation score by aggregating all prediction scores according to a desired level of popularity-bias robustness. We theoretically analyze PROBE by defining six key properties for reliable KGC evaluation and prove that PROBE satisfies all the properties, while existing metrics fail to satisfy some. In particular, due to the open-world nature of KGs, an evaluation metric should preserve the relative performance of KGC models even when only incomplete facts are observed. We show that PROBE better maintains such consistency, providing a more reliable estimate of intrinsic model performance than existing metrics. Extensive experiments with six KGC models on six real-world KGs reveal that existing metrics may over- or under-estimate model performance depending on different evaluation perspectives, whereas PROBE enables a more comprehensive, flexible, and consistent evaluation of KGC models.

## Full Text


<!-- PDF content starts -->

Generalized Rank-based Evaluation for Knowledge Graph Completion:
Perspectives, Framework, and Analyses
SOOHO MOON,Chung-Ang University, South Korea
JIAN KANG,Mohamed bin Zayed University of Artificial Intelligence, UAE
YUNYONG KO∗,Chung-Ang University, South Korea
Knowledge graph completion (KGC) aims to predict missing facts from an observed knowledge graph (KG), playing a crucial role in
a wide range of real-world applications such as drug discovery, recommender systems, and retrieval-augmented generation (RAG).
Although numerous KGC models have been proposed, the evaluation of KGC remains underexplored, despite its critical role in
reliably assessing model performance and selecting appropriate models for real-world applications. In this paper, we introduce two
important perspectives for KGC evaluation that are overlooked by existing evaluation metrics,(P1)predictive sharpness— the degree
of strictness in penalizing inaccurate predictions — and(P2)popularity-bias robustness— the ability to evaluate predictions in a
popularity-aware manner by accounting for both entity and relation popularity bias. To address both perspectives, we propose a
generalized evaluation framework,PROBE, which consists of a rank transformer (RT) that estimates the score of each prediction
based on a desired level of predictive sharpness and a rank aggregator (RA) that determines the final evaluation score by aggregating
all prediction scores according to a desired level of popularity-bias robustness. We theoretically analyzePROBEby defining six key
properties for reliable KGC evaluation and prove thatPROBEsatisfies all the properties, while existing metrics fail to satisfy some.
In particular, due to the open-world nature of KGs, an evaluation metric should preserve the relative performance of KGC models
even when only incomplete facts are observed. We show thatPROBEbetter maintains such consistency, providing a more reliable
estimate of intrinsic model performance than existing metrics. Extensive experiments with six KGC models on six real-world KGs
reveal that existing metrics may over- or under-estimate model performance depending on different evaluation perspectives, whereas
PROBEenables a more comprehensive, flexible, and consistent evaluation of KGC models. Our code and datasets are available at
https://github.com/potato2734/probe-kgc-evaluation.
CCS Concepts:•Information systems →Data mining;•Computing methodologies →Knowledge representation and
reasoning.
Additional Key Words and Phrases: knowledge graphs, knowledge graph completion, rank-based evaluation, open-world assumption
1 Introduction
A knowledge graph (KG) is a graph-structured representation of real-world knowledge, where an entity is represented as
a node and a relation between two entities is represented as an edge in the form of a triple (ℎ,𝑟,𝑡) . KGs [ 4,8,29,37,40]
have been widely used in a wide range of applications such as question answering (QA) [ 15,17,56,58,62], news
classification [ 12,20,61], recommender systems [ 49,50,64], drug discovery [ 5,23,59,60], and retrieval-augmented
generation (RAG) [ 9,11,13,14,33]. For example, in RAG, a KG can be used to better understand a user query and
retrieve more query-relevant information, thereby enabling a large language model (LLM) to generate high-quality
responses [ 19,25,28]. In the pharmaceutical domain, biomedical KGs can be applied to drug discovery by providing
useful information about complex relationships among diseases, genes, and compounds, which significantly reduces
the cost and time required to develop new medications [59].
∗Corresponding author.
Authors’ Contact Information: Sooho Moon, Chung-Ang University, Seoul, South Korea, moonwalk725@cau.ac.kr; Jian Kang, Mohamed bin Zayed
University of Artificial Intelligence, Abu Dhabi, UAE, jian.kang@mbzuai.ac.ae; Yunyong Ko, Chung-Ang University, Seoul, South Korea, yyko@cau.ac.kr.
1arXiv:2606.08921v1  [cs.LG]  8 Jun 2026

2 Moon et al.
Rank Prediction Rank Transformation Rank Aggregation
A
𝓖𝒕𝒔𝒕
B
1
2
50
2
2
50.51
0.66MRR 
Hits@5
MRR 
Hits@51
1 / 2
1 / 50
1
1 
0
0.40
1.001 / 2
1 / 2
1 / 5
1
1 
1?
?
?
Fig. 1. Rank-based evaluation protocol for knowledge graph completion: (1) prediction, (2) transformation, and (3) aggregation.
Real-world KGs, however, are inherently incomplete [ 42,46], i.e., a number of facts are missing. This fundamental
limitation can hinder the potential of KGs in real-world applications. To address this limitation,knowledge graph
completion(KGC) has been widely studied [ 7,24,34–36,42,46,53,63], which aims to infer missing facts based on the
observed KG structure. Specifically, given a KG, it predicts the missing entity when either the head or the tail entity is
unknown, i.e., in the form of (ℎ,𝑟, ?)or(?,𝑟,𝑡) . Despite recent breakthroughs in KGC models, little attention has been
paid to theevaluationof KGC models. Without appropriate evaluation, a suboptimal model may be selected, leading to
low-quality knowledge and degraded performance in downstream tasks.
Motivated by this, we revisit the evaluation protocol for KGC models. To evaluate KGC models,rank-basedevaluation
metrics (e.g., mean rank (MR), mean reciprocal rank (MRR), and Hits@K) are commonly adopted [ 7,16,30,31,53,55]
due to theopen-world assumption(OWA) [ 16,55], where the absence of a triple in a KG does not necessarily imply
that it is false. Specifically, the rank-based evaluation process is as follows: Given a trained KGC model and a set of
test triples,(1)(Prediction) it computes the probability that each candidate entity is the missing entity for each triple –
predicting the missing entity in the form of (ℎ,𝑟, ?)or(?,𝑟,𝑡) – and ranks the candidates based on their probabilities;(2)
(Transformation) it transforms the rank of the missing entity into a prediction score; and(3)(Aggregation) it aggregates
the scores of all predictions to compute the final accuracy. Depending on the evaluation metric, the specific procedures
for the (2) transformation and (3) aggregation steps can vary.
Examples.As illustrated in Figure 1, consider two KGC models A and B evaluated on three test triples. Suppose that
model A ranks the target entities at [1,2,50], whereas model B ranks them at [2,2,5]. From the perspective of the
evaluation procedure, their difference arises from how ranks are transformed into scores and aggregated. For MRR, each
rank is transformed using the reciprocal function 𝑓(𝑟)=1
𝑟. Thus, the transformed scores are
1,1
2,1
50
for model A and1
2,1
2,1
5
for model B. In the aggregation step, the final score is computed as the average of the transformed scores:
MRR𝐴=1
3 1+1
2+1
50=0.51,andMRR𝐵=1
3 1
2+1
2+1
5=0.40. For Hits@5, the transformation step assigns a binary
score depending on whether the rank of the target entity is within the top 5 predictions: 𝑓(𝑟)= 1if𝑟≤ 5,otherwise 0.
Accordingly, the transformed scores of model A and model B are [1,1,0] and [1,1,1], respectively. In the aggregation
step, the final accuracy is computed as the average of the transformed scores: Hits@5𝐴=1
3(1+1+0)=0.66and
Hits@5𝐵=1
3(1+1+1)=1.00. This example shows that the preferred model depends on the evaluation metric: MRR
favors model A due to its stronger emphasis ontop-rankedpredictions, whereas Hits@5 favors model B because it
considers only whether the correct entity appears within the top-𝐾results.

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 3
Ⅰ
Approve!
ⅢⅡⅠ
ⅢⅡⅠ
Ⅰ
ⅡⅠ
Ⅰ
ⅢⅡ
< Distribution of Rank Predictions >
Obs 1.“To mitigate the cost and delays of trial failures,
we require models with high confidence”
“Identifying novel relationships involving 
low-popularity entities provides greater utility”
ⅢⅡⅠ
ⅢⅡⅠ
ⅢⅡⅠ
target
?
?
Obs 2.< Performance Difference On Entity Popularity >
associatepopular
rare
Fig. 2. Motivating examples illustrating the importance of(P1)predictive sharpness and(P2)popularity-bias robustness in KGC
evaluation. Depending on the desired evaluation perspective, different KGC models may be preferred.
1.1 Motivation
However, we observe that existing rank-based evaluation metrics overlook two subtle yet important perspectives for
evaluating KGC models: (P1)predictive sharpnessand (P2)popularity-bias robustness. We present motivating examples
and conduct preliminary experiments to justify the importance of these perspectives for KGC evaluation.
(P1) Predictive sharpness.Naturally, we may answer the question “how strictly should we evaluate each individual
prediction?" differently depending on the application. For example, when biomedical KGs are used for drug discovery,
inaccurate (i.e., less faithful) predictions may lead to serious risks, such as harmful side effects, reduced effectiveness,
and substantial clinical trial costs [ 48,59]. Thus, in this case, as predictions become less accurate, a larger penalty should
be imposed on the KGC model. In other words, a high level ofpredictive sharpnessis required in evaluating KGC models.
In contrast, when commonsense KGs are used in recommender systems, a relatively low level of predictive sharpness
may be acceptable as long as the newly predicted facts are beneficial for better understanding users and items.
Existing metrics, however, do not consider predictive sharpness in KGC evaluation. To verify our claim, we analyze
the distribution of predictions from two state-of-the-art KGC models (RotatE [ 42] and RNNLogic [ 35]). As shown in
Figure 2, RNNLogic produces more 1st-rank predictions, whereas RotatE performs better across ranks 2–100. This
implies that RNNLogic (resp. RotatE) is preferable when a high (resp. low) level of predictive sharpness is required.
However, existing metrics consistently assign higher scores to RNNLogic than RotatE.
(P2) Popularity-bias robustness.Real-world graphs generally follow a power-law degree distribution [ 18,21,51,52],
meaning that most entities appear in only a few triples (low popularity), while a small number of entities appear in a
large number of triples (high popularity). Thus, KGC models may exhibitpopularity bias, since high-popularity entities
and relations are used much more frequently during training. From an application perspective, the robustness of KGC
models to popularity bias is often important in [ 6,57]. For example, in biomedical KGs for drug discovery, practitioners
are typically more interested in discovering new facts associated with low-popularity entities and relations (e.g., rare
diseases, less-investigated genes, or novel compounds), rather than repeatedly confirming well-known facts associated
with high-popularity ones [2, 39, 65]. Thus, a high level ofpopularity-bias robustnessis required in KGC evaluation.
However, existing metrics fail to reflect this perspective and tend to overestimate KGC models that perform well
on high-popularity triples. To verify this, we measure the KGC accuracy (MRR) of RotatE and RNNLogic across
different popularity levels. As shown in Figure 2, both models achieve much higher accuracy on high-popularity

4 Moon et al.
triples, confirming strong popularity bias. Interestingly, although RotatE outperforms RNNLogic across most popularity
groups, RNNLogic achieves higher overall accuracy, indicating that existing metrics fail to properly reflect robustness
to popularity bias in KGC evaluation.
1.2 Our Work
To reflect both perspectives(P1)and(P2), we propose a generalized framework for KGC evaluation, named Predictive
shaRpness and p Opularity- Bias robustness aware Evaluation (PROBE).PROBEconsists of a rank transformer (RT)
and a rank aggregator (RA). For each test triple, the RT transforms the rank of the missing entity predicted by a KGC
model into a score, based on the required level of predictive sharpness(P1). Then, the RA assigns a weight to each test
triple based on the desired level of popularity-bias robustness(P2), and computes the final accuracy score by taking
the weighted average of all transformed scores. In particular, for(P2), we observe that popularity bias varies across
entities and relations. This is because the popularity of a triple is jointly determined by entities and relations, i.e.,
even a high-popularity entity may rarely appear with a specific relation, and vice versa. Based on this observation,
PROBEseparately measures(P2)-(i)entity popularity bias and(P2)-(ii)relation popularity bias for each test triple, and
determines the final weight based on the two bias measures. Note that existing metrics can be interpreted as special
cases ofPROBEunder specific configurations of RT and RA.
In addition, we theoretically analyze thatPROBEsatisfies six key properties required for reliable KGC evaluation,
while existing metrics fail to satisfy some of them. Importantly, under the open-world nature of KGs, accurately assessing
the intrinsic performance of a KGC model is a critical requirement for evaluation metrics, i.e., a metric should preserve
the relative performance of models even when only a partial set of facts is observed. In other words, if two models have
similar performance in the open world, their performance should remain consistent when evaluated in a closed-world
setting where all facts are observable. In this regard, we show thatPROBEprovides a reliable estimate of the intrinsic
performance of KGC models in open-world settings.
Through comprehensive experiments with six KGC models on six real-world KGs, we reveal the following findings.
(1)Existing rank-based metrics implicitly assume a high level of predictive sharpness, favoring KGC models that produce
top-ranked predictions while underestimating those that produce imperfect yet high-quality predictions across broader
rank ranges (e.g., 2nd–5th ranks).(2)KGC models highly evaluated by existing metrics tend to exhibit strong popularity
bias. Such models often perform poorly on low-popularity triples. These findings reveal inherent limitations of existing
rank-based metrics in evaluating KGC models, as they overlook the crucial perspectives of predictive sharpness (P1)
and popularity-bias robustness (P2). In contrast,(3)PROBEenable more comprehensive evaluation of KGC models by
flexibly reflecting different levels of predictive sharpness and popularity-bias robustness. Finally,(4)PROBEconsistently
preserves the relative performance of KGC models across open-world and closed-world settings, demonstrating reliable
evaluation under the open-world assumption.
1.3 Contributions
The main contributions of this work are summarized as follows.
•New Perspectives: We introduce two key perspectives for KGC evaluation,(P1)predictive sharpness and
(P2)popularity-bias robustness, which capture important aspects of KGC model performance overlooked by
existing evaluation metrics.

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 5
•Generalized Framework: We propose a generalized evaluation framework,PROBE, that comprehensively
evaluates KGC models by flexibly incorporating different levels of(P1)predictive sharpness and(P2)popularity-
bias robustness.
•Theoretical Analysis: We provide a theoretical analysis showing thatPROBEsatisfies six key properties
required for reliable KGC evaluation. We further analyze model evaluation under open-world and closed-world
settings, demonstrating thatPROBEpreserves the relative performance of KGC models across the two settings.
•Comprehensive Experiments: Through extensive experiments using six KGC models on six real-world KGs,
we quantitatively and qualitatively demonstrate thatPROBEprovides more reliable and consistent evaluation
results, particularly in preserving the relative performance of KGC models under the open-world assumption.
The paper is organized as follows. Section 2 briefly reviews recent KGC models and existing evaluation metrics for
KGC models. Section 3 describes our evaluation framework,PROBE. Section 4 theoretically analyzes key properties of
PROBEand the ability to assess model performance under the open-world setting. Section 5 presents the experimental
setup and results with in-depth analysis. Finally, Section 6 concludes this paper.
2 Related Work
In this section, we review common protocols and rank-based metrics for KGC evaluation, and introduce recent knowledge
graph completion (KGC) models.
2.1 Evaluation of Knowledge Graph Completion
KGC models are commonly evaluated by rank-based metrics such as mean rank (MR), mean reciprocal rank (MRR),
and Hits@K. Despite their widespread adoption, these metrics have several limitations, including sensitivity to the
open-world assumption, inability to account for popularity bias, and difficulty in comparing results across datasets. To
address these limitations, a handful of studies have revisited rank-based evaluation for KGC. Yang et al . [55] showed
that existing metrics yield inconsistent results under the open-world assumption and proposed alternative variants
such as log-MRR and p-MRR. However, these metrics do not address popularity bias. Mohamed et al . [30] introduced
stratified metrics (e.g., strat-MRR and strat-Hits@K) that assign different weights to test triples based on entity and
relation popularity. While effective for mitigating popularity bias, they do not control the level of predictive sharpness.
Berrendorf et al . [3] observed that rank-based scores are generally comparable only within the same dataset, since rank
interpretation depends on the number of candidate entities. They proposed the adjusted mean rank index (AMRI), a
normalized version of MR that enables meaningful comparison across datasets. Hoyt et al . [16] further extended AMRI
to z-adjusted MRR (ZMRR), which normalizes scores using the mean and variance of the rank distribution. Similarly,
Tiwari et al . [44] proposed weighted geometric mean rank (WMR) that assigns higher importance to predictions made
over larger candidate sets, thereby reflecting the intuition that achieving the same rank in a more difficult setting should
be more highly rewarded.
Those rank-based metrics are commonly computed under the filtered setting [ 7], which removes candidate entities
that correspond to other true triples when ranking the target entity. This prevents unfair penalization due to the
incompleteness of real-world KGs, where multiple entities may form valid triples for the same query. Another important
protocol concerns tie-breaking [ 43]. Since multiple entities may receive identical scores from a KGC model, assigning
the best possible rank to the target entity can lead to overly optimistic evaluation. To address this issue, Sun et al . [43]

6 Moon et al.
proposed a tie-breaking protocol to assign the average rank within the tied group as the final rank, preventing expected
overestimation of model performance.
Recent work [ 31] introduced two important perspectives for KGC evaluation:(P1)predictive sharpness and(P2)
popularity-bias robustness. The work proposed a new evaluation framework reflecting these two perspectives and
analyzed several limitations of existing rank-based metrics. However, its theoretical analysis was limited to only two
properties, i.e., fixed optimum and fixed pessimum, without investigating the reliability of KGC evaluation under the
open-world assumption (OWA). In addition, popularity-bias robustness was modeled only using the frequency of the
missing entity, limiting its ability to capture popularity bias induced by entity-relation interactions. The experimental
evaluation was also limited to four KGC models and two KGs. In this paper, we substantially extend [ 31] in several
directions. First, we extend popularity-bias robustness by incorporating both entity popularity and entity-conditioned
relation popularity. Second, we provide more in-depth theoretical analysis on six key properties for reliable KGC
evaluation as well as consistency under OWA. Third, we conduct comprehensive experiments using six KGC models on
six real-world KGs under diverse evaluation perspectives.
2.2 Knowledge Graph Completion Models
Embedding-based approaches [ 1,7,22,42,46,53] represent entities and relations as vectors in a latent space, preserving
the semantic meaning of triples through algebraic operations. TransE [ 7] models each relation as a translation vector
such that the head entity translated by the relation is close to the tail entity. RotatE [ 42] represents relations as rotations
in complex space to capture diverse relation patterns such as symmetry and inversion. HousE [ 22] employs Householder
transformations to model both rotation and projection for more expressive relations. DistMult [ 53] adopts a bi-linear
scoring function with diagonal matrices to model symmetric relations. ComplEx [ 46] extends DistMult to complex-
valued embeddings to capture asymmetric relations. TuckEr [ 1] applies Tucker tensor decomposition to model rich
interactions between entities and relations.
Rule-based approaches [ 35,36,54] learn logical patterns from relational paths (i.e., sequences of relations), enabling
interpretable reasoning beyond observed triples. NeuralLP [ 54] adopts a recurrent architecture with attention and
auxiliary memory to learn variable-length logical rules. pLogicNet [ 36] employs a probabilistic logic neural network to
learn logical rules while modeling their uncertainty. RNNLogic [ 35] treats logical rules as latent variables and jointly
optimizes a rule generator and a reasoning predictor using the EM algorithm. In addition, various deep learning-based
approaches have also been studied [ 38,41,47,63,66]. R-GCN [ 38] extends graph convolutional networks (GCN)
by applying relation-specific transformations. CompGCN [ 47] jointly learns entity and relation representations via
compositional message passing. NBFNet [ 66] applies a neural Bellman-Ford framework to encode path-based relational
information. RED-GNN [ 63] optimizes propagation in GNNs via dynamic programming for improved efficiency.
MLSAA [ 41] enhances inductive capability by integrating pretrained language models with graph neural networks
through adaptive aggregation and multi-level sampling.
3 Proposed Framework: PROBE
In this section, we present a generalized framework for KGC evaluation, namedPROBE( Predictive sha Rpness and
pOpularity- Bias robustness aware Evaluation. First, we introduce the notations used in this paper and formulate the
problems that we consider. Then, we describe two key components ofPROBE: a rank transformer (Section 3.2) and a
rank aggregator (Section 3.3). Finally, we present a geometric interpretation ofPROBE(Section 3.4).

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 7
Table 1. Notations and their descriptions
Notation Description
G=(E,R,T)Knowledge graph consisting of entities, relations, and triples
E,R,TSets of entities, relations, and triples
(ℎ,𝑟,𝑡)Triple with head entityℎ, relation𝑟, and tail entity𝑡
𝑛Number of test triples times two (𝑛=|T 𝑡𝑒𝑠𝑡|∗2)
𝜃(·)KGC model that assigns a score to a triple
𝑓(·),𝑎𝑔𝑔(·)Rank transformation and aggregation functions
𝑟∈NRank of the correct entity among candidate entities
r=[𝑟 1,...,𝑟𝑛],c=[𝑐 1,...,𝑐𝑛]Rank vector and transformed score vector
𝛼,𝛽Parameters controlling predictive sharpness and popularity-bias robustness
𝛿𝑒,𝛿𝑟|𝑒 Entity popularity and entity-conditioned relation popularity
𝑤𝑒,𝑤𝑟|𝑒 Entity weight and entity-conditioned relation weight
w=[𝑤 1,...,𝑤𝑛]Weight vector for test triples
3.1 Notations and Problem Formulation
The notations used in this paper are described in Table 1. This work focuses onthe evaluation of knowledge graph
completion (KGC). Given 𝑘candidate models, each model is first trained to solve the KGC task, and then evaluated to
determine which model more accurately predicts missing facts. To formalize this process, we consider the following
two related problems: (1) knowledge graph completion and (2) rank-based KGC evaluation.
Problem 1(Knowledge Graph Completion). Given a knowledge graph (KG) G=(E,R,T) , whereEis the set of
entities,Ris the set of relations, and T={((ℎ,𝑟,𝑡)|ℎ,𝑡∈E,𝑟∈R} is the set of triples (i.e., facts), the goal of knowledge
graph completion (KGC) is to infer missing facts based on the observed KGG.
Given a KGC model 𝜃(ℎ,𝑟,𝑡) , its model parameters are typically trained based on the objective that encourage
positive triples to obtain higher scores than negative triples generated through negative sampling. To achieve this
objective, various loss functions have been used, including margin-based ranking loss [ 7], logistic loss [ 22,46], and
cross-entropy loss [1, 35, 46].
Problem 2(Rank-based KGC Evaluation). Given a trained KGC model 𝜃(·)and a set of test triples T𝑡𝑒𝑠𝑡, the goal
of rank-based KGC evaluation is to measure how accurately the model predicts missing entities for each triple (e.g.,
(ℎ,𝑟, ?)or(?,𝑟,𝑡) ) under the open-world assumption. Specifically, rank-based evaluation assesses model performance
based on how highly the correct entity is ranked relative to other candidate entities across test triples, where candidates
are ordered according to the scores produced by the model𝜃(·).
Thus, an effective rank-based metric should appropriately reflect the relative ordering of predicted entities in the
open-world setting and provide reliable assessment of model performance across different prediction scenarios.
Overview ofPROBE.Given a trained KGC model 𝜃(·)and a set of test triples T𝑡𝑒𝑠𝑡,PROBEevaluates the model 𝜃
based on the rank-based evaluation protocol [ 7,22,35,43]. As illustrated in Figure 3,PROBEconsists of the following
three steps: (1) prediction, (2) transformation, and (3) aggregation.
(1) Prediction: For each test triple (ℎ,𝑟,𝑡)∈T𝑡𝑒𝑠𝑡 , two queries are generated by masking either the head or tail entity,
i.e.,(ℎ,𝑟, ?)and(?,𝑟,𝑡) . A KGC model 𝜃(·) then estimates the likelihood of each candidate entity 𝑒′∈E being the

8 Moon et al.
𝛅𝐞
𝛅𝐫|𝐞
rank(A) (B)
scorePopularity𝒢𝑡𝑠𝑡
𝒢𝑡𝑟𝑛
L1 Norm𝐜
WeightPopularity
WeightTransformation 2 Aggregation3
Final Score1Prediction
?
??1
1
0|ℰi|(A)
(B)Models
𝐰𝐞
𝐰𝐫|𝐞𝛼
𝛽Entity popularity
Relation popularity0.15
0.1
0.05
0.16
0.5
0.0
Fig. 3. Overview of PROBE: (1) (prediction) generating ranks for test triples using a KGC model; (2) (transformation) converting ranks
into scores via a predictive sharpness-aware function controlled by 𝛼; and (3) (aggregation) computing the final evaluation score using
popularity-aware weights that capture triple-level popularity controlled by𝛽.
missing entity. Candidate entities are ranked according to their scores, producing the rank1 ≤𝑟≤|E| of the correct
entity. For all queries, a rank vectorr =[𝑟 1,𝑟2,...,𝑟𝑛]is produced, where 𝑛=|T𝑡𝑒𝑠𝑡|∗2(∵two queries for each triple).
(2) Transformation: Givenr, a rank transformation function 𝑓(𝑟,𝛼) :N→R converts each rank into a score based
on the desired level of predictive sharpness 𝛼. The function 𝑓(·)is defined to beanti-monotone, ensuring that lower
numeric rank values yield higher scores. This step produces a score vectorc=[𝑐 1,𝑐2,...,𝑐𝑛].
(3) Aggregation: Givenc, a rank aggregation function 𝑎𝑔𝑔(c,𝛽):R𝑛→R computes the final evaluation score as a
weighted average of the scores inc. The weights are determined based on the popularity of the corresponding triples,
reflecting the desired level of popularity-bias robustness controlled by𝛽.
3.2 Rank Transformation
The rank transformer (RT) ofPROBEconverts the rank of each prediction into a score, considering the required level of
predictive sharpness. Formally, given a rank 𝑟∈rand apredictive sharpness control factor 𝛼, the score is computed as:
𝑓(𝑟,𝛼)=1
𝑟𝛼(𝑟∈r).(1)
When𝛼<0, the function becomesmonotonically increasing, assigning larger scores to worse predictions (larger ranks),
leading undesirable evaluation behavior (e.g., sensitive to worse predictions) [ 16,44]. Thus, we focus on 𝛼>0, ensuring
that lower ranks receive higher scores. Figure 4(a) illustrates the transformed scores for different values of 𝛼. Larger
𝛼imposes heavier penalties on large ranks, emphasizing top-ranked predictions. Conversely, smaller 𝛼reduces the
relative penalty on large ranks, allowing moderately ranked predictions to contribute more to the final score.
We note that this formulation can generalize various rank transformation rules used in existing metrics. For example,
when𝛼=1, the transformation reduces to the reciprocal rank 𝑓(𝑟,1)=1
𝑟used in MRR. As 𝛼=− 1, it degenerates to
the identity function𝑓(𝑟,−1)=𝑟, corresponding to directly using the rank values as in MR.
Improving the distinguishability of RT . As shown in Figure 4(a), the lower bound of the RT increases as 𝛼ap-
proaches zero. Consequently, the score range shrinks, i.e., |E𝑖|−𝛼≤𝑓(𝑟,𝛼)≤ 1, where|E𝑖|is the number of candidate
entities for a test queryiin the filtered setting. A narrower score range reduces the differences between scores corre-
sponding to different ranks, limiting the ability of the metric to distinguish model performance. This issue becomes
more severe when |E𝑖|is small, as the pessimum (minimum) score increases further. To address this limitation, we

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 9
/uni00000013/uni0000004e /uni00000015/uni00000011/uni00000018/uni0000004e /uni00000018/uni0000004e /uni0000001a/uni00000011/uni00000018/uni0000004e
/uni00000035/uni00000044/uni00000051/uni0000004e/uni00000003/uni00000039/uni00000044/uni0000004f/uni00000058/uni00000048/uni00000013/uni00000011/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000014/uni00000011/uni00000013/uni00000013/uni00000037/uni00000055/uni00000044/uni00000051/uni00000056/uni00000049/uni00000052/uni00000055/uni00000050/uni00000048/uni00000047/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000020/uni00000013/uni00000011/uni00000014/uni00000015/uni00000018
/uni00000020/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000020/uni00000013/uni00000011/uni00000018
/uni00000020/uni00000014/uni00000011/uni00000013
/uni00000013/uni0000004e /uni00000015/uni00000011/uni00000018/uni0000004e /uni00000018/uni0000004e /uni0000001a/uni00000011/uni00000018/uni0000004e
/uni00000035/uni00000044/uni00000051/uni0000004e/uni00000003/uni00000039/uni00000044/uni0000004f/uni00000058/uni00000048/uni00000013/uni00000011/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000014/uni00000011/uni00000013/uni00000013/uni00000037/uni00000055/uni00000044/uni00000051/uni00000056/uni00000049/uni00000052/uni00000055/uni00000050/uni00000048/uni00000047/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000020/uni00000013/uni00000011/uni00000014/uni00000015/uni00000018
/uni00000020/uni00000013/uni00000011/uni00000015/uni00000018
/uni00000020/uni00000013/uni00000011/uni00000018
/uni00000020/uni00000014/uni00000011/uni00000013
(a) Original RT (b) Affine RT
Fig. 4. Rank transformers of PROBE: the affine RT rescales the range of scores to[0,1].
/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003(r|e)
/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003(e)
/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003(r|e)
/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003(e)
/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003(r|e)
/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003(e)
(a) FB15k237 (b) WN18RR (c) YAGO3-10
Fig. 5. Relationship between entity popularity 𝛿(𝑒) and entity-conditioned relation popularity 𝛿(𝑟|𝑒) in three real-world KGs. The
numerous intersecting lines indicate weak correlation between the two measures.
apply anaffinetransformation to the RT to rescale the scores from [|E𝑖|−𝛼,1]to[0,1](see Figure 4(b)), mapping the
worst prediction to the pessimum score of0. The rank transformer ofPROBEis re-defined as:
𝑓∗(𝑟𝑖,𝛼)=𝑓(𝑟𝑖,𝛼)−1
1−(|E𝑖|−𝛼)+1.(2)
This formulation improves the distinguishability ofPROBEby enlarging the effective score range while maintaining
the consistency of rank order, thereby enhancing its suitability as a rank-based evaluation metric for KGC models.
3.3 Rank Aggregation
The rank aggregator (RA) ofPROBEassigns a weight to each prediction based on the popularity of its corresponding
triple. A key question is“how to estimate the popularity of a target triple?"One possible way is to use the popularity
of a target entity in a triple. However, the popularity of a triple is often not fully captured by entity frequency alone.
For example,Barack Obamaappears in a number of triples with relations such asbornIn,presidentOf, oreducatedAt,
yet rarely with relations such asmemberOfSportsTeam. Thus, the triple (Obama,memberOfSportsTeam,𝑡) should be
regarded as a low-popularity query despite the high popularity of the entity. Similarly, relation frequency alone is
insufficient to characterize triple popularity. Some relations (e.g.,locatedIn,typeOf,hasGender) occur frequently overall,
but may still be rare for specific entities.
To empirically verify our claim, we analyze the relationship between entity popularity and entity-conditioned
relation popularity in real-world KGs. Figure 5 visualizes the entity and entity-conditioned relation popularity for triples
sampled from three real-world KGs, where the bottom horizontal axis represents entity popularity 𝛿(𝑒)=𝑑(𝑒)/(|T|∗ 2),
where𝑑(𝑒) is the number of triples where entity 𝑒appears and|T|is the total number of triples in a KG, and the
top axis represents entity-conditioned relation popularity 𝛿(𝑟|𝑒)=𝑑(𝑒⇒𝑟)/𝑑(𝑒) , where𝑑(𝑒⇒𝑟) is the number of
triples where 𝑒participate in the relation 𝑟. Each triple is represented as a line connecting its entity popularity and
entity-conditioned relation popularity values. If entity popularity and relation popularity are strongly correlated, most
lines would appear roughly parallel with few crossings. However, we observe a large number of intersecting lines across
all six datasets, indicating that entity popularity and relation popularity are not strongly correlated.

10 Moon et al.
From this observation, we characterize popularity bias at the triple level by separately measuring (i) entity popularity
and (ii) entity-conditioned relation popularity, and combining them to determine the final weight of a test query.
Formally, the weight of a test query𝑞=(ℎ,𝑟,?)(resp.𝑞=(?,𝑟,𝑡)) is defined as:
𝑤𝑡=𝑤𝑒·𝑤𝑟|𝑒,where𝑤 𝑒=1
(𝜖𝑒+𝛿(𝑒))𝛽, 𝑤𝑟|𝑒=1
(𝜖𝑟|𝑒+𝛿(𝑟|𝑒))𝛽,(3)
where𝑒is the target entity, 𝛿(𝑒) is the𝑒’s popularity, 𝛿(𝑟|𝑒) is the popularity of the relation 𝑟conditioned on 𝑒,𝛽is
a factor to control the level of popularity-bias robustness, and 𝜖is a small constant to prevent the division-by-zero
problem. Then, the weights for all test triples are normalized as follows:
¯w=w
||w|| 1,w=[𝑤 1,𝑤2,...,𝑤𝑛],(4)
where||w𝑡||1=Í𝑛
𝑖=1𝑤𝑖denotes the L1 norm. This normalization ensures thatPROBEcomputes a weighted average of
the transformed scores, preventing the magnitude of the weights from affecting the final evaluation score.
Therefore, given a KGC model 𝜃(·) and a setT𝑡𝑒𝑠𝑡of test triples,PROBEcomputes the final evaluation score of
the model by taking the weighted average of the transformed scoresc ∈R𝑛produced by the RT, where each score is
weighted by its corresponding popularity-aware weight ¯w∈R𝑛determined by the RA:
PROBE(𝜃,T 𝑡𝑒𝑠𝑡)=𝑎𝑔𝑔(c, ¯w)=1
𝑛𝑛∑︁
𝑖=1𝑤𝑖·𝑐𝑖.(5)
Figure 6(a) shows the weight of an entity (resp. relation) according to their popularity across different levels of
popularity-bias robustness 𝛽. When𝛽>0, RA assigns smaller weights to more popular entities (resp. relations) and
relatively larger weights to less popular ones, mitigating the dominance of high-popularity triples in evaluation. When
𝛽= 0, all entities (resp. relations) receive the same weight of 1, reducing the evaluation to the simple average of
transformed scores without considering popularity as in existing metrics such as MR, MRR, and Hits@K. Figures 6(b)-(d)
show the final triple weights according to entity popularity 𝛿(𝑒) and entity-conditioned relation popularity 𝛿(𝑟|𝑒) .
Since𝑤𝑡is defined as the product 𝑤𝑒·𝑤𝑟|𝑒, a triple is considered highly popular only when both popularity measures
are high. Thus, our weighting scheme captures popularity at the triple level.
3.4 Geometric Interpretation of PROBE
InPROBE, the transformed scores and their corresponding popularity-aware weights can be viewed as vectors in
an𝑛-dimensional space, i.e.,c =[𝑐 1,𝑐2,···,𝑐𝑛]and ¯w=[ ¯𝑤1,¯𝑤2,···, ¯𝑤𝑛], where𝑛=2∗|T𝑡𝑒𝑠𝑡|and each dimension
corresponds to a test query. Under this representation, the final evaluation score of a KGC model can be interpreted as
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000018 /uni00000014/uni00000011/uni00000013
/uni00000033/uni00000052/uni00000053/uni00000058/uni0000004f/uni00000044/uni00000055/uni0000004c/uni00000057/uni0000005c/uni00000014/uni00000013/uni00000013/uni00000014/uni00000013/uni00000014/uni00000014/uni00000013/uni00000015/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000056/uni00000020/uni00000013/uni00000011/uni0000001b
/uni00000020/uni00000013/uni00000011/uni00000017
/uni00000020/uni00000013/uni00000011/uni00000015
/uni00000020/uni00000013/uni00000011/uni00000014
/uni00000020/uni00000013/uni00000011/uni00000013
e/uni00000003/uni0000000b/uni0000000c
/uni0000000b/uni0000000c/uni00000003r|e
wt/uni00000003/uni0000000b/uni0000000c
e/uni00000003/uni0000000b/uni0000000c
/uni0000000b/uni0000000c/uni00000003r|e
wt/uni00000003/uni0000000b/uni0000000c
e/uni00000003/uni0000000b/uni0000000c
/uni0000000b/uni0000000c/uni00000003r|e
wt/uni00000003/uni0000000b/uni0000000c
/uni0000002b/uni0000004c/uni0000004a/uni0000004b
/uni0000002f/uni00000052/uni0000005a
(a) Weight functions (b)𝛽=0.0(c)𝛽=0.1(d)𝛽=0.8
Fig. 6. (a) Weight functions with varying 𝛽and (b) the final weights of triples depending on their entity and conditioned relation
popularity with three different𝛽=0.0,𝛽=0.1, and𝛽=0.8.

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 11
𝛼= 1.0 𝛽= 0.0 𝛼= 0.25 𝛽= 0.0 𝛼= 1.0 𝛽= 0.8 (a) (b) (c)𝐫𝐀
𝐫𝐁
𝐫𝐂
𝜹(𝐫|𝐞)𝜹(𝐞)?
?𝑟1
𝑟1
𝑟2 𝒯𝑡𝑒𝑠𝑡 𝐜𝐀
𝐜𝐁
𝐜𝐂300 2 1
10 3 2
5 4 4
0.1 0.5 0.5
0.25 0.8 0.8𝐰
Fig. 7. Geometric interpretation of PROBE: The parameter 𝛼controls the geometry of the transformed score space, while 𝛽determines
the evaluation direction through popularity-aware weighting. Therefore, the final evaluation score can be interpreted as the alignment
between the score vector and the popularity-aware evaluation direction.
the inner product between the two vectors:
PROBE(𝜃,T 𝑡𝑒𝑠𝑡)=𝑎𝑔𝑔(c, ¯w)=c· ¯w.(6)
From this geometric viewpoint, the RT ofPROBEmaps the rank vectorrinto a transformed space whose geometry
reflects the desired level of predictive sharpness. Larger 𝛼amplifies score differences among top-ranked predictions,
increasing the relative importance of correctly ranking entities near the best position (rank 1), resulting in sharper
separation between top and lower ranks. Conversely, smaller 𝛼compresses score differences across ranks, reducing the
penalty for moderately ranked predictions. Meanwhile, the weight vector ¯wdefines an evaluation direction that reflects
the desired level of popularity-bias robustness. Each element ¯𝑤𝑖determines the contribution of the corresponding test
query to the final score, indicating the relative importance of different regions in the evaluation space. Larger 𝛽tilts the
evaluation direction toward low-popularity triples by assigning relatively larger weights to them. Thus, 𝛼controls the
geometry of the transformed score space, while𝛽controls the evaluation direction.
Under this interpretation, the final score measures the alignment between the score vector and the popularity-aware
evaluation direction. High evaluation performance can be achieved when model predictions align well with both
evaluation perspectives: (P1) predictive sharpness and (P2) popularity-bias robustness. Consequently, different KGC
models may exhibit different degrees of alignment with this direction, leading to different evaluation outcomes.
Examples.Consider three KGC models evaluated on three test triples. They produce the rank vectors rA=[1,2,300],
rB=[2,3,10], and rC=[4,4,5]. Figure 7 shows that the final evaluation scores vary depending on the values of 𝛼
and𝛽, which can lead to different preferred models. When 𝛼=1and𝛽=0(a), this setting imposes a strong penalty
on incorrect predictions (i.e., high predictive sharpness) while not considering popularity bias. In this case, model A
achieves the best performance. When 𝛼=0.25and𝛽=0(b), the required level of predictive sharpness is reduced,
allowing moderately ranked predictions to contribute positively to the overall evaluation score. As a result, a model
that consistently predicts reasonable ranks, even if not top-ranked, may be preferred, and model B achieves the best
performance. When 𝛼=1and𝛽=0.8(c), the required level of popularity-bias robustness is increased, which assigns
greater weights to the predictions on low-popularity triples. In this case, a model that precisely predicts the ranks of
low-popularity triples may be preferred, and model C achieves the best performance.

12 Moon et al.
4 Theoretical Analysis of PROBE
In this section, we present six key properties for serving as a meaningful rank-based metric for KGC models and show
thatPROBEsatisfies all the properties (Section 4.1). We further analyze how reliably our proposed metric can estimate
the intrinsic predictive performance of KGC models under the open-world assumption (OWA) (Section 4.2).
4.1 Key Properties of a Rank-based Metric
Property 1(Fixed optimum). A rank-based metric satisfies thefixed optimumproperty if its transformation function
𝑓(·)assigns a constant optimal score to the best possible prediction, i.e., when the target entity is ranked first by a KGC
model: i.e.,𝑓(𝑟 𝑖)=𝑐𝑜𝑝𝑡, where𝑟𝑖=1.
Analysis underProperty 1.PROBEsatisfies this property since 𝑓∗(1)=1=𝑐𝑜𝑝𝑡according to Eq. (2). MR,
Hits@K, MRR, p-MRR, and Strat-MRR also satisfy this property, because each assigns its optimal value when 𝑟𝑖=1.
Specifically, MR achieves its minimum value, while Hits@K, MRR and the variants of MRR yield the maximum value of
1. Therefore, these metrics satisfy the fixed optimum property.
Property 2(Fixed pessimum). A rank-based metric satisfies thefixed pessimumproperty if its transformation function
𝑓(·)assigns a constant pessimum score to the worst possible prediction, i.e., when the target entity is ranked last by a
KGC model: i.e.,𝑓(𝑟 𝑖)=𝑐𝑝𝑒𝑠, where𝑟𝑖=|E𝑖|.
Analysis underProperty 2.PROBEsatisfies this property since 𝑓∗(|E𝑖|)= 0=𝑐𝑝𝑒𝑠according to Eq. (2). In
contrast, MRR, p-MRR, and Strat-MRR do not guarantee a fixed pessimum score, because their transformation function
𝑓(𝑟𝑖)=1/𝑟𝑖only approaches zeroasymptoticallyas 𝑟𝑖→∞ . Thus, these metrics do not provide a constant lower
bound for the worst prediction. In contrast,PROBEassigns a fixed pessimum value, improving interpretability and
discrimination among poorly ranked predictions.
Property 3(Anti monotonicity). A rank-based metric isanti-monotoneif its transformation function 𝑓(·)assigns a
higher score to a better rank, i.e., a lower rank value: i.e.,𝑟 𝑖>𝑟𝑗→𝑓(𝑟𝑖)<𝑓(𝑟𝑗).
Analysis underProperty 3.PROBEsatisfies this property since the RT assigns higher scores to better ranks, i.e.,
for any𝑟𝑖>𝑟𝑗,𝑓(𝑟𝑖,𝛼)<𝑓(𝑟 𝑗,𝛼)(see Figure 4). MRR, p-MRR, and Strat-MRR also satisfy this property because their
transformation function is monotonically decreasing with rank. In contrast, MR does not satisfy this property since
𝑓MR(𝑟𝑖)=𝑟𝑖assigns larger values to worse ranks. Hits@K also violates this property because it assigns identical scores
to different ranks within the same interval Although this property is not strictly required, it improves interpretability
by ensuring that better predictions consistently receive higher scores within a bounded range (e.g.,[0,1]).
Property 4(Candidate-size awareness). A rank-based metric iscandidate-size awareif its transformation function
𝑓(·) assigns different scores to the same rank depending on the size of the candidate entity set. Formally, for two
predictions with identical rank 𝑟𝑖=𝑟𝑗on two test triples with different candidate sizes |E𝑖|<|E𝑗|, it assigns different
scores: i.e.,|E 𝑖|<|E𝑗|→𝑓(𝑟𝑖)<𝑓(𝑟𝑗).
Analysis underProperty 4.PROBEsatisfies this property since the RT explicitly incorporates the candidate size
|E𝑖|into the transformation function (Eq. (2)). For the same rank 𝑟, predictions evaluated over larger candidate sets
receive higher scores, reflecting the greater difficulty of achieving the same rank among more candidates. Hits@K, MR,
MRR, p-MRR, and Strat-MRR depend only on rank 𝑟𝑖and ignore candidate size, assigning identical scores to predictions
with the same rank. In contrast, 𝑊𝑀𝑅 and AMRI account for candidate size, enabling more meaningful comparison
across predictions with different candidate spaces.

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 13
Table 2. Comparison of PROBE with existing metrics based on the six key properties.
Property MR MRR Hits@K p-MRR Strat-MRR 𝑊𝑀𝑅AMRI PROBE
Fixed optimum✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
Fixed pessimum- -✓- -✓ ✓ ✓
Anti monotonicity-✓-✓ ✓ ✓ ✓ ✓
Candidate-size awareness- - - - -✓ ✓ ✓
Sharpness awareness- -✓ ✓- - - ✓
Popularity awareness- - - - - - - ✓
Property 5(Sharpness awareness). A rank-based metric issharpness-awareif its transformation function 𝑓(·)can
adjust the sensitivity of evaluation to rank differences (i.e., varying levels of predictive sharpness). Formally, for a fixed
rank𝑟and two different sharpness levels𝛼≠𝛼′, the metric assigns different scores: i.e.,𝑓 𝛼(𝑟)≠𝑓𝛼′(𝑟).
Analysis underProperty 5.PROBEsatisfies this property since the RT incorporates a controllable parameter 𝛼
that adjusts the sensitivity of evaluation to rank differences. Larger 𝛼imposes larger penalties on worse ranks, leading
to sharper discrimination between high- and low-quality predictions. Among existing metrics, p-MRR reflects predictive
sharpness through its exponent parameter 𝑝. However, p-MRR does not satisfy the fixed pessimum property (Property
2), as it cannot guarantee a constant lower bound for worst-ranked predictions.
Property 6(Popularity awareness). A rank-based metric ispopularity-awareif its aggregation function assigns
different importance weights to prediction queries according to the popularity of their corresponding triples. Formally,
for a query triple 𝑞∈T𝑡𝑒𝑠𝑡, the aggregation weight is determined by its popularity: i.e., 𝑤𝑞∝𝛿(𝑞) , where𝛿(𝑞) denotes
the popularity of query𝑞.
Analysis underProperty 6.PROBEsatisfies this property since its RA assigns weights based on the popularity of
each query according to Eq. (3), assigning lower weights to high-popularity triples and higher weights to low-popularity
triples. This enables the evaluation metric to reflect the robustness of KGC models to popularity bias. On the other
hand, most existing metrics assign equal weights to all test triples and therefore ignore popularity differences across
triples. Although Strat-MRR introduces weighting across query groups, its weights depend on the grouping structure
of the test set rather than the popularity structure of the KG, and thus cannot capture popularity-bias robustness. In
contrast,PROBEexplicitly incorporates popularity information into the evaluation process, thereby enabling systematic
assessment of model robustness to popularity bias.
As summarized in Table 2,PROBEsatisfies all six key properties, whereas existing metrics satisfy only a subset of
them. This indicates thatPROBEprovides a more comprehensive and principled evaluation framework for KGC models,
while remaining a generalized metric that can degenerate to existing metrics under specific parameter settings.
4.2 Evaluation under the Open-World Assumption (OWA)
Since real-world KGs are inherently incomplete [ 42,46], the absence of a triple does not necessarily imply that it is
false. Thus, an observed KG can be viewed as apartial observationof an underlying complete KG containing all true
facts, i.e., the closed-world assumption (CWA). Therefore, an evaluation metric for KGC models should estimate the
model’s intrinsic predictive performance of the complete KG, even when evaluated is performed on an incomplete KG,
i.e., the open-world assumption (OWA). In particular, if an incomplete KG is obtained by removing a subset of true facts
from a complete KG, the intrinsic predictive performance of a model should remain unchanged. Thus, the evaluation
metric should provide consistent performance estimates across the two settings. Formally, for a model 𝜃, a set of test

14 Moon et al.
with filtering
fullfilteringQuebec Bellechasse_Municipality
Quebec
QuebecYamachicheYamachiche Maria-Chapdelaine Côte-Nord
Maria-ChapdelaineRoberval_Airport Ville-Marie Rosemère
Côte-Nord Roberval_Airport Ville-Marie Montréal-Nord LaSalle RosemèreMontréal-Nord LaSalle
Rank 146 / ℰ
RosemèreRank 9 / ℰ𝑖
Rank 2 / ℰ −ℰ𝑖+w/o filtering
Fig. 8. Example illustrating the challenge of evaluation under the OWA. For the query (?,isLocatedIn, Quebec), a model ranks the
target entityRosemèreas(1)146th without filtering,(2)9th after removing 137 known true entities with filtering, and(3)2nd after
additionally accounting for even unknown true entities (full filtering).
triplesT, a desirable evaluation metric𝑀(·)should satisfy:
ET∼D𝑂𝑊𝐴[𝑀(𝜃,T)]≈ET∼D𝐶𝑊𝐴[𝑀(𝜃,T)].(7)
However, accurate estimation of intrinsic predictive performance under the OWA remains challenging, because a
number of true facts may be missing from the observed KG, and the extent of such missing facts is unknown. While the
filtered setting [ 7] improves evaluation reliability by removing known true entities during rank computation, it remains
unclear how many additional true entities are still missing from the KG. Consequently, the ranks computed by a KGC
model may still underestimate the model’s performance since unobserved true entities may be ranked higher than the
target entity but are incorrectly treated as negatives [26].
Figure 8 illustrates this challenge with an example query (?,isLocatedIn, Quebec), where the target answer isRosemère.
We train ComplEx [ 46] on the YAGO3-10 dataset and make a prediction on the query.(1)Withoutfiltering, the target
entity is ranked 146th due to many higher-ranked true entities, resulting in a severely underestimated evaluation result.
(2)Withfiltering, 137 known true entities, ranked higher than the target entity, are removed, thereby improving the
rank to 9. However, we observe that 7 of the remaining 8 higher-ranked entities (e.g.,Yamachinche,Maria-Chapdelanine,
andLaSalle) are also true yet unknown. After accounting for these unknown true entities (i.e.,(3)fullfiltering), the
model ranks the target entity as 2nd among all true answers. This example highlights that the intrinsic predictive
performance of a KGC model should be evaluated based on the latent true rank (e.g., rank 2 in this example).
Motivated by the aforementioned challenge of KGC evaluation under the OWA, Yang et al . [55] theoretically analyzed
how reliably an evaluation metric reflects true model strength. Specifically, they examined how the expected value of a
metric changes with respect to the model’s true performance. Formally, the derivative of the expected metric value
with respect to true model strengthℓis defined as:
dˆE(𝑀)
dℓ=1
ℓ𝛾(|E+
𝑖|+1)E𝑟∼B(|E+
𝑖|+1,ℓ𝛾)𝑔(𝑟),(8)
where𝑀denotes the evaluation metric, ℓrepresents the true model strength (i.e., model performance under CWA), 𝛾is
the sparsity of the KG, |E+
𝑖|is the total number of true entities with respect to a given a test queryiin the complete KG,
and𝑟is the predicted rank following a binomial distribution B(|E+
𝑖|+1,ℓ𝛾). The function 𝑔(𝑟)=𝑟𝑓(𝑟) depends on the
rank transformation function 𝑓(·)defined by the metric. Thus, Eq. (8) provides a theoretical basis for analyzing how
reliably an evaluation metric reflects intrinsic predictive performance under the OWA.
Now, we analyze how reliablyPROBEreflects true model strength under the OWA. Specifically, we analyze how
the derivative of the expectedPROBEscore with respect to true model strength varies as a function of the sharpness
control factor 𝛼, by examining the sign of the partial derivative ofdˆE(PROBE)
dℓwith respect to 𝛼. Since the denominator
in Eq. (8) and the binomial term B(|E+
𝑖|+1,ℓ𝛾) do not depend on 𝛼, the behavior ofdˆE(PROBE)
dℓwith respect to 𝛼is
determined solely by𝑔(𝑟). Therefore, this analysis reduces to understanding the sign of𝜕
𝜕𝛼𝑔(𝑟).

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 15
Then, we apply Eq. (8) to the RT ofPROBEand rewrite the derivative with respect to𝛼as follows:
𝜕
𝜕𝛼dˆE(PROBE)
dℓ
∝𝜕
𝜕𝛼𝑔(𝑟)=𝜕
𝜕𝛼 𝑟𝑓∗
𝛼(𝑟)∝𝜕
𝜕𝛼𝑓∗
𝛼(𝑟) (∵𝑟is independent to𝛼)
=𝜕
𝜕𝛼(𝑓(𝑟,𝛼)−1
1−(|E𝑖|−𝛼)+1) (∵𝑓∗
𝛼(𝑟)=𝑓(𝑟,𝛼)−1
1−(|E𝑖|−𝛼)+1)(9)
∝(|E𝑖|
𝑟)𝛼((𝑟𝛼−1)ln|E 𝑖|+(1−|E 𝑖|𝛼)ln𝑟).(10)
Since(|E𝑖|
𝑟)𝛼>0, the sign of𝜕
𝜕𝛼
dˆE(PROBE)
dℓ
is determined by the sign of𝐹 𝛼(𝑟)=(𝑟𝛼−1)ln|E 𝑖|+(1−|E 𝑖|𝛼)ln𝑟.
By showing that 𝐹𝛼(𝑟)does not change the sign, i.e., ∀𝑟,𝐹𝛼(𝑟)≤ 0or𝐹𝛼(𝑟)≥ 0, we can ensure that the expected
PROBEscore changesmonotonicallywith respect to the true model strength ℓ. In other words, this indicates that 𝛼
directly controls how reliabilyPROBEreflects intrinsic model performance under the OWA.
From this observation, we present the following lemma:
Lemma 1.Let ℓand𝛼denote the true model strength under the CWA and the sharpness parameter, respectively. Then,
the derivative of the expectedPROBEscore with respect toℓis a monotonically decreasing function of𝛼∈R(𝛼≠0):
𝜕
𝜕𝛼dˆE(PROBE)
dℓ
≤0.
Proof for Lemma 1. This proof is equivalent to showing that 𝐹𝛼(𝑟)≤ 0for all𝑟<|E𝑖|. The derivative of 𝐹𝛼(𝑟)with
respect to𝛼is defined as:
𝜕
𝜕𝛼𝐹𝛼(𝑟)=𝜕
𝜕𝛼((𝑟𝛼−1)ln|E 𝑖|+(1−|E 𝑖|𝛼)ln𝑟)
=(𝑟𝛼−|E𝑖|𝛼)ln𝑟ln|E 𝑖|.(11)
We first identify the feasible domain of the rank 𝑟. Letm∼B(|E+
𝑖|,𝛾)denote the number of missing true entities
among all the true entities |E+
𝑖|. Since the number of filtered true entities excluding the target entity is |E+
𝑖|−m−1,
and the number of remaining candidate entities is |E𝑖|=|E|−(|E+
𝑖|−m−1). Thus, we have|E|−|E+
𝑖|+1≤|E𝑖|
(∵m≥0). Then, from the observation in [ 55] that for any queryi, the total number of true entities is typically less
than 10% of the total candidate entities, i.e.,|E+
𝑖|/|E|<10%, we obtain the following inequality:
1≤𝑟≤|E+
𝑖|+1<9|E+
𝑖|+1<|E|−|E+
𝑖|+1≤|E 𝑖|≤|E|.(12)
Based on this inequality, by applying𝑟<|E 𝑖|and1≤|E 𝑖|to Eq. 11, we obtain the following sign of the derivative:
𝜕
𝜕𝛼𝐹𝛼(𝑟) 
≤0if𝛼>0
=0if𝛼→0
≥0if𝛼<0.(13)
This shows that 𝐹𝛼(𝑟)remains non-positive over the domain of interest, i.e., ∀𝑟,𝐹𝛼(𝑟) ≤ 0. Recall that since
|E𝑖|
𝑟𝛼
>0, the sign of𝜕
𝜕𝛼
dˆE(PROBE)
dℓ
is determined solely by𝐹 𝛼(𝑟). Therefore, we conclude that
𝜕
𝜕𝛼dˆE(PROBE)
dℓ
≤0.(14)
□

16 Moon et al.
5 Experimental Validation
In this section, we aim to answer the following research questions:
•RQ1. To what extent does (P1) the predictive sharpness affect the evaluation score of KGC models?
•RQ2. To what extent does (P2) the popularity-bias robustness affect the evaluation score of KGC models?
•RQ3. DoesPROBEprovide a more comprehensive evaluation of KGC models, compared to existing metrics?
•RQ4. How reliable are the evaluation results produced byPROBEunder the open-world assumption (OWA)?
5.1 Experimental Setup
KG datasets and KGC models.In our experiments, we use six real-world knowledge graphs (KGs), which were also
used in [ 1,10,22,35,42,46,53]: FB15k237 [ 45], WN18RR [ 29], YAGO3-10 [ 27], Family [ 54], UMLS [ 32], and Kinship [ 32].
Table 3 shows the data statistics. We use six state-of-the-art KGC models: 4 embedding-based models (RotatE [ 42],
ComplEx [ 46], HousE [ 22], and TuckEr [ 1]) and 2 rule-based models (pLogicNet [ 36] and RNNLogic [ 35]). For all KGC
models, we use the official source codes provided by the authors. Also, we release the code and datasets used in this
paper at https://github.com/potato2734/probe-kgc-evaluation.
Table 3. Statistics of six real-world KGs used in our experiments.
|E| |R| |T|𝛿(𝑞) 𝑎𝑣𝑔𝛿(𝑞)𝑚𝑎𝑥
FB15k237 14,541 237 272,115 37.4 7,614
WN18RR 40,943 11 86,835 4.2 482
YAGO3-10 123,182 37 1,079,040 17.5 61,044
Family 3,007 12 23,483 15.6 94
UMLS 135 46 1,959 29.0 140
Kinship 104 25 3,206 61.7 80
Evaluation protocol.We use the evaluation protocol exactly same as that used in [ 35,43,63]. For each query, we
remove its corresponding known true entities from candidate entities during ranking (i.e., the filtered setting) and
assign the average rank of the tied group as the final rank (i.e., the tie-breaking protocol) for fair evaluation. During
training, a KGC model is optimized to minimize its specific loss function (e.g., margin-based ranking loss, logistic loss,
or cross-entropy loss). At each epoch, the model is evaluated on the validation set using a target evaluation metric (e.g.,
MRR, p-MRR, orPROBE), and the best-performing model is saved. Then, the selected model is evaluated on the test set
using the same evaluation metric. We report the averaged performance over three runs with different random seeds.
5.2 RQ1. Effects of predictive sharpness
In this experiment, we measure the accuracy of six KGC models with the varying sharpness control factor 𝛼, while
fixing the level of popularity-bias robustness ( 𝛽=0). As shown in 4, the results clearly demonstrate that the accuracy
of KGC models are highly sensitive to the level of predictive sharpness controlled by 𝛼. As𝛼varies, not only the
absolute evaluation scores but also the relative rankings of models substantially change across datasets, indicating
that different levels of predictive sharpness may favor different KGC models. This observation suggests that selecting
an inappropriate level of predictive sharpness for a target application (e.g, drug discovery or recommendation) may
lead to the selection of a suboptimal model. For instance, on FB15k-237, ComplEx is ranked only 4th when 𝛼=1.0

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 17
Table 4. The accuracy of KGC Models evaluated by PROBE with varying levels of predictive sharpness 𝛼. The gold, silver, and bronze
indicate the best, the second, and the third best results, respectively.
ModelsFB15k-237 WN18RR YAGO3-10
𝛼=1.0𝛼=0.5𝛼=0.0𝛼=-0.5𝛼=-1.0 𝛼=1.0𝛼=0.5𝛼=0.0𝛼=-0.5𝛼=-1.0 𝛼=1.0𝛼=0.5𝛼=0.0𝛼=-0.5𝛼=-1.0
RotatE 0.3140 0.4241 0.7124 0.9380 0.9844 0.4641 0.5213 0.6952 0.8549 0.9017 0.4743 0.5640 0.7971 0.9555 0.9813
ComplEx 0.3146 0.4283 0.7213 0.9425 0.9855 0.4754 0.5179 0.6594 0.8121 0.8700 0.4496 0.5467 0.7979 0.9588 0.9815
HousE 0.3073 0.4100 0.6910 0.9244 0.9778 0.4184 0.4913 0.6991 0.8823 0.9346 0.4538 0.5297 0.7500 0.9267 0.9647
TuckER 0.3530 0.4608 0.7379 0.9479 0.9884 0.4625 0.4999 0.6368 0.7961 0.8583 0.5562 0.6246 0.8144 0.9550 0.9806
pLogicNet 0.3241 0.4311 0.7149 0.9386 0.9847 0.4957 0.5672 0.7593 0.9002 0.9292 0.4535 0.5467 0.7927 0.9576 0.9834
RNNLogic 0.3197 0.4218 0.6969 0.9213 0.9723 0.4807 0.5225 0.6567 0.7929 0.8401 OOM OOM OOM OOM OOM
ModelsFamily UMLS Kinship
𝛼=1.0𝛼=0.5𝛼=0.0𝛼=-0.5𝛼=-1.0 𝛼=1.0𝛼=0.5𝛼=0.0𝛼=-0.5𝛼=-1.0 𝛼=1.0𝛼=0.5𝛼=0.0𝛼=-0.5𝛼=-1.0
RotatE 0.9487 0.9645 0.9834 0.9922 0.9939 0.7784 0.8368 0.9054 0.9552 0.9771 0.6336 0.7236 0.8372 0.9276 0.9719
ComplEx 0.9434 0.9619 0.9837 0.9942 0.9967 0.6046 0.6842 0.7936 0.8879 0.9386 0.5394 0.6462 0.7857 0.9008 0.9594
HousE 0.9757 0.9818 0.9893 0.9938 0.9953 0.8354 0.8775 0.9271 0.9637 0.9802 0.6808 0.7620 0.8621 0.9400 0.9770
TuckER 0.9784 0.9832 0.9895 0.9939 0.9959 0.8439 0.8849 0.9323 0.9669 0.9825 0.7067 0.7808 0.8716 0.9421 0.9759
pLogicNet 0.8119 0.8600 0.9355 0.9837 0.9939 0.5781 0.6608 0.7748 0.8764 0.9346 0.3771 0.4501 0.5628 0.6803 0.7641
RNNLogic 0.9737 0.9781 0.9839 0.9882 0.9901 0.7942 0.8455 0.9068 0.9526 0.9739 0.6417 0.7289 0.8394 0.9277 0.9712
/uni00000014/uni00000010/uni00000015 /uni00000016/uni00000010/uni00000018 /uni00000019/uni00000010/uni00000014/uni00000013 /uni00000014/uni00000014/uni00000010/uni00000015/uni00000013 /uni00000015/uni00000014/uni00000010/uni00000018/uni00000013 /uni00000018/uni00000014/uni00000010/uni00000014/uni00000013/uni00000013/uni00000014/uni0000004e/uni00000015/uni0000004e /uni00000006/uni00000003/uni00000053/uni00000055/uni00000048/uni00000047/uni0000004c/uni00000046/uni00000057/uni0000004c/uni00000052/uni00000051/uni0000000e/uni00000017/uni00000015/uni00000016
/uni0000000e/uni00000015/uni0000001c/uni00000016/uni0000000e/uni00000014/uni0000001a/uni00000018/uni0000000e/uni00000014/uni00000015/uni00000018 /uni0000000e/uni00000014/uni00000019/uni00000014/uni0000000e/uni0000001a/uni0000001c/uni0000002b/uni00000052/uni00000058/uni00000056/uni00000028 /uni00000035/uni00000031/uni00000031/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046
/uni00000014/uni00000010/uni00000015 /uni00000016/uni00000010/uni00000018 /uni00000019/uni00000010/uni00000014/uni00000013 /uni00000014/uni00000014/uni00000010/uni00000015/uni00000013 /uni00000015/uni00000014/uni00000010/uni00000018/uni00000013 /uni00000018/uni00000014/uni00000010/uni00000014/uni00000013/uni00000013/uni00000014/uni0000004e/uni00000015/uni0000004e/uni00000017/uni0000004e /uni00000006/uni00000003/uni00000053/uni00000055/uni00000048/uni00000047/uni0000004c/uni00000046/uni00000057/uni0000004c/uni00000052/uni00000051/uni0000000e/uni00000014/uni0000001c/uni00000013
/uni0000000e/uni00000016/uni00000016/uni0000001a
/uni0000000e/uni00000015/uni00000018/uni00000014 /uni0000000e/uni00000015/uni00000013/uni00000015 /uni0000000e/uni00000014/uni00000019/uni0000001a/uni0000000e/uni00000014/uni00000016/uni00000014/uni00000053/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni00000031/uni00000048/uni00000057 /uni0000002b/uni00000052/uni00000058/uni00000056/uni00000028
Fig. 9. Breakdown of predictions of two KGC models according to their ranks on the WN18RR and YAGO3-10 datasets.
(i.e., a high-sharpness setting similar to MRR), but its ranking progressively improves to 3rd at 𝛼=0.5and further to
2nd when𝛼=0.0. Similarly, on YAGO3-10, ComplEx eventually becomes the best-performing model when 𝛼=− 0.5,
despite being ranked only 5th under𝛼=1.0.
As𝛼decreases,PROBEimposes weaker penalties on non-top-ranked predictions, allowing moderately accurate
predictions to contribute more positively to the final score. Thus, models that consistently produce relatively high
ranks tend to improve their relative rankings under low predictive sharpness settings. In contrast, models that rely on
a relatively small number of highly accurate top-ranked predictions tend to lose their advantage as 𝛼decreases. For
example, on WN18RR, HousE improves from 6th at 𝛼=1.0to 1st at𝛼=− 1.0, whereas RNNLogic shows the opposite
trend. This suggests that HousE produces more stable rank predictions across broader rank ranges, while RNNLogic
relies more heavily on top-ranked predictions. A similar pattern is observed on YAGO3-10, where pLogicNet becomes
increasingly competitive under lower predictive sharpness settings despite weaker performance under high predictive
sharpness settings. Specifically, as shown in Figure 9, although RNNLogic achieves slightly more rank-1 and rank-2
predictions, HousE substantially outperforms RNNLogic across ranks 3–100. Consequently, RNNLogic is favored under
high predictive sharpness settings, while HousE becomes increasingly competitive as 𝛼decreases. A similar pattern is
also observed between pLogicNet and HousE on YAGO3-10. Therefore, different applications may require different
levels of predictive sharpness, and failing to reflect this perspective may lead to suboptimal model selection.
Finding (1). Existing evaluation metrics for KGC models (e.g., MRR) implicitly assume high predictive sharpness,
favoring KGC models producing top-ranked predictions while underestimating models with more stable ranking quality
across broader rank ranges. This suggests that a single fixed metric may be insufficient across diverse application
scenarios with different evaluation requirements.

18 Moon et al.
Table 5. The accuracy of KGC Models evaluated by PROBE with varying levels of popularity-bias robustness 𝛽. The gold, silver, and
bronze indicate the best, the second, and the third best results, respectively.
ModelsFB15k-237 WN18RR YAGO3-10
𝛽=0.0𝛽=0.2𝛽=0.4𝛽=0.6𝛽=0.8 𝛽=0.0𝛽=0.2𝛽=0.4𝛽=0.6𝛽=0.8 𝛽=0.0𝛽=0.2𝛽=0.4𝛽=0.6𝛽=0.8
RotatE 0.3140 0.1747 0.1043 0.0866 0.0820 0.4641 0.3326 0.2097 0.1462 0.1203 0.4743 0.3525 0.2207 0.1249 0.0791
ComplEx 0.3146 0.1801 0.1097 0.0897 0.0821 0.4754 0.3330 0.1998 0.1309 0.1029 0.4496 0.3476 0.2188 0.1241 0.0789
HousE 0.3073 0.1651 0.0948 0.0775 0.0731 0.4184 0.3017 0.1941 0.1383 0.1153 0.4538 0.3268 0.1828 0.0778 0.0297
TuckER 0.3530 0.2033 0.1286 0.1073 0.0980 0.4625 0.3125 0.1751 0.1058 0.0790 0.5654 0.4260 0.2675 0.1505 0.0947
pLogicNet 0.3241 0.1912 0.1244 0.1103 0.1097 0.4957 0.3478 0.2221 0.1601 0.1354 0.4535 0.3391 0.2167 0.1279 0.0861
RNNLogic 0.3197 0.1745 0.1058 0.0901 0.0862 0.4807 0.3243 0.1820 0.1099 0.0817 OOM OOM OOM OOM OOM
ModelsFamily UMLS Kinship
𝛽=0.0𝛽=0.2𝛽=0.4𝛽=0.6𝛽=0.8 𝛽=0.0𝛽=0.2𝛽=0.4𝛽=0.6𝛽=0.8 𝛽=0.0𝛽=0.2𝛽=0.4𝛽=0.6𝛽=0.8
RotatE 0.9487 0.9405 0.9259 0.9035 0.8745 0.7784 0.7537 0.7077 0.6310 0.5254 0.6336 0.6273 0.6199 0.6113 0.6016
ComplEx 0.9434 0.9419 0.9344 0.9185 0.8944 0.6046 0.6126 0.6092 0.5888 0.5512 0.5394 0.5354 0.5305 0.5247 0.5178
HousE 0.9757 0.9672 0.9513 0.9266 0.8950 0.8354 0.8128 0.7677 0.6903 0.5820 0.6808 0.6709 0.6594 0.6464 0.6320
TuckER 0.9784 0.9700 0.9548 0.9318 0.9026 0.8439 0.8111 0.7597 0.6878 0.6054 0.7067 0.6912 0.6733 0.6531 0.6308
pLogicNet 0.8119 0.7913 0.7662 0.7393 0.7131 0.5781 0.5590 0.5258 0.4774 0.4202 0.3771 0.3795 0.3817 0.3834 0.3847
RNNLogic 0.9737 0.9612 0.9401 0.9102 0.8748 0.7942 0.7608 0.7063 0.6251 0.5233 0.6417 0.6230 0.6016 0.5775 0.5510
5.3 RQ2. Effects of popularity-bias robustness
In this experiment, we evaluate six KGC models with varying levels of popularity-bias robustness 𝛽, while fixing the
predictive sharpness level ( 𝛼=1). As shown in Table 5, increasing 𝛽significantly changes the evaluation behavior across
all datasets. These results indicate that the perceived effectiveness of a KGC model can vary considerably depending
on how importantly the evaluation emphasizes robustness to popularity bias (e.g., discovering a new fact or allowing
common facts). Therefore, ignoring popularity bias during evaluation may result in selecting models that perform well
mainly on highly popular facts, while overlooking models that better generalize to less-observed new knowledge.
Specifically, as 𝛽increases,PROBEassigns larger weights, i.e., 𝛿(𝑒) and𝛿(𝑟|𝑒) , to low-popularity triples while
assigning smaller weights to high-popularity triples. Consequently, KGC models that rely heavily on predictions on
high-popularity triples tend to lose their advantage as 𝛽increases. In contrast, models that maintain high-quality
predictions on low-popularity triples become increasingly competitive. For example, on FB15k-237, TuckER is ranked
1st when𝛽=0, but its ranking decreases under more popularity-robust settings, whereas pLogicNet becomes the
best-performing model when 𝛽≥0.6. Similarly, on WN18RR, RNNLogic rapidly drops from 2nd to 5th as 𝛽increases,
while RotatE improves from 4th to 2nd.
To better understand why the relative rankings of KGC models change as 𝛽increases, we conduct an in-depth
analysis by comparing their prediction behaviors across different popularity levels. Specifically, we divide test queries
into four groups according to their triple popularity weight𝑤 𝑡and sample representative queries from each group.
Case study 1: pLogicNet vs TuckER on FB15k-237. Figure 10(a) shows the prediction results of pLogicNet and TuckER
across four popularity groups. For the most popular group (e.g., Top 0.34%), TuckER achieves a more accurate prediction
(rank 2) than pLogicNet (rank 5), indicating that TuckER performs particularly well on highly frequent facts. However, as
popularity decreases, pLogicNet consistently maintains relatively strong prediction quality, achieving ranks 6, 7, and 20
across the remaining groups. In contrast, TuckER’s prediction quality substantially deteriorates, producing much lower-
ranked predictions (ranks 15, 18, and 50). These results explain why pLogicNet becomes increasingly favored under higher
popularity-bias robustness settings. We further analyze the models from the perspective of entity-conditioned relation
popularity𝛿(𝑟|𝑒) . As shown in Figure 10(a) (below pie charts), TuckER performs extremely well on highly frequent
entity-relation patterns. For example, in the query (Brunei,organization, ?), the relationorganizationaccounts for 91% of

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 19
/uni00000053/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni00000031/uni00000048/uni00000057 /uni00000037/uni00000058/uni00000046/uni0000004e/uni00000028/uni00000035/uni00000018 /uni00000015/uni00000037/uni00000052/uni00000053/uni00000003/uni00000013/uni00000011/uni00000016/uni00000017/uni00000008
/uni00000053/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni00000031/uni00000048/uni00000057 /uni00000037/uni00000058/uni00000046/uni0000004e/uni00000028/uni00000035/uni00000019
/uni00000014/uni00000018/uni00000037/uni00000052/uni00000053/uni00000003/uni00000015/uni0000001c/uni00000011/uni00000018/uni00000018/uni00000008
/uni00000053/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni00000031/uni00000048/uni00000057 /uni00000037/uni00000058/uni00000046/uni0000004e/uni00000028/uni00000035/uni0000001a
/uni00000014/uni0000001b/uni00000037/uni00000052/uni00000053/uni00000003/uni00000019/uni00000013/uni00000011/uni0000001a/uni0000001b/uni00000008
/uni00000053/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni00000031/uni00000048/uni00000057 /uni00000037/uni00000058/uni00000046/uni0000004e/uni00000028/uni00000035/uni00000015/uni00000013
/uni00000018/uni00000013/uni00000037/uni00000052/uni00000053/uni00000003/uni0000001b/uni0000001b/uni00000011/uni00000019/uni0000001b/uni00000008
Commonwealth of Nations
(91%)
organizationBrunei
?Rank 2
Rank 1Independent film
(3%)
genrePulp Fiction
?Rank 8
Rank 38
Tenor saxophone
(11%)
roleCharlie Parker
?Rank 5
Rank 12Tak Fujimoto
(0%)
cinematographyThat Thing You Do!
?Rank 3
Rank 45
(a) Case study 1: pLogicNet vs TuckER on FB15k-237
/uni00000026/uni00000052/uni00000050/uni00000053/uni0000004f/uni00000028/uni0000005b /uni00000035/uni00000031/uni00000031/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni00000015/uni00000014/uni00000037/uni00000052/uni00000053/uni00000003/uni00000014/uni00000011/uni00000017/uni0000001b/uni00000008
/uni00000026/uni00000052/uni00000050/uni00000053/uni0000004f/uni00000028/uni0000005b /uni00000035/uni00000031/uni00000031/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni00000015
/uni0000001a/uni00000037/uni00000052/uni00000053/uni00000003/uni00000015/uni00000019/uni00000011/uni00000019/uni0000001a/uni00000008
/uni00000026/uni00000052/uni00000050/uni00000053/uni0000004f/uni00000028/uni0000005b /uni00000035/uni00000031/uni00000031/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni0000001c
/uni00000015/uni00000016/uni00000037/uni00000052/uni00000053/uni00000003/uni0000001b/uni00000013/uni00000011/uni0000001a/uni00000017/uni00000008
/uni00000026/uni00000052/uni00000050/uni00000053/uni0000004f/uni00000028/uni0000005b /uni00000035/uni00000031/uni00000031/uni0000002f/uni00000052/uni0000004a/uni0000004c/uni00000046/uni0000001c
/uni00000019/uni00000017/uni00000037/uni00000052/uni00000053/uni00000003/uni0000001c/uni00000015/uni00000011/uni00000018/uni0000001c/uni00000008
Occupation or discipline
(96%)
issue inMammal
?Rank 3
Rank 1Genetic function
(1%)
measurement ofClinical attribute
?Rank 1
Rank 5
virus
(14%)
interacts withBacterium
?Rank 2
Rank 4behavior
(0%) 
exhibitsvertebrate
?Rank 3
Rank 9 (b) Case study 2: ComplEx vs RNNLogic on UMLS
Fig. 10. Case studies on popularity-bias robustness across varying levels of entity popularity. While TuckER and RNNLogic achieve
highly accurate predictions on popular facts, their prediction quality substantially degrades on low-popularity queries. In contrast,
pLogicNet and ComplEx maintain more stable prediction quality across broader popularity ranges.
all relations associated with the entity, and TuckER successfully predicts the correct answer at rank 1. However, on less
frequent entity-relation patterns, such as (Tenor saxophone,𝑟𝑜𝑙𝑒, ?)and(Tak Fujimoto,cinematography, ?), pLogicNet
substantially outperforms TuckER. These observations indicate that TuckER relies more heavily on highly popular
entity-relation patterns, whereas pLogicNet demonstrates stronger robustness to sparse and less-observed facts.
Case study 2: ComplEx vs RNNLogic on UMLS. We observe a similar tendency on WN18RR when comparing
ComplEx and RNNLogic as shown in Figure 10(b). Although RNNLogic achieves rank 1 on the most popular query
group (Top 1.48%), its prediction quality rapidly degrades, producing substantially lower-ranked predictions (e.g., ranks
23 and 64) as query popularity decreases. In contrast, ComplEx maintains consistently strong prediction quality across
all popularity groups, achieving ranks 2, 2, 9, and 9 even for low-popularity queries. This explains why ComplEx
becomes increasingly competitive under higher popularity-bias robustness settings. From the perspective of entity-
conditioned relation popularity 𝛿(𝑟|𝑒) , RNNLogic achieves highly accurate predictions for highly frequent entity-relation
patterns, such as(𝑀𝑎𝑚𝑚𝑎𝑙,𝑖𝑠𝑠𝑢𝑒𝑖𝑛, ?)where the relation accounts for 96% of the entity’s associated relations. However,
on rare entity-relation patterns, such as (?,𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑚𝑒𝑛𝑡𝑜𝑓,𝐺𝑒𝑛𝑒𝑡𝑖𝑐𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛) and(?,𝑒𝑥ℎ𝑖𝑏𝑖𝑡𝑠,𝑏𝑒ℎ𝑎𝑣𝑖𝑜𝑟) , ComplEx
outperforms RNNLogic.
Finding (2). Existing metrics implicitly assume uniform importance across all triples, favoring KGC models that rely
heavily on highly popular entities and relations (i.e., exhibiting strong popularity bias), while underestimating models
robust to less-observed yet important facts. As a result, conventional evaluation settings may overestimate the practical
effectiveness of KGC models that fail to generalize beyond frequently observed knowledge.
5.4 RQ3. Comprehensiveness of PROBE
In this experiment, we investigate how comprehensivelyPROBEevaluates KGC models across varying levels of
predictive sharpness and popularity-bias robustness. Unlike existing metrics that evaluate KGC models from a single
fixed perspective,PROBEallows flexible evaluation under diverse combinations of 𝛼and𝛽, making researchers and
practitioners easy to evaluate their KGC models under diverse real-world requirements.
To analyze the relative performance of KGC models across different evaluation perspectives, we evaluate each model
under multiple(𝛼,𝛽) settings. Since the primary goal of KGC evaluation is to select the best-performing model under
a given perspective, in this experiment, we apply min-max normalization to the scores of the six comparing models
within each(𝛼,𝛽) coordinate. Figure 11 visualizes the performance of each KGC model on each dataset as a heatmap.

20 Moon et al.
RotatE ComplEx HousE TuckER pLogicNet RNNLogicFB15k-237
 WN18RR
 YAGO3-10
/uni00000032/uni00000011/uni00000032/uni00000011/uni00000030 Family
 UMLS
 Kinship
Fig. 11. Comprehensive evaluation of KGC models by PROBE under varying levels of predictive sharpness 𝛼and popularity-bias
robustness𝛽. The results demonstrate that model superiority substantially changes across different evaluation perspectives.
In each heatmap, the 𝑥-axis represents the predictive sharpness factor 𝛼, ranging from−1.0to1.0with a step size of
0.5, while the 𝑦-axis represents the popularity-bias robustness factor 𝛽, ranging from0 .0to0.8with a step size of0 .2.
O.O.M indicates an out-of-memory case. The results reveal that no single KGC model achieves the best performance
across all datasets and evaluation settings. For example, although TuckER performs best on most datasets under many
different evaluation settings, it is outperformed by HousE and pLogicNet on WN18RR and by ComplEx on YAGO3-10.
These results indicate that the preferred KGC model fundamentally depends on the desired evaluation perspective as we
claimed. In addition, model superiority cannot be fully characterized by a single fixed evaluation metric. In other words,
even KGC models that achieve similar performance under conventional metrics often exhibit different performance
patterns under varying levels of predictive sharpness and popularity-bias robustness.
Finding (3). Existing metrics with a single fixed evaluation perspective fail to fully characterize the performance of KGC
models whose relative performance can substantially vary depending on the desired evaluation perspective.PROBE
enables comprehensive evaluation of KGC models across diverse levels of predictive sharpness and popularity-bias
robustness, revealing performance characteristics that cannot be captured by existing fixed metrics.
5.5 RQ4. Evaluation Reliability under the Open-World Assumption
In this experiment, we evaluate how reliablyPROBEreflects the intrinsic performance of KGC models under the open-
world assumption (OWA). To properly assess intrinsic model performance under OWA, the corresponding performance
under the closed-world assumption (CWA) must be known. However, in real-world KGs, obtaining all true facts is
practically impossible. Therefore, we use the artificially constructed family-tree KG introduced in [ 55], where all facts are
fully observable under CWA. The KG consists of 6,004 entities, 23 relations, and 192,532 triples. From this closed-world
KG, OWA settings can be simulated by intentionally removing triples, enabling direct comparison between the intrinsic

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 21
Table 6. Different hyperparameters for two KGC models used in the experiment for RQ4.
# label KGC model Batch size # of negative samples Dimension gamma alpha learning rate L3 reg. weight
0 RotatE 1,024 128 1,000 24 1.0 5e-5 0.0
1 RotatE 512 128 500 24 1.0 5e-5 0.0
2 RotatE 128 128 200 24 1.0 5e-5 0.0
3 RotatE 128 512 500 18 1.0 5e-5 0.0
4 RotatE 128 128 1,000 12 0.5 5e-5 0.0
5 ComplEx 1,024 128 1,000 500 1.0 1e-3 1e-5
6 ComplEx 512 128 1,000 200 1.0 1e-4 1e-5
7 ComplEx 128 128 1,000 500 0.5 5e-3 1e-5
8 ComplEx 256 128 1,000 200 0.5 1e-3 1e-5
9 ComplEx 512 256 1,000 500 1.0 1e-3 1e-5
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000018 /uni00000014/uni00000011/uni00000013
/uni00000014/uni00000011/uni00000013/uni00000010/uni00000030/uni00000035/uni00000035/uni00000003/uni0000000b/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000014/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000016/uni00000014/uni00000011/uni00000013/uni00000010/uni00000030/uni00000035/uni00000035/uni00000003/uni0000000b/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c/uni00000014/uni00000011/uni00000013/uni00000010/uni00000030/uni00000035/uni00000035
/uni0000004c/uni00000047/uni00000048/uni00000044/uni0000004f
/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048
/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000003/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000018 /uni00000014/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000019/uni0000001a/uni00000010/uni00000030/uni00000035/uni00000035/uni00000003/uni0000000b/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni0000001a/uni00000010/uni00000030/uni00000035/uni00000035/uni00000003/uni0000000b/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c/uni00000013/uni00000011/uni00000019/uni0000001a/uni00000010/uni00000030/uni00000035/uni00000035
/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048
/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000003/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018 /uni00000013/uni00000011/uni00000018/uni00000013 /uni00000013/uni00000011/uni0000001a/uni00000018 /uni00000014/uni00000011/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000016/uni00000016/uni00000010/uni00000030/uni00000035/uni00000035/uni00000003/uni0000000b/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni00000016/uni00000016/uni00000010/uni00000030/uni00000035/uni00000035/uni00000003/uni0000000b/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c/uni00000013/uni00000011/uni00000016/uni00000016/uni00000010/uni00000030/uni00000035/uni00000035
/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048
/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000003/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018 /uni00000013/uni00000011/uni00000018/uni00000013 /uni00000013/uni00000011/uni0000001a/uni00000018 /uni00000014/uni00000011/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018/uni00000010/uni00000030/uni00000035/uni00000035/uni00000003/uni0000000b/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni00000015/uni00000018/uni00000010/uni00000030/uni00000035/uni00000035/uni00000003/uni0000000b/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c/uni00000013/uni00000011/uni00000015/uni00000018/uni00000010/uni00000030/uni00000035/uni00000035
/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048
/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000003/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000018 /uni00000014/uni00000011/uni00000013
/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000014/uni00000011/uni00000013/uni0000000c/uni00000003/uni0000000b/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000014/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000016/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000014/uni00000011/uni00000013/uni0000000c/uni00000003/uni0000000b/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c
/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000014/uni00000011/uni00000013/uni0000000c
/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048
/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000003/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000018 /uni00000014/uni00000011/uni00000013
/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000013/uni00000011/uni00000018/uni0000000c/uni00000003/uni0000000b/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000013/uni00000011/uni00000018/uni0000000c/uni00000003/uni0000000b/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c
/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000013/uni00000011/uni00000018/uni0000000c
/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048
/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000003/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018 /uni00000013/uni00000011/uni00000018/uni00000013 /uni00000013/uni00000011/uni0000001a/uni00000018 /uni00000014/uni00000011/uni00000013/uni00000013
/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000013/uni00000011/uni00000013/uni0000000c/uni00000003/uni0000000b/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c
/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001b/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000013/uni00000011/uni00000013/uni0000000c/uni00000003/uni0000000b/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c
/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000013/uni00000011/uni00000013/uni0000000c
/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048
/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000003/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013
/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000010/uni00000013/uni00000011/uni00000018/uni0000000c/uni00000003/uni0000000b/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c
/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001b/uni00000014/uni00000011/uni00000013/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000010/uni00000013/uni00000011/uni00000018/uni0000000c/uni00000003/uni0000000b/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048/uni00000003/uni00000057/uni00000048/uni00000056/uni00000057/uni00000003/uni00000056/uni00000048/uni00000057/uni0000000c
/uni00000033/uni00000035/uni00000032/uni00000025/uni00000028/uni0000000b/uni00000020/uni00000010/uni00000013/uni00000011/uni00000018/uni0000000c
/uni00000056/uni00000053/uni00000044/uni00000055/uni00000056/uni00000048
/uni00000049/uni00000058/uni0000004f/uni0000004f/uni00000003/uni00000003/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013
Fig. 12. Comparison of intrinsic model performance under CWA ( 𝑥-axis) and observed performance under OWA ( 𝑦-axis). A reliable
evaluation metric should preserve the relative ranking of models and assign similar scores to models with comparable intrinsic
performance under CWA, even when evaluated under OWA.
performance of KGC models under CWA and their observed evaluation results under OWA. Specifically, the KG dataset
consists of a training set and two test sets.(1) Full test set: all facts from the complete family-tree KG are used, enabling
evaluation under CWA.(2) Sparse test set: only 75% of the facts are used, simulating evaluation under OWA.
We select RotatE and ComplEx as the base KGC models and use five model variants with different hyperparameters
for each KGC model as described in Table 6. Each model is trained on the training set and evaluated on both the full test
set and sparse test set by using three evaluation metrics, including p-MRR, Hits@K, andPROBE. For p-MRR, we use four
values for the hyperparameter 𝑝, 1.0, 0.67, 0.33, and 0.25, following the settings used in the original work [ 55]. Figure 12
shows the results, where the 𝑥-axis represents the model performance on the full test set, i.e., the intrinsic performance
under the closed-world assumption (CWA), while the 𝑦-axis represents the observed performance on the sparse test

22 Moon et al.
set under the open-world assumption (OWA). For Hits@K and p-MRR, models with the same intrinsic performance
under CWA are often evaluated quite differently under OWA, and the relative rankings of models substantially change
between the two settings. In contrast,PROBEconsistently assigns similar evaluation scores to models with the same
CWA performance, while well-preserving their relative rankings under OWA.
Finding (4). Existing metrics often fail to reliably reflect the intrinsic performance of KGC models under the open-world
assumption, as their evaluation results can substantially vary depending on the incompleteness of observable facts. In
contrast,PROBEmore consistently preserves the relative performance and rankings of KGC models between CWA and
OWA settings, enabling more reliable evaluation under incomplete knowledge graphs.
6 Conclusion
In this paper, we point out that existing rank-based metrics for KGC evaluation overlook two important perspectives:
(P1) predictive sharpness and (P2) popularity-bias robustness. To address these limitations, we proposePROBE, a
generalized KGC evaluation framework consisting of a rank transformer (RT) and a rank aggregator (RA), which flexibly
control predictive sharpness and popularity-bias robustness, respectively. We theoretically analyzePROBEby defining
six key properties for reliable KGC evaluation and prove thatPROBEsatisfies all of them, unlike existing metrics.
In particular, we show thatPROBEmore consistently preserves the intrinsic performance and relative rankings of
KGC models under the open-world assumption. Extensive experiments with six KGC models on six real-world KGs
demonstrate that existing rank-based metrics implicitly assume fixed evaluation perspectives, which can over- or
under-estimate KGC models depending on their ranking behavior and robustness to popularity bias, whilePROBE
enables more comprehensive, flexible, and reliable evaluation across diverse evaluation perspectives.
In future work, we plan to further extendPROBEto a broader range of KGC models and real-world KGs. We also plan
to develop differentiable extensions ofPROBEthat can be directly incorporated into the training objective, enabling
KGC models to learn application-specific ranking behaviors and robustness requirements in an end-to-end manner.
References
[1] Ivana Balazevic, Carl Allen, and Timothy Hospedales. 2019. TuckER: Tensor Factorization for Knowledge Graph Completion. InProceedings of the
2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP). Association for Computational Linguistics, Hong Kong, China, 5185–5194. doi:10.18653/v1/D19-1522
[2]Dongmin Bang, Sangsoo Lim, Sangseon Lee, and Sun Kim. 2023. Biomedical knowledge graph learning for drug repurposing by extending
guilt-by-association to multiple layers.Nature Communications14, 1 (2023), 3570.
[3] Max Berrendorf, Evgeniy Faerman, Laurent Vermue, and Volker Tresp. 2020. On the ambiguity of rank-based evaluation of entity alignment or link
prediction methods.arXiv preprint arXiv:2002.06914(2020).
[4] Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring
human knowledge. InProceedings of the 2008 ACM SIGMOD International Conference on Management of Data. 1247–1250.
[5] Stephen Bonner, Ian P Barrett, Cheng Ye, Rowan Swiers, Ola Engkvist, Andreas Bender, Charles Tapley Hoyt, and William L Hamilton. 2022. A
review of biomedical datasets relating to drug discovery: a knowledge graph perspective.Briefings in Bioinformatics23, 6 (2022), bbac404.
[6] Stephen Bonner, Ufuk Kirik, Ola Engkvist, Jian Tang, and Ian P Barrett. 2022. Implications of topological imbalance for representation learning on
biomedical knowledge graphs.Briefings in bioinformatics23, 5 (2022), bbac279.
[7]Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling
multi-relational data. InAdvances in Neural Information Processing Systems, Vol. 26.
[8] Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam Hruschka, and Tom Mitchell. 2010. Toward an architecture for never-ending
language learning. InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 24. 1306–1313.
[9]Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang. 2025. Pathrag: Pruning graph-based
retrieval augmented generation with relational paths.arXiv preprint arXiv:2502.14902(2025).
[10] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. 2018. Convolutional 2D knowledge graph embeddings. InProceedings of
the AAAI Conference on Artificial Intelligence, Vol. 32.

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 23
[11] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness,
and Jonathan Larson. 2024. From local to global: A graph RAG approach to query-focused summarization.arXiv preprint arXiv:2404.16130(2024).
[12] Shangbin Feng, Zilong Chen, Wenqian Zhang, Qingyao Li, Qinghua Zheng, Xiaojun Chang, and Minnan Luo. 2021. Kgap: Knowledge graph
augmented political perspective detection in news media.arXiv preprint arXiv:2108.03861(2021).
[13] Qingyu Guo, Fuzhen Zhuang, Chuan Qin, Hengshu Zhu, Xing Xie, Hui Xiong, and Qing He. 2020. A survey on knowledge graph-based recommender
systems.IEEE Transactions on Knowledge and Data Engineering34, 8 (2020), 3549–3568.
[14] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. Lightrag: Simple and fast retrieval-augmented generation.arXiv preprint
arXiv:2410.05779(2024).
[15] Yanchao Hao, Yuanzhe Zhang, Kang Liu, Shizhu He, Zhanyi Liu, Hua Wu, and Jun Zhao. 2017. An end-to-end model for question answering over
knowledge base with cross-attention combining global knowledge. InProceedings of the 55th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 221–231.
[16] Charles Tapley Hoyt, Max Berrendorf, Mikhail Galkin, Volker Tresp, and Benjamin M Gyori. 2022. A unified framework for rank-based evaluation
metrics for link prediction in knowledge graphs.arXiv preprint arXiv:2203.07544(2022).
[17] Xiao Huang, Jingyuan Zhang, Dingcheng Li, and Ping Li. 2019. Knowledge graph embedding based question answering. InProceedings of the twelfth
ACM international conference on web search and data mining. 105–113.
[18] Myung-Hwan Jang, Yunyong Ko, Hyuck-Moo Gwon, Ikhyeon Jo, Yongjun Park, and Sang-Wook Kim. 2023. SAGE: A Storage-Based Approach for
Scalable and Efficient Sparse Generalized Matrix-Matrix Multiplication. InProceedings of the ACM International Conference on Information and
Knowledge Management (CIKM). 923–933.
[19] Xinke Jiang, Ruizhe Zhang, Yongxin Xu, Rihong Qiu, Yue Fang, Zhiyuan Wang, Jinyi Tang, Hongxin Ding, Xu Chu, Junfeng Zhao, et al .2025.
HyKGE: A hypothesis knowledge graph enhanced RAG framework for accurate and reliable medical LLMs responses. InProceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 11836–11856.
[20] Yunyong Ko, Seongeun Ryu, Soeun Han, Youngseung Jeon, Jaehoon Kim, Sohyun Park, Hanghang Han, Kyungsik Tong, and Sang-Wook Kim. 2023.
KHAN: Knowledge-Aware Hierarchical Attention Networks for Accurate Political Stance Prediction. InProceedings of the ACM Web Conference
(WWW)(Austin, TX, USA). Association for Computing Machinery (ACM), 1572–1583. doi:10.1145/3543507.3583300
[21] Yunyong Ko, Jae-Seo Yu, Hong-Kyun Bae, Yongjun Park, Dongwon Lee, and Sang-Wook Kim. 2021. MASCOT: A Quantization Framework for
Efficient Matrix Factorization in Recommender Systems. InProceedings of the IEEE International Conference on Data Mining (ICDM). IEEE, 290–299.
[22] Rui Li, Jianan Zhao, Chaozhuo Li, Di He, Yiqi Wang, Yuming Liu, Hao Sun, Senzhang Wang, Weiwei Deng, Yanming Shen, et al .2022. House:
Knowledge graph embedding with householder parameterization. InInternational Conference on Machine Learning. PMLR, 13209–13224.
[23] Xuan Lin, Zhe Quan, Zhi-Jie Wang, Tengfei Ma, and Xiangxiang Zeng. 2020. KGNN: Knowledge graph neural network for drug-drug interaction
prediction.. InIJCAI, Vol. 380. 2739–2745.
[24] Shuwen Liu, Bernardo Grau, Ian Horrocks, and Egor Kostylev. 2021. Indigo: GNN-based inductive knowledge graph completion using pair-wise
encoding. InAdvances in Neural Information Processing Systems, Vol. 34. 2034–2045.
[25] Haoran Luo, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin Kuang, Meina Song, Yifan Zhu, et al .2025.
Hypergraphrag: Retrieval-augmented generation via hypergraph-structured knowledge representation.arXiv preprint arXiv:2503.21322.
[26] Xin Lv, Yankai Lin, Yixin Cao, Lei Hou, Juanzi Li, Zhiyuan Liu, Peng Li, and Jie Zhou. 2022. Do pre-trained models benefit knowledge graph
completion? a reliable evaluation and a reasonable approach. InFindings of the association for computational linguistics: ACL 2022. 3570–3581.
[27] Farzaneh Mahdisoltani, Joanna Biega, and Fabian M Suchanek. 2013. YAGO3: A knowledge base from multilingual Wikipedias. InCIDR.
[28] Nicholas Matsumoto, Jay Moran, Hyunjun Choi, Miguel E Hernandez, Mythreye Venkatesan, Paul Wang, and Jason H Moore. 2024. KRAGEN: a
knowledge graph-enhanced RAG framework for biomedical problem solving using large language models.Bioinformatics40, 6 (2024), btae353.
[29] George A Miller. 1995. WordNet: a lexical database for English.Commun. ACM38, 11 (1995), 39–41.
[30] Aisha Mohamed, Shameem Parambath, Zoi Kaoudi, and Ashraf Aboulnaga. 2020. Popularity agnostic evaluation of knowledge graph embeddings.
InConference on Uncertainty in Artificial Intelligence. PMLR, 1059–1068.
[31] Sooho Moon and Yunyong Ko. 2026. How Sharp and Bias-Robust is a Model? Dual Evaluation Perspectives on Knowledge Graph Completion. In
Proceedings of the nineteenth ACM international conference on web search and data mining.
[32] Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel, et al .2011. A three-way model for collective learning on multi-relational data.. InIcml, Vol. 11.
3104482–3104584.
[33] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. 2024. Unifying large language models and knowledge graphs: A
roadmap.IEEE Transactions on Knowledge and Data Engineering36, 7 (2024), 3580–3599.
[34] Kunxun Qi, Jianfeng Du, and Hai Wan. 2024. Bi-directional Learning of Logical Rules with Type Constraints for Knowledge Graph Completion. In
Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 1899–1908.
[35] Meng Qu, Junkun Chen, Louis-Pascal Xhonneux, Yoshua Bengio, and Jian Tang. 2021. RNNLogic: Learning Logic Rules for Reasoning on Knowledge
Graphs. InInternational Conference on Learning Representations (ICLR). https://openreview.net/forum?id=tGZu6DlbreV
[36] Meng Qu and Jian Tang. 2019. Probabilistic logic neural networks for reasoning. InAdvances in Neural Information Processing Systems, Vol. 32.
[37] Tara Safavi and Danai Koutra. 2020. CoDEx: A Comprehensive Knowledge Graph Completion Benchmark. InProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP).

24 Moon et al.
[38] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max Welling. 2018. Modeling relational data with graph
convolutional networks. InThe Semantic Web: 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3–7, 2018, Proceedings 15.
Springer, 593–607.
[39] Harry Shomer, Wei Jin, Wentao Wang, and Jiliang Tang. 2023. Toward degree bias in embedding-based knowledge graph completion. InProceedings
of the ACM web conference 2023. 705–715.
[40] Fabian M Suchanek, Gjergji Kasneci, and Gerhard Weikum. 2007. YAGO: a core of semantic knowledge. InProceedings of the 16th International
Conference on World Wide Web. 697–706.
[41] Kai Sun, Huajie Jiang, Yongli Hu, and Baocai Yin. 2024. Incorporating multi-level sampling with adaptive aggregation for inductive knowledge
graph completion.ACM transactions on knowledge discovery from data18, 5 (2024), 1–16.
[42] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. 2019. RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. In
International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=HkgEQnRqYQ
[43] Zhiqing Sun, Shikhar Vashishth, Soumya Sanyal, Partha Talukdar, and Yiming Yang. 2020. A Re-evaluation of Knowledge Graph Completion
Methods. InProceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics,
Online, 5516–5522. doi:10.18653/v1/2020.acl-main.489
[44] Sudhanshu Tiwari, Iti Bansal, and Carlos R Rivero. 2021. Revisiting the evaluation protocol of knowledge graph completion methods for link
prediction. InProceedings of the Web Conference 2021. 809–820.
[45] Kristina Toutanova and Danqi Chen. 2015. Observed versus latent features for knowledge base and text inference. InProceedings of the 3rd Workshop
on Continuous Vector Space Models and Their Compositionality. 57–66.
[46] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. 2016. Complex embeddings for simple link prediction. In
Proceedings of International Conference on Machine Learning. PMLR, 2071–2080.
[47] Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha Talukdar. 2020. Composition-based Multi-Relational Graph Convolutional Networks.
InProceedings of International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=BylA_C4tPr
[48] Daniel Vella and Jean-Paul Ebejer. 2022. Few-shot learning for low-data drug discovery.Journal of Chemical Information and Modeling63, 1 (2022),
27–42.
[49] Hongwei Wang, Fuzheng Zhang, Xing Xie, and Minyi Guo. 2018. DKN: Deep knowledge-aware network for news recommendation. InProceedings
of the Web Conference (WWW). 1835–1844.
[50] Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, and Minyi Guo. 2019. Knowledge graph convolutional networks for recommender systems. InThe
world wide web conference. 3307–3313.
[51] Zihao Wang, Kwunping Lai, Piji Li, Lidong Bing, and Wai Lam. 2019. Tackling Long-Tailed Relations and Uncommon Entities in Knowledge Graph
Completion. InProceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, Hong Kong, China, 250–260. doi:10.18653/v1/D19-1024
[52] Wenhan Xiong, Mo Yu, Shiyu Chang, Xiaoxiao Guo, and William Yang Wang. 2018. One-Shot Relational Learning for Knowledge Graphs. In
Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Brussels,
Belgium, 1980–1990. doi:10.18653/v1/D18-1223
[53] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding Entities and Relations for Learning and Inference in
Knowledge Bases. InProceedings of International Conference on Learning Representations (ICLR), Poster Track. https://arxiv.org/abs/1412.6575
[54] Fan Yang, Zhilin Yang, and William W Cohen. 2017. Differentiable learning of logical rules for knowledge base reasoning. InAdvances in Neural
Information Processing Systems, Vol. 30.
[55] Haotong Yang, Zhouchen Lin, and Muhan Zhang. 2022. Rethinking knowledge graph evaluation under the open-world assumption. InAdvances in
Neural Information Processing Systems, Vol. 35. 8374–8385.
[56] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. 2021. QA-GNN: Reasoning with Language Models and
Knowledge Graphs for Question Answering. InProceedings of the 2021 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies. 535–546.
[57] Qing Ye, Chang-Yu Hsieh, Ziyi Yang, Yu Kang, Jiming Chen, Dongsheng Cao, Shibo He, and Tingjun Hou. 2021. A unified drug–target interaction
prediction framework based on knowledge graph and recommendation system.Nature communications12, 1 (2021), 6775.
[58] Wen-tau Yih, Xiaodong He, and Christopher Meek. 2014. Semantic parsing for single-relation question answering. InProceedings of the 52nd Annual
Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 643–648.
[59] Xiangxiang Zeng, Xinqi Tu, Yuansheng Liu, Xiangzheng Fu, and Yansen Su. 2022. Toward better drug discovery with knowledge graph.Current
Opinion in Structural Biology72 (2022), 114–126.
[60] Rui Zhang, Dimitar Hristovski, Dalton Schutte, Andrej Kastrin, Marcelo Fiszman, and Halil Kilicoglu. 2021. Drug repurposing for COVID-19 via
knowledge graph completion.Journal of Biomedical Informatics115 (2021), 103696.
[61] Wenqian Zhang, Shangbin Feng, Zilong Chen, Zhenyu Lei, Jundong Li, and Minnan Luo. 2022. KCD: Knowledge Walks and Textual Cues Enhanced
Political Perspective Detection in News Media. InProceedings of the 2022 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies. 4129–4140.
[62] Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander Smola, and Le Song. 2018. Variational reasoning for question answering with knowledge
graph. InProceedings of the AAAI conference on artificial intelligence, Vol. 32.

Generalized Rank-based Evaluation for Knowledge Graph Completion: Perspectives, Framework, and Analyses 25
[63] Yongqi Zhang and Quanming Yao. 2022. Knowledge graph reasoning with relational digraph. InProceedings of the ACM web conference 2022.
912–924.
[64] Sijin Zhou, Xinyi Dai, Haokun Chen, Weinan Zhang, Kan Ren, Ruiming Tang, Xiuqiang He, and Yong Yu. 2020. Interactive recommender system via
knowledge graph-enhanced reinforcement learning. InProceedings of the 43rd international ACM SIGIR conference on research and development in
information retrieval. 179–188.
[65] Chaoyu Zhu, Xiaoqiong Xia, Nan Li, Fan Zhong, Zhihao Yang, and Lei Liu. 2023. RDKG-115: Assisting drug repurposing and discovery for rare
diseases by trimodal knowledge graph embedding.Computers in Biology and Medicine164 (2023), 107262.
[66] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. 2021. Neural Bellman-Ford networks: A general graph neural network
framework for link prediction. InAdvances in Neural Information Processing Systems, Vol. 34. 29476–29490.