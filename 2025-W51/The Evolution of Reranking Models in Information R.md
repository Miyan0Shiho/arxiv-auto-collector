# The Evolution of Reranking Models in Information Retrieval: From Heuristic Methods to Large Language Models

**Authors**: Tejul Pandit, Sakshi Mahendru, Meet Raval, Dhvani Upadhyay

**Published**: 2025-12-18 06:29:37

**PDF URL**: [https://arxiv.org/pdf/2512.16236v1](https://arxiv.org/pdf/2512.16236v1)

## Abstract
Reranking is a critical stage in contemporary information retrieval (IR) systems, improving the relevance of the user-presented final results by honing initial candidate sets. This paper is a thorough guide to examine the changing reranker landscape and offer a clear view of the advancements made in reranking methods. We present a comprehensive survey of reranking models employed in IR, particularly within modern Retrieval Augmented Generation (RAG) pipelines, where retrieved documents notably influence output quality.
  We embark on a chronological journey through the historical trajectory of reranking techniques, starting with foundational approaches, before exploring the wide range of sophisticated neural network architectures such as cross-encoders, sequence-generation models like T5, and Graph Neural Networks (GNNs) utilized for structural information. Recognizing the computational cost of advancing neural rerankers, we analyze techniques for enhancing efficiency, notably knowledge distillation for creating competitive, lighter alternatives. Furthermore, we map the emerging territory of integrating Large Language Models (LLMs) in reranking, examining novel prompting strategies and fine-tuning tactics. This survey seeks to elucidate the fundamental ideas, relative effectiveness, computational features, and real-world trade-offs of various reranking strategies. The survey provides a structured synthesis of the diverse reranking paradigms, highlighting their underlying principles and comparative strengths and weaknesses.

## Full Text


<!-- PDF content starts -->

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
The Evolution of Reranking Models in
Information Retrieval: From Heuristic Methods
to Large Language Models
Tejul Pandit1[0009-0006-4376-1063], Sakshi Mahendru1∗, Meet Raval2∗, and
Dhvani Upadhyay3
1Palo Alto Networks, USA
2University of Southern California, USA
3Dhirubhai Ambani University, India
tejulpandit96@gmail.com
Abstract.Reranking is a critical stage in contemporary information re-
trieval (IR) systems, improving the relevance of the user-presented final
results by honing initial candidate sets. This paper is a thorough guide
to examine the changing reranker landscape and offer a clear view of the
advancements made in reranking methods. We present a comprehensive
survey of reranking models employed in IR, particularly within mod-
ern Retrieval Augmented Generation (RAG) pipelines, where retrieved
documents notably influence output quality.
We embark on a chronological journey through the historical trajectory
of reranking techniques, starting with foundational approaches, before
exploring the wide range of sophisticated neural network architectures
such as cross-encoders, sequence-generation models like T5, and Graph
NeuralNetworks(GNNs)utilizedforstructuralinformation.Recognizing
the computational cost of advancing neural rerankers, we analyze tech-
niques for enhancing efficiency, notably knowledge distillation for creat-
ing competitive, lighter alternatives. Furthermore, we map the emerging
territory of integrating Large Language Models (LLMs) in reranking, ex-
amining novel prompting strategies and fine-tuning tactics. This survey
seeks to elucidate the fundamental ideas, relative effectiveness, compu-
tational features, and real-world trade-offs of various reranking strate-
gies. The survey provides a structured synthesis of the diverse rerank-
ing paradigms, highlighting their underlying principles and comparative
strengths and weaknesses.
Keywords:Rerankers,InformationRetrieval(IR),RetrievalAugmented
Generation (RAG), Learning-to-rank, Neural rerankers, cross-encoders,
T5, Graph Neural Networks (GNN), knowledge distillation, Large Lan-
guage Models (LLM)
*Equal contributionarXiv:2512.16236v1  [cs.IR]  18 Dec 2025

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
2 T. Pandit et al.
1 Introduction
Information retrieval (IR) systems are essential in today’s digital world, with
their ability to power anything from online search and recommender systems to
sophisticated question-answering and knowledge management platforms. A crit-
ical component within these systems is the reranking stage. Reranking carefully
reorders these candidates to show the user the most pertinent results after an
initial, often rapid, retrieval phase that produces an extensive collection of can-
didate documents. This dramatically improves the overall quality and efficacy of
theIRsystem[1].Notably,thesignificantdevelopmentinLargeLanguageModels
(LLMs) has subsequently driven substantial improvements and interest in Re-
trieval Augmented Generation (RAG) pipelines, where the precision of retrieved
context, refined by reranking, is crucial for generating accurate and relevant
outputs[2]. Based on [3], we represent the typical placement of the reranking
component within a RAG pipeline in Figure 1.
Fig. 1.RAG approach highlighting a post-retrieval step of Reranking documents.
Reranking methods have evolved significantly, from heuristic scoring and
traditional learning-to-rank models to deep learning and, more recently, LLMs,
enabling richer semantic understanding and inter-document relationships.
This survey provides a comprehensive exploration of the evolving field of
reranking models in IR. We trace the development trajectory from classical
learning-to-rank methods in Section 3, followed by Deep learning rerankers in
Section 4. We also cover methods aimed at improving the efficiency of reranking
models, with a particular focus on knowledge distillation techniques in Section
5. Finally, we examine the cutting-edge integration of LLM-based rerankers in
Section 6. This study aims to provide scholars and practitioners with a valuable
guide for navigating this ever-evolving subject by compiling significant advance-
ments and empirical findings throughout the reranking timeline.

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
The Evolution of Reranking Models in IR 3
2 Background
Ranking approaches, which form the basis for many reranking techniques, can
broadlybecategorizedbasedontheirsupervisionstrategiesandhowtheyprocess
documents during training and scoring: pointwise, pairwise, and listwise.
Pointwise approachesconsider a single document at a time within the loss
function. They train a classifier or regressor to predict the relevance score of an
individual document for a given query. The final ranking is then produced by
sorting documents based on these predicted scores.
Pairwise approachesfocus on pairs of documents. Given a pair of docu-
ments, the model aims to determine the optimal relative ordering between them,
comparing this prediction against the ground truth.
Listwise approachesdirectlyconsiderandoptimizetherankingofanentire
list of documents. These methods aim to produce the optimal ordering for the
complete set of retrieved candidates.
3 Learning to Rank in Reranking Models
Learning drastically changed the reranking phase of contemporary information
retrieval (IR) systems to Rank (LTR). This section of the survey charts LTR’s
evolutionary trajectory in reranking, focusing on methodological shifts, core
problems, and model sophistication, from foundational probabilistic methods
to pairwise learning, listwise optimization, and introduction to deep learning
models.
Early LTR (late 1980s- early 1990s) applied statistical methods to estimate
relevance. Key efforts included [4] using polynomial regression for direct rele-
vance probability estimation, [5] with multi-stage logistic regression for compos-
ite clues, and [6] learning optimal linear combinations of expert scores, presaging
meta-search. These established data-driven rankings’ viability.
The late 1990s- early 2000s saw pairwise learning ascend, driven by machine
learning advances. Friedman’s [7] provided the Gradient Boosted Decision Tree
(GBDT)foundationcrucialforlaterLTR.Algorithmslike[8]appliedboostingto
learn from pairwise preferences, while [9] introduced influential neural networks
for pairwise ranking by minimizing cross-entropy between predicted and target
pair probabilities.
Limitations of generic pairwise losses spurred refinement towards IR-specific
metrics. [10] Adapted Ranking SVM by weighting misranked pairs based on
IR relevance. The pivotal [11] enabled direct optimization of non-smooth IR
metrics (like NDCG) by defining gradients (λ-values) based on metric changes
from swaps, guiding pairwise models to optimize listwise objectives. Other works
like[12]and[13]alsoexploredpairwiseregressionforpreferencesanddirectMAP
optimization.
This led to explicit listwise approaches, treating entire document lists as
training instances. [14] pioneered a listwise loss based on top-one probabilities.

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
4 T. Pandit et al.
Others like[15] and [16] (both2007) proposedquery-level losses. Boostingframe-
works like [17] were adapted to optimize IR metrics directly. Probabilistic list
models, such as [18] using the Plackett-Luce model, gained traction. The highly
successful [19] integrated MART (GBDTs) with LambdaRank’s gradients, be-
coming a state-of-the-art baseline. The scalability of such tree-based models was
later significantly enhanced by systems like [20].
From the mid-2010s, deep learning (DL) began to dominate LTR, offering
powerful representation learning and contextual modeling. Using techniques like
[21] triplet loss for metric learning, neural IR models began learning potent
query-document embeddings. Contextual DL models, using RNNs or Trans-
formers (e.g., [22], [23], [24]), excelled at capturing document interdependen-
cies within lists. Simultaneously, DL spurred innovations in preference modeling
(e.g., [25]) and end-to-end metric optimization using differentiable surrogates
or black-box methods (e.g., [26], [27]). We elaborate on more sophisticated DL-
based rerankers in Section 4.
Alongside these mainstream trends, diverse strategies continued to enrich
LTR. Evolutionary algorithms [28], [29] directly optimized IR metrics. Other
approaches included ordinal classification [30], incorporating preference magni-
tudes [31], and hybrid models combining objectives or data sources [32].
4 Deep Learning Techniques for Reranking
Deep learning models have significantly advanced the reranking task by captur-
ing intricate semantic relationships between queries and documents.
4.1 Transformer-based Rerankers (BERT and T5)
Transformerarchitecturesdominaterecentrerankingadvancements,notablyBERT-
style encoders and T5-style sequence-to-sequence models.
BERT-like Cross-EncodersCross-encoders based on BERT[33] and its vari-
ants jointly encode query-document pairs, enabling rich token-level interactions
via self-attention[34]. They output a relevance score.
Training pipelines are crucial. [35] uses sequential sparse retrieval (BM25),
pointwise BERT reranking, and pairwise BERT reranking. Integrated pipelines
like [36] combine retrieval, reranking, and generation using a BERT-based model
trained on MS MARCO[37].
Efficiency is a key challenge. [38] offers a scalable alternative using contex-
tualized late interaction. It encodes queries and documents independently with
BERT, preserving token embeddings. Similarity is calculated via Maximum Sim-
ilarity (MaxSim), summing the max dot products between query and document
tokens.Thisallowsdocumentprecomputation,drasticallyreducinglatencywhile
maintaining accuracy. Adaptation for multilingual scenarios uses methods like
Adapters or SFTMs[39] with mBERT, decoupling language and task adaptation
to mitigate the "curse of multilinguality"[40].

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
The Evolution of Reranking Models in IR 5
T5-based Sequence-to-Sequence RerankersSeq2seq models like T5[41]
offer an alternative generative approach. They frame relevance prediction as a
text-to-text task, leveraging pretraining benefits.
The seminal work [42] fine-tuned T5 to generate "true" or "false" given a
query-document pair, using the probability of generating "true" as the relevance
score. [43] proposes directly outputting numerical scores using the T5 encoder-
decoderorencoder-onlyarchitecture,supportingfine-tuningwithstandardrank-
ing losses. Listwise losses often yield better performance. [44] can improve gener-
alization by jointly training on ranking and auxiliary tasks like passage-to-query
generation. [45] adapts T5 for listwise reranking using the Fusion-in-Decoder
(FiD) architecture and an efficient m-ary tournament sort inference, mitigating
positional biases seen in other LLM-based listwise methods[46].
4.2 Advanced Interaction and Alternative Architectures
Research also explores models that explicitly model interactions between multi-
ple candidates or use different architectural foundations.
Methods modeling joint comparison between candidates, like [47], use a
Transformer encoder to jointly encode the query and multiple candidates, cap-
turing inter-candidate relationships efficiently within multi-stage pipelines (e.g.,
BE-CMC or BE-CMC-CE). [48] uses a specialized ListAttention mechanism al-
lowing passages to attend to each other, combined with Circle Loss[49] for im-
proved contrastive learning.
Beyond Transformers, other models are explored. [50] builds a document
graph from retrieved documents, using Graph Convolutional Networks[51] to
update representations based on relationships derived from Abstract Meaning
Representation (AMR). [52] evaluates the Mamba architecture[53] for document
reranking, showing competitive performance against Transformers but highlight-
ing current implementation speed challenges.
5 Efficiency Techniques: Knowledge Distillation
Modern search engines and RAG systems have demonstrated considerable effi-
cacy in question answering. However, their performance often encounters limita-
tions when confronted with complex, knowledge-intensive tasks that demand nu-
anced reasoning. Such tasks, including the identification of coding/configuration
errors, medical diagnosis, financial forecasting, and strategic marketing approach
formulation based on case studies, necessitate more than superficial relevance
matching, domain-specific knowledge.
5.1 Distillation Training Strategies for Reasoning-Aware Reranking
Based on our research across recent papers, the methodologies for distilling rea-
soning capabilities into smaller LMs for reranking can be broadly classified based

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
6 T. Pandit et al.
on the nature of the supervisory signal transferred from the teacher to the stu-
dent model.
Wedelineatetwoprimarycategories-LabelDistillation&Rationale-Enhanced
Distillation.
[54] claims a KARD-trained 250M T5 model surpassed a fine-tuned 3B T5
model and proposes a unique distillation process to transfer the learning from
LLMs to small LM using supervised fine tuning process. It follows a two-stage
training process (training the small language model (LM) with knowledge, and
training a helper "reranker") followed by an inference process that uses both
these trained components to answer new questions effectively.
In the first stage, we train the reranker via a teacher LLM that gener-
ates rationales for ground truth questions and documents, making the reranker
"rationale-aware" or "reasoning-aware," bridging the gap between optimal re-
trieval for rationale generation and practical inference-time retrieval. In the next
stage, small LLM is trained using explicit knowledge augmentation in addition
to query and rational, which is theoretically motivated to reduce the memoriza-
tion burden on small LMs. [55] presents novel frameworks like RADIO (RAtio-
nale DIstillation) to mitigate the "preference misalignment" gap in the RAG
pipeline. This gap can result in the generator receiving documents that, while
generally relevant, lack the precise information needed for high-quality, reasoned
responses. Recent research has begun to address this by focusing on how the rea-
soning process itself can inform document selection. It involves fine-tuning the
reranker component, denoted as R_θand typically instantiated as a powerful
cross-encoder architecture, utilizing a contrastive learning objective.
[56] and [57] addresses the lack of explainability and transparency in neural
document reranking models. It introduces a novel framework that separates and
distills both direct relevance and comparative reasoning from a LLM teacher
into LLaMA. Both the models were evaluated across a wider range of datasets.
The seminal work [58] introduced the concept of knowledge distillation, using
"soft targets" (probabilities from the teacher’s softmax, often with a tempera-
ture parameter T) and a modified cross-entropy loss (or matching logits, which it
showsisaspecialcase)totransferknowledge.Itdefinesthebasicdistillationpro-
cess and loss. However, this standard approach can suffer from over-calibration,
where the student too rigidly imitates an imperfect teacher. [59] proposes an im-
proved distillation process by integrating an alternative contrastive loss (LBKL)
with the standard Kullback-Leibler (KL) Divergence loss[60]. This LBKL al-
lows the student model to more conservatively and adaptively learn from the
teacher, permitting deviation when the teacher’s guidance might be suboptimal
for certain examples. The strategy aims to balance the benefits of teacher imita-
tion with an explicit contrastive objective, demonstrably improving re-ranking
performance by making the knowledge transfer more robust to teacher errors.
5.2 Comparison & Observations
[54]’s explicit knowledge augmentation strategy might be advantageous for tasks
requiring complex knowledge synthesis beyond document content. [56] is specif-

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
The Evolution of Reranking Models in IR 7
ically optimized for generating structured, interpretable rationales and perform-
ing precise reranking of a given set of documents within a single, efficient student
model. For pure document reranking with explainability, the authors found uni-
fied distillation particularly effective. [57] extends [56] distilling approach on
web-scraped Q&A to specifically refine the student’s ability to generate these
step-by-step explanations before relevance judgment.
The development of such models is crucial for democratizing access to sophis-
ticated reasoning capabilities, overcoming the deployment barriers of LLMs, and
paving the way for more transparent and trustworthy IR systems. While not a
distillation method itself, frameworks like Rank1[61], which focus on optimizing
test-time computation for reranking, stand to benefit significantly from highly
efficient yet effective distilled models that can perform sophisticated reasoning
within tight latency budgets.
6 LLM Based Rerankers
LLM-based rerankers are effective in listwise reranking settings where the goal
is to produce an ordered document list without explicit relevance scores for
individual documents [62, 63, 64, 65, 66]. [62] evaluates the performance of LLMs
in RAG using prompting strategies and introduces RankGPT. To tackle the
issue of context length limitations, it proposes a sliding window-based approach,
breaking the primary prompt into smaller prompts and then merging the results
of these smaller prompts. It then evaluates the model on various datasets to test
the efficacy of its proposed solutions. To counter concerns that LLMs might have
been trained on existing benchmark data, the paper introduces a new test set,
NovelEval, with information post the knowledge cutoff date of GPT-4[67]. Other
models have also researched sliding window techniques, for example, [68] shares
a SlideGar (Sliding-window-based Graph Adaptive Retrieval) algorithm to use
adaptive retrieval.
[69] investigates the challenges in the recommendation system due to the dis-
parity between the pre-training LLMs and the specific requirements of recom-
mendationtasks.ItintroducesaDirectMulti-preferenceoptimizationframework
and fine-tunes the model by incorporating multiple negative samples and opti-
mizing the model to increase the probability of correct predictions and decrease
incorrect predictions. [70] uses supervised fine tuning (SFT) on LLaMA[71] for
recommendation systems and introduces two models, RepLLaMa and RankL-
LaMa. RepLLaMa encodes queries and documents into a vector representation
and computes the relevance using the dot product, while RankLLaMa performs
pairwise ranking of the inputs. In contrast to SFT, [72] conducts training on a
large weakly supervised corpus and proposes a two-stage training framework to
adapt decoder-only LLMs to text ranking progressively using Continuous Pre-
Training(CPT)followedbySFT.ThebaseLLMundergoesCPTonalarge-scale,
diverse,weakly-supervisedtext-pairdatasetfromvarioussources.Thisstageuses
the Next Token Prediction (NTP) objective to initially orient the LLM towards
different aspects of relevance inherent in the text pairs. The model is fine-tuned

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
8 T. Pandit et al.
usinghigh-qualitysupervisedrankingdata.Itproposesanewoptimizationstrat-
egy using a ranking loss based on the generation probability of the entire query
given the document (P(q|d)), calculated via NTP, rather than relying on a last-
token representation.
LLMshavealsobeenproventobeeffectivetextrankerswhenusedinpairwise
ranking settings. [73] uses a pairwise approach where LLM is given a query and
two passages to determine the relevant passage. All the comparisons made create
a final score for the passage. [74] also shares a pairwise prompting approach
called PRP-Graph, which leverages LLMs’ output probabilities for target labels,
transforming these outputs into numerical scores for enhanced text ranking.
Researchers have also used model distillations to create open-source models.
[63],aLLaMa-basedmodelisthefirstfullyopen-sourceLLMcapableofperform-
ing high-quality listwise reranking in a zero-shot setting. It is trained in zero-shot
settingsusingRankGPT_3.5asateachermodelusing100KqueriesfromtheMS
Marco dataset. However, [63] still lags behind the state-of-the-art RankGPT4
in effectiveness. Thus, [64] introduces RankZephyr, another open source model
which bridges the gap and, in a few cases, outperforms RankGPT_4 with orders
of magnitude fewer parameters. [75] studies how to construct effective GPT-free
listwise rerankers based on open-source LLM models, where they used Code-
LLaMA-Instruct[76] for their experimental analysis.
In addition to these techniques, research has considered prompt engineering
for reranking. [77] shares a novel approach to reduce human effort and unlock
the potential of prompt optimization in the reranking approach by generating
refined prompts through feedback and preference optimization. [78] integrates
a soft prompt with a passage-specific embedding layer to form a new learnable
prompt module, then concatenates with the embeddings of the raw passage to
serve as new input for the LLMs.
7 Conclusion
This survey tracked the evolution of reranking algorithms from foundational
Learning to Rank (LTR) methods, which established principles for optimizing
result order, through the transformative impact of deep learning that enabled
sophisticated pattern capture and list-wise modeling. We then examined how
knowledge distillation became crucial for deploying complex models efficiently,
and finally, explored the emerging role of LLMs, which offer potential for deeper
semantic understanding. However, current rerankers, particularly those leverag-
ing deep learning and LLMs, grapple with challenges such as substantial com-
putational overhead, making real-time deployment in high-throughput systems
difficult. Furthermore, issues related to bias amplification and the need for ex-
tensive, high-quality labeled data remain significant hurdles in achieving robust
andfairrerankingperformanceacrossdiversequeriesandcontexts.Thisdevelop-
ment demonstrates an ongoing push for more efficient and contextually sensitive
result ordering, with future research probably concentrating on combining these

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
The Evolution of Reranking Models in IR 9
approaches and tackling the real-world difficulties of implementing sophisticated
rerankers.
References
[1]What Are Rerankers and How Do They Enhance Information Retrieval?
2024.url:https://zilliz.com/learn/what-are-rerankers-enhance-
information-retrieval.
[2]Rerankers and Two-Stage Retrieval.url:https://www.pinecone.io/
learn/series/rag/rerankers/.
[3] Yunfan Gao et al.Retrieval-Augmented Generation for Large Language
Models: A Survey. 2024. arXiv:2312 . 10997 [cs.CL].url:https : / /
arxiv.org/abs/2312.10997.
[4] NorbertFuhr.“Optimumpolynomialretrievalfunctionsbasedontheprob-
abilityrankingprinciple”.In:ACMTrans. Inf. Syst.7.3(July1989),pp.183–
204.issn: 1046-8188.doi:10.1145/65943.65944.url:https://doi.
org/10.1145/65943.65944.
[5] William S. Cooper, Fredric C. Gey, and Daniel P. Dabney. “Probabilistic
retrieval based on staged logistic regression”. In:Proceedings of the 15th
Annual International ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval. SIGIR ’92. Copenhagen, Denmark: As-
sociation for Computing Machinery, 1992, pp. 198–210.isbn: 0897915232.
doi:10.1145/133160.133199.url:https://doi.org/10.1145/133160.
133199.
[6] Brian T. Bartell, Garrison W. Cottrell, and Richard K. Belew. “Auto-
matic combination of multiple ranked retrieval systems”. In:Proceedings
of the 17th Annual International ACM SIGIR Conference on Research
and Development in Information Retrieval. SIGIR ’94. Dublin, Ireland:
Springer-Verlag, 1994, pp. 173–181.isbn: 038719889X.
[7] JeromeH.Friedman.“GreedyFunctionApproximation:AGradientBoost-
ingMachine”.In:The Annals of Statistics29.5(2001),pp.1189–1232.issn:
00905364, 21688966.url:http://www.jstor.org/stable/2699986(vis-
ited on 05/11/2025).
[8] Yoav Freund et al. “An efficient boosting algorithm for combining pref-
erences”. In:J. Mach. Learn. Res.4.null (Dec. 2003), pp. 933–969.issn:
1532-4435.
[9] Chris Burges et al. “Learning to rank using gradient descent”. In:Proceed-
ings of the 22nd International Conference on Machine Learning. ICML
’05. Bonn, Germany: Association for Computing Machinery, 2005, pp. 89–
96.isbn: 1595931805.doi:10 . 1145 / 1102351 . 1102363.url:https :
//doi.org/10.1145/1102351.1102363.
[10] Yunbo Cao et al. “Adapting ranking SVM to document retrieval”. In:
Proceedings of the 29th Annual International ACM SIGIR Conference on
Research and Development in Information Retrieval. SIGIR ’06. Seattle,
Washington, USA: Association for Computing Machinery, 2006, pp. 186–

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
10 T. Pandit et al.
193.isbn: 1595933697.doi:10.1145/1148170.1148205.url:https:
//doi.org/10.1145/1148170.1148205.
[11] Christopher Burges, Robert Ragno, and Quoc Le. “Learning to Rank with
Nonsmooth Cost Functions”. In: Jan. 2006, pp. 193–200.
[12] Tapio Pahikkala et al. “Learning to Rank with Pairwise Regularized Least-
Squares”. In:SIGIR 2007 Workshop on Learning to Rank for Information
Retrieval(Jan. 2007).
[13] Yisong Yue et al. “A support vector method for optimizing average preci-
sion”. In:Proceedings of the 30th Annual International ACM SIGIR Con-
ference on Research and Development in Information Retrieval. SIGIR
’07. Amsterdam, The Netherlands: Association for Computing Machinery,
2007, pp. 271–278.isbn: 9781595935977.doi:10.1145/1277741.1277790.
url:https://doi.org/10.1145/1277741.1277790.
[14] Zhe Cao et al.Learning to Rank: From Pairwise Approach to Listwise
Approach. Tech. rep. MSR-TR-2007-40. Apr. 2007, p. 9.url:https://
www.microsoft.com/en- us/research/publication/learning- to-
rank-from-pairwise-approach-to-listwise-approach/.
[15] Tao Qin et al. “Query-level loss functions for information retrieval”. In:In-
formation Processing & Management44.2 (2008). Evaluating Exploratory
Search Systems Digital Libraries in the Context of Users’ Broader Activ-
ities, pp. 838–855.issn: 0306-4573.doi:https://doi.org/10.1016/j.
ipm.2007.07.016.url:https://www.sciencedirect.com/science/
article/pii/S0306457307001276.
[16] Ming-feng Tsai et al.FRank: A Ranking Method with Fidelity Loss. Tech.
rep. MSR-TR-2006-155. Nov. 2006, p. 10.url:https://www.microsoft.
com/en-us/research/publication/frank-a-ranking-method-with-
fidelity-loss/.
[17] Jun Xu and Hang Li. “AdaRank: a boosting algorithm for information
retrieval”. In:Proceedings of the 30th Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval. SIGIR
’07. Amsterdam, The Netherlands: Association for Computing Machinery,
2007, pp. 391–398.isbn: 9781595935977.doi:10.1145/1277741.1277809.
url:https://doi.org/10.1145/1277741.1277809.
[18] Fen Xia et al. “Listwise approach to learning to rank: theory and algo-
rithm”. In:Proceedings of the 25th International Conference on Machine
Learning.ICML’08.Helsinki,Finland:AssociationforComputingMachin-
ery, 2008, pp. 1192–1199.isbn: 9781605582054.doi:10.1145/1390156.
1390306.url:https://doi.org/10.1145/1390156.1390306.
[19] Chris J. C. Burges et al.Ranking, Boosting, and Model Adaptation. Tech.
rep. MSR-TR-2008-109. Oct. 2008, p. 18.url:https://www.microsoft.
com/en- us/research/publication/ranking- boosting- and- model-
adaptation/.
[20] Tianqi Chen and Carlos Guestrin. “XGBoost: A Scalable Tree Boosting
System”. In:Proceedings of the 22nd ACM SIGKDD International Con-
ference on Knowledge Discovery and Data Mining. KDD ’16. ACM, Aug.

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
The Evolution of Reranking Models in IR 11
2016, pp. 785–794.doi:10.1145/2939672.2939785.url:http://dx.
doi.org/10.1145/2939672.2939785.
[21] Florian Schroff, Dmitry Kalenichenko, and James Philbin. “FaceNet: A
unified embedding for face recognition and clustering”. In:2015 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR). IEEE,
June 2015, pp. 815–823.doi:10.1109/cvpr.2015.7298682.url:http:
//dx.doi.org/10.1109/CVPR.2015.7298682.
[22] Qingyao Ai et al. “Learning a Deep Listwise Context Model for Ranking
Refinement”. In:The 41st International ACM SIGIR Conference on Re-
search & Development in Information Retrieval. SIGIR ’18. Ann Arbor,
MI, USA: Association for Computing Machinery, 2018, pp. 135–144.isbn:
9781450356572.doi:10.1145/3209978.3209985.url:https://doi.
org/10.1145/3209978.3209985.
[23] Liang Pang et al. “SetRank: Learning a Permutation-Invariant Ranking
Model for Information Retrieval”. In:Proceedings of the 43rd International
ACM SIGIR Conference on Research and Development in Information Re-
trieval. SIGIR ’20. Virtual Event, China: Association for Computing Ma-
chinery, 2020, pp. 499–508.isbn: 9781450380164.doi:10.1145/3397271.
3401104.url:https://doi.org/10.1145/3397271.3401104.
[24] Changhua Pei et al.Personalized Re-ranking for Recommendation. 2019.
arXiv:1904.06813 [cs.IR].url:https://arxiv.org/abs/1904.06813.
[25] Marius Köppel et al.Pairwise Learning to Rank by Neural Networks Re-
visited: Reconstruction, Theoretical Analysis and Practical Performance.
2019. arXiv:1909.02768 [cs.IR].url:https://arxiv.org/abs/1909.
02768.
[26] Fatih Cakir et al. “Deep Metric Learning to Rank”. In:2019 IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR). 2019,
pp. 1861–1870.doi:10.1109/CVPR.2019.00196.
[27] Robin Swezey et al.PiRank: Scalable Learning To Rank via Differentiable
Sorting. 2021. arXiv:2012.06731 [cs.LG].url:https://arxiv.org/
abs/2012.06731.
[28] Jen-Yuan Yeh et al. “Learning to rank for information retrieval using ge-
netic programming”. In:Proceedings of ACM SIGIR 2007 Workshop on
Learning to Rank for Information Retrieval(July 2012).doi:10.1109/
CyberneticsCom.2012.6381614.
[29] Osman Ali Sadek Ibrahim and Dario Landa-Silva. “ES-Rank: evolution
strategy learning to rank approach”. In:Proceedings of the Symposium on
Applied Computing. SAC ’17. Marrakech, Morocco: Association for Com-
puting Machinery, 2017, pp. 944–950.isbn: 9781450344869.doi:10.1145/
3019612.3019696.url:https://doi.org/10.1145/3019612.3019696.
[30] Ping Li, Chris J.C. Burges, and Qiang Wu.Learning to Rank Using Clas-
sification and Gradient Boosting. Tech. rep. MSR-TR-2007-74. Advances
in Neural Information Processing Systems 20. Jan. 2008.url:https :
//www.microsoft.com/en-us/research/publication/learning-to-
rank-using-classification-and-gradient-boosting/.

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
12 T. Pandit et al.
[31] Chenguang Zhu et al. “A general magnitude-preserving boosting algorithm
for search ranking”. In:Proceedings of the 18th ACM Conference on Infor-
mation and Knowledge Management. CIKM ’09. Hong Kong, China: Asso-
ciationforComputingMachinery,2009,pp.817–826.isbn:9781605585123.
doi:10.1145/1645953.1646057.url:https://doi.org/10.1145/
1645953.1646057.
[32] D. Sculley. “Combined regression and ranking”. In:Proceedings of the 16th
ACM SIGKDD International Conference on Knowledge Discovery and
Data Mining. KDD ’10. Washington, DC, USA: Association for Comput-
ing Machinery, 2010, pp. 979–988.isbn: 9781450300551.doi:10.1145/
1835804.1835928.url:https://doi.org/10.1145/1835804.1835928.
[33] Jacob Devlin et al.BERT: Pre-training of Deep Bidirectional Transform-
ers for Language Understanding. 2019. arXiv:1810.04805 [cs.CL].url:
https://arxiv.org/abs/1810.04805.
[34] Ashish Vaswani et al.Attention Is All You Need. 2023. arXiv:1706.03762
[cs.CL].url:https://arxiv.org/abs/1706.03762.
[35] Rodrigo Nogueira et al.Multi-Stage Document Ranking with BERT. 2019.
arXiv:1910.14424 [cs.IR].url:https://arxiv.org/abs/1910.14424.
[36] Michael Glass et al.Re2G: Retrieve, Rerank, Generate. 2022. arXiv:2207.
06300 [cs.CL].url:https://arxiv.org/abs/2207.06300.
[37] Payal Bajaj et al.MS MARCO: A Human Generated MAchine Reading
COmprehension Dataset. 2018. arXiv:1611.09268 [cs.CL].url:https:
//arxiv.org/abs/1611.09268.
[38] Omar Khattab and Matei Zaharia.ColBERT: Efficient and Effective Pas-
sage Search via Contextualized Late Interaction over BERT. 2020. arXiv:
2004.12832 [cs.IR].url:https://arxiv.org/abs/2004.12832.
[39] Robert Litschko, Ivan Vulić, and Goran Glavaš.Parameter-Efficient Neu-
ral Reranking for Cross-Lingual and Multilingual Retrieval. 2022. arXiv:
2204.02292 [cs.CL].url:https://arxiv.org/abs/2204.02292.
[40] Tyler A. Chang et al.When Is Multilinguality a Curse? Language Model-
ing for 250 High- and Low-Resource Languages. 2023. arXiv:2311.09205
[cs.CL].url:https://arxiv.org/abs/2311.09205.
[41] Colin Raffel et al.Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer.2023.arXiv:1910.10683 [cs.LG].url:https:
//arxiv.org/abs/1910.10683.
[42] Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin.Document Ranking
with a Pretrained Sequence-to-Sequence Model. 2020. arXiv:2003.06713
[cs.IR].url:https://arxiv.org/abs/2003.06713.
[43] Honglei Zhuang et al.RankT5: Fine-Tuning T5 for Text Ranking with
Ranking Losses. 2022. arXiv:2210.10634 [cs.IR].url:https://arxiv.
org/abs/2210.10634.
[44] Jia-Huei Ju, Jheng-Hong Yang, and Chuan-Ju Wang.Text-to-Text Multi-
view Learning for Passage Re-ranking. 2022. arXiv:2104.14133 [cs.IR].
url:https://arxiv.org/abs/2104.14133.

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
The Evolution of Reranking Models in IR 13
[45] Soyoung Yoon et al.ListT5: Listwise Reranking with Fusion-in-Decoder
Improves Zero-shot Retrieval. 2024. arXiv:2402 . 15838 [cs.IR].url:
https://arxiv.org/abs/2402.15838.
[46] Nelson F. Liu et al.Lost in the Middle: How Language Models Use Long
Contexts. 2023. arXiv:2307.03172 [cs.CL].url:https://arxiv.org/
abs/2307.03172.
[47] JonghyunSongetal.Comparing Neighbors Together Makes it Easy: Jointly
Comparing Multiple Candidates for Efficient and Effective Retrieval. 2024.
arXiv:2405.12801 [cs.CL].url:https://arxiv.org/abs/2405.12801.
[48] Junlong Liu et al.ListConRanker: A Contrastive Text Reranker with List-
wise Encoding. 2025. arXiv:2501.07111 [cs.CL].url:https://arxiv.
org/abs/2501.07111.
[49] Yifan Sun et al.Circle Loss: A Unified Perspective of Pair Similarity Op-
timization. 2020. arXiv:2002.10857 [cs.CV].url:https://arxiv.org/
abs/2002.10857.
[50] Jialin Dong et al.Don’t Forget to Connect! Improving RAG with Graph-
based Reranking.2024.arXiv:2405.18414 [cs.CL].url:https://arxiv.
org/abs/2405.18414.
[51] Si Zhang et al. “Graph convolutional networks: a comprehensive review”.
In:Computational Social Networks6 (Nov. 2019).doi:10.1186/s40649-
019-0069-y.
[52] ZhichaoXu.RankMamba: BenchmarkingMamba’s Document RankingPer-
formance in the Era of Transformers. 2024. arXiv:2403.18276 [cs.IR].
url:https://arxiv.org/abs/2403.18276.
[53] Albert Gu and Tri Dao.Mamba: Linear-Time Sequence Modeling with
Selective State Spaces. 2024. arXiv:2312.00752 [cs.LG].url:https:
//arxiv.org/abs/2312.00752.
[54] Minki Kang et al.Knowledge-Augmented Reasoning Distillation for Small
Language Models in Knowledge-Intensive Tasks. 2023. arXiv:2305.18395
[cs.CL].url:https://arxiv.org/abs/2305.18395.
[55] Pengyue Jia et al.Bridging Relevance and Reasoning: Rationale Distilla-
tion in Retrieval-Augmented Generation.2024.arXiv:2412.08519 [cs.CL].
url:https://arxiv.org/abs/2412.08519.
[56] Yuelyu Ji et al.ReasoningRank: Teaching Student Models to Rank through
Reasoning-BasedKnowledgeDistillation.2025.arXiv:2410.05168 [cs.CL].
url:https://arxiv.org/abs/2410.05168.
[57] Chris Samarinas and Hamed Zamani.Distillation and Refinement of Rea-
soning in Small Language Models for Document Re-ranking. 2025. arXiv:
2504.03947 [cs.IR].url:https://arxiv.org/abs/2504.03947.
[58] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.Distilling the Knowledge
in a Neural Network. 2015. arXiv:1503.02531 [stat.ML].url:https:
//arxiv.org/abs/1503.02531.
[59] Yingrui Yang et al. “Balanced Knowledge Distillation with Contrastive
Learning for Document Re-ranking”. In:Proceedings of the 2023 ACM SI-
GIR International Conference on Theory of Information Retrieval. ICTIR

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
14 T. Pandit et al.
’23. Taipei, Taiwan: Association for Computing Machinery, 2023, pp. 247–
255.isbn: 9798400700736.doi:10.1145/3578337.3605120.url:https:
//doi.org/10.1145/3578337.3605120.
[60] I. Csiszar. “I-Divergence Geometry of Probability Distributions and Mini-
mization Problems”. In:The Annals of Probability3.1 (1975), pp. 146–158.
doi:10.1214/aop/1176996454.url:https://doi.org/10.1214/aop/
1176996454.
[61] Orion Weller et al.Rank1: Test-Time Compute for Reranking in Informa-
tion Retrieval. 2025. arXiv:2502.18418 [cs.IR].url:https://arxiv.
org/abs/2502.18418.
[62] Weiwei Sun et al.Is ChatGPT Good at Search? Investigating Large Lan-
guage Models as Re-Ranking Agents. 2024. arXiv:2304.09542 [cs.CL].
url:https://arxiv.org/abs/2304.09542.
[63] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin.RankVicuna:
Zero-Shot Listwise Document Reranking with Open-Source Large Language
Models. 2023. arXiv:2309.15088 [cs.IR].url:https://arxiv.org/
abs/2309.15088.
[64] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin.RankZephyr:
Effectiveand RobustZero-Shot Listwise Rerankingis aBreeze!2023.arXiv:
2312.02724 [cs.IR].url:https://arxiv.org/abs/2312.02724.
[65] Xueguang Ma et al.Zero-Shot Listwise Document Reranking with a Large
Language Model.2023.arXiv:2305.02156 [cs.IR].url:https://arxiv.
org/abs/2305.02156.
[66] Shuoqi Sun et al.An Investigation of Prompt Variations for Zero-shot
LLM-based Rankers. 2025. arXiv:2406.14117 [cs.IR].url:https://
arxiv.org/abs/2406.14117.
[67] OpenAITeam.GPT-4Technical Report.2024.arXiv:2303.08774 [cs.CL].
url:https://arxiv.org/abs/2303.08774.
[68] MandeepRathee,SeanMacAvaney,andAvishekAnand.Guiding Retrieval
using LLM-based Listwise Rankers. 2025. arXiv:2501 . 09186 [cs.IR].
url:https://arxiv.org/abs/2501.09186.
[69] Zhuoxi Bai et al.Finetuning Large Language Model for Personalized Rank-
ing. 2024. arXiv:2405.16127 [cs.IR].url:https://arxiv.org/abs/
2405.16127.
[70] Xueguang Ma et al.Fine-Tuning LLaMA for Multi-Stage Text Retrieval.
2023. arXiv:2310.08319 [cs.IR].url:https://arxiv.org/abs/2310.
08319.
[71] Hugo Touvron et al.LLaMA: Open and Efficient Foundation Language
Models. 2023. arXiv:2302.13971 [cs.CL].url:https://arxiv.org/
abs/2302.13971.
[72] Longhui Zhang et al.A Two-Stage Adaptation of Large Language Models
for Text Ranking. 2024. arXiv:2311 . 16720 [cs.IR].url:https : / /
arxiv.org/abs/2311.16720.

This version of the article has been accepted for publication in Springer CCIS, volume 2775, after peer review (when ap-
plicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The
Version of Record is available online at springnature. Use of this Accepted Version is subject to the publisher’s Ac-
cepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscriptterms.
The Evolution of Reranking Models in IR 15
[73] Chao-Wei Huang and Yun-Nung Chen.InstUPR : Instruction-based Un-
supervised Passage Reranking with Large Language Models. 2024. arXiv:
2403.16435 [cs.CL].url:https://arxiv.org/abs/2403.16435.
[74] Jian Luo et al. “PRP-Graph: Pairwise Ranking Prompting to LLMs with
Graph Aggregation for Effective Text Re-ranking”. In:Proceedings of the
62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). Ed. by Lun-Wei Ku, Andre Martins, and Vivek
Srikumar. Bangkok, Thailand: Association for Computational Linguistics,
Aug. 2024, pp. 5766–5776.doi:10.18653/v1/2024.acl-long.313.url:
https://aclanthology.org/2024.acl-long.313/.
[75] Xinyu Zhang et al.Rank-without-GPT: Building GPT-Independent List-
wiseRerankers onOpen-SourceLargeLanguage Models.2023.arXiv:2312.
02969 [cs.CL].url:https://arxiv.org/abs/2312.02969.
[76] Baptiste Rozière et al.Code Llama: Open Foundation Models for Code.
2024. arXiv:2308.12950 [cs.CL].url:https://arxiv.org/abs/2308.
12950.
[77] Can Jin et al.APEER: Automatic Prompt Engineering Enhances Large
LanguageModel Reranking.2024.arXiv:2406.14449 [cs.AI].url:https:
//arxiv.org/abs/2406.14449.
[78] Xuyang Wu et al.Passage-specific Prompt Tuning for Passage Reranking
in Question Answering with Large Language Models. 2024. arXiv:2405.
20654 [cs.CL].url:https://arxiv.org/abs/2405.20654.