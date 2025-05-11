# LLM-Independent Adaptive RAG: Let the Question Speak for Itself

**Authors**: Maria Marina, Nikolay Ivanov, Sergey Pletenev, Mikhail Salnikov, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Alexander Panchenko, Viktor Moskvoretskii

**Published**: 2025-05-07 08:58:52

**PDF URL**: [http://arxiv.org/pdf/2505.04253v1](http://arxiv.org/pdf/2505.04253v1)

## Abstract
Large Language Models~(LLMs) are prone to hallucinations, and
Retrieval-Augmented Generation (RAG) helps mitigate this, but at a high
computational cost while risking misinformation. Adaptive retrieval aims to
retrieve only when necessary, but existing approaches rely on LLM-based
uncertainty estimation, which remain inefficient and impractical. In this
study, we introduce lightweight LLM-independent adaptive retrieval methods
based on external information. We investigated 27 features, organized into 7
groups, and their hybrid combinations. We evaluated these methods on 6 QA
datasets, assessing the QA performance and efficiency. The results show that
our approach matches the performance of complex LLM-based methods while
achieving significant efficiency gains, demonstrating the potential of external
information for adaptive retrieval.

## Full Text


<!-- PDF content starts -->

LLM-Independent Adaptive RAG:
Let the Question Speak for Itself
Maria Marina2,1, Nikolay Ivanov1, Sergey Pletenev2,1, Mikhail Salnikov2,1,
Daria Galimzianova4,Nikita Krayko4,Vasily Konovalov2,5,
Alexander Panchenko1,2, Viktor Moskvoretskii1,3
1Skoltech,2AIRI,3HSE University,4MTS AI,5MIPT
{Maria.Marina, A.Panchenko, Mikhail.Salnikov}@skol.tech
Abstract
Large Language Models (LLMs) are prone to
hallucinations, and Retrieval-Augmented Gen-
eration (RAG) helps mitigate this, but at a high
computational cost while risking misinforma-
tion. Adaptive retrieval aims to retrieve only
when necessary, but existing approaches rely
on LLM-based uncertainty estimation, which
remain inefficient and impractical. In this study,
we introduce lightweight LLM-independent
adaptive retrieval methods based on external
information. We investigated 27 features, or-
ganized into 7 groups, and their hybrid com-
binations. We evaluated these methods on 6
QA datasets, assessing the QA performance
and efficiency. The results show that our ap-
proach matches the performance of complex
LLM-based methods while achieving signif-
icant efficiency gains, demonstrating the po-
tential of external information for adaptive re-
trieval.
1 Introduction
Large Language Models (LLMs) excel in tasks
like question answering (QA) (Yang et al., 2018;
Kwiatkowski et al., 2019), but remain vulnerable to
hallucinations (Yin et al., 2024; Ding et al., 2024).
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020) mitigates this by incorporating external
information, although it introduces risks such as
error accumulation (Shi et al., 2023) and external
hallucinations (Ding et al., 2024).
Adaptive retrieval techniques (Moskvoretskii
et al., 2025; Ding et al., 2024; Jeong et al.,
2024) aim to balance LLM knowledge with exter-
nal resources by estimating uncertainty to decide
whether retrieval is needed.
However, existing methods primarily frame this
task as uncertainty estimation based on LLM inter-
nal states or outputs, leading to significant computa-
tional overhead. This can offset the efficiency gains
from reduced retrieval calls and limit practicality,
especially with larger models.
106
105
104
103
102
101
Inference PFLOPs (1e15) (Log-scale)49.049.550.050.551.0InAccuracy
Question
complexityQuestion
typeKnowledgabilityEntity
popularity
Context relevanceAdaptiveRAGMaxTokenEntropyEigValLaplacianFigure 1: PFLOPs-Inaccuracy trade-off for proposed
features vs the most efficient alternative adaptive re-
trieval methods for the NQ dataset. Radius of the points
is proportional to the number of LLM calls. Green dot-
ted line indicate Always RAG approach.
In this study, we address this issue by introduc-
ing LLM-independent adaptive retrieval methods
that leverage external information, such as entity
popularity and question type. Our methods achieve
comparable quality while being significantly more
efficient, eliminating the need for LLMs entirely.
Our evaluation, shown in Figure 1, shows that
our proposed features are much more efficient in
terms of PFLOPs and LLM calls, with downstream
performance comparable to other adaptive retrieval
methods.
Our contributions and findings are as follows:
1.We introduce 7 groups of lightweight external
information features, encompassing 27 fea-
tures, for LLM-independent adaptive retrieval.
2.Our approach significantly improves effi-
ciency by eliminating the need for LLM-based
uncertainty estimation while maintaining QA
performance.
3.We show that our methods outperform
uncertainty-based adaptive retrieval methods
for complex questions.
We make data and all models publicly available.1
1https://github.com/marialysyuk/External_
Adaptive_RetrievalarXiv:2505.04253v1  [cs.CL]  7 May 2025

2 Related Work
Adaptive Retrieval-Augmented Generation re-
duces unnecessary retrievals by determining
whether external knowledge is needed. This de-
cision can be based on LLM output (Trivedi et al.,
2023), consistency checks (Ding et al., 2024), inter-
nal uncertainty signals (Jiang et al., 2023; Su et al.,
2024; Yao et al., 2024), or trained classifiers (Jeong
et al., 2024).
External Information methods can enhance re-
trieval, such as integrating knowledge graphs and
the popularity of the entity. KG structures have
been incorporated into LLM decoding to enable
reasoning on graphs for more reliable answers (Luo
et al., 2024). Popularity and graph frequency im-
prove retrieval efficiency, as shown in LightRAG
and MiniRAG, which prioritize frequently accessed
entities and relationships (Guo et al., 2024; Fan
et al., 2025). Graph-based features, including entity
properties (Lysyuk et al., 2024), popularity (Mallen
et al., 2023a), and structural attributes (Salnikov
et al., 2023), have also been shown to be effective
in QA systems.
3 Methods
Our baselines include the following adaptive re-
trieval methods:
Adaptive RAG uses a T5-large-based classifier
to determine whether retrieval is needed (Jeong
et al., 2024). FLARE triggers retrieval when to-
ken probability falls below a threshold (Jiang et al.,
2023). DRAGIN estimates uncertainty based on
token probabilities and attention weights, exclud-
ing stopwords (Su et al., 2024). Rowen relies on
consistency checks across languages and models to
trigger retrieval (Ding et al., 2024). SeaKR moni-
tors internal state consistency to trigger retrieval, re-
ranking snippets to reduce uncertainty (Yao et al.,
2024). EigValLaplacian assesses uncertainty us-
ing graph features based on pairwise consistency
scores (Lin et al., 2023). Max Token Entropy
measures uncertainty by aggregating the maximum
entropy of token distributions (Fomicheva et al.,
2020). Hybrid UEincludes 5 uncertainty features
relevant to the task (Moskvoretskii et al., 2025):
Mean Token Entropy, Max Token Entropy, SAR,
EigValLaplacian, Lex-Similarity.
3.1 External Information Methods
In this section, we describe the proposed exter-
nal information methods for adaptive retrieval.Each group may contain multiple features used
to train a classifier to predict retrieval needs, fol-
lowing Moskvoretskii et al. (2025); Jeong et al.
(2024).
Graph features capture information about the
entities in question from a knowledge graph, includ-
ing the minimum, maximum, and mean number of
triples per subject and object, where the subject or
object corresponds to an entity from the question.
Popularity features include the minimum, maxi-
mum, and mean number of Wikipedia page views
per entity in the question.
Frequency features include the minimum, max-
imum and mean frequencies of entities in a refer-
ence text collection2, along with the frequency of
the least common n-gram in the question.
Knowledgability features assign a score to each
entity, reflecting the LLM’s verbalized uncertainty
about its knowledge. By pre-computing these
scores for entities in the Wikidata Knowledge
Graph, retrieval decisions can be made without
querying the LLM at inference time.
Question Type features include probabilities for
nine categories: ordinal, count, generic, superla-
tive, difference, intersection, multihop, compara-
tive, and yes/no.
Question Complexity reflects the difficulty of a
question, considering the reasoning steps required.
Context Relevance features include the mini-
mum, maximum and mean probabilities that a con-
text is relevant to the question, along with the con-
text length.
Hybrid External includes all external features.
Hybrid ¬UFPincludes all external features ex-
cept frequency and popularity, as they are highly
correlated with graph features.
Hybrid ¬FPincludes uncertainty and all external
features except frequency and popularity.
The details of all methods are described in the
Appendix A.
4 Experimental Setup
In this section, we briefly discuss the implementa-
tion details and the evaluation setup.
4.1 Implementation Details
We use LLaMA 3.1-8B-Instruct (Dubey et al.,
2024) and the BM25 retriever (Robertson et al.,
1994) as the main components of our approach,
2https://www.inf.uni-hamburg.de/en/inst/ab/lt/
resources/data/depcc.html

MethodNQ SQuAD TQA 2Wiki HotPot Musique
InAcc↑LMC↓RC↓InAcc↑LMC↓RC↓InAcc↑LMC↓RC↓InAcc↑LMC↓RC↓InAcc↑LMC↓RC↓InAcc↑LMC↓RC↓
Never RAG 44.6 1.0 0.00 17.6 1.0 0.00 63.6 1.0 0.00 31.8 1.0 0.00 28.6 1.0 0.00 10.6 1.0 0.00
Always RAG 49.6 1.0 1.00 31.2 1.0 1.00 61.0 1.0 1.00 37.4 1.0 1.00 41.0 1.0 1.00 10.0 1.0 1.00
Multi-Step Adaptive Retrieval
AdaptiveRAG 49.6 2.0 0.98 28.6 2.0 0.97 62.8 1.5 0.54 45.4 5.2 2.64 41.4 4.6 2.34 14.0 3.6 3.63
DRAGIN 48.0 4.5 2.24 29.8 4.3 2.14 66.6 4.1 2.06 45.6 5.8 2.92 43.0 5.1 2.56 13.4 6.3 3.15
FLARE 45.0 3.1 2.07 23.8 3.1 2.08 64.8 2.1 1.39 42.4 3.9 2.85 37.2 5.1 4.07 9.0 4.1 3.10
Rowen CM 49.4 29.5 7.27 19.6 29.2 7.20 65.6 28.7 7.12 44.4 32.9 7.87 35.6 31.9 7.70 10.4 42.1 9.52
Seakr 40.6 14.6 1.00 26.8 14.6 1.00 65.6 14.6 1.00 39.8 12.3 2.44 42.4 9.9 1.76 11.8 12.3 2.40
Uncertainty Estimation
EigValLaplacian 51.2 1.8 0.81 31.4 2.0 0.10 64.0 1.3 0.26 38.4 2.0 0.98 41.0 1.9 0.91 10.2 2.0 1.00
MaxTokenEntropy 50.6 1.7 0.58 31.2 2.0 0.10 65.0 1.2 0.22 37.6 2.0 0.95 41.4 2.0 0.99 10.6 2.0 0.87
Hybrid UE 50.2 1.7 0.77 31.4 2.0 0.98 63.8 1.3 0.27 37.4 2.0 0.98 41.2 1.9 0.94 10.6 1.8 0.75
External Features
Graph 49.0 1.0 0.87 30.4 1.0 0.95 63.6 1.0 0.32 35.8 1.0 0.67 40.8 1.0 0.97 10.0 1.0 1.00
Popularity 49.8 1.0 0.92 31.2 1.0 1.00 63.2 1.0 0.15 35.6 1.0 0.84 41.0 1.0 0.94 10.0 1.0 0.96
Frequency 49.8 1.0 0.96 31.0 1.0 0.99 63.2 1.0 0.04 37.4 1.0 0.84 41.0 1.0 0.94 10.0 1.0 0.96
Knowledgability 49.6 1.0 0.95 31.2 1.0 1.00 63.0 1.0 0.28 38.4 1.0 0.89 41.0 1.0 1.00 9.8 1.0 0.61
Question type 49.6 1.0 0.88 30.4 1.0 0.97 64.0 1.0 0.29 35.6 1.0 0.74 39.6 1.0 0.88 10.0 1.0 1.00
Question complexity 49.6 1.0 1.00 31.2 1.0 1.00 63.6 1.0 0.10 36.8 1.0 0.94 41.0 1.0 1.00 10.6 1.0 0.95
Context relevance 49.0 1.0 1.00 31.0 1.0 1.00 62.8 1.0 1.00 36.0 1.0 1.00 41.0 1.0 1.00 10.6 1.0 1.00
Hybrids with External Features
Hybrid UFP 47.8 1.0 1.00 30.8 1.0 1.00 63.4 1.0 1.00 36.4 1.0 1.00 39.8 1.0 1.00 10.6 1.0 1.00
Hybrid External 46.0 1.8 1.00 30.2 2.0 1.00 63.2 1.3 1.00 37.0 2.0 1.00 39.2 1.9 1.00 12.2 1.8 1.00
Hybrids with Uncertainty and External Features
Hybrid FP 48.4 1.8 1.00 31.2 2.0 1.00 64.6 1.3 1.00 37.8 2.0 1.00 41.0 1.9 1.00 12.2 1.8 1.00
All 47.6 1.8 1.00 30.8 2.0 1.00 64.2 1.3 1.00 37.8 2.0 1.00 37.8 1.9 1.00 11.2 1.8 1.00
Ideal 60.8 1.6 0.55 36.0 1.8 0.82 73.6 1.4 0.36 50.0 1.7 0.68 46.0 1.7 0.71 16.4 1.9 0.89
Table 1: QA Performance of adaptive retrieval and uncertainty methods. ‘Ideal’ represents the performance of a
system with an oracle providing ideal predictions for the need to retrieve. ‘InAcc’ denotes In-Accuracy, measuring
the QA system’s performance. ‘LMC’ indicates the mean number of LM calls per question, and ‘RC’ represents the
mean number of retrieval calls per question. The SOTA results are highlighted in bold, as well as the best results for
the external methods.
following Yao et al. (2024); Jeong et al. (2024);
Moskvoretskii et al. (2025).
4.2 Datasets
We evaluate on single-hop SQuAD v1.1 (Rajpurkar
et al., 2016), Natural Questions (Kwiatkowski et al.,
2019), TriviaQA (Joshi et al., 2017) and multi-hop
MuSiQue (Trivedi et al., 2022), HotpotQA (Yang
et al., 2018), 2WikiMultiHopQA (2wiki) (Ho et al.,
2020) QA datasets to ensure real-world query com-
plexity, following Trivedi et al. (2023); Jeong et al.
(2024); Su et al. (2024); Yao et al. (2024). We use
500-question subsets from the original test sets, as
in Moskvoretskii et al. (2025); Jeong et al. (2024).
4.3 Evaluation
We evaluate both the quality and efficiency of the
adaptive retrieval system. For quality, we use In-
Accuracy (InAcc) , which measures whether the
LLM output contains the ground-truth answer, as
it is a reliable metric based on Moskvoretskii et al.
(2025); Mallen et al. (2023b); Jeong et al. (2024);
Asai et al. (2024); Baek et al. (2023).
Following Jeong et al. (2024); Moskvoretskii
et al. (2025), for efficiency we adopt Retrieval
Calls (RC) – the average number of retrievals perquestion, and LM Calls (LMC) – the average num-
ber of LLM calls per question, including uncer-
tainty estimation. Further details are provided in
Appendix B.
5 Results
In the following sections, we present the results of
the end-to-end and UE methods, as well as groups
of external features, focusing on downstream per-
formance and efficiency. For comparison, we also
include the ‘Never RAG’, ‘Always RAG’, and
‘Ideal’ benchmarks. The ‘Ideal’ benchmark rep-
resents the performance of a system with an oracle
providing perfect retrieval predictions.
Downstream Performance First, we assess
whether external methods can replace uncertainty-
based approaches. As shown in Table 1, at least
one external feature matches the performance of
the uncertainty estimation methods for each dataset.
Combining external features even increases InAc-
curacy for the Musique dataset. Compared to Multi-
Step Adaptive Retrieval, using only external fea-
tures yields similar results across all datasets, ex-
cept for 2wiki.
Second, we examine whether external methods

Is ComplexMax Context RelevanceIntersection Question ProbabilityLLaMA KnowledgeMin Context Relevance
External Features
0 5 10 15 20
Importance ScoreMax T oken EntropyMean T oken EntropyLexical Similarity (ROUGE-L)SAREigValLaplacian NLI Entailment Score
All Features(a) trivia
Min Context RelevanceMax Subject GraphMean Context RelevanceCount Question ProbabilityMin Object Graph
External Features
0.00 0.05 0.10 0.15 0.20
Importance ScoreMean T oken EntropyOrdinal Question ProbabilityMax T oken EntropyMean Context RelevanceMultihop Question Probability
All Features (b) musique
Figure 2: Feature importances for one of the best algorithms for only external features vs all features for TriviaQA
(simple) and Musique (complex) datasets.
Uncertainty
Popularity
Graph
Frequency
Question
 type
Question
 complexity
Knowledgability
Context
 relevance
Context
 length
Class labelUncertainty
Popularity
Graph
Frequency
Question
 type
Question
 complexity
Knowledgability
Context
 relevance
Context
 length
Class label2wiki1 0.1 0.17 0.09 0.08 0.28 0.19 0.22 0.13 0.37
0 1 0.33 0.02 0.16 0.01 0.08 0.09 0.05 0.08
0.01 0.19 1 0.03 0.01 0.07 0.16 0.02 0.08 0.14
0.04 0.01 0.02 1 0.02 0.01 0.15 0.1 0.12 0.08
0.21 0.06 0.24 0.03 1 0.05 0.02 0.04 0.05 0.08
0.3 0.01 0.01 0 0.01 1 0.08 0.11 0.03 0.22
0.11 0.13 0.21 0.05 0.04 0.11 1 0.2 0.08 0.06
0.06 0.04 0.14 0.03 0.25 0 0.07 1 0.14 0.06
0.02 0.02 0.12 0.02 0.03 0.05 0.04 0.12 1 0.04
0.19 0.01 0.09 0.08 0.41 0.09 0.21 0.18 0.01 1Trivia
0.00.20.40.60.81.0
Figure 3: Heatmap of different groups of features for
TriviaQA and 2WikiMultiHopQA (2wiki) datasets. Up-
per right triangle states for the absolute correlations on
the TriviaQA, while down left states for the absolute
correlations on the 2WikiMultiHopQA
complement uncertainty-based approaches. Our
findings show that hybrids with uncertainty fea-
tures do not outperform any external feature com-
binations, suggesting that these features are more
substitutive than complementary.
Efficiency Performance External features sig-
nificantly reduce LLM calls, addressing a key effi-
ciency bottleneck that worsens with LLM scaling.
However, they lead to slightly more conservative
behavior with increased Retrieval Calls, though
still fewer than Multi-Step approaches. Since ex-
ternal information features are pre-computed, no
additional LLM calls are required during inference.6 Features Reciprocity
We identify four key aspects that influence adap-
tive retrieval performance: LLM knowledge (un-
certainty features, knowledgability), question type
(simple vs. complex reasoning), context rele-
vance (irrelevant context reduces performance),
and entity rarity (approximated by entity popularity
groups). Figure 2 shows that for the simple Triv-
iaQA dataset, the Top-5 features are uncertainty-
based, while for complex datasets, question type
and context relevance become more important.
Thus, relying solely on uncertainty-based features
is insufficient for efficient adaptive retrieval.
External features tend to be more substitutive
than complementary, as they often exhibit strong
correlations despite their differences. As shown in
Figure 3, for simple questions, uncertainty strongly
correlates with graph features, question complexity,
knowledgability, and context relevance. For com-
plex questions, uncertainty correlates with question
complexity, type, and knowledgability. Heatmaps
and feature importances for other datasets could be
found in Appendix D.
7 Conclusion
In this work, we introduced 7 groups of lightweight
external features for LLM-independent adaptive
retrieval, improving efficiency by eliminating the
need for LLM-based uncertainty estimation while
preserving QA performance. Our approach out-
performs uncertainty-based methods for complex
questions and offers a detailed analysis of the com-
plementarity between uncertainty and external fea-
tures.

Limitations
•We evaluate model performance using six
widely adopted QA datasets. Incorporating
a broader range of datasets, particularly those
tailored to specific domains, could offer more
comprehensive insights and showcase the ver-
satility of our approach.
•Our study focuses on the LLaMA3.1-8B-
Instruct model, a top-performing open-source
model within its parameter range. Expanding
the analysis to additional architectures could
further strengthen the generalizability of our
results.
Ethical Considerations
Text retrieval systems can introduce biases into re-
trieved documents, which may inadvertently steer
the outputs of even ethically aligned LLMs in un-
intended directions. Consequently, developers in-
tegrating RAG and Adaptive RAG pipelines into
user-facing applications should account for this po-
tential risk.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024 . OpenReview.net.
Jinheon Baek, Soyeong Jeong, Minki Kang, Jong C.
Park, and Sung Ju Hwang. 2023. Knowledge-
augmented language model verification. In Proceed-
ings of the 2023 Conference on Empirical Methods
in Natural Language Processing, EMNLP 2023, Sin-
gapore, December 6-10, 2023 , pages 1720–1736. As-
sociation for Computational Linguistics.
Lars Buitinck, Gilles Louppe, Mathieu Blondel, Fabian
Pedregosa, Andreas Mueller, Olivier Grisel, Vlad
Niculae, Peter Prettenhofer, Alexandre Gramfort,
Jaques Grobler, Robert Layton, Jake VanderPlas, Ar-
naud Joly, Brian Holt, and Gaël Varoquaux. 2013.
API design for machine learning software: experi-
ences from the scikit-learn project. In ECML PKDD
Workshop: Languages for Data Mining and Machine
Learning , pages 108–122.
Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen,
and Xueqi Cheng. 2024. Retrieve only when it
needs: Adaptive retrieval augmentation for halluci-
nation mitigation in large language models. CoRR ,
abs/2402.10612.Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, et al. 2024. The llama 3 herd of models. arXiv
preprint arXiv:2407.21783 .
Research Facebook. 2019. fvcore: A light-weight core
library for computer vision frameworks. https://
github.com/facebookresearch/fvcore .
Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao
Huang. 2025. Minirag: Towards extremely
simple retrieval-augmented generation. Preprint ,
arXiv:2501.06713.
Marina Fomicheva, Shuo Sun, Lisa Yankovskaya,
Frédéric Blain, Francisco Guzmán, Mark Fishel,
Nikolaos Aletras, Vishrav Chaudhary, and Lucia Spe-
cia. 2020. Unsupervised quality estimation for neural
machine translation. Transactions of the Association
for Computational Linguistics , 8:539–555.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2024. Lightrag: Simple and fast retrieval-
augmented generation. Preprint , arXiv:2410.05779.
John T Hancock and Taghi M Khoshgoftaar. 2020. Cat-
boost for big data: an interdisciplinary review. Jour-
nal of big data , 7(1):94.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing A multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. In Proceedings of the 28th International
Conference on Computational Linguistics, COLING
2020, Barcelona, Spain (Online), December 8-13,
2020 , pages 6609–6625. International Committee on
Computational Linguistics.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. In Proceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), NAACL 2024, Mexico City, Mexico, June
16-21, 2024 , pages 7036–7050. Association for Com-
putational Linguistics.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics, ACL
2017, Vancouver, Canada, July 30 - August 4, Volume
1: Long Papers , pages 1601–1611. Association for
Computational Linguistics.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur P. Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: a benchmark for question answering
research. Trans. Assoc. Comput. Linguistics , 7:452–
466.
Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Advances in Neu-
ral Information Processing Systems 33: Annual Con-
ference on Neural Information Processing Systems
2020, NeurIPS 2020, December 6-12, 2020, virtual .
Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2023.
Generating with confidence: Uncertainty quantifi-
cation for black-box large language models. arXiv
preprint arXiv:2305.19187 .
Linhao Luo, Zicheng Zhao, Chen Gong, Gholamreza
Haffari, and Shirui Pan. 2024. Graph-constrained rea-
soning: Faithful reasoning on knowledge graphs with
large language models. Preprint , arXiv:2410.13080.
Maria Lysyuk, Mikhail Salnikov, Pavel Braslavski, and
Alexander Panchenko. 2024. Konstruktor: A strong
baseline for simple knowledge graph question an-
swering. CoRR , abs/2409.15902.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023a.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023b.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), ACL 2023, Toronto, Canada,
July 9-14, 2023 , pages 9802–9822. Association for
Computational Linguistics.
Viktor Moskvoretskii, Maria Lysyuk, Mikhail Sal-
nikov, Nikolay Ivanov, Sergey Pletenev, Daria Gal-
imzianova, Nikita Krayko, Vasily Konovalov, Irina
Nikishina, and Alexander Panchenko. 2025. Adap-
tive retrieval without self-knowledge? bringing uncer-
tainty back home. arXiv preprint arXiv:2501.12835 .
Mikhail Plekhanov, Nora Kassner, Kashyap Popat,
Louis Martin, Simone Merello, Borislav Kozlovskii,
Frédéric A. Dreyer, and Nicola Cancedda. 2023.
Multilingual end to end entity linking. CoRR ,
abs/2306.08896.Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100, 000+ questions
for machine comprehension of text. In Proceedings
of the 2016 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2016, Austin,
Texas, USA, November 1-4, 2016 , pages 2383–2392.
The Association for Computational Linguistics.
Stephen E. Robertson, Steve Walker, Susan Jones,
Micheline Hancock-Beaulieu, and Mike Gatford.
1994. Okapi at TREC-3. In Proceedings of The Third
Text REtrieval Conference, TREC 1994, Gaithers-
burg, Maryland, USA, November 2-4, 1994 , volume
500-225 of NIST Special Publication , pages 109–
126. National Institute of Standards and Technology
(NIST).
Mikhail Salnikov, Hai Le, Prateek Rajput, Irina Nik-
ishina, Pavel Braslavski, Valentin Malykh, and
Alexander Panchenko. 2023. Large language models
meet knowledge graphs to answer factoid questions.
InProceedings of the 37th Pacific Asia Conference
on Language, Information and Computation , pages
635–644, Hong Kong, China. Association for Com-
putational Linguistics.
Priyanka Sen, Alham Fikri Aji, and Amir Saffari.
2022. Mintaka: A complex, natural, and multi-
lingual dataset for end-to-end question answering.
InProceedings of the 29th International Confer-
ence on Computational Linguistics, COLING 2022,
Gyeongju, Republic of Korea, October 12-17, 2022 ,
pages 1604–1619. International Committee on Com-
putational Linguistics.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed H. Chi, Nathanael Schärli,
and Denny Zhou. 2023. Large language models can
be easily distracted by irrelevant context. In Interna-
tional Conference on Machine Learning, ICML 2023,
23-29 July 2023, Honolulu, Hawaii, USA , volume
202 of Proceedings of Machine Learning Research ,
pages 31210–31227. PMLR.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024. DRAGIN: dynamic retrieval
augmented generation based on the real-time informa-
tion needs of large language models. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
ACL 2024, Bangkok, Thailand, August 11-16, 2024 ,
pages 12991–13013. Association for Computational
Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics , 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),

ACL 2023, Toronto, Canada, July 9-14, 2023 , pages
10014–10037. Association for Computational Lin-
guistics.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry
Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny
Zhou, Quoc Le, and Thang Luong. 2023. Freshllms:
Refreshing large language models with search engine
augmentation. Preprint , arXiv:2310.03214.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. In Proceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
Brussels, Belgium, October 31 - November 4, 2018 ,
pages 2369–2380. Association for Computational
Linguistics.
Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao,
Linmei Hu, Weichuan Liu, Lei Hou, and Juanzi
Li. 2024. Seakr: Self-aware knowledge retrieval
for adaptive retrieval augmented generation. CoRR ,
abs/2406.19215.
Xunjian Yin, Xu Zhang, Jie Ruan, and Xiaojun Wan.
2024. Benchmarking knowledge boundary for large
language models: A different perspective on model
evaluation. In Proceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), ACL 2024, Bangkok, Thai-
land, August 11-16, 2024 , pages 2270–2286. Associ-
ation for Computational Linguistics.

A External Methods
Graph Using the BELA entity linking module (Plekhanov et al., 2023), the entities from the question
are linked to the corresponding IDs in the Wikidata knowledge graph. Then, for each entity the number of
triples where this entity is either an object or a subject is retrieved. Finally, six features are calculated: the
minimum/maximum/mean number of triples per subject and object.
Popularity Using the BELA NER module (Plekhanov et al., 2023) the entities are retrieved from
the question. Then, for each entity the mean amount of views per Wikipedia page is calculated using
Wikimedia API3for last year. Finally, there are three features: the minimum/maximum/mean number of
views per entity per question.
Knowledgability The prompt to the LLaMA 3.1-8B-Instruct model to approximate its interal knowl-
edge:
Answer the following question based on your internal knowledge with one or few words.
If you are sure the answer is accurate and correct, please say ‘100’. If you are not confident with
the answer, please range your knowledgability from 0 to 100, say just number. For example, ‘40’.
Question: {question}. Answer:
Question type Using the train part of the Mintaka dataset (Sen et al., 2022), we train a classifier based
on the bert-base-uncased model4to predict whether a question belongs to one of the 12 question types:
‘ordinal’, ‘count’, ‘generic’, ‘superlative’, ‘difference’, ‘intersection’, ‘multihop’, ‘yesno’, ‘intersection’,
‘comparative’, ‘multihop’, ‘yes/no’. As a result, we get twelve probabilities that the question belongs to a
certain class. The accuracy classification score on the validation part of the Mintaka dataset is 0.93.
Question Complexity is based on N-hop feature from FreshQA (Vu et al., 2023) dataset:
•One-hop, where the question is explicit about all the relevant information needed to complete the
task, so no additional inference is needed.
•Multi-hop, where the question requires one or more additional inference steps to gather all the
relevant information needed to complete the task.
The dataset consists of 500 training and 100 test examples. As a training model, we used a Distil-bert5
model. The final F1 score on the test set is 0.82.
Context relevance Each question with one context at a time is passed to the cross-encoder model based
on the uncased model of the bert base. A question and a context are passed via the [SEP] token with the
additional classification head over the base model. The final probabilities of each context being relevant
are aggregated via minimum/maximum/mean across all contexts. Additionally, there is the fourth feature
that calculates the context length.
B Technical Details
Train setting. We conduct all experiments using the LLaMA 3.1-8B-Instruct model with its default
generation parameters. The responses generated, with and without the retriever, are sourced from previous
studies (Moskvoretskii et al., 2025), following the AdaptiveRAG framework (Jeong et al., 2024). The
baseline results are also adopted from prior work, as we employ the exact same settings and generation
configurations.
3https://foundation.wikimedia.org/wiki/Api/
4https://hf.co/google-bert/bert-base-uncased
5https://hf.co/distilbert/distilbert-base-uncased

We implemented classifiers using Scikit-learn (Buitinck et al., 2013), CatBoost (Hancock and Khoshgof-
taar, 2020), and performed hyperparameter tuning using a validation set of 100samples randomly selected
from the training set, testing with three different random seeds for each dataset. We evaluated seven
classifiers: Logistic Regression, KNN, MLP, Decision Tree, CatBoosting, Gradient Boosting, and Random
Forest. Data preprocessing involved standard scaling. For the final model, we used a V otingClassifier,
combining the two best-performing classifiers from the validation set, each trained with their optimal
hyperparameters. Performance was evaluated based on the In-accuracy metric, and the top classifiers were
retrained on the full training set with these selected hyperparameters.
Hyperparameters grid. Logistic Regression : C: [0.01, 0.1, 1], solver: [lbfgs, liblinear], class_weight:
[balanced, 0: 1, 1: 1, None], max_iter: [10000, 15000, 20000]
KNN : n_neighbors: [5, 7, 9, 11, 13, 15], metric: [euclidean, manhattan], algorithm: [auto, ball_tree,
kd_tree], weights: [uniform, distance]
MLP : hidden_layer_sizes: [(50,), (100,), (50, 50), (100, 50), (100, 100)], activation: [relu, tanh], solver:
[adam, sgd], alpha: [0.00001, 0.0001, 0.001, 0.01], learning_rate: [constant, adaptive], early_stopping:
True, max_iter: [200, 500]
Decision Tree : max_depth: [3, 5, 7, 10, None], max_features: [0.2, 0.4, sqrt, log2, None], criterion:
[gini, entropy], splitter: [best, random]
CatBoosting : iterations: [10, 50, 100, 200], learning_rate: [0.001, 0.01, 0.05], depth: [3, 4, 5, 7, 9],
bootstrap_type: [Bayesian, Bernoulli, MVS]
Gradient Boosting : n_estimators: [25, 35, 50], learning_rate: [0.001, 0.01, 0.05], max_depth: [3, 4, 5,
7, 9], max_features: [0.2, 0.4, sqrt, log2, None]
Random Forest : n_estimators: [25, 35, 50], max_depth: [3, 5, 7, 9, 11], max_features: [0.2, 0.4, sqrt,
log2, None], bootstrap: [True, False], criterion: [gini, entropy], class_weight: [balanced, 0: 1, 1: 1, None]
C FLOPs calculation
MethodNQ
Mean Upper bound
AdaptiveRAG 0.0216 0.4389
SeaKR 0.3504 2.4548
DRAGIN 0.2608 1.0129
FLARE 0.09699 0.9290
Rowen 1.865 15.9677
EigValLaplacian 0.10517 0.3291
MaxTokenEntropy 0.027116 0.22121
Entity_popularity 0.0181238962 0.210304
Is_complex 0.0181418 0.2082277
Llama_know 0.018291 0.22747
Context_relevance 0.018327 0.2084429
Question_type 0.01812162 0.2073669
Table 2: A comparison of FLOPs usage across different methods on the Natural Questions (NQ) dataset. The “Mean”
column shows the average PFLOPs ( 1015FLOPs) per question, while the “Upper bound” column represents the
theoretical maximum FLOPs assuming the LLaMA 3.1 8B model (in FP16 precision) runs at 100% GPU utilization
for the entire processing of a single sample. The row labeled “Entity_popularity” reflects the computational overhead
required for graph/popularity/frequency features. It is important to note that for features such as "Entity_popularity",
"Is_complex", "Llama_know", "Context_relevance", "Question_type" the generation of final answer for a question
(after precomputing these features) accounts for more than 99% of the total FLOPs.
To calculate floating-point operations (FLOPs), we used the fvcore (Facebook, 2019) library developed
by Facebook Research. This library provides a flexible and efficient interface for analyzing the computa-
tional complexity of PyTorch models. Specifically, we wrapped our model generation process with the
FlopCountAnalysis class, which automatically traces the model forward pass and counts the number of
FLOPs for each layer. The theoretical analysis includes an approximate formula to calculate an upper

bound per sample:
Total FLOPs ≈ 
Total TFLOPs
×1012×(Elapsed Seconds ),
where Total TFLOPs = (TFLOPs per GPU )×(Number of GPUs ),assuming 100% utilization.
D Heatmaps and feature importances for all datasets
Intersection Question ProbabilityContext LengthMean Context RelevanceMax Context RelevanceMin Context Relevance
External Features
0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07
Importance ScoreLexical Similarity (ROUGE-L)Max T oken EntropyMean T oken EntropySARMean Context Relevance
All Features
(a) nq
Comparative Question ProbabilityMin PopularitySuperlative Question ProbabilityMin Context RelevanceDifference Question Probability
External Features
0.0 0.1 0.2 0.3 0.4
Importance ScoreSARIs ComplexMean T oken EntropyIntersection Question ProbabilityMin Inexact Frequency
All Features (b) squad
Comparative Question ProbabilityYes/No Question ProbabilityMultihop Question ProbabilityCount Question ProbabilityMean Context Relevance
External Features
0.00 0.05 0.10 0.15 0.20 0.25
Importance ScoreComparative Question ProbabilityYes/No Question ProbabilityMultihop Question ProbabilityCount Question ProbabilityMax Context Relevance
All Features
(c) 2wikimultihop
Yes/No Question ProbabilityComparative Question ProbabilityMultihop Question ProbabilityOrdinal Question ProbabilityMax Exact Frequency
External Features
0.00 0.02 0.04 0.06 0.08 0.10
Importance ScoreMax T oken EntropyComparative Question ProbabilityYes/No Question ProbabilityOrdinal Question ProbabilityMultihop Question Probability
All Features (d) hotpot
Figure 4: Feature importances for one of the best algorithms for only external features vs all features for NQ,
TriviaQA (simple) and HotpotQA, Musique (complex) datasets.

Uncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class labelUncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class label1 0.04 0.04 0.02 0.02 0.19 0.17 0.12 0.18 0.2
0.04 1 0.03 0 0.01 0.01 0.02 0.03 0.02 0.05
0.04 0.03 1 0.01 0.02 0.02 0.06 0.02 0.04 0.03
0.02 0 0.01 1 0.04 0.02 0.03 0.21 0.03 0.02
0.02 0.01 0.02 0.04 1 0.03 0.08 0.01 0.08 0.08
0.19 0.01 0.02 0.02 0.03 1 0.01 0.06 0.04 0.21
0.17 0.02 0.06 0.03 0.08 0.01 1 0.14 0.17 0.02
0.12 0.03 0.02 0.21 0.01 0.06 0.14 1 0.08 0.05
0.18 0.02 0.04 0.03 0.08 0.04 0.17 0.08 1 0.07
0.2 0.05 0.03 0.02 0.08 0.21 0.02 0.05 0.07 1
0.00.20.40.60.81.0(a) nq
Uncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class labelUncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class label1 0.11 0.05 0.01 0.1 0.31 0.19 0.05 0.06 0.24
0.11 1 0.4 0.05 0.15 0.03 0.12 0.15 0.01 0.03
0.05 0.4 1 0.07 0.03 0.05 0.3 0.11 0.03 0.05
0.01 0.05 0.07 1 0.02 0.05 0.03 0.04 0.06 0.05
0.1 0.15 0.03 0.02 1 0 0.02 0.04 0.02 0.06
0.31 0.03 0.05 0.05 0 1 0.17 0.16 0.01 0.23
0.19 0.12 0.3 0.03 0.02 0.17 1 0.15 0.09 0.09
0.05 0.15 0.11 0.04 0.04 0.16 0.15 1 0.13 0.08
0.06 0.01 0.03 0.06 0.02 0.01 0.09 0.13 1 0.03
0.24 0.03 0.05 0.05 0.06 0.23 0.09 0.08 0.03 1
0.00.20.40.60.81.0 (b) squad
Uncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class labelUncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class label1 0.1 0.17 0.09 0.08 0.28 0.19 0.22 0.13 0.37
0.1 1 0.33 0.02 0.16 0.01 0.08 0.09 0.05 0.08
0.17 0.33 1 0.03 0.01 0.07 0.16 0.02 0.08 0.14
0.09 0.02 0.03 1 0.02 0.01 0.15 0.1 0.12 0.08
0.08 0.16 0.01 0.02 1 0.05 0.02 0.04 0.05 0.08
0.28 0.01 0.07 0.01 0.05 1 0.08 0.11 0.03 0.22
0.19 0.08 0.16 0.15 0.02 0.08 1 0.2 0.08 0.06
0.22 0.09 0.02 0.1 0.04 0.11 0.2 1 0.14 0.06
0.13 0.05 0.08 0.12 0.05 0.03 0.08 0.14 1 0.04
0.37 0.08 0.14 0.08 0.08 0.22 0.06 0.06 0.04 1
0.00.20.40.60.81.0
(c) trivia
Uncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class labelUncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class label1 0.04 0.12 0.05 0.14 0.28 0.16 0.14 0.01 0.29
0.04 1 0.25 0.32 0.03 0.04 0.04 0.01 0.05 0.07
0.12 0.25 1 0.18 0.09 0.03 0.2 0.04 0.03 0.07
0.05 0.32 0.18 1 0.04 0.04 0.01 0.12 0.04 0.06
0.14 0.03 0.09 0.04 1 0.13 0 0.06 0.02 0.21
0.28 0.04 0.03 0.04 0.13 1 0.08 0.16 0.04 0.16
0.16 0.04 0.2 0.01 0 0.08 1 0.13 0.09 0.11
0.14 0.01 0.04 0.12 0.06 0.16 0.13 1 0.03 0.08
0.01 0.05 0.03 0.04 0.02 0.04 0.09 0.03 1 0.03
0.29 0.07 0.07 0.06 0.21 0.16 0.11 0.08 0.03 1
0.00.20.40.60.81.0 (d) hotpot
Uncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class labelUncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class label1 0 0.01 0.04 0.21 0.3 0.11 0.06 0.02 0.19
0 1 0.19 0.01 0.06 0.01 0.13 0.04 0.02 0.01
0.01 0.19 1 0.02 0.24 0.01 0.21 0.14 0.12 0.09
0.04 0.01 0.02 1 0.03 0 0.05 0.03 0.02 0.08
0.21 0.06 0.24 0.03 1 0.01 0.04 0.25 0.03 0.41
0.3 0.01 0.01 0 0.01 1 0.11 0 0.05 0.09
0.11 0.13 0.21 0.05 0.04 0.11 1 0.07 0.04 0.21
0.06 0.04 0.14 0.03 0.25 0 0.07 1 0.12 0.18
0.02 0.02 0.12 0.02 0.03 0.05 0.04 0.12 1 0.01
0.19 0.01 0.09 0.08 0.41 0.09 0.21 0.18 0.01 1
0.00.20.40.60.81.0
(e) 2wikimultihop
Uncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class labelUncertainty
Popularity
Graph
Frequency
Question type
Question complexity
Knowledgability
Context relevance
Context length
Class label1 0.09 0.14 0.15 0.04 0.11 0.21 0.02 0.11 0.17
0.09 1 0.2 0.22 0.01 0.03 0 0.01 0.01 0.04
0.14 0.2 1 0.42 0.01 0.05 0.07 0.03 0.09 0.07
0.15 0.22 0.42 1 0.03 0.02 0.1 0.06 0.11 0.06
0.04 0.01 0.01 0.03 1 0.01 0.04 0.05 0.04 0.11
0.11 0.03 0.05 0.02 0.01 1 0.02 0.07 0 0.15
0.21 0 0.07 0.1 0.04 0.02 1 0.08 0.01 0.07
0.02 0.01 0.03 0.06 0.05 0.07 0.08 1 0.12 0.12
0.11 0.01 0.09 0.11 0.04 0 0.01 0.12 1 0.02
0.17 0.04 0.07 0.06 0.11 0.15 0.07 0.12 0.02 1
0.00.20.40.60.81.0 (f) musique
Figure 5: Absolute correlation of features from different groups of external features with class label