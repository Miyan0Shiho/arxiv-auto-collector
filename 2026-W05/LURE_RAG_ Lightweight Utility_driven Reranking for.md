# LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG

**Authors**: Manish Chandra, Debasis Ganguly, Iadh Ounis

**Published**: 2026-01-27 12:26:31

**PDF URL**: [https://arxiv.org/pdf/2601.19535v1](https://arxiv.org/pdf/2601.19535v1)

## Abstract
Most conventional Retrieval-Augmented Generation (RAG) pipelines rely on relevance-based retrieval, which often misaligns with utility -- that is, whether the retrieved passages actually improve the quality of the generated text specific to a downstream task such as question answering or query-based summarization. The limitations of existing utility-driven retrieval approaches for RAG are that, firstly, they are resource-intensive typically requiring query encoding, and that secondly, they do not involve listwise ranking loss during training. The latter limitation is particularly critical, as the relative order between documents directly affects generation in RAG. To address this gap, we propose Lightweight Utility-driven Reranking for Efficient RAG (LURE-RAG), a framework that augments any black-box retriever with an efficient LambdaMART-based reranker. Unlike prior methods, LURE-RAG trains the reranker with a listwise ranking loss guided by LLM utility, thereby directly optimizing the ordering of retrieved documents. Experiments on two standard datasets demonstrate that LURE-RAG achieves competitive performance, reaching 97-98% of the state-of-the-art dense neural baseline, while remaining efficient in both training and inference. Moreover, its dense variant, UR-RAG, significantly outperforms the best existing baseline by up to 3%.

## Full Text


<!-- PDF content starts -->

LURE-RAG: Lightweight Utility-driven
Reranking for Efficient RAG
Manish Chandra1, Debasis Ganguly1, and Iadh Ounis1
University of Glasgow, Glasgow, United Kingdom
m.chandra.1@research.gla.ac.uk, Debasis.Ganguly@glasgow.ac.uk,
iadh.ounis@glasgow.ac.uk
Abstract.Most conventional Retrieval-Augmented Generation (RAG)
pipelines rely on relevance-based retrieval, which often misaligns with
utility – that is, whether the retrieved passages actually improve the
quality of the generated text specific to a downstream task such as ques-
tion answering or query-based summarization. The limitations of exist-
ing utility-driven retrieval approaches for RAG are that, firstly, they
are resource-intensive typically requiring query encoding, and that sec-
ondly, they do not involve listwise ranking loss during training. The
latter limitation is particularly critical, as the relative order between
documents directly affects generation in RAG. To address this gap, we
propose Lightweight Utility-driven Reranking for Efficient RAG (LURE-
RAG), a framework that augments any black-box retriever with an effi-
cient LambdaMART-based reranker. Unlike prior methods, LURE-RAG
trains the reranker with a listwise ranking loss guided by LLM utility,
thereby directly optimizing the ordering of retrieved documents. Exper-
iments on two standard datasets demonstrate that LURE-RAG achieves
competitive performance, reaching 97–98% of the state-of-the-art dense
neural baseline, while remaining efficient in both training and inference.
Moreover, its dense variant, UR-RAG, significantly outperforms the best
existing baseline by up to 3%.
Keywords:RAG, listwise ranking, lightweight reranker
1 Introduction
Large Language Models (LLMs) have shown strong ability to generate fluent and
often factually grounded text, yet they are limited by their parametric knowl-
edge (which is fixed at training time) [28,4]. They may also hallucinate or fail on
domain-specific or newly emerging information. Retrieval-Augmented Genera-
tion (RAG) has emerged as a prominent framework to address these limitations
by combining external retrieval of relevant documents with language model gen-
eration [38,12]. RAG systems first retrieve documents or passages from some cor-
pus given an input, then augment the LLM’s input with those documents, and
finally generate a response grounded on both retrieved and internal knowledge.
RAG has become a core paradigm in knowledge-intensive NLP tasks [18,14].arXiv:2601.19535v1  [cs.IR]  27 Jan 2026

2 M. Chandra et al.
RAG systems usually consist of multiple modules including at least a retriever
and a generator. Some systems may have other modules to further enhance effec-
tiveness on downstream tasks, like a reranker [13] or a decision maker deciding
when to retrieve [22,11].
In many RAG pipelines [4,12], relevance is defined in the traditional IR sense,
suchaslexicalorsemanticsimilaritybetweendocumentsandthequery.However,
in the RAG setting, semantic relevance to the query alone does not necessarily
translate into utility for the LLM’s downstream generation [40,9,42], that is,
whether the retrieved documents actually help the model produce more accu-
rate, coherent, or useful answers. This discrepancy arises because, even when
documents are individually relevant, concatenating them into a single context
may introduce incoherence or inconsistency, which can in turn mislead the LLM
and degrade the quality of the generated output. Moreover, even if a document
is semantically similar to an input question, it may miss crucial facts required for
the correct answer generation. Recent works [11,38] showed that using utility-
driven signals (e.g. metrics derived from LLM outputs and the ground-truth
answers) as supervision can better align retrieval (and subsequent reranking in
some cases) to what improves generation.
Despite the progress in the direction of utility-driven signals, there remain
important gaps. Firstly, the loss functions used in the retriever or reranker train-
ing do not reflect order or comparative ranking (i.e., a document in rank 1 vs
rank 3 matters). This formulation treats all mis-rankings equally, failing to re-
flect the importance of ranking errors at the top of the list versus those deeper
in the ranking. It has indeed been shown that the order of documents matter
in RAG [31,9]. Secondly, the existing utility-driven methods are not efficient
in terms of training and inference cost. To the best of our knowledge, there is
no existing work focusing on lightweight, efficient reranking methods that still
integrate utility supervision in a RAG setup.
Inthiswork,weproposeaframeworkthattreatstheretrieverandtheLLMas
black-boxes, and adds a lightweight reranker on top of the retrieved candidates,
which is efficient both for training and inference. We train the reranker using
listwise ranking loss derived from supervision given by the LLM utility, i.e., how
much each document/passage contributes to the quality of generation. We call
our proposed frameworkLURE-RAG(Lightweight Utility-driven Reranking
for Efficient RAG).
Our Contributions. Our main contributions are summarized below.
–We propose ablack-box retriever, lightweight reranker and black-box generator
architecture for RAG, called LURE-RAG.
–We propose a listwise ranking-loss based training scheme for the reranker that
uses LLM utility signals (answer correctness) to supervise which retrieved
documents should be ranked higher.
–We demonstrate that LURE-RAG shows a competitive performance, coming
within 97-98% of the strongest dense neural reranker baseline, while being
computationallyefficientwithmodesttrainingandinferenceoverhead,making
it practical for real-world RAG systems.

LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 3
–We demonstrate that the dense neural counterpart of LURE-RAG, which we
call UR-RAG (Utility-driven Reranking for RAG), significantly outperforms
competing baselines on multiple knowledge-intensive tasks.
2 Related Work
In this section, we position LURE-RAG within the literature on RAG, distin-
guishing between those methods that do not explicitly use utility-driven supervi-
sion, and those that do. We then cover learning to rank using listwise or pairwise
ranking losses, since LURE-RAG leverages those approaches.
2.1 Retrieval Augmented Generation
RAG [28] refers broadly to systems which combine a retrieval component (ob-
taining a ranked list of the most likely relevant documents from a collection)
together with a generative (LLM) component that uses those retrieved passages
to output answers. RAG helps address LLM limitations like outdated knowledge,
hallucinations, or inability to access domain-specific or emerging information
[28,12].
Relevance-based RAG. Early approaches to RAG rely primarily on the tra-
ditional IR notion of relevance, rather than explicitly optimizing for downstream
LLM utility [4,12]. In these methods, retrieval is guided by query-document lex-
ical or semantic similarity, often using sparse methods such as BM25 [37] or
dense dual-encoder retrievers [25]. Once retrieved, the top-kpassages are either
directly provided to the generator [28,8], or fused using models such as Fusion-
in-Decoder (FiD) [20].
Supervised Retriever (Utility-driven RAG). These methods incorporate a
downstream LLM output utility (e.g., the correctness of answers in a factoid QA
task) into the training of retriever or reranker. A noticeable example is RePlug
[38], which integrates a dense retriever (e.g., Contriever [21]) into a black-box
LLM pipeline. Unlike classical supervised retrievers trained on relevance labels,
RePlugleveragesLLMutilitysignals,i.e.,howmucharetrievedpassageimproves
the LLM’s likelihood of producing the correct answer. This methodology aligns
the retriever with the needs of the generator rather than with human-annotated
relevance judgments. The training objective in RePlug relies on KL divergence
between the retriever’s predicted distribution over passages and the LLM-utility-
derived distribution.
While RePlug is described as a plug-in approach, in practice, it depends on
dense retrievers such as Contriever. This makes it incompatible with lightweight
or non-parametric retrievers such as BM25, because BM25 does not produce
differentiable representations that can be trained against utility-derived soft la-
bels.Asaresult,RePlugcannotdirectlyoptimizeBM25orothernon-parametric

4 M. Chandra et al.
retrievers, limiting its generality in resource-constrained or latency-sensitive set-
tings. Furthermore, the KL divergence objective in RePlug does not take into
account the utility-driven ranking of the documents, i.e., it treats misclassifica-
tions at the top of the list the same as those at the bottom. In contrast, our
proposed method LURE-RAG makes use of a ranking loss and is compatible
with any retriever since it assumes the retriever to be a black-box.
Another recent work [42] highlights the misalignment between topical rele-
vance and actual utility in RAG, proposing that LLMs can serve as zero-shot
utility judges through a k-sampling listwise approach to mitigate inherent posi-
tion biases. While effective, this method is computationally expensive at infer-
ence time, as it requires multiple LLM calls to shuffle and aggregate document
scores. In contrast, our proposed LURE-RAG framework distills this utility-
driven intelligence into a lightweight LambdaMART-based reranker, providing
an efficient and scalable way to integrate utility signals without requiring multi-
ple LLM calls. We do not adopt this approach as a baseline in our experiments
since its reliance on iterative LLM calls makes it computationally prohibitive at
scale, and its design (direct utility judgment rather than reranking) targets a
different problem formulation than ours.
2.2 Lightweight Learning to Rank
Learning to Rank (LTR) in IR focuses on training models to order documents by
relevance.AmongthemostinfluentialapproachesareRankNet[5],LambdaRank
[6], and LambdaMART [7].
RankNet introduced a neural network–based pairwise ranking model that
learns by minimizing a cross-entropy loss between predicted pairwise prefer-
ences and ground-truth labels. RankNet demonstrated the effectiveness of using
gradient-based optimization for ranking, a key contribution for subsequent neu-
ral ranking models such as [34,2]. LambdaRank improved upon RankNet by
introducing the concept of lambdas: gradient signals that depend not only on
the pairwise preference but also on the impact of swapping two documents on
an evaluation metric such as NDCG (Normalized Discounted Cumulative Gain).
This allowed models to optimize ranking metrics more directly without needing
their gradients explicitly.
LambdaMART combined LambdaRank with MART (Multiple Additive Re-
gression Trees) [10], a boosted tree ensemble method. By applying the lambda-
based gradients to tree-based learners, LambdaMART achieved state-of-the-art
results in large-scale learning-to-rank benchmarks and became widely deployed
in industrial search [7,41] and recommendation systems [17]. LambdaMART is
particularly appealing due to its efficiency, scalability, and interpretability, which
make it suitable as a lightweight reranker in real-world pipelines. In our work,
we build on these strengths by employing LambdaMART within LURE-RAG
as the core reranker, using LLM-derived utility signals to provide supervision.
This allows us to capture task-specific document preferences while retaining the
efficiency and transparency of tree-based models.

LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 5
Document 
CollectionTrain set
RetrieverA training data 
instanceRetrieved documents
The inaugural Nobel Prize 
in Physics was bestowed 
upon German scientist 
Wilhelm …
The Nobel Prizes, 
including the one in 
Physics, are presented at 
an annual ceremony …
.
.
.
In the late 19th century, 
physics was considered 
a preeminent science, 
and the recognition of …Who got the first Nobel 
prize in Physics?
Utility -based 
SortingTrain set 
for 
reranker
Ranked list1Train 
LambdaMARTPrompts 
Constructor
Generator2
Utility 
Computation
34Reranker  Training
Inference
Trained 
lambdaMART
Test 
queryRetrieverPrompts 
Constructork documents
Generator
Fig.1: Schematic diagram of LURE-RAG workflow. 1○For the given query, documents
are retrieved using a (black-box) retriever. 2○Each of these documents is used as a
context for the query and the prompts are constructed. These prompts are fed into
a generator one by one to get the LLM predictions. 3○LLM’s posteriors are used to
compute the utility of each document. The documents are then sorted based on the
utility scores and only top-kof them are retained. These form the training instances
for training a reranker. 4○LambdaMART is trained using thus obtained training data.
3 Proposed Methodology
In this section, we first provide the problem definition and then the details of
our proposed strategy for lightweight utility-driven reranking for efficient RAG
(LURE-RAG). Figure 1 presents an overarching view of the proposed approach.
3.1 Problem Definition
LetQdenote a set of queries (or user questions), where each queryq∈Qis
represented as a sequence of tokensq= (w 1, . . . , w |q|). For a given queryq, a
retrieverRreturns a candidate set ofNdocuments:D q={d 1, . . . , d N}, di∈D,
whereDis the document collection. In a traditional retrieval setting, documents
are ranked according to relevance scores. However, in RAG, what ultimately
matters is theutilityof a document, i.e., how much it improves the LLM’s
ability to generate correct responses.
We formalize the utility-driven reranking problem as follows: given a query
q, a set of candidate documentsD q, and a utility functionU(q, d)estimated via

6 M. Chandra et al.
LLM outputs, learn a ranking function:f θ: (q, d)7→R, such that the induced
ranking maximizes the expected downstream utility. The rerankerf θassigns a
new relevance scores q,d=fθ(q, d)to each documentd∈D q.
3.2 Utility-driven Supervision
Following prior work ([38]), we define document utility by measuring the effect
of includingdin the LLM’s context on the downstream task-specific score (e.g.,
F1 or Exact Match). Formally, ify qis the gold answer andˆy q(d)is the model’s
output when conditioned ond, thenU(q, d) =P(y q,ˆyq(d)), whereP(·)is a task-
specific evaluation metric. The utilitiesU(q, d)serve as supervision signals for
training the reranker.
3.3 Feature Construction
Each query–document pair(q, d)is represented by a feature vectorϕ(q, d)∈Re
such that:ϕ(q, d) = (f 1(q, d), f 2(q, d), . . . , f e(q, d)), whereeis the total number
of features. Specifically, we use lexical and IR-based features as follows.
Query statistics. Since longer queries may provide more discriminative con-
text, we use the total number of query terms and the distinct number of query
terms as features, i.e.,f 1=|q|andf 2=|{w∈q}|. We also use features that
capture how “rare” or “specific” the query terms are, i.e.,f 3= min w∈qIDF(w),
f4= max w∈qIDF(w)andf 5=1
|q|P
w∈qIDF(w).
Document statistics. We also use the corresponding features for documents,
i.e.,f 6=|d|andf 7=|{w∈d}|. We use these features because longer docu-
ments might get unfairly favored because of their higher term frequencies. Sim-
ilarly, we use the informativeness-related feature on the document side, i.e.,
f8= min w∈dIDF(w),f 9= max w∈dIDF(w)andf 10=1
|d|P
w∈dIDF(w).
Query-Documentfeatures. Asameasureoflexicalmatch,weusethenumber
of overlapping terms as a feature, i.e.,f 11=q∩d. Furthermore, as a typically
strong baseline for query-document matching, we use the retrieval score [32], i.e.,
if BM25 is used for retrieval,f 12=BM25(q, d).
Topicfeatures. LatentDirichletAllocation(LDA)[3]providesalower-dimensional
topical representation that captures co-occurrence patterns beyond surface word
overlap. These topic-level signals help identify cases where a query and a can-
didate document aresemanticallyaligned even if they share few exact terms,
making them a complementary feature to lexical statistics. Theϕvector for a
specific topic is the probability distribution over all words in the vocabulary and
theθvector for a specific document is the probability distribution over all topics.
Letθ q∈RMandθ d∈RMbe the topic distributions for queryqand docu-
mentdoverMtopics. We use cosine similarity to capture the semantic similarity

LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 7
beyond words overlap, i.e.,f 13=θq·θd/|θq||θd|. Furthermore, in order to cap-
ture the intuition that if the query’s dominant topics are also well covered in
the document, then the document is likely useful, we use the sum of document
topic probabilities corresponding to the query’s top topics as a feature. Mathe-
matically,f 14=P
m∈Topa(θq)θd(m), where Topa(θq)denotes the topatopics in
the query andθ d(m)denotes topicm’s probability in the documentd.
3.4 Listwise Learning to Rank with LambdaMART
LambdaMART [7] is a listwise learning-to-rank algorithm that combines Lamb-
daRank’s[6]gradientformulationwithgradient-boostedregressiontrees(MART)[10].
For each queryq, we have a candidate document setD qwith corresponding util-
itylabels{U(q, d)|d∈D q}.LambdaMARTtrainsanensembleofregressiontrees
to predict scoress q,dsuch that the induced predicted ranking aligns with the
ground-truth ranking based on the utilities. The objective is to minimize a loss
that directly correlates with ranking quality. In LambdaMART, ifd iis ranked
higher thand jfor the queryq, the gradients are defined as follows:
λij=−σ·1
1 + expσ(sq,di−sq,dj)·∆NDCG ij,(1)
where,s q,diands q,djare predicted scores for documentsd iandd jrespectively,
σ >0is a scaling parameter and∆NDCG ijis the change in NDCG ifd iandd j
were swapped. The reranker updates its parameters using theseλvalues, which
approximate the gradient of NDCG with respect to the predicted scores.
3.5 Inference
At test time, given a query, the first-stage retriever first retrievesNdocu-
ments/passages. The trained lambdaMART model is then used to rank them
and select topk. Thesekdocuments are then used to construct the prompt for
the LLM. We note that while at training time we score each training example
independently, at test time the LLM observes a prompt, i.e., a sequence of ex-
amples. We leave modeling the dependence between different training examples
to future work.
4 Experiment Setup
4.1 Research Questions and Datasets
We investigate the following research questions:
– RQ-1: Does utility-driven reranking lead to a better downstream performance
than traditional relevance-based retrieval across different LLMs and datasets?
– RQ-2: Does training a reranker with listwise ranking loss derived from LLM
utility lead to a better downstream performance?

8 M. Chandra et al.
Table 1: Statistics of the datasets used in our experiments.
Dataset Domain #Train #Dev #Test
NQ-Open Google queries 72,209 8,757 2,889
TriviaQA Trivia questions 78,785 8,837 11,313
– RQ-3:CanalightweightLambdaMART-basedreranker(LURE-RAG)achieve
competitive performance compared to dense neural rerankers (RePlug), while
being more computationally efficient?
– RQ-4: Is the reranker transferrable, i.e., can the reranker trained on one
LLM’s utilities perform well when used with other downstream LLMs of dif-
ferent sizes?
Datasets. We conduct our experiments on two standard open-domain ques-
tion answering (QA) datasets, namely Natural Questions-Open (NQ-Open) and
TriviaQA (see Table 1). More details on each dataset are as follows.
NQ-Open (NQ): The NQ-open dataset [27], a subset of the NQ dataset [26],
differs by removing the restriction of linking answers to specific Wikipedia pas-
sages, thereby mimicking a more general information retrieval scenario similar to
web searches. Following the methodology of [27], our primary source for answer-
ing queries is the English Wikipedia dump as of 20 December 2018. Consistent
with the Dense Passage Retrieval (DPR) approach [25], each Wikipedia article
in this dump is segmented into non-overlapping passages of 100 words.
TriviaQA (TQA): The TriviaQA dataset [23] is a large-scale benchmark con-
sisting of question–answer pairs originally collected from trivia enthusiasts and
quiz-league websites. Each question is accompanied by one or more evidence
documents, either from Wikipedia or from web search results. For open-domain
QA experiments, prior work [25,28,20] commonly discards the original evidence
and evaluates on a standardized English Wikipedia corpus similar to NQ-Open.
The “filtered” version ensures that all retained questions have at least one answer
string present in the passage collection, yielding a cleaner evaluation set. Similar
to [15,19], we perform evaluation on the dev set.
4.2 Methods Investigated
Baselines. We compare LURE-RAG1against the following baselines:
– 0-shot: We use the base models as the baseline, i.e., without providing any
retrieved context in the prompt.
– k-shot: We use the top-kdocuments/passages retrieved by the first-stage
retriever. This baseline helps to check whether the notion of relevance in IR
translates to RAG directly.
– RePlug[38]: The original RePlug framework optimizes a retriever by align-
ing retrieval scores with downstream LLM utility. However, RePlug has two
1Code available athttps://github.com/ManishChandra12/LURE-RAG

LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 9
practical limitations in our setting. First, it is incompatible with fixed or non-
learnable retrievers such as BM25, since its learning relies on updating the
retriever parameters. Second, it is resource-intensive, as the retriever opti-
mization requires periodic index refreshes to incorporate updated document
representations. To enable a fair comparison, we adapt RePlug into a black-
box retriever setting: the retriever is fixed, and only a reranker is trained.
Specifically, we retain RePlug’s KL-divergence–based objective, but apply it
to align the reranker scores with downstream utility signals derived from the
LLM outputs. This modified baseline preserves RePlug’s utility-driven su-
pervision while removing its dependency on the retriever updates and index
refreshes,makingitdirectlycomparabletoourproposedLURE-RAGmethod.
Variants of LURE-RAG. To evaluate the effect of ranking loss in isola-
tion (i.e., without efficiency consideration), we employ the following variants
of LURE-RAG.
– LURE-RAG: We employ Phi-3-mini-4K-instruct [30] as the LLM for obtain-
ing utilities in order to train the LambdaMART reranker.
– UR-RAG: We also introduce a dense variant, UR-RAG, where we replace
LambdaMART with SBERT [36]. SBERT has been widely adopted for seman-
tic retrieval and reranking, and it has been extensively benchmarked [39,16]
in retrieval settings. We fine-tune SBERT using the following loss function
[6,29] to inject the ranking signal into the reranker.
Lrank(q) =X
di,dj∈Dq:U(q,d i)>U(q,d j)1
r(di)−1
r(dj)
log(1 +e(sq,dj−sq,di)),(2)
wherer(·)is the ranking induced byU(·). IfU(q, d i)> U(q, d j),diis ranked
higher thand jand therefore,r(d i)< r(d j). Whend ihas a much higher rank
thand j, e.g,r(d i) = 1andr(d j) = 10, the weight component of the loss will
be high and the loss function will strongly draws(q, d i)up froms(q, d j). This
variant provides insights into the effectiveness-efficiency trade-off.
4.3 Evaluation Metrics and Hyper-parameter Configurations
Accuracy.Datasets like NQ-Open and TriviaQA provide a range of potential
answers for each query. Frequently, these answers are different variants of the
same concept (e.g., “President D. Roosevelt” or “President Roosevelt”). To eval-
uate the LLM accuracy on these datasets, which accept a range of valid answers
for each query, we employ a binary assessment consistent with [24,9]. A response
is deemed accurate if it contains at least one of the predefined correct answers;
otherwise, it is inaccurate.
F1.F1-score is useful when answers involve partial overlap (e.g. multi-token
answers), or when one wants to reward partial correctness or penalize missing
parts of the answer. For a queryq, precision and recall are defined as:P q=
|yq∩ˆyq|/|ˆyq|andR q=|y q∩ˆyq|/|yq|respectively. We then define the F1 score

10 M. Chandra et al.
Table 2: A comparison between our proposed LURE-RAG approach and the baselines
(usingBM25forretrieval)onthetwodatasetsintermsofAccuracy(Acc)andF1-score.
The best Acc and F1 values are bold-faced for each dataset and LLM combinations.
llama-1B phi-3.8B Qwen-14B
Dataset Rank-By Method Acc F1 Acc F1 Acc F1
NQRelevance0-shot .1215 .0616 .1970 .1026 .2323 .2518
k-shot .2876 .1590 .2676 .1866 .3548 .3498
UtilityRePlug .2932 .1894 .2895 .2007 .3775 .3700
LURE-RAG .2891 .1868 .2822 .1963 .3712 .3655
LURE-RAG (w/o topics) .2863 .1829 .2810 .1961 .3700 .3627
UR-RAG.2999 .2068 .2933 .2088 .3822 .3854
TQARelevance0-shot .2258 .2137 .2779 .2214 .3566 .3587
k-shot .2841 .2600 .3486 .3177 .3976 .3978
UtilityRePlug .3027 .3017 .3603 .3321 .4112 .4125
LURE-RAG .2968 .3010 .3569 .3272 .4057 .4047
LURE-RAG (w/o topics) .2947 .2973 .3544 .3261 .4008 .4001
UR-RAG.3111 .3104 .3676 .3472 .4186 .4239
as:F1 = (1/|Q|)P
q∈Qmax yq∈Cq(2PqRq/(Pq+Rq)), whereC qdenotes the set
of all available ground-truth answers for the queryq.
LURE-RAG relies on the LLM predictions for the reranker training. To anal-
yse the variations that may be caused due to this choice of LLM, we conduct
our experiments on three different LLMs - each from different families. The ob-
jective is to analyse the variations in LURE-RAG’s performance corresponding
to different sizes of the models, and variations in characteristics of models across
different families, thereby allowing to answer RQ-4. In particular, we use Meta’s
Llama-3.2-1B-Instruct [33], Microsoft’s Phi-3-mini-3.8B-4K-instruct [1] and Al-
ibaba’s Qwen2.5-14B-Instruct-1M [35]. Furthermore, we conduct two sets of ex-
periments - one involving BM25 [37] for sparse retrieval and the other involving
Contriever [21] for dense retrieval.
In our experiments, we set the number of first-stage retrieved documents
using BM25/Contriever (N) to 10 and RAG context size (k) to 5. As a reranker
for RePlug baseline and UR-RAG, we finetune SBERT all-MiniLM-L6-v2 [36].
For fine-tuning SBERT, we use a training batch size of 32, a learning rate of 1e-5
and early stopping with a patience of 3 epochs. As an optimiser, we use Adam
with weight decay. The optimal values of the hyperparameters are obtained via a
grid search. For LDA, we use a random subset of 1M documents from the corpus
and set the number of topicsMto 100 and the threshold for ‘top’ topics (a) to
20.

LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 11
Table 3: A comparison between our proposed LURE-RAG approach and the baselines
(using Contriever for retrieval) on the two datasets in terms of Accuracy (Acc) and
F1-score. The best Acc and F1 values are bold-faced for each dataset and LLM com-
binations.
llama-1B phi-3.8B Qwen-14B
Dataset Rank-By Method Acc F1 Acc F1 Acc F1
NQRelevance0-shot .1215 .0616 .1970 .1026 .2323 .2518
k-shot .2911 .1782 .2631 .1835 .3508 .3401
UtilityRePlug .2976 .1853 .2799 .1994 .3751 .3696
LURE-RAG .2945 .1811 .2722 .1949 .3687 .3669
LURE-RAG (w/o topics) .2940 .1798 .2720 .1904 .3666 .3664
UR-RAG.3027 .1915 .2883 .2076 .3811 .3800
TQARelevance0-shot .2258 .2137 .2779 .2214 .3566 .3587
k-shot .2834 .2620 .3509 .3137 .3966 .3954
UtilityRePlug .3111 .3042 .3582 .3400 .4010 .4077
LURE-RAG .3059 .2951 .3635 .3304 .3994 .4039
LURE-RAG (w/o topics) .2994 .2897 .3628 .3220 .4010 .4017
UR-RAG.3184 .3098 .3699 .3497 .4177 .4203
5 Results
Tables 2 and 3 report the performance of our proposed method relative to the
baselines on two open-domain QA benchmarks, Natural Questions (NQ) and
TriviaQA (TQA), across three different LLMs of increasing model sizes — llama-
1B, phi-3.8B, and Qwen-14B.
Comparison between relevance-based and utility-driven paradigms.
Across the dataset and LLM choices, we observe that utility-driven approaches
consistently outperform the relevance-based ones. While k-shot prompting im-
proves over 0-shot in the relevance-based setting (e.g., +0.16 Acc improve-
ment for llama-1B on NQ using sparse retrieval), the gap between k-shot and
utility-driven reranking remains substantial. This confirms that document or-
dering based solely on classical IR relevance is insufficient for optimal down-
stream performance, and utility-driven reranking better aligns retrieval with the
LLM’s actual answer quality. Therefore, in answer toRQ-1, we conclude that
utility-drivenrerankingleadstobetterdownstreamperformancethantraditional
relevance-based retrieval across different LLMs and datasets.
Effect of listwise ranking loss. While RePlug remains a strong utility-driven
baseline,weobservethatUR-RAGsignificantlyoutperformsitacrossallsettings.
Forexample,onNQwithllama-1Busingsparseretrieval,RePlugachieves0.1894
F1, whereas UR-RAG achieves 0.2068, yielding noticeable improvements. Paired
t-tests at 5% significance level reveal that the improvements in the evaluation

12 M. Chandra et al.
metrics obtained using UR-RAG are statistically significant across all dataset
and LLM combinations (tests are conducted separately) compared to Replug. A
key reason for this gain lies in the ranking loss employed by UR-RAG, which
explicitly encourages the reranker to preserve correct document orderings ac-
cording to the downstream utility. In contrast, RePlug does not incorporate
any ranking-aware objective, relying solely on the KL divergence between the
reranker scores and utility distributions. This difference in performance is ob-
served because order of documents matter in RAG [9]. Therefore, in answer to
RQ-2, we conclude that training a reranker with listwise ranking loss derived
from LLM utility leads to better downstream performance.
Comparisonbetweendenseandlight-weightreranker. LURE-RAGachieves
results that are competitive with the dense reranker baseline (RePlug), while re-
quiring only a lightweight LambdaMART model for training and inference. The
F1-scores and accuracies for the NQ dataset using LURE-RAG (in sparse re-
trieval setup) are always above97.8%and97.4%, respectively, of that achieved
using RePlug. Similar numbers for TQA dataset are98.1%and98.05%, respec-
tively. These results indicate that a lightweight reranker, when guided bylist-
wise rankingutility-driven supervisionprovidesa muchmore efficientalternative
with only minor effectiveness trade-offs compared to a dense neural reranker not
trained with a ranking loss. Therefore, in answer toRQ-3, we conclude that a
lightweight LambdaMART-based reranker can achieve competitive performance
compared to dense neural rerankers that are trained without ranking loss, while
being more computationally efficient.
Along the expected lines, UR-RAG, which fine-tunes SBERT with a utility-
driven ranking loss, yields the highest accuracy and F1 scores across all dataset-
model combinations. For instance, on TQA with Qwen-14B (in sparse retrieval
setup),UR-RAGachieves0.4186accuracyand0.4239F1,thestrongestresultsin
Table 2. The F1-scores and accuracies for the NQ dataset using LURE-RAG are
always above90.3%and96.2%, respectively, of that achieved using UR-RAG.
Similar numbers for TQA dataset are94.2%and95.4%, respectively. These
results indicate that when a lightweight reranker and a dense neural reranker,
both trained using a ranking loss, are compared, the performance deterioration
is larger.
Furthermore, we also see improvements across different LLMs. Note that we
train the reranker using the utilities obtained from the phi-3 model, and reuse
the same trained reranker for other LLMs as well. Therefore, in answer toRQ-4,
we conclude that the reranker trained on one LLM’s utilities can perform well
when used with other downstream LLMs of different sizes.
Interpretability. ThebenefitofusingalightweightmodellikelambdaMARTis
thatit’sinterpretable.Tounderstandwhichsignalsdrivethereranker’sdecisions,
weanalysethefeatureimportancescoresfromthetrainedLambdaMARTmodels
(see Figure 2). Feature importances are computed using the gain metric, which
measures the relative contribution of each feature to the model’s decision splits.
For every split in the trees, the “gain” (improvement in the objective function)

LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 13
(a) NQ
 (b) TQA
Fig.2: Feature importance scores of the top-8 features obtained from the trained lamb-
daMART model for LURE-RAG.
is calculated. The total gain for a feature is summed across all splits where it
is used, providing a measure of its importance. The most influential feature is
the BM25 similarity score (bm25_score), which dominates the learned model.
This suggests that classical lexical similarity remains a strong signal for utility-
driven reranking. Beyond this, the number of (unique) document terms and
document-level IDF statistics also exhibit notable importance, indicating that
documents containing rare, high-value terms often align better with downstream
LLM utility. Moreover, query-document term overlap and features derived from
LDA provide additional, though relatively weaker, signals.
Ablation study. To examine the contribution of the LDA topic features, we
conduct an ablation study by removing them from the feature set. As expected
from the feature importance analysis (Figure 2), topic features exhibit relatively
low importance compared to lexical and BM25-based features. Consistently, we
observe only marginal degradation in performance across LLMs when these fea-
tures are excluded, as shown in Tables 2 and 3 corresponding to ‘LURE-RAG
(w/o topics)’ rows. This result suggests that while topic-level information can
provide additional semantic signal, the reranker primarily relies on stronger lex-
ical and retrieval-based signals. Nonetheless, the inclusion of LDA features does
offer small but consistent improvements.
6 Conclusions
In this work, we proposed LURE-RAG, a lightweight utility-driven reranking
framework that improves retrieval-augmented generation by incorporating a
ranking-aware objective, offering a more efficient alternative to existing ap-
proaches. We further introduced UR-RAG, a dense variant that achieves the
strongest results, significantly improving accuracy and F1 by up to 3%. Our
results demonstrate that utility-driven reranking using principled ranking losses
providesapowerfulandpracticalapproachtoaligningretrievalwithdownstream
generation tasks in retrieval-augmented generation. In future work, we aim to
explore richer feature representations including semantic signals.

14 M. Chandra et al.
Disclosure of Interests.The authors have no competing interests to declare that
are relevant to the content of this article.
References
1. Abdin, M.I., Ade Jacobs, S., Awan, A.A., Aneja, J., Awadallah, A., Has-
san Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A., Behl, H., Benhaim, A.,
Bilenko, M., Bjorck, J., Bubeck, S., Cai, M., Mendes, C.C.T., Chen, W., Chaud-
hary, V., Chopra, P., Giorno, A.D., de Rosa, G., Dixon, M., Eldan, R., Iter, D.,
Goswami, A., Gunasekar, S., Haider, E., Hao, J., Hewett, R.J., Huynh, J., Java-
heripi, M., Jin, X., Kauffmann, P., Karampatziakis, N., Kim, D., Khademi, M.,
Kurilenko, L., Lee, J.R., Lee, Y.T., Li, Y., Liang, C., Liu, W., Lin, X.E., Lin, Z.,
Madan, P., Mitra, A., Modi, H., Nguyen, A., Norick, B., Patra, B., Perez-Becker,
D., Portet, T., Pryzant, R., Qin, H., Radmilac, M., Rosset, C., Roy, S., Saarikivi,
O., Saied, A., Salim, A., Santacroce, M., Shah, S., Shang, N., Sharma, H., Song,
X., Ruwase, O., Wang, X., Ward, R., Wang, G., Witte, P., Wyatt, M., Xu, C., Xu,
J., Xu, W., Yadav, S., Yang, F., Yang, Z., Yu, D., Zhang, C., Zhang, C., Zhang, J.,
Zhang, L.L., Zhang, Y., Zhang, Y., Zhou, X.: Phi-3 technical report: A highly capa-
ble language model locally on your phone. Tech. Rep. MSR-TR-2024-12, Microsoft
(August 2024),https://www.microsoft.com/en-us/research/publication/
phi-3-technical-report-a-highly-capable-language-model-locally-on-your-phone/
2. Ai, Q., Bi, K., Guo, J., Croft, W.B.: Learning a deep listwise context model for
rankingrefinement.In:The41stInternationalACMSIGIRConferenceonResearch
& Development in Information Retrieval. pp. 135–144 (2018)
3. Blei, D.M., Ng, A.Y., Jordan, M.I.: Latent dirichlet allocation. Journal of Machine
Learning Research (JMLR)3, 993–1022 (2003).https://doi.org/10.1162/jmlr.
2003.3.4-5.993
4. Brown, A., Roman, M., Devereux, B.: A systematic literature review of retrieval-
augmented generation: Techniques, metrics, and challenges. Applied Sciences9(12)
(2025).https://doi.org/10.48550/arxiv.2508.06401
5. Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., Hul-
lender, G.: Learning to rank using gradient descent. In: Proceedings of the 22nd
InternationalConferenceonMachineLearning.p.89–96.ICML’05,Associationfor
Computing Machinery, New York, NY, USA (2005).https://doi.org/10.1145/
1102351.1102363,https://doi.org/10.1145/1102351.1102363
6. Burges, C.J.C., Ragno, R., Le, Q.V.: Learning to rank with nonsmooth cost func-
tions. In: Proceedings of the 20th International Conference on Neural Information
ProcessingSystems.p.193–200.NIPS’06,MITPress,Cambridge,MA,USA(2006)
7. Burges, C.J.: From ranknet to lambdarank to lambdamart:
An overview. Tech. Rep. MSR-TR-2010-82, Microsoft Research
(2010),https://www.microsoft.com/en-us/research/publication/
from-ranknet-to-lambdarank-to-lambdamart-an-overview/
8. Chandra, M., Ganguly, D., Ounis, I.: One size doesn’t fit all: Predicting the num-
ber of examples for in-context learning. In: Advances in Information Retrieval:
47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy,
April 6–10, 2025, Proceedings, Part I. p. 67–84. Springer-Verlag, Berlin, Heidelberg
(2025).https://doi.org/10.1007/978-3-031-88708-6_5,https://doi.org/10.
1007/978-3-031-88708-6_5

LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 15
9. Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Campagnano, C., Maarek,
Y., Tonellotto, N., Silvestri, F.: The power of noise: Redefining retrieval for rag
systems. In: Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval. p. 719–729. SIGIR ’24, Asso-
ciation for Computing Machinery, New York, NY, USA (2024).https://doi.org/
10.1145/3626772.3657834,https://doi.org/10.1145/3626772.3657834
10. Friedman, J.H.: Greedy function approximation: A gradient boosting machine.
Annals of Statistics29(5), 1189–1232 (2001).https://doi.org/10.1214/aos/
1013203451
11. Gao, J., Li, L., Ji, K., Li, W., Lian, Y., yuzhuo fu, Dai, B.: SmartRAG: Jointly
learn RAG-related tasks from the environment feedback. In: The Thirteenth In-
ternational Conference on Learning Representations (2025),https://openreview.
net/forum?id=OCd3cffulp
12. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M.,
Wang, H.: Retrieval-augmented generation for large language models: A survey
(2024),https://arxiv.org/abs/2312.10997
13. Glass, M., Rossiello, G., Chowdhury, M.F.M., Naik, A., Cai, P., Gliozzo, A.:
Re2G: Retrieve, rerank, generate. In: Carpuat, M., de Marneffe, M.C., Meza Ruiz,
I.V. (eds.) Proceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technolo-
gies. pp. 2701–2715. Association for Computational Linguistics, Seattle, United
States (Jul 2022).https://doi.org/10.18653/v1/2022.naacl-main.194,https:
//aclanthology.org/2022.naacl-main.194/
14. Gupta, S., Ranjan, R., Singh, S.N.: A comprehensive survey of retrieval-augmented
generation(rag):Evolution,currentlandscapeandfuturedirections(2024),https:
//arxiv.org/abs/2410.12837
15. Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.W.: Retrieval-augmented
language model pre-training. In: International Conference on Machine Learning
(ICML). pp. 3929–3938. PMLR (2020)
16. Hofstätter, S., Althammer, S., Khattab, O., Mitra, B., Hanbury, A.: Efficiently
teaching an effective dense retriever with balanced topic aware sampling. In: Pro-
ceedings of the 44th International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval. pp. 113–122. ACM (2021)
17. Hu, Z., Wang, Y., Peng, Q., Li, H.: Unbiased lambdamart: An unbiased pairwise
learning-to-rank algorithm. In: Proceedings of The World Wide Web Conference
(WWW ’19). pp. 2271–2281 (2019).https://doi.org/10.1145/3308558.3313768
18. Huang, Y., Huang, J.: A survey on retrieval-augmented text generation for large
language models (2024),https://arxiv.org/abs/2404.10981
19. Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., Grave,
E.: Atlas: Few-shot learning with retrieval augmented language models. In: In-
ternational Conference on Machine Learning (ICML). pp. 13182–13210. PMLR
(2023)
20. Izacard, G., Grave, E.: Leveraging passage retrieval with generative models for
open domain question answering. In: International Conference on Learning Repre-
sentations (ICLR) (2021),https://arxiv.org/abs/2007.01282
21. Izacard, G., Lewis, P., Riedel, S., Karpukhin, V., Minervini, P., Petroni, F., Grave,
E.: Unsupervised dense information retrieval with contrastive learning. Transac-
tions of the Association for Computational Linguistics10, 665–681 (2022)
22. Jeong, S., Baek, J., Cho, S., Hwang, S.J., Park, J.: Adaptive-RAG: Learning
to adapt retrieval-augmented large language models through question complex-

16 M. Chandra et al.
ity. In: Duh, K., Gomez, H., Bethard, S. (eds.) Proceedings of the 2024 Con-
ference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 7035–
7049. Association for Computational Linguistics, Mexico City, Mexico (Jun 2024).
https://doi.org/10.18653/v1/2024.naacl-long.389,https://aclanthology.
org/2024.naacl-long.389
23. Joshi, M., Choi, E., Weld, D., Zettlemoyer, L.: Triviaqa: A large scale distantly
supervisedchallengedatasetforreadingcomprehension.In:Proceedingsofthe55th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers). pp. 1601–1611. Association for Computational Linguistics (2017).https:
//doi.org/10.18653/v1/P17-1147,https://aclanthology.org/P17-1147
24. Kandpal, N., Deng, H., Roberts, A., Wallace, E., Raffel, C.: Large language models
struggle to learn long-tail knowledge. In: Krause, A., Vreeken, J., Yoav, S. (eds.)
Proceedings of the 40th International Conference on Machine Learning. vol. 202,
pp. 15886–15904. PMLR (2023)
25. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih,
W.t.: Dense passage retrieval for open-domain question answering. In: Proceed-
ings of the 2020 Conference on Empirical Methods in Natural Language Process-
ing (EMNLP). pp. 6769–6781. Association for Computational Linguistics (2020).
https://doi.org/10.18653/v1/2020.emnlp-main.550,https://aclanthology.
org/2020.emnlp-main.550
26. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C.,
Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey,
M., Chang, M.W., Dai, A.M., Uszkoreit, J., Le, Q., Petrov, S.: Natural questions:
A benchmark for question answering research. Transactions of the Association for
Computational Linguistics7, 453–466 (2019)
27. Lee, K., Chang, M.W., Toutanova, K.: Latent retrieval for weakly supervised
open domain question answering. In: Proceedings of the 57th Annual Meeting
of the Association for Computational Linguistics. pp. 6086–6096. Association
for Computational Linguistics (2019).https://doi.org/10.18653/v1/P19-1612,
https://aclanthology.org/P19-1612
28. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive nlp tasks. In: Proceedings of the 34th Interna-
tional Conference on Neural Information Processing Systems. NIPS ’20, Curran
Associates Inc., Red Hook, NY, USA (2020)
29. Li, X., Lv, K., Yan, H., Lin, T., Zhu, W., Ni, Y., Xie, G., Wang, X., Qiu, X.:
Unified demonstration retriever for in-context learning. In: Rogers, A., Boyd-
Graber, J., Okazaki, N. (eds.) Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). pp. 4644–
4668. Association for Computational Linguistics, Toronto, Canada (Jul 2023),
https://aclanthology.org/2023.acl-long.256
30. Li, Y., Bubeck, S., Eldan, R., Del Giorno, A., Gunasekar, S., Lee, Y.T.: Text-
booksareallyouneedii:phi-1.5technicalreport.arXivpreprintarXiv:2309.05463
(2023)
31. Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang,
P.: Lost in the middle: How language models use long contexts. Transactions of
the Association for Computational Linguistics12, 157–173 (2024).https://doi.
org/10.1162/tacl_a_00638,https://aclanthology.org/2024.tacl-1.9/

LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 17
32. Liu,T.Y.:Learningtorankforinformationretrieval.FoundationsandTrendsinIn-
formation Retrieval3(3), 225–331 (2009).https://doi.org/10.1561/1500000016
33. Meta AI: Llama 3.2: Revolutionizing edge ai and vision with open, customizable
models. Tech. rep., Meta Platforms, Inc. (Sep 2024),https://ai.meta.com/blog/
llama-3-2-connect-2024-vision-edge-mobile-devices/, accessed: 2025-05-22
34. Pasumarthi, R.K., Bruch, S., Wang, X., Bendersky, M., Najork, M., Wang, J.,
Li, Y., Anil, R., Cheng, H., Chappidi, R., et al.: Tf-ranking: Scalable tensorflow
library for learning-to-rank. In: Proceedings of the 25th ACM SIGKDD Interna-
tional Conference on Knowledge Discovery & Data Mining. pp. 2970–2978 (2019)
35. Qwen Team: Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 (2024),
submitted December 2024
36. Reimers, N., Gurevych, I.: Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. In: Inui, K., Jiang, J., Ng, V., Wan, X. (eds.) Proceedings
of the 2019 Conference on Empirical Methods in Natural Language Process-
ing and the 9th International Joint Conference on Natural Language Process-
ing (EMNLP-IJCNLP). pp. 3982–3992. Association for Computational Linguis-
tics, Hong Kong, China (Nov 2019).https://doi.org/10.18653/v1/D19-1410,
https://aclanthology.org/D19-1410
37. Robertson, S.E., Zaragoza, H.: The probabilistic relevance framework: Bm25 and
beyond. Foundations and Trends in Information Retrieval3(4), 333–389 (2009).
https://doi.org/10.1561/1500000019
38. Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer,
L., Yih, W.t.: REPLUG: Retrieval-augmented black-box language models. In:
Duh, K., Gomez, H., Bethard, S. (eds.) Proceedings of the 2024 Conference
of the North American Chapter of the Association for Computational Linguis-
tics: Human Language Technologies (Volume 1: Long Papers). pp. 8371–8384.
Association for Computational Linguistics, Mexico City, Mexico (Jun 2024).
https://doi.org/10.18653/v1/2024.naacl-long.463,https://aclanthology.
org/2024.naacl-long.463/
39. Thakur, N., Reimers, N., Daxenberger, J., Gurevych, I.: BEIR: A heterogeneous
benchmarkforzero-shotevaluationofinformationretrievalmodels.In:Proceedings
of the 43rd International ACM SIGIR Conference on Research and Development
in Information Retrieval. pp. 2308–2318. ACM (2021)
40. Tian, F., Ganguly, D., Macdonald, C.: Is relevance propagated from retriever to
generatorinrag?In:AdvancesinInformationRetrieval:47thEuropeanConference
on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings,
Part I. p. 32–48. Springer-Verlag, Berlin, Heidelberg (2025).https://doi.org/10.
1007/978-3-031-88708-6_3,https://doi.org/10.1007/978-3-031-88708-6_3
41. Wu, Q., Burges, C.J.C., Svore, K.M., Gao, J.: Adapting boosting for information
retrieval measures. Information Retrieval13(3), 254–270 (2010).https://doi.
org/10.1007/s10791-009-9112-1
42. Zhang, H., Zhang, R., Guo, J., de Rijke, M., Fan, Y., Cheng, X.: Are large
language models good at utility judgments? In: Proceedings of the 47th In-
ternational ACM SIGIR Conference on Research and Development in Informa-
tion Retrieval. p. 1941–1951. SIGIR ’24, Association for Computing Machin-
ery, New York, NY, USA (2024).https://doi.org/10.1145/3626772.3657784,
https://doi.org/10.1145/3626772.3657784