# Online-Optimized RAG for Tool Use and Function Calling

**Authors**: Yu Pan, Xiaocheng Li, Hanzhao Wang

**Published**: 2025-09-24 09:08:46

**PDF URL**: [http://arxiv.org/pdf/2509.20415v1](http://arxiv.org/pdf/2509.20415v1)

## Abstract
In many applications, retrieval-augmented generation (RAG) drives tool use
and function calling by embedding the (user) queries and matching them to
pre-specified tool/function descriptions. In this paper, we address an
embedding misalignment issue that often arises in practical applications due to
imperfect embedding models or noisy descriptions; such misalignment may lead to
incorrect retrieval and task failure. We introduce Online-Optimized RAG, a
deployment-time framework that continually adapts retrieval embeddings from
live interactions using minimal feedback (e.g., task success). Online-Optimized
RAG applies lightweight online gradient updates with negligible per-query
latency and requires no changes to the underlying LLM. The method is
plug-and-play: it supports both single- and multi-hop tool use, dynamic tool
inventories, and $K$-retrieval with re-ranking. We provide a problem-dependent
theoretical analysis that quantifies how the method's performance depends on
the initialization quality of the embeddings and other related quantities.
Across diverse tool-use and document-retrieval scenarios, our Online-Optimized
RAG consistently improves tool selection accuracy and end-task success, thus
providing a simple, practical path to robust, self-improving RAG systems.

## Full Text


<!-- PDF content starts -->

Online-Optimized RAG for Tool Use and Function Calling
Yu Pan†, Xiaocheng Li‡, Hanzhao Wang†
†The University of Sydney Business School, The University of Sydney,
‡Imperial College Business School, Imperial College London
Abstract
In many applications, retrieval-augmented generation (RAG) drives tool use and function calling
by embedding the (user) queries and matching them to pre-specified tool/function descriptions. In
this paper, we address an embedding misalignment issue that often arises in practical applications
due to imperfect embedding models or noisy descriptions; such misalignment may lead to incorrect
retrieval and task failure. We introduce Online-Optimized RAG, a deployment-time framework that
continually adapts retrieval embeddings from live interactions using minimal feedback (e.g., task
success). Online-Optimized RAG applies lightweight online gradient updates with negligible per-
query latency and requires no changes to the underlying LLM. The method is plug-and-play: it
supports both single- and multi-hop tool use, dynamic tool inventories, andK-retrieval with re-
ranking. We provide a problem-dependent theoretical analysis that quantifies how the method’s
performance depends on the initialization quality of the embeddings and other related quantities.
Across diverse tool-use and document-retrieval scenarios, our Online-Optimized RAG consistently
improves tool selection accuracy and end-task success, thus providing a simple, practical path to
robust, self-improving RAG systems.
1 Introduction
Modernlargelanguagemodels(LLMs)increasinglyrelyonretrieval-augmentedgeneration(RAG)(Lewis
et al., 2020) to ground responses in external data. In tool-use settings, an agent encodes the user task,
retrieves a tool or function (e.g., an API), and executes it: a query is embedded into a vector space
and matched against a catalog of tool/function descriptions that are likewise embedded; the retriever
proposes candidates by similarity (often top-k), and an executor (e.g., a function-calling API or tool
wrapper) carries out the selected call (Patil et al., 2024; Qin et al., 2023; Lumer et al., 2024).
However, when the system cannot incorporate domain feedback, RAG can still yield incorrect calls
and answers. Retrieval quality degrades whenever the (trained) embedding geometry drifts from the
operational environment. For example, such misalignments can arise from (i) noisy or incomplete tool
documentation, (ii) outdated or suboptimal embedding models, (iii) shifts in user intent or phrasing
relative to training or others. In such cases, semantically related tools may be mapped far apart (or vice
versa), causing the retriever to surface the wrong candidate; the downstream LLM is then bottlenecked
by what it is given, leading to unnecessary backtracking or failed tasks. Figure 1 shows two examples
of the degradation of retrieval performance caused by bad documentation and a poor embedding model.
Existing deployments typically freeze embeddings and indices after offline training (Zeighami et al.,
2024; Qin et al., 2023; Patil et al., 2024; Li et al., 2023), leaving no principled, low-cost way to repair
performance at deployment. While recent work adaptscontrollersat inference time to decide when or
how much to retrieve (Asai et al., 2024; Jeong et al., 2024) or tunes top-kand retrieval strategies (e.g., no
Correspondence to yu.pan@sydney.edu.au, hanzhao.wang@sydney.edu.au
1arXiv:2509.20415v1  [cs.SE]  24 Sep 2025

Good Documentation
"This model predicts the punctuation
of English, Italian, French and
German texts. It was developed to
restore transcribed..."
Bad Documentation
"punctuation prediction"0.4294
0.2664
0.4201Recall@10
Recall@10 (r aw)
Recall@10 (optimiz ed)(a) Good documentation vs. bad documentation
 (b) Larger model vs. smaller model
Figure 1: We measure retrieval with Recall@10, where larger values indicate better performance. We
defineoptimizedas the performance after applying Online-Optimized RAG andrawas the performance
before optimization. (a) Poor documentation weakens semantic alignment and lowers retrieval quality,
while Online-Optimized RAG mitigates this mismatch. (b) t-SNE visualization of the same samples
for two embedding models before and after optimization. The larger modeltext-embeded-3-large
generallyoutperformstext-embeded-v4. Thus, applying asmall modelcanyield low rawperformance in
practice. However, the embeddings after optimization from both models move toward similar regions and
achieve comparable performance, demonstrating our approach’s effectiveness. We refer more numerical
experiments to Section 4 and the experimental setup for generating these two subfigures is provided in
Appendix C.
retrieval / one-shot / multi-step) via multi-armed-bandit or reinforcement-learning approaches (Fu et al.,
2024; Tang et al., 2024; Sun et al., 2025), these methods only adjust global hyperparameters but do not
update the underlying embedding space. We defer further discussion of related literature to Appendix A.
We introduceOnline-Optimized RAG, a deployment-time framework that continuously updates re-
trieval embeddings from online interactions for tool use and function calling. The core idea is simple:
treat the tool retriever as an object to be optimized at test time using minimal observable feedback (e.g.,
whether the task is solved). After each interaction, we apply lightweight online gradient updates to the
item (tool) embeddings to improve future retrieval accuracy without modifying the underlying LLM,
planner, or executors. The procedure is plug-and-play, adds negligible latency, requires no privileged
access to model internals, and also applies beyond tools to general document retrieval.
Our contributions are summarized as follows:
•Problem formulation for online retrieval.We cast the problem of RAG tool and function
selection under an online learning framework with bandit-style execution feedback (success/failure
signals only for the chosen tool), updating the retrieval geometry on the fly after collecting each
feedback.
•A simple, scalable update rule.We propose an online gradient descent variant that adjusts
embeddingsperinteractionusinganimportance-weightedestimator. Theupdateoftheembeddings
keeps computation overhead minimal for large catalogs and high-throughput systems; it is both
intuitive and theoretically supported.
•Versatility across real retrieval settings.The same update mechanism applies to tooland
document retrieval, single- and multi-hop pipelines, dynamic tool inventories, and multiple re-
trievals with reranking. This enables a straightforward integration with common function-calling
frameworks and LLM agents without altering the LLM.
•Principled adaptation guarantees.We derive a problem-dependent performance analysis clar-
ifying how performance depends on the quality of the initial embeddings: strong initializations
2

accelerate convergence toward the optimum, while weaker ones still improve steadily under online
updates.
•Empirical evidence on comprehensive tasks.Across diverse scenarios, our Online-Optimized
RAG consistently improves tool selection and downstream task success, and the performance also
transfers to general document retrieval. This demonstrates our method as a simple and robust
path to self-improving RAG.
Online
Optimiz e Query
ArrivalEmbedded
Query Retriev erEmbedding
ModelChoose 
Tool 
Feedback:
Success/F ailTool Embeddings
Update to
 Success
FailQuery emb . 
Selected tool emb . 
Updated  emb. 
Unselected tool  emb. 
Updated  emb. 
Selected tool update 
Selected tool  update 
Unselected tool updateOnline Optimize 
(a) Online-Optimized RAG
Figure 2: Online-Optimized RAG updates tool embeddings at deployment time for each incoming query
qt. When the selected tooli tsucceeds, its embedding moves toward the query to increase similarity.
When it fails, the embedding ofi tmoves away to reduce similarity. Embeddings of all unselected tools
are also pushed away from the query. See Algorithm 1 and its discussion for details.
2 Problem Setup
2.1 RAG for tool use and function calling
In this section, we describe the problem of RAG and present its mathematical setup. The setup can be
viewed as a simplified version of the most general RAG systems. The aim here is to generate intuitions
and to give us a rigorous language to describe our algorithm, and we will discuss several extensions that
cover a broad range of different RAG application contexts.
Specifically, for a RAG system, an embedding model such as OpenAI or Gemini embeddings API,
maps an input query (e.g., a prompted question or task for the users) to a query embeddingq∈Rd. On
the other hand, a database or a tool pool containsIcandidate items (e.g., documents for answering the
question, functions in MCP, or tools for solving the task). The goal of RAG is to retrieve a proper tool
from theIitems that best fits the queryq. For the basic setup, we consider the case that for each query
q, there exists an (unobserved) optimal itemi∗∈ {1, . . . , I}that best answers the query or solves the
task. The case is motivated by that in most tool-use and function-calling applications, the optimal tool
or function is usually unique. However, our online-optimized framework and the algorithm also apply
to more general setups of K-retrievals (with rerankers), time-varying database, and multi-hop retrievals
which we defer to Section 3.1.
Next, givenq, the RAG system produces a distributionp= (p 1, . . . , p I), wherep iis the probability
of selecting itemi. The cosine similarity-based RAG (most commonly used) represents each itemiwith
3

an embedding vectorθ i∈Rd,
Θ= [θ 1, . . . ,θ I]⊤∈RI×d.
Then each itemiis scored by the softmax of the inner product
pi(q,Θ) =exp(q⊤θi)PI
i′=1exp(q⊤θi′).(1)
In this light, the retrieval problem can also be viewed as a multiclass classification with inputqand label
i∗under a softmax classifier parameterized byΘ. The loss function is
l(Θ; (q, i∗)) =−logp i∗(q,Θ).
2.2 Online-optimized framework for RAG
We now present an online-learning setting for the RAG problem where at each timet= 1, . . . , T, a query
arrives represented by the embeddingq t. Importantly, we allow changing embeddingsΘ t∈RI×dindexed
by timet. The benefit is that this admits an imperfect initial embeddingΘ 1and allows embeddings to
be better learned and improved over time.
In hindsight of seeing all the data{(q t, i∗
t)}T
t=1, the optimal embedding should be
Θ∗= arg min
ΘTX
t=1l(Θ; (q t, i∗
t)).(2)
In the online setting as how these RAG systems are usually deployed in practice, at each timet, we can
choose the embeddingsΘ tbased on the past observation historyH t={q s, is,1{i s=i∗
s}}t−1
s=1. Hereq sis
the query embedding at times,i sis the chosen tool, and the indicator variable tells whether the chosen
tool is the correct/optimal one or not. And thus the RAG performance is measured by
TX
t=1l(Θt; (qt, i∗
t)).
We make several important remarks about the setup. First, the online setup is mainly motivated by
the sequentially arriving nature of the user queries in RAG systems, and the nature makes it possible a
continual refinement of the embeddingsΘ t. The setup also allows a distribution shift ofq tover time, and
ideally,Θ tshould be online optimized to adapt to the shift over time. Second, the feedback structure
1{it=i∗
t}is mild as it doesn’t require knowing the optimali∗
tbut only whether the chosen onei tequals
i∗
tor not. Such abandit-styleorpartial-observationfeedback system removes the need for additional
data annotations on the optimali∗
tat each time (which users of the RAG may not even know), but
1{it=i∗
t}can be simply obtained by users’ interactions (thumb-up or -down) or rule-based judges of
task success. Third, we choose to optimize the database/toolbase embeddingsΘ tinstead of the query
embedding model that givesq tfor two reasons: (a) the query embedding models are sometimes blackbox
APIs and don’t provide a fine-tuning option, and (b) they are often general embedding models that are
used simultaneously for other RAG tasks, and fine-tuning against one RAG task may deteriorate its
performances on others. Lastly, we note our idea of online-optimizingΘ tcan be viewed as a lightweight
implementation of the tool description rewriting idea in building MCP-based agents (Anthropic, 2025);
we optimize in the embedding space, whereas Anthropic (2025) optimizes in the language space, both
for the tool use and function callings.
4

3 Online-Optimized RAG: Algorithms and Variants of RAG
In this section, we present our algorithm of online-optimized RAG and show how it can be applied to
several extensions beyond the main setup.
Algorithm 1Online-Optimized RAG (ORAG)
Input:Initial embeddingsΘ 1= [θ 1,1,θ1,2, . . . ,θ 1,I]⊤∈RI×d; learning rateη >0
1:fort= 1,2, . . .do
2:Observe query embeddingq t∈Rd.
3:Compute sampling probabilities from the currentΘ tby (1):
pt,i=pi(qt,Θt), i= 1, . . . , I.(3)
4:Sample an item (tool/document/function)i t∼pt= (p t,1, . . . , p t,I)and get feedback1{i t=i∗
t}.
5:Compute the (stochastic) gradient estimateg t,ifor each itemi:
gt,i=
pt,i−1{i=i t}1{i t=i∗
t}
pt,it
qt.(4)
6:Update embeddings forΘ t+1= [θ t+1,1, . . . ,θ t+1,I]⊤by
θt+1,i =θt,i−η·g t,i.
▷Optional:projectθ t+1,iinto some desired subspace (such as unit ball{θ∈Rd:∥θ∥ 2≤1})
7:end for
Algorithm 1 implements the standard RAG pipeline when handling a stream of user queries, except
for Step 5 and Step 6 where it updates the embeddingsΘ t.Essentially, the update performs a stochastic
gradient descent with respect to the loss function (2). We note that in calculating the update (4), it only
requires the knowledge of1{i=i∗
t}, i.e., we only need to know whether the chosen itemi tis the correct
one or not, but no need to knowi∗
t. As mentioned earlier, this creates much convenience in annotation
– no need for hiring annotators to labeli∗
t.The most important structural property of the update is
described by the following lemma.
Lemma 3.1.Fori= 1, ..., I,
E[gt,i] =∂l(Θ; (q t, i∗
t))
∂θi
Θ=Θ t
where the expectation is over the tool selectioni t∼ptas defined in Algorithm 1.
The lemma states that the update term at timetcan be viewed as a stochastic gradient of thet-th
term in the loss function (2). This enables a clean theoretical analysis of the algorithm, which we defer
to Section 5. The key to achieve this property in Lemma 3.1 is the coefficient beforeq t. Such a design
often appears for bias correction in adversarial online learning (Kakade et al., 2008; Auer et al., 2002).
Intuitively, for the chosen itemi=i t, if the choice is incorrect (i t̸=i∗
t), theng t,it=pt,it·qtand the
updateθ t+1,i t=θt,it−η p t,it·qtmovesθ t,itaway fromq t, decreasing their similarity. If the choice is
correct (i t=i∗
t), theng t,it=
pt,it−1
pt,it
qt, sop t,it−1
pt,it≤0and the update movesθ t,ittowardq t,
increasing similarity. The magnitude of this correction is proportional topt,it−1
pt,it, which is larger
when the model’s current confidencep t,itis smaller, i.e., we correct more aggressively when we were
unsure yet happened to be right. For all other itemsi̸=i t, the updateθ t+1,i =θt,i−η p t,i·qtmoves
θt,iaway fromq t, decreasing their similarity. This nonzero adjustment for unchosen items is because
the loss couples all items through the softmax normalization, and hence increasing probability on the
(unknown) correct item necessarily requires decreasing probability on the others. The dynamics are also
visualized in Figure 2.
5

We make the following remarks about the algorithm:
Learning rate.The parameterηcontrols the learning rate of embedding updates. As shown later,
a proper choice ofηyields convergence of the loss toward the optimum. In practice, a small constant
(e.g.,η= 10−5) prevents overly large changes. One can also use a time–varying schedule, e.g.,η t=c/√
t
withc >0, to taper updates as more (online) data arrive.
Session scope.The algorithm imposes no requirements on how(q t, i∗
t)are generated. In practice,
the embedding updates may aggregate interactions from a broad user population (to adapt universally)
or from a single user (to personalize the system). The prototypical version of Algorithm 1 performs
updates upon every sample, but one may easily convert it into a batched version by batching samples
from multiple timestamps or even an offline version.
Computation.As noted earlier, the algorithm is lightweight, and it performs one gradient update at
each time step. There is an even more efficient version which only performs an update to the embedding
of the chosen itemi t. We defer more details to Appendix B.2.
Exploration.Unlike Banditron (Kakade et al., 2008) for online multiclass prediction, which enforces
uniform exploration with a fixed probability, our procedure utilizes the inherent randomness of the vector
pt. The advantage of this exploration-free design is that it doesn’t sacrifice the current user experience
for future improvement of the system.
3.1 Variants of the RAG setup
Now we show how Algorithm 1 can be applied to more general RAG settings than the setup in the last
section. We report its numerical performance in the next section, and defer more implementation details
to Appendix B.3.
Kretrievals with reranker.Algorithm 1 retrieves one single tool/function per round. A practical
extension is to retrieveK≥2candidates and pass them to a reranker (e.g., a cross-encoder or an LLM
judge) that selects the best among them (Qin et al., 2023; Xu et al., 2024). In Algorithm 2, we deal with
the RAG system with a reranker; instead of sampling one item, it samples multiple items and lets the
reranker decide the best one. The algorithm thus modifies the sampling step of Algorithm 1 (line 4) by
inserting a reranking block.
Time-varying database.Algorithm 1 is also compatible with a dynamic toolbox{1, . . . , I}that
changes over time. In its variant Algorithm 3, at the start of roundt, it first updates the available
set of items and adjusts the embedding matrixΘ taccordingly (adds rows for new items and removes
rows for obsolete ones). Then computep tand proceed as usual, i.e., ensureΘ tcontains exactly the
items available at timet. This operates smoothly because (i) the sampling distribution is softmax-based
(it automatically re-normalizes over the current items), and (ii) the updates are item-wise (lines 5-6 of
Algorithm 1). This setting captures the case where the optimal tool for certain queries may not exist
in early phases (smallt) and only becomes available at a later stage. For example, the optimal tooli′
for a queryq′is unavailable fort <10, andq t=q′for allt≤10. Even without seeingi′, Algorithm 3
will repeatedly push the existing item embeddings away fromq′whenever the sampled item is incorrect.
This decreases their logitsq′⊤θt,iand thus their softmax probabilities relative to the (eventual) optimal
item. Wheni′is introduced att= 10, even with an untouched, reasonable initialization aligned toq′, its
selection probability will be comparatively higher, improving retrieval without any special warm start.
Multi-hop retrieval.Some RAG tasks requiremulti-hopretrieval to select multiple items that
jointly solve the task (Tang and Yang, 2024). A common strategy is to use a planner (e.g., an LLM)
to decompose the input into sub-tasks (Shen et al., 2023; Qin et al., 2023; Lumer et al., 2024). In such
a setting, we can apply Algorithm 1 at each hop by reducing the multi-hop query/task to a sequence
of single-hop sub-tasks. Concretely, in the variant Algorithm 4, at hoph(theh-th sub-task), we run
6

Algorithm 1 to select an item and obtain feedback from a judge (e.g., an LLM or rule-based judge when
a human is unavailable) indicating whether the selection advances or answers the query. These per-hop
updates align the embeddings across the entire multi-hop pipeline.
Algorithms 2, 3 and 4 are all formally described in Appendix B.3.
4 Experiments
For Algorithm 1 and its variants, we evaluate them on both tool calling and information retrieval tasks
and conduct experiments on several open source benchmarks. We summarize the experiment setup here
and defer the implementation details to Appendix C. Unless otherwise noted, all results of our methods
are computed as the average of five independent runs.
Datasets.For tool use, we adoptUltraTool(Huang et al., 2024) and three sub-tasks fromTool-
Ret(Shi et al., 2025):ToolRet-Web,ToolRet-Code, andToolRet-Customized. For information retrieval,
we useFiQAbenchmark (Thakur et al., 2021). For multi-hop reasoning, we useMultiHopRAG(Tang
and Yang, 2024). These datasets provide challenging real-world scenarios for retrieval tasks.
Baselines.We compare our method against a strong suite of retrieval models following the method-
ology in Shi et al. (2025). The baselines include a sparse retriever based onBM25(Huang et al.,
2024), competitive dense retrievers accessed via API: OpenAI’stext-embedding-3-largeand Qwen’s
text-embedding-v4, and also two state-of-the-art cross-encoder models of different sizes based on pre-
vious research and benchmark reports (Muennighoff et al., 2022; Tang and Yang, 2024; Shi et al., 2025):
Qwen3-Reranker-0.6Bandbge-reranker-v2-gemma.
Metrics.For all retrieval tasks, we report performance using standard information retrieval metrics:
Recall@k (R@k) and NDCG@k (N@k). Following common practice, we choosek= 10. For tool-use
simulation experiments, we also report the function-call accuracy.
4.1 Retrieval performance
We begin by assessing Algorithm 1 through a comparison with strong baselines in the retrieval literature.
We report metrics after applying the method with an average of3000updates to the embeddings (exact
numbersvarybasedontheselectedbatchsizeanddatasetsize), andweincludetheinitialmodelswithout
online updates. Table 1 presents the results, whereOursdenotes the results of Algorithm 1.
Table 1: Retrieval performance at k=10. The best result in each column is highlighted. Percentage
improvements of our methods over their baselines are shown below each score. Dataset names are
abbreviated:U-Tool(UltraTool),T-Web(ToolRet-Web),T-Code(ToolRet-Code), andT-Custom
(ToolRet-Customized).
MethodU-Tool FiQA T-Web T-Code T-Custom
R@10 N@10 R@10 N@10 R@10 N@10 R@10 N@10 R@10 N@10
BM25 0.3208 0.2003 0.2955 0.2326 0.1778 0.1428 0.3446 0.2421 0.4922 0.3816
bge-reranker-v2-gemma 0.8448 0.5852 0.75000.4655 0.4849 0.3486 0.6081 0.53220.6455 0.5221
Qwen3-Reranker-0.6B 0.7200 0.4590 0.5500 0.4361 0.3622 0.1897 0.5802 0.4781 0.6274 0.4923
text-embedding-v4 0.7451 0.5064 0.5335 0.4604 0.2701 0.1453 0.5291 0.3770 0.5066 0.4097
text-embedding-3-large 0.8356 0.6067 0.6258 0.5462 0.3243 0.1675 0.5347 0.3582 0.6378 0.5204
Ours (text-emb.-v4)0.8256
(+8.05%)0.5982
(+8.28%)0.5464
(+1.29%)0.4698
(+0.94%)0.3657
(+9.56%)0.1968
(+5.15%)0.5960
(+6.69%)0.4280
(+5.10%)0.5739
(+6.73%)0.4398
(+3.01%)
Ours (text-emb.-3-L.)0.8682
(+3.26%)0.6522
(+4.55%)0.6421
(+1.63%)0.5680
(+2.18%)0.3780
(+5.37%)0.2065
(+3.90%)0.5849
(+5.02%)0.4070
(+4.88%)0.6937
(+5.59%)0.5735
(+5.31%)
Results.The results give several key insights. First, our proposed method demonstrates a significant
and consistent improvement over its base dense retrieval models. For example, on the ToolRet-Code
7

benchmark, both thetext-embedding-large-3and thetext-embedding-v4baselines gain significant
performance improvements, and the initially underperformedtext-embedding-v4even outperforms the
text-embedding-large-3after the optimization via our method. This shows our approach is not only
effective but also versatile, enhancing strong existing models without requiring architectural changes.
Second, traditional sparse retrieval methods likeBM25, which rely on lexical matching, consistently
underperform across all benchmarks. This highlights the necessity of semantic understanding for the
nuanced task of tool retrieval, where the user’s intent may not share keywords with the tool’s description.
Finally, while powerful reranker models can achieve high performance on specific tasks, their practical
utility is often limited by high computational costs, making them unsuitable for real-time applications.
As visualized in Figure 3, our method provides a much more balanced and practical solution, achieving
state-of-the-art performance while maintaining low inference latency.
Figure 3: Performance vs. time cost of different retrieval methods. The performance is the arithmetic
average of R@10 results in Table 1, and the time cost is evaluated and recorded on the same GPU server.
The embedding model time cost is obtained by using theQwen3-Embedding-4Bas the proxy.
4.2 Algorithm 1’s variants evaluation
We now evaluate the adaptability of our method across several practical scenarios, including integration
with rerankers, time-varying databases, and multi-hop retrieval tasks (see the variants discussed in Sec-
tion 3.1). We use theUltraToolbenchmark for experiments on dynamic databases and integration with
rerankers, and theMultiHopRAGbenchmark for the multi-hop retrieval task. The detailed experiment
setup is provided in Appendix C.
Integration with Rerankers.We consider a pipeline where an LLM reranks the top candidates
retrieved by our model before a final tool is selected. For each query, a reranker model reranks the
sampled 10 tool documentations, and the success of the final tool call provides the gradients for our
algorithm as shown in Algorithm 2. For reproducibility, we employ RankGPT (Sun et al., 2023) with
gpt-4.1-nano-2025-04-14as the reranker. We compare this LLM-as-reranker approach against our
standard method that samples directly from the learned policy and also the baseline where we make
no updates to embeddings. The results are presented in Figure 4. We observe that during the early
stage, it is indifferent whether to use a reranker or not. Later, the reranker accelerates improvement in
retrieval performance. The reason is that a stronger reranker increases the probability of selecting the
correct item. Intuitively, a successful retrieval yields a precise signal that the chosen item is correct,
while a failed retrieval only indicates that the chosen item is incorrect without revealing which item is
correct. By increasing the rate of successful retrievals, the reranker provides more informative feedback
for subsequent learning.
Time-varying database.We study a setting where the toolbase changes over time. At the start,
only a random subset of tools is available, and the remaining tools are introduced after half of the
queries have been processed. Under this setup, part of the embeddings cannot be updated during the
8

Figure 4: Performance onUltraToolwith and without an LLM-based reranker. Integrating an LLM
reranker provides a stronger signal, accelerating learning and further boosting retrieval performance.
first phase, and for some queries, the ground truth optimal tool may be temporarily unavailable. Even
so, our method can improve the performance as discussed in Section 3.1. We compare this dynamic
setting, labeled asDynamic DB (Algorithm 3), with a static baseline where all tools are available from
the beginning, labeled asDefault (Algorithm 1). As in Figure 5, though removing embeddings at the
beginning can reduce recall, our method adapts to the changing set of tools and achieves consistent gains.
Figure 5: Performance onUltraToolin static vs. dynamic database settings. Our method demonstrates
robust adaptation, maintaining consistent improvements even when the toolset changes midway through
the experiment.
Multi-hop retrieval.We evaluate our method in a multi-hop setting, where solving an input
task requires a sequence of successful tool retrievals. The plug-and-play nature of our algorithm enables
straightforwardintegrationintotheexistingmulti-hopframeworks. Weimplementaquerydecomposition
pipeline in which a planner first decomposes the input task into several subtasks, and Algorithm 1 is
applied to each subtask, as discussed in Section 3.1. Here, for each subtask query, we retrieve only 5
documents. We evaluate on theMultiHopRAGbenchmark, and the performance changes are shown in
Figure 6. This integration yields a substantial improvement in end-to-end question answering accuracy,
from0.55to0.68.
5 Theoretical Analysis
In this section, we provide a theoretical analysis of Algorithm 1. The aim is not so much for a theoretical
peace of mind but to derive more insights for implementing the algorithm in practice. Generally, the
9

Figure 6: Performance changes on theMultiHopRAGbenchmark. The baseline is computed using the
same retrieval and question-answering workflow (see Appendix C) with rawtext-embedding-3-large
embeddings. Integrating our method into a task decomposition pipeline demonstrates stable learning,
leading to improved multi-hop QA performance.
performance of an online algorithm/policyπis measured by itsregret
Regπ 
{(qt, i∗
t)}T
t=1
=TX
t=1l(Θt; (qt, i∗
t))−TX
t=1l(Θ∗; (qt, i∗
t)),
whereΘ tis the embedding at timetspecified by the policyπ, Algorithm 1 in our context, andΘ∗is
the optimal embedding defined by (2) upon optimizing over all the queries in a hindsight manner. As
noted earlier, we make no assumption on the generation ofq tandi∗
t.
Theorem 5.1.For any sequence{(q t, i∗
t)}T
t=1, Algorithm 1 (ORAG) with initializationΘ 1and learning
rateη >0satisfies
Eh
RegORAG 
{(qt, i∗
t)}T
t=1i
≤∥Θ1−Θ∗∥2
F
2η+η
2TX
t=11
pt,i∗
t−2p t,i∗
t+ 1
∥qt∥2
2,
where∥ · ∥ Fdenotes the Frobenius norm and the expectation is with respect to the randomness ofi t’s.
Theorem 5.1 gives a problem-dependent regret bound for Algorithm 1. The first term depends on the
initializationΘ 1, whereas the second depends on the probabilitiesp t,i∗
tof selecting the optimal items.
For the initialization quality, the term∥Θ 1−Θ∗∥2
Fquantifies how close the initial embeddings are to the
optimum. If the initialization is good (i.e., close toΘ∗), only minor updating is needed. For the second
term, we can interpret it as the confidence in the optimal item. The summation grows when the model
assigns low probabilities to the optimal item. Intuitively, lower confidence (smallerp t,i∗
t) incurs larger
regret. In particular, ifp t,i∗
t= 1then the contribution at timetis zero and the corresponding gradient
gt,ivanishes for alliand there is no need to adjust the embeddings. Further, in an unrealistically ideal
case, ifp t,i∗
t≡1for allt, thenRegORAG= 0and Algorithm 1 leavesΘ t≡Θ 1unchanged. In the light of
Lemma 3.1, the proof follows the standard analysis of online gradient descent (Hazan et al., 2016) and
is deferred to Appendix D.
With an appropriate choice ofηto tradeoff these two aspects, Algorithm 1 achieves sublinear regret
inT; equivalently, the average regret tends to zero and the loss approaches the optimum ofΘ∗:
Corollary 5.2.Assume there exist constants ¯Θ>0,p ∈(0,1), and¯q >0such that∥Θ 1−Θ∗∥2
F≤¯Θ,
10

Figure 7: Cumulative regret for Algorithm 1. See Appendix C for the setup.
pt,i∗
t≥p, and∥q t∥2
2≤¯qfor allt. Then, withη=r
p¯Θ
¯q(1−p )(1+2p )T, we have
Eh
RegORAG 
{(qt, i∗
t)}T
t=1i
≤s¯Θ ¯q(1−p )(1 + 2p )T
p=O(√
T).
Corollary5.2showsthatAlgorithm1attainsO(√
T)regretrelativetotheoptimalembeddings(know-
ing all incoming queries in the hindsight). While the choice ofηabove depends on several parameters,
in practice (and in our experiments) a small constant with a time-varying schedule, e.g.,η t=c/√
twith
c= 10−5(as used in standard online convex optimizations (Hazan et al., 2016)) can work well across
different contexts. Figure 7 empirically verifies a sublinear cumulative regret for Algorithm 1. We also
draw a connection between the cross-entropy loss (used in the above regret analysis) and the accuracy
metric, and we refer to Appendix B.
6 Conclusion
We introduceOnline-Optimized RAG, a deployment-time framework for tool use and function calling
thatcontinuallyimprovesretrievalbyupdatingembeddingsfromliveinteractionswithminimalfeedback.
Our method casts retrieval as online classification and employs lightweight gradient-style updates that
preserve latency and throughput, scale to large catalogs, and integrate seamlessly with existing LLM
pipelines without retraining the generator. We provide theoretical guarantees and an analysis linking
initial embedding quality to downstream performance, supported by empirical evaluation on real retrieval
workloads. We hope this work catalyzes future deployment of self-improving RAG systems.
References
Alazraki, Lisa, Marek Rei. 2024. Meta-reasoning improves tool use in large language models.arXiv
preprint arXiv:2411.04535.
Anthropic. 2025. How we built our multi-agent research system.
https://www.anthropic.com/engineering/multi-agent-research-system.
Asai, Akari, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection .
Auer, Peter, Nicolo Cesa-Bianchi, Yoav Freund, Robert E Schapire. 2002. The nonstochastic multiarmed
bandit problem.SIAM journal on computing32(1) 48–77.
11

Bartlett, Peter L, Michael I Jordan, Jon D McAuliffe. 2006. Convexity, classification, and risk bounds.
Journal of the American Statistical Association101(473) 138–156.
Braunschweiler, Norbert, Rama Doddipatla, Tudor-Catalin Zorila. 2025. Toolreagt: Tool retrieval for
llm-based complex task solution via retrieval augmented generation.Proceedings of the 3rd Workshop
on Towards Knowledgeable Foundation Models (KnowFM). 75–83.
Chen, Jiuhai, Jonas Mueller. 2023. Quantifying uncertainty in answers from any language model and
enhancing their trustworthiness.arXiv preprint arXiv:2308.16175.
Fu, Jia, Xiaoting Qin, Fangkai Yang, Lu Wang, Jue Zhang, Qingwei Lin, Yubo Chen, Dongmei Zhang,
Saravan Rajmohan, Qi Zhang. 2024. Autorag-hp: Automatic online hyper-parameter tuning for
retrieval-augmented generation.arXiv preprint arXiv:2406.19251.
Gao, Luyu, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham
Neubig. 2023. Pal: Program-aided language models.International Conference on Machine Learning.
PMLR, 10764–10799.
Goswami, Dipam, Liying Wang, BartĹ Twardowski, Joost van de Weijer, et al. 2025. Query drift
compensation: Enabling compatibility in continual learning of retrieval embedding models.arXiv
preprint arXiv:2506.00037.
Gutiérrez, Bernal Jiménez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su. 2025. From rag to memory:
Non-parametric continual learning for large language models.arXiv preprint arXiv:2502.14802.
Hao, Shibo, Tianyang Liu, Zhen Wang, Zhiting Hu. 2023. Toolkengpt: Augmenting frozen language
models with massive tools via tool embeddings.Advances in neural information processing systems
3645870–45894.
Hazan, Elad, et al. 2016. Introduction to online convex optimization.Foundations and Trends®in
Optimization2(3-4) 157–325.
Huang, Shijue, Wanjun Zhong, Jianqiao Lu, Qi Zhu, Jiahui Gao, Weiwen Liu, Yutai Hou, Xingshan
Zeng, Yasheng Wang, Lifeng Shang, Xin Jiang, Ruifeng Xu, Qun Liu. 2024. Planning, creation, usage:
Benchmarking llms for comprehensive tool utilization in real-world complex scenarios .
Jeong, Soyeong, JinheonBaek, SukminCho, SungJuHwang, JongCPark.2024. Adaptive-rag: Learning
to adapt retrieval-augmented large language models through question complexity.arXiv preprint
arXiv:2403.14403.
Jimenez Gutierrez, Bernal, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su. 2024. Hipporag: Neuro-
biologically inspired long-term memory for large language models.Advances in Neural Information
Processing Systems3759532–59569.
Kakade, Sham M, Shai Shalev-Shwartz, Ambuj Tewari. 2008. Efficient bandit algorithms for online
multiclass prediction.Proceedings of the 25th international conference on Machine learning. 440–447.
Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances in neural information processing systems33
9459–9474.
12

Li, Minghao, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang,
Yongbin Li. 2023. Api-bank: A comprehensive benchmark for tool-augmented llms.arXiv preprint
arXiv:2304.08244.
Liu, Linyu, Yu Pan, Xiaocheng Li, Guanting Chen. 2024a. Uncertainty estimation and quantification for
llms: A simple supervised approach.arXiv preprint arXiv:2404.15993.
Liu, Shengjie, Alex Lu, Li Dong, Jason Zhu, Manish Gawali, Alice Zhou. 2025. Toposem: In-context
planning with semantically-informed tooling graph similarity .
Liu, Shudong, Zhaocong Li, Xuebo Liu, Runzhe Zhan, Derek Wong, Lidia Chao, Min Zhang. 2024b.
Can llms learn uncertainty on their own? expressing uncertainty effectively in a self-training manner.
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 21635–
21645.
Lumer, Elias,VamseKumarSubbiah,JamesABurke, PradeepHonaganahalliBasavaraju,AustinHuber.
2024. Toolshed: Scale tool-equipped agents with advanced rag-tool fusion and tool knowledge bases.
arXiv preprint arXiv:2410.14594.
Muennighoff, Niklas, Nouamane Tazi, Loïc Magne, Nils Reimers. 2022. Mteb: Massive text embedding
benchmark.arXiv preprint arXiv:2210.07316doi:10.48550/ARXIV.2210.07316. URLhttps://arxiv.
org/abs/2210.07316.
Nikitin, Alexander, Jannik Kossen, Yarin Gal, Pekka Marttinen. 2024. Kernel language entropy: Fine-
grained uncertainty quantification for llms from semantic similarities.Advances in Neural Information
Processing Systems378901–8929.
Oche, Agada Joseph, Ademola Glory Folashade, Tirthankar Ghosal, Arpan Biswas. 2025. A systematic
review of key retrieval-augmented generation (rag) systems: Progress, gaps, and future directions.
arXiv preprint arXiv:2507.18910.
Patil, Shishir G, Tianjun Zhang, Xin Wang, Joseph E Gonzalez. 2024. Gorilla: Large language model
connected with massive apis.Advances in Neural Information Processing Systems37126544–126565.
Qin, Yujia, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru
Tang, Bill Qian, et al. 2023. Toolllm: Facilitating large language models to master 16000+ real-world
apis.arXiv preprint arXiv:2307.16789.
Reddy, Revanth Gangi, Pradeep Dasigi, Md Arafat Sultan, Arman Cohan, Avirup Sil, Heng Ji, Han-
naneh Hajishirzi. 2023. Refit: Relevance feedback from a reranker during inference.arXiv preprint
arXiv:2305.11744.
Schick, Timo, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke
Zettlemoyer, Nicola Cancedda, Thomas Scialom. 2023. Toolformer: Language models can teach them-
selves to use tools.Advances in Neural Information Processing Systems3668539–68551.
Shen, Yongliang, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang. 2023. Hugginggpt:
Solvingaitaskswithchatgptanditsfriendsinhuggingface.Advances in Neural Information Processing
Systems3638154–38180.
Shi, Zhengliang, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, Zhaochun
Ren. 2025. Retrieval models aren’t tool-savvy: Benchmarking tool retrieval for large language models.
arXiv preprint arXiv:2503.01763.
13

Su, Weihang, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia
Zhou, Yiqun Liu. 2025. Parametric retrieval augmented generation.Proceedings of the 48th Interna-
tional ACM SIGIR Conference on Research and Development in Information Retrieval. 1240–1250.
Sun, Jiashuo, Xianrui Zhong, Sizhe Zhou, Jiawei Han. 2025. Dynamicrag: Leveraging outputs of large
language model as feedback for dynamic reranking in retrieval-augmented generation.arXiv preprint
arXiv:2505.07233.
Sun, Weiwei, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin,
Zhaochun Ren. 2023. Is chatgpt good at search? investigating large language models as re-ranking
agents.arXiv preprint arXiv:2304.09542.
Tang, Xiaqiang, Qiang Gao, Jian Li, Nan Du, Qi Li, Sihong Xie. 2024. Mba-rag: a bandit ap-
proach for adaptive retrieval-augmented generation through question complexity.arXiv preprint
arXiv:2412.01572.
Tang, Yixuan, YiYang.2024. Multihop-rag: Benchmarkingretrieval-augmentedgenerationformulti-hop
queries.arXiv preprint arXiv:2401.15391.
Tewari, Ambuj, Peter L Bartlett. 2007. On the consistency of multiclass classification methods.Journal
of Machine Learning Research8(5).
Thakur, Nandan, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evaluation of information retrieval models.Thirty-fifth
Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).
URLhttps://openreview.net/forum?id=wCu6T5xFjeJ.
Wang, Renxi, Xudong Han, Lei Ji, Shu Wang, Timothy Baldwin, Haonan Li. 2024. Toolgen: Unified
tool retrieval and calling via generation.arXiv preprint arXiv:2410.03439.
Xu, Qiancheng, Yongqi Li, Heming Xia, Wenjie Li. 2024. Enhancing tool retrieval with iterative feedback
from large language models.arXiv preprint arXiv:2406.17465.
Xu, Qiancheng, Yongqi Li, Heming Xia, Fan Liu, Min Yang, Wenjie Li. 2025. Petoolllm: Towards
personalized tool learning in large language models.arXiv preprint arXiv:2502.18980.
Yuan, Siyu, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, Deqing
Yang. 2024. Easytool: Enhancing llm-based agents with concise tool instruction.arXiv preprint
arXiv:2401.06201.
Zeighami, Sepanta, Zac Wellmer, Aditya Parameswaran. 2024. Nudge: Lightweight non-parametric
fine-tuning of embeddings for retrieval.arXiv preprint arXiv:2409.02343.
Zhang, Tianjun, ShishirGPatil, NamanJain, ShengShen, MateiZaharia, IonStoica, JosephEGonzalez.
2024. Raft: Adapting language model to domain specific rag.arXiv preprint arXiv:2403.10131.
Zhang, William, Yiwen Zhu, Yunlei Lu, Mathieu Demarne, Wenjing Wang, Kai Deng, Nutan Sahoo,
Katherine Lin, Miso Cilimdzic, Subru Krishnan. 2025. Flair: Feedback learning for adaptive informa-
tion retrieval.arXiv preprint arXiv:2508.13390.
14

A Related Work
A.1 Retrieval augmented generation
Retrieval-augmentedgeneration(RAG)(Lewisetal.,2020)augmentsLLMswitharetrieverthatsupplies
passages from an external knowledge base and instructs the model to answer using those passages. By
exposing sources, RAG reduces the risk of hallucinations and improves factuality. We refer to the survey
paper Oche et al. (2025) for a comprehensive review of RAG.
The first line of work adapts when and how much to retrieve at inference time rather than changing
the retriever itself. SELF-RAG (Asai et al., 2024) lets the model decide when to retrieve and critique
its own outputs, reducing hallucinations over standard RAG, while Adaptive-RAG (Jeong et al., 2024)
learns a lightweight router that sends easy questions to zero/single-shot retrieval and harder ones to
multi-step pipelines, trading accuracy for latency. Building on online signals, AutoRAG-HP (Fu et al.,
2024) frames top-k(and related knobs) as a hierarchical bandit tuned from live feedback; MBA-RAG
(Tang et al., 2024) treats whole retrieval policies (none/one-shot/multi-step) as arms; and DynamicRAG
(Sun et al., 2025) optimizes a reranker via reinforcement learning to reorder passages and choosekper
query. These methods primarily tune controller hyper-parameters rather than updating the embedding
space as we do.
Another line of works swaps or augments the retrieval substrate itself. Parametric RAG (Su et al.,
2025) pre-parameterizes documents as small LoRA adapters so the model retrieves by merging adapters
instead of consuming long contexts, while HippoRAG and its follow-up (Jimenez Gutierrez et al., 2024;
Gutiérrez et al., 2025) build an open knowledge graph and use graph walks to achieve multi-hop, context-
aware retrieval.
Closer to our aim of aligning the retriever with usage signals, several papers adjust representations
at inference or through fine-tuning. RAFT (Zhang et al., 2024) fine-tunes generators to quote the
right spans, improving faithfulness under noisy top-k. For continual retriever training, Goswami et al.
(2025) estimates query-embedding drift for new tasks and compensates it at retrieval time to preserve
compatibility with an existing index. ReFIT (Reddy et al., 2023) distills a cross-encoder reranker into
the query embedding on the fly and re-retrieves with the updated query vector; FLAIR (Zhang et al.,
2025) leverages user/synthetic indicator feedback to re-rank via a two-track scoring scheme; and NUDGE
(Zeighami et al., 2024) fine-tunes document embeddings through offline training and validation datasets
with positive feedback. However, these approaches are controller-tuning, offline, or/and require labeled
offline datasets. In contrast, our method performs lightweight online gradient updates to the retrieval
embeddings from minimal deployment feedback (e.g., solved/unsolved).
A.2 Tool use and function calling
Tool use and function calling are now core capabilities of modern LLMs: models can invoke external
resourcestocompletetasksbycallingAPIs(Qinetal.,2023;Patiletal.,2024;Lietal.,2023), executinga
Python interpreter (Gao et al., 2023), or orchestrating other AI models (Shen et al., 2023). In particular,
RAG is commonly employed in the function-call setting for tool-augmented LLMs: given a user query,
the system retrieves tool/function specifications and examples from a catalog so the model can select
and parameterize the correct call (e.g., Shen et al. (2023); Liu et al. (2025); Lumer et al. (2024); Alazraki
and Rei (2024); Xu et al. (2024)).
To strengthen tool use, methods generally combine two phases: offline training and online inference.
Offline approaches fine-tune LLMs on curated tool-use corpora (Qin et al., 2023; Patil et al., 2024; Li
et al., 2023; Hao et al., 2023; Wang et al., 2024; Schick et al., 2023). Online techniques improve calling
performance at inference time by supplying clearer tool descriptions, leveraging the model’s reasoning,
15

and incorporating feedback loops (Yuan et al., 2024; Alazraki and Rei, 2024; Lumer et al., 2024; Xu et al.,
2024; Shen et al., 2023). Within the feedback-driven line, PEToolLLaMA (Xu et al., 2025) personalizes
tool learning through supervised fine-tuning and direct preference optimization, while Xu et al. (2024)
iteratively refines queries using tool feedback to improve retrieval accuracy at the cost of additional
latency. Both frameworks require offline model updates and/or multi-step inference. By contrast, our
approach targets the retrieval layer that underpins function selection: we perform lightweight online
gradient updates to the retrieval embeddings using minimal deployment feedback, aligning tool retrieval
without fine-tuning the LLM or adding complex controllers. This yields a plug-and-play mechanism for
robust, self-improving tool use during deployment.
B More Discussions
B.1 Discussion for cross-entropy loss
We choose the cross-entropy loss since its convexity inΘand also its surrogate property for the0–1
loss (Tewari and Bartlett, 2007; Bartlett et al., 2006): optimizingΘby minimizing cross-entropy is
statistically aligned with maximizing top-1 retrieval accuracy.
The retrieval task can be cast as multiclass prediction with inputqand labeli∗. TheBayesian0–1
risk of a decision ruleg(q)∈ {1, . . . , I}is
R0–1(g) = Pr 
g(Q)̸=I∗
=E[1{g(Q)̸=I∗}],
whose Bayes-optimal rule isg⋆(q) = arg max iηi(q), whereη i(q):= Pr(I∗=i|Q=q)and the above
probability is with respect to the randomness of(Q,I∗). Directly minimizing the0–1risk is intractable;
a standard approach is to minimize the (population) cross-entropy (CE) risk of a probabilistic predictor
p(q,Θ)with parameterΘ,
RCE(Θ) =E[−logp I∗(Q,Θ)].
TheCElossisacalibrated surrogateforthe0–1loss: itsconditionalminimizerpredictsthetrueposteriors
η(q), and any sequence of models whose CE risk approaches its minimum induces decision rules whose
0–1risk approaches the Bayes risk. Thus, trainingΘby minimizing cross-entropy (withp igiven by the
softmax in (1)) is statistically aligned with maximizing top-1 retrieval accuracy as shown in Proposition
B.1, which follows the standard analysis of surrogate properties (Tewari and Bartlett, 2007; Bartlett
et al., 2006).
Proposition B.1.Let(Q, I∗)be distributed according to some unknown law. For any measurable
p(q,Θ)∈∆I−1and the induced classifierg Θ(q):= arg max ipi(q,Θ), define
ηi(q):= Pr(I∗=i|Q=q),∆(q) :=η(1)(q)−η (2)(q),
whereη (1)≥η(2)≥ ···are the sorted coordinates ofη(q). Then:
(i) (Conditional optimality)For each fixedq, the conditional CE risk
L(p;η(q)) :=E[−logp I∗|Q=q] =−IX
i=1ηi(q) logp i
is uniquely minimized overp∈∆I−1atp=η(q).
16

(ii) (Excess-risk decomposition)
RCE(Θ)−inf
pRCE=E
KL 
η(Q)∥p(Q,Θ)
,
whereinf pRCE=E
H(η(Q))
, withHthe Shannon entropy.
(iii) (Bayes consistency / classification calibration)SupposePr 
∆(Q) = 0
= 0(no ties almost surely).
If a sequenceΘ nsatisfiesR CE(Θn)→inf pRCE, then
R0–1 
gΘn
−→inf
gR0–1=R 0–1(g⋆).
Proof. (i)For fixedη,L(p;η) =−P
iηilogp iis minimized atp=ηby Gibbs’ inequality, since
−X
iηilogp i=H(η) + KL(η∥p)≥H(η),
with equality iffp=η.
(ii)Taking expectation overQin the identity above yields
RCE(Θ) =E[H(η(Q))] +E
KL 
η(Q)∥p(Q,Θ)
,
and the infimum over allpis attained byp=ηpointwise.
(iii)By (ii),R CE(Θn)↓infR CEimpliesE
KL 
η(Q)∥p(Q,Θ n)
→0. Pinsker’s inequality gives,
for eachn,
∥η(Q)−p(Q,Θ n)∥TV≤q
1
2KL 
η(Q)∥p(Q,Θ n)
,
hence the total variation distance converges to0inL1and along a subsequence almost surely. Wherever
∆(Q)>0, this forcesarg max ipi(Q,Θ n) = arg max iηi(Q)for all largen. Thereforeg Θn(Q)→g⋆(Q)
almost surely, and by bounded convergence,R 0–1(gΘn)→ R 0–1(g⋆).
Remark.In our formulation,p i(q,Θ)is the softmax in (1), which maps any score vector to a valid
probability vector. Minimizing the sample average of−logp i∗(q,Θ)is therefore an empirical proxy for
minimizingR CE(Θ), and by the proposition it targets the Bayes-optimal top-1 retrieval rule under the
0–1criterion. We also note that the retrieval probabilitiesp i(q,Θ)are induced directly by the embedding
scores and could potentially be improved via calibration (Chen and Mueller, 2023; Liu et al., 2024a,b;
Nikitin et al., 2024), which is an interesting direction for future work.
B.2 More efficient gradient update
Relative to using fixed embeddings (e.g.,Θ 1), Algorithm 1 adds only two per-round operations: gradient
computation and embeddings update. For very large tool catalogs, an even more efficient variant updates
only the chosen itemθ t,iteach round: compute the (stochastic) gradient estimate for the sampled item
it,
gt,it=
1−1{it=i∗
t}
pt,it
qt,
and update
θt+1,i =

θt,i−ηg t,it,ifi=i t,
θt,i,otherwise.
17

With a similar analysis of Lemma 3.1, we can showg′
t,i=1{i=i t}
1−1{it=i∗
t}
pt,it
qt(which matches
the above variant update by noting the indicator1{i=i t}nulls the unchosen items) is also an unbiased
estimator for gradients for alli. Because only the sampled item is modified at each iteration, this variant
is attractive when the number of items is large.
B.3 Variants of Algorithm 1
Algorithm 2ORAG withKretrievals
Input:Initial embeddingsΘ 1∈RI×d; learning rateη >0; beam sizeK∈ {1, . . . , I}; reranker
Rerank(q,I)→i∈ I
1:fort= 1,2, . . .do
2:Observe query embeddingq t∈Rd.
3:Compute sampling probabilities from currentΘ tvia (1):
pt,i=pi(qt,Θt), i= 1, . . . , I.
4:Sample a setI tof sizeKwithout replacementfromp t= (p t,1, . . . , p t,I).
5:Obtain final choicei t←Rerank(q t,It)and observe feedback1{i t=i∗
t}.
6:Compute the (stochastic) gradient estimateg t,ifor each itemi:
gt,i=
pt,i−1{i=i t}1{i t=i∗
t}
pt,it
qt.(5)
7:Update embeddings forΘ t+1= [θ t+1,1, . . . ,θ t+1,I]⊤:
θt+1,i =θt,i−η·g t,i.
8:end for
Algorithm 3ORAG with Dynamic Database
Input:Initial item setI 1and embeddingsΘ 1∈R|I1|×d; learning rateη >0; initializerInitEmbed(i)∈
Rdfor new items
1:fort= 1,2, . . .do
2:Observe current available item setI t(additions/removals relative toI t−1).
3:Maintain embeddings forΘ t:
•For eachi∈ I t\ It−1(new item), add a rowθ t,i←InitEmbed(i)toΘ t.
•For eachi∈ I t−1\ It(removed item), delete rowθ t−1,ifromΘ t−1.
4:Observe query embeddingq t∈Rd.
5:Compute probabilities over available items via (1):
pt,i=pi(qt,Θt), i∈ I t.
6:Samplei t∼ptand observe1{i t=i∗
t}.
7:Compute the (stochastic) gradient estimateg t,ifor each itemi:
gt,i=
pt,i−1{i=i t}1{i t=i∗
t}
pt,it
qt.(6)
8:Update embeddings forΘ t+1= [θ t+1,1, . . . ,θ t+1,I]⊤:
θt+1,i =θt,i−η·g t,i.
9:end for
18

Algorithm 4ORAG with Multi-Hop
Input:Initial embeddingsΘ 1= [θ 1,1, . . . ,θ 1,I]⊤∈RI×d; learning rateη >0
1:fort= 1,2, . . .do
2:Observe a sequence of sub-task embeddingsQ t={q(h)
t}Ht
h=1.
3:InitializeΘ(1)
t←Θ t.
4:forh= 1,2, . . . , H tdo▷sub-tasks within roundt
5:Compute sampling probabilities via (1):
pt,h,i=pi
q(h)
t,Θ(h)
t
, i= 1, . . . , I.(7)
6:Sample an itemi t,h∼pt,h= (p t,h,1, . . . , p t,h,I)and obtain judge feedbacky t,h∈ {0,1}.
7:Compute the (stochastic) gradient estimateg t,h,ifor each itemi:
gt,h,i=
pt,h,i−1{i=i t,h}yt,h
pt,h,it,h
q(h)
t.(8)
8:Update embeddings:
θ(h+1)
t,i←θ(h)
t,i−η·g t,h,i.
9:end for
10:SetΘ t+1←Θ(Ht+1)
t.
11:end for
C Experiment Details
C.1 Dataset and prompt details
This part provides details on the construction of queries and document/tool representations for each
benchmark. To ensure reproducibility, we outline the exact data fields and templates used to generate
the text for embedding.
C.1.1 UltraTool
Following prior work (Braunschweiler et al., 2025), we decompose each annotated plan step into a stan-
dalone retrieval query. For each sample, we construct the query from the top-level question (question
column) and the specific plan step (stepcolumn) using the following template:
Given the following task:"{question}", select the best tool provided in the context to
solve the following substep:"{step}".
The resulting text is used as the input for the query embedding model.
For each tool, we create a single text representation (stored astext_representationcolumn) by
concatenating the following fields in order. Fields that are empty are omitted.
•Name:name
•Description:description
•Arguments:arguments(parsed as a JSON string)
•Results:results(parsed as a JSON string)
This concatenated string is used to embed the tool documentation.
19

C.1.2 ToolRet
We use all 35 sub-tasks from the ToolRet benchmark (Shi et al., 2025). Following the original paper,
we use an instruction-based format for queries, concatenating the providedinstructionand the user
query:
{instruction}\n{query}
For tool documentation, we perform a schema-aware extraction from the raw JSON object. We
extract and join the following fields with newlines:
•ToolRet-Code:name,description,func_description,functionality
•ToolRet-Web/Customized:name,description
If a field is not present or the documentation is not a valid JSON object, we fall back to using the raw
documentation string for embedding.
C.1.3 FiQA
FortheFiQAbenchmark(Thakuretal.,2021),wefollowthestandardsetupfromtheMTEBtoolkit(Muen-
nighoff et al., 2022). For both corpus and queries, we embed the content of thetextfield.
C.1.4 MultiHopRAG
For each originalquery, we generate a sequence of sub-queries (decomposed_questions) using an LLM-
based query decomposition strategy. We use the following prompt for this task:
You are an expert research analyst specializing in breaking down complex questions into a logical sequence
of simple, answerable sub-questions.
Your task is to decompose a given ’Original Question’ into a series of smaller, ordered sub-questions. This
decomposition will be used to query a retrieval system containing various factual reports.
Your Goal:Create a step-by-step reasoning path. The answer to a later sub-question should ideally
depend on or build upon the answer to a previous one, creating a logical chain.
Key Constraints:
(a)Logical Flow:The sub-questions must follow a logical order. The sequence should represent the
steps a human researcher would take to find the final answer.
(b)Self-Contained:Each sub-question must be understandable and answerable on its own.
(c)Fact-Focused:All sub-questions must be aimed at retrieving factual information from the reports.
Do not ask about the publication source or publisher unless it is essential for resolving ambiguity.
(d)Completeness:The combined answers to your sub-questions should contain all the information
necessary to answer the Original Question.
(e)No Direct Answers:Do not try to answer the Original Question yourself. Only generate the
sub-questions.
Wethenformacompactretrievalstring(formatted_query)foreachsub-questionusingthetemplate:
Context: {original_query} | Focus: {sub_question}
For the document corpus, we create a standardized text representation by sorting all key-value pairs
of a document’s JSON object and joining them into a single string with the format{key}:{value}on
20

each line. This approach ensures a consistent representation that includes all available information (e.g.,
category, title, body).
C.1.5 Common Preprocessing and Embedding Details
Before embedding, we apply light text normalization to all inputs, including stripping whitespace and
replacing newlines for API stability. If an input exceeds the model’s length limit, we progressively
truncate it (e.g., to 8192 characters and then shorter) and skip any samples that remain too long. The
output dimension for all embedding models is set to 1536.
C.2 Trainer
Our algorithm is implemented using PyTorch. We employ the AdamW optimizer with default parameters
and a learning rate schedule that decays proportionally to1/√
t, wheretis the training/update step.
Key hyperparameters, including the initial learning rate and batch size, were tuned via a Bayesian-
optimization-based grid search. The search space for each hyperparameter is detailed below:
•Initial Learning Rate (η 0):{1e-8,2e-8,5e-8,1e-7,2e-7,5e-7,1e-6,2e-6,5e-6,1e-5}
•Batch Size:{5,10,20,30,40,50}
C.3 Data enhancement
The main results presented in Section 4.1 utilize a data augmentation strategy where each query is
processed multiple times to accelerate convergence. We refer to this as the multiple exposure setting.
For a more realistic online deployment scenario, we also evaluate a single exposure setting where each
query is seen only once. Table 2 presents the results for this setting. For this experiment, we used the
same hyperparameters tuned for the multiple exposure setting. We note that performance could likely
be further improved by re-tuning the hyperparameters specifically for the single exposure scenario.
Table 2: Retrieval performance in the single exposure setting (no data augmentation). Our method
still consistently improves over the base dense retrieval models, albeit with smaller margins than in the
multiple exposure setting reported in the main paper.
MethodU-Tool FiQA T-Web T-Code T-Custom
R@10 N@10 R@10 N@10 R@10 N@10 R@10 N@10 R@10 N@10
text-embedding-v4 0.7451 0.5064 0.5335 0.4604 0.2701 0.1453 0.5291 0.3770 0.5066 0.4097
text-embedding-3-large 0.8356 0.6067 0.6258 0.5462 0.3243 0.1675 0.5347 0.3582 0.6378 0.5204
Ours (text-emb.-v4)0.7614
(+1.63%)0.5209
(+1.45%)0.5382
(+0.47%)0.4646
(+0.42%)0.2900
(+1.99%)0.1579
(+1.26%)0.5484
(+1.93%)0.3891
(+1.21%)0.5202
(+1.36%)0.4237
(+1.40%)
Ours (text-emb.-3-L.)0.8540
(+1.86%)0.6180
(+1.13%)0.6284
(+0.26%)0.5483
(+0.21%)0.3458
(+2.15%)0.1804
(+1.29%)0.5430
(+0.83%)0.3671
(+0.89%)0.6561
(+1.83%)0.5324
(+1.20%)
C.4 More details on the experiments
We offer more details of the experiments included in our main paper here.
Illustration experiments.The experiment illustrated in Figure 1a is conducted on a subset
of theToolRet-Codedataset. We only use the tool items whosedocumentationcolumn contains a
functionalitykey, where the corresponding value is a highly compact and ambiguous description of the
tool item. We refer to the full documentation as a “good documentation”, and the onlyfunctionality
description as a “bad documentation”. The visualization in Figure 1b is extracted from an experiment
21

run onUltraTooldataset in Table 1. We randomly sample 100 tool items and inspect their embeddings
and performance metrics across different settings.
Variant experiments.All the variant experiments are conducted without the data enhancement
techniques mentioned in Section C.3 to evaluate the practical performance under an online setting. Also,
considering the high costs of LLM-involved experiments, we did not fully tune the parameters during
experiments integrated with the LLM reranker, and only ran 1 round of them.
Varying database experiment.For the varying database experiment shown in Figure 5, we
provide half of the tools in the beginning, and only add the other half as available tools when half of the
queries are processed. The queries are not manipulated.
Multi-Hop retrieval experiment.Figure 8 provides a visual depiction of the combined online-
optimizing and inference workflow for the multi-hop retrieval experiments shown in Figure 6. The
multi-hop pipeline uses an LLM agent (backed bygpt-4o-mini-2024-07-18) for two key steps: (1)
reranking retrieved documents for each sub-task query, and (2) synthesizing a final answer from the
collected evidence, with the support of OpenAI’s JSON mode. The prompts for these steps are provided
below.
You are an impartial and meticulous AI judge. Your task is to determine which of the provided documents
contains useful information to answer the given question, especially the “Focus” one.
Carefully review each document and respond with a JSON object containing the 0-based index of the
relevant document. Smaller index is more relevant.
Question:
{question}
Retrieved Documents:
{formatted_docs}
Based on the question, which document is the most relevant?
You are a concise QA assistant. Given a main question and evidence documents, provide the final short
answer only. If uncertain, provide your best effort.
Main Question:
{question}
Evidence Documents:
{formatted_joined_docs}
Provide the final answer only with no explanation.
Regret analysis experiment.We perform our regret analysis on theToolRet-Codedataset,
where the result is displayed in Figure 7. To evaluate performance across different time horizons (T),
we truncate the query set to various lengths while keeping all other hyperparameters identical to those
used for the results in Table 1. Regret is calculated as the difference between the cumulative loss of our
online Algorithm 1 and an oracle baseline trained with full-information gradients (see Section 3), with
the cross-entropy loss being the loss function.
22

Get batch of sub-
task embeddings
Compute sam-
pling probabilities
Sample5candi-
date items per query
Fetch candidate documen-
tations and concatenate
JudgeLLM selects
best related item
Obtain feedback
by LLM selection
Compute gradient
Update embeddingsUpdate Loop
Item documentation
collected for each sub-task
Aggregate the documenta-
tions by the original query
Feed the query and
reference documenta-
tions to the LLM agent
for final prediction
Figure8: Workflowforthemulti-hopexperiment. TheleftpanelshowstheupdateloopforourAlgorithm
4, which leverages the decomposed sub-task query embeddings and optimizes the document embedding.
The right panel illustrates the inference process where, for each sub-query, retrieved documents are
collected and then synthesized by an LLM agent to produce the final answer.
D Appendix for Proofs
D.1 Proof for Lemma 3.1
Proof.The (full-information withi∗
tobservable) gradient ofl(Θ; (q t, i∗
t))with respect toθ iis
∂l(Θ; (q t, i∗
t))
∂θi
Θ=Θ t= 
pt,i−1{i=i∗
t}
qt,
wherep t,iis defined in (3).
By the definitions, for anyiwe have
Eit[gt,i] =IX
it=1pt,it·
pt,i−1{i=i t}1{it=i∗
t}
pt,it
qt
=pt,i·
1−1{i=i∗
t}
pt,i
qt
= (p t,i−1{i=i∗
t})qt
23

D.2 Proof of Theorem 5.1
Proof.The proof follows a standard regret analysis for online convex optimization. Let⟨·,·⟩ Fdenote
the Frobenius inner product and∥ · ∥ Fthe Frobenius norm. Define the gradient (estimator) matrices
Gt= [g t,1, . . . ,g t,I]⊤and ˜Gt= [˜gt,1, . . . , ˜gt,I]⊤, whereg t,iis as in Section 3 and
˜gt,i=∂l(Θ; (q t, i∗
t))
∂θi
Θ=Θ t
is the (full-information withi∗
tobservable) gradient ofl(Θ; (q t, i∗
t))with respect toθ i.
By the update in Algorithm 1, for eacht,
∥Θt+1−Θ∗∥2
F=∥Θ t−ηG t−Θ∗∥2
F
=∥Θ t−Θ∗∥2
F−2η⟨G t,Θt−Θ∗⟩F+η2∥Gt∥2
F.
Summing and rearranging yields
TX
t=1⟨Gt,Θt−Θ∗⟩F=TX
t=1∥Θt−Θ∗∥2
F− ∥Θ t+1−Θ∗∥2
F
2η+η
2TX
t=1∥Gt∥2
F
=∥Θ1−Θ∗∥2
F− ∥Θ T+1−Θ∗∥2
F
2η+η
2TX
t=1∥Gt∥2
F
≤∥Θ1−Θ∗∥2
F
2η+η
2TX
t=1∥Gt∥2
F.
Becausel(Θ; (q t, i∗
t))is convex inΘ, for anyt(conditioning onq t, i∗
t,Θt),
l(Θt; (qt, i∗
t))−l(Θ∗; (qt, i∗
t))≤ ⟨ ˜Gt,Θt−Θ∗⟩F=
E[G t],Θt−Θ∗
F,
where the equality uses the unbiasednessE[G t] =˜Gtfrom Lemma 3.1 (the expectation is overi t∼pt
given the history). Summing overtand applying the previous bound gives
TX
t=1Eh
l(Θt; (qt, i∗
t))−l(Θ∗; (qt, i∗
t))i
≤∥Θ1−Θ∗∥2
F
2η+η
2TX
t=1E
∥Gt∥2
F
.
It remains to boundE
∥Gt∥2
F
. By definition,∥G t∥2
F=PI
i=1∥gt,i∥2
2, and
E
∥Gt∥2
F
=IX
i=1E
∥gt,i∥2
2
=∥q t∥2
2IX
i=1Eit"
pt,i−1{i=i t}1{i t=i∗
t}
pt,it2#
=∥q t∥2
2IX
i=1pt,i 
p2
t,i−21{i=i∗
t}+1{i=i∗
t}
p2
t,i!
≤1
pt,i∗
t−2p t,i∗
t+ 1
∥qt∥2
2.
Plugging this into the previous inequality establishes the claimed bound.
24

D.3 Proof of Corollary 5.2
Proof.Under the assumptions given with
η=s
p¯Θ
¯q(1−p )(1 + 2p )T,
the corollary is a direct result of Theorem 5.1.
25