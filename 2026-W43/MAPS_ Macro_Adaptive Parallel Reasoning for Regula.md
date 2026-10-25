# MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in Retrieval-Augmented Generation

**Authors**: Xinze Li, Dingwen Zhang, Zhenghao Liu, Yukun Yan, Shuo Wang, Yu Gu, Ge Yu, Maosong Sun

**Published**: 2026-10-21 14:19:17

**PDF URL**: [https://arxiv.org/pdf/2610.29792v1](https://arxiv.org/pdf/2610.29792v1)

## Abstract
Retrieval-Augmented Generation (RAG) improves the performance of Large Language Models (LLMs) by retrieving external knowledge. Recently, some RAG methods interleave reasoning and retrieval, enabling LLMs to progressively generate sub-queries and retrieve external knowledge. However, these methods may suffer from redundant or ineffective sub-queries and insufficiently diverse query exploration. Moreover, due to the tightly interleaved nature of reasoning and retrieval, it remains unclear how the evolving reasoning process influences query intent. To characterize the evolution of query intent during reasoning, we introduce search momentum, which captures the accumulated retrieval tendency induced by historical changes in generated sub-queries along the reasoning trajectory. Through momentum-based analysis, we find that the query generation process tends to continuously evolve along existing retrieval trajectories. Based on this observation, we propose MAPS, a Macro-Adaptive Parallel Reasoning framework for regulating search momentum and mitigating query-intent drift. MAPS first retains the retrieved knowledge and a refined reasoning state, thereby reducing the influence of historical reasoning trajectories on subsequent sub-query generation. Subsequently, MAPS extends the conventional sequential reasoning process into a parallel reasoning paradigm. At each iteration, MAPS identifies complementary retrieval directions, generates corresponding sub-queries for parallel retrieval, and consolidates the retrieved evidence to update the knowledge and reasoning states. Experiments on six question-answering benchmarks demonstrate that MAPS consistently outperforms representative RAG methods across different backbone models. All code and resources are available at https://github.com/NEUIR/MAPS.

## Full Text


<!-- PDF content starts -->

Arxiv Preprint.
MAPS: MACRO-ADAPTIVEPARALLELREASON-
ING FORREGULATINGSEARCHMOMENTUM IN
RETRIEVAL-AUGMENTEDGENERATION
Xinze Li1∗, Dingwen Zhang1∗, Zhenghao Liu1†, Yukun Yan2, Shuo Wang2,
Yu Gu1, Ge Yu1, Maosong Sun2
1School of Computer Science and Engineering, Northeastern University, Shenyang, China
2Department of Computer Science and Technology, Tsinghua University, Beijing, China
ABSTRACT
Retrieval-Augmented Generation (RAG) improves the performance of Large Lan-
guage Models (LLMs) by retrieving external knowledge. Recently, some RAG
methods interleave reasoning and retrieval, enabling LLMs to progressively gen-
erate sub-queries and retrieve external knowledge. However, these methods may
suffer from redundant or ineffective sub-queries and insufficiently diverse query
exploration. Moreover, due to the tightly interleaved nature of reasoning and re-
trieval, it remains unclear how the evolving reasoning process influences query
intent. To characterize the evolution of query intent during reasoning, we intro-
duce search momentum, which captures the accumulated retrieval tendency in-
duced by historical changes in generated sub-queries along the reasoning trajec-
tory. Through momentum-based analysis, we find that the query generation pro-
cess tends to continuously evolve along existing retrieval trajectories. Based on
this observation, we propose MAPS, a Macro-Adaptive Parallel Reasoning frame-
work for regulating search momentum and mitigating query-intent drift. MAPS
first retains the retrieved knowledge and a refined reasoning state, thereby re-
ducing the influence of historical reasoning trajectories on subsequent sub-query
generation. Subsequently, MAPS extends the conventional sequential reasoning
process into a parallel reasoning paradigm. At each iteration, MAPS identifies
complementary retrieval directions, generates corresponding sub-queries for par-
allel retrieval, and consolidates the retrieved evidence to update the knowledge
and reasoning states. Experiments on six question-answering benchmarks demon-
strate that MAPS consistently outperforms representative RAG methods across
different backbone models. All code and resources are available athttps:
//github.com/NEUIR/MAPS.
1 INTRODUCTION
Large Language Models (LLMs) (Jaech et al., 2024; Guo et al., 2025) have demonstrated remark-
able capabilities in knowledge-intensive question answering and complex reasoning. However, they
remain susceptible to hallucinations (Jiang et al., 2023; Ji et al., 2023), leading to factually incor-
rect or unsupported responses. Retrieval-Augmented Generation (RAG) (Lewis et al., 2020; Izacard
et al., 2023) addresses this issue by enabling LLMs to access external knowledge during infer-
ence, thereby improving the factuality and reliability of their responses. However, for multi-hop
queries (Yang et al., 2018), the required evidence is often distributed across multiple documents and
interconnected through dependencies, making single-step retrieval insufficient to comprehensively
acquire the information needed to answer the query (Fang et al., 2025; Wu et al., 2026).
To address this limitation, recent RAG approaches have explored leveraging the reasoning capabil-
ities of LLMs to adaptively acquire external evidence. These methods (Li et al., 2025a; Jin et al.,
2025; Song et al., 2025) interleave reasoning and retrieval, allowing LLMs to identify missing in-
∗Equal Contribution.
†Corresponding Authors.
1
arXiv:2610.29792v1  [cs.CL]  21 Oct 2026

Arxiv Preprint.
formation, generate sub-queries, and retrieve relevant documents to progressively support the orig-
inal query. Specifically, an LLM is prompted to generate sub-queries when additional knowledge
is necessary, retrieve relevant documents, and incorporate the retrieved evidence into subsequent
reasoning. However, as reasoning and retrieval proceed iteratively, these methods can suffer from
redundant or ineffective sub-queries, excessively long retrieval chains, or insufficient exploration
diversity (Ji et al., 2025; Martinez et al., 2025), ultimately degrading retrieval and downstream rea-
soning performance. Due to the tightly interleaved nature of reasoning and retrieval, it remains
unclear how the evolving reasoning process influences the search direction, particularly when com-
prehensive evidence requires exploration beyond the current retrieval trajectory.
To characterize how search intentions evolve throughout the reasoning process, we introducesearch
momentum, which captures the accumulated retrieval tendency induced by historical changes in gen-
erated sub-queries along the reasoning trajectory. At each step, we update the search momentum by
aggregating the current sub-query with the accumulated momentum through a stepwise learnable
weight. We learn these weights by using the updated momentum to predict the query-intent drift
at the current step. By analyzing the resulting search momentum across reasoning steps, we ob-
serve that the deviation directions of subsequent sub-queries become increasingly aligned with the
accumulated search momentum as retrieval proceeds deeper. This finding suggests that query gener-
ation tends to follow the existing retrieval trajectory, rather than correcting or escaping accumulated
deviations through subsequent reasoning (Guo et al., 2025). Such momentum-driven behavior can
progressively amplify early search deviations, causing the retrieval process to drift away from the
original information need. This motivates us to regulate the accumulation of search momentum and
preserve alignment between the evolving retrieval process and the original query intent.
In this paper, we propose MAPS, a Macro-Adaptive Parallel Reasoning framework for regulating
search momentum and mitigating query-intent drift. MAPS introduces a parallel reasoning paradigm
that reduces the risk of search momentum being dominated by a single deviated retrieval direction,
thereby maintaining better alignment with the original query intent. Specifically, MAPS first disen-
tangles reasoning traces from the sub-queries that propagate historical retrieval tendencies, retaining
only the refined retrieval knowledge and reasoning state to reduce the influence of historical search
deviations on subsequent query generation. It then extends the conventional sequential reasoning
process into a parallel reasoning paradigm, enabling the model to calibrate the search direction and
construct a more robust search momentum. Specifically, MAPS identifies multiple complementary
retrieval directions from the current reasoning state and generates corresponding sub-queries in par-
allel to explore these directions. The resulting evidence is consolidated to update the accumulated
knowledge and refined reasoning state, which serves as the basis for the next reasoning iteration.
Our experiments on multiple datasets demonstrate that MAPS achieves significant improvements
over all baseline methods across different backbone models, highlighting its strong generalization
ability and robustness. Further analyses show that MAPS effectively regulates the evolution of
search momentum, preventing it from being dominated by a single retrieval direction. Our experi-
ments further demonstrate that MAPS effectively mitigates query intent drift and maintains stronger
relevance between generated sub-queries and supporting evidence throughout iterative reasoning.
Moreover, MAPS enables the model to acquire the prerequisite knowledge required for reasoning at
an earlier stage, while reducing unnecessary retrieval iterations.
2 ANALYZINGQUERYINTENTDRIFTDURINGLLM REASONING
Recent RAG methods (Li et al., 2025a) enable LLMs to actively retrieve relevant documents and
incorporate them into the reasoning process, allowing them to progressively acquire the information
required to answer a given query. However, the sub-queries generated during this iterative process
do not necessarily remain aligned with the original information need. This is because subsequent
sub-query generation depends on the preceding retrieval trajectory, which may contain sub-queries
deviating from the original information need and noisy retrieved knowledge, thereby influencing the
intent of subsequent sub-queries and ultimately degrading downstream reasoning performance (Niu
et al., 2026). To characterize how the history retrieval trajectory influences the subsequent query
intent, we first formalize the accumulated change in retrieval intent along the reasoning trajectory
assearch momentum, and then investigate whether it can serve as a predictor of subsequent query
intent drift.
2

Arxiv Preprint.
Table 1: Fitted Momentum Coefficients and Directional Alignment Between Search Momentum
and Subsequent Query Intent Deviations.S tdenotes the mean directional alignment scoreS tacross
trajectories in the independent test set.β tcontrols the retention of previously accumulated retrieval
tendencies when constructing search momentum. The Pooled-MHQA fits the momentum coeffi-
cients on the mixed training data from the three datasets and evaluates directional alignment on the
corresponding test sets. Detailed experimental settings and results are provided in Appendix A.1.
Datasetβ 1 S1 β2 S2 β3 S3 β4 S4 Avg.S t
2WikiQA 0.0000 0.4051 0.6352 0.5698 0.7552 0.6483 0.8761 0.6955 0.5797
MuSiQue 0.0000 0.3105 0.5400 0.5196 0.7214 0.6489 0.8252 0.6913 0.5426
HotpotQA 0.0000 0.4427 0.6161 0.6840 0.8151 0.7731 0.8836 0.7750 0.6687
Pooled-MHQA 0.0000 0.3875 0.5959 0.5651 0.7559 0.6774 0.8630 0.7279 0.5895
Uncovering Search Momentum in LLM Reasoning.To characterize how retrieval intent evolves
during iterative reasoning, we introducesearch momentum, which captures the accumulated seman-
tic shift in search behavior across reasoning steps. Given an initial queryq 0, suppose that the LLM
has completed the reasoning trajectory up to the(t−1)-th step:
Rt−1= (r 1, q1, o1, . . . , r t−1, qt−1, ot−1),(1)
wherer idenotes the reasoning step,q iis the generated sub-query, ando idenotes the knowledge
observation obtained from the retrieved evidence. Conditioned on the initial queryq 0and the pre-
ceding reasoning trajectoryR t−1, the LLMMiteratively generates the next reasoning step and
sub-queryq tuntil reaching the maximum number of reasoning stepsT:
rt, qt=M(q 0, Rt−1).(2)
The generated sub-queryq tis then used to retrieve a set of documentsD t, from which the LLM
derives the corresponding observationo t. The reasoning trajectory is subsequently updated as:
Rt=Rt−1⊕(r t, qt, ot).(3)
We represent thet-th step queryq tby its semantic embeddingq tand characterize the evolution of
retrieval intent through the semantic displacement between these sub-queries{q 0, q1, . . . , q T}:
qt= Emb(q t),(4)
where Emb(·)denotes the embedding model, which can be instantiated with the E5 model (Wang
et al., 2022). We then define the search momentumm tat thet-th reasoning step as:
mt=βtmt−1+ (1−β t)(qt−qt−1),(5)
wherem 0=0and0≤β t<1. The coefficientβ tcontrols how strongly previously accumulated
search dynamics are retained when incorporating the latest semantic shift. In this formulation,m t
is an exponentially weighted aggregation of historical query movements along the trajectoryq 0→
··· →q t. Its magnitude reflects the strength of the accumulated semantic shift, while its direction
characterizes the predominant direction in which the retrieval intent has evolved.
Tracking Query Intent Drift with Search Momentum.After establishing search momentum as an
explicit representation of accumulated query movement, we next examine whether it can predict the
direction of subsequent query drift during LLM reasoning. Specifically, ifm tis consistently aligned
with the deviation of the next sub-query from the original query, it suggests that historical search
dynamics are propagated to subsequent query generation and that search momentum can serve as an
effective signal for tracking intent drift.
For each reasoning trajectory, we therefore measure the directional alignment between the search
momentumm tand the deviation of the subsequent sub-queryq t+1from the original queryq 0:
St= cos(m t,qt+1−q0),(6)
whereq t+1−q0represents the semantic deviation of the next sub-query from the original infor-
mation need. A largerS tindicates that the accumulated search dynamics captured bym tare more
3

Arxiv Preprint.
consistent with the direction in which the subsequent query deviates from the original query, sug-
gesting stronger predictive correspondence between search momentum and future query drift.
To obtain the momentum at each reasoning step, we fit the coefficientβ tin Eq. 5 by maximizing
the corresponding directional alignmentS t. Sinceβ tis constrained to[0,1), we solve the resulting
bounded optimization problem using the L-BFGS-B quasi-Newton method (Byrd et al., 1995). Ta-
ble 1 reports the momentum coefficients fitted on the training set and the resulting directional align-
ment scoresS tevaluated on a held-out set. The complete procedure is provided in Appendix A.1.
We observe a consistent increase in bothβ tandS tas reasoning progresses. The increasingβ tindi-
cates that later reasoning steps place progressively greater weight on previously accumulated search
dynamics, suggesting that historical queries exert an increasingly persistent influence on subsequent
query generation as the retrieval process becomes deeper. Meanwhile, the increasingS tshows that
the next query becomes more aligned with the current search momentum, suggesting that the rea-
soning process tends to perpetuate the existing search trajectory rather than effectively correcting or
escaping it through reflection (Guo et al., 2025). This observation suggests that the reasoning pro-
cess does not naturally eliminate the accumulated search momentum through subsequent reasoning
steps. Taken together, these observations motivate us to investigate how to regulate the accumulation
and propagation of search momentum, with the goal of reducing the persistent influence of historical
search intents and maintaining better alignment with the original information need.
3 MAPS: REGULATINGINTENTDRIFT THROUGHPARALLELREASONING
In this section, we propose MAPS, a Macro-Adaptive Parallel Reasoning-based RAG framework
for regulating search momentum. MAPS introduces a parallel reasoning paradigm to reduce the
influence of accumulated search momentum and maintain better alignment with the original query
intent. Specifically, MAPS first disentangles historical reasoning traces and sub-queries that may
propagate previously established retrieval tendencies, thereby reducing their influence on subsequent
query generation. It then employs parallel reasoning to explore multiple complementary search
directions, enabling the retrieval process to calibrate its search direction at an early stage and further
mitigate query intent drift.
Reasoning Disentanglement for Search Momentum Regulation.Given an initial queryq 0, exist-
ing iterative RAG methods (Li et al., 2025a) maintain a reasoning trajectoryR t−1accumulated up
to the(t−1)-th iteration:
Rt−1= (r 1, q1, o1, . . . , r t−1, qt−1, ot−1),(7)
wherer i,qi, ando idenote the reasoning trace, generated sub-query, and knowledge observation
obtained from retrieval at thei-th iteration, respectively. As discussed in Section 2, historical sub-
queries can exert a stronger influence on subsequent query generation than the reasoning process
itself, causing later queries to inherit and progressively reinforce previously established retrieval
tendencies. Such trajectory-dependent propagation can consequently lead to query intent drift.
To mitigate this effect, MAPS disentangles the accumulated knowledge from the historical retrieval
trajectory by not explicitly preserving the reasoning tracesr iand sub-queriesq iinRt−1. Instead, it
retains the knowledge acquired over the firstt−1iterations as a knowledge observation set:
Ot−1= (o 1, . . . , o t−1).(8)
Meanwhile, rather than retaining the complete sequence of historical reasoning tracesr 1:t−1 =
(r1, . . . , r t−1), MAPS maintains a compact refined reasoning state˜r t−1that summarizes the es-
sential reasoning information accumulated from previous iterations. This state is updated at each
iteration while avoiding explicit exposure to the historical sequence of sub-queries.
Accordingly, at thet-th iteration, the subsequent reasoning generated by the modelM Reason is
conditioned on the original queryq 0, the accumulated knowledge observationsO t−1, and the refined
reasoning state˜r t−1, rather than the complete historical trajectoryR t−1:
(rt, qt, ot) =M Reason (q0, Ot−1,˜rt−1),(9)
where˜r t−1provides the reasoning information necessary for the current iteration, whileO t−1pre-
serves the knowledge acquired from previous retrievals. By removing the explicit dependency on
4

Arxiv Preprint.
historical sub-queries and reasoning traces, MAPS reduces the propagation of previously accumu-
lated retrieval tendencies into subsequent query generation.
The reasoning process terminates when the accumulated knowledgeO t= (o 1, . . . , o t−1, ot)satis-
fies the current information needs, or when the maximum number of iterations is reached. After the
T-th iteration, the LLM generates the final answerybased on the original query, the accumulated
knowledge observations, and the refined reasoning state:
y=M Reason (q0, OT,˜rT),(10)
whereTdenotes the termination iteration or the maximum number of allowed reasoning iterations.
Parallel Reasoning for Search Direction Calibration.To further regulate the search direction at
thet-th iteration, MAPS introduces parallel reasoning to explore multiple complementary retrieval
intents before generating the sub-queries. Specifically, we extend the sequential reasoning process
in Eq. 9 to a parallel reasoning paradigm, where multiple retrieval directions are first identified and
subsequently instantiated into independent sub-queries.
First, instead of directly generating a sub-query from the current reasoning state, we employ the
search direction proposal modelM Intent to identify multiple complementary retrieval directions:
{s1
t, . . . , sN
t}=M Intent (q0, Ot−1,˜rt−1),(11)
where each retrieval direction is instantiated as an individual sub-query. We then collect the gener-
ated sub-queries into a query setQ t={q1
t, . . . , qN
t}. Unlike sequential query generation, in which
each query is directly conditioned on the preceding retrieval trajectory, parallel query generation
explores multiple retrieval directions from the same reasoning state. This expands the search space
at an early stage and increases the likelihood of retrieving evidence that addresses the original infor-
mation needs, thereby reducing the risk that the search momentum becomes dominated by a single
deviated retrieval direction.
Next, each sub-queryqi
t∈Q tindependently retrieves the top-Kdocuments from the document
collectionD. The retrieved documents are then aggregated into a unified evidence set:
qi
t=M Query (q0, si
t,˜rt−1),(12)
where each retrieval direction is associated with a generated sub-query. Subsequently, MAPS col-
lects the sub-queries generated from all retrieval directions to form the query setQ t={q1
t, . . . , qN
t}.
By generating multiple sub-queries in parallel, MAPS expands the search space at an early stage and
increases the opportunity to generate sub-queries that retrieve useful evidence for the original query.
Then, we use each sub-queryqi
tindependently retrieves the top-Kdocuments from the document
collectionDand collect all retrieved documents:
Dt=[
qi
t∈QtTop-K(qi
t,D).(13)
Finally, the refinement agentM Refine consolidates the retrieved evidence from different retrieval
directions into a knowledge observationo tand updates the refined reasoning state from˜r t−1to˜rt:
ot,˜rt=M Refine (q0,{s1
t, . . . , sN
t}, Qt,Dt, Ot−1,˜rt−1).(14)
The resulting knowledge observationo tand refined reasoning state˜r tare then regarded as the out-
puts of Eq. 9 to support subsequent reasoning iterations.
4 EXPERIMENTALMETHODOLOGY
This section first describes datasets and baselines. Then, we introduce the implementation details of
our experiments.
Datasets.We evaluate different methods on three categories of benchmarks covering different task
scenarios. For single-hop QA tasks, we use NQ (Kwiatkowski et al., 2019) and PopQA (Mallen
et al., 2023). For multi-hop QA tasks, we evaluate MAPS on 2WikiQA (Ho et al., 2020), Hot-
potQA (Yang et al., 2018), and MuSiQue (Trivedi et al., 2022). For reasoning QA tasks, following
previous work (Li et al., 2025a), we evaluate on the Diamond subset of GPQA (Rein et al., 2023),
5

Arxiv Preprint.
Table 2: Overall Performance of MAPS across All Benchmarks. The best results are inboldand the
second-best are underlined .
Method NQ PopQA 2WikiQA HotpotQA MuSiQue GPQA Avg.
Qwen3-32B
Vanilla LLM 25.0 19.4 27.3 21.3 7.1 44.4 24.1
Vanilla RAG 49.0 52.1 30.0 38.3 9.9 44.2 37.2
Self-RAG 39.2 37.9 29.8 31.5 17.2 45.1 33.4
IRCOT 38.2 37.6 38.8 33.2 22.3 54.2 37.4
Search-o1 53.1 56.5 43.7 40.0 23.7 46.4 43.9
S3 59.5 60.6 37.2 38.3 19.4 60.7 45.9
ParallelSearch 57.2 60.4 53.6 43.5 30.9 49.8 49.2
MAPS63.4 64.8 59.4 48.8 37.6 62.1 56.0
Qwen3.6-35B-A3B
Vanilla LLM 39.4 39.4 42.1 34.8 19.4 76.8 42.0
Vanilla RAG 55.6 61.4 49.1 45.5 26.8 75.7 52.3
Self-RAG 45.5 52.3 44.8 39.7 27.3 67.6 46.2
IRCOT 49.0 45.9 45.8 43.2 28.6 64.7 46.2
Search-o1 53.0 62.0 56.3 44.0 34.1 76.3 54.3
S3 57.4 59.8 34.5 41.9 18.9 73.7 47.7
ParallelSearch 58.4 56.8 55.9 53.7 41.5 77.5 57.3
MAPS64.1 69.3 65.8 55.1 45.2 79.2 63.1
which contains198graduate-level science questions. Except for GPQA, we randomly sample1,000
questions from each dataset for evaluation.
Baselines.We compare MAPS with multiple representative baselines. Vanilla LLM prompts the
LLM to generate answers without external knowledge, and Vanilla RAG (Lewis et al., 2020) utilizes
the original query to retrieve documents and generates answers based on them. For adaptive RAG
models, Self-RAG (Asai et al., 2024) prompts the LLM to determine whether retrieval is needed
and whether retrieved documents are used during decoding. For reasoning-driven RAG models,
IRCoT (Trivedi et al., 2023) interleaves reasoning with retrieval, where retrieved documents are in-
corporated into subsequent reasoning. Search-o1 (Li et al., 2025a) further extends this paradigm by
introducing agentic query generation and the Reason-in-Documents module. For sub-query opti-
mization RAG models, S3 (Jiang et al., 2025) trains a lightweight model to generate more effective
sub-queries. ParallelSearch (Zhao et al., 2025) iteratively decomposes the original query into multi-
ple complementary sub-queries and retrieves them in parallel to explore diverse knowledge.
Implementation Details.We use Qwen3-32B (Yang et al., 2025) and Qwen3.6-35B-A3B (Qwen
Team, 2026) as the backbone models. The retrieval corpus is the English Wikipedia December 2018
dump (Karpukhin et al., 2020), and we employ E5-base-v2 (Wang et al., 2022) as the dense retriever
with FAISS (Douze et al., 2026) for efficient similarity search over the pre-built index. Each retrieval
step returns the top-5 most relevant passages, and the maximum number of execution iterations is
set to 10. Following previous work (Song et al., 2025; Li et al., 2026), we adopt Cover-EM as the
primary evaluation metric.
5 EVALUATIONRESULTS
5.1 OVERALLPERFORMANCE
As shown in Table 2, we compare MAPS with a range of representative baselines across different
tasks. Overall, MAPS consistently outperforms all baselines, with improvements of 6.8% and 5.8%
on Qwen3-32B and Qwen3.6-35B-A3B, demonstrating the effectiveness of our parallel reasoning
paradigm for mitigating query intent drift. Notably, MAPS delivers consistent gains across different
backbone models, indicating its robustness and generalizability across model architectures.
6

Arxiv Preprint.
Table 3: Ablation Study. The best results are inboldand the second-best are underlined .
Method NQ PopQA 2WikiQA HotpotQA MuSiQue GPQA Avg.
Qwen3-32B
MAPS63.4 64.8 59.4 48.8 37.6 62.3 56.1
w/o Retrieval Direction 63.0 64.3 55.8 46.9 35.6 46.0 51.9
w/o Parallel Reasoning 60.1 61.2 54.1 45.2 33.1 45.3 49.8
w/o Disentanglement 61.0 58.8 53.1 48.3 34.3 45.1 50.1
Qwen3.6-35B-A3B
MAPS64.1 69.3 65.8 55.1 45.2 79.2 63.1
w/o Retrieval Direction 62.3 69.2 62.2 51.6 43.1 68.1 59.4
w/o Parallel Reasoning 56.3 66.8 62.8 49.8 39.5 63.8 56.5
w/o Disentanglement 54.0 64.9 64.4 52.9 42.6 78.1 59.5
Compared with Naive-RAG, MAPS achieves substantial improvements, showing that reasoning-
guided iterative retrieval can acquire more comprehensive knowledge than one-pass retrieval. More
importantly, compared with iterative reasoning-based RAG methods such as Search-o1, MAPS im-
proves performance by over 9%. This result suggests that repeatedly extending the retrieval process
along a single reasoning trajectory can reinforce previously explored retrieval directions, whereas
parallel retrieval provides complementary search directions and reduces the risk of search momen-
tum being dominated by a single trajectory. In addition, MAPS consistently outperforms Paral-
lelSearch across all datasets. This comparison shows that simply generating multiple sub-queries
to broaden the search space is insufficient to fully mitigate the influence of accumulated retrieval
tendencies on subsequent query generation. In contrast, MAPS explicitly explores complemen-
tary retrieval directions while retaining only the retrieved knowledge, rather than the full historical
reasoning trajectory, as the context for subsequent retrieval. By decoupling subsequent query gener-
ation from the accumulated reasoning trajectory, MAPS effectively mitigates search intent drift and
enables more diverse and targeted knowledge acquisition.
5.2 ABLATIONSTUDY
In this subsection, we conduct ablation studies to evaluate the effectiveness of different components
in MAPS. In the experiments, we compare MAPS with three ablated models. MAPS w/o Retrieval
Direction directly generates multiple sub-queries in parallel based on the accumulated knowledge
and refined reasoning state, without first identifying multiple retrieval directions to guide sub-query
generation. MAPS w/o Parallel Reasoning replaces the multi-directional parallel reasoning mech-
anism with sequential single-query generation and retrieval, which generates only one retrieval di-
rection and its corresponding sub-query at each iteration. MAPS w/o Disentanglement does not dis-
entangle the retrieval trajectory, but instead preserves the complete reasoning traces and sub-queries
generated at each iteration for subsequent reasoning.
As shown in Table 3, MAPS consistently outperforms all ablated variants, demonstrating the effec-
tiveness of the proposed framework. Specifically, compared with MAPS w/o Retrieval Direction,
MAPS achieves over a 4% improvement, indicating that explicitly identifying complementary re-
trieval directions helps guide parallel retrieval toward diverse information needs. MAPS also outper-
forms MAPS w/o Parallel Reasoning, demonstrating that the parallel reasoning mechanism expands
the search space to acquire supporting evidence at early stages, thereby preventing the subsequent
retrieval process from being dominated by a single retrieval direction and causing query intent drift.
Moreover, comparing MAPS w/o Disentanglement with MAPS, we observe additional gains, sug-
gesting that retaining the complete historical retrieval trajectory introduces stronger dependencies on
previous retrieval tendencies, propagating accumulated deviations to subsequent query generation.
5.3 EVOLUTION OFSEARCHMOMENTUM DURINGITERATIVEREASONING
As shown in Figure 1, we analyze the evolution of search momentum across different methods
and investigate whether MAPS can effectively regulate its directional drift. In this experiment, we
7

Arxiv Preprint.
Component 1Component 21
2
3
4 Iteration
Original Question
Search-o1
MAPS
(a) Search Momentum of MAPS and
Search-o1.
Component 1Component 21
2
3
4 Iteration
Original Question
MAPS
w/o Disentanglement
(b) Search Momentum of MAPS and
MAPS w/o Disentanglement.
0.00 -0.05 -0.10 -0.15 -0.20 -0.25 -0.30 -0.35 -0.40
Cumulative offset of ΔSm
t1
2
3
4Momentum iteration t
w/o Disentanglement
Search-o1
MAPS(c) Cumulative Directional Off-
set of Search Momentum.
Figure 1: Evolution of Search Momentum during Iterative Reasoning. Figures 1a and 1b visualize
the trajectories of search momentum through t-SNE. Figure 1c shows the cumulative directional
offset of search momentum with respect to the original query across reasoning iterations.
randomly sample 500 instances from all benchmark datasets. Detailed experimental settings and
additional results are provided in Appendix A.2.
As shown in Figures 1a and 1b, we apply t-SNE to jointly project the embeddings of the orig-
inal queries and their corresponding search momentum at different reasoning steps into a two-
dimensional space. Compared with MAPS, Search-o1 exhibits a larger directional deviation of
search momentum from the original query. This observation suggests that, compared with the
reasoning-based RAG method, MAPS mitigates the tendency of search momentum to become dom-
inated by a single retrieval direction through its parallel reasoning mechanism. Furthermore, MAPS
w/o Disentanglement exhibits weaker alignment between search momentum and the original query
than MAPS, suggesting that disentangling accumulated knowledge from the reasoning states and
generated sub-queries in the historical reasoning trajectory can reduce the influence of historical
retrieval tendencies on the evolution of search momentum. Finally, for Figure 1c, we compute the
change in cosine similarity between the search momentum and the original query across consecutive
iterations and accumulate these changes to quantify the directional deviation of search momentum.
The results demonstrate the effectiveness of MAPS in calibrating the search direction through paral-
lel retrieval and reasoning disentanglement during reasoning. As the reasoning process progresses,
the search momentum of MAPS becomes more stable than that of the other methods, indicating that
MAPS can better maintain coherent search directions established in the early reasoning steps.
5.4 ANALYSIS OFMAPS RETRIEVALBEHAVIOR DURINGITERATIVEREASONING.
01234≥ 5Search-o1
0
1
2
3
4
≥ 5
01234≥ 5
MAPS0
1
2
3
4
≥ 5100.0 86.9 83.2 80.1 78.5 78.1
100.0 100.0 86.6 82.8 80.3 79.9
89.9 100.0 100.0 85.4 81.0 81.3
84.7 85.9 100.0 100.0 83.8 83.6
85.2 86.4 90.0 100.0 100.0 88.5
84.8 86.0 89.8 90.4 100.0 100.0
84.4 85.6 89.2 90.1 90.5 100.0
(a) Similarity between
Queries.
1 2 3 4≥5
Iteration0.760.780.800.82Similarity
MAPS
Search-o1(b) Similarity between
Queries and Golden
Evidence.
Figure 2: Evolution of Query Similarity
and Query-Evidence Similarity during Iterative
Reasoning.In this section, we further analyze the retrieval be-
havior of MAPS during iterative reasoning. We
use a pooled set of all datasets with ground-truth
documents.
As shown in Figure 2, we first analyze how the
similarity between sub-queries and the original
query, as well as the golden evidence, evolves
across iterations. As shown in Figure 2a, we
compute the cosine similarity between the queries
generated at different reasoning iterations and the
original query. The result shows that, during the
early stages of reasoning, the similarity between
the generated sub-queries and the original query
decreases rapidly for both Search-o1 and MAPS,
reflecting the need to explore additional informa-
tion beyond the original query. However, as rea-
soning progresses, the sub-queries generated by
MAPS consistently maintain higher similarity with the original query than those generated by
8

Arxiv Preprint.
Search-o1 and gradually become more stable. This suggests that MAPS calibrates search momentum
at the early stages through parallel reasoning and reasoning disentanglement, enabling subsequently
generated sub-queries to remain more closely aligned with the information need expressed by the
original query. As shown in Figure 2b, for each sample, we compute the similarity between the sub-
query generated at each iteration and each of its golden evidence, and then average these similarity
scores. The results show that the similarity between the sub-queries generated by MAPS and the
ground-truth evidence decreases during the early stages, but subsequently recovers and stabilizes.
In contrast, the similarity for Search-o1 continues to decline as reasoning progresses. This indi-
cates that, after exploring different retrieval directions through parallel reasoning at the early stages,
MAPS can progressively calibrate the query direction, enabling subsequent queries to more stably
follow retrieval directions that are aligned with the target evidence.
MAPS Search-o1 ParallelSearch12345
Termination Step
Average Step
1.8%62.7%15.3%5.3%14.9%
10.2%50.4%20.8%6.2%12.5%
13.3%58.5%19.1%3.9%5.1%
(a) Termination Ratio at
Each Iteration.
1 2 3 4≥5
T ermination Step0.10.20.30.40.50.60.7 Cover EM
MAPS Search-o1 ParallelSearch(b) Performance by Ter-
mination Step.
Figure 3: Termination Distribution and Answer
Performance across Different Methods.As shown in Figure 3, we further analyze at which
iteration different methods terminate during iter-
ative reasoning and their corresponding answer
performance. Figure 3a reports the proportion of
samples terminating at each iteration for differ-
ent methods. The results show that MAPS has a
larger proportion of samples terminating at early
iterations and also exhibits a higher proportion of
samples requiring five or more retrieval rounds.
This suggests that, for simple queries, MAPS
can calibrate the query generation at early stages
through parallel reasoning, enabling the model to
rapidly acquire sufficient evidence with fewer it-
erations and terminate in a timely manner. For
complex questions, MAPS regulates search mo-
mentum to reduce the accumulation of historical retrieval tendencies, allowing subsequent queries
to continue exploring relevant knowledge. Figure 3b further compares the answer performance of
models at different termination iterations. The results show that MAPS maintains strong answer
performance across different termination steps. This suggests that its early termination typically
occurs after sufficient evidence has been acquired, rather than resulting from erroneous stopping
caused by retrieval drift. Meanwhile, MAPS continues to achieve higher performance at later termi-
nation steps, indicating that, after regulating search momentum, subsequent sub-queries consistently
retrieve along directions aligned with the original information need.
6 RELATEDWORK
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving
external knowledge during inference (Lewis et al., 2020; Izacard et al., 2023; Li et al., 2025b),
thereby providing additional factual support and mitigating hallucinations (Ji et al., 2023). Early
RAG systems typically follow a retrieve-then-read paradigm, where the model first retrieves rele-
vant documents based on the original query and then feeds the retrieved documents as contextual
input to prompt the LLM to generate responses (Shi et al., 2024; Izacard et al., 2023; Lin et al.,
2024). Furthermore, some studies investigate how to enable LLMs to adaptively determine when
to retrieve external knowledge during response generation, thereby improving the reliability of gen-
erated responses (Asai et al., 2024). However, for complex questions, retrieving solely with the
original query is often insufficient to acquire all relevant evidence, as the required information is
typically distributed across multiple documents (Fang et al., 2025; Wu et al., 2026).
Recent reasoning-driven RAG methods leverage the reasoning capabilities of LLMs to enable pro-
gressive reasoning and sub-query generation for acquiring external knowledge, while incorporat-
ing the retrieved evidence into subsequent reasoning processes (Li et al., 2025a; Jin et al., 2025).
However, these methods heavily rely on the quality of generated sub-queries, where irrelevant or
erroneous sub-queries may lead to deviations in subsequent retrieval trajectories. Therefore, some
studies have begun to investigate how to optimize the sub-query generation process. S3 (Jiang et al.,
2025) trains a lightweight model specifically for sub-query generation, enabling the retrieval of more
accurate knowledge. ParallelSearch (Zhao et al., 2025) prompts the model to generate multiple sub-
queries at each iteration and performs parallel retrieval, thereby acquiring more comprehensive ev-
9

Arxiv Preprint.
idence. However, these methods do not eliminate the influence of historical retrieval trajectories on
subsequent sub-query generation.
7 CONCLUSION
In this paper, we introduce search momentum, which captures the accumulated retrieval tendency
induced by historical changes in generated sub-queries along the reasoning trajectory. Our exper-
iments show that the sub-queries generated by the model tend to continuously evolve along the
accumulated retrieval trajectory, leading to deviations in query intent. Based on this observation, we
propose MAPS, a Macro-Adaptive Parallel Reasoning framework for regulating search momentum
and mitigating query-intent drift. By disentangling historical reasoning trajectories and introducing
a parallel reasoning mechanism, MAPS effectively alleviates query-intent drift.
AI USESTATEMENT
Generative AI tools are used in this paper solely to check grammatical errors and assist with language
polishing, with the aim of improving the clarity of the manuscript. All research ideas, experimental
design, implementation, analysis, and conclusions are independently developed and verified by the
authors without the use of any AI tools.
REFERENCES
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to
retrieve, generate, and critique through self-reflection. InThe Twelfth International Conference on
Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.
URLhttps://openreview.net/forum?id=hSyW5go0v8.
Richard H Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. A limited memory algorithm for
bound constrained optimization.SIAM Journal on scientific computing, 16(5):1190–1208, 1995.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-
Emmanuel Mazar ´e, Maria Lomeli, Lucas Hosseini, and Herv ´e J´egou. The faiss library.IEEE
Trans. Big Data, 12(2):346–361, 2026. doi: 10.1109/TBDATA.2025.3618474. URLhttps:
//doi.org/10.1109/TBDATA.2025.3618474.
Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald. Kirag: Knowledge-driven iterative retriever
for enhancing retrieval-augmented generation. InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), pp. 18969–18985, 2025.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu
Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhi-
hong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng,
Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie
Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li,
Hao Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li,
Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang,
Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean
Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan
Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming
Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi
Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu,
Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou,
Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, Tao Wang,
Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao,
Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin
Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue
Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou,
Xianzu Wang, Xinxia Shan, Y . K. Li, Y . Q. Wang, Y . X. Wei, Yang Zhang, Yanhong Xu, Yao
10

Arxiv Preprint.
Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying
He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou,
Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You,
Yuxuan Liu, Yuyang Zhou, Y . X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunx-
ian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean
Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui
Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng
Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through rein-
forcement learning.Nat., 645(8081):633–638, 2025. doi: 10.1038/S41586-025-09422-Z. URL
https://doi.org/10.1038/s41586-025-09422-z.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning steps. InProceedings of the 28th Interna-
tional Conference on Computational Linguistics, pp. 6609–6625, 2020.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning
with retrieval augmented language models.J. Mach. Learn. Res., 24:251:1–251:43, 2023. URL
https://jmlr.org/papers/v24/23-0037.html.
Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec
Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card.arXiv
preprint arXiv:2412.16720, 2024.
Yuelyu Ji, Rui Meng, Zhuochun Li, and Daqing He. Curriculum guided reinforcement learning for
efficient multi hop retrieval augmented generation.arXiv preprint arXiv:2505.17391, 2025.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.ACM
computing surveys, 55(12):1–38, 2023.
Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, and Jiawei
Han. s3: You don’t need that much data to train a search agent via rl. InProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing, pp. 21610–21628, 2025.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of the
2023 conference on empirical methods in natural language processing, pp. 7969–7992, 2023.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and
Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement
learning.arXiv preprint arXiv:2503.09516, 2025.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. InPro-
ceedings of the 2020 conference on empirical methods in natural language processing (EMNLP),
pp. 6769–6781, 2020.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research.Transactions of the
Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl a00276. URL
https://aclanthology.org/Q19-1026/.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In
Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-
Tien Lin (eds.),Advances in Neural Information Processing Systems 33: Annual Con-
ference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12,
2020, virtual, 2020. URLhttps://proceedings.neurips.cc/paper/2020/hash/
6b493230205f780e1bc26945df7481e5-Abstract.html.
11

Arxiv Preprint.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. InProceedings
of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5420–5438,
2025a.
Xinze Li, Sen Mei, Zhenghao Liu, Yukun Yan, Shuo Wang, Shi Yu, Zheni Zeng, Hao Chen, Ge Yu,
Zhiyuan Liu, Maosong Sun, and Chenyan Xiong. RAG-DDR: optimizing retrieval-augmented
generation using differentiable data rewards. InThe Thirteenth International Conference on
Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025b.
URLhttps://openreview.net/forum?id=Pnktu2PBXD.
Xinze Li, Yuqing Lan, Zhenghao Liu, Haidong Xin, Yukun Yan, Shuo Wang, Zheni Zeng, Sen Mei,
Ge Yu, and Maosong Sun. Seek: Steering llm reasoning for rag via internal reasoning sketches,
2026.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard James, Pedro Ro-
driguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. RA-
DIT: retrieval-augmented dual instruction tuning. InThe Twelfth International Conference on
Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.
URLhttps://openreview.net/forum?id=22OTbutug9.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi.
When not to trust language models: Investigating effectiveness of parametric and non-parametric
memories. InProceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (ACL 2023), 2023.
Damian Martinez, Catalina Riano, and Hui Fang. Preqrag - classify and rewrite for enhanced RAG,
2025. URLhttps://doi.org/10.48550/arXiv.2506.17493.
Qingying Niu, Ruiyang Ren, Wayne Xin Zhao, and Yaliang Li. Bound: Brief-guided corrective
preference distillation at search-control boundaries.arXiv preprint arXiv:2608.08768, 2026.
Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, April 2026. URL
https://qwen.ai/blog?id=qwen3.6-35b-a3b.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien
Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof Q&A
benchmark.arXiv preprint arXiv:2311.12022, 2023.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. In
Proceedings of the 2024 conference of the north american chapter of the association for com-
putational linguistics: Human language technologies (volume 1: Long papers), pp. 8371–8384,
2024.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang,
and Ji-Rong Wen. R1-Searcher: Incentivizing the search capability in LLMs via reinforcement
learning.arXiv preprint arXiv:2503.05592, 2025.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multi-
hop questions via single-hop question composition. InTransactions of the Association for Com-
putational Linguistics, 2022.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving re-
trieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. InProceed-
ings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023), pp.
10014–10037, 2023.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Ma-
jumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.arXiv
preprint arXiv:2212.03533, 2022.
12

Arxiv Preprint.
Mingyan Wu, Zhenghao Liu, Xinze Li, Yuqing Lan, Yukun Yan, Shuo Wang, Cheng Yang, Minghe
Yu, Zheni Zeng, and Maosong Sun. Finding what matters: Anchoring context knowledge with
evolving indices for iterative retrieval.arXiv preprint arXiv:2601.16462, 2026.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu,
Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang,
Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui
Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang
Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger
Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan
Qiu. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing (EMNLP), 2018.
Shu Zhao, Tan Yu, Anbang Xu, Japinder Singh, Aaditya Shukla, and Rama Akkiraju. ParallelSearch:
Train your LLMs to decompose query and search sub-queries in parallel with reinforcement learn-
ing.arXiv preprint arXiv:2508.09303, 2025.
13

Arxiv Preprint.
A APPENDIX
A.1 FITTING THESEARCHMOMENTUMCOEFFICIENTS
In this section, we provide a detailed description of the experimental settings and methodology for
fitting the momentum coefficients introduced in Section 2. Specifically, we first describe how the
datasets used for momentum fitting are constructe. Then, we present the detailed procedure for
estimating the search momentum coefficients. Finally, we report the fitted momentum results of the
MAPS model.
Data Construction for Momentum Fitting.We collect 1,000 questions from each of 2WikiQA,
MuSiQue, and HotpotQA to construct the training data asD train for fitting the momentum coeffi-
cientsβ t. For each question, we run Search-o1 with Qwen3-32B as the backbone model and record
the complete trajectory of iterative reasoning and retrieval. We discard trajectories with fewer than
three retrieval iterations to retain those containing sufficient query transitions for analyzing accu-
mulated search momentum. For each retained trajectory, we encode the original question and all
generated sub-queries,{q 0, . . . , q T}, using the E5 embedding model, obtaining their semantic rep-
resentations{q 0, . . . ,q T}. To evaluate the directional alignment between search momentum and
subsequent query deviations, we collect an additional 1,000 questions from each dataset asD test
, with no overlap withD train , and construct the held-out test data using the same filtering, and
encoding procedures as train data do. The search momentum coefficients are fitted exclusively on
the training dataD train and then fixed for evaluation on the test dataD test. For the Pooled-MHQA
setting, we merge the training data from the three datasets into a pooled training set and similarly
combine their test data into a pooled test set.
Fitting Objective.For each trajectory in the training setD train, we first compute the stepwise
directional alignment scores{S 1, . . . , S T−1}according to Eq. 6 in section 2. We then retain the
firstKalignment scores from each trajectory, whereK= 4, since the vast majority of retrieval
trajectories terminate within five iterations. Including later steps would substantially reduce the
number of available trajectories at those positions and may lead to unreliable coefficient estimation
due to the limited amount of training data.
The overall fitting objective averages the stepwise alignment scores equally:
J(β) =1
KX
i∈D trainKX
t=1S(i)
t(β).(15)
For each trajectoryi, we compute the momentum from Eq. 5 withm 0=0and use Eq. 6 to obtain its
stepwise alignment scoreS(i)
t(β). For MAPS and MAPS w/o Disentanglement, which use multiple
sub-queries for retrieval at each iteration, we average the embeddings of all sub-queries generated
at iterationtand use the resulting vector asq t. we estimate the search momentum coefficient
βby minimizing the fitting objectiveJ(β)using the L-BFGS-B quasi-Newton method with 500
iterations.
Table 4: Fitted Momentum Coefficients and Directional Alignment between Search Momentum and
Subsequent Query Intent Deviations for MAPS.
Datasetβ 1 S1 β2 S2 β3 S3 β4 S4 Avg.
2WikiQA 0.0000 0.5264 0.5408 0.8206 0.7472 0.8689 0.8698 0.8929 0.7772
MuSiQue 0.0000 0.5120 0.5272 0.8297 0.7404 0.8844 0.8394 0.9050 0.7828
HotpotQA 0.0000 0.5851 0.5256 0.8438 0.7259 0.9037 0.8557 0.9131 0.8115
Pooled MHQA 0.0000 0.5331 0.5328 0.8276 0.7408 0.8803 0.8570 0.9006 0.7854
Evaluation and Results across Methods.After fitting, we freezeβand compute the held-outS t
over eligible trajectories inD test. We also reportAvg.=1
KPK
t=1St. Table 1 reports the Search-o1
results in the main text; Tables 4 and 5 report the corresponding results for MAPS and MAPS w/o
Disentanglement. Each table gives the fitted coefficients and held-out alignment scores for all four
dataset settings.
14

Arxiv Preprint.
Table 5: Fitted Momentum Coefficients and Held-out Alignment Scores for MAPS w/o Disentan-
glement on Three Multi-hop Datasets and Pooled MHQA.
Datasetβ 1 S1 β2 S2 β3 S3 β4 S4 Avg.
2WikiQA 0.0000 0.5036 0.5469 0.8102 0.7543 0.8548 0.8527 0.8533 0.7555
MuSiQue 0.0000 0.4855 0.5369 0.7824 0.7538 0.8329 0.8629 0.8545 0.7388
HotpotQA 0.0000 0.5435 0.5467 0.8086 0.7607 0.8236 0.8651 0.8596 0.7588
Pooled MHQA 0.0000 0.5065 0.5422 0.7978 0.7549 0.8396 0.8589 0.8547 0.7497
A.2 EXPERIMENTALDETAILS FORANALYZING THEEVOLUTION OFSEARCHMOMENTUM
In this section, we describe how the three-dimensional search momentum visualizations in Fig-
ures 1a and 1b are constructed. We first use t-SNE to jointly project the original query embeddings
and search momentum from all methods into a two-dimensional space. We then translate the pro-
jected points to position the original questions near the origin and center the momentum points at
each iteration. During translation, all points within the same group are shifted by the same amount,
preserving their relative positions. We then sample trajectories separately for each method and
construct three-dimensional plots with the iteration index on the z-axis. For each method at each
iteration, we draw a smooth closed contour around the corresponding point cloud to illustrate how
search momentum evolves across iterations.
A.3 CASESTUDY
Table 6 and Figure 4 present a qualitative case study on a 2WikiQA question asking which ofLa
Mˆome Vert-De-GrisandThe Pod Peoplehas the earlier-born director.
Although Search-o1’s first retrieval, for “La M ˆome Vert-De-Gris director,” identifies Olivier Dahan,
its reasoning then invents a version ofThe Pod Peopledirected by Natasha Lyonne and Amy Nichol-
son. It next queries Amy Nicholson’s birth date and drifts further toward documentary production,
1930s organizations, and finally the Farm Holiday Association’s membership, following this unsup-
ported interpretation and deviating from the original question . In contrast, MAPS’s parallel queries
identify both relevant directors, Bernard Borderie and Juan Piquer Sim ´on, in the first iteration and
focus subsequent retrieval on their birth dates. A biographical entry retrieved in the second iteration
records Sim ´on’s birth on 16 February 1935; later film entries reconfirm Borderie’s 1953 directorial
debut, from which MAPS infers that he was born earlier, although his exact birth date is not re-
trieved. This broader initial exploration and evidence-guided refinement enable MAPS to answer
correctly without following Search-o1’s misleading local trajectory.
A.4 PROMPTTEMPLATES
We detail the prompt templates employed in MAPS. As shown in Figure 5, it assesses whether
the accumulated knowledge is sufficient to answer the original question and either produces a final
answer or proposes complementary retrieval directions. As shown in Figure 6, it prompts the LLM
to convert these directions into concrete sub-queries for parallel retrieval. Finally, the prompt in
Figure 7 extracts relevant information from the retrieved documents and updates the accumulated
reasoning traces, providing evidence and guidance for subsequent iterations.
15

Arxiv Preprint.
Search-o1
MAPSOriginal question
Golden evidenceLa môme
vert-de-grisThe Pod People
Bernard
BorderieJuan Piquer
SimónOriginal
question
(a) Document Embedding Space
La môme
vert-de-grisThe Pod People
Bernard
BorderieJuan Piquer
Simón1
1a1b
1c1dOriginal
question (b) Iteration 1
La môme
vert-de-grisThe Pod People
Bernard
BorderieJuan Piquer
Simón1
21a1b
1c1d
2a2bOriginal
question (c) Iteration 2
La môme
vert-de-grisThe Pod People
Bernard
BorderieJuan Piquer
Simón1
2
31a1b
1c1d
2a2b
3aOriginal
question
(d) Iteration 3
La môme
vert-de-grisThe Pod People
Bernard
BorderieJuan Piquer
Simón1
2
3
41a1b
1c1d
2a2b
3a4aOriginal
question (e) Iteration 4
La môme
vert-de-grisThe Pod People
Bernard
BorderieJuan Piquer
Simón1
2
3
451a1b
1c1d
2a2b
3a4a5a
5cOriginal
question (f) Iteration 5
Figure 4: T-SNE Projection of E5 Embeddings for the Case-study Trajectories in Table 6. The
black diamond marks the original question, and gold stars mark the four gold documents. Orange
points trace Search-o1’s sequential queries; blue points mark MAPS’s parallel queries, with letters
identifying branches. Blue outlines indicate the range covered by MAPS’s queries in each iteration.
16

Arxiv Preprint.
Table 6: Case Study Comparing Search-o1 and MAPS. Red highlights misleading observations,
off-topic evidence associated with query drift, and the incorrect final output; blue marks correctly
identified key entities and the correct final answer.
Question:Which film has the director who was born first, La Môme Vert-De-Gris or The Pod People?
Gold answer:La Môme Vert-De-Gris.
Search-o1
Iteration 1. Query:La Môme Vert-De-Gris director
Evidence:Documents:The search identifies Olivier Dahan as the director of La Môme Vert-De-Gris. The retrieved result does not support
the subsequent claims about a different version of The Pod People.Refined observation:After retrieving Olivier Dahan, the reasoning
invents a version of The Pod People directed by Natasha Lyonne and Amy Nicholson. It then shifts to comparing those directors and
follows that unsupported interpretation in later searches.
Iteration 2. Query:Amy Nicholson birth date
Evidence:Documents:Results concern Meredith Nicholson (an author born in 1866), Amy Hill, Richard Nicholson, Christopher Nichol-
son, and Lorraine Nicholson (an actress and director born in 1990), rather than Amy Nicholson.Refined observation:The results do
not identify Amy Nicholson or provide a birth date; they concern Meredith Nicholson, Amy Hill, Richard and Christopher Nicholson,
and Lorraine Nicholson. The reasoning then recasts the task as comparing film profit margins and veers into queries about making a
documentary, without establishing either director’s birth date.
Iteration 3. Query:Process of making a documentary film
Evidence:Documents:Filmmaking articles describe production stages such as screenwriting, casting, shooting, sound recording, editing,
and screening. Documentary references discuss factual subjects, actuality footage, and commentary; none bears on the films or directors
in the question.Refined observation:Summarizes production stages, then asserts that the user wants the number of people in a group and
introduces “The 1930s: A Decade of Discontent” as the current search query.
Iteration 4. Query:The 1930s: A Decade of Discontent group profile
Evidence:Documents:Results describe the Great Depression, labor movements, and 1930s political and economic conditions. One
document says the League for Industrial Democracy organized New Deal radio broadcasts, but none identifies a group named “The 1930s:
A Decade of Discontent” or gives its membership.Refined observation:Treats the request as identifying an organization associated with
1930s discontent and selects the League for Industrial Democracy as the closest match. The observation continues the historical-group
topic rather than returning to either film.
Iteration 5. Query:Farm Holiday Association 1930s membership numbers
Evidence:Documents:The Farmers’ Holiday Association was organized by Milo Reno in May 1932 and advocated withholding farm
products during the Great Depression, but its retrieved entry gives no membership count. Other results give figures for different organiza-
tions: 24,500 and 100,000 for the 19th-century Farmers’ Alliance, and 15,000 for the National Federation of Young Farmers’ Clubs. None
establishes membership of the queried association or answers the film question.
Final answer:The Farm Holiday Association reached its peak membership in the mid-1930s, with estimates ranging between 10,000 to
15,000 members.
MAPS
Iteration 1. Parallel queries:(1) Director of La Môme Vert-De-Gris; (2) The Pod People director; (3) birth date of director of La Môme
Vert-De-Gris; (4) The Pod People director birth date
Evidence:La Môme Vert-De-Gris, also released as Poison Ivy, is a 1953 French crime film and Bernard Borderie’s directorial debut. The
Pod People is a 1983 French-Spanish science fiction film directed by Juan Piquer Simón. The pooled documents identify both directors,
but neither birth date is available.
Iteration 2. Parallel queries:(1) Bernard Borderie director La Môme Vert-De-Gris birth date; (2) Juan Piquer Simón birth date
Evidence:Simón’s biography records his birth on 16 February 1935 in Valencia, Spain, and identifies him as the director of Pieces (1982)
and Slugs: The Movie (1988). Film documents again confirm Borderie’s association with La Môme Vert-De-Gris, but do not provide his
birth date.
Iteration 3. Query:Bernard Borderie birth date
Evidence:Documents concern Bernard Francis Law, Bernard Jenkin, Bernard Le Nail, and Bernard King, rather than Borderie. No
relevant birth date is found. The refined observation incorrectly interprets these irrelevant results as a reason to question Borderie’s
identity.
Iteration 4. Query:Bernard Borderie filmography and birth date. A second planned branch produces an empty query.
Evidence:Results include the Pool Group and its film Borderline (1930), actor Bernard Blier (born 11 January 1916), and actor/director
Paul Henreid (born 10 January 1908). None establishes Borderie’s birth date; the refined observation again incorrectly rejects his director
attribution.
Iteration 5. Parallel queries:(1) director of La Môme Vert-De-Gris birth date; (2) Juan Piquer Simón birth date; (3) a malformed query
containing generated text instead of a concise search.
Evidence:Film entries reconfirm Bernard Borderie’s 1953 debut with La Môme Vert-De-Gris; Juan Piquer Simón’s biography again gives
16 February 1935. Borderie’s birth date remains absent from the documents.
Final answer:La Môme Vert-De-Gris.
17

Arxiv Preprint.
System:Youareanavigatoragentinamulti-stageretrievalreasoningsystemtoanswertheuser'squestion.Youreceivetheoriginalquestion,thehistoricalglobalrefinementreports,andaccumulatedretrievalexperiencesrefinedoverpreviousiterations.Yourtask:1.Thinkgloballyinside<think>...</think>.2.Carefullyassesswhethertheinformationalreadygatheredissufficientforaconfidentanswer.Inthefirstround,whenlittleornoinformationhasbeengathered,considerretrievingusefulevidencebeforeanswering.3.Iftheavailableinformationissufficient,useittooutputthefinalconciseshortanswerinside<answer>...</answer>.4.Ifimportantinformationisstillmissing,outputasetofretrievaldirectionsinside<search_directions>...</search_directions>.Eachdirectionmustbewrappedin<directionid="k">...</direction>.IMPORTANT--Learningfromaccumulatedretrievalexperiences:Reviewtheaccumulatedretrievalexperiencesbelow.Theyhavebeencontinuouslyrefinedacrossiterations--capturingwhatsearchstrategiesandquerypatternswork,whatdoesn't,andhowtoimprove.Usetheselessonstoproducebetter-targeteddirectionsthatavoidpastmistakesandreplicatepastsuccesses.AlsousetheHistoricalRefinedInformationastheevidencealreadygathered:identifywhatitalreadyresolves,whatremainsuncertain,andwhetheranothersearchwouldaddusefulinformation.CRITICAL:Everydirectionmustbeimmediatelysearchable.Useonlyknownentitynames--neveruseplaceholders,variablereferences,orunresolvedreferenceslike"[Name]","[Entity]".Ifadirectiondependsoninformationfromanotherdirection,mergethemoromitthedependentdirection.COMMITANDPRODUCEOUTPUT:Afteryour<think>reasoning,immediatelyemiteither<answer>...</answer>or<search_directions>...</search_directions>.Donotre-deliberate,donotrestatetheseinstructions,donotiterateonwording--committoyourfirstreasonableplan.YouroutputMUSTendwithacomplete<answer>...</answer>or<search_directions>...</search_directions>block.User:<OriginalQuestion>{question}</OriginalQuestion><HistoricalRefinedInformation>{refinementreportsR_1,...,R_{t-1}frompreviousiterations,or"None"inthefirstround}</HistoricalRefinedInformation><AccumulatedRetrievalExperiences>{experiencestoreE_t}</AccumulatedRetrievalExperiences>Decidewhethertoanswernoworproposethenextretrievaldirections.
Figure 5: Prompt Template Used in MAPS for Generating Retrieval Directions.
18

Arxiv Preprint.
System:YouareahelpfulassistantthathelpstheNavigatorAgentanswertheuser'squestion.Youreceivetheoriginalquestion,oneassignedretrievaldirection,andaccumulatedretrievalexperiencesrefinedoverpreviousiterations.Yourtaskistoconverttheretrievaldirectionintooneconcretesearch-engine-friendlyquery.IMPORTANT--Learningfromaccumulatedretrievalexperiences:Reviewtheaccumulatedretrievalexperiencesbelow.Theyhavebeencontinuouslyrefinedacrossiterations--capturingquerypatternsthatyieldedusefulornon-usefulresults.Usetheselessonstowriteabetterqueryforthecurrentdirection.Rules:-UseonlyentitynamesexplicitlymentionedinthedirectionorintheNavigator'sthoughts.-Neveruseplaceholderslike"[Name]","[Entity]",orunresolvedreferences.-Ifthedirectiondependsonanunknownentity,skipandoutput<search></search>instead.-Reasonlocallyinside<think>...</think>,thenoutputexactlyonequeryinside<search>...</search>.-Donotiterateonquerywordingorsecond-guessyourself--committoyourfirstreasonablequeryandoutputit.YouroutputMUSTendwithacomplete<search>...</search>block.User:<OriginalQuestion>{question}</OriginalQuestion><AssignedDirection{k}>{directiontextofretrievaldirectionk}</AssignedDirection{k}><AccumulatedRetrievalExperiences>{experiencestoreE_t}</AccumulatedRetrievalExperiences>Nowthinkandputyoursearchqueryin<search>...</search>.
Figure 6: Prompt Template Used in MAPS for Parallel Sub-Query Generation.
19

Arxiv Preprint.
System:Youareahelpfulassistanttaskedwithanalyzingthecurrentiteration'ssearchresultsandproducingBOTHhelpfulinformationANDanupdatedretrievalexperiencestore.Yourobjectives:1.EXTRACTHELPFULINFORMATIONFORTHEANSWER:Readthesearcheddocumentsandextractinformationrelevanttotheconcretesearchqueriesandtheoriginalquestion.IntegratethisintotheNavigatorAgent'sreasoning.Whenpresentingafact,preferattributingittotheconcretesearchquerythatretrievedthesupportingdocumentratherthantoitsDocID,sotheretrievalcontextremainsclear.Whendocumentsmentiondifferententitieswiththesameorsimilarname,preservetheirdistinguishingcontextratherthanmergingtheirfacts.Outputthecompleterefinedinformationinside<refined_information>and</refined_information>.Ifnohelpfulinformationisfound,summarizewhatwassearchedandretrieved.2.UPDATETHERETRIEVALEXPERIENCESTORE(CRITICAL):BelowistheCURRENTexperiencestoreaccumulatedfrompreviousiterations.BasedonTHISiteration'ssearchresults,produceaCOMPLETEUPDATEDexperiencestore.Foreachexistingexperience,decidewhetherto:-KEEPitas-is(stillvalid)-REFINE/MODIFYitbasedonnewevidencefromthisround-DELETEit(provedunreliableornolongerrelevant)Additionally,ADDnewexperiencesbasedonwhatyoulearnedinTHISround:Whatquerypatternsorsearchstrategiesworkedwell?Whichonesdidnot?Whatcouldbedonedifferentlynexttime?OutputtheCOMPLETEUPDATEDsetofexperiences,witheachlessoninside<experience>and</experience>.IncludeALLexperiencesthatshouldbekept(refinedoras-is),notjustthenewones.Forexample:<experience>Querieswithspecificentitynames(e.g."AliceChen")yieldmorerelevantdocumentsthanquerieswithgenericroles(e.g.'theCEO').</experience>User:<OriginalQuestion>{question}</OriginalQuestion><PreviousNavigatorAgentReasoning>{thinkblockoftheNavigatorAgentinthisiteration}</PreviousNavigatorAgentReasoning><AbstractSearchDirections>-Direction1:{direction1}-...-Directionm:{directionm}</AbstractSearchDirections><ConcreteSearchQueries>-SearchDirection:{k}|SearchQuery:{queryofparallelagentfordirectionk}-...</ConcreteSearchQueries><GlobalSearchedDocumentPool>DocID:{d_1}Sources:{directionsthatretrievedd_1}SourceDirections:{correspondingdirections}SourceQueries:{correspondingqueries}Content:{contentsofd_1}...(oneblockperpooleddocument)</GlobalSearchedDocumentPool><CurrentRetrievalExperienceStore(toreviewandupdate)>1.{experience1}...n.{experiencen}</CurrentRetrievalExperienceStore(toreviewandupdate)>
Figure 7: Prompt Template Used in MAPS for Refined Reasoning.
20