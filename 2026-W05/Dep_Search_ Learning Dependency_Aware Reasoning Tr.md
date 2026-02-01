# Dep-Search: Learning Dependency-Aware Reasoning Traces with Persistent Memory

**Authors**: Yanming Liu, Xinyue Peng, Zixuan Yan, Yanxin Shen, Wenjie Xu, Yuefeng Huang, Xinyi Wang, Jiannan Cao, Jianwei Yin, Xuhong Zhang

**Published**: 2026-01-26 18:42:33

**PDF URL**: [https://arxiv.org/pdf/2601.18771v1](https://arxiv.org/pdf/2601.18771v1)

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, particularly when augmented with search mechanisms that enable systematic exploration of external knowledge bases. The field has evolved from traditional retrieval-augmented generation (RAG) frameworks to more sophisticated search-based frameworks that orchestrate multi-step reasoning through explicit search strategies. However, existing search frameworks still rely heavily on implicit natural language reasoning to determine search strategies and how to leverage retrieved information across reasoning steps. This reliance on implicit reasoning creates fundamental challenges for managing dependencies between sub-questions, efficiently reusing previously retrieved knowledge, and learning optimal search strategies through reinforcement learning. To address these limitations, we propose Dep-Search, a dependency-aware search framework that advances beyond existing search frameworks by integrating structured reasoning, retrieval, and persistent memory through GRPO. Dep-Search introduces explicit control mechanisms that enable the model to decompose questions with dependency relationships, retrieve information when needed, access previously stored knowledge from memory, and summarize long reasoning contexts into reusable memory entries. Through extensive experiments on seven diverse question answering datasets, we demonstrate that Dep-Search significantly enhances LLMs' ability to tackle complex multi-hop reasoning tasks, achieving substantial improvements over strong baselines across different model scales.

## Full Text


<!-- PDF content starts -->

Dep-Search: Learning Dependency-Aware Reasoning
Traces with Persistent Memory
Yanming Liu1∗, Xinyue Peng2∗, Zixuan Yan1, Yanxin Shen1, Wenjie Xu3,
Yuefeng Huang1, Xinyi Wang, Jiannan Cao4, Jianwei Yin1, Xuhong Zhang1†
1Zhejiang University,2Intel Corporation,3Tsinghua University,
4Massachusetts Institute of Technology
{oceann24, yanzixuan, ssyysyx, zhangxuhong, zjuyjw}@zju.edu.cn
{xuwj24}@mails.tsinghua.edu.cn, jiannan@mit.edu, xinyue.peng@intel.com
Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in
complex reasoning tasks, particularly when augmented with search mechanisms
that enable systematic exploration of external knowledge bases. The field has
evolved from traditional retrieval-augmented generation (RAG) frameworks to
more sophisticated search-based frameworks that orchestrate multi-step reasoning
through explicit search strategies. However, existing search frameworks still rely
heavily on implicit natural language reasoning to determine search strategies and
how to leverage retrieved information across reasoning steps. This reliance on
implicit reasoning creates fundamental challenges for managing dependencies
between sub-questions, efficiently reusing previously retrieved knowledge, and
learning optimal search strategies through reinforcement learning. To address these
limitations, we propose Dep-Search, a dependency-aware search framework that
advances beyond existing search frameworks by integrating structured reasoning,
retrieval, and persistent memory through GRPO. Dep-Search introduces explicit
control mechanisms that enable the model to decompose questions with depen-
dency relationships, retrieve information when needed, access previously stored
knowledge from memory, and summarize long reasoning contexts into reusable
memory entries. Through extensive experiments on seven diverse question answer-
ing datasets, we demonstrate that Dep-Search significantly enhances LLMs’ ability
to tackle complex multi-hop reasoning tasks, achieving substantial improvements
over strong baselines across different model scales.
1 Introduction
Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning
tasks, particularly when augmented with search mechanisms that enable systematic exploration of
external knowledge bases [ 1,7,12]. The field has evolved from traditional retrieval-augmented
generation (RAG) frameworks to more sophisticated search-based frameworks that orchestrate
multi-step reasoning through explicit search strategies [ 14]. Recent work such as Search-R1 [ 9],
DeepResearcher [ 43], Chain-of-Agents [ 13], and Kimi-K2 [ 30] have shown that search frameworks
can effectively decompose complex questions, retrieve relevant information from multiple sources,
and synthesize answers through structured multi-step reasoning [ 33]. However, existing search
frameworks still rely heavily on implicit natural language reasoning to determine search strategies and
how to leverage retrieved information across reasoning steps [ 33]. This reliance on implicit reasoning
∗Equal contribution.
†Corresponding author.
Preprint.arXiv:2601.18771v1  [cs.CL]  26 Jan 2026

creates fundamental challenges for managing dependencies between sub-questions, efficiently reusing
previously retrieved knowledge, and learning optimal search strategies through reinforcement learning
[44].
The fundamental problem lies in the lack of explicit dependency modeling and persistent memory
management in current search frameworks. Existing approaches decompose questions into sub-
questions but fail to explicitly model dependencies between these sub-questions, leading to inefficient
search patterns where the same information may be retrieved multiple times or sub-questions are
answered out of dependency order [ 15]. Moreover, existing systems treat each reasoning episode
independently, discarding valuable knowledge extracted during search that could be reused across
questions or even within the same multi-step reasoning process [ 38]. This knowledge loss is partic-
ularly problematic in complex scenarios where retrieved facts from early steps are needed in later
dependent steps, forcing redundant searches and increasing computational costs [ 16]. Addition-
ally, training search-based LLMs to learn optimal search strategies remains challenging, as existing
reinforcement learning approaches struggle with the sparse reward signals and the need to jointly
optimize decomposition, retrieval, memory access, and reasoning behaviors [2].
To address these limitations, we proposeDep-Search, a dependency aware search framework that
advances beyond existing search frameworks by integrating structured reasoning, retrieval, and
persistent memory through GRPO. Dep-Search introduces explicit control mechanisms that enable
the model to decompose questions with dependency relationships, retrieve information when needed,
access previously stored knowledge from memory, and summarize long reasoning contexts into
reusable memory entries. By combining dependency aware question decomposition with a persistent
memory system and GRPO for trajectory-level learning, Dep-Search ensures that reasoning follows
explicit dependency structures, retrieved knowledge is efficiently stored and reused, and the policy
learns to optimize the entire search-reasoning-memory pipeline jointly. Unlike existing search
frameworks that rely on heuristic search strategies, Dep-Search treats all tokens uniformly in the
policy, enabling end-to-end learning of when to decompose, what to retrieve, when to access memory,
and how to synthesize final answers, while the explicit memory state provides verifiable knowledge
accumulation throughout the reasoning process.
Our Contributions.Our contributions are detailed as follows.
•We present Dep-Search, a novel framework that formalizes multi-hop reasoning through dependency
aware decomposition and explicit control tokens, providing structured reasoning traces and efficient
knowledge reuse.
•We introduce a persistent memory system that automatically stores summarized facts from searches
and enables efficient memory access through embedding-based similarity search, addressing the
knowledge loss problem in existing search frameworks.
•We demonstrate that QDMR-based decomposition enables adaptive dependency modeling that
significantly outperforms sequential decomposition approaches, allowing the model to determine
both the number of reasoning steps and their dependency structure dynamically.
2 Related Work
2.1 Agentic Reinforcement Learning
Recent advances in agentic reinforcement learning (RL) have explored how LLM-based agents can
interact with environments, tools, and external knowledge sources to solve complex tasks through
trial-and-error learning [ 8,23,24]. Early work focused on using RL to fine-tune language models on
synthetic reasoning tasks or instruction-following benchmarks, typically with short-horizon rewards
and limited interaction structure [ 18,26]. Recent frameworks introduce multi-step decision processes
in which the agent can iteratively call tools, plan, and revise its strategy [ 13,30]. These systems
demonstrate that explicit interaction loops and environment feedback can significantly improve the
robustness and adaptability of LLMs on complex tasks such as web navigation, code generation,
and multi-hop question answering [ 4,40]. A common theme across these approaches is the use
of policy optimization balance exploration and exploitation, such as entropy-balanced objectives
that encourage diverse exploration while maintaining exploitation of promising strategies [ 2], and
experience replay mechanisms that enable agents to learn from past trajectories more effectively
[16]. The emphasis on trajectory-level learning, where agents learn to optimize sequences of actions
2

rather than individual decisions, enabling better credit assignment and long-term planning in complex
multi-step reasoning scenarios.
2.2 Agentic Memory
A growing line of work studies how LLM agents can maintain and exploit persistent memory to
improve long-term coherence, personalization, and knowledge reuse [ 19,21,37]. Early memory-
augmented systems typically log past interactions or retrieved documents in a buffer and naively
prepend them to the prompt, which quickly becomes inefficient and noisy as the context grows
[3,41]. Subsequent approaches introduce memory retrieval modules based on dense embeddings,
enabling agents to select relevant past experiences or facts conditioned on the current query [ 29,34].
Recent agentic memory frameworks go further by allowing agents to write structured summaries
into memory, compressing long trajectories into reusable high-level knowledge that can be recalled
[6,35,36]. A common evolution across these approaches is the shift from passive memory storage to
active memory management, where agents not only retrieve but also strategically write and organize
memory content to optimize knowledge reuse across different reasoning episodes.
3 Methodology
3.1 Problem Overview
Let the problem distribution be D, where each instance is a natural-language question Q∼ D . Our
goal is to generate an answer Athrough a dependency aware search process by maximizing the
expected trajectory return:
max
θEQ∼D, τ∼π θ(·|Q)
R(τ)
,(1)
where πθis the Dep-Search policy, τ= (a 1, . . . , a T)is a complete reasoning trajectory containing
intermediate actions and the final answer, andR(τ)is the trajectory-level return.
At stept, the search state is defined as
St= (T t,Ct,M t),(2)
where Ttis the current dependency aware reasoning trace, recording decomposed sub-questions and
their dependency relations; Ctis the current context, including the system prompt, the question Q,
previously generated text, retrieved evidence, and the explicitly exposed memory content; and Mtis
the memory buffer that stores fact sentences extracted during reasoning.
In implementation, the state Stis encoded in the already generated token prefix x1:t−1 . The policy
defines the conditional distribution of the next token on this prefix:
pθ(at|St)≡p θ(at|x1:t−1).(3)
3.2 Data Collection and Design
During data collection, we use the current policy πθoldto interact with the environment and sam-
ple multiple complete reasoning trajectories for each question Q, which are then used for GRPO
optimization.
Given a questionQ, the initial state is
S0= (T 0,C0,M 0),(4)
whereT0is empty, C0consists of the system prompt and Q, andM0is the fixed initial memory. The
conditional probability of a full trajectoryτ= (a 1, . . . , a T)is:
pθold(τ|Q) =TY
t=1πθold(at|x1:t−1),(5)
where x1:t−1 denotes the token prefix before step t, and the state sequence Stis induced by the
environment transition operator St+1=G(S t, at)that updates the state based on the generated token
at. Specifically, when the model emits control tokens, the environment updates the state components
3

Figure 1: Overview of the Dep-Search framework. The agent decomposes questions into dependent
sub-questions, retrieves relevant information, accesses stored knowledge from memory, and synthe-
sizes answers through trajectory-level reinforcement learning.
as follows: (1) <Decompose> updatesT tby adding new sub-questions and their dependency edges;
(2)<Retrieve> updates Ctby appending retrieved documents and automatically summarizes them into
memory entries; (3) <Memory> updates Ctby appending retrieved memory facts; (4) <Conclusion>
updates Mtby summarizing the current context into new memory entries. For regular reasoning
tokens, Ctis updated by appending the generated tokens to the context. The complete rollout
procedure is detailed in Algorithm 1.
Decompose token.When the model emits <Decompose> , it decomposes the question QintoK
dependent sub-questions {q1, . . . , q K}, where dependencies between sub-questions form a directed
acyclic graph (DAG) structure. Unlike sequential decomposition that processes sub-questions linearly,
Dep-Search models explicit dependency relationships where each sub-question qkmay depend on
the results of one or more prerequisite sub-questions, forming a multi-level tree-like dependency
structure. The model then solves these sub-questions following a topological ordering, ensuring that
prerequisite sub-questions are resolved before dependent ones, similar to QDMR decomposition
strategies. The decomposition updates the reasoning trace as Tt+1=Tt∪ {(q k,deps(q k))}, where
deps(q k)denotes the set of prerequisite sub-questions for qk. The detailed training prompt template
is provided in Appendix D.
Retrieve token.When the model emits <Retrieve> followed by a query r1:Land the closing tag
</Retrieve> , the environment immediately performs retrieval. The retrieval process consists of two
stages: first usingqwen3-embeddingfor dense retrieval to obtain a candidate set Dcandwith similarity
scores sdense(di, r) =cosine(E emb(di),E emb(r)), then applyingqwen3-rerankerfor re-ranking with
scoress rerank(di, r)to select the top-kdocuments:
Dt=Top-k(D cand, srerank),(6)
whereEembandErerank denote the embedding functions for dense retrieval and reranking, respectively.
The retrieved documents are formatted and inserted as <Retrieve_result> Dt</Retrieve_result> im-
mediately after the closing </Retrieve> tag, updating the context as Ct+1=Ct∪ Dt. The model
generates the query autonomously, allowing it to determine what information to retrieve based on
the current reasoning context. Additionally, information from the retrieved documents is automat-
ically summarized into fact sentences and stored in memory using the LRU rule, updating Mt+1
accordingly.
Memory token.When the model emits <Memory> followed by a query and the closing tag
</Memory> , the environment fetches relevant summarized facts from the memory buffer. Concretely,
it always includes the most recently written memory items and augments them with additional
entries retrieved by running qwen3-embedding over all m∈ M t, computing cosine similarities
4

between the query and memory embeddings, and selecting those whose similarity exceeds a threshold.
The concatenated memory snippets are then inserted as <Memory_result> Mread
t</Memory_result>
immediately after the closing </Memory> tag, updating the context as Ct+1=Ct∪ Mread
t. The
model generates the query autonomously, allowing it to determine what knowledge to fetch from
memory based on the current reasoning needs, enabling effective knowledge reuse during reasoning.
Conclusion token.When the model emits <Conclusion> , it asks the environment to summarize the
preceding reasoning and retrieved evidence that have been resolved into a compact natural-language
conclusion. The environment runs the LLM in summarization mode over the current contextC tand
writes the resulting fact sentences into the memory buffer as new entries, updating Mt+1via the
LRU rule. These stored facts can later be accessed via <Memory> to avoid redundant retrieval and
reasoning. Note that memory entries are not deleted but accumulated, with older entries evicted only
when the memory capacity is exceeded.
3.3 Memory Module
We model the memory Mtas an LRU buffer with a fixed capacity, storing fact sentences extracted
from retrieved documents and summarized reasoning during the search process. The memory and
its update rule have no trainable parameters and are treated as part of the state and the environment
transition.
State representation.At time stept, the memory is a finite set:
Mt={m(t)
1, . . . , m(t)
nt}, n t≤C mem.(7)
Each entry mcontains a fact sentence f(m) expressed in natural language and metadata such as
source and write time. For example, memory entries may contain fact sentences like “Beijing hosted
the 2022 Winter Olympics” or “Tom Hanks starred in Forrest Gump, which grossed $678 million
worldwide.” These fact sentences are stored as plain text, allowing the model to reuse previously
extracted knowledge without re-retrieving or re-reasoning. The overall search state is
St= (T t,Ct,M t).(8)
In practice, Mtis verbalized into a short “known facts” segment and injected into the token prefix so
that the policy can explicitly read the current memory.
When the accumulated context becomes long or contains several resolved sub-questions, the model
can emit <Conclusion> to compress the preceding reasoning and evidence that have been resolved
into new memory entries. We denote the set of newly written memory items at steptby
Ft=Summarize(C t),(9)
where Summarize(·) is implemented by prompting the LLM to produce a few natural-language fact
sentences that capture reusable knowledge from the resolved reasoning steps. These fact sentences
are then stored in Mtwithout deleting existing entries, allowing knowledge accumulation throughout
the reasoning process. We maintain a recency marker ℓt(m)∈N for each memory entry m, and
update it whenevermis written/added byF t. Let the candidate set be:
˜Mt+1=M t∪ Ft,(10)
and update the recency markers:
ℓt+1(m) =t, m∈ F t,
ℓt(m), m∈ M t\ Ft.(11)
Then we perform capacity truncation to obtain the updated memory:
Mt+1= arg max
M⊆ ˜Mt+1|M|≤C memX
m∈Mℓt+1(m).(12)
Equivalently, we keep up toC mementries with the largestℓ t+1(m)and evict those with the smallest
ℓt+1(m).
During training, each episode starts from a fixed initial memory M0, which is typically empty, and
memory is not shared across episodes. During inference, one may inject a cross-question long-term
memory asM 0to evaluate knowledge accumulation.
5

3.4 RL with Dep-Search
In the reinforcement learning stage, we optimize the Dep-Search policy πθusing GRPO. The policy
generates trajectories τ= (a 1, . . . , a T)that interleaves control tokens with reasoning tokens, where
all tokens are modeled uniformly.
We use trajectory-level rewards since the final answer quality depends on the entire reasoning process.
For each questionQ, we sampleKtrajectories:
G(Q) ={τ 1, . . . , τ K}, τ k∼πθold(· |Q),(13)
and compute trajectory-level returnsR(τ k). We define group-relative advantages:
A(τk) =R(τ k)−¯R(Q), ¯R(Q) =1
KKX
k=1R(τk),(14)
which naturally handle varying question difficulty by comparing trajectories within the same group.
Let the token sequence of τkbe{(sk,t, ak,t)}t, where sk,tencodes the current state St. We update
the policy using the clipped GRPO objective:
LGRPO(θ) =E Q∼D, τ k∼πθold(·|Q), th
min
ρk,t(θ)·A(τ k),clip(ρ k,t(θ),1−ϵ,1 +ϵ)·A(τ k)i
−β·E Q∼D, τ k∼πθold(·|Q), th
KL 
πθold(· |s k,t)∥π θ(· |s k,t)i
,
(15)
where ρk,t(θ) =π θ(ak,t|sk,t)/πθold(ak,t|sk,t). Since A(τk)is shared across all tokens in τk, this
enables joint optimization of decomposition, retrieval, memory access, and reasoning behaviors.
The policy πθ(at|x1:t−1)always conditions on the current state St: it decides whether to retrieve
(<Retrieve> ), whether to decompose ( <Decompose> ), and how to generate subsequent reasoning
given the available memory and evidence. Memory writing and LRU updates are purely environment
rules and do not receive gradients; reinforcement learning only updates θ, learning when to retrieve
and how to leverage memory to achieve high-quality answers with fewer retrievals. Since memory is
explicitly included in the state, Dep-Search remains a standard MDP and the dynamic memory does
not violate the assumptions of GRPO.
3.5 Reward Model
The trajectory return R(τ) is primarily driven by answer quality and imposes a curved penalty on
excessive retrieval and decomposition. We define A(τ) as the final answer produced by trajectory τ,
A⋆as the gold answer, Nret(τ)as the number of <Retrieve> calls in τ, andNdec(τ)as the number of
<Decompose> calls inτ.
The total return is
R(τ) =R ans(τ)−R ret(τ)−R dec(τ),(16)
where Rans(τ)is the answer quality reward, and Rret(τ)andRdec(τ)are penalties for excessive
retrieval and decomposition, respectively.
For the answer-quality term Rans(τ), we use exact match (EM) or F1 score between the generated
answerA(τ)and the gold answerA⋆:
Rans(τ) =EM 
A(τ), A⋆
orR ans(τ) =F1 
A(τ), A⋆
,(17)
where both metrics are normalized to[0,1].
For both retrieval and decomposition penalties, we apply a linear penalty only when the opera-
tion count exceeds a threshold. Let k1andk2be the thresholds for retrieval and decomposition,
respectively. The penalty functions are:
Rret(τ) =(
0, N ret(τ)≤k 1,
λret 
Nret(τ)−k 1
, N ret(τ)> k 1,Rdec(τ) =(
0, N dec(τ)≤k 2,
λdec 
Ndec(τ)−k 2
, N dec(τ)> k 2,
(18)
6

where λret>0andλdec>0control the penalty slopes for retrieval and decomposition, respectively.
This design makes the return mainly driven by answer quality, while penalizing retrieval and de-
composition only after surpassing reasonable thresholds, encouraging efficient dependency aware
search. Moreover, within each GRPO group for a given question, trajectories are rolled out under the
same initial memory configuration and environment rules, so the model observes consistent memory
dynamics across the group. This shared memory context allows GRPO to provide stable supervision
for learning when and how to emit <Memory> to effectively exploit stored facts.
4 Experimental Setup
4.1 Datasets
We evaluate Dep-Search on six multi-hop question answering datasets that require complex reasoning
over multiple documents.HotpotQA[ 39] is a widely-used benchmark featuring questions that require
reasoning over multiple Wikipedia paragraphs, with both distractor and full-wiki settings.2WikiMul-
tihopQA[ 5] focuses on multi-hop questions that require comparing and contrasting information from
different Wikipedia articles.Musique[ 31] presents questions that need to aggregate information
across multiple paragraphs, with explicit reasoning chains.Bamboogle[ 20] is a challenging dataset
that requires searching through multiple web pages to answer questions.TriviaQA[ 10] contains
question-answer pairs with evidence from Wikipedia and web sources, testing the model’s ability to
retrieve and synthesize information.PopQA[ 17] focuses on popular entity questions that require
up-to-date knowledge retrieval. These datasets cover diverse question types, from factoid queries
to complex multi-step reasoning, providing comprehensive evaluation of Dep-Search’s dependency
aware search capabilities.
4.2 Baselines
We compare Dep-Search against ten baseline methods: Directly Inference, Vanilla RAG, IRCoT
[32], RA-ISF [ 15], Search-O1 [ 14], Search-R1 [ 9], R1-Searcher [ 25], HierSearch [ 28], O2-Searcher
[27], and ZeroSearch [ 27]. These baselines cover the spectrum from simple retrieval-augmented
generation to sophisticated search-based reasoning. For more details, please refer to Appendix A.
4.3 Models and Metrics
We conduct experiments using two model sizes:Qwen2.5-3B-InstructandQwen2.5-7B-Instruct
[22]. These models provide a good balance between performance and computational efficiency,
allowing us to evaluate Dep-Search’s effectiveness across different model scales. For evaluation
metrics, we useExact Match (EM)for multiple-choice questions where the answer format is
constrained, andF1 scorefor open-ended questions where partial credit is appropriate. Both metrics
are normalized to [0,1] and align with the reward function used during training, as described in
Section 3.5. Detailed information about retrieval corpus and implementation details are provided in
Appendix A.
5 Experiments
5.1 Main Results
Table 1 presents the comprehensive evaluation results across six question answering datasets. Our
Dep-Search method achieves the best overall performance on both model scales, demonstrating
consistent improvements over existing baseline methods.
Overall Performance.Dep-Search achieves average scores of 39.29 and 49.77 on Qwen2.5-3B-
Instruct and Qwen2.5-7B-Instruct, respectively, outperforming all baseline methods. On the 3B
model, Dep-Search scores 39.29, about 3 points higher than HierSearch, which reaches 36.31, and
about 4 points higher than O2-Searcher at 35.21. On the 7B model, Dep-Search reaches 49.77,
improving over HierSearch at 46.66 by roughly 3 points and over O2-Searcher at 45.70 by roughly
4 points. These results demonstrate that our dependency aware decomposition, persistent memory
mechanism, and GRPO-based training effectively improve multi-hop reasoning capabilities.
7

Table 1: Main experimental results on single-hop and multi-hop question answering datasets.Bold
numbers indicate the best performance among all methods for each model.
MethodSingle-Hop QA Multi-Hop QA
Avg.
NQ TriviaQA PopQA HotpotQA 2WikiMHQA Musique Bamboogle
Qwen2.5-3B-Instruct
Directly Inference 12.40 30.60 12.40 16.00 19.20 4.40 16.80 16.00
Vanilla RAG 13.80 29.20 14.60 13.40 17.20 3.20 14.40 15.11
IRCoT 14.20 34.80 20.80 19.60 28.40 6.40 5.56 18.54
RA-ISF 15.60 36.20 22.40 20.80 29.60 7.20 6.20 19.71
Search-O1 16.60 31.00 24.80 14.80 22.40 5.20 22.40 19.77
Search-R1 35.80 55.80 26.00 33.20 26.00 7.60 12.50 28.13
R1-Searcher 37.60 56.20 32.20 31.20 29.80 9.40 18.50 29.85
HierSearch 44.80 61.0048.8035.00 34.80 12.40 22.56 36.31
O2-Searcher 44.20 60.40 40.40 34.60 34.40 12.00 21.44 35.21
ZeroSearch 36.20 54.40 25.10 29.00 28.20 8.80 16.67 27.54
Dep-Search 47.20 65.00 47.40 38.00 38.80 14.60 24.00 39.29
Qwen2.5-7B-Instruct
Directly Inference 11.60 35.60 13.20 16.40 22.20 4.80 14.40 16.89
Vanilla RAG 13.20 36.80 15.40 17.60 23.40 5.60 15.20 17.60
IRCoT 27.60 47.40 27.40 21.00 29.20 9.80 27.78 27.17
RA-ISF 28.80 49.20 29.20 22.40 30.60 10.60 28.90 28.67
Search-O1 19.40 40.60 25.60 17.00 27.00 8.60 30.40 24.06
Search-R1 42.40 63.40 51.60 32.80 33.20 17.40 26.39 38.17
R1-Searcher 44.20 64.80 53.20 34.20 35.80 18.60 28.89 39.85
HierSearch 48.20 67.0061.6038.80 39.60 20.40 32.00 46.66
O2-Searcher 47.40 66.20 58.20 38.20 39.00 20.00 30.89 45.70
ZeroSearch 41.60 62.60 50.40 32.20 32.80 16.80 25.56 37.54
Dep-Search 53.80 72.00 60.20 44.40 45.20 22.20 30.56 49.77
Single-Hop vs. Multi-Hop QA.Dep-Search shows strong performance across both single-hop and
multi-hop question types. On single-hop datasets such as NQ, TriviaQA, and PopQA, Dep-Search
achieves average scores of 53.07 and 62.00 on the 3B and 7B models, respectively. HierSearch
slightly surpasses Dep-Search on PopQA, where HierSearch obtains 48.80 compared to 47.40 for
Dep-Search on 3B, and 61.60 compared to 60.20 on 7B, while our method achieves the best results
on NQ and TriviaQA. On multi-hop datasets including HotpotQA, 2WikiMHQA, Musique, and
Bamboogle, Dep-Search achieves average scores of 28.85 and 35.59 on 3B and 7B models, showing
larger gains over baselines. This suggests that explicit dependency modeling and memory reuse are
particularly valuable for complex reasoning chains that require information from multiple sources.
Model Scale Analysis.The performance gap between 3B and 7B models highlights the importance
of model capacity for complex reasoning tasks. Dep-Search improves from 39.29 on the 3B model
to 49.77 on the 7B model, an increase of about 10.5 points that is larger than for most baselines,
suggesting that larger models better leverage the structured reasoning and memory mechanisms. On
the 7B model, Dep-Search achieves particularly strong performance on multi-hop datasets: on Hot-
potQA, Dep-Search scores 44.40, about 12 points higher than Search-R1 at 32.80; on 2WikiMHQA,
Dep-Search reaches 45.20, about 12 points higher than Search-R1 at 33.20. Dep-Search’s depen-
dency aware decomposition allows the model to answer sub-questions in the correct order, while
the persistent memory mechanism reduces redundant retrievals by storing and reusing previously
extracted facts.
5.2 Ablation Study
To better understand the contribution of different components in Dep-Search, we conduct a compre-
hensive ablation study on the Qwen2.5-3B-Instruct model across all seven datasets. We systematically
remove the QDMR-style decomposition, the memory module, and the explicit conclusion-based
summarization to evaluate their individual contributions.
The results in Table 2 show that all three components contribute consistently across both single-hop
and multi-hop datasets. Removing the memory module causes the largest performance drop, with an
average decrease of 5.25 points. The degradation is particularly severe on multi-hop datasets such as
Musique, where performance drops from 14.60 to 11.00, and Bamboogle, where it decreases from
24.00 to 19.60, demonstrating that reusing summarized facts across reasoning steps is essential for
8

Table 2: Ablation study on Qwen2.5-3B-Instruct across all datasets. We report scores when removing
QDMR-style decomposition, the memory module, and the conclusion-based summarization mecha-
nism.∆denotes the average performance drop compared to the full model.
VariantSingle-Hop QA Multi-Hop QA
Avg.∆
NQ TriviaQA PopQA HotpotQA 2WikiMHQA Musique Bamboogle
Full Dep-Search 47.20 65.00 47.40 38.00 38.80 14.60 24.00 39.29–
w/o QDMR Decompose 43.80 61.20 44.00 34.00 35.20 12.40 21.20 35.97 -3.32
w/o Memory Module 41.60 58.40 42.20 32.50 33.00 11.00 19.60 34.04 -5.25
w/o Conclusion 45.00 62.80 45.60 35.50 36.60 13.20 22.40 37.30 -1.99
complex multi-hop reasoning. Removing QDMR-style decomposition leads to the second largest
drop, with an average decrease of 3.32 points. The effects are more pronounced on multi-hop datasets
such as HotpotQA and Musique, where performance decreases substantially, compared to single-hop
datasets like TriviaQA, where the impact is more moderate, confirming that explicit dependency
aware decomposition is crucial for structuring reasoning across sub-questions. The conclusion-based
summarization mechanism contributes a smaller but consistent improvement, with an average gain
of 1.99 points, suggesting that explicitly distilling long reasoning traces into compact, reusable
summaries further stabilizes the search process.
5.3 Reward Function Threshold Analysis
4 6 8 10 12 155
8
10
12
15
2038.5 39.8 41.2 41.5 40.9 40.2
41.8 43.4 44.6 44.3 44.0 43.2
43.0 44.7 47.0 45.4 45.1 44.3
42.1 43.8 45.3 45.0 44.6 43.9
40.6 42.3 43.5 43.2 42.8 42.1
39.0 40.7 41.9 41.6 41.2 40.5
38394041424344454647
Performance Score
Figure 2: Reward function threshold sensitiv-
ity analysis on 2WikiMHQA.We investigate the sensitivity of Dep-Search to
the reward function thresholds k1andk2on the
2WikiMHQA dataset using Qwen2.5-7B-Instruct.
These thresholds control when penalties are applied
for excessive retrieval and decomposition operations,
balancing between allowing necessary operations and
discouraging wasteful ones.
Figure 2 presents the performance across different
combinations of k1andk2. The optimal configura-
tion is k1= 10 andk2= 8, achieving a score of
47.0 on 2WikiMHQA. As the retrieval threshold k1
decreases, the penalty is applied earlier, discourag-
ing necessary retrieval operations and limiting the
model’s ability to gather sufficient information. Con-
versely, as k1increases, excessive retrieval operations
waste computational resources without improving
answer quality. Similarly, when the decomposition
threshold k2decreases, the model is penalized for
necessary decomposition steps, preventing proper
question breakdown. When k2increases, the model over-decomposes questions into unnecessary
fine-grained steps. The optimal thresholds strike a balance that allows sufficient operations for com-
plex multi-hop reasoning while preventing wasteful ones, demonstrating the importance of careful
hyperparameter tuning for reward function design.
5.4 Action Usage Analysis
To understand how Dep-Search adapts its search strategy to different question types, we analyze the
frequency of different action calls across various datasets. This analysis reveals how the framework
adjusts its decomposition, retrieval, memory access, and summarization behaviors based on dataset
characteristics. Figure 3 presents the average frequency of each action type across different datasets.
Decomposition.Multi-hop datasets trigger more frequent decomposition operations, with frequencies
ranging from 1.8 to 3.4 calls per question, as the model needs to explicitly break down complex
questions into dependent sub-problems. This enables the framework to structure reasoning chains
with clear dependencies, allowing each sub-question to leverage results from previous steps.
9

Retrieval.Multi-hop datasets trigger extensive retrieval operations, with frequencies ranging from
3.2 to 8.2 calls per question, as the model needs to gather evidence from different documents or
paragraphs to answer dependent sub-questions. The framework strategically performs retrievals at
different stages of reasoning, targeting specific information needed for each step.
NQ TriviaQA PopQA HotpotQA 2WikiMHQA Musique Bamboogle012345678
1.82.12.42.83.2
3.03.43.6
3.25.26.8
5.98.2
7.3
1.5
1.32.22.9
2.63.5
3.2
1.2
1.01.82.5
2.33.1
2.8Decompose
Retrieve
Fet_Memory
Conclusion
Figure 3: Action call frequency per question across
different datasets on Qwen2.5-7B-Instruct.Memory Access.Memory access frequencies
range from 1.3 to 3.5 calls per question, typ-
ically 40% to 50% of the retrieval frequency,
indicating selective utilization of stored knowl-
edge. This enables efficient knowledge reuse
across reasoning chains, particularly in multi-
hop scenarios where early retrieved facts are
needed in later dependent steps.
Conclusion.Conclusion frequencies range from
1.0 to 3.1 calls per question, with multi-hop
datasets showing higher frequencies as they gen-
erate longer reasoning chains that require com-
pression. The model summarizes intermediate
results into memory entries, helping manage
context length and enabling knowledge reuse in
subsequent reasoning steps.
5.5 Memory Capacity Sensitivity Analysis
1510 15 20 25 30 35 40 45 50
Memory Capacity37383940414243Performance Score
38.141.242.3
40.8Performance
Optimal (15 entries)
1510 15 20 25 30 35 40 45 50
Memory Capacity051015202530354045Memory Reuse Percentage (%)
Memory Reuse %
Avg Retrievals
5.756.006.256.506.757.007.257.50
Average Retrievals
Figure 4: Memory capacity sensitivity analysis on 2WikiMHQA.We investigate how memory ca-
pacity affects Dep-Search per-
formance by varying the mem-
ory buffer size from 1 to 50
entries on 2WikiMHQA using
Qwen2.5-7B-Instruct. This anal-
ysis helps understand the trade-
off between memory capacity
and performance, identifying the
optimal capacity for efficient
knowledge reuse.
Figure 4 presents the perfor-
mance across different memory
capacities, tested at intervals of
5 entries from 1 to 50.Performance peaks at 15 entries with a score of 42.3, demonstrating that
moderate memory capacity provides optimal knowledge reuse. Performance increases steadily from 1
to 15 entries, with scores improving from 38.1 to 42.3, as the memory buffer becomes large enough to
store relevant facts without excessive overhead. However, beyond 15 entries, performance gradually
decreases, dropping to 40.8 at 50 entries. This pattern suggests that while larger memory buffers can
store more information, they may introduce noise or make it harder for the model to identify the most
relevant entries, leading to suboptimal memory access decisions.Memory reuse percentage peaks
at 10 entries with 40.5% of entries being reused, then decreases rapidly as capacity increases, drop-
ping to 9.2% at 50 entries. This indicates that smaller capacities enable more frequent reuse of stored
knowledge, while larger buffers store more one-time-use information. The optimal performance at
15 entries occurs despite lower reuse percentage compared to 10 entries, suggesting that a balance
between reuse frequency and memory capacity is crucial for overall reasoning quality. The average
number of retrievals decreases with larger memory capacity, suggesting that the optimal capacity
balances between storing sufficient knowledge and maintaining efficient memory access. These
results demonstrate that a capacity of 15 entries provides the optimal balance between performance
and efficiency for 2WikiMHQA.
10

6 Conclusions
In this work, we introduced Dep-Search, a dependency aware search framework that enables LLMs
to perform structured multi-hop reasoning through explicit dependency modeling and persistent
memory management. Unlike existing search frameworks that rely on implicit natural language
reasoning to determine search strategies, Dep-Search integrates structured reasoning, retrieval, and
persistent memory through GRPO, allowing autonomous decomposition, strategic retrieval, and
efficient knowledge reuse across reasoning steps. Through extensive experiments on seven diverse
question answering datasets, we demonstrated that Dep-Search significantly enhances LLMs’ ability
to tackle complex multi-hop reasoning tasks, achieving substantial relative improvements over strong
baselines across different model scales, with larger models showing greater absolute gains while
smaller models benefit from more pronounced relative improvements. Our analysis also provides key
insights into RL training strategies for dependency aware search-augmented reasoning, particularly
regarding reward function design and the interplay between decomposition, retrieval, and memory
access behaviors. Looking ahead, future work can explore expanding Dep-Search to support broader
reasoning scenarios, including more sophisticated dependency modeling mechanisms, dynamic
memory management strategies, and integration with diverse external knowledge sources beyond
Wikipedia.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG:
Learning to retrieve, generate, and critique through self-reflection. InThe Twelfth International
Conference on Learning Representations, 2024. URL https://openreview.net/forum?
id=hSyW5go0v8.
[2]Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jing-
han Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, et al. Agentic entropy-balanced policy
optimization.arXiv preprint arXiv:2510.14545, 2025.
[3]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models:
A survey.arXiv preprint arXiv:2312.10997, 2(1), 2023.
[4]Yiwei Guo, Shaobin Zhuang, Kunchang Li, Yu Qiao, and Yali Wang. Transagent: Transfer
vision-language foundation models with heterogeneous agent collaboration.Advances in Neural
Information Processing Systems, 37:98419–98444, 2024.
[5]Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps. InProceedings of the
28th International Conference on Computational Linguistics, pages 6609–6625, 2020.
[6]Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin,
Honglin Guo, Shihan Dou, Zhiheng Xi, et al. Memory in the age of ai agents.arXiv preprint
arXiv:2512.13564, 2025.
[7]Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of
the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992,
2023.
[8]Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Za-
mani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with
reinforcement learning.arXiv preprint arXiv:2503.09516, 2025.
[9]Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O Arik, Dong Wang, Hamed
Zamani, and Jiawei Han. Search-r1: Training LLMs to reason and leverage search engines
with reinforcement learning. InSecond Conference on Language Modeling, 2025. URL
https://openreview.net/forum?id=Rwhi91ideu.
11

[10] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale
distantly supervised challenge dataset for reading comprehension. In Regina Barzilay and
Min-Yen Kan, editors,Proceedings of the 55th Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada,
July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL
https://aclanthology.org/P17-1147/.
[11] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InProceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 6769–6781, 2020.
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Advances in neural information processing
systems, 33:9459–9474, 2020.
[13] Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xinpeng Liu, Jiayu Zhang, Zhenqiang
Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, et al. Chain-of-agents: End-to-end agent
foundation models via multi-agent distillation and agentic rl.arXiv preprint arXiv:2508.13167,
2025.
[14] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang,
and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. In Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors,Proceedings
of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5420–
5438, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN
979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.276. URL https://aclanthology.
org/2025.emnlp-main.276/.
[15] Yanming Liu, Xinyue Peng, Xuhong Zhang, Weihao Liu, Jianwei Yin, Jiannan Cao, and Tianyu
Du. Ra-isf: Learning to answer and understand from retrieval augmentation via iterative
self-feedback. InFindings of the Association for Computational Linguistics ACL 2024, pages
4730–4749, 2024.
[16] Fanbin Lu, Zhisheng Zhong, Shu Liu, Chi-Wing Fu, and Jiaya Jia. Arpo: End-to-end policy
optimization for gui agents with experience replay.arXiv preprint arXiv:2505.16282, 2025.
[17] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Ha-
jishirzi. When not to trust language models: Investigating effectiveness of parametric and
non-parametric memories. InProceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, 2023.
[18] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to
follow instructions with human feedback.Advances in neural information processing systems,
35:27730–27744, 2022.
[19] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica,
and Joseph E Gonzalez. Memgpt: Towards llms as operating systems.arXiv preprint
arXiv:2310.08560, 2023.
[20] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis.
Measuring and narrowing the compositionality gap in language models. InFindings of the
Association for Computational Linguistics: EMNLP 2023, pages 5687–5711, 2023.
[21] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Heyi Tang, Feng Ren, Teng Ma, Shangming Cai,
Yineng Zhang, Mingxing Zhang, et al. Mooncake: A kvcache-centric disaggregated architecture
for llm serving.ACM Transactions on Storage, 2024.
[22] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu,
Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu,
12

Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji
Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang
Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5
technical report, 2025. URLhttps://arxiv.org/abs/2412.15115.
[23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint arXiv:2402.03300, 2024.
[24] Zhihong Shao, Yuxiang Luo, Chengda Lu, ZZ Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong
Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning.
arXiv preprint arXiv:2511.22570, 2025.
[25] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang,
and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement
learning.arXiv preprint arXiv:2503.05592, 2025.
[26] Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian
Min, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. Smart-searcher: Incentivizing the dynamic
knowledge acquisition of LLMs via reinforcement learning. In Christos Christodoulopoulos,
Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors,Findings of the Association for
Computational Linguistics: EMNLP 2025, pages 13572–13586, Suzhou, China, November 2025.
Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.
findings-emnlp.731. URLhttps://aclanthology.org/2025.findings-emnlp.731/.
[27] Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan
Zhang, Fei Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of llms
without searching.arXiv preprint arXiv:2505.04588, 2025.
[28] Jiejun Tan, Zhicheng Dou, Yan Yu, Jiehan Cheng, Qiang Ju, Jian Xie, and Ji-Rong Wen.
Hiersearch: A hierarchical enterprise deep search framework integrating local and web searches.
arXiv preprint arXiv:2508.08088, 2025.
[29] Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long Le, Yiwen Song, Yanfei Chen,
Hamid Palangi, George Lee, et al. In prospect and retrospect: Reflective memory management
for long-term personalized dialogue agents. InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), pages 8416–8439, 2025.
[30] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen,
Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence.arXiv
preprint arXiv:2507.20534, 2025.
[31] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique:
Multihop questions via single-hop question composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022.
[32] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In
Proceedings of the 61st annual meeting of the association for computational linguistics (volume
1: long papers), pages 10014–10037, 2023.
[33] Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. Astute rag: Over-
coming imperfect retrieval augmentation and knowledge conflicts for large language models.
InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 30553–30571, 2025.
[34] Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei
Huang, and Huajun Chen. Wise: Rethinking the knowledge memory for lifelong model
editing of large language models.Advances in Neural Information Processing Systems, 37:
53764–53797, 2024.
13

[35] Zixuan Wang, Bo Yu, Junzhe Zhao, Wenhao Sun, Sai Hou, Shuai Liang, Xing Hu, Yinhe Han,
and Yiming Gan. Karma: Augmenting embodied ai agents with long-and-short term memory
systems. In2025 IEEE International Conference on Robotics and Automation (ICRA), pages
1–8. IEEE, 2025.
[36] Tianxin Wei, Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning,
Mengting Ai, Yunzhe Li, Jingrui He, Ed H Chi, et al. Evo-memory: Benchmarking llm agent
test-time learning with self-evolving memory.arXiv preprint arXiv:2511.20857, 2025.
[37] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem:
Agentic memory for LLM agents. InThe Thirty-ninth Annual Conference on Neural Information
Processing Systems, 2025. URLhttps://openreview.net/forum?id=FiM0M8gcct.
[38] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem:
Agentic memory for llm agents.arXiv preprint arXiv:2502.12110, 2025.
[39] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 conference on empirical methods in natural language
processing, pages 2369–2380, 2018.
[40] Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr, Saining
Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making
agents via reinforcement learning.Advances in neural information processing systems, 37:
110935–110971, 2024.
[41] Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, LEI BAI, and Xiang Wang. Multi-agent
architecture search via agentic supernet. InForty-second International Conference on Machine
Learning, 2025. URLhttps://openreview.net/forum?id=imcyVlzpXh.
[42] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding
and reranking through foundation models.arXiv preprint arXiv:2506.05176, 2025.
[43] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and
Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world
environments.arXiv preprint arXiv:2504.03160, 2025.
[44] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale
Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables
complex reasoning in large language models. InThe Eleventh International Conference on
Learning Representations, 2023.
A Experimental Setup Details
A.1 Datasets
We evaluate Dep-Search on six multi-hop question answering datasets that require complex reasoning
over multiple documents:
•HotpotQA[ 39]: A widely-used benchmark featuring questions that require reasoning over multiple
Wikipedia paragraphs, with both distractor and full-wiki settings. The dataset contains over 113,000
question-answer pairs, where each question requires combining information from at least two
paragraphs to answer correctly. Questions are designed to test various reasoning types including
comparison, bridge, and intersection queries. The distractor setting includes irrelevant paragraphs
to test the model’s ability to filter noise, while the full-wiki setting requires searching through
the entire Wikipedia corpus. The dataset’s emphasis on multi-paragraph reasoning makes it an
ideal testbed for dependency aware reasoning, as models must identify which paragraphs contain
prerequisite information before answering dependent questions.
14

•2WikiMultihopQA[ 5]: Focuses on multi-hop questions that require comparing and contrasting
information from different Wikipedia articles. The dataset contains over 190,000 question-answer
pairs that explicitly require reasoning across multiple Wikipedia articles. Questions often involve
identifying relationships between entities mentioned in different articles, such as comparing birth
dates, locations, or achievements. This dataset emphasizes the need for explicit dependency
modeling, as questions frequently require identifying prerequisite information from one article
before querying related information from another. The structured nature of Wikipedia articles
and the explicit multi-hop requirements make this dataset particularly suitable for evaluating
dependency aware search frameworks.
•Musique[ 31]: Presents questions that need to aggregate information across multiple paragraphs,
with explicit reasoning chains. The dataset is constructed by composing simpler single-hop
questions into complex multi-hop queries, resulting in over 25,000 questions. Each question
comes with annotated reasoning paths that specify the sequence of information needed to answer
correctly. The dataset provides explicit reasoning chains, enabling evaluation of whether models
can correctly structure multi-step reasoning. Questions require aggregating facts from multiple
paragraphs, often involving temporal reasoning, numerical comparisons, or logical deductions. The
composition-based construction ensures that questions have clear dependency structures, making it
valuable for testing dependency aware decomposition strategies.
•Bamboogle[ 20]: A challenging dataset that requires searching through multiple web pages to
answer questions. The dataset contains questions that simulate real-world web search scenarios,
where answers are distributed across different web pages. Questions are designed to test the model’s
ability to navigate complex information spaces, follow links between pages, and synthesize infor-
mation from multiple sources. The dataset emphasizes the importance of managing dependencies
across different sources, as information from one page may be needed to understand or locate
information on another page. This dataset tests the model’s ability to handle noisy web content and
manage search dependencies in unstructured information spaces.
•TriviaQA[ 10]: Contains question-answer pairs with evidence from Wikipedia and web sources,
testing the model’s ability to retrieve and synthesize information. The dataset includes over
650,000 question-answer-evidence triples, with questions authored by trivia enthusiasts. Each
question comes with multiple evidence documents (approximately six per question on average),
providing high-quality distant supervision. The dataset includes both reading comprehension
and open-domain question answering formats, testing different aspects of retrieval and reasoning
capabilities. Questions exhibit considerable syntactic and lexical variability between questions
and corresponding answer-evidence sentences, requiring models to perform cross-sentence reason-
ing. The large scale and diverse question types make TriviaQA a comprehensive benchmark for
evaluating retrieval-augmented reasoning systems.
•PopQA[ 17]: Focuses on popular entity questions that require up-to-date knowledge retrieval. The
dataset contains questions about popular entities that are frequently queried, testing the model’s
ability to retrieve and utilize factual knowledge. Questions often involve temporal reasoning, as
they may ask about recent events or current information about well-known entities. This dataset
emphasizes the importance of memory mechanisms for storing and reusing factual knowledge, as
questions about the same entity may appear multiple times with different aspects. The focus on
popular entities ensures that questions have sufficient context and evidence available, while still
requiring sophisticated reasoning to combine multiple facts about the same entity.
These datasets cover diverse question types, from factoid queries to complex multi-step reasoning,
providing comprehensive evaluation of Dep-Search’s dependency aware search capabilities.
A.2 Baselines
We compare Dep-Search against ten baseline methods that represent different approaches to multi-hop
question answering:
•Directly Inference: Uses the base language model without any retrieval or search mechanisms,
serving as a lower bound to demonstrate the importance of external knowledge access. This baseline
directly generates answers from the model’s parametric knowledge, without accessing external
documents or performing any search operations. It helps quantify the performance gain achieved
by incorporating retrieval and search mechanisms.
15

•Vanilla RAG: Retrieves relevant documents using dense embeddings and directly generates answers
from the retrieved context, representing the simplest form of retrieval-augmented generation. This
baseline performs a single retrieval step using dense embeddings, retrieves top- kdocuments, and
generates answers directly from the concatenated retrieved context. It demonstrates the baseline
performance achievable with simple retrieval-augmented generation without iterative reasoning or
search strategies.
•IRCoT[ 32]: Integrates iterative retrieval with chain-of-thought reasoning, retrieving documents
at each reasoning step. The method alternates between generating reasoning steps and retrieving
relevant documents based on the current reasoning context. This approach demonstrates the benefits
of interleaving retrieval and reasoning, allowing the model to refine its search queries based on
intermediate reasoning results. The iterative process enables the model to progressively gather
information needed for answering complex questions.
•RA-ISF[ 15]: Employs retrieval-augmented inference with iterative search and filtering, using
feedback mechanisms to refine retrieval queries. The method performs multiple rounds of retrieval,
where each round uses feedback from previous retrievals to improve query formulation. It employs
iterative search and filtering mechanisms to progressively narrow down relevant information,
enabling more targeted retrieval as reasoning progresses.
•Search-O1[ 14]: Integrates agentic retrieval mechanisms with large reasoning models, orchestrating
multi-step reasoning through explicit search strategies. The framework combines large reasoning
models with autonomous search capabilities, allowing the model to decide when and what to search
based on reasoning needs. Search-O1 employs explicit search tokens and integrates retrieval results
into the reasoning process, enabling coordinated search and reasoning behaviors.
•Search-R1[ 9]: Uses reinforcement learning to train models to reason and leverage search engines,
representing a recent search-based framework for multi-step reasoning. The method trains language
models to autonomously decide when to search and how to formulate search queries through
reinforcement learning. Search-R1 demonstrates the effectiveness of learning search strategies
through RL, enabling models to develop effective search behaviors through trial and error.
•R1-Searcher[ 25]: Implements recursive search mechanisms based on R1 architecture for complex
queries, enabling deeper exploration of search spaces. The method employs recursive search strate-
gies that allow the model to iteratively refine queries and explore search spaces more thoroughly.
R1-Searcher’s recursive approach enables handling of complex queries that require multiple levels
of reasoning, allowing for deeper information exploration.
•HierSearch[ 28]: Employs hierarchical search strategies that decompose questions at multiple
granularity levels, allowing for more structured reasoning processes. The framework decomposes
questions hierarchically, creating multiple levels of abstraction that guide the search process.
HierSearch’s multi-granularity approach enables more structured reasoning by organizing search at
different levels of detail.
•O2-Searcher[ 27]: Focuses on optimizing search efficiency through advanced architectural designs.
The method employs sophisticated search mechanisms designed to minimize unnecessary search
operations while maintaining high answer quality. O2-Searcher optimizes the trade-off between
search cost and performance, enabling efficient search-augmented reasoning.
•ZeroSearch[ 27]: Aims to incentivize search capabilities without explicit search operations,
representing a state-of-the-art search framework. The method trains models to internalize search
behaviors without requiring explicit search API calls, encouraging the model to develop search-
like reasoning patterns. ZeroSearch demonstrates an alternative approach to search-augmented
reasoning by learning implicit search strategies.
These baselines cover the spectrum from simple retrieval-augmented generation to sophisticated
search-based reasoning, allowing us to assess Dep-Search’s improvements in dependency modeling
and memory management.
A.3 Models
We conduct experiments using two model sizes:Qwen2.5-3B-InstructandQwen2.5-7B-Instruct
[22]. These models provide a good balance between performance and computational efficiency,
allowing us to evaluate Dep-Search’s effectiveness across different model scales. Qwen2.5 is a family
16

of large language models that demonstrate strong reasoning capabilities and instruction-following
abilities. The models are trained with extensive instruction tuning and demonstrate competitive
performance on various reasoning benchmarks. The 3B variant offers faster inference and lower
memory requirements, making it suitable for resource-constrained environments, while the 7B
variant provides stronger reasoning capabilities and better instruction understanding. Both variants
support the control tokens and structured reasoning required by Dep-Search, enabling comprehensive
evaluation of the framework’s dependency aware search mechanisms across model scales. The choice
of these model sizes allows us to demonstrate that Dep-Search’s improvements are consistent across
different model capacities, suggesting that the framework’s benefits are not limited to larger models.
A.4 Retrieval Corpus
All retrieval operations are performed over theWikipedia 2018corpus [ 11], which contains approxi-
mately 5.9 million passages from English Wikipedia articles. This corpus provides a comprehensive
knowledge base for multi-hop reasoning tasks and is consistent with the evaluation setup used in most
baseline methods. We use the same corpus for both training and evaluation to ensure fair comparison.
The corpus is preprocessed into passages of approximately 100 words each, enabling efficient dense
retrieval and reranking operations.
A.5 Implementation Details
We implement Dep-Search using PyTorch and the HuggingFace Transformers library. For retrieval,
we useqwen3-embeddingfor dense retrieval andqwen3-rerankerfor re-ranking [ 42], with top-
k= 5 documents retrieved per query. The memory buffer has a fixed capacity of Cmem= 20 entries,
managed using LRU eviction. For GRPO training, we sample K= 4 trajectories per question and
use a learning rate of 1×10−5with AdamW optimizer. The reward function uses thresholds k1= 10
for retrieval and k2= 8for decomposition, with penalty coefficients λret= 0.1 andλdec= 0.05 . We
train for 3 epochs with batch size 2 and gradient accumulation steps of 4. During inference, we use
temperature 0.7 and top-p 0.9 for generation, with a maximum of 16384 new tokens per trajectory.
B Dep-Search Algorithm
Algorithm 1Dep-Search Rollout with Search Call
Require:QuestionQ, policy modelπ θ, retriever Retr, initial memoryM 0, step budgetT
Ensure:Final answerAand trajectoryτ
1:T 0← ∅,C 0←[instr;Q], S 0←(T 0,C0,M 0), x← ∅, t←0
2:whileAnot emitted andt < Tdo
3:Samplea t∼πθ(· |x),x←x+a t
4:ifa tis<Decompose> then
5:UpdateT t+1by adding new sub-questions and dependency edges
6:else ifa tcloses <Retrieve> tag with queryr tthen
7:D t←Retr(r t)
8:Append <Retrieve_result> Dt</Retrieve_result> tox,C t+1← C t∪ Dt
9:Fret
t←Summarize(D t), updateM t+1withFret
t
10:else ifa tcloses <Memory> tag with queryqmem
tthen
11:SelectMread
t⊆ M tby recency and embedding similarity toqmem
t
12:Append <Memory_result> Mread
t</Memory_result> tox,C t+1← C t∪ Mread
t
13:else ifa tis<Conclusion> then
14:F t←Summarize(C t), updateM t+1via the LRU rule withF t
15:else ifa tis an answer-closing tokenthen
16:ExtractAfromxandbreak
17:end if
18:t←t+ 1
19:end while
20:Construct trajectoryτfrom(S t, at)tandreturn(A, τ)
17

C Decomposition Strategy Analysis
To analyze the effectiveness of dependency aware decomposition, we compare different decompo-
sition strategies on HotpotQA and 2WikiMHQA using Qwen2.5-7B-Instruct. Multi-hop reasoning
requires breaking down complex questions into dependent sub-questions, where later steps often rely
on results from earlier steps. However, existing approaches either ignore dependencies entirely or use
fixed decomposition patterns that cannot adapt to question complexity. We evaluate three strategies:
Sequential Decomposition that processes sub-questions without explicit dependencies, Two-step
Dependencies that models step-to-step dependencies, and QDMR Decomposition that allows the
model to determine both the number of steps and their dependency structure adaptively. This analysis
is crucial because explicit dependency modeling enables the model to structure reasoning chains
correctly, ensuring that prerequisite information is gathered before dependent steps are executed,
which is essential for accurate multi-hop reasoning.
Table 3: Decomposition strategy comparison on multi-hop datasets on Qwen2.5-7B-Instruct.
Strategy HotpotQA 2WikiMHQA Dependency Accuracy Avg.
Sequential Decomposition 38.2 39.1 0.0% 38.7
Two-step Dependencies 40.5 41.2 72.3% 40.9
QDMR Decomposition42.8 43.5 81.2% 43.2
Table 3 presents the comparison of different decomposition strategies.QDMR Decomposition
achieves the best performance, with an average score of 43.2 across the two datasets, outperforming
Sequential Decomposition by 4.5 points. The QDMR strategy achieves 81.2% dependency accuracy,
correctly identifying relationships between sub-questions in most cases. Sequential Decomposition,
similar to approaches like RA-ISF that process sub-questions without explicit dependency modeling,
shows the lowest performance, confirming that explicit dependency modeling is crucial for multi-hop
reasoning. Two-step Dependencies achieves intermediate performance, demonstrating that even
simple dependency modeling improves reasoning quality, but adaptive QDMR decomposition that
allows the model to determine both step count and dependency structure provides the best results.
These results validate that explicit dependency modeling enables correct reasoning order, where
prerequisite information is gathered before dependent steps, leading to more accurate multi-hop
reasoning.
18

D Training Prompt Template
Training Prompt Template
System Instruction:You are a helpful AI assistant that solves complex questions through
dependency aware reasoning with persistent memory.
Available Actions:
1.<Decompose> steps with dependencies </Decompose>
Break down the question into 2-4 steps with explicit dependencies. Each step must reference
prerequisite steps using “(step_number)” format. Dependencies form a directed acyclic graph
where sub-questions are solved in topological order, ensuring prerequisites are resolved
before dependent steps.
Example: “(1) Identify key entity. (2) Use (1) to locate factual reference. (3) Resolve
dependent query using (2). (4) Synthesize final answer from (1)–(3).”
2. <Retrieve> query </Retrieve> <Retrieve_result> retrieved documents
</Retrieve_result>
Upon emitting <Retrieve> followed by a query and the closing </Retrieve> , the environment
immediately performs two-stage retrieval (dense retrieval + reranking) and returns the top- k
retrieved documents wrapped in <Retrieve_result> tags. The model generates the query
autonomously, allowing it to determine what information to retrieve based on the current
reasoning context.
3.<Memory> query </Memory> <Memory_result> retrieved facts </Memory_result>
Upon emitting <Memory> followed by a query and the closing </Memory> , the envi-
ronment returns a mixture of Most Recent Memory entries and additional memory items
whose qwen3-embedding cosine similarity with the query exceeds a threshold, wrapped
in<Memory_result> tags. The model generates the query autonomously, allowing it to
determine what knowledge to fetch from memory based on the current reasoning needs.
4.<Conclusion> summary </Conclusion>
When long reasoning and evidence have accumulated, emit <Conclusion> to summarize the
preceding context into a few natural-language sentences, which are then written into the
memory buffer and can be reused via <Memory> .
E Case Studies
This section presents detailed case studies demonstrating how Dep-Search handles complex multi-hop
reasoning questions through dependency aware decomposition, retrieval, and memory management.
19

Case Study 1: Bridge Question on HotpotQA
Question:What is the nationality of the author of the book that won the Pulitzer Prize for
Fiction in 2018?
Dep-Search Reasoning Process:
To answer this question, I need to find the author of the book that won the Pulitzer Prize for
Fiction in 2018, and then determine that author’s nationality. Let me first check my memory to
see if I have relevant information stored.
<Memory> Pulitzer Prize Fiction 2018 author nationality </Memory> <Memory_result> (No
relevant memory found for this specific question) </Memory_result>
The memory doesn’t have this specific information. Let me retrieve information about the 2018
Pulitzer Prize winner.
<Retrieve> Pulitzer Prize for Fiction 2018 winner </Retrieve> <Retrieve_result> The Pulitzer
Prize for Fiction in 2018 was awarded to "Less" by Andrew Sean Greer. The novel is a comedic
story about a middle-aged novelist who travels around the world to avoid attending his ex-
boyfriend’s wedding. </Retrieve_result>
Good, I found that "Less" by Andrew Sean Greer won the prize. The retrieved information has
been automatically stored in memory. However, this result doesn’t include information about
the author’s nationality. Let me check memory again to see if I can find the author’s nationality
information.
<Memory> Andrew Sean Greer nationality birthplace </Memory> <Memory_result> "Less"
by Andrew Sean Greer won the 2018 Pulitzer Prize for Fiction. </Memory_result>
The memory only contains information about the book award, not the author’s nationality. I need
to retrieve more specific information about Andrew Sean Greer’s nationality.
<Retrieve> Andrew Sean Greer biography nationality birthplace </Retrieve> <Retrieve_result>
Andrew Sean Greer is an American novelist and short story writer. He was born on November
21, 1970, in Washington, D.C., United States. Greer has written several novels including "Less"
and "The Confessions of Max Tivoli." </Retrieve_result>
Perfect! I now have all the information needed. I’ve accumulated information from multiple
retrievals and memory checks. The context is getting long, so let me summarize the key findings
before proceeding.
<Conclusion> The 2018 Pulitzer Prize for Fiction was awarded to "Less" by Andrew Sean
Greer. Andrew Sean Greer is an American novelist born in Washington, D.C., United States.
</Conclusion>
Based on the information I’ve gathered, I found that Andrew Sean Greer won the Pulitzer Prize
for Fiction in 2018, and from the biography retrieval, I confirmed that he is an American author,
born in Washington, D.C., United States. Therefore, the nationality of the author is American.
Final Answer:American
20

Case Study 2: Comparison Question on 2WikiMultihopQA
Question:Which actor appeared in both the movie directed by Christopher Nolan that won the
Academy Award for Best Picture and the movie where Leonardo DiCaprio won Best Actor?
Dep-Search Reasoning Process:
This is a complex comparison question that requires finding information about two different
movies and then comparing their casts. I need to identify: (1) Christopher Nolan’s Best Picture
winner, and (2) the movie where Leonardo DiCaprio won Best Actor. These two sub-questions
can be answered independently, so I can retrieve them in parallel or sequentially. Let me start
with the first one.
<Retrieve> Christopher Nolan Best Picture Academy Award winner
</Retrieve> <Retrieve_result> Christopher Nolan’s "Oppenheimer" won the Academy
Award for Best Picture in 2024. The film stars Cillian Murphy, Emily Blunt, Robert Downey Jr.,
Matt Damon, and Florence Pugh. </Retrieve_result>
Good, I found that "Oppenheimer" won Best Picture. Now let me retrieve information about
Leonardo DiCaprio’s Best Actor win.
<Retrieve> Leonardo DiCaprio Best Actor Academy Award winner movie
</Retrieve> <Retrieve_result> Leonardo DiCaprio won the Academy Award for Best
Actor for his role in "The Revenant" (2015), directed by Alejandro González Iñárritu. The film
also stars Tom Hardy, Domhnall Gleeson, and Will Poulter. </Retrieve_result>
Now I have both movies: "Oppenheimer" and "The Revenant". Let me check my memory to
see if I have stored information about these casts, which would help me compare them more
efficiently.
<Memory> actors in Oppenheimer and The Revenant </Memory> <Memory_result> Oppen-
heimer (2024) cast: Cillian Murphy, Emily Blunt, Robert Downey Jr., Matt Damon, Florence
Pugh. The Revenant (2015) cast: Leonardo DiCaprio, Tom Hardy, Domhnall Gleeson, Will
Poulter. </Memory_result>
Comparing the casts: "Oppenheimer" has Cillian Murphy, Emily Blunt, Robert Downey Jr.,
Matt Damon, Florence Pugh. "The Revenant" has Leonardo DiCaprio, Tom Hardy, Domhnall
Gleeson, Will Poulter. I don’t see any obvious overlap. However, I recall that Matt Damon and
Leonardo DiCaprio have worked together before. Let me verify if Matt Damon appeared in "The
Revenant" or if there’s any connection.
<Retrieve> Matt Damon Leonardo DiCaprio movies together The Revenant
</Retrieve> <Retrieve_result> Matt Damon and Leonardo DiCaprio have appeared to-
gether in "The Departed" (2006) and "The Revenant" (2015), where Matt Damon had a cameo
role. However, Matt Damon did not appear in "Oppenheimer." </Retrieve_result>
I see that Matt Damon had a cameo in "The Revenant", but he did not appear in "Oppenheimer".
After careful comparison, no actor appears in both films. I’ve now accumulated multiple retrieval
results and memory accesses. The context is getting long with information about both movies,
their casts, and additional verification queries. Let me summarize the key findings to compress
the context and store them in memory for potential future use.
<Conclusion> Christopher Nolan’s "Oppenheimer" won Best Picture in 2024 with cast including
Cillian Murphy, Emily Blunt, Robert Downey Jr., Matt Damon, and Florence Pugh. Leonardo
DiCaprio won Best Actor for "The Revenant" in 2015, which also stars Tom Hardy, Domhnall
Gleeson, and Will Poulter. </Conclusion>
Now let me analyze the information I’ve gathered. From the retrievals and memory, I have the
complete cast lists for both films. "Oppenheimer" stars: Cillian Murphy, Emily Blunt, Robert
Downey Jr., Matt Damon, and Florence Pugh. "The Revenant" stars: Leonardo DiCaprio, Tom
Hardy, Domhnall Gleeson, and Will Poulter. I also verified that Matt Damon had a cameo in
"The Revenant" but did not appear in "Oppenheimer". After comparing all the actors from both
films, I can confirm that no actor appears in both films.
Final Answer:No actor appeared in both films.
Final Answer:No actor appeared in both films.
21

Case Study 3: Multi-hop Question with Memory Reuse
Question:What is the capital city of the country where the author of "1984" was born?
Dep-Search Reasoning Process:
This is a multi-hop question that requires several steps. Let me first check my memory to see if I
have relevant information.
<Memory> author of 1984 birth country capital city </Memory> <Memory_result> (No rele-
vant memory found for this specific question) </Memory_result>
Memory doesn’t have this information. Let me try retrieving information about the author of
"1984".
<Retrieve> author of 1984 novel </Retrieve> <Retrieve_result> "1984" is a dystopian
novel published in 1949. The novel explores themes of totalitarianism and surveillance.
</Retrieve_result>
This retrieval result doesn’t include the author’s name or birth information. The retrieved
information is not sufficient for answering the question. Let me try a more specific query.
<Retrieve> author of 1984 novel George Orwell </Retrieve> <Retrieve_result> "1984" is a
dystopian novel written by George Orwell, published in 1949. George Orwell is the pen name of
Eric Arthur Blair, born on June 25, 1903, in Motihari, Bengal Presidency, British India (now in
Bihar, India). </Retrieve_result>
Excellent! I found that George Orwell (Eric Arthur Blair) was born in British India, which is
now India. The retrieved information has been automatically summarized and stored in memory.
Now I know the author was born in India. Let me check my memory to retrieve this stored fact.
<Memory> George Orwell birth country India </Memory> <Memory_result> "1984" was
written by George Orwell (Eric Arthur Blair), who was born in British India (now India).
</Memory_result>
Perfect! I retrieved the birth country information from memory. Now I need to find the capital
city of India. Let me check memory first to see if I have this information.
<Memory> India capital city </Memory> <Memory_result> "1984" was written by George
Orwell (Eric Arthur Blair), who was born in British India (now India). </Memory_result>
The memory has information about India but not the capital city. Let me retrieve this information.
<Retrieve> capital city of India </Retrieve> <Retrieve_result> The capital city of India is New
Delhi. New Delhi serves as the capital of India and is part of the larger Delhi metropolitan area.
</Retrieve_result>
Perfect! I found that New Delhi is the capital of India. The retrieved information has been
automatically stored in memory. Let me verify by checking memory.
<Memory> New Delhi India capital </Memory> <Memory_result> The capital city of India is
New Delhi. New Delhi serves as the capital of India and is part of the larger Delhi metropolitan
area. </Memory_result>
Now I have all the information needed to answer the question. From my retrievals and memory,
I found that: (1) The author of "1984" is George Orwell, (2) George Orwell was born in British
India, which is now India, (3) The capital city of India is New Delhi. Therefore, the capital city
of the country where the author of "1984" was born is New Delhi.
Final Answer:New Delhi
22

Case Study 4: Complex Dependency Chain
Question:What is the population of the city where the university that awarded the Nobel Prize
winner in Physics 2023 their PhD is located?
Dep-Search Reasoning Process:
This is a complex multi-hop question with a long dependency chain. Let me try retrieving
information about the 2023 Nobel Prize winners.
<Retrieve> Nobel Prize Physics 2023 </Retrieve> <Retrieve_result> The Nobel Prize in
Physics recognizes outstanding contributions to the field of physics. The prize has been awarded
annually since 1901. </Retrieve_result>
This retrieval result is too general and doesn’t contain information about the 2023 winners or their
PhD universities. The retrieved information is not useful for answering this specific question.
Given the complexity of this multi-hop question, let me decompose it into clear dependent steps
to better structure my reasoning.
<Decompose> (1) Identify the Nobel Prize winner(s) in Physics 2023. (2) Find the university
where the winner from (1) received their PhD. (3) Determine the city where the university from
(2) is located. (4) Find the population of the city from (3). </Decompose>
Now let me proceed step by step following the decomposition. Step 1: Find the 2023 Physics
Nobel Prize winners.
<Retrieve> Nobel Prize Physics 2023 winner </Retrieve> <Retrieve_result> The 2023 Nobel
Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz, and Anne L’Huillier for their
work on attosecond physics. Anne L’Huillier received her PhD from the University of Paris in
1986. </Retrieve_result>
Great! I found that there were three winners, and I also see that Anne L’Huillier received her
PhD from the University of Paris. The retrieved information has been automatically summarized
and stored in memory. Step 2: Find the university location. Let me check my memory first.
<Memory> Anne L’Huillier PhD university University of Paris </Memory> <Memory_result>
The 2023 Nobel Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz, and Anne
L’Huillier. Anne L’Huillier received her PhD from the University of Paris. </Memory_result>
I retrieved the university information from memory. Step 3: Find the city where the University
of Paris is located.
<Retrieve> University of Paris location city </Retrieve> <Retrieve_result> The University of
Paris (also known as Sorbonne University) is located in Paris, France. The university has multiple
campuses throughout Paris, with its main campus in the Latin Quarter. </Retrieve_result>
Perfect! The University of Paris is located in Paris, France. Step 4: Find the population of Paris.
I’ve now accumulated several retrieval results. The context is getting long with multiple facts.
Let me summarize the key information before proceeding to the final retrieval.
<Conclusion> The 2023 Nobel Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz,
and Anne L’Huillier. Anne L’Huillier received her PhD from the University of Paris, which is
located in Paris, France. </Conclusion>
Now I can proceed to find the population of Paris. Let me check memory first to see if I have
this information stored.
<Memory> Paris France population </Memory> <Memory_result> The 2023 Nobel Prize
in Physics was awarded to Pierre Agostini, Ferenc Krausz, and Anne L’Huillier. Anne
L’Huillier received her PhD from the University of Paris, which is located in Paris, France.
</Memory_result>
The memory has information about Paris but not the population. Let me retrieve this information.
<Retrieve> Paris France population 2023 </Retrieve> <Retrieve_result> Paris, the capital city
of France, has a population of approximately 2.1 million people within the city limits (as
of 2023). The greater Paris metropolitan area has a population of over 12 million people.
</Retrieve_result>
Perfect! I found the population information. The retrieved information has been automatically
stored in memory. Let me verify by checking memory one more time.
<Memory> Paris population 2023 </Memory> <Memory_result> Paris, the capital city of
France, has a population of approximately 2.1 million people within the city limits (as of
2023). The greater Paris metropolitan area has a population of over 12 million people.
</Memory_result>
Now I have completed all four steps of the decomposition. Therefore, the population of the city
where the university that awarded the Nobel Prize winner in Physics 2023 their PhD is located is
approximately 2.1 million within city limits.
Final Answer:Approximately 2.1 million (within city limits) or over 12 million (metropolitan
area)23

Case Study 5: Decomposition with Parallel Sub-questions
Question:What are the birth years of both the director of "Inception" and the lead actor of "The
Dark Knight"?
Dep-Search Reasoning Process:
This question requires finding information about two different entities (a director and an actor)
from two different movies. These sub-questions can be answered independently, but I need to
structure them clearly. Let me decompose the question first.
<Decompose> (1) Find the director of "Inception" and their birth year. (2) Find the lead actor
of "The Dark Knight" and their birth year. (3) Combine the birth years from (1) and (2).
</Decompose>
Now let me proceed step by step. Step 1: Find the director of "Inception" and their birth year.
<Retrieve> director of Inception movie </Retrieve> <Retrieve_result> "Inception" (2010) is a
science fiction film directed by Christopher Nolan. The film stars Leonardo DiCaprio, Marion
Cotillard, Tom Hardy, and Ellen Page. </Retrieve_result>
Good, I found that Christopher Nolan directed "Inception". The retrieved information has been
automatically stored in memory. Now I need to find Christopher Nolan’s birth year.
<Retrieve> Christopher Nolan birth year biography </Retrieve> <Retrieve_result> Christopher
Nolan is a British-American filmmaker born on July 30, 1970, in London, England. He is known
for directing films such as "Inception", "The Dark Knight", and "Interstellar". </Retrieve_result>
Perfect! Christopher Nolan was born in 1970. Step 2: Find the lead actor of "The Dark Knight"
and their birth year.
<Retrieve> lead actor The Dark Knight movie </Retrieve> <Retrieve_result> "The Dark
Knight" (2008) is a superhero film directed by Christopher Nolan. The film stars Christian
Bale as Bruce Wayne/Batman, Heath Ledger as the Joker, and Aaron Eckhart as Harvey Dent.
</Retrieve_result>
I found that Christian Bale is the lead actor (playing Batman). The retrieved information has
been automatically stored in memory. Now I need to find Christian Bale’s birth year.
<Memory> Christian Bale birth year </Memory> <Memory_result> "The Dark Knight" (2008)
stars Christian Bale as Bruce Wayne/Batman. </Memory_result>
The memory has information about the movie but not the birth year. Let me retrieve Christian
Bale’s birth year.
<Retrieve> Christian Bale birth year biography </Retrieve> <Retrieve_result> Christian Bale
is a British actor born on January 30, 1974, in Haverfordwest, Wales. He is known for his roles
in "The Dark Knight" trilogy, "American Psycho", and "The Fighter". </Retrieve_result>
Perfect! Christian Bale was born in 1974. The retrieved information has been automatically
stored in memory. Step 3: Combine the birth years. I’ve now gathered information about both
birth years. Let me verify by checking memory.
<Memory> Christopher Nolan Christian Bale birth years </Memory> <Memory_result>
Christopher Nolan is a British-American filmmaker born on July 30, 1970. Christian Bale
is a British actor born on January 30, 1974. </Memory_result>
Perfect! I have confirmed both birth years from memory. I’ve accumulated information from
multiple retrievals. The context is getting long, so let me summarize the key findings.
<Conclusion> Christopher Nolan directed "Inception" and was born in 1970. Christian Bale
starred as the lead actor in "The Dark Knight" and was born in 1974. </Conclusion>
Now I have completed all three steps of the decomposition. From my retrievals and memory, I
found that: (1) Christopher Nolan directed "Inception" and was born in 1970, (2) Christian Bale
was the lead actor of "The Dark Knight" and was born in 1974. Therefore, the birth years are:
Christopher Nolan was born in 1970, and Christian Bale was born in 1974.
Final Answer:Christopher Nolan was born in 1970, and Christian Bale was born in 1974.
24

Case Study 6: Decomposition with Complex Dependencies
Question:What is the release year of the first movie directed by the person who won the
Academy Award for Best Director in 2014?
Dep-Search Reasoning Process:
This is a complex question with multiple dependent steps. I need to find the Best Director winner
in 2014, then find their first directed movie, and finally find that movie’s release year. Each
step depends on the previous one. Let me decompose this question to clarify the dependency
structure.
<Decompose> (1) Identify the winner of the Academy Award for Best Director in 2014. (2)
Find the first movie directed by the person from (1). (3) Determine the release year of the movie
from (2). </Decompose>
Now let me proceed step by step following the dependencies. Step 1: Find the Best Director
winner in 2014.
<Retrieve> Academy Award Best Director 2014 winner </Retrieve> <Retrieve_result> The
Academy Award for Best Director in 2014 was awarded to Alejandro González Iñárritu for
"Birdman or (The Unexpected Virtue of Ignorance)". Iñárritu is a Mexican filmmaker known for
his distinctive visual style and complex narratives. </Retrieve_result>
Excellent! I found that Alejandro González Iñárritu won Best Director in 2014. The retrieved
information has been automatically stored in memory. Step 2: Find the first movie directed by
Alejandro González Iñárritu.
<Memory> Alejandro González Iñárritu first movie directed </Memory> <Memory_result>
The Academy Award for Best Director in 2014 was awarded to Alejandro González Iñárritu for
"Birdman". </Memory_result>
The memory has information about the award but not about his first movie. Let me retrieve
information about Iñárritu’s filmography.
<Retrieve> Alejandro González Iñárritu first movie filmography debut
</Retrieve> <Retrieve_result> Alejandro González Iñárritu made his directorial debut
with "Amores perros" (Love’s a Bitch) in 2000. The film is a Mexican crime drama that follows
three interconnected stories set in Mexico City. It was Iñárritu’s first feature film and received
critical acclaim. </Retrieve_result>
Perfect! I found that "Amores perros" (2000) was Iñárritu’s first directed movie. The retrieved
information has been automatically stored in memory. Step 3: The release year is already
included in the previous result - 2000.
<Memory> Amores perros release year </Memory> <Memory_result> Alejandro González
Iñárritu made his directorial debut with "Amores perros" in 2000. </Memory_result>
I can confirm from memory that "Amores perros" was released in 2000. I’ve successfully
completed all three steps of the decomposition. Let me verify the complete chain by checking
memory one more time.
<Memory> Alejandro González Iñárritu Best Director 2014 first movie Amores perros
</Memory> <Memory_result> The Academy Award for Best Director in 2014 was awarded to
Alejandro González Iñárritu for "Birdman". Alejandro González Iñárritu made his directorial
debut with "Amores perros" in 2000. </Memory_result>
Perfect! The memory confirms the complete chain: Best Director 2014→Alejandro González
Iñárritu→first movie "Amores perros" →release year 2000. I’ve accumulated information
from multiple steps. The context is getting long, so let me summarize the key findings.
<Conclusion> Alejandro González Iñárritu won the Academy Award for Best Director in 2014.
He made his directorial debut with "Amores perros". </Conclusion>
Now I have completed all three steps of the decomposition. From my retrievals and memory, I
found that: (1) Alejandro González Iñárritu won the Academy Award for Best Director in 2014,
(2) His first directed movie was "Amores perros", (3) "Amores perros" was released in 2000.
Therefore, the release year of the first movie directed by the person who won the Academy
Award for Best Director in 2014 is 2000.
Final Answer:2000
25

Case Study 7: Decomposition After Failed Retrieval
Question:What is the total number of books written by the author who won the Nobel Prize in
Literature in 2017?
Dep-Search Reasoning Process:
This question requires finding the Nobel Prize winner in Literature for 2017, then finding how
many books that author has written. Let me first check my memory.
<Memory> Nobel Prize Literature 2017 author books </Memory> <Memory_result> (No rele-
vant memory found for this specific question) </Memory_result>
Memory doesn’t have this information. Let me try retrieving information about the 2017 Nobel
Prize in Literature.
<Retrieve> Nobel Prize Literature 2017 </Retrieve> <Retrieve_result> The Nobel Prize in
Literature is awarded annually to an author who has produced outstanding work in the field of
literature. </Retrieve_result>
This retrieval result is too general and doesn’t contain information about the 2017 winner or their
books. The retrieved information is not useful for answering this specific question. Given that
this is a multi-step question, let me decompose it to better structure my reasoning.
<Decompose> (1) Identify the winner of the Nobel Prize in Literature in 2017. (2) Find the total
number of books written by the author from (1). </Decompose>
Now let me proceed step by step. Step 1: Find the 2017 Nobel Prize in Literature winner.
<Retrieve> Nobel Prize Literature 2017 winner </Retrieve> <Retrieve_result> The Nobel Prize
in Literature for 2017 was awarded to Kazuo Ishiguro "who, in novels of great emotional force,
has uncovered the abyss beneath our illusory sense of connection with the world". Ishiguro is
a British novelist of Japanese origin, known for works such as "The Remains of the Day" and
"Never Let Me Go". </Retrieve_result>
Excellent! I found that Kazuo Ishiguro won the Nobel Prize in Literature in 2017. The retrieved
information has been automatically stored in memory. Step 2: Find the total number of books
written by Kazuo Ishiguro.
<Memory> Kazuo Ishiguro books bibliography </Memory> <Memory_result> The Nobel
Prize in Literature for 2017 was awarded to Kazuo Ishiguro, a British novelist known for works
such as "The Remains of the Day" and "Never Let Me Go". </Memory_result>
The memory mentions some books but doesn’t have the total count. Let me retrieve more specific
information about Ishiguro’s bibliography.
<Retrieve> Kazuo Ishiguro total number of books bibliography complete list
</Retrieve> <Retrieve_result> Kazuo Ishiguro has published eight novels: "A Pale View of
Hills" (1982), "An Artist of the Floating World" (1986), "The Remains of the Day" (1989),
"The Unconsoled" (1995), "When We Were Orphans" (2000), "Never Let Me Go" (2005), "The
Buried Giant" (2015), and "Klara and the Sun" (2021). He has also published a collection of
short stories and several screenplays. </Retrieve_result>
Perfect! I found that Kazuo Ishiguro has published eight novels. The retrieved information has
been automatically stored in memory. Let me verify by checking memory.
<Memory> Kazuo Ishiguro eight novels bibliography </Memory> <Memory_result> The No-
bel Prize in Literature for 2017 was awarded to Kazuo Ishiguro. Kazuo Ishiguro has published
eight novels: "A Pale View of Hills" (1982), "An Artist of the Floating World" (1986), "The
Remains of the Day" (1989), "The Unconsoled" (1995), "When We Were Orphans" (2000),
"Never Let Me Go" (2005), "The Buried Giant" (2015), and "Klara and the Sun" (2021).
</Memory_result>
Confirmed! Kazuo Ishiguro has published eight novels. I’ve successfully completed both steps
of the decomposition. The context is getting long, so let me summarize the key findings.
<Conclusion> Kazuo Ishiguro won the Nobel Prize in Literature in 2017. He has published
multiple novels and other works. </Conclusion>
Now I have completed both steps of the decomposition. From my retrievals and memory, I
found that: (1) Kazuo Ishiguro won the Nobel Prize in Literature in 2017, (2) He has published
eight novels, plus additional works including short stories and screenplays. Therefore, the total
number of books written by the author who won the Nobel Prize in Literature in 2017 is eight
novels (plus additional works including short stories and screenplays).
Final Answer:Eight novels (plus additional works including short stories and screenplays)
26