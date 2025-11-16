# Q-RAG: Long Context Multi-step Retrieval via Value-based Embedder Training

**Authors**: Artyom Sorokin, Nazar Buzun, Alexander Anokhin, Oleg Inozemcev, Egor Vedernikov, Petr Anokhin, Mikhail Burtsev, Trushkov Alexey, Yin Wenshuai, Evgeny Burnaev

**Published**: 2025-11-10 17:31:02

**PDF URL**: [https://arxiv.org/pdf/2511.07328v1](https://arxiv.org/pdf/2511.07328v1)

## Abstract
Retrieval-Augmented Generation (RAG) methods enhance LLM performance by efficiently filtering relevant context for LLMs, reducing hallucinations and inference cost. However, most existing RAG methods focus on single-step retrieval, which is often insufficient for answering complex questions that require multi-step search. Recently, multi-step retrieval approaches have emerged, typically involving the fine-tuning of small LLMs to perform multi-step retrieval. This type of fine-tuning is highly resource-intensive and does not enable the use of larger LLMs. In this work, we propose Q-RAG, a novel approach that fine-tunes the Embedder model for multi-step retrieval using reinforcement learning (RL). Q-RAG offers a competitive, resource-efficient alternative to existing multi-step retrieval methods for open-domain question answering and achieves state-of-the-art results on the popular long-context benchmarks Babilong and RULER for contexts up to 10M tokens.

## Full Text


<!-- PDF content starts -->

Under review as a conference paper
Q-RAG: LONGCONTEXTMULTI-STEPRETRIEVAL
VIAVALUE-BASEDEMBEDDERTRAINING
Artyom Sorokin1,2, Nazar Buzun3, Alexander Anokhin1, Oleg Inozemcev1, Egor Vedernikov1,
Petr Anokhin2, Mikhail Burtsev4, Trushkov Alexey6, Yin Wenshuai5, Evgeny Burnaev1,2
1Applied AI, Moscow, Russia
2Learnable Intelligence Lab, Moscow, Russia
3CILAB.AI, Moscow, Russia
4London Institute for Mathematical Sciences, London, UK
5Higher School of Economics, Moscow, Russia
6Independent Researcher
griver29@gmail.com, n.buzun@cilab.ai
ABSTRACT
Retrieval-Augmented Generation (RAG) methods enhance LLM performance by
efficiently filtering relevant context for LLMs, reducing hallucinations and infer-
ence cost. However, most existing RAG methods focus on single-step retrieval,
which is often insufficient for answering complex questions that require multi-
step search. Recently, multi-step retrieval approaches have emerged, typically
involving the fine-tuning of small LLMs to perform multi-step retrieval. This type
of fine-tuning is highly resource-intensive and does not enable the use of larger
LLMs. In this work, we propose Q-RAG, a novel approach that fine-tunes the Em-
bedder model for multi-step retrieval using reinforcement learning (RL). Q-RAG
offers a competitive, resource-efficient alternative to existing multi-step retrieval
methods for open-domain question answering and achieves state-of-the-art results
on the popular long-context benchmarks Babilong and RULER for contexts up to
10M tokens.
1 INTRODUCTION
Large language models (LLMs) have achieved impressive results across a wide range of tasks
(Novikov et al., 2025; Guo et al., 2025; Yang et al., 2025). However, they still face some sev-
eral fundamental limitations such as static knowledge, computational inefficiency on long contexts,
degraded performance caused by attention dilution, and hallucinations (Hsieh et al., 2024; Kuratov
et al., 2024; Liu et al., 2025). Retrieval-Augmented Generation (RAG) is one of the most widely
used techniques to address these issues (Yu et al., 2024).
RAG works by extracting only the most relevant parts from a large external corpus or context, such as
newly added knowledge or lengthy texts. This allows LLMs to operate on shorter and more focused
inputs, improving efficiency and output quality. Most current RAG methods rely on single-step re-
trieval. This setup performs well in relatively simple tasks like Needle-in-a-Haystack (Hsieh et al.,
2024). Still, more complex problems require multi-step interaction with the context. Multi-step
retrieval can be viewed as a form of search-based reasoning. There are several existing approaches
to multi-step retrieval reasoning. One direction involves constructing a knowledge graph from the
retrieved information (Ma et al., 2025; Li et al., 2024). These methods are often slow at infer-
ence time, since the LLM must process the entire context to build the graph for each new input.
Another line of work uses LLM agents, which interleave RAG queries with LLM-generated instruc-
tions (Singh et al., 2025; Anokhin et al., 2024). These systems are sensitive to noisy or inaccurate
retrieved passages, which may disrupt the generation of future queries. This shows the need for
joint optimization of the retrieval and generation components. Recently, methods have emerged that
fine-tune LLMs to interact more effectively with retrieval tools (Song et al., 2025; Jin et al., 2025;
Chen et al., 2025). These methods tend to perform better, but they require expensive fine-tuning
of the LLM itself. This makes them impractical for large models and limits accessibility for most
researchers and practitioners.
1arXiv:2511.07328v1  [cs.LG]  10 Nov 2025

Under review as a conference paper
In this work, we focus on developing a resource-efficient multi-step RAG approach using reinforce-
ment learning. Instead of fine-tuning an LLM, we train an agent that performs retrieval directly in
the latent space of text chunk embeddings. This allows us to learn a compact and efficient model
using value-based RL methods.
Our approach achieves state-of-the-art results on long-context commonsence reasoning, multi-hop
QA, and NIAH tasks with contexts up to 10 million tokens. It also performs competitively on
open-domain QA benchmarks such as Musique and HotpotQA (Yang et al., 2018;?), while being
significantly faster and cheaper to train and run compared to existing multi-step RAG methods. Our
contributions are the following:
• We propose a new method for training a multi-step retrieval agent using temporal difference
reinforcement learning.
• We achieve state-of-the-art results on benchmarks that require commonsense reasoning and
NIAH tasks over ultra long contexts (up to 10M tokens).
• We introduce a new way to incorporate temporal information into the multi-step embedder,
enabling temporal reasoning during retrieval. Our temporal reasoning mechanism general-
izes well to long contexts at inference time.
2 RELATEDWORKS
There are several main directions for tackling complex retrieval scenarios on long context tasks.
A highly popular approach involves building fine-tuning free LLM Agents that combine off-the-shelf
retrievers with LLMs, such as Search-o1 (Li et al., 2025). Many of these works further enhance
retrieval quality by constructing large knowledge graphs over the context, which, while requiring
little additional training, are extremely slow at inference due to the need for LLMs to process the
entire context, e.g. GraphReader (Li et al., 2024), HippoRAG (Jimenez Gutierrez et al., 2024),
AriGraph (Anokhin et al., 2024).
Another line of work fine-tunes LRMs to perform multi-step retrieval, allowing the model to gener-
ate intermediate search queries inside the reasoning for long contexts. The first work to apply this
idea was IM-RAG (Yang et al., 2024), which fine-tuned the LLM with a frozen embedder using
PPO (Schulman et al., 2017). More recent papers, such as R1-Searcher (Song et al., 2025), Search-
R1 (Jin et al., 2025), RAG-RL (Huang et al., 2025), and ReSearcher (Chen et al., 2025), extended
this direction by employing GRPO (Shao et al., 2024) for the task. Unlike these methods, which
freeze the embedder and fine-tune the LLM, our approach fine-tunes only the embedder, allowing it
to pair with LLMs of any size, including proprietary ones, while keeping fine-tuning efficient and
inexpensive.
A different approach is to fine-tune the retriever itself using feedback from the LLM, as in Re-
Plug (Shi et al., 2024). This direction is most similar to ours, but RePlug did not address multi-step
reasoning or use reinforcement learning in this setting. BeamRetriever (Zhang et al., 2024) achieves
state-of-the-art results on short-context QA by training a reranker for BeamSearch-style planning.
In contrast, Q-RAG trains the embedder with reinforcement learning, enabling faster inference and
better scalability to long contexts through efficient vector similarity instead of transformer-based
trajectory scoring.
Extremely long-sequence processing is demonstrated by models that combine recurrence with
Transformer architecture. The Mamba family of state space models (Gu & Dao, 2024) replaces at-
tention with structured recurrent dynamics, offering linear-time scalability and strong performance
on long sequences, though often at the cost of weaker in-context learning and less expressive token-
to-token interaction compared to Transformer-based architectures. The Recurrent Memory Trans-
former (RMT) (Bulatov et al., 2022) introduces segment-level recurrence by passing memory tokens
between fixed-size segments, enabling Q&A on sequences up to 10M tokens. Titans (Behrouz et al.,
2024) frames recurrent memory training as a meta-learning problem, showing scaling beyond 2M
tokens. Building on this idea, ATLAS (Behrouz et al., 2025) increases memory capacity, achieving
better long-context performance than both RMT and Titans. The Associative Recurrent Memory
Transformer (ARMT) (Rodkin et al., 2024) employs quasi-linear, associative attention in each layer
2

Under review as a conference paper
and attains the best long-context scores among recurrent models. Our approach outperforms all of
these models on contexts beyond 1M tokens while belonging to a different class of methods.
LongRoPE2 (Shang et al., 2025) tackles the positional encoding bottleneck, extending the effective
context window of pre-trained LLMs to 128K tokens while retaining short-context performance
through RoPE rescaling and mixed-window training.
3 METHODS
...John 
stayed 
late 
at 
the 
office...
Q-RAG 
Agent
Environment
...He 
briefly 
stopped 
by 
home...
 
hebought 
something 
at 
the 
pharmacy...
...that, 
he 
spent 
twenty 
minutes 
at 
a 
café...
Outside 
his 
place, 
he 
realized 
his 
keys 
were 
...He 
stayed 
overnight 
at 
his 
neighbor?s 
place...
Question: 
Where 
could 
John 
have 
forgotten 
his 
keys?
Answer: 
cafe, 
pharmacy
State 
Embedder
Action 
Embedder 
Q 
values
reward 
function 
/ 
critic. 
. 
....that, 
he 
spent 
twenty 
minutes 
at 
a 
café...
Long-Context 
Document
Next 
timestep
Figure 1: Q-RAG agent interacts with multi-step retrieval environment. The starting states 0contains
the initial queryq. At the start of the episode, the agent embeds all chunks of the long contextC.
At each stept, the agent computes a vector embedding of the current states t, which includesq
and all previously selected chunks. For every chunkci∈At, the utility of retrieving it is evaluated
by theQ-functionQ θ(st, a=ci). The policyπ θselects the next chunk fromA twith probability
proportional to itsQ θ(st, ci)value.
3.1 PRELIMINARIES
LetDbe a dataset of triples(C, q, y), whereCis a long context,qis an initial query, andyis
the gold answer. The queryqcan be either a user question aboutCor a generated claim whose
factuality or consistency with earlier parts ofCmust be verified. We assumeCis pre-segmented
into non-overlapping1text chunksC={c(i)}m
i=1in document order. The agent’s goal is to identify
the information inCthat is missing fromqbut necessary to produce the correct answery. We model
multi-step retrieval as a finite-horizon Markov Decision Process, or MDP(S,A, p, r, γ), whereA
is the action space,Sis the state space,ris the reward function,pis the (deterministic) transition
function, andγ∈[0,1]is the discount factor. At stept= 0, the action set isA 0=C, where
an actiona t∈A tselects one chunk. At later steps, previously selected chunks are removed so
At=C\ {a 0, . . . , a t−1}. Superscripts indicate document positions and subscripts indicate episode
timesteps. The notationai(equivalentlyc(i)) denotes the chunk/action at positioniin the document;
selecting the chunk with indexiat steptis writtenai
t. Symbolscandaare used interchangeably,
depending on context.
States are ordered lists that always begin with the query,s t= ord([q, a 0, . . . , a t−1]), whereord(·)
sorts by the original document order to avoid permutation ambiguity; the initial state contains only
the query,s 0= [q]. Transitions are deterministic,p(s t, at) = ord([q, a 0, . . . , a t−1, at]). An episode
terminates either when a step budgetTis reached or when a special STOPaction is taken.
1Chunk overlapping may complicate the explanation but does not affect our proposed solution.
3

Under review as a conference paper
When supervision provides a set of support factsF⋆⊆C, we use a sparse terminal reward: the
reward is0at all intermediate steps, and at the end of the episode it is1if all support facts are
included in the final state (otherwise0). When only answer supervision is available, one could
instead use an LLM to generateˆyfrom the final state and define a terminal reward via an answer-
quality metric (e.g., exact match or F1). In this work we do not pursue LLM-based rewards; all
reported experiments rely on the support-fact signal, and exploring LLM-based reward design is left
for future work.
3.2 VALUE-BASEDRLFOREMBEDDERFINE-TUNING
Action selection in multi-step retrieval is performed by a value-based agent. Specifically, maximum-
entropy reinforcement learning (Ziebart, 2010; Haarnoja et al., 2018) is adopted together with the
corresponding definitions of the softQπandVπvalue functions for a policyπ:
Qπ(s, a) =r(s, a) +γVπ(s′=p(s, a))(1)
Vπ(s) =E a∼π(·|s) [Qπ(s, a)−αlogπ(a|s)](2)
Here,α >0is a temperature that controls the strength of exploration. This choice is primarily mo-
tivated by the need for effective exploration in the long-context multi-step retrieval environment. In
Q-RAG, Q function is approximated using two embedders for states and actions. The state embed-
derE s(st;θ1)∈Rdproduces vector embedding for the current states t, while the action embedder
Ea(ai, i;θ 2)∈Rdemploy rotary position embeddings to encode both the candidate chunk content
and its document-position indexi. Q values are then estimated by an inner product between two em-
beddings:Q θ(s, ai) =⟨E s(s;θ 1), Ea(ai, i;θ 2)⟩. This factorization is theoretically grounded; we
derive its convergence guarantees with explicit rates in Appendix A. GivenQ θ, the chunk selection
probability is computed using a Boltzmann policy:
π(at|st) =exp1
α(Qθ(st, at)−q)P
a∈A texp1
α(Qθ(st, a)−q)(3)
withq= max a∈A tQθ(st, a)and temperatureαannealed from an initial value to zero during train-
ing (proportionally to the learning rate).
As the backbone Temporal Difference learning algorithm, we adopt the recent PQN method by
Gallici et al.. Compared to DQN (Mnih et al., 2015), PQN removes the need for a replay buffer. In
our setting with a large number of chunks, a replay buffer would require re-embedding all document
chunks for each sample drawn from the replay buffer to estimateV/Qvalues for subsequent states
st+1. Which significantly slows the training process and increases memory space requirements.
Using PQN enables an on-policy value-based training that avoids these costs. The key departures
in Q-RAG, relative to the original PQN backbone, are the use of soft value functions and target
networks. Ablation results demonstrating the benefit of these choices are reported in Section 4.5.
As the training target, rather than the one-step return (see r.h.s. in Eq. 1), aλ-return is used to
improve stability and learning speed:
Gλ
t= (1−λ)T−t−1X
n=1λn−1Gt:t+n +λT−t−1Gt,
whereG t:t+n =Pn
k=1γk−1rt+k+Vθ′(st+n). The approximation of the state value function can
be computed from Q values in the case of discrete actions:
Vθ′(st) =αlogX
a∈A texpQθ′(st, a)
α
(4)
Hereθ′denotes slowly updated target network parameters. The model parametersθare finetuned to
minimize the mean squared error to theλ-returns:
LQ=E[(Q θ(st, at)−Gλ
t)2](5)
The Q-RAG pseudocode is presented in Algorithm 1.
4

Under review as a conference paper
Algorithm 1Q-RAG
1:Hyperparameters:
2:Environments countK, retrieval stepsT, temperatureα, TD parameterλ, EMAτ.
3:Initialize:
4:State embedderE s(s;θ 1)
5:Action embedderE a(ai, i;θ 2)with positioni
6:CriticQ θ(s, ai) =E s(s;θ 1)TEa(ai, i;θ 2)
7:Critic targetQ θ′(s, ai)
8:procedureCOMPUTETARGETS({s t, at, rt, vt}T+1
t=1)
9:Initializeλ-returnsG T=rT+γv T+1
10:fort=T−1downto1do
11:G t=rt+γ
(1−λ)v t+1+λG t+1
12:end for
13:return{G t}T
t=1
14:end procedure
15:Training (one update step)
16:forenvk∈1, . . . , Kin parallel do
17:s 1,A1=ResetQueryAndContext()
18:ComputeE a=E a(A;θ)andE′
a=E a(A;θ′)
19:forstept∈1, . . . , T+ 1do
20:a t∼softmax a∈A t1
αEs(s;θ)TEa
21:v t=αlogP
a∈Aexp1
αEs(s;θ′)TE′
a
22:r t=ComputeReward(s t, at)
23:s t+1=concatenate(s t, at)
24:A t+1=A t\ {at}
25:end for
26:B={s t, at, rt, vt}T+1
t=1
27:{Gk
t}T
t=1=ComputeTargets(B)
28:end for
29:∇L Q=1
TKPK
k=1PT
t=1∇θ(Qθ(sk
t, ak
t)−Gk
t)2
30:Updateθusing∇L Q
31:Update target parameters:θ′←τθ+ (1−τ)θ′
3.3 TEMPORAL REASONING FOR LONG-CONTEXT SEARCH
When dealing with narrative text, the information contained in a text chunkcmay be insufficient
to determine whetherchelps us answer the questionq. For example, we may need to know what
happened before some specific event. A standard retriever can find several relevant text chunks that
specify the character’s location, but choosing the correct one can be impossible without taking into
account temporal information. To address this, we propose arelative postional encodingof chunks
that explicitly encodes their position with respect to the facts already extracted into the state. At
stept, letS t={i 1<···< i k}be the (sorted) document indices of selected chunks andA tthe set
of available actions. The indices inS tpartition the document intok+1disjoint intervals: “before
the earliest selected fact”, “between consecutive selected facts”, and “after the latest selected fact.”
The relative positional mappingρ t:N→R+assigns to every original chunk index a real-valued
index that (i) identifies the interval it belongs to and (ii) preserves the relative order between chunks.
This mapping makes explicitbetween which extracted factsa chunk lies, while remaining invariant
to global shifts of absolute positions.
Formally, the interval boundaries are defined asb 0=1,b j=ijforj=1:k, andb k+1=m+1forC=
{c(i)}m
i=1. To compute relative indexρ t(i)for a chunkci, find the uniquejsuch thatb j≤i < b j+1
and set
ρt(i) =j δ+ℓi−b j
bj+1−bj,(6)
whereδ >0is the inter-interval step andℓ∈(0, δ)controls the within-interval resolution (e.g.,
δ=10,ℓ=9in our experiments). In the action embedder, the absolute position is replaced by the
5

Under review as a conference paper
relative one,
Ea 
ai, i;θ 2
⇒E a 
ai, ρt(i);θ 2
,(7)
which allows the Q-function to exploit the spatial relation of candidates to already retrieved evidence
while retaining local order within each interval. This design allows the retrieval agent to perform
strongly not only on fact-finding over disjoint document collections, but also on long-form narrative
tasks, enabling Q-RAG to compete with recurrent transformers (Bulatov et al., 2022; Rodkin et al.,
2024; Behrouz et al., 2025; 2024) and other long context approaches.
4 EXPERIMENTS
4.1 EXPERIMENTALSETUP
We evaluate our approach, Q-RAG, on tasks that cover commonsence reasoning, temporal reason-
ing, a bunch of needle in a haystack tasks and open-domain multi-hop question answering tasks on
context lengths that range from 4k tokens to 10M tokens per sample. For commonsence and tempo-
ral reasoning we useBabilongbenchmark (Kuratov et al., 2024), for Needle-in-a-Haystack we use
RULERbenchmark Hsieh et al. (2024). For open-domain multi-hop QA we useHotpotQAYang
et al. (2018),MusiqueTrivedi et al. (2022) andRULERbenchmarks. Babilong and RULER require
long contexts. Musique and HotpotQA use short contexts.
Baselines differ by task. Computing a uniform set of baselines across all datasets is difficult and
time-consuming. Many methods do not release code. Some methods were evaluated only on some
of these datasets. Even when the tasks match, the experimental settings often differ for the same
benchmarks. Some baselines provide code but require heavy resources (e.g., at least 8×A100 GPUs
Jin et al. (2025); Song et al. (2025); Huang et al. (2025)) to fine-tune, which are unavailable for us.
Therefore, we report three types of baselines, and we mark each baseline in tables accordingly:
•×Ablation: baselines that test the effectiveness of our proposed modifications.
•✓Reproduced: baselines that we finetuned and/or evaluated on our datasets using released
code or publicly available checkpoints.
•◦Reported: baselines whose scores we take directly from the original papers.
4.2 COMMONSENSE REASONING ON ULTRA-LONG CONTEXTS
On the BabiLong Kuratov et al. (2024) benchmark, we compared our method with the state-of-the-
art long-context processing approaches, including Titans Behrouz et al. (2024), Atlas Behrouz et al.
(2025), ARMT Rodkin et al. (2024), RMT Bulatov et al. (2022), as well as proprietary LLMs and
LLM-based agents. The results for most of these baselines were taken directly from the respective
original papers. As shown in Figure 3c, our approach achieves the highest average performance
(a)
 (b)
Figure 2: Comparison of answer accuracy on the long-context benchmark Babilong. Solid lines de-
note methods fine-tuned on the Babilong, while dashed lines denote zero-shot methods.a)Average
performance across tasks Q1–QA5.b)Performance on the hardest task, QA3, which requires the
longest reasoning chain and temporal awareness.
6

Under review as a conference paper
on BabiLong in ultra-long contexts ranging from 1 to 10 million tokens, demonstrating superior
generalization to long contexts compared to other specialized long-context methods.
In Figure 3a, we present separate results for the QA3 subtask, which is the hardest subtask in the Ba-
bilong benchmark, which specifically requires the multistep search of at least 3 different facts and
temporal reasoning. Experimental results show that the majority of models perform worst on the
QA3 subtask. As the results indicate, alternative long-context approaches show even greater perfor-
mance degradation on this task with increasing context length. In contrast, Q-RAG shows virtually
no degradation, with the largest performance gap over all baselines observed on this most chal-
lenging subtask. We additionally fine-tuned the Beam-Retriever baseline specifically on the QA3
subtasks, given its strong performance on open-domain QA datasets. However, this method failed
to solve the task. Note that some methods, such as Titans Behrouz et al. (2024) and Atlas Behrouz
et al. (2025), are absent from the Figure as they did not report detailed breakdowns by a subtask.
4.3 NEEDLE IN AHAYSTACK ANDLONGCONTEXTQA
While reasoning tasks are crucial for evaluating advanced retrieval systems, a substantial portion
of real-world applications reduces to Needle-in-a-Haystack (NIAH) problems, making it equally
important that models deliver consistently strong performance on these tasks.
RULER is a dataset that includes many long-context tasks. Most of these tasks follow the NIAH for-
mulation. The NIAH setup evaluates the ability to retrieve a specific “needle” from a long distracting
“haystack”.
For RULER benchmark we use Titans Behrouz et al. (2024), Atlas Behrouz et al. (2025), Mamba2
Waleffe et al. (2024), and LongRope2 Shang et al. (2025) as baselines. Titans, Atlas are recurrent
transformers. Mamba2 is a state space model (SSM) that combines transformer components with
Table 1: Results on the RULER benchmark, evaluating long-context retrieval performance across
various context lengths.S(Single-needle): Find one value for one key.MK(Multi-keys): Find one
value for one key among many.MV(Multi-values): Find all values for one key.MQ(Multi-query):
Answer multiple questions over the context.MH QA: open domain multi-hop question answering.
Length MethodsS MKMV MQ NIAH Avg. MH QA
1-st 2-nd 3-rd 1-st 2-nd 3-rd
4K◦Titans 98.4 99.8 89.4 n/a n/a n/a n/a n/a n/a n/a
◦Atlas 99.2 100 90.6 n/a n/a n/a n/a n/a n/a n/a
◦Mamba2-Hybrid 100 100 95.7 89.5 95.5 96 97.9 97.6 96.5 48.8
◦LongRoPe2-8B 100 100 99 100 100 100 99 99.7 99.7 60
✓Beam-Retriever 100 100 98 98 98 97 98 99 98.5 28.3
Q-RAG 100 100 100 100 100 100 100 100100 67
16K◦Titans 96.2 80.2 n/a n/a n/a n/a n/a n/a n/a n/a
◦Atlas 97 84 n/a n/a n/a n/a n/a n/a n/a n/a
◦Mamba2-Hybrid 100 100 81.5 92 92.2 83 89.8 90.2 91.1 44
◦LongRoPe2-8B 100 100 100 99 100 98 95 98.2 98.8 58
✓Beam-Retriever 100 100 97 96.5 96 95 80 98 95.3 28.3
Q-RAG 100 100 100 100 100 100 100 100100 67
32K◦Mamba2-Hybrid 100 100 96.7 84 76.5 81.5 84.3 80.9 88.0 38.5
◦LongRoPe2-8B 100 100 100 99 98 100 98 96.2 98.9 55
Q-RAG 100 100 100 100 100 100 100 100100 67
128K◦LongRoPe2-8B 100 100 99 96 91 94 96.5 97 96.7 50
Q-RAG 100 100 100 100 100 100 100 100100 62
1M Q-RAG 100 100 100 100 98.5 99.0 100 10099.7 57
7

Under review as a conference paper
SSM. LongRope2 is a method for extending the effective context window of LLMs. All methods
were fine-tuned either directly on RULER (Titans, Atlas, Mamba2) or on related synthetic NIAH-
style datasets (LongRope2). Q-RAG was also fine-tuned on the NIAH subtasks. For the Multi-hop
QA RULER subtask, Q-RAG was fine-tuned on HotpotQA and evaluated on the Multi-hop QA
subtask out-of-distribution.
The results are shown in Table 1. Q-RAG achieves near-perfect performance on all NIAH subtasks.
Q-RAG embedder was trained on 4K-length documents and generalizes to context lengths up to 1M
tokens without loss of accuracy. On the Multi-hop QA subtask, Q-RAG shows significantly better
results than all our baselines at all context lengths we consider. Some degradation with increasing
context length starts only from 128K.
4.4 OPEN-DOMAINQUESTIONANSWERING
For our experiments on the HotPotQA and Musique datasets, we compared our method against
several strong baselines. The first baseline is Beam Retriever, which enables multi-step retrieval
by training a model to score sequences of retrieved chunks. During evaluation, Beam-Retriever is
given the oracle number of supporting facts (i.e., the gold hop count) and always retrieves exactly
that many facts. Although this approach is slower than traditional retrieval methods and does not
scale well to longer contexts, it achieves state-of-the-art results on HotPotQA. Another baseline
we considered is SearchR1, a recent method from a family of approaches that train the LLM itself
to compose text queries for multi-step retrieval. Additionally, we evaluated the performance of
LLM-agent-based methods, including GraphReader and HippoRAG. Q-RAG and Beam-Retriever
were fine-tuned on HotPotQA and evaluated on Musique for out-of-distribution testing. Baseline
numbers were taken directly from the corresponding papers. Missing entries indicate metrics not
reported by the original authors.
The comparison results are presented in Table 2. Our method achieves fact retrieval accuracy on par
with Beam Retriever, surpasses all other baselines on HotPotQA, and matches the performance of
full-LLM-tuning Search-R1 while outperforming all alternatives on the out-of-distribution Musique
dataset, resulting in the best overall performance across benchmarks. For both methods involving
retrieval mechanism fine-tuning (Q-RAG and Beam Retriever), we used the QwQ-32B model to
produce the final answer.
Table 2: Comparison of methods on HotPotQA and Musique benchmarks. Bold text and underline
denote the best and second best scores respectively.
HotPotQA Musique (OOD) Avg
Methods Fact F1 Fact EM Ans F1 Ans EM Fact F1 Fact EM Ans F1 Ans EM Ans F1 Ans EM
Finetuned on HotPotQA
Plan Q-RAG 0.95 0.91 0.76 0.60 0.69 0.53 0.51 0.36 0.64 0.48
Q-RAG 0.93 0.89 0.76 0.59 0.71 0.55 0.520.37 0.64 0.48
✓Beam-Retriever0.97 0.94 0.77 0.61 0.61 0.36 0.40 0.27 0.59 0.44
✓Search-r1 0.81 0.66 0.65 0.52 0.71 0.550.510.39 0.58 0.46
◦RAG-RL 0.82 – 0.69 – 0.65 – 0.47 – 0.58 –
×Multi-step RAG w.o. FT 0.73 0.54 0.65 0.50 0.51 0.30 0.40 0.27 0.53 0.39
Zero Shot methods
✓GraphReader – – 0.46 0.24 – – 0.40 0.20 0.43 0.22
✓Single step RAG – – 0.53 0.39 – – 0.28 0.17 0.41 0.28
4.5 ABLATIONSTUDY
To assess the impact of the architectural choices in Q-RAG, an ablation study was conducted on the
Babylon-QA3 task. This benchmark was selected because it is among the most challenging long-
context tasks used in the experiments and it supports evaluation at arbitrary context lengths. The
following baselines were compared against Q-RAG:
8

Under review as a conference paper
0.00 0.01 0.02 0.03 0.04 0.05
T emperature parameter 
0.860.880.90.920.940.96Average return
QA2 QA3
(a)
0.4 0.5 0.6 0.7 0.8
TD parameter 
0.930.940.950.960.970.98Average return
QA2 QA3 (b)
 (c)
Figure 3: Ablation for (a) policy entropy coefficient (α) in soft Q function and (b) forλ-return
parameter. Inference runtime comparison (c), context length, tokens on x-axes.
•Multi-step RAG w.o. FT.This baseline reproduces the full Q-RAG retrieval pipeline and
uses the same state and action embedders, but relies on their original pretrained weights
without any reinforcement learning fine-tuning. This setting tests whether RL fine-tuning
of the embedders is beneficial for multi-step retrieval quality.
•Multi-step RAG w. SFT.This baseline applies supervised fine-tuning using ground-truth
support facts as supervision. The loss follows the objective used in BeamRetriever for
trajectory supervision, adapted to the multi-step retrieval setting. This setting isolates the
effect of RL by comparing it to supervised learning on the same supervision signal.
•Q-RAG w.o. target.This variant removes target networks from the PQN-based value
learning, following the original PQN recipe without target parameters. It measures the
contribution of target networks to stability and performance in the Q-RAG training loop.
•Q-RAG w.o. Soft-Q.This variant replaces the maximum-entropy (soft) value functions
with standard (non-entropy-regularized) Q-learning objectives. It evaluates the effect of
entropy regularization and the soft value formulation on retrieval performance.
All baselines were evaluated with three random seeds. Table 3 reports results at a 32k-token con-
text length on QA3. Figure 3 shows the sensitivity of Q-RAG to theλ-return parameter and the
temperatureα(the strength of entropy regularization) on QA2 and QA3.
Table 3: Ablation results on Babilong QA3. Table shows F1 score for support facts retrieval. All
values are averaged over 3 runs with different seeds.
Method 1K 4K 32K 128K 1M
Q-RAG 97.8±0.17 97.4±0.14 97.1±0.08 96.8±0.08 96.5±0.16
×Q-RAG w.o. Soft-Q 95.9±0.70 95.5±0.80 94.5±0.50 94.0±0.30 93.3±0.45
×Q-RAG w.o. Target 79.2±26.0 78.1±26.6 77.6±27.2 77.4±27.3 75.9±28.2
×Multi-Step RAG w. SFT 20.33±0.32 20.87±0.35 20.10±0.20 18.30±0.36—
×Multi-Step RAG w.o. FT 15.52±0.11 16.38±0.10 15.51±0.16 15.34±0.12—
5 CONCLUSION
This work introduced Q-RAG, a resource-efficient method for multi-step retrieval trained with re-
inforcement learning directly in the latent space of text-chunk embeddings. Across long-context
benchmarks (e.g.,Babilong,RULER) and open-domain QA datasets (e.g.,Musique,HotpotQA),
Q-RAG attains state-of-the-art or highly competitive results. Its advantage over baselines widens as
context length grows, and performance shows minimal degradation even at ultra-long scales.
A key practical benefit is compute efficiency: all training was performed on a single A100 GPU
with 80 GB memory, whereas recent RL-based multi-step retrievers such as Search-R1/R1-Searcher
typically report training on clusters of about eight A100 GPUs. By fine-tuning only the embedder
while keeping the LLM frozen, Q-RAG remains easy to pair with powerful pre-trained or proprietary
LLMs, enabling efficient training, flexible deployment, and strong retrieval over very long contexts.
9

Under review as a conference paper
Looking ahead, promising directions include using structured LLM feedback as a reward signal,
strengthening compositional and temporal reasoning directly in the embedding space, and exploring
tighter integration with generation while preserving the method’s efficiency and scalability.
REFERENCES
Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry Evseev, Andrey Kravchenko, Mikhail Burt-
sev, and Evgeny Burnaev. Arigraph: Learning knowledge graph world models with episodic
memory for llm agents.arXiv preprint arXiv:2407.04363, 2024.
Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time.arXiv
preprint arXiv:2501.00663, 2024.
Ali Behrouz, Zeman Li, Praneeth Kacham, Majid Daliri, Yuan Deng, Peilin Zhong, Meisam Raza-
viyayn, and Vahab Mirrokni. Atlas: Learning to optimally memorize the context at test time.
arXiv preprint arXiv:2505.23735, 2025.
M. Sh. Birman and M. Z. Solomyak. Spectral asymptotics of nonsmooth elliptic operators. I, II.
Trudy Moskovskogo Matematicheskogo Obshchestva, 27:3–52, 1972.
Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. Recurrent memory transformer.Advances in
Neural Information Processing Systems, 35:11079–11091, 2022.
Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z Pan,
Wen Zhang, Huajun Chen, Fan Yang, et al. Learning to reason with search for llms via reinforce-
ment learning.arXiv preprint arXiv:2503.19470, 2025.
Matteo Gallici, Mattie Fellows, Benjamin Ellis, Bartomeu Pou, Ivan Masmitja, Jakob Nicolaus
Foerster, and Mario Martin. Simplifying deep temporal difference learning. InThe Thirteenth
International Conference on Learning Representations.
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2024.
URLhttps://arxiv.org/abs/2312.00752.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning.arXiv preprint arXiv:2501.12948, 2025.
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy
maximum entropy deep reinforcement learning with a stochastic actor. InInternational confer-
ence on machine learning, pp. 1861–1870. Pmlr, 2018.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang
Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language
models?arXiv preprint arXiv:2404.06654, 2024.
Jerry Huang, Siddarth Madala, Risham Sidhu, Cheng Niu, Hao Peng, Julia Hockenmaier, and Tong
Zhang. Rag-rl: Advancing retrieval-augmented generation via rl and curriculum learning.arXiv
preprint arXiv:2503.12759, 2025.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobi-
ologically inspired long-term memory for large language models.Advances in Neural Information
Processing Systems, 37:59532–59569, 2024.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and
Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement
learning.arXiv preprint arXiv:2503.09516, 2025.
Yury Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and
Mikhail Burtsev. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack.
Advances in Neural Information Processing Systems, 37:106519–106554, 2024.
10

Under review as a conference paper
Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu,
Yangguang Li, Wanli Ouyang, et al. Graphreader: Building graph-based agent to enhance long-
context abilities of large language models. InFindings of the Association for Computational
Linguistics: EMNLP 2024, pp. 12758–12786, 2024.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models.arXiv preprint
arXiv:2501.05366, 2025.
Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang,
Chenchen Zhang, Ge Zhang, Jiebin Zhang, et al. A comprehensive survey on long context lan-
guage modeling.arXiv preprint arXiv:2503.17407, 2025.
Chuangtao Ma, Yongrui Chen, Tianxing Wu, Arijit Khan, and Haofen Wang. Large language mod-
els meet knowledge graphs for question answering: Synthesis and opportunities, 2025. URL
https://arxiv.org/abs/2505.20099.
V olodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Belle-
mare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level
control through deep reinforcement learning.nature, 518(7540):529–533, 2015.
Erich Novak and Henryk Wo ´zniakowski.Tractability of Multivariate Problems: Volume I: Linear
Information, volume 6 ofEMS Tracts in Mathematics. European Mathematical Society, Z ¨urich,
2008.
Alexander Novikov, Ng ˆan V ˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt
Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian,
et al. Alphaevolve: A coding agent for scientific and algorithmic discovery.arXiv preprint
arXiv:2506.13131, 2025.
Ivan Rodkin, Yuri Kuratov, Aydar Bulatov, and Mikhail Burtsev. Associative recurrent memory
transformer.arXiv preprint arXiv:2407.04841, 2024.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms.arXiv preprint arXiv:1707.06347, 2017.
Ning Shang, Li Lyna Zhang, Siyuan Wang, Gaokai Zhang, Gilsinia Lopez, Fan Yang, Weizhu
Chen, and Mao Yang. Longrope2: Near-lossless llm context window scaling.arXiv preprint
arXiv:2502.20082, 2025.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, et al. Deepseekmath: Pushing the limits of mathematical reasoning in
open language models.arXiv preprint arXiv:2402.03300, 2024.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. In
Proceedings of the 2024 Conference of the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 8364–8377,
2024.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented
generation: A survey on agentic rag, 2025. URLhttps://arxiv.org/abs/2501.09136.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang,
and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement
learning.arXiv preprint arXiv:2503.05592, 2025.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop
questions via single-hop question composition.Transactions of the Association for Computational
Linguistics, 10:539–554, 2022.
Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert
Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, et al. An empirical study of mamba-
based language models.arXiv preprint arXiv:2406.07887, 2024.
11

Under review as a conference paper
Chenghan Yang, Ruiyu Zhao, Yang Liu, and Ling Jiang. Survey of specialized large language model.
arXiv preprint arXiv:2508.19667, 2025.
Diji Yang, Jinmeng Rao, Kezhen Chen, Xiaoyuan Guo, Yawen Zhang, Jie Yang, and Yi Zhang.
Im-rag: Multi-round retrieval-augmented generation through learning inner monologues. InPro-
ceedings of the 47th International ACM SIGIR Conference on Research and Development in In-
formation Retrieval, pp. 730–740, 2024.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pp. 2369–2380, 2018.
Tan Yu, Anbang Xu, and Rama Akkiraju. In defense of rag in the era of long-context language
models.arXiv preprint arXiv:2409.01666, 2024.
Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Liu Yong, and Shen Huang. End-to-end beam
retrieval for multi-hop question answering. InProceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technolo-
gies (Volume 1: Long Papers), pp. 1718–1731, 2024.
Brian D Ziebart.Modeling purposeful adaptive behavior with the principle of maximum causal
entropy. Carnegie Mellon University, 2010.
12

Under review as a conference paper
A INNER PRODUCT APPROXIMATION FORQ-FUNCTION
The Universal Approximation Theorem (UAT) states that neural networks with a single hidden layer
can approximate any continuous function arbitrarily well under mild conditions. In this section, we
prove a variant of the UAT for functions decomposed as an inner product involving Rotary Position
Embedding (RoPE). Specifically, we show that any continuous q-functionQ(s, ai)defined on a
compact domain can be approximated by functions of the form:
F(s, ai) =⟨E s(s), E a(ai, i)⟩, E a(ai, i) =R pos(i)Ea(ai),(8)
whereE sandE aare continuous vector functions (e.g., neural networks) andR tis the RoPE matrix
of dimensionr(even) parameterized byt=pos(i):
Rt=r/2M
j=1
cos(θ jt)−sin(θ jt)
sin(θ jt) cos(θ jt)
,(9)
whereθ jare fixed frequencies. For notational simplicity in the following derivations, we introduce
the following conventions:
(x, y) := (s, a), t:=pos(i), h(x) :=E s(s), g(y) :=E a(ai).
For simplicity, we assume the domains ofx,yandtare continuous, corresponding to the embed-
dings of text tokens.
Theorem 1.LetX⊂Rdx,Y⊂Rdy, andT⊂Rbe compact sets, and define the compact domain
K=X×Y×T. LetC(K,R)be the space of continuous real-valued functions onKequipped with
the uniform norm. LetR tbe the RoPE matrix of dimensionr, defined as a block-diagonal rotation
matrix (9). Define the function class:
A={F(x, y, t) =⟨h(x), R tg(y)⟩ |h∈C(X,Rr), g∈C(Y,Rr)}.(10)
ThenAis dense inC(K,R). That is, for anyf∈C(K,R)andϵ >0, there exist continuous
functionsh:X→Rdandg:Y→Rdsuch that:
sup
(x,y,t)∈K|f(x, y, t)− ⟨h(x), R tg(y)⟩|< ϵ.(11)
Proof.We prove the result via the Stone-Weierstrass theorem, which states that if a subalgebra
A ⊂C(K,R)contains the constant functions and separates points, thenAis dense inC(K,R).
Thus, we show thatAsatisfies these requirements.
Ais a subalgebra.We prove closure under addition, scalar multiplication, and multiplication of
two arbitrary elements.
Scalar multiplication: LetF(x, y, t) =⟨h(x), R tg(y)⟩ ∈ Aandc∈R. Defineh′(x) =ch(x).
ThencF(x, y, t) =⟨h′(x), R tg(y)⟩ ∈ A.
Addition: LetF 1(x, y, t) =⟨h 1(x), R tg1(y)⟩andF 2(x, y, t) =⟨h 2(x), R tg2(y)⟩. Defineh(x) =
[h1(x);h 2(x)]∈R2dandg(y) = [g 1(y);g 2(y)]∈R2d, and let eRtbe a block-diagonal extension of
Rt. Then
⟨h(x),eRtg(y)⟩=⟨h 1(x), R tg1(y)⟩+⟨h 2(x), R tg2(y)⟩=F 1(x, y, t) +F 2(x, y, t)∈ A.
Multiplication: LetF 1andF 2as above. Note that:
F1(x, y, t)F 2(x, y, t) =⟨h 1(x)⊗h 2(x),(R tg1(y))⊗(R tg2(y))⟩.
Since(R tg1(y))⊗(R tg2(y)) = (R t⊗Rt)(g1(y)⊗g 2(y)), andR t⊗Rtis a block-diagonal rotation
matrix with anglesθ j+θk(a RoPE matrix of dimensiond2), defineh(x) =h 1(x)⊗h 2(x)∈Rd2,
g(y) =g 1(y)⊗g 2(y)∈Rd2, and let eRtbe the RoPE matrix with frequencies{θ j+θk}. Then:
F1(x, y, t)F 2(x, y, t) =⟨h(x), eRtg(y)⟩ ∈ A.
Thus,Ais a subalgebra.
13

Under review as a conference paper
Acontains the constant functions.Show the constant function1is inA. Augment the dimen-
sion: letd′=d+1, and defineh(x) = (1,0, . . . ,0)T∈Rd′,g(y) = (1,0, . . . ,0)T∈Rd′. Define a
modified RoPE matrixR′
tthat acts as the identity on the first coordinate and asR ton the remaining
dcoordinates. Then
⟨h(x), R′
tg(y)⟩= 1.
Aseparates points.Let(x 1, y1, t1)̸= (x 2, y2, t2)∈K. ConstructF∈ Asuch that
F(x 1, y1, t1)̸=F(x 2, y2, t2).
Case 1:x 1̸=x 2ory1̸=y2. Chooseg(y) =v(a constant non-zero vector) and lethbe continuous
withh(x 1)̸=h(x 2). ThenF(x, y, t) =⟨h(x), R tv⟩. SinceR tvtraces a circle (forvwith at least
two non-zero components), for genericv,R t1vandR t2vare not orthogonal toh(x 1)−h(x 2), so
F(x 1, y1, t1)̸=F(x 2, y2, t2). The case wheny 1̸=y2is identical to the 1st case.
Case 2:t 1̸=t2. Chooseh(x) =wandg(y) =v. ThenF(x, y, t) =⟨w, R tv⟩. Sincet7→R tv
is injective (forv̸= 0and non-zero frequencies),R t1v̸=R t2v. Choosewnot orthogonal to
Rt1v−R t2v, soF(x 1, y1, t1)̸=F(x 2, y2, t2).
Thus, by the Stone-Weierstrass theorem,Ais dense inC(K,R).
Theorem 1 establishes that our architecture is capable of approximating any continuous function
arbitrarily well. However, it does not specify how complex the network needs to be to achieve a
given accuracy. The following quantitative result addresses this by providing an explicit convergence
rate dependent on the smoothness of the target function.
Theorem 2(Convergence Rate).Let a target functionf∈C(K,R)belong to the Sobolev space
Hs,∞(K), i.e.,fhas bounded derivatives up to orders. Then, for any integerr >0, there exist
feature mapsh:X→Rrandg:Y→Rrsuch that
sup
(x,y,t)∈K|f(x, y, t)− ⟨h(x), R tg(y)⟩| ≤C·r−s/(2d x+2dy+2),
whereCdepends on∥f∥ Hs,∞, the diameters ofX, Y, T, and the frequenciesθ j.
Proof.SinceR tincorporates trigonometric functions, we first expandfin a Fourier series int:
f(x, y, t) =X
k∈Zak(x, y)eikt,
where the Fourier coefficients satisfy:
|ak(x, y)| ≤∥f∥Hs,∞
(1 +|k|)s.
Truncate to|k| ≤N, with error:
f(x, y, t)−X
|k|≤Nak(x, y)eikt≤C 1N−s.
Make an approximation ofa k(x, y)by inner products:
ak(x, y)≈ ⟨h k(x), g k(y)⟩.
Using the result on inner product approximation by Lemma 1, for eachkthere exist functionsh k, gk
such that:
|ak(x, y)− ⟨h k(x), g k(y)⟩| ≤C 2r−s/(d x+dy)
k.
Chooser k∼r1/(2N+1)so that total dimensionP
|k|≤N rk∼(2N+1)r1/(2N+1). Define functions
h, gand the RoPE matrix to have frequenciesθ j=jforj= 1, . . . , Nsuch that
⟨h(x), R tg(y)⟩=X
|k|≤N⟨hk(x), R(k)
tgk(y)⟩,
14

Under review as a conference paper
whereR(k)
tis the block corresponding to frequencyk. By design,R(k)
tgk(y)produces terms like
eiktgk(y), so:
⟨h(x), R tg(y)⟩=X
|k|≤N⟨hk(x), g k(y)⟩eikt.
The error is bounded by:
ϵ=|f(x, y, t)− ⟨h(x), R tg(y)⟩| ≤C 1N−s+C 2X
|k|≤Nr−s/(d x+dy)
k.
Withr k∼r1/(2N+1)andN∼logr, we obtain:
ϵ≤C·r−s/(2d x+2dy+2).
Lemma 1.LetΩ x⊂RdxandΩ y⊂Rdybe compact domains with Lipschitz boundaries. Consider
a symmetric, positive-definite kernela: Ω x×Ω y→Rsuch thata∈Hs(Ωx×Ω y)for some
smoothness parameters >(d x+dy)/2. Letd=d x+dybe the total dimension.
Then, for any integerr >0, there exist feature mapsh: Ω x→Rrandg: Ω y→Rrsuch that the
following uniform approximation bound holds:
sup
x∈Ωx, y∈Ω y|a(x, y)− ⟨h(x), g(y)⟩ Rr| ≤C·r−s/d· ∥a∥ Hs.
Here,C >0is a constant depending ons,d x,dy, and the domainsΩ x,Ωy, but independent ofr
anda.
Proof.Consider a countable orthonormal basis of eigenfunctions{ϕ i} ⊂L2(Ωx)and{ψ i} ⊂
L2(Ωy)with corresponding non-negative eigenvalues{λ i}(ordered non-increasingly,λ 1≥λ 2≥
... >0) such that:
a(x, y) =∞X
i=1λiϕi(x)ψ i(y),
where the convergence is absolute and uniform onΩ x×Ωy. The optimal rank-rapproximation is
given by truncating to the firstrterms:
ar(x, y) :=rX
i=1λiϕi(x)ψ i(y).
Denote the approximation error as the tail of the infinite series:
er(x, y) =a(x, y)−a r(x, y) =∞X
i=r+1λiϕi(x)ψ i(y),
such that
|er(x, y)| ≤∞X
i=r+1λi|ϕi(x)||ψ i(y)|.
The key is that the smoothness ofagoverns the decay rate of the eigenvaluesλ i. From Theorem 3.1
Birman & Solomyak (1972) follows that for an operator with kernel inHs(Ωx×Ωy), the eigenvalues
satisfy:
λi≤C 1·i−(1+2s/d)· ∥a∥ Hs,
whereC 1depends on the domain and dimension.
The conditions > d/2implies thatHsis continuously embedded in the space of continuous func-
tions. Furthermore, one can show the eigenfunctions are uniformly bounded:
∥ϕi∥L∞(Ωx)≤C 2,∥ψ i∥L∞(Ωy)≤C 2,
whereC 2is a constant independent ofi.
15

Under review as a conference paper
Combining the eigenvalue decay and eigenfunction bounds gives a pointwise error estimate:
|er(x, y)| ≤C2
2∞X
i=r+1λi≤C 1C2
2∥a∥Hs∞X
i=r+1i−(1+2s/d).
The tail of the series can be bounded by an integral:
∞X
i=r+1i−(1+α)≤Z∞
rt−(1+α)dt=1
αr−α,whereα=2s
d.
Substitutingα= 2s/dyields:
|er(x, y)| ≤C 1C2
2∥a∥Hs·d
2s·r−2s/d.
A more refined analysis (Novak & Wo ´zniakowski (2008)) shows the supremum norm error decays
asO(r−s/d). Thus, we obtain:
sup
x,y|er(x, y)| ≤C·r−s/d· ∥a∥ Hs.
B PLANNING FORMULTI-STEPRETRIEVAL
Wecanapplyplanningat the multi-step retrieval stage, formulating source selection as a search over
the space of action trajectories; see § 4.4 for an application. In the spirit ofBeam-Retriever, we can
run beam search where candidates are ranked by the learned action-valueQ θ(s, a). However, our
planning is computationally cheaper becauseQ θis computed as adot productof state and action
embeddings,Q θ(s, a) =⟨E s(s), E a(a)⟩,so no new transformer forward passes are required for
each candidate chunk, whereasBeam-Retrieverrelies on a transformer reranker over trajectories,
incurring fresh forward passes at every expansion. Details of the embedding-based scoring are
provided in § 3.2. At inference, we performbeam search overQanddeterministicallyexpand the
top-kactions byQ θ.
C TRAINING DETAILS
We trained the model with AdamW (learning rate1.5×10−5,β1=0.9,β 2=0.98,ϵ=10−6, weight
decay5×10−4). The learning rate followed alinearschedule: we used a warm-up of1,000steps,
then linearly decayed the rate to10%of its initial value over the remaining training steps. We applied
gradient clipping with a maximumℓ 2norm of2.0and used gradient accumulation for8steps. The
base mini-batch size was12; with accumulation this yields an effective batch size of12×8 = 96
per update (scaled by the number of devices if using distributed training).
In the objective and algorithmic components we setγ=0.99,α=0.05,λ=0.5, andτ=0.02. Action
representations were capped at a maximum length of220tokens.
Models per benchmark.For open-domain QA benchmarks (HotPotQA,Musique), we
trained anmultilingual-e5-largeencoder. ForRulerandBabilong, we trained
facebook/contriever. The end-to-end training of a single model did not exceed 12 hours
on a singleA100-80GBGPU.
16