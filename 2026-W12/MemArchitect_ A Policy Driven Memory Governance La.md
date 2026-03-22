# MemArchitect: A Policy Driven Memory Governance Layer

**Authors**: Lingavasan Suresh Kumar, Yang Ba, Rong Pan

**Published**: 2026-03-18 22:37:05

**PDF URL**: [https://arxiv.org/pdf/2603.18330v1](https://arxiv.org/pdf/2603.18330v1)

## Abstract
Persistent Large Language Model (LLM) agents expose a critical governance gap in memory management. Standard Retrieval-Augmented Generation (RAG) frameworks treat memory as passive storage, lacking mechanisms to resolve contradictions, enforce privacy, or prevent outdated information ("zombie memories") from contaminating the context window.
  We introduce MemArchitect, a governance layer that decouples memory lifecycle management from model weights. MemArchitect enforces explicit, rule-based policies, including memory decay, conflict resolution, and privacy controls.
  We demonstrate that governed memory consistently outperforms unmanaged memory in agentic settings, highlighting the necessity of structured memory governance for reliable and safe autonomous systems.

## Full Text


<!-- PDF content starts -->

MemArchitect: A Policy Driven Memory Governance Layer
Lingavasan Suresh Kumar*
Arizona State UniversityYang Ba†
Arizona State UniversityRong Pan‡
Arizona State University
Abstract
As Large Language Models (LLMs) transition into persistent agents, they face a critical “Governance Gap.”
Standard Retrieval Augmented Generation (RAG) treats memory as a passive storage bucket, lacking the controls
necessary to manage contradictions, enforce privacy, or prevent “zombie memories” (outdated facts) from polluting
the context window. We introduceMemArchitect, a distinct governance layer that decouples memory lifecycle
management from model weights. By enforcing explicit, rule based policies ranging from memory decay to user
privacy enforcement,MemArchitectaddresses these challenges. We demonstrate that governed memory outperforms
raw memory, highlighting the necessity of governance in autonomous agentic environments.
1 Introduction
As Large Language Models (LLMs) evolve from stateless chatbots to autonomous agents, they face a critical failure
mode:Memory Hallucination(Iyidogan and Ozkes, 2025). Unlike standard hallucinations, these occur when an agent
retrieves conflicting or outdated facts from its own history (e.g., retrieving “User is single” from 2024 alongside “User
is married” from 2025). Simply expanding context windows exacerbatescontext pollution, rendering agents unreliable
for long-horizon tasks with frequent state changes.
The agentic AI field currently addresses these challenges through three distinct architectural paradigms: operating
systems, efficient streaming, and structured retention. First, frameworks such as MemGPT (Packer et al., 2024) and
MemOS (Li et al., 2025) treat the LLM as an operating system, using virtual memory paging to manage context, while
recent innovations such as Titans (Behrouz et al., 2024) propose “neural memory” modules that allow models to learn
context at test time. Second, to sustain these lifelong interactions, approaches such as SimpleMem (Liu et al., 2025)
and LLMLingua (Jiang et al., 2023) focus on semantic compression, building on the foundational Attention Sink
(Xiao et al., 2023) mechanisms that enable stable processing of infinite token streams. Finally, the focus on semantic
organization is evolving rapidly; while Generative Agents (Park et al., 2023) and MemoryBank (Zhong et al., 2023)
introduced reflection and decay to mimic human retention, newer approaches such as A-MEM (Xu et al., 2025) and
the Random Tree Model of Meaningful Memory (Zhong et al., 2025) advocate using the hierarchical Zettelkasten-
style evolutionary structures to replace flat logs. Commercial-grade implementations, such as Mem0 (Chhikara et al.,
2025), further extend this approach by employing graph-enhanced retrieval to persist user preferences across sessions.
However, these systems suffer from a distinct “Governance Gap.” They treat memory as a passive, append-only
log, lacking mechanisms to resolve state contradictions, enforce “Right-to-be-Forgotten” compliance, or actively prune
obsolete data based on utility. Therefore, we propose MemArchitect, a governance middleware that shifts the paradigm
from passive memory retrieval to active adjudication. Unlike standard RAG systems that treat memory as a static
storage bucket, MemArchitect interposes a “Triage & Bid” economy between the user and the agent, where every
piece of information must actively compete for the context window. This policy engine enforces a unified governance
protocol in four critical domains:Lifecycle & Hygiene, Consistency & Truth, Provenance & Trust, and Efficiency &
Safety,ensuring that the agent’s context is not only retrieved but proactively managed for decay, factuality, lineage,
and compliance standards.
*Email:lsuresh4@asu.edu
†Email:Yangba@asu.edu
‡Email:Rong.Pan@asu.edu
1arXiv:2603.18330v1  [cs.AI]  18 Mar 2026

2 The Governance Engine: Policy Layer
2.1 The Policy Suite
To transition memory from a passive artifact to an active cognitive resource, we establish a unified governance pro-
tocol anchored in four foundational pillars:Lifecycle,Consistency,Retrieval, andEfficiency. We argue that for an
autonomous agent to function indefinitely without succumbing to “context collapse,” it must systematically address
the entropy of time, the drift of truth, the complexity of access, and the scarcity of computing resource. By governing
these four dimensions, we ensure that the memory store operates not merely as a database, but as a highly alerted
environment where information actively competes for relevance. (For the complete mathematical formulations and
implementation details of all policies, please see Appendix A.)
1.Lifecycle & Hygiene (The “Forgetting” Engine):
•FSRS Decay:We replace exponential decay with theFree Spaced Repetition Scheduler(FSRS v4)Ye et al.
(2022). Memory decay mimics a biological forgetting curve where retrievability (R) depends on stability
(S) and time (t).
•Entropy-Triggered Consolidation:A background process monitors information density. High redundancy
(Entropy Ratio<0.4) triggers a cleanup cycle where fading episodic memories (0.3≤R≤0.7) are
compressed into stable semantic facts, and “forgettable” noise (R <0.3) is actively pruned.
2.Consistency & Truth (The “Utility” Engine):
•The Reflection Loop (Kalman Filter):We treat the retrieval utility as an alert signal. A Kalman Filter
tracks the “Trust Score” (U) of every memory. Useful memories gain inertia (U↑), while hallucinations
are mathematically down-weighted.
•Relevance Discriminator (Veto Gate):A secondary Cross-Encoder acts as a “Veto” gate. Even if a memory
passes the vector search, it is discarded if it does not meet a strict entailment threshold (i.e.,<0.1). This
prevents highly similar but irrelevant noise from polluting the context.
3.Adaptive Retrieval (The “Auction” Engine):
•Multi-Hop Decomposition:Complex queries are broken into sub-questions (e.g., “Where is Tokyo?” +
“Capital of that country?”) to retrieve disjointed facts.
•Adaptive Scoring:Queries are classified (Fact, Temporal, Reasoning) for dynamically tuning retrieval
weights, where old memories are penalized for immediate reasoning tasks (λ= 1.0), and facts are treated
as timeless (λ= 0).
•Hebbian Graph Expansion:To bridge non-semantic gaps, we implement a Hebbian Graph Expansion
policy, inspired by the foundational principle of associative learning (Hebb, 1949), where “memories that
fire together are wired together.” That is, if Memory A and B are frequently used together (e.g.,P(B|A)>
0.7), retrieving A automatically pulls B into context.
4.Efficiency (The “Budget” Engine):
•Adaptive Token Budgeting:The system dynamically allocates the context window. High-confidence re-
trieval sets a “Reasoning Reserve”(30%) for the LLM to think, while low-confidence retrieval maximizes
the “Recall Reserve” (10%) to seek for facts.
2.2 The Governance Cycle & Dependencies
These policies do not operate in isolation; they form a tightly coupled functional loop, as illustrated in Figure 1. The
architecture splits the agent’s operation into three distinct execution paths:
1.Read Path (The Retrieval Auction):Upon receiving a query, theClassifierfirst determines the active scoring
logic (λ, β) based on intent (e.g., Fact vs. Reasoning). SimultaneousDecompositionbreaks complex queries
into atomic sub-questions. Candidates then enter theAuction, where they are ranked by their computed scores,
rigorously filtered by theDiscriminator(Cross-Encoder) for semantic entailment, and expanded viaHebbian
Linksto retrieve associated context before filling theAdaptive Token Budget.
2

2.Reflect Path (The Feedback Loop):Post-generation, the system closes the learning loop.Usage Detection
algorithms analyze the generated output to identify which specific memories contributed to the answer. This
binary feedback signal triggers immediate updates to theKalman Utility(U) state (reinforcing trust) and the
FSRS Stability(S) metric (to reset the forgetting curve).
3.Background Path (The Hygiene Engine):During system idle time, theConsolidationpolicy scans the vector
store. Driven by the updated Stability scores, it enforces the lifecycle rules: low-retrievability noise is physically
deleted, while fading episodic memories are compressed into stable semantic facts, ensuring the long-term health
of the memory graph.
Figure 1:The MemArchitect Governance Workflow.The system operates as a continuous closed loop across three
paths:(1) Read Path (Top):Incoming queries are classified by intent to customize the search. Memories are retrieved,
filtered for safety, and ranked by value to fit the limited context budget.(2) Reflect Path (Right):A feedback loop
detects which memories were actually used to generate the answer, strengthening useful information while penalizing
distractions.(3) Background Path (Bottom):During idle cycles, a maintenance process automatically cleans the
memory store by deleting forgotten noise and compressing fading details into permanent facts.
3 Experiment & Analysis
3.1 Experiment Setup and Baseline
MemOS (Memory Operating System) (Li et al., 2025) and SimpleMem (Liu et al., 2025) capture two complementary
extremes, respectively; i.e., system-level governance and algorithmic efficiency. They form natural baselines for
evaluating recall quality and token efficiency on long-horizon benchmarks such as LoCoMo(Maharana et al. (2024)).
Qwen-3B and Llama-3.1-8B are used in our experiments and evaluated in accuracy.
3

MemOS reframes LLM memory as a system-level resource (“MemCube”) and manages its lifecycle through an
OS-style module, called MemScheduler. It represents a maximalist, full-stack approach. Like MemOS, we treat
memory as a governed resource. However, MemArchitect operates strictly as an external governance layer without
modifying model parameters, making it lightweight and model-agnostic. SimpleMem targets token inefficiency using a
three-stage pipeline – compression, consolidation, and adaptive retrieval. It achieves strong efficiency by maximizing
information density per token. In comparison, MemArchitect shares the efficiency goal but differs in mechanism.
While SimpleMem relies on compression, we rely onpolicy-driven governanceincluding FSRS decay, utility tracking,
and auctions.
3.2 Quantitative Results
Table 1: Comparison of Memory Systems on LoCoMo-10 Benchmark
Model Task Baseline (Acc) MemArchitect (Acc)
Qwen-3BSingle-Hop 79.4% (SimpleMem) 95.0% ( +15.6%)
Multi-Hop 66.7% (SimpleMem) 95.8% ( +29.1%)
Temporal Reasoning 56.1% (SimpleMem) 95.3% ( +39.2%)
Open Domain 67.1% (SimpleMem) 93.1% ( +26.0%)
Llama-3.1-8BSingle-Hop 74.0% (MemOS) 53.2% ( -20.8%)
Multi-Hop 36.0% (MemOS) 40.1% ( +4.1%)
Temporal Reasoning 72.0% (MemOS) 27.1% ( -44.9%)
Open Domain 45.0% (MemOS) 56.3% ( +11.3%)
Average62.04%69.49% ( +7.45%)
Table 1 demonstrates the superior performance of MemArchitect in most tasks and shows an aggregated improve-
ment of+7.45%accuracy across all tasks. When compared against the algorithmic baseline (SimpleMem), MemAr-
chitect achieves a dominant sweep, confirming that policy-driven adjudication is superior to semantic compression for
preserving context fidelity. SimpleMem’s compression indiscriminately discards temporal markers, leading to poor
performance in Temporal Reasoning, whereas our FSRS policy actively prioritizes them (95.3% accuracy, a+39.2%
gain).
However, the lower half of the Table reveals a critical trade-off against the MemOS, which treats memory as an
infinite OS resource, dominating in raw recall tasks (Single-Hop and Temporal) because it retains all logs. Our active
decay policy, by contrast, aggressively prunes “one-shot” facts (such as specific timestamps) that lack repetition,
resulting in a -44.9% drop in Temporal recall. Yet, MemArchitect outperformed MemOS inOpen Domain (+11.3%)
andMulti-Hop (+4.1%)tasks. This indicates that while active governance sacrifices eidetic recall, it significantly
enhances abstract synthesis and long-horizon coherence. It shifts the failure mode from dangerous “Context Pollution”
to tunable “Over-Pruning”, which is a manageable parameter rather than a structural flaw.
4 Conclusion and Future Work
We introducedMemArchitect, a middleware layer transforming memory from passive storage to active adjudica-
tion. By enforcing policies for lifecycle, consistency, retrieval, and efficiency, we address the critical “Governance
Gap.” Our prototype demonstrates a strong aggregate improvement (+7.45%) and outperforms SimpleMem (+27.5%
in subgroup analysis), proving that adjudication preserves semantic integrity better than compression. While aggres-
sive decay currently limits raw recall compared to the maximalist baseline, MemOS, these results validate our thesis:
Governance shifts failure modes from unchecked hallucination to managed pruning.
4

To advance MemArchitect from a prototype into a trustworthy and robust policy-driven architecture, we will
continue to evaluate the system on established benchmarks such asLongMemEval(Wu et al., 2024), which challenges
model capacity with significantly more question-answer turns than LoCoMo. In parallel, we aim to validate the
prototype on persona-centric benchmarks likePreFEval(Zhao et al., 2025) andPersonaMem(Jiang et al., 2025) to
ensure stable, long-horizon identity preservation. Furthermore, we will fine-tune the architecture by calibrating FSRS
for low-frequency fact retention, implementing the remaining governance policies, and expanding our evaluation to
include comparisons against other state-of-the-art and learning-based approaches for long-term memory management
in agentic environments.
References
Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. 12 2024. URLhttp:
//arxiv.org/abs/2501.00663.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready
ai agents with scalable long-term memory. 4 2025. URLhttp://arxiv.org/abs/2504.19413.
Donald O Hebb.The Organization of Behavior: A Neuropsychological Theory. Wiley, New York, 1949.
Engin Iyidogan and Ali I Ozkes. Agentic ai and hallucinations.Economics Letters, 255:112520, 2025. ISSN
0165-1765. doi: https://doi.org/10.1016/j.econlet.2025.112520. URLhttps://www.sciencedirect.com/
science/article/pii/S016517652500357X.
Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J Taylor, and
Dan Roth. Know me, respond to me: Benchmarking llms for dynamic user profiling and personalized responses at
scale.arXiv preprint arXiv:2504.14225, 2025.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for
accelerated inference of large language models. 12 2023. URLhttp://arxiv.org/abs/2310.05736.
Zhiyu Li, Shichao Song, Chenyang Xi, Hanyu Wang, Chen Tang, Simin Niu, Ding Chen, Jiawei Yang, Chunyu Li,
Qingchen Yu, Jihao Zhao, Yezhaohui Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai
Chen, Kehang Li, Zhen Tao, Huayi Lai, Hao Wu, Bo Tang, Zhenren Wang, Zhaoxin Fan, Ningyu Zhang, Linfeng
Zhang, Junchi Yan, Mingchuan Yang, Tong Xu, Wei Xu, Huajun Chen, Haofen Wang, Hongkang Yang, Wentao
Zhang, Zhi-Qin John Xu, Siheng Chen, and Feiyu Xiong. Memos: A memory os for ai system. 8 2025. URL
http://arxiv.org/abs/2507.03724.
Jiaqi Liu, Yaofeng Su, Peng Xia, Yiyang Zhou, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao.
Simplemem: Efficient lifelong memory for llm agents.arXiv preprint arXiv:2601.02553, 2025. URLhttps:
//github.com/aiming-lab/SimpleMem.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating
very long-term conversational memory of llm agents, 2024. URLhttps://arxiv.org/abs/2402.17753.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Memgpt:
Towards llms as operating systems. 2 2024. URLhttp://arxiv.org/abs/2310.08560.
Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein.
Generative agents: Interactive simulacra of human behavior. InUIST 2023 - Proceedings of the 36th Annual ACM
Symposium on User Interface Software and Technology. Association for Computing Machinery, Inc, 10 2023. ISBN
9798400701320. doi: 10.1145/3586183.3606763.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking
chat assistants on long-term interactive memory. 2024. URLhttps://arxiv.org/abs/2410.10813.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with
attention sinks.arXiv, 2023.
5

Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zujie Liang, and Yongfeng Zhang. A-mem: Agentic memory for llm
agents. 7 2025. URLhttp://arxiv.org/abs/2502.12110.
Junyao Ye, Jingyong Su, and Yilong Cao. A stochastic shortest path algorithm for optimizing spaced repetition
scheduling. InProceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining,
pages 4381–4390, 2022. doi: 10.1145/3534678.3539081.
Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, and Kaixiang Lin. Do llms recognize your preferences?
evaluating personalized preference following in llms, 2025. URLhttps://arxiv.org/abs/2502.09597.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models
with long-term memory. 5 2023. URLhttp://arxiv.org/abs/2305.10250.
Weishun Zhong, Tankut Can, Antonis Georgiou, Ilya Shnayderman, Mikhail Katkov, and Misha Tsodyks. Random
tree model of meaningful memory.Physical review letters, 134:237402, 6 2025. ISSN 10797114. doi: 10.1103/
g1cz-wk1l.
A GOVERNANCE LAYER: THE COMPLETE POLICY LAYER
This appendix provides the complete technical specifications for the MemArchitect policy engine. We categorize poli-
cies into four functional domains: Lifecycle, Consistency, Retrieval, and Safety. Each entry details the mathematical
formulation, critical constants derived from our implementation, and the research rationale.
A.1 Domain A: Lifecycle & Hygiene (The “Forgetting” Engine)
Goal:To actively manage memory retention based on biological decay and information density.
Policy 1: FSRS Retrievability (Active)We implement the Free Spaced Repetition Scheduler (FSRS v4) to model
the probability of recall (R) over time ((Ye et al., 2022)).
R(t) =
1 +19
9·t
S−1
(1)
•Constant The19
9(≈2.11):Empirically calibrated on millions of flashcard reviews. It ensures that when elapsed
time (t) equals stability (S), retrievability drops to exactly32%(R≈0.32), which is identified as the optimal
forgetting index for efficient relearning.
Policy 2: Stability Update (Active)Successful retrieval strengthens a memory’s stability (S), which determines
how long it will persist.
Snew=Sold×
1 + 0.5×(11−D)×(e1.5(1−R)−1)
(2)
•Growth Factor (0.5):Matches FSRS ‘w8‘ parameter for balanced growth, preventing memories from becoming
“immortal” too quickly.
•Desirable Difficulty (e1.5(1−R)−1):The exponent1.5creates a non-linear bonus for “close calls.” If a memory
is retrieved whenR≈0.1(almost forgotten), it receives a massive2.86×stability boost compared to easy
recalls.
Policy 3: Entropy-Triggered Consolidation (Active)A background process monitors the information density of
the episodic log to trigger compression cycles.
Trigger⇐⇒Size(Gzip(Content))
Size(Content)<0.4(3)
6

•Threshold (0.4):Text streams compressing to<40%of original size indicate high redundancy (>60%repe-
tition).
•Action:
– Delete (R <0.3):Memories with<30%retrievability are effectively forgotten and pruned.
– Consolidate (0.3≤R≤0.7):Fading memories are synthesized into semantic facts.
– Keep (R >0.7):Active memories are preserved.
A.2 Domain B: Consistency & Truth (The “Utility” Engine)
Goal:To maximize factuality and reduce hallucinations through active monitoring.
Policy 4: Kalman Utility Filter (Active)We estimate the “True Utility” (U) of a memory source by treating usage
feedback as a noisy signal.
Kk=Pk−1
Pk−1+Rnoise;U k=Uk−1+K k·(zk−Uk−1)(4)
•Process Noise (Q= 0.05):Assumes memory utility changes slowly over time.
•Measurement Noise (R= 0.1):Reflects that our Usage Detection heuristic (word overlap) has moderate
uncertainty.
•Kalman Gain (K≈0.6):The system weighs new observations at60%and historical inertia at40%.
Policy 5: Relevance Discriminator (Active)A Cross-Encoder model acts as a “Veto Gate” after vector retrieval to
filter high-similarity but irrelevant noise.
Entailment(Q, M)<0.1 =⇒Drop (5)
•Threshold (0.1):Acts as a safety floor to discard candidates that are semantically disjoint, even if they passed
the vector search.
Policy 6: Conflict Resolution (Planned)Future work involves an NLI-based arbitrator. WhenM newconflicts with
Mold:
Winner= max(SourceAuth(M)×Recency(M))(6)
This policy will resolve contradictions (e.g., “User lives in NYC” vs “User lives in London”) to prevent “schizophrenic”
agent responses.
A.3 Domain C: Adaptive Retrieval (The “Auction” Engine)
Goal:To dynamically value information based on user intent and task complexity.
Policy 7: Adaptive Scoring (Active)The system assigns a Retrieval Score based on similarity (S), retrievability
(R), and utility (U).
Score=Sim×Rλ×(1 +β·U)(7)
•Decay Penalty (λ):
–λ= 0(Fact): Facts are timeless; age is ignored.
–λ= 1.0(Reasoning): Immediate context is prioritized; old memories are penalized.
•Utility Boost (β):
–β= 1.5(Multi-hop): Trust proven memories (U) for complex reasoning chains.
7

Policy 8: Hebbian Graph Expansion (Active)We track co-occurrence to solve the “missing link” problem. If
memoryAis retrieved, memoryBis also retrieved if:
P(B|A) =Count(A∩B)
Count(A)>0.7(8)
•Threshold (0.7):Requires a70%co-occurrence rate to establish a strong associative link, filtering out random
noise.
Policy 9: Adaptive Token Budgeting (Active)Token allocation varies with the confidence of the retrieval set.
GenReserve=(
2048if AvgScore(Top5)>0.4 (Reasoning Mode)
300otherwise(Recall Mode)(9)
•Reasoning Mode:High-quality retrieval allows the LLM to utilize a large context for analysis.
•Recall Mode:Low-quality retrieval forces a maximized context window to find needle-in-haystack facts.
A.4 Domain D: Governance & Safety (The “Compliance” Engine)
Goal:To enforce safety standards and data privacy.
Policy 10: Toxic Memory Filter (Planned)A write-path guardrail scans all incoming user text. If “Prompt In-
jection” or toxic content is detected, the system refuses to write the entry to long-term storage, preventing persistent
poisoning attacks.
Policy 11: “Right-to-be-Forgotten” Cascade (Planned)To comply with GDPR, this policy ensures that deleting a
root memory triggers a recursive purge of all derived summaries and insights. This prevents the agent from retaining
“Zombie Memories”-secrets that persist in summarized forms after the user has requested deletion.
B Evaluation Setup
To ensure a fair and direct comparison, we conducted two distinct evaluation runs, strictly matching the backbone
models and configurations of each respective baseline.
B.1 Experiment 1: MemOS Comparison Setup
For the MemOS comparison, we followed the official MemOS evaluation pipeline and reproduced its experimental
setup using the same foundation models and evaluation scripts. Both MemOS and MemArchitect were evaluated under
identical backbone and embedding configurations to isolate the effect of memory governance.
•Backbone LLM:We usedMeta-Llama-3.1-8B-Instructas the reasoning and generation model for both sys-
tems.
•Embedding Model:Both systems employedBGE-M3for dense retrieval and semantic similarity.
•Evaluation Protocol:We ran the original MemOS evaluation script on the LoCoMo benchmark and then eval-
uated MemArchitect using the same script, prompts, and judging logic, replacing only the memory backend.
Answer correctness was assessed using an LLM-as-a-Judge based on Llama-3.1-8B, following MemOS’s re-
ported methodology.
This controlled setup ensures that the performance differences reported in Table 1 arise from the memory architecture
and governance policies, rather than differences in models or evaluation procedures.
8

B.2 Experiment 2: SimpleMem Comparison Setup
We evaluate Memory Architect against SimpleMem under a strictly controlled setup to isolate the impact of the
memory architecture. Both systems use the same backbone LLM, embedding model, dataset, and evaluation protocol,
differing only in their memory management strategy.
•Models:Both systems useQwen2.5-3B-Instructas the generative backbone andnomic-embed-text-v1.5for
dense retrieval. To ensure consistency, the same LLM (Qwen2.5-3B-Instruct) is also used as an LLM-as-a-
Judge.
•Evaluation Protocol:We follow the official SimpleMem LoCoMo-10 evaluation script and apply the exact
same judging logic to Memory Architect, replacing only the memory backend. Answer correctness is deter-
mined via an LLM-as-a-Judge using a binary pass/fail decision based on semantic match, core fact preservation,
and acceptable temporal or lexical variance.
B.3 Dataset Configuration
Both experiments utilized theLoCoMo-10dataset, a long-horizon conversational benchmark consisting of 1,986 QA
pairs across multi-session dialogues. Results are reported by category (Single-Hop, Temporal, Multi-Hop, and Open-
Domain) to highlight specific architectural strengths.
9