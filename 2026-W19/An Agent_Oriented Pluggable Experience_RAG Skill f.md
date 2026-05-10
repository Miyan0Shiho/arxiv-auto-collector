# An Agent-Oriented Pluggable Experience-RAG Skill for Experience-Driven Retrieval Strategy Orchestration

**Authors**: Dutao Zhang, Tian Liao

**Published**: 2026-05-05 17:10:25

**PDF URL**: [https://arxiv.org/pdf/2605.03989v2](https://arxiv.org/pdf/2605.03989v2)

## Abstract
Retrieval-augmented generation systems often assume that one fixed retrieval pipeline is sufficient across heterogeneous tasks, yet factoid question answering, multi-hop reasoning, and scientific verification exhibit different retrieval preferences. We present Experience-RAG Skill, an agent-oriented pluggable retrieval orchestration layer positioned between the agent and the retriever pool. The proposed skill analyzes the current scene, consults an experience memory, selects an appropriate retrieval strategy, and returns structured evidence to the agent. Under a fixed candidate pool, Experience-RAG Skill achieves an overall nDCG@10 of 0.8924 on BeIR/nq, BeIR/hotpotqa, and BeIR/scifact, outperforming fixed single-retriever baselines and remaining competitive with Adaptive-RAG-style routing. The results suggest that retrieval strategy selection can be productively encapsulated as a reusable agent skill rather than being hard-coded in the upper workflow.

## Full Text


<!-- PDF content starts -->

An Agent-Oriented Pluggable Experience-RAG Skill for
Experience-Driven Retrieval Strategy Orchestration
zhangdutao
Macao Polytechnic University
p2527418@mpu.edu.moTian Liao
Macao Polytechnic University
p2421086@mpu.edu.mo
Abstract
Retrieval-augmented generation systems often assume that one fixed retrieval pipeline is
sufficient across heterogeneous tasks, yet factoid question answering, multi-hop reasoning,
and scientific verification exhibit different retrieval preferences. We present Experience-RAG
Skill, an agent-oriented pluggable retrieval orchestration layer positioned between the agent
and the retriever pool. The proposed skill analyzes the current scene, consults an experience
memory, selects an appropriate retrieval strategy, and returns structured evidence to the
agent. Under a fixed candidate pool, Experience-RAG Skill achieves an overall nDCG@10 of
0.8924 on BeIR/nq, BeIR/hotpotqa, and BeIR/scifact, outperforming fixed single-retriever
baselines and remaining competitive with Adaptive-RAG-style routing. The results suggest
that retrieval strategy selection can be productively encapsulated as a reusable agent skill
rather than being hard-coded in the upper workflow.
1 Introduction
Retrieval-augmented generation (RAG) improves knowledge access in large language models by
introducing external evidence before or during generation [1, 2, 3, 4]. Existing work has ex-
plored query rewriting, hierarchical retrieval, active retrieval, corrective retrieval, self-reflective
retrieval, and long-context retrieval [5, 6, 7, 8, 9, 10]. However, many practical systems still
assume that one fixed retrieval strategy is sufficient across all tasks.
This assumption is problematic in realistic agent settings. Factoid question answering,
multi-hop retrieval, and scientific claim verification do not rely on the same retrieval behavior.
A dense retriever may be strong for direct fact questions, while hybrid retrieval is often more
stable for multi-hop and scientific evidence matching. For an agent, the challenge is therefore
not only whether knowledge can be retrieved, but also how retrieval should be organized for
the current task.
We address this problem by redefining Experience-RAG as an agent-oriented pluggable skill
rather than a single retrieval method. This framing is related to recent work on tool-using
and acting language agents [11, 12, 13, 14, 15], but focuses specifically on retrieval strategy
orchestration. The proposed Experience-RAG Skill acts as a middle layer between the agent
and the retriever pool, with four core responsibilities: scene analysis, experience lookup, strategy
routing, and result packaging. The current study focuses on a fixed candidate pool setting and
investigates whether such a skill layer yields practical value even before open-world candidate
onboarding is solved.
Our contributions are threefold. First, we formulate retrieval strategy selection as an agent
skill orchestration problem. Second, we build an experience memory that records scene features,
multi-retriever performance, and strategy margins. Third, we evaluate the proposed skill on
heterogeneous retrieval tasks and compare it with both traditional and modern baselines.
1arXiv:2605.03989v2  [cs.AI]  7 May 2026

Figure 1: Overall framework of Experience-RAG Skill. The skill is positioned between the agent
and the retriever pool. It transforms the incoming query, history, and metadata into structured
scene features, consults reusable experience records, selects a retrieval strategy, invokes the
candidate retriever pool, and returns a packaged evidence object to the agent.
2 Experience-RAG Skill
2.1 Framework
Given a queryq, dialogue historyh, task metadatam, and a candidate retriever poolR,
Experience-RAG Skill first constructs a structured scene representations, then routes the query
using experience memoryM, and finally returns a standardized retrieval package:
s=Analyze(q, h, m), r∗=Route(s, M), o=RetrieveAndPackage(q, r∗).(1)
The framework contains six modules: Skill Interface, Scene Analyzer, Experience Memory,
Strategy Router, Retriever Pool, and Result Packager. The agent only interacts with the unified
skill interface, while the lower retrieval methods remain encapsulated inside the pool.
2.2 Experience Memory and Routing
The experience memory stores complete records rather than only the best-strategy label:
e= (scene features, score vector, best strategy, best margin).(2)
In the current version, scene features include task type, domain, context length, question
complexity, query style, and document structure. These features support both rule-based and
learned routing.
Our strongest current setting is rule-based routing. It mapsdirecttasks todense, and both
multi hopandscientifictasks tohybrid rrf. Learned routing variants were also explored,
but under the present experience scale they did not outperform the rule-based baseline.
2.3 Difference from Adaptive-RAG
Adaptive-RAG-style methods focus on complexity-driven strategy selection. In contrast, Experience-
RAG Skill focuses on an agent-facing orchestration layer with explicit experience reuse, unified
2

Table 1: Overall retrieval performance.
Method Recall@10 MRR@10 nDCG@10
BM25 0.9012 0.8556 0.8426
Rewrite-BM25 0.8956 0.8566 0.8412
Dense 0.9136 0.8819 0.8627
Hybrid RRF 0.9428 0.8846 0.8802
Experience-RAG Skill0.9428 0.9006 0.8924
Table 2: Comparison with modern baselines.
Method Recall@10 MRR@10 nDCG@10
Adaptive-RAG-style0.9428 0.9019 0.8934
Experience-RAG Skill0.94280.9006 0.8924
HyDE 0.8983 0.8533 0.8326
LongRAG-style 0.7399 0.7407 0.7095
RAPTOR-style 0.7155 0.7223 0.6841
interfaces, and structured evidence packaging. This difference is important because our goal is
not only to choose a retriever, but also to make retrieval strategy selection reusable as an agent
capability.
3 Evaluation
3.1 Setup
We evaluate on three public retrieval benchmarks: BeIR/nq, BeIR/hotpotqa, and BeIR/scifact,
using 120 queries from each dataset and sampled candidate corpora. We compare againstbm25,
rewrite bm25,dense,hybrid rrf, andexperience rag. We report Recall@10, MRR@10, and
nDCG@10.
3.2 Main Results
Table 1 shows that Experience-RAG Skill achieves the best overall retrieval quality, with
nDCG@10 of 0.8924, Recall@10 of 0.9428, and MRR@10 of 0.9006. It outperforms the strongest
fixed single-method baselinehybrid rrf(0.8802 nDCG@10) as well asdense(0.8627),bm25
(0.8426), andrewrite bm25(0.8412). At the dataset level, Experience-RAG ties with the
strongest fixed strategy on each dataset while outperforming all fixed strategies on the mixed
workload, indicating that its main advantage lies in heterogeneous orchestration rather than
dominance on every single subtask.
3.3 Modern Baseline Comparison
We further expand the candidate pool withHyDE,RAPTOR-style,LongRAG-style, andAdaptive-RAG-style,
covering query expansion, hierarchical retrieval, long-context retrieval, and adaptive routing
[16, 6, 10, 17]. Table 2 shows that Adaptive-RAG-style is slightly better overall (0.8934 vs. 0.8924
nDCG@10), but the gap is very small. In contrast, HyDE, LongRAG-style, and RAPTOR-style
are clearly weaker under the current sampled-corpus and retrieval-only setting. These results
suggest that Experience-RAG Skill is already competitive with the strongest modern routing
baseline while providing a clearer agent-facing systems boundary.
3

4 Brief Discussion
The current results support three conclusions. First, retrieval strategy selection can be pro-
ductively encapsulated as an agent skill rather than being embedded in the upper workflow.
Second, experience-driven orchestration is effective on heterogeneous retrieval workloads. Third,
the skill remains competitive after modern baselines are added to the same fixed candidate pool.
At the same time, the current version has clear limitations. The experiments are based
on sampled rather than full-scale corpora, learned routing does not yet outperform the rule-
based baseline, and the current implementation still assumes a fixed candidate pool. Moreover,
the present workflow case analysis is qualitative and does not yet constitute a full end-to-end
interactive agent benchmark. These limitations make the current version more appropriate as
a short paper or preprint than as a full systems paper.
5 Conclusion
We introduced Experience-RAG Skill, an agent-oriented pluggable retrieval orchestration layer
that combines scene analysis, experience memory, and strategy routing. Under a fixed candidate
pool, the proposed skill outperforms fixed single-retriever baselines and remains competitive
with Adaptive-RAG-style routing on heterogeneous retrieval tasks. The study suggests that
retrieval strategy selection is a reusable agent capability in its own right. Future work will
extend this line toward dynamic candidate onboarding, capability-level routing, and full end-
to-end agent evaluation.
References
[1] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. arXiv preprint arXiv:2005.11401, 2020.
[2] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM:
Retrieval-Augmented Language Model Pre-Training. In Proceedings of ICML, 2020.
[3] Gautier Izacard and Edouard Grave. Leveraging Passage Retrieval with Generative Models
for Open Domain Question Answering. arXiv preprint arXiv:2007.01282, 2021.
[4] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, et al. Improving Language Models
by Retrieving from Trillions of Tokens. In Proceedings of ICML, 2022.
[5] Xueguang Ma, Yeyun Gong, Pengcheng He, et al. Query Rewriting in Retrieval-Augmented
Large Language Models. In Proceedings of EMNLP, 2023.
[6] Parth Sarthi, Salman Abdullah, Aman Tuli, et al. RAPTOR: Recursive Abstractive Pro-
cessing for Tree-Organized Retrieval. arXiv preprint arXiv:2401.18059, 2024.
[7] Zhengbao Jiang, Frank F. Xu, Luyu Gao, et al. Active Retrieval Augmented Generation.
arXiv preprint arXiv:2305.06983, 2023.
[8] Shengnan Yan et al. CRAG: Corrective Retrieval Augmented Generation. arXiv preprint
arXiv:2401.15884, 2024.
[9] Akari Asai et al. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-
Reflection. arXiv preprint arXiv:2310.11511, 2023.
[10] Jinhao Jiang, Yuhui Xu, Ming Feng, et al. LongRAG: Enhancing Retrieval-Augmented
Generation with Long-context LLMs. arXiv preprint arXiv:2406.15319, 2024.
4

[11] Shunyu Yao, Jeffrey Zhao, Dian Yu, et al. ReAct: Synergizing Reasoning and Acting in
Language Models. In Proceedings of ICLR, 2023.
[12] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, et al. Toolformer: Language Models Can
Teach Themselves to Use Tools. arXiv preprint arXiv:2302.04761, 2023.
[13] Yongliang Shen, Kaitao Song, Xu Tan, et al. HuggingGPT: Solving AI Tasks with Chat-
GPT and its Friends in Hugging Face. In Advances in Neural Information Processing
Systems, 2023.
[14] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large
Language Model Connected with Massive APIs. arXiv preprint arXiv:2305.15334, 2023.
[15] Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. Cog-
nitive Architectures for Language Agents. Transactions on Machine Learning Research,
2023.
[16] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise Zero-Shot Dense Retrieval
without Relevance Labels. arXiv preprint arXiv:2212.10496, 2022.
[17] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-
RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question
Complexity. arXiv preprint arXiv:2403.14403, 2024.
[18] Nandan Thakur, Nils Reimers, Andreas Rueckle, et al. BEIR: A Heterogeneous Benchmark
for Zero-shot Evaluation of Information Retrieval Models. arXiv preprint arXiv:2104.08663,
2021.
A Supplementary Experiments
A.1 Ablation Study
To better understand where the gains of Experience-RAG Skill come from, we also evaluated
several ablated variants. The full Experience-RAG configuration achieves nDCG@10 of 0.8924,
whereas fixing the strategy tohybrid rrfyields 0.8802, fixing it todenseyields 0.8627, and
fixing it tobm25yields 0.8426. These results indicate that the main advantage comes from
task-aware orchestration rather than from a single strong retriever.
Table 3: Ablation results.
Variant Recall@10 MRR@10 nDCG@10
Full Experience-RAG0.9428 0.9006 0.8924
Fixed Hybrid RRF 0.9428 0.8846 0.8802
Fixed Dense 0.9136 0.8819 0.8627
Fixed BM25 0.9012 0.8556 0.8426
A.2 Learned Routing Attempts
We additionally tested learned routing variants in order to examine whether the experience
memory can support automatic router learning. A hard-classification router reached nDCG@10
of 0.8778, while a score-regression router reached 0.8627. Both remained below the rule-based
variant. Under the current scale of the experience memory, these results suggest that learned
routing is feasible but not yet mature enough to replace the stronger rule-based skill.
5

A.3 Preliminary Agent Workflow Observation
We also carried out a qualitative workflow inspection with three representative tasks: factoid
question answering, multi-hop retrieval, and scientific claim verification. Across these examples,
Experience-RAG Skill consistently made the routing decision explicit and inspectable, whereas
fixed-retriever agents lacked a dedicated mechanism for selecting retrieval behavior. This ob-
servation is supportive but still qualitative, and should not be interpreted as a full end-to-end
agent benchmark.
6