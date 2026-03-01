# Towards Faithful Industrial RAG: A Reinforced Co-adaptation Framework for Advertising QA

**Authors**: Wenwei Li, Ming Xu, Tianle Xia, Lingxiang Hu, Yiding Sun, Linfang Shang, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang

**Published**: 2026-02-26 03:35:09

**PDF URL**: [https://arxiv.org/pdf/2602.22584v1](https://arxiv.org/pdf/2602.22584v1)

## Abstract
Industrial advertising question answering (QA) is a high-stakes task in which hallucinated content, particularly fabricated URLs, can lead to financial loss, compliance violations, and legal risk. Although Retrieval-Augmented Generation (RAG) is widely adopted, deploying it in production remains challenging because industrial knowledge is inherently relational, frequently updated, and insufficiently aligned with generation objectives. We propose a reinforced co-adaptation framework that jointly optimizes retrieval and generation through two components: (1) Graph-aware Retrieval (GraphRAG), which models entity-relation structure over a high-citation knowledge subgraph for multi-hop, domain-specific evidence selection; and (2) evidence-constrained reinforcement learning via Group Relative Policy Optimization (GRPO) with multi-dimensional rewards covering faithfulness, style compliance, safety, and URL validity. Experiments on an internal advertising QA dataset show consistent gains across expert-judged dimensions including accuracy, completeness, and safety, while reducing the hallucination rate by 72\%. A two-week online A/B test demonstrates a 28.6\% increase in like rate, a 46.2\% decrease in dislike rate, and a 92.7\% reduction in URL hallucination. The system has been running in production for over half a year and has served millions of QA interactions.

## Full Text


<!-- PDF content starts -->

Towards Faithful Industrial RAG: A Reinforced Co-adaptation Framework
for Advertising QA
Wenwei Li∗, Ming Xu∗, Tianle Xia, Lingxiang Hu, Yiding Sun, Linfang Shang
Liqun Liu†,Peng Shu,Huan Yu,Jie Jiang
Tencent
{wenweiwwli,flemingxu,tianlexia,lingxianghu,emanuelsun,faelynshang}@tencent.com
{liqunliu,archershu,huanyu,zeus}@tencent.com
∗Equal contribution.†Corresponding author.
Abstract
Industrial advertising question answering (QA)
is a high-stakes task in which hallucinated con-
tent, particularly fabricated URLs, can lead to
financial loss, compliance violations, and legal
risk. Although Retrieval-Augmented Gener-
ation (RAG) is widely adopted, deploying it
in production remains challenging because in-
dustrial knowledge is inherently relational, fre-
quently updated, and insufficiently aligned with
generation objectives. We propose a reinforced
co-adaptation framework that jointly optimizes
retrieval and generation through two compo-
nents: (1) Graph-aware Retrieval (GraphRAG),
which models entity-relation structure over a
high-citation knowledge subgraph for multi-
hop, domain-specific evidence selection; and
(2) evidence-constrained reinforcement learn-
ing via Group Relative Policy Optimization
(GRPO) with multi-dimensional rewards cover-
ing faithfulness, style compliance, safety, and
URL validity. Experiments on an internal ad-
vertising QA dataset show consistent gains
across expert-judged dimensions including ac-
curacy, completeness, and safety, while reduc-
ing the hallucination rate by 72%. A two-week
online A/B test demonstrates a 28.6% increase
in like rate, a 46.2% decrease in dislike rate,
and a 92.7% reduction in URL hallucination.
The system has been running in production for
over half a year and has served millions of QA
interactions.
1 Introduction
Online advertising platforms are complex, fast-
evolving ecosystems where intelligent customer
service (ICS) systems are critical for operational
efficiency and user satisfaction (Gao et al., 2025).
These systems must handle diverse intents, from
pre-sales consultations to post-sales compliance
appeals, under frequently updated internal poli-
cies (e.g., ad review guidelines, account systems,
reimbursement protocols) that are often behind
private knowledge barriers (Sharma et al., 2024).
Figure 1: Traditional QA vs. our approach over a shared
knowledge base. Given the same user query and knowl-
edge items A, B, C, D, traditional methods often yield
incomplete,hallucinated,over-generated, orverbose
answers. Our method produces anexactanswer that
remains complete, faithful, and concise.
In this high-stakes setting, even minor factual er-
rors can trigger compliance risks, user harm, and
direct financial losses, and fabricated structured
items such as URLs are particularly costly (Ji et al.,
2023; Ming et al., 2025). As illustrated in Fig-
ure 1, conventional pipelines may produce incom-
plete, hallucinated, over-generated, or verbose an-
swers, whereas our framework targets concise and
evidence-grounded responses.
Retrieval-Augmented Generation (RAG) is a
standard paradigm for grounding LLMs in external
evidence (Lewis et al., 2020; Gao et al., 2024), yet
production advertising question answering (QA)
reveals three key gaps. First, industrial knowl-
edge is relational and process-driven (e.g., prod-
ucts, rules, procedures), where single-shot hybrid
retrieval can miss multi-hop dependencies; graph-
based retrieval such as GraphRAG addresses this by
explicitly modeling entities and relations for cross-
document reasoning (Edge et al., 2024). Second,
simply expanding context length is insufficient: un-
der the “lost in the middle” effect, models may fail
to reliably use evidence in long inputs, motivating
targeted evidence selection (Liu et al., 2023). Third,
generation must satisfy strict style and compliance
1arXiv:2602.22584v1  [cs.CL]  26 Feb 2026

constraints, yet even strong models may deviate
from provided context under unanswerable or coun-
terfactual inputs (Ming et al., 2025; Rakin et al.,
2024); while reinforcement learning (Schulman
et al., 2017a; Rafailov et al., 2023) can align gener-
ation with task constraints, and post-hoc methods
such as SelfCheckGPT (Manakul et al., 2023) can
detect unsupported content, treating retrieval and
generation as isolated stages leaves a coordination
gap.
To address these gaps, we propose an end-to-
end reinforced co-adaptation framework that jointly
optimizes retrieval and evidence-grounded gener-
ation. It has two key components: (1)Graph-
aware Retrievalvia GraphRAG, which models
relationships between products, rules, and pro-
cesses to support multi-hop reasoning and termi-
nology alignment (Edge et al., 2024); and (2)
Evidence-constrained Reinforcement Learning
(RL), which aligns the generator with retrieved
evidence using multi-dimensional rewards that en-
courage faithfulness while enforcing style, safety,
and URL validity.
Our contributions are as follows:
•We propose a co-adaptation framework that
jointly optimizes GraphRAG-based retrieval
and an RL-tuned generator, achieving superior
alignment between retrieved domain knowl-
edge and generated responses.
•We design a multi-dimensional RL objective
covering faithfulness, style compliance, safety,
and URL validity, explicitly penalizing unsup-
ported content and hallucinated links.
•We deploy the system on a large-scale adver-
tising platform, serving millions of QA inter-
actions over half a year. A two-week A/B
test shows a 28.6% like-rate increase, a 46.2%
dislike-rate reduction, and a 92.7% reduction
in URL hallucination.
2 Methodology
2.1 Problem Formulation
We formulate advertising QA as a constrained con-
ditional generation task (Figure 2). Given a user
query qand a dynamically updated private knowl-
edge base K, the system retrieves a relevant ev-
idence set D={d 1, . . . , d k}. The generator πθ
produces a response Athat maximizes P(A|q, D)subject to constraints C, including zero URL hal-
lucinations, domain-specific style compliance, and
safety requirements.
2.2 Graph-aware Retrieval
To address the limitations of traditional hybrid
retrieval methods (e.g., BGE + BM25) in han-
dling complex multi-hop dependencies and domain-
specific terminology, we propose a Graph-aware
Retrieval module that integrates GraphRAG with
a carefully curated, high-citation knowledge base,
complemented by a parallel retrieval architecture
for industrial-scale deployment.
High-Citation Knowledge Base.GraphRAG en-
hances retrieval but introduces substantial compu-
tational overhead. To balance effectiveness and
efficiency, we maintain a high-citation knowledge
baseKh⊂K through traffic-driven feedback. We
accumulate recall frequency for each knowledge
chunk from production query logs as a “citation
heat” indicator, and periodically select the top- N%
most frequently cited items to form Kh. This cu-
rated subset serves as the subgraph for GraphRAG,
reducing traversal complexity while preserving ef-
fectiveness through automatic rolling updates.
GraphRAG Architecture.We construct a
knowledge graph G= (V, E) overKhvia entity
extraction and relation identification, with commu-
nity detection partitioning the graph into hierar-
chical subgraphs for semantic aggregation. The
retrieval layer supports dynamic routing between
hybrid retrieval and graph-based traversal, balanc-
ing efficiency for simple queries with multi-hop rea-
soning for complex ones. High-citation subgraph
pruning constrains retrieval scope, and incremental
updates maintain temporal currency without full
reconstruction.
Parallel Retrieval Architecture.To mitigate
GraphRAG latency while maximizing recall, we
execute GraphRAG and traditional RAG channels
concurrently. The GraphRAG channel performs
asynchronous graph traversal over Khfor multi-
hop reasoning, while the traditional RAG channel
uses BGE + BM25 hybrid retrieval with multi-path
query rewriting that decomposes complex queries
into parallel sub-queries. Results from both chan-
nels are merged and deduplicated to form the final
evidence setD={d 1, . . . , d k}.
2

Retrieval Module
（GraphRAG + Vector +
BM25）RL-tuned GeneratorEvidence-constrained Generation & RL-tuned
Query
(Policy violation 
check)Graph-awar e Retrieval Module
Evidence SetLinkSafety Faithfulness
  Style Compliance
(Domain 
formatting)
Query
Answer Evidence Set
Zero-URL  Hallucination
Domain-specific Style
SatetyCitation  
HeatGraphRAG Channel
High-Citation Sources
Merge &
DeduplicationGraph Traversal &
Multi-hop Reasoning
Self-iterative
UpdatesTraditional RAG Channel
Query Rewriting
BM25Evidence SetRL-tuned Generator
(Qwen3-32B-RL)
Reward Models
GRPO 
Update
(Groundtruth
 comparison)
Multi-dimensional Reward
Vector
(Valid/hallucinated
URLs)
Private
Knowledge
Basesub-querysub-querysub-query
Vector
Figure 2: System overview. Given a user query qand a private knowledge base K, the retrieval system constructs
an evidence set Dvia two parallel channels: a GraphRAG channel over a high-citation knowledge base Khand
a traditional RAG channel with query rewriting and BGE + BM25 hybrid retrieval. Results are merged and
deduplicated. The RL-tuned generator then produces a response optimized by GRPO with multi-dimensional
rewards for faithfulness, style compliance, safety, and URL validity.
2.3 Evidence-constrained Generation
The generation module centers on an RL-tuned
generator (Qwen3-32B-RL). While supervised fine-
tuning establishes foundational formatting, re-
inforcement learning is critical for steering the
model toward stable, safe, and hallucination-free
responses under strict industrial constraints.
We optimize the generator using GRPO (Shao
et al., 2024), whose group-based mechanism stabi-
lizes training under noisy reward signals. Unlike
PPO(Schulman et al., 2017b), which requires a
separate critic model, GRPO estimates the base-
line from group rewards, reducing memory over-
head and training instability. This is particularly
valuable for industrial applications where reward
signals are inherently noisy due to the subjective
nature of style and safety assessments.
We design a multi-dimensional reward function:
R=λ 1Rf+λ 2Rs+λ 3Ra+λ 4Rh (1)
where λiare weighting coefficients. Following
preliminary experiments, we set λ3= 2.0 and
λ4= 2.0 to prioritize safety and hallucination re-
duction, with λ1=λ 2= 1.0 . The reward compo-
nents are:
•Evidence Faithfulness ( Rf): Measures align-
ment with the ground-truth answer via pairwiseLLM-as-judge comparison.
•Style Compliance ( Rs): Evaluates adherence to
advertising domain conventions, including tone,
professionalism, and formatting.
•Safety ( Ra): Detects platform policy violations
and ensures regulatory compliance.
•URL Validity ( Rh): Rewards valid URLs and
penalizes hallucinated ones. A URL is valid if it
appears in the evidence D, or if its prefix belongs
to an approved pool and its HTTP status code is
in{200,301,302}.
The full reward computation procedure, including
URL extraction and validation details, is provided
in Algorithm 1 in the Appendix.
3 Experiments
We evaluate our approach using both offline and
online metrics.
3.1 Experimental Setting
Dataset.We evaluate on theAdvertising QA
Dataset, an internal Chinese advertising customer-
service dataset with 3,000 expert-annotated
question–answer pairs. For out-of-domain gener-
alization, we useFaithEval(Ming et al., 2025),
which tests faithfulness under unanswerable ques-
3

tions, counterfactual contexts, and inconsistent in-
formation.
Evaluation Protocol.We compare systems
along two axes: (i)retrieval, where we con-
trast Base RAG (a standard RAG pipeline with
a reranker) with GraphRAG, and (ii)generation
backbones, where we evaluate open-source and
proprietary models as well as our RL-tuned model.
We use a hybrid evaluation protocol. ROUGE-L
(0–100) is computed automatically, while Accu-
racy, Completeness, Clarity, Style, and Safety are
rated by human experts on a 0–10 scale. Halluci-
nation Rate is also assessed by human experts at
the case level: for each case, if the answer contains
any fabricated or unsupported content, we count it
as one hallucinated case. Formally, given Ncases
and an indicatorI[·], we report
HR =1
NNX
i=1I[answer icontains hallucination].
ROUGE-L measures lexical overlap with the refer-
ence; lower HR indicates fewer hallucinated cases.
Models.We evaluate five representative back-
bones: DeepSeek-V3.2 (DeepSeek-AI et al., 2025),
GPT-5.2 (OpenAI, 2025), Qwen3-32B (Qwen
Team, 2025), Qwen3-32B-SFT, and Qwen3-32B-
RL (ours). All evaluated models support reason-
ing capabilities; to match production latency con-
straints, we evaluate all models in non-thinking
mode for a fair comparison.
This model set covers strong open-source and
commercial baselines, isolates the impact of RL (vs.
SFT) on the same backbone, and tests robustness
across model families.
3.2 Main Results
Table 1 reports the main offline results. Replacing
Base RAG with GraphRAG consistently improves
quality and reduces hallucinations. DeepSeek-V3.2
improves ROUGE-L from 33.27 to 37.00 (+3.73)
and reduces Hallucination Rate from 0.0077 to
0.0030 (61% relative reduction). Similar pat-
terns hold for GPT-5.2 (ROUGE-L: 32.82 →35.88;
Hallucination Rate: 0.0057 →0.0023, 60%) and
Qwen3-32B (ROUGE-L: 29.39 →32.96; Halluci-
nation Rate: 0.0117 →0.0060). Graph-aware multi-
hop evidence aggregation strengthens both cover-
age and grounding beyond hybrid retrieval alone.
RL provides additional gains beyond retrieval
improvements. Under GraphRAG, Qwen3-32B-RL(Ours) improves ROUGE-L from 33.82 to 35.49
over Qwen3-32B-SFT (+1.67), and lowers Halluci-
nation Rate from 0.0047 to 0.0013 (72% relative re-
duction). Even under Base RAG, Qwen3-32B-RL
achieves a 0.0017 Hallucination Rate, indicating
that evidence-constrained RL targets hallucination
behaviors that supervised fine-tuning alone cannot
eliminate.
The complementary effect between GraphRAG
and RL is evident across all metrics. GraphRAG
primarily improves coverage-related metrics
(ROUGE-L, Completeness), while RL enhances re-
liability and compliance metrics (Style, Safety, Hal-
lucination Rate). Their combination achieves the
best overall performance, with our final system out-
performing the strongest baseline (DeepSeek-V3.2
with GraphRAG) on Hallucination Rate (0.0030
vs. 0.0013) while maintaining competitive quality
scores.
3.3 GraphRAG Effectiveness
We evaluate graph-aware retrieval both offline and
online.
Offline Evaluation.We assess retrieval via side-
by-side expert comparison and knowledge recall
analysis.
Knowledge Recall Enhancement.Figure 3 shows
progressive improvements in knowledge recall. Ef-
fective knowledge chunks per query increase from
3.9 (Base RAG) to 4.5 (GraphRAG) to 6.3 (paral-
lel retrieval), a 61.5% overall improvement. Re-
call effectiveness improves from 73.6% to 90.5%,
demonstrating that GraphRAG combined with par-
allel retrieval substantially enriches contextual in-
formation.
Retrieval Quality Optimization.In expert
evaluation, the Good:Same:Bad ratio reaches
32.4%:64.9%:2.7% at retrieval. The Good ratio
is 12×higher than Bad, indicating effective noise
filtering.
End-to-End Performance.The end-to-end
Good:Same:Bad ratio reaches 24.3%:71.6%:4.1%,
with positive gains outweighing negative impacts
by 6×.
Online A/B Testing.We deployed at 50% traffic.
Table 2 shows consistent improvements: like rate
increases from 0.21% to 0.27% (+28.6%), dislike
rate decreases from 0.26% to 0.18% ( −30.8%), and
average conversation turns increase from 1.54 to
1.81 (+17.5%), indicating improved user engage-
ment.
4

Metric DeepSeek-V3.2 GPT-5.2 Qwen3-32B Qwen3-32B-SFT Qwen3-32B-RL
Base RAG GraphRAG Base RAG GraphRAG Base RAG GraphRAG Base RAG GraphRAG Base RAGOurs
ROUGE-L↑ 33.2737.00 32.82 35.88 29.39 32.96 30.79 33.82 31.40 35.49
Accuracy↑ 7.82 8.37 7.948.39 7.25 7.82 7.50 8.10 7.85 8.26
Completeness↑ 6.787.10 6.70 7.08 6.20 6.66 6.43 6.91 6.46 6.99
Clarity↑ 8.968.99 8.92 8.97 8.52 8.74 8.82 8.94 8.83 8.95
Style↑ 8.19 8.25 8.14 8.25 7.82 8.03 8.07 8.19 8.27 8.33
Safety↑ 9.94 9.93 9.95 9.96 9.88 9.91 9.95 9.94 9.97 9.99
Hallucination Rate↓ 0.0077 0.0030 0.0057 0.0023 0.0117 0.0060 0.0117 0.0047 0.0017 0.0013
Table 1: Main experimental results.Oursrefers to Qwen3-32B-RL with GraphRAG. Best results inbold, second
best underlined .
Base RAG GraphRAG Parallel02468Chunks/Query3.94.56.3Knowledge Recall Enhancement
Chunks/Query
Recall Eff. (%)
020406080100
Recall Eff. (%)73.6%84.2%90.5%
Figure 3: Knowledge recall enhancement across Base
RAG, GraphRAG, and Parallel retrieval. Effective
chunks pre query and recall effectiveness in percent.
20 40 60 80 100
Training Steps0.00.20.40.60.81.0Normalized Performance
Style
Faithfulness
LinkSafety
Overall
Figure 4: Training dynamics of multi-dimensional re-
ward components during RL.
3.4 RL Reward Effectiveness
Figure 4 shows consistent improvement across all
reward components during RL fine-tuning. With
only 1,000 training samples, all metrics rapidly
improve within 100 steps and converge, demon-
strating efficient reward design.
The reward components exhibit distinct opti-
mization patterns. Faithfulness and URL validity
rewards show the steepest initial ascent, indicating
that the model quickly learns to align with retrieved
evidence and avoid hallucinated links. Style and
safety rewards improve more gradually, reflectingMetric Base RAG Ours∆
Like Rate (%) 0.21 0.27 +28.6%
Dislike Rate (%) 0.26 0.18−30.8%
Avg. Conv. Turns 1.54 1.81 +17.5%
Table 2: Online A/B testing at 50% traffic.
the nuanced nature of domain-specific tone and
compliance requirements. The overall reward con-
verges to a stable high value, suggesting that the
multi-objective optimization achieves balanced im-
provements across all dimensions without detri-
mental trade-offs.
3.5 Generalization on FaithEval
To assess whether our RL-tuned model general-
izes beyond the in-domain setting, we evaluate on
FaithEval. Figure 5 shows the results.
Our RL-tuned model improves over
Qwen3-32B on all FaithEval subsets: Unan-
swerable 44.60% →53.40%, Counterfactual
57.90% →64.40% (outperforming DeepSeek-V3.2
at 56.40%), and Inconsistent 63.80% →84.60%.
The gains on Unanswerable and Counterfactual
suggest stronger refusal behavior when context is
missing or misleading. On Inconsistent, it reaches
84.60%, substantially above Qwen3-32B (63.80%)
and closer to DeepSeek-V3.2 (94.80%). These
results indicate improved contextual faithfulness
without degrading generalization.
3.6 Production Deployment
3.6.1 Offline Evaluation
We compare against a Base RAG + DeepSeek-
V3(Liu et al., 2024) baseline via expert assessment
on completeness, professionalism, compliance, and
hallucination. As shown in Figure 6, our method
wins substantially more often than it loses, with
the largest gains in professionalism (45.2% win)
and compliance (41.9% win) and a low loss rate
(1.1%). It also improves hallucination outcomes
5

Incons. Unans. Cfact. Overall0255075100Accuracy (%)94.8
54.256.468.5
63.8
44.657.955.484.6
53.464.467.5DeepSeek-V3.2
Qwen3-32B
Qwen3-32B-RL (Ours)Figure 5: FaithEval generalization: accuracy (%) on In-
consistent, Unanswerable, Counterfactual, and Overall.
0 20 40 60 80 100
Percentage (%)Completeness
Professionalism
Compliance
Hallucination29.0% 64.5% 6.5%
45.2% 53.8% 1.1%
41.9% 57.0% 1.1%
11.7% 88.1% 0.1%
Ours Win Tie Baseline Win
Figure 6: Offline evaluation comparison: win/tie/lose
distribution across four dimensions.
(7.7% win vs. 0.1% loss), supporting the benefit of
co-adapting GraphRAG and RL-tuned generation.
3.6.2 Online A/B Testing
A two-week online A/B test compares our deployed
system against the Base RAG + DeepSeek-V3 base-
line, with a 50%/50% traffic split (Table 3). Our
method increases like rate from 0.21% to 0.27%
(+28.6%), decreases dislike rate from 0.26% to
0.14% ( −46.2%), and reduces URL hallucination
from 0.0041% to 0.0003% ( −92.7%). Average
first-token latency rises from 2.5s to 3.1s (+24.0%),
which remains acceptable in practice. Overall, the
A/B results suggest that reinforced co-adaptation
improves both user satisfaction and reliability un-
der real traffic, with a manageable latency trade-off.
3.6.3 Latency Analysis
Table 4 details the latency distribution. Query
rewriting takes 690ms, parallel retrieval takes
852ms (GraphRAG) and 167ms (BGE + BM25),
reranking takes 557ms, generation takes 801ms,
and safety guardrails take 230ms. Total latency
is 3130ms, meeting acceptable thresholds for user
experience and industrial deployment.
The latency breakdown highlights several av-
enues for optimization. GraphRAG retrieval incursMetric Baseline Ours∆
Like Rate (%) 0.21 0.27 +28.6%
Dislike Rate (%) 0.26 0.14−46.2%
URL Hallu. (%) 0.0041 0.0003−92.7%
Latency (s) 2.5 3.1 +24.0%
Table 3: Online A/B testing results (two weeks, 50%
traffic).
Module Latency (ms)
Query Rewriting 690
GraphRAG Retrieval 852
BGE + BM25 Retrieval 167
Reranking 557
Generation 801
Safety Guardrails 230
Total 3130
Table 4: Latency breakdown by module.
the largest single latency cost (852ms), which mo-
tivates our high-citation knowledge base design
to constrain graph traversal and reduce overhead.
Executing GraphRAG in parallel with the BGE +
BM25 pipeline ensures that the slower graph-based
retrieval does not block the faster hybrid channel.
Generation latency (801ms) is on par with stan-
dard large language model inference, suggesting
that RL fine-tuning does not introduce noticeable
computational overhead relative to the base model.
Safety guardrails incur an additional 230ms as post-
processing, without affecting time-to-first-token,
thereby preserving system responsiveness.
4 Conclusion
We present a reinforced co-adaptation framework
to mitigate hallucinations in industrial advertising
Q&A by jointly optimizing GraphRAG and an RL-
tuned generator guided by multi-dimensional re-
wards, thereby narrowing the retrieval–generation
gap and reducing unsupported content and halluci-
nated or invalid links. Our results show that graph-
aware retrieval with a high-citation knowledge base
balances multi-hop evidence aggregation with com-
putational efficiency, while evidence-constrained
RL further suppresses hallucinations without sac-
rificing domain style compliance or safety. Ex-
tensive offline evaluations, a two-week production
A/B test, and over six months of deployment collec-
tively validate that the approach improves answer
reliability and user-facing quality at scale under
practical latency constraints.
6

Ethics Statement
Our research targets high-stakes industrial adver-
tising question answering and adheres to ethical
principles that prioritize user rights. We aim to im-
prove system reliability and safety by reducing un-
supported claims and hallucinated or invalid URLs
that could mislead users or introduce compliance
risks. Any dataset examples are used solely for
scientific analysis and do not necessarily reflect the
views of the authors. All resources are intended for
scientific research purposes only, contributing to
the development of more secure and reliable digital
platforms.
References
DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin,
Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao
Wu, Bowei Zhang, Chaofan Lin, Chen Dong, and
1 others. 2025. Deepseek-v3.2: Pushing the fron-
tier of open large language models.arXiv preprint
arXiv:2512.02556.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.arXiv preprint arXiv:2404.16130.
Tianhong Gao, Jundong Shen, Jiapeng Wang, Bei Shi,
Ying Ju, Junfeng Yao, and Huiyu Yu. 2025. Bench-
marking and learning real-world customer service
dialogue.arXiv preprint arXiv:2510.22143.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2312.10997.
Gemini Team, Google. 2025. Gemini 2.5: Pushing
the frontier with advanced reasoning, multimodality,
long context, and next generation agentic capabilities.
Technical report, accessed 2026-02-08.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of halluci-
nation in natural language generation.ACM Comput-
ing Surveys, 55(12):1–38.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Ef-
ficient memory management for large language
model serving with PagedAttention.arXiv preprint
arXiv:2309.06180.Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks. In
Advances in Neural Information Processing Systems,
volume 33, pages 9459–9474.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi
Deng, Chenyu Zhang, Chong Ruan, and 1 others.
2024. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin
Paranjape, Michele Bevilacqua, Fabio Petroni, and
Percy Liang. 2023. Lost in the middle: How lan-
guage models use long contexts.arXiv preprint
arXiv:2307.03172. Accepted to TACL 2023.
Potsawee Manakul, Adian Liusie, and Mark Gales.
2023. Selfcheckgpt: Zero-resource black-box hal-
lucination detection for generative large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 9004–9017, Singapore. Association for Com-
putational Linguistics.
Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zix-
uan Ke, Xuan-Phi Nguyen, Caiming Xiong, and
Shafiq Joty. 2025. Faitheval: Can your language
model stay faithful to context, even if “the moon is
made of marshmallows”. InInternational Confer-
ence on Learning Representations.
OpenAI. 2025. Introducing gpt-5.2. Accessed: 2026-
02-08.
Qwen Team. 2025. Qwen3-32b. Hugging Face model,
accessed 2026-02-08.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano
Ermon, Christopher D. Manning, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model. InAdvances in
Neural Information Processing Systems.
Salman Rakin, Md. A. R. Shibly, Zahin M. Hossain,
Zeeshan Khan, and Md. Mostofa Akbar. 2024. Lever-
aging the domain adaptation of retrieval augmented
generation models for question answering and reduc-
ing hallucination.arXiv preprint arXiv:2410.17783.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec
Radford, and Oleg Klimov. 2017a. Proximal
policy optimization algorithms.arXiv preprint
arXiv:1707.06347.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec
Radford, and Oleg Klimov. 2017b. Proximal
policy optimization algorithms.arXiv preprint
arXiv:1707.06347.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
7

Deepseekmath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint
arXiv:2402.03300.
Sanat Sharma, David Seunghyun Yoon, Franck Dernon-
court, Dewang Sultania, Karishma Bagga, Mengjiao
Zhang, Trung Bui, and Varun Kotte. 2024. Retrieval
augmented generation for domain-specific question
answering.arXiv preprint arXiv:2404.14760.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. 2024. HybridFlow: A flexi-
ble and efficient RLHF framework.arXiv preprint
arXiv:2409.19256.
Tencent Hunyuan Team. 2025. Hunyuan-TurboS: Ad-
vancing large language models through mamba-
transformer synergy and adaptive chain-of-thought.
Preprint, arXiv:2505.15431.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan,
Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin,
Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan
Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and
16 others. 2025. DAPO: An open-source LLM rein-
forcement learning system at scale.arXiv preprint
arXiv:2503.14476.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176.
Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang,
Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu,
Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda
Chen. 2025. Swift: A scalable lightweight infrastruc-
ture for fine-tuning. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 39, pages
29733–29735.
8

A Implementation Details
This section details the parameter settings, end-to-
end pipeline implementation, and training specifics
that instantiate the method described in the main
paper.
Query rewriting and retrieval.Multi-route
query rewriting produces three rewritten variants
in parallel while retaining the original user query,
yielding four queries in total for retrieval. The
GraphRAG component follows the standard Mi-
crosoft GraphRAG design (Edge et al., 2024),
with local search used for graph traversal. The
high-citation knowledge subgraph is built from
the top- N% most frequently cited items, with
N= 10 . The traditional RAG channel uses hybrid
retrieval with BGE + BM25 (run jointly); results
from both channels are merged and deduplicated,
then reranked by a lightweight Qwen3-4B reranker
(Zhang et al., 2025). Finally, to fit the model con-
text window, we truncate the reranked evidence to
8K tokens.
SFT stage.The first stage is supervised fine-
tuning with LoRA on the base model Qwen3-32B,
implemented with the SWIFT infrastructure (Zhao
et al., 2025). We use a learning rate of 1×10−4,
train for 5 epochs on 8 ×NVIDIA H20 GPUs, and
use 1k human-annotated dialogue samples.
RL stage.The second stage uses reinforcement
learning via the VERL framework (Sheng et al.,
2024) and the GRPO algorithm. Training is again
LoRA-based on 16 ×H20 GPUs, with 1k prompts
and responses labeled by Gemini 2.5 Pro (Gem-
ini Team, Google, 2025) for reward learning. The
judger used to compute rewards is Hunyuan Tur-
boS (Tencent Hunyuan Team, 2025). We train for
120 steps with batch size 16, set generation temper-
ature to 1.0, set the maximum response length to
2K tokens, and use 8 rollouts per prompt. All re-
ward terms are normalized before combination. For
reward weights, we set higher weights for safety
and hallucination-related terms. We set λ3= 2.0
for safety and λ4= 2.0 for hallucination and link
penalty, while other weights are set to 1. Following
a DAPO-style setup (Yu et al., 2025), the reference-
model KL term is removed.
Safety guardrails.During streaming generation,
safety guardrails post-process the output to detect
and filter policy violations and hallucinated URLsbefore serving, enforcing zero-hallucination and
strict safety constraints in the final response.
B Reward Computation Algorithm
Algorithm 1Multi-dimensional Reward Computa-
tion
Require: Generated answer A, retrieved evidence
D={d 1, . . . , d k}, ground truth answer Agt,
URL prefix candidate poolC p
Ensure:Total rewardR
1:Extract URLs via regex: U ←
ExtractURLs re(A)
2:Extract evidence URLs via regex: UD←
ExtractURLs re(D)
3:HTTP status set:S ← {200,301,302}
4:URLs in evidence:U evi← U ∩ U D
5:URLs not in evidence:U out← U \ U D
6:Prefix-approved URLs: Upref← {u∈ U out|
Prefix(u)∈ C p}
7:HTTP-valid URLs: Uhttp← {u∈ U pref|
code(u)∈ S}
8:Valid URLs:U valid← U evi∪ Uhttp
9:R f←f faithful (A, A gt){Pairwise comparison
with ground truth using LLM-as-judge}
10:R s←f style(A){Style evaluation using LLM-
as-judge}
11:R a←f safety(A){Safety check using LLM-as-
judge}
12:R+
h←Reward(U valid){Positive reward for
valid links}
13:R−
h←Penalty(U \ U valid){Negative penalty
for invalid links}
14:R h←R+
h−R−
h
15:R←λ 1Rf+λ 2Rs+λ 3Ra+λ 4Rh
16:returnR
C Prompt
LLM Judger Prompt
You are an expert evaluator for advertising
customer service answer quality.
Evaluate Answer B on the three
dimensions below.
- Evidence Faithfulness: compare Answer A and
Answer B; judge whether Answer B
is G, meaning better, S, meaning tie, or B,
meaning worse, than Answer A and give
a brief reason.
- Style Compliance and Safety: score Answer B
only, for example on a 0-10 scale, and
do not use G, S, or B.
Dimensions:
9

1. Evidence Faithfulness:
- How well does the answer align with the
provided materials through
pairwise comparison? Consider semantic
consistency and factual accuracy
given the user query and dialogue
history; penalize unsupported or
contradictory claims.
2. Style Compliance, score 0-10:
- Does the answer adhere to advertising
domain conventions, including
tone, professionalism, and domain-
specific formatting requirements?
- Scoring: 0-2 poor. This includes
informal style, off-tone responses, or
wrong format.
3-4 below average. This indicates
partial compliance.
5-6 acceptable. This indicates general
compliance with minor gaps.
7-8 good. This indicates professional
responses with consistent tone and
format.
9-10 excellent. This indicates full
alignment with domain conventions.
3. Safety, score 0-10:
- Does the answer avoid platform policy
violations and comply with
regulatory standards and safety
guidelines?
- Scoring: 0-2 severe violations. This
indicates policy breach or harmful or
non-compliant content.
3-4 notable issues. This indicates
multiple issues or serious
compliance gaps.
5-6 acceptable. This indicates minor or
ambiguous issues.
7-8 good. This indicates compliant
responses with isolated
imperfections.
9-10 excellent. This indicates full
compliance with no risk.
---
### Input:
[Query]: {query}
[Dialogue History]: {dialogue_history}
[Materials]: {file}
[Answer A]: {ans_a}
[Answer B]: {ans_b}
---
### Output in JSON only, example:
{
"scores": {
"Evidence Faithfulness": {"reason": "...",
"grade": "G"},"Style Compliance": 8,
"Safety": 9
}
}
D Factual QA under Distracting Context
Setting and results.We construct a factual QA
evaluation set from 1,000 knowledge items sam-
pled from the production environment. Retrieved
context is obtained from the actual recall pipeline
so that all models receive identical inputs. All mod-
els are evaluated in non-thinking mode. Figure 7
reports accuracy for our online deployed model and
leading commercial flagship models.
Ours
DeepSeek-V3.2Kimi K2.5 Doubao 1.8HY 2.06080100Accuracy (%)99.3%
93.6%95.0%92.2%96.5%
Figure 7: Accuracy on the factual QA evaluation set.
The input context includes all relevant knowledge and
distracting retrieved passages.
E Example Dialogue Comparison
Setting.This example compares responses to an
account ID query. The previous online answer con-
tains hallucinated links marked in red, while our an-
swer uses validated links. Sensitive platform names
and domains in our answer are replaced with place-
holders: [Platform Name] represents the advertis-
ing platform name, and [platform-domain.com]
represents the platform domain.
Observation.As shown in Figure 8, the base-
line answer provides generic instructions with two
hallucinated example links (https://example.com)
that do not correspond to actual platform re-
sources. In contrast, our answer delivers a struc-
tured, scenario-specific response that distinguishes
between uncertified and certified account work-
flows, includes validated platform links with opera-
tion screenshots, and provides additional guidance
for service provider and recharge account queries.
This comparison illustrates how our framework
10

eliminates hallucinated URLs while improving an-
swer completeness and practical utility.
F Service Deployment
Full-stack integration.The service is deployed
end-to-end on a large-scale advertising platform
and integrated into multiple user-facing entry
points, including an official website sidebar, a mini-
program client, and a standalone website. To avoid
disclosing sensitive details, we describe the deploy-
ment in terms of functional modules rather than
proprietary product names.
Online serving.The generator is served with
the vLLM framework (Kwon et al., 2023) on 2 ×
NVIDIA H20 GPUs. The knowledge base index
is hosted on an Elasticsearch cluster for scalable
retrieval. On top of these components, we build the
production workflow on an internal agent orches-
tration platform, which connects the link-related
model and other internal services as tools, executes
the retrieval–generation–post-check pipeline, and
exposes an external-facing streaming API with an
interface compatible with OpenAPI.
Example dialogue case.Figure 9 shows a real
dialogue from our deployed advertising QA sys-
tem. The user asks about recent restrictions on
mini-program landing pages in the financial credit
industry. The system responds with a structured,
evidence-grounded answer covering (i) qualifi-
cation and entity requirements, (ii) content and
presentation standards (e.g., risk disclaimers and
interest-rate disclosure), (iii) interaction and tech-
nical requirements (e.g., privacy, consent, and no
forced actions), and (iv) prohibited scenarios (e.g.,
unauthorized downloads and payment-information
placement), along with compliant production sug-
gestions. This case illustrates that the frame-
work can deliver policy-aligned, comprehensive
responses without hallucinated or non-compliant
content.
11

Query:Where can I find the Account ID?
Previous Asnwer
You can view the Account ID in the advertiser list of
the service provider system. The specific steps are as
follows:
1. Log in to the service provider system;
2. Click “Account List” in the workbench;
3.In the account list, you can view the names and IDs
of each account.
In addition, you can use the filtering function for account
names and IDs to perform precise or fuzzy searches.
Account names support fuzzy matching queries, and
Account IDs support exact matching queries.
For further queries on Account IDs for certified or un-
certified accounts, please refer to the detailed steps and
screenshots in the following links:
•Account ID query for certified accounts
https://example.com
•Account ID query for uncertified accounts
https://example.comOur Method
The method for viewing advertising Account IDs varies
depending on the account certification status and opera-
tion path, as detailed below:
I. Account ID Query for Uncertified Advertising Ac-
counts
1.Visit the [Platform Name] websitehttps://[platform-
domain.com], click the “Login” button in the upper
right corner to scan and log in;
2.In the account selection panel, click the “Uncer-
tified Accounts” section to view the correspond-
ing Account ID. (See operation position example
athttps://[platform-cdn.com]/img1.png)
II. Account ID Query for Certified Advertising Ac-
counts
1.Log in to the [Platform Name] website and enter the
“[Platform Name] Workbench”;
2.Click “Management” and select the “Entity/Business
Unit” where the account is located on the left;
3.Click “Account Management” to view the Ac-
count ID. (See operation position examples
athttps://[platform-cdn.com]/img2.pngand
https://[platform-cdn.com]/img3.png)
III. Query via Service Provider System
1.Log in to the service provider system and enter the
“Account List” in the workbench;
2.View account names and IDs in the account list, sup-
porting searches by account name (fuzzy matching)
or ID (exact matching).
IV . Recharge Account UID Query
To view the UID of a recharge account, please click
“Account Settings” in “Financial Management” and view
it in “Account Information”.
✗Hallucinated✓Validated
Figure 8: Comparison of dialogue responses: the previous online answer (left, red border) contains hallucinated
links marked in red, while our answer (right, green border) uses validated links with platform-specific terms replaced
by placeholders.
12

Figure 9: A real dialogue case from our deployed system: the user inquires about restrictions on mini-program
landing pages in the financial credit industry; the system responds with a structured, policy-grounded answer
covering qualifications, content standards, interaction design, prohibited scenarios, and compliant production
suggestions.
13