# RankArena: A Unified Platform for Evaluating Retrieval, Reranking and RAG with Human and LLM Feedback

**Authors**: Abdelrahman Abdallah, Mahmoud Abdalla, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali, Adam Jatowt

**Published**: 2025-08-07 15:46:53

**PDF URL**: [http://arxiv.org/pdf/2508.05512v1](http://arxiv.org/pdf/2508.05512v1)

## Abstract
Evaluating the quality of retrieval-augmented generation (RAG) and document
reranking systems remains challenging due to the lack of scalable,
user-centric, and multi-perspective evaluation tools. We introduce RankArena, a
unified platform for comparing and analysing the performance of retrieval
pipelines, rerankers, and RAG systems using structured human and LLM-based
feedback as well as for collecting such feedback. RankArena supports multiple
evaluation modes: direct reranking visualisation, blind pairwise comparisons
with human or LLM voting, supervised manual document annotation, and end-to-end
RAG answer quality assessment. It captures fine-grained relevance feedback
through both pairwise preferences and full-list annotations, along with
auxiliary metadata such as movement metrics, annotation time, and quality
ratings. The platform also integrates LLM-as-a-judge evaluation, enabling
comparison between model-generated rankings and human ground truth annotations.
All interactions are stored as structured evaluation datasets that can be used
to train rerankers, reward models, judgment agents, or retrieval strategy
selectors. Our platform is publicly available at https://rankarena.ngrok.io/,
and the Demo video is provided https://youtu.be/jIYAP4PaSSI.

## Full Text


<!-- PDF content starts -->

RankArena: A Unified Platform for Evaluating Retrieval,
Reranking and RAG with Human and LLM Feedback
Abdelrahman Abdallah
University of Innsbruck
Innsbruck, Tyrol, AustriaMahmoud Abdalla
Chungbuk National University
Cheongju-si, Cheongju, Republic of
KoreaBhawna Piryani
University of Innsbruck
Innsbruck, Tyrol, Austria
Jamshid Mozafari
University of Innsbruck
Innsbruck, Tyrol, AustriaMohammed Ali
University of Innsbruck
Innsbruck, Tyrol, AustriaAdam Jatowt
University of Innsbruck
Innsbruck, Tyrol, Austria
ABSTRACT
Evaluating the quality of retrieval-augmented generation (RAG)
and document reranking systems remains challenging due to the
lack of scalable, user-centric, and multi-perspective evaluation tools.
We introduce RankArena, a unified platform for comparing and
analysing the performance of retrieval pipelines, rerankers, and
RAG systems using structured human and LLM-based feedback as
well as for collecting such feedback. RankArena supports multiple
evaluation modes: direct reranking visualisation, blind pairwise
comparisons with human or LLM voting, supervised manual docu-
ment annotation, and end-to-end RAG answer quality assessment.
It captures fine-grained relevance feedback through both pairwise
preferences and full-list annotations, along with auxiliary metadata
such as movement metrics, annotation time, and quality ratings.
The platform also integrates LLM-as-a-judge evaluation, enabling
comparison between model-generated rankings and human ground
truth annotations. All interactions are stored as structured evalu-
ation datasets that can be used to train rerankers, reward models,
judgment agents, or retrieval strategy selectors. Our platform is
publicly available at https://rankarena.ngrok.io/, and the Demo
video is provided1.
CCS CONCEPTS
•Information systems →Language models ;Top-k retrieval
in databases .
KEYWORDS
Ranking, RAG, LLM-as-a-Judge, Retriever
ACM Reference Format:
Abdelrahman Abdallah, Mahmoud Abdalla, Bhawna Piryani, Jamshid Moza-
fari, Mohammed Ali, and Adam Jatowt. 2025. RankArena: A Unified Platform
1https://youtu.be/jIYAP4PaSSI
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-2040-6/2025/11. . . $15.00
https://doi.org/10.1145/XXXXXXX.XXXXXXXfor Evaluating Retrieval, Reranking and RAG with Human and LLM Feed-
back . In Proceedings of the 34th ACM International Conference on Information
and Knowledge Management (CIKM ’25), November 10–14, 2025, Seoul, Re-
public of Korea. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/
XXXXXXX.XXXXXXX
1 INTRODUCTION
Recent advancements in large language models (LLMs) and retrieval-
augmented generation (RAG) [ 1,3,5,7,19,28,33] systems have
significantly expanded the scope of natural language understanding
and generation. These systems combine information retrieval with
powerful generative models, enabling applications such as open-
domain question answering [ 20,22,30,44,45], summarization [ 17],
and chat assistants [ 36]. However, as models and pipelines become
more complex, evaluating their performance in alignment with
human preferences remains a persistent challenge. Conventional
benchmarks [ 42,43] often relying on static datasets and predefined
metrics fail to capture user-centric notions of quality such as docu-
ment relevance ,answer usefulness , orfaithfulness in generation . To
address this gap, we introduce RankArena , a unified, open-source
evaluation and annotation platform that supports multi-faceted
assessment of retrieval ,reranking , and generation quality . Inspired
by recent efforts such as MT-Bench [ 9] and Chatbot Arena [ 15], our
platform extends the evaluation paradigm beyond model-centric
metrics by integrating human-in-the-loop andLLM-as-a-judge feed-
back [ 21,47]. Users interact with multiple reranking and RAG
pipelines across various tasks, cast preferences in blind pairwise
comparisons, annotate ranked document lists, and evaluate gener-
ated answers, all of which are structured into reusable datasets.
RankArena features five complementary evaluation modes:
(1)Reranker Comparison: Direct and blind pairwise battles
between ranked document lists.
(2)Manual Annotation: Supervised full-list ranking with qual-
ity labels, tracking annotation effort and movement metrics.
(3)LLM Judgment: GPT-based voting on document orderings
or RAG answers.
(4)Comprehensive Reranker Leaderboard: Evaluation and
comparison of multiple rerankers across diverse retrieval
tasks, using shared datasets and unified metrics.
(5)RAG Output Evaluation: Qualitative and preference-based
evaluation of generated answers from end-to-end RAG pipelines.arXiv:2508.05512v1  [cs.IR]  7 Aug 2025

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Abdallah et al.
1 2 3 4 5 6 7
8
9RAG Arena:  Evaluate 
retrieval -augmented 
generation (RAG) 
pipelines.
   Direct Reranker : Inspect 
and assess single reranker  
outputs.
   1v1 Reranker : Conduct 
pairwise battles between 
two rerankers .
   Retriever + Reranker : 
Benchmark retriever -
reranker  pipelines jointly.
   Anonymous Reranker : 
Blind comparisons without 
revealing reranker  identity.
   BEIR Dataset 
Evaluation:  Evaluate 
models on standard BEIR 
benchmarks.
   Document Annotation:  
Perform detailed manual 
document annotations.
   Arena Leaderboard:  
View aggregated 
performance rankings and 
statistics.
    System Information:  
Access details about 
system status and setup.1
2
3
4
5
6
7
8
9R
R
Figure 1: Screenshot of the RankArena interface with added explanations of each tab (1-9) showcasing the system’s main
evaluation modes and tools.
RankArena
Comprehensive Reranking 
Evaluation Platform
Information
System Guide
Leaderboard
Rankings
Annotation
Manual Labeling
RAG
End -to-End RAG
Direct Reranker
Single Method Test
1v1 Arena
Head -to-Head
Retriever
Full Pipeline
Anonymous
Blind Evaluation
BEIR Eval
BenchmarksLLM Judge
Voting
Figure 2: The overview of RankArena highlighting its main
evaluation modules.
(6)Dataset Collection for Reward and Judge Model Train-
ing: Systematically collect aligned human and LLM prefer-
ence data, creating reusable datasets to train reward models
for retrieval and to build or fine-tune LLM judges capable of
comparing retrieval or RAG systems effectively.
Unlike previous works that focus solely on chatbot alignment or
answer preference [ 15,29], RankArena supports retrieval-specific
evaluation signals , making it uniquely suited to train and bench-
mark rerankers, reward models, and retrieval agents. Our platform
captures diverse signals, including user preferences, annotation
metadata, rank movements, generation ratings, and LLM feedback,
all stored in a structured, extensible dataset format. These datasets
can be used for training new models , analyzing alignment gaps be-
tween humans and LLMs, or developing retrieval strategies adaptive
to user feedback .2 RANKARENA FUNCTIONALITY
RankArena fills this gap by providing a unified platform for collect-
ing pairwise preferences, listwise annotations, and LLM-as-a-judge
feedback at scale. This system is important because it enables: (i) cre-
ation of reusable datasets for training supervised rerankers and re-
ward models; (ii) study of human-LLM agreement in ranking tasks;
(iii) adaptive benchmarking that evolves with real user queries and
feedback; and (iv) reproducible, open evaluations combining human
and automated signals.
The overview of our system is illustrated in Figure 1 and 2,
which highlights the various modules working in concert to support
rich evaluation scenarios. Our platform facilitates blind head-to-
head comparisons, direct inspection of retrieval outputs, large-scale
annotation, and statistical aggregation of pairwise preferences into
global model rankings.
Arena Battles and Preference Collection: At the heart of RankArena
lies the 1v1 Arena , a head-to-head evaluation interface where two
rerankers, retrieval systems, or RAG pipelines are presented in a
blind and randomized manner. Either human or LLM-based judges
interact with the system by comparing the outputs of these two
systems on the same query and voting for the preferred result. This
design simplifies the cognitive load on evaluators compared to ab-
solute scoring, as it only requires relative preference judgments. We
support over 24 reranking methods [ 2,4,11,14,16,18,24,26,27,
31,34,35,37,41,41,48] spanning pairwise, listwise, and pointwise
approaches, with a total of 84 models available for evaluation in
these head-to-head battles.
LLM-as-a-Judge Evaluation: To complement human annotations,
RankArena integrates an LLM judge module, where large language
models (e.g., GPT-4) provide automated pairwise preferences be-
tween retrieval or RAG outputs. The LLM-as-a-judge mechanism
enables rapid scaling of evaluations and offers a cost-effective alter-
native to human voting. We adopt a structured prompting strategy

RankArena: A Unified Platform for Evaluating Retrieval, Reranking and RAG with Human and LLM Feedback CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
that asks LLMs to consider relevance, ranking order quality, and
overall usefulness in answering the query. The prompt clearly in-
structs the LLM to review two reranked document lists for a given
query and respond strictly within a predefined format, specifying
the winning reranker and providing concise reasoning. This ap-
proach ensures consistency, interpretability, and comparability of
LLM judgments across different evaluation tasks.
End-to-End RAG:. RankArena also supports the evaluation of
full RAG pipelines , where annotators or LLM judges assess gener-
ated answers along with their supporting document lists. This inte-
grated evaluation allows for measuring not only document ranking
quality but also how well retrieved evidence supports final answer
generation. Such evaluation is essential for understanding system
performance in complex, multi-stage pipelines. We support five dif-
ferent retrievers (DPR [ 25], Colbert [ 26], Contreiver [ 23], BGE [ 12],
BM25 [ 38]) operating over two indexing corpora— Wikipedia and
MS MARCO as well as online retrievers for live document retrieval,
providing flexibility across static and dynamic knowledge sources.
For RAG evaluations, we extend our LLM-as-a-Judge module to
assess both the generated answers and their associated supporting
documents from two different rerankers. The LLM judge considers
not only the relevance and order of retrieved contexts but also the
quality of the final answers and their alignment with the retrieved
evidence. This method enables scalable, automated assessment of
how effectively different rerankers support answer generation in
RAG systems. We adopt a structured prompt for RAG evaluations
that asks the LLM judge to compare two RAG outputs each con-
sisting of a document list, a generated answer, and an indication of
which document the answer was primarily drawn from. The LLM
is instructed to evaluate results based on: (1) Relevance of retrieved
documents, (2) Quality of document ranking, (3) Usefulness and
correctness of the generated answers, and (4) Faithfulness of the
answers to the supporting documents. This structured prompting
is illustrated in Figure 3, enabling consistent and interpretable LLM
voting for end-to-end RAG pipelines.
Direct Reranker and Full-List Annotation: Beyond head-to-head
comparisons, RankArena supports two complementary modes for
evaluating single reranker outputs. In the direct reranker mode,
users provide both the query and the candidate documents, and the
system applies a single reranker to produce a ranked list. This mode
allows users to directly examine how a specific reranker orders
documents for a given query. In contrast, the full-list annotation
mode involves users providing a query (or selecting one), after
which the system retrieves documents from online or offline cor-
pora. The user then manually reorders or assigns relevance grades
to these documents, creating high-quality listwise supervision data.
This annotated data can be used to train supervised rerankers, list-
wise ranking models, or reward models for preference learning.
Leaderboard Aggregation and Statistical Modeling: The prefer-
ences collected from head-to-head battles, direct evaluations, and
LLM judges are aggregated into a comprehensive reranker leader-
board . We compute win rates 𝑤𝑖=wins 𝑖
total votes 𝑖and transform these
into an ELO-style rating:
𝑅𝑖=1200+32·(𝑤𝑖−0.5)·min(log(total votes 𝑖+1),5.0)LLM Judge RAG Prompt Template
System Prompt: You are an expert evaluator for RAG
systems and document reranking.
Instruction: Given a query and two RAG outputs, de-
termine which system provides the better overall result.
Evaluate both:
•The relevance and order of retrieved documents
•The usefulness and correctness of the generated
answer
•How well the answer aligns with the supporting
document
Query: "example query here"
=== Model A Results ===
Documents:
1. Document text A1 [...]
2. Document text A2 [...]
Answer: Generated answer A [...]
Source Document: Document A1
=== Model B Results ===
Documents:
1. Document text B1 [...]
2. Document text B2 [...]
Answer: Generated answer B [...]
Source Document: Document B2
Respond with ONLY ONE of these options:
•Model A if the first system is better
•Model B if the second system is better
•Tie if both are equally good
Then provide a brief explanation (2-3 sentences) of your
reasoning.
Response Format:
•WINNER: [Model A / Model B / Tie]
•REASONING: [Your explanation]
Figure 3: Illustration of our LLM Judge RAG prompt template.
The LLM evaluates RAG results by comparing retrieved doc-
uments, generated answers, and evidence alignment.
This formulation reflects both win rate and the confidence in that
estimate (via number of votes). Traditional benchmark scores, such
as BEIR averages, are calculated as: BEIR Avg𝑖=1
𝑁Í𝑁
𝑗=1𝑆𝑖𝑗where
𝑆𝑖𝑗is the score on benchmark 𝑗. Models are ranked by descend-
ing ELO rating, with BEIR performance as a secondary criterion.
Our leaderboard thus integrates user and LLM preference signals
with standardized benchmarks, offering a holistic view of model
performance.
3 RESULTS AND ANALYSIS
Experiment Setup. Our experiments were conducted using the
RankArena platform, developed on top of the Rankify [6], Gra-
dio [ 8] and PyTorch [32] framework. The system is deployed on
a server equipped with two NVIDIA A40 GPUs and 250 GB of
RAM, providing sufficient resources to support continuous 24-hour
operation for large-scale evaluation and annotation.
RankArena integrates diverse reranking and retrieval compo-
nents for comprehensive benchmarking. The system currently in-
cludes: (1) 24 reranking methods spanning pointwise, pairwise,
and listwise approaches, (2) 84 reranker models (as some methods
apply different models) evaluated across different retrieval tasks,
and (3) 5 retrievers operating on two corpora: Wikipedia and
MS MARCO , along with online retrieval for dynamic evaluation.
Below, we report the results and analyses based on the following
Arena Statistics : (1)Total User Votes: 102. (2) Active Models: 36
(models that have received human votes). (3) Benchmark Models:
80 (models with BEIR benchmark scores). (4) Methods Tested: 25.
(5)Average User-LLM Agreement: 74.2%
BEIR Benchmark Correlations. We analyze the correlation be-
tween different BEIR benchmark [ 42] datasets to understand how
model performance generalizes across tasks. Figure 4 (available also
in the leaderboard) illustrates the correlation matrix between BEIR
Average and individual benchmarks (DL19, DL20, Covid, NFCorpus,
Touche, DBPedia, SciFact).

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Abdallah et al.
BEIR_Avg DL19 DL20 Covid NFCorpus T ouche DBPedia SciFactBEIR_Avg DL19 DL20 Covid NFCorpus T ouche DBPedia SciFact1.00 0.75 0.70 0.83 0.27 0.79 0.84 0.62
0.75 1.00 0.90 0.61 0.16 0.44 0.69 0.56
0.70 0.90 1.00 0.58 0.10 0.47 0.66 0.63
0.83 0.61 0.58 1.00 0.24 0.52 0.89 0.75
0.27 0.16 0.10 0.24 1.00 -0.01 0.20 0.24
0.79 0.44 0.47 0.52 -0.01 1.00 0.63 0.35
0.84 0.69 0.66 0.89 0.20 0.63 1.00 0.76
0.62 0.56 0.63 0.75 0.24 0.35 0.76 1.00Correlation between BEIR Benchmarks
0.00.20.40.60.81.0
Figure 4: Correlation matrix between BEIR Average and in-
dividual benchmarks.
52.8
51.5
51.5
50.4
50.3
50.2
49.6
49.5
48.3
47.9
47.2
47.2
46.9
46.8
46.2
44.2
43.7
43.1
39.5
35.9
34.4
33.5
26.2
twolarrankgpt-apilistt5rankt5inrankerzephyr rerankermonot5apirankertransformer rankervicuna rerankerlit5distmonobertsplade rerankercolbert rankerPRPrankgptlit5scoreSentence transformer rerankeruprechorankflashrankPromptagator++incontext reranker01020304050Method Performance
MethodsAverage BEIR Score
Figure 5: Average BEIR performance of different reranking
methods.
T he BEIR Average shows strong correlations with DBPedia
(0.84), Covid (0.83), and DL19 (0.75), indicating these datasets con-
tribute substantially to the aggregate BEIR metric. In contrast, NF-
Corpus displays weak correlation (0.27) with BEIR Average and
other benchmarks, suggesting it measures distinct capabilities. Such
correlation patterns provide insight into the diversity of tasks
within BEIR and the challenges of designing universal rankers.
Models Performance on BEIR. Figure 5 (also displayed in the
leaderboard tab) presents the average BEIR scores across evaluated
methods. The highest-performing method, twolar [10], achieves an
average score of 52.8, while other top methods (e.g., rankgpt-api [40],
listt5 [46]) exhibit scores above 50. Methods like UPR[39] and
incontext reranker [13] perform considerably worse, reflecting
a gap between the strongest and weakest reranking strategies.
Human-LLM Agreement. We assess next the alignment between
human votes and LLM judge decisions. Figure 6 shows a scatter plot
of human preference rates versus LLM judge rates across the evalu-
ated models. The results demonstrate a generally strong positive
correlation between human and LLM judgments, with most models
t5-smallmonot5-base-msmarco-10k
t5-basemonot5-3b-msmarco-10k
monot5-large-msmarco-10k
llamav3.1-8b inranker-small
inranker-base listt5-3bmonot5-large-msmarco
0 20 40 60 80 100020406080100
Perfect agreement
020406080100Agreement (%)Human vs. LLM judge agreement
Human win rate (%)LLM win rate (%)Agreement rate: 53.5%
Models: 10Figure 6: Scatter plot of human-LLM agreement across mod-
els.
0 5 10 15 20 25 30 35
T otal User VotesPRPPromptagator++Sentence_transformer_rerankerapirankercolbert_rankerflashranklit5distincontext_rerankerlit5scorevicuna_rerankersplade_rerankerrankgpt-apizephyr_rerankerechorankmonoberttwolarsentence_transformer_rerankerblender_rerankerrankt5listt5inrankerrankgpttransformer_rankermonot5uprMethod
000000000000011112447882738Method Popularity
Figure 7: Distribution of user votes across different reranking
methods in RankArena. The figure shows that a few methods
receive
clustering near the diagonal line. Notable deviations are observed
for certain models, indicating areas where human preferences and
automated judgments diverge, warranting deeper qualitative anal-
ysis.
Method Popularity and User Engagement. Figure 7 (shown also
under the leaderboard tab) illustrates the popularity of different
reranking methods in RankArena, measured by the total number
of user votes received during head-to-head evaluations. Our data
shows a clear skew in user engagement across methods: a small
number of rerankers dominate in terms of participation, while many
others receive few or no votes.
4 CONCLUSION
We presented RankArena, a unified platform for evaluating re-
trieval, reranking, and RAG systems using human and LLM feed-
back. By combining pairwise preferences, full-list annotations, and
automated judgments, RankArena enables the creation of reusable
datasets and open benchmarking of retrieval pipelines.

RankArena: A Unified Platform for Evaluating Retrieval, Reranking and RAG with Human and LLM Feedback CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
5 GENAI USAGE DISCLOSURE
We used OpenAI’s ChatGPT for minor language editing, specifically
to rephrase sentences and correct grammatical errors.
REFERENCES
[1]Abdelrahman Abdallah and Adam Jatowt. 2023. Generator-retriever-generator:
A novel approach to open-domain question answering. arXiv preprint
arXiv:2307.11278 (2023).
[2]Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Mohammed M Ab-
delgwad, and Adam Jatowt. 2024. DynRank: Improving Passage Retrieval with
Dynamic Zero-Shot Prompting Based on Question Classification. arXiv preprint
arXiv:2412.00600 (2024).
[3]Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Mohammed Ali,
and Adam Jatowt. 2025. From Retrieval to Generation: Comparing Different
Approaches. arXiv preprint arXiv:2502.20245 (2025).
[4]Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, and Adam Jatowt.
2025. Asrank: Zero-shot re-ranking with answer scent for document retrieval.
arXiv preprint arXiv:2501.15245 (2025).
[5]Abdelrahman Abdallah, Bhawna Piryani, and Adam Jatowt. 2023. Exploring the
state of the art in legal QA systems. Journal of Big Data 10, 1 (2023), 127.
[6]Abdelrahman Abdallah, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali,
and Adam Jatowt. 2025. Rankify: A comprehensive python toolkit for retrieval,
re-ranking, and retrieval-augmented generation. arXiv preprint arXiv:2502.02464
(2025).
[7]Abdelrahman Abdallah, Bhawna Piryani, Jonas Wallat, Avishek Anand, and Adam
Jatowt. 2025. Tempretriever: Fusion-based temporal dense passage retrieval for
time-sensitive questions. arXiv preprint arXiv:2502.21024 (2025).
[8]Abubakar Abid, Ali Abdalla, Ali Abid, Dawood Khan, Abdulrahman Alfozan, and
James Zou. 2019. Gradio: Hassle-free sharing and testing of ml models in the
wild. arXiv preprint arXiv:1906.02569 (2019).
[9]Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jiaheng Liu, Zhanhui Zhou, Zhuoran
Lin, Wenbo Su, Tiezheng Ge, Bo Zheng, et al .2024. Mt-bench-101: A fine-grained
benchmark for evaluating large language models in multi-turn dialogues. arXiv
preprint arXiv:2402.14762 (2024).
[10] Davide Baldelli, Junfeng Jiang, Akiko Aizawa, and Paolo Torroni. 2024. TWOLAR:
a TWO-step LLM-Augmented distillation method for passage Reranking. In
European Conference on Information Retrieval . Springer, 470–485.
[11] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau,
Nicolas Chapados, and Siva Reddy. 2024. LLM2Vec: Large Language Models Are
Secretly Powerful Text Encoders. arXiv:2404.05961 [cs.CL] https://arxiv.org/abs/
2404.05961
[12] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.
Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text
embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216
(2024).
[13] Shijie Chen, Bernal Jiménez Gutiérrez, and Yu Su. 2024. Attention in Large
Language Models Yields Efficient Zero-Shot Re-Rankers. arXiv preprint
arXiv:2410.02642 (2024).
[14] Zijian Chen, Ronak Pradeep, and Jimmy Lin. 2024. An Early FIRST Reproduc-
tion and Improvements to Single-Token Decoding for Fast Listwise Reranking.
arXiv:2411.05508 [cs.IR] https://arxiv.org/abs/2411.05508
[15] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos,
Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E
Gonzalez, et al .2024. Chatbot arena: An open platform for evaluating llms by
human preference. In Forty-first International Conference on Machine Learning .
[16] Prithiviraj Damodaran. 2023. FlashRank, Lightest and Fastest 2nd Stage Reranker
for search pipelines. https://doi.org/10.5281/zenodo.10426927
[17] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization. arXiv preprint arXiv:2404.16130 (2024).
[18] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant.
2022. From distillation to hard negative sampling: Making sparse neural ir models
more effective. In Proceedings of the 45th international ACM SIGIR conference on
research and development in information retrieval . 2353–2359.
[19] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey. arXiv preprint arXiv:2312.10997
2, 1 (2023).
[20] Raphael Gruber, Abdelrahman Abdallah, Michael Färber, and Adam Jatowt. 2024.
Complextempqa: A large-scale dataset for complex temporal question answering.
arXiv preprint arXiv:2406.04866 (2024).
[21] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu,
Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al .2024. A survey on
llm-as-a-judge. arXiv preprint arXiv:2411.15594 (2024).[22] Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan Wang, Lan Liu,
William Yang Wang, Bonan Min, and Vittorio Castelli. 2024. Rag-qa arena: Evalu-
ating domain robustness for long-form retrieval augmented question answering.
arXiv preprint arXiv:2407.13998 (2024).
[23] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised Dense Infor-
mation Retrieval with Contrastive Learning. https://doi.org/10.48550/ARXIV.
2112.09118
[24] Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. 2023. LLM-Blender: Ensem-
bling Large Language Models with Pairwise Ranking and Generative Fusion.
arXiv:2306.02561 [cs.CL] https://arxiv.org/abs/2306.02561
[25] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu,
Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for
Open-Domain Question Answering.. In EMNLP (1) . 6769–6781.
[26] Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage
search via contextualized late interaction over bert. In Proceedings of the 43rd
International ACM SIGIR conference on research and development in Information
Retrieval . 39–48.
[27] Thiago Laitz, Konstantinos Papakostas, Roberto Lotufo, and Rodrigo Nogueira.
2024. InRanker: Distilled Rankers for Zero-shot Information Retrieval.
arXiv:2401.06910 [cs.IR] https://arxiv.org/abs/2401.06910
[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems 33 (2020), 9459–9474.
[29] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Qingwei Lin, Jian-Guang Lou,
Shifeng Chen, Yansong Tang, and Weizhu Chen. 2024. Wizardarena: Post-training
large language models via simulated offline chatbot arena. Advances in Neural
Information Processing Systems 37 (2024), 111544–111570.
[30] Jamshid Mozafari, Abdelrahman Abdallah, Bhawna Piryani, and Adam Jatowt.
2024. Exploring hint generation approaches in open-domain question answering.
arXiv preprint arXiv:2409.16096 (2024).
[31] Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019. Multi-Stage
Document Ranking with BERT. arXiv:1910.14424 [cs.IR] https://arxiv.org/abs/
1910.14424
[32] A Paszke. 2019. Pytorch: An imperative style, high-performance deep learning
library. arXiv preprint arXiv:1912.01703 (2019).
[33] Bhawna Piryani, Abdelrahman Abdallah, Jamshid Mozafari, and Adam Jatowt.
2024. Detecting temporal ambiguity in questions. arXiv preprint arXiv:2409.17046
(2024).
[34] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. RankVicuna:
Zero-Shot Listwise Document Reranking with Open-Source Large Language
Models. arXiv:2309.15088 [cs.IR] https://arxiv.org/abs/2309.15088
[35] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023.
RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a
Breeze! arXiv:2312.02724 [cs.IR] https://arxiv.org/abs/2312.02724
[36] Mahimai Raja, E Yuvaraajan, et al .2024. A rag-based medical assistant especially
for infectious diseases. In 2024 International Conference on Inventive Computation
Technologies (ICICT) . IEEE, 1128–1133.
[37] Muhammad Shihab Rashid, Jannat Ara Meem, Yue Dong, and Vagelis Hristidis.
2024. EcoRank: Budget-Constrained Text Re-ranking Using Large Language
Models. arXiv:2402.10866 [cs.CL] https://arxiv.org/abs/2402.10866
[38] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond. Foundations and Trends ®in Information Retrieval
3, 4 (2009), 333–389.
[39] Devendra Singh Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau
Yih, Joelle Pineau, and Luke Zettlemoyer. 2022. Improving Passage Retrieval with
Zero-Shot Question Generation. (2022). https://arxiv.org/abs/2204.07496
[40] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin
Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT good at search? investigat-
ing large language models as re-ranking agents. arXiv preprint arXiv:2304.09542
(2023).
[41] Manveer Singh Tamber, Ronak Pradeep, and Jimmy Lin. 2023. Scaling Down, LiT-
ting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder
Models. arXiv:2312.16098 [cs.IR] https://arxiv.org/abs/2312.16098
[42] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of
information retrieval models. arXiv preprint arXiv:2104.08663 (2021).
[43] Ellen M Voorhees. 2001. The TREC question answering track. Natural Language
Engineering 7, 4 (2001), 361–378.
[44] Jonas Wallat, Abdelrahman Abdallah, Adam Jatowt, and Avishek Anand.
2025. A study into investigating temporal robustness of llms. arXiv preprint
arXiv:2503.17073 (2025).
[45] Yi Yang, Wen-tau Yih, and Christopher Meek. 2015. Wikiqa: A challenge dataset
for open-domain question answering. In Proceedings of the 2015 conference on
empirical methods in natural language processing . 2013–2018.
[46] Soyoung Yoon, Eunbi Choi, Jiyeon Kim, Hyeongu Yun, Yireun Kim, and Seung-
won Hwang. 2024. Listt5: Listwise reranking with fusion-in-decoder improves

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Abdallah et al.
zero-shot retrieval. arXiv preprint arXiv:2402.15838 (2024).
[47] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al .2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information
Processing Systems 36 (2023), 46595–46623.[48] Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai Hui, Ji Ma, Jing Lu, Jianmo Ni,
Xuanhui Wang, and Michael Bendersky. 2022. RankT5: Fine-Tuning T5 for Text
Ranking with Ranking Losses. arXiv:2210.10634 [cs.IR] https://arxiv.org/abs/
2210.10634