# DaPT: A Dual-Path Framework for Multilingual Multi-hop Question Answering

**Authors**: Yilin Wang, Yuchun Fan, Jiaoyang Li, Ziming Zhu, Yongyu Mu, Qiaozhi He, Tong Xiao, Jingbo Zhu

**Published**: 2026-03-19 16:23:04

**PDF URL**: [https://arxiv.org/pdf/2603.19097v1](https://arxiv.org/pdf/2603.19097v1)

## Abstract
Retrieval-augmented generation (RAG) systems have made significant progress in solving complex multi-hop question answering (QA) tasks in the English scenario. However, RAG systems inevitably face the application scenario of retrieving across multilingual corpora and queries, leaving several open challenges. The first one involves the absence of benchmarks that assess RAG systems' capabilities under the multilingual multi-hop (MM-hop) QA setting. The second centers on the overreliance on LLMs' strong semantic understanding in English, which diminishes effectiveness in multilingual scenarios. To address these challenges, we first construct multilingual multi-hop QA benchmarks by translating English-only benchmarks into five languages, and then we propose DaPT, a novel multilingual RAG framework. DaPT generates sub-question graphs in parallel for both the source-language query and its English translation counterpart, then merges them before employing a bilingual retrieval-and-answer strategy to sequentially solve sub-questions. Our experimental results demonstrate that advanced RAG systems suffer from a significant performance imbalance in multilingual scenarios. Furthermore, our proposed method consistently yields more accurate and concise answers compared to the baselines, significantly enhancing RAG performance on this task. For instance, on the most challenging MuSiQue benchmark, DaPT achieves a relative improvement of 18.3\% in average EM score over the strongest baseline.

## Full Text


<!-- PDF content starts -->

DAPT: A DUAL-PATH FRAMEWORK FOR
MULTILINGUAL MULTI-HOP QUESTION ANSWERING
Yilin Wang1, Yuchun Fan1, Jiaoyang Li1, Ziming Zhu1, Yongyu Mu1,
Qiaozhi He, Tong Xiao1,2†, Jingbo Zhu1,2
1School of Computer Science and Engineering, Northeastern University, Shenyang, China
2NiuTrans Research, Shenyang, China
ABSTRACT
Retrieval-augmented generation (RAG) systems have made signif-
icant progress in solving complex multi-hop question answering
(QA) tasks in the English scenario. However, RAG systems in-
evitably face the application scenario of retrieving across multilin-
gual corpora and queries, leaving several open challenges. The first
one involves the absence of benchmarks that assess RAG systems’
capabilities under the multilingual multi-hop (MM-hop) QA setting.
The second centers on the overreliance on LLMs’ strong semantic
understanding in English, which diminishes effectiveness in mul-
tilingual scenarios. To address these challenges, we first construct
multilingual multi-hop QA benchmarks by translating English-only
benchmarks into five languages, and then we propose DaPT, a novel
multilingual RAG framework. DaPT generates sub-question graphs
in parallel for both the source-language query and its English trans-
lation counterpart, then merges them before employing a bilingual
retrieval-and-answer strategy to sequentially solve sub-questions.
Our experimental results demonstrate that advanced RAG systems
suffer from a significant performance imbalance in multilingual sce-
narios. Furthermore, our proposed method consistently yields more
accurate and concise answers compared to the baselines, signifi-
cantly enhancing RAG performance on this task. For instance, on
the most challenging MuSiQue benchmark, DAPT achieves a rela-
tive improvement of 18.3% in average EM score over the strongest
baseline.1
Index Terms—Multilingual, RAG, Multi-hop QA, Sub-question
Decomposition
1. INTRODUCTION
Retrieval-Augmented Generation (RAG) has emerged as a key tech-
nique for large language models (LLMs) [1–3] to solve knowledge-
intensive tasks. It introduces retrieval into the inference process, en-
abling models to refer to the provided relevant context and then gen-
erate the final answer, thus reducing hallucination [4–7]. Although
the RAG paradigm performs effectively for simple, fact-based ques-
tions, it still struggles with the multi-hop question answering (QA)
task. In this scenario, the clues remain in several documents, and
models need to integrate the clues from the retrieved documents to
form a conclusion. Related works along this line of research involve
two aspects. One stream of research is structure-free optimizations
of RAG systems. A prominent approach is to employ iterative re-
trieval to progressively refine the supporting clues found across mul-
†Corresponding author.
1The code and data will be available at https://github.com/f6ster/DaPT.tiple documents [8, 9]. Another stream of research leverages struc-
tured knowledge (e.g., trees or graphs) to either enhance the retrieval
of documents dense with clues or to serve as a structural guide for
the LLM to pinpoint the clues within the retrieved context [10–14].
Notably, these advances have been demonstrated almost exclu-
sively in English-centric scenarios. This presents a critical gap, as
real-world applications must handle the complexities of multilin-
gual queries and documents. This gap can be attributed to two main
aspects: (i) a scarcity of multilingual multi-hop (MM-hop) bench-
marks has hindered systematic evaluation, while naive ’translate-
then-retrieve’ strategies introduce noise and lose source-language
context; and (ii) more fundamentally, the reasoning mechanisms of
current methods are deeply reliant on the LLM’s robust English com-
prehension, causing performance to degrade in cross-lingual scenar-
ios [15–20].
As for aspect (i), we first constructed three new MM-hop QA
benchmarks by using GPT-4o-mini to translate three mainstream QA
datasets—HotpotQA [21], 2WikiMultiHopQA [22], and Musique
[23]—into five diverse languages: Swahili (Sw), Thai (Th), German
(De), Spanish (Es), and Chinese (Zh). To address the issue of (ii) we
seek to develop a method that can leverage the robust English com-
prehension of LLMs without losing crucial source-language context.
To this end, we propose aDual-Path (DaPT) Framework for MM-
hop QA. Our method DaPT, inspired by recent studies on enhancing
model performance with cross-lingual context [24, 25], first decom-
poses both the source query and its English translation counterpart
into parallel sub-question graphs. Then DaPT aligns and merges
these graphs based on the semantic similarity of each node, yield-
ing a unified, bilingual reasoning structure. This fused graph then
enables a bilingual retrieval strategy, allowing the model to solve
sub-questions sequentially by drawing evidence from a richer, cross-
lingual context pool.
Extensive experimental results across three benchmarks show
that DaPT achieves superior performance in multilingual multi-hop
QA. For instance, DaPT raises the average EM score by 3.2 points
over the HippoRAG2 on the HotpotQA benchmark. Besides, DaPT
facilitates answering questions in low-resource languages, with a 4.3
F1-point improvement over HippoRAG2 in Thai on the challeng-
ing MuSiQue dataset. Moreover, DaPT demonstrates significant im-
provements in the Exact Match (EM) score across all benchmarks,
suggesting that DaPT generates more precise and concise answers.
2. METHODOLOGY
In this section, we aim to elaborate on our methodology. As shown in
Fig. 1, we consider this a three-stage process: (1) Parallel Planning:
This stage involves translating the original query into English andarXiv:2603.19097v1  [cs.CL]  19 Mar 2026

Original Query English QueryTranslation
Merge Similar Sub-Questions
SortTopological
Decomposition Decomposition
Retrieval
&
Answer
Retrieval
&
AnswerRetrieval
&
Answer
Original Query
 Final Answer
Bilingual Retrieval and Answering
Re-inference
Sub Questions Graph 
Similar Sub-Questions
 Bilingual Sub-Questions Unsimilar Sub-Questions
 Judge
Step1 Answer
 Step4 Answer Step3 Answer
Fig. 1: Our proposed framework for multilingual multi-hop QA named DaPT. The framework consists of three main stages: (i) Parallel
Planning: Translating the source query into English and decomposing both into two parallel sub-question graphs. (ii) Node fusion: Fusing the
two graphs into a single bilingual reasoning graph based on the embedding similarity of their node of sub-questions. (iii) Bilingual Retrieval
and Answering: Sequentially solving each node on the fused graph by gathering evidence with a bilingual retrieval strategy.
decomposing both versions into sub-question graphs, where each
node is a single-hop question and the edges represent the dependen-
cies between them; (2) Node Fusion: Fusing the two sub-question
graphs into a single, bilingual reasoning structure by aligning nodes
with high embedding similarity; and (3) Bilingual Retrieval and An-
swering: The nodes of the fused graph are solved sequentially. For
each bilingual node, a dual-path retrieval strategy gathers context
from the corpus for the LLM to generate a sub-answer. Finally, the
complete reasoning path, consisting of all solved sub-questions and
their answers, is synthesized to produce the final response.
2.1. Parallel Planning
Given an original queryq l, wherel∈ {sw,zh,de,th,es}, DaPT
first translatesq linto its English counterpartq en. Next, an LLM de-
composes each question into an interdependent sub-question graph,
yielding two separate graphs:G landG en, and ensures that the
resulting sub-question graphs are acyclic without circular depen-
dencies. Formally, each graphG= (V, E)consists of node set
V={v 1, . . . , v n}representing the sub-questions. A directed edge
(vi, vj)∈Eindicates that the answer to sub-questionv iis required
to solvev j. This decomposition process can be formalized as:
GL(VL, EL) = Decompose
f(qL)
L∈ {l,en},(1)
wheref(·)is the prompting function that prepares the input question
qLfor the LLM.
As shown in Fig 1, the original query and its English counterpart
are decomposed into two separate sub-question graphs, consisting of
three and four sub-questions, respectively.2.2. Node Fusion
Inspired by the concept of critical paths, we identify pairs of seman-
tically similar sub-questions across the two graphs as pivotal nodes.
The rationale is that while the decomposition process may vary be-
tween languages, a sub-question appearing in both graphs likely rep-
resents a robust and indispensable hub. Correctly answering these
pivotal nodes is therefore critical for the success of the entire rea-
soning process.
In two sub-question graphsG landG en, we identify the pair of
nodes(v∗
l, v∗
en)which has the highest embedding similarity across
all possible pairs. This selection is formalized as:
(v∗
l, v∗
en) = arg max
(vl,ven)∈Vl×Vensim(ϕ(v l), ϕ(v en)),(2)
where sim(·,·)is the cosine similarity function.
This pair is only considered for fusion if its similarity score ex-
ceeds a predefined thresholdτ. If no such pair exists, the process
terminates.
After identifying the valid pair(v∗
l, v∗
en), the source-language
nodev∗
lis updated to become a bilingual node, incorporating the
sub-question from its English counterpartv∗
en. All incoming and
outgoing edges originally connected tov∗
enare then redirected to
the updated nodev∗
l. Finally, the redundant English nodev∗
enis
removed from the graph. This process repeats until no more pairs
above the similarity threshold can be found, resulting in the final
fused graphG f.
The last step of this stage is to establish a valid execution se-
quence for solving the sub-questions. To achieve this, we perform

Table 1: EM and F1 scores across six languages on the three multilingual multi-hop QA datasets we constructed. Avg. denotes the average
score across languages.Boldvalues indicate the best performance within each group.
Type MethodSw Th De Es Zh En Avg.
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
HotpotQA
SF-BaselinesZero-shot 17.9 28.1 17.2 21.8 23.2 32.1 23.9 34.3 15.9 33.7 27.9 38.4 21.0 31.4
CoT 16.8 27.3 16.7 21.2 24.6 34.6 24.3 35.1 15.3 33.5 29.8 40.5 21.3 32.0
Vanilla RAG 35.1 48.1 38.4 47.7 44.0 55.5 42.956.234.9 35.5 50.4 62.3 41.0 50.9
SA-BaselinesGraphRAG 18.9 24.8 16.8 20.6 11.2 14.8 23.6 31.3 13.5 23.6 29.0 39.3 18.8 25.7
HippoRAG2 42.257.438.949.245.759.846.2 57.0 34.6 56.2 54.069.043.658.1
DaPT 44.553.240.746.951.559.550.7 60.3 35.255.158.367.546.857.1
2WikiMultiHopQA
SF-BaselinesZero-shot 22.5 26.1 27.0 28.5 28.4 32.5 27.0 31.0 21.5 21.6 26.0 31.4 25.4 28.5
CoT 16.8 27.0 16.8 21.5 27.8 31.2 25.7 30.9 23.9 34.0 28.8 36.1 23.3 30.1
Vanilla RAG 23.6 37.9 37.5 41.0 38.1 43.1 38.4 43.5 36.7 36.8 38.5 43.7 37.0 41.0
SA-BaselinesGraphRAG 22.3 25.2 24.8 27.1 29.3 32.9 27.4 31.4 24.2 32.7 35.4 40.1 27.2 31.6
HippoRAG2 47.4 55.7 46.2 53.752.761.757.065.330.0 43.858.9 68.148.758.1
DaPT 46.5 53.1 45.4 51.655.159.157.562.839.0 51.156.8 67.250.157.5
MuSiQue
SF-BaselinesZero-shot 1.3 9.3 1.1 3.0 3.9 11.2 2.6 13.2 1.9 21.1 4.9 14.8 2.6 12.1
CoT 1.4 9.6 1.1 3.5 3.7 11.2 2.7 11.9 1.7 19.3 4.9 14.1 2.6 11.6
Vanilla RAG 6.4 17.2 7.1 12.5 13.7 21.9 12.5 23.1 9.5 28.5 19.0 29.1 11.4 22.0
SA-BaselinesGraphRAG 2.3 8.2 1.7 8.1 2.2 10.3 3.2 13.5 2.8 20.7 4.7 13.3 2.8 12.4
HippoRAG2 18.129.712.5 19.5 23.8 34.6 22.8 36.5 13.333.031.143.620.2 32.8
DaPT 21.329.515.5 23.8 27.4 35.0 29.3 38.3 18.632.831.241.423.9 33.5
a topological sort on the nodes ofG f. This operation yields a lin-
ear sequence of nodesS= (v 1, v2, . . . , v m), wheremis the total
number of nodes in the fused graph. This sequenceSrepresents the
final, ordered reasoning path, which contains the updatedv∗
lnodes
from the previous step.
2.3. Bilingual Retrieval and Answering
In this stage, we solve the sub-questions iteratively according to the
sequenceS. For each step, the query for the current sub-question
vi+1is dynamically formed by combining the answer from the im-
mediately preceding sub-question,a i. This query update process is
formalized as:
qi+1=combine(a i, vi+1),(3)
wherea iis the answer to the previous sub-question, and combine(·)
is a function that merges them into a new query. Subsequently, the
queryq i+1will be used to retrieve relevant context from a corpus
in the same language asq i+1, enabling the LLM to answer the sub-
questionv i+1with reference to this context and produce a response
ai+1.
For each bilingual nodev∗
l, DaPT treats the bilingual sub-
question as two parallel queries for retrieval, producing candidate
answers from the source-language and English paths, respectively.
We employ a lightweight LLM-based verification step, denoted as
Judge(·,·), to assess whether the two candidate answers are semanti-
cally consistent and sufficient to resolve the current sub-question. If
the verification succeeds, the final answer is selected via Select(·,·),
which prioritizes the source-language answer when both candidates
are valid. Otherwise, DaPT triggers a self-correction mechanism
ReGen(·,·), prompting the LLM to regenerate the answer using the
accumulated bilingual reasoning path as context. Formally,
ai=(
Select(al
i, aen
i),if Judge(al
i, aen
i) =True,
ReGen(v i,Path(v i)),otherwise,(4)This process improves robustness against retrieval noise and
translation discrepancies, while maintaining a lightweight reasoning
pipeline.
In Fig. 1, the process of solving sub-questions begins with the
nodes that have no dependencies. These sub-questions serve as
queries for context retrieval, after which the LLM generates the
corresponding answers. Once an LLM answer is deemed correct, it
becomes the answer of the previous sub-question, combining with
the sub-question of step 2, serving as the query for step 2. When
JudgeLLM determines that step 2 has failed, the LLM will reuse
the previous reasoning path and context to attempt a new response
in step 3 until it succeeds. This process will continue until all
sub-questions are fully resolved.
Finally, to alleviate the burden of long contexts and reduce po-
tential noise, we combine all sub-questions with their corresponding
answers into a reliable logical chain. This chain, together with the
retrieval results of the original question, is provided to the LLM,
which performs integrated reasoning to produce the final answer.
3. EXPERIMENT
3.1. Experiment Setups
Benchmarks. To evaluate the effectiveness of DaPT in MM-hop
question answering, we conduct experiments on three widely used
public multi-hop QA benchmarks: HotpotQA [21], 2WikiMul-
tiHopQA [22], MuSiQue [23]. To ensure a fair and consistent
comparison across methods, we align our entire evaluation setup
with that of HippoRAG2. Specifically, we adopt their methodology
of using a smaller, sampled benchmark of 1,000 questions from each
validation set for testing, rather than the full datasets. Furthermore,
all retrieval is conducted using the exact same corpus to maintain
consistency. To evaluate the performance of RAG systems in mul-
tilingual multi-hop question-answering, we utilized GPT-4o-mini
2https://github.com/OSU-NLP-Group/HippoRAG

to translate the three test datasets into five language versions. We
calibrated the consistency of key entities across translations us-
ing specific rules and human verification, ensuring the integrity of
multi-hop reasoning chains and the translation quality of the data.
Baseline. Our baselines can be divided into two categories ac-
cording to whether structured knowledge is retrieved in multi-hop
reasoning: Structure-Free (SF) baselines and Structure-Aware (SA)
baselines. The Structure-Free baselines setting includes three base-
lines. Zero-shot LLM answers the question using only the model’s
internal knowledge, without any external corpus retrieval. Vanilla
RAG is a standard RAG pipeline that uses the original question di-
rectly as a query to retrieve documents for contextual answering.
CoT Prompting enhances the vanilla RAG by appending the “Let’s
think step by step.” prompt to elicit a detailed reasoning process
from the model. Structure-Aware baselines setting includes two re-
cent, high-performance graph-based RAG methods, GraphRAG [10]
and HippoRAG2 [14].
Metrics. By convention, we employ answer-based Exact Match
(EM) score and token-based F1 score as evaluation metrics. EM
score means the proportion of predictions in the test set that per-
fectly match the reference answer. F1 score evaluates the overlap
between the predicted and reference answer tokens. We follow the
implementation approach of HippoRAG, normalizing both predicted
values and ground truth answers.
Implementation details. To ensure a fair comparison, we uti-
lize GPT-4o-mini as the default LLM across all methods. Based on
previous work [15,16], we use the advanced and available BGE-m33
as the embedding model across all methods. For the Top-k parame-
ter of each method, we setk= 5for HippoRAG2 andk= 3for all
other baselines. The fusion thresholdτis set to 0.8. All experiments
are conducted on 8*NVIDIA GeForce RTX 3090 GPUs.
3.2. Main Results
Advanced RAG systems exhibit significant performance imbal-
ances in multilingual scenarios.As shown in Table 1, while the
HippoRAG2 achieves a strong performance of 54.0 EM score on the
HotpotQA and 31.1 on the harder benchmark MuSiQue in English,
its performance significantly declines in moderately low-resource
languages, such as Thai. Specifically, it shows a decrease of 28.0%
(54.0→38.9) on the HotpotQA and 59.8% (31.1→12.5) on
MuSiQue test sets.
DaPT enhances multilingual performance on MM-hop ques-
tion answering.As shown in Table 1, DaPT demonstrates a consis-
tent advantage over all baselines on the EM score across the three
MM-hop datasets. In particular, DaPT achieves average improve-
ments of 6.8% on HotpotQA and 15.5% on MuSiQue compared with
HippoRAG2. These results highlight the effectiveness of DaPT in
solving MM-hop question answering. Remarkably, DaPT leads to
substantial gains for high-resource languages, with improvements of
4.6% (52.7→55.1) in German and 30% (30.0→39.0) in Chinese
compared with HippoRAG2. This suggests that DaPT, by decom-
posing the query into a sub-question graph and fusing it with their
parallel English versions for reasoning, effectively bridges the lan-
guage gap to some extent. In addition, we note that DaPT does not
always achieve the highest F1 score, which suggests its preference
for offering the most concise and precise responses.
3https://huggingface.co/BAAI/bge-m3Table 2: Ablation performance on multilingual HotpotQA. EM and
F1 represent the average scores for respective non-English scores.
Method EM F1
DaPT (Full Model)44.5 55.1
w/o Decomposition 28.3 (↓16.2) 46.8 (↓8.2)
w/o Fusion 42.7 (↓1.9) 53.1 (↓1.9)
Translate Q&A 41.4 (↓3.2) 51.5 (↓3.5)
3.3. Ablation Study
We present extensive ablation studies of DaPT on the effectiveness
of decomposition and fusion frameworks. Our ablation study on the
multilingual HotpotQA dataset investigates the impact of DaPT’s de-
composition and fusion framework. We compare DaPT with three
variants: (i) DaPT removes the decomposition stage, (ii) DaPT re-
moves the graph fusion stage (iii) Translate Q&A. The Translate
Q&A means a simple method where the source query is translated
to English for the entire reasoning process, and the final answer is
translated back to original language. The performance of these ab-
lated methods on English queries would be redundant: the variant
without decomposition behaves identically to the Vanilla RAG base-
line in English, while the other two variants are identical to the full
DaPT framework when handling English questions. We only report
the average EM scores and F1 scores on the non-English languages.
As shown in Table 2, DaPT achieves the best performance com-
pared to the framework without its decomposition and fusion com-
ponents and the Translate Q&A method. The results demonstrate
the effectiveness and the necessity of the decomposition and fusion
framework within DaPT. Furthermore, it confirms that our method
is inherently more advanced than a simple Translate Q&A strategy.
4. CONCLUSION
In this work, we address the challenge of multilingual multi-hop QA
with two key contributions. First, we construct a new set of multilin-
gual multi-hop benchmarks by translating three mainstream bench-
marks, HotpotQA, 2WikiMultiHopQA, MuSiQue into five diverse
languages, ranging from low- to high-resource. Second, we propose
DaPT, a novel multilingual RAG framework for MM-hop question
answering. DaPT first translates the original-language query into a
parallel English counterpart, then decomposes and fuses both into a
bilingual sub-question graph, and finally performs sequential reason-
ing over this unified structure leading to richer reasoning paths, more
accurate results, and better interpretability. Extensive experiments
on our benchmarks demonstrate that DaPT consistently outperforms
structure-aware approaches.
5. ACKNOWLEDGEMENTS
This work was supported in part by the National Science Foundation
of China (Nos. 62276056 and U24A20334), the Yunnan Fun-
damental Research Projects (No.202401BC070021), the Yunnan
Science and Technology Major Project (No. 202502AD080014),
the Fundamental Research Funds for the Central Universities (Nos.
N25BSS054 and N25BSS094), and the Program of Introducing
Talents of Discipline to Universities, Plan 111 (No.B16009).

6. REFERENCES
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad,
Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko
Altenschmidt, Sam Altman, Shyamal Anadkat, et al., “Gpt-4
technical report,”arXiv preprint arXiv:2303.08774, 2023.
[2] AI Anthropic, “The claude 3 model family: Opus, sonnet,
haiku. claude-3 model card, 2024,” inConference on Natural
Language Processing, vol. 1, pp. 4582–4597.
[3] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu,
Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang,
Chong Ruan, et al., “Deepseek-v3 technical report,”arXiv
preprint arXiv:2412.19437, 2024.
[4] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua
Peng, Xiaocheng Feng, Bing Qin, et al., “A survey on hallu-
cination in large language models: Principles, taxonomy, chal-
lenges, and open questions,”ACM Transactions on Informa-
tion Systems, vol. 43, no. 2, pp. 1–55, 2025.
[5] Pengcheng Huang, Zhenghao Liu, Yukun Yan, Haiyan Zhao,
Xiaoyuan Yi, Hao Chen, Zhiyuan Liu, Maosong Sun, Tong
Xiao, Ge Yu, et al., “Parammute: Suppressing knowledge-
critical ffns for faithful retrieval-augmented generation,” inThe
Thirty-ninth Annual Conference on Neural Information Pro-
cessing Systems, 2025.
[6] Lei Huang, Xiaocheng Feng, Weitao Ma, Yuchun Fan, Xia-
chong Feng, Yuxuan Gu, Yangfan Ye, Liang Zhao, Weihong
Zhong, Baoxin Wang, et al., “Alleviating hallucinations from
knowledge misalignment in large language models via selec-
tive abstention learning,” inProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), 2025, pp. 24564–24579.
[7] Lei Huang, Xiaocheng Feng, Weitao Ma, Yuchun Fan, Xia-
chong Feng, Yangfan Ye, Weihong Zhong, Yuxuan Gu, Baoxin
Wang, Dayong Wu, et al., “Improving contextual faithfulness
of large language models via retrieval heads-induced optimiza-
tion,”arXiv preprint arXiv:2501.13573, 2025.
[8] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and
Ashish Sabharwal, “Interleaving retrieval with chain-of-
thought reasoning for knowledge-intensive multi-step ques-
tions,”arXiv preprint arXiv:2212.10509, 2022.
[9] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan
Duan, and Weizhu Chen, “Enhancing retrieval-augmented
large language models with iterative retrieval-generation syn-
ergy,”arXiv preprint arXiv:2305.15294, 2023.
[10] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex
Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky,
Robert Osazuwa Ness, and Jonathan Larson, “From local to
global: A graph rag approach to query-focused summariza-
tion,”arXiv preprint arXiv:2404.16130, 2024.
[11] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang,
“Lightrag: Simple and fast retrieval-augmented generation,”
arXiv preprint arXiv:2410.05779, 2024.
[12] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Ya-
sunaga, and Yu Su, “Hipporag: Neurobiologically inspired
long-term memory for large language models,”Advances in
Neural Information Processing Systems, vol. 37, pp. 59532–
59569, 2024.[13] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna,
Anna Goldie, and Christopher D Manning, “Raptor: Recur-
sive abstractive processing for tree-organized retrieval,” in
The Twelfth International Conference on Learning Represen-
tations, 2024.
[14] Bernal Jim ´enez Guti ´errez, Yiheng Shu, Weijian Qi, Sizhe
Zhou, and Yu Su, “From rag to memory: Non-parametric con-
tinual learning for large language models,” 2025.
[15] Nadezhda Chirkova, David Rau, Herv ´e D´ejean, Thibault For-
mal, St ´ephane Clinchant, and Vassilina Nikoulina, “Retrieval-
augmented generation in multilingual settings,”arXiv preprint
arXiv:2407.01463, 2024.
[16] Jeonghyun Park and Hwanhee Lee, “Investigating language
preference of multilingual rag systems,”arXiv preprint
arXiv:2502.11175, 2025.
[17] Jirui Qi, Raquel Fern ´andez, and Arianna Bisazza, “On the
consistency of multilingual context utilization in retrieval-
augmented generation,”arXiv preprint arXiv:2504.00597,
2025.
[18] Yongyu Mu, Hengyu Li, Junxin Wang, Xiaoxuan Zhou, Chen-
glong Wang, Yingfeng Luo, Qiaozhi He, Tong Xiao, Guocheng
Chen, and Jingbo Zhu, “Boosting text-to-image generation
via multilingual prompting in large multimodal models,” in
ICASSP 2025 - 2025 IEEE International Conference on Acous-
tics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.
[19] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian,
and Zheng Liu, “Bge m3-embedding: Multi-lingual, multi-
functionality, multi-granularity text embeddings through self-
knowledge distillation,”arXiv preprint arXiv:2402.03216,
2024.
[20] Yuchun Fan, Yongyu Mu, Yilin Wang, Lei Huang, Junhao
Ruan, Bei Li, Tong Xiao, Shujian Huang, Xiaocheng Feng,
and Jingbo Zhu, “Slam: Towards efficient multilingual rea-
soning via selective language alignment,”arXiv preprint
arXiv:2501.03681, 2025.
[21] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William W Cohen, Ruslan Salakhutdinov, and Christopher D
Manning, “Hotpotqa: A dataset for diverse, explainable multi-
hop question answering,”arXiv preprint arXiv:1809.09600,
2018.
[22] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and
Akiko Aizawa, “Constructing a multi-hop qa dataset for com-
prehensive evaluation of reasoning steps,”arXiv preprint
arXiv:2011.01060, 2020.
[23] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and
Ashish Sabharwal, “Musique: Multihop questions via single-
hop question composition,”Transactions of the Association for
Computational Linguistics, vol. 10, pp. 539–554, 2022.
[24] Ercong Nie, Sheng Liang, Helmut Schmid, and Hinrich
Sch¨utze, “Cross-lingual retrieval augmented prompt for low-
resource languages,”arXiv preprint arXiv:2212.09651, 2022.
[25] Yongyu Mu, Peinan Feng, Zhiquan Cao, Yuzhang Wu, Bei Li,
Chenglong Wang, Tong Xiao, Kai Song, Tongran Liu, Chun-
liang Zhang, et al., “Large language models are parallel multi-
lingual learners,”arXiv preprint arXiv:2403.09073, 2024.