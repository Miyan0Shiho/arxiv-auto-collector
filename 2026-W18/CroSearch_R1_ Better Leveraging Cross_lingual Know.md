# CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation

**Authors**: Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, Kaiyu Huang

**Published**: 2026-04-28 03:41:36

**PDF URL**: [https://arxiv.org/pdf/2604.25182v1](https://arxiv.org/pdf/2604.25182v1)

## Abstract
A multilingual collection may contain useful knowledge in other languages to supplement and correct the facts in the original language for Retrieval-Augmented Generation (RAG). However, the vanilla approach that simply concatenates multiple pieces of knowledge from different languages into the context may fail to improve effectiveness due to the potential disparities across languages. To better leverage multilingual knowledge, we propose CroSearch-R1, a search-augmented reinforcement learning framework to integrate multilingual knowledge into the Group Relative Policy Optimization (GRPO) process. In particular, the approach adopts a multi-turn retrieval strategy with cross-lingual knowledge integration to dynamically align the knowledge from other languages as supplementary evidence into a unified representation space. Furthermore, we introduce a multilingual rollout mechanism to optimize reasoning transferability across languages. Experimental results demonstrate that our framework effectively leverages cross-lingual complementarity and improves the effectiveness of RAG with multilingual collections.

## Full Text


<!-- PDF content starts -->

CroSearch-R1: Better Leveraging Cross-lingual Knowledge for
Retrieval-Augmented Generation
Rui Qi
School of Computer Science and
Technology, Beijing Jiaotong
University
Beijing, China
cherry@bjtu.edu.cnFengran Mo
RALI, Université de Montréal
Montréal, Québec, Canada
fengran.mo@umontreal.caSijin Lu
School of Computer Science and
Technology, Beijing Jiaotong
University
Beijing, China
lusijin@bjtu.edu.cn
Yufeng Chen
School of Computer Science and
Technology, Beijing Jiaotong
University
Beijing, China
chenyf@bjtu.edu.cnJian-Yun Nie
RALI, Université de Montréal
Montréal, Québec, Canada
nie@iro.umontreal.caKaiyu Huang *
School of Computer Science and
Technology, Beijing Jiaotong
University
Beijing, China
kyhuang@bjtu.edu.cn
Abstract
A multilingual collection may contain useful knowledge in other
languages to supplement and correct the facts in the original lan-
guage for Retrieval-Augmented Generation (RAG). However, the
vanilla approach that simply concatenates multiple pieces of knowl-
edge from different languages into the context may fail to improve
effectiveness due to the potential disparities across languages. To
better leverage multilingual knowledge, we proposeCroSearch-R1,
a search-augmented reinforcement learning framework to integrate
multilingual knowledge into the Group Relative Policy Optimiza-
tion (GRPO) process. In particular, the approach adopts a multi-turn
retrieval strategy with cross-lingual knowledge integration to dynam-
ically align the knowledge from other languages as supplementary
evidence into a unified representation space. Furthermore, we intro-
duce a multilingual rollout mechanism to optimize reasoning trans-
ferability across languages. Experimental results demonstrate that
our framework effectively leverages cross-lingual complementarity
and improves the effectiveness of RAG with multilingual collections.
CCS Concepts
•Information systems→Question answering.
Keywords
Retrieval-Augmented Generation, Multilingualism, Cross-lingual
Retrieval
ACM Reference Format:
Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, and Kaiyu
Huang. 2026. CroSearch-R1: Better Leveraging Cross-lingual Knowledge
for Retrieval-Augmented Generation. InProceedings of the 49th Interna-
tional ACM SIGIR Conference on Research and Development in Information
*Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3809921Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Australia.ACM,
New York, NY , USA, 7 pages. https://doi.org/10.1145/3805712.3809921
1 Introduction
Retrieval-augmented generation (RAG) leverages external knowl-
edge from rich sources to enhance large language models (LLMs)
in QA, so as to mitigate hallucinations and improve response accu-
racy [ 4,9,27,37,48,49]. While most investigations on RAG have
focused primarily on monolingual scenario, external collections in
different languages can provide complementary knowledge to RAG
due to their different coverage, diverse linguistic expressions and cul-
tural features [ 11,17,21]. Therefore, a valuable practical scenario
to investigate is RAG that exploits knowledge from multilingual
collections, and in particular, from collections in other languages
(that we call cross-lingual collections).
This problem has been overlooked in the existing research. RAG
approaches using multilingual collections simply concatenate multi-
ple pieces of knowledge from different languages based solely on
the relevance [ 7,10,21,26]. This simplistic solution leads to two
main limitations. First, all retrieved knowledge is treated equally
regardless of language. This may penalize the knowledge in the
original language of the query, which is usually assumed to be more
influential. Second, there is a representation gap between different
languages in LLMs [ 29], so that the question representation is only
weakly aligned with that of documents in other languages, leading
to mismatch. As a result, the existing methods trying to incorporate
cross-lingual external knowledge often fail to improve effectiveness
due to the potential disparities across languages.
To better leverage cross-lingual knowledge for RAG, a core issue
is to design an appropriate method to project cross-lingual knowl-
edge into a unified representation space. This task is challenging
because it is difficult to determine what cross-lingual knowledge
is useful for a question and when it should be combined [ 5,12].
Search-augmented reinforcement learning (RL) provides a possible
solution to this problem [ 14,15,24,43,45,47]. It enables LLMs
to engage in dynamic interactions with search engines and perform
multi-turn reasoning. In so doing, the entire framework is optimizedarXiv:2604.25182v1  [cs.CL]  28 Apr 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Trovato et al.
holistically within a search-and-reasoning framework where knowl-
edge can be integrated at multiple stages in various forms depending
on the needs.
In this paper, we proposeCroSearch-R1, a search-augmented RL
framework to address this issue. It uses Group Relative Policy Opti-
mization (GRPO) [ 34] to achieve efficient and dynamic utilization of
cross-lingual knowledge. Specifically, CroSearch-R1 adopts acom-
plemental knowledge retrievalstrategy where LLMs dynamically de-
termine the timing to invoke the search engine and perform multiple
retrieval turns when encountering knowledge insufficiency. The pro-
gressive strategy prioritizes local collection for primary knowledge
in the same language during the first turn, then adaptively expands to
global collections for complementary evidence in subsequent steps
when needed. This strategy that prioritizes the knowledge in the
same language can mitigate cross-lingual interference. Moreover, to
address semantic gap in cross-lingual documents, we perform mul-
tilingual knowledge normalization in the process ofcross-lingual
knowledge integration, bridging disparate knowledge into a unified
representation space to alleviate contextual conflict, which often
stems from inconsistent factual claims or linguistic biases inherent
in different language corpora. In reinforcement learning optimiza-
tion, CroSearch-R1 introduce aSynchronized Cross-Lingual Policy
Optimization, which enables the model to explore a single query
through diverse cross-lingual “thinking” patterns. Instead of standard
repetitive sampling in an invariable linguistic setting, each rollout
within a group is guided to initiate its retrieval and reasoning in the
cross-lingual manner. By transforming sampling into this multi-path
cross-lingual exploration, queries can adapt to their respective exter-
nal retrieval requirements and implicitly learn the transferability of
knowledge from other languages. Experimental results demonstrate
that CroSearch-R1 effectively leverages cross-lingual knowledge
collections as complements and improves the effectiveness of RAG
with cross-lingual collections.
2 Methodology
2.1 Complemental Knowledge Retrieval
As shown in Alg. 1, we formulate the LLM response generation as
an adaptive, multi-turn retrieval-augmented process, where external
cross-lingual evidence is incrementally incorporated only when nec-
essary. Following an iterative framework [ 14], the model alternates
between internal reasoning and external search calls. The system in-
struction explicitly guides the model to structure its generation using
special action tokens, including <think>, <search>, <information>,
and <answer>. Formally, the generation process is defined as:
𝑦∼𝜋𝜃(·|𝑥(𝐿),M),withM=Ø
𝑙∈GS𝑙
Here,𝑥(𝐿)denotes the user query in language 𝐿, andS𝑙represents
evidences retrieved from collection of language 𝑙using specific
search operators{𝜔𝑙}𝑙∈G. Upon detecting search tokens, the method
extracts the query and retrieves evidenceS 𝑙, then appends it within
<information> and </information> tags as additional context for the
next generation step. Subsets of S𝑙are adaptively queried according
to the model’s intermediate reasoning state.
Specifically, the model first calls the search operator 𝜔𝐿for the
query language 𝐿by inserting 𝑥(𝐿)within <search> and </search>Algorithm 1:Multi-Turn Dynamic Retrieval
Require :Input query𝑥(𝐿), policy model𝜋 𝜃, cross-lingual
search operatorsW={𝜔 𝑙}𝑙∈G, max budget𝐵.
Ensure :Final response𝑦.
1Initialize context𝐶←{𝑥(𝐿)}and search turn count𝑏←0
2while𝑏<𝐵do
3 Generate trajectory segment 𝜏∼𝜋𝜃(·|𝐶) until anAction
Markeris reached;𝐶←𝐶∪{𝜏}
4if𝜏triggers a Search Actionthen
5Extract search query𝑞from𝜏
6𝑏←𝑏+1
// Complemental Knowledge Retrieval
7if𝑏=1then
8S←𝜔 𝐿(𝑞)
9else if𝑏=2then
10S←Ð
𝑙∈G𝜔𝑙(𝑞)
11else
12S←𝜔 𝑒𝑛(𝑞)
13𝐶←𝐶∪{KnowledgeNorm(S)}
14else if𝜏triggers a Final Response Actionthen
15returnFinal answer extracted from𝐶
16else
17𝐶←𝐶∪{Self-Correction Message}
18returnFinal response from𝐶
tags, aiming to establish a backbone using culturally and contex-
tually relevant local evidence. If the model still lacks information
after interacting with relevant documents, it triggers a complemen-
tal turn to search cross-lingual collections M\S𝐿. In addition,
the model retrieves from a high-resource language collection (e.g.,
English𝜔𝑒𝑛) to provide the remaining complementary knowledge
when the retrieval process is not completed after more than two
turns. This iterative process continues until the model generates its
final response within <answer> and </answer> tags, or reaches the
maximum search budget𝐵.
In contrast to strategies that simply concatenate knowledge from
all languages, our multi-turn retrieval strategy leverages query lan-
guage evidence as a foundational anchor, maintaining the dominance
of the original context. By introducing cross-lingual knowledge as
a selective supplement to fill specific factual gaps, this approach
effectively prevents redundant information from creating noise or
interfering with the reasoning process.
2.2 Cross-lingual Knowledge Integration
To fully exploit the potential of knowledge collections across lan-
guages, we propose a cross-lingual knowledge integration mecha-
nism aimed at harmonizing multiple knowledge and bridging seman-
tic gaps.
Multilingual Knowledge Normalization.To address the semantic
gap prevalent in non-native documents, we first perform the multi-
lingual knowledge normalization to transfer the representation from
other languages to the original language. Given a set of retrieved

CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
global documentsD𝑔𝑙𝑜𝑏𝑎𝑙 in various languages, we apply a transla-
tion operatorTto map each document 𝑑∈D𝑔𝑙𝑜𝑏𝑎𝑙 into a unified
semantic space aligned with the query language𝐿:
ˆD𝑔𝑙𝑜𝑏𝑎𝑙 ={T(𝑑,𝐿)|𝑑∈D 𝑔𝑙𝑜𝑏𝑎𝑙}
where ˆD𝑔𝑙𝑜𝑏𝑎𝑙 represents the normalized global evidence, facilitating
the seamless fusion of multilingual information.
Fact-based Evidence Reconstruction.Building upon the normal-
ization, the model is explicitly guided to reconstruct the normalized
global evidence into brief fact-statements Fwithin the <think> tags.
Steered by the specialized prompt P𝑟𝑒𝑐𝑜𝑛 , the model is asked to
contrast the newly acquired cross-lingual evidence against the first-
round results in the original language to resolve any factual conflicts
or inherent biases. Formally, the reasoning process is defined as:
F∼𝜋𝜃(·|𝑥(𝐿),D𝑛𝑎𝑡𝑖𝑣𝑒,ˆD𝑔𝑙𝑜𝑏𝑎𝑙 ;P𝑟𝑒𝑐𝑜𝑛)
whereD𝑛𝑎𝑡𝑖𝑣𝑒 is the original evidence acquired in the first retrieval
turn, and P𝑟𝑒𝑐𝑜𝑛 denotes the specialized prompt for fact reconstruc-
tion. These reconstructed global facts Fare not a simple concatena-
tion of the original documents, but rather serve as information-dense
complementary signals to refine the original knowledge.
Finally, the policy model 𝜋𝜃synthesizes the predictive response
𝑦by integrating the foundational original evidence with the recon-
structed cross-lingual facts:
𝑦∼𝜋𝜃(·|𝑥(𝐿),D𝑛𝑎𝑡𝑖𝑣𝑒,F)
The reconstruction leverages cross-lingual complementarity to bridge
core information gaps while preserving the centrality of the context
in the original language of the query, thereby minimizing interfer-
ence due to knowledge disparities.
2.3 Synchronized Cross-Lingual Policy
Optimization
To handle the complexities of cross-lingual environment, we intro-
duce a synchronized optimization strategy built upon GRPO. This
design aims to train the model to develop universal reasoning capa-
bility for cross-lingual scenarios by comparing the thinking paths
for the same question in different languages.
Parallel Trajectory Sampling.To foster a unified understanding
across languages, we explore the source query 𝑥through a set of
cross-lingual thinking modes L={𝑙 1,𝑙2,...,𝑙𝐺}, where each mode
𝑙𝑖guides the model to initiate its reasoning and retrieval within
a specific language context. This approach ensures that the core
semantic intent remains unchanged while the reasoning trajectory
is conditioned on different languages. During the rollout phase,
we perform 𝐺independent samplings for the query 𝑥. For the𝑖-th
sampling (𝑖∈{1,...,𝐺} ), the model generates a reasoning response
𝑟𝑖under the guidance of the cross-lingual thinking mode𝑙 𝑖:
𝑟𝑖∼𝜋𝜃(·|𝑥,E𝑖,𝑙𝑖)
whereE𝑖represents the evidence retrieved from the global collection
in𝑖-th sampling. We optimize the policy𝜋 𝜃by maximizing:LCLPO(𝜃)=E{𝜏𝑖}𝐺
𝑖=1"
1
𝐺𝐺∑︁
𝑖=11
|T𝑖|∑︁
𝑡∈T𝑖
min
𝜌𝑖,𝑡(𝜃)˜𝐴𝑐𝑟𝑜𝑠𝑠
𝑖,𝑡,clip(𝜌𝑖,𝑡(𝜃),1−𝛿,1+𝛿) ˜𝐴𝑐𝑟𝑜𝑠𝑠
𝑖,𝑡
−𝜆·1
𝐺𝐺∑︁
𝑖=1KL(𝜋𝜃(·|𝑥(𝑙𝑖))∥𝑝 ref(·|𝑥(𝑙𝑖)))#
where𝜌𝑖,𝑡(𝜃)denotes the importance ratio between the updated pol-
icy and the behavior policy at step 𝑡,˜𝐴𝑐𝑟𝑜𝑠𝑠
𝑖,𝑡is the advantage baseline
computed using cross-lingual sampling group, and T𝑖indexes the
time steps associated with reasoning, retrieval invocation, and fact
synthesis actions. The standard GRPO follows a uniform sampling
procedure where multiple responses are generated from a given
query, resulting in relatively homogeneous reasoning. In contrast,
we guide the query to think in different languages, sampling across
a group of semantically equivalent queries. These samples are com-
plementary because they represent diverse reasoning and retrieval
trajectories triggered by the unique knowledge in each language.
By exposing the policy to rollouts conditioned on the cross-lingual
thinking modes, the model is trained to process the trajectories
in various expressions of the same semantics. The cross-lingual
rollout can acquire diverse training samples. This ensures that the
model learns a broader range of semantic retrieval and generation
relationships, thereby enhancing its cross-lingual transferability.
N-gram Outcome Reward.Rule-based outcome rewards based on
exact string matching, as adopted in prior studies such as Search-R1,
are ill-suited for cross-lingual settings due to their sensitivity to
surface-level lexical variation. To address this limitation, we employ
a character-level similarity metric based on 3-gram recall to evaluate
answer correctness across languages.
Formally, the outcome score for trajectory𝜏 𝑖is defined as:
𝑟out(𝑖)=Char-3-Recall( ˆ𝑦𝑖,𝑦★)
where ˆ𝑦𝑖denotes the extracted answer span and 𝑦★is the reference
answer. This metric provides a smooth and dense reward signal that
mitigates abrupt reward discontinuities, reducing the likelihood of
unstable gradient updates caused by extreme importance weight
amplification.
3 Experiments
3.1 Experimental Settings
We evaluate our method on both multilingual knowledge-intensive
benchmark MKQA [ 22] in Latin languages (EN, FR) and non-Latin
languages (TH, AR) and representative English benchmarks includ-
ing HotpotQA [ 44], PopQA [ 23] and 2WikimultihopQA [ 6]. For
all benchmarks, we utilize a set of Wikimedia_dump1from dif-
ferent languages (French, English, Arabic, and Thai) as the cross-
lingual collections. All methods in the experiment are allowed to
utilize the full collections for a fair comparison. Retriever and Trans-
lator are selected as multilingual E5 [ 40] and nllb-200-distilled-
600M [ 38], respectively. To ensure a comprehensive evaluation, we
evaluate against: (1) Direct Inference without Retrieval; (2) Inference
1https://huggingface.co/datasets/wikimedia/wikipedia

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Trovato et al.
Table 1: Performance on MKQA and English benchmarks. Background colors indicate training-free and post-training paradigms.
MethodsMKQA (cross-lingual, fEM / c3Recall) English Datasets (monolingual, fEM / c3Recall)
ENFRTHARAve HOTPOTQA POPQA 2WIKIAve
Qwen2.5-3B-Instruct
Direct Inference 41.11 / 51.80 29.21 / 40.04 11.43 / 18.75 7.83 / 17.60 22.40 / 32.05 19.60 / 35.97 18.50 / 30.50 36.60 / 44.79 24.90 / 37.09
IRCoT 38.83 / 51.49 39.88 / 49.29 15.46 / 21.13 12.46 / 20.61 26.66 / 35.63 35.42 / 42.70 52.68 / 60.44 34.17 / 44.58 40.75 / 49.24
Search-o1 41.66 / 53.23 38.62 / 51.05 15.25 / 20.35 13.67 / 21.81 27.30 / 36.61 30.44 / 41.68 54.04 / 62.50 33.12 / 45.65 39.20 / 49.94
RAG 52.08 / 60.73 40.72 / 50.20 14.02 / 19.04 11.59 / 17.42 29.60 / 36.85 31.50 / 43.84 50.10 / 59.47 38.60 / 45.90 40.07 / 49.74
SFT 56.85 / 63.60 48.00 / 56.56 16.91 / 29.06 16.84 / 28.23 34.65 / 44.36 39.20/ 50.79 59.50 / 65.18 38.90 / 45.90 45.87 / 53.96
Search-R1 63.38 / 77.31 44.73 / 61.35 22.36 / 33.95 16.89 / 39.49 36.82 / 53.03 35.35 / 62.65 59.96 / 78.39 52.93 / 65.76 49.41 / 68.93
Ours 67.68/83.22 54.00/72.39 24.41/42.26 17.48/47.87 40.89/61.44 38.67 /66.88 65.04/83.46 55.26/70.42 52.99/73.59
Qwen2.5-7B-Instruct
Direct Inference 49.96 / 60.00 35.79 / 46.86 14.02 / 20.75 10.73 / 19.69 27.63 / 36.83 26.50 / 43.83 24.90 / 37.06 33.80 / 41.59 28.40 / 40.83
IRCoT 40.88 / 53.67 41.52 / 51.44 18.06 / 23.65 15.28 / 23.84 28.94 / 38.15 40.84 / 47.67 31.43 / 50.22 33.34 / 44.85 35.20 / 47.58
Search-o1 44.24 / 56.06 41.48 / 53.57 18.54 / 23.65 15.16 / 24.53 29.86 / 39.45 36.32 / 47.62 30.48 / 47.06 33.92 / 46.88 33.57 / 47.19
RAG 55.05 / 64.12 44.64 / 53.53 16.76 / 22.55 16.37 / 24.43 33.21 / 41.16 37.60 / 50.34 10.00 / 55.00 39.70 / 46.69 29.10 / 50.68
SFT 59.91 / 66.86 51.92 / 59.53 23.26 / 34.69 19.81 / 29.25 38.73 / 47.58 43.00 / 54.61 62.60/ 68.40 39.80 / 46.25 48.47 / 56.42
Search-R1 69.43 / 79.82 57.91 / 70.36 25.29 / 37.13 18.65 / 34.68 42.82 / 55.50 44.14/ 67.17 58.20 / 70.96 56.05 / 66.08 52.80 / 68.07
Ours 72.07/82.68 59.67/74.04 27.83/45.98 24.12/41.63 45.92/61.08 44.14/67.70 61.33 /78.04 61.13/73.64 55.53/73.13
Table 2: Average results of CroSearch-R1 without each module
based onQwen2.5-3B-Instructfor MKQA.
fEM c3Recall
Our Full Framework 40.89 61.44
w/o Cross-lingual Knowledge Integration 38.46 55.68
w/o Cross-lingual Policy Optimization 38.40 56.44
w/o c3Recall Reward 28.85 33.49
with Retrieval, including naive RAG [ 18], IRCoT [ 39] and Search-
o1 [20]; and (3) Fine-Tuning Methods such as Supervised fine-
tuning (SFT) [ 3] and a cross-lingual reproduction of Search-R1 [ 14].
We employ Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct [ 30] as
the primary backbones for our experiments. For evaluation, we fol-
low Chirkova et al . [2] and use both flexible exact match (fEM) and
recall on character 3-gram level (c3Recall) [33].
3.2 Results
Overall.The left half of Table 1 shows the performance on MKQA,
where CroSearch-R1consistently surpasses strong baselines on LLMs
of different sizes with respect to fEM and c3Recall. While Search-R1
demonstrates robust cross-lingual transferability, it remains hindered
by optimization difficulties and marginal performance gains. In ad-
dition, as shown in the right half of Table 1, the results on English
benchmarks demonstrate that cross-lingual resources generally make
positive impact with CroSearch-R1, while they may degrade the per-
formance with other methods due to knowledge disparities. This phe-
nomenon is particularly pronounced for the resource-rich language
- English. In fact, existing methods simply concatenates original
and cross-lingual evidence. It is easy for cross-lingual knowledge
to introduce noise that interferes with that of the original language.Our method retains the dominance of the original language while
introducing cross-lingual thinking patterns during sampling to gen-
erate trajectories with linguistic diversity. This expands the state
space for cross-lingual retrieval and generation, allowing the model
to better leverage cross-lingual knowledge and brige the semantic
gap, compared with Search-R1.
Ablation Studies.Table 2 investigates the impact of different mod-
ules in CroSearch-R1. The results demonstrate that the removal of
each module leads consistently to a performance degradation. Specif-
ically, cross-lingual knowledge integration and cross-lingual policy
optimization have greater impacts. This indicates that it is more cru-
cial to eliminate representation discrepancies when injecting cross-
lingual knowledge, as LLMs can better understand such knowledge
effectively for the query language. Compared to the outcome-based
reward function for exact matching, the n-gram score broadens the
scope of rewards, allowing for greater flexibility in generated an-
swers, particularly when the length of outputs varies significantly
the for cross-lingual scenario.
3.3 Analysis
Scalability of Increasing Language Counts.Figure 1a illustrates
the variation in retrieval performance with increasing language di-
versity. In contrast to other RAG methods, whose performance sig-
nificantly declines with the increase in the number of languages, our
method consistently maintains an upward trend. This performance
degradation is primarily caused by the interference of cross-lingual
knowledge, where the potential disparities in knowledge across lan-
guages may confuse the model. This situation corresponds to the
research question we identified and our method addresses this by em-
ploying a complemental injection strategy that selectively integrates
cross-lingual evidence when necessary, ensuring that the original
context remains the dominant anchor for the final response.

CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
Ul Ul+En All3035404550556065Average Score
RAG Search-R1 CroSearch-R1
(a) Number of language collections.
-En -Fr -Th -ArEn
Fr
Th
ArQuery Language1.13 1.00 0.29
9.74 2.43 4.30
4.45 1.85 1.35
5.26 3.29 2.10
2468
Relative Performance Drop (%)
(b) Type of language collections.
Figure 1: Performance variation caused by language collections.
Ul of (a) indicates the query language and the value within
each block in (b) represents the degradation when document
collections in another language (X-axis) are removed.
Effect of Language-specific Collections.As shown in Figure 1b,
we investigate the impact of language-specific collections for differ-
ent query languages. The results show that the exclusion of English
knowledge leads to a substantial performance degradation across all
other languages and the performance on English queries remains
robust regardless of the removal of any other language collections. It
reveals the high density of valuable information within the English
collection, which is directly proportional to its predominant cor-
pus scale. Interestingly, we identify significant mutual knowledge
synergy between specific language pairs. For instance, French and
Arabic exhibit high reciprocal utility, where external knowledge
from one language significantly enhances the response quality of
the other. However, the synergy between Thai and Arabic is rela-
tively limited, with each language providing only marginal gains for
queries in the other.4 Related Work
Retrieval-Augmented Generation [ 18,25,35] improves question
answering by incorporating external knowledge during content gen-
eration, using additional context to achieve better accuracy for
LLMs [ 36,42,46]. In cross-lingual settings, relevant knowledge is
often distributed across documents in different languages, and direct
integration may introduce noise or lead to conflicting information,
which harms generation quality. Existing cross-lingual RAG meth-
ods commonly adopt a translation-based strategy [ 1,2,13,28,31],
translating documents into a single language before retrieval and
generation. This approach reduces language gaps to some extent,
but noise introduced by translation errors may lead to the loss of
key information. Other methods focus on modifying the knowl-
edge injection process [ 8,41] or the reasoning strategy [ 16,19,32].
However, modifying knowledge injection often disrupts semantic
integrity, while complex reasoning strategies remain prone to "lan-
guage drift" [ 19] and hallucination without robust filtering of cross-
lingual evidence.
In contrast to these methods, we fully leverage cross-lingual
knowledge through Cross-lingual Knowledge Integration and multi-
path sampling optimization, while employing a progressive search
strategy with an iterative RL method to mitigate the interference of
cross-lingual knowledge from different languages.
5 Conclusions
In this paper, we present CroSearch-R1, a search-augmented RL
framework designed to effectively leverage cross-lingual knowl-
edge for RAG. By integrating a multi-turn retrieval and cross-lingual
knowledge integration strategy into a synchronized cross-lingual pol-
icy optimization framework, the approach enables dynamic knowl-
edge complement and multi-path reasoning transfer across languages.
Experimental results demonstrate that CroSearch-R1 effectively ex-
ploits cross-lingual knowledge retrieval and significantly bridges the
gap between language representations.
Acknowledgments
This work is supported by the Fundamental Research Funds for
the Central Universities of China under Grant 2024JBGP008 and
the National Natural Science Foundation of China (No. 62406018,
62476023, 61976016, 62376019), and the authors would like to
thank the anonymous reviewers for their valuable comments and
suggestions to improve this paper.
References
[1]Syed Rameel Ahmad. 2024. Enhancing Multilingual Information Retrieval in
Mixed Human Resources Environments: A RAG Model Implementation for Mul-
ticultural Enterprise. arXiv:2401.01511 [cs.IR] https://arxiv.org/abs/2401.01511
[2]Nadezhda Chirkova, David Rau, Hervé Déjean, Thibault Formal, Stéphane Clin-
chant, and Vassilina Nikoulina. 2024. Retrieval-augmented generation in multi-
lingual settings. InProceedings of the 1st Workshop on Towards Knowledgeable
Language Models (KnowLLM 2024). 177–188.
[3]Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus,
Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al .2024.
Scaling instruction-finetuned language models.Journal of Machine Learning
Research25, 70 (2024), 1–53.
[4]Chenxu Cui, Haihui Fan, Jinchao Zhang, Lin Shen, Bo Li, and Weiping Wang.
2025. CIRAG: Retrieval-Augmented Language Model with Collective Intelligence.
InProceedings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval(Padua, Italy)(SIGIR ’25). Association for

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Trovato et al.
Computing Machinery, New York, NY , USA, 1316–1326. doi:10.1145/3726302.
3729921
[5]Ping Guo, Yubing Ren, Yue Hu, Yanan Cao, Yunpeng Li, and Heyan Huang.
2024. Steering Large Language Models for Cross-lingual Information Retrieval.
InProceedings of the 47th International ACM SIGIR Conference on Research
and Development in Information Retrieval(Washington DC, USA)(SIGIR ’24).
Association for Computing Machinery, New York, NY , USA, 585–596. doi:10.
1145/3626772.3657819
[6]Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics, Donia Scott, Nuria Bel, and Chengqing Zong (Eds.). International
Committee on Computational Linguistics, Barcelona, Spain (Online), 6609–6625.
doi:10.18653/v1/2020.coling-main.580
[7]Md. Asif Hossain, Nabil Subhan, Mantasha Rahman Mahi, and Jannatul Ferdous
Nabila. 2026. Cost-Efficient Cross-Lingual Retrieval-Augmented Generation
for Low-Resource Languages: A Case Study in Bengali Agricultural Advisory.
arXiv:2601.02065 [cs.CL] https://arxiv.org/abs/2601.02065
[8]Md. Asif Hossain, Nabil Subhan, Mantasha Rahman Mahi, and Jannatul Ferdous
Nabila. 2026. Cost-Efficient Cross-Lingual Retrieval-Augmented Generation
for Low-Resource Languages: A Case Study in Bengali Agricultural Advisory.
arXiv:2601.02065 [cs.CL] https://arxiv.org/abs/2601.02065
[9]Yuntong Hu, Zhihan Lei, Zhongjie Dai, Allen Zhang, Abhinav Angirekula, Zheng
Zhang, and Liang Zhao. 2025. CG-RAG: Research Question Answering by Cita-
tion Graph Retrieval-Augmented LLMs. InProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval
(Padua, Italy)(SIGIR ’25). Association for Computing Machinery, New York, NY ,
USA, 678–687. doi:10.1145/3726302.3729920
[10] Chao Huang, Fengran Mo, Yufeng Chen, Changhao Guan, Zhenrui Yue, Xinyu
Wang, Jinan Xu, and Kaiyu Huang. 2025. Boosting Data Utilization for Multi-
lingual Dense Retrieval. InProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, Christos Christodoulopoulos, Tanmoy
Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computa-
tional Linguistics, Suzhou, China, 12362–12378. doi:10.18653/v1/2025.emnlp-
main.624
[11] Kaiyu Huang, Fengran Mo, Xinyu Zhang, Hongliang Li, You Li, Yuanchi Zhang,
Weijian Yi, Yulong Mao, Jinchen Liu, Yuzhuang Xu, et al .2026. A survey on
large language models with multilingualism: Recent advances and new frontiers.
Artificial Intelligence Review(2026).
[12] Andreea Iana, Goran Glavaš, and Heiko Paulheim. 2024. MIND Your Language:
A Multilingual Dataset for Cross-lingual News Recommendation. InProceedings
of the 47th International ACM SIGIR Conference on Research and Development
in Information Retrieval(Washington DC, USA)(SIGIR ’24). Association for
Computing Machinery, New York, NY , USA, 553–563. doi:10.1145/3626772.
3657867
[13] Can Iscan, Muhammet Furkan Ozara, and Akhan Akbulut. 2024. Enhancing RAG
Pipeline Performance with Translation-Based Embedding Strategies for Non-
English Documents. In2024 Innovations in Intelligent Systems and Applications
Conference (ASYU). 1–6. doi:10.1109/ASYU62119.2024.10756977
[14] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong
Wang, Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs
to Reason and Leverage Search Engines with Reinforcement Learning.
arXiv:2503.09516 [cs.CL] https://arxiv.org/abs/2503.09516
[15] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi
Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al .2024.
Training language models to self-correct via reinforcement learning.arXiv preprint
arXiv:2409.12917(2024).
[16] Somnath Kumar, Vaibhav Balloli, Mercy Ranjit, Kabir Ahuja, Sunayana Sitaram,
Kalika Bali, Tanuja Ganu, and Akshay Nambi. 2025. Bridging the Language Gap:
Dynamic Learning Strategies for Improving Multilingual Performance in LLMs.
InProceedings of the 31st International Conference on Computational Linguistics,
Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di
Eugenio, and Steven Schockaert (Eds.). Association for Computational Linguistics,
Abu Dhabi, UAE, 9209–9223. https://aclanthology.org/2025.coling-main.619/
[17] Dawn Lawrie, Efsun Kayi, Eugene Yang, James Mayfield, Douglas W. Oard,
and Scott Miller. 2025. Generate-Distill: Training Cross-Language IR Models
with Synthetically-Generated Data. InProceedings of the 48th International ACM
SIGIR Conference on Research and Development in Information Retrieval(Padua,
Italy)(SIGIR ’25). Association for Computing Machinery, New York, NY , USA,
2926–2930. doi:10.1145/3726302.3730201
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[19] Bo Li, Zhenghua Xu, and Rui Xie. 2025. Language Drift in Multilingual
Retrieval-Augmented Generation: Characterization and Decoding-Time Mitiga-
tion. arXiv:2511.09984 [cs.CL] https://arxiv.org/abs/2511.09984[20] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu,
Peitian Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic search-enhanced
large reasoning models.arXiv preprint arXiv:2501.05366(2025).
[21] Wei Liu, Sony Trenous, Leonardo F. R. Ribeiro, Bill Byrne, and Felix Hieber. 2025.
XRAG: Cross-lingual Retrieval-Augmented Generation. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2025, Christos Christodoulopoulos,
Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for
Computational Linguistics, Suzhou, China, 15669–15690. doi:10.18653/v1/2025.
findings-emnlp.849
[22] Shayne Longpre, Yi Lu, and Joachim Daiber. 2021. MKQA: A Linguistically
Diverse Benchmark for Multilingual Open Domain Question Answering.Trans-
actions of the Association for Computational Linguistics9 (2021), 1389–1406.
doi:10.1162/tacl_a_00433
[23] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating
Effectiveness of Parametric and Non-Parametric Memories. InProceedings of the
61st Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.).
Association for Computational Linguistics, Toronto, Canada, 9802–9822. doi:10.
18653/v1/2023.acl-long.546
[24] Fengran Mo, Yifan Gao, Sha Li, Hansi Zeng, Xin Liu, Zhaoxuan Tan, Xian
Li, Jianshu Chen, Dakuo Wang, and Meng Jiang. 2026. Agentic Conversational
Search with Contextualized Reasoning via Reinforcement Learning.arXiv preprint
arXiv:2601.13115(2026).
[25] Fengran Mo, Yifan Gao, Chuan Meng, Xin Liu, Zhuofeng Wu, Kelong Mao,
Zhengyang Wang, Pei Chen, Zheng Li, Xian Li, et al .2025. Uniconv: Unifying
retrieval and response generation for large language models in conversations. In
Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 6936–6949.
[26] Fengran Mo, Kelong Mao, Ziliang Zhao, Hongjin Qian, Haonan Chen, Yiruo
Cheng, Xiaoxi Li, Yutao Zhu, Zhicheng Dou, and Jian-Yun Nie. 2025. A survey
of conversational search.ACM Transactions on Information Systems43, 6 (2025),
1–50.
[27] Fengran Mo, Zhan Su, Yuchen Hui, Jinghan Zhang, Jia Ao Sun, Zheyuan Liu,
Chao Zhang, Tetsuya Sakai, and Jian-Yun Nie. 2026. Opendecoder: Open large
language model decoding to incorporate document quality in rag.arXiv preprint
arXiv:2601.09028(2026).
[28] Jeonghyun Park and Hwanhee Lee. 2025. Investigating Language Preference of
Multilingual RAG Systems. InFindings of the Association for Computational
Linguistics: ACL 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and
Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics,
Vienna, Austria, 5647–5675. doi:10.18653/v1/2025.findings-acl.295
[29] Rui Qi, Zhibo Man, Yufeng Chen, Fengran Mo, Jinan Xu, and Kaiyu Huang. 2025.
SoT: Structured-of-Thought Prompting Guides Multilingual Reasoning in Large
Language Models. InFindings of the Association for Computational Linguistics:
EMNLP 2025. 11024–11039.
[30] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin,
Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou,
Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li,
Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang,
Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang,
Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5
Technical Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
[31] Leonardo Ranaldi, Barry Haddow, and Alexandra Birch. 2025. Mul-
tilingual Retrieval-Augmented Generation for Knowledge-Intensive Task.
arXiv:2504.03616 [cs.CL] https://arxiv.org/abs/2504.03616
[32] Leonardo Ranaldi, Federico Ranaldi, Fabio Massimo Zanzotto, Barry Haddow, and
Alexandra Birch. 2025. Improving Multilingual Retrieval-Augmented Language
Models through Dialectic Reasoning Argumentations. InProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing, Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.).
Association for Computational Linguistics, Suzhou, China, 9075–9096. doi:10.
18653/v1/2025.emnlp-main.461
[33] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli,
Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves to use tools.Advances in
Neural Information Processing Systems36 (2023), 68539–68551.
[34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi,
Haowei Zhang, Mingchuan Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
Models. arXiv:2402.03300 [cs.CL] https://arxiv.org/abs/2402.03300
[35] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning
Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented
generation. InProceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 1240–1250.
[36] Zhan Su, Fengran Mo, Jinghan Zhang, Yuchen Hui, Jiaao Sun, and Jian-yun Nie.
2025. Parametric retrieval-augmented generation using latent routing of lora

CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
adapters.arXiv preprint arXiv:2511.17044(2025).
[37] Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and
Xueqi Cheng. 2025. Boosting Retrieval-Augmented Generation with Generation-
Augmented Retrieval: A Co-Training Approach. InProceedings of the 48th Inter-
national ACM SIGIR Conference on Research and Development in Information
Retrieval(Padua, Italy)(SIGIR ’25). Association for Computing Machinery, New
York, NY , USA, 2441–2451. doi:10.1145/3726302.3729907
[38] NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad,
Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht,
Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood,
Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John
Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit,
Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov,
Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn,
Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and
Jeff Wang. 2022. No Language Left Behind: Scaling Human-Centered Machine
Translation. arXiv:2207.04672 [cs.CL] https://arxiv.org/abs/2207.04672
[39] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving retrieval with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of the 61st annual meeting of the
association for computational linguistics (volume 1: long papers). 10014–10037.
[40] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and
Furu Wei. 2024. Multilingual e5 text embeddings: A technical report.arXiv
preprint arXiv:2402.05672(2024).
[41] Wenmin Wang, Peilin Zhang, Ge Liu, Ruihua Wu, and Guixiang Song. 2024. Inves-
tigating on the External Knowledge in RAG for Zero-Shot Cross-Language Trans-
fer. In2024 IEEE 4th International Conference on Electronic Technology, Com-
munication and Information (ICETCI). 1479–1484. doi:10.1109/ICETCI61221.
2024.10594504
[42] Suhang Wu, Jialong Tang, Baosong Yang, Ante Wang, Kaidi Jia, Jiawei Yu,
Junfeng Yao, and Jinsong Su. 2024. Not All Languages are Equal: Insights
into Multilingual Retrieval-Augmented Generation. arXiv:2410.21970 [cs.CL]
https://arxiv.org/abs/2410.21970[43] Diji Yang, Jinmeng Rao, Kezhen Chen, Xiaoyuan Guo, Yawen Zhang, Jie Yang,
and Yi Zhang. 2024. IM-RAG: Multi-Round Retrieval-Augmented Generation
Through Learning Inner Monologues. InProceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval
(Washington DC, USA)(SIGIR ’24). Association for Computing Machinery, New
York, NY , USA, 730–740. doi:10.1145/3626772.3657760
[44] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, Ellen Riloff,
David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for
Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-
1259
[45] Hansi Zeng, Zoey Li, Yifan Gao, Chenwei Zhang, Xiaoman Pan, Tao Yang,
Fengran Mo, Jiacheng Lin, Xian Li, and Jingbo Shang. 2026. SynPlanResearch-
R1: Encouraging Tool Exploration for Deep Research with Synthetic Plans.arXiv
preprint arXiv:2603.07853(2026).
[46] Jinghan Zhang, Fengran Mo, Tharindu Cyril Weerasooriya, Ruimin Dai, Xi-
aoyan Han, Yanjie Fu, Dakuo Wang, and Kunpeng Liu. 2026. StaRPO: Stability-
Augmented Reinforcement Policy Optimization.arXiv preprint arXiv:2604.08905
(2026).
[47] Jinghan Zhang, Xiting Wang, Yiqiao Jin, Changyu Chen, Xinhao Zhang, and Kun-
peng Liu. 2024. Prototypical reward network for data-efficient rlhf. InProceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics (Vol-
ume 1: Long Papers). 13871–13884.
[48] Jinghan Zhang, Xiting Wang, Fengran Mo, Yeyang Zhou, Wanfu Gao, and Kun-
peng Liu. 2025. Entropy-based exploration conduction for multi-step reasoning. In
Findings of the Association for Computational Linguistics: ACL 2025. 3895–3906.
[49] Jinghan Zhang, Xiting Wang, Weijieying Ren, Lu Jiang, Dongjie Wang, and
Kunpeng Liu. 2025. Ratt: A thought structure for coherent and correct llm
reasoning. InProceedings of the AAAI Conference on Artificial Intelligence,
V ol. 39. 26733–26741.