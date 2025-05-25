# SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis

**Authors**: Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, Lei Fang, Zhongyuan Wang, Ji-Rong Wen

**Published**: 2025-05-22 16:05:02

**PDF URL**: [http://arxiv.org/pdf/2505.16834v1](http://arxiv.org/pdf/2505.16834v1)

## Abstract
Retrieval-augmented generation (RAG) systems have advanced large language
models (LLMs) in complex deep search scenarios requiring multi-step reasoning
and iterative information retrieval. However, existing approaches face critical
limitations that lack high-quality training trajectories or suffer from the
distributional mismatches in simulated environments and prohibitive
computational costs for real-world deployment. This paper introduces
SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap
through strategic data engineering rather than complex training paradigms. Our
approach synthesizes high-quality training data by simulating realistic user
interactions in live web search environments, coupled with a multi-criteria
curation strategy that optimizes the diversity and quality of input and output
side. Experiments on five benchmarks across diverse domains demonstrate that
SFT on only 871 curated samples yields significant improvements over RL-based
baselines. Our work establishes SFT as a viable pathway by systematically
addressing the data-scarce bottleneck, offering practical insights for
efficient deep search systems. Our code is available at
https://github.com/RUCAIBox/SimpleDeepSearcher.

## Full Text


<!-- PDF content starts -->

SimpleDeepSearcher: Deep Information Seeking via Web-Powered
Reasoning Trajectory Synthesis
Shuang Sun1*, Huatong Song2*, Yuhao Wang2, Ruiyang Ren2,
Jinhao Jiang2, Junjie Zhang2, Fei Bai3, Jia Deng2,
Wayne Xin Zhao2†, Zheng Liu4, Lei Fang5, Zhongyuan Wang4, Ji-Rong Wen2
1School of Computer Science and Engineering, Northeastern University
2Gaoling School of Artificial Intelligence, Renmin University of China
3School of Informatics, Xiamen University
4Beijing Academy of Artificial Intelligence5DataCanvas Alaya NeW
shuangsun@stumail.neu.edu.cn, batmanfly@gmail.com
Abstract
Retrieval-augmented generation (RAG) sys-
tems have advanced large language mod-
els (LLMs) in complex deep search scenar-
ios requiring multi-step reasoning and iterative
information retrieval. However, existing ap-
proaches face critical limitations that lack high-
quality training trajectories or suffer from the
distributional mismatches in simulated environ-
ments and prohibitive computational costs for
real-world deployment. This paper introduces
SimpleDeepSearcher, a lightweight yet effec-
tive framework that bridges this gap through
strategic data engineering rather than complex
training paradigms. Our approach synthesizes
high-quality training data by simulating realis-
tic user interactions in live web search environ-
ments, coupled with a multi-criteria curation
strategy that optimizes the diversity and quality
of input and output side. Experiments on five
benchmarks across diverse domains demon-
strate that SFT on only 871 curated samples
yields significant improvements over RL-based
baselines. Our work establishes SFT as a viable
pathway by systematically addressing the data-
scarce bottleneck, offering practical insights
for efficient deep search systems. Our code is
available at https://github.com/RUCAIBox/
SimpleDeepSearcher .
1 Introduction
In recent years, retrieval-augmented genera-
tion (RAG) methods have significantly enhanced
LLMs by incorporating external knowledge re-
trieval mechanisms (Zhao et al., 2024; Gao et al.,
2024). Recent advancements have extended these
capabilities to complex deep search scenarios that
demand multi-step reasoning with iterative infor-
mation retrieval and synthesis(Alzubi et al., 2025).
To address the complex reasoning demands in
deep search scenarios, early research explored
*Equal contributions.
†Corresponding author.prompt-based strategies that guide models to de-
compose questions, generate queries, and retrieve
information iteratively (Jiang et al., 2024; Teng
et al., 2025; Li et al., 2025a). Other studies have
attempted to improve model performance through
supervised fine-tuning (SFT) (Wang et al., 2025),
but due to the lack of high-quality interaction tra-
jectories for reasoning and retrieval, SFT methods
often fail to fully activate the model’s autonomous
search abilities (Schick et al., 2023). To further
enhance the model’s autonomous search capabil-
ities, reinforcement learning (RL) (Sutton et al.,
1999) is considered as a promising solution to train
models through real-time interaction with the envi-
ronment (Song et al., 2025; Jin et al., 2025; Zheng
et al., 2025). However, most RL-based approaches
operate within artificial environments using static
document corpora, creating a distributional mis-
match with real-world web dynamics. Moreover,
the inherent computational intensity of RL training
escalates exponentially when interfacing with live
search APIs (Sun et al., 2025).
Given the overhead and complexity of RL-based
training, we posits that SFT remains a viable path-
way for building efficient deep search systems.
While SFT offers a streamlined training process, it
faces the critical challenge of lacking high-quality
training data in deep search scenarios. On the in-
put side, existing QA datasets often lack the di-
versity and complexity of questions and search-
oriented purposes on the Web, which are essential
for deep search training. On the output side, tradi-
tional answer annotations omit the critical reason-
ing traces (search operations, evidence synthesis,
and efficient decision paths) required for teaching
search-integrated reasoning strategies.
In this paper, we propose SimpleDeepSearcher,
an efficient search-with-think framework that uti-
lizes strategic data engineering rather than com-
plex training paradigms. Our core methodology
centers on a three-fold process for constructingarXiv:2505.16834v1  [cs.CL]  22 May 2025

What is the birthdate of …. who plays …and 
who broke the record … that has ever …?
When  holding  … in two hands, what  is … ?
What  award did … receive in 2008?…
…
Where is the location  of 
the newly constructed …?
What is the population  of 
the city in which …?check
add
populationlocation author
birthday
language…Keywords Set
Domain Heterogeneity
Keyword Diversity
Complexity of Knowledge Units checkQuestion 1 Question n
…… S1 S0 S2 S9 S0 S1 S2 S9…
Correct ratios 0.2
 Correct ratios 1
Question Difficulty
… 𝑟0𝑡𝑠𝑞0𝑟1𝑟3 a
… a 𝑟0 𝑟2
… 𝑟0𝑡𝑠𝑞0𝑡𝑒𝑟6 a
… 𝑟0𝑡𝑠𝑞0𝑡𝑒𝑟2 aFormat Standardization
Reasoning Path Control
Search Effectiveness
Retained Solution
S0
S1
S2
S9Response Curation Query SamplingArena Essex Raceway is built 
alongside an out -of-town shopping 
centre  constructed on the site of a 
former what?Okay, let's tackle this question about 
Arena Essex Raceway. The user is 
asking what the site of the out -of-
town … <𝑡𝑠>Arena Essex … <𝑡𝑠>Chalk 
QuarryThe site of the Lakeside Shopping 
Centre, next to which Arena Essex 
Raceway was built, was previously 
used as a chalk quarry. This is 
confirmed by Web Page 7 …Search
Arena Essex Raceway …Reason SummarizationMulti -turn
QuestionInference
Search Data Other DataAnswer
Pure Reasoning Data
…
Supervised Fine -tuning
pr q d r a … 𝑡𝑠 𝑡𝑒
Mask Mask
Reinforcement Learning
Offline
RejectedDPOQuestion
PoolREINFO
RCE++
Top-k
Webpages
ChosenOnline
Qu
est
io
n
Po
ol
Qu
est
io
n
Po
olFigure 1: Overall framework of our proposed SimpleDeepSearcher approach. pdenotes the prompt, rdenotes the
reasoning content, qrepresents the search query, and drefers to the retrieved document after summarization. tsand
teare special tokens indicating the beginning and end of the search query, and adenotes the final answer.
high-quality training data. First, we develop a data
synthesis framework grounded in real web search
environments, simulating realistic user search be-
haviors to generate multi-turn reasoning trajecto-
ries. Second, we propose a diversity-aware query
sampling strategy to optimize domain coverage,
semantic complexity, and knowledge unit density.
Moreover, we adopt a four-dimensional response
curation that enforces format standardization, rea-
soning efficiency, question difficulty, and search
effectiveness. By systematically addressing both
query and response-side quality through automated
pipelines, SimpleDeepSearcher can obtain high-
quality supervised signals based on real web search
for complex reasoning to facilitate the SFT process.
Experimental results show that our SFT method
significantly boosts model performance on five rep-
resentative benchmarks with only 871 high-quality
training samples. Compared to prompt-based meth-
ods, SimpleDeepSearcher achieves a 48.3% im-
provement, and compared to RL-based RAG meth-
ods achieves a 24.9% improvement. This demon-
strates that our framework effectively balances per-
formance and efficiency, providing a simple yet
powerful approach to enhancing deep search ca-
pabilities. Furthermore, our framework is highly
extensible that can be combined with other types
of training data, the framework is also applicable
to RL-based training.Our main contributions are as follows:
•We propose a real web-based data synthesis
framework that simulates realistic user search be-
haviors, generating multi-turn reasoning and search
trajectories.
•We design a multi-criteria data curation strat-
egy that jointly optimizes both input question selec-
tion and output response filtering through orthogo-
nal filtering dimensions.
•Experimental results demonstrate that SFT on
only 871 samples enables SimpleDeepSearcher to
outperform strong baselines (especially RL-based
baselines) on both in-domain and out-of-domain
benchmarks.
2 Method
2.1 Overview
In this section, we propose SimpleDeepSearcher
for complex deep search tasks by leveraging multi-
stage data construction strategies.
To address the resource-intensive limitations
of deep search systems, we propose Sim-
pleDeepSearcher, a framework that achieves in-
telligence search through efficient supervised fine-
tuning (SFT) with minimal training data. For con-
structing high-quality SFT data, we establish a sys-
tematically designed data synthesis and curation
pipeline, as illustrated in Figure 1.

First, we replace static document retrieval with
real-time network interactions, simulating human
search behavior through an iterative cycle of
“reasoning-searching-summarizing-generating.” By
directly processing raw HTML content via com-
mercial search APIs, we capture diverse web in-
formation features—ranging from structured data
snippets to unstructured narrative discourse. Based
on this, we first filter input queries using domain
heterogeneity, keyword diversity, and knowledge
unit complexity to construct a maximally informa-
tive training foundation while ensuring selected
queries align with real-world web search scenar-
ios. Additionally, we apply a filtering mechanism
to LLM-synthesized responses, implementing a
four-dimensional quality filter that simultaneously
optimizes format standardization, reasoning path
control, question difficulty, and search effective-
ness to guarantee response quality.
The framework’s modular design offers three dis-
tinctive advantages over conventional approaches:
First, it exposes the model to authentic search ar-
tifacts and noise patterns through real web inter-
actions. Second, our multidimensional filtering
strategy enables state-of-the-art performance with
remarkably small SFT datasets, eliminating de-
pendency on resource-heavy RL training. Third,
the decoupled architecture between data synthesis
and model constraints provides exceptional flex-
ibility that our curated datasets can enhance any
LLMs while maintaining compatibility with emerg-
ing reasoning architectures and alternative training
paradigms including RL. Since the searched con-
tent is not generated by the LLM, we mask out
these tokens during the SFT process.
Our methodology achieves unprecedented effi-
ciency in search-oriented model training, reducing
computational demands while maintaining com-
petitive performance through strategic data quality
optimization rather than brute-force data quantity.
2.2 Data Synthesis in Real Web Environment
Typically, traditional retrieval-augmented gen-
eration (RAG) systems rely on closed and static
knowledge corpora. Such knowledge corpora ex-
hibit two primary limitations: firstly, the content
they contain often consists of refined and con-
densed segments (Barnett et al., 2024); secondly,
the information within these knowledge corpora
lacks timeliness (Ouyang et al., 2025). Conse-
quently, RAG systems are limited in their ability to
simulate authentic user search behaviors, as userstypically search within open, dynamic, and com-
plex web environments where the information is
not only diverse in format and varied in quality but
is also frequently accompanied by redundancy and
noise. In light of this, our data synthesis approach
does not rely on curated document collections but
is instead grounded in the real, open web environ-
ment. This authentic web environment also places
greater demands on the model’s capabilities for
information extraction, synthesis, and reasoning.
Building upon the widely adopted iterative deep
search process (Li et al., 2025a) of reason-search-
summarize-generate, we develop an automated
pipeline for large-scale training data synthesis. For
each query, our framework at each iteration (1)
initiates web searches via commercial APIs, (2) ex-
tracts and processes raw HTML content, (3) applies
an LLM to reason over multi-source evidence, and
(4) continues for the next iteration or stop iteration.
By sampling multiple reasoning paths per query,
we capture nuanced decision-making processes in-
herent to real-world information synthesis.
Our data synthesis strategy is firmly rooted in
real web scenarios, which substantially enriches the
diversity and authenticity of training samples. Our
strategy generates more practical and representative
supervisory signals for SFT, thereby addressing
critical limitations in conventional SFT paradigms.
2.3 Diversity-aware Query Sampling
To engineer a deep search architecture with ad-
vanced query comprehension and reasoning capa-
bilities, we implement a strategic repurposing of
open-domain question answering (QA) resources.
These curated datasets offer natural language ques-
tions that inherently require multi-hop informa-
tion retrieval operations, thereby exhibiting strong
task alignment with the cognitive demands of deep
search systems. Our selection protocol combines
single-hop and multi-hop QA benchmarks through
principled composition, ensuring coverage of both
atomic and composite reasoning paradigms.
However, empirical evidence suggests that naive
dataset scaling yields diminishing returns in SFT.
The efficacy of such approaches fundamentally de-
pends on the intrinsic diversity and informational
entropy of training instances. While existing open-
domain QA corpora provide substantial volume,
systematic analysis reveals three critical limita-
tions: (1) domain-specific overrepresentation cre-
ating skewed knowledge distributions, (2) repeti-
tive syntactic patterns reducing linguistic variabil-

Algorithm 1 Diversity-aware Query Sampling
Input: Annotated dataset Dwith domains
d1, d2, . . . , d m, target number of queries N
1:Nd←N/m
2:S← ∅ ▷Initialize the target set
3:fori= 1 to ndo
4: Ddi← {x∈D|domain (x) =di}
5: SortDdiby descending interrogative
6: words
7: while|Sdi|< N dandDdi̸=∅do
8: K← ∅ ▷Initialize the keyword set
9: foreach sample xinDdido
10: if|Sdi| ≥Ndthen
11: break
12: end if
13: kw←keywords (x)
14: ifx /∈Sandkw∩K=∅then
15: S←S∪ {x}
16: K←K∪keywords (x)
17: Ddi←Ddi\ {x}
18: end if
19: end for
20: end while
21:end for
22:return S
ity, and (3) semantic simplicity thresholds below
real-world query complexity. These factors collec-
tively induce model brittleness and constrain cross-
domain generalization potential. To address these
critical limitations, we introduce a diversity-aware
query sampling strategy to implement systematic
data filtering through tripartite orthogonal criteria:
Domain Heterogeneity encompasses the system-
atic classification of query semantics across distinct
knowledge domains ( e.g., history, science, politics).
This dimension ensures a balanced distribution of
questions across different domains, thereby reduc-
ing domain-specific biases and enhancing general-
ization capabilities.
Keyword Diversity focuses on the distributional
diversity of core semantic constituents. we ensure
non-redundant exposure to low-frequency concep-
tual entities, multi-order relational dependencies,
and contextually ambiguous referential expressions.
Such systematic variation compels the model to
transcend superficial lexical pattern matching, in-
stead developing reasoning architectures essential
for interpreting complex entity interactions.Complexity of knowledge units captures the fre-
quency of interrogative terms used in questions
(e.g., what, when), which serve as indicators of
syntactic and semantic complexity. Questions with
greater inquiry potential are given priority, ensur-
ing comprehensive modeling of implicit reasoning
chains triggered by diverse question formulations.
We developed a systematic query selection
framework incorporating three complementary di-
mensions: domain heterogeneity, keyword diver-
sity, and complexity of knowledge units. First, we
partition the dataset into domain-specific clusters
using the LLM-generated semantic classifications.
Within each domain cluster, queries are ranked by
knowledge unit complexity scores derived from
conceptual density analysis. Subsequently, we per-
form iterative selection using a greedy algorithm
that maximizes keyword diversity while maintain-
ing inter-domain balance. The detailed procedure
for query sampling is presented in Algorithm 1.
2.4 Multi-Dimention Response Curation
Building upon the aforementioned data synthe-
sis and query sampling strategies, we have success-
fully generated high-quality training data derived
from real-world web environments. However, due
to the inherent unpredictability of LLM reasoning,
the quality of synthesized data exhibits consider-
able variability despite meticulous control over in-
put and generation processes. Three primary issues
are observed: (i) Formatting irregularities, such
as inconsistent reasoning languages, non-standard
formats for search and reasoning steps, and hetero-
geneous answer formats; (ii) reasoning redundancy,
including hypothesis overgeneration, fabricated re-
trieval content, and excessive validation loops; (iii)
inefficient search strategies, including redundant
search exploration, contextual myopia and failure
to retrieve relevant information.
The presence of low-quality reasoning outputs
in language models not only compromises perfor-
mance and transparency but also introduces noise
into training signals, leading to inefficient compu-
tational resource utilization. To address these chal-
lenges, we developed a systematic filtering protocol
that selects optimal solutions through rigorous and
comprehensive evaluation of multiple responses
per query.
To mitigate these issues, we impose strict con-
straints on both the format and content of sam-
pled responses, retaining only those that satisfy all
predefined criteria. Our filtering strategy, struc-

tured around four pillars, ensures retention of high-
quality reasoning data while promoting efficient
search integration.
Format Standardization . Filter out responses
with mixed reasoning languages or incorrect rea-
soning and search formats, and correct answers
with formatting errors to ensure consistency and
standardization across all responses. Responses
exhibiting mixed languages, irregular reasoning
structures, or formatting inconsistencies were ex-
cluded. Automated correction aligned remaining
answers with standardized templates.
Reasoning Path Control . Strictly limit the use of
reflection expressions ( e.g., alternatively, wait, etc.)
and control the length of reasoning to avoid unnec-
essary and redundant reasoning steps. Reasoning
models tend to hypothesize, infer, and reflect based
on internal knowledge, often resulting in delayed
use of search tools and inefficient reasoning. By
regulating the reasoning path, the model can learn
to seamlessly integrate search into its inference pro-
cess and adopt more efficient reasoning strategies.
Question Difficulty . Filter out questions with con-
sistently high accuracy across multiple reasoning
attempts and prioritize those with lower accuracy.
Accuracy obtained from multiple samples can serve
as a proxy for question difficulty. Selecting more
challenging questions helps enhance the model’s
ability to handle complex queries.
Search Effectiveness . Among multiple candidate
responses, prioritize those with fewer search steps
and more diverse search content. This encourages
the model to not only invoke search capabilities
but also to learn how to formulate effective sub-
queries based on the original question for efficient
information retrieval.
Based on the above dimensions, we first collect
metadata for each response, such as the number of
search steps, reasoning length, and accuracy. Sub-
sequently, responses are filtered sequentially based
onformat standardization andreasoning path con-
trol. Then, based on question difficulty , questions
with high accuracy are removed. For each remain-
ing question, we retain multiple high-quality re-
sponses that meet all constraints and sort them by
search steps. According to search effectiveness , the
response with the fewest search steps is selected
as the final answer. Through this process, we ulti-
mately obtained 871 high-quality question-answerpairs. This multi-criteria approach not only en-
hances model training efficiency but also provides
insights into optimal human-AI reasoning patterns.
3 Experiments
3.1 Experimental Setup
Datasets. We sample training data from single-
hop and multi-hop knowledge-intensive QA
datasets to cover a wide range of domains and ques-
tion difficulty. For single-hop questions, we use
Natural Questions (Kwiatkowski et al., 2019) and
SimpleQA (Wei et al., 2024). For multi-hop ques-
tions, we use HotpotQA (Yang et al., 2018), 2Wiki-
MultiHopQA (Ho et al., 2020), MuSiQue (Tang
and Yang, 2024), and MultiHopRAG (Tang and
Yang, 2024). To test the model’s performance on
out-of-domain data, we select Bamboogle (Press
et al., 2022), FRAMES (Krishna et al., 2024), and
GAIA (Mialon et al., 2023) as extra test sets. These
datasets are not used during training and help eval-
uate how well the model works on new domains.
We evaluate our approach on 500 randomly sam-
pled instances from the validation sets of Hot-
potQA, 2WikiMultiHopQA, and MuSiQue. For
Bamboogle and FRAMES, we use their full test
sets. For GAIA, we use 103 examples from the
text-only validation subset (Li et al., 2025b).
Metrics. We report results using two metrics: F1
score and LLM-as-Judge (LasJ). The F1 score cap-
tures the word-level similarity between the pre-
dicted and golden answers, while LasJ leverages
GPT-4o-mini to evaluate the correctness of the pre-
dicted response.
Baselines. We consider following type of base-
lines: Naive Generation : Direct generation of
answers without retrieval. Standard RAG (Zhao
et al., 2024): Directly retrieve relevant documents
by querying the original question. Search-o1 (Li
et al., 2025a): Encourages the model to perform
self-initiated retrieval using prompts. RAG-RL :
R1-Searcher (Song et al., 2025) and DeepRe-
searcher (Zheng et al., 2025), the open-source
7B model trained with reinforcement learning to
enable self-initiated retrieval. We conduct ex-
periments using the following model backbones
with an online search engine, including Qwen-2.5-
7B-Instruct, Qwen-2.5-32B-Instruct, Deepseek-
Distilled-Qwen-2.5-32B, and QwQ-32B.
Implementation Details. Our experimental
setup consists of four main components: query

Models Methods2Wiki†MuSiQue†Bamboogle‡Frames‡GAIA‡
F1 LasJ F1 LasJ F1 LasJ F1 LasJ F1 LasJ
Qwen-7BDirectly Gen 27.7 26.8 9.6 6.2 18.2 17.6 12.6 10.1 13.6 6.8
Standard RAG 34.8 34.8 17.2 14.6 31.5 31.2 13.9 13.5 - -
Search-o1 48.0 51.2 21.5 20.6 57.9 59.2 30.9 35.0 24.3 21.4
R1-Searcher 63.4 66.4 29.0 26.8 68.2 68.8 34.4 40.3 24.1 20.4
DeepResearcher 59.7∗66.6∗27.1∗29.3∗71.0∗72.8∗- - - -
SimpleDeepSearcher 70.6 79.8 28.2 29.4 74.5 76.8 44.9 55.3 39.3 36.9
Qwen-32BDirectly Gen 31.7 31.2 13.3 12.4 25.7 25.6 15.6 14.2 18.6 13.9
Standard RAG 43.7 45.0 19.5 16.8 40.8 40.8 19.4 19.4 - -
Search-o1 64.9 74.8 29.1 30.6 74.4 78.4 47.2 56.8 36.5 34.0
SimpleDeepSearcher 71.9 81.2 30.6 33.0 78.1 80.0 50.1 60.8 42.1 40.8
DDQ-32BDirectly Gen 36.9 36.2 19.6 16.0 32.6 32.8 27.8 29.2 14.8 9.7
Standard RAG 48.1 50.0 24.0 21.6 42.6 46.4 26.5 28.9 - -
Search-o1 49.6 55.2 25.4 23.8 65.7 68.0 32.2 38.7 23.2 24.3
SimpleDeepSearcher 69.0 77.4 32.9 33.6 80.5 83.2 52.2 63.8 42.0 41.7
QwQ-32BDirectly Gen 39.6 39.8 18.9 17.4 29.6 29.6 28.1 31.3 16.8 11.7
Standard RAG 48.4 50.6 21.8 19.4 42.5 46.4 27.4 31.6 - -
Search-o1 69.4 78.0 34.3 36.4 78.7 78.4 51.6 64.4 38.3 37.9
SimpleDeepSearcher 75.6 84.4 34.8 37.4 83.4 88.0 56.8 68.8 48.9 50.5
Table 1: Performance comparisons between SimpleDeepSearcher and the baselines on QA benchmarks. The best
results are in bold and the second-best are underlined .†/‡represents in-domain/out-domain datasets. Results
marked with * are cited from their official paper or report. Qwen-7B ,Qwen-32B ,DDQ-32B are the abbreviations of
Qwen-2.5-7B-Instruct, Qwen-2.5-32B-Instruct, and Deepseek-Distilled-Qwen-2.5-32B, respectively.
sampling, data synthesis, generation, and SFT.
During query sampling, we used QwQ-32B
to annotate each query with its corresponding
domain and keywords. For data synthesis, we
employed QwQ-32B as the reasoning model and
Google Search API as the search engine, with a
maximum of 10 search calls and 15 reasoning
turns per query. For each query, we sampled 10
candidate responses. For generation, all models
are configured with a maximum sequence length
of 20,480 tokens, temperature of 0.6, top -p of
0.95, and top -k of 40. In the SFT phase, we use
a total batch size of 64 and train for 6 epochs
with a learning rate of 1e -5, warmup ratio of 0.03,
and a sequence length of 30,000 tokens. During
fine-tuning, external retrieval documents are
masked to avoid learning from noisy or spurious
information. All prompts used in the experiments
are provided in Appendix E.
3.2 Main Results
Table 1 presents the main results of the proposed
SimpleDeepSearcher and baselines across five rep-
resentative datasets.
Firstly, SimpleDeepSearcher consistently out-
performs all existing baseline methods across five
benchmark datasets. Specifically, it achieves the
best performance not only on in-domain datasets
(i.e., 2Wiki, MuSiQue) but also shows substan-
tial improvements on out-of-domain datasets ( i.e.,Category MethodBamboogle GAIA
F1 LasJ F1 LasJ
Ours 74.5 76.8 39.3 36.9
Query Samplingw/o DH 69.7 70.4 35.6 35.8
w/o KD 73.2 76.0 32.9 31.1
w/o CKU 71.7 74.4 32.1 29.1
Environment w/o Online 74.0 74.4 30.4 28.2
Response Curationw/o FS 72.8 75.2 38.0 36.9
w/o RPC 71.7 74.4 31.6 30.1
w/o QD 67.1 70.4 32.9 32.0
w/o SE 72.6 73.6 37.7 35.0
Table 2: Results of variants of SimpleDeepSearcher on
Bamboogle and GAIA.
Bamboogle, FRAMES, GAIA), demonstrating its
strong generalization ability.
Besides, SimpleDeepSearcher consistently out-
performs reinforcement learning-based methods
such as R1-Searcher and DeepResearcher across
most evaluation metrics. These approaches are
trained on large-scale datasets using complex re-
inforcement learning algorithms. In contrast, our
method relies on supervised fine-tuning with only
871 training examples. This demonstrates that
our framework achieves strong performance while
maintaining high data efficiency, offering a simple
yet effective alternative for improving deep search
capabilities.
Thirdly, SimpleDeepSearcher achieves stable
and substantial performance improvements across

MethodBamboogle GAIA
F1 LasJ F1 LasJ
Distilled (Ours) 74.5 76.8 39.3 36.9
w. DPO 75.0 79.2 39.0 37.9
w. Reinforce++ 73.8 75.8 29.4 24.3
Table 3: Evaluation Results of RL-based Methods.
Model #Alternatively #Search Output Length
QwQ-32B 7.933 2.390 867.148
QwQ-32B-SFT 4.051 2.329 581.731
Table 4: Statistical analysis of model outputs.
models with diverse backbones and parameter
scales, ranging from 7B to 32B. For instance, com-
pared to Search-o1, it achieves relative improve-
ments of 48.3%, 42.6%, and 11.5% on Qwen2.5-
7B-Instruct, DeepSeek-R1-Distill-Qwen-2.5-32B,
and QwQ-32B, respectively. This demonstrates the
strong generalization ability of our distillation and
self-distillation strategies, with the selected data
consistently leading to performance gains across
heterogeneous model architectures.
4 Further Analysis
4.1 Ablation Study
To validate the effectiveness of the proposed
SimpleDeepSearcher, we conduct a comprehen-
sive ablation analysis using Qwen2.5-7B-Instruct
on the Bamboogle and GAIA datasets. We conduct
detailed ablation studies on three main aspects: (1)
Query Sampling: w/o DH removes domain hetero-
geneity filter, w/o KD removes keyword diversity
filter, w/o CKU removes complexity of knowledge
units filter; (2) Environment: w/o Online uses local
dense dense retrieval to synthesize training data;
(3) Response Curation: w/o FR removes format
regularization filter, w/o RPC removes reasoning
path control filter, w/o QD removes question diffi-
culty filter, w/o SC search count filter. As shown
in Table 2, all ablated variants exhibit a decline
in performance compared to our full method, un-
derscoring the integral contribution of each com-
ponent. Among them, w/o QD leads to the most
significant performance drop, suggesting that ques-
tion difficulty plays a crucial role in training. More
challenging questions are more likely to stimulate
the model’s autonomous retrieval capabilities dur-
ing reasoning.Model Plan. Search Summ.
Qwen-7B 0.416 0.455 0.363
Qwen-7B-SFT 0.590 0.677 0.584
QwQ-32B 0.623 0.680 0.594
QwQ-32B-SFT 0.629 0.713 0.624
Table 5: Proportion of instances containing the correct
answer at each stage of the inference process (Planning,
Search, and Summarization), before and after SFT.
4.2 Effect of Post-SFT RL
Recent studies have investigated the integration
of RL and RAG (Song et al., 2025; Jin et al., 2025;
Zheng et al., 2025). We further examine the advan-
tages and limitations of applying RL after SFT.
We apply DPO and REINFORCE++ to conduct
offline and online reinforcement learning, respec-
tively. As shown in Table 3, the model trained
with DPO achieves further improvements over the
SFT baseline, demonstrating the effectiveness of
offline preference optimization. In contrast, the
model trained with REINFORCE++ produces sig-
nificantly shorter responses (see Appendix C) and
shows notable performance degradation on both the
Bamboogle and GAIA benchmarks. This suggests
that online RL mainly triggers retrieval behavior,
but brings little benefit to models that are already
good at retrieval. We hypothesize that the suc-
cess of offline DPO stems from its ability to lever-
age high-quality trajectories generated by a strong
LLM. These trajectories provide informative pref-
erence signals and stable supervision, allowing the
model to refine its reasoning and search strategies.
4.3 Response Redundancy
In this part, we analyze how SFT impacts redun-
dant reasoning and search behavior. Specifically,
we focus on three indicators: (1) the frequency of
the reflective word “alternatively”, which signals
hesitation or divergent reasoning; (2) the average
length of reasoning chains, measured by output
length; and (3) the number of search calls made dur-
ing inference. Our analysis is based on the QwQ-
32B model, evaluated on the 2Wiki, MuSiQue, and
Bamboogle datasets. As shown in Table 4, the
average use of “alternatively” and the overall out-
put length are both significantly reduced after SFT.
Moreover, the model issues fewer search queries.
These results indicate that our self-distillation ap-
proach improves both the reasoning clarity and
search efficiency of the model. This improvement

ModelsSummarization
ModelsBamboogle GAIA
F1 LasJ F1 LasJ
Qwen-7B-SFTbefore training 70.8 71.2 28.0 26.2
after training 67.5 68.8 23.9 21.4
QwQ-32B 74.5 76.8 39.3 36.9
GPT-4o-mini 70.9 76.8 33.7 32.0
QwQ-32B-SFTbefore training 83.5 88.0 48.9 50.5
after training 83.9 86.4 43.2 47.6
GPT-4o-mini 80.0 80.8 40.5 44.7
Table 6: Performance comparison across two bench-
marks using different summarization models.
Training DataBamboogle GAIA AIME
F1 LasJ F1 LasJ F1 LasJ
- Reasoning 74.5 76.8 39.3 36.9 13.3 13.3
+ Reasoning 76.9 80.8 37.2 37.9 20.0 20.0
Table 7: Results of the SimpleDeepSearcher trained w/
and w/o reasoning data across three benchmarks.
can be attributed to the high-quality training data
selected through our proposed method.
4.4 Stage-wise Analysis
In this part, we analyze how training improves
the performance of each sub-task in our approach,
including iterative search, planning, and summa-
rization. We evaluate the proportion of cases in
which the final answer appears during each sub-
process to quantify the efficiency of that stage.
To eliminate interference from the summarization
stage, all summarization models are kept identical
during inference, with detailed settings provided
in Appendix D. The results are shown in Table 5.
We can observe substantial improvements across
all components, with the search component show-
ing the most significant gain. This suggests that
training effectively enhances the model’s ability
to generate more coherent reasoning and search
trajectories, leading to more accurate information
retrieval and improved overall model performance.
4.5 Effect of Summarization Model
This part investigates the impact of the summa-
rization model on overall performance. We fix the
reasoning model and conduct a comparative analy-
sis of overall performance using different summa-
rization models. As shown in Table 6, QwQ-32B
demonstrates the strongest summarization capa-
bility and is therefore selected as the summariza-
tion model for all reasoning models. Furthermore,
using fine-tuned models for summarization leads
to performance degradation on downstream tasks
Figure 2: Average reasoning length across three bench-
marks w/ and w/o reasoning data for training.
Training DataSearch Count
Bamboogle GAIA AIME
- Reasoning 1.552 1.757 0
+ Reasoning 1.672 1.845 0
Table 8: Average search count across three benchmarks
of the model trained w/ and w/o reasoning data.
compared to their pre-trained counterparts. This
might be attributed to the reduced long-text summa-
rization ability of the fine-tuned models, due to the
distributional shifts on a limited task and domain of
the training data. This decline is more pronounced
for models with fewer parameters.
4.6 Effect of Additional Reasoning Data
We further investigate the impact of incorpo-
rating complex mathematical reasoning data on
Qwen2.5-7B-Instruct. As shown in Table 7, this
leads to consistent performance gains across all
benchmarks. Furthermore, Figure 2 and Table 8
reveals significant alterations in the model’s behav-
ioral patterns on two kinds of tasks: for tasks em-
phasizing complex reasoning ( e.g., AIME, GAIA),
the model generates longer and more in-depth rea-
soning outputs; for search tasks ( e.g., Bamboogle),
the model performs more searches and explores
more thoroughly. These findings suggest that incor-
porating complex reasoning data helps the model
learn to adapt its reasoning and search strategies to
the specific demands of a task. This adaptability is
critical for addressing complex and diverse queries.
5 Conclusion
In this work, we present SimpleDeepSearcher, a
lightweight yet effective framework for deepsearch
tasks, addressing the limitations of existing RAG
methods that rely heavily on complex training

paradigms or suffer from distributional mismatches.
By leveraging realistic web search simulations and
a multi-criteria data curation strategy, we construct
high-quality training trajectories that enable effi-
cient supervised fine-tuning. Despite using only
871 curated samples, our method achieves substan-
tial gains over RL-based baselines across diverse
in-domain and out-of-domain benchmarks. Our
results highlight the potential of strategic data engi-
neering to empower deep search reasoning.
Limitation
Despite our substantial efforts, this work is sub-
ject to two limitations stemming. Due to limitations
in training resources and hardware, we conducted
distillation training on 7B and 32B models. In fu-
ture work, we plan to train and evaluate our frame-
work on larger-scale models ( i.e.,72B) to further
verify its generalization capability and robustness.
Additionally, because of the inherent difficulty in
synthesizing multi-hop data, the original data used
for distillation primarily consisted of relatively sim-
ple multi-hop questions. If more realistic and chal-
lenging multi-hop queries can be synthesized in the
future, applying our framework for filtering and
training may yield even better performance.
References
Salaheddin Alzubi, Creston Brooks, Purva Chiniya,
Edoardo Contente, Chiara von Gerlach, Lucas Irwin,
Yihan Jiang, Arda Kaz, Windsor Nguyen, Sewoong
Oh, and 1 others. 2025. Open deep search: Democ-
ratizing search with open-source reasoning agents.
arXiv preprint arXiv:2503.20201 .
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024 . OpenReview.net.
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu,
Zach Brannelly, and Mohamed Abdelrazek. 2024.
Seven failure points when engineering a retrieval
augmented generation system. In Proceedings of
the IEEE/ACM 3rd International Conference on AI
Engineering-Software Engineering for AI , pages 194–
199.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. arXiv preprint arXiv:2011.01060 .
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. arXiv preprint
arXiv:2403.14403 .
Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie
Wang, Wayne Xin Zhao, Yang Song, and Tao Zhang.
2024. Rag-star: Enhancing deliberative reasoning
with retrieval augmented verification and refinement.
CoRR , abs/2412.12881.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-r1:
Training llms to reason and leverage search engines
with reinforcement learning. CoRR , abs/2503.09516.
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha,
and Jinwoo Shin. 2024. Sure: Summarizing re-
trievals using answer candidates for open-domain QA
of LLMs. In The Twelfth International Conference
on Learning Representations .
Satyapriya Krishna, Kalpesh Krishna, Anhad Mo-
hananey, Steven Schwarcz, Adam Stambler, Shyam
Upadhyay, and Manaal Faruqui. 2024. Fact,
fetch, and reason: A unified evaluation of
retrieval-augmented generation. arXiv preprint
arXiv:2409.12941 .
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research. Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025a. Search-o1: Agentic search-enhanced
large reasoning models. CoRR , abs/2501.05366.
Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yu-
tao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng
Dou. 2025b. Webthinker: Empowering large rea-
soning models with deep research capability. arXiv
preprint arXiv:2504.21776 .
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference ef-
ficiency of large language models. In Proceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing , pages 6342–6353, Singa-
pore. Association for Computational Linguistics.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf,
Yann LeCun, and Thomas Scialom. 2023. Gaia: a
benchmark for general ai assistants. In The Twelfth
International Conference on Learning Representa-
tions .
Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan,
Yucong Luo, Jiaying Lin, and Qi Liu. 2025. Hoh: A
dynamic benchmark for evaluating the impact of out-
dated information on retrieval-augmented generation.
arXiv preprint arXiv:2503.04800 .
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2022. Measuring
and narrowing the compositionality gap in language
models. arXiv preprint arXiv:2210.03350 .
Ruiyang Ren, Yuhao Wang, Junyi Li, Jinhao Jiang,
Wayne Xin Zhao, Wenjie Wang, and Tat-Seng Chua.
2025. Holistically guided monte carlo tree search
for intricate information seeking. arXiv preprint
arXiv:2502.04751 .
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves
to use tools. In Advances in Neural Information Pro-
cessing Systems 36: Annual Conference on Neural
Information Processing Systems 2023, NeurIPS 2023,
New Orleans, LA, USA, December 10 - 16, 2023 .
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. Enhanc-
ing retrieval-augmented large language models with
iterative retrieval-generation synergy. arXiv preprint
arXiv:2305.15294 .
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
CoRR , abs/2503.05592.
Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan
Hou, Yong Jiang, Pengjun Xie, Fei Huang, and Yan
Zhang. 2025. Zerosearch: Incentivize the search
capability of llms without searching. arXiv preprint
arXiv:2505.04588 .
Richard S Sutton, Andrew G Barto, and 1 others. 1999.
Reinforcement learning. Journal of Cognitive Neuro-
science , 11(1):126–134.
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries. arXiv preprint arXiv:2401.15391 .
Fengwei Teng, Zhaoyang Yu, Quan Shi, Jiayi Zhang,
Chenglin Wu, and Yuyu Luo. 2025. Atom of
thoughts for markov llm test-time scaling. arXiv
preprint arXiv:2502.12018 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrievalwith chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings of the
61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) , pages
10014–10037.
Liang Wang, Haonan Chen, Nan Yang, Xiaolong
Huang, Zhicheng Dou, and Furu Wei. 2025.
Chain-of-retrieval augmented generation. CoRR ,
abs/2501.14342.
Jason Wei, Nguyen Karina, Hyung Won Chung,
Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John
Schulman, and William Fedus. 2024. Measuring
short-form factuality in large language models. arXiv
preprint arXiv:2411.04368 .
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models. Advances
in neural information processing systems , 35:24824–
24837.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren
Wang, Yunteng Geng, Fangcheng Fu, Ling Yang,
Wentao Zhang, and Bin Cui. 2024. Retrieval-
augmented generation for ai-generated content: A
survey. CoRR , abs/2402.19473.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025.
Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments. arXiv
preprint arXiv:2504.03160 .

A Related Work
Retrieval-Augmented LLMs. To improve the fac-
tual precision of LLM-generated texts (Zhao et al.,
2024), researchers enhance LLMs with retrieval-
augmented generation (RAG) (Guu et al., 2020).
Various approaches have been proposed, such
as branching-based methods (Kim et al., 2024),
summarization-based methods (Li et al., 2023), and
adaptive retrieval techniques (Jeong et al., 2024).
With the increase in model parameters, LLMs have
demonstrated chain-of-thought reasoning capabil-
ities (Wei et al., 2022), and many researchers to
integrated such reasoning with RAG via prompt
engineering (Shao et al., 2023; Trivedi et al., 2023).
Other studies have attempted to distill retrieval
abilities into smaller models through supervised
fine-tuning (Asai et al., 2024). However, these ap-
proaches limit the model’s capacity with a fixed
reasoning path.
Enhancing LLMs with Search. Recently, several
deep search frameworks are proposed (Ren et al.,
2025; Alzubi et al., 2025). They integrate large
language models with search engines in a more
flexible and dynamic manner. Search-o1 (Li et al.,
2025a) simulates deep search in LLMs through
prompt engineering, allowing them to retrieve in-
formation independently during multi-step reason-
ing. R1-Searcher (Song et al., 2025) and Search-
R1 (Jin et al., 2025) equip large language models
with retrieval tools and train them end-to-end us-
ing reinforcement learning. This approach effec-
tively enhances the model’s ability to interleave
reasoning with retrieval during inference. However,
due to the inherent complexity of RL and its high
computational demands, conducting large-scale ex-
periments on full-sized LLMs remains challeng-
ing. SimpleDeepSearcher synthesizes high-quality
training data via broad query sampling and precise
filtering, enabling strong deep search performance
with minimal training cost.
B DPO Detailed Settings
Our objective was to identify answer trajec-
tories that were both correct and demonstrated
efficient reasoning and search paths. To this
end, we construct preference pairs (Rw, Rl),
where Rwdenotes the preferred trajectory and
Rlthe rejected one. We repurpose our previ-
ously established pipeline for query sampling
and data synthesis. During the data synthesis
Figure 3: Changes in Sequence Length and Reward
During REINFORCE++ Training.
stage, we generate responses using the strongest
SFT-trained model, SimpleDeepSearcher-QwQ-
32B-SFT, and the target model to be optimized,
SimpleDeepSearcher-Qwen-7B-SFT. Responses
generated by SimpleDeepSearcher-QwQ-32B-SFT
that pass both the formatting and reasoning path
control checks are treated as chosen examples,
while those generated by SimpleDeepSearcher-
Qwen-7B-SFT that fail these checks are treated
as rejected examples. Ultimately, we construct a
dataset consisting of approximately 875 training
pairs.
For Direct Preference Optimization (DPO) train-
ing, we utilize a learning rate of 5×10−7, aβof
0.1, training for 5epochs with a batch size of 256,
a warm-up ratio of 0.1, and a maximum sequence
length of 10000 .
C REINFORCE++ Detailed Settings
To construct the reinforcement learning (RL)
dataset, we utilized the model that had been trained
though SimpleDeepSearcher to perform rollout
sampling on the training sets of 2Wiki and Hot-
potQA. For each question, eight candidate re-
sponses were generated. From this pool, we se-
lected 2480 samples corresponding to questions
with one to six correct answers, ensuring diversity
in the RL training data.
The reward function employed in REIN-
FORCE++ consists of two components: an answer
reward and a format penalty. The answer reward
is calculated as the F1 score between the predicted
answer and the reference answer, providing a di-
rect measure of response accuracy. In addition, a
discrete format penalty of −2is applied if any of
the following undesirable behaviors are detected:
•Self-Retrieved Content: The model fabricates
content that is not retrieved from external sources.
•Contains Gibberish: The generated output con-

tains nonsensical, irrelevant, or corrupted text seg-
ments.
•Excessive Analytical Markers: The response
contains more than 5 occurrences of phrases such
asAlternatively ,Wait, orHmm , which are treated
as signals of incoherent reasoning.
•Lack of Boxed Answers or Excessive Reason-
ing Length: The model either executes more than 8
retrieval steps or the token length of the analytical
content between any two retrievals exceeds 8,096
tokens.
If none of these conditions are met, no penalty
is applied. To maintain on-policy training through-
out the RL process, we adjusted the batch size to
ensure that learning was based on the most recent
policy rollouts. Figure 3 shows the variations in
response length and reward values observed during
the training process.
D Stage-wise Analysis Settings
We conduct a comparative analysis of Qwe2.5-
7B-Instruct and QwQ-32B before and after training
across the 2Wiki, MuSiQue, and Bamboogle bench-
marks. During inference, we fix the summarization
model to QwQ-32B across all comparisons to elim-
inate potential interference from the summarization
component (the impact of the summarization model
will be further discussed in Section 4.5).
E Instruction Templates
Figure 4 shows the instruction for annotating the
domain and keywords of questions. Figure 5 shows
the instruction for LLM as a judge. Figure 6 shows
the instruction for the reasoning model. Figure 7
shows the instruction for the summarization model.

Instruction for Annotation
You are an advanced semantic analyzer. For the given question, perform the following tasks step by step:
1. **Domain Identification**:
- Determine the broad subject category (domain) this question belongs to.
- Examples: film, history, biology, geography, politics, technology, etc (or any other suitable domain)
2. **Key Point Extraction**:
- Identify 2 -4 core semantic components that are crucial for answering
- Include:
• Key entities (e.g., films, people, locations)
• Critical attributes (e.g., age, duration, population)
• Core relationships (e.g., comparison, causality)
• Measurement dimensions (e.g., time, quantity)
-Exclude filler words and non -essential descriptors \n
**Output Requirements**:
- Use JSON format: {{"domain": "...", " key_points ": [...]}}
- Keep key_points  concise (1 -2 words each)
- Use lowercase for all outputs
- Separate multiple key_points  with commas \n
**Examples**:
Question: "Which film whose director is younger, Charge It To Me or Danger: Diabolik ?"
Output: {{"domain": "film", " key_points ": ["director", "age "]}}\n
**Now process this question:**
{{Question}}Figure 4: Instruction for annotation.
Instruction for LLM as Judge
Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it 
fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct 
and False otherwise.
Golden Answer may have multiple options, and matching any one of them is considered correct. \n
Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}
Figure 5: Instruction for LLM as a judge.

Instruction for Reasoning Model
You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. 
You have special tools: \n\n
- To perform a search: write <| begin_search_query |> your query here <| end_search_query |>.\n
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format 
<|begin_search_result |> ...search results... <| end_search_result |>.\n\n
Whenever you encounter a topic, fact, or piece of information you are uncertain about or need further details on, please 
perform a search to gather more accurate, up -to-date, or specific information. You can repeat the search process multiple 
times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}. \n\n
Once you have all the information you need, continue your reasoning. \n\n
Remember: \n
- Use <| begin_search_query |> to request a web search and end with <| end_search_query |>.\n
- When done searching, continue your reasoning. \n
- Do not generate <| begin_search_result |> and <| end_search_result |> tags yourself. \n\n
Please answer the following question. You should think step by step to solve it. \n\n
Provide your final answer in the format \\boxed{YOUR_ANSWER}. \n\n
Question: \n{question} \n\nFigure 6: Instruction for reasoning model.

Instruction for Summarization Model
**Task Instruction:** \n\n
You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, 
**Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for 
**Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the 
**Previous Reasoning Steps** to continue reasoning for the original question. \n
**Guidelines:** \n
1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
-Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for 
the original question. \n
2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning 
Steps**
- Ensure that the extracted information is accurate and relevant. \n
3. **Output Format:**
- Present the helpful information for current search query: beginning with `**Final Information**` as shown below.
**Final Information** \n
[Helpful information] \n
**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning }\n
- **Current Search Query:**  
{search_query }\n
- **Searched Web Pages:**  
{document} \n
Now you should analyze each web page and find helpful information based on the current search query "{ search_query }" 
and previous reasoning steps.Figure 7: Instruction for summarization model.