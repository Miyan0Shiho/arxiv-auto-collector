# SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA

**Authors**: Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu Zhang, Zibin Zheng

**Published**: 2025-12-09 17:58:36

**PDF URL**: [https://arxiv.org/pdf/2512.08867v1](https://arxiv.org/pdf/2512.08867v1)

## Abstract
The Development Knowledge Question Answering (Dev Knowledge QA) task aims to provide natural language answers to knowledge-seeking questions during software development. To investigate its importance and to what extent it has been explored, we analyze real user-LLM dialogues from WildChat and find that: (1) The Dev Knowledge QA task accounts for 39.6% of interactions(highest among all tasks), revealing broad knowledge needs beyond code generation (32.3%). (2) Only 27.5% of real Dev Knowledge QA dialogues focus on code understanding, leaving out development knowledge-seeking. (3) Only 17.1% of real-world Dev Knowledge QA dialogues can be used for constructing a benchmark. Existing benchmarks have two primary limitations for evaluating the Dev Knowledge QA capability of LLMs. First, existing benchmarks offer a limited development knowledge scope, mainly focusing on code understanding and neglecting broader knowledge during development. Second, some benchmarks are not built from real user queries. To bridge this gap, we design a three-phase pipeline that transforms real-world dialogue into simple development knowledge-seeking QA pairs. Through this pipeline, we introduce SimpleDevQA, a multilingual benchmark derived from real user dialogues. It contains 2,740 QA pairs in three languages (English, Chinese, and Russian), and focuses on questions with unique, short, and verifiable answers for accurate and simple evaluation. Experiments show that: Code LLMs generally outperform general LLMs of similar scale; Knowledge injection with the Retrieval-Augmented Generation (RAG) strategy can boost LLM accuracy by 11.3% on average; LLMs show systematic overconfidence in Dev Knowledge QA, and the answering accuracy of LLMs shows a positive correlation with their stated confidence; Generally, LLMs with stronger code generation performance also exhibit stronger performance in Dev Knowledge QA.

## Full Text


<!-- PDF content starts -->

SimpleDevQA: Benchmarking Large Language Models on
Development Knowledge QA
JING ZHANG,Sun Yat-Sen University, China
LIANGHONG GUO,Sun Yat-Sen University, China
YANLIN WANG∗,Sun Yat-Sen University, China
MINGWEI LIU,Sun Yat-Sen University, China
JIACHI CHEN,Sun Yat-Sen University, China
YUCHI MA,Huawei Cloud Computing Technologies, China
ENSHENG SHI,Huawei Cloud Computing Technologies, China
TERRY YUE ZHUO,Monash University, Australia
HONGYU ZHANG,Chongqing University, China
ZIBIN ZHENG,Sun Yat-Sen University, China
The Development Knowledge Question Answering (Dev Knowledge QA) task aims to provide accurate natural
language answers to knowledge-seeking questions during software development. To investigate the importance
of Dev Knowledge QA in AI-assisted software development scenarios and to what extent it has been explored,
we conduct a preliminary study on 1 million real user-LLM dialogues from WildChat. We discover that: (1) The
Dev Knowledge QA task accounts for 39.6% of user interactions with LLMs (highest among all tasks), revealing
broad knowledge needs beyond code generation (32.3%). (2) Only a small portion of real Dev Knowledge QA
dialogues (27.5%) focus on code understanding, leaving out development knowledge-seeking. (3) Only 17.1% of
real-world Dev Knowledge QA dialogues can be used for constructing a benchmark. Existing benchmarks have
two primary limitations for evaluating the Dev Knowledge QA capability of LLMs. First, existing benchmarks
offer a limited development knowledge scope, mainly focusing on code understanding and neglecting broader
knowledge during development. Second, some benchmarks are not built from real user queries, failing to
reflect genuine developer task demands and query patterns. To bridge this gap, we design a three-phase
pipeline that transforms real-world dialogue into simple development knowledge-seeking QA pairs. Through
this pipeline, we introduce SimpleDevQA, a multilingual Dev Knowledge QA benchmark derived from real
user dialogues. This dataset contains 2,740 QA pairs in three languages (English, Chinese, and Russian), and
focuses on questions with unique, short, and verifiable answers, making evaluation more accurate and simple.
We obtain several findings through extensive experiments on SimpleDevQA with 18 mainstream LLMs: (1)
Closed-source models typically surpass open-source ones, and code LLMs generally outperform general LLMs
∗Corresponding authors
Authors’ Contact Information: Jing Zhang, zhangj777@mail2.sysu.edu.cn, Sun Yat-Sen University, China; Lianghong
Guo, guolh8@mail2.sysu.edu.cn, Sun Yat-Sen University, China; Yanlin Wang, wangylin36@mail.sysu.edu.cn, Sun Yat-Sen
University, China; Mingwei Liu, liumw26@mail.sysu.edu.cn, Sun Yat-Sen University, China; Jiachi Chen, chenjch86@mail.
sysu.edu.cn, Sun Yat-Sen University, China; Yuchi Ma, mayuchi1@huawei.com, Huawei Cloud Computing Technologies,
China; Ensheng Shi, shiensheng@huawei.com, Huawei Cloud Computing Technologies, China; Terry Yue Zhuo, terry.
zhuo@monash.edu, Monash University, Australia; Hongyu Zhang, hyzhang@cqu.edu.cn, Chongqing University, China;
Zibin Zheng, zhzibin@mail.sysu.edu.cn, Sun Yat-Sen University, China.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXX
, Vol. 1, No. 1, Article . Publication date: December 2025.arXiv:2512.08867v1  [cs.SE]  9 Dec 2025

2Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
of similar scale. (2) Knowledge injection with the Retrieval-Augmented Generation (RAG) strategy can boost
LLM accuracy by 11.3% on average, enabling smaller models to achieve performance comparable to larger
ones. (3) LLMs show systematic overconfidence in Dev Knowledge QA, and the answering accuracy of LLMs
shows a positive correlation with their stated confidence. (4) Generally, LLMs with stronger code generation
performance also exhibit stronger performance in Dev Knowledge QA.
CCS Concepts:•Software and its engineering→Software development techniques.
Additional Key Words and Phrases: Benchmark, Large Language Models, Development Knowledge QA
ACM Reference Format:
Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo,
Hongyu Zhang, and Zibin Zheng. 2025. SimpleDevQA: Benchmarking Large Language Models on Development
Knowledge QA. InProceedings of Make sure to enter the correct conference title from your rights confirmation
email (Conference acronym ’XX).ACM, New York, NY, USA, 24 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
The Development Knowledge Question Answering (Dev Knowledge QA) task aims to address
knowledge-seeking questions posed during the software development process by providing accurate
and relevant natural language answers. Effectively responding to these diverse, knowledge-seeking
questions is crucial for boosting developer productivity and enhancing software quality. With the
rapid advancements and increasing integration of LLMs into developer workflows, Dev Knowledge
QA has emerged as a critical area for research and evaluation [ 11,15,18,35,44,53,58,60,77,110].
To investigate the importance of Dev Knowledge QA in AI-assisted software development
scenarios, we conduct a preliminary study on WildChat [ 106], a dataset including one million
user–ChatGPT [ 90] conversations. Considering the large scale and topic diversity of WildChat, we
filter and sample it to obtain WildChat-Dev-Lite, a software development-related subset of 8,786
conversations used for our preliminary study. Through preliminary studies on the WildChat-Dev-
Lite dataset, we obtain three findings based on preliminary study research questions (PSRQs):
•The Dev Knowledge QA task accounts for the highest proportion (39.6%) of all tasks observed
during user interactions with LLMs, showing the importance of Dev Knowledge QA task.
•In real Dev Knowledge QA dialogues, only 27.5% of dialogues focus on code understanding.
•Only 17.1% of real Dev Knowledge QA dialogues could be used for constructing a benchmark.
Through the preliminary study, we identify the importance of constructing a Dev Knowledge
QA benchmark. Although there are existing Dev Knowledge QA benchmarks [ 11,35,44,53,58,60],
they have the following two key problems:
P1: Limited Development Knowledge Scope.Existing Dev Knowledge QA benchmarks
mainly focus on code understanding [ 11,35,44,53,58,60] and do not cover other knowledge
related to the development process. For instance, CS1QA [ 44] evaluates code understanding pri-
marily within programming education scenarios. Cases are shown in Figure 1. However, based
on PSRQ2 from our preliminary study, practical software development demands a much broader
understanding, encompassing knowledge of underlying systems, databases, network protocols,
algorithmic principles, etc.
P2: Not built from real user queries.Some existing Dev Knowledge QA benchmarks [ 58,60]
are not built from real user queries from real-world development scenarios, which can not reflect the
real demands of developers. For example, RepoQA [ 60] only evaluates a model’s code understanding
ability on long code context tasks, but this fails to fully reflect genuine developer task demands
and query patterns in real development scenarios.
In this paper, we propose SimpleDevQA, a Dev Knowledge QA benchmark built from real devel-
oper dialogues, to address the aforementioned problems. Based on the findings in our preliminary
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 3
question: “I wrote a Fibonacci function and ran 
it, but the Fibonacci sequence not input, but 
only appeared. It comes out as [], not None.”
answer: "It looks like the direction of the 
inequality sign in the while statement is wrong.”
questionType : “logical”QA Pair
“def fibonacci (upper bound):
i=0
k=1
fibo_list =[]
while i>upper_bound :
fibo_list.append (i)
fibo_list.append (k)
i=i+k
k=i+k
return fibo_list
print( fibonacci (1000))”Input Codequestion: “In Java, what is the default maximum age of a 
cookie if not set explicitly with setMaxAge ()?"
answer: “-1.”
language: “English”QA Pair
(a) Case in CS1QA (b) Case in SimpleDevQAquestion: “ 在使用 WIN32 API 的Windows 文件处理中，当
FindFirstFileA 找不到文件或目录时，它会返回什么？ ”
answer: “ INVALID_HANDLE_VALUE.”
language: “ Chinese”QA Pair
question: “Каков режим дизайна в разработке игр, который 
включает в себя создание многоразовых компонентов, 
позволяющих динамически взаимодействовать, например, 
прыгать или перенаправлять объекты в игре? ”
answer: “Component Pattern .”
language: “Russian ”QA Pair
Fig. 1. Case comparison between CS1QA and SimpleDevQA (CS1QA focuses on code-specific QA, while
SimpleDevQA addresses knowledge-seeking QA beyond code snippets)
study, we design and implement a three-phase data construction pipeline to convert real-world
SE-related conversations into a Dev Knowledge QA benchmark. When constructing the benchmark,
we focus only on software development knowledge-seeking questions that have a single correct
answer, which helps us avoid the open-endedness of language models, following prior work [ 31,88].
This is important because it makes measuring factuality much easier, though it may leave out
questions that could have multiple valid responses [25, 102].
To build a high-quality benchmark, we collect 3,000 real user conversations from WildChat as a
seed dataset. We then process the dataset through a three-stage pipeline. First,in the Reference
Collection stage, we extract core software engineering topics from real-world dialogues. Using
these topics, we then retrieve relevant reference documents from the web, ensuring that the
generated QA pairs are both factually grounded and faithful to the original user intent. Second,
in the QA-pair Generation stage, the original real-world dialogues and their purified reference
documents are jointly provided to an LLM, which is guided to produce QA pairs. This design
ensures that the constructed pairs are both realistic and reliable. Third,in the QA-pair Filtering
stage, we implement a rigorous multi-layer pipeline: (1) An LLM is used to remove low-quality
questions, and an additional LLM with Retrieval-Augmented Generation (RAG) is employed to
verify the factual correctness of answers; (2) To increase difficulty, four powerful LLMs filter out
QA pairs deemed too easy; (3) Human annotators perform verification to ensure the overall quality
and correctness of the remaining QA pairs.
Through our construction pipeline, we finally build the SimpleDevQA benchmark, which con-
tains 2,740 Dev Knowledge QA pairs and covers three languages (English, Chinese, and Russian).
The benchmark focuses on questions with single, correct answers, making evaluation more accurate
and simple. As shown in Figure 1, each data consists of a question inquiring about development
knowledge; a single, unambiguous, and correct reference answer; and the language used. Addition-
ally, each QA pair is also accompanied by multiple web-retrieved references, which can be used to
verify the factual accuracy of the answer. Following prior works on factual benchmarks [ 31,88],
, Vol. 1, No. 1, Article . Publication date: December 2025.

4Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
Table 1. Comparison of existing Dev Knowledge QA benchmarks.
Benchmark Year Size Data Source Built from Real
User QueriesDiverse Dev
Knowledge
CodeQA 2021 190,000 Code Snippets% %
CS1QA 2022 9,237 Textbooks, Edu Materials! %
CodeRepoQA 2024 585,687 Real Code Repositories! %
RepoQA 2024 500 Real Code Repositories% %
InfiBench 2024 234 Real questions! %
CoReQA 2025 1,563 Real issues and comments! %
SimpleDevQA 2025 2,740 Real User Dialogues, Web! !
we evaluate SimpleDevQA by prompting a separate judge LLM with the model’s prediction and
the reference answer to verify correctness.
We conduct extensive experiments on SimpleDevQA with existing mainstream LLMs: (1) We
evaluate the performance of 18 mainstream LLMs on SimpleDevQA, and the results show that
LLMs’ performance in Dev Knowledge QA varies considerably, with closed-source models typically
surpassing open-source models, and code-specific LLMs generally outperform general-purpose
models of similar scale. (2) We conduct experiments to evaluate whether the knowledge injection
strategy [ 46] can improve the performance of LLMs. The experimental results demonstrate that
the knowledge injection strategy can improve LLMs’ accuracy in Dev Knowledge QA and can
enable smaller LLMs to achieve performance comparable to that of larger LLMs. (3) According to
previous studies on factuality benchmarks [ 31,41,88], we can evaluate the correlation between the
confidence of LLMs and their accuracy on SimpleDevQA to assess whether they “know what they
know”, thereby helping developers make better decisions about when to trust or validate the output
of LLMs. The results show that LLMs’ accuracy generally increases with their stated confidence in
Dev Knowledge QA [ 85]. (4) Based on the results of our preliminary study, we find that users place
the highest demands on LLMs’ code generation ability and understanding of software knowledge.
Therefore, we conduct experiments to analyze the relationship between these two abilities in LLMs.
The results reveal that LLMs with stronger code generation performance also exhibit stronger
performance in Dev Knowledge QA, where o3-mini and DeepSeek-R1 show the best comprehensive
performance. Besides, both code generation and Dev Knowledge QA performance improve as the
model scale increases.
We summarize the main contributions as follows:
•We conduct a preliminary study to investigate the importance of Dev Knowledge QA in
AI-assisted software development scenarios.
•We propose SimpleDevQA, a Dev Knowledge QA benchmark derived from real developer
dialogues, to assess LLMs’ understanding capability of software development knowledge in
real programming scenarios.
•We construct a data construction pipeline that can convert real-world conversations into a
Dev Knowledge QA benchmark.
•We conduct extensive experiments on SimpleDevQA with existing mainstream LLMs, thereby
providing a comprehensive evaluation of their performance to understand and apply software
development knowledge in real-world scenarios.
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 5
2 Related Work
2.1 Dev Knowledge QA Benchmarks
In recent years, there are many previous works evaluating the abilities of LLMs in different software
engineering fields such as code generation [ 4,12,13,29,32,86,105,107,108], commit message
generation [ 80,81,84,103], code summarization [ 39,75,96], code search [ 17,19,24,49,76,93],
issue resolution [ 27,28,38,94,95], etc. To evaluate the ability of LLMs in Dev Knowledge QA,
numerous Dev Knowledge QA benchmarks have emerged in recent years.
For example, CodeQA [ 58] emphasizes syntax understanding, API usage, and basic logic analysis
at the code snippet. CS1QA [ 44] adopts an educational perspective, extracting QA pairs gathered
from chat logs in an introductory programming class. CoReQA [ 11] focuses on evaluating a model’s
ability to answer relevant development questions based on code review history, including code
changes constructed from GitHub issues and comments. As the first repository-level comprehension
benchmark, CodeRepoQA [ 35]is a multi-turn QA benchmark, which contains contextual questions
but remains limited to basic scenarios like code retrieval and dependency analysis. RepoQA [ 60] is a
benchmark to evaluate LLMs on long-context code understanding, designed to test a model’s ability
to understand code, dependencies, and project structure at the code repository level. InfiBench [ 54]
is the first large-scale freeform QA benchmark for code, built from high-quality Stack Overflow
questions. The comparison between the SimpleDevQA and other Dev Knowledge QA benchmarks
can be found in Table 1. These Dev Knowledge QA benchmarks are mainly designed around
questions grounded in a given code snippet. These benchmarks primarily assess LLMs’ ability to
code understanding and do not cover other knowledge related to the development process.
2.2 Real-World Software QA Datasets
There are numerous software QA datasets that have been constructed by extracting real developer
interactions from online QA communities such as Stack Overflow [ 5] and GitHub [ 42] platforms [ 8,
16,87]. For example, the StaQC [ 100] focuses on matching natural language questions with relevant
code snippets, and the work by Wu et al. [ 91] also demonstrates how API knowledge can be
retrieved from Stack Overflow to construct datasets based on QA pairs from Stack Overflow. The
TechQA corpus [ 8] consists of real user questions from a technical forum, rather than questions
generated specifically for a competition or a task. According to PSRQ3 in our preliminary study
described in Section 3, these real QA datasets often result in verbose answers, may contain subjective
opinions, or provide multiple solutions, making it difficult to directly and effectively assess LLMs’
performance on the QA task.
2.3 Factuality Benchmarks
In recent years, factual evaluation of large-scale language models has garnered significant academic
attention, leading to the development of several representative benchmarks [ 14,36,40,50,79,
99,105,111]. MMLU [ 33] systematically assesses models’ breadth of knowledge and reasoning
capabilities, TruthfulQA [ 56] employs carefully designed adversarial questions to specifically assess
whether models generate answers that sound reasonable but are actually false (i.e., “hallucinations”).
For systematic hallucination evaluation, HallEval [ 51] establishes a standardized framework to
quantitatively measure hallucination across QA, summarization, and dialogue tasks. Additionally,
OpenAI recently released SimpleQA [ 88], a dataset of 4,326 concise fact-seeking questions that en-
able reliable factual accuracy assessment. Following this, Chinese SimpleQA [ 31] was subsequently
proposed to evaluate factual performance in Chinese LLMs, containing 3,000 high-quality questions
across six major themes. However, current factual benchmarks remain limited to general domains,
leaving professional fields like software development without dedicated evaluation frameworks.
, Vol. 1, No. 1, Article . Publication date: December 2025.

6Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
WildChatModel
ClassificationLLM 
Verification
SE-related DataStratified 
Sampling
WildChat -Dev WildChat -Dev-Lite
Fig. 2. Data processing workflow for extracting software development–related conversations.
3 Preliminary Study
In this section, to investigate the importance of the Dev Knowledge QA task in real-world develop-
ment scenarios, we conduct a preliminary study on WildChat [ 106], a dataset including one million
user–ChatGPT conversations.
Given the diverse topics in WildChat’s dialogues, we extract software development-related
dialogues for our research. Following WildChat [ 106], we use a prompt-classification model [ 83] to
filter the dataset and obtain 376,888 dialogues related to software engineering. To further enhance
data quality, we employ Llama3-Instruct-8B [ 64] to process the dialogues, filtering out conversations
irrelevant to software development. This step results in the WildChat-Dev dataset, which includes
103,112 dialogues focused on development. To facilitate our manual annotation and analysis of
real dialogues, we apply stratified sampling to 103,112 dialogues based on user language, dialogue
length, and the interacted model within the data, which results in the WildChat-Dev-Lite dataset.
To compute the sample size, we follow previous studies [ 97,98], use the random sampling method
based on the confidence interval [ 1]. We set a confidence interval of 1 and a confidence level of 95%,
and compute that the sample size is 8,786 [ 2]. At the end, we obtain the WildChat-Dev-Lite dataset,
which includes 8,786 real dialogues. The whole data processing workflow is shown in Figure 2.
Based on the WildChat-Dev-verified dataset, we design the following preliminary study research
questions (PSRQs):
•PSRQ1:What is the importance of Dev Knowledge QA tasks in real-world development scenar-
ios?
•PSRQ2:How well can existing benchmarks evaluate real Dev Knowledge QA capabilities?
•PSRQ3:What challenges arise when constructing a Dev Knowledge QA benchmark from real
user dialogues?
3.1 PSRQ1: Importance Analysis of Dev Knowledge QA
To investigate the importance of Dev Knowledge QA, we analyze the distribution of Dev Knowledge
QA tasks in real-world development scenarios when users utilize LLMs. First, we use the open card
sorting method [ 45] to summarize the tasks involved in the dialogues from WildChat-Dev-Lite, and
the final eight categories are inspired by prior works [ 34,65,104], as shown in Table 2. Then, we
manually filter dialogues from WildChat-Dev-Lite that are unrelated to software development and
classify the remaining dialogues into 8 task categories. Specifically, we first conduct a pilot study
in which three annotators jointly label a small sample of dialogues and discuss the criteria. During
the annotation process, two annotators independently label each dialogue. They then cross-check
the results and resolve discrepancies through discussion. If no agreement can be reached, a third
annotator is involved, and final decisions are made via majority voting. To assess annotation
reliability, we calculate Krippendorff’s alpha [ 43], which is 0.908, indicating strong agreement. The
final average annotation accuracy is 0.924. Finally, we perform a statistical analysis on the classified
WildChat-Dev-Lite dataset in the 3.
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 7
Table 2. Description of software engineering-related tasks in real scenarios
Task Category Task Description
Dev Knowledge QA QA that questions knowledge during development
Code Generation Generating code examples based on requirements
Code Debugging Identifying errors in existing code
Program Repair Fixing defective or buggy code
Code Translation Converting code from source to target language
Code Editing Modifying or enhancing functionality of existing code
Comment Generation Generating explanatory comments for code
Test Generation Creating test cases for code
As illustrated in the Figure 3 below, we find that the Dev Knowledge QA task (39.6%) accounts
for the highest proportion among all tasks, surpassing the code generation task (32.3%), illustrating
the high popularity of the Dev Knowledge QA task in real-world development scenarios.
PSRQ1 Summary:The Dev Knowledge QA task accounts for the highest proportion (39.6%)
of all tasks observed during user interactions with LLMs, showing the importance of the Dev
Knowledge QA task.
3.2 PSRQ2: Topic Analysis of Dev Knowledge QA
Since previous Dev Knowledge QA benchmarks primarily focus on code understanding [ 35,44,
58,60], we analyze whether existing benchmarks can fully evaluate real Dev Knowledge QA
capabilities. Through manual annotation of 3,098 Dev Knowledge QA dialogues, we find that only
851 (approximately 27.5%) of these dialogues involved questions centered on specific code snippets,
programming language details, code usage, or debugging issues. In contrast, the remaining 2,870
dialogues (approximately 72.5%) consisted of queries seeking factual information about broader
development topics, such as system design principles, operational procedures, underlying principles,
and environment configuration, etc. The result demonstrates that in real-world development
scenarios, Dev Knowledge QA tasks not only involve questions centered around code but also
encompass inquiries about other aspects of development knowledge. Thus, existing Dev Knowledge
QA benchmarks can not fully evaluate real Dev Knowledge QA capabilities.
PSRQ2 Summary:Existing Dev Knowledge QA benchmarks mainly focus on code under-
standing, which accounts for only a small portion of real interactions (27.5%), leaving out other
abilities such as development knowledge-seeking.
3.3 PSRQ3: Challenge Analysis of Constructing Dev Knowledge QA Benchmark
We further investigate whether these real-world dialogues related to development can be used
for constructing a Dev Knowledge QA benchmark. We randomly sample 1,000 dialogues from the
Dev Knowledge QA dialogues to facilitate manual verification. Then, we manually verify each
Dev Knowledge QA instance against three critical criteria proposed by previous research [ 31,88]:
(1) the question must have a single answer, (2) reference answers should not change over time,
and (3) reference answers must be supported by evidence. The results reveal that only 17.11% of
these real-world dialogues could be used for constructing a Dev Knowledge QA benchmark. Other
dialogues may lack a single definitive answer. For instance, a user’s question,“ When using SVN to
, Vol. 1, No. 1, Article . Publication date: December 2025.

8Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
Dev Knowledge QA
39.60%
Code Generation
32.30%Code Edit
10%Program Repair :6.80%Debugging :8.90%Code Translation :1.70%Code Summarization :0.40%Test Generation :0.30%
Dev Knowledge QA
Code Generation
Code Edit
Program Repair
Debugging
Code Translation
Code Summarization
Test Generation
Fig. 3. The distribution of SE tasks in WildChat-Dev-Lite dialogues.
Search 
Web Pages
Seed 
ConversationPhase I. Reference Collection
Phase III. QA -pair Filtering
Quality
VericationDifficulty 
StratificationHuman 
Verification
Reference 
Web PagesReference 
Content
Extract
References
QA pairs With 
Reference AnswerGenerate 
QA pairs
: LLM
 : search tool
Phase II. QA -pair Generation
SimpleDevQA
 Verified
QA pairsHard
QA pairs
Fig. 4. SimpleDevQA Construction Pipeline.
modify files in Linux, how to prevent others from modifying them at the same time?” can have
multiple valid or context-dependent answers. This finding reveals the necessity of constructing a
dedicated benchmark to automatically evaluate LLM performance on Dev Knowledge QA tasks in
real-world development scenarios.
PSRQ3 Summary:Only 17.1% of real-world Dev Knowledge QA dialogues can be used for
constructing a benchmark.
4 SimpleDevQA Benchmark Construction
In this section, we design a three-stage benchmark construction framework aimed at converting
real-world SE-related conversations into a Dev Knowledge QA benchmark. An overview of our
pipeline is illustrated in Figure 4. The framework consists of three stages: the Reference Collection
stage, the QA-pair Generation stage, and the QA-pair Filtering stage.
Our preliminary study in Section 3 shows that most real-world Dev Knowledge QA dialogues are
unsuitable for benchmark construction. First, many answers in the dialogues are incorrect and thus
cannot serve as reference answers. Second, even when answers are correct, the associated questions
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 9
often have multiple valid responses, making it difficult to define a single reference answer. To
overcome these problems, we adopt a Retrieval-Augmented Generation (RAG) approach to construct
high-quality QA pairs from real-world dialogues. Specifically, our method first leverages an LLM
to perform a web search based on the user dialogue and retrieve relevant documents containing
factual evidence related to the dialogue. We then extract information from these documents to
guide the LLM in generating QA pairs. By combining real user questions with externally verified
content, we generate QA pairs whose answers are grounded in factual evidence and suitable for
use as reference answers in evaluation.
In our data generation process, we randomly sample 3,000 real-world conversations from
WildChat-Dev to serve as the seed dataset. These seed datasets provide diverse and represen-
tative developer dialogues, which are used as the initial input for constructing our Dev Knowledge
QA benchmark.
4.1 Reference Collection
To ensure that the generated QA pairs are grounded in factual evidence while remaining faithful to
the original user intent, we collect high-quality reference documents for each real-world developer
dialogue in this stage.
Starting from the seed dataset, each dialogue is analyzed by GPT-4o-mini [ 67] to identify core
software engineering topics and then generate search queries that capture the main intent and
technical focus of the dialogue. These queries are submitted to the Google Search API [ 70] to retrieve
relevant web page URLs. For each dialogue, we collect 10 candidate web pages. After collecting
web pages related to each dialogue, we use the Goose3 [ 21] tool to remove irrelevant content from
the collected web pages. Finally, we collect 30,000 reference documents for the QA-pair Generation
stage.
4.2 QA-pair Generation
Once the reference documents have been collected, this stage is aimed at generating Dev Knowledge
QA pairs using the original real-world dialogue and its corresponding reference documents. The
motivation for this design is to ensure both the realism and reliability of the constructed QA pairs.
On the one hand, using actual developer dialogues helps preserve the authenticity of the questions
and ensures that they reflect real concerns in software development. On the other hand, reference
documents provide factual support for generating accurate and verifiable answers.
The generation process begins with the preprocessing of the reference content. For documents
that are too long to be directly used as input for the LLM due to context window limits or irrelevant
information, we first apply the GPT-4o-mini [ 66] to extract their main content. Then, all reference
documents linked to the same dialogue are combined into a single reference text. This unified
reference text, along with the original real-world dialogue, is fed into an LLM to produce candidate
QA pairs. Here, we use GPT-4o [ 67] to generate Dev Knowledge QA pairs. The prompts for
generating QA pairs are presented in Figure 5. Here, we request that LLM generate 3 QA pairs for
one dialogue at the same time. Besides, we provide 10 demonstrations to enhance the quality of
generated Dev Knowledge QA pairs from LLMs. Finally, we generate 9,000 Dev Knowledge QA
pairs from the initial 3,000 Dev Knowledge dialogues.
4.3 QA-pair Filtering
In this stage, we design a QA-pair Filtering pipeline to improve the quality of SimpleDevQA.
Quality Verification.To improve the quality of SimpleDevQA, we separately verify the quality
of both questions and answers. First, to improve the quality of questions in SimpleDevQA, we
employ an LLM to filter out low-quality questions. Specifically, following previous studies[ 31],
, Vol. 1, No. 1, Article . Publication date: December 2025.

10Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
Prompt Template for QA -pair Generation
###Task Description###
You areanexpert software programmer .Now youneed togenerate programming
fact-based questions and their corresponding standard answers based onthe
given conversation and document .The QA pairs must meet the following
requirements :
1.**Relevance toSoftware Engineering** :The questions must relate toconcrete
programming knowledge orconcepts insoftware engineering, such asalgorithms,
data structures, design patterns, libraries, APIs, orbest practices .Avoid subjective
oropinion -based questions .
2.**Language Consistency** :The language used inthegenerated question -
answer pairs must beconsistent with thelanguage used intheconversation .
3.**Clarity and Precision** :Each question should have one and only one clear
and undisputed answer .Avoid ambiguous oropen -ended questions .
4.**Timeless Answers** :The answers must betimeless and not subject to
change .Avoid questions about roles, events that may change over time.
5.**Conciseness** :Answers should beasconcise aspossible while remaining
accurate .
6.**Educational Value** :Questions should have acertain level ofdifficulty to
pose achallenge and beeducational forlearning software engineering concepts .
Thegenerated questions cannot beeasily answered correctly
###Output Format###
[Format]
###Examples ofHigh -Quality QAPairs###
[Example]
###Input###
Here isthegiven conversation :{conversation} ,document :{document}
Fig. 5. The prompt template for generating QA pairs.
we use three core standards for QA pair quality verification: (1) Each question must have exactly
one unambiguous and uncontroversial answer, avoiding ambiguous or open-ended questions. (2)
Questions should exhibit sufficient complexity to evaluate LLMs’ depth of understanding of software
engineering concepts. (3) All questions must have verifiable answers based on publicly available
information by December 31, 2024. In this step, we apply GPT-4o-mini [ 66] to eliminate samples
that fail to meet any of these criteria, resulting in 6,020 QA pairs. Second, to improve the quality of
answers, we implement an LLM with Retrieval-Augmented Generation(RAG) [ 46] system to verify
the factual correctness of answers. For our RAG system, we utilize LlamaIndex [ 59] to build the
search tool and select Google Search [ 70] as the search engine. This tool is used to search related
web pages as references for subsequent answer verification. Here, we use GPT-4o-mini [ 66] with
our RAG system to classify the QA pairs with incorrect answers. Then, we manually correct the
answers of these QA pairs.
Difficulty Filtering.To increase the difficulty of SimpleDevQA, we employ LLMs to filter out
low-difficulty QA pairs. Specifically, following previous research [ 31], we use four LLMs with strong
general factual capability: GPT-4o [ 67], Llama-3-70B-Instruct [ 22], Qwen2.5-72B-Instruct [ 72] and
GLM-4-Plus [ 6]. If a question could be correctly answered by all four strong models, it is considered
a easy question. After this step, we obtain 3231 difficult QA pairs and 2,789 easy QA pairs.
Human Verification.Finally, we conduct human verification to further enhance the quality of
SimpleDevQA. We follow the criteria in Quality Verification to verify the data quality. Each QA pair
is evaluated independently by two annotators. They first check whether it meets the predefined
standards—if either annotator rejects it, the pair is removed. Then, annotators search for supporting
evidence using search engines and provide answers with citations from authoritative sources (at
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 11
least two URLs per answer). If the two annotators disagree, a third annotator reviews the case and
makes the final decision, considering both previous evaluations. The human-annotated answers
are compared with LLM-generated responses, and only QA pairs with full agreement are kept to
ensure high precision and consistency with the standards. Finally, we obtain 2,740 QA pairs.
In summary, throughout the construction and annotation of SimpleDevQA, numerous low-quality
samples are discarded. Initially, we generate 9,000 QA pairs based on 3,000 real conversations.
After Quality Verification, approximately 6,020 pairs (67%) were retained, with 33% discarded.
Subsequently, Difficulty Filtering using multiple models yielded 3,231 hard samples. Finally, rigorous
manual review led to the removal of an additional 491 samples, resulting in a high-quality benchmark
with 2,740 QA pairs.
5 Benchmark Characteristics
SimpleDevQA is a Dev Knowledge QA benchmark designed for the software development domain,
containing 2,740 manually verified QA pairs. Our dataset covers the three most prevalent languages
from the WildChat-Dev-Lite dataset: English, Chinese, and Russian. Specifically, it includes 624
English QA pairs, 1,341 Chinese QA pairs, and 775 Russian QA pairs.
On average, questions in our benchmark have a length of 27.74 tokens1, while answers have a
length of 7.99 tokens, as shown in Table 3. In addition to these, each QA pair is also accompanied by
multiple web-retrieved references, which include corresponding webpage URLs and text snippets
that can be used to verify the factual accuracy of the answers. This benchmark fills the gap in existing
evaluation sets for assessing broad development knowledge in the real world, providing a reliable
tool for evaluating LLMs’ understanding of development knowledge. In summary, SimpleDevQA
includes the following key characteristics:
Built from Real User Queries.The questions in SimpleDevQA are built from real user queries
from real-world development scenarios, which can reflect the real task demands of developers.
Diverse Development KnowledgeSimpleDevQA includes questions spanning a broader range
of development knowledge domains. We manually classify the questions into knowledge domains
using a taxonomy adapted from a previous study [61].
•Syntactic questions:These focus on programming language grammar and API usage, such as
language-specific functions or common library usage, etc.
•Semantic questions:These target more abstract programming concepts, such as algorithms,
data structures, and object-oriented principles, etc.
The classification results, shown in Table 3, indicate that the benchmark contains 2,305 syntactic
questions and over 435 semantic questions, covering 9 distinct development knowledge domains.
By covering a wide range of Dev knowledge, SimpleDevQA can more realistically reflect the
development knowledge QA ability of an LLM.
Multilingualism.SimpleDevQA incorporates three popular languages, including Chinese,
English, and Russian. This dataset can be used to evaluate the multilingual Dev Knowledge QA
capabilities of different LLMs.
Validated Quality.We apply a comprehensive and rigorous three-stage filtering process to
improve the quality of the SimpleDevQA.
Static.In SimpleDevQA, each QA pair’s reference answers are grounded in stable development
knowledge, which remains invariant over time or external changes. Each QA pair is equipped
with reference web documents, which serve as static sources of knowledge. This design ensures
long-term reproducibility for model evaluation,
1The code of computing tokens is from https://github.com/openai/tiktoken.
, Vol. 1, No. 1, Article . Publication date: December 2025.

12Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
Table 3. SimpleDevQA Statistics.
(a) Knowledge Domain.
Category Domain Size
Syntactic knowledgeAPIs & Frameworks 1764
Programming language syntax 541
Semantic knowledgeAlgorithms & Data structures 133
Software Development & Engineering 92
Database Management & SQL 92
Computer organization & Architecture 55
System design 33
Object-oriented programming 27
Compiler design 2(b) Language Usage.
Language Ratio
English 22.78%
Chinese 48.94%
Russian 28.28%
(c) Average Token Length.
QA Length
Question 27.74
Answer 7.99
Easy-to-evaluate.By manual verification, we retain only those questions with single and
unambiguous answers. According to previous studies [ 31,88], this ensures that we can evaluate
the correctness of predicted answers simply and accurately using the LLM-as-a-judge method [ 30,
31, 48, 78, 88, 112].
6 Evaluation Setup
6.1 Evaluation Details
Model Selection.In evaluation, we choose 18 widely used LLMs in both closed-source and open-
source categories. For the closed-source models, we include Claude-3.5-Haiku [ 3], o3-mini [ 68],
GPT-3.5-Turbo[ 7], DeepSeek-V3[ 57], GPT-4o [ 70], and DeepSeek-R1. For the open-source models,
we cover twelve representative models drawn from five major series: the Qwen2.5 series (7B, 14B,
32B) [ 72], the InternLM2.5 series (7B, 20B) [ 82], the Llama3 series (8B, 70B) [ 22], the DeepSeek-
Coder series (6.7B, V2-lite) [ 26], and the Qwen2.5-Coder series (7B, 32B) [ 37]. We also include
CodeLlama-7B-instruct [73] in our open-source evaluation.
Experiment Settings.During LLM inference, we set the temperature to 0.7 and the top-p to 0.95.
During evaluation, we set the temperature of the judge model to 0.5 and top-p to 1. This setting
mitigates the risk of extreme bias from greedy decoding while avoiding excessive noise, thereby
producing judgments that are both consistent and reasonably diverse [ 23,48]. Nevertheless, it may
introduce a certain degree of randomness and potential bias. To mitigate randomness, we conduct
each experiment three times and average the outcomes. All experiments are run on a server under
Ubuntu 20.04.6 LTS, equipped with 128 Intel®Xeon®Platinum 8336C @ 2.30 GHz CPUs and eight
NVIDIA A800 80 GB PCIe GPUs.
6.2 Evaluation Metrics
Grading Method:Following previous studies[ 31,88], we evaluate the correctness of the model’s
predicted answers by prompting a separate judge model with both the prediction and the reference
answer and asking it to assign one of three labels: Correct, Not Attempted, or Incorrect, to each
prediction. And we conduct a human evaluation and measured agreement with the LLM judge. The
Cohen’s Kappa score [ 63] between the LLM and human evaluation reaches 0.83 with an overall
accuracy of 0.91. These results indicate a high level of agreement (Kappa > 0.8), supporting the
high reliability of the LLM judge. Thus, we here use the GPT-4o-mini [66] as a judge model.
Metrics:Following previous studies[ 31,88], we use these metrics to evaluate the performance
of LLMs on SimpleDevQA:
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 13
/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni0000002c/uni00000051/uni00000046/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057
/uni00000026/uni0000002a/uni00000024
/uni00000029/uni00000010/uni00000056/uni00000046/uni00000052/uni00000055/uni00000048/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a/uni00000027/uni00000048/uni00000048/uni00000053/uni00000056/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000059/uni00000016
/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni0000002c/uni00000051/uni00000046/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057
/uni00000026/uni0000002a/uni00000024
/uni00000029/uni00000010/uni00000056/uni00000046/uni00000052/uni00000055/uni00000048/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000026/uni00000052/uni00000047/uni00000048/uni00000055/uni00000010/uni00000039/uni00000015/uni00000010/uni0000002f/uni0000004c/uni00000057/uni00000048/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni0000002c/uni00000051/uni00000046/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057
/uni00000026/uni0000002a/uni00000024
/uni00000029/uni00000010/uni00000056/uni00000046/uni00000052/uni00000055/uni00000048/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001a/uni00000013/uni00000045
/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni0000002c/uni00000051/uni00000046/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057
/uni00000026/uni0000002a/uni00000024
/uni00000029/uni00000010/uni00000056/uni00000046/uni00000052/uni00000055/uni00000048/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a
/uni00000016/uni000000ee/uni00000014/uni00000013/uni00000014
/uni00000052/uni00000016/uni00000010/uni00000050/uni0000004c/uni00000051/uni0000004c
/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni0000002c/uni00000051/uni00000046/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057
/uni00000026/uni0000002a/uni00000024
/uni00000029/uni00000010/uni00000056/uni00000046/uni00000052/uni00000055/uni00000048/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000016/uni00000015/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni0000002c/uni00000051/uni00000046/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057
/uni00000026/uni0000002a/uni00000024
/uni00000029/uni00000010/uni00000056/uni00000046/uni00000052/uni00000055/uni00000048/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000026/uni00000052/uni00000047/uni00000048/uni00000055/uni00000010/uni00000016/uni00000015/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000026/uni0000004b/uni0000004c/uni00000051/uni00000048/uni00000056/uni00000048 /uni00000028/uni00000051/uni0000004a/uni0000004f/uni0000004c/uni00000056/uni0000004b /uni00000035/uni00000058/uni00000056/uni00000056/uni0000004c/uni00000044/uni00000051
Fig. 6. Results of different models between different language subsets.
•Correct (CO):The predicted answer fully includes the reference answer without contradiction.
•Not Attempted (NA):The predicted answer does not fully match the reference answer but also
does not contradict it.
•Incorrect (IN):The predicted answer contradicts the reference answer at any point, even if the
contradiction is later corrected.
•Correct Given Attempted (CGA):The ratio of correct answers among all attempted answers
(including both correct and incorrect responses).
•F-score:The harmonic mean of the overall percentage of correctly answered questions and the
metric “Correct Given Attempted”.
•Average Tokens:The average sum of input and output tokens per question that can evaluate
aspects of LLM’s potential invocation costs.
7 Evaluation
In this section, we conduct experiments on SimpleDevQA to address the following research ques-
tions (RQs):
•RQ1:How do different LLMs perform in SimpleDevQA?
•RQ2:Can knowledge injection improve LLMs’ factual accuracy for software engineering factual
QA tasks?
•RQ3:How does LLMs’ stated confidence correlate with their accuracy on Dev Knowledge QA
tasks?
•RQ4:What is the correlation between an LLM’s code generation performance and its compre-
hension of software development knowledge?
7.1 RQ1: Performance Comparison Analysis
We evaluate 18 mainstream LLMs on SimpleDevQA. The results are presented in the Table 4.
First, we can find that LLMs’ performance on SimpleDevQA varies considerably, and o3-mini and
DeepSeek-R1 perform best on SimpleDevQA, achieving F-scores of 0.719 and 0.7, respectively.
Second, at similar parameter scales, specialized code LLMs significantly outperformed general
, Vol. 1, No. 1, Article . Publication date: December 2025.

14Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
Table 4. Results of different models on difficult QA pairs from SimpleDevQA. NA is short for Not Attempted;
CGA is short for Correct Given Attempted.
Model Type Correct Incorrect NA CGA F-score Avg.
Tokens
Closed-Source Large Language Models
Claude-3.5-Haiku General LLM 0.536 0.324 0.14 0.623 0.576 321.62
o3-mini General LLM 0.718 0.278 0.004 0.721 0.719 229.86
GPT-3.5-Turbo General LLM 0.677 0.307 0.016 0.688 0.682 117.05
DeepSeek-V3 General LLM 0.560 0.432 0.007 0.564 0.562 527.57
GPT-4o General LLM 0.619 0.373 0.008 0.624 0.622 254.81
DeepSeek-R1 General LLM 0.679 0.262 0.059 0.722 0.7 918.02
Open-Source Large Language Models
Qwen2.5-32B-Instruct General LLM 0.543 0.442 0.015 0.551 0.547 309.92
Qwen2.5-14B-Instruct General LLM 0.498 0.489 0.013 0.505 0.501 317.04
Qwen2.5-7B-Instruct General LLM 0.445 0.534 0.021 0.454 0.45 360.53
InternLM2.5-7B-chat General LLM 0.411 0.572 0.017 0.418 0.414 521.66
InternLM2.5-20B-chat General LLM 0.466 0.517 0.017 0.474 0.47 458.14
Llama-3.1-8B General LLM 0.449 0.535 0.016 0.456 0.453 304.89
Llama-3.1-70B General LLM 0.538 0.451 0.011 0.544 0.541 304.82
CodeLlama-7B-
InstructCode LLM 0.389 0.594 0.017 0.396 0.393 403.62
DeepSeek-Coder-V2-
Lite-InstructCode LLM 0.518 0.472 0.01 0.523 0.521 403.87
DeepSeek-Coder-6.7B-
InstructCode LLM 0.511 0.477 0.012 0.517 0.514 269.88
Qwen2.5-Coder-7B-
InstructCode LLM 0.573 0.412 0.015 0.582 0.578 116.9
Qwen2.5-Coder-32B-
InstructCode LLM 0.574 0.412 0.014 0.582 0.578 278.35
LLMs. For instance, while Qwen2.5-32B-Instruct scored only 0.551 on the F-score, Qwen2.5-Coder-
32B-Instruct achieved 0.582. This difference suggests training on large-scale code data can improve
models’ Dev Knowledge QA ability effectively. Third, we find that closed-source models generally
performed better than open-source models, with closed-source models consistently achieving
higher F-scores. For example, the open-source Llama-3.1-70B has an F-score of 0.544, while the
closed-source Claude-3.5-Haiku achieved 0.623. Besides, there is a notable performance gap between
the best-performing closed-source model, o3-mini (F-score=0.719), and the top open-source model,
Qwen2.5-Coder-32B-Instruct (F-score=0.578), indicating that current open-source models still have
room for improvement in understanding professional software development knowledge. Fourth,
we find that models’ performance improves as their scale increases based on many model series,
such as the InternLM2.5 series, the Llama3 series. Finally, we find that Claude-3.5-Haiku achieves
the highest score on the Not Attempt metric among all evaluated LLMs. For questions about which
it is uncertain, it tends to abstain rather than provide an incorrect answer.
We also analyze the performance of six LLMs on the Chinese, English, and Russian subsets
of SimpleDevQA, with the results detailed in Figure 6. From these experimental outcomes, we
identify several key findings: First, o3-mini demonstrates the strongest performance, achieving the
leading score across all three languages. Furthermore, most Chinese-developed models, such as the
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 15
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025 /uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000025 /uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000039/uni00000016 /uni0000002a/uni00000033/uni00000037/uni00000010/uni00000016/uni00000011/uni00000018/uni00000010/uni00000037/uni00000058/uni00000055/uni00000045/uni00000052
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000056/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000026/uni00000052/uni00000047/uni00000048/uni00000055/uni00000010/uni00000039/uni00000015/uni00000010/uni0000002f/uni0000004c/uni00000057/uni00000048/uni00000027/uni00000048/uni00000048/uni00000053/uni00000056/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000026/uni00000052/uni00000047/uni00000048/uni00000055/uni00000010/uni00000019/uni00000011/uni0000001a/uni000000250.00.20.40.60.81.0/uni00000029/uni00000010/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000013/uni00000011/uni00000017/uni00000018 /uni00000013/uni00000011/uni00000017/uni00000018/uni00000013/uni00000011/uni00000018/uni00000019/uni00000013/uni00000011/uni00000019/uni0000001b
/uni00000013/uni00000011/uni00000018/uni00000015 /uni00000013/uni00000011/uni00000018/uni00000014/uni00000013/uni00000011/uni00000019/uni0000001b
/uni00000013/uni00000011/uni00000019/uni00000013 /uni00000013/uni00000011/uni00000019/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000015/uni00000013/uni00000011/uni0000001a/uni00000013/uni00000013/uni00000011/uni00000019/uni00000019/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048
/uni0000005a/uni0000004c/uni00000057/uni0000004b/uni00000003/uni00000035/uni00000024/uni0000002a
Fig. 7. The effect of the knowledge injection strategy.
/uni00000013 /uni00000015/uni00000013 /uni00000017/uni00000013 /uni00000019/uni00000013 /uni0000001b/uni00000013 /uni00000014/uni00000013/uni00000013
/uni00000036/uni00000057/uni00000044/uni00000057/uni00000048/uni00000047/uni00000003/uni00000026/uni00000052/uni00000051/uni00000049/uni0000004c/uni00000047/uni00000048/uni00000051/uni00000046/uni00000048/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001b/uni00000014/uni00000011/uni00000013/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000045
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000016/uni00000015/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000026/uni00000052/uni00000047/uni00000048/uni00000055/uni00000010/uni0000001a/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni0000004c/uni00000051/uni00000057/uni00000048/uni00000055/uni00000051/uni0000004f/uni00000050/uni00000015/uni00000042/uni00000018/uni00000010/uni0000001a/uni00000045/uni00000010/uni00000046/uni0000004b/uni00000044/uni00000057
/uni00000033/uni00000048/uni00000055/uni00000049/uni00000048/uni00000046/uni00000057/uni00000003/uni00000026/uni00000044/uni0000004f/uni0000004c/uni00000045/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051
Fig. 8. Calibration of model outputs using self-
reported stated confidence scores.
/uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013 /uni0000001a/uni00000013
/uni00000029/uni00000010/uni00000056/uni00000046/uni00000052/uni00000055/uni00000048/uni00000017/uni00000013/uni00000018/uni00000013/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000033/uni00000044/uni00000056/uni00000056/uni00000023/uni00000014/uni00000026/uni0000004f/uni00000044/uni00000058/uni00000047/uni00000048/uni00000010/uni00000016/uni00000010/uni00000018/uni00000010/uni0000004b/uni00000044/uni0000004c/uni0000004e/uni00000058/uni00000052/uni00000016/uni00000010/uni00000050/uni0000004c/uni00000051/uni0000004c
/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000016/uni00000011/uni00000018/uni00000010/uni00000057/uni00000058/uni00000055/uni00000045/uni00000052/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000059/uni00000016/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000017/uni00000052/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000055/uni00000014
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000016/uni00000015/uni00000025/uni00000010/uni0000004c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025/uni00000010/uni0000004c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025/uni00000010/uni0000004c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000045/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001a/uni00000013/uni00000045
/uni00000026/uni00000052/uni00000047/uni00000048/uni0000004f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni0000001a/uni00000025/uni00000010/uni0000004c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000027/uni00000048/uni00000048/uni00000053/uni00000056/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000026/uni00000052/uni00000047/uni00000048/uni00000055/uni00000010/uni00000019/uni00000011/uni0000001a/uni00000025/uni00000010/uni0000004c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000026/uni00000052/uni00000047/uni00000048/uni00000055/uni00000010/uni00000016/uni00000015/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057
/uni00000014/uni00000013
/uni00000013/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013
/uni00000035/uni00000044/uni00000051/uni0000004e/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000046/uni00000052/uni00000051/uni00000056/uni0000004c/uni00000056/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni00000056/uni00000046/uni00000052/uni00000055/uni00000048Fig. 9. Results of different LLMs on SimpleDevQA
and HumanEval.
Qwen and DeepSeek series, perform best in Chinese. The result indicates that the model performs
differently across languages.
RQ1 Summary:First, LLMs’ performance in Dev Knowledge QA varies considerably, with
closed-source models o3-mini and DeepSeek-R1 leading. Second, closed-source LLMs typically
surpass open-source counterparts, indicating room for growth in the latter’s understanding of
specialized software development knowledge. Third, code-specific LLMs generally outperform
general-purpose models of similar scale.
7.2 RQ2: Impacts of the Knowledge Injection Strategy
Previous studies [ 47,52,92] have demonstrated that the Retrieval Augmentation Generation [ 46]
(RAG) strategy is one common method to inject factual knowledge into LLMs to improve LLMs’
factual accuracy on general QA tasks. In this section, we investigate its impact on LLMs’ per-
formances in the Dev Knowledge QA task. Following previous studies [ 31], we implement our
, Vol. 1, No. 1, Article . Publication date: December 2025.

16Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
Prompt Template for Confidence Output
###Task Description###
Please provide your best answer tothegiven question and indicate your confidence
intheanswer using ascore from 0to100.Please respond inthefollowing JSON
format :
{
"answer" :"your answer",
"confidence score" :your confidence score
}
###Input###
Given aquestion :{question}
Fig. 10. The prompt for guiding LLM to output confidence.
RAG pipeline using LlamaIndex [ 59] and integrate the Google Search API [ 20] to retrieve relevant
software-engineering knowledge from the web.
The experimental results are presented in the Figure 7. First, we can find that after implementing
the RAG strategy, all evaluated LLMs show significant performance improvements on SimpleDevQA,
with an average improvement of 11.3%. For instance, with the RAG strategy, Llama-3.1-8B’s F-score
increases by 0.145. DeepSeek-V3, despite having the smallest performance gain from RAG, still has
its score rise by 0.033. Second, the RAG approach effectively reduces the performance gap between
different LLMs. For instance, without RAG, the F-score gap between Qwen2.5-7B-instruct and
GPT-3.5-Turbo is as high as 23.2%, whereas, after RAG integration, this gap substantially decreases
to just 3.7%. The result shows that smaller LLMs achieve comparable performance with larger LLMs
after the RAG strategy.
RQ2 Summary:The knowledge injection strategy can improve LLMs’ accuracy in Dev Knowl-
edge QA, and can enable smaller LLMs to achieve performance comparable to that of larger
LLMs.
7.3 RQ3: Stated Confidence and Accuracy Correlation Analysis
According to previous studies [ 31,88], a factual QA benchmark like SimpleQA [ 88] can not only
evaluate the factual accuracy of LLM but also serve as a reliable calibration test, assessing the align-
ment between model confidence and factual accuracy. A well-calibrated LLM enables developers to
judge the trustworthiness of their answers based on their stated confidence. Following the previous
study, we conduct a calibration analysis by directly instructing the LLM to state its confidence after
its answer when responding to a question for the specific prompt, as shown in Figure 10. Then, we
plot the correlation between the LLM’s stated confidence and its actual accuracy. The results are
presented in Figure 8.
As shown in Figure 8, the results indicate that all LLMs’ QA accuracy increases as their confidence
increases, which can boost LLM accuracy by 11.3% on average. This suggests that users can utilize
the prompt shown in Figure 10 to select answers for which LLMs express higher confidence,
potentially leading to more accurate outcomes. However, all evaluated LLMs exhibit a degree
of overconfidence. Specifically, their performance falls well below the Y=X ideal calibration line,
indicating that these LLMs tend to overestimate the accuracy of their answers. These findings
indicate that LLM calibration still requires substantial improvement. When consulting an LLM
for Dev Knowledge, users should not fully trust every answer from LLMs, but can rely more on
answers with higher confidence.
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 17
question: Какое предложение SQL используется для 
группирования строк с одинаковым значением в 
указанном столбце ?
answer: GROUP BYquestion: In bash scripting within a Nix Flake mkDerivation , 
what command would configure Git to set core.ignorecase to 
false?
answer: git config --global  core.ignorecase false
Database Management & SQLAPIs & Frameworksquestion: How to enable Dey Tools project on IntelliJ2021.2 
usiny maven and observe thechanges in code without having 
to restart the Tomcat server ?
answer: Step 1: Install Tomcat in IntelliJ ...  Step 2: Add 
Tomcat Deployment for Your Maven Project ... 
(a) Question examples in prior benchmark (b) Question examples in SimpleDevQAquestion: How can l increase the laravel 8 dd() limitations? l 
have lots of nested relationship and l can't view most of them 
due to many data.
answer: 1. Use config() ... 2. Use dump() ... 3. Use toArray () ... 
multiple valid answerslengthy  and complex answershort and unique  answer
Fig. 11. Case studies contrasting SimpleDevQA with prior benchmarks.
RQ3 Summary:LLMs’ accuracy generally increases with their stated confidence in Dev
Knowledge QA. However, they tend to overestimate the accuracy of their answers.
7.4 RQ4: Capability Correlation Study
Based on our preliminary study in Section 3.1, code generation and Dev Knowledge QA are the
most common tasks when developers interact with LLMs. To investigate how LLM performance
relates across these two tasks, we compare evaluated LLMs on two benchmarks: HumanEval [ 13]
for code generation and SimpleDevQA for Dev Knowledge QA. We gather each model’s HumanEval
pass@12score alongside its SimpleDevQA F-score and plot these paired metrics in a scatter plot.
The results are shown in Figure 9.
Experimental results in Figure 9 demonstrate that LLMs that achieve higher pass@1 scores tend
to also score higher on SimpleDevQA, indicating a strong relationship between code generation
capability and development knowledge comprehension. Among them, o3-mini and DeepSeek-
R1 exhibit outstanding performance on both datasets, demonstrating superior comprehensive
capabilities. However, special cases exist. For example, Qwen2.5-7B-Instruct achieves nearly 80%
Pass@1 on code generation benchmarks, yet its F-score is only about 45%. This gap suggests that
strong code generation ability does not guarantee equally strong performance on Dev Knowledge
QA tasks. Furthermore, we find that both code generation and Dev Knowledge QA performance
improve as the model scale increases. This trend holds true for both the Qwen2.5 series and the
Llama 3.1 series.
RQ4 Summary:Generally, LLMs with stronger code generation performance also exhibit
stronger performance in Dev Knowledge QA, where o3-mini and DeepSeek-R1 show the
best comprehensive performance. Besides, both code generation and Dev Knowledge QA
performance improve as model scale increases.
7.5 Case Study
In the QA pairs we observed, we categorize and analyze cases where SimpleDevQA exhibits
significant differences compared to prior benchmarks [ 11,54]. We identify the following aspects
that highlight the advantages of SimpleDevQA.
Simple and Accurate Evaluation.SimpleDevQA focuses on closed-ended, knowledge-seeking
questions with short, unique answers, enabling simple, accurate, and scalable automated evaluation.
For example, questions in SimpleDevQA directly ask “what”, “which command” rather than “how”, as
shown in Figure 11. This question style dictates that the answers are short and unique. The answers
2We collect pass@1 scores of evaluated LLMs on HumanEval from https://evalplus.github.io/leaderboard.html.
, Vol. 1, No. 1, Article . Publication date: December 2025.

18Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
to the examples above are “GROUP BY” and “git config –global core.ignorecase false” respectively.
This one-to-one correspondence is key to accurate evaluation. In contrast, prior benchmarks [ 54]
often feature open-ended, procedural questions (e.g., starting with “How”) that seek solutions with
multiple valid answers or require lengthy and complex answers. This ambiguity can compromise the
accuracy and consistency of automated evaluation, often necessitating more complex procedures
to ensure a fair assessment. Consequently, assessing such benchmarks often requires complex,
execution-based setups or relies on similarity metrics (e.g., ROUGE [ 55], BLEU [ 69]). The design of
SimpleDevQA avoids these issues, as its concise, factual answers allow for simple and accurate
evaluation using an LLM as a judge.
Broad Development Knowledge Coverage.The questions in SimpleDevQA cover multiple
aspects of software development knowledge. As the examples show in Figure 11, it evaluates
knowledge in Database Management & SQL and APIs & Frameworks, unlike prior benchmarks that
often focus on specific code implementations. This design allows SimpleDevQA to truly assess an
LLM’s understanding of broad software knowledge, not just its ability to comprehend or generate
code.
8 Threats to Validility
We have identified the following potential threats to our study.
Limited benchmark size and coverage.A potential threat to validity is that the size and
coverage of the benchmark are limited. The current version contains only 2,740 instances across 9
domains and 3 languages, constrained by the limited size and coverage of the seed dataset during
the data generation stage. Specifically, we sampled 3,000 real Dev Knowledge dialogues as the seed
dataset, given the high cost of LLM inference and the labor-intensive manual verification required
in the filtering stage. Nevertheless, the collection pipeline we provided is general for converting
real-world dialogues into high-quality QA pairs. With this pipeline, additional data covering more
languages and domains can be obtained. In future work, we plan to expand the benchmark and
address long-tail and multilingual challenges by collecting more real dialogues.
Focus on a single task.Another potential threat to validity is that our correlation analysis
focuses only on the code generation task. In Section 7.4, we analyze only the correlation between
an LLM’s code generation performance and its Dev Knowledge QA ability. We choose the code
generation task because, according to the findings in Section 3, it is the second most popular task
among developers. In future work, we will investigate how Dev Knowledge QA ability correlates
with performance on other tasks such as code translation [ 89], bug detection [ 71], code editing [ 9],
etc.
Potential bias from model generation and judgment.A potential threat to validity is that
the use of LLMs in both benchmark construction and evaluation may introduce bias. In benchmark
construction, the QA pairs are generated by an LLM using the original dialogues and references as
input. Although we apply a multi-stage filtering process to ensure correctness, this process may still
introduce subtle biases, such as phrasing that reflects the LLM’s style rather than organic developer
language. In evaluation, we set the temperature of the judge model = 0.5 and top-p = 1 to balance
determinism and diversity. While this reduces the risk of extreme bias from greedy decoding,
inherent randomness remains even after multiple runs, which may still affect the consistency,
reproducibility, and introduce potential bias [ 10,101,109]. In future work, we plan to explore
techniques such as prompt engineering [ 62,74] to improve stylistic realism and mitigate these
biases.
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 19
9 Conclusion
In this paper, we propose SimpleDevQA, a multilingual Dev Knowledge QA benchmark built
from real developer dialogues that covers broader development knowledge. To obtain this dataset,
we design a three-step pipeline to convert real-world conversations into a Dev Knowledge QA
benchmark. Through our construction pipeline, we generate 2,740 challenging Dev Knowledge
QA pairs from real Dev Knowledge dialogues. Each QA pair is verified manually to ensure the
correctness of the question and answer. Based on this dataset, we conduct extensive experiments
with 17 existing mainstream LLMs. Experimental results show that first, closed-source models
typically surpass open-source ones, and code LLMs generally outperform general LLMs of similar
scale. Second, the Retrieval-Augmented Generation (RAG) strategy can enable smaller models to
achieve performance comparable to larger models. Third, we find that LLMs’ accuracy generally
increases with their stated confidence in SimpleDevQA. However, they tend to overestimate the
accuracy of their answers. Finally, we find that LLMs with stronger code generation performance
also exhibit stronger performance in the Dev Knowledge QA task.
10 Data Availability
To facilitate the replication study, we have released our data and code at: https://github.com/
DeepSoftwareAnalytics/SimpleDevQA.
References
[1]2023. Confidence interval. Wikipedia, The Free Encyclopedia. https://en.wikipedia.org/wiki/Confidence_interval
[Online; accessed May 31, 2025].
[2] 2023. Sample Size Calculator. https://www.surveysystem.com/sscalc.htm. Accessed: 2025-05-31.
[3] Anthropic. 2024. Claude Haiku. https://www.anthropic.com/claude/haiku Accessed: 2025-05-28.
[4]Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang,
Carrie Cai, Michael Terry, Quoc Le, et al .2021. Program Synthesis with Large Language Models.arXiv preprint
arXiv:2108.07732(2021).
[5]Anton Barua, Stephen W Thomas, and Ahmed E Hassan. 2014. What are developers talking about? an analysis of
topics and trends in stack overflow.Empirical software engineering19 (2014), 619–654.
[6] BigModel Team. 2024. GLM-4 Usage Guide. https://open.bigmodel.cn/dev/howuse/glm-4 Accessed: 2025-05-28.
[7]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan,
Pranav Shyam, Girish Sastry, Amanda Askell, et al .2020. Language models are few-shot learners.Advances in neural
information processing systems33 (2020), 1877–1901.
[8]Vittorio Castelli, Rishav Chakravarti, Saswati Dana, Anthony Ferritto, Radu Florian, Martin Franz, Dinesh Garg,
Dinesh Khandelwal, Scott McCarley, Mike McCawley, Mohamed Nasr, Lin Pan, Cezar Pendus, John Pitrelli, Saurabh
Pujar, Salim Roukos, Andrzej Sakrajda, Avirup Sil, Rosario Uceda-Sosa, Todd Ward, and Rong Zhang. 2019. The
TechQA Dataset. arXiv:1911.02984 [cs.CL] https://arxiv.org/abs/1911.02984
[9]Saikat Chakraborty, Yangruibo Ding, Miltiadis Allamanis, and Baishakhi Ray. 2020. Codit: Code editing with tree-based
neural models.IEEE Transactions on Software Engineering48, 4 (2020), 1385–1399.
[10] Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. 2024. Humans or llms as the judge? a
study on judgement biases.arXiv preprint arXiv:2402.10669(2024).
[11] Jialiang Chen, Kaifa Zhao, Jie Liu, Chao Peng, Jierui Liu, Hang Zhu, Pengfei Gao, Ping Yang, and Shuiguang
Deng. 2025. CoReQA: Uncovering Potentials of Language Models in Code Repository Question Answering.
arXiv:2501.03447 [cs.SE] https://arxiv.org/abs/2501.03447
[12] Jiachi Chen, Qingyuan Zhong, Yanlin Wang, Kaiwen Ning, Yongkun Liu, Zenan Xu, Zhe Zhao, Ting Chen, and Zibin
Zheng. 2024. Rmcbench: Benchmarking large language models’ resistance to malicious code. InProceedings of the
39th IEEE/ACM International Conference on Automated Software Engineering. 995–1006.
[13] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards,
Yuri Burda, Nicholas Joseph, Greg Brockman, et al .2021. Evaluating large language models trained on code.arXiv
preprint arXiv:2107.03374(2021).
[14] I Chern, Steffi Chern, Shiqi Chen, Weizhe Yuan, Kehua Feng, Chunting Zhou, Junxian He, Graham Neubig, Pengfei
Liu, et al .2023. FacTool: Factuality Detection in Generative AI–A Tool Augmented Framework for Multi-Task and
Multi-Domain Scenarios.arXiv preprint arXiv:2307.13528(2023).
, Vol. 1, No. 1, Article . Publication date: December 2025.

20Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
[15] José Wellington Franco da Silva, Amanda Drielly Pires Venceslau, Juliano Efson Sales, José Gilvan Rodrigues Maia,
Vládia Célia Monteiro Pinheiro, and Vânia Maria Ponte Vidal. 2020. A short survey on end-to-end simple question
answering systems.Artificial Intelligence Review53, 7 (2020), 5429–5453.
[16] Bhuwan Dhingra, Kathryn Mazaitis, and William W Cohen. 2017. Quasar: Datasets for question answering by search
and reading.arXiv preprint arXiv:1707.03904(2017).
[17] Luca Di Grazia and Michael Pradel. 2023. Code search: A survey of techniques for finding code.Comput. Surveys55,
11 (2023), 1–31.
[18] Dennis Diefenbach, Vanessa Lopez, Kamal Singh, and Pierre Maret. 2018. Core techniques of question answering
systems over knowledge bases: a survey.Knowledge and Information systems55, 3 (2018), 529–569.
[19] Jing Gong, Yanghui Wu, Linxi Liang, Zibin Zheng, and Yanlin Wang. 2024. CoSQA+: Enhancing code search dataset
with matching code.arXiv e-prints(2024), arXiv–2406.
[20] Google Developers. 2024. Custom Search API Documentation. https://developers.google.com/custom-search
Accessed: 2025-05-28.
[21] Goose3 Contributors. 2024. Goose3: A Python Library for Web Content Extraction. https://github.com/goose3/goose3
Accessed: 2025-05-28.
[22] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al .2024. The llama 3 herd of models.arXiv preprint
arXiv:2407.21783(2024).
[23] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma,
Honghao Liu, et al. 2024. A survey on llm-as-a-judge.arXiv preprint arXiv:2411.15594(2024).
[24] Wenchao Gu, Yanlin Wang, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, and Michael Lyu. 2022. Accelerating
Code Search with Deep Hashing and Code Classification. InProceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), Smaranda Muresan, Preslav Nakov, and Aline Villavicencio
(Eds.). Association for Computational Linguistics, Dublin, Ireland, 2534–2544. doi:10.18653/v1/2022.acl-long.181
[25] Melody Y Guan, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz Barak, Alec Helyar, Rachel Dias, Andrea Vallone,
Hongyu Ren, Jason Wei, et al .2024. Deliberative alignment: Reasoning enables safer language models.arXiv preprint
arXiv:2412.16339(2024).
[26] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li,
et al.2024. DeepSeek-Coder: When the Large Language Model Meets Programming–The Rise of Code Intelligence.
arXiv preprint arXiv:2401.14196(2024).
[27] Lianghong Guo, Wei Tao, Runhan Jiang, Yanlin Wang, Jiachi Chen, Xilin Liu, Yuchi Ma, Mingzhi Mao, Hongyu Zhang,
and Zibin Zheng. 2025. Omnigirl: A multilingual and multimodal benchmark for github issue resolution.Proceedings
of the ACM on Software Engineering2, ISSTA (2025), 24–46.
[28] Lianghong Guo, Yanlin Wang, Caihua Li, Pengyu Yang, Jiachi Chen, Wei Tao, Yingtian Zou, Duyu Tang, and Zibin
Zheng. 2025. SWE-Factory: Your Automated Factory for Issue Resolution Training Data and Evaluation Benchmarks.
arXiv preprint arXiv:2506.10954(2025).
[29] Lianghong Guo, Yanlin Wang, Ensheng Shi, Wanjun Zhong, Hongyu Zhang, Jiachi Chen, Ruikai Zhang, Yuchi Ma,
and Zibin Zheng. 2024. When to stop? towards efficient code generation in llms with excess token prevention. In
Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis. 1073–1085.
[30] Junda He, Jieke Shi, Terry Yue Zhuo, Christoph Treude, Jiamou Sun, Zhenchang Xing, Xiaoning Du, and David Lo.
2025. From code to courtroom: Llms as the new software judges.arXiv preprint arXiv:2503.02246(2025).
[31] Yancheng He, Shilong Li, Jiaheng Liu, Yingshui Tan, Weixun Wang, Hui Huang, Xingyuan Bu, Hangyu Guo, Chengwei
Hu, Boren Zheng, et al .2024. Chinese simpleqa: A chinese factuality evaluation for large language models.arXiv
preprint arXiv:2411.07140(2024).
[32] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir
Puranik, Horace He, Dawn Song, and Jacob Steinhardt. 2021. Measuring Coding Challenge Competence With APPS.
NeurIPS(2021).
[33] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020.
Measuring massive multitask language understanding.arXiv preprint arXiv:2009.03300(2020).
[34] Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, and Haoyu
Wang. 2024. Large language models for software engineering: A systematic literature review.ACM Transactions on
Software Engineering and Methodology33, 8 (2024), 1–79.
[35] Ruida Hu, Chao Peng, Jingyi Ren, Bo Jiang, Xiangxin Meng, Qinyun Wu, Pengfei Gao, Xinchen Wang, and Cuiyun
Gao. 2024. CodeRepoQA: A Large-scale Benchmark for Software Engineering Question Answering.arXiv preprint
arXiv:2412.14764(2024).
[36] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv,
Yikai Zhang, Yao Fu, et al .2023. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models.
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 21
Advances in Neural Information Processing Systems36 (2023), 62991–63010.
[37] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming
Lu, et al. 2024. Qwen2. 5-coder technical report.arXiv preprint arXiv:2409.12186(2024).
[38] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2023.
Swe-bench: Can language models resolve real-world github issues?arXiv preprint arXiv:2310.06770(2023).
[39] Xin Jin, Jonathan Larson, Weiwei Yang, and Zhiqiang Lin. 2023. Binary code summarization: Benchmarking
chatgpt/gpt-4 and other large language models.arXiv preprint arXiv:2312.09601(2023).
[40] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised
challenge dataset for reading comprehension.arXiv preprint arXiv:1705.03551(2017).
[41] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac
Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al .2022. Language models (mostly) know what they know.
arXiv preprint arXiv:2207.05221(2022).
[42] Eirini Kalliamvakou, Georgios Gousios, Kelly Blincoe, Leif Singer, Daniel M German, and Daniela Damian. 2014. The
promises and perils of mining github. InProceedings of the 11th working conference on mining software repositories.
92–101.
[43] Klaus Krippendorff. 2011. Computing Krippendorff’s Alpha-Reliability. (2011).
[44] Changyoon Lee, Yeon Seonwoo, and Alice Oh. 2022. CS1QA: A dataset for assisting code-based question answering
in an introductory programming course.arXiv preprint arXiv:2210.14494(2022).
[45] Krystal M Lewis and Peter Hepburn. 2010. Open card sorting and factor analysis: a usability case study.The electronic
library28, 3 (2010), 401–416.
[46] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[47] Changmao Li and Jeffrey Flanigan. 2024. RAC: Efficient LLM Factuality Correction with Retrieval Augmentation.
arXiv preprint arXiv:2410.15667(2024).
[48] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. 2024. Llms-as-judges:
a comprehensive survey on llm-based evaluation methods.arXiv preprint arXiv:2412.05579(2024).
[49] Haochen Li, Chunyan Miao, Cyril Leung, Yanxian Huang, Yuan Huang, Hongyu Zhang, and Yanlin Wang. 2022.
Exploring Representation-level Augmentation for Code Search. InProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for
Computational Linguistics, Abu Dhabi, United Arab Emirates, 4924–4936. doi:10.18653/v1/2022.emnlp-main.327
[50] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. 2023.
Cmmlu: Measuring massive multitask language understanding in chinese.arXiv preprint arXiv:2306.09212(2023).
[51] Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2023. Halueval: A large-scale hallucination
evaluation benchmark for large language models.arXiv preprint arXiv:2305.11747(2023).
[52] Jiarui Li, Ye Yuan, and Zehua Zhang. 2024. Enhancing llm factual accuracy with rag to counter hallucinations: A case
study on domain-specific queries in private knowledge-bases.arXiv preprint arXiv:2403.10446(2024).
[53] Linyi Li, Shijie Geng, Zhenwen Li, Yibo He, Hao Yu, Ziyue Hua, Guanghan Ning, Siwei Wang, Tao Xie, and Hongxia
Yang. 2024. InfiBench: Evaluating the Question-Answering Capabilities of Code Large Language Models. InAdvances
in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and
C. Zhang (Eds.), Vol. 37. Curran Associates, Inc., 128668–128698. https://proceedings.neurips.cc/paper_files/paper/
2024/file/e888eb9400fe14bb70e057aa1d719188-Paper-Datasets_and_Benchmarks_Track.pdf
[54] Linyi Li, Shijie Geng, Zhenwen Li, Yibo He, Hao Yu, Ziyue Hua, Guanghan Ning, Siwei Wang, Tao Xie, and Hongxia
Yang. 2024. Infibench: Evaluating the question-answering capabilities of code large language models.Advances in
Neural Information Processing Systems37 (2024), 128668–128698.
[55] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. InText summarization branches out.
74–81.
[56] Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods.
arXiv preprint arXiv:2109.07958(2021).
[57] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu
Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report.arXiv preprint arXiv:2412.19437(2024).
[58] Chenxiao Liu and Xiaojun Wan. 2021. CodeQA: A question answering dataset for source code comprehension.arXiv
preprint arXiv:2109.08365(2021).
[59] Jerry Liu. 2022.LlamaIndex. doi:10.5281/zenodo.1234
[60] Jiawei Liu, Jia Le Tian, Vijay Daita, Yuxiang Wei, Yifeng Ding, Yuhan Katherine Wang, Jun Yang, and Lingming Zhang.
2024. Repoqa: Evaluating long context code understanding.arXiv preprint arXiv:2406.06025(2024).
, Vol. 1, No. 1, Article . Publication date: December 2025.

22Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
[61] Dung Nguyen Manh, Thang Phan Chau, Nam Le Hai, Thong T Doan, Nam V Nguyen, Quang Pham, and Nghi DQ
Bui. 2024. CodeMMLU: A Multi-Task Benchmark for Assessing Code Understanding Capabilities of CodeLLMs.arXiv
preprint arXiv:2410.01999(2024).
[62] Ggaliwango Marvin, Nakayiza Hellen, Daudi Jjingo, and Joyce Nakatumba-Nabende. 2023. Prompt engineering in
large language models. InInternational conference on data intelligence and cognitive informatics. Springer, 387–402.
[63] Mary L Mchugh. 2012. Interrater reliability: the kappa statistic.Biochemia Medica22, 3 (2012), 276–282.
[64] Meta AI. 2024. Meta Llama 3 8B Instruct. https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct. Accessed:
2025-05-31.
[65] Changan Niu, Chuanyi Li, Bin Luo, and Vincent Ng. 2022. Deep learning meets software engineering: A survey on
pre-trained models of source code.arXiv preprint arXiv:2205.11739(2022).
[66] OpenAI. 2024. GPT-4o mini. https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/. Accessed:
2025-05-29.
[67] OpenAI. 2024. Hello GPT-4o. https://openai.com/index/hello-gpt-4o/ Accessed: 2025-05-28.
[68] OpenAI. 2024. OpenAI o3 Mini. https://openai.com/index/openai-o3-mini/ Accessed: 2025-05-28.
[69] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation
of machine translation. InProceedings of the 40th annual meeting of the Association for Computational Linguistics.
311–318.
[70] Jan Piasecki, Marcin Waligora, and Vilius Dranseika. 2018. Google search as an additional source in systematic
reviews.Science and engineering ethics24 (2018), 809–810.
[71] Michael Pradel and Koushik Sen. 2018. Deepbugs: A learning approach to name-based bug detection.Proceedings of
the ACM on Programming Languages2, OOPSLA (2018), 1–25.
[72] Team Qwen. [n. d.]. Qwen2. 5: A party of foundation models, September 2024.URL https://qwenlm. github. io/blog/qwen2
5 ([n. d.]).
[73] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu,
Romain Sauvestre, Tal Remez, et al .2023. Code llama: Open foundation models for code.arXiv preprint arXiv:2308.12950
(2023).
[74] Sander Schulhoff, Michael Ilie, Nishant Balepur, Konstantine Kahadze, Amanda Liu, Chenglei Si, Yinheng Li, Aayush
Gupta, HyoJung Han, Sevien Schulhoff, et al .2024. The prompt report: a systematic survey of prompt engineering
techniques.arXiv preprint arXiv:2406.06608(2024).
[75] Ensheng Shi, Yanlin Wang, Lun Du, Junjie Chen, Shi Han, Hongyu Zhang, Dongmei Zhang, and Hongbin Sun. 2022.
On the evaluation of neural code summarization. InProceedings of the 44th international conference on software
engineering. 1597–1608.
[76] Ensheng Shi, Yanlin Wang, Wenchao Gu, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, and Hongbin Sun. 2023.
Cocosoda: Effective contrastive learning for code search. In2023 IEEE/ACM 45th International Conference on Software
Engineering (ICSE). IEEE, 2198–2210.
[77] Ensheng Shi, Yanlin Wang, Fengji Zhang, Bei Chen, Hongyu Zhang, Yanli Wang, Daya Guo, Lun Du, Shi Han,
Dongmei Zhang, and Hongbin Sun. 2025. SoTaNa: An Open-Source Software Engineering Instruction-Tuned Model.
In2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge). 1–12.
doi:10.1109/Forge66646.2025.00010
[78] Guijin Son, Hyunwoo Ko, Hoyoung Lee, Yewon Kim, and Seunghyeok Hong. 2024. Llm-as-a-judge & reward model:
What they can and cannot do.arXiv preprint arXiv:2409.11239(2024).
[79] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R
Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al .2022. Beyond the imitation game: Quantifying and
extrapolating the capabilities of language models.arXiv preprint arXiv:2206.04615(2022).
[80] Wei Tao, Yanlin Wang, Ensheng Shi, Lun Du, Shi Han, Hongyu Zhang, Dongmei Zhang, and Wenqiang Zhang.
2021. On the evaluation of commit message generation models: An experimental study. In2021 IEEE International
Conference on Software Maintenance and Evolution (ICSME). IEEE, 126–136.
[81] Wei Tao, Yanlin Wang, Ensheng Shi, Lun Du, Shi Han, Hongyu Zhang, Dongmei Zhang, and Wenqiang Zhang. 2022.
A large-scale empirical study of commit message generation: models, datasets and evaluation.Empirical Software
Engineering27, 7 (2022), 198.
[82] I Team. 2024. Internlm2 technical report.arXiv preprint arXiv:2403.17297(2024).
[83] valpy. 2024. Prompt Classification Model. https://huggingface.co/valpy/prompt-classification Accessed: 2025-05-28.
[84] Haoye Wang, Xin Xia, David Lo, Qiang He, Xinyu Wang, and John Grundy. 2021. Context-aware retrieval-based deep
commit message generation.ACM Transactions on Software Engineering and Methodology (TOSEM)30, 4 (2021), 1–30.
[85] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny
Zhou. 2022. Self-consistency improves chain of thought reasoning in language models.arXiv preprint arXiv:2203.11171
(2022).
, Vol. 1, No. 1, Article . Publication date: December 2025.

SimpleDevQA: Benchmarking Large Language Models on Development Knowledge QA 23
[86] Yanlin Wang, Tianyue Jiang, Mingwei Liu, Jiachi Chen, Mingzhi Mao, Xilin Liu, Yuchi Ma, and Zibin Zheng. 2025.
Beyond functional correctness: Investigating coding style inconsistencies in large language models.Proceedings of
the ACM on Software Engineering2, FSE (2025), 690–712.
[87] Zhiruo Wang, Shuyan Zhou, Daniel Fried, and Graham Neubig. 2022. Execution-based evaluation for open-domain
code generation.arXiv preprint arXiv:2212.10481(2022).
[88] Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and
William Fedus. 2024. Measuring short-form factuality in large language models.arXiv preprint arXiv:2411.04368
(2024).
[89] Justin D Weisz, Michael Muller, Steven I Ross, Fernando Martinez, Stephanie Houde, Mayank Agarwal, Kartik
Talamadupula, and John T Richards. 2022. Better together? an evaluation of ai-supported code translation. In
Proceedings of the 27th International Conference on Intelligent User Interfaces. 369–391.
[90] Philip Welsby and Bernard MY Cheung. 2023. ChatGPT. 1047–1048 pages.
[91] Di Wu, Xiao-Yuan Jing, Hongyu Zhang, Yang Feng, Haowen Chen, Yuming Zhou, and Baowen Xu. 2023. Retrieving
API knowledge from tutorials and stack overflow based on natural language queries.ACM Transactions on Software
Engineering and Methodology32, 5 (2023), 1–36.
[92] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, and Huaxiu Yao. 2024. Rule: Reliable
multimodal rag for factuality in medical vision language models. InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing. 1081–1093.
[93] Shuhan Yan, Hang Yu, Yuting Chen, Beijun Shen, and Lingxiao Jiang. 2020. Are the code snippets what we are
searching for? a benchmark and an empirical study on code search with natural-language queries. In2020 IEEE 27th
International Conference on Software Analysis, Evolution and Reengineering (SANER). IEEE, 344–354.
[94] John Yang, Carlos E Jimenez, Alex L Zhang, Kilian Lieret, Joyce Yang, Xindi Wu, Ori Press, Niklas Muennighoff,
Gabriel Synnaeve, Karthik R Narasimhan, et al .2024. Swe-bench multimodal: Do ai systems generalize to visual
software domains?arXiv preprint arXiv:2410.03859(2024).
[95] John Yang, Kilian Leret, Carlos E Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir
Press, Ludwig Schmidt, and Diyi Yang. 2025. Swe-smith: Scaling data for software engineering agents.arXiv preprint
arXiv:2504.21798(2025).
[96] Kang Yang, Xinjun Mao, Shangwen Wang, Yanlin Wang, Tanghaoran Zhang, Bo Lin, Yihao Qin, Zhang Zhang,
Yao Lu, and Kamal Al-Sabahi. 2025. Large Language Models Are Qualified Benchmark Builders: Rebuilding Pre-
Training Datasets for Advancing Code Intelligence Tasks. In2025 IEEE/ACM 33rd International Conference on Program
Comprehension (ICPC). 298–309. doi:10.1109/ICPC66645.2025.00038
[97] Shuo Yang, Jiachi Chen, and Zibin Zheng. 2023. Definition and detection of defects in NFT smart contracts. In
Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis. 373–384.
[98] Shuo Yang, Xingwei Lin, Jiachi Chen, Qingyuan Zhong, Lei Xiao, Renke Huang, Yanlin Wang, and Zibin Zheng. 2025.
Hyperion: Unveiling DApp Inconsistencies Using LLM and Dataflow-Guided Symbolic Execution. In2025 IEEE/ACM
47th International Conference on Software Engineering (ICSE). 2125–2137. doi:10.1109/ICSE55347.2025.00015
[99] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering.arXiv preprint
arXiv:1809.09600(2018).
[100] Ziyu Yao, Daniel S Weld, Wei-Peng Chen, and Huan Sun. 2018. Staqc: A systematically mined question-code dataset
from stack overflow. InProceedings of the 2018 World Wide Web Conference. 1693–1703.
[101] Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang,
Pin-Yu Chen, et al .2024. Justice or prejudice? quantifying biases in llm-as-a-judge.arXiv preprint arXiv:2410.02736
(2024).
[102] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong
Chen, et al .2025. Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models.Computational
Linguistics(2025), 1–46.
[103] Yuxia Zhang, Zhiqing Qiu, Klaas-Jan Stol, Wenhui Zhu, Jiaxin Zhu, Yingchen Tian, and Hui Liu. 2024. Automatic
commit message generation: A critical review and directions for future work.IEEE Transactions on Software Engineering
50, 4 (2024), 816–835.
[104] Ziyin Zhang, Chaoyu Chen, Bingchang Liu, Cong Liao, Zi Gong, Hang Yu, Jianguo Li, and Rui Wang. 2023. Unifying the
perspectives of nlp and software engineering: A survey on language models for code.arXiv preprint arXiv:2311.07989
(2023).
[105] Ziyao Zhang, Chong Wang, Yanlin Wang, Ensheng Shi, Yuchi Ma, Wanjun Zhong, Jiachi Chen, Mingzhi Mao, and Zibin
Zheng. 2025. Llm hallucinations in practical code generation: Phenomena, mechanism, and mitigation.Proceedings of
the ACM on Software Engineering2, ISSTA (2025), 481–503.
, Vol. 1, No. 1, Article . Publication date: December 2025.

24Jing Zhang, Lianghong Guo, Yanlin Wang, Mingwei Liu, Jiachi Chen, Yuchi Ma, Ensheng Shi, Terry Yue Zhuo, Hongyu
Zhang, and Zibin Zheng
[106] Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. 2024. Wildchat: 1m chatgpt
interaction logs in the wild.arXiv preprint arXiv:2405.01470(2024).
[107] Dewu Zheng, Yanlin Wang, Ensheng Shi, Hongyu Zhang, and Zibin Zheng. 2024. How well do llms generate code for
different application domains? benchmark and evaluation.arXiv preprint arXiv:2412.18573(2024).
[108] Dewu Zheng, Yanlin Wang, Ensheng Shi, Ruikai Zhang, Yuchi Ma, Hongyu Zhang, and Zibin Zheng. 2025. HumanEvo:
An Evolution-Aware Benchmark for More Realistic Evaluation of Repository-Level Code Generation. InProceedings
of the IEEE/ACM 47th International Conference on Software Engineering(Ottawa, Ontario, Canada)(ICSE ’25). IEEE
Press, 1372–1384. doi:10.1109/ICSE55347.2025.00228
[109] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan
Li, Dacheng Li, Eric Xing, et al .2023. Judging llm-as-a-judge with mt-bench and chatbot arena.Advances in neural
information processing systems36 (2023), 46595–46623.
[110] Zibin Zheng, Kaiwen Ning, Qingyuan Zhong, Jiachi Chen, Wenqing Chen, Lianghong Guo, Weicheng Wang, and
Yanlin Wang. 2025. Towards an understanding of large language models in software engineering tasks.Empirical
Software Engineering30, 2 (2025), 50.
[111] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan
Duan. 2024. AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models. InFindings of the Association
for Computational Linguistics: NAACL 2024, Kevin Duh, Helena Gomez, and Steven Bethard (Eds.). Association for
Computational Linguistics, Mexico City, Mexico, 2299–2314. doi:10.18653/v1/2024.findings-naacl.149
[112] Terry Yue Zhuo. 2024. ICE-Score: Instructing Large Language Models to Evaluate Code. InFindings of the Association
for Computational Linguistics: EACL 2024. 2232–2242.
, Vol. 1, No. 1, Article . Publication date: December 2025.