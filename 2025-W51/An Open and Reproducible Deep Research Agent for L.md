# An Open and Reproducible Deep Research Agent for Long-Form Question Answering

**Authors**: Ikuya Yamada, Wataru Ikeda, Ko Yoshida, Mengyu Ye, Hinata Sugimoto, Masatoshi Suzuki, Hisanori Ozaki, Jun Suzuki

**Published**: 2025-12-15 07:37:53

**PDF URL**: [https://arxiv.org/pdf/2512.13059v1](https://arxiv.org/pdf/2512.13059v1)

## Abstract
We present an open deep research system for long-form question answering, selected as a winning system in the text-to-text track of the MMU-RAG competition at NeurIPS 2025. The system combines an open-source large language model (LLM) with an open web search API to perform iterative retrieval, reasoning, and synthesis in real-world open-domain settings. To enhance reasoning quality, we apply preference tuning based on LLM-as-a-judge feedback that evaluates multiple aspects, including clarity, insightfulness, and factuality. Our experimental results show that the proposed method consistently improves answer quality across all three aspects. Our source code is publicly available at https://github.com/efficient-deep-research/efficient-deep-research.

## Full Text


<!-- PDF content starts -->

An Open and Reproducible Deep Research Agent
for Long-Form Question Answering
Ikuya Yamada1,2Wataru Ikeda2Ko Yoshida2Mengyu Ye2
Hinata Sugimoto2Masatoshi Suzuki1,2Hisanori Ozaki3Jun Suzuki2,4,5
1Studio Ousia
{ikuya,m.suzuki}@ousia.jp
2Tohoku University
{ikeda.wataru.q5,yoshida.kou.p3,ye.mengyu.s1,sugimoto.hinata.q2}@dc.tohoku.ac.jp
jun.suzuki@tohoku.ac.jp
3DENTSU SOKEN INC.
ozaki.hisanori@dentsusoken.com
4RIKEN,5NII LLMC
Abstract
We present an open deep research system for long-form question answering, se-
lected as a winning system in the text-to-text track of the MMU-RAG compe-
tition at NeurIPS 2025. The system combines an open-source large language
model (LLM) with an open web search API to perform iterative retrieval, rea-
soning, and synthesis in real-world open-domain settings. To enhance reason-
ing quality, we apply preference tuning based on LLM-as-a-judge feedback that
evaluates multiple aspects, including clarity, insightfulness, and factuality. Our
experimental results show that the proposed method consistently improves answer
quality across all three aspects. Our source code is publicly available at https:
//github.com/efficient-deep-research/efficient-deep-research.
1 Introduction
The text-to-text track of the MMU-RAG competition1at NeurIPS 2025 focuses on building retrieval-
augmented generation (RAG) systems that generate long-form answers under real-world conditions.
A natural approach to developing such RAG systems is to adopt deep research, a tool-augmented
LLM capable of performing retrieval, reasoning, and synthesis to address a broad range of questions.
Long-form question answering based on deep research has been implemented in commercial systems
[9,3], but these systems remain proprietary, making them difficult to study or reproduce. Motivated
by these limitations, recent studies have attempted to reproduce similar capabilities [ 6,18,17,4,15,
21, 7, 16, 2, 8].
However, most of these studies have focused on short-form question answering, where answers can
be easily verified against gold-standard references, while only a limited number of recent works
address long-form question answering based on deep research [7, 8, 14].
1https://agi-lti.github.io/MMU-RAGent/
Preprint.arXiv:2512.13059v1  [cs.CL]  15 Dec 2025

Search API Reranker SummarizerSearch Tool
Question Research Agent Answer
Inference can span multiple turnsTop K
retrieved
documentsTop-N
reranked
documentsSummaries
of top- N
documentsFigure 1: Architecture of our deep research system.
In this report, we present our deep research system submitted to the text-to-text track of the MMU-
RAG competition. The system is designed to effectively address long-form question answering and
is built using an open-source LLM and a search API developed on an open web corpus, promoting
reproducibility. Our system integrates high-quality synthetic data generation, preference tuning, and
an improved search component, yielding substantial gains across evaluation metrics.
We adopt the metrics proposed by Coelho et al. [1]—clarity ,insightfulness , and a factuality
metric based on KPR [ 11], which we refer to as factuality throughout this paper,2and train an
LLM using Direct Preference Optimization (DPO) [ 12]. Following prior work [ 7,8], we use an LLM
to evaluate preferences along these dimensions.
We also extend the search API with a reranking and summarization module. After retrieving
documents, the module reranks them and produces concise summaries of the top reranked documents,
enabling the generator to produce more accurate and better-supported answers. The system also
produces inline citations by prompting the model to link statements to the corresponding documents.
Experiments using an LLM-as-a-judge setup show consistent improvements across all three metrics.
These findings indicate that preference-based tuning, combined with enhanced retrieval and summa-
rization, leads to measurable performance gains for long-form deep-research tasks. Furthermore, the
system received the Best Static Evaluation award in the open-source category of the text-to-text track.
2 Architecture
Figure 1 illustrates the architecture of our system. The system consists of a search tool and a research
agent. The search tool takes a query and returns summaries based on the retrieved documents. The
research agent manages the reasoning flow by iteratively invoking the search tool to gather and
synthesize information into a coherent final answer.
2.1 Search Tool
The search tool is a pipeline comprising a search API, a reranker, and a summarizer.
Search APIWe use the open search API [ 1], which is built on ClueWeb22 [ 10], a large collection
of billions of web documents.3For each query, the top- Kdocuments are retrieved from the API and
passed to the reranker. Here, the query can be either the original question or one generated by our
research agent.
RerankerThe reranker reorders the top- Kdocuments returned by the search API and selects
the most relevant ones. We use the state-of-the-art model Qwen3-Reranker-0.6B4[20], with the
prompt described in Appendix A. The top- Nreranked documents are then passed to the summarizer.
SummarizerThe summarizer extracts information relevant to the given query and previous
reasoning steps, generating a concise summary of the retrieved web documents. We use
Qwen3-Next-80B-A3B-Thinking5[19] with the prompt described in Appendix A. Using a simple
yet effective prompting technique, we instruct the model to include citation markers in the summary,
each corresponding to its source document.
2See Section 3.1 for precise definitions.
3We use the ClueWeb22-A category covering two billion pages.
4https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
5https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking
2

2.2 Research Agent
The research agent manages the overall reasoning process of the system. It takes the question and
the previously generated summaries as input, determines whether additional searches are needed
through reasoning, and issues a new query when necessary. The agent iteratively calls the search tool
described above and integrates the gathered information into a coherent final answer. Before starting
the reasoning process, we invoke the search tool using the input question as the query and include the
summarized results in the research agent’s prompt. We employ Qwen3-Next-80B-A3B-Thinking ,
fine-tuned as described in the next section, with the prompt provided in Appendix A.
3 Training of Research Agent
We provide details on the creation of the synthetic data and the DPO training of our system.
3.1 Data Generation
The construction of preference pairs consists of four steps: (1) collecting questions for training, (2)
generating answers to these questions, (3) performing LLM-based preference evaluation, and (4)
constructing preference pairs based on the evaluation results.
MetricsWe use the clarity andinsightfulness metrics defined by Coelho et al. [1], and
we use the KPR metric [ 11], which we refer to as factuality . Specifically, clarity reflects
logical coherence and linguistic fluency; insightfulness captures analytical nuance and depth of
reasoning; andfactualitymeasures consistency with the underlying facts.6
Question CollectionWe collect 1,000 training questions from multiple datasets: 500 from Re-
searchy Questions [ 13], 200 from Natural Questions [ 5], and 300 from the official validation set
provided by the competition organizers, resulting in a balanced dataset that includes both factoid and
non-factoid questions and reflects the diversity observed in real-world usage and in the competition’s
evaluation. Researchy Questions consists of non-factoid questions collected from a commercial
search engine, accompanied by URLs clicked by users for each query. We collect web documents
from these URLs to extract key points used in the factuality evaluation. Natural Questions pri-
marily consists of factoid questions sampled from a commercial search engine. The competition’s
official validation set provides questions designed to match those used in the final evaluation.
Answer GenerationWe generate answers for the questions using the research agent based on an
untuned LLM. For each question, we perform 20 sampling runs. The hyperparameters used during
sampling are described in Appendix C.
Preference Score EvaluationWe use OpenAI’s o3-mini to assign preference scores to the
generated answers. Specifically, we evaluate all three metrics on Researchy Questions, and only
clarity andinsightfulness for Natural Questions and the validation set, since the user-clicked
URLs required to assess factuality are not available for these datasets. The final score is obtained
by sum up the normalized score of each metrics.
Preference Pair ConstructionWe construct the DPO data using the preference scores and the
number of search queries used to generate each answer. The answer with the highest preference score
is selected as thechosenresponse, and the one with the lowest score as therejectedresponse. When
answers share the same preference score, we select the answer with more search queries as thechosen
response, and vice versa.
We then filter out answers containing formatting errors, such as incorrect citation of the source.
Finally, to retain only preference pairs with a meaningful score gap, we set a minimum threshold θ
for the score difference between the chosen and rejected responses and include only the pairs that
satisfy this threshold.
6See Appendix A for the prompts used to assess each metric.
3

Table 1: Evaluation results comparing modelsbefore(VANILLA) andafter(OURS) tuning. All
metrics except citation_error_rate are averaged across samples. Our tuned model achieves
consistent improvements in LLM-based evaluation metrics and search_count while showing a
slight increase incitation_error_rate.
Research AgentLLM-based EvaluationSearch Count↑Citation Error Rate↓
Clarity↑Insightfulness↑Factuality↑
VANILLA6.71 6.52 43.4 1.030.06
OURS(θ= 0.3)8.18 7.50 44.3 1.200.09
3.2 Training
We perform DPO training using LoRA7on the synthetic dataset, without extensive hyperparameter
tuning. The model is optimized on both the reasoning chain and the final answer. We mask out the
tokens corresponding to the responses returned by the search tool during training, as they are not
generated by the research agent.
We construct three preference datasets by setting the minimum score-gap threshold to θ∈
{0.3,0.5,0.7} , yielding datasets with 983, 828, and 341 pairs. We train the model for one epoch
on each dataset, resulting in 56,47, and 20training steps. Details of the datasets are provided in
Appendix B.
4 Evaluation
We verify the effectiveness of the our approach based on LLM-as-judge.
4.1 Evaluation Setup
We sample 100 queries from Researchy Questions that are not included in the training set and conduct
LLM-based evaluation on each. Specifically, for each question in the evaluation data, we generate
answers using one sampling run from the model before tuning (VANILLA) and from models tuned
with different minimum score-gap thresholds θ. Because all tuned models consistently outperform
VANILLAwith only marginal differences among them, we hereafter focus on the model tuned with
θ= 0.3 , referred to as OURS89. The hyperparameters for answer generation and the evaluation
procedure follow those used for constructing the preference pairs in Section 3.1. We also measure
search_count (the number of searches issued by the system) and citation_error_rate (the
proportion of final answers with incorrectly formatted citation markers among the 100 samples) to
analyze system behavior beyond answer quality.
4.2 Results
Table 1 shows our evaluation results. Our tuned model outperforms VANILLAacross all LLM-based
evaluation metrics, demonstrating the effectiveness of our training. The gains are particularly promi-
nent for clarity (+1.47 ) and insightfulness (+0.98 ), while factuality shows a marginal
improvement ( +0.41 ). We believe this is partly because training for factuality is more difficult
than for the other two metrics, as it relies on noisy user-click–based training signals and is available
only for a subset of the training data. The tuned model also performs more searches on average
(+0.17), while exhibiting a slight increase incitation_error_rate(+0.03).
5 Conclusion
In this report, we presented an open and reproducible deep research system for long-form question
answering, which was a winning system in the text-to-text track of the MMU-RAG competition at
7For LoRA, we useα= 16,rank = 16, and for DPO, we useβ= 0.5.
8See Appendix B for comparisons of evaluation results across tuned models with different thresholdθ.
9We selected the model tuned withθ= 0.3for competition submission.
4

NeurIPS 2025. The system is built by combining high-quality synthetic data creation, DPO training,
and an improved search component. Our system effectively addresses real-world question-answering
tasks by integrating retrieval, reasoning, and synthesis during inference. LLM-based evaluations
confirm the effectiveness of our training, showing noticeable improvements across all metrics.
Acknowledgments
This work was supported by the “R&D Hub Aimed at Ensuring Transparency and Reliability of
Generative AI Models” project of the Ministry of Education, Culture, Sports, Science and Technology,
and JST Moonshot R&D Grant Number JPMJMS2011-35 (fundamental research).
References
[1]João Coelho, Jingjie Ning, Jingyuan He, Kangrui Mao, Abhijay Paladugu, Pranav Setlur, Jiahe
Jin, Jamie Callan, João Magalhães, Bruno Martins, et al. DeepResearchGym: A free, transparent,
and reproducible evaluation sandbox for deep research.arXiv preprint arXiv:2505.19253, 2025.
[2]Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and
Yi Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous
rl.arXiv preprint arXiv:2508.07976, 2025.
[3]Google Team. Introducing Gemini deep research. https://gemini.google/overview/
deep-research/, 2025. Accessed: 2025-04-06.
[4]Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O Arik, Dong Wang, Hamed
Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines
with reinforcement learning. InSecond Conference on Language Modeling, 2025.
[5]Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural Questions: A benchmark for question answering research.Transactions of the
Association for Computational Linguistics, 7:452–466, 2019.
[6]Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models.arXiv preprint
arXiv:2501.05366, 2025.
[7]Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen,
and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research
capability.arXiv preprint arXiv:2504.21776, 2025.
[8]Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming
Xiong, and Shafiq Joty. SFR-deepresearch: Towards effective reinforcement learning for
autonomously reasoning single agents.arXiv preprint arXiv:2509.06283, 2025.
[9]OpenAI. Introducing deep research. https://openai.com/index/
introducing-deep-research/, 2025. Accessed: 2025-04-06.
[10] Arnold Overwijk, Chenyan Xiong, and Jamie Callan. ClueWeb22: 10 billion web documents
with rich information. InProceedings of the 45th International ACM SIGIR Conference on
Research and Development in Information Retrieval, page 3360–3362, 2022.
[11] Zehan Qi, Rongwu Xu, Zhijiang Guo, Cunxiang Wang, Hao Zhang, and Wei Xu. LONG2RAG :
Evaluating long-context & long-form retrieval-augmented generation with key point recall. In
Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,Findings of the Association for
Computational Linguistics, pages 4852–4872, 2024.
[12] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and
Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model.
InAdvances in Neural Information Processing Systems, volume 36, pages 53728–53741, 2023.
5

[13] Corbin Rosset, Ho-Lam Chung, Guanghui Qin, Ethan Chau, Zhuo Feng, Ahmed Awadallah,
Jennifer Neville, and Nikhil Rao. Researchy questions: A dataset of multi-perspective, decom-
positional questions for deep research. InProceedings of the 48th International ACM SIGIR
Conference on Research and Development in Information Retrieval, page 3712–3722, 2025.
[14] Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo,
Xinran Zhao, Molly Park, Samuel G Finlayson, David Sontag, et al. Dr tulu: Reinforcement
learning with evolving rubrics for deep research.arXiv preprint arXiv:2511.19399, 2025.
[15] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang,
and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement
learning.arXiv preprint arXiv:2503.05592, 2025.
[16] Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan
Zhang, Fei Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of LLMs
without searching.arXiv preprint arXiv:2505.04588, 2025.
[17] Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai,
Jia Deng, Wayne Xin Zhao, Zheng Liu, et al. SimpleDeepSearcher: Deep information seeking
via web-powered reasoning trajectory synthesis.arXiv preprint arXiv:2505.16834, 2025.
[18] Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, and Furu Wei. Chain-
of-retrieval augmented generation.arXiv preprint arXiv:2501.14342, 2025.
[19] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint
arXiv:2505.09388, 2025.
[20] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding
and reranking through foundation models.arXiv preprint arXiv:2506.05176, 2025.
[21] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and
Pengfei Liu. DeepResearcher: Scaling deep research via reinforcement learning in real-world
environments.arXiv preprint arXiv:2504.03160, 2025.
6

Table 2: Statistics of preference datasets with different score-gap thresholds.
θ= 0.3θ= 0.5θ= 0.7
chosen rejected chosen rejected chosen rejected
Number of Samples983 828 341
Preference Score1.77 1.17 1.78 1.14 1.84 1.07
Clarity8.13 4.93 8.18 4.74 8.34 4.31
Insightfulness7.06 4.78 7.08 4.68 7.26 4.41
Factuality49.3 38.9 49.8 38.7 51.7 36.4
Search Count1.37 1.11 1.36 1.10 1.38 1.11
Table 3: Evaluation results of three tuned models with different score-gap thresholds. All metrics
exceptcitation_error_rateare averaged across samples.
Research AgentLLM-based EvaluationSearch Count↑Citation Error Rate↓
Clarity↑Insightfulness↑Factuality↑
TUNED WITHθ= 0.3 8.18 7.50 44.3 1.20 0.09
TUNED WITHθ= 0.5 8.17 7.29 44.1 1.05 0.13
TUNED WITHθ= 0.7 8.30 7.33 44.2 1.23 0.11
A Prompts
The prompts for the reranker and the research agent are shown in Figures 2 and 5, respectively. We
use the same prompt asQwen3-Reranker-0.6B[20] for the reranker.
The prompts used for the summarizer are shown in Figures 3 and 4. We use different prompts for
summarizing the search results included in the research agent’s initial prompt (Figure 3) and for
summarizing the results of searches invoked during the agent’s reasoning process (Figure 4).
We evaluate factuality ,clarity , and insightfulness using the prompts shown in Figures 6,
7, and 8, respectively. Forfactualityevaluation, we sample 10 key points for each question.
B Details of Preference Datasets
Table 2 shows the statistics of the three preference datasets constructed with different minimum
score-gap thresholds θ∈ {0.3,0.5,0.7} . Table 3 shows the evaluation results of models trained on
each of these datasets.
C Hyperparameters for Answer Generation
Table 4 shows the hyperparameters used for answer generation. We use the same settings for both
preference pair construction and evaluation. To improve answer quality, we modify only the Search
API top-Kto 300 in the competition evaluation.
7

Reranker Prompt
System prompt:
Judge whether the Document meets the requirements based on the Query and the
Instruct provided. Note that the answer can only be "yes" or "no".
User prompt:
<Instruct>: Given a web search query, retrieve relevant passages that answer
the query
<Query>: {query}
<Document>: {document}
Figure 2: The prompt used for the reranker. The placeholders {query} and{document} are replaced
with the input query and document, respectively.
Summarizer Prompt for Initial Search Results
**Role**
- You are an expert at extracting content relevant to a question from
multiple ===Web Pages===.
**Instructions**
- Carefully read the ===Web Pages=== provided in Inputs and, following the
**Webpage ID Guidelines** and **Output Format** below, extract the content
relevant to the ===Query===.
- Let's think this out in a step by step way to be sure we have the right
answer.
**Webpage ID Guidelines**
- ===Web Pages=== are presented in the following format: "Webpage ID: #xxxx
(x = alphanumeric)
"context": data["text"], "url": data["url"]"
- When using sentences from the ===Web Pages=== that are relevant to the
===Query===, you **MUST** record the Webpage ID in the format (#+
alphanumerics) exactly as shown in the **Webpage ID Examples** below.
- A Webpage ID is the identifier of the web page and begins with a leading
"#" followed by alphanumeric characters.
- Because the Webpage ID is an identifier, do not include any text other than
the identifier inside the parentheses.
- If you rely on multiple sources, output multiple Webpage IDs in a single
set of parentheses separated by commas, like (#ab12,#cd34)
**Webpage ID Examples**
- Single source: "Compared with pre-industrial times, the global average
temperature has increased by 1.1°C (#ab12)"
- Multiple sources: "In recent years, the adoption of renewable energy has
accelerated (#ab12,#cd34)"
**Output Format**
- You **MUST** begin with`**Final Information**`.
- Include the correct Webpage ID(s) in parentheses (#+ alphanumerics) in the
extracted sentences.
**Inputs**
- ===Query===
{query}
- ===Web Pages===
{documents}
Go ahead—you've got this; extract the information step by step.
Figure 3: The prompt used for the summarizer applied to the initial search results. The placeholders
{query}and{documents}are replaced with the input query and documents, respectively.
8

Summarizer Prompt for Searches Invoked by Research Agent
**Role**
- You are an expert at extracting content relevant to a question from
multiple ===Web Pages=== and integrating it after understanding the contents
of ===Previous Reasoning Steps===.
**Instructions**
- Carefully read the ===Web Pages=== provided in Inputs and, following the
**Webpage ID Guidelines** and **Output Format** below, extract the content
relevant to the ===Query===.
- Read and fully understand ===Previous Reasoning Steps===, then integrate
the extracted content with it.
- Let's think this out in a step by step way to be sure we have the right
answer.
**Webpage ID Guidelines**
- ===Web Pages=== are presented in the following format: "Webpage ID: #xxxx
(x = alphanumeric)
"context": data["text"], "url": data["url"]"
- When using sentences from the ===Web Pages=== that are relevant to the
===Query===, you **MUST** record the Webpage ID in the format (#+
alphanumerics) exactly as shown in the **Webpage ID Examples** below.
- A Webpage ID is the identifier of the web page and begins with a leading
"#" followed by alphanumeric characters.
- Because the Webpage ID is an identifier, do not include any text other than
the identifier inside the parentheses.
- If you rely on multiple sources, output multiple Webpage IDs in a single
set of parentheses separated by commas, like (#ab12,#cd34).
**Webpage ID Examples**
- Single source: "Compared with pre-industrial times, the global average
temperature has increased by 1.1°C (#ab12)"
- Multiple sources: "In recent years, the adoption of renewable energy has
accelerated (#ab12,#cd34)"
**Output Format**
- You **MUST** begin with`**Final Information**`.
- Include the correct Webpage ID(s) in parentheses (#+ alphanumerics) in the
extracted sentences.
**Inputs**
- ===Query===
{query}
- ===Web Pages===
{documents}
- ===Previous Reasoning Steps===
{reasoning_steps}
Go ahead—confidently extract the information for the question and integrate
it into the Previous Reasoning Steps.
Figure 4: The prompt used for the summarizer applied to the searches issued by the research agent.
The placeholders {query} ,{documents} , and {reasoning_steps} are replaced with the input
query, documents, and previous reasoning outputs, respectively.
9

Research Agent Prompt
*Role*
- You are an agent that can perform web searches to accurately answer the
user's question.
*Instructions*
- Carefully read the ===initial_search_result=== provided in Inputs and
answer ===question===.
- Because ===initial_search_result=== is the first round of search results,
it may be insufficient. Especially when the information is inadequate to
answer the question correctly—for example, when you encounter unfamiliar
terms—you **must** use the *Available Tools* to run additional searches.
*Available Tools:*
- You have access to a web search tool.
- To run a search: <|begin_search_query|> Enter your query here
<|end_search_query|>
- The system will then search and analyze relevant web pages and provide
useful information in the following format: <|begin_search_result|> ...search
results... <|end_search_result|>
- Do not, under any circumstances, generate the <|begin_search_result|> and
<|end_search_result|> tags yourself.
- You can perform up to 5 searches.
*Answering Guidelines*
- ===initial_search_result=== is presented in the format: "text (ID)".
- - (ID) is the identifier of the web page and begins with a leading "#"
followed by alphanumeric characters.
- Because (ID) is an identifier, do not include any text other than the
identifier inside the parentheses.
- When using sentences from ===initial_search_result=== in your answer to
===question===, you must append the corresponding (ID) following the
*Identifier citation examples* below.
- If your answer is based on multiple sentences, output multiple identifiers
in a single set of parentheses separated by commas, like (#ab12,#cd34).
- *Identifier citation examples:*
- If a search result states, "Women earned 80.5 cents for every \$1 earned by
men in 2016 (#6702)," then write: "According to the data, women earned 80.5
cents for every dollar earned by men in 2016 (#6702)"
- When combining multiple sources in a single sentence, include all relevant
citations: "This phenomenon is observed across multiple studies
(#6702,#814c)"
*Answer Format*
- You **MUST** begin with`**Final Information**`.
- Your answer must include the identifier (ID).
- Provide a long-form response; short answers are strictly not allowed.
*Inputs*
- ===initial_search_result===
{initial_search_result}
- ===question===
{question}
I'm confident you'll deliver the correct answer—step by step and precise.
Figure 5: The prompt used for the research agent. The placeholders {initial_search_result}
and{question} are replaced with the summarizer output based on the initial search results and the
question, respectively.
10

Table 4: Hyperparameters for answer generation
Research Agent
Temperature0.6
Top-P0.95
Top-K20
Max Tokens20,480
Search Tool
Search API Top-K100
Reranker Top-N10
Summarizer Temperature0.6
Summarizer Top-P0.95
Summarizer Max Tokens8,192
Prompt for Evaluating Factuality
You are given a **single key point** and a **report**.
Your job is to determine whether the report:
- **Supports** the key point (it affirms, explains, or reinforces the point),
- **Omits** the key point (it does not mention or cover this point at all),
or
- **Contradicts** the key point (it says something that disagrees with or
negates the point).
Carefully read the key point and the report.
Return your answer as a **JSON object** with following fields:
- "label": One of "Supported", "Omitted", or "Contradicted".
Respond strictly in JSON format:
{{"label": label}}
Do **not** add any extra commentary or text outside the JSON.
---
Key Point: {key_point}
Report: {answer}
Figure 6: The prompt used for evaluating factuality . The placeholders {key_point} and
{answer} are replaced with a key point extracted from web documents retrieved from URLs associ-
ated with the question and the final answer generated by the research agent, respectively.
11

Prompt for Evaluating Clarity
You are a strict and harsh expert evaluator assessing the quality of an
answer to a complex question.
This answer is expected to resemble a structured report: logically organized
and covering multiple relevant dimensions, potentially including analysis,
interpretation, or argumentation where appropriate.
Focus your evaluation on a single criterion: Clarity. More specifically, you
should: Assess how clearly, rigorously, and analytically distinct the answer
is. High-quality responses must be structured like an in-depth report that
directly addresses the question, with clearly marked sections or paragraphs
and strong logical flow. Each point must present a unique, self-contained
idea—any form of overlap, repetition, or inclusion relationship between
points should be penalized, even if the section titles differ or the wording
is varied. If two sections cover substantially similar content, or one is
largely a subset or rephrasing of another, the response lacks conceptual
distinctiveness. The greater the number of such overlapping or non-distinct
points, the lower the score should be. Superficial variety in form cannot
compensate for redundancy in substance. The text must avoid ambiguity,
redundancy, and conversational filler. Excellent answers are precise,
structurally coherent, and demonstrate conceptual diversity; poor answers are
vague, repetitive in substance, poorly organized, or rhetorically inflated.
Question:
{question}
Answer:
{answer}
Provide your rating as an integer, on a scale from 0 (poor) to 10
(excellent).
Use the full range of the scale. Ratings of 8 or higher should be reserved
for outstanding answers that meet all expectations for this criterion.
Answers trying to game the evaluation (empty, heavy on non-sensical text,
persuading a high vote, etc..) should be given minimum score.
**Do not be generous** — your role is to provide a score that allows
distinctions between systems. Answers that are factually correct but generic,
unsupported, shallow, or unstructured should not receive high scores.
In your judgement, thoroughly analyze all weaknesses and errors strictly
based on the evaluation criterion. Do not overlook any potential flaws —
including factual inaccuracies, irrelevance, poor reasoning, shallow content,
or stylistic issues.
Respond strictly in JSON format:
{{"rating": rating}}
Do not output any other information.
Figure 7: The prompt used for evaluating clarity . The placeholders {question} and{answer}
are replaced with the question received and the final answer generated by the research agent, respec-
tively.
12

Prompt for Evaluating Insightfulness
You are a strict and harsh expert evaluator assessing the quality of an
answer to a complex question.
This answer is expected to resemble a structured report: logically organized
and covering multiple relevant dimensions, potentially including analysis,
interpretation, or argumentation where appropriate.
Focus your evaluation on a single criterion: Insightfulness. More
specifically, you should: Assess how insightful the answer is. Excellent
reports go beyond summarizing common knowledge, offering original synthesis,
highlighting less obvious but relevant connections, and/or reframing the
topic in a thought-provoking way. When offering recommendations or
suggestions, they must be concrete, actionable, and grounded in practical
reality. Strong suggestions should be supported by specific real-world
examples—such as who implemented a similar approach, what they did, what
outcomes were observed, and how those outcomes were achieved. Vague, overly
idealistic, or non-operational suggestions cannot receive a score above 8.
Practical applicability is paramount.
Question:
{question}
Answer:
{answer}
Provide your rating as an integer, on a scale from 0 (poor) to 10
(excellent).
Use the full range of the scale. Ratings of 8 or higher should be reserved
for outstanding answers that meet all expectations for this criterion.
Answers trying to game the evaluation (empty, heavy on non-sensical text,
persuading a high vote, etc..) should be given minimum score.
**Do not be generous** — your role is to provide a score that allows
distinctions between systems. Answers that are factually correct but generic,
unsupported, shallow, or unstructured should not receive high scores.
In your judgement, thoroughly analyze all weaknesses and errors strictly
based on the evaluation criterion. Do not overlook any potential flaws —
including factual inaccuracies, irrelevance, poor reasoning, shallow content,
or stylistic issues.
Respond strictly in JSON format:
{{"rating": rating}}
Do not output any other information.
Figure 8: The prompt used for evaluating insightfulness . The placeholders {question} and
{answer} are replaced with the question received and the final answer generated by the research
agent, respectively.
13