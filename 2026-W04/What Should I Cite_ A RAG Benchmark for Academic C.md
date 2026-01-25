# What Should I Cite? A RAG Benchmark for Academic Citation Prediction

**Authors**: Leqi Zheng, Jiajun Zhang, Canzhi Chen, Chaokun Wang, Hongwei Li, Yuying Li, Yaoxin Mao, Shannan Yan, Zixin Song, Zhiyuan Feng, Zhaolu Kang, Zirong Chen, Hang Zhang, Qiang Liu, Liang Wang, Ziyang Liu

**Published**: 2026-01-21 12:51:47

**PDF URL**: [https://arxiv.org/pdf/2601.14949v1](https://arxiv.org/pdf/2601.14949v1)

## Abstract
With the rapid growth of Web-based academic publications, more and more papers are being published annually, making it increasingly difficult to find relevant prior work. Citation prediction aims to automatically suggest appropriate references, helping scholars navigate the expanding scientific literature. Here we present \textbf{CiteRAG}, the first comprehensive retrieval-augmented generation (RAG)-integrated benchmark for evaluating large language models on academic citation prediction, featuring a multi-level retrieval strategy, specialized retrievers, and generators. Our benchmark makes four core contributions: (1) We establish two instances of the citation prediction task with different granularity. Task 1 focuses on coarse-grained list-specific citation prediction, while Task 2 targets fine-grained position-specific citation prediction. To enhance these two tasks, we build a dataset containing 7,267 instances for Task 1 and 8,541 instances for Task 2, enabling comprehensive evaluation of both retrieval and generation. (2) We construct a three-level large-scale corpus with 554k papers spanning many major subfields, using an incremental pipeline. (3) We propose a multi-level hybrid RAG approach for citation prediction, fine-tuning embedding models with contrastive learning to capture complex citation relationships, paired with specialized generation models. (4) We conduct extensive experiments across state-of-the-art language models, including closed-source APIs, open-source models, and our fine-tuned generators, demonstrating the effectiveness of our framework. Our open-source toolkit enables reproducible evaluation and focuses on academic literature, providing the first comprehensive evaluation framework for citation prediction and serving as a methodological template for other scientific domains. Our source code and data are released at https://github.com/LQgdwind/CiteRAG.

## Full Text


<!-- PDF content starts -->

What Should I Cite? A RAG Benchmark for
Academic Citation Prediction
Leqi Zheng∗
Tsinghua University
Beijing, China
zhenglq24@mails.tsinghua.edu.cnJiajun Zhang∗
zhangjiajun519@mail.ustc.edu.cn
University of Science and
Technology of China
Hefei, ChinaCanzhi Chen∗
Beijing Institute of
Technology
Beijing, China
chencanzhi@bit.edu.cnChaokun Wang†
chaokun@tsinghua.edu.cn
Tsinghua University
Beijing, China
Hongwei Li
Tsinghua University
Beijing, ChinaYuying Li
Tsinghua University
Beijing, ChinaYaoxin Mao
Beijing Institute of
Technology
Beijing, ChinaShannan Yan
Tsinghua University
Beijing, China
Zixin Song
Tsinghua University
Beijing, ChinaZhiyuan Feng
Tsinghua University
Beijing, ChinaZhaolu Kang
Peking University
Beijing, ChinaZirong Chen
Tsinghua University
Beijing, China
Hang Zhang
Tsinghua University
Beijing, ChinaQiang Liu
qiang.liu@nlpr.ia.ac.cn
Institute of Automation,
Chinese Academy of
Sciences
Beijing, ChinaLiang Wang
wangliang@nlpr.ia.ac.cn
Institute of Automation,
Chinese Academy of
Sciences
Beijing, ChinaZiyang Liu
Tsinghua University
Beijing, China
Abstract
With the rapid growth of Web-based academic publications, more
and more papers are being published annually, making it increas-
ingly difficult to find relevant prior work. Citation prediction aims
to automatically suggest appropriate references, helping scholars
navigate the expanding scientific literature. Here we presentCit-
eRAG, the first comprehensive retrieval-augmented generation
(RAG)-integrated benchmark for evaluating large language models
on academic citation prediction, featuring a multi-level retrieval
strategy, specialized retrievers, and generators. Our benchmark
makes four core contributions: (1) We establish two instances of the
citation prediction task with different granularity. Task 1 focuses on
coarse-grained list-specific citation prediction, while Task 2 targets
fine-grained position-specific citation prediction. To enhance these
two tasks, we build a dataset containing 7,267 instances for Task 1
and 8,541 instances for Task 2, enabling comprehensive evaluation
∗These authors contributed equally to this research.
†Chaokun Wang is the corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
WWW ’26, Dubai, United Arab Emirates.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2307-0/2026/04
https://doi.org/10.1145/3774904.3792075of both retrieval and generation. (2) We construct a three-level large-
scale corpus with 554k papers spanning many major subfields, using
an incremental pipeline. (3) We propose a multi-level hybrid RAG
approach for citation prediction, fine-tuning embedding models
with contrastive learning to capture complex citation relationships,
paired with specialized generation models. (4) We conduct exten-
sive experiments across state-of-the-art language models, including
closed-source APIs, open-source models, and our fine-tuned gen-
erators, demonstrating the effectiveness of our framework. Our
open-source toolkit enables reproducible evaluation and focuses on
academic literature, providing the first comprehensive evaluation
framework for citation prediction and serving as a methodological
template for other scientific domains. Our source code and data are
released at https://github.com/LQgdwind/CiteRAG.
CCS Concepts
•Information systems→Retrieval tasks and goals.
Keywords
Large Language Models, Retrieval Augmented Generation
ACM Reference Format:
Leqi Zheng, Jiajun Zhang, Canzhi Chen, Chaokun Wang, Hongwei Li, Yuy-
ing Li, Yaoxin Mao, Shannan Yan, Zixin Song, Zhiyuan Feng, Zhaolu Kang,
Zirong Chen, Hang Zhang, Qiang Liu, Liang Wang, and Ziyang Liu. 2026.arXiv:2601.14949v1  [cs.IR]  21 Jan 2026

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Leqi Zheng et al.
What Should I Cite? A RAG Benchmark for Academic Citation Prediction.
InProceedings of the ACM Web Conference 2026 (WWW ’26), April 13–17,
2026, Dubai, United Arab Emirates.ACM, New York, NY, USA, 12 pages.
https://doi.org/10.1145/3774904.3792075
1 Introduction
With the rapid advancement of scientific research on the Web, an
unprecedented number of academic papers are published every
day, forming a massive and ever-growing body of online scholarly
knowledge. At the heart of this Web-based scientific corpus lies
the citation network, which connects individual studies through
citation relationships and continuously integrates new findings into
the existing body of science. As citation networks play a central
role in the evolution of scientific knowledge, citation prediction,
which aims to forecast the prior studies a new paper will cite, has at-
tracted growing research attention. From an individual researcher’s
perspective, citation prediction can facilitate the discovery of rele-
vant prior work, improve the quality of scholarly references, and
foster interdisciplinary innovation. From a broader perspective, it
provides a powerful analytical tool for computational social science
[3,11,17], revealing the hidden structure of citation networks and
offering valuable insights into how knowledge propagates, trans-
forms, and gives rise to innovation.
Despite extensive research on citation prediction [ 4,35], there re-
mains a notable absence of a comprehensive benchmark for eval-
uating models in this domain. Existing benchmarks suffer from
several key limitations: (1)oversimplified task formulations, as
real-world users often query related studies using specific sections
of a paper, such as the introduction, methodology, or experimental
details, to identify the most relevant and fine-grained prior work [ 4],
whereas current benchmarks typically treat citation prediction as
a binary classification or document-level retrieval problem within
a fixed corpus, an oversimplification that diverges from practical
scenarios; (2)ignorance of the hierarchical structure of scien-
tific literature, since scientific texts are inherently layered and
rhetorically organized [ 16,30,35], yet current benchmarks often
flatten this complexity into a textual knowledge graph assuming
full-text availability, thus neglecting the multi-level organization of
scientific discourse; and (3)suboptimal methodological design,
as most existing approaches rely on fine-tuning simple embedding-
based retrievers [ 8,18,26], which struggle to capture the nuanced,
structural nature of citation relationships and tend to lose retrieval
effectiveness as the candidate set expands [ 18]; (4)a lack of sys-
tematic evaluation framework, as current benchmarks typically
focus on citation network construction and simple embedding-
based methods while failing to provide a comprehensive evaluation
of more advanced approaches such as retrieval-augmented genera-
tion (RAG) [ 13,19]. Other platforms, such as Deep Research-style
agents [ 43], focus on literature exploration rather than serving as
standardized benchmarks, and cannot serve as formal benchmarks
for citation prediction.
To address this gap, we introduce a comprehensive benchmark
for citation prediction that encompasses two complementary tasks
operating at different levels of granularity. List-specific citation
prediction (Task 1) evaluates a system’s ability to generate completeTable 1: Comparison of our proposed benchmark with existing ci-
tation datasets [ 1,5,27,31] across five key dimensions. Multidi-
mensional Evaluation means our work assesses not just retrieval or
classification, but also hallucination and diversity in RAG systems.
Dataset SciDocs unarXive REASONS LitSearch Ours
Multi-domain" " " " "
Structured Full-Text% " % % "
Support for RAG Tasks% % " " "
Multi-level Corpus% % % % "
Multidimensional Evaluation% % % % "
reference lists for papers. Position-specific citation prediction (Task
2) assesses a system’s capability to predict precise citations for
individual reference placeholders within the paper text. Building
on these two tasks, we collected a large corpus of scientific texts
from Google Scholar, resulting in a multidisciplinary, hierarchical
corpus of 554k papers. Each paper is represented with multi-level
textual information and associated with a large candidate set. To
facilitate the evaluation of citation prediction tasks, we further
curated 7,267 samples for Task 1 and 8,541 samples for Task 2.
Given the massive size of the candidate pool and the limitations
of conventional embedding-based methods in capturing citation
relations, we design a multi-level hybrid retrieval-augmented gen-
eration (RAG) approach for citation prediction. Our approach in-
tegrates a carefully fine-tuned, state-of-the-art 8B retriever with a
suite of task-specific large language models (LLMs). To establish a
solid foundation for future research, we conduct a comprehensive
evaluation of our RAG pipeline alongside a wide range of base-
line methods, including traditional embedding-based retrievers and
both open- and closed-source LLMs. The results highlight the in-
trinsic complexity of the citation prediction task and confirm the
effectiveness of our proposed approach. Finally, we release an open-
source toolkit that enables reproducible evaluation and provides
the first comprehensive framework dedicated to citation prediction.
Table 1 summarizes the key features of our benchmark compared to
existing datasets. Our benchmark is multidisciplinary and supports
RAG-based baselines. It also includes the evaluation for hallucina-
tion and diversity, is structured as a hierarchical RAG pipeline, and
provides a reproducible open-source toolkit.
In summary, our contributions are fourfold:
(1)Task Definition and Granularity Distinction.We define two
novel instances of the citation prediction task: Task 1, focusing on
coarse-grained list-specific prediction, and Task 2, targeting fine-
grained position-specific prediction. The dataset we built contains
7,267 instances for Task 1 and 8,541 instances for Task 2, enabling
comprehensive evaluation of both retrieval and generation across
different task granularities. (Section 2 and Section 3)
(2)Multi-Level Incremental Corpus Construction.We con-
struct a large-scale hierarchical corpus with 554,719 papers, built
using an incremental pipeline that supports continuous updates.
This enables flexible access to multi-level information and scalable
expansion of the academic knowledge base for citation prediction
tasks. (Section 3)

What Should I Cite? A RAG Benchmark for
Academic Citation Prediction WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
(3)Multi-Level Hybrid Retrieval and Generation Framework.
We propose a multi-level hybrid RAG approach, combining spe-
cialized retrievers tuned for different corpus levels with generation
models fine-tuned to capture complex citation relationships. This
multi-level hybrid RAG approach improves both retrieval precision
and context-aware citation generation compared to single-level
methods. (Section 4)
(4)Open-Source RAG Evaluation Toolkit.We establish a com-
prehensive evaluation framework covering the full citation predic-
tion pipeline, from retrieval effectiveness to end-to-end generation
performance. The toolkit provides standardized metrics, evalua-
tion protocols, and extensive comparison across state-of-the-art
models, including closed-source APIs, open-source LLMs, and our
fine-tuned generators, enabling reproducible benchmarking and
facilitating further research. We hope this toolkit serves as a foun-
dation for future work. (Section 5, Section 6 and Section 7)
2 Preliminaries and Task Definition
We formulate academic citation prediction as a retrieval-augmented
generation (RAG) problem operating at two distinct granularities to
evaluate the capabilities of different large language models. The first
task addresses coarse-grained citation prediction for entire papers,
while the second focuses on fine-grained citation prediction for
specific textual positions.
2.1 Task 1: List-Specific Citation Prediction
The list-specific citation prediction task aims to predict the complete
reference list for a given academic paper. Let C={𝑐 1,𝑐2,...,𝑐𝑛}
denote the corpus of available papers. Given a query paper 𝑞with
title𝑇𝑞and abstract 𝐴𝑞, the task generates a ranked list of reference
papers:
𝑅𝑞={𝑟 1,𝑟2,...,𝑟𝑘}, 𝑟𝑖∈C(1)
where𝑘is the number of predicted citations and the ranking func-
tion𝑓:(𝑇𝑞,𝐴𝑞)→𝑅𝑞maps the input to an ordered sequence
reflecting citation relevance.
This coarse-grained formulation employs a fixed-length prefix chunk-
ing strategy, using title and abstract (typically 200-400 tokens) to
represent each paper. This design addresses practical constraints in
academic literature where full-text access is frequently restricted by
paywalls or copyright, while titles and abstracts remain universally
accessible through bibliographic databases. The approach balances
computational efficiency with real-world applicability across het-
erogeneous corpus conditions.
2.2 Task 2: Position-Specific Citation Prediction
The position-specific citation prediction task operates at finer gran-
ularity, predicting the specific paper to cite at individual reference
placeholders within the text. Let 𝑞be a query paper with title 𝑇𝑞, ab-
stract𝐴𝑞, and textual sections 𝑆𝑞containing reference placeholders
{[𝑟𝑒𝑓] 1,[𝑟𝑒𝑓] 2,...,[𝑟𝑒𝑓]𝑚}. For each reference placeholder[𝑟𝑒𝑓] 𝑖
with surrounding context X𝑖⊂𝑆𝑞, the task predicts the appropriate
citation:ˆ𝑟𝑖=𝑔(𝑇𝑞,𝐴𝑞,X𝑖;C), ˆ𝑟𝑖∈C(2)
where𝑔is the prediction function that selects the most relevant
paper from corpusCbased on the local context X𝑖and paper meta-
data.
This fine-grained formulation requires understanding the imme-
diate textual context surrounding each reference placeholder. The
prediction system must analyze specific claims, methodologies, or
concepts being discussed, identify the required citation support
type, and select the most appropriate reference. This reflects the
actual writing process where authors make precise decisions about
citations at each point in their argumentation, considering chrono-
logical precedence, methodological relevance, and argumentative
support.
3 Corpus and Dataset Construction
This section details the construction of the RAG benchmark dataset,
which encompasses both the core citation prediction dataset and
comprehensive external knowledge corpora. The dataset construc-
tion process involves three primary phases: systematic data col-
lection, rigorous preprocessing and quality control measures, and
the creation of specialized retrieval corpora to support end-to-end
citation prediction tasks.
3.1 Data Collection
To construct a comprehensive corpus for academic citation pre-
diction, we systematically collected papers from Google Scholar
over the past decade, ensuring broad temporal coverage of evolving
research trends and citation patterns. The collection framework
implements adaptive query strategies using year-based filtering for
categories with fewer than 2,000 papers, month-by-month querying
for 2,000-12,000 papers, and week-by-week collection for categories
exceeding 12,000 papers to prevent API-related data loss.
3.2 Corpus and Dataset Pre-processing
3.2.1 Corpus Pre-processing.The corpus preprocessing pipeline
implements a three-tier hierarchical structure for multi-granular
citation prediction evaluation. Level 1 contains domain classifica-
tions, titles, and abstracts for rapid relevance assessment. Level
2 adds introduction sections to Level 1 components, enabling a
deeper understanding of research motivation and methodology.
Level 3 incorporates full-text content with references removed,
plus conclusion sections, providing comprehensive coverage while
maintaining clean input for generation models. The preprocess-
ing framework employs pattern-matching algorithms to extract
introduction sections and removes citation markers to eliminate
prediction interference. All levels maintain consistent metadata,
including authorship, publication year, and venue information, sup-
porting both coarse-grained and fine-grained citation prediction
tasks through appropriate information density matching.
3.2.2 Dataset Pre-processing.The List-Specific Citation Prediction
(Task 1) preprocessing establishes quality control measures to iden-
tify papers with sufficient citation information for meaningful eval-
uation. We filter out papers with non-standard citations (including

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Leqi Zheng et al.
Figure 1: Overview of the CiteRAG benchmark pipeline comprising three stages: (1) Data Collection and Corpus Pre-processing constructs a
three-level hierarchical corpus from web-crawled papers; (2) Dataset Pre-processing filters and formats data into Task 1 and Task 2 QA pairs
with test instances removed from the corpus; (3) Retrieval-Augmented Generation Pipeline trains embedding models and retrievers, then
applies multi-level hybrid retrieval to feed generators for task-specific citation prediction and evaluation.
duplicate citations, citation errors, etc.) and retain essential compo-
nents, including identifiers, titles, abstracts, and complete reference
collections, while removing extraneous metadata. This approach
ensures adequate citation diversity for comprehensive list-level
prediction assessment.
The Position-Specific Citation Prediction (Task 2) preprocessing
employs a more sophisticated multi-stage approach to create fine-
grained citation scenarios. We also initially filter out papers with
non-standard citations, then analyze citation distribution patterns
to identify the three most frequently cited sections within each
paper. For these high-citation sections, we preserve the top three
most frequent citations with valid identifiers while removing other
citation markers, creating clean textual contexts with precisely
positioned reference placeholders. Papers where any single refer-
ence appears more than ten times within a section are excluded to
prevent citation bias.
To ensure rigorous evaluation integrity and prevent data contam-
ination, all papers successfully processed and incorporated into
either Task 1 or Task 2 datasets are permanently removed from the
original corpus.
3.3 Corpus and Dataset Statistics
The comprehensive statistics of our benchmark corpus and evalua-
tion datasets are presented in Figure 2. The datasets are formatted as
question-answer pairs following the ChatML convention detailed in
Appendix A, enabling seamless integration with standard language
model training frameworks.4 Retriever Design
Academic papers exhibit hierarchical textual structures with vary-
ing information density across different granularity levels. To ad-
dress the challenges of multi-granular information access and lim-
ited full-text availability, we develop a specialized retriever opti-
mized for academic citation prediction tasks.
4.1 Multi-Level Hybrid Retrieval Strategy
We implement a retriever system Mthat directly leverages the
three-level corpus hierarchy constructed in Section 3. The retriever
system is formally defined as M=(Encode,Retrieve,𝑆) where
Encode represents the embedding function, Retrieve denotes the
similarity search function, and 𝑆is the ranking fusion strategy. This
architecture enables multi-granular information access across the
pre-existing corpus levels D1,D2,D3, which correspond to Level
1, Level 2, and Level 3 corpus representations respectively.
Our chunking strategy employs title and abstract content as stan-
dardized query input, maintaining consistency with Task 1 require-
ments while enabling efficient retrieval across all corpus granu-
larities. Given query paper 𝑞consisting of title and abstract, the
retrieval system performs parallel top-k similarity search across
the three pre-established corpus levels:
R1
𝑘=Retrieve(𝑞,D1),(3)
R2
𝑘=Retrieve(𝑞,D2),(4)
R3
𝑘=Retrieve(𝑞,D3).(5)

What Should I Cite? A RAG Benchmark for
Academic Citation Prediction WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Figure 2: Corpus and dataset statistics. The left sunburst chart shows three-level organization with 554,719 total papers across 10 fields,
training/test distributions, and metadata structure. The upper right panels display citation frequency distributions for Task 1 and Task 2. The
lower right panels show token composition breakdowns and response token length distributions for both tasks.
Results are merged using reciprocal rank fusion:
R∗
𝑘=𝑆 R1
𝑘,R2
𝑘,R3
𝑘.(6)
This architecture enhances retrieval diversity while maintaining
robustness across varying corpus availability conditions.
4.2 Contrastive Learning Fine-tuning
Standard embedding models are inadequately optimized for aca-
demic citation relationships, which encompass complex interdepen-
dencies beyond semantic similarity. We fine-tune Qwen-3-embedding-
8B as CitationRetriever-8B using contrastive learning to capture
citation-specific patterns.
From the training corpus, we select 30,000 query papers and their
cited references, constructing 400,000 query-positive pairs (𝑞,𝑑+)
across all granularity levels. We train the model using InfoNCE loss
with temperature scaling:
L=−logexp(sim(𝑞,𝑑+)/𝜏)Í
𝑑−∈N(𝑞) exp(sim(𝑞,𝑑−)/𝜏)+exp(sim(𝑞,𝑑+)/𝜏)(7)
where sim(·,·) denotes cosine similarity, 𝜏is the temperature pa-
rameter, andN(𝑞) represents negative samples from approximate
nearest neighbors and in-batch negatives.
The resulting retriever model, CitationRetriever-8B, demonstrates
enhanced capability in identifying complex citation relationships
beyond surface-level semantic matching.5 Evaluation Metrics
The evaluation framework employs distinct metrics tailored to each
component of the RAG pipeline, ensuring comprehensive assess-
ment of both retrieval effectiveness and generation quality. The
metrics are designed to capture different aspects of system perfor-
mance while maintaining consistency with established practices in
information retrieval and citation analysis.
5.1 Retriever Evaluation Metrics
We employ two standard information retrieval metrics to evaluate
retriever performance: Recall@k and Mean Reciprocal Rank@k
(MRR@k). Detailed mathematical formulations are provided in
Appendix B.
5.2 Task 1 Evaluation Metrics
Task 1 evaluation employs three standard metrics for reference
generation: Recall@k, Normalized Discounted Cumulative Gain@k
(NDCG@k), and Hit@k. Complete definitions are provided in Ap-
pendix B.
5.3 Task 2 Evaluation Metrics
Task 2 evaluation employs Position-Aware Citation Accuracy@k
(PACA@k), a novel metric we propose to assess position-specific
citation prediction quality.
Position-Aware Citation Accuracy@k (PACA@k)accounts for
both correctness and ranking positions:
PACA@k=1
𝑁𝑁∑︁
𝑖=1∑︁
𝑗∈correct@k
1−rank𝑗−1
𝑘
(8)

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Leqi Zheng et al.
where𝑁is the total number of reference placeholders, correct@k
denotes correct predictions within top-k, and rank𝑗is the position
of correct prediction𝑗.
PACA@k assigns higher scores to correct predictions at earlier
ranks, reflecting that users typically consider only top-ranked sug-
gestions when selecting citations for specific positions.
5.4 Diversity and Quality Metrics
Beyond retrieval accuracy, we propose two complementary met-
rics to assess the quality and diversity of predicted reference lists:
Citation Diversity Entropy and Hallucination Rate.
Citation Diversity Entropy (CDE)measures the diversity of
predicted citations across different research categories. For list-
specific citation prediction (Task 1), given a predicted reference
list with citations distributed across 𝐶categories, let 𝑝𝑖denote the
proportion of citations in category 𝑖. The Citation Diversity Entropy
is computed as:
CDE=−𝐶∑︁
𝑖=1𝑝𝑖log2(𝑝𝑖)(9)
where higher entropy values indicate more diverse citation patterns
across research areas, while lower values suggest concentration
in fewer categories. The maximum possible entropy is log2(𝐶),
achieved when citations are uniformly distributed across all cate-
gories.
Hallucination Rate (Halluc.)quantifies the proportion of pre-
dicted citations that refer to non-existent papers, indicating model
reliability. For a predicted reference list 𝑃, let𝑉denote the set
of papers that can be verified to exist in the real world through
bibliographic databases. The Hallucination Rate is defined as:
Halluc.=|𝑃\𝑉|
|𝑃|×100%(10)
where𝑃\𝑉 represents predictions that cannot be verified as real
papers. Lower hallucination rates indicate better model grounding
and reduced generation of fabricated references, which is critical
for maintaining academic integrity in citation prediction systems.
6 Experimental Setup
6.1 Baselines
6.1.1 Closed-source LLMs.We evaluate leading commercial lan-
guage models from major AI companies, including OpenAI (GPT-o3
and GPT-5), Anthropic (Claude-3.5, Claude-3.7, and Claude Sonnet
4), Google (Gemini-2.5-Pro and Gemini-2.0-Flash), and xAI (Grok-3).
6.1.2 Open-source LLMs.The open-source evaluation encompasses
prominent publicly available models, including Moonshot AI (Kimi-
K2), DeepSeek (DeepSeek-V3), OpenAI (GPT-oss-120B), and Qwen
(Qwen3-Coder-480B). Additionally, we employ the Qwen3 series
as our primary experimental models CitationGenerator to demon-
strate specialized model adaptation for academic citation prediction
tasks.6.2 Experimental Settings
We conduct the following fourfold experiments: (1)Few-shot In-
context Learning:Leverages large language models’ contextual
understanding without parameter updates by constructing prompts
with representative training examples, task-specific instructions,
and JSON output formats, emphasizing holistic paper comprehen-
sion for List-Specific Citation Prediction and contextual analysis
around reference placeholders for Position-Specific Citation Predic-
tion, while incorporating reasoning to prevent hallucination and
ensure grounded predictions. (2)Retrieval Augmented Genera-
tion:Integrates external knowledge retrieval with generation in a
two-stage pipeline, querying a hierarchical corpus using paper titles
and abstracts to return top-R relevant papers (R=5 or R=10) based
on semantic similarity, which are then incorporated into genera-
tion prompts alongside target paper content to ground predictions
in actual corpus content. (3)Supervised Fine-tuning:Adapts
pre-trained language models for citation prediction through pa-
rameter optimization on task-specific data, using question-answer
formats with paper metadata as input and citation predictions in
JSON schemas as targets, with quality-filtered and consistently for-
matted training data to ensure stable convergence. (4)Supervised
Fine-tuning with Retrieval Augmented Generation:Combines
parametric adaptation with external knowledge access by first fine-
tuning on citation-specific tasks and then integrating retrieved
papers during training and inference, enabling models to leverage
both adapted parametric knowledge and external corpus content.
6.3 Implementation Details
Our retriever module was fine-tuned from the Qwen3-Embedding-
8B [39] model using the Swift framework. The training was config-
ured with a learning rate of 5e-5, a batch size of 16, and a maximum
sequence length of 4,608. For the generator, we fine-tuned the
Qwen3-4B [ 34] and Qwen3-Coder-30B [ 12] models utilizing the
Megatron-LM [ 28] framework. These models were trained with a
learning rate of 7e-6, a batch size of 64, and a maximum sequence
length of 131,072.
For inference, all open-source models were served using vLLM,
while closed-source models were queried via their official APIs. To
ensure deterministic and focused outputs, we set the generation
temperature to 0.1 and the presence penalty to 1.0 for all model
requests, wherever supported by the respective API1. All training
and deployment were conducted on a cluster of 32 NVIDIA A100
GPUs.
7 Experimental Results
7.1 Results on Task 1 and Task 2
7.1.1 Overall Ratings.Table 2 presents comprehensive performance
comparisons across both citation prediction tasks. RAG consistently
improves performance across all models, with retrieval depth sig-
nificantly impacting prediction quality. Increasing retrieval from
R=5 to R=10 yields substantial improvements, with models like
Grok-3 and Claude-3.5 showing over 80% relative improvement in
Task 1 Recall@20. Task 2 exhibits even more pronounced benefits,
1GPT-o3 does not support custom temperature or max_tokens settings. Claude models
do not support presence penalty configuration.

What Should I Cite? A RAG Benchmark for
Academic Citation Prediction WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Table 2: Performance comparison across citation prediction tasks. Bold indicates best overall performance, underline indicates best baseline
performance. * indicates statistical significance (𝑝<0.05).
ModelTasks Task1 Task2
Metrics Recall@20 Recall@40 NDCG@20 NDCG@40 Hit@20 Hit@40 PACA@10 PACA@20 PACA@40
Closed-source LLMs
GPT-5w/o RAG 0.061 0.090 0.263 0.212 0.115 0.176 0.053 0.069 0.081
RAG (R=5) 0.065↑6.56% 0.093↑3.33% 0.306↑16.3% 0.231↑8.96% 0.124↑7.83% 0.181↑2.84% 0.086↑62.3% 0.109↑58.0% 0.131↑61.7%
RAG (R=10) 0.071↑16.4% 0.099↑10.0%0.324↑23.2% 0.248↑17.0% 0.136↑18.3% 0.193↑9.66%0.098↑84.9% 0.121↑75.4% 0.143↑76.5%
GPT-o3w/o RAG 0.029 0.034 0.166 0.113 0.059 0.068 0.030 0.036 0.041
RAG (R=5) 0.044↑51.7% 0.051↑50.0% 0.230↑38.6% 0.157↑38.9% 0.086↑45.8% 0.101↑48.5% 0.039↑30.0% 0.049↑36.1% 0.060↑46.3%
RAG (R=10) 0.055↑89.7% 0.064↑88.2% 0.264↑59.0% 0.184↑62.8% 0.104↑76.3% 0.126↑85.3% 0.045↑50.0% 0.058↑61.1% 0.068↑65.9%
Claude-3.5w/o RAG 0.028 0.034 0.140 0.098 0.055 0.067 0.024 0.029 0.034
RAG (R=5) 0.044↑57.1% 0.050↑47.1% 0.230↑64.3% 0.157↑60.2% 0.084↑52.7% 0.098↑46.3% 0.046↑91.7% 0.056↑93.1% 0.064↑88.2%
RAG (R=10) 0.054↑92.9% 0.062↑82.4% 0.264↑88.6% 0.180↑83.7% 0.103↑87.3% 0.121↑80.6% 0.057↑138% 0.069↑138% 0.078↑129%
Claude-3.7w/o RAG 0.042 0.054 0.197 0.144 0.081 0.107 0.040 0.054 0.071
RAG (R=5) 0.057↑35.7% 0.071↑31.5% 0.275↑39.6% 0.196↑36.1% 0.110↑35.8% 0.140↑30.8% 0.063↑57.5% 0.081↑50.0% 0.099↑39.4%
RAG (R=10) 0.065↑54.8% 0.081↑50.0% 0.300↑52.3% 0.217↑50.7% 0.126↑55.6% 0.162↑51.4% 0.082↑105% 0.105↑94.4% 0.125↑76.1%
Claude-4-Sonnetw/o RAG 0.038 0.051 0.187 0.139 0.076 0.103 0.046 0.064 0.080
RAG (R=5) 0.054↑42.1% 0.068↑33.3% 0.268↑43.3% 0.193↑38.8% 0.100↑31.6% 0.136↑32.0% 0.076↑65.2% 0.097↑51.6% 0.115↑43.8%
RAG (R=10) 0.064↑68.4% 0.080↑56.9% 0.296↑58.3% 0.213↑53.2% 0.122↑60.5% 0.157↑52.4% 0.080↑73.9% 0.101↑57.8% 0.122↑52.5%
Gemini-2.0-Flashw/o RAG 0.023 0.030 0.115 0.084 0.045 0.060 0.028 0.039 0.051
RAG (R=5) 0.035↑52.2% 0.043↑43.3% 0.180↑56.5% 0.127↑51.2% 0.068↑51.1% 0.086↑43.3% 0.082↑193% 0.105↑169% 0.134↑163%
RAG (R=10) 0.048↑109% 0.058↑93.3% 0.230↑100% 0.162↑92.9% 0.092↑104% 0.114↑90.0% 0.107↑282% 0.138↑254% 0.171↑235%
Gemini-2.5-Prow/o RAG 0.055 0.069 0.272 0.195 0.108 0.138 0.071 0.092 0.113
RAG (R=5) 0.069↑25.5% 0.087↑26.1% 0.329↑21.0% 0.235↑20.5% 0.134↑24.1% 0.170↑23.2% 0.099↑39.4% 0.129↑40.2% 0.155↑37.2%
RAG (R=10) 0.076↑38.2%0.097↑40.6% 0.346↑27.2%0.252↑29.2%0.144↑33.3%0.188↑36.2% 0.127↑78.9%0.170↑84.8%0.201↑77.9%
Grok-3w/o RAG 0.016 0.017 0.090 0.060 0.031 0.034 0.030 0.043 0.054
RAG (R=5) 0.024↑50.0% 0.025↑47.1% 0.134↑48.9% 0.087↑45.0% 0.047↑51.6% 0.049↑44.1% 0.082↑173% 0.092↑114% 0.098↑81.5%
RAG (R=10) 0.040↑150% 0.042↑147% 0.192↑113% 0.122↑103% 0.077↑148% 0.079↑132% 0.101↑237% 0.122↑184% 0.133↑146%
Open-source LLMs
Kimi-K2w/o RAG 0.036 0.046 0.177 0.127 0.071 0.091 0.048 0.066 0.080
RAG (R=5) 0.052↑44.4% 0.064↑39.1% 0.245↑38.4% 0.175↑37.8% 0.100↑40.8% 0.126↑38.5% 0.085↑77.1% 0.104↑57.6% 0.120↑50.0%
RAG (R=10) 0.061↑69.4% 0.075↑63.0% 0.273↑54.2% 0.195↑53.5% 0.117↑64.8% 0.146↑60.4% 0.098↑104% 0.123↑86.4% 0.143↑78.7%
Deepseek-v3w/o RAG 0.028 0.033 0.143 0.099 0.055 0.067 0.048 0.063 0.073
RAG (R=5) 0.050↑78.6% 0.058↑75.8% 0.248↑73.4% 0.171↑72.7% 0.096↑74.5% 0.114↑70.1% 0.103↑115% 0.131↑108% 0.153↑110%
RAG (R=10) 0.060↑114% 0.070↑112% 0.278↑94.4% 0.193↑94.9% 0.111↑102% 0.137↑104% 0.108↑125% 0.135↑114% 0.158↑116%
Qwen3-Coder-480Bw/o RAG 0.019 0.023 0.100 0.071 0.037 0.047 0.042 0.053 0.065
RAG (R=5) 0.037↑94.7% 0.042↑82.6% 0.177↑77.0% 0.121↑70.4% 0.072↑94.6% 0.083↑76.6% 0.101↑140% 0.121↑128% 0.138↑112%
RAG (R=10) 0.049↑158% 0.054↑135% 0.216↑116% 0.147↑107% 0.095↑157% 0.106↑126% 0.113↑169% 0.140↑164% 0.161↑148%
GPT-oss-120Bw/o RAG 0.016 0.018 0.095 0.063 0.033 0.037 0.024 0.028 0.031
RAG (R=5) 0.027↑68.8% 0.031↑72.2% 0.155↑63.2% 0.103↑63.5% 0.054↑63.6% 0.060↑62.2% 0.069↑188% 0.082↑193% 0.090↑190%
RAG (R=10) 0.043↑169% 0.046↑156% 0.219↑131% 0.144↑129% 0.083↑152% 0.090↑143% 0.087↑263% 0.108↑286% 0.122↑294%
CitationGenerator-4Bw/o All 0.006 0.007 0.029 0.021 0.013 0.017 0.009 0.010 0.012
w/o SFT (R=5) 0.011↑83.3% 0.012↑71.4% 0.056↑93.1% 0.034↑61.9% 0.024↑84.6% 0.027↑58.8% 0.051↑467% 0.064↑540% 0.074↑517%
w/o SFT (R=10) 0.021↑250% 0.023↑229% 0.122↑321% 0.078↑271% 0.041↑215% 0.045↑165% 0.063↑600% 0.078↑680% 0.092↑667%
w/o RAG 0.012 0.019 0.055 0.047 0.022 0.039 0.017 0.029 0.042
Full (R=5) 0.023↑91.7% 0.034↑78.9% 0.107↑94.5% 0.084↑78.7% 0.042↑90.9% 0.065↑66.7% 0.087↑412% 0.096↑231% 0.101↑140%
Full (R=10) 0.027↑125% 0.039↑105% 0.129↑135% 0.098↑109% 0.053↑141% 0.074↑89.7% 0.119↑600% 0.148↑410% 0.164↑290%
CitationGenerator-30Bw/o All 0.010 0.011 0.057 0.038 0.020 0.023 0.014 0.016 0.017
w/o SFT (R=5) 0.021↑110% 0.024↑118% 0.117↑105% 0.076↑100% 0.039↑95.0% 0.042↑82.6% 0.076↑443% 0.084↑425% 0.089↑424%
w/o SFT (R=10) 0.031↑210% 0.035↑218% 0.153↑168% 0.100↑163% 0.059↑195% 0.063↑174% 0.096↑586% 0.114↑613% 0.124↑629%
w/o RAG 0.043 0.069 0.225 0.198 0.113 0.130 0.149 0.165 0.191
Full (R=5) 0.072↑67.4% 0.113↑63.8% 0.338↑50.2% 0.273↑37.9% 0.183↑61.9% 0.204↑56.9% 0.268↑79.9% 0.291↑76.4% 0.318↑66.5%
Full (R=10) 0.076∗
↑76.7%0.122∗
↑76.8%0.367∗
↑63.1%0.295∗
↑49.0%0.193∗
↑70.8%0.219∗
↑68.5%0.273∗
↑83.2%0.303∗
↑83.6%0.342∗
↑79.1%
with Gemini-2.0-Flash achieving over 200% improvement in PACA
metrics with RAG integration. Supervised fine-tuning demonstrates
remarkable effectiveness, as CitationGenerator-30B without RAG
achieves competitive performance with leading closed-source mod-
els that use retrieval, particularly excelling in Task 2 with PACA@20
of 0.165. The combination of fine-tuning and RAG produces syn-
ergistic effects, with CitationGenerator-30B (R=10) achieving the
highest performance: Recall@20 of 0.076, NDCG@20 of 0.367 for
Task 1, and PACA@20 of 0.303 for Task 2. Baseline capabilities vary
considerably, with closed-source models demonstrating stronger
zero-shot performance, though this gap narrows substantially with
domain-specific fine-tuning.7.1.2 Ablation Study.Table 2 demonstrates that supervised fine-
tuning and retrieval augmentation contribute through distinct yet
complementary mechanisms in our CitationGenerator models. Fine-
tuning provides the primary performance gain, with w/o RAG con-
figurations substantially outperforming w/o SFT variants across
both model sizes, demonstrating that domain adaptation is essen-
tial for capturing academic citation patterns. Retrieval augmenta-
tion further enhances fine-tuned models, with Full configurations
achieving significant improvements over w/o RAG baselines. Model
capacity plays a critical role, as CitationGenerator-4B shows limited
effectiveness across all configurations while CitationGenerator-30B
achieves state-of-the-art performance, highlighting the importance
of sufficient parameter scale for complex citation prediction tasks.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Leqi Zheng et al.
Table 3: Results on retriever comparison of different models.
Model Recall@20 Recall@50 MRR@20 MRR@50
TF-IDF 0.0212 0.0332 0.1109 0.1145
BM25 0.0283 0.0426 0.1402 0.1435
BGE-M3 0.0326 0.0481 0.1611 0.1643
All-MPNet-Base-V2 0.0366 0.0566 0.1670 0.1703
Multilingual-E5-Large 0.0332 0.0496 0.1634 0.1667
Qwen3-Embedding-8B 0.0522 0.0775 0.2163 0.2186
CitationRetriever-8B0.0970 0.1386 0.3219 0.3232
Figure 3: Performance Comparison of Single-Level vs. Multi-Level
Fused Retrieval Strategies
7.2 Retriever Comparison
7.2.1 Retriever Performance Comparison.Table 3 demonstrates
substantial performance advantages of neural embedding mod-
els over traditional sparse retrieval methods for academic citation
prediction. Traditional approaches including TF-IDF and BM25
achieve limited effectiveness with MRR@50 of 0.1145 and 0.1435.
General-purpose neural models including BGE-M3 and All-MPNet-
Base-V2 show moderate improvements with MRR@50 around 0.16-
0.17, yet remain insufficient for citation-specific patterns. Qwen3-
Embedding-8B exhibits stronger performance with MRR@50 of
0.2186 due to large-scale pre-training, though it lacks specialized
adaptation for citation relationships. Our CitationRetriever-8B achieves
MRR@50 of 0.3232, representing 47.9% improvement over Qwen3-
Embedding-8B and nearly double the performance of general-purpose
models, demonstrating that contrastive learning on citation data
enables capturing complex relationships including shared method-
ologies and field-specific conventions.
7.2.2 Ablation on Multi-level Corpus.Figure 3 illustrates that multi-
level fusion consistently outperforms single-level retrieval across all
embedding models. Traditional models show modest improvements
of 0.2-2.7% in MRR@50, while Qwen3-Embedding-8B demonstrates
1.1% improvement. CitationRetriever-8B exhibits the most substan-
tial gains with 3.7% improvement over its best single-level config-
uration, indicating that contrastive fine-tuning enables learning
complementary relevance signals across granularities where Level
1 captures topical alignment, Level 2 incorporates methodologi-
cal context, and Level 3 provides comprehensive content coverage.
These results validate that academic papers require multi-level
representation for effective citation prediction.8 Related Work
8.1 Retrieval-Augmented Generation
Traditional retrieval methods [ 40–42] are no longer sufficient to
meet the demands of the era of large language models [ 7,10,12,15,
20,29,36–38], retrieval-augmented generation (RAG) has become
a key paradigm for mitigating large language models’ limitations
in domain-specific or time-sensitive tasks, where models often
hallucinate or generate outdated responses [ 32]. RAG addresses this
via a three-stage pipeline: retrieval from external knowledge bases,
query–context integration, and grounded response generation [ 2].
Subsequent work improves RAG through enhanced retrievers [ 19],
adaptive retrieval strategies [ 14], reasoning-based prompting such
as IR-CoT [ 33], and graph-structured retrieval like GraphRAG [ 6].
However, these general frameworks lack the hierarchical corpus
organization and multi-granularity retrieval needed for academic
literature, where citation reasoning spans titles, abstracts, and full
texts.
8.2 Citation Generation and Evaluation
Recent RAG systems have also enabled citation-aware text gen-
eration for verifiability. Early models such as WebGPT [ 22] and
reinforcement-based methods [ 21] laid the groundwork, while later
systems including ReGen [ 23] and WebCPM [ 24] refined citation in-
tegration through improved training. Recent advances in few-shot
learning, self-reflection mechanisms, and fine-grained reward mod-
eling have further improved citation quality. Evaluation has evolved
from manual annotation (AIS [ 25]) to automated and benchmark-
based approaches such as ALCE [ 9]. However, existing work mainly
targets Web-based question answering and lacks support for aca-
demic citation prediction, which requires both document-level ref-
erence modeling and position-specific citation reasoning.
9 Conclusion
In this paper, we proposeCiteRAG, the first comprehensive RAG-
integrated benchmark for evaluating large language models on aca-
demic citation prediction. The benchmark features dual-granularity
tasks, a hierarchical corpus of 554k papers, and standardized eval-
uation framework. Extensive experiments across state-of-the-art
language models demonstrate that RAG integration consistently
enhances prediction accuracy and reduces hallucination rates. Our
results reveal that specialized domain adaptation through super-
vised fine-tuning provides substantial performance gains, while
retrieval augmentation offers complementary benefits through ex-
ternal knowledge grounding. Our open-source toolkit provides the
research community with reproducible evaluation protocols and
comprehensive baselines, establishing a methodological foundation
for advancing citation prediction systems.
Acknowledgments
This work is supported in part by the National Natural Science
Foundation of China (No. 62372264 and No. 92467203) and Sina
Weibo Corp. Chaokun Wang is the corresponding author.

What Should I Cite? A RAG Benchmark for
Academic Citation Prediction WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
References
[1]Anirudh Ajith, Mengzhou Xia, Alexis Chevalier, Tanya Goyal, Danqi Chen, and
Tianyu Gao. 2024. LitSearch: A Retrieval Benchmark for Scientific Literature
Search. InProceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing. 15068–15083.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
[3]Valerio Ciotti, Moreno Bonaventura, Vincenzo Nicosia, Pietro Panzarasa, and
Vito Latora. 2016. Homophily and missing links in citation networks. doi:10.
1140/epjds/s13688-016-0062-2
[4]Arman Cohan, Waleed Ammar, Madeleine van Zuylen, and Field Cady. 2019.
Structural Scaffolds for Citation Intent Classification in Scientific Publications.
3586–3596 pages. doi:10.18653/v1/N19-1361
[5]Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S. Weld.
2020. SPECTER: Document-level Representation Learning using Citation-
informed Transformers. InProceedings of the 58th Annual Meeting of the Associ-
ation for Computational Linguistics. Association for Computational Linguistics,
2270–2282. doi:10.18653/v1/2020.acl-main.207
[6]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization.
[7]Zhiyuan Feng, Zhaolu Kang, Qijie Wang, Zhiying Du, Jiongrui Yan, Shubin Shi,
Chengbo Yuan, Huizhi Liang, Yu Deng, Qixiu Li, et al .2025. Seeing Across Views:
Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes.
arXiv preprint arXiv:2510.19400(2025).
[8]Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive
learning of sentence embeddings. https://aclanthology.org/2021.emnlp-main.
552/
[9]Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling Large
Language Models to Generate Text with Citations. 6465–6488 pages.
[10] Yingjie He, Zhaolu Kang, Kehan Jiang, Qianyuan Zhang, Jiachen Qian, Chunlei
Meng, Yujie Feng, Yuan Wang, Jiabao Dou, Aming Wu, et al .2026. How Order-
Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction.
arXiv preprint arXiv:2601.08626(2026).
[11] Jake M. Hofman, Duncan J. Watts, Susan Athey, Filiz Garip, Thomas L. Griffiths,
Jon Kleinberg, Helen Margetts, Sendhil Mullainathan, Matthew J. Salganik, Simine
Vazire, Alessandro Vespignani, and Tal Yarkoni. 2021. Integrating explanation and
prediction in computational social science. 181–188 pages. doi:10.1038/s41586-
021-03659-0
[12] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu
Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al .2024. Qwen2. 5-coder technical
report.
[13] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. https://aclanthology.
org/2021.eacl-main.74/
[14] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. 7029–7043 pages.
[15] Zhaolu Kang, Junhao Gong, Wenqing Hu, Shuo Yin, Kehan Jiang, Zhicheng Fang,
Yingjie He, Chunlei Meng, Rong Fu, Dongyang Chen, et al .2026. QuantEval: A
Benchmark for Financial Quantitative Tasks in Large Language Models.arXiv
preprint arXiv:2601.08689(2026).
[16] Abhinav Ramesh Kashyap, Yajing Yang, and Min-Yen Kan. 2023. Scientific doc-
ument processing: challenges for modern learning methods. 283–309 pages.
doi:10.1007/s00799-023-00352-7
[17] David Lazer, Alex Pentland, Lada Adamic, Sinan Aral, Albert-László Barabási,
Devon Brewer, Nicholas Christakis, Noshir Contractor, James Fowler, Myron Gut-
mann, Tony Jebara, Gary King, Michael Macy, Deb Roy, and Marshall Van Alstyne.
2009. Computational Social Science. 721–723 pages. doi:10.1126/science.1167742
[18] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval
for weakly supervised open domain question answering. 6086–6096 pages.
https://aclanthology.org/P19-1612/
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, et al .2020. Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. https://arxiv.org/abs/2005.11401
[20] Yuying Li, Siyi Qian, Hao Liang, Leqi Zheng, Ruichuan An, Yongzhen Guo,
and Wentao Zhang. 2025. CapGeo: A Caption-Assisted Approach to Geometric
Reasoning.arXiv preprint arXiv:2510.09302(2025).
[21] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song,
et al.2022. Teaching language models to support answers with verified quotes.
[22] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, et al .2021.
WebGPT: Browser-assisted question-answering with human feedback.
[23] Hongjin Qian, Yutao Zhu, Zhicheng Dou, Haoqi Gu, Xinyu Zhang, et al .2023.
WebBrain: Learning to generate factually correct articles for queries by grounding
on large web corpus.
[24] Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao Liang, Kunlun Zhu, Yankai Lin,
Xu Han, Ning Ding, Huadong Wang, et al .2023. WebCPM: Interactive WebSearch for Chinese Long-form Question Answering.
[25] Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins,
Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reit-
ter. 2023. Measuring attribution in natural language generation models. 777–
840 pages.
[26] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings
using Siamese BERT-networks. https://aclanthology.org/D19-1410/
[27] Tarek Saier, Johan Krause, and Michael Färber. 2023. unarXive 2022: All arXiv
Publications Pre-Processed for NLP, Including Structured Full-Text and Citation
Network. In2023 ACM/IEEE Joint Conference on Digital Libraries (JCDL). IEEE
Computer Society, 66–70. doi:10.1109/JCDL57899.2023.00020
[28] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper,
and Bryan Catanzaro. 2020. Megatron-LM: Training Multi-Billion Parameter
Language Models Using Model Parallelism. arXiv:1909.08053 [cs.CL] https:
//arxiv.org/abs/1909.08053
[29] Qwen Team. 2025. Qwen3-coder: Agentic coding in the world.Blog post(2025).
[30] Simone Teufel, Jean Carletta, and Marc Moens. 1999. An annotation scheme
for discourse-level argumentation in research articles. 110–117 pages. https:
//aclanthology.org/E99-1015/
[31] Deepa Tilwani, Yash Saxena, Ali Mohammadi, Edward Raff, Amit Sheth, Srini-
vasan Parthasarathy, and Manas Gaur. 2024. REASONS: A benchmark for RE-
trieval and Automated citationS Of scieNtific Sentences using Public and Propri-
etary LLMs.arXiv preprint arXiv:2405.02228(2024).
[32] SM Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha,
and Amitava Das. 2024. A comprehensive survey of hallucination mitigation
techniques in large language models.
[33] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. 10014–10037 pages.
[34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report.
[35] Haopeng Zhang, Xiao Liu, and Jiawei Zhang. 2023. Contrastive Hierarchical
Discourse Graph for Scientific Document Summarization (CHANGES). https:
//aclanthology.org/2023.codi-1.4/
[36] Hang Zhang, Chaokun Wang, Hongwei Li, Cheng Wu, Songyao Wang, Yabin
Liu, Gengyuan Shi, and Ziyang Liu. 2025. PLForge: Enhancing Language Models
for Natural Language to Procedural Extensions of SQL.Proc. ACM Manag. Data
3, 6, Article 348 (Dec. 2025), 28 pages. doi:10.1145/3769813
[37] Jiajun Zhang, Jianke Zhang, Zeyu Cui, Jiaxi Yang, Lei Zhang, Binyuan Hui, Qiang
Liu, Zilei Wang, Liang Wang, and Junyang Lin. 2025. PlotCraft: Pushing the
Limits of LLMs for Complex and Interactive Data Visualization.arXiv preprint
arXiv:2511.00010(2025).
[38] Lei Zhang, Mouxiang Chen, Ruisheng Cao, Jiawei Chen, Fan Zhou, Yiheng Xu,
Jiaxi Yang, Liang Chen, Changwei Luo, Kai Zhang, et al .2026. MegaFlow: Large-
Scale Distributed Orchestration System for the Agentic Era.arXiv preprint
arXiv:2601.07526(2026).
[39] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al .2025. Qwen3 Embedding:
Advancing Text Embedding and Reranking Through Foundation Models.
[40] Leqi Zheng, Chaokun Wang, Canzhi Chen, Jiajun Zhang, Cheng Wu, Zixin
Song, Shannan Yan, Ziyang Liu, and Hongwei Li. 2025. LAGCL4Rec: When
LLMs Activate Interactions Potential in Graph Contrastive Learning for Recom-
mendation. InFindings of the Association for Computational Linguistics: EMNLP
2025. Association for Computational Linguistics, Suzhou, China, 1163–1184.
doi:10.18653/v1/2025.findings-emnlp.61
[41] Leqi Zheng, Chaokun Wang, Ziyang Liu, Canzhi Chen, Cheng Wu, and Hongwei
Li. 2025. Balancing Self-Presentation and Self-Hiding for Exposure-Aware Rec-
ommendation Based on Graph Contrastive Learning. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 2027–2037.
[42] Leqi Zheng, Chaokun Wang, Zixin Song, Cheng Wu, Shannan Yan, Jiajun Zhang,
and Ziyang Liu. 2025. Negative Feedback Really Matters: Signed Dual-Channel
Graph Contrastive Learning Framework for Recommendation. InThe Thirty-ninth
Annual Conference on Neural Information Processing Systems.
[43] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui
Lu, and Pengfei Liu. 2025. Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments.
A QA Format
Both citation prediction tasks are formatted as question-answer
pairs following the ChatML format. The System role defines task re-
quirements, specifying input format (paper metadata and retrieved
context) and output format (JSON object with predicted titles and

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Leqi Zheng et al.
Figure 4: Question-answer format specifications for Task 1 and Task 2.
reasoning). The User role provides the query paper’s title, abstract,
and retrieved papers from the corpus, with Task 2 additionally
including section text containing reference placeholders. The Assis-
tant role generates predictions in strict JSON format, producing a
ranked list of reference titles for Task 1 or position-specific citations
for Task 2, accompanied by detailed reasoning to ground predic-
tions in the provided context and prevent hallucination. Figure 4
illustrates the complete ChatML prompt structure and expected
response format for both tasks.
B Detailed Metric Definitions
For the retriever evaluation,Recall@kmeasures the proportion of
relevant papers successfully retrieved within the top-k positions:
Recall@k=|Retrieved@k∩Relevant|
|Relevant|(11)
For the retriever evaluation,Mean Reciprocal Rank@k (MRR@k)
evaluates the ranking quality by considering the position of the
first relevant item:
MRR@k=1
|𝑄||𝑄|∑︁
𝑖=11
rank𝑖(12)
where ranks beyond k contribute zero to the average.
For the task 1 evaluation,Recall@kmeasures the proportion of
ground truth references successfully predicted:
Recall@k=|Predicted@k∩GroundTruth|
|GroundTruth|(13)For the task 1 evaluation,Normalized Discounted Cumulative
Gain@k (NDCG@k)evaluates both relevance and ranking qual-
ity:
NDCG@k=DCG@k
IDCG@k(14)
where
DCG@k=𝑘∑︁
𝑖=12rel𝑖−1
log2(𝑖+1)(15)
For the task 1 evaluation,Hit@kquantifies the absolute number
of correct predictions within the top-k results.
C External Experimental Results
C.1 Citation Diversity and Hallucination
Analysis
Figure 5 reveals the complementary effects of RAG on citation di-
versity and hallucination reduction. RAG integration consistently
increases citation diversity entropy, with CitationGenerator-30B
achieving 3.196 with RAG compared to 2.968 without, indicating
broader subcategory exploration. Closed-source models exhibit
moderate diversity improvements of 0.3-0.5 entropy units while
maintaining balanced distributions. More critically, RAG dramati-
cally reduces hallucination rates, with CitationGenerator-30B de-
creasing from 17.4% to 4.9%, and Gemini-2.5-Pro, GPT-5, and Claude-
4-Sonnet reducing from 22.8%, 23.7%, and 25.9% to 11.5%, 13.8%, and
10.6% respectively.

What Should I Cite? A RAG Benchmark for
Academic Citation Prediction WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
With RAG Without RAG With RAG Without RAG0.00.51.01.52.02.53.03.5Citation Diversity Entropy (CDE)
3.196
2.9682.422
2.3372.153
2.0362.179
2.048
Citation Diversity Entropy (CDE) Hallucination Rate (%)CitationGenerator-30B Gemini-2.5-Pro GPT-5 Claude-4-Sonnet
051015202530
Hallucination Rate (%)4.9%
17.4%11.5%
22.8%13.8%
23.7%10.6%
25.3%
Figure 5: Citation Quality Assessment: Diversity Entropy and Hallu-
cination Rate Across Models.
0 5 10 15 20 25 30 35 40
Top-R-40-35-30-25-20-15-10-50Decrease of NDCG@20(%)
Gemini-2.5-Pro
GPT-5
Claude-4-Sonnet
CitationGenerator-30B
0 5 10 15 20 25 30 35 40
Top-R-50-40-30-20-100Decrease of PACA@20(%)
Gemini-2.5-Pro
GPT-5
Claude-4-Sonnet
CitationGenerator-30B
Figure 6: Retrieval Depth vs. Prediction Quality Trade-off Analysis:
Identifying Optimal Context Length for List-Specific and Position-
Specific Citation Prediction.
C.2 Retrieval Depth vs. Quality Trade-off
Analysis
Figure 6 demonstrates complex relationships between retrieval
depth and prediction quality influenced by context processing ca-
pabilities. Most models exhibit initial improvements from R=5-15,
followed by performance degradation beyond optimal depth as
excessive context introduces noise and exceeds attention span lim-
its. Notably, Gemini-2.5-Pro maintains consistent performance im-
provement across all retrieval depths, attributed to its extended con-
text window capability that effectively processes longer sequences
without quality degradation. CitationGenerator-30B demonstrates
competitive robustness within moderate ranges but shows limita-
tions with extensive context. Considering the trade-offs between in-
ference latency, token consumption costs, and prediction accuracy,
R=10 emerges as the optimal configuration, providing substantial
quality improvements over shallow retrieval while maintaining
computational efficiency and avoiding the noise interference ob-
served at deeper retrieval levels across most model architectures.
C.3 Noise Robustness Analysis
Table 4 evaluates model robustness against retrieval noise by in-
troducing varying proportions of irrelevant papers into retrieved
contexts. Closed-source models including GPT-5, Claude-4-Sonnet,
and Gemini-2.5-Pro demonstrate moderate robustness, maintaining
80-90% of their clean performance at 20% noise levels but declining
to 75-85% at higher noise ratios. CitationGenerator-30B demon-
strates superior robustness due to its fine-tuning, retaining approxi-
mately 80% of clean performance even at 40% noise and maintainingTable 4: Results on noise robustness comparison of different models
across two tasks.
ModelTasks Task1 Task2
Metrics Recall@20 NDCG@20 PACA@20
GPT-5w/o RAG 0.061 0.263 0.069
Noise 0% 0.071 0.324 0.121
Noise 20% 0.066 0.302 0.099
Noise 40% 0.068 0.298 0.102
Noise 80% 0.062 0.270 0.078
Noise 100% 0.055 0.247 0.064
Claude-4-Sonnetw/o RAG 0.038 0.187 0.064
Noise 0% 0.064 0.296 0.101
Noise 20% 0.057 0.257 0.090
Noise 40% 0.051 0.227 0.074
Noise 80% 0.038 0.177 0.073
Noise 100% 0.031 0.151 0.059
Gemini-2.5-Prow/o RAG 0.055 0.272 0.092
Noise 0% 0.076 0.346 0.170
Noise 20% 0.072 0.323 0.123
Noise 40% 0.067 0.304 0.119
Noise 80% 0.059 0.280 0.096
Noise 100% 0.056 0.270 0.090
CitationGenerator-30Bw/o RAG 0.043 0.225 0.165
Noise 0% 0.076 0.367 0.303
Noise 20% 0.056 0.298 0.242
Noise 40% 0.052 0.262 0.210
Noise 80% 0.047 0.240 0.175
Noise 100% 0.045 0.230 0.168
reasonable effectiveness at 80% noise levels where other models
collapse. These results demonstrate that retrieval noise induces
gradual performance degradation rather than complete system fail-
ure, validating the robustness of RAG-based citation prediction in
realistic noisy retrieval environments.
C.4 Use Case Analysis
Figure 7 demonstrates two representative applications correspond-
ing to our benchmark tasks. The literature review generation use
case aligns with Task 1’s coarse-grained objective, where the sys-
tem retrieves a comprehensive list of papers for a research topic
on efficient Transformer architectures, with reasoning explaining
each paper’s relevance. The assisted citation writing use case corre-
sponds to Task 2’s fine-grained objective, where the system recom-
mends specific papers for individual reference placeholders based
on surrounding textual context, successfully identifying relevant
works addressing hallucination and factuality challenges. These
scenarios validate the practical utility of both task formulations for
real-world academic workflows.
C.5 Difficult Case Analysis
Figure 8 illustrates two challenging scenarios that expose current
system limitations. The left panel shows a complex nested clause
case where the citation appears within a deeply embedded subordi-
nate clause, causing models to focus on the main clause about gen-
eralization rather than the nested context discussing momentum-
based updates, resulting in incorrect predictions. The right panel
demonstrates technical detail overload, where multiple scattered
concepts in a lengthy sentence distract models from identifying the
correct transformer mechanism reference, leading to recency bias
toward superficial keyword matches.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Leqi Zheng et al.
Figure 7: From Benchmark to Practice: Illustrative Use Cases of RAG-Enhanced Citation Prediction in Academic Research Workflows.
Figure 8: Difficult Case Analysis in Practice: Syntactic Complexity and Technical Overload as Critical Challenge Patterns.