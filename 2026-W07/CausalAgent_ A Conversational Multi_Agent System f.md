# CausalAgent: A Conversational Multi-Agent System for End-to-End Causal Inference

**Authors**: Jiawei Zhu, Wei Chen, Ruichu Cai

**Published**: 2026-02-12 03:36:29

**PDF URL**: [https://arxiv.org/pdf/2602.11527v1](https://arxiv.org/pdf/2602.11527v1)

## Abstract
Causal inference holds immense value in fields such as healthcare, economics, and social sciences. However, traditional causal analysis workflows impose significant technical barriers, requiring researchers to possess dual backgrounds in statistics and computer science, while manually selecting algorithms, handling data quality issues, and interpreting complex results. To address these challenges, we propose CausalAgent, a conversational multi-agent system for end-to-end causal inference. The system innovatively integrates Multi-Agent Systems (MAS), Retrieval-Augmented Generation (RAG), and the Model Context Protocol (MCP) to achieve automation from data cleaning and causal structure learning to bias correction and report generation through natural language interaction. Users need only upload a dataset and pose questions in natural language to receive a rigorous, interactive analysis report. As a novel user-centered human-AI collaboration paradigm, CausalAgent explicitly models the analysis workflow. By leveraging interactive visualizations, it significantly lowers the barrier to entry for causal analysis while ensuring the rigor and interpretability of the process.

## Full Text


<!-- PDF content starts -->

CausalAgent: A Conversational Multi-Agent System for
End-to-End Causal Inference
Jiawei Zhu∗
3123005571@mails.gdut.edu.cn
Guangdong University of Technology
Guangzhou, Guangdong, ChinaWei Chen∗†
chenweidelight@gmail.com
Guangdong University of Technology
Guangzhou, Guangdong, ChinaRuichu Cai
cairuichu@gmail.com
Guangdong University of Technology
Guangzhou, Guangdong, China
Peng Cheng Laboratory
Shenzhen, China
Figure 1: Overview of the CausalAgent Interface and Workflow.
Abstract
Causal inference holds immense value in fields such as health-
care, economics, and social sciences. However, traditional causal
analysis workflows impose significant technical barriers, requir-
ing researchers to possess dual backgrounds in statistics and com-
puter science, while manually selecting algorithms, handling data
quality issues, and interpreting complex results. To address these
challenges, we propose CausalAgent, a conversational multi-agent
system for end-to-end causal inference. The system innovatively
∗Both authors contributed equally to this research.
†Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
IUI Companion ’26, Paphos, Cyprus
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1985-1/2026/03
https://doi.org/10.1145/3742414.3794777integrates Multi-Agent Systems (MAS), Retrieval-Augmented Gen-
eration (RAG), and the Model Context Protocol (MCP) to achieve
automation from data cleaning and causal structure learning to
bias correction and report generation through natural language
interaction. Users need only upload a dataset and pose questions
in natural language to receive a rigorous, interactive analysis re-
port. As a novel user-centered human-AI collaboration paradigm,
CausalAgent explicitly models the analysis workflow. By leveraging
interactive visualizations, it significantly lowers the barrier to entry
for causal analysis while ensuring the rigor and interpretability of
the process.
CCS Concepts
•Human-centered computing →Natural language interfaces;
•Computing methodologies →Multi-agent systems;Causal rea-
soning and diagnostics.arXiv:2602.11527v1  [cs.AI]  12 Feb 2026

IUI Companion ’26, March 23–26, 2026, Paphos, Cyprus Zhu, Chen and Cai, et al.
Keywords
Multi-Agent System, Causal Inference, Large Language Models,
RAG, Model Context Protocol, Interactive Data Analysis
ACM Reference Format:
Jiawei Zhu, Wei Chen, and Ruichu Cai. 2026. CausalAgent: A Conversa-
tional Multi-Agent System for End-to-End Causal Inference. InCompanion
Proceedings of the 31st International Conference on Intelligent User Interfaces
(IUI Companion ’26), March 23–26, 2026, Paphos, Cyprus.ACM, New York,
NY, USA, 4 pages. https://doi.org/10.1145/3742414.3794777
1 Introduction
Causal inference is crucial for decision-making in many fields like
healthcare [ 12], economics [ 8], Neuroscience [ 4], and financial anal-
ysis [ 6]. However, traditional workflows impose high barriers, de-
manding researchers master complex algorithms and manually ad-
dress engineering challenges [ 5,15]. While Large Language Models
(LLMs) offer potential for causal reasoning, single models struggle
with hallucination in long-process tasks [ 7,9]. Moreover,existing
tools (e.g., DoWhy [ 14], Causal-learn [ 18]) often lack intuitive, end-
to-end automation.
To address these challenges, we introduceCausalAgent1, a
system encapsulating the causal analysis process into collaborative
agent services [ 17]. Users interact via natural language to complete
a closed loop: from data processing and RAG-based knowledge
retrieval [ 11] to causal structure learning and report generation.
By explicitly modeling the workflow and employing Supervised
Fine-Tuning (SFT), CausalAgent mitigates hallucination risks and
ensures rigorous, reproducible analysis.
In this demonstration, we showcase how CausalAgent guides
users through the entire causal discovery pipeline—from raw data
processing to interactive, multi-round causal reasoning—via a uni-
fied conversational interface.
2 System Overview
2.1 Architecture
The core of CausalAgent lies in its highly scalable Multi-Agent
System (MAS). As illustrated in Figure 1, the system follows a
modular design utilizing LangGraph [ 10] for intelligent routing.
The architecture is composed of three primary components:Data
Processing Agent,Causal Structure Learning Agent, andRe-
porting Agent. Each agent focuses solely on its own logic and
I/O constraints, preserving context information within the shared
state to ensure consistency across different analysis stages [ 16]. The
specific details of these components are provided in the following
subsections.
2.2 Data Processing Agent
This agent handles both preprocessing and quality detection.Pre-
processing:It validates uploaded files, summarizes statistics (e.g.,
missing rates, unique values), and infers the meaning from variables’
names. The system provides a “causal analysis friendliness" rating
and generates visualizations (histograms, correlation heatmaps) to
highlight data quality issues.Quality Detection:It utilizes graph
algorithms to detect cycles violating the Directed Acyclic Graph
1Source code: https://github.com/DMIRLAB-Group/CausalAgent(DAG) assumption [ 12]. If cycles are found, the agent proposes edge
modifications based on domain priors and statistical associations,
requiring the LLM to provide confidence-weighted decisions.
2.3 Causal Structure Learning Agent
This agent acts as an algorithm scheduler based on natural lan-
guage prompts in the state and knowledge priors, utilizing the
Model Context Protocol (MCP) [ 2]. It invokes causal learning algo-
rithms from the tool server to construct causal graphs. Currently,
the algorithm library integrates the PC algorithm2[15] for search-
ing DAG structures based on conditional independence tests, and
OLC-based algorithms3[3,5] for handling latent confounders. The
agent selects the most appropriate algorithm based on natural lan-
guage instructions and data characteristics, outputting a node list
and adjacency matrix. The causal graph structure generated in
this phase serves as the core input for post-processing and report
generation modules, and is persisted as a checkpoint to facilitate
subsequent multi-turn inquiries.
2.4 Reporting Agent
By integrating RAG [ 11], this agent synthesizes diagnostic metrics
and causal structures into comprehensive reports. These reports
comprehensively cover methodological details, key causal findings,
and model limitations. Furthermore, the system exports the graph
to an interactive frontend component, allowing users to explore
complex dependencies through visual operations like node manip-
ulation and zooming.
3 Key Implementation
To ensure robustness, we employ the following four strategies:
(1)Retrieval-Augmented Generation (RAG).We constructed
a vectorized knowledge base from causal textbooks to ground
LLM explanations in established theory.
(2)Supervised Fine-Tuning (SFT).The base model was fine-
tuned to enhance instruction-following and accuracy in trans-
lating statistical results into natural language.
(3)Prompt Engineering & MCP.Adopting MCP [ 2] standard-
izes the interface between LLMs and external tools, decoupling
reasoning from execution.
(4)LLM Backbone.We utilize GLM-4.6 [ 1], leveraging its reason-
ing capabilities for task distribution and conflict resolution.
4 Demonstration Scenario
To demonstrate the efficacy and interactivity of CausalAgent in
scientific discovery, we present a case study using the widely-cited
Sachs Protein Signaling Dataset [13]. This dataset consists of flow
cytometry measurements of 11 phosphorylated proteins and phos-
pholipids (e.g., Raf, Mek, Erk) derived from human immune system
cells, serving as a gold standard for validating causal structure
learning algorithms.
4.1 Workflow Walkthrough
As shown in Figure 2, the analysis workflow proceeds as follows:
2Source code: https://github.com/py-why/causal-learn
3Source code: https://github.com/DMIRLAB-Group/CDMIR

CausalAgent: A Conversational Multi-Agent System for End-to-End Causal Inference IUI Companion ’26, March 23–26, 2026, Paphos, Cyprus
Figure 2: Demonstration of CausalAgent on the Sachs Dataset. (A) CausalAgent Running process. (B) Correlation heatmap of
variables. (C) Generated Causal diagram.
Step 1: Data Upload and Query.The user uploads thesachs.csv
file and inputs a high-level directive:“Please conduct a causal anal-
ysis on the file ‘sachs_dataset.csv’. "
Step 2: Multi-Agent Collaboration.CausalAgent automati-
cally profiles the dataset. By analyzing data distribution characteris-
tics within this biological context (dense network), it autonomously
selects the PC algorithm for pathway reconstruction—without re-
quiring the user to interpret complex raw adjacency matrices.
Step 3: Report Generation.Following Processes B and C de-
picted in Figure 2, CausalAgent synthesizes algorithmic outputs
with its built-in knowledge base to generate a comprehensive anal-
ysis report. The report identifiesAktandPkaas master regulators,
explicitly recommending them as priority targets for drug interven-
tion studies.
Step 4: Interactive Refinement.Following the report, the user
asks a follow-up question:“What if we intervene on Mek? How would
Erk change?". The system responds by synthesizing thederived
causal analysis resultswithdomain priors.
5 Conclusion and Future Work
We presented CausalAgent, a conversational multi-agent system
for end-to-end causal inference. The system integrates MAS, RAG,
and multiple causal algorithms encapsulated by the MCP protocol.
While ensuring high extensibility, it provides users with a conve-
nient interactive experience, significantly reducing the barrier to
the causal domain and providing a better platform for expanding
causal inference development.Future work will focus on two directions: first, at the algorithmic
level, integrating structure learning methods supporting Latent
Confounders and counterfactual reasoning modules to achieve
quantitative causal effect estimation; second, regarding domain
adaptation, introducing expert feedback mechanisms for high-risk
scenarios like healthcare and finance to further enhance system
safety and reliability.
Acknowledgments
This research was supported in part by the National Science and
Technology Major Project (2021ZD0111502), the Natural Science
Foundation of China (U24A20233, 62206064), the National Science
Fund for Excellent Young Scholars (62122022), the Guangdong Ba-
sic and Applied Basic Research Foundation (2025A1515010172),
and the Guangzhou Basic and Applied Basic Research Foundation
(2024A04J4384).
GenAI Usage Disclosure
Generative AI tools (Gemini, Gemini-3.0-pro) were used to improve
the readability of the manuscript and assist in prototyping the agent
interaction logic. The entire content, including code and text, was
manually verified by the authors to ensure scientific rigor.
References
[1]Zhipu AI. 2025. GLM-4.6. https://docs.bigmodel.cn/cn/guide/models/text/glm-4.6.
Acessed:2025-11-27.
[2]Anthropic. 2024. Model Context Protocol: A Standard for Connecting AI Models
to Data. https://www.anthropic.com/news/model-context-protocol. Accessed:

IUI Companion ’26, March 23–26, 2026, Paphos, Cyprus Zhu, Chen and Cai, et al.
2025-11-27.
[3]Ruichu Cai, Zhiyi Huang, Wei Chen, Zhifeng Hao, and Kun Zhang. 2023. Causal
discovery with latent confounders based on higher-order cumulants. InInterna-
tional conference on machine learning. PMLR, 3380–3407.
[4]Ruichu Cai, Yunjin Wu, Xiaokai Huang, Wei Chen, Tom ZJ Fu, and Zhifeng Hao.
2024. Granger causal representation learning for groups of time series.Science
China Information Sciences67, 5 (2024), 152103.
[5]Wei Chen, Zhiyi Huang, Ruichu Cai, Zhifeng Hao, and Kun Zhang. 2024. Identifi-
cation of causal structure with latent variables based on higher order cumulants.
InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 20353–
20361.
[6]Wei Chen, Linjun Peng, Zhiyi Huang, Ruichu Cai, Zhifeng Hao, and Kun Zhang.
2025. Higher Order Cumulants-Based Method for Direct and Efficient Causal
Discovery.IEEE Transactions on Neural Networks and Learning Systems(2025).
[7]Wei Chen, Jiahao Zhang, Haipeng Zhu, Boyan Xu, Zhifeng Hao, Keli Zhang,
Junjian Ye, and Ruichu Cai. 2025. Causal-aware Large Language Models: Enhanc-
ing Decision-Making Through Learning, Adapting and Acting.arXiv preprint
arXiv:2505.24710(2025).
[8]Guido W Imbens and Donald B Rubin. 2015.Causal Inference for Statistics,
Social, and Biomedical Sciences. Cambridge University Press. doi:10.1017/
CBO9781139025751
[9]Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in
natural language generation.Comput. Surveys55, 12 (2023), 1–38. doi:10.1145/
3571730
[10] LangChain. 2024. LangGraph: Building Language Agents as Graphs. https:
//github.com/langchain-ai/langgraph. Accessed: 2025-11-27.[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in Neural Information Processing Systems33 (2020), 9459–9474.
[12] Judea Pearl. 2009.Causality. Cambridge university press. doi:10.1017/
CBO9780511803161
[13] Karen Sachs, Omar Perez, Dana Pe’er, Douglas A Lauffenburger, and Garry P
Nolan. 2005. Causal protein-signaling networks derived from multiparameter
single-cell data.Science308, 5721 (2005), 523–529. doi:10.1126/science.1105809
[14] Amit Sharma and Emre Kiciman. 2020. DoWhy: An End-to-End Library for Causal
Inference.arXiv preprint arXiv:2011.04216(2020). doi:10.48550/arXiv.2011.04216
[15] Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. 2000.
Causality, prediction, and search. MIT press. doi:10.7551/mitpress/1754.001.0001
[16] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li
Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al .2023. Autogen: Enabling
next-gen llm applications.arXiv preprint arXiv:2308.08155(2023). doi:10.48550/
arXiv.2308.08155
[17] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yi Ding, Boyang Hong, Ming
Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al .2023. The rise and potential
of large language model based agents: A survey.arXiv preprint arXiv:2309.07864
(2023). doi:10.48550/arXiv.2309.07864
[18] Yujia Zheng, Biwei Huang, Wei Chen, Joseph Ramsey, Mingming Gong, Ruichu
Cai, Shohei Shimizu, Peter Spirtes, and Kun Zhang. 2024. Causal-learn: Causal
Discovery in Python.Journal of Machine Learning Research25, 60 (2024), 1–8.
http://jmlr.org/papers/v25/23-0970.html