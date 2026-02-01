# "ENERGY STAR" LLM-Enabled Software Engineering Tools

**Authors**: Himon Thakur, Armin Moin

**Published**: 2026-01-27 06:40:57

**PDF URL**: [https://arxiv.org/pdf/2601.19260v1](https://arxiv.org/pdf/2601.19260v1)

## Abstract
The discussion around AI-Engineering, that is, Software Engineering (SE) for AI-enabled Systems, cannot ignore a crucial class of software systems that are increasingly becoming AI-enhanced: Those used to enable or support the SE process, such as Computer-Aided SE (CASE) tools and Integrated Development Environments (IDEs). In this paper, we study the energy efficiency of these systems. As AI becomes seamlessly available in these tools and, in many cases, is active by default, we are entering a new era with significant implications for energy consumption patterns throughout the Software Development Lifecycle (SDLC). We focus on advanced Machine Learning (ML) capabilities provided by Large Language Models (LLMs). Our proposed approach combines Retrieval-Augmented Generation (RAG) with Prompt Engineering Techniques (PETs) to enhance both the quality and energy efficiency of LLM-based code generation. We present a comprehensive framework that measures real-time energy consumption and inference time across diverse model architectures ranging from 125M to 7B parameters, including GPT-2, CodeLlama, Qwen 2.5, and DeepSeek Coder. These LLMs, chosen for practical reasons, are sufficient to validate the core ideas and provide a proof of concept for more in-depth future analysis.

## Full Text


<!-- PDF content starts -->

‘ENERGY STAR’ LLM-Enabled Software Engineering Tools
Himon Thakur
hthakur@uccs.edu
Department of Computer Science
University of Colorado Colorado Springs (UCCS)
United StatesArmin Moin
amoin@uccs.edu
Department of Computer Science
University of Colorado Colorado Springs (UCCS)
United States
Abstract
The discussion around AI-Engineering, that is, Software Engineer-
ing (SE) for AI-enabled Systems, cannot ignore a crucial class of soft-
ware systems that are increasingly becoming AI-enhanced: Those
used to enable or support the SE process, such as Computer-Aided
SE (CASE) tools and Integrated Development Environments (IDEs).
In this paper, we study the energy efficiency of these systems. As AI
becomes seamlessly available in these tools and, in many cases, is
active by default, we are entering a new era with significant impli-
cations for energy consumption patterns throughout the Software
Development Lifecycle (SDLC). We focus on advanced Machine
Learning (ML) capabilities provided by Large Language Models
(LLMs). Our proposed approach combines Retrieval-Augmented
Generation (RAG) with Prompt Engineering Techniques (PETs) to
enhance both the quality and energy efficiency of LLM-based code
generation. We present a comprehensive framework that measures
real-time energy consumption and inference time across diverse
model architectures ranging from 125M to 7B parameters, including
GPT-2, CodeLlama, Qwen 2.5, and DeepSeek Coder. These LLMs,
chosen for practical reasons, are sufficient to validate the core ideas
and provide a proof of concept for more in-depth future analysis.
Keywords
case tools, ide, ai, energy efficiency, llm
ACM Reference Format:
Himon Thakur and Armin Moin. 2026. ‘ENERGY STAR’ LLM-Enabled Soft-
ware Engineering Tools. In.ACM, New York, NY, USA, 2 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
A vital class of AI-enabled software systems includes tools sup-
porting the Software Engineering (SE) process, such as Computer-
Aided SE (CASE) tools and Integrated Development Environments
(IDEs). Modern IDEs, such as the Jupyter Notebook, have already
integrated advanced AI capabilities. Consequently, AI has become
considerably ubiquitous and accessible throughout the software
development process. What has remained largely understudied
is the energy-efficiency aspect of the new AI-enabled trends in
the modern era of SE. In this paper, we focus on Large Language
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXXModels (LLMs) used for code generation, for example, automated
suggestions of code snippets, in modern IDEs. We propose a novel
approach based on Retrieval-Augmented Generation (RAG) and
Prompt Engineering Techniques (PETs).
The contribution of this paper is to answer the following research
questions (RQ) through an experimental study:RQ1:Can we reduce
LLMs’ energy consumption or inference time through Retrieval
Augmented Generation (RAG) pipelines?RQ2:How do several
selected LLMs with different architectures compare in terms of
energy consumption and inference latency?RQ3:Can we find a
correlation between the LLM’s model size and any possible energy
efficiency benefits of using RAG across different LLM architectures?
RQ4:Can RAG enable smaller and more resource-efficient LLMs to
achieve code generation performance comparable to larger LLMs
while maintaining low energy usage?
The remainder of this extended abstract is structured as follows.
Section 2 briefly reviews the literature. In Section 3, we propose our
novel approach. Section 4 presents a summary of our experimental
study. Finally, we conclude and suggest the future work in Section 5.
2 Related Work
Rubei et al. [ 5] demonstrated that strategic prompt engineering
significantly reduces energy consumption in code completion tasks
using Llama 3. Their optimal prompt configuration with custom
tags reduced energy usage by 99% for one-shot and 83% for few-shot
prompting compared to baselines, while improving the accuracy
of the exact match by up to 44%. This showed that a thoughtful
prompt design could substantially reduce the environmental impact
of LLM inference without compromising performance. Moreover,
Islam et al. [ 2] showed that LLM-generated code can be up to 450
times less energy-efficient than code written by human developers.
3 Proposed Approach
We propose a novel framework that utilizes RAG pipelines to en-
hance the energy efficiency and/or inference time for LLM-based
code generation or improvement tasks in software development
tools or environments. The framework measures the resulting im-
pact on energy consumption of the LLM and the total inference
time. The proposed framework integrates several components: i)
LLM:Our framework allows selecting from a range of LLMs vary-
ing in size and architecture, including GPT-2 (125M parameters),
CodeLlama (7B parameters), and Qwen 2.5 (7B parameters). There-
fore, we support model portability. The LLM is deployed to support
automated code generation and improvements. ii)RAG Pipeline:
We implement an embedding-based retrieval mechanism using Sen-
tenceTransformers [ 4] to identify and incorporate relevant code
examples based on natural language queries. Our RAG implemen-
tation combines dense vector retrieval with prompt augmentation.arXiv:2601.19260v1  [cs.SE]  27 Jan 2026

Conference’17, July 2017, Washington, DC, USA Thakur and Moin
For each natural language code description, we perform the follow-
ing steps: a)Embedding Generation:We encode the query using
the SentenceTransformer model (specifically, the all-MiniLM-L6-v2
model) to create dense vector representations. b)Similarity Search
and Retrieval:We retrieve the most similar examples from the
training corpus based on cosine similarity between embeddings. We
use the Facebook AI Similarity Search (FAISS) open-source library
for efficient vector search when available, with PyTorch fallback
for compatibility across platforms. c)Prompt Augmentation:For
RAG-enhanced generation, we augment the input prompt with
the retrieved examples. For baseline comparisons, we use only the
original query without examples. d)Example Selection:We dy-
namically select the number of examples to include based on the
model’s context window, ensuring that we do not exceed maximum
token limits. For our main experiments, we typically include 2-3 ex-
amples. iii)External Knowledge Base:The RAG pipeline requires
an external knowledge base for the search and retrieval of relevant
contextual information in order to augment the input prompt as
explained above. This knowledge base must be compatible with
the specific programming languages supported by the SE tool or
environment. iv)Energy Monitoring:We utilize the CodeCarbon
[6] library. Note that we do not rely on the CO2 emission estimates,
as it is practically impossible to know the exact source of energy.
4 Experimental Study
We use the CodeXGLUE dataset [ 3], specifically the CONCODE
text-to-code subset, which contains natural language descriptions
paired with corresponding Java code. We also use Kaggle’s Natu-
ral Language to Python Code dataset for a similar task in Python
[1]. Moreover, the source code is available as open-source soft-
ware at https:// github.com/qas-lab/himon-thakur/tree/main/Rag-
Enhanced-PETs-for-LLM-Energy-Consumption-Optimization. Our
experimental study confirmed the findings reported by Rubei et al.
[5] that well-designed prompts can reduce the LLMs’ energy con-
sumption. In addition, we observed a possibility of a reduction in
the inference time. In the following, we revisit our RQs and answer
them based on the achieved experimental results.RQ1.Based on
the results, the answer is yes, this can happen, although with mixed
results. The findings showed reductions in energy consumption
when using RAG for GPT-2 and CodeLlama. However, DeepSeek
Coder and Qwen indicated increased energy consumption with
RAG. For the inference time, only CodeLlama showed improve-
ments (25% faster with RAG), while other models experienced in-
creased inference times. CodeLlama achieved the most promising
results with both reduced energy consumption and 25% faster infer-
ence time while drastically improving code quality. Other models
showed trade-offs: GPT-2 had a slightly better energy efficiency, but
it was slower in inference; DeepSeek and Qwen consumed more
energy and time, but they could generally produce higher quality
code with RAG.RQ2.The energy consumption levels varied sig-
nificantly across the selected models, with GPT-2 being the most
efficient, followed by CodeLlama, while DeepSeek Coder and Qwen
used approximately 3x more energy. For inference time, Qwen with-
out RAG was the fastest, followed by GPT-2, DeepSeek Coder, and
CodeLlama.RQ3.The answer is no based on our evidence here.
The data showed no clear relationship between the model size andachieving any RAG-based energy efficiency benefits. Of the models
studied, only GPT-2 (i.e., the smallest in size among the choices)
and CodeLlama showed energy reduction with RAG, while the
other models saw increased energy consumption regardless of their
size.RQ4.Based on our outcomes in this study, the answer is yes.
With RAG, GPT-2 on the Kaggle dataset achieved a code quality
score of 0.6, which matched DeepSeek Coder’s performance while
using approximately 3.5x less energy. This demonstrated that RAG
could help smaller, more efficient models achieve competitive code
generation quality.
5 Conclusion and Future Work
In this paper, we have examined how RAG affects energy consump-
tion and inference speed in code generation across different LLM
architectures for SE tools. Our experimental results have shown
that the impacts of RAG pipelines varied across the studied LLMs:
CodeLlama experienced 25% faster inference times and substantial
quality improvements, while smaller models like GPT-2 showed
mixed efficiency results despite modest energy savings. For future
work, we intend to use more powerful servers. In particular, cloud
servers require some deeper understanding of the cloud provider
infrastructure to be aware of any potential unwanted influence on
our measurements. In addition, we will incorporate well-established
code quality metrics, such as CodeBleu, as well as static and dy-
namic analyses, inspections, and testing to better assess the code
quality. We will also explore a combination of RAG and the Model
Context Protocol (MCP). Finally, we plan to explore the energy
efficiency implications for quantum computing, such as Python
code generation for quantum SDKs.
Acknowledgments
This work is in part supported by a grant from the Colorado OEDIT.
In preparing this work, we used generative AI models and tools
(e.g., GPT-5), to assist with generating or revising content and code.
References
[1][n. d.]. Kaggle - Natural Language to Python Code. https://www.kaggle.com/
datasets/linkanjarad/coding-problems-and-solution-python-code Accessed 2025-
10-23.
[2]Md Arman Islam, Devi Varaprasad Jonnala, Ritika Rekhi, Pratik Pokharel, Sid-
dharth Cilamkoti, Asif Imran, Tevfik Kosar, and Bekir Turkkan. 2025. Evaluating
the Energy-Efficiency of the Code Generated by LLMs. arXiv:2505.20324 [cs.SE]
https://arxiv.org/abs/2505.20324
[3]Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio
Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou,
Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan,
Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. 2021. CodeXGLUE:
A Machine Learning Benchmark Dataset for Code Understanding and Generation.
arXiv preprint arXiv:2102.04664(2021). doi:10.48550/arXiv.2102.04664
[4]Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP). Hong Kong, China, 3982–3992.
doi:10.18653/v1/D19-1410
[5]Riccardo Rubei, Aicha Moussaid, Claudio Di Sipio, and Davide Di Ruscio. 2025.
Prompt engineering and its implications on the energy consumption of Large
Language Models. InProceedings of the 2025 IEEE/ACM 9th International Workshop
on Green and Sustainable Software(Ottawa, ON, Canada)(GREENS ’25). IEEE Press,
60–67. doi:10.1109/GREENS66463.2025.00014
[6]Victor Schmidt, Kamal Goyal, Arturo Joshi, Boris Feld, Liam Conell, Nikos Laskaris,
Doug Blank, Jonathan Wilson, Sorelle Friedler, and Sasha Luccioni. 2021. CodeCar-
bon: Estimate and Track Carbon Emissions from Machine Learning Computing.
doi:10.5281/zenodo.4658424