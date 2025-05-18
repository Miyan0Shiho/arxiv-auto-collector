# OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit

**Authors**: Arun S. Maiya

**Published**: 2025-05-12 15:36:27

**PDF URL**: [http://arxiv.org/pdf/2505.07672v2](http://arxiv.org/pdf/2505.07672v2)

## Abstract
We present OnPrem$.$LLM, a Python-based toolkit for applying large language
models (LLMs) to sensitive, non-public data in offline or restricted
environments. The system is designed for privacy-preserving use cases and
provides prebuilt pipelines for document processing and storage,
retrieval-augmented generation (RAG), information extraction, summarization,
classification, and prompt/output processing with minimal configuration.
OnPrem$.$LLM supports multiple LLM backends -- including llama$.$cpp, Ollama,
vLLM, and Hugging Face Transformers -- with quantized model support, GPU
acceleration, and seamless backend switching. Although designed for fully local
execution, OnPrem$.$LLM also supports integration with a wide range of cloud
LLM providers when permitted, enabling hybrid deployments that balance
performance with data control. A no-code web interface extends accessibility to
non-technical users.

## Full Text


<!-- PDF content starts -->

OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit
OnPrem.LLM: A Privacy-Conscious
Document Intelligence Toolkit
Arun S. Maiya amaiya@ida.org
Institute for Defense Analyses
Alexandria, VA, USA
Abstract
We present OnPrem .LLM, a Python-based toolkit for applying large language models (LLMs)
to sensitive, non-public data in offline or restricted environments. The system is designed for
privacy-preserving use cases and provides prebuilt pipelines for document processing and
storage, retrieval-augmented generation (RAG), information extraction, summarization,
classification, and prompt/output processing with minimal configuration. OnPrem .LLM
supports multiple LLM backends—including llama.cpp, Ollama, vLLM, and Hugging Face
Transformers—with quantized model support, GPU acceleration, and seamless backend
switching. Although designed for fully local execution, OnPrem .LLM also supports inte-
gration with a wide range of cloud LLM providers when permitted, enabling hybrid de-
ployments that balance performance with data control. A no-code web interface extends
accessibility to non-technical users.
Keywords: generative AI, large language models, LLM, NLP, machine learning
1. Introduction
Large Language Models (LLMs) such as GPT-4, LLaMA, Mistral, and Claude have sig-
nificantly advanced the state of natural language processing, achieving strong performance
across tasks including text generation, summarization, question answering, and code synthe-
sis (Anthropic (2024); Jiang et al. (2023); OpenAI et al. (2024); Hugo Touvron et al. (2023)).
While many of these models are accessible through cloud-based APIs, their application to
sensitive, non-public data remains challenging in regulated or restricted environments.
Organizations in domains such as defense, healthcare, finance, and law often operate un-
der strict data privacy and compliance requirements. These constraints frequently prohibit
the use of general-purpose external services, particularly in environments with firewalls,
air-gapped networks, or classified (or otherwise sensitive) workloads.
Common approaches to LLM deployment under these constraints include: (1) running
lightweight, local models with reduced performance; (2) self-hosting larger models at signif-
icant infrastructure cost; or (3) implementing complex, hybrid pipelines to leverage cloud
APIs without violating data governance. Each path incurs trade-offs in accuracy, scalabil-
ity, latency, and maintainability. Prior work on privacy-focused systems ( e.g., PrivateGPT,
LocalGPT, GPT4All) typically lack comprehensive end-to-end pipelines for common tasks,
do not support a diverse range of LLM backends, offer minimal no-code options for non-
1arXiv:2505.07672v2  [cs.CL]  13 May 2025

Maiya
technical users, and are often constrained by a narrow focus on specific retrieval strategies
(Anand et al. (2023); Iv´ an Mart´ ınez (2023); PromptEngineer (2023)).
To address this gap, we introduce OnPrem .LLM, a modular, production-ready toolkit for
applying LLMs to private document workloads in constrained environments. The system
supports both fully local execution and secure integration with privacy-compliant cloud
endpoints, giving organizations flexible control over data locality and model placement. It
includes a suite of prebuilt pipelines for common document intelligence tasks such as ad-
vanced document processing (e.g., table extraction, OCR, markdown conversion), retrieval-
augmented generation (RAG), information extraction, text classification, semantic search,
and summarization.
OnPrem .LLM supports a range of LLM backends—including llama.cpp for efficient
quantized inference, Hugging Face Transformers for broad model compatibility, Ollama
for simplified local model orchestration, and vLLM for high-throughput, GPU-accelerated
inference. In addition, the system provides optional connectors to privacy-compliant cloud
providers ( e.g., AWS GovCloud, Azure Government).1While the toolkit prioritizes privacy,
publicly accessible cloud LLMs (e.g., OpenAI, Anthropic) can also easily be used for appli-
cations involving public or non-sensitive data (e.g., government policy documents, scientific
publications), enabling hybrid deployments that balance performance with data control. A
unified API enables seamless backend switching, while a no-code web interface allows non-
technical users to perform complex document analysis without programming. Key design
principles include:
•Data control – Local processing by default; external access is opt-in and configurable
•Deployment flexibility – Operates on consumer-grade machines or GPU-enabled in-
frastructure
•Ease of integration – Python API, point-and-click web interface, and prebuilt work-
flows streamline setup and execution
•Real-world focus – Built-in pipelines solve practical document-centric tasks out of the
box
OnPrem .LLMis open-source, free to use under a permissive Apache license, and available
on GitHub at: https://github.com/amaiya/onprem . The toolkit has been applied
to a wide range of use cases in the public sector including horizon scanning of scientific
and engineering research, analyses of government policy, qualitative survey analyses, and
resume parsing for talent acquisition. Next, we discuss the core modules in the package.
2. Core Modules
OnPrem .LLMis organized into four primary modules that together provide a comprehensive
framework for document intelligence:
1.OnPrem .LLM was originally named to reflect its exclusive focus on private, local LLMs. Support for
cloud-based models was added later to enable hybrid local/cloud deployments and to support privacy-
compliant cloud endpoints.
2

OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit
2.1 LLM Module
The core engine for interfacing with large language models. It provides a unified API
for working with various LLM backends including llama.cpp, Hugging Face Transformers,
Ollama, vLLM, and a wide-range of cloud providers (Anthropic (2024); Georgi Gerganov
et al. (2023); Ollama Contributors (2025); Kwon et al. (2023); OpenAI et al. (2024); Wolf
et al. (2020)). This module abstracts the complexity of different model implementations
through a consistent interface while handling critical operations such as model loading with
inflight quantization support, easy accessibility to LLMs served through APIs, agentic-like
RAG, and structured LLM outputs.
2.2 Ingest Module
A comprehensive document processing pipeline that transforms raw documents into retriev-
able knowledge. It supports multiple document formats with specialized loaders, automated
OCR for image-based text, and extraction of tables from PDFs. The module offers three
distinct vector storage approaches:
1.Dense Store : Implements semantic search using sentence transformer embeddings
and ChromaDB for similarity-based retrieval using HNSW indexes (Huber et al.
(2025)).
2.Sparse Store : Provides both on-the-fly semantic search and traditional keyword
search through Whoosh2with stemming analyzers and a rich field schema
3.Dual Store : Combines both approaches by maintaining parallel stores, enabling hy-
brid retrieval that leverages both semantic similarity and keyword (or field) matching
2.3 Pipelines Module
Pre-built workflows for common document intelligence tasks with specialized submodules:
•Extractor : Applies prompts to document units (sentences/paragraphs/passages) to
extract structured information with Pydantic model validation (Colvin et al. (2025))
•Summarizer : Provides document summarization with multiple strategies including
map-reduce for large documents and concept-focused summarization3
•Classifier : Implements text classification through scikit-learn wrappers ( SKClassifier ),
Hugging Face transformers ( HFClassifier ), and few-shot learning with limited ex-
amples ( FewShotClassifier ) (Pedregosa et al. (2011); Tunstall et al. (2022); Wolf
et al. (2020))
•Guider : Constrains LLM outputs to enforce structural consistency in generated text
2.4 App Module
As shown in Figure 1, a Streamlit-based web application makes the system accessible to non-
technical users through five specialized interfaces (Streamlit Contributors (2025)). The web
2. We use whoosh-reloaded (available at https://github.com/Sygil-Dev/whoosh-reloaded ), a
fork and continuation of the original Whoosh project.
3.Concept-focused summarization is a technique available in OnPrem .LLM to summarize documents with
respect to a user-specified concept of interest.
3

Maiya
interfaces offer easy, point-and-click access to: 1) interactive chat with conversation history,
2) document-based question answering with source attribution to mitigate hallucinations, 3)
custom prompt application to individual document passages with Excel export capabilities,
4) keyword and semantic search with filtering, pagination, and result highlighting, and 5)
an administrative interface for document ingestion, folder management, and application
configuration. For more information, refer to the web UI documentation.4
3. Usage Examples
The package documentation provides numerous practical examples and applications.5To
illustrate the software’s ease of use, Example 1 presents a concise example of retrieval-
augmented generation (RAG). We download and ingest the House Report on the 2024
National Defense Authorization Act (NDAA) to answer a related policy question.
Example 1: A Basic RAG Pipeline in OnPrem .LLM
from onprem import LLM,utils
# STEP 1 : download the 2024 NDAA report
url ='https ://www. congress . gov /118/ crpt / hrpt125 /CRPT −118hrpt125 . pdf '
utils .download (url,'/tmp/ndaa2024/ report . pdf ')
# STEP 2 : load the d e f a u l t LLM using the d e f a u l t LLM backend
llm =LLM(n_gpu_layers =−1)
# STEP 3 : i n g e s t documents into d e f a u l t vector s t o r e
llm.ingest ('/tmp/ndaa2024 ')
# STEP 4 : ask questions
result =llm.ask('What i s said about hypersonics ? ')
p r i n t ( f”ANSWER: \n{r e s u l t [ 'answer ']}” )
ANSWER :
The context provided highlights the importance of expanding programs related to
hypersonic technology .The House Committee on Armed Services has directed the
Secretary of Defense to . . . ( answer truncated due to space constraints )
4. Conclusion
OnPrem .LLM addresses the critical need for privacy-preserving document intelligence in re-
stricted environments. By combining local large language model (LLM) inference, along
with cloud-based LLM options, and a modular architecture with prebuilt pipelines, the
toolkit enables organizations to implement advanced NLP workflows without compromis-
ing data governance. Its design allows for flexible control over the system footprint, mak-
ing it suitable for a variety of application environments. As LLMs continue to improve
in capability and efficiency, the demand for frameworks that support privacy-conscious,
resource-aware use will only grow.
4.https://amaiya.github.io/onprem/webapp.html
5.https://amaiya.github.io/onprem/
4

OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit
Figure 1: The OnPrem .LLM Web UI.
References
Yuvanesh Anand, Zach Nussbaum, Adam Treat, Aaron Miller, Richard Guo, Ben Schmidt,
GPT4All Community, Brandon Duderstadt, and Andriy Mulyar. Gpt4all: An ecosystem
of open source compressed language models. 2023. URL https://arxiv.org/abs/
2311.04931 .
Anthropic. The claude 3 model family: Opus, sonnet, haiku. https://www.anthropic.
com/claude-3-model-card , 2024. Accessed: 2025-05-07.
Samuel Colvin, Eric Jolibois, Hasan Ramezani, Adrian Garcia Badaracco, Terrence Dorsey,
David Montague, Serge Matveenko, Marcelo Trylesinski, Sydney Runkle, David He-
witt, Alex Hall, and Victorien Plot. Pydantic. https://github.com/pydantic/
pydantic , April 2025. URL https://docs.pydantic.dev/latest/ . MIT Li-
cense. Accessed: 2025-05-07.
Georgi Gerganov et al. llama.cpp: Llm inference in c/c++. https://github.com/
ggml-org/llama.cpp , 2023. Accessed: 2025-05-07.
Jeff Huber, Anton Troynikov, and Chroma Contributors. Chroma: The ai-native open-
source embedding database. https://github.com/chroma-core/chroma , 2025.
Version 1.0.8. Accessed: 2025-05-07.
5

Maiya
Hugo Touvron et al. Llama: Open and efficient foundation language models. 2023. URL
https://arxiv.org/abs/2302.13971 .
Iv´ an Mart´ ınez. Privategpt. May 2023. URL https://github.com/zylon-ai/
private-gpt . Software available at https://www.zylon.ai/ .
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh
Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lu-
cile Saulnier, L´ elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timoth´ ee Lacroix, and William El Sayed. Mistral 7b.
2023. URL https://arxiv.org/abs/2310.06825 .
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for
large language model serving with pagedattention. In Proceedings of the ACM SIGOPS
29th Symposium on Operating Systems Principles , 2023. URL https://github.com/
vllm-project/vllm .
Ollama Contributors. Ollama. https://github.com/ollama/ollama , 2025. Ac-
cessed: 2025-05-07.
OpenAI et al. Gpt-4 technical report. 2024. URL https://arxiv.org/abs/2303.
08774 .
Fabian Pedregosa, Ga¨ el Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand
Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent
Dubourg, et al. Scikit-learn: Machine learning in python. Journal of machine learn-
ing research , 12(Oct):2825–2830, 2011.
PromptEngineer. localgpt. https://github.com/PromtEngineer/localGPT , 2023.
Accessed: 2025-05-07.
Streamlit Contributors. Streamlit: A faster way to build and share data apps. https:
//github.com/streamlit/streamlit , 2025. Accessed: 2025-05-07.
Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe
Wasserblat, and Oren Pereg. Efficient few-shot learning without prompts. 2022. URL
https://arxiv.org/abs/2209.11055 .
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, An-
thony Moi, Pierric Cistac, Tim Rault, R´ emi Louf, Morgan Funtowicz, Joe Davison,
Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu,
Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M.
Rush. Huggingface’s transformers: State-of-the-art natural language processing. 2020.
URL https://arxiv.org/abs/1910.03771 .
6