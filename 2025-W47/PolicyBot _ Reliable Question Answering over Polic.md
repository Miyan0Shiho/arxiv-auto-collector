# PolicyBot - Reliable Question Answering over Policy Documents

**Authors**: Gautam Nagarajan, Omir Kumar, Sudarsun Santhiappan

**Published**: 2025-11-17 15:26:10

**PDF URL**: [https://arxiv.org/pdf/2511.13489v1](https://arxiv.org/pdf/2511.13489v1)

## Abstract
All citizens of a country are affected by the laws and policies introduced by their government. These laws and policies serve essential functions for citizens. Such as granting them certain rights or imposing specific obligations. However, these documents are often lengthy, complex, and difficult to navigate, making it challenging for citizens to locate and understand relevant information. This work presents PolicyBot, a retrieval-augmented generation (RAG) system designed to answer user queries over policy documents with a focus on transparency and reproducibility. The system combines domain-specific semantic chunking, multilingual dense embeddings, multi-stage retrieval with reranking, and source-aware generation to provide responses grounded in the original documents. We implemented citation tracing to reduce hallucinations and improve user trust, and evaluated alternative retrieval and generation configurations to identify effective design choices. The end-to-end pipeline is built entirely with open-source tools, enabling easy adaptation to other domains requiring document-grounded question answering. This work highlights design considerations, practical challenges, and lessons learned in deploying trustworthy RAG systems for governance-related contexts.

## Full Text


<!-- PDF content starts -->

POLICYBOT- RELIABLEQUESTIONANSWERING OVERPOLICY
DOCUMENTS
Gautam Nagarajan
School of Engineering
Shiv Nadar University
Noida, India
gautam.nraj@gmail.comOmir Kumar
Centre for Responsible AI
IIT Madras
Chennai, India
omir@cerai.inSudarsun Santhiappan
Centre for Responsible AI
Wadhwani School of Data Science & AI
IIT Madras
Chennai, India
sudarsun@dsai.iitm.ac.in
ABSTRACT
All citizens of a country are affected by the laws and policies introduced by their government. These
laws and policies serve essential functions for citizens. Such as granting them certain rights or
imposing specific obligations. However, these documents are often lengthy, complex, and difficult to
navigate, making it challenging for citizens to locate and understand relevant information. This work
presents PolicyBot , a retrieval-augmented generation (RAG) system designed to answer user queries
over policy documents with a focus on transparency and reproducibility. The system combines domain-
specific semantic chunking, multilingual dense embeddings, multi-stage retrieval with reranking, and
source-aware generation to provide responses grounded in the original documents. We implemented
citation tracing to reduce hallucinations and improve user trust, and evaluated alternative retrieval and
generation configurations to identify effective design choices. The end-to-end pipeline is built entirely
with open-source tools, enabling easy adaptation to other domains requiring document-grounded
question answering. This work highlights design considerations, practical challenges, and lessons
learned in deploying trustworthy RAG systems for governance-related contexts.
KeywordsCitation Tracing ·Hallucination Mitigation ·Policy Documents ·Question Answering ·Retrieval-Augmented
Generation·Semantic Chunking
1 Introduction
Public policy documents form the backbone of governance, defining laws, regulations, entitlements, and procedures
that affect citizens, institutions, and industries. However, these documents are often dense, verbose, lengthy, and
inaccessible to non-experts. Legal jargon, bureaucratic phrasing, and intricate cross-referencing create significant
barriers to understanding, even for professionals familiar with the policy domain. As a result, individuals seeking
specific information to make decisions, verify claims, or understand their rights often rely on secondhand summaries,
expert intermediaries, or extensive manual reading. These approaches can be slow, expensive, error-prone, and prone to
omissions. Therefore, it may be argued that laws and policies are often inaccessible to the very people (citizens) for
whom they are made.
Existing question answering (QA) and retrieval-augmented generation (RAG) systems have shown promise in domains
such as general web search, customer support, and scientific literature. However, they face particular challenges in
the legal and policy document domain. Conventional methods often struggle with the length and complexity of these
documents, resulting in incomplete retrieval, loss of context, or factual errors in the generated responses. Moreover,
many systems lack reliable citation tracing, making it difficult for users to verify information—a critical need in
governance and legal contexts. Hallucination, where systems fabricate content not present in the source, further
undermines trust in automated policy QA tools.
To address these challenges, we presentPolicyBot, a retrieval-augmented generation system designed specifically for
policy document question answering. PolicyBot integrates domain-specific semantic chunking with a multi-stagearXiv:2511.13489v1  [cs.ET]  17 Nov 2025

PolicyBot - Reliable Question Answering over Policy Documents
retrieval pipeline that includes multilingual dense embeddings, RAG-Fusion, Hypothetical Document Embeddings
(HyDE), and reranking via a cross-encoder model. It employs source-aware generation and citation tracing to ensure
factual accuracy and reduce hallucinations. The system runs efficiently on consumer-grade hardware, utilizing the
Gemma 3n1language model viaOllama2, which enables deployment in low-bandwidth or offline settings. It supports
context-aware, multi-turn conversations by preserving relevant query history and maintaining coherence over extended
interactions.
The key contributions of this work are:
•Factual accuracy and traceability:A multi-layered hallucination control strategy combining document-
grounded generation, direct quotations, and explicit source chunk display.
•Advanced retrieval pipeline:Domain-specific semantic chunking and multi-stage retrieval with HyDE,
RAG-Fusion, and reranking to improve relevance and handle ambiguous queries.
•Hardware-efficient local deployment:An open-source, lightweight LLM architecture enabling private,
low-latency inference on consumer-grade hardware.
•Context-aware multi-turn interaction:Support for coherent, multi-turn dialogue grounded in prior conversa-
tion history.
PolicyBot is designed to benefit diverse user groups, including students seeking clarity on laws, policies, and
regulations; journalists fact-checking policy claims; NGOs and citizens verifying their rights; public sector professionals
reviewing operational rules; and researchers examining law and policy. Its offline capability makes it particularly
relevant for under-resourced communities, enhancing equitable access to reliable policy information. The PolicyBot
not only helps improve accessibility to applicable laws and policies for citizens, but also ensures that respective
governments and related institutions can reach a greater number of people.
Paper Outline.The remainder of this paper is organized as follows. Section 2 describes the system architecture and
design choices in detail. Section 3 presents the experimental setup, including datasets, metrics, and tested configurations.
Section 4 discusses deployment considerations and key lessons learned. Section 5 addresses ethical considerations
relevant to policy document QA. Section 6 outlines potential extensions to enhance scalability, multilingual support, and
user-centered evaluation Section 7 concludes by reflecting on how the proposed PolicyBot system advances reliable,
transparent, and accessible question answering over policy documents. Section 8 provides the links to the live demo
and the source code of the proposedPolicyBotsystem.
2 System Design and Architecture
PolicyBot is implemented as a modular retrieval-augmented generation (RAG) pipeline optimized for the charac-
teristics of public policy documents. The architecture comprises seven main components: data ingestion, semantic
chunking, embedding and indexing, retrieval and reranking, generation, citation, and hallucination control, as well as
the user interface. Figure 1 illustrates the overall workflow.
2.1 Data Ingestion
The current implementation only supports PDF files as input. The PyMuPDF3library is used for fast and accurate text
extraction. Each page of the input PDF is treated as a separate document, and its text content is stripped of leading and
trailing whitespace before further processing. For each page, metadata is stored that includes the page number and the
original file name. No additional preprocessing, such as header or footer removal or language detection, is performed,
as the text extraction capabilities ofPyMuPDFare sufficient for the target documents.
2.2 Semantic Chunking
To segment documents into semantically coherent units, PolicyBot utilizes the experimental semantic text splitter
[1] of LangChain4. This splitter segments the text based on logical and semantic boundaries rather than fixed-size
windows. The segmentation process uses standard_deviation as the breakpoint_threshold_type and sets the
value of breakpoint_threshold_amount to1.0, enabling the chunking algorithm to adapt to the natural structure
1https://deepmind.google/models/gemma/gemma-3n/
2https://github.com/ollama/ollama
3https://pymupdf.readthedocs.io/en/latest/
4https://python.langchain.com/docs/introduction/
2

PolicyBot - Reliable Question Answering over Policy Documents
Figure 1: Architecture of thePolicyBotsystem.
of the policy documents. This approach preserves the integrity of legal clauses and section-level semantics, reducing the
likelihood of losing critical contextual dependencies. Empirically, this method consistently outperformed alternatives,
such as fixed-length or recursive chunking, in the domain of policy documents.
2.3 Embedding and Indexing
We use alibaba/gte-multilingual-base [2] to create the rich embeddings of the semantic chunks. We chose the
model for its strong performance across multilingual factual question-answering tasks in the SciFact dataset [ 3] from the
BEIR benchmarking framework [ 4]. While the model supports multiple languages, the current deployment primarily
focuses on policy documents in English.
We use ChromaDB [ 5] to store the vector embeddings locally. ChromaDB is configured with the default HNSW
(Hierarchical Navigable Small World) [ 6] vector index for efficient approximate nearest neighbor (ANN) search. Cosine
similarity is used as the distance metric for retrieval. This combination of dense embeddings and ANN indexing
provides a balance of semantic fidelity and query-time efficiency, making it suitable for interactive use.
3

PolicyBot - Reliable Question Answering over Policy Documents
2.4 Retrieval and Reranking
PolicyBotemploys a multi-stage retrieval pipeline to optimize recall and relevance before generating answers.
HyDE Retrieval[ 7]: In the first stage, the Gemma 3n model is prompted with a carefully crafted system prompt, the
user’s query, and a summary of the entire document to generate ahypothetical answerthat might appear in the policy
text. The embedding of this hypothetical answer is then used to retrieve a set of candidate chunks from ChromaDB.
Multi-query Generation[ 8]: The system generates five semantically diverse re-wordings of the user’s original query,
again using the document summary for context. Each reworded query is embedded and used to retrieve additional
candidate chunks.
Reciprocal Rank Fusion (RRF)[ 9]: Candidate sets from the HyDE and multi-query retrieval stages are merged
using Reciprocal Rank Fusion. RRF assigns a score to each chunk based on its rank across the different retrieval lists,
prioritizing chunks that consistently appear near the top in multiple rankings. Top- pfiltering is applied to the fused
results to select a dynamically determined set of highly relevant chunks.
Reranking:The final candidate set from RRF is passed to the BAAI/bge-reranker-base [10] cross-encoder model,
which assigns a fine-grained relevance score for each chunk concerning the original user query. A second top- pfiltering
step selects the highest-confidence chunks for downstream generation.
2.5 Generation
Answer generation is performed by the Gemma 3n model running locally via Ollama . The model is configured with a
context length of 32,000 tokens, although the full limit is rarely reached. The temperature is set to 0.1 to encourage
deterministic, factual outputs over creative variation. No explicit maximum output length is imposed. The generation
process is orchestrated by a custom LangChain pipeline using a domain-specific system prompt that enforces several
constraints:
• Only information from retrieved chunks should be used to generate the answer.
• Direct quotations from the source are preferred wherever applicable.
•If sufficient supporting context is absent, the model must respond with “not enough context” rather than
attempting to answer.
This prompt design serves as a central safeguard against hallucination, ensuring that all generated outputs are grounded
in the policy document.
2.6 Citation and Hallucination Control
PolicyBot employs a dual approach to ensure factual accuracy and maintain traceability. First, hallucination control
is enforced through the generation prompt described above, which explicitly restricts the model from introducing
knowledge that’s outside the supplied Policy documents. Second, citations are provided directly to the user via the
front-end. Beneath each answer, a collapsible section displays the exact text chunks that were passed to the generation
model, along with their metadata. This allows users to verify every statement against the original source. Dynamic top- p
filtering during retrieval further reduces noise in the input context, thereby minimizing the likelihood of unsupported
content.
2.7 User Interface
We implement an indicative user interface using Streamlit5, providing an interactive and accessible front end. The
interface supports free-form queries, displays multi-turn conversation history, and embeds interactive citations that
expand on demand. This design is intended to serve diverse audiences, including policy officers, journalists, NGOs,
and citizens, without requiring technical expertise. While the current deployment is local, the architecture allows
straightforward adaptation to cloud or on-premises environments.
5https://docs.streamlit.io/
4

PolicyBot - Reliable Question Answering over Policy Documents
3 Experimental Setup
To evaluate and select the optimal components for the retrieval-augmented generation (RAG) pipeline, a series of
controlled experiments was conducted. The evaluation focused on three key components: the large language model
(LLM) for answer generation, the embedding model for retrieval, and the chunking strategy for document segmentation.
3.1 LLM Decision
The large language model (LLM) is the central component of the PolicyBot ’s answer generation pipeline, responsible
for producing responses that are factually correct and contextually relevant to user queries. For this application, the
LLM needed to satisfy three crucial requirements:
1. Minimize hallucinations and ensure that all generated content is grounded in the retrieved context.
2. Maintain a professional and neutral tone suitable for interpreting policy-related queries.
3. Operate efficiently on consumer-grade hardware without significant degradation in quality.
To evaluate candidate models, a custom benchmark dataset was created consisting of policy-related questions, their
associated source document excerpts, and reference answers. This dataset was specifically curated to reflect the
linguistic and structural complexity of real-world policy documents, including long clauses, conditional statements, and
nested definitions. Six open-source LLMs available via theOllamaframework were selected for testing:
•deepseek-r1:8b[11]
•mistral:7b[12]
•llama3.1:8b[13]
•qwen2.5:7b[14]
•gemma:7b[15]
•gemma 3n:e4b[16]
Each model was evaluated using the same retrieval context and identical system prompt. The prompt was designed
to explicitly restrict the model to using only the provided context, encourage the inclusion of direct quotations where
relevant, and instruct the model to respond with “not enough context” if it could not confidently answer the question.
This ensured that the primary difference in performance would stem from the model’s inherent capabilities rather than
variations in retrieval quality or prompt structure. The evaluation procedure involves the following steps:
• Running each model locally on identical hardware to ensure a fair comparison in latency and resource usage.
•Assessing each generated answer on three primary axes:factual accuracy(measured against the reference
answer),adherence to the system prompt(compliance with hallucination control and citation requirements),
andstylistic quality(professional tone, clarity, and neutrality).
•Computing a set of quantitative metrics for each model: similarity, ROUGE-L [ 17], BLEU, METEOR, BERT-
based precision, recall, and F1scores [ 18]. These metrics capture both lexical overlap and semantic similarity
between generated answers and reference answers.
•Recording the proportion of cases in which the model explicitly responded with “not enough context,” serving
as an indicator of conservative generation behavior where relevant information wasn’t available in the retrieved
context.
Across these metrics, gemma 3n:e4b consistently achieved the highest performance, demonstrating a strong balance
between factual accuracy, semantic fidelity, and stylistic appropriateness, while adhering closely to the constraints on
hallucination mitigation. From this evaluation, gemma 3n:e4b emerged as the most balanced choice. It consistently
provided accurate, well-grounded responses, demonstrated strong adherence to hallucination mitigation instructions, and
maintained a clear, professional tone. Larger models such as llama3.3:8b andqwen2.5:7b occasionally generated
more verbose answers, but these often contained unsupported statements. Importantly, gemma 3n:e4b achieved
this performance while running efficiently, making it well-suited for the local, resource-constrained deployment
environments targeted byPolicyBot.
5

PolicyBot - Reliable Question Answering over Policy Documents
3.2 Embedding Model Selection
High-quality dense embeddings are critical to the retrieval stage of any RAG system: they determine whether semanti-
cally relevant document chunks are surfaced for downstream reranking and generation. For PolicyBot , the embedding
model needed to satisfy several practical and domain-specific requirements:
1.Semantic fidelity:The embeddings must capture subtle semantic relationships in policy text (e.g., conditionals,
exceptions, references) so that relevant clauses are retrieved even when queries are phrased differently from
the source wording.
2.Robustness to domain wording:The model should handle formal, legalistic, and bureaucratic language
common in policy documents.
3.Multilingual support:While the current system primarily handles English documents, multilingual capability
was desirable for potential future extensions.
4.Retrieval efficiency and compatibility:The embeddings should work well with approximate nearest neighbor
(ANN) indexing (ChromaDB with HNSW) and enable fast interactive queries.
To perform a standardized and reproducible comparison, we used the BEIR evaluation framework [ 4] with the SciFact
dataset [ 3]. SciFact was chosen because it provides a factual QA-style retrieval task that stresses semantic matching
between queries and scientific claims—an appropriate proxy for the factual, clause-oriented matching required in
policy documents. Retrieval performance was evaluated at multiple retrieval depths ( k), and models were compared
using precision, recall, and F1score at these depths to quantify both exact-match and soft-match retrieval quality. The
candidate embedding models are listed below.
•Alibaba-NLP/gte-multilingual-base[2]
•intfloat/multilingual-e5-large-instruct[19]
•jinaai/jina-embeddings-v3[20]
•Lajavaness/bilingual-embedding-large[21, 22, 23]
•Qwen/Qwen3-Embedding-0.6B[24]
•Snowflake/snowflake-arctic-embed-l-v2.0[25]
The evaluation procedure consists of the following steps:
•Index construction:For each candidate model, embeddings were generated for the SciFact corpus and
indexed in a local ChromaDB instance configured with the default HNSW ANN index and cosine similarity as
the distance metric.
•Retrieval experiments:For each model, standard BEIR retrieval scripts were used to run queries against the
index, collecting retrieval results at severalkvalues.
•Metric computation:Precision, recall, and F1score were computed at each kto assess how reliably each
model surfaced relevant documents across retrieval depths. These metrics capture the trade-off between
precision at shallow depths and recall at deeper retrievals.
•Controlled conditions:All experiments were executed on the same hardware and indexing configurations
to ensure comparability across models. Embedding generation was batched to reflect realistic production
batching behavior, and caching was consistently used across runs to minimize measurement variance resulting
from I/O differences.
Figure 2 summarizes the comparative retrieval performance across the tested models.
Alibaba-NLP/gte-multilingual-base consistently achieved the highest F1scores across a range of
retrieval depths, indicating a favorable balance between precision and recall for this task. In particular,
gte-multilingual-base exhibited stronger performance at moderate depths (e.g., k= 5 andk= 10 ), which are
commonly used in RAG pipelines to assemble candidate contexts for reranking and generation.
Beyond raw retrieval metrics, practical considerations also informed the selection. gte-multilingual-base demon-
strated stable behavior with ChromaDB’s HNSW index and produced embeddings that integrated efficiently into our
multi-stage retrieval pipeline (HyDE + multi-query + RRF + reranker). Its multilingual capability provides flexibility
for future extensions without compromising current English-language performance. Based on these quantitative and
qualitative observations,Alibaba-NLP/gte-multilingual-basewas selected as the embedding model for PolicyBot ’s
production pipeline.
6

PolicyBot - Reliable Question Answering over Policy Documents
(a) Recall scores for different embedding models
(b) Precision scores for different embedding models
(c)F 1scores for different embedding models
Figure 2: Comparative performance of embedding models at various retrieval depths
7

PolicyBot - Reliable Question Answering over Policy Documents
3.3 Chunking Strategies
The chunking strategy directly determines the semantic coherence of document chunks and, consequently, the quality of
retrieval in a RAG system. For the PolicyBot , chunking is especially critical because policy documents often contain
long, legally precise clauses, where splitting at arbitrary points can distort meaning or omit essential context.
3.3.1 Dataset Preparation and Modification
To benchmark different chunking strategies, we use the Microsoft/Wiki_QA dataset [ 26] as the base. This dataset
comprises pairs of natural language questions and corresponding sentences from Wikipedia articles, with each sentence
provided separately. For this evaluation, the dataset was modified as follows:
1.All sentences belonging to the same Wikipedia article title were concatenated into a single paragraph, preceded
by the article title as a heading. This was followed immediately by the next article title and its corresponding
concatenated content, producing a continuous, multi-article corpus.
2.The resulting structure created a long, uninterrupted text stream suitable for testing various chunking mecha-
nisms under realistic retrieval conditions.
3.The original question–answer mappings were preserved at the sentence level, allowing precise verification of
whether a retrieved chunk contained the gold-standard answer sentence.
3.3.2 Relevance Definition
For evaluation purposes, a retrieved chunk was consideredrelevantto a query if it contained the complete sentence
that was labeled as the correct answer in the original Wiki_QA dataset. If the answer sentence was not fully contained
within the chunk, the retrieval was marked as non-relevant, even if partial information was present. This ensured strict
relevance criteria aligned to return self-contained, verifiable policy clauses.
3.3.3 Chunking Strategies Tested
Two primary segmentation approaches, each with multiple configurations, were evaluated:
•Recursive Chunking[ 27]: Implemented using LangChain’s RecursiveCharacterTextSplitter , this
method segments text based on a fixed character length, attempting to break on natural boundaries (paragraphs,
sentences) when possible, but without semantic analysis. Multiple configurations were tested by varying the
chunk size (e.g., 750, 800, 1000 characters) and overlap length (200–250 characters) to assess the trade-off
between retrieval coverage and precision.
•Semantic Chunking[ 1]: Implemented using LangChain’s experimental semantic chunker, which leverages
statistical variance in sentence embeddings to determine natural breakpoints in the text. Several parameteriza-
tions were explored across different breakpoint detection methods, including gradient, percentile, and standard
deviation thresholds, to evaluate their impacts on semantic coherence and retrieval performance.
3.3.4 Evaluation Procedure
For each chunking strategy, multiple configurations were tested to account for variations in chunk size, overlap, and
sensitivity to semantic breakpoints. In the case of recursive chunking, configurations included fixed sizes of 750, 800,
and 1000 characters with overlaps ranging from 200–250 characters. For semantic chunking, configurations varied
across breakpoint detection methods, including gradient, percentile, and standard deviation thresholds (e.g., gradient at
0.75, percentile at 0.9, and standard deviation at 1.0). For each configuration:
1. The modified dataset documents were segmented according to the chosen method and parameter set.
2.Chunks were embedded using the selected production embedding model ( gte-multilingual-base ) and
indexed in ChromaDB with an HNSW index and cosine similarity.
3.Original Wiki_QA questions were used as retrieval queries, returning the top- kchunks for multiple values of
k.
4.Retrieval performance was measured using precision, recall, and F1score, where relevance was determined
using the strict criterion described above.
8

PolicyBot - Reliable Question Answering over Policy Documents
(a) Recall scores for different chunking strategies
(b) Precision scores for different chunking strategies
(c)F 1scores for different chunking strategies
Figure 3: Comparison of chunking strategies at various retrieval depths
9

PolicyBot - Reliable Question Answering over Policy Documents
3.3.5 Results
The semantic chunking configuration consistently outperformed recursive chunking across low kvalues. Its ability
to preserve complete logical units of text—such as entire clauses or multi-sentence definitions—led to higher recall
without sacrificing precision. This property is particularly valuable in the policy document setting, where losing part of
a clause can alter the interpretation of the text.
4 Deployment & Lessons Learned
The system was deployed as a web-based application, designed to operate in both online and offline settings. It uses a
Streamlit interface for interaction and runs entirely on local infrastructure, leveraging Ollama to host the Gemma 3n
model and ChromaDB for vector storage. This self-contained design enables deployment in environments with limited
or no internet connectivity, ensuring privacy of sensitive policy documents and eliminating reliance on external APIs.
The deployment target was to create a tool suitable for diverse stakeholders, such as students, journalists, NGOs, and
public officials.
4.1 Key Deployment Challenges
During deployment, several practical challenges emerged. One significant factor was the trade-off between latency
and accuracy. Increasing the number of retrieved chunks or expanding the context length generally improves response
accuracy but at the cost of slower inference. This necessitated careful tuning of retrieval parameters to strike a balance
between relevance and response time.
A second challenge was multilingual capability. While the chosen embedding model ( gte-multilingual-base ) sup-
ports multiple languages, the current implementation was evaluated primarily on English policy documents. Extending
reliable performance to other languages will require additional benchmarking.
Hardware constraints also influenced the deployment strategy. The system was tested on a consumer-grade NVIDIA
GeForce GTX 1650 GPU. This limited the size of models that could be used without excessive latency or memory
pressure. As a result, the architecture prioritized a lightweight LLM and an efficient retrieval pipeline.
4.2 Lessons Learned
Several insights were gained during the deployment process. First, incorporating direct citation tracing in the interface
significantly improved the ease of cross-verifying responses. By showing the exact text chunks that supported a
generated answer, the system allowed rapid validation of the output against the source document.
Second, semantic chunking proved critical for this domain. Maintaining the logical coherence of policy clauses within a
chunk substantially improved retrieval quality compared to fixed-size segmentation.
Finally, local inference was found to be a viable strategy for policy document QA. By carefully selecting and optimizing
models and retrieval parameters, it was possible to deliver grounded, context-aware responses with minimal infrastructure
requirements.
5 Ethical Considerations
The deployment of a retrieval-augmented generation (RAG) chatbot for policy documents raises several critical ethical
considerations. Given that the system is intended for contexts where the accuracy and transparency of information have
direct social and governance implications, it is crucial to address potential risks in bias, fairness, transparency, and
privacy.
5.1 Bias and Fairness
Large Language Models (LLMs) can inherit biases from their training data, which may manifest in subtle ways when
interpreting or summarizing policy documents. Such biases could lead to the prioritization of specific interpretations
over others or the omission of relevant but less frequently represented perspectives. This is particularly significant
in policy contexts, where nuanced phrasing can alter the perceived meaning of a clause. While the current system
minimizes this risk by constraining the model to operate strictly within the retrieved document context, residual bias
may still arise through the ranking and selection of chunks. Future work should explore bias detection frameworks and
the integration of fairness-aware retrieval techniques to ensure balanced representation in retrieved content.
10

PolicyBot - Reliable Question Answering over Policy Documents
5.2 Transparency in Governance Contexts
Transparency is essential for maintaining accountability in policy interpretation. The chatbot addresses this by
implementing a citation tracing mechanism, wherein the exact text chunks used for answer generation are displayed
alongside the output. This design enables rapid cross-verification of responses against the source material, reducing the
likelihood of unsubstantiated claims. In governance contexts, where multiple stakeholders may scrutinize decisions, the
ability to trace an answer to its originating source is a safeguard against misinformation and misinterpretation. However,
transparency also depends on the completeness of retrieved content; if the retrieval pipeline fails to include key relevant
passages, even a fully cited answer may be incomplete.
5.3 Privacy and Data Protection
Policy documents can range from publicly available legislation to internal regulatory guidelines that may contain
sensitive or confidential information. The local-first design of the system, with no external API calls during query
processing, mitigates the risk of sensitive data leakage. All document parsing, embedding, and inference occur within
the local deployment environment, ensuring that the content never leaves the host machine. Nevertheless, when used in
shared or institutional settings, additional safeguards—such as encrypted storage of embeddings and access control for
the user interface—should be considered to prevent unauthorized access. Furthermore, if future versions incorporate
multilingual or cross-border policy analysis, varying data protection regulations (e.g., GDPR, DPDP Act) will need to
be explicitly addressed in the system design.
5.4 Ethical Deployment Practices
Ethical deployment of such systems requires an understanding that they are assistive tools, not authoritative sources of
legal interpretation. Outputs should be positioned as aids to human decision-making, rather than definitive rulings. Dis-
claimers in the interface can reinforce this distinction, and careful user training can further mitigate risks of over-reliance.
Additionally, reproducibility and auditability of system outputs—achieved through open-source implementation and
fixed-version component releases—help ensure that results can be independently verified.
In summary, while the current architecture incorporates mechanisms for transparency and privacy, continuous attention
to bias mitigation, data governance, and the framing of outputs is necessary for the responsible deployment of policy
contexts. These considerations are not static; they must evolve in tandem with advances in LLM capabilities, retrieval
methods, and the legal frameworks governing digital policy tools.
6 Future Work
While the current system demonstrates the feasibility and benefits of a domain-specific Retrieval-Augmented Generation
(RAG) pipeline for policy documents, several promising avenues for further development remain. These directions span
technical, linguistic, and user-centered aspects to expand applicability and impact.
6.1 Domain Adaptation to Broader Legal and Regulatory Texts
The present work focuses on public policy documents; however, the architecture can be extended to handle diverse legal
corpora, including statutes, judicial opinions, regulatory guidelines, and international treaties. Each of these domains
presents unique structural and linguistic challenges: statutes often have deeply nested numbering schemes, judicial
opinions contain extensive citations and precedent references, and treaties incorporate multilingual parallel clauses.
Domain adaptation will require refining chunking strategies to preserve logical integrity in these formats, as well as
integrating embeddings fine-tuned on legal corpora to improve retrieval accuracy.
6.2 Multilingual and Cross-Jurisdictional Support
Although the selected embedding model supports multilingual input, the current deployment primarily operates in
English. Expanding to fully multilingual policy datasets would make the system valuable in multilingual governance
contexts—such as Indian state government portals, EU policy repositories, or UN reports. This extension would require
curated multilingual corpora, rigorous cross-lingual retrieval benchmarking, and evaluation of translation fidelity when
retrieved content is displayed alongside its original-language source.
11

PolicyBot - Reliable Question Answering over Policy Documents
6.3 Enhanced Hallucination Control
The current system enforces hallucination control through strict prompting, citation tracing, and top- pfiltering. Future
enhancements could include retrieval confidence estimation (e.g., using similarity score thresholds), automated factuality
checks against authoritative sources, and fine-tuning the LLM on citation-heavy legal datasets to strengthen grounded
generation. Another promising approach is multi-model cross-verification, where two independent models must
converge on the same answer before it is presented to the user.
6.4 User-Centric Design and Evaluation
A key next step is to integrate user feedback systematically. While the current system was primarily evaluated through
technical benchmarks, human-centered studies can reveal usability barriers, preferred answer formats, and the extent to
which citation tracing enhances comprehension. Structured evaluations involving journalists, NGO workers, and policy
analysts could measure perceived trustworthiness, cognitive effort saved, and the likelihood of adopting such tools in
daily workflows.
6.5 Scalability and Efficiency Improvements
Scaling the system to process and serve thousands of long policy documents will require advances in both retrieval
infrastructure and inference optimization. Techniques such as optimized HNSW index parameter tuning, distributed
vector search, and model quantization can reduce latency while maintaining retrieval accuracy. These optimizations are
especially critical for deployment in resource-constrained or offline environments.
7 Conclusion
This work presented PolicyBot , a retrieval-augmented generation pipeline designed explicitly for navigating dense,
jargon-heavy policy documents. By combining semantic chunking, multilingual dense embeddings, multi-stage retrieval
with reranking, and strict hallucination control mechanisms, the system delivers factually grounded answers with
transparent source citations. The architecture prioritizes privacy and low-latency inference through local deployment,
making it particularly suitable for under-resourced environments where cloud-based solutions are impractical.
The experimental evaluation systematically benchmarked critical components of the pipeline, including the choice
of LLM, embedding model, and chunking strategy, using standardized datasets and relevance-based retrieval metrics.
The resulting configuration—centered on the Gemma 3n:e4b ,gte-multilingual-base embeddings, and semantic
chunking—strikes a balance between accuracy, efficiency, and transparency.
Beyond technical contributions, this work highlights a broader principle: domain-specific RAG systems, when designed
with traceability and verifiability as primary goals, can make complex governance information more accessible without
compromising trust. The citation tracing interface not only provides factual grounding but also enables independent
verification, empowering users to interpret and act upon policy information with greater confidence.
Looking ahead, the outlined future work opens a path toward a multilingual, legally aware, and user-validated
PolicyBot that could serve as a dependable assistant for policy professionals, civil society organizations, and citizens
worldwide. By maintaining a focus on transparency, fairness, and privacy, such systems can make a meaningful
contribution to the democratization of policy understanding in an increasingly complex information landscape.
8 Demo & Source
The source code of the PolicyBot is available on GitHub6. The instructions for using the demo are provided on the
demo page. The datasets used in this study will be shared upon reasonable request.
Acknowledgments
We thank the patrons of the Centre for Responsible AI7at IIT Madras for their continued support and assistance in
building the PolicyBot.
6https://github.com/cerai-iitm/policybot.git
7https://cerai.iitm.ac.in/
12

PolicyBot - Reliable Question Answering over Policy Documents
References
[1]LangChain. How to split text based on semantic similarity | langchain. https://python.langchain.com/
docs/how_to/semantic-chunker/, 2025.
[2]Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun
Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. mgte: Generalized long-context text representation
and reranking models for multilingual text retrieval. In Franck Dernoncourt, Daniel Preotiuc-Pietro, and Anastasia
Shimorina, editors,Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing:
EMNLP 2024 - Industry Track, Miami, Florida, USA, November 12-16, 2024, pages 1393–1412. Association for
Computational Linguistics, 2024.
[3]David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh
Hajishirzi. Fact or fiction: Verifying scientific claims. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu,
editors,Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP
2020, Online, November 16-20, 2020, pages 7534–7550. Association for Computational Linguistics, 2020.
[4]Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A heterogeneous
benchmark for zero-shot evaluation of information retrieval models. In Joaquin Vanschoren and Sai-Kit Yeung,
editors,Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS
Datasets and Benchmarks 2021, December 2021, virtual. NeurIPS Datasets and Benchmarks 2021, 2021.
[5] Chroma Team. Chroma documentation.https://docs.trychroma.com/, 2025.
[6]Yu A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical
navigable small world graphs.IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4):824–836,
2020.
[7]Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval without relevance
labels, 2022.
[8]Kouji Tahata and Kohei Matsuda. Diagonals-parameter symmetry model and its property for square contingency
tables with ordinal categories, 2023.
[9]Gordon V . Cormack, Charles L A Clarke, and Stefan Buettcher. Reciprocal rank fusion outperforms condorcet and
individual rank learning methods. InProceedings of the 32nd International ACM SIGIR Conference on Research
and Development in Information Retrieval, SIGIR ’09, page 758–759, New York, NY , USA, 2009. Association for
Computing Machinery.
[10] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Packaged resources to advance general
chinese embedding, 2023.
[11] DeepSeek. Deepseek-r1:8b.https://ollama.com/library/deepseek-r1:8b, 2025.
[12] Mistral AI. Mistral-7b-instruct-v0.2. https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 ,
2023.
[13] Meta AI. Llama 3.1 8b.https://ollama.com/library/llama3.1:8b, 2024.
[14] Qwen Team. Qwen2.5: A party of foundation models, September 2024.
[15] Thomas Mesnard Gemma Team, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Laurent Sifre, Morgane
Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, and et al. Gemma. 2024.
[16] Gemma Team. Gemma 3n. 2025.
[17] Alessia Auriemma Citarella, Marcello Barbella, Madalina G. Ciobanu, Fabiola De Marco, Luigi Di Biasi, and
Genoveffa Tortora. Assessing the effectiveness of ROUGE as unbiased metric in extractive vs. abstractive
summarization techniques.J. Comput. Sci., 87:102571, 2025.
[18] Hadeel Saadany and Constantin Orasan. Bleu, meteor, bertscore: Evaluation of metrics performance in assessing
critical translation errors in sentiment-oriented text.CoRR, abs/2109.14250, 2021.
[19] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Multilingual E5 text
embeddings: A technical report.CoRR, abs/2402.05672, 2024.
[20] Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang,
Georgios Mastrapas, Andreas Koukounas, Andreas Koukounas, Nan Wang, and Han Xiao. jina-embeddings-v3:
Multilingual embeddings with task lora, 2024.
[21] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán,
Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation
learning at scale.arXiv preprint arXiv:1911.02116, 2019.
13

PolicyBot - Reliable Question Answering over Policy Documents
[22] Iryna Gurevych Nils Reimers. Sentence-bert: Sentence embeddings using siamese bert-networks.
https://arxiv.org/abs/1908.10084, 2019.
[23] Nandan Thakur, Nils Reimers, Johannes Daxenberger, and Iryna Gurevych. Augmented sbert: Data augmentation
method for improving bi-encoders for pairwise sentence scoring tasks.arXiv e-prints, pages arXiv–2010, 2020.
[24] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and
reranking through foundation models.CoRR, abs/2506.05176, 2025.
[25] Snowflake Inc. Snowflake arctic embed l v2.0. https://huggingface.co/Snowflake/
snowflake-arctic-embed-l-v2.0, 2024.
[26] Yi Yang, Wen-tau Yih, and Christopher Meek. WikiQA: A challenge dataset for open-domain question answering.
InProceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 2013–2018,
Lisbon, Portugal, September 2015. Association for Computational Linguistics.
[27] LangChain. Recursivecharactertextsplitter — langchain api reference. https://python.langchain.
com/api_reference/text_splitters/character/langchain_text_splitters.character.
RecursiveCharacterTextSplitter.html, 2025.
14