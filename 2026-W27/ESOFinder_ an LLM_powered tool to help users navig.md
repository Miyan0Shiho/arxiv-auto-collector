# ESOFinder: an LLM-powered tool to help users navigate ESO documentation

**Authors**: P. Sánchez-Sáez, C. Reinero, M. Vioque, M. Wittkowski, M. Rejkuba, M. Romaniello, A. Barnes, J. Pritchard, M. Marsset

**Published**: 2026-06-29 09:28:29

**PDF URL**: [https://arxiv.org/pdf/2606.30029v1](https://arxiv.org/pdf/2606.30029v1)

## Abstract
The large amount and diversity of documentation available for users of the European Southern Observatory (ESO), spanning the full observing lifecycle from proposal preparation and observation planning to data reduction and archival access, makes it increasingly challenging for the astronomical community to efficiently find relevant information. To address this, we have developed ESOFinder, an in-house chatbot powered by Large Language Models (LLMs) and Retrieval Augmented Generation (RAG). ESOFinder integrates public information from instrument manuals, phase 1/2/3 documentation, data reduction pipeline manuals, the ESO Knowledge Base, and key web resources (spanning more than 3500 links and over 100 manuals) to provide concise, context-aware, and reference-linked answers to user queries about proposal or observation preparation, data retrieval, and data processing. Built on open-source LLMs, running on a local server, ESOFinder ensures data privacy, transparency, and complete control over the knowledge base. Its multi-step architecture allows verification of retrieved documents and generated answers, reducing the risk of hallucinations and improving the reliability of responses compared to commercial tools. The current version of ESOFinder is being tested internally at ESO to evaluate its performance, assess its integration with internal workflows, and identify limitations in coverage and accuracy. These tests will guide further improvements, including the incorporation of additional documentation, and enhanced retrieval strategies. Ultimately, ESOFinder aims to become an interface for users to navigate ESO's complex documentation ecosystem and to support both staff and community astronomers in their daily tasks.

## Full Text


<!-- PDF content starts -->

ESOFinder: an LLM-powered tool to help users navigate
ESO documentation
Paula S´ anchez-S´ aeza, Claudio Reineroa, Miguel Vioquea, Markus Wittkowskia, Marina
Rejkubaa, Martino Romanielloa, Ashley Barnesa, John Pritcharda, and Micha¨ el Marsseta
aEuropean Southern Observatory, Karl-Schwarzschild-Strasse 2, 85748 Garching bei M¨ unchen,
Germany
ABSTRACT
The large amount and diversity of documentation available for users of the European Southern Observatory (ESO)
– spanning the full observing lifecycle from proposal preparation and observation planning to data reduction and
archival access – makes it increasingly challenging for the astronomical community to efficiently find relevant
information. To address this, we have developed ESOFinder, an in-house chatbot powered by Large Language
Models (LLMs) and Retrieval-Augmented Generation (RAG). ESOFinder integrates public information from
instrument manuals, phase 1/2/3 documentation, data reduction pipeline manuals, the ESO Knowledge Base,
and key web resources (spanning more than 3500 links and over 100 manuals) to provide concise, context-aware,
and reference-linked answers to user queries about proposal/observation preparation, data retrieval, and data
processing. Built on open-source LLMs (e.g., Mistral AI models) running on a local server, ESOFinder ensures
data privacy, transparency, and complete control over the knowledge base. Its multi-step architecture allows
verification of retrieved documents and generated answers, reducing the risk of hallucinations and improving the
reliability of responses compared to commercial tools. The current version of ESOFinder is being tested internally
at ESO to evaluate its performance, assess its integration with internal workflows, and identify limitations in
coverage and accuracy. These tests will guide further improvements, including the incorporation of additional
documentation, and enhanced retrieval strategies. Ultimately, ESOFinder aims to become an interface for users
to navigate ESO’s complex documentation ecosystem and to support both staff and community astronomers in
their daily tasks.
Keywords:AI, LLMs, RAG, Instrument Manuals, Pipelines, Observatory Documentation, Chatbot
1. INTRODUCTION
Modern astronomical observatories generate vast and continuously evolving documentation ecosystems that
users must navigate to successfully plan and execute their observing programmes. At the European Southern
Observatory (ESO), the documentation of La Silla Paranal Observatory spans more than 3500 links across ESO
science webpages∗and Knowledge base†, and over 100 manuals distributed as PDF documents, totalling over
6.2 million words (roughly eight times the length of the King James Bible, or the equivalent of over 17 days of
continuous reading at average pace). It encompasses a broad and heterogeneous set of resources covering the full
lifecycle of an observational programme: from Phase 1, the proposal submission stage in which astronomers define
their science case, describe the programme feasibility, and provide a target list, instrument configuration, and time
request for evaluation and allocation; through Phase 2, the observation preparation and execution stage in which
accepted proposals are translated into detailed observing blocks (OBs) specifying telescope pointings, instrument
setups, and execution constraints, and in which the observing programmes are executed and monitored; to Phase
3, the data product submission stage in which principal investigators contribute science-ready reduced data
back to the ESO archive for public release. Beyond these phase-specific resources, users must also navigate
instrument user manuals, data reduction pipeline documentation, and archive interface instructions. Locating
Further author information: (Send correspondence to P.S.S.)
P.S.S.: E-mail: paula.sanchezsaez@eso.org
∗https://www.eso.org/sci.html
†https://support.eso.org/en-GB/kbarXiv:2606.30029v1  [astro-ph.IM]  29 Jun 2026

precise, up-to-date information within this extensive corpus represents a significant and growing challenge, both
for the astronomical community and for ESO staff, who assist users on a daily basis and must maintain curated,
current documentation across all of these resources. Motivated by this need, we present ESOFinder, a question-
answering system designed to assist ESO users by leveraging recent advances in natural language processing
(NLP), specifically large language models (LLMs) and retrieval-augmented generation (RAG).
An LLM is an artificial intelligence (AI) model designed to understand and generate human-like text. Most
modern LLMs are based on the Transformer architecture introduced by Vaswani et al. (2017),1and are trained
on large, diverse text corpora that enable them to perform tasks such as translation, summarisation, and open-
ended question answering. The term “large” refers to the substantial number of parameters these models con-
tain (typically in the billions), which allows them to capture complex linguistic patterns and generate coherent
responses. While general-purpose LLMs such as those powering ChatGPT,2Gemini,3or Claude4have demon-
strated impressive capabilities, they present several limitations: their training data has a fixed cutoff date and
cannot incorporate domain-specific or frequently updated knowledge; they are prone to generating hallucinated
answers, as they are designed to always produce a response rather than acknowledge uncertainty; and their use
may raise privacy and security concerns when sensitive or confidential information is involved. These limitations
are particularly pronounced in an observatory context, where documentation is continuously revised to reflect
new instrument capabilities, operational policies, and observing modes. Moreover, a substantial amount of out-
dated information persists on the Internet (like legacy instrument manuals or webpages from previous observing
periods), making it difficult for general-purpose LLMs to distinguish current guidelines and policies from ob-
solete ones. To address this, ESOFinder employs RAG,5an approach that augments a generative model with
an external retrieval mechanism. In RAG, when a user submits a query, a retrieval component first searches a
curated knowledge base to identify relevant documents; this retrieved content is then provided as context to the
generative model, which synthesizes a response grounded in both its pre-trained knowledge and the dynamically
retrieved information. This three-stage pipeline – retrieve, augment, generate – makes RAG particularly well-
suited to knowledge-intensive and domain-specific applications such as observatory user support, where accuracy
and timeliness of information are critical.
In this paper, we introduce ESOFinder, an in-house RAG-based chatbot developed within the User Support
Department (USD) at ESO Headquarters. We describe its design, including the hybrid retrieval pipeline, the
answer-generation strategy, and the multi-step architecture used to reduce hallucinations and improve response
reliability. We also present a quality assessment of the tool based on feedback collected from ESO staff during
internal testing. Section 2 describes the system architecture; Section 3 presents the evaluation results; and
Section 4 summarises our findings and outlines future directions.
2. ESOFINDER ARCHITECTURE
ESOFinder is composed of two operationally distinct phases: anoffline ingestion pipelinethat constructs the
searchable knowledge base from ESO documentation, and anonline inference pipelinethat handles user queries
at runtime. The following sections provide a high-level overview of the system components and their interactions.
2.1 Document Ingestion Pipeline
Before the system can answer questions, all relevant documentation must be processed and indexed. This offline
pipeline transforms heterogeneous source material (both PDF manuals and web-based content) into a structured,
searchable knowledge base through three main steps.
Content extraction and parsing:ESO documentation spans a wide range of formats and topics. PDF
documents are parsed into structured text that preserves the original heading hierarchy and layout. Web-based
documentation is scraped and converted into a consistent plain-text representation. All content is organised into
domain-specific collections covering proposal preparation, instrument configuration, data delivery and access,
data reduction, and general observatory support, with dedicated coverage for La Silla Paranal Observatory
operations. ALMA documentation is maintained as a separate collection, as its instrumentation, operating
modes, and documentation conventions differ substantially from those of other ESO facilities.
Chunking and contextual enrichment:The extracted text is split into overlapping chunks using a
hierarchical splitting strategy that respects document structure: sections are first divided at heading boundaries

before a secondary token-level split is applied to longer segments, keeping a maximum of approximately 1,000
tokens (roughly 750 words) per chunk with a small overlap between adjacent chunks. This ensures that chunk
boundaries align with natural document divisions rather than arbitrary character counts. In addition, we apply
a contextual retrieval‡approach, where each chunk is passed to the LLM together with a summary of its parent
document, and the model is asked to generate a short description situating the chunk within its broader context.
That enriched description is then added to the chunk before embedding, improving the quality of vector search
for passages that would otherwise appear ambiguous when read in isolation.
Dual indexing:Each enriched chunk is indexed in two complementary ways. Dense vector embeddings are
produced with a sentence-transformer model (nomic-embed-text-v16) and stored in a ChromaDB§vector store,
with separate collections per documentation domain. Simultaneously, all chunks are indexed in an Elasticsearch¶
BM257keyword index. The rationale for maintaining both representations is discussed in Section 2.2. The inges-
tion step also supports zero-downtime updates: a rebuilt index is first written to a staging area and then instantly
replaces the existing one, so the live service continues answering queries uninterrupted while documentation is
updated.
2.2 Hybrid Retrieval and Reranking
When a query arrives, the system retrieves candidate documents through a three-stage pipeline designed to
maximise both recall and precision. A summary of the RAG strategy is presented in Figure 1, and further
explained bellow.
Parallel hybrid search:The user’s question is submitted simultaneously to two search systems, shown as
theSemanticandKeyword-basedbranches in Figure 1. TheSemanticbranch uses vector search: the query
is converted into a dense embedding and compared against the chunk embeddings pre-computed during the
documentation ingestion and stored in the ChromaDB collections (Section 2.1). Passages are ranked by cosine
similarity,8a measure of how closely their meaning matches the query, allowing the system to retrieve relevant
content even when the wording differs from the original question. To avoid returning near-duplicate passages,
the top results are further refined using Maximal Marginal Relevance (MMR9), which balances relevance to
the query against diversity among the selected passages. TheKeyword-basedbranch uses BM25,7a classical
ranking function that scores passages according to the frequency and specificity of query terms appearing in the
text. This makes it particularly effective for exact matches on technical vocabulary such as instrument names,
acronyms, and OB templates, terms that semantic search may fail to distinguish because they do not separate
well in embedding space. Combining both methods is a deliberate design choice: semantic search captures
meaning and handles paraphrasing, while keyword search ensures that precise technical terms are not missed.
Each branch independently selects 16 candidate documents, which are then merged for reranking.
Cross-encoder reranking:The candidate document chunks selected in the previous step are reranked
by a dedicated cross-encoder model (BAAI/bge-reranker-base‖). Unlike the embedding-based retriever, which
scores documents independently, the cross-encoder evaluates each (query, passage) pair jointly, producing a more
accurate relevance estimate.
Weighted score combination:
Results from both search branches are merged into a single ranked list using a weighted scoring formula. Each
passage receives a final score that combines two components: a semantic score, based on how well the passage
matches the meaning of the query, and a keyword score, based on how well it matches the exact terms used. The
semantic component is weighted more heavily (α= 0.7), reflecting its generally stronger retrieval performance:
sfinal=α·svec
reranker ·wcoll+ (1−α)·0.1·sBM25
reranker
rBM25 + 1,(1)
‡https://www.anthropic.com/engineering/contextual-retrieval
§https://github.com/chroma-core/chroma
¶https://github.com/elastic/elasticsearch
‖https://huggingface.co/BAAI/bge-reranker-base

Figure 1. ESOFinder RAG strategy summary, including semantic and keyword-based document retrieval and reranking
strategy.
wheresvec
reranker andsBM25
reranker are the cross-encoder relevance scores for candidates retrieved by the semantic and
keyword branches respectively, andr BM25 is the original rank of the passage in the keyword results; together
with the damping factor 0.1, this ensures that lower-ranked keyword matches contribute progressively less to the
final score. The termw collis a boost factor that favors documentation collections most relevant to the type of
question being asked, as determined by the query classifier (Section 2.3). This allows the system to prioritize the
most relevant subset of documentation without excluding any collection entirely. A final filtering step removes
low-scoring passages and limits the total context size before passing it to the answer generation step.
2.3 Agentic Query-Answer Workflow
The query-to-answer loop is implemented as a self-correcting agentic workflow using LangGraph∗∗, a framework
for building structured, multi-step LLM pipelines. The workflow performs several LLM-driven steps before and
after retrieval, and includes automatic retry logic to improve answer quality. Figure 2 summarizes the agentic
loop strategy used by ESOFinder.
All LLM calls within the workflow (query classification, question phrasing, message condensation, answer
generation, and answer grading) are handled by a single local model,Mistral-Small-3.1-24B-Instruct-2503††
∗∗https://github.com/langchain-ai/langgraph
††https://mistral.ai/news/mistral-small-3-1

Figure 2. ESOFinder agentic query-answer loop summary. The yellow arrows indicate steps that make use of the
mistral-small-3.1LLM. The retrieve step follows the strategy presented in Figure 1.
(hereaftermistral-small-3.1), served via vLLM.10Using a single model for all tasks simplifies deployment and
ensures consistent behaviour across the pipeline.
Query understanding:Before retrieval begins, two preparatory steps are applied. For multi-turn conversa-
tions, the latest user message is condensed into a fully self-contained question that clarifies implicit references to
previous turns. The question is then classified into one of seven observatory-specific domains (proposal prepara-
tion, instrument configuration, data delivery, pipeline reduction, ALMA, observing support, and general queries)
to set the collection weighting (w collin Eq. 1) for retrieval (classify query). Next, an alternative phrasing of
the question (review query) is generated and submitted alongside the original to broaden retrieval coverage.
Document retrieval and grading:The hybrid retrieval pipeline (Section 2.2) is executed for each question
variant, and the resulting candidate documents are merged, deduplicated, and ranked by their final combined
score (retrieve). A grading step then discards passages whose combined retrieval score falls below a minimum
threshold and retains at most eight documents to be passed to the generation step (grade documents). If
no documents survive filtering, the query is rewritten (transform query) and retrieval is retried up to two
additional times; if no relevant documents are found after all retries, a fallback response is returned directly
without invoking the generation step (nodocs endnode).
Answer generation:Themistral-small-3.1LLM generates a structured answer grounded in the retrieved
documents, including inline citations and links to the source documentation (generate). All prompts include
an ESO-specific terminology glossary to assist the model with observatory acronyms and instrument names.

Self-correction loop:Two grading steps evaluate the generated answer using themistral-small-3.1
LLM: one checks whether the answer is factually grounded in the retrieved documents (supported), and another
checks whether it addresses the question asked (useful). If both checks pass, the generated answer is presented
to the user (end node). If either check fails, the system rewrites the query (transform query) and retries the
retrieval-generation cycle up to two additional times before returning a fallback response (error endnode). This
loop reduces the rate of hallucinated or off-topic answers without requiring human intervention.
2.4 Deployment and User Interface
The ESOFinder pipeline is served through a streaming HTTP backend (FastAPI‡‡) that returns intermediate
status events and the final answer incrementally, providing users with real-time feedback on system progress.
The LLM inference server runs on two GPUs (NVIDIA RTX 6000 Ada Generation) in tensor-parallel mode; the
cross-encoder reranker is kept on CPU to prevent memory contention with the LLM inference server.
Users can access ESOFinder in two ways. The primary interface is a web application that supports multi-
turn conversations, i.e., the user can ask follow-up questions within the same session, and the system remembers
the context of previous exchanges. Multiple independent interfaces can be deployed simultaneously for different
user groups, with access currently restricted to ESO staff, fellows, and students during the evaluation phase. In
addition, ESOFinder exposes an Application Programming Interface (API) service that is currently integrated
with the USD helpdesk system: when a new support ticket is received, the API is queried automatically, and the
system’s answer is made available to the support astronomer handling the ticket, who may use it to streamline
the response process. Both interfaces collect user feedback for system evaluation
3. ESOFINDER’S FEEDBACK EXERCISE
ESOFinder has undergone three formal feedback exercises since its initial deployment, each informing a subse-
quent iteration of the system. In all these exercises, users were asked to evaluate each system response using
a two-part feedback form: a five-star overall rating, and a qualitative classification selected from a fixed set of
options –Perfect answer, accurate;Correct answer, but missing information;Correct answer, but too verbose;
Correct, but not useful in the context of the question;Irrelevant;Incorrect answer;Confusing; andCannot be
answered by ESOFinder. The star rating provides an overall performance score, while the qualitative labels helps
identify the specific nature of each success or failure. The version presented in this paper represents the third
generation of the tool, incorporating lessons learned from the first two rounds of evaluation, and is currently
undergoing a broader assessment by ESO staff, fellows, and students.
The first feedback exercise was conducted with ESO’s user support astronomers, who are responsible for
providing support for VLT observations (12 members of the ESO’s User Support Group – USG), using an early
version of ESOFinder that employed a simpler agentic workflow, less extensive documentation, and a less refined
document retrieval strategy than the architecture described in Section 2. The overall rating averaged 3.0 out
of 5 stars, with negative marks driven primarily by irrelevant or incorrect answers. In approximately one third
of the queries, the system’s response was judged to be confusing or incorrect. It is worth noting, however, that
a fraction of these cases could be traced back to gaps or ambiguities in the underlying documentation rather
than failures in the retrieval or generation pipeline. Analysis of the feedback revealed several recurring patterns:
questions about specific instruments (such as VISIR) received proportionally more negative reviews than average,
the system struggled with convoluted or multi-part questions, and answers were often perceived as overly verbose,
which was identified as the primary source of confusion. These findings motivated the incorporation of additional
documentation and adjustments to the system prompts to improve answer conciseness, resulting in a second
version of the tool with an improved agentic loop and a more diverse knowledge base.
The second feedback exercise was conducted with a broader audience, including staff responsible for main-
taining ESO’s data pipelines, operating the science archive, and supporting observers throughout the observing
process (∼20 members of the ESO Data Management and Operations – DMO– division), yielding an average
rating of 3.1 out of 5 stars. The response quality distribution was notably bimodal, with a higher proportion of
answers rated either as perfect or as incorrect compared to the first round, suggesting that the system performs
‡‡https://fastapi.tiangolo.com

well on straightforward queries but still fails on a subset of more challenging ones. As in the first round, a frac-
tion of incorrect answers could be attributed to documentation shortcomings. Specific issues identified included
retrieval failures on documents containing large tables, the absence of ALMA documentation from the indexed
knowledge base, incomplete coverage of Astronomical Data Query Language (ADQL) query documentation, and
Phase 3 documentation being retrieved in response to general queries for which it was not the most appropriate
source. The results of this second exercise motivated a substantial redesign of the system, resulting in a third
version with an improved agentic loop and a more structured retrieval pipeline, in particular a refined RAG
strategy and a reorganisation of the domain-specific document collections.
Figure 3. Summary of the third feedback exercise. The left panel shows the distribution of the five-star overall rating,
while the right panel shows the more detailed qualitative classification.
The current version of ESOFinder, described in this paper, incorporates the improvements driven by the first
two feedback exercises and is currently being evaluated by ESO staff astronomers, fellows, and students through
the web interface described in Section 2.4. Preliminary results collected between April and May 2026 indicate
a clear improvement in performance: the average star rating has increased to 3.84 out of 5, with a median of
5 stars. When only DMO staff are considered in the feedback evaluation, the average star rating is 4.08, with
a median of 5 stars. The majority of detailed feedback responses classify the answers as eitherPerfect answer,
accurateorCorrect answer, but missing information, and the fraction of responses labelled as incorrect has
notably reduced compared to previous evaluation rounds. Figure 3 shows the distribution of star ratings and
detailed qualitative feedback labels collected during this evaluation period.
It should be noted that the three evaluation exercises were conducted with different audiences: the first two
rounds involved primarily instrument and operations experts from the DMO division, while the current exercise
covers a broader and more representative sample of ESO users. This evolution in audience composition is an
important caveat when comparing results across rounds, but it also means that the latest evaluation provides a
more realistic picture of system performance, as would be expected when used by the broader ESO astronomical
community.
under real-world usage conditions.
4. CONCLUSIONS
We have presented ESOFinder, a domain-specific question-answering system developed in-house at ESO to
assist staff and community astronomers in navigating the observatory’s extensive documentation ecosystem.
The system is built on a RAG architecture combining hybrid retrieval (semantic search and BM25 keyword
indexing), cross-encoder reranking, and a self-correcting agentic workflow powered by a locally hosted open-
source LLM. By grounding all responses in a curated, versioned knowledge base, ESOFinder provides concise,
reference-linked answers while avoiding the limitations of general-purpose tools, in particular their fixed training
cutoff and lack of domain-specific knowledge.
Three successive feedback exercises have guided the iterative development of the system. Each round identi-
fied concrete weaknesses in retrieval coverage, documentation completeness, and response quality, and directly

motivated targeted improvements to the agentic loop, the RAG strategy, and the organisation of the domain-
specific document collections. Preliminary results from the ongoing third evaluation round, covering a broader
and more representative user population, show a clear improvement in performance relative to earlier versions,
with an average star rating of 3.84 out of 5 and a median of 5 stars, and a shift in the qualitative feedback
distribution towards correct answers.
Several limitations remain and will guide future development. Retrieval quality on documents containing
large tables and specific information presented as figures (instead of text) requires further work. The ALMA
knowledge base, currently maintained as a separate collection, will be expanded and better integrated with the
rest of the documentation. Furthermore, the feedback exercise has revealed inconsistencies and gaps in the
underlying ESO documentation itself, highlighting the importance of maintaining a well-curated and up-to-date
knowledge base.
Looking ahead, we identify several directions for future work. We plan to leverage ESOFinder more sys-
tematically as a tool for documentation quality control and improvement, using it to identify contradictions,
ambiguities, and outdated content across the documentation on a regular basis. On the technical side, we plan
to explore alternative LLM serving strategies and hierarchical multi-agent architectures that could improve both
response speed and the system’s ability to navigate the layered structure of ESO documentation. We also aim to
extend the coverage of the knowledge base by integrating the helpdesk ticket history from the DeskPro system,
which would allow ESOFinder to retrieve relevant past tickets and their solutions when support astronomers
handle new queries, while ensuring that access to sensitive ticket data remains restricted to authorised USD staff.
For this, the use of local LLMs presents a clear advantage, as sensitive information can be fed to the LLM without
risking data leaks. In parallel, we are exploring the feasibility of adapting the ESOFinder framework to serve
the needs of Paranal operations staff, for example through a dedicated interface tailored to on-site operational
documentation and tools used by the telescope and instrument operators. We plan to make the system available
to the broader ESO user community, establishing ESOFinder as a standard interface for documentation queries
across the observatory.
ACKNOWLEDGMENTS
The authors thank all members of the User Support Group (USG) and the Data Management and Operations
(DMO) division who participated in the first two feedback exercises, providing detailed and constructive evalu-
ations that directly shaped the development of ESOFinder. We also thank the ESO staff, fellows, and students
who are contributing to the ongoing third evaluation round. Their collective effort in testing the system and
providing feedback has been invaluable in improving the quality and reliability of the tool.
The authors acknowledge the use of Claude (Anthropic) as a language editing tool during the preparation of
this manuscript.
REFERENCES
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin,
I., “Attention is all you need,” in [Advances in Neural Information Processing Systems], Guyon, I., Luxburg,
U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R., eds.,30, Curran Associates,
Inc. (2017).
[2] OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Leoni Aleman, F., et al., “GPT-4
Technical Report,”arXiv e-prints, arXiv:2303.08774 (Mar. 2023).
[3] Gemini Team, Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., et al., “Gemini: A
Family of Highly Capable Multimodal Models,”arXiv e-prints, arXiv:2312.11805 (Dec. 2023).
[4] Anthropic, “The Claude 3 model family: Opus, Sonnet, Haiku,” tech. rep., Anthropic (2024).
[5] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler, H., Lewis, M., Yih, W.-t.,
Rockt¨ aschel, T., Riedel, S., and Kiela, D., “Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks,”arXiv e-prints, arXiv:2005.11401 (May 2020).
[6] Nussbaum, Z., Morris, J. X., Duderstadt, B., and Mulyar, A., “Nomic Embed: Training a Reproducible
Long Context Text Embedder,”arXiv e-prints, arXiv:2402.01613 (Feb. 2024).

[7] Robertson, S. E. and Walker, S., “Some simple effective approximations to the 2-poisson model for proba-
bilistic weighted retrieval,” in [SIGIR ’94], Croft, B. W. and van Rijsbergen, C. J., eds., 232–241, Springer
London, London (1994).
[8] Manning, C. D., Raghavan, P., and Sch¨ utze, H., [Introduction to Information Retrieval], Cambridge Uni-
versity Press, Cambridge, UK (2008).
[9] Carbonell, J. G. and Goldstein, J., “The use of mmr, diversity-based reranking for reordering documents
and producing summaries,” in [Proceedings of the 21st Annual International ACM SIGIR Conference on
Research and Development in Information Retrieval], 335–336, ACM (1998).
[10] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Hao Yu, C., Gonzalez, J. E., Zhang, H., and Stoica, I.,
“Efficient Memory Management for Large Language Model Serving with PagedAttention,”arXiv e-prints,
arXiv:2309.06180 (Sept. 2023).