# TechGraphRAG: An Agentic Graph-Augmented RAG Framework for Technical Literature Reasoning

**Authors**: Kanwar Bharat Singh

**Published**: 2026-06-01 03:01:51

**PDF URL**: [https://arxiv.org/pdf/2606.01613v1](https://arxiv.org/pdf/2606.01613v1)

## Abstract
This paper presents an agentic retrieval-augmented generation (RAG) framework for domain-specific technical reasoning support, instantiated over a curated corpus of approximately 2,100 academic papers in intelligent tires, vehicle dynamics, and vehicle control. Unlike conventional single-pass RAG systems, the proposed architecture employs a 13-step autonomous pipeline that classifies queries by intent, scores evidence sufficiency against a multi-dimensional rubric, performs agentic retry with drift-guarded query reformulation, searches external academic databases (Crossref, OpenAlex, Semantic Scholar) through iterative optimize--search--vet loops, traverses a Neo4j knowledge graph for relational context, verifies citation integrity, and applies post-generation quality checks with automatic regeneration. Key contributions include a 100-point evidence sufficiency scoring framework across five dimensions with relevance damping and hybrid rule-based/LLM review; a route-dependent external search architecture with iterative agentic loops; a knowledge graph constructed via LLM-based entity extraction and OpenAlex author validation with intra-corpus citation resolution; and a self-correcting generation loop with citation verification and quality assessment. The framework is presented as a practical, implemented case study illustrating how agentic, evidence-grounded RAG can support literature navigation and technical reasoning over large, domain-specific corpora.

## Full Text


<!-- PDF content starts -->

TechGraphRAG: An Agentic Graph-Augmented RAG
Framework for Technical Literature Reasoning
Kanwar Bharat Singh
Global Tire Intelligence and Solutions (GTIS)
The Goodyear Tire & Rubber Company
Abstract
This paper presents an agentic retrieval-augmented generation (RAG) framework
for domain-specific technical reasoning support, instantiated over a curated corpus
of approximately 2,100 academic papers in intelligent tires, vehicle dynamics,
and vehicle control. Unlike conventional single-pass RAG systems, the proposed
architecture employs a 13-step autonomous pipeline that classifies queries by intent,
scores evidence sufficiency against a multi-dimensional rubric, performs agentic
retry with drift-guarded query reformulation, searches external academic databases
(Crossref, OpenAlex, Semantic Scholar) through iterative optimize–search–vet
loops, traverses a Neo4j knowledge graph for relational context, verifies citation
integrity, and applies post-generation quality checks with automatic regeneration.
Key contributions include a 100-point evidence sufficiency scoring framework
across five dimensions with relevance damping and hybrid rule-based/LLM re-
view; a route-dependent external search architecture with iterative agentic loops;
a knowledge graph constructed via LLM-based entity extraction and OpenAlex
author validation with intra-corpus citation resolution; and a self-correcting gen-
eration loop with citation verification and quality assessment. The framework is
presented as a practical, implemented case study illustrating how agentic, evidence-
grounded RAG can support literature navigation and technical reasoning over large,
domain-specific corpora.
1 Introduction
Recent advances in large language models (LLMs), enabled by the Transformer architecture Vaswani
et al. [2017] and scaled through models such as GPT-3 Brown et al. [2020], have demonstrated
significant potential to accelerate knowledge discovery, synthesis, and reasoning across a wide range
of scientific and engineering domains. Techniques such as chain-of-thought prompting Wei et al.
[2022], Kojima et al. [2022] have further enhanced the reasoning capabilities of these models. By
enabling natural-language interaction with complex information, LLMs offer new opportunities to
streamline research workflows and support technical decision-making. However, when applied to
highly specialized technical fields, general-purpose LLMs exhibit fundamental limitations related to
domain specificity, knowledge freshness, context scalability, and traceability.
Over the past fifteen years, the author has been actively engaged in research and applied development
across domains including intelligent tire systems, vehicle state estimation, vehicle dynamics, and
vehicle control. This sustained involvement has necessitated continuous interaction with a broad and
evolving body of specialized technical literature. As a natural outcome of this long-term research
activity, the author has accumulated and curated a personal technical corpus comprising more than
2,100 academic papers, yielding approximately 24,000 indexed text chunks. The collection spans
topics such as tire–road interaction modeling, intelligent tire sensing, vehicle and tire state estimation,
sensor fusion, and advanced vehicle control algorithms, and has served as a foundational reference
base supporting ongoing research and development efforts.
Preprint.arXiv:2606.01613v1  [cs.IR]  1 Jun 2026

While this curated collection represents a rich and authoritative knowledge base, its growing scale and
interdisciplinary nature introduce significant practical challenges for effective utilization. Traditional
keyword-based search tools and manual literature review processes become progressively less effec-
tive as corpus size and technical diversity increase. Relevant prior art, methodological insights, and
cross-domain relationships are often obscured by inconsistent terminology, fragmented publication
venues, and the cognitive burden associated with navigating large volumes of dense technical material.
As a result, valuable knowledge may remain underutilized, despite being present within the available
literature.
The target users of such a system are engineers, researchers, and technical leaders working in
research-heavy product development who need to connect technical literature and emerging research
to generate evidence-backed insights, clarify the state of the art, drive design decisions, and support
technical discussions. For these users, the ability to rapidly navigate a large body of literature with
confidence in source attribution is essential.
The motivation of this work is therefore to develop a scalable, domain-specific technical knowledge
engine capable of:
• Systematically organizing and indexing large volumes of specialized literature
• Enabling semantic, meaning-based retrieval beyond conventional keyword search
•Rapidly surfacing relevant prior art, methods, and insights across adjacent technical domains
•Autonomously evaluating evidence sufficiency and dynamically supplementing from external
academic sources when internal evidence is insufficient
•Maintaining structured knowledge representations through a graph database capturing
relationships between papers, authors, methods, topics, and citations
•Supporting research, system design, and technical decision-making in intelligent tire and
vehicle dynamics applications
To address these needs, this paper presents an agentic retrieval-augmented generation (RAG) frame-
work tailored for domain-specific technical reasoning support. Drawing on advances in agentic LLM
architectures that interleave reasoning with action Yao et al. [2023] and learn from prior attempts
through self-reflection Shinn et al. [2023], the proposed system goes beyond conventional single-
pass RAG to employ a multi-step agentic pipeline comprising 13 orchestrated stages. The pipeline
autonomously classifies queries by intent, scores evidence sufficiency against a multi-dimensional
rubric, performs agentic retry with drift-guarded query reformulation when local evidence is weak,
searches external academic databases (Crossref, OpenAlex, Semantic Scholar) through iterative
optimize–search–vet loops, traverses a Neo4j knowledge graph for relational context, verifies citation
integrity, and applies post-generation quality checks with automatic regeneration. By grounding
model outputs in explicitly retrieved and vetted evidence from both internal and external sources, the
framework enables more reliable, transparent, and verifiable technical analysis.
The central contribution of TechGraphRAG is not a new retrieval model, but an evidence-gated
agentic architecture that coordinates hybrid retrieval, graph traversal, external academic search,
citation verification, and answer quality control into a reproducible workflow for domain-specific
technical literature reasoning. This paper makes the following contributions:
•An autonomous query-time, 13-step agentic RAG pipeline that goes beyond single-pass
retrieval to include query classification, evidence sufficiency scoring, agentic retry, multi-
source external search, knowledge graph enrichment, citation verification, and quality-
checked generation.
•A multi-dimensional evidence sufficiency scoring framework (100-point rubric across five
dimensions) with relevance damping and hybrid rule-based/LLM review, enabling the
system to autonomously decide whether local evidence is sufficient or external academic
sources are needed.
•An agentic external search architecture with iterative optimize–search–vet–reformulate
loops across Crossref, OpenAlex, and Semantic Scholar, with route-dependent strategies for
content, bibliometric, trend, and current-world queries.
2

•A Neo4j knowledge graph constructed via LLM-based entity extraction and OpenAlex author
validation, capturing paper–author–topic–method–citation relationships with intra-corpus
citation resolution, enabling co-citation traversal and structured relational retrieval.
•A practical architecture for controlled yet autonomous LLM-assisted technical reasoning,
with built-in citation verification, answer quality assessment, and automatic regeneration,
enabling grounded analysis over authoritative sources while preserving transparency and
auditability.
The remainder of this paper is organized as follows. Section 2 surveys related work. Section 3
discusses the fundamental limitations of standalone large language models in domain-specific settings.
Section 4 motivates the adoption of retrieval-augmented generation. Sections 5 and 6 describe the
proposed agentic system architecture and implementation considerations, respectively. Section 7
presents evaluation results. Section 8 discusses limitations, followed by concluding remarks in
Section 9.
2 Related Work
Retrieval-Augmented Generation.Lewis et al. Lewis et al. [2020] introduced RAG by com-
bining a pre-trained retriever with a seq2seq generator, demonstrating that grounding generation
in retrieved passages improves factuality on knowledge-intensive tasks. Dense Passage Retrieval
(DPR) Karpukhin et al. [2020] showed that learned bi-encoder representations substantially out-
perform BM25 for open-domain question answering. Subsequent surveys Gao et al. [2024] have
catalogued the rapid evolution of RAG architectures along the dimensions of retrieval, augmentation,
and generation. TechGraphRAG builds on this foundation but differs in two respects: it fuses dense
and sparse retrieval via Reciprocal Rank Fusion Cormack et al. [2009] with cross-encoder rerank-
ing Nogueira and Cho [2019], and it augments the retriever with a structured knowledge graph and
external academic database search rather than relying on a single retrieval pass.
Self-Correcting and Adaptive RAG.Self-RAG Asai et al. [2024] trains a single LM to emit
special reflection tokens that decide when to retrieve, whether retrieved passages are relevant, and
whether the generated output is supported—enabling on-demand retrieval without a separate scoring
module. Corrective RAG (CRAG) Yan et al. [2024] introduces a lightweight retrieval evaluator that
classifies documents as correct, ambiguous, or incorrect, and triggers web search when confidence is
low. REPLUG Shi et al. [2024] treats the retriever as a plug-in to a frozen black-box LM and tunes
retrieval via the LM’s perplexity signal. TechGraphRAG shares the adaptive retrieval philosophy of
these systems but implements it through an explicit evidence sufficiency scoring rubric (100-point,
five-dimension) with rule-based scoring, relevance damping, and LLM-based review, rather than
through learned reflection tokens or perplexity-based signals. This design choice trades end-to-end
learnability for interpretability and auditability, which are priorities in engineering domains where
users need to understand why the system sought additional evidence.
Agentic and Multi-Step Reasoning.ReAct Yao et al. [2023] demonstrated that interleaving
reasoning traces with tool-use actions enables LLMs to solve multi-step tasks more reliably than
reasoning or acting alone. Reflexion Shinn et al. [2023] extended this idea by adding verbal self-
reflection, where an agent critiques its own failures and carries forward lessons to subsequent attempts.
IRCoT Trivedi et al. [2023] interleaves retrieval with chain-of-thought reasoning steps for multi-hop
questions. TechGraphRAG draws on both ReAct-style action sequencing and Reflexion-style retry
(the agentic retry in Step 4 uses retrieval shortcomings to guide query reformulation), but wraps these
mechanisms in a fixed 13-stage pipeline with explicit drift guards and accept/reject gates rather than
an open-ended agent loop. This bounded design ensures predictable cost, latency, and auditability per
query.
Graph-Enhanced Retrieval.GraphRAG Edge et al. [2024] constructs a hierarchical community
graph from source documents and uses LLM-generated community summaries to answer global,
corpus-level queries that standard vector retrieval handles poorly. TechGraphRAG takes a comple-
mentary approach: rather than summarizing communities, it builds a typed entity graph (papers,
authors, topics, methods, metrics, applications) with LLM-based extraction, OpenAlex author vali-
dation, and intra-corpus citation resolution. At query time, the graph provides co-citation traversal,
3

Table 1: Architectural comparison with related systems.
Feature Standard Self-RAG CRAG GraphRAG TechGraphRAG
RAG (Ours)
Hybrid dense+sparse retrieval No No No No Yes
Cross-encoder reranking No No No No Yes
Explicit evidence scoring No Partial∗Partial∗No Yes
Agentic retry with drift guard No No No No Yes
External academic search No No Web No Yes
Knowledge graph traversal No No No Yes Yes
Citation verification No No No No Yes
Post-generation quality check No Yes No No Yes
∗Self-RAG uses learned reflection tokens; CRAG uses a retrieval evaluator. Both differ from TechGraphRAG’s
explicit multi-dimensional rubric with interpretable scores.
author lookup, and relational context that enriches—but does not replace—the hybrid vector/keyword
retrieval results.
Positioning.Table 1 summarizes the key architectural differences. TechGraphRAG is distinguished
not by any single retrieval innovation but by the integration of evidence-gated control flow, multi-
source retrieval (local hybrid, external academic APIs, knowledge graph), citation verification,
and self-correcting generation into a reproducible, auditable pipeline for domain-specific technical
literature reasoning.
3 Limitations of General-Purpose Standalone Large Language Models in
Domain-Specific Settings
Despite their strong language generation and reasoning capabilities, standalone/pre-trained LLMs are
not well-suited for domain-critical engineering applications without augmentation Gao et al. [2024].
Key limitations include the following.
3.1 Domain Knowledge Gaps
Pre-trained LLMs are trained on broad, general-purpose datasets and often lack deep coverage of
niche, proprietary, or highly specialized technical domains. As a result, their responses may be
incomplete, overly generic, or misaligned with domain-specific assumptions and constraints common
in tire mechanics and vehicle dynamics.
3.2 Knowledge Staleness
LLMs are trained on static datasets with a fixed knowledge cutoff. Incorporating newly published
research, internal reports, or evolving best practices requires retraining or fine-tuning, which is
computationally expensive and operationally complex. This makes standalone LLMs ill-suited for
fast-moving research environments.
3.3 Limited Context Window
Large language models can process only a finite number of input tokens per inference, which
fundamentally constrains their ability to reason over large documents or extensive collections of
technical material. This limitation is intrinsic to transformer-based architectures Vaswani et al. [2017],
where both computational cost and attention complexity scale with context length.
Even with extended-context models (e.g., on the order of 32k tokens), the usable input typically
corresponds to only 15–20 pages of dense technical text. This is insufficient for many real-world
engineering and research workflows that require simultaneous reasoning across dozens of papers,
multi-chapter reports, appendices, or large patent portfolios.
Moreover, increasing context length does not linearly translate to improved reasoning performance.
As the input grows, attention mechanisms exhibit degradation in effective recall and focus, particularly
4

for information appearing earlier in the prompt. Consequently, simply “feeding more text” into a
standalone LLM is neither scalable nor reliable for domain-critical technical analysis.
3.4 Lack of Source Attribution and Traceability
A critical limitation of standalone large language models is their inability to reliably provide explicit
source attribution for the information they generate. LLM outputs are produced by synthesizing
patterns learned during training, rather than by retrieving and citing specific documents, sections, or
evidence. As a result, generated responses are typically uncited, non-verifiable, and difficult to trace
back to authoritative sources.
This limitation is particularly problematic in research and development settings, where technical
claims must be supported by verifiable references. Without clear provenance, it becomes challenging
to assess the validity of assumptions, reproduce results, compare methodologies, or build confi-
dence in engineering decisions derived from LLM-generated content. Moreover, the absence of
source grounding increases the risk of hallucinated or conflated information. In safety-relevant or
domain-critical engineering workflows, such behavior is unacceptable, as it undermines transparency,
auditability, and accountability.
4 Motivation for Retrieval-Augmented Generation
Collectively, the limitations of standalone large language models outlined in the preceding section
motivate retrieval-augmented generation (RAG) approaches Gao et al. [2024], which decouple
knowledge storage from reasoning and enable LLMs to operate over large, evolving technical corpora
in a controlled, transparent, and traceable manner. The present use case is particularly well suited for
a RAG architecture due to the nature, scale, and technical value of the underlying knowledge corpus.
The corpus consists of approximately 2,100 highly specialized academic papers, technical reports,
and research documents focused on intelligent tires, smart tire sensing, tire–road interaction, vehicle
dynamics, and vehicle control systems. This domain is characterized by dense technical content,
mathematical formulations, domain-specific terminology, and a strong dependence on modeling
assumptions, operating conditions, and experimental regimes.
Traditional keyword-based search is insufficient in this context for several reasons:
•Relevant concepts are frequently expressed using different terminology across authors,
disciplines, and time periods
•Critical insights are often embedded within explanatory or methodological text, rather than
appearing as explicit keywords
•Cross-disciplinary relationships—such as the influence of tire mechanics on ABS or ESC
control behavior—are difficult to uncover through lexical matching alone
At the same time, pure LLM-based approaches without retrieval are inadequate for research-grade
and safety-relevant technical workflows, as they:
• Lack direct access to the full domain-specific corpus
• Risk hallucinating or oversimplifying technical details
• Provide limited traceability to original sources, prior art, and supporting evidence
Retrieval-augmented generation directly addresses these limitations by:
• Enabling semantic, meaning-based retrieval across the entire corpus
• Grounding generated responses in retrieved, authoritative source documents
• Preserving transparency by allowing inspection and verification of supporting text
• Scaling efficiently to thousands of documents through dense vector indexing
However, basic single-pass RAG systems—which retrieve a fixed set of chunks and pass them directly
to a language model—remain limited in their ability to handle the diversity of query types encountered
in research workflows. In practice, the queries that researchers and engineers pose fall into distinct
categories, each requiring different retrieval strategies:
5

•Technical research: What methods exist for a given problem (e.g., friction estimation,
aquaplaning prediction, camera-based road condition sensing)?
•Trend discovery: Who is publishing on a topic, what subjects are emerging, and how is the
research landscape evolving over time?
•Literature synthesis: What is the state of the art, how do methods compare, and where are
the gaps?
•Research recall: Which papers in the corpus previously discussed a specific concept or
method?
A content question about tire modeling requires different retrieval strategies than a bibliometric
query about publication trends or an author’s body of work. Furthermore, a single retrieval pass may
produce insufficient or weakly relevant evidence, particularly for niche or cross-domain queries, with
no mechanism to detect or recover from such failures.
These observations motivate the evolution from basic RAG to an agentic RAG architecture. Recent
work on agentic LLM frameworks has demonstrated that interleaving reasoning with actions such as
tool use and search Yao et al. [2023] and incorporating self-reflection to learn from prior failures Shinn
et al. [2023] substantially improve task performance. Similarly, graph-based retrieval approaches Edge
et al. [2024] have shown that structured knowledge representations can answer broad, corpus-level
questions more effectively than standard vector retrieval. Drawing on these advances, the proposed
system autonomously classifies query intent, evaluates evidence quality, performs iterative retrieval
with drift-guarded reformulation, searches external academic databases when needed, leverages
structured knowledge graph relationships, and applies post-generation quality controls. The following
section presents this agentic architecture in detail.
5 System Architecture and Methodology
The proposed agentic RAG system is a modular, research-grade technical knowledge engine whose
architecture is domain-agnostic, while its instantiation in this work is tailored to large-scale, domain-
specific literature in intelligent tires, vehicle state estimation, vehicle dynamics, and vehicle control.
The system is explicitly designed to address challenges related to semantic retrieval, context scalability,
evidence sufficiency, and reasoning faithfulness in safety-relevant engineering domains.
The pipeline comprises 13 orchestrated stages that execute autonomously for each user query,
organized into two agentic loops (Figure 1).Loop 1 (Evidence Gathering)encompasses query
classification, rewriting, local retrieval, evidence sufficiency scoring, agentic retry, external academic
search, LLM vetting, and knowledge graph traversal. Its objective is to assemble the strongest
possible evidence set from both internal and external sources.Loop 2 (Answer Curation and
Verification)takes the assembled evidence and performs prompt construction, citation verification,
answer generation, and quality assessment with optional automatic regeneration. This separation
ensures that evidence quality is established before synthesis begins, and that the synthesis itself is
independently verified.
All computationally intensive preprocessing steps—including document parsing, structure-aware
chunking, embedding generation, index construction, entity extraction, and knowledge graph
construction—are performed offline. Query-time execution spans the full 13-step pipeline.
5.1 Pipeline Overview
The 13 pipeline stages are summarized in Table 2.
5.2 Query Classification and Routing (Step 0)
Before retrieval begins, GPT-4o-mini classifies each incoming query into one or more retrieval routes.
This classification determines which external search strategy is used in Step 5 and influences the
number of agentic retry attempts and whether the knowledge graph performs author lookups. The
four routes are:
•Content: Technical/engineering questions about how or what. Uses Crossref + OpenAlex +
Semantic Scholar (3 attempts).
6

Loop 1: Evidence Gathering
Step 0 Query Classification →Step 1 Query Rewrite →Step 2 Local Retrieval
(FAISS+BM25+Rerank) →Step 3 Evidence Sufficiency Scoring →Step 4 Agentic Retry
(if WEAK) →Step 5 External Search (Crossref+OpenAlex+SS) →Step 6 LLM Vetting →Step 7
Knowledge Graph Traversal
Loop 2: Answer Curation & Verification
Step 8 Build Augmented Prompt →Step 9 Citation Verification →Step 10 Generate Answer →
Step 11Quality Check (auto-regenerate if needed)→Step 12Complete
Figure 1: TechGraphRAG pipeline architecture: 13 stages organized into two agentic loops. Loop 1
gathers and scores evidence from local, external, and graph sources. Loop 2 curates, verifies, and
generates the final answer.
Table 2: Agentic RAG pipeline: 13 orchestrated stages.
Step Name Description LLM?
0 Query Classification GPT-4o-mini classifies query route: content, bibliometric, trend,
or current_worldYes (tiny)
1 Query Rewrite GPT-4o-mini rewrites the user’s question into optimized retrieval
termsYes (tiny)
2 Local Retrieval FAISS (semantic) + BM25 (keyword) search with cross-encoder
rerankingNo
3 Evidence Sufficiency 100-point rubric + LLM reviewer scores how well local evidence
answers the questionYes
4 Agentic Retry If evidence is WEAK: reformulate query, drift guard, re-retrieve,
second hybrid checkConditional
5 External Search Route-dependent: Crossref + OpenAlex + SS (3 attempts) or
OpenAlex agentic loop (5 attempts)Yes
6 LLM Vetting Display vetting summary—how many external papers passed rele-
vance check—
7 Knowledge Graph Neo4j traversal: related papers, co-citing papers, author lookup No
8 Build Prompt Merge local chunks + external abstracts + graph context into
structured promptNo
9 Citation Verification GPT-4o-mini checks for gaps or contradictions in sources Yes (tiny)
10 Generate Answer Selected model produces the final answer Yes
11 Quality Check GPT-4o-mini assesses answer quality; auto-regenerates once if
issues foundYes
12 Complete Final cost summary and pipeline status No
•Bibliometric: Queries seeking papers on a topic, latest publications, or author searches.
Uses OpenAlex agentic loop (5 attempts).
•Trend: Queries about research evolution or publication volume over time. Uses OpenAlex
trend aggregation (group_by=publication_year).
•Current_world(auxiliary): Company strategies, market data, regulations, industry news.
Uses web search, bypassing academic APIs. Although the primary focus of this work is
technical literature reasoning, this auxiliary route is included to handle queries that fall
outside the academic corpus.
A query can have multiple routes (e.g., “What recent work builds on X and how do they improve it?”
is both bibliometric and content). The classifier returns a primary route and a list of all applicable
routes. Local retrieval (FAISS + BM25) always runs regardless of route, except for pure current_world
queries which skip the local retrieval steps entirely as there is no academic corpus value.
5.3 Structure-Aware Chunking (Offline)
Section-Aware Chunking.Each document is segmented based on detected structural markers,
including explicit section headers (e.g., Abstract, Introduction, Methods, Results), numbered headings,
and common scholarly section conventions. Chunk boundaries are constrained to lie entirely within
7

individual sections; chunks are never permitted to span across section boundaries. This design
enforces conceptual separation between fundamentally different content types—such as background
theory, methodology, and experimental results—and significantly reduces semantic dilution during
retrieval.
Paragraph-Aware Chunking.Within each section, text is further segmented into complete para-
graphs using layout-based cues and newline heuristics. Chunk construction adheres to a strict
integrity rule: only whole paragraphs are aggregated, and sentences or paragraphs are never split.
This preserves semantic coherence and ensures that definitions, modeling assumptions, equations,
and explanatory context remain intact within a single retrievable unit.
Chunk Sizing and Overlap.Chunks are constructed using empirically selected parameters to
balance semantic completeness and retrieval efficiency. The target chunk size is approximately 3,500
characters, with a minimum of approximately 300 characters and a maximum of approximately 6,000
characters. A conservative overlap of approximately 400 characters is applied to maintain continuity
across adjacent chunks. Each resulting chunk represents a self-contained semantic unit aligned with
the document’s natural structure and intent.
Implementation.Document parsing and text extraction are performed using the PyMuPDF li-
brary Artifex Software [2024], which provides reliable page-level access to structured text from
academic PDFs. Section and paragraph boundaries are identified using rule-based heuristics derived
from detected headers and layout cues. For each chunk, metadata including document identifier,
section label, page range, and chunk unique identifier is retained to support traceability, citation, and
linkage to the knowledge graph.
5.4 Embedding and Lexical Representations (Offline)
Dense Semantic Embeddings.Each document chunk is embedded using
sentence-transformers/all-MiniLM-L6-v2 Reimers and Gurevych [2019], producing
a 384-dimensional dense vector capturing overall semantic meaning. The compact embedding
dimensionality enables efficient indexing and low-latency similarity search on standard workstation
hardware. At query time, the (optionally rewritten) user query is embedded using the same model to
ensure compatibility within the shared embedding space.
Sparse Lexical Representation (BM25).In parallel, each chunk is represented using a sparse
lexical encoding via BM25 Robertson and Zaragoza [2009] to support exact keyword-based retrieval.
Chunk text is tokenized using a domain-aware normalization pipeline that preserves alphanumeric
tokens, acronyms, and technical symbols commonly used in engineering literature (e.g., ABS, µ,
RLS, slip). Unlike dense embeddings, BM25 computes relevance scores based on term frequency,
inverse document frequency, and document length normalization, making it particularly effective for
retrieving chunks containing exact terminology, acronyms, and formula-driven language.
5.5 Query Understanding and LLM-Based Query Rewriting (Step 1)
Engineering queries posed by users are often informal, abbreviated, or underspecified relative to
the language used in academic literature. Prior to retrieval, GPT-4o-mini generates a semantically
equivalent but technically explicit rewritten query. The objectives of this step are to expand abbrevi-
ated or implicit technical terminology, normalize phrasing toward literature-style language, preserve
the original user intent without introducing new assumptions, and improve alignment between the
query and both dense and lexical retrieval mechanisms. The rewritten query is used for retrieval; the
original user query is retained for transparency.
5.6 Hybrid Retrieval (Step 2)
To improve recall and robustness across heterogeneous technical writing styles, the system employs a
hybrid retrieval strategy combining dense vector similarity search via FAISS Johnson et al. [2021]
with lexical keyword-based retrieval via BM25.
8

Given an embedded query, FAISS returns a ranked list of candidate chunks based on cosine sim-
ilarity (inner product over L2-normalized embeddings). In parallel, BM25 produces a ranked list
based on lexical relevance scores. The two candidate lists are fused using Reciprocal Rank Fusion
(RRF) Cormack et al. [2009], which computes a hybrid base score as a function of rank position
in each retrieval list. This fusion strategy balances semantic relevance and lexical precision while
remaining insensitive to score scale differences across modalities.
Cross-Encoder Reranking.Following RRF fusion, a transformer-based cross-encoder model
(cross-encoder/ms-marco-MiniLM-L-6-v2 ) Nogueira and Cho [2019] jointly encodes each
query–chunk pair and directly predicts a relevance score. Unlike bi-encoder embeddings, the cross-
encoder attends jointly over both inputs, enabling fine-grained matching of technical phrasing,
assumptions, and methodological detail. This stage significantly improves precision at top ranks,
particularly for queries requiring close alignment between problem formulation and methodological
content.
5.7 Evidence Sufficiency Scoring (Step 3)
A distinguishing feature of the proposed agentic architecture is the autonomous assessment of whether
retrieved local evidence is sufficient to answer the user’s question, or whether external academic
sources are needed. The system scores retrieved evidence across five dimensions using a 100-point
rubric.
Scoring Dimensions.
•Retrieval Confidence(40 points): Cross-encoder reranker scores—are the top chunks
actually relevant?
•Answer Specificity(25 points): Do chunks contain specific data, methods, equations,
results?
•Source Diversity(15 points): How many distinct papers contribute relevant evidence?
•Metadata Completeness(10 points): Do chunks have section labels, page numbers, ex-
tractable year?
•Recency / Intent Fit(10 points): Are sources recent enough given the query type?
Decision Thresholds.
• 80–100 (STRONG): Answer from internal corpus only. External search is skipped.
•50–79 (MODERATE): Answer from internal corpus, optionally enrich with external
sources.
•0–49 (WEAK): Trigger agentic retry (Step 4) followed by external academic search (Step 5).
Relevance Damping.When retrieval confidence is low (irrelevant chunks retrieved), the other four
dimensions are scaled down proportionally using a damping factor:
damping= max
minretrieval_score
25,1.0
,0.2
(1)
This prevents metadata or recency scores of off-topic chunks from inflating the overall score.
LLM Reviewer (Hybrid Check).After rule-based scoring, GPT-4o-mini reviews evidence snippets
and can downgrade (never upgrade) the verdict if the chunks do not actually address the question.
This hybrid approach combines the efficiency of rule-based scoring with the semantic judgment of an
LLM.
5.8 Agentic Retry (Step 4)
When evidence is WEAK, the system attempts a smarter search before resorting to external sources.
This retry mechanism draws on the principle of verbal self-reflection Shinn et al. [2023], where the
system uses the shortcomings of the initial retrieval to guide reformulation:
9

•Query Reformulation: GPT-4o-mini generates alternative search terms based on what
concepts are missing from the initial retrieval.
•Drift Guard: A rule-based check ensures reformulated results still relate to the original
question ( ≥30% term overlap required). If drift is detected, the retry is rejected without a
second LLM call.
•Second Hybrid Check: If drift is acceptable, the full hybrid evidence check (rule-based
+ LLM reviewer) runs on the new results. This is the “double LLM safety net”—the first
hybrid check ran on original results (Step 3), and this second one evaluates the retry results.
•Accept/Reject: Retry results are used only if the new score exceeds the original. This is a
safety net—if it does not help, the system keeps original results and proceeds to external
search.
5.9 External Academic Search (Step 5)
External search behavior depends on the query route determined in Step 0. For content routes,
external search runs when evidence is MODERATE or WEAK (score <80 ). For bibliometric and
trend routes, external search always runs, as the primary purpose is finding external papers or trends.
External Sources.
•CrossrefCrossref [2024] (PRIMARY): Reliable journal-article search with no rate limits
and good abstracts. Always queried first on the content route.
•OpenAlexPriem et al. [2022] (SECOND): Large coverage ( ∼250 M works), free, no rate
limits, relevance-sorted. Has built-in progressive broadening (retries without year filter,
shortens queries automatically). Primary source for bibliometric/trend routes.
•Semantic ScholarAllen Institute for AI [2024] (BONUS): Only queried when an API key
is configured. Better neural relevance ranking but aggressive rate limiting without a paid
key. SupportspublicationTypes,fieldsOfStudy, andyearfilters.
Content Route (up to 3 attempts).All three sources are queried on every attempt (Semantic
Scholar only if the API key exists). Results are merged and deduplicated by title similarity. Each
attempt follows the cycle: (1) optimize query via GPT-4o-mini, (2) search Crossref + OpenAlex +
Semantic Scholar, (3) LLM vet to select only directly relevant papers, (4) check threshold ( ≥2vetted
papers = success), (5) if below threshold and not the last attempt, reformulate query and retry. The
best result across all attempts is kept.
Bibliometric/Trend Route (up to 5 attempts).Uses OpenAlex only with the same optimize–
search–vet–reformulate pattern, but with more attempts. Works search employs progressive broad-
ening (year filter →no year filter →shortened query). Trend aggregation uses a single OpenAlex
group_by=publication_year call, independent of the retry loop. Author search retrieves Ope-
nAlex author profiles and top works by citation count.
5.10 Knowledge Graph (Step 7)
The system maintains a Neo4j Neo4j, Inc. [2024] knowledge graph built from entity extraction
over the local PDF corpus. Inspired by graph-based retrieval approaches that leverage structured
knowledge representations to answer broad, corpus-level questions more effectively than standard
vector RAG Edge et al. [2024], Step 7 traverses this graph to enrich the augmented prompt with
structured relationships that vector/keyword search cannot capture.
Graph Schema.The graph contains eight node types: Paper ,Author ,Topic ,Method ,Metric ,
Application ,CitationPaper , and Chunk . Relationships include AUTHORED ,BELONGS_TO ,
USES_METHOD ,REPORTS_METRIC ,APPLIES_TO ,CITES (external citations), CITES_PAPER (re-
solved intra-corpus citations), and HAS_CHUNK (bridging to FAISS/BM25). Paper nodes carry
properties including doc_id ,title ,pdf_filename ,year ,author_keywords (as listed by the
paper’s authors), and extracted_keywords (additional specific terms identified by the LLM, stored
in lowercase for search and filtering).
10

Graph Construction Pipeline.
1.Chunking: PDFs are split into ∼3,500 -character chunks with section labels during indexing.
2.Priority section selection: The first chunk (header/title block containing title and authors)
is always included, followed by abstract, introduction, and conclusion chunks, up to a
6,000-character budget.
3.Entity extraction: GPT-4o-mini extracts title, authors, year, topics, methods, metrics,
applications, author_keywords, extracted_keywords, and up to 10 cited paper titles from the
excerpt (∼$0.0004/paper).
4.Author validation: LLM-extracted authors are cross-checked against OpenAlex publisher
metadata. If OpenAlex returns a confident title match, its author list replaces the LLM’s
extraction, preventing misattribution ( ∼10% of papers had incorrect LLM-extracted authors).
Falls back to LLM authors when OpenAlex has no match (∼40%of papers).
5.Graph ingestion: Extracted entities are written as nodes and relationships using idempotent
MERGEqueries.
6.Chunk linking: All chunk UIDs for a paper are linked via
(Paper)-[:HAS_CHUNK]->(Chunk), bridging the vector database to the graph.
7.Citation resolution: A post-ingestion step matches CitationPaper.title against
Paper.title (case-insensitive, with containment matching for slight title variations).
When a match is found, the CITES edge is rewired to a direct CITES_PAPER edge be-
tween the two Paper nodes, and the orphaned CitationPaper node is deleted ( ∼433
resolved intra-corpus links across ∼2,100 papers, with ∼9,100 remaining as truly external
citations).
Query-Time Graph Features.
•Related papers: Papers sharing topic/method overlap with retrieved chunks (always runs if
Neo4j is available).
•Co-citing papers: Papers that share references with retrieved chunks, using both CITES
(external) andCITES_PAPER(intra-corpus) edges (always runs if Neo4j is available).
•Title map: Maps PDF filenames to proper paper titles for display (always runs if Neo4j is
available).
•Author lookup: Finds all papers by a queried author in the graph (bibliometric queries
only).
Co-Citation Traversal.The co-citation query uses both relationship types to find papers that
cite the same works: (Paper)-[:CITES]->(CitationPaper)<-[:CITES]-(Paper) for shared
external citations, and (Paper)-[:CITES_PAPER]->(Paper)<-[:CITES_PAPER]-(Paper) for
shared intra-corpus citations.
Graph Contribution Beyond Vector Retrieval.To illustrate the value of the knowledge graph,
consider the query“What methods exist for real-time tire friction estimation?”FAISS and BM25
retrieve chunks that explicitly mention friction estimation methods (e.g., extended Kalman filters
applied to tire force models). However, several relevant papers in the corpus address the same problem
using different terminology—for instance, papers ontire–road interaction observersor µ-estimation
via wheel dynamics—and do not share sufficient lexical or embedding overlap to appear in the top- K
retrieval results. The knowledge graph surfaces these papers through two mechanisms: (i)topic
overlap—papers linked to the same Topic nodes (e.g., “friction estimation,” “tire–road interaction”)
are retrieved even when their chunk text differs; and (ii)co-citation traversal—papers that cite
the same foundational references (e.g., Pacejka’s tire models) are identified via shared CITES and
CITES_PAPERedges. In observed development queries, graph enrichment often surfaced additional
related papers that were not present in the top-20 vector/keyword retrieval results, providing the
answer model with broader methodological context.
11

5.11 Augmented Prompt Construction (Step 8)
The final prompt sent to the answer model is constructed by merging local chunks, external abstracts,
and graph context into a structured format. When local evidence is very weak (score <30 AND
verdict is WEAK), local chunks are dropped entirely from the prompt, and the answer is built from
external evidence only. This prevents low-quality local chunks from misleading the answer model.
The augmented prompt follows a structured template:
•System instruction establishing the role as a technical research assistant constrained to use
only the provided evidence
• The user’s original question
•EVIDENCEsection: condensed evidence from top retrieved PDFs with source metadata
(filename, section, pages)
•EXTERNAL EVIDENCEsection: vetted academic papers with title, authors, year, and
abstract
•TASKinstructions: structured engineering-focused response, source citation requirements
([SOURCE n] or[EXTERNAL n] ), distinction between internal and external claims, conflict
identification, and explicit gap acknowledgment
5.12 Citation Verification (Step 9)
Before answer generation, GPT-4o-mini performs a citation verification pass over the assembled
sources. This step checks for gaps in source coverage relative to the question, contradictions between
sources, and potential attribution issues. Any findings are incorporated into the prompt to guide the
answer model.
5.13 Answer Generation (Step 10)
The selected answer model (e.g., a lightweight model for standard queries or a stronger model
for premium queries) produces a structured natural-language response using only the constructed
augmented prompt. The LLM functions as a constrained synthesis component: it is not permitted to
perform external search, tool invocation, or autonomous retrieval. Where the supplied evidence is
insufficient, ambiguous, or internally inconsistent, the model is instructed to explicitly state these
limitations.
5.14 Answer Quality Check (Step 11)
After generating an answer, GPT-4o-mini evaluates the response across multiple dimensions:
• Does it address the question?
• Are sources properly cited?
• Is it factually grounded (no hallucinations)?
• Are there obvious gaps?
If issues are found, the system automatically regenerates once with the critique appended to the
prompt. The user interface indicates whether regeneration occurred. This self-correcting behavior
improves answer quality without manual intervention.
5.15 Cost Tracking (Step 12)
All LLM calls throughout the pipeline are tracked with exact token counts (input and output) and
associated costs. The final step reports a complete cost summary. Typical per-query costs are shown
in Table 3.
12

Table 3: Typical per-query costs.
Scenario Typical Cost
Strong evidence (local only)∼$0.0006(<0.1cent)
Full pipeline with external search (1 attempt)∼$0.003(∼0.3cents)
Full pipeline with external search (3 attempts)∼$0.005(∼0.5cents)
Premium model for answer∼$0.008(∼0.8cents)
6 Implementation Considerations
The proposed agentic RAG framework is implemented as a modular system designed to support both
autonomous operation and interactive expert workflows while preserving reproducibility, data control,
and inspection transparency. All document preprocessing steps—including PDF parsing, structure-
aware chunking, embedding generation, lexical index construction, FAISS index construction, entity
extraction, and knowledge graph construction—are performed offline. Query-time execution spans
the full 13-step pipeline but remains interactive, with real-time progress reporting.
6.1 User Interface
A lightweight Gradio-based user interface provides interactive access to the pipeline. The interface
exposes query input, model selection, and real-time pipeline progress with step-by-step status
indicators. The UI displays retrieval results, evidence sufficiency scores, external search vetting
summaries, knowledge graph context, the constructed augmented prompt, and the generated answer
with cost tracking. Gradio functions solely as a presentation and interaction layer; all retrieval and
reasoning logic is implemented within the local RAG engine.
6.2 Precision-Oriented Retrieval Design
A central design choice is the intentional prioritization of retrieval precision over recall. In RAG
systems, retrieved content is consumed directly by a language model as contextual input rather
than presented to a human for selective review. Retrieval errors—particularly weakly relevant,
redundant, or contradictory content—can disproportionately degrade downstream synthesis quality.
This is especially problematic in technical and safety-relevant domains, where multiple modeling
assumptions and experimental regimes may coexist under a shared topical label.
Accordingly, the framework adopts a precision-oriented retrieval strategy. Initial candidates are
obtained using complementary dense semantic similarity (FAISS) and sparse lexical matching
(BM25), merged via Reciprocal Rank Fusion, and refined through cross-encoder reranking. The
evidence sufficiency scoring mechanism (Section 5.7) further ensures that only when local evidence
is genuinely insufficient does the system expand its search to external sources.
6.3 Incremental Updates
When new papers are added to the corpus, an incremental indexing pipeline processes only the new
PDFs and appends to the existing FAISS and BM25 indices. A manifest file tracks what has been
indexed. Entity extraction and knowledge graph ingestion run incrementally as well, with idempotent
MERGEqueries ensuring that re-processing a paper does not create duplicate nodes or relationships.
6.4 PDF Processing Pipeline
The offline processing pipeline proceeds as follows: PDF files (sourced from Zotero and OneDrive
folders) undergo PyMuPDF text extraction, whitespace normalization, and section detection. Text
is grouped into paragraphs, built into ∼3,500 -character chunks, and written to JSONL. Chunks
are then embedded with MiniLM-L6-v2 and indexed in FAISS ( IndexFlatIP ) and BM25. PDFs
exceeding 40 pages are skipped. The current corpus comprises approximately 2,100 papers yielding
approximately 24,000 indexed chunks.
13

Table 4: Route-level evaluation results across 6 test queries spanning 3 route types.
Q Query (abridged) Route Ev. Str. Retry Ext. Cost Time
1 Friction estimation for AEB Content 74 Moderate No 4 $0.005 14.3s
2 Tire slip angle from accelerometer Content 68 Moderate Yes 5 $0.007 22.1s
3 Pacejka vs. brush model comparison Content 81 Strong No 3 $0.005 13.8s
4 Recent intelligent tire sensor papers Bibliom. – – No 11 $0.004 19.4s
5 Piezoelectric energy harvesting in tires Bibliom. – – No 9 $0.004 17.2s
6 EU tire labeling regulations Current – – No 0 $0.003 10.9s
Ev. = evidence sufficiency score (content route only). Str. = evidence strength verdict. Ext. = external papers
vetted and accepted. Cost = total LLM cost. Time = wall-clock time.
7 Results
The system was evaluated on representative technical queries drawn from day-to-day engineering and
research workflows in intelligent tires, vehicle dynamics, and vehicle control. Evaluation focused on
(i) query route classification accuracy, (ii) evidence sufficiency scoring behavior, (iii) effectiveness
of agentic retry, (iv) end-to-end answer quality with citation grounding, and (v) cost and latency.
Assessment was performed through a combination of automated pipeline metrics and expert-driven
qualitative review.
7.1 Route-Level Evaluation
A test set of six queries spanning three route types (content, bibliometric, and current_world) was
evaluated end-to-end through the full 13-step pipeline. Table 4 summarizes the per-query results.
Key Findings.
•Route classification: All 6 queries were routed correctly (6/6), indicating correct intent
classification across content, bibliometric, and current_world query types on this test set.
•Answer quality: All 6 answers passed the automated quality check (Step 11) without
requiring regeneration, suggesting adequate citation grounding and factual coherence on the
evaluated queries.
•Evidence sufficiency scoring: Content queries produced evidence scores of 68–81 (mean
74). The scoring correctly distinguished between queries with strong local coverage (Q3,
score 81, classified STRONG) and those with moderate coverage (Q1, score 74; Q2, score
68, both classified MODERATE). The retry mechanism fired on Q2 (the lowest-scoring
query) and improved coverage.
•Route-dependent behavior: Bibliometric queries correctly bypassed local retrieval and
evidence scoring, proceeding directly to external academic search via OpenAlex. The
current_world query correctly bypassed both local retrieval and academic search, using web
search instead.
•Cost efficiency: Total cost across all 6 queries was $0.027 (mean $0.0045/query). Content
queries with external search cost $0.005–$0.007; the current_world query with no academic
search cost $0.003.
•Latency: Wall-clock time ranged from 10.9s (current_world, minimal pipeline) to 22.1s
(content with retry and external search), with a mean of 16.4s.
7.2 Retrieval Ablation Study
To quantify the contribution of individual retrieval components, the system includes a programmatic
evaluation harness that supports ablation studies across six standard configurations. Table 5 reports
preliminary results on a development set of 10 queries with expert-labeled relevant chunks, evaluated
atK=5.
The preliminary results suggest that each retrieval component contributes positively: hybrid fusion
improves over either modality alone, cross-encoder reranking provides the largest single-component
14

Table 5: Retrieval ablation results (development set, 10 queries, K=5 ). Each row adds one component
to the previous configuration.
Configuration P@5 R@5 NDCG@5 MRR
BM25 only 0.52 0.38 0.49 0.61
FAISS only (dense) 0.58 0.44 0.55 0.68
Hybrid (FAISS + BM25, RRF) 0.66 0.52 0.63 0.76
Hybrid + cross-encoder reranking 0.74 0.56 0.71 0.83
Full pipeline (+ query rewrite + keyword boost) 0.78 0.60 0.76 0.87
P@5 = precision at 5; R@5 = recall at 5; MRR = mean reciprocal rank. Results are preliminary and based on a
small development set; expansion to a larger labeled query set is ongoing.
gain (+8 points P@5 over hybrid), and query rewriting with keyword boost provides an additional
incremental improvement. The harness additionally supports parameter sweeps for RRF k(10–200),
cross-encoder candidate pool size (20–200), and keyword boost weight (0.0–1.0). Comprehensive
ablation results across a larger labeled query set with statistical significance testing are the subject of
ongoing work.
7.3 Qualitative Case Study
To illustrate the pipeline’s end-to-end behavior, we present a representative content query in detail.
Original Query:How can you determine aquaplaning using vehicle CAN data?
Rewritten Query (Step 1):What methods can be used to determine aquaplaning conditions using
vehicle Controller Area Network data?
The pipeline classified this as acontentroute query (Step 0). Local retrieval (Step 2) produced
18 candidate chunks from the corpus. Evidence sufficiency scoring (Step 3) assessed the local
evidence asMODERATE(score in the 50–79 range), indicating partial but incomplete coverage.
The agentic retry (Step 4) did not fire as the score was above the WEAK threshold. External search
(Step 5) queried Crossref and OpenAlex, returning 4 vetted papers covering wheel-speed-based
detection, accelerometer-based approaches, vibration profile analysis, and force-deviation methods.
The knowledge graph (Step 7) surfaced co-citing papers sharing references with the retrieved local
evidence. The combined evidence set—local chunks, vetted external papers, and graph context—was
assembled into the augmented prompt (Step 8), and citation verification (Step 9) confirmed no gaps
or contradictions.
The quality check (Step 11) assessed the generated answer as satisfactory, and no regeneration was
required. Total pipeline cost for this query was approximately $0.003 with a wall-clock time of
approximately 14 seconds.
8 Limitations
Several limitations of the current work should be acknowledged.
Proprietary Corpus.The technical corpus used in this study is proprietary and cannot be publicly
released, which limits direct reproducibility. However, the complete pipeline—including prepro-
cessing, indexing, entity extraction, and retrieval—is documented in sufficient detail (Section 6 and
Appendix A) to enable faithful reproduction on equivalent domain corpora.
Small Evaluation Set.The route-level evaluation (Table 4) covers only six queries, and the retrieval
ablation (Table 5) uses a development set of ten queries. These sample sizes are insufficient to draw
statistically robust conclusions about system performance. Expanding to a larger labeled query set
(20–50 queries across all route types) with inter-annotator agreement is a priority for future work.
LLM-as-Judge.The automated quality check (Step 11) and evidence sufficiency LLM reviewer
(Step 3) use GPT-4o-mini as a judge. LLM-based evaluation is known to exhibit biases including
15

position bias, verbosity preference, and self-enhancement bias. No human evaluation or inter-rater
reliability analysis has been conducted to validate these automated assessments.
External API Dependence.The pipeline depends on cloud-based LLM APIs (OpenAI) for query
rewriting, evidence scoring, entity extraction, and answer generation, as well as on Crossref, Ope-
nAlex, and Semantic Scholar for external search. API availability, rate limits, cost changes, and
model deprecation represent operational risks that are outside the system’s control.
Text-Only Ingestion.The current pipeline processes only the text content of academic PDFs.
Figures, tables, charts, diagrams, and equations rendered as images are not extracted, indexed, or
retrieved. This is a meaningful limitation in engineering domains where technical content is frequently
conveyed through visual representations.
9 Conclusion
This paper presented an agentic retrieval-augmented generation framework for domain-specific
technical reasoning support, instantiated over a curated corpus of approximately 2,100 academic
papers in intelligent tires, vehicle dynamics, and vehicle control. The system goes significantly
beyond basic single-pass RAG through a 13-step autonomous pipeline that classifies queries by
intent, scores evidence sufficiency against a multi-dimensional rubric, performs agentic retry with
drift-guarded reformulation, searches external academic databases through iterative optimize–search–
vet loops, traverses a Neo4j knowledge graph for relational context, verifies citations, and applies
post-generation quality checks with automatic regeneration.
Key architectural contributions include the evidence sufficiency scoring framework, which au-
tonomously determines whether local evidence is sufficient or external sources are needed; the
route-dependent external search architecture spanning Crossref, OpenAlex, and Semantic Scholar;
the knowledge graph with LLM-based entity extraction, OpenAlex author validation, and intra-corpus
citation resolution; and the self-correcting generation loop with citation verification and quality
assessment.
Overall, the implemented system illustrates how an agentic, evidence-grounded RAG architecture can
serve as a practical foundation for literature navigation and technical reasoning support over large,
domain-specific corpora. The autonomous evidence evaluation and multi-source retrieval capabilities
are designed to reduce the manual effort required for literature review while maintaining transparency,
traceability, and engineering rigor. While the full technical corpus used in this study is proprietary,
the complete preprocessing, indexing, entity extraction, and retrieval pipeline has been documented
to enable faithful reproduction on equivalent domain corpora.
Several directions for future work are identified. First, the current text-only pipeline does not
exploit figures, tables, charts, and diagrams embedded in academic papers; extending the system
tomultimodal RAG—where visual elements are captioned, indexed, and retrieved alongside text—
would significantly enrich the evidence available for technical reasoning. Second, the current
architecture depends on cloud-based LLM APIs (OpenAI) for query rewriting, evidence scoring,
entity extraction, and answer generation; replacing these withlocally hosted open-weight models
(e.g., Llama, Mistral) would eliminate external API dependence, reduce per-query cost, and address
data privacy concerns for proprietary corpora. Third, incorporatinghuman-in-the-loop feedback—
allowing users to rate responses as useful, incorrect, or incomplete—would enable continuous
improvement of retrieval ranking, prompting strategies, and evidence sufficiency thresholds through
reinforcement from expert judgment. Finally, expanding the evaluation to a larger labeled query set
with comprehensive retrieval ablations (using the evaluation harness described in Section 7.2) would
provide stronger quantitative evidence for the contribution of each pipeline component.
References
Allen Institute for AI. Semantic scholar academic graph API. https://api.semanticscholar.
org/, 2024.
Artifex Software. PyMuPDF documentation.https://pymupdf.readthedocs.io/, 2024.
16

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection. InProc. Int. Conf. Learning Representations
(ICLR), 2024.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. InAdvances in Neural Information Processing Systems, volume 33, pages
1877–1901, 2020.
Gordon V . Cormack, Charles L. A. Clarke, and Stefan Buettcher. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. InProc. 32nd Int. ACM SIGIR Conf., pages
758–759, 2009. doi: 10.1145/1571941.1572114.
Crossref. Crossref REST API.https://api.crossref.org/, 2024.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A
graph RAG approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2024. doi: 10.48550/arXiv.2312.10997.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs.IEEE
Trans. Big Data, 7(3):535–547, 2021. doi: 10.1109/TBDATA.2019.2921572.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. InProc.
Conf. Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, 2020.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners. InAdvances in Neural Information Processing Systems,
volume 35, 2022.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems, volume 33, pages 9459–9474, 2020.
Neo4j, Inc. Neo4j graph database.https://neo4j.com/, 2024.
Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with BERT.arXiv preprint
arXiv:1901.04085, 2019.
Jason Priem, Heather Piwowar, and Richard Orr. OpenAlex: A fully-open index of scholarly works,
authors, venues, institutions, and concepts.arXiv preprint arXiv:2205.01833, 2022.
Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese BERT-
networks. InProc. Conf. Empirical Methods in Natural Language Processing (EMNLP), pages
3982–3992, 2019.
Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond.
Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/1500000019.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. REPLUG: Retrieval-augmented black-box language models.arXiv
preprint arXiv:2301.12652, 2024.
Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion:
Language agents with verbal reinforcement learning. InAdvances in Neural Information Processing
Systems, volume 36, pages 8634–8652, 2023.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval
with chain-of-thought reasoning for knowledge-intensive multi-step questions. InProc. Annual
Meeting of the Association for Computational Linguistics (ACL), pages 10014–10037, 2023.
17

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. InAdvances in Neural Information
Processing Systems, volume 30, 2017.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V .
Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In
Advances in Neural Information Processing Systems, volume 35, 2022.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation.
InProc. Int. Conf. Learning Representations (ICLR), 2024.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
ReAct: Synergizing reasoning and acting in language models. InProc. Int. Conf. Learning
Representations (ICLR), 2023.
A Reproducibility and Implementation Details
To support reproducibility and independent verification of the proposed agentic RAG framework, this
appendix documents the exact software environment, library versions, and indexing configuration
used. All components were executed in a fully local Python environment, and dependency versions
were captured at runtime usingpip freeze.
Core Libraries and Versions.Python 3.13.6; Windows 11 (build 10.0.22631, 64-bit).
Table 6: Core dependencies.
Library Version
faiss-cpu 1.13.2
sentence-transformers 5.2.0
transformers 4.57.3
torch 2.9.1
gradio 5.49.1
PyMuPDF (fitz) 1.26.7
neo4j (Python driver) —
openai 2.7.1
numpy 2.4.0
pandas 2.3.3
FAISS Indexing Configuration.Embedding model: sentence-transformers/all-MiniLM-L6-v2
(384 dimensions). Index type: IndexFlatIP (exact, non-approximate). Similarity metric: cosine
similarity via inner product (all embeddings L2-normalized). No approximate search structures (IVF,
HNSW, PQ) are employed. Top-Kretrieval is configurable at query time (typicalK= 10–25).
Deduplication and Filtering.Chunk-level deduplication via deterministic SHA-1 hash of nor-
malized text content. Reference-section exclusion via section-header matching. Each chunk retains
document identifier, section label, page range, andchunk_uidmetadata.
Knowledge Graph Configuration.Database: Neo4j (local instance). Eight node types: Paper ,
Author ,Topic ,Method ,Metric ,Application ,CitationPaper ,Chunk . Entity extraction model:
GPT-4o-mini ( ∼$0.0004 /paper). Author validation: OpenAlex API cross-check ( ∼60% of papers
validated). Citation resolution: ∼433 intra-corpus links resolved, ∼9,100 external citations retained.
18

Table 7: Technology stack summary.
Component Technology
Vector index FAISS (IndexFlatIP, cosine similarity)
Keyword index BM25Okapi (rank_bm25)
Embedding modelall-MiniLM-L6-v2(384d)
Cross-encoderms-marco-MiniLM-L-6-v2
Knowledge graph Neo4j
Entity extraction GPT-4o-mini + OpenAlex author validation
PDF extraction PyMuPDF (fitz)
LLM orchestration OpenAI API (gpt-4o-mini for tooling)
External search Crossref + OpenAlex + Semantic Scholar
UI Gradio
Language Python
19