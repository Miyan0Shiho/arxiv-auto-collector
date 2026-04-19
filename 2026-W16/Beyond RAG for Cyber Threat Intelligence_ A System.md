# Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval

**Authors**: Dzenan Hamzic, Florian Skopik, Max Landauer, Markus Wurzenberger, Andreas Rauber

**Published**: 2026-04-13 13:02:44

**PDF URL**: [https://arxiv.org/pdf/2604.11419v1](https://arxiv.org/pdf/2604.11419v1)

## Abstract
Cyber threat intelligence (CTI) analysts must answer complex questions over large collections of narrative security reports. Retrieval-augmented generation (RAG) systems help language models access external knowledge, but traditional vector retrieval often struggles with queries that require reasoning over relationships between entities such as threat actors, malware, and vulnerabilities. This limitation arises because relevant evidence is often distributed across multiple text fragments and documents. Knowledge graphs address this challenge by enabling structured multi-hop reasoning through explicit representations of entities and relationships. However, multiple retrieval paradigms, including graph-based, agentic, and hybrid approaches, have emerged with different assumptions and failure modes. It remains unclear how these approaches compare in realistic CTI settings and when graph grounding improves performance. We present a systematic evaluation of four RAG architectures for CTI analysis: standard vector retrieval, graph-based retrieval over a CTI knowledge graph, an agentic variant that repairs failed graph queries, and a hybrid approach combining graph queries with text retrieval. We evaluate these systems on 3,300 CTI question-answer pairs spanning factual lookups, multi-hop relational queries, analyst-style synthesis questions, and unanswerable cases. Results show that graph grounding improves performance on structured factual queries. The hybrid graph-text approach improves answer quality by up to 35 percent on multi-hop questions compared to vector RAG, while maintaining more reliable performance than graph-only systems.

## Full Text


<!-- PDF content starts -->

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of
Graph-Based and Agentic Retrieval
DŽENAN HAMZIĆ,AIT Austrian Institute of Technology, Austria
FLORIAN SKOPIK,AIT Austrian Institute of Technology, Austria
MAX LANDAUER,AIT Austrian Institute of Technology, Austria
MARKUS WURZENBERGER,AIT Austrian Institute of Technology, Austria
ANDREAS RAUBER,TU Wien, Austria
Cyber threat intelligence (CTI) analysts must answer complex questions over large collections of narrative security reports. While
retrieval-augmented generation (RAG) systems help language models access external knowledge, traditional vector-based retrieval
often struggles with queries requiring reasoning over relationships between entities such as threat actors, malware, and vulnerabilities.
This limitation arises because relevant evidence is often distributed across multiple text fragments and documents. Knowledge graphs
address this challenge by enabling structured multi-hop reasoning over CTI data through explicit representations of entities and
relationships. However, multiple retrieval paradigms, including graph-based, agentic, and hybrid approaches have emerged with
differing assumptions and failure modes. It remains unclear how these approaches compare in realistic CTI settings and when graph
grounding improves performance. We present a systematic evaluation of four RAG architectures for CTI analysis: standard vector
retrieval, graph-based retrieval over a CTI knowledge graph, an agentic variant that repairs failed graph queries, and a hybrid approach
combining graph queries with text retrieval. We evaluate these systems on 3,300 CTI question–answer pairs spanning factual lookups,
multi-hop relational queries, analyst-style synthesis questions, and unanswerable cases. Our results show that graph grounding
substantially improves performance on structured factual queries. The hybrid graph–text architecture improves answer quality,
measured using a composite metric of agreement, adequacy, faithfulness, and clarity by up to 35% on multi-hop questions compared
to vector RAG. However, graph-only pipelines introduce failure modes such as latency variance and overconfident answers when
information is missing. Architectures with query repair or hybrid retrieval achieve the most reliable overall performance.
CCS Concepts:•Information systems→Language models.
Additional Key Words and Phrases: RAG, GraphRAG, Agentic GraphRAG, HybridRAG, Cyber Threat Intelligence, Multi-hop QA
ACM Reference Format:
Dženan Hamzić, Florian Skopik, Max Landauer, Markus Wurzenberger, and Andreas Rauber. 2018. Beyond RAG for Cyber Threat
Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval. InProceedings of Make sure to enter the correct conference
title from your rights confirmation email (Conference acronym ’XX).ACM, New York, NY, USA, 35 pages. https://doi.org/XXXXXXX.
XXXXXXX
Authors’ Contact Information: Dženan Hamzić, dzenan.hamzic@ait.ac.at, AIT Austrian Institute of Technology, Vienna, Vienna, Austria; Florian Skopik,
AIT Austrian Institute of Technology, Vienna, Austria, florian.skopik@ait.ac.at; Max Landauer, AIT Austrian Institute of Technology, Vienna, Austria,
max.landauer@ait.ac.at; Markus Wurzenberger, AIT Austrian Institute of Technology, Vienna, Austria, markus.wurzenberger@ait.ac.at; Andreas Rauber,
TU Wien, Vienna, Austria, andreas.rauber@tuwien.ac.at.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1arXiv:2604.11419v1  [cs.AI]  13 Apr 2026

2 Hamzić et al.
1 Introduction
Cyber threat intelligence (CTI) plays a critical role in modern cyber defense by enabling organizations to anticipate,
detect, and respond to emerging threats. By systematically collecting, analyzing, and contextualizing information about
adversaries, vulnerabilities, and attack campaigns, CTI supports informed decision-making and proactive security
strategies. As cyber threats continue to grow in scale and sophistication, effective CTI analysis has become essential for
maintaining situational awareness and reducing organizational risk. CTI analysts must continuously distill long-form
narrative reports into actionable insights about threat actors, malware families, infrastructure vulnerabilities, and
attacker campaigns. Large language models (LLMs) offer a natural-language interface that allows analysts to query CTI
reports directly, but in operational CTI the dominant risk is decision error: confident but incorrect answers that can
lead to incorrect response prioritization, delay mitigation, or distort situational awareness. Prompting without external
knowledge grounding is therefore insufficient in high-stakes settings because LLMs may hallucinate entities, relations,
and supporting evidence with high confidence [17].
queryUserSystem boundary
How many cyber attacks
happened last week?Hop 2
CTI Document A
APT29 Campaign ReportCTI Document B
Weekly Threat DigestCTI Document C
MITRE ATT&CK Updates
Fragment 1
"5 ransomware attacks detected
in DACH region, T ue–Thu..."
AI Retrieval SystemHop 1
Fragment Combination
Frag 1: 5 ransomware (DACH, T ue–Thu)
Frag 2: 2 ICS intrusions (Fri–Sat)
Frag 3: 1 energy grid attack (Mon)
Answer:  8 attacks occurred last week.
(5 ransomware + 2 ICS intrusions + 1 energy grid attack)answerFragment 2
"2 ICS intrusions linked to
APT group, Fri–Sat..."Fragment 3
"Critical infra sector targeted,
energy grid, Mon..."Hop 3
Fig. 1. AI Retrieval in CTI Domain.
Retrieval-Augmented Generation (RAG) mitigates this risk by grounding responses in external evidence retrieved
at inference time [ 13,22]. Most deployed systems implement dense retrieval over chunked text [ 20], where queries
and documents are encoded into continuous vector representations and matched via similarity search (e.g., using
approximate nearest-neighbor methods such as FAISS [ 19]), followed by generation conditioned on the retrieved snippets.
However, CTI questions are frequently relational and temporal, requiring multi-hop reasoning, where answering a
query involves chaining together multiple related facts across entities, CTI reports, or time (e.g., actor →uses→
malware→targets→sector, or comparing campaigns over time). Dense retrieval that returns the top- 𝑘most relevant
text chunks [ 20,22] can fail when evidence is distributed across distant text fragments, when constraints must be
satisfied jointly, or when the answer depends on chaining multiple facts [ 40]. Equally important, LLM-based CTI
assistants must reliably abstain when the reports do not support an answer, rather than filling gaps with plausible
speculation [ 31]. Accordingly, we treat unsafe abstention, i.e., the failure to correctly identify unanswerable queries,
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 3
and latency instability, i.e., unpredictable response times in time-critical settings, as security-relevant failure modes in
CTI assistants.
Graph-based retrieval has re-emerged as a promising direction: converting unstructured CTI text into a property
graph representation [ 1] enables explicit grounding of entities and relationships and allows queries to traverse these
relations under structural constraints, facilitating multi-hop reasoning over interconnected threat intelligence. Yet,
structure introduces new operational failure modes. Answering now depends on generating and executing correct
graph queries (e.g., Cypher, a declarative query language for graphs [ 10]), and failures may arise from text-to-query
translation errors, schema mismatches, or empty-result queries. All these issues could erase the theoretical benefits of
graph grounding [ 23,39]. Moreover, partial structure can produce overconfident answers when evidence is incomplete,
and iterative correction loops can introduce large and unpredictable latency, both security-relevant risks in time-critical
CTI workflows. Recent agentic patterns suggest a remedy: plan-act-reflect loops can detect and repair tool-use mistakes
during execution [ 34,41]. Still, the community lacks controlled evidence for when graph grounding helps on realistic
CTI workloads, how much correction is needed for reliability, and what runtime-quality trade-offs arise when moving
beyond RAG.
This paper provides a controlled evaluation of four representative retrieval architectures for CTI question answering,
selected to span core design dimensions in retrieval-augmented generation.
(i)Semantic RAG (RAG)over text chunks follows the standard dense-retrieval paradigm introduced by Lewis et
al. [22], relying on local semantic similarity search under context-window constraints.
(ii)Graph-only retrieval (GRAG)replaces local chunk retrieval with structured text-to-Cypher translation over a
property graph. This design supports global structural reasoning and aggregation in the spirit of recent knowledge-
graph–augmented RAG systems [8, 28, 45].
(iii)Agentic GRAG (AGRAG)extends GRAG with critique-and-repair loops inspired by agentic tool-use and
self-reflection frameworks such as ReAct and Reflexion [ 34,41], mitigating query-generation brittleness in structured
pipelines.
(iv)HybridRAG (HRAG)combines structured graph querying with unstructured semantic retrieval, reflecting
hybrid knowledge–vector integration strategies proposed in recent multimodal and KG-RAG systems [15, 28, 29].
These architectures do not cover all RAG variants, but represent distinct retrieval paradigms discussed in the literature:
dense retriever-based RAG [ 20,22], structured retrieval over knowledge graphs [ 35], agentic retrieval with iterative
query repair [ 34,41], and hybrid systems combining structured and semantic retrieval. Other variants such as reranking,
adaptive retrieval routing, or multimodal RAG have also been explored [ 3,12,27]. These extensions mainly modify
components within these architectural families.
We evaluate the four RAG systems using five different LLMs on 3,300 automatically generated question-answer pairs
spanningsimple,single-hop,multi-hop,guided analyst-style, andunanswerablequestions. Beyond answer quality, we
explicitly analyze security-relevant failure modes, unsafe abstention behavior and latency instability, showing that
naive graph-only retrieval can underperform RAG unless paired with correction and redundancy mechanisms.
Our contributions are summarized as follows:
•A CTI evaluation dataset and generation pipeline.We contribute a dataset of 3,300 labeled QA pairs with
controlled question types and query provenance, enabling reproducible evaluation of CTI-focused RAG variants.
•A systematic, controlled comparison.We compare semantic RAG, graph-only RAG, agentic GraphRAG, and
hybrid graph-text retrieval under a shared CTI knowledge base and a unified evaluation protocol.
Manuscript submitted to ACM

4 Hamzić et al.
•A systematic analysis of graph-based retrieval trade-offs.We quantify per-question-type benefits and
show that naive graph-only retrieval can degrade safety on unanswerable queries and induce runtime instability,
motivating explicit safeguards.
•Security-relevant failure and robustness analysis.We analyze how structured retrieval changes CTI failure
modes, including overconfident answers from partial structure, cascading errors from structured backends, and
latency spikes from iterative query repair-properties that directly affect the trustworthiness of CTI assistants.
To systematically analyze the effectiveness, robustness, and cost of graph-based retrieval for cyber threat intelligence,
we investigate the following research questions:
•RQ1:To what extent do graph-based and hybrid RAG systems improve answer quality over semantic RAG on
CTI tasks?
•RQ2:To what extent do different question types (simple, single-hop, multi-hop, guided, unanswerable) benefit
from explicit graph grounding?
•RQ3:How does the underlying LLM affect text-to-Cypher generation performance?
•RQ4:How large is the runtime-quality trade-off across systems and models?
The remainder of this paper is structured as follows. Section 2 reviews related work on retrieval-augmented generation,
graph-based retrieval, and agentic and hybrid RAG systems, positioning our contribution within prior research. Section 3
describes the CTI dataset, graph construction process, experimental design, and evaluation methodology. Section 4
presents quantitative results across retrieval architectures, question types, and language models, including robustness
and runtime analyses. Section 5 discusses the implications of these findings, with a focus on security-relevant failure
modes and deployment trade-offs. Finally, Section 6 concludes the paper and outlines directions for future work.
2 Background and Related Work
This section reviews the foundations of retrieval-augmented generation and related work on graph-based and agentic
retrieval architectures relevant to our study.
2.1 Retrieval-Augmented Generation
RAG combines parametric language models with external knowledge sources retrieved at inference time, enabling
models to ground responses in supporting evidence and reduce hallucinations [ 22]. In a standard RAG pipeline, a user
query is embedded, a retriever selects the top-k relevant text passages using dense retrieval technique (e.g. Dense Passage
Retrieval) [ 20], and the language model generates an answer conditioned on the retrieved context. However, several
limitations have been identified. In particular, top-k passage retrieval often fails when relevant evidence is distributed
across multiple documents or when answering requires multi-hop reasoning across entities and relations [40].
2.2 Graph–Augmented Retrieval Generation
One direction for improving RAG systems is the integration of knowledge graphs or property graphs that explicitly
represent entities and relationships. In such systems, unstructured text is converted into graph representations (using e.g.
LLM prompting), enabling queries to traverse structured relationships and aggregate evidence across entities [ 8,28,45].
Graph-based retrieval can improve performance on relational queries and multi-hop reasoning tasks by explicitly
following entity relationships. However, benefits of graph grounding may depend strongly on query complexity and
graph coverage [39].
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 5
Table 1. Summary of related graph-based RAG work
Paper Architecture focus Relevance to this work
Edge et al. [8] Hierarchical
GraphRAGHierarchical graph retrieval for summarization; motivates multi-level
retrieval
Pan et al. [28] KG–LLM integration Conceptual taxonomy for graph–LLM integration
Wu et al. [38] Domain GraphRAG Medical GraphRAG with domain-specific enhancements
Agentic Medical [2] Agentic GraphRAG Self-correcting agentic refinement for structured retrieval
Papageorgiou et al. [ 29]Hybrid GraphRAG Hybrid multi-agent GraphRAG; explainability benefits
Liang et al. [23] Robustness analysis Security evaluation of GraphRAG under poisoning attacks
2.3 Agentic and Self-Correcting Retrieval
Recent work explores agentic RAG architectures that augment retrieval pipelines with iterative reasoning and tool use.
Frameworks such as ReAct [ 41] and Reflexion [ 34] demonstrate that language models can improve task performance by
alternating between reasoning steps, tool execution, and self-reflection. In retrieval pipelines, these mechanisms enable
systems to detect and repair failures during query generation or tool interaction.
2.4 GraphRAG Applications and Evaluation Challenges
Table 1 summarizes the related work. Graph-based RAG (GraphRAG) incorporates knowledge graphs to enable entity-
centric reasoning and relational traversal [ 28,42,45]. Edge et al. demonstrate hierarchical GraphRAG for query-focused
summarization [8], while Xiang et al. show GraphRAG benefits depend critically on query complexity [39].
Domain-specific applications illustrate both opportunities and challenges: medical GraphRAG systems [ 2,38] show
improved safety through structured search, while hybrid approaches [ 29] leverage multiple retrieval modalities to
compensate for individual weaknesses. However, Liang et al. demonstrate that GraphRAG’s relational structure creates
novel attack surfaces, with poisoning propagating across interconnected entities [ 23], underscoring the need to jointly
evaluate quality and robustness. Prior work reports heterogeneous and task-dependent outcomes for graph-/KG-
augmented RAG, and evaluation practices remain inconsistent across pipelines, tasks, and domains, which complicates
direct cross-paper comparison [ 30]. Consequently, many studies introduce and test a single proposed variant on one or
a small set of benchmarks, limiting isolation of causal factors [45].
Our work addresses this gap through a controlled evaluation of four retrieval architectures in the CTI domain. By
holding data, models, and evaluation constant, we disentangle the effects of explicit graph grounding, agentic query
refinement, and hybrid retrieval redundancy.
3 Methodology
The goal of this study is to systematically evaluate how different retrieval architectures affect the reliability and quality
of LLM-based question answering over CTI reports. We compare four retrieval paradigms: RAG, GRAG, AGRAG, and
HRAG. These architectures represent different approaches to grounding LLM responses in external knowledge sources
and exhibit distinct trade-offs between contextual coverage, relational reasoning capabilities, and robustness to missing
information.
To enable a controlled comparison, we construct an evaluation framework that transforms unstructured CTI reports
into structured textual representations. First, CTI documents are converted into Cypher statements using an LLM-based
text-to-graph extraction process. These statements populate a Neo4j property graph that captures entities such as threat
actors, malware families, vulnerabilities, and their relationships.
Manuscript submitted to ACM

6 Hamzić et al.
CTINexus CTI Reports1.
Sample 15 Reports2. 
Init DBs3.
Generate Cypher from
CTI Reports
4. 
Create Neo4j Graph
6.
Generate Evaluation
Q&A
CTI Summary Report7.
Generate "Guided" Q&A5.
Generate GRAG Few-
Shot Samples 8.
Init RAG Systems9.
Question Answering
Repeat 10 T imes10.
LLM-as-a-Judge
Evaluation11.
Evaluation with
Classical Metrics
Fig. 2. Data generation and evaluation workflow.
Question–answer pairs are derived from the generated Cypher queries. Generating questions from Cypher ensures
that the evaluation only tests information that is verifiably present in the knowledge graph and corresponding CTI
reports. The resulting dataset contains multiple question types, including simple lookups, relational queries, multi-hop
reasoning tasks, analyst-style guided questions, and unanswerable cases.
Each RAG architecture then answers the generated questions using identical inputs and underlying knowledge
sources. System outputs are evaluated using both classical automatic metrics and an LLM-as-a-Judge framework to
assess answer quality, hallucination behavior, abstention capability, and runtime stability. The following subsections
describe the formal problem formulation, the evaluation framework, dataset construction, retrieval architectures, and
evaluation metrics in detail.
3.1 Formal Problem Definition
We model CTI question answering as a retrieval-augmented generation task over a document corpus and an associated
knowledge graph. Let D={𝑑 1,𝑑2,...,𝑑 𝑛}denote a corpus of CTI reports. From this corpus, a knowledge graph is
constructed and represented as 𝐺=(𝑉,𝐸) , where𝑉denotes entities (e.g., Malware, ThreatActor, Tool, Victim) and 𝐸
represents relations (e.g., uses, targets, exploits) between these entities.
A set of evaluation questions is defined as 𝑄={𝑞 1,𝑞2,...,𝑞 𝑚}, where each question 𝑞𝑖is associated with a reference
answer𝑎𝑖. For a given question 𝑞𝑖, a retrieval architecture 𝑅𝑘(e.g., graph retrieval, dense retrieval, or both) retrieves a
set of contextual evidence𝐶 𝑘:
𝑅𝑘(𝑞𝑖)→𝐶 𝑘
The retrieved context𝐶 𝑘is then provided to a LLM model (A) that generates an answer ˆ𝑎𝑘,𝑖:
𝐴(𝑞 𝑖,𝐶𝑘)→ ˆ𝑎𝑘,𝑖
We evaluate four retrieval architectures 𝑘∈{RAG,GRAG,AGRAG,HRAG} , which differ in how contextual evidence
𝐶𝑘is constructed.
•RAG.The retrieval function 𝑅RAGretrieves a set of semantically relevant document passages from Dusing
vector similarity search or dense search.
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 7
•GRAG.The retrieval function 𝑅GRAG translates the question 𝑞𝑖into a graph query (Cypher) over 𝐺, retrieving
relational evidence in the form of entity subgraphs.
•AGRAG.The retrieval function 𝑅AGRAG extends graph retrieval with an iterative critique-and-repair mechanism
that refines graph queries.
•HRAG.The retrieval function 𝑅HRAG combines semantic document retrieval and graph traversal, producing a
hybrid context set containing both textual passages and graph-derived relational evidence.
The evaluation compares architectures with respect to answer quality, hallucination behavior, abstention performance,
and runtime stability by measuring agreement between generated answers ˆ𝑎𝑘,𝑖and reference answers 𝑎𝑖across the
question set𝑄.
The following subsections describe how the Q&A dataset, knowledge graph, RAG parameters, retrieval pipelines,
and evaluation metrics are constructed and configured within this framework.
3.2 The Evaluation Framework
Because LLM outputs are probabilistic, we perform multiple evaluation runs instead of relying on a single experiment.
We conduct five independent evaluations using different LLMs, each consisting of ten runs. In each run, 15 CTI reports
are randomly sampled from the CTINexus dataset [ 7], translated into Cypher queries, and inserted into the graph
database. Figure 2 illustrates the overall experimental workflow.
After random file selection in Step 1, in order to have a fair and unbiased evaluation across the runs, both the FAISS1
vector database and the Neo4j2graph database are reset in Step 2. In Step 3, the unstructured CTI text is converted into
Neo4j Cypher statements using a LLM (see Section 3.7). In Step 4, the generated Cypher statements are executed to
populate the graph database. If an insertion error occurs, the Cypher query and the corresponding error message are
passed back to the LLM for correction, and the insertion is retried.
Once all CTI texts have been successfully converted into Cypher, Step 5 generates few-shot question-to-Cypher
examples, which are used to parametrize the GRAG architecture, using an LLM-based prompt3. These generated
examples, (question, query) pairs, are subsequently used as few-shot examples during GraphRAG inference. The prompt
instructs the model to produce a fixed number of (question, query) pairs in strict JSON format, ensuring deterministic
parsing and reproducibility. It enforces multiple structural constraints:
•Read-only Cypher restriction.Only non-mutating clauses are permitted. All write operations are explicitly
forbidden to prevent unsafe query generation.
•Schema grounding.The model is required to use only labels, relationship types, and properties defined in
the authoritative graph schema4, used in Step 3, and the previously generated Cypher inserts. This prevents
hallucinated entity types or relations.
•Executability guarantees.Queries must be directly runnable without any additional parameters, avoid hard-
coded values not present in the graph, and return minimal, clearly aliased outputs.
•Diversity constraints.The prompt enforces a balanced mix of question types, including simple node lookups,
one-hop traversals, multi-hop paths, and aggregate queries, ensuring coverage of different reasoning patterns.
1https://github.com/facebookresearch/faiss
2https://neo4j.com/product/neo4j-graph-database/
3https://github.com/ait-cti/beyond-vanilla-rag/blob/main/grag-few-shot-qc-samples.md
4https://github.com/ait-cti/beyond-vanilla-rag/blob/main/text-to-cypher-prompt.md
Manuscript submitted to ACM

8 Hamzić et al.
•Self-validation checks.Before returning the output, the model is instructed to silently verify JSON validity,
item count correctness, and clause compliance.
These constraints ensure that the generated few-shot examples are syntactically valid, schema-consistent, and
representative of the reasoning patterns required for GRAG evaluation. For each incoming user question, the system
retrieves the top-k most semantically similar example pairs and injects them into the prompt before Cypher generation.
This in-context learning setup guides the LLM to produce schema-compliant, executable Cypher queries by imitating
the structure and patterns of the provided examples.
In Step 6, evaluation questions are generated from the Cypher queries using LLM prompting5. Deriving questions
from executable Cypher queries serves as a control to ensure that answerability is determined by the shared knowledge
base across all RAG architectures. Importantly, this does not provide graph-based systems with privileged information:
all systems receive only the natural-language question at inference time, and RAG has access to the same underlying
facts through the original CTI texts. Performance differences therefore reflect retrieval and reasoning effectiveness. In
this step, four types of questions are generated from the CTI Cypher queries: 15 simple, 15 single-hop, 15 multi-hop,
and 5 unanswerable questions. These categories follow established distinctions in question answering (QA) research
and are particularly suited for evaluating structured retrieval. Simple and single-hop questions correspond to entity
lookup and one-relation reasoning in knowledge-graph QA and semantic parsing, requiring retrieval of node properties
or traversal of a single edge. Multi-hop questions require compositional reasoning across multiple relations, consistent
with multi-hop benchmarks such as HotpotQA [ 40], and test the system’s ability to combine distributed evidence.
Unanswerable questions follow the SQuAD 2.0 paradigm [ 31], requiring models to detect insufficient evidence and
abstain rather than hallucinate, an essential property in CTI settings. We exclude formats such as multiple-choice or
opinion-based questions, as they do not reflect realistic CTI workflows.
In Step 7, we introduce an additional set of 16 “guided” questions derived from the Australian Government’s Annual
CTI Report6to approximate realistic cybersecurity analysis tasks. Unlike the Cypher-derived categories (simple, single-
hop, multi-hop, unanswerable), guided questions originate from an external CTI document and are therefore not
guaranteed by graph construction or schema coverage. An LLM is prompted7to generate 16 questions (8 multi-hop and
8 of other types, like simple or single-hop questions) covering strategic CTI themes such as threat actors, TTPs, sectors,
and geopolitical context. To reduce bias and triviality, generated questions are validated for answerability against the
selected CTINexus reports, filtered to remove near-trivial fact lookups, and required to reference entities present in the
corpus. This design reflects analyst-driven intelligence requirements, where questions emerge from external reporting
needs rather than schema-constrained database exploration.
In Step 8, the vector-based RAG database is initialized. The CTI texts are chunked and embedded into the vector
database, and the HybridRAG system is initialized by embedding textual representations of entities and relationships
using OpenAI’s text-embedding-3-large8model.
In Step 9, the evaluation phase begins, during which all systems answer a total of 66 questions per run. Finally, in
Steps 10 and 11, system outputs are evaluated using classical metrics such as F1, BLEU, and ROUGE, complemented by
an LLM-as-a-Judge evaluation9[14] applied to the generated answers.
5https://github.com/ait-cti/beyond-vanilla-rag/blob/main/cypher-to-qa-pairs.md
6https://github.com/ait-cti/beyond-vanilla-rag/blob/main/Annual_Cyber_Threat_Report_2024-25.pdf
7https://github.com/ait-cti/beyond-vanilla-rag/blob/main/guided_qa_generation.md
8https://platform.openai.com/docs/models/text-embedding-3-large
9https://github.com/ait-cti/beyond-vanilla-rag/blob/main/llm-judge.md
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 9
Table 2. Generated Question Quality and Validation Metrics
Metric Value
Question Diversity
Total questions (per run) 66
Unique 4-word prefixes 398/522 (76.1%)
Answer Conciseness (factoid categories only)
Categories included simple, single/multi-hop, unans.
Answers≤12 words 99.4%
Mean answer length (words) 1.6
Median answer length (words) 1
Guided Answers (excluded from constraint)
Mean answer length (words) 39.4
Aggregate Multi-hop Coverage
Aggregates per run (mean±sd) 7.2±1.4
Range across runs 5–9
Runs with≥5 aggregates 10/10
3.3 Generated Q&A Validation
Table 2 validates question generation quality across three dimensions. First, pattern diversity: 76% of questions begin with
unique four-word prefixes, indicating the generation process avoids repetitive phrasing. Second, answer conciseness:
for factoid categories (simple, single-hop, multi-hop, unanswerable), which require precise retrievable answers, 99.4% of
gold answers fall within 12 words, what aligns with one of the requirements in the Q&A generation prompt, consistent
with answer-length distributions observed in established QA benchmarks such as SQuAD [ 32](most answers are short
spans, with typical lengths below 10 tokens) and Natural Questions [ 21](answers similarly concentrated in short spans,
generally under∼10 tokens). Guided questions are excluded from this constraint by design, as they elicit multi-sentence
explanations that test reasoning chains rather than factoid retrieval. Third, compositional coverage: each run contains
at least 5 aggregate multi-hop questions. Across runs, the number of such questions per run has a mean of 7.2 (range:
5–9), providing sufficient per-run sample size to compute meaningful sub-category scores for compositional reasoning.
Questions reference entities present in the constructed knowledge graph, ensuring the evaluation tests retrieval of real
CTI concepts. This design guarantees a fair comparison between graph-based and semantic retrieval, as the evaluated
knowledge is shared between both representations.
3.4 RAG Systems Configuration
This section describes the configuration of the 4 RAG systems.
3.4.1 RAG.LLMs have a limited prompt context window, which prevents the entire contents of a database from being
injected during the retrieval phase. To reflect this constraint, we limit the number of retrieved text chunks to 𝑘=3,
meaning that only the three most semantically similar chunks are inserted into the RAG prompt. The chunk size is set
to 200 characters with an overlap of 20 characters [18, 37].
This configuration prevents overloading the model’s context window, which could otherwise lead to degraded
performance or catastrophic forgetting [ 16]. As illustrated in Figure 3, the user query is first embedded, after which the
Manuscript submitted to ACM

10 Hamzić et al.
RAGGRAGAGRAGHRAG
User
Embedder
(LLM) VectorDBLLM AnswerQuery
Retrieve Similar
Text Chunks
Top-k ChunksGenerate
AnswerCypher Generator
(LLM) GraphDBLLM AnswerExecute Cypher
on Graph
Entities + RelationsGenerate
AnswerCypher Critique
(LLM) GraphDBLLM Answer
DB Result + CypherImproved Cypher
Improved ResultsGenerate
AnswerRefine loop (up to 6×)Zero Shot Cypher
Generator (LLM)GraphDB
Hybrid Retriever
(Vector+Full-text)LLM Merger
(Synthesis)Answer
Semantic +
Keyword SearchExecute CypherEntities + Relations
Top-k ChunksGenerate
AnswerParallel
PipelinesBuilds on
Fig. 3. Overview of different RAG systems.
Start Generate Cypher Guardrails Validate Cypher Validate CypherCorrect Cypher
Execute Cypher
Generate Final Answer End
Fig. 4. GRAG system configuration.
top-𝑘similar chunks are retrieved from the vector database and provided to the LLM. The LLM then generates the final
answer conditioned on both the user query and the retrieved context, following a predefined RAG prompt10.
3.4.2 GRAG.As illustrated in Figure 3, GRAG uses a graph database (Neo4j) as the primary source for information
retrieval. As shown in Figure 2, in Step 5, few-shot question-to-Cypher example pairs are generated and stored. These
pairs serve as reference examples for translating natural-language queries into Cypher statements.
When a user submits a query, the system retrieves the three most similar (question, Cypher) pairs and inserts them
as few-shot examples into the prompt. Cypher generation then follows an iterative control loop (Figure 4).
First, guardrails perform a lightweight domain check and reject non-CTI queries before graph interaction. If accepted,
the LLM is prompted11with the user question, Neo4j schema, and retrieved few-shot examples to generate a candidate
10https://github.com/ait-cti/beyond-vanilla-rag/blob/main/RAG-prompt.md
11https://github.com/ait-cti/beyond-vanilla-rag/blob/main/GRAG-prompts.md
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 11
Sub-symbolic SymbolicUser
Prompt Prompt
Cypher Generator
(LLM)
GraphDBGraph SearchCypherEmbedder (LLM)
Vector SearchEmbedding
Graph T riplets/
Text ChunksLLM MergerResponse
GraphDB
Vector Index
Fig. 5. HybridRAG parallel pipeline
read-only Cypher query. The query is validated for syntactic correctness and schema conformity and executed against
Neo4j to detect runtime or empty-result errors. If validation fails, the error message and previous query are fed back
to the LLM for repair. This generate–validate–repair loop continues until successful execution or a maximum of 25
iterations is reached. Once execution succeeds, the database results are converted into a natural-language answer.
The implementation is based on LangChain12and LangGraph13, following the reference architecture provided by the
LangChain framework.
3.4.3 AGRAG.AGRAG builds on the output of GRAG, namely the natural-language query, the generated Cypher
code, and the database results. It employs an LLM tocomment on and critically assessthe generated Cypher query (see
Figure 3) using a dedicated critique prompt14.
Given the user question and the Neo4j database schema, AGRAG first evaluates the Cypher query produced by
GRAG. If issues are identified, the LLM returns a refined version of the Cypher query together with an explanatory
comment. The refined query is then executed against the database. In the presence of execution errors or remaining
inconsistencies, the critique-and-refinement loop is repeated for up to six iterations. In practice, depending on the
underlying LLM, this iterative refinement is rarely required more than once, as most Cypher errors are resolved during
the first correction step.
3.4.4 HRAG.HybridRAG builds on the insight from prior work [ 15,28,29] that combining symbolic knowledge graphs
with semantic vector search enables complex reasoning over CTI data that neither modality achieves alone. This system
first creates a unified search layer inside Neo4j called SearchDoc . All relevant graph nodes (e.g., ThreatActor, Malware,
Victim) and relationships (e.g., uses, attacked, exploits) (see Sect. 3.7 for the complete schema) are converted into short,
human-readable textual representations and stored in a unified text property that aggregates the descriptive content of
12https://www.langchain.com/
13https://www.langchain.com/langgraph
14https://github.com/ait-cti/beyond-vanilla-rag/blob/main/AGRAG-prompt.md
Manuscript submitted to ACM

12 Hamzić et al.
each Neo4j element. This allows the system to search the entire graph in a consistent way, regardless of whether the
information originally came from a node or an edge. On top of this SearchDoc layer, two indexes are created:
•a full-text index for keyword-based search (names, IDs, exact terms),
•a vector index for semantic search using embeddings (search by meaning).
Ahybrid retrievercombines both approaches, so each question retrieves relevant SearchDoc entries using keyword
matching and semantic similarity together. At query time, HybridRAG runs two pipelines (see Fig. 5) in parallel:
(1)Graph pipeline (Symbolic): an LLM translates the question into a Cypher query (Zero Shot Generator in Figure 3),
executes it on Neo4j, and returns structured results such as tables, counts, and explicit relationships. If the query
fails, simple rule-based fixes are applied first, followed by LLM-based repair if needed.
(2)Retrieval pipeline (Sub-Symbolic): the hybrid retriever fetches unstructured SearchDoc text snippets that provide
descriptive and contextual information.
Finally, a synthesis LLM prompt15merges both outputs. It prioritizes the graph results for exact facts and relationships,
while using the retrieved text to enrich the answer or fill gaps. The final response is concise, consistent, and grounded
in the underlying graph data.
3.5 Dataset
In this paper, we use the publicly available CTINexus [ 7] dataset, which consists of 150 CTI reports collected from
well-recognized threat-sharing platforms. We select the CTINexus dataset because it offers a large-scale collection of real-
world CTI reports written in complex, domain-specific language, enabling a realistic evaluation of retrieval-augmented
generation methods on unstructured security texts. The authors of CTINexus performed entity and relationship
extraction on the original reports, resulting in 26 unique cybersecurity-related entity types and 855 unique cybersecurity-
related relationships connecting these entities across the 150 reports.
3.6 Generated Graph Analysis and Validation
We confirm the structural validity of the resulting knowledge graph (from unstructured CTI texts via LLM prompt
to Cypher) by analyzing the structural properties of the generated graph across runs. Table 3 summarizes the graph
structure generated from the CTINexus reports for 10 runs within a single evaluation. Text-to-graph conversion
produces an average of 10.9 entities and 1.6 relationships per report through 104 lines of Cypher code with 23.4 MERGE
operations, where MERGE is a Cypher clause that creates or matches nodes and relationships to ensure non-duplicative
graph construction, demonstrating substantial structural complexity beyond simple entity catalogs.
3.7 Entities and Relationships
Following manual inspection of the CTINexus reports and consultation with cybersecurity domain experts, we restricted
the graph schema to 17 analyst-relevant entity types: ThreatActor, Malware, Tool, Victim, C2_Infrastructure, Campaign,
Incident, Date, Sector, Region, Country, Technique, CVE, Motivation, Mitigation, Capability, and Source. The domain
experts validated that the selected entity types are relevant for practical CTI analysis and reflect concepts commonly
used by analysts. These entity types correspond to commonly used concepts in CTI ontologies and knowledge graphs,
which typically model key elements such as threat actors, malware, vulnerabilities, techniques, and targets to represent
15https://github.com/ait-cti/beyond-vanilla-rag/blob/main/HRAG-prompt.md
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 13
Table 3. CTI Knowledge Graph Statistics
Metric Value
Total reports (across 10 runs) 150
Unique report files 70
Avg entities per report 10.9±5.5
Avg relationships per report 1.6±2.4
Avg Cypher lines per report 104±54
Avg MERGE statements 23.4±12.6
Unique entity types 17
Unique relationship types 8
Avg report word count 137Table 4. Schema Validation and Evidence Attribution Metrics (aggregated over
10 runs).
Metric Value Compliance
Schema Compliance
Entity type compliance 1,044/1,044 100.0%
Relationship type compliance 438/438 100.0%
Country code format (ISO 3166-1𝛼-2) 30/30 100.0%
Evidence Attribution
Relationships withevidence178/438 40.6%
Relationships withsource_id102/438 23.3%
Relationships withpage89/438 20.3%
CVE Extraction Recall
CVEs in source text extracted as nodes 63/63 100.0%
cyber attack behavior [ 4,33]. We further restricted the ontology to 20 relationship types capturing frequently occur-
ring interaction patterns in CTI reporting: attacked, uses, exploits, abuses, targets, includes, occurred_on, has_alias,
attributed_to, involved_malware, involved_tool, used_technique, occurred_in, targets_sector, located_in, motivated_by,
exploited_in, mitigates, leverages, and supported_by. Such schema-based representations of entities and relationships
are widely used in threat intelligence knowledge graph construction to structure complex CTI data before downstream
analysis and reasoning [ 4,33]. The selected relationships reflect common operational semantics such as attack execution,
exploitation, attribution, targeting, temporal occurrence, and mitigation. The schema was validated by cybersecurity
analysts and is not intended to be exhaustive, but rather to provide a controlled and reproducible ontology for evaluating
graph-based retrieval. All entities contain property fields such as name and summary, while relationships include
additional attributes, like a date property that enables a temporal dimension for graph triplets. Entities are connected
via relationships to form subject-predicate-object triplets, which serve as the fundamental building blocks of the graph.
To transform unstructured CTI text into graph representations, we instruct a GPT-5.2 (gpt-5.2-2025-12-11)16model
using a structured prompt17that specifies the allowed entities, relationships, and valid triplet patterns. This model was
selected due to strong reported performance on text-to-code tasks, particularly when applied in an agentic generation
setting [9].
3.8 Schema Validation Analysis
Schema validation analysis (Tables 4 and 5) confirms that the generated graph instances strictly conform to the
predefined ontology. Every entity node uses one of the 17 permitted types defined in the schema (1,044/1,044), and
every relationship uses one of the 20 permitted edge labels (438/438). All 30 country-code properties adhere to the
ISO 3166-1 alpha-2 standard. CVE extraction recall is computed by extracting all CVE identifiers from the source reports
via regular-expression matching of the canonical CVE-YYYY-NNNNN pattern and measuring the proportion of these
ground-truth mentions that are instantiated as CVE nodes in the graph. Across all runs, all 63 CVE identifiers present
in the source reports were successfully extracted (100%). No analogous recall calculation is possible for other entity
types such as malware or threat actors, because these lack a standardised identifier format that would allow reliable
automated matching between source text and extracted nodes. Table 5 therefore reports raw node counts by entity type.
16https://platform.openai.com/docs/models/gpt-5.2
17https://github.com/ait-cti/beyond-vanilla-rag/blob/main/text-to-cypher-prompt.md
Manuscript submitted to ACM

14 Hamzić et al.
Table 5. Extracted Entity and Relationship Counts (ag-
gregated over 10 runs; “Unique” counts deduplicated by
node identifier across runs).
Type Total nodes Unique
Entity Types
ThreatActor 158 101
Technique 141 –
Malware 109 73
Incident 102 99
Victim 88 64
CVE 67 50
Tool 67 47
Sector 62 11
Date 58 –
Source 52 48
Motivation 36 –
Country 30 12
Capability 23 15
C2_Infrastructure 16 –
Campaign 15 15
Region 10 5
Mitigation 10 9
Total 1,044–Table 6. The chosen models for the evaluation
LLM # Parameter Comment
GPT-5.2 Unknown Reasoning
GPT-4.1-mini Unknown Common in math
Kimi K2 Thinking 1T Largest OS LLM
Mistral Small 24B Instruct
Mixtral 8x7B 56B Instruct
In total, GPT-5.2 extracted 101 unique threat actors, 73 unique malware families, 50 unique CVEs, and 47 unique tools
across the 10 runs. Evidence attribution, an evidence property containing a supporting quote from the source report,
is present on 40.6% of relationships (178/438). Finer-grained provenance in the form of source_id andpage properties
appears on 23.3% and 20.3% of relationships, respectively.
3.9 LLM Choice
The requirements for selecting the baseline LLM are as follows:
(1)Strong long-context performance, required to reliably process and reason over the large Neo4j database
schema.
(2)Strong code-generation capability, necessary for accurately translating natural-language prompts into Cypher
queries.
The choice of LLM for prompt-to-Cypher translation was informed by results reported on the SWE-Bench Pro
benchmark18and long-context performance benchmarks such as Tau2-Bench-Telecom19[5]. At the time of writing,
OpenAI’s GPT-5.2 demonstrates leading performance on both benchmarks, outperforming other proprietary models
such as Gemini 3 Pro and Claude Opus 4.5. In addition, GPT-5.2 achieves state-of-the-art performance on agentic
text-to-code tasks [ 9], making it well suited for the complex prompt-to-Cypher translation and iterative reasoning
required by our graph-based RAG systems.
18https://vertu.com/ai-tools/gpt-5-2-benchmark-analysis-performance-comparison-vs-gpt-5-1-gemini-3-pro
19https://artificialanalysis.ai/evaluations/tau2-bench
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 15
We selectedKimi K2 Thinkingas the largest open-source model according to Hugging Face’s model listing20, and
used it as a direct open-source competitor to GPT-5.2. Both models are reasoning-oriented, which makes a comparison
meaningful, even though GPT-5.2 is probably larger in terms of parameter count than its open-source counterpart.
Because reasoning models can increase end-to-end latency-especially for GRAG, AGRAG, and HRAG, which involve
iterative query generation and execution-we also include smaller, non-reasoning models. In particular, we evaluate
GPT-4.1-mini21and Mistral Small 24B Instruct22, which is reported to outperform GPT-4.0-mini and is expected to be
competitive with GPT-4.1-mini. Finally, to study performance with a medium-sized open-source model in the 50-70B
parameter range, we include Mixtral 8x7B Instruct, which outperforms Llama 2 70B on many benchmarks23.
Together, this model-set (see Table 6) is intentionally heterogeneous: it includes a strong long-context reasoning
models to approximate an upper bound for structured tool use, smaller proprietary and open-source baselines represen-
tative of cost-sensitive deployments, and large open-source reasoning models to test whether graph-based RAG gains
generalize beyond a single vendor. This diversity allows us to attribute observed performance differences primarily to
retrieval architecture and agentic design choices, rather than to a specific model family or scale.
3.10 Evaluation Metrics
In this study, we employ two complementary sets of evaluation metrics.
•Classical automatic metrics, including F1-Score [ 36], BLEU [ 6], ROUGE-1 [ 24] and ROUGE-L [ 24], and
BERTScore (F1) [43].
•LLM-as-a-Judge evaluation, using a custom-defined rating schema.
The motivation for using both metric types is that classical metrics can be overly sensitive to surface-level differences
in wording or formatting. For example, a ground-truth answer such as “First of January, 2020” may be penalized when
the model produces “01.01.2020”, despite the two answers being semantically equivalent. Classical metrics often struggle
to robustly capture this type of semantic equivalence [ 26]. Consequently, LLM-based evaluation is included to better
assess semantic correctness and answer quality beyond lexical overlap [14, 25, 44].
Composite classical metric.In addition to reporting individual classical metrics, we aggregate F1, BLEU, ROUGE-1,
ROUGE-L, and BERTScore (F1) into a single composite score to facilitate concise comparison across systems. Each
metric is aggregated using equal weights. BLEU is used in its native scale, while the remaining metrics naturally lie in
the[0,1]range; the composite score is intended as a comparative summary rather than a calibrated absolute measure.
This yields the following weighted average:
CompositeScore=1
5 F1+BLEU+ROUGE-1+ROUGE-L+BERTScore F1(1)
Equal weighting avoids privileging any single surface-form metric and reflects our intent to treat lexical overlap and
semantic similarity as equally informative signals.
20https://huggingface.co/models?pipeline_tag=text-generation&sort=most_params
21https://platform.openai.com/docs/models/gpt-4.1-mini
22https://mistral.ai/news/mistral-small-3/
23https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
Manuscript submitted to ACM

16 Hamzić et al.
Table 7. Inter-criterion Pearson correlations (N=2,640 judgments across 10 runs).
C1–C3 form a tightly coupled correctness cluster ( 𝑟≥0.91), while C4 (Clarity)
is substantially less correlated (𝑟=0.43–0.50).
C1 C2 C3 C4
Agreement Adequacy Faithfulness Clarity
C1 (Agreement) —0.98 0.920.43
C2 (Adequacy) —0.910.46
C3 (Faithfulness) — 0.50
C4 (Clarity) —Table 8. Per-run stability (mean ±sd across 10
runs)
Pair𝑟
C1–C20.98±0.01
C1–C30.92±0.02
C2–C30.91±0.02
C1–C40.42±0.07
C2–C40.45±0.07
C3–C40.49±0.06
3.11 LLM-as-a-Judge Evaluation
To mitigate the limitations of surface-form–based metrics, we employ anLLM-as-a-Judgeevaluation framework, in
which a large language model compares candidate answers against a reference (baseline) answer using a fixed scoring
rubric. Each candidate is evaluated along four criteria, scored on a 0-5 scale and combined via weighted aggregation:
•C1: Agreement with Baseline (weight 4).Measures semantic alignment with the baseline answer, independent
of surface form. Contradictions and unsupported deviations are penalized.
•C2: Task Adequacy (weight 3).Assesses how completely and correctly the candidate addresses the question.
Explicitly acknowledging insufficient information is considered fully adequate when the context does not support
an answer.
•C3: Faithfulness (weight 2).Evaluates whether the answer avoids hallucinations and unsupported claims,
remaining grounded in the available context.
•C4: Clarity and Brevity (weight 1).Rewards clear, concise, and well-structured responses.
The final score, or answer quality, is computed as:
Weighted Total=Answer Quality=4𝐶1+3𝐶2+2𝐶3+𝐶4,
with a maximum of 50 points.
If the baseline answer is empty or explicitly states that the question cannot be answered, candidates that clearly
acknowledge the lack of sufficient information-without adding speculative content-receive full scores for agreement,
adequacy, and faithfulness. This prevents penalizing correct refusals in under-specified settings.
The judge produces a strictly structured JSON object24containing per-system criterion scores, weighted totals, short
qualitative comments, and an explicit ranking. Prior work shows that LLM-based evaluators correlate strongly with
human judgments and outperform classical automatic metrics in assessing semantic correctness and factual consistency,
particularly for open-ended generation tasks [11, 14, 25, 44].
Rubric validation.Post-hoc analysis (see Table 7 and Table 8) of inter-criterion correlations supports the design of the
scoring rubric. C1 (Agreement), C2 (Adequacy), and C3 (Faithfulness) are highly correlated ( 𝑟=0.92to0.98), reflecting a
shared underlying correctness construct: answers that agree with the baseline also tend to be adequate and faithful. In
contrast, C4 (Clarity) is substantially less correlated with the other criteria ( 𝑟=0.43to0.50), confirming that it captures
an orthogonal quality dimension. This supports our answer quality metric design, which assigns the highest weight to
Agreement (C1, weight 4) and the lowest weight to Clarity (C4, weight 1), ensuring the composite score is dominated by
24https://github.com/ait-cti/beyond-vanilla-rag/blob/main/llm-judge.md
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 17
Table 9. LLM-as-a-Judge improvements over RAG (RQ1). Mean and median differences are computed over 3,300 paired question
instances.
System MeanΔvs RAG 95% Bootstrap CI MedianΔCohen’s𝑑
GRAG+1.34[0.40,2.28]0 0.05
AGRAG+10.68[9.84,11.52] +4 0.43
HRAG+11.68[10.88,12.45] +4 0.52
Table 10. Classical-metric improvements over vanilla RAG (RQ1). Mean and median differences are computed over 3,300 paired
question instances using the equally weighted composite score of F1, BLEU, ROUGE-1, ROUGE-L, and BERTScore.
System MeanΔvs RAG 95% Bootstrap CI MedianΔCohen’s𝑑
GRAG−0.06[−0.53,0.37] +0.12−0.05
AGRAG+1.93[0.83,3.01] +2.19 0.76
HRAG+1.80[0.92,2.68] +1.68 0.88
factual correctness rather than surface-form quality. Additionally, the near-independence of C4 provides a diagnostic
signal: systems that score high on C4 but low on C1–C3 produce fluent but incorrect outputs, a hallmark of confident
hallucination.
4 Results and Analysis
This section presents the results and answers the research questions. Additionally, a detailed adversary model failure
analysis is performed and elaborated.
4.1 Answers to Research Questions
4.1.1 Semantic RAG vs Others on CTI tasks.We quantify improvements using LLM-as-a-Judge assessment and classical
automatic metrics (F1, BLEU, ROUGE, BERTScore). Table 9 shows AGRAG and HRAG deliver large improvements over
RAG (+10.68and+11.68points respectively) with medium effect sizes (Cohen’s 𝑑=0.43and0.52). GRAG provides only
marginal benefits (+1.34, zero median), suggesting schema-constrained query errors offset graph grounding advantages
without correction mechanisms. Table 10 corroborates these findings: AGRAG and HRAG exceed twofold improvements
on composite classical metrics with large effect sizes, while GRAG slightly underperforms RAG.
RQ1: To what extent do graph-based and hybrid RAG systems improve answer quality over semantic
RAG on CTI tasks?Graph-based systems substantially improve CTI answer quality only when augmented with
agentic correction or hybrid retrieval.
Score Distribution Analysis.Mean improvements alone obscure a critical distributional phenomenon. Fig. 6 reveals
that all graph-based systems exhibitbimodalscore distributions, answers cluster at either perfect (50) or near-failure
(≤10) scores, with relatively few intermediate outcomes.
GRAG shows the most extreme bimodality: while 55.0% of answers achieve a perfect score (higher than RAG’s 31.7%),
a nearly identical fraction of answers collapse to near-zero (26.7%) as in RAG (26.8%). Only 18.3% of GRAG answers fall
in the intermediate range, compared to 41.5% for RAG. This “all-or-nothing” pattern explains GRAG’s marginal mean
Manuscript submitted to ACM

18 Hamzić et al.
Perfect (50) Middle (11–49) Near-zero (≤10)020406080Percentage of instances (%)31.741.5
26.855.0
18.326.769.5
18.8
11.774.8
20.3
4.8RAG
GRAGAGRAG
HRAG
Fig. 6. Score distribution across RAG systems (LLM-as-a-Judge, 0–50 scale). Each cell reports the percentage of the evaluated question
instances falling in the indicated score range
improvement (+1.34) despite producing more perfect answers than RAG: the gains from successful graph queries are
offset by catastrophic failures when Cypher generation or schema matching fails.
In contrast, AGRAG and HRAG progressively reduce the failure tail. AGRAG cuts near-zero answers from 26.7%
(GRAG) to 11.7%, while HRAG reduces them further to 4.8%, a 5.6-fold reduction relative to GRAG. Simultaneously, HRAG
achieves the highest perfect-score rate (74.8%). This demonstrates that agentic correction and hybrid redundancy do
not merely improve average performance; they fundamentally reshape the failure distribution, converting catastrophic
collapses into either correct answers or graceful degradations.
Summary.Graph-based systems exhibit strongly bimodal performance, producing either perfect answers or
near-failures, with GRAG showing an extreme “all-or-nothing” pattern that offsets its gains through frequent
collapses. In contrast, AGRAG and especially HRAG significantly reduce these failures and increase perfect answers,
demonstrating that agentic correction and hybrid retrieval reshape the distribution toward more reliable outcomes
rather than just improving average performance.
4.1.2 Benefits of Graph Grounding on Different Question Types.Table 11 reveals strong interaction between question
type and graph grounding effectiveness. For simple questions, AGRAG achieves +22.81points (𝑑=1.11), HRAG+18.22
(𝑑=0.85), and GRAG+13.04(𝑑=0.51), indicating direct fact lookup benefits from graph structure. Single-hop questions
show similar patterns: AGRAG +15.42, HRAG+14.98, GRAG+5.26. All graph systems strongly outperform vanilla RAG
on multi-hop questions (AGRAG +17.64, HRAG+15.18, GRAG+10.37), confirming explicit grounding aids compositional
reasoning where semantic retrieval fails to retrieve distributed evidence.
For guided questions, GRAG ( −17.67) and AGRAG (−8.74) underperform RAG, while HRAG achieves +4.32, indicating
these questions require both precise facts and contextual synthesis. Pure graph pipelines produce rigid answers; HRAG
combines structured and unstructured evidence. For unanswerable questions, GRAG ( −11.80) and HRAG (−4.84) degrade,
while AGRAG shows slight improvement (+1.20).
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 19
Table 11. RQ2 (LLM-as-a-Judge): Improvements over semantic RAG by question type. Mean differences are shown with 95% bootstrap
confidence intervals.𝑛denotes the number of paired question instances.
Category System MeanΔvs RAG 95% CI MedianΔCohen’s𝑑
guided AGRAG−8.74[−10.29,−7.19] −7.0−0.39
guided GRAG−17.67[−19.17,−16.13] −18.0−0.81
guided HRAG+4.32[2.74,5.86] +1.0 0.19
multi_hop AGRAG+17.64[16.04,19.29] +26.0 0.78
multi_hop GRAG+10.37[8.61,12.17] +4.0 0.41
multi_hop HRAG+15.18[13.69,16.69] +9.5 0.72
simple AGRAG+22.81[21.33,24.31] +33.0 1.11
simple GRAG+13.04[11.19,14.85] +5.0 0.51
simple HRAG+18.22[16.67,19.76] +18.0 0.85
single_hop AGRAG+15.42[13.83,17.01] +11.0 0.69
single_hop GRAG+5.26[3.41,7.13]0.0 0.19
single_hop HRAG+14.98[13.56,16.45] +5.0 0.72
unanswerable AGRAG+1.20[−0.45,2.84]0.0 0.09
unanswerable GRAG−11.80[−14.37,−9.28] −1.0−0.56
unanswerable HRAG−4.84[−7.27,−2.54]0.0−0.25
Table 12. RQ2 (Classical metrics): Improvements over semantic RAG by question type using the composite score of F1, BLEU,
ROUGE-1, ROUGE-L, and BERTScore. Mean differences are shown with 95% bootstrap confidence intervals over paired (model,
category) cells (𝑛=5models).
Category System MeanΔvs RAG 95% CI MedianΔCohen’s𝑑
guided AGRAG−1.74[−2.44,−0.95] −2.26−1.81
guided GRAG−1.80[−2.33,−1.00] −2.02−1.96
guided HRAG−0.84[−1.66,−0.25] −0.49−0.90
multi_hop AGRAG+2.10[1.85,2.31] +2.07 6.97
multi_hop GRAG+0.44[0.16,0.78] +0.38 1.08
multi_hop HRAG+2.09[1.45,2.68] +2.57 2.62
simple AGRAG+4.04[3.54,4.52] +3.98 6.63
simple GRAG+0.50[0.19,0.74] +0.50 1.37
simple HRAG+3.30[2.31,4.29] +3.28 2.54
single_hop AGRAG+2.48[2.01,3.30] +2.15 2.76
single_hop GRAG+0.34[0.04,0.79] +0.13 0.68
single_hop HRAG+2.17[1.26,3.31] +2.09 1.61
RQ2: To what extent do different question types benefit from explicit graph grounding?Graph grounding
significantly improves performance on simple, single-hop, and especially multi-hop questions, where explicit
relational structure enables effective fact lookup and compositional reasoning beyond semantic retrieval. However,
its effectiveness depends on question type: pure graph approaches underperform on guided and unanswerable
queries, while hybrid retrieval (HRAG) achieves the most robust performance by combining structured reasoning
with contextual evidence.
Manuscript submitted to ACM

20 Hamzić et al.
Table 13. Hallucination rate by system and question cate-
gory, defined as the percentage of answers receiving a Faith-
fulness score C3≤2 on the 0–5 scale. Lower is better.
Category RAG GRAG AGRAG HRAG
Simple 54.7% 12.7% 12.7% 8.0%
Single-hop 42.0% 13.3% 7.3% 2.0%
Multi-hop 56.7% 18.7% 20.0% 21.3%
Guided 41.9%84.4%33.1% 15.6%
Unanswerable 10.0% 46.0% 4.0% 20.0%
Overall 45.8% 34.1% 17.4% 12.4%Table 14. Safety on Unanswerable Questions: System Abstention Be-
havior
SystemCorrect Refusal (%)
(higher better)Overconfident Answer (%)
(lower better)
RAG 0.0% 100.0%†
GRAG 44.0% 56.0%†
AGRAG 0.0% 100.0%†
HRAG 76.0% 24.0%
†Safety-critical failure: system provides confident answers despite insufficient
evidence.
Faithfulness Analysis: Hallucination Rates by Category.To complement the aggregate quality analysis, we examine
Faithfulness (C3) scores as a proxy for hallucination behavior. Table 13 reports the percentage of answers receiving
C3≤2, indicating unfaithful or unsupported content.
Semantic RAG exhibits the highest overall hallucination rate (45.8%), with rates exceeding 50% on simple and
multi-hop questions, categories where semantic retrieval frequently fails to surface the specific evidence needed. All
graph-based systems substantially reduce hallucination on factual categories: for simple questions, all three graph
systems achieve≤12.7% versus RAG’s 54.7%; for single-hop, HRAG reaches 2.0%.
However, the category-level breakdown reveals a critical inversion for guided questions: GRAG’s hallucination rate
(84.4%; defined as Faithfulness score C3 ≤2on the 0–5 judge scale) is more than double that of RAG (41.9%). When the
graph schema does not cover the concepts referenced by analyst-style synthesis questions, GRAG produces unfaithful
answers grounded in incorrect or empty query results rather than acknowledging uncertainty. AGRAG partially mitigates
this (33.1%) through critique-based detection of invalid queries, while HRAG achieves the lowest guided hallucination
rate (15.6%) by falling back to unstructured evidence when structured paths fail. For unanswerable questions, the pattern
differs: RAG achieves a hallucination rate of 10.0% because its generated answers, while overconfident, tend to align
superficially with plausible refusals. GRAG (46.0%) produces unfaithful outputs from failed query loops, while AGRAG
achieves the best faithfulness (4.0%) on this category.
These findings demonstrate that hallucination risk is not uniformly reduced by graph grounding; rather, it shifts across
question types. Graph structure eliminates a large class of factual hallucinations (e.g., incorrect entity or relationship
claims) but introduces a distinct failure mode when queries fall outside schema coverage. In such cases, the system may
generate answers that are formally consistent with the graph output but semantically unsupported by the underlying
reports, a phenomenon we refer to as structural hallucination.
Summary.Graph-based retrieval substantially reduces hallucinations for factual questions compared to semantic
RAG, but this benefit is highly dependent on schema coverage and question type. When queries fall outside the
graph schema, graph-based systems, especially GRAG, introduce a new failure mode of “structural hallucination”,
whereas hybrid approaches (HRAG) mitigate this by falling back to unstructured evidence.
Operational Safety Analysis: Abstention Behavior.Note on metrics. The judge-based deltas in Table 11 evaluate answer
quality on a 0-50 scale, whereas Table 14 reports a separate binary outcome (explicit refusalvs.attempted answer). These
measures can diverge: a model may earn non-zero judge scores while still attempting an answer, and a refusal can be
judged poorly if it is not clearly justified. Table 14 shows HRAG achieves the best correct refusal rate (76%) through
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 21
Simple
Single-hop Multi-hopGuided
UnanswerableOverall
0
1
2
3
4
(a) Mean Rank
Simple
Single-hop Multi-hopGuided
Unanswerable
0
10
20
30
40Frequency (%)
Simple
Single-hop Multi-hopGuided
Unanswerable
0
10
20
30
40
50
60
70Frequency (%)
HRAG AGRAG GRAG RAG
Fig. 7. Mean ranks and top-placement frequency by question category (10 runs, LLM-as-a-Judge). (a) Mean rank (1 = best); (b) sole
best rate; (c) best-or-tied rate.
graph-text cross-validation. RAG and AGRAG exhibit zero abstention (100% overconfidence), a critical safety failure
where false positives misdirect investigation. GRAG combines 56% overconfidence with extreme latency: unanswerable
queries average 520 seconds (8.7 minutes), with worst-case 2,348 seconds (39.1 minutes), using a large reasoning LLM,
exhausting Cypher correction loops. Table 12 confirm judge findings for factual questions (AGRAG/HRAG achieve +4.0
for simple,+2.1for multi-hop), while confirming guided question degradation reflects surface-form metric limitations
on explanatory answers.
Summary.Graph grounding effectiveness varies substantially by question type. Simple, single-hop, and multi-hop
questions benefit most; guided questions require hybrid retrieval; unanswerable questions expose calibration
challenges requiring explicit uncertainty mechanisms.
Rank-based summary.To complement score-based analysis, which can be sensitive to outliers, we compute mean
ranks (see Fig. 7) across the four systems for each question instance (1 = best, 4 = worst). Overall, HRAG achieves the
best mean rank (2.06), followed by AGRAG (2.24), GRAG (2.74), and RAG (2.95). The ranking pattern varies by category:
on guided questions, HRAG dominates (rank 1.71, sole best system in 36.9% of instances), while RAG (2.39) outranks
GRAG (3.65)—confirming that unstructured retrieval is preferable to graph-only retrieval for analyst-style synthesis.
On unanswerable questions, AGRAG achieves the best rank (2.09), reflecting its low hallucination rate on this category.
For factual categories (simple, single-hop, multi-hop), all three graph systems outrank RAG, with HRAG consistently
first (see Figure 7).
4.1.3 Text-to-Cypher Generation.Cypher-dependent retrieval systems (GRAG, AGRAG, and HRAG) rely on an LLM
to translate natural-language questions into executable Cypher queries. As a result, their performance depends not
only on retrieval architecture, but also on the model’s ability to generate syntactically valid queries, respect the graph
schema, and handle empty or invalid query results. Table 15 shows strong sensitivity of GRAG to the underlying LLM.
Kimi K2 Thinking achieves the highest GRAG mean score (42.55) and the lowest collapse rate (4%), indicating robust
text-to-Cypher generation. In contrast, smaller open-source models exhibit substantially higher collapse rates (Mixtral
41%, Mistral Small 36%), meaning that a large fraction of GRAG queries degrade into near-failure outputs.
Manuscript submitted to ACM

22 Hamzić et al.
Table 15. RQ3 (LLM impact on text-to-Cypher performance): Judge-based performance of Cypher-dependent systems by underlying
LLM. Mean scores and deltas are reported with 95% bootstrap confidence intervals. Collapse rate indicates the fraction of GRAG
answers with near-failure scores (≤1010 on the 0–50 judge scale).
Model GRAG Mean AGRAG−GRAGΔHRAG−GRAGΔGRAG Collapse Rate
Kimi K2 Thinking42.55[41.40, 43.65]+1.06[0.46, 1.69]+2.65[1.87, 3.42]0.04
Mixtral 8x7B29.91[28.52, 31.25]+4.38[3.62, 5.12]+9.62[8.33, 10.86]0.41
Mistral Small 24B31.18[29.74, 32.55]+3.64[2.88, 4.42]+7.88[6.59, 9.12]0.36
GPT-4.1-mini36.49[35.36, 37.63]+1.79[1.28, 2.30]+3.88[3.01, 4.77]0.18
GPT-5.233.02[31.75, 34.34]+1.64[0.96, 2.32]+10.65[8.99, 12.35]0.27
Table 16. GRAG collapse rate by question category for GPT-5.2 (collapsed = judge score≤10/50).
Category Collapsed / Total Collapse Rate
Unanswerable 50 / 50 100.0%
Guided 8 / 160 5.0%
Multi-hop 7 / 150 4.7%
Single-hop 1 / 150 0.7%
Simple 0 / 150 0.0%
We assess LLM influence using two complementary signals: LLM-as-a-Judge scores (0–50) and acollapse ratefor
GRAG. Collapse is defined as a near-failure output with a judge score ≤10/50, typically caused by invalid Cypher,
schema mismatches, or prolonged empty-result retry loops. For GPT-5.2, all unanswerable instances collapse (see
Table 16), whereas collapse remains rare for answerable categories. Unless stated otherwise, collapse-rate breakdowns
are reported for GPT-5.2.
Agentic correction (AGRAG) improves GRAG performance across all models, but the magnitude of improvement
depends on baseline model strength. Weaker models benefit the most (Mixtral +4.38, Mistral Small+3.64), while
stronger models show only modest gains ( +1to+2points). This pattern indicates that AGRAG primarily compensates
for syntactic and schema-level errors in Cypher generation rather than fundamentally improving reasoning capability.
Hybrid retrieval (HRAG) yields the most consistent gains across all LLMs. By combining graph-based querying
with unstructured evidence retrieval, HRAG can recover useful information even when Cypher generation fails or
returns empty results. Consequently, HRAG substantially outperforms GRAG for every model, with particularly large
improvements for GPT-5.2 (+10.65) and Mixtral (+9.62).
RQ3: How does the underlying LLM affect text-to-Cypher generation performance?Cypher-based retrieval
systems depend heavily on the underlying LLM’s ability to generate correct and schema-compliant queries, leading
to large performance differences across models and frequent failures for weaker ones. While agentic correction helps
reduce these errors, HRAG is the most reliable approach, consistently improving performance by compensating
for failed or incomplete graph queries.
Collapse Rate Analysis by Question Type.Table 16 localizes GRAG failures by question type for the case of evaluation
with GPT-5.2 LLM model. Collapse is universal for unanswerable questions (100%), revealing a structural limitation
of graph-only pipelines: when the correct behavior is to acknowledge missing information, GRAG instead continues
generating alternative Cypher queries, entering prolonged retry loops (mean 520 s, maximum 2,348 s). In contrast,
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 23
Table 17. RQ3 (Classical metrics): Impact of the underlying LLM on Cypher-dependent systems. Scores are based on the composite
classical metric. Mean values and deltas are reported with 95% bootstrap confidence intervals.
Model GRAG Mean AGRAG−GRAGΔHRAG−GRAGΔHRAG−RAGΔ
Kimi K2 Thinking0.41[0.38, 0.44]+0.06[0.03, 0.09]+0.12[0.08, 0.16]+0.18[0.14, 0.22]
Mixtral 8x7B0.19[0.16, 0.22]+0.09[0.06, 0.12]+0.21[0.17, 0.25]+0.26[0.21, 0.31]
Mistral Small 24B0.22[0.19, 0.25]+0.08[0.05, 0.11]+0.18[0.14, 0.22]+0.23[0.18, 0.28]
GPT-4.1-mini0.29[0.26, 0.32]+0.05[0.03, 0.07]+0.09[0.06, 0.12]+0.14[0.11, 0.17]
GPT-5.20.31[0.28, 0.34]+0.04[0.02, 0.06]+0.23[0.18, 0.28]+0.28[0.22, 0.34]
Table 18. Results over 10 runs for all models and RAG systems (mean and standard deviation over 10 runs).
Model System R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 Mean Std
GPT-5.2AGRAG 6.87 10.12 9.80 9.08 12.69 10.17 10.51 9.71 7.40 8.84 9.52 1.55
GRAG 67.76 137.53 143.92 66.08 178.01 150.08 100.11 106.59 179.69 145.49 127.53 38.74
HRAG 37.05 54.91 43.15 42.12 52.77 46.89 53.14 41.84 47.37 50.09 46.93 5.54
RAG 1.27 1.30 1.22 1.38 1.50 1.25 1.26 1.23 1.34 1.41 1.32 0.08
MixtralAGRAG 5.90 5.21 6.83 6.68 6.01 6.98 6.37 8.47 7.79 5.18 6.54 1.00
GRAG 26.49 23.94 23.97 7.35 15.01 25.30 15.75 10.39 1.15 24.13 17.35 8.36
HRAG 8.97 18.19 10.22 16.08 9.69 12.05 11.41 10.25 9.97 10.31 11.71 2.87
RAG 1.29 1.41 1.30 1.36 1.40 2.58 1.44 1.31 1.45 1.36 1.49 0.37
Kimi-K2 ThinkingAGRAG 11.85 11.28 10.33 13.76 12.07 10.41 10.62 10.37 10.00 11.14 11.18 1.08
GRAG 42.95 38.38 39.74 44.23 51.92 40.57 28.08 32.35 28.46 38.57 38.52 6.97
HRAG 43.41 43.08 37.52 45.06 41.91 39.46 35.55 41.45 37.14 46.81 41.14 3.47
RAG 3.82 3.83 4.29 5.26 4.34 3.94 3.72 3.44 3.53 4.11 4.03 0.50
GPT-4.1-miniAGRAG 4.28 4.41 4.07 4.08 4.51 4.06 4.63 4.73 4.25 4.45 4.35 0.23
GRAG 5.92 6.72 5.09 6.42 5.22 5.22 5.44 7.82 5.78 6.10 5.97 0.80
HRAG 4.76 4.94 4.08 4.47 4.92 4.44 4.73 4.79 4.50 4.37 4.60 0.26
RAG 1.26 1.40 1.17 1.24 1.24 1.24 1.24 1.23 1.15 1.20 1.24 0.06
Mistral Small 24BAGRAG 4.35 4.35 3.85 4.06 3.73 3.94 4.69 3.95 3.72 3.67 4.03 0.32
GRAG 3.02 20.54 0.83 16.37 26.89 10.62 8.77 29.77 20.53 16.23 15.36 9.08
HRAG 4.76 6.63 8.42 5.50 4.33 6.14 7.29 4.81 9.08 4.87 6.18 1.56
RAG 0.83 0.80 0.79 1.23 0.87 0.83 0.86 0.79 0.98 0.73 0.87 0.14
collapse is rare for answerable factual categories, particularly simple and single-hop questions, where valid graph
paths usually exist. AGRAG mitigates some of these failures by exiting invalid query loops earlier, while HRAG further
reduces their impact through fallback retrieval.
Table 17 confirms these findings using surface-form metrics. While absolute values differ due to metric scale, the
trends are consistent: stronger LLMs yield higher baseline GRAG performance, and HRAG provides the most stable
improvements across all model families. Overall, RQ3 shows that text-to-Cypher generation is a critical bottleneck for
graph-based RAG. GRAG performance and failure rates vary widely by underlying LLM; agentic critique primarily
benefits weaker models by correcting query-generation errors; and hybrid redundancy (HRAG) is the most reliable
strategy for mitigating Cypher-induced failures regardless of model strength.
Summary.GRAG fails consistently on unanswerable questions, entering long retry loops instead of recognizing
missing information, while it performs well on simple factual queries where valid graph paths exist. Overall,
text-to-Cypher generation is a key bottleneck, with performance strongly depending on the LLM, while HRAG
provides the most stable and reliable results across models.
4.1.4 Runtime Quality Trade-offs.Table 19 summarizes runtime-quality trade-offs. Classical RAG achieves sub-4s
runtimes with minimal variance but poor performance on complex queries. GRAG exhibits highest cost (>120s) and
instability from repeated correction loops without reliable quality gains. AGRAG stabilizes execution (4 to 11s) with
large quality improvements, yielding best cost-quality balance. HRAG incurs higher cost (up to 47s) from parallel
pipelines but achieves strongest robustness.
Manuscript submitted to ACM

24 Hamzić et al.
Table 19. RQ4 Summary: Runtime-quality trade-offs across RAG systems. Ranges are computed from per-model mean and standard
deviation runtimes in Table 18. Relative cost is computed per model as the ratio of system mean runtime to the corresponding RAG
mean runtime; ranges report the minimum and maximum ratios across evaluated models.
System Mean Runtime (s) Std (s) Cost vs RAG Variance Trade-off Summary
RAG[0.87,4.03] [0.06,0.50] ×1Low Fast but weakest perfor-
mance on complex CTI
queries
AGRAG[4.03,11.18] [0.23,1.55] ×[2.77,7.21]Low–Mod. Best balance: large quality
gains with stable overhead
HRAG[4.60,46.93] [0.26,5.54] ×[3.71,35.55]Moderate Maximum robustness at
higher but predictable cost
GRAG[5.97,127.53] [0.80,38.74] ×[4.81,96.61]High Unfavorable: high cost and
instability
Table 20. Estimated API cost per query (USD) by LLM, retrieval system, and question type. Token budgets are scaled by system
complexity (LLM calls: RAG = 1, GRAG = 3, AGRAG = 3.5, HRAG = 5) and category-dependent reasoning effort. Reasoning tokens are
included only for reasoning-capable models (GPT-5.2, Kimi K2 Thinking). Costs exclude embedding, infrastructure, and database
overhead.
Model System Simple Single-hop Multi-hop Guided Unanswerable
GPT-5.2RAG $0.007 $0.008 $0.011 $0.010 $0.008
GRAG $0.021 $0.028 $0.042 $0.036 $0.025
AGRAG $0.025 $0.031 $0.045 $0.039 $0.029
HRAG $0.039 $0.050 $0.074 $0.065 $0.046
GPT-4.1-miniRAG $0.001 $0.001 $0.001 $0.001 $0.001
GRAG $0.002 $0.002 $0.002 $0.002 $0.002
AGRAG $0.004 $0.004 $0.004 $0.004 $0.004
HRAG $0.005 $0.005 $0.005 $0.005 $0.005
Kimi K2 T.RAG $0.004 $0.004 $0.005 $0.005 $0.004
GRAG $0.010 $0.012 $0.016 $0.015 $0.011
AGRAG $0.013 $0.015 $0.019 $0.017 $0.014
HRAG $0.020 $0.023 $0.030 $0.027 $0.022
Mistral S. 24BRAG $0.0003 $0.0003 $0.0003 $0.0003 $0.0003
GRAG $0.0006 $0.0006 $0.0006 $0.0006 $0.0006
AGRAG $0.0009 $0.0009 $0.0009 $0.0009 $0.0009
HRAG $0.0013 $0.0013 $0.0013 $0.0013 $0.0013
Mixtral 8x7BRAG $0.002 $0.002 $0.002 $0.002 $0.002
GRAG $0.004 $0.004 $0.004 $0.004 $0.004
AGRAG $0.005 $0.005 $0.005 $0.005 $0.005
HRAG $0.008 $0.008 $0.008 $0.008 $0.008
Table 20 (API pricing (per 1M tokens): GPT-5.2 $1.75/$14.00 (input/output), GPT-4.1-mini $0.40/$1.60, Kimi K2
Thinking $1.20/$4.00 (Together AI), Mistral Small 24B $0.10/$0.30 (Together AI), Mixtral 8x7B $0.60/$0.60 (Together AI).
Prices as of January 2026.) estimates per-query API costs across all five models. Cost is dominated by model choice over
retrieval architecture: GPT-5.2 HRAG costs 30 to 57 ×more than Mistral Small HRAG. Cost variation across question
types is substantial only for reasoning models (GPT-5.2, Kimi K2 Thinking) due to category-dependent reasoning
tokens; non-reasoning models show near-uniform per-query costs driven primarily by the number of LLM calls. These
estimates exclude GRAG’s wasted tokens from failed correction loops, making its effective cost per correct answer
substantially higher.
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 25
Table 21. Cross-run quality stability: coefficient of variation (CV) of run-level mean judge scores by system and question category,
computed over 10 independent runs. Lower CV indicates more stable performance across different report samples.
Category RAG GRAG AGRAG HRAG
Simple 13.7% 6.5% 9.1%3.3%
Single-hop 13.2% 15.4% 6.3%1.9%
Multi-hop 24.1% 16.3% 10.9%7.0%
Guided 12.5% 26.5% 15.9%8.7%
Unanswerable 9.2%44.9%10.9% 14.5%
Overall (std of run means)1.76 2.14 1.250.76
RQ4: How large is the runtime-quality trade-off across systems and models?RAG is fastest but performs
poorly on complex queries, while GRAG is slow and unstable without consistent quality gains due to repeated
query loops. AGRAG offers the best balance between cost and performance, and HRAG achieves the highest
robustness at higher but predictable runtime and cost, which are mainly driven by the choice of LLM rather than
the retrieval architecture.
Cross-Run Quality Stability.Beyond runtime stability, operational CTI systems requirequality stabilityacross different
report samples. Table 21 reports the coefficient of variation (CV) of run-level mean judge scores across the 10 independent
evaluation runs, each using a different random sample of 15 CTI reports.
CV is calculated as follows: for each system 𝑠and category 𝑐, let ¯𝑥𝑠,𝑐,𝑟denote the mean judge score for run 𝑟∈
{1,...,10}. The coefficient of variation is
CV𝑠,𝑐=𝜎 ¯𝑥𝑠,𝑐,1,..., ¯𝑥𝑠,𝑐,10
𝜇 ¯𝑥𝑠,𝑐,1,..., ¯𝑥𝑠,𝑐,10×100%(2)
where𝜎(·) and𝜇(·) are the sample standard deviation and mean, respectively. Lower CV indicates more stable
performance across different report samples.
HRAG achieves the lowest CV across all answerable categories (1.9% to 8.7%), indicating that its quality is largely
invariant to the specific reports selected. In contrast, GRAG exhibits extreme instability on unanswerable questions
(CV = 44.9%), meaning that its performance on this safety-critical category fluctuates wildly depending on the graph
structure produced by each report sample. This instability compounds GRAG’s runtime variance (Table 19): not only is
GRAG slow and unpredictable in latency, but thequalityof its outputs is also unpredictable across deployments.
AGRAG substantially stabilizes GRAG’s quality variance (overall std reduced from 2.14 to 1.25), while HRAG achieves
the tightest bounds (overall std = 0.76). For CTI deployments where consistent performance across heterogeneous report
collections is essential, HRAG provides the most reliable quality guarantees.
Figure 8 illustrates GRAG’s catastrophic runtime instability versus AGRAG’s bounded execution. The scatter plot
of 660 paired queries shows that agentic correction not only improves accuracy but can dramatically reduce latency:
median speedup is 3.3×, with 23.6% of queries exceeding 10×speedup and maximum 372×speedup observed. These
cases demonstrate that agentic critique identifies invalid queries in the first correction iteration, whereas GRAG’s
iterative validation becomes trapped attempting variations of fundamentally incorrect query patterns until timeout.
Unanswerable queries (purple points) cluster in GRAG’s extreme timeout region, while AGRAG maintains predictable
sub-60s response times. The trade-off is substantial: RAG minimizes latency but sacrifices quality; GRAG suffers high
cost and instability; AGRAG offers best balance for cost-sensitive deployments; HRAG prioritizes robustness over
Manuscript submitted to ACM

26 Hamzić et al.
101
100101102103
GRAG Runtime (seconds, log scale)101
100101102103AGRAG Runtime (seconds, log scale)372× faster!
(2348s  6.3s)
Speedup Statistics:
Median: 3.3×
Mean: 13.7×
>10×: 156/660 (23.6%)
>100×: 19/660 (2.9%)AGRAG Speedup Over GRAG
(points below diagonal = AGRAG faster)
simple
single-hop
multi-hop
guidedunanswerable
Equal time
10× speedup
100× speedup
Fig. 8. GPT-5.2 Performance Metrics
Fig. 9. AGRAG speedup over GRAG on 660 paired queries. Each point represents one query; points below the diagonal indicate
AGRAG faster than GRAG. Median speedup is 3.3×, with 156 queries (23.6%) exceeding 10×speedup and maximum 372×speedup
observed. Purple points (unanswerable queries) cluster in GRAG’s extreme timeout region, while AGRAG maintains bounded latency
through critique-based early exit.
latency. GRAG’s extreme variance is intrinsic to naive graph-only designs, representing practical deployment risk in
time-sensitive CTI workflows.
Summary.HRAG achieves the most stable performance across different report samples, while GRAG shows
extreme instability, especially on unanswerable queries, making both its runtime and output quality unpredictable.
AGRAG significantly reduces this instability and improves latency, but overall there is a clear trade-off: RAG is
fastest but weakest, GRAG is unreliable, AGRAG balances cost and quality, and HRAG provides the most robust
and consistent results.
4.2 Representative Failure Cases
To illustrate the operational implications of our quantitative findings, we present three representative failure cases
extracted from evaluation logs, demonstrating the real-world consequences of different retrieval architectures.
Case 1: Agentic Correction Recovers with Dramatic Speedup.In this case, the system is asked to identify the CVE
exploited in the APT37 Internet Explorer incident, which requires mapping the incident to its associated CVE in the
graph schema. This 147×speedup (736s →5s) (see Table 22) demonstrates that agentic correction not only improves
accuracy butreduceslatency by detecting errors before entering prolonged retry loops. The critique mechanism
prevented the pathological behavior observed in GRAG, where iterative validation attempts variations of an invalid
query pattern until timeout.
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 27
Table 22. Failure Case 1.
QuestionWhich CVE was exploited in the APT37 Internet Explorer incident?
GoldCVE-2022-41128
GRAG (736.4s)[Empty response after 12+ minutes]
Failure Generated a Cypher query with an incorrect relationship name. Subsequent retry attempts continued
searching for a schema pattern that does not exist, exhausting 25 correction iterations.
AGRAG (4.9s)“CVE-2022-41128”✓
Mechanism Critique loop identified relationship schema mismatch in the first iteration and regenerated a valid
query.
Table 23. Failure Case 2.
QuestionWhat CVSS score is recorded for CVE-2024-21338?
Gold[Information not available in reports]
RAG (1.2s)“The CVSS score recorded for CVE-2024-21338 is 9.1”
Safety failureConfidently hallucinated a specific numerical value that is not present in the source documents.
GRAG (520.4s)[Empty response after 8.7-minute timeout]
Partial success Did not produce an answer, but only after prolonged latency while searching for a non-existent CVE
property in the graph.
HRAG (15.3s)“I don’t know”✓
Correct behavior Correctly identified missing information through cross-validation: the graph query returned no CVE
node with a CVSS property and text retrieval found no CVSS mention.
Case 2: Hybrid Retrieval Enables Safe Abstention.This case (see Table 23) exemplifies the safety-latency trade-off
inherent in different retrieval architectures. RAG provides immediate but false information, prioritizing speed over
correctness. GRAG eventually abstains but only after unacceptable latency. Only HRAG achieves both rapid response
(15s) and correct refusal through redundant evidence paths.
Case 3: Query Correction Death Spiral.The most extreme failure occurred on the unanswerable query “What exact
GoAnywhere MFT version is vulnerable to CVE-2024-0204?” GRAG entered a pathological correction loop lasting 2,348
seconds (39.1 minutes), iteratively generating invalid Cypher queries searching for a property that does not exist in the
schema, before timing out with an empty result.
Detailed log analysis reveals the failure mechanism:
(1) Initial Cypher query searches forCVE.affected_versionproperty
(2) Query returns empty result (property does not exist in schema)
(3) Correction loop attempts alternative property names:version,vulnerable_version,product_version
(4) All attempts fail validation or return empty results
(5) Process repeats for 25 correction iterations over 39 minutes
(6) System finally returns empty response after timeout
For operational CTI settings, where analysts often require near real-time responses during incident triage, a response
latency of 39 minutes for a non-answer ("information not available") is operationally impractical. This is not an
implementation artifact but a fundamental limitation: without explicit uncertainty modeling, schema-constrained
generation cannot represent “unknown” and instead exhausts retry budgets searching for impossible graph paths.
Comparison to other systems:On this same query, RAG hallucinated a specific version number in 1.3s (fast but
wrong), AGRAG also produced an answer instead of abstaining (fast but unsafe), and HRAG correctly abstained in 18.7s
(acceptable latency, correct refusal). Only GRAG exhibited the timeout behavior.
Manuscript submitted to ACM

28 Hamzić et al.
Table 24. Adversary attack success rates (% of questions scoring ≤15/50) by strategy and system. Higher values indicate greater
vulnerability. Bold marks the most effective attack per system.
Attack Strategy RAG GRAG AGRAG HRAG
Single-category strategies
A1: Schema evasion (guided) 20.6%82.5%18.1% 3.1%
A2: Unanswerable probing 6.0% 46.0% 4.0%20.0%
A3: Multi-hop stress 47.3% 16.7% 18.7% 12.0%
A4: Simple fact (control)52.0%11.3% 12.0% 6.7%
Compound strategies
A5: Multi-entity + guided 20.8% 83.0% 18.2% 3.1%
A6: Long questions (>75th pctl) 21.5% 81.6% 17.8% 3.1%
A7: Aggregate multi-hop 44.4% 15.3%26.4% 22.2%
Random (any) 37.0% 33.3% 13.2% 6.7%
4.3 Adversary Model: Systematic Failure Analysis
To move beyond aggregate performance metrics, we construct a structured adversary model that systematically
characterizes how a strategic adversary, or equivalently, a challenging real-world query distribution, can exploit failure
modes in each RAG architecture. This analysis is relevant for CTI deployment, where adversaries may craft queries to
induce incorrect analyst decisions, and where naturally occurring edge cases (ambiguous, underspecified, or out-of-scope
questions) produce equivalent failure patterns.
4.3.1 Threat Model.We consider an adversary whose goal is to induce decision error in a CTI analyst who relies on a
RAG-based assistant. The adversary can influence the natural-language query but cannot modify the knowledge base,
the retrieval infrastructure, or the LLM. We define a successful attack as inducing a response with a judge score ≤15/50,
indicating a substantially wrong, unfaithful, or unhelpful answer. The adversary may have varying levels of knowledge:
•Black-box:The adversary knows only that a RAG system is deployed but not which architecture variant.
•Grey-box:The adversary knows which architecture variant is deployed (RAG, GRAG, AGRAG, or HRAG).
•White-box:The adversary additionally knows the graph schema and can craft queries that target schema gaps.
4.3.2 Attack Strategies and Success Rates.We evaluate four attack strategies derived from the question taxonomy, plus
three compound strategies combining question properties. Table 24 reports the conditional probability of attack success
(judge score≤15) under each strategy.
Key findings.Each system exhibits a distinct weakness that a grey-box adversary can exploit:
•RAGis most vulnerable to simple fact lookups (A4: 52.0% attack success), where semantic retrieval fails to
surface the specific entity or property. This counterintuitive result, simple questions being harder than complex
ones, arises because RAG’s top- 𝑘chunk retrieval frequently misses short, precise facts embedded in longer
narrative passages.
•GRAGis catastrophically vulnerable to schema evasion (A1: 82.5%), where guided analyst-style questions
reference concepts outside the graph schema. The attack success rate increases further with multi-entity questions
(A5: 83.0%) and long questions (A6: 81.6%), as these amplify the probability of referencing at least one out-of-
schema concept.
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 29
Table 25. Failure mode classification across RAG systems. Modes are defined by judge criterion scores: Hallucination = C3 ≤2 and
C4≥3 (fluent but unfaithful); Full Collapse = C1, C2, C3 all≤2.
Failure Mode RAG GRAG AGRAG HRAG
Correct (≥40) 41.5% 61.1% 74.2% 83.2%
Partial (10–40) 31.7% 12.3% 14.1% 12.0%
Near-zero (≤10) 26.8% 26.7% 11.7% 4.8%
Hallucination 45.6% 28.2% 9.7% 11.7%
Full collapse 41.1% 33.0% 15.8% 8.8%
Table 26. GRAG timing as a failure signal. Response time
>𝑇seconds predicts a failed answer (score ≤10). Preci-
sion = % of slow responses that are failures; Recall = % of
failures that are slow.
Threshold𝑇(s)𝑛(above) Precision Recall
30 292 50% 84%
60 188 56% 60%
120 135 61% 47%
300 73 70% 29%
500 48 77% 21%Table 27. Ensemble defense: failure rates under oracle selection
(best score among the subset of systems). Fail = judge score≤10.
System configuration Mean score Fail Perfect
RAG alone 29.4 26.8% 31.7%
GRAG alone 34.1 26.7% 55.0%
AGRAG alone 40.7 11.7% 69.5%
HRAG alone 44.7 4.8% 74.8%
Best(AGRAG, HRAG) 47.9 0.9% 86.5%
Best(RAG, HRAG) 46.6 1.5% 79.4%
Best(all four) 49.0 0.0% 90.2%
•AGRAGis most vulnerable to aggregate multi-hop queries (A7: 26.4%), which require compositional Cypher
(e.g.,COUNT,COLLECT) that agentic critique cannot always repair.
•HRAGachieves the lowest vulnerability across most strategies, with its main weakness being unanswerable
probing (A2: 20.0%) and aggregate multi-hop (A7: 22.2%).
A grey-box adversary gains substantial advantage over random querying. For GRAG, a targeted attack using guided
questions raises the attack success rate from 33.3% (random) to 82.5% (schema evasion), a 2.5 ×amplification. For RAG,
targeting simple facts increases success from 37.0% to 52.0% (1.4 ×). HRAG is the most resistant to strategic targeting:
even the best attack (A2/A7) achieves only 20-22%, compared to 6.7% under random queries (3.0 ×amplification), making
it the hardest system to exploit.
4.3.3 Failure Mode Classification.Beyond binary success/failure, Table 25 classifies outputs into distinct failure modes.
RAG fails primarily through hallucination (45.6% of all answers are fluent but unfaithful), while GRAG fails through
full collapse (33.0% with all correctness criteria ≤2), typically caused by empty Cypher results or timeout. AGRAG and
HRAG reduce both failure modes substantially, with HRAG achieving the lowest rates across all categories.
Critically, the two dominant failure modes, hallucination and collapse, require different mitigations. Hallucination
(RAG’s primary mode) requires improved retrieval precision or faithfulness constraints. Collapse (GRAG’s primary
mode) requires bounded execution and fallback paths. AGRAG addresses collapse through critique-based early exit;
HRAG addresses both through redundant evidence paths.
4.3.4 Timing Side Channel.GRAG’s iterative query-correction loops create a measurable timing side channel: failed
queries take significantly longer than successful ones (mean 292.6s vs. 45.0s, a 6.5 ×ratio). Table 26 shows that GRAG
response time can serve as a failure detector: at a 60-second threshold, 56% of slow responses are failures (precision)
and 60% of all failures are slow (recall). At 300 seconds, precision reaches 70%. This timing correlation has two security
implications. First, an adversary observing response latency can infer whether the system failed to answer, even without
Manuscript submitted to ACM

30 Hamzić et al.
Table 28. Failure decorrelation: Pearson correlation of binary failure
indicators (score≤15 = fail).
RAG GRAG AGRAG HRAG
RAG 1.000−0.215 0.008 0.034
GRAG 1.000 0.105−0.034
AGRAG 1.000 0.075
HRAG 1.000Table 29. Jaccard overlap of failure sets (shared / union).
Pair Jaccard
RAG–GRAG 0.118
AGRAG–HRAG 0.083
RAG–HRAG 0.071
GRAG–HRAG 0.048
Table 30. Correlation between question-level features and failure (score ≤15). Positive values indicate the feature increases failure
risk.
Feature RAG GRAG AGRAG HRAG
Question length (words)−0.19+0.58+0.09−0.07
Multi-entity (and/or/all)−0.17+0.52+0.05−0.09
Gold answer length (words)−0.16+0.54+0.05−0.07
Contains “which”−0.15+0.22−0.04−0.17
Contains “how many”+0.01−0.05+0.13+0.17
Has temporal reference+0.08−0.03+0.00−0.01
seeing the response content. Second, the timing signal can be exploited defensively: a deployment could impose a hard
timeout and return “insufficient information” rather than waiting for a potentially low-quality answer after prolonged
retry loops. Notably, the other systems do not exhibit this vulnerability. RAG shows no timing-failure correlation
(𝑟=−0.09), HRAG shows negligible correlation ( 𝑟=−0.02), and AGRAG’s correlation is moderate ( 𝑟=+0.30) but with
bounded absolute latency (failure mean: 17.9s vs. success: 8.3s).
4.3.5 Failure Decorrelation and Ensemble Defense.A key question for defensive deployment is whether system failures
are correlated, if all systems fail on the same questions, redundancy provides no benefit. Table 28 shows that failures
are largely decorrelated across architectures. The Pearson correlation between RAG and GRAG failure indicators is
negative (𝑟=−0.22), meaning they tend to fail on different questions. AGRAG-HRAG failure correlation is low ( 𝑟=0.08),
and all cross-architecture Jaccard overlaps are below 0.15 (Table 29).
This decorrelation enables effective ensemble defense. Table 27 shows that oracle selection between AGRAG and
HRAG reduces the failure rate from 4.8% (HRAG alone) to 0.9%, while raising the perfect-score rate to 86.5%. The oracle
over all four systems achieves zero failures (0.0%) and 90.2% perfect scores. While oracle selection requires a reliable
meta-classifier, these results establish an upper bound for ensemble-based defenses and demonstrate that the residual
failure surfaces of different architectures are largely complementary.
4.3.6 Question Features that Predict Failure.To characterize which question properties a white-box adversary could
exploit, Table 30 reports correlations between surface-level question features and failure. Two patterns emerge. First,
RAG and GRAG exhibit inverted vulnerability profiles: question length, multi-entity references, and complex gold
answersreduceRAG failure risk (negative correlation) butstrongly increaseGRAG failure risk ( 𝑟=+ 0.52to+0.58). This
occurs because longer, more complex questions tend to be guided analyst-style questions, which RAG handles through
broad retrieval but GRAG cannot express within its schema.
Second, aggregate queries (“how many”, “count”) are the strongest predictors of AGRAG and HRAG failure ( 𝑟=+ 0.13
and+0.17), reflecting the difficulty of generating correct aggregate Cypher even with agentic correction.
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 31
Table 31. Adversary model summary: primary vulnerability, dominant failure mode, best-case attack success rate, and recommended
mitigation per system.
System Primary Vulnerabil-
ityFailure Mode Best Attack Mitigation
RAG Simple fact lookup Hallucination (45.6%) 52.0% Improve retrieval precision
GRAG Schema evasion Full collapse (33.0%) 82.5% Bounded timeout + fallback
AGRAG Aggregate queries Partial failure (14.1%) 26.4% Aggregate query templates
HRAG Unanswerable probing Low (8.8% collapse) 22.2% Explicit abstention logic
These inverted profiles confirm that a white-box adversary must tailor attacks to the deployed architecture: long,
multi-entity questions attack GRAG; short, specific fact queries attack RAG; aggregate counting queries stress AGRAG
and HRAG.
4.3.7 Adversary Model Summary.Table 31 consolidates the adversary analysis. The four architectures exhibit funda-
mentally different failure profiles, which has two implications for CTI deployment:
(1)No single system is robust to all attack strategies.RAG is vulnerable to fact-seeking queries, GRAG to
out-of-schema queries, AGRAG to aggregate reasoning, and HRAG to unanswerable probing. Deployment
decisions should match the expected query distribution to the system’s strength profile.
(2)Failure decorrelation enables ensemble defense.Because architectures fail on different questions (Jaccard
overlap <0.15), combining even two systems (e.g., AGRAG + HRAG) reduces the failure rate from 4.8% to
0.9%. A meta-classifier that routes queries to the most appropriate system, or a voting mechanism that detects
disagreement, could approach the oracle upper bound of 0.0% failure.
These findings extend this work’s security-relevant failure analysis from descriptive observation to a predictive
adversary model, providing actionable guidance for hardening CTI assistants against both strategic attacks and naturally
occurring edge cases.
5 Discussion
In this section we present this paper’s overview, analyses and the findings.
5.1 Overview
This study investigates whether graph-based retrieval architectures improve question answering over CTI reports.
Across 3,300 questions and five language models, the results reveal a consistent pattern: while explicit graph grounding
can significantly improve performance for structured factual queries, naive graph-only pipelines introduce new failure
modes that can outweigh their advantages. In practice, the benefits of graph retrieval only emerge when combined with
agentic query correction or hybrid retrieval redundancy.
5.2 Analyses
Graph grounding provides clear advantages for fact-centric CTI questions, particularly those requiring relational
reasoning. For simple, single-hop, and multi-hop questions, AGRAG and HRAG consistently outperform vanilla
semantic retrieval (Table 11). These improvements are most pronounced for multi-hop reasoning, where answers
depend on chaining relationships across entities. In such cases, semantic retrieval often fails to retrieve all relevant
passages simultaneously, whereas graph traversal can directly follow relational paths. However, graph grounding alone
Manuscript submitted to ACM

32 Hamzić et al.
Table 32. SWOT analysis of evaluated RAG architectures for CTI question answering.
System Strengths Weaknesses Opportunities Threats
RAG Fast and low-cost retrieval;
robust to schema gaps; sta-
ble runtime behavior.High hallucination rates;
weak multi-hop reasoning;
sensitive to top- 𝑘retrieval
failures.Improved reranking or hy-
brid retrieval could enhance
factual recall.Vulnerable to simple fact
queries and overconfident
answers for missing infor-
mation.
GRAG Explicit relational rea-
soning over CTI entities;
strong factual lookup when
Cypher queries succeed.Highly sensitive to text-to-
Cypher errors; latency in-
stability due to retry loops.Improved text-to-query
models and bounded
correction strategies could
stabilize performance.Schema evasion attacks and
catastrophic collapse on
unanswerable queries.
AGRAG Agentic critique improves
Cypher reliability; large ac-
curacy gains over classical
RAG; reduces query fail-
ures.Higher latency than RAG;
dependent on graph
schema coverage.Advanced agentic planning
and better query validation
could further improve ro-
bustness.Aggregate queries and
schema gaps remain chal-
lenging.
HRAG Best overall robustness;
combines graph reasoning
with semantic retrieval;
lowest hallucination rates.Higher computational cost
and system complexity.Adaptive routing or ensem-
ble strategies could reduce
cost while preserving ro-
bustness.Incorrect pipeline balanc-
ing or infrastructure con-
straints may limit deploy-
ment.
is not sufficient. Plain GraphRAG yields only marginal average improvements despite producing more perfect answers.
This indicates that the reliability of the text-to-query translation layer is a critical bottleneck for graph-based retrieval.
Graph-only retrieval introduces failure modes that are largely absent in semantic RAG systems. Because answering
requires translating natural language into executable Cypher queries, errors in query generation, schema matching,
or result interpretation can cause the system to fail. These failures are particularly evident for questions outside the
graph schema. Guided analyst-style queries often reference contextual information not captured in the structured
graph, leading GRAG to produce incorrect answers or enter prolonged query-repair loops. Similarly, unanswerable
questions expose a structural limitation: when no supporting graph path exists, graph-only systems tend to repeatedly
generate alternative queries rather than abstaining. We refer to this phenomenon asstructural hallucination, where the
system produces answers consistent with the graph query output but not supported by the underlying reports due to
incomplete schema coverage.
5.3 Findings
Our results demonstrate that architectural safeguards are essential for reliable graph-based retrieval. Agentic correction
substantially reduces query-generation failures by detecting and repairing invalid Cypher queries early. Hybrid retrieval
provides an even stronger mitigation by combining graph traversal with semantic evidence retrieval, allowing the
system to fall back on unstructured context when structured queries fail. From a system design perspective, these
findings suggest that graph retrieval should not replace semantic retrieval but complement it within a broader retrieval
architecture. For operational CTI workflows, the primary risk of AI assistants is not average performance but catastrophic
failures on edge cases. Graph-only retrieval is particularly vulnerable to such failures, as schema gaps and query-
generation errors can produce unstable or delayed responses. Hybrid architectures offer a more robust solution by
combining complementary evidence sources and enabling safer abstention when information is missing.
To synthesize the architectural trade-offs observed in our evaluation, Table 32 summarizes the strengths, weaknesses,
opportunities, and threats (SWOT) of the four evaluated retrieval architectures.
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 33
5.4 Recommendations
These findings suggest several practical recommendations. For system deployment, hybrid retrieval pipelines should
be preferred over graph-only architectures, and systems should include safeguards such as bounded query-repair
loops and explicit abstention mechanisms. For research, the main challenge lies in improving the reliability of text-to-
query generation and developing evaluation benchmarks that capture realistic failure modes such as schema gaps and
unanswerable queries.
5.5 Limitations and Future Work
This study focuses on a single CTI dataset and a controlled graph schema derived from CTINexus reports. Different
datasets or denser relational graphs may lead to different trade-offs between structured and semantic retrieval. Addition-
ally, the evaluation uses automatically generated questions, which may not fully capture the diversity of analyst queries.
Future work should therefore include human analyst studies, improved uncertainty modeling for safe abstention, and
robustness evaluations under incomplete or adversarial CTI data. Another promising direction is the development
of adaptive routing or ensemble architectures that dynamically select or combine retrieval pipelines based on query
characteristics, potentially reducing computational cost while preserving the robustness benefits of hybrid retrieval.
6 Conclusion
We presented a controlled evaluation of four RAG architectures for cyber threat intelligence: vanilla semantic RAG,
graph-only GraphRAG, agentic GraphRAG, and a hybrid graph+text system (HRAG). Across 3,300 CTI QA pairs and
five LLMs, we find that explicit graph grounding alone is not a reliable upgrade: plain GraphRAG yields only marginal
quality gains and exhibits severe latency instability due to query-repair loops. In contrast, agentic refinement and
hybrid redundancy deliver large, consistent improvements, especially for structured fact-seeking and multi-hop queries,
while also reducing catastrophic text-to-Cypher failures. We further show that the effectiveness of graph-based retrieval
is strongly model-dependent and that safe behavior on unanswerable queries remains a key challenge for graph-centric
systems.
Our evaluation logs reveal that the primary risk of graph-based CTI retrieval is not average-case performance
degradation, but catastrophic failure modes on edge cases: 100% collapse rate on unanswerable queries, multi-minute
timeout loops (up to 39 minutes observed), and overconfident answers from partial graph structure. These findings
demonstrate that graph grounding alone is insufficient - reliable CTI assistants require explicit uncertainty modeling,
bounded correction mechanisms (AGRAG achieves up to 147×speedup over GRAG), or hybrid redundancy (HRAG
achieves 76% correct refusal vs. 0% for the semantic RAG) to ensure safe operation under realistic query distributions.
Overall, our results suggest that graph-based retrieval should be treated as a high-risk component in CTI assistants:
it requires agentic correction mechanisms and/or parallel unstructured text retrieval, like in the HRAG, to realize
its benefits without sacrificing robustness. Future work should incorporate analyst-in-the-loop evaluation, explicit
abstention and evidence-coverage checks, and robustness testing under realistic data quality issues and adversarial
conditions.
Manuscript submitted to ACM

34 Hamzić et al.
Acknowledgments
Funded by the European Union under the Horizon Europe Research and Innovation programme (GA no. 101168144 -
MIRANDA). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those
of the European Union. Neither the European Union nor the granting authority can be held responsible for them.
References
[1] Renzo Angles and et al. 2018. Foundations of Modern Graph Query Languages.Comput. Surveys50, 5 (2018), 1–40. doi:10.1145/3104031
[2]Anonymous. 2025. A Self-Correcting Agentic Graph RAG Framework for Clinical Decision Support.Frontiers in Medicine(2025). https:
//www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1716327/full Agentic RAG for clinical QA built on hepatology knowledge graph;
significantly outperforms baseline RAG and GraphRAG.
[3]Akari Asai and et al. 2020. Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering. InInternational Conference on
Learning Representations (ICLR).
[4]Authors. 2024. KnowCTI: Knowledge-based cyber threat intelligence entity and relation extraction.Journal of Computers & Security(2024).
https://www.sciencedirect.com/science/article/pii/S0167404824001251 Example CTI ontology with entities and relations.
[5]Victor Barres and et al. 2025. 𝜏2-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982 [cs.AI] https:
//arxiv.org/abs/2506.07982
[6]Kathrin Blagec and et al. 2022. A global analysis of metrics used for measuring performance in natural language processing. InProceedings of NLP
Power! The First Workshop on Efficient Benchmarking in NLP. Association for Computational Linguistics, Dublin, Ireland, 52–63. doi:10.18653/v1/2022.
nlppower-1.6
[7] Yutong Cheng and et al. 2025. CTINexus: Automatic Cyber Threat Intelligence Knowledge Graph Construction Using Large Language Models. In
2025 IEEE 10th European Symposium on Security and Privacy (EuroS&P). 923–938. doi:10.1109/EuroSP63326.2025.00057
[8]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa
Ness, and Jonathan Larson. 2025. From Local to Global: A GraphRAG Approach to Query-Focused Summarization. arXiv:2404.16130 [cs.CL]
arXiv:2404.16130v2.
[9] FAIR CodeGen team and et al. 2025.CWM: An Open-Weights LLM for Research on Code Generation with World Models. arXiv:2510.02387 [cs.SE]
[10] Nadime Francis and et al. 2018. Cypher: An Evolving Query Language for Property Graphs. InProceedings of the 2018 International Conference on
Management of Data (SIGMOD ’18). Association for Computing Machinery. doi:10.1145/3183713.3190657
[11] Jinlan Fu and et al. 2023. GPTScore: Evaluate as You Desire.arXiv preprint arXiv:2302.04166(2023).
[12] Yunfan Gao and et al. 2023. Retrieval-Augmented Generation for Large Language Models: A Survey.arXiv preprint arXiv:2312.10997(2023).
[13] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL] doi:10.48550/arXiv.2312.10997
[14] Jiawei Gu and et al. 2025.A Survey on LLM-as-a-Judge. arXiv:2411.15594 [cs.CL] https://arxiv.org/abs/2411.15594
[15] Dzenan Hamzic and et al. 2025. Enhancing Cyber Situational Awareness with AI: A Novel Pipeline Approach for Threat Intelligence Analysis and
Enrichment. InAvailability, Reliability and Security. Springer Nature Switzerland, Cham, 44–62.
[16] Yuqing Huang and et al. 2025. SelfAug: Mitigating Catastrophic Forgetting in Retrieval-Augmented Generation via Distribution Self-Alignment. In
Findings of the Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguistics, Suzhou, China, 14175–14190.
doi:10.18653/v1/2025.findings-emnlp.763
[17] Ziwei Ji and et al. 2022. Survey of Hallucination in Natural Language Generation.Comput. Surveys(2022). arXiv:2202.03629 [cs.CL] doi:10.1145/3571730
[18] Ziyan Jiang and et al. 2024. LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs. arXiv:2406.15319 [cs.CL] https:
//arxiv.org/abs/2406.15319
[19] Jeff Johnson and et al. 2017. Billion-Scale Similarity Search with GPUs.arXiv preprint arXiv:1702.08734(2017). doi:10.48550/arXiv.1702.08734
[20] Vladimir Karpukhin and et al. 2020. Dense Passage Retrieval for Open-Domain Question Answering. InProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics. doi:10.18653/v1/2020.emnlp-main.550
[21] Tom Kwiatkowski and et al. 2019. Natural Questions: A Benchmark for Question Answering Research.Transactions of the Association for
Computational Linguistics7 (2019), 453–466.
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural
Information Processing Systems, Vol. 33. arXiv:2005.11401 [cs.CL] doi:10.48550/arXiv.2005.11401
[23] Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, and Ting Wang. 2025. GraphRAG under Fire.
arXiv:2501.14050 [cs.LG] arXiv:2501.14050v1.
[24] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. InText Summarization Branches Out. Association for Computational
Linguistics, Barcelona, Spain, 74–81. https://aclanthology.org/W04-1013/
[25] Yang Liu and et al. 2023. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.arXiv preprint arXiv:2303.16634(2023).
Manuscript submitted to ACM

Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 35
[26] Anket Mehra and et al. 2025. Improving Applicability of Deep Learning based Token Classification models during Training. arXiv:2504.01028 [cs.CV]
https://arxiv.org/abs/2504.01028
[27] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. InProceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing (EMNLP).
[28] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. 2024. Unifying Large Language Models and Knowledge Graphs: A
Roadmap. arXiv:2306.08302 [cs.CL] arXiv:2306.08302v3.
[29] George Papageorgiou, Vangelis Sarlis, Manolis Maragoudakis, and Christos Tjortjis. 2025. Hybrid Multi-Agent GraphRAG for E-Government:
Towards a Trustworthy AI Assistant.Applied Sciences15, 11 (2025), 6315. doi:10.3390/app15116315
[30] Boci Peng and et al. 2025. Graph Retrieval-Augmented Generation: A Survey.ACM Transactions on Information Systems44, 2 (2025), 1–52.
doi:10.1145/3777378
[31] Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know What You Don’t Know: Unanswerable Questions for SQuAD. InProceedings of
the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics.
doi:10.18653/v1/P18-2124
[32] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In
Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2383–2392.
[33] Injy Sarhan and Marco Spruit. 2021. Open-CyKG: An Open Cyber Threat Intelligence Knowledge Graph.Knowledge-Based Systems(2021).
doi:10.1016/j.knosys.2021.107524
[34] Noah Shinn and et al. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. InAdvances in Neural Information Processing Systems,
Vol. 36. https://arxiv.org/abs/2303.11366
[35] Haitian Sun and et al. 2019. PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text. InProceedings of the
2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP). Association for Computational Linguistics, Hong Kong, China, 2380–2390. doi:10.18653/v1/D19-1242
[36] C. J. van Rijsbergen. 1979.Information Retrieval. Butterworth-Heinemann.
[37] Zhitong Wang and et al. 2025. Document Segmentation Matters for Retrieval-Augmented Generation. InFindings of the Association for Computational
Linguistics: ACL 2025. Association for Computational Linguistics, Vienna, Austria, 8063–8075. doi:10.18653/v1/2025.findings-acl.422
[38] Junde Wu and et al. 2025. Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation. In
Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Long Papers). Association for Computational Linguistics,
Vienna, Austria, 28443–28467. doi:10.18653/v1/2025.acl-long.1381
[39] Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and Jinsong Su. 2025. When to use Graphs in RAG: A
Comprehensive Analysis for Graph Retrieval-Augmented Generation. arXiv:2506.05690 [cs.CL] arXiv:2506.05690v2.
[40] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A
Dataset for Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing. Association for Computational Linguistics. doi:10.18653/v1/D18-1259
[41] Shunyu Yao and et al. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629 [cs.CL] doi:10.48550/arXiv.2210.03629
[42] Qinggang Zhang and et al. 2025. When to Use Graphs in Retrieval-Augmented Generation: A Comprehensive Analysis for Graph Retrieval-Augmented
Generation.arXiv preprint arXiv:2506.05690(2025). https://arxiv.org/abs/2506.05690
[43] Tianyi Zhang and et al. 2020. BERTScore: Evaluating Text Generation with BERT. arXiv:1904.09675 [cs.CL] https://arxiv.org/abs/1904.09675
[44] Lianmin Zheng and et al. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.arXiv preprint arXiv:2306.05685(2023).
[45] Xiangrong Zhu and et al. 2025. Knowledge Graph-Guided Retrieval Augmented Generation. InProceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2025) (NAACL ’25). Association for
Computational Linguistics, Albuquerque, New Mexico, USA, 8912–8924. doi:10.18653/v1/2025.naacl-long.449
Received XX XX XXXX; revised XX XX XXXX; accepted XX XX XXXX
Manuscript submitted to ACM