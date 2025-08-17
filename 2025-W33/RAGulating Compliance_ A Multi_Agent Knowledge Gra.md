# RAGulating Compliance: A Multi-Agent Knowledge Graph for Regulatory QA

**Authors**: Bhavik Agarwal, Hemant Sunil Jomraj, Simone Kaplunov, Jack Krolick, Viktoria Rojkova

**Published**: 2025-08-13 15:51:05

**PDF URL**: [http://arxiv.org/pdf/2508.09893v1](http://arxiv.org/pdf/2508.09893v1)

## Abstract
Regulatory compliance question answering (QA) requires precise, verifiable
information, and domain-specific expertise, posing challenges for Large
Language Models (LLMs). In this work, we present a novel multi-agent framework
that integrates a Knowledge Graph (KG) of Regulatory triplets with
Retrieval-Augmented Generation (RAG) to address these demands. First, agents
build and maintain an ontology-free KG by extracting subject--predicate--object
(SPO) triplets from regulatory documents and systematically cleaning,
normalizing, deduplicating, and updating them. Second, these triplets are
embedded and stored along with their corresponding textual sections and
metadata in a single enriched vector database, allowing for both graph-based
reasoning and efficient information retrieval. Third, an orchestrated agent
pipeline leverages triplet-level retrieval for question answering, ensuring
high semantic alignment between user queries and the factual
"who-did-what-to-whom" core captured by the graph. Our hybrid system
outperforms conventional methods in complex regulatory queries, ensuring
factual correctness with embedded triplets, enabling traceability through a
unified vector database, and enhancing understanding through subgraph
visualization, providing a robust foundation for compliance-driven and broader
audit-focused applications.

## Full Text


<!-- PDF content starts -->

RAGulating Compliance: A Multi-Agent Knowledge Graph for
Regulatory QA
Bhavik Agarwal, Hemant Sunil Jomraj, Simone Kaplunov, Jack Krolick, Viktoria Rojkova
MasterControl AI Research
{bagarwal,hjomraj,skaplunov, jkrolick,vrojkova }@mastercontrol.com
Abstract
Regulatory compliance question answering (QA) requires precise, verifiable information, and domain-
specific expertise, posing challenges for Large Language Models (LLMs). In this work, we present a novel
multi-agent framework that integrates a Knowledge Graph (KG) of Regulatory triplets with Retrieval-
Augmented Generation (RAG) to address these demands. First, agents build and maintain ontology-free
KG by extracting subject–predicate–object (SPO) triplets from regulatory documents and systemati-
cally cleaning, normalizing, deduplicating, and updating them. Second, these triplets are embedded and
stored along with their corresponding textual sections and metadata in a single enriched vector database,
allowing for both graph-based reasoning and efficient information retrieval. Third, an orchestrated agent
pipeline leverages triplet-level retrieval for question answering, ensuring high semantic alignment between
user queries and the factual ’who-did-what-to-whom’ core captured by the graph. Our hybrid system
outperforms conventional methods in complex regulatory queries, ensuring factual correctness with em-
bedded triplets, enabling traceability through a unified vector database, and enhancing understanding
through subgraph visualization, providing a robust foundation for compliance-driven and broader audit-
focused applications.
1 Introduction
The growing regulatory complexities in healthcare, pharmaceuticals, and medical devices shape market
access and patient care [HCB24]. The extensive guidance and rules of the FDA [FDA25] require strict
compliance with approvals, post-market surveillance, and quality systems [CDW22]. Meanwhile, LLMs
such as GPT-o1 [Z+24], Qwen-2.5 [Y+24] and Pi-4 [A+24] excel in text tasks but face unique challenges in
precision, verifiability, and domain specialization in high-stakes regulatory contexts [WZ24]. Hallucination
risks and limited contextual understanding underscore the need for robust guardrails, particularly in safety-
critical applications [H+24a], [L+24b]. How can we ensure the domain specificity and reliability required for
compliance?
Our work proposes a three-fold innovation for regulated compliance: first, we construct and refine triplet
graphs from regulatory documents, building on knowledge graph research [N+15]; second, we integrate
these graphs with RAG techniques, inspired by open-domain QA [L+21] and healthcare question-answering
[Y+25], to reduce hallucinations [J+24]; and third, a multi-agent architecture oversees graph construction,
RAG database enrichment, and the final question-answering process, ultimately grounding responses in
factual relationships to enhance precision, reliability, and verifiability—key for demonstrating compliance to
regulators and stakeholders.
2 Relevant Work
A line of research tackles hallucination and domain-specific gaps by integrating language models with knowl-
edge graphs (KGs), which encode domain knowledge for semantic linking, inference, and consistency checks
[H+21, N+15, C+20]. In regulatory settings, KGs capture complex relationships among rules and guidelines
[C+24], and when combined with retrieval-augmented generation (RAG) [L+21], reduce factual errors by
putting outputs in authoritative data [L+24a]. Although RAG has excelled in open-domain QA [L+21, K+20],
1arXiv:2508.09893v1  [cs.AI]  13 Aug 2025

its application in regulatory compliance, particularly synthesizing structured (KG) and unstructured text,
remains underexplored. Multi-agent systems [SLB08, Woo09] offer autonomous agents for data ingestion, KG
construction, verification, and inference, enabling modularity and scalability [Wei00, Z+13]. This approach
is well suited to dynamic regulatory environments that require constant updates.
2.1 Knowledge Graphs in Regulatory Compliance
Knowledge graphs excel at representing complex regulatory information, facilitating semantic relationships
[H+21]. Notable examples include enterprise KGs for market regulations [Ers23] and frameworks for med-
ical device policies [C+24], while [X+25] underscores KG reasoning techniques that bridge structured and
unstructured data.
2.2 Retrieval-Augmented Generation in Regulatory Compliance
RAG [L+21] integrates retrieval mechanisms with generative language models, improving the factual accuracy
[H+24b]. In the pharmaceutical domain, a chatbot that uses RAG successfully navigated complex guidelines
by retrieving and storing responses in relevant documents [KM24].
2.3 Multi-Agent Systems and Their Application
Multi-agent systems enable specialized agents to coordinate complex tasks [SLB08, Woo09], facilitating
robust data integration and knowledge engineering [Z+13]—a key advantage in rapidly evolving regulatory
contexts.
3 Ontology Free Knowledge Graph
Knowledge graphs often rely on predefined ontologies (e.g. DBpedia [L+15], YAGO [SKW07]), yet an alterna-
tive ’schema-light’ approach defers rigid schemas in favor of flexible bottom-up extraction [EFC+11, FSE14].
This method quickly adapts to new data domains [CBK+10], reduces initial overhead [EFC+11], and allows
partial schemas to emerge naturally [HBC+21], making it especially valuable in regulatory settings where
rules evolve rapidly, data formats vary [PEK06], and open-ended queries can reveal hidden legal connec-
tions [FSE14]. In order to demonstrate how the schema-light strategy operates in practice, we extracted
triplets from the data from the Electronic Code of Federal Regulations, focusing on specific sections that
share references and time constraints. The resulting relationships form a small subgraph that illustrates
both shallow hierarchical structures ( e.g., parts, and subparts) and interlinked regulatory requirements. As
seen in Figure 1, these extracted triplets reveal how different sections converge on the same 15-day appeal
timeframe, underscoring the flexibility of an ontology-free approach in capturing cross-references and shared
procedural deadlines.
4 Triplet-Based Embeddings for Regulatory QA with Textual Ev-
idence
In this section, we introduce a formulation for leveraging embedded triplets to enable precise, fact-centric
retrieval in a regulatory question-answering system. Unlike purely text-based approaches, our method not
only encodes concise Subject-Predicate-Object relationships, but also links each triplet to the original text
sections from which it was extracted. At query time, the system retrieves both the relevant triplets and
corresponding text evidence, feeding them into an LLM for the final generation of the answer.
4.1 Corpus, Sections, and Triplet Extraction
LetCbe a corpus of regulatory documents. We partition Cinto atomic text sections —- for instance,
paragraphs, clauses, or semantically coherent fragments —-using a function
Ω :C → X ,
2

( partOf )
CHAPTER I <------------- SUBCHAPTER B
^ ^
| ( partOf ) |
PART 117 -------------- SUBPART E ( Withdrawal of QF Exemption )
^ ^
| ( inSubpart ) | ( inSubpart )
§§117.257 , 117.260 , 117.264 , 117.267
| ^ ^ ^
| ( references ) ( references ) ( references )
| \\ $117 .264 \\ $117 .267 \\ $117 .264
|
[ timeframe : 15 days to appeal the order ]
Figure 1: Independent sections converge on a single requirement, only discernible through triplet-driven
interconnections.
where Xis the set of all text fragments {x1, x2, . . . , x m}.
We then apply an information extraction pipeline Φ to each section xj. The pipeline identifies the
subject-predicate-object relationships of that section, thus producing triplets:
Φ 
Ω(C)
=
ti|ti= (si, pi, oi)	
.
We define a linking function Λ such that each triplet tiis associated with one or more text sections xj.
Formally,
Λ :T → 2X,
where Tis the set of all extracted triplets, and Λ 
ti
yields the subset of sections from which tiwas extracted.
Hence, each triplet tihasprovenance —a reference to its original textual source(s).
4.2 Embedding Triplets
For each triplet ti= (si, pi, oi), we create a short textual representation f(ti). A typical choice is a
concatenation of S-P-O, for example:
f(ti) = concat( si, pi, oi).
We then define an embedding function
E:X ∪ T → Rd,
where dis the dimensionality of the embedding space. Specifically, for any triplet ti,
eti=E 
f(ti)
∈Rd
We also embed queries and (optionally) text sections themselves via the same or a compatible model.
The resulting vectors are stored in a vector index Vsuch that:
V=
(eti, ti,Λ(ti))1≤i≤N	
,
where etiis the triplet embedding and Λ( ti) is the set of associated text sections.
4.3 Embedding Function
To enhance query processing and retrieval, we developed an embedding model based on Transformer’s
methodology, specifically leveraging transformer-based architectures such as BERT. This embedding model
was trained on textual data extracted from the eCFR, capturing semantic nuances specific to the regulatory
language. The embedding process involves encoding cleaned textual chunks into high-dimensional vector
representations, which enable efficient semantic search and retrieval in downstream tasks, significantly im-
proving the precision and relevance of responses to regulatory queries.
3

4.4 Query Embedding and Retrieval
Given a user query Q∈ Q —- for example, “Which agency is responsible for Regulation 2025-X?” —- we
embed Qas
eQ=E(Q).
We perform a k-nearest neighbor search in Vusing a similarity measure sim( ·,·), typically cosine similarity.
We obtain:
TQ= TopK
sim 
eQ,eti
which yields the top- ktriplets most relevant to the query. For each retrieved triplet ti∈ TQ, we can also
retrieve its associated text sections through Λ( ti). Formally:
XQ=[
ti∈TQΛ 
ti
soXQis the set of sources’ text sections that support the discovered triplets.
4.5 LLM-Based QA with Triplets and Text
To finalize the answer, we define a function
Γ :Q × 2T×2X→ A ,
where Ais the set of possible answers. Essentially, Γ is an LLM that accepts: User query Q, Retrieved
triplets TQ, Relevant Text Sections XQ. The LLM then produces an answer A∈ A. In symbolic form:
A= Γ 
Q,TQ,XQ
.
In practice, the LLM input might be a prompt that includes the user question plus concatenated or selectively
summarized triplets and text sections. By examining both structured (triplet) facts and verbatim textual
evidence, the LLM generates a more accurate and explainable response.
4.6 Theoretical Considerations
Completeness and Consistency: Tis complete if every relevant statement in Cis represented by at
least one SPO triplet and consistent if Φ does not introduce contradictory or spurious triplets.
Retrieval Sufficiency: With sim 
eQ,eti
as a semantic relatedness measure and an embedding function
Ethat preserves factual relationships, the top- ktriplets in TQshould suffice to answer Q.
Text Sections as Evidence: Because each tilinks back to its source text, users or downstream models
can verify and clarify relationships by referring to the original regulatory language, thus mitigating ambigu-
ities not fully captured by the triplet alone.
5 Multi Agents System
We use a multiagent system to orchestrate ingestion, extraction, cleaning, and query-answering in a modular,
scalable manner. Each agent specializes in a core function, such as document ingestion, triplet extraction,
or final answer generation, so they can run independently and be refined without disrupting the rest.
Figure 2: Multi Agents High Level Architecture
4

5.1 Agents for ontology free knowledge graph constructions
The document ingestion agent segments raw regulatory text, captures metadata, and outputs structured
fragments. The extraction agent uses an LLM to detect subject-predicate-object triplets (e.g., ’FDA requires
submission within 15 days’). Normalization and Cleaning Agent merges duplicates, standardizes entities,
and resolves synonyms to produce clean triplets. Triplet Store and Indexing Agent embeds and stores triplets
in a vector database for easy retrieval.
5.2 Agentic Retrieval-Augmented Generation System
Our second agentic system utilizes the custom embedding model to retrieve semantically similar triplets from
the knowledge graph. Initially, the retrieval agent identifies relevant triplets based on semantic proximity to
user queries. Subsequently, the story-building agent compiles and synthesizes the textual chunks associated
with these triplets into a coherent narrative. Finally, the generation agent processes this cohesive story to
formulate precise and contextually relevant responses. This approach ensures that responses to regulatory
inquiries are accurate, traceable, and grounded in verified regulatory content.
6 Retrieved Subgraph Visualization
Additionally, we supplement the responses with an interactive visual of the relevant subgraphs of the retrieved
triplets. This visual aid significantly improves user comprehension and provides greater contextual clarity,
facilitating informed decision making in regulatory compliance tasks.
Figure 3: Navigational Facility of triplets
5

7 Evaluation
In this section, we outline our methodology for evaluating the system’s ability to (1) retrieve the correct
sections of a regulatory corpus, (2) generate factually accurate answers and (3) demonstrate flexibility of
navigation through the interconnection of triplets in related sections. We detail our sampling procedure, the
construction of queries, the measurement of section-level overlap, the assessment of factual correctness, and
the analysis of triplet-based navigation.
Figure 4: Evaluation Methodology
7.1 Sampling and Ground Truth Construction
Random Sampling of Sections. LetS={s1, s2, . . . , s N}be the full set of sections of the regulatory
corpus. We draw a random subset
S′=
si1, si2, . . . , s ik	
⊂ S,
where each sijis considered a target section for evaluation, and k≪N.
Identifying All Ground Truth Mentions. For each sampled section sij, we locate all other sections in
the corpus that reference or expand upon the same regulatory ideas or entities. Formally, let
M(sij) =
sm1, sm2, . . .	
denote the set of sections that contain overlaps or references relevant to sij. We then create a re-told story
by concatenating sijwith all sections in M(sij):
esij= 
sij∥sm1∥sm2∥. . .
.
This concatenated text esijis treated as the ground truth context for the focal section sij.
7.2 LLM-Generated Questions and Answers
We employ a Large Language Model, denoted LLM gen, to produce a set of questions and corresponding
reference answers based on each concatenated text esij. Formally,
(Qij,Aij) = LLM gen 
esij
,
where Qij={q1, q2, . . . , q m}andAij={a1, a2, . . . , a m}. Each pair ( qr, ar) is presumed to be responsible
via the original information in esij.
7.3 System Inference and Evaluations
7.3.1 Section-Level Overlap
To answer each question qr∈ Q ij, our system retrieves a set of sections Rij,rdeemed relevant (based on
embedding retrieval, triplet matching, or both). We measure the level of overlap between the recovered
sections Rij,rand the ground truth target section sij(along with its reference set M(sij)).
6

Definition: Overlap score. LetGij={sij} ∪M(sij) be the set of ground truth sections. Suppose that
the system returns Rij,r={r1, r2, . . . , r ℓ}. We define the overlap score Ofor question qras
O 
Rij,r,Gij
=Rij,r∩ Gij
Rij,r.
Thus,
•ifRij,r∩ Gij=∅, then O= 0;
•ifRij,rreturns exactly one section, r1, and r1=sij, then O= 1;
•if, for instance, the system returns three sections, only one of which matches any in Gij, then O= 1/3.
We can further refine this measure by applying a similarity threshold θfor the equivalence between the
retrieved sections and the ground truth sections (e.g., if the sections partially overlap or are highly similar).
In that case,Rij,r∩ Gij=X
r∈Rij,r1h
sim 
r, sg
≥θfor some sg∈ Giji
.
7.3.2 Factual Correctness of Answers
Once the system retrieves relevant sections and processes them through the QA pipeline (with or without
associated triplets), it produces an answer a⋆
r. We compare a⋆
rwith the reference answer arfrom LLM gen.
LLM-Based Fact Checking. We use a secondary evaluation model LLM evalor a domain expert to assess
whether a⋆
risfactually correct with respect to the original text esij. We denote:
F(a⋆
r, ar) =(
1,ifa⋆
ris factually correct and consistent with ar,
0,otherwise .
We measure correctness with two conditions:
1.With Triplets : The system’s answer is grounded in the set of triplets that directly link to the retrieved
sections.
2.Without Triplets : The system response is derived purely from the retrieval of raw text, without
referencing the triplet data structure.
By comparing the correctness scores for these two conditions, we quantify the impact of structured triplets
in factual precision.
7.3.3 Navigational Facility of Triplets
We also investigate how triplet interconnections facilitates follow-up questions. In many regulatory contexts,
a concept from one section leads to further questions about a related section. To do this, we define the
following.
Triplet Overlap Across Sections. LetTbe the global set of extracted triplets. For sections sijand
smℓ∈M(sij), we look at triplets that are shared or linked between these sections:
T(sij) ={t∈ T | tis extracted from section sij},
T(smℓ) ={t∈ T | tis extracted from section smℓ}.
We then analyze:
T(sij)∩ T(smℓ),
which denotes shared triplets that link the heads / tail entities in sections. A single triplet may appear in
multiple sections if those sections refer to the same entity relationships; or it may connect a head entity in
sijto a tail entity in smℓ.
7

Navigational Metric. We define a metric Nav( S′) to capture average fraction of shared or sequentially
linked triplets among sections that mention the same ground-truth concepts. Let
Nav(S′) =1
kkX
j=1P
smℓ∈M(sij)T(sij)∩ T(smℓ)
P
smℓ∈M(sij)T(sij)∪ T(smℓ).
A higher value indicates stronger overlap (and thus navigational facility ), suggesting that triplets help the
system move seamlessly between related sections.
By integrating section-level overlap analysis, factual correctness checks, and a triplet interconnection
navigation metric, this evaluation framework measures retrieval accuracy, answer precision, and knowledge
connectivity - ensuring robust compliance support, domain-specific Q&A, and effective scalability in real-
world regulatory settings.
Table 1: Evaluation Results for Section Overlap, Answer Accuracy, and Navigation Metrics
Metric Without Triplets With Triplets
1. Section Overlap (Similarity Threshold)
0.50 0.0812 0.0745
0.60 0.2700 0.2143
0.75 (stricter) 0.1684 0.2888 (highest accuracy)
2. Answer Accuracy (Scale: 1-5)
Average Accuracy 4.71 4.73
3. Navigation Metrics
Average Degree 1.2939 (less interconnected) 1.6080 (more interconnected)
Unconnected Sections Linked 5014 unconnected section 5011 connected sections
Avg. Shortest Path 2.0167 (slower information flow) 1.3300 (faster information flow)
The table compares system performance with and without triplets across three evaluation criteria: retrieval
accuracy (section overlap at varying similarity thresholds), factual correctness of generated answers, and effi-
ciency of navigation through related regulatory sections. Triplets yield highest accuracy at higher threshold.
Triplets network significantly enhances connectivity and navigation.
8 Discussion
Throughout this work, we presented a multi-agent system that uses triplet-based knowledge graph construc-
tion and retrieval-augmented generation (RAG) to enable transparent, verifiable question-answering on a
regulatory corpus. By delegating ingestion, triplet extraction, KG maintenance, and query orchestration to
specialized agents, unstructured text becomes a structured data layer for precise retrieval. The synergy of
KG and RAG provides high-confidence, explainable facts alongside fluent responses to the large language
model, as Section 7 demonstrates through accurate section retrieval, factual correctness and navigational
queries (Figure 3). Grounding answers with triplets reduces LLM hallucinations, and provenance links enable
robust auditing.
8.1 Challenges
An ontology-free approach facilitates rapid ingestion and incremental refinement but can lead to vocabulary
fragmentation; canonicalization and entity resolution [GTHS14, SWLW14] help unify concepts, and advanced
reasoning tasks may still benefit from partial or emergent schemas [RYM13]. Extraction quality directly
affects the integrity of the KG, as domain-specific jargon or ambiguous references can produce missing or
erroneous triples, and deeper inferences or temporal constraints may require additional rule-based or symbolic
logic. Large-scale RAG pipelines also require careful optimization for embedding, indexing, and retrieval.
8

8.2 Future Directions
Looking ahead, we see multiple avenues for enhancing and extending the system: although our current
pipeline supports factual lookups, more complex regulatory questions demand deeper logical reasoning or
chaining of evidence, and integration with advanced reasoning LLMs can address multistep analysis and
domain-specific inference needs. By including user feedback or expert annotations, we could iteratively re-
fine triplet quality and reduce extraction errors. Active learning or weakly supervised methods may help
identify ambiguous relationships, prompting relabeling or model retraining. Over time, such feedback loops
would yield higher-precision knowledge graphs. Regulatory corpora often change rapidly (e.g., new guide-
lines, amendments). We aim to develop incremental update mechanisms that re-ingest altered documents
and regenerate only those triples affected by the changes, minimizing downtime and ensuring continuous
compliance coverage. Although we focus on health life science regulatory compliance , the underlying archi-
tecture of multi–agent ingestion, knowledge graph construction, and RAG QA—can be generalized to other
domains with high stakes factual queries (e.g., clinical trials, financial regulations, or patent law). Tailoring
the extraction logic and graph schema of each agent to domain-specific requirements would enable a larger
impact.
References
[A+24] Marah Abdin et al. Phi-4 technical report. arXiv preprint , 2024.
[C+20] Xiaojun Chen et al. A review: Knowledge reasoning over knowledge graph. Expert Systems with
Applications , 2020.
[C+24] Subhankar Chattoraj et al. Semantically rich approach to automating regulations of medical
devices. Technical report, UMBC, 2024.
[CBK+10] Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam Hruschka, and Tom
Mitchell. Toward an architecture for never-ending language learning. In Proceedings of the 24th
AAAI Conference on Artificial Intelligence (AAAI) , pages 1306–1313, 2010.
[CDW22] Joseph J. Cordes, Susan E. Dudley, and Layvon Washington. Regulatory compliance burden.
GW Regulatory Studies Center , 2022.
[EFC+11] Oren Etzioni, Anthony Fader, Janara Christensen, Stephen Soderland, and Mausam. Open
information extraction: The second generation. In Proceedings of the 22nd International Joint
Conference on Artificial Intelligence (IJCAI) , pages 3–10, 2011.
[Ers23] Vladimir Ershov. A case study for compliance as code with graphs and language models. arXiv
preprint , 2023.
[FDA25] FDA. Fda guidance documents. FDA Regulatory Information , 2025.
[FSE14] Anthony Fader, Stephen Soderland, and Oren Etzioni. Open information extraction for the web.
Communications of the ACM , 57(9):80–86, 2014.
[GTHS14] Luis Gal´ arraga, Christina Teflioudi, Klaus Hose, and Fabio Suchanek. Canonicalizing open
knowledge bases. In Proceedings of the 23rd ACM International Conference on Information and
Knowledge Management (CIKM) , pages 1679–1688, 2014.
[H+21] Aidan Hogan et al. Knowledge graphs. arXiv preprint , 2021.
[H+24a] Joe B. Hakim et al. The need for guardrails with large language models in medical safety-
critical settings: An artificial intelligence application in the pharmacovigilance ecosystem. arXiv
preprint , 2024.
[H+24b] Lars Hillebrand et al. Advancing risk and quality assurance: A rag chatbot for improved regu-
latory compliance. https://ieeexplore.ieee.org/document/10825431 , 2024.
9

[HBC+21] Aidan Hogan, Eva Blomqvist, Michael Cochez, et al. Knowledge graphs. Synthesis Lectures on
Data, Semantics, and Knowledge , 12(2):1–257, 2021.
[HCB24] Yu Han, Aaron Ceross, and Jeroen Bergmann. More than red tape: exploring complexity in
medical device regulatory affairs. BMJ Innovations , 2024.
[J+24] Ziwei Ji et al. Survey of hallucination in natural language generation. arXiv preprint , 2024.
[K+20] Vladimir Karpukhin et al. Dense passage retrieval for open-domain question answering. arXiv
preprint , 2020.
[KM24] Jaewoong Kim and Moohong Min. From rag to qa-rag: Integrating generative ai for pharma-
ceutical regulatory compliance process. arXiv preprint , 2024.
[L+15] Jens Lehmann et al. Dbpedia – a large-scale, multilingual knowledge base extracted from
wikipedia. Semantic Web , 2015.
[L+21] Patrick Lewis et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. arXiv
preprint , 2021.
[L+24a] Jiarui Li et al. Enhancing llm factual accuracy with rag to counter hallucinations: A case study
on domain-specific queries in private knowledge-bases. arXiv preprint , 2024.
[L+24b] Chen Ling et al. Domain specialization as the key to make large language models disruptive: A
comprehensive survey. arXiv preprint , 2024.
[N+15] Maximilian Nickel et al. A review of relational machine learning for knowledge graphs. arXiv
preprint , 2015.
[PEK06] Florian Probst, Sascha Eck, and Werner Kuhn. Scalable semantics: A case study on ontology-
driven geographic information integration. International Journal of Geographical Information
Science , 20(5):563–583, 2006.
[RYM13] Sebastian Riedel, Limin Yao, and Andrew McCallum. Relation extraction with matrix factor-
ization and universal schemas. In Proceedings of NAACL-HLT 2013 , pages 74–84, 2013.
[SKW07] Fabian M. Suchanek, Gjergji Kasneci, and Gerhard Weikum. Yago: A core of semantic knowledge
unifying Wikipedia and WordNet. Proceedings of the 16th International Conference on World
Wide Web , 2007.
[SLB08] Yoav Shoham and Kevin Leyton-Brown. Multiagent systems: Algorithmic, game-theoretic,
and logical foundations. https://www.eecs.harvard.edu/cs286r/courses/fall08/files/
SLB.pdf , 2008.
[SWLW14] Wei Shen, Jianyong Wang, Ping Luo, and Min Wang. A survey on entity linking: Methods, tech-
niques, and applications. IEEE Transactions on Knowledge and Data Engineering , 27(2):443–
460, 2014.
[Wei00] Gerhard Weiss. Multiagent systems: A modern approach to distributed artificial intelligence.
https://ieeexplore.ieee.org/book/6267355 , 2000.
[Woo09] Michael Wooldridge. An Introduction to MultiAgent Systems . Wiley, 2009. https://www.wiley.
com/en-be/An+Introduction+to+MultiAgent+Systems%2C+2nd+Edition-p-9780470519462 .
[WZ24] Dandan Wang and Shiqing Zhang. Large language models in medical and healthcare fields:
applications, advances, and challenges. Artificial Intelligence Review , 2024.
[X+25] Yunfei Xiang et al. Integrating knowledge graph and large language model for safety management
regulatory texts. In Lecture Notes in Computer Science , volume 14250, pages 976–988. 2025.
[Y+24] An Yang et al. Qwen2.5 technical report. arXiv preprint , 2024.
10

[Y+25] Rui Yang et al. Retrieval-augmented generation for generative artificial intelligence in health
care. npj Digital Medicine , 2025.
[Z+13] Anna Zygmunt et al. Agent-based environment for knowledge integration. arXiv preprint , 2013.
[Z+24] Tianyang Zhong et al. Evaluation of openai o1: Opportunities and challenges of agi. arXiv
preprint , 2024.
11