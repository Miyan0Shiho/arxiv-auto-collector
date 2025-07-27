# Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning in Real-World Applications

**Authors**: Jean Lelong, Adnane Errazine, Annabelle Blangero

**Published**: 2025-07-22 12:03:10

**PDF URL**: [http://arxiv.org/pdf/2507.16507v1](http://arxiv.org/pdf/2507.16507v1)

## Abstract
Conventional Retrieval-Augmented Generation (RAG) systems enhance Large
Language Models (LLMs) but often fall short on complex queries, delivering
limited, extractive answers and struggling with multiple targeted retrievals or
navigating intricate entity relationships. This is a critical gap in
knowledge-intensive domains. We introduce INRAExplorer, an agentic RAG system
for exploring the scientific data of INRAE (France's National Research
Institute for Agriculture, Food and Environment). INRAExplorer employs an
LLM-based agent with a multi-tool architecture to dynamically engage a rich
knowledge base, through a comprehensive knowledge graph derived from open
access INRAE publications. This design empowers INRAExplorer to conduct
iterative, targeted queries, retrieve exhaustive datasets (e.g., all
publications by an author), perform multi-hop reasoning, and deliver
structured, comprehensive answers. INRAExplorer serves as a concrete
illustration of enhancing knowledge interaction in specialized fields.

## Full Text


<!-- PDF content starts -->

Agentic RAG with Knowledge Graphs for Complex
Multi-Hop Reasoning in Real-World Applications
Jean Lelonga, Adnane Errazineaand Annabelle Blangeroa
aEkimetrics, France
Abstract. Conventional Retrieval-Augmented Generation (RAG)
systems enhance Large Language Models (LLMs) but often fall short
on complex queries, delivering limited, extractive answers and strug-
gling with multiple targeted retrievals or navigating intricate entity
relationships. This is a critical gap in knowledge-intensive domains.
We introduce INRAExplorer, an agentic RAG system for exploring
the scientific data of INRAE (France’s National Research Institute
for Agriculture, Food and Environment). INRAExplorer employs an
LLM-based agent with a multi-tool architecture to dynamically en-
gage a rich knowledge base, through a comprehensive knowledge
graph derived from open access INRAE publications. This design
empowers INRAExplorer to conduct iterative, targeted queries, re-
trieve exhaustive datasets (e.g., all publications by an author), per-
form multi-hop reasoning, and deliver structured, comprehensive an-
swers. INRAExplorer serves as a concrete illustration of enhancing
knowledge interaction in specialized fields.
1 Introduction
Effective digital knowledge utilization demands relevant, exhaustive
and structured information retrieval. While Retrieval-Augmented
Generation (RAG) grounds Large Language Models (LLMs) in cu-
rated, trustworthy information [5], prevalent architectures —termed
classical RAG— exhibit key limitations. Classical RAG typically re-
trieves a limited set (top-k) of semantically similar text chunks via
vector search [4, 2] to contextualize an LLM. While valuable for
anchoring LLM responses and extractive question answering, this
’top-k snippet’ approach is often insufficient for queries requiring
exhaustive lists, synthesis from multiple distinct data points, or navi-
gating complex relational paths (e.g., author to publications to fund-
ing projects).
To address these shortcomings, we introduce INRAExplorer, a
method that synergizes agentic RAG [10] for dynamic reasoning and
Knowledge Graph (KG)-enhanced RAG [12, 13] for structured, ex-
haustive retrieval. INRAExplorer deeply incorporates KG querying
as a core agentic capability, enabling it to overcome the single-pass,
limited-context nature of classical RAG. This fusion delivers precise,
relationally-aware retrieval from KGs, combined with the adaptive,
multi-hop reasoning of an LLM-driven agent.
While the integration of Knowledge Graphs with LLMs is gaining
traction, many current approaches primarily use KGs by perform-
ing a sophisticated form of map-reduce summarization over graph-
retrieved data [1]. In contrast, INRAExplorer, drawing inspiration
from the need for deeper reasoning similar to human investigative
processes, focuses on enabling the LLM agent to construct chains
of thought leveraging several ways to retrieve information. Our sys-tem empowers the agent to dynamically navigate between different
tools, gathering evidence, evaluating intermediate findings, and plan-
ning subsequent steps. This allows INRAExplorer to act more like a
human researcher, meticulously assembling pieces of information to
construct a comprehensive and nuanced answer, rather than simply
summarizing pre-existing snippets of information.
2 INRAExplorer: Agentic RAG with Knowledge
Graphs
INRAExplorer is a system designed to implement and showcase the
capabilities of this agentic, KG-enhanced RAG methodology. It em-
ploys an LLM-based agent that orchestrates a suite of specialized
tools to interact with a hybrid knowledge base, specifically tailored
to the complexities of real-world scientific data exploration.
2.1 Knowledge Base Construction
The foundation of INRAExplorer is a comprehensive knowledge
base constructed from INRAE’s scientific output, focusing on publi-
cations from January 2019 to August 2024 and restricted to Open Ac-
cess documents. Core data originates from an internal join of pub-
lication metadata from HAL (Hyper Articles en Ligne), valued for
its qualitative and exhaustive INRAE scope, and OpenAire, used for
deduplication and enriched author/project information. This merged
dataset was further enriched with other public sources using DOIs,
including BBI (Base Bibliographique INRAE) for validation, ScanR
for additional publication and project details, and dataset repositories
for links to underlying research data.
Full-text content of Open Access PDF publications was processed
using GROBID for structured text extraction [6], isolating sections
such as title, abstract, keywords, introduction, and conclusion to form
meaningful chunks.
A significant component of the structured knowledge comes from
theINRAE Thesaurus . This was integrated into the KG as a dedi-
cated subgraph, where hierarchical terms form ‘Domain‘ nodes and
more specific, leaf-level terms are represented as ‘Concept‘ nodes
(see Table 1). Publications are then linked to these ‘Concept‘ nodes
using exact matches in selected sections of publication texts, provid-
ing a controlled vocabulary for semantic exploration and enabling the
model to reference the thesaurus for understanding domain-specific
query vocabulary.
The processed information is stored in a hybrid knowledge base :
•AVector Database stores representations of the textual chunks.
For each publication, key sections (title, abstract, introduction,arXiv:2507.16507v1  [cs.AI]  22 Jul 2025

Table 1. Distribution of Node Types in the INRAExplorer Knowledge Graph (Total Nodes: 417,030)
Node Type Count Percentage Description
Author 233,728 56.0% Researchers and authors of scientific publications
Keyword 96,588 23.2% Keywords associated with publications (declared by authors)
Publication 38,791 9.3% Scientific articles and other academic publications
Software 21,617 5.2% Software developed or used in research
Concept 13,591 3.3% Concepts from the INRAE thesaurus identified in publications
Journal 5,563 1.3% Scientific journals where works are published
Project 3,999 1.0% Funded research projects
Domain 2,595 0.6% Thematic domains of the INRAE thesaurus
ResearchUnit 299 0.1% INRAE research units and laboratories
Dataset 240 0.1% Datasets used or produced in research
Region 19 0.0% Geographic regions where research units are located
conclusion) were concatenated to form these chunks. Two types of
vectors were computed for each chunk to support hybrid search: a
dense vector using a Jina v3 embedding model for semantic sim-
ilarity [2], and a sparse vector using BM25 for keyword-based
matching [7].
•AKnowledge Graph models the core entities and their mul-
tifaceted relationships [8]. The graph comprises various node
types—including ‘Publication‘, ‘Author‘, ‘Keyword‘, but also
‘Concept‘ and ‘Domain‘ from INRAE’s thesaurus, and several
other complementary meta-information. Table 1 details these node
types and their distributions. The graph contains more than 1
million relationships that link these entities. This KG serves as
the backbone for precise, structured queries, multi-hop reasoning
[1, 3], and the retrieval of exhaustive, interconnected information
sets.
2.2 Agent-Driven Multi-Tool Orchestration
At the heart of INRAExplorer is an LLM-based agent utilizing the
open-weight model deepseek-r1-0528. Its primary responsibilities
include: understanding the user’s query, decomposing it if necessary,
formulating a plan of action, dynamically selecting and invoking the
appropriate tools, and synthesizing the information gathered from
multiple tool calls into a coherent and comprehensive response.
The agent has access to a set of specialized tools, each designed
for a specific type of interaction with the knowledge base:
•SearchGraph (Knowledge Graph Querying): This is the
main tool, pivotal for deep interaction with the Neo4j KG. It al-
lows the agent to send Cypher queries to retrieve specific entities,
traverse complex relationship paths, and gather exhaustive lists
(e.g., all publications by a specific author, all projects associated
with a particular research unit). This tool is critical for obtaining
structured and complete answers that go beyond simple text snip-
pet retrieval.
•SearchPublications (Hybrid Publication Search): This
helper tool interfaces with the vector database to perform hybrid
searches (semantic [4, 2] and keyword-based [7]) over the cor-
pus of publication texts. It allows the agent to find relevant entry
points into the graph to start its reasoning and chain of retrieval,
for instance by identifying initial sets of documents for further ex-
ploration via the KG. A reranker (e.g., Cohere) is used to refine
the top results and merge the results of the two retrieval methods.
•SearchConceptsKeywords (Concept/Keyword Search):
This other helper tool allows the agent to find relevant entry points
into the graph. It helps bridge the gap between user queries and
the structured vocabulary of the knowledge base by allowing the
agent to search for relevant concepts from the integrated thesaurior the author keywords indexed in the KG. This is useful for query
disambiguation, suggesting related terms, or finding precise entry
points for subsequent graph traversal.
•IdentifyExperts (Expert Identification): This refined tool
encapsulates complex domain-specific knowledge for identify-
ing experts on a given topic. It allows for more reproducibil-
ity through different answers by the model. The tool uses
SearchPublications to find highly relevant papers, then use
SearchGraph to analyze the authors of these papers, their num-
ber of citations on this given topic, their collaboration networks,
their involvement in related projects, and other structural indica-
tors of expertise, before synthesizing a ranked list based on a com-
posite score.
3 Illustrative Scenarios and Capabilities
The INRAExplorer method, through its agentic and Knowledge
Graph (KG)-centric design, offers significant advantages over classi-
cal RAG systems, particularly for complex information needs requir-
ing exhaustive and structured answers. This section illustrates these
capabilities through two representative scenarios, highlighting multi-
hop sequential reasoning and the use of specialized, modular tools.
3.1 Multi-Hop Sequential Reasoning for Complex
Queries
Many real-world queries necessitate navigating multiple types of re-
lationships across different entities, a task where classical RAG often
falters. For example, consider the query: "Find INRAE authors who
have published on ’climate change adaptation strategies’, identify the
projects that funded these publications, and list other key topics these
funding projects are related to."
INRAExplorer’s agent tackles this through a sequence of reasoned
steps, demonstrating multi-hop reasoning:
1.Initial Information Gathering : The agent first uses
SearchPublications orSearchConceptsKeywords
to identify an initial set of relevant publications and their authors
related to ’climate change adaptation strategies’. This step
grounds the query in the available literature.
2.First Hop - Identifying Funding : Using the SearchGraph
tool, the agent then queries the KG to find ‘Project‘ nodes linked to
these initial publications via a ‘FUNDED_BY‘ relationship. This
constitutes the first "hop" in the reasoning chain, connecting pub-
lications to their funding sources.
3.Second Hop - Exploring Related Project Topics : Subsequently,
for each identified funding project, the agent again employs

SearchGraph . This time, it seeks other ‘Concept‘ nodes (repre-
senting topics) linked to these projects via relationships like ‘DE-
SCRIBES‘. This is the second "hop," broadening the understand-
ing of the projects’ thematic scope, through common publication
nodes.
4.Synthesis : Finally, the agent synthesizes these interconnected
findings into a structured answer. This response explicitly shows
the chain: authors →publications on ’climate change adaptation
strategies’ →funding projects →other related research topics.
This process illustrates how the agent can meticulously assemble
pieces of information, retrieve complete sets of entities and their spe-
cific relationships from the KG, and construct a comprehensive an-
swer that goes far beyond simple snippet retrieval.
3.2 Modular and Controlled Expertise Identification
INRAExplorer’s architecture supports the integration of specialized,
high-level tools that encapsulate complex, domain-specific logic, of-
fering more controlled and reproducible outputs compared to relying
solely on raw tool access for every task. The IdentifyExperts
tool is a prime example of this modularity.
Consider a query such as: "Identify leading INRAE experts
on ’zoonoses’." Instead of the agent needing to devise a multi-
step plan from scratch using basic tools, it can leverage the
IdentifyExperts tool. This tool is designed to provide a repli-
cable and controlled method for expert identification:
1.Tool Invocation : The agent recognizes the query’s intent and calls
theIdentifyExperts tool with the topic ’zoonoses’.
2.Encapsulated Workflow : The IdentifyExperts tool exe-
cutes a predefined sequence of actions, which itself involves calls
to other foundational tools:
•It first uses SearchPublications to find highly relevant
papers on ’zoonoses’.
•Then, it employs SearchGraph to extract the authors of these
papers.
•For each author, it calculates a composite expertise score based
on multiple weighted metrics: average relevance of their arti-
cles to the topic, number of articles in the top 10% of results,
total number of relevant publications, citation counts for these
publications, period of activity in the domain, and recency of
their latest publication.
3.Structured Output : The tool returns a ranked list of experts on
’zoonoses’, along with their expertise scores and the breakdown
of these scores.
4.Synthesis by Agent : The agent then presents this structured in-
formation to the user.
This approach ensures consistent, domain-aware execution for
tasks like expert identification and simplifies agent decision-making
with powerful abstract tools. Such modularity is key for future ex-
pansion, enabling the agent to flexibly choose between direct graph
access (e.g., via SearchGraph ) for novel queries and controlled
tools for established needs, enhancing adaptability and reliability.
3.3 System Design for Real-World Application and AI
Advancement
INRAExplorer demonstrates the effective application of multiple AI
techniques to a significant real-world challenge: navigating and rea-
soning over complex scientific knowledge. Its distinctiveness arisesfrom the synergistic integration of an agentic framework [10] with
deep Knowledge Graph querying capabilities [12]. This combination
enables sophisticated multi-hop reasoning and the retrieval of ex-
haustive, structured answers, addressing needs beyond typical RAG
systems. The system’s architecture leverages key open-source com-
ponents—including Mirascope for agent orchestration, Qdrant for
vector storage, Neo4j for the knowledge graph, GROBID for PDF
processing [6]—and utilizes the open-weight model deepseek-r1-
0528. This open design ensures inherent adaptability, meaning IN-
RAExplorer not only serves as a potent tool for its current domain
but also offers an extensible architecture. Such a foundation can fa-
cilitate the practical integration and evaluation of new AI approaches
for advanced information interaction and complex reasoning tasks.
4 Conclusion and Future Work
The INRAExplorer method advances beyond conventional Retrieval-
Augmented Generation [5] by synergizing an agentic, multi-tool
framework [10] with the deep, structured querying capabilities of
Knowledge Graphs (KGs) [12]. This distinct combination allows for
exhaustive retrieval and sophisticated handling of complex relational
queries, offering greater flexibility in problem decomposition and
dynamic tool use [11, 9] than systems focusing narrowly on either
general agency or KG querying alone [1, 3]. INRAExplorer thus
delivers comprehensive, structured answers crucial for knowledge-
intensive domains and represents an open, deployable, and adaptable
solution for advanced knowledge exploration, fostering accessibility
and community development.
Future work will focus on developing a tailored evaluation frame-
work for INRAExplorer. Standard benchmarks fail to capture the
complexity of scientific, multi-hop queries central to our use case. A
meaningful assessment requires collaboration with domain experts
to define realistic tasks, gold standards, and success criteria. This
co-designed approach will enable rigorous validation of both system
performance and underlying technical choices in real-world condi-
tions.
Furthermore, looking beyond the initial evaluation, another
promising direction for enhancing INRAExplorer involves special-
izing the core agent model. Techniques inspired by reinforcement
learning, such as Reinforcement Learning from Verifiable Feedback
(RLVF), could be explored to further refine smaller, more efficient
language models. This approach could train them to better navigate
the complexities and variable nature of multi-hop reasoning tasks,
potentially leading to more robust and nuanced system performance.
Acknowledgements
The authors wish to express their gratitude to several individuals and
institutions for their contributions to this work. We thank Tristan Sa-
lord, Alban Thomas, Eric Cahuzac, François-Xavier Sennesal, Odile
Hologne and Hadi Quesneville from INRAE for their assistance in
the data gathering process, for providing expert guidance on the rules
for merging diverse data sources, for their insightful expert opinions,
and for INRAE’s role in co-financing this project. We also extend our
sincere thanks to Nicolas Chesneau from Ekimetrics for his support
and insightful contributions.
References
[1] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson. From local to global:

A graph rag approach to query-focused summarization, 2025. URL
https://arxiv.org/abs/2404.16130.
[2] M. Günther, L. Milliken, J. Geuter, G. Mastrapas, B. Wang, and H. Xiao.
Jina embeddings: A novel set of high-performance sentence embedding
models, 2023. URL https://arxiv.org/abs/2307.11224.
[3] X. He, Y . Tian, Y . Sun, N. V . Chawla, T. Laurent, Y . LeCun, X. Bresson,
and B. Hooi. G-retriever: Retrieval-augmented generation for textual
graph understanding and question answering, 2024. URL https://arxiv.
org/abs/2402.07630.
[4] V . Karpukhin, B. O ˘guz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen,
and W. tau Yih. Dense passage retrieval for open-domain question an-
swering, 2020. URL https://arxiv.org/abs/2004.04906.
[5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel,
and D. Kiela. Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. In H. Larochelle, M. Ranzato, R. Had-
sell, M. Balcan, and H. Lin, editors, Advances in Neural Informa-
tion Processing Systems , volume 33, pages 9459–9474. Curran Asso-
ciates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.pdf.
[6] P. Lopez. GROBID: Combining Automatic Bibliographic Data Recog-
nition and Term Extraction for Scholarship Publications. In Research
and Advanced Technology for Digital Libraries, 13th European Confer-
ence, ECDL 2009, Corfu, Greece, September 27-October 2, 2009. Pro-
ceedings , volume 5714 of Lecture Notes in Computer Science , pages
473–474. Springer, 2009. doi: 10.1007/978-3-642-04346-8_62. URL
https://doi.org/10.1007/978-3-642-04346-8_62.
[7] S. E. Robertson and H. Zaragoza. The Probabilistic Relevance Frame-
work: BM25 and Beyond. Foundations and Trends in Information Re-
trieval , 3(4):333–389, 2009. doi: 10.1561/1500000019. URL https:
//doi.org/10.1561/1500000019.
[8] D. S. Roll, Z. Kurt, Y . Li, and W. L. Woo. Augmenting Orbital Debris
Identification with Neo4j-Enabled Graph-Based Retrieval-Augmented
Generation for Multimodal Large Language Models. Sensors , 25(11):
3352, 2025. doi: 10.3390/s25113352. URL https://www.mdpi.com/
1424-8220/25/11/3352.
[9] T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettle-
moyer, N. Cancedda, and T. Scialom. Toolformer: Language Models
Can Teach Themselves to Use Tools. arXiv preprint arXiv:2302.04761 ,
2023. URL https://arxiv.org/abs/2302.04761.
[10] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei. Agentic Retrieval-
Augmented Generation: A Survey on Agentic RAG. arXiv preprint
arXiv:2501.09136 , jan 2025. URL https://arxiv.org/abs/2501.09136.
[11] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y . Cao.
ReAct: Synergizing Reasoning and Acting in Language Models. In
International Conference on Learning Representations (ICLR) , 2023.
URL https://arxiv.org/abs/2210.03629.
[12] Q. Zhang, S. Chen, Y . Bei, Z. Yuan, H. Zhou, Z. Hong, J. Dong,
H. Chen, Y . Chang, and X. Huang. A Survey of Graph Retrieval-
Augmented Generation for Customized Large Language Models. arXiv
preprint arXiv:2501.13958 , jan 2025. URL https://arxiv.org/abs/2501.
13958.
[13] X. Zhu, Y . Xie, Y . Liu, Y . Li, and W. Hu. Knowledge Graph-Guided
Retrieval Augmented Generation. In Proceedings of the 2025 Annual
Conference of the North American Chapter of the Association for Com-
putational Linguistics (NAACL) , 2025. URL https://aclanthology.org/
2025.naacl-long.449/. arXiv:2502.06864.