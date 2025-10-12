# Deterministic Legal Retrieval: An Action API for Querying the SAT-Graph RAG

**Authors**: Hudson de Martim

**Published**: 2025-10-07 15:04:23

**PDF URL**: [http://arxiv.org/pdf/2510.06002v1](http://arxiv.org/pdf/2510.06002v1)

## Abstract
The Structure-Aware Temporal Graph RAG (SAT-Graph RAG) addresses core
limitations of standard Retrieval-Augmented Generation in the legal domain by
providing a verifiable knowledge graph that models hierarchical structure,
temporal evolution, and causal events of legal norms. However, a critical gap
remains: how to reliably query this structured knowledge without sacrificing
its deterministic properties. This paper introduces the SAT-Graph API, a formal
query execution layer centered on canonical actions-atomic, composable, and
auditable primitives that isolate probabilistic discovery from deterministic
retrieval. These actions enable: (i) high-precision hybrid search; (ii) robust
reference resolution; (iii) point-in-time version retrieval; and (iv) auditable
causal tracing. We demonstrate how planner-guided agents can decompose complex
queries into Directed Acyclic Graphs (DAGs) of these actions. This two-layer
architecture transforms retrieval from an opaque black box to a transparent,
auditable process, directly addressing Explainable AI (XAI) requirements for
high-stakes domains.

## Full Text


<!-- PDF content starts -->

Deterministic Legal Retrieval:
An Action API for Querying the SAT-Graph RAG
Hudson de Martim1
1Federal Senate of Brazil ,hudsonm@senado.leg.br
Abstract
The Structure-Aware Temporal Graph RAG (SAT-Graph RAG) addresses core limitations of standard
Retrieval-Augmented Generation in the legal domain by providing a verifiable knowledge graph that
models hierarchical structure, temporal evolution, and causal events of legal norms. However, a critical
gap remains: how to reliably query this structured knowledge without sacrificing its deterministic
properties. This paper introduces the SAT-Graph API, a formal query execution layer centered on
canonical actions—atomic, composable, and auditable primitives that isolate probabilistic discovery
from deterministic retrieval. These actions enable: (i) high-precision hybrid search; (ii) robust reference
resolution; (iii) point-in-time version retrieval; and (iv) auditable causal tracing. We demonstrate how
planner-guided agents can decompose complex queries into Directed Acyclic Graphs (DAGs) of these
actions. This two-layer architecture transforms retrieval from an opaque black box to a transparent,
auditable process, directly addressing Explainable AI (XAI) requirements for high-stakes domains.
Keywords:Legal AI, Knowledge Graphs, Retrieval-Augmented Generation (RAG), Agentic Frameworks,
Explainable AI, Trustworthy AI, Temporal Data Modeling, Legal Ontology.
1 Introduction
The advent of Large Language Models (LLMs) has catalyzed a paradigm shift in information retrieval, with
Retrieval-Augmented Generation (RAG) emerging as a dominant architecture for grounding model outputs
in factual data [ 1]. By retrieving relevant text chunks to supplement the model’s parametric knowledge,
RAG systems have shown remarkable success in reducing hallucinations. However, the direct application
of standard "flat-text" RAG to high-stakes, structured domains such as law reveals fundamental limitations.
Legal corpora are not mere collections of text; they are complex, interlinked systems defined by a rigid
hierarchical structure, a continuous temporal evolution, and a clear causal chain of legislative actions. Recent
studies confirm that traditional RAG frameworks, which rely on surface-level semantic similarity, fail to
capture the nuanced relevance required by legal statutes and are insufficient for domain-specific applications
without significant adaptation [2, 3].
Standard RAG architectures are ill-equipped to handle this complexity. Even when enhanced with sophis-
ticated semantic segmentation that correctly parses a legal text into its constituent articles and paragraphs, the
underlying retrieval mechanism remains fundamentally flawed. It treats these well-defined segments as a
simple "flat-text" collection. More critically, it lacks any inherent understanding of versioning. A query about
a legal provision is meaningless without a temporal anchor, as the correct answer changes over time. This
structural and causal blindness renders such systems fundamentally unreliable for applications demanding
legal precision and auditability.
1arXiv:2510.06002v1  [cs.AI]  7 Oct 2025

To address these deficiencies, our prior work introduced the Structure-Aware Temporal Graph RAG (SAT-
Graph RAG) [ 4]. This framework moves beyond flat text by modeling legal norms as a verifiable knowledge
graph whose ontology explicitly represents the hierarchy, diachronic evolution, and causal events of legislation,
as illustrated in Figure 1. While the SAT-Graph provides the necessary verifiable knowledge substrate, a
critical gap remains: how can external systems, such as agentic reasoning frameworks or query engines,
reliably and flexibly interact with this structured knowledge without sacrificing its deterministic properties?
Unstructured access could easily bypass the graph’s guarantees, undermining the very trustworthiness it was
designed to provide.
Figure 1: From "flat-text" RAG to SAT Graph RAG
This paper bridges that critical gap. We argue that a trustworthy knowledge base requires a formally
defined and auditable interaction protocol. To this end, we introduce a formal query execution layer, which we
specify as the SAT-Graph API. The core of this API is a library ofcanonical actions—atomic, composable,
and auditable primitives designed to serve as a secure low-level interface for querying the SAT-Graph. Instead
of a monolithic query engine, we provide the essential building blocks that higher-level systems can use
to construct complex retrieval plans. These actions provide a comprehensive toolkit for: (i) high-precision
hybrid search combining semantic, lexical, and structured metadata filters; (ii) robust resolution of textual
references to formal identifiers; (iii) deterministic retrieval of point-in-time versions; and (iv) complete,
auditable tracing of causal lineage.
We demonstrate how a planner-guided agent, such as those proposed in advanced Graph RAG frame-
works [ 5], can leverage this API by decomposing a user’s natural language query into an executable Directed
Acyclic Graph (DAG) of these actions. This two-layer architecture—with the SAT-Graph as the verifiable
knowledge base and our Action API as the secure interaction protocol—establishes a new blueprint for
building trustworthy legal AI. It is a direct architectural response to the broader challenge of Explainable
AI (XAI), moving the system’s reasoning from an opaque "black box" to a transparent process that can be
audited and understood, a critical requirement for high-stakes domains [ 6]. This allows advanced reasoning
engines to operate on legal knowledge while ensuring every step of the retrieval process remains explicit,
verifiable, and grounded in the structure of the law.
The main contributions of this paper are as follows:
•We formally specify a Canonical Action API, a library of primitive and composable actions for
interacting with legal knowledge graphs, covering primitives for retrieval, temporal resolution, and
causal analysis.
2

•We demonstrate how this API serves as a foundational layer for agentic architectures, enabling them to
decompose complex questions into verifiable retrieval plans.
•We articulate how this architecture decouples the knowledge representation (the graph) from the query
logic (the agent), enabling a robust and explainable approach to answering legal questions with LLMs.
The remainder of this paper is structured as follows. Section 2 discusses related work in legal AI and
graph-based retrieval. Section 3 details the specification of our Canonical Action API. Section 4 presents use
cases demonstrating its application, and Section 5 concludes with a discussion of the implications and future
directions.
2 Related Work
Our work is situated at the intersection of legal informatics, retrieval-augmented generation, knowledge
graphs, and agentic AI systems. We position our contribution by reviewing the state of the art in these areas.
2.1 AI in Legal Information Retrieval
The application of artificial intelligence to legal text is a long-standing field of research. Early systems relied
on handcrafted rules and expert knowledge to perform legal reasoning [ 7]. While the advent of machine
learning brought significant advances through models like Legal-BERT [ 8], a common thread remains the
treatment of legal documents as unstructured text. This perspective often overlooks the deep, formal semantics
embedded in the law’s structure.
Our approach, in contrast, aligns with a mature research tradition that seeks to formalize legal knowledge
using ontologies. The challenge of managing the "diachronic evolution" of legal norms has long been
recognized, with early work proposing ontologies to identify and track legal resources over time [ 9]. Our use
of a formal, LRMoo-based model [ 10] builds upon this foundation. Furthermore, the explicit modeling of
legislative changes via our ‘Action‘ nodes is a direct response to the recognized need for formal languages to
describe and manage changes in complex knowledge graphs, a problem addressed by frameworks like the
Knowledge Graph Change Language (KGCL) [ 11]. By adopting a formal, top-down ontological model, we
enable more sophisticated, machine-readable applications that respect the inherent structure of the law.
2.2 Retrieval-Augmented Generation and its Limitations
Retrieval-Augmented Generation (RAG) has become the de-facto standard for enabling LLMs to answer
questions over specific corpora [ 1]. In its canonical form, RAG operates by embedding and indexing text
chunks, retrieving the most relevant via vector similarity search.
While effective for unstructured text, this "flat-text" RAG paradigm falters in the legal domain. Recent
benchmarks confirm that the retrieval stage remains a critical bottleneck for legal RAG systems [ 12]. Our
claims of "context blindness" and "temporal naivety" are not hypothetical; they reflect well-documented
failure modes. Studies show that a reliance on surface-level similarity fails to capture the "nuanced relevance"
of legal statutes [ 3] and that standard RAG is insufficient without domain-specific adaptations to handle
the complexity of legal queries [ 2]. The inability of these systems to guarantee the retrieval of the correct
historical version of a law is a fatal flaw for any serious legal application.
2.3 Knowledge Graphs for Retrieval (Graph RAG)
To overcome the limitations of flat-text retrieval, a growing body of work has focused on Graph RAG [ 13].
Frameworks like Youtu-GraphRAG have demonstrated the power of this approach [ 5]. The architectural
3

advantage of a knowledge graph is that it provides a persistent, structured, and connected memory that
an external agent can query, enabling more intricate reasoning tasks than are possible with isolated text
chunks [14].
Our SAT-Graph RAG [ 4] aligns with this trend but makes a critical distinction. Unlike generalist
approaches that discover a graph "bottom-up" from unstructured text, the SAT-Graph is constructed "top-
down," mapping thepre-existing, formal structureof the law onto a deterministic, temporal ontology. The
most important entities are not named persons or places, but the structural components of the law itself
and the legislative events that alter them. While this provides a robust knowledge base, the question of a
standardized interaction protocol remains open.
2.4 Agentic Frameworks for Reasoning and Retrieval
Recent advances have seen LLMs employed not just as generators, but as reasoning agents capable of
planning and tool use [ 15]. These agents can decompose complex questions and utilize a "library of tools" to
gather information. The API we propose is, in effect, this specialized tool library for the legal graph.
A critical insight from recent research is the symbiotic relationship between architectural innovation and
evaluation benchmarks. The legal domain has historically suffered from a lack of benchmarks that require
deep, multi-hop reasoning over structured, temporal data. However, new, more challenging benchmarks are
emerging, such as "Bar Exam QA" and "Housing Statute QA," which are specifically designed to require
"reasoning-intensive" queries rather than simple lexical overlap [ 16]. These new benchmarks expose the deep
limitations of traditional RAG and create a clear need for architectures like ours. The SAT-Graph, combined
with our Canonical Action API, is precisely the type of verifiable, structure-aware framework needed to
successfully address these next-generation evaluation tasks.
2.5 Positioning Our Contribution
Our work synthesizes these threads. We build upon the foundation of our SAT-Graph, which addresses the
representation problem in legal AI. We acknowledge the power of agentic reasoning frameworks for flexible
query resolution. However, we identify a critical missing component: thesecure and auditable bridge
between the reasoning agent and the structured knowledge.
While prior work has focused on either the graph structure itself or the high-level agentic logic, our
contribution is the formal specification of the intermediate layer: aCanonical Action API. This is not another
monolithic retrieval system, but a low-level protocol that provides the necessary safety guarantees for the
legal domain. It enables advanced agents to perform complex reasoning while ensuring that every interaction
with the knowledge base is explicit, verifiable, and grounded in the temporal and structural reality of the law.
3 Specification of the Canonical Action API
The central thesis of our work is that trustworthy interaction with a structured knowledge base like the
SAT-Graph requires a formal, auditable protocol. We introduce this protocol as the Canonical Action
API, a library ofatomic and composable actions. This API serves as an abstraction layer, hiding the
underlying graph database implementation (e.g., Cypher queries over a Neo4j database). Crucially, it provides
a unified interface for bothprobabilistic discovery actions(like resolving natural language references) and
fully deterministic retrieval actionsthat operate on formal identifiers. In this pattern, an external agentic
framework is designed to interact with the knowledge base primarily through these actions, ensuring that
every step in a retrieval plan is explicit, verifiable, and its level of certainty is understood, as shown in Figure
2.
4

Figure 2: Diagram illustrating a reasoning agent decomposing a user prompt into tasks that are executed
through actions provided by the SAT-Graph API.
3.1 Design Principles
The design of the API is guided by three core principles that, together, ensure trustworthiness even when
interfacing with probabilistic language models.
•Maximal Determinism:We acknowledge that any system interpreting natural language has inherently
probabilistic components. Our design philosophy is toisolatethis non-determinism at the API’s entry
points. Actions for discovery and search are probabilistic by nature. However, the core architectural
contribution of this API is to enable the orchestration offully deterministic retrieval plansonce a
formal identifier is obtained. It is this capability for achieving determinism in the final, grounded query
that fundamentally distinguishes our approach and justifies the "Deterministic Retrieval" paradigm.
Once a formal id (URI/URN) is acquired, all subsequent actions that operate on it are guaranteed to be
deterministic.
•Composability:Actions are designed as atomic "building blocks." They can be chained together by
a planning agent to construct complex query workflows. This allows for flexibility while ensuring
that the overall plan is composed of individually verifiable steps, clearly distinguishing between initial
probabilistic grounding and subsequent deterministic traversal.Server-side execution of composed
actions ensures efficient processing even for complex multi-step plans.
•Verifiability through Auditability:This principle is our safeguard against the initial non-determinism.
Every action, including the probabilistic ones, must return not just a result but also a confidence score
or justification. When an agent executes a plan, the sequence of action calls and their outputs forms a
complete, human-readable audit trail. This log makes the entire processverifiable, even if not every
step is deterministic. A user can inspect the log and see, "The system resolved ’Article 6’ to id ’...’
with 98% confidence," allowing them to trust or question the initial grounding step.
5

3.2 Core Data Models
The actions in our API operate on and return a set of core data objects that correspond directly to the entities in
the SAT-Graph ontology, as illustrated in Figure 3. The ontology makes a clear distinction between versioned
normative entities (Item) and atemporal conceptual entities (Theme).
Figure 3: SAT-Graph Ontology
Item: Represents a versioned, structural entity of a legal norm. It is instantiated as one of two concrete
subtypes:WorkorWork Component.
•id (ID):A unique, canonical identifier (e.g. URI/URN).
•category (string):The architectural class (‘Work‘ or ‘Work Component‘).
•type (string):The specific semantic subtype (‘Constitution‘, ‘Article‘, ‘Chapter‘).
•label (string):The human-readable label.
•parent (id)?:(Optional) The id of its single structural parent. A ‘Work‘ has no parent.
•children (list[id])?: (Optional) A list of ids for its structural children. Leaf nodes
have no children.
•metadata (JSON)?: (Optional) A flexible json store for structured, atemporal properties of
the item—examples of metadata description of a legal norm’s Work in [17].
Theme: Represents an atemporal, conceptual entity used to classify Item s and to organize knowledge.
Themes can be arranged in their own taxonomy.
•id (ID):A unique, canonical identifier.
•label (string):The human-readable label.
•parent (list[id])?: (Optional) A list of ids of broader themes. Themes can have
multiple parents, creating a directed acyclic graph (DAG) structure. Root themes have an empty
list.
6

•children (list[id])?:(Optional) A list of ids of more specific themes.
•members (list[id])?:(Optional) A list of ids ofItems classified under this theme.
Architectural Note: Separation of Normative and Conceptual EntitiesA foundational design principle
of this framework is the strict separation of normative entities ( Item ) from conceptual, classificatory
entities ( Theme ).Item s represent the formal, versioned structure of the law itself. Theme s, in contrast,
form an independent, atemporal ontology used to organize and discover knowledge. An Item ’s structural
relationships (‘parent‘, ‘children‘) define its fixed position within a document, forming a strict hierarchy.
ATheme ’s relationships define both its taxonomic position relative to other themes (‘parent‘, ‘children‘)
and its classificatory relationship to the law (‘members‘). This clear separation allows an agent to navigate
the formal structure of a norm (or component) and the conceptual map of the legal domain through distinct,
unambiguous API actions.
The ‘Version‘ (Item Temporal Version) and ‘Action‘ data models build upon this DAG foundation to
create a fully versioned and causally-linked graph.
Version: Represents a specific, time-bound version of an Item . An Version captures not only the content
but also the structural position of an item at a specific point in time.
•id (ID):A unique identifier for this specific version.
•item_id (ID):The id of the abstractItemit is a version of.
•validity_interval ([date, date])?: A tuple representing the start and end dates
of its validity. The end date is optional (or null) to represent a currently valid or open-ended
version.
•parent (list[ID])?: (Optional) A list of identifiers for its parent Version s. A ‘Work‘’s
Version has no parents.
•children (list[ID])?: (Optional) A list of identifiers for its children Version s, repre-
senting the complete structure at this point in time.
•metadata (JSON)?: (Optional) A flexible json store for structured properties of the version.
Action: Represents a reified legislative event as the most granular, atomic unit of change in the graph.
It is a first-class entity that models a single, auditable causal link, connecting a specific clause of an
authorizing law to the specific change it enacts.
•id (ID):A unique identifier for the action.
•type (string): The type of legislative action (e.g., ‘Amendment‘, ‘Revocation‘, ‘Cre-
ation‘).
•date (date):The effective date of the action.
•source_version_id (ID): The identifier for the specific Version of the legal text that
authorizesthis single change.
•terminates_version? (ID): (Optional) The specific Version whose validity ister-
minatedby this action.
•produces_version (ID):The specificVersionthat iscreatedby this action.
TextUnit: Represents a concrete piece of textual information. It is a flexible object used to hold any
"aspect" of text associated with any primary entity in the graph. This could be the canonical text of a
versioned item (an Version ), a detailed description of a Theme , or a textualized metadata aspect of
anItem(like a publication info.).
7

•id (string):A unique identifier for this specific piece of text.
•source_node_type (string): The type of the source node (e.g., " Item ", "Version ",
"Action", "Theme"), to allow for unambiguous retrieval by an agent.
•source_node_id (ID): The unique identifier of the node in the graph to which this text is
attached.
•language (string):The language of the content (e.g., "pt-br").
•aspect (string): The semantic role of the text relative to its source node (e.g., "canonical"
text of a Version, "description" of a Theme, "textual_metadata" of a Work).
•content (string):The raw textual content.
3.3 API Action Specification
The API is organized into functional categories based on agentic intent. A core design guarantee governs all
actions: any action that accepts a formal ID as its primary input is fully deterministic. The only exceptions
are the discovery and search actions, which interpret natural language text. This section details the signature
and purpose of each action.
Bridging Baseline RAG with a Temporal Knowledge GraphA key design challenge is to provide a
search interface that is both simple enough to replicate the behavior of a standard, "temporally-naïve" RAG
system, while also exposing the full power of the underlying temporal graph. Traditional RAG pipelines
typically index a single, current version of a text. A query like “Which social rights are guaranteed?” executes
a semantic search over these current texts.
The SAT-Graph, however, containsall historical versions. A simple semantic search would return a mix
of current and outdated, yet relevant, texts. To manage this complexity, the API’s primary search primitive,
searchTextUnits , is explicitly designed to handle two core user intents through a simple timestamp
parameter:
•Default (Current State Query):When the timestamp is omitted, the search is automatically constrained
to only the currently valid texts, thus replicating the expected behavior of a baseline RAG.
•Temporal Query:When a timestamp is provided, the search is constrained to the historical snapshot of
texts that were valid on that specific date.
This design provides a single, powerful action that elegantly bridges the gap between simple RAG queries
and complex historical analysis.
3.3.1 Discovery and Search Actions
These actions serve as the primary entry points for an agent, used when it needs to find entities based on
ambiguous natural language references or semantic content.
1.resolveItemReference(reference_text: string, context_id?: ID,
top_k?: int)→list[{ID, confidence: float}]
•Description:Translates an ambiguous, natural-language reference to a document/component into
aranked list of candidate Item ids. This is the critical first step for any query that names a
specific item, acting as the bridge between user language and the graph’s formal identifiers.
•Parameters:
–reference_text : A string such as "Article 5 of the Constitution", "the Civil Code", or a
relative reference like "the previous section".
8

–context_id : (Optional) The id of an Item that provides a structural context. This allows
the system to accurately resolve relative references and disambiguate ambiguous ones (e.g.,
resolving "Article 10" to the one within the same law as the context id).
–top_k : (Optional) The maximum number of candidate matches to return (e.g., defaults to
3).
•Returns:A ranked list of objects, each containing a probable id and a confidence score indicating
the certainty of the match.
•Discussion:This action is indispensable for grounding agentic workflows. Document/component
references are often highly ambiguous ("Article 1" could exist in hundreds of laws) or relative
("the following paragraph"). This function handles this by returning a ranked list of candidates
rather than forcing a single, potentially incorrect, guess. The optionalcontext_idparameter
significantly enhances its precision, enabling sophisticated, context-aware interactions. This
design makes the entire system safer and more robust, preventing critical errors that arise from
incorrect initial grounding.
2.resolveThemeReference(reference_text: string, top_k?: int)→
list[{ID, confidence: float}]
•Description:Translates a natural-language name or label for a legal theme into aranked list
of candidate Theme ids. This is the primary entry point for grounding a query in a specific
conceptual category.
•Parameters:
–reference_text : A string representing the name of a theme, such as "Social Security"
or "Environmental Law".
–top_k : (Optional) The maximum number of candidate matches to return (e.g., defaults to
3).
•Returns:A ranked list of objects, each containing a probableThemeid and a confidence score.
•Discussion:This action is the conceptual counterpart to resolveItemReference . It per-
forms entity linking for thematic nodes, allowing an agent to deterministically identify a theme
mentioned by name before using it as a scope in other actions likesearchTextUnits.
3.searchTextUnits(version_ids?: list[ID], item_ids?: list[ID],
theme_ids?: list[ID], metadata_filter?: MetadataFilter,
timestamp?: date, semantic_query?: string, lexical_query?: string,
language?: string, aspects?: list[string], top_k?: int)→
list[{text_unit: TextUnit, score: float}]
•Description:Performs a hybrid search to discover TextUnit s nodes by combining semantic,
lexical, and structured criteria returning a ranked list of the most relevant results, ready to be used
as context for a generation model. This is the primary, optimized entry point for all standard RAG
queries.
•Parameters:
–version_ids : (Optional) A list of specific Version (Item Temporal Version) identifiers.
Provides an explicit, pre-resolved versioned scope. This parameter ismutually exclusive
with the combination ofitem_ids/theme_idsandtimestamp.
9

–item_ids : (Optional) A list of Item ids. Defines the structural scope. The search includes
these items and their descendants.
–theme_ids : (Optional) A list of Theme ids. Expands the search scope to include all
member Item s of the specified themes and all of their descendant themes. For example,
searching within the "Tribute Law" theme will automatically include items from its sub-
themes like "Federal Taxes" and "State Taxes".
–metadata_filter : (Optional) A structured object ( MetadataFilter ) that specifies
structured filters to be applied. This object can contain clauses like:
*item_metadata_filter (dict) : (Optional) A set of key-value pairs to filter
against themetadataof theItems (e.g., ‘"jurisdiction": "federal"‘).
*version_metadata_filter (dict) : (Optional) A set of key-value pairs to
filter against the metadata of the Version itself (e.g., ‘"publication_date": ">=2010-
01-01"‘).
–timestamp : (Optional) A date to constrain the search when using item_ids ortheme
_ids . If omitted, defaults to the current date ("now"). This parameter is ignored if
version_idsis provided.
–semantic_query (string): (Optional) The natural language query string.
–lexical_query (string) : (Optional) A set of keywords or phrase for a full-text
syntactic search.
–language: (Optional) A code for the desired language.
–aspects : (Optional) A list of strings specifying the desired textual aspects (e.g., [‘"canoni-
cal"‘, ‘"summary"‘]). Defaults to [‘"canonical"‘].
–top_k: (Optional) The maximum number of results to return.
•Returns:A ranked list of objects, each containing a matching TextUnit and its similarity score.
•Discussion:This action’s dual-mode design for scoping provides maximum flexibility.
When to use Mode 1 (Explicit Version Scope):Use when you have already identified a specific
set of Versions (e.g., from ‘getItemHistory‘) and need to search exclusively within that curated
set. This provides a direct, high-performance path for advanced analytical tasks.
When to use Mode 2 (Conceptual Scope):Use for natural language queries like "What did
Law X say about topic Y in 2005?" The agent defines the "what" (item_ids/theme_ids) and the
"when" (timestamp), and the API handles temporal resolution automatically.fic set of Version s
of interest (e.g., from getItem History ) and needs to perform a semantic search exclusively
within that curated set. Using this mode provides a direct, high-performance path for advanced
analytical tasks.
Furthermore, the aspects parameter enables a truly holistic search across the entity hierarchy.
By default, the search targets the "canonical" text of the relevant Version s. However,
by specifying other aspects (e.g., ["canonical", "textual_metadata"] ), the search
space expands to include all TextUnit s matching those aspects, regardless of whether they are
attached to the filtered Version s themselves or to their timeless parent Item s. This allows an
agent to, in a single query, find a match in a Work ’s "publication info." just as easily as in an
Version’s official text, creating a comprehensive and context-aware retrieval process.
4.searchItems(item_ids?: list[ID], theme_ids?: list[ID],
item_metadata_filter?: dict, semantic_query?: string,
lexical_query?: string, top_k?: int)→
list[{item: Item, score: float}]
10

•Description:Performs a hybrid search to discover Item nodes (the structural entities of the
graph) by combining semantic, lexical, and structured criteria. This action searches across the
entire version history of each item’s text to ensure comprehensive discovery.
•Parameters:
–item_ids: (Optional) A list ofItemids to define the search scope.
–theme_ids: (Optional) A list ofThemeids to expand the search scope.
–item_metadata_filter (dict) : (Optional) A set of key-value pairs to filter against
themetadataof theItem(e.g., ‘"jurisdiction": "federal"‘).
–semantic_query (string): (Optional) The natural language query string.
–lexical_query (string) : (Optional) A set of keywords or phrase for a full-text
syntactic search.
–top_k: (Optional) The maximum number of results to return.
•Returns:A ranked list of objects, each containing a matchingItemand its similarity score.
•Discussion:This action’s purpose is structural discovery, not content retrieval. When an agent
usessearchItems , it is attempting to find thelocusof a concept within the graph’s formal
structure. To support this, the search must be comprehensive, and is therefore intentionally
designed to be time-agnostic, considering content from an item’s entire version history.
The distinction fromsearchTextUnitsis critical:
–To findthe textabout "housing" that was valid in 2005, an agent must use searchTextUnits(
..., timestamp="2005-01-01").
–To findthe itemthat, at any point in its history, has dealt with "housing", an agent must use
searchItems.
Adding a timestamp to this function would create a confusing overlap and undermine its core
purpose as a high-recall structural discovery tool. The item_metadata_filter parameter,
in contrast, enhances this purpose by enabling high-precision discovery based on the atemporal
metadata of the structural nodes themselves.
5.searchThemes(semantic_query?: string, lexical_query?: string,
top_k?: int)→list[{theme: Theme, score: float}]
•Description:Performs a hybrid search to discover Theme nodes based on a descriptive query.
This action searches exclusively on the textual descriptions associated with the themes themselves,
not on the content of their member items.
•Parameters:
–semantic_query (string): (Optional) The natural language query string.
–lexical_query (string) : (Optional) A set of keywords or phrase for a full-text
syntactic search.
–top_k: (Optional) The maximum number of candidate themes to return.
•Returns:A ranked list of candidate objects, each containing a theme’s id and a confidence score.
•Discussion:This action’s purpose is to map a user’s description of a topic to the most appropriate
formal concept in the graph’s ontology. To ensure a precise and high-performance conceptual
match, the search is intentionally scoped to the curated textual descriptions of the Theme nodes.
Searching the content of all member Item s would introduce significant semantic noise and lead
to inaccurate results, as large themes could overshadow more specific ones. This design choice
maintains a clear separation of concerns: use searchThemes to find the right conceptual
category, then use searchTextUnits with the resulting theme’s id to find relevant content
within that category.
11

3.3.2 Deterministic Fetch Actions
Once an agent has acquired a specific ID, these actions are used to retrieve the full, corresponding data objects
in a deterministic, non-ambiguous way.
6.getItem(id: ID)→Item
•Description:Retrieves the fullItemobject given its unique id.
•Parameters:
–id: The canonical id of the item.
7.getTheme(id: ID)→Theme
•Description:Retrieves the fullThemeobject given its unique id.
•Parameters:
–id: The canonical id of the theme.
8.getAction(id: ID)→Action
•Description:Retrieves the full Action data object given its unique identifier. This provides
access to all details of a action, including its precise ‘source_version_id‘.
•Parameters:
–id: The unique identifier of the action.
•Returns:The correspondingActiondata object.
9.getValidVersion(item_id: ID, timestamp: date-time, policy:
TemporalPolicy)→Version
•Description:The core temporal resolution function. It finds the single Version that was legally
in force on a specific date.
•Parameters:
–item_id: The id of the item of interest.
–timestamp: The specific date for which to check validity.
–policy : An enum specifying the resolution logic, e.g., ‘TemporalPolicy.SnapshotLast‘
which finds the version whose validity interval contains the timestamp.
•Returns:The uniqueVersionobject valid on the given date.
•Discussion:The explicit ‘policy‘ parameter is essential for guaranteeing determinism. It allows
the agent to specify the exact temporal logic required, such as retrieving the last valid version
within a day (snapshot semantics) or handling boundary conditions in a predictable manner. The
policy parameter controls temporal resolution: PointInTime : Returns the version valid at
exactly the specified timestamp. SnapshotLast : Returns the last known version before or at
the timestamp. Default: PointInTime.
10.getTextForVersion(version_id: ID, language: string)→TextUnit
12

•Description:Deterministically retrieves a specific textual representation (a TextUnit ) for
a single, known Version . This is a direct-access, primary-key-based lookup, not a search
operation.
•Parameters:
–version_id: The unique identifier of theVersionobject whose text is requested.
–language: A code for the desired language (e.g., "pt-br", "en").
•Returns:The single, correspondingTextUnitobject.
•Discussion:This action is fundamentally different from searchTextUnits . While the latter
is a probabilistic discovery tool based on semantic similarity, getTextForVersion is a
deterministic fetch operation. It serves the critical use case where an agent has already identified
a specific Version (e.g., via getValidVersion orgetItemHistory ) and requires its
exact textual content. Maintaining this as a distinct, highly-optimized action ensures API clarity,
guarantees determinism, and provides a clear audit trail of the agent’s execution plan.
3.3.3 Structural Navigation Actions
These actions are used to traverse the atemporal, hierarchical, and thematic structures of the graph.
11.enumerateItems(item_ids?: list[ID], theme_ids?: list[ID],
depth?: int)→list[Item]
•Description:Enumerates the structural or thematic members of a given scope. This is the primary
action for navigating the graph’s hierarchy and membership relationships.
•Parameters:
–item_ids : (Optional) A list of Item ids. Defines the structural scope. The search includes
these items and their descendants.
–theme_ids : (Optional) A list of Theme ids. Expands the scope to include all member
items of these themes.
–depth : (Optional) An integer controlling the traversal depth for hierarchical Item scopes.
If ‘depth=1‘, only direct children are returned. If omitted or negative, all descendants are
returned recursively. This parameter is ignored forThemescopes.
•Returns:A list of Item objects. The behavior is polymorphic based on the type of the input
‘scope_id‘:
–For a structuralItemscope: it returns its hierarchical children/descendants.
–For aThemescope: it returns all of its memberItems.
•Discussion:This action’s purpose is structural enumeration and navigation, not search filtering.
It is the correct tool for tasks like building a table of contents or programmatically exploring the
members of a conceptual theme. For performing content-based searches within a scope, agents
should pass the scope’s id directly to the item_ids ortheme_ids parameters of the search
functions ( searchTextUnits ,searchItems ), as this is significantly more performant than
retrieving the members first and then passing a long list of IDs.
12.getAncestors(item_id: ID)→list[Item]
13

•Description:Navigates the hierarchy upwards from a specific structural Item to its parent,
grandparent, and so on, up to the root Work . This is crucial for retrieving the full contextual path
of any given provision.
•Parameters:
–item_id: The id of the startingItem.
•Returns:An ordered list of Item objects representing the hierarchical path, from the highest-
level ancestor (e.g., a ‘Title‘) down to the immediate parent of the input item. The root Work
itself is typically excluded.
•Discussion:This action is the fundamental tool for contextualization. The legal meaning of
a provision is often determined by the Chapter or Title it belongs to. After a search action
likesearchTextUnits identifies a relevant item, an agent should use getAncestors to
retrieve its structural "breadcrumb". This allows the generation of richer, context-aware responses,
significantly improving the quality and explainability of the final answer.
13.getThemesForItem(item_id: ID)→list[Theme]
•Description:Retrieves a list of all Theme s that are directly associated with a specific Item .
This provides the inverse navigation from the structural hierarchy to the conceptual ontology.
•Parameters:
–item_id: The id of theItemwhose thematic classifications are requested.
•Returns:A list of allThemeobjects that classify the givenItem.
•Discussion:This action is the essential counterpart to using a Theme as a scope in enumerate
Items . It makes the relationship between items and themes fully bidirectional, enabling an
agent to efficiently answer questions like "What topics does this article cover?". Without this
primitive, an agent would need to resort to a highly inefficient brute-force search across the entire
thematic ontology. This function completes the API’s structural navigation toolkit.
14.getItemContext(item_id: ID)→StructuralContext
•Description:Retrieves the complete, immediate structural context of a single Item in one
efficient call. This includes its parent, siblings, and direct children.
•Parameters:
–item_id: The id of theItemfor which to retrieve the context.
•Returns:A StructuralContext object containing the full data for the target item and its
immediate hierarchical neighbors.
–target (Item):The full object of the item in question.
–parent (Item)?:The full object of its immediate parent.
–siblings (list[Item])?: A list of full objects for its siblings (other items with the
same parent).
–children (list[Item])?:A list of full objects for its immediate children.
•Discussion:This is a high-utility, performance-oriented action designed for applications that
need to render a contextual view of an item (e.g., a document browser or a table of contents).
Replicating this functionality would require multiple, less efficient API calls to fetch ancestors,
descendants, and then compute the sibling set. By providing this as a single, atomic operation,
the API dramatically simplifies the logic for common UI and agentic navigation tasks.
14

3.3.4 Causal & Lineage Analysis Actions
These actions enable agents to traverse the temporal and causal dimensions of the SAT-Graph, understanding
the history and evolution of legal items.
15.getItemHistory(item_id: ID)→list[Action]
•Description:Provides a complete, ordered timeline of all actions that have affected a given Item .
•Parameters:
–item_id: The id of theItemwhose history is requested.
•Returns:A time-ordered list of all Action objects that have either created or terminated a
version (Version) of the specifiedItem.
•Discussion:This action provides a complete, auditable narrative of an item’s evolution. An agent
can iterate through the returned list of Action s to reconstruct every state transition, accessing the
"before" state via ‘action.terminates_version‘ and the "after" state via ‘action.produces_version‘
for each step.
16.traceCausality(version_id: ID)→{creating_action: Action,
terminating_action?: Action}
•Description:Traces the direct causal links for a single Version . It identifies the specific
Action that brought this Version into existence (its provenance) and, if applicable, the
Actionthat terminated its validity.
•Parameters:
–version_id: The unique identifier of theVersion.
•Returns:An object containing the Action that created the Version and, optionally, the
Actionthat terminated it.
•Discussion:This action is crucial for pinpointing the exact legislative act responsible for a specific
version of a legal text, facilitating precise auditability.
17.getVersionsInInterval(item_id: ID, start_date: date,
end_date: date)→list[Version]
•Description:Retrieves all Version s of a given Item that were legally valid at any point within
a specified time interval.
•Parameters:
–item_id: The id of the item of interest.
–start_date: The start date of the time interval for the query.
–end_date: The end date of the time interval for the query.
•Returns:A time-ordered list of all Version objects whose validity period overlaps with the
specified interval.
•Discussion:This function serves as a critical performance optimization for historical analysis.
While getItemHistory retrieves the causalevents(‘Actions‘), this action retrieves the re-
sultingstates(‘Versions‘) within a specific temporal window. Replicating this on the client-side
would require fetching the entire history and performing complex filtering. By providing this
15

as a server-side primitive, the API enables an agent to efficiently answer questions like "show
all versions of this article during a specific presidential term," which is essential for building
responsive and scalable legal research applications.
18.compareVersions(version_id_A: ID, version_id_B: ID)→TextDiffReport
•Description:Performs a granular, structure-aware comparison between the textual content of two
differentVersions of the sameItem. It highlights additions, deletions, and modifications.
•Parameters:
–version_id_A: Identifier of the firstVersion.
–version_id_B: Identifier of the secondVersion.
•Returns:A structuredTextDiffReportobject detailing the changes.
•Discussion:This action provides a powerful, server-side analysis that goes beyond a simple
textual ‘diff‘. Because the server has access to the full graph structure, it can perform astructural
comparison, identifying not just changed words but also added or removed components (e.g.,
a new paragraph). This transforms the comparison from a mere text-level operation into a true
structural impact analysis, which is fundamental for legal practitioners. Centralizing this complex
logic ensures high performance and a consistent, high-quality analysis for all client applications.
19.getActionsBySource(source_work_id: ID, action_types?: list[string])
→list[Action]
•Description:An essential action for impact analysis that provides forward causal tracing. Given
the id of a source Work (e.g., an amending law), it retrieves all atomic Action s authorized by
any clause within that Work.
•Parameters:
–source_work_id: The id of the sourceWork(e.g., "Law No. 50 of 2001").
–action_types : (Optional) A list of strings to filter the results to specific types of actions
(e.g., [‘Revocation‘, ‘Creation‘]).
•Returns:A comprehensive list of all Action objects caused by the specified source that match
the filter criteria.
•Discussion:This action is the logical inverse of traceCausality . While the latter traces
backwards from a single effect to its granular cause, getActionsBySource traces forwards
from a high-level cause (an entire legislative document) to all of its granular effects. It is the
primary, high-performance primitive for answering the critical question: "What did Law X do?".
The optionalaction_typesfilter allows for more targeted impact analysis, such as isolating
only the revocations or creations enacted by a specific law.
3.3.5 Macro-Level Analysis Actions
Beyond entity retrieval and causal tracing, the Canonical Action API provides a distinct class of actions
designed for high-level, aggregate analysis. These functions represent a deliberate architectural choice to
offload computationally intensive processing from the client to the server. While an agent could theoretically
replicate their logic by making numerous, sequential calls to lower-level primitives, doing so would be highly
inefficient and impractical for large-scale analysis.
By treating server-side aggregation as a core primitive, the API enables agents and applications to
efficiently answer broad, macro-level questions about legislative evolution. Actions in this category typi-
cally do not return lists of full entity objects, but rather structured, lightweight summary objects (like an
16

ImpactReport ) containing aggregate statistics and lists of identifiers, designed to be selectively hydrated
by the agent as needed.
20.summarizeImpact(item_ids?: list[ID], theme_ids?: list[ID],
time_interval: TimeInterval, action_types?: list[string])→
ImpactReport
•Description:An aggregate analysis action that provides a high-level summary of changes that
occurred within a specific scope during a defined time period.
•Parameters:
–item_ids: (Optional) A list ofItemids to define the structural scope of the analysis.
–theme_ids: (Optional) A list ofThemeids to define the thematic scope of the analysis.
–time_interval: A start and end date for the analysis window.
–action_types : (Optional) A list of strings to filter the summary to specific action types
(e.g., [‘Revocation‘, ‘Creation‘]). If omitted, all action types are included.
•Returns:A structured ImpactReport object, containing aggregate statistics (e.g., counts of
each action type) and lists of identifiers for the relevantActions and affectedItems.
•Discussion:This powerful action enables macro-level analysis and is a deliberate design choice
to offload complex processing to the server. Replicating its logic on the client-side would require
numerous, inefficient calls. By treating server-side aggregation as a core primitive, the API
ensures efficiency and simplifies agent logic for this common legislative analysis use case. The
ImpactReport intentionally contains lightweight identifiers, allowing the agent to selectively
hydrate full details using batch operations likegetBatchActionsas needed.
3.3.6 Introspection & Metadata Actions
To enable agents to be more adaptive and robust, the API provides a set of introspection actions that allow
them to discover the capabilities and boundaries of the knowledge graph at runtime.
21.getTemporalCoverage(item_id: ID)→TimeInterval
•Description:Returns the start and (optional) end dates of the known version history for a specific
Item.
•Returns:A tuple containing the start date of the first Version and the end date of the last known
Version. The end date is null if the item is still considered active.
•Discussion:This action enables an agent to be aware of its own knowledge boundaries. It should
be used to proactively validate a user’s temporal query before attempting retrieval, allowing for
graceful handling of requests for dates outside the known history and improving the overall user
experience.
22.getAvailableLanguages()→list[string]
•Description:Returns a list of all language codes (e.g., "pt-br", "en") for which textual content is
available in the graph.
•Discussion:This enables an agent or user interface to dynamically present language options to
the user, ensuring that requests are only made for languages that the system actually supports.
17

23.getSupportedActionTypes()→list[string]
•Description:Returns a list of all action types (e.g., ‘Amendment‘, ‘Revocation‘) recognized by
the graph’s ontology.
•Discussion:This action provides the canonical list of values that can be used in the action_types
filter of other functions like summarizeImpact . It allows a client application to dynamically
build filtering interfaces without hardcoding ontology-specific values.
24.getRootThemes()→list[Theme]
•Description:Retrieves a list of the full Theme objects that are at the root of the thematic
taxonomy (i.e., themes that have no parents).
•Returns:A list of all rootThemeobjects.
•Discussion:This action serves as the primary and most efficient entry point for an agent or user
interface to explore the thematic ontology. By returning the full root objects, it allows a client to
render the top-level categories without requiring subsequent API calls.
3.3.7 Architectural Rationale and Modeling Principles
Discussion: Modeling of State TransitionsOur framework is designed to handle various types of leg-
islative state transitions with precision. A notable example is the handling of revocations. A recommended
practice for maintaining a continuous version timeline is to model a revocation not just as the termination
of the last valid Version , but also as the creation of a new, subsequent Version with a specific textual
content, such as "(Revoked)". However, the API is flexible enough to model this differently. This flexibility
is fully represented in the Action data model, which can cleanly represent an original creation (with only
aproduces_version_id ), a simple repeal (with only a terminates_version_id ), or a standard
amendment (using both).
Discussion: Atomic Actions and Emergent Event CohesionA core design principle of the SAT-Graph is
the atomicity of causal events. Our model defines an Action as the most granular unit of change possible to
ensure maximum auditability. A key consequence is that a single, complex legislative event—such as the
enactment of "Constitutional Amendment No. 99"—is not represented by a monolithic node. Instead, its
total effect is captured by anemergent collectionof many individual Action nodes, whose cohesion is
established by their respective source_version_id properties all tracing back to the same source Work .
This design provides the best of both worlds. An agent can use a single, high-level API action to
reconstruct the full scope of an event:
•Strategy: Aggregate Actions by Source Work.The agent first identifies the amending law as a
Work (e.g., via resolveItemReference ). It then makes a single call to getActionsBySource(
source_work_id=...)to gather all the individualActionnodes authorized by that amendment.
This powerful aggregation enables high-level impact analysis ("What were all the effects of this amend-
ment?"), while the atomic nature of each returned Action preserves the granularity needed for pinpoint-
accurate causal tracing.
Discussion: Multi-Aspect Retrieval via TextUnitThe TextUnit is the foundational object for the
multi-aspect retrieval capability of the SAT-Graph. By decoupling textual content from the core graph
entities, we can associate multiple, distinct textual representations with any node. This is achieved via
thesource_node_type andsource_node_id properties on the TextUnit , allowing it to point
to any other entity in the graph. For example, an entire Work can have a TextUnit with an aspect of
18

"textual_metadata", a Theme can have one with an aspect of "description", and a Version can have both a
"canonical" text and a "summary".
This flexibility is directly exposed through the API, particularly via the aspects parameter in functions
likesearchTextUnits . AllTextUnit s, regardless of their source entity or aspect, are candidates for
embedding and semantic retrieval. This creates a rich, multi-faceted search space where an agent can retrieve
information based not just on the law’s formal content, but also on its context, metadata, and conceptual
definitions in a single, unified query.
3.4 Practical Implementation Considerations
While the actions specified above define the conceptual and logical interface to the SAT-Graph, a production-
grade implementation should include additional features for efficiency and robust operation.
Batch OperationsTo mitigate network latency in scenarios requiring numerous lookups—such as recon-
structing the state of a constitutional chapter with 50 articles at a specific date, which would otherwise require
50 sequential API calls—, such as resolving the point-in-time structure of a complex item or analyzing its
full history, the API should providebatch-enabled versionsof its core retrieval actions. Key batch actions
include:
25.getBatchItems(ids: list[ID])→list[Item]
•Description:A batch-optimized version of ‘getItem‘. Given a list of ids, it efficiently retrieves
the full data object for each correspondingItemin a single request.
•Parameters:
–ids: A list of ids for the items of interest.
•Returns:A list of the requested Item objects. The returned list may be shorter than the input list
if some ids were not found.
•Discussion:This is a fundamental utility action used after any discovery process. When an agent
obtains a list of ids from functions like ‘searchItems‘ or ‘enumerateItems‘, this batch action
allows it to hydrate those references into full objects to access their labels, types, or hierarchical
relationships without incurring the high cost of N+1 individual requests.
26.getBatchValidVersions(item_ids: list[ID], timestamp: date-time,
policy: TemporalPolicy)→list[Version]
•Description:A batch-optimized version of ‘getValidVersion‘. Given a list of item ids, it efficiently
retrieves all of their correspondingVersions that were valid on a single, specified date.
•Parameters:
–item_ids: A list of ids for the items of interest.
–timestamp: The specific date for which to check validity for all items.
–policy: The temporal resolution policy to apply uniformly.
•Returns:A list of the valid Version objects. Note that the returned list may be shorter than the
input list if some items did not have a valid version on the given date.
•Discussion:This action is essential for efficiently reconstructing the state of a complex hier-
archical item at a specific point in time. It is designed to be used directly with the output of
‘enumerateItems‘, avoiding the "N+1 query problem" and drastically improving performance.
19

27.getBatchTexts(version_ids: list[ID], language: string,
aspects?: list[string])→list[TextUnit]
•Description:A batch-optimized version of getTextForVersion . Given a list of Version
identifiers, it efficiently retrieves their corresponding textual content.
•Parameters:
–version_ids: A list of identifiers for the Versions of interest.
–language: The language to retrieve.
–aspects : (Optional) A list of textual aspects to retrieve for all Versions (e.g., [‘"canonical"‘,
‘"summary"‘]). Defaults to [‘"canonical"‘].
•Returns:A list of the requestedTextUnitobjects.
•Discussion:This action is critical for use cases that require inspecting the content of multiple
versions, such as finding the exact point of a textual change within a long version history.
It prevents the severe performance degradation that would result from iterating and calling
getTextForVersionindividually for each version.
28.getBatchActions(ids: list[ID])→list[Action]
•Description:A batch-optimized version of ‘getAction‘. Given a list of action identifiers, it
efficiently retrieves the full data object for each correspondingActionin a single request.
•Parameters:
–ids: A list of unique identifiers for the actions of interest.
•Returns:A list of the requested Action objects. The returned list may be shorter than the input
list if some identifiers were not found.
•Discussion:This action is essential for the "hydration" step of analytical queries. After an agent
discovers a set of relevant ‘Action‘ IDs via a function like ‘summarizeImpact‘, this batch operation
allows it to retrieve the full details of each action (such as their specific ‘source_version_id‘)
to build a complete and auditable report without the high performance cost of N+1 individual
requests.
3.5 API as a Foundation for Agentic Reasoning
The specified actions collectively form a comprehensive yet controlled interface to the legal knowledge
graph. They are intentionally designed to be low-level. An agentic framework is not expected to simply call
one action to answer a user’s query. Instead, a planning module is responsible for decomposing a complex
question into a sequence or a Directed Acyclic Graph (DAG) of these actions. The power of this approach is
best illustrated with examples.
Example 1: Point-in-Time Retrieval PlanA fundamental question like "What was the text of Article
6 of the Constitution in 1999?" would be decomposed into a clean, sequential plan. The agent’s goal is to
deterministically find the single correct textual version for that specific date.
1.Ground the reference:It first calls resolveItemReference(referenceText="Article
6 of the Constitution")to get the item’s canonical ID.
2.Find the valid version:It then calls getValidVersion , passing the item’s ID and the target
timestamp ("1999-01-01T12:00:00Z"), to get the unique historicalVersionobject.
3.Retrieve the text:Finally, it calls getTextForVersion , passing the retrieved version’s ID and
the desired language (e.g.,"pt-BR"), to get the final, correctTextUnit.
20

This simple chain demonstrates how a natural language query is translated into a fully deterministic and
auditable retrieval plan once the initial probabilistic grounding is complete.
Example 2: Thematic AnalysisA broader, exploratory question like "Summarize the evolution of all
constitutional provisions related to ’Digital Security’ since 2000" would require a thematic analysis plan that
leverages the API’s powerful server-side aggregation:
1.First, the agent discovers the relevant conceptual node by calling searchThemes(semantic_
query="Digital Security").
2.Then, instead of fetching all items and their individual histories ( which would require O(N) calls where
N is the number of articles in the chapter—potentially hundreds of API calls for a single constitutional
chapter), the agent makes a single, powerful call to summarizeImpact , passing the theme’s id and
the specified time interval.
The server performs the complex aggregation, returning a concise ImpactReport . This report contains
the aggregate statistics and the identifiers of all relevant changes. If a more detailed narrative is required, the
agent can then make targeted calls to getBatchActions andgetBatchItems to hydrate the specific
details needed for the final summary. This demonstrates how the API transforms a potentially massive
client-side processing task into a single, efficient query.
Example 3: Robustness and Multilingual FallbackThe API’s atomic nature enables agents to build
robust and user-friendly behaviors, such as multilingual fallback. An agent serving a user who prefers
Portuguese ("pt-BR") can be programmed to handle cases where a specific text is not available in that
language. Instead of simply failing, the agent can implement a graceful fallback strategy:
1.The agent first attempts to retrieve the text in the user’s preferred language using getTextFor
Version(version_id, "pt-BR").
2.If this call fails (e.g., by returning a 404 error), the agent does not give up. It makes a second call to re-
trieve the text in a default fallback language, such as English: getTextForVersion(version_id,
"en").
3.When presenting the result, the agent can now inform the user about the substitution: "The requested
text was not available in Portuguese. Displaying the English version instead."
This pattern, enabled by the API’s clear separation of concerns, demonstrates how composable primitives
can be used to build sophisticated, resilient, and transparent agentic behaviors that go beyond simple query-
response loops.
3.6 Formal OpenAPI Specification
The complete, machine-readable specification of this API is provided as an OpenAPI 3.0 YAML document,
available in the project repository1. The OpenAPI specification includes: Complete request/response schemas
for all actions; Example requests and responses; Detailed error response specifications; Authentication
requirements; Rate limiting policies.
Developers can use this specification to generate client SDKs, test suites, and interactive documentation
using standard OpenAPI tooling.
4 Use Cases and Application
To demonstrate the practical utility and expressive power of the Canonical Action API, this section presents
three representative use cases. These scenarios are designed to highlight how an agentic framework can
1https://github.com/hmartim/sat-graph-api
21

leverage the API to answer complex legal questions that are intractable for standard RAG systems. Each use
case follows a pattern: a user’s query is presented, the challenge for conventional systems is outlined, and an
executable plan composed of our API actions is detailed.
4.1 Use Case 1: Point-in-Time Comparison and Causal Pinpointing
User Query: "What were the exact textual differences in Article 6 of the Brazilian Constitution
before and after the amendment that introduced the right to ’housing’ (direito à moradia)?"
ChallengeA standard RAG system would likely retrieve multiple, conflicting versions of Article 6 from
its index, unable to deterministically isolate the versions immediately preceding and succeeding a specific
conceptual change. It lacks the causal understanding to link the introduction of a word to a specific legislative
event.
Agent Execution PlanThe agent’s plan is to find the specific Action that introduced the change, which
inherently contains the identifiers for the "before" and "after" states.
1.Identify the Item:The agent first grounds the query by resolving the textual reference to its canonical
identifier.
candidates =resolveItemReference(reference_text="Article 6 of the
→Constitution")
target_item_id =candidates[0].id //Agent proceeds with the top
→candidate
2.Retrieve the Full Historical Lineage:The agent fetches the complete, time-ordered history of all
legislative events that affected the item.
history =getItemHistory(item_id=target_item_id) //Returns a list[
→Action]
3.Pinpoint the Pivotal Action (Agent Logic):The agent must now find the first Action in the
history whose resulting text contains the keyword "moradia". To do this efficiently, it first collects
allproduces_version_id s from the ‘history‘ list. It then makes a single getBatchTexts
call to retrieve all corresponding texts at once. By iterating through these texts, it can identify the
pivotal Action responsible for the change. By iterating through the returned texts and their associated
source_node_ids, the agent can map the first occurrence of the keyword back to the pivotal Action.
4.Compare the "Before" and "After" States:The pivotal Action object itself contains the direct
references to the state it terminated and the state it created. The agent uses these to get a granular
difference report.
//pivotal_action is the Action object found in the previous step
version_id_before =pivotal_action.terminates_version_id
version_id_after =pivotal_action.produces_version_id
diff_report =compareVersions(
version_id_a=version_id_before,
version_id_b=version_id_after
)
22

Synthesized OutcomeThe agent receives a structured TextDiffReport . This verifiable data, along
with details from the pivotal_action itself (like its date andsource_version_id ), is passed to
an LLM. The model can then generate a precise, auditable, and natural-language summary: "The right to
’housing’ (moradia) was introduced into Article 6 by an action on [ date from action], which terminated the
validity of version [ IDbefore] and created version [ IDafter]. The exact change was the addition of the word
’moradia’...". This answer is precise, verifiable, and causally grounded, demonstrating a capability far beyond
standard RAG systems.
4.2 Use Case 2: Causal Lineage Tracing for Legal Audit
User Query: "A legal case from 2012 references Article 227, paragraph 4 of the Constitution.
Which specific law introduced this paragraph, and is its text still the same today?"
ChallengeThis query has two distinct requirements that are impossible for standard systems: determining
historical provenance ("which law introduced...") and confirming current validity with precision. A standard
RAG system lacks a causal model to trace origins and cannot guarantee that a retrieved text is the most
current, valid version.
Agent Execution PlanThe agent recognizes two parallelizable sub-tasks and starts by grounding the
reference to get a stable identifier.
//Initial, common step for both tasks
candidates =resolveItemReference(
reference_text="Article 227, paragraph 4 of the Constitution"
)
target_item_id =candidates[0].id //Agent proceeds with the top candidate
The agent can now execute the two tasks in parallel.
Task A: Determine Provenance1. Retrieve the Past Version:The agent gets the specific Version
that was valid on a given date in 2012.
//Timestamp is in ISO 8601 UTC format for consistency
version_2012 =getValidVersion(
item_id=target_item_id, timestamp="2012-01-01T12:00:00Z"
)
2.Trace Causality to the Source:It then traces this specific version back to the action that created
it.
causality =traceCausality(version_id=version_2012.id)
creating_action =causality.creating_action
3.Hydrate the Source for Readability (Agent Logic):The creating_action object contains
thesource_version_id . To provide a human-readable answer, the agent "hydrates" this ID
(viagetVersion and then getItem ) and traverses its ancestors (via getAncestors ) to
find the rootWork(the amending law) and get itslabel.
Task B: Determine Current State1. Retrieve the Current Version:Using the same item ID, the agent
fetches the version valid as of "now".
23

//"NOW" is represented by the current UTC timestamp
version_current =getValidVersion(
item_id=target_item_id, timestamp="2025-10-06T10:30:00Z"
)
2.Compare Historical and Current Versions:To check if the text has changed, the agent compares
the version from 2012 with the current one.
diff_report =compareVersions(
version_id_a=version_2012.id,
version_id_b=version_current.id
)
//If the diff_report is empty, the text is unchanged.
Synthesized OutcomeThe agent’s planner combines the results. From Task A, it has the identity of the
amending law. From Task B, it knows whether the text has been altered. This structured data is passed to an
LLM, which can now generate a highly accurate and useful answer: "The version of Article 227, paragraph
4, valid in 2012, was introduced by **[Name of Amending Law from Task A]**, effective on [ date from
creating_action ]. A comparison with the current version shows that the text **has not changed**. It
reads: ’...’". This demonstrates how an agent can compose API primitives to build a complete and auditable
answer to a complex, multi-part question.
4.3 Use Case 3: Hierarchical Impact Summarization
User Query: "Provide a summary of all legislative changes, specifically creations and revocations,
within the ’National Tax System’ chapter of the Constitution between January 1, 2019, and
December 31, 2022."
ChallengeThis query is impossible for systems that are not hierarchy-aware. A flat-text system would
perform a keyword search for "tax," retrieving irrelevant articles and missing the implicit scope of the
"National Tax System" chapter. Furthermore, it cannot differentiate between different types of legislative
changes.
Agent Execution PlanThe agent leverages the API’s powerful server-side aggregation capabilities to
construct a highly specific and efficient query plan.
1.Identify the Scope:The agent first resolves the reference to the chapter to get its canonical id.
candidates =resolveItemReference(
reference_text="National Tax System chapter of the Constitution"
)
scope_id =candidates[0].id
2.Request a Filtered Impact Summary:The agent makes a single call to the aggregate analysis action.
It constructs a request object and sends it as the body of a POST request to the summarizeImpact
endpoint.
//The agent constructs a request body for a POST call
request_body ={
"item_ids": [scope_id],
"time_interval": [
24

"2019-01-01T00:00:00Z",
"2022-12-31T23:59:59Z"
],
"action_types": ["Creation", "Revocation"]
}
report =summarizeImpact(body=request_body)
3.Hydrate Details for Reporting (Agent Logic):The returned ImpactReport contains lightweight
lists of identifiers. To build a rich, human-readable narrative, the agent "hydrates" these IDs into full
objects using efficient batch calls.
//The response ‘report‘ contains direct lists of IDs
action_ids =report.actions
affected_item_ids =report.affected_items
//Get full data for the legislative events
full_actions =getBatchActions(ids=action_ids)
//Get full data for the affected constitutional articles
full_items =getBatchItems(ids=affected_item_ids)
//Additional hydration can be done to find the names of the amending
→laws
Synthesized OutcomeThe agent now possesses a complete, structured dataset of all relevant changes.
This verifiable data is passed to an LLM for synthesis. The resulting summary is not a guess, but a factual
report grounded in the graph: "Between 2019 and 2022, the ’National Tax System’ chapter saw the following
changes matching your criteria: Article 156-A was created by Constitutional Amendment No. 110/2019, and
**Article 149-A** was revoked by Constitutional Amendment No. 109/2021." This demonstrates a complete,
efficient, and auditable workflow, from high-level aggregation to detailed, human-readable reporting.
4.4 Use Case 4: Distinguishing Structural and Normative Predecessors
User Query: "What was the previous rule that governed this matter?"
ChallengeThis seemingly simple question hides a deep legal ambiguity. "Previous rule" can have two
distinct meanings: (1) thestructural predecessor(the immediately preceding version of the same article) or
(2) thenormative predecessor(an entirely different article, possibly in another law, that was revoked and
replaced). Standard systems can handle neither, as they lack both versioning and a semantic understanding of
legislative succession.
Agent Execution PlanOur API empowers an agent to disambiguate and solve for both interpretations.
After establishing the context (getting the ‘target_item_id‘ for "this matter"), it can pursue both paths.
For the Structural Predecessor (Deterministic Causal Path): The agent’s plan is to find the direct causal
antecedent of the current version by tracing the event that created it.
1. It retrieves the current version of the item:
current_version = getValidVersion(item_id=target_item_id,
timestamp="2025-10-06T10:30:00Z", policy="SnapshotLast")
25

2. It then traces this version’s immediate cause by callingtraceCausality:
causality = traceCausality(version_id=current_version.id)
3.The returned CausalityTrace object contains the Action that created the current version.
ThisAction also holds the ID of the version that it terminated. This is the structural predecessor:
predecessor_version_id = causality.creating_action.
terminates_version_id
This plan is fully deterministic and provides a precise, auditable answer by directly traversing the
causal graph primitives.
For the Normative Predecessor (Heuristic Path): The agent executes a more complex, discovery-oriented
plan to find a semantically similar but structurally distinct predecessor.
1. It first gathers information about the current version: its text, and the start of its validity.
current_version =getValidVersion(
item_id=target_item_id, timestamp="2025-10-06T10:30:00Z", policy="
→SnapshotLast"
)
change_date =current_version.validity_interval[0] //Start of the
→interval
current_text =getTextForVersion(version_id=current_version.id,
→language="pt-BR").content
2.It then uses this information to search for semantically similartextsthat were valid immediately
beforethe change occurred.
//Agent constructs a request body for a POST call
//It calculates a timestamp immediately prior to the change_date
previous_timestamp =calculate_prior_time(change_date) //e.g., change_
→date - 1 second
request_body ={
"semantic_query": current_text,
"timestamp": previous_timestamp
}
candidates =searchTextUnits(body=request_body)
3.The agent can then analyze the top candidates, filtering out the structural predecessor (which
might also appear) and presenting the most likely normative predecessor(s) to the user.
This second plan, while heuristic, is powerfully grounded in the verifiable temporal snapshots provided
by the API, making its discovery process far more reliable than a blind search.
Synthesized OutcomeThe agent can now provide a sophisticated, multi-faceted answer that addresses the
user’s ambiguous query head-on: "The query ’previous rule’ can have two meanings. The direct previous
version of **Article 10** (itsstructural predecessor) had the following text: ’...’. However, this new version
was created by Law XYZ, which also revoked **Article 150 of a different law** that previously governed this
matter (its likelynormative predecessor). Would you like to see the text of Article 150?". This demonstrates
the framework’s ability to support the nuanced reasoning required for deep legal analysis.
26

5 Conclusion
In this paper, we addressed a critical challenge in the application of AI to the legal domain: the need for a
trustworthy and auditable method for interacting with complex, versioned knowledge. While our previous
work, the SAT-Graph RAG, established a robust model for representing the structural and temporal reality
of legal norms, it left open the question of how to query this representation without sacrificing its inherent
guarantees. We have demonstrated that the solution is not a more complex, monolithic query engine, but
rather a more disciplined and transparent interaction layer.
Our primary contribution is the formal specification of theSAT-Graph API. The core of this API is a
library ofcanonical actions—low-level, composable, and deterministic primitives that enforce auditability
by design. By providing this API as the primary, formally-specified interface to the graph, we create a
powerful abstraction layer. This ensures that agentic frameworks can operate on the knowledge base safely
and predictably, without needing to understand the complexities of the underlying data model, making the
retrieval process as rigorous as the knowledge base itself. We have shown how this architecture enables
a powerful synergy: advanced agentic frameworks can provide the flexibility to interpret complex natural
language queries, while our API provides the unyielding determinism required to execute the resulting
retrieval plans safely. The agent’s plan becomes a complete, human-readable audit trail, transforming a
potentially opaque reasoning process into a series of explicit, verifiable steps. This shift from probabilistic
retrieval to deterministic, planned execution is a direct architectural response to the challenge of Explainable
AI (XAI), making it possible to build trustworthy AI systems for regulated domains.
5.1 Implications
The implications of this two-layer architecture are significant, marking a paradigm shift away from end-to-end,
black-box systems and toward a more modular and trustworthy approach.
Decoupling Knowledge from ReasoningThe most crucial implication is the decoupling of the knowledge
representation from the reasoning logic. The SAT-Graph serves as the stable, verifiable "source of truth,"
which can be curated and maintained independently. The reasoning layer—the agentic frameworks that plan
and execute queries—can evolve and improve at a rapid pace, incorporating newer and more powerful LLMs.
Generalizability Beyond the Legal DomainFurthermore, our deliberate use of a generic Item data
model—representing not just a document (‘Work‘) and its components (‘Work Component‘), but also abstract
Concept s and Term s—means that the SAT-Graph and its accompanying API are not limited to the legal
domain. This architectural pattern is directly applicable to other fields that rely on versioned, structured,
and evolving documents, such as technical standards (ISO, IETF), regulatory compliance documents (FDA,
financial regulations), and complex engineering manuals. In essence, any domain wherewhat wasis as
important aswhat iscan benefit from this architectural pattern.
Architectural BoundariesFinally, it is critical to distinguish between the responsibilities of the SAT-Graph
API and those of the higher-level agentic framework that consumes it. Our API is intentionally designed as
a set of verifiable primitives for retrieving structural, temporal, and causal facts, answering "what is?" and
"what was?". Complex legal reasoning tasks, such as conflict detection or transitive dependency analysis,
are the responsibility of the reasoning agent. This deliberate separation of concerns is a core strength of
our architecture: the API provides the stable and auditable foundation of truth, while the agent provides the
flexible, evolving layer of strategic reasoning.
27

5.2 Limitations
While this work provides a formal specification and demonstrates the conceptual power of the SAT-Graph
API through use cases, several limitations should be acknowledged:
Implementation Complexity: Building a production-grade implementation requires significant engineer-
ing effort, particularly for the server-side aggregation and temporal indexing capabilities.
Ontology Dependency: The API’s effectiveness depends on the quality and completeness of the underlying
SAT-Graph. Incomplete or incorrectly modeled events will propagate to retrieval results.
Agent Planning Capability: The architecture assumes the availability of sophisticated planning agents.
Current LLMs vary widely in their ability to decompose complex queries into optimal action sequences.
Evaluation Gap: This paper focuses on specification and design. Comprehensive empirical evaluation
against legal AI benchmarks remains future work.
5.3 Future Directions
This work lays a foundational layer, and in doing so, opens several avenues for future research.
First, the API specified in this paper focuses on the "verifiable" graph of hierarchical, temporal, and
causal links. As highlighted by use cases like impact analysis, a critical next step is the formalization of
a parallel "semantic overlay" graph. The conceptual groundwork for this involves introducing a first-class
Relation entity, capable of modeling a rich vocabulary of typed, directed, and potentially time-bound links
between Item s (e.g., ‘cites‘, ‘succeeds‘, ‘applies_to‘). This would be exposed through a new set of ‘Graph
Traversal Actions‘ in the API, enabling the tracing of transitive impacts and other complex relational queries.
The practical design and formal OpenAPI specification for the Relation entity and its associated actions
are being actively developed as part of an open-source project, which serves as the reference implementation
for the concepts discussed in this paper.2
Second, the sophistication of the planner agent is a key area for improvement. Future work could explore
fine-tuning LLMs specifically for the task of translating complex questions into optimal execution DAGs using
our API. This could involve creating a synthetic dataset of questions and their corresponding gold-standard
action plans for various structured document domains.
Finally, this framework can be extended to handle cross-document analysis across an entire corpus.
This would involve connecting multiple SAT-Graphs and introducing actions that can trace references and
dependencies not just within a single document, but between different standards, regulations, and their
historical precedents, moving closer to a comprehensive model of a complete knowledge ecosystem.
References
[1]Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V , Goyal N, et al. Retrieval-augmented generation for
knowledge-intensive nlp tasks. In: Advances in Neural Information Processing Systems. vol. 33; 2020.
p. 9459-74.
[2]Rahman SMW, Kim S, Choi H, Bhatti DS, Lee HN. Legal Query RAG (LQ-RAG). IEEE Access.
2025;PP(99):1-1.
2The SAT-Graph API specification project is publicly available at: https://github.com/hmartim/sat-graph-api
28

[3]Ajay Mukund S, Easwarakumar KS. Optimizing Legal Text Summarization Through Dynamic Retrieval-
Augmented Generation and Domain-Specific Adaptation. Symmetry. 2025;17(5):633.
[4]De Martim H. An Ontology-Driven Graph RAG for Legal Norms: A Structural, Temporal, and
Deterministic Approach; 2025. [preprint]. Accessed on: August 23, 2025. Available from: https:
//arxiv.org/abs/2505.00039. arXiv preprint arXiv:2505.00039.
[5]Edge D, Trinh H, Cheng N, Bradley J, Chao A, Mody A, et al.. From local to global: A graph RAG
approach to query-focused summarization; 2024. Accessed on: August 23, 2025. arXiv preprint
arXiv:2404.16130.
[6]Tjoa E, Guan C. A survey on explainable artificial intelligence (XAI): Toward medical XAI. IEEE
Transactions on Neural Networks and Learning Systems. 2020;32(11):4793-813.
[7]Sergot MJ, Sadri F, Kowalski RA, Kriwaczek F, Hammond P, Cory H. The logic programming
representation of the British Nationality Act. In: Proceedings of the 3rd international conference on
Logic programming. Springer; 1986. p. 430-53.
[8]Chalkidis I, Kamallakis M, Androutsopoulos I. Legal-bert: The muppets straight out of law school. In:
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics; 2020. p.
2898-904.
[9]De Oliveira Lima JA, Palmirani M, Vitali F. Moving in the time: An ontology for identifying legal
resources. In: Legal Knowledge and Information Systems. Berlin: Springer Berlin Heidelberg; 2008. p.
15-24.
[10] De Martim H. Modeling the Diachronic Evolution of Legal Norms: An LRMoo-Based, Component-
Level, Event-Centric Approach to Legal Knowledge Graphs; 2025. [preprint]. Accessed on: August 23,
2025. Available from:https://arxiv.org/abs/2506.07853.
[11] Hegde H, Vendetti J, Goutte-Gattat D, Caufield JH, Graybeal JB, Harris NL, et al. A change language
for ontologies and knowledge graphs. Database. 2025;2025:baae133.
[12] Pipitone N, Alami GH. LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the
Legal Domain; 2024.
[13] Gao Y , Xiong Y , Gao X, Jia K, Pan J, Bi Y , et al.. Retrieval-Augmented Generation for Large Language
Models: A Survey; 2023.
[14] Zhao X, et al. AGENTiGraph: A Multi-Agent Knowledge Graph Framework for Interactive, Domain-
Specific LLM Chatbots. In: Proceedings of the 34th ACM International Conference on Information and
Knowledge Management (CIKM 2025); 2025. Forthcoming.
[15] Yao S, Zhao J, Yu D, Du N, Shafran I, Narasimhan KR, et al.. ReAct: Synergizing Reasoning and
Acting in Language Models; 2022.
[16] Zheng L, Guha N, Arifov JD, Chen ST, Siyavash SM, Manning CD, et al.. A Reasoning-Focused Legal
Retrieval Benchmark; 2025. Forthcoming. Proceedings of the 4th ACM Symposium on Computer
Science and Law (CS&Law ’25).
[17] Martim HD. Legal Knowledge Graph Foundations, Part I: URI-Addressable Abstract Works (LRMoo
F1→schema.org); 2025. [preprint]. Accessed on: August 23, 2025. Available from: https:
//arxiv.org/abs/2508.00827.
29