# FashionKG-RAG: Knowledge Graph-Enhanced Retrieval-Augmented Generation for Fashion Question Answering

**Authors**: Yujuan Ding, Linyin Luo, Shijie Wang, Xu Yuan, Yunshan Ma, Yi Bin, Wenqi Fan, Qing Li

**Published**: 2026-08-24 01:01:06

**PDF URL**: [https://arxiv.org/pdf/2608.22688v1](https://arxiv.org/pdf/2608.22688v1)

## Abstract
Fashion is a knowledge-intensive domain in which effective decision-making depends on integrating multiple types of knowledge. Although Large Language Models (LLMs) have transformed many areas, their application in fashion remains limited by hallucinations and weak domain specialization. Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) offers a promising way to add structured knowledge to LLMs. However, existing fashion KGs are typically restricted to product-level attributes or item relations, and fail to capture the broader fashion ecosystem. To bridge these gaps, we propose \textbf{FashionEcoKG}, a comprehensive, domain-wide knowledge graph built with expert-level precision and professionalism. It is constructed through a three-stage agentic pipeline that extracts high-fidelity knowledge cores from authoritative textbooks and strengthens structural connectivity through cross-domain augmentation and generative expansion. To leverage this resource, we further develop \textbf{PG-RAG} (Pruning-Grounding RAG), a training-free framework designed to handle the conceptual density and linguistic noise of fashion queries. Specifically, we introduce a Dual-Granularity Path Re-Ranking (DGPR) module of two stages. The Pruning-based Semantic Ranking (PSR) module distills each query into a skeleton form to improve retrieval recall, while the Grounding-based Agentic Ranking (GAR) performs point-wise scrutiny of candidate paths against the original full query to ensure global relevance. Experiments on a curated fashion QA dataset show that PG-RAG effectively leverages FashionEcoKG to improve retrieval and answer accuracy, outperforming both non-RAG and existing KG-RAG baselines.

## Full Text


<!-- PDF content starts -->

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 1
FashionKG-RAG: Knowledge Graph-Enhanced Retrieval-Augmented
Generation for Fashion Question Answering
Yujuan Ding, Linyin Luo, Shijie Wang, Xu Yuan, Yunshan Ma, Yi Bin, Wenqi Fan, Qing Li,Fellow, IEEE
Abstract—Fashion is a knowledge-intensive domain in which
effective decision-making depends on integrating multiple types
of knowledge. Although Large Language Models (LLMs) have
transformed many areas, their application in fashion remains
limited by hallucinations and weak domain specialization. Knowl-
edge Graph (KG)-based Retrieval-Augmented Generation (RAG)
offers a promising way to add structured knowledge to LLMs.
However, existing fashion KGs are typically restricted to product-
level attributes or item relations, and fail to capture the broader
fashion ecosystem. To bridge these gaps, we propose Fash-
ionEcoKG, a comprehensive, domain-wide knowledge graph built
with expert-level precision and professionalism. It is constructed
through a three-stage agentic pipeline that extracts high-fidelity
knowledge cores from authoritative textbooks and strengthens
structural connectivity through cross-domain augmentation and
generative expansion. To leverage this resource, we further
develop PG-RAG (Pruning-Grounding RAG), a training-free
framework designed to handle the conceptual density and lin-
guistic noise of fashion queries. Specifically, we introduce a Dual-
Granularity Path Re-Ranking (DGPR) module of two stages.
The Pruning-based Semantic Ranking (PSR) module distills each
query into a skeleton form to improve retrieval recall, while the
Grounding-based Agentic Ranking (GAR) performs point-wise
scrutiny of candidate paths against the original full query to
ensure global relevance. Experiments on a curated fashion QA
dataset show that PG-RAG effectively leverages FashionEcoKG
to improve retrieval and answer accuracy, outperforming both
non-RAG and existing KG-RAG baselines.
Index Terms—Retrieval-augmented generation, Knowledge
graph for RAG, Fashion question answering
I. INTRODUCTION
Fashion is a cornerstone of the global economy and a critical
application domain for Artificial Intelligence (AI) [1], [2].
While often perceived as a field driven by subjective aesthet-
ics, fashion is fundamentally knowledge-intensive: successful
decision-making relies on accumulated expertise spanning
design, production, marketing, and communication1. Current
computational fashion research primarily focuses on visual
perception, generation, and outfit-related tasks such as item
retrieval [3] and recommendation [4]. However, the more
fundamental and knowledge-intensive Question Answering
(QA) task remains significantly underexplored [5], [6].
Even though LLMs have demonstrated extraordinary capac-
ity [7], [8] in general-domain tasks, they still face hallucina-
tion problems and struggle with specialized queries due to
Yujuan Ding, Shijie Wang, Xu Yuan and Qing Li are with Department
of Computing, Hong Kong Polytechnic University, Linyin Luo is with
School of Computer Science and Engineering, Sun Yat-sen University; Wenqi
Fan is with Deprtment of Computing and Department of Management and
Marketing, Hong Kong Polytechnic University; Yunshan Ma is with School
of Computing and Information Systems, Singapore Management University;
Yi Bin is with School of Computer Science and Technology, Tongji University.
1https://www.mckinsey.com/industries/retail/our-insights/state-of-fashion#/
Fig. 1: Illustration of knowledge-intensive fashion QA task
addressed by various solutions. Off-the-shelf LLMs lack the
necessary fashion knowledge for accurate responses. Even
with well-structured knowledge graphs, naive KG-RAG meth-
ods struggle to precisely target relevant information due to
the interference from query noise. In contrast, our PG-RAG
facilitates effective knowledge retrieval by pruning distractors
and extracting the most relevant knowledge path(s).
insufficient domain knowledge. As the example illustrated in
Figure 1, answering fashion questions with high conceptual
density is challenging for general-purpose LLMs. Although
Retrieval-Augmented Generation (RAG) [9], [10] incorpo-
rates external references, standard flat text-based methods fail
to leverage underlying structural knowledge. By neglecting
relations between concepts separated across multiple pages,
these retrievers provide fragmented information lacking long-
range dependencies, which is insufficient for complex or deep
reasoning [11].
To address these limitations, Knowledge Graph (KG) inte-
gration offers a promising solution by providing structured do-
main intelligence without costly fine-tuning [12], with success-
ful applications in other knowledge-intensive domains [13]–
[16]. However, general-purpose KGs [17] often lack the spe-
cialized details required for advanced fashion reasoning. While
fashion-specific KGs have been developed to fill this gap [18],
they remain constrained by narrow focuses: product-level
attributes for e-commerce [19]–[21], basic relational patterns
between items [3], [4], or transient social media trends [22],
[23]. Although these product-focused KGs may be effective
and useful to enhance tasks such as recommendation, they are
too narrow to capture the broaderFashion Ecosystem, which
encompasses critical dimensions beyond fashion products,
including business strategy, cultural history, and production
science. Furthermore, because these resources are typically
built on large-scale observational data rather than authoritative
expert knowledge, they lack the professional rigor necessary
arXiv:2608.22688v1  [cs.IR]  24 Aug 2026

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 2
for deep, knowledge-intensive analysis in the fashion domain.
To close these gaps, we proposeFashionEcoKG, a compre-
hensive domain-wide knowledge graph. Unlike previous works
that rely on unstructured web data, we utilize LLM agents
to anchor our construction process in authoritative fashion
textbooks, ensuring the fidelity and professional quality of
the resulting knowledge. While LLMs show potential for
automated KG construction [15], [24]–[27], building a high-
quality expert-level graph from such dense academic sources
presents unique challenges. Specifically, restricting our source
to canonical references initially yields an inherently narrow
core with limited situational coverage. Furthermore, because
textbooks often present concepts through high-level exemplifi-
cation without explicitly mapping every relational dependency,
the initial extraction often results in fragmented or isolated
entities. To achieve both domain expertise and structural con-
nectivity, we propose a novel three-stage hierarchical pipeline
based on LLM agents that evolves a sparse textbook core into
a larger and more comprehensive network. First, LLM agents
extract concise entities and triples from canonical references
to establish a high-quality foundation. Next, we augment this
core with general-domain KGs to bridge isolated entities and
ensure structural connectivity. Finally, we expand the graph’s
coverage and further enhance its relational structure using
validated, LLM-generated knowledge.
As illustrated in Figure 1, this work focuses on knowledge-
intensive fashion QA, where queries typically exhibit high
conceptual density and frequently span diverse sub-domains.
This multifaceted nature inevitably introduces semantic noise,
as mentions or concepts that appear relevant may actually
serve as distractors, complicating both the retrieval and gen-
eration phases. This distraction may be more amplified in a
KG-RAG pipeline than in a pure LLM due to error accu-
mulation. To address these challenges, we proposePG-RAG
(Pruning-Grounding RAG), a KG-enhanced framework utiliz-
ing a multi-stage path retrieval and re-ranking process. Unlike
conventional entity-based reasoning, our approach organizes
the FashionEcoKG into multi-hop paths to capture the struc-
tural topology. After an initial entity-based coarse retrieval,
we introduce Dual-Granularity Path Re-Ranking (DGPR), an
architecture that re-ranks the candidate knowledge paths to rec-
oncile retrieval breadth with precision. This dual-stage design
begins with Pruning-based Semantic Ranking (PSR), which
distills the conceptually dense fashion query into a simplified
skeleton to neutralize combinational distractors. PSR serves
as a necessary filter to isolate core intent and ensure a high-
recall semantic search without interference from linguistic
noise. This significant reduction in candidates subsequently
enables Grounding-based Agentic Ranking (GAR), where an
LLM agent performs point-wise scrutiny against the original
full query. Through thisPruning-to-Groundingtransition, PG-
RAG ensures that the final output is anchored in semantic
intent and also considers global context and details.
Our work addresses the limitations of existing fashion QA
models through the following technical contributions:
•We introduceFashionEcoKG, a comprehensive, domain-
wide Knowledge Graph anchored by authoritative textbooks.
By employing an effective three-stage construction pipeline(Extraction, Augmentation, Expansion), we evolve sparse
academic concepts into a professional network that captures
the broader fashion ecosystem knowledge beyond simple
product-centric relations.
•We proposePG-RAG(Pruning-Grounding RAG), a KG-
enhanced framework featuring Dual Granularity Path Re-
Ranking (DGPR). This multi-stage re-ranking architecture
reconciles retrieval breadth with reasoning precision by
transitioning from Pruning-based Semantic Ranking (PSR)
for noise neutralization to Grounding-based Agentic Rank-
ing (GAR) for logical and detailed verification against the
original query.
•We evaluate our method through extensive experiments on a
fashion QA dataset with a curated set of multi-type, multi-
complexity domain-specific questions. The results show that
PG-RAG significantly outperforms traditional non-RAG,
text-based RAG, and existing KG-RAG baselines, proving
its superior ability to leverage structured domain intelligence
for better QA precision.
II. RELATEDWORK
This section reviews related studies on Knowledge Graph-
based RAG, Fashion KG, and RAG for Fashion.
A. Knowledge Graph-based RAG
RAG is widely adopted in LLMs to mitigate issues such as
outdated knowledge and hallucinations [9], [28]. By dynami-
cally integrating external information with parametric knowl-
edge, RAG has been proven effective for knowledge-intensive
tasks across diverse domains [13], [15], [29], [30]. However,
conventional RAG methods primarily rely on unstructured
knowledge representations and often overlook logical relation-
ships among knowledge elements, limiting their capacity for
complex or multi-hop reasoning [31].
To address this limitation, Knowledge Graph-based RAG
(KG-RAG) has recently emerged as a paradigm for augment-
ing LLMs with structured knowledge. Based on how graph
structures are utilized, KG-RAG approaches can be broadly
categorized into three types [31]. (1)Knowledge-based KG-
RAGtreats graphs as structured knowledge carriers to enable
efficient retrieval and multi-step reasoning. Early works focus
on reasoning over existing KGs, such as KAPING [32] and
ToG [33]. Another line of research converts unstructured
corpora into structured graphs. For example, GraphRAG [34]
and LightRAG [35] leverage LLMs to automatically construct
attributed graphs with enriched node and edge semantics,
while StructRAG [36] explores richer structures by introducing
five task-specific graph types. (2)Index-based KG-RAGuses
graphs as indexing structures to organize and retrieve raw text
chunks for LLM inference. GNN-ret [37] constructs passage-
level graphs based on structural and keyword similarities
to support general-purpose retrieval. KGP [38] extends this
approach by building question-aware indexing graphs that
capture lexical, semantic, and entity-level relationships. (3)
Hybrid KG-RAGutilizes graph structures both as carriers of
knowledge and as indexing tools [39], [40]. A representative

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 3
Fig. 2: FashionEcoKG construction pipeline. The initialization
stage extracts a concise and professional knowledge core from
authoritative textbooks, which then evolves into a network
with better connectivity and coverage through two phases of
augmentation.
approach is LinearRAG [40], which builds a lightweight hier-
archical graph over entities, sentences, and passages, enabling
passage retrieval through local semantic bridging and global
importance aggregation.
B. Fashion KG and RAG for Fashion
Knowledge graphs in the fashion domain have tradition-
ally focused on product-centric tasks. Early work, such as
FashionPedia [18], focuses on fine-grained visual understand-
ing of apparel, enabling product classification and attribute
detection. Similarly, [4] constructs a fashion KG based on
item- and outfit-level attributes for outfit recommendation.
Recent studies incorporate multimodal information into fash-
ion KGs. FashionKlip [3] propose a multimodal fashion KG
in which entities represent physical product attributes and
edges encode semantic similarity, with a focus on e-commerce
applications. Another line of work builds fashion KGs from
social media data [22], [23] to capture user behavior pat-
terns and fashion trends. While effective for recommendation
and retrieval [19]–[21], [41], these methods are restricted to
product- or outfit-level concepts. They largely neglect broader
fashion knowledge, including cultural, historical, business,
and design-related aspects, limiting their utility for complex
domain reasoning.
0 500 1000
Other_AbstractConcept
BrandStrategy
FashionStyle
HistoricalPeriod
Combined_AbstractConc…AbstractConcept
0 500 1000
ClothingItem
Other_ApparelItem
MarketRoleProduct
SignatureItem
AccessoryApparelItem
0 200 400 600 800
DesignMethod
Produc�onMethod
FabricTreatment
Illustra�onTechnique
Other_ProcessOrMethodProcessOrMethod0 200 400
GarmentPart
Color
DesignPrinciple
Pa�ern
StructuralElement
Silhoue�e
Other_DesignElementDesignElement
0 500 1000
Fabric
Hardware
Combined_Material
Other_MaterialMaterial
0 200 400
Brand
Combined_Organiza�on
Other_Organiza�onOrganiza�on
0 500
HistoricalEvent
Combined_Others
Other_OthersOthers
0 200 400
Designer
Illustrator
Other_PersonPerson
0 1000 2000
AbstractConcept
ApparelItem
ProcessOrMethod
DesignElement
Material
Organiza�on
Others
PersonEn�ty Type Distribu�onFig. 3: Entity type distribution of FashionEcoKG. The top-left
panel illustrates the primary entity types, while the remaining
subplots provide a granular breakdown of secondary entity
types for each respective primary type.
Motivated by advances in KG-RAG, recent work explores
integrating fashion KGs with LLMs. For instance, Fashion-
RAG [42] and FITMag [43] combine multimodal LLMs with
graph-structured knowledge and social signals for fashion
content generation. VestiMe [44] and VisioRAG [45] lever-
age multimodal retrieval to support conversational exploration
and recommendation. Despite these developments, the funda-
mental knowledge-intensive question answering task remains
significantly underexplored in the fashion domain. Existing
models [5], [46] can identify a garment or garment-relevant
concepts but often fail to abstract concepts, such as a de-
signer’s underlying inspiration or the strategic rationale of
a style. Such limitations hinder fashion AI from achieving
expert-level decision-making and from addressing complex
queries that require long-range reasoning across the broader
fashion ecosystem.
III. FASHIONECOKG CONSTRUCTION
Our construction methodology follows a divergent yet
seeded approach, designed to overcome the inherent trade-
offs between knowledge authority and graph connectivity.
We propose a Knowledge Initialization and Augmentation
pipeline that transitions from a condensed, textbook-based core
to a deep and large-scale knowledge network, as shown in
Figure 2.
A. FashionEcoKG Initialization
The initialization of FashionEcoKG systematically extracts
domain entitiesEand relationsRfrom a corpus of authori-
tative fashion textbooksT. To preserve the authority of the

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 4
TABLE I: Prompts for FashionEcoKG Initialization.
NER Prompt
System prompt:I will provide a section about the ontology design of the
fashion knowledge graph and a chapter of content from a fashion-related
book. Your task is to perform the NER task based on this content.
Requirements:
1) List all results as completely as possible, in Chinese.
2) Do not include categories that are not in the ontology content.
3) Entities should be specific instances rather than general categories
and must not duplicate the existing entries in the ontology.
4) Output as a JSON list in the format “name”: “entity name”, “label”:
“entity category”, where label includes all hierarchical categories
separated by “;”. Output only the result directly without explanation.
User prompt:
1) Ontology Design: data[“ontology”]
2) Text Content: data[“text”]
3) Begin Output:
RD Prompt
System prompt:I will provide content related to the fashion domain
knowledge graph, including a chapter from a fashion-related book, the
corresponding NER results, and the designed relations. Your task is to
perform the relation extraction task based on this content. You should
identify as many entity relations as possible, where entities are from the
provided NER results and relations are from the provided relation design.
The output format should be triples.
User prompt:
1) NER Results: data[“NER”]
2) Relation Design: data[“ontology”]
3) Book Content: data[“book content”]
4) Begin Output:
source material, which draws on interviews with students
at leading Chinese fashion universities, the original meta-
data is maintained in Chinese; all triples are subsequently
translated into English to support broader generalization.
Although translation may introduce subtle nuances, modern
LLMs substantially reduce cross-lingual barriers, rendering
this concern secondary to our primary objectives. Given the
advanced capability of LLMs in perceiving textual semantics
and resolving well-defined extraction tasks from unstructured
text [15], [27], we adopt an LLM agent-based approach for
both Named Entity Recognition (NER) and Relation Detection
(RD). A decoupled, sequential pipeline processes the raw
textbook content through these two phases, and both are
executed chapter-by-chapter to preserve local semantic focus.
For effective NER, the agent is first instructed with an
expert-crafted ontologyof the fashion ecosystem, which
defines eight primary categories and multiple secondary sub-
categories, as shown in Figure 3. This ontology was co-
developed by two fashion experts, audited by a third, and
iteratively refined according to the error patterns surfaced
during preliminary NER/RD testing. To ensure extraction
quality, we benchmarked multiple LLM backbones against
manual annotations produced by a fashion-major researcher
for a representative chapter, which served as the ground-truth
baseline. The model achieving the highest alignment with both
the hierarchical ontology and the human-annotated results was
then selected as the primary agent for ontologically grounded
NER across the textbook corpus.
For the RD phase, the agent is supplied with the raw chapterTABLE II: Prompts for FashionEcoKG Expansion.
Fetch articles prompt
System prompt:I will provide a specific fashion entity keyword and its associated
category label (context). Your task is to convert this keyword into a highly effective
search query to retrieve professional content.
Requirements:
1) The goal is to find high-quality articles, fashion magazines, or academic
papers.
2) Use the “Label” only as context to understand the domain (e.g., distinguishing
a brand from a material), but do NOT clutter the search query with generic
domain words unless necessary for disambiguation.
3) The query must focus primarily on the “Keyword” itself.
4) Output ONLY the raw query text string. Do not include quotes, explanations,
or any other text.
5) The query should be in English.
User prompt:
1) Keyword: data[“keyword”]
2) Label: data[“label”]
3) Begin Output:
Generate triples prompt
System prompt:I will provide a keyword and a section of text content retrieved
from fashion-related sources. Your task is to extract structured Knowledge Graph
triples based on this content.
Requirements:
1) Extract triples in the format (head, relation, tail).
2) You must ONLY use relations from the following allowed list: [designs,
creativeDirectorOf, isCommonlyMadeOf, hasSilhouetteCommonly, hasStyle,
influencedBy, originatedIn, usesTechnique, followsPrinciple, hasPart, isAsso-
ciatedWith, isCharacterizedBy, isPopularAt, happenIn, pairWith, inspiredBy].
3) Adhere to strict type constraints:
- “originatedIn”: head must be Brand/Person, tail must be Location/Time.
- “hasPart”: head must be Product/Brand, tail must be Feature/Sub-brand.
- “isCommonlyMadeOf”: tail must be Material/Fabric.
4) Ensure both “head” and “tail” are specific entities found in or inferred from
the text. 5. Output as a JSON list in the format [“head”: “X”, “relation”: “Y”,
“tail”: “Z”].
5) Output only the JSON result directly without explanation. If no valid triples
are found, output an empty list [].
User prompt:
1) Keyword: data[“keyword”]
2) Text Content: data[“combined text”]
3) Begin Output:
Nodes labeling prompt
System prompt:I will provide a specific entity name that is missing a category label
in the Knowledge Graph, along with a list of valid domain labels. Your task is to
categorize the entity using ONLY the provided labels.
Requirements:
1) Select the most specific and accurate label from the “Available Labels” list
provided in the input.
2) Do NOT invent new labels. You must strictly match one of the strings from
the provided list.
3) If the entity is ambiguous, select the most relevant fashion/design-related label.
4) Output as a JSON object in the format: “name”: “entity name”, “label”:
“SelectedLabel”.
5) Return ONLY the JSON object. Do not include markdown formatting (like
“‘json), explanations, or notes.
User prompt:
1) Entity: data[“node name”]
2) Available Labels: data[“labels str”]
3) Begin Output:
text, the entity set identified by NER, and a dictionary ofpre-
defined relationsestablished by domain experts (Supplemen-
tary Materials). This closed relational vocabulary constrains
the model to detecting edges within a fixed semantic set.
Similar to NER, several LLM backbones are compared, and
an additional LLM serves as a reflection agent to further
refine the outputs. The final production model was selected
through human sampling evaluation, ensuring that the resulting
triples preserve the structural rigor required for expert-level

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 5
reasoning. Following NER and RD, the preliminary triples of
FashionEcoKG were generated and then subjected tomanual
review, a labor-intensive validation process totaling approxi-
mately 100 man-hours contributed by eight fashion-expert an-
notators, with a domain-specialized annotator (fashion major)
verifying and correcting the generated triples to guarantee data
fidelity. Specific prompts applied for NER and RD are given
in Table I.
While the initialization phase ensures that the foundation
of the FashionEcoKG is anchored in academically rigorous
and high-precision domain knowledge, the resulting graph
topology is limited bystructural sparsity. Analysis of the
initial graph indicates that the condensed nature of textbook
corpora often leads to concepts being introduced in isolation,
resulting in a fragmented architecture with restricted relational
connections. To mitigate these structural bottlenecks and fa-
cilitate more complex reasoning, we introduce two additional
augmentation phases to bridge gaps between isolated expert-
level entities as well as to enrich the graph with stronger
relational connections.
B. FashionEcoKG Augmentation
We propose a two-faceted augmentation stage for our
FashionEcoKG with two processes: Cross-domain Relational
Augmentation and Generative KG Expansion.
Cross-domain Relational Augmentation.To address the
topological fragmentation identified in the initial graph, we
implement a Cross-domain Relational Augmentation phase
utilizing a domain-restricted slice of DBpedia. Given the
extreme scale and inherent noise of the full DBpedia cor-
pus, we isolate the Fashion category subgraph2as a high-
fidelity reference set to maintain efficiency and precision. The
augmentation begins with Entity Alignment, where a coarse
matching process is executed between the entities distilled
in the previous stage and the resources within the DBpedia
fashion slice. For each successfully aligned entity, we perform
Neighborhood Subgraph Extraction to retrieve all associated
triples. To ensure structural integrity, the resulting candidate
relations are manually evaluated. Relations that are low quality
or semantically divergent from the fashion ecosystem are
removed. The remaining qualified triples are merged into
FashionEcoKG, serving as the ‘structural glue’ that converts
isolated clusters into a connected, multi-hop topology.
Generative KG ExpansionTo grow the narrow core into a
deep, comprehensive resource, we deploy an LLM-driven ex-
pansion pipeline that uses the verified entities of the grounded
FashionEcoKG base as seeds for querying open web data. For
each seed entitys, an automatic search aggregates a corpus
of unstructured documents from the open web, which then
serve as the dynamic context for an LLM agent. To keep the
generated knowledge domain-aligned, we condition the LLM
on the relation set established in previous phases.
Candidate triples undergo two-facet verification before in-
tegration. First, the agent checks whether each triple’s en-
tities map to categories of the expert-crafted ontology, a
domain filter that confines new concepts to the established
2https://dbpedia.org/page/Fashionfashion ecosystem. Second, to prevent graph inflation and
preserve topological efficiency, each candidate triple is com-
pared against the current graph state and discarded if redun-
dant. Through retrieval, regulated generation, and verification,
FashionEcoKG grows substantially in scale and relational
connectivity, enabling expert-level reasoning while preserving
authoritative precision. Specific prompts applied for Fash-
ionEcoKG expansion are given in Table II. Statistics of the
different stages of FashionEcoKG are given in Table III.
Beyond the consistent gains FashionEcoKG yields across
multiple RAG strategies, we further assess triple-level quality
via a multi-agent LLM check: Gemini 2.5 and GPT-5 indepen-
dently cross-validate the extracted triples, achieving over 94%
individual correctness and 72% joint agreement. This confirms
both the reliability of our automated extraction pipeline and
the overall quality of the curated knowledge.
Finally, FashionEcoKG supports incremental updates: newly
acquired triples are mapped to established ontological anchors
and verified using the same procedure introduced in the
augmentation process.
TABLE III: FashionEcoKG Statistics
KG-V1 KG-V2 KG-V3
# Entity 4930 5320 7299
# Triple 8041 8770 10592
IV. MODEL: PRUNING-GROUNDINGKNOWLEDGEGRAPH
RETRIEVAL-AUGMENTEDGENERATION
In this section, we introduce our Pruning-Grounding
Knowledge Graph Retrieval-Augmented Generation (PG-
RAG) method, which leverages the constructed FashionEcoKG
for knowledge-intensive fashion QA. Given a questionq,
the ultimate task is to design a functionfto predict the
answera, i.e.,a=f(q)We denote our FashionEcoKG
asG={< e h, r, et>|e h, et∈ E, r∈ R}, whereE
andRare the set of entities and relations respectively, and
< eh, r, et>denotes a triple. With FashionEcoKG, we aim
to extract effective informationIbased on the question with
an effective retriever (g):I=g(q,G), thereby helping the
generator (f) for answer prediction:a=f(q,I).
A. Coarse-grained Path Retrieval
Given a well-constructed KGG, our PG-RAG is a path-
based RAG pipeline that first encodes the KG into multi-
hop paths with maximum path length beingT, which can
be conducted offline. The whole path set can be denoted
asP={P1, . . . ,PT}, wherePidenotes the collection of
paths withi-hop connections. Each path can be indicated with
entities in it, which means every entity in the KG is associated
with a set of paths, denoted asP e. For example, a 2-hop
path can be denoted asp=e 0r0− →e 1r1− →e 2,p∈ P e0and
p∈ P e2. We treat each questionqas the original query for the
retrieval system. To support entity retrieval, we first conduct
entity extraction with an LLM agentf eefor a given query to
obtain entity mention setE q=fee(q) ={e q}, eq∈q. After
that, we can retrieve relevant knowledge paths for queryq

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6
Fig. 4: Overview of the PG-RAG Pipeline. Based on coarse-grained path retrieval, PG-RAG introduces a Dual-Granularity Re-
ranking module to address conceptual density and query distraction in fashion QA. The PSR module extracts query skeletons
for high-recall path retrieval, followed by the GAR module for point-wise candidate path scrutiny against the full query to
ensure global relevance.
through entity matching:P q={p e|e∈ E q}. For each queryq,
the entity-matching path retrieval results offer coarse-grained
relevant candidates.
B. Dual-Granularity Path Re-Ranking (DGPR)
The initial retrievalP qoften suffers from noise due to
the conceptual density and descriptive distractors present in
fashion queries, as shown in the example illustrated in Fig-
ure 4. Purely entity-based retrieval also neglects relational
information. To refine this issue, we propose Dual-granularity
Path Re-Ranking (DGPR), comprising a semantic-alignment
pruning stage and a structural-verification grounding stage.
1) Pruning-based Semantic Ranking (PSR):Fashion
queries typically involve combinational subjects where only a
subset of mentions is critical for the reasoning task. To isolate
the core intent, we utilize an LLM agentf qpas thequery
pruner, transforming the original queryqinto a compact query
skeletonq s=fqp(q). This process prunes linguistic noise and
non-essential entity mentions that might lead to semantic drift
during retrieval. The semantic similarity between the query
skeletonq sand each candidate pathp∈ P qobtained from
initial stage is further computed:
s(qs, p) =sim(h qs, hp),(1)
whereh qsandh pare semantic embeddings for them respec-
tively, which can be obtained with a pre-trained language
model. We select the top-k 1pathsP psr⊆ Pqwith the highest
similarity scores. By prioritizing the skeleton, PSR ensures a
high-recall set of path candidates while effectively neutralizing
combinational distractors.
2) Grounding-based Agentic Ranking (GAR):While PSR
identifies semantically related paths, purely embedding-based
retrieval often fails to capture the structural logic or over-
looks fine-grained constraints. To overcome this limitation,
we introduce Grounding-based Agentic Ranking (GAR). This
stage utilizes a discriminative evaluator agentf eto performpoint-wise scoring on the pruned candidate setP psrusing the
original queryq. The original query is chosen here rather than
the simplified skeleton used in PSR to ensure that no expert-
level constraints or nuanced requirements are lost during the
grounding process. While PSR prioritizes retrieval breadth by
neutralizing noise, GAR provides the necessary precision by
scrutinizing each candidate pathp i∈ Ppsragainst every detail
of the user’s intent. Unlike semantic ranking, which relies
on global vector proximity, the agent explicitly interprets the
relational transitions within a path (e.g.,e 0r0− →e 1r1− →e 2) to
determine if the knowledge chain logically satisfies the query.
For each path, the agent assigns a grounding scoreω i∈[0,1]:
ωi=fe(q, pi), p i∈ Ppsr.(2)
The final set of retrieved informationIis constructed by
selecting the top-k 2paths with the highest evaluation scores:
I={p i|top-k 2sorted byω i}.(3)
By transitioning from the broad semantic filtering of PSR
to the rigorous agentic grounding of GAR, the PG-RAG
framework ensures that the final generation is anchored in
logically verified, highly relevant knowledge.
V. EXPERIMENTS
In this section, we evaluate the effectiveness of our proposed
PG-RAG framework through extensive experiments.
A. Experimental Setups
Implementation Details.We implement the proposed method
using the open-source PyTorch [47] framework. All ex-
periments are conducted on an NVIDIA RTX 3090 GPU.
We adopt Qwen2.5-7B-Instruct [48] as the pruning model,
SBERT [49] as the semantic embedding model, and Qwen3-
Reranker-4B [50] as the GAR model. The top 10 paths
retrieved by the PSR stage are retained as candidates for

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7
TABLE IV: Fashion QA accuracy of compared methods with regard to different question types and complexities. Defi.: Defi-
nition, Infl.: Influence, Part.: Partonomy, Char.: Characteristic, Exem.: Exemplification, Insp.: Inspiration, Comp.: Comparison.
ModelQuestion Category ComplexityOverall
Defi. Infl. Part. Char. Origin Exem. Insp. Comp. Easy Medium Hard
open-source VLMs
Qwen2.5-VL-7B [48]82.32 64.10 55.14 56.27 53.94 52.73 43.82 41.38 82.01 53.75 57.76 59.86
Qwen3-VL-8B [50]93.91 71.79 64.49 65.37 66.76 57.58 53.93 31.03 93.81 65.63 59.03 69.44
InternVL3-8B [53]86.96 60.26 62.62 64.33 61.81 47.88 44.94 31.03 87.02 60.97 56.74 64.90
InternVL3.5-8B [54]92.17 66.67 58.88 63.43 65.31 56.97 55.06 37.93 92.04 63.80 57.51 67.69
Gemma-3-12B [55]88.99 62.82 65.42 66.72 58.60 58.79 50.56 44.83 89.09 63.89 58.02 67.31
MiniCPM-V-4.5-8B [56]92.17 73.08 63.55 62.84 67.35 53.94 55.06 27.59 92.04 63.35 60.05 67.96
GLM-4.1V-9B [57] 92.75 70.51 66.36 67.46 66.76 61.21 58.43 41.38 92.63 67.82 60.05 70.76
open-source LLMs
DeepSeek-R1-Qwen-7B [8]71.01 39.74 38.32 38.21 37.32 36.36 37.08 17.24 70.50 37.29 38.68 43.76
GLM-4-9B [58]75.94 62.82 42.99 50.60 58.89 47.27 34.83 27.59 75.81 51.55 49.36 55.59
Llama3.1-8B [59]85.17 61.54 52.34 49.25 56.25 47.27 48.31 34.48 84.91 50.61 53.18 57.60
Qwen2.5-7B [48] 93.60 70.51 57.01 62.99 63.12 54.55 51.69 34.48 93.79 61.90 58.02 67.04
Qwen3-8B [50] 91.30 69.23 59.81 64.93 60.64 57.58 52.81 34.48 91.15 63.16 58.02 67.25
closed-source LLMs
Gemini-2.5 Flash [61] 92.46 74.36 64.49 69.85 65.89 57.23 56.18 37.93 92.63 67.49 61.83 70.94
Grok-4 Fast [63]95.65 71.79 75.70 67.76 68.51 63.03 52.81 48.28 95.87 68.28 63.36 72.34
GPT-5 Nano [7]93.91 78.21 66.36 67.31 64.43 58.18 56.18 34.48 94.10 66.64 60.05 70.32
KG-RAG Methods
TOG Qwen2.5-7B [33]88.24 70.91 63.38 66.26 59.85 60.00 58.47 40.62 87.57 65.78 58.73 67.51
KAPING Qwen2.5-7B [32]90.67 77.98 65.49 65.42 65.90 59.29 63.87 46.88 90.27 67.94 60.08 69.56
KG-Triple-RAG Qwen2.5-7B 89.07 73.39 63.38 66.76 70.23 60.62 68.91 46.88 88.79 70.97 57.77 70.58
KG-Path-RAG Qwen2.5-7B 88.80 73.39 60.56 66.62 70.99 64.60 64.71 46.88 88.20 69.26 62.57 70.63
PG-RAG Methods
PG-RAG Qwen2.5-7B 92.53 83.49 69.01 68.36 73.28 67.26 73.95 46.88 92.04 73.23 64.88 74.17
PG-RAG Qwen3-8B 89.87 84.40 71.13 70.36 70.48 69.91 73.95 56.25 89.09 72.92 68.91 74.50
PG-RAG GLM-4.1V-9B 90.67 81.65 74.65 73.70 73.79 73.01 78.15 56.25 89.97 75.64 72.17 77.06
PG-RAG Gemini-2.5 Flash 94.13 83.33 83.10 78.50 75.83 78.32 76.47 50.00 93.81 80.47 72.88 80.74
Easy
17.95%
Medium
60.72%Hard
21.33%18.42%
4.24% 5.98%36.63%
18.42%
9.53%5.24%1.54%
00.10.20.30.4
Deﬁni�on InﬂuencePartonomyCharacteris�cOrigin
Exempliﬁca�onInspira�onComparison
(a) Ques�on Type Distribu�on (a) Ques�on Complexity Distribu�on
Fig. 5: Distribution of question categories (Left) and question
complexity (Right) in the tested fashion QA dataset.
GAR, and the top-ranked GAR result is used as the context
for question answering. The candidates for coarse-grained
path retrieval include 1-hop, 2-hop, and 3-hop paths. For
FashionEcoKG construction, we evaluate three closed-source
LLM backbones: Gemini-2.5 Pro, Claude Sonnet 4, and GPT-
5. Gemini-2.5 Pro is selected as the primary agent for both
NER and RD tasks across all stages.
Datasets.We construct a question-answering dataset using a
KG-driven multi-complexity generation pipeline, following a
methodology similar to that of concurrent work [51]. Specifi-
cally, we employ LLM agents to generate questions grounded
in knowledge paths with different hop-level complexities in
the KG. The resulting dataset contains 2,153 multiple-choice
questions (MCQs) across three complexity levels: Easy (0-
hop), Medium (1-hop), and Hard (≥2-hop). Each sample
consists of a ground-truth answer and three plausible dis-tractors. To ensure data integrity, all generated samples are
manually verified and refined to correct hallucinations and
structural errors, including low-quality distractors, disordered
options, answer leakage in the question stem, and logically
weak phrasing. The questions are further categorized into eight
types to support fine-grained analysis of model performance
across multiple dimensions. In Fig. 6 we illustrate several
QA examples with different question types and complexity
levels. To complement the synthetic benchmark and to mitigate
the concern that questions generated in reverse from a graph
may overfit graph structure or retrieval paths, we prepare a
tiny human-authored testing subset FashionQA-mini (200 QA
samples). This subset is generated independently by human
annotators without referring to FashionKG while using only
the original textbook as reference material.
Baselines.To evaluate the effectiveness of the proposed PG-
RAG model, we compare it with a comprehensive set of base-
lines, including mainstream Vision-Language Models (VLMs),
LLMs, and established KG-RAG methods. Foropen-source
VLMs, we consider Qwen 2.5VL [48], Qwen3-VL [50],
InternVL3 [53],InternVL3.5 [54] Gemma-3 [55], MiniCPM-
V-4.5 [56], and GLM-4.1V [57]. Since fashion is a vision-
intensive domain, these models are included because their mul-
timodal training may help them acquire and represent fashion
knowledge more effectively than text-only LLMs. Foropen-
source LLMs, we include DeepSeek-R1-Qwen [8], GLM-
4 [58], LLaMA 3.1 [59], Qwen 2.5 [48], and Qwen3 [50],

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 8
TABLE V: PG-RAG prompts
Query pruning prompt
System prompt:Your task is to perform simplification on the question and extract
the main question that best simplifies the original question. If the question is already
a simple question, just use the original question as a simplified question.
Format: Use Markdown headings (“# 1. Analysis” and “# 2. Final Answer”) to
structure your output clearly.
CRITICAL: Conclude your analysis by providing the single best simplified question
after “# 2. Final Answer”.
User prompt:Question: data[”question”]. Begin Output:Topic entity extraction prompt
System prompt:You are an expert in text entity recognition. Extract all nouns from
the given question. Output in JSON format, e.g.,{“entities”: [“entity1”, “entity2”]}.
Note: 1. Each line must be valid JSON. 2. If no entities are found, output “No entities
found”.
User prompt:Question: data[“question”]. Begin Output:
No RAG QA prompt
System prompt:You are an expert fashion connoisseur. Your task is to analyze the
question and context, and then strictly follow the steps below to output your final
answer:
1) Internal Knowledge and Analysis: First, reason step-by-step why each option
is correct or incorrect based on both the provided context and your internal
knowledge
2) Selection:Based on your analysis, select the single best option. If, after
thorough reasoning, you cannot confidently choose one, randomly select one
option (A, B, C, or D) as your final answer to avoid non-response.
3) Format: Use Markdown headings (1. Analysis and 2. Final Answer) to structure
your output clearly.
4) Final Answer: Conclude your analysis by providing the single best option letter
inside< Answer >and< Answer >tags.
CRITICAL:The entire output MUST strictly start with the< Answer >tag,
followed by a newline, then your detailed analysis.
User prompt:
1) Question: data[“question”]
2) Options: data[“options”]
3) Begin Output:RAG QA prompt
System prompt:You are an expert fashion connoisseur. Your task is to analyze the
question and context, and then strictly follow the steps below to output your final
answer:
1) Provided Information: The context includes several contexts retrieved by the
question. The contexts are provided in decreased similarity with the question.
2) Knowledge and Analysis: First, reason step-by-step why each option is correct
or incorrect based on **both the provided context and your internal knowledge.
3) Selection: Based on your analysis, select the single best option. If, after
thorough reasoning, you cannot confidently choose one, randomly select one
option (A, B, C, or D) as your final answer to avoid non-response.
4) Format: Use Markdown headings (1. Analysis and 2. Final Answer) to structure
your output clearly.
5) Final Answer: Conclude your analysis by providing the single best option letter
inside< Answer >and< Answer >tags.
CRITICAL:The entire output MUST strictly start with the< Answer >tag,
followed by a newline, then your detailed analysis. Remember to conclude your
analysis by providing the single best option letter inside< Answer >and<
Answer >tags.
User prompt:
1) Context: data[“context”]
2) Question: data[“question”]
3) Options: data[“options”]
4) Begin Output:
In the fashionﬁeld, what is the "Ugly
Look" classiﬁed as?
A. Historical Period  B. Fashion Style
C. Apparel Item       D. Designer
(Ques�on Type: Deﬁni�on)Easy QA sample 1
"Gender-neutral tailoring" belongs to 
which type of design element?
A. Silhoue�eB. Color
C. Material             D. Pa�ern
(Ques�on Type: Deﬁni�on)Easy QA sample 2
Which of the following colors is a 
feature of "Elegant Style"?
A. Red                 B. Blue
C. WhiteD. Black
(Ques�on Type: Characteris�c)Medium QA sample 1
The forma�on of "Street Style" was 
mainly inﬂuenced by what culture?
A. Court Culture  B. Royal Fashion
C. SubcultureD. Industrial Revolu�on
(Ques�on Type: Inﬂuence)Medium QA sample 2
Hippie style popular in the 1970s is 
characterized by the use of which fabric?
A. GabardineB. Calico
C. Velvet               D. Flannel
(Ques�on Type: Characteris�c)Hard QA sample 1
Which cra� technique popular in the 
1980s is o�en used to make Outerwear?
A. Embroidery      B. Plea�ng
C. Laser cu�ng    D. Digital prin�ng
(Ques�on Type: Exempliﬁca�on)Hard QA sample 2
Fig. 6: Fashion QA examples with different question types
and levels of complexity. The correct answer is highlighted in
yellow.
[60]. Forclosed-source LLMs, we use Gemini-2.5 Flash [61],
[62], Grok-4 Fast [63], and GPT-5 Nano [7].
We further include several training-free KG-RAG baselines.
In addition to basic Triple-RAG and Path-RAG, which rely
on triple- and path-level semantic similarity for retrieval, wecompare our method with KAPING [32] for entity-based triple
retrieval and ToG [33] for structured reasoning over graphs.
We exclude approaches that construct their own KGs from
scratch, because their performance would be confounded by
differences in KG construction quality. To ensure a fair eval-
uation of retrieval and reasoning, we compare only methods
that operate directly on our FashionEcoKG.
Evaluation Metrics.We evaluate retrieval performance using
the Pseudo Relevance Recall (PRRecall@K) [64], [65] metric.
A document is considered pseudo-relevant if it contains any
answers. PRRecall@K measures whether the retrievedK
documents include at least one pseudo-relevant document. For
QA performance, we report the accuracy.
B. Fashion QA Performance
The experimental results summarized in Table IV reveal
several key insights into the efficacy of our proposed method.
First, PG-RAG consistently achieves the best performance
across evaluation settings, outperforming both vanilla LLMs
and existing KG-RAG baselines. This advantage demonstrates
the effectiveness of DGPR in identifying high-quality reason-
ing paths that contain useful knowledge. The effectiveness of
KG path-based retrieval is further supported by the qualitative
cases will be discussed in Sec. V-F.
When examining the impact of different generators, we
observe that the backbone model’s inherent capabilities re-
main influential; GLM-4.1-9B, which is the strongest no-
RAG baseline, also achieves the best overall performance
when integrated into the PG-RAG pipeline, notably surpass-

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 9
TABLE VI: QA accuracy comparison with graph-based RAG baselines. (All methods use Qwen-2.5-7B as the QA model. Best
result per column is inbold.
ModelQuestion Category ComplexityOverall
Defi. Infl. Part. Char. Origin Exem. Insp. Comp. Easy Medium Hard
No RAG93.6070.51 57.01 62.99 63.12 54.55 51.69 34.4893.7961.90 58.02 67.04
LightRAG [35] 88.53 72.48 52.11 58.88 63.61 50.44 52.94 53.12 88.50 60.86 55.28 63.87
LinearRAG [40] 89.33 62.39 57.75 60.48 59.03 53.98 57.98 43.75 89.09 60.47 56.81 64.10
PG-RAG 92.5383.49 69.01 68.36 73.28 67.26 73.9546.88 92.0473.23 64.88 74.17
ing variants that use Qwen as the generator. Furthermore,
our comparison of various LLMs and VLMs in no-RAG
settings provides a broader understanding of how different
architectures handle specialized fashion knowledge. We find
that performance varies significantly across different versions
of the same backbone, with closed-source models generally
exhibiting stronger reasoning and domain knowledge than
open-source alternatives.
To mitigate overfitting to graph structures in synthetic
benchmarks, FashionQA-mini provides 200 human-written
questions constructed independently from textbooks without
FashionKG access. Four annotators formulate natural, chal-
lenging queries testing reasoning, domain expertise, and long-
tail concepts, serving as a robustness check for real-world
generalization beyond KG-based generation. On FashionQA-
mini, our method achieves the best QA accuracy of 85.5%
among all compared settings. It outperforms the No RAG base-
line by 13.5% and KAPING by 6.5%. Since FashionQA-mini
is manually constructed without referring to FashionEcoKG,
this improvement suggests that our method generalizes be-
yond KG-derived question patterns to realistic, human-written
fashion-domain questions.
Comparison with Graph-based RAG Baselines: We ad-
ditionally evaluate two representative graph-based RAG meth-
ods, LightRAG [35] and LinearRAG [40]. Both methods
construct graph indexes to enhance retrieval and have shown
effectiveness on relatively well-organized corpora and standard
QA benchmarks. However, they are not specifically designed
for knowledge-intensive fashion QA, where critical informa-
tion is embedded in heterogeneous, descriptive, and weakly
structured raw text.
As shown in Table VI, both methods underperform the
No-RAG base LLM in aggregate accuracy, with scores of
63.87 and 64.10 compared to 67.04. Although they bring
improvements on some question types that benefit from rela-
tional reasoning, such as Com., and for LinearRAG also Part.
and Insp., these gains are outweighed by performance drops
on Defi., Char., and Exam. This suggests that graph indexes
automatically constructed from weakly structured fashion text
can help when explicit relational reasoning is required, but
may introduce noise or retrieval mismatches for questions that
the base LLM can already answer reliably.
We attribute this degradation to the nature of fashion
knowledge. Unlike cleaner corpora where entities and rela-
tions can be readily extracted, fashion knowledge is often
latent, fine-grained, and context-dependent, appearing within
descriptive explanations rather than explicit relational struc-tures. Existing graph-based RAG methods therefore struggle
to uncover and exploit such hidden knowledge effectively.
These results further motivate our domain-specific knowledge
modeling strategy: FashionEcoKG explicitly organizes deeply
embedded fashion knowledge into a structured, ecosystem-
level representation, while our PG-RAG framework retrieves
and reasons over this structure more effectively.
C. Ablation Studies on PG-RAG
In Table VII, we analyze the impact of different modules
of PG-RAG. Specifically, we evaluate the following configura-
tions (using Qwen-2.5 / Gemini-2.5 as the question answering
model): (1)No RAG: answering without using any retrieved
contexts. (2)Ours (w/o GAR): using only the PSR retrieval
results. (3)Ours (w OSR): using DGPR retrieval results but
removing query pruning in the PSR stage, that is, using the
original query to compute semantic similarity with candidate
paths. (4)Ours (w PAR): using the pruned query instead of
the original one to compute GAR ranking scores. (5)Ours:
PG-RAG.
The experimental results lead to three main observations.
(1) The two ranking modules, PSR and GAR, both contribute
to QA performance. As shown in the second line, PSR alone
improves the overall QA performance by 5.08% (Qwen-2.5)
and 7.66% (Gemini-2.5), indicating the effectiveness of the
augmented knowledge. (2) As shown in the last line, adding
the GAR module further brings up the performance by 1.86%
(Qwen-2.5) and 2.14% (Gemini-2.5). The GAR module sig-
nificantly improves Top-1 PSRecall by computing fine-grained
scores, especially for medium- and hard-level queries. (3)
Query pruning is effective in PSR, as reflected by the results
of Ours (w OSR), but has a negative effect in GAR. These
findings are consistent with our motivation for designing the
DGPR module.
We further evaluate the effects of the number of paths
retained after PSR (k 1) and GAR (k 2). As shown in Figure 7,
the optimal settings depend on question complexity. For Easy
and Medium cases, the best results are achieved withk 1= 10
andk 2= 1, indicating that a single accurately located path
is sufficient for simpler queries. For Hard cases, performance
peaks atk 1= 20andk 2= 5, suggesting that complex multi-
hop reasoning requires a broader set of knowledge paths to
provide sufficient context and satisfy expert-level constraints.
D. Ablation Studies on FashionEcoKG
To validate the essential role of our knowledge source, we
evaluated RAG performance across three distinct formats: our

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 10
TABLE VII: Ablation study results. OSR means PSR with the Original query, while PAR means GAR with the Pruning query.
PSRecall@1 Accuracy
Qwen-2.5-7B Gemini-2.5 Flash
Ablation Easy Medium Hard Overall Easy Medium Hard Overall Easy Medium Hard Overall
No RAG - - - - 90.27 64.90 57.97 67.23 92.63 67.49 61.83 70.94
Ours (w/o GAR) 2.36 24.12 17.66 19.11 91.15 71.28 62.57 72.31 93.22 78.13 70.25 78.60
Ours (w OSR) 2.65 35.25 25.53 27.74 89.3874.3262.19 73.75 94.1080.16 70.06 79.91
Ours (w PAR) 2.36 34.4727.6427.74 89.68 71.3666.2273.01 93.22 79.7774.2880.56
Ours2.95 35.49 27.64 28.44 92.0473.23 64.8874.17 93.8180.4772.8880.74
80859095
1 5 10K2 (GAR)
80859095
5 10 20K1 (PSR)
68707274
5 10 20K1 (PSR)
68707274
1 5 10K2 (GAR)
62646668
1 5 10K2 (GAR)
62646668
5 10 20K1 (PSR)
7172737475
1 5 10K2 (GAR)
7172737475
5 10 20K1 (PSR)Easy Medium
Hard Overall
Fig. 7: QA accuracy of PG-RAG with differentk 1,k2settings.
TABLE VIII: QA accuracy of RAG methods with different
knowledge sources and retriever backbones.
Retriever K-Type K-Source Easy Medium Hard Overall Impro.
SBERTChunksLen=100 88.79 59.38 58.35 63.78 -5.13%
Len=1000 88.20 57.98 54.89 62.00 - 7.78%
KGDBPediaFashion 87.32 61.63 54.13 63.87 - 5.00%
FashionEcoKG 88.79 70.97 57.77 70.58 + 4.98%
ContrieverChunksLen=100 89.38 60.54 54.13 63.54 - 5.49%
Len=1000 85.84 60.39 52.98 62.61 - 6.83%
KGDBPediaFashion 89.68 60.86 57.97 64.71 - 3.75%
FashionEcoKG 89.68 67.70 57.58 68.72 + 2.22%
LightRAGLen=1000 88.50 60.86 55.28 63.87 -4.72%
LinearRAGLen=100 89.09 60.47 56.81 64.10 -4.39%
FashionEcoKG, raw Fashion Textbook Chunks, and DBPedi-
aFashion using two retriever backbones. To ensure a fair and
direct comparison, we use a basic semantic-similarity retrieval
strategy across all resources: triple-level retrieval for the KGs
and chunk-level retrieval for the text-based resources. From
the experimental results shown in Table VIII, we have the
following observations: (1) Traditional text-based RAG con-
sistently degraded performance (dropping accuracy by 5–7%).
Larger chunks introduced excessive noise, obscuring relevant
information. (2) DBPediaFashion also had a negative impact.
Despite its large scale (1,072,977 nodes, 2,691 relations, and
over 2.36 million triples), it lacks the professional precision of
FashionEcoKG and contains too much noise and redundancy.
(3) Our FashionEcoKG consistently showed positive results.
Its success stems from its knowledge compactness and author-
itative, expert-anchored content.
We further compare two advanced KG-RAG methods on
textbook chunks, namely LightRAG [35] and LinearRAG [40],
which construct KG based on raw chunks in their distinct man-
ner. While these sophisticated methods outperformed naive se-
mantic retrieval, they still failed to achieve a positive net effect
6466687072
No
RAGV1 V2 V3QA Accuracy (%)
SBERT Contriever
-0.200.20.40.60.8
Deﬁni�on Inﬂuence Partonomy Characteris�cOrigin
Exempliﬁca�onInspira�on ComparisonFashionEcoKG-0.100.10.20.3
Deﬁni�onInﬂuencePartonomy
Characteris�cOrigin
Exempliﬁca�onInspira�onComparisonText Chunk
-0.100.10.20.3
Deﬁni�onInﬂuencePartonomy
Characteris�cOrigin
Exempliﬁca�onInspira�onComparisonDBPediaFashion
V1V2V3Fig. 8: QA accuracy improvement for different question types
with different knowledge sources (bars) and overall QA accu-
racy with three versions of FashionEcoKG under two retriever
backbones (chart).
on LLM answering accuracy. This observation suggests that
in expert-intensive domains characterized by high conceptual
density, imprecise knowledge extraction or the introduction of
irrelevant structural noise can have a detrimental side effect on
the generation process. Even advanced graph-based indexing
cannot fully overcome the inherent noise and fragmentation
in raw text, further underscoring the critical necessity of our
structured, high-fidelity FashionEcoKG.
To investigate the effectiveness of our KG construction
pipeline, we experiment on the three sequential versions of
FashionEcoKG (V1: Initialized Extraction, V2: Augmentation,
V3: Expansion) alongside an analysis across eight question
types in Figure 8. From the experimental results, we can
observe that: (1) QA accuracy improved consistently from V1
to V3 across both retrievers. This trend empirically validates
our hierarchical construction pipeline, proving that each stage,
from foundational expertise to relational densification, directly
enhances reasoning capacity. (2) The final FashionEcoKG
(V3) achieved positive gains in 7 out of 8 categories, rep-
resenting the most robust performance among all sources. In
contrast, DBPediaFashion provided only marginal support in
InspirationandComparison, while text chunks often caused
detrimental side effects in categories such asOriginandExem-
plification. (3) The highest gains were observed inInspiration
andComparison, suggesting that while LLMs possess basic
fashion facts, they lack the structural intelligence required
for complex cross-concept synthesis. Conversely, no source
improved performance in theDefinitiondomain, implying

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 11
Fig. 9: Two cases. Highlights are detected entities for coarse-
grained path retrieval for FashionEcoKG.
that LLMs’ parametric memory is already sufficient for basic
terminology, rendering external augmentation redundant for
simple identification.
E. Theoretical Cost Analysis of PG-RAG
We conduct efficiency and cost comparisons for different
retrieval methods. For PG-RAG, the embeddings of questions,
nodes, and knowledge paths are pre-computed offline. There-
fore, the online efficiency bottleneck mainly comes from node
similarity search and LLM-based reranking/answer generation.
Specifically, PG-RAG involves a one-time preprocessing stage
and a per-query online inference stage, with notations defined
in Table IX.
TABLE IX: Notations for cost analysis.
Notation Description
NNumber of questions in the dataset
VNumber of nodes in KB
Ptotal Number of KB paths, including 1,2,3-hops
CLLM Cost of one LLM call
Cenc Cost of one text-to-embedding operation
dEmbedding dimension
Ncand Number of candidate paths for a question
topk Number of candidates retained for reranking
During preprocessing, PG-RAG generates pruned questions
and extracts entities with LLM calls, incurringO(N·C LLM),
and encodes questions, KB paths, and nodes with costs of
O(N·C enc),O(P total·Cenc), andO(V·C enc), respectively.
Although path encoding dominates due to the large number
of paths, it is performed only once during KB construction.
Moreover, sinceC enc≈0.02seconds and encoding supports
batching, this offline cost is manageable.
For online inference, each query first performs coarse re-
trieval through entity encoding and node similarity search,
with costsO(e·C enc)andO(e·V·d), respectively. It
then applies PSR scoring over candidate paths with costO(N cand·d), followed by GAR reranking of the top-kpaths
using LLM calls, incurringO(topk·CLLM). Finally, answer
generation requiresO(C LLM). Thus, the overall online cost is
mainly governed by node similarity search and LLM reranking
over a small set of top-kcandidates.
In our experiments, we set topk= 10, which keeps the
average online retrieval time of PG-RAG at approximately
1.5 seconds per query. The subsequent QA latency mainly
depends on the chosen LLM, with an average inference time
of approximately 5–8 seconds per query. By comparison,
TOG requires about 60 seconds per query for retrieval, while
KAPING takes around 1 second per query when embeddings
are pre-computed. These results show that PG-RAG achieves
a practical balance between retrieval effectiveness and online
efficiency.
F . Case Study
In Figure 9, we provide two illustrative examples of fashion
QA, showing the original question, the pruned query, the
retrieved context, and the final generated answer. These cases
demonstrate how different modules in PG-RAG work together
to improve retrieval reliability and reasoning accuracy.
In the first case, the original question contains multiple
potentially distracting concepts, includingFull Skirts,Trench
Coats,Trench Coat Design, andWar. Direct retrieval based on
the full question may be affected by irrelevant terms such as
Full Skirts, which is not central to answering the question. Our
query pruning module removes such distracting information
and reformulates the question into a concise semantic skeleton:
identifying the century of the war that influenced trench coat
design. This pruned query better captures the core reasoning
intention. As a result, the retrieval module successfully obtains
a relevant KG pathindicating thatthe trench coat was influ-
enced by World War I. The generated answer then combines
the retrieved context with temporal knowledge that World War
I occurred from 1914 to 1918, leading to the correct answer,
i.e., the 20th century. This example shows that query pruning
can reduce semantic noise and improve the alignment between
the query and the relevant knowledge path.
The second case further illustrates the advantage of re-
trieving structured context from FashionEcoKG. The original
question asks where the Hippie-influenced style of Bermuda
Shorts originated, but it also contains several related fashion
concepts that may confuse retrieval, such asBermuda Shorts,
Shorts, andHippie style. PG-RAG prunes the query into
a more focused form that directly asks for the origin of
the Hippie-influenced style. Based on this refined query, the
retrieval module obtainsa compact but informative context
path:Bermuda shorts are associated with shorts inspired by
Hippie style, which originated in San Francisco. This retrieved
path provides explicit evidence for the model to compare
the candidate options and select the correct answer. The
final response therefore grounds its reasoning in the retrieved
context rather than relying only on parametric knowledge.
Overall, these examples reveal three advantages of PG-
RAG. First, the pruning step improves query clarity by re-
moving distracting or less relevant concepts while preserving

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 12
the key reasoning intent. Second, the path-based retrieval
over FashionEcoKG provides concise and structured evidence
that connects fashion items, styles, influences, and origins.
Third, the grounding and answer generation process enables
the LLM to reason over the retrieved context and produce
interpretable answers. Therefore, PG-RAG not only improves
final QA accuracy, but also enhances transparency by exposing
intermediate retrieval and reasoning results.
VI. CONCLUSION
This paper addresses knowledge-intensive fashion question
answering with FashionEcoKG, an expert-anchored knowl-
edge graph built through a three-stage hierarchical pipeline
that grounds academic concepts in authoritative textbooks and
expands them via cross-domain KG alignment and open-world
data, capturing the broader fashion ecosystem overlooked by
existing resources. To leverage this structured intelligence,
we propose PG-RAG, whose Dual-Granularity Path Selec-
tion module combines Pruning-based Semantic Ranking to
neutralize query noise and Grounding-based Agentic Ranking
to provide logical verification. PG-RAG significantly out-
performs text-based and KG-RAG baselines, confirming that
while LLMs hold sufficient parametric memory for basic
terminology, our framework supplies the structural intelli-
gence required for complex cross-concept synthesis in expert-
intensive vertical domains such as fashion.
REFERENCES
[1] N. Oh, E. Ko, and M. Cho, “Fashion ai across the value chain: A
comprehensive literature review and future agenda,”Journal of Global
Scholars of Marketing Science, vol. 35, no. 4, pp. 516–538, 2025.
[2] L. Liu, H. Zhang, D. Zhou, and J. Shi, “Toward fashion intelligence
in the big data era: State-of-the-art and future prospects,”IEEE
Transactions on Consumer Electronics, vol. 70, no. 1, pp. 36–57, 2023.
[3] X. Wang, C. Wang, L. Li, Z. Li, B. Chen, L. Jin, J. Huang, Y . Xiao, and
M. Gao, “Fashionklip: Enhancing e-commerce image-text retrieval with
fashion multi-modal conceptual knowledge graph,” inACL: Industry
Track, 2023, pp. 149–158.
[4] H. Zhan, J. Lin, K. E. Ak, B. Shi, L.-Y . Duan, and A. C. Kot, “a3-fkg:
Attentive attribute-aware fashion knowledge graph for outfit preference
prediction,”TMM, vol. 24, pp. 819–831, 2021.
[5] M. Wang, A. Mahjoubfar, and A. Joshi, “Fashionvqa: A domain-
specific visual question answering system,” inCVPR, 2023, pp. 3514–
3519.
[6] Y . Ding, Y . Ma, W. Fan, Y . Yao, T.-S. Chua, and Q. Li, “Fashionregen:
Llm-empowered fashion report generation,” inCompanion Proceedings
of WWW, 2024, pp. 991–994.
[7] OpenAI, “Introducing gpt-5,” OpenAI Official Website, 2025. [Online].
Available: https://openai.com/index/introducing-gpt-5/
[8] DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J.-M. Song, R. Zhang
et al., “Deepseek-r1: Incentivizing reasoning capability in llms via
reinforcement learning,”ArXiv, 2025.
[9] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and
Q. Li, “A survey on rag meeting llms: Towards retrieval-augmented
large language models,” inKDD, 2024, pp. 6491–6501.
[10] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
H. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,”arXiv preprint arXiv:2312.10997, 2023.
[11] N. Xu, Z. Lu, H. Tian, R. Kang, J. Cao, Y . Zhang, A. Liu, “Learning to
supervise knowledge retrieval over a tree structure for visual question
answering, ”IEEE Transactions on Multimedia, 2024.
[12] Liao, J., Liu, C., Ding, Y ., Wang, H., Tang, Z., Li, K. and Li,
K. “Enhancing Large Language Models Reasoning Via Multi-Path
Optimization on Knowledge Graph”.IEEE Transactions on Knowledge
and Data Engineering, 2026.[13] N. Matsumoto, J. Moran, H. Choi, M. E. Hernandez, M. Venkatesan,
P. Wang, and J. H. Moore, “Kragen: a knowledge graph-enhanced
rag framework for biomedical problem solving using large language
models,”Bioinformatics, vol. 40, no. 6, p. btae353, 2024.
[14] H. Qi, J. Li,Y . Ge, H. Wu, Y . Cai, S. Che, T. Sheng, Q. Zhao,
C. Chen, “DUAL-Know: A Description-Augmented and Uncertainty-
Aware GraphRAG Framework for Anesthesiology Question Answer-
ing,”IEEE Transactions on Knowledge and Data Engineering, 2025.
[15] S. Gao, K. Yu, Y . Yang, S. Yu, C. Shi, X. Wang, N. Tang, and H. Zhu,
“Large language model powered knowledge graph construction for
mental health exploration,”Nature Communications, vol. 16, no. 1,
p. 7526, 2025.
[16] Shang, S., Cheng, X., Xiong, Y ., Guo, F., Gao, S., Chen, X., Wang, F.,
Wang, Y ., Zhao, D. and Yan, R., “Personalized Review Summarization
by Using Graph-Based Retrieval Augmemted Generation, ”IEEE
Transactions on Knowledge and Data Engineering, 2025.
[17] S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R. Cyganiak, and Z. Ives,
“Dbpedia: A nucleus for a web of open data,” inInternational Semantic
Web Conference, 2007, pp. 722–735.
[18] M. Jia, M. Shi, M. Sirotenko, Y . Cui, C. Cardie, B. Hariharan,
H. Adam, and S. Belongie, “Fashionpedia: Ontology, segmentation,
and an attribute localization dataset,” inECCV, 2020.
[19] J. Barroca, A. Shivkumar, B. Q. Ferreira, E. Sherkhonov, and J. Faria,
“Enriching a fashion knowledge graph from product textual descrip-
tions,”arXiv preprint arXiv:2206.01087, 2022.
[20] Q. Jia, Y . Liu, D. Wu, S. Xu, H. Liu, J. Fu, R. V ollgraf, and
B. Wang, “Kg-flip: Knowledge-guided fashion-domain language-image
pre-training for e-commerce,” inACL: Industry Track, 2023, pp. 81–88.
[21] Y . Ding, P. Mok, Y . Bin, X. Yang, and Z. Cheng, “Modeling multi-
relational connectivity for personalized fashion matching,” inACM
MM, 2023, pp. 7047–7055.
[22] Y . Ma, X. Yang, L. Liao, Y . Cao, and T.-S. Chua, “Who, where, and
what to wear? extracting fashion knowledge from social media,” in
ACM MM, 2019, pp. 257–265.
[23] Y . Yuan, W. Zhang, Y . Deng, and W. Lam, “Multimodal fashion
knowledge extraction as captioning,” inProceedings of the Annual
International ACM SIGIR Conference on Research and Development
in Information Retrieval in the Asia Pacific Region, 2023, pp. 52–62.
[24] J. Han, N. Collier, W. Buntine, and E. Shareghi, “Pive: Prompting
with iterative verification improving graph-based generative capability
of llms,” inFindings of ACL, 2024, pp. 6702–6718.
[25] R. Yang, B. Yang, X. Zhao, F. Gao, A. Feng, S. Ouyang, M. Blum,
T. She, Y . Jiang, F. Lecueet al., “Graphusion: A rag framework for
scientific knowledge graph construction with a global perspective,” in
Companion Proceedings of WWW, 2025, pp. 2579–2588.
[26] H. Chen, X. Shen, Q. Lv, J. Wang, X. Ni, and J. Ye, “Sac-kg: Exploiting
large language models as skilled automatic constructors for domain
knowledge graphs,”arXiv preprint arXiv:2410.02811, 2024.
[27] X. Feng, H. Cui, K. Hayashi, H. H. Vu, K. T. Suzuki, N. Nishida,
H. Kamigaito, Y . Matsumoto, T. Watanabe, and C. Yang, “Llmkg+:
Systematically improving knowledge quality and coverage in kgs using
llms–a case study in medical domain,”AI Open, 2025.
[28] B. Ni, Z. Liu, L. Wang, Y . Lei, Y . Zhao, X. Cheng, Q. Zeng,
L. Dong, Y . Xia, K. Kenthapadiet al., “Towards trustworthy retrieval
augmented generation for large language models: A survey,”arXiv
preprint arXiv:2502.06872, 2025.
[29] A. Dutta, M. Singh, G. Verbruggen, S. Gulwani, and V . Le, “Rar:
Retrieval-augmented retrieval for code generation in low resource
languages,” inEMNLP, 2024, pp. 21 506–21 515.
[30] X. Yuan, L. Ning, W. Fan, and Q. Li, “mkg-rag: Multimodal knowledge
graph-enhanced rag for visual question answering,”arXiv preprint
arXiv:2508.05318, 2025.
[31] Zhang, Q., Chen, S., Bei, Y ., Yuan, Z., Zhou H., Hong Z., and others,
“A survey of graph retrieval-augmented generation for customized large
language models,”arXiv preprint arXiv:2501.13958, 2025.
[32] J. Baek, A. F. Aji, and A. Saffari, “Knowledge-augmented language
model prompting for zero-shot knowledge graph question answering,”
arXiv preprint arXiv:2306.04136, 2023.
[33] J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y . Gong, L. M. Ni,
H.-Y . Shum, and J. Guo, “Think-on-graph: Deep and responsible
reasoning of large language model on knowledge graph,”arXiv preprint
arXiv:2307.07697, 2023.
[34] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global:
A graph rag approach to query-focused summarization,”arXiv preprint
arXiv:2404.16130, 2024.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 13
[35] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “LightRAG: Simple and
fast retrieval-augmented generation,” inEMNLP, 2025, pp. 10 746–
10 761.
[36] Z. Li, X. Chen, H. Yu, H. Lin, Y . Lu, Q. Tang, F. Huang, X. Han,
L. Sun, and Y . Li, “StructRAG: Boosting knowledge intensive reason-
ing of LLMs via inference-time hybrid information structurization,” in
ICLR, 2025.
[37] Z. Li, Q. Guo, J. Shao, L. Song, J. Bian, J. Zhang, and R. Wang,
“Graph neural network enhanced retrieval for question answering of
large language models,” inACL-HLT, 2025, pp. 6612–6633.
[38] Y . Wang, N. Lipka, R. A. Rossi, A. Siu, R. Zhang, and T. Derr,
“Knowledge graph prompting for multi-document question answering,”
inAAAI, vol. 38, no. 17, 2024, pp. 19 206–19 214.
[39] X. Zhu, Y . Xie, Y . Liu, Y . Li, and W. Hu, “Knowledge graph-guided
retrieval augmented generation,” inACL-HLT, 2025, pp. 8912–8924.
[40] L. Zhuang, S. Chen, Y . Xiao, H. Zhou, Y . Zhang, H. Chen, Q. Zhang,
and X. Huang, “Linearrag: Linear graph retrieval augmented generation
on large-scale corpora,”arXiv preprint arXiv:2510.10114, 2025.
[41] C. Yan, Y . Chen, and L. Zhou, “Differentiated fashion recommendation
using knowledge graph and data augmentation,”IEEE ACCESS, vol. 7,
pp. 102 239–102 248, 2019.
[42] F. Sanguigni, D. Morelli, M. Cornia, and R. Cucchiara, “Fashion-rag:
Multimodal fashion image editing via retrieval-augmented generation,”
arXiv preprint arXiv:2504.14011, 2025.
[43]J. Han1, M. Guo, S. Li, K. Thiyagarajan, Z. Cheng, and L. Zhao, “Fit-
mag: A framework for generating fashion journalism using multimodal
llms, social media influence, and graph rag,”Artificial Intelligence and
Social Computing, vol. 163, p. 79, 2025.
[44] A. Maduro, C. Rodr ´ıguez, and S. Ortiz-Chamorro, “Vestime: A rag-
based multimodal fashion outfit recommender system,”The 51st Latin
American Computing, 2025.
[45] A. Balachandran and M. Masum, “Visiorag: A multimodal framework
for enhancing recommendation system using vision transformers and
rag,” inIEEE CAI, 2025, pp. 114–119.
[46] C. Xu, Y . Kang, Q. Feng, J. Hua, P. Li, F. Wu, H. Wei, X. Chen, S.-J.
Huang, and S. Chen, “Itemrag: Retrieval-augmented generation with
item-based knowledge computing for e-commerce product question
answering,”Big Data Mining and Analytics, 2025.
[47] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,
T. Killeen, Z. Lin, N. Gimelshein, L. Antigaet al., “Pytorch: An
imperative style, high-performance deep learning library,” inNeurIPS,
vol. 32, 2019.
[48] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yuet al., “Qwen2.5
technical report,”arXiv preprint arXiv:2412.15115, 2024.
[49] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings
using siamese bert-networks,”arXiv preprint arXiv:1908.10084, 2019.
[50] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zhenget al., “Qwen3
technical report,”arXiv preprint arXiv:2505.09388, 2025.
[51] N. Tatarinov, V . Kannan, H. Srinivasa, A. Raj, H. S. Anand, V . Singh,
A. Luthra, R. Lade, A. Shah, and S. Chava, “Kg-qagen: A knowledge-
graph-based framework for systematic question generation and long-
context llm evaluation,”arXiv preprint arXiv:2505.12495, 2025.
[52] P. Zhou, F. Zhang, X. Peng, Z. Xu, J. Ai, Y . Qiu, C. Li, Z. Li,
M. Li, Y . Fenget al., “Mdk12-bench: A multi-discipline benchmark
for evaluating reasoning in multimodal large language models,”arXiv
preprint arXiv:2504.05782, 2025.
[53] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y . Duan,
W. Su, J. Shaoet al., “Internvl3: Exploring advanced training and
test-time recipes for open-source multimodal models,”arXiv preprint
arXiv:2504.10479, 2025.
[54] W. Wang, Z. Gao, L. Gu, Z. Liu, H. Pu, L. Cui, X. Wei,et al., “In-
ternVL3.5: Advancing Open-Source Multimodal Models in Versatility,
Reasoning, and Efficiency,”arXiv preprint arXiv:2508.18265, 2025.
[55] Gemma Team, “Gemma 3 technical report,”ArXiv, arXiv:2503.19786.
[56] T. Yu, Z. Wang, C. Wang, F. Huang, W. Ma, Z. He, T. Cai,
W. Chen, Y . Huang, Y . Zhaoet al., “Minicpm-v 4.5: Cooking efficient
mllms via architecture, data, and training recipe,”arXiv preprint
arXiv:2509.18154, 2025.
[57] W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi,
J. Ji, L. Panet al., “Glm-4.1 v-thinking: Towards versatile multimodal
reasoning with scalable reinforcement learning,”arXiv e-prints, pp.
arXiv–2507, 2025.
[58] T. G. A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Rojaset al.,
“Chatglm: A family of large language models from glm-130b to glm-4
all tools,”ArXiv, vol. abs/2406.12793, 2024. [Online]. Available:
https://api.semanticscholar.org/CorpusID:270562306[59] A. Dubey, Abhinav. J, A. Pandey, A. Kadian, A. Al-Dahle,et al., ,
“The llama 3 herd of models,”arXiv–2407, 2024.
[60] X. Zheng, Y . Li, H. Chu, Y . Feng, X. Ma, J. Luo, J. Guo, H. Qin,
M. Magno, and X. Liu, “An empirical study of qwen3 quantization,”
arXiv preprint arXiv:2505.02214, 2025.
[61] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva,
I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosenet al., “Gemini
2.5: Pushing the frontier with advanced reasoning, multimodality,
long context, and next generation agentic capabilities,”arXiv preprint
arXiv:2507.06261, 2025.
[62] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut,
J. Schalkwyk, A. M. Dai, A. Hauth, K. Millicanet al., “Gem-
ini: a family of highly capable multimodal models,”arXiv preprint
arXiv:2312.11805, 2023.
[63] xAI, “Announcing grok,” xAI Blog, 2023. [Online]. Available:
https://x.ai/blog/grok
[64] W. Lin, J. Chen, J. Mei, A. Coca, and B. Byrne, “Fine-grained
late-interaction multi-modal retrieval for retrieval augmented visual
question answering,” inNeurIPS, 2023.
[65] M. Luo, Y . Zeng, P. Banerjee, and C. Baral, “Weakly-supervised visual-
retriever-reader for knowledge-based question answering,” inEMNLP,
2021.
[66] X. Zhao, Y . Zhang, W. Zhang, and X.-M. Wu, “Unifashion: A unified
vision-language model for multimodal fashion retrieval and genera-
tion,”arXiv preprint arXiv:2408.11305, 2024.