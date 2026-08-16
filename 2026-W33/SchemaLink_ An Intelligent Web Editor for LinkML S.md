# SchemaLink: An Intelligent Web Editor for LinkML Schema Curation

**Authors**: Emanuele Cavalleri, Paolo Perlasca, J. Harry Caufield, Justin Reese, Christopher J. Mungall, Marco Mesiti

**Published**: 2026-08-12 19:01:00

**PDF URL**: [https://arxiv.org/pdf/2608.12529v1](https://arxiv.org/pdf/2608.12529v1)

## Abstract
Motivation: LinkML is a suitable language for the representation of the structural and content constraints of different kinds of biomedical data. Even if it is a quite recent proposal, it has been applied in several biomedical contexts. Developing and maintaining LinkML schemas presents several challenges, particularly for novice curators. Non-expert bio-curators may struggle with LinkML syntax and best practices, requiring significant time and effort to develop well-structured schemas.
  Results: In this paper we propose SchemaLink, a web-based environment for the graphical construction and enhancement of LinkML schemas that address the following requirements: $(i)$ introduce a graphical language for the specification of LinkML schemas, $(ii)$ make uniform the specification of schemas in similar contexts, $(iii)$ simplify the design and curation processes by exploiting a RAG-based approach to assist curators in creating new schemas from scratch and editing already developed ones. Several experimental analyses show the quality of the produced LinkML schemas through the AI-based editing facilities.
  Availability and Implementation: SchemaLink is available online at: https://SchemaLink.biodata.di.unimi.it. SchemaLink code and testing data are available as open-source on GitHub at: https://github.com/AnacletoLAB/{schemalink-webapp,schemalink-api}.

## Full Text


<!-- PDF content starts -->

Journal Title Here,2022, 1–11
doi: DOI HERE
Advance Access Publication Date: Day Month Year
Paper
SchemaLink: An Intelligent Web Editor for LinkML
Schema Curation
Emanuele Cavalleri ,1Paolo Perlasca ,1J. Harry Caufield ,2Justin Reese ,2
Christopher J. Mungall2and Marco Mesiti1,2,∗
1Department of Computer Science, University of Milano, Via Celoria 18, Milano, 20133, Italy and
2Biosystems Data Science, Lawrence Berkeley National Lab, Calvin Rd, Berkeley, 94705, CA, USA
∗Corresponding author. marco.mesiti@unimi.it
FOR PUBLISHER ONLY Received on Date Month Year; revised on Date Month Year; accepted on Date Month Year
Abstract
Motivation:LinkML is a suitable language for the representation of the structural and content constraints
of different kinds of biomedical data. Even if it is a quite recent proposal, it has been applied in several
biomedical contexts. Developing and maintaining LinkML schemas presents several challenges, particularly
for novice curators. Non-expert bio-curators may struggle with LinkML syntax and best practices, requiring
significant time and effort to develop well-structured schemas.
Results:In this paper we propose SchemaLink, a web-based environment for the graphical construction
and enhancement of LinkML schemas that address the following requirements:(i)introduce a graphical
language for the specification of LinkML schemas,(ii)make uniform the specification of schemas in similar
contexts,(iii)simplify the design and curation processes by exploiting a RAG-based approach to assist
curators in creating new schemas from scratch and editing already developed ones. Several experimental
analyses show the quality of the produced LinkML schemas through the AI-based editing facilities.
Availability and Implementation:SchemaLink is available online at:
https://SchemaLink.biodata.di.unimi.it. SchemaLink code and testing data are available as
open-source on GitHub at:https://github.com/AnacletoLAB/{schemalink-webapp,schemalink-api}.
Key words:schema design, graphical language, RAG, intelligent interface
1. Introduction
LinkML [Moxon et al., 2025] is a flexible modeling language
for creating YAML-based schemas across various biomedical
domains. It allows the precise specification of biomedical
entities with their relationships and distinctive properties.
Unlike OBO ontologies [Smith et al., 2007] and biomedical
controlled vocabularies (e.g. KEGG [Kanehisa and Goto, 2000],
Rfam [Kalvari et al., 2020]) that characterize entire domains
of interest, LinkML schemas focus on specific aspects that
are of interest for conducting a given analysis. Classes and
relationships can be grounded in ontologies to represent their
semantics and leverage their identification schemes. Descriptions
and instances of schema elements can be included to improve the
characterization of key entities and document their roles. Although
it is a relatively recent framework, LinkML has already been
applied in several biomedical contexts, including the identification
and extraction of biomedical entities and relationships [Cavalleri
et al., 2024], cancer data harmonization [NIH National CancerInstitute, 2021], environmental genomics [National Microbiome
Data Collaborative, 2022], and knowledge graph integration [Unni
et al., 2022].
Developing and maintaining LinkML schemas, however,
presents several challenges, particularly for novice curators.
Non-expert bio-curators may struggle with LinkML syntax and
best practices, requiring significant time and effort to develop
well-structured schemas. Moreover, the following issues impact
LinkML schema curation: (i) The richness and flexibility of
LinkML offer diverse tree- or graph-like schema structural forms,
which can lead to inconsistencies, making schemas difficult
to compare and integrate across similar application domains.
(ii) Understanding and validating existing LinkML schemas
typically involves manually inspecting hundreds of lines of code,
which is both time-consuming and error-prone. Updating and
versioning schemas may introduce inconsistencies. (iii) Classes
and properties imported from biomedical ontologies may be
subject to semantic ambiguity. Misalignment between ontology
©The Author 2022. Published by Oxford University Press. All rights reserved. For permissions, please e-mail: journals.permissions@oup.com1
arXiv:2608.12529v1  [cs.DB]  12 Aug 2026

2 E. Cavalleri et al.
Fig. 1: Graphical representation of a schema.
terms and LinkML classes can introduce both syntactic and
semantic errors, complicating schema interoperability.
In this paper, we introduce SchemaLink, a web-based
environment for the graphical construction and enhancement of
LinkML schemas that address the previously outlined challenges.
SchemaLink extends thearrows.app[Jones and Karaca, 2019]
graphical environment proposed by Neo4j to develop property
graphs. Whilearrows.appis intended to draw graph nodes
and relationships at the instance-level, SchemaLink has been
conceived to work at the schema-level, allowing schema curators to
define classes and specify associative and inheritance relationships
among them. Beyond class and relationship modeling, SchemaLink
enables the definition of various schema constraints, including
attribute types, primary keys, mandatory and optional properties,
cardinality constraints on attributes and relationships, default
values, controlled vocabularies, descriptions, and examples taken
from ontologies. All these characteristics can be exploited via
simple graphical artifacts, avoiding the use of LinkML syntax.
Validation mechanisms are included in the platform to check the
adherence of the schemas to LinkML syntax. Generated graphical
schemas can then be serialized according to different schema
structural forms relying on the user’s needs.
An intelligent component has been integrated in SchemaLink
that exploits a vector database, fed with LinkML schemas
developed in several Monarch projects [Putman et al., 2023], and
general-purpose large language models (LLMs) combined with a
Retrieval-Augmented Generation (RAG) approach [Lewis et al.,
2020] to assist curators in creating new schemas from scratch
and enhancing already developed ones. For refining a schema
S, different kinds of schema modification prompts have been
formulated (like adding a class, including new relations between
two classes, explaining the role of a class/relation in the schema)
that can be issued to a LLM along with schema samples similar to
S(extracted from the vector database) to suggest a new version
that includes the proposed modification. This is an innovative
contribution of SchemaLink that can be exploited both by novice
and expert LinkML curators.
Several tools, proposed by different research communities,
provide a graphical abstraction of the schema adopted in a
given domain. Besides the general purpose modeling systems (e.g.
Entity-Relationship diagrams, UML class diagrams, and ontology
editor frameworks such as Prot´ eg´ e [Musen, 2015]), specific ones
have been introduced to describe schemas. Approaches can be
classified into three main paradigms: Labeled-property-graph
(LPG) diagrams [Di Pierro et al., 2023], Object-role modeling
(ORM) diagrams [Halpin, 2007], Ontology-based diagrams (e.g.
OntoPad [Arndt et al., 2021b], SHAPEness [Paciello et al., 2023]).These models provide different expressive powers for capturing
schema constraints. However, to the best of our knowledge, none
of them incorporates artificial intelligence features to assist end-
users in designing schemas, nor they provide support for LinkML
structural forms.
A key requirement that should be guaranteed by an intelligent
application like the one we propose is that the proposed
suggestions are meaningful and lead to identifying a reasonable
schema that fulfills the user’s expectations. However, identifying
expert curators to calibrate the system is often unfeasible in real-
world settings, due to both the limited availability of qualified
experts and the costs associated with their involvement. For this
reason, in this paper, we envisioned the use of the LLM-as-a-
Judge [Gu et al., 2025] technique for evaluating the quality of the
obtained schemas. This approach allowed us to conduct extensive
testing of different use cases and to limit the use of domain experts
to a very limited subset of questionable cases.
2. System and Methods
SchemaLink relies on an abstract graph data model according
to which LinkML schemas can be easily represented. The use
of graphical artifacts for representing the schema components
makes easy the exploration and development of LinkML schemas.
Moreover, an intelligent component has been realized to further
support the curators in the design and enhancement of schemas
that exploits a RAG-based method. Furthermore, our schemas can
be easily serialized in different structural forms, according to which
real-world LinkML schemas are generated.
2.1. A Graph Model for LinkML Schemas
A schemaSis a triple⟨C,R,M⟩, whereCandRare the set of
classes and relationships of the schema with the description of
their attributes, andMis a set of metadata associated with the
schema. SchemaLink allows the graphical specification of a schema
Sby means of different graphical artifacts that are described in
this section. The graphical representation is internally translated
into a LinkML schema because its textual representation facilitates
the use of LLMs for suggesting schema modifications.
Example 1Suppose a biomedical curator wishes to integrate -
omics data from heterogeneous sources structuring knowledge for
downstream tasks such as differential expression and pathway
analysis and functional enrichment studies. For this purpose,
the curator needs to create a schema for representing genes
participating in specific biological processes and their localization

SchemaLink: An Intelligent Web Editor for LinkML Schema Curation3
Fig. 2: Graphical artifacts for the specification of classes.
within cellular components (annotations can be retrieved from
Gene Ontology terms [Consortium et al., 2023]). Moreover, gene
expression can be measured via wet lab experiments such as PCR,
microarrays, and RNA-seq. For the generation of this schema, he
can exploit SchemaLink and develop the diagram in Fig. 1.
Classes.A classC∈ Cis visually represented as a circle that
can be added to the main canvas. When the circle is highlighted
an inspector panel is opened on the right side of the canvas (as
the one in Fig. 2). The user can specify the class name (denoted
class name) unique in the entire schema, a description, ontologies
to annotate the class with, and a sample of class instances (the
latter can be used for entity extraction through few-shot and
LLMs [Caufield et al., 2024]). The backend application supports
the user in the specification of OBO ontologies and in suggesting
samples of OBO terms to be used as examples through the
Ontology Lookup Service API [Cˆ ot´ e et al., 2006]. Through the
inspector, the user can manage the class attributes. For each
attribute, the name should be declared along with a textual
description and its type (simple types likeint,bool,string,date,
composite types likelist,set,tupleof simple types, Regex-
based types for pattern validation). Alternatively, attributes can
reference other classes. This feature of embedding classes as
Fig. 3: Graphical artifacts for the specification of associations.
attributes allows the specification of inlined instances in models
that support this functionality [Moxon et al., 2025]. Attributes
can also be marked as mandatory and used as identifiers. Fig. 2
shows the inspector panel for theGO Termclass of our running
example. For this class, the curator has identified the properties
goid,label, and a list ofsynonyms.
For the sake of readability, the canvas reports the class
name, the ontologies associated with it, and the basic properties
of its attributes. The inspector panel reports details with the
visualization options for drawing the class and its properties.
Association and Inheritance Relations.A class can be
linked to another one to draw a relationR∈ Rby hovering
outside a node and dragging the cursor. This feature is inherited
from thearrows.appapplication and permits to intuitively connect
classes and build schemas. Then, the panel inspector can be shown
to specify the relationship type. By default, a link between two
classes is an association relationship, and the user can specify
its name (denotedrel name) and cardinality constraints (zero-to-
one,one-to-one,zero-to-many,one-to-many,many-to-many, as
well as the option of defining a custom minimum and maximum
cardinality for both the subject and the object). Depending on
the adopted cardinality constraint, a different kind of arrow is
drawn. The user may also report the way in which the relationship
should be navigated when serializing the schema (from left to
right, from right to left, or both directions). Moreover, the
user can annotate the relationship using ontologies that formally
define the relation type and retrieve samples of relationships from
the considered ontologies (backend facilities similar to the one
described for classes are available here). Finally, relationships
can be characterized through attributes and their specification is
similar to the one for classes. In this case, attributes of type class
are used to representn-ary relationships (given its LPG roots,

4 E. Cavalleri et al.
Fig. 4: Internal representation of LinkML schemas and schema
portions for feeding the collections of the vector database.
SchemaLink facilitates the specification of binary relationships
while offering flexibility to other relationship types). Fig. 3 shows
the panel inspector for the associationparticipates inbetween
the classesGeneandBiologicalProcess. For this association, the
curator has identified an instance sample of plausible relationships
and an attribute namedexpreferencing the classExperiment. The
name of the relationship (i.e.participates in) is grounded in
the corresponding Relation Ontology (RO [Mungall et al., 2023])
property.
Once an association is drawn, it can be updated into an
inheritance one that is rendered as a UML-like empty arrow.
When this option is chosen, the inspector panel is simplified
by maintaining only the relation type. Once the inheritance
relationship is specified, the child class inherits all the properties
and associations of the parent class. Fig. 1 shows thatBio-
logicalProcessandCellularComponentare subclasses ofGoTerm.
2.2. The SchemaLink Intelligent Component
SchemaLink integrates a RAG approach to support both the
semi-automatic generation of schemas from scratch and the
enhancement of existing schemas through a structured set of
parametric schema editing prompts. These prompts, instantiated
on the selected schema items, guide a general-purpose LLM
in suggesting coherent extensions, refinements, or explanations
consistent with the existing schema structure.
When generating a schema from scratch, the user provides a
textual description of the target domain. Based on this description,
the system proposes an initial LinkML schema in the editing
canvas, including candidate classes and relationships. The user
may then refine the schema either manually (by interacting with
the graphical artifacts) or by invoking the intelligent editing
functionality to apply modification operations to the entire schema
or to selected portions of it.In the remainder, we describe the organization of the vector
database, the taxonomy of schema modification prompts, and the
overall workflow of the intelligent SchemaLink component.
Vector Database.The vector database is organized into the
following four collections, designed to support specific schema
editing operations during the RAG: (i) entire LinkML schemas
(collection (S)); (ii) only classes extracted (collection (C)); (iii)
only relationships (collection (R)); and (iv) relationships enhanced
with the specifications of the classes they connect (collection
(C+R)). Examples of schemas are initially introduced in the (S)
collection, while the other collections are automatically populated.
A Levenshtein distance [Levenshtein, 1966] filter is used to avoid
the presence of near-duplicate classes and relationships within a
collection (details in the Supplementary Materials).
Example 2Consider the simple schema reported in Fig. 4.a
that describes the cell lines involved in diseases. Fig. 4.b shows
the internal representation within SchemaLink, whereas Fig. 4.c
shows the corresponding graph-like form that is stored in the
collection(S). Finally, Fig. 4.d shows the portions of schemas
that are stored in the other collections.
Schema Editing Operations.SchemaLink offers a set
of 43 schema editing operations. These operations have been
classified according to two dimensions: (i) the target of the
modification, i.e. the portion of the schema to which the
operation applies (a class, relationship, subgraph involving
multiple classes and relationships), and (ii) the type of operation
performed on that target. Four operation types have been
identified:Add, which includes operations that introduce new
classes/relationships/attributes or annotate existing ones;Fix,
which encompasses refinements aimed at improving the quality
of existing schema items, including enhancements to class names,
attribute types, and cardinality of relationships;Explain, which
provides human-readable explanations of the role and semantics of
a selected schema portion within the overall schema;Reification,
which includes operations that promote attributes from a class to
new classes linked through appropriate relationships.
Moreover, we define a repository of schema editing prompts
containing pairs⟨op, M op⟩, whereopdenotes an editing operation
andM opis its associated parametric prompt template. At run-
time, the templateM opis instantiated according to the selected
schema portion (i.e. the class, relationship, or subgraph selected by
the user) and associated parameters (including the element name,
attributes, reference ontologies, and instance sample).
Listing 1 reports the parametric template prompt associated
with the operationAdd associationfor the generation of a new
association between two classes. Parameters (rendered in blue) are
dynamically replaced with the names of the selected classes.
Listing 1.Parametric prompt for generating a new association between two classes.
From the LinkML schema provided below , add one or more new
semantically meaningful relationships between the classes
class name 1and class name 2.
Finally, eachopis associated with a specific collection of
the vector database, which determines the type of examples
retrieved through the RAG. The selected collection reflects the
contextual needs of the operation (e.g. full-schema context, class-
level patterns, relationship-level patterns, or both classes and
relationships) and takes advantage of an empirical evaluation that
we have conducted (details will be provided in the experiments).

SchemaLink: An Intelligent Web Editor for LinkML Schema Curation5
Target Operation Schema Modification Coll.
classaddadd a new class semantically similar toclass name S
add a new class in relation withclass namethrough the relationrel name C
add [attributes|attributes description|parent class|child class] toclass name S
annotateclass namewith relevant [ontologies|samples|description] C
fixrenameclass name R
enhance attributes [name|description] ofclass name R
update attributes type forclass name C+R
update ontologies that annotateclass name S
enhance [sample] forclass name C
reification extract attributes fromclass nameto create a new class S
explain explain in human-friendly terms the role ofclass namein the schema -
relationaddadd a new association betweenclass name 1andclass name 2 C+R
add relevant attributes torel name S
annotaterel namewith relevant [ontologies|samples|description] C+R
introduce attribute description forrel name S
fixrenamerel name C+R
enhance cardinality ofrel name S
enhance attributes [name|type] ofrel name S
update the attribute description forrel name R
enhance [ontologies|samples] forrel name S
explain explain in human-friendly terms the role ofrel namein the schema -
Table 1.Description of main schema editing operations.
Table 1 summarizes the main modification operations
supported by SchemaLink for classes and relationships, along
with their target, operation type, and associated vector database
collection (Coll.). Operations on subgraphs extend or combine
those defined for classes and relationships.
Intelligent SchemaLink Workflow.A textual description
of the target domainTis provided by the curator that needs
to develop a new schema from scratch. The collection (S)
of the vector database is queried using the embedding ofT
to identify previously curated schemas in semantically related
domains. Although schemas represent structured knowledge,
LinkML schemas are stored as textual artifacts in the repository.
As a result, a LinkML schema embedding captures thematic and
structural features comparable to those present in the textual
domain descriptionT. The Top-kmost similar schemas are
therefore used as contextual examples to guide the LLM in
generating a schema. If the LLM hallucinates and the generated
output is not a valid instance of the LinkML language, the system
applies syntactic correction routines and, if necessary, re-prompts
the LLM until a valid schema is obtained.
The workflow is more articulated when modifying an existing
schema. In this case, the canvas already contains a schemaS, and
the curator selects the portionP⊆Sto be enhanced (e.g. a class, a
relationship, or a subgraph). For example, consider the schema on
the left-hand side of Fig. 6, where the curator selects the classes
GeneandCellularComponent. By means of a drop-down menu,
the curator identifies the operationopto apply (in our example,
op=Add association). At this point, the intelligent component
receives the triplet⟨S, P, op⟩as input (step 1).
Using⟨P, op⟩, SchemaLink retrieves the corresponding prompt
templateM opfrom the repository and instantiates it according
to the selected portionP, yieldingM op(P) (step 2). The pair
⟨Mop(P), S⟩is embedded to retrieve contextually similar examples
CXfrom the vector database (step 3). This retrieval strategy
differs from a na¨ ıve RAG setup in which only the textual
description ofSis embedded. By jointly embeddingM op(P)
together with the current schemaS, retrieval becomes schema-
Fig. 5: Example of LLM prompt for generating a new association.
and operation-aware. As a result, the retrieved contextCXalso
depends on the intended operation kind required by the curator.
Finally, SchemaLink combines⟨M op(P), S, C X⟩to construct a
LLM prompt for the generation of an updated schemaS′(step 4).
Fig. 5 illustrates the structure of the prompt in the running
example. The generated schemaS′undergoes automatic syntactic
validation to address potential hallucinations as discussed above.
After validation,S′is rendered on the canvas for user inspection
(step 5). The curator may accept the proposed schema or further
refine it either manually or by applying additional schema editing
operations. In our example, the system proposes a new association
namedlocated inbetweenGeneandCellularComponent. A
complete example illustrating how the SchemaLink intelligent
component can be used to enhance our running example in
a transcriptomics context is provided in the Supplementary
Materials.
2.3. Representation of LinkML Structural Forms
The graphical artifacts introduced in the previous sections support
the generation and serialization of different LinkMLstructural
forms[LinkML Authors, 2025b] that allow schema designers to
choose the most appropriate representation for their data and
technical constraints.

6 E. Cavalleri et al.
Fig. 6: Intelligent SchemaLink workflow.
Selecting a structural form depends on the complexity of
the domain, the nature of the relationships, interoperability
requirements, query performance, and the intended data
infrastructure (e.g. document-oriented, property/RDF graphs).
In the current implementation, SchemaLink supports the
two most commonly adopted structural forms: tree-like and
graph-like representations. Schemas expressed in either form can
be imported into SchemaLink to generate the corresponding
graphical representation, and vice-versa, schemas created within
SchemaLink can be serialized into these forms. During import,
schemas undergo syntactic validation using standard LinkML
community tools to ensure correctness.
The tree-like form organizes data in a nested manner, where
associations are typically embedded within classes as attributes.
This style is well-suited for hierarchical or document-oriented data
models where nested groupings play a crucial role. Classes may
referenceclasses or can be nested using theinlinedclause. Class
inheritance is supported via theis aclause, enabling reuse of
attributes and annotations. This form is thus ideal to model
complex, interrelated data elements that are not easily captured in
a flat structure (e.g. organizational structures or product catalogs).
Data conforming to this form are usually serialized in JSON-like
or document-based databases (e.g. MongoDB [MongoDB, 2009]).
Example 3Consider the schema excerpt in Fig. 7. In the
tree-like form (left panel), the associationGene-participates
in-BiologicalProcessis defined as an attribute of the classGene.
Itsmany-to-manycardinality is represented using themultivalued
clause. The classBiologicalProcessis declared as a subclass of
GOTermvia theis aclause.
The graph-like form represents entities as nodes and
associations as edges, with properties attached directly to nodes
and edges. Inheritance is represented as in the tree-like form.
The graph-like form is well-suited for detailed, attribute-rich
models. Data conforming with this form are usually represented
as property graphs, e.g. via Neo4j [Neo4j, 2007]) or the
RDF-star [Arndt et al., 2021a] data model. For this form,
SchemaLink uses the OntoGPT [Caufield et al., 2024] core classes
(mainlyNamedEntityandTriple) and the property graph schema
paradigm [LinkML Authors, 2025a].
Example 4The schema in Fig. 7 can also be represented in
the graph-like form (right panel). Here, the association betweenGeneandBiologicalProcessis explicitly modeled as a class with
attributes defining thesubject,predicate(i.e. the relationship
type), andobjectof the relationship.
This dual representation of schemas has the positive side effect
that schemas represented through the tree-structured form can be
translated into the graph-structured form and vice-versa through a
heuristic algorithm that we have developed [Perlasca et al., 2025].
3. Implementation
The ChromaDB [Huber and Troynikov, 2025] vector database was
used for indexing schemas via the OpenAItext-embedding-3-large
model. For feeding the vector database we used a repository of
60 expert-curated LinkML schemas from the OntoGPT project
(available athttps://github.com/monarch-initiative/ontogpt/
tree/main/src/ontogpt/templates). To reduce variability arising
from heterogeneous schema formats and LinkML dialects, we
generated a graph-like schema form through the SchemaLink
import/export routines, followed by manual quality control.
Starting from the collection (S), we generated the downstream
collections. Specifically, collection (C) contains 157 classes,
collection (R) contains 127 relationships, and collection (C+
R) contains the same number of relationships, but enriched
with the class specifications. The LLM used for the intelligent
component isgpt-4o-minibecause it provides a good balance
between responsiveness and output quality. We settemperature
to 0 and the number of retrieved examples via RAG to 10 items
for each intelligent operation.
Users can register with SchemaLink to access its intelligent
functionalities. Registration is required because the RAG-based
component relies on GPT models, whose usage incurs operational
costs. Registered curators can introduce their schemas to enrich
the vector database. This community-driven enrichment process
improves retrieval quality and broadens the semantic coverage of
the RAG component. The application and the schema collection
are released as open source. This allows users to inspect indexed
schemas, run SchemaLink locally, and configure the system with
their own API keys (details at:https://anacletolab.github.io/
schemalink-docs).
To evaluate the SchemaLink intelligent component (both the
generation of schemas from scratch and refined schemas obtained
by the application of the schema editing operations), we considered
four state-of-the-art conversational LLM systems – ChatGPT

SchemaLink: An Intelligent Web Editor for LinkML Schema Curation7
Fig. 7: Tree- and graph-like LinkML structural forms.
(OpenAI), DeepSeek, Claude (Anthropic), and Gemini (Google) –
alongside ten expert human curators. Although human curators
serve as the primary reference standard, LLMs provide more
scalable estimates.
The LLM evaluation followed the LLM-as-a-judge paradigm [Gu
et al., 2025]. In the LLM-as-a-judge setting, more capable
models (in terms of parameter count and reasoning ability)
are employed to evaluate the outputs of the lighter-weight
model used as SchemaLink engine (i.e.gpt-4o-mini). To better
approximate realistic usage scenarios in which schema proposals
are iteratively refined, we decided to exploit the official web-based
conversational interfaces of these systems because they preserve
interaction history (the use of the stateless API calls would have
required reconstruction of dialog context at each step). Moreover,
using independent LLM judges improves fairness and reduces
vendor-specific bias (e.g. avoiding the situation in which only
ChatGPT is used to evaluate a schema produced by a GPT-based
engine). Prompts for interacting with LLM-judges are provided in
the Supplementary Materials.
4. Discussion
Several experiments have been conducted to evaluate, on the
one hand, the quality of the schema generated by the intelligent
component and, on the other hand, its efficiency and applicability
in a web environment. Specifically, we describe the experiments
for evaluating the quality of the schemas generated from scratch
and the quality of the enhanced schemas obtained through the
intelligent operations. Moreover, we have evaluated the impact of
different vector database collections on the RAG-based retrieval
strategy. Finally, we discuss the execution times of the intelligent
operations.
Disease Drug Protein Pathway RNA
ChatGPT 4 4 3 4 3
DeepSeek 3 4 3 3 3
Claude 3 4 3 4 3
Gemini 4 4 3 4 3
Curators 2.4±.5 3.7±.5 3.0±.8 3.7±.5 4.5±.5
Table 2.Evaluation of schemas generated from scratch.4.1. Quality Evaluation of Schemas generated from Scratch
To evaluate the quality of the schema developed from scratch, we
defined five case studies in collaboration with biomedical LinkML
curators. Each case corresponds to a textual description of a
domain of interest, and SchemaLink is used to generate an initial
schema. Then, the expert is asked to assign a discrete score from 1
to 5 to the generated schema according to the following guidelines:
1 = schema unintelligible or not applicable to the domain; 2 =
largely incorrect with major inconsistencies; 3 = broadly usable
but requiring several revisions; 4 = correct and coherent with only
minor refinements needed; 5 = fully correct and well-suited to the
domain. Adopting a five-value scale for schema quality aligns with
research practices for subjective evaluation [Likert, 1932].
Use cases correspond to schemas that model: (i) entities
involved in diseases (Disease); (ii) drugs and their mechanisms
of action (Drug); (iii) proteins, including functional domains
and post-translational modifications (Protein); (iv) biological
pathways and their participating entities (genes, RNAs, proteins,
metabolites;Pathway); and (v) relations among RNA molecules,
including subtypes such as ncRNAs and mRNAs, and their further
specializations such as miRNAs and siRNAs (RNA). Prompts issued
to the intelligent component of SchemaLink for generating schemas
from scratch in these domains are provided in the Supplementary
Materials.
Fig. 8: Evaluation pipeline for SchemaLink intelligent operations.

8 E. Cavalleri et al.
Table 2 reports the scores assigned to theGenerateoperation
across the five use cases. For domain experts, we also include the
standard deviation across the ten evaluations. Scores provided
by LLM-judges are consistent across models, and 9/20 ratings
(45%) are greater than or equal to 4, which (according to our
protocol) indicates that the generated schema is coherent and
needs minor refinements.ProteinandRNA, which require more
specialized subclassing, are typically judged semantically correct
but incomplete. We also observe that ChatGPT and Gemini
assign slightly higher average scores (both 3.6) than Claude (3.4)
and DeepSeek (3.2); this may reflect family alignment with the
GPT-based engine used in SchemaLink (gpt-4o-mini), which was
likely trained on the same corpus of data as the ChatGPT model.
Domain expert evaluations are overall aligned with those of LLM-
judges: on average, 4/5 use cases receive scores above 3, with
RNAbeing considered the most complete (4.5). The reported
standard deviations are generally low since expert ratings typically
differ by at most one point. Qualitative feedback from domain
experts consistently highlights that classes and hierarchies are well
modeled, while relationship and attribute types are often the least
semantically refined components.
4.2. Quality Evaluation of Schema Editing Operations
To evaluate the quality of the schema editing operations, we have
adopted the pipeline described in Fig. 8. Starting from the schema
generated in the previous step, the curator is asked to choose an
intelligent operation (step 1 in the figure). Then, the SchemaLink
intelligent component proposes an updated schema to the user
(step 2). Finally, the curator is asked to score on a 0–5 scale
the relevance of the obtained schema (step 3). The guideline for
assigning the score is the following: 0 = failure to display any
changes; 1 = failure to accomplish the requested change; 2 =
substantial manual repair needed; 3 = the change is achieved with
minor issues; 4 = correct change with negligible adjustments; and,
5 = fully correct change with no adjustments. This procedure is
repeated six times per schema (i.e. six intelligent operations are
evaluated) to capture performance across a representative editing
sequence.
Fig. 9 summarizes the average scores obtained for each case
and judge. Out of 420 evaluations, 363 (86.4%) achieve a score
greater than or equal to 3, and 279 (66.4%) achieve a score
greater than or equal to 4. This means the vast majority of the
editing operations are semantically sound. It is worth noting that
9 evaluations referring to operations that modify attribute types
were assigned a score of 0 by LLM-judges. We inspected such
cases and noticed that the target already had an appropriate type,
meaning no change was required (i.e. the LLM hallucinated). We
re-prompted the LLMs, and they confirmed that these low scores
reflected an inappropriate choice of operation rather than a failure
of the intelligent component.
Overall,Addoperations achieve the highest performance
among all categories, with an average score of around 4, and they
were judged more than 200 times by human curator and more than
20 times by each LLM-judge. A plausible explanation is that these
operations benefit from a larger set of relevant examples stored in
the RAG collections, which enhances contextual grounding during
generation. Feedback from expert curators highlights that the
generated classes and relationships are generally appropriate and
well-formed, requiring only minor adjustments (e.g. refinement
of attribute types).Fixoperations–which require contextualEditing Operation Evaluation
5
35.0%
(147)
4
31.4%
(132)
3
20.0%
(84)0(9) 2.1%
2
10.2%
(43)11.2% (5)
Disease Drug Protein Pathway RNA
ChatGPT 4.3 4.3 4.5 3.5 4.0
DeepSeek 3.7 4.5 3.5 4.0 3.3
Claude 3.8 3.3 3.5 2.7 2.7
Gemini 3.3 3.7 3.2 2.3 3.5
Curators 4.0 3.7 4.2 4.0 3.7
Fig. 9: Average scores for intelligent editing operations.
reasoning over schema structure and constraints–achieve an overall
average score of 3.3. They were invoked 70 times by human
curators (mean score 3.7), 8 times by ChatGPT (mean score 3.0),
Gemini (2.2), and Claude (1.7), and 10 times by DeepSeek (3.0).
Feedback from curators suggests that fixes applied to subgraphs
are particularly effective and appreciated, as they enable one-
shot refinements of multiple elements. However, more fine-grained
constraints on attributes (e.g. enforcing that ap-valuemust be
lower than 0.05) are sometimes under-specified and could be
improved. TheReifycategory was invoked ten times by expert
curators (average score 4.5) and only once by LLMs (Gemini),
which assigned a score of 3 to the resulting schema. Curators noted
that reifications capture the intended structural transformation;
however, they also expected the inclusion of specialized attributes
or ontology for the resulting subclasses.
Fig. 10 details the five highest-performing intelligent operations
among those invoked at least once by each evaluator. Bar
height represents the average score assigned by each evaluator,
while the number displayed above each bar indicates how
many times the corresponding operation was selected and
evaluated.Addoperations are both frequently invoked and
highly rated. In particular,AddClassDescriptionwas invoked
27 times and achieved an average score of 4.7. Similarly,
AddClassOntologiesandAddRelAttributesreceived consistently
strong evaluations across judges. The assessments provided by
expert curators are closely aligned with those of LLM-judges
for these operations. AmongFixoperations,FixRelCardinality
achieves a mean score of 3 across 22 invocations, suggesting
that the intelligent component can support schema refinements.
Notably,FixRelCardinalityandAddClassAssociatedWithClass
receive higher evaluations from curators (exceeding LLM scores
by more than one point on average). This suggests human

SchemaLink: An Intelligent Web Editor for LinkML Schema Curation9
Fig. 10: Top-5 operations by average score.
evaluators better appreciate structurally sound and context-
dependent refinements, particularly when modeling complex
relationships.
Finally, we asked expert curators to answer three additional
questions assessing (i) the overall quality of the final schema, (ii)
the effectiveness of the enhancement process from the initial to
the refined schema, and (iii) the level of effort that would have
been required to perform the same refinement manually, without
intelligent assistance. On a 1–5 scale, the average score is around 4
across all questions. Qualitative feedback from curators highlights
that relationships are often less accurate than classes; however,
generated schemas are semantically acceptable and solid starting
points for further refinement.
We also asked curators to evaluate whether the final schema
was structurally coherent and meaningful (Yes/Partially/No). In
28 cases (56%), the answer wasYes, further supporting the overall
quality of the system.
4.3. Impact of “Custom” RAG Collections
We tested whether the organization in four collections of the
vector database improves the quality of intelligent schema editing
compared with a single collection containing only entire LinkML
schemas. We evaluate each use case four times by considering the
four collections (C,R,C+R,S) as example sources for the RAG
component. We relied only on the LLM-as-a-Judge technique to
scale this analysis because a complete human evaluation of all
configurations would be prohibitively expensive (we consider the
4 LLM-judges, 5 running use cases, 4 retrieval collections, and 6
operations resulting in 480 interactions with the system).
Fig. 11 reports the average score differences obtained by
using our custom collections depending on the editing operation
with respect to the baseline (S) strategy. Similar trends
are observed across the other collections; the corresponding
heatmaps and detailed statistics are provided in Supplementary
Materials.Overall, our strategy improves generation quality. Gains
reach up to +2.8 points (+56%) in theDruguse case under
DeepSeek, with an average improvement of +0.8 points (+16%)
across all models and domains. Average improvements per LLM-
judge are +0.56 (ChatGPT), +1.38 (DeepSeek), +0.20 (Claude),
and +1.04 (Gemini), indicating that the benefit of targeted
retrieval is robust across judges. Only one decrease is observed,
for theRNAcase under Claude (-1.10).
Fig. 11: Impact of our custom RAG collections wrt. (S).
4.4. Time Performance
To evaluate time performance, we executed the full set of 43
intelligent operations three times each on three different schemas
using a commodity laptop connected to the SchemaLink web
application via theGoogle Chromebrowser (release138, its
developer tool was used to assess time latencies). Table 3 reports
the average end-to-end latency and standard deviations, grouped
by both operation category and target type (class, relationship,
or subgraph). Overall, the measured latencies are often below
15 seconds. Moreover, the results indicate thatGenerate,Reify,
and operations acting on a subgraph require noticeably more time.
This is expected, as these operations either create new schema
components from scratch or manipulate sub-schemas that typically
involve multiple classes and relationships.
Latency also benefited from custom RAG collections. When
relying only on the (S) collection, complex operations could take
several minutes due to the large context window. Retrieving
targeted schema portions reduces the context size and leads to
faster inference. This is consistent with prior findings showing
that smaller context windows improve LLM inference efficiency
and latency [Gao et al., 2024, Liu et al., 2024, Chung et al., 2025].
Operation Mean±Std.(s)
Generate 9.68±1.99
Add 5.05±1.84
Fix 6.47±4.05
Reify 14.61±4.87Target Mean±Std.(s)
Class 5.71±3.81
Relation 5.48±3.52
Subgraph 7.71±3.05
Table 3.Average execution times by operation category and target type.
4.5. Concluding Remarks
The experimental campaign demonstrates the effectiveness of our
approach. Experts’ evaluations yield high scores and indicate
that the system is effective in supporting schema creation and
enrichment tasks. At the same time, they highlight important
directions for future improvements, as operations involving
complex relationships (i.e. those requiring structural reasoning
and constraint handling) remain more challenging. End-to-end
latencies are typically below 15 seconds, even for complex
operations such as schema generation or reification.
The graphical representation of schemas adopted in this paper
can be seen as a “bridge” across the several LinkML structural
forms. Translation algorithms have been defined to move from
one representation to the others. Thus, users can adopt the most
convenient form for their domains. At the current stage, the

10 E. Cavalleri et al.
platform supports tree-like and graph-like forms, but further forms
(e.g. tabular and relational) can be included.
Future developments will focus on extending supported
LinkML structural forms and on broadening import/export
capabilities to other schema modeling paradigms such as RDF
Schema [W3C Working Group, 2025], OWL [W3C Working Group,
2012], SHACL [SHACL Working Group, 2017], and PG-
schema [Angles et al., 2023]. Furthermore, we plan to enhance
schema modeling by allowing users to specify additional structural
and semantic constraints that go beyond syntactic validation. For
instance, users could define temporal consistency constraints (e.g.
an attributeyearassociated with a person cannot refer to a date
later than the person’s death) and ontology-based constraints (e.g.
instances of a classGeneassociated with aSpeciesmust comply
with the identifiers for that species). These constraints can be
integrated into the intelligent component so that schema semantic
inconsistencies are reduced during generation and editing.
References
R. Angles et al. Pg-schema: Schemas for property graphs.Proc.
ACM Manag. Data, 1(2), June 2023. doi: 10.1145/3589778.
D¨ orthe Arndt et al. RDF-star and SPARQL-star.https://w3c.
github.io/rdf-star/cg-spec/2021-12-17.html, 2021a.
N. Arndt et al. A visual shacl shapes editor based on ontopad. In
Int’l Conf. on Semantic Systems, 2021b.
J Harry Caufield et al. Structured Prompt Interrogation
and Recursive Extraction of Semantics (SPIRES): a method
for populating knowledge bases using zero-shot learning.
Bioinformatics, 40(3):btae104, 02 2024. ISSN 1367-4811. doi:
https://doi.org/10.1093/bioinformatics/btae104. URLhttps:
//doi.org/10.1093/bioinformatics/btae104.
Emanuele Cavalleri et al. SPIREX: Improving LLM-based relation
extraction from RNA-focused scientific literature using graph
machine learning. InProceedings of Workshops at the 50th
International Conference on Very Large Data Bases, VLDB
2024, Guangzhou, China, August 26-30, 2024. VLDB.org, 2024.
Yeounoh Chung et al. Is Long Context All You Need? Leveraging
LLM’s Extended Context for NL2SQL.Proc. VLDB Endow.,
18(8):2735–2747, April 2025. ISSN 2150-8097. doi: 10.14778/
3742728.3742761. URLhttps://doi.org/10.14778/3742728.
3742761.
The Gene Ontology Consortium et al. The Gene Ontology
knowledgebase in 2023.Genetics, 224(1):iyad031, 03 2023.
ISSN 1943-2631. doi: 10.1093/genetics/iyad031. URLhttps:
//doi.org/10.1093/genetics/iyad031.
Richard G Cˆ ot´ e et al. The ontology lookup service, a lightweight
cross-platform tool for controlled vocabulary queries.BMC
Bioinformatics, 7(1), February 2006. ISSN 1471-2105. doi:
10.1186/1471-2105-7-97. URLhttp://dx.doi.org/10.1186/
1471-2105-7-97.
D. Di Pierro et al. Lpg-based knowledge graphs: A survey, a
proposal and current trends.Information, 14(3), 2023.
Muhan Gao et al. Insights into LLM long-context failures:
When transformers know but don’t tell. In Yaser Al-Onaizan,
Mohit Bansal, and Yun-Nung Chen, editors,Findings of
the Association for Computational Linguistics: EMNLP 2024,
pages 7611–7625, Miami, Florida, USA, November 2024.
Association for Computational Linguistics. doi: 10.18653/
v1/2024.findings-emnlp.447. URLhttps://aclanthology.org/
2024.findings-emnlp.447/.Jiawei Gu et al. A Survey on LLM-as-a-Judge, 2025. URLhttps:
//arxiv.org/abs/2411.15594.
T. Halpin.Fact-Oriented Modeling: Past, Present and
Future, pages 19–38. Springer-Verlag, 2007. doi: 10.1007/
978-3-540-72677-7 2.
J. Huber and A. Troynikov. Chroma, 2025. URLhttps://www.
trychroma.com/.
Alistair Jones and Irfan Karaca. Arrows.https://arrows.app/,
2019.
Ioanna Kalvari, Eric P Nawrocki, Nancy Ontiveros-Palacios,
et al. Rfam 14: expanded coverage of metagenomic, viral
and microRNA families.Nucleic Acids Research, 49(D1):
D192–D200, November 2020. ISSN 1362-4962. doi: http:
//dx.doi.org/10.1093/nar/gkaa1047. URLhttp://dx.doi.org/
10.1093/nar/gkaa1047.
Minoru Kanehisa and Susumu Goto. KEGG: Kyoto Encyclopedia
of Genes and Genomes.Nucleic Acids Research, 28(1):27–30, 01
2000. ISSN 0305-1048. doi: 10.1093/nar/28.1.27. URLhttps:
//doi.org/10.1093/nar/28.1.27.
Vladimir I. Levenshtein. Binary Codes Capable of Correcting
Deletions, Insertions, and Reversals.Soviet Physics
Doklady, 1966. URLhttps://nymity.ch/sybilhunting/pdf/
Levenshtein1966a.pdf.
Patrick Lewis et al. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th
International Conference on Neural Information Processing
Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran
Associates Inc. ISBN 9781713829546.
R. Likert. A technique for the measurement of attitudes.Archives
of Psychology, 22 140:55–55, 1932.
LinkML Authors. How to make a property graph schema.https:
//linkml.io/linkml/howtos/model-property-graphs.html,
2025a.
LinkML Authors. How to recognize and work with
different structural forms.https://linkml.io/linkml/howtos/
recognize-structural-forms.html, 2025b.
Nelson F. Liu et al. Lost in the middle: How language
models use long contexts.Transactions of the Association for
Computational Linguistics, 12:157–173, 2024. doi: 10.1162/
tacla00638. URLhttps://aclanthology.org/2024.tacl-1.
9/.
MongoDB. MongoDB.https://mongodb.com, 2009.
Sierra A T Moxon et al. LinkML: an open data modeling
framework.GigaScience, 15:giaf152, 12 2025. ISSN 2047-217X.
doi: 10.1093/gigascience/giaf152. URLhttps://doi.org/10.
1093/gigascience/giaf152.
Chris Mungall et al. oborel/obo-relations: 2023-08-18
release. Available athttps://doi.org/10.5281/zenodo.
8263469, August 2023. URLhttps://doi.org/10.5281/
zenodo.8263469.
M.A. Musen. The prot´ eg´ e project: a look back and a look forward.
AI Matters, 1(4):4–12, 2015. doi: 10.1145/2757001.2757003.
URLhttps://doi.org/10.1145/2757001.2757003.
National Microbiome Data Collaborative. National
Microbiome Data Collaborative Schema.https:
//github.com/microbiomedata/nmdc-schema, 2022.
Neo4j. Neo4j Graph Database.https://neo4j.com, 2007.
NIH National Cancer Institute. CRDC-H schema.https://
cancerdhc.github.io/ccdhmodel/v1.1, 2021.
Rossana Paciello et al. SHAPEness: A SHACL-Driven Metadata
Editor. In Emmanouel Garoufallou and Andreas Vlachidis,

SchemaLink: An Intelligent Web Editor for LinkML Schema Curation11
editors,Metadata and Semantic Research, pages 274–288,
Cham, 2023. Springer Nature Switzerland.
Paolo Perlasca et al. From the graphical representation to the
LinkML code. Technical report, University of Milan, 2025.
Tim E Putman et al. The monarch initiative in 2024: an
analytic platform integrating phenotypes, genes and diseases
across species.Nucleic Acids Research, 52(D1):D938–D949,
11 2023. ISSN 0305-1048. doi: 10.1093/nar/gkad1082. URL
https://doi.org/10.1093/nar/gkad1082.
SHACL Working Group. Shapes Constraint Language (SHACL)
- W3C recommendation.https://www.w3.org/TR/shacl, July
2017.Barry Smith et al. The OBO foundry: coordinated evolution
of ontologies to support biomedical data integration.Nat.
Biotechnol., 25(11):1251–1255, November 2007.
Deepak R. Unni et al. Biolink Model: A universal schema
for knowledge graphs in clinical, biomedical, and translational
science.Clinical and Translational Science, 15(8):1848–1855,
June 2022. ISSN 1752-8062. doi: 10.1111/cts.13302. URL
http://dx.doi.org/10.1111/cts.13302.
W3C Working Group. Web Ontology Language (OWL) - W3C
recommendation.https://www.w3.org/OWL, December 2012.
W3C Working Group. RDF Schema (RDFS) - W3C
recommendation.https://www.w3.org/TR/rdf12-schema,
February 2025.

SchemaLink: An Intelligent Web Editor for LinkML Schema Curation
Supplementary material
Emanuele Cavalleri, Paolo Perlasca, J. Harry Caufield, Justin Reese, Christopher J. Mungall, Marco Mesiti
1 Vector Database Indexing Filter
Schemas are indexed in the vector database as textual representations, meaning that classes and relationships are embedded as
strings. Since different schemas may contain syntactically different but semantically equivalent entities (e.g., classes named
GeneandGenes), we apply a near-duplicate filtering procedure before indexing items into each collection.
For any pair of classes or relationships ciandcj, we compute a string-based distance over their names: dname(ci,cj) =
Lev 
name(c i),name(c j)
,where Levdenotes the Levenshtein distance. If dname(ci,cj)<0.2 , the two elements are considered
potential duplicates. In this case, additional comparisons are performed on their descriptions, attribute definitions, and ontology
annotations. If ddesc(ci,cj)<0.3 &d attr(ci,cj)<0.3 &d ann(ci,cj)<0.3 , the two entities are considered duplicates, and
only the element with the largest total character length is indexed in the database.
2 SchemaLink at Work
Suppose a biomedical curator is interested in organizing data about functional genomics experiments that were performed across
research groups. A common scenario involves transcriptomic studies (e.g. RNA-seq experiments) in which gene expression is
measured under specific experimental conditions, and genes are subsequently annotated with functional information. In such
contexts, it is essential to: (i)uniquely identify genes to avoid duplicate or ambiguous entries; (ii)characterize the experimental
setting (treatment, biological sample, perturbation, etc.); and(iii)annotate genes with GO terms.
Although LinkML can be profitably leveraged to design the schema for this domain and include usual entities, relationships,
and properties that applications need, its manual writing is time-consuming, and domain curators should inspect previously
developed schemas to generate an accurate new one.
By means of SchemaLink, curators can start from a simple textual description of the domain and obtain an initial schema
that takes advantage of the schema already loaded in the vector database. For example, with the following prompt“Generate a
LinkML schema involving experiments that measure the expression of genes, and include the cellular components in which
genes are located. Cellular components can be modeled as specializations of Gene Ontology terms. ”, the initial schema
in Supplementary Fig. S1 is generated. Its structure captures key requirements: genes are uniquely identified via HGNC
identifiers, experiments are explicitly represented as first-class entities, and annotations are structured according to the desired
GO categories.
Although genes are already linked to CellularComponent , functional genomics studies, like gene enrichment, typ-
ically require broader functional annotations, since genes are also interpreted in terms of the processes, functions, and
pathways in which they participate. To support this extended functional characterization, through the drop-down menu
for the class Gene , the curator invokes the operationAdd class associated with. SchemaLink introduces the new class
named BiologicalProcess , modeled as a specialization of GOTerm , together with the corresponding association
Gene-participates in-BiologicalProcess . After adding a few attributes to the newly introduced elements,
the schema evolves into the one shown in Fig. 1.
Furthermore, gene expression experiments are frequently performed in different organisms (e.g. human, mouse, Drosophila),
and results depend on the species under investigation. For example, differential expression patterns observed in a mouse
model may need to be compared with their human orthologs in comparative and translational studies. For this reason, to
explicitly represent the organisms involved in genomic experiments, the curator manually introduces the class Species . Then,
through Add operations, the intelligent component enriches this class by proposing the ontology NCBITaxon, a description (“A
taxonomic species, representing a group of organisms capable of interbreeding. ”), and a set of candidate attributes. The set of
attributes includes: (i)name (required), described as “the common name of the species”; (ii)scientific_name (required),
representing “the scientific (Latin) name of the species”; (iii)habitat (optional), “the natural environment in which the
species lives”; (iv)conservation_status (optional), “the extinction risk category”; and (v)synonyms (optional), “a
1

list of alternative names for the species”.
Further enhancements can be applied to the schema either manually or through the intelligent component. For example,
basic string types of attributes can be enhanced to more specific types (such as a controlled vocabulary for the attribute
conservation_status ) or instance-level examples that can annotate classes or associations. Further associations can also
be suggested by the SchemaLink engine (like Experiment-conducted on-Species andExperiment-has-Species ).
The curator can inspect alternatives and retain only the semantically appropriate ones (e.g. discardinghasbecause it is too
generic).
More advanced modeling patterns are also supported. For example, a recursive relationship on Gene can be introduced to
represent homologous genes. After its creation, the Fix operation can be applied to improve its name. The system proposes1
to 1 homologous to, which is aligned with RO.
At the end of this iterative refinement process, the schema in Supplementary Fig. S2 is obtained. The classes Experiment
andSpecies are introduced, with attributes and ontology annotations. Moreover, Species is embedded in Gene
via a required attribute species , ensuring that each gene instance is associated with an organism. The relationships
Experiment-conducted on-Species andGene-1 to 1 homologous to-Gene are proposed. Since it is
grounded in established bio-ontologies, it provides a framework for integrating in-house experiments and data retrieved
from centralized repositories.
Supplementary Fig. S1.Intermediate schema and menu with intelligent operations.
Supplementary Fig. S2.Graphical representation of the schema obtained after editing operations.
2/5

3 LLM-as-a-Judge Prompts
This supplementary section reports the prompts used within the LLM-as-a-Judge evaluation framework. The evaluation protocol
consists of three stages, each associated with a specific prompt: (i)evaluating the initial schema generated via SchemaLink
(Supplementary Listing S1, shown for the Disease use case); (ii)selecting an intelligent editing operation to improve the
schema (Supplementary Listing S2); and (iii) evaluating the updated schema after applying the operation (Supplementary
Listing S3).
Supplementary Listing S1.Prompt to evaluate the intial schema.
You are an expert LinkML schema designer working in the biological and biomedical fields. Note that in your domain of
expertise, LinkML relations are represented using the Triple core class and their predicates are represented using the
RelationshipType core class.
Rate my LinkML schema that is described as "a LinkML schema involving diseases" from 1 to 5, where:
1: Poorly written and poorly coherent with the description I gave you
2: Contains significant issues in structure, consistency, but a partial attempt at schema design is visible.
3: Adequately written, but lacking clarity, completeness according to your knowledge.
4: Well-written, mostly coherent with the intended schema description, with only minor issues or inconsistencies.
5: Excellent, fully coherent with the schema description, structurally consistent, and clearly written.
Schema:
{LinkML schema}
Supplementary Listing S2.Prompt to request an intelligent operation.
You are an expert LinkML schema designer working in the biological and biomedical fields. Note that in your domain of
expertise, LinkML relations are represented using the Triple core class and their predicates are represented using the
RelationshipType core class. I am asking you as an expert to choose an operation to improve the LinkML schema proposed
below. Choose an operation from the following list and specify the class, relationship, or schema portion on which you
would like to
execute the operation.
Schema:
{LinkML schema}
List:
{List of intelligent operations with their descriptions}
Supplementary Listing S3.Prompt to evaluate the intelligent operation.
I have updated the LinkML schema according to the operation you chose. Rate my LinkML schema from 0 to 5, where:
0: No apparent modification in the schema, even a minimal one.
1: Poor update, poorly coherent with the modification you gave me.
2: Contains significant issues, but a partial attempt of update is visible.
3: Adequately written, but lacking clarity, completeness according to the chosen modification.
4: Well-written, mostly coherent with the intended schema modification task, with only minor issues or inconsistencies.
5: Excellent, fully coherent with the schema modification task, structurally consistent, and clearly written
{Updated schema}
4 Generate Prompts
The following prompts were submitted to SchemaLink for the generation of schemas from scratch in the considered use cases:
•Disease:Generate a LinkML schema involving diseases.
•Drug:Generate a LinkML schema involving drugs and their mechanisms of action.
•Protein :Generate a LinkML schema involving proteins, including functional domains and post-translational modifi-
cations.
•Pathway :Generate a LinkML schema involving biological pathways and their participating entities (genes, RNAs,
proteins, metabolites).
•RNA :Generate a LinkML schema involving relations between RNA molecules (including subtypes such as ncRNA, mRNA,
and subtypes such as miRNA, siRNA, etc.).
3/5

5 Intelligent Operations and Collections
Supplementary Table S1 reports the retrieval collection that achieved the highest average score according to the LLM-judges
for each intelligent operation. The column Avg indicates the mean score assigned by the judges, whereas the total number of
invocations is reported in Times . In 18 cases ( ≈42%), the most effective collection belongs to one of the specialized subsets
C,R, orC+R . Ten operations (e.g.,FixClassExamples,AddSubschemaExamples) are not included in the table because they
were never selected by any LLM-judge during the experimental evaluation. We associate them with(S).
Finally, Supplementary Fig. S3 reports the average score differences obtained by using the “custom” collection strategy
with respect to the individual retrieval policies (C),(R), and (C+R) . The same trend observed in the comparison against (S)is
confirmed: selecting the most appropriate collection for each editing operation outperforms on average single fixed retrieval
strategy.
Operation Coll. Avg Times
GenerateS3.5 20
AddClassSimilarToClassS3.3 3
AddClassAssociatedWithClassC2.9 11
AddAttributesToRelationshipC+R4.2 6
AddClassesSimilarToEntitiesC+R3.3 3
ReifyClassS3.0 1
FixClassNameR2.8 8
FixClassOntologyS3.0 3
FixRelationshipCardinalityS2.9 12
AddAttributesToClassS4.0 8
AddAttributesDescriptionS4.5 2
AddParentClassS3.0 4
AddChildClassS3.7 6
AddClassOntologyC4.6 7
AddClassExamplesC4.4 5
AddClassDescriptionC4.8 16
FixClassDescriptionR3.9 9
FixClassAttributesNameR4.0 2
FixClassAttributesDescriptionR5.0 1
FixClassAttributesTypeC+R0.63 8
AddRelationshipAttributesDescriptionS5 1
AddRelationshipOntologyC+R4.3 4
AddRelationshipExamplesC+R5.0 2
AddRelationshipDescriptionC+R4.8 5
FixRelationshipNameC+R2.0 1
FixRelationshipDescriptionS3.0 1
FixRelationshipAttributesNameS4.0 1
FixRelationshipAttributesTypeS1.0 3
AddAssociationsSimilarToEntitiesC+R4.0 2
AddSubschemaDescriptionC+R2.0 1
FixClassesAndAssociationsNameS2.0 1
FixClassesAndAssociationsDescriptionR4.0 1
FixSubschemaOntologyS5.0 1
Supplementary Table S1.Best-performing retrieval collection for each operation.
4/5

(C)
(R)
(C+R)
Supplementary Fig. S3.Impact of our custom RAG collections wrt.(C),(R), and(C+R).
5/5