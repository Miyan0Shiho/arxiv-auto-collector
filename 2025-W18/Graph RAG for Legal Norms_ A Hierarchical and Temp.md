# Graph RAG for Legal Norms: A Hierarchical and Temporal Approach

**Authors**: Hudson de Martim

**Published**: 2025-04-29 18:36:57

**PDF URL**: [http://arxiv.org/pdf/2505.00039v1](http://arxiv.org/pdf/2505.00039v1)

## Abstract
This article proposes an adaptation of Graph Retrieval Augmented Generation
(Graph RAG) specifically designed for the analysis and comprehension of legal
norms, which are characterized by their predefined hierarchical structure,
extensive network of internal and external references and multiple temporal
versions. By combining structured knowledge graphs with contextually enriched
text segments, Graph RAG offers a promising solution to address the inherent
complexity and vast volume of legal data. The integration of hierarchical
structure and temporal evolution into knowledge graphs - along with the concept
of comprehensive Text Units - facilitates the construction of richer,
interconnected representations of legal knowledge. Through a detailed analysis
of Graph RAG and its application to legal norm datasets, this article aims to
significantly advance the field of Artificial Intelligence applied to Law,
creating opportunities for more effective systems in legal research,
legislative analysis, and decision support.

## Full Text


<!-- PDF content starts -->

Graph RAG for Legal Norms:
A Hierarchical and Temporal Approach
Hudson de Martim1
1Federal Senate of Brazil , hudsonm@senado.leg.br
Abstract
This article proposes an adaptation of Graph Retrieval Augmented Generation (Graph RAG) specif-
ically designed for the analysis and comprehension of legal norms, which are characterized by their
predefined hierarchical structure, extensive network of internal and external references and multiple
temporal versions. By combining structured knowledge graphs with contextually enriched text segments,
Graph RAG offers a promising solution to address the inherent complexity and vast volume of legal
data. The integration of hierarchical structure and temporal evolution into knowledge graphs - along
with the concept of comprehensive Text Units - facilitates the construction of richer, interconnected
representations of legal knowledge. Through a detailed analysis of Graph RAG and its application to
legal norm datasets, this article aims to significantly advance the field of Artificial Intelligence applied to
Law, creating opportunities for more effective systems in legal research, legislative analysis, and decision
support.
Keywords: Retrieval-Augmented Generation; Knowledge Graphs; Legal Tech; Legal AI; Legal Information
Retrieval; Knowledge Representation.
1 Introduction
In the contemporary era, where dependence on precise and accessible information is growing, the legal
domain presents a unique challenge to professionals and analysis systems due to its vast volume and
intrinsic complexity. Navigating this mass of data and efficiently extracting relevant information becomes
a progressively arduous task. This difficulty not only burdens legal professionals but also exposes the
limitations of conventional review and analysis methods – whether manual or digital – which are often slow
and susceptible to errors. This complexity is aggravated by legal language itself, which, with its specific
terminology and requirement for deep contextual understanding, presents essential nuances for rigorous
interpretation and analysis.
The challenge is amplified by three defining characteristics of legal norms: the predefined hierarchical
structure, the extensive cross-references (internal and external) and the multiplicity of temporal versions. In
this scenario, Artificial Intelligence (AI) emerges as a promising field, offering tools and techniques capable
of overcoming the barriers of traditional legal analysis and optimizing access to and comprehension of this
vast knowledge.
Legal frameworks are inherently hierarchical, with constitutions, statutes, regulations, and jurisprudence
organized in a structured manner. Understanding the relationships between the different levels of this hierarchy
is essential for the interpretation and application of Law. Higher-level legal principles (e.g., constitutional
provisions) frequently influence or restrict lower-level norms (e.g., specific regulations) – regarding the
hierarchy of norms and legal principles, see [1].
1arXiv:2505.00039v1  [cs.CL]  29 Apr 2025

Additionally, the dynamic nature of legal norms, evolving through amendments, repeals, and new
interpretations, results in multiple temporal versions. The ability to track and understand this evolution
[2] is, therefore, essential for determining the applicable legislation at a given moment. These defining
characteristics of legal norms, their structural hierarchy (alongside the cross-references) and their temporal
dimension, although representing significant analytical challenges, are indispensable for correct application
and normative interpretation. Consequently, they demand analysis approaches that explicitly capture both
dimensions – precisely the objective of the adapted Graph-based Retrieval-Augmented Generation (Graph
RAG) methodology in this article.
Graph RAG emerges as an evolution of traditional RAG [ 3], which combines information retrieval with
generation by Large Language Models (LLMs)1to enhance the relevance and precision of content. By
enabling LLMs to base their answers on external knowledge sources, going beyond the data used in their
training, RAG mitigates inherent limitations of these models, such as the propensity for hallucinations2.
By utilizing knowledge graphs to represent and structure the relationships between concepts and entities
extracted from these knowledge sources, Graph RAG [ 4] goes beyond traditional RAG by capturing the
intricate and interconnected relationships within and between documents, which is particularly valuable for
complex domains like Law. Legal knowledge is highly relational. Statutes refer to other statutes, clauses
reference other clauses, and jurisprudence is based on precedents. Knowledge graphs are well-suited to
represent these complex relationships.
Graph RAG presents significant potential for the analysis of legal norms. This technique can integrate
structured information from knowledge graphs with unstructured textual content to create richer and more
interconnected representations of legal knowledge. This integration allows capturing the complex relationships
between legal entities, such as laws, doctrine, jurisprudence, concepts, and the temporal evolution of legal
texts. By representing these legal entities, and their relationships, in a graph and linking them to the actual
text of the legal norms and their different versions, Graph RAG can provide a more holistic understanding of
Law [5].
The objective of this article is to contribute to the advancement of knowledge in Artificial Intelligence
applied to Law, detailing the analysis of Graph RAG and its application to datasets of legal norms, with the
goal of facilitating more effective systems to support legislative and legal work.
This paper is structured as follows. Section 2 first defines the core structural entities – Norm, Component,
and Version – essential for modeling legal documents, discussing their hierarchical relationships and the
challenges of representing temporal changes. Section 3 then elaborates on our proposed adaptations to
the Graph RAG framework, introducing the concept of cumulative Text Units, explaining the rationale
for modeling versions as aggregations rather than simple compositions, detailing the inclusion of essential
metadata and alteration actions, and discussing strategies for structure-aware retrieval. Lastly, Section 4
concludes the paper by highlighting the key contributions and suggesting avenues for future investigation.
2 Structural Entities in the Graph
A Graph RAG system operates through the combination of several essential components. At its core lies a
knowledge graph [ 6], which serves as a structured representation of knowledge. In this graph, nodes represent
entities such as legal concepts, articles of law, or judicial cases, while edges symbolize the relationships
between these entities, such as definitions, citations, or modifications. The knowledge graph provides a
semantic foundation for understanding these connections. It explicitly models relationships, which are often
1LLMs (Large Language Models), such as GPT-4 or Gemini 2.0, are AI models trained to understand and generate natural
language. For legal applications, see [16].
2LLMs are trained on vast amounts of data but may lack specific or updated information. RAG addresses this problem by
retrieving relevant documents and providing them as context to the LLM before it generates a response.
2

implicit in unstructured text, making them accessible for retrieval and reasoning.
The initial step in both Graph RAG and traditional RAG approaches is to divide each text in the input
dataset into segments, or “ text chunks ”. These segments are then converted into embeddings [ 7] and stored
in a vector database [ 8]. Various segmentation strategies can be employed, such as sentence segmentation,
paragraph segmentation, window-based splitting (according to the embedding model’s window size), and
semantic segmentation (dividing text according to its semantic blocks), among others.
The second step involves extracting Entities , along with their properties and relationships, from each text
segment generated in the previous step. These extracted entities serve as the “building blocks” or nodes of the
knowledge graph, with edges interconnecting them to represent the relationships derived from the normative
texts.
In the context of legal norms, the entities extracted from the texts can be categorized into two groups:
1.Content Entities
2.Structural Entities
Considering the first group, we can find traditional entities in the text of a legal norm such as types:
Person ,Organization ,Location ,Event ,Concept (metamodel entity3),Legal Norm (normative reference4) etc.
Regarding the second group, we find entities that represent the hierarchical structure of the text of a legal
norm (its articulation ), according to the formation rules of this type of text, and those that represent the
versions of this structure over time.
2.1 Legal Norm Structure
To structure the text of a Brazilian legal norm [9], the following components are defined5:
•da Parte Inicial (of the Initial Part):
◦Fórmula de Promulgação (Promulgation Formula): indicates the authority or institution respon-
sible for creating the norm.
◦Epígrafe (Epigraph): presents the full name of the norm.
◦Ementa (Abstract/Summary): provides a concise summary of the norm’s content.
◦Preâmbulo (Preamble): offers an introductory text that indicates the proposing body of the norm
and, when applicable, its legal basis.
•Articulação (Articulation):
◦Encompasses the legal provisions, structured into Dispositivos6(provisions) as illustrated in
figures 1 and 2.
•da Parte Final (of the Final Part):
3Legal metamodel, in this context, refers to a conceptual model that defines the fundamental types of entities and relationships
in the domain of Law, such as abstract legal concepts, principles, values, etc. Examples would be: "Right to Life", "Principle of
Legality", "Social Contract".
4Normative reference represents a remission or citation to another legal norm, either in its entirety or to a specific provision.
5In a more precise representation of the structuring of a legal norm, there would be the element Componente Autônomo
Articulado (Autonomous Articulated Component), which would group all parts of the norm, including annexes. But, for didactic
simplification, it will not be presented.
6ADispositivo (provision) of a legal norm is each element/component of the text that contains the prescription, i.e., the command,
the rule of conduct, the permission, the prohibition, or the authorization that the norm establishes. It forms the normative core of the
legal text.
3

◦Fecho (Closing): details the place and date of issuance of the norm.
◦Assinaturas e Assinaturas Complementares (Signatures and Complementary Signatures):
include the necessary signatory information.
This hierarchical organization not only facilitates the systematic interpretation of legal texts but also
enables precise navigation and retrieval of specific legal provisions. The basic unit of a Brazilian legal norm’s
articulation is the Artigo (Article). Each Article can be further subdivided into additional provisions, as
demonstrated in Figure 1.
Figure 1: Diagram of the hierarchical structure of possible subdivisions for the Article (Artigo) provision in
Brazilian legislation. (Diagram shows: Artigo (Article) →Caput; Artigo (Article) →Parágrafo (Paragraph);
Caput, Parágrafo (Paragraph) →Inciso (Item); Inciso (Item) →Alínea (Subitem); Alínea (Subitem) →Item
(Sub-subitem))
Articles can also be organized into Dispositivos de Agrupamento (Grouping Provisions), as illustrated
in Figure 2.
Figure 2: Diagram of the hierarchical structure of grouping for the "Article" (Artigo) provision in Brazilian
legislation (Diagram shows hierarchy: Articulação (Articulation) →Parte (Part) →Livro (Book) →Título
(Title)→Capítulo (Chapter) →Seção (Section) →Subseção (Subsection) →Artigo (Article))
4

As an example, Figure 3 presents the provision “Art. 12” of the “Constituição Federal Brasileira de 1988”
(Federal Constitution of Brazil (1988)), displaying its grouping provisions (“Título II” and “Capítulo III”)
and some of its child provisions (or subdivisions). Highlighted is the article, the basic unit of articulation.
Annotations in orange are representing the type of each provision.
Figure 3: Example of articulated text for Art. 12 of the Federal Constitution of Brazil (1988) with annotations
indicating the types of hierarchical provisions
As proposed, there will be 3 (three) types of Structural Entities for building the Knowledge Graph:
1.Norm (as a Whole): represents a legal norm.
2.Component: represents an element of the hierarchical structure of a legal norm. It can optionally be
composed of other components (its subdivisions).
3.Version: version of the content (e.g., label and text) of a component, or of a norm (as a whole), over
time – will be detailed further ahead.
TheNorm andComponent types are found in the Hierarchy dimension . The Version type is in the Time
dimension [10].
2.2 Legal Norm Text Segmentation
In the original Graph RAG proposal, the first step involves segmenting the entire document into smaller
text segments, or “text chunks”. In the subsequent step, each segment is processed—typically using a large
language model (LLM)—to extract the identified entities, along with their properties and relationships, which
are then added to the Knowledge Graph. However, when dealing with legal texts, the hierarchical structure
is a fundamental characteristic that must be preserved to ensure accurate interpretation. Therefore, it is
essential to extract and represent this hierarchical structure within the Knowledge Graph by mapping it to the
previously defined structural entities.
5

Ideally, semantic segmentation7should be employed to divide the document into text segments that corre-
spond directly to its structural elements, as illustrated in Figure 3. In this approach, the initial segmentation
step would simultaneously extract the structural entities, thereby capturing the hierarchical organization from
the outset. The subsequent step would focus on extracting content entities, as outlined in the original Graph
RAG proposal.
This proposed new first step (semantic segmentation oriented by the hierarchical structure of the legal
norm text) could be done using an LLM (with fine-tuning or prompt engineering) prepared to recognize the
structure of legal norms in the worked domain, generating a structured output in XML or JSON, for example.
Another option would be to use a specialized Parser8for segments of legal norm texts that recognizes
and processes this structure – in both options, human review is recommended, given the complexity and
importance of precision in interpreting the legal structure, and to ensure the correctness and quality of
segmentation and extraction.
As an example, a passage of the the “Constituição Federal Brasileira de 1988” (Federal Constitution of
Brazil (1988)) received the semantic segmentation (like showed in Figure 3), producing the entity of type
Norm , represented by a circle in green, and those of type “ Component ”, represented in blue in the hierarchy
presented in Figure 4.
Figure 4: Illustration of hierarchical semantic segmentation applied to a passage of the Federal Constitution
of Brazil (1988), representing the "Norm" (green), "Components" (blue) and their associated texts
7Semantic segmentation is a division of the document based on the meaning and structure of the text, rather than merely on
superficial textual criteria such as sentences or paragraphs.
8In the Brazilian case, an implementation of a Parser is available in the LexML project, which receives an unstructured text of a
Brazilian legal norm and produces an XML file with the structured text. This parser, developed specifically for Brazilian legislation,
represents a valuable tool for applying the proposed structured segmentation, as it automates the recognition and extraction of the
hierarchical structure of norms, facilitating the creation of the Knowledge Graph. LeXML Brazil, Legislative Integration Project.
Available at: https://projeto.lexml.gov.br . Accessed on: 15 Mar 2025.
6

2.3 Text Versioning over Time
Version entities represent the view over time of a norm/component, which can be extracted from its original
text, but also from the “ compilação multivigente ”9(multi-temporal compilation) of its text.
A component always has an original version, which corresponds to the text, and label, of the component
on the signature date of the original norm. It optionally has versions created by subsequent norms that
commanded, for example, alteration of the provision to new content (e.g., alteration of the wording of an
article by an amendment), or commanded a repeal (e.g., an article being explicitly repealed by a later law),
or a suppression (e.g., an item being suppressed by a later norm), or a renumbering (e.g., an article being
renumbered, label altered, by a later law) etc. of that component.
Since the text of a norm is constructed by hierarchically grouping the texts of its individual components,
the creation of a new version for any component automatically generates a new version for the entire norm as
well as for all aggregating elements/components that include that component.
For example, Figure 5 illustrates the Version entities for the provision “Art. 6” of the "Constituição
Federal brasileira” (Brazilian Federal Constitution)10, demonstrating how different versions reflect changes
over time.
Figure 5: Structure of the provision "Art. 6" of the "Constituição Federal Brasileira" (Brazilian Federal
Constitution), illustrating the Version entities (light gray circles) and their respective events and contents
over time. The versions represent the different wordings of "Art. 6" on different dates, due to constitutional
amendments.
2.4 Building the Graph
After semantically segmenting the text of a legal norm into a hierarchical structure (tree), as per the proposal
presented above, the elements of the hierarchy are then inserted as nodes of the types Norm (with its associated
9“Compilação” (Compilation) is a process in which, starting from the original version of a legal norm, all modifications
commanded by subsequent norms are applied to its text, up to the specified date, thus producing an unofficial updated text of this
norm on this date. It is "Multivigente” (Multi-temporal) when all dates on which the norm underwent alteration are considered in the
Compilation, producing an unofficial updated text (version) for each identified date.
10The structuring of the "Constituição Federal Brasileira de 1988" (Brazilian Federal Constitution of 1988) into "Norm", "Compo-
nents", and "Versions" can be visualized at https://normas.leg.br/?urn=urn:lex:br:federal:constituicao:
1988-10-05;1988 . Accessed on 15.03.2025.
7

properties, such as URN - Uniform Resource Name, name, signature date etc.) and “ Component ” (with its
properties such as URN, inclusion date etc.) into the Knowledge Graph, as illustrated in Figure 4, which
demonstrates the semantic segmentation of a Constitution segment.
Next, for each component, a node of the type Version will be inserted (and its properties such as URN,
version date, label/name, text etc.), with its associated text segment [ 11], as presented in Figure 6 – this
associated text segment (text chunk) is fundamental for the Retrieval Augmented Generation process, as
it represents the textual content of that specific version that will be retrieved and used as context for the
responses generated by the LLM.
Figure 6: Representation in the knowledge graph of the relationship between hierarchical structure entities
"Norm", "Component”, their original versions "Version", and the associated Text Chunks.
For example, the Norm11node/entity “Constituição Federal de 1988” (Federal Constitution of 1988),
identified by urn12
urn:lex:br:federal:constituicao:1988-10-05;1988 , has a Dispositivo13(“Provision”)
node/entity “caput do Art. 6º” (caput is the main text of the article), identified by
urn:lex:br:federal:constituicao:1988-10-05;1988!art6_cpt , which has associ-
ated the original version Version14node/entity, identified by
urn:lex:br:federal:constituicao:1988-10-05;1988@1988-10-05!art6_cpt , that
represents the version on the date 1988-10-05 (original version) of the text of the caput of Art. 6 of Federal
Constitution of 1988. For this Version node/entity, we have the associated Text chunk with the text bellow, as
illustrated in Figure 5:
"São direitos sociais a educação, a saúde, o trabalho, o lazer, a segurança, a previdência so-
cial, a proteção à maternidade e à infância, a assistência aos desamparados, na forma desta
Constituição."
11For a more precise representation in the Graph, specialized entity types can be created, derived from Norm , for each type of
norm in the treated legal system. In Brazil, there are, for example: "Constituição Federal" (Federal Constitution), “Lei Ordinária"
(Ordinary Law) etc. Thus, the entity representing a legal norm could be included in the graph using the specialized type.
12The formation of the urns presented in this work follows the standard proposed by the Brazilian LexML project.
13“Dispositivo ” (Provision) is a subtype of Component . For an even more precise representation in the Graph, specialized entity
types can be created, derived from Dispositivo , for each type of provision found in the treated legal system. For example, in the
Brazilian case, there are types like "Artigo" (Article), "Inciso” (Item), “Alínea" (Subitem), “Título” (Title), “Capítulo” (Chapter),
presented in figures 1 and 2. Thus, the entity representing a component could be included in the graph using the specialized type.
14For a more precise representation in the Graph, specialized entity types can be created, derived from Version , for each type of
version found in the treated legal system. For example, in Brazil, there are types like "Original”, “Alteração de Texto” (Text Change),
“Renumeração” (Renumbering), “Revogação” (Repeal) etc. Thus, the entity representing the version could be included in the graph
using the specialized type.
8

Unlike other fields, where the text remains static over time (for example, the transcribed speech of a
Parliamentarian in a Congress session will not change over time), the text over time of a legal norm, or of a
provision, can be represented by a set of Versions, as exemplified by Figure 7. In this figure, four versions of
the caput of Art. 6, found in Figure 5, are displayed: one original and three resulting from changes promoted
by “Emendas Constitucionais” (Constitutional Amendments).
Figure 7: Representation of the multiple "Version" entities for the caput do Art. 6 °provision (...!art6_cpt),
illustrating its textual evolution over time and the link of each version to its Text Chunk.
The proposal of this work is not only to incorporate the entities of the hierarchical structure of the legal
norm’s text into the knowledge graph, but also to include the Version entities of each component of the norm.
It is important to note that the text segments (text chunks), identified by the original text and by the versions
of the text over time of the norm, will be linked to the Version entities, and not to Component entities, as they
represent the text of the component at different moments in time, as shown in Figure 7 - in practice, this link
could be materialized as a property of the Version entity in the graph, just like a property for the embedding
of its content.
This temporal representation will allow answering questions about the text of a component considering
the temporal context: for example, if there was any change in the text of a component over time; what changes
the component underwent over time, or within a period; what was the text of the component on a specific
date, etc.
3 Adapted approach
3.1 Text Units
Regarding the text segment (text chunk) associated with each Version entity, in cases of subdivisions of a caput,
such as “ parágrafos ” (paragraphs) and enumeration elements (“ incisos ” (items) and “ alíneas ” (subitems)),
we suggest adopting the strategy proposed by [ 12]. In this strategy, the text segment to be considered
for a component is not only the text directly resulting from the semantic segmentation of the norm (or its
compilation over time) but the semantically complete text of the component, derived from post-segmentation
(and post-compilation) processing. The cited author recognizes the hierarchical and interdependent nature
of the components of a legal norm, where, for example, the meaning of an enumeration component (e.g.,
“inciso ” (item)) is frequently dependent on the context of its parent component (e.g., the caput).
In other words, when including in the graph, for example, the original Version of the provision "inciso I
do caput do Art. 5 º” (item I of caput of Art. 5), presented in Figure 4, the text linked to this entity will not be
just:
9

"I - homens e mulheres são iguais em direitos e obrigações, nos termos desta Constituição;"
("I - men and women are equal in rights and obligations, under the terms of this Constitution; ")
but rather the text formed by the content of the article’s caput (on the same date) plus the content of the
item itself, which can also be visualized in Figure 8:
"[Todos são iguais perante a lei, sem distinção de qualquer natureza, garantindo-se aos brasileiros
e aos estrangeiros residentes no País a inviolabilidade do direito à vida, à liberdade, à igualdade,
à segurança e à propriedade, nos termos seguintes:] I - homens e mulheres são iguais em direitos
e obrigações, nos termos desta Constituição;”
("[Everyone is equal before the law, without distinction of any kind, guaranteeing to Brazilians
and foreigners residing in the Country the inviolability of the right to life, liberty, equality,
security, and property, under the following terms:] I – Men and women are equal in rights and
obligations, under the terms of this Constitution; ")
Figure 8: Example of contextual (cumulative) Text Units for "incisos’ (items) of caput of Art. 5, demonstrating
the inclusion of the parental caput text to provide complete semantic context.
The version of an article’s caput will have as linked text its own text accumulated with the text from the
tree of its child components, as exemplified in Figure 9. The same applies to the article, the groupers and
the norm (as a whole), accumulating its own text with the text from their respective trees. According to the
cited author, this would bring more “aboutness”15to the texts linked to the entities [ 13]. In other words, the
cumulative “ Text Unit ”, by including the textual context of the parent, or child, components, represents the
central theme and the legal meaning of the component in question more completely and comprehensively,
increasing how well the text captures "what it is about" the provision.
Considering this new approach, we will call the texts associated with the entities Text Units , instead of
Text Chunks, as they are not necessarily the pieces of text resulting directly from the segmentation of the
norms’ texts, but rather the contextually complete content of each component.
3.2 Versions as Aggregations
In addition to the original version of a component C of a legal norm N, for each date/time T on which a
subsequent norm Y commands an alteration in the content of C, a new version T must be produced for
component C, as presented in Figure 10.
15The term "aboutness" refers to the capacity of a text to faithfully represent its central theme. For theoretical discussions, see [ 17].
10

Figure 9: Example of cumulative Text Unit for the caput of Art. 5 of the "Constituição Federal" (Federal
Constitution).
Figure 10: Versions of component C over time
Considering the proposal of Text Units resulting from the accumulation of text in the hierarchy, conse-
quently, a new version T must also be produced for every component of norm N that:
1. groups, directly or indirectly, component C, including the norm (as a whole, as level 0);
2.has its Text Unit formed by the accumulation of the text of C as a prefix (for example: " parágrafos ”
(paragraphs), “ incisos ” (items) and “ alíneas ” (subitems)).
In Figure 11, we observe that on the norm’s original date (1988-10-05), the original version of the
component "tit2" is formed by the original versions of its children. That is, the text of "Título II” (Title II) on
that date consists of the texts of its children from that same date.
On 1993-09-14, a new version for “tit2 cap4" was produced, either by its own alteration or indirectly by
the alteration of one or more components in its tree. A more common implementation would be to produce
new versions for other children of "tit2" on 1993-09-14 and then produce a 1993-09-14 version for "tit2",
which would be a composition16of its children’s versions on the same date - similar to the formation of the
"tit2" text on the original date.
However, the other children of "tit2" were not altered on 1993-09-14. Our proposal, therefore, is not
to produce versions for the unaltered children (for dates other than the original) and instead reuse the most
recent versions of these children. Instead of composition, the 1993-09-14 version for "tit2" would be an
aggregation17of the most recent versions of the children.
16In UML terms, Composition represents a strong whole-part relationship where the "part" belongs exclusively to one "whole"
and shares its lifecycle; if the whole is deleted, the part is too. In our initial comparison, this would mean a parent version strictly
owns child versions from the exact same date.
17Aggregation, in UML, represents a weaker whole-part relationship where the "part" can exist independently of the "whole" and
11

We can verify the formation of these version aggregations in Figure 11, where new versions of the
component "tit2" were produced as a consequence of new versions of one or more of its children on each
date.
Figure 11: New versions of the component "tit2" derived from new versions of some of its children
Thus, in the construction of the graph, besides each version being existentially linked to its component,
this work proposes that each version, when it is not a leaf, be a agregation with its child versions. In other
words, a child version does not "belong" exclusively to a single parent version, but rather contributes to the
formation of multiple parent versions at different moments in time.
When a version of the child component Cx does not exist for a date Tn, the most recent version of
Cx, prior to Tn, will be used in the aggregation, as exemplified by V1 of C3, which is being used in the
formation of version V2 of C1 (examples of this situation also appear in Figure 11). In this case, Version
V1 of component C3 contributes to version V1 of component C1 and to version V2 of component C1,
demonstrating that a child version can participate in multiple parent versions in different temporal contexts.
The orange arrows indicate the aggregation relationship (Figure 12).
Figure 12: Diagram of aggregation relationships between versions, illustrating how child versions can
contribute to multiple parent versions at different times and how earlier versions are reused in the absence of
subsequent changes.
can potentially be shared by multiple wholes. In our proposed model, this reflects how a child version (the "part") can be reused by
multiple parent versions (the "wholes") across different times without its existence depending on any single one.
12

This representation of versions as aggregation ensures uniqueness in the representation of a version. In a
traditional approach, there is a complete version of the articulation tree for each change date. That is, for
each component of the tree, there is a version for the date, even if the content of that component was not
changed on that date. In the proposed approach, there is a version for the component on the original date
and on each date its content was actually changed, thus eliminating redundant and unnecessary copies of the
version, contributing to a non-ambiguous and economical representation.
In the context of RAG, it allows the system to retrieve not only the specific version of a component, but
also the context of its parent and child versions, providing a more holistic and precise understanding of the
norm at a given moment, facilitating more complex queries about the temporal evolution of norms [ 2] - for
example, a legal researcher will be able to ask what the wording of a certain article was on a specific date and
obtain the precise answer thanks to the temporal component of Graph RAG.
3.3 Additional Context for Versions
This work also proposes that, for each Version inserted into the graph, information describing its validity status
and, when terminated, the type of action that promoted its termination (“ revogação ” (repeal), “ alteração de
texto ” (text change), “ supressão ” (suppression), “ derrubada de veto ” (veto override), among other examples)
and the responsible norm (the legal norm, and, ideally, the specific component of that norm, that promoted or
commanded the action) be added, when available, as additional context of the version.
This information of a version will be registered by the property Contexto Adicional (Additional Context)
of the Version entity in the graph. If new alteration actions occur between subsequent and prior norms, this
status can be updated over time.
For example, consider the caput of Art. 6 º(presented in Figure 5) of the “Constituição Federal Brasileira
de 1988” (Brazilian Federal Constitution of 1988), signed and published on 1988-10-05. Thus, at the time
of the Constitution’s publication, the Additional Context property of its original version (1988-10-05) will
contain information about the start of its validity:
"Essa versão tem vigência de 1988-10-05 até hoje."
("This version is valid from 1988-10-05 until today. ")
The version of a component is created using the signature date of its norm (when original) or the
signature date of the altering norm, when produced by an action. On the other hand, the validity of the
version starts from the publication date (by default in Brazil)18of the norm (when original) or the publication
date of the altering norm, when produced by an action – and the previous version of the component, when
exists, has its validity terminated using the date before this publication date.
For example, with the publication, on 2000-02-15, of “Emenda Constitucional n º26 de 14/02/2000”
(Brazilian Constitutional Amendment No. 26), signed on 2000-02-14, an action was commanded, through the
caput of its Art. 1 º, to alter the wording of the caput provision of Art. 6 °of the Brazilian Federal Constitution
of 1988 – the action is represented in the graph19by the Action20entity (in brown in Figure 13). This action
terminated the validity of the 1988-10-05 version and produced a new version, dated 2000-02-14 (with the
new text), for the altered component.
18Defining the start of validity for a legal norm and its components is a complex issue, due to a wide variety of possibilities.
Therefore, it will not be the focus of this article and will be explored in future work.
19The modeling of the “Action presented in Figure 13 was simplified for didactic purposes. It can be improved using ontol-
ogy techniques, such as those proposed in UFO (Unified Foundational Ontology), found at https://nemo.inf.ufes.br/
projetos/ufo/ . Accessed on 22.03.2025.
20For a more precise representation in the Graph, specialized entity types can be created, derived from Action , for each type of
action between norms found in the treated legal system. For example, in the Brazilian case, types like "Alteração de Texto" (Text
Change), "Renumeração" (Renumbering), "Revogação" (Repeal) etc. Thus, the entity representing the action could be included in
the graph using the specialized type.
13

Figure 13: Representation in the knowledge graph of the text change action
The new version, dated 2000-02-14 (signature date of the Amendment), of the caput of Art. 6 º
of the Brazilian Federal Constitution of 1988, will then have as the content of the Additional Context,
at the time of the Amendment’s publication :
"Essa versão tem vigência de 2000-02-15 até hoje."
("This version is valid from 2000-02-15 until today. ")
Furthermore, the content of the Additional Context of the previous version (dated 1988-10-05) of the
caput of Art. 6 °of the Brazilian Federal Constitution of 1988 will be updated, with a complement textually
describing the reason for and the entity responsible for the termination of its validity:
"Essa versão tem vigência de 1988-10-05 a 2000-02-14. Sua vigência foi encerrada pela nova
redação de texto dada a esse dispositivo pelo caput do Art. 1 ºda Emenda Constitucional n º26
de 14/02/2000."
("This version was valid from 1988-10-05 to 2000-02-14. Its validity was terminated by the new
text wording given to this component by the caput of Art. 1 ºof Constitutional Amendment No.
26")
The additional contexts can be persisted directly as textual metadata (property) of the version or be
generated automatically from the actions (and from start-of-validity andend-of-validity information)
inserted in the graph.
Either way, the content of the Additional Context should not be used in the creation of the Text Unit
embedding of its version, so as not to add noise to its semantic understanding.
The Additional Context is important to enrich the response generated by the LLM in the RAG process.
When the RAG system retrieves a specific Version as relevant context, the Additional Context associated
with this version can be sent along with the Text Unit to the LLM, providing valuable information about
the history and validity of the component, allowing for more complete and contextually rich responses – in
more advanced implementations, the additional context would be sent to the LLM only when it is sensed that
it is necessary for generating the response. This will provide context for the LLM to display, for example,
besides the texts in the timeline of a component, also information about the validity of each text and, in case
of termination, by whom and for what reason its validity was terminated.
3.4 Alteration Actions as Text Units
As previously presented, a subsequent norm can mandate an amendment to the wording of a provision of a
prior norm, a repeal of a provision, or of a prior norm in its entirely, etc. An additional proposal, therefore, is
to create a Text Unit representing each alteration action between legal norms/components. Considering the
action illustrated in Figure 13, the content of the new Text Unit will be:
14

"A Emenda Constitucional n º26, de 14 de fevereiro de 2000, por meio do caput do seu Art.
1º, deu uma nova redação ao caput do Art. 6 ºda Constituição Federal de 1988. Essa alteração
encerrou em 2000-02-14 a vigência da versão original desse dispositivo (de 1988-10-05) e
instituiu uma nova versão com vigência a partir de 2000-02-15, cujo texto passou a ser: ’São
direitos sociais a educação, a saúde, o trabalho, a moradia, o lazer, a segurança, a previdência
social, a proteção à maternidade e à infância, a assistência aos desamparados, na forma desta
Constituição.”
("Constitutional Amendment no. 26, of February 14, 2000, through the caput of its Art. 1 º,
provided a new wording for the caput of Art. 6 ºof the Brazilian Federal Constitution of
1988. This alteration terminated on 2000-02-14 the validity of the original version of this
provision (from 1988-10-05) and established a new version effective from 2000-02-15, whose
text became: ’Social rights include education, health, work, housing, leisure, security, social
security, protection of motherhood and childhood, and assistance to the destitute, in the manner
prescribed by this Constitution. ’ ")
Although this alteration command already appears in the text of the caput (main section) of Art. 1 º
of Constitutional Amendment no. 26, as shown in Figure 14, generating a Text Unit for the Action entity
structures and completes the information about the action/command, making explicit, for example, the validity
periods of the text versions.
Figure 14: Text of the Emenda Constitucional nº 26 de 14/02/2000 (Constitutional Amendment no. 26)
The generated Text Unit is then associated with the Action entity in the graph. And it receives its own
embedding, allowing, besides Version Text Units embeddings, for more sophisticated analyses of legislative
evolution, facilitates the traceability of normative alterations, and improves the understanding of the life cycle
of legal norms in the knowledge graph [14], in the context of RAG.
3.5 Metadata as Text Units
Besides the texts of legal norms, other sources of information can be used to feed the knowledge graph. One
possibility would be to generate Text Units with the metadata21, in textual format, to associate them with the
Norm andComponent entities, similar to the original proposal of entity summaries from Graph RAG – the
difference here is that this information, generally, is not contained within the text of the norm itself; therefore,
these summaries could not be extracted from the text chunks, as in the original proposal.
For example, for the Norm entity representing the legal norm “Lei n º10.406 de 10/01/2002” (Law no.
10,406 of 2002-01-10), a Text Unit would be generated textually describing the entity’s properties22:
21Metadata is data about data. In this context, structured or textual information describing the properties and characteristics of
legal norms, such as publication date, authorship, type of norm, relationships with organizations, etc.
22This Text Unit was generated from the structured metadata registered in the knowledge graph of the "Sistema de Gestão de
15

"São apresentados a seguir os metadados gerais da norma jurídica federal ’Lei n º10.406 de 10/01/2002’:
é do tipo ’Lei Ordinária’; está no nível ’Legal’ do ordenamento jurídico brasileiro; possui como urn:
’urn:lex:br:federal:lei:2002-01-10;10406’; possui como URL principal (uri): ’ https://normas.leg.
br/?urn=urn:lex:br:federal:lei:2002-01-10;10406 ’; possui como URLs alternativas:
’http://www.lexml.gov.br/urn/urn:lex:br:federal:lei:2002-01-10;10406 ’ e
’http://legis.senado.leg.br/legislacao/DetalhaSigen.action?id=552282 ’; foi
apreciada e aprovada pelo ’Congresso Nacional’; foi assinada em Brasília, 10 de janeiro de 2002; 181 ºda
Independência e 114 ºda República; foi assinada por FERNANDO HENRIQUE CARDOSO, Aloysio
Nunes Ferreira Filho; foi oficialmente publicada pela ’Imprensa Nacional’, tendo como data de publicação
original e oficial: 11/01/2002; tem como nomes alternativos: ’LEI-10406-2002-01-10’ e ’Código Civil
(2002) (CC)’; tem como ementa: ’Institui o Código Civil’."
("The general metadata for the federal legal norm Law no. 10,406 of 2002-01-10 are presented below:
it is of the type ’Ordinary Law’; it is at the ’Legal’ level of the Brazilian legal system; its URN is:
’urn:lex:br:federal:lei:2002-01-10;10406’; its main URL (URI) is: ’ https://normas.leg.br/
?urn=urn:lex:br:federal:lei:2002-01-10;10406 ’; its alternative URLs are: ’ http:
//www.lexml.gov.br/urn/urn:lex:br:federal:lei:2002-01-10;10406 ’ and ’ http:
//legis.senado.leg.br/legislacao/DetalhaSigen.action?id=552282 ’; it was re-
viewed and approved by the ’National Congress’; it was signed in Brasília, on January 10, 2002; 181st
year of Independence and 114th year of the Republic; it was signed by FERNANDO HENRIQUE CAR-
DOSO, Aloysio Nunes Ferreira Filho; it was officially published by the ’National Press’, with the original
and official publication date being: 2002-01-11; it has alternative names: ’LEI-10406-2002-01-10’ and
’Civil Code (2002) (CC)’; its summary is: ’Institutes the Civil Code’. ")
Informative relationships between legal norms, such as regulation, succession, correlation, etc., do not
promote actions that alter the text of a norm, thus not generating temporal versions. Nevertheless, they are
important in the context of semantic retrieval of legal information. Therefore, we could also generate Text
Units with the textual descriptions of these relationships.
Considering the legal norm ’CONSTITUIÇÃO DO BRASIL (1967)’ (Constitution of Brazil (1967)), the
following are the Text Units23for its relationships presented in the graph of Figure 15.
Figure 15: Example representation of inter-normative relationships (succession) between different normative
acts in the knowledge graph.
A norma jurídica ’CONSTITUIÇÃO DO BRASIL (1967)’ sucedeu a norma jurídica ’CONSTITUIÇÃO
DOS ESTADOS UNIDOS DO BRASIL (1946)’
(The legal norm ’Constitution of Brazil (1967)’ succeeded the legal norm ’Constitution of the United
States of Brazil (1946)’ .)
A norma jurídica ’Constituição da República Federativa do Brasil’ sucedeu a norma jurídica ’CONSTI-
TUIÇÃO DO BRASIL (1967)”
(The legal norm ’Constitution of the Federative Republic of Brazil’ succeeded the legal norm ’Constitution
of Brazil (1967)’ )
Normas Jurídicas (Sigen)” (Legal Norm Management System) of the Federal Senate of Brazil. This metadata can also be viewed
publicly, in structured form, through the Normas.leg.br portal, as in the example: https://normas.leg.br/?urn=urn:
lex:br:federal:lei:2002-01-10;10406 . Accessed on 2025-03-22.
23It is not necessary to generate a Text Unit for the other side of the relation, since an LLM and a RAG retrieval already semantically
understand this reversal ("succeeded” vs “was succeeded by", for example).
16

This approach aligns with the concept of “ Multi-vector embeddings ” [15], which proposes using multiple
vectors to represent information at different levels of detail. Instead of generating a single, complete Text
Unit for an entity encompassing all its properties and relationships, this work proposes generating one Text
Unit for the entity’s intrinsic properties (general metadata), as exemplified above, and a separate Text Unit for
each of its relationships.
Thus, instead of having just one vector/point in the vector space representing an entity, we will have
multiple vectors/points associated with it. Each vector will capture/encode a different aspect of the information
related to the entity (potentially viewed as a different " lens" through which the entity is represented), thus
offering a more contextualized and granular view of the entity, which can lead to significant improvements in
semantic retrieval by proximity.
It is also proposed to generate Text Units for Component entities. For example:
"O dispositivo ’Art. 5 º, caput, XII –’ da ’Constituição da República Federativa do Brasil’ é
correlato da norma jurídica ’Lei nº 14.063 de 23/09/2020’."
("The provision ’Art. 5 º, caput, XII -’ of the ’Constitution of the Federative Republic of Brazil’ is
correlated with the legal norm ’Law no. 14,063 of 2020-09-23’. ")
These metadata Text Units about the norm, or component, can be generated externally (from other
information sources) and inserted directly into the graph, or be generated automatically from the properties
and/or relationships of the entity in the graph24- although the original metadata might be structured in the
graph, converting them to textual format allows their uniform integration and processing within the Graph
RAG workflow.
The metadata Text Unit for a norm or component will have its own embedding generated and will
also be subject to semantic retrieval, like the content Text Units derived from versions and alteration
actions. Consequently, vector search will be able to retrieve information about the metadata (properties and
relationships) of a legal norm (or a component), even though this information is generally not present in the
text of the norm itself.
We can see these Text Units derived from metadata, additional to those derived from the texts of the
norms/components, in Figure 16. The insertion of this type of Text Unit brings additional information to the
Knowledge Graph and, consequently, to the vector database of contexts.
Figure 16: Knowledge graph illustrating the Text Units derived from both the texts of the norms and their
metadata.
24As these properties (publication date, alternative names, level in the legal system, etc.) and relationships (correlation, succession,
etc.) generally cannot be directly derived from the texts of the legal norms, they must be populated in the graph either manually or
through semi-automatic extraction processes using other information sources.
17

3.6 Hierarchical Grouping as Communities
The grouping components of legal norms function in practice as hierarchical levels for thematic grouping,
chosen as the systematizing criterion for the articulation. As observed in the example of Figure 4, Art. 5 º
(Article 5) is grouped under “CAPÍTULO I – DOS DIREITOS E DEVERES INDIVIDUAIS E COLE-
TIVOS” (CHAPTER I – OF INDIVIDUAL AND COLLECTIVE RIGHTS AND DUTIES), which in turn
is grouped under “TÍTULO II - DOS DIREITOS E GARANTIAS FUNDAMENTAIS” (TITLE II - OF
FUNDAMENTAL RIGHTS AND GUARANTEES). These groupings then ascend in level until reaching the
Norm (as a Whole), representing natural hierarchical and thematic structures within legal norms. This directly
aligns with the concept of communities in a graph proposed by the Graph RAG approach – in this approach,
“communities ” refer to groups of nodes in the graph that are more densely interconnected among themselves
than with the rest of the graph, representing thematic or conceptual clusters. Therefore, the hierarchical
structure of legal norms can be viewed as an intrinsic form of organization into thematic communities.
Legal norms, in general, are classified into one or more themes25, which can be considered intertextual,
or inter-norm, aggregators, transversally gathering norms on cohesive subjects, as exemplified in Figure 17.
In this figure, the Norm (as a whole) are represented by circles in green, and the Themes , in orange.
Figure 17: Diagram illustrating the inter-norm aggregation by legal themes (orange), represented as higher-
level communities that group "Norms" (green) in the knowledge graph.
Components of legal norms could also be classified under one or more themes, as demonstrated in
Figure 18. This allows broadening the classification of the components beyond their groupers within the
hierarchical structure of the legal norm.
The themes would then be inserted as Entities/Nodes Theme in the Knowledge Graph, each having an
associated Text Unit with its description, which would be produced by a human domain expert, ensuring the
curation of the produced texts26.
25Brazilian federal norms are classified according to the “Sistema de Classificação de Normas Jurídicas do Congresso Nacional”
(System for Classification of Legal Norms of the National Congress), with examples of themes including: “Children and Adolescents”,
“Extraordinary Credit”, “Indigenous Population”, “Basic Sanitation”, ’General Social Security System’, etc.
26An alternative would be to use an LLM to generate the theme descriptions, producing summaries from the descriptions of the
sub-themes or, in the case of leaf themes, from the summaries of the associated norms. However, this approach has disadvantages:
besides requiring intensive processing and high costs (considering that a theme can cover hundreds of norms), the summary’s
effectiveness would heavily depend on the quality of the algorithms and models used.
18

Figure 18: Diagram illustrating associations between components and legal themes.
3.7 Retrieval considering Structure
Another advantage of using a graph to represent structural entities, reflecting the hierarchical structure of legal
texts, is that the user could choose the scope of their query (level of aggregation), for example, by defining
that they wish to query about a specific theme (which would encompass all its sub-themes and associated
norms, with their components and versions), or about a specific title of a norm (which would encompass
all its subdivisions and versions), or about a specific article (which would encompass its subdivisions and
versions), or about a specific version of a norm or a component (which would encompass its child versions),
as presented in Figure 19.
Figure 19: Diagram illustrating the possibility of retrieving Text Units considering the hierarchical structure
of the knowledge graph
This choice would define a filter on the hierarchical index of the Text Units (and vectors/embeddings)
to be retrieved, making the context retrieval for the response more focused. The hierarchical structure of
19

the graph would then allow for more precise semantic navigation between different levels of detail. This
“semantic navigation ” manifests in the user’s ability to direct the search to a specific level of the legal
hierarchy (theme, title, chapter, article, etc.), filtering the search space and focusing retrieval on the Text
Units most relevant to their interest at that level of detail. Instead of a generic search across the entire graph,
the user can explore legal knowledge in a more structured and hierarchy-guided manner.
4 Conclusion and Future Work
Aiming to enhance the analysis and comprehension of legal norms in the era of Artificial Intelligence,
this article presented a detailed proposal for the construction of a legal knowledge graph, enriched with
contextually relevant Text Units. The central objective was to significantly expand the scope of queries
regarding legal norms, allowing Artificial Intelligence systems to address questions ranging from literal text
retrieval to complex inquiries requiring an understanding of the hierarchical structure, the intrinsic semantics,
the intra- and inter-relations and the temporal evolution of law.
Incorporating such structural and temporal knowledge is expected to significantly enhance the precision
of legal AI responses, thereby reducing contextual errors and the risk of citing incorrect versions of the law.
In summary, the main contributions of this work include:
•A detailed proposal for building a legal knowledge graph, integrating the predefined hierarchical
structure, temporal versioning, and cumulative and contextually enriched Text Units, with an initial
focus on structural entities.
•The modeling of Versions as Aggregations , reflecting hierarchical interdependence and the propagation
of changes over time, allowing for an accurate and efficient representation of normative evolution.
•The suggestion to incorporate Additional Context into versions, enriching the graph with descriptive
metadata about the history and lifecycle of norms, providing additional context during response
generation by the LLM.
•The proposal to generate specific Text Units for Alteration Actions as Text Units (e.g., repeal, text
amendment), making the norm-modifying events themselves and their effects semantically retrievable.
•The generation of Text Units from “ structured Metadata ” (intrinsic properties and relationships such
as succession or correlation), allowing non-explicitly textual information in norms to be included in
vector search and aligning with concepts like multi-vector embeddings.
•The perspective of Hierarchical Grouping as Communities , interpreting the hierarchical structure and
legal themes (curated by specialists) as thematic communities within the graph, contributing to more
targeted and semantic information retrieval strategies.
•The proposal of Retrieval considering Structure , enabling users to define the search scope and semanti-
cally navigate the legal hierarchy (whether by theme, title, article, version, etc.), refining the precision
and relevance of responses.
•A Graph RAG approach that seeks the lowest possible cost for graph creation by prioritizing pre-
existing or curated hierarchical and thematic structures, avoiding computationally intensive techniques
such as automatic community detection and the construction of complex summaries by LLMs for this
purpose.
20

The main innovation lies in the adaptation and expansion of Graph RAG to the legal field, with a particular
focus on structural entities, considered essential for identifying robust legal grounds. Semantic segmentation
guided by the hierarchical structure, combined with the representation of cross-references, versioning and
grouping into thematic communities, provides a natural and rich contextual basis for legal RAG strategies.
The proposal to use the hierarchy of communities and themes designed by domain experts reflects the original
intent of legislators and ensures informational curation, an essential aspect for the reliability of AI systems in
law, in addition to minimizing the need for complex and costly graph construction processes.
For future work, several research possibilities open up:
•It is fundamental to empirically validate the effectiveness of the proposed Graph RAG approach
through experiments with legal norm datasets and rigorous evaluations of the quality and precision of
the responses generated by LLMs.
•Although the proposal of this work significantly advances the representation of normative complexity,
it primarily focused on modeling the hierarchical structure and temporality. The deep extraction and
semantic integration of Content Entities (people, organizations, legal concepts etc.) and their complex
interrelationships [5] remain a challenge reserved for future investigations.
•Investigating the use of Artificial Intelligence techniques, such as language models and clustering
algorithms, to support human curation in detecting thematic communities and suggesting hierarchical
structures represents another promising direction.
•Exploring the practical application of legal Graph RAG in legal decision support systems, legislative
impact analysis tools, and platforms for facilitated access to legal information for legal professionals
and citizens can demonstrate the concrete value and feasibility of the approach in real-world and
diverse scenarios.
•The proposed hierarchical structure for legal norms aligns with the Placing Graph approach of Bigraphs
[18], which organizes entities into contextual hierarchies (e.g., temporal, thematic). Furthermore,
the Linking Graph approach (hypergraph) could enrich the modeling by allowing multidimensional
associations between components of norms, such as cross-references between articles or case law.
As future work, it is proposed to explore the integration of Bigraphs to unify the hierarchical and
associative modeling of legal norms. This approach could enhance the representation of spatio-
temporal relationships (e.g., effectiveness of norms in different jurisdictions) and allow dynamic
memory operations, such as automatic consolidation of versions or the identification of normative
conflicts via semantic link analysis.
•Investigating the applicability and potential adaptation of the proposed Graph RAG approach beyond
legal norms to other fundamental types of legal documents, such as jurisprudence (case law) and
contracts. While the current focus leverages the predefined hierarchy of norms, exploring how
this framework could be extended or modified to handle the different structural conventions (e.g.,
case structure, contract clauses) and temporal dynamics (e.g., precedential relationships, contract
amendments) inherent in these other domains would be a valuable direction. This analysis would help
clarify the approach’s scope of generality and its limitations, further contextualizing its contribution
within the broader field of Legal AI.
Ultimately, by integrating structure, cross-references, temporality, and semantics into a legal knowledge
graph, this work offers a significant step towards more intelligent, transparent, and effective Artificial Intelli-
gence systems for the complex and fundamental domain of law. By facilitating precise information retrieval
and making legislative analysis more accessible through a navigable and contextualized representation, this
21

approach has the potential to democratize access to legal information and empower professionals and citizens
in an increasingly regulated world.
References
[1]Vinx L. Hans Kelsen’s pure theory of law: legality and legitimacy. Oxford: Oxford University Press;
2007.
[2]Chen Z, Li D, Zhao X, Hu B, Zhang M. Temporal knowledge question answering via abstract reasoning
induction. arXiv preprint arXiv:2311.09149v2 [Preprint]. 2023 [cited 2024 Apr 23]. Available from:
https://arxiv.org/abs/2311.09149v2
[3]Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V , Goyal N, et al. Retrieval-augmented generation for
knowledge-intensive NLP tasks. Adv Neural Inf Process Syst. 2020;33:9459-74.
[4]Edge D, Trinh H, Cheng N, Bradley J, Chao A, Mody A, et al. From local to global: A graph rag
approach to query-focused summarization. arXiv preprint arXiv:2404.16130 [Preprint]. 2024 [cited
2024 Apr 23]. Available from: https://arxiv.org/abs/2404.16130
[5]Li J, Yang G, Xu C, Zhang Y , Li H, Huang Z, et al. Construction of legal knowledge graph based on
knowledge-enhanced large language models. Information. 2024;15(11):666.
[6]Kejriwal M. Knowledge graphs: A practical review of the research landscape. Information.
2022;13(4):161.
[7]Mikolov T, Chen K, Corrado G, Dean J. Efficient estimation of word representations in vector space.
arXiv preprint arXiv:1301.3781 [Preprint]. 2013 [cited 2024 Apr 23]. Available from: https://
arxiv.org/abs/1301.3781
[8]Dang DV , Le NT, Nguyen NT, Takasu A. Information retrieval from legal documents with ontology and
graph embeddings approach. In: Proceedings of the International Conference on Industrial, Engineering
and Other Applications of Applied Intelligent Systems; 2023. p. 300-12.
[9]Brazil. Complementary Law No. 95 of February 26, 1998. Provides for the consolidation, codifica-
tion, and drafting of laws. Diário Oficial da União. 1998 Feb 27 [cited 2024 Apr 23]. Available
from: https://normas.leg.br/?urn=urn:lex:br:federal:lei.complementar:
1998-02-26;95
[10] Roddick JF, Spiliopoulou M. A survey of temporal knowledge discovery paradigms and methods. IEEE
Trans Knowl Data Eng. 2002;14(4):750-67.
[11] He X, Tian Y , Sun Y , Chawla N, Laurent T, LeCun Y , et al. G-retriever: Retrieval-augmented generation
for textual graph understanding and question answering. Adv Neural Inf Process Syst. 2023;36:132876-
907.
[12] Lima JAO. Unlocking legal knowledge with multi-layered embedding-based retrieval. arXiv preprint
arXiv:2411.07739 [Preprint]. 2024 [cited 2024 Apr 23]. Available from: https://arxiv.org/
abs/2411.07739
[13] Libal T. Legal linguistic templates and the tension between legal knowledge representation and reasoning.
Front Artif Intell. 2023;6:1136263.
22

[14] Tang Y , Gu Y , Qu S, Cheng X. CaseGNN: Graph neural networks for legal case retrieval with text-
attributed graphs. In: Proceedings of the European Conference on Information Retrieval; 2024. p.
80-95.
[15] Reddy RG, Garg D, Oh A, Kumar S, Muresan S, Richardson F, et al. AGRaME: Any-granularity ranking
with multi-vector embeddings. arXiv preprint arXiv:2405.15028 [Preprint]. 2024 [cited 2024 Apr 23].
Available from: https://arxiv.org/abs/2405.15028
[16] Chalkidis I, Fergadiotis M, Malakasiotis P, Aletras N, Androutsopoulos I. LEGAL-BERT: The muppets
straight out of law school. arXiv preprint arXiv:2010.02559 [Preprint]. 2020 [cited 2024 Apr 23].
Available from: https://arxiv.org/abs/2010.02559
[17] Salton G. Automatic Text Processing: The Transformation, Analysis, and Retrieval of Information by
Computer. Boston: Addison-Wesley; 1989.
[18] Pavlyshyn V . Bigraphs for AI agent memory [Internet]. Medium. 2023 [cited 2024
Apr 27]. Available from: https://volodymyrpavlyshyn.medium.com/
bigraphs-for-ai-agent-memory-3bdae9809f4a
23