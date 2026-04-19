# Knowledge Graph RAG: Agentic Crawling and Graph Construction in Enterprise Documents

**Authors**: Koushik Chakraborty, Koyel Guha

**Published**: 2026-04-14 05:26:50

**PDF URL**: [https://arxiv.org/pdf/2604.14220v1](https://arxiv.org/pdf/2604.14220v1)

## Abstract
This research paper addresses the limitations of semantic search in complex enterprise document ecosystems. Traditional RAG pipelines often fail to capture hierarchical and interconnected information, leading to retrieval inaccuracies. We propose Agentic Knowledge Graphs featuring Recursive Crawling as a robust solution for navigating superseding logic and multi-hop references. Our benchmark evaluation using the Code of Federal Regulations (CFR) demonstrates that this Knowledge Graph-enhanced approach achieves a 70% accuracy improvement over standard vector-based RAG systems, providing exhaustive and precise answers for complex regulatory queries.

## Full Text


<!-- PDF content starts -->

Knowledge Graph RAG: Agentic Crawling and Graph
Construction in Enterprise Documents
Koushik Chakraborty and Koyel Guha
Google AI, Global Services Delivery
March 30, 2026
Abstract
This research paper addresses the limitations of semantic search in complex enterprise document
ecosystems. TraditionalRAG pipelinesoftenfailtocapturehierarchical andinterconnectedinformation,
leadingtoretrievalinaccuracies. WeproposeAgenticKnowledgeGraphsfeaturingRecursiveCrawling as
arobustsolutionfornavigatingsupersedinglogicandmulti-hopreferences. Ourbenchmarkevaluation
usingtheCodeofFederalRegulations(CFR)demonstratesthatthisKnowledgeGraph-enhancedapproach
achievesa70%accuracyimprovementoverstandardvector-basedRAGsystems,providingexhaustiveand
precise answers for complex regulatory queries.
1 Introduction
In massive enterprise document ecosystems—legal, construction, specifications—information is not flat. It
is deeply hierarchical and interconnected. A cohesive "answer" to a query is rarely found in a single text
chunk. Instead, it is scattered across a "Base Contract," multiple "Addendum," "Technical Manuals," and
"Amendments."StandardRAGpipelines,whichdependonsemanticsimilarity,arefundamentallyinadequate
in this domain. These systems retrieve text based on the most similar content rather than the referenced
content. Furthermore, they fail to account for the temporal precedence of recent amendments over earlier
versions. Tosolvethis,wemustbuildMultiAgentWorkflowswithComplexGraphDataStorethatactas
graph traverser: crawling precise citations and constructing a knowledge graph of validity.
The core contributions of this paper are:
1.The Agentic Knowledge Graph (AKG) Framework: A hybrid RAG architecture that moves beyond
semantic retrieval to incorporate deterministic graph traversal.
2.TheTemporalGraphSchema: ModelingrelationshipsusingexplicitSUPERSEDESandREFERS_TO
edges to resolve versioning and multi-hop dependency conflicts.
3.The Recursive Reference Crawler: An autonomous agent implementation that programmatically
follows citation paths to assemble a complete, contextually valid answer.
4.Quantitative Validation: Demonstration of a 70% accuracy improvement over vector-only RAG
systems in the complex Code of Federal Regulations (CFR) domain.
This deep dive focuses on the two core engineering pillars of this architecture:
1.The Recursive Reference Crawler: Identifying and fetching linked nodes.
2.The Knowledge Graph: Structuring these nodes for intelligent retrieval.
1arXiv:2604.14220v1  [cs.IR]  14 Apr 2026

1.1 Sample documents:
1.1.1 Document 1: The Base Contract
Document ID:Base_Contract_Vol1
Date:2020-01-01
Description:The foundational construction agreement.
[Content] Section 4: Structural Materials & Methods Clause 4.2All permanent concrete structures,
including retaining walls and station boxes, shall utilizeGrade 25 Concrete, unless otherwise specified. For
curing times, refer to Clause 4.5.
Clause 4.5Concrete curing times shall be a minimum of 14 days under standard atmospheric conditions.
Clause 4.8Earth Retaining Stabilizing Structures (ERSS) shall be constructed using standard contiguous
boredpile(CBP)methods. Nospecificintegrationisrequiredbetweenthemainstationboxandsecondary
entrances unless directed by the Chief Engineer.
Section 9: Particular Specifications Clause 9.3.1General boundary walls shall have a minimum
thickness of 600mm.
1.1.2 Document 2: Amendment 01
Document ID:Amendment_01_Vol2
Date:2022-06-15
Description:First major revision due to updated geological surveys.
[Content] Item 1: Revision to MaterialsDelete Clause 4.2 of the Base Contract (Base_Contract_Vol1)
initsentiretyandreplaceitwiththefollowing: RevisedClause4.2: Allpermanentundergroundconcrete
structures shall utilize Grade 30 Waterproof Concrete.
Item 2: Revision to ERSS MethodsAmend Clause 4.8 of the Base Contract. Add the following
subsection: Clause4.8(a): Forhigh-water-tableareas,theERSSmethodmustdeviatefromstandardCBP.
Refer to the updated requirements in Particular Specification Clause 9.3.5.
Item3: NewSpecificationsAddnewClause9.3.5: Wherewatertablesarewithin2metersofthesurface,
secant pile walls shall be used in accordance with environmental safety guidelines.
1.1.3 Document 3: Tender Addendum No. 03
Document ID:Tender_Addendum_03_Vol1to6
Date:2024-10-10
Description:Final critical updates right before tender submission. Incorporates user-clarification requests.
[Content] Clarification No. 12: Concrete GradesFurther to the revisions in Amendment_01_Vol2, delete
the revised Clause 4.2 and replace with the following: Final Clause 4.2: Due to new structural loads, all
permanentstationboxstructuresmustutilizeGrade40Concrete(HighStrength). Retainingwallsnotattached
to the station box may remain at Grade 30.
2

ClarificationNo. 45: StationBoxERSS&EntrancesDeleteClause4.8(a)asintroducedinAmend-
ment_01_Vol2. AddnewClause4.8(g)totheMainText: Clause4.8(g): Diaphragmwallsshallbeadoptedfor
all permanent Earth Retaining Stabilizing Structures (ERSS) of the station box. For the associated structural
requirements, refer to Clause 9.3.10.5 of the Particular Specification.
ClarificationNo. 46: IntegrationofEntrancesAddnewClause9.3.10.5totheParticularSpecification:
Clause9.3.10.5: Diaphragmwallthicknessforthestationboxshallbeaminimumof1200mm. Entrances
1 & 3 are integrated with the structure of the station box and shall be treated as part of the station box for
structuralpurposes. ForacleardemarcationsketchofboundariesbetweenstationboxstructuresandEntrance
structures in plan & section, refer toDrawing_17.3.1_Demarcation_Plan.
1.2 Why RAG on these documents are tricky
StandardRAG pipelinesrelyon CosineSimilarityto measuresemanticrelevance betweenaquery vectorA
and a document chunk vectorB:
cos(𝜃)=A·B
∥A∥∥B∥(1)
Whileeffectiveforthematicmatches,thismetricisfundamentallyincapableofdiscerningthetemporal
validity or structural context of the information. The graph-based approach presented here shifts the retrieval
foundation from this probabilistic measure to a deterministic, set-theoretic traversal logic.
1.2.1 Example 1: The Cascading Amendment Problem
User Query:"What grade of concrete must I use for the permanent station box?"
TheCorrectAnswer:Grade40Concrete(HighStrength). Retainingwallsnotattachedtothestation
box may remain at Grade 30.
Why this is correct (and the complexity behind it):To arrive at this correct answer, a human reader (or
a system) must trace a multi-year chain of deletions and replacements across three separate documents:
1. They must first findClause 4.2in the Base Contract (2020), which statesGrade 25should be used.
2.Theymustthenrealizethattwoyearslater,Amendment01(2022)legallydeletedtheoriginalClause
4.2 and replaced it with a requirement forGrade 30.
3.Finally,theymustdiscoverthatrightbeforesubmission,TenderAddendum03(2024)explicitlydeleted
the revision from Amendment 01, establishingGrade 40as the final, binding requirement for the
station box.
The Standard RAG Failure:A standard semantic search engine will simply return all three text chunks
because they all highly match the keywords "grade of concrete permanent station box". It has no way
of knowing that the 2024 document invalidates the 2022 and 2020 documents, practically guaranteeing a
hallucinated or conflicting answer.
3

1.2.2 Example 2: The Multi-Hop Breadcrumb Trail
UserQuery:"HowshouldweconstructtheERSSforthestationbox,andwherecanIfindtheexactboundary
measurements for Entrances 1 and 3?"
TheCorrectAnswer:TheERSSforthestationboxmustbeconstructedusingDiaphragmwallswith
a minimum thickness of 1200mm. Because Entrances 1 & 3 are integrated with the station box, the exact
boundary measurements and demarcation lines can be found inDrawing_17.3.1_Demarcation_Plan.
Why this is correct (and the complexity behind it):Answering this question requires navigating a
convoluted, multi-step breadcrumb trail where no single document or clause contains the complete answer:
1.First, the reader must bypass outdated ERSS rules in the Base Contract and Amendment 01, eventually
finding the currently validClause 4.8(g)in Tender Addendum 03.
2.Clause 4.8(g) reveals the method (Diaphragm walls) but does not answer the second part of the user’s
question about entrances. Instead, it provides an instruction: "refer to Clause 9.3.10.5".
3.ThereadermustthenhuntdownClause9.3.10.5. Thisclauseexplainsthattheentrancesarestructurally
integrated with the station box, but still doesn’t give the exact measurements.
4. Instead, Clause 9.3.10.5 provides a final instruction: look atDrawing 17.3.1.
The Standard RAG Failure:If a system only retrieves the most "semantically similar" paragraphs to
theuser’sprompt,itwilllikelyjustpullClause4.8(g). Itwillstopthere,completelymissingthestructural
requirementsin9.3.10.5andfailingtotelltheuserwhichdrawingholdstheactualmeasurements. Thesystem
mustknowhowto"read"aninstructioninthetextandphysicallyjumptothenextlocationtoassemblethe
full picture.
2 Related Work
The field of Retrieval-Augmented Generation (RAG) is predominantly anchored in vector space models,
leveragingsemanticsimilarityforchunkretrieval. Thispaperdirectlyaddressesthelimitationsofstandard
RAG by introducing structural awareness.
2.1 Limitations of Standard Vector RAG
TraditionalRAGsystems,whichrelysolelyonvector-basedretrieval(suchasthoseusingCosineSimilarity),
treat documents as an amorphous collection of text chunks. This architecture fails when:
1.TemporalConflicts:Asemanticallydense,butoutdated,clauseisretrievedoveraconcise,superseding
amendment (the "temporal hallucination" problem).
2.Contextual Fragmentation:A complete answer requires explicitly following a chain of citations
(multi-hop traversal), which vector search cannot perform deterministically.
2.2 Advancement over GraphRAG Frameworks
Our proposed framework—which focuses onTemporal-Hierarchical RAG—advances existing Graph RAG
conceptsbyintroducingspecifictemporalandlogicaledgetypes. WhilemanyGraphRAGsolutionsmodel
general relationships, our system prioritizes the engineering ofSUPERSEDESandREFERS_TOedges to
enforce document validity and structural dependency, which are crucial for complex, regulated domains. The
core innovation lies in using the Knowledge Graph not just for enriched vector indexing, but as a primary,
deterministic traversal mechanism guided by an autonomous agent (the Recursive Reference Crawler).
4

3 System Architecture and Methodology
TheReferenceCrawlerisanautonomousagentdesignedtonavigatethe"SupersedingLogic"ofenterprise
documentation. Unlikeaprobabilisticsearchengine,itfollowsexplicitinstructionsfoundinthetext(e.g.,
"Delete Clause 12.1 and see Amendment 3").
It operates on a cycle ofExtraction -> Traversal -> Aggregation.
3.1 Core Architecture
The crawler uses a queue-based traversal algorithm (BFS/DFS) to follow citations (Edges) from one Clause
(Node)to another. It usesaspecializedLLM calltoextract structuredcitationsfromunstructured legaltext.
Figure 1: Building Knowledge Graph based on the documents
5

Figure 2: Query Processing & Retrieval
3.2 Recursive Crawler Implementation
Please refer to Appendix for further details.
4 Building the Knowledge Graph
While the crawler navigates primarily at query-time, we can optimize global retrieval by pre-building a
Knowledge Graph. This graph explicitly models the complex relationships between documents, allowing us
to query for "The Valid Clause" rather than just "The Text."
4.1 Graph Schema
•Nodes:Documents/Clauses (Attributes: timestamp, version, text_content)
•Edges:
1.SUPERSEDES:Directional edge from a newer clause to an older one (superseding).
2.REFERS_TO:Citational edge indicating dependency.
3.CONTAINS:Hierarchical edge (Document contains Clause).
4.2 Graph Construction
Please refer to Appendix for further details.
Hereareexamplesofhowuserquestionswillinteractwiththegraphbuiltfromthethreedocumentsgiven
in section1.1 Sample Documents:
6

4.2.1 Query Example 1: Testing Temporal Precedence (The "Supersedes" Edge)
User Query:"What grade of concrete must I use for the permanent station box?"
Correct Answer:should returnGrade 40 as the legally valid answer, ignoring Grades 25 and 30.
Expected Graph Traversal:
1. Standard Search hitsBase_Contract_Vol1::Clause 4.2(Says Grade 25).
2. Graph checks forSUPERSEDESedges targeting this node.
3. Redirects toAmendment_01_Vol2::Item 1(Says Grade 30).
4. Graph checks forSUPERSEDESedges targeting this node.
5. Redirects toTender_Addendum_03_Vol1to6::Clarification No. 12(Says Grade 40).
6.Crawler Output:Returns Grade 40 as the legally valid answer, ignoring Grades 25 and 30.
4.2.2 Query Example 2: Testing Recursive Traversal (The "Refers_To" Edge)
User Query:"How should we construct the ERSS for the station box, and where can I find the exact
boundary measurements for Entrances 1 and 3?"
Expected Graph Traversal:
1.StandardSearchhitsTender_Addendum_03_Vol1to6::Clause4.8(g)(IdentifiesDiaphragmwallsare
needed).
2.TheCrawler’sLLMextractsthereference:target_document: Tender_Addendum_03,target_section:
9.3.10.5.
3.Crawler hops toClause 9.3.10.5(Identifies 1200mm thickness and that Entrances 1 & 3 are integrated).
4.TheCrawler’sLLMextractsthenextreference:target_document: Drawing_17.3.1_Demarcation_Plan.
5.Crawler Output:The LLM aggregates all three hops into a single context window. It answers
theERSSmethod(Diaphragmwalls),thethickness(1200mm),notestheintegrationofEntrances1
& 3, and successfully directs the user toDrawing 17.3.1for the exact boundary plan—a complete,
hallucination-free answer that a standard semantic search would miss.
5 Benchmark Performance Evaluation
5.1 Dataset Profile: The Code of Federal Regulations (CFR)
ThebenchmarkutilizestheCodeofFederalRegulations(CFR),sourcedinstructuredXMLformatfrom
https://www.govinfo.gov/bulkdata/CFR. The CFR was selected as the primary corpus because its structural
complexity serves as a rigorous stress test for information retrieval systems. Specifically, it exhibits two
characteristics that highlight the limitations of flat vector stores compared to Knowledge Graphs (KGs):
•Inherent Hierarchy:The CFR follows a strictly nested taxonomy, descending fromTitle through
Chapter, Subchapter, Part, Subpart, and Section, down to individual Paragraphs.
•DenseCross-Referencing:Regulatorylanguageischaracterizedbyfrequentexplicitcitations(e.g.,
"pursuant to 261.14(a)(4)"), creating a complex web of interdependent legal authorities.
7

5.2 Methodology and Accuracy Results
To evaluate performance, we curated a gold-standard set of 20 complex regulatory questions. We com-
paredtheretrievalaccuracyofaKnowledgeGraph-enhancedapproachagainstastandardVector-based
Retrieval-Augmented Generation (RAG)baseline.
Accuracy Improvement
The Knowledge Graph approach demonstrated a70% improvement in accuracyover the standard RAG
baseline. WhiletheRAGsystemfailedtoprovideacompleteorcorrectresponsefor70%ofthequeries—often
due to incomplete context or retrieval "hallucinations"—the Knowledge Graph system successfully navigated
the nested structures to provide exhaustive and precise answers in every instance.
Table 1: Retrieval Performance Comparison
Metric Vector RAG Approach Knowledge Graph Approach
Correct/Complete Answers 5 (25%)19 (95%)
Incomplete/Inaccurate 8 (40%) 1 (5%)
Refusals/No Answer 7 (35%) 0 (0%)
Total Questions 20 20
Figure 3: Performance Comparison: Knowledge Graph Vs. RAG Approach
8

Figure 4: Average Faithfulness Vs. Approach
We utilized the Code of Federal Regulations (CFR) as a stress test for our architecture due to its inherent
hierarchy and dense cross-referencing. Accuracy was measured using theOverlap Coefficientformula:
Score=|KeywordsAnswer∩KeywordsSource|
|KeywordsAnswer|(2)
•Numerator:Count of unique, meaningful words found in both the Answer and the Source.
•Denominator:Total count of unique, meaningful words in the Answer.
•Result:A score between0 and 1 (or 0% and 100%).
Example:
5.3 Comparative Analysis: KG vs. Vector RAG
TheexperimentaldataidentifiesthreecriticaldimensionswhereKnowledgeGraphsoutperformtraditional
semantic search:
9

Question Accordingto561.2from2002,whatconstitutesan’account’,
and what are some of the specific, distinct types of accounts
that fall under this definition?
Golden Snippet 561.2 Account. "The term account means any savings ac-
count, demand account, certificate account, tax and loan
account, note account, United States Treasury general ac-
count or United States Treasury time deposit-open account,
whetherintheformofadepositorashare,heldbyanaccount
holder in a savings association."
Knowledge Graph Approach Answer Based on the 2002 version of 561.2, an ’account’ is defined
as anysavings account, demand account, certificate ac-
count, tax and loan account, note account, United States
Treasurygeneralaccount,orUnitedStatesTreasurytime
deposit-openaccount,heldbyanaccountholderinasavings
association.Specific examples of account types mentioned
within this definition includesavings accounts, demand ac-
counts, and certificate accounts.
KG Score 57.14%
RAG Approach Answer Accordingto561.2from2002,an"account"meansanyaccount
established to provide benefits, a pension, a retirement plan, or
compensationofanykind. Specific,distincttypesofaccounts
thatfallunderthisdefinitioninclude: Benefitaccounts,Pension
accounts, Retirement plan accounts, Compensation accounts.
RAG Score 9.52%
Table 2: Comparative Sample Answer Quality for CFR Question 561.2
Feature Vector RAG Limitation Knowledge Graph Advantage
Structural Awareness Treats citations as mere text strings;
lacks the logic to recognize them as
pointers to external authorities.Modelscitationsasexplicitedgesbe-
tween nodes, allowing deterministic
traversal of the regulatory web.
Temporal Precision Struggles to differentiate between se-
manticallysimilarchunksfromdiffer-
ent years, leading to versioning errors.Encodes temporal attributes (e.g.,
has_version orvalid_in_year )as
metadata or nodes to isolate specific
timeframes.
Logical Relationships Relies on semantic proximity, which
often fails to capture "logical leaps"
like exemptions or negations.Explicitly maps relational logic (e.g.,
IS_CONSTRAINED_BY orDEFINES),
ensuring the model respects the hierar-
chy of definitions.
Table 3: Comparative Analysis: Vector RAG Limitations vs. Knowledge Graph Advantages
10

StructuralandHierarchicalNavigationInastandardRAGworkflow,thesystemmayretrieveaspecific
section but fail to capture the parent definitions or child clauses that govern it. Conversely, the KG treats
theserelationshipsasnavigablepaths. Bymodelingthehierarchy,thesystemensuresthatifa"Section"is
retrieved, its governing "Part" and "Subpart" context is preserved.
HandlingSemanticSimilarityvs. ExplicitLogicVector-basedmodelsprioritizehowsimilarwords
"feel"inahigh-dimensionalspace. However,regulatorycompliancerequiresidentifyingwhenoneruleis
superseded by another. The Knowledge Graph excels here by utilizingtyped relationships, such as linking a
specific "Savings Account" node to a broader "Account Definition" node, ensuring the response reflects the
precise legal taxonomy rather than just a linguistic match.
Addendum: NoteonTemporalPrecedenceThetemporalaspectofdocumentvalidity—determining
whichdocumentisthecurrentsourceoftruth—isimplicitlyhandledbythegraphstructureabove. Bytreating
"Time" as a property of the Node (timestamp) and "Validity" as a property of the Edge (Supersedes), we
solve the temporal hallucination problem. The graph traversal logic (get_valid_clause) automatically respects
temporal precedence without strictly needing a separate "Temporal Search" engine. The graph topology
encodes time.
6 Conclusion
The integration of Agentic Knowledge Graphs and recursive traversal significantly enhances the reliability of
informationretrievalincomplexdomains. Bymodelingdocumentrelationshipsthroughexplicitedgesand
temporalmetadata,theproposedarchitectureovercomestheinherentlimitationsofflatvectorstores. The
observed 70% accuracy improvement confirms that structural awareness is critical for maintaining truth and
consistency in enterprise-scale knowledge management.
References
[1]P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T.
Rocktäschel, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."Advances
in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9459–9474, 2020.
[2] Google Cloud. "GoogleADK Documentation"https://google.github.io/adk-docs/agents/.
[3]Office of the Federal Register. "Code of Federal Regulations (CFR) Bulk Data."National Archives and
Records Administration, 2026. [Online]. Available:https://www.govinfo.gov/bulkdata/CFR.
[4]S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and J. Wu. "Unifying Large Language Models and
Knowledge Graphs: A Roadmap."IEEE Transactions on Knowledge and Data Engineering, 2024.
7 Appendix
7.1 Code Sample: Recursive Crawler Implementation
The following implementation demonstrates the crawler’s logic: analyzing a text, finding its outbound
references, and recursively fetching those targets to build a complete context.
11

1import time
2from typing import List , Set, Tuple
3from pydantic import BaseModel , Field
4import google.generativeai as genai
5
6# --- Data Structures ---
7class ClauseReference(BaseModel):
8target_document_name: str = Field(..., description="Name of the
referenced document")
9target_section_id: str = Field(..., description="The clause or section
number referenced")
10reasoning: str = Field(..., description="Why this reference is relevant
")
11
12class ExtractionResult(BaseModel):
13has_references: bool
14references: List[ClauseReference]
15
16# --- Core Crawler Logic ---
17def reference_crawler(start_doc: str, start_clause: str, initial_text: str,
max_depth: int = 3) -> str:
18"""
19Traverses document references starting from a root clause.
20Returns a consolidated context string containing the lineage of clauses
.
21"""
22# Queue stores: (Document , Clause , CurrentDepth)
23queue = [(start_doc , start_clause , 0)]
24visited: Set[str] = set()
25
26# Store the full context to return to the LLM
27aggregated_context = f"ROOT SOURCE: [{start_doc} - {start_clause}]\
nContent: {initial_text}\n\n"
28visited.add(f"{start_doc}::{start_clause}")
29
30while queue:
31current_doc , current_clause , depth = queue.pop(0)
32
33if depth >= max_depth:
34continue
35
36# 1. Fetch text (Simulated fetch from DB/Vector Store)
37text_content = fetch_clause_text(current_doc , current_clause)
38
39# 2. Extract references using a specialized minimal LLM call
40references = extract_references_with_llm(text_content)
41
42if references.has_references:
43aggregated_context += f"--- References found in {current_doc} {
current_clause} ---\n"
44
45for ref in references.references:
46node_key = f"{ref.target_document_name}::{ref.
12

target_section_id}"
47
48if node_key not in visited:
49visited.add(node_key)
50
51# Fetch the content of the referenced node
52ref_text = fetch_clause_text(ref.target_document_name ,
ref.target_section_id)
53aggregated_context += f">> REFERENCED: [{ref.
target_document_name} {ref.target_section_id}]\n"
54aggregated_context += f" Content: {ref_text}\n"
55
56# Add to queue to deeper traversal
57queue.append((ref.target_document_name , ref.
target_section_id , depth + 1))
58
59return aggregated_context
60
61def extract_references_with_llm(text: str) -> ExtractionResult:
62"""
63Uses a targeted prompt to extract citations from legal text.
64"""
65prompt = f"""
66Analyze the following text: "{text}"
67Identify any specific references to other documents or clauses.
68Return JSON matching the ExtractionResult schema.
69"""
70# Call your LLM here (e.g. Gemini 1.5 Flash for speed)
71return call_llm_with_schema(prompt , ExtractionResult)
Listing 1: Recursive Crawler Implementation
7.2 Code Sample: Graph Construction
This example uses networkx to build a graph that can resolve superseding logic dynamically.
1import networkx as nx
2from datetime import datetime
3
4class LegalKnowledgeGraph:
5def __init__(self):
6self.graph = nx.DiGraph()
7
8def add_clause_node(self , doc_id: str, clause_id: str, content: str,
date: str):
9node_id = f"{doc_id}::{clause_id}"
10self.graph.add_node(
11node_id ,
12content=content ,
13date=datetime.strptime(date , "%Y-%m-%d"),
14type="clause"
15)
16return node_id
17
13

18def add_superseding_relationship(self , newer_node_id: str,
older_node_id: str):
19"""
20Model that Newer Node replaces Older Node.
21"""
22self.graph.add_edge(newer_node_id , older_node_id , relation="
SUPERSEDES")
23
24def add_reference_relationship(self , source_node_id: str,
target_node_id: str):
25"""
26Model that Source refers to Target for information.
27"""
28self.graph.add_edge(source_node_id , target_node_id , relation="
REFERS_TO")
29
30def get_valid_clause(self , base_clause_node_id: str) -> str:
31"""
32Finds the most current version of a clause by traversing incoming
SUPERSEDES edges.
33"""
34current_node = base_clause_node_id
35
36# Traverse ’backwards’ along SUPERSEDES edges
37# (Find the node that claims to SUPERSEDE the current node)
38
39while True:
40# Find predecessors connected by a ’SUPERSEDES’ edge
41superseders = [
42n for n in self.graph.predecessors(current_node)
43if self.graph[n][current_node].get("relation") == "
SUPERSEDES"
44]
45
46if not superseders:
47break
48
49# If multiple superseders exist , pick the most recent one
50superseders.sort(key=lambda n: self.graph.nodes[n][’date’],
reverse=True)
51current_node = superseders[0]
52print(f" -> Redirected to superseding clause: {current_node}"
)
53
54return self.graph.nodes[current_node][’content’]
55
56# Usage Simulation
57kg = LegalKnowledgeGraph()
58
59# 1. Ingest Base Contract (Old)
60base_id = kg.add_clause_node("BaseContract", "12.1", "Grade 25 Concrete", "
2020-01-01")
61
62# 2. Ingest Amendment (New)
14

63amend_id = kg.add_clause_node("Amendment_07", "Item_5", "Grade 40 Concrete
(High Strength)", "2024-06-01")
64
65# 3. Define Relationship: Amendment 07 -> Supersedes -> Base Contract 12.1
66kg.add_superseding_relationship(amend_id , base_id)
67
68# 4. Query
69print("--- Querying Clause 12.1 ---")
70result = kg.get_valid_clause(base_id)
71print(f"Result: {result}")
72# Output: Grade 40 Concrete (High Strength)
Listing 2: Knowledge Graph Construction
15