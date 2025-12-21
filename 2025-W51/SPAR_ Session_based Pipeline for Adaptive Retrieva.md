# SPAR: Session-based Pipeline for Adaptive Retrieval on Legacy File Systems

**Authors**: Duy A. Nguyen, Hai H. Do, Minh Doan, Minh N. Do

**Published**: 2025-12-15 02:54:10

**PDF URL**: [https://arxiv.org/pdf/2512.12938v1](https://arxiv.org/pdf/2512.12938v1)

## Abstract
The ability to extract value from historical data is essential for enterprise decision-making. However, much of this information remains inaccessible within large legacy file systems that lack structured organization and semantic indexing, making retrieval and analysis inefficient and error-prone. We introduce SPAR (Session-based Pipeline for Adaptive Retrieval), a conceptual framework that integrates Large Language Models (LLMs) into a Retrieval-Augmented Generation (RAG) architecture specifically designed for legacy enterprise environments. Unlike conventional RAG pipelines, which require costly construction and maintenance of full-scale vector databases that mirror the entire file system, SPAR employs a lightweight two-stage process: a semantic Metadata Index is first created, after which session-specific vector databases are dynamically generated on demand. This design reduces computational overhead while improving transparency, controllability, and relevance in retrieval. We provide a theoretical complexity analysis comparing SPAR with standard LLM-based RAG pipelines, demonstrating its computational advantages. To validate the framework, we apply SPAR to a synthesized enterprise-scale file system containing a large corpus of biomedical literature, showing improvements in both retrieval effectiveness and downstream model accuracy. Finally, we discuss design trade-offs and outline open challenges for deploying SPAR across diverse enterprise settings.

## Full Text


<!-- PDF content starts -->

SPAR: SESSION-BASEDPIPELINE FORADAPTIVERETRIEVAL
ONLEGACYFILESYSTEMS
Duy A. Nguyen
Siebel school of Computing and Data Science
University of Illinois, Urbana-Champaign
duyan2@illinois.eduHai H. Do
School of Communication and Information Technology
Hanoi University of Science and Technology
haidh@illinois.edu
Minh Doan
Bioimaging Analytics
GlaxoSmithKline, Collegeville, PA, USA
minh.x.doan@gsk.comMinh N. Do
Electrical and Computer Engineering
University of Illinois, Urbana-Champaign
minhdo@illinois.edu
ABSTRACT
The ability to extract value from historical data is essential for enterprise decision-making. However,
much of this information remains inaccessible within large legacy file systems that lack structured
organization and semantic indexing, making retrieval and analysis inefficient and error-prone. We
introduceSPAR(Session-based Pipeline for Adaptive Retrieval), a conceptual framework that inte-
grates Large Language Models (LLMs) into a Retrieval-Augmented Generation (RAG) architecture
specifically designed for legacy enterprise environments. Unlike conventional RAG pipelines, which
require costly construction and maintenance of full-scale vector databases that mirror the entire file
system, SPAR employs a lightweight two-stage process: a semantic Metadata Index is first created,
after which session-specific vector databases are dynamically generated on demand. This design
reduces computational overhead while improving transparency, controllability, and relevance in
retrieval. We provide a theoretical complexity analysis comparing SPAR with standard LLM-based
RAG pipelines, demonstrating its computational advantages. To validate the framework, we apply
SPAR to a synthesized enterprise-scale file system containing a large corpus of biomedical literature,
showing improvements in both retrieval effectiveness and downstream model accuracy. Finally, we
discuss design trade-offs and outline open challenges for deploying SPAR across diverse enterprise
settings.
KeywordsLegacy file system·Retrieval-Augmented Generation (RAG)·Session-based RAG
1 Introduction
Context.The exponential growth of enterprise data has created an urgent need for robust systems capable of extracting,
managing, and leveraging information from heterogeneous file repositories. Legacy file systems, which frequently store
vast collections of historical data in unstructured or poorly structured formats, pose a persistent challenge in this regard
[1,2]. Such systems typically rely on rigid storage hierarchies and encompass diverse modalities, including unstructured
documents (e.g., PDFs, DOCX), semi-structured files (e.g., JSON, CSV), and specialized formats (e.g., medical images
in healthcare). The absence of semantic indexing and structured metadata further hinders their integration into modern
data processing and retrieval pipelines.
In response to these challenges, Retrieval-Augmented Generation (RAG) pipelines powered by Large Language Models
(LLMs) have emerged as a promising solution [ 3,4,5,6,7]. As shown in Figure 1a, traditional RAG systems
employ a two-stage process that synergistically combines retrieval and generation to augment the capabilities of LLMs,
particularly for enterprise dedicated tasks. The first stage involves constructing a comprehensive vector database,
which serves as the central knowledge base for the system. This database is created by encoding data from large filearXiv:2512.12938v1  [cs.IR]  15 Dec 2025

LLM-based Retrieval-Augmented Generation on Legacy File Systems
Legacy File SystemsVector DatabaseConstructUpdate
Query Command
Retriever (e.g.Similarity Search)QueryQueryReturn top-k documents
Query with additional context
LLM Agent
(a) Ordinary LLM-based RAG
Metadata IndicesReturn relevant documents
Query Command
Retriever (e.g.Similarity Search)QueryLegacy File SystemsConstructUpdate
File Retrieval CommandVector DatabaseQueryParserQueryReturn keywords & metadata
Query with additional context
LLM AgentReturn top-k documents (b) SPAR
Figure 1: Overview of ordinary LLM-based RAG pipeline versus proposed SPAR pipeline.
systems into dense vector representations using advanced embedding models. These embeddings effectively capture
the semantic relationships within the data, enabling meaningful and efficient retrieval. In the second stage, the system
focuses on retrieval and generation. When a user submits a query, the system retrieves relevant vectors from the
database through similarity search mechanisms, ranking and fetching data points that closely align with the query’s
semantic meaning. These retrieved results are then passed as context to the LLM, enabling it to generate responses
that are accurate, contextually grounded, and aligned with the user’s needs. This retrieval step narrows the scope of
information the LLM processes and tailored with enterprise context, enhancing efficiency while mitigating issues such
as hallucination.
Despite its utility, this approach is not without significant limitations:
1.Resource Intensive: Constructing and maintaining a large-scale vector database tightly mirroring an existing
file system demands substantial computational and storage resources, which can be a barrier for resource-
constrained systems.
2.Dynamic Syncing Issues: Continuous synchronization between the file system and the vector database
introduces overheads and risks of inconsistencies, especially as datasets grow in size and complexity.
3.Scalability Constraints: Querying large vector datasets often results in inefficiencies, reduced accuracy, and
limited user control over the relevance and specificity of retrieved outputs.
These inherent drawbacks highlight the need for more adaptive and resource-efficient solutions, motivating the
exploration of innovative approaches to RAG pipeline design.
Our method.To address these limitations, we present SPAR (Session-based Pipeline for Adaptive Retrieval), a novel
and modular RAG pipeline tailored specifically for legacy enterprise file systems (Figure 1b). This approach departs
from conventional RAG methods [ 3,4,5] by eliminating the need for extensive, static vector databases that mirror
large file systems. Instead, SPAR introduces a common strategy for constructing lightweight metadata indices that
encapsulate file metadata and enterprise-defined tags. These indices are not only highly adaptable to diverse enterprise
data and domains but also significantly more efficient and manageable than traditional methods. Building on these
indices, SPAR employs a session-based approach in which each query session begins with a specialized prompt that
leverages the metadata to retrieve relevant files. This prompt dynamically generates a temporary vector database,
customized to meet the session’s unique requirements. The resulting database acts as a flexible and persistent knowledge
base for RAG operations throughout the session, with on-demand updates enabled via user prompts to ensure relevance
and accuracy.
By combining the efficiency of metadata indices with the flexibility of on-the-fly vector database construction,SPAR
addresses key limitations of traditional RAG systems:
•Efficiency and Scalability: The use of lightweight metadata indices and session-specific vector databases
introduces minimal resource overhead, making SPAR adaptable to enterprise workloads of varying scale.
•Seamless Synchronization: Metadata indices can be efficiently updated in real time to reflect changes in the
underlying file system. Meanwhile, session-based vector databases are decoupled from persistent storage,
eliminating the need for continuous synchronization.
2

LLM-based Retrieval-Augmented Generation on Legacy File Systems
•Accuracy and Usability: Enterprise-defined tags enable precise filtering of relevant files during retrieval. The
smaller scope of session-specific vector databases also enhances retrieval precision and interpretability.
Through a comprehensive theoretical analysis encompassing construction, maintenance, and synchronization complexity,
we demonstrate that SPAR provides a more efficient and adaptable solution for applying RAG to legacy file systems
than conventional approaches. Beyond theory, we validate the framework on a synthesized enterprise-scale file system
containing a large corpus of biomedical literature. Compared with a standard LLM-based RAG pipeline built on a global
vector database, SPAR achieves higher retrieval accuracy and improved downstream model performance, underscoring
its potential for practical enterprise deployment.
The remainder of the paper is structured as follows. Section 2 reviews conventional LLM-based RAG pipelines, their
extensions, and their application in enterprise settings. Section 3 details the design principles and key components of
the proposed SPAR framework. Section 4 provides a complexity-based comparison between SPAR and traditional
RAG pipelines. Section 5 demonstrates SPAR in enterprise use cases, while Section 6 discusses design trade-offs,
deployment considerations, and open research challenges.
2 Preliminary: LLM-based RAG pipelines
The ordinary RAG framework [ 8,9,10] follows a “retrieve-then-generate” paradigm. Enterprise data from di-
verse sources—such as financial reports, customer support logs, technical documentation, and other domain-specific
repositories—is first ingested and preprocessed. To meet the input constraints of LLMs, the data is segmented into
smaller chunks and transformed into dense vector representations using embedding models such as BERT [ 11] or
Sentence Transformers [ 12,13]. These vectors, which capture semantic relationships, are stored in vector databases like
FAISS or Pinecone [ 14], optimized for large-scale enterprise workloads. When a user submits a query—for example, to
generate a report, analyze business trends, or draft a customer response—the system encodes the query into a vector
representation. This query vector is matched against stored embeddings to retrieve the most relevant information.
The retrieved content is then passed to the generative stage, where the LLM synthesizes a response that integrates
enterprise-specific evidence with its inherent generative capabilities. In this way, the final output is not only coherent
and fluent but also grounded in the enterprise’s operational context.
RAG is frequently highlighted for its ability to mitigate hallucinations in LLMs by grounding them in external evidence.
Within enterprise contexts, this grounding capability opens new opportunities and challenges alike [ 15,16,17,18]. On
the one hand, RAG pipelines enable enterprises to incorporate diverse and unstructured data—ranging from documents
and spreadsheets to knowledge repositories—into LLM-driven workflows. This integration allows organizations to
transform raw data into actionable insights, producing outputs that are both operationally meaningful and tailored to
enterprise-specific needs. Moreover, RAG-based systems can dynamically incorporate updates to the knowledge base,
ensuring responsiveness to evolving contexts. For instance, new records from ongoing projects or real-time customer
interactions can be integrated into the retrieval process without the need to retrain the underlying model.
On the other hand, ordinary RAG pipelines encounter several limitations when deployed at enterprise scale. First, the
construction and maintenance of large vector databases is computationally expensive, particularly for heterogeneous
file systems spanning decades of historical data. Second, exclusive reliance on similarity-based retrieval introduces
usability concerns: users often lack control over the selection of retrieved content and must carefully craft queries to
obtain relevant results. This misalignment between query formulations and document embeddings can lead to irrelevant
or noisy retrievals. Third, as the volume of enterprise data continues to grow, system scalability is strained, reducing
retrieval efficiency and degrading output precision. Consider, for example, a financial enterprise maintaining decades of
annual and quarterly reports in mixed formats. When an analyst queries the system for“summarize revenue growth
for Q2 2019”, a conventional RAG pipeline may return documents discussing“quarterly revenue trends”or“Q2
performance”but from different years. Because such pipelines typically lack metadata-aware filtering for temporal
constraints, the analyst must manually refine queries or sift through irrelevant results, leading to inefficiency and
possible misinterpretation. This example highlights how seemingly minor gaps in retrieval control can have significant
consequences in high-stakes enterprise settings.
Extensions to the basic RAG framework have been developed with enterprise use cases in mind, aiming to address
specific shortcomings of ordinary systems. To enhance retrieval precision in complex environments, researchers have
explored fine-grained indexing [ 19,20,21] (e.g., FKE, which retrieves sentence-level knowledge via chain-of-thought
prompting [ 21]), query rewriting strategies [ 22,23,24] (e.g., Rewrite–Retrieve–Read [ 24]), and metadata-driven
reranking [ 25]. Further improvements include hybrid retrieval systems that combine dense and sparse representations
(e.g., Blended RAG [ 26]), as well as diagnostic and benchmarking tools such as RAGChecker and RankArena [ 27,28]
for evaluating and refining pipeline performance. Despite these advances, such methods typically presuppose a
static, centralized vector database that mirrors the underlying file system. This assumption creates critical barriers in
3

LLM-based Retrieval-Augmented Generation on Legacy File Systems
Natural Language
API LayerbuildUpdateDtb()resetDtb()queryDtb()Query ParserExtract CommandExtract Metadata & KeywordsStructured QueryLegacy File Systems
Metadata Monitor ModuleMetadata DatabaseBuild/UpdateTemporary Vector Database
QueryUpdate Status
Add/Modify/DeleteRetrieveVector QueryText EncoderSimilarity SearchFetchDelete
Return
Top-k Vector Document
Figure 2: Detailed components of SPAR pipeline.
enterprise contexts: while indexing, query rewriting, and reranking can improve precision, they do not address the
heavy computational cost of constructing and continuously synchronizing a global vector database with dynamic and
evolving file systems.
To meet domain-specific needs, modular RAG variants have also been explored in areas such as legal document
analysis [ 29], customer support automation [ 30], and financial forecasting [ 31]. Although such systems can deliver
tailored functionality once the data has been ingested, they inherit the same scalability and maintenance bottlenecks,
limiting their applicability to heterogeneous enterprise repositories that often lack structured metadata and span multiple
formats accumulated over decades. Consequently, despite these extensions, existing RAG pipelines remain difficult to
operationalize in real-world legacy environments.
In contrast,SPARovercomes these barriers by decoupling retrieval from large-scale persistent vector stores. Its use of
lightweight metadata indices and session-based, on-demand vector databases directly addresses the synchronization,
efficiency, and scalability issues that current extensions leave unresolved. The next section introduces the design
principles and key components of SPAR, showing how this framework provides a practical and adaptive foundation for
enterprise-scale retrieval and generation.
3 SPAR - Session-based Pipeline for Adaptive Retrieval
3.1 Overview
We proposeSPAR(Session-based Pipeline for Adaptive Retrieval), a modular RAG framework designed for efficient
retrieval and generation over legacy enterprise file systems (Figure 2). Unlike conventional RAG architectures that rely
on static, large-scale vector databases mirroring entire file systems, SPAR adopts a flexible and lightweight strategy.
At its core is a metadata-driven indexing scheme that encodes essential file attributes and enterprise-defined tags,
yielding an enterprise-orientedMetadata Index. This index provides a scalable, semantically enriched representation
of enterprise data and serves as the foundation for targeted retrieval and adaptive vector construction.
Building on this index, SPAR operates in a session-oriented manner through what we call theSession-based RAG
Process. Each user query initiates a session that begins with metadata-aware filtering to identify relevant files. From
this subset, a temporary vector database is generated on demand, serving as a session-specific knowledge base. This
database supports RAG operations throughout the session and can be interactively updated, enabling more precise and
contextually relevant retrieval.
Together, these two components reduce computational overhead, enhance retrieval controllability, and ensure tighter
alignment with enterprise data semantics. The following subsections detail each stage in turn: first, the construction of
the Metadata Index, and second, the session-based RAG process that enables adaptive retrieval and generation.
4

LLM-based Retrieval-Augmented Generation on Legacy File Systems
1234Design the database schemeAutomate extraction of file tagsConstruct metadata indexInsert data into metadata indexSynchronize/update/delete continuously
Figure 3: Conceptual steps to build and maintain a Metadata Index.
3.2 Metadata Index construction
Instead of constructing a global vector database, we introduce aMetadata Index, which captures high-level descriptors
of enterprise data while being more efficient and easier to update. Figure 3 illustrates the conceptual steps. At the heart
of the index is a hierarchical system of enterprise-defined tags and structured metadata that together enable efficient,
controllable, and semantically meaningful retrieval.
Scheme Design.The Metadata Index can be modeled as a relational database with two core tables: Files andTags
(Table 1). Each file in the legacy system is represented in Files by its path, metadata, and associated tags. The Tags
table encodes both enterprise-defined and automatically extracted tags, organized into a hierarchy of parent and child
concepts. This structure provides a coarse but interpretable descriptor of the file corpus, enabling fast categorization
and retrieval.
Table 1: Design scheme of the Metadata Index as a relational database, consisting of two tables –FilesandTags.
Files
File_IDUnique ID in Metadata Index
File_PathActual path in the system
Tag_IDEnterprise-defined or LLM assigned tags
MetadataOther metadata specific for the fileTags
Tag_IDUnique ID in Metadata Index
Tag_ValueThe tag content
Parent_Tag_IDIDs of parent if exist
Children_Tag_IDIDs of children if exist
File Tag Assignment and Hierarchy Contruction.
Enterprise-defined flat tag set.Tags constitute the primary descriptors in the proposed Metadata Index, serving to
condense and represent files at a coarse level. We assume the availability of a flat set of enterprise-defined tags,
corresponding to the finest level of granularity and later forming the leaf nodes in the tag hierarchy. These tags are
typically specified by system owners to reflect the enterprise’s internal structure, domain logic, and organizational
conventions. By design, this flat tag set encodes institutional knowledge about how information is classified and
organized, ensuring that file categorization aligns with the enterprise’s operational context.
File-tag assignment.Once the tag vocabulary is established, each file in the legacy system must be mapped to one
or more tags that best describe its content. This assignment can be performed manually, semi-automatically, or fully
automatically using LLM-assisted tools [ 32,33,34]. Manual assignment offers high accuracy but is costly at scale,
whereas automated assignment is efficient but may introduce inconsistencies or noise. A pragmatic strategy is to
combine both: LLMs provide initial assignments, while human reviewers validate a portion of them, particularly for
critical datasets. This one-time effort for existing repositories is still significantly more tractable than constructing
and maintaining a global vector database mirroring the entire file system (see Section 4). For newly ingested files,
assignment can be integrated into the data pipeline as part of regular workflows, ensuring the index remains up to date.
Additional hierarchical taxonomy.To avoid flat tag sets becoming unmanageable or overlapping, we propose construct-
ing an enterprise-specific hierarchical taxonomy. Figure 4 illustrates such a structure in a healthcare setting: raw tags act
as leaf nodes, while intermediate nodes group related tags into broader categories based on domain logic or statistical
co-occurrence. Intermediate nodes can be defined manually by domain experts or automatically induced by LLMs,
which cluster semantically related leaf tags into higher-order groups. This taxonomy enables multi-level queries, where
selecting an intermediate node implicitly includes all its descendant tags, allowing users to broaden or narrow retrieval
scope depending on their intent. It also provides extensibility: new tags can be added over time and integrated into
existing categories, maintaining structural coherence as the dataset evolves.
5

LLM-based Retrieval-Augmented Generation on Legacy File Systems
Medical Data
Clinical Imaging
ChemotherapyDiagnosis
Neurological CardiovascularTreatment Radiology
X-ray MRI
Figure 4: Example of aHierarchical File Tagin a medical application. Leaf tags (orange) are defined by system
owners or extracted with LLMs, while intermediate tags are generated to group related concepts. Multi-level queries
allow broader or narrower retrieval based on user intent.
Taken together, enterprise-defined tags and hierarchical organization form a principled foundation for scalable retrieval
over large and heterogeneous datasets, without relying on full-text semantic embeddings. The hierarchy preserves
structural coherence by reflecting enterprise-specific categorization, while supporting expressive and interpretable
queries at varying levels of granularity. This design ensures transparent alignment between user intent and retrieval
results, while remaining extensible as enterprise data grows.
Metadata Characterization and Structuring.Beyond tags, enterprise files often include rich metadata (e.g., modality,
acquisition date, or institution for medical images; department or document type for clinical text). These attributes
are consistent across workflows and should be treated as first-class retrieval constraints rather than post-hoc filters. In
SPAR, metadata constraints are applied jointly with tag-based filtering, allowing users to combine conditions such as
time range, modality, or department with semantic tags. This narrows the search space before embeddings are created,
improving both efficiency and interpretability.
Comparison to Traditional Approaches.In conventional RAG pipelines, retrieval operates over a centralized vector
database, with metadata applied only after semantic similarity search. This post-hoc filtering unnecessarily enlarges
the candidate set and dilutes metadata precision when embedded into dense vectors. Such designs are especially
problematic for periodic or task-specific enterprise queries, where only a small, well-defined subset of data is relevant.
SPAR eliminates the global index and instead builds a lightweight metadata index defined explicitly by tag and metadata
constraints. This enables first-class filter of high-level and semantically rich information, allowing efficient retrieval of
relevant subset, reduces computational overhead, and keeps enterprise-defined structure central to the process. The
result is a faster, more precise, and more interpretable retrieval pipeline tailored to enterprise knowledge work. The next
subsection details how this Metadata Index is leveraged during theSession-based RAG Process, where session-specific
vector databases are dynamically constructed to support adaptive retrieval and generation.
3.3 Session-based Retrieval RAG
3.3.1 Workspaces: Contextualized Environments for Targeted Retrieval
To operationalize metadata- and tag-driven filtering, we introduce the concept ofWorkspaces. A workspace defines a
semantically scoped, time-bounded retrieval environment aligned with a specific project, reporting task, or analytic
thread. Each workspace is instantiated by filtering the global corpus according to metadata and tag constraints, ensuring
that only files relevant to the task are included. Figure 5 illustrates this design in a healthcare application: a workspace
focused ondiabetesmaintains a different document pool—and thus a distinct temporary vector database—than one
6

LLM-based Retrieval-Augmented Generation on Legacy File Systems
Workspace - Diabetes Care Program Q2 ReviewThread - Policy & Compliance AuditThread - Education & Outreach MaterialsThread - Medication Utilization ReviewDocument URIProcessed DataQ2_HbA1c_Analysis.csvPatient_Readmission_Rates_Report.pdfEndocrinology_Notes_2025-03.docxWorkspace – Obesity Care Program Q2 Review
Workspace – Metabolic Syndrome & Risk ReductionProcessed Documents
Figure 5: Example ofSession-based Retrievalin a medical application. Each workspace corresponds to an active task
with its own temporary vector database, while processed files can be cached and reused across workspaces to reduce the
overhead of workspace creation.
Retrieve all CT scans and associated documents for patient A dated May 25.
Figure 6: Example of a file retrieval prompt. Red text corresponds to metadata, while green text indicates extracted
keywords.
investigatingobesityormetabolic syndrome. Multiple discussion threads can coexist within a workspace, all sharing its
scoped database while maintaining thematic consistency.
This design eliminates the need for a centralized global index and instead constructs lightweight, task-specific vector
databases that exist only for the duration of the session. Retrieval is performed solely within this filtered subset, enabling
faster and more precise query resolution. Because the candidate pool is tightly scoped from the outset, embeddings
and normalized representations can be reused across queries within the same workspace, improving computational
efficiency and reducing latency. Workspaces therefore provide not only targeted retrieval but also a natural structure for
organizing enterprise knowledge around ongoing tasks.
3.3.2 Main functions within workspaces
Unlike ordinary RAG pipelines, which focus primarily on retrieval and query resolution, SPAR’s workspace framework
extends the functionality beyond simple querying. In particular, it introduces two additional core operations: the
creation and updating of temporary vector databases for active sessions, and their removal once the session concludes.
Files Retrieval and Updates.We introduce a dedicated command for file retrieval ( buildUpdateDtb() , see Figure 2),
which builds or updates the workspace’s temporary vector database with files relevant to the user’s query. Through
this command, the user specifies the type of documents to retrieve. From the prompt, the system automatically derives
filtering conditions by extracting keywords and metadata. An example prompt is shown in Figure 6, where red highlights
illustrate extracted metadata and green highlights show keywords used for filtering.
To perform this extraction, a language tool [ 35,36,37,38] is applied to identify salient keywords and metadata terms.
Extracted keywords are embedded and aligned with the existing tag hierarchy to find semantically related tags. To avoid
redundancy, a hierarchy-aware pruning step ensures that if both an ancestor and its descendant are retrieved, only the
ancestor is retained. This implicitly selects all the descendant tags underlying while prevents overlapping matches and
preserves broader conceptual categories–particularly useful when users express high-level intent without knowing the
precise vocabulary.
7

LLM-based Retrieval-Augmented Generation on Legacy File Systems
Medical Data
Clinical Imaging
ChemotherapyDiagnosis
Neurological CardiovascularTreatment Radiology
X-ray MRISelected Tags
Inferred Raw TagsNot Selected Tags
Figure 7: Example of tag selection from the hierarchy in a medical workspace.
Metadata constraints (e.g., time ranges, file formats, or modality types) are mapped directly to structured filters, as they
are typically explicit in the prompt and require no disambiguation.
The final filtering logic combines both dimensions: a file must match at least one tag from each selected tag group (after
hierarchical expansion) and satisfy all metadata constraints to be included. This two-stage process–first by tags, then by
metadata–substantially narrows the candidate pool, ensuring downstream retrieval operates only on documents aligned
with user intent.
As illustrated in Figure 7, this architecture also enhances transparency. The highlighted tag hierarchy could be returned to
users and provides a clear mapping from abstract query concepts to retrieved files, helping users understand why specific
documents were selected. In cases of few or no results, the structured filters can be visualized and inspected, allowing
users to diagnose whether the issue lies in mismatched tags, restrictive metadata, or ambiguous phrasing—facilitating
more effective query refinement.
File processing and vector database enrichment.Each candidate file is first normalized into a modality-specific
intermediate format (e.g., text extraction from documents, image preprocessing for scans), then embedded into a shared
semantic space using a multimodal encoder [ 39,40,41]. The resulting embeddings are stored in the workspace’s
temporary vector database, which serves as the basis for subsequent retrieval.
To avoid redundant computation, the system maintains a cache of normalized representations and embeddings. If a file
is reused across multiple workspaces, its cached representation is retrieved directly, significantly reducing processing
time and resource consumption. When a file is modified, its cached representation is automatically invalidated and
refreshed during the next processing cycle, ensuring that stale embeddings are never propagated across workspaces.
This mechanism supports consistent retrieval results even in dynamic file systems.
Workspace archival and termination.Workspaces are designed to be temporary and task-scoped. When inactive, a
workspace can either be archived or terminated. Archival removes the workspace’s vector database and intermediate
embeddings while preserving metadata, filtering criteria, and access history. After a predefined expiration period, this
process is triggered automatically.
Because the original file list and filtering rules are retained, a user can later reactivate the workspace and reconstruct its
state without full reprocessing or manual configuration. Archived workspaces may also be shared or transferred between
collaborators, enabling reproducibility and collaborative exploration without duplicating heavy preprocessing steps.
This design avoids the need for continuous background updates to a global index—an issue common in traditional RAG
systems, where file system changes can introduce inconsistencies and maintenance overhead. Instead, our approach
shifts this burden to the workspace creation phase, which is lightweight and explicitly aligned with user context. A
detailed theoretical comparison is provided in Section 4, with trade-offs discussed in Section 6.
8

LLM-based Retrieval-Augmented Generation on Legacy File Systems
4 Theoretical Comparison - SPAR versus ordinary LLM-based RAG
Throughout these analyses, we follow the system of notations defined in Table 2.
Table 2: Symbolic notations and definition for complexity analyses.
Symbol Definition
NTotal number of files provided by the enterprise.
MTotal number of predefined tags used to annotate files.
The tag space is semantically meaningful and interpretable.
Typically,M≪N
Tproc Average time required to process a single file.
Includes format conversion and embedding of the resulting content.
Ncandidates Number of files returned by the metadata/tag index for a query (coarse filtering).
This is the candidate setbeforepolicy/format/quality gates and cache checks.
TypicallyN filtered≤N candidates ;
when there are no extra gates or cache reuse,N candidates =N filtered .
Nfiltered Number of files actually admitted to the workspaceafterpolicy/format/quality gates
and cache checks (i.e., those we (re)process/embed and index).
TypicallyN filtered≪N, due to early-stage narrowing.
4.1 Construction and Maintenance Cost
Traditional RAG Pipeline
In traditional RAG systems, the entire file corpus must be preprocessed and indexed prior to retrieval. This involves:
•Preprocessing time:O(N·T proc)
•Index construction:Using an algorithm like HNSW, which requires O(NlogN) time in low-dimensional
settings [42]
Thus, the total construction time is:
TRAG=O(N·T proc) +O(NlogN)
Proposed Approach
SPAR performs targeted construction by leveraging enterprise tags and metadata as a first-stage filter, followed by
just-in-time processing and indexing per user session (workspace).
•Offline tag index:Built once by embedding the enterprise tag vocabulary and constructing an index (e.g.,
HNSW), with a one-time cost ofO(MlogM).
•Per-workspace vector database construction:Each user file retrieval command triggers the following steps:
–Tag search:Performed via approximate nearest neighbor search over the tag index using HNSW, yielding
complexityO(logM).
–File filtering:Via the Metadata Index (see Section 3.2 and Table 1), we perform indexed lookups. The
cost isO(N candidates ).
– Preprocessing:Each filtered file is processed and embedded, contributingO(N filtered·Tproc)time.
–Indexing:The resulting embeddings are added to a session-specific HNSW index, requiring
O(N filtered logN filtered)time.
9

LLM-based Retrieval-Augmented Generation on Legacy File Systems
Per-session construction time.Let p:=N filtered/Ndenote selectivity and assume indexed filtering so that Ncandidates ≈
Nfiltered =pN. Then the per-session cost decomposes as
Tours=O(MlogM)|{z}
one-time tag index+O(logM)|{z}
tag lookup+O(N candidates )|{z}
indexed filtering
+O(N filtered Tproc)|{z }
(re)processing/embedding+O(N filtered logN filtered)| {z }
workspace ANN index(1)
≈O(MlogM) +O(N filtered [1 +T proc+ logN filtered]),(⋆)
which shows thatT oursscales with thefilteredset sizeN filtered rather than the global corpus sizeN.
Break-even (back-of-the-envelope).Comparing a one-time ordinary build against WSPAR sessions/workspaces,
SPAR is preferable whenever
O(N T proc+NlogN)> O(MlogM) +WX
w=1O(N filtered,w [1 +T proc+ logN filtered,w ]).(2)
If sessions have similar selectivity pso that Nfiltered,s ≈pN , this reduces to the rule of thumb W p≪1 , i.e., a few
targeted sessions over narrow slices of the corpus amortize better than a full global prebuild.
Maintenance / ingestion
For ordinary RAG, whenkfiles arrive or change, the system pays
O(k T proc) +O(klog(N+k))
to (re)embed and insert them into the global ANN index. For SPAR, the Metadata Index is updated online with cost
O(tag_assign(k)),
while vector embeddings are computed just-in-time during sessions (no global vector reindex is required). This design
removes the “fall-out-of-sync” failure mode that can trigger costly full reconstructions in ordinary pipelines.
4.2 Search Time
Traditional RAG pipeline.In text-book analyses, graph-based approximate nearest-neighbor (ANN) methods such
as HNSW offer (near-)logarithmic query time, e.g., O(logN) under low effective dimensionality and fixed graph
degree [ 42]. In modern LLM-/VLM-derived embeddings, however, theeffectivedimension is high; distance concen-
tration and sparsity dilute ANN guarantees (“curse of dimensionality” [ 43]). In practice, maintaining recall requires
increasing search-time hyperparameters—e.g., HNSW’s dynamic candidate list size ef(efSearch ) and its maximum
neighborhood degree M—or analogous beam/queue widths in other graph ANNS such as NSG and DiskANN; these
raise per-query constant factors and can push latency beyond interactive budgets [44, 45, 46, 47].
SPAR’s selective search.SPAR narrows the search spacebeforevector lookup using the Metadata Index (Section 3.2),
yielding a filtered set of sizeN filtered (Table 2), typicallyN filtered≪N.
LetTANN(N, d, θ) denote the empirical cost of an ANN query as a function of database size N, effective dimension d,
and tuning parametersθ(e.g.,efSearch, graph degree). Then a single query costs
T(q)
RAG≈T ANN(N, d, θ)vs.T(q)
SPAR≈T ANN(Nfiltered, d, θ′),
withθ′free to be more recall-oriented because Nfiltered is much smaller. Since TANNis (empirically) increasing in both
Nandθ, there exists a settingθ′≥θ(increased recall-oriented knobs - e.g.,efSearch, graph degree) such that
TANN(Nfiltered, d, θ′)≤T ANN(N, d, θ),
while achieving equal or higher recall. This mitigates the high-dimensionality penalty observed in large, global indices.
•Latency-preserving regime:Fix recall and reduce latency by replacingNwithN filtered .
•Quality-raising regime:Keep latency flat and increase recall by raising θ→θ′(e.g., larger efSearch ) made
affordable by the smallerN filtered .
10

LLM-based Retrieval-Augmented Generation on Legacy File Systems
Amortization within a workspace.Each workspace builds an ANN index over its filtered set once at cost
O 
Nfiltered logN filtered
. ForQdownstream queries in the same session, the effective per-query overhead is
ONfiltered logN filtered
Q
,
which is negligible for interactive Q(e.g., multi-turn retrieval or re-ranking passes). Filtering time itself is accounted in
Section 4.1 via the Metadata Index and does not scale withN(linear scans are avoided).
By shrinking the candidate setbeforevector search, SPAR avoids the scalability bottlenecks of ever-growing global
indices and enables a favorable speed–quality trade-off: either faster responses at fixed quality or higher-quality results
at fixed latency, with index-build costs amortized over session queries.
4.3 Memory Usage
The memory footprint in vector-based retrieval systems consists of two main components: the storage of the embedding
vectors themselves and the additional overhead of the indexing structure (e.g., graph connections in HNSW). Let vbe
bytes per embedding andobe index overhead per vector.
Traditional RAG pipeline.AllNembeddings are stored once in a global vector index, resulting in:
Mem global =N·(v+o)
SPAR.Each workspace maintains a local index over a smaller filtered subset Nfiltered , but embeddings may be duplicated
across multiple indices if the same file is relevant to multiple workspaces. If Wworkspaces are active, the RAM used
by workspace-scoped indices is
Mem SPAR≈δ·
E[#unique files across active workspaces]
·(v+o),
whereδ∈[1, W]is the average duplication factor (how many workspaces, on average, index the same file).
A sufficient “win” condition vs. a global index of sizeNis
δ·W·E[N filtered]< N.
It should be noted that the memory footprint in our approach is further optimized by the workspace archival mechanism.
Inactive workspaces can be archived, which removes the associated vector index and intermediate embeddings while
retaining only the necessary metadata and access history. This reduces δfurther over time and ensures no unnecessarily
memory.
5 Application in synthesized biomedical literature corpus
To illustrate and provide a preliminary quantitative assessment of the SPAR pipeline, we synthesize a small file system
containing biomedical literature. On this corpus, we generate a limited set of multiple-choice questions to evaluate both
file retrieval accuracy and model performance in answering questions with the retrieved documents, comparing against
a conventional LLM-based RAG pipeline. This experiment is intended solely as a toy example and does not reflect the
full application scope of SPAR, which is designed for deployment on enterprise datasets at much larger scale.
5.1 Dataset Description
For this experiment, we sample 1000 full-text articles from the PMC Open Access Subset [ 48], which contains millions
of biomedical papers in raw text format released under permissive reuse licenses. Each article is linked to a PubMed ID
(PMID), which we use to retrieve additional metadata, including MeSH descriptors, from the 2025 PubMed Baseline
dataset [49].
5.2 Baseline Construction
We construct both an ordinary RAG system and our SPAR pipeline to work with this toy corpus for comparison. Both
systems are backed by a same LLM agent - Qwen2.5-VL 3B [50] to ensure no model bias is introduced.
Ordinary RAG system.For the baseline, we construct a global vector database covering the entire set of 1000 articles.
Each article is split into passages using a fixed window size, and the resulting chunks are embedded with a sentence-level
11

LLM-based Retrieval-Augmented Generation on Legacy File Systems
encoder to produce dense representations. These embeddings are indexed in Pinecone database [ 51] and serve as the
knowledge base for retrieval. At query time, the user question is embedded, nearest neighbors are retrieved from the
global index, and the retrieved passages are provided as context to the LLM for answer generation. This setup mirrors
the standard RAG pipeline widely used in practice, where a single, centralized vector store is maintained across the
entire corpus.
SPAR system.To correctly set up the SPAR pipeline, the major component needed to construct is the tag hierarchy and
Metadata Index.
MeSH Hierarchy Construction.To substitute for enterprise-defined tags, we adopt the Medical Subject Headings
(MeSH) taxonomy as the structured tag system for biomedical literature. MeSH is organized hierarchically through
TreeNumber identifiers, which encode parent–child relationships. For example, G07is a parent of G07.025 , which
in turn is a parent of G07.025.133 . Table 3 illustrates this structure with the descriptorAdaptation, Physiological
(D000222 ), which appears under both G07.025 andG16.012.500 , and itself serves as a parent to terms such as
AcclimatizationandBody Temperature Regulation. This overlapping DAG structure naturally supports hierarchical tag
expansion, enabling queries to be broadened or narrowed based on user intent.
To operationalize this hierarchy, we traverse the MeSH XML descriptors and store them in the Tags table of our
Metadata Index. Hierarchical relationships are derived by comparing TreeNumber values: if one is a prefix of another,
it is treated as an ancestor. In parallel, each MeSH descriptor is embedded into the vector database to support semantic
similarity during file retrieval latter. Embeddings are constructed using both the descriptor’s name and its associated
metadata fields (e.g., notes, annotations), yielding richer semantic representations.
Table 3: Excerpt of MeSH hierarchy illustrating the DAG structure centered onAdaptation, Physiological.
TreeNumber MeSH ID Term Name
G07 D010829 Physiological Phenomena
G16.012 D000220 Adaptation, Biological
G07.025 D000222 Adaptation, Physiological
G16.012.500 D000222 Adaptation, Physiological
G07.025.133 D000064 Acclimatization
G16.012.500.133 D000064 Acclimatization
G16.012.500.535 D001833 Body Temperature Regulation
Metadata Index Finalization.Finally, the curated files from the previous step are linked to their corresponding MeSH
descriptors by populating the Files table with file paths, metadata fields, and associated MeSH IDs. This completes
the Metadata Index, equipping the system to perform tag-based filtering, scoped embedding, and hierarchical navigation
of biomedical content at query time. Implementation details, including specific databases and tools, are provided in
Appendix A.1.
5.3 Experimental Setup
To evaluate SPAR against a conventional RAG baseline, we design an experiment that jointly measures retrieval
accuracy and downstream answer accuracy. The evaluation relies on a set of automatically generated multiple-choice
questions, each tied to a specific article in the corpus.
Question design.From the full set of 1000 biomedical articles, we select a subset of 335target articles that collectively
cover a broad range of MeSH tags, ensuring representative coverage of the corpus. The remaining 665articles serve as
distractors, thereby introducing realistic noise into the retrieval process. For each target article, a single multiple-choice
question is generated using a larger variant of the chosen LLM agent—Qwen-2.5VL 7B [ 50]. Each question consists of
four candidate answers, with exactly one correct option, and is grounded in the main text of the associated article. The
prompting procedure used for question generation is detailed in Appendix A.1.
Retrieval accuracy.We measure retrieval performance by computing the true positive rate, defined as the proportion of
queries for which at least one relevant passage from the ground-truth article is retrieved among the top- kresults (with
k= 5 in all experiments). For SPAR, retrieval and querying are decoupled into two commands: (1) file retrieval, which
returns relevant documents, and (2) normal querying, which answers user questions using the workspace database. To
ensure fairness, we apply the same set of questions to both systems, but strip away the multiple-choice options when
performing file retrieval for SPAR. Sample questions are provided in Appendix A.2.
12

LLM-based Retrieval-Augmented Generation on Legacy File Systems
Answer accuracy.Answer accuracy is assessed by the true positive rate of the LLM agent’s responses to the full
multiple-choice questions, i.e., the proportion of cases where the correct answer is selected. Both SPAR and the baseline
system use the same agent—Qwen2.5-VL 3B [ 50]—for answer generation, ensuring that differences arise solely from
retrieval design.
5.4 Results and Interpretation
Table 4: Preliminary quantitative results of two RAG systems on the toy biomedical corpus.
Ordinary RAG SPAR (Ours)
Retrieval Accuracy 80.3% 89.5%
Average Retrieval Time 0.039s 0.015s
Answer Accuracy 65.1% 68.1%
Table 4 reports the quantitative comparison between the ordinary RAG baseline and SPAR. Overall, SPAR demonstrates
consistent improvements in retrieval accuracy, answer accuracy, and efficiency while using the same LLM agent.
Specifically, SPAR achieves a 9.2% absolute improvement in retrieval accuracy. This gain highlights the benefit of
hierarchical routing and metadata-aware filtering, which narrow the candidate search space and improve relevance.
Importantly, to ensure fairness, we did not tailor the queries or prompts specifically for file retrieval. With optimized
prompting strategies, we expect the retrieval gap could widen further in favor of SPAR.
In addition, the average retrieval time in SPAR is reduced by more than half compared to ordinary RAG ( 0.015 second
vs.0.039 second), reflecting the efficiency of constructing smaller, task-specific vector databases rather than operating
over a large global index.
Finally, by providing the LLM with more relevant context, SPAR improves answer accuracy by approximately 3%.
While the gain is modest, it demonstrates the downstream impact of more precise retrieval and points toward further
improvements when combined with optimized retrieval–generation integration.
6 Discussion
6.1 Motivation Recap
As outlined earlier, SPAR prioritizes resource efficiency by avoiding global vector indexes when working with large
legacy file-based system and instead relying on on-demand, task-scoped workspaces. This design reduces persistent
storage and maintenance costs while aligning with the nature of enterprise workflows, where retrieval is typically
short-term and context-specific. The architecture also emphasizes modularity, transparency, and controllability, all of
which are often overlooked in conventional RAG pipelines.
While these benefits are evident, SPAR also introduces limitations when applied outside its intended scope. We discuss
these trade-offs below.
6.2 Trade-offs of On-Demand Workspace Design
Because vector indexes are constructed on demand, each workspace must preprocess and embed its candidate file
set whenever retrieval is initiated. For small or infrequent tasks this overhead is acceptable, but in larger sessions or
environments where similar workspaces are repeatedly created, the setup cost becomes more pronounced.
To reduce redundant computation, we cache normalized and embedded file representations at the file level. This
prevents re-embedding for previously processed files, but filtering and index construction must still be repeated for each
workspace. In high-throughput or multi-user settings, this design may result in avoidable overhead.
Another challenge is storage redundancy. Files reused across different sessions may be embedded multiple times into
separate workspace-local indexes. Although these indexes are ephemeral—archived or discarded once a workspace
is closed—the transient duplication still increases memory and disk footprint when many workspaces are active
simultaneously. Without further optimizations such as deduplication or shared indexing, storage costs can scale linearly
with the number of concurrent workspaces.
In practice, we expect these issues to emerge primarily under atypical usage patterns (e.g., excessive or overlapping
workspace creation). For the scoped, project-oriented tasks common in enterprise environments, SPAR is likely to
13

LLM-based Retrieval-Augmented Generation on Legacy File Systems
operate efficiently. Nonetheless, future improvements such as incremental index construction or cross-workspace
sharing mechanisms could further mitigate redundancy.
6.3 Dependency on Metadata Quality
SPAR’s file filtering logic relies on the structured metadata (e.g., timestamps, enterprise-defined tags) as a prerequisite.
In practice, however, metadata may be incomplete, inconsistently applied, or altogether absent. Enterprise-defined tags,
in particular, are often curated manually, which can make them noisy, sparse, or unavailable in domains where ongoing
maintenance is resource-intensive.
To alleviate this burden, we proposed LLM-assisted file tagging and hierarchy construction. While effective in scaling
tag assignment, this approach introduces its own trade-offs. First, consistency: similar files may be assigned slightly
different tags, reducing retrieval reliability. This can be addressed through normalization rules, clustering-based
reconciliation, or periodic audits with human oversight. Second, interpretability: automatically generated hierarchies
may suffer from over-grouping (merging distinct concepts) or mis-grouping (placing tags under inappropriate parents).
These issues can be mitigated with lightweight expert supervision, constraint-based clustering, or iterative refinement.
Despite these risks, hybrid strategies—enterprise-defined leaf tags combined with LLM-augmented grouping—offer a
pragmatic balance between scalability and semantic control.
When metadata is missing or unreliable, recall becomes the primary concern. Relevant documents may be excluded too
early in the retrieval process, preventing them from reaching the embedding stage. This limitation is especially acute in
legacy systems with inconsistent archival practices. SPAR partially mitigates this issue by exposing the tag and metadata
predicates used in retrieval, giving users visibility into the filtering logic and the ability to relax overly strict constraints.
However, the root cause lies in organizational data hygiene: robust metadata management and standardization are
prerequisites for maximizing SPAR’s effectiveness.
6.4 Summary of Limitations and Future Directions
In summary, SPAR trades persistent global indexing for on-demand workspace construction, achieving efficiency and
controllability at the cost of repeated setup overhead and potential redundancy under heavy use. Likewise, its reliance
on metadata offers interpretability and precision but inherits the weaknesses of incomplete or inconsistent metadata
ecosystems. These trade-offs highlight opportunities for future research, such as incremental or shared indexing across
workspaces, adaptive caching policies, and more reliable metadata generation pipelines. Addressing these challenges
would further strengthen SPAR’s practicality for deployment in diverse enterprise environments.
References
[1]Michael L Brodie. The promise of distributed computing and the challenges of legacy information systems. In
Interoperable database systems (DS-5), pages 1–31. Elsevier, 1993.
[2]Sanjay Jha, Meena Jha, Liam O’Brien, and Marilyn Wells. Integrating legacy system into big data solutions: Time
to make the change. InAsia-Pacific World Congress on computer science and engineering, pages 1–10. IEEE,
2014.
[3]Zixuan Ke, Weize Kong, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. Bridging the
preference gap between retrievers and LLMs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 10438–10451, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
[4]Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael
Zeng, and Meng Jiang. Generate rather than retrieve: Large language models are strong context generators. In
ICLR. OpenReview.net, 2023.
[5]Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt,
Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy.
Recommender systems with generative retrieval. InThirty-seventh Conference on Neural Information Processing
Systems, 2023.
[6]Yubo Ma, Yixin Cao, Yong Ching Hong, and Aixin Sun. Large language model is not a good few-shot information
extractor, but a good reranker for hard samples! InThe 2023 Conference on Empirical Methods in Natural
Language Processing, 2023.
14

LLM-based Retrieval-Augmented Generation on Legacy File Systems
[7]Zhebin Zhang, Xinyu Zhang, Yuanhang Ren, Saijiang Shi, Meng Han, Yongkang Wu, Ruofei Lai, and Zhao Cao.
IAG: Induction-augmented generation framework for answering reasoning questions. InThe 2023 Conference on
Empirical Methods in Natural Language Processing, 2023.
[8]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[9]Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. A comprehensive survey of retrieval-augmented
generation (rag): Evolution, current landscape and future directions.arXiv preprint arXiv:2410.12837, 2024.
[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang.
Retrieval-augmented generation for large language models: A survey.arXiv preprint arXiv:2312.10997, 2023.
[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional
transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors,Proceedings
of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June
2019. Association for Computational Linguistics.
[12] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei
Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer.Journal of
machine learning research, 21(140):1–67, 2020.
[13] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave,
and Guillaume Lample. Llama: Open and efficient foundation language models.ArXiv, abs/2302.13971, 2023.
[14] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus.IEEE Transactions on
Big Data, 7(3):535–547, 2021.
[15] Tianyang Zhang, Zhuoxuan Jiang, Shengguang Bai, Tianrui Zhang, Lin Lin, Yang Liu, and Jiawei Ren. Rag4itops:
A supervised fine-tunable and comprehensive rag framework for it operations and maintenance.arXiv preprint
arXiv:2410.15805, 2024.
[16] Mathieu Bourdin, Anas Neumann, Thomas Paviot, Robert Pellerin, and Samir Lamouri. An agile method for
implementing retrieval augmented generation tools in industrial smes.arXiv preprint arXiv:2508.21024, 2025.
[17] Isaac Shi, Zeyuan Li, Wenli Wang, Lewei He, Yang Yang, and Tianyu Shi. esapiens: A real-world nlp framework
for multimodal document understanding and enterprise knowledge processing.arXiv preprint arXiv:2506.16768,
2025.
[18] Rajat Khanda. Agentic ai-driven technical troubleshooting for enterprise systems: A novel weighted retrieval-
augmented generation paradigm.arXiv preprint arXiv:2412.12006, 2024.
[19] Binita Saha, Utsha Saha, and Muhammad Zubair Malik. Advancing retrieval-augmented generation with inverted
question matching for enhanced qa performance.IEEE Access, 2024.
[20] Yuxuan Chen, Daniel Röder, Justus-Jonas Erker, Leonhard Hennig, Philippe Thomas, Sebastian Möller, and
Roland Roller. Retrieval-augmented knowledge integration into language models: A survey. In Sha Li, Manling
Li, Michael JQ Zhang, Eunsol Choi, Mor Geva, Peter Hase, and Heng Ji, editors,Proceedings of the 1st Workshop
on Towards Knowledgeable Language Models (KnowLLM 2024), pages 45–63, Bangkok, Thailand, August 2024.
Association for Computational Linguistics.
[21] Jingxuan Han, Zhendong Mao, Yi Liu, Yexuan Che, Zheren Fu, and Quan Wang. Fine-grained knowledge
enhancement for retrieval-augmented generation. InFindings of the Association for Computational Linguistics:
ACL 2025, pages 10031–10044, 2025.
[22] Wenjun Peng, Guiyang Li, Yue Jiang, Zilong Wang, Dan Ou, Xiaoyi Zeng, Derong Xu, Tong Xu, and Enhong
Chen. Large language model based long-tail query rewriting in taobao search. InCompanion Proceedings of the
ACM on Web Conference 2024, pages 20–28, 2024.
[23] Duy A Nguyen, Rishi Kesav Mohan, Shimeng Yang, Pritom Saha Akash, and Kevin Chen-Chuan Chang. Minielm:
A lightweight and adaptive query rewriting framework for e-commerce search optimization. InFindings of the
Association for Computational Linguistics: ACL 2025, pages 6952–6964, 2025.
[24] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in retrieval-augmented large
language models. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing,
pages 5303–5315, 2023.
15

LLM-based Retrieval-Augmented Generation on Legacy File Systems
[25] Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H. Chi, Quoc V Le, and Denny
Zhou. Take a step back: Evoking reasoning via abstraction in large language models. InThe Twelfth International
Conference on Learning Representations, 2024.
[26] Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj Solanki. Blended rag: Improving rag (retriever-augmented
generation) accuracy with semantic search and hybrid query-based retrievers. In2024 IEEE 7th international
conference on multimedia information processing and retrieval (MIPR), pages 155–161. IEEE, 2024.
[27] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang
Wang, Shichao Sun, Huanyu Li, et al. Ragchecker: A fine-grained framework for diagnosing retrieval-augmented
generation.Advances in Neural Information Processing Systems, 37:21999–22027, 2024.
[28] Abdelrahman Abdallah, Mahmoud Abdalla, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali, and Adam
Jatowt. Rankarena: A unified platform for evaluating retrieval, reranking and rag with human and llm feedback.
arXiv preprint arXiv:2508.05512, 2025.
[29] Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stewart Massie, Ikechukwu Nkisi-Orji,
Ruvan Weerasinghe, Anne Liret, and Bruno Fleisch. Cbr-rag: Case-based reasoning for retrieval augmented
generation in llms for legal question answering. In Juan A. Recio-Garcia, Mauricio G. Orozco-del Castillo, and
Derek Bridge, editors,Case-Based Reasoning Research and Development, pages 445–460, Cham, 2024. Springer
Nature Switzerland.
[30] Jaswinder Singh. How rag models are revolutionizing question-answering systems: Advancing healthcare, legal,
and customer support domains.Distributed Learning and Broad Applications in Scientific Research, 5:850–866,
2019.
[31] Xiang Li, Zhenyu Li, Chen Shi, Yong Xu, Qing Du, Mingkui Tan, and Jun Huang. Alphafin: Benchmarking
financial analysis with retrieval-augmented stock-chain framework. InProceedings of the 2024 Joint International
Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages
773–783, 2024.
[32] Ruiming Tang, Chenxu Zhu, Bo Chen, Weipeng Zhang, Menghui Zhu, Xinyi Dai, and Huifeng Guo. Llm4tag:
Automatic tagging system for information retrieval via large language models. InProceedings of the 31st ACM
SIGKDD Conference on Knowledge Discovery and Data Mining V . 2, pages 4882–4890, 2025.
[33] Chen Li, Yixiao Ge, Jiayong Mao, Dian Li, and Ying Shan. Taggpt: Large language models are zero-shot
multimodal taggers.arXiv preprint arXiv:2304.03022, 2023.
[34] Emily Johnson, Xavier Holt, and Noah Wilson. Improving the accuracy and efficiency of legal document tagging
with large language models and instruction prompts.arXiv preprint arXiv:2504.09309, 2025.
[35] Tim Schopf, Simon Klimek, and Florian Matthes. Patternrank: Leveraging pretrained language models and part
of speech for unsupervised keyphrase extraction.arXiv preprint arXiv:2210.05245, 2022.
[36] Matej Martinc, Blaž Škrlj, and Senja Pollak. Tnt-kid: Transformer-based neural tagger for keyword identification.
Natural Language Engineering, 28(4):409–448, 2022.
[37] Reza Yousefi Maragheh, Chenhao Fang, Charan Chand Irugu, Parth Parikh, Jason Cho, Jianpeng Xu, Saranyan
Sukumar, Malay Patel, Evren Korpeoglu, Sushant Kumar, et al. Llm-take: Theme-aware keyword extraction using
large language models. In2023 IEEE International Conference on Big Data (BigData), pages 4318–4324. IEEE,
2023.
[38] Nacef Ben Mansour, Hamed Rahimi, and Motasem Alrahabi. How well do large language models extract
keywords? a systematic evaluation on scientific corpora. InProceedings of the 1st Workshop on AI and Scientific
Discovery: Directions and Opportunities, pages 13–21, 2025.
[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language
supervision. InInternational conference on machine learning, pages 8748–8763. PmLR, 2021.
[40] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and
Ishan Misra. Imagebind: One embedding space to bind them all. InProceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 15180–15190, 2023.
[41] Yunhao Liu, Suyang Xi, Shiqi Liu, Hong Ding, Chicheng Jin, Chenxi Yang, Junjun He, and Yiqing Shen.
Multimodal medical image binding via shared text embeddings.arXiv preprint arXiv:2506.18072, 2025.
[42] Yu A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical
navigable small world graphs.IEEE Trans. Pattern Anal. Mach. Intell., 42(4):824–836, April 2020.
16

LLM-based Retrieval-Augmented Generation on Legacy File Systems
[43] Michel Verleysen and Damien François. The curse of dimensionality in data mining and time series prediction. In
International work-conference on artificial neural networks, pages 758–770. Springer, 2005.
[44] Yu A Malkov and Dmitry A Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical
navigable small world graphs.IEEE transactions on pattern analysis and machine intelligence, 42(4):824–836,
2018.
[45] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. Fast approximate nearest neighbor search with the
navigating spreading-out graph.arXiv preprint arXiv:1707.00143, 2017.
[46] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnawamy, and Rohan
Kadekodi. Diskann: Fast accurate billion-point nearest neighbor search on a single node.Advances in neural
information processing Systems, 32, 2019.
[47] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré,
Maria Lomeli, Lucas Hosseini, and Hervé Jégou. The faiss library.arXiv preprint arXiv:2401.08281, 2024.
[48] U.S. National Library of Medicine. Pubmed central (pmc) open access subset. https://pmc.ncbi.nlm.nih.
gov/tools/openftlist/, 2025. Accessed: 2025-09-08.
[49] U.S. National Library of Medicine. Pubmed 2025 baseline dataset. https://lhncbc.nlm.nih.gov/ii/
tools/pm-baseline.html, 2025. Accessed: 2025-09-08.
[50] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang,
Jun Tang, et al. Qwen2. 5-vl technical report.arXiv preprint arXiv:2502.13923, 2025.
[51] Pinecone Systems, Inc. Pinecone vector database. https://www.pinecone.io/ , 2025. Accessed: 2025-09-08.
[52] Mintplex Labs. Anything llm. https://github.com/Mintplex-Labs/anything-llm , 2025. Accessed:
2025-09-08.
[53] Ollama. Ollama: Framework for running large language models locally. https://ollama.com/ , 2025. Accessed:
2025-09-08.
[54] PostgreSQL Global Development Group. Postgresql. https://www.postgresql.org/ , 2025. Relational
database system, accessed: 2025-09-08.
[55] Jeongmin M Lee, Juyeong Kim, Kyeongeun Kim, Yesol Yim, Yerin Hwang, Selin Woo, and Dong Keon Yon.
Association between residential greenness and allergic diseases among adolescents in south korea: A nationwide
representative study.Pediatric Allergy and Immunology, 36(9):e70199, 2025.
[56] Hyunji Park, Jiwon Cheon, Hyojung Kim, Jihye Kim, Jihyun Kim, Jeong-Yong Shin, Hyojin Kim, Gaeun Ryu,
In Young Chung, Ji Hun Kim, et al. Gut microbial production of imidazole propionate drives parkinson’s
pathologies.Nature Communications, 16(1):8216, 2025.
[57] Chieu-Hoang Ly Luong, Lisa Kalisch Ellett, Nicole Pratt, Kirsten Staff, and Jack Janetzki. Trends in use of
direct-acting antivirals for treatment of hepatitis c virus infection in australia 2016–2024.Journal of Viral Hepatitis,
32(10):e70082, 2025.
17

LLM-based Retrieval-Augmented Generation on Legacy File Systems
A Appendix
A.1 Implementation details of SPAR in Biomedical Corpus application.
System Construction
We implement both the ordinary RAG baseline and the SPAR prototype by extending the codebase of the open-source
RAG frameworkAnythingLLM[52].
For the ordinary RAG pipeline, the system is backed by a Pinecone vector database [ 51], which stores dense embeddings
for the entire corpus. Querying and response generation are handled by an open-source LLM chatbot agent integrated
through the Ollama framework [ 53]. This setup reflects the standard retrieve-and-generate paradigm widely adopted in
practice.
For SPAR, we replace the global vector index with aMetadata Indeximplemented in PostgreSQL [ 54], following
the relational schema described in Table 1. To support tag extraction and comparison during user retrieval commands,
we additionally maintain a lightweight Pinecone vector database containing embeddings of the tag vocabulary. This
auxiliary index enables semantic similarity checks over tags while the Metadata Index governs structural filtering and
hierarchy management. Together, these components realize SPAR’s session-based retrieval logic.
Prompts used to query LLM Agent
To generate our set of multiple choice questions, the following prompt is fed into Qwen2.5-VL 7B along with the
corresponding text articles:
[ARTICLE_METADATA]
Title: TITLE
PMID: PMID
MeSH tags: MESH_TAGS_COMMA_SEPARATED
[ARTICLE_MAIN_TEXT]
PLAIN_TEXT_MAIN_BODY
[TASK]
Create ONE grounded 4-option MCQ as JSON following the schema:
{
"pmid": "<PMID>",
"mesh_tags_used": ["<MeSH 1>", "<MeSH 2>", "..."], // choose 1–3 most relevant
"question": "<clear, self-contained stem>",
"options": { "A": "...", "B": "...", "C": "...", "D": "..." },
"correct_option": "A|B|C|D",
}
Target any of these facet (pick ONE that the text supports best):
- precise finding (Results),
- mechanism/pathway explanation,
- methodology detail (e.g., study design, sample, assay),
- clinical implication or limitation/assumption.
Constraints & style:
- The stem must be answerable from the provided text alone.
- Options A–D must be plausible and mutually exclusive; only one is fully supported by the text.
- If the text supports multiple options, revise distractors to be close but definitively
incorrect (alter condition, scope, comparator, unit, cohort, or time frame).
- Avoid “not/except” stems unless absolutely necessary.
If a unique correct answer cannot be guaranteed, output:
{
"skip": true,
"reason": "unanswerable or multiple correct"
}
[OUTPUT]
JSON only. No commentary.
18

LLM-based Retrieval-Augmented Generation on Legacy File Systems
A.2 Examples of multiple choice questions
In this section, we present some MCQs generated by Qwen2.5-VL 7B for the toy experiment.
For article [55] that studies residential greenness and allergic diseases, here is the generated question:
Which satellite product and resolution did the study use to compute NDVI for residential
greenness?
A. Landsat-8 OLI; 30 m, monthly, UTM projection
B. Terra MODIS MOD13A3 (v6.1); 1 km, monthly, sinusoidal projection
C. Sentinel-2 MSI; 10 m, weekly, WGS-84 projection
D. VIIRS; 500 m, daily, Lambert conformal projection
Correct answer: B
Similarly, for [56]:
In antibiotic-treated mice colonized with Streptococcus mutans, which observation best
supports that mTORC1 activation mediates the PD-like pathology?
A. Rapamycin lowered plasma and brain imidazole propionate (ImP) to baseline but did not
change neuronal pathology.
B. Rapamycin left pS6/4E-BP1 phosphorylation unchanged yet improved motor function.
C. Rapamycin suppressed mTORC1 signaling in TH+ neurons and rescued neurodegeneration and
motor deficits, even though ImP levels remained elevated.
D. Pirfenidone reduced dopaminergic neurodegeneration in vivo without affecting ImP levels.
Correct answer: C
For [57]:
During the study period, when did Australia’s HCV DAA dispensings peak, and what was the
approximate count?
A. March 2016; ~11,400 dispensings
B. June 2016; ~20,200 dispensings
C. July 2018; ~10,000 dispensings
D. January 2021; ~4,000 dispensings
Correct answer: B
19