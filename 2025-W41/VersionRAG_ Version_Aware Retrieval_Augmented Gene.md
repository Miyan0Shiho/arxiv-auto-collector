# VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents

**Authors**: Daniel Huwiler, Kurt Stockinger, Jonathan Fürst

**Published**: 2025-10-09 11:48:58

**PDF URL**: [http://arxiv.org/pdf/2510.08109v1](http://arxiv.org/pdf/2510.08109v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems fail when documents evolve
through versioning-a ubiquitous characteristic of technical documentation.
Existing approaches achieve only 58-64% accuracy on version-sensitive
questions, retrieving semantically similar content without temporal validity
checks. We present VersionRAG, a version-aware RAG framework that explicitly
models document evolution through a hierarchical graph structure capturing
version sequences, content boundaries, and changes between document states.
During retrieval, VersionRAG routes queries through specialized paths based on
intent classification, enabling precise version-aware filtering and change
tracking. On our VersionQA benchmark-100 manually curated questions across 34
versioned technical documents-VersionRAG achieves 90% accuracy, outperforming
naive RAG (58%) and GraphRAG (64%). VersionRAG reaches 60% accuracy on implicit
change detection where baselines fail (0-10%), demonstrating its ability to
track undocumented modifications. Additionally, VersionRAG requires 97% fewer
tokens during indexing than GraphRAG, making it practical for large-scale
deployment. Our work establishes versioned document QA as a distinct task and
provides both a solution and benchmark for future research.

## Full Text


<!-- PDF content starts -->

VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving
Documents
Daniel Huwiler Kurt Stockinger Jonathan Fürst
Zurich University of Applied Sciences, Switzerland
daniel.huwiler@proton.me, kurt.stockinger@zhaw.ch, jonathan.fuerst@zhaw.ch
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems fail when documents evolve through ver-
sioning—a ubiquitous characteristic of tech-
nical documentation. Existing approaches
achieve only 58-64% accuracy on version-
sensitive questions, retrieving semantically sim-
ilar content without temporal validity checks.
We present VersionRAG, a version-aware RAG
framework that explicitly models document
evolution through a hierarchical graph structure
capturing version sequences, content bound-
aries, and changes between document states.
During retrieval, VersionRAG routes queries
through specialized paths based on intent clas-
sification, enabling precise version-aware fil-
tering and change tracking. On our Ver-
sionQA benchmark—100 manually curated
questions across 34 versioned technical doc-
uments—VersionRAG achieves 90% accu-
racy, outperforming naive RAG (58%) and
GraphRAG (64%). VersionRAG reaches 60%
accuracy on implicit change detection where
baselines fail (0-10%), demonstrating its ability
to track undocumented modifications. Addi-
tionally, VersionRAG requires 97% fewer to-
kens during indexing than GraphRAG, making
it practical for large-scale deployment. Our
work establishes versioned document QA as a
distinct task and provides both a solution and
benchmark for future research.
1 Introduction
Retrieval-Augmented Generation (RAG) has
emerged as the dominant paradigm for grounding
large language models (LLMs) in external knowl-
edge, enabling accurate question answering over
large document collections (Fan et al., 2024; Gao
et al., 2023). However, a critical yet underexplored
challenge arises when these documents evolve over
time through versioning—a ubiquitous character-
istic of technical documentation, API references,
legal documents, and scientific literature. In such
Figure 1: Standard RAG fails to correctly associate
version-specific context. Although version 15.14.0 con-
tains the correct answer, the model is misled by seman-
tically similar but temporally irrelevant versions.
environments, even minor version changes can fun-
damentally alter the validity of an answer, yet exist-
ing RAG methods lack mechanisms to distinguish
between versioned document states:
Challenge 1: Version Conflation.Consider a
software development team querying their doc-
umentation:“What is the stability level of as-
sert.CallTracker in Node.js version 15.14.0”As
illustrated in Figure 1, standard RAG retrieves
semantically similar content from multiple ver-
sions (14.21.3, 15.14.0, 16.20.2), creating ambi-
guity where version 16.20.2 marks the function as
deprecated while earlier versions show it as experi-
mental or stable. This version conflationleads to
incorrectorcontradictoryanswers inversion-spe-
cific queries.
Challenge 2: No Tracking of Implicit Changes.
The challenge extends beyond simple version iden-
tification. When asked“In which version was as-
sert.deepEqual() removed?"”, systems must rea-
son about state transitions across versions. Fig-
ure 2 shows how current approaches fail this
task—retrieving multiple deprecation notices with-
out detecting the actual removal event, as they
lack mech anisms totrack changes between doc-
ument states. Recent graph-based extensions like
GraphRAG (Zhang et al., 2025) model seman-
tic relationships between concepts but still fail
1arXiv:2510.08109v1  [cs.IR]  9 Oct 2025

Figure 2: RAG system fails to identify the version in
which a function was removed, despite retrieving multi-
ple relevant deprecation entries.
Figure 3: GraphRAG fails to answer version-specific
questions despite having version nodes, as it lacks ex-
plicit version-to-version connections and change track-
ing.
on versioned documents. As shown in Figure 3,
while GraphRAG captures relationships between
functions and modules, it cannot determine when
assert.deepEqual() was removed because ver-
sion transitions are not explicitly modeled in the
graph structure. Our experiments confirm this limi-
tation: GraphRAG achieves only 10% accuracy on
implicit change detectiontasks(see Section 6).
To address both challenges, we presentVer-
sionRAG (Version-Aware Retrieval-Augmented
Generation for Evolving Documents), a novel
retrieval-augmented generation method that explic-
itly models document versioning through a struc-
tured graph representation. Unlike existing ap-
proaches that treat documents as static entities,
VersionRAG constructs a hierarchical graph dur-
ing indexing that captures: (1) version sequences
and relationships, (2) explicit and implicit changes
between versions, and (3) version-specific con-
tent boundaries. This structure enables precise
version-aware retrieval while maintaining compat-
ibility with standard vector-based search for non-
versioned queries.
Our key insight is that versioned document QA
requires fundamentally different retrieval strategies
depending on query intent. We identify three dis-
tinct query types—(i) content retrieval, (ii) version
listing, and (iii) change retrieval—each demand-
ing specialized graph traversal and filtering mech-
anisms. By classifying queries and routing them
through appropriate retrieval paths, VersionRAG
achieves 90% accuracy onourevaluationdatasetcompared to58% forstandard RAG and64% for
GraphRAG.
Beyond effectiveness, VersionRAG demon strates
remark able efficiency.Our structured graph rep-
resentation requires 97% fewer tokens duringin-
dexingcompared toGraphRAG, as version rela-
tionships are encoded as graph edges rather than
extracted through expensive LLM calls. This ef-
ficiency makes VersionRAG practical for large-
scale deployment on continuously evolving doc-
ument collections. We make the followingcon-
tributions: 1Versioned document QA task:
We formalize the versioned document QA task
and identify three fundamental query types that
require distinct retrieval strategies, providing a
framework for future research in this area (Sec-
tion 4). 2VersionRAG:We propose Version-
RAG, a novel graph-based retrieval framework
that explicitly models version relationships and
changes, enabling accurate reasoning over docu-
ment evolution while maintaining efficiency (Sec-
tion 4). 3VersionQA Dataset:We create a com-
prehensive benchmark of 100 manually curated
question-answer pairs across 34 versioned techni-
cal documents, spanning six evaluation categories
designed to test different aspects of version-aware
reasoning (Section 5). 4Evaluation:We demon-
strate through extensive experiments that Version-
RAG achieves 90% accuracy, significantly outper-
forming both standard RAG (58%) and GraphRAG
(64%), while requiring 97% fewer indexing tokens
than GraphRAG (Section 6). All our code and
dataset ispublicly available: https://github.
com/danielhuwiler/versionrag
2 Related Work
Retrieval-Augmented Generation.RAG has be-
come the standard approach for grounding LLMs in
external knowledge (Lewis et al., 2021; Gao et al.,
2023). Recent advances like Active Retrieval Aug-
mented Generation (Jiang et al., 2023) and R²AG
(Ye et al., 2024) have introduced sophisticated re-
trieval strategies, but these systems assume static
corpora without temporal or version constraints.
Recent work confirms that standard RAG fails on
outdated information (Ouyang et al., 2025), achiev-
ing only 58% accuracy on our versioned document
benchmark.
Graph-Enhanced Retrieval.GraphRAG and its
variants enhance retrieval through structured knowl-
edge representations (Zhang et al., 2025). While
2

GraphRAG excels at multi-hop reasoning, system-
atic comparisons (Han et al., 2025) show it strug-
gles with temporal dynamics. Our experiments
confirm this limitation: GraphRAG achieves only
64% overall accuracy on versioned documents and
fails catastrophically (10% accuracy) on implicit
change detection, while requiring 16× more index-
ing tokens than our approach.
Temporal and Version-Aware Approaches.Ex-
isting temporal approaches address different chal-
lenges than versioned documents. Time-Aware
Language Models (Dhingra et al., 2022) modify
LM internals to handle facts that expire, but re-
quire retraining and cannot handle discrete ver-
sion numbers. Temporal KGQA systems like
CRONKGQA (Saxena et al., 2021) answer ques-
tions over temporal knowledge graphs but assume
pre-existing temporal structures rather than extract-
ing version relationships from documents. Tem-
pRALM (Gade and Jetcheva, 2024) introduces
time-sensitive retrieval but focuses on chronolog-
ical grounding (e.g., "What happened in 2023?")
rather than version-specific retrieval (e.g., "What
changed in version 2.3?"). None of these systems
handle version-to-version transitions essential for
technical documentation.
Comparison with Existing Approaches.Table 1
summarizes how VersionRAG uniquely combines
capabilities for versioned document QA. To our
knowledge, this is the first system specifically de-
signed for retrieval-augmented generation over ver-
sioned documents, achieving 90% accuracy while
requiring 97% fewer tokens than GraphRAG. Ad-
ditional related work is discussed in Appendix A.
Table 1: Comparison of RAG approaches for version-
specific retrieval requirements. VersionRAG is the only
method that directly addresses versioned documents,
distinct from more generic temporal retrieval methods.
MethodVersion
AwarenessChange
TrackingTemporal
FilteringGraph
ReasoningHybrid
RetrievalEfficient
Indexing
Standard RAG✗ ✗ ✗ ✗ ✗ ✓
GraphRAG✗ ✗ ✗ ✓ ✓ ✗
Time-Aware LM✗ ✗ ✓ ✗ ✗N/A
CRONKGQA✗ ✗ ✓ ✓ ✗ ✗
TempRALM✗ ✗ ✓ ✗ ✗ ✓
VersionRAG✓ ✓ ✓ ✓ ✓ ✓
3 Problem and Data Structure
We first define the problem of version-aware RAG
with its three fundamental query types and then
design our data structure based on the identified
query types.3.1 Problem Formulation.
LetD={d 1, d2, ..., d n}be a collection of docu-
ments where each document diexists in multiple
versionsV di={v i,1, vi,2, ..., v i,m}. Given a query
q, the versioned document QA task requires re-
trieving the correct context cfrom the appropriate
version(s) to generate answer a. We identify three
fundamental query types that require distinct re-
trieval strategies: (1)Content Retrieval- retrieving
information from a specific version, (2)Version
Retrieval- identifying available versions or ver-
sion metadata, and (3)Change Retrieval- detecting
modifications between versions. Standard RAG sys-
tems failbecause they optimizep(c|q) withoutcon-
sideringversion constraints, while ourapproach
modelsp(c|q, v) where vrepresents version con-
text.
3.2 Version-Aware Graph Structure.
VersionRAG constructs a hierarchical graph G=
(N, E) - where Nare the nodes and Ethe edges
- during indexing to explicitly model version rela-
tionships. As shown in Figure 4, the graph con-
sists of five levels: (1)Category nodesorganize
documents into semantic groups, (2)Document
nodesrepresent unique documents with multiple
versions, (3)Version nodescapture specific docu-
ment versions with edges indicating temporal se-
quence, (4)Content nodesstore references to vector
embeddings of document chunks, and (5)Change
nodesrepresent modifications between versions,
either extracted from changelogs (explicit) or in-
ferred through difference analysis (implicit). This
structure enables both efficient graph traversal for
version-specific queries and compatibility with vec-
tor search for content retrieval.
4 VersionRAG
VersionRAG consists of three main parts: (1)In-
dexing, which creates our version-aware graph data
structure; (2)Retrieval, which implements a hy-
brid retrieval strategy, including intend detection,
retrieval mode selection and version-aware filter-
ing; (3)Generation, which contains the answer
generation for the user based on version-specific
context (see Figure 5).
4.1 Indexing Pipeline.
The indexing process transforms raw documents
into the version-aware graph through five steps.
First,attribute extractionuses an LLM to extract
3

Figure 4: Version-aware graph structure with hierarchi-
cal organization from categories to individual changes,
enabling precise version-specific retrieval.
document metadata (title, version, type) from the
first pages. Second,document clusteringgroups
versions of the same document and organizes them
into categories using semantic similarity. Third,
graph constructioncreates the hierarchical struc-
ture with proper version sequencing. Fourth,con-
tent indexingchunks documents and stores embed-
dings in a vector database with version metadata.
Finally,change extractionidentifies modifications
either from explicit changelogs or by computing
differences between consecutive versions using the
DeepDiff library (Dehpour), then employs an LLM
to generate semantic descriptions of changes.
4.2 Hybrid Retrieval Strategy.
VersionRAG employs a query-aware retrieval strat-
egy that routes queries through appropriate retrieval
paths based on their intent. The retrieval pipeline
consists of three components: (1)Query parsing
uses an LLM to classify the query type and to ex-
tract relevant parameters (category, document, ver-
sion), achieving 92% classification accuracy with
DeepSeek-R1 70B. (2)Retrieval mode selection
determines whether to use graph traversal (for ver-
sion/change queries) or vector search (for content
queries). (3)Version-aware filteringconstrains vec-
tor search to specific versions when needed, en-
suring temporal accuracy. For change retrieval,
the system can traverse the graph to find explicit
change nodes or perform semantic search over in-
dexed changes. This hybrid approach maintains
the efficiency of vector search while adding precise
version control through graph-based filtering.4.3 Generation.
After retrieval, the generation component synthe-
sizes answers using the retrieved context. The
LLM receives both the original query and version-
specific context, with explicit instructions to
ground responses in the provided information. Un-
like standard RAG where context may contain
temporally inconsistent information, VersionRAG
ensures that the retrieved context aligns with the
query’s version requirements, enabling accurate
and consistent answer generation.
Efficiency Considerations.VersionRAG
achieves significant efficiency gains compared to
GraphRAG through its structured approach. By en-
coding version relationships as graph edges rather
than extracting them through LLM analysis of en-
tire documents, VersionRAG requires 97% fewer
tokens during indexing (186K vs 2,970K tokens
with DeepSeek-R1 70B). The graph structure also
enables incremental updates—new versions can
be added without reprocessing existing documents,
making the system practical for continuously evolv-
ing documentation.
5 Dataset and Experimental Setup
To the best of our knowledge there currently does
not exist a version-aware document QA dataset. We
therefore construct a new benchmark dataset based
on publicly available documentation and change
logs of open-source software projects. Our dataset
VersionQA is publicly available for the com-
munity ( https://github.com/danielhuwiler/
versionrag)
5.1 VersionQA: A Versioned Document QA
Dataset
Dataset Construction.We create VersionQA,
the first benchmark specifically designed for evalu-
ating versioned document question answering. The
dataset comprises 100 manually curated question-
answer pairs across 34 technical documentation
files totaling over 700 pages. Documents span
four topics as shown in Table 2: Apache Spark
releases (6 changelog files), Bootstrap releases (6
changelog files), Node.js Assert module (13 doc-
umentation files across versions), and Node.js Er-
rors (9 documentation files across versions). This
mix of changelogs and versioned documentation
enables evaluation of both explicit and implicit
change detection capabilities.
4

Figure 5: VersionRAG framework with three main components: indexing constructs the version-aware graph,
retrieval routes queries through appropriate paths, and generation produces answers using retrieved context.
Query Categories.Based on our query type tax-
onomy, we design six evaluation categories to com-
prehensively test version-aware reasoning capabili-
ties. Table 3 shows the distribution and examples.
Content Retrieval(20 Q&A pairs) tests basic in-
formation extraction.Content Retrieval Complex
(20 pairs) requires synthesizing multiple pieces of
information.Content Retrieval Version-Specific
(20 pairs) tests precise version-aware retrieval (e.g.,
“What is the stability level of assert.CallTracker in
Node.js version 20.19.0?”).Version Listing & In-
quiry(20 pairs) evaluates version metadata under-
standing.Change Retrieval Explicit(10 pairs) tests
retrieval of documented changes from changelogs.
Change Retrieval Implicit(10 pairs) requires de-
tecting undocumented changes between versions
(e.g., “When was assert.deepEqual() removed?”).
5.2 Experimental Configuration
Baseline Systems.We compare VersionRAG
against two baselines representing different re-
Table 2: VersionQA dataset composition with version
ranges.
Topic Type Files Versions
Apache Spark Changelog 6 2.4.7 - 3.5.5
Bootstrap Changelog 6 5.2.3 - 5.3.5
Node.js Assert Documentation 13 11.15.0 - 23.11.0
Node.js Errors Documentation 9 15.14.0 - 23.11.0trieval paradigms.Naive RAGimplements stan-
dard retrieval-augmented generation following
(Gao et al., 2023), chunking documents into 512-
token segments and using text-embedding-3-small
for embedding. It retrieves top-5 chunks based
on cosine similarity without version awareness.
GraphRAGuses the official Neo4j implementation
(Neo4j, Inc., 2024) with HybridCypherRetriever,
combining vector search with graph traversal. It
constructs knowledge graphs by extracting entities
and relationships using LLMs but lacks explicit
version modeling.
Language Models.We evaluate across three
LLMs with varying capacities to assess robustness:
Llama 3 8B (8,192 token context), GPT-4o Mini
(128,000 token context), and DeepSeek-R1 70B
(128,000 token context). All systems use OpenAI’s
text-embedding-3-small (OpenAI, 2024) for vector
embeddings and the same LLM for both indexing
and generation within each experimental configu-
Table 3: Distribution of query categories in VersionQA.
60% of queries require version-aware reasoning.
Category Q&A Pairs Version-Sensitive
Content Retrieval 20 No
Content Retrieval Complex 20 No
Content Retrieval Version-Specific 20 Yes
Version Listing & Inquiry 20 Yes
Change Retrieval (Explicit) 10 Yes
Change Retrieval (Implicit) 10 Yes
5

ration to ensure fair comparison.
Implementation Details.VersionRAG is imple-
mented using Neo4j (Neo4j, Inc., 2025) for graph
storage and Milvus Lite (Zilliz, 2025) for vector
database operations. Documents are chunked into
512-token segments with 50-token overlap. The
DeepDiff library (Dehpour) performs line-level
comparisons for implicit change detection. Few-
shot prompting guides LLM behavior across all
tasks without fine-tuning (see Appendix E for fur-
ther details).
Evaluation Metrics.We measure accuracy as the
proportion of correctly answered questions. Each
answer is evaluated using GPT-4.1 as an LLM
judge (Gu et al., 2025), comparing generated re-
sponses against ground truth. This automated eval-
uation is manually reviewed for all 100 questions to
ensure reliability. While the dataset size precludes
extensive statistical testing, the manual curation en-
sures high-quality evaluation across diverse query
types.
5.3 Resource Considerations
All experiments were conducted using cloud APIs
with costs tracked. Table 4 shows the dramatic
efficiency difference: VersionRAG requires 186K
input tokens versus GraphRAG’s 2,970K tokens for
indexing the same documents with DeepSeek-R1
70B, translating to $0.17 versus $6.67 in API costs
and 25 minutes versus 5.2 hours processing time.
This 97% reduction in resource consumption makes
VersionRAG practical for large-scale deployment
on continuously evolving documentation.
Table 4: Indexing resource consumption with DeepSeek-
R1 70B.
Approach Tokens [K] Cost [$] Time
GraphRAG 2,970 6.67 5h 12min
VersionRAG (ours) 186 0.17 25min
6 Results and Analysis
Table 5 presents the overall accuracy across all
systems and LLMs. VersionRAG with DeepSeek-
R1 70B achieves 90% overall accuracy, signifi-
cantly outperforming both Naive RAG (58%) and
GraphRAG (64%). This 32 percentage point im-
provement over Naive RAG and 26 points over
GraphRAG demonstrates the critical importance
of version-aware retrieval. Notably, VersionRAG
maintains its advantage across all model sizes, witheven the smallest model (Llama 3 8B) achieving
comparable performance to GraphRAG’s best con-
figuration.
Table 5: Overall accuracy by query category. Version-
RAG (our approach) consistently outperforms baselines
across all configurations.
LLM Approach Overall Content Version Change
[%] Retrieval [%] Listing [%] Retrieval [%]
DeepSeek-R1 70BNaive RAG 58.0 76.6 35.0 25.0
GraphRAG 64.0 70.0 80.0 30.0
VersionRAG (ours)90.0 93.3 100.0 70.0
GPT-4o MiniNaive RAG 53.0 70.0 20.0 35.0
GraphRAG 57.0 61.6 65.0 35.0
VersionRAG (ours)79.0 80.0 90.0 65.0
Llama 3 8BNaive RAG 47.0 61.6 25.0 25.0
GraphRAG 43.0 45.0 70.0 10.0
VersionRAG (ours)55.0 61.6 70.020.0
6.1 Performance by Query Type
Content Retrieval.For standard content retrieval
without version constraints, VersionRAG achieves
93.3% accuracy with DeepSeek-R1 70B, slightly
outperforming Naive RAG (76.6%). The im-
provement stems from VersionRAG’s metadata-
enhanced retrieval, which filters out temporally
irrelevant content even when version-specificity
is not explicitly required. However, for version-
specific content retrieval, the gap widens dramat-
ically: VersionRAG achieves perfect 100% accu-
racy while Naive RAG drops to 55%. Figure 6 illus-
trates this failure mode—Naive RAG retrieves se-
mantically similar content from multiple versions,
creating ambiguity that prevents correct answers.
Figure 6: Naive RAG fails on version-specific retrieval
due to conflicting content from multiple versions.
Version Listing and Inquiry.VersionRAG
demonstrates exceptional performance on version-
related queries, achieving 100% accuracy with
DeepSeek-R1 70B compared to 35% for Naive
RAG and 80% for GraphRAG. The structured
graph representation enables direct traversal to ver-
sion nodes, providing complete and accurate ver-
sion information. Naive RAG fails because version
metadata is scattered across documents without
explicit connections. GraphRAG performs better
by capturing some version entities but lacks the
explicit version sequencing that VersionRAG pro-
vides through its hierarchical structure.
6

Change Retrieval.The most challenging cat-
egory reveals VersionRAG’s unique capabilities.
For explicit change retrieval from changelogs, Ver-
sionRAG achieves 80-90% accuracy compared to
50-60% for baselines. For implicit change detec-
tion—requiring inference of undocumented modifi-
cations—VersionRAG reaches 60% accuracy with
DeepSeek-R1 70B while both baselines essentially
fail (0-10% accuracy). This dramatic difference
stems from VersionRAG’s preprocessing step that
computes and semantically indexes differences be-
tween versions, making implicit changes directly
retrievable rather than requiring runtime inference.
6.2 Impact of Model Capacity
Model size significantly affects performance, par-
ticularly for VersionRAG. The accuracy progres-
sion from Llama 3 8B (55%) to GPT-4o Mini (79%)
to DeepSeek-R1 70B (90%) suggests that larger
models better leverage the structured graph repre-
sentation. Analysis of retrieval mode classifica-
tion reveals that DeepSeek-R1 70B correctly iden-
tifies query intent in 92% of cases versus 76% for
Llama 3 8B. This classification accuracy directly
impacts downstream performance, as misrouted
queries cannot benefit from version-aware retrieval.
6.3 Efficiency Analysis
Table 6 compares resource consumption across ap-
proaches. VersionRAG’s structured indexing re-
quires 97% fewer tokens than GraphRAG (186K
vs 2,970K) and 92% less time (25 minutes vs 5.2
hours). This efficiency gain arises from encoding
version relationships as graph edges rather than ex-
tracting them through LLM analysis. GraphRAG
generates 11,761 nodes and 39,438 relationships by
processing entire documents, while VersionRAG
creates a compact 896-node graph focused on ver-
sion structure. Despite this smaller graph, Ver-
sionRAG achieves superior accuracy through its
targeted design for versioned document retrieval.
Table 6: Efficiency comparison with DeepSeek-R1 70B.
VersionRAG achieves highest accuracy with minimal
resource consumption.
Metric Naive RAG GraphRAG VersionRAG (ours)
Index Tokens [K] 0 2,970 186
Graph Nodes 0 11,761 896
Accuracy [%] 58.0 64.0 90.0
Cost [$] 0.00 6.67 0.176.4 Error Analysis
Examining VersionRAG’s 10% error rate reveals
two primary failure modes. First, query misclassifi-
cation (8% of errors) occurs when the LLM incor-
rectly identifies query intent, particularly for am-
biguous questions that could be interpreted as either
content or change retrieval. Second, incomplete
change extraction (2% of errors) happens when sub-
tle modifications between versions are not captured
during indexing. Interestingly, GraphRAG’s fail-
ures are more systemic—it achieves only 10% accu-
racy on implicit change detection because its entity-
relationship extraction cannot capture version-to-
version transitions that are not explicitly stated in
the text.
6.5 Ablation Study
To understand component contributions, we eval-
uate VersionRAG variants with different indexing
and retrieval LLMs. Using DeepSeek-R1 70B for
indexing but Llama 3 8B for retrieval yields 68%
accuracy, demonstrating that high-quality graph
construction partially compensates for weaker re-
trieval models. Conversely, using Llama 3 8B
for indexing limits accuracy to 74% even with
DeepSeek-R1 70B retrieval, highlighting the criti-
cal importance of accurate change extraction dur-
ing indexing. These results confirm that both com-
ponents contribute to overall performance, with
indexing quality setting an upper bound on achiev-
able accuracy.
7 Discussion
Key Insights.Our results demonstrate that ver-
sioned document QA requires fundamentally differ-
ent retrieval mechanisms than standard RAG. The
32-point accuracy gap between VersionRAG and
Naive RAG cannot be bridged by simply retrieving
more context or using larger models—it stems from
the inability to distinguish between version-valid
and version-invalid information. Even GraphRAG,
despite its sophisticated entity-relationship model-
ing, achieves only marginal improvements (64% vs
58%) because version transitions are not explicit
relationships that can be extracted from text. They
must be modeled as first-class citizens in the re-
trieval system.
Practical Deployment Strategy.The correlation
between model capacity and performance (55%
to 90% accuracy) suggests a tiered deployment.
For simple version lookups and content retrieval
7

where VersionRAG maintains reasonable accuracy
(61.6%) even with Llama 3 8B, smaller models suf-
fice. Complex change detection and multi-version
reasoning tasks require larger models for accurate
query classification and change interpretation. Or-
ganizations can optimize costs by routing queries
based on complexity—a strategy our query classifi-
cation component naturally supports.
Generalizability Beyond Technical Documenta-
tion.While evaluated on technical documenta-
tion, VersionRAG’s principles apply to any domain
with discrete document versions. Legal documents
undergo formal revisions with tracked changes.
Scientific papers have preprint versions and post-
review updates. Medical guidelines and regulatory
compliance documents follow strict versioning for
liability reasons. The key requirement is identi-
fiable version markers and structural consistency
across versions. Continuous update streams (news
articles) or unstructured versioning (social media)
would require different approaches, particularly for
change detection.
Efficiency-Flexibility Trade-off.VersionRAG’s
97% reduction in indexing tokens compared to
GraphRAG represents a design philosophy: lever-
aging domain knowledge about versioning pat-
terns rather than discovering all possible relation-
ships. This specialization enables practical deploy-
ment for large document collections while main-
taining superior accuracy for version-specific tasks.
GraphRAG’s flexibility to discover unexpected re-
lationships is valuable for exploratory analysis. Fu-
ture work could explore hybrid approaches that
combine VersionRAG’s efficient version tracking
with GraphRAG’s open-ended relationship discov-
ery for comprehensive document understanding.
Implications for RAG System Design.Our
work highlights that domain-specific structures in
documents—whether versions, sections, or tempo-
ral markers—should be explicitly modeled rather
than hoping generic retrieval will capture them.
The success of purpose-built components (version-
aware graph, query classification, change detec-
tion) over general-purpose solutions suggests that
RAG systems benefit from domain-aware architec-
tures. This challenges the trend toward ever-larger,
general-purpose models and retrieval systems, ad-
vocating instead for structured approaches that en-
code domain knowledge into the system architec-
ture.8 Conclusion
We presented VersionRAG, the first retrieval-
augmented generation framework specifically de-
signed for versioned documents. By introducing a
version-aware graph structure that explicitly mod-
els document evolution and changes, VersionRAG
achieves 90% accuracy on version-sensitive ques-
tions—a 32-point improvement over standard RAG
and 26-point improvement over GraphRAG. Be-
yond effectiveness, VersionRAG demonstrates re-
markable efficiency, requiring 97% fewer tokens
during indexing than GraphRAG while maintaining
superior performance.
Our work makes three key contributions to the
field. First, we formalize the versioned document
QA task and identify three fundamental query types
that require distinct retrieval strategies, providing a
framework for future research. Second, we develop
VersionRAG’s novel graph-based architecture that
bridges the gap between efficient vector retrieval
and precise version-aware filtering. Third, we cre-
ate VersionQA, the first benchmark for evaluating
version-aware document QA systems, comprising
100 manually curated questions across 34 versioned
documents.
The implications extend beyond technical docu-
mentation. As organizations increasingly rely on
LLMs to navigate complex document repositories,
the ability to reason about document evolution be-
comes critical for applications in compliance, de-
bugging, and knowledge management. Version-
RAG’s efficient indexing and incremental update
capabilities make it practical for deployment on
continuously evolving document collections.
Future work should address three key areas: (1)
expanding evaluation to other domains such as le-
gal and medical documents, (2) developing more
sophisticated change detection mechanisms that
can capture subtle semantic shifts, and (3) explor-
ing hybrid approaches that combine VersionRAG’s
version awareness with GraphRAG’s open-ended
relationship discovery. As LLMs become more
capable, we also envision opportunities for learn-
ing version patterns from data rather than imposing
predefined structures.
8

Limitations
While VersionRAG demonstrates strong perfor-
mance on versioned document QA, several limi-
tations constrain its applicability:
Dataset Scale and Diversity.Our evaluation
uses 100 manually curated question-answer pairs
across 34 documents. While manual curation en-
sures high quality and diverse query types, this
scale limits statistical testing and fine-grained er-
ror analysis. The dataset focuses exclusively on
technical documentation (Node.js, Apache Spark,
Bootstrap). Generalization to other domains with
different versioning conventions (legal documents
with amendment structures, medical guidelines
with approval stages, scientific papers with revi-
sion rounds) remains unexplored and may require
domain-specific adaptations.
Change Detection Limitations.VersionRAG
achieves 60% accuracy on implicit change detec-
tion—our most challenging task. The DeepDiff-
based comparison may miss subtle semantic
changes that don’t manifest as clear textual dif-
ferences (e.g., a security implication from chang-
ing default values). Additionally, smaller LLMs
like Llama 3 8B produce 42% fewer change nodes
than larger models, directly linking change detec-
tion quality to model capacity. Production systems
requiring high precision for compliance or security-
critical change tracking may find this insufficient.
Version Extraction Requirements.The system
assumes documents contain extractable version in-
formation in their first pages and maintain consis-
tent structure across versions. Documents with ver-
sion information embedded in filenames only, irreg-
ular formatting, or versioning schemes that don’t
follow semantic conventions (e.g., date-based ver-
sions, code names) may fail during indexing. The
sequential processing of versions also cannot cap-
ture complex dependencies between non-adjacent
versions (e.g., a feature deprecated in v2.0 but reim-
plemented differently in v4.0).
Model and Infrastructure Dependencies.Per-
formance degrades significantly with model size:
query classification accuracy drops from 92%
(DeepSeek-R1 70B) to 76% (Llama 3 8B), directly
impacting downstream retrieval. This creates de-
ployment challenges for resource-constrained envi-
ronments. Infrastructure requirements include both
a graph database (Neo4j) and vector store (Mil-vus), increasing operational complexity compared
to pure vector-based solutions. The system also re-
quires 186K tokens for indexing our 34-document
test collection—while 97% less than GraphRAG,
this still represents substantial API costs for orga-
nizations with thousands of documents.
Temporal Reasoning Gaps.While VersionRAG
handles discrete version numbers effectively, it
cannot reason about continuous temporal aspects
within versions. Questions like "What features
were added in Q3 2023?" require mapping calendar
time to version releases, which our current system
does not support. Similarly, questions about devel-
opment velocity ("How quickly are deprecations
typically resolved?") or version lifecycle patterns
are beyond the current scope.
Error Propagation.The pipeline architecture
means errors compound: misclassified queries (8%
error rate) lead to incorrect retrieval paths, version
extraction errors affect all downstream process-
ing, and incorrect change detection permanently
impacts the graph structure. Unlike end-to-end
learned systems, these errors cannot be corrected
through backpropagation, requiring manual inter-
vention or complete reindexing to fix.
Ethical Considerations
Environmental Impact.While VersionRAG re-
duces computational costs by 97% compared to
GraphRAG, the indexing process still requires sub-
stantial LLM API calls (186K tokens for our 34-
document collection). Organizations should con-
sider the environmental impact of processing large
document repositories and explore batch process-
ing or caching strategies to minimize redundant
computations.
Fairness and Bias.VersionRAG’s performance
depends heavily on the underlying LLM’s capabili-
ties, inheriting any biases present in these models.
Our evaluation focused on English technical doc-
umentation; performance may vary significantly
for other languages or domains. The system’s 92%
query classification accuracy with large models
versus 76% with smaller models could create dis-
parities in service quality based on available com-
putational resources.
Data Privacy.Versioned documents often con-
tain sensitive information about deprecated fea-
tures, security vulnerabilities, or internal changes.
9

While VersionRAG processes documents locally af-
ter initial LLM-based indexing, organizations must
carefully consider which documents to include and
ensure appropriate access controls on the resulting
knowledge graph.
Transparency and Accountability.The sys-
tem’s 10% error rate, particularly in change de-
tection, could lead to incorrect technical decisions
if users over-rely on its outputs. We recommend
clearly communicating confidence levels and pro-
viding source attribution for all retrieved informa-
tion. The version-aware graph structure actually en-
hances accountability by maintaining clear prove-
nance for all information.
Broader Impacts.Improved access to versioned
documentation could democratize technical knowl-
edge and reduce barriers for developers working
with evolving systems. However, it might also
enable more sophisticated attacks if applied to se-
curity documentation. We encourage responsible
deployment with appropriate access controls and
monitoring.
AI Assistants In Research Or WritingWe have
used GPT-5, Gemini 2.5 pro and Claude Opus 4.1
to help us refine our writing. All final text in the
paper has been verified by the authors.
References
Omar Adjali, Olivier Ferret, Sahar Ghannay, and Hervé
Le Borgne. 2024. Multi-level information retrieval
augmented generation for knowledge-based visual
question answering. InProceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing, pages 16499–16513, Miami, Florida,
USA. Association for Computational Linguistics.
Anab Maulana Barik, Wynne Hsu, and Mong-Li Lee.
2024. Time matters: An end-to-end solution for tem-
poral claim verification. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing: Industry Track, pages 657–664,
Miami, Florida, US. Association for Computational
Linguistics.
Bin Chen, Chunjing Xiao, and Fan Zhou. 2024. Natural
evolution-based dual-level aggregation for temporal
knowledge graph reasoning. InFindings of the Asso-
ciation for Computational Linguistics: EMNLP 2024,
pages 9274–9284, Miami, Florida, USA. Association
for Computational Linguistics.
Qinyuan Cheng, Xiaonan Li, Shimin Li, Qin Zhu,
Zhangyue Yin, Yunfan Shao, Linyang Li, Tianxi-
ang Sun, Hang Yan, and Xipeng Qiu. 2024. Unified
active retrieval for retrieval augmented generation.InFindings of the Association for Computational
Linguistics: EMNLP 2024, pages 17153–17166, Mi-
ami, Florida, USA. Association for Computational
Linguistics.
Sep Dehpour. DeepDiff.
Bhuwan Dhingra, Jeremy R. Cole, Julian Martin
Eisenschlos, Daniel Gillick, Jacob Eisenstein, and
William W. Cohen. 2022. Time-aware language mod-
els as temporal knowledge bases.Transactions of the
Association for Computational Linguistics, 10:257–
273.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models.Preprint,
arXiv:2405.06211.
Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald.
2024. Trace the evidence: Constructing knowledge-
grounded reasoning chains for retrieval-augmented
generation. InFindings of the Association for Com-
putational Linguistics: EMNLP 2024, pages 8472–
8494, Miami, Florida, USA. Association for Compu-
tational Linguistics.
Anoushka Gade and Jorjeta Jetcheva. 2024. It’s
about time: Incorporating temporality in re-
trieval augmented language models.Preprint,
arXiv:2401.13222.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang,
and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun
Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni,
and Jian Guo. 2025. A survey on llm-as-a-judge.
Preprint, arXiv:2411.15594.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Ming-Wei Chang. 2020. Realm: Retrieval-
augmented language model pre-training. InInter-
national Conference on Machine Learning, pages
3929–3938.
Haoyu Han, Harry Shomer, Yu Wang, Yongjia Lei, Kai
Guo, Zhigang Hua, Bo Long, Hui Liu, and Jiliang
Tang. 2025. Rag vs. graphrag: A systematic evalua-
tion and key insights.Preprint, arXiv:2502.11371.
Palak Jain, Livio Baldini Soares, and Tom Kwiatkowski.
2024. From RAG to riches: Retrieval interlaced with
sequence generation. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 8887–8904, Miami, Florida,
USA. Association for Computational Linguistics.
10

Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing, pages 6769–6781.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR Conference, pages 39–48.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2021.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Preprint, arXiv:2005.11401.
Yading Li, Dandan Song, Changzhi Zhou, Yuhang Tian,
Hao Wang, Ziyi Yang, and Shuhao Zhang. 2024a.
A framework of knowledge graph-enhanced large
language model based on question decomposition
and atomic retrieval. InFindings of the Association
for Computational Linguistics: EMNLP 2024, pages
11472–11485, Miami, Florida, USA. Association for
Computational Linguistics.
Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei,
and Michael Bendersky. 2024b. Retrieval augmented
generation or long-context LLMs? a comprehensive
study and hybrid approach. InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing: Industry Track, pages 881–
893, Miami, Florida, US. Association for Computa-
tional Linguistics.
Kelong Mao, Zheng Liu, Hongjin Qian, Fengran Mo,
Chenlong Deng, and Zhicheng Dou. 2024. Rag-
studio: Towards in-domain adaptation of retrieval
augmented generation through self-alignment. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2024, pages 725–735, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Neo4j, Inc. 2024. Neo4j GraphRAG Python Documen-
tation. Accessed: 2025-05-02.
Neo4j, Inc. 2025. Neo4j graph database. https://
neo4j.com. Accessed: 2025-10-05.
OpenAI. 2024. New embedding models and
api updates. https://openai.com/blog/
new-embedding-models-and-api-updates .
Accessed: 2025-10-05.Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan,
Yucong Luo, Jiaying Lin, and Qi Liu. 2025. Hoh: A
dynamic benchmark for evaluating the impact of out-
dated information on retrieval-augmented generation.
Preprint, arXiv:2503.04800.
Xinying Qian, Ying Zhang, Yu Zhao, Baohang Zhou,
Xuhui Sui, Li Zhang, and Kehui Song. 2024.
TimeR4: Time-aware retrieval-augmented large lan-
guage models for temporal knowledge graph question
answering. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language Process-
ing, pages 6942–6952, Miami, Florida, USA. Associ-
ation for Computational Linguistics.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 338–354,
Mexico City, Mexico. Association for Computational
Linguistics.
Apoorv Saxena, Soumen Chakrabarti, and Partha Taluk-
dar. 2021. Question answering over temporal knowl-
edge graphs. InProceedings of the 59th Annual
Meeting of the Association for Computational Lin-
guistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long
Papers), pages 6663–6676, Online. Association for
Computational Linguistics.
Fuda Ye, Shuangyin Li, Yongqi Zhang, and Lei Chen.
2024. R²ag: Incorporating retrieval information into
retrieval augmented generation. InFindings of the
Association for Computational Linguistics: EMNLP
2024, pages 11584–11596, Miami, Florida, USA.
Association for Computational Linguistics.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu,
Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang
Wang, Dawei Yin, and Jiliang Tang. 2024. The good
and the bad: Exploring privacy issues in retrieval-
augmented generation (RAG). InFindings of the As-
sociation for Computational Linguistics: ACL 2024,
pages 4505–4524, Bangkok, Thailand. Association
for Computational Linguistics.
Qinggang Zhang, Shengyuan Chen, Yuanchen Bei,
Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan
Dong, Hao Chen, Yi Chang, and Xiao Huang. 2025.
A survey of graph retrieval-augmented generation
for customized large language models.Preprint,
arXiv:2501.13958.
Yu Zhang, Kehai Chen, Xuefeng Bai, Zhao Kang, Quan-
jiang Guo, and Min Zhang. 2024. Question-guided
knowledge graph re-scoring and injection for knowl-
edge graph question answering. InFindings of the
Association for Computational Linguistics: EMNLP
2024, pages 8972–8985, Miami, Florida, USA. Asso-
ciation for Computational Linguistics.
11

Qingfei Zhao, Ruobing Wang, Yukuo Cen, Daren Zha,
Shicheng Tan, Yuxiao Dong, and Jie Tang. 2024.
LongRAG: A dual-perspective retrieval-augmented
generation paradigm for long-context question an-
swering. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 22600–22632, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Zilliz. 2025. Milvus lite: In-process vector database for
dev and test. https://milvus.io/docs/milvus_
lite.md. Accessed: 2025-10-05.
12

A Additional Related Work
This section provides a comprehensive overview of related work beyond the concise discussion in Section
2.
A.1 Evolution of RAG Systems
The development of retrieval-augmented generation has seen significant advances beyond the foundational
work. REALM (Guu et al., 2020) pioneered retrieval-enhanced pre-training, while dense retrieval methods
(Karpukhin et al., 2020) and efficient architectures like ColBERT (Khattab and Zaharia, 2020) improved
retrieval quality.
Recent innovations have introduced more sophisticated strategies. RICHES (Jain et al., 2024) unifies
retrieval with generation by directly decoding document contents, eliminating the need for separate
retriever and generator components. Multi-Level Information RAG (Adjali et al., 2024) enhances answer
generation through entity retrieval and query expansion with joint-training loss. These advances demon-
strate the field’s progression toward more integrated and efficient architectures, though none address
version-specific challenges.
A.2 Graph-Based Retrieval Systems
Beyond GraphRAG, several systems have explored graph-enhanced retrieval. TRACE (Fang et al., 2024)
employs a KG Generator to create knowledge graphs from retrieved documents and constructs reasoning
chains for multi-hop QA, achieving up to 14.03% improvement over baseline methods. Knowledge
Graph-Enhanced LLMs (Li et al., 2024a) use question decomposition and atomic retrieval to improve
reasoning over structured knowledge. Q-KGR (Zhang et al., 2024) introduces question-guided re-scoring
to eliminate noisy pathways in retrieved subgraphs.
While these approaches demonstrate the value of graph-based reasoning, they focus on semantic
relationships rather than temporal or version-specific connections. The computational costs remain
significant—our analysis shows GraphRAG requires 11,761 nodes and 39,438 relationships for our test
corpus, while VersionRAG achieves superior accuracy with only 896 nodes.
A.3 Temporal Reasoning in NLP
Temporal reasoning has been extensively studied across different aspects. TimeR4(Qian et al., 2024)
integrates temporal knowledge from Temporal Knowledge Graphs into LLMs through a Time-aware
Retrieve-Rewrite-Retrieve-Rerank framework, achieving 47.8% and 22.5% relative gains on temporal
reasoning tasks. Time Matters (Barik et al., 2024) addresses temporal claim verification by extracting
temporal cues and applying temporal reasoning for fact-checking. Natural Evolution-based approaches
(Chen et al., 2024) model asynchronous characteristics of event evolution in temporal KGs using dual-level
aggregation.
These approaches focus on continuous temporal dimensions (dates, time periods) rather than discrete
version numbers. The distinction is crucial: version transitions in technical documentation follow semantic
versioning conventions (major.minor.patch) that encode compatibility and change severity, which temporal
reasoning systems cannot capture.
A.4 Domain Adaptation in RAG
Several recent works have explored adapting RAG to specific domains. RAG-Studio (Mao et al., 2024)
generates domain-specific training data through self-alignment, fine-tuning both LLMs and retrievers to
work cohesively for domain-specific tasks. LongRAG (Zhao et al., 2024) introduces a dual-perspective
paradigm for long-context QA, processing information at both global and local levels to handle documents
up to 100K tokens. Unified Active Retrieval (Cheng et al., 2024) addresses the challenge of when to
retrieve through multiple orthogonal criteria, achieving better retrieval timing judgments with negligible
extra inference cost.
While these systems improve domain-specific performance, they do not address the fundamental
challenge of document versioning where the same query may have different answers depending on the
version context.
13

A.5 RAG Evaluation and Analysis
The evaluation of RAG systems has become increasingly sophisticated. ARES (Saad-Falcon et al., 2024)
provides an automated evaluation framework using synthetic training data and lightweight LM judges to
assess context relevance, answer faithfulness, and answer relevance. A comprehensive study (Li et al.,
2024b) comparing RAG with long-context LLMs reveals that while long-context models achieve better av-
erage performance when sufficiently resourced, RAG maintains a significant cost advantage—particularly
relevant given VersionRAG’s 97% reduction in indexing tokens.
Privacy and security concerns have also emerged. Research on privacy issues in RAG (Zeng et al.,
2024) demonstrates vulnerabilities in leaking private retrieval databases, highlighting the need for careful
system design—an aspect we address through VersionRAG’s structured graph approach that maintains
clear data provenance.
A.6 Historical Context: Early Temporal Work
While not directly comparable to modern neural approaches, early work in temporal information retrieval
laid important foundations. Research on temporal question answering in the pre-neural era established
taxonomies of temporal questions and identified key challenges in handling time-sensitive information.
These insights inform our query classification system, though the technical approaches differ substantially
given advances in neural architectures and the specific challenge of versioned documents.
B Extended Experimental Results
B.1 Detailed Performance Breakdown
Table 7 presents the complete accuracy breakdown across all query subcategories from the thesis evalua-
tion.
Table 7: Detailed accuracy breakdown. CR: Content Retrieval, CR-C: Content Retrieval Complex, CR-VS: Content
Retrieval Version-Specific, VLI: Version Listing & Inquiry, Ch-E: Change Explicit, Ch-I: Change Implicit.
LLM Approach CR CR-C CR-VS VLI Ch-E Ch-I
[%] [%] [%] [%] [%] [%]
DeepSeek-R1 70BBaseline 95.0 80.0 55.0 35.0 50.0 0.0
GraphRAG 85.0 80.0 100.0 80.0 50.0 10.0
VersionRAG 100.0 80.0 100.0 100.0 80.0 60.0
GPT-4o MiniBaseline 85.0 75.0 50.0 20.0 60.0 10.0
GraphRAG 70.0 65.0 50.0 65.0 60.0 10.0
VersionRAG 80.0 90.0 70.0 90.0 90.0 40.0
Llama 3 8BBaseline 65.0 80.0 40.0 25.0 50.0 0.0
GraphRAG 70.0 25.0 40.0 70.0 20.0 0.0
VersionRAG 65.0 80.0 40.0 70.0 30.0 10.0
B.2 Query Classification Performance
The accuracy of query intent classification directly impacts VersionRAG’s routing mechanism:
Table 8: Query classification accuracy showing smaller models struggle with change retrieval identification.
LLM Overall [%] Content Retrieval [%] Version Listing [%] Change Retrieval [%]
DeepSeek-R1 70B 92.0 93.3 90.0 90.0
GPT-4o Mini 91.0 95.0 85.0 85.0
Llama 3 8B 76.0 73.3 95.0 65.0
14

B.3 Resource Consumption Comparison
Table 9 shows the dramatic efficiency difference between approaches:
Table 9: Resource consumption showing VersionRAG’s 97% reduction in tokens and 92% reduction in processing
time.
LLM Approach Input [K] Output [K] Cost [$] Time
DeepSeek-R1 70BGraphRAG 2,970 5,922 6.67 5h 12min
VersionRAG 186 38 0.17 25min
GPT-4o MiniGraphRAG 2,970 1,631 1.42 2h 45min
VersionRAG 184 54 0.07 20min
Llama 3 8BGraphRAG 2,970 2,541 0.35 2h 11min
VersionRAG 194 32 0.01 24min
B.4 Graph Structure Statistics
Comparison of graph structures generated by different approaches shown in Table 10.
Table 10: Graph statistics showing VersionRAG’s compact structure (896 nodes) versus GraphRAG’s expansive
graph (11,761+ nodes).
VersionRAGTotal Nodes Differ Nodes Extract Nodes Relationships
DeepSeek-R1 70B 896 146 642 922
GPT-4o Mini 893 142 643 919
Llama 3 8B 776 85 584 801
GraphRAGTotal Nodes Relationships
DeepSeek-R1 70B 11,761 39,438
GPT-4o Mini 13,236 31,894
Llama 3 8B 11,231 24,538
B.5 Cross-Model Indexing and Retrieval
Complete results when using different LLMs for indexing versus retrieval listed in Table 11.
Table 11: Cross-model performance showing self-combinations achieve highest accuracy.
Indexing LLM Retrieval LLM Overall [%] Content [%] Version [%] Change [%]
Llama 3 8B Llama 3 8B 55.0 61.6 70.0 20.0
Llama 3 8B GPT-4o Mini 67.0 65.0 75.0 65.0
Llama 3 8B DeepSeek-R1 70B 74.0 76.6 85.0 55.0
GPT-4o Mini Llama 3 8B 54.0 51.6 90.0 25.0
GPT-4o Mini GPT-4o Mini 79.0 80.0 90.0 65.0
GPT-4o Mini DeepSeek-R1 70B 79.0 83.3 90.0 55.0
DeepSeek-R1 70B Llama 3 8B 68.0 75.0 60.0 55.0
DeepSeek-R1 70B GPT-4o Mini 78.0 76.6 95.0 65.0
DeepSeek-R1 70B DeepSeek-R1 70B 90.0 93.3 100.0 70.0
15

B.6 Change Retrieval Breakdown
Detailed performance on explicit versus implicit change detection in Table 12.
Table 12: Change retrieval showing implicit changes are significantly harder to detect.
Indexing LLM Retrieval LLM Explicit [%] Implicit [%]
Llama 3 8B Llama 3 8B 30.0 10.0
Llama 3 8B GPT-4o Mini 90.0 40.0
Llama 3 8B DeepSeek-R1 70B 70.0 40.0
GPT-4o Mini Llama 3 8B 50.0 0.0
GPT-4o Mini GPT-4o Mini 90.0 40.0
GPT-4o Mini DeepSeek-R1 70B 100.0 10.0
DeepSeek-R1 70B Llama 3 8B 80.0 30.0
DeepSeek-R1 70B GPT-4o Mini 80.0 50.0
DeepSeek-R1 70B DeepSeek-R1 70B 80.0 60.0
B.7 GraphRAG with Alternative Indexing
Testing whether GraphRAG’s limitations stem from indexing or retrieval in Table 13.
Table 13: GraphRAG performance showing marginal improvement (+4%) with better indexing.
Indexing LLM Retrieval/Generation LLM Overall Accuracy [%]
Llama 3 8B Llama 3 8B 43.0
DeepSeek-R1 70B Llama 3 8B 47.0
B.8 Document Coverage
The dataset covers four main documentation sources listed in Table 14.
Table 14: Document coverage in VersionQA dataset.
Topic Type Files Pages
Apache Spark Releases Changelog 6 150
Bootstrap Releases Changelog 6 120
Node.js Assert Module Documentation 13 260
Node.js Errors Documentation 9 180
Total 34 710
16

C Additional Ablation Studies
C.1 Cross-Model Indexing and Retrieval Performance
Table 15 shows the complete results when using different LLMs for indexing versus retrieval/generation,
revealing how indexing quality affects downstream performance.
Table 15: Complete cross-model performance matrix showing that self-combinations generally achieve highest
accuracy.
Indexing Retrieval Overall Content Version Change
LLM LLM [%] Retrieval [%] Listing [%] Retrieval [%]
Llama 3 8B Llama 3 8B 55.0 61.6 70.0 20.0
Llama 3 8B GPT-4o Mini 67.0 65.0 75.0 65.0
Llama 3 8B DeepSeek-R1 70B 74.0 76.6 85.0 55.0
GPT-4o Mini Llama 3 8B 54.0 51.6 90.0 25.0
GPT-4o Mini GPT-4o Mini 79.0 80.0 90.0 65.0
GPT-4o Mini DeepSeek-R1 70B 79.0 83.3 90.0 55.0
DeepSeek-R1 70B Llama 3 8B 68.0 75.0 60.0 55.0
DeepSeek-R1 70B GPT-4o Mini 78.0 76.6 95.0 65.0
DeepSeek-R1 70B DeepSeek-R1 70B 90.0 93.3 100.0 70.0
C.2 Detailed Change Retrieval Performance
Table 16 breaks down change retrieval into explicit and implicit categories across model combinations.
Table 16: Change retrieval performance showing explicit changes are easier to retrieve than implicit ones.
Indexing LLM Retrieval LLM Change Explicit [%] Change Implicit [%]
Llama 3 8B Llama 3 8B 30.0 10.0
Llama 3 8B GPT-4o Mini 90.0 40.0
Llama 3 8B DeepSeek-R1 70B 70.0 40.0
GPT-4o Mini Llama 3 8B 50.0 0.0
GPT-4o Mini GPT-4o Mini 90.0 40.0
GPT-4o Mini DeepSeek-R1 70B 100.0 10.0
DeepSeek-R1 70B Llama 3 8B 80.0 30.0
DeepSeek-R1 70B GPT-4o Mini 80.0 50.0
DeepSeek-R1 70B DeepSeek-R1 70B 80.0 60.0
C.3 GraphRAG Performance with Different Indexing Models
As noted in Section 4.2.3, we tested whether GraphRAG’s poor performance with Llama 3 8B stemmed
from indexing or retrieval limitations.
Table 17: GraphRAG performance showing that better indexing provides only marginal improvement (+4%) when
paired with a weak retrieval model.
Indexing LLM Retrieval/Generation LLM Overall Accuracy [%]
Llama 3 8B Llama 3 8B 43.0
DeepSeek-R1 70B Llama 3 8B 47.0
17

C.4 Query Classification Accuracy by Model
The accuracy of retrieval mode derivation directly impacts VersionRAG’s performance as shown in Table
18.
Table 18: Query intent classification accuracy showing that smaller models struggle particularly with change
retrieval queries.
LLM Overall [%] Content Retrieval [%] Version Listing [%] Change Retrieval [%]
DeepSeek-R1 70B 92.0 93.3 90.0 90.0
GPT-4o Mini 91.0 95.0 85.0 85.0
Llama 3 8B 76.0 73.3 95.0 65.0
C.5 Graph Structure Complexity by Model
Different LLMs produce varying graph structures during indexing as shown in Table 19.
Table 19: Graph structure statistics showing VersionRAG’s focused approach versus GraphRAG’s comprehensive
entity extraction.
VersionRAG GraphsTotal Nodes Differ Nodes Extraction Nodes Relationships
DeepSeek-R1 70B 896 146 642 922
GPT-4o Mini 893 142 643 919
Llama 3 8B 776 85 584 801
GraphRAG GraphsTotal Nodes Entity Nodes Relationships
DeepSeek-R1 70B 11,761 11,761 - 39,438
GPT-4o Mini 13,236 13,236 - 31,894
Llama 3 8B 11,231 11,231 - 24,538
D Qualitative Analysis
D.1 Retrieval Process Examples
We illustrate the qualitative differences between approaches through concrete examples from our experi-
ments.
D.1.1 Version-Specific Content Retrieval
When asked "What is the stability level of the assert.partialDeepStrictEqual method in Node.js version
23.11.0?", the baseline RAG system retrieves semantically similar chunks from multiple versions:
• "assert.partialDeepStrictEqual method is 1_2 - Release candidate" (v23.11.0)
• "assert.partialDeepStrictEqual method is 1_0 - Early development" (v22.14.0)
This conflicting information prevents correct answer generation. In contrast, VersionRAG:
1. Classifies the query as version-specific content retrieval
2. Extracts the version parameter (23.11.0)
3. Constrains vector search to that specific version
4. Returns only the correct information
D.1.2 Change Detection
VersionRAG successfully identifies both explicit and implicit changes:
Explicit (from changelog):"Upgraded Avro to version 1.11.4" (Apache Spark 3.5.5)
Implicit (from diff analysis):"Added new method assert.partialDeepStrictEqual" (detected between
Node.js 21.7.3→22.14.0)
GraphRAG fails on implicit changes (10% accuracy) because its entity-relationship extraction cannot
capture transitions not explicitly stated in text.
18

D.2 Graph Structure Analysis
VersionRAG’s focused graph structure enables efficient retrieval as shown in Table 20.
Table 20: Resource comparison showing VersionRAG’s 13× smaller graph achieves 26 points higher accuracy.
Component VersionRAG GraphRAG
Total Nodes 896 11,761
Relationships 922 39,438
Processing Tokens 186K 2,970K
Indexing Time 25 min 312 min
D.3 Failure Mode Analysis
Analysis of errors reveals distinct patterns:
VersionRAG Failures (10% of queries).
• Query misclassification (8%): Ambiguous queries incorrectly routed
• Incomplete change extraction (2%): Subtle modifications missed by diff analysis
GraphRAG Failures (36% of queries).
• Version confusion (20%): Cannot distinguish between temporally different states
• Missing change detection (16%): No mechanism for implicit change identification
Baseline RAG Failures (42% of queries).
• Version mixing (25%): Retrieves conflicting information from multiple versions
• No change awareness (17%): Cannot identify modifications between documents
D.4 Model Capacity Effects
Smaller models show degraded performance in specific areas:
•Change extraction: Llama 3 8B generates 42% fewer change nodes than DeepSeek-R1 70B
•Query classification: 24% misclassification rate with Llama 3 8B versus 8% with DeepSeek-R1
70B
•Context integration: Limited 8,192 token window constrains complex reasoning
D.5 Efficiency-Performance Trade-off
VersionRAG achieves superior accuracy while requiring 97% fewer resources than GraphRAG. This
efficiency stems from encoding version relationships as explicit graph edges rather than discovering them
through expensive LLM analysis. The focused graph structure (896 nodes) provides better signal-to-noise
ratio than GraphRAG’s comprehensive entity graph (11,761 nodes), demonstrating that domain-specific
design outperforms general-purpose approaches for specialized tasks.
19

E Implementation Details
E.1 System Architecture and Tools
VersionRAG was implemented using the following tools as specified in the experimental setup:
•Llama 3 8B: Transformer-based model with 8 billion parameters, 8,192 token context window
•DeepSeek-R1 70B: Distilled LLaMA-based model with 70 billion parameters, 128,000 token
context window
•GPT-4o Mini: Fast and cost-efficient model (context window undisclosed)
•GPT-4.1: Used specifically for LLM-as-a-judge evaluation
•text-embedding-3-small: OpenAI embedding model for all text representations
•Milvus Lite v2.4.11: Vector database for storing embeddings
•Neo4j v1.6.1: Locally hosted graph database for VersionRAG implementation
•Neo4j AuraDB: Cloud-based graph database used for GraphRAG
E.2 Document Processing
Input Formats.The system supports both PDF and Markdown files as input. All PDF documents are
converted to Markdown format to preserve structural information while enabling text processing.
Chunking Configuration.Documents are segmented into:
•Chunk size: 512 tokens
•Chunk overlap: 50 tokens
•Purpose: Balances context preservation with model token limits
E.3 Indexing Process Details
Attribute Extraction.Each document’s first page is processed by an LLM to extract:
• Topic title
• Brief content summary
• Version information
At least the first ten pages are analyzed to determine whether the document is a changelog or general
documentation.
Document Clustering.Documents are clustered using an LLM instructed to group based on:
• Title similarity for version grouping
• Topic similarity for category assignment
Change Detection.Two methods are employed:
•Explicit changes: Extracted from documents identified as changelogs using LLM analysis
•Implicit changes: Detected using the DeepDiff library for line-level comparison between consecutive
versions, with LLM interpretation of the differences
20

E.4 LLM Configuration
Prompting Strategy.All LLM tasks use few-shot prompting without fine-tuning. Example input-output
pairs guide the model’s behavior for:
• Query classification into three categories (VersionRetrieval, ChangeRetrieval, ContentRetrieval)
• Parameter extraction (category, document, version)
• Change interpretation from diff outputs
API Access.
• GPT-4o Mini: Accessed via OpenAI API
• Llama 3 8B and DeepSeek-R1 70B: Accessed via GroqCloud
E.5 Retrieval Configuration
Query Classification.User queries are classified into three retrieval modes with the following accuracy:
• DeepSeek-R1 70B: 92% classification accuracy
• GPT-4o Mini: 91% classification accuracy
• Llama 3 8B: 76% classification accuracy
Retrieval Strategies by Query Type.
•VersionRetrieval: Graph traversal to version nodes
•ChangeRetrieval: Semantic search over change nodes or direct graph access
•ContentRetrieval: Vector similarity search with optional version filtering
21

E.6 Baseline RAG Implementation
Following standard RAG methodology:
• Documents chunked into 512-token segments
• Embeddings generated using text-embedding-3-small
• Top-5 chunks retrieved based on cosine similarity
• No version awareness or metadata filtering
The baseline implementation serves as a minimal, naive RAG setup that enables direct comparison with
more advanced approaches. Each document is split into fixed-size chunks, embedded, and stored in a
vector database for semantic retrieval. At query time, the top-ksemantically similar chunks are retrieved
purely based on cosine similarity, without the use of metadata or contextual filtering.
The overall structure of this baseline pipeline is illustrated in Figure 7. This setup establishes a
consistent foundation for evaluating the benefits of graph-based retrieval.
Figure 7: Baseline Naive RAG Framework
22

E.7 GraphRAG Implementation
Using the official Neo4j GraphRAG implementation:
• HybridCypherRetriever for combining vector search with graph traversal
• Default parameters as specified in Neo4j documentation
• Entity and relationship extraction performed by LLMs during indexing
The GraphRAG system extends the baseline approach by introducing structured knowledge graph
representations. During indexing, entities and their relationships are extracted from text using LLMs and
stored as nodes and edges in a Neo4j database. This process is visualized in Figure 8, which outlines the
main components of the GraphRAG pipeline.
Figure 8: GraphRAG Framework
The retrieval process is handled by theHybridCypherRetriever, which combines vector-based semantic
retrieval with graph traversal. Retrieved nodes can be expanded through connected relationships (e.g.,
:MENTIONS ,:RELATED_TO ) to gather additional, contextually relevant information. This hybrid process,
depicted in Figure 9, allows the system to integrate both implicit semantic similarity and explicit relational
context.
Figure 9: HybridCypherRetrieval process combining semantic similarity with graph relationships.
E.8 Resource Tracking
All experiments tracked:
• Input and output token counts
• API costs (OpenAI and GroqCloud pricing)
• Processing time for each phase
• Generated graph sizes (nodes and relationships)
The implementation and dataset are publicly available at: https://github.com/danielhuwiler/
versionrag
23

F Dataset Examples
F.1 Sample Questions and Answers by Category
We present representative examples from each of the six evaluation categories in VersionQA, taken directly
from our dataset.
F.1.1 Content Retrieval (Standard)
Q1:What are the four categories of errors that Node.js applications generally experience?
A:The four categories are: Standard JavaScript errors, system errors, user-specified errors, and Assertion-
Errors.
Q2:What is the purpose of the error.code property in Node.js errors?
A:The error.code property is a string label that identifies the kind of error and is the most stable way to
identify an error across Node.js versions.
Q3:What was fixed in Apache SPARK-31967?
A:SPARK-31967 fixed the issue where loading the jobs UI page took 40 seconds.
F.1.2 Content Retrieval Complex
Q1:What is the difference between assert.deepEqual() and assert.deepStrictEqual() in Node.js?
A:’deepEqual()’ allows coercion (uses ’==’), while ’deepStrictEqual()’ uses ’Object.is()’ for strict
comparison; the latter is more reliable.
Q2:How does assert.throws() validate errors in Node.js?
A:It checks that a function throws an error matching a specified class, RegExp, validator function, or
object; throws AssertionError if validation fails.
Q3:What happens if no ’error’ event handler is provided for an EventEmitter in Node.js?
A:The process crashes unless an ’uncaughtException’ handler is registered, making proper ’error’
handling essential.
F.1.3 Content Retrieval Version-Specific
Q1:What is the stability level of the assert.CallTracker in Node.js version 20.19.0?
A:Stability: 0 - Deprecated
Q2:What is the stability level of the assert.partialDeepStrictEqual method in Node.js version 23.11.0?
A:In Node.js v23.11.0, the stability level of the assert.partialDeepStrictEqual method is 1_2 - Release
candidate.
Q3:What type of release is Apache Spark 2.4.7?
A:Spark 2.4.7 is a maintenance release containing stability, correctness, and security fixes.
F.1.4 Version Listing & Inquiry
Q1:What is the latest NodeJs version you know of?
A:23.11.0
Q2:What Apache Spark versions are available?
A:Version 2.4.7, Version 3.3.4, Version 3.4.4, Version 3.5.3, Version 3.5.4, Version 3.5.5.
Q3:Does NodeJs version 21.7.3 exist in the system?
A:Yes
Q4:How many Bootstrap versions are you aware of?
A:5.2.3, 5.3.1, 5.3.2, 5.3.3, 5.3.4, 5.3.5. That makes a total of six versions.
F.1.5 Change Retrieval (Explicit)
Q1:What dependency was upgraded in Apache Spark 3.5.5?
A:Avro was upgraded to version 1.11.4 in Spark 3.5.5.
Q2:What change was made to the way badges handle text readability in Bootstrap v5.3.3?
A:Badges now use the .text-bg-* text utilities to ensure that the text is always readable, especially when
customized colors differ in light and dark modes.
24

Q3:What was fixed in the selector engine in Bootstrap v5.3.3?
A:A regression in the selector engine that wasn’t able to handle multiple IDs anymore was fixed.
F.1.6 Change Retrieval (Implicit)
Q1:When was the assert method partialDeepStrictEqual added to Node.js Assert?
A:The assert.partialDeepStrictEqual method was added to Node.js in version 22.14.0.
Q2:With what Node.js version was error code ERR_ACCESS_DENIED added?
A:The error code ERR_ACCESS_DENIED was added in Node.js version 16.20.2.
Q3:Was the CallTracker object introduced as new feature in Node.js Assert version 14.21.3?
A:Yes, the CallTracker object was introduced as a new feature in Node.js Assert version 14.21.3.
F.2 Dataset Statistics
Based on the dataset composition described in Section 5.
Table 21: Dataset distribution showing 60% of queries require version-aware reasoning.
Category Q&A Pairs Version-Sensitive
Content Retrieval 20 No
Content Retrieval Complex 20 No
Content Retrieval Version-Specific 20 Yes
Version Listing & Inquiry 20 Yes
Change Retrieval (Explicit) 10 Yes
Change Retrieval (Implicit) 10 Yes
Total 100 60%
25