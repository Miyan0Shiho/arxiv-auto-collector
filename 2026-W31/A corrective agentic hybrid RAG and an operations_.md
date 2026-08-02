# A corrective agentic hybrid RAG and an operations-grounded evaluation for a scientific facility

**Authors**: Rajat Sainju, Dariusz Jarosz, Hairong Shang, Michael Prince, Ryan M. Aydelott, Mathew J. Cherukara, Yine Sun, Michael D. Borland

**Published**: 2026-07-27 17:01:30

**PDF URL**: [https://arxiv.org/pdf/2607.24663v1](https://arxiv.org/pdf/2607.24663v1)

## Abstract
Scientific user facilities accumulate decades of operational knowledge that no single search index covers: electronic logbooks, technical documents, internal wikis, operations chat messages, maintenance records, and live control-system data. We present APS-RAG, Advanced Photon Source Retrieval Augmented Generation, a deployed platform that makes the institutional knowledge at the Advanced Photon Source (APS) accessible to staff through natural-language queries, along with an operations-grounded evaluation. The retrieval engine fuses dense, sparse, and knowledge-graph (KG) channels with query-type-adaptive reciprocal-rank fusion, adds a corrective agentic loop, and runs a native-tool ReAct executor over a Model Context Protocol (MCP) tooling layer. We construct APS-Bench, a 50-question, question-answering (QA) dataset with auditable gold answers. Every retrieval-augmented variant numerically improves strict vital-nugget recall over a naive BM25 baseline (63.8%), with the full corrective Agentic GraphRAG scoring (70.3%). The cross-encoder reranker contributes significantly to answer quality: removing it and allowing the LLM to score relevance drastically reduces strict vital recall by 32.8%. The graph channel and corrective loop contribute positively as expected, but the performance gains are marginal. Additionally, we also compare the performance of open-source and closed-source LLMs in final answer synthesis. We release the APS-Bench construction methodology, the six-layer evaluation harness, and the underlying codebase, along with the '/aps-rag' retrieval agent skill framework, to support reproduction and adoption at other facilities. Together, the deployed platform and its operations-grounded evaluation present a promising workflow for trustworthy, statistically grounded AI assistance in facility operations, transferable to other large scientific instruments.

## Full Text


<!-- PDF content starts -->

APS-RAG
A corrective agentic hybrid RAG and an operations-grounded
evaluation for a scientific facility
Rajat Sainju,1,a)Dariusz Jarosz,1Hairong Shang,1Michael Prince,1Ryan M. Aydelott,1Mathew J. Cherukara,1
Yine Sun,1and Michael Borland1
Advanced Photon Source, Argonne National Laboratory, Lemont, Illinois 60439, USA
(Dated: 28 July 2026)
Scientific user facilities accumulate decades of operational knowledge that no single search index covers: electronic
logbooks, technical documents, internal wikis, operations chat messages, maintenance records, and live control-system
data. We present APS-RAG—Advanced Photon Source Retrieval Augmented Generation—a deployed platform that
makes the institutional knowledge at the Advanced Photon Source (APS) accessible to staff through natural-language
queries, along with an operations-grounded evaluation. The retrieval engine fuses dense, sparse, and knowledge-graph
(KG) channels with query-type-adaptive reciprocal-rank fusion, adds a corrective agentic loop, and runs a native-tool
ReAct executor over a Model Context Protocol (MCP) tooling layer. We construct APS-Bench, a 50-question, question-
answering (QA) dataset with auditable gold answers. Every retrieval-augmented variant numerically improves strict
vital-nugget recall over a naive BM25 baseline (63.8%→65.5–70.3%), with the full corrective Agentic GraphRAG
scoring (70.3%). The cross-encoder reranker contributes significantly to the answer quality: removing it and allowing
the LLM to score the relevance drastically reduces strict vital recall by 32.8% (95% CI[−47.4,−19.1];p<10−4).
The graph channel and corrective loop contribute positively as expected, but the performance gains are marginal. Ad-
ditionally, we also compare the performance of open-source and closed-source LLMs in final answer synthesis. We
release the APS-Bench construction methodology, the six-layer evaluation harness, and the underlying codebase, along
with the/aps-ragretrieval agent skill framework, to support reproduction and adoption at other facilities. Together, the
deployed platform and its operations-grounded evaluation present a promising workflow for trustworthy, statistically
grounded AI assistance in facility operations, transferable to other large scientific instruments.
I. INTRODUCTION
Large-scale scientific user facilities operate continuously
and accumulate operational knowledge in forms that are usu-
ally not available through a unified search interface. These
include scientific logbooks, internal technical documentation,
manuals, policy documents, maintenance work requests, oper-
ations chat messages, downtime records, internal group wikis,
and live control-system process variables (PVs). For an X-ray
light source such as the Advanced Photon Source (APS)1,2,
which operates over decades, the volume of knowledge re-
quired for operation, maintenance, and improvement accumu-
lates at a rate that exceeds the capacity of any individual to
absorb it. Much of the knowledge is tacit, residing within
the records of specialized groups (e.g., Controls, Diagnostics,
Radio Frequency, Magnetics, Beamline Controls and Data
Acquisition). For example, a beam-loss event may be doc-
umented by a operations staff in BELY , the APS’s scientific
logbook3; the subsystem fault that caused it may have been
filed in a maintenance work request days earlier; the resolu-
tion lives in an operator’s wiki; the subsystem specification
is a technical document in the Integrated Content Manage-
ment System (ICMS); and the relevant setpoint is a live pro-
cess variable accessed via the archiving system like EPICS4.
This fragmentation imposes high costs. When solutions to re-
curring problems cannot be found quickly, downtime length-
ens; when experienced staff members rotate off shift or retire,
the reasoning behind past decisions can leave with them. Ef-
a)Author to whom correspondence should be addressed: rsainju@anl.govfective knowledge management is therefore essential for min-
imizing downtime and preserving institutional knowledge at
large-scale scientific facilities.
Different research groups at APS require tailored access
to operational knowledge. For example, scientists preparing
machine studies require provenance information and config-
uration details of previous measurements. Engineers design-
ing subsystem upgrades depend on controlling specifications
and the history of related modifications. Technicians diag-
nosing faults at 3 a.m. need to know how the same symp-
tom was resolved the last three times it occurred. Although
such information is recorded, locating it necessitates famil-
iarity with the relevant systems and their query mechanisms.
A large language model (LLM) interface could address this
challenge by enabling natural language queries5,6; however,
an LLM alone lacks facility-specific knowledge and may gen-
erate fluent yet unsupported responses, which is unsuitable
for operational decision-making. Retrieval-augmented gener-
ation (RAG) addresses this limitation by grounding responses
in passages retrieved from facility records rather than relying
solely on LLM’s memory7,8. This approach supplies essential
operations knowledge, enhances factual accuracy, and enables
source citation for each LLM-synthesized answer. Thus, min-
imizing unsupported or speculative claims.
In this work, we present APS-RAG (Fig. 1), a deployed
agentic knowledge-retrieval platform that transforms natu-
ral language queries from staff into a source-cited answer
drawn from the facility’s dispersed databases. To evaluate the
system’s performance, we constructed a benchmark dataset,
APS-Bench, using a methodology similar to InPars9. This en-
ables us to generate question-and-answer pairs from the oper-
ational corpus and to go beyond the usually reported recall-
arXiv:2607.24663v1  [physics.acc-ph]  27 Jul 2026

APS-RAG 2
FIG. 1. APS-RAG deployment overview. Through a single natural-language interface, APS staff can query the facility’s dispersed operational
record. The platform pools information from nine distinct APS data sources: eight indexed knowledge bases are kept current by a daily
diff-and-upsert pipeline, and the EPICS Archiver is not pre-indexed and is accessed at query time for live and historical process-variable
data through the Model Context Protocol (MCP) tooling layer. Three user-selectable query modes are available:APS-RAG (Fast)for quicker
responses,APS-RAG (Thinking), which enables the corrective agentic loop, andAI Troubleshoot, which retrieves knowledge-graph-targeted
fault-resolution steps.
style or qualitative results. During the last two-month de-
ployment, the system processed an average of eight queries
per day. The system was iteratively improved based on user
feedback. The primary contribution of this study is the op-
erational integration of established techniques, including hy-
brid retrieval that combines dense10, sparse11, and knowledge
graph12retrieval; adaptive reciprocal-rank fusion (RRF)13,
corrective retrieval14, a persistent Model Context Protocol
(MCP) layer that combines retrieved documents with live fa-
cility data and generates code for statistics-related queries
(e.g., "How many work requests entered in year 2026 exist
for kickers?"), and nugget-based evaluation15. We also report
controlled ablations of each component. We find that agentic
RAG shows great promise for fast, accurate, and reproducible
actionable knowledge retrieval in operational settings.
II. RELATED WORK
APS-RAG sits at the intersection of three rapidly develop-
ing lines.General RAG.We build on retrieval-augmented
generation16,17with dense retrieval10, BM2511, reciprocal-
rank fusion13, and cross-encoder reranking18; on knowledge-
graph-augmented RAG12,19,20(surveyed in Ref.21; Ref.22an-
alyzes when graph retrieval helps); and on agentic and self-
correcting RAG — corrective RAG14, Self-RAG23, Adaptive-
RAG24, and the ReAct paradigm25(surveyed in Ref.26) —
over a Model-Context-Protocol tooling layer27. We evalu-
ate with nugget recall15, a claim-level faithfulness metric in-
spired by FaithJudge28with an HHEM-2.1 backstop29, and a
RAGAS-style proxy30, with benchmark questions generated
following the InPars9method. Similar automated self-datacuration has been shown to be effective for adapting and eval-
uating RAG systems in a new domain31.
Accelerator and scientific-facility RAG.Machine learn-
ing methods are now well established for accelerator con-
trol and diagnostics, as documented in recent lterature32–38.
More recently, large language models have been applied
to facility operations, with several systems pairing an
LLM with document retrieval and tool integration. No-
table examples include CALMS at Argonne39, GAIA at
DESY/European XFEL40, a RAG summarization agent for
the Electron-Ion Collider41, a multi-facility logbook-RAG ef-
fort42with companion logbook-retrieval and analytics stud-
ies43, and a RAG demo for technical documentation44. Rel-
ative to all of these, APS-RAG adds query-type-adaptive
dense+sparse+knowledge-graph fusion, a corrective loop with
a knowledge-graph-grounding trigger and a self-critique gate,
and — most importantly — a domain-specific operational
evaluation, where prior facility systems report recall-style or
qualitative results. A second, adjacent line targets control
rather than knowledge access: agentic experiment execution
at the Advanced Light Source45, agents that operate scien-
tific instruments46, the agentic-control vision of Ref.47, and
natural-language tuning48. We share the plan-first, bounded-
tool design philosophy of Ref.45, but answer questions and
read instruments rather than steer the machine. The near-
est methodological analog, agentic hybrid RAG for muon-
collider analysis49, likewise combines hybrid retrieval with
agentic query decomposition and a benchmark that carries
key points and unanswerable questions, but targets the offline
high-energy-physics literature. The systems above do not con-
tain a knowledge-graph fault-chain channel and do not report
claim-level faithfulness under cross-family judging, which is

APS-RAG 3
TABLE I. Composition of the APS-RAG retrieval corpus. Each row
corresponds to one institutional data source ingested and stored as an
indexed collection. Character counts reflect raw UTF-8 text extracted
after document preprocessing; token counts are computed with the
GPT-4/cl100k byte-pair-encoding tokenizer.
CollectionUnique documents
(ct)Characters
(ct)Tokens
(GPT-4/cl100k)
BELY 4.5k 14.0M 4.2M
ICMS 5.8k 72.0M 20.6M
Work Requests 17.3k 3.0M 0.8M
Teams Chat 9.4k 2.4M 0.8M
Internal Wikis 0.2k 0.8M 0.1M
Beam Downtime 103.0k 7.9M 1.9M
SDDS 0.6k 1.4M 0.4M
OAG Apps 0.7k 0.8M 0.2M
part of what we show in this work. To our knowledge, APS-
RAG is the first accelerator-operations RAG system to add
cross-database search, allowing users to access multiple insti-
tutional knowledge bases through one natural language inter-
face.
III. THE APS-RAG PLATFORM AND SYSTEM
ARCHITECTURE
A. APS databases
APS-RAG connects to nine databases that collectively en-
compass the facility’s written operational knowledge. An
overview of these connected sources is shown in Fig. 1. Five
of these databases correspond to live operational systems
that staff interact with daily.BELY3serves as APS’s sci-
entific logbook and maintains a time-ordered record of shift
entries, beam-loss events, machine experiments, and trou-
bleshooting notes. TheIntegrated Content Management Sys-
tem (ICMS)functions as the facility’s controlled-document
repository for procedures, specifications, technical notes, and
reports. TheWork Request systemtracks maintenance and
equipment-replacement jobs from request through comple-
tion; the operationalMicrosoft Teams chat historycaptures
the routine operational communication among operators, en-
gineers, and scientists, and theinternal wikiswhich are group-
and subsystem-maintained reference web pages.) Three ad-
ditional sources provide supporting documentation: theMCR
beam downtime records(a curated, structured record of down-
time events and their descriptions, complementary to the
BELY entries), theSDDS documentationset53(current SDDS
user guides), and theOAG codebase and application docu-
mentation, which describes operator-facing graphical user in-
terface and tool descriptions. The ninth source, the EPICS
Archiver, contains live and historical process-variable data.
This source is accessed at query time through the Model-
Context-Protocol tooling layer, as its data are continuous and
timestamped rather than document-based, and thus are not
pre-indexed. Table I summarizes the document, character, and
token counts for each source.B. Data ingestion and indexing
Database-backed sources such as BELY , ICMS, Work Re-
quests, and Beam downtime records are pulled automatically
via REST APIs and export libraries. The APS wikipage con-
tents are acquired using a dedicated web-crawler that traverses
the internal wiki’s page-and-attachment graph, removing nav-
igation boilerplate. We have a separate in-house pipeline that
performs optical character recognition (OCR) on Teams con-
versation screenshots to extract the conversation thread54,55.
SDDS has standard documents, while the OAG Apps code-
base and usage description were included as markdown. Each
ingested record is converted into a collection-specific repre-
sentation with source-specific metadata (shown in Figs. S1
and S2 of the supplementary material). Attachments, such as
figures and schematics, are linked to their parent records. The
deployed ingestion layer is broader than those included in the
evaluation. The benchmark corpus and all results in this paper
cover six of these collections (BELY , ICMS, Work Requests,
Teams, SDDS, OAG Apps), whereas the APS wiki and beam-
downtime records were added to the production index only
after the evaluation corpus was frozen. Consequently, these
sources are present in the deployed system but are excluded
from the benchmark, thereby preserving the frozen snapshot
that underpins the reported results.
a. Per-source schemas:A normalized schema was de-
veloped for each source to preserve the fields required for fil-
tering, temporal scoping, citation, and to preserve the record-
ing conventions specific to each contributing team. Then, it
maps the free-text fields to the indexed body (see Fig. S1 of
the supplementary material). The schemas retain native iden-
tifiers, such as BELY_ID, WRQ_ID, ICMS_ID, and likewise
for other sources, ensuring that an inline citation resolves back
to the source record and its canonical web URL. The inclusion
of inline citations enables any staff member to verify the an-
swer at their original sources.
b. Data freshness via daily diffs and upserts:A sched-
uled daily pipeline maintains index freshness without requir-
ing a full corpus rebuild. For each data source, the pipeline
performs an incremental diff operation, comparing the content
hash of each record against a persisted per-document synchro-
nization state. This process ensures that only new or modified
records are re-fetched. Updated records are upserted, using
the document ID as the key, into both the dense vector store
(Qdrant50and the keyword-based store (Elasticsearch51). Si-
multaneously, an incremental update is applied to the Neo4j
knowledge graph for the corresponding documents. Deletions
are similarly propagated based on hash comparison. Conse-
quently, the computational cost of re-indexing is determined
by the daily change set rather than the entire corpus. This
design ensures that time-sensitive information, such as recent
shift logs, open work requests, and current downtime, remains
consistent and up-to-date across all three storage backends.
c. Parent-child indexing and embeddings:Retrieval is
conducted over two complementary channels on a shared
chunk store. Documents use a two-level parent-child scheme
that decouples the unit embedded or scored from the unit re-
turned56. A parent record retains the full context of a complete

APS-RAG 4
FIG. 2. APS-RAG system architecture. A user query first undergoes a preprocessing stage — temporal parsing, synonym addition, acronym ex-
pansion, and multi-query generation — after this a router and planner which includes query intent detector (e.g., factual, procedure, multi-hop)
distributes the query across eight knowledge bases through three retrieval channels: a dense vector index (Qdrant50, e5-large-v2 embeddings),
a sparse keyword index (Elasticsearch51, BM25), and a Neo4j52knowledge graph queried via Cypher. Contexts retrieved from these channels
are merged through query-type-adaptive reciprocal-rank fusion, reranked by a cross-encoder, and forwarded to a corrective orchestrator, which
retries retrieval with a broadened query whenever answer confidence falls below threshold. This same pipeline is exercised end-to-end by the
APS-RAG benchmark and evaluation harness.
record or section, such as a complete logbook file or tech-
nical report, while each parent is split into atomic children
through recursive character-based splitting, using the uniform
two-stage sizes given in Sec. III D and a 200-character over-
lap. Each child is prepended with a concise header contain-
ing the source file_id, document title, and date, ensuring self-
descriptiveness. Both retrieval channels utilize the same set
of children. In the dense retrieval channel, each child is en-
coded as a 1024-dimensional vector using e5-large-v257and
stored in Qdrant for approximate nearest neighbor search. The
e5 model was selected for its robust zero-shot retrieval per-
formance on out-of-domain text, which is particularly rel-
evant for the terse, acronym-rich language characteristic of
accelerator operations. In the sparse retrieval channel, the
same children are indexed in Elasticsearch and scored using
BM25. This redundancy is intentional: dense retrieval en-
ables recovery of paraphrased or semantically related entries,
while BM25 provides reliable matching for exact identifiers
such as process-variable names (e.g.,LTS:H1:CurrentAO,S-
DCCT:CurrentM), sector names, and equipment labels, which
are often not captured by dense encoders. As a result, match-
ing is achieved at fine granularity, while the enclosing parent
record is supplied to the generator to provide necessary con-
text for synthesis.
C. APS-RAG architecture overview
APS-RAG, as shown in Fig. 2, turns a user’s natural-
language query into a source-cited answer drawn from the fa-
cility’s dispersed operational databases. A question is routed
through one of three service modes:APS-RAG (Fast), a singlepass through the pipeline;APS-RAG (Thinking), which en-
ables bounded corrective retrieval rounds (Sec. III E); andAI
Troubleshoot, for fault-resolution queries (Sec. III F). An en-
tered query is first preprocessed by (i) detecting the query in-
tent - factoid (single value), range (temporal), multi-hop, pro-
cedural, comparison, or diagnostic; (ii) resolving the search
date range; (iii) adding synonyms for domain keywords; (iv)
expanding acronyms while retaining the original acronym;
and (v) generatingN qparaphrases (multi-query generation;
Nq=5 in the deployed configuration). Depending on the
query and selected mode, the retrieval channels run in parallel
across the indexed corpus: a dense vector store, a sparse key-
word store, and optionally a knowledge graph. Their ranked
lists are merged by reciprocal-rank fusion13with query-type-
adaptive weights, reranked by a cross-encoder over the top
50 fused candidates (each channel retrieves top-k=100 be-
fore fusion;reranker_top_k=50 in the deployed configura-
tion), and passed to a frontier LLM served through Argonne’s
ARGO gateway (provides data-secure programmable access
to multiple LLMs exclusively for internal users), which syn-
thesizes an answer whose inline citations link directly to the
original records. The dense-plus-sparse hybrid with adaptive
fusion and reranking is the deployed default; the corrective or-
chestration loop and the Neo4j knowledge-graph channel are
deployed as optional modes and are described in Secs. III E
and III F.
D. Adaptive three-way retrieval fusion
The dense, sparse, and graph channels are combined by
reciprocal-rank fusion13,58: a documentdat rankr c(d)in

APS-RAG 5
FIG. 3. State-machine representation of the corrective agentic loop (LangGraph). Nodes are atomic operations: a planner decomposes the
task, a router dispatches it to hybrid retrieval and, when needed, MCP tool calls (Archiver, data analysis, image search), and a chain-of-thought
reasoning step drafts the answer, which the corrective evaluator scores via Eq. (2). On a pass, the response is published; on a fail (dashed edge),
control returns to the planner, which replans and retries with an escalated retrieval strategy if confidence falls below the configured threshold or
more than 50% of the retrieved documents fail the relevance evaluation. The schematic shows the state machine’s maximum-rounds parameter
at five; the deployed configuration caps corrective rounds at two (Sec. III E).
channelcreceives the fused score
RRF(d) = ∑
cwc
k+r c(d),k=60,(1)
where the sum runs over the channels in whichdis retrieved,
andkis the smoothing constant. The three channels com-
bining via RRF is shown in Fig. 4. What distinguishes APS-
RAG is that the channel weightsW V(vector),W K(keyword),
andW G(knowledge graph) are not fixed. A query router
classifies each request into one of eight operational intents,
and each intent maps to its own(dense/BM25/graph)pro-
file: causal, multi-hop, and troubleshooting queries up-weight
the graph channel (weights 0.3/0.2/0.5), factual, and specific
value queries up-weight BM25 (0.3/0.5/0.2, exploiting the
literal date and shift tokens that lexical search handles well),
and mathematical or code queries up-weight the dense chan-
nel. The actual weight setting for various query intents is tab-
ulated in Table S3. The weights are hand-set from operational
experience rather than learned. The performance on different
query types is presented in Fig. S3 of the supplementary ma-
terial. It shows that troubleshooting and multi-hop questions
still seem to be areas where the answers have alleviated hal-
lucination and lower faithfulness scores. Note: A re-retrieval
with an expanded time range is triggered when no records are
found within the defined time range.
E. Corrective agentic loop and self-critique gate
The orchestrator is a LangGraph59state machine (Fig. 3)
whose corrective cycle scores each draft answer as
s=0.4ρ+0.3g+0.3c−0.05n KG,(2)
whereρis the relevance of the retrieved evidence,gthe
grounding of the draft in that evidence,cits completeness,
andn KGthe number of answer entities that cannot be verified
against the knowledge graph. Similar to the fusion weights
of Sec. III D, the coefficients are manually tuned from opera-
tional experience: relevance weighted highest, grounding and
FIG. 4. The APS-RAG retrieval and generation pipeline com-
prises three retrieval channels: a dense vector channel implemented
with Qdrant and e5-large-v2, a sparse keyword channel based on
Elasticsearch and BM25, and an optional knowledge-graph chan-
nel utilizing Neo4j. Each channel returns candidate passages, which
are merged using reciprocal-rank fusion with query-type-adaptive
weights. The merged candidates are subsequently reranked by a
cross-encoder that jointly evaluates each passage with the query.
Controlled ablation experiments indicate that this reranking stage
strongly contributes to APS-RAG’s performance. The highest-
ranked passages, together with the query, are then provided to a fron-
tier large language model accessed via Argonne’s ARGO gateway,
GPT-5.4. This model synthesizes an answer that includes inline ci-
tations, linking each claim to its corresponding source passage and
web-accessible record.
completeness equally, and a small per-entity penalty for un-
verifiable knowledge-graph mentions rather than learned or
tuned; they were fixed before either benchmark was scored.
Re-retrieval is triggered when the scoresfalls below a con-

APS-RAG 6
figured threshold, withτ=0.4 used in the deployed configu-
ration. The default threshold and the impact of the corrective
trigger on an ablation subset are presented in Fig. S4 of the
supplementary material. Substantive improvements in RAG
answer quality are observed at the threshold and afterτ=0.3.
The corrective trigger is activated when the supported fraction
of retrieved evidence is less than one-half, or when at least
three knowledge-graph entities remain unverified.
Upon trigger activation, the retrieval strategy advances to a
more intensive loop. For instance, a stalled three-way search
is replaced by a dense-only retrieval strategy, and the query
is broadened using an LLM-generated reformulation for up to
two additional rounds. After these iterations, the orchestrator
returns the best available draft with a floored confidence value,
thereby preventing indefinite looping. Three answer metrics
were evaluated: strict vital recall (V strict), and faithfulness15.
These metrics increase modestly fromτ=0.3 and plateau
nearτ=0.7 to 0.8, while the answer change rate saturates
atτ=0.5. Prior to this comparatively expensive corrective
loop, a self-critique gate is applied. A faster model (Claude
Haiku 4.5, served through ARGO) assesses whether the draft
appropriately answers the query, is well-cited, and is concise.
If the draft fails this assessment, a single targeted rewrite is
triggered. This gate is designed to resolve straightforward
failures at a fraction of the computational cost required for
a full corrective round.
F. A troubleshooting-oriented knowledge graph
We built an offline pipeline to extract entities and rela-
tions from six collections with a custom two-stage LLM ex-
tractor, loading the results into Neo4j. The graph contains
96,517 nodes and 84,222 edges across 14 types, including
Fault, Component, System, and LogEntry. Retrieval sup-
ports entity lookups, path-based traversals, and shortest-path
searches to reconstruct cause-to-effect-to-resolution chains
beyond what flat vector retrieval can achieve. Graph analyt-
ics use native Cypher queries in Neo4j Community Edition,
with degree-centrality and community-membership proxies
and exact shortest-path traversal12. These serve as prox-
ies, not a full analytics stack. The graph is purpose-
built to reconstruct troubleshooting procedures from his-
torical records. We connect LogEntry, Fault, Compo-
nent, System, Procedure, WorkRequest, and beam-downtime
nodes via cause-and-resolution relations (CAUSED_BY ,
RESOLVED_BY , HAS_FAULT, ABOUT_SYSTEM, FOL-
LOWS_PROCEDURE, CONTACT_PERSON). This enables
tracing symptom-to-cause-to-action-to-resolution chains, mir-
roring how experienced staff resolve issues. The system re-
trieves similar historical fault-resolution chains and generates
actionable, citation-backed troubleshooting steps based on
proven solutions. This is especially effective for causal, multi-
hop, and troubleshooting queries where evidence is distributed
across records and vector similarity is insufficient. We assess
this using the fault-chain-completeness metric (Sec. V), prior-
itizing fault-chain coverage over complete procedural correct-
ness.G. Native-tool ReAct and the MCP tooling layer
APS-RAG executes tools using a dual-path ReAct execu-
tor, which prioritizes a native function-calling approach while
retaining a prompt-based fallback for legacy compatibility.
The system integrates eight tools, provided by four FastMCP
servers (archiver, graph, compute, and image; see Table S1
in the supplementary material), each operating behind a stan-
dardized error envelope and supported by a persistent ses-
sion pool. This design amortizes the substantial index loading
overhead, which would otherwise significantly increase per-
call latency. The resulting architecture allows APS-RAG to
synthesize live PV readings with document-based evidence to
generate unified responses. As this capability is not reflected
in the primary evaluation metrics, further details are provided
in Sec. S1 of the supplementary material. Deployment within
operational facility infrastructure necessitated ongoing reli-
ability engineering to address infrastructure drift, including
gateway migrations, modifications to response envelopes, and
silently disabled preprocessing chains. A summary of rep-
resentative incidents and the corresponding regression safe-
guards is presented in Sec. S3 of the supplementary material.
H. The deployed interface and representative query and
responses
The pipeline of Secs. III C–III F reaches staff through
the OpenWebUI60chat interface shown in Fig. 5(a): a
mode selector exposesAPS-RAG (Fast)andAPS-RAG (Think-
ing)alongside companion assistants (image search, a sched-
uled daily-brief summarizer, and a help assistant), and
suggested entry points advertise the most common needs
live machine status (answered through the tooling layer of
Sec. III G), acronym resolution (the query-enhancement chain
of Sec. III C), and technical-document search. Panels (b)-(g)
of Fig. 5 show six live query–response pairs that span the
query-type taxonomy of Sec. IV A and make the preceding
subsections concrete. Panel (b) reconstructs the first-stored-
beam commissioning sequence as a dated narrative stitched
from multiple logbook entries, from the April 2024 injec-
tion attempts through “first light at 35-BM” (BELY_2612).
Panel (c) answers a parametric study question by retriev-
ing and differentiating four beam-lifetime-versus-fill-pattern
studies (81–432 bunch configurations). Panel (d) is the
fault-chain behavior Sec. III F targets: for a stored-beam-
loss query, it returns the proximate cause and timestamp,
the step-by-step recovery timeline, and a linked mainte-
nance record (WRQ_52149) that it flags asrelated con-
text while declining to assert it as the cause, alongside
an explicit statement that the records do not support an
insertion-device attribution—the citation discipline the self-
critique gate of Sec. III E is designed to enforce. Panel (e)
shows Fast-mode technical-document lookup returning two
linked ICMS reports on ELEGANT spin-evolution simula-
tions. Panel (f) executes a personnel- and method-scoped
search (MOGA/MOBO lifetime optimization) and honors the
requested reverse-chronological ordering with per-entry quan-

APS-RAG 7
FIG. 5. The APS-RAG interface offers multiple retrieval modes, including Fast and Thinking, as well as image-search capabilities and
scheduled-summary and help assistants. The interface provides suggested entry points for tasks such as live machine-status lookups, acronym
resolution, and technical-document searches. Six representative query–response pairs demonstrate the system’s performance on operational
queries, such as (b) tracing the first-stored-beam commissioning sequence, (c) identifying beam-lifetime-versus-fill-pattern studies, (d) diag-
nosing stored-beam-loss events, (e) retrieving technical documentation, (f) history of lifetime optimization, and (g) finding the causes of stored
beam lost over a week’s time. Each response contains inline citations to underlined record identifiers, including BELY , Work Request, and
ICMS file IDs, which link directly to the relevant logbook entry, work request, or technical document. This design allows for systematic
verification of the provenance of each response. The outputs presented are generated by the deployed system and illustrate the response format
and citation traceability.
titative outcomes. Panel (g) resolves the relative window
“last week” into absolute dates (the temporal-parsing step of
Sec. III C), aggregates the stored-beam-loss events in it with
their likely proximate causes, and states plainly which days
had none. The common thread is traceability: every claim
carries an inline citation to a native record identifier (BELY ,
WRQ, or ICMS ID) that resolves to the source record, so a
staff member can verify each fact where it lives, and the visi-
ble reasoning-time indicators expose the Thinking mode’s de-
liberation cost. These outputs are from the deployed system
and illustrate response format and citation traceability. Lastly,
APS-RAG can be directly reached through Claude Code or
Codex using the/aps-ragskill (see Sec. S4 for details).
IV. BENCHMARKS AND EVALUATION METHODOLOGY
Evaluation is an integral part of model development, crucial
for new domain adaptation, and critical for safety-first appli-
cations such as accelerator operations, so we describe it in
some detail. It is designed to answer three questions: whether
retrieval augmentation improves answer quality on the facil-
ity’s institutional knowledge, which pipeline components con-
tribute reliably to that improvement, and how performance on
a corpus-derived benchmark transfers to verbatim staff ques-
tions. The results in Sec. V are organized around these ques-tions.
A. Real operations data benchmark
APS-Bench (corpus-derived,n=50; 49 answerable, one
abstention item) is built from the live corpus, not handwritten
(Fig. 6). For each (query-type×collection) cell, we sample
a full record from a collection which is part of the APS cor-
pus and generate type-conditioned questions with an answer
set using a GPT-family model (GPT 5.4via ARGO, temper-
ature 0.0); gold answers are then constructed from the sam-
pled document, with grounded inline citations, and each an-
swer is decomposed into 3–10 atomic nuggets labeledvitalor
okay(at least two vital). This method is similar to the tech-
nique described in the InPars9. In Fig. 6, a sampled logbook
titled ’Analysis of M1 excursion MPS trip’ is sent to a gen-
erator LLM with a question-answer generation prompt to cre-
ate a question and answer pair from the full logbook record.
The percentage of each question type (e.g., lookup, factoid,
or other) to be selected in APS-Bench is preset. Every item
carries an auditable provenance trail (prompt, questions, ver-
ified answers, original document IDs, and prompt-template
hashes; see Sec. S9 of the supplementary material), and APS-
Bench’s gold remains highly BM25-retrievable (mean gold-
document minimum rank 0.41; Table S4 of the supplemen-

APS-RAG 8
FIG. 6. Construction of the corpus-derived benchmark (APS-Bench). The questions are generated from source passages from the APS Corpus.
Documents are sampled from the indexed corpus, the union of the per-database collections (Sec. III B), and passed, with a question-type
taxonomy controlling the mix of query types, to a generator LLM that produces a question paired with a reference answer grounded in the
sampled passage; each item is retained only if its gold passage is retrievable from the corpus. The example shown is illustrative: a multi-hop
causal question generated from a logbook entry on an M1-excursion machine-protection-system trip, with an answer that both attributes the
cause and resolves whether a follow-up action was required. The taxonomy shown is schematic; the realized question-type and collection
composition of the 50-question benchmark are summarized in Table S4, as well as its difficulty strata, and its vital-nugget annotation is
reported in Sec. S9 of the supplementary material.
tary material, which reports the composition of the bench-
mark dataset). Construction details and the benchmark dataset
are given in Sec. S8– S9 of the supplementary material. One
thing to note here is that, in the question-prompt for bench-
mark dataset creation, five real APS staff questions were used
as few-shot examples for the question-generation prompt. The
benchmark carries one abstention item - a question the knowl-
edge base cannot answer - that tests whether a system de-
clines rather than fabricates; abstention items are excluded
from nugget recall and faithfulness and scored only by the
abstention-correctness sub-metric.
We deliberately usen=50 because each question requires
several LLM inference calls beyond a single-pass RAG query.
These include intent classification, temporal parsing, multi-
query generation, planning and routing, synthesis, a self-
critique gate with targeted rewrite, and up to two corrective
rounds. The five RAG variants and three-seed design then
multiply this by 3×end-to-end runs before judging. Nugget-
presence judging adds roughly six calls per question, based
on an average of 6.3 nuggets, and faithfulness evaluation re-
quires a separate call to decompose the answer into claims.
Altogether, a single 50-question, five-system evaluation costs
on the order of∼5,000 LLM inferences.B. A six-layer evaluation
Five pipeline variants are evaluated in an ablation ladder,
beginning with a naive BM25 baseline (Naive RAG), followed
by Hybrid RAG, Agentic RAG, and GraphRAG, culminating
in the full corrective Agentic GraphRAG which is deployed in
production as APS-RAG across six layers: (1) retrieval qual-
ity (MRR@10, graded NDCG@10, Recall@k); (2) nugget
answer quality15metrics, including strict vital recallV strict,
vital recallV, and all-nugget weighted recallW(headline);
(3) claim-level faithfulness using an HHEM-2.1 backstop29
(headline), which employs a decompose-then-verify halluci-
nation rate inspired by FaithJudge28but omits the human-
annotated peer-response pool required by FaithJudge, and is
therefore not presented as FaithJudge itself; (4) a RAGAS-
style30lexical-overlapproxy(for comparability only); (5) the
domain-specific metrics for institutional knowledge described
above; and (6) paired statistics. Two answer-match metrics
(token-level F1 and semantically judged containment accu-
racy) link these results to the short-form QA literature and
are defined and reported in Sec. S5 of the supplementary ma-
terial. Traditional reference-based metrics (such as BLEU,
ROUGE, and BERTScore) are intentionally excluded from the
primary evaluation, as they assess surface-level text similarity
rather than the accurate conveyance of underlying institutional
knowledge.

APS-RAG 9
C. Judging, tiering, and statistics
Answers are generated by a GPT-family model (GPT 5.4
via ARGO) and subsequently evaluated by a large language
model judge. This evaluation protocol is used instead of
reference-based lexical overlap metrics, which are insufficient
for determining whether a long-form operational answer ac-
curately reflects the intended institutional fact. Automated
evaluation using large language models as judges has become
standard for open-ended and retrieval-augmented generation
tasks, with robust judges demonstrating high correlation with
human preference assessments61. In this protocol, the judge
receives the question, the gold answer annotated with atomic
nuggets, and the system’s response, and produces per-nugget
presence decisions for vital-nugget recall as well as per-claim
support decisions based on the retrieved evidence for the faith-
fulness metric30. It is important to note that large language
model judges are subject to systematic biases, including po-
sition and verbosity effects61, and, in this context, a self-
enhancement bias in which a model tends to assign higher
scores to outputs from its own family. This bias persists even
when evaluators can identify their own generations62.
To avoid confounding answer quality with self-preference,
the evaluation protocol enforces a strict separation between
generation and judgment model families. For each evaluation,
the generator and judge are drawn from distinct model fami-
lies. Answers produced by the GPT family are evaluated by
Claude, which serves as the primary judge. A separate Gemini
model family is used for secondary verification. Judge assign-
ments are determined by call site: Opus 4.6 is used for tasks
requiring heavy reasoning, while Sonnet 4.6 is used for lighter
or bulk evaluation steps. These assignments are fixed across
all five systems, as detailed in Table II. This approach iso-
lates architectural effects and maintains a single, fixed judge
per headline metric. Absolute differences are reported with
95% paired bootstrap confidence intervals, computed using
10,000 resamples per question, and exact McNemar tests are
applied for paired binary outcomes. Evaluation is performed
using seeds(42,123,456). Because RAG inference is cached
per (system, question) and scoring layers operate determinis-
tically at temperature 0, the standard deviations reported are
near zero by construction. These values should not be inter-
preted as evidence of low run-to-run variance. The appro-
priate measure of uncertainty is the bootstrap confidence in-
terval over questions, rather than the standard deviation over
seeds. Since a single judge, even when cross-family, may still
introduce idiosyncratic biases, the recommended practice of
assessing inter-judge agreement is also followed5. The full
system’s APS-Bench answers were re-scored with a fully in-
dependent third-family judge (Gemini), disjoint from both the
generator and the primary judge, and substantial per-nugget
agreement (κ≈0.8) was observed, with the headline strict vi-
tal recall only marginally affected under the swap. This proce-
dure bounds, but does not eliminate, single-judge sensitivity.V. RESULTS AND DISCUSSION
Unless stated otherwise, all results are on APS-Bench (n=
50), evaluated against the triangulated gold; table values are
computed from a single frozen evaluation source. The single
most important framing point comes first: at this benchmark
size, with most ablation CIs spanning zero (Table IV), the only
component contribution that is statistically reliable on the cur-
rent data is the reranker, and that finding survives multiplicity
correction: under Holm/Benjamini-Hochberg adjustment the
reranker remains significant (paired-bootstrapp<10−4; Bon-
ferroni across all tests), while every other component- and
system-level difference is not detected — a count consistent
with the≈0.4 false positives expected by chance across the
ablation family. Accordingly, we describe differences without
characterizing small gaps as definitive wins.
A. Answer quality and faithfulness
The primary result indicates that all retrieval-augmented
variants achieve higher strict vital recall compared to the naive
BM25 baseline, increasing from 63.8% to a range of 65.5%–
70.3%, corresponding to gains of+1.7 to+6.4 points (Ta-
ble II). The full Agentic GraphRAG configuration attains the
highest recall at 70.3%, followed by GraphRAG at 69.1%.
However, paired statistical tests reported in Table II show
that none of the ten pairwiseV strict comparisons reach sta-
tistical significance, either nominally or after Holm correc-
tion. Similarly, exact McNemar tests on binary per-question
outcomes do not identify significant differences (minimum
Holmp=0.92), indicating a lack of detection at this bench-
mark size rather than equivalence. Weighted recall demon-
strates three nominal improvements over the naive baseline
(hybrid, GraphRAG, and the full system), but none remain
significant after Holm correction (see supplementary mate-
rial). Component ablation (Table IV) reveals that neither
the graph channel nor the corrective loop alone is individ-
ually decisive at this benchmark size. Claim-level halluci-
nation rates range from 7.9%–11.3% across systems, with
the lowest value observed for the corrective Agentic RAG
(7.9%). Analysis of the four augmented variants shows that
the corrective loop reduces hallucination relative to its hy-
brid base (9.4%→7.9%), while adding the graph channel
alone maintains hallucination at the naive level (11.3% ver-
sus 11.2%). Combining the graph channel with the correc-
tive loop reduces hallucination from 11.3%→8.7%. The
only starred interval in Table II (Agentic RAG versus naive)
is nominal and does not remain significant after Holm correc-
tion. The answer-match metrics are consistent with these find-
ings: accuracy ranges from 58.8%–65.0% and best-span F1
from 63.1%–73.3%, with the hybrid configuration achieving
the highest values (65.0%/73.3%) and the full system second
(64.1%/71.2%). All retrieval-augmented variants outperform
the naive baseline on every answer-quality metric, although
no ordering among the augmented variants is statistically sup-
ported. These results suggest that the primary benefit of the
corrective loop is improved faithfulness, while the advantage

APS-RAG 10
TABLE II. Generation quality of the five RAG variants on APS-Bench (n=50): strict vital recallV strict, claim-level hallucination rate (lower is
better), with 95% paired-bootstrap CIs of the difference versus the naive baseline. Accuracy and F1 scores are also measured. Values are means
over three seeds: generation, cross-family judging, and test protocols in Sec. IV C. Details onV strict, hallucination rate, and the answer-match
metrics (Acc./F1) are in the supplementary materials. Note: Agentic GraphRAG is deployed in production as APS-RAG.
System Retrieval StrategyStrict Vital Recall
Vstrict(%)∆Vstrictvs. Naive
[95% CI]Hallucination
Rate (%)∆Halluc. vs. Naive
[95% CI]Accuracy
(%)F1
(%)
Naive RAG BM25 63.8 – 11.2 – 58.8 63.1
Hybrid RAG Vector + BM25 65.5+1.7[−4.1,+7.8]9.4−1.7[−8.5,+4.3]65.0 73.3
Agentic RAG Vector + BM25 66.1+2.3[−4.4,+9.2]7.9−3.3[−15.5,−1.9]60.6 67.1
GraphRAG Vector + BM25 + KG 69.1+5.3[−2.5,+13.6]11.3+0.1[−7.8,+7.9]61.9 69.2
Agentic GraphRAG (ours) Vector + BM25 + KG 70.3+6.4[−2.1,+15.6]8.7−2.5[−10.2,+4.4]64.1 71.2
TABLE III. Retrieval quality was evaluated on APS-Bench (n=50). NDCG was calculated using graded relevance (primary=2, supporting
=1), and rankings were deduplicated to address document granularity. Reported values represent means over three random seeds. According
to Holm correction applied across each metric’s ten system pairs, no comparison reached statistical significance. Nominally, the hybrid
method (MRR@10+8.3[+1.6,+16.6]versus naive) and GraphRAG (MRR@10+6.3[+0.3,+14.2]) outperformed the naive baseline, while
NDCG@10 and Recall@10/20 did not exhibit nominal differences.
System MRR@10 (%) NDCG@10 (%) Recall@5 (%) Recall@10 (%) Recall@20 (%)
Naive RAG 80.7 69.6 63.5 67.3 71.2
Hybrid RAG 89.0 73.6 62.6 70.4 76.3
Agentic RAG 83.8 71.0 67.8 76.5 74.0
GraphRAG 87.0 73.3 61.9 69.1 76.7
Agentic GraphRAG (ours) 85.6 72.2 62.1 69.1 75.1
of the graph channel is conditional. Further discussion is pro-
vided in Sec. V E.
B. Answer quality across difficulty categories
Analysis of the full configuration’s answer quality by dif-
ficulty stratum (easyn=15, mediumn=24, hardn=10,
n=1 is abstention; out of the 50 questions). The ques-
tion composition in Table S4 reveals a non-monotonic pat-
tern. Strict vital recall is highest for the easy band (77.9%),
remains similar for the hard band (72.6%), and is lowest for
the medium band (64.5%). Mean claim-level hallucination
follows the inverse order (3.4%, 10.1%, and 21.6%, respec-
tively). Given the sample sizes of 10-24 questions per cate-
gory, no significance tests are attached to these differences;
the following analysis is descriptive. The easy band behaves
as intended, consisting of single-hop factual questions an-
chored to precise operational identifiers, where both the sparse
channel and the reranker demonstrate optimal performance.
For example, when queried about which after-scan PVs must
be zeroed to prevent 9-BM 2D scans from hanging (APS-
BENCH-006), the system returns both PV names verbatim
(9bmc:scan1.AWAIT,9bmc:scan2.AWAIT), the “NoWait”
qualifier, and the failure mode, citing the correct entry
[BELY_10635] (strict vital recall 100%, hallucination 0%).
This level of grounding also extends to multi-entry lookups
requiring negative discrimination. For the mispolarized-BTS-
quadrupole discovery (APS-BENCH-036), the system re-
trieves both 2024-04-12 entries with accurate timestamps and
the cable-swap resolution [BELY_1651, BELY_1653] while
explicitly excluding an earlier magnet-testing entry as relatedbut not the discovery itself [BELY_721].
The medium band is empirically the most challenging,
with its performance deficit attributable to composition ef-
fects rather than a decline in system competence. This stratum
is characterized by terse operational comparisons and tem-
poral texts, where evidence is dispersed across sparse log-
book and work-request records, and reference answers fre-
quently hedge or abstain appropriately. When evidence is
well localized, medium-band questions are addressed as ef-
fectively as those in the easy band. For instance, the linac
macro-pulse question (APS-BENCH-042) yields the com-
plete quantitative result—a single macro pulse at 13 kV
that splits into two at 14–15 kV [BELY_5290], along with
the companion RG2 measurement of two pulses of approx-
imately 2 and 9 ns separated by approximately 7 ns at 15.2
kV [BELY_7482]—achieving 100% strict vital recall and
zero hallucination. The cross-beamline comparison (APS-
BENCH-029) retrieves the 7 mrad Pt-stripe HR-mirror set-
ting and calculates the 3 mrad differential relative to the 20-
BM AuPd setup at 4 mrad, again without hallucination. How-
ever, its supporting citation refers to an adjacent entry from the
same test series rather than the gold-annotated record, result-
ing in content accuracy but citation imprecision. The band’s
21.6% mean hallucination rate is primarily driven by items
with scattered evidence, where retrieval provides only partial
context, and the generator compensates with under-supported
specifics. This is the failure mode that inexpensive gating
components are designed to suppress in ablation section.
The hard band performs better than its designation implies
(62.6%), primarily due to its composition. It is dominated
by multi-hop and causal questions that are grounded in sta-
ble, well-documented reference materials such as the SDDS

APS-RAG 11
Toolkit and ICMS. Although the evidence is dispersed, it re-
mains accessible. Two key behaviors are observed. The first
is causal discipline: when queried about whether a three-
screen emittance measurement improved linac-to-PAR charge
transmission (APS-BENCH-041), the system avoids over-
attribution. It reports the improvement, enumerates the con-
founded co-interventions (gun-front-end optimization, RF and
trajectory tuning, kicker timing) [BELY_4799, BELY_8520],
explicitly states that the log does not isolate the measure-
ment as the cause, and cites the standard procedure to clar-
ify its actual role as beam matching rather than a standalone
charge booster [APS_1684426]. This matches the reference
at zero hallucination, demonstrating the guarded behavior that
the self-critique gate and knowledge-graph grounding are in-
tended to produce. The second behavior is the characteris-
tic hard-band failure: the issue is coverage rather than fabri-
cation. For example, the two-month spin-tracking synthesis
(APS-BENCH-037) recovers the correct three-thread struc-
ture (implementation, validation, applications) and the correct
time window, but omits vital per-document details and cites a
partially different document subset, resulting in 25% strict vi-
tal recall and only 8% hallucination. Thus, hard multi-hop er-
rors are typically omissions rather than fabrications; recall de-
clines before faithfulness. This mirrors, at the question level,
the recall/faithfulness decoupling and aligns with the condi-
tional, per-type interpretation of the graph channel’s benefit.
C. Retrieval quality
Hybrid dense+sparse fusion gives the best first-stage rank-
ing quality: MRR@10 89.0% versus 80.7% for naive
BM25 (a nominal gain of+8.3 points[+1.6,+16.6]) and
NDCG@10 73.6% versus 69.6%; GraphRAG follows at
MRR@10 87.0% (+6.3[+0.3,+14.2], likewise nominal),
and neither gain survives Holm correction (Table III). The
graph-bearing variants rank slightly below hybrid (MRR@10
85.6–87.0% versus 89.0%), consistent with the graph channel
adding relational candidates that dilute ranking precision even
as they feed downstream reasoning, while the corrective agen-
tic variant recovers recall at deeper cutoffs (Recall@5 67.8%
and Recall@10 76.5%, the best of the five; Recall@20 is
led by GraphRAG at 76.7%). NDCG@10 and Recall@10/20
show no nominal differences across the ten system pairs. That
the full Agentic GraphRAG nonetheless posts the highest an-
swer quality (70.3%, Table II) despite trailing hybrid on first-
stage ranking (85.6% versus 89.0% MRR@10) suggests its
corrective reasoning compensates downstream for first-stage
ranking it does not itself lead.
D. Component ablation
The ablation experiments utilize a hard-biased, type-
stratified subset of APS-Bench, referred to as the ablation sub-
set. Each question receives a score of+2 if its difficulty label
is hard and+2 if its query type is multi-hop, causal, or trou-
bleshooting. The subset is populated in a round-robin man-ner across all eight query types, prioritizing the most diffi-
cult questions. This process results in 19 questions (8 hard,
4 medium, 7 easy), ensuring representation from all query
types. The subset serves as a stratified sample, distinct from
the difficulty stratum referenced in Table S4. On this subset
(Table IV), one component is particularly influential: substi-
tuting the Jina v3 cross-encoder63with the legacy LLM scorer
leads to a reduction of 32.8 points in strict vital recall (95%
CI[−47.4,−19.1]). Two important qualifications constrain
this result. First, the per-query distribution indicates that the
legacy scorer not only ranks less effectively but often fails
entirely: NDCG@10 falls below 0.02 (indicating no usable
ranking) for 63% of the subset questions and below 0.10 for
68%. On a per-question basis, strict vital recall is lower for
12 out of 19 questions, unchanged for 7, and never higher, re-
sulting in an aggregate NDCG@10 of 13.4 compared to 69.5
for the full configuration. This row should therefore be inter-
preted as a comparison between a functioning cross-encoder
and a degraded fallback, rather than as a general estimate of
the marginal value of reranking. The comparison among com-
petent rerankers, by contrast, is a single-digit lever: on a fixed
10k-document diagnostic index constructed to isolate reranker
choice, swapping the Jina v3 cross-encoder for bge-reranker-
v2-m3 or Qwen3-Reranker-4B shifts MRR@10 only from
86.9 to 83.3–84.7 over a BM25 first stage (Tables S6– S7, and
Fig. S6 of the supplementary material, which also report the
no-rerank baseline). Second, the validity of this finding is not
compromised by multiple comparisons. The reranker effect
remains significant under all applied corrections: its paired-
bootstrapp<10−4(none of 100,000 resamples crossed zero),
and it persists after Bonferroni correction, whether applied to
the nine ablation rows alone or to the nine rows plus the ten
system pairs of Table II, as well as after Benjamini–Hochberg
correction.
Removal of the corrective loop (−6.6,[−21.4,6.2]), the
graph retriever (−6.4,[−20.7,5.0]), or the adaptive weights
(−3.5,[−16.8,6.7]) results in directional decreases in recall,
though the confidence intervals remain wide atn=19. The
self-critique gate and the query-enhancement steps do not ex-
hibit individually significant effects on this subset. These re-
sults indicate that, at the current benchmark size, recall per-
formance is primarily determined by retrieval and reranking
quality. Any recall contribution from agentic components ap-
pears limited and will require a larger, pre-registered study for
definitive assessment.
Excluding the reranker outlier, the per-ablation changes in
strict vital recall and hallucination rate are positively corre-
lated (r= +0.63): components that best preserve recall are
also those whose removal most increases hallucination. For
example, removal of the self-critique gate leaves strict vital
recall essentially unchanged (+0.4, CI[−15.8,14.9]) but in-
creases the hallucination rate by 3.7 points (7.7%→11.4%).
The adaptive RRF weights and the synonym resolver exhibit
similar patterns (−3.5 and−6.3 recall, both CIs spanning
zero, with hallucination increases of+1.8 and+1.4 points,
respectively). Temporal parsing is the only component whose
removal slightly reduces hallucination (7.7%→6.9%) while
decreasing recall (−6.7), consistent with the hypothesis that

APS-RAG 12
date-scoped retrieval elicits more specific and thus more fal-
sifiable claims, whereas its absence leads to vaguer responses
that miss nuggets but assert less. These results indicate that
several inexpensive gating and preprocessing components pri-
marily contribute to faithfulness rather than recall. A recall-
only interpretation of Table IV would underestimate the value
of the self-critique gate, which functions as a faithfulness
guard by pruning weakly grounded claims without recover-
ing new evidence. The corrective loop demonstrates a similar
effect at full-benchmark scale, directionally lowering halluci-
nation from 9.4% to 7.9% over its hybrid base (Table II). This
decoupling is reported as directional, since the hallucination-
rate deltas do not include per-row confidence intervals at
n=19. The∆-recall-versus-∆-hallucination scatter, provided
in the supplementary material, is consistent across four of the
nine rows. These findings support the practical recommenda-
tion to prioritize reranking and to retain inexpensive faithful-
ness gates, as their removal has minimal impact on recall but
increases unsupported claims by 6-9 points.
E. When graph retrieval helps, and what it costs
The honest reading of these results is nuanced rather than
triumphal, and we think the nuance is the contribution. The
full corrective agentic GraphRAG achieves the highest strict
vital recall on the full benchmark, with a score of 70.3%
(Table II). However, the recommendation to deploy this sys-
tem is based on a numerical advantage that is not statisti-
cally supported by the paired tests in Table II. For organiza-
tions implementing a knowledge-management system of this
type, the primary recommendation is to prioritize investment
in reranking mechanisms before introducing agentic compo-
nents. The inexpensive gating components should be retained,
as their contribution is primarily to faithfulness rather than re-
call (Sec. V D).
The benefit of the knowledge graph is directional. The
graph-bearing variants achieve the highest answer quality on
the full benchmark, with scores of 69.1% and 70.3% (Ta-
ble II). However, the ablation removing the graph is incon-
clusive, with a difference of -6.4 points and a 95% confi-
dence interval of [-20.7, 5.0] (Table IV). The graph channel
does not improve first-stage ranking precision, as indicated
by MRR@10 values of 85.6% to 87.0% compared to 89.0%
for the hybrid system (Table III). Without the corrective loop,
the graph-based system exhibits the highest hallucination rate
among the five systems, at 11.3% (Table II). The value of
the knowledge graph appears to be concentrated in causal,
multi-hop, and troubleshooting query types, where it is associ-
ated with increased fault-chain completeness, expert finding,
and procedural correctness. Figure S1 of the supplementary
material provides the per-question faithfulness breakdown by
query type. This observation is hypothesis-generating rather
than established. The per-query-type weights were fixed prior
to evaluation, which reduces but does not eliminate the risk of
post-hoc stratum selection. The conditions under which graph
retrieval is beneficial remain an open question in general-
domain RAG, as noted by Xiang et al.22. A pre-registered, ad-equately powered per-stratum test is required to resolve this,
and such an evaluation is identified as future work rather than
a current claim.
F. Effect of the embedding model and corrective rounds
We evaluated four embedding models on a fixed 10,000-
document diagnostic index (Fig. S7 and Table S5 of the sup-
plementary material), observing a clear hierarchy in perfor-
mance. The bge-m3 model yielded the highest vector-only
MRR@10 at 79.2%, with the deployed e5-large-v2 achieving
70.2%. The all-MiniLM-L12-v2 and domain-pretrained ac-
cphysbert models performed less effectively, with MRR@10
values between 58.5% and 60.6%. Hybrid fusion narrowed
the performance gap, producing MRR@10 values in the range
of 74.6% to 85.1%. The first-stage BM25 baseline also
achieved strong results (82.1%). Despite overlapping boot-
strap confidence intervals, two principal observations can be
made. First, the domain-pretrained encoder does not surpass
general-purpose models on this corpus. Second, retrieval per-
formance is influenced more by the cross-encoder reranker
than by the choice of embedding model (Table 1; Tables S6
and S7 of the supplementary material). A two-dimensional t-
SNE64projection of e5-large-v2 embeddings (Fig. S5 of the
supplementary material) shows the source collections form-
ing distinct yet partially overlapping clusters, with no single
dense neighborhood covering the facility’s full documented
knowledge. This heterogeneity motivates the hybrid, graph-
augmented architecture, in which the knowledge graph sup-
plies cross-collection links, such as those connecting a fault
reported in a logbook entry to its resolution in a work request,
which are not guaranteed by embedding proximity and would
otherwise require manual reconstruction across disparate sys-
tems.
A hyperparameter sweep (Fig. S9(a) of the supplementary
material) was conducted to assess the impact of varying the
maximum number of corrective rounds on APS-Bench. The
primary performance improvement is observed after the first
corrective round, with additional rounds yielding no consis-
tent further benefit. This pattern of diminishing returns sup-
ports using a two-round cap in deployment. Full-benchmark
results indicate that the corrective loop predominantly im-
proves faithfulness rather than recall, reducing hallucination
from 9.4% to 7.9% relative to the hybrid baseline (Table II).
Most questions are adequately grounded after a single pass, so
the observed faithfulness gains are concentrated among a sub-
set of complex, multi-hop questions. This improvement is ac-
companied by increased latency: in the May 2026 deployment
study (Table S2 and Fig. S9(b) of the supplementary mate-
rial), end-to-end wall time for a 10-query advanced-reasoning
workload ranged from 40 to 178 seconds, depending on the
planner model. Retrieval fan-out accounted for the majority of
this time (55–104 seconds), with reranking contributing∼18
seconds.

APS-RAG 13
TABLE IV . Component ablation was conducted on the19-questionstratified ablation subset (selection rule in Sec. V D; 8 hard, 4 medium, 7
easy; all eight query types) by removing or swapping a single component of the full configuration, evaluated on the triangulated gold standard.
The∆column reports the absolute change in strict vital recall compared to the full configuration (percentage points), with a 95% paired
bootstrap confidence interval. The reranker row is the only interval that excludes 0, and this result remains significant under both Bonferroni
and Benjamini–Hochberg correction (paired-bootstrapp<10−4). Reported values are means over three seeds (deterministic, temperature = 0,
cached scoring). Confidence intervals are computed only for the strict-vital-recall delta; the remaining columns, including hallucination rates
discussed in Sec. V D, are directional at this subset size. The LLM-scorer row represents a degraded fallback, not a comparison to a competent
reranker (Sec. V D).
ConfigurationStrict Vital Recall
Vstrict(%)Weighted Recall
W(%)Hallucination
Rate (%)NDCG@10
(%)∆Strict Vital Recall (pts)
[95% CI]Accuracy
(%)F1
(%)
Agentic GraphRAG (full, ours) 64.5 70.2 7.7 69.5 — (reference) 78.9 72.6
−Corrective loop (single-pass) 57.9 63.3 7.1 63.1−6.6[−21.4,6.2]73.7 73.0
−Knowledge-graph channel 58.1 64.1 8.2 58.6−6.4[−20.7,5.0]78.9 71.6
−Adaptive RRF weights (fixed) 61.0 61.4 9.5 61.5−3.5[−16.8,6.7]73.7 73.0
−Self-critique gate 64.9 66.5 11.4 60.1+0.4[−15.8,14.9]78.9 72.4
−Reranker 31.8 43.9 30.0 13.4−32.8[−47.4,−19.1]42.1 61.0
−Temporal parsing 57.9 60.4 6.9 61.5−6.7[−20.7,3.9]78.9 73.2
−Synonym resolver 58.2 63.2 9.1 62.7−6.3[−19.7,3.9]84.2 72.5
−Acronym resolver 58.2 62.3 9.4 60.3−6.4[−18.9,2.9]68.4 72.0
−Multi-query generator 61.8 63.0 10.0 62.2−2.8[−15.8,6.4]78.9 71.9
G. Effect of the synthesis model
Replacing the synthesis large language model (LLM)
across ten generators sourced from four vendors, while keep-
ing retrieval and fusion components constant, as well as
reranking and evaluation criteria fixed, results in a change
in strict vital recall of only approximately 12 percentage
points (58.1–70.3%)—the best open-weight generator (GPT-
OSS 120B, 67.7%) sits within 2.6 points of the deployed GPT-
5.4 (70.3%)—while the hallucination rate spreads more than
fivefold (5.6–30.6%) and tracks neither recall nor parameter
count (Fig. S8 of the supplementary material). The 31B open-
weight Gemma4 is the most faithful generator tested (5.6%
hallucination at 62.3% recall) and is small enough to serve
and fine-tune, on facility hardware, whereas the larger open
models reach near-frontier recall at hallucination rates (23.8–
30.6%) we would not deploy against a safety-relevant record
without the pipeline’s faithfulness guards. The component-
level decoupling as noted in Sec. V D thus reappears at the
generator level—recall behaves like a retrieval property, faith-
fulness like a generator property—so hallucination rate, rather
than recall, should be the primary criterion for generator se-
lection in operational retrieval-augmented generation (RAG)
systems.
H. Limitations
Several factors constrain the scope of the present study and
inform the direction for subsequent deployment. The current
evaluation is limited to a single facility, the APS, and the ex-
tent to which these findings generalize to other accelerator
environments remains to be established. Benchmark ques-
tions are generated automatically, as detailed in Sections S8
and S9 of the supplementary material. The primary nugget-
recall and faithfulness metrics are based on gold nuggets ex-tracted by large language models and on the nugget-presence
decisions of an LLM judge. Validation against human judg-
ment, particularly given the brevity and technical nature of
the domain text, is planned for future work. As an interim
measure, cross-family inter-judge agreement on per-nugget
support decisions is substantial, with Cohen’s kappa approx-
imatelyk=0.8 when compared to an independent Google-
family judge, and the strict vital recall metric remains un-
changed under this judge substitution. While this reduces the
risk of artifacts arising from reliance on a single judge, it does
not eliminate it. The expert audit therefore remains the defini-
tive standard.
Citation precision on APS-Bench is approximately 73%,
which merits discussion. At the level of base-document gran-
ularity, precision is defined as the proportion of cited docu-
ments that are included in the gold evidence pool, while recall
is the proportion of gold documents that are cited. This pre-
cision is partly attributable to the fact that each gold answer
is supported by only two to three evidence passages. Never-
theless, this result indicates that a substantial number of non-
gold passages are retrieved as supporting evidence, which has
implications for traceability when staff seek to verify facts at
their source. Citation recall, which ranges from 57% to 66%,
is additionally reported as a more informative complementary
metric. Future work will prioritize claim-level citation scoring
using an expanded evidence pool.
Comparative claims in this study are intentionally limited
to internal systems. The reported standard deviations corre-
spond to deterministic replication and do not capture run-to-
run variability. Although the tooling layer has been deployed,
it has not yet been evaluated using the primary metrics, and its
systematic assessment will proceed as deployment continues.
These limitations do not detract from the main contributions
of this work: the establishment of a benchmark and metric
suite for institutional knowledge, a systematic comparison of
five systems, and a reranker result that remains robust after
correction for multiple comparisons. Collectively, these con-

APS-RAG 14
siderations define the set of questions that ongoing in-service
operation is expected to address.
VI. CONCLUSION
We have introduced APS-RAG, a knowledge-management
platform designed to make a large accelerator facility’s dis-
persed institutional knowledge accessible to all staff, includ-
ing scientists, engineers, operators, and technicians, through
natural-language queries. The platform integrates a cor-
rective, agentic GraphRAG engine, adaptive three-channel
retrieval, and an evaluation framework tailored for opera-
tional deployment. On APS-Bench, a corpus-derived, nugget-
annotated benchmark with auditable gold standards, all
retrieval-augmented variants demonstrate clear improvements
in answer quality over naive RAG. The Agentic GraphRAG
integration achieves the highest point estimate (70.3% versus
63.8% in strict vital recall compared to the naive baseline), al-
though per-system paired tests do not distinguish this lead at
the current benchmark size, so we do not present it as defini-
tive. The necessity of a functioning cross-encoder reranker
is evident in the component-level recall analysis, while the
inexpensive gating components primarily contribute to faith-
fulness rather than recall. We are releasing the code, bench-
mark dataset, evaluation harness, and a reusable agentic re-
trieval agent skill. Effective knowledge management trans-
forms a facility’s accumulated experience into a resource that
reduces downtime and mitigates staff turnover. Our goal is
for the benchmark creation methodology and metric family to
be broadly applicable, supporting any organization or facility
seeking to maximize the utilization and value of its institu-
tional knowledge.
SUPPLEMENTARY MATERIAL
The supplementary material details the experimental pro-
tocols, extended analyses, and reference tables that support
the main text. Section S1 outlines the implementation of
the native-tool ReAct executor and the Model Context Pro-
tocol (MCP) tooling layer. Section S2 presents the results of
the component-latency measurements and the reranker A/B
evaluation. Section S3 summarizes the reliability-engineering
effort. Section S4 describes the design and deployment of
the reusable /aps-rag retrieval skill. Section S5 specifies the
exact-match, F1, and accuracy metrics applied in the answer-
match evaluations reported in Table II. Section S6 compiles
supplementary figures, including the faithfulness breakdown
by query type (Fig. S1) and the sensitivity of model an-
swer evaluation between open-source and closed-source mod-
els (Fig. S8). Sections S7 through S9 document the corpus
schema, the APS-Bench construction protocol, and the prove-
nance of the benchmark dataset.ACKNOWLEDGMENTS
R.S would like to thank AOP Group Members, Greg Fys-
tro, Randy Flood, Brendan Chandler, Elaine Chandler, and
Felix M. Lacap for productive discussions and support in
developing APS-RAG. The research is supported by the U.
S. Department of Energy, Office of Science, under Contract
No. DE-AC02-06CH11357. We gratefully acknowledge the
computing resources provided on Improv and Swing, a high-
performance computing cluster operated by the Laboratory
Computing Resource Center at Argonne National Laboratory.
VII. DATA AVAILABILITY STATEMENT
The APS-RAG code is available on GitHub (), benchmark
(APS-Bench), the evaluation harness, and the/aps-ragre-
trieval skill (see Sec. S4), which lets the platform’s institu-
tional knowledge be queried from Claude Code and Codex,
will be released publicly; the derived metrics that support the
findings are reported in the article and its supplementary ma-
terial. The underlying institutional-knowledge data were gen-
erated at the Advanced Photon Source, a large-scale user fa-
cility, and derived data are available from the corresponding
author upon reasonable request, subject to the facility’s data-
handling policy.
VIII. REFERENCES
1J. N. Galayda, “The Advanced Photon Source,” inProc. PAC’95(JACoW
Publishing, Geneva, Switzerland) pp. 4–8.
2M. Borlandet al., “The Upgrade of the Advanced Photon Source,” inProc.
IPAC’18(JACoW Publishing, Geneva, Switzerland) pp. 2872–2877.
3D. Jarosz, E. Chandler, G. Shen, L. Xiao, N. Arnold, and S. Veseli, “Log-
ging a new era at the APS using BELY,” , 1482–1486.
4M. V . Shankar, L. F. Li, M. A. Davidsaver, and M. G. Konrad, “The
EPICS archiver appliance,” inProceedings of the 15th International Con-
ference on Accelerator and Large Experimental Physics Control Systems
(ICALEPCS’15)(JACoW, Melbourne, Australia, 2015) pp. 761–764.
5J. Gu, X. Jiang, Z. Shi, H. Tan, X. Zhai, C. Xu, W. Li, Y . Shen, S. Ma,
H. Liu, Y . Wang, and J. Guo, “A survey on LLM-as-a-judge,” (2024),
arXiv:2411.15594 [cs.CL].
6T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-
V oss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler,
J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray,
B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever,
and D. Amodei, “Language Models are Few-Shot Learners,” (2020),
arXiv:2005.14165 [cs].
7P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler,
M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, “Retrieval-
Augmented Generation for Knowledge-Intensive NLP Tasks,” .
8Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, M. Wang, and
H. Wang, “Retrieval-Augmented Generation for Large Language Models:
A Survey,” (2024), arXiv:2312.10997 [cs.CL].
9L. Bonifacio, H. Abonizio, M. Fadaee, and R. Nogueira, “InPars: Unsuper-
vised Dataset Generation for Information Retrieval,” inProceedings of the
45th International ACM SIGIR Conference on Research and Development
in Information Retrieval(ACM, Madrid Spain, 2022) pp. 2387–2392.
10V . Karpukhin, B. O ˘guz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and
W.-t. Yih, “Dense passage retrieval for open-domain question answering,”

APS-RAG 15
inProceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP)(2020) pp. 6769–6781.
11S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework:
BM25 and Beyond,” Found. Trends Inf. Retr.3, 333–389 (2009).
12D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and
J. Larson, “From local to global: A graph RAG approach to query-focused
summarization,” (2024), arXiv:2404.16130.
13G. V . Cormack, C. L. A. Clarke, and S. Buettcher, “Reciprocal rank fusion
outperforms condorcet and individual rank learning methods,” inProceed-
ings of the 32nd international ACM SIGIR conference on Research and
development in information retrieval(ACM, Boston MA USA, 2009) pp.
758–759.
14S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling, “Corrective retrieval aug-
mented generation,” (2024), arXiv:2401.15884.
15R. Pradeep, N. Thakur, S. Upadhyay, D. Campos, N. Craswell, and J. Lin,
“Initial nugget evaluation results for the TREC 2024 RAG track with the
AutoNuggetizer framework,” (2024), arXiv:2411.09607.
16P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler,
M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, “Retrieval-
augmented generation for knowledge-intensive NLP tasks,” inAdvances in
Neural Information Processing Systems, V ol. 33 (2020) pp. 9459–9474.
17Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, M. Wang,
and H. Wang, “Retrieval-augmented generation for large language models:
A survey,” (2023), arXiv:2312.10997.
18R. Nogueira and K. Cho, “Passage re-ranking with BERT,” (2019),
arXiv:1901.04085.
19Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “LightRAG: Simple and fast
retrieval-augmented generation,” (2024), arXiv:2410.05779.
20B. Jiménez Gutiérrez, Y . Shu, Y . Gu, M. Yasunaga, and Y . Su, “Hip-
poRAG: Neurobiologically inspired long-term memory for large language
models,” inAdvances in Neural Information Processing Systems, V ol. 37
(2024) arXiv:2405.14831.
21B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, and
S. Tang, “Graph retrieval-augmented generation: A survey,” (2024),
arXiv:2408.08921.
22Z. Xiang, C. Wu, Q. Zhang, S. Chen, Z. Hong, X. Huang, and J. Su, “When
to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-
Augmented Generation,” (2026), arXiv:2506.05690 [cs.CL].
23A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-RAG: Learning
to retrieve, generate, and critique through self-reflection,” inInternational
Conference on Learning Representations (ICLR)(2024) arXiv:2310.11511.
24S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. Park, “Adaptive-RAG:
Learning to Adapt Retrieval-Augmented Large Language Models through
Question Complexity,” inProceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Hu-
man Language Technologies (Volume 1: Long Papers)(Association for
Computational Linguistics, Mexico City, Mexico, 2024) pp. 7036–7050.
25S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and
Y . Cao, “ReAct: Synergizing reasoning and acting in language models,”
(2023), international Conference on Learning Representations (ICLR);
arXiv:2210.03629.
26A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei, “Agentic
retrieval-augmented generation: A survey on agentic RAG,” (2025),
arXiv:2501.09136.
27Anthropic, “Introducing the Model Context Protocol,” (2024),https://
www.anthropic.com/news/model-context-protocol.
28M. S. Tamber, F. S. Bao, C. Xu, G. Luo, S. Kazi, M. Bae, M. Li,
O. Mendelevitch, R. Qu, and J. Lin, “Benchmarking LLM faithfulness in
RAG with evolving leaderboards,” inProceedings of the 2025 Conference
on Empirical Methods in Natural Language Processing: Industry Track
(2025) arXiv:2505.04847.
29F. Bao, M. Li, R. Luo, and O. Mendelevitch, “HHEM-2.1-Open:
Hughes hallucination evaluation model,” (2024), vectara;https://
huggingface.co/vectara/hallucination_evaluation_model.
30S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “RAGAs: Auto-
mated evaluation of retrieval augmented generation,” inProceedings of the
18th Conference of the European Chapter of the Association for Computa-
tional Linguistics: System Demonstrations(2024) pp. 150–158.
31K. Mao, Z. Liu, H. Qian, F. Mo, C. Deng, and Z. Dou, “RAG-Studio: To-
wards In-Domain Adaptation of Retrieval Augmented Generation ThroughSelf-Alignment,” inFindings of the Association for Computational Lin-
guistics: EMNLP 2024(Association for Computational Linguistics, Miami,
Florida, USA, 2024) pp. 725–735.
32A. Edelen, C. Mayes, D. Bowring, D. Ratner, A. Adelmann, R. Ischebeck,
J. Snuverink, I. Agapov, R. Kammering, J. Edelen, I. Bazarov, G. Valentino,
and J. Wenninger, “Opportunities in machine learning for particle accelera-
tors,” (2018), arXiv:1811.03172.
33J. Kaiser, C. Xu, A. Eichler, A. Santamaria Garcia, O. Stein, E. Brün-
dermann, W. Kuropka, H. Dinter, F. Mayet, T. Vinatier, F. Burkart, and
H. Schlarb, “Reinforcement learning-trained optimisers and Bayesian op-
timisation for online particle accelerator tuning,” Scientific Reports14,
15733 (2024).
34R. Humble, W. Colocho, F. O’Shea, D. Ratner, and E. Darve, “Resilient
V AE: Unsupervised Anomaly Detection at the SLAC Linac Coherent Light
Source,” EPJ Web of Conferences295, 09033 (2024).
35W. Blokland, K. Rajput, M. Schram, T. Jeske, P. Ramuhalli, C. Peters,
Y . Yucesan, and A. Zhukov, “Uncertainty aware anomaly detection to pre-
dict errant beam pulses in the Oak Ridge Spallation Neutron Source accel-
erator,” Physical Review Accelerators and Beams25, 122802 (2022).
36K. Rajput, M. Schram, W. Blokland, Y . Alanazi, P. Ramuhalli, A. Zhukov,
C. Peters, and R. Vilalta, “Robust errant beam prognostics with conditional
modeling for particle accelerators,” Machine Learning: Science and Tech-
nology5(2024), dOI: 10.1088/2632-2153/ad2e18.
37I. Lobach and M. Borland, “Recurrent neural networks for anomaly detec-
tion in magnet power supplies of particle accelerators,” Machine Learning
with Applications18, 100585 (2024).
38J. P. Edelen and N. M. Cook, “Anomaly Detection in Particle Accelerators
using Autoencoders,” Tech. Rep. (2021) arXiv:2112.07793 [physics.acc-
ph].
39M. H. Prince, H. Chan, A. Vriza, T. Zhou, V . K. Sastry, Y . Luo, M. T. Dear-
ing, R. J. Harder, R. K. Vasudevan, and M. J. Cherukara, “Opportunities for
retrieval and tool augmented large language models in scientific facilities,”
npj Computational Materials10, 251 (2024).
40F. Mayet, “GAIA: A General AI Assistant for Intelligent Accelerator Oper-
ations,” (2024), arXiv:2405.01359 [cs.CL].
41K. Suresh, N. Kackar, L. Schleck, and C. Fanelli, “Towards a RAG-
based Summarization Agent for the Electron-Ion Collider,” (2024),
arXiv:2403.15729 [cs.CL].
42A. Sulc, A. Bien, A. Eichler, D. Ratner, F. Rehm, F. Mayet, G. Hart-
mann, H. Hoschouer, H. Tuennermann, J. Kaiser, J. S. John, J. Maldonado,
K. Hazelwood, R. Kammering, T. Hellert, T. Wilksen, V . Kain, and W.-
L. Hu, “Towards Unlocking Insights from Logbooks Using AI,” (2024),
arXiv:2406.12881 [physics.acc-ph].
43J. Maldonado, “Enhancing Electronic Logbooks Using Machine Learning,”
.
44L. Stuhlmann, M. Sapinski, Q. Dai, and R. Ischebeck, “Intelligent Retrieval
Systems for Accelerator Physics: Matching Retrieval Approaches to Use
Cases,” .
45T. Hellert, D. Bertwistle, S. C. Leemann, A. Sulc, and M. Venturini, “Agen-
tic artificial intelligence for multistage physics experiments at a large-scale
user facility particle accelerator,” Physical Review Research8, L012017
(2026), arXiv:2509.17255.
46A. Vriza, M. H. Prince, T. Zhou, H. Chan, and M. J. Cherukara, “Operating
advanced scientific instruments with AI agents that learn on the job,” npj
Computational Materials12, 160 (2026).
47A. Sulc, T. Hellert, R. Kammering, H. Hoschouer, and J. St. John, “To-
wards agentic AI on particle accelerators,” inMachine Learning and the
Physical Sciences Workshop, 38th Conference on Neural Information Pro-
cessing Systems (NeurIPS)(2024) arXiv:2409.06336.
48J. Kaiser, A. Lauscher, and A. Eichler, “Large language models for
human-machine collaborative particle accelerator tuning through natural
language,” Science Advances11, eadr4173 (2025).
49R. Jiang, D. Fu, C. Jiang, T. Yang, Z. Wang, Y . Wu, Y . Ban, Y . Mao, and
Q. Li, “Agentic Hybrid RAG for Evidence-Grounded Muon Collider Anal-
ysis,” (2026), arXiv:2606.10381 [hep-ex].
50Qdrant, “Qdrant: Vector database for the next generation of AI applica-
tions,”https://qdrant.tech(2026), accessed: 2026-08-22.
51Elastic N.V ., “Elasticsearch: The official distributed search & analytics
engine,”https://www.elastic.co/elasticsearch(2026), accessed:
2026-08-22.

APS-RAG 16
52Neo4j, Inc., “Neo4j graph database platform,”https://neo4j.com
(2026), accessed: 2026-08-22.
53M. Borland, “User‘s guide for SDDS toolkit Version 1.4,” Tech. Rep.
ANL/ASD/RP–88322, 179270 (1995).
54Artifex Software, “Pymupdf: A high performance python library for data
extraction, analysis, conversion & manipulation of pdf documents,” (2026).
55R. Smith, “An overview of the tesseract ocr engine,” inNinth International
Conference on Document Analysis and Recognition (ICDAR 2007), V ol. 2
(IEEE Computer Society, 2007) pp. 629–633.
56P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning,
“Raptor: Recursive abstractive processing for tree-organized retrieval,”
(2024), arXiv:2401.18059 [cs.CL].
57L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and
F. Wei, “Text embeddings by weakly-supervised contrastive pre-training,”
(2022), arXiv:2212.03533.
58S. Bruch, S. Gai, and A. Ingber, “An Analysis of Fusion Functions for
Hybrid Retrieval,” ACM Transactions on Information Systems42, 1–35
(2024), arXiv:2210.11934 [cs.IR].
59LangChain AI, “LangGraph,” (2024),https://github.com/
langchain-ai/langgraph.
60J. Baek, A. Hussain, D. Liu, N. Vincent, and L. H. Kim, “Open we-
bui: An open, extensible, and usable interface for ai interaction,” (2025),
arXiv:2510.02546 [cs.HC].
61L. Zheng, W.-L. Chiang, Y . Sheng, S. Zhuang, Z. Wu, Y . Zhuang, Z. Lin,
Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judg-
ing LLM-as-a-judge with MT-bench and Chatbot Arena,” inAdvances in
Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and
Benchmarks Track(2023) arXiv:2306.05685.
62A. Panickssery, S. R. Bowman, and S. Feng, “LLM evaluators recognize
and favor their own generations,” inAdvances in Neural Information Pro-
cessing Systems 37 (NeurIPS 2024)(2024) arXiv:2404.13076.
63F. Wang, Y . Li, and H. Xiao, “jina-reranker-v3: Last but not late interaction
for listwise document reranking,” (2025), arXiv:2509.25085.
64L. van der Maaten and G. Hinton, “Visualizing data using t-SNE,” Journal
of Machine Learning Research9, 2579–2605 (2008).
65Author to whom correspondence should be addressed: rsainju@anl.gov.
66X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for lan-
guage image pre-training,” inProceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV)(2023) pp. 11975–11986.
67M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré,
M. Lomeli, L. Hosseini, and H. Jégou, “The Faiss library,” (2024),
arXiv:2401.08281.
68J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, “M3-embedding:
Multi-linguality, multi-functionality, multi-granularity text embeddings
through self-knowledge distillation,” inFindings of the Association for
Computational Linguistics: ACL 2024(2024) arXiv:2402.03216.
69W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “MiniLM: Deep
self-attention distillation for task-agnostic compression of pre-trained trans-
formers,” inAdvances in Neural Information Processing Systems, V ol. 33
(2020) pp. 5776–5788.
70N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings us-
ing Siamese BERT-networks,” inProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing and the 9th In-
ternational Joint Conference on Natural Language Processing (EMNLP-
IJCNLP)(2019) pp. 3982–3992.
71T. Hellert, J. Montenegro, M. Venturini, and A. Pollastro, “Domain-specific
text embedding model for accelerator physics,” Physical Review Accelera-
tors and Beams28, 044601 (2025).
72Y . Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang,
D. Liu, J. Lin, F. Huang, and J. Zhou, “Qwen3 Embedding: Advanc-
ing Text Embedding and Reranking Through Foundation Models,” (2025),
arXiv:2506.05176 [cs].

APS-RAG 17
Supplementary material
A corrective agentic hybrid RAG and an operations-grounded
evaluation for a scientific facility
Rajat Sainju,65Dariusz Jarosz, Hairong Shang, Michael Prince,
Ryan M. Aydelott, Mathew J. Cherukara, Yine Sun, and Michael D. Borland
Advanced Photon Source, Argonne National Laboratory, Lemont, Illinois 60439, USA
S1. NATIVE-TOOL REACT AND THE MCP TOOLING
LAYER
APS-RAG executes tools using a dual-path ReAct executor.
The primary execution path employs native function calling
through a vendor-aware /chat/ client, NativeToolsArgoLLM,
which generates structured tool calls compatible with Ope-
nAI, Gemini, and Anthropic backbone models. This client ac-
commodates three distinct response-envelope formats: object,
list, and bare string. A legacy prompt-based TOOL_CALL
regular expression path is maintained as a fallback mecha-
nism. Tool functionality is provided by four FastMCP servers,
each registered in .mcp.json (see Table S1). The archiver
server supports PV and device resolution as well as historical
PV data access through channel_find and channel_read. The
graph server enables knowledge-graph search. The compute
server provides calculator operations, sandboxed code execu-
tion, and tabular analysis. The image server supports reverse-
image and text-to-image search over logbook attachments us-
ing SigLIP66and FAISS67. In total, eight tools are available.
Each tool returns a standardized envelope containing status,
result, and error fields. In-tool exceptions are intercepted by
a decorator, ensuring that exceptions do not propagate across
the standard I/O boundary. A persistent session pool main-
tains a dedicated client session for each server throughout the
host process lifetime. This approach amortizes the substantial
memory requirements of SigLIP and FAISS models, as well
as the initialization overhead of Argo HTTP, both of which
would otherwise contribute significantly to per-call latency. If
a transport failure occurs, the affected session is terminated
and reestablished. The robustness of this layer is validated by
818 tests, comprising 754 unit tests and 64 integration tests.
These include contract tests aligned with each stage for naive-
RAG redirection, native function calling, MCP server opera-
tions, and the persistent session pool, as well as an integration
test that verifies ReAct retry behavior in response to tool er-
rors.
TABLE S1. The four FastMCP servers and their eight tools.
Server Tools
archiver channel_find, channel_read
graph graph_search
compute calculator, code_executor, data_analysis
image image_search [reverse-image, text-to-image]S2. COMPONENT-LATENCY STUDY AND RERANKER
A/B
Timing instrumentation for each pipeline component was
achieved using a non-invasive harness implemented via class-
level monkey patching withperf_counter. Query-level tim-
ing isolation was accomplished through a context variable.
For the naive path, a module-global timing dictionary was
utilized, with retrieval operations distributed across a thread
pool. Table S2 summarizes deployment-latency measure-
ments by planner model, evaluated on a 10-query advanced-
reasoning workload. Measurements were collected in May
2026 using the pre-freeze deployment stack. These results
characterize the deployed service and are not intended as
benchmark measurements on a frozen stack.
The planner is the language model responsible for gener-
ating the structured multi-step retrieval and reasoning plan; it
serves as the independent variable and is varied across rows,
while all other stages use fixed models (claude-sonnet-4.6 for
the router and enhancer, GPT-5.4 for synthesis, and Jina v3 for
reranking), subject to the isolation caveat noted above. Wall
time is the end-to-end query latency from submission to the
final cited answer, computed per query and averaged over the
query set; it is less than the sum of the component columns be-
cause sub-retrievers execute in parallel and reranking is nested
within retrieval. The enhancer expands the input query into
semantic variants and, in a single combined call, also per-
forms intent routing and — for agentic queries — plan gener-
ation, consolidating functions previously handled by separate
router and planner calls. Retrieval is hybrid, fusing concurrent
dense-vector retrieval (Qdrant with e5-large-v2 query embed-
dings), lexical retrieval (BM25 in Elasticsearch), and multi-
hop knowledge-graph retrieval (Neo4j) via reciprocal-rank fu-
sion; it includes query embedding, and its wall time encom-
passes the reranker stage. The reranker applies cross-encoder
reranking (Jina v3, GPU-local, top-k = 50) to the fused can-
didate pool to select the final context set, and although re-
ported separately, executes within retrieval. Finally, synthe-
sis composes a grounded, citation-annotated answer from the
reranked evidence.
A five-batch reranker A/B evaluation compared the Jina v3
cross-encoder63with the legacy LLM scorer on advanced-
reasoning queries, usingreranker_top_k=50 and active
corrective rounds. Results show that both approaches have
similar wall time, although the LLM scorer exhibits a high-
variance tail. This comparison addresses latency only. In
contrast, the quality comparison, as reported in Table IV of
the main text, strongly favors Jina v3, which constitutes the
principal finding. Per-query analysis highlights the quality
gap: the legacy scorer not only produces inferior rankings

APS-RAG 18
TABLE S2. Per-component wall time (s) by planner model, 10 advanced-reasoning queries (deployment study, May 2026, pre-freeze stack).
Component times can sum to more than the wall time because retrieval fans out in parallel. The per-component wall time also depends on
Argo latency, traffic, chain-of-thinking triggers, and corrective loops. Note: enhancer = query-preprocessing + router + planner. The added
latency by embedding models is∼30 ms.
Planner Wall (s) Enhancer (s) Retrieval (s) Reranker (s) Synthesis (s)
Gemini 2.5 Flash 88.8±24.3 31.0±29.1 65.1±21.6 18.2±16.6 6.5±1.5
Gemini 2.5 Pro 96.0±29.1 29.7±12.6 56.1±29.8 19.2±17.4 8.9±3.3
GPT-5.4 105.9±11.6 28.4±13.5 60.0±31.4 18.8±17.0 8.5±3.8
Claude Opus 4.6 140.4±44.5 71.7±59.6 109.7±34.3 18.8±17.1 8.5±3.8
but also frequently fails to generate usable results. Specifi-
cally, NDCG@10 is less than 0.02 for 63% of the 19 subset
questions and less than 0.10 for 68%, with strict vital recall
per question worse on 12 of 19, unchanged on 7, and better
on none. The comparison among competent rerankers is re-
flected in the single-digit spread of Tables S6 and S7. The
LLM-scorer default row in main-text Table IV should be in-
terpreted as a degraded fallback rather than as a peer reranker.
A five-batch reranker A/B evaluation compared the Jina v3
cross-encoder63with the legacy LLM scorer on advanced-
reasoning queries, usingreranker_top_k=50 and active
corrective rounds. Results show that both approaches have
similar wall time, although the LLM scorer exhibits a high-
variance tail. This comparison addresses latency only. In
contrast, the quality comparison, as reported in Table IV of
the main text, strongly favors Jina v3, which constitutes the
principal finding. Per-query analysis highlights the quality
gap: the legacy scorer not only produces inferior rankings
but also frequently fails to generate usable results. Specifi-
cally, NDCG@10 is less than 0.02 for 63% of the 19 subset
questions and less than 0.10 for 68%, with strict vital recall
per question worse on 12 of 19, unchanged on 7, and better
on none. The comparison among competent rerankers is re-
flected in the single-digit spread of Tables S6 and S7. The
LLM-scorer default row in main-text Table IV should be in-
terpreted as a degraded fallback rather than as a peer reranker.
S3. RELIABILITY-FOCUSED ENGINEERING EFFORT
Deployment within live operational environments necessi-
tated robust adaptation to ongoing infrastructure drift. The
following representative events were identified, each detected
or confirmed through the latency harness and safeguarded
by the test suite. First, an Argo gateway migration was ob-
served, which eliminated persistent 502 retry storms. Second,
a response-envelope mismatch in the/chat/endpoint, char-
acterized by a bare-string response shape, silently disrupted
planner functionality until appropriate handling was imple-
mented. Third, an acronym-resolution chain was silently dis-
abled in production due to a path, import, or API drift bug.
Fourth, phantom corrective rounds were triggered by scoring
a post-formatted draft, which was resolved by instead scoring
the raw draft. Finally, planner-sentinel false positives were
observed in tool-skip accounting. These events collectively
illustrate the need for comprehensive monitoring and rigorous
testing to ensure system resilience amid infrastructure evolu-tion.
S4. THE APS-RAG RETRIEVAL SKILL
The retrieval layer is implemented as a self-contained agent
skill,aps-rag, which can be invoked directly from coding as-
sistants such as Claude Code and Codex. The skill exposes the
corpus as a retrieval-as-a-service primitive: rather than em-
bedding a generator within the tool, it returns evidence pas-
sages for synthesis by the host model. Retrieval and genera-
tion are thus decoupled; the skill is responsible solely for sur-
facing relevant passages, while the calling model composes
and cites the answer. This design enables agents that process
tool outputs to reuse the indexed knowledge base without re-
quiring the full agentic stack.
The skill is structured as three decoupled tiers. (i) A state-
less FastAPI gateway performs lexical (BM25) retrieval di-
rectly against the live Elasticsearch indices; it contains no
LLM and is strictly read-only. The gateway shares only the in-
dices with the production agentic system, with no shared pro-
cess, model, or datastore beyond Elasticsearch. As a result,
the lightweight skill path and the corrective-agentic service
are operationally independent and do not affect each other’s
stability. (ii) A thin MCP server fronts the gateway and ex-
poses two tools to the host model:query_aps_rag, which is-
sues a query and returns the top-kranked passages, each anno-
tated with its source collection, document identifier, passage
text, and retrieval score, andaps_rag_health, which reports
gateway liveness and index availability. The MCP layer con-
tains no retrieval or generation logic and serves solely as a
protocol bridge between the agent runtime and the gateway.
(iii) The host model (Claude Code or Codex) consumes the re-
turned passages and synthesizes the final response, attributing
each claim to the (collection, document id) of the supporting
passage, ensuring that the answer includes verifiable, corpus-
grounded citations.
The design is intentionally minimal. Limiting the skill
to BM25 retrieval over shared indices ensures low and pre-
dictable latency, deterministic behavior, and read-only safety,
while avoiding redundant generation and any dependency on
the embedding, reranking, knowledge-graph, or LLM compo-
nents of the agentic system. The skill can be deployed wher-
ever the Elasticsearch indices are accessible. Distribution is
via an internal GitLab repository, accompanied by an install
script that registers the MCP server in the agent configuration
and specifies the gateway endpoint, enabling/aps-ragas a

APS-RAG 19
first-class command within existing agent sessions.
S5. ANSWER-MATCH METRICS: EXACT MATCH, F1,
AND ACCURACY
To facilitate comparison with the short-form question an-
swering literature, such as SQuAD and Natural Questions,
which report exact match (EM), F1, and containment accu-
racy metrics as used in Adaptive-RAG, Self-RAG, FLARE,
and related retrieval-augmented QA studies, we compute three
corresponding answer-match metrics. Accuracy and F1 are
presented alongside the main-text generation results in Tables
II and IV , while EM is detailed below. Two methodological
adaptations are necessary for this comparison. First, the APS-
Bench gold standard consists of long-form answers, compris-
ing a grounded paragraph and atomic nuggets, rather than a
short factoid span. Therefore, we derive a concise reference
answer for each question. Second, our systems generate long-
form operational answers rather than short spans, which ren-
ders a verbatim whole-response match uninformative and lim-
its the precision of token overlap when compared to a short
reference. Both adaptations are addressed in the following
sections.
a. Short reference answers.For each answerable ques-
tion we extract a short canonical answera⋆(a value with units,
a PV/device name, a command, or a minimal phrase) together
with an alias setAof equivalent surface forms (with/with-
out units, numeric vs. spelled, common abbreviations) from
the long-form gold answer and its vital nuggets, using the
generator-family model (gpt54, temperature 0). These are
LLM-extracted and carry the same These short reference an-
swers are subject to the same validation-pending status as the
gold standard. We define the setG={a⋆}∪A, where ˆyde-
notes the system’s answer. All string comparisons are per-
formed using SQuAD normalizationN(·), which includes
lowercasing, removal of punctuation and articles (a, an, the),
and whitespace normalization.
b. EM (exact match; strictest).The exact match metric
assigns EM=1 if there exists ag∈Gsuch thatN(g)appears
as a contiguous substring withinN(ˆy); otherwise, EM=0.
For paragraph-length answers, whole-response string equal-
ity, as used in standard EM, is nearly always zero. There-
fore, we report exact-span containment, which represents the
strictest form of meaningful containment. In this context, EM
serves as a conservative lower bound on verbatim agreement
and should not be interpreted as a measure of overall system
accuracy.
c. F1 (token overlap).The standard token-levelF1 be-
tween the reference and the best-matching span of the re-
sponse. ForN(g)of token lengthℓwe slide windows of
lengthℓ±2 over the response tokens and take the maximum
over windows and overg∈G,
F1=max
g∈Gmax
windoww2PR
P+R,P=|w∩g|
|w|,R=|w∩g|
|g|,
Here,| · ∩ · |denotes the multiset token overlap. Comput-
ing F1 over the best-matching span, rather than the entire re-sponse, mitigates the loss of precision that would otherwise
result from comparing a long answer to a short reference.
d. Acc (containment accuracy).Whether the response
contains the correct answer. Lexical matching systematically
under-credits correct paraphrases, synonyms, and equivalent
units (e.g. 14.5 nC” vs. 14.5 nanocoulombs’‘, accumulator
ring” vs. Particle Accumulator Ring’‘); we verified that token
F1/EM track the semantic vital-nugget recall while genuine
correctness is higher. Accuracy is therefore judgedsemanti-
callyby the cross-family primary judge (claudeopus46; An-
thropic̸=generatorgpt54, OpenAI—the same generator,̸=
,judge guard used for every other LLM-judged metric here).
Given the question, the gold answer (anda⋆), and the re-
sponse, the judge returns a binary decision on whether the
response states the correct answer, explicitly accepting para-
phrase, synonymy, equivalent units, and additional correct
context; Acc=1 for “yes”.
Aggregation and status. Each metric is computed as the
average over all answerable questions and across three eval-
uation seeds, with standard deviation approximately zero at
temperature zero. The metrics are additive, as they are de-
rived from cached system outputs without requiring additional
system runs, and they do not affect nugget-recall, faithful-
ness, or operational evaluation layers. On APS-Bench, the
Agentic GraphRAG system achieves EM, F1, and Acc val-
ues of 32.7%, 70.3%, and 69.4%, respectively, which are con-
sistent with its semantic strict-vital-nugget recall. The con-
cordance between the independent lexical (F1) and seman-
tic (Acc, nugget recall) metrics indicates that the answer-
correctness measurement is robust and not an artifact of any
single matching approach. Since the systems generate long-
form operational answers rather than short spans, EM is nec-
essarily low and should not be interpreted as a measure of
system accuracy. Instead, Acc, which reflects semantic con-
tainment, is the appropriate metric for determining whether
the response contains the correct information. Both the short
reference answers and the Acc judge are produced by large
language models.

APS-RAG 20
TABLE S3. Query-type-adaptive reciprocal-rank-fusion weights. The router recognizes eight operational intents; they carry a hand-set weight
profile(dense,BM25,graph). Weights are set from operational experience, not learned.
Intent Dense BM25 Graph Emphasis
factual 0.20.50.3 BM25
value 0.10.60.3 BM25 (numerical)
temporal 0.30.50.2 BM25 (date/shift tokens)
troubleshooting 0.3 0.20.5graph
multi_hop 0.3 0.20.5graph
comparative0.40.3 0.3 balanced
code 0.5 0.5 0.0 balanced
general0.40.3 0.3 balanced
TABLE S4. Construction-time composition of the two benchmarks (n=50 each), reported side by side and never pooled. “Gold min-rank” is
the mean rank at which the first gold document is retrieved (lower is easier).
Property APS-Bench
Answerable / abstention 50 / 0
Mean nuggets per question 6.27
Mean gold-doc min-rank 0.41
Query types
factual / temporal 7 / 6
multi-hop / procedural 8 / 8
causal / comparative 6 / 8
troubleshooting / parametric 5 / 2
Difficulty
easy / medium / hard 15 / 25 / 10
TABLE S5. Effect of embedding model on retrieval (APS-Bench; 10k-index diagnostic, first-stage, no reranker; mean over subsamples).
Models: e5-large-v257, bge-m368, all-MiniLM-L12-v269,70, and the domain-pretrained accphysbert71.
Embedding Model Model Size MRR@10 (%) NDCG@10 (%) Recall@10 (%)
BM25 — 82.1 69.4 63.1
e5-large-v2 vector 335M 70.2 57.2 52.8
e5-large-v2 hybrid 335M 79.2 67.5 61.9
bge-m3 vector 568M 79.2 65.6 60.6
bge-m3 hybrid 568M 85.1 72.3 65.7
all-MiniLM-L12-v2 vector 33M 60.6 48.6 43.9
all-MiniLM-L12-v2 hybrid 33M 77.8 64.1 59.9
accphysbert vector∼110M 58.5 43.1 37.1
accphysbert hybrid∼110M 74.6 60.9 57.0
TABLE S6. Effect of re-ranker model over BM25 first-stage retrieval (APS-Bench; 10k-index diagnostic). Rerankers: jina-reranker-v363,
bge-reranker-v2-m368, and qwen3-reranker72. Precision@10 is not reported because each gold answer spans 2–3 passages.
Reranker Model Model Size MRR@10 (%) NDCG@10 (%) Recall@10 (%)
(no rerank) — 62.1 49.4 43.1
jina-reranker-v3 0.6B 86.9 73.8 64.1
bge-reranker-v2-m3 0.6B 83.3 71.5 64.6
qwen3-reranker-4B 4B 84.7 71.9 66.4
TABLE S7. Effect of re-ranker model over hybrid (BM25 + vector, RRF) first-stage retrieval (APS-Bench; 10k-index diagnostic). Preci-
sion@10 is not reported because each gold answer spans 2–3 passages.
Reranker Model Model Size MRR@10 (%) NDCG@10 (%) Recall@10 (%)
(no rerank) — 64.2 57.6 47.2
jina-reranker-v3 0.6B 86.1 73.7 64.8
bge-reranker-v2-m3 0.6B 83.0 71.3 65.3
qwen3-reranker-4B 4B 85.7 71.9 67.3

APS-RAG 21
S6. SUPPLEMENTARY FIGURES

APS-RAG 22
{
"$schema": "corpus_v2",
"$id": "aps-rag/corpus-record.json",
"title": "APS-RAG unified corpus record",
"$comment": "One indexed chunk. Every collection is normalized to CorpusRecord; work_requests adds chunk-provenance fields.",
"$defs": {
"CorpusRecord": {
"type": "object",
"additionalProperties": false,
"required": ["doc_id", "content", "title", "author", "collection", "system", "creation_datetime", "file_id"],
"properties": {
"doc_id": {"type": "string", "pattern": ".+__chunk_\\d+$", "description": "chunk id ’<source>__chunk_<n>’"},
"content": {"type": "string", "minLength": 1, "description": "chunk body: BM25-analyzed and embedded"},
"title": {"type": "string"},
"author": {"type": "string", "description": "may be empty"},
"collection": {
"type": "string",
"enum": ["bely_logbook", "icms_collection", "work_requests",
"teams_messages", "sdds_collection", "oag_collection"]
},
"system": {"type": "string", "description": "accelerator system; may be empty"},
"creation_datetime": {
"anyOf": [{"type": "string", "format": "date-time"},
{"type": "string", "maxLength": 0}]
},
"file_id": {"type": "string"}
}
},
"WorkRequestRecord": {
"$comment": "work_requests: CorpusRecord plus chunk-provenance fields",
"allOf": [{"$ref": "#/$defs/CorpusRecord"}],
"unevaluatedProperties": false,
"properties": {
"parent_id": {"type": "string"},
"chunk_index": {"type": "integer", "minimum": 0},
"chunk_type": {"type": "string"}
},
"required": ["parent_id", "chunk_index", "chunk_type"]
}
},
"oneOf": [
{"allOf": [{"$ref": "#/$defs/CorpusRecord"}],
"properties": {"collection": {"not": {"const": "work_requests"}}}},
{"allOf": [{"$ref": "#/$defs/WorkRequestRecord"}],
"properties": {"collection": {"const": "work_requests"}}}
]
}
FIG. S1. A unified corpus-record schema, based on JSON Schema, is implemented across all eight collections. Each indexed chunk is validated
againstCorpusRecord, while thework_requestscollection is further validated againstWorkRequestRecord, constrained by the terminal
oneOf. The eight required fields facilitate standardized filtering, temporal scoping, and source citation across otherwise heterogeneous sources.
Among these,contentis the only field subjected to BM25 analysis and embedding.

APS-RAG 23
{
"$schema": "v2",
"$id": "aps-rag/bely-logbook-source.json",
"title": "BELY logbook native source schema (nested: file -> entry -> sub-entry)",
"$comment": "24 files, 4536 entries, 29567 sub-entries. Each entry is flattened to CorpusRecord at index time.",
"$ref": "#/$defs/BelyLogbookFile",
"$defs": {
"BelyLogbookFile": {
"type": "object",
"additionalProperties": false,
"required": ["subfolder_name", "total_entries", "processing_timestamp", "entries"],
"properties": {
"subfolder_name": {"type": "string", "description": "beamline/section, e.g. ’25id’, ’studies-sr’"},
"total_entries": {"type": "integer", "minimum": 0},
"processing_timestamp":{"type": "string", "format": "date-time"},
"entries": {"type": "array", "items": {"$ref": "#/$defs/BelyLogbookEntry"}}
}
},
"BelyLogbookEntry": {
"type": "object",
"$comment": "Operational entries leave studies_plan/summary empty; SR studies entries populate them (subtype by value of ’logbook
’).",
"additionalProperties": false,
"required": [
"BELY_ID", "version", "logbook", "machine/system", "bely_filename",
"bely_creation_datetime", "created_by", "topic", "personnel_involved",
"keywords", "studies_plan", "studies_summary", "bely_sub_entries",
"processing_timestamp"
],
"properties": {
"BELY_ID": {"type": "string", "description": "numeric-string id, e.g. ’10617’"},
"version": {"type": "string"},
"logbook": {"type": "string", "description": "source logbook / discriminator, e.g. ’Operations’, ’SR’, ’25-ID’"},
"machine/system":{"type": "string"},
"bely_filename": {"type": "string"},
"bely_creation_datetime": {"type": "string", "format": "date-time"},
"created_by": {"type": "string"},
"topic": {"type": "string"},
"personnel_involved": {"type": "array", "items": {"type": "string"}},
"keywords": {"type": "array", "items": {"type": "string"}},
"studies_plan": {"type": "string", "description": "populated only for SR studies; ’’ otherwise"},
"studies_summary": {"type": "string", "description": "populated only for SR studies; ’’ otherwise"},
"bely_sub_entries": {"type": "array", "items": {"$ref": "#/$defs/BelySubEntry"}},
"processing_timestamp": {"type": "string", "format": "date-time"}
}
},
"BelySubEntry": {
"type": "object",
"additionalProperties": false,
"required": ["entry_text", "attachment_urls"],
"properties": {
"entry_text": {"type": "string"},
"attachment_urls": {"type": "array", "items": {"type": "string", "format": "uri"}}
}
}
}
}
FIG. S2. The BELY logbook native source schema is unique in its composite, three-level nested structure. Each group corresponds to a
BelyLogbookFiletemplate (24 files in the frozen index), which containsBelyLogbookEntryrecords (4,536). Each entry subsequently
holds chronologicalBelySubEntryposts. During indexing, entries and their sub-entries are flattened into the unified corpus record.

APS-RAG 24
FIG. S3. Per-question faithfulness on APS-Bench is reported, with
each cell representing the per-question faithfulness (1−hallucination
rate) of a system. Questions are grouped by query type along the ver-
tical axis. The overlay indicates whether each cell is hallucination-
free (v) or hallucinated (x), based on the HHEM-2.1 backstop with a
threshold of 0.5. Failures cluster by question rather than by system;
a small subset of challenging items, particularly two troubleshoot-
ing questions, are hallucinated by all five variants. Consequently, the
faithfulness tail is primarily determined by question difficulty rather
than system architecture.
FIG. S4. The sensitivity of answer quality to the corrective-trigger
thresholdτis evaluated on the APS-Bench ablation subset (main-text
Sec. V C), with the round cap fixed at three. The left axis presents
strict vital recall, lenient vital recall, and claim-level faithfulness
(with 95% bootstrap confidence interval bands), while the right axis
shows the fraction of answers differing from theτ=0.3 baseline.
The vertical dotted line indicates the deployed defaultτ=0.4. All
three answer metrics increase modestly fromτ=0.3, plateau near
τ=0.7 to 0.8, and decrease atτ=0.9. The answer-change rate
reaches saturation at 1.0 byτ=0.5, as higher thresholds trigger re-
retrieval for nearly every question. Given the wide and overlapping
confidence intervals at this subset size, the sweep suggests low sensi-
tivity around the deployed default, rather than providing justification
for retuning.
FIG. S5. A two-dimensional t-SNE64projection of e5-large-v2 em-
beddings is presented for corpus documents from six APS collections
(BELY , ICMS, Work Requests, Teams, SDDS, OAG) in APS-Bench.
The collections occupy distinct but partially overlapping regions,
with cross-collection overlap where the same devices and faults re-
cur. This observed heterogeneity provides empirical motivation for
the hybrid, graph-augmented design. The projection is performed at
the corpus level and remains independent of the benchmark.

APS-RAG 25
FIG. S6. The effect of the cross-encoder reranker on first-stage re-
trieval quality is evaluated on APS-Bench, with the top-100 can-
didates reranked over a fixed 10,000-document diagnostic corpus.
MRR@10 is reported for (a) a BM25 (Naive-RAG) first stage and
(b) an RRF hybrid (Hybrid-RAG) first stage, each evaluated with-
out reranking and with three cross-encoders (jina-reranker-v3, bge-
reranker-v2-m3, Qwen3-Reranker-4B). Error bars represent 95%
bootstrap confidence intervals. The jina-reranker-v3 achieves the
highest point estimate in both settings, but the differences among
the three rerankers are within single digits, and their confidence in-
tervals overlap. This comparison is among functioning rerankers and
should be interpreted alongside the component ablation (main-text
Table IV), where removing the cross-encoder with direct LLM as-
sessment, rather than a competent alternative, results in a substantial
performance drop.
FIG. S7. Retrieval quality by embedding model is assessed on APS-
Bench using a fixed 10,000-document diagnostic corpus, with 95%
bootstrap confidence interval error bars, for both vector-only and
hybrid (vector plus BM25) retrieval. Metrics reported include (a)
MRR@10 and (b) Recall@10. Four models are compared: the de-
ployed default e5-large-v2, bge-m3, all-MiniLM-L12-v2, and the
domain-pretrained accphysbert. The dashed line indicates first-stage
naive BM25. The results show an ordered spread (Table S5): vector-
only MRR@10 is highest for bge-m3 (≈79%), followed by e5-large-
v2 (≈70%), with all-MiniLM-L12-v2 and accphysbert lower (≈59–
61%). Hybrid fusion increases these values to approximately 75–
85%, and first-stage BM25 alone is strong (≈82%). Notably, the
95% confidence intervals overlap, so the ordering is suggestive rather
than definitive. The domain-pretrained accphysbert does not outper-
form the general-purpose models; bge-m3 only marginally surpasses
e5-large-v2, and the cross-encoder reranker remains the most signif-
icant retrieval improvement (main-text Table IV).

APS-RAG 26
FIG. S8. This figure examines the sensitivity of strict vitalV strictand hallucination rate to the choice of synthesis large language model (LLM).
The complete Agentic GraphRAG pipeline was re-executed on APS-Bench, systematically substituting the synthesis model across ten LLMs
from four vendors, while keeping retrieval, fusion, reranking, and the cross-family judge constant. For each model, the dark bar represents
Vstrict(left axis; higher values indicate better performance), and the light bar indicates the claim-level hallucination rate (right axis; lower values
are preferable). Vendor is encoded by color; hatching differentiates open-weight from closed-weight models; release month is displayed below
each model; and parameter counts are provided for open-weight models. Anthropic models are intentionally excluded because Claude serves
as the fixed judge, ensuring that the generator and judge families are always distinct. The deployed generator (GPT-5.4,gpt54via ARGO)
matches the full-system values reported in main-text Table II (70.3%/8.7%). Two primary patterns emerge from this analysis. First, strict
vital recall demonstrates relative insensitivity to the generator; the ten-model range is 58.1–70.3%, and the best open-weight model (GPT-OSS
120B, 67.7%) is within 2.6 points of the closed-weight frontier. This finding aligns with the expectation that recall is primarily determined
by the fixed retrieval and reranking front end. Second, hallucination rates vary by more than a factor of five (5.6–30.6%) and do not correlate
with either recall or parameter count. Notably, the 31B open-weight Gemma4 model exhibits the highest faithfulness (5.6%, outperforming the
deployed GPT-5.4 at 8.7%), while the 400B–675B open-weight models are the least faithful (30.1–30.6%). Additionally, faithfulness improves
across successive generations within the OpenAI model line (22.8% to 10.3% to 8.7%). The decoupling of recall and faithfulness observed in
the component ablation (main-text Sec. V C) is thus also evident at the generator level. All values are point estimates based on a single frozen
answer set; recall differences of a few points fall within the bootstrap confidence interval widths reported in main-text Table II, so the precise
ranking should be interpreted as suggestive rather than definitive.

APS-RAG 27
FIG. S9. Corrective-loop behavior is analyzed on the APS-Bench ablation subset (main-text Sec. V C). (a) Claim-level faithfulness score (left
axis; mean with 95% bootstrap confidence interval band) and the fraction of answers differing from the single-pass baseline (answer-change
rate vs. cap=0; right axis, dashed) are reported as a function of the maximum number of configured corrective rounds. The sweep is conducted
at a diagnostic trigger threshold ofτ=0.9, intentionally set above the deployedτ=0.4 so that the corrective loop activates for nearly every
question; absolute levels are therefore not directly comparable to the deployed-configuration tables. Allowing a single round results in nearly
every answer being revised (answer-change rate saturates at≈1.0 from round 1 onward), while faithfulness remains statistically stable from
zero to three rounds (≈0.89→0.91 at round 1, a gain of≈0.02 within the confidence interval band) and declines slightly at four to five
rounds (≈0.81–0.84). The effect of the loop is concentrated in the first corrective round, with additional rounds altering answers without
improving grounding. (b) Per-query wall-clock latency versus faithfulness for the same runs, colored by the configured round cap, shows
latency concentrated at 50–150 s with a tail to≈200 s and no systematic dependence on either the round cap or achieved faithfulness. These
results support the deployed two-round cap: rounds beyond the first increase latency and revision churn without measurable faithfulness
improvement.

APS-RAG 28
S7. CORPUS COLLECTIONS AND DOCUMENT SCHEMA
The retrieval corpus consists of 130,663 indexed records
obtained from eight operational sources. For sparse re-
trieval, records are indexed in Elasticsearch using BM25,
while dense retrieval is performed in Qdrant using the e5-
large-v2 model57with 1024-dimensional embeddings. Main
text Table I summarizes the collections and their respec-
tive sizes. Each indexed chunk, irrespective of its source,
is normalized into a unified corpus record comprising eight
fields:doc_id,content,title,author,collection,
system,creation_datetime, andfile_id. Thecontent
field contains the body text. Among all collections, only
thework_requestscollection splits records into multiple
chunks per parent record. For this collection, three addi-
tional chunk-provenance fields are included:parent_id,
chunk_index, andchunk_type.
Five of the eight sources exhibit a flat structure. In the case
of ICMS, SDDS, and OAG, each file contains a single JSON
document. For Teams and Work Requests, each file contains a
single list of records. These flat sources map directly onto the
unified corpus record format. In contrast, the BELY collection
employs a three-level nested schema, as illustrated in Fig.??.
Each beamline is represented by a file, resulting in a total of
24 files. Each file contains logbook entries, with 4,536 en-
tries in total. Each entry comprises 14 fields, including array-
valued fields such aspersonnel_involvedandkeywords.
Within each entry, there are chronological sub-entries, total-
ing 29,567, where each sub-entry consists ofentry_text
andattachment_urls. The schema includes two entry sub-
types that share the same structure and are distinguished by
the value of thelogbookfield. Routine operational entries
have emptystudies_planandstudies_summaryfields,
while storage-ring studies entries populate these fields. There
are 1,020 storage-ring studies entries among the 4,536 total
entries. During indexing, each BELY entry, together with its
sub-entries, is serialized into a single unified corpus record.
S8. BENCHMARK CONSTRUCTION DETAILS AND
DATASHEET
APS-Bench is constructed as a corpus-derived benchmark
rather than through manual annotation. Question–answer
pairs are generated from sampled corpus passages follow-
ing the InPars methodology9, where a large language model,
provided with a document and a small set of in-context ex-
emplars, generates a query that is answerable by the doc-
ument. While InPars employs such synthetic pairs for re-
triever training, we adapt this approach for evaluation pur-
poses, aligning more closely with automated self-data cura-
tion protocols for domain adaptation in retrieval-augmented
generation systems31. This adaptation introduces type con-
ditioning, structure-based difficulty assignment, retrievability
filtering, and a gold answer construction stage that is explic-
itly separated from the generation process. The pipeline com-
prises five stages, with each stage recording its outputs in the
released provenance record (Sec. S9).a. Stage 1: Seed discovery and passage sampling. For
each cell in the query type by collection grid, defined by eight
operational query types (factual, temporal, multi-hop, proce-
dural, causal, comparative, troubleshooting, parametric) and
eight indexed collections, at least five BM25 seed queries are
issued against the live corpus (2026-05-22). The top-ranked
results are pooled, and seed queries are constructed using the
operational vocabulary specific to each cell, including device
and subsystem names, fault terminology, and date- or shift-
specific phrasing. The pooled results are deduplicated at the
document level. Candidate source passages are then sampled
from each cell’s pool, ensuring that every generated question
is grounded in a specific, indexable passage rather than rely-
ing on the generator’s parametric knowledge.
b. Stage 2: Type-conditioned question–answer genera-
tion (InPars-style). Each sampled passage is provided to a
GPT-family generator (gpt54 via ARGO, temperature 0.7) us-
ing a type-conditioned prompt template. This template speci-
fies the target query type and its structural requirements. For
example, a multi-hop question must require at least two dis-
tinct evidence spans, a temporal question must include an ex-
plicit date or shift scope, and a troubleshooting question must
request a symptom to cause to resolution chain. The tem-
plate also supplies in-context exemplars of the target type and
instructs the model to return a single JSON object contain-
ing the question, a draft reference answer grounded only in
the supplied passage, and the supporting passage spans. Two
key modifications to the original InPars protocol are intro-
duced: first, full question–answer pairs are generated rather
than queries alone, as the subsequent nugget annotation (Stage
5) requires a grounded answer for decomposition; second,
generation is conditioned on the operational query-type taxon-
omy, ensuring that the benchmark’s type distribution is deter-
mined by design. Each per-type template is fixed and hashed,
and the hash is stored in every record it produces for inclusion
in the released package.
SYSTEM: generate evaluation questions for an accelerator-
operations
knowledge base
INPUT:
passage <- sampled corpus passage (doc_id, title, date,
text)
query_type <- one of the eight types + its structural
requirements
exemplars <- in-context (passage, question, answer)
examples
of the target type
OUTPUT (one JSON object):
question <- answerable from the passage; must satisfy the
type’s structural requirements
draft_answer <- grounded only in the supplied passage, with an
inline [doc_id] citation
support_spans <- character spans of the passage backing the
answer
c. Stage 3: Structure-first difficulty assignment.Diffi-
culty (easy, medium, hard) is determined based on the struc-
tural properties of the question, including the number of re-
quired evidence spans, the need for temporal scoping or cross-
collection linking, and the specificity of the requested quan-
tity. Assignment of difficulty is never based on observed sys-
tem performance. This structure-based approach ensures that
difficulty labels remain independent of the five systems under
evaluation, thereby preventing any potential feedback from

APS-RAG 29
evaluation outcomes into the benchmark design.
d. Stage 4: Over-generation and filtering.Candidate
question–answer pairs are over-generated for each cell and
subsequently filtered. The primary filtering criterion is re-
trievability: a candidate is retained only if its source passage
can be recovered from the full index by an independent re-
triever. This step eliminates questions for which the support-
ing evidence is not accessible at retrieval time. Remaining
candidates are further deduplicated to remove near-duplicate
questions within each cell and are screened for type confor-
mance and answerability. The final pool is balanced across
the type-by-collection grid to yield the released set of n=50. It
is important to note that retrieveability does not guarantee the
source is present in the final LLM synthesis.
e. Stage 5: Nugget annotation. The generator’s draft
answer from Stage 2 is not used as the released gold an-
swer; it serves solely to specify the intent of the question.
The gold answer is reconstructed using an anti-circular, multi-
retriever protocol. Evidence is retrieved independently by
BM25, dense, and graph retrievers, and the answer is synthe-
sized based on the pooled independent support with grounded
inline [doc_id] citations. Each answer is decomposed into 3
to 10 atomic nuggets labeled as vital or okay, with at least
two vital nuggets per question, following the AutoNuggetizer
convention15. Vital nuggets are required in any acceptable an-
swer, while okay nuggets provide correct supporting detail.
Each nugget is associated with a verified passage_span and
a chain_role in cause, effect, resolution, none. Reconstruct-
ing the gold answer from independent retrieval ensures that
no evaluated system’s output, and no artifact of the generation
step other than the question text, is included in the released
gold or its nuggets.
f. Provenance.Every stage writes to the record: seed
queries and retrieved doc-ids (Stage 1), prompt-template hash
and generation seed (Stage 2), the difficulty rationale (Stage
3), filter outcomes (Stage 4), and the gold-discovery trace
(regold_provenance, Stage 5), together with a package-
level SHA-256 (Sec. S9).
g. Known BiasesTwo biases are inherent to InPars-style
construction and bound the claims made on this benchmark,
as stated in the main-text Limitations. First, corpus-derived
questions echo the vocabulary of their source passages, which
makes them intrinsically easier to retrieve than verbatim oper-
ator questions; absolute scores should therefore be read as an
upper bound on staff-question performance. Second, the ques-
tions, gold answers, and nugget labels are all LLM-produced,
so the benchmark inherits the generator’s stylistic and topical
preferences within each cell even though the type mix is con-
trolled. The expert-validation protocol is the gate on both: at
least two APS experts adjudicate a stratified calibration sam-
ple for question validity and answerability, vital-nugget agree-
ment against the auto-extracted nuggets, and inter-annotator
agreement (Cohen’sκ, Krippendorff’sα, percent agreement),
using the risk-first audit sheets of Sec.??S9. BENCHMARK DATASET: SCHEMA, PROVENANCE,
AND FULL LISTING
We release the benchmark in full. Each record is one JSON
object carrying thequestion, itsquery_type,difficulty,
andsource_collection; thegold_answerwith inline
[doc_id]citations; the supportingevidence_passages;
and the atomicnuggets, each labeledvitalorokay. Ques-
tions were generated by a GPT-family model (gpt54via
ARGO) from the live APS corpus using the above-defined In-
Pars methodology.

APS-RAG 30
A. APS-Bench: full question listing (n=50, corpus-derived)
APS-BENCH-001parametric·ICMS·easy
Q.For the PAR THz beamline measurement setup described in
DIAG-TN-2024-018, what bunch-charge setpoint should I request if I want
to run at the highest tested charge from the November 13 and December 9,
2024 studies?
Gold.The highest tested bunch charge in the November 13 and December 9,
2024 PAR THz beamline studies was 16 nC, so the bunch-charge setpoint to
request is 16 nC [APS_2294687__chunk_0].
Nuggets.[V]The highest tested bunch charge in the November 13 and
December 9, 2024 PAR THz beamline studies was 16 nC.[V]The
bunch-charge setpoint to request is 16 nC. [o] The answer applies
specifically to the PAR THz beamline measurement setup described in
DIAG-TN-2024-018.
Evidence.APS_2294687__chunk_0(primary)
APS-BENCH-002procedural·ICMS·easy
Q.How would you set up the PAR to serve as the injector for the Linac
Extension Area instead of the usual photocathode or RF electron gun source,
including the basic sequence of using the PAR as a damping or accumulator
ring before extracting and transporting the beam to LEA?
Gold.To set up the PAR as the injector for the Linac Extension Area instead
of the usual photocathode or RF electron gun source, one would use the
Particle Accumulator Ring (PAR) as a damping or accumulator ring, then
transport the extracted beam to LEA [APS_2189598__chunk_0]. The
passage also states that the typical LEA source is the photocathode electron
gun and that it is also possible to use electron beams from radiofrequency
electron guns [APS_2189598__chunk_0]. The passages do not provide
further setup details beyond this basic sequence [APS_2189598__chunk_0].
Nuggets.[V]The Particle Accumulator Ring (PAR) can be used as the
injector source for the Linac Extension Area (LEA).[V]In this setup, the
PAR is used as a damping ring or accumulator ring before beam delivery to
LEA.[V]After use in the PAR, the beam is extracted from the PAR and
transported to LEA. [o] The usual source for LEA is the photocathode
electron gun. [o] Electron beams from radiofrequency electron guns can
also be used for LEA. [o] The provided passage does not give additional
setup details beyond the basic sequence of using PAR, extracting the beam,
and transporting it to LEA.
Evidence.APS_2189598__chunk_0(primary)
APS-BENCH-003factual·OAG·easy
Q.In the APS accelerator control system, what diagnostic GUI would an
operator use to verify that OAG data loggers are actively writing files, that
the RunControl PVs are running, and to identify any workstations or
processes needing attention when data archiving problems are suspected?
Gold.The operator would use the checkDataLoggers diagnostic GUI, which
checks whether data loggers are actively writing files, verifies RunControl
PVs are running, and helps identify which workstations or processes need
attention when data archiving problems are suspected
[checkDataLoggers__chunk_0].
Nuggets.[V]The diagnostic GUI to use is checkDataLoggers.
[V]checkDataLoggers verifies whether OAG data loggers are actively
writing files.[V]checkDataLoggers verifies that the RunControl PVs are
running. [o] checkDataLoggers helps identify which workstations need
attention when data archiving problems are suspected.
[o] checkDataLoggers helps identify which processes need attention when
data archiving problems are suspected.
Evidence.checkDataLoggers__chunk_0(primary)
APS-BENCH-004factual·ICMS·easy
Q.What is the principal function of the PAR at the APS?
Gold.The principal function of the PAR at the APS is to act as an
accumulator ring, damping multiple (< 30) injections from the linac into a
single bunch [APS_2170123__chunk_0].
Nuggets.[V]At the APS, the PAR’s principal function is to act as an
accumulator ring.[V]The PAR damps injections from the linac.
[o] The PAR handles multiple injections, fewer than 30, from the linac.
[V]The PAR combines the multiple linac injections into a single bunch.
Evidence.APS_2170123__chunk_0(primary)
APS-BENCH-005factual·ICMS·easyQ.According to DIAG-TN-2022-007, what ring is being considered as a
damping or accumulator ring to inject beam into the Linac Extension Area?
Gold.According to DIAG-TN-2022-007, the ring being considered as a
damping or accumulator ring to inject beam into the Linac Extension Area is
the Particle Accumulator Ring (PAR) [APS_2189598__chunk_0].
Nuggets.[V]DIAG-TN-2022-007 considers the Particle Accumulator Ring
(PAR) as the ring for injecting beam into the Linac Extension Area.
[V]The ring is being considered in the role of a damping ring.[V]The
ring is being considered in the role of an accumulator ring. [o] The
acronym PAR stands for Particle Accumulator Ring.
Evidence.APS_2189598__chunk_0(primary)
APS-BENCH-006factual·BELY·easy
Q.For 2D scans on 9-BM using EPICS scan control, which after-scan PVs
need to be set to 0 with "NoWait" to prevent the scan from hanging at the
end of random lines?
Gold.For 2D scans on 9-BM using EPICS scan control, the after-scan fields
must set 9bmc:scan1.AWAIT to 0 and 9bmc:scan2.AWAIT to 0 with
"NoWait"; otherwise the scan can hang at the end of random lines
[10635__chunk_0].
Nuggets.[V]For 2D scans on 9-BM using EPICS scan control, the
after-scan field 9bmc:scan1.AWAIT must be set to 0.[V]For 2D scans on
9-BM using EPICS scan control, the after-scan field 9bmc:scan2.AWAIT
must be set to 0.[V]The after-scan settings for these PVs should use
"NoWait". [o] If these after-scan PVs are not set this way, the scan can
hang at the end of random lines.
Evidence.10635__chunk_0(primary)
APS-BENCH-007factual·SDDS·easy
Q.When preparing an orbit response matrix for use in feedback, which
SDDS Toolkit command would you use to transpose the tabular data so the
output has one column for each input row?
Gold.The command is "sddstranspose()" because it "transposes the tabular
data in the input file, so that the output file contains one column for each row
in the input" and the example usage is to "transpose an orbit response matrix
as part of preparing to use it for feedback" [SDDS_OAG_User’s Guide for
SDDS Toolkit Version 5.0_chunk_19__chunk_1].
Nuggets.[V]The SDDS Toolkit command to use is sddstranspose().
[V]sddstranspose() transposes the tabular data in the input file.[V]After
transposition, the output file contains one column for each row in the input.
[o] A documented use of sddstranspose() is transposing an orbit response
matrix when preparing it for feedback.
Evidence.SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_19__chunk_1(primary)
APS-BENCH-008procedural·SDDS·easy
Q.How would you use the SDDS toolkit to prepare an orbit response matrix
and then generate a correction vector for step-by-step feedback, including
which commands you would run to transpose the response matrix and
multiply it by the error vector?
Gold.Using the SDDS toolkit, you would prepare the orbit response matrix
for feedback by running sddstranspose on the orbit response matrix file,
since sddstranspose transposes tabular data and is used to transpose an orbit
response matrix as part of preparing it for feedback [SDDS_OAG_User’s
Guide for SDDS Toolkit Version 5.0_chunk_19__chunk_1]. Then you
would generate the correction vector for step-by-step feedback by running
sddsmatrixmult to multiply the error vector by the correction matrix, since
sddsmatrixmult multiplies the tabular data in two input files and is used to
multiply a vector of errors with a correction matrix to obtain a vector of
corrections to apply in a step-by-step feedback system [SDDS_OAG_User’s
Guide for SDDS Toolkit Version 5.0_chunk_19__chunk_1]. The passages
do not provide the exact command-line syntax, only the toolkit commands to
use [SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_19__chunk_1].
Nuggets.[V]Use the SDDS command sddstranspose to transpose the orbit
response matrix file when preparing it for feedback. [o] Transposing the
orbit response matrix is part of preparing the matrix for feedback.[V]Use
the SDDS command sddsmatrixmult to multiply the error vector by the
correction matrix.[V]Multiplying the error vector by the correction
matrix produces the correction vector to apply in step-by-step feedback.
[o] The cited SDDS passages identify which toolkit commands to use but do
not provide exact command-line syntax.

APS-RAG 31
Evidence.SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_19__chunk_1(primary)
APS-BENCH-009procedural·SDDS·easy
Q.How do I use sddsplot to display the Twiss functions for APS lattices so
that each function is on its own plotting page while each lattice data page
appears with a different line type?
Gold.Use: sddsplot -columnNames=s,’(beta?,etax)’ APS.twi
-graphic=line,vary -split=page -groupby=nameIndex -separate=nameIndex
[SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_126__chunk_0]
Nuggets.[V]Use the command ‘sddsplot -columnNames=s,’(beta?,etax)’
APS.twi -graphic=line,vary -split=page -groupby=nameIndex
-separate=nameIndex‘.[V]Specify ‘-columnNames=s,’(beta?,etax)’‘ to
plot the Twiss-function columns against ‘s‘.[V]Use ‘-split=page‘ so each
function is placed on its own plotting page.[V]Use ‘-graphic=line,vary‘
so different lattice data pages are distinguished by different line types.
[o] Use ‘-groupby=nameIndex‘. [o] Use ‘-separate=nameIndex‘.
[o] The input lattice file in the example is ‘APS.twi‘.
Evidence.SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_126__chunk_0(primary)
APS-BENCH-010factual·SDDS·easy
Q.Using sddsplot, what command would you use to plot the horizontal beta
function versus s for the APS design lattice file APS0.twi?
Gold.The command is "sddsplot -columnNames=s,betax APS0.twi"
[SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_126__chunk_0].
Nuggets.[V]The required command uses the program ‘sddsplot‘.[V]The
command specifies the columns ‘s‘ and ‘betax‘ with
‘-columnNames=s,betax‘.[V]The input file in the command is
‘APS0.twi‘.[V]The full command is ‘sddsplot -columnNames=s,betax
APS0.twi‘. [o] In this command, ‘s‘ is the horizontal axis variable.
[o] In this command, ‘betax‘ is the horizontal beta function being plotted.
Evidence.SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_126__chunk_0(primary)
APS-BENCH-011procedural·ICMS·easy
Q.How is the PAR terahertz diagnostic intended to be used operationally to
monitor longitudinal bunch instabilities during accumulation of multiple
linac injections into a single bunch?
Gold.The PAR terahertz diagnostic is intended to be used to observe
longitudinal electron bunch instabilities [APS_2170123__chunk_0] during
operation of the Particle Accumulator Ring, whose principal function is
damping multiple (< 30) injections from the linac into a single bunch
[APS_2170123__chunk_0]. The passage does not fully specify the
operational procedure for monitoring those instabilities during accumulation.
Nuggets.[V]The PAR terahertz diagnostic is intended to observe
longitudinal electron bunch instabilities.[V]The diagnostic is intended
for use during operation of the Particle Accumulator Ring (PAR).[V]The
PAR accumulates multiple injections from the linac into a single bunch.
[o] The number of linac injections accumulated is fewer than 30. [o] A
principal function of the PAR is to damp the accumulated injections.
[V]The passage does not specify a detailed operational procedure for how
the terahertz diagnostic monitors instabilities during accumulation.
Evidence.APS_2170123__chunk_0(primary)
APS-BENCH-012procedural·BELY·easy
Q.If you’re trying to refine chromaticity during SR studies and the
chromaticity knobs are not behaving as expected, what measurement setup
should you use to get more reliable chromaticity results, and what
tune-adjustment precaution should you take given the quadrupole current
limits and the observed conditioning behavior?
Gold.Use a chromaticity measurement setup with 15 points, since that gave
much more reliable chromaticity results when the chromaticity knobs were
not behaving as expected [3640__chunk_0]. For tune adjustment, take care
because some quadrupoles are already near the swap-out safety/current limit,
which caused issues in tune adjustment in one case, and also because
conditioning the quadrupoles after adjusting the tune knobs once per plane
did not restore the tunes; the passages do not specify a complete corrective
procedure beyond this precaution [3640__chunk_0].
Nuggets.[V]Use a chromaticity measurement setup with 15 points to obtain
more reliable chromaticity results.[V]The 15-point chromaticitymeasurement setup was specifically recommended when the chromaticity
knobs were not behaving as expected.[V]When adjusting tune, be
cautious because some quadrupoles are already near the swap-out
safety/current limit. [o] Quadrupoles being near the safety/current limit
caused tune-adjustment issues in at least one case.[V]Conditioning the
quadrupoles after adjusting the tune knobs once per plane did not restore the
tunes. [o] The source passage does not provide a complete corrective
procedure beyond warning to take this precaution.
Evidence.3640__chunk_0(primary)
APS-BENCH-013procedural·SDDS·easy
Q.How do I compute an inverse response matrix for trajectory correction
from a previously measured response matrix, and what machine element
subset can I select when doing it?
Gold.You compute an inverse response matrix by using the “Compute
Inverse Response Matrix” function to compute a trajectory correction matrix
from a previously measured response matrix [SDDS_OAG_Control System
High-Level Applications_chunk_18__chunk_0]. When doing this, you can
select a subset of correctors and BPMs from that previously measured
response matrix [SDDS_OAG_Control System High-Level
Applications_chunk_18__chunk_0].
Nuggets.[V]An inverse response matrix is computed by using the
"Compute Inverse Response Matrix" function.[V]The inverse response
matrix is computed from a previously measured response matrix. [o] The
computed inverse response matrix is a trajectory correction matrix.
[V]When computing the inverse response matrix, you can select a subset of
correctors from the previously measured response matrix.[V]When
computing the inverse response matrix, you can select a subset of BPMs
from the previously measured response matrix.
Evidence.SDDS_OAG_Control System High-Level
Applications_chunk_18__chunk_0(primary)
APS-BENCH-014factual·SDDS·easy
Q.In the OAG control system high-level applications, what does the
“Compute Inverse Response Matrix” function let you calculate from a
previously measured response matrix?
Gold.It lets you calculate a trajectory correction matrix from a subset of
correctors and BPMs in a previously measured response matrix
[SDDS_OAG_Control System High-Level
Applications_chunk_18__chunk_0].
Nuggets.[V]The “Compute Inverse Response Matrix” function calculates a
trajectory correction matrix.[V]The calculation uses a previously
measured response matrix as input.[V]The trajectory correction matrix
can be calculated from a subset of correctors.[V]The trajectory
correction matrix can be calculated from a subset of BPMs.
Evidence.SDDS_OAG_Control System High-Level
Applications_chunk_18__chunk_0(primary)
APS-BENCH-015procedural·BELY·easy
Q.If we can’t inject into the new sextupole settings on first beam, what
bootstrap procedure should we follow to recover stored beam and transition
back to the new sextupoles while maintaining orbit control?
Gold.If we can’t inject into the new sextupole settings on first beam, the
bootstrap procedure is: restore the old sextupoles by merely setting them
without standardizing [6944__chunk_1]; fill 10 mA beam and run orbit
correction [6944__chunk_1]; ramp to the new sextupoles while the beam is
stored and orbit correction is running [6944__chunk_1]; save correctors
[6944__chunk_1]; measure and correct tunes [6944__chunk_1]; and save
quadrupoles [6944__chunk_1]. The passage also indicates orbit control must
actually be active during the sextupole ramp, since a partial beam loss
occurred when x-plane orbit correction had been left in a suspend state
[6944__chunk_1].
Nuggets.[V]If injection into the new sextupole settings fails on first beam,
first restore the old sextupole settings by setting them without standardizing.
[V]After restoring the old sextupoles, fill 10 mA of beam.[V]Run orbit
correction on the stored 10 mA beam.[V]Ramp from the old sextupoles
to the new sextupoles while the beam remains stored.[V]Orbit correction
must be actively running during the sextupole ramp. [o] After reaching the
new sextupole settings, save the corrector settings. [o] Measure and
correct the tunes after the sextupole transition. [o] Save the quadrupole
settings after tune correction. [o] A partial beam loss occurred when
x-plane orbit correction was left suspended during the sextupole ramp.

APS-RAG 32
Evidence.6944__chunk_1(primary)
APS-BENCH-016parametric·ICMS·medium
Q.For the PAR integrating current transformer absolute calibration, what
bunch-current range in mA corresponds to the calibrated bunch-charge range
at the 9.77 MHz repetition rate?
Gold.The calibrated bunch-charge range of 1-21.9 nC corresponds to a
bunch-current range of 9.8-214 mA at the 9.77 MHz repetition rate
[APS_2032345__chunk_0].
Nuggets.[V]The PAR integrating current transformer absolute calibration
uses a calibrated bunch-charge range of 1 nC to 21.9 nC.[V]The
repetition rate for this conversion is 9.77 MHz.[V]A bunch charge of 1
nC corresponds to a bunch current of 9.8 mA at 9.77 MHz.[V]A bunch
charge of 21.9 nC corresponds to a bunch current of 214 mA at 9.77 MHz.
[V]The calibrated bunch-current range is 9.8 mA to 214 mA.
Evidence.APS_2032345__chunk_0(primary)
APS-BENCH-017procedural·OAG·medium
Q.How do I start a cathode bake-in on RG1 or RG2, including which
parameters I need to set before hitting Ramp Cathode and how to stop and
reset the procedure if I need to abort?
Gold.To start a cathode bake-in, first select the RF Gun (RG1 or RG2), then
set the maximum heater power in Watts, set the time to hold at maximum
power, configure the current step size in Amps, and configure the pause time
between steps [BakeInCathode__chunk_1]. After these parameters are set,
use Ramp Cathode to start the bake-in procedure
[BakeInCathode__chunk_1]. If you need to abort, use Abort to stop the
bake-in procedure and reset [BakeInCathode__chunk_1].
Nuggets.[V]Select the RF Gun to bake in, choosing either RG1 or RG2,
before starting the procedure.[V]Set the maximum heater power in Watts
before starting the cathode bake-in.[V]Set the time to hold at maximum
power before starting the cathode bake-in.[V]Configure the current step
size in Amps before starting the cathode bake-in.[V]Configure the pause
time between steps before starting the cathode bake-in.[V]Start the
cathode bake-in by using Ramp Cathode after the required parameters are
set.[V]Use Abort to stop the cathode bake-in procedure if you need to
abort. [o] Using Abort also resets the cathode bake-in procedure.
Evidence.BakeInCathode__chunk_1(primary)
APS-BENCH-018troubleshooting·WORK_REQUESTS·medium
Q.Injection is blocked because B:IS is showing negative current readbacks,
and the only recovery so far was a power cycle. Given the reported MTime
power supply failure, what component should we suspect and what
corrective action should we take to restore reliable injection?
Gold.We should suspect the MTime power supply because a reported
MTime power supply failure requires that the power supply be swapped
[50362__chunk_0]. The corrective action indicated to restore reliable
injection is to swap the power supply [50362__chunk_0]. The passages do
not explicitly state that the MTime power supply failure is the confirmed
cause of B:IS negative current readbacks; they only say B:IS showed
negative current readbacks preventing injection and was recovered via power
cycle [50694__chunk_0].
Nuggets.[V]The component to suspect is the MTime power supply.
[V]The corrective action is to swap the MTime power supply.
[V]Swapping the power supply is indicated as the way to restore reliable
injection. [o] The evidence does not explicitly confirm that the MTime
power supply failure caused the B:IS negative current readbacks. [o] B:IS
negative current readbacks were preventing injection. [o] (resolution) A
power cycle had previously recovered the issue.
Evidence.50362__chunk_0(primary),50694__chunk_0(supporting)
APS-BENCH-019temporal·TEAMS·medium
Q.On the 10/8/24 shift after the beam dump and before attempting a reset,
did Operations have any way to verify whether the alignment issue still
existed without first filling and dumping beam again?
Gold.Based on the 10/8/24 passage, Operations did not have a stated way to
verify whether the alignment issue still existed after the beam dump and
before attempting a reset without filling and dumping beam again; Elaine
Chandler explicitly said, "I don’t know how to check alignment without
filling and dumping beam," and asked Weixing Cheng whether there was
another way [teams_Borland, Michael David_e79c556151b8__chunk_0].
The passages do not provide any answer from Weixing Cheng or anyone else
confirming an alternative method, so the passages do not fully answerwhether such a way existed [teams_Borland, Michael
David_e79c556151b8__chunk_0].
Nuggets.[V]The 10/8/24 passage does not state any method Operations
could use to verify whether the alignment issue still existed after the beam
dump and before attempting a reset without filling and dumping beam again.
[V]Elaine Chandler said, "I don’t know how to check alignment without
filling and dumping beam." [o] Elaine Chandler asked Weixing Cheng
whether there was another way to check alignment.[V]The provided
passages do not include any answer from Weixing Cheng or anyone else
confirming an alternative verification method. [o] Because no alternative
method is given in the passages, the evidence does not fully establish that
such a way existed.
Evidence.teams_Borland, Michael David_e79c556151b8__chunk_0
(primary),teams_Borland, Michael
David_d9889662890a__chunk_0(supporting)
APS-BENCH-020temporal·TEAMS·medium
Q.During the 10/8/24 beam-dump/reset troubleshooting shift, before trying
a reset on the suspected alignment issue, who did Elaine suggest checking
with for an alternate way to verify alignment without filling and dumping
beam, and on the later 1/26/25 BESOCM support shift, who was Andrew’s
fallback contact if Rob Keane and a Diagnostics secondary contact could not
be reached?
Gold.Before trying a reset on the suspected alignment issue during the
10/8/24 shift, Elaine suggested checking with Weixing Cheng for another
way to verify alignment without filling and dumping beam [teams_Borland,
Michael David_e79c556151b8__chunk_0]. On the 1/26/25 BESOCM
support shift, Andrew’s fallback contact if Rob Keane and a Diagnostics
secondary contact could not be reached was Guobao [teams_Borland,
Michael David_d9889662890a__chunk_0].
Nuggets.[V]During the 10/8/24 beam-dump/reset troubleshooting shift,
Elaine suggested checking with Weixing Cheng before trying a reset on the
suspected alignment issue. [o] Elaine suggested Weixing Cheng as
someone who might offer another way to verify alignment without filling
and dumping beam.[V]During the 1/26/25 BESOCM support shift,
Andrew’s fallback contact was Guobao if Rob Keane could not be reached.
[V]During the 1/26/25 BESOCM support shift, Andrew’s fallback contact
was Guobao if a Diagnostics secondary contact also could not be reached.
Evidence.teams_Borland, Michael David_e79c556151b8__chunk_0
(primary),teams_Borland, Michael
David_d9889662890a__chunk_0(primary)
APS-BENCH-021comparative·BELY·medium
Q.At 9-BM, how did the relative ion-chamber noise behavior change
between the Ni K-edge tests and the later 24 keV dummy scan after all
SR570s were manually set to low-noise/no-filter—specifically, which signal
chain or chamber was identified as noisier in the Ni-edge study, and what
gain/gas configuration was then used for the He-filled I0 versus the N2-filled
chambers in the 24 keV comparison?
Gold.In the Ni K-edge tests, the noisier behavior was identified in the scaler
3 signal chain rather than the ion chamber itself, and separately the I0 signal
was noted to have higher noise than IRef, with I0 being He-filled and IRef
the ADC ion chamber [7413__chunk_0]. In the later 24 keV dummy scan
after all SR570s were manually set to low-noise/no-filter, the He-filled I0-1
used 1 pA/V , while the N2-filled chambers were adjusted to 50 pA/V for all
N2-filled ion chambers [7582__chunk_0].
Nuggets.[V]In the Ni K-edge tests, the noisier behavior was attributed to
the scaler 3 signal chain rather than to the ion chamber itself. [o] In the
Ni-edge study, the I0 signal was observed to be noisier than IRef. [o] In
that Ni-edge comparison, I0 was the He-filled chamber. [o] In the later 24
keV dummy scan, all SR570s had been manually set to low-noise mode with
no filter.[V]In the 24 keV comparison, the He-filled I0-1 was set to a
gain of 1 pA/V .[V]In the 24 keV comparison, the N2-filled ion chambers
were set to a gain of 50 pA/V .
Evidence.7413__chunk_0(primary),7582__chunk_0(primary)
APS-BENCH-022comparative·WORK_REQUESTS·medium
Q.Between the Booster dipole slave power supply SCR-failure trip and the
ioclet1bpm reboot to add event timestamps for injection-cycle data
collection, which issue is a hardware fault versus a controls/data-acquisition
change, and how would their expected operational impact differ during
Booster injection operations?

APS-RAG 33
Gold.The Booster dipole slave power supply issue is a hardware fault: the
power supply needed repair because it "tripped on SCR failure"
[45927__chunk_0]. The ioclet1bpm issue is a controls/data-acquisition
change: it required a reboot "to add event timestamps for injection cycle data
collection" [44960__chunk_0]. Their exact operational impact during
Booster injection operations is not fully stated in the passages, but the first
implies a failed power-supply condition requiring repair [45927__chunk_0],
whereas the second implies a data-collection/timestamping change for
injection-cycle measurements [44960__chunk_0].
Nuggets.[V]The Booster dipole slave power supply issue was a hardware
fault.[V]The hardware fault was that the power supply tripped on SCR
failure. [o] The Booster dipole slave power supply required repair.
[V]The ioclet1bpm issue was a controls/data-acquisition change.[V]The
ioclet1bpm reboot was done to add event timestamps for injection-cycle data
collection. [o] The expected operational impact of the power-supply issue
was a failed hardware condition affecting operation until repaired. [o] The
expected operational impact of the ioclet1bpm change was on
injection-cycle measurement data collection rather than a hardware failure.
Evidence.45927__chunk_0(primary),44960__chunk_0(primary)
APS-BENCH-023troubleshooting·WORK_REQUESTS·medium
Q.B:IS is showing negative current readbacks and injection is
blocked—should the first recovery step be to power cycle the supply, and if
that doesn’t hold do we need to swap the MTime power supply?
Gold.Yes. The passages indicate that when B:IS showed negative current
readbacks preventing injection, it was recovered via power cycle
[50694__chunk_0]. Separately, they state that for an MTime power supply
failure, the power supply needs to be swapped [50362__chunk_0]. However,
the passages do not explicitly say that if the power-cycle recovery does not
hold for B:IS, then the next step is to swap the MTime power supply
[50694__chunk_0][50362__chunk_0].
Nuggets.[V](resolution) When B:IS shows negative current readbacks and
injection is blocked, a power cycle recovered it.[V]The first recovery
step for the B:IS negative-current-readback issue is to power cycle the
supply.[V]For an MTime power supply failure, the documented action is
to swap the power supply.[V]The passages do not explicitly state that if
the B:IS power-cycle recovery does not hold, the next step is to swap the
MTime power supply. [o] The evidence for power-cycling B:IS comes
from source 50694__chunk_0. [o] The evidence for swapping the MTime
power supply comes from source 50362__chunk_0.
Evidence.50694__chunk_0(primary),50362__chunk_0(supporting)
APS-BENCH-024comparative·ICMS·medium
Q.Compared with normal operation without VESD, what happened to the
PAR THz beamline CSR signal at 1 nC when the LTP:FL1 flag was inserted?
Gold.At 1 nC, inserting the LTP:FL1 flag attenuated the beam so much that
the PAR THz beamline CSR signal became indistinguishable from the
background, and no CSR signal was detected, compared with normal
operation without VESD [APS_2294687__chunk_6].
Nuggets.[V]The condition discussed is at a bunch charge of 1 nC.
[V]With the LTP:FL1 flag inserted, the beam was strongly attenuated.
[V]Under that condition, the PAR THz beamline CSR signal became
indistinguishable from background.[V]No PAR THz beamline CSR
signal was detected when the LTP:FL1 flag was inserted. [o] This
outcome is in contrast to normal operation without VESD.
Evidence.APS_2294687__chunk_6(primary),APS_2294687__chunk_0
(supporting)
APS-BENCH-025temporal·WORK_REQUESTS·medium
Q.During the 2024-10-18 shift, was there any active PAR machine-repair
work like the 2020 harmonic-system troubleshooting, or was the only
relevant work request the Linac RG2 kicker interlock sum fault
investigation?
Gold.During the 2024-10-18 shift, the passages show a completed work
request to investigate the Linac RG2 kicker interlock sum fault
[50660__chunk_0]. The only PAR machine-repair example provided is from
2020-09-28 to troubleshoot and repair the PAR harmonic system
[43541__chunk_0]. The passages do not show any active PAR
machine-repair work on 2024-10-18, so based only on these passages, the
only relevant work request shown for 2024-10-18 was the Linac RG2 kicker
interlock sum fault investigation [50660__chunk_0].
Nuggets.[V]On 2024-10-18, the passages show a completed work request
to investigate the Linac RG2 kicker interlock sum fault.[V]The passagesdo not show any active PAR machine-repair work on 2024-10-18. [o] The
only PAR machine-repair example provided in the passages is from
2020-09-28. [o] The 2020-09-28 PAR machine-repair example was to
troubleshoot and repair the PAR harmonic system.[V]Based only on
these passages, the only relevant work request shown for 2024-10-18 was the
Linac RG2 kicker interlock sum fault investigation.
Evidence.50660__chunk_0(primary),43541__chunk_0(supporting)
APS-BENCH-026temporal·BELY·medium
Q.During the 2026-02-05 to 2026-02-06 pumpdown of the 9-ID Grand
Tube, what was the GT vacuum reading just before GV11 was opened, and if
we wanted to use the 9-BM Ge8 xspress3 setup for a synchronized scan in
that same run window, which [main] section trigger/sync PVs would have
failed the caget existence check?
Gold.Just before GV11 was opened during the 2026-02-05 to 2026-02-06
9-ID Grand Tube pumpdown, the GT vacuum reading was 3.6e04 at
2026-02-06_14:34:06 [10119__chunk_0]. For the 9-BM Ge8 xspress3
setup, the [main] section trigger/sync PVs that would have failed the caget
existence check were the d#ScanStart2.OUTN scan-start PVs and the
scalerd#sync.INBN scan-sync PVs [10086__chunk_0].
Nuggets.[V]Just before GV11 was opened during the 2026-02-05 to
2026-02-06 9-ID Grand Tube pumpdown, the GT vacuum reading was
3.6e04. [o] The timestamp for the GT vacuum reading just before GV11
was opened was 2026-02-06_14:34:06.[V]For the 9-BM Ge8 xspress3
setup, the [main] section d#ScanStart2.OUTN scan-start PVs would have
failed the caget existence check.[V]For the 9-BM Ge8 xspress3 setup,
the [main] section scalerd#sync.INBN scan-sync PVs would have failed the
caget existence check.
Evidence.10119__chunk_0(primary),10086__chunk_0(primary)
APS-BENCH-027comparative·WORK_REQUESTS·medium
Q.Which issue is tied to adding event timestamps for injection cycle data
collection—the ioclet1bpm reboot request or the MTime power supply
failure—and what corrective action is requested for each?
Gold.Adding event timestamps for injection cycle data collection is tied to
the ioclet1bpm reboot request; the requested corrective action is to reboot
ioclet1bpm [44960__chunk_0]. The MTime issue is a power supply failure;
the requested corrective action is to swap the power supply
[50362__chunk_0].
Nuggets.[V]The issue tied to adding event timestamps for injection cycle
data collection is the ioclet1bpm reboot request.[V]The requested
corrective action for the injection cycle event timestamp issue is to reboot
ioclet1bpm.[V]The MTime issue is a power supply failure.[V]The
requested corrective action for the MTime power supply failure is to swap
the power supply.
Evidence.44960__chunk_0(primary),50362__chunk_0(primary)
APS-BENCH-028troubleshooting·ICMS·medium
Q.SPX0 keeps dropping RF and staying latched RF-OFF after a fast
interlock event, with the equipment protection display showing an
ARC-related trip and no automatic recovery. Based on the cryomodule and
master fast interlock scheme, what fault path is most likely causing the
shutdown, and what operator action is required to restore klystron drive?
Gold.The most likely fault path is an ARC-related cryomodule fast interlock
fault—specifically an arc detector on the forward power coupler or LOM
damper waveguide RF windows—being detected by the fast cryomodule
interlock, which sends a latching trip command via fiber optic link to the
master fast RF interlock in the amplifier [APSU_1430567__chunk_1037].
That master fast interlock then interrupts RF drive to the klystron and stays
latched in the RF-off condition [APSU_1430567__chunk_1037]. To restore
klystron drive, the operator must issue a manual reset command, because
there is no automatic recovery described in the passage
[APSU_1430567__chunk_1037].
Nuggets.[V](cause) The shutdown is most likely caused by an ARC-related
cryomodule fast interlock fault. [o] (cause) A likely source of the ARC
fault is an arc detector on the forward power coupler RF window. [o]
(cause) A likely source of the ARC fault is an arc detector on the LOM
damper waveguide RF window.[V](cause) The cryomodule fast interlock
sends a latching trip command to the master fast RF interlock via a fiber
optic link.[V]The master fast RF interlock interrupts RF drive to the
klystron.[V](cause) The master fast RF interlock remains latched in the
RF-OFF condition after this trip.[V](resolution) Restoring klystron drive

APS-RAG 34
requires the operator to issue a manual reset command. [o] (cause) There
is no automatic recovery for this fault path described in the passage.
Evidence.APSU_1430567__chunk_1037(primary),
APSU_1430567__chunk_1034(supporting)
APS-BENCH-029comparative·BELY·medium
Q.Compared with the 20-BM AuPd nanoparticles setup where the HR
mirror was backed off to 4 mrad, what HR mirror angle was used for the
9-BM 10 keV ion-chamber test on the Pt stripe, and how do the two mirror
settings differ operationally?
Gold.For the 9-BM 10 keV ion-chamber test on the Pt stripe, the HR mirror
was inserted to 6.0 mrad [8773__chunk_0]. Compared with the 20-BM
AuPd nanoparticles setup, where the HR mirror was at 5 mrad and then
backed off to 4 mrad [9300__chunk_0], the 9-BM test used a higher angle
by 2.0 mrad relative to the final 4 mrad setting
[8773__chunk_0][9300__chunk_0]. Operationally, the 20-BM setting
involved backing the mirror off from 5 to 4 mrad [9300__chunk_0], while
the 9-BM test involved inserting the HR mirror onto the Pt stripe at 6.0 mrad
during the 10 keV test after removing it, setting energy to 10 keV , and
detuning the mono by 25% [8773__chunk_0].
Nuggets.[V]For the 9-BM 10 keV ion-chamber test on the Pt stripe, the HR
mirror angle was 6.0 mrad.[V]In the 20-BM AuPd nanoparticles setup,
the HR mirror’s final backed-off setting was 4 mrad.[V]The 9-BM 6.0
mrad setting is 2.0 mrad higher than the 20-BM final 4 mrad setting.
[o] At 20-BM, the HR mirror was first at 5 mrad before being backed off to 4
mrad.[V]Operationally, the 20-BM configuration involved backing the
HR mirror off from 5 mrad to 4 mrad.[V]Operationally, the 9-BM test
involved inserting the HR mirror onto the Pt stripe at 6.0 mrad. [o] During
the 9-BM procedure, the mirror insertion at 6.0 mrad occurred during a 10
keV test after the mirror had been removed, with the monochromator
detuned by 25%.
Evidence.8773__chunk_0(primary),9300__chunk_0(supporting)
APS-BENCH-030troubleshooting·ICMS·medium
Q.The SPX0 cavity RF is latched off and won’t come back after a
coupler/window arc trip. What should I check in the fast cryomodule
interlock path, and what operator action is required to restore RF?
Gold.You should check the fast cryomodule interlock path for a latching
fault from the arc detectors on the forward power coupler and RF windows,
and more generally for any fast cryomodule interlock fault being relayed by
the fiber-optic link to the master fast RF interlock chassis in the amplifier
[APSU_1430567__chunk_1037]. The trip causes the master fast interlock to
interrupt RF drive to the klystron and latch RF off until a manual reset
command is issued by the system operator [APSU_1430567__chunk_1037].
The passages do not provide any more specific SPX0-only recovery step
beyond that manual operator reset [APSU_1430567__chunk_1037].
Nuggets.[V](cause) Check the fast cryomodule interlock path for a latching
fault from the arc detectors on the forward power coupler.[V]
(cause) Check the fast cryomodule interlock path for a latching fault from
the arc detectors on the RF windows.[V](cause) Check for any fast
cryomodule interlock fault being relayed by the fiber-optic link to the master
fast RF interlock chassis in the amplifier. [o] (cause) The trip causes the
master fast interlock to interrupt RF drive to the klystron.[V]
(resolution) The master fast interlock latches RF off until a manual reset
command is issued.[V](resolution) The required recovery action is a
manual reset by the system operator. [o] (resolution) No more specific
SPX0-only recovery step is provided beyond the manual operator reset.
Evidence.APSU_1430567__chunk_1037(primary),
APSU_1430567__chunk_1034(supporting)
APS-BENCH-031comparative·BELY·medium
Q.Compared with the 9-BM Cu foil tests where switching the SR570s to
low-noise/no filters and increasing the N2 IC HV to 450 V reduced glitches
and improved the spectra, how did the later UW PSIC Mo foil runs compare
operationally in terms of SR570 filter/sensitivity settings and the
Dtheta-pitch setpoint used during the alternating knife-edge/XANES
sequence?
Gold.In the later UW PSIC Mo foil runs, the SR570 settings were initially
low-noise mode with no filters, no offsets, and 50 pA/V sensitivity
[8468__chunk_9], then changed to 12 dB low-pass, 1 kHz filter frequency,
low-noise mode, no offset, 50 pA/V sensitivity [8468__chunk_9], and later
changed again to no filters, low-noise mode, no offset, 10 pA/V sensitivity[8468__chunk_9]. During the alternating knife-edge/XANES sequence, the
Dtheta-pitch setpoint was 7965 steps on 2025/10/17 and 2025/10/18, and
7964 steps on 2025/10/19 [8468__chunk_9]. By comparison, the earlier
9-BM Cu foil tests reported setting all SR570s to low-noise, no filters and
increasing the N2-filled IC HV to 450 V , which reduced glitches and
improved the spectra [7323__chunk_0].
Nuggets.[V]In the later UW PSIC Mo foil runs, the SR570s were initially
set to low-noise mode with no filters, no offset, and 50 pA/V sensitivity.
[V]Later in the UW PSIC Mo foil runs, the SR570s were changed to
low-noise mode with a 12 dB low-pass filter at 1 kHz, no offset, and 50
pA/V sensitivity.[V]Later again in the UW PSIC Mo foil runs, the
SR570s were changed to low-noise mode with no filters, no offset, and 10
pA/V sensitivity.[V]During the alternating knife-edge/XANES sequence,
the Dtheta-pitch setpoint was 7965 steps on 2025/10/17. [o] During the
alternating knife-edge/XANES sequence, the Dtheta-pitch setpoint was 7965
steps on 2025/10/18.[V]During the alternating knife-edge/XANES
sequence, the Dtheta-pitch setpoint was 7964 steps on 2025/10/19.
Evidence.8468__chunk_9(primary),7323__chunk_0(supporting)
APS-BENCH-032causal·ICMS·medium
Q.Why would using the Particle Accumulator Ring as the injector for the
Linac Extension Area change LEA beam delivery compared with the usual
photocathode or RF gun source?
Gold.Using the Particle Accumulator Ring (PAR) as the injector would
change LEA beam delivery because the usual LEA source is the
photocathode electron gun, with RF electron guns also possible, whereas the
proposed PAR-based scheme would instead use PAR as a damping or
accumulator ring before transporting the extracted beam to LEA
[APS_2189598__chunk_0]. The passage does not provide further details on
exactly how the delivered beam properties or delivery mode would change
compared with the usual photocathode or RF gun source
[APS_2189598__chunk_0].
Nuggets.[V]The usual injector source for LEA is the photocathode electron
gun. [o] RF electron guns are also possible injector sources for LEA.
[V]In the proposed scheme, the Particle Accumulator Ring would be used as
a damping or accumulator ring before beam extraction to LEA.[V]Using
PAR would therefore replace the usual direct photocathode or RF gun
injection path with a PAR-based injection path. [o] The cited passage does
not specify exactly how the delivered beam properties would change relative
to the usual sources. [o] The cited passage does not specify exactly how
the beam delivery mode would change relative to the usual sources.
Evidence.APS_2189598__chunk_0(primary),APS_2189598__chunk_7
(supporting)
APS-BENCH-033multi_hop·ICMS·medium
Q.For PAR photon diagnostics, if I need the monitor with the larger
light-gathering acceptance, which instrument should I use and what are the
accepted synchrotron-radiation half-angles for it and for the streak camera?
Gold.Use SLM1 for the larger light-gathering acceptance, because it accepts
a larger synchrotron-radiation half-angle than the streak camera
[APS_2181087__chunk_0]. The accepted synchrotron-radiation half-angle
is 5.3 mrad for SLM1 [APS_2181087__chunk_0] and 1.9 mrad for the
streak camera [APS_2181087__chunk_0].
Nuggets.[V]The instrument to use for larger light-gathering acceptance is
SLM1.[V]SLM1 accepts a synchrotron-radiation half-angle of 5.3 mrad.
[V]The streak camera accepts a synchrotron-radiation half-angle of 1.9
mrad. [o] SLM1 has a larger accepted synchrotron-radiation half-angle
than the streak camera.
Evidence.APS_2181087__chunk_0(primary),APS_2181087__chunk_6
(supporting)
APS-BENCH-034multi_hop·TEAMS·medium
Q.If the FID pulsers are locked out and the IOC is showing timeout errors
because the devices look unresponsive, and you also can’t reach Rob Keane
and don’t have a secondary Diagnostics contact for the BESOCM system,
who should you call next for help?
Gold.If the FID pulsers are locked out, the IOC will see an unresponsive
device and encounter timeout errors [teams_Fors, Thomas
L._f01fd5c1b495__chunk_0]. If you can’t reach Rob Keane and do not have
a secondary Diagnostics contact for the BESOCM system, you should call
Guobao next for help [teams_Johnson, Andrew
N._16bc9ca6ec27__chunk_0].

APS-RAG 35
Nuggets.[V]Locked-out FID pulsers appear unresponsive to the IOC.
[o] When the FID pulsers appear unresponsive, the IOC shows timeout
errors.[V]If Rob Keane cannot be reached, another person should be
contacted for BESOCM help.[V]If there is no secondary Diagnostics
contact for the BESOCM system, the next person to call is Guobao.
Evidence.teams_Fors, Thomas L._f01fd5c1b495__chunk_0
(supporting),teams_Johnson, Andrew N._16bc9ca6ec27__chunk_0
(primary)
APS-BENCH-035causal·TEAMS·medium
Q.Why did the IOC start showing timeout errors on the FID pulsers after the
33ID PSS trip, if the trip was caused by the BPLD from the MPS side?
Gold.The 33ID PSS trip was caused by the BPLD from the MPS side
[teams_Shen, Guobao_b50b74042679__chunk_0]. The IOC started showing
timeout errors on the FID pulsers because the FID pulsers do not have
separate control power, so when they are locked out, the IOC sees them as
unresponsive devices and encounters timeout errors [teams_Fors, Thomas
L._f01fd5c1b495__chunk_0].
Nuggets.[V](cause) The 33ID PSS trip was caused by the BPLD from the
MPS side.[V]The FID pulsers do not have separate control power.
[V]When the FID pulsers are locked out, the IOC sees them as unresponsive
devices.[V](cause) The IOC timeout errors occurred because it was
trying to communicate with locked-out, unresponsive FID pulsers.
Evidence.teams_Shen, Guobao_b50b74042679__chunk_0(primary),
teams_Fors, Thomas L._f01fd5c1b495__chunk_0(primary)
APS-BENCH-036comparative·ICMS·medium
Q.Compared with a 2M undulator, how many additional motor power and
motor feedback cables should ops expect for a 4M undulator when checking
the HPMU cable/port organization?
Gold.A 2M undulator has 2 motor power cables and 2 motor feedback
cables, while a 4M undulator has 4 motor power cables and 4 motor
feedback cables [APSU_2185843__chunk_3]. Therefore, compared with a
2M undulator, ops should expect 2 additional motor power cables and 2
additional motor feedback cables for a 4M undulator
[APSU_2185843__chunk_3].
Nuggets.[V]A 2M undulator has 2 motor power cables.[V]A 2M
undulator has 2 motor feedback cables.[V]A 4M undulator has 4 motor
power cables.[V]A 4M undulator has 4 motor feedback cables.
[V]Compared with a 2M undulator, a 4M undulator has 2 additional motor
power cables.[V]Compared with a 2M undulator, a 4M undulator has 2
additional motor feedback cables.
Evidence.APSU_2185843__chunk_3(primary),
APSU_2185843__chunk_14(supporting)
APS-BENCH-037temporal·ICMS·medium
Q.During the June 2021 PAR diagnostics review for APS-U single-bunch
accumulation, which photon monitor had the smaller accepted
synchrotron-radiation half-angle, and why would that matter operationally
when trying to observe longitudinal bunch instabilities at the high-charge
condition called out for PAR?
Gold.The streak camera had the smaller accepted synchrotron-radiation
half-angle: 1.9 mrad versus 5.3 mrad for SLM1 [APS_2181087__chunk_0].
Operationally, this would matter because PAR needs diagnostics to observe
longitudinal electron bunch instabilities at the high-charge single-bunch
condition of up to 20 nC, where measurement and control of beam
instabilities may be important [APS_2170123__chunk_0]. The passages do
not explicitly state the operational consequence of the smaller half-angle
beyond identifying it as a smaller accepted synchrotron-radiation half-angle
[APS_2181087__chunk_0].
Nuggets.[V]The streak camera had the smaller accepted
synchrotron-radiation half-angle.[V]The streak camera’s accepted
synchrotron-radiation half-angle was 1.9 mrad.[V]SLM1’s accepted
synchrotron-radiation half-angle was 5.3 mrad.[V]PAR diagnostics were
needed to observe longitudinal electron bunch instabilities in single-bunch
operation.[V]The high-charge single-bunch condition called out for PAR
was up to 20 nC. [o] At that high-charge condition, measurement and
control of beam instabilities may be important operationally. [o] The cited
passages do not explicitly state any more specific operational consequence of
the smaller half-angle than that it is smaller.
Evidence.APS_2181087__chunk_0(primary),APS_2170123__chunk_0
(supporting)APS-BENCH-038temporal·BELY·medium
Q.During the 2026-02-03 9-BM Ge8 with xspress3 setup, which [main]
xsp3.ini scan-trigger PVs should we flag as missing with caget, and how
does that differ from the 2025-01-10 ISN_V ortex_ME7 note where the
EPICS IOC was running but still showed high reset counts and pileup on all
channels except ch 7?
Gold.For the 2026-02-03 9-BM Ge8 with xspress3 setup, the [main]
xsp3.ini scan-trigger PVs to flag as missing with caget are the
"d#ScanStart2.OUTN" PVs and the "scalerd#sync.INBN" PVs, i.e. the
configured d1/d2/d3 scan-start and scan-sync PVs do not exist
[10086__chunk_0]. This differs from the 2025-01-10 ISN_V ortex_ME7
note because there the EPICS IOC was running ("Using EPICS IOC
(ioc-7Channel)") but, despite that, all channels except ch 7 showed high
reset counts and pileup in the last channel [6297__chunk_0].
Nuggets.[V]In the 2026-02-03 9-BM Ge8 with xspress3 setup, the [main]
xsp3.ini scan-trigger PVs to flag as missing with caget are the
d#ScanStart2.OUTN PVs.[V]In the 2026-02-03 9-BM Ge8 with
xspress3 setup, the [main] xsp3.ini scan-trigger PVs to flag as missing with
caget are the scalerd#sync.INBN PVs. [o] The missing PVs correspond to
the configured d1, d2, and d3 scan-start and scan-sync PVs not existing.
[V]In the 2025-01-10 ISN_V ortex_ME7 note, the EPICS IOC was running.
[o] The 2025-01-10 ISN_V ortex_ME7 note specifically says "Using EPICS
IOC (ioc-7Channel)".[V]In the 2025-01-10 ISN_V ortex_ME7 note, all
channels except channel 7 showed high reset counts.[V]In the
2025-01-10 ISN_V ortex_ME7 note, pileup was present in the last channel.
Evidence.10086__chunk_0(primary),6297__chunk_0(supporting)
APS-BENCH-039multi_hop·ICMS·medium
Q.For the new PAR terahertz diagnostic, if we add the in-vacuum mirror
assembly and transmission viewport on the visible light port extension,
would that vacuum hardware be expected to fall outside ASME
pressure-vessel code scope, and what design limits in the spec determine
that?
Gold.Yes. The specification says vacuum vessels designed under this
specification are expected to be outside the scope of the ASME Boilers and
Pressure Vessel Code because of the design pressure differential being less
than 15 psi and the size being less than 6 inches diameter regardless of length
[APS_2176697__chunk_19]. The new PAR terahertz diagnostic includes an
in-vacuum mirror assembly and transmission viewport as deliverables for the
visible light diagnostics port extension [APS_2176697__chunk_0].
Nuggets.[V]The in-vacuum mirror assembly and transmission viewport for
the new PAR terahertz diagnostic would be expected to fall outside ASME
pressure-vessel code scope.[V]The specification states that vacuum
vessels under this specification are expected to be outside the scope of the
ASME Boilers and Pressure Vessel Code.[V]One design limit in the
specification is a design pressure differential of less than 15 psi.
[V]Another design limit in the specification is a vessel size of less than 6
inches in diameter. [o] The less-than-6-inches diameter limit applies
regardless of vessel length. [o] The new PAR terahertz diagnostic includes
an in-vacuum mirror assembly as a deliverable for the visible light
diagnostics port extension. [o] The new PAR terahertz diagnostic includes
a transmission viewport as a deliverable for the visible light diagnostics port
extension.
Evidence.APS_2176697__chunk_19(primary),APS_2176697__chunk_0
(supporting)
APS-BENCH-040causal·ICMS·medium
Q.Why would accumulating up to 20 nC in a single PAR bunch during
APS-U make it important to add a terahertz diagnostic for observing
longitudinal bunch instabilities?
Gold.During APS-U, the PAR is required to accumulate charge in a single
bunch up to 20 nC [APS_2170123__chunk_0]. At such high charges,
measurement and control of electron beam instabilities may be important
[APS_2170123__chunk_0]. The terahertz diagnostic is intended to be used
to observe longitudinal electron bunch instabilities
[APS_2170123__chunk_0]. Therefore, accumulating up to 20 nC in one
PAR bunch makes adding a terahertz diagnostic important so those
longitudinal bunch instabilities can be observed, measured, and controlled
[APS_2170123__chunk_0].
Nuggets.[V]During APS-U, the PAR must accumulate up to 20 nC of
charge in a single bunch.[V]At such high single-bunch charge, electron
beam instabilities become important to measure and control.[V]The

APS-RAG 36
terahertz diagnostic is intended to observe longitudinal electron bunch
instabilities.[V](cause) Adding the terahertz diagnostic is important
because it enables observation of the longitudinal instabilities that can arise
at 20 nC single-bunch operation. [o] Observing those instabilities supports
their measurement and control.
Evidence.APS_2170123__chunk_0(primary),APS_2170123__chunk_22
(supporting)
APS-BENCH-041multi_hop·WORK_REQUESTS·hard
Q.If the Booster Dipole Slave power supply tripped on an SCR failure and,
separately, the MTime power supply has failed and needs to be swapped,
which two power-supply issues should operations coordinate to address
before returning the affected systems to service?
Gold.Operations should coordinate (1) repairing the Booster Dipole Slave
power supply because it tripped on an SCR failure [45927__chunk_0] and
(2) swapping the failed MTime power supply [50362__chunk_0].
Nuggets.[V]The Booster Dipole Slave power supply tripped on an SCR
failure.[V]The Booster Dipole Slave power supply needs to be repaired
before the affected system is returned to service.[V]The MTime power
supply has failed.[V]The failed MTime power supply needs to be
swapped before the affected system is returned to service. [o] Operations
should coordinate addressing both the Booster Dipole Slave power supply
issue and the MTime power supply issue.
Evidence.45927__chunk_0(primary),50362__chunk_0(primary)
APS-BENCH-042comparative·ICMS·hard
Q.For PAR commissioning readiness, how did the status of the injector-side
ESH/HP shield design review implementation compare with the
post-installation vacuum checkout sign-offs, specifically in terms of whether
each was accepted and what operational controls or interlocks were
demonstrated on the vacuum system?
Gold.The injector-side ESH/HP shield design review implementation was
accepted as implemented per the Injector Safety Assessment document
[APS_1271753__chunk_170]. The post-installation vacuum checkout
sign-offs were also accepted [APS_1271753__chunk_137]. On the vacuum
system, the demonstrated/accepted operational controls and interlocks
included manual override demonstrated, system control of pumps and valves
accepted, over-pressure interlock demonstrated, ground-fault interlock
demonstrated, and local control and override demonstrated
[APS_1271753__chunk_137].
Nuggets.[V]The injector-side ESH/HP shield design review implementation
was accepted as implemented.[V]The post-installation vacuum checkout
sign-offs were accepted.[V]A manual override on the vacuum system
was demonstrated.[V]System control of vacuum pumps was accepted.
[V]System control of vacuum valves was accepted.[V]An over-pressure
interlock on the vacuum system was demonstrated.[V]A ground-fault
interlock on the vacuum system was demonstrated. [o] Local control on
the vacuum system was demonstrated. [o] Local override on the vacuum
system was demonstrated.
Evidence.APS_1271753__chunk_170(primary),
APS_1271753__chunk_137(primary)
APS-BENCH-043troubleshooting·ICMS·hard
Q.On SPX0, after an arc detector or cavity vacuum/over-field/quench trip
from the cryomodule interlock, the klystron RF drive drops out and stays
latched RF-off even though the protection status looks normal again. Based
on the SPX0 interlock architecture, what should the operator check and do to
restore RF, and where are the trip thresholds set?
Gold.After an arc detector trip or a cavity vacuum/over-field/quench trip
from the cryomodule interlock, the fault sends a latching trip command to
the master fast RF interlock in the amplifier, which interrupts RF drive to the
klystron and latches RF off [APSU_1430567__chunk_1037]. Even if
protection status appears normal again, the operator should check whether
the master fast interlock is still latched RF-off and issue a manual reset
command to restore RF [APSU_1430567__chunk_1037]. The trip
thresholds for the fast interlock signals are set manually by potentiometers in
the fast interlock chassis [APSU_1430567__chunk_1037].
Nuggets.[V](cause) An arc detector trip sends a latching trip command to
the master fast RF interlock in the amplifier.[V](cause) A cavity vacuum
trip from the cryomodule interlock sends a latching trip command to the
master fast RF interlock in the amplifier.[V](cause) A cavity over-field
trip from the cryomodule interlock sends a latching trip command to themaster fast RF interlock in the amplifier.[V](cause) A cavity quench trip
from the cryomodule interlock sends a latching trip command to the master
fast RF interlock in the amplifier.[V]The master fast RF interlock
interrupts RF drive to the klystron and latches RF off.[V]The operator
should check whether the master fast RF interlock is still latched RF-off.
[V](resolution) The operator must issue a manual reset command to the
master fast RF interlock to restore RF. [o] (cause) The trip thresholds for
the fast interlock signals are set manually by potentiometers in the fast
interlock chassis.
Evidence.APSU_1430567__chunk_1037(primary),
APSU_1430567__chunk_1034(supporting)
APS-BENCH-044causal·BELY·hard
Q.Why did the jumps in the I_T/scaler 3 signal disappear after the SR570
settings were manually corrected and the ini files were later updated to turn
all SR570 filters off and align the PV/channel mapping with beamline.ini,
even though LabVIEW had been showing different ion chamber settings?
Gold.The jumps disappeared because the problem was in the scaler
3/SR570 signal chain settings rather than the ion chamber itself: after
swapping cables, scaler 3 remained noisy, which indicated an issue with the
signal chain, not the IC [7413__chunk_0]. LabVIEW was not actually
controlling the ion chamber mode and filter settings, so the settings shown
there could be wrong [7413__chunk_0][7583__chunk_0]. When all SR570s
were manually set to low-noise with no filters, the jumps in scaler 3
disappeared [7413__chunk_0]. The later ini-file changes then made the
configured SR570 settings consistent by turning all SR570 filters off in
sr570.ini and changing beamline.ini channels to match sr570.ini
[7583__chunk_0]. Beamline server reads the PVs to set the SR570, while
LabVIEW is not updated when PVs change, so LabVIEW could still show
different ion chamber settings even though the actual PV/channel
configuration had been corrected [7583__chunk_0].
Nuggets.[V]After swapping cables, scaler 3 remained noisy, showing the
problem was in the scaler 3/SR570 signal chain rather than in the ion
chamber itself.[V]LabVIEW was not actually controlling the ion
chamber mode and filter settings.[V](cause) Because LabVIEW was not
controlling those settings, the ion chamber settings displayed in LabVIEW
could be wrong.[V]When all SR570 units were manually set to low-noise
mode with no filters, the jumps in scaler 3 disappeared. [o] The later
sr570.ini update turned all SR570 filters off. [o] The later beamline.ini
update changed channel assignments to match sr570.ini. [o] The beamline
server reads PVs to set the SR570 configuration. [o] LabVIEW is not
updated when the PVs change.[V](resolution) The jumps disappeared
because the actual SR570/PV/channel configuration was corrected, even
though LabVIEW still showed different ion chamber settings.
Evidence.7413__chunk_0(primary),7583__chunk_0(primary)
APS-BENCH-045causal·SDDS·hard
Q.Why would the predicted BPM change come out incorrect if we first used
sddsmatrixmult with the response matrix and correctorChange file and then
reviewed the result in sddsplot, but the matrix multiplication had the left- and
right-hand inputs reversed instead of using the response matrix on the left?
Gold.The predicted BPM change would be incorrect because
sddsmatrixmult interprets the first input file as the left-hand matrix and the
second input file as the right-hand matrix [SDDS_OAG_User’s Guide for
SDDS Toolkit Version 5.0_chunk_106__chunk_1]. In the stated BPM
example, the correct computation is the response matrix multiplied by the
columns of the correctorChange file to produce bpmExpectedChange
[SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_106__chunk_1]. If the inputs were reversed without using the
option that swaps left and right roles, then the multiplication order would not
match the documented BPM example, so the resulting predicted BPM
change would come out incorrect [SDDS_OAG_User’s Guide for SDDS
Toolkit Version 5.0_chunk_106__chunk_1]. The passages do not provide
any additional explanation about sddsplot causing the error; they only show
that sddsplot is used for plotting data [SDDS_OAG_User’s Guide for SDDS
Toolkit Version 5.0_chunk_126__chunk_0].
Nuggets.[V]sddsmatrixmult interprets its first input file as the left-hand
matrix.[V]sddsmatrixmult interprets its second input file as the
right-hand matrix.[V]In the BPM example, the correct computation is
response matrix multiplied by the columns of the correctorChange file.
[V]Reversing the response matrix and correctorChange inputs changes the
multiplication order from the documented BPM computation.[V]If the

APS-RAG 37
inputs are reversed without using the option that swaps left and right roles,
the predicted BPM change result is incorrect. [o] The cited passages do
not attribute the error to sddsplot. [o] The cited passages only show
sddsplot being used to plot data.
Evidence.SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_106__chunk_1(primary),SDDS_OAG_User’s Guide for
SDDS Toolkit Version 5.0_chunk_126__chunk_0(supporting)
APS-BENCH-046causal·OAG·hard
Q.Why would a new VSWR or ArcTrip event on one of the linac RF
stations show up in LinacRFFaultReview for operator annotation and
printing only after LinacRFFaultLogger detects a fault-count change and
writes the timestamped snapshot to the historical SDDS file?
Gold.A new VSWR or ArcTrip event would show up in
LinacRFFaultReview only after LinacRFFaultLogger detects a fault-count
change because the logger is the component that automatically detects new
faults by monitoring count PVs and then saves fault data with timestamps,
including fault snapshots, to historical SDDS files
[LinacRFFaultLogger__chunk_0]. LinacRFFaultReview is a reviewer
application that displays RF fault events with timestamps, allows operators
to annotate them, supports Review/Print reports, and saves comment updates
to SDDS files, but the passages do not say that it detects faults itself
[LinacRFFaultReview__chunk_0]. Therefore, the event appears there for
operator annotation and printing after the logger has written the timestamped
fault record to the historical SDDS file that the review workflow uses
[LinacRFFaultLogger__chunk_0][LinacRFFaultReview__chunk_0].
Nuggets.[V](cause) LinacRFFaultLogger detects new RF faults by
monitoring fault-count PVs for count changes.[V](cause) When
LinacRFFaultLogger detects a fault-count change, it writes a timestamped
fault snapshot to a historical SDDS file.[V](cause) LinacRFFaultReview
displays RF fault events with timestamps from the stored fault records. [o]
(cause) LinacRFFaultReview supports operator annotation of RF fault
events. [o] (cause) LinacRFFaultReview supports review and print
reporting for RF fault events.[V]The documentation does not indicate
that LinacRFFaultReview detects VSWR or ArcTrip faults itself.
[V]Therefore, a new VSWR or ArcTrip event appears in
LinacRFFaultReview only after LinacRFFaultLogger has created the
timestamped historical SDDS record used by the review workflow.
Evidence.LinacRFFaultLogger__chunk_0(primary),
LinacRFFaultReview__chunk_0(supporting)
APS-BENCH-047multi_hop·SDDS·hard
Q.If you need to verify an orbit-response prediction by first multiplying the
response matrix with a corrector change vector to generate expected BPM
changes, and then visually compare the result page-by-page in an interactive
X-windows plot, which SDDS tool would you use for the matrix product,
what is the required left-hand/right-hand file ordering for that multiplication,
and which plotting tool provides the zoom/pan and cursor-readout GUI for
reviewing the output?
Gold.Use sddsmatrixmult to multiply the response matrix by the corrector
change vector to generate expected BPM changes [SDDS_OAG_User’s
Guide for SDDS Toolkit Version 5.0_chunk_106__chunk_1]. For that
multiplication, the left-hand matrix must be in the first file (file1) and the
right-hand matrix must be in the second file (file2); in the example, response
is first and correctorChange is second [SDDS_OAG_User’s Guide for SDDS
Toolkit Version 5.0_chunk_106__chunk_1]. The plotting tool is sddsplot,
whose X-windows GUI provides zoom/pan and cursor readout for
interactive review page-by-page [SDDS_OAG_User’s Guide for SDDS
Toolkit Version 5.0_chunk_126__chunk_0].
Nuggets.[V]The SDDS tool to multiply the response matrix by the
corrector change vector is sddsmatrixmult.[V]In sddsmatrixmult, the
left-hand matrix must be supplied in the first input file (file1).[V]In
sddsmatrixmult, the right-hand matrix must be supplied in the second input
file (file2).[V]For this specific multiplication, the response matrix is the
left-hand matrix and therefore goes in file1.[V]For this specific
multiplication, the corrector change vector is the right-hand matrix and
therefore goes in file2.[V]The plotting tool for interactive visual review
is sddsplot. [o] sddsplot provides an X-windows GUI for reviewing the
output page-by-page. [o] The sddsplot GUI includes zoom and pan
capabilities. [o] The sddsplot GUI includes cursor readout.
Evidence.SDDS_OAG_User’s Guide for SDDS Toolkit Version
5.0_chunk_106__chunk_1(primary),SDDS_OAG_User’s Guide forSDDS Toolkit Version 5.0_chunk_126__chunk_0(primary)
APS-BENCH-048multi_hop·ICMS·hard
Q.Before we install and commission the new PAR terahertz
synchrotron-radiation diagnostic, which APS Tech Note documents should
operations review to cover both the engineering specification and the physics
requirements, and how do those documents relate to the measurement
capability being proposed for PAR?
Gold.Operations should review the PAR terahertz diagnostic engineering
specification, DIAG-TN-2021-009 (Rev. 3), titled "Particle Accumulator
Ring Terahertz Diagnostic Engineering Specification Document"
[APS_2176698__chunk_7], and the physics requirements document,
DIAG-TN-2020-049 (Rev. 1), titled "Particle Accumulator Ring Terahertz
Diagnostic Physics Requirements Document" [APS_2176698__chunk_7].
These documents relate to the proposed PAR measurement capability
because the design-review response states that "a new measurement
capability is proposed for the Particle Accumulator Ring (PAR)" and that
"Terahertz frequency synchrotron radiation is proposed"
[APS_2176698__chunk_0]. The passages do not further explain how each
of the two tech notes maps onto specific measurement capabilities beyond
identifying them as the engineering specification and physics requirements
documents for the PAR terahertz diagnostic [APS_2176698__chunk_7].
Nuggets.[V]Operations should review APS Tech Note
DIAG-TN-2021-009, Revision 3.[V]DIAG-TN-2021-009 (Rev. 3) is
titled "Particle Accumulator Ring Terahertz Diagnostic Engineering
Specification Document."[V]Operations should review APS Tech Note
DIAG-TN-2020-049, Revision 1.[V]DIAG-TN-2020-049 (Rev. 1) is
titled "Particle Accumulator Ring Terahertz Diagnostic Physics
Requirements Document."[V]The proposed capability is a new
measurement capability for the Particle Accumulator Ring (PAR).[V]The
proposed measurement uses terahertz-frequency synchrotron radiation.
[o] The cited passages identify the two tech notes as the
engineering-specification and physics-requirements documents for the PAR
terahertz diagnostic. [o] The cited passages do not further explain how
each tech note maps to specific measurement capabilities beyond those
document roles.
Evidence.APS_2176698__chunk_7(primary),APS_2176698__chunk_0
(supporting)
APS-BENCH-049multi_hop·BELY·hard
Q.Before relying on 9-BM’s new DTcorrI0 from the vortex ME4, what
beam-based check should I run to confirm the DXP deadtime-correction PVs
are actually working, and if the result is not flat as I change the incident flux,
which Wenachee2 deadtime-correction parameters would I need to adjust?
Gold.Run a 1D dummy scan with about 0.3 s integration time per point and
plot Element_total / deadtime corrected i0 while changing the incident flux
by putting in attenuators of 10% to 50%; if that plot stays constant, the
deadtime-corrected I0 is working well [7096__chunk_0]. Before relying on
9-BM’s new DTcorrI0, this beam-based check is needed because the setup
was only tested without beam and the DXP PVs still need to be verified with
beam [10660]. If the result is not flat as I0 changes, adjust the Wenachee2
DT-calculation parameters: change the 2.5e-6 value so the plot stays
constant when i0 changes, and also change the divided-by-12 factor if you
are not using 12 elements [7096__chunk_0].
Nuggets.[V]Run a 1D dummy scan as the beam-based check before relying
on 9-BM’s new DTcorrI0. [o] Use about 0.3 s integration time per point in
that scan.[V]Vary the incident flux during the scan by inserting
attenuators spanning roughly 10% to 50%.[V]Plot Element_total divided
by deadtime-corrected I0 to evaluate whether the DXP deadtime-correction
PVs are working.[V]If that plot remains constant as incident flux
changes, the deadtime-corrected I0 is working well. [o] This beam-based
verification is needed because the setup had only been tested without beam.
[V]If the plot is not flat as I0 changes, adjust the Wenachee2
deadtime-calculation parameter currently set to 2.5e-6.[V]If the plot is
not flat, also adjust the Wenachee2 factor that divides by 12 when the
detector is not using 12 elements.
Evidence.7096__chunk_0(primary),10660(supporting)
APS-BENCH-050multi_hop·BELY·hard
Q.At 9-BM, once beam is back and you need to verify the new caqtdm
DTcorrI0 setup for the vortex ME4, what beam-based check should you run
and which parameter(s) would you adjust if the deadtime-corrected I0 does
not stay constant as attenuation is changed?

APS-RAG 38
Gold.Once beam is back, run a 1D dummy scan with about 0.3 sec
integration time per point, plot Element_total / deadtime corrected I0, and
insert attenuators from about 10% to 50% to see whether the plot stays
constant [7096__chunk_0]. If the deadtime-corrected I0 does not stay
constant as attenuation is changed, adjust the DT calculation parameters on
the Wenachee2 computer: change the 2.5e-6 value so the plot stays constant
when I0 changes, and also change the divide-by-12 factor if you are not
using 12 elements [7096__chunk_0]. The 9-BM note says the setup was
tested without beam and that with beam you still need to test that the dxp
PVs are updating [10660].
Nuggets.[V]Once beam is back, run a 1D dummy scan to verify the newcaqtdm DTcorrI0 setup for the vortex ME4. [o] Use about 0.3 seconds
integration time per point for the dummy scan.[V]Plot Element_total
divided by deadtime-corrected I0 during the check.[V]Insert attenuators
over roughly the 10% to 50% range during the scan.[V]The verification
criterion is that Element_total / deadtime-corrected I0 should stay constant
as attenuation is changed.[V]If deadtime-corrected I0 does not stay
constant with attenuation, adjust the DT calculation parameters on the
Wenachee2 computer.[V]Change the 2.5e-6 parameter until the plot
stays constant when I0 changes.[V]Change the divide-by-12 factor if the
detector is not using 12 elements. [o] With beam, also verify that the dxp
PVs are updating.
Evidence.7096__chunk_0(primary),10660(supporting)