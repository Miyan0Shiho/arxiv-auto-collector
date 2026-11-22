# KnowledgeClaw: Towards Agentic Knowledge Engineering via Omnimodal Hypergraphs

**Authors**: Haoran Luo, Jiapu Wang, Linhao Luo, Luu Anh Tuan

**Published**: 2026-11-18 12:06:06

**PDF URL**: [https://arxiv.org/pdf/2611.24294v1](https://arxiv.org/pdf/2611.24294v1)

## Abstract
Knowledge engineering (KE) turns raw files into a reasoning-ready knowledge base, a task now within reach of LLM agents. KE has progressed from chunk and graph retrieval-augmented generation (RAG) to memory agents, agentic KE and multimodal KE. However, a live knowledge base needs omnimodality, a closed-loop lifecycle and multi-granular application at once, yet existing systems meet at most two and no benchmark measures all three. We formalize Omnimodal Knowledge Engineering (OmniKE) and release OmniKE-Bench, six tasks with 1,829 model-free items over 3,960 real files in five modalities. We further propose KnowledgeClaw, an agentic system that builds an omnimodal typed hypergraph, keeps it current with a closed-loop lifecycle agent and routes each query to the granularity it needs. Against 30 baselines, it ranks first on all six tasks with an overall OKE of 55.9, +14.6 over the strongest baseline. Our software and data are publicly available at https://github.com/LHRLAB/KnowledgeClaw.

## Full Text


<!-- PDF content starts -->

Preprint.
KNOWLEDGECLAW: TOWARDSAGENTICKNOWLEDGE
ENGINEERING VIAOMNIMODALHYPERGRAPHS
Haoran Luo1†, Jiapu Wang2∗, Linhao Luo3, Luu Anh Tuan1,4
1Nanyang Technological University, Singapore
2Nanjing University of Science and Technology3Monash University4VinUniversity
†Project Lead.∗Corresponding Author.
haoran.luo@ntu.edu.sg, jiapu.wang@njust.edu.cn, anhtuan.luu@ntu.edu.sg
ABSTRACT
Knowledge engineering (KE) turns raw files into a reasoning-ready knowledge
base, a task now within reach of LLM agents. KE has progressed from chunk and
graph retrieval-augmented generation (RAG) to memory agents, agentic KE and
multimodal KE. However, a live knowledge base needs omnimodality, a closed-
loop lifecycle and multi-granular application at once, yet existing systems meet
at most two and no benchmark measures all three. We formalizeOmnimodal
Knowledge Engineering(OmniKE) and releaseOmniKE-Bench, six tasks with
1,829model-free items over3,960real files in five modalities. We further propose
KnowledgeClaw, an agentic system that builds an omnimodal typed hypergraph,
keeps it current with a closed-loop lifecycle agent and routes each query to the
granularity it needs. Against30baselines, it ranks first on all six tasks with an
overall OKE of55.9,+14.6over the strongest baseline. Our software and data are
publicly available athttps://github.com/LHRLAB/KnowledgeClaw.
1 INTRODUCTION
Personal Files
LLM AgentHypergraph
Multi-Granular Application
Closed-Loop Lifecycle
Omnimodality
Graph RAG static corpus
Memory Agents flat stores
Multimodal RAG static corpus
Agentic KE text only
KnowledgeClaw (Ours)
read extract
link revise
Figure 1: Existing KE systems each miss at least
one of the three properties that a live knowledge
base needs, while the proposed KnowledgeClaw
achieves all three of the above at the same time.Knowledge engineering (KE) distills raw files
into a reasoning-ready knowledge base (Studer
et al., 1998), but manual construction long kept
it from scaling (Pan et al., 2024). LLM agents
lift this limit (OpenAI et al., 2024; Yao et al.,
2023; Wang et al., 2024): an agent can read
files, extract entities, link records and revise its
store (Xu et al., 2024), curating all of a person’s
growing files into a single live knowledge base.
Knowledge structures grew from passages in
RAG (Lewis et al., 2020) to entity graphs in
GraphRAG (Edge et al., 2025) andn-ary hyper-
edges in HyperGraphRAG (Luo et al., 2025).
Memory agents (Packer et al., 2024) keep a
writable store that MemPro (Liu et al., 2026d)
evolves, agentic KE (OpenClaw contributors,
2026) lets one agent build and keep it, and
Omni-SimpleMem (Liu et al., 2026a) reaches
audio and video. Yet no system unites them all.
A personal knowledge base needs all of this at once: papers arrive as PDFs, talks as audio or video
and notes as text; papers are revised or withdrawn; queries range from one-file lookups to multi-file
joins and topic synthesis. This raises three challenges (Figure 1); each existing line meets at most
two.(i) Omnimodality:an entity in a PDF and a recording should be one node, yet only multimodal
RAG reads beyond text.(ii) Closed-loop lifecycle:every edit should reach everything derived from
its file, yet graph and multimodal RAG index a static corpus.(iii) Multi-granular application:
each query should use the file, subgraph or community level, yet memory agents keep flat stores.
1
arXiv:2611.24294v1  [cs.AI]  18 Nov 2026

Preprint.
Three Challenges
Omnimodality
Closed-Loop Lifecycle
Multi-Granular Application
Existing Systems: At Most TwoOmniKE-Bench No Model-Made Answers
10 Personas 3,960 Files 1,829 Items
KnowledgeClaw
Ingest Once Into a Knowledge Base
and Answer Later Queries
C 1Omnimodal
Typed Hypergraph
Construction
T1
T2
C 2Closed-Loop
Lifecycle Agent
A D D
 D E L
R E F R E S H
Maintenance
T5
C 3Multi-Granular
Retrieval
?
File
Subgraph
Community
Application
T3
T4
T6
Results
O u r s: OKE 55.9
+ 1 4 . 6 over M e m P r o
vs. 30 Baselines
T1T2T3T4T5T6
w/o C1 T1 − 2 0 . 0
w/o C2 T5 − 4 6 . 0
w/o C3 OKE − 2 9 . 2T1
T2
T3
T4
T5
T6
Six Tasks
Graph
RAGMemory
Agents
Multimodal
RAGAgentic
KECatalogue
Record𝒦 = 𝒮 ( 𝒟 )
Figure 2: Overview of KnowledgeClaw on OmniKE. C1 builds an omnimodal hypergraph (T1, T2),
C2 maintains it (T5) and C3 routes each query (T3, T4, T6); it leads on all six tasks (OKE=55.9).
No benchmark measures all three, so we formalizeOmnimodal Knowledge Engineering(OmniKE):
a system ingests a person’s files once into a knowledge base and answers later queries from it. We
releaseOmniKE-Bench, six tasks on construction (T1, T2), application (T3, T4, T6) and mainte-
nance (T5). Its1,829model-free items cover3,960real files in five modalities from ten researchers.
On top of OmniKE-Bench, we proposeKnowledgeClaw, an agentic system of three components
(Figure 2).(C1) Omnimodal typed hypergraphreads each file natively into a catalogue record
and merges all records into one hypergraph, where files of any modality meet at shared entities.
(C2) Closed-loop lifecycle agenttracks every node’s source files and commits each ADD, DELor
REFRESHalong them, so no stale knowledge survives an edit.(C3) Multi-granular retrieval with
intent routingdraws on just the tier that each query needs, so all named files enter the evidence.
Against30baselines in ten groups, KnowledgeClaw ranks first on all six tasks with an overall OKE
of55.9,+14.6over the strongest baseline, MemPro. Removing C1, C2 or C3 lowers T1 by20.0, T5
by46.0or OKE by29.2. Even with only256evidence tokens, it beats every full-context baseline.
2 RELATEDWORK
Our setting joins the two lines of work reviewed below, and Appendix G surveys both lines in full.
LLM-based knowledge engineering.KE structures grew from chunks (StandardRAG, Lewis et al.,
2020; RAPTOR, Sarthi et al., 2024) to entity graphs (GraphRAG, Edge et al., 2025; LightRAG, Guo
et al., 2025b; HippoRAG 2, Guti ´errez et al., 2025), agentic graph RAG (KAG, Liang et al., 2025;
Youtu-GraphRAG, Dong et al., 2026a), curated KGs (iText2KG, Lairgi et al., 2025; OntoKG, Li
et al., 2026c) and hypergraphs (HyperGraphRAG, Luo et al., 2025; Hyper-KGGen, Huang et al.,
2026; HyperRAG, Lien et al., 2026). Memory agents (MemGPT, Packer et al., 2024; A-MEM, Xu
et al., 2025; Mem0, Chhikara et al., 2025; Graphiti, Rasmussen et al., 2025; MIRIX, Wang & Chen,
2025; MemOS, Li et al., 2025), self-evolving systems (MemEvolve, Zhang et al., 2026; ATOM,
Lairgi et al., 2026; MemPro, Liu et al., 2026d) and agentic KE (OpenClaw, 2026; Graphify, 2026;
Cognee, Markovic et al., 2025; LLM Wiki, Karpathy, 2026; OKF, Google Cloud, 2026) make the
store writable, but edit single facts or pages, not everything derived from a changed source file.
Multimodal retrieval-augmented generation.Multimodal RAG extends this line to visual–textual
graphs (MMGraphRAG, Wan & Yu, 2026), audio-visual KGs (M3KG-RAG, Park et al., 2026a)
and multimodal memories (Omni-SimpleMem, Liu et al., 2026a), but none maintains its store.
KnowledgeClaw joins both lines in one omnimodal hypergraph kept consistent under file edits.
2

Preprint.
(A) Six OmniKE Tasks
Construction · What Each File Holds
Application · Composes Records
Application Maintenance
T 1
 4 0 0
Knowledge
Acquisition
T 2
 4 0 0
Cross-Modal
Coreference
T 3
 4 0 0
Multi-Hop
Retrieval
T 4
 1 1 0
Cross-Modal
Reasoning
T 6
 3 8 2
Knowledge
Discovery
T 5
 1 3 7
Lifecycle
Consistency
A
 B(B) OmniKE-Bench Corpus
3,960
Real
Files
LAW
EDU
FIN
PSY
HUM
 GAM
POL
HC
BLD
MUS
10 Personas
 Per Persona
90 180 18
 18
15.8 h Media
 6.9 GB(C) Four-Phase Construction
No Model-
Made Gold
1,829 Items
1Collect
Human-Made Gold
Redistributable
2Lift
Normalize,
Nine Contracts
4Balance Answers
Recompute
Verify
 3Fill Templates
From Rows
Instantiate
(D) Model-Free Scoring No Model Call
Predicted
Gold
Token F1 ≥ 0.5
Set F1: T1, T3, T4
Accuracy: T2, T5
Balanced Accuracy: T6
T5: Half per Correct Phase Task (0–100)
 PersonaMean
of 10Mean
of Tasks
OKEΣ = 1,829 items
Figure 3:(A)Six OmniKE tasks in three groups: construction (T1, T2), application (T3, T4, T6)
and maintenance (T5).(B)OmniKE-Bench spans ten personas, five modalities and3,960real files.
(C)Four phases build its1,829items with no model-made gold.(D)Model-free scoring and OKE.
3 THEOMNIKE TASKS ANDBENCHMARK
Task definition.Apersonais one researcher’s file collectionDover the modalitiesM=
{text,pdf,image,audio,video}. A systemSreadsDonce, atINGEST, and commits an artefact
K=S(D), its knowledge store; every later query is answered throughKalone, each in a fresh
context. The six tasksT 1–T6(Figure 3A, Table 5) form three groups, one per property.Construction
asks whetherKcaptures what each file holds: extract a file’s entities of given types (T 1) and match
a media file to its authoritative description (T 2).Applicationasks whetherKcomposes records:
answer 2–4-hop joins (T 3), joins across modalities (T 4) and a file’s controlled-vocabulary labels
(T6).Maintenance(T 5) asks whetherKabsorbs an edit∆∈ {ADD,DEL,REFRESH}: the system
answers, commits the next version as a REFRESHthat changes the answer, and answers it again.
Benchmark and statistics.OmniKE-Bench holds ten personas from healthcare to games, each with
90documents as PDF and text,180images,18audio and18video clips:3,960files,15.8h of media
and6.9GB in all (Figure 3B, Table 6). Its1,829items comprise400each forT 1–T3,110forT 4,137
forT 5and382forT 6, and, unlike prior omnimodal benchmarks, every answer traces to human-made
metadata, deterministic parsing, official transcripts or version chains, never to a model’s output.
Construction.Four phases (Figure 3C) follow three rules: answers come only from authoritative
structure, a model only paraphrases, and no human edits or picks an item.Collectadmits redis-
tributable sources with human-made annotation, so ASR, OCR and model descriptions may be sys-
tem input but never gold.Liftnormalizes every file and maps each source onto nine canonical con-
tracts (records, alignments, events, . . . ), so each task is written once for all personas.Instantiatefills
templates from contract rows; GPT-5-nano only rewrites a stem into prose, and deterministic checks
discard any item whose slots changed or whose stem leaks the answer.Verifybalances answers so
that priors carry no signal and recomputes them with an independent implementation that agrees on
all1,829items, so every answer is reproducible from the raw source files alone (Appendix D.2).
Evaluation metrics.Scoring calls no model (Figure 3D). Free-form answers (T 1,T3,T4) are
matched to gold values one-to-one and leniently (a date at the gold’s precision in any format, any
other value by normalized equality or a word-token F1 of at least0.5) and scored by the set F1 of
the matches.T 2andT 5are four-way choices, aT 5item scoring half per correct phase, andT 6scores
balanced accuracy over positive and negative candidates. A task score is the mean item score (0–
100), a persona score the mean over the tasks it holds, and OKE the mean of the ten persona scores,
so each persona weighs the same whatever its number of items (full definitions in Appendix D.3).
3

Preprint.
Entity IndexHyperedge Index
Segment Index BM25
(A) C1: Omnimodal
Typed Hypergraph(B) C2: Closed-Loop
Lifecycle Agent(C) C3: Multi-Granular
RetrievalText
PDF
Text
Projection
Native
ReadingCatalogue
Record
Author
Subject
Identifier
Events
Provenance π(x)
Record Event Cross-ModalA D D DEL REFRESH
Lifecycle
AgentReceive
EditCommit Along
Provenance
Rebuild
IndicesAudit
Invariant𝒦 𝒦
Every Node Rests
On Present FilesQuery q Handle h
Anchor
FilesAnchor
Entities
Intent
Router
File
TierSubgraph
TierCommunity
Tier
Anchor Cards
Linked-Work Cards
Top- k Hyperedges
PassagesAnswer
Every Named File
Enters EvidenceSegmentsTitle
Prune π(x)
No Model Call Handle h LLM Fallback
Frozen ΩDel: 9 ms Refresh: 2.6 sSpan ψ
Image
Audio
Video
LookupJoin Synthesis
Edit Δ
1
 2
3
 4
Figure 4:KnowledgeClaw. (A)C1 merges catalogue records into one omnimodal typed hypergraph
K.(B)C2 commits each edit in place along provenance.(C)C3 routes each query to the right tier.
4 METHODOLOGY: KNOWLEDGECLAW
KnowledgeClaw (Figure 4) realizes the systemSof §3 with three components, one per task group:
C1 reads a personaDonce into an omnimodal typed hypergraphK(§4.1), C2 commits each edit∆
toKalong source files (§4.2), and C3 answers each queryqfrom just the tier ofKit needs (§4.3).
4.1 C1: OMNIMODALTYPEDHYPERGRAPH
The artefact.Nodes are the filesa∈ Dand their entitiesv∈ V; a hyperedgee∈ Ejoins nodes and
is typed by a kindκ(e), and the provenanceπ(x)gives the source files of an entity or hyperedgex:
K= (D,V,E, π),E ⊆2D∪V, κ:E → {record,event,cross-modal}, π:V ∪ E →2D.(1)
Native reading.A fileaof modalityµ(a)∈ Mbecomes a text surface˜a: text and PDF via a shared
text projectionproj, media natively via the frozen omnimodal modelΩon the raw fileraw(a)under
a promptp µ(a)(promptsp ·in Appendix A);splitcuts˜ainton asegmentsσ a,jof∼1,000tokens:
˜a=proj(a), µ(a)∈ {text,pdf},
Ω 
raw(a);p µ(a)
, µ(a)∈ {image,audio,video},(σa,1, . . . , σ a,na) = split(˜a).(2)
Catalogue records and identity.One callΩ(˜a;p rec)writes the catalogue record ofa: dated events
and a value setZ a,γper fieldγof a fixed schemaΓ(title, author, subject, id, . . . ). A valuezbecomes
the entityι(z), each field arecordhyperedge and each event aneventhyperedge{a}with its date:
ea,γ={a} ∪
ι(z) :z∈Z a,γ	
, π(e a,γ) ={a}, π(v)←π(v)∪ {a} ∀v∈e a,γ∩ V,
ι(z) = norm(z),V(a) =S
γ∈Γ 
ea,γ∩ V
,[v] =
v′∈ V:ι p(v′) =ι p(v)	
,(3)
wherenormcase-folds a value and trims its edge punctuation, so records meet on shared entities,
and the name keyι p(first initial, last name) puts “M. Langtry” and “Max Langtry” in one alias class.
Cross-modal coreference.An entity whose files span≥2modalities (µapplied elementwise) gets
across-modalhyperedge, and files of different modalities sharing a title or identifier aretwins:
e×
v={v}∪π(v)if|µ(π(v))| ≥2, a≈b⇐⇒µ(a)̸=µ(b)∧b∈π 
V ∩(e a,title∪ea,id)
,(4)
and an encoderϕembeds entities, hyperedges and segments into dense indicesI V, IE, Iσ; BM25
indexes every surface˜a, and each index key names the entity, hyperedge or segment that it encodes.
4

Preprint.
Proposition 1.InK, any two filesa, b∈ D, of the same or different modalities, whose records name
a common entityvare joined within two hyperedges, namely their record hyperedges containingv.
Proof.Empirical support is in §5.4 (Figure 6) and the proof in Appendix B.1.
4.2 C2: CLOSED-LOOPLIFECYCLEAGENT
Edits.A handlehnames a document across its versions. LetD tbe the files present after thet-th
edit∆ tandη t(h)the files of the newest version ofh; the three edits of §3 act as
Dt=

Dt−1∪A,∆ t=ADD(A),
Dt−1\ηt−1(h),∆ t=DEL(h), 
Dt−1\ηt−1(h)
∪A′,∆ t=REFRESH(h→A′),after whichη t(h) =A′,(5)
whereAholds added files,A′a new version,K t= (D t,Vt,Et, πt)is the artefact after∆ tand
Σt={σ a,j:a∈ D t}its segments. ADD(A)runs C1 onAalone; DEL(h)calls no model: it prunes
ηt−1(h)from eachπ(x), so an edit touches only what rests on changed files; REFRESH(h→A′)is
DEL(h)then ADD(A′). Commits restoreprovenance soundness, withkeys(I)being the keys ofI:
Vt∪ Et=
x∈ V t−1∪ Et−1:πt−1(x)̸⊆η t−1(h)	
, π t(x) =π t−1(x)\η t−1(h),(6)
∀x∈ V t∪ Et:∅ ̸=π t(x)⊆ D t,keys(I V)∪keys(I E)∪keys(I σ)⊆ V t∪ Et∪Σt.(7)
Proposition 2.After any sequence of edits∆ 1, . . . ,∆ t, all ofK trests on present files, i.e., it satisfies
Eq. (7), so no evidence is stale: no later query draws evidence from a deleted or superseded file.
Proof.Empirical support is in §5.5 (Figure 7) and the proof in Appendix B.2.
4.3 C3: MULTI-GRANULARRETRIEVAL WITHINTENTROUTING
Anchors and intent routing.Handleshdl(q)inqresolve viaη tto anchor filesH q, and each quoted
spanψ∈Ψ qto the entityι(ψ)or its nearest entity inI Vat cosine≥θ= 0.86, giving anchor entities
Vq. Relational, lookup and synthesis cuesrel(q),look(q),syn(q)∈ {0,1}routeqto a tierg(q):
Hq=[
h∈hdl(q)ηt(h), g(q) =

file, H q̸=∅, V q=∅,¬rel(q),
subgraph,else ifH q∪Vq̸=∅ ∨(look∧ ¬syn)(q),
community, H q∪Vq=∅ ∧syn(q),
Ω(q;p route),otherwise.(8)
Tiers and answering.Thefiletier reads only the cardscard(a)(record, events, opening text) of
Hq. A join walks from the anchors, scoring each other filebwith1[·]the indicator,π([v])the files
of[v],⊑containment inhead(b)(title, subject) or text ˜b, andβ ψ(b)∈[0,1]a BM25 score:
sq(b) =X
v∈Vq101
b∈π([v])
+X
ψ∈Ψ q
61
ψ⊑head(b)
+ 31
ψ⊑˜b
+ 2β ψ(b)
+X
a∈HqX
u∈V(a)log 
1 +|D t|/|π(u)|
1
b∈π([u])
,(9)
skipping hubs, entities in overmax(20,|D t|/5)files. Twins form worksw;W qkeeps up to six
within0.6of the best score¯s q(w). For synthesis,Ωreports on the64largest Louvain clusters under
pcom. The evidenceX qholds the anchor cards, those ofW q, the four bestI σsegments and the top-k
hyperedges by rank fusionrrf qof listsL q,V(hyperedges of the2kentities nearestqinI V) andL q,E
(the2knearest inI E), or the nearest reports;ˆy q= Ω(q,X q;pans)answersq(withλ= 4,k= 10):
¯sq(w) =X
b∈wsq(b),rrf q(e) =X
L∈{L q,V, Lq,E}1[e∈L] 
λ+ rank L(e)
.(10)
Proposition 3.For every queryq, each filea∈H qthatqnames enters its evidenceX qascard(a),
whateverk,λ,θand the size ofD t, which top-ksimilarity with anykbelow that size cannot ensure.
Proof.Empirical support is in §5.6 (Figure 8) and the proof in Appendix B.3.
5

Preprint.
Table 1:Main results on OmniKE-Bench.T 1–T6: per-task scores over ten personas; HC–GAM:
per-persona means over tasks (descriptive);OKE: mean of persona columns.Bold: best; underline :
best baseline;italic: ablation equal to the full model. Tags: input ( /align-lef◎text, ⊷multimodal,
♂photo-videoomnimodal), structure ( none, chunk, graph, hypergraph) and lifecycle ( /toggle-onyes,/toggle-offno).
Tasks Personas
MethodT 1T2T3T4T5T6HC FINEDULAWPSYPOLHUMBLDMUSGAM OKE
No knowledge engineering
/align-lef◎/toggle-off NaiveGeneration 0.0 28.0 0.8 1.8 37.2 59.8 17.9 13.4 22.0 10.8 22.8 9.2 10.0 16.9 18.4 18.7 16.0
Chunk / hierarchical retrieval
/align-lef◎/toggle-off StandardRAG 6.1 27.2 9.5 12.7 51.8 55.7 30.4 18.6 29.2 13.9 32.8 7.5 14.0 19.3 23.9 20.6 21.0
/align-lef◎/toggle-off RAPTOR 6.4 28.5 6.2 0.9 39.4 58.0 24.6 17.5 25.0 15.0 30.1 11.0 12.0 16.2 21.7 20.2 19.3
Graph RAG
/align-lef◎/toggle-off LightRAG 13.6 31.0 9.5 2.7 50.0 58.0 30.6 24.5 29.7 22.2 33.3 6.7 18.9 20.7 24.7 27.7 23.9
/align-lef◎/toggle-off GraphRAG 3.5 26.8 10.5 6.4 32.8 52.9 26.0 15.3 22.6 12.8 33.3 8.1 13.3 14.8 15.8 18.2 18.0
/align-lef◎/toggle-off HippoRAG 2 3.1 24.5 8.8 4.5 46.7 53.1 27.7 18.1 22.1 11.6 23.8 7.5 11.2 19.2 17.3 26.2 18.5
Agentic graph RAG
/align-lef◎/toggle-off KAG / OpenSPG 6.7 22.8 7.0 2.7 46.7 53.4 24.9 18.4 28.7 14.2 23.7 8.3 12.8 18.3 21.5 16.1 18.7
/align-lef◎/toggle-off Youtu-GraphRAG 1.6 27.5 9.0 4.5 32.8 53.0 23.1 18.3 20.8 8.7 30.8 7.8 11.7 17.4 15.7 17.8 17.2
Knowledge graph construction
/align-lef◎/toggle-off iText2KG 1.3 25.2 2.2 0.0 27.7 53.8 17.4 9.5 20.2 10.8 21.7 5.0 12.4 17.8 14.5 14.8 14.4
/align-lef◎/toggle-off OntoKG 4.8 29.2 2.0 1.8 31.8 56.2 19.6 13.2 25.0 7.5 24.7 7.8 15.8 14.6 19.2 20.9 16.8
n-ary / hypergraph knowledge engineering
/align-lef◎/toggle-off HyperGraphRAG 4.4 29.5 7.8 5.5 40.1 53.0 25.1 15.6 27.2 15.6 30.1 8.5 12.0 17.4 23.9 17.5 19.3
/align-lef◎/toggle-off Hyper-KGGen 1.4 29.8 3.0 0.0 27.7 54.6 18.8 12.1 21.2 16.3 22.5 10.0 13.7 12.5 16.7 17.9 16.2
/align-lef◎/toggle-off HyperRAG 1.1 30.2 6.5 2.7 31.8 53.1 25.4 13.3 23.7 9.2 30.4 10.8 13.3 13.8 18.0 13.0 17.1
Memory agents
/align-lef◎/toggle-on MemGPT / Letta 8.8 27.2 4.8 4.5 34.3 55.4 26.7 18.2 21.9 17.1 25.6 10.1 11.4 18.6 18.7 16.2 18.4
/align-lef◎/toggle-on A-MEM 7.8 25.2 10.2 10.0 49.3 58.1 29.2 20.1 28.0 12.1 31.5 8.9 13.1 20.2 26.3 20.2 21.0
/align-lef◎/toggle-on Mem0 2.4 29.5 3.5 2.7 38.0 53.0 21.9 14.5 23.6 15.1 23.3 6.2 13.3 19.6 18.7 18.2 17.4
/align-lef◎/toggle-on Graphiti (Zep) 10.9 66.0 12.0 5.5 58.8 68.5 37.0 35.6 43.1 35.0 50.1 18.3 37.2 26.6 30.5 38.8 35.2
/align-lef◎/toggle-on MIRIX 1.6 34.0 3.2 1.8 38.7 55.1 21.7 13.9 23.9 13.8 25.5 12.0 13.7 19.1 24.1 18.5 18.6
/align-lef◎/toggle-on MemOS 13.6 28.8 5.0 2.7 47.8 54.8 26.8 22.2 31.8 15.3 24.0 9.2 18.7 17.4 24.2 26.6 21.6
Self-evolving / closed-loop lifecycle
/align-lef◎/toggle-on MemEvolve 0.0 27.0 0.5 4.5 32.5 60.6 19.9 13.0 22.1 10.0 23.7 11.7 8.3 15.9 17.6 10.0 15.2
/align-lef◎/toggle-on ATOM 5.1 30.5 4.2 0.0 30.3 54.8 22.4 14.8 23.7 10.8 21.7 11.7 11.6 16.7 22.3 19.7 17.5
/align-lef◎/toggle-on MemPro 11.4 78.2 13.5 8.2 78.8 69.5 43.5 45.2 50.7 33.4 53.6 23.6 36.5 37.4 43.4 45.5 41.3
Agentic knowledge engineering
/align-lef◎/toggle-on OpenClaw 2.9 61.5 2.8 0.9 41.2 65.2 27.1 23.0 36.9 22.5 36.2 19.9 31.8 17.2 29.0 28.3 27.2
/align-lef◎/toggle-on Graphify 0.9 35.8 1.2 0.0 35.8 52.4 18.3 20.2 22.8 15.9 28.3 10.2 10.5 14.2 20.7 16.4 17.8
/align-lef◎/toggle-on Cognee 12.8 44.0 5.0 2.7 45.3 59.7 28.2 28.2 38.1 18.1 30.9 21.5 11.7 19.4 28.9 30.4 25.6
/align-lef◎/toggle-on LLM Wiki 20.0 66.2 6.5 0.0 50.7 68.1 33.9 36.5 45.8 32.5 44.5 21.6 42.4 24.2 36.5 34.8 35.3
/align-lef◎/toggle-on OKF 27.2 68.5 8.8 17.3 54.4 67.3 45.5 44.3 46.3 38.9 49.7 16.8 37.6 27.0 38.7 40.6 38.5
Omnimodal / multimodal knowledge engineering
♂photo-video/toggle-off Omni-SimpleMem 10.3 70.2 10.2 10.9 52.2 68.8 40.6 34.0 44.7 29.9 48.3 20.2 26.6 26.0 36.4 37.9 34.5
⊷/toggle-off MMGraphRAG 1.6 32.0 10.5 1.8 32.5 51.2 23.9 17.8 21.9 14.7 31.9 9.2 14.0 16.7 19.1 18.6 18.8
♂photo-video/toggle-off M3KG-RAG 17.2 60.8 1.5 0.0 32.8 60.9 30.4 32.6 39.3 30.1 41.8 15.1 29.7 17.7 25.7 25.9 28.8
Ablations
/align-lef◎/toggle-on w/o C1 (no hypergraph) 31.6 81.2 11.0 7.3 95.3 74.6 51.7 55.2 57.8 42.6 58.1 25.8 51.8 43.4 53.5 50.5 49.1
♂photo-video/toggle-off w/o C2 (no lifecycle)51.6 82.8 15.8 20.949.676.0 60.952.3 50.546.6 65.2 28.3 57.335.3 53.3 54.5 50.4
♂photo-video/toggle-on w/o C3 (flat retrieval) 15.7 34.8 8.0 5.5 62.8 57.5 28.2 22.3 36.7 15.6 35.2 16.7 20.7 31.6 30.0 29.5 26.7
♂photo-video/toggle-onKnowledgeClaw (ours) 51.6 82.8 15.8 20.9 95.6 76.0 60.9 64.4 59.2 46.6 65.2 28.3 57.3 47.3 64.5 65.7 55.9
5 EXPERIMENTS
We answer nine research questions (RQs).RQ1–RQ2:Does KnowledgeClaw outperform the30
baselines, and does each component count?RQ3–RQ5:Does C1 join files across modalities
(Proposition 1), C2 keep answers current (Proposition 2) and C3 supply the evidence each query
needs (Proposition 3)?RQ6–RQ7:Is it efficient and robust to its settings?RQ8:Is the lead fair to
the baselines?RQ9:How does it solve, from raw media, a query sixteen of the30baselines miss?
5.1 EXPERIMENTALSETUP
Datasets and Evaluation Metrics.OmniKE-Bench (§3) holds1,829items over3,960real files of
ten personas in five modalities. Scoring calls no model: set F1 forT 1,T3andT 4, accuracy forT 2and
T5, balanced accuracy forT 6, and OKE averages the ten persona scores (detailed in Appendix D).
6

Preprint.
Baselines and Ablations.Thirty baselines in the ten groups of Table 1 span every line of §2, from
NaiveGeneration, the frozen model with no corpus, through chunk, graph and hypergraph RAG to
memory agents, agentic KE and omnimodal KE; each is re-implemented from its code or paper with
its prompts and defaults, and all attempt the six tasks. The ablations w/o C1 (no hypergraph), w/o C2
(no lifecycle) and w/o C3 (flat retrieval) each remove one component of §4 (detailed in Appendix E).
Implementation Details.All methods share one frozen omnimodal modelΩ(GEMINI2.5 FLASH
LITE; Google, 2025), encoderϕ(GEMINIEMBEDDING2,3,072dimensions) and the answer prompt
pans, so methods differ only in the evidence they supply; the omnimodal group and KnowledgeClaw
read raw media natively, all other methods a shared media-to-text bridge. KnowledgeClaw uses
θ= 0.86,λ= 4,k= 10and64communities for every persona and task (detailed in Appendix E.6).
5.2 MAINRESULTS(RQ1)
As shown in Table 1, KnowledgeClaw ranks first on all six tasks and all ten personas with55.9
OKE,+14.6over the strongest baseline, MemPro. Figure 5a scales each task to its best method,
which is KnowledgeClaw on all six, and in Figure 5c it beats the strongest baseline of every persona
by+4.7to+21.1(two-sided sign test,p= 0.002). The five strongest baselines have a lifecycle
(MemPro, OKF, LLM Wiki, Graphiti) or native media reading (Omni-SimpleMem), never both, and
none exceeds41.3OKE: the three properties of OmniKE pay off only when they are all combined.
5.3 ABLATIONSTUDY(RQ2)
Every component is necessary, and each costs most where it acts (Figure 5b): without C1,T 1and
T4drop most, as only bridged text segments remain; without C2, onlyT 5falls, as the superseded
version stays; without C3, flat top-kloses on all six tasks. Without C1 or C2, OKE still stays above
every baseline (49.1and50.4against MemPro’s41.3), so the three components are complementary.
KnowledgeClaw MemPro OKF LLM Wiki Omni-SimpleMem strongest baseline
0 50 100
% of the best per taskT6T5T4T3T2T1 52.7
94.6
85.7
82.6
82.4
91.5
(a) Share of the best per task
w/o C1 w/o C2 w/o C3
T1
T2
T3
T4
T5
T6
OKE-20.0 0-35.9
-1.6 0-48.0
-4.8 0 -7.8
-13.6 0-15.4
-0.3 -46.0 -32.8
-1.4 0-18.5
-6.8 -5.5 -29.2 (b) Ablation minus full model
HC
Fin
Edu
Law
PsyPol
+4.7HumBldMus
+21.1Gam
 (c) Per persona
Figure 5:RQ1–RQ2.(a) Right: best baseline. (b) Table 1. (c) Task mean; blue: extreme leads.
5.4 HYPERGRAPHANALYSIS(RQ3)
KnowledgeClaw w/o C1 best text baseline best omnimodal baseline median range
0 50 100
ScoreT6 imageT4 img+PDFT3 PDFT2 imageT1 imageT1 PDF +25.6
+13.9
+4.3
+6.4
+3.6
+4.6
(a) Per-modality slices
2 3 4
Hops01020Set F1
19.3
13.5
7.1 (b) Join depth
TextPDFImgAudPDF
Img
Aud
Vid8.2k
179204
21820696
216208146214 (c) Cross-modal hyperedges
Figure 6:C1 analysis.(a) Label: gain over the best baseline. (b)T 3andT 4joins. (c) All personas.
Typed records lift every modality and hop.KnowledgeClaw leads the best text-bridge and omni-
modal baselines on all six modality slices, most onT 1PDFs and images (Figure 6a). It also leads at
every join depth, while the median score across the30baselines falls from7.5to0.0(Figure 6b).
Joins cross every modality.C1 reads all216image, audio and video files of each persona natively
(Table 12). Each cell of Figure 6c counts, over the ten personas, the hyperedges joining its two
modalities: every pair is joined, from96image–audio to8.2k text–PDF;95.4–99.2%of files reach
another file in two hyperedges (Proposition 1), and the largest join gains fall on shared entities
(+6.0) and file-name anchors (+4.9; Table 11), where similarity on its own has nothing to match.
7

Preprint.
5.5 LIFECYCLEANALYSIS(RQ4)
KnowledgeClaw MemPro OKF w/o C2 lifecycle baseline static baseline
0 50 100
Phase A050100Phase B93.4, 97.8
77.4, 80.3
93.4, 5.8 A = B
(a) Before vs. after the edit
0 50 100
Share of answers (%)KnowledgeClaw
MemPro
OKF
Graphiti
w/o C2current
superseded2.2
19.0
65.0
61.3
94.2 (b) Phase-B answers
1 100 200
Edits per persona0510Stale (k)10,148
0 (c) Edit stream
Figure 7:C2 analysis.(a, b)T 5accuracy and answers. (c) Nodes without a present file, all personas.
Each of the137T 5items is answered before (phase A) and after (phase B) its edit is committed,
and Figure 7c streams2,000random edits.Current after every edit.KnowledgeClaw scores93.4
before and97.8after the edit, while eight of the nine baselines above70before it fall below36after
it (Figure 7a); it returns the superseded value for only2.2%, far less than every baseline (Figure 7b).
Sound and cheap under a long edit stream.Over2,000edits, Eq. (7) never fails and no node
without a present file survives (Proposition 2), while an append-only store accumulates10,148such
nodes (Figure 7c). An edit touches a median0.33%of the hypergraph and a REFRESHtakes2.6s,
against118s to ADDthe whole corpus, so each edit costs far less than a full rebuild of the corpus.
5.6 RETRIEVALANALYSIS(RQ5)
KnowledgeClaw w/o C3 (flat top-k)
0 50 100
Named file found (%)T6T5T4T3T2T1 93.8
66.5
85.4
23.1
63.5
86.6100
100
100
100
100
100
(a) Named file found
T1T2T3T4T5T61001k10kMedian tokens1.9×9.1×
1.9×2.4×
1.8×10.5× (b) Evidence length
FileWalkCom.FlatOurs
T1
T2
T3
T4
T5
T6
OKE51.636.14.115.751.6
82.881.527.834.882.8
1.515.81.28.015.8
3.620.90.05.520.9
95.695.328.862.895.6
76.072.654.257.576.0
52.251.215.826.755.9 (c) Forced granularity
Figure 8:C3 analysis.(a) Items with all named files in the evidence. (b) Per item. (c) Bold: best.
A file isnamedwhen the question carries its handle. Figure 8c forces one granularity on every
query: anchor cards (File), the walk, community reports (Com.) or flat top-k.Named files arrive,
in less evidence.By Proposition 3, C3 brings every named file’s card into the evidence, where flat
top-kmisses it on6–77%(Figure 8a), with1.8–10.5×shorter evidence for each item (Figure 8b).
Routing matches the best fixed granularity.Each forced granularity fails on some task, file only
on the joins (1.5and3.6) and walking onT 1(36.1). The router equals the best fixed one on all six
tasks and reaches55.9OKE, above File (52.2), Walk (51.2), Flat (26.7) and Com. (15.8) (Figure 8c).
5.7 EFFICIENCY ANDROBUSTNESS(RQ6–RQ7)
KnowledgeClaw MemPro OKF w/o C3
$0.0001 $0.001 $0.01
USD per item (log)204060OKE55.941.3
(a) Query cost
256 512 1k 2k full
Tokens204060OKE
MemPro 41.3
OKF 38.551.255.9 (b) Evidence budget
Figure 9:Efficiency (RQ6).(a) Dashed: the best OKE
at or below each cost. (b) Dashed: MemPro and OKF.Figure 9a places each method (grey: other
baselines) by its query cost (Table 9) and
OKE, and Figure 9b cuts the evidence of
each item to its firstBtokens and answers
it again.The best score at a low cost.
KnowledgeClaw spends$0.32per persona
to ingest,$0.0004per query and3minutes
to build (Table 9), less than MemPro on all
three, by27.8×per query (see Figure 9a).
Short evidence is enough.Cut to the first
256tokens, its evidence still yields51.2
OKE, above every baseline even at full
context, and512tokens give55.0, while
8

Preprint.
flat top-kstays within20.6–26.7at every budget (Figure 9b); the routed evidence is also shorter
than flat top-k’s and scores+7.8to+48.0points higher, on every task (Table 14 in Appendix F.5).
θ keep linksk0204060Score
OKE 55.8–56.3
T3+T4 16.3–19.8MemPro 41.3
12.4
Figure 10:Robustness (RQ7).One
hyperparameter varied; star: default.Figure 10 varies one hyperparameter at a time around its
default (star): the anchor thresholdθ, the keep ratio, the
cap on linked works andk.Stable across settings.Over its
15further settings, OKE stays within55.8–56.3andT 3+T4
within16.3–19.8, always above MemPro’s41.3and12.4
(Figure 10): no setting has to be tuned per persona or per
task to keep KnowledgeClaw ahead of every baseline. No
setting changesT 1,T2,T5orT6, and OKE moves by at most
0.4from the default (full sweep in Table 15), so none of the
defaults is tuned to any task, and none of them needs to be.
5.8 MATCHEDINPUTS ANDSCORERS(RQ8)
Table 2:Matched inputs (RQ8). +H: the ingested text
of each named file is put before the baseline’s own ev-
idence;+H+R: KnowledgeClaw’s record card instead.
BaselineT 5(the lifecycle, C2) is not rerun. Bold: best.
MethodT 1T2T3T4T5T6OKE
KnowledgeClaw 51.6 82.8 15.8 20.9 95.6 76.0 55.9
MemPro 11.4 78.2 13.5 8.2 78.8 69.5 41.3
+ H 28.3 78.8 13.2 7.3 78.8 73.8 45.9
+ H + R 45.2 81.0 13.5 7.3 78.8 75.8 50.7
OKF 27.2 68.5 8.8 17.3 54.4 67.3 38.5
+ H + R 43.7 78.2 8.8 17.3 54.4 72.6 45.6
StandardRAG 6.1 27.2 9.5 12.7 51.8 55.7 21.0
+ H + R 43.1 76.5 10.0 10.9 51.8 71.8 44.4We give the strongest baselines the in-
puts KnowledgeClaw uses (Table 2) and
rescore every method (Figure 11), to test
whether its lead comes from the system
or from the evaluation. Builds and re-
trieval are replayed, so only the evidence
changes, and every answer goes through
the same answerer and scorer as in Ta-
ble 1 (the full tables are in Appendix F.7).
Ahead with matched inputs.Given
the same handle lookup and Knowledge-
Claw’s record cards, MemPro rises from
41.3to50.7OKE, still5.2below Knowl-
edgeClaw, which stays first on all six
tasks (Table 2); OKF and StandardRAG
reach only45.6and44.4. The records
give the largest lift, onT 1, where they raise MemPro from11.4to45.2and StandardRAG from
6.1to43.1: the typed catalogue of C1 is what the baselines lack, yet none reaches Knowledge-
Claw’s51.6, which also reads the anchor file natively and routes the query. Figure 11a plots its lead
over each baseline given +H+R, remaining where lookup and records cannot help, on the joins (T 4
20.9vs. at most17.3) and the lifecycle (T 595.6vs.78.8), which only the walk and commits supply.
KnowledgeClaw MemPro OKF other baseline ablation
T1 T4 T5 (stored)0204060Lead of
KnowledgeClaw
6.4
13.6
16.87.9
3.6
41.28.5
10.0
43.8
(a) Lead over +H+R
20 40 60
Lenient OKE204060Strict OKE41.3, 37.955.9, 54.6
strict = lenientKendall τ = 0.90 (b) Strict vs. lenient scorer
T1T3T4All050100Agreement (%)95.196.098.095.7agree (c) Matcher vs. LLM judge
Figure 11:Matched inputs and scorers.(a) KnowledgeClaw minus each baseline given +H+R;
grey: StandardRAG; light: storedT 5(not rerun). (b) All34rows of Table 1. (c) Cohen’sκ= 0.91.
Independent of the scorer, agreed by a judge.Both checks reuse the stored predictions of Table 1,
so no answer is regenerated. Each point of Figure 11b is one row of Table 1 at its lenient and strict
OKE (diagonal: equal scores). KnowledgeClaw sits top right: under exact matching it stays first
on every task, its margin grows from14.6to16.7OKE, and the two scorers agree closely on the
ranking of all34rows (Kendallτ b= 0.90; per-task scores in Table 17). In Figure 11c, on sampled
answers, the matcher and an LLM judge agree on95.1,96.0and98.0%of the decisions onT 1,T3
andT 4(95.7%of all decisions; Table 18), and the deterministic matcher is the stricter of the two.
Of the values that C1 copies into records,97.8%occur in the source text that it read (Appendix F.7).
9

Preprint.
Table 3:Case studyon itemT2 building 3261e3912859. Frames, waveform and transcript
come from the files, answers are saved predictions, and the last two rows follow their saved traces.
Query(T 2, building).Which candidate description is the authoritative description of the video
f75bfdd876eb3d22.mp4?The four candidates open four lectures of one MIT class by the same speaker
and differ only in the topic: (A) transportation systems, (B) the grid, (C) solar, (D) a just energy transition.
Gold:(D).
0:01 0:06 0:26 1:06 1:46
audio
[0:00] “…talk about cities and a just energy transition”sibling lectures in the corpus
(A)
transport(B)
grid(C)
solar
all four by David Hsu, one MIT class
Method How the video reaches the answerer, and what is retrieved Answer
NaiveGeneration No corpus: the query names only a hashed file name, which carries no
topic.(C) solar✗
StandardRAG Text bridge of the video (30s windows) cut into1,200-token chunks;
top-5by similarity to the query.(B) grid✗
LightRAG Entity-relation graph from the text bridge; query keywords retrieve en-
tities, relations and their chunks.(A) transport✗
HippoRAG 2 Graph of triples from the text bridge; personalized PageRank from
nodes matched to the query.(A) transport✗
HyperGraphRAGn-ary hyperedges from the text bridge; query entities and segments re-
trieve hyperedges and chunks.(B) grid✗
Mem0 Memories distilled from the text bridge; recalled by similarity to the
query.(A) transport✗
Cognee Knowledge graph and chunk summaries from the text bridge; the top-5
triplets nearest to the query.(A) transport✗
MMGraphRAG Text graph over the bridged transcript (raw media only for images);
top-5nodes by similarity.(A) transport✗
w/o C3 Our hypergraph with flat top-k: ten hyperedges, five of them identifiers
of other videos, and six passages, none of them from the named video
(10.5k characters).(B) grid✗
KnowledgeClaw
Handle in q
f75bf….mp4C3: Anchorfile in HqCard read natively by C1
title: Cities and a Just Energy
Transition; person: T. Roosevelt
evidence: this card alone, 3.7k characters (D) just
transition✓
5.9 CASESTUDY(RQ9)
Table 3 traces an item whose video is one of four sibling lectures with the same speaker, class and
near-identical openings, which similarity to a query naming only a file cannot separate: sixteen of
the30baselines choose a sibling’s description. KnowledgeClaw resolves the handle to the anchor
file and answers from its card alone, a record that C1 read from the raw video, so the topic, the one
field in which the four candidates differ, is in the evidence, and it chooses the gold description (D)
from2.9×less evidence than w/o C3 (3.7k vs.10.5k characters), all of it from the named video.
This case isolates lookup (C3) and native reading (C1); Tables 1 and 2 measure the overall gain.
6 CONCLUSION
We formalized Omnimodal Knowledge Engineering (OmniKE), which asks a live knowledge base
for omnimodality, a closed-loop lifecycle and multi-granular application at once, and released
OmniKE-Bench, six model-free tasks over the real files of ten personas in five modalities. Our
KnowledgeClaw reads every file natively into an omnimodal typed hypergraph (C1), keeps it sound
under edits with a closed-loop lifecycle agent (C2), and routes each query to the granularity it needs
(C3). Against30baselines it ranks first on all six tasks and all ten personas and stays current after
every edit. Given its handle lookup and records, the strongest baseline still trails it by5.2OKE.
10

Preprint.
AIUSE STATEMENT
Large language models enter this work only as studied components: the frozen backbone shared by
all evaluated systems (§5.1) and the stem paraphraser of OmniKE-Bench (§3), which never writes a
gold answer. Beyond these roles, they were used only to polish language and to assist with coding.
The authors conceived the research, designed and ran every experiment, drew every conclusion, and
reviewed, tested and verified all AI-assisted text and code, for which they take full responsibility.
ETHICS STATEMENT
OmniKE-Bench is built only from public sources that permit redistribution (Table 7): public-domain
government records, open-access papers, and open museum, media and course collections. Every
released file keeps its original licence, including the non-commercial terms of some sources. The
benchmark involves no human subjects and collects no private data: its ten personas are assembled
from public material, speech and video come only from public proceedings and openly licensed
recordings, and every file is stripped of metadata, original names and source URLs. No gold answer
is produced by a model, so it adds no model-generated misinformation. OmniKE-Bench is intended
for research use; systems that, like KnowledgeClaw, index personal files should run only on data
their users own or are entitled to process, and every use must honour the licence of each source file.
REPRODUCIBILITY STATEMENT
All material needed to reproduce our results is given in the appendices. Appendix A lists every
prompt of KnowledgeClaw verbatim with its inputs and output limits; Appendix B proves the three
propositions of §4; Appendix C gives the full algorithm of construction, lifecycle commits and
retrieval. Appendix D specifies the six tasks, the seven-stage construction of OmniKE-Bench from
its licensed sources and the model-free metrics; a SHA-256 manifest seals the release, and one
command rebuilds every item from the raw files. Appendix E fixes the shared deployment protocol
(one frozen model and encoder, temperature0by default, cached calls) and the configuration of
every baseline, KnowledgeClaw and its ablations. Appendix F reports the evaluation protocol, cost,
and the setting and full results of every analysis, all recomputed from saved predictions and traces.
REFERENCES
Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammad-
khani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, and Ehsaneddin
Asgari. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented gen-
eration. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar
(eds.),Findings of the Association for Computational Linguistics: ACL 2025, pp. 16776–16809,
Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-
5. doi: 10.18653/v1/2025.findings-acl.861. URLhttps://aclanthology.org/2025.
findings-acl.861/.
Zafar Ali, Asad Khan, Aalia Malik, and Pavlos Kefalas. Graphloom: Reliability-calibrated graph
evidence routing for multimodal kg-rag, 2026. URLhttps://arxiv.org/abs/2608.
15056.
Nguyen Phu An, Dung Nguyen Quang, Luu Hieu An, Linh Ngo Van, Trung Le, and Thien Huu
Nguyen. HyperProve: Answer-guided hypergraph expansion for multi-hop question answer-
ing. In Hung-Yi Lee, Sunipa Dev, Jessy Li, and Desmond Elliott (eds.),Proceedings of the
2026 Conference on Empirical Methods in Natural Language Processing, pp. 4867–4882, Bu-
dapest, Hungary, October 2026. Association for Computational Linguistics. ISBN 979-8-89176-
459-0. doi: 10.18653/2026.emnlp-main.212. URLhttps://aclanthology.org/2026.
emnlp-main.212/.
Weidong Bao, Yilin Wang, Ruyu Gao, Fangling Leng, Yubin Bao, and Ge Yu. Dial-kg: Schema-free
incremental knowledge graph construction via dynamic schema induction and evolution-intent
assessment. In Hyungsoo Jung, Tianzheng Wang, Masashi Toyoda, Hyuk-Yoon Kwon, and Jae-
woong Lee (eds.),Database Systems for Advanced Applications, pp. 357–372, Singapore, 2026.
Springer Nature Singapore. ISBN 978-981-92-0378-9.
11

Preprint.
Jiacheng Cai, Zijin Hong, Zheng Yuan, Huachi Zhou, Qinggang Zhang, and Xiao Huang. Cross-
modalqa: A cross-modal and multi-hop benchmark for multimodal retrieval-augmented genera-
tion, 2026. URLhttps://arxiv.org/abs/2609.05518.
Shenao Chen, Yidan Xu, Xiangmin Han, Rundong Xue, Duanpo Wu, Yuhan Gao, Chenggang Yan,
and Yue Gao. Hypergraph-based multimodal retrieval-augmented generation with incremental
refinement, 2026. URLhttps://arxiv.org/abs/2608.16628.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav.Mem0: Build-
ing Production-Ready AI Agents with Scalable Long-Term Memory. IOS Press, October 2025.
ISBN 9781643686318. doi: 10.3233/faia251160. URLhttp://dx.doi.org/10.3233/
faia251160.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-modal
retrieval is what you need for multi-page multi-document understanding, 2024. URLhttps:
//arxiv.org/abs/2411.04952.
Yuzhuo Cui, Zongye Zhang, and Qingjie Liu. LivingRAG: Augmenting graph RAG with ex-
perience. In Hung-Yi Lee, Sunipa Dev, Jessy Li, and Desmond Elliott (eds.),Findings of
the Association for Computational Linguistics: EMNLP 2026, pp. 33604–33628, Budapest,
Hungary, October 2026a. Association for Computational Linguistics. ISBN 979-8-89176-462-
0. doi: 10.18653/2026.findings-emnlp.1578. URLhttps://aclanthology.org/2026.
findings-emnlp.1578/.
Zeyang Cui, Jiannong Cao, Zhiyuan Wen, Bo Yuan, Junlan Feng, and Shengyuan Chen. Edge-
mem: Llm-free agent memory construction and retrieval via evidence-preserving multi-anchor
hypergraph, 2026b. URLhttps://arxiv.org/abs/2609.05553.
Sijun Dai, Qiang Huang, Xiaoxing You, and Jun Yu. MG2-RAG: Multi-granularity graph for mul-
timodal retrieval-augmented generation. In Paolo Favaro, Zuzana Kukelova, Atsuto Maki, Anna
Rohrbach, Konrad Schindler, and Federico Tombari (eds.),Computer Vision – ECCV 2026, pp.
554–577, Cham, 2026. Springer Nature Switzerland. ISBN 978-3-032-37167-6.
Junnan Dong, Siyu An, Yifei Yu, QIANWEN ZHANG, Linhao Luo, Xiao Huang, di yin, Yunsheng
Wu, and Xing Sun. Youtu-graphrag: Vertically unified agents for graph retrieval-augmented
complex reasoning. In C. V ondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust
(eds.),International Conference on Learning Representations, volume 2026, pp. 143735–143755,
2026a. URLhttps://proceedings.iclr.cc/paper_files/paper/2026/file/
e8618f2038b0209705e4d5e3cd496e1d-Paper-Conference.pdf.
Junnan Dong, Linhao Luo, Senlei Zhang, Gong Chen, Taian Guo, Yifei Yu, Rong Tao, Tao Guo,
Qian-Wen Zhang, Siyu An, Ruizhi Qiao, and Xing Sun. Wfm: Wiki foundation model for com-
plex agentic reasoning, 2026b. URLhttps://arxiv.org/abs/2609.18182.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Tru-
itt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global:
A graph rag approach to query-focused summarization, 2025. URLhttps://arxiv.org/
abs/2404.16130.
Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin
Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang.
Lightmem: Lightweight and efficient memory-augmented generation. In C. V on-
drick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.),Inter-
national Conference on Learning Representations, volume 2026, pp. 98706–98729,
2026. URLhttps://proceedings.iclr.cc/paper_files/paper/2026/file/
a05b72653ec5b473732129829ae04195-Paper-Conference.pdf.
Yichao Feng, Ran Zhang, Haoran Luo, Zhenghong Lin, Carl Yang, and Anh Tuan Luu. Diachronic
hypergraphs for orchestrated multi-agent multimodal memory curation, 2026. URLhttps:
//arxiv.org/abs/2608.29678.
12

Preprint.
Zhenbo Fu, Yuanzhe Zhang, Qiange Wang, Hao Yuan, Yuehao Xu, Enze Yi, Yanfeng Zhang, and
Ge Yu. Evorag: Making knowledge graph-based rag automatically evolve through feedback back-
propagation, 2026. URLhttps://arxiv.org/abs/2604.15676.
Xian Gao, Jinpeng Wang, Jiacheng Ruan, Guangyu Cao, Ting Liu, and Yuzhuo Fu. Memo:
Multimodal evidence memory organization for long-horizon llm agents, 2026. URLhttps:
//arxiv.org/abs/2609.07471.
Google. Gemini 2.5 Flash-Lite. Official model documentation, 2025. URLhttps:
//ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite. Ac-
cessed September 22, 2026.
Google Cloud. Open Knowledge Format. GitHub repository, 2026. URLhttps://github.
com/GoogleCloudPlatform/open-knowledge-format. Accessed September 22,
2026.
Graphify contributors. Graphify. GitHub repository, 2026. URLhttps://github.com/
Graphify-Labs/graphify. Accessed September 22, 2026.
Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. Rag-anything: All-in-one rag
framework, 2025a. URLhttps://arxiv.org/abs/2510.12323.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. LightRAG: Simple and fast
retrieval-augmented generation. In Christos Christodoulopoulos, Tanmoy Chakraborty, Car-
olyn Rose, and Violet Peng (eds.),Findings of the Association for Computational Linguistics:
EMNLP 2025, pp. 10746–10761, Suzhou, China, November 2025b. Association for Computa-
tional Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.568. URL
https://aclanthology.org/2025.findings-emnlp.568/.
Bernal Jim ´enez Guti ´errez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From RAG to memory:
Non-parametric continual learning for large language models. In Aarti Singh, Maryam Fazel,
Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry
Zhu (eds.),Proceedings of the 42nd International Conference on Machine Learning, volume 267
ofProceedings of Machine Learning Research, pp. 21497–21515. PMLR, 13–19 Jul 2025. URL
https://proceedings.mlr.press/v267/gutierrez25a.html.
Chengrui Han and Zesheng Cheng. Raga: Reading-and-graph-building-agent for autonomous
knowledge graph construction and retrieval-augmented generation, 2026. URLhttps://
arxiv.org/abs/2605.17072.
Xin He, Yili Wang, Wenqi Fan, Qing Li, Qinggang Zhang, Yi Chang, and Xin Wang. Hvm-
graphrag: Holistic-view multimodal graph retrieval-augmented generation on complex document,
2026. URLhttps://arxiv.org/abs/2607.24861.
Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, and Chu-song Chen.
MegaRAG: Multimodal knowledge graph-based retrieval augmented generation. In Maria Li-
akata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.),Proceedings of the 64th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 48031–48059, San Diego, California, United States, July 2026. Association for Compu-
tational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.2218. URL
https://aclanthology.org/2026.acl-long.2218/.
Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue, Xingliang Hou, Zhiqiang Tian, Yue Gao, and Shaoyi
Du. Cog-rag: Cognitive-inspired dual-hypergraph with theme alignment retrieval-augmented
generation.Proceedings of the AAAI Conference on Artificial Intelligence, 40(37):31032–31040,
March 2026a. ISSN 2159-5399. doi: 10.1609/aaai.v40i37.40363. URLhttp://dx.doi.
org/10.1609/aaai.v40i37.40363.
Ruofan Hu, Shengyang Xu, Minjie Hong, Xiaoda Yang, Sashuai Zhou, Ke Lei, Tao Jin, and
Zhou Zhao. Doc-REFRAG: Rethinking multimodal document retrieval-augmented generation.
In Hung-Yi Lee, Sunipa Dev, Jessy Li, and Desmond Elliott (eds.),Proceedings of the 2026
Conference on Empirical Methods in Natural Language Processing, pp. 8624–8644, Budapest,
13

Preprint.
Hungary, October 2026b. Association for Computational Linguistics. ISBN 979-8-89176-459-
0. doi: 10.18653/2026.emnlp-main.377. URLhttps://aclanthology.org/2026.
emnlp-main.377/.
Xia Hu, Honglei Zhuang, Brian Potetz, Alireza Fathi, Bo Hu, Babak Samari, and Howard Zhou.
Life-bench: A benchmark and knowledge graph framework for multimodal personalization be-
yond concept recognition, 2026c. URLhttps://arxiv.org/abs/2602.19001.
Rizhuo Huang, Yifan Feng, Rundong Xue, Shihui Ying, Jun-Hai Yong, Chuan Shi, Shaoyi Du, and
Yue Gao. Hyper-kggen: A skill-driven knowledge extractor for high-quality knowledge hyper-
graph generation. InProceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery
and Data Mining V .2, pp. 1898–1909. ACM, August 2026. doi: 10.1145/3770855.3818198. URL
http://dx.doi.org/10.1145/3770855.3818198.
Jingbo Ji, Lingyi Li, Xilong Cheng, Yuhao Zhou, Wenji Zhang, Yuting Tan, and Yunxiao Qin.
Ripplemem: From isolated retrieval to associative recollection for long-term agent memory, 2026.
URLhttps://arxiv.org/abs/2608.13334.
Dongming Jiang, Yi Li, Guanpeng Li, and Bingzhe Li. MAGMA: A multi-graph based agentic
memory architecture for AI agents. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David
Jurgens (eds.),Proceedings of the 64th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 36848–36865, San Diego, California, United States,
July 2026. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/
v1/2026.acl-long.1709. URLhttps://aclanthology.org/2026.acl-long.1709/.
Andrej Karpathy. LLM Wiki. GitHub Gist, 2026. URLhttps://gist.github.com/
karpathy/442a6bf555914893e9891c11519de94f. Accessed September 22, 2026.
Houda Khrouf, Pedro Fillastre, and Sebastiao Correia. Optimizing hypergraph-based rag: Toward
better fact extraction and chunk retrieval, 2026. URLhttps://arxiv.org/abs/2607.
20506.
Yassir Lairgi, Ludovic Moncla, R ´emy Cazabet, Khalid Benabdeslem, and Pierre Cl ´eau. itext2kg:
Incremental knowledge graphs construction using large language models. In Mahmoud Barhamgi,
Hua Wang, and Xin Wang (eds.),Web Information Systems Engineering – WISE 2024, pp. 214–
229, Singapore, 2025. Springer Nature Singapore. ISBN 978-981-96-0573-6.
Yassir Lairgi, Ludovic Moncla, Khalid Benabdeslem, R ´emy Cazabet, and Pierre Cl ´eau. ATOM:
AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs. In Vera
Demberg, Kentaro Inui, and Llu ´ıs Marquez (eds.),Findings of the Association for Computational
Linguistics: EACL 2026, pp. 950–966, Rabat, Morocco, March 2026. Association for Compu-
tational Linguistics. ISBN 979-8-89176-386-9. doi: 10.18653/v1/2026.findings-eacl.49. URL
https://aclanthology.org/2026.findings-eacl.49/.
Christopher Latimer, Nicol `o Boschi, Andrew Neeser, Chris Bartholomew, Gaurav Srivastava, Xuan
Wang, and Naren Ramakrishnan. Hindsight: Structured agent memory that retains, recalls, and
reflects. In Greg Durrett and Ping Jian (eds.),Proceedings of the 64th Annual Meeting of the
Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 275–285, San
Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN
979-8-89176-392-0. doi: 10.18653/v1/2026.acl-demo.27. URLhttps://aclanthology.
org/2026.acl-demo.27/.
EunKyeong Lee, Kyeong-Jin Oh, Jinwon Kim, Hye Woo Lee, Minsang Song, Hyeongjun Jang, and
Junyoung Youn. Mosaic: Query-aware exploration policy adaptation for graphrag, 2026. URL
https://arxiv.org/abs/2609.11065.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In
H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.),Advances in Neu-
ral Information Processing Systems, volume 33, pp. 9459–9474. Curran Associates, Inc.,
2020. URLhttps://proceedings.neurips.cc/paper_files/paper/2020/
file/6b493230205f780e1bc26945df7481e5-Paper.pdf.
14

Preprint.
Geng Li, Yuhao Wang, Dong Li, Jianye Hao, and Yuxin Peng. Graphmemix: Query-aware evidence
forests for long-term multimodal agent memory, 2026a. URLhttps://arxiv.org/abs/
2608.26983.
Jiazheng Li, Chi-Hao Wu, Yunze Liu, Kaize Ding, Jundong Li, and Chuxu Zhang. Bridging modal-
ities, spanning time: Structured memory for ultra-long agentic video reasoning, 2026b. URL
https://arxiv.org/abs/2605.08271.
Yitao Li, Zhanlin Liu, Anuranjan Pandey, and Muni Srikanth. Ontokg: Ontology-oriented knowl-
edge graph construction with intrinsic-relational routing, 2026c. URLhttps://arxiv.org/
abs/2604.02618.
Zhaoyang Li, Zenghuang Fu, Qiuyuan Ai, Ping Jiang, Haoyu Wu, Minghui Wu, Chenxu Zhao,
Jie Song, and Guannan He. Coevokg: Co-evolving knowledge graphs with self-evolving search
agents, 2026d. URLhttps://arxiv.org/abs/2608.01904.
Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, Boyu Chen, Shichao Song, Simin Niu, Hanyu
Wang, Jiawei Yang, Chen Tang, Qingchen Yu, Jihao Zhao, Yezhaohui Wang, Peng Liu, Zehao
Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhen Tao, Huayi Lai,
Hao Wu, Bo Tang, Zhengren Wang, Zhaoxin Fan, Ningyu Zhang, Linfeng Zhang, Junchi Yan,
Mingchuan Yang, Tong Xu, Wei Xu, Huajun Chen, Haofen Wang, Hongkang Yang, Wentao
Zhang, Zhi-Qin John Xu, Siheng Chen, and Feiyu Xiong. Memos: A memory os for ai system,
2025. URLhttps://arxiv.org/abs/2507.03724.
Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao, Mengshu Sun,
Zhiqiang Zhang, Jun Zhou, Wenguang Chen, Wen Zhang, and Huajun Chen. Kag: Boosting
llms in professional domains via knowledge augmented generation. InCompanion Proceedings
of the ACM on Web Conference 2025, WWW ’25, pp. 334–343. ACM, May 2025. doi: 10.1145/
3701716.3715240. URLhttp://dx.doi.org/10.1145/3701716.3715240.
Wen-Sheng Lien, Yu-Kai Chan, Hao-Lung Hsiao, Bo-Kai Ruan, Meng-Fen Chiang, Chien-An Chen,
Yi-Ren Yeh, and Hong-Han Shuai. Hyperrag: Reasoning n-ary facts over hypergraphs for re-
trieval augmented generation. InProceedings of the ACM Web Conference 2026, pp. 2465–2476.
ACM, April 2026. doi: 10.1145/3774904.3792710. URLhttp://dx.doi.org/10.1145/
3774904.3792710.
Jiashi Lin, Changhong Jiang, Xiangru Lin, Ruifei Zhang, Xinyi Zhu, Jiyao Liu, Cheng Tang, Ye Du,
Shujian Gao, Junzhi Ning, Lihao Liu, Ziyan Huang, Tianbin Li, Jin Ye, and Junjun He. Evograph-
r1: Self-evolving multimodal knowledge hypergraphs for agentic retrieval. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 756–765, June
2026.
Jiaqi Liu, Zipeng Ling, Shi Qiu, Yanqing Liu, Siwei Han, Peng Xia, Haoqin Tu, Zeyu Zheng, Cihang
Xie, Charles Fleming, Mingyu Ding, and Huaxiu Yao. Omni-simplemem: Autoresearch-guided
discovery of lifelong multimodal agent memory, 2026a. URLhttps://arxiv.org/abs/
2604.01007.
Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu
Yao. SimpleMem: Efficient lifelong memory for LLM agents. In Tong Zhang, Miroslav Dudik,
Martin Jaggi, Alekh Agarwal, Sharon Li, Dale Schuurmans, Jerry Zhu, Felix Berkenkamp, Hanze
Dong, and Alberto Bietti (eds.),Proceedings of the 43rd International Conference on Machine
Learning, volume 306 ofProceedings of Machine Learning Research, pp. 78500–78513. PMLR,
06–11 Jul 2026b. URLhttps://proceedings.mlr.press/v306/liu26dx.html.
Junming Liu, Siyuan Meng, Yanting Gao, Song Mao, Pinlong Cai, Guohang Yan, Yirong Chen, Zilin
Bian, Ding Wang, and Botian Shi. Aligning vision to language: Annotation-free multimodal
knowledge graph construction for enhanced llms reasoning. InProceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pp. 981–992, October 2025.
Junming Liu, Yifei Sun, Weihua Cheng, Haodong Lei, Yirong Chen, Licheng Wen, Xuemeng Yang,
Daocheng Fu, Pinlong Cai, Nianchen Deng, Yi Yu, Shuyue Hu, Botian Shi, and Ding Wang.
Memverse: Multimodal memory for lifelong learning agents, 2026c. URLhttps://arxiv.
org/abs/2512.03627.
15

Preprint.
Qingshan Liu, Guoqing Wang, Wen Wu, Jingqi Huang, Xinqi Tao, Dejia Song, Jie Zhou, and Liang
He. MemPro: Agentic memory systems as evolvable programs. In Hung-Yi Lee, Sunipa Dev,
Jessy Li, and Desmond Elliott (eds.),Proceedings of the 2026 Conference on Empirical Methods
in Natural Language Processing, pp. 31124–31143, Budapest, Hungary, October 2026d. Associa-
tion for Computational Linguistics. ISBN 979-8-89176-459-0. doi: 10.18653/2026.emnlp-main.
1364. URLhttps://aclanthology.org/2026.emnlp-main.1364/.
Zeyu Liu, Jian Zhong, Rongduo Han, Ziyang Wu, Shunye Tang, Chenghao He, Yaxuan Yang,
Yihang Qiu, Ailing Wang, Xiao Liang, Guohuan Xie, Xiaokang Xue, Gongchen Li, Haining
Zhang, and Wei Wang. EvalMem: An operation-level diagnostic framework for long-term mem-
ory systems. In Hung-Yi Lee, Sunipa Dev, Jessy Li, and Desmond Elliott (eds.),Findings
of the Association for Computational Linguistics: EMNLP 2026, pp. 24002–24034, Budapest,
Hungary, October 2026e. Association for Computational Linguistics. ISBN 979-8-89176-462-
0. doi: 10.18653/2026.findings-emnlp.1134. URLhttps://aclanthology.org/2026.
findings-emnlp.1134/.
Yuxing Lu, Wei Wu, Xukai Zhao, Rui Peng, and Jinzhuo Wang. Karma: Leveraging multi-agent llms
for automated knowledge graph enrichment. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Ko-
niusz, M. Ghassemi, and N. Chen (eds.),Advances in Neural Information Processing Systems,
volume 38, Main Conference, pp. 56421–56453. Curran Associates, Inc., 2025. doi: 10.52202/
085713-1887. URLhttps://proceedings.neurips.cc/paper_files/paper/
2025/file/517f9b9c227b9dd51dba4560f37165ed-Paper-Conference.pdf.
Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika
Lin, Yu Feng, Zemin Kuang, Meina Song, Yifan Zhu, and Anh Tuan Luu. Hyper-
graphrag: Retrieval-augmented generation via hypergraph-structured knowledge represen-
tation. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi,
and N. Chen (eds.),Advances in Neural Information Processing Systems, volume 38,
Main Conference, pp. 152206–152234. Curran Associates, Inc., 2025. doi: 10.52202/
085713-5089. URLhttps://proceedings.neurips.cc/paper_files/paper/
2025/file/df55ee6e59f8ac4a625219e11fe9ddba-Paper-Conference.pdf.
Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina
Song, Xiaobao Wu, Yifan Zhu, and Anh Tuan Luu. Graph-r1: Towards agentic GraphRAG
framework via end-to-end reinforcement learning. In Tong Zhang, Miroslav Dudik, Martin Jaggi,
Alekh Agarwal, Sharon Li, Dale Schuurmans, Jerry Zhu, Felix Berkenkamp, Hanze Dong, and
Alberto Bietti (eds.),Proceedings of the 43rd International Conference on Machine Learning,
volume 306 ofProceedings of Machine Learning Research, pp. 82220–82240. PMLR, 06–11 Jul
2026. URLhttps://proceedings.mlr.press/v306/luo26c.html.
Menghe Ma, Siqing Wei, Yuecheng Xing, Ziyue Zhu, Zhenghong Lin, Yaheng Wang, Fanhong
Meng, Peijun Han, Luu Anh Tuan, and Haoran Luo. Onote: Hypergraph-grounded omnimodal
reasoning for computational music science, 2026a. URLhttps://arxiv.org/abs/2604.
20719.
Xiaokang Ma, Yifan Sun, Zhihong Jin, Jie Gu, Yudong Luo, Shenyi Shao, Chu Tang, Jingmin Chen,
and Li Pu. Reflectworld-mm: An entity-oriented multimodal memory system for open-ended
video streams, 2026b. URLhttps://arxiv.org/abs/2607.09759.
Vasilije Markovic, Lazar Obradovic, Laszlo Hajdu, and Jovan Pavlovic. Optimizing the interface
between knowledge graphs and llms for complex reasoning, 2025. URLhttps://arxiv.
org/abs/2505.24478.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red
Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Moham-
mad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher
Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brock-
man, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann,
Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis,
Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey
16

Preprint.
Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux,
Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila
Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix,
Sim´on Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gib-
son, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan
Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hal-
lacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan
Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu,
Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun
Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Ka-
mali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook
Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel
Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen
Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel
Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez,
Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv
Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney,
Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick,
Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel
Mossing, Tong Mu, Mira Murati, Oleg Murk, David M ´ely, Ashvin Nair, Reiichiro Nakano, Ra-
jeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe,
Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel
Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe
de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny,
Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl,
Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra
Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders,
Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Sel-
sam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor,
Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky,
Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang,
Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Pre-
ston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cer ´on Uribe, Andrea Vallone, Arun Vi-
jayvergiya, Chelsea V oss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan
Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng,
Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Work-
man, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming
Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao
Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL
https://arxiv.org/abs/2303.08774.
OpenClaw contributors. OpenClaw. GitHub repository, 2026. URLhttps://github.com/
openclaw/openclaw. Accessed September 22, 2026.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E.
Gonzalez. Memgpt: Towards llms as operating systems, 2024. URLhttps://arxiv.org/
abs/2310.08560.
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large
language models and knowledge graphs: A roadmap.IEEE Transactions on Knowledge and Data
Engineering, 36(7):3580–3599, 2024. doi: 10.1109/TKDE.2024.3352100.
Hyeongcheol Park, Jiyoung Seo, Jaewon Mun, Hogun Park, Wonmin Byeon, Sung June Kim,
Hyeonsoo Im, JeungSub Lee, and Sangpil Kim. Mˆ3kg-rag: Multi-hop multimodal knowledge
graph-enhanced retrieval-augmented generation. InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 14968–14978, June 2026a.
Jinyoung Park, Sanghyeok Lee, Omar Zia Khan, Hyunwoo J. Kim, and Joo-Kyung Kim. Hyper-
graphpro: Progress-aware reinforcement learning for structure-guided hypergraph rag, 2026b.
URLhttps://arxiv.org/abs/2601.17755.
17

Preprint.
Tithi Rakshit, Hongkuan Zhou, Lavdim Halilaj, and Yuqicheng Zhu. Less is more: Graph-free mul-
timodal RAG via multi-signal late fusion. In Pepa Atanasova, Gholamreza Haffari, and Minghan
Wang (eds.),Proceedings of the 2026 Conference on Empirical Methods in Natural Language
Processing: Industry Track, pp. 1962–1975, Budapest, Hungary, October 2026. Association for
Computational Linguistics. ISBN 979-8-89176-460-6. doi: 10.18653/2026.emnlp-industry.115.
URLhttps://aclanthology.org/2026.emnlp-industry.115/.
Shi-Ju Ran, Kun Zhang, Xi Wu, Liu-Si Yang, and Wen-Jun Li. Llms interpret, embeddings orga-
nize, graphs emerge: Agent-driven compilation of scientific knowledge, 2026. URLhttps:
//arxiv.org/abs/2608.29612.
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A
temporal knowledge graph architecture for agent memory, 2025. URLhttps://arxiv.org/
abs/2501.13956.
Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag:
Retrieval-augmented generation with extreme long-context videos. InProceedings of the 32nd
ACM SIGKDD Conference on Knowledge Discovery and Data Mining V .1, pp. 2390–2401.
ACM, April 2026. doi: 10.1145/3770854.3783944. URLhttp://dx.doi.org/10.1145/
3770854.3783944.
Florian Ruosch and Luca Rossetto. Mediagraph: A content-aware data model and query framework
for multimodal knowledge graphs, 2026. URLhttps://arxiv.org/abs/2608.18895.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher
Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In
B. Kim, Y . Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y . Sun (eds.),In-
ternational Conference on Learning Representations, volume 2024, pp. 32628–32649,
2024. URLhttps://proceedings.iclr.cc/paper_files/paper/2024/file/
8a2acd174940dbca361a6398a4f9df91-Paper-Conference.pdf.
Manuel Serna-Aguilera, Raegan Anderes, Page Dobbs, and Khoa Luu. Nico-rag: Multimodal hyper-
graph retrieval-augmented generation for understanding the nicotine public health crisis, 2026a.
URLhttps://arxiv.org/abs/2603.02047.
Manuel Serna-Aguilera, Raegan Anderes, Page Dobbs, and Khoa Luu. Prism-rag: Multimodal
hypergraph retrieval-augmented generation for tobacco product and legislative policy reasoning,
2026b. URLhttps://arxiv.org/abs/2609.23769.
Yi Ting Shen, Kentaroh Toyoda, and Alex Leung. Revoked but still authoritative: An empirical
study of revocation enforcement in agent-memory systems, 2026. URLhttps://arxiv.
org/abs/2609.08258.
Rudi Studer, V .Richard Benjamins, and Dieter Fensel. Knowledge engineering: Principles and
methods.Data & Knowledge Engineering, 25(1):161–197, 1998. ISSN 0169-023X. doi: https:
//doi.org/10.1016/S0169-023X(97)00056-6. URLhttps://www.sciencedirect.com/
science/article/pii/S0169023X97000566.
Susheel Suresh, Hazel Mak, Sahil Bhatnagar, Chhaya Methani, and Alejandro Gutierrez Munoz.
Grounding agent memory: Environment-probing curation for enterprise agents, 2026. URL
https://arxiv.org/abs/2609.11060.
Yang Tian, Fan Liu, Jingyuan Zhang, Zhenyang Li, Yupeng Hu, and Liqiang Nie. Mm-contextfold:
Context folding for multimodal agentic retrieval, 2026. URLhttps://arxiv.org/abs/
2609.23121.
Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu,
and Yangqiu Song. AutoGraph-R1: End-to-end reinforcement learning for knowledge graph
construction. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.),
Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), pp. 23351–23374, San Diego, California, United States, July 2026. Association
for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.1070.
URLhttps://aclanthology.org/2026.acl-long.1070/.
18

Preprint.
Xueyao Wan and Hang Yu. Mmgraphrag: Bridging vision and language with interpretable multi-
modal knowledge graphs, 2026. URLhttps://arxiv.org/abs/2507.20804.
Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, and Ning Jiang. Cross-granularity hy-
pergraph retrieval-augmented generation for multi-hop question answering.Proceedings of the
AAAI Conference on Artificial Intelligence, 40(39):33368–33376, March 2026. ISSN 2159-5399.
doi: 10.1609/aaai.v40i39.40623. URLhttp://dx.doi.org/10.1609/aaai.v40i39.
40623.
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai
Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large
language model based autonomous agents.Frontiers of Computer Science, 18(6):186345, Mar
2024. ISSN 2095-2236. doi: 10.1007/s11704-024-40231-1. URLhttps://doi.org/10.
1007/s11704-024-40231-1.
Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents, 2025. URL
https://arxiv.org/abs/2507.07957.
Ziming Wang. Toki: A bitemporal operator algebra for contradiction resolution in llm-agent persis-
tent memory, 2026. URLhttps://arxiv.org/abs/2606.06240.
Keshu Wu, Chenchen Kuai, Zihao Li, Jiwan Jiang, Shiyu Shen, Shian Wang, Chan-Wei Hu,
Zhengzhong Tu, and Yang Zhou. Knowledge is not static: Order-aware hypergraph rag for lan-
guage models, 2026a. URLhttps://arxiv.org/abs/2604.12185.
Zongyu Wu, Yilong Wang, Xiaochen Wang, Minhua Lin, Zhichao Xu, Fenglong Ma, Xiang Zhang,
and Suhang Wang. Multi-granularity context-enhanced rag over multimodal knowledge graphs,
2026b. URLhttps://arxiv.org/abs/2608.25986.
Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng
Zheng, Yang Wang, and Enhong Chen. Large language models for generative informa-
tion extraction: a survey.Frontiers of Computer Science, 18(6):186357, Nov 2024. ISSN
2095-2236. doi: 10.1007/s11704-024-40555-y. URLhttps://doi.org/10.1007/
s11704-024-40555-y.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem:
Agentic memory for llm agents. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz,
M. Ghassemi, and N. Chen (eds.),Advances in Neural Information Processing Systems, vol-
ume 38, Main Conference, pp. 17577–17604. Curran Associates, Inc., 2025. doi: 10.52202/
085713-0593. URLhttps://proceedings.neurips.cc/paper_files/paper/
2025/file/19909c36f51abc4856b4560aff3d36d6-Paper-Conference.pdf.
Meriem Yacoubi, Pia Schmidt, Nenad Petrovic, Ahmed Frikha, Martin Kirchhoff, and Alois Knoll.
Memorylace: Memory lifecycle-aware consolidation and evidence retrieval, 2026. URLhttps:
//arxiv.org/abs/2609.03201.
Neeraj Yadav. Temporal validity in retrieval memory: Eliminating stale-fact errors for ai agents over
evolving knowledge, 2026. URLhttps://arxiv.org/abs/2606.26511.
Cehao Yang, Xiaojun Wu, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Yuanliang Sun, Jia Li, Hui
Xiong, and Jian Guo. Graphsearch: An agentic deep searching workflow for graph retrieval-
augmented generation, 2025. URLhttps://arxiv.org/abs/2509.22009.
Chang Yang, Chuang Zhou, Yilin Xiao, Su Dong, Luyao Zhuang, Yujing Zhang, Zhu Wang, Zijin
Hong, Zheng Yuan, Zhishang Xiang, Shengyuan Chen, Huachi Zhou, Qinggang Zhang, Ninghao
Liu, Jinsong Su, Xinrun Wang, Yi Chang, and Xiao Huang. Graph-based agent memory: Taxon-
omy, techniques, and applications, 2026a. URLhttps://arxiv.org/abs/2602.05665.
Kairui Yang, Minghao An, Xunkai Li, Ziheng Yi, Zekai Chen, Guangyuan He, and Rong-Hua Li.
Mace: Memory-agent co-evolution with adaptive memory graphs for multi-agent systems, 2026b.
URLhttps://arxiv.org/abs/2609.21533.
19

Preprint.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan
Cao. React: Synergizing reasoning and acting in language models. InThe Eleventh International
Conference on Learning Representations, 2023. URLhttps://openreview.net/forum?
id=WE_vluYUL-X.
Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, and Sung Ju Hwang. Worldmm: Dynamic multi-
modal memory agent for long video reasoning. InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 25599–25609, June 2026.
Li Yuan, Qingfei Huang, Bingshan Zhu, Yi Cai, Qingbao Huang, Changmeng Zheng, Zikun Deng,
and Tao Wang. Hybrid-dmkg: A hybrid reasoning framework over dynamic multimodal knowl-
edge graphs for multimodal multihop qa with knowledge editing.Proceedings of the AAAI Con-
ference on Artificial Intelligence, 40(33):28032–28040, March 2026a. ISSN 2159-5399. doi: 10.
1609/aaai.v40i33.40028. URLhttp://dx.doi.org/10.1609/aaai.v40i33.40028.
Xu Yuan, Liangbo Ning, Qingqing Ye, Wenqi Fan, and Qing Li. mkg-rag: Leveraging multimodal
knowledge graphs in retrieval-augmented generation for knowledge-intensive vqa. InProceedings
of the 49th International ACM SIGIR Conference on Research and Development in Information
Retrieval, pp. 2274–2285. ACM, 2026b. doi: 10.1145/3805712.3809680. URLhttp://dx.
doi.org/10.1145/3805712.3809680.
Juwei Yue, Chuanrui Hu, Jiawei Sheng, Zuyi Zhou, Wenyuan Zhang, Tingwen Liu, Li Guo, and
Yafeng Deng. HyperMem: Hypergraph memory for long-term conversations. In Maria Li-
akata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.),Proceedings of the 64th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 35237–35254, San Diego, California, United States, July 2026. Association for Compu-
tational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.1627. URL
https://aclanthology.org/2026.acl-long.1627/.
Xiangjun Zai, Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, and Wenjie Zhang. Proh: Dy-
namic planning and reasoning over knowledge hypergraphs for retrieval-augmented generation,
2026. URLhttps://arxiv.org/abs/2510.12434.
Guangzhe Zhang. Correct now, insufficient later: Auditing update sufficiency in context compres-
sion, 2026. URLhttps://arxiv.org/abs/2609.20045.
Guibin Zhang, Haotian Ren, Chong Zhan, Junhao Wang, He Zhu, Wangchunshu Zhou, and
Shuicheng Yan. MemEvolve: Meta-evolution of agent memory systems. In Tong Zhang,
Miroslav Dudik, Martin Jaggi, Alekh Agarwal, Sharon Li, Dale Schuurmans, Jerry Zhu, Fe-
lix Berkenkamp, Hanze Dong, and Alberto Bietti (eds.),Proceedings of the 43rd International
Conference on Machine Learning, volume 306 ofProceedings of Machine Learning Research,
pp. 158215–158234. PMLR, 06–11 Jul 2026. URLhttps://proceedings.mlr.press/
v306/zhang26fa.html.
Suifeng Zhao, Zida Liu, Xinyu Lei, Lei Sun, Jun Gao, and Sujian Li. MCite-RL: Towards reli-
able multimodal RAG via citation-enhanced agentic reinforcement learning. In Hung-Yi Lee,
Sunipa Dev, Jessy Li, and Desmond Elliott (eds.),Proceedings of the 2026 Conference on Em-
pirical Methods in Natural Language Processing, pp. 32172–32187, Budapest, Hungary, October
2026. Association for Computational Linguistics. ISBN 979-8-89176-459-0. doi: 10.18653/2026.
emnlp-main.1408. URLhttps://aclanthology.org/2026.emnlp-main.1408/.
Chulun Zhou, Chunkang Zhang, Guoxin Yu, Fandong Meng, Jie Zhou, Wai Lam, and Mo Yu. HG-
Mem: Hypergraph-based working memory to improve multi-step RAG for long-context complex
relational modeling. In Tong Zhang, Miroslav Dudik, Martin Jaggi, Alekh Agarwal, Sharon
Li, Dale Schuurmans, Jerry Zhu, Felix Berkenkamp, Hanze Dong, and Alberto Bietti (eds.),
Proceedings of the 43rd International Conference on Machine Learning, volume 306 ofPro-
ceedings of Machine Learning Research, pp. 166312–166333. PMLR, 06–11 Jul 2026. URL
https://proceedings.mlr.press/v306/zhou26bk.html.
Yifan Zhu, Xinyu Mu, Tao Feng, Zhonghong Ou, Yuning Gong, and Haoran Luo. Omnirag-agent:
Agentic omnimodal reasoning for low-resource long audio-video question answering, 2026. URL
https://arxiv.org/abs/2602.03707.
20

Preprint.
APPENDIX
A PROMPTSUSED INKNOWLEDGECLAW
This section gives, verbatim, every prompt that KnowledgeClaw sends to the frozen modelΩ
(GEMINI2.5 FLASHLITE); Table 4 places each in §4. In the boxes,{. . .}marks a slot filled
at run time and⟨. . .⟩a raw media part. All calls run at temperature0, in JSON mode for JSON
replies. C1 owns four prompts and C3 three; C2 adds none: DELcalls no model, and ADDand
REFRESHrun C1’s prompts on the added files, so edits and construction share one extraction path.
Table 4: The prompts of KnowledgeClaw;Maxis the output-token limit of the call.
Prompt Used in Input Output Calls Max
pimage C1, Eq. (2) image,≤1,536px description, visible text per image 1,000
paudio C1, Eq. (2) audio,16kHz mono summary, timed transcript per recording 3,000
pvideo C1, Eq. (2)≤6keyframes, audio description, transcript per video 3,000
prec C1, Eq. (3) first9,000chars of˜aJSON record per file 2,500
proute C3, Eq. (8) queryqone label if no rule routesq5
pcom C3, communities entities, propositions report,≤200words per community 400
pans C3, answering evidenceX q, question JSON answer per query 600
A.1 READING PROMPTSp µ(a)
Eq. (2) reads each image, audio and video file once, natively, under the prompt of its modality
(Figure 12). Since the reading˜ais what the record prompt catalogues, each prompt asks for what
tells a file apart from its look-alikes rather than for a caption: the object type, depicted content,
technique and every legible word of an image, with an inventory for a floor plan; the speakers and
a time-marked transcript of a recording; and both, with on-screen text, for a video, sent as up to six
time-labelled keyframes and its audio track. Each reading is then split into segments of about1,000
tokens that keep their file’s modality and together form the passage indexI σthat C3 searches.
Image reading promptp image
Y ou are indexing one image for a personal knowledge base, so that it can later be told apart from similar images and
matched to its catalogue record. Write plain prose (at most 250 words), then one line "Visible text:" with every legible
word verbatim (or "none").
Cover, when present: what kind of object the image is (photograph, photomicrograph, electron micrograph, engraving,
drawing, painting, portrait, map, chart, floor plan, diagram, screenshot, poster, document scan, object photo, ...); what it
depicts -- people (how many, age group, sex, activity, clothing), organisms, anatomy, objects, instruments, buildings,
places, setting and apparent period; technique (stain, magnification, colour or black-and-white).
For a floor plan or technical diagram, add a line "Inventory:" that counts what is drawn, by type: rooms (each room type
with its count), walls, windows, doors, stairs, columns, railings, and every fixed fitting or furniture item (sinks, toilets,
showers, bathtubs, cabinets, closets, stoves, appliances, sauna benches, ...) with its count.
Do not guess names, places or dates that are not visible.
⟨image⟩
Audio reading promptp audio
Y ou are indexing one audio recording for a personal knowledge base. First give a 1-3 sentence summary: what the
recording is (lecture, interview, talk, song, sound effect, ...), who speaks (names if said), and the topic or musical
character (genre, instruments, mood, tempo). Then write "Transcript:" and transcribe all speech verbatim, with a time
mark such as [0:30] about every 30 seconds. If there is no speech, write "Transcript: none".
⟨audio⟩
Video reading promptp video
Y ou are indexing one video for a personal knowledge base. Y ou are given keyframes (with their times) and the audio
track. First describe what the video shows in 2-4 sentences: kind of video (lecture, interview, demonstration, gameplay,
animation, ...), people, setting, and all on-screen text verbatim. Then write "Transcript:" and transcribe all speech
verbatim, with a time mark such as [0:30] about every 30 seconds (or "Transcript: none").
Frame at 0:10:⟨keyframe⟩Frame at 0:30:⟨keyframe⟩. . . Frame at 1:50:⟨keyframe⟩Audio track:⟨audio⟩
Figure 12: Reading promptsp µ(a) of Eq. (2) for image, audio and video files; each prompt is sent
together with the raw media part(s) shown last; the reply is the surface˜aread by the record prompt.
21

Preprint.
A.2 RECORD PROMPTp rec
Eq. (3) writes each file’s record with one call of the prompt in Figure 13 on the first9,000characters
of˜a. The files of all ten personas share one schemaΓof13general cataloguing fields and dated
events. As a join matches forms, not topics,subjectasks for the preferred form of standard vocabu-
laries (arXiv codes, MeSH descriptors and check tags), andelementfor canonical class names.
Record promptp rec
Y ou are cataloguing one file of a personal knowledge base. From the file content below, write its catalogue record. Copy
names, identifiers and dates exactly as they appear; never invent a value that the content does not support.
Return JSON only:
{"title": "<title as printed, or a short descriptive title>",
"kind": "<what the file is: preprint, journal article, report, bill, thesis, abstract, photograph, photomicrograph, drawing,
floor plan, lecture recording, interview, song, ...>",
"summary": "<one sentence>",
"fields": {
"author": ["<each author’s full name, in order>"],
"affiliation": ["<each distinct affiliation, the full string as printed>"],
"subject": ["<the controlled-vocabulary subject terms a library catalogue would assign to this file, each in the
vocabulary’s exact preferred form: arXiv category codes (e.g. cs.CL, eess.AS, q-bio.NC) for arXiv preprints; 5-15
MeSH descriptors for biomedical articles, written as MeSH writes them (plural nouns, inverted forms with
commas), covering the population studied, the conditions, organisms, interventions, settings and methods, plus
check tags (Humans, Male, Female, age groups) when the study population supports them; policy areas for
government reports; subject headings otherwise>"],
"keyword": ["<author keywords or key topics>"],
"identifier": ["<DOI, arXiv id, report or bill number, PMID, ...>"],
"venue": ["<journal, conference, publisher or series>"],
"organization": ["<other organisations named>"],
"person": ["<other people named: speakers, editors, people shown>"],
"place": ["<places>"],
"depicts": ["<for media: what is shown or heard: people, organisms, objects, activities>"],
"object_type": ["<for media: the kind of object, e.g. Photographs, Portraits, Engravings, Maps>"],
"element": ["<for plans, diagrams and building photos: every room type and building element shown, each as a
canonical singular class name (e.g. Bedroom, Kitchen, Door, Window, Wall, Stairs, Balcony, Roof)>"],
"language": ["<language of the content>"]
},
"events": [{"event": "<submitted | received | revised | accepted | published | published online | published print | created |
recorded | issued | ...>", "date": "<YYYY -MM-DD, YYYY -MM or YYYY>"}]}
Rules: leave a field as [] when the content does not state it. An arXiv stamp such as "arXiv:2205.01234v2 [cs.CL] 24
May 2022" gives the identifier, the primary subject category and the submission date of that version. Journal front matter
gives received / accepted / published dates.
FILE:
{content}
Figure 13: Record promptp recof Eq. (3);{content}is the opening of˜a.
From record to hyperedges.Before any write, fields outsideΓ, placeholder values (“unknown”,
“n/a”, “none”, “null”), values of400or more characters and duplicates are dropped, and an event is
kept only if its date parses. The title and each non-empty field become onerecordhyperedge holding
the file and the field’s values, and each event aneventhyperedge dated YYYY-MM-DD, YYYY-
MM or YYYY . Every value enters throughι, so a value shared by two records is one entity with two
source files; Eq. (4) then adds a cross-modal hyperedge for each entity shared across modalities.
A.3 ROUTING PROMPTp route
Eq. (8) routes by rule a query that names a file or entity, is short, or has a lookup or synthesis cue;
others go to Figure 14, whose one-word reply picks the subgraph or community tier or both (hybrid).
Routing promptp route
Classify the query into ONE retrieval strategy over a personal knowledge hypergraph.
subgraph: a precise fact, definition, who/when/where/which. community: open-ended synthesis or a thematic overview.
hybrid: ambiguous, or needs both.
Reply with exactly one word: subgraph, community or hybrid.
Query:{query}
Figure 14: Routing promptp route, the last case of Eq. (8).
22

Preprint.
A.4 COMMUNITY REPORT PROMPTp com
The community tier summarises each of the64largest Louvain communities with the prompt of
Figure 15. Its slots hold the names of up to15member entities, up to12propositions, each the text
of a record hyperedge on a member, and the modalities those hyperedges were read from. Louvain
runs with a fixed seed on the entity graph whose edge weight counts the record hyperedges two
entities share. The reports, at most200words in four fixed parts, form the dense index that the
community tier searches withϕ(q); any edit invalidates them, and they are rebuilt on next use.
Community report promptp com
Y ou are summarising one community of a personal knowledge hypergraph.
Entities:{entities}
Propositions:
{propositions}
Modalities:{modalities}
Write a report of at most 200 words, plain prose: (1) THEME, one sentence; (2) KEY ENTITIES, the 3-5 most central
and their role; (3) SUMMARY of the facts and relationships; (4) CROSS-MODAL NOTES if several modalities occur. No
generic boilerplate.
Figure 15: Community report promptp comof the community tier (§4.3).
A.5 ANSWER PROMPTp ans
Every method of Table 1 that reads the corpus answers through the same prompt (Figure 16): a
system message, then the evidence, the question and the JSON output contract of the task’s answer
type, so that methods differ only in the evidence they supply. KnowledgeClaw fills the context with
Xq, anchor cards first. On phase B of aT 5item it retrieves again from the editedK, and the question
carries the update notice shown in the box. The contract follows the item’s answer type: a choice
index for the four-way items ofT 2andT 5, the entity names forT 1, the asked value for the joins ofT 3
andT 4, and the covered candidates forT 6; the reply is parsed into the prediction that §3 scores. For
the video item of Table 3, KnowledgeClaw sends only the card of the named video,3,659characters
of catalogue record and timestamped transcript, andΩchooses the gold option (D) from it alone.
Answer promptp ans
System:Y ou may use the CONTEXT below (retrieved automatically over the released corpus; it may be irrelevant,
incomplete, or -- in the ablation condition -- deliberately missing one modality). Do not assume anything not stated in the
context or the question.
User:CONTEXT:
{evidence}
QUESTION:{question}(phase B ofT 5:)QUESTION: (An update to the underlying record may have been filed
since any earlier question.){question}
T2,T5:Options:
[1]{option 1}
[2]{option 2}
[3]{option 3}
[4]{option 4}
Respond with JSON only: {"choice": <1-4>}
T1:Respond with JSON only: {"entities": ["<entity name>", "<entity name>", ...]}. Each element must be ONL Y the
entity’s own name/surface form, exactly as it appears -- for example a person’s name, an organisation, a subject term, or
a physical part or feature. Do NOT list the entity TYPES you were asked about (e.g. do not answer with "author" or
"subject" themselves), do NOT wrap entities in objects with their own type/modality fields, and do NOT answer
"unknown" -- if you cannot find any, respond with an empty list.
T3,T4:Answer the question directly. Respond with JSON only: {"answer": ["<value>"]} -- a list containing the specific
value(s) the question asks for (usually exactly one, e.g. a single date, name or identifier), not a list of unrelated entities
found in the context.
T6:Candidates:
-{candidate 1}
- . . .
Respond with JSON only: {"covered": ["<candidate>", ...]}. Each element must be one of the candidate strings above,
copied exactly, as a plain string (not wrapped in an object).
Figure 16: Answer promptp ansof §4.3, shared by every method that reads the corpus: a system
message and a user message that ends with the output contract of the item’s task (600output tokens).
23

Preprint.
B THEORETICALPROOF
The notation is that of §4;K t= (D t,Vt,Et, πt)is the artefact after thet-th edit,K 0is empty, and
anelementis an entity or hyperedgex∈ V t∪ Et. Apathof lengthmbetween nodesx 0andx m
is an alternating sequencex 0∈e1∋x1∈e2∋ ··· ∈e m∋xmof nodes and hyperedges ofE t,
andd t(x0, xm)is the length of a shortest one (∞if none). Three facts of the construction are used
throughout: (F1) node identity isι, a deterministic function of a value, soV tholds at most one node
per key; (F2) a record write changesKonly as in Eq. (3), creating an element with provenance{a}
or addingato an existing provenance; (F3) every edit is one of the three cases of Eq. (5), and edits
are committed one after another, so each step of a proof acts on one fixed, committed artefactK t−1.
B.1 PROOF OFPROPOSITION1
Proposition 1.InK, any two filesa, b∈ D, of the same or different modalities, whose records name
a common entityvare joined within two hyperedges, namely their record hyperedges containingv.
Proof.We first show that merging is order-free.Lemma.After the records of a setAof files
are merged, in any order,V ⊇ {ι(z) :z∈Z a,γ, a∈A, γ∈Γ}and, for each such nodev,
π(v)⊇ {a∈A:∃γ, v∈e a,γ}. Indeed, by (F1) and (F2) the write ofareaches the unique
node of keyι(z): the first write creates it and every later one unions its file intoπ(v). Set union is
commutative and associative, so the result does not depend on the order of the writes, nor on whether
they come from one ADDor from several, so buildingKat once or by later ADDs gives the sameK.
Now leta, b∈ D thold valuesz∈Z a,γandz′∈Zb,γ′withι(z) =ι(z′) =v. By the lemma the
nodevexists and the record hyperedgese a,γ∋a, vande b,γ′∋v, bare both inE t, whence the path
a∈e a,γ∋v∈e b,γ′∋b, d t(a, b)≤2.(11)
If moreoverµ(a)̸=µ(b), thena, b∈π(v)gives|µ(π(v))| ≥2, so the cross-modal hyperedge
e×
v={v}∪π(v)of Eq. (4) contains both files andd t(a, b) = 1. No step depends on the modalities:
Eq. (2) maps every modality to a text surface, one record call maps every surface onto the same
schemaΓ, andιacts on values alone, so the bound holds for all|M|2= 25modality pairs alike.
For person names the walk works in the quotient ofK tby the alias classes[v]; replacingvby[v]in
Eq. (11) gives the same bound for two files that write “M. Langtry” and “Max Langtry” alike.
Chains.Ifa 0, . . . , a m∈ Dtand each consecutive pair names a common entity, then concatenating
thempaths of Eq. (11) givesd t(a0, am)≤2m: a join overmrecords is a walk of at most2msteps
over one structure, whatever the modalities along the way, as the joins ofT 3andT 4require.
Stability.The join holds exactly while both records are present. If an edit removes a filec /∈ {a, b},
Eq. (6) only shrinksπ(v)⊇ {a, b}to a non-empty set, sovis kept, whilee a,γande b,γ′, whose
provenance is{a}and{b}, are untouched; if it removesa, the hyperedgee a,γleaves with it, and
so does the path. A store of segments, by contrast, has no node forv:aandbmeet only if one
similarity search happens to return both, which no structure of that segment store ensures.
Empirically,95.4–99.2%of each persona’s files lie within two hyperedges of another file (Table 12).
B.2 PROOF OFPROPOSITION2
Proposition 2.After any sequence of edits∆ 1, . . . ,∆ t, all ofK trests on present files, i.e., it satisfies
Eq. (7), so no evidence is stale: no later query draws evidence from a deleted or superseded file.
Proof.We first show by induction ontthat Eq. (7) holds after every edit. Fort= 0,K 0is empty
and the claim is vacuous; assume it holds forK t−1. By (F3) there are exactly three cases to check.
Case∆ t=ADD(A). By (F2) a write ofa∈Agives each element it creates the provenance{a}and
addsato the provenance of each element that exists; a cross-modal hyperedge takesπ(e×
v) =π(v).
Every provenance thus stays non-empty and withinD t−1∪A=D t, and index keys are added only
for the elements just written and for the segments ofA, which lie inΣ t; nothing is removed.
Case∆ t=DEL(h). LetR=η t−1(h), so thatD t=D t−1\R. By Eq. (6) withη t−1(h) =R,
the edit removes, with its index keys, every element whose provenance lies withinR, and deletesR
24

Preprint.
from every other provenance, which therefore stays non-empty:
∅ ̸=π t(x) =π t−1(x)\R⊆ D t−1\R=D t for every surviving elementx.(12)
The segments ofRleaveI σwith their files. Dropping orphans only removes nodes, and recomputing
e×
vfrom the prunedπ(v)keepsπ(e×
v) =π(v)⊆ D t. Every elementxwithπ t−1(x)∩R=∅keeps
its provenance unchanged, so the edit is local: it touches only the elements thatRsupports.
Case∆ t=REFRESH(h→A′). It is DEL(h)followed by ADD(A′)and preserves the invariant
because both do; re-pointingη t(h) =A′⊆ D tchanges no provenance, and a later query namingh
resolves toA′, so it reads the new values and never the old ones. This completes the induction.
Now letc /∈ D tbe a superseded or deleted file. The evidenceX qof §4.3 is assembled from (i)
the cards of files inH qand of linked files, all inD tbecauseη tmaps every handle intoD tand the
walk scores only present files; (ii) hyperedges ofE tand entities ofV t, whose provenance lies inD t
by Eq. (7); (iii) passages keyed inI σ, which belong to files ofD tby Eq. (7); and (iv) the keyword
index and the community summaries, which every edit invalidates and which are rebuilt fromK t.
The record and event hyperedges that stated the values ofchad provenance{c}and were removed
with it, and a value ofcthat survives inK tis, up toι, stated by a present file. Henceccontributes to
no part ofX q, for any queryqand any sequence∆ 1, . . . ,∆ t, so no evidence is stale.
Empirically, Eq. (7) held after each of the2,000edits of the stress test in §5.5.
B.3 PROOF OFPROPOSITION3
Proposition 3.For every queryq, each filea∈H qthatqnames enters its evidenceX qascard(a),
whateverk,λ,θand the size ofD t, which top-ksimilarity with anykbelow that size cannot ensure.
Proof. Part 1: KnowledgeClaw.Letqname a handlehand leta∈η t(h). By Eq. (8),a∈H q, hence
Hq∪Vq̸=∅and Eq. (8) routesqto thefileor thesubgraphgranularity, never to the community tier
alone. At either granularity the evidenceX qopens with the card of every file ofH q; the character
budget applies only to the linked works and passages that follow, and no later step removes a card.
Hence
Pr
card(a)∈ X q
= 1for everya∈H q,(13)
whateverk,λ,θor the size ofD t, andX qenters the answer prompt whole. By Proposition 2,
a∈ D t, so this card is also current. At thesubgraphgranularity the named file also seeds the
walk: for every filebthat shares a record entityu∈ V(a)witha, the third sum of Eq. (9) adds
log(1 +|D t|/|π(u)|)>0tos q(b), since|π(u)| ≤ |D t|by Eq. (7). Every file that Proposition 1
joins toathrough a non-hub entity is thus ranked by the walk, and the evidence is anchored on the
named file before any linked work is added. The guarantee covers a handle that names several files
as well: everya∈η t(h), e.g. a PDF and its text twin after a REFRESH, enters as its own card.
Part 2: top-ksimilarity.Let a retriever return thekitems of a corpus most similar toϕ(q)under
a fixed encoder, and let the corpus holdkfilesb 1, . . . , b kwhose surfaces embed in the direction of
ϕ(q),cos(ϕ(q), ϕ( ˜bi)) = 1, beside the named fileawithcos(ϕ(q), ϕ(˜a))<1. The retriever returns
b1, . . . , b kand missesa. The same happens, without exact ties, wheneverklook-alike files lie closer
toϕ(q)thanadoes, as the same-collection candidates of aT 2item do. More generally, suppose
aandℓlook-alike files are exchangeable to the encoder givenq, i.e. the joint law of their scores
is invariant under permutation, as it is when a content-hash handle carries no meaning an encoder
could use; ties are broken uniformly. By symmetry each of theℓ+ 1files is equally likely to hold
each rank, and the topkholdmin(k, ℓ+ 1)of them, so
Pr
a∈top-k(q)
=min(k, ℓ+ 1)
ℓ+ 1<1wheneverℓ≥k.(14)
Fusing several similarity lists, as Eq. (10) does, does not lift this bound: each list is a top-2kby
similarity, and an item absent from every list has fused score0. So no choice ofksmaller than the
corpus lets similarity search ensure that a named file is retrieved, while Eq. (13) holds for everyk;
KnowledgeClaw therefore uses similarity only to add evidence, never to decide whether a named
file is read. Propositions 1–3 thus give the join, maintenance and lookup guarantees of C1–C3.
Empirically, flat top-kmisses a named file on33.5%of theT 2items and76.9%of theT 4items,
while C3 brings every named file into the evidence (Table 14), exactly as Eq. (13) predicts.
25

Preprint.
C KNOWLEDGECLAWALGORITHMDETAILS
Algorithm 1 gives the full workflow of KnowledgeClaw (§4) in three phases:(1) Omnimodal hy-
pergraph construction (C1).Each fileais read once into a surface˜aby Eq. (2) and catalogued
by one call ofΩ; the records merge intoKby Eq. (3), with cross-modal hyperedges and twins by
Eq. (4).(2) Closed-loop lifecycle (C2).Each edit∆ tof Eq. (5) is committed in place: DELretracts
provenance by Eq. (6) with no model call, ADDreuses phase 1 on the added files only, and each
commit restores the soundness invariant of Eq. (7).(3) Intent-routed multi-granular retrieval
(C3).A query is anchored and routed by Eq. (8) to a tier; the subgraph tier walks from the anchors
by Eq. (9) and fuses two dense paths by Eq. (10); one call ofΩanswers from the evidenceX q.
Algorithm 1KnowledgeClaw: construct, maintain and query an omnimodal typed hypergraph
Require:Persona filesD, edit stream∆ 1,∆2, . . ., queriesq, frozen modelΩ, encoderϕ
Ensure:HypergraphK t= (D t,Vt,Et, πt)and answersˆy q
1:// 1: Omnimodal Hypergraph Construction (C1), ADD(D)on an emptyK
2:functionADD(A)
3:foreacha∈Ain parallel do▷model calls write nothing
4:˜a←proj(a)ifµ(a)∈ {text,pdf}, elseΩ(raw(a);p µ(a));(σ a,1, . . . , σ a,na)←split(˜a)
5: Catalogue record: 
{Za,γ}γ∈Γ,events
←Ω(˜a;p rec)
6:end for
7:foreacha∈A, each fieldγ∈Γdo▷sequential merge, Eq. (3)
8:e a,γ← {a} ∪ {ι(z) :z∈Z a,γ};E ← E ∪ {e a,γ};π(e a,γ)← {a}
9:π(v)←π(v)∪ {a}forv∈e a,γ∩ V; add alias keyι p(v)to class[v]for person names
10:end for
11: Add a datedeventhyperedge{a}per event; adde×
vand twinsa≈bby Eq. (4)
12: Insertϕ(v), ϕ(e), ϕ(σ a,j)intoI V, IE, Iσand˜ainto BM25
13:end function
14:// 2: Closed-loop Lifecycle (C2), for each edit∆ tof Eq. (5)
15:functionDEL(h)▷no model call
16:foreacha∈η(h), eachx∈ V ∪ Ewitha∈π(x)do
17:ifπ(x) ={a}thendeletexand its index keyselseπ(x)←π(x)\ {a}▷Eq. (6)
18:end for
19: Drop orphans; recompute touchede×
v;D ← D \η(h)
20:end function
21:functionREFRESH(h→A′): DEL(h); ADD(A′);η(h)←A′
22: After each commit: assert Eq. (7); mark derived indices for rebuild on next use
23:// 3: Intent-routed Multi-granular Retrieval (C3), for each queryq
24:H q←S
h∈hdl(q)η(h);V q← {ι(ψ)or nearest inI Vatcos≥θ:ψ∈Ψ q};g←g(q)by Eq. (8)
25:ifg=filethen
26:X q←full cards of the files inH q
27:else ifg=subgraphthen
28: Score eachb /∈H q(nor its twin) bys q(b)of Eq. (9), skipping hubs; merge twins into worksw
29:W q←works with¯s q(w)≥0.6 max w′¯sq(w′), at most six
30:L q,V←hyperedges of the top-2kentities ofI V;L q,E←top-2khyperedges ofI E
31:X q←cards ofH q∥cards ofW q∥top-khyperedges byrrf q(Eq. 10)∥four passages ofI σ
32:else
33:X q←the three nearest reports of the64largest Louvain communities, eachΩ(·;p com)
34:end if
35:returnˆy q= Ω(q,X q;pans)
Complexity analysis.LetN=|D|,n mthe number of media files,R=P
a|V(a)|the total record
size anddthe embedding dimension. First, construction issues exactlyN+n mcalls ofΩon
bounded inputs (Table 4), i.e.O(N)parallel calls, plusO(R)to merge andO 
(|V|+|E|+|Σ|)d
to index; no call sees two files together, so no step is quadratic inN. Second, an edit costs only what
it touches: DEL(h)visits the nodes whose provenance holds a file ofη(h),O(P
a∈η(h) |V(a)|)
with no model call, and ADD(A)costsO(|A|)calls, independent of|D t|, soTedits avoid the
O(TN)calls of rebuilding. Third, a query costs at most two calls ofΩ(routing only if no rule
applies, and the answer); the file tier isO(|H q|), the walk visits only files linked to an anchor,
O P
v∈Vq∪V(H q)|π([v])|
, and the two dense paths costO 
(|V|+|E|)d
. The evidence is bounded
(six works,khyperedges, four passages), so a query costs$0.0004and a1,024-token budget keeps
55.8of the full55.9OKE (§5.7); all three phases thus scale at most linearly with the corpus.
26

Preprint.
D OMNIKE-BENCHDETAIL
This appendix details the tasks, construction and metrics of §3. Personadhas corpusD d; aunituis
one released work (a document as PDF and text, an image, or a clip), arecordrone of its files with
modalitym(r)∈ M, andK d=S(D d)the artefact that a systemScommits atINGESTtime.
Table 5:The six OmniKE tasks.All gold comes from human-made metadata or description,
deterministic parsing or official transcripts, never from a model.M(t): evidence modalities (items).
Query→answer Gold source MetricM(t)
Construction
T1PDF or image+closed type
inventory→its entities of those
typescatalogued authors, affiliations, subject
terms, depicted persons; parsed floor-plan
elementsset F1pdf 344, image
56
T2media file→its authoritative
description among4candidatescatalogue record, creator description or
transcript; all four candidates from one
collection, each of a different subjectaccuracyimage 397,
audio/video 3
Application
T32–4-hop question→one
catalogued valuedeterministic join over catalogue records;
each join condition is one hopset F1image 275, pdf
125
T4asT3, but the chain joins image
and PDF records (2–3 hops)as above set F1 image+pdf 110
T6PDF or image+candidate
labels→those assigned to ithuman-assigned controlled-vocabulary
labels; negatives never co-assigned with
its labelsbalanced
acc.image 304, pdf
78
Maintenance
T5a document’s title, authors or
abstract; answer, commit∆,
answer againofficial version chain; the corpus holds
the prior version,∆the nextaccuracy,
both phasespdf 137
D.1 TASKDESIGN
Canonical contracts.Adapters lift each source onto nine contracts: REC (a record with its unit,
modality and parent), ORD (order by date, version or recording time), MEN (an entity of a cata-
logued class), ALN (a cross-modal alignment), IDN (an agent identity with its identifier basis), EVT
(a dated event), STA (a state value in one version), NRM (a validity norm and its deviations) and
COV (an assigned controlled-vocabulary term). Every row carries its provenance, and builders read
contracts only, so each task is one builder that runs unchanged over all ten personas alike.
Instantiation.T 1–T3run in every persona, the other
three tasks wherever their structure yields at least20items
(Table 6), never padded with model-made items:T 4needs
two modalities sharing authority-controlled entities (in
healthcare, NLM subject headings of CDC images and
PubMed Central papers),T 5an official version chain and
T6a controlled vocabulary with enough balanced labels.
T1(knowledge acquisition).Given a unitu(a PDF or
an image) and a closed type inventoryΓ = Γ d,m(u) , re-
turn the entities of those types inu,T 1: (u,Γ)7→ ˆE,
with goldE∗(u) ={surf(x) :x∈MEN(u),cls(x)∈
Γ,own(x, u)}. The ruleown(x, u)keeps only what the
unit itself states: a whole-word surface in its own human
text or aligned transcript, or, for an image, a row from a
human structural layer or a class naming what is depicted.
Γd,mholds every class with such a row in(d, m), never
tailored to any single item; on average, an item asks for
2.4entity types and holds6.0gold entities of those types.Table 6:Items per persona and task.
Each persona has90documents (PDF
and text),180images,18audio and18
video clips (2.9,2.4min on average);
15.8h of media,6.9GB in all. A blank
cell: the task is not run in that persona.
T1T2T3T4T5T6 All
HC 40 40 41 110 166 397
FIN41 40 41 29 151
EDU40 40 41 22 46 189
LAW36 40 41 117
PSY40 40 40 170 290
POL40 40 41 121
HUM41 40 41 122
BLD40 40 32 29 141
MUS41 40 41 29 151
GAM41 40 41 28 150
Total 400 400 400 110 137 382 1,829
T2(cross-modal coreference).Given a media recordr, choose among four candidates the de-
scription joined torby an ALN row,T 2: (r,{c 1, . . . , c 4})7→ˆı. All candidates come from the same
27

Preprint.
Table 7:Sources per persona.PD: public domain (US federal work); WC: Wikimedia Commons,
pages before 30 Nov 2022, each file keeping its licence (PD, CC0, CC BY or CC BY-SA); OCW:
MIT OpenCourseWare.NC: CC BY-NC-SA, kept per file. No source carries a no-derivatives term.
Persona text / pdf image audio video
HealthcarePubMed Central OA
(CC BY)CDC Public Health Image
Library (PD)PriMock57 (CC BY);
Senate HELP hearings
(PD)Senate HELP
hearings (PD)
Finance arXiv q-fin WCFOMC press
conferences (PD);
Earnings-21/22 (CC
BY-SA)FOMC press
conferences with
official transcripts
(PD)
Education arXivSmithsonian Open Access
(CC0)OCWNCOCWNC
LawSCOTUS opinions
(PD); DOAJ CC-BY
journalsWCSCOTUS oral
arguments, official
transcripts (PD)Senate Judiciary
hearings (PD)
Psychology PsyArXiv, PMCWellcome Collection
(PD/CC0/CC BY/BY-SA)OCW 9.00NC OCW 9.00,
captioned lecturesNC
PoliticsCRS reports with
version chains (PD);
World Bank OKR
(CC BY)WCV oxPopuli, European
Parliament (CC0)White House
briefings, official
captions (PD)
HumanitiesDOAJ CC-BY
humanities journalsMet and Cleveland
Museum Open Access with
curatorial records (CC0)Open Yale CoursesNC;
LibriV ox with
Gutenberg text (PD)Open Yale CoursesNC
BuildingarXiv; building codes
(CC0); NIST (PD)CubiCasa5K plansNC; CMP
facades (CC BY-SA)OCW planningNC OCW planning; site
footageNC
Music arXivMet musical instruments
(CC0)OpenMIC-2018 labels
(CC BY); FMA, no
NC/NDOCW 21M,
captioned lecturesNC
Games arXivWC screenshots;
OpenGameArt (CC0)OpenGameArt, creator
descriptions (CC0)WC, captioned game
videos (CC BY)
parentp(r)(transcript, explicit parent, partition key or collection), so distractors are of the same kind
and place; descriptions have at least40characters, and no two candidates share a subject group.
T3(multi-hop retrieval).Givenqencoding a join pathπ=r 0j1− → ···jh− →v,h∈ {2,3,4}, return
the terminal value,T 3: (q,K d)7→ˆa. Each hop is one contract join (an ALN step, the one other
record sharing a field, a unique-entity lookup or a terminal value step). A path is admissible only if
every record-valued hop yields one unvisited record, the terminal hop yields one value and no earlier
record gives the answer, so every hop is needed. The400items have300,72and28paths of2,3
and4hops.T 4(cross-modal reasoning)is the same over paths through two or more modalities,
one non-textual; its110items, disjoint fromT 3, join image and PDF records via NLM headings.
T5(lifecycle consistency).Over an official version chain, the system answersqon a document’s
state attribute (phase A, goldv old), commits the released next version∆ = (REFRESH, A ∆)with
A∆∩ Dd=∅, and answers again (phase B, goldv new̸=vold). Attributes are titles (49items),
abstracts (69) and author lists (19); spelling-only pairs are dropped, and no stem or option carries a
version label, date or other order cue, so only the committed edit separates the two phases.
T6(knowledge discovery).Given a unituand candidatesC=C+∪C−from one vocabulary,
return the covered ones,T 6: (u, C)7→ ˆC+⊆C.C+holds the unit’s human-assigned terms and
C−terms never co-assigned with them, with|C+|=|C−| ≥2; vocabularies are MeSH (171
items), LCSH (127), Smithsonian object types (46), LC Names (24) and three others (14).
D.2 BENCHMARKCONSTRUCTION
The four phases of Figure 3C run as seven stages, each SHA-256-ordered, fully cached and score-
blind.(1) Collect.Sources (Table 7) obey three rules. Gold must be human-made: ASR, OCR
and model descriptions may be system input but never gold, ruling out three ASR-captioned video
28

Preprint.
archives. Media must be redistributable, and crowd-written gold is limited to pages created before 30
Nov 2022.(2) Normalise and lift.Every file is re-encoded, renamed by content hash and stripped
of metadata; images come with their own catalogue record, never cut from a paper. Adapters map
each annotation to one contract row.(3) Instantiate, rewrite and verify.A template filled from
contract rows yields a stem, its slots and its answer. GPT-5-nano paraphrases the stem blind to
answer and options; six deterministic checks require verbatim slots, numbers and dates, no answer
in the stem and no task-declared pattern, e.g. a version label inT 5; failing items are discarded, never
repaired.(4) Balance and (5) recompute.No answer value exceeds10%of a task or persona share,
T2andT 5rotate gold positions, and frequentT 6labels are positive in1/3to2/3of their uses. An
independent second implementation recomputes every answer from the raw sources and agrees on
all1,829items.(6) Gate and (7) freeze.Gates confirm that no released file keeps an original name,
source URL or gold sentence; image-level records drop a redaction-box probe onT 2from77.3to
chance (25.0–27.5). Each persona is then cut to Table 6’s size, keeping every unit an item touches,
so no gold answer changes; a SHA-256 manifest seals the release, and one command rebuilds every
item from the raw sources, so the benchmark itself is fully reproducible from public data.
D.3 EVALUATIONMETRICDESIGN
Protocol.The harness buildsK donce per persona and answers every task, each item in a fresh
context, with a value list (T 1,T3,T4), an option (T 2,T5per phase) or a candidate subset (T 6); a
missing or unparseable answer scores0. Reads are not sandboxed (MMGraphRAG reopens images
at query time); answers are scored by the deterministic functions below, never by a model.
Lenient value matching.A predicted valuepand a gold valueghave similarityσ(p, g), where
nrm(·)is the normalisation (lower-case, trim punctuation, drop a leading article, collapse spaces),
tok(·)the lower-cased token set andγ(g)∈ {year,month,day}the precision of a gold date:
σ(p, g) =

1
date(p)≡ γ(g)g
ga date of granularityγ(g),
1 nrm(p) = nrm(g),
2|tok(p)∩tok(g)|/(|tok(p)|+|tok(g)|)otherwise,(15)
so “May 24, 2022” matches “2022-05-24” and a short affiliation its full catalogued form. Pairs with
σ≥τ= 0.5are matched one-to-one, greedily by decreasingσ, giving the match setM( ˆA, A∗).
Item and task scores.With ˆA,A∗the predicted and gold values and ˆC+the selected candidates,
s2=1[ˆı=g]for gold optiong,s 5=1
2(1[ˆa A=gA] +1[ˆa B=gB]), and
s1,3,4=2|M( ˆA, A∗)|
|ˆA|+|A∗|, s 6=|ˆC+∩C+|
2|C+|+|C−\ˆC+|
2|C−|.(16)
OnT 5always returning the old value, or always the new, scores exactly50, as does selecting all or
none of the candidates onT 6, so only a committed edit and real discrimination rise above half credit.
Ass∈[0,1]has variance at most1/4, the worst-case95%half-width of ann-item mean is98/√n:
±4.9forT 1–T3,±5.0forT 6,±8.4forT 5and±9.3forT 4; per-persona scores are descriptive.
Admission references.Table 8 gives the me-
chanical floorf(best of position, length, overlap
and prior heuristics), the zero-model shortcut probe
P, closed-bookC, flat retrievalRandR −m, with
CandRthe best of two models before the cut.C
is0.0on the three free-form tasks and below the
floor onT 2andT 5, so neither the stem nor mem-
orised knowledge answers it;Rreaches only2.5
and0.0on the joinsT 3andT 4, so retrieval alone
does not compose records; andR −mfalls below
RwhereverRscores, by up to56.0onT 2, so the
named modality is needed. NeitherPnorCex-
ceedsRon any task, so no shortcut beats reading
the corpus, the skill OmniKE-Bench measures.Table 8:Reference scores.R: media bridge
and three BM25 passages;R −m:Rwithout
the item’s modality.
T1T2T3T4T5T6
Itemsn400 400 400 110 137 382
Floorf12.1 33.5 2.0 0.0 50.0 50.0
ProbeP0.1 26.6 0.3 0.0 25.4 49.2
Closed-bookC0.0 32.3 0.0 0.0 35.8 58.4
RetrievalR5.3 82.3 2.5 0.0 64.2 69.9
R−m 0.0 26.3 0.0 0.0 32.8 50.3
29

Preprint.
E BASELINEDETAIL
This appendix specifies how every row of Table 1 is deployed: the shared protocol (§E.1), the30baselines in
the order and grouping of the table (§E.2–§E.5), and KnowledgeClaw with its three ablations (§E.6). For each
method we state whatINGESTbuilds, what a query retrieves and how theT 5edit∆is committed; every number
is the value used in our runs, and a token is4characters throughout, as in the evidence budgets of §5.7.
E.1 SHARED DEPLOYMENT PROTOCOL
One frozen stack.Every method calls the same modelΩ(GEMINI2.5 FLASHLITE) and encoderϕ
(GEMINIEMBEDDING2,3,072dimensions) through one metering proxy that caches each call by the SHA-256
of model, messages and parameters, at temperature0unless a method’s official default differs (MemPro,0.3).
One input.A text or PDF file enters as its opening4,000tokens, cut at a paragraph break and headed by its
file name, the only handle items use. Text-reading methods receive image, audio and video through the shared
media-to-text bridge (image captions, time-windowed transcripts); the omnimodal group and KnowledgeClaw
read raw media natively.One answerer.Whatever a method retrieves, or gathers with its own agent loop,
ends in the answer prompt of Appendix A.5 with600output tokens, so methods differ only in the evidence
they supply.One protocol.Per persona a method builds once over the base corpus and answers each item
in a fresh context at24-way concurrency. OnT 5all phase-A answers come first, then every item’s∆(the
next released version, headed “a new version ofh”) is committed through the method’s own write path, then
all phase-B answers, so no phase-A answer can see an edit. Every method attempts all six tasks.Faithful
re-implementation.Each baseline is re-implemented on this stack from its official code (from the paper where
none is released: OntoKG, M3KG-RAG), with its prompts copied verbatim and its defaults kept; in-process
indices replace databases, the frozen stack replaces each method’s own models (NV-Embed, MiniLM, CLIP,
Whisper, Qwen), and gleaning is0wherever offered. Other deviations are listed per method; costs are in Ta-
ble 9, metered by the same proxy for every method and split into ingest and query by when each call was made,
with the shared media-to-text bridge charged once for the whole benchmark rather than per method.
E.2 NO KNOWLEDGE ENGINEERING,CHUNK,GRAPH AND AGENTIC GRAPHRAG
NaiveGeneration(Google, 2025). No ingest, retrieval or write:Ωanswers from the question and options
alone, the closed-book control, which shows what the frozen model knows without any corpus.
StandardRAG(Lewis et al., 2020).Ingest:1,200-token windows with100overlap, embedded byϕ.Query:
cosine top-5windows.Edit:the windows of∆are appended beside those of the prior version.
RAPTOR(Sarthi et al., 2024).Ingest:sentence-bounded leaves of≤100tokens; up to5layers of soft GMM
clustering (k≤50by BIC, membershipp >0.1, PCA to10dimensions for UMAP, clusters over3,500tokens
re-clustered), each cluster summarised in≤100tokens.Query:collapsed-tree top-10over all layers within
3,500tokens.Edit:the tree is rebuilt over the base corpus and∆, which still holds the prior version.
LightRAG(Guo et al., 2025b) (v1.4.14).Ingest:1,200/100-token chunks; one call per chunk extracts entities
of11types and keyworded relations, merged by name and pair with map-reduce description summaries.Query:
hybrid mode; a keyword call sends low-level keys to the entity store and high-level keys to the relation store
(top-40each, cosine>0.2), plus20related chunks under6,000/8,000/30,000-token entity, relation and total
budgets.Edit:native incremental insert, merging the entities and relations of∆by name and pair.
GraphRAG(Edge et al., 2025).Ingest:1,200/100-token units; one call per unit extracts entities of four types
and weighted relations, descriptions are summarised in≤500words, recursive Louvain (communities above
10members split again) stands in for hierarchical Leiden, and each community receives a rated report.Query:
local search from the top-20entities within12,000tokens, split15/35/50%over reports, entity-relation tables
and text units.Edit:the official update mode extracts∆, merges by title and pair, and re-summarises.
HippoRAG 2(Guti ´errez et al., 2025).Ingest:one passage per file; OpenIE by an NER call and a conditioned
triple call; phrase, fact and passage nodes with synonymy edges at cosine≥0.8.Query:the top-5facts pass
the recognition-memory filter (10shots) and seed Personalized PageRank (damping0.5, passage weight0.05);
returning the top-5passages.Edit:incremental indexing, synonymy edges recomputed.
KAG/OpenSPG(Liang et al., 2025).Ingest:1,200/100-token chunks; three schema-free calls a chunk (NER,
standardisation, triples) into an entity-chunk graph with a PageRank matrix.Query:its solver plans a logical
form, runs≤3retrieval steps (exact linking at0.9, else fuzzy at0.8with PPR chunks, RRF-fused with vector
chunks at0.65), summarising each step,≤5calls.Edit:incremental extraction, PPR matrix rebuilt.
Youtu-GraphRAG(Dong et al., 2026a).Ingest:1,200/100-token chunks (official5,000/200), one schema-
guided call each, into a four-level tree of attributes, triples, keywords and≤100FastTreeComm communities.
Query:the agent splitsqinto≤4sub-questions, retrieves triples by node, relation and3-hop paths (top-20),
and runs IRCoT for≤2steps (official5).Edit:full rebuild of the tree over the base corpus and∆.
E.3 KNOWLEDGE GRAPH CONSTRUCTION ANDn-ARY/HYPERGRAPHKE
iText2KG, OntoKG, Hyper-KGGen and ATOM construct a graph but define no query path; their facts, each
tagged with its source file, are served by one shared layer, cosine top-30overϕ, the same for all four.
30

Preprint.
iText2KG(Lairgi et al., 2025).Ingest:1,200-token chunks are distilled into facts, one section per file; entity
and relation calls follow, entities resolve on exact (name, label) match or cosine>0.92(0.6name+ 0.4
label), relation names at0.9, and one round revisits isolated entities (official3); the official0.8/0.7thresholds
collapsed relations onto∼30names and were recalibrated.Edit:∆is ingested onto the graph.
OntoKG(Li et al., 2026c).Ingest:one call designs a modular schema over8seed categories from24sampled
files, one refinement call repairs the types and properties a probe extraction leaves unrouted, and600-token
chunks are extracted with the paper’s Listing 2 and routed to modules by exact or cosine-≥0.9match.Edit:
∆is extracted and routed into the frozen schema.
HyperGraphRAG(Luo et al., 2025).Ingest:1,200/100-token chunks; one call per chunk yieldsn-ary knowl-
edge segments with scored entities, stored as a bipartite hyperedge-entity graph.Query:entity and segment
keys parsed fromqdrive a local (top-60entities) and a global (top-60hyperedges) path at cosine>0.2,
returning entities, hyperedges and source chunks,4,000tokens each.Edit:incremental insert.
Hyper-KGGen(Huang et al., 2026).Ingest:the official recursive splitter (1,000characters,200overlap); per
chunk a mention call and a hyperedge call over the known nodes; nodes merge by name and hyperedges by
member set, and the skill library is empty as none is released.Edit:∆runs the same pipeline and merges in.
HyperRAG(Lien et al., 2026).Ingest:the HyperGraphRAG hypergraph, built with its own calls.Query:
topic entities ofqseed an LLM-scored beam search over hyperedges and entities (width3, depth3,≤100
candidates per entity) with a sufficiency check per depth; paths and their chunks enter within12,000tokens.
Edit:incremental insert through the HyperGraphRAG write path, which keeps the prior version in the graph.
E.4 MEMORY AGENTS AND SELF-EVOLVING LIFECYCLE
MemGPT/Letta(Packer et al., 2024).Ingest:512/50-token passages into archival memory beside the core
personaandhumanblocks.Query:a tool-calling loop under the verbatimletta v1system prompt
searches archival memory (top-5,≤20) and edits core memory for≤5steps.Edit:passages are appended.
A-MEM(Xu et al., 2025).Ingest:one note per1,200/100-token chunk, analysed into keywords, context and
tags; each new note runs the evolution call over its5nearest notes, strengthening links or updating neighbours.
Query:top-5notes and up to5linked notes.Edit:∆is added and evolved.
Mem0(Chhikara et al., 2025).Ingest:a fact-extraction call per chunk, then one update call over each fact’s
top-5memories decides ADD, UPDATE, DELETEor NONE.Query:top-100memories with their sources.Edit:
the same path, so∆can update or delete memories.
Graphiti (Zep)(Rasmussen et al., 2025).Ingest:one episode per chunk at a reference time; entity extraction,
resolution (exact name,3-gram Jaccard≥0.9, else an LLM dedupe over15candidates), fact extraction with
validity times, and edge resolution that invalidates contradicted facts.Query:RRF of BM25 and cosine over
facts and entities (top-10each) and BM25 episodes, with validity dates.Edit:∆is ingested at a later time and
invalidates what it contradicts. To fit the shared budget, a fact call has3,072output tokens and15facts.
MIRIX(Wang & Chen, 2025).Ingest:a meta-manager routes each chunk to the core, episodic, semantic,
resource and knowledge-vault managers, each writing its store by a JSON function call against its BM25 top-
10; procedural memory stays empty, as released.Query:a topic call, then core memory, the10latest episodes
and BM25 top-10per store.Edit:the managers insert, merge or replace items.
MemOS(Li et al., 2025).Ingest:the structured document reader turns each chunk into key-value-tag mem-
ories, with a20-item working memory.Query:a task-parse call yields keys, tags and2–5expansions; graph
recall (key or≥2tags) and vector recall are reranked by cosine to the top-20.Edit:new memories are added.
MemEvolve(Zhang et al., 2026). It evolves agent memory and has no document ingest, so each chunk enters its
fusion-memory provider as a successful trajectory “memorise⟨file⟩”.Ingest:a summarise call writes insight,
pattern, playbook and checklist nodes, linked at similarity≥0.75and consolidated every50ingests.Query:a
focus call, hybrid search (0.2TF-IDF+ 0.8cosine, top-5,3-neighbour expansion) and a synthesis call give a
≤350-character note.Edit:∆is written through the same path.
ATOM(Lairgi et al., 2026).Ingest:380-token chunks stamped with the date each file states; atomic facts, then
temporal5-tuples(s, p, o, t start, tend), merged with entities at cosine≥0.9and served by the shared fact layer
of §E.3.Edit:∆is observed later; a relation its old file supported that the new file does not re-observe loses
that source, and is closed att endonce none remains.
MemPro(Liu et al., 2026d). The released evolved program is used without re-evolution.Ingest:2,000-token
pages, each abstracted by its memory agent, into dense and BM25 indices.Query:planning over all abstracts,
keyword, vector and page search (top-5), focused excerpts of≤3pages, integration and reflection, for one
research iteration (official3) to fit the shared budget.Edit:new pages are abstracted against the memory and
appended, so the prior version’s pages stay searchable (19.0%of its phase-B answers superseded, Table 13).
E.5 AGENTIC KNOWLEDGE ENGINEERING AND OMNIMODAL KNOWLEDGE ENGINEERING
OpenClaw, LLM Wiki and OKF build markdown knowledge bases for an agent to read, so they share one
reader: hybrid search (0.7cosine+ 0.3BM25, top-6over1,500-character windows), open and list tools,≤5
steps and≤24,000characters gathered, so they differ only in the knowledge base each one writes.
31

Preprint.
OpenClaw(OpenClaw contributors, 2026).Ingest:the corpus is split into32sessions; each file is one
agent turn under the verbatimAGENTS.mdmemory rules, appending to the session’s daily note, with the
pre-compaction flush beyond32,000tokens.Edit:∆runs as a new session that writes a new note.
Graphify(Graphify contributors, 2026).Ingest:the verbatim skill prompt extracts nodes and edges from
batches of22files (one per image); Louvain (seed42) is its fallback for Leiden.Query:term hits in label and
source pick5start nodes, expanded by BFS to depth2within2,000tokens.Edit:--updateon unseen files.
Cognee(Markovic et al., 2025).Ingest:cognify; per file a graph-extraction and a summary call into doc-
ument, chunk, summary, entity and type nodes.Query:graph completion; top-100per collection, unmatched
elements penalised6.5, and the5best triplets by summed distance.Edit:cognifyrun on∆alone.
LLM Wiki(Karpathy, 2026). The idea file gives no prompts, so our schema instantiates it.Ingest:in waves of
16, each source gets a page and a plan over its12nearest entity and concept pages (≤5updates,≤3new), then
one merge rewrite per touched page.Edit:a new version plans against its old source page first, then merges.
OKF(Google Cloud, 2026).Ingest:one concept document per file, written by the reference instruction with
the concept list in view, plus per-directory indices and a log.Edit:a new version refines its base concept in
place, with the existing document as input and the update logged, without ever duplicating the concept page.
Omni-SimpleMem(Liu et al., 2026a).Ingest:multimodal atomic units from raw media:500/50-token text
chunks, redundancy-filtered and summarised; two captions per image; audio transcripts; video at1fps through
the HSV scene trigger,≤24captioned frames and its audio.Query:dense top-10and BM25 top-10.Edit:∆
adds units beside those of the prior version, which both searches can still return after the edit is committed.
MMGraphRAG(Wan & Yu, 2026).Ingest:a text graph (1,200/100-token chunks, transcripts included)
fused with an image graph built from the raw images (description, image entities, enhancement, whole-image
matching by spectral clustering), one graph per persona.Query:top-5nodes at0.2with their tables, plus a
vision answer over each of≤3retrieved images.Edit:a full rebuild over the base corpus and∆, as at ingest.
M3KG-RAG(Park et al., 2026a).Ingest:2,000-token text items and a raw-media caption per image, audio and
video file; triples by the extractor and normaliser, descriptions by the searcher-inspector (pass≥7,≤3rounds).
Query:items withinL 2≤1.1of the top-5plus named files,2-hop expansion scored bycos(h, q) + cos(t, q),
and GRASP pruning.Edit:∆adds its items, captions and triplets to the graph built at ingest time.
E.6 KNOWLEDGECLAW AND ITS ABLATIONS
KnowledgeClaw.Ingest (C1):text and PDF from the shared projection; image (≤1,536px), audio (16kHz)
and video (≤6keyframes, one per20s, with audio) read natively byΩ(Appendix A.1), all2,160media files of
the benchmark; surfaces split into∼1,000-token segments; one record call per file over its first9,000characters
fills13typed fields, title and events; record, event and cross-modal hyperedges and twins by Eqs. (3)–(4);
indicesI V, IE, Iσand BM25.Query (C3):anchors atθ= 0.86and routing by Eq. (8);7,000-character anchor
cards; the walk of Eq. (9) keeps works within0.6of the best, at most six; RRF withλ= 4,k= 10, and four
passages (Eq. 10), within22,000characters;64Louvain communities (seed0) reported in≤200words; one
retrieval pass and one answer call.Edit (C2):a new version of a live handlehcommits as REFRESH(h→A′):
DELwith provenance pruning, ADD(A′)and re-pointingh(Eqs. 5–6), each commit audited and checked
against Eq. (7); a median DELtakes9ms and a REFRESH2.6s, against118s to ADDthe whole corpus (§5.5).
w/o C1 (no hypergraph).Media arrive through the shared bridge instead of native reading and no record call
is made, so only the segments of the text projection are indexed (I σ), with no entity, hyperedge or community.
C2 and C3 run unchanged on this store: file handles still resolve to their cards and REFRESHstill retracts the
old version, so it keeps97.1phase-B accuracy (Table 13). It reaches49.1OKE, losing most onT 1(−20.0)
andT 4(−13.6), where native media reading and the typed hyperedges that join files matter most.
w/o C2 (no lifecycle).C1 and C3 are unchanged; every∆is appended as new files, with no DELof the old
version and no re-pointing ofh, so the handle keeps resolving to the version it named. It reaches50.4OKE,
and onlyT 5falls (49.6against95.6), as94.2%of its phase-B answers return the superseded value (Table 13).
w/o C3 (flat retrieval).C1 and C2 are unchanged; a query skips anchoring, routing, cards, the walk and the
communities, and its evidence is the RRF top-10hyperedges of Eq. (10) over the top-20ofI VandIE, plus the
top-6passages ofI σ. It reaches26.7OKE and trails KnowledgeClaw on every task, most onT 2(34.8against
82.8), where flat top-kmisses the named file on33.5%of the items, exactly as Proposition 3 predicts.
The ablations share KnowledgeClaw’s call cache and protocol, so each differs only in the component it removes,
and Figure 5b plots each of its losses on all six tasks, relative to the full model’s own scores in Table 1.
Settings in the analyses.The values above are the defaults of every KnowledgeClaw row. The sweeps of §5.7
(Figure 10) varyθ, the keep ratio, the cap on linked works andkaround them, and the forced granularities of
§5.6 (Figure 8c) fix the tier of Eq. (8) instead of routing; every other setting stays as listed, and none is tuned
per persona or per task. Table 15 lists the full sweep, and the default reproduces Table 1 exactly (Appendix F.1).
On this stack KnowledgeClaw spends$0.32per persona on ingest and$0.0004per query, and builds a persona
in a median of3minutes of wall-clock time at24-way concurrency (Table 9), below MemPro on all three.
Every KnowledgeClaw number in this paper uses these settings on the frozen stack of §E.1.
32

Preprint.
F EXPERIMENTDETAIL
F.1 PROTOCOL AND COST
Saved outputs.Every number of §5.2–§5.9, apart from the matched inputs and the judge of §5.8,
which call the model again, is recomputed from the main run: the per-item predictions of all rows of
Table 1, scored as in §3, and the traces, audit logs and hypergraphs saved by KnowledgeClaw and
its ablations. A difference quoted in the text is taken between rounded entries.
Analyses with new answers.For the evidence budget (Figure 9b), each item’s saved evidence is
cut to its firstBtokens at a line break and answered again; the full-length point is the main-table
prediction. For the forced granularities and the sweep (Figures 8c, 10), KnowledgeClaw is rebuilt
from the main run’s call cache and every query is re-retrieved on the main protocol; an item whose
evidence is unchanged keeps its prediction, so the default reproduces Table 1 exactly. The edit
stream (Figure 7c) applies200random DELand re-ADDedits per persona after theT 5commits and
checks Eq. (7) after each. The three analyses together spent only$2.6of API calls in total. The
matched inputs and the judge of §5.8 add only$0.006of API calls in total (Appendix F.7).
Cost.KnowledgeClaw builds a persona in3minutes at24-way concurrency, beaten only by
Graphify (2, OKE17.8), and spends$0.32to ingest it and$0.0004per query,27.8×below MemPro;
only methods with OKE≤21.0are cheaper or faster, far below its55.9(Table 9).
Table 9:Cost. inis ingest USD per persona,queryUSD per item andminwall-clock minutes per
persona at 24-way concurrency (median over the ten personas). Spend is the metered API cost of
each method’s calls on the frozen stack, split into ingest and query by when each call was made; the
shared media-to-text bridge is built once for the whole benchmark and charged separately.
Methodin query minMethodin query minMethodin query min
StandardRAG 0.11 0.0004 3 A-MEM 0.88 0.0009 46 OpenClaw 0.37 0.0016 5
RAPTOR 0.27 0.0005 15 Mem0 0.52 0.0006 36 Graphify 0.22 0.0003 2
LightRAG 1.10 0.0040 11 Graphiti (Zep) 2.14 0.0014 76 Cognee 0.52 0.0010 15
GraphRAG 0.89 0.0015 9 MIRIX 1.18 0.0018 17 LLM Wiki 0.78 0.0009 14
HippoRAG 2 0.39 0.0010 18 MemOS 0.46 0.0005 10 OKF 0.76 0.0010 4
KAG/OpenSPG 1.37 0.0019 19 MemEvolve 0.23 0.0002 6 Omni-SimpleMem 0.38 0.0011 9
Youtu-GraphRAG 0.80 0.0028 17 ATOM 1.75 0.0004 111 MMGraphRAG 1.07 0.0017 12
iText2KG 1.01 0.0003 113 MemPro 0.42 0.0111 13 M3KG-RAG 1.26 0.0011 26
OntoKG 0.72 0.0001 73 HyperGraphRAG 1.25 0.0041 11
Hyper-KGGen 1.18 0.0002 98 HyperRAG 1.28 0.0052 16KnowledgeClaw 0.32 0.0004 3
MemGPT/Letta 0.12 0.0009 5
F.2 MAIN RESULTS AND ABLATIONS(RQ1–RQ2)
Setting.Figure 5a divides each task score by the best score of any row of Table 1; Figure 5c
compares each persona score with that of the strongest of the30baselines on that persona (two-
sided sign test). Each ablation (Appendix E.6) shares the call cache and removes one component.
Results.KnowledgeClaw leads on all ten personas by+13.3on average (Table 10), while the
strongest baseline changes with the persona (MemPro seven times, OKF twice, LLM Wiki once);
onT 1the four baselines of Figure 5a reach only20.0–52.7%of its score. Each ablation loses most
where its component acts: C1 onT 1(−20.0) andT 4(−13.6), C2 only onT 5(−46.0), and C3 on
every task (−7.8to−48.0), so the three components are complementary rather than redundant.
F.3 OMNIMODAL TYPED HYPERGRAPH(RQ3)
Setting.A slice holds the items of one task whose unit has the given modality. The510items of
T3andT 4are grouped by hops, by join shape (alookup chainfollows one value at a time, ashared-
entityjoin meets a file on a common entity, aconjunctiveone needs two entities) and by anchor (a
named file or a quoted entity). A file isjoinedwhen another file lies within two hyperedges.
Results.KnowledgeClaw leads the best baseline on every slice, depth, join shape and anchor (Ta-
ble 11), most where reading and joining are hardest: onT 1PDFs and images, where the best omni-
modal baseline gets18.8and7.7, and at four hops (7.1against3.6); removing C1 costs up to21.3on
33

Preprint.
Table 10:Per-persona comparison (RQ1).Persona score of KnowledgeClaw and of the strongest
of the30baselines on each persona, with the name of that baseline and KnowledgeClaw’s gain.
HC FINEDULAWPSYPOLHUMBLDMUSGAM
KnowledgeClaw60.9 64.4 59.2 46.6 65.2 28.3 57.3 47.3 64.5 65.7
Best baseline 45.5 45.2 50.7 38.9 53.6 23.6 42.4 37.4 43.4 45.5
which OKF MemPro MemPro OKF MemPro MemPro LLM Wiki MemPro MemPro MemPro
Gain+15.4 +19.2 +8.5 +7.7 +11.6 +4.7 +14.9 +9.9 +21.1 +20.2
a slice. The baseline median falls from7.5at two hops to2.6at three and0.0at four, while Knowl-
edgeClaw keeps7.1, and only37of the510items (7%) are misread, answered wrongly though their
evidence holds a gold value. Per persona the hypergraph joins95.4–99.2%of files in3.6–6.1MB
(Table 12), with cross-modal hyperedges for every pair of the five modalities: the180audio and180
video files reach text by218and216hyperedges, PDF by206and208, and each other by214.
Table 11:Omnimodal typed hypergraph (RQ3).Top: score per modality slice (nitems). Bottom:
set F1 onT 3andT 4by depth, join shape and anchor;best: strongest of the30baselines.
SlicenOurs Best text-bridge Best omnimodal w/o C1
T1PDF 34456.430.8 (OKF) 18.8 (M3KG-RAG) 35.1
T1image 5621.65.3 (MemPro) 7.7 (M3KG-RAG) 10.0
T2image 39782.978.6 (MemPro) 70.0 (Omni-SimpleMem) 81.6
T3PDF 12546.440.0 (MemPro) 32.8 (MMGraphRAG) 32.0
T4image+PDF 11020.917.3 (OKF) 10.9 (Omni-SimpleMem) 7.3
T6image 30471.466.8 (MemPro) 65.5 (Omni-SimpleMem) 69.6
JoinnOurs Best w/o C1 JoinnOurs Best w/o C1
2 hops 32619.316.9 (MemPro) n/a shared entity 8414.38.3 (MemPro) 7.1
3 hops 15613.512.2 (OKF) n/a conjunctive 519.85.9 (OKF) 3.9
4 hops 287.13.6 (RAPTOR) n/a file anchor 12212.37.4 (OKF) 5.7
lookup chain 37518.414.7 (MemPro) 11.7 entity anchor 38818.314.2 (MemPro) 11.6
Table 12:Committed hypergraph per persona (RQ3).Hyperedges by kind;joined: % of files
within two hyperedges of another file;216media files per persona are read natively by C1.
HC FINEDULAWPSYPOLHUMBLDMUSGAM
Files 396 417 413 396 396 396 396 417 417 418
Entities 6,023 4,302 4,487 4,584 5,606 3,759 4,358 3,782 3,039 3,981
Record 2,966 2,779 2,667 2,967 2,897 2,913 2,740 2,632 2,302 2,399
Event 597 223 201 396 468 222 311 194 141 133
Cross-modal 1,319 373 325 1,797 985 1,273 1,260 642 360 367
Joined (%) 98.2 98.6 97.3 99.2 98.7 98.7 96.2 99.0 95.4 98.6
Size (MB) 6.1 4.3 4.2 5.6 5.3 5.7 4.6 4.7 3.6 3.8
F.4 CLOSED-LOOP LIFECYCLE(RQ4)
Setting.Each of the137T 5items is answered (phase A), its next version is committed as REFRESH,
and it is answered again (phase B); its score is the mean of both. A phase-B answer issuperseded
when it returns the replaced value. Latencies are single commits read from the audit logs.
Results.KnowledgeClaw leads every baseline in both phases (93.4and97.8, against at most81.8
and80.3; Table 13); after the commit its evidence holds the new value on95.5%of the items and
the old one on none. Over2,000edits,100of them on audio and84on video files, Eq. (7) always
holds, while an append-only store ends with10,148stale nodes and1,190stale segments; a commit
touches a median0.33%of the hypergraph (0.86%at the 95th percentile), so DELtakes9ms and
REFRESH2.6s against118s to ADDthe whole corpus, a small fraction of a rebuild.
F.5 INTENT-ROUTED MULTI-GRANULAR RETRIEVAL(RQ5)
Setting.A file isnamedwhen the question carries its handle. Evidence length is the median per
task, and ours is split by section headers into cards, linked-work cards, hyperedges and passages.
34

Preprint.
Table 13:Lifecycle consistency per phase (RQ4). A,B:T 5accuracy before and after the edit;
Sup.: % of phase-B answers that return the superseded value (lower is better); ours in bold.
Method A B Sup. Method A B Sup. Method A B Sup.
NaiveGeneration 34.3 40.1 34.3 HyperRAG 27.7 35.8 27.0 Cognee 73.7 16.8 68.6
StandardRAG 73.0 30.7 64.2 MemGPT/Letta 32.8 35.8 34.3 LLM Wiki 73.0 28.5 70.8
RAPTOR 44.5 34.3 47.4 A-MEM 77.4 21.2 73.7 OKF 73.7 35.0 65.0
LightRAG 72.3 27.7 63.5 Mem0 38.7 37.2 28.5 Omni-SimpleMem 75.9 28.5 70.1
GraphRAG 27.7 38.0 29.2 Graphiti (Zep) 81.8 35.8 61.3 MMGraphRAG 26.3 38.7 29.2
HippoRAG 2 51.1 42.3 43.1 MIRIX 33.6 43.8 27.7 M3KG-RAG 24.1 41.6 28.5
KAG/OpenSPG 57.7 35.8 53.3 MemOS 57.7 38.0 56.9 w/o C1 93.4 97.1 2.9
Youtu-GraphRAG 29.9 35.8 32.1 MemEvolve 27.0 38.0 30.7 w/o C2 93.4 5.8 94.2
iText2KG 19.0 36.5 24.8 ATOM 27.7 32.8 26.3 w/o C3 78.1 47.4 23.4
OntoKG 27.7 35.8 24.1 MemPro 77.4 80.3 19.0KnowledgeClaw 93.4 97.8 2.2
HyperGraphRAG 35.0 45.3 33.6 OpenClaw 48.9 33.6 51.1
Hyper-KGGen 24.8 30.7 26.3 Graphify 32.1 39.4 28.5
Results.Routing gives lookups anchor cards alone and joins a walk made mostly of linked-work
cards (Table 14); on every task its evidence is shorter than flat top-kand scores higher, by+7.8to
+48.0. Each fixed tier wins only for its own intent; the router matches the best on every task. On
T4, flat top-kbrings all named files into the evidence on only23.1%of items, against100%for C3.
Table 14:Evidence per task (RQ5).Named: % of items with all named files in the evidence;
tokens: median evidence length; make-up of KnowledgeClaw’s evidence in %; score as in Table 1.
Named (%) Tokens Make-up of ours (%) Score
Task Ours Flat Ours Flat Card Linked Edge Passage Ours Flat
T1 10093.8 1,972 3,686 100 0 0 051.615.7
T2 10066.5 248 2,250 100 0 0 082.834.8
T3 10085.4 1,416 2,647 25.2 50.4 7.5 17.015.88.0
T4 10023.1 1,986 4,761 7.8 61.5 6.8 24.020.95.5
T5 10063.5 2,145 3,840 100 0 0 095.662.8
T6 10086.6 295 3,103 100 0 0 076.057.5
F.6 EFFICIENCY AND ROBUSTNESS(RQ6–RQ7)
Setting.Evidence is cut toBtokens; the sweep varies one default of Appendix E.6 at a time.
Results.512tokens already give55.0OKE and1,024tokens55.8, while flat top-kstays within
20.6–26.7at every budget (Table 15). No setting of the sweep changesT 1,T2,T5orT6, and OKE
moves by at most0.4; a larger keep ratio even liftsT 4to32.7: the defaults are not tuned.
Table 15:Efficiency and robustness (RQ6–RQ7).Left: OKE with evidence cut toBtokens.
Right: one hyperparameter varied (default underlined);T 3,4: mean over allT 3andT 4items.
BOurs FlatθT 3,4 OKE keepT 3,4 OKE linksT 3,4 OKEkT 3,4 OKE
25651.220.6 0.80 16.7 55.9 0.4 17.6 56.0 2 18.2 56.2 5 17.1 56.0
51255.022.0 0.83 16.9 55.9 0.5 17.5 56.0 4 16.9 55.9 10 16.9 55.9
1,02455.822.4 0.86 16.9 55.9 0.6 16.9 55.9 6 16.9 55.9 15 17.1 56.0
2,04855.324.8 0.89 16.7 55.9 0.7 19.8 56.3 8 16.7 55.9 20 16.9 56.0
full55.926.7 0.92 16.3 55.8 0.8 19.6 56.3 10 16.9 56.0
F.7 MATCHED INPUTS AND SCORERS(RQ8)
Setting.The matched inputs rerun MemPro, OKF and StandardRAG with builds and retrieval re-
played from the original call cache, so only the final answer call is new;T 5is not rerun. A replay
check of20items per persona reproduces the stored predictions of StandardRAG on all ten personas
and of MemPro and OKF on nine (education8/20, building18/20). The scorer study rescores
the stored predictions of all34rows of Table 1 by exact set F1 (strict) and by the matcher of Ap-
pendix D.3 (lenient). The judge,Ωat temperature0, decides for each gold value whether a predicted
value denotes it, on300non-empty predictions sampled with seed0,25per task from Knowledge-
Claw, MemPro, OKF and StandardRAG onT 1,T3andT 4, the value-list tasks;275replies give one
35

Preprint.
decision per gold value. The grounding check reads the committed hypergraphs of the main run: a
copied record value is grounded when its normalised form occurs in the text C1 read for its file.
Results.The ingested text alone lifts each baseline less than the record cards do (45.9,41.3and39.6
against50.7,45.6and44.4OKE; Table 16); with the records, OKF and StandardRAG gain16.5and
37.0onT 1and7.1and23.4OKE, still below55.9. KnowledgeClaw is first on every task under both
scorers, and only0.5%of its answers cannot be parsed, against5.7%for MemPro and a median of
2.2%(Table 17). The matcher agrees with the judge on95.7%of the decisions (Table 18);97.8%
of the29,633copied values are grounded, only32of4,059files get an empty record, and Eq. (7)
holds on all ten committed hypergraphs and over all2,000edits of the stream (Table 19).
Table 16:Matched inputs, all rows (RQ8).Scored as in Table 1;+Hand+H+Ras in Table 2,
which omits the +H rows of OKF and StandardRAG. BaselineT 5is not rerun. Bold: best.
MethodT 1 T2 T3 T4 T5 T6 OKE
KnowledgeClaw 51.6 82.8 15.8 20.9 95.6 76.0 55.9
MemPro 11.4 78.2 13.5 8.2 78.8 69.5 41.3
+ H 28.3 78.8 13.2 7.3 78.8 73.8 45.9
+ H + R 45.2 81.0 13.5 7.3 78.8 75.8 50.7
OKF 27.2 68.5 8.8 17.3 54.4 67.3 38.5
+ H 32.2 71.8 9.0 17.3 54.4 70.6 41.3
+ H + R 43.7 78.2 8.8 17.3 54.4 72.6 45.6
StandardRAG 6.1 27.2 9.5 12.7 51.8 55.7 21.0
+ H 24.3 75.0 9.5 10.9 51.8 71.0 39.6
+ H + R 43.1 76.5 10.0 10.9 51.8 71.8 44.4
Table 17:Strict vs. lenient scorer (RQ8).S/L: exact set F1 / matcher (T 2,T5,T6score alike);
None: unparsed answers; Empty: empty answers onT 1,3,4; Kendallτ b= 0.90over34rows.
T1 T3 T4 OKE None Empty ans.
MethodS L S L S L S L (%) (%)
KnowledgeClaw 48.4 51.6 14.0 15.8 20.0 20.9 54.6 55.9 0.58.6
MemPro 9.9 11.4 1.5 13.5 2.7 8.2 37.9 41.3 5.7 21.5
OKF 24.2 27.2 1.8 8.8 2.7 17.3 35.7 38.5 1.3 7.5
LLM Wiki 17.6 20.0 1.0 6.5 0.0 0.0 33.2 35.3 1.6 14.8
Graphiti (Zep) 9.2 10.9 5.5 12.0 1.8 5.5 33.1 35.2 2.1 24.6
Omni-SimpleMem 8.5 10.3 0.8 10.2 2.7 10.9 31.7 34.5 3.5 20.7
Median of30baselines 3.6 4.9 1.1 6.4 0.5 2.7 17.4 18.7 2.2 20.7
Table 18:Matcher vs. LLM judge (RQ8).
Matcher, Judge: gold values matched by that
side only; Cohen’sκ= 0.91over all decisions.
SliceAns. Gold Agr. (%) Matcher Judge
All 275 585 95.7 9 16
T1 75 385 95.1 9 10
T3 100 100 96.0 0 4
T4 100 100 98.0 0 2Table 19:Grounding of copied values (RQ8).
Author, affiliation, identifier, venue, person, or-
ganization and place values of C1’s records.
ModalityCopied Grounded (%)
PDF 16,058 98.2
Text 9,682 97.5
Image, audio, video 3,893 96.7
All 29,633 97.8
F.8 CASE STUDY(RQ9)
Setting.The case is theT 2item whose media is a video, the150-second lecture
f75bfdd876eb3d22.mp4that C1 reads natively from its keyframes and audio; its candidates
describe four lectures of one MIT class by the same speaker, so only the topic tells them apart.
Results.Sixteen of the30baselines choose a sibling: seven (A) (LightRAG, HippoRAG 2, KAG,
HyperRAG, Mem0, Cognee, MMGraphRAG), seven (B) (StandardRAG, GraphRAG, iText2KG,
OntoKG, HyperGraphRAG, Hyper-KGGen, ATOM) and two (C) (NaiveGeneration, A-MEM).
KnowledgeClaw’s3,659-character card holds the titleCities and a Just Energy Transition, the peo-
ple discussed and a timestamped transcript stating the topic; the10,497characters of w/o C3 hold
identifiers of five other videos and no passage of the named one, and it picks the grid lecture (B).
36

Preprint.
G EXTENDEDRELATEDWORK
G.1 LLM-BASEDKNOWLEDGEENGINEERING
Construction and retrieval.Beyond passages (Lewis et al., 2020) and recursive summaries (Sarthi
et al., 2024), graph RAG adds entity communities (Edge et al., 2025), local-global retrieval (Guo
et al., 2025b) or passage-entity links (Guti ´errez et al., 2025). Construction iterates via entity reso-
lution (Lairgi et al., 2025), ontology routing (Li et al., 2026c), agent teams (Lu et al., 2025), task
rewards (Tsang et al., 2026), joint reading and building (Han & Cheng, 2026) or Wiki compila-
tion (Ran et al., 2026); retrieval uses logical forms (Liang et al., 2025), schema-bound agents (Dong
et al., 2026a), multi-channel search (Yang et al., 2025) or trained and executable policies (Luo et al.,
2026; Lee et al., 2026). Hypergraphs keepn-ary facts (Luo et al., 2025; Huang et al., 2026), fuse
entity and passage levels (Wang et al., 2026; Hu et al., 2026a; Khrouf et al., 2026) and plan or learn
traversal (Zai et al., 2026; Lien et al., 2026; Park et al., 2026b; An et al., 2026; Wu et al., 2026a).
Persistent and evolving memory.Agent memory spans extraction to evolution (Yang et al., 2026a):
tiered context (Packer et al., 2024), evolving notes (Xu et al., 2025), salient facts (Chhikara et al.,
2025), graph-LLM links (Markovic et al., 2025), a memory OS (Li et al., 2025), compression (Fang
et al., 2026; Liu et al., 2026b), fact/belief stores (Latimer et al., 2026), associative graphs (Jiang
et al., 2026; Ji et al., 2026) and hypergraphs (Zhou et al., 2026; Yue et al., 2026; Cui et al., 2026b).
Updates use (bi)temporal graphs (Rasmussen et al., 2025; Lairgi et al., 2026; Bao et al., 2026;
Wang, 2026), supersession and provenance (Yadav, 2026; Yacoubi et al., 2026), feedback (Fu et al.,
2026; Li et al., 2026d; Cui et al., 2026a) or evolving architectures (Zhang et al., 2026; Liu et al.,
2026d; Yang et al., 2026b), yet none defines how dependent indices follow a removed source file.
Compiled Wikis (Karpathy, 2026; Dong et al., 2026b) and agentic KE tools (OpenClaw contributors,
2026; Graphify contributors, 2026; Karpathy, 2026; Google Cloud, 2026) also persist knowledge.
Diagnostics separate failure stages (Liu et al., 2026e), ground memories (Suresh et al., 2026), test
revocation (Shen et al., 2026) and show that compression can lose what updates need (Zhang, 2026).
KnowledgeClaw adds a file-level contract: provenance-guided edits update all of its derived state.
G.2 MULTIMODALRETRIEVAL-AUGMENTEDGENERATION
Cross-modal construction and granularity.As surveyed by Abootorabi et al. (2025), systems
align vision to language to build KGs (Liu et al., 2025; Ruosch & Rossetto, 2026), link textual
and cross-modal graphs (Guo et al., 2025a; Wan & Yu, 2026; Yuan et al., 2026b) and keep visual
evidence end to end (Hsiao et al., 2026; He et al., 2026); others set granularity by pages (Cho
et al., 2024), chunk expansion (Hu et al., 2026b), grounded regions (Dai et al., 2026; Wu et al.,
2026b), reliability-routed subgraphs (Ali et al., 2026), citation-aware RL (Zhao et al., 2026) or
context folding (Tian et al., 2026), and TrioRAG (Rakshit et al., 2026) fuses signals without a graph.
Higher-order, memory and maintenance settings.Hypergraphs link text, images and tables (Chen
et al., 2026) or serve domains (Serna-Aguilera et al., 2026a;b; Ma et al., 2026a); audiovisual graphs
add long-video context and multi-hop selection (Ren et al., 2026; Zhu et al., 2026; Park et al., 2026a).
Memories coordinate typed stores (Wang & Chen, 2025; Liu et al., 2026c;a), keep multi-type video
memory (Yeo et al., 2026; Ma et al., 2026b; Li et al., 2026b) or budget evidence (Li et al., 2026a; Gao
et al., 2026). Closest are multi-hop and agentic hypergraph editing (Yuan et al., 2026a; Lin et al.,
2026) and higher-order memory lifecycles (Feng et al., 2026); we instead propagate file edits through
shared entities, cross-modal links and indices. Beyond personalization and cross-modal multi-hop
benchmarks (Hu et al., 2026c; Cai et al., 2026), OmniKE-Bench also tests post-edit maintenance.
H FUTUREWORK
Only three OmniKE-Bench items involve audio or video (Table 5), so the evaluation evidence for
audio and video is limited; adding such items is our next step. C1 can read spreadsheets, slides and
code with one more reading prompt (Appendix A.1) into the same typed record (Appendix A.2),
and richer record fields can join files at finer entities. C2 can serve knowledge bases shared by many
users and follow continuous edit streams. C3 can add tiers, such as timelines or tables, as further
routing labels (Appendix A.3) and learn routing from usage. At larger scale, the hypergraph can be
partitioned by team or domain, thereby serving enterprise, scientific and personal knowledge bases.
37