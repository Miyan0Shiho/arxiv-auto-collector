# EGCForest: Structured Agent Memory Organization for Enterprise Group Chats

**Authors**: Lingxi Cui, Jinhao Ruan, Wenqing Wei, Guo Yi, Li Rao, Ke Chen, Lidan Shou

**Published**: 2026-10-14 09:07:53

**PDF URL**: [https://arxiv.org/pdf/2610.19836v1](https://arxiv.org/pdf/2610.19836v1)

## Abstract
Enterprise group chats record project progress, decisions, task ownership, and other organizational information valuable to enterprise agents. Reliable retrieval is challenging because evidence spans interleaved threads, channels, and time, with meaning and authority conditioned on the speaker. Neither RAG's independent chunk retrieval nor memory systems' summary-based consolidation can reliably recover such distributed evidence. We present EGCForest, a novel multi-view hierarchical memory framework for enterprise group chats. EGCForest organizes an immutable message log into three complementary forests: a Channel forest for temporal workflows, an Actor forest for speaker-conditioned context, and an Artifact forest for deliverable lifecycles. Cross-view connections are enabled by a Placement Map and sparse structural bridges. Built on this memory backbone, EGCForest retrieves and assembles evidence by jointly leveraging message-level relevance, hierarchical context from forest summaries, and cross-view links, recovering distributed evidence. Experiments on three enterprise-chat benchmarks show that EGCForest consistently outperforms RAG and memory-system baselines, improving answer accuracy by up to 20.85 points over MemOS while supporting efficient online retrieval.

## Full Text


<!-- PDF content starts -->

EGCForest: Structured Agent Memory Organization for Enterprise Group
Chats
Lingxi Cui1,2, Jinhao Ruan1,2, Wenqing Wei1,2, Guo Yi3, Li Rao3, Ke Chen3, Lidan Shou1,2✉
1The State Key Laboratory of Blockchain and Data Security, Zhejiang University
2Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security
3Xiaohongshu
{cuilingxi.cs, capt, chenk, should}@zju.edu.cn
{zengzicheng, yutian5, liuxin6}@xiaohongshu.com
Abstract
Enterprise group chats record project progress,
decisions, task ownership, and other organiza-
tional information valuable to enterprise agents.
Reliable retrieval is challenging because ev-
idence spans interleaved threads, channels,
and time, with meaning and authority condi-
tioned on the speaker. Neither RAG’s inde-
pendent chunk retrieval nor memory systems’
summary-based consolidation can reliably re-
cover such distributed evidence. We present
EGCForest , a novel multi-view hierarchical
memory framework for enterprise group chats.
EGCForest organizes an immutable message
log into three complementary forests: a Chan-
nel forest for temporal workflows, an Actor
forest for speaker-conditioned context, and an
Artifact forest for deliverable lifecycles. Cross-
view connections are enabled by a Placement
Map and sparse structural bridges. Built on
this memory backbone, EGCForest retrieves
and assembles evidence by jointly leveraging
message-level relevance, hierarchical context
from forest summaries, and cross-view links,
recovering distributed evidence. Experiments
on three enterprise-chat benchmarks show that
EGCForest consistently outperforms RAG and
memory-system baselines, improving answer
accuracy by up to 20.85 points over MemOS
while supporting efficient online retrieval.
The code is available at https://anonymous.
4open.science/r/EGCForest-MAD12/.
1 Introduction
Enterprise group chats have become the default
medium for daily collaboration inside modern or-
ganizations (Yang et al., 2026). Beyond casual
exchanges, they carry a substantial amount of task-
related knowledge such as project progress, own-
ership assignments, and design rationales. Recent
progress in agent systems has raised the demand for
reliable enterprise knowledge sources (Sun et al.,
2026), and group-chat logs provide large-scale,
Group : Backend
09:30 Bob Owner
Need QA  sign-of f before release.09:21CCarol QA
Lunch at 12?09:18DDave Engineer
Staging looks good.09:10AAlice PM
PR for                  is merged. DOC-042 @Bob
Lily10:31LQA
passed QA.Group: Release
10:02CCarol QA
Check                 . @Lily DOC-042
DOC-042
10:40BBob Owner
Approved. Release                  today . DOC-042Was DOC-042 released,
and who approved it?
Yes. DOC-042 was 
released on Dec 03,
after Bob's approval.
Knownledge in 
Enterprise Group Chats
Project progress
Ownership assignments
Final decisions and rationale
......
Challenge 1: multi-level
conversational structure1
......Evidence across channels
Several concurrent topics
Reply chains
Challenge 2: identity
conditioned semantics2
Speaker-dependent authority
Peer-specific jargon ......① Interleaved topics② Peer-speciﬁc jargon
② Different authority
 ① Reply chain 
BFigure 1: Enterprise group chats contain valuable orga-
nizational knowledge, but deploying them as reliable
agent memory is challenging.
fine-grained evidence of organizational activity. Ex-
isting research on group chats, in contrast, has
largely focused onparticipatingin the conversa-
tion (Sapkota et al., 2025), for instance predicting
the next speaker, recognizing the addressee, or dis-
entangling interleaved threads, rather than onex-
tractingthe organizational knowledge that these
conversations contain. Turning raw chat logs into
an external memory that an agent can query accu-
rately, with traceable evidence back to the original
messages, therefore remains an open problem.
Two properties of enterprise group chats make
this task particularly challenging. First, reliable
memory for enterprise group chats must preserve
theirmulti-level conversational structure. Mes-
sages are temporally ordered, but topics interleave
within channels, reply chains mix substantive and
tangential exchanges, and decision evidence may
span multiple threads, channels, and time peri-
ods. As shown in Figure 1, answering an artifact-
status question requires tracing its lifecycle across
two channels and four participants. Retrieval-
augmented generation (RAG) systems typically
treat messages as independent chunks, favoring
arXiv:2610.19836v1  [cs.CL]  14 Oct 2026

locally similar matches while missing the full ev-
idence chain (Karpukhin et al., 2020). Memory
systems instead compress chats into facts or sum-
maries, obscuring or conflating evidence from dis-
tinct threads and decisions (Chhikara et al., 2025).
Second, capturingspeaker-conditioned seman-
tics is challenging yet essential in enterprise group
chats. Unlike dyadic conversations, single-user
agent interactions, or participant-agnostic docu-
ments, group-chat participants’ roles, responsibil-
ities, and task relations shape message interpreta-
tion. The same claim may carry different authority
across speakers (e.g., a project owner versus an
intern), while teams may refer to the same artifact
differently. RAG systems typically represent mes-
sages as plain text and use speaker identity only
as a metadata filter, thereby overlooking speaker-
conditioned meaning and evidential relevance (Ma-
harana et al., 2024). Memory systems often encode
identity as a namespace or auxiliary tag, which can
similarly conflate claims from participants with
different responsibilities (Packer et al., 2023).
We address both challenges with EGCForest , a
structured external memory organized as three com-
plementary forests over a shared message log. The
Channelforest partitions each group chat into hi-
erarchical temporal intervals, preserving local con-
text and topic evolution. TheActorforest cen-
ters each participant, with separate branches for
messages authored by that participant and mes-
sages that mention them, making identity a first-
class semantic axis rather than a passive meta-
data field. TheArtifactforest organizes mes-
sages around deliverables and represents their life-
cycle across group chats. These forests are ex-
plicitly interconnected. When a message appears
in multiple forests, aPlacement Maplinks its
view-specific placements to a single immutable log
entry, enabling direct cross-view correspondence
without content duplication. EGCForest further
adds sparseBridgesthat connect distinct messages
through native reply links and shared salient iden-
tifiers. Built on this backbone, a four-stage re-
trieval pipeline combines view-aware leaf recall,
summary-guided supplementation, structural com-
pletion, and budget-aware assembly.
Contributions.We make the following contri-
butions:(1) A novel memory framework.We
proposeEGCForest , a structured memory frame-
work featuring three complementary forests to or-
ganize scattered knowledge from enterprise groupchats while preserving conversational structure,
speaker context, and traceability to original mes-
sages.(2) Three-view forest architecture.We
construct complementary Channel, Actor, and
Artifact forests that capture temporal workflow,
speaker-conditioned context, and artifact lifecy-
cles, respectively, with a Placement Map and sparse
Bridges enabling efficient cross-view navigation.
(3) Structure-aware evidence retrieval.We com-
bine view-aware tree routing and evidence expan-
sion through summaries and cross-view links to
recover distributed evidence.(4) Empirical valida-
tion.Across three benchmarks, EGCForest deliv-
ers consistent improvements over strong RAG and
memory-system baselines, outperforming MemOS
by up to 20.85 points.
2 Preliminaries
2.1 Problem Formulation
We consider an enterprise workspace with a set of
channels1Cand a set of users U. A chat history
is an ordered, immutable sequence of messages
M={m 1, m2, . . .} , where each message is a
tuple m= (t m, um, cm, rm, xm)with timestamp
tm, sender um∈ U, channel cm∈ C, optional
reply targetr m∈ M ∪ {∅}, and raw textx m.
Our goal is to organize Minto a structured exter-
nal memory Kthat an agent can query. As memory
quality is hard to assess directly, we use question
answering as a measurable proxy for it. Concretely,
given a natural-language query qposed by an ask-
ing user uq∈ U, the task is to retrieve a compact
evidence set Eq=R(q, u q;K)⊆ M via memory
K, and feed Eqtogether with qto an answer agent
to produce an answera.
2.2 Related Work
Agent memory systemsmaintain persistent mem-
ory across sessions with a single user, typically by
compressing conversations into facts, summaries,
or memory items (Xu et al., 2026; Kang et al., 2025;
Latimer et al., 2025). MemGPT (Packer et al.,
2023) treats the context window as limited main
memory and swaps information with an external
store, whereas Mem0 (Chhikara et al., 2025) in-
crementally adds, merges, or replaces memory en-
tries. Although effective for single-user assistants,
these compression-oriented designs can obscure
1We usechannelto refer to a persistent, multi-party group
chat, following common enterprise chat platform terminology
(e.g., Slack, Microsoft Teams).

Enterprise Group ChatImmutable Message Log
ID: 095  User:  Carol   Group:  Release
T ime: 25-12-03 10:02
T ext: @Lily PR Check DOC-042.ID: 096   User:  Cici   Group:  Market
T ime: 25-12-03 10:15
T ext: @Jimmy New customer demands.ID: 097   User:  Lily   Group:  Release
T ime: 25-12-03 10:31
T ext: DOC-042 passed QA.......Three Complementary Forests
Channel Forest
Release
... 095 097 ...Actor Forest
Bob
authored mentioned
... ... ... ...Artifact Forest
DOC-042
... ... 095 097...
...... ... DOC-042 QA...
...Ofﬂine Memory Construction
......
Merge QA...Online Retrieval
BM25
VecView-aware 
Leaf Retrieval
Query...
Message Log
view
routing Top 
Leaves Channel Actor ArtifactSummary-guided
Supplementation
Summary Node
Supplementation Leaves
Structural
Completion
Top 
LeavesChannel Actor Artifact
Placement MapBridgeBudget-aware
Assembly
Candidate 
LeavesDeduplicate
&RankToken Budget
Assembled context AgentS1 S2
S3 S4Top 
LeavesFigure 2: Overview ofEGCForest: offline three-view memory construction and online four-stage retrieval.
group-chat structure by treating speaker identity as
metadata, merging conflicting claims, and limiting
access to raw-message evidence.
Retrieval-augmented generation (RAG)typ-
ically organizes text corpora as flat text chunks,
hierarchical summaries, or graph-structured rep-
resentations. BM25 (Robertson and Zaragoza,
2009) and dense retrieval (Karpukhin et al., 2020)
rank individual chunks by lexical or semantic
relevance. Hierarchical methods such as RAP-
TOR (Sarthi et al., 2024) recursively cluster and
summarize chunks or retrieval across abstraction
levels. GraphRAG (Edge et al., 2024), Ligh-
tRAG (Guo et al., 2025), and HippoRAG (Gutier-
rez et al., 2024) organize entities and relations for
retrieval, with GraphRAG additionally summariz-
ing graph communities. These approaches primar-
ily infer structure from text, whereas group chats
expose native structural signals, including reply
links, participant roles, and cross-channel member-
ship, that are not fully captured by text alone.
Multi-party conversation understanding
spans three directions: inferring participant states
(e.g., emotion recognition (Ghosal et al., 2019)),
modeling discourse structure (e.g., conversation
disentanglement (Kummerfeld et al., 2019)), and
summarizing group interactions (e.g., dialogue
summarization (Gliwa et al., 2019), decision
detection (Addlesee et al., 2023)). Recent work
couples group chats with LLM agents for ongoing
participation, for example by jointly generating
responses and predicting the next speaker (Wang
et al., 2025) or orchestrating multi-party tool
use (Houde et al., 2025). In contrast, we treat
completed group chats as a knowledge source and
organize them into structured, evidence-grounded
memory for downstream agents.3 Methodology
EGCForest organizes enterprise group chats into
three complementary forests over a shared, im-
mutable message log, preserving structural signals
lost when chats are compressed into summaries
or treated as independent chunks (Figure 2). Of-
fline, it constructs aChannel forestfor temporal
workflows, anActor forestfor speaker-conditioned
context, and anArtifact forestfor cross-channel
deliverable lifecycles. Online, four stages retrieve
message anchors, supplement them through sum-
maries, recover linked evidence, and assemble a
token-bounded context for the answer agent.
3.1 Structured Memory Construction
3.1.1 Three-View Forest Architecture
Immutable Message Log.The three forests
share an immutable log of verbatim messages.
Each forest leaf references a single log entry rather
than copying its content, preserving traceability
to the original evidence and avoiding duplication
across views. For each message m, the log stores
its timestamp tm, sender um, channel cm, optional
reply target rm, and raw text xm. We further derive
a mention set mentions(m)⊆ U and structured
identifiers, such as document codes, from xmusing
lightweight regular expressions.
Channel forest.The Channel forest organizes
each channel as a temporal hierarchy, making con-
versational context retrievable at multiple temporal
scales. For each channel c∈ C , we construct a
balanced k-ary tree Tcwhose leaves are references
to the immutable log entries of messages in c, or-
dered chronologically. Every kconsecutive leaves
or child nodes are recursively grouped into a parent
until a single root remains. Each internal node sum-

marizes its contiguous time interval, focusing on
participants, workflows, decisions, commitments,
blockers, and unresolved issues. Unlike semantic
clustering hierarchies, this construction preserves
temporal contiguity and event order, supporting re-
trieval of project progress and decision evolution.
The running example summarizes the #backend
workflow from merge to release, preserving trace-
ability to message-level evidence, including the QA
requirement that governs the transition.
/* In all running examples, [R] denotes a root node, [N] an
internal node with an LLM-generated summary, and [L] a leaf
node that references an immutable log entry. */
[R]#backend [channel]
+–[N]Dec 1: merge and release gate
| +–[L]m 091: @Bob, PR for DOC-042 is merged.
| +–[L]m 092: Staging looks good.
| +–[L]m 093: Lunch at 12?
| +–[L]m 094: Need QA sign-off before release.
+–[N]Dec 3: release confirmation
+–[L]m 099: Released after Bob’s approval.
Actor forest.The Actor forest addresses speaker-
conditioned semantics by organizing messages
around each participant, enabling cross-channel
tracing of an employee’s responsibilities, commit-
ments, interactions, and reported context. For each
participant u∈ U , we construct an actor tree Tu
whose root separates messages authored by ufrom
messages that explicitly mention u. Within each
branch, messages are partitioned by channel, or-
dered chronologically, and recursively grouped us-
ing the same k-ary construction as the Channel
forest. Internal-node summaries emphasize the par-
ticipant’s speech acts, responsibilities, expertise,
ownership, and reporting relations. The running
example aggregates Bob’s authored and mentioned
messages, forming a participant-centered record
of his project involvement, responsibilities, and
interaction patterns.
[R]Bob [actor]
+–[N]authored-by
| +–[N]#backend
| | +–[N]Dec 1: release requirement
| | +–[L]m 094: Need QA sign-off before release.
| +–[N]#release
| +–[N]Dec 3: release approval
| +–[L]m 098: Approved. Release DOC-042 today.
+–[N]mentioned-in
+–[N]#backend
+–[N]Dec 1–3: merged and released
+–[L]m 091: @Bob, PR for DOC-042 is merged.
+–[L]m 099: Released after Bob’s approval.
Artifact forest.The Artifact forest organizes
messages around enterprise deliverables, enabling
cross-channel tracing of an artifact’s lifecycle, sta-
tus, and ownership. We identify artifacts offlineby clustering messages that discuss the same ar-
tifact. We extract artifact-like identifiers, such as
document codes, from messages using lightweight
rules and cluster the associated messages using
farthest-first clustering. Candidate clusters with
highly similar centroids are merged automatically,
while an LLM judge resolves borderline cases. We
retain clusters with sufficient lifecycle transitions.
For each retained artifact a, we construct a tree
Tawhose leaves reference its member messages
pooled across channels and ordered chronologi-
cally. Messages are recursively grouped using
the same k-ary construction as the other forests.
Internal-node summaries emphasize lifecycle tran-
sitions, responsible participants, and cross-channel
status changes. The running example links merge,
QA, approval, and release evidence from two chan-
nels into a traceable lifecycle for DOC-042.
[R]DOC-042 [artifact]
+–[N]Dec 1: merged and awaiting QA
| +–[L]m 091: @Bob, PR for DOC-042 is merged.
| +–[L]m 094: Need QA sign-off before release.
+–[N]Dec 3: QA passed, approved, released
+–[L]m 095: @Lily PR Check DOC-042.
+–[L]m 097: DOC-042 passed QA.
+–[L]m 098: Approved. Release DOC-042 today.
+–[L]m 099: Released after Bob’s approval.
3.1.2 Cross-View Connections
To surface cross-view evidence efficiently at query
time, we build cross-view connections through a
Placement Mapand a sparse set ofBridges.
Placement Map.Each forest indexes refer-
ences to messages in the same immutable mes-
sage log under a different routing rule. Conse-
quently, a message may have placements in multi-
ple trees across views, providing a direct struc-
tural connection between them. To make this
link efficiently queryable, we precompute for ev-
ery message mthe placement table Pl(m) =
{(view,tree_id,parent_id,leaf_pos)} indexed by
msg_id in a hash map. The Placement Map sup-
ports three structural retrieval primitives through di-
rect lookup rather than tree traversal. trees_of(m)
returns the trees containing m.window(m, T, w)
returns the wmessages adjacent to min the leaf
order of tree T.cotenants(m) returns the sib-
ling leaves that share a parent node with macross
its placements. These primitives make cross-view
placements and local structural neighborhoods di-
rectly addressable, thereby supporting the multi-
view joins used later in retrieval.

Bridges.Whereas the Placement Map connects
multiple placements of the same message, bridges
connect distinct messages through strong structural
or evidential relations. We construct two bridge
types from native reply links and the leaf-level
identifiers defined in Section 3.1.1.Reply bridges
connect a message mwith its explicit reply target
rmwhen rm̸=∅.Identifier bridgesconnect mes-
sages sharing a structured identifier, such as a doc-
ument code or ticket ID. To avoid dense all-pairs
connectivity, each message links to at most bsame-
identifier messages ranked by temporal proximity.
These sparse bridges connect local conversational
dependencies and cross-channel artifact evidence,
supporting the bridge-hop operator in retrieval.
3.2 Forest Retrieval
Given a query q, retrieval produces a compact ev-
idence set Equnder a fixed token budget through
four stages. S1retrieves primary evidence via
view-aware leaf recall, S2supplements it with
summary-guided nodes, S3completes the chain
through bounded cross-view operations, and S4
assembles the final context. Before S1, we parse
qinto a lightweight query specification using de-
terministic rules that require no LLM calls. The
specification records the asker and mentioned enti-
ties, including participants, channels, and artifact
identifiers, to guide subsequent view selection. We
detailS 1throughS 4below.
3.2.1S 1View-aware Leaf Retrieval
S1combines global message seeding with view-
aware tree routing. Global retrieval identifies fine-
grained message-level anchors, while tree routing
lifts these anchors to coarser structural contexts that
capture broader workflows, participant histories,
and artifact lifecycles. The selected trees define the
search space for subsequent summary expansion
and cross-view completion.
Candidate seeding.Given a query q, we retrieve
messages using BM25 for lexical matching and
dense retrieval for semantic matching. We fuse
the resulting rankings through reciprocal rank fu-
sion (RRF), which combines ranks rather than raw
scores with incompatible scales:
shyb(m, q) =1
η+ rank bm25(m)+λ d·1
η+ rank den(m),
(1)
where ηis a smoothing constant and λdweights
dense retrieval. The top- Kmessages undershyb(m, q) form the anchor pool. Rather than be-
ing returned directly, these anchors are projected
to their forest placements to identify relevant trees
and select leaf evidence within each view.
View-aware routing.Using the anchor pool, we
score each candidate tree with three complemen-
tary signals.Leaf2treeprojects each anchor to its
host trees through Pl(·) and aggregates weighted
anchor votes.Rootmatches qagainst each tree’s
root summary.Collapsedcaptures locally relevant
content that may be obscured by a tree’s coarse
root summary. It scans internal-node summaries by
cosine similarity and assigns each tree the score of
its best-matching summary. We rank each signal
within each view rather than globally, and fuse the
resulting within-view ranks using RRF. We select
top-ranked trees for subsequent stages and retain
anchor messages as primary leaf evidence under
fixed per-view quotas, preventing larger views from
crowding out complementary ones.
3.2.2S 2Summary-guided Supplementation
S2supplements the message-level evidence re-
turned by S1with summary-level retrieval over the
trees selected in S1. A message may weakly match
qin isolation even when its surrounding segment is
highly relevant. We therefore score internal-node
summaries against q, recovering such messages and
providing layered context for the answer model.
For each internal node nwith summary embedding
sn, we compute cos(q,s n). To avoid overlap, we
retain only the highest-scoring node on each root-
to-leaf path and expand it into a bounded number
of descendant leaves. These leaves are appended
with their ancestor summary chain asnavigation
headers. We place S2leaves after S1messages
rather than merging raw scores, as RRF and cosine
similarity are not directly comparable.
3.2.3S 3Structural Completion
S3recovers cross-view evidence via bounded
Placement Map joins and bridge hops.
Placement Map join.To avoid unnecessary ex-
pansion, S3invokes a Placement Map join only
when the query mentions at least two resolvable
entities. It first locates an evidence message mA
for one entity AfromS1, then uses Pl(m A)to en-
ter a related placement in the selected view. For
queries with an explicit temporal cue, candidates
are restricted to leaves occurring before or after
mA. Candidates are ranked by similarity to the

remaining entityBandq, and the top candidate is
retained only if its score exceeds a threshold.
Bridge hop. S3expands high-confidence leaves
fromS1andS2through the sparse bridges defined
in Section 3.1.2. For each candidate leaf m, we
inspect its incident reply and identifier bridges, and
add each reached message m′to the candidate pool
as a bridge anchor. For each m,S3follows at most
one bridge into a tree outside the evidence pool,
choosing the endpoint most relevant to q. We then
inspect a bounded window around the selected end-
point in that tree and adds the most query-relevant
neighboring leaf. A small hop budget prevents
uncontrolled expansion while recovering reply de-
pendencies and cross-channel artifact evidence.
3.2.4S 4Budget-aware Assembly
S4merges and deduplicates candidates from S1,
S2, andS3. Since scores are not comparable across
stages, we preserve within-stage ranks and add can-
didates in the order S1,S2, and S3. Each stage is
subject to a fixed allocation, and selection stops
when the token budget is reached. Selected mes-
sages are rendered verbatim with lightweight meta-
data (channel, sender, lifecycle markers). Summary
headers from S2are attributed to their originating
nodes and capped at a small fraction of the bud-
get, allowing the answer model to distinguish raw
evidence from generated navigation. This design
preserves traceability to the original chat log.
4 Experiments
We evaluate EGCForest through four key ques-
tions:Q1 (Section 4.2):Does EGCForest out-
perform RAG and memory-system baselines?Q2
(Section 4.3):Do the Channel, Actor, and Arti-
fact views provide complementary benefits?Q3
(Section 4.4):Does EGCForest retrieve more rele-
vant evidence than RAG baselines, and how does
each retrieval stage ( S1–S4) contribute?Q4 (Sec-
tion 4.5):How efficient is EGCForest in terms
of offline construction cost, storage footprint, and
online retrieval latency?
4.1 Experimental Setup
Datasets.We employEverMemBench(Hu
et al., 2026),GroupMemBench(Yang et al.,
2026), and a real-world internal enterprise bench-
markCompany. Details are in Appendix A.2.
Baselines.RAG baselines include BM25 (Robert-
son and Zaragoza, 2009), Dense (Karpukhinet al., 2020), their reciprocal-rank-fusion vari-
antBM25+Dense (Cormack et al., 2009), and
GraphRAG (Edge et al., 2024). Agent memory-
system baselines include MemoBase (Team, 2025),
Mem0 (Chhikara et al., 2025), A-Mem (Xu et al.,
2026), andMemOS(Li et al., 2025).
Metrics.We reportAnswer Accuracy, mea-
sured against gold answers LLMs. ForCompany,
we additionally reportPrecision@ kandRecall@ k
against gold evidence messages.
Implementation.Following the official protocol,
we use GPT-5 as the answer model onGroupMem-
Benchand GPT-4.1-mini onEverMemBench.
We also use GPT-4.1-mini onCompany. We use
the frozen Qwen3-Embedding-0.6B encoder, set
the branching factor to k= 16 , and retrieve under
a 3,000-token budget. Details are in Appendix A.3.
4.2 Q1: Main Results
Overall.Table 1 shows that EGCForest achieves
the highest average accuracy on all three bench-
marks, exceeding the strongest baseline by 4.77,
8.59, and 12.61 percentage points, respectively.
These consistent gains demonstrate its robustness
across diverse dialogue settings and question types.
Dataset and category analysis. EGCForest
ranks first or second on most subcategories. On
EverMemBench, which contains long-horizon,
multi-channel project discussions, EGCForest per-
forms especially well on Fine-grained Recall and
Memory Awareness, which requirelinking evi-
dence across channels and time. Profile Un-
derstanding remains difficult because Style, Skill,
and Role must be inferred from diffuse behav-
ioral patterns rather than explicit event-level ev-
idence. OnGroupMemBench, which empha-
sizes threaded conversations and asker-conditioned
queries,EGCForest achieves 55.10 on Implicit and
40.57 on Ambiguity. Implicit questions require
tracing first-person references through the asker’s
history, whereas Ambiguity questions require re-
solving terminology across speakers. These re-
sults are consistent with the benefit of the Actor
forest for retrievingasker-conditioned context.
EGCForest is less dominant on Abstention ( 81.29 ),
where recall-oriented retrieval may surface plausi-
ble but non-supporting messages. OnCompany,
EGCForest reaches 61.90 on Temporal and 64.06
on Update, where the Channel forest preserves
workflow chronology and the Artifact forest links

EverMemBench
MethodFine-grained Recall Memory Awareness Profile UnderstandingAvg.
Single Multi Temp. Const. Proact. Update Style Skill Role
BM25 87.45 5.62 11.33 70.60 62.20 37.28 30.94 43.44 53.63 44.72
Dense 86.23 6.83 14.33 79.60 70.2059.93 31.49 45.70 55.24 49.95
BM25+Dense 81.38 4.42 14.00 75.20 68.40 49.48 32.6046.61 57.2647.70
GraphRAG 69.64 5.22 14.33 79.00 49.00 52.9638.1245.70 50.00 44.89
MemoBase 60.09 12.85 18.00 64.68 36.77 30.60 17.05 29.59 38.78 34.27
Mem0 55.40 11.24 6.33 66.17 52.46 51.87 22.73 31.36 36.22 37.09
A-Mem 88.26 6.83 11.67 77.20 62.20 48.78 34.81 46.6150.00 47.37
MemOS 71.3618.8815.67 69.90 51.99 45.15 28.98 32.54 48.47 42.55
OURS89.8817.27 33.33 81.00 70.20 61.32 38.1245.70 55.65 54.72
GroupMemBench
Method Multi-Hop Update Ambiguity Implicit Temporal Abstention Avg.
BM25 40.66 28.04 16.04 46.94 56.17 83.45 47.11
Dense 35.16 24.30 22.64 44.90 43.83 77.70 42.28BM25+Dense 32.42 22.43 23.58 30.61 43.8384.1741.74GraphRAG 17.03 19.63 17.92 20.41 6.79 76.26 26.58
MemoBase 15.93 3.74 15.09 20.41 20.99 82.73 27.92Mem0 23.63 14.02 16.04 24.49 7.41 79.86 28.19A-Mem 31.87 27.10 26.42 36.73 21.60 78.42 37.18
MemOS 28.57 21.50 24.53 18.37 25.93 82.01 35.70
OURS47.25 32.71 40.57 55.10 68.5281.2955.70
Company Dataset
MethodFine-grained Recall Memory Awareness Profile UnderstandingAvg.
Single Multi-hop Temp. Const. Update Implicit Abstention Style Skill Role Ambiguity
BM25 58.82 44.87 41.27 75.00 42.19 45.05 50.00 2.13 8.33 5.00 75.00 37.45
Dense 57.98 48.72 47.62 66.25 35.94 45.05 62.50 1.06 7.14 2.50 80.00 36.58
BM25+Dense 62.18 53.85 44.44 72.50 48.44 55.86 37.50 4.26 5.95 2.5085.0040.70
GraphRAG 56.30 55.13 33.33 57.50 34.38 48.65 37.50 4.26 5.95 7.50 75.00 35.71
MemoBase 36.97 34.62 23.81 37.50 29.69 34.23 12.50 0.00 5.95 3.75 60.00 24.22
Mem0 52.94 39.74 33.33 63.75 34.38 42.34 37.50 2.13 10.71 5.00 70.00 33.33
A-Mem 59.66 52.56 47.62 67.50 39.06 48.65 62.50 3.19 7.14 6.25 75.00 38.58
MemOS 56.30 33.33 36.51 70.00 29.69 42.34 12.50 1.06 5.95 2.50 65.00 32.46
OURS78.15 65.38 61.90 86.25 64.06 73.87 75.00 7.45 14.29 12.50 85.00 53.31
Table 1: Main results on three enterprise group benchmarks. We use the official taxonomies forEverMemBench
andGroupMemBench, and a unified group_chat taxonomy forCompany. Bold: best; underlined: second-best.
lifecycle evidence across channels.
Understanding the baseline gap.Memory sys-
tems may merge claims across speakers or threads
when consolidating messages into facts, profiles,
or summaries, obscuring attribution, updates, and
verbatim evidence. OnCompany, MemOS ob-
tains only 33.33/36.51 on Multi-hop/Temporal, ver-
sus 65.38/61.90 for EGCForest . Raw-message
RAG preserves message details but ranks messages
independently. OnCompany, BM25+Dense ob-
tains 53.85/44.44 on Multi-hop/ Temporal ques-
tions requiring evidence across channels and time.
GraphRAG further abstracts messages into entity-
relation graphs and community summaries, diluting
reply, attribution, and workflow signals.4.3 Q2: Component Study forEGCForest
Figure 3 evaluates the contribution of the three
views2. The full model (V4) achieves the best
weighted average on both datasets. Compared with
V1, V2 improves Fine-grained Recall and Memory
Awareness on the Company dataset by 8.08 and
5.32 points, respectively, showing the benefit of
speaker-centered context. Compared with V1, V3
improves Fine-grained Recall by 0.87/2.31 points
and Profile Understanding by 6.61/2.88 points
on GroupMemBench/Company, respectively. By
grouping messages about the same deliverable, the
Artifact view exposes cross-channel lifecycle and
2On GroupMemBench, Fine-grained Recall (Multi-hop +
Temporal),Memory Awareness (Update + Implicit + Absten-
tion), and Profile Understanding(Ambiguity).

53.20
56.97
54.07
57.2757.97
57.97
55.93
59.3233.96
38.68
40.57
40.5752.35
54.77
52.89
55.70V1
V2
V3
V457.69
65.77
60.00
70.3865.40
70.72
63.88
75.2813.67
15.47
16.55
16.5544.94
49.94
46.19
53.31(a) GroupMemBench
Fine-grainedMemory Profile Avg.(b) Company Dataset
Fine-grainedMemory Profile Avg.Figure 3: Performance of different view combinations
onGroupMemBenchandCompany. V1: Channel;
V2: Channel+Actor; V3: Channel+Artifact; V4: Full.
ownership evidence useful for evidence recall and
profile inference. Finally, V4 outperforms V2 by
0.93 points on GroupMemBench and 3.37 points on
the Company dataset, confirming that Artifact evi-
dence complements the Channel and Actor views.
4.4 Q3: Retrieval Performance
4.4.1 Comparison with Standard Retrieval
EGCForestachieves the strongest overall retrieval
performance, attaining the highest Precision@ k
at every budget while maintaining competitive
Recall@ k3. This result is consistent with its multi-
view organization and structural retrieval pipeline,
which consider temporal, participant, and artifact
context in addition to message-level relevance.
GraphRAG obtains the highest Recall@20 but at
substantially lower precision, suggesting noisier
expansion. Lexical and dense baselines improve
more slowly as the retrieval budget increases, likely
because they rank messages independently and can-
not exploit multi-view structural relations.
20 30 40 50k468Precision (%)
8.20
6.67
5.63
4.82(a) Precision@k
20 30 40 50k405060Recall (%)
45.8153.0757.8460.79(b) Recall@k
BM25 Vec BM25+Vec GraphRAG Ours
Figure 4: Retrieval Performance onCompany.
4.4.2 Stage-wise Contribution
Table 2 shows the cumulative contribution of each
retrieval stage under the same context budget. S1
reaches 40.3 R@20, suggesting that hybrid mes-
sage anchors and routing through complementary
Channel, Actor, and Artifact views recover much
3Precision@ kis inherently low: with 3.6 gold messages
per question,P@20 is upper-bounded at roughly 18%.of the directly relevant evidence. S2 adds summary-
guided supplementation and yields modest gains
in R@20 and accuracy ( 40.3→40.8 and45.4→
45.8), indicating that S1 covers much of the di-
rectly relevant evidence. S3 improves P@20 from
7.1 to 8.0 and R@20 from 40.8 to 44.5 through
Placement Map joins and bridge hops to recover
evidence distributed across views. S4 yields small
retrieval gains but improves accuracy from 47.9 to
53.3, showing that effective evidence selection and
organization are crucial for downstream answering.
Variant P@20 R@20 Acc.
S1: View-aware Leaf Retrieval 7.0 40.3 45.4
+ S2: Summary-guided Supplementation 7.1 40.8 45.8
+ S3: Structural Completion 8.0 44.5 47.9
Full (+ S4: Budget-aware Assembly) 8.2 45.8 53.3
Table 2: Stage-wise retrieval ablation ofEGCForest.
4.5 Q4: Efficiency
EGCForest incurs moderate offline cost. Its three-
forest construction is slower than lexical and dense
indexing but faster than GraphRAG, MemoBase,
and Mem0, with a 206 MB storage footprint. At
query time, EGCForest achieves the fastest re-
trieval latency at 65 ms per query, enabled by
cached node summaries and constant-time Place-
ment Map lookups. Its answer-token usage is com-
parable to other memory systems, averaging 2,365
input and 1,300 output tokens per query.
MethodOffline Construction Online Retrieval Online Answer
Build Time Storage Ret. Time / Q Answer Tok. / Q
BM25 0.6 s – 253 ms 1454 / 103
Vec 3.4 s 83 MB 72 ms 1462 / 112
BM25 + Vec 8.9 s 83 MB 710 ms 1452 / 106
GraphRAG 38200 s 641 MB 2107 ms 3936 / 1283
MemoBase 7994 s – 1634 ms 3153 / 1422
Mem0 7706 s 1.4 GB 15339 ms 492 / 1249
A-Mem 434 s 24 MB 97 ms 2305 / 1311
MemOS 1401 s – 573 ms 632 / 1207
Ours 1842 s 206 MB 65 ms 2365 / 1300
Table 3: Efficiency comparison on GroupMemBench.
5 Conclusion
As an initial exploration of agent memory for enter-
prise group chats, EGCForest organizes immutable
message evidence through complementary Chan-
nel, Actor, and Artifact forests. This structured-
memory paradigm enables agents to navigate trace-
able organizational evidence across time, partici-
pants, and artifacts. Experiments on three bench-
marks demonstrate consistent improvements across
diverse question categories.

Limitations
EGCForest is evaluated through question answer-
ing on three enterprise-chat benchmarks. This
setting enables controlled comparison of memory
quality, while leaving its benefits for broader agent
workflows, such as task coordination and long-term
planning, to be explored. Moreover, EGCForest ’s
offline construction time remains manageable at the
evaluated scale, making periodic memory updates
through full reconstruction feasible. For larger or
more frequently updated workspaces, incremental
forest construction and selective updates to sum-
maries and cross-view connections are promising
directions for balancing memory freshness with
maintenance cost. Finally, our current formula-
tion focuses on text messages and their associ-
ated metadata. Images, document attachments, and
voice messages can contain relevant organizational
knowledge beyond the message text. Future work
could extend EGCForest to integrate these sources
into a unified multimodal memory while preserving
traceability to the original evidence.
References
Angus Addlesee, Weronika Siei ´nska, Nancie Gunson,
Daniel Hernandez Garcia, Christian Dondrup, and
Oliver Lemon. 2023. Multi-party goal tracking with
LLMs: Comparing pre-training, fine-tuning, and
prompt engineering. InProceedings of the 24th An-
nual Meeting of the Special Interest Group on Dis-
course and Dialogue, pages 229–241.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet
Singh, and Deshraj Yadav. 2025. Mem0: Building
production-ready ai agents with scalable long-term
memory.arXiv preprint arXiv:2504.19413.
Gordon V Cormack, Charles LA Clarke, and Stefan
Buettcher. 2009. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. In
Proceedings of the 32nd international ACM SIGIR
conference on Research and development in informa-
tion retrieval, pages 758–759.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Deepanway Ghosal, Navonil Majumder, Soujanya Poria,
Niyati Chhaya, and Alexander Gelbukh. 2019. Dia-
logueGCN: A graph convolutional neural network for
emotion recognition in conversation. InProceedings
of the 2019 Conference on Empirical Methods in Nat-
ural Language Processing and the 9th InternationalJoint Conference on Natural Language Processing
(EMNLP-IJCNLP), pages 154–164.
Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Alek-
sander Wawer. 2019. SAMSum corpus: A human-
annotated dialogue dataset for abstractive summariza-
tion. InProceedings of the 2nd Workshop on New
Frontiers in Summarization, pages 70–79. Associa-
tion for Computational Linguistics.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025, pages
10746–10761.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. HippoRAG: Neu-
robiologically inspired long-term memory for large
language models.
Stephanie Houde, Kristina Brimijoin, Michael Muller,
Steven I Ross, Dario Andres Silva Moran, Gabriel En-
rique Gonzalez, Siya Kunde, Morgan A Foreman,
and Justin D Weisz. 2025. Controlling ai agent par-
ticipation in group conversations: A human-centered
approach. InProceedings of the 30th International
Conference on Intelligent User Interfaces, pages 390–
408.
Chuanrui Hu, Tong Li, Xingze Gao, Hongda Chen,
Yi Bai, Dannong Xu, Tianwei Lin, Xiaohong Li, Yun-
yun Han, Jian Pei, and Yafeng Deng. 2026. Evalu-
ating long-horizon memory for multi-party collabo-
rative dialogues. InProceedings of the 32nd ACM
SIGKDD Conference on Knowledge Discovery and
Data Mining V .2, page 9082–9090.
Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai.
2025. Memory OS of AI agent. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing, pages 25961–25970, Suzhou,
China. Association for Computational Linguistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 conference on empirical methods in natural
language processing (EMNLP), pages 6769–6781.
Jonathan K. Kummerfeld, Sai R. Gouravajhala, Joseph J.
Peper, Vignesh Athreya, Chulaka Gunasekara, Jatin
Ganhotra, Siva Sankalp Patel, Lazaros C Poly-
menakos, and Walter Lasecki. 2019. A large-scale
corpus for conversation disentanglement. InProceed-
ings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 3846–3856. As-
sociation for Computational Linguistics.
Chris Latimer, Nicoló Boschi, Andrew Neeser, Chris
Bartholomew, Gaurav Srivastava, Xuan Wang, and
Naren Ramakrishnan. 2025. Hindsight is 20/20:
Building agent memory that retains, recalls, and re-
flects.arXiv preprint arXiv:2512.12818.

Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding
Chen, Jiawei Yang, Chenyang Xi, Huayi Lai, Ji-
hao Zhao, Yezhaohui Wang, Junpeng Ren, Zehao
Lin, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li,
Zhiqiang Yin, Qingchen Yu, Bo Tang, and 3 others.
2025. Memos: An operating system for memory-
augmented generation (mag) in large language mod-
els.Preprint, arXiv:2505.22101.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov,
Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
2024. Evaluating very long-term conversational
memory of llm agents. InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 13851–
13870.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang,
Shishir G Patil, Ion Stoica, and Joseph E Gonzalez.
2023. Memgpt: Towards llms as operating systems.
arXiv preprint arXiv:2310.08560.
Stephen Robertson and Hugo Zaragoza. 2009. The prob-
abilistic relevance framework: Bm25 and beyond.
Foundations and trends® in information retrieval,
4(1-2):1–174.
Sagar Sapkota, Mohammad Saqib Hasan, Mubarak
Shah, and Santu Karmaker. 2025. Multi-party
conversational agents: A survey.arXiv preprint
arXiv:2505.18845.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher Manning.
2024. Raptor: Recursive abstractive processing for
tree-organized retrieval. InInternational Conference
on Learning Representations, volume 2024, pages
32628–32649.
Yiqun Sun, Pengfei Wei, and Lawrence B. Hsieh. 2026.
Don’t retrieve, navigate: Distilling enterprise knowl-
edge into navigable agent skills for QA and RAG. In
First Workshop on Agent Skills.
Memodb Team. 2025. Memobase: User profile-based
long-term memory for ai chatbot applications.
Xiaoyu Wang, Ningyuan Xi, Qingqing Gu, and Luo Ji.
2025. Multi-party supervised fine-tuning of language
models for multi-party dialogue generation. In2025
International Joint Conference on Neural Networks
(IJCNN), pages 1–9. IEEE.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao
Tan, and Yongfeng Zhang. 2026. A-mem: Agentic
memory for llm agents.Advances in Neural Informa-
tion Processing Systems, 38:17577–17604.
Jingbo Yang, Kwei-Herng Lai, Xiaowen Wang, Shiyu
Chang, Yaar Harari, and Evgeniy Gabrilovich. 2026.
Groupmembench: Benchmarking llm agent mem-
ory in multi-party conversations.arXiv preprint
arXiv:2605.14498.

A Appendix
A.1 Additional Experiments
A.1.1 Answer-Model Robustness
EGCForest consistently outperforms memory-
system baselines across all answer-model back-
bones, with gains of 13.83–18.52 points over the
strongest baseline. While baseline rankings vary
across backbones, EGCForest remains the top-
performing method. It achieves 57.18% with GLM-
5.3 and 55.70% with GPT-5, showing that its struc-
tured evidence remains effective across answer
models.
Backbone MemoBase Mem0 MemOS A-Mem Ours
Qwen3.5-35B-A3B 25.23 24.7 32.08 33.6947.52
DeepSeek-V4-Flash 28.46 26.44 32.75 37.9952.35
GLM-5.3 31.01 29.66 35.44 40.5457.18
GPT-5 (default) 27.92 28.19 35.70 37.1855.7
Table 4: Agent backbone ablation on GroupMemBench.
A.2 Dataset Details
We evaluate EGCForest on two public enterprise
group-chat memory benchmarks and an internal
enterprise-chat benchmark. Table 5, Table 6, and
Table 7 summarize their scales and evaluation dis-
tributions.
A.2.1 EverMemBench
EVERMEMBENCH(Hu et al., 2026) evaluates long-
horizon collaborative memory over nterprise group
chats spanning multiple channels. We use its offi-
cial evaluation protocol and all released question-
answer pairs.
Category Statistic Value
Corpus Projects 5
Channels per project 3
Participants 170
Time span 365 days
Messages 51,023
Dialogue tokens 4.23M
Fine-grained Recall Single-hop / Multi-hop / Temporal 213 / 249 / 300
Memory Awareness Constraint / Proactivity / Update 402 / 427 / 268
Profile Understanding Style / Skill / Role 176 / 169 / 196
Evaluation Total QA pairs 2,400
Table 5: Statistics and question distribution of EVER-
MEMBENCH.
A.2.2 GroupMemBench
GROUPMEMBENCH(Yang et al., 2026) evaluates
memory over threaded enterprise group chats with
user-conditioned queries. We use the official cor-
pus and evaluation protocol.Category Statistic Value
CorpusDomains 4
Projects 33
Topics 125
Phases 295
Participants 45
Messages 120,000
Message TypePosts 27,222
Threaded replies 79,766
Cross-project replies 13,012
Decision changes 352
EvaluationMulti-Hop 182
Update 107
Ambiguity 106
Implicit 49
Temporal 162
Abstention 139
Total QA pairs 745
Table 6: Statistics and evaluation-question distribution
of GroupMemBench.
A.2.3 Internal Enterprise-Chat Benchmark
Our internal benchmark is constructed from
anonymized enterprise group-chat logs. It evalu-
ates channel workflow, actor responsibility, artifact
lifecycle, cross-channel tracking, and multi-hop ev-
idence retrieval. Each question is annotated with
an answer and supporting message evidence.
Category Statistic Value
Corpus Persistent channels 3
Participants 39
Messages 10,222
Tracked artifact clusters 22
Artifact-bearing messages 1,158
Evaluation Fine-grained Recall 260
Memory Awareness 263
Profile Understanding 278
Total QA pairs 801
Table 7: Statistics and question distribution of the inter-
nal enterprise-chat benchmark.
A.3 Implementation Details
This section details the implementation choices
used for forest construction, retrieval, and answer
generation. Unless otherwise specified, we use
Qwen3-Embedding-0.6B as the frozen encoder for
both message and node-summary embeddings. Ta-
ble 8 summarizes the main hyperparameters and
dataset-specific answer-model settings.
Baseline Evaluation.We use Qwen3-
Embedding-0.6B for EGCForest and for all

reproduced baselines that require embeddings.
On EverMemBench, we rerun BM25, Dense,
BM25+Dense, GraphRAG, and A-Mem; results
for MemoBase, Mem0, and MemOS are reported
from the original benchmark paper. On Group-
MemBench, we rerun all methods. All baselines on
Company are evaluated by us. Because the filtered
evaluation subsets used in prior studies were
unavailable in the public releases we accessed, our
reproduced runs use the full publicly available data
for the corresponding task categories.
Forest Construction.All three forests use a com-
mon branching factor of k= 16 . Messages within
each tree are ordered chronologically and recur-
sively grouped into k-way internal nodes. This pro-
duces shallow trees while keeping the input to each
node-summary generation step manageable. For
example, a tree with approximately 104leaves has
height four. The artifact construction script has a
default centroid-merge threshold of θmerge = 0.88 .
The following example shows an immutable log
entry and its derived fields.
msg_idm 091
timestamp 2025-12-01 09:10
sender alice
channel #backend
reply_tom 090
text “@Bob, PR for DOC-042
is merged.”
mentions {bob}(derived)
identifiers {DOC-042}(derived)
placements channel:#backend
actor:alice(auth)
actor:bob(ment)
artifact:DOC-042
Retrieval and Assembly.Table 8 reports the
default settings for question answering, includ-
ing a final context cap of 20 messages. For
the retrieval-budget sweep in Figure 4, we ad-
just the final message cap to each evaluated cutoff
k∈ {20,30,40,50} . The tree-level RRF constant
ηtree= 0gives reciprocal-rank weights of 1/rank .
The path and leaf limits in S2apply to candidate
generation before final selection.
The expansion quota is shared by S2andS3, in-
cluding bridge messages. Verbatim messages are
included first; summary headers use the remain-
ing space, capped at 35% of the total token bud-
get (1,050 tokens). This is an upper bound, not a
reserved allocation. Node summaries are precom-
puted and cached during construction, and online
LLM guidance is disabled; retrieval requires no
LLM calls for query guidance, memory traversal,
or summarization.Parameter Value
EncoderQwen3-Embedding-0.6B
Branching factork16
RRF constants (message / tree) 60 / 0
Tree weights (Root / Leaf2tree
/ Collapsed)0.8 / 2.0 / 0.8
S2paths / leaf cap per path 8 / 4
Bridge fanout / depth 2 / 1 hop
Bridge messages/query (max.) 2
S1/ expansion quotas 14 / 6
Final messages (default max.) 20
Context budget 3,000 tokens
Header budget (max.) 35%
Answer (GroupMemBench) GPT-5
Answer (EverMemBench,
Company)GPT-4.1-mini
Judge (GroupMemBench) GPT-5
Judge (EverMemBench,
Company)Gemini-3-Flash-Preview
Table 8: Main implementation settings forEGCForest.
A.4 LLM Prompt Templates
We present condensed prompt templates. The
answer-generation and grading templates are
condensed from the officialEverMemBench
prompts (Hu et al., 2026). Multiple-choice ques-
tions require a single uppercase option letter and
are scored by direct matching; open-ended answers
use an LLM judge. Summaries are generated of-
fline, and online retrieval uses no LLM calls.
Node summaries.Inputs specify the view, tree
label, interval, level, and ordered source text: orig-
inal messages at L1, and consecutive child sum-
maries above it.
Write a compact index entry for routing to orig-
inal evidence. Preserve names, speaker attribu-
tion, dates, quantities, versions, statuses, deci-
sions, commitments, blockers, and unresolved
questions. Keep events chronological and express
state changes as “X →Y”. Omit unsupported
causes, outcomes, judgments, and advice; retain
specific names rather than replacing them with
vague categories.
Channel:identify participants and their contri-
butions; keep parallel workstreams distinct. For
reply-thread inputs, preserve the opening issue,
reply context, and stated resolution.Actor:distin-
guish the person’s own speech acts from others’
claims about them; preserve channel attribution
and available evidence IDs.Artifact:track life-
cycle transitions, responsible speakers, versions,
statuses, decisions, blockers, and channel prove-
nance.
Non-root summaries use one paragraph, capped
at 200/220/180 words for Channel/Actor/Artifact;

Actor summaries use 2–4 compact sentences. Root
cards use labelled prose: Channel records span,
participants, workstreams, decisions, and open
threads; Actor records supported roles, channels,
responsibilities, explicit decision authority, ob-
served phrasing, others’ attributed descriptions, and
at most three evidence IDs; Artifact records aliases,
owners, channels, current status, lifecycle, and key
decisions. Unsupported fields are omitted. Actor
root cards are capped at 160 words.
Open-ended answer generation.Input: re-
trieved memories and the question.
Answer using only the supplied memories. Track
who said what and when; use direct evidence and
prioritize the latest memory when accounts con-
flict. Convert relative time expressions to specific
dates using message timestamps. Output only
a concise answer, preferably under 5–6 words,
without reasoning or extra explanation.
Open-ended answer grading.Input: the ques-
tion, gold answer, and generated answer.
Judge whether the generated answer contains the
gold answer’s key information. Accept equivalent
wording and date formats; allow a ±1-day date
difference for timezone variation. Return a JSON
labelofCORRECTorWRONG.
A.5 Ethics Statement
We hereby confirm that the entire work presented in
this paper, including all content, analysis, and con-
clusions, is the original work of the human authors.
While the core research, analysis, and conclusions
are entirely the work of the human authors, gener-
ative AI was consulted for the limited purpose of
improving the grammar and readability of the text.