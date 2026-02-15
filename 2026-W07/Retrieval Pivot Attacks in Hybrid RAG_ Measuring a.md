# Retrieval Pivot Attacks in Hybrid RAG: Measuring and Mitigating Amplified Leakage from Vector Seeds to Graph Expansion

**Authors**: Scott Thornton

**Published**: 2026-02-09 13:55:04

**PDF URL**: [https://arxiv.org/pdf/2602.08668v2](https://arxiv.org/pdf/2602.08668v2)

## Abstract
Hybrid Retrieval-Augmented Generation (RAG) pipelines combine vector similarity search with knowledge graph expansion for multi-hop reasoning. We show that this composition introduces a distinct security failure mode: a vector-retrieved "seed" chunk can pivot via entity links into sensitive graph neighborhoods, causing cross-tenant data leakage that does not occur in vector-only retrieval. We formalize this risk as Retrieval Pivot Risk (RPR) and introduce companion metrics Leakage@k, Amplification Factor, and Pivot Depth (PD) to quantify leakage magnitude and traversal structure.
  We present seven Retrieval Pivot Attacks that exploit the vector-to-graph boundary and show that adversarial injection is not required: naturally shared entities create cross-tenant pivot paths organically. Across a synthetic multi-tenant enterprise corpus and the Enron email corpus, the undefended hybrid pipeline exhibits high pivot risk (RPR up to 0.95) with multiple unauthorized items returned per query. Leakage consistently appears at PD=2, which we attribute to the bipartite chunk-entity topology and formalize as a proposition.
  We then show that enforcing authorization at a single location, the graph expansion boundary, eliminates measured leakage (RPR near 0) across both corpora, all attack variants, and label forgery rates up to 10 percent, with minimal overhead. Our results indicate the root cause is boundary enforcement, not inherently complex defenses: two individually secure retrieval components can compose into an insecure system unless authorization is re-checked at the transition point.

## Full Text


<!-- PDF content starts -->

Retrieval Pivot Attacks in Hybrid RAG:
Measuring and Mitigating Amplified Leakage from
Vector Seeds to Graph Expansion
Scott Thornton
scott@perfecxion.ai
Abstract
Hybrid Retrieval-Augmented Generation (RAG) pipelines
increasingly combine vector similarity search with knowl-
edge graph expansion to support multi-hop reasoning. We
show this composition introduces a distinct security failure
mode: a semantically retrieved “seed” chunk can pivot via
entity linking into sensitive graph neighborhoods, causing
data leakage that does not exist in vector-only retrieval. We
formalize this risk asRetrieval Pivot Risk (RPR)and de-
fine companion metricsLeakage@k,Amplification Factor
(AF), andPivot Depth (PD)to quantify leakage probability,
magnitude, amplification over vector baselines, and structural
distance to the first unauthorized node. We further present a
taxonomy of fourRetrieval Pivot Attacksthat exploit the
vector-to-graph boundary with injection budgets as small as
10–20 chunks, and we demonstrate that adversarial injection
is not required: naturally shared entities (e.g., vendors, in-
frastructure, compliance standards) create cross-tenant pivot
paths organically. In a synthetic multi-tenant enterprise corpus
(1,000 documents; 2,785 graph nodes; 15,514 edges) evalu-
ated with 500 queries, the undefended hybrid pipeline exhibits
RPR≈0.95 andAF(ε)≈160–194× relative to vector-only
retrieval, with leakage occurring at PD=2 hops as a structural
consequence of the bipartite chunk–entity topology. We vali-
date these findings on two real-world corpora: the Enron email
corpus (50,000 emails; RPR=0.695 ) and SEC EDGAR 10-
K filings (887 sections across 20 companies; RPR=0.085 ).
RPR scales with entity connectivity density, but the struc-
tural invariant (PD=2) persists across all three corpora. We
propose five layered defenses and find that a single place-
ment fix—per-hop authorization at the graph expansion
boundary—eliminates all measured leakage ( RPR→0.0 )
across all three corpora, all queries, and all attack variants
with negligible latency overhead, indicating the vulnerability
is primarily a boundary enforcement problem rather than a
defense complexity problem.1 Introduction
Enterprise adoption of Retrieval-Augmented Generation
(RAG) has accelerated rapidly: 30–60% of enterprise AI
use cases now rely on RAG architectures [11], and the vec-
tor database market reached $1.73 billion in 2024 [14]. To
improve multi-hop reasoning over complex organizational
knowledge, practitioners increasingly deployhybridRAG
pipelines that combine vector similarity search with knowl-
edge graph expansion [15, 16]. These systems retrieve an
initial set of text chunks via embedding similarity, thenpivot
through entity mentions into a knowledge graph to gather
structurally related context before passing the assembled re-
sults to a large language model (LLM). We call the transi-
tion between vector retrieval and graph expansion thepivot
boundary: the architectural point where a retrieved seed be-
comes a graph traversal capability.
This boundary is aboundary placementbug. The vector
store enforces tenant policybeforeretrieval. The graph expan-
sion must enforce itagainafter entity linking—otherwise an
authorized seed chunk becomes an uncontrolled entry point
into the knowledge graph. Consider an analyst at an engineer-
ing firm who queries about Kubernetes cluster configurations.
Vector retrieval returns authorized engineering documents.
But those documents mention shared entities (“CloudCorp,”
“auth-service”) that also appear in the knowledge graph con-
nected to confidential HR salary records, restricted security
credentials, and financial audit data belonging to other ten-
ants. A 2-hop graph expansion traverses through these shared
entities into unauthorized neighborhoods, silently injecting
sensitive cross-tenant data into the analyst’s context window.
Prior work falls into two separate buckets that do not
address this boundary.(1) Vector-only attacks: Poisone-
dRAG achieves 90% attack success rate by injecting 5 ma-
licious texts into vector stores [28]; CorruptRAG [24] and
CtrlRAG [18] extend these results.(2) Graph-only attacks:
GRAGPoison demonstrates 98% success via relation-centric
poisoning within GraphRAG [10]; TKPA [20] and RAG-
Safety [25] target graph-side integrity.(3) The hybrid bound-
1arXiv:2602.08668v2  [cs.CR]  10 Feb 2026

ary—how vector outputs become graph seeds—has not been
systematically measured or formalized. The OWASP LLM
Top 10 (2025) introduced LLM08 (Vector and Embedding
Weaknesses) but does not address graph components [2].
MITRE ATLAS treats RAG as a monolithic system [1]. The
SoK on RAG privacy explicitly identifies hybrid RAG security
as an open problem [4].
This paper makes three contributions:
1.Composition vulnerability + metrics.We formalizeRe-
trieval Pivot Risk (RPR)and companion metrics (AF, PD,
Leakage@k, severity-weighted leakage) that quantify a
compound attack surface emerging from composing two
individually secure retrieval modalities. In our bipartite
chunk–entity graph, all leakage occurs at PD =2 hops—
a structural signature of the pivot boundary (§5).
2.Cross-dataset validation + organic leakage.We
demonstrate the vulnerability on three corpora: a syn-
thetic enterprise corpus (1,000 documents, 4 tenants;
RPR=0.95), the Enron email corpus (50,000 emails, 5
departments; RPR =0.70), and SEC EDGAR 10-K fil-
ings (887 sections, 4 sectors; RPR =0.09)—all without
adversarial injection. We present four Retrieval Pivot
Attacks (A1–A4) exploiting the pivot boundary with
injection budgets of 10–20 chunks (§4, §7).
3.Defense placement analysis.We evaluate five layered
defenses (D1–D5) and show that authorization must be
re-checked at the graph expansion boundary. D1 (per-
hop authorization) alone eliminates all measured leakage
(RPR→0.0 ) across both corpora, all queries, all attack
variants, and metadata mislabel rates up to 5%. D2–D5
serve as utility optimizers and defense-in-depth layers
(§8, §7).
The complete codebase, data generators, attack im-
plementations, defense suite, and experimental results
are available at https://github.com/scthornton/
hybrid-rag-pivot-attacks.
2 Background
2.1 Vector Retrieval in RAG
Standard RAG systems encode documents as dense vector
embeddings using models such as all-MiniLM-L6-v2 [26]
and retrieve the top- kchunks by cosine similarity to the
query embedding. In multi-tenant deployments, vector stores
apply metadata prefilters (e.g., tenant ID) before similarity
search, ensuring that retrieval respects organizational bound-
aries. This prefiltering makes vector-only RAG robust against
cross-tenant leakage: our experiments confirm RPR=0.0 for
vector-only pipelines across all evaluation queries on all three
corpora (§7).2.2 Knowledge Graph RAG
GraphRAG [15] and related approaches construct knowledge
graphs from document corpora via named entity recognition
(NER) and relation extraction, then leverage graph struc-
ture for multi-hop reasoning. Systems like AgCyRAG [9]
use LLM-driven graph traversal for complex queries. Graph-
based retrieval provides superior comprehensiveness on multi-
hop queries—86% versus 57% for vector RAG [16]—but
introduces graph-specific attack surfaces: GRAGPoison [10]
achieves 98% attack success through relation-centric poison-
ing, and TKPA [20] drops QA accuracy from 95% to 50% by
modifying just 0.06% of corpus text.
2.3 Hybrid RAG and the Pivot Boundary
Hybrid RAG combines both retrieval modalities in a pipeline:
1.Vector retrieval:Query embedding →cosine similarity
→top-kseed chunksS(q).
2.Entity linking:Extract named entities from S(q) via
NER; map entity mentions to knowledge graph node
IDs.
3.Graph expansion:BFS/DFS traversal from linked entity
nodes to depthd, gathering structurally related nodes.
4.Context merge:Combine vector-retrieved chunks with
graph-expanded context for LLM generation.
We define thepivot boundaryas the transition between
steps 1–2 and steps 3–4: the point where vector retrieval
results become graph traversal seeds. This boundary is the at-
tack surface we study. Vector-side prefilters operatebeforethe
pivot; graph expansion operatesafterit. If graph expansion
does not independently enforce access controls, any entity
mentioned in an authorized seed chunk becomes an uncon-
trolled entry point into the knowledge graph.
Running example.Consider an INTERNAL-clearance en-
gineer at Acme Corp who queries “What infrastructure does
auth-service depend on?” The vector store returns authorized
Acme engineering documents mentioning auth-service. En-
tity linking maps “auth-service” to a shared entity node in the
knowledge graph. BFS expansion then walks: auth-service
→CloudCorp (shared vendor entity, hop 1) →Umbrella Se-
curity’s CONFIDENTIAL incident-response documents (hop
2). The engineer’s authorized query silently pulls CONFI-
DENTIAL cross-tenant data into the context—a 2-hop pivot
through a legitimate shared entity.
3 Threat Model
3.1 System Model
We consider a multi-tenant enterprise hybrid RAG system
with the following components:
2

•Avector store(ChromaDB) containing document chunk
embeddings with tenant and sensitivity metadata.
•Aknowledge graph(Neo4j) containing entity
nodes, chunk nodes, and typed edges (MENTIONS,
DEPENDS_ON, BELONGS_TO, RELATED_TO) ex-
tracted from the document corpus.
•Anentity linkerthat maps NER-extracted entities from
vector-retrieved chunks to graph node IDs.
•Agraph expanderthat performs BFS traversal from
linked entities to depth d(default d=2 ), imple-
mented via Neo4j’s apoc.path.spanningTree for hop-
distance tracking.
•Four organizationaltenantswith distinct data ownership
boundaries.
•Foursensitivity tiers: PUBLIC ( ≈40% of corpus),
INTERNAL (30%), CONFIDENTIAL (20%), RE-
STRICTED (10%).
3.2 Attacker Capabilities
We consider two attacker models:
Injection attacker.The attacker holds a legitimate account
in one tenant ( acme_engineering ) and can inject documents
into the shared corpus through standard channels—wiki edits,
ticket updates, shared document repositories. Injected docu-
ments undergo normal ingestion: chunking, NER-based en-
tity extraction, embedding, and indexing into both vector and
graph stores. The attacker’s injection budget is modest: 10–20
chunks, consistent with the budgets shown effective in prior
work [24, 28].
No-injection attacker.The attacker holds a legitimate ac-
count and crafts queries that mention bridge entities (shared
infrastructure names, vendor references, cross-team person-
nel) to maximize pivot probability through naturally occurring
cross-tenant entity connections. This attacker requireszero
document injection—the organic structure of the multi-tenant
knowledge graph provides the pivot paths. Our evaluation
shows that this attacker model achieves RPR=0.95 through
benign queries alone (§7), demonstrating that the vulnerability
isstructuralrather than injection-dependent.
3.3 Attacker Goals
•G1: Cross-tenant access.Cause the retrieval context
for a query in the attacker’s tenant to include chunks or
entities belonging to other tenants.
•G2: Sensitivity escalation.Cause an INTERNAL-
clearance user’s context to include CONFIDENTIAL
or RESTRICTED items.•G3: Amplified leakage.Achieve leakage in the hybrid
pipeline that exceeds what is possible with vector-only
retrieval (i.e., AF>1).
3.4 Adaptive Attacker Considerations
An adaptive attacker aware of D1 (per-hop authorization)
might attempt to bypass it through: (a)metadata spoofing—
forging tenant or sensitivity labels during document injection,
or (b)same-tenant escalation—crafting documents within
the attacker’s own tenant that link to higher-sensitivity content
within the same tenant boundary. D1 is robust against (a) if
and only if the ingestion pipeline enforces metadata integrity
(the metadata is assigned by the system, not the uploader).
We evaluate D1’s robustness under metadata corruption in
§7.11 and discuss mitigation strategies in §8.2. Same-tenant
escalation (b) targetssensitivityboundaries rather thantenant
boundaries and requires the attacker to already have write
access to the target tenant’s corpus.
3.5 Metadata Integrity Assumption
D1 (per-hop authorization) relies on node metadata—tenant
labels and sensitivity tiers—being trustworthy.We assume
metadata is system-assigned during document ingestion
(the uploader cannot choose tenant or sensitivity labels). If an
attacker can forge metadata, D1 can be bypassed. We evaluate
robustness under random metadata corruption in §7.11 and
discuss additional integrity mechanisms in §8.2.
3.6 Scope
We measure unauthorized items in the retrieval context
window, not exfiltrated tokens or generated text. We focus
exclusively on the retrieval layer. We do not evaluate LLM
generation quality, jailbreaking, or prompt injection. The at-
tacker does not have direct database access, cannot modify
model weights, and cannot alter the pipeline configuration.
We omit a graph-only baseline (P2) because graph-only RAG
without vector seeding is a fundamentally different retrieval
paradigm—our question isamplification created by compo-
sition, not graph-only security, which is addressed by prior
work [10, 20].
4 Retrieval Pivot Attacks
We define four non-adaptive attacks (A1–A4) that exploit the
pivot boundary with increasing structural sophistication, and
three adaptive attacks (A5–A7) that target the defense mech-
anisms themselves. Each attack operates within the threat
model of §3: the attacker injects a small number of crafted
documents that undergo standard ingestion.
3

4.1 A1: Seed Steering
Objective.Maximize the probability that attacker-crafted
chunks are retrieved by target queries while embedding entity
mentions that link to sensitive graph neighborhoods.
Mechanism.The attacker estimates the query centroid
for target query families (e.g., “infrastructure monitoring”)
and crafts 10 chunks with high semantic overlap to this cen-
troid. Each chunk embeds 2 pivot entities—entities that are
within 1–2 hops of sensitive nodes in the knowledge graph
(e.g., k8s-prod-cluster ,auth-service ). Chunks are la-
beled PUBLIC with low provenance scores (0.3) to bypass
any trust-based filtering. When retrieved, entity linking maps
the embedded entity mentions to graph nodes, seeding expan-
sion into sensitive neighborhoods.
Entry point.Vector retrieval (cosine similarity).
Pivot mechanism.Entity linking from retrieved chunk to
graph node.
4.2 A2: Entity Anchor Injection
Objective.Force entity linking to create dense connections
between injected chunks and sensitive graph neighborhoods.
Mechanism.The attacker identifies anchor entities adja-
cent to sensitive nodes (e.g., entities 1 hop from restricted
credentials) and crafts chunks with dense entity mentions: 3+
mentions of the primary target entity per chunk, plus 2 related
entities. NER extraction creates MENTIONS edges from each
chunk to the target entities. Any query that retrieves one of
these chunks—even incidentally—triggers graph expansion
toward the sensitive area.
Entry point.Entity extraction during ingestion.
Pivot mechanism.MENTIONS edges from chunk to target
entity nodes.
4.3 A3: Neighborhood Flooding
Objective.Inflate the local density around a target entity to
increase the volume of sensitive content reachable through
BFS expansion.
Mechanism.The attacker injects 20 chunks, each mention-
ing a high-value target entity near a sensitive subgraph. Each
chunk also mentions a different neighbor entity, creating a
dense web of edges. The result is a “graph gravity well”: the
target entity accumulates many more edges than the graph av-
erage, causing BFS expansion to gather a larger neighborhood
when traversing through this entity. Note that our pipeline uses
uniform BFS traversal (not degree-weighted or PageRank-
based expansion), so the flooding increases reachable volume
rather than biasing traversal order.
Entry point.Graph topology manipulation via ingestion.
Pivot mechanism.Inflated local density increases BFS ex-
pansion volume.Algorithm 1General Retrieval Pivot Attack
Require: Target query family QT, injection budget B, pivot
entitiesE p, target neighborhoodN T
Ensure:Injected chunksC atk
1:fori=1 toBdo
2:c i←CRAFTCHUNK(Q T,Ep)▷High similarity to
QT
3:c i.entities←EMBEDENTITIES(E p,NT)▷Mentions
nearN T
4:c i.sensitivity←PUBLIC
5:c i.provenance←0.3
6:C atk←C atk∪{c i}
7:end for
8:INGEST(C atk)▷Standard pipeline: chunk, NER, embed,
index
9:returnC atk
4.4 A4: Bridge Node Attack
Objective.Create artificial cross-tenant edges that enable
graph traversal from the attacker’s tenant into a target tenant.
Mechanism.The attacker crafts 15 chunks that co-mention
entities from both the attacker’s tenant and the target tenant
within the same document. NER extraction creates entity
nodes on both sides of the tenant boundary, and relation ex-
traction creates RELATED_TO edges between them. After
ingestion, BFS expansion from the attacker’s authorized sub-
graph can traverse these artificial bridge edges into the target
tenant’s data.
Entry point.Cross-tenant entity co-mention.
Pivot mechanism.Artificial RELATED_TO edges spanning
tenant boundaries.
Algorithm 1 formalizes the general pivot attack framework.
5 Metrics
We introduce metrics to quantify retrieval pivot risk. Let q
denote a query, ua user with tenant tuand clearance level ℓu,
Sk(q)the top- kcontext set, and Sensitive(x,u) a predicate that
is true when item xhas sensitivity > ℓ uor tenant ̸=tu(entity
nodes with empty-string tenants are excluded from cross-
tenant counts, as they are tenant-neutral shared concepts—not
leaked items themselves but rather the pivot bridges through
which leakage propagates). Let Seeds(q)⊆S k(q)denote the
set of entity nodes that are linked from vector-retrieved chunks
via NER extraction—these are the graph entry points that
initiate BFS expansion.
Item serialization.Eachitemin Sk(q)is a node serial-
ized for the LLM context: chunk nodes are serialized as
their source text; entity nodes are serialized as {name, type,
top-3 relations} . Context size counts the total number of
serialized nodes (chunks + entities) presented to the LLM.
4

Table 1: Notation summary.
Symbol Definition
q Query
u User with tenantt uand clearanceℓ u
Sk(q) Top-kcontext set (serialized items)
Seeds(q) Entity nodes linked from vector-retrieved chunks
Sensitive(x,u) xhasℓ x> ℓuortx̸=tu
RPR Prob. of any sensitive item in context
Leakage@k Count of sensitive items in context
SWL Severity-weighted leakage
AF(ε) Leakage ratio: hybrid / max(vector,ε)
PD Min hops from seed to first sensitive node
The default pipeline configuration (depth d=2 , branching
≤10 , total≤100 nodes, top- k=10 vector results) produces
∼110 items per query in the undefended hybrid pipeline (P3).
Table 1 summarizes the notation used throughout.
5.1 Retrieval Pivot Risk (RPR)
RPR measures the probability that a query’s retrieval context
contains any unauthorized item:
RPR(u) =Pr
q∼Q[∃x∈S k(q): Sensitive(x,u)](1)
RPR is operationalized as the fraction of evaluation queries
whose context contains at least one sensitive item. We re-
port RPR with 95% bootstrap confidence intervals (10,000
resamples).
5.2 Leakage@k
Leakage@k counts the number of unauthorized items in the
context:
Leakage@k(q,u) = |{x∈S k(q): Sensitive(x,u)} |(2)
While RPR captureswhetherleakage occurs, Leakage@k
captures itsseverity. A context with 1 leaked item and one
with 20 both yield RPR=1 , but their Leakage@k values
differ by 20×.
5.3 Severity-Weighted Leakage
Not all leakage is equally harmful. A PUBLIC-clearance user
receiving INTERNAL content is a lesser violation than re-
ceiving RESTRICTED content. We define severity-weighted
leakage as:
SWL(q,u) = ∑
x∈S k(q)
Sensitive(x,u)w(x,u)(3)where the per-item weight w(x,u) captures two distinct policy
violations:
w(x,u) =(
ℓx−ℓuifℓx> ℓ u
1 ift x̸=tu∧ℓx≤ℓ u(4)
Hereℓxis the sensitivity level (PUBLIC=0, INTERNAL=1,
CONFIDENTIAL=2, RESTRICTED=3) and txthe tenant of
item x. Over-clearance items contribute weight proportional
to the sensitivity gap. Cross-tenant items that do not exceed
the user’s clearance receive a penalty of 1, reflecting the pol-
icy violation of accessing another tenant’s data regardless
of sensitivity. In our experiments, the INTERNAL-clearance
user observing RESTRICTED items produces weight 2 per
item, CONFIDENTIAL produces weight 1, and cross-tenant
PUBLIC items produce weight 1.
5.4 Amplification Factor
AF quantifies the leakage increase that hybrid retrieval intro-
duces compared to vector-only retrieval:
AF=E[Leakage@k] hybrid
E[Leakage@k] vector(5)
When the vector baseline produces zero leakage (which our
experiments confirm), AF=∞ for any nonzero hybrid leakage.
To provide a finite, plottable alternative, we also report AF(ε) :
AF(ε) =E[Leakage@k] hybrid
max(E[Leakage@k] vector,ε)(6)
withε=0.1 , alongside the absolute difference ∆Leakage=
E[Leakage@k] hybrid−E[Leakage@k] vector .
Because the vector-only baseline produces zero leakage in
our setting, ratio metrics can be ill-conditioned ( AF=∞ ). We
therefore use ∆Leakage as the primary magnitude measure
and AF( ε) only as a secondary, plottable indicator. The central
security finding is not the exact amplification ratio but that
hybridizationcreates nonzero leakagedespite secure vector
prefiltering.
5.5 Pivot Depth (PD)
PD measures the minimum graph distance from a seed node
to the first sensitive node reached during expansion:
PD(q) =min{d(s,x):s∈Seeds(q),x∈S k(q),Sensitive(x,u)}
(7)
where d(s,x) is the shortest-path distance in the knowl-
edge graph from seed sto node x, computed via
apoc.path.spanningTree which yields paths with explicit
hop counts. We report PD as a distribution (min, median,
max) across queries that exhibit leakage, rather than a single
summary statistic. Operationally, PD identifies theminimum
5

traversal depth at which enforcement must occur: if all leak-
age has PD=d , limiting expansion to depth <dor inserting
an authorization check at depthdis sufficient to prevent it.
6 Experimental Setup
6.1 Synthetic Enterprise Corpus
We generate a multi-tenant enterprise corpus of
1,000 documents across four organizational tenants:
acme_engineering ,globex_finance ,initech_hr , and
umbrella_security . Documents are produced by 12
domain-specific generators (3 per tenant) that embed
realistic entity mentions—system names, personnel, projects,
compliance standards—using curated entity pools.
Sensitivity tiers follow a realistic distribution: PUBLIC
(40%), INTERNAL (30%), CONFIDENTIAL (20%), and
RESTRICTED (10%). Each document includes ground-truth
entity annotations and sensitivity labels.
Bridge entities.We inject 15 bridge entities across 5 cate-
gories (shared vendors, shared infrastructure, shared person-
nel, shared compliance standards, and shared projects) that
naturally appear in documents across multiple tenants. For ex-
ample, “CloudCorp” appears in both engineering and finance
documents, and “auth-service” appears in both engineering
and security documents. After NER-based entity extraction,
the knowledge graph contains 40 naturally shared entities
across tenant boundaries—not through adversarial injection,
but through legitimate cross-team references that any multi-
tenant organization would exhibit.
6.2 Knowledge Graph Construction
Documents undergo chunking (300-token windows, 50-token
overlap), producing ∼2,000 chunks from 1,000 source docu-
ments (average ∼2 chunks per document). Chunks are pro-
cessed through spaCy NER extraction ( en_core_web_sm ),
two-pass relation extraction (ground-truth relation resolu-
tion followed by pattern-based extraction across 5 typed rela-
tions: DEPENDS_ON, OWNED_BY , BELONGS_TO, CON-
TAINS, DERIVED_FROM, plus RELATED_TO fallback),
and embedding via all-MiniLM-L6-v2 (384 dimensions). The
resulting knowledge graph contains 2,785 nodes (785 en-
tity nodes + 2,000 chunk nodes) and 15,514 edges (7,386
extracted relations plus MENTIONS, CONTAINS, and BE-
LONGS_TO structural edges added during graph construc-
tion).
6.3 Enron Email Corpus
To validate that Retrieval Pivot Risk is not an artifact of syn-
thetic construction, we evaluate on the Enron email corpus—
a public-record dataset of ∼500,000 corporate emails from∼150 Enron employees, released during the 2001 FERC in-
vestigation. We subsample to 50,000 emails from the most
active employees and assign tenants based on departmental
structure:
•trading— Trading, West/East Power Trading
•legal— Legal, Government Affairs
•finance— Finance, Risk Management, Accounting
•energy_services— Energy Services, Pipeline, ENA
•executive— Executive, Office of the Chairman
Sensitivity labels are assigned by keyword matching: RE-
STRICTED (attorney-client privilege markers, password
shares, strategy memos), CONFIDENTIAL (deal negotiations,
valuations, board communications), INTERNAL (standard
departmental communications), PUBLIC (company-wide an-
nouncements). The resulting distribution (92% PUBLIC,
6.7% CONFIDENTIAL, 0.7% RESTRICTED, 0.3% INTER-
NAL) is skewed compared to the synthetic corpus—reflecting
realistic email traffic where most messages are routine.
Chunking (300-token windows, 50-token overlap) produces
152,064 chunks. spaCy NER extraction ( en_core_web_sm )
identifies 2.07M entity mentions. The resulting knowledge
graph contains 376,000 nodes (174,000 entity nodes +
152,000 chunk nodes + 50,000 document nodes) and 2.3M
edges. Nineteen entities are naturally shared across depart-
mental boundaries—cross-department executives (Ken Lay,
Jeff Skilling, Andy Fastow), external organizations (Arthur
Andersen, Vinson & Elkins), deal names (Project Raptor,
LJM2), and internal systems (EnronOnline).
Key differences from synthetic.The Enron graph is 135×
larger than the synthetic graph (376K vs. 2.8K nodes) but has
fewer intentional bridge entities (19 vs. 40). Email text pro-
duces noisier NER extractions (dates, monetary amounts, par-
tial names) than curated synthetic documents, creating more
spurious entity connections but also more entity fragmenta-
tion. These properties make the Enron corpus a stress test for
whether the pivot vulnerability generalizes beyond controlled
conditions.
6.4 EDGAR 10-K Corpus
As a third corpus, we evaluate on SEC EDGAR 10-K an-
nual reports—public filings from 20 companies across four
industry sectors (tech, finance, healthcare, energy). Each
sector serves as a tenant, and filing sections receive sen-
sitivity labels based on a Material Non-Public Informa-
tion (MNPI) framework: Items 1–4 (PUBLIC), Items 7–
8 (INTERNAL), Items 1A/7A/9A/11–12 (CONFIDENTIAL),
Items 5/13 (RESTRICTED).
The EDGAR graph contains 19,527 nodes (887 documents,
3,692 chunks, 14,948 entities) and 427,415 edges. Cross-
sector bridge entities arise naturally from shared auditors
6

Table 2: Pipeline variants and their defense configurations.
ID Pipeline Defenses
P1 Vector-only Tenant prefilter
P3 Hybrid baseline None
P4 Hybrid + D1 Per-hop authz
P5 Hybrid + D1,D2 + Edge allowlist
P6 Hybrid + D1–D3 + Budgeted traversal
P7 Hybrid + D1–D4 + Trust weighting
P8 Hybrid + D1–D5 + Merge filter
(Big 4 firms), institutional investors (BlackRock, Vanguard),
and board members serving on companies in multiple sec-
tors. NER extraction on formal financial language produces
dense entity graphs ( ∼4 entity mentions per chunk) but the
standardized vocabulary creates fewer distinctive cross-sector
entity connections than either the synthetic or Enron corpus.
Key differences from other corpora.EDGAR filings use
formal, standardized language with fewer organic entity con-
nections across sector-tenants than Enron’s informal email
traffic. This produces the lowest RPR of the three corpora
(0.085 vs.0.695 Enron, 0.954 synthetic) while preserving the
PD=2 structural invariant.
6.5 Pipeline Variants
We evaluate 7 pipeline configurations (Table 2):
We omit a graph-only baseline (P2) because graph-only re-
trieval without vector seeding is a fundamentally different
paradigm that does not involve the pivot boundary under
study. Its security properties are addressed by prior work
on GraphRAG attacks [10, 20].
6.6 Evaluation Protocol
Synthetic corpus.We evaluate each pipeline on 500 template-
generated queries: 350 benign queries (standard domain ques-
tions stratified across 4 tenants and 3 clearance levels) and
150 adversarial queries (queries that mention bridge entities
or target cross-tenant pivot paths, stratified across attack types
A1–A4). The evaluation user belongs to acme_engineering
with INTERNAL clearance.
Enron corpus.We evaluate P1, P3, and P4 on 200 queries
(100 benign + 100 adversarial) generated from department-
specific templates. The evaluation user belongs to trading
with INTERNAL clearance.
For each query, we measure RPR, Leakage@k, severity-
weighted leakage, AF( ε),∆Leakage, PD distribution, latency,
and context size. All RPR and Leakage@k values are reported
with 95% bootstrap confidence intervals (10,000 resamples,
seed 42). Graph expansion uses apoc.path.spanningTree
for BFS with hop-distance tracking: each expanded noderecords its minimum distance from the nearest seed, enabling
precise PD measurement.
Statistical methodology.All metrics are reported with
95% bootstrap confidence intervals (10,000 resamples, per-
centile method, seed 42). We use non-parametric bootstrap
because RPR values near 0 or 1 violate normal approxima-
tions, and leakage distributions are highly skewed. Our query
sets (500 synthetic, 200 Enron) each provide power >0.80
for detecting 5pp differences (Appendix F provides sample
size justification, multiple comparison corrections, and εsen-
sitivity analysis).
Reproducibility.All experiments use a single fixed ran-
dom seed (42) for corpus generation, query sampling, and
bootstrap resampling. Pipeline configurations are built pro-
grammatically (not from YAML) to ensure parameter con-
sistency across variants. Appendix A provides the full repro-
duction sequence; Appendix B details the corpus generator;
Appendix C lists all query templates. The codebase includes
255 passing unit tests.
6.7 Attack Evaluation
We evaluate all four attacks (A1–A4) against the undefended
hybrid pipeline (P3) and three defense configurations (P4,
P6, P8). For each attack, payloads are injected into a clean
corpus, and 10 adversarial queries are executed against each
pipeline variant. After each attack evaluation, the graph is
rebuilt from the clean corpus to prevent cross-contamination
between attack experiments.
7 Results
7.1 Hybrid RAG Amplifies Leakage
Table 3 presents security metrics across all pipeline variants.
The central finding is stark:the undefended hybrid pipeline
(P3) leaks massively while the vector-only baseline (P1)
leaks nothing.
P1 achieves RPR=0.0 on both query sets—the vector
store’s tenant prefilter prevents any cross-tenant or sensitivity-
escalated content from reaching the context. In contrast,
P3 achieves RPR=0.954 [95% CI: 0.931, 0.974] on 350
benign queries (334 of 350 queries produce leakage) and
RPR=0.947 [0.907, 0.980] on 150 adversarial queries (142
of 150 queries leak). Mean Leakage@k reaches 16.0 items for
benign queries and 19.4 for adversarial queries—meaning that
roughly 15–18% of the 110-item context consists of unau-
thorized content. Severity-weighted leakage averages 22.9
(benign) and 26.4 (adversarial), indicating that leaked items
skew toward higher sensitivity tiers.
Because the vector baseline (P1) produces zero leakage,
the absolute difference ∆Leakage=16.0 (benign) and 19.4
(adversarial) is the most direct measure of the hybrid penalty:
each query exposes 16–19 additional unauthorized items
7

Table 3: Defense ablation: security metrics across pipeline variants (500 queries: 350 benign + 150 adversarial). RPR = Retrieval
Pivot Risk with 95% bootstrap CI, Leak = mean Leakage@k, SWL = severity-weighted leakage, PD = Pivot Depth (hops), Ctx =
mean context size. “–” indicates no leakage occurred (PD undefined).
Benign Queries (n=350) Adversarial Queries (n=150)
Variant RPR [CI] Leak SWL PD Ctx RPR [CI] Leak SWL PD Ctx
P1 (Vector) 0.000 0.0 0.0 – 10 0.000 0.0 0.0 – 10
P3 (Hybrid) .954 [.931,.974] 16.0 22.9 2.0 110 .947 [.907,.980] 19.4 26.4 2.0 110
P4 (+D1) 0.000 0.0 0.0 – 56 0.000 0.0 0.0 – 50
P5 (+D1,D2) 0.000 0.0 0.0 – 57 0.000 0.0 0.0 – 51
P6 (+D1–D3) 0.000 0.0 0.0 – 29 0.000 0.0 0.0 – 28
P7 (+D1–D4) 0.000 0.0 0.0 – 28 0.000 0.0 0.0 – 24
P8 (All) 0.000 0.0 0.0 – 20 0.000 0.0 0.0 – 20
Figure 1: Retrieval Pivot Risk with 95% bootstrap CIs across
pipeline variants. P3 (undefended hybrid) shows RPR≈0.95 .
All defended variants (P4–P8) achieve RPR=0.0.
purely from graph expansion. The classical Amplification
Factor is formally AF=∞ (division by zero); the regularized
variant AF(ε=0.1) =160 –194× provides a finite proxy.1The
hybrid architecture does not merely amplify existing risk—it
creates risk from nothing: P1 achieves RPR=0.0 while P3
reaches RPR=0.95 on identical queries.
Figure 1 shows RPR with bootstrap confidence intervals
across pipeline variants.
7.2 The PD = 2 Structural Signature
All leakage in P3 occurs at exactly PD=2 hops: the PD
distribution is (min=2,median=2,max=2) across all 476
leaking queries. This uniformity is not coincidental—it is a
structural consequence of the bipartite chunk-entity graph
topology in our construction. The pivot path is:
1.Hop 0: Authorized seed chunk (retrieved by vector simi-
larity, passes tenant prefilter).
2.Hop 1: Shared entity node (linked via NER from seed
chunk text; entity nodes carry no tenant ownership).
1AF(ε) is sensitive to the choice of ε; see Appendix F for a robustness
check acrossε∈ {0.01,0.05,0.1,0.5}.3.Hop 2: Unauthorized chunk (connected to the shared en-
tity via MENTIONS edge; belongs to a different tenant
or higher sensitivity tier).
This 2-hop pattern is inherent to any hybrid RAG system
that constructs a bipartite graph between chunks and entities
with expansion depth d≥2 : the entity-linking step creates a
structural bridge between vector-retrieved content and graph-
stored content. The PD=2 finding isstructural in our bipartite
construction—knowledge graphs with richer entity-to-entity
relationships (e.g., ontological hierarchies, multi-hop infer-
ence chains) may exhibit leakage at deeper pivot depths. We
discuss how PD varies with traversal parameters in §7.9.
7.3 Organic Leakage: No Injection Required
A critical observation from our 350 benign queries is that
RPR=0.954 without any adversarial injection. The 40 nat-
urally shared entities across tenant boundaries—shared in-
frastructure, vendors, personnel, compliance standards, and
projects—provide sufficient pivot paths for massive leakage
through ordinary queries. In our corpus, 334 of 350 benign
queries (95.4%) trigger cross-tenant leakage through organic
entity connections. This means the vulnerability isstructural:
it exists in any multi-tenant hybrid RAG deployment where
tenants share real-world entities, regardless of whether an
attacker injects content.
Bridge category analysis.To understand which types of
shared entities drive leakage, we analyze the hop-1 pivot
nodes in each leaking query’s traversal path and classify them
against the 5 bridge entity categories (Table 4).
Personnel entities (shared employees like “Maria Chen”)
dominate: they appear at hop 1 in 31% of benign leaking
queries and 47% of adversarial leaking queries, account-
ing for 23.6% and 42.8% of attributed leakage respectively.
Compliance entities (SOC2, PCI-DSS, ISO27001) rank sec-
ond. Notably, 52% of benign leaking queries haveno rec-
ognized bridge entityat hop 1—non-bridge entities (mone-
tary amounts, dates, generic organizational terms extracted
by spaCy NER) also create unintended cross-tenant paths,
8

Table 4: Organic leakage by bridge entity category under P3
(no injection). Queries = leaking queries with that bridge type
at hop 1. Leak = attributed leaked items.
Bridge Category Benign Adversarial
Queries Leak Queries Leak
Personnel 104 1,321 67 1,244
Compliance 46 782 33 489
Infrastructure 43 406 16 213
Vendor 0 0 30 218
Project 7 87 24 228
(No bridge at hop 1) 175 — 22 —
Total leaking 334 5,595 142 2,907
Table 5: Connectivity sweep: effect of bridge entity count on
P3 RPR and mean Leakage@k. RPR is remarkably stable;
mean leakage increases monotonically with connectivity.
Bridges P3 RPR Mean Leak PD
0 0.93 21.5 2.0
5 0.95 26.6 2.0
10 0.95 30.5 2.0
15 0.95 31.5 2.0
25 0.95 32.5 2.0
40 0.94 34.5 2.0
indicating that the leakage risk is broader than just named
shared entities.
7.4 Connectivity Sensitivity
To measure how shared-entity density affects leakage,
we regenerate the corpus with bridge entity counts ∈
{0,5,10,15,25,40} , rebuild all indexes for each count, and
run 100 adversarial queries through P3 (Table 5).
Two findings emerge. First,RPR is remarkably stable
(0.93–0.95) regardless of bridge count—even with zero in-
tentional bridge entities, organic entity overlap from spaCy
NER produces RPR =0.93 . Second,mean leakage scales
monotonicallyfrom 21.5 (0 bridges) to 34.5 (40 bridges),
a 60% increase. Bridge entities do notenableleakage (BFS
already reaches cross-tenant nodes through generic NER enti-
ties), but theyamplifythe volume of leaked content per query.
PD remains uniformly 2.0 across all bridge counts, confirming
the structural bipartite pivot.
7.5 Embedding Model Sensitivity
To verify that the pivot vulnerability is structural rather than
embedding-dependent, we repeat the P1 and P3 evaluations
using all-mpnet-base-v2 (768 dimensions, higher retrieval
quality) in addition to our default all-MiniLM-L6-v2 (384Table 6: Embedding model sensitivity: P3 RPR and mean
Leakage@k across two embedding models. The vulnerability
persists regardless of embedding quality.
Model Benign Adversarial
RPR Leak RPR Leak
MiniLM-L6 (384d) 0.954 16.0 0.947 19.4
MPNet (768d) 0.994 19.4 0.980 28.4
Table 7: RPR under each attack type across pipeline vari-
ants. Mean Leakage@k shown in parentheses. All attacks fail
against D1-defended pipelines.
Attack P3 P4 P6 P8
A1 (Seed Steer) 1.00 (20.5) 0.00 (0.0) 0.00 (0.0) 0.00 (0.0)
A2 (Entity Anchor) 1.00 (20.5) 0.00 (0.0) 0.00 (0.0) 0.00 (0.0)
A3 (Nbhd Flood) 1.00 (20.5) 0.00 (0.0) 0.00 (0.0) 0.00 (0.0)
A4 (Bridge Node) 1.00 (20.5) 0.00 (0.0) 0.00 (0.0) 0.00 (0.0)
dimensions). We rebuild the ChromaDB collection for each
model and run the full 500-query evaluation (Table 6).
The higher-quality MPNet model actuallyincreasesRPR:
from 0.954 to 0.994 on benign queries, and from 0.947 to
0.980 on adversarial queries. Mean leakage also rises—from
16.0 to 19.4 (benign) and from 19.4 to 28.4 (adversarial). Bet-
ter embeddings retrieve more relevant seed chunks, which in
turn mention more entities, which seed more graph expan-
sion into cross-tenant neighborhoods. Both models produce
PD=2.0 uniformly, confirming that the bipartite pivot struc-
ture is model-independent. P1 achieves RPR=0.0 under
both models, confirming that vector-side tenant prefiltering
remains effective regardless of embedding quality.
7.6 Attack Evaluation
Table 7 presents RPR under each attack across pipeline vari-
ants. All four attacks achieve RPR=1.0 against the unde-
fended hybrid pipeline (P3)—every adversarial query pro-
duces cross-tenant leakage. Critically, all four attacks achieve
RPR=0.0 against every defended variant (P4, P6, P8).
Table 8 provides injection details for each attack. The at-
tacks span a range of injection budgets (9–20 chunks) and
entity strategies (1–2 target entities, 1–6 MENTIONS edges
created).
Two aspects of this uniformity are themselves findings.
Leakage is bounded by the expansion window, not the
attack mechanism: all four attacks produce identical Leak-
age@k ( ≈20.5) against P3 because the bottleneck is the
unguarded graph traversal—all four simply steer queries
into the same undefended 2-hop expansion path, and the to-
tal_nodes budget (100) caps the leaked volume. This means
more sophisticated injection strategies yield no additional
9

Table 8: Attack injection details. All attacks achieve RPR=1.0
on P3 and RPR=0.0 on P4/P6/P8.
Attack Payloads Chunks Entities Mentions
A1 9 9 1 3
A2 10 10 1 6
A3 20 20 1 4
A4 15 15 2 6
Table 9: RPR under adaptive attacks (A5–A7) across pipeline
variants. A5 tests metadata forgery at three rates; A6 tests en-
tity manipulation; A7 tests query manipulation. Leakage@k
shown in parentheses.
Attack Rate P3 P4 P7 P8
A5 1% 1.00 (7.8) 0.00 (0.0) 0.00 (0.0) 0.00 (0.0)
A5 5% 1.00 (7.8) 0.00 (0.0) 0.00 (0.0) 0.00 (0.0)
A5 10% 1.00 (7.8) 0.00 (0.0) 0.00 (0.0) 0.00 (0.0)
A6 — 1.00 (7.8) 0.00 (0.0) 0.00 (0.0) 0.00 (0.0)
A7 — query-only (no injection)
leakage beyond what organic entity overlap already provides.
Second, D1 blocksallfour strategies because it operates on
nodeproperties(tenant, sensitivity) rather than graphstruc-
ture(paths, edges). The attacks manipulate paths—optimizing
similarity (A1), creating dense connections (A2), inflating
density (A3), or bridging boundaries (A4)—but D1 filters on
properties at each hop, making the path irrelevant.
7.7 Adaptive Attacks (A5–A7)
We extend the attack taxonomy with three adaptive strategies
that target the defense mechanisms themselves (Table 9).
A5 (Metadata Forgery)relabels injected nodes with the
target tenant’s name to bypass D1’s tenant check. At forgery
rates of 1%, 5%, and 10%, A5 achieves RPR=1.0 against P3
butRPR=0.0 against all defended pipelines (P4, P7, P8). D1
remains effective because the forged metadata only affects
the attacker’s own injected nodes—the organic entity nodes
that create the pivot path still carry empty-string tenant labels
and are filtered.
A6 (Entity Manipulation)creates documents mentioning
entities from the target tenant’s namespace, attempting to
create new shared entity nodes. A6 also fails against D1:
the newly created entity nodes still carry empty-string tenant
labels, and the defense filters them regardless of how they
were created.
A7 (Query Manipulation)crafts queries mentioning tar-
get tenant entity names to steer NER-based entity linking
toward sensitive neighborhoods. This is a query-only attack
(no injection) and produces the same RPR as organic benign
queries—confirming that the vulnerability is in the graphTable 10: Generation contamination metrics across datasets
and LLMs. ECR = Entity Contamination Rate, ILS = Infor-
mation Leakage Score, FCR = Factual Contamination Rate,
GRR = Generation Refusal Rate. n= queries with leakage
evaluated.
Dataset LLM ECR ILS FCR GRR n
SyntheticGPT-4o 0.077 0.305 0.050 0.800 10
Claude Sonnet 4.5 0.321 0.352 0.072 0.047 64
DeepSeek-V3 0.147 0.333 0.044 0.045 134
EnronGPT-5.2 0.082 0.285 0.005 0.000 181
Claude Sonnet 4.5 0.133 0.280 0.004 0.000 40
DeepSeek-V3 0.053 0.260 0.056 0.050 101
EDGARClaude Sonnet 4.5 0.040 0.358 0.000 0.019 53
DeepSeek-V3 0.025 0.345 0.016 0.101 139
Note:Synthetic Claude Sonnet 4.5 and EDGAR Claude Sonnet 4.5 used
self-judging; all other rows used Claude Sonnet 4.5 as judge. Synthetic
GPT-4o ( n=10 ) used self-judging. ECR drops from synthetic to Enron to
EDGAR as entity distinctiveness decreases. EDGAR shows the lowest
ECR (≤0.04 ) because financial entity names appear naturally in any
financial answer.
expansion, not in query processing.
7.8 Generation Impact
To measure whether leaked context actually contaminates
LLM-generated answers, we evaluate production LLMs on
queries with known leakage across all three corpora (Ta-
ble 10).
The results reveal model-dependent and dataset-dependent
contamination behavior. On the synthetic corpus, Claude
Sonnet 4.5 exhibits the highest Entity Contamination Rate
(ECR=0.32 ,n=64 ), indicating it readily incorporates leaked
entities into generated answers. GPT-4o shows the opposite
pattern: ECR=0.08 andGRR=0.80 (n=10 ), largely ignor-
ing leaked context. DeepSeek-V3 falls between ( ECR=0.15 ,
n=134 ) with low GRR ( 0.05) and FCR ( 0.04). All three mod-
els show GRR ≤0.05 at scale, suggesting that generation
refusal is rare when leaked context is topically relevant.
On the Enron corpus ( n=181 for GPT-5.2, n=40 for
Claude, n=101 for DeepSeek), ECR drops substantially
across all models (GPT-5.2 0.08; Claude 0.13; DeepSeek
0.05) while GRR approaches zero (all models ≤0.05 ). Real
email content is contextually relevant enough that LLMs
rarely ignore it, but leaked entities from adjacent departments
are less distinctive than synthetic ones, producing lower entity
contamination. GPT-5.2 shows moderate ECR ( 0.082 ) with
near-zero FCR ( 0.005 ), suggesting it surfaces entity names
but avoids reproducing substantive facts from leaked context.
These findings confirm that retrieval-level leakage translates
to generation-level contamination across all three corpora,
that the severity depends on the model and the domain, and
that the effect is robust across sample sizes (n=40–181).
10

Figure 2: Traversal parameter sweep: context size vs. latency,
colored by RPR. The total node budget is the primary leakage-
controlling parameter.
7.9 Traversal Parameter Sweep
To understand which traversal parameters control leakage, we
sweep across 27 configurations: depth ∈ {1,2,3} , branching
factor∈ {5,10,25} , and total node budget ∈ {25,50,100} ,
running 100 adversarial queries per configuration against the
undefended hybrid pipeline (P3). Figure 2 shows the results.
Three findings emerge:
Total node budget is the primary control.Regardless of
depth or branching, total_nodes=25 yields RPR=0 (insuffi-
cient expansion to reach cross-tenant content), total_nodes=
50yields RPR≈0.66 , and total_nodes=100 yields RPR≈
0.93. This is because the total node budget caps how many
graph nodes are gathered, and once this cap prevents expan-
sion from reaching the 2-hop cross-tenant chunks, leakage is
eliminated.
Depth must be ≥2for leakage to occur.At depth=1 ,
RPR=0 regardless of branching or total node budget, because
single-hop expansion cannot cross the chunk →entity→chunk
pivot path. This confirms the structural PD=2 signature.
Branching factor is irrelevant given a total node cap.
At fixed depth and total node budget, branching ∈ {5,10,25}
produces identical RPR values. The BFS expansion fills the
node budget regardless of how many children are explored
per node.
7.10 Latency and Overhead
Table 11 presents latency and context size across pipeline
variants.
D1 adds negligible latency overhead—P3 to P4 shows
<1ms increase at p50 (26.7 →26.5ms benign). More striking,
the full defense stack (P8) is actuallyfasterthan undefendedTable 11: Latency (ms) and context size across pipeline vari-
ants.
Variant p50 p95 Mean Ctx
Benign Queries
P1 (Vector) 12.6 19.9 16.2 10
P3 (Hybrid) 26.7 38.1 32.4 110
P4 (+D1) 26.5 34.4 30.4 56
P6 (+D1–D3) 23.1 29.8 26.4 29
P8 (All) 22.7 28.6 25.6 20
Adversarial Queries
P1 (Vector) 12.3 34.6 23.5 10
P3 (Hybrid) 27.3 40.2 33.7 110
P4 (+D1) 26.3 32.2 29.3 50
P6 (+D1–D3) 23.1 33.7 28.4 28
P8 (All) 22.9 30.1 26.5 20
P3 (22.7 vs. 26.7ms) because budgeted traversal (D3) and
trust filtering (D4) reduce the number of nodes expanded
and processed, decreasing both graph query time and context
assembly overhead.
7.11 Metadata Integrity Stress Test
An adaptive attacker might attempt to circumvent D1 by
corrupting sensitivity labels. We test D1’s robustness by
randomly flipping sensitivity labels on r%of graph nodes
(r∈ {0.1,0.5,1.0,2.0,5.0} ) and measuring RPR under P4.
Results:D1 maintains RPR=0.0 at all mislabel rates up
to 5%.This robustness arises because D1’s primary protec-
tion is thetenantfilter (not the sensitivity filter): cross-tenant
leakage requires traversing to a node with a different tenant
label, and sensitivity mislabeling does not affect tenant as-
signments. We note that context size decreases slightly at
higher mislabel rates (50 →48 items at 5%) as some autho-
rized nodes have their sensitivity erroneously raised above
the user’s clearance.
7.12 Defense Ablation
The defense stack shows a clear pattern (Figure 4):
•D1 alone(P4): RPR drops from 0.95 to0.0. Context re-
duces from 110 to 50–56 items (the unauthorized items
are removed; authorized graph-expanded content is re-
tained).
•D1 + D2(P5): No additional security improvement; con-
text remains similar (51–57). Edge allowlisting provides
defense in depth but does not further reduce leakage
already eliminated by D1.
11

Figure 3: D1 robustness under metadata corruption. RPR
remains 0.0 even at 5% mislabel rate. Context size decreases
slightly as erroneously up-labeled nodes are filtered.
Figure 4: Mean context size under progressive defenses. D1
alone reduces context from 110 to 50–56 items (removing
unauthorized content). D3–D5 further reduce noise, reaching
19–20 items with the full defense stack.
•D1–D3(P6): Context drops to 28–29. Budgeted traversal
caps the number of expanded nodes, reducing context
noise from authorized but irrelevant graph content.
•D1–D4(P7): Context drops to 24–28. Trust-weighted
filtering removes low-provenance nodes.
•D1–D5(P8): Context reaches 19–20 items. The merge
filter provides a final defense-in-depth check, reducing
context by 82% from the undefended baseline.
The key insight is that D1 is bothnecessary and sufficient
for security (eliminating all leakage), while D2–D5 serve
asutility optimizers(reducing context noise and providing
defense in depth).
7.13 Utility Impact
A critical question for practitioners is whether defenses de-
grade retrieval quality. Since our evaluation corpus uses syn-
thetic documents without human-annotated relevance labels,Table 12: Security-utility tradeoff across pipeline variants (be-
nign queries, n=350 ). Auth. Items = mean authorized items
in context. Auth. Rate = fraction of context that is authorized.
Retention = authorized items relative to P3’s authorized base-
line (94 items). D1 eliminates all leakage while retaining
5.6×more content than vector-only.
Variant RPR Ctx Auth. Auth. Rate Retention
P1 (vector) 0.000 10 10.0 1.00 —
P3 (hybrid) 0.954 110 94.0 0.85 1.00
P4 (+D1) 0.000 56 56.0 1.00 0.60
P6 (+D1–3) 0.000 29 29.0 1.00 0.31
P8 (+D1–5) 0.000 20 20.0 1.00 0.21
we measure utility through two proxy metrics: (1)autho-
rized context size—the number of items passing authoriza-
tion checks in the final context, and (2)authorization rate—
the fraction of context items that are authorized (i.e., not
leaked). Together these quantify how much useful content
each defense preserves and how much noise it eliminates.
Table 12 presents the security-utility tradeoff. P3 (unde-
fended hybrid) returns 110 items per query, but only 85.5%
are authorized—the remaining 16 items (14.5%) are cross-
tenant or over-clearance leakage. P1 (vector-only) returns
10 fully authorized items, establishing the vector-only utility
baseline.
D1 (P4) retains 56 authorized items per query—a 40%
reduction from P3’s 94 authorized items, driven entirely by
entity node removal (§8). However, P4 still provides 5.6 ×
more authorized content than P1’s vector-only baseline (56
vs. 10), confirming thatgraph expansion retains substantial
utility even after D1 filtering. The lost items are entity nodes
and their dependent traversal paths, not authorized chunks
from the user’s own tenant.
D3–D5 progressively reduce context size (56 →29→20)
by capping traversal depth, filtering low-provenance nodes,
and removing noise at the merge stage. Even the most ag-
gressive configuration (P8) retains 2 ×more content than
vector-only retrieval. Under adversarial queries, the pattern
is similar: P4 retains 50 authorized items (retention 0.55 vs.
P3’s 90.6 authorized baseline).
The practical conclusion:D1 eliminates all leakage while
preserving 5.6 ×the content of vector-only retrieval.The
additional defenses (D3–D5) are context quality optimizers
that reduce noise for downstream LLM generation. Organi-
zations prioritizing breadth should deploy D1 alone; those
prioritizing signal-to-noise ratio should add D3 and D4.
7.14 Cross-Dataset Validation
To verify that Retrieval Pivot Risk is not an artifact of synthetic
corpus construction, we repeat the baseline evaluation (P1,
P3, P4) on the Enron email corpus (§6.3) and EDGAR 10-
12

Table 13: Cross-dataset baseline: retrieval security metrics for
undefended (P3) and defended (P4) hybrid RAG across three
corpora. RPR = Retrieval Pivot Risk, Leak = mean leaked
items per query, PD = Pivot Depth (hops), Ctx = mean context
size.
Dataset Pipeline RPR Leak PD Ctx
SyntheticP1 (Vector) 0.000 0.0 – 10
P3 (Hybrid) 0.954 16.0 2.0 110
P4 (+D1) 0.000 0.0 – 56
EnronP1 (Vector) 0.000 0.0 – 10
P3 (Hybrid) 0.695 7.1 2.0 45
P4 (+D1) 0.000 0.0 – 25
EDGARP1 (Vector) 0.000 0.0 – 10
P3 (Hybrid) 0.085 0.4 2.0 39
P4 (+D1) 0.000 0.0 – 22
Note:Synthetic corpus: 1,000 documents, 4 tenants, 40 bridge entities.
Enron corpus: 50,000 emails, 5 departments, 19 cross-department entities.
EDGAR corpus: 887 10-K filing sections, 4 sector-tenants, 13 cross-
sector entities. All corpora evaluated with 200 benign queries.
K filings (§6.4). Table 13 compares results across all three
corpora.
Four findings emerge from the cross-dataset comparison:
The vulnerability generalizes.Enron’s undefended hybrid
pipeline (P3) produces RPR=0.695 on benign queries—139
of 200 queries leak cross-department content, with a mean of
7.1 leaked items per query. EDGAR produces RPR=0.085 —
lower but still affecting 17 of 200 queries. The vulnerability
is not an artifact of synthetic construction: real corporate data
with natural organizational boundaries exhibits the same pivot
mechanism.
RPR scales with entity connectivity.The three corpora
span a wide range: synthetic ( RPR=0.954 , 40 bridge enti-
ties), Enron (RPR=0.695, 19 shared entities), and EDGAR
(RPR=0.085 , 13 cross-sector entities). RPR correlates with
the number and density of cross-tenant entity connections.
The synthetic graph has curated cross-references, Enron has
informal cross-department email traffic, and EDGAR’s formal
financial language creates fewer organic entity connections
across sectors. This confirms that RPR is a function of entity
connectivity density, not corpus size.
D1 eliminates all leakage on all three corpora.P4
achieves RPR=0.0 on Enron (context size 25), EDGAR
(context size 22), and synthetic (context size 56). The defense
generalizes without modification: the same per-hop tenant-
and-sensitivity check eliminates leakage regardless of corpus
size, entity distribution, or graph topology.
PD = 2 persists.All leakage in all three corpora occurs
at exactly PD=2 , confirming that the bipartite pivot (chunk
→entity→chunk) is a structural invariant of hybrid RAG
systems that construct knowledge graphs via entity linking,
independent of the underlying document domain.Table 14: Defense mechanisms, pipeline integration points,
and experimental impact.
Defense Stage Effect
D1: Per-hop authz Post-expansion RPR→0.0
D2: Edge allowlist Traversal query Defense in depth
D3: Budget Traversal params Ctx 110→28
D4: Trust weight Post-expansion Low-prov removal
D5: Merge filter Post-merge Final backstop
8 Mitigations
We propose five defenses that operate at different stages of
the hybrid pipeline, forming a defense-in-depth architecture
(Table 14).
8.1 D1: Per-Hop Authorization
Per-hop authorization is anintentionally conservative
minimum-viable guardrail: it re-checks access control pred-
icates on every node reached during graph expansion. For
each expanded nodev, the defense evaluates:
auth(u,v) = (v.tenant=t u)∧(v.sensitivity≤ℓ u)(8)
Nodes failing this check are removed from the expansion
result before context assembly. In our implementation, D1
operates as apost-expansion filter: the BFS traversal runs
unrestricted, and the authorization check is applied to the re-
sult set. This is simpler to implement than in-traversal pruning
(which would stop expansion at unauthorized nodes) but is
less efficient because it expands nodes that will be discarded.
A true per-hop pruning implementation would stop expansion
at unauthorized nodes, preventing their children from being
explored. We chose post-expansion filtering for implemen-
tation simplicity; the security properties are identical (both
produce RPR=0.0 ), but in-traversal pruning would further
reduce latency and context processing overhead.
D1 is the most effective defense: it reduces RPR from 0.95
to 0.0 across all query types and all attack variants, with <1ms
latency overhead.
Entity tenant semantics.A subtle but critical design deci-
sion concerns entity nodes. In our knowledge graph, entity
nodes (extracted via NER) represent shared concepts—people,
systems, vendors—that may appear in documents across mul-
tiple tenants. These nodes carry tenant = "" (empty string)
rather than any specific tenant label, because they aretenant-
neutral: the concept “CloudCorp” belongs to no single tenant.
Under D1’s authorization check, empty-tenant nodes fail the
tenant-match predicate ( v.tenant̸=t u) and are filtered from
the expansion result. This means D1 removesall entity nodes
from the context—not just unauthorized ones.
13

This entity-level filtering is precisely the mechanism by
which D1 eliminates cross-tenant leakage. The 2-hop pivot
path (chunk →entity→chunk) requires traversal through
a shared entity node. By filtering entity nodes, D1 severs
this path at hop 1, preventing the traversal from reaching
the unauthorized chunk at hop 2. The trade-off is that entity
information (which may be useful for answer generation) is
excluded from the final context. We quantify this impact in
Section 9.
We acknowledge that D1’s entity filtering is functionally
equivalent to disabling entity traversal—a valid criticism.
However, this is theminimum viable defense, not the opti-
mal one. A finer-grainedentity-aware authorizationscheme
would: (a) label each entity with the set of tenants whose doc-
uments mention it (e.g., “CloudCorp” →{acme, globex}), (b)
allow traversalthroughentities whose tenant set includes tu,
and (c) apply the authorization check only atchunknodes (the
terminal nodes that carry sensitive text). This design would
preserve entity context for within-tenant traversals while still
blocking cross-tenant chunk access. We leave its implementa-
tion and security analysis to future work, and position D1 as
the conservative baseline that demonstrates theexistenceof
the boundary problem.
The critical insight is thatexisting graph databases already
store the metadata needed for this check(tenant labels, sensi-
tivity tiers). The defense does not require new infrastructure—
only the discipline to re-check authorization at the graph layer
rather than relying solely on the vector prefilter.
8.2 D1 Metadata Integrity Assumption
D1 assumes that node metadata (tenant, sensitivity) is trust-
worthy. If an attacker can forge metadata during document
injection, D1 can be bypassed. Our metadata stress test (§7.11)
shows that D1 is robust to random mislabeling up to 5%, but
targeted metadata forgery is a different threat. Mitigations
include: (a) system-assigned metadata during ingestion (the
uploader cannot choose tenant or sensitivity labels), (b) crypto-
graphic signing of metadata at ingestion time, and (c) periodic
metadata audits that compare node labels against source-of-
truth identity systems. In practice, most enterprise document
management systems already enforce system-assigned tenant
labels, making (a) the natural deployment model.
8.3 D2: Edge Allowlist
Edge allowlisting restricts graph traversal to pre-approved
edge types per query class. The expander’s BFS query in-
cludes a relationshipFilter parameter that limits which
edge types can be traversed. For example, general queries may
traverse MENTIONS, DEPENDS_ON, and BELONGS_TO
edges, while RELATED_TO edges (which are the primary
vector for bridge attacks) are excluded.8.4 D3: Budgeted Traversal
Budgeted traversal enforces hard limits on BFS expansion:
maximum hop depth, maximum branching factor per node,
and maximum total expanded nodes. Our traversal sweep
(§7.9) shows that total_nodes≤25 eliminates leakage even
without D1, while the default budget ( dmax=2, branching
≤10 , total≤50 ) reduces context from 56 items (D1 only) to
28 items, removing authorized but irrelevant graph content
that adds noise.
8.5 D4: Trust-Weighted Expansion
Trust-weighted expansion filters expanded nodes by their
provenance score. Each node carries a provenance score ( 0.0–
1.0) based on its source reliability. The defense applies a
minimum threshold (default: 0.6), removing low-provenance
content. This is particularly effective against injected attack
payloads, which carry provenance scores of 0.3–0.4 in our
attack implementations.
8.6 D5: Merge-Time Policy Filter
The merge-time policy filter performs a final access control
check after vector and graph results are combined. This de-
fense acts as a backstop: even if earlier defenses miss an
unauthorized item, the merge filter removes any item whose
sensitivity exceeds the user’s clearance before the context
reaches the LLM.
9 Discussion
9.1 Security-Utility Tradeoff
Our results reveal a favorable security-utility tradeoff. D1
alone achieves complete security ( RPR=0.0 ) while retaining
50% of the graph-expanded context ( 110→50–56 items).
The retained items are authorized content that contributes to
answer quality. Adding D3–D5 further reduces context to 19–
20 items—an 82% reduction from the undefended baseline—
but the additional defenses trade context volume for noise
reduction rather than security improvement.
Entity over-filtering.D1’s security guarantee comes at a
specific cost: entity nodes are excluded from the final context.
In our bipartite construction, the undefended P3 pipeline re-
turns approximately 110 items per query, of which roughly
25–35 are entity nodes. After D1 filtering, these entity nodes
are removed entirely, leaving 50–56 chunk-only items. The
entity information they carried (names, relationships, organi-
zational context) is lost. For queries that benefit from entity-
level knowledge (e.g., “Who works on project Alpha?”), this
may reduce answer quality compared to an oracle that retains
only authorized entity nodes. We consider this an acceptable
14

trade-off: the entity nodes are exactly the pivot vectors that
enable cross-tenant leakage, and their removal is the mech-
anism by which D1 achieves RPR=0.0 . Future work on
multi-tenant entity authorization (Section 10) could recover
entity utility while preserving security.
The practical recommendation is clear:deploy D1 imme-
diately as a minimum viable defense.Add D3 and D4 if
context noise is impacting LLM generation quality. D2 and
D5 provide defense in depth for compliance-sensitive deploy-
ments. Practically, this means systems should treat tenant-
neutral entity nodes asprivileged pivot infrastructurethat
must carry explicit authorization semantics (or be excluded)
in multi-tenant deployments; leaving them unlabeled creates
a structurally inevitable cross-tenant pivot path.
9.2 Why PD = 2 Across All Three Corpora
Proposition (Pivot depth in bipartite entity-link graphs).
In a bipartite graph constructed from chunk nodes and entity
nodes via NER-based entity linking, any entity co-mentioned
by chunks from different tenants induces a length-2 path be-
tween those chunks (chunk →entity→chunk). Therefore,
if cross-tenant leakage occurs via shared entities under ex-
pansion depth d≥2 , the minimum pivot depth to the first
unauthorized chunk is PD=2.
This follows directly from the bipartite construction: the
entity-linking step creates a 2-hop path between any two
chunks that mention the same entity. Chunk A (hop 0) →
shared entity (hop 1) →Chunk B (hop 2). Every shared entity
is a potential pivot, and the number of pivot opportunities
scales with the number of cross-boundary entity mentions.
The Enron and EDGAR corpora confirm this structural
invariant: despite graphs ranging from 2.8K to 376K nodes,
varying NER quality, and document types from curated tem-
plates to informal email to formal financial filings, all leakage
occurs at exactly PD=2 . The synthetic corpus has 40 shared
entities, Enron has 19, and EDGAR has 13. All three produce
the same pivot depth. This confirms that PD=2 is a conse-
quence of the bipartite construction, not of corpus-specific
properties.
Knowledge graphs with richer topologies—ontological hi-
erarchies, multi-hop inference chains, entity-to-entity rela-
tionships beyond simple co-mention—may exhibit leakage
at PD >2. Our traversal sweep confirms that depth=1 pre-
vents all leakage in the bipartite construction, but graphs with
entity-entity edges would require analysis at deeper traversal
depths.
9.3 The Amplification Mechanics
The core insight of this work is not simply “apply ACLs to
graph expansion” but rather theidentification and character-
ization of a compound attack surfacethat arises from com-
bining two individually secure retrieval modalities. Vectorretrieval with tenant prefiltering is secure ( RPR=0.0 ). Graph
retrieval within a single tenant’s subgraph is secure. But con-
necting vector retrieval outputs to graph expansion inputs—
the pivot boundary—creates leakage that neither modality
exhibits alone.
This is analogous to composition vulnerabilities in other
security domains: two individually secure components can
produce an insecure system when composed. The amplifica-
tion factor quantifies the severity: AF(ε)≈160–194× on the
synthetic corpus and AF(ε)≈70× on Enron. Hybrid RAG
does not incrementally increase leakage—it creates 70–194×
more leaked items than the vector baseline from which it
amplifies.
The defense implication is equally specific: the authoriza-
tion check must be placedat the boundary, after graph ex-
pansion and before context assembly. Authorization at the
vector layer alone (which every production system already
has) is insufficient. Authorization at the LLM layer (e.g.,
instruction-following) is unreliable. Only authorization at the
graph expansion layer—the pivot boundary itself—addresses
the root cause.
9.4 Comparison with Prior Art
GRAGPoison [10] demonstrates that graph structure amplifies
small poisoning inputs (98% ASR from relation injection).
Our work extends this insight to hybrid architectures: the
combinationof vector retrieval and graph expansion creates
a compound attack surface where authorized vector results
seed unauthorized graph traversals.
PoisonedRAG [28] achieves 90% ASR by injecting 5 docu-
ments into vector stores. In vector-only RAG, this is contained
by prefilters. In hybrid RAG, even a single retrieved document
that mentions a shared entity can trigger graph expansion into
sensitive neighborhoods, making the effective attack surface
orders of magnitude larger.
The Graph RAG Privacy Paradox [12] observes that graph
RAG reduces text leakage but increases structural informa-
tion extraction. Our findings align: the hybrid pipeline’s graph
expansion systematically transforms authorized text into unau-
thorized structural access.
9.5 Implications for Agentic Systems
The pivot attack is especially dangerous in agentic RAG de-
ployments (LangGraph, CrewAI) where graph traversal is per-
formed autonomously by an LLM agent [8]. In these systems,
the agent decides traversal depth, edge types, and expansion
strategies without human oversight. A poisoned vector seed
can manipulate the agent into performing unrestricted graph
exploration, and research shows that a single compromised
agent can poison 87% of downstream decision-making within
4 hours [8]. Per-hop authorization is even more critical in
15

agentic settings, where there is no human in the loop at the
expansion step.
10 Limitations
Three corpora.We evaluate on a synthetic enterprise corpus,
the Enron email corpus, and SEC EDGAR 10-K filings. The
synthetic corpus validates the vulnerability under controlled
conditions; the Enron and EDGAR corpora confirm general-
ization to real-world data with varying entity connectivity den-
sities. However, all three use a bipartite chunk–entity graph
topology that produces a uniform PD=2 signature. Knowledge
graphs with richer entity-to-entity relationships (ontological
hierarchies, multi-hop inference chains) may exhibit leakage
at deeper pivot depths. We do not evaluate on production enter-
prise graphs with RBAC/ABAC policies, which may exhibit
different connectivity patterns.
Generation evaluation scope.Our generation contamina-
tion experiment (§7.8) evaluates three LLMs across all three
corpora ( n=40 –181 leaking queries on Enron, n=64 –134 on
synthetic, n=53 –139 on EDGAR) and confirms that leaked
context contaminates generated answers (ECR up to 0.32,
FCR up to 0.07). The GPT-4o synthetic sample ( n=10 ) pre-
dates our expanded query pool; larger samples across Enron
and EDGAR provide more robust estimates. We do not eval-
uate chain-of-thought reasoning, which might amplify the
impact by synthesizing cross-tenant connections.
Two embedding models.We evaluate two open-source
models (all-MiniLM-L6-v2, 384d, and all-mpnet-base-v2,
768d) and confirm the vulnerability persists across both (§7.5).
However, we do not test commercial embedding models (e.g.,
OpenAI text-embedding-3-large) which may produce differ-
ent similarity distributions. The pivot vulnerability is struc-
tural (entity linking, not embedding quality), so we expect
model independence to hold broadly.
Entity over-filtering.D1 removesallentity nodes from the
expansion result because entities carry empty-string tenant
labels. This is the mechanism that eliminates leakage, but it
also removes potentially useful entity context. A finer-grained
approach would assign entities to the set of tenants whose
documents mention them, or introduce a SHARED tenant class
with explicit cross-tenant grants. Such designs would recover
entity utility at the cost of more complex authorization logic
and a larger attack surface.
Adaptive attacker scope.We evaluate three adaptive at-
tacks (A5–A7) including targeted metadata forgery at rates up
to 10%, entity manipulation, and query manipulation (§7.7).
All defended pipelines maintain RPR=0.0 . However, we do
not evaluate attackers who craft high-provenance payloads to
bypass D4 specifically, or adversarial query phrasing to ma-
nipulate D2’s query classifier. Our metadata forgery assumes
the attacker can relabel node properties but cannot modify the
graph schema or index structure.NER and entity linker dependence.The pivot path ex-
ists because entity linking creates shared nodes across tenant
boundaries. All three corpora use spaCy’sen_core_web_sm
NER model, which has known recall limitations for domain-
specific entities. On the Enron corpus, this produces noisier
entity extractions (dates, monetary amounts, partial names)
that create spurious cross-tenant connections—yet RPR is
lower (0.695 vs. 0.954), suggesting that NER noise creates
fragmented rather than dense pivot paths. A higher-recall
linker would extract more shared entities, potentially increas-
ing organic leakage. We do not evaluate how linker quality
affects RPR, nor do we test production entity linking systems
that perform coreference resolution or cross-document entity
merging.
Simplified policy model.Our authorization model uses
tenant labels and four sensitivity tiers (PUBLIC through
RESTRICTED). Production enterprises typically employ
richer access control: role-based (RBAC) and attribute-based
(ABAC) policies, group memberships, temporary grants, le-
gal holds, and need-to-know exceptions. D1’s per-hop check
generalizes to any predicate that can be evaluated on node
metadata, but we do not demonstrate this generality exper-
imentally. The gap between our flat tenant model and real-
world RBAC/ABAC hierarchies remains an open integration
question.
Graph-only baseline (P2).We omit a graph-only RAG
baseline because graph-only retrieval without vector seeding
does not involve the pivot boundary under study. This means
our results characterize thehybrid-specificvulnerability but
do not compare against graph-only retrieval’s independent
security properties.
11 Related Work
11.1 Vector RAG Poisoning
PoisonedRAG [28] formalized knowledge corruption attacks
on RAG, achieving 90% ASR with 5 injected documents.
CorruptRAG [24] demonstrated single-document attacks with
higher stealth. CtrlRAG [18] achieved 90% ASR on GPT-4o
via black-box feedback optimization. RIPRAG [21] applied
reinforcement learning to optimize poisoning without model
access. NeuroGenPoisoning [27] targeted specific neurons
for >90% success. These works study vector-side attacks in
isolation; none address what happens when poisoned vector
results seed graph expansion.
11.2 GraphRAG Security
GRAGPoison [10] is the closest related work, demonstrat-
ing relation-centric poisoning of GraphRAG with 98%
ASR. TKPA [20] showed that modifying 0.06% of corpus
text drops GraphRAG accuracy by 45 percentage points.
16

RAGCrawler [22] achieved 84.4% knowledge extraction cov-
erage through graph-guided probing. The Graph RAG Privacy
Paradox [12] established that graph RAG increases structural
leakage while reducing text leakage. Our work extends these
graph-side insights to the hybrid setting, where the vector-to-
graph transition creates a compound attack surface.
11.3 RAG Privacy and Extraction
The SoK on RAG Privacy [4] systematized all known pri-
vacy attack vectors in RAG systems and explicitly noted that
hybrid RAG privacy risks remain under-studied. Riddle Me
This [13] demonstrated membership inference on RAG sys-
tems. Traceback of RAG Poisoning [23] provided forensic
methods for identifying responsible documents, but traces
only through the vector retrieval path without following graph
expansion chains.
11.4 RAG Defenses
RAGuard [5] detects poisoning via retrieval pattern analy-
sis. SeCon-RAG [17] applies semantic consistency filtering.
RevPRAG [19] achieves 98% detection via LLM activation
analysis. SDAG [7] partitions context into trusted and un-
trusted segments with block-sparse attention. SD-RAG [3]
implements selective disclosure policies. All operate within
a single retrieval modality. Our defense suite (D1–D5) is the
first designed specifically for the cross-store boundary, with
per-hop authorization as the cornerstone mechanism.
Ethical Considerations
Experiments use three corpora: a synthetic corpus gener-
ated by the authors, the Enron email corpus (a public-record
dataset released during the 2001 FERC investigation and
widely used in NLP research [6]), and SEC EDGAR 10-K
filings (publicly available regulatory documents). No produc-
tion enterprise data, user accounts, or confidential documents
were used. The attack implementations (A1–A7) target our
own evaluation infrastructure and are designed to characterize
vulnerabilities, not exploit production systems. Our primary
contribution is the defense (D1), which we release alongside
the attack code to ensure that the net effect of publication is
protective. The released code does not include tools for tar-
geting external systems; all components require a local Neo4j
and ChromaDB deployment to operate.
Open Science
We release the complete research artifact: source code, syn-
thetic data generators, Enron ingestion pipeline, all seven
attack implementations (A1–A7), the five-layer defense suite
(D1–D5), evaluation harness, query templates, and raw ex-
perimental results. The repository includes 255 passing unittests and deterministic reproduction via a fixed random seed
(42). All experiments run on commodity hardware ( <16 GB
RAM, CPU-only) using open-source models and databases
(spaCy, Sentence Transformers, Neo4j Community Edition,
ChromaDB). The Enron email corpus is a public-record
dataset available from Carnegie Mellon University [6]. No
proprietary models, commercial APIs, or restricted datasets
are required to reproduce the core retrieval pivot results. The
generation contamination evaluation (§7.8) uses commercial
LLM APIs (OpenAI, Anthropic, DeepSeek) and is therefore
dependent on API availability and pricing; we include cached
results for reproducibility without API access.
12 Conclusion
Hybrid RAG pipelines that combine vector retrieval with
knowledge graph expansion introduce an attack surface at the
vector-to-graph boundary that has received limited explicit
treatment in prior work. We formalized this threat as Retrieval
Pivot Risk (RPR) and demonstrated across three corpora—a
synthetic enterprise dataset (1,000 documents, 4 tenants), the
Enron email corpus (50,000 emails, 5 departments), and SEC
EDGAR 10-K filings (887 sections, 4 sector-tenants)—that
undefended hybrid pipelines exhibit RPR=0.95 ,0.70, and
0.09 respectively, compared to 0.0for vector-only retrieval.
All leakage occurs at 2 hops through the entity-linking pivot
in our bipartite graph construction, a structural invariant that
holds across all three corpora.
Our taxonomy of seven Retrieval Pivot Attacks (A1–A4
non-adaptive, A5–A7 adaptive) shows that small injection
budgets (10–20 chunks) can exploit this boundary, and that
even without adversarial injection, naturally shared entities
create organic pivot paths that leak cross-tenant data. The
Enron and EDGAR corpora confirm this on real data: 19
naturally shared entities across Enron departmental bound-
aries produce RPR =0.695, while 13 cross-sector entities in
EDGAR 10-K filings produce RPR =0.085 on benign queries
alone.
The most important finding is practical:per-hop
authorization—re-checking tenant and sensitivity labels
at each graph expansion step—eliminates all measured
retrieval pivot leakage across all three corpora, all seven
attack variants, and metadata forgery rates up to 10%.
This defense is simple, requires no model changes, adds <1ms
latency, and uses metadata already present in graph databases.
We recommend it as the minimum viable security control for
any hybrid RAG deployment.
Our traversal parameter sweep reveals that the total node
budget is the primary leakage-controlling parameter: limiting
expansion to ≤25 nodes eliminates leakage even without
authorization checks, while expansion depth must reach ≥2
for the pivot to occur. The full defense stack (D1–D5) reduces
context noise by 82% with zero residual leakage.
The cross-dataset comparison reveals that RPR scales with
17

entity connectivity density: the synthetic corpus (40 bridge
entities, curated cross-references) produces higher RPR than
Enron (19 natural bridges, sparser cross-department traffic).
This confirms that the vulnerability is structural but its severity
depends on the knowledge graph’s topology—denser entity
sharing means more pivot opportunities.
Our work opens several directions for future research: eval-
uation on production enterprise graphs with richer entity-
to-entity topologies (where PD >2 may emerge), high-
provenance payload crafting to bypass D4, multi-tenant entity
authorization (recovering entity context without re-enabling
the pivot path), and extension to agentic hybrid RAG where
graph traversal is autonomously controlled.
References
[1] MITRE ATLAS update october 2025, 2025.
[2]OWASP top 10 for large language model applications
2025, 2025.
[3]Aiman Al Masoud, Marco Arazzi, and Antonino No-
cera. SD-RAG: Prompt-injection-resilient selective dis-
closure.arXiv preprint arXiv:2601.11199, 2026.
[4]Andreea-Elena Bodea, Stephen Meisenbacher, Alexan-
dra Klymenko, and Florian Matthes. SoK: Privacy risks
and mitigations in RAG systems. InIEEE SaTML, 2026.
[5]Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan,
Zhuqing Liu, Xiaohua Hu, and Minghong Fang. RA-
Guard: Secure retrieval-augmented generation against
poisoning attacks. InIEEE BigData, 2025.
[6]William W. Cohen. The Enron email dataset. Technical
report, Carnegie Mellon University, 2015. Public record
released by FERC during the Enron investigation.
[7]Sagie Dekel, Moshe Tennenholtz, and Oren Kurland.
Addressing corpus knowledge poisoning using sparse at-
tention.arXiv preprint arXiv:2602.04711, 2026. Block-
sparse attention preventing cross-document interactions.
[8]Savi Grover. Vulnerabilities and risk analysis of multi-
agentic AI-RAG systems.European Journal of Artificial
Intelligence, 5(1), 2026.
[9]Kabul Kurniawan, Rayhan Firdaus Ardian, Elmar Kies-
ling, and Andreas Ekelhart. AgCyRAG: An agentic
knowledge graph based RAG framework for automated
security analysis.CEUR Workshop Proceedings, 4079,
2025.
[10] Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi
Zhu, Tanqiu Jiang, Neil Gong, and Ting Wang.
GraphRAG under fire. InIEEE Symposium on Security
and Privacy (S&P), 2026. 98% ASR via relation-centric
poisoning; to appear.[11] Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang,
Hanyu Wang, Feiyu Xiong, Jason Zhaoxin Fan,
Bo Tang, Shichao Song, Mengwei Wang, and Ji-
awei Yang. SafeRAG: Benchmarking security
in retrieval-augmented generation.arXiv preprint
arXiv:2501.18636, 2025.
[12] Jiale Liu, Jiahao Zhang, and Suhang Wang. Expos-
ing privacy risks in graph RAG.arXiv preprint
arXiv:2508.17222, 2025. Graph RAG reduces text leak-
age but increases structured data extraction.
[13] Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaud-
hari, Alina Oprea, and Amir Houmansadr. Riddle me
this! stealthy membership inference for RAG. InACM
CCS, 2025.
[14] Pinecone. RAG with access control, 2024.
[15] Microsoft Research. GraphRAG: Unlocking LLM dis-
covery on narrative private data. 2024.
[16] Bhaskarjit Sarmah, Benika Hall, Rohan Rao, Sunil Pa-
tel, Stefano Pasquali, and Dhagash Mehta. HybridRAG:
Integrating knowledge graphs and vector retrieval aug-
mented generation for efficient information extraction.
arXiv preprint arXiv:2408.04948, 2024.
[17] Xiaonan Si, Meilin Zhu, Simeng Qin, Lijia Yu, Lijun
Zhang, Shuaitong Liu, Xinfeng Li, Ranjie Duan, Yang
Liu, and Xiaojun Jia. SeCon-RAG: A two-stage seman-
tic filtering and conflict-free framework for trustworthy
RAG. InNeurIPS, 2025.
[18] Runqi Sui. CtrlRAG: Black-box document poisoning
attacks for retrieval-augmented generation of large lan-
guage models.arXiv preprint arXiv:2503.06950, 2025.
90% ASR on GPT-4o with 5 poisoned documents per
query.
[19] Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping
Chen, and Jun Dai. RevPRAG: Revealing poisoning
attacks in retrieval-augmented generation through LLM
activation analysis. InFindings of EMNLP, 2025. 98%
TPR detection rate.
[20] Jiayi Wen, Tianxin Chen, Zhirun Zheng, and Cheng
Huang. A few words can distort graphs: Knowledge
poisoning attacks on graph-based RAG.arXiv preprint
arXiv:2508.04276, 2025. 93.1% success modifying
0.06% of text; UKPA drops QA from 95% to 50%.
[21] Meng Xi, Sihan Lv, Yechen Jin, Guanjie Cheng, Naibo
Wang, Ying Li, and Jianwei Yin. RIPRAG: Hack a
black-box RAG QA system with reinforcement learning.
2025.
18

[22] Mengyu Yao, Ziqi Zhang, Ning Luo, Shaofei Li, Yifeng
Cai, Xiangqun Chen, Yao Guo, and Ding Li. Connect the
dots: Knowledge graph-guided crawler attack on RAG
systems.arXiv preprint arXiv:2601.15678, 2026. 84.4%
corpus coverage within 1000 queries using KG-guided
extraction.
[23] B. Zhang et al. Traceback of poisoning attacks to RAG.
InACM Web Conference, 2025.
[24] Baolei Zhang, Yuxi Chen, Zhuqing Liu, Lihai Nie, Tong
Li, Zheli Liu, and Minghong Fang. Practical poisoning
attacks against retrieval-augmented generation. InACM
SACMAT, 2026. Single-document poisoning with higher
stealth.
[25] Tianhang Zhao et al. RAG safety: Exploring knowledge
poisoning attacks to KG-RAG.Information Fusion,
2025.
[26] Zexuan Zhong, Zhengxuan Huang, Alexander Wettig,
and Danqi Chen. Poisoning retrieval corpora by inject-
ing adversarial passages. InEMNLP, 2023.
[27] Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng,
and Long Jiao. NeuroGenPoisoning: Neuron-guided
attacks on RAG via genetic optimization. InNeurIPS,
2025.
[28] Wei Zou et al. PoisonedRAG: Knowledge corruption
attacks to retrieval-augmented generation of large lan-
guage models. InUSENIX Security Symposium, 2025.
A Artifact Appendix
A.1 Repository
The complete codebase, synthetic data generators, attack
implementations, defense suite, evaluation harness, and ex-
perimental results are available at: https://github.com/
scthornton/hybrid-rag-pivot-attacks
A.2 System Requirements
•Python 3.11+ with dependencies: chromadb ,neo4j ,
spacy,sentence-transformers,pydantic,numpy
•Neo4j 5.15+ with APOC plugin (for
apoc.path.spanningTree)
• ChromaDB server (latest)
• spaCy model:en_core_web_smA.3 Reproduction Steps
Synthetic corpus:
1.Clone repository and install: pip install -e
".[dev]"
2.Start services: docker compose up -d (Neo4j + Chro-
maDB)
3.Generate corpus: python
scripts/make_synth_data.py
4.Generate queries: python
scripts/generate_queries.py
5. Build indexes:python scripts/build_indexes.py
6.Run experiments: python
scripts/run_experiments.py -bootstrap
7.Run attack experiments: python
scripts/run_attack_experiments.py
8.Run sweeps: python
scripts/run_sweep_experiments.py
-traversal-sweep -mislabel-sweep
-connectivity-sweep
Enron corpus:
1.Ingest Enron emails: python
scripts/ingest_enron.py
2.Build indexes: python scripts/build_indexes.py
-dataset enron
3.Run experiments: python
scripts/run_experiments.py -dataset enron
Run tests:pytest tests/ -v(255 tests passing).
A.4 Runtime Estimates
Synthetic:Corpus generation and index building complete in
∼5 minutes. The full 500-query evaluation across 7 pipeline
variants takes ∼20 minutes. Attack experiments ( 4×4 con-
figurations) require ∼10 minutes including graph rebuilds.
Sweep experiments total∼45 minutes.
Enron:Ingestion and NER extraction takes ∼40 minutes
(2.07M entity mentions across 152K chunks). Neo4j graph
loading completes in ∼2 minutes using batch UNWIND
queries. The 200-query evaluation across P1/P3/P4 takes ∼10
minutes.
All experiments require <16 GB RAM and run on CPU-
only hardware.
19

B Corpus Generator Details
The synthetic corpus is produced by 12 domain-specific docu-
ment generators (3 per tenant), each instantiating one of three
genre templates:architecture description,policy/procedure,
andstatus report. Each generator populates templates from
curated entity pools:
•Engineering:12 system names, 15 technologies, 6
projects (e.g., auth-service ,Kubernetes ,Project
Alpha).
•Finance:10 vendors, 6 accounts, 7 regulations (e.g.,
Deloitte,SOX,Capital Expenditure 2025).
•HR:12 departments, 7 benefits, 10 named employees
(e.g.,Engineering,401k matching,Maria Chen).
•Security:6 CVEs, 8 tools, 6 frameworks (e.g.,
CVE-2025-41923,Splunk SIEM,NIST CSF).
Bridge entities span 5 categories: shared vendors ( CloudCorp ,
DataSyncInc ,SecureNetLLC ), shared infrastructure
(k8s-prod-cluster ,splunk-siem ,auth-service ),
shared personnel ( Maria Chen ,James Rodriguez ,Aisha
Patel ), shared compliance ( SOC2-audit ,PCI-DSS-cert ,
ISO27001 ), and shared projects ( ProjectNexus ,
ProjectHorizon ,ProjectArcade ). Sensitivity tiers
are assigned per-document: PUBLIC 40%, INTERNAL 30%,
CONFIDENTIAL 20%, RESTRICTED 10%. All generation
uses seed 42 for reproducibility.
C Query Templates
Benign queries(350 total, 5 templates ×4 domains ×vari-
able entity fills). Example templates:
Engineering: "What is the architecture of {system}?"
Finance: "Summarize the contract with {vendor}."
HR: "What are the policies for {department}?"
Security: "What is the severity of {cve}?"
Adversarial queries(150 total) are stratified across 4 attack
types:
A1 (Seed Steering): "What systems like {system}
handle {other_tenant_technology}?"
A2 (Entity Anchor): "Tell me about {bridge_entity}
and its role in {target_tenant_domain}."
A3 (Neighborhood Flood): "List everything related to
{bridge_entity}."
A4 (Bridge Node): "What connections exist between
{bridge_entity} across departments?"
Template slots are filled from aligned entity pools (iden-
tical to those in the corpus generator) to ensure query
entities appear in the knowledge graph. Each query car-
ries metadata: user_tenant (always acme_engineering ),user_clearance (stratified across PUBLIC, INTERNAL,
CONFIDENTIAL), and query_type (benign or adversarial
with attack subtype).
D Defense Implementation
D1 (Per-hop authorization).After BFS expansion via
apoc.path.spanningTree, each returned node is checked:
def is_node_authorized(node, user):
tier = SensitivityTier(node.sensitivity)
if tier > user.clearance:
return False
if node.tenant not in user.allowed_tenants:
return False
return True
Entity nodes carry tenant="" (empty string), so they always
fail the tenant check. This is the mechanism that severs the
chunk→entity→chunk pivot path.
D3 (Budgeted traversal).The Cypher query enforces a
global node budget via LIMIT $max_total . Per-hop branch-
ing is enforced post-query: expanded nodes are grouped by
hop depth, and each group is truncated to max_branching
entries.
BFS Cypher query(simplified):
UNWIND $seed_ids AS seed_id
MATCH (start {node_id: seed_id})
CALL apoc.path.spanningTree(start, {
maxLevel: $max_hops,
limit: $max_total
}) YIELD path
WITH last(nodes(path)) AS node,
length(path) AS depth
RETURN node.node_id, node.tenant,
node.sensitivity, min(depth)
ORDER BY hop_depth LIMIT $max_total
Thelength(path) return value feeds the Pivot Depth metric
directly.
E Entity-Aware Authorization
D1 eliminates all entity nodes because they carry empty-string
tenant labels. This is effective but coarse: it removes poten-
tially useful entity context (names, types, relations) from the
retrieval result. A finer-grained scheme would distinguish
traversal authorizationfrominclusion authorization:
1.Entity tenant-set labeling.During graph construction,
label each entity with the set of tenants whose documents
mention it: tenant_set = {acme, globex} . Shared
entities retain multiple tenant labels.
20

2.Traversal-through permission.Allow BFS to traverse
throughan entity node if the user’s tenant is in the en-
tity’s tenant_set . The entity itself may appear in con-
text.
3.Destination check.At each chunk node reached via an
entity, re-check the chunk’s tenant and sensitivity against
the user’s policy. This preserves D1’s security guarantee
at the chunk level while recovering entity utility.
This scheme would increase context size (recovering the en-
tity nodes D1 currently removes) without re-enabling the
chunk→entity→unauthorized chunkpivot, because the desti-
nation check still blocks cross-tenant chunks. The key trade-
off is implementation complexity: entity tenant_set must
be maintained as documents are added or removed, and the au-
thorization predicate becomes a set-membership check rather
than a simple equality.
F Statistical Methodology
Bootstrap procedure.All confidence intervals use the non-
parametric percentile bootstrap (10,000 resamples, seed 42).
For binary RPR indicators, the bootstrap distribution is con-
structed by resampling the n=500 per-query binary outcomes
{0,1} with replacement and computing the mean for each re-
sample. The 95% CI is the [2.5%,97.5%] percentile interval
of the bootstrap distribution.
Sample size justification.With n=500 queries, we
achieve power >0.80 for detecting a 5 percentage-point dif-
ference in RPR (from 0.0 to 0.05) at α=0.05 , computed via
the exact binomial test. For leakage means, the standard error
of the bootstrap is SE<0.5 items, sufficient to distinguish
mean differences of≥1 item.
Multiple comparisons.We test 7 pipeline variants against
2 query types (14 comparisons per metric). Applying a Bonfer-
roni correction ( αadj=0.05/14=0.0036 ), the core findings
(P1/P4–P8: RPR =0.0 ; P3: RPR >0.90 ) remain significant
atp<10−6.
εsensitivity.The regularized amplification factor AF(ε)
usesε=0.1 . We verify that conclusions are robust across
ε∈ {0.01,0.05,0.1,0.5} :AF(ε) ranges from 1,599 ( ε=0.01 )
to 32 (ε=0.5), all confirming>30×amplification.
21