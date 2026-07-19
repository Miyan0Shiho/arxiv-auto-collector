# Cost-Governed RAG: Unified Per-Tenant Cost Attribution Across Retrieval and Generation in Multi-Tenant LLM Systems

**Authors**: Navnit Shukla

**Published**: 2026-07-13 22:16:58

**PDF URL**: [https://arxiv.org/pdf/2607.12188v1](https://arxiv.org/pdf/2607.12188v1)

## Abstract
Enterprise Retrieval-Augmented Generation (RAG) deployments face a critical governance gap: while LLM generation cost is metered per token, the retrieval layer - vector memory, similarity compute, and embedding API calls - remains an unattributed shared cost, enabling invisible cross-subsidization among tenants. We present Cost-Governed RAG, an architecture that integrates a codebook-oblivious vector index (TurboVec) with a multi-tenant LLM governance gateway, creating a unified observability stack where embedding, retrieval, and generation costs are jointly attributable per tenant. The architecture exploits TurboVec's deterministic, closed-form memory formula to enable near-exact per-tenant retrieval cost calculation - a property unavailable in graph-based indexes with non-linear memory overhead. Deployed on Snowpark Container Services within a cloud data platform's governance boundary, the system achieves 99.96% end-to-end cost attribution accuracy across 100 simulated tenants (10M vectors, log-normal size distribution) with telemetry overhead below 0.04% of query latency. The architecture reduces retrieval infrastructure cost by 3.1-9.0x compared to managed vector database services under the pricing assumptions detailed in Section IV. We formalize a three-layer cost model and demonstrate that codebook-oblivious quantization enables deterministic per-tenant cost attribution while also removing the shared-codebook leakage surface present in trained quantizers - the latter observation being exploratory and subject to the limitations described in Section VII.

## Full Text


<!-- PDF content starts -->

Cost-Governed RAG: Unified Per-Tenant Cost
Attribution Across Retrieval and Generation in
Multi-Tenant LLM Systems
Navnit Shukla
AI Platform Engineering
Snowflake Inc.
Tustin, California, USA
https://orcid.org/0009-0005-8801-3344
Abstract—Enterprise Retrieval-Augmented Generation (RAG)
deployments face a critical governance gap: while LLM gen-
eration cost is metered per token, the retrieval layer—vector
memory, similarity compute, and embedding API calls—remains
an unattributed shared cost, enabling invisible cross-subsidization
among tenants. We present Cost-Governed RAG, an architecture
that integrates a codebook-oblivious vector index (TurboVec)
with a multi-tenant LLM governance gateway, creating a unified
observability stack where embedding, retrieval, and generation
costs are jointly attributable per tenant. The architecture exploits
TurboVec’s deterministic, closed-form memory formula to en-
able near-exact per-tenant retrieval cost calculation—a property
unavailable in graph-based indexes with non-linear memory
overhead. Deployed on Snowpark Container Services within a
cloud data platform’s governance boundary, the system achieves
99.96% end-to-end cost attribution accuracy across 100 simulated
tenants (10M vectors, log-normal size distribution) with teleme-
try overhead below 0.04% of query latency. The architecture
reduces retrieval infrastructure cost by 3.1–9.0×compared to
managed vector database services under the pricing assumptions
detailed in Section IV . We formalize a three-layer cost model
and demonstrate that codebook-oblivious quantization enables
deterministic per-tenant cost attribution while also removing the
shared-codebook leakage surface present in trained quantizers—
the latter observation being exploratory and subject to the
limitations described in Section VII.
Index Terms—retrieval-augmented generation, cost gover-
nance, multi-tenant AI systems, vector quantization, AI observ-
ability, cloud data platforms
I. INTRODUCTION
Enterprise LLM deployments increasingly use Retrieval-
Augmented Generation (RAG) [1] to ground model outputs
in proprietary knowledge bases. While recent work addresses
thegenerationcost of LLM inference through model cas-
cading [5], prompt compression [6], and token-level meter-
ing [17], theretrievalcost—memory consumed by vector
indexes, compute spent on similarity search, and embedding
API calls—remains a blind spot in enterprise cost governance.
This gap is consequential in multi-tenant deployments
where multiple business units or customers share infrastruc-
ture. A tenant with 10M documents indexed at full precision
(FP32) consumes∼57 GiB of vector memory, while a tenant
with 100K documents consumes∼0.57 GiB—yet both maybe charged the same flat infrastructure rate. Without retrieval-
layer cost attribution, cross-subsidization is invisible to FinOps
teams [21].
The challenge is compounded by the opacity of modern
vector indexes. Graph-based indexes (HNSW [14]) have non-
linear, graph-topology-dependent memory overhead that defies
simple per-tenant attribution. Trained codebook quantizers
(PQ [11]) share codebook state across tenants, creating both a
cost-attribution ambiguity and a potential privacy concern [3].
We address this gap by integrating two complementary
systems:
1)TurboVec[2], [3]: A codebook-oblivious vector index
whose memory consumption is exactlyn t×d×b/8per
tenant—deterministic, linear, and trivially attributable.
2)GovLLM: A multi-tenant LLM governance gateway
providing per-tenant token attribution, authentication,
model RBAC, and observability tracing [4].
Together, they form aCost-Governed RAGstack where
retrieval cost (memory, QPS), generation cost (tokens, credits),
and embedding cost (API calls) all close into a unified per-
tenant observability layer within the cloud data platform—
without external data movement.
Contributions.
1) A three-layer cost model formalizing embedding, re-
trieval, and generation costs per tenant, showing that
codebook-oblivious quantization enables near-exact re-
trieval attribution with bounded residual error (Sec-
tion III-B).
2) An architecture deploying the full Cost-Governed RAG
stack on Snowpark Container Services with integrated
telemetry (Section III).
3) Empirical validation on a 100-tenant simulation (10M
vectors) showing 99.96% end-to-end cost attribution
accuracy, 3.1–9.0×cost reduction versus managed alter-
natives, and sub-0.04% telemetry overhead (Section IV).
II. BACKGROUND ANDMOTIVATION
A. The RAG Cost Stack
A complete RAG query incurs cost at three distinct layers,
each with different attribution characteristics:
arXiv:2607.12188v1  [cs.AI]  13 Jul 2026

TABLE I
RAG COSTLAYERS ANDATTRIBUTIONCHARACTERISTICS
Layer Cost Driver Attribution Status
Embedding Tokens→API credits Per-call Solved
Retrieval Memory + compute Shared infraGap
Generation Tokens→LLM credits Per-call Solved
Existing LLM observability platforms (Langfuse [17],
LangSmith [18], Helicone [19]) trace generation cost per
tenant but treat retrieval as a fixed infrastructure cost amortized
uniformly. This creates a systematic governance failure: ten-
ants with large corpora subsidize tenants with small corpora,
and retrieval-heavy workloads (many queries, large top-k) are
invisible to cost dashboards.
B. Why Existing Vector Indexes Resist Cost Attribution
HNSW [14]:Graph-based indexes store navigable small-
world graphs whose edge structure is corpus-dependent and
shared across all vectors. A tenant’s vectors participate in
edges connecting to other tenants’ vectors, making per-tenant
memory isolation impossible without physical partitioning.
Trained PQ [11]:Product Quantization learnsm×K
centroids viak-means on the full corpus. This codebook is
shared across all tenants and encodes corpus-wide distribu-
tional structure. Per-tenant codebook attribution is undefined,
and the codebook itself may leak cross-tenant statistics [3].
TurboVec [2]:Codebook-oblivious quantization derives
boundaries analytically from the known marginal distribution
of L2-normalized vectors after random rotation. The codebook
contains zero corpus-dependent information and zero per-
tenant state. Total serving memory per tenant decomposes as:
Mt=nt×d×b
8| {z }
quantized codes+nt×4|{z}
norms+nt×8|{z}
IDs+n t×α|{z}
block metadata+d2×4
T|{z}
rotation (shared)
(1)
wheren t= tenant’s vector count,d= dimension,b= bit width,
α≈0.5bytes/vector for 32-vector block headers, andT=
number of tenants sharing the rotation matrix. All terms except
the last are deterministic and linear inn t; the rotation matrix
(d2×4 = 9.0MB atd= 1536) is shared and negligible at
scale. At 100K vectors and 4-bit:
•Codes:100K×1536×4/8 = 76.8MB
•Norms:100K×4 = 0.4MB
•IDs:100K×8 = 0.8MB
•Block metadata:100K×0.5 = 0.05MB
•Total per-tenant:≈78.0MB (payload-dominated)
This determinism—every component is a closed-form func-
tion ofn t,d, andb—is the key enabler of near-exact cost
attribution.
C. GovLLM: Multi-Tenant LLM Governance
GovLLM [4] provides per-tenant token attribution via a
FastAPI gateway with:•Per-tenant authentication (PAT/JWT/OAuth) with config-
urable rate limits
•Model RBAC restricting which tenants access which
LLMs and at what tier
•Langfuse-compatible tracing with telemetry closure into
the cloud data platform
•Per-query cost records written to platform-native tables
III. ARCHITECTURE
A. Deployment on Snowpark Container Services
The Cost-Governed RAG stack deploys within Snow-
park Container Services (SPCS), a container runtime within
Snowflake’s governance boundary. This deployment topology
ensures:
•Data never leaves the platform’s governance perimeter
•Compute is billed through the platform’s native credit
metering
•Telemetry writes directly to platform tables (no external
egress)
•Network isolation via VPC-level controls
Governance BoundaryTenant Request
GovLLM Gateway
Embedding Service
TurboVec (SPCS)
LLM (Cortex API)
Response + Cost Recordauth + rate limit
Cembed logged
Cretrieve logged
Cgenerate logged
Fig. 1. Cost-Governed RAG architecture. Each layer emits per-tenant cost
telemetry into a unified observability table within the platform governance
boundary.
B. Per-Tenant Cost Attribution Model
For tenanttmaking queryq, the total attributed cost is:
C(t, q) =C embed(q) +C retrieve(t, q) +C generate (q)(2)
Where each component is defined as:
Cembed(q) =tokens(q)×r embed (3)
Cretrieve(t, q) =Mt
Mtotal×r mem
| {z }
memory share+blocks scanned(t, q)×r cpu| {z }
compute
(4)
Cgenerate (q) = (T in+T out)×r model (5)

The retrieval cost decomposes into two attributable compo-
nents:
•Memory share:Tenantt’s fraction of total index mem-
ory, computed from (1). Because TurboVec’s memory is
deterministic and linear inn t, this fraction has bounded
error arising only from the amortized shared rotation
matrix (<0.1% at 10M vectors).
•Compute share:The number of 32-vector SIMD blocks
actually scanned for tenantt’s query. With kernel-level
allowlist filtering, only the tenant’s own blocks are scored,
making compute attribution precise.
Key insight:TurboVec’s codebook-oblivious design makes
retrieval cost attributionnear-exact(99.88%+ accuracy) be-
cause: (1) memory is deterministic per tenant with only a
small shared rotation matrix to amortize, and (2) kernel-level
filtering ensures compute is physically isolated per query.
This combination is, to our knowledge, unavailable in HNSW
or PQ-based indexes where shared graph edges or trained
codebook state resist clean per-tenant decomposition.
C. Telemetry Pipeline
Each TurboVec search emits a structured telemetry record:
{tenant_id, query_id, timestamp,
vectors_scanned, blocks_skipped,
latency_us, k_returned,
index_memory_bytes_tenant,
bit_width, compression_ratio}
This record is joined with the GovLLM generation trace
(tokens consumed, model used, cost) in the platform’s unified
telemetry table, producing a complete per-query cost record
across all three layers. Standard SQL queries against this table
power cost dashboards, anomaly detection, and chargeback
reports.
D. Cost Dashboard Integration
The unified telemetry table enables SQL-native cost analyt-
ics:
SELECT tenant_id,
SUM(embed_cost + retrieve_cost
+ generate_cost) as total_cost,
SUM(retrieve_cost)/SUM(total_cost)
as retrieval_pct
FROM cost_telemetry
WHERE ts >= DATEADD(’day’, -30, CURRENT_DATE)
GROUP BY tenant_id ORDER BY total_cost DESC;
This query is not straightforward with external vector
databases (Pinecone, Qdrant) because they do not, to our
knowledge, expose per-tenant memory utilization APIs that
feed directly into the governance layer’s tables.
IV. EVALUATION
A. Experimental Setup
We simulate a 100-tenant deployment with the following
configuration:•Total corpus:10M vectors,d= 1536(OpenAI text-
embedding-3-large withdimensions=1536)
•Tenant distribution:Log-normal (µ= 11.5,σ= 1.0),
ranging from∼10K to∼500K vectors per tenant
•Query load:1,000 QPS aggregate, distributed propor-
tionally to tenant size
•Quantization:TurboVec 4-bit (8×compression)
•Compute:SPCSCPU_X64_Sinstance family (1 node)
for benchmarks at 100K scale
B. Cost Attribution Accuracy
We compare attributed cost (from telemetry records) against
actual resource consumption measured at the infrastructure
level. Attribution accuracy for each tenantiis defined as
Ai= 1− | ˆCi−Ci|/Ci, where ˆCiis the telemetry-derived
attributed cost andC iis the ground-truth cost computed
from direct memory measurement and CPU cycle counters.
The reported end-to-end figure is computed as follows: for
each of the 10,000 simulated queries, we compute per-query
attribution error; we then average across queries within each
tenant to obtainA i; finally, we report the mean ofA iacross all
100 tenants. Because the simulation uses a fixed random seed
(42) and all index operations are deterministic (flat-scan with
no stochastic components), repeated runs produce identical
results—variance arises only from the tenant size distribution,
not from algorithmic randomness.
TABLE II
COSTATTRIBUTIONACCURACYACROSSLAYERS(100 TENANTS, 10M
VECTORS, LOG-NORMALDISTRIBUTION). ACCURACY=MEAN OF
1− |ˆCi−C i|/C iACROSS TENANTS;MAX ERROR=WORST-CASE
SINGLE-TENANT DEVIATION.
Layer Accuracy Max Error Error Source
Embedding 100.00% 0.0% —
Retrieval 99.88% 1.2% Shared overhead
Generation 99.96% 0.4% Rounding
End-to-end 99.96% 0.8%—
The 0.12% retrieval attribution error arises solely from the
shared rotation matrix term in (1): atd= 1536, the matrix
occupiesd2×4 = 9.0MB, which is amortized uniformly
across tenants rather than attributed proportionally to corpus
size. All other terms (codes, norms, IDs, block metadata) are
exactly linear inn tand thus perfectly attributable. At 10M
total vectors, the shared 9.0 MB represents<0.12% of the
total 7.3 GiB index, producing the observed attribution gap.
C. Memory Cost Reduction
The “Attribution” column highlights a key differentiator:
managed vector databases do not, to our knowledge, provide
APIs for per-tenant memory consumption reporting, making
cost governance difficult at the retrieval layer. Self-hosted
FP32 indexes can theoretically be attributed (memory is linear
in vector count) but lack the compression benefits.
Pricing assumptions.TurboVec SPCS cost assumes a sin-
gleCPU_X64_Snode at Snowflake’s published SPCS credit

TABLE III
MONTHLYRETRIEVALINFRASTRUCTURECOSTCOMPARISON(10M
VECTORS,d= 1536, 1000 QPS). MEMORY DERIVED FROMEQ. 1.
Configuration Memory†$/month Attribution‡
Uncompressed FP32 57.2 GiB $580 Possible
TurboVec 4-bit (SPCS) 7.3 GiB $248Exact
TurboVec 2-bit (SPCS) 3.7 GiB $124Exact
Pinecone (managed) — $770 Opaque
Qdrant Cloud (managed) — $1,116 Opaque
Reduction—3.1–9.0×—
†From Eq. 1: codes (10M×1536×b/8) + norms (10M×4) + IDs (10M×8)
+ block metadata (10M×0.5) + rotation (9 MB shared). 4-bit:
7.68 + 0.04 + 0.08 + 0.005 + 0.009 = 7.81GB= 7.27GiB≈7.3GiB.
Cost estimates from public pricing, June 2026; do not equalize recall/latency/SLAs.
‡Exact: closed-form per-tenant from Eq. 1;Possible: linear but no API;
Opaque: no per-tenant memory API exposed to consumers.
rate (us-west-2, June 2026). Pinecone estimate assumes the
Serverless plan at 10M stored vectors with 1,000 reads/sec
(no writes); Qdrant estimate assumes a dedicated cluster
sized for 10M vectors atd= 1536in AWS us-east-1. All
estimates exclude replication, backups, and availability SLAs.
These figures are illustrative scenarios, not rigorous iso-quality
benchmarks.
D. SPCS Deployment Performance
TABLE IV
TURBOVECSPCS SERVICEPERFORMANCE(100K VECTORS,d= 1536,
4-BIT,CPU_X64_SINSTANCEFAMILY)
Metric Value
Ingest throughput 19,966 vectors/sec
Search latency (unfiltered) 13.0 ms (77 QPS)
Search latency (10-tenant filtered) 8.0 ms (125 QPS)
Recall@5 vs exact FP32 0.962
Telemetry overhead 0.005 ms (0.04%)
Index memory (codes only) 76.8 MB
Total serving memory (Eq. 1) 78.1 MB
Compression ratio vs FP32 7.8×
Filtered search isfasterthan unfiltered because the ker-
nel short-circuits SIMD blocks with no allowed tenant vec-
tors. Telemetry record creation adds 0.005 ms per query—
negligible relative to search latency—enabling per-query cost
attribution without measurable performance degradation.
E. Comparison Against Alternative Index Architectures
To contextualize TurboVec’s governance advantage, we
benchmark alternative FAISS index types at 999K vectors
(d= 1536) from the same DBpedia dataset:
HNSW achieves the highest recall (0.984) but its graph-
based memory is non-decomposable per tenant—edges cross
tenant boundaries, making exact cost attribution structurally
impossible without physical partitioning. We note that the
HNSW comparison is at 500K rather than 999K due to
hardware memory constraints (>6.4 GiB required for HNSW-
Flat at 999K on our 16 GiB test machine); at 999K, HNSW
memory would be approximately 6.4 GiB, further wideningTABLE V
INDEXARCHITECTURECOMPARISON AT999K VECTORS: RECALL,
LATENCY, MEMORY,ANDCOSTATTRIBUTIONPROPERTIES
Method R@5 ms/q Mem Train Attrib.
TurboQuant 4-bit 0.968 17.1 767 MB None Exact
TurboQuant 2-bit 0.901 9.1 384 MB None Exact
FAISS PQ 4-bit 0.883 154 384 MB 61s Ambiguous
FAISS PQ 8-bit 0.961 222 767 MB 132s Ambiguous
FAISS IVF-PQ 0.840 5.7 390 MB 35s Ambiguous
HNSW-Flat∗0.984 0.5 3200 MB 758s Impossible
∗HNSW measured at 500K (infeasible at 999K on 16 GB test machine).
the memory gap versus TurboVec’s 767 MB. IVF-PQ and
flat PQ share trained codebook state across tenants, creating
attribution ambiguity for the codebook memory component.
Only TurboVec’s codebook-oblivious, flat-scan architecture
supports exact per-tenant memory decomposition via (1).
F . Tenant Scaling Analysis
To understand cost attribution behavior as tenant count
varies, we measure attribution accuracy acrossT∈
{10,50,100,500,1000}tenants (10M vectors, fixed total):
TABLE VI
COSTATTRIBUTIONACCURACY VS. TENANTCOUNT(10M VECTORS
TOTAL, TURBOVEC4-BIT)
Tenants Attribution Max Error Avg Vecs/Tenant
10 99.99% 0.1% 1,000,000
50 99.95% 0.5% 200,000
100 99.88% 1.2% 100,000
500 99.72% 2.8% 20,000
1000 99.53% 4.7% 10,000
Attribution accuracy degrades gracefully as tenant count
increases because the fixed overhead (O fixed) is amortized
over more tenants with smaller per-tenant allocations. Even at
1000 tenants, attribution remains above 99.5%—well within
acceptable FinOps tolerance [21].
G. Cost Breakdown Analysis
We analyze the relative contribution of each cost layer
across tenant sizes:
TABLE VII
COSTLAYERBREAKDOWN BYTENANTSIZE(1000 QPS TOTAL, CLAUDE
3.5 SONNETGENERATION)
Tenant Size Embed % Retrieve % Generate %
Small (10K vecs) 8% 5% 87%
Medium (100K vecs) 8% 12% 80%
Large (500K vecs) 8% 28% 64%
For large tenants, retrieval constitutes up to 28% of total
RAG cost—far from negligible. Without retrieval-layer attri-
bution, these tenants would be cross-subsidized by smaller
tenants under flat-rate pricing.

V. DISCUSSION
Why cloud-native deployment matters for governance.
External vector databases (Pinecone, Qdrant, Weaviate) cannot
export per-tenant memory utilization back to the enterprise
governance layer. By deploying TurboVec within SPCS, re-
trieval telemetry writes to the same Snowflake tables as gener-
ation telemetry—enabling unified SQL-based cost dashboards,
anomaly alerts, and automated chargeback without cross-
platform data movement or API polling.
Codebook-oblivious quantization as a governance en-
abler.The connection between codebook design and cost
governance is under-explored. TurboVec’s codebook contains
no corpus-dependent state, which simultaneously: (a) removes
the shared-codebook leakage surface present in trained PQ
codebooks [3], and (b) makes per-tenant memory determinis-
tically attributable (no shared learned state to amortize). This
dual benefit suggests that leakage-resistant system design can
be synergistic with—rather than in tension with—governance
requirements. We note that this observation is limited to the
codebook surface; it does not constitute end-to-end privacy
guarantees, which would require a formal threat model and
empirical evaluation beyond the scope of this paper.
Sustainability implications.At enterprise scale (100M+
vectors), the 8×memory reduction from 4-bit quantization
translates directly to fewer compute nodes. Concretely: 100M
vectors at FP32 require∼572 GiB of index memory, ne-
cessitating∼5 r6i.4xlarge instances (128 GiB RAM each, at
90% utilization). At 4-bit, the same corpus fits in∼112 GiB,
requiring∼1 instance—a reduction of 4 nodes. At∼0.25 kW
per instance and US average grid carbon intensity of 0.39 kg
CO 2/kWh [23], the annual savings are approximately4×
0.25×8760×0.39≈3.4tonnes CO 2per 100M-vector
deployment—aligning enterprise RAG infrastructure with sus-
tainability goals [22].
Integration with FinOps frameworks.The three-layer cost
model maps directly to the FinOps Foundation’s FOCUS spec-
ification [21]: each telemetry record contains a resource iden-
tifier (tenant id), usage quantity (vectors scanned, tokens),
and unit rate—enabling standard FinOps tooling (CloudHealth,
Apptio) to process RAG costs alongside traditional cloud
spend.
VI. RELATEDWORK
LLM cost optimization.UCCI [5] routes queries to cheaper
models via cascade policies. ProCut [6] compresses prompts
to reduce input token costs. FrugalGPT [7] combines model
selection, caching, and query adaptation. These operate exclu-
sively at the generation layer; to our knowledge, none address
retrieval cost attribution.
RAG system optimization.RAG-Stack [8] co-optimizes
retrieval quality and throughput but does not formalize cost
per tenant. RAGO [9] provides systematic RAG serving per-
formance optimization. HyperRAG [10] addresses distributed
retrieval scaling. To our knowledge, none provide per-tenant
cost observability.Vector quantization for ANN.Product Quantization [11]
and OPQ [12] achieve high compression via corpus-dependent
training. TurboQuant [2] derives boundaries analytically with-
out training. RaBitQ [13] provides theoretical error bounds.
Our contribution is connecting quantization design choices to
cost governance properties.
Multi-tenant AI systems.OptiLeak [15] demonstrates
prompt reconstruction risks in shared LLM KV-caches. Multi-
tenant database isolation is well-studied [16], but multi-tenant
vector index isolation for both leakage reduction and cost
attribution is, to our knowledge, novel.
MLOps and AI observability.Langfuse [17] and Lang-
Smith [18] provide LLM tracing. MLflow [20] tracks ML
experiments. Helicone [19] provides LLM cost analytics. To
our knowledge, none integrate retrieval-layer cost attribution
into the observability pipeline.
FinOps for AI.The FinOps Foundation’s FOCUS specifi-
cation [21] standardizes cloud cost data but has no provisions
for AI-specific cost components (embeddings, retrieval, gen-
eration). Our three-layer model extends FinOps principles to
the RAG stack.
VII. LIMITATIONS ANDFUTUREWORK
1)Simulated workload only.The 100-tenant evaluation
uses synthetic log-normal tenant distributions and pro-
portional query loads. Real enterprise workloads exhibit
bursty, time-varying patterns that may stress the attri-
bution model differently. Validation against anonymized
production traces from a multi-tenant RAG deployment
is planned as immediate future work.
2)CPU flat-scan only.TurboVec currently uses flat-scan
search. Graph-based indexes (HNSW) have non-linear
memory profiles that would require a different attri-
bution formula. Extending the cost model to graph-
augmented TurboVec is future work.
3)Single pricing snapshot.Cost comparisons use 2026
public pricing that may change. The architecture is
pricing-model-agnostic; only the rate constants (r mem,
rcpu,rmodel) require updating.
4)No dynamic rebalancing.The current model assumes
static tenant-to-vector assignments. Tenant growth,
churn, and data migration would require periodic rec-
onciliation of attribution baselines.
5)Single embedding model.All cost projections assume
d= 1536(OpenAI text-embedding-3-large with the
dimensions=1536parameter; the model’s native
output isd= 3072). Lower-dimensional embeddings
would reduce per-tenant memory proportionally, po-
tentially shifting the cost balance toward generation-
dominated workloads.
Future Work.Natural extensions include: (a) integration
with graph-based search layers and corresponding non-linear
attribution models; (b) real production deployment with ac-
tual chargeback validation; (c) cost-aware query routing that
dynamically selects bit-width based on tenant SLA tier; and

(d) extending the model to multimodal RAG (image + text
embeddings with heterogeneous dimensionality).
VIII. CONCLUSION
Cost-Governed RAG demonstrates that codebook-oblivious
vector quantization simultaneously removes the shared-
codebook leakage surface and enables deterministic per-tenant
cost attribution—bridging two communities (retrieval systems
and AI governance) that have operated largely independently
despite serving the same enterprise deployments. Deployed
on a cloud data platform’s container service, the architecture
achieves 99.96% end-to-end cost attribution accuracy while
reducing retrieval infrastructure cost by 3.1–9.0×compared to
managed alternatives under the pricing assumptions described.
For large tenants, retrieval constitutes up to 28% of total
RAG cost—a governance blind spot that our unified observ-
ability stack makes visible. As enterprise AI moves toward
production-grade multi-tenant deployment, cost governance
must evolve from token-level metering to full-stack attribution
spanning every layer of the RAG pipeline.
ACKNOWLEDGMENT
Navnit Shukla conceived the Cost-Governed RAG archi-
tecture, designed the three-layer cost model, conducted all
deployment experiments and cost analysis, and authored the
complete manuscript.
REFERENCES
[1] P. Lewis et al., “Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks,” inProc. NeurIPS, 2020.
[2] J. Zhang et al., “TurboQuant: Online Vector Quantization with Near-
optimal Distortion Rate,” inProc. ICLR, 2026.
[3] N. Shukla, “TurboVec: A Case Study in Cost-Efficient Private Retrieval
for Enterprise RAG via Codebook-Oblivious Quantization,” inProc.
FLLM, 2026.
[4] N. Shukla, “Building Multi-Tenant AI Chat Applications with Cortex
REST API,” Snowflake Developer Guide, 2026.
[5] V . Kotte, “UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade
Routing,” arXiv:2605.18796, 2026.
[6] Y . Xu et al., “ProCut: LLM Prompt Compression via Attribution
Estimation,” arXiv:2508.02053, 2025.
[7] L. Chen et al., “FrugalGPT: How to Use Large Language Models While
Reducing Cost and Improving Performance,” arXiv:2305.05176, 2023.
[8] Y . Chen et al., “RAG-Stack: Co-Optimizing RAG Quality and Perfor-
mance From the Vector Database Perspective,” arXiv:2510.20296, 2025.
[9] Z. Li et al., “RAGO: Systematic Performance Optimization for Retrieval-
Augmented Generation Serving,” inProc. ISCA, 2025.
[10] X. Liu et al., “Towards Hyper-Efficient RAG Systems in VecDBs:
Distributed Parallel Multi-Resolution Vector Search,” arXiv:2511.16681,
2025.
[11] H. J ´egou, M. Douze, and C. Schmid, “Product Quantization for Nearest
Neighbor Search,”IEEE Trans. PAMI, vol. 33, no. 1, pp. 117–128, 2011.
[12] T. Ge, K. He, Q. Ke, and J. Sun, “Optimized Product Quantization for
Approximate Nearest Neighbor Search,” inProc. IEEE CVPR, 2013, pp.
2946–2953.
[13] J. Gao et al., “RaBitQ: Quantizing High-Dimensional Vectors with a
Theoretical Error Bound for Approximate Nearest Neighbor Search,” in
Proc. SIGMOD, 2024.
[14] Y . Malkov and D. Yashunin, “Efficient and Robust Approximate Nearest
Neighbor Search Using Hierarchical Navigable Small World Graphs,”
IEEE Trans. PAMI, vol. 42, no. 4, pp. 824–836, 2020.
[15] Z. Wang et al., “OptiLeak: Efficient Prompt Reconstruction via Rein-
forcement Learning in Multi-tenant LLM Services,” arXiv:2602.20595,
2026.[16] S. Aulbach et al., “Multi-Tenant Databases for Software as a Service:
Schema-Mapping Techniques,” inProc. ACM SIGMOD, 2008, pp. 1195–
1206.
[17] M. Kilian et al., “Langfuse: Open-Source LLM Engineering Platform,”
2024. [Online]. Available: https://langfuse.com
[18] LangChain, “LangSmith: LLM Application Observability,” 2024. [On-
line]. Available: https://smith.langchain.com
[19] Helicone, “LLM Observability for Developers,” 2025. [Online]. Avail-
able: https://helicone.ai
[20] M. Zaharia et al., “Accelerating the Machine Learning Lifecycle with
MLflow,”IEEE Data Eng. Bull., vol. 41, no. 4, 2018.
[21] FinOps Foundation, “FOCUS: FinOps Open Cost and Usage Specifica-
tion,” 2024. [Online]. Available: https://focus.finops.org
[22] R. Schwartz et al., “Green AI,”Communications of the ACM, vol. 63,
no. 12, pp. 54–63, 2020.
[23] US EPA, “eGRID Summary Tables 2022,” 2024. [Online]. Available:
https://www.epa.gov/egrid