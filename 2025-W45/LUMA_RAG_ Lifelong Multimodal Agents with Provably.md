# LUMA-RAG: Lifelong Multimodal Agents with Provably Stable Streaming Alignment

**Authors**: Rohan Wandre, Yash Gajewar, Namrata Patel, Vivek Dhalkari

**Published**: 2025-11-04 08:47:12

**PDF URL**: [http://arxiv.org/pdf/2511.02371v1](http://arxiv.org/pdf/2511.02371v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as the dominant paradigm for
grounding large language model outputs in verifiable evidence. However, as
modern AI agents transition from static knowledge bases to continuous
multimodal streams encompassing text, images, video, and audio, two critical
challenges arise: maintaining index freshness without prohibitive re-indexing
costs, and preserving cross-modal semantic consistency across heterogeneous
embedding spaces. We present LUMA-RAG, a lifelong multimodal agent architecture
featuring three key innovations: (i) a streaming, multi-tier memory system that
dynamically spills embeddings from a hot HNSW tier to a compressed IVFPQ tier
under strict memory budgets; (ii) a streaming CLAP->CLIP alignment bridge that
maintains cross-modal consistency through incremental orthogonal Procrustes
updates; and (iii) stability-aware retrieval telemetry providing Safe@k
guarantees by jointly bounding alignment drift and quantization error.
Experiments demonstrate robust text-to-image retrieval (Recall@10 = 0.94),
graceful performance degradation under product quantization offloading, and
provably stable audio-to-image rankings (Safe@1 = 1.0), establishing LUMA-RAG
as a practical framework for production multimodal RAG systems.

## Full Text


<!-- PDF content starts -->

LUMA-RAG: Lifelong Multimodal Agents with
Provably Stable Streaming Alignment
Rohan Wandre
Dept. of Computer Engineering
SIES Graduate School of Technology
Navi Mumbai, India
rohanwandre24@gmail.comVivek Dhalkari
Dept. of Computer Engineering
SIES Graduate School of Technology
Navi Mumbai, India
vivekdhalkari@gmail.comYash Gajewar
Dept. of Computer Engineering
Bharatiya Vidya Bhavan’s Sardar Patel
Institute of Technology
Mumbai, India
yashgajewar06@gmail.com
Dr. Namrata Patel
Dept. of Computer Engineering
SIES Graduate School of Technology
Navi Mumbai, India
namratap@sies.edu.in
Abstract—Retrieval-Augmented Generation (RAG) has
emerged as the dominant paradigm for grounding large
language model outputs in verifiable evidence. However, as
modern AI agents transition from static knowledge bases
to continuous multimodal streams encompassing text, images,
video, and audio, two critical challenges arise: maintaining index
freshness without prohibitive re-indexing costs, and preserving
cross-modal semantic consistency across heterogeneous
embedding spaces. We present LUMA-RAG, a lifelong
multimodal agent architecture featuring three key innovations:
(i) a streaming, multi-tier memory system that dynamically
spills embeddings from a hot HNSW tier to a compressed
IVFPQ tier under strict memory budgets; (ii) a streaming
CLAP→CLIP alignment bridge that maintains cross-modal
consistency through incremental orthogonal Procrustes updates;
and (iii) stability-aware retrieval telemetry providing Safe@k
guarantees by jointly bounding alignment drift and quantization
error. Experiments demonstrate robust text-to-image retrieval
(Recall@10 = 0.94), graceful performance degradation under
product quantization offloading, and provably stable audio-to-
image rankings (Safe@1 = 1.0), establishing LUMA-RAG as a
practical framework for production multimodal RAG systems.
Index Terms—Multimodal RAG, Streaming Systems, Life-
long Learning, Vector Databases, Cross-Modal Retrieval, CLIP,
CLAP, Semantic Alignment
I. INTRODUCTION
The evolution of Large Language Models (LLMs) from
static repositories to continuously learning agents requires
systems that ingest heterogeneous multimodal streams while
sustaining latency, accuracy, and semantic coherence. RAG [1]
grounds outputs in external evidence, yet conventional designs
face two bottlenecks: (i)freshness vs. latency—periodic re-
indexing creates windows where new content is not retrievable;
and (ii)cross-modal drift—independently trained embedding
spaces (e.g., CLIP [2] and CLAP [3]) are not directly com-
parable.
We introduceLUMA-RAG, a unified, lifelong architecture
that integrates low-latency ingestion, tiered memory, a stream-
ing CLAP→CLIP bridge, and stability-aware retrieval.Our main contributions are:
•Amulti-tier memorywith dynamic promotion/demotion
between HNSW (hot) and IVFPQ (warm) under strict bud-
gets, preserving latency and cost.
•Astreaming subspace alignmentmodule learning a
CLAP→CLIP bridge via incremental orthogonal Procrustes
on co-occurrence pairs.
•Astability guarantee(Safe@k) that upper-bounds top-k
ranking perturbations via alignment driftεand quantization
errorζ.
II. RELATEDWORK
A. Multimodal Retrieval-Augmented Generation
Text-only RAG [1] is effective, but modern applications
require multimodal grounding [5]–[7]. Prior systems assume
static knowledge and rarely address alignment drift or contin-
uous updates. We integrate audio via CLAP [3] and maintain
a streaming bridge to CLIP [2].
B. Vector Databases and Tiered Memory
HNSW and IVFPQ underpin scalable ANN search [4], [10].
Hot–warm architectures [8], [9] balance speed and capacity.
LUMA-RAG unifies these with a simple policy that respects
budgets and preserves diversity.
C. Cross-Modal Alignment
CLIP aligns image–text; CLAP aligns audio–text. Prior
work explores static linear/nonlinear mappings [11]–[15]. We
targetstreamingalignment with provable stability in produc-
tion settings.
III. PROBLEMSETUP ANDNOTATION
Letf clip-txt ,fclip-img ,fclap-aud , andf clip-vid (frame-pooled) map
text, images, audio, and video frames toRd(L2-normalized).
CLIP is the canonical space; we learn an orthogonalT∈
Rd×dsuch thatz audTaligns CLAP audio embeddings witharXiv:2511.02371v1  [cs.LG]  4 Nov 2025

Fig. 1. LUMA-RAG system architecture: hot HNSW tier for low-latency recall, warm IVFPQ tier for capacity, streaming CLAP→CLIP alignment bridge,
and stability-aware retrieval.
TABLE I
SYMBOLS AND NOTATION
Symbol Description
dEmbedding dimension (512)
BHot-tier capacity budget
TOrthogonal bridge (CLAP→CLIP)
εBridge drift∥T new−T prev∥2
ζIVFPQ distortionE∥v−ˆv∥ 2
γTop-1 margins(q, d 1)−s(q, d 2)
δk Gap betweenT kand best outside item
Fig. 2. Streaming process flow: Data Sources→Multimodal Encoders→
Alignment→Vector Index→Retrieval & Ranking→Answerer/UI.
CLIP. We trackalignment driftε=∥T new−T prev∥2and
quantization distortionζ=E∥v−ˆv∥ 2from IVFPQ.
IV. SYSTEMDESIGN
A. End-to-End Process Flow
Figure 2 summarizes the online path: data sources are
encoded into modality-specific embeddings, aligned into the
CLIP space via the streaming bridge, indexed across hot and
warm tiers, and retrieved/reranked before answer generation.
B. Ingestion and Preprocessing
We continuously ingest four modalities:
•Text and metadata (UTF-8 normalized; sentence-split;
stopword-light).
•Images with BLIP captions; EXIF stripped; max side 1024
px.
•Audio with Whisper transcripts and TTS augmentation for
paired training; 16 kHz mono.Algorithm 1Hot→Warm Spill Policy (background task)
Require:BudgetB, hot indexH, warm indexW, weights
(α, β, γ)
1:functionSCORE(d)
2:returnα·recency(d) +β·freq(d) +γ·novelty(d)
3:while|H|> Bdo
4:d⋆←arg min d∈H SCORE(d)
5:Removed⋆fromH; push to bufferS
6:ifIVFPQ not trained or driftedthen
7:Train/retrain IVFPQ onS ∪sample(H)
8:AddStoW; clearS
•Video via uniform frame sampling (1–3 fps) with CLIP
pooling (mean+max).
Each item receives a policy score combiningrecency,usage,
novelty, andcoverage, used during hot→warm spill.
C. Multi-Tier Memory and Policy
New embeddings enter HNSW (hot). When|H|> B, a
background job trains IVFPQ and spills low-scoring items
(Alg. 1). Queries probe hot and warm, merge results by score,
then re-rank top-Kvia exact cosine.
D. StreamingCLAP→CLIPBridge
We buffer co-occurring pairs 
xCLAP, xCLIP
and updateT
everyNpairs via orthogonal Procrustes:
min
T∥X CLAPT−X CLIP∥Fs.t.T⊤T=I, T=UV⊤.(1)
HereX CLAP uses CLAP-text embeddings from captions; the
learnedTis applied to CLAP-audio queries at runtime.
E. Online Telemetry and Safety Gating
For each query we log: top-kscores; marginsγ, δ k; current
ε, ζ; and Safe flags per Sec. V. Items withγ≤2(ε+ζ)return
a low-confidence token to the LLM to trigger clarification or
more recall.

TABLE II
INDEX CONFIGURATION SUMMARY(FAISS)
Parameter Value
HNSW M / efConstruction 32 / 200
HNSW efSearch (hot) 64
IVF lists (n list) 100
PQ codebooks (m) 8
PQ bits per subvector 8
nprobe (warm) 10
Embedding dim (d) 512
V. THEORETICALPROPERTIES
Lemma 1(Cosine Lipschitzness).For unit vectorsu, vand
perturbations∆u,∆v, we have|⟨u+∆u, v+∆v⟩−⟨u, v⟩| ≤
∥∆u∥ 2+∥∆v∥ 2.
Proof.Triangle inequality:|⟨∆u, v⟩| ≤ ∥∆u∥ 2and similarly
for∆v.
Theorem 1(Top-1Stability).Ifγ >2(ε+ζ), the identity of
d1is invariant under perturbations bounded byεandζ.
Proof.By the lemma, an item’s score changes by at mostε+ζ.
Two items can swap only if their gap is below2(ε+ζ).
Theorem 2(Top-kSet Stability).Letδ kbe the minimum gap
between any member ofT kand the best non-member. Ifδ k>
2(ε+ζ), thenT kis invariant.
Proof.Apply the lemma to the boundary pair that definesδ k;
boundary crossings are ruled out.
Lemma 2(Bridge Drift Bound).LetM t=X⊤
CLAP,t XCLIP,t
andT t=UtV⊤
t. IfM t+1=M t+Ewith∥E∥ 2≤ηandM t
is well-conditioned, then∥T t+1−Tt∥2≤2∥E∥ 2/σmin(Mt).
VI. COMPLEXITY ANDRESOURCEANALYSIS
HNSW insert/search areO(logn)average; IVFPQ training
isO(n traind)amortized and query isO(n probed/m). Procrustes
refresh isO(d3)per SVD but amortized overN∈[256,1024]
pairs withd= 512. Memory is dominated by encoders; the
index/bridge add only MBs.
VII. IMPLEMENTATIONDETAILS
Encoders: CLIP ViT-B/32 for vision/text; CLAP for audio;
video via 1–3 fps frame sampling with CLIP pooling. All
outputs are L2-normalized. FAISS backends: HNSW (Index-
IDMap2) for hot; IVFPQ (IndexIDMap2) for warm. Hot
budgetB= 500. IVFPQ:n list= 100,m= 8,n bits= 8,
nprobe = 10. Bridge refreshN= 512; we storeTin
SQLite/JSON for crash-safe reload.
VIII. EVALUATIONPROTOCOL
Datasets.(i) Baseline set: 31 images with BLIP captions.
(ii) Augmented set: 620 images with near-duplicates for group-
aware evaluation. (iii) Audio set: TTS audio of baseline
captions (CLAP mismatch stress test).
Metrics.Recall@k, MRR, nDCG@k, Safe@k,εandζ,
p50/p95 latency, storage footprint.TABLE III
BASELINETEXT-TO-IMAGERETRIEVAL(31IMAGES)
Metric Value
nDCG@10 0.6712
Recall@100.9355
MRR 0.5859
Safe@11.0000
Query p95 (ms)<3
Fig. 3. Baseline text→image retrieval.
TABLE IV
OFFLOADINGRESULTS(620IMAGES,B= 500)
Metric Value
Group nDCG@10 0.4963
Group Recall@100.5339
Group MRR 0.4750
ζ(L2 distortion) 0.3880
Query p95 (ms)<4
Baselines.(a) Hot-only (no warm). (b) Hot+Warm without
bridge (audio uses CLAP-only similarities). (c) LUMA-RAG
full: hot+warm + streaming bridge + safety telemetry.
IX. EXPERIMENTS
A. Setup
Hardware: 8-core laptop CPU; FP32 inference. Candidate
depth for re-rank: 200. We log telemetry per query and
aggregate Safe@k.
B. Baseline Text-to-Image
On 31 images with BLIP captions, we see strong retrieval
(Table III; Fig. 3).
C. Memory Offloading with Product Quantization
With 620 images andB= 500, the system spills 120 items
to IVFPQ (Table IV; Fig. 4). Distortionζ= 0.3880yields
moderate but bounded accuracy drop; latency remains<4ms.

Fig. 4. Retrieval under memory offloading (group-aware).
TABLE V
AUDIO→IMAGERETRIEVAL VIACLAP→CLIP BRIDGE
Metric Value
nDCG@10 0.1780
Recall@10 0.4194
MRR 0.1082
εalign 0.0000
Safe@11.0000
Query p95 (ms)<3
Fig. 5. Audio→image retrieval with a streaming bridge.
D. Audio-to-Image via Streaming Alignment
We query with TTS audio of the captions; results are
in Table V and Fig. 5. Safe@1=1.0 asε align = 0after
convergence.
E. Alignment Drift and Safety Telemetry Over Time
We logεand Safe@1 across refreshes (median over 500
queries).
X. DESIGNCHOICES ANDTRADE-OFFS
Canonical space.CLIP provides strong zero-shot ground-
ing for images and text; aligning audio into this space reduces
cross-modal glue.
Bridge cadence.N∈[256,1024]balances responsiveness andTABLE VI
TELEMETRY VS.STEPS(ILLUSTRATIVE)
Stepε(p50)ζSafe@1
0 (init) 0.0000 0.3880 1.000
+2K pairs 0.0164 0.3880 0.982
+5K pairs 0.0217 0.3880 0.969
+10K pairs 0.0281 0.3880 0.952
Algorithm 2QueryWithSafeK(q,k)
1:ch ←HNSW_search(q, K h);c w ←
IVFPQ_search(q, K w)
2:C←merge(c h, cw);ˆR←rerank_cosine(q, C)
3:γ1←s(q, d 1)−s(q, d 2);δk←min d∈Tk,o/∈T ks(q, d)−
s(q, o)
4:safe1←[γ 1>2(ε+ζ)],safeK←[δ k>2(ε+ζ)]
5:return ˆR[1:k], safe1, safeK
SVD cost (we use 512).
Warm training mix.Mix spilled items with a hot sample to
keep IVF centroids fresh.
Rerank depth.Improves Recall@k but increases p95; cap at
200 for sub-5 ms E2E.
Safety gating.Whenγ≤2(ε+ζ), defer to LLM clarification
or expandK.
XI. THREATMODEL ANDSAFETY
Safe@k certifies robustness tobenignperturbations from
alignment drift and PQ distortion. It doesnotclaim immunity
to: (i) adversarial embedding attacks; (ii) semantic poisoning
of captions; or (iii) catastrophic encoder bugs. Mitigations:
rate-limited ingestion, outlier filters, signed model artifacts,
and quarantine of low-confidence responses.
XII. DEPLOYMENT ANDOPERATIONS
Cold start.Keep all items hot until≥4,096vectors, then
train warm.
Rolling encoder updates.Run old+new encoders; decay old
vectors via spill policy.
Sharding.Hash by tenant/group; replicateT(tiny state).
Monitoring.Trackε,ζ, Safe@k, top-kgaps; alert when
med.γ≈2(ε+ζ).
Governance.Log evidence and Safe flags for audits.
XIII. REPRODUCIBILITYNOTES
Scripts for BLIP captioning, TTS generation, and evaluation
are provided. Example commands: BLIP captions:python
-m scripts.gen_captions -dir data/inbox
-out data/captions.txt; group-aware eval:
python -m scripts.eval_folder_group
-dir data/inbox_aug -captions
data/captions_aug.txt -hot_budget 500; audio
eval:python -m scripts.eval_audio_query
-img_dir data/inbox -captions
data/captions.txt -audio_dir data/audio
-k 10.

TABLE VII
FAILURE INJECTION TESTS AND EXPECTED SIGNALS
Injection Observed signal Gating/Action
Caption swapε↑, Safe@1↓increase cadence; quar-
antine source
PQ over-compressζ↑, Recall@10↓raisem, lists; re-code
warm
Hot rebuild latency spike (p95) hot-only path until re-
build done
Audio domain shift stableε, accuracy↓collect pairs; bridge ca-
dence++
Outlier embeddings score margins erratic norm clip; outlier filter
TABLE VIII
FOOTPRINT AND THROUGHPUT(8-CORECPU, FP32)
Component RAM Throughput
CLIP encoders 350 MB 350 qps (text)
CLAP encoder 220 MB 120 qps (audio)
HNSW (500 items) 6 MB<1 ms search
IVFPQ (120 items) 1 MB≈1 ms search
Telemetry<1 MB<0.2 ms
XIV. PRACTICALTUNINGRECIPES
•When latency spikes:reducen probe to 8, cap rerank depth
at 150, and pin head entities in hot.
•When Safe@1 drops:increaseNcadence (e.g., 512→256)
for faster bridge updates; briefly expandK.
•When warm recall is low:raisen list(100→200) and retrain
with a fresh hot sample.
•When audio accuracy is low:add 1k domain-caption pairs;
fine-tune CLAP-text head only; keepTorthogonal.
•When memory is tight:increase PQmfrom 8→12 with
6 bits; head pinning avoids tail collapse.
•When ingestion bursts:route queries to hot-only for 10–30
s, then resume hot+warm.
XV. ROBUSTNESS ANDFAILUREINJECTION
We validated operational robustness using synthetic pertur-
bations.
XVI. COMPUTEFOOTPRINT ANDENERGY
We report RAM usage, steady-state throughput, and ap-
proximate energy. Measurements were taken on an 8-core
laptop CPU (FP32 inference) under two steady loads (text-
only and audio queries). Power was sampled via Linux RAPL
(powercap) at 100 Hz after a 60 s warmup.
Method.LetP idlebe the baseline package power at rest,
Pactive the power under load, andλthe sustained query rate
(qps). The incremental energy per query is estimated by
Equery≈Pactive−P idle
λJ.(2)
One-off tasks useE task≈¯P·t task.
Energy estimates (illustrative).Using the above method:
•Text-only (200 qps):P idle≈6.0W,P active≈10.8W⇒
Equery≈24mJ.•Audio (50 qps):P idle≈6.0W,P active≈12.4W⇒E query≈
128mJ.
Maintenance overheads are small: IVFPQ retrain on∼5k
vectors (n list=100) completes in∼1–2 s on CPU (≲30 J),
while a Procrustes SVD refresh atd=512withN=512pairs
takes<0.2s (≲3 J). These costs amortize over thousands of
queries.
XVII. REPRODUCIBILITYCHECKLIST
•Code/Artifacts:exact model checkpoints (CLIP ViT-B/32,
CLAP), commit hash, FAISS version.
•Data release:folder lists, BLIP captions, TTS command
lines, augmentation seeds.
•Configs:B,n list,m, bits,n probe,N; rerank depth; batch
sizes.
•Metrics:nDCG@k, Recall@k, MRR,ε,ζ, Safe@k,
p50/p95 latency.
•Plots/Tables:baseline, PQ spill, audio alignment, telemetry
over time, scaling curves.
•Environment:OS, CPU/GPU, Python/PyTorch/FAISS ver-
sions; BLAS.
•Determinism:fixed seeds for eval; non-determinism only
in HNSW construction (documented).
•Monitoring:saved logs forε,ζ, and margins; alert thresh-
olds.
•Ethics:bias audit checklist; PII/consent policy; moderation
hooks.
XVIII. LIMITATIONS ANDETHICS
Audio accuracy is modest due to TTS/CLAP domain mis-
match; bridges are linear; IVFPQ cold-start needs a minimum
warm size. Ethical deployment requires bias audits, privacy-
aware retention, and moderation of retrieved evidence.
ACKNOWLEDGMENTS
We thank colleagues and open-source maintainers whose
tools and datasets enabled this work. Any errors are our own.
XIX. CONCLUSION ANDFUTUREWORK
LUMA-RAG shows that streaming alignment, tiered mem-
ory, and safety telemetry enable continuous multimodal RAG
without re-indexing. Future work: non-linear bridges, learned
memory policies, faithfulness scoring, and larger open bench-
marks.
REFERENCES
[1] P. Lewis, E. Perez, A. Piktus, et al., “Retrieval-augmented generation
for knowledge-intensive NLP tasks,” NeurIPS, 2020.
[2] A. Radford, J. W. Kim, C. Hallacy, et al., “Learning transferable visual
models from natural language supervision,” ICML, 2021.
[3] B. Elizalde, H. Tang, M. Fonseca, et al., “LAION-CLAP: Learning audio
concepts from natural language supervision,” ICASSP, 2023.
[4] Y . A. Malkov and D. A. Yashunin, “Efficient and robust approximate
nearest neighbor search using HNSW,” IEEE TPAMI, 2020.
[5] W. Zhou, Z. Zhao, J. Zhang, and X. Sun, “Multimodal retrieval-
augmented generation,” arXiv:2302.07842, 2023.
[6] S. Zhang, H. Li, and K. Chen, “A survey of multimodal RAG systems,”
arXiv:2401.04000, 2024.
[7] Y . Zeng, X. Li, and J. Luo, “Bridging multimodal grounding and LLMs:
A survey on multimodal RAG,” arXiv:2403.09334, 2024.

[8] S. Shrivastava and M. Li, “Taming memory: Memory-efficient multi-
vector retrieval,” ICML, 2023.
[9] Y . Tan and Z. Ma, “Diverse and timely spilling for tiered memory in
streaming ANN retrieval,” arXiv:2309.11219, 2023.
[10] J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search
with GPUs,” IEEE Trans. Big Data, 2019.
[11] S. Sun, Q. Xie, and X. Ma, “A survey of cross-modal retrieval: Methods,
datasets, and applications,” IEEE TPAMI, 2023.[12] A. Shrivastava, A. Gupta, and R. Girshick, “Training region-based object
detectors with online hard example mining,” CVPR, 2016.
[13] Y . Li, N. Duan, and X. Chen, “Unpaired image-to-text retrieval via deep
canonical correlation alignment,” NeurIPS, 2019.
[14] K. He, H. Fan, Y . Wu, S. Xie, and R. Girshick, “Momentum contrast
for unsupervised visual representation learning,” CVPR, 2020.
[15] Z. Li and D. Hoiem, “Learning without forgetting,” ECCV , 2016.