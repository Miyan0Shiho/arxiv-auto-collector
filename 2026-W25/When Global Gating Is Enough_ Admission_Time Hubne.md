# When Global Gating Is Enough: Admission-Time Hubness Control in Anisotropic Vector Retrieval Systems

**Authors**: Prashant Kumar Pathak, Tarun Kumar Sharma

**Published**: 2026-06-18 01:40:36

**PDF URL**: [https://arxiv.org/pdf/2606.19692v1](https://arxiv.org/pdf/2606.19692v1)

## Abstract
Vector hubness, where a few points become nearest neighbors of many queries, creates a poisoning risk in retrieval-augmented generation (RAG): one injected document can influence unrelated requests. Existing defenses use periodic reverse-kNN scans, leaving an exposure window and repeated corpus-wide work. We study admission-time control, scoring each candidate against sentinel queries and quarantining hub-like documents before insertion. Across two 100,000-document corpora, five encoders, and disjoint attacker and defender query sets, a global gate achieves recall 1.0 at the decisive embedding-space point (>=0.92 across the effective range) and 0.91 +/- 0.07 on HotFlip attacks, with 1% false positives on general documents. A per-topic gate provides no reliable benefit, consistent with anisotropy coupling local and global visibility. Thresholds are maintained incrementally, with corpus-size-independent insertion cost and amortized deletion cost. On HNSW, admission adds about 3.1% to ingestion latency, scoring remains flat to 10^6 vectors, and 1.2% of decisions flip under approximate indexing, none involving attacks. Provenance complements the gate for natural or tight-domain hubs.

## Full Text


<!-- PDF content starts -->

When Global Gating Is Enough: Admission-Time Hubness Control in Anisotropic Vector
Retrieval Systems
Prashant Kumar Pathak Tarun Kumar Sharma
Santa Clara, CA, USA Colonia, NJ, USA
prashant.pathak@ieee.org tarun.sharma@ieee.org
Abstract
Vectorhubness—the tendency of a few points to be nearest neighbours of disproportionately many queries in high-dimensional
space—is an attack surface for retrieval-augmented generation (RAG): an adversary injects one crafted document that is retrieved for
many unrelated queries, turning a single record into a broad poisoning vector. The prevailing defence isdetection: periodically scan
the index, estimate reverse- kNN influence, and remove outliers, which leaves an exposure window and requires repeated corpus-wide
rescans. We studyadmission-timecontrol—scoring each document’s hub behaviour against sentinel queries and refusing it before it
becomes retrievable—and the systems question of whether the control can be maintainedincrementallyrather than by rescans. On a
100,000-document corpus assembled from four BEIR collections (500 topics, 5 ,571 sentinels, bge-large-en-v1.5 , exact kNN)
and in an evasion setting where attacker anchors aredisjointfrom the defender’s sentinels, a singleglobaladmission gate recovers
effective adversarial hubs on embedding-space attacks at recall 1 .0 at the decisive operating point ( ≥0.92 across the effective range)
and AUROC≈1.0 at an≈1% false-positive rate on in-domain general documents (organic natural hubs remain a residual requiring
provenance-based adjudication); on harder gradient-realised (HotFlip) attacks the global gate averages 0 .91±0.07 recall across
five seeds, at least as good as a per-topic gate (0 .85±0.05). The gate matches a recent reverse- kNN detector and dominates a
provenance-only baseline that would require trusting 99% of sources. A more elaboratedomain-aware(per-topic) gate provides no
statistically reliable benefit: its marginal over the global gate is+0 .000 in the cross-encoder (five encoders, 384–1024 dimensions:
MiniLM, BGE, GTE, E5) and cross-corpus (two compositionally distinct 100k-document corpora) evaluations, while the five-seed
suppression sweep yields+0 .018±0.025 (a confidence interval including zero). We give a geometric account: embedding spaces are
anisotropic, so per-topic query centroids are positively aligned with the global centroid (strongly in four of five encoders), so a hub
cannot easily be simultaneously topic-loud and globally-quiet; a direct vector-space optimisation over the unit sphere did not find
such a vector in the evaluated searches. On the systems side we maintain the admission thresholds incrementally—exact against
full recomputation, with per-write cost independent of corpus size (amortised for deletion)—and evaluate the full pipeline on a real
approximate (HNSW) index: admission adds ≈3.1% to ingestion latency (0 .6% for candidate scoring), its candidate-scoring cost is
flat to N=106, it tolerates approximate indexing (1 .2% of admit/quarantine decisions flip under HNSW, none of them attacks), and
concurrent ingestion scales 5 .9×across eight threads. The contribution is a preventive, incrementally-maintainableglobaladmission
gate, a reproducible negative result on domain-aware refinement, and a geometric explanation of when global control suffices; we
treat provenance as a complementary control for the residual (weak or tight-domain hubs).
Keywords:retrieval-augmented generation, vector database security, nearest-neighbour hubness, data poisoning, admission control,
incremental maintenance, embedding anisotropy, approximate nearest neighbour search
1. Introduction
Retrieval-augmented generation (RAG) [ 3,4] grounds a lan-
guage model on documents fetched from a vector store by em-
bedding similarity. The retrieved passages are treated as trusted
context, so the store is a security boundary: a document retrieved
for a query can steer the model’s output for that query.Hub-
ness—the concentration, in high-dimensional spaces, of reverse-
nearest-neighbour (reverse- kNN) counts on a few points [ 1]—
turns this into an attack surface. An adversary who crafts ahub
document can have it retrieved across many unrelated queries,
so a single injected record influences a large fraction of interac-
tions [11, 9, 8].The prevailing defence is detection: periodically scan the
index, estimate reverse- kNN influence, and remove statistical
outliers [ 16]. Detection after the fact has two structural costs: an
exposure windowbetween a hub’s insertion and the next scan,
during which it hijacks queries, and the expense of rescanning
a large store. This paper asks whether the same statistic can
be appliedat admission—evaluated on each documentbefore
it becomes retrievable, so a hub never enters the index—and
whether that control can be maintainedincrementallyper write
rather than by repeated full rescans. A useful gate must also
separate adversarial hubs fromlegitimatelygeneral documents
(glossaries, overviews) that are naturally hub-like, or it wouldarXiv:2606.19692v1  [cs.CR]  18 Jun 2026

break normal ingestion.
A natural refinement is to condition the statistic on topic
(“domain-aware” detection), on the intuition that a concept-
specific hub may be loud within one topic yet quiet globally and
thus slip past a global threshold. Our central empirical finding
is that, on dense single-vector retrieval, this refinement buys
nothing: across the evaluated encoders, corpora, attacks, and
operating points, domain-aware gating provides no statistically
reliable benefit. We explain this geometrically and show that
the global gate is cheap enough to run synchronously on the
ingestion path of a real approximate-nearest-neighbour (ANN)
index.
Contributions.
1.Apreventiveadmission-time formulation of hubness con-
trol that evaluates documents before they are retrievable,
removing the exposure window left by detective scanning
(Sections 3, 5).
2.Ascalable incremental-maintenancescheme for the ad-
mission thresholds—exact against full recomputation, with
per-write cost independent of corpus size (amortised for
deletion)—and a real-index systems evaluation (HNSW;
end-to-end latency, scalability to N=106, approximate-
index robustness, concurrency) showing the gate adds
negligible relative overhead in the evaluated ingestion
pipeline (Sections 5, 9).
3.Areproducible negative result: a domain-aware per-topic
gate provides no measurable benefit over the global gate
across five encoders (384–1024 d; MiniLM/BGE/GTE/E5),
two compositionally distinct corpora, two concept defini-
tions, and a five-seed sweep (Sections 7, 11).
4.Ageometric explanation—embedding anisotropy couples
local and global visibility—that provides a geometric ac-
count of the observed sufficiency of global gating and
identifies a tight-domain exception (Sections 8, 10).
2. Threat Model and Assumptions
System.A vector store holds unit-normalised embeddings of
single-passage documents under a fixed encoder; queries retrieve
the top- kby cosine similarity (equivalently inner product on
normalised vectors).Adversary.The attacker injects documents
through the ingestion path but cannot alter the encoder, other
stored vectors, or the retrieval code. The goal is a hub retrieved
for manyanchorqueries the attacker cares about. We grant the
attacker knowledge of the defence and the ability to optimise
hubs by gradient methods, and—as the realistic evasion case—
hold the attacker’s anchor queriesdisjointfrom the defender’s
sentinels (measured overlap 0 .000).Out of scope.Targeted
single-query poisoning [ 9,10], multi-vector/late-interaction
retrieval [ 6], and model-internal poisoning are out of scope and
revisited as limitations.
3. Admission-Time Global Gate
Let the corpus be C={x1,...,x N},xi∈Rd,∥xi∥=1, and
letQ={q1,...,q M}be sentinel-query embeddings. For eachAlgorithm 1Candidate scoring and admission
Require:candidated, sentinelsQ, thresholds{τ j}, budgetθ
1:h←1
MP
j1[⟨d,q j⟩>τ j]
2:ifh≤θthen
3:returnadmit
4:else
5:returnquarantine
6:end if
sentinel qjdefine a threshold τjas the similarity of its k-th
nearest neighbour in C, so a candidate dwould enter qj’s top- k
iff⟨d,q j⟩>τ j. Theglobal hub scoreis the fraction of sentinels
the candidate would displace into,
h(d)=1
MMX
j=11h
⟨d,q j⟩>τ ji
,(1)
and the gate admits difh(d)≤θ, quarantining it otherwise, with
θcalibrated to a target false-positive budget on held-out natural
documents. A normal single-topic document beats only a small
fraction of sentinels; being retrieved by many is the adversarial
signal.
4. Domain-Aware Alternative and Geometric Hypothesis
The natural refinement conditions on topic. With per-topic
local hub rate st(d) over topic t’s sentinels, a worst-topic robust- z
score
z(d)=max
tst(d)−med t
max 1.4826 MAD t,1/n t
(here MAD tis the median absolute deviation; the scale floored
at the rate resolution 1 /ntavoids a degenerate zero-MAD blow-
up) is intended to catch a hub that is quiet globally but loud
within one topic. Thegeometric hypothesiswe test is that, under
anisotropy, no such hub exists, so z(d) carries no information
beyondh(d) (Section 8).
5. Incremental Systems Architecture
The gate sits synchronously on the ingestion path: admit-
ted documents are indexed, flagged documents quarantined for
review. An external, slowly-updated baseline supplies θand re-
sists a dilution attack (mass injection that shifts the corpus’s own
statistics); a velocity monitor watches for many sub-threshold
documents converging on one region; a periodic detective scan
remains as a backstop for queries outside the sentinel set. Each
sentinel maintains its current top- Bsimilarities ( B=k+c) in
a buffer with row ids, so τjis the k-th entry and scoring is one
pass over the sentinels (Algorithms 1–2).
Complexity.Scoring and insertion cost O(Md) and O(Md+
alogk ) (a=displaced sentinels); buffered deletion costs O(MB)
for the membership check with an occasional O(N) refill that
amortises to a constant (a sentinel underflows only every ∼(B−k)
buffer hits, and the buffer-hit rate is ∼1/N). Insertions and
2

Algorithm 2Incremental threshold maintenance (insert/delete)
1:procedureInsert(d)
2: forjwith⟨d,q j⟩>minB jdosplice⟨d,q j⟩into Bj,
keep top-B;τ j←B j[k]
3:end for
4:end procedure
5:procedureDelete(x)
6: forjsuch that x∈B j(scan, O(MB) total)dodrop x
fromB j
7:if|B j|<kthenrefillB jby one rescan▷rare
8:elseτ j←B j[k]
9:end if
10:end for
11:end procedure
Figure 1: Incremental threshold maintenance on the real bge-large embed-
dings: per-write insert/delete cost is flat in Nand exact against full recomputa-
tion, whereas the naive recompute isO(N).
ordinary deletions are N-independent; deletion is amortised N-
independent under the stated random-delete workload, with an
O(N) worst-case refill. Per-write costs grow with the sentinel
count Mand embedding dimension d(and, under an approxi-
mate index, with the ANN query used to maintain τ). Space is
O(MB+Md). The maintained thresholds areexact: on the
realbge-large embeddings at thedeployedsentinel count
(M=5,571, k=10,B=50, 200 randomly generated update op-
erations at each tested corpus size) the per-write update is flat
inN(≈0.72 ms insert,≈0.08–0.11 ms delete from N=104to
8×104) with maximum error 0 and zero buffer refills, while a
naive full recompute grows linearly (0 .56 s→4.2 s); see Fig. 1.
Section 9 evaluates the gate end-to-end on a real approximate
index.
6. Experimental Methodology
Corpus. N=100,000 documents from four BEIR collec-
tions [ 34], gold-prioritised then subsampled: FiQA (finance;
the concept target; 57 ,318), TREC-COVID (33 ,866), SciFact
(5,183), NFCorpus (3 ,633); passages under 80 characters dropped;
10,200 grounded queries; 328 /500 topics contain a query.En-
coder. BAAI/bge-large-en-v1.5 (dim 1024) [ 29] viasentence-
transformers [32];float32 ,max_seq_length 256, L2-normalised
once (cosine=inner product); the query-side instruction isprepended to queries only. Hardware: Apple-silicon (arm64)
with the MPS backend.Topics. T=500 clusters from MiniBatchKMeans
on the L2-normalised embeddings; a query’s topic is the major-
ity topic of its gold documents.Search.Exact brute-force kNN
(k=10) for the separation study; the systems study (Section 9)
uses a real Hierarchical Navigable Small World (HNSW) in-
dex [ 18].τjis each sentinel’s k-th-NN similarity over the clean
corpus.Sentinels. M=5,571: held-out evaluation queries (half
of each topic’s queries) plus the 500 cluster centroids.Splits.
Within each topic, queries are split evenly into sentinels and at-
tacker anchors; sentinel–anchor overlap 0 .000.Negatives.The
headline natural-negative class is N-organic ∪N-curated: 1 ,000
organicnatural hubs (top-1% by reverse- k-occurrence over the
queries) and 1 ,500curatedin-domain definitional/overview pas-
sages mined from the corpus (excluding organic). Two sanity
floors: 1,500 cross-domain Simple-English-Wikipedia passages
and 2,000 random documents. A disjoint benign calibration
set (5,000 random documents) supplies the robust- zbaseline;
attacks and evaluation negatives are never used for calibration.
Calibration.The operating threshold θis selected to give a 1%
false-positive rate (FPR) on thedisjoint5 ,000-document benign
calibration set (never the evaluation negatives), then frozen; we
report attack recall and the per-population false-positive rate at
that fixed threshold (Table 2). For corpus-derived benign candi-
dates (organic, curated, random, calibration), admission scores
use leave-one-out thresholds, so each document is evaluated as
a pre-insertion candidate; external candidates (injected attacks,
cross-domain Wikipedia) use the standard threshold.Attacks.
Universal hubs (24, random anchor subsamples); concept hubs
by a gradient hub-rate search on the unit sphere across a suppres-
sion sweep λ∈{ 0,0.5,1,2,4,8,16}(Table 3 lists a representa-
tive subset; 8 targets, 5 seeds; an effective concept hub retrieves
≥20% of its target topic), with a closed-form mean-cosine attack
as reference; and HotFlip-through-BGE [ 12] text-constrained
hubs (4 targets×5 seeds, length 32, 120 steps), realising the
λ∗=2 direction as valid but non-fluent token sequences. Cisco’s
open detector [ 16] (the hubscan software [ 17]) is run as a base-
line on the same corpus, sentinels, and hubs.Reproducibility.
Five seeds{20260612,7,42,123,2024}reuse the cached cor-
pus embedding and re-randomise everything downstream that
drives the verdict.Software.Python 3.11 with torch 2.8,
sentence-transformers 5.2,scikit-learn 1.7,hnswlib
0.8,numpy /scipy , on arm64 macOS (MPS); encoder model
cards: BAAI/bge-large-en-v1.5 ,BAAI/bge-base-en-v1.5 ,
intfloat/e5-large-v2 ,thenlper/gte-large ,sentence-transformers/all-MiniLM-L6-v2 .
7. Security Effectiveness
Benign floor (no attacks).Curated in-domain passages are
statistically indistinguishable from random (curated-vs-random
area under the ROC curve, AUROC, 0 .518): genuinely-general
documents are not hub-like; the cross-domain Wikipedia set
is below random (0 .414). Organic natural hubs are separable
from random (AUROC 0 .992): they genuinely behave like hubs.
Table 1 reports the three gate variants. With the threshold frozen
on the disjoint calibration set (Section 6), the false-positive
rate splits sharply by population (Table 2): genuinely-general
3

Figure 2: Global and per-topic recall coincide for every effective hub; per-topic
never exceeds global and falls below once the attack is ineffective. Target hit-
rate declines with the global hub rate as λgrows—the coupling predicted by
anisotropy.
documents are admitted cleanly (curated 1 .1%, random 1 .2%,
Wikipedia 0 .0%), butorganic natural hubs are flagged at86%.
This is not a tuning artefact: an organic natural hub and an
adversarial hub are both hub-like, so the statistic alone cannot
separate them. The gate therefore admits genuinely-general
content cleanly and detects all universal and concept hubs at
the pre-specified decisive operating point (recall 1 .0); organic
natural hubs fall in the flagged set and constitute the residual
that provenance (Section 10) must adjudicate.
Global vs. domain-aware (embedding-space attacks).Across
the suppression sweep (Table 3, Fig. 2) per-topic recall equals
global recall for every effective hub and never exceeds it; the
maximum marginal is+0 .000. Atλ=8 the hub already re-
trieves only 7% of its target topic—it has ceased to be effective—
and there per-topic (0 .342) is below global (0 .355). On the
embedding-space attacks the global gate reaches recall 1 .0 on
universal hubs and on concept hubs at the decisive operating
pointλ∗(AUROC 1.0;≥0.92 across the effective λrange, Ta-
ble 3).
Fig. 3 shows the decisive score distributions: adversarial
hubs separate cleanly from the natural-general negatives un-
der the global gate: adversarial hubs separate from genuinely-
general documents, while naturally-occurring hubs overlap with
the adversarial population and require provenance-based adjudi-
cation.
Baselines.On the same corpus, sentinels, and hubs the gate
matches the reverse- kNN detector [ 16] at recall 1.0/AUROC 1 .0
(Table 4); in our reproduction on this corpus both the global and
domain-aware modes are recall-saturated, so this setting has no
headroom to exhibit a domain-aware advantage; the detector’s
own paper reports settings where domain-scoped scanning re-
covers targeted attacks that evade global detection, consistent
with our geometric account that such an advantage appears only
where domains are isolated (the tight-domain residual of Sec-
tion 10). A provenance-only baseline flags every injected hub
but also every untrusted natural document, so a provenance-only
policy that rejects every untrusted document meets a 1% benign
false-positive rate only if at least 99% of benign sources are
already classified as trusted.Realised (HotFlip) attacks.HotFlip-through-BGE hubs
reach mean cosine 0 .864 (0.84–0.89) of the embedding-space
optimum (valid but non-fluent tokens). On the pre-specified run
the global gate recovers them at recall 1 .0 (AUROC 1 .0) and the
per-topic gate at 0 .93 (AUROC 0 .997; this pre-specified seed
is not bitwise-reproducible across machines, so we report the
five-seed reproduction value of Section 11), at a 1 .1% curated-
document false-positive rate at the frozen operational threshold.
The realised attack is the hardest case for both gates and is
analysed across seeds in Section 11.
Robustness to encoder.Table 5 repeats the comparison on
five encoders spanning 384–1024 dimensions and four architec-
tures. The per-topic marginal over global is+0 .000 on every
encoder; the global gate’s universal recall is stable throughout
(concept recall is 1 .0 except on the most anisotropic encoder,
e5-large, at 0 .58—a case that is hard forbothgates). On the
most isotropic encoder (MiniLM), the per-topic gate’s curated-
document FPR rises to 0 .33 under MiniLM (Table 5) and its
calibration is unstable across clusterings while the global gate
remains stable—another reason to prefer global. Anisotropy is
reported as the mean cosine between per-topic query centroids
and the global query centroid; the per-topic marginal does not
vary with it (Fig. 4).
Robustness to corpus.To test generality beyond the assem-
bled finance/biomedical corpus, we rebuilt the entire pipeline
on a compositionally distinct 100 ,000-document general-web/
encyclopedic corpus (NQ, HotpotQA, DBpedia-entity, Quora;
37,033 grounded queries; 8 ,000 sentinels selected as in Sec-
tion 6—the eval half of each domain’s queries, capped at 7 ,500,
plus the 500 cluster centroids). The finding reproduces exactly
(Table 6): the benign floor holds (curated-vs-random AUROC
0.514, organic-vs-random 0 .996), the global gate recovers every
effective concept hub at recall 1 .0 (AUROC 0 .999) (curated-
document FPR 0.9%, organic-hub flag rate 94%, mirroring cor-
pus 1), and the per-topic marginal over global is again+0 .000
ateverysuppression level. On this corpus effective hubs persist
to higherλ(the decisive hub at λ∗=8 still retrieves 41% of its
target topic and is caught at recall 1 .0), so the global gate’s suffi-
ciency is, if anything, more pronounced. The negative result is
thus robust across two corpora of different domain composition
in addition to the five encoders above.
8. Geometric Analysis
The negative result is consistent with an anisotropy-based
geometric mechanism (we show alignment, not sole causa-
tion). In these embedding spaces, per-topic query centroids
are positively aligned with the global query centroid (mean
topic-centroid·global cosine 0 .47–0.96; strong in four of five en-
coders), so the component of a candidate that raises its topic hub
rate also raises its global hub rate: a hub cannot be topic-loud
and globally-quiet at once. A direct vector-space optimisation
over the unit sphere—maximising topic hub rate subject to a
global-rate ceiling—did not find a topic-loud/global-quiet vector
in the evaluated searches, and the target-topic hit-rate falls in
lock-step with the global rate as λincreases (Table 3). This
anisotropy is the well-documented narrow-cone geometry of
4

Table 1: Benign false-positive floor (no attacks). Scores: global is a rate; per-topic and rknn-zare robustz-scores.
Gate N-organic p50/p99 N-curated p99 organic-vs-random AUROC
global 0.0011/0.0029 0.0007 0.992
per-topic 3.0/9.0 3.0 0.965
rknn-z6.0/16.0 4.0 0.992
Figure 3: Score distributions for natural-general negatives vs. universal and concept adversarial hubs. Effective hubs separate cleanly; the natural-general docs sit at
low scores. This panel is descriptive (score distributions); it is not evaluated at the frozen operating threshold—per-population false-positive rates at that threshold are
in Table 2.
Table 2: Leakage-free, leave-one-out benign false-positive rate per population:
the operating threshold is frozen on the disjoint 5 ,000-document calibration set
(global gate, θ=0.0007); corpus-derived candidates are scored pre-insertion
(leave-one-out). Genuinely-general documents are admitted cleanly; organic
natural hubs are not separable from adversarial hubs by the statistic.
benign populationNFP FPR
organic natural hubs (top-1% rev-k) 1,000 862 0.862
curated (definitional/overview) 1,500 16 0.011
random corpus documents 2,000 24 0.012
cross-domain Wikipedia 1,500 0 0.000
attack recall at the same frozen threshold: universal 1.000, concept 1.000
Figure 4: Across five encoders the per-topic gate’s marginal recall over the global
gate is≈0 regardless of anisotropy: domain-aware refinement adds nothing on
dense single-vector retrieval.
contextual and sentence embeddings [ 23,24,25,26,28]; our
result connects it to a security consequence: where the cone
is narrow, a single global threshold is sufficient and a domain-
aware statistic is redundant.9. Systems Evaluation
We evaluate the gate end-to-end on arealapproximate index
(HNSW via hnswlib [ 18]) on the real bge-large embeddings,
measuring the cost of running admission synchronously on the
ingestion path.
End-to-end ingestion.Per-document ingestion has four
synchronous stages: encode 26 .4 ms, candidate scoring 0 .18 ms,
threshold maintenance 0 .72 ms (incremental τupdate at the
deployed M=5,571; p95 1.1, p99 1.4 ms), and HNSW insert
2.0 ms (encode/scoring/insert p99 51 .6/0.32/3.8 ms)—a full in-
gestion path of 29 .3 ms. Theadmission-control overhead—
candidate scoring plus synchronous threshold maintenance—is
(0.18+0.72)/29.3≈3.1%, of which candidate scoring alone
is 0.6%. The four stages are timed independently (scoring:
the sentinel matmul and threshold compare; maintenance: the
buffer update; insertion: the HNSW add) and reported as non-
overlapping medians, so the 3 .1% does not double-count. Bulk
index construction (hnswlib’s multi-threaded add_items ) runs
at 6,900 docs/s and adds 440 MB for 100k vectors.
Scalability.We separate two claims.Exactincremental
threshold maintenance was evaluated through N=8×104
(Section 5, Fig. 1). Candidate-scoring and HNSW-insert latency
were probed on an index grown to N=106—real corpus vectors
through N=105, then synthetically extended for insert-timing
only (Fig. 5, left); the gate’s per-write scoring latency isflat(139–
273µs at every N), while HNSW insertion grows sub-linearly
(1.4 ms→6.2 ms). The gate’s N-independence—the design’s
central claim—holds on a real index, not just in the data-structure
model of Section 5. Peak resident memory atN=106is 9 GB.
Approximate-index robustness.The right question is not
how farτdrifts but whether the admit/quarantinedecisionchanges
whenτis maintained against a real HNSW index instead of ex-
actkNN. With HNSW at recall@10 ≈0.93–0.96, only 1.2% of
5

Table 3: Gradient concept attack across the off-cluster suppression sweep λ(embedding-space). Concept recall is 1 .0 at the pre-specified decisive point λ∗=2; the
minimum over the effective range (λ≤4, target hit-rate≥20%) is 0.925.
λglobal rate target hit-rate recall global recall per-topic
0 0.211 0.93 1.000 1.000
2 (λ∗) 0.066 0.59 1.000 1.000
4 0.011 0.25 0.925 0.925
8 0.000 0.07 0.355 0.342
16 0.000 0.00 0.000 0.000
Table 4: Reverse-kNN detector baseline [16] on the same data.
variant family r@0.4% r@1% AUROC separation
global universal 1.000 1.000 1.000 12.00
global concept 1.000 1.000 1.000 11.56
concept-aware universal 1.000 1.000 1.000 6.22
concept-aware concept 1.000 1.000 1.000 6.22
Table 5: Per-topic gating adds nothing across encoders (full N=105). Anisotropy is the mean topic-centroid ·global-centroid cosine; “marginal” is the maximum
per-topic−global recall over theλsweep.
encoder dim anisotropy per-topic marginal concept recall (global) per-topic curated FPR
all-MiniLM-L6-v2 384 0.466+0.000 1.000 0.330
bge-base-en-v1.5 768 0.848+0.000 1.000 0.000
bge-large-en-v1.5 1024 0.860+0.000 1.000 0.000
gte-large 1024 0.952+0.000 1.000 0.000
e5-large-v2 1024 0.961+0.000 0.580 0.000
decisions flip over all candidates, andnoattack decision flips
(universal and concept recall stay 1 .0); the flips concentrate in
the borderline organic-hub population (58 /1000, 5.8%); curated
(5/1500), random (8 /2000), and Wikipedia (1 /1500) popula-
tions flip<0.5% and the attack families not at all (0 /24 univer-
sal, 0/40 concept) (underlying threshold drift 0 .0015 mean, p99
0.017). The gate’s decisions are robust to the approximate index
that production systems actually run.
Concurrency and throughput.The gate is a matmul and
vectorises: batched scoring reaches 142 ,000 docs/s; HNSW in-
sertion remains the dominant throughput bottleneck, though
concurrent gate execution introduces measurable contention
at higher thread counts. Under concurrent ingestion (Python
threads, each gating and inserting into a shared index), through-
put scales 5 .9×across eight threads to 3 ,523 docs/s—a≈29%
reduction versus insert-only at eight threads (4 ,951→3,523
docs/s), attributable to Python-level contention and shared-index
memory traffic (Fig. 6).
Faithfulness of the sentinel statistic.The gate approxi-
mates a full reverse- kNN scan by a sentinel sample; faithfulness
is the Spearman correlation between h(d) and the full reverse-
kNN count. With the evaluation configuration (sentinels drawn
from the evalhalfof queries, ≈50% coverage) it is 0 .77 (Spear-
man over a 20 ,000-document sample; a smaller 4 ,000-document
sample gives 0 .75); this is a coverage artefact, not a mechanismlimit. Sweeping sentinel coverage Q(Fig. 7) raises it monoton-
ically to 0.92 at Q=8,000 and 0.98 at full query coverage (a
pure subsample reaches 1 .000, so the τmechanism itself loses
almost nothing). Because candidate-scoring latency scales with
Mand is 0.18 ms at M=5,571; it remains below 0 .5 ms at full
query coverage. Total admission overhead, including threshold
maintenance (which also grows with the sentinel count), should
be considered separately.
10.Tight-Domain Residual and Complementary Provenance
Defining concepts as real corpusdomains(rather than k-
means clusters) reproduces the+0 .000 per-topic marginal. The
exception bounds the central claim: in a tight, distinct, under-
sampled domain (TREC-COVID, 50 queries) an effective con-
cept hub can be globally quiet and evadebothgates, and there
the concept-aware variant isworsethan global because natu-
ral in-domain documents are themselves concentrated, raising
the within-domain floor. This is consistent with the geometric
account: an isolated domain is less collinear with the global
centroid. For such residual cases, documentprovenance(source
trust) is a promising complementary control, orthogonal to the
geometric signal; we report a provenance-onlybaseline and
do not claim an integrated gate-plus-provenance policy as an
implemented contribution.
6

Table 6: The result reproduces on a second, compositionally distinct corpus (embedding-space attacks,N=105, bge-large). Per-topic gating again adds nothing.
metric corpus 1 (finance/biomed) corpus 2 (general-web)
curated-document FPR 0.011 0.009
random-document FPR 0.012 0.011
organic-hub flag rate 0.862 0.936
universal recall 1.000 1.000
concept recall atλ∗1.000 1.000
min. recall across effectiveλ0.925 1.000
per-topic marginal over global+0.000+0.000
Figure 5: Real-HNSW systems evaluation. Left: candidate-scoring latency remains flat through N=106, while HNSW insertion grows with index size. Right:
median ingestion latency by stage (encode, candidate scoring, threshold maintenance, HNSW insert)—admission control adds ≈3.1% to median ingestion latency;
candidate scoring alone contributes 0.6%.
11. Reproducibility
The comparison between the global and domain-aware gates
rests on two metrics near a decision boundary and sensitive to
nondeterminism in clustering, the anchor split, and attack con-
struction. Re-running the downstream pipeline for five seeds
(reusing the cached corpus embedding) gives, on the realised
HotFlip attack, global recall 0 .91±0.07, per-topic recall 0 .85±
0.05, and per-topic marginal over global+0 .018±0.025 (Ta-
bles 7, 8, Fig. 8): no statistically reliable improvement from
domain-aware refinement. The core separation is stable (uni-
versal recall 1 .000, embedding-space concept recall 1 .0 at the
pre-specified decisive point (minimum recall 0 .925 across the
effective suppression range), a 1 .1% curated-document false-
positive rate at the frozen threshold, attack realisability ≈0.86).
The pre-specified configuration is the most favourable of the five
for the per-topic gate, so we report the verdict conservatively: a
single favourable run should not be read as a benefit. A paired
analysis (experimental unit: seed, n=5) of the realised-attack
(per-topic−global) recall gives a mean of −0.061 (95% boot-
strap CI [−0.091,−0.023]; Cohen’s d=−1.37): per-topic is, if
anything,worse, and the interval excludes zero on the side that
favours global. The embedding-space sweep marginal is+0 .018(95% bootstrap CI [0 .000,0.044], including zero). A two-sided
sign test is underpowered at n=5 ( p=0.38); every interval is
nonetheless consistent with no benefit from domain-aware refine-
ment. The full pipeline, frozen thresholds, and per-seed artefacts
support full replication and are available from the author upon
request.
12. Related Work
Hubness.The concentration of reverse-nearest-neighbour
counts in high-dimensional data was characterised by Radovanovi ´c
et al. [ 1] and mitigated by similarity re-scaling [ 2]; we use it as
a security signal rather than a retrieval-quality one.RAG and
retrieval poisoning.RAG [ 3,4] over dense retrievers [ 5,6,7]
is vulnerable to corpus poisoning [8], knowledge-corruption at-
tacks (PoisonedRAG [ 9]), SEO/embedding attacks (GASLITE [ 10]),
and adversarial hubness [ 11]; adversarial text is realised by gra-
dient token attacks [ 12,13]. Defences include certified output
robustness (RobustRAG [ 14]), reranking/graph defences [ 15],
and reverse- kNN hubness detection [ 16]; provenance and source-
trust approaches secure the channel rather than the statistic. We
add a preventive, admission-time complement with a negative
result on domain-aware refinement.ANN systems.Production
7

Figure 6: Left: batched gate throughput (vectorised matmul). Right: concurrent ingestion scales with threads; HNSW insertion remains the dominant operation,
although admission control introduces measurable contention at higher concurrency.
Table 7: Five-seed sweep, realised (HotFlip) attack: per-seed recall and AUROC. The pre-specified seed is the most favourable for the per-topic gate. SDs are
population SD (ddof=0).
seed global recall per-topic recall AUROC pt
20260612 (pre-spec.) 1.000 0.930 0.997
7 0.900 0.850 0.975
42 0.900 0.807 0.995
123 0.950 0.850 0.993
2024 0.790 0.800 0.979
mean±std 0.910±0.070 0.847±0.046 0.988±0.009
Figure 7: Gate faithfulness is coverage-limited: Spearman against the full
reverse- kNN scan rises with sentinel coverage Q, reaching≥0.90 by Q≈8,000
and 0.98 at full query coverage.
stores use HNSW [ 18], IVF-PQ [ 19], FAISS [ 20], ScaNN [ 21],
and DiskANN [ 22]; we show admission is cheap relative to
insertion in such an index.Embedding anisotropy.Contextual
and sentence embeddings occupy a narrow cone [ 23,24], ad-
dressed by all-but-the-top [ 25], whitening/flow [ 26,27], and
contrastive training [ 28]; encoders used here are BGE [ 29],
E5 [30], GTE [ 31], MiniLM/SBERT [ 32,33], benchmarked
with BEIR [ 34]/MTEB [ 35]. We tie anisotropy to a security suf-Table 8: Embedding-space suppression-sweep marginal (per-topic −global, max
overλ) by seed—a distinct quantity from the realised-attack recalls of Table 7.
seed emb-space sweep marginal
20260612 (pre-spec.)+0.065
7+0.000
42+0.000
123+0.000
2024+0.025
mean±std+0.018±0.025
ficiency condition.Incremental maintenance and admission
control.Our threshold buffers are an instance of incremen-
tal view maintenance [ 36,37]; admission control is classic in
systems [ 38].Data poisoning.Training-time and web-scale
poisoning [ 39,40] target learning; we target the retrieval channel
at write time.
13. Limitations and Discussion
Realised-attack recall.On HotFlip the global gate is 0 .91±
0.07, not perfect, and dips to 0 .79 on one seed; both gates leave
8

Figure 8: Five-seed distribution of the realised-attack recall: per-topic never
reliably exceeds global, and the pre-specified seed is the most favourable for the
per-topic gate.
a realised-attack residual.Tight-domain residual.An effec-
tive hub in an isolated, under-sampled domain can evade both
gates (Section 10); a controlled domain-compactness/size sweep
and an integrated provenance policy are future work.Encoder
dependence.Concept-hub separation is encoder-dependent (e5-
large at 0.58); theper-topic-adds-nothingresult is robust across
encoders, but absolute separation is not uniform.Single node.
The systems study measures one machine; multi-node/sharded
distributed ingestion is future work, as is end-to-end evaluation
against live indirect-prompt-injection payloads.Attack scope.
Targeted single-query poisoning, multi-vector retrieval, and
adaptive low-amplitude/multi-document attacks are out of scope.
Hardware.All systems measurements use an Apple-silicon
(arm64) laptop with the MPS backend; the absolute latency and
throughput figures should be re-validated on x86 server hard-
ware with a discrete GPU, where the gate’s N-independence and
small relative overhead are expected to hold but the absolute
numbers will differ.Statistical power.The paired bootstrap
and effect size reported in Section 11 rest onn=5 seeds; more
seeds—particularly on the realised (HotFlip) attack—and more
encoders would tighten the interval, though the point estimates
already favour the global gate.
14. Conclusion
A preventive, admission-timeglobalstatistical gate separates
effective adversarial hubs from genuinely-general documents at
a low measured false-positive rate (curated 1 .1%, random 1 .2%,
Wikipedia 0 .0%), in a realistic evasion setting and on par with
a recent detector; naturally-occurring hubs remain statistically
ambiguous and require provenance-based adjudication. The gate
is maintainable incrementally at a per-write cost that is exact and
N-independent (amortised for deletion), and—on a real HNSW
index—adds≈3.1% to the ingestion path (candidate scoring
0.6%), with 1 .2% of admit/quarantine decisions flipping under
approximate indexing (none of them attacks). A domain-aware
refinement provides no benefit in any evaluated setting, a result
consistent with an anisotropy-based mechanism in which topic-
local and global visibility are coupled. We advocate a global
admission gate with provenance as a complementary control forthe tight-domain residual, and identify multi-node scalability
and adaptive-adversary evaluation as the primary next steps.
Declarations
Competing interests.The authors declare that they have no
known competing financial interests or personal relationships
that could have appeared to influence the work reported in this
paper.Funding.This research received no external funding.
Data availability.The code and data supporting this study are
available from the authors upon reasonable request.CRediT
author contributions. Prashant Kumar Pathak:Conceptu-
alization, Methodology, Software, Validation, Formal analysis,
Visualization, Writing – original draft, Writing – review and edit-
ing.Tarun Kumar Sharma:Conceptualization, Methodology,
Validation, Writing – review and editing.
Declaration of generative AI and AI-assisted technologies in
the manuscript preparation process
LLMs were used for editorial purposes in this manuscript,
and all outputs were inspected by the authors to ensure accuracy
and originality. The construction, experiments, and reported
numbers are the authors’ own work; the LLM did not run experi-
ments or generate findings. No proprietary data was shared with
the LLM, and the authors take responsibility for the contents of
this paper.
References
[1]M. Radovanovi ´c, A. Nanopoulos, M. Ivanovi ´c, Hubs in space:
popular nearest neighbors in high-dimensional data,Journal of
Machine Learning Research11 (2010) 2487–2531.
[2]D. Schnitzer, A. Flexer, M. Schedl, G. Widmer, Local and global
scaling reduce hubs in space,Journal of Machine Learning Re-
search13 (2012) 2871–2902.
[3]P. Lewis et al., Retrieval-augmented generation for knowledge-
intensive NLP tasks, in:NeurIPS, 2020.
[4]K. Guu, K. Lee, Z. Tung, P. Pasupat, M.-W. Chang, REALM:
retrieval-augmented language model pre-training, in:ICML,
2020.
[5]V . Karpukhin et al., Dense passage retrieval for open-domain
question answering, in:EMNLP, 2020.
[6]O. Khattab, M. Zaharia, ColBERT: efficient and effective passage
search via contextualized late interaction over BERT, in:SIGIR,
2020.
[7]G. Izacard et al., Unsupervised dense information retrieval with
contrastive learning (Contriever),TMLR(2022).
[8]Z. Zhong, Z. Huang, A. Wettig, D. Chen, Poisoning retrieval
corpora by injecting adversarial passages, in:EMNLP, 2023.
[9]W. Zou, R. Geng, B. Wang, J. Jia, PoisonedRAG: knowledge
corruption attacks to retrieval-augmented generation of large
language models, in:USENIX Security, 2025.
9

[10] M. Ben-Tov, M. Sharif, GASLITE-ing the retrieval: exploring
vulnerabilities in dense embedding-based search, in:ACM CCS,
2025.
[11] T. Zhang et al., Adversarial hubness in multi-modal retrieval,
arXiv:2412.14113 (2024).
[12] J. Ebrahimi, A. Rao, D. Lowd, D. Dou, HotFlip: white-box
adversarial examples for text classification, in:ACL, 2018.
[13] A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, M. Fredrikson,
Universal and transferable adversarial attacks on aligned language
models, arXiv:2307.15043 (2023).
[14] C. Xiang et al., Certifiably robust RAG against retrieval corrup-
tion, arXiv:2405.15556 (2024).
[15] J. Zheng, A. P. Gema, G. Hong, X. He, P. Minervini, Y . Sun,
Q. Xu, GRADA: graph-based reranking against adversarial docu-
ments attack, in:EMNLP, 2025 (arXiv:2505.07546).
[16] I. Habler, V . S. Narajala, S. Koren, A. Chang, T. Saade, Adversar-
ial Hubness Detector: Detecting Hubness Poisoning in Retrieval-
Augmented Generation Systems, arXiv:2602.22427 (2026).
[17] Cisco AI Defense, hubscan : Adversarial Hubness Detec-
tor (software), https://github.com/cisco-ai-defense/
adversarial-hubness-detector(2026).
[18] Yu. A. Malkov, D. A. Yashunin, Efficient and robust approximate
nearest neighbor search using hierarchical navigable small world
graphs,IEEE TPAMI42 (4) (2020) 824–836.
[19] H. Jégou, M. Douze, C. Schmid, Product quantization for nearest
neighbor search,IEEE TPAMI33 (1) (2011) 117–128.
[20] J. Johnson, M. Douze, H. Jégou, Billion-scale similarity search
with GPUs,IEEE Transactions on Big Data7 (3) (2021) 535–
547.
[21] R. Guo et al., Accelerating large-scale inference with anisotropic
vector quantization (ScaNN), in:ICML, 2020.
[22] S. J. Subramanya et al., DiskANN: fast accurate billion-point
nearest neighbor search on a single node, in:NeurIPS, 2019.
[23] K. Ethayarajh, How contextual are contextualized word represen-
tations? Comparing the geometry of BERT, ELMo, and GPT-2
embeddings, in:EMNLP, 2019.
[24] J. Gao et al., Representation degeneration problem in training
natural language generation models, in:ICLR, 2019.
[25] J. Mu, S. Bhat, P. Viswanath, All-but-the-top: simple and effective
postprocessing for word representations, in:ICLR, 2018.
[26] B. Li et al., On the sentence embeddings from pre-trained lan-
guage models (BERT-flow), in:EMNLP, 2020.
[27] J. Su et al., Whitening sentence representations for better seman-
tics and faster retrieval, arXiv:2103.15316 (2021).
[28] T. Gao, X. Yao, D. Chen, SimCSE: simple contrastive learning
of sentence embeddings, in:EMNLP, 2021.
[29] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, J.-Y . Nie, C-
Pack: packed resources for general Chinese embeddings (BGE),
arXiv:2309.07597 (2023).[30] L. Wang et al., Text embeddings by weakly-supervised contrastive
pre-training (E5), arXiv:2212.03533 (2022).
[31] Z. Li et al., Towards general text embeddings with multi-stage
contrastive learning (GTE), arXiv:2308.03281 (2023).
[32] N. Reimers, I. Gurevych, Sentence-BERT: sentence embeddings
using Siamese BERT-networks, in:EMNLP, 2019.
[33] W. Wang et al., MiniLM: deep self-attention distillation for task-
agnostic compression of pre-trained transformers, in:NeurIPS,
2020.
[34] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych,
BEIR: a heterogeneous benchmark for zero-shot evaluation of
information retrieval models, in:NeurIPS Datasets and Bench-
marks, 2021.
[35] N. Muennighoff, N. Tazi, L. Magne, N. Reimers, MTEB: massive
text embedding benchmark, in:EACL, 2023.
[36] J. A. Blakeley, P.-A. Larson, F. W. Tompa, Efficiently updating
materialized views, in:ACM SIGMOD, 1986.
[37] A. Gupta, I. S. Mumick, Maintenance of materialized views:
problems, techniques, and applications,IEEE Data Engineering
Bulletin18 (2) (1995) 3–18.
[38] M. Welsh, D. Culler, Adaptive overload control for busy internet
servers, in:USENIX Symp. Internet Technologies and Systems
(USITS), 2003.
[39] B. Biggio, B. Nelson, P. Laskov, Poisoning attacks against support
vector machines, in:ICML, 2012.
[40] N. Carlini et al., Poisoning web-scale training datasets is practical,
in:IEEE S&P, 2024.
10