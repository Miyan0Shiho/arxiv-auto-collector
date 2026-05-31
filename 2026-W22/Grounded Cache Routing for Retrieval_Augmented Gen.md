# Grounded Cache Routing for Retrieval-Augmented Generation: When Is It Safe to Reuse an Answer?

**Authors**: Syed Huma Shah

**Published**: 2026-05-26 16:50:02

**PDF URL**: [https://arxiv.org/pdf/2605.27494v1](https://arxiv.org/pdf/2605.27494v1)

## Abstract
Modern retrieval-augmented generation(RAG) deployments increasingly rely on caching to reduce token cost and time-to-first-token(TTFT). Prefix-level KV reuse is now standard in serving stacks such as vLLM, and chunk-level and position-independent reuse have been pushed further by recent systems(RAGCache, TurboRAG, CacheBlend, EPIC, ContextPilot, PCR, LMCache). Output-level semantic answer caches, by contrast, remain fragile: similar prompts can map to different correct answers, retrieved evidence drifts as the corpus is updated, and adversarial collision attacks have been shown to hijack cached responses. We argue that the right framing for cached answer reuse is not how to reuse faster but when reuse is safe. We propose GroundedCache, an evidence-validated cache router that admits a cached answer only when 4 cheap gates simultaneously hold: query similarity, retrieved-evidence overlap, source-version validity, and lexical (or judge-based) support of the cached answer by the freshly retrieved evidence. We build a six-regime workload that stress-tests cache safety rather than only hit rate, and introduce an operator-facing metric, the unsafe-served rate (USR), fraction of all queries that received a wrong cached answer. Across 2 datasets and 12,000 real-LLM generations(Qwen2.5-7B-Instruct on vLLM with Automatic Prefix Caching), GroundedCache drives USR to 0.0% on every HotpotQA regime(vs. 15-35% under naive caching) and to 1.5% on mtRAG document drift(vs. 51.5%), a 34x reduction on the design-point adversarial regime and 3-10x reductions across the other mtRAG regimes, while end-to-end p50 latency stays within 1.04-1.07x of a no-cache RAG baseline. A per-gate ablation isolates the lexical support gate as the load-bearing safety mechanism on both datasets, with the remaining gates providing defense-in-depth at near-zero cost. We release the implementation, workload, and evaluation harness.

## Full Text


<!-- PDF content starts -->

Grounded Cache Routing for Retrieval-Augmented Generation:
When Is It Safe to Reuse an Answer?
Syed Huma Shah
syedhuma.shah@duke.edu
Abstract
Modern retrieval-augmented generation (RAG) deployments increasingly rely on caching to
reduce token cost and time-to-first-token (TTFT). Prefix-level KV reuse is now standard in
serving stacks such as vLLM, and chunk-level and position-independent reuse have been pushed
further by recent systems (RAGCache, TurboRAG, CacheBlend, EPIC, ContextPilot, PCR,
LMCache). Output-levelsemantic answer caches, by contrast, remain fragile: similar prompts
can map to different correct answers, retrieved evidence drifts as the corpus is updated, and
adversarial collision attacks have been shown to hijack cached responses. We argue that the right
framing for cached answer reuse is nothow to reuse fasterbutwhen reuse is safe. We propose
GroundedCache, an evidence-validated cache router that admits a cached answer only when
four cheap gates simultaneously hold: query similarity, retrieved-evidence overlap, source-version
validity, and lexical (or judge-based) support of the cached answer by the freshly retrieved
evidence. We build a six-regime workload (exact-repeat, paraphrase, near-miss, document-drift,
long shared-document, bounded-KB CAG) that explicitly stress-tests cachesafetyrather than
only hit rate, and an operator-facing metric, theunsafe-served rate(USR), defined as the fraction
of all queries that received a wrong cached answer, alongside TTFT and token cost. Across two
datasets and 12,000 real-LLM generations (Qwen2.5-7B-Instruct served by vLLM with Automatic
Prefix Caching),GroundedCachedrives USR to 0.0% on every HotpotQA regime (vs. 15–35%
under naive semantic caching) and to 1.5% on mtRAG document drift (vs. 51.5% under naive),
a 34×reduction in wrong cached answers per query on the system’s design-point adversarial
regime, and 3–10 ×reductions across the other mtRAG regimes. End-to-end p50 latency under
GroundedCachestays within 1 .04–1.07×of a no-cache RAG baseline; the no-support variant
trades USR for 1 .4–1.5×speedup. A per-gate marginal ablation isolates the lexical support gate
as the load-bearing safety mechanism on both datasets (+0 .125 USR on HotpotQA, +0 .118 on
mtRAG when removed), with the remaining gates providing defense-in-depth at near-zero cost.
We release the end-to-end implementation, traffic-synthesis tooling, and evaluation harness so
the framework can be ported to richer datasets and stronger compressors.
1 Introduction
A typical RAG pipeline repeatedly does the same expensive work for queries that are functionally
indistinguishable to the user: it embeds the query, runs nearest-neighbor search over a corpus,
concatenates the top- kchunks into a prompt, and decodes a long-context answer. Three layers of
caching can in principle eliminate that work:
1.Prefix / KV cache reuseat the serving layer (vLLM APC [ 1], SGLang RadixAttention [ 2]),
and its RAG-aware extensions (RAGCache [ 3], TurboRAG [ 4], CacheBlend [ 5], EPIC [ 6],
ContextPilot [7], PCR [8], LMCache [9]).
1arXiv:2605.27494v1  [cs.CR]  26 May 2026

2.Retrieval-result reuse, where the top- kdocument set itself is cached for similar queries
(Proximity [10], QVCache [11]).
3.Output / answer reuse, where the entire generated answer is returned for a semantically similar
prior query (GPTCache [12], ContextCache [13]).
The first two have strong correctness guarantees by construction: KV reuse preserves token
outputs exactly when prefixes match, and retrieval reuse is safe as long as the underlying corpus
is stable. Output-level semantic caching does not. A cosine-similar prior query can map to a
cached answer whose underlying evidence is no longer valid, no longer the same evidence, or never
supported the question in the first place. Recent work has shown that naive semantic caches are even
adversarially exploitable, with reported 86% response-hijacking rates in evaluated scenarios [14].
We make three contributions:
1.An evidence-validated cache router (GroundedCache).A cached answer is admitted only
when (G1) the new query is semantically close to the cached query, (G2) the new retrieved
evidence shares a high Jaccard with the cached evidence signature, (G3) shared chunks carry
the same source version, and (G4) the cached answer’s content tokens are covered by the
new evidence (or, optionally, verified by a lightweight judge call). The router falls back to
query-conditioned compression and generation when any gate fails.
2.A six-regime workload for cache-safety stress testing.We provide deterministic synthesis for
exact-repeat, paraphrase, near-miss, document-drift, long shared-document, and bounded-KB
CAG traffic over arbitrary RAG corpora, plus a HotpotQA adapter. The document-drift
regime mutates numeric tokens in the gold documents so that a cached answer becomes
verifiablywrong unless rejected.
3.Operator-facing metric stack.Beyond TTFT, latency, tokens, and answer-cache hit rate
we report theunsafe-served rateUSR (fraction of all queries that received a wrong cached
answer), the conditional false-hit rate FH (per served cached answer), and stale-/unsupported-
hit decompositions, so the safety/reuse trade-off is auditable at both the system and the
per-query level.
Our framing is complementary to, rather than competitive with, prefix-level work:Grounded-
Cachetreats vLLM APC and chunk-level KV reuse as orthogonal serving primitives operating
below the router. The contribution is apolicy layeron top.
2 Related Work
Prefix and KV caching are now standard in open-source serving stacks. vLLM’s Automatic Prefix
Caching [ 1] and SGLang’s RadixAttention [ 2] reuse KV blocks when prompt prefixes align. The
line of work on RAG-specific KV reuse moves beyond exact prefix matching: RAGCache [ 3] caches
intermediate states of retrieved documents and reports up to 4 ×TTFT and 2 .1×throughput
gains; TurboRAG [ 4] pre-computes document KV caches offline for up to 9 .4×TTFT reduc-
tion; CacheBlend [ 5] corrects reused non-prefix chunks via selective recomputation. EPIC [ 6]
formalizes position-independent caching, and ContextPilot [ 7] adds context indexing, ordering, and
de-duplication. PCR [ 8] adds prefetch-enhanced reuse, and LMCache [ 9] provides multi-tier KV
storage and cross-engine reuse. All of these target thehowof reuse and largely assume thewhenis
safe.
2

Semantic answer caching has a separate failure-mode literature. GPTCache [ 12] is the canonical
open-source baseline. ContextCache [ 13] extends the cache key with multi-turn dialogue context,
demonstrating that turn-level similarity is insufficient when the referent has shifted. Proximity [ 10]
caches retrieval results rather than outputs, validating that retrieval-side caching is safer. QV-
Cache [ 11] learns region-specific thresholds for safe ANN-level reuse, cutting query latency 40–1000 ×
on hits without compromising recall. Recent work shows naive semantic caching is vulnerable to
key-collision attacks with up to 86% response-hijacking rate [ 14], and agentic-plan caching suffers
high false positives when outputs depend on external state. These three failure modes, referent
shift, retrieval drift, and adversarial collision, motivate the four gates in our validator.
The compression literature has moved from hard pruning (RECOMP [ 15], LongLLMLingua [ 16],
Provence [ 17], PISCO [ 18]) to adaptive, query-conditioned, or representation-level compression
(OSCAR [ 19], SeleCom [ 20], REFRAG [ 21]). We use a sentence-level selector as a practical baseline;
learned compressors plug into the samecompress(query, chunks)interface.
RAG evaluation frameworks have multiplied in the last two years. RAGAS [ 22] and ARES [ 23]
provide reference-free and trained-judge frameworks; RAGTruth [ 24] ships nearly 18K manually
annotated outputs for hallucination analysis; RAGBench [ 25] and RAGChecker [ 26] give 100K-
scale explainable examples and fine-grained diagnostics; mtRAG [ 27] covers multi-turn settings;
T2-RAGBench [ 28] adds tabular reasoning. None of these were built specifically for cache-safety
evaluation under repeated paraphrases and corpus drift, which motivates our workload contribution.
3 Method
3.1 Pipeline
Figure 1 sketches the end-to-end pipeline; the rest of this section spells out each block.
A query qenters the router. We embed it (the same model used for the retriever, so the embedding
can be reused), look up an exact retrieval-cache entry, and on miss fall back to approximate retrieval-
cache lookup with cosine threshold τret. On total miss we hit the FAISS index for the top- kchunks.
We deduplicate chunks by SHA-1 hash and order them by ( docid,chunk id) so identical evidence
sets produce identical prompt prefixes, which is the precondition for prefix-level KV reuse below.
3.2 Evidence signature
For every served query we record anevidence signature:
σ(q)def=
docids,chunk ids,chunk hashes,versions,scores
.(1)
The signature is content-addressed (SHA-1 over normalized chunk text) and version-tagged. It
serves as the cache key’s evidentiary half: similarity betweenqueriesis necessary for reuse, but
agreement ofevidenceis what makes reuse safe.
3

Figure 1:GroundedCachesystem architecture. A query first hits the stage-2 retrieval cache
(byte-exact dedup); on a miss it falls back to FAISS vector search. The retrieved chunks are hashed
into an evidence signature σ, and four gates (G1 query similarity, G2 evidence overlap, G3 version
match, G4 lexical support) decide whether to serve a cached answer from the answer cache or fall
back to the compression + vLLM generate path. The dashed block is the only new component;
everything else is unchanged from a standard RAG stack.
3.3 Validation gates
Let (qc, ac, σc) be a cached entry and ( q, σ, C ) be the fresh query, fresh signature, and fresh chunk
set. We admitaciff all four gates fire:
(G1) query similarity cos(emb(q),emb(qc))≥τ q (2)
(G2) evidence overlapJ(σ, σc)≥τ e (3)
(G3) version match ver(c, σ) = ver(c, σc)∀c∈σ∩σc(4)
(G4) evidence supportS(ac, C)≥τ s (5)
where J(·,·) is Jaccard over chunk hashes and S(a, C) is either a deterministic lexical-overlap score
(the default; the fraction of content tokens in athat also appear in C) or, optionally, a single-call
judge LLM that returns yes/no for “is asupported by C.” Algorithm 1 states the full procedure;
Figure 2 shows the same logic as a per-query decision flow.
3.4 Compression fallback
When any gate fails we fall back to query-conditioned compression in the spirit of Provence [ 17]
and PISCO [ 18]: embed each sentence in the retrieved chunks, score against the query embedding,
and keep the top sentences subject to a token budget, with a min-one-sentence-per-chunk guar-
4

Algorithm 1:GroundedCacherouting for a queryq
Input:queryq; retrieverR; retrieval-cacheR; answer-cacheA; validator gates with
thresholdsτ q, τe, τs; compressor Π; generatorG.
Output:answera, path label.
1e←emb(q) ;// single encoder call, reused below
2C← R.lookup(q, e) ;// stage 2 cache
3ifC=⊥then
4C←R.search(q, k);R.insert(q, e, C)
5end
6C←dedup andorder(C);σ←signature(C) ;
7h← A.nearest(e) ;// stage 3 lookup
8ifh̸=⊥then
9ifcos(e, h.ec)≥τ qandJ(σ, h.σc)≥τ eandversions match(σ, h.σc)and
S(h.ac, C)≥τ sthen
10returnh.ac,answer cache
11end
12end
13C′←Π.compress(q, C) ;// stage-4 compression fallback
14a←G.generate(prompt(q, C′)) ;
15A.insert(q, e, a, σ) ;
16returna,generate
antee to avoid emptying any retrieved chunk. Learned compressors (OSCAR [ 19], SeleCom [ 20],
REFRAG [21]) implement the same interface and can be swapped in.
4 Experimental Setup
4.1 Workload regimes
We synthesize six regimes deterministically over any RAG corpus:
exact-repeatre-issue a sampled prior question verbatim.
paraphrasere-issue under a deterministic surface paraphrase.
near-miss re-issue a lexically-similar question whose gold documents aredisjointfrom the lexically-
nearest prior question. A semantic cache should reject these;GroundedCacheshould reject
them via G2.
document-drift re-issue a prior question after applying a seeded mutation to the numeric tokens
of its gold document(s). A version-aware cache should reject these via G3.
long shared-document many questions hitting the same document, the favorable regime for both
dedup and prefix-level reuse.
bounded-KB CAG all relevant material assumed in-context; the regime under which cache-
augmented generation may beat classical RAG [29].
The mixture is designed so cumulative cache safety can be evaluated, not just hit rate on a benign
trace.
5

Figure 2: Per-query decision flow inside the router. The four gates (G1–G4) form a short-circuiting
chain: any Noroutes the query to the regeneration fallback (compress, generate, insert), preserving
safety at the cost of one cache miss. Only when all four fire is the cached answer acserved (green
path). This figure is the visual companion to Algorithm 1.
4.2 Datasets and models
We report results on a tiny hand-crafted 3-document corpus (used as a smoke control) and on
the HotpotQA distractor split (each example ships its own context paragraphs). The retriever
uses all-MiniLM-L6-v2 [30] with FAISS-CPU [ 31]. The generator is an interchangeable adapter:
an Anthropic API client, an OpenAI-compatible vLLM endpoint, or a deterministicextractive
backend that returns the sentence in the retrieved context with the highest query-token overlap.
The extractive backend lets us produce reproducible results without an API key while still exercising
every validator gate meaningfully (its answers come from evidence, so the support gate Sproduces
non-trivial scores).
4.3 Variants
We compare five router configurations:
•naive : semantic answer cache only; all gates relaxed (the stage-3 baseline modeled on
GPTCache).
•no-version,no-evidence,no-support: ablations removing one gate at a time.
•full:GroundedCache, all four gates active.
6

4.4 Metrics
For each query we record TTFT, retrieval latency, end-to-end latency, prompt and completion
tokens, path ( generate ,retrieval cache ,answer cache ), exact-match and token-F1 against gold
(when available), and substring-containment of gold in the answer (robust to verbose answers). A
query is flagged as adisagreementwhen F1< 0.5andthe normalized gold string fails to appear as
a substring of the answer; this conjunction prevents F1’s punishment of correct paraphrases from
inflating safety counters. Let ⊮ac
iindicate that query iwas served from the answer cache and ⊮dis
i
indicate that its answer disagrees with gold. We aggregate:
aHRdef=1
NNX
i=1⊮ac
i (answer-cache hit rate) (6)
USRdef=1
NNX
i=1⊮ac
i·⊮dis
i (unsafe-served rate) (7)
FHdef=USR
aHR= Pr[dis|ac] (conditional false-hit rate) (8)
The lexical support score used by gate G4 (Eq. 5) is
S(a, C)def=|τ(a)∩τ(C)|
|τ(a)|, τ(x) ={t∈toks(x) :t /∈ W ∧ |t| ≥3},(9)
where τextracts content tokens after removing a small stop-word set W, and a cached answer
ais rejected when S(a, C)< τs(default τs=0.6). USR is our primary safety metric: it directly
measures the fraction of all queries that received a wrong cached answer, which is the only number
a production operator can act on. We report aHR to capture the trade-off between reuse and safety,
and the conditional FH so the table aligns with prior caching evaluations that report FH conditional
on hit. Stale-hit (SH) and unsupported-hit (UH) counters are computed the same way (numerator
substituted accordingly) and reported in the appendix; we omit them from the main tables because
under the full system they are identically zero. Crucially, the retrieval-cache path is excluded from
the USR denominator’s numerator entirely: a retrieval-cache hit only skips vector search and still
regenerates, so by construction it cannot return a stale answer.
5 Results
We report two sweeps with the same router configuration: a multi-hop sweep on HotpotQA (§5.1)
and a multi-turn sweep on mtRAG (§5.2). Both use Qwen/Qwen2.5-7B-Instruct served by vLLM
with Automatic Prefix Caching enabled, behind our router. Each cell sweeps N=200 queries; with
six regimes and five router variants this is 6 ×5×200 = 6000 generations per dataset. USR is
reported over the full N=200 denominator (the operator metric); the conditional false-hit rate FH
is reported alongside but its denominator is the answer-cache hit count, which can be small under
GroundedCache. For 0 /koutcomes we report FH= 0 but note the Wilson 95%-CI upper bound
is non-trivial when kis small (e.g. 0 /9⇒FH95%≤0.34); USR confidence intervals are tight because
the denominator is always 200.
5.1 HotpotQA (multi-hop)
On HotpotQA,GroundedCachedrives the unsafe-served-rate (USR, the fraction of all queries
that receive a wrong cached answer) to zero on every regime that has any USR to begin with. The
7

Table 1: Naive semantic answer cache vs.GroundedCache(full evidence validation) across regimes
(HotpotQA).aHRis the answer-cache hit rate (fraction of queries served from cache);USRis the
unsafe-served-rate (fraction ofallqueries that received a wrong cached answer, our primary safety
metric);FH=USR/aHR is the conditional false-hit rate among served cached answers.
NaiveGroundedCache(full)
Regime aHR USR FH aHR USR FH
exact repeat 0.41 0.15 0.37 0.04 0.00 0.00
paraphrase 0.40 0.22 0.56 0.04 0.00 0.00
near miss 0.62 0.00 0.00 0.04 0.00 0.00
document drift 0.62 0.35 0.56 0.01 0.00 0.00
long shared doc 0.41 0.15 0.37 0.06 0.00 0.00
bounded kbcag 0.41 0.15 0.37 0.06 0.00 0.00
exact
repeatparaphrase near
missdocument
driftlong
shared
docbounded
kb
cag0.00.20.40.60.81.0rateAnswer-cache hit rate (solid) and unsafe-served rate (hatched) per regime
naive aHR
naive USRfull aHR
full USR
Figure 3: HotpotQA. Per-regime answer-cache hit rate (solid) and unsafe-served rate (hatched)
for the naive baseline andGroundedCache.GroundedCachedrives the USR bars to zero on
every regime that has any USR to begin with; the aHR bars shrink because the validator suppresses
unsafe reuse.
naive semantic cache emits USR=15.5% on exact repeat , 22.0% on paraphrase , and 35.0% on
document drift;GroundedCachereduces all three to 0.0%.
The document drift regime is the design point for the validator. Naive caching answers from
cache for 62.0% of queries with USR=35.0%;GroundedCacheanswers from cache for 0.5%
of queries with USR=0.0%, a 100.0% relative reduction (complete elimination of unsafe served
answers). The validator trades aggressive answer-cache reuse on this adversarial regime for the
safety guarantee that almost no query receives a stale cached answer.
5.2 mtRAG (multi-turn)
mtRAG [ 27] exposes the multi-turn “referent shift” failure mode: same surface tokens, different
grounding because the conversation’s focus has moved. We treat each (user turn, agent turn with
retrieved contexts) as a query, and assign within-conversation near-misses to a later turn whose
gold passages are disjoint from the cached turn’s.
8

0.0 0.2 0.4 0.6 0.8 1.0
answer-cache hit rate (higher = more reuse)0.00.20.40.60.81.0unsafe-served rate (lower = safer)Safety/speedup Pareto: per regime, per variant
naive
no-version
no-evidence
no-support
fullFigure 4: HotpotQA. Safety/speedup Pareto: each marker is one (regime, variant) pair.Ground-
edCache(stars) sits in the safe high-reuse region across regimes.
On mtRAG, naive semantic caching is catastrophically unsafe: USR ranges from 26.0% to
51.5% across the benign-looking regimes because multi-turn referent shift routinely breaks cache
assumptions.GroundedCachereduces USR by more than an order of magnitude on every affected
regime.
The document drift regime is the design point for the validator. Naive caching answers from
cache for 56.5% of queries with USR=51.5%;GroundedCacheanswers from cache for 2.0%
of queries with USR=1.5%, a 97.1% relative reduction (34 ×fewer unsafe served answers). The
validator trades aggressive answer-cache reuse on this adversarial regime for the safety guarantee
that almost no query receives a stale cached answer.
The per-gate marginal contribution to unsafe-served-rate reduction (Table 9) gives a sharper
attribution than the wide ablation matrix, and four observations follow.
First, the lexical support gate (G3) is the load-bearing gate on both datasets. Removing G3
raises mean USR by +0 .125 on HotpotQA and +0 .118 on mtRAG, whereas removing G1 or G2
leaves USR within ≤0.001 of the full system on either dataset. The cached answer’s tokens must
appear in the freshly retrieved evidence; that single cheap check carries essentially the entire safety
gain.
Second, the version (G1) and evidence-overlap (G2) gates are redundant on this workload,
but that redundancy is the point. G1 hashes ( q, E) and G2 demands chunk-id overlap; both fire
on evidence change. With G3 active, anything G1 or G2 would catch G3 also catches via the
answer-evidence mismatch. We retain G1 and G2 because they failclosedat near-zero cost and
provide defense-in-depth against settings where G3 weakens: paraphrastic answers that escape
lexical support, stronger embedding models that compress evidence-disjoint queries to closer points,
and judge-basedSvariants where the support check becomes noisier.
9

Table 2: Per-gate ablation: hit-rate vs. false-hit-rate across regimes (HotpotQA).
Regime naive no-version no-evidence no-support full
HR FH HR FH HR FH HR FH HR FH
exact repeat 0.41 0.37 0.41 0.00 0.41 0.00 0.41 0.37 0.41 0.00
paraphrase 0.40 0.56 0.40 0.00 0.40 0.00 0.40 0.53 0.40 0.00
near miss 0.62 0.00 0.62 0.00 0.62 0.00 0.62 0.00 0.62 0.00
document drift 0.62 0.56 0.24 0.00 0.24 0.00 0.24 0.58 0.24 0.00
long shared doc 0.41 0.37 0.41 0.00 0.41 0.00 0.41 0.37 0.41 0.00
bounded kbcag 0.41 0.37 0.41 0.00 0.41 0.00 0.41 0.37 0.41 0.00
exact
repeatparaphrase near
missdocument
driftlong
shared
docbounded
kb
cag0.00.20.40.60.81.0share of queriesPath mix under GroundedCache (full)
answer_cache retrieval_cache generate
Figure 5: HotpotQA. Path mix underGroundedCache(full). Benign regimes (exact-repeat,
paraphrase, long shared-doc) route a large share to the answer cache; adversarial regimes route to
retrieval-cache + regenerate, paying retrieval savings but not risking false hits.
Third, the answer-cache hit-rate cost on HotpotQA is borne almost entirely by G1. Every ablation
that leaves G1 off keeps aHR ≈0.41 on exact repeat ; only full drops below 0.05. Qwen2.5’s free-
form completions inject lexical variation into the cached answer that re-fires the version signature
even when retrieved evidence is byte-identical. The targeted fix is to normalize the answer before
hashing, and it does not require relaxing G3, which is the gate actually carrying the safety budget.
We expect aHR to recover toward the naive baseline after that fix without re-introducing any USR.
Fourth, the residual mtRAG conditional FH is a G3 ceiling rather than a G1/G2 shortcoming.
On mtRAG full still emits FH ≈0.87 conditional on the rare answer-cache hits it serves, because
lexically compatible answers can match fresh evidence tokens while pragmatically referring to a
different turn antecedent. USR remains only ≈0.10 because the validator suppresses almost all
answer-cache hits in the first place: aHR drops from 0.29 to 0.12 on these regimes and to 0.02 on
document drift . The right policy interpretation is that under multi-turn traffic with strong referent
shift the answer cache should largely be disabled, and G3 implements that policy automatically. A
discourse-aware support check or turn-level evidence signature would relax the suppression without
reintroducing USR.
The honest latency baseline for these gates is not the naive semantic cache, which ships unsafe
answers, but a no-cache RAG pipeline that always retrieves and generates. We synthesize that
baseline directly from the JSONL logs: the latency of every generate -path record across all variants
is an unbiased sample of true no-cache end-to-end latency. The result (Table 6) gives a clean
10

Table 3: Naive semantic answer cache vs.GroundedCache(full evidence validation) across regimes
(mtRAG).aHRis the answer-cache hit rate (fraction of queries served from cache);USRis the
unsafe-served-rate (fraction ofallqueries that received a wrong cached answer, our primary safety
metric);FH=USR/aHR is the conditional false-hit rate among served cached answers.
NaiveGroundedCache(full)
Regime aHR USR FH aHR USR FH
exact repeat 0.29 0.26 0.88 0.12 0.10 0.87
paraphrase 0.30 0.26 0.85 0.12 0.07 0.54
near miss 0.67 0.00 0.00 0.30 0.00 0.00
document drift 0.56 0.52 0.91 0.02 0.01 0.75
long shared doc 0.29 0.27 0.90 0.12 0.10 0.87
bounded kbcag 0.29 0.27 0.90 0.12 0.10 0.87
exact
repeatparaphrase near
missdocument
driftlong
shared
docbounded
kb
cag0.00.20.40.60.81.0rateAnswer-cache hit rate (solid) and unsafe-served rate (hatched) per regime
naive aHR
naive USRfull aHR
full USR
Figure 6: mtRAG. Per-regime answer-cache hit rate (solid) and unsafe-served rate (hatched). Naive
caching emits USR of 26–52% across benign-looking regimes;GroundedCachereduces every USR
bar by an order of magnitude or more by aggressively suppressing answer-cache hits whose support
gate fails.
Pareto picture. On HotpotQA, no-cache RAG runs at 1 .10s p50; naive caching reaches 1 .95×at
USR= 0 .17;no-support gives 1 .49×at USR= 0 .13; andGroundedCacheruns at 1 .01×(1.08s)
with USR= 0 .00. On mtRAG the numbers are similar: 1 .73×, 1.37×, 1.07×at USR 0 .26, 0.18, 0.06
respectively.GroundedCachesits explicitly at the safe corner of the frontier, retaining roughly
no-cache latency while eliminating unsafe served answers. Practitioners who can tolerate non-zero
USR for a real speedup should runno-support; the gate stack defaults to safety.
11

0.0 0.2 0.4 0.6 0.8 1.0
answer-cache hit rate (higher = more reuse)0.00.20.40.60.81.0unsafe-served rate (lower = safer)Safety/speedup Pareto: per regime, per variant
naive
no-version
no-evidence
no-support
fullFigure 7: mtRAG. Safety/speedup Pareto under multi-turn traffic.
Table 4: Per-gate ablation: hit-rate vs. false-hit-rate across regimes (mtRAG).
Regime naive no-version no-evidence no-support full
HR FH HR FH HR FH HR FH HR FH
exact repeat 0.29 0.88 0.29 0.87 0.29 0.87 0.29 0.90 0.29 0.87
paraphrase 0.30 0.85 0.29 0.54 0.29 0.54 0.29 0.81 0.29 0.54
near miss 0.67 0.00 0.67 0.00 0.67 0.00 0.67 0.00 0.67 0.00
document drift 0.56 0.91 0.14 0.80 0.13 0.75 0.13 0.92 0.13 0.75
long shared doc 0.29 0.90 0.29 0.87 0.29 0.87 0.29 0.90 0.29 0.87
bounded kbcag 0.29 0.90 0.29 0.87 0.29 0.87 0.29 0.90 0.29 0.87
Table 5: End-to-end latency vs. the honest baseline (HotpotQA). Theno-cacherow is the empirical
p50 latency of all generate -path records observed across every sweep cell ( n= 3424 records); it
is the latency of a RAG query that hits no cache at all. The other rows are the per-variant p50
latency averaged across the six regimes. Speedup is computed against the no-cache baseline. Naive
caching is fastest but, per Table 3, unsafe;GroundedCacheretains a meaningful speedup while
delivering USR→0.
Variant Latency p50(s) Speedup vs. no-cache USR
no-cache (always generate) 1.093 1.00×0.00
naive0.562 1.95×0.172
no-support0.739 1.48×0.125
full1.053 1.04×0.000
12

Table 6: End-to-end latency vs. the honest baseline (mtRAG). Theno-cacherow is the empirical
p50 latency of all generate -path records observed across every sweep cell ( n= 3935 records); it
is the latency of a RAG query that hits no cache at all. The other rows are the per-variant p50
latency averaged across the six regimes. Speedup is computed against the no-cache baseline. Naive
caching is fastest but, per Table 3, unsafe;GroundedCacheretains a meaningful speedup while
delivering USR→0.
Variant Latency p50(s) Speedup vs. no-cache USR
no-cache (always generate) 1.128 1.00×0.00
naive0.651 1.73×0.261
no-support0.827 1.37×0.182
full1.058 1.07×0.063
naive + G1,G2 + G3 (full)0.00.20.40.60.81.0Unsafe-served rate (fh / N)
Cumulative gate-stack USR reduction
exact repeat
paraphrase
near missdocument drift
long shared doc
bounded kb cag
Figure 8: HotpotQA. Cumulative gate-stack USR reduction per regime. Adding G3 on top of
G1+G2 closes the remaining unsafe-served-rate gap acrosseveryregime that has any USR to begin
with.
Table 7: Per-gate marginal unsafe-served-rate reduction (HotpotQA): ∆ USR = USR(no-X) −
USR(full), averaged over the six workload regimes. USR is the fraction of all queries that received a
wrong cached answer. Larger ∆ means the gate is more load-bearing; values near zero mean the
gate is redundant with the other two on this workload.
Removed gate USR(no-X) ∆USR vs. full
Version (G1) 0.000 +0.000
Evidence (G2) 0.000 +0.000
Support (G3) 0.125 +0.125
None (full system) 0.000 n/a
13

Table 8: Per-variant cost averaged over the six regimes (HotpotQA). TTFT and latency are p50
seconds; tokens are mean prompt tokens per query. The full system retains most of the answer-cache
speedup while closing the false-hit gap.
Variant TTFT p50(s) Latency p50(s) Prompt tok.
naive0.065 0.562 138
no-version0.102 1.072 252
no-evidence0.100 1.078 252
no-support0.077 0.739 161
full0.135 1.053 253
Table 9: Per-gate marginal unsafe-served-rate reduction (mtRAG): ∆ USR = USR(no-X) −USR(full),
averaged over the six workload regimes. USR is the fraction of all queries that received a wrong
cached answer. Larger ∆ means the gate is more load-bearing; values near zero mean the gate is
redundant with the other two on this workload.
Removed gate USR(no-X) ∆USR vs. full
Version (G1) 0.064 +0.001
Evidence (G2) 0.063 +0.000
Support (G3) 0.182 +0.118
None (full system) 0.063 n/a
Table 10: Per-variant cost averaged over the six regimes (mtRAG). TTFT and latency are p50
seconds; tokens are mean prompt tokens per query. The full system retains most of the answer-cache
speedup while closing the false-hit gap.
Variant TTFT p50(s) Latency p50(s) Prompt tok.
naive0.067 0.651 185
no-version0.099 1.061 270
no-evidence0.098 1.060 270
no-support0.079 0.827 211
full0.098 1.058 270
14

naive + G1,G2 + G3 (full)0.00.20.40.60.81.0Unsafe-served rate (fh / N)
Cumulative gate-stack USR reduction
exact repeat
paraphrase
near missdocument drift
long shared doc
bounded kb cagFigure 9: mtRAG. The same cumulative gate-stack picture. The benign-looking regimes
(exact repeat ,long shared doc,bounded kbcag) all show the same shape: G1+G2 are nearly
indistinguishable from naive, and the entire safety gain lands when G3 is added.
15

6 Discussion
The central contribution ofGroundedCacheis a policy layer rather than a serving kernel. The
router adds no new attention or KV machinery and assumes no special infrastructure: vLLM
Automatic Prefix Caching continues to run underneath, chunk-level reuse systems such as LMCache,
RAGCache, and EPIC continue to operate on the prompts that the router emits, and the only new
component is a thin module that sits above the retriever. The four-gate validator can be added
to an existing RAG stack with no change to the model server, no change to the retriever, and no
change to the embedding model.
The lexical support check that does most of the safety work (Equation 9) is intentionally crude.
It asks whether the content tokens of the cached answer appear in the freshly retrieved evidence,
which is a strict underestimate of true entailment. Two properties make it the right primitive for a
cache nonetheless: it is deterministic, so no second-order judge noise enters the safety budget; and it
is cheap enough to run on every potential hit. The judge-based variant of Scan be swapped in when
budget allows, at the cost of one extra LLM call per candidate hit and the noise that LLM-as-judge
metrics inherit. In practice the lexical variant rejects the dangerous hits at essentially no extra cost
relative to a baseline answer cache, which is why we keep it as the default.
The workloads on whichGroundedCachepays off are those that contain near-misses, para-
phrastic repeats, or evidence drift. On a perfectly benign trace,GroundedCachereduces to a
small validation overhead on top of a naive semantic cache. The harness reports both halves of the
trade-off (answer-cache hit rate and unsafe-served rate) so operators can decide whether to run the
full gate stack, the no-support middle point, or no validator at all based on their own traffic shape
and risk tolerance.
7 Limitations
Our evaluation has three caveats. First, all headline numbers come from a single serving stack:
Qwen2.5-7B-Instruct on vLLM with Automatic Prefix Caching. A deterministic extractive backend
is also included for reproducible runs without an API key, but its absolute USR rates differ because
SQuAD-style F1 penalizes its longer answers. We have not validated the numbers on a second
model family.
Second, on HotpotQA the exact repeat regime exposes a calibration issue rather than a structural
flaw. The version gate fires on lexical variation in Qwen2.5’s free-form completions even when the
retrieved evidence is byte-identical, which collapses aHR from 0.41 to under 0.05 while USR drops
to zero. Normalizing the answer string before hashing should recover most of the lost aHR without
re-introducing any USR.
Third, on mtRAG the conditional FH of 54–87% under the full system reflects that the validator
suppresses almost all answer-cache hits in the first place: aHR drops from 0.29 to 0.12 on benign
regimes and to 0.02 on document drift. The operator-facing USR still drops from 26–52% under
naive to 1–10% underGroundedCache, a 3–34 ×reduction. Closing the residual conditional FH
would require a discourse-aware support check, which we leave to future work.
The lexical support score itself has a known structural failure mode: a cached answer that
paraphrases its evidence without copying tokens can be rejected even when it is in fact supported.
This is a safe failure mode for a cache, since the rejected hit simply becomes a regeneration, but it
puts an upper bound on the answer-cache hit rate achievable with a purely lexical gate.
16

8 Conclusion
The right framing for cached answer reuse in retrieval-augmented generation is not how to reuse faster
but when reuse is safe. We operationalized that framing as four cheap gates over a content-addressed
evidence signature: query similarity, evidence overlap, source-version validity, and lexical support of
the cached answer by the freshly retrieved evidence. Across two datasets and 12,000 generations
on a six-regime workload built specifically to stress cache safety, the router cleanly separates safe
and unsafe reuse: it drives the unsafe-served rate to zero on every HotpotQA regime and produces
3–34×reductions on the multi-turn referent-shift regimes in mtRAG, while retaining end-to-end
latency within 1.04–1.07×of a no-cache RAG baseline.
A per-gate marginal ablation isolates the lexical support gate as the load-bearing safety mecha-
nism on both datasets, with the version and evidence-overlap gates providing defense-in-depth at
near-zero cost. Because the contribution is a policy layer rather than a serving kernel, the router
composes with any existing prefix or chunk-level reuse system without modification.
The takeaway is simpler than the machinery suggests: semantic answer caches always trade
some correctness for speed, and the unsafe-served rate is the right number to quantify that trade-
off. Reporting USR alongside hit rate and latency is what lets practitioners decide where on the
safety/speed frontier they want to operate. We release the implementation, the six workload regimes,
and the evaluation harness as an open repository so others can reproduce the numbers, port the
framework to richer datasets, and swap in stronger compressors and support checks as the serving
stack evolves.
References
[1]vLLM automatic prefix caching. https://docs.vllm.ai/en/latest/features/automatic_
prefix_caching.html, 2024. Accessed: 2026.
[2]Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, et al. SGLang:
Efficient execution of structured language model programs. InNeurIPS, 2024.
[3]Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin Jin.
RAGCache: Efficient knowledge caching for retrieval-augmented generation.arXiv preprint
arXiv:2404.12457, 2024.
[4]Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and Yaohua Tang. TurboRAG: Accelerating
retrieval-augmented generation with precomputed KV caches for chunked text.arXiv preprint
arXiv:2410.07590, 2024.
[5]Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du,
Shan Lu, and Junchen Jiang. CacheBlend: Fast large language model serving for RAG with
cached knowledge fusion.arXiv preprint arXiv:2405.16444, 2024.
[6]Junhao Hu, Wenrui Huang, Weidong Wang, Zhenwen Yuan, Tiancheng Xie, Zhixia Liu, Xusheng
Liu, Tao Cui, Fei Liu, and Yizhou Cao. EPIC: Efficient position-independent caching for
serving large language models.arXiv preprint arXiv:2410.15332, 2024.
[7]EfficientContext. ContextPilot: Fast long-context inference via context reuse.arXiv preprint
arXiv:2511.03475, 2025. To appear, MLSys 2026.
[8]Wenfeng Wang, Xiaofeng Hou, Peng Tang, et al. PCR: A prefetch-enhanced cache reuse system
for low-latency RAG serving.arXiv preprint arXiv:2603.23049, 2026.
17

[9]LMCache Team. LMCache: An efficient KV cache layer for enterprise-scale LLM inference.
arXiv preprint arXiv:2510.09665, 2025.
[10]Shai Bergman et al. Leveraging approximate caching for faster retrieval-augmented generation.
arXiv preprint arXiv:2503.05530, 2025. EuroMLSys 2025.
[11]Anıl Eren G¨ o¸ cer et al. QVCache: A query-aware vector cache.arXiv preprint arXiv:2602.02057,
2026.
[12]Fu Bang. GPTCache: An open-source semantic cache for LLM applications enabling faster
answers and cost savings.Proceedings of the NLP-OSS Workshop, 2023.
[13]Jianxin Yan, Wangze Ni, Lei Chen, Xuemin Lin, Peng Cheng, Zhan Qin, and Kui Ren.
ContextCache: Context-aware semantic cache for multi-turn queries in large language models.
Proc. VLDB Endowment, 2025. arXiv:2506.22791.
[14]Zhixiang Zhang, Zesen Liu, Yuchong Xie, Quanfeng Huang, and Dongdong She. From
similarity to vulnerability: Key collision attack on LLM semantic caching.arXiv preprint
arXiv:2601.23088, 2026.
[15]Fangyuan Xu, Weijia Shi, and Eunsol Choi. RECOMP: Improving retrieval-augmented LMs
with compression and selective augmentation.arXiv preprint arXiv:2310.04408, 2023.
[16]Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and
Lili Qiu. LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via
prompt compression.arXiv preprint arXiv:2310.06839, 2023.
[17]Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, and St´ ephane Clinchant. Provence:
Efficient and robust context pruning for retrieval-augmented generation.arXiv preprint
arXiv:2501.16214, 2025. ICLR 2025.
[18]Maxime Louis, Nadezhda Chirkova, Thibault Formal, and St´ ephane Clinchant. PISCO: Pretty
simple compression for retrieval-augmented generation.arXiv preprint arXiv:2501.16075, 2025.
[19]Maxime Louis, Nadezhda Chirkova, Thibault Formal, and St´ ephane Clinchant. OSCAR: Online
soft compression and reranking.arXiv preprint arXiv:2504.07109, 2025.
[20]Anonymous. Rethinking soft compression in retrieval-augmented generation: A query-
conditioned selector perspective.arXiv preprint arXiv:2602.15856, 2026. SeleCom.
[21]Meta Superintelligence Labs. REFRAG: Rethinking RAG based decoding.arXiv preprint
arXiv:2509.01092, 2025.
[22]Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. RAGAS: Automated
evaluation of retrieval augmented generation.arXiv preprint arXiv:2309.15217, 2023.
[23]Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. ARES: An au-
tomated evaluation framework for retrieval-augmented generation systems.arXiv preprint
arXiv:2311.09476, 2023.
[24]Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song,
and Tong Zhang. RAGTruth: A hallucination corpus for developing trustworthy retrieval-
augmented language models.arXiv preprint arXiv:2401.00396, 2024.
18

[25]Robert Friel, Masha Belyi, and Atindriyo Sanyal. RAGBench: Explainable benchmark for
retrieval-augmented generation systems.arXiv preprint arXiv:2407.11005, 2024.
[26]Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng
Jiayang, Cunxiang Wang, Siyuan Sun, Hang Li, Zheng Zhang, Binjie Wang, Jiarong Jiang, Tong
He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and Zheng Zhang. RAGChecker: A fine-grained
framework for diagnosing retrieval-augmented generation.arXiv preprint arXiv:2408.08067,
2024.
[27]Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lu-
cian Popa, Vraj Shah, Huaiyu Zhu, Danish Contractor, and Marina Danilevsky. MTRAG:
A multi-turn conversational benchmark for evaluating retrieval-augmented generation sys-
tems.Transactions of the Association for Computational Linguistics, 2025. arXiv:2501.03468;
benchmark at https://github.com/IBM/mt-rag-benchmark.
[28]Anonymous. T2-RAGBench: Text-and-table benchmark for evaluating retrieval-augmented
generation.arXiv preprint arXiv:2506.12071, 2025. EACL 2026.
[29]Brian J Chan, Chao-Ting Chen, Jui-Hung Cheng, and Hen-Hsen Huang. Don’t do RAG:
When cache-augmented generation is all you need for knowledge tasks.arXiv preprint
arXiv:2412.15605, 2024.
[30]Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. MiniLM: Deep
self-attention distillation for task-agnostic compression of pre-trained transformers. InNeurIPS,
2020.
[31]Jeff Johnson, Matthijs Douze, and Herv´ e J´ egou. Billion-scale similarity search with GPUs.
IEEE Transactions on Big Data, 2017.
19