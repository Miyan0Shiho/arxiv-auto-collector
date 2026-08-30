# WARP: Wasserstein-Aligned RAG for Population Opinions

**Authors**: Aman Singh Thakur, Aditya Agrawal, Alwarappan Nakkiran, Alex Karlsson

**Published**: 2026-08-24 06:41:23

**PDF URL**: [https://arxiv.org/pdf/2608.22859v1](https://arxiv.org/pdf/2608.22859v1)

## Abstract
RAG systems are increasingly used to summarize what large collections of documents say. A user asks "What do people think about X?" and receives an answer that reads as consensus. But standard top-k retrieval ranks documents by query similarity, not by how faithfully they represent the population, so minority views quietly disappear. Existing fixes fall short. Diversity re-rankers like MMR and DPP spread retrieved documents apart, but with no target distribution to aim for. Calibration methods based on KL or JS divergence do target one, yet treat opinion bins as unordered: confusing strong positive with strong negative costs no more than an adjacent-bin miss.
  We introduce WARP, a family of post-retrieval algorithms that calibrate retrieved evidence to the population's opinion distribution. WARP first recovers underrepresented opinions that cosine ranking may bury, then uses Wasserstein-1 distance to select documents whose sentiment-intensity distribution matches the population target, capturing the ordinal structure ignored by KL and JS divergence. We develop three variants for dense, sparse, and variable candidate pools, trading off calibration quality and speed. Across three review domains spanning 35K documents, 156 queries, and 26 entities, WARP's domain-matched variants reduce distributional error by at least 43% with sub-second latency. These gains carry through to generation: a five-judge LLM panel prefers WARP-generated answers in 86% of decided comparisons at k <= 5.

## Full Text


<!-- PDF content starts -->

WARP: Wasserstein-Aligned RAG for Population Opinions
Aman Singh Thakur*Aditya Agrawal Alwarappan Nakkiran Alex Karlsson
Amazon.com
Abstract
RAG systems are increasingly used to summa-
rize what large collections of documents say.
A user asks “What do people think about X?”
and receives an answer that reads as consensus.
But standard top- kretrieval ranks documents
by query similarity, not by how faithfully they
represent the population, so minority views qui-
etly disappear. Existing fixes fall short. Di-
versity re-rankers like MMR and DPP spread
retrieved documents apart, but with no target
distribution to aim for. Calibration methods
based on KL or JS divergence do target one,
yet treat opinion bins as unordered: confusing
strong positive with strong negative costs no
more than an adjacent-bin miss.
We introduce WARP, a family of post-retrieval
algorithms that calibrate retrieved evidence to
the population’s opinion distribution. WARP
first recovers underrepresented opinions that co-
sine ranking may bury, then uses Wasserstein-1
distance to select documents whose sentiment-
intensity distribution matches the population
target, capturing the ordinal structure ignored
by KL and JS divergence. We develop three
variants for dense, sparse, and variable can-
didate pools, trading off calibration quality
and speed. Across three review domains span-
ning 35K documents, 156 queries, and 26 enti-
ties, WARP’s domain-matched variants reduce
distributional error by at least 43% with sub-
second latency. These gains carry through to
generation: a five-judge LLM panel prefers
WARP-generated answers in 86% of decided
comparisons atk≤5.
1 Introduction
Users used to browse reviews themselves - click-
ing through results, weighing contradictions, build-
ing a mental picture. Noisy, but transparent: you
could see disagreement. RAG systems (Lewis et al.,
*Correspondence:amanzing@amazon.com2020) replace this with a single synthesized answer,
but a new failure mode emerges. Standard top- kre-
trieval selects evidence by query similarity, without
factoring spread of opinions in the corpus. Take a
hotel where 60% of reviewers are satisfied and 40%
complain about noise. Retrieval over-represents the
majority because similarity scoring favors the dom-
inant cluster. Every claim in the resulting summary
traces to a real review, yet the user never learns
that two in five guests had a bad experience. This
leads to distributional distortion. A synthesized
response from a skewed sample would read factual
but would not present the whole picture.
For opinion queries, uncertainty is aleatoric: it
reflects genuine disagreement. A faithful system
must preserve the distribution without collapsing
it. Agrawal et al. (2026) formalize this distinction;
Nayeem and Rafiei (2025) build a scalable opinion
summarization pipeline but do not optimize the
retriever for distributional fidelity.What’s missing
is the engineering: in a production stack with
sub-second latency budgets, how do you select
a small evidence set whose opinion distribution
actually matches the population?
A standard cosine retriever maximizes query
similarity, producing relevant but homogeneous
evidence - four glowing reviews when two would
suffice. Diversity re-rankers (MMR (Carbonell and
Goldstein, 1998), DPP (Kulesza and Taskar, 2012))
push back by maximizing pairwise spread, which
helps until each sentiment bin has one representa-
tive; past that point they quietly revert to relevance
ordering. KL and JS calibration (Steck, 2018; Dang
and Croft, 2012) come closer by explicitly optimiz-
ing against a target distribution. But they remain
categorical. Confusing “positive” with “negative”
costs the same as confusing ”positive” and ”neu-
tral”. Opinions are ordinal and we need a metric
which supports this.
Wasserstein-1 respects this ordering. Its closed-
form CDF computation ( O(m) formsentiment
arXiv:2608.22859v1  [cs.IR]  24 Aug 2026

Figure 1: WARP pipeline.Stage 1: semantic retrieval returns a Top- Ncandidate pool.Stage 1.5: deficit-aware
pool expansion recovers under-represented sentiment poles via re-retrieval (Entity-GatedorAdaptive Expansion;
§3.3; self-bypasses when the pool is already balanced).Stage 2: W1re-ranking selects the final kdocuments whose
empirical opinion distribution matches the population target Ppop(W1Minimizerfor dense pools, W1-MMRfor
sparse pools, orWassRank OTas a tuning-free fallback; §3.4). Pool density drives the variant choice (decision
matrix in Table 3).
bins) makes per-candidate evaluation cheap enough
for runtime re-ranking without offline precomputa-
tion. We built WARP (Figure 1) around this in two
stages. First, a deficit-aware pool expansion pass
compares the retrieved distribution against the pop-
ulation target, identifies under-represented poles,
and issues targeted re-retrievals only where gaps
exist. Then a greedy re-ranker selects the final evi-
dence set by minimizing W1distance to Ppop. The
pipeline instantiates in three variants (Section 3.4):
the Minimizer greedily selects from entity-matched
candidates - effective for dense pools; for sparser
conditions, W1-MMR blends relevance with cal-
ibration, and WassRank OT solves the full slot-
assignment as rectangular optimal transport requir-
ing no per-domain tuning.
Our contributions:
1.Pool expansion that recovers what cosine
retrieval buries.Entity-gated and adaptive
re-retrieval recover under-represented opin-
ions with near-complete entity coverage. Self-
bypasses at zero cost when already balanced.
2.W 1re-rankers ensure faithful population
representation.A greedy minimizer that
matches opinion distribution for entity dense
domains and the hybrid variants trade opinion
diversity for relevance on sparse domains.3.Deployment characterization across 3 do-
mains.Pool density determines algorithm
choice. All variants add under 330 ms re-
ranking latency (p99) and generalize across
labeling methods and index types.
2 Related Work
Retrieval diversity.MMR (Carbonell and Gold-
stein, 1998) penalizes redundancy via pairwise dis-
similarity; DPP (Kulesza and Taskar, 2012) maxi-
mizes determinantal spread; xQuAD (Santos et al.,
2010) diversifies over query intents. More recent
variants-BQP (Lu and Sidiropoulos, 2026), Ada-
GReS (Peng et al., 2025), MUSS (Nguyen and Kan,
2026)-refine these ideas in various ways (see Wu
et al. (2024) for a survey). However, diversity is the
wrong objective here as the signal runs out, once
every sentiment bin has one representative (5–7
selections).
Optimal transport (OT) in IR.Wasserstein dis-
tances (Peyré and Cuturi, 2019) have appeared in
IR before-Word Mover’s Distance (Kusner et al.,
2015) for document similarity, WassRank (Yu et al.,
2019) as a listwise training loss, OTExtSum (Tang
et al., 2022) for extractive summarization, Wasser-
stein coresets (Claici et al., 2018) for data summa-
rization. However, all these function at training

1 5 10 15 2005101520W1 distance
Seller Forums (Sparse)
1 5 10 15 20
Yelp Hotels (Dense)
1 5 10 15 20
OpinRank Cars (Dense)
Top kTop-k relevant (baseline) W₁ Minimizer WassRank OT W₁-MMRk* (optimal)Figure 2: W1re-rankers (no re-retrieval) reduce distributional error at least 43% vs. Top- k(gray dashed) across
three domains — 43% Seller Forums ( W1-MMR), 79% Yelp ( W1Minimizer), 69% OpinRank ( W1Minimizer);
see Table 1. Curves converge at k∗=3–5documents (black dot; Appendix C.3) as most entities’ opinion mass
concentrates in 2–3 of 7 bins.N=200.
time, over generic content. None use Wasserstein
as aruntimeselection objective conditioned on an
opinion distribution.
Opinion-aware retrieval.Agrawal et al. (2026)
audit 30+ RAG benchmarks and find a strik-
ing gap: only one addresses opinion synthesis.
They formalize the epistemic/aleatoric distinction
- the posterior converges to the population dis-
tribution Ppop(θ)for opinion queries - and pro-
pose a three-term objective combining coverage
(W2), fidelity, and demographic fairness. How-
ever, their work is limited to 2 domains without
practical retrieval strategies. Nayeem and Rafiei
(2025) solve a different problem entirely: gen-
erating opinion highlights from thousands of re-
views via retrieve-then-synthesize with AOS-triplet
verification. Their retriever is standard seman-
tic search without targeting distributional fidelity.
We instantiate the Agrawal et al. (2026) cover-
age objective using W1instead of W2- its closed-
form CDF difference enables O(m) per-candidate
evaluation at runtime - and test across three do-
mains.
Calibrated selection and diversity methods.
Calibrated recommendation via KL minimization
(Steck, 2018) and proportional slot allocation (PM-
2; Dang and Croft 2012) do address distributional
fidelity across facets - but assume categorical la-
bels. Fairness-aware ranking (Singh and Joachims,
2018; Zehlike et al., 2017; Oesterling et al., 2024)
enforces demographic constraints without model-
ing ordinal opinion at all. On the sentiment side,Aktolga and Allan (2013) applies bias modes to
retrieval, aspect-level summarization (Angelidis
and Lapata, 2018; Jiang et al., 2023) diversifies
across topical facets. All of these methods use
categorical penalty structures that do not exploit
ordinal distance between opinion bins. WARP ad-
dresses this gap: an ordinal W1ground cost applied
at query time to opinion-distribution matching in
RAG, drawing on the calibrated-selection tradi-
tion (Steck, 2018) and OT-in-ranking (Yu et al.,
2019) while adapting both to a training-free run-
time re-ranker across three review domains.
3 Methodology
Below we describe the experimental setup (Sec-
tions 3.1–3.2), pool expansion (Section 3.3), and
W1re-ranking (Section 3.4) for stress-testing
WARP.
3.1 Datasets
3 review corpora give us the spread we need across
different domains:
•Amazon Seller Forums1(∼8K posts). Re-
trieved via Amazon Bedrock Knowledge
Bases (Amazon Web Services, 2024b) with
hybrid search.
•Yelp Open Dataset(Yelp, 2024) ( ∼14K hotel
reviews). Retrieved via local FAISS (Johnson
1https://sellercentral.amazon.com/sel
ler-forums

et al., 2019) + MiniLM-L6 (Wang et al., 2020;
Reimers and Gurevych, 2019).
•OpinRank(Ganesan and Zhai, 2012) ( ∼13K
automotive reviews). Retrieved via local
FAISS + MiniLM-L6.
Entity selection follows Agrawal et al. (2026):
LLM-based entity extraction, diversity filtering
(Shannon entropy H≥0.6 , minority sentiment
≥10%), then ranking by entropy.
We use six questions per entity (2 breadth, 2
polar, 2 segment; templates in Appendix A.2)
which gives us 156 total queries, across all do-
mains. Entity and sentiment-intensity labels come
from a single LLM pass per document (prompt
in Appendix A.3.1), mapped to a 7-bin ordinal
scaleS={−30,−20,−10,0,+10,+20,+30}
(Table 4, Appendix A.1).
Full dataset statistics appear in Table 5 (Ap-
pendix A.1). We also ablate with V ADER (Hutto
and Gilbert, 2014) lexicon scores on OpinRank
to confirm labeling-method independence (Ap-
pendix F.1).
Defaults:N=200candidate pool,k=20output.
3.2 Baselines
Five baselines. Top- kis plain cosine-similarity
retrieval. MMR (Carbonell and Goldstein,
1998) penalizes semantic redundancy ( λ=0.5 ).
DPP (Kulesza and Taskar, 2012) maximizes de-
terminantal spread over 1D sentiment-intensity fea-
tures. OpinionMMR penalizes opinion-bin dis-
tance rather than embedding distance. KL (JS)
Minimizer implements Steck (2018)’s calibrated-
recommendation objective with Jensen–Shannon
divergence - identical to our W1Minimizer but
with an order-agnostic metric (Appendix C.1).
Metrics.Distributional fidelity: Wasserstein-1
distance W1(lower is better). Entity relevance:
Entity Match rate (EM%, hereafter EM; fraction
of selected documents matching the queried entity,
higher is better). We also report re-ranking latency
in milliseconds. For generation evaluation, Cohen’s
κ(Cohen, 1960) measures inter-judge alignment.
3.3 Pool Expansion via Re-Retrieval
For entity sparse domains, the relevance-based re-
trieval step is expected to include a % of documents
which do not discuss the queried entity - due to
near-misses leaked in from other entities sharingAlgorithm 1W 1Minimizer
Require:PoolC,entitye,targetP pop,output sizek
1:C e← {d∈C:entity(d) =e}
2:S← ∅
3:while|S|< kdo
4:d∗←arg min d∈C e\SW1(EmpDist(S∪
{d}), P pop)
5:ties broken by relevance score↓
6:S←S∪ {d∗}
7:end while
8:returnS
the same index. We tackle this with two expansion
strategies (full algorithms in Appendix B):
Entity-Gated Re-retrieval.Pass 1 pulls the stan-
dard top- Nby relevance. Pass 2 then digs into
the tail of the ranked list, fishing out documents
that match the queried entity but fell below the
cutoff, ranking them by opinion extremity (abso-
lute sentiment-intensity score - strongly negative
reviews surface before mild ones). The passes are
merged and deduplicated.
Adaptive Expansion.We compare the pool’s
opinion breakdown against Ppop. If any bin is
under-represented (pool fraction below threshold
of the population fraction), a targeted retrieval fires
for entity-matched documents in the deficit bin us-
ing pole-biased queries (Prompt A.3.4). If the pool
already mirrors the target, nothing happens - zero
overhead (Algorithm 3).
3.4 Wasserstein Re-Ranking Algorithms
Problem Formulation.For opinion queries, a
faithful answer must reflect how viewpoints dis-
tribute along an ordinal axis - one where positions
have a natural ordering and distances between them
reflect severity of disagreement. The re-ranking ob-
jective follows directly: select kdocuments whose
empirical distribution along this axis minimizes
transport distance to the target distribution Ppop.
We instantiate the axis as sentiment-intensity (SI),
a 7-bin ordinal scale S={−30,−20, . . . ,+30} .
At indexing time each document dreceives an SI
label; per entity ewe derive Ppop(s)as the corpus-
observed fraction of documents for ein each bin,
or as an externally-specified target (star ratings,
survey-calibrated priors) when one is available —
Ppopis an input to the re-ranker, not a claim about
the true underlying population (§Limitations; tol-

Table 1: Main results: Diversity baselines and calibrated methods vs WARP re-rankers. W1↓: distributional distance
to the population opinion target. EM% ↑: fraction of selected documents matching the queried entity. EG/AE =
pool expansion strategies (Entity-Gated / Adaptive Expansion). N=200 candidates, k=20 output. Paired Wilcoxon
vs. Top- k:∗∗∗p<0.001 ,∗∗p<0.01 ,∗p<0.05 . Bold: best per column.†Shared FAISS index across all OpinRank
entities (EM=82.9%); entity-specific indices giveW 1=0.95with 100% EM (Appendix F.3).
Seller Forums Yelp Hotels OpinRank Cars Latency
MethodW 1↓EM%W 1↓EM%W 1↓EM%(ms)
Top-k(baseline) 10.33 42.9 13.03 40.1 13.33 27.00
MMR*10.31 43.5 8.67 14.2 9.64 11.3 12
DPP***11.69 41.2 5.63 78.8 7.38 69.2 45
OpinionMMR**10.44 43.5 8.59 12.4 10.02 10.6 12
KL (JS) Minimizer*8.87 51.4 3.00 93.8 4.54 85.89
W1-MMR***5.86 44.6 4.02 19.43.4121.8 154
W1Minimizer***8.84 51.4 2.76 93.8 4.1485.8 9
WassRank OT***5.7144.4 2.52 92.2 3.87†82.9 89
EG +W 1Min***8.5963.5 1.52 99.24.14 85.8 13
AE +W 1Min***8.5963.51.66 95.9 4.14 85.88
erance to partial estimates in Section 4.4). Given
a candidate pool CofNdocuments, pick k≪N
into result setSminimizing:
W1(P, Q) =m−1X
i=1|CDF P(si)−CDF Q(si)| ·∆s
(1)
where CDF is the cumulative distribution func-
tion, m=7 bins, and ∆s=10 is the bin spacing.
The ordinal structure matters: confusing +30 with
−30 costs Wmax
1=60, while adjacent-bin errors
cost only 10. Evaluation runs in O(m) per can-
didate; rank ordering is preserved under W2(Ap-
pendix C.4).
We implemented three variants that minimize
this objective. They differ in how they cope with
entity pool density and the relevance-calibration
tradeoff:
W1Minimizer.Filter the pool to entity-matched
candidates, then greedily add whichever document
pulls the empirical distribution closest to Ppop,
breaking ties by relevance score (Algorithm 1).
Needs a dense pool where most candidates match
the queried entity.
W1-MMR.Built for sparse pools where entity-
matched candidates alone cannot fill kslots. This
variant works on the full candidate set - no entity
filtering - and scores each candidate by blending re-
trieval relevance with the calibration gain it would
contribute:
score(d) =λ·rel(d) + (1−λ)·∆W 1
Wcur
1(2)rel(d) is the original retrieval score, λcontrols the
relevance-calibration blend, and dividing by Wcur
1
normalizes to percentage improvement so the cal-
ibration term does not vanish as the distribution
tightens. Full algorithm in Appendix B.3.
WassRank OT.Instead of greedy selection,
WassRank (Yu et al., 2019) solves the assignment
globally: allocate kslots proportionally to Ppop,
then find the minimum-cost candidate-to-slot bi-
jection where cost blends normalized ordinal dis-
tance with a relevance penalty. This minimizes
blended transport cost to a proportional discretiza-
tion of Ppop, which empirically tracks W1(Ap-
pendix F.3). We solve the rectangular assign-
ment (Crouse, 2016) in O(k2·N). Originally a
listwise training loss; we repurpose it here as an
inference-time re-ranker with no learned parame-
ters and call it WassRank OT.
4 Results
Four questions drove our experiments: does it work,
does the metric choice matter, do retrieval gains
actually reach the user, and which variant belongs
where? Sections 4.1–4.4 take them in order.
4.1 Main Results
In Table 1, the gains are large and consistent,
concentrating on entities with the most skewed
baseline distributions (per-entity breakdowns in
Appendix C.5). Our three W1re-ranking algo-
rithms (Section 3.4) cut distributional error (Equa-
tion 1) by at least 43% relative to Top- kacross
all three domains — 43% on sparse Seller Fo-

rums ( W1-MMR), 79% on dense Yelp ( W1Mini-
mizer, rising to 88% with pool expansion), and 69%
on OpinRank ( W1Minimizer). Entity relevance
tracks along: pairing Entity-Gated re-retrieval
(Section 3.3) with W1Minimizer pushes Yelp
EM% from 40.1% (Top- kbaseline) to 99.2% (post-
reranking). As a sanity check, 5% Gaussian per-
turbation of retrieval scores yields only 2–3% W1
improvement, non-significant. This confirms that
the score structure drives these results and not the
arbitrary reshuffling. (Appendix F.2)
4.2 The Ordinal Metric Makes a Measurable
Difference
The metric matters. Swapping JS divergence for
W1while keeping the same greedy loop and en-
tity filtering cuts error by 8.7% on Yelp and 9.7%
on OpinRank (Appendix C.1). JS penalizes a
+30↔ −30 confusion no more than an adjacent-
bin slip; W1charges six times more. Entity filtering
alone accounts for 43.4% W1reduction on Yelp;
W1Minimizer adds 62.6% beyond that, and on en-
tity sparse domains, like Seller Forums, W1-MMR
adds 37.7% beyond the entity-filter baseline (Ap-
pendix C.2). We also stress-tested under noise:
corrupting both SI labels andP popsimultaneously,
W1Minimizer never falls below Top- keven at 50%
label error on dense pools, and still outperforms
Top-kby 72% at 20% Dirichlet perturbation (Ap-
pendix F.1).
4.3 Retrieval Gains Propagate to Generation
Do retrieval gains actually reach users? We gen-
erated answers from each method’s retrieved evi-
dence (prompt in Appendix A.3.2) and ran blind
pairwise comparisons against Top- kusing a 5-
judge LLM panel (Claude Sonnet 4 (Anthropic,
2025), Llama 3.3 70B (Meta, 2024), Mistral Large
3 (Mistral AI, 2025), Amazon Nova Pro (Ama-
zon AGI, 2025), DeepSeek v3.2 (DeepSeek-AI,
2025)) with position control (Zheng et al., 2023)
and 3/5 majority vote (Appendix G.1). All five
judges from independent model families agree di-
rectionally at κd=0.61 –1.00; the random baseline
correctly shows no significance.
Atk=5, calibration methods win 70–89% of de-
cided comparisons (Table 2). At k=10 this drops
to 73–88% (Table 20) as larger context lets the
LLM average out distributional imbalance (Ap-
pendix G.1.3). A prompt ablation (Appendix G.1.4)
disentangles two evaluation criteria: when judges
assessproportional accuracy(does the answer re-Table 2: Generation evaluation ( k=5): 5-judge major-
ity vote, position-controlled blind pairwise vs. Top- k
(N=156 queries). Fair% = W/(W+L). Win% = W/ N.
All approaches significant at p<0.001 (sign test) except
Random.k=10in Appendix G.1.3.
Approach W/L/T Fair% Win%
Random 16/36/104 31% 10%
MMR∗39/8/109 83% 25%
DPP∗54/7/9589%35%
WassRank OT∗66/28/62 70%42%
W1-MMR∗44/10/102 81% 28%
W1Minimizer∗51/8/9786%33%
flect how common each view is?), W1Minimizer
leads DPP by 8 pp and matches an oracle stratified-
sampler within 1 pp (Table 21); under pureview-
point coverage(does it mention all perspectives?),
the gap narrows to parity. W1’s proportional advan-
tage persists through generation, not just retrieval.
Metric isolation.On sparse pools, W1-MMR ap-
pears to lose to KL(JS)-MMR (Appendix C.1), but
the comparison is confounded: KL(JS)-MMR se-
lects 1.7–2 ×more entity-matched documents than
W1-MMR on Yelp and OpinRank. Once we con-
trol for EM%, the Minimizer comparison flips: W1
wins on both domains (Appendix C.1), and a single-
judge generation evaluation confirms the same di-
rection, with W1Minimizer leading KL(JS) Mini-
mizer by 7 pp Fair% cross-domain (Appendix G.2).
Two other patterns are worth noting. First,
semantic-diversity baselines fade as kgrows:
MMR and OpinionMMR Fair% collapse to 29–
33% at k=20 , while distributional methods keep
gaining (WassRank OT: 70% →73%→92% across
k∈{5,10,20} ; Appendix G.1.3). Second, DPP and
W1optimize different targets — bin diversity vs.
bin ratios — which happen to align at k=5 with
2–3 active bins but diverge at k=20 . On OpinRank,
near-uniform Ppopcompresses ordinal distances, so
the KL/ W1gap narrows; the ordinal advantage is
largest where distributions are skewed (Yelp, Seller
Forums).
4.4 Deployment Characterization
Which algorithm to pick comes down to one num-
ber: entity match percentage (Table 3). Dense
pools, where all retrieved documents discuss the
query entity, are easy. When EM ≥85% ,W1Mini-
mizer delivers 69–81% reduction, statistically in-
distinguishable from oracle stratified sampling.

Table 3: Deployment decision matrix. Pool density de-
termines algorithm choice; all methods SLA-compliant
(N=200 ,k=20 ,m=7 bins).|Ce|= entity-matched can-
didates. Noise tolerance details in Appendix F.1.
Pool Condition Method ms Complexity
Dense (EM≥85%)W 1Minimizer 9O(|C e| ·k·m)
Sparse (baseline EM∼40%)W 1-MMR 154O(N·k·m)
Variable / unknown EG +W 1Min 13O(|C e| ·k·m)
Entity-Gated re-retrieval pushes Yelp further ( W1:
2.76→1.52, EM%: 93.8 →99.2); on OpinRank
both expansion strategies detect no deficit and self-
bypass. Sparse entity pools are a different story. On
Seller Forums (Top- kbaseline EM ∼40%; W1Min-
imizer reaches ∼51% post-reranking), W1-MMR
scores all 200 candidates regardless of entity match,
reaching 43% reduction while maintaining 81%
generation propagation (Table 2). This outperforms
WassRank OT (70%) despite comparable retrieval-
sideW1, making W1-MMR the preferred hybrid
for sparse and variable pools. Re-ranking latency
stays under 330 ms across the board; end-to-end
depends on retrieval infrastructure (Appendix B.5).
For dense pools N=100 suffices; W1-MMR bene-
fits from largerNon sparse pools (Appendix F.3).
An exact Ppopturns out to be unnecessary: fifty
entity-matched documents preserve 74–98% of or-
acle improvement; smoothing with a domain prior
(τ=50 ) matches the oracle’s harm rate. Cold-start
entities fall back to this prior while still beating
Top-k(Appendix F.4). Label source is flexible -
both LLM-extracted and V ADER work, as do man-
aged cloud vector stores and local FAISS indices.
4.5 Independent-Target Validation
A natural concern is circularity: our LLM-derived
sentiment-intensity (SI) labels define both the pop-
ulation target Ppopused at inference and the W1
metric applied to the retrieved evidence. If the
labeler had systematic biases, calibrating and mea-
suring against it could inflate the numbers. To rule
this out, we cross-validate against Yelp’s native 1–5
star ratings on all 13,533 reviews. Stars are ordinal,
human-authored at review time, and participate in
no partof the WARP pipeline — neither retrieval,
nor re-ranking, nor the Ppoptarget. Mapped to the
same 7-bin SI scale, star-derived and LLM-derived
labels correlate at Spearman ρ=0.881 (p<10−300)
and Pearson r=0.908 , with 81.3% same-valence
agreement and MAE 7.2 SI units — less than one
bin (Appendix E).
Correlation on individual labels is necessary butnot sufficient; a labeler could still consistently
mis-estimate the entity-level Ppop. We therefore
recompute Ppopfrom star ratings alone for each
of the 10 Yelp entities and evaluate WARP’s re-
trievals against this fully independent target. Mean
W1gap between star-derived and LLM-derived
Ppop: 2.88 (Appendix Table 16). Triangle in-
equality then gives W1(WARP,star-P pop)≤5.64
andW1(Top-k,star-P pop)≥10.15 , i.e., a worst-
case reduction of at least 44.4%against a target
our labels never touched. This complements two
other circularity defenses — the V ADER labeling
ablation and the mislabel sensitivity study (Ap-
pendix F), where W1Minimizer never crosses the
Top-kbaseline on Yelp even at 50% label corrup-
tion — and the FDR-corrected significance testing
and entity-clustered bootstrap in Appendix D. The
gains are not an artifact of the labeler or of any
single evaluation protocol.
5 Conclusion
As enterprise document corpora grow, RAG sys-
tems have become the default interface between
users and organizational knowledge, returning rel-
evant evidence in sub-second re-ranking latencies.
Yet when the underlying corpus is diverse and
highly opinionated, relevance-ranked retrieval se-
lects only for topical similarity; the retrieved set
may not reflect population opinions proportionally,
producing summaries that appear grounded but mis-
represent the balance of views.
WARP closes this gap by re-ranking retrieved ev-
idence against the target population distribution, re-
quiring no model fine-tuning, no retriever changes,
and no additional inference calls. The result is at
least a 43% reduction in distributional distance to
the population target ( W1), all within a 330 ms la-
tency envelope suitable for production deployment.
Limitations
Offline evaluation under production constraints.
All reported results come from offline experiments,
engineered to satisfy the sub-330 msre-ranking
budget typical of a large-scale e-commerce RAG
setting (end-to-end depends on retrieval infrastruc-
ture; Appendix B.5). We therefore characterize
retrieval and generation-side fidelity, not live user
outcomes: whether proportionally faithful sum-
maries measurably shift user trust, engagement,
or decision quality under live traffic is untested,
and offline W1reductions need not map one-to-

one onto user-perceived faithfulness. A controlled
online A/B test is the natural next step.
Corpus bias vs. retrieval bias. Ppopiscorpus-
observed, not a true population target: review
corpora are self-selected, motivated writers with
strong positive or negative experiences are over-
represented, and some segments never write at all.
WARP removes theretrieval-inducedbias — Top-
klayers cosine-similarity skew on top of the under-
lying corpus bias — but does not remove the corpus
bias itself. Faithful matching over a biased sam-
ple can lend false authority: the summary reflects
the writers’ distribution, not the underlying popu-
lation’s. Because Ppopis aninputto the re-ranker
rather than something it learns, external signals
(native star ratings, demographic priors, survey-
calibrated distributions) can be plugged in directly
to reweight the target — a natural extension requir-
ing no architectural change.
Pre-computed labels required.Per-document
sentiment-intensity annotations are needed at index-
ing time. A single LLM pass or V ADER suffices,
and 50–100 labels per entity are enough - but the
cost–quality tradeoff of labeling strategies is not
evaluated.
Scale and Temporal Drift.156 queries across 26
entities; paired Wilcoxon provides adequate power,
but scaling beyond 14K documents is untested.
Ppopdegrades gracefully under perturbation (72%
better than Top- katε=0.2 ) but temporal drift is
not addressed.
Single ordinal axis; multi-issue opinions.
WARP operates on a single ordinal axis (senti-
ment intensity). Multi-issue opinion spaces — e.g.,
a review that praises price but criticizes support,
or patient-experience, employee-engagement, and
multi-issue polling settings where several stance
dimensions matter jointly — would require multi-
marginal optimal transport, a natural extension we
do not evaluate here.
Domain scope and generation.All domains are
product/service reviews requiring entity-anchored
opinions and an ordinal scale. Generation signifi-
cance is driven by Yelp (97% decided) and Opin-
Rank (94%); Seller Forums differentiates weakly
(54%) due to sparse pools. A prompt ablation (Ap-
pendix G.1.4) shows reported gaps are criterion-
dependent.Ethics Statement
Distributional fidelity proportionally surfaces mi-
nority views, including potentially harmful ones. A
content-safety filter should gate re-ranker output:
excluded opinions are removed from both Ppopand
the candidate poolbeforecalibration. Because re-
ranking is post-retrieval, it is compatible with any
upstream safety filter without modification.
Use of AI Writing Assistance
The research (design, experiments, analysis, and
writing) is the authors’ own work. We used Claude
(Anthropic) as a proofreader and sounding board: it
flagged unclear phrasing, suggested structural edits,
and helped scaffold parts of the experimental code,
all of which we reviewed and revised ourselves.
Figure 1 was generated with ChatGPT (OpenAI) to
give readers a quick visual overview of the pipeline;
we verified its accuracy. No AI system produced
research claims or final prose.

References
Aditya Agrawal, Alwarappan Nakkiran, Darshan
Fofadiya, Alex Karlsson, Harsha Aduri, and
Aman Singh Thakur. 2026. Retrieval-augmented gen-
eration must move beyond factual grounding to rep-
resent diverse opinions.Preprint, arXiv:2604.12138.
Elif Aktolga and James Allan. 2013. Sentiment diversi-
fication with different biases. InProceedings of ACM
SIGIR, pages 593–602.
Amazon AGI. 2025. The Amazon Nova family of
models: Technical report and model card.Preprint,
arXiv:2506.12103.
Amazon Web Services. 2024a. Amazon Titan text
embeddings v2. https://docs.aws.amazo
n.com/ai/responsible-ai/titan-text
-embeddings/overview.html . Accessed:
2026-05.
Amazon Web Services. 2024b. Knowledge
bases for Amazon Bedrock. https:
//docs.aws.amazon.com/bedrock/la
test/userguide/knowledge-base.html .
Accessed: 2026-05.
Stefanos Angelidis and Mirella Lapata. 2018. Sum-
marizing opinions: Aspect extraction meets senti-
ment prediction and they are both weakly supervised.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
3675–3686, Brussels, Belgium. Association for Com-
putational Linguistics.
Anthropic. 2025. The Claude model family.
https://www-cdn.anthropic.com/6b
e99a52cb68eb70eb9572b4cafad13df32e
d995.pdf. Accessed: 2026-05.
Jaime Carbonell and Jade Goldstein. 1998. The use of
MMR, diversity-based reranking for reordering doc-
uments and producing summaries. InProceedings of
ACM SIGIR, pages 335–336.
Sebastian Claici, Aude Genevay, and Justin Solomon.
2018. Wasserstein measure coresets.arXiv preprint
arXiv:1805.07412.
Jacob Cohen. 1960. A coefficient of agreement for
nominal scales.Educational and Psychological Mea-
surement, 20(1):37–46.
David F. Crouse. 2016. On implementing 2D rectan-
gular assignment algorithms.IEEE Transactions on
Aerospace and Electronic Systems, 52(4):1679–1696.
Van Dang and W. Bruce Croft. 2012. Diversity by pro-
portionality: An election-based approach to search
result diversification. InProceedings of ACM SIGIR,
pages 65–74.
DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the
frontier of open large language models.Preprint,
arXiv:2512.02556.Kavita Ganesan and ChengXiang Zhai. 2012. Opinion-
based entity ranking.Information Retrieval,
15(2):116–150.
C.J. Hutto and Eric Gilbert. 2014. V ADER: A parsi-
monious rule-based model for sentiment analysis of
social media text. InProceedings of AAAI ICWSM.
Han Jiang, Rui Wang, Zhihua Wei, Yu Li, and Xinpeng
Wang. 2023. Large-scale and multi-perspective opin-
ion summarization with diverse review subsets. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2023, pages 5641–5656, Singapore.
Association for Computational Linguistics.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.
Billion-scale similarity search with GPUs.IEEE
Transactions on Big Data, 7(3):535–547.
Alex Kulesza and Ben Taskar. 2012. Determinantal
point processes for machine learning.Foundations
and Trends in Machine Learning, 5(2–3):123–286.
Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Wein-
berger. 2015. From word embeddings to document
distances. InProceedings of ICML, pages 957–966.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Qiheng Lu and Nicholas D. Sidiropoulos. 2026. Prin-
cipled and scalable diversity-aware retrieval via
cardinality-constrained binary quadratic program-
ming.Preprint, arXiv:2604.02554. Concurrent
work.
Meta. 2024. Llama 3.3 model card. https:
//www.llama.com/docs/model-cards
-and-prompt-formats/llama3_3/ . Ac-
cessed: 2026-05.
Mistral AI. 2025. Mistral large 3. https:
//docs.mistral.ai/models/model-c
ards/mistral-large-3-25-12 . Accessed:
2026-05.
Mir Tafseer Nayeem and Davood Rafiei. 2025. Opin-
ioRAG: Towards generating user-centric opinion
highlights from large-scale online reviews. InSecond
Conference on Language Modeling.
Vu Nguyen and Andrey Kan. 2026. MUSS: Multilevel
subset selection for relevance and diversity.Preprint,
arXiv:2503.11126.
Alex Oesterling, Claudio Mayrink Verdun, Carol Xuan
Long, Alexander Glynn, Lucas Monteiro Paes, Sa-
jani Vithana, Martina Cardone, and Flavio P. Calmon.
2024. Multi-group proportional representation in
retrieval. InAdvances in Neural Information Pro-
cessing Systems, volume 37, pages 114601–114655.
Curran Associates, Inc.

Chao Peng, Bin Wang, Zhilei Long, and Jinfang Sheng.
2025. AdaGReS: Adaptive greedy context selection
via redundancy-aware scoring for token-budgeted
RAG.Preprint, arXiv:2512.25052.
Gabriel Peyré and Marco Cuturi. 2019. Computational
optimal transport.Foundations and Trends in Ma-
chine Learning, 11(5–6):1–257.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using siamese BERT-
networks. InProceedings of EMNLP-IJCNLP, pages
3982–3992.
Rodrygo L.T. Santos, Craig Macdonald, and Iadh Ou-
nis. 2010. Exploiting query reformulations for web
search result diversification. InProceedings of
WWW, pages 881–890.
Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, We-
icheng Ma, and Soroush V osoughi. 2025. Judging the
judges: A systematic study of position bias in LLM-
as-a-judge. InProceedings of the 14th International
Joint Conference on Natural Language Processing
and the 4th Conference of the Asia-Pacific Chapter of
the Association for Computational Linguistics, pages
292–314, Mumbai, India. The Asian Federation of
Natural Language Processing and The Association
for Computational Linguistics.
Ashudeep Singh and Thorsten Joachims. 2018. Fairness
of exposure in rankings. InProceedings of ACM
SIGKDD, pages 2219–2228.
Harald Steck. 2018. Calibrated recommendations. In
Proceedings of ACM RecSys, pages 154–162.
Peggy Tang, Kun Hu, Rui Yan, Lei Zhang, Junbin Gao,
and Zhiyong Wang. 2022. OTExtSum: Extractive
text summarisation with optimal transport. InFind-
ings of the Association for Computational Linguis-
tics: NAACL 2022, pages 1128–1141, Seattle, United
States. Association for Computational Linguistics.
Aman Singh Thakur, Kartik Choudhary, Venkat Srinik
Ramayapally, Sankaran Vaidyanathan, and Dieuwke
Hupkes. 2025. Judging the judges: Evaluating align-
ment and vulnerabilities in LLMs-as-judges. In
Proceedings of the Fourth Workshop on Generation,
Evaluation and Metrics (GEM2), pages 404–430, Vi-
enna, Austria and virtual meeting. Association for
Computational Linguistics.
Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei
Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong,
Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large lan-
guage models are not fair evaluators. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 9440–9450, Bangkok, Thailand. Association
for Computational Linguistics.
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan
Yang, and Ming Zhou. 2020. MiniLM: Deep self-
attention distillation for task-agnostic compression of
pre-trained transformers. InProceedings of NeurIPS.Haolun Wu, Yansen Zhang, Chen Ma, Fuyuan Lyu,
Bowei He, Bhaskar Mitra, and Xue Liu. 2024. Re-
sult diversification in search and recommendation: A
survey.Preprint, arXiv:2212.14464.
Yelp. 2024. Yelp open dataset. https://www.ye
lp.com/dataset. Accessed: 2026-05.
Hai-Tao Yu, Adam Jatowt, Hideo Joho, Joemon M. Jose,
Xiao Yang, and Long Chen. 2019. WassRank: List-
wise document ranking using optimal transport the-
ory. InProceedings of the Twelfth ACM International
Conference on Web Search and Data Mining, pages
24–32. Association for Computing Machinery.
Meike Zehlike, Francesco Bonchi, Carlos Castillo, Sara
Hajian, Mohamed Megahed, and Ricardo Baeza-
Yates. 2017. FA*IR: A fair top-k ranking algorithm.
InProceedings of the 2017 ACM on Conference on In-
formation and Knowledge Management, pages 1569–
1578. ACM.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-Bench and chatbot arena.
InAdvances in Neural Information Processing Sys-
tems, volume 36.
A Experimental Setup
A.1 Dataset Details
Opinion Scale.Sentiment-Intensity (SI) is
mapped to a discrete ordinal scale:
Table 4: Sentiment-Intensity (SI) mapping to the 7-bin
ordinal scale.Mixedlabels (praise-and-criticism co-
occurring in one review) collapse to the neutral bin;
they account for 4–8% of extractions across domains
andW1Minimizer never crosses the Top- kbaseline
on Yelp even at 50% label corruption (§F.1), so this
pooling is bounded in impact. Multi-issue axes require
multi-marginal transport — outside our current scope
(§Limitations).
Sentiment Intensity SI Score
Positive High / Med / Low +30 / +20 / +10
Neutral Any 0
Mixed — 0
Negative High / Med / Low−30/−20/−10
Entity Selection.Three stages. First, LLM-
based entity extraction with domain-specific seed
lists. Then diversity filtering: Shannon entropy
H≥0.6 , minority sentiment ≥10% , mention
count≥100 . Finally, top- nselection ranked by
entropy descending.

Table 5: Dataset summary across three evaluation domains.
Dataset Domain Source Size Entities Queries Search Embedding Chunking Labeling
Seller Forums E-comm. seller Public∼8K 6 36 Hybrid Titan V2 (Amazon Web Services, 2024a) (1024d) 300 tok / 20% overlap LLM-enriched metadata
Yelp Hotels Hospitality Yelp Open 14K 10 60 Semantic MiniLM-L6 (384d) Whole review LLM-extracted (Claude)
OpinRank Cars Automotive OpinRank 13K 10 60 Semantic MiniLM-L6 (384d) Whole review LLM-extracted; V ADER
A.2 Query Templates
Table 6 lists the exact query templates used to gen-
erate evaluation questions. Each entity is instanti-
ated into all 6 templates, yielding 156 total queries
(36 Seller Forums + 60 Yelp + 60 OpinRank).
Entities. Seller Forums(6): A+ Content, Ac-
count Health Rating, Brand Stores, Community
Management, Coupons, Seller Education.Yelp Ho-
tels(10): Booking Experience, Breakfast, Break-
fast Quality, Business Center, Casino, Loyalty Pro-
gram, Nightly Rate, Pool, Room Quality, Shuttle
Service.OpinRank Cars(10): Oil Change Inter-
val, Dealer Experience, Vehicle Size, Sales Expe-
rience, Fuel Tank Size, Resale Value, City MPG,
Remote Start, Climate Control, Ground Clearance.
A.3 Prompt Library
All prompts used in the WARP pipeline. Template
variables shown in{braces}.
A.3.1 Opinion Extraction
Prompt A.3.1: Registry-Guided Extraction (Open-
Domain)
A single LLM call per document extracts enti-
ties and sentiment-intensity labels using a domain-
specific tiered seed list while remaining open to
discovering new entities.
System:You extract structured opinion data from reviews.
Output ONLY valid JSON arrays. No markdown, no
explanation.
User:Extract entity-level opinions from this review.
[{registry_block} -- domain-specific
tiered entity list injected here]
Instructions:
1. Identify ALL entities/aspects the reviewer discusses.
2. If an entity matches the registry above, use the EXACT
registry name. Prefer the most specific tier (Tier 3 > Tier
2 > Tier 1).
3. If the reviewer discusses something NOT in the registry,
create a short descriptive entity name (open discovery).
4. For EACH entity mentioned, output:
- entity: the entity name (registry canonical or new)
- sentiment: “positive”, “negative”, “neutral”, or “mixed”
- intensity: “high”, “medium”, or “low”
- evidence: exact quote from the review supporting this
opinion (max 120 chars)
Only include entities explicitly discussed. Return []if
none found.
Output ONLY a JSON array.
Review:
{review_text}For closed-domain settings (e.g., Yelp Hotels), the
extraction prompt uses a fixed aspect list with key-
word mappings instead of the tiered registry, but
the output schema is identical.
A.3.2 Answer Generation
Prompt A.3.2: Opinion Summary Generator
Instructs the LLM to synthesize a proportional opin-
ion summary from retrieved documents.
System:You are summarizing community opinions from
review data. Use ONLY the provided documents. Repre-
sent opinions proportionally — include minority views,
not just the majority. Be specific and cite reviewer experi-
ences where possible.
User:Question:{question}
Reviews:
{docs}
Summarize what the community thinks. Be thorough —
cover the full range of opinions proportionally.
Do not add opinions beyond what’s in the documents.
A.3.3 Pairwise Generation Evaluation (Judge)
Each answer pair is evaluated via position-
controlled blind pairwise comparison (positions
swapped in the second call for debiasing). The
judge scores three dimensions; theOPINION FAIR-
NESSdimension is swapped between variants in
our ablation (§G.1.4).
Prompt A.3.3: Pairwise Judge Template
Full evaluation template: blind comparison on opin-
ion fairness, informativeness, and groundedness.

Table 6: Query templates per domain. Each template is instantiated with every selected entity (6 for Seller Forums,
10 for Yelp/OpinRank), producing 6 questions per entity (2 breadth, 2 polar, 2 segment).
Type Idx Seller Forums Yelp Hotels OpinRank Cars
breadth 0 What do sellers think about {en-
tity}?What do guests think about {en-
tity}?What do owners think about {en-
tity}?
breadth 1 Summarize seller opinions on {en-
tity}.Summarize guest opinions on {en-
tity}.Summarize owner opinions on
{entity}.
polar 0 What are the biggest complaints
about {entity}?What are the biggest complaints
about {entity}?What are the biggest complaints
about {entity}?
polar 1 What positive experiences have
sellers had with {entity}?What positive experiences have
guests had with {entity}?What do owners praise most about
{entity}?
segment 0 How do small sellers vs large sell-
ers feel about {entity}?How do business travelers vs
leisure travelers feel about {en-
tity}?How do commuters vs enthusiasts
feel about {entity}?
segment 1 Do new sellers and experienced
sellers differ in their views of {en-
tity}?Do solo guests and families differ
in their views of {entity}?Do new buyers and long-term
owners differ in their views of
{entity}?
System:You are an expert evaluator comparing two an-
swers that summarize community opinions. You do NOT
know which retrieval method produced which answer.
Judge strictly on the quality of the answers themselves.
User:You are evaluating two answers to the same ques-
tion about community opinions. Both answers were gener-
ated from different sets of reviews retrieved for the same
question.
Question:{question}
— Answer A —
{answer_a}
— Answer B —
{answer_b}
Compare the answers on these dimensions. For each, pick
“A”, “B”, or “tie”:
1. OPINION FAIRNESS: [fairness variant
inserted -- see Prompts A.3.3.1 and
A.3.3.2 below]
2. INFORMATIVENESS: Which answer is more helpful,
specific, and actionable for someone trying to understand
what the community thinks? Consider breadth of points
covered, specificity, and usefulness.
3. GROUNDEDNESS: Which answer appears more
grounded in actual community member experiences?
Look for attribution to specific perspectives vs vague gen-
eralizations that could be hallucinated.
Respond ONLY with JSON:
{"fairness": "A"|"B"|"tie",
"informativeness": "A"|"B"|"tie",
"groundedness": "A"|"B"|"tie",
"reasoning": "<1-2 sentences>"}
The baseline evaluation uses a generic fairness in-
struction (“which answer more proportionally rep-
resents the full range of community opinions, in-
cluding minority views?”). Our ablation replaces it
with the two variants below.
Prompt A.3.4: Fairness Variant A — Proportional
Accuracy
Provides ground-truth Ppopto the judge; penalizes
over-representing minority opinions.OPINION FAIRNESS (Proportional Accuracy): The ac-
tual community opinion distribution for this topic is:
{p_pop_description} . Which answer more accu-
rately reflects these real proportions? The ideal answer
should make the reader walk away with a correct sense
of how common each view is: a dominant opinion should
dominate the summary, a rare opinion should be men-
tioned but not over-emphasized. An answer that gives
equal weight to a 10% minority and a 70% majority is
MISLEADING, even if well-intentioned. Pick the answer
whose emphasis better matches the actual distribution
above.
Where {p_pop_description} is dynamically
filled per entity from the ground-truth Ppop, e.g.:
“approximately 65% positive (mostly satisfied),
25% negative (dissatisfied), 10% neutral.”
Prompt A.3.5: Fairness Variant B — Viewpoint Cov-
erage
Rewards distinct viewpoint coverage; favors sur-
facing rare perspectives even if disproportionate.
OPINION FAIRNESS (Viewpoint Coverage): Which an-
swer covers MORE DISTINCT VIEWPOINTS from the
community, giving voice to all perspectives including rare
and minority ones? An answer that only represents the
majority view — even if that majority is large — is LESS
fair than one that surfaces unique perspectives readers
might not otherwise encounter. The ideal answer ensures
no viewpoint goes unheard, even if that means giving
disproportionate space to rare opinions.
A.3.4 Retrieval Augmentation
Prompt A.3.6: Pole-Biased Multi-Query Templates
During pool expansion (Stage 1.5), three sentiment-
pole queries recover documents that cosine retrieval
buries.
Positive:What positive experiences have sellers had with
{entity}?
Negative:What are the biggest complaints about
{entity}?
Neutral:What factual information do sellers share about
{entity}?

Evaluation queries (6 per entity) are generated from
three templates—breadth (“What do {persona}
think about {entity} ?”), polar (“What are
the biggest complaints...”), and segment (“How
do{segment_A} vs{segment_B} feel...”)—
with domain-adapted persona and segment terms
(e.g., “sellers” / “small sellers vs large sellers”).
B Algorithm Details
Table 7 summarizes the key symbols used through-
out.
Table 7: Notation reference.
Symbol Meaning
dA candidate document
eTarget entity
kOutput size (documents returned)
NCandidate pool size
SI(d)Sentiment-intensity label ofd
Ppop Ground-truth population distribution
Ppool Pool-level empirical distribution
W1(P, Q)Wasserstein-1 distance betweenPandQ
λRelevance–calibration trade-off weight
δDeficit threshold for pool expansion
nexp Budget (docs per deficit pole)
rel(d)Relevance score of documentd
EM Entity-matched subset of pool
B.1 Entity-Gated Re-retrieval
The two-pass entity-gated pool construction recov-
ers entity-matched documents that pure relevance
ranking buries.
Algorithm 2Entity-Gated Re-retrieval
Require: Results R(by score), entity e, sizes
n1, n2, thresholdτ
1:P 1←R[1 :n 1](top-n 1by relevance)
2:C 2← {d∈R\P 1:entity(d) =e∧
score(d)≥τ}
3:SortC 2by|SI(d)|descending
4:P 2←C 2[1 :n 2]
5:Pool←dedup(P 1∪P2), sorted by score
6:returnPool
B.2 Adaptive Distribution-Aware Pool
Expansion
When entity-matched documents under-represent a
sentiment pole relative to Ppop, pole-biased queries
(§A.3.4) retrieve additional candidates before re-
ranking. The cold-start path ( EM=∅ ) treats every
pole with non-negligible mass as a deficit, ensuring
new entities still receive balanced pools. In prac-
tice, one to three deficit poles trigger per entity;
nexpcaps total expansion cost at a fixed multiple of
k.Algorithm 3Adaptive Pool Expansion
Require: PoolC0, entity e,Ppop, threshold δ, bud-
getn exp
1:EM← {d∈C 0:entity(d) =e}
2:ifEM=∅then
3:DeficitPoles← {s:P pop(s)>0.01}
4:else
5:P pool(s)← |{d∈EM:SI(d) =s}|/|EM|
6: DeficitPoles ← {s:P pool(s)< δ·P pop(s)}
7:end if
8:ifDeficitPoles=∅then
9:returnC 0
10:end if
11:Retrieve nexpentity-matched docs per deficit
pole
12:Pool←dedup(C 0∪C exp)
13:returnPool
B.3W 1-MMR
Algorithm 4W 1-MMR
Require: PoolC, scores rel(·) , target Ppop,λ, size
k
1:S← ∅
2:while|S|< kdo
3:Wcur
1 ←(
60ifS=∅
W1(EmpDist(S), P pop)else
4:foreachd∈C\Sdo
5:∆W 1←Wcur
1−W 1(EmpDist(S∪
{d}), P pop)
6:score(d)←λ·rel(d) + (1−λ)·
∆W 1/Wcur
1
7:end for
8:d∗←arg max d∈C\S score(d)
9:S←S∪ {d∗}
10:end while
11:returnS
B.4 Label Acquisition
Per-document SI labels come from a single LLM
pass at indexing time ( <200 ms/doc). Labeling
is cheap. The Seller Forums corpus ( ∼8K posts)
labels in under 4 minutes with 150 concurrent work-
ers. V ADER (Hutto and Gilbert, 2014) works as a
zero-cost alternative for well-structured review text.
Ppopis just a count aggregation per entity ( O(n) ,
milliseconds); incremental updates on ingestion
avoid recomputation entirely.

B.5 End-to-End Latency Breakdown
Hardware.Intel Xeon Platinum 8175M @
2.50 GHz, 8 cores (single-threaded execution), no
GPU.
WARP’s re-ranking step runs entirely in NumPy
on the retrieved candidate distribution — no addi-
tional model calls. End-to-end latency therefore
decomposes into three components (Table 8):
Re-ranking latency is dominated by CDF differ-
encing over m=7 bins and is backend-independent.
Retrieval and re-retrieval dominate managed-
endpoint deployments; for local FAISS the whole
pipeline fits inside a 500 ms sub-second SLA.
TheW1Minimizer’s near-zero p50 on Seller Fo-
rums (0.1 ms) reflects the sparse entity-filtered pool:
few candidates match the queried entity, so the
greedy loop terminates almost immediately. p99
stays under 310 ms across all WARP variants on
both domains, and end-to-end with local FAISS
retrieval fits in <365 ms — comfortably inside a
sub-second SLA. Managed vector stores add ∼1.5 s
of retrieval latency that WARP does not remove.
C Retrieval Ablations
We ablate key components of the retrieval pipeline:
distance metric choice, entity filtering, pool expan-
sion, convergence, per-entity variation, and sensi-
tivity to hyperparameters.
C.1 JS vs.W 1: Metric Comparison and
Entity-Match Confound
This section unpacks the JS-vs- W1comparison
summarized in Section 4.2. We rerun the four cali-
brated methods (KL(JS) Minimizer, W1Minimizer,
KL(JS)-MMR, W1-MMR) on Yelp and OpinRank
atk=20 , holding the greedy structure and entity
filtering fixed so only the distance metric varies
within each row-pair.
Table 10 shows KL(JS)-MMR beating W1-
MMR on Yelp and OpinRank W1. On its face
this reverses the ordinal-metric advantage the Min-
imizer comparison establishes. It doesn’t: the two
hybrids retrieve very different candidate pools, and
the metric is being scored on top of that difference.
In the top block (hybrids, uncontrolled EM),
KL(JS)-MMR retrieves nearly twice as many
entity-matched documents as W1-MMR (38.2% vs.
19.4% on Yelp; 36.1% vs. 21.8% on OpinRank),
and its lower W1tracks that EM gap rather than
the choice of ordinal-vs-categorical ground cost.
In the bottom block (Minimizers, controlled EM),both algorithms filter to the same entity-matched
pool (EM =93.8% Yelp, 85.8% OpinRank), and
W1wins on both domains — 2.76 vs. 3.00 on Yelp,
4.14 vs. 4.54 on OpinRank. The controlled compar-
ison is where the ordinal metric earns its billing; the
uncontrolled hybrid comparison is a retrieval-pool
artifact.
C.2 Entity-Filtered Control (Yelp)
On sparse Seller Forums (Top- kbaseline
EM∼40%; entity-filter ceiling ∼51%), entity
filtering provides only −11.6%W 1reduction (vs.
−43.4% on Yelp, where the entity filter reaches
100% EM), confirming that W1-MMR’s full-pool
scoring drives gains in entity-sparse domains
(−37.7%beyond the entity-filter baseline).
C.3 Adaptive-kConvergence
W1Minimizer hits diminishing returns at k∗=3
(Seller Forums), k∗=4(Yelp), k∗=5(OpinRank)
(Figure 3). An adaptive- kstrategy could there-
fore feed far fewer documents to the LLM without
losing distributional fidelity. This also implies ro-
bustness to impreciseP popestimates.
C.4W 2vs.W 1Comparison
A natural concern is whether optimizing W1(lin-
ear ground cost) sacrifices performance under
higher-order transport metrics. We compute W2
(quadratic ground cost) for all methods on Yelp
Hotels ( k=20 ). The rank ordering across methods
is perfectly preserved: every method that achieves
lower W1also achieves lower W2, with a linear re-
lationship (slope = 1.19 ,R2>0.99 ). This occurs
because our 7-bin ordinal scale has uniform spac-
ing (∆s= 10 ), which makes W1andW2mono-
tonically related for any pair of distributions on
this support. The practical implication is that our
choice of W1, motivated by its O(m) closed-form
computation, does not trade off optimality under
alternative Wasserstein orders.
C.5 Per-Entity Breakdowns
Aggregate results in Table 1 mask substantial per-
entity variation. The gains concentrate on entities
whose baseline distributions are most skewed, pre-
cisely where minority-opinion misrepresentation
does the most damage. We break out per-entity W1
for all three domains below.

Table 8: Per-component latency breakdown. Re-ranking is hardware-only (no model inference); retrieval and
re-retrieval depend on the vector-store backend. p99 for re-ranking; typical/median for the rest.
Component Local FAISS Managed Bedrock Knowledge Base
Retrieval<50 ms∼1.5 s
Re-retrieval†<5 ms∼100 ms
Re-ranking (p99)<310 ms<310 ms
End-to-end<365 ms∼1.9 s
†Only fires when pool expansion (Entity-Gated / Adaptive) detects a deficit.
Table 9: Per-query re-ranking latency percentiles ( N=96 queries: 36 Seller Forums + 60 Yelp Hotels, single-
threaded on the hardware above). p99 drives production SLAs; max reported for tail auditing. All WARP variants
stay under 310 ms at p99.
Domain Method Queries p50 (ms) p95 (ms) p99 (ms) Max (ms)
Seller ForumsW 1Minimizer 360.1 38.7 45.5 48.0
Seller ForumsW 1-MMR 36 156.8 163.1 166.2 167.4
Seller Forums EG +W 1Min 36 0.6 211.0 216.2 218.8
Seller Forums DPP 36 47.8 49.1 51.9 53.0
Yelp HotelsW 1Minimizer 6065.0270.8 289.9 295.3
Yelp HotelsW 1-MMR 60 81.6 284.7 301.6 303.5
Yelp Hotels EG +W 1Min 60 145.7 302.0 307.6 312.5
Yelp Hotels DPP 60 103.5127.4 129.3 131.3
Figure 3: W1Minimizer convergence vs. output size k. Median convergence point k∗(where ∆W 1<0.5 per
additional document): Seller Forumsk∗=3, Yelpk∗=4, OpinRankk∗=5.
Table 10: Entity-match confound in hybrid variants.
KL(JS)-MMR retrieves ∼1.7–2×more entity-matched
documents than W1-MMR on both Yelp and OpinRank.
In the controlled Minimizer comparison (same entity
filter, EM equalized) W1wins on both domains — the
KL(JS)-MMR W1advantage in the uncontrolled com-
parison traces to the EM gap, not to metric superiority.
Yelp OpinRank
MethodW 1 EM%W 1 EM%
KL (JS)-MMR2.5138.2%3.3236.1%
W1-MMR 4.02 19.4% 3.41 21.8%
KL (JS) Minimizer 3.00 93.8% 4.54 85.8%
W1Minimizer2.7693.8%4.1485.8%
C.5.1 Yelp Hotels (k=20)
W1Minimizer lands below 2.0 on 7 of 10 enti-
ties. That is near-perfect calibration. Two outliersTable 11: Entity-filter control experiment (Yelp Hotels,
k=20 ). Entity filtering alone reduces W1by 43.4%; W1
Minimizer achieves 62.6% additional reduction beyond
the entity-filter control.
Approach EM%W 1∆vs Top-K∆vs EF+Top-K
Top K (no filter) 40.1 13.03 — —
EntityFilter + Top-K 100.0 7.38−43.4%—
EntityFilter + MMR 100.0 6.99−46.4%−5.3%
EntityFilter + DPP 100.0 5.64−56.7%−23.6%
W1Minimizer 93.8 2.76−78.8%−62.6%
W1-MMR 19.4 4.02−69.2%−45.5%
EG +W 1Min 99.2 1.52−88.3%−79.4%
AE +W 1Min 95.9 1.66−87.3%−77.5%
remain: Business Center (9.76) and Shuttle Ser-
vice (5.78), both corresponding to entities with
extreme distributional skew and thin pool diversity.
Even greedy optimization cannot fully close the
gap within k=20 selections under those conditions.

Table 12: Per-entity breakdown (Yelp Hotels, 10 as-
pects).W 1Minimizer achieves<2.0 on 7/10 entities.
Entity Top-k W 1W1MinW 1-MMR WassRank OT
Booking Experience 18.30 1.95 3.78 1.58
Breakfast 7.55 1.48 2.87 1.42
Breakfast Quality 5.78 0.87 3.79 0.65
Business Center 24.51 9.76 6.72 9.26
Casino 14.79 0.86 4.07 0.62
Loyalty Program 16.31 4.07 4.49 2.95
Nightly Rate 7.49 0.82 3.45 0.66
Pool 10.79 0.73 2.68 0.46
Room Quality 8.55 1.24 2.67 2.18
Shuttle Service 21.95 5.78 5.70 5.38
C.5.2 OpinRank Cars (k=20, top 10 by
entropy)
OpinRank entities vary widely in mention count
(39–1,465) and minority fraction (19–45%). High-
count entities with moderate minority shares cal-
ibrate well: City MPG, for instance, has 1,465
mentions at 25% minority and reaches W1=0.38 .
Hard cases persist. Vehicle Size has only 42 men-
tions; Dealer Experience starts from a W1=23.07
baseline. Both resist full correction.
D Statistical Validity
Two orthogonal concerns for the significance test-
ing: multiple comparisons across many method ×
domain combinations, and non-independent sam-
ples within a domain (queries drawn from the same
entity share an underlying opinion distribution).
We address both.
D.1 Benjamini–Hochberg Correction
We ran 57 paired Wilcoxon tests across three do-
mains and eight re-ranking variants (baselines and
W1family). Benjamini–Hochberg FDR correction
atα=0.05 leaves 39/57 tests surviving. Table 14
shows the 14 W1-family tests: every one survives
on every domain. The 18 non-surviving tests come
from methods outside our contribution set (Stance-
based re-ranking, generic MMR, DPP-Evidence
variants).
D.2 Entity-Clustered Bootstrap
Paired-Wilcoxon assumes independent paired ob-
servations. Queries drawn from the same entity are
not independent, since they share the same underly-
ing opinion distribution. We therefore resample at
the entity level: 10,000 bootstrap iterations, draw-
ing entities with replacement and recomputing the
mean per-query W1improvement. All 95% CIs ex-
clude zero for every W1method on every domain
(Table 15).
Seller Forums’ wider CIs (e.g., W1Minimizer
[0.67,10.38] ) reflect its small entity count ( N=4 ),not a qualitatively different effect — the lower
bound still exceeds zero. The aggregate signal is
cross-domain consistency across 10 Yelp, 9 Opin-
Rank, and 4 Seller Forums entities.
E Independent-Target Validation
Our LLM-derived sentiment-intensity labels define
both the population target Ppopused at inference
and the W1evaluation metric applied to the re-
trieved evidence. If the labeler had systematic bi-
ases, calibrating and measuring against it could in-
flate the numbers. We cross-validate against Yelp’s
native 1–5 star ratings, which are ordinal, human-
authored at review time, and participate inno part
of the WARP pipeline — neither retrieval, nor re-
ranking, nor the Ppopused at inference. Section 4.5
in the main body summarizes; this appendix reports
the underlying numbers.
A high correlation on individual labels is a nec-
essary but not sufficient condition: the entity-level
Ppopis an aggregate, and a labeler could still con-
sistently mis-estimate it. We therefore recompute
Ppopfrom star ratings alone for each of the 10
Yelp entities and quantify the divergence from the
LLM-derived Ppopactually used at inference (Ta-
ble 16). Mean W1divergence: 2.88. Under a
triangle-inequality bound, WARP’s retrievals stay
within W1=5.64 of the star-derived target while
Top-kis at least W1=10.15 away — a worst-case
reduction of at least 44.4% against a target the la-
beler never touched.
F Robustness & Sensitivity
F.1 Noise Tolerance
Two studies stress-test the system under realistic
noise. First, Dirichlet perturbation of Ppopsim-
ulates inaccurate population estimates. Second,
mislabel sensitivity corrupts both document SI la-
bels and Ppopat once. These jointly reveal how
much noise each algorithm can absorb, and where
deployment breaks down.
F.1.1 Dirichlet Perturbation ofP pop
We perturb Ppopwith Dirichlet noise at ε∈
{0,0.1,0.2,0.5} (N=30 samples per entity) to
simulate estimation error in the population distri-
bution (Figure 4):
On Yelp, which has a dense candidate pool, the
W1Minimizer at ε=0.2 degrades by 31%, but it
still comes in 72% ahead of Top- k.W1-MMR
barely moves under the same perturbation ( <1%

Table 13: Per-entity breakdown (OpinRank, LLM-extracted entities). Count = opinion mentions, H = Shannon
entropy, Min% = minority sentiment fraction.
Entity Count H Min% Top-k W 1MinW 1-MMR WassRank OT
Oil Change Interval 144 1.49 21% 3.96 1.08 2.95 0.78
Dealer Experience 425 1.47 30% 23.07 11.71 3.40 10.28
Vehicle Size 42 1.37 19% — 8.31 6.19 8.31
Sales Experience 776 1.33 28% 15.66 3.01 3.19 2.57
Fuel Tank Size 269 1.32 36% 13.67 3.06 3.25 3.05
Resale Value 427 1.30 39% 12.19 2.24 2.40 2.33
City MPG 1465 1.29 25% 9.12 0.38 2.78 0.38
Remote Start 39 1.25 41% 8.80 4.52 3.54 4.52
Climate Control 727 1.24 45% 9.95 0.95 2.43 0.94
Ground Clearance 201 1.22 33% 23.55 6.12 4.00 5.50
Figure 4: Dirichlet perturbation robustness across all three domains. Dashed red line = Top- kbaseline. At ε=0.2 ,
W1Minimizer remains 72% better than Top-kon Yelp.
Table 14: Benjamini–Hochberg FDR correction
(α=0.05 ) on paired-Wilcoxon p-values for the W1fam-
ily. Every W1-family algorithm survives on every do-
main (14/14). Reduction is per-query paired reduc-
tion relative to Top- k; aggregate reductions in Table 1
weight queries differently.∗∗∗padj<10−3,∗∗padj<10−2,
∗padj<0.05.
Domain ApproachNReductionp adj
YelpW 1Minimizer∗∗∗57+81.4%<0.000001
YelpW 1-MMR∗∗∗57+70.2%<0.000001
Yelp EG +W 1Min∗∗∗57+89.8%<0.000001
Yelp AE +W 1Min∗∗∗57+89.1%<0.000001
Yelp WassRank OT∗∗∗57+83.3%<0.000001
OpinRankW 1Minimizer∗∗∗54+72.4%<0.000001
OpinRankW 1-MMR∗∗∗54+76.7%<0.000001
OpinRank EG +W 1Min∗∗∗54+72.4%<0.000001
OpinRank AE +W 1Min∗∗∗54+72.4%<0.000001
OpinRank WassRank OT∗∗∗54+74.7%<0.000001
Seller ForumsW 1Minimizer∗∗∗24+59.1%0.000427
Seller ForumsW 1-MMR∗∗24+20.4%0.004552
Seller Forums EG +W 1Min∗24+53.7%0.049
Seller Forums AE +W 1Min∗24+53.7%0.049
degradation). On Amazon Seller Forums the pic-
ture is similar in shape: all methods degrade grace-
fully, and W1Minimizer at ε=0.2 stays 56% above
Top-k. OpinRank is the most sensitive of the three
because the unperturbed baseline is already tight
(W1Min sits at 0.87 at ε=0), so relative degrada-Table 15: Entity-clustered bootstrap (10,000 resamples).
Mean improvement in W1relative to Top- k; 95% CIs
computed by resampling entities with replacement. All
CIs exclude zero.
Dataset Method #Entities Mean∆W 1 95% CI
YelpW 1Minimizer 10 10.85[8.26,13.37]
Yelp EG +W 1Min 10 12.09[8.72,15.37]
YelpW 1-MMR 10 9.58[6.40,12.80]
OpinRankW 1Minimizer 9 9.65[6.92,12.34]
OpinRankW 1-MMR 9 10.22[6.48,14.17]
Seller ForumsW 1Minimizer 4 5.32[0.67,10.38]
Seller ForumsW 1-MMR 4 1.84[0.49,3.25]
tion looks larger, but even there ε=0.2 holds 45%
better than Top- k. Across all three domains, W1-
MMR degrades the least, which we attribute to its
hybrid objective partially shielding it from target
noise.
F.1.2 Mislabel Sensitivity (SI Label Error)
A harder question: at what SI label error rate does
theW1advantage drop below significance? We cor-
rupt candidate document labels AND Ppopsimulta-
neously, the realistic scenario where labeling errors
propagate into the population estimate. Error rates
∈ {0,0.05,0.10,0.20,0.30,0.50} ,N=20 repeti-
tions per setting (Figure 5). “Crossover” marks

Table 16: Star-derived vs. LLM-derived
Ppopper Yelp entity. Mean W1gap: 2.88.
Bounds: W1(WARP,star-P pop)≤5.64 ,
W1(Top-k,star-P pop)≥10.15 ; worst-case reduc-
tion≥44.4%.
EntityNreviewsW 1(star-P pop,LLM-P pop)
Booking Experience 1,372 3.66
Breakfast 788 1.44
Breakfast Quality 637 4.15
Business Center 142 3.23
Casino 208 1.36
Loyalty Program 607 1.57
Nightly Rate 2,481 4.60
Pool 1,318 2.10
Room Quality 2,557 1.78
Shuttle Service 280 4.90
Mean—2.88
the error rate at which a method’s W1exceeds the
Top-kbaseline:
Dense pools are forgiving. W1Minimizer never
crosses the Top- kbaseline on Yelp, even at 50%
label error. Sparse pools need more care; use W1-
MMR regardless of label quality, as its hybrid ob-
jective guarantees a floor.
F.2 Noise Control
We need to rule out a trivial explanation: maybe
any score perturbation induces diversity that looks
like opinion-aware signal. So we add 5% Gaussian
noise to retrieval scores and re-rank:
Table 17: Noise control: 5% Gaussian perturbation
of retrieval scores yields negligible W1improvement
(2–3%, p >0.05 ), confirming that our 43–88% distri-
butional improvements ( W1-MMR on sparse pools; W1
Minimizer / EG+ W1Min on dense) represent genuine
distributional optimization.
Method SellerW 1YelpW 1OpinRankW 1
Top-k(baseline) 10.33 13.03 13.33
Top-k+ 5% Noise 10.07 12.63 12.97
Reduction 2.5% 3.1% 2.7%
Significance ns (p >0.05, all domains)
W1Minimizer 8.84 2.76 4.14
Reduction 14.4% 78.8% 69.0%
Random score perturbation yields 2–3% W1
change (non-significant). Our methods hit 14–79%
on the same pools.
F.3 Hyperparameter Sensitivity
λ(relevance–calibration tradeoff).Hybrid
methods ( W1-MMR and WassRank OT) each in-
troduce λ. We sweep λ∈ {0.1,0.3,0.5,0.7,0.9}
across all three domains (Figure 6). W1-MMR
holds steady at λ≤0.5 on Seller Forums and Yelp,
then degrades once relevance dominates. Opin-
Rank behaves differently: λ=0.7 is optimal there(denser pools tolerate more calibration weight).
WassRank OT is nearly λ-invariant on dense pools:
Yelp gives W1≈2.52 forλ≤0.5 (identical to
3 decimal places) and OpinRank yields W1=0.95
with 100% EM across that range.2No per-domain
tuning needed.
Pool size N.We vary N∈ {25,50,100,200}
with fixed k=20 (Figure 7). N=100 suffices for
near-optimal performance on dense pools. W1-
MMR benefits more from larger pools because it
draws from the full (unfiltered) candidate set. On
OpinRank, even N=50 approaches the N=200 re-
sult because entity-specific FAISS indices guaran-
tee most candidates are already entity-matched. On
sparse Seller Forums, larger pools become essen-
tial for W1-MMR to locate cross-entity documents
that fill distributional gaps.
F.4 Population Estimation and Cold-Start
Robustness
We evaluate whether Ppopcan be reliably esti-
mated from indexed data and whether the sys-
tem degrades safely at cold-start. For each en-
titye, the full-corpus distribution P∗
pop,e serves
as our oracle evaluation target. We sample sub-
sets of n∈ {10,25,50,100,250,500} entity-
matched documents, compute ˆP(n)
pop,e, and measure
W1(ˆP(n)
pop,e, P∗
pop,e). Crucially, the re-ranker uses
ˆPpopbut selected evidence is evaluated against
P∗
pop; this avoids circularity.
Estimation convergence.Figure 8 shows estima-
tion error vs. sample size across all three domains.
Convergence is rapid. At n=50 , mean W1between
estimated and oracle distributions drops below 3.0;
atn=100 , below 2.0. Bootstrap CIs (95%) narrow
monotonically, which indicates stable estimation
with modest data.
Cold-start behavior.When an entity has few la-
beled documents, the entity-level Ppopestimate
gets noisy. We simulate this by capping obser-
vation count at n∈ {10,25,50,100} . Even at
n=10 , all estimation strategies beat uncalibrated
Top-k. Smoothed interpolation with a domain
prior ( ˜P=α ˆPe+ (1−α)P domain ,α=n/(n+50) )
works best at every count, achieving 61% W1re-
2Theλ-sensitivity results use entity-specific FAISS indices
(EM=100%), whereas Table 1 uses a single shared FAISS
index across all OpinRank entities (EM=82.9%). This ac-
counts for the OpinRank WassRank OT W1difference (3.87
in Table 1 vs. 0.95 here).

Figure 5: Mislabel sensitivity: W1vs. label error rate. Dashed red line = Top- kbaseline. W1Minimizer never
crosses Top-keven at 50% error on Yelp.
Figure 6: λ-sensitivity for W1-MMR (left) and OpinionMMR (right). W1-MMR is stable at λ≤0.5 and degrades
atλ=0.9. OpinionMMR shows higherW 1across all domains with limited sensitivity toλ.
Figure 7: W1vs. pool size N(k=20 ).N=100 suffices for dense pools; W1-MMR benefits from larger pools on
sparse data.
duction at n=10 . Our recommended production
rule: use entity-level Ppopwhen n≥25 and at
least 2 sentiment bins have mass >0.05 ; otherwise
fall back to the smoothed domain prior.G Generation Evaluation
G.1 Judge Validation & Positional Bias
Generation evaluation should not hinge on a single
model’s quirks; leniency bias and prompt sensi-
tivity are well-documented failure modes (Thakur
et al., 2025). We validate with five independent

10 25 50 100 250 500
Documents sampled (n)01234567W1(̂Ppop,P*
pop)
Ppop Estimation Convergence
Seller Forums
Yelp
OpinRankFigure 8: W1(ˆPpop, P∗
pop)vs. entity-matched documents
sampled. With 50–100 documents per entity, estimation
error is comparable to or below the re-ranking improve-
ment magnitude itself.
model families from different providers: Claude
Sonnet 4 (Anthropic), Llama 3.3 70B (Meta),
Mistral Large 3 (Mistral AI), Amazon Nova Pro
(Amazon), and DeepSeek v3.2 (DeepSeek). Ap-
pendix A.3.3 contains the full judge prompt with
its three evaluation dimensions. Below we con-
firm positional bias, report directional inter-judge
agreement, and present majority-vote consensus.
G.1.1 Positional Bias
LLM judges exhibit positional bias, a preference
for whichever answer appears first (Wang et al.,
2024; Shi et al., 2025). We confirm this across
all five judges in our panel. Our position-swap
protocol converts inconsistent position-dependent
preferences into conservative ties, motivating the
directional κdanalysis in §G.1.2. Bias rates differ
across models, producing asymmetric tie distribu-
tions:
Table 18: Consensus tie rates across the 5-judge
panel. Higher tie rates indicate stronger positional bias
(position-swap converts inconsistent preferences to ties).
Judge Consensus Tie Rate (k=5) Consensus Tie Rate (k=10)
Claude Sonnet 4 46% 46%
Mistral Large 3 48% 61%
DeepSeek v3.2 63% 71%
Llama 3.3 70B 73% 86%
Amazon Nova Pro 79% 86%
When judges do commit to a preference, they
agree directionally 85–100% of the time.
G.1.2 Directional Inter-Judge Agreement ( κd)
We want to separate genuine directional disagree-
ment from artifacts of unequal tie rates. So we com-
puteκdover only those pairs where both judges
commit to a non-tie preference:Table 19: Mean directional κdper judge vs. all others
(fairness dimension). Non-Anthropic judges agree at
κd≥0.95.
Judge (vs. others)k=5κ dk=5Agree%k=10κ dk=10Agree%
Claude vs. others 0.68 88.2% 0.86 93.5%
Llama 3.3 vs. others 0.89 96.0% 0.97 98.8%
Mistral L3 vs. others 0.89 95.8% 0.95 97.8%
DeepSeek vs. others 0.93 97.4% 0.95 97.9%
Nova Pro vs. others 0.88 95.3% 0.94 97.3%
All five judges agree directionally at κd=0.61 –
1.00. Among the four non-Anthropic judges,
pairwise κdranges 0.95–1.00 at both kvalues.
Claude’s lower κd(0.68/ k=5, 0.86/ k=10 ) traces
to its lower tie rate: it commits more often and
occasionally flags nuances others collapse into ties.
G.1.3 Majority Vote Results (3/5 Judges Must
Agree)
Under majority vote (3/5 agreement required), cali-
bration methods reach strong significance at k=5
while Random correctly fails. This is a critical
sanity check. At k=10 , fewer approaches survive;
context-window averaging attenuates distributional
differences. We extend to k=20 under the same
protocol; the scaling regime distinguishes semantic-
diversity from distributional methods.
Atk=5 with only 2–3 active sentiment bins, di-
versity and calibration objectives partly overlap and
MMR/OpinionMMR win 83%/82% (Table 2). By
k=20 the pool has room for finer-grained propor-
tions and pure semantic-diversity has run out of sig-
nal; MMR and OpinionMMR Fair% drop to 29%
and 33%. Distributional methods maintain their
advantage: WassRank OT rises from 70% ( k=5) to
73% ( k=10 ) to 92% ( k=20 ) as budget lets propor-
tional targeting express itself in the answer. DPP
is a partial exception: it also grows with k(89% at
k=20 ) because its determinantal spread lands one
representative per bin when there is enough budget.
Domain decomposition at k=10 reveals a clear
density dependency. On dense Yelp (60 questions,
10 aspects), all calibration methods achieve 86–
97% fairness decided rate; W1Minimizer hits 97%
(28W/1L). Dense OpinRank (60 questions, 10 en-
tities) confirms propagation: DPP leads at 100%
(19W/0L), W1Minimizer at 94% (16W/1L), and
5 of 8 approaches reach p <0.001 . Sparse Seller
Forums (36 questions, 6 entities, EM ∼40%) is the
outlier: only OpinionMMR achieves significance
(10W/2L, 83%, p=0.019 );W1Minimizer and W1-
MMR hover near coin-flip (54–55%), confirming
that sparse pools cap document quality regardless
of re-ranking strategy. Random fails on all three

Table 20: 5-judge majority-vote generation results (cross-domain, 156 questions) at k∈{5,10,20} . All calibration
methods achieve p <0.001 atk=5; retain significance at k=10 (p≤0.021 ) and k=20 (p≤0.004 ). Random
fails at all kvalues. At k=20 , MMR/OpinionMMR Fair% collapse (29%/33%) as semantic diversity saturates;
distributional methods maintain or grow their advantage; WassRank OT rises 70% →73%→92%. Fair% = W/(W+L).
Win% = W/N.
k=5k=10k=20
Approach W/L/T Fair% Win% W/L/T Fair% Win% W/L/T Fair% Win%
Random 16/36/104 31% 10% 12/40/104 23% 8% 3/26/127 10% 2%
MMR∗∗39/8/109 83% 25% 21/9/126 70% 13% 5/12/139 29% 3%
OpinionMMR∗∗50/11/95 82% 32% 25/8/123 76% 16% 7/14/135 33% 4%
DPP∗∗54/7/95 89% 35% 42/6/108 88% 27% 25/3/128 89% 16%
WassRank OT∗∗66/28/62 70% 42% 75/28/53 73% 48% 33/3/12092%21%
W1-MMR∗∗44/10/102 81% 28% 24/7/125 77% 15% 13/2/141 87% 8%
W1Minimizer∗∗51/8/97 86% 33% 32/8/116 80% 21% 27/5/124 84% 17%
domains (13–53%), validating the sanity check.
G.1.4 Judge Prompt Sensitivity
Our main fairness criterion conflates two signals:
proportional accuracyandviewpoint coverage. We
disentangle them by re-judging all answer pairs
under two contrasting definitions (Prompt A and
Prompt B in Appendix A.3.3), same 5-judge panel
with position-swap (Zheng et al., 2023) and 3/5
majority vote:
•Prompt A (Proportional Accuracy):Judges
receive the ground-truth Ppopand are asked
which answer’s emphasis better matches
the actual distribution. An answer over-
representing a 10% minority relative to a 70%
majority is penalized as misleading.
•Prompt B (Viewpoint Coverage):Judges
are asked which answer covers more distinct
viewpoints, including rare ones. Giving dis-
proportionate space to minority views is re-
warded.
INFORMATIVENESS and GROUNDEDNESS
dimensions remain identical across both prompts,
serving as controls.
Tables 21 and 22 reveal three patterns. (1) Under
proportional accuracy, W1Minimizer leads DPP
by 8 pp at k=5 and 6 pp at k=10 ; retrieval cal-
ibration propagates when judges can verify pro-
portions against ground truth. Stratified oracle
sampling reaches 35% Fair% (Table 21) — W1
Minimizer sits within 1 pp of the oracle, confirm-
ing that Prompt A tracks the ground-truth Ppop
rather than any specific retrieval strategy. (2) Un-
der viewpoint coverage, the gap narrows to 3 pp
(k=5) and 2 pp ( k=10 ). DPP’s diversity advantage
is real but modest once W1’s calibrated selectionTable 21: Judge prompt ablation: fairness win rate (%)
under proportional vs. coverage definitions (3 domains,
156 questions per approach). W1Minimizer dominates
under proportional accuracy; the gap narrows under
coverage. Stratified sampling (oracle) confirms the
Prompt A criterion tracks the ground-truth distribution,
not any specific retrieval strategy — W1Minimizer
matches oracle Fair% within 1 pp atk=5.
k=5k=10
Approach Prop. (A) Cov. (B) Prop. (A) Cov. (B)
Stratified (oracle) 35% — — —
W1Minimizer 34% 38% 30% 27%
DPP 26% 35% 24% 25%
W1-MMR 23% 36% 18% 17%
OpinionMMR 21% 30% 13% 19%
MMR 13% 26% 10% 13%
Random 16% 13% 12% 7%
Table 22: Control dimensions remain stable across
prompt variants ( k=5,∆= Prompt B −Prompt A).
Shifts≤4 pp confirm the fairness prompt change does
not contaminate other evaluation axes.
Approach Info% (A) Info% (B) Grnd% (A) Grnd% (B)
W1Minimizer 44% 40% 28% 24%
DPP 38% 40% 27% 24%
Random 19% 18% 15% 8%
already covers most sentiment bins. (3) Inter-judge
agreement rises under Prompt B (Llama–Mistral
κ: 0.39→0.62 at k=5), which suggests viewpoint
counting is more objectively evaluable than propor-
tional matching without ground truth. Our original
conflated criterion thus understates W1’s advantage
at its design objective (proportional faithfulness)
while slightly overstating DPP’s coverage benefit.
G.2 Single-Judge Metric Isolation
(Generation)
The 5-judge majority-vote protocol is conservative
— positional-bias-driven ties inflate the tie column.
To isolate the ordinal metric’s end-to-end contri-
bution with more statistical power, we ran a con-

trolled single-judge (Claude Sonnet 4.6) position-
controlled pairwise evaluation between W1Min-
imizer and KL(JS) Minimizer at k=5 (Table 23).
Same algorithm, same entity-filtered pool; only the
distance function differs.
Table 23: W1Minimizer vs. KL(JS) Minimizer, gen-
eration evaluation at k=5 (156 queries, single-judge
Claude Sonnet 4.6, position-controlled, blind pairwise
vs. Top- k).W1leads KL(JS) by +7 pp cross-domain;
both significantly beat Top-k(p<0.001).
Domain MethodNW/L/T Fair% Win%
Seller ForumsW 1Minimizer 36 10/9/1753% 28%
KL (JS) Minimizer 36 7/10/19 41% 19%
YelpW 1Minimizer 60 39/4/1791% 65%
KL (JS) Minimizer 60 31/8/21 79% 52%
OpinRankW 1Minimizer 60 29/8/23 78% 48%
KL (JS) Minimizer 60 35/10/15 78%58%
Cross-domainW 1Minimizer 156 78/21/5779% 50%
KL (JS) Minimizer 156 73/28/55 72% 47%
W1leads KL(JS) by +7 pp Fair% cross-domain,
and the retrieval-side 8–10% W1advantage (Ap-
pendix C.1) does not invert at generation time. Yelp
shows the largest gap (91% vs. 79%). Seller Fo-
rums shows the same directional advantage at a
smaller magnitude (53% vs. 41%). OpinRank is a
partial exception: Fair% ties at 78% and KL(JS)
edges W1on Win Rate (58% vs. 48%), which we
attribute to OpinRank’s near-uniform Ppopcom-
pressing ordinal distances — when adjacent-bin
errors and pole-to-pole errors carry similar cost in
the target itself, the ordinal ground cost has less to
do.