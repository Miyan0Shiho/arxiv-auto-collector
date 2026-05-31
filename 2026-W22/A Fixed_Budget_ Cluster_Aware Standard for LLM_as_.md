# A Fixed-Budget, Cluster-Aware Standard for LLM-as-a-Judge Evaluation: A Multi-Hop RAG Stress Test

**Authors**: Camilo Chacón Sartori, José H. García

**Published**: 2026-05-27 00:12:02

**PDF URL**: [https://arxiv.org/pdf/2605.27789v1](https://arxiv.org/pdf/2605.27789v1)

## Abstract
Retrieval-augmented generation (RAG) systems are often compared by asking a large language model (LLM) judge which answer is better. For multi-hop RAG, this has become a measurement problem as much as a modeling problem: the same score can reflect retrieval quality, answer length, lexical overlap, or a statistical test that ignores clustered data. We ask what happens when these choices are made explicit.
  We propose a minimum measurement standard for LLM-as-a-judge comparisons in RAG. The standard fixes the top-100 candidate pool, evidence budget, answer cap, generator, and prompt; it also requires pre-registered hypotheses, cluster-aware inference, an exact cluster sign-flip check when feasible, and second-judge replication. Clustered benchmarks can overstate progress; the field should adopt this standard. We stress-test it with Genetic Algorithm Decoder for Multi-hop Evidence Composition (GADMEC), an evolutionary evidence selector, on 400 multi-hop questions in computer science/machine learning (CS/ML) and Materials Science. The protocol changes the empirical story. A binomial test makes all four semantic-baseline comparisons look significant; cluster-aware inference leaves only one Bonferroni-significant result. BM25 beats pure semantic GADMEC under the same budget, while a lexical-semantic hybrid recovers in CS/ML and narrows the Materials Science gap.

## Full Text


<!-- PDF content starts -->

A Fixed-Budget, Cluster-Aware Standard for LLM-as-a-Judge Evaluation:
A Multi-Hop RAG Stress Test
Camilo Chacón Sartori*José H. García
Catalan Institute of Nanoscience and Nanotechnology (ICN2), CSIC and BIST,
Campus UAB, Bellaterra, Barcelona, Spain
camilo.chacon@icn2.cat josehugo.garcia@icn2.cat
Code and materials: github.com/camilochs/fixed-budget-llm-judge-rag
*Corresponding author
Abstract
Retrieval-augmented generation (RAG) sys-
tems are often compared by asking a large lan-
guage model (LLM) judge which answer is
better. For multi-hop RAG, this has become a
measurement problem as much as a modeling
problem: the same score can reflect retrieval
quality, answer length, lexical overlap, or a sta-
tistical test that ignores clustered data. We ask
what happens when these choices are made ex-
plicit.
We propose a minimum measurement stan-
dard for LLM-as-a-judge comparisons in RAG.
The standard fixes the top-100 candidate pool,
evidence budget, answer cap, generator, and
prompt; it also requires pre-registered hypothe-
ses, cluster-aware inference, an exact clus-
ter sign-flip check when feasible, and second-
judge replication. Clustered benchmarks can
overstate progress; the field should adopt this
standard. We stress-test it with Genetic Algo-
rithm Decoder for Multi-hop Evidence Compo-
sition (GADMEC), an evolutionary evidence
selector, on 400 multi-hop questions in com-
puter science/machine learning (CS/ML) and
Materials Science. The protocol changes the
empirical story. A binomial test makes all
four semantic-baseline comparisons look sig-
nificant; cluster-aware inference leaves only
one Bonferroni-significant result. BM25 beats
pure semantic GADMEC under the same bud-
get, while a lexical-semantic hybrid recovers in
CS/ML and narrows the Materials Science gap.
1 Introduction
RAG systems are increasingly compared with pair-
wise LLM-as-a-judge protocols: two answers are
shown to a judge model, and the preferred answer
is counted as a win. This protocol is attractivebecause it is simple and cheap to scale. It also
hides important choices. A method can look bet-
ter because it selected better evidence, because it
induced longer answers, because it matched lexi-
cal cues that dense retrieval missed, or because the
statistical test treated clustered examples as inde-
pendent. In multi-hop RAG, these mechanisms are
easy to mix.
This paper asks whether the headline of a RAG
comparison survives when those choices are con-
trolled. In our experiment, the answer is mixed. A
binomial test would make all four pre-registered
semantic-baseline comparisons look significant.
Cluster-aware inference leaves only one compar-
ison significant after Bonferroni correction, with
two more significant only before correction. The
empirical story therefore depends not only on the
selector, but also on the measurement protocol used
to evaluate it.
We make two claims. First, pairwise LLM-as-a-
judge evaluation for multi-hop RAG is more fragile
than standard reporting suggests. Second, the field
should adopt a minimum measurement standard
for cluster-structured LLM-as-a-judge benchmarks.
The standard fixes evidence and answer budgets.
It separates confirmatory from exploratory analy-
ses through pre-registration and a public deviation
log, uses cluster-aware inference, adds an exact
sign-flip check when the cluster count permits it,
and replicates headline results with a second judge.
These components are not individually new. The
contribution is to make them operate together in
one controlled RAG comparison and show how the
conclusions change.
We stress-test the standard with Genetic Al-
gorithm Decoder for Multi-hop Evidence Com-
position (GADMEC), a Biased Random-Key Ge-
1arXiv:2605.27789v1  [cs.AI]  27 May 2026

netic Algorithm (BRKGA; Gonçalves and Resende,
2011) for evidence subset selection. GADMEC
is the instrument, not the protagonist. Its evolu-
tionary search is separated from the decoder that
enforces budget and diversity constraints, so a
random-fitness ablation can test whether the fitness
function contributes beyond the decoder machin-
ery.
Contributions.
1.Afixed-budget evaluation design for LLM-
as-a-judge comparisons in RAG: same top-
100 candidate pool, 2000-token evidence bud-
get, 300-token answer cap, and generator set-
tings for all methods.
2.Apre-registered analysis protocol : four
Bonferroni-corrected primary hypotheses, an
addendum for length matching and ablations,
and a deviation log for post-hoc analyses.
3.Across-domain benchmark : 687 arXiv pa-
pers (2024–2026), a 3-level taxonomy, 10
cross-subfield combinations per area, 20 ques-
tions per combination, and 200 contrastive
multi-hop questions per area.
4.Amethodological demonstration : the head-
line of a multi-hop RAG comparison can
change when inference respects the cluster
structure. The binomial test would have
reported all four primary tests as signifi-
cant; wild-cluster bootstrap leaves only one
Bonferroni-significant result. The same proto-
col exposes a lexical-vs-semantic axis: BM25
beats pure semantic GADMEC, while a hy-
brid recovers in CS/ML.
The paper unfolds as follows. Section 2 situates
fixed-budget evaluation relative to LLM-as-a-judge
evaluation, multi-hop question answering (QA),
subset selection, and pre-registration. Sections 3
and 4 define GADMEC, the baselines, and the
controlled evaluation protocol. Section 5 reports
the pre-registered results, length controls, content-
distance diagnostics, and ablations. Sections 6–8
interpret the mechanisms, conclude, and state the
limitations.
2 Related Work
LLM-as-a-Judge Evaluation for RAG.
Reference-free LLM-as-a-judge evaluation
is now common in RAG research. RAGASformalised the setup for retrieval-augmented
generation (Es et al., 2024), and recent surveys
catalogue its variants (Li et al., 2025). The closest
concern for this paper is length bias (Zheng et al.,
2023; Dubois et al., 2024). We control both input
and output budgets, then ask what signal remains
after length matching.
Multi-hop QA benchmarks. HotpotQA, 2Wiki-
MultiHop, MuSiQue, and GRADE target chained-
evidence multi-hop reasoning (Lee et al., 2025).
Our questions have a different shape: contrastive
composition (“how do XandYdiffer on as-
pectZ”). This makes evidence selection budget-
sensitive because a good answer must cover several
sub-aspects at once.
Combinatorial subset selection in retrieval.
Maximal Marginal Relevance (MMR; Carbonell
and Goldstein, 1998) and Determinantal Point Pro-
cesses (DPPs; Kulesza and Taskar, 2012) are clas-
sical instances. Genetic algorithms have been ap-
plied to RAG primarily for adversarial attack and
corpus poisoning (Cho et al., 2024). We use a
BRKGA constructively: the candidate pool is the
search space, and the selected subset is the evidence
plan.
Pre-registration in NLP. Pre-registration is rare
in natural language processing (NLP). We follow
social-science practice with a primary registration,
timestamped addenda, and a deviation log separat-
ing confirmatory and exploratory analyses.
Together, these strands motivate an evaluation
instrument rather than only a new retriever. The
selector is allowed to vary; the evidence budget,
answer cap, judge protocol, and evidential status
of each analysis are not.
3 Method
We first define the selector, then the controls that
make the comparisons interpretable. Figure 1 sum-
marizes the full pipeline. The key rule is simple:
every method sees the same candidates and uses
the same generation settings. Only the evidence
plan changes.
3.1 GADMEC fitness
GADMEC’s fitness over a candidate evidence plan
Pcombines five components:
f(P) =αCOV+βDIV+γCOST +δCOH+εSUB
2

1. Corpus construction
Taxonomy 3 levels
2 areas×8 subareas×5 topics
arXiv search per combo
700 candidate combos
10 approved combos / area
5 TOP + 5 NICHO
687 papers
chunked + embedded2. Question generation
Contrastive prompt
“How do X and Y differ in...”
Sample 3 papers / combo
×2 paper triplets
GPT-5.4-mini
T= 0.7
200 questions / area
multi-hop, no method names3. Evaluation
Retrieval
top-100 chunks by cosine
4 evidence plans
GADMEC / Greedy / MMR / BM25
GPT-5.4-mini answers
T= 0, seed=42
Opus 4.7 judge (API)
randomized A/B positionsFigure 1: GADMEC pipeline. The protocol first builds a taxonomy-filtered corpus, then generates contrastive
multi-hop questions, and finally evaluates four selectors under the same candidate pool, evidence budget, generator,
answer cap, and judge (Claude Opus 4.7, randomised answer-A/answer-B positions). The only intended source of
variation is the evidence selector. We re-judge the six headline comparisons with a second strong proprietary judge
from a different provider (DeepSeek V4 Pro, thinking mode) to assess inter-judge robustness (§5.5).
Here COV rewards mean query similarity, and
DIVrewards pairwise dissimilarity among selected
chunks. COST is a normalised token penalty, COH is
centroid–query similarity, and SUB measures cover-
age of GPT-5.4-mini query sub-aspects embedded
with all-MiniLM-L6-v2.
We use α= 0.30, β= 0.15, γ= 0.00, δ=
0.15, ε= 0.40, with sub-coverage threshold 0.40.
The BM25 lexical component ζis0for pure se-
mantic GADMEC and activated only in the hybrid
analyses.
3.2 BRKGA decoder
BRKGA (Gonçalves and Resende, 2011) repre-
sents each plan as random keys in [0,1]nand leaves
feasibility to a decoder. The decoder sorts chunks
by key and accepts them greedily until the plan
reaches the 2000-token budget. It also enforces
minimum query similarity 0.15, redundancy thresh-
old0.80, and at least three k-means thematic clus-
ters. The search uses population size 20, elite
fraction 0.24, elite-inheritance probability 0.70, at
most 50 generations, early stopping after 15 stag-
nant generations, and seed 42.
3.3 Baselines
We compare against greedy top- kbudget-fill, Max-
imal Marginal Relevance (MMR) with λ= 0.5
(Carbonell and Goldstein, 1998), and the BM25
lexical retriever ( k1= 1.5, b= 0.75). All meth-
ods operate on the same top- 100cosine-pre-filtered
candidate pool. Win-rate differences therefore re-flect how methods choose evidence from the same
pool, not whether one method had access to better
candidates.
3.4 Question generation and judging
Question generation uses a contrastive multi-hop
prompt with GPT-5.4-mini at T= 0.7. Each
area has 10 combinations, with 20 questions
per combination. Answers use GPT-5.4-mini at
T= 0, seed 42, evidence-only prompting, and
max_completion_tokens=300 . Claude Opus 4.7
judges randomised answer-A/answer-B pairs in a
single pass using a fixed pairwise prompt. The
prompt asks for a global preference based on
factual correctness, completeness, evidence sup-
port, clarity, specificity of evidence-backed claims,
multi-source synthesis, and coverage of the ques-
tion aspects. Ties are allowed but reserved for
answers that are truly equivalent; ties are recorded
and excluded from the win-rate denominator.
4 Experimental Setup
The setup applies the fixed-budget design to two
scientific domains. Each pairwise judgment com-
pares evidence selection under the same retrieval
substrate, input budget, and answer-generation bud-
get.
4.1 Corpus
The corpus is organised by a 3-level taxonomy:
area→subarea →topic. We use two areas, Com-
puter Science/Machine Learning (CS/ML; 341 pa-
3

pers) and Materials Science (MatSci; 346 papers),
drawn from arXiv between 2024-01 and 2026-05.
Each area contributes 10 cross-subfield combina-
tions: five high-density TOP-regime combinations
with at least 100 arXiv papers and five lower-
density NICHO combinations with 10–50 papers.
The manifest includes SHA-256 hashes over arXiv
IDs.
4.2 Fixed-budget control
All methods use a 2000-token evidence budget with
budget-fill. For compact tables and figures, we
abbreviate GADMEC as GA. The resulting input
lengths are tightly matched (median evidence to-
kens over 200 questions per area; parentheses give
IQRs when shown):
Selector CS/ML MatSci
GA 1979 (1960 –1993 )1973 (1938 –1992 )
Greedy 1986 (1966 –1994 )1979 (1960 –1992 )
MMR 1984 1982
BM25 1986 (1972 –1995 )1987 (1967 –1994 )
All methods fill approximately 99% of the budget.
Answer generation also uses identical caps and set-
tings. Thus systematic answer-length differences
are not caused by unequal budgets; they reflect
what the selected evidence makes available to the
generator.
4.3 Pre-registration
The primary registration locks four hypotheses
of the form WR(GA vs. BASE)>0.5, where
WR denotes win rate. Each hypothesis uses one-
sided cluster bootstrap over combination identifiers
(10 000 resamples) and Bonferroni α= 0.0125 .
The addenda lock the ≤200-character length-
matched bin, ablation predictions, BM25 parity
checks, and hybrid predictions before judging. The
deviation log records analyses added later, includ-
ing bin sensitivity, content-distance slicing, and hy-
brid configurations. It also records two drift items:
Opus 4.7 rather than the pre-registered 4.6 judge,
and 200 rather than 100 questions per area.
5 Results
5.1 Main results
We report results in four blocks. Block 1 gives the
pre-registered semantic comparison against Greedy
and MMR. Block 2 adds BM25 as a lexical base-
line under the same budget. Block 3 tests whether
the BRKGA machinery alone explains the gains.
Block 4 asks whether adding lexical signal to theGADMEC fitness recovers the BM25 gap. The
order is diagnostic: each block removes a different
explanation for the observed win rates.
Block 1 — Primary (pre-registered, Bonfer-
roniα= 0.0125 over 4 tests). Table 1 reports the
four primary semantic comparisons.
Table 1: Primary semantic comparisons.
Comparison Area WR % pVpWpPBonf
GADMEC vs Greedy CS/ML 57.4 0.0251 0.0515 0.0586 ×
GADMEC vs Greedy MatSci 58.9 0.0205 0.0394 0.0391 •
GADMEC vs MMR CS/ML 60.1 0.0000 0.0043 0.0078 ⋆
GADMEC vs MMR MatSci 61.7 0.0025 0.0201 0.0234 •
We report three p-values, each with a separate
role. pVcomes from a vanilla cluster bootstrap
by combination identifier (10,000 resamples). pW
comes from a pivotal wild-cluster bootstrap with
Webb 6-point weights (Cameron et al., 2008), cen-
tred at the null H0: WR = 0 .5.pPcomes from
an exact cluster sign-flip permutation test. Because
each area has 10 clusters, the exact test can enumer-
ate all 210= 1024 sign assignments. It therefore
does not rely on large-cluster asymptotics.
The wild bootstrap and the exact sign-flip test
agree to within 0.005 on every primary com-
parison. This makes the small-cluster concern
visible rather than hidden in a single number.
For the four-test primary family, Bonferroni de-
cisions use pWatα= 0.0125 :⋆passes Bon-
ferroni, •passes uncorrected α= 0.05, and×
fails. One of four primary tests passes Bonfer-
roni (GADMEC vs MMR in CS/ML); two more
pass only before correction. The gap between pV
and the cluster-aware p-values reflects per-cluster
heterogeneity (cross-combination standard devi-
ation of win rate ≈12percentage points; see
results/all_bootstrap_pvalues.json ). We
return to this point in §6.
Block 2 — Secondary (BM25 retrieval base-
line, addendum, descriptive). Table 2 adds the
lexical BM25 baseline under the same budget.
Table 2: BM25 lexical-baseline comparisons.
Comparison Area WR % pWBonf
GADMEC vs BM25 CS/ML 40.2 0.0188 •
GADMEC vs BM25 MatSci 39.2 0.0007 ⋆
BM25 beats pure semantic GADMEC in both
domains under the same budget. The effect is
Bonferroni-significant in MatSci ( pW= 0.0007 )
and significant at uncorrected α= 0.05in CS/ML
4

20 30 40 50 60 70 80 90 100
Win rate over decisive cases (%) — 95\% cluster-bootstrap CIGADMEC vs Greedy
GADMEC vs MMR
GADMEC vs BM25
Random fitness vs Greedy
No sub-aspect vs Greedy
Hybrid vs Greedy
Hybrid vs BM25
Hybrid vs GADMEC (pure sem.)57.458.9
60.161.7
40.239.2
42.937.2
57.455.8
69.365.7
57.043.7
63.563.5Primary
BM25 baseline
Ablations
Hybrid (ζ=0.3)
CS/ML
Materials ScienceGADMEC variant wins
Hybrid variant winsFocal variant losesFigure 2: Main results. Each row is one pairwise com-
parison. Markers: circles = CS/ML, triangles = Materi-
als Science. Colours: Slate Blue = GADMEC variant
wins, Dark Teal = hybrid variant wins, grey = focal
variant loses. Horizontal bars are 95% cluster-bootstrap
confidence intervals (CIs) by combination identifier;
dashed vertical line is the 50% no-effect reference.
(pW= 0.0188 ). Dense pre-filtering therefore does
not subsume lexical signal. Block 4 tests whether
adding that lexical signal to GADMEC’s fitness
recovers the gap.
Block 3 — Ablations (descriptive, addendum-
locked). Table 3 tests whether the BRKGA ma-
chinery alone explains the gains.
Table 3: Ablations against Greedy.
Ablation vs Greedy CS/ML WR MatSci WR
random_fitness 42.9 [36.7, 49.2] 37.2 [29.6, 43.2]
random_fitness + no_constr. 39.8 [34.7, 45.9] 36.7 [29.9, 43.0]
no_subaspect ( ε=0) 57.4 [51.0, 63.5] 55.8 [49.2, 62.8]
Block 4 — Hybrid GADMEC ( ζ= 0.3, ex-
ploratory, locked before judging).
Block 2 shows that lexical signal matters. We
therefore activate the BM25 component of the sur-
rogate fitness ( ζ= 0.3) while leaving the other
fitness weights unchanged. For each question, we
compare the hybrid against Greedy, BM25, and
original semantic GA ( ζ= 0). Table 4 reports the
hybrid comparisons:
Block 4 is a separate family of six compar-
isons. With within-family Bonferroni correction
(α= 0.05/6≈0.0083 ), five of six hybrid com-
parisons pass under wild-cluster inference. TheTable 4: Hybrid GADMEC comparisons.
Hybrid vs Area WR % pWBonf
Greedy CS/ML 69.3 0.0016 ⋆
Greedy MatSci 65.7 0.0021 ⋆
BM25 CS/ML 57.0 0.0007 ⋆
BM25 MatSci 43.7 0.0335 •
Pure semantic GADMEC CS/ML 63.5 0.0012 ⋆
Pure semantic GADMEC MatSci 63.5 0.0015 ⋆
exception is hybrid vs BM25 in MatSci: it passes
at uncorrected α= 0.05, but not Bonferroni, and it
remains directionally a loss(win rate 43.7%).
Taken together, the hybrid reverses the BM25
deficit in CS/ML (hybrid vs BM25 pW= 0.0007 ),
narrows but does not eliminate the BM25 deficit
in MatSci ( −10.8percentage points before, −6.3
after), and improves over both semantic baselines
and pure semantic GADMEC. Pre-registered pre-
dictions for ζ= 0.3(deviations entry 7, locked
before judging) were vs Greedy [55,65]% , vs
BM25 [45,60]% , and vs pure semantic GADMEC
[50,60]% . Observed values met or exceeded the
predicted interval in 5 of 6 cells; the exception was
hybrid vs BM25 in MatSci ( 43.7%), marginally be-
low the [45,60]lower bound. Figure 2 summarizes
Blocks 1–4 with confidence intervals.
5.2 Length-matched robustness
Length matching is a mechanism check, not a sec-
ond primary endpoint. If the advantage is mostly a
length artifact, it should weaken when GADMEC
and Greedy answers have similar character counts.
Bracketed intervals are 95% confidence intervals
(CIs). Table 5 reports the pre-registered ≤200-
character matched bin:
Table 5: Length-matched robustness for GADMEC vs
Greedy.
Subset CS/ML WR [CI] MatSci WR [CI]
All pairs 57.4 [50.0, 64.8] 58.9 [50.5, 66.8]
In matched bin 50.0 [39.8, 60.4] 56.1 [46.4, 65.2]
Out of matched bin 64.4 [56.9, 71.6] 62.2 [51.0, 73.8]
We then vary the bin width as an exploratory
sensitivity sweep (Table 6; Figure 3).
Table 6: Bin-width sensitivity for length matching.
Bin width CS/ML WR % MatSci WR %
≤50 44.8 54.8
≤100 50.9 58.7
≤150 50.0 58.0
≤200 (pre-reg) 50.0 56.1
≤300 53.7 55.5
≤400 53.3 56.3
all 57.4 58.9
5

≤50 ≤100 ≤150 ≤200
(pre-reg)≤300 ≤400 all
|len(ans_GADMEC) − len(ans_Greedy)| bin (chars)3540455055606570GA win rate vs Greedy (%)455150 50545357
555958
56555659CS/ML
MatSciFigure 3: Bin sensitivity. The figure varies the al-
lowed answer-length difference between GADMEC and
Greedy. MatSci remains stable across bin widths, while
CS/ML moves toward parity as the bin tightens; this is
consistent with a less length-entangled MatSci advan-
tage.
MatSci remains in the 54–58 % range across bin
widths. CS/ML moves toward parity and drops to
44.8 % at the tightest bin, where most answers are
too short to differ structurally. This domain split
is exploratory rather than confirmatory. The pre-
registered ≤200-character bin is not Bonferroni-
significant in either area, so the stable MatSci range
should be read as a sensitivity pattern for follow-up
work, not as a settled domain difference.
5.3 Content-distance slicing within matched
bin (exploratory)
Within the matched bin, we ask whether answer
pairs that differ more in content show a stronger
GADMEC signal. Table 7 slices pairs by quar-
tiles of Jaccard similarity between GA and Greedy
answers:
Table 7: Jaccard slices within the matched bin.
Jaccard slice CS/ML WR % MatSci WR %
Q1 (most distinct) 62.5 ( n=24 ) 65.4 ( n=26 )
Q2 45.8 55.6
Q3 66.7 44.4
Q4 (most similar) 25.0 59.3
Across four additional slicing specifications,
MatSci’s most-distinct bucket stays at 60–66 %.
CS/ML is specification-dependent: lexical Jac-
card shows a 62.5 % Q1 win rate, but cosine
Q1 drops to 45.8 %. One plausible reason is
geometric: CS/ML answers occupy a narrower
embedding-space region (cosine range 0.67–0.97)
than MatSci (0.53–1.00), so cosine slices discrimi-
nate less in CS/ML. Figure 4 compares the lexical
Q1 distinct Q2 Q3 Q4 similar20304050607080GA win rate vs Greedy (%)62
4667
2565
56
4459(a) Slice by Jaccard (lexical) within matched bin
CS/ML
MatSci
Q1 distinct Q2 Q3 Q4 similar20304050607080GA win rate vs Greedy (%)465054
5065
52
4859(b) Slice by cosine (semantic) within matched bin
CS/ML
MatSciFigure 4: Content-distance slicing within the matched
bin. Panel (a) groups answer pairs by lexical overlap;
panel (b) groups them by semantic cosine similarity.
MatSci is stable in the most-distinct bucket under both
definitions, while CS/ML depends on the distance mea-
sure.
and semantic-distance slices.
5.4 Ablation discussion
Fitness signal contribution. Randomising the
surrogate fitness while keeping the decoder machin-
ery drops win rate to 42.9 % in CS/ML and 37.2 %
in MatSci. Disabling constraints as well lowers per-
formance slightly further (39.8 %, 36.7 %). Thus
the fitness signal, not merely the BRKGA decoder,
accounts for the 14.5–21.7 percentage-point gap to
full GADMEC.
Sub-aspect coverage. Removing the εsub-
aspect component is inert in CS/ML (57.4 % vs.
57.4 %) and costs 3.1 percentage points in MatSci
(58.9 % to 55.8 %). This is a domain finding rather
than a calibration failure: MatSci appears to benefit
from explicit decomposable-query coverage, while
CS/ML’s remaining gain is more entangled with
length and lexical content distance.
5.5 Inter-judge agreement
We re-judge the four primary tests and the two
hybrid-vs-BM25 comparisons with DeepSeek V4
Pro (thinking mode), a second strong proprietary
judge from a different provider. We use the same
prompt, answer-A/answer-B randomisation, and
tie-handling rule; ties are recorded but excluded
from the win-rate denominator. MatSci primary
usesn= 199 shared pairs (one Opus pair was
a parse-error retry; Opus WR is recomputed on
the matched subset for this section). The two
judges come from different providers (Anthropic
and DeepSeek). Table 8 reports Cohen’s κand
Gwet’s agreement coefficient (AC1):
Cohen’s κranges from moderate (0.535, MatSci
hybrid vs BM25) to substantial (0.745, CS/ML GA
vs Greedy) per Landis and Koch (1977). Gwet’s
6

Table 8: Inter-judge agreement.
Comparison Opus WR DeepSeek WR raw % κ AC1
CS/ML GA vs Greedy 57.4 57.4 87.0 0.745 0.826
CS/ML GA vs MMR 60.1 62.2 85.5 0.705 0.808
MatSci GA vs Greedy 58.7 61.1 84.9 0.693 0.800
MatSci GA vs MMR 62.1 59.4 86.4 0.731 0.819
CS/ML hybrid vs BM25 57.0 52.0 83.0 0.658 0.663
MatSci hybrid vs BM25 43.7 41.7 77.0 0.535 0.694
AC1 is in the substantial-to-almost-perfect range
(0.663–0.826). Win-rate point estimates differ by
at most 5 percentage points, and the directional ver-
dict matches in all six comparisons. This reduces
the risk that the headline depends on one judge
model. It does not replace human gold annotation,
which we leave to follow-up work.
6 Discussion
The results are best read as a mechanism map, not
as a ranking of retrievers. Fixed budgets remove
one structural confound. What remains separates
into three signals: semantic composition, lexical
matching, and answer-length effects.
Composition vs. length under fixed budgets.
Even after matching input budgets and answer caps,
a method can win in several ways. Its evidence may
induce longer answers. It may support denser an-
swers of similar length. Or the judge may still
reward small length differences. The domain split
suggests that these mechanisms should not be col-
lapsed into a single “better retrieval” story.
MatSci: composition-driven (exploratory).
Across seven bin widths and five content-distance
specifications, GADMEC’s matched-bin advan-
tage is stable (54–66 %). In the pre-registered
matched bin, GADMEC is slightly longer more
often than Greedy ( 63vs.44cases at ≤200chars).
This suggests that composition can show up as
length in MatSci, rather than length simply caus-
ing the judged advantage. The reading remains
exploratory: the MatSci CI in the pre-registered
bin is [46.4,65.2], which crosses 50%.
CS/ML: composition entangled with length.
CS/ML behaves differently. The composition sig-
nal appears under lexical content-distance (Q1 Jac-
card 62.5 %) but not under semantic distance (Q1
cosine 45.8 %). CS/ML answers cluster in a nar-
rower embedding region, so semantic-distance slic-
ing has lower discriminating power. The mecha-
nism is more entangled with length in CS/ML than
in MatSci.BM25 as complementary signal. BM25’s dom-
inance over pure semantic GADMEC shows that
lexical signal still matters after dense pre-filtering.
The hybrid beats BM25 in CS/ML and narrows,
but does not close, the MatSci gap. We therefore
treat lexical matching and semantic composition as
complementary signals.
Inference under cluster heterogeneity. The gap
between pVandpWin Block 1 is not a quirk. It
is what happens when cluster correlation is taken
seriously in a small-cluster design. Dual reporting
of vanilla and wild-cluster bootstrap is standard in
econometrics, but rare in NLP evaluation. We rec-
ommend it for cluster-structured NLP benchmarks
where some categories, domains, or sources may
be easier than others. The test choice can change
the headline, so the cost of that choice should be
visible to the reader.
Scope of the proposed standard. The cluster
issue is not specific to this benchmark. Many LLM-
as-a-judge evaluations have natural clusters. MT-
Bench groups 80 multi-turn questions into eight ca-
pability categories; AlpacaEval draws 805 instruc-
tions from five source datasets; MMLU spans 57
subjects in four broad domains; and MTEB spans
eight task families across 58 datasets. Multi-hop
RAG benchmarks such as HotpotQA and 2Wiki-
MultiHop also build questions around bridge or
comparison entities. In each case, some clusters
may be easier, harder, or more favorable to one
method. A binomial p-value over pooled pairs can
therefore overstate confidence. We do not test the
standard on those benchmarks here, but the recom-
mendation is meant for cluster-structured pairwise
judging designs in general.
7 Conclusion
Fixed-budget evaluation makes multi-hop RAG
comparisons easier to interpret. Under matched
evidence budgets and answer caps, pure semantic
GADMEC beats Greedy and MMR directionally
in all four pre-registered tests. Under wild-cluster
bootstrap, however, only one test passes Bonferroni
and two more pass only at uncorrected α= 0.05.
A standard binomial test would have reported all
four as significant.
The same protocol also changes how we read
retrieval signals. BM25 beats pure semantic GAD-
MEC in both domains, showing that lexical match-
ing remains important under the same budget. A
7

lexical-semantic hybrid reverses the BM25 gap in
CS/ML and narrows it in MatSci without clos-
ing it. Length-matched analyses suggest a do-
main split. This claim remains exploratory: in
the pre-registered ≤200-character bin neither area
reaches Bonferroni significance and the MatSci CI
[46.4,65.2]crosses 50%. Code, data manifests,
registrations, deviations, prompts, and judgments
are available in the project repository. Appendix A,
Appendix B, and Appendix C provide descriptive
diagnostics, cost logs, and registration summaries.
8 Limitations
The conclusions are bounded by the evaluation in-
strument. The main limits are the judge, the num-
ber of domains, the exploratory status of several
diagnostics, the small number of clusters, and the
current granularity of the mechanism analysis.
Judge robustness. The primary judge is Claude
Opus 4.7. We replicate the six pairwise compar-
isons that drive the headline (4 primary + 2 hy-
brid vs BM25) with DeepSeek V4 Pro in think-
ing mode, a second strong proprietary judge from
a different provider. Agreement is moderate-to-
substantial across all six comparisons. Cohen’s κ∈
[0.535,0.745] (Landis and Koch, 1977), Gwet’s
AC1∈[0.663,0.826], raw agreement is 77–87%,
and the directional verdict matches in all six (§5.5).
Win-rate point estimates differ by at most 5 percent-
age points. We did not replicate the BM25-only and
ablation comparisons; full multi-judge replication
is future work.
Two domains. The domain-dependent mecha-
nism is based on CS/ML and MatSci only. Replica-
tion on biomedical, legal, or social-science corpora
is needed.
Exploratory analyses. The bin-sensitivity
sweep, content-distance slicing, hybrid ζ= 0.3,
and MatSci ζ= 0.5analyses were added after the
primary results, although their predictions were
locked before judging where applicable. Primary
inference rests on the pre-registered tests. The
MatSci matched-bin CI [46.4,65.2]crosses 50%,
so the strongest MatSci robustness claim should be
read as an exploratory pattern.
Cluster count. We have 10 combinations per
area, which is at the lower edge of what wild-cluster
bootstrap is recommended for ( nclusters ≥20–30
in Cameron et al., 2008). We address this by also re-porting the exact cluster sign-flip permutation test.
It enumerates all 210= 1024 sign assignments and
is unaffected by the small cluster count. The two
tests agree to within 0.005on every primary com-
parison (and within 0.01across all reported com-
parisons), so the Block 1 inference is not driven
by wild-bootstrap asymptotics alone. Replication
on benchmarks with more cluster-level diversity
would improve external validity and precision.
Mechanistic granularity. We do not yet quantify
which chunks GADMEC selects differently from
Greedy. Chunk-overlap and evidence-structure
analyses are needed to isolate the CS/ML resid-
ual mechanism.
Necessity of each standard component. The
proposed measurement standard has four compo-
nents. This paper directly tests two: fixed-budget
design (controlled in the experimental setup; §4)
and cluster-aware inference (Block 1 dual report-
ing plus the permutation check). Pre-registration
with a deviation log and second-judge replication
constrain reporting, but they are not variables we
can ablate after the fact. We therefore do not claim
that removing pre-registration would change con-
clusions in this dataset. Future work should test
each component across multiple benchmarks.
Acknowledgments
Camilo Chacón Sartori acknowledges fund-
ing from R&D Project PID2022-138283NBI00,
funded by MICIU/AEI/10.13039/501100011033
and “FEDER – A way of making Europe”.
References
A. Colin Cameron, Jonah B. Gelbach, and Douglas L.
Miller. 2008. Bootstrap-based improvements for in-
ference with clustered errors. Review of Economics
and Statistics , 90(3):414–427. doi:10.1162/rest.
90.3.414 .
Jaime Carbonell and Jade Goldstein. 1998. The use of
MMR, diversity-based reranking for reordering doc-
uments and producing summaries. In Proceedings
of the 21st Annual International ACM SIGIR Confer-
ence on Research and Development in Information
Retrieval , pages 335–336. doi:10.1145/290941.
291025 .
Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho
Hwang, and Jong C. Park. 2024. Typos that broke
the RAG’s back: Genetic attack on RAG pipeline
by simulating documents in the wild via low-level
perturbations. In Findings of the Association for
8

Computational Linguistics: EMNLP 2024 .doi:10.
18653/v1/2024.findings-emnlp.161 .
Yann Dubois, Balázs Galambosi, Percy Liang, and Tat-
sunori B. Hashimoto. 2024. Length-controlled Al-
pacaEval: A simple way to debias automatic eval-
uators. In Proceedings of the Conference on Lan-
guage Modeling (COLM) .doi:10.48550/arXiv.
2404.04475 .
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2024. RAGAs: Automated eval-
uation of retrieval augmented generation. In Pro-
ceedings of the 18th Conference of the European
Chapter of the Association for Computational Lin-
guistics: System Demonstrations , pages 150–158.
Association for Computational Linguistics. doi:
10.18653/v1/2024.eacl-demo.16 .
José Fernando Gonçalves and Mauricio G. C. Re-
sende. 2011. Biased random-key genetic al-
gorithms for combinatorial optimization. Jour-
nal of Heuristics , 17(5):487–525. doi:10.1007/
s10732-010-9143-1 .
Alex Kulesza and Ben Taskar. 2012. Determinantal
point processes for machine learning. Foundations
and Trends in Machine Learning , 5(2–3):123–286.
doi:10.1561/2200000044 .
J. Richard Landis and Gary G. Koch. 1977. The
measurement of observer agreement for categorical
data. Biometrics , 33(1):159–174. doi:10.2307/
2529310 .
Jeongsoo Lee, Daeyong Kwon, and Kyohoon Jin.
2025. GRADE: Generating multi-hop QA and fine-
gRAined difficulty matrix for RAG evaluation. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2025 .doi:10.18653/v1/2025.
findings-emnlp.236 .
Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad
Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhat-
tacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu,
Kai Shu, Lu Cheng, and Huan Liu. 2025. From gen-
eration to judgment: Opportunities and challenges
of LLM-as-a-judge. In Proceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP) .doi:10.18653/v1/
2025.emnlp-main.138 .
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judg-
ing LLM-as-a-judge with MT-Bench and Chatbot
Arena. In Advances in Neural Information Process-
ing Systems 36 (NeurIPS 2023) .doi:10.48550/
arXiv.2306.05685 .
9

A Supplementary Diagnostics
The appendix is descriptive. It does not introduce
new primary claims; instead, it makes the aggregate
results easier to audit. Figure 5, Figure 6, and Fig-
ure 7 answer three diagnostic questions: whether
the aggregate win rates are driven by a few combi-
nations, whether wins against Greedy and MMR
tend to co-occur, and whether the TOP/NICHO
corpus regime changes the interpretation.
Table 9 adds three logged examples to make
the mechanisms less abstract. These examples are
illustrative only: they are selected from existing
answer-pair and verdict logs, and they do not define
a new hypothesis test.
Logged example Preferred answer adds Mechanism illustrated
CS/ML, GAD-
MEC >Greedy,
q=3Covers reward/Kullback–
Leibler (KL) infringement
control and spatial-aware
compositional generation;
Greedy supports only the
infringement side.Semantic composition
improves sub-aspect
coverage under the same
budget.
CS/ML, BM25 >
GADMEC, q=2Gives details for the two-
stage structural pipeline and
entity-relation mining; pure
semantic GADMEC is cor-
rect but terser.Lexical matching recov-
ers method and con-
trol terms missed by
embedding-only fitness.
CS/ML, Hybrid >
BM25, q=4Retrieves constraint-
feedback and semantic
action/entity control ev-
idence; BM25 lacks the
law-grounded comparison.Lexical signal inside se-
mantic fitness can recover
part of the BM25 gap.
Table 9: Illustrative examples from logged answer pairs
and judge verdicts. The table is descriptive and is not
used for statistical inference. It shows how the aggre-
gate mechanisms in the main text appear in individual
comparisons.
A.1 Per-combination win rates
Figure 5 should be read as a stability check, not as
a separate hypothesis test. The main text reports
cluster-bootstrap intervals over combination identi-
fiers; this figure shows the same source of variation
directly. It also helps identify where a future mech-
anistic analysis should focus: combinations with
high semantic-GADMEC win rates likely contain
evidence sets where coverage and diversity help,
while low-rate combinations are candidates for lex-
ical or length-driven failure modes.
A.2 Joint outcome matrix
Figure 6 gives a more concrete view of what the
aggregate primary tests mean. In both domains,
many questions are wins against both semantic
baselines, but there are also substantial off-diagonal
cells. Thus Greedy and MMR are not redundant
controls: they overlap, but they do not define the
same comparison.A.3 Stratified TOP/NICHO analysis
Figure 7 is included because the corpus construc-
tion deliberately mixes common and niche cross-
subfield combinations. The figure is reassuring
only in a limited sense: it shows no obvious col-
lapse in either regime, but the intervals are wide
enough that the main paper should rely on the
pooled, pre-registered tests rather than regime-
specific claims.
B Cost log
Total judging cost across all conditions (primary
+ BM25 + 3 ablations + hybrid ζ= 0.3+ hybrid
ζ= 0.5MatSci): approximately C82 ($95 USD,
May 23 2026 European Central Bank rate 0.8605
EUR/USD), via the Claude Opus 4.7 application
programming interface (API), billed at $15/1M in-
put tokens and $75/1M output tokens. Per-run logs
are included in results/cost_v3_judging.json
andresults/cost_v3_extras_judging.json .
The cost log is reported for audit, not as a met-
ric. It makes the evaluation scale visible: every re-
ported comparison uses the same evidence budget,
generator, answer cap, judge, and answer-position
randomisation, so the paired design carries the in-
ference rather than any cost normalisation.
C Pre-registration documents
The prepared release archive includes the primary
pre-registration, dated 2026-05-19; the addendum,
dated 2026-05-23; and the running deviation log.
The primary registration fixes the four pairwise
hypotheses, the Bonferroni correction, tie handling,
and cluster-bootstrap reporting before judging. The
addendum records analyses that became necessary
after inspecting the design, including length match-
ing and ablations. The deviation log separates
later descriptive checks, such as bin-width sensi-
tivity, content-distance slicing, and hybrid lexical
weights.
This structure is meant to keep the paper read-
able without blurring evidential status. The main
claims come from registered fixed-budget compar-
isons. The remaining analyses explain where the
signal is stable, where it is domain-dependent, and
which mechanisms should be tested in follow-up
work rather than treated as settled here.
10

vs Greedy vs MMRTOP: detection × transformer
TOP: diffusion × generation
TOP: expressivity × transformer
TOP: prompting × sft
TOP: quantization × transformer
NICHO: agents × gnn
NICHO: distillation × gnn
NICHO: gradient free × meta learning
NICHO: imitation × tools
NICHO: meta learning × quantization40%
(n=20)50%
(n=20)
53%
(n=19)55%
(n=20)
75%
(n=20)63%
(n=19)
65%
(n=20)65%
(n=20)
60%
(n=20)50%
(n=20)
75%
(n=20)75%
(n=20)
42%
(n=19)50%
(n=20)
63%
(n=19)79%
(n=19)
55%
(n=20)60%
(n=20)
45%
(n=20)55%
(n=20)CS/ML
vs Greedy vs MMRTOP: afm × superconductors
TOP: color centers × quantum sensors
TOP: dft × superconductors
TOP: md × thermal cond
TOP: photovoltaics × solar cells
NICHO: ferroelectric × ml potentials
NICHO: heat dissipation × ml potentials
NICHO: magnonics × monte carlo
NICHO: md × phase transformation
NICHO: nonlinear optics × tmds40%
(n=20)42%
(n=19)
63%
(n=19)74%
(n=19)
74%
(n=19)72%
(n=18)
40%
(n=20)60%
(n=20)
60%
(n=20)55%
(n=20)
70%
(n=20)90%
(n=20)
63%
(n=19)50%
(n=20)
75%
(n=20)75%
(n=20)
65%
(n=20)50%
(n=20)
40%
(n=20)50%
(n=20)Materials Science
2030405060708090
GADMEC win rate (%)
Per-combination GADMEC win ratesFigure 5: Pure semantic GADMEC win rate per combination, stratified by domain and corpus regime (TOP/NICHO).
Each panel is one domain; each row is a cross-subfield combination; the two columns report win rates against
Greedy and MMR. The heatmap is a descriptive stability check: it exposes heterogeneity behind the aggregate rates
without adding a separate hypothesis test.
win tie loss
GA vs MMR outcomewin tie lossGA vs Greedy outcome79 0 34
0 1 2
40 1 43CS/ML
win tie loss
GA vs MMR outcomewin tie lossGA vs Greedy outcome82 1 33
0 1 2
39 2 40Materials ScienceJoint distribution of GA outcomes
Figure 6: Joint outcome matrix for GADMEC against
Greedy and MMR, by area. Rows give verdicts against
Greedy and columns give verdicts against MMR. Off-
diagonal cells show where the two semantic baselines
expose different failure modes.
vs Greedy vs MMR40506070GA win rate (%)CS/MLTOP regime NICHO regime
vs Greedy vs MMR40506070Materials Science
Bars: GA WR±95% bootstrap CI per regime. Dashed line: 50% reference.Figure 7: Pure semantic GADMEC win rates strati-
fied by corpus regime. TOP denotes dense topic pairs;
NICHO denotes lower-density pairs. Bars show win
rates and 95% bootstrap intervals within each regime.
11