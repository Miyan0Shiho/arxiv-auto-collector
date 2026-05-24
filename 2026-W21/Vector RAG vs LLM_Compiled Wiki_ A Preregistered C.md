# Vector RAG vs LLM-Compiled Wiki: A Preregistered Comparison on a Small Multi-Domain Research

**Authors**: Theodore O. Cochran

**Published**: 2026-05-18 14:41:16

**PDF URL**: [https://arxiv.org/pdf/2605.18490v1](https://arxiv.org/pdf/2605.18490v1)

## Abstract
We preregistered a comparison of two ways to help an LLM answer questions over a small research corpus: a single-round Vector RAG system and an LLM-compiled markdown wiki. Both systems answered the same 13 questions over 24 papers using the same answer-generating model, and their answers were scored by blinded LLM judges.
  The wiki scored much better at connecting findings across papers, but its advantage in answer organization was not strong after judge adjustment. RAG met the preregistered test for single-fact lookup questions. The clean query-side cost result went against the expected wiki advantage: under the tested setup, the wiki used far more query tokens than RAG, so it could not recover any upfront build cost through cheaper queries.
  Two exploratory analyses changed how we interpret the result. First, claim-level citation checking favored the wiki: its cited pages more often supported the exact claims being made, even though RAG scored better on the overall groundedness rubric. Second, a decomposition-based RAG variant recovered most of the wiki's advantage on cross-paper synthesis at lower LLM-token cost, but it did not recover the wiki advantage in claim-by-claim citation support.
  The main conclusion is that grounded research synthesis is not a single capability. Systems can differ in how well they organize evidence, how well their citations support each claim, and how much they cost to run. In this study, no architecture was best on all three.

## Full Text


<!-- PDF content starts -->

Vector RAG vs LLM-Compiled Wiki:
A Preregistered Comparison on a Small Multi-Domain Research
Corpus
Theodore O. Cochran
AI for Altruism (A4A)
theo@ai4altruism.org
Abstract
We preregistered a comparison of two ways to help an LLM answer questions over a small
research corpus: a single-round Vector RAG system and an LLM-compiled markdown wiki. Both
systems answered the same 13 questions over 24 papers using the same answer-generating model,
and their answers were scored by blinded LLM judges.
The wiki scored much better at connecting findings across papers, but its advantage in
answer organization was not strong after judge adjustment. RAG met the preregistered test for
single-fact lookup questions. The clean query-side cost result went against the expected wiki
advantage: under the tested setup, the wiki used far more query tokens than RAG, so it could
not recover any upfront build cost through cheaper queries.
Two exploratory analyses changed how we interpret the result. First, claim-level citation
checking favored the wiki: its cited pages more often supported the exact claims being made, even
though RAG scored better on the overall groundedness rubric. Second, a decomposition-based
RAG variant recovered most of the wiki’s advantage on cross-paper synthesis at lower LLM-token
cost, but it did not recover the wiki advantage in claim-by-claim citation support.
The main conclusion is that grounded research synthesis is not a single capability. Systems
can differ in how well they organize evidence, how well their citations support each claim, and
how much they cost to run. In this study, no architecture was best on all three.
1 Introduction
An “LLM Wiki” turns a research corpus into a cross-linked wiki at ingest time, and at question
time the model browses that wiki rather than retrieving raw chunks from the original papers.
This architecture was popularized in informal commentary, notably Andrej Karpathy’s “agentic
markdown wiki” framing. The idea suggests three possible tradeoffs: the wiki may help with
multi-paper synthesis; rewriting papers into wiki pages may lose source fidelity; and the wiki may
move cost from query time to ingest time, with a break-even point after enough questions. To our
knowledge no quantitative head-to-head comparison against Vector RAG has been published.1
This paper reports a preregistered, blinded, two-judge comparison of Vector RAG and LLM
Wiki on a fixed 24-paper corpus answering 13 evaluation questions across six difficulty tiers. Both
systems use the same query model (Claude Opus 4.7 at xhigh); the cross-family judge is GPT-5.4
1We searched Google Scholar, Semantic Scholar, arXiv, ACL Anthology, and OpenReview on 2026-05-08 for
combinations of “LLM wiki”, “agentic markdown wiki”, “compiled wiki”, “RAG comparison”, and “retrieval augmented
generation wiki”. We found related hierarchical-, graph-, and summary-memory work (Section 2) but no preregistered
head-to-head comparison of an LLM-compiled markdown wiki against chunk-vector RAG.
1arXiv:2605.18490v1  [cs.CL]  18 May 2026

with Gemini 2.5 Pro for inter-rater reliability. Corpus, question set, rubric, decision rules, and
Bayesian model were locked at the OSF preregistration tag before any judge run.
Contributions.
•Preregistered comparison.The wiki was much better at connecting papers; its advantage
in answer organization was weaker after judge adjustment, positive but below the registered H 1
+2.0threshold (the rubric criteria are inter_paper_mapping and structural_integrity ).
RAG met the preregistered test for single-fact source grounding on groundedness (small-
effect-fragile under IRR adjustment). The expected wiki cost advantage failed in the opposite
direction: the wiki spent about21×more per query than RAG.
•Evaluation-method finding.The two LLM judges behaved differently. Gemini 2.5 Pro often
gave near-perfect scores on holistic criteria, while GPT-5.4 used more of the scale. The judges
agreed most on the most concrete criterion, inter_paper_mapping . A post-hoc claim-level
grounding analysis disagreed with the rubric groundedness score by direction, indicating that
holistic and per-citation grounding metrics measure related but distinct properties.
•Exploratory RAG ablation (single-judge).A decomposition-retrieval variant of RAG
recovered most of the wiki’s advantage on cross-paper synthesis, closing ∼88%of the gap on
the H 1subset and reducing the remaining wiki advantage below the registered+2 .0threshold.
It didnotrecover the wiki advantage in claim-by-claim citation support. The result is a
three-way tradeoff among single-round RAG, decomp-RAG, and wiki along synthesis structure,
claim-citation alignment, and cost.
The five preregistered run artifacts, four post-hoc artifacts, and reproducer scripts are deposited
on OSF / GitHub respectively.
2 Related Work
RAG and multi-hop retrieval.Retrieval-augmented language modeling was introduced by
REALM[Guuetal.,2020]andLewisetal.[Lewisetal.,2020]withdensepassageretrieval[Karpukhin
et al., 2020]. Our Vector RAG is a single-round retrieve-then-generate instantiation with multi-query
expansion, hybrid retrieval, Cohere reranking, and CRAG-inspired [Yan et al., 2024] corrective valida-
tion; it does not perform retrieval-during-generation (cf. IRCoT [Trivedi et al., 2023], FLARE [Jiang
et al., 2023]). Multi-hop benchmarks (HotpotQA [Yang et al., 2018], MuSiQue [Trivedi et al., 2022],
MultiHop-RAG [Tang and Yang, 2024]) document that single-round retrieval is insufficient for
connected reasoning across documents.
Hierarchical and abstractive memory.The LLM Wiki architecture sits closest to compiled
abstractive memory: an offline LLM-built representation queried in lieu of (or alongside) raw
chunks. RAPTOR [Sarthi et al., 2024] builds a recursive cluster-summary tree; RECOMP [Xu
et al., 2024] compresses retrieved documents; GraphRAG [Edge et al., 2024] builds an entity graph
with community summaries; knowledge-graph variants include REANO [Fang et al., 2024] and
GNN-RAG [Mavromatis and Karypis, 2025]. Our wiki is conceptually an LLM-built version of this
family with a markdown-rendered intermediate representation. OpenScholar [Asai et al., 2026] and
PaperQA2 [Skarlinski et al., 2024] apply similar primitives at much greater scale.
2

Citation-faithfulness and LLM-as-judge.ALCE [Gao et al., 2023] introduced citation-
precision/recall benchmarks for long-form answers; Liu et al. [2023a] found commercial generative
search systems often produce fluent but unsupported answers; our claim-level grounding analysis
follows this protocol family. Zheng et al. [2023] document position, verbosity, and self-enhancement
biases in LLM judges; G-Eval [Liu et al., 2023b] reports improved human alignment under structured
form-filling but notes possible bias toward LLM-generated text; Wang et al. [2024] show ranking
can change with response order. Our IRR ceiling-judge finding (Section 5.3) is consistent with this
literature.
3 Methods
Corpus and questions.24 peer-reviewed papers across three domains (8 each: AI ethics &
law, climate science, precision medicine), 2017–2026. 13 evaluation questions across six tiers:
chronological, conflict, multi-hop, emergence, policy ( n= 2each, preregistered as wiki-favoring), and
bias-check ( n= 3, preregistered as RAG-favoring with extra power for H 2). Inclusion: peer-reviewed
or arXiv-preprint research papers; exclusion: review-only summaries, books, non-research artifacts.
Corpus list with DOIs and full question text in the OSF deposit.
Vector RAG.RAG retrieves passages once, reranks and validates them, then writes the answer.
Implementation: document-aware markdown-header chunking with token-limited secondary splitting
(target 512 tokens, 50-token overlap), per-chunk contextual enrichment, multi-query expansion,
dense+sparse hybrid retrieval, Cohere reranking to top-5, CRAG-inspired [Yan et al., 2024] corrective
validation, and single-round answer generation. “Single-round” means no retrieval-during-generation
(cf. [Trivedi et al., 2023, Jiang et al., 2023]); the pipeline is otherwise multi-stage.
LLM Wiki.The wiki system first compiles the papers offline into a persistent cross-linked
markdown wiki (entities, concepts, sources, and analyses with cross-references). At question time, a
tool-using agent lists pages, reads selected pages, and submits an answer via three tools ( list_pages ,
read_page ,submit_answer ;MAX_TURNS =30, no query-side prompt caching). Two minor telemetry
caveats (≤7%thinking-token over-count; ≤6.5%upper-bound caching saving forgone on the query
side) do not materially affect the per-query comparison.
Both systems use the same query model (Claude Opus 4.7 at xhigh), corpus, and evaluation set;
the comparison is fully crossed and within-question paired. This is a comparison of two practical
architectures, not a clean causal isolation of “knowledge organization”: the systems differ along
multiple axes (offline LLM rewriting; compiled markdown vs vector store; tool-loop vs single-round
retrieval; multi-call vs single-call context).
Judging.Primary: GPT-5.4 at mediumreasoning, cross-family from the answer model. IRR:
Gemini 2.5 Pro, same prompt, fixed seed offset. Per-question {RAG, Wiki} →{A, B} blinding via
random.Random(seed=42) (primary) / (seed=43) (secondary). The judge prompt instructs the
model to score each system independently on each criterion. Rubric: four 1–10 anchored criteria with
explicit anchors at 1/5/10: groundedness (claims trace to source material), structural_integrity
(unified narrative vs concatenated fragments), conflict_awareness (naming of contradictory
findings),inter_paper_mapping(multi-hop synthesis across≥2papers).
Preregistration and decision rules (paraphrased).
3

•H1(synthesis): Wiki −RAG≥+2.0on inter_paper_mapping andstructural_integrity
for the multi-hop+emergence tier ( n= 4). Supported if both clear; weakly supported if both
positive but at least one below; refuted if either negative.
•H 2(point-source): RAG−Wiki≥0ongroundednessfor the bias-check tier (n= 3).
•H 3(cost): bothT ingest[wiki]>T ingest[rag]ANDT query[rag]>T query[wiki].
The preregistration is explicit that the primary analyses are directional comparisons of means against
magnitude thresholds, not frequentist tests, because nis small and LLM scores have non-i.i.d. error
structure. Two robustness checks are also preregistered (bootstrap percentile CI; Bayesian model
withµ∼N(0,4),σ∼HalfNormal(4), NUTS).
IRR-triggered adjustment.If the maximum primary–secondary score delta exceeds 2 on any
criterion, that criterion’s mean_tier values are recomputed using the mean of the two judges’
per-question scores in place of the primary-judge-only mean for the H 1/H2tests.
4 Experimental setup
Artifacts.Five preregistered run artifacts plus four post-hoc artifacts (claim-level grounding on
the original run + decomp ablation run, judge, and grounding) are deposited on OSF with SHA-256
hashes; reproducer scripts are on GitHub.Manipulation check.All 26 (question, system) cells used
claude-opus-4-7atxhigh; zero exclusions, zero ingest/query/judge failures.
Cost-accounting scope.H 3follows the preregistered token-accounting (sum of prompt + comple-
tion at ingest; prompt + completion + thinking at query) and excludes embedding spend, reranker
spend, vector-store storage, and dollar normalization.Prompt-cache caveat (load-bearing).The
wiki ingest artifact rolls uncached+cache_creation+cache_read into one field. Anthropic bills
cache reads at∼10%of base input rate, so the gross figure overstates billable cost by an order of
magnitude on cache-heavy workloads. We therefore report H3 only on per-query (H 3b), captured
uncached on both sides; H 3a is not adjudicated from the deposited artifact.
Robustness checks (preregistered).Bootstrap: 95% percentile CI, 10,000 resamples per
criterion×tier.Bayesian: per criterion ×tier,score_diff∼N (µ,σ2)withµ∼N (0,4),σ∼
HalfNormal (4), NUTS (4 chains ×2,000 iter×2,000 warmup, random_seed=42 ); all fits met
the preregistered convergence criteria ( ˆR≤ 1.003, min ESS≥1,907). The Bayesian “strongly
corroborated” threshold isP(µ≥θ)≥0.95(θ= +2for H 1,θ= 0for H 2).
IRR adjustment.The preregistration specified a 3-question IRR subset; during execution we
extended secondary judging to all 13 questions, a disclosed rigor-increasing deviation. The max-delta
trigger fires on all four criteria under both the prereg n= 3and the expanded n= 13coverage; the
adjustment rule is invariant to the deviation.
Post-hoc claim-level grounding.For each of the 26 answer cells, Claude Opus 4.7 (medium
adaptive thinking) atomizes the answer into atomic claims with cited_source_idx ; for each cited
claim, GPT-5.4 (medium, cross-family) scores against the cited chunks as supported /partial
/contradicted /unsupported . Uncited claims auto-receive unsupported . Full prompts in the
OSF supplement.
4

5 Results
We report each hypothesis under theprimary-onlyreading (the pre-IRR baseline) and theIRR-
adjustedreading (judge-mean recomputation; see Section 3). The trigger and adjustment rule
were both preregistered. The IRR-adjusted reading is the more conservative of the two and is the
basis for our headline verdicts.
5.1 Confirmatory analyses
H1: Synthesis advantage ( n= 4).Per-question paired diffs (Wiki −RAG) on the multi-hop +
emergence subset:
structural_integrity inter_paper_mapping
Question primary secondary avg primary secondary avg
T3-mia-as-copyright-evidence+2 0 +1.0 +8 +7 +7.5
T3-rwd-validity-for-side-effects+1 0 +0.5 +2 +4 +3.0
T4-cross-domain-audit+1 +2 +1.5 +6 +8 +7.0
T4-eelgrass-mrv-gaps+4 +3 +3.5 +9 +9 +9.0
mean+2.000+1.250+1.625 +6.250+7.000+6.625
Primary-only:∆ struct = +2.000(lands exactly at threshold),∆ mapping = +6.250(clears).
Both criteria≥+2.0⇒supported.IRR-adjusted (headline):∆ struct = +1.625(misses by
0.375),∆ mapping = +6.625(clears)⇒weakly supportedper the preregistered three-way deci-
sion rule. The inter_paper_mapping advantage is large and never close to the threshold; the
structural_integrityadvantage is threshold-sensitive.
H2: Point-source parity ( n= 3).Per-question paired diffs (RAG −Wiki) on groundedness
for the bias-check tier:
Question primary secondary avg
B1-devote3-confidence-interval+1 +3 +2.0
B2-he-tyka-equilibration-ratios+3−6−1.5
B3-mia-attack-success-rate+2 +1 +1.5
mean+2.000−0.667+0.667
The B2 secondary entry is the largest single judge disagreement in the experiment: GPT-5.4
scored RAG=9 / wiki=6, Gemini 2.5 Pro scored RAG=4 / wiki=10. Both readings nonetheless
return∆≥0, satisfying the preregistered decision rule.Primary-only:∆ = +2 .000⇒supported.
IRR-adjusted:∆ = +0 .667⇒supported. H 2is supported on the registered rubric metric; Section 5.4
reports a post-hoc analysis that nuances the underlying mechanism.
H3: Cost asymmetry.The preregistration expected the wiki to be expensive to build but
cheap to query. The clean query-side data showed the opposite: the wiki cost more per query,
so the amortization story cannot hold. The query-side test (H 3b) is directly adjudicable from
per-query telemetry; the ingest-side test (H 3a) isnot adjudicated from the deposited artifactdue to
a prompt-cache accounting issue we discovered post-hoc (see Section 7).
5

Quantity (sum across 13 questions) Value
Tquery[rag](prompt + completion + thinking)78,093
Tquery[wiki](prompt + completion + thinking)1,651,357
H3b supported/parenleftbig
Tquery[rag]>T query[wiki]/parenrightbig
False(21×opposite direction)
The wiki query-side harness explicitly disabled prompt caching (companion §4.3); the per-query
figures above are uncached input + completion + thinking and are billable-equivalent. RAG
per-query is similarly uncached. The21×per-query gap is unambiguous.
H3as a conjunctive hypothesis requires both H 3a and H 3b to be supported, so H 3b’s opposite-
direction result alone is enough to refute it. The wiki cost intuition was that pre-compilation
pays a large up-front fee in exchange for cheaper queries, breaking even at some N. The opposite
happened: the user paysmoreper query, not less, so the preregistered wiki-amortization mechanism
is impossible under the observed per-query costs. The preregistered crossover-queries formula’s
denominator( Tquery[rag]−Tquery[wiki])/13 =−121,020is negative; the formula returns a negative
N, mathematically encoding “no positive-query crossover”.
We do not quote an ingest cost ratio. The wiki-side ingest telemetry aggregated uncached +
cache_creation +cache_read tokens into a single tokens_in field; Anthropic bills cache reads at
∼10%of base input rate, so summing them at face value over-counts billable cost by an order
of magnitude or more on cache-heavy workloads. Without a four-column breakdown (uncached /
cache_create / cache_read / output) we cannot reconstruct billable cost.H3a is not adjudicated
from the deposited artifact.The H3 verdict rests on H3b alone, which is conjunctively sufficient.
5.2 Robustness checks (preregistered)
Table 1: Bootstrap percentile CI (10,000 resamples) and preregistered Bayesian posterior ( µ∼
N(0,4),σ∼HalfNormal (4), NUTS) for each H 1/H2cell. “Strong” marks P(µ≥θ )≥0.95per the
preregistered Bayesian threshold (H 1θ= +2; H 2θ= 0).
Test Reading mean bootstrap 95% CI posteriorµ P(µ≥θ)strong?
H1struct judge-avg+1.625 [+0.75,+2.88] +1.500 0.283no
H1mapping judge-avg+6.625 [+4.13,+8.50] +5.5890.967yes
H1struct primary+2.000 [+1.00,+3.25] +1.819 0.440no
H1mapping primary+6.250 [+3.50,+8.50] +5.060 0.938close
H2ground judge-avg+0.667 [−1.50,+2.00] +0.522 0.660no
H2ground primary+2.000 [+1.00,+3.00] +1.808 0.937close
Only H 1inter_paper_mapping (judge-avg) clears the preregistered “strongly corroborated”
bar ofP(µ≥θ )≥0.95. The Bayesian posterior shrinks point estimates toward zero (e.g., H 1
primary structural_integrity sample mean+2 .000shrinks to posterior+1 .819), reflecting the
regularising prior at small n. The two robustness checks agree on direction but disagree onhow
oftenthe threshold is robustly cleared.
5.3 Inter-rater reliability: a ceiling-judge finding
The preregistered IRR rule fired on all four criteria; max-deltas across the expanded n= 13
secondary coverage are 6 ( groundedness ), 4 ( structural_integrity ), 8 ( conflict_awareness ),
6

6 (inter_paper_mapping ). Computed instead on the preregistered n= 3IRR subset, max-deltas
are 6 / 4 / 4 / 3, still triggering on all four criteria.
The disagreement is largely aboutlevel, notranking. The two judges agreed on the sign of
every criterion-level gap on the H 1confirmatory subset and on three of four 13-question criterion-
level means. groundedness is the exception: across all 13 questions GPT-5.4 rates RAG higher
(7.00 vs 6.31), while Gemini 2.5 Pro rates wiki higher (8.15 vs 9.69). The H 2subset (bias-check
groundedness ) shows the same level of fragility: primary∆ = +2 .000, secondary∆ = −0.667,
judge-avg∆ = +0 .667, supported under the registered decision rule but the secondary judge alone
would refute. This reinforces the interpretation that holistic groundedness is the criterion most
sensitive to judge calibration.
Table 2: Per-judge per-system means across all 13 questions. Gemini 2.5 Pro ratings on Wiki
saturate near 10/10 except on inter_paper_mapping , the only criterion with a concrete operational
definition.
Judge RAG gwiki gRAG swiki sRAG cwiki cRAG mwiki m
GPT-5.4 (primary) 7.00 6.31 6.85 8.92 6.77 8.38 5.38 9.46
Gemini 2.5 Pro (secondary) 8.159.698.159.858.549.625.2310.00
Gemini 2.5 Pro ratings on the wiki output saturate near 10/10 across three of four criteria. GPT-
5.4 ratings spread between 6 and 9 with greater variance. The exception is inter_paper_mapping :
both judges’ RAG means differ by 0.15 (5.38 vs 5.23) and both judges’ wiki means by 0.54
(9.46 vs 10.00). The two judges agree most closely on the criterion with the most concrete
operational definition. We surface as a transferable methodological finding:rubric criteria with
concrete operational definitions show inter-judge convergence; criteria depending on holistic stylistic
assessment show ceiling-judge calibration drift, consistent with prior LLM-as-judge literature [Zheng
et al., 2023, Liu et al., 2023b, Wang et al., 2024].
5.4 Post-hoc claim-level grounding
We ran a post-hoc claim-level grounding analysis on all 466 atomic claims.This analysis is not
preregisteredand is reported as exploratory robustness on the H 2mechanism.
Important interpretive caveat.For RAG claims, the cited evidence is the original PDF chunk
that the retrieval system surfaced; for wiki claims, the cited evidence is a wiki-page excerpt, which is
itself a compilation of source PDFs and not an original PDF passage. The analysis below therefore
testsevidence-artifact claim alignment(does each claim follow from what the system cited?),not
original-source fidelity(does each claim follow from the underlying PDF?). A reader who wants
source fidelity for wiki must additionally ask: does the wiki page itself accurately reflect the source
PDF? We do not measure this; Section 7 flags it as the primary follow-up.
Aggregate by system.Total claims: RAG 150, Wiki 316. Cited rate (claims with non-empty
cited_source_idx ): RAG 88.0%, Wiki 76.3%. Wiki produces about twice as many claims per
answer (24.3 vs 11.5) and cites a smaller fraction of them.
Treating uncited claims as unsupported (all-claimsview), the headline rates are: RAG 16.7%
supported / 42.0% unsupported; Wiki 30.7% / 28.5%. The conditional-on-cited view (citation-
precisionsense) is:
7

Verdict (conditional on cited) RAG (n= 132) Wiki (n= 241)
supported18.9%(macro18.4%)40.2%(macro37.3%)
partial 44.7% 53.1%
contradicted 2.3% 0.4%
unsupported34.1%(macro35.8%)6.2%(macro8.0%)
Macro-averages by question (each of 13 answers weighted equally regardless of claim count) are
reported alongside the micro-averages to address the concern that long wiki answers might inflate
wiki’s per-claim rate. The qualitative conclusion is invariant: wiki cited claims are ∼2×more often
supported and∼4–5×less often unsupported than RAG cited claims.
Of 4 directly-contradicted claims in the experiment, 3 are RAG (B1: HR/CI mismatch between
cited table and answer; T4-cross-domain: feature-drift direction inverted; T5-glp1: false description
of paper content) and 1 is Wiki (T2-oae: “in 95%” rendered as “over 95%”, a single-word paraphrase
artefact).
By tier.The largest gap is on thebias-check tier, H 2’s home turf: rubric∆ groundedness =−2.00
(RAG ahead, the registered metric) but claim-level supported% is 5.3% RAG vs 51.9% Wiki. The
two metrics point in opposite directions on the same tier (Table 3).
Table 3: Claim-level grounding rates by tier, conditional on cited.†Rightmost column: rubric
groundednessdelta as Wiki−RAG (positive = Wiki ahead on rubric).
Tier RAG sup% Wiki sup% RAG unsup% Wiki unsup% rubric∆ ground†
bias-check5.3% 51.9% 78.9%7.4%−2.00
chronological 8.0% 50.8% 60.0% 1.6%−0.50
conflict 22.7% 42.9% 9.1% 8.6%+2.00
emergence 25.0% 37.5% 30.0% 0.0%−1.00
multi-hop 32.0% 35.6% 20.0% 11.1%−3.50
policy 19.0% 22.0% 9.5% 9.8%+1.50
Absolute-rate caveat.These rates should not be read as high absolute citation reliability: even
the wiki’s cited claims are more often partial than strictly supported (53.1% vs 40.2% across all
13 questions). The result is a relative architecture comparison under a strict supported/partial
boundary, not a benchmark-level claim of high citation precision.
Reading.The two metrics measure related but distinct properties. The rubric judge evaluates
holisticanswer-evidence trace and rewards RAG’s short, citation-heavy answers anchored to verbatim
chunk text. The claim-level analysis evaluatesstrict per-citation matchingand catches RAG’s
tendency to retrieve a chunk and then synthesize/extrapolate beyond it, including the B1 confidence-
interval error, where the chunk reports HR=1 .37but the answer says HR=1 .38. The observed
pattern is consistent with wiki compilation pre-positioning evidence in claim-shaped artifacts (so
when the answer cites a wiki page, the cited page typically contains the supporting sentence directly),
although we do not measure header/claim alignment as an independent variable; this is not the
same as source fidelity.
This does not invalidate the preregistered H 2verdict.H 2was a registered test on rubric
groundedness and that verdict (RAG ≥Wiki) holds on the registered metric. It does materially
nuance the underlyingmechanism: the prereg conjectured that wiki’s lossy summarization would
8

lose point-source fidelity; under claim-level scrutiny, RAG’s chunk-retrieve-and-synthesize pipeline
shows worseevidence-artifact alignmentthan wiki’s compilation-and-traverse pipeline does. Whether
wiki’s compilation step preserves source fidelity at a higher rate than RAG’s retrieval step is a
separate question that this analysis does not adjudicate.
The methodological lesson:rubric-style holistic grounding scores and claim-level citation align-
ment can disagree by direction, not just magnitude. RAG/Wiki evaluations should report both.
5.5 Other preregistered exploratory analyses
The preregistration listed seven secondary analyses planned but not adjudicative. We summarize
here; full data in the OSF supplement.
•Bias-tag stratification.The question taxonomy holds: bias=rag questions show RAG winning
groundedness by 2.0; bias=wiki questions show Wiki winning all four criteria (largest on
inter_paper_mappingat+5.75).
•Latency.RAG total 3.3 min, Wiki total 22.0 min (6 .6×); the wall-clock ratio is sublinear in
tokens because adaptive thinking overlaps with input processing.
•Adaptive-thinking engagement.RAG triggered nonzerothinking_tokenson 2/13 questions;
Wiki on 13/13. Both at the samexhighflag: task shape, not flag, drives engagement.
•Cost-vs-quality scatter.RAG cluster ( ≤8k tokens, scores 3.75–8.25) and Wiki cluster ( ∼30k–
210k tokens, 6.75–9.25) are not globally dominated. On 4 of 13 questions RAG matches or
beats Wiki on rubric mean at an order-of-magnitude lower cost (T1-cardio, T3-rwd, B2, B3).
The Pareto frontier is task-dependent.
6 Exploratory decomp-RAG ablation
A reviewer hypothesis flagged that single-round retrieval is a known weakness on multi-hop synthe-
sis [Trivedi et al., 2023, Jiang et al., 2023], so wiki’s H1 advantage might be partly an artefact of
the registered RAG configuration. We test this with a post-hoc decomposition-retrieval ablation.
Method.For each question: Claude Opus 4.7 decomposes it into 2–5 sub-questions; the existing
single-round RAG pipeline (Section 3) runs per sub-question, keeping cited sources only; cited chunks
are deduplicated; Claude Opus 4.7 generates the final answer using thesameanswer-generation
system prompt as the single-round baseline (verbatim from the codebase). Same query model,
corpus, vector store, and reranker; only the question-to-retrieval-context step changes. Three
predictions were committed before running:P1decomp closes ≥50%of the wiki–single-RAG gap
oninter_paper_mapping ;P2per-query token cost between5–10 ×single-round;P3claim-level
supported-rate improves on multi-hop. GPT-5.4 was re-invoked with the same prompt, rubric, and
seed (42) on the (decomp, wiki) pair.The decomp ablation was scored only by GPT-5.4;
we did not rerun Gemini 2.5 Pro. Given that the original analysis triggered a full IRR adjustment
from judge calibration drift (Section 5.3),decomp results below should be read as exploratory and
judge-sensitive, not as a preregistered re-test.
Three-way result. Cost (P2).Decomp’s per-query LLM-token cost (under the H3 token-
accounting scope, excluding reranker and embedding spend) is6 .3×single-round and3 .4×cheaper
than wiki, satisfying P2. Wall-clock isslowerthan wiki despite fewer tokens, because decomp
9

Table 4: Three-way comparison summary. Cost rows sum across the 13 questions. Synthesis-shape
rows use the H1 confirmatory subset ( n= 4); the Reading column reports the within-judge-run
shrinkage from the original judge run to the decomp judge run. Claim-level row uses cited claims
across all 13 questions.aSingle-RAG, scored in the original judge run.bDecomp-RAG, scored in
the post-hoc decomp judge run with the same prompt, rubric, and seed.cWiki: range across the
two judge runs (same answers, scored slightly differently).
Metric SingleaDecompbWikicReading
Total query tokens 78k491k1,651k decomp3.4×cheaper
than wiki on LLM to-
kens
Total wall-clock (min) 3.326.822.0 decomp slower than
wiki (sequential)
H1inter_paper_mappingmean 3.509.009.00–9.75 wiki gap shrinks
+6.25→+0.75
H1structural_integritymean 7.008.758.92–9.00 wiki gap shrinks
+2.00→+0.25
Cited claim supported % 18.9 19.2 40.2 decomp doesnot
close citation gap
performs sequential retrieval-then-rerank-then-validate calls per sub-question; the LLM-token picture
is favorable for decomp, the wall-clock picture is not, and the embedding/reranker cost picture is
not measured in this study.
Synthesis structure (P1).On the H 1confirmatory subset, the decomp run supports P1.
The wiki advantage shrinks from+6 .25to+0.75on inter_paper_mapping (about88%of the gap
closed) and from+2 .00to+0.25on structural_integrity (about87.5%). Against the decomp
baseline, both criteria fall below the preregistered+2 .0threshold; the wiki would not clear the
registered H 1thresholds in this comparison.This does not alter the preregistered H 1verdict against
single-round RAG. It indicates that the synthesis-structure advantage is sensitive to the RAG baseline.
Decomp also outperforms wiki on holistic groundedness (+1.15across all 13 questions;+3 .00on
the multi-hop tier specifically) and satisfies the H 2parity test on bias-check groundedness (+2 .33).
Claim-level grounding (P3 partial).We re-ran the §5.4 atomize-and-score pipeline on
the 232 decomp claims (94.4% cited, vs 88.0% single, 76.3% wiki). Decomp’s overall cited-claim
supported-rate is 19.2%, essentially equal to single-round (18.9%), and roughly half wiki’s (40.2%).
The decomp claim-level advantage over single-round is concentrated on bias-check (5 .3→43.5%,
fixing chunk-recombination errors like the B1 confidence-interval mistake) and modestly on multi-hop
(32.0→37.0%). On chronological / conflict / emergence / policy, decomp is similar to or slightly
lower than single-round.
Reading.The pattern is consistent with two separable mechanisms underlying the wiki advantage.
Retrieval coverageis substantially mitigated by decomposition retrieval in this ablation: when
RAG breaks each question into sub-questions, it recovers most of the wiki’s advantage on cross-paper
synthesis.Representation alignmentisnotclosed by broader retrieval: the wiki still does better
at citing pages that directly support the specific claim being made. One plausible explanation is
that wiki pages pre-position evidence in claim-shaped artifacts, while RAG chunks are aligned to the
query rather than to each downstream answer claim. We do not directly measure this mechanism.
Future work combining iterative retrieval with claim-grounded post-hoc citation repair would test
10

whether the alignment gap can be closed too.
Caveats summary.Single-judge (no IRR for the ablation); cross-run wiki absolute scores differ by
≤0.6on any criterion (within-judge-run deltas in Table 4 are robust to this); n= 4on the H 1subset.
Because the ablation is exploratory and small- n, we report point estimates only and treat them as
mechanism evidence rather than confirmatory inference; one decomposition prompt was fixed before
the run. Reproducer:experiments/run_decomp_rag.py+run_claim_grounding_decomp.py.
7 Discussion and limitations
What the findings mean.The wiki advantage appears to decompose into two separable
mechanisms. Theretrieval-coverageportionofthewiki’ssynthesis-shapeadvantageissubstantially
mitigated by decomposition retrieval in our ablation (Section 6). Therepresentation-alignment
portion (the claim-level citation precision criterion) isnotclosed by broader retrieval. The observed
pattern is consistent with wiki compilation pre-positioning evidence in claim-shaped artifacts (so
when the answer cites a wiki page, the cited page typically contains the supporting sentence directly),
while RAG retrieval aligns chunks with thequery, not with each specific claim the answer ends up
making. We do not measure header/claim alignment as an independent variable; this is a mechanism
hypothesis consistent with the data, not a demonstrated causal account.
Karpathy’s three intuitions are partially borne out. (i) Pre-compilation helps with synthesis
(supported on the registered baseline; substantially mitigable with decomp). (ii) Pre-compilation
may sacrifice point-source fidelity (supported on the rubric; reversed under claim-level scrutiny).
The relevant comparison there is evidence-artifact alignment, not original-PDF fidelity, which we
do not measure. (iii) Pre-compilation shifts cost from query to ingest with finite-N amortization
(refuted: per-query goesup, not down).
When to pick which.Single-round RAG for cost-sensitive point-source retrieval. Decomp-RAG
for multi-paper synthesis where structure matters and the budget can absorb ∼6×single-round
LLM-token cost (excluding the additional reranker/embedding spend from multiple sub-question
retrievals); tested at one operating point in this study. Wiki for evidence-artifact claim-citation
alignment, despite the ∼21×per-query token premium, pending source-PDF validation. H 3’s
per-query result rules out wiki at scale on per-query-cost grounds regardless of N.
What the findings do not show.No human evaluation(the most consequential gap): all rubric
scoring, claim-level scoring, and decomp judging are LLM-based; the IRR finding (Section 5.3) shows
LLM-judge calibration drift is large and direction-flipping on holistic criteria. Human validation is
the highest-priority future work; until then the claim-level analysis should be read as LLM-scored
evidence-artifact alignment, not human-validated source fidelity. The §6 decomp ablation should
similarly be read as exploratory and judge-sensitive (single-judge, no IRR).Small n(n= 4for H 1,
n= 3for H 2): standard errors are large and the decomp ablation reports point estimates only.
Single corpus and single query model: 24 papers, frontier LLM; transfer to large-corpus or different
model families unverified.Style confound: wiki produces continuous prose, RAG bullet-and-cite;
not controlled for.One decomposer promptfixed before the run, not iterated.Cost telemetry
under prompt caching: the wiki ingest tokens_in field aggregates uncached + cache_creation +
cache_read at face value; cache reads bill at ∼10%of base input rate, so the gross figure overstates
billable cost by an order of magnitude. The H 3b verdict is unaffected (query-side caching was
disabled on both sides).Methods note for replicators:cached-vs-uncached cost-comparison studies
11

must capture four columns per LLM call (uncached / cache_creation / cache_read / output), not a
rolled-uptokens_in.
Beyond this study.Future work should compare claim-level grounding against original PDF
passages (testing source fidelity for wiki) rather than against the cited evidence artifact only;
run Gemini 2.5 Pro on the decomp pair to add IRR coverage; test retrieval-during-generation
methods [Trivedi et al., 2023, Jiang et al., 2023] that may close more of the wiki advantage; and
test hybrid wiki+source-RAG architectures that could backstop wiki’s rare paraphrase artefacts (1
of 4 contradicted claims) with source-chunk verification.
8 Conclusion
The most defensible conclusion of this preregistered comparison is not that compiled wiki memory
beats RAG, nor the reverse. Grounded research synthesis is not a single capability: a system
can organize evidence well, cite evidence well for each specific claim, or run cheaply, and in this
experiment no architecture did all three best. Single-round RAG minimizes cost; decomp-RAG
approaches the wiki on synthesis-shape rubric criteria at ∼3.4×lower per-query LLM-token cost;
wiki retains the strongest LLM-scored evidence-artifact claim-citation alignment, at high per-query
cost and pending source-PDF validation. Future small-corpus RAG/Wiki evaluations should report
synthesis structure, claim-citation alignment, and cost separately rather than collapsing them into a
single grounded-synthesis verdict.
References
Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, et al. Synthesizing scientific literature with
retrieval-augmented language models.Nature, 2026.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, and Jonathan Larson. From local to global: A Graph RAG approach to query-focused
summarization, 2024. arXiv preprint, Microsoft Research technical report.
Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald. REANO: Optimising retrieval-augmented
reader models through knowledge graph generation. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (ACL), 2024.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate
text with citations. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing (EMNLP), 2023.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-
augmented language model pre-training. InProceedings of the 37th International Conference on
Machine Learning (ICML), 2020.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen tau Yih. Dense passage retrieval for open-domain question answering. InProceed-
ings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP),
2020.
12

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems (NeurIPS), 2020.
Nelson F. Liu, Tianyi Zhang, and Percy Liang. Evaluating verifiability in generative search engines.
InFindings of the Association for Computational Linguistics: EMNLP, 2023a.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-Eval: NLG
evaluation using GPT-4 with better human alignment. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing (EMNLP), 2023b.
Costas Mavromatis and George Karypis. GNN-RAG: Graph neural retrieval for efficient large
language model reasoning on knowledge graphs. InFindings of the Association for Computational
Linguistics (ACL Findings), 2025.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D.
Manning. RAPTOR: Recursive abstractive processing for tree-organized retrieval. InInternational
Conference on Learning Representations (ICLR), 2024.
Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J.
Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, and Andrew D. White. Language
agents achieve superhuman synthesis of scientific knowledge (PaperQA2), 2024. arXiv preprint,
FutureHouse technical report.
Yixuan Tang and Yi Yang. MultiHop-RAG: Benchmarking retrieval-augmented generation for
multi-hop queries. InConference on Language Modeling (COLM), 2024.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop
questions via single-hop question composition.Transactions of the Association for Computational
Linguistics, 2022.
HarshTrivedi, NiranjanBalasubramanian, TusharKhot, andAshishSabharwal. Interleavingretrieval
with chain-of-thought reasoning for knowledge-intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023.
Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu
Liu, and Zhifang Sui. Large language models are not fair evaluators. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (ACL), 2024.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. RECOMP: Improving retrieval-augmented LMs
with context compression and selective augmentation. InInternational Conference on Learning
Representations (ICLR), 2024.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation
(CRAG), 2024. arXiv preprint.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing (EMNLP), 2018.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. InAdvances in Neural Information
Processing Systems (NeurIPS), 2023.
13