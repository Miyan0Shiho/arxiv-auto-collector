# Answer Presence Drives RAG Rewriting Gains

**Authors**: Yuejie Li, Yueying Hua, Ke Yang, Li Zhang, Yueping He, Yueping He, Ruiqi Li, Bolin Chen, Tao Wang, Bowen Li, Chengjun Mao

**Published**: 2026-06-04 03:00:42

**PDF URL**: [https://arxiv.org/pdf/2606.05633v1](https://arxiv.org/pdf/2606.05633v1)

## Abstract
Retrieval-augmented QA pipelines often route retrieved passages through an LLM \emph{rewriter} before a smaller reader, lifting F1 by tens of points on multi-hop benchmarks; this gain is typically credited to improved evidence quality. We ask whether that lift is causally driven by the gold answer string appearing in the rewritten context rather than by curation per se, using a controlled intervention audit. For each rewritten context we re-run the reader after one of four controlled edits to the compile output: removing the gold answer span, replacing a length-matched random non-answer span (placebo), or injecting the gold into rewrites where it was absent (at the prefix or at a midpoint sentence boundary). Across twelve completed (cell, baseline) intervention runs spanning three reader families (Qwen2.5-7B, Qwen3.5-35B, GLM-4.7), two datasets (HotpotQA, 2WikiMultihopQA), and three compiler arrangements (MA-only, MB-only, MA$+$verify), removing the gold answer drops reader F1 by $28$ to $64$ points beyond the length-matched placebo on paired \texttt{answer-in-compile} strata, and prepending the gold into rewrites that lacked it raises F1 by $+0.7$ to $+9.7$ points in $10$ of $12$ (cell, baseline) combinations. A companion five-sentinel audit shows the conventional single-\texttt{[MASK]} probe is itself sentinel-fragile: on 2Wiki it reports a $+4.12$~F1 ``non-leakage residual'' that flips to $-3.33$ to $-7.81$~F1 under four alternative sentinels and fails an equivalence test for three of those four ($1/4$~pass). We do not propose a new rewriter or mitigation; we release the intervention runner and the sentinel panel so that other rewriter-gain claims can be tested against the same standard.

## Full Text


<!-- PDF content starts -->

Answer Presence Drives RAG Rewriting Gains
Yuejie Li*†Yueying Hua*Ke Yang*Li Zhang*Yueping He*
Ruiqi Li*Bolin Chen*Tao Wang*Bowen Li*Chengjun Mao*
Ant Group
{liyuejie.lyj, huayueying.hyy, zhulang.yk, lier.zl, heyueping.hyp,
liruiqi.lrq, bolin.cbl, taoran.wt, zhikong.lbw, chengjun.mcj}@antgroup.com
Abstract
Retrieval-augmented QA pipelines often route
retrieved passages through an LLMrewriter
before a smaller reader, lifting F1 by tens of
points on multi-hop benchmarks; this gain
is typically credited to improved evidence
quality. We ask whether that lift is causally
driven by the gold answer string appearing
in the rewritten context rather than by cura-
tion per se, using a controlled intervention
audit. For each rewritten context we re-run
the reader after one of four controlled edits
to the compile output: removing the gold an-
swer span, replacing a length-matched ran-
dom non-answer span (placebo), or injecting
the gold into rewrites where it was absent (at
the prefix or at a midpoint sentence bound-
ary). Across twelve completed (cell, baseline)
intervention runs spanning three reader fami-
lies (Qwen2.5-7B, Qwen3.5-35B, GLM-4.7),
two datasets (HotpotQA, 2WikiMultihopQA),
and three compiler arrangements (MA-only,
MB-only, MA +verify), removing the gold an-
swer drops reader F1 by 28to64points be-
yond the length-matched placebo on paired
answer-in-compile strata, and prepending
the gold into rewrites that lacked it raises F1 by
+0.7 to+9.7 points in 10of12(cell, baseline)
combinations. A companion five-sentinel audit
shows the conventional single- [MASK] probe
is itself sentinel-fragile: on 2Wiki it reports a
+4.12 F1 “non-leakage residual” that flips to
−3.33 to−7.81 F1 under four alternative sen-
tinels and fails an equivalence test for three of
those four ( 1/4pass). We do not propose a new
rewriter or mitigation; we release the interven-
tion runner and the sentinel panel so that other
rewriter-gain claims can be tested against the
same standard.
1 Introduction
Retrieval-augmented question answering (RAG)
pipelines increasingly route retrieved passages
*All authors contributed equally.
†Corresponding author:liyuejie.lyj@antgroup.comthrough a stronger LLMrewriter—a compiler, sum-
marizer, or compressor—before a smaller reader
produces the final answer (Lewis et al., 2020; Gao
et al., 2023; Asai et al., 2024). On multi-hop bench-
marks the rewriter lifts reader F1 by tens of points,
and the lift is typically credited to improved evi-
dence quality: better organization, denoising, multi-
hop chaining. We ask a more basic question: how
much of the lift is caused by the gold answer string
being surfaced in the rewritten context, rather than
by curation alone? In these same multi-hop settings,
the rewriter also surfaces the gold answer string in
roughly 80% of records, so the two explanations—
curation and answer-string surfacing—are observa-
tionally entangled in the aggregate F1 gains used
to justify the pipeline.
The conventional way to break this entangle-
ment is to substring-mask the gold answer in the
rewritten context with a sentinel token, most com-
monly [MASK] , and re-run the reader: a collapse
to raw-retrieval F1 is taken as evidence of answer-
string leakage, while a significantly positive resid-
ual is taken as evidence of a non-leakage channel.
We show in §4 that this single-sentinel probe is it-
self unreliable. On 2WikiMultihopQA the [MASK]
leaves a +4.12 F1 residual over raw retrieval,
but under four alternative sentinels ( [REMOVED] ,
a natural-language deletion phrase, a generic word,
and a symbol string) on the same paired examples,
the residuals instead range from −3.33 to−7.81
F1, and the equivalence criterion is met for only
one of the four sentinels; the apparent residual is
largely a [MASK] -token artifact. A masking diag-
nostic that can flip sign with the choice of sentinel
cannot, by itself, separate answer-string surfacing
from genuine evidence curation.
We therefore replace single-sentinel masking
with a controlled intervention audit. For each
rewritten context we re-run the reader under
four controlled edits:removethe gold answer
span, replace a length-matched random non-
1arXiv:2606.05633v1  [cs.AI]  4 Jun 2026

answer span as aplacebo, orinsertthe gold
answer string into rewrites that lack it, either
at the prefix or at a midpoint sentence bound-
ary. The remove-minus-placebo contrast on the
paired answer-in-compile stratum is a direct,
on-distribution estimate of the causal dependence
of reader F1 on the gold answer string being
present. On the complementary subset, insertion
tests whether restoring the gold answer string re-
covers F1.
Across twelve evaluated (cell, baseline) interven-
tion runs, spanning three reader families (Qwen2.5,
Qwen3.5, GLM), two datasets (HotpotQA, 2Wiki-
MultihopQA), and three compiler configurations
(MA-only, MB-only, MA +verify), removing the
gold answer drops reader F1 by 28to64points
beyond the length-matched placebo, and prepend-
ing the gold to rewrites that lacked it raises F1 by
+0.7 to+9.7 points in 10of12(cell, baseline)
combinations.
We make three contributions. First, we present
the first controlled answer-presence intervention
audit for compile-then-read RAG. The remove /
placebo / insert design yields a remove-minus-
placebo F1 drop of 28to64points across twelve
(cell, baseline) intervention runs, and reveals that
insertion effects depend on position. Second, we
give a negative result for single-sentinel masking
diagnostics: in a five-sentinel audit on 2WikiMulti-
hopQA, the positive [MASK] residual reverses un-
der all four alternative sentinels. Third, we release
a reusable audit kit, including an intervention run-
ner and a sentinel panel, so that future rewriter-gain
claims can be tested against a common standard.
§2 defines the setup, §3 specifies the audit protocol,
§4 reports the results, and §5 discusses what the
interventions identify and what they do not.
2 Setup
Pipeline.A QA question qis answered from a
long retrieved context Cqby a reader Main one
of four settings:B 1raw retrieval (reader sees Cq);
B2MA-only compile ( MA(C q, q));B 3MB-only
compile, with MB a different model family from
MA;B4MA compile then MB-verify, which may
rewrite unsupported sentences.
Cells.The audit is run on four (reader, compiler-
family, dataset) cells:S1(Qwen2.5-7B / Qwen2.5-
72B / HotpotQA),S2(Qwen2.5-7B / Qwen2.5-72B
/ 2Wiki),S3(GLM-4.7 / GLM-5 / HotpotQA),S5
(Qwen3.5-35B / Qwen3.5-27B / HotpotQA), withDeepSeek-V3 as MB in every cell. Each cell ex-
ercises B 1–B4on the same 1,000 -question subset.
The suite labels follow our internal run IDs: S4
is a verifier-variant pilot reported only in the ap-
pendix and is not part of the main answer-presence
intervention grid. The four cells cover three reader
families, two datasets, and two MA families, so
that no single contrast in §4 is identified by a single
(reader, compiler) combination.
Datasets and decoding.HotpotQA distractor
split (Yang et al., 2018) and 2WikiMultihopQA
(Ho et al., 2020) are both multi-hop benchmarks
whose distractor pools already contain every gold
supporting paragraph for every evaluated query, so
compile gain cannot be attributed to compensating
for missing retrieved evidence. Records with gold
strings shorter than two characters are excluded.
Token-level F1 is computed against the original
gold (Yang et al., 2018). Across cells, the compile
output surfaces the gold answer string in roughly
80% of records; this answer-surfacing rate is the
observational entanglement the intervention audit
is designed to break. Reader: temperature=0.01 ,
max_tokens=512 ; compilers: temperature=0.2 ,
max_tokens=2048.
3 Audit Protocol
The audit has two layers. Thecausal-intervention
layer (§3.1) is the main test of whether gold-answer
presence in the rewritten context causally drives
reader F1. Thesentinellayer (§3.2) controls a sep-
arate concern: the same intervention can mislead if
its implementation (e.g. the choice of mask token)
leaks exploitable structure to the reader.
3.1 Causal Interventions
Motivation.Measuring how reader F1 changes
when the rewriter is added is observational: in a
typical cell, B 2both re-organises retrieved evidence
and surfaces the gold answer string into the rewrit-
ten context. To estimate the on-distributioncausal
effect of answer presence we must edit the rewrit-
ten context to add or remove the gold while holding
everything else as close to constant as possible, and
we must distinguish that edit’s effect from the effect
ofanyedit of the same size.
Design.For each B 2/B3/B4compile output c
we apply one of four edits:removeevery case-
insensitive match of the gold answer in cwith
[MASK] ;placeboreplace a length-matched random
2

non-answer span (deterministic seed 1729 ) with
[MASK] ;insert_prependprepend “ Note: <gold>. ”
toc;insert_midinsert “ <gold>. ” at the midpoint
sentence boundary. The remove/placebo edits tar-
getans_in_compile=1 ; the insert edits target
ans_in_compile=0.
Stratification.Each record is tagged with two bi-
nary flags: ans_in_b1∈ {0,1} (the gold appears
in raw Cq) and ans_in_compile∈ {0,1} . The
compile output moves a record into one of four tran-
sition buckets, (ans_in_b1,ans_in_compile)∈
{0→0,0→1,1→1,1→0} . Theremoveand
placebointerventions apply only where
ans_in_compile= 1 ; theinsertinterven-
tions apply only where ans_in_compile= 0 .
Within each (cell, baseline, intervention) we
report the F1 mean and a paired ∆against
the unperturbed compile output, computed by
bootstrap (1,000resamples, seed42,95%CI).
Identification.Bothremoveandplaceboedit
a span of identical word count from the same
rewriter output and differ only in whether the
deleted content is the gold answer. We therefore
read∆causal = ∆ remove−∆ placebo on the 1→1 stra-
tum ( ans_in_b1=ans_in_compile=1 ), where
both perturbations apply and the gold is already
retrievable from raw context, and interpret it as
the average treatment effect of gold-answer pres-
ence on reader F1 in 1→1 contexts. Positive ∆
on the 0→0 insertbuckets is the complementary
estimate of how much F1 the rewriter would have
provided had it surfaced the answer. Because both
arms write [MASK] intocand differ only in whether
the masked span is the gold, the common sentinel-
token main effect cancels in ∆remove−∆ placebo ,
so the sentinel-fragility concern of §3.2 applies to
single-sentinel leakage residuals but not to ∆causal
itself.
3.2 Sentinel-Fragility Audit
Motivation.A sentinel the reader can ex-
ploit would produce a spurious “non-leakage
residual” in the older mask-and-see literature
and, to the extent that its effect interacts with
the surrounding context rather than acting
only as a common additive shift, could leak a
second-order term into ∆causal even after the
main-effect cancellation of §3.1. We audit this
separately on the ans_in_compile=1 stratum of
cells S1 and S2, following the sentinel-ablationprotocol in scripts/preregistration_99c_
sentinel_ablation.md.
Design.Five replacement tokens are applied to
the same paired stratum:MASK [MASK] (the con-
ventional choice),REMOVED [REMOVED] (brack-
eted sentinel without standard placeholder se-
mantics),NATURAL“the answer was removed”
(natural-language deletion),WORDthing(generic
noun), andSYMBOL ### (symbol string). A
length-matched PLACEBO replaces a random non-
answer span of equal word count with[MASK].
Equivalence criteria. C2a (sentinel equiva-
lence).A non- [MASK] sentinel passes if its paired
∆against [MASK] has CI containing zero, or |∆|<
1.0F1, or |∆|<0.20× the original [MASK] -vs-
B2effect on the same stratum; the audit passes
if≥3/4 alternative sentinels pass, otherwise the
[MASK] residual is judged sentinel-fragile.C2b
(placebo).The B 2-vs-PLACEBO paired ∆has CI
containing zero or |∆|<0.50× the original effect,
confirming that any masking-side F1 collapse is
answer-specific rather than perturbation-generic.
4 Results
Causal ∆on the 1→1 stratum.Table 1 reports
the headline quantity: for each (cell, baseline), the
paired remove and placebo ∆s on the 1→1 stratum,
and their difference ∆causal = ∆ remove−∆ placebo .
Across the twelve (cell, baseline) intervention
runs—three baselines per reader for S1, S2, S3,
S5—removing the gold answer drops reader F1 by
37to65points, the length-matched placebo drops
F1 by only 0to13points (and is mildly positive
in S3 and S5), and ∆causal ranges from −28.2 (S1,
B2) to−64.1 (S3, B 2) F1. All twelve ∆causal values
have the same sign and exceed 25F1 in magnitude.
The S5 cell (Qwen3.5-35B reader) also has a mean
∆placebo that is mildlypositive( +1.9 to+4.2 F1),
which strengthens the contrast: deleting a same-
sized non-answer span is not on average harmful
in that cell, yet deleting the gold answer collapses
F1 by tens of points.
Insertion in the 0→0 bucket.Prepending
“Note: <gold>. ” to compile outputs thatlacked
the gold answer raises reader F1 by a positive ∆
in10/12 cell–baseline combinations (range +0.7
to+9.7 F1, with S1 and S5 clustered near +8to
+10 in B 2/B3runs and S3 showing a smaller +2
to+6effect). Inserting the same string at the mid-
point sentence boundary instead of the prefix gives
3

Cell Bn pair ∆rm ∆pl∆causal [95% CI]
S1 B 2598−40.7−12.5−28.2[−31.1,−25.3]
S1 B 3549−44.2−11.4−32.8[−36.1,−29.7]
S1 B 4565−37.3−8.6−28.7[−31.9,−25.6]
S2 B 2808−44.9−12.6−32.3[−34.9,−29.8]
S2 B 3834−39.6−6.3−33.3[−35.6,−31.1]
S2 B 4788−38.8−8.1−30.7[−33.3,−28.2]
S3 B 2745−65.3−1.2−64.1[−67.2,−61.4]
S3 B 3700−48.6 +0.7−49.4[−52.7,−46.1]
S3 B 4737−60.5 +0.3−60.8[−64.0,−57.8]
S5 B 2822−37.1 +3.8−41.0[−43.7,−38.1]
S5 B 3754−44.3 +4.2−48.5[−51.5,−45.5]
S5 B 4766−47.3 +1.9−49.1[−52.4,−46.0]
Table 1: Causal-intervention audit on the 1→1 stra-
tum ( ans_in_b1=ans_in_compile=1 ).∆rmand∆pl
are paired F1 deltas (perturbed −original compile)
for the remove and placebo interventions respectively;
∆causal= ∆ rm−∆ plis the same-stratum causal estimate,
with95% paired bootstrap CI ( 1,000 resamples, seed
42) over the per-qid difference on the npairqids where
both arms applied. All values are F1 percentage points
(×100).
a mostly non-positive ∆(range −13.3 to+5.5 F1,
9/12 negative), so the reader’s use of an injected
gold is position-sensitive: prefix-injected gold lifts
F1 in the direction that removing the gold lowered
it, but mid-context injection does not.
Sentinel-fragility (companion).Table 2 reports
the audit on S1 and S2. On S1, all five sentinels
collapse post-mask F1 below raw retrieval and 4/4
alternatives pass C2a (sentinel-robust). On S2 the
[MASK] sentinel reports a +4.12 F1 “non-leakage
residual” that the four alternatives invert to between
−3.33 and−7.81 F1 (1/4pass), judging the resid-
ual a [MASK] -token artifact. Both cells pass C2b.
Because ∆causal in Table 1 is computed against
the matched placebo and not against the [MASK] -
residual, it does not inherit this sentinel-token ex-
posure.
Identity sanity check.We first rule out re-
run instability as an explanation. When the
gold answer is absent from the compiled context
(ans_in_compile=0 ), theremoveedit changes
nothing. Rerunning the reader on these unchanged
contexts reproduces the original compile F1, with
median per-question |∆|=0.000 on both datasets.
The intervention deltas are therefore not artifacts
of a second reader call.HotpotQAN=4772WikiN=829
Condition F1∆ B1 F1∆ B1
B1 raw 23.93 — 17.52 —
B2 compile 54.38+30.4562.15+44.64
[MASK]19.95−3.98∗21.63+4.12∗
[REMOVED]15.08−8.85∗13.21−4.31∗
NATURAL 17.25−6.67∗11.16−6.36∗
WORD 14.86−9.07∗9.71−7.81∗
SYMBOL 17.34−6.59∗14.19−3.33∗
PLACEBO†50.64−3.74∗60.22−1.93∗
C2a sentinel-equiv.PASS (4/4) FAIL (1/4 pass)
C2b placeboPASS PASS
Table 2: Sentinel-fragility audit on cells S1 and
S2 (paired ans_in_compile=True strata), token-F1
×100 .∗marks 95% bootstrap CI excluding zero;†the
PLACEBO ∆column is reported vs B 2so that C2b di-
rectly compares perturbation to compile. C2a passes if
≥3/4 alternative sentinels have ∆vs[MASK] with95%
CI containing zero, |∆|<1.0 F1, or |∆|<0.20×
the original [MASK] -vs-B 2effect; C2b passes if the
B2-vs-PLACEBO ∆has95% CI containing zero or
|B2−PLACEBO|<0.50× the same effect. This table
reports the sentinel-layer audit; the causal layer is Ta-
ble 1.
5 Discussion
Across twelve (cell, baseline) intervention runs, the
remove-minus-placebo ∆causal in Table 1 lies in
[−64.1,−28.2] F1 with the same sign in every cell;
even accepting the upper-end placebo collapse as
residual confounding, the gold answer is a neces-
sary input to the bulk of the F1 lift the rewriter deliv-
ers on these multi-hop benchmarks. The 0→0 inser-
tion results (prefix positive in 10/12 (cell, baseline)
combinations, midpoint mostly non-positive: 9/12
negative) are the symmetric statement: when the
rewriter fails to surface the gold, prefix-injecting it
recovers between +0.7 and+9.7 F1, a nontrivial
fraction of the ∼30 F1 B 2lift. We do not interpret
these as the entire story—rewriters plausibly also
de-clutter evidence—but they bound how much of
the lift can be credited to “curation quality” without
further controlled evidence: most of it cannot. The
sentinel-fragility audit (Table 2) rules out a simpler
“do the older masking diagnostic bigger” rebuttal:
on 2Wiki the [MASK] probe’s positive “non-leakage
residual” flips under four alternative sentinels and
fails C2a, so the design’s reliance on a remove-vs-
placebo paired contrast—rather than on a [MASK] -
residual vs raw retrieval—is necessary. For deploy-
ment audits, claims of a non-answer-string compile
channel should be paired with both layers (sentinel
4

+ placebo on the masking side; remove vs. placebo
on the ans_in_compile=1 stratum); the released
audit kit ( scripts/p0_intervention.py and the
sentinel runner) supplies the standard.
6 Related Work
Leakage and perturbation in QA/RAG.Train–
test overlap and exploitation of contaminated data
have been documented and debated in open-domain
QA (Lewis et al., 2021; Magar and Schwartz,
2022; Sainz et al., 2023); in RAG, entity pertur-
bation (Longpre et al., 2021), when-does-retrieval-
help analyses (Mallen et al., 2023; Wen et al.,
2024), and noise-sensitivity audits (Cuconasu et al.,
2024; Yoran et al., 2023) characterize context-vs-
parametric reliance. Counterfactual data (Kaushik
et al., 2019), contrast sets (Gardner et al., 2020),
behavioral testing (Ribeiro et al., 2020), and
distractor-sensitivity probes (Shi et al., 2023) pro-
vide the methodological tradition our placebo con-
dition continues. The missing piece our paper sup-
plies is an in-passageremove-vs-placebointerven-
tion on the rewriter’s output, on the same paired
stratum, identifying the causal effect of answer-
string surfacing.
Sentinel sensitivity and statistical practice.
Prompt- and format-fragility is well-attested (Sclar
et al., 2024; Webson and Pavlick, 2022; Lu et al.,
2022; V oronov et al., 2024; Liu et al., 2024);
Liao et al. (2022) treats [MASK] as an information-
gathering token in pre-training, the only close
neighbour to our sentinel-flip result. On the sta-
tistical side, we follow standard rigour arguments
(Card et al., 2020; Dodge et al., 2019; Dror et al.,
2018); paired bootstrap CIs adapt Koehn (2004),
and the equivalence thresholds follow TOST (Lak-
ens, 2017). We combine these tools into a single
audit for one specific RAG failure mode.
Limitations
The intervention audit covers four (reader, com-
piler, dataset) cells—three reader families, two
datasets, three compiler arrangements—over
twelve (cell, baseline) intervention runs. We do
not claim ∆causal magnitudes transfer outside this
grid. The remove/insert interventions are string-
level: aliases, paraphrasings, and entity-mediated
cues are not edited, so a rewriter that consistently
restates the gold in different words would still pass
the remove condition (paraphrastic leakage is notdetected). Our scope is restricted toonline, per-
queryrewriting; in offline LLM-curated corpora
(Gunasekar et al., 2023; Allal et al., 2024; Maini
et al., 2024; Su et al., 2025; Long et al., 2024) the
rewriter is query-agnostic and cannot selectively
surface a specific gold span, so our magnitudes do
not transfer, though we view analogous answer-
removal/placebo controls as a reasonable precon-
dition for attributing such gains to curation quality.
We do not propose a new rewriter, mitigation, or
alias-aware masking scheme; the contribution is
diagnostic.
Reproducibility.Exact prompts, decoding pa-
rameters, model endpoint identifiers, and the
suite-ID to log-file mapping are listed in Ap-
pendix M. The intervention runner and sen-
tinel panel are released as part of the audit kit
(scripts/p0_intervention.py and the sentinel-
ablation runner).
References
Loubna Ben Allal, Anton Lozhkov, and Daniel van
Strien. 2024. Cosmopedia: how to create large-scale
synthetic data for pre-training.Hugging Face Blog,
page 56.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
InInternational conference on learning representa-
tions, volume 2024, pages 9112–9141.
Dallas Card, Peter Henderson, Urvashi Khandelwal,
Robin Jia, Kyle Mahowald, and Dan Jurafsky. 2020.
With little power comes great responsibility. In
Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 9263–9274.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. InProceedings of the 47th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 719–729.
Jesse Dodge, Suchin Gururangan, Dallas Card, Roy
Schwartz, and Noah A Smith. 2019. Show your
work: Improved reporting of experimental results. In
Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pages 2185–2194.
Rotem Dror, Gili Baumer, Segev Shlomov, and Roi
Reichart. 2018. The hitchhiker’s guide to testing
statistical significance in natural language processing.
5

InProceedings of the 56th annual meeting of the
association for computational linguistics (volume 1:
Long papers), pages 1383–1392.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6465–6488.
Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan
Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi,
Dheeru Dua, Yanai Elazar, Ananth Gottumukkala,
and 1 others. 2020. Evaluating models’ local deci-
sion boundaries via contrast sets. InFindings of the
Association for Computational Linguistics: EMNLP
2020, pages 1307–1323.
Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio
César Teodoro Mendes, Allie Del Giorno, Sivakanth
Gopi, Mojan Javaheripi, Piero Kauffmann, Gus-
tavo de Rosa, Olli Saarikivi, and 1 others. 2023.
Textbooks are all you need.arXiv preprint
arXiv:2306.11644.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. InProceedings of the 28th International Con-
ference on Computational Linguistics, pages 6609–
6625.
Divyansh Kaushik, Eduard Hovy, and Zachary C Lipton.
2019. Learning the difference that makes a differ-
ence with counterfactually-augmented data.arXiv
preprint arXiv:1909.12434.
Philipp Koehn. 2004. Statistical significance tests for
machine translation evaluation. InProceedings of
the 2004 conference on empirical methods in natural
language processing, pages 388–395.
Daniël Lakens. 2017. Equivalence tests: A practical
primer for t tests, correlations, and meta-analyses.So-
cial psychological and personality science, 8(4):355–
362.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Patrick Lewis, Pontus Stenetorp, and Sebastian Riedel.
2021. Question and answer test-train overlap in open-
domain question answering datasets. InProceedings
of the 16th Conference of the European Chapter of
the Association for Computational Linguistics: Main
Volume, pages 1000–1008.
Baohao Liao, David Thulke, Sanjika Hewavitharana,
Hermann Ney, and Christof Monz. 2022. Mask more
and mask later: Efficient pre-training of masked lan-
guage models by disentangling the [mask] token. InFindings of the Association for Computational Lin-
guistics: EMNLP 2022, pages 1478–1492.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the association
for computational linguistics, 12:157–173.
Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao
Ding, Gang Chen, and Haobo Wang. 2024. On llms-
driven synthetic data generation, curation, and evalu-
ation: A survey. InFindings of the Association for
Computational Linguistics: ACL 2024, pages 11065–
11082.
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.
2021. Entity-based knowledge conflicts in question
answering. InProceedings of the 2021 conference on
empirical methods in natural language processing,
pages 7052–7063.
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel,
and Pontus Stenetorp. 2022. Fantastically ordered
prompts and where to find them: Overcoming few-
shot prompt order sensitivity. InProceedings of the
60th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
8086–8098.
Inbal Magar and Roy Schwartz. 2022. Data contamina-
tion: From memorization to exploitation. InProceed-
ings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers),
pages 157–165.
Pratyush Maini, Skyler Seto, Richard Bai, David Grang-
ier, Yizhe Zhang, and Navdeep Jaitly. 2024. Rephras-
ing the web: A recipe for compute and data-efficient
language modeling. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 14044–
14072.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st annual meeting of
the association for computational linguistics (volume
1: Long papers), pages 9802–9822.
Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin,
and Sameer Singh. 2020. Beyond accuracy: Behav-
ioral testing of nlp models with checklist. InProceed-
ings of the 58th annual meeting of the association for
computational linguistics, pages 4902–4912.
Oscar Sainz, Jon Campos, Iker García-Ferrero, Julen
Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre.
2023. Nlp evaluation in trouble: On the need to
measure llm data contamination for each benchmark.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 10776–10787.
6

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane
Suhr. 2024. Quantifying language models’ sensitiv-
ity to spurious features in prompt design or: How i
learned to start worrying about prompt formatting.
InInternational Conference on Learning Representa-
tions, volume 2024, pages 25055–25083.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed H Chi, Nathanael Schärli,
and Denny Zhou. 2023. Large language models can
be easily distracted by irrelevant context. InInter-
national Conference on Machine Learning, pages
31210–31227. PMLR.
Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings,
Brandon Norick, Markus Kliegl, Mostofa Patwary,
Mohammad Shoeybi, and Bryan Catanzaro. 2025.
Nemotron-cc: Transforming common crawl into a
refined long-horizon pretraining dataset. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 2459–2475.
Anton V oronov, Lena Wolf, and Max Ryabinin. 2024.
Mind your format: Towards consistent evaluation of
in-context learning improvements. InFindings of
the Association for Computational Linguistics: ACL
2024, pages 6287–6310.
Albert Webson and Ellie Pavlick. 2022. Do prompt-
based models really understand the meaning of their
prompts? InProceedings of the 2022 conference
of the north american chapter of the association for
computational linguistics: Human language tech-
nologies, pages 2300–2344.
Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao
Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu,
Wendy Gao, Jiaxin Xu, and 1 others. 2024. Bench-
marking complex instruction-following with multiple
constraints composition.Advances in Neural Infor-
mation Processing Systems, 37:137610–137645.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2023. Making retrieval-augmented language
models robust to irrelevant context.arXiv preprint
arXiv:2310.01558.
7

Supplementary Material
This appendix collects supplementary experiments
around the controlled intervention audit reported
in the main text. None is required for the main-
text tables, but together they provide the wider
evidence base that motivated and constrains it.
The sections cover: pipeline-level numbers — full
main-suite results (§A), reader-scale attenuation
(§B), and cross-rewriter sweeps (§C); mechanism
probes — verification-mode ablations (§D), length-
controlled regressions on answer preservation (§E),
the V 5-vs-Control length-quartile breakdown (§F),
per-question oracle headroom and the selection
gap (§G), and the question-blind containment au-
dit (§H); and descriptive / reproducibility material
— operator-by-reader pattern (§I), the full answer-
mask ablation and dashboard cross-walk (§J), the
2Wiki alias spot-check (§K), prompts (§L), and
model endpoints with run-ID and log-file mapping
(§M).
Suite heterogeneity warning.The appendix col-
lates experiments that differ from the main-text
answer-mask cells along several axes; their abso-
lute numbers are not interchangeable with the main
text. The main-text S1/S2 results (Table 2) use
the B 2compile rewriter on a paired 1,000 -question
subset of each dataset with the Qwen2.5-7B reader
attemperature=0.01 . The appendix departs from
this baseline in three ways: (i) the scale sweep (§B)
and the length / oracle / OLS analyses (§§E–G) sub-
stitute the V 5quote-first rewriter and run on the full
N=7,405 HotpotQA / N=1,005 Qasper sets; (ii)
the cross-rewriter sweep (§C) additionally swaps
the rewriter model; (iii) the verification-mode ab-
lations (§D) add a verifier stage. The resulting
differences in rewriter setting, sample, and reader
produce different absolute F1s (e.g. raw HotpotQA
= 23.0 on the main-text 1k subset but 65.13 on the
cross-rewriter full set), so appendix tables should
be read for their within-cell deltas, not against main-
text numbers. The S1/S2/S3/S5 cells in §A corre-
spond to the suites used in the main-text interven-
tion audit; the masked-ablation tables in §J share
the S1/S2 answer-mask baseline.
A Full Main-Suite Pipeline Results
Table 3 reports the full pipeline F1 across all suites
we ran (S1–S5 on HotpotQA/2Wiki, Q1/Q2 on
Qasper). The short-paper main text uses S1 (Qwen-
7B / HotpotQA), S2 (Qwen-7B / 2Wiki), S3 (GLM-Suite Dataset Reader B 1 B2 ∆F1
S1 HotpotQA Qwen2.5-7B 0.230 0.502+0.272
S2 2Wiki Qwen2.5-7B 0.153 0.540+0.387
S3 HotpotQA GLM-4.7 0.737 0.721−0.016
S4 HotpotQA GLM-4.7∗0.726 0.705−0.021
S5 HotpotQA Qwen3.5-35B 0.399 0.477+0.078
Q1 Qasper Qwen2.5-7B 0.383 0.367−0.015
Q2 Qasper Qwen2.5-72B 0.403 0.378−0.025
Table 3: Main-suite pipeline effects across reader
regimes.∗S4 uses a GLM-5 verifier in place of
DeepSeek-V3. Compile (B 2) helps a small Qwen2.5
reader on multi-hop QA (S1/S2), is near-null or negative
for a strong GLM reader (S3/S4), helps a Qwen3.5-
family reader (S5), and slightly hurts both 7B and
72B Qwen2.5 readers on Qasper (Q1/Q2). Num-
bers from full-suite logs; logs/S{1..5}_*.jsonl ,
logs/Q{1,2}_*.jsonl.
4.7 / HotpotQA), and S5 (Qwen3.5-35B / Hot-
potQA) for the intervention audit, and S1, S2 for
the sentinel-fragility companion. The remaining
suites (S4 verifier pilot, Q1, Q2 on Qasper) doc-
ument that the per-cell rewriter effect can shrink
toward zero or flip sign as the reader strengthens
or the dataset changes; this is the broader empiri-
cal context for the sentinel fragility we report on
HotpotQA vs. 2Wiki.
B Reader-Scale Attenuation
To isolate a within-family reader effect, we hold the
rewriter setting fixed (V 5quote-first, Qwen2.5-72B
rewriter) and sweep Qwen2.5 reader scale from
0.5B to 72B on both HotpotQA and Qasper. Table 4
reports, at each reader scale, the paired F1 delta
between the rewriter output and raw retrieval.
Two consequences for the main-text intervention.
First, the attenuation curve in HotpotQA explains
why the main-text mask_applied=True stratum
compile lift (Table 2, +30.45 F1 for Qwen2.5-7B)
is much larger than the equivalent lift for a 72B
reader (a near-null +1.03 F1 from the sweep): the
audit verdicts are therefore specific to the 7B reader
cell, and we do not extrapolate them across reader
scales. Second, the Qasper column shows that
a uniform compile policy already fails the sign-
monotonicity test on a third dataset; the answer-
mask intervention is not performed on Qasper be-
cause its free-form, abstractive answers do not ad-
mit clean substring masking (Limitations).
8

Reader scale HotpotQA∆F1 Qasper∆F1
(V5−Raw) (V5−Raw)
Qwen2.5-0.5B+0.1966 +0.0257
Qwen2.5-1.5B+0.1465−0.0118
Qwen2.5-3B+0.2121 +0.0105
Qwen2.5-7B+0.0931−0.0154
Qwen2.5-14B+0.0260−0.0562
Qwen2.5-32B+0.0133−0.0071
Qwen2.5-72B+0.0103−0.0250
Table 4: V 5quote-first rewriter setting held fixed while
Qwen2.5 reader scale is swept. HotpotQA: positive
at every scale; effect broadly attenuates with reader
strength (non-monotonic at the small end: 3B peaks at
+0.212 above 0.5B’s +0.197 ). Qasper: the effect is neg-
ative at every scale from 1.5B through 72B except 3B.
The sign of the rewriting effect is determined by dataset
×reader rather than by the rewriter alone. HotpotQA:
N=7,405 per scale; Qasper: N=1,005 per scale; both
paired.
C Cross-Rewriter Sweep: Stronger
Rewriter Helps but Does Not Close the
Reader-Dataset Gap
One alternative account of the negative Qasper lifts
in Appendix B is that the rewriter is underpowered.
We test this by swapping Qwen2.5-72B-Instruct
for Qwen3-235B-A22B-Instruct-2507 under the
same question-conditioned Control rewriter setting
and the same reader (Qwen2.5-7B). The stronger
rewriter improves both datasets: on HotpotQA,
Control rewrite F1 moves from 74.86 to75.50
(+10.37 over the raw baseline of 65.13 ); on Qasper,
from 38.50 to40.94 (+2.69 over38.25 ). Rewriter
scale is not irrelevant. With the same reader, how-
ever, the HotpotQA / Qasper gap persists: Hot-
potQA remains a high-gain setting and Qasper re-
mains a low-gain one, ruling out the strongest ver-
sion of a writer-only account in the range we can
measure. The mediating factor must lie outside
rewriter capacity alone — which is consistent with
the dataset-dependent residual reported in the main
text under a single rewriter.
D Verification-Mode Ablations
The main-text intervention grid covers B 2, B3, and
B4. For completeness, this section ablates the veri-
fication mode applied to the compiled context: Ta-
ble 5 reports three downstream verification variants
— hard rewriting (B 4, the verifier may delete or
rewrite sentences from the compile output), soft an-
notation (B 4s, the verifier tags but does not modify),
and label-only verification (B 4lr, the verifier classi-Suite Variant EM F1 Hall%
AblF, Qwen2.5B2compiler-only0.287 0.49919.1
B4hard rewrite 0.227 0.440 18.3
B4ssoft annotate 0.148 0.356 18.4
AblF2, Qwen3.5B2compiler-only 0.266 0.480 14.9
B4hard rewrite0.353 0.57815.8
B4ssoft annotate 0.254 0.45913.3
AblE3, Qwen2.5B2compiler-only0.270 0.48719.7
B4hard rewrite 0.224 0.43117.2
B4lrlabel-only 0.265 0.473 18.2
Table 5: Verification-mode ablations on HotpotQA.
AblF and AblF2 contrast Qwen2.5 versus Qwen3.5
reader families; AblE3 isolates the label-only variant.
Soft annotation (B 4s) is worse than hard rewriting (B 4)
in both reader families: keeping more text is not what
helps. Label-only verification (B 4lr) recovers most of
the hard-rewrite loss against compiler-only, i.e. much
of the verifier damage comes from rewriting compiler
sentences rather than from judging them. Hallucination
rates are LLM-judge diagnostics, not the basis for the
claim.N=1,000per row.
fies each compile-output sentence as supported or
unsupported but copies it verbatim either way).
These ablations are reported here only as the
broader experimental context within which the
answer-mask intervention sits: they do not them-
selves identify answer surfacing as the operative
mechanism. That identification is made by the
main-text intervention together with the legibility
regression in Appendix E.
E Length-Controlled Regression of
Per-Question F1 on Answer Surfacing
As a correlational complement to the main-text in-
tervention, we regress per-question F1 on (i) the
rewriter setting (Baseline / Control / V5), (ii) log
rewriter-output length, (iii) an answer-in-rewrite in-
dicator (substring presence of any normalised gold
alias), and setting ×length interactions, with cluster-
robust SEs on qid. Table 6 and Table 7 report the
coefficients for HotpotQA and Qasper across two
readers each. The answer-in-rewrite indicator car-
ries a coefficient of +0.29 to+0.35 atp <10−50
in every cell, an order of magnitude larger than any
setting-only coefficient (all |ˆβsetting| ≤0.09 ). This
is the correlational counterpart to the main-text in-
tervention: when the answer is in the rewrite the
reader scores about 30 F1 points higher per ques-
tionat the same rewrite length, regardless of which
rewriter setting produced it.
9

Predictor Coef SEp
Reader = Qwen2.5-7B-Instruct(n=22,215,R2=0.094)
Setting=Control−0.05900.03670.108
Setting=V5+0.01730.03520.624
log(len)−0.05790.01811.4×10−3
Answer-in-rewrite+0.34640.01194.6×10−187
Control×log(len)−0.03850.02140.072
V5×log(len) +0.04920.02160.023
Reader = Qwen3-8B(n=22,215,R2=0.087)
Setting=Control−0.08990.03540.011
Setting=V5−0.01960.03430.567
log(len)−0.06850.01759.2×10−5
Answer-in-rewrite+0.32970.01239.7×10−158
Control×log(len)−0.01880.02080.364
V5×log(len) +0.06600.02071.4×10−3
Table 6: HotpotQA OLS: F1∼setting + log(len) +
ans_in + setting×log(len).
Predictor Coef SEp
Reader = Qwen2.5-7B-Instruct(n=3,015,R2=0.218)
Setting=Control−0.23030.07910.004
Setting=V5−0.17940.08520.035
log(len)−0.07840.02734.0×10−3
Answer-in-rewrite+0.28740.01935.2×10−50
Reader = Qwen3-8B(n=3,015,R2=0.239)
Setting=Control−0.15760.07780.043
Setting=V5−0.04100.08520.630
log(len)−0.06570.02690.015
Answer-in-rewrite+0.29750.01853.0×10−58
Table 7: Qasper OLS: same specification as Table 6, plus
setting ×answer-type interactions (with the extractive
answer type as the reference level; interaction columns
omitted for space). The coefficient on the answer-in-
rewrite indicator is similar in sign and magnitude across
both readers and both datasets.
F Length-Quartile Breakdown of V5 vs.
Control on HotpotQA
The interaction term V5×log(len) in Table 6 is
positive and significant in both HotpotQA panels.
Table 8 stratifies the V5 −Control delta by length
quartile of the joint Control+V5 length distribution,
confirming that V 5underperforms Control on short
rewrites and reverses on long ones for both readers.
G Per-Question Oracle Headroom and
the Selection Gap (Qasper)
Even on Qasper, where a fixed rewriter setting does
not yield reliable aggregate gains, there is substan-
tial per-question variation in which rewrite helps.
We quantify this with a per-question oracle over
rewrite variants and contrast it with a representativeBinnAvg len∆F1 Ctrl V5
Reader = Qwen2.5-7B-Instruct
Q1 (short) 1890 58−2.5579.04 76.48
Q2 1822 77−1.0476.75 75.71
Q3 1858 91+1.3373.32 74.64
Q4 (long) 1835 117+0.6170.25 70.86
Reader = Qwen3-8B
Q1 (short) 1890 58−1.7779.16 77.39
Q2 1822 77−1.2777.33 76.05
Q3 1858 91+0.8275.24 76.06
Q4 (long) 1835 117+0.1771.56 71.74
Table 8: Length-matched V5 −Control comparison on
HotpotQA, binned by quartile of the joint Control+V5
length distribution.
Reader (Qasper) Qwen2.5-7B Qwen3-8B
Raw 38.25 39.50
Control freeform 38.50 39.91
V5 quote-first 36.72 39.10
LLM router (Qwen2.5-72B) 38.88 40.11
Per-question oracle (2-action) 46.63 47.72
Per-question oracle (6-action)58.66 58.65
Table 9: Per-question oracle versus router baselines
on Qasper ( N=1,005 ). The 2-action oracle picks per
question between Control and V5; the 6-action oracle
further adds Raw, Ctrl-235B, V6, and V7. The LLM
router selects within the same expanded space.
LLM-based router that uses only the question text
to select.
The 2-action oracle leaves +8.13 F1 of reach-
able headroom over the better single policy on
Qwen2.5-7B ( +7.81 on Qwen3-8B); the 6-action
oracle bounds the reachable variation at roughly
+20 F1 over raw. The LLM router (Qwen2.5-72B)
recovers +0.38 F1 (p=0.30 ) over the better sin-
gle policy and lies −7.75 F1 below the 2-action
oracle and −19.78 F1 below the 6-action oracle
(p<0.001 on both). The selection problem is there-
fore nontrivial and is not solved by writer scaling:
a stronger Qwen3-235B Control rewrite as single
policy reaches 40.94 F1, which still leaves a +5.69
F1 oracle gap.
H Question-Blind Containment Audit
If question-conditioning is the operational mecha-
nism that produces answer surfacing in the compile
output, removing the question from the rewriter
prompt should cut answer-string containment sub-
stantially. Table 10 reports the audit for both com-
pilers and both multi-hop / long-document datasets.
10

Setting Dataset Compiler Cont.% Ans-only% Len
Question-conditioned
HotpotQA Qwen2.5-72B 79.3 20.0 66.5
HotpotQA Qwen3-235B 78.4 38.8 55.9
Qasper Qwen2.5-72B 39.0 3.0 111.9
Qasper Qwen3-235B 39.7 2.0 129.6
Question-blind
HotpotQA Qwen2.5-72B 49.8 0.0 246.6
HotpotQA Qwen3-235B 54.4 0.0 305.9
Qasper Qwen2.5-72B 20.5 0.0 192.8
Qasper Qwen3-235B 24.3 0.0 244.1
Table 10: Containment audit: question-conditioned
vs. question-blind rewriters. “Cont.%” = fraction of
rewrites containing the gold answer string; “Ans-only%”
= fraction of rewrites under 50 tokens that also con-
tain it (a near-extractive form); “Len” = average length
in tokens. Removing the question halves containment
in three of four cells and zeroes the answer-only rate
in all four; blind rewrites are also nearly four times
longer than the conditioned counterparts. HotpotQA:
N=7,405; Qasper:N=1,005per cell.
Setting Compile (B 2) Verifier-edit
(B4)
Weak Qwen2.5 (0.5–
3B), Hotpot/2Wikistrongly helps
(+19to+21)mostly hurts
Mid Qwen2.5 (7–
14B), Hotpot/2Wikihelps ( +1 to
+5)mixed
Large Qwen2.5
(72B), Hotpot/2Wikinear-null near-null
Weak Qwen2.5 (0.5–
3B), Qaspersmall/mixed hurts
Mid+large Qwen2.5
(7B+), Qaspermostly hurts
(−0.7to−5.6)mostly hurts
Qwen3.5-35B, Hot-
potQAhelps ( +0.078 ,
S5)—
Table 11: Descriptive recap of the sign and rough mag-
nitude of each operator’s per-cell effect. Observational
only; the per-question oracle in Appendix G shows sub-
stantial within-cell variation that this table does not
capture.
I Descriptive Operator-by-Reader
Pattern
Table 11 compresses the observed sign and rough
magnitude of compile ( B2) and verifier-edit ( B4)
effects across the reader ×dataset cells we ran.
This is descriptive only; no rule is fit, and we dis-
courage reading any single cell as a deployment
recommendation without independent in-domain
validation.
JAnswer-Mask Ablation: Full Table and
Cross-Walk
Procedure.For every B2compile record on Hot-
potQA and 2Wiki we (i) extract the question and
the compile-output context from the prompt sentStratumn∆ B2vsB1∆mask vsB 1
HotpotQA
all 994+27.15−8.13
mask_applied=True 631+29.75−5.43
2Wiki
all 1000+38.72 +5.13
mask_applied=True 829+44.64 +4.12
mask_applied=False 171+10.04 +10.04
Table 12: Full answer-mask ablation including the all-
records and mask_applied=False strata that are ab-
sent from the main-text [MASK] row of Table 2. The
HotpotQA mask_applied=False row is omitted be-
cause the reader’s free-form rerun phrasing on yes/no
questions ( n=252 of363in that stratum) drifts non-
deterministically between the original logged call and
the masked-rerun call, which would dominate the ag-
gregate; on the gold-length ≥3 , non-yes/no subset
of the same stratum the per-question median |∆|is
0.0000 F1 (Sec. 4, “Identity sanity check”). The 2Wiki
mask_applied=False original_f1 and masked_f1
columns match because no yes/no rerun drift exists in
that stratum.
to the Qwen2.5-7B reader; (ii) case-insensitively
substring-replace the gold answer string in the
context with the literal token [MASK] , exclud-
ing records with gold length <2 characters;
(iii) re-render with the standard reader prompt
and re-call Qwen2.5-7B ( temperature=0.01 ,
max_tokens=512 ); and (iv) score F1 against the
original gold. We stratify on mask_applied : True
when the gold string was present in the compile out-
put (the interventional test), False when masking is
the identity (sanity check).
Full table.Table 12 adds the all and
mask=False strata to the main-text [MASK]
row of Table 2. The mask=False aggregates
do not match the original B2lift exactly on
HotpotQA because reader rerun phrasing varies
non-deterministically on yes/no questions; the
per-question median absolute difference on the
non-yes/no subset is <0.0001 F1 (cf. the identity
sanity-check paragraph in the main text).
K Alias Spot-Check: Reproducibility
Details
The main-text alias spot-check uses a deliberately
permissive heuristic detector. For each 2Wiki
mask=True record, after the gold answer string
is replaced with [MASK] , we test whether any of
the following surface forms still appears case-
insensitively in the masked compile context:
11

1.The gold answer with leading “the” stripped
(e.g. “the United States”→“united states”).
2.The last token of a multi-word gold answer,
lower-cased and stripped of trailing punctua-
tion, with length ≥3(e.g. “Ridley Scott” →
“scott”).
3.A capital-initial acronym of a multi-word gold
answer with length ≥2, lower-cased (e.g.
“National Basketball Association”→“nba”).
4.A 4-digit year extracted from a date-form gold
(e.g. “June 12, 1987”→“1987”).
Variants identical to the lower-cased full gold are
removed; any variant shorter than 3 characters is
discarded. The detector fires on a record if any
variant occurs as a case-insensitive substring of the
masked context.
Result.On the n=829 2Wiki mask=True stra-
tum,96records ( 11.6% ) are flagged. The remain-
ing733 records have masked-vs- B1lift+5.20
F1 (95% CI [+2.67,+7.70] ,2,000 -resample boot-
strap, seed=42 ). The flagged-subset mean is ≈
−4.13 F1, so dropping it mechanically pulls the
clean-subset mean above the +4.12 F1 of the full
set; we treat the clean-subset value as a robustness
check, not an adjusted effect estimate.
Permissiveness.Manual inspection of detector
fires shows two recurrent patterns: (a) true alias
preservation (“scott” surviving for “Ridley Scott”,
“united states” for “the United States”); and (b)
generic-last-token false positives (“heart” surviv-
ing for “Heart attack”, “school” for “High school”).
Pattern (b) means the 11.6% figure is an upper
bound on heuristically-detectable alias leakage, and
the clean-subset residual is therefore an underes-
timate of the surviving non-answer-string compo-
nent.
Reproducibility.The full alias-spot-check script
is at scripts/99d_2wiki_alias_check.py ;
it reads the released S2_2wiki.jsonl and
S3_answer_masked_2wiki.jsonl logs, re-
derives the masked context per record, applies
the detector, and prints the firing rate plus the
clean-subset bootstrap CI used above.
L Prompts
Compiler (B 2/V5).The rewriter receives the
question and all raw passages:Identify all facts in the source documents that are
relevant to the query. Rewrite those facts into a
concise, logically structured summary. Do not
fabricate information not present in the source
documents.
Reader.The reader prompt is the same for B1
(raw), B2(compile), and the masked B2context.
Only theContext:field changes:
Context: <evidence>
Question: <question>
Answer concisely:
Hard verifier (used in Appendix D only).
Compare the summary against the source docu-
ments. If the summary contains entities or claims
not present in the source documents, output a re-
vised version with unsupported claims removed.
If it is fully faithful, output the summary as is.
Soft / label-only verifiers (used in Appendix D
only).The soft verifier keeps all content but
prefixes uncertain or unsupported claims with
[UNCERTAIN] or[UNSUPPORTED] . The label-only
verifier classifies each sentence as [SUPPORTED] or
[UNSUPPORTED]but copies the sentence verbatim.
M Model Endpoints, Decoding, and
Run-ID Mapping
Decoding parameters.Rewriter
calls: temperature=0.2 , top_p=0.95 ,
max_tokens=2048 . Reader calls:
temperature=0.01 , top_p=0.95 ,
max_tokens=512 (default) or max_tokens=32
(strict 1–5-word reader-scale sweep). Failed calls
were retried up to 3 times with exponential backoff;
examples where any variant failed after all retries
were excluded from the paired comparison for
that suite. Paired bootstrap intervals use NumPy
with seed=42 ,1,000 resamples for the main-text
intervention audit ( 2,000 in the per-question
headroom analysis of Appendix G, 10,000 in the
long-form mechanism analyses of Appendix E).
Model endpoints.Table 13 lists the exact API
model -field strings used. These are submitted
field values, not guaranteed immutable checkpoint
names; the table documents the evaluated endpoints
but does not remove service-side reproducibility
risk. Full-suite runs were produced between April
21 and April 26, 2026 on the same provider plat-
form.
12

Paper name API model field
Qwen2.5-3Bqwen2.5-3b-instruct
Qwen2.5-7Bqwen2.5-7b-instruct
Qwen2.5-14Bqwen2.5-14b-instruct
Qwen2.5-32Bqwen2.5-32b-instruct
Qwen2.5-72Bqwen2.5-72b-instruct
Qwen3-8Bqwen3-8b
Qwen3-14Bqwen3-14b
Qwen3-235Bqwen3-235b-a22b
Qwen3.5-27Bqwen3.5-27b
Qwen3.5-35B-A3Bqwen3.5-35b-a3b
GLM-5glm-5
GLM-4.7glm-4.7
DeepSeek-V3deepseek-v3
Qwen-Maxqwen-max
Table 13: Exact API endpoint identifiers used in the
experiments.
Suite ID Log-file pattern
S1 (Hotpot)logs/S1_hotpotqa_*.jsonl
S2 (2Wiki)logs/S2_2wiki_*.jsonl
S3, S4 (Hotpot, GLM)logs/S{3,4}_*.jsonl
S5 (Hotpot, Qwen3.5)logs/S5_*.jsonl
Q1, Q2 (Qasper)logs/Q{1,2}_qasper_*.jsonl
Scale-Hotpotlogs/scale_hotpotqa_qwen25-*b_*.jsonl
Scale-Qasperlogs/scale_qasper_qwen25-*b_*.jsonl
AblF, AblF2, AblE3logs/abl{F,F2,E3}_*.jsonl
AblE6 oraclelogs/ablE6_*_oracle6_*.jsonl
Qblindlogs/qblind_*_{72b,235b}_*.jsonl
OLS panelslogs/lr_panel_*_*.jsonl
Answer-masklogs/S3_answer_masked_{hotpot,2wiki}.jsonl
Table 14: Suite-ID to log-file mapping. Each suite is one
(reader, rewriter, verifier, dataset, N) cell. Filenames
include the rewriter / reader endpoints (cf. Table 13) and
a date stamp.
Run-ID mapping.Table 14 maps the suite
IDs used throughout the appendix to the on-disk
JSONL log files released alongside the code. Every
numeric claim in the paper is derived from these
files via the aggregation scripts inscripts/.
13