# Align-RAG: Alignment Is All You Need for TSFM In-Context Learning

**Authors**: Mohammad Asadi, Soheil Hor, Bardiya Akhbari, Jack W. O'Sullivan, Tahoura Nedaee, Layne C. Price, Raviteja Anantha, Euan Ashley, Ehsan Adeli

**Published**: 2026-08-06 03:43:45

**PDF URL**: [https://arxiv.org/pdf/2608.05571v1](https://arxiv.org/pdf/2608.05571v1)

## Abstract
Retrieval-augmented forecasting promises to adapt frozen Time Series Foundation Models (TSFMs) to new domains without fine-tuning, but recent methods typically rely on learned fusion modules, i.e., trained adapters that merge retrieved examples into the backbone's forecast, based on the assumption that frozen backbones cannot dynamically incorporate retrieved context on their own. We show this assumption is unnecessary. We introduce Align-RAG, a training-free method that applies a closed-form per-pair amplitude rescaling and integer-lag phase shift to retrieved past-future windows before they enter a frozen backbone's context. With no learned parameters, Align-RAG outperforms the state-of-the-art trained retrieval adapter on a frozen Chronos-Bolt on all seven datasets of the standard benchmark (avg -3.75% MSE), showing that the gains previously attributed to learned fusion are recoverable without any training. Align-RAG further improves zero-shot MSE on four additional frozen TSFMs with various architectures by 2.5% to 13.7% per backbone with no per-backbone tuning. To probe why alignment helps, we compare the frozen backbone's prediction shift under aligned demonstrations to the closed-form ridge prediction shift on the same pairs. We find that aligned demonstrations induce prediction shifts that track a closed-form ridge predictor on the same pairs, with a future-shuffle control ruling out a futures-averaging account. Together, these results indicate that frozen TSFMs already support dynamic in-context use of retrievals, and that closed-form alignment should be the default baseline for retrieval-augmented forecasting before any fusion module is trained. Code available at: https://github.com/masadi-99/align-rag

## Full Text


<!-- PDF content starts -->

Align-RAG: Alignment Is All You Need for TSFM
In-Context Learning
Mohammad Asadi1Soheil Hor2Bardiya Akhbari2
Jack W. O’Sullivan1Tahoura Nedaee1Layne C. Price2
Raviteja Anantha2Euan Ashley1Ehsan Adeli1
1Stanford University2Amazon
masadi@stanford.edu
Abstract
Retrieval-augmented forecasting promises to adapt frozen Time Series Foundation
Models (TSFMs) to new domains without fine-tuning, but recent methods typically
rely on learned fusion modules, i.e., trained adapters that merge retrieved examples
into the backbone’s forecast, based on the assumption that frozen backbones cannot
dynamically incorporate retrieved context on their own. We show this assumption
is unnecessary. We introduceAlign-RAG, a training-free method that applies a
closed-form per-pair amplitude rescaling and integer-lag phase shift to retrieved
past–future windows before they enter a frozen backbone’s context. With no
learned parameters, Align-RAG outperforms the state-of-the-art trained retrieval
adapter on a frozen Chronos-Bolt on all seven datasets of the standard benchmark
(avg−3.75% MSE), showing that the gains previously attributed to learned fusion
are recoverable without any training. Align-RAG further improves zero-shot MSE
on four additional frozen TSFMs with various architectures by 2.5% to13.7%
per backbone with no per-backbone tuning. To probe why alignment helps, we
compare the frozen backbone’s prediction shift under aligned demonstrations to
the closed-form ridge prediction shift on the same pairs. We find that aligned
demonstrations induce prediction shifts that track a closed-form ridge predictor on
the same pairs, with a future-shuffle control ruling out a futures-averaging account.
Together, these results indicate that frozen TSFMs already support dynamic in-
context use of retrievals, and that closed-form alignment should be the default
baseline for retrieval-augmented forecasting before any fusion module is trained.
Code available at:https://github.com/masadi-99/align-rag
1 Introduction
Time series forecasting underpins decision-making in domains as varied as energy management,
finance, healthcare, and climate science. Time Series Foundation Models (TSFMs) such as
Chronos [Ansari et al., 2024], Chronos-2 [Ansari et al., 2025], TimesFM [Das et al., 2024],
Moirai [Woo et al., 2024], and Toto [Cohen et al., 2024] have established a zero-shot forecast-
ing paradigm by learning shared temporal representations across heterogeneous corpora. Once
pretrained, a frozen TSFM can be applied to unseen series without further training, but its accuracy
on a target distribution depends on whatever the pretraining corpus exposed.
Retrieval-augmented generation [Lewis et al., 2020] addresses an analogous gap for large language
models by providing relevant context at inference time. The corresponding move for time series
is to retrieve past–future windows from a reference set whose dynamics resemble the query and
supply them to the model as additional context. For example, a hospital deploying a frozen TSFM
Preprint.
arXiv:2608.05571v1  [cs.LG]  6 Aug 2026

Figure 1: Align-RAG overview. (1) For each query past q, retrieve top- Kpast–future windows from
a reference corpus. (2) Apply two closed-form, training-free transforms per pair so that each retrieved
past matches the query while the past–future correspondence is preserved. (3) Pack the aligned pairs
as in-context demonstrations using MMR diversification and a tiered token-budget layout. (4) A
single forward pass through a frozen Time Series Foundation Model. No learned fusion module is
introduced. (5) The final forecast combines the model’s in-context output with a distance-weighted
prior built from the retrieved aligned futures.
might retrieve patterns from related ICU traces and an energy operator might retrieve consumption
windows from similar load regimes. Done well, this offers dynamic adaptation without labelled data
or per-domain fine-tuning.
Recent retrieval-augmented forecasting on frozen TSFMs has converged on a single design pattern:
pair the backbone with a learned fusion module that mediates between retrievals and the model’s
representation. TS-RAG (Neurips 2025) [Ning et al., 2025], the state-of-the-art in retrieval-augmented
forecasting for time series foundation models to the best of our knowledge, states the position cleanly,
claiming that pretrained TSFMs “lack inherent mechanisms for domain adaptation, as they cannot
incorporate external contextual knowledge dynamically”, and trains an Adaptive Retrieval Mixer
to fuse encoded retrievals with the backbone. TimeRAF [Zhang et al., 2025] trains an end-to-end
retriever and a Channel Prompting module; RATD [Liu et al., 2024] trains a reference-modulated
attention block inside a diffusion denoiser; RAFT [Han et al., 2025] trains projection and prediction
heads on top of the retrieved patterns; TimesFM-ICF [Faw et al., 2025] re-pretrains the foundation
model itself to consume in-context retrieval examples. The underlying assumption is uniform: a
learned ad-hoc element is required to make a frozen TSFM benefit from retrieved context. We
challenge this assumption.
We introduce Align-RAG, a training-free retrieval-augmented forecasting method that extends a
frozen TSFM by aligning retrieved past–future windows to the query in amplitude and phase before
they enter the backbone’s context. Given a query window, Align-RAG retrieves nearest neighbours
from a reference set of (past, future) pairs and applies a closed-form per-pair amplitude rescaling
and phase shift so that each retrieved past matches the query. The aligned pairs are then provided as
in-context demonstrations to the backbone, whose parameters remain unchanged.
Align-RAG outperforms TS-RAG on a frozen Chronos-Bolt backbone on 7/7datasets in MSE and
6/7in MAE across the TS-RAG benchmark protocol (ETTh1, ETTh2, ETTm1, ETTm2, Weather,
Exchange, Electricity; input length 512, horizon 64), with an average MSE reduction of 3.75% and
gains significant under a paired moving-block bootstrap on 5of6small-dataset comparisons. The
same procedure transfers across four additional frozen TSFMs (Chronos-2, TimesFM-2.0, Moirai,
Toto), improving zero-shot MSE by between 2.5% and13.7% per backbone, with the largest gains on
Moirai (avg −13.7% , single cells reaching −30% on ETTm1 and −20% on Weather), and without
any per-backbone tuning. The comparison is non-trivial: TS-RAG’s Adaptive Retrieval Mixer is
2

trained on 26million context–horizon pairs sampled from Chronos’s pretraining corpus [Ning et al.,
2025], while Align-RAG introduces no learned parameters at all.
A2×2 ablation isolates where the gain comes from. Holding the backbone and benchmark fixed, we
cross retriever (trained TS-RAG retriever vs. random retrieval) with alignment (on vs. off). Switching
from the trained retriever to a random retriever, with alignment on, costs 0.84 percentage points of
MSE on average and still beats the trained TS-RAG mixer on 4of7datasets. Turning alignment off,
with either retriever, costs more than 20percentage points of MSE. The ranking of retrieved windows
accounts for under one point of the headline; the alignment of the windows that are retrieved accounts
for the rest.
We also probe the mechanism behind this gain. For each test window we compute the cosine similarity
ρGDbetween the frozen backbone’s prediction shift, induced by replacing zero-shot context with
aligned demonstrations, and the closed-form ridge prediction shift on the same demonstration pairs.
On Chronos-Bolt, ρGDrises from 0.30 without alignment to 0.45 with alignment, and a paired
moving-block bootstrap excludes zero on 7/7datasets. As a causal control, we shuffle the futures
of the retrieved pairs while leaving the pasts and the alignment intact. On ETTm1, the 4.2% MSE
reduction from alignment flips to a 12.2% MSE increase under the future shuffle, in the direction
the closed-form ridge predicts on the same shuffled pairs, ruling out an explanation in which the
backbone is averaging over retrieved futures and ignoring the past–future relationship.
Contributions.
•Align-RAG, a training-free retrieval-augmented forecasting method that aligns retrieved
past–future windows to the query in amplitude and phase before they enter a frozen TSFM’s
context, without modifying the backbone or training any new component.
•Evaluation against the trained TS-RAG mixer on the standard seven-dataset benchmark with
a frozen Chronos-Bolt (Section 4.2) and across four further frozen TSFMs (Section 4.4).
•A behavioural mechanism analysis (Section 5): under our closed-form alignment, the
frozen backbone’s prediction shift tracks the closed-form ridge prediction shift on the same
demonstrations, and a future-shuffle causal control confirms the model is regressing on the
(past, future) pairs rather than averaging over their futures.
2 Related work
Retrieval-augmented forecasting for TSFMs.A growing body of work attaches trained com-
ponents to a frozen TSFM to consume retrievals. TS-RAG [Ning et al., 2025] trains an Adaptive
Retrieval Mixer on top of frozen Chronos-Bolt [Ansari et al., 2024]; TimeRAF [Zhang et al., 2025]
trains an end-to-end retriever and a Channel Prompting module; RAFT [Han et al., 2025] trains
projection and prediction heads; RATD [Liu et al., 2024] trains a diffusion denoiser with a learned
reference-modulated attention block; TimeRAG [Yang et al., 2025] inserts a learned reprogramming
layer; Advanced RAF [Tire et al., 2024] fine-tunes the backbone end-to-end (while Naive RAF,
although a training-free variant, is outperformed substantially by TS-RAG[Ning et al., 2025]). Each
of these methods relies on a learned ad-hoc element interposed between the retrieval and the back-
bone. We compare numerically against TS-RAG, which is the most directly comparable and the
strongest trained-fusion baseline at the time of writing; we adopt its protocol throughout for fair
comparison. The corresponding move on the LLM side has typically been simpler. Methods such as
REPLUG [Shi et al., 2024], kNN-LM [Khandelwal et al., 2020], and In-Context RALM [Ram et al.,
2023] concatenate or interpolate retrieved context with a frozen LM directly.
In-context learning as implicit regression.A line of theoretical work frames in-context learning
as implicit regression. Garg et al. [2022] show that transformers trained on linear functions match
the least-squares estimator; Akyürek et al. [2023] and von Oswald et al. [2023] demonstrate that
the algorithm such transformers implement is consistent with gradient descent or closed-form ridge
regression on the in-context demonstrations. Faw et al. [2025] extends a related lens to TSFMs by
re-pretraining the foundation model to consume in-context retrieval examples. We use this body of
work as a behavioural reference: in Section 5 we compare the frozen TSFM’s prediction shift under
our alignment to the closed-form ridge predictor’s shift on the same demonstrations.
3

Phase, amplitude, and structural similarity in time series.Dynamic time warping [Cuturi
and Blondel, 2017] and its shape variants [Zhao and Itti, 2018] compare misaligned series; k-
Shape [Paparrizos and Gravano, 2015] clusters by phase-invariant Pearson similarity; the matrix
profile [Yeh et al., 2016] indexes motifs under z-normalised distance. These methods treat amplitude
and phase as nuisance directions and remove them before downstream learning.
3 Method: Align-RAG
Align-RAG operates on each retrieved past–future window, before tokenization, with two closed-form
transforms. The first rescales the retrieved past so its first and second moments agree with the query’s,
through a regularised affine fit. The second shifts the retrieved past in time so its principal lag with
respect to the query is zero, as an integer-sample shift chosen by cross-correlation. Both transforms
have the same future-side counterparts: the same affine map and the same lag, applied to the retrieved
future so the (pi,fi)pair stays consistent. Neighbour diversification, packing into the backbone’s
context budget, future blending, and a no-demonstration second forward pass are further training-free
components that complement the alignment and improve zero-shot accuracy.
Letq∈RSdenote the query past of length S(S=512 for Chronos-Bolt) and y∈RHits target
future of horizon H. A retrieval index returns a pool P={(p i,fi, di)}20
i=1of (past, future, distance)
triples per query channel. We writef θ(C)for the frozen TSFM’s forecast given a contextC.
3.1 Amplitude alignment
For each retrieved neighbour, indexed by i, we fit a single affine map on the past and apply it to both
past and future. Let µq, σqdenote the mean and standard deviation of the query past q, and µpi, σpi
those of the retrieved pastp i. The Wiener-style shrunk slopea iand interceptb iare
ai=σqσpi
σ2pi+ (σ q/M)2, b i=µ q−aiµpi (1)
where aiis the regularised counterpart of the plug-in scale ratio σq/σpi, in the spirit of Wiener
filtering for noisy estimation [Wiener, 1949]. The shrinkage parameter Mcontrols the strength of
regularisation: as σpi→0,aishrinks smoothly to zero, and a flat low-variance neighbour collapses
to its own mean rather than amplifying noise. We set M=5 . The aligned past and future are
˜pi=aipi+biand˜fi=aifi+bi, respectively.
3.2 Phase alignment
We shift the rescaled neighbour by the integer lag that maximises its sample cross-correlation with
the query past:
τ∗
i= arg max
|τ|≤S/41
S− |τ|X
t 
qt+τ−¯q ˜pi,t−¯˜pi
,(2)
where ¯qand¯˜piare the sample means. This is the maximum-likelihood lag estimator under stationary
noise [Knapp and Carter, 1976]. Restricting |τ| ≤S/4 keeps the shift from consuming more than a
quarter of the context. The sameτ∗
iis applied to ˜fito keep the past–future correspondence intact.
3.3 Auxiliary mechanics
Diversification.Top- Knearest neighbours in time-series corpora are typically near-duplicates (e.g.,
same day-of-week, same hour-of-day). We apply Maximal Marginal Relevance [Carbonell and
Goldstein, 1998] with λ=0.3 to select K=10 neighbours from a retrieval pool of 20, maximising
diversity among the chosen demonstrations while preserving relevance to the query.
Token-budget layout.Chronos-Bolt has a 2048 -token budget; ten unmodified demonstrations of
S+H=576 tokens overflow it. We pack a heterogeneous layout Md(Sd+H) +M s(Ss+H) +S q≤
2048 with Md=2,Sd=256 ,Ms=8,Ss=32. Two detailed demonstrations carry medium-range
context; eight dense ones carry the most recent 32steps and the full 64-step future. Futures are never
truncated.
4

Future blend.The in-context prediction is the convex combination
byu= (1−β)f θ(C) +βKX
i=1wi˜fi, β=0.15,(3)
where wi∝exp(−d i/τ)withτ= median({d i})/5 weights aligned futures by retrieval distance, C
is the tiered context of the previous step, and βcontrols the contribution of the retrieval-based prior
to the backbone forecast.
No-demonstration second pass.A second forward pass on the query alone, blended in the same
way, gives byc. The final prediction averages the two, by=αbyu+ (1−α) byc, with α=0.60 . Please
note that we use this full configuration for the main headline numbers, but the central claim does not
depend on these two post-processing terms: setting β= 0 and removing the consensus pass leaves a
pure aligned-ICL variant whose only model-facing intervention is the aligned demonstration context.
Table 7 shows that the pure aligned-ICL setting still outperforms TS-RAG in 6/7 datasets.
The full pseudocode is Algorithm 1. Hyperparameter values are listed in Appendix A; the same
configuration is used throughout the paper.
Algorithm 1Align-RAG.
Require:query pastq, retrieval poolP={(p i,fi, di)}20
i=1, frozen TSFMf θ
1:S ←MMR(q,P, λ=0.3, K=10)
2:for(p i,fi, di)∈ Sdo
3:(a i, bi)←Wiener affine fit ofp itoq(Eq. 1,M=5)
4: ˜pi,˜fi←a ipi+bi, aifi+bi
5:Phase-align by integer-lag cross-correlation (Eq. 2)
6:end for
7:C←TIEREDLAYOUT(q,{ ˜pi,˜fi}K
i=1)
8:w i∝exp(−d i/τ),τ= median({d i})/5
9:byu←(1−β)f θ(C) +βP
iwi˜fi
10:byc←(1−β)f θ(q) +βP
iwi˜fi
11:return by←0.60 byu+ 0.40 byc
4 Experiments
4.1 Setup
We evaluate on the seven datasets used by TS-RAG [Ning et al., 2025]: ETTh1, ETTh2, ETTm1,
ETTm2, Weather, Exchange, and Electricity. Context length S=512 and horizon H=64 follow
theTS-RAG protocol; longer-horizon results at H∈ {96,192} are reported in Appendix G and
preserve the gain on every (dataset, horizon) cell. Metrics are MSE and MAE. The frozen backbone
for the main comparison is Chronos-Bolt-Base [Ansari et al., 2024]; cross-backbone results also
use Chronos-2 [Ansari et al., 2025], TimesFM-2.0 [Das et al., 2024], Moirai [Woo et al., 2024],
and Toto [Cohen et al., 2024]. For the head-to-head comparison against TS-RAG (Table 1) we use
TS-RAG ’s own released top- 20retrieval index per (query, channel) so that any improvement isolates
the effect of alignment from differences in retrieval quality. The TS-RAG numbers are produced by
its released code and checkpoint and match the paper’s reported values to≤0.3%.
4.2 Main result on Chronos-Bolt
Table 1 reports MSE and MAE on the full test sets, together with paired moving-block bootstrap
CIs on per-window squared-error differences (block 24,N=2000 resamples, Künsch, 1989, Liu and
Singh, 1992).
Align-RAG attains the lower MSE on every benchmark and the lower MAE on six of seven. Five
MSE wins exceed 2%relative reduction; five of six small-dataset bootstrap CIs lie strictly above zero.
Chronos-Bolt inference is deterministic, so the only source of randomness in a paired comparison is
the choice of test windows.
5

Table 1: Align-RAG vs. TS-RAG [Ning et al., 2025] on a frozen Chronos-Bolt. Bold marks the
lower of MSE/MAE per row. The right column is the paired moving-block bootstrap 95% CI on
per-window MSE differences ( TS-RAG− Align-RAG); positive means Align-RAG wins. Five of six
small-dataset MSE wins are significant atα=0.05.
Datasetn test Align-RAG TS-RAG∆MSE / MAE paired95%CI
ETTh1 19,7190.3434 / 0.36120.3556 / 0.3623−3.44%/−0.30% [+0.0078,+0.0171]
ETTh2 19,7190.2379 / 0.29640.2451 / 0.2981−2.93%/−0.57% [+0.0029,+0.0117]
ETTm1 80,1990.2671 / 0.30470.2904 / 0.3113−8.01%/−2.12% [+0.0182,+0.0287]
ETTm2 80,1990.1384 / 0.21880.1465 / 0.2230−5.51%/−1.90% [+0.0061,+0.0102]
Weather 219,9960.1372 / 0.17290.1453 / 0.1770−5.60%/−2.34% [+0.0069,+0.0097]
Exchange 11,6320.0620 / 0.17150.0624 / 0.1716−0.69%/−0.05% [−0.0015,+0.0025]
Electricity 1,668,2370.1119/ 0.2018 0.1120 / 0.2003−0.07%/+ 0.73%—
Wins / avg7/7MSE (avg−3.75%);6/7MAE (avg−0.93%)
4.3 Alignment, not retrieval ranking, drives the improvement
We isolate the contribution of alignment from that of retrieval ranking with a 2×2 ablation on
Chronos-Bolt (Table 2). The retriever is either TS-RAG’s released top- 20index (ranked) or uniform
random sampling from the leakage-safe train split (random). Alignment is either applied (aligned,
the amplitude and phase steps of Section 3) or skipped (no align; retrieved windows enter the context
unscaled and unshifted).
Table 2: Retriever ×alignment ablation on Chronos-Bolt (MSE). “Aligned” applies steps (ii)+(iii)
(amplitude and phase); “no align” skips both, leaving retrieved windows unscaled and unshifted. ∆is
relative to zero-shot.
Variant ETTh1 ETTh2 ETTm1 ETTm2 Weather Exch. Elec. avg∆
Zero-shot 0.3615 0.2516 0.3107 0.1486 0.1524 0.0671 0.11320%
Ranked retriever, aligned 0.3461 0.24620.2900 0.14380.14590.0652 0.1108−3.65%
Ranked retriever, no align 0.3656 0.3131 0.3008 0.2121 0.1515 0.1271 0.1130+22.12%
Random retriever, aligned0.3450 0.24370.2929 0.14500.14550.0677 0.1130−2.81%
Random retriever, no align 0.3535 0.2972 0.3015 0.2061 0.1508 0.1252 0.1178+20.19%
With alignment fixed, retriever choice costs at most 0.84 pp average MSE: switching from the ranked
retriever to uniform random windows still beats the trained TS-RAG mixer on 4/7datasets in absolute
MSE, and on four datasets (ETTh1, ETTh2, Weather, Electricity) the random retriever is within 0.5pp
of the ranked one. Retriever quality matters more on Exchange and Electricity, the two non-periodic
datasets where shape-matched retrieval carries channel-specific information. Even there, random
retrieval is competitive.
With the retriever fixed, removing alignment under either choice regresses MSE by 20–22pp versus
zero-shot. We further investigate the mechanism through which alignment changes the model’s
behaviour in Section 5.
4.4 Cross-backbone generalisation
We apply the same method, with the same configuration, to five frozen TSFMs on the same seven
datasets, at S=512 ,H=64 . The retriever is rebuilt from scratch as a per-dataset, per-channel,
train-split-onlyz-normalised Euclidean nearest-neighbour index, with a strict leakage audit.
All five backbones reduce MSE on average, ranging from −2.45% (Toto) to −13.73% (Moirai).
Moirai, the backbone furthest from the Chronos lineage on which TS-RAG was trained, gains
−13.73% on average and wins on every dataset, with the largest single cell a −29.95% MSE
reduction on ETTm1 and −20.18% on Weather. TimesFM-2.0 wins on the four ETT datasets at up to
−12.43% but regresses on Exchange ( +2.03% ). Bolt and Chronos-2 win every benchmarked dataset
where they are evaluated, with Bolt’s Electricity result ( +0.04% ) effectively a tie. Toto is the most
uneven case: it gains on ETTh1, ETTm1, and Weather but regresses by +0.04% to+2.20% on the
other four datasets, and its−2.45%average is the smallest of the five backbones.
6

Table 3: Cross-backbone results: zero-shot MSE (ZS) and Align-RAG’s relative MSE change vs. that
backbone’s zero-shot ( ∆%); negative ∆% means Align-RAG wins. Weather and Electricity were
excluded from zero-shot evaluation for TimesFM in Ning et al. [2025]’s protocol on the grounds
of pretraining-corpus overlap, so we omit those two cells for TimesFM-2.0 to remain consistent
with their leakage treatment. All five backbones use the same in-house z-normalised Euclidean
nearest-neighbour retriever; absolute MSE values therefore differ from Table 1’s Align-RAG cells,
which use TS-RAG’s released top-20index.
Dataset Bolt Chronos-2 TimesFM Moirai Toto row avg
ZS∆%ZS∆%ZS∆%ZS∆%ZS∆% ∆%
ETTh10.3046−4.890.3158−6.390.3165−7.100.3359−5.300.2931−4.01−5.54
ETTh20.2278−0.270.2444−4.080.2451−1.120.2696−9.060.2217 +0.04−2.89
ETTm10.2633−11.130.2645−11.650.2674−9.050.3962−29.950.2652−16.09−15.58
ETTm20.1403−5.290.1408−0.130.1605−12.430.1822−16.600.1422 +2.20−6.45
Weather0.1521−6.260.1618−1.97— —0.2132−20.180.1413−0.24−7.16
Exch.0.0651−1.200.0584−0.680.0636 +2.03 0.0599−1.240.0460 +0.25−0.17
Elec.0.1043 +0.04 0.1067−1.87— —0.1628−13.820.1149 +0.69−3.74
wins6/7 7/7 4/5 7/7 3/7
avg∆%−4.14−3.83−5.53−13.73−2.45−5.96
4.5 Component ablation
Table 4 reports a cumulative-add ablation on the five small datasets, with Exchange and Electricity
reported separately under the table. Each row adds one component on top of the previous.
Table 4: Cumulative ablation on Chronos-Bolt, MSE change vs. TS-RAG . The last column is the
average across the five smaller datasets; Electricity ( ntest=1.67 M) and Exchange ( ntest=11.6 k) span
two orders of magnitude in test-set size and are reported separately under the table.
Variant ETTh1 ETTh2 ETTm1 ETTm2 Weather avg
Zero-shot Chronos-Bolt+1.7% +2.7% +7.0% +1.4% +4.9% +3.5%
+CC align,K=2,β=0.1−2.1% +1.6%−4.1%−1.1%−5.5%−2.2%
+tiered layout,K=10, affine−2.0%−2.1%−5.7%−2.6%−3.5%−3.2%
+MMR (λ=0.3) [steps i–v]−2.9%−2.1%−7.4%−4.0%−5.4%−4.4%
+consensus pass (α=0.60) [full]−3.4%−2.9%−8.0%−5.5%−5.6%−5.1%
On Electricity, steps (i)–(v) regress to +2.7% MSE and the second pass (vi) recovers a tie at −0.07% ;
on Exchange the full method lands at −0.69% . The amplitude and phase steps alone close most
of the gap to TS-RAG on the small datasets; the layout and diversification each contribute about a
point of average MSE; the second-pass average contributes a further point and resolves the Electricity
calibration regression that the in-context steps alone introduce on a 321-channel corpus (Appendix C).
A leave-one-out variant on ETTh1 and ETTm1 confirms that removing any single component costs 0
to2.8pp MSE (Appendix B).
5 Aligned demonstrations elicit ridge-like behaviour
In this section, we examine the in-context behaviour of the frozen TSFM under our alignment, and ask
whether the model’s demonstration-induced prediction shift is consistent with the closed-form ridge
prediction shift on the same (pi,fi)pairs. The investigation combines a cosine signature relating the
two prediction shifts and a future-shuffle causal control that distinguishes regression on the pairs
from averaging over their futures.
Probe.Let bymodel andbyzsdenote the model’s forecast with and without the aligned demonstrations,
respectively. The closed-form ridge predictor is
byridge =q⊤(P⊤P+λI)−1P⊤F, λ= 1,
7

unaligned cc-only amp-only aligned aligned
(fut-shuf)0.250.300.350.400.450.50ρGD  (Pearson cosine)(a) Implicit-GD signature
unaligned cc-only amp-only aligned aligned
(fut-shuf)0.300.340.380.42Decoder cross-attention
to demo region(b) Demo-conditioned attention
unaligned cc-only amp-only aligned aligned
(fut-shuf)−3−2−10123456% MSE change vs zero-shot(c) Behavioural outcomeFigure 2: Mechanism evidence on Chronos-Bolt across five demonstration conditions ( N=200
windows per cell, averaged over the seven datasets). (a) Implicit-GD signature ρGD: amplitude
alignment lifts ρGDfrom 0.30 (unaligned ,cc_only ) to∼0.45 (amp_only ,aligned ); the future-
shuffle condition matches that level. (b) Decoder cross-attention to the demonstration region rises
in the same conditions. (c) MSE change versus zero-shot: amp_only andaligned reduce MSE,
whereas aligned_fut_shuf reverses the gain even though ρGDstays high in panel (a) – the model
continues to track an implicit ridge solution, but on corrupted demonstration pairs.
where P∈RK×Sstacks the K=10 aligned demonstration pasts and F∈RK×Hstacks their aligned
futures. The implicit-GD signature is the per-window cosine
ρGD(w) = cos bymodel(w)−byzs(w),byridge(w)−byzs(w)
.
A pretrained TSFM also handles seasonality, trend, and recent-value extrapolation, so ρGD=1is
not reachable; the question is whether a regression-on-pairs component shows up above those
other behaviours. We evaluate ρGDunder five conditions per window: unaligned (raw retrievals),
amp_only,cc_only,aligned, andaligned_fut_shuf(aligned demonstrations with futures per-
muted within the set, breaking the pair correspondence). Confidence intervals are computed by paired
moving-block bootstrap with block length 20andN= 2000 resamples over 200paired test windows
per dataset.
Result. ρGDrises from [0.17,0.48] unaligned (mean 0.30) to[0.37,0.59] aligned (mean 0.45). The
paired bootstrap of ρaligned
GD −ρunaligned
GD excludes zero on 7/7Chronos-Bolt datasets (smallest gap
+0.07 on Electricity, largest +0.25 on Exchange). Cross-attention to the demonstration region rises
on6/7datasets (Fig. 2b). Of the two alignment components, amplitude carries the effect: amp_only
matchesalignedon7/7datasets andcc_onlyreaches significance only on ETTh2.
Future-shuffle causal control.The shuffle permutes {fi}within each demonstration set, leaving
the pasts and the alignment intact. Under a futures-averaging account the shuffle is innocuous: the
set of futures is unchanged. Under regression on the pairs it biases the closed-form ridge solution in
a known direction, and a model that tracks that solution should inherit the bias. The latter is what
we observe. The shuffle worsens MSE on 4/7datasets (mean +2.6 pp, median +0.9 pp); the largest
reversal is on ETTm1, where the −4.2% gain from alignment flips to +12.2% . The remaining three
datasets, whose alignment-vs-unaligned gap is already small, move by less than 3pp in either MSE
orρGD. Across all seven, ρGDremains high under the shuffle (mean 0.46, range [0.34,0.61] ): the
model continues to implement an implicit ridge regression, now on the corrupted pairs (pi,f′
i), and
inherits whichever direction of bias that corruption induces. A futures-averaging account predicts
neither the directional MSE shift nor the preservation ofρ GD.
The probe replicates on Chronos-2’s group-attention pathway with ρaligned
GD > ρunaligned
GD on6/7
datasets (13/14cells across the two backbones, Appendix D). Figure 3 shows one ETTm1 window:
raw retrievals miss the query in both amplitude and phase; alignment collapses the demonstration
cloud onto the query’s regime, the model’s prediction shift becomes visibly proportional to the
closed-form ridge shift on the same pairs, and MSE drops from2.06(zero-shot) to0.70(aligned).
8

0 100 200 300 400 500
time step (S=512 past, P=64 future)−2−101234value (z-score)(a) Raw retrieved demonstrations
query past
query future
0 100 200 300 400 500
time step (S=512 past, P=64 future)−2−101234(b) Align-RAG aligned demonstrations
0 10 20 30 40 50 60
forecast horizon012345value (z-score)(c) Future predictions
zero-shot  (MSE=2.056)
unaligned  (MSE=1.945)aligned     (MSE=0.695)
true future
0 10 20 30 40 50 60
forecast horizon−0.50.00.51.01.52.02.5Δ̂y (vs zero-shot)(d) Model shift vs. ridge shift  (ρaln
GD=+0.94, ρun
GD=+0.26)
zero-shot anchor
Δ̂ymodel unaligned
Δ̂yridge unalignedΔ̂ymodel aligned
Δ̂yridge alignedFigure 3: One ETTm1 query window. (a) Raw retrieved demonstrations have the wrong amplitude
and phase relative to the query, and the resulting unaligned ICL prediction (MSE 1.94,ρGD=+0.26 )
misses the true future. (b) Aligned demonstrations cluster around the query. (c) Aligned ICL recovers
the trajectory (a 2.9× reduction in MSE). (d) Under aligned, the model’s prediction shift visibly
tracks the closed-form ridge prediction shift on the same demos ( ρGD=+0.94 ); under unaligned, the
two are nearly orthogonal (ρ GD=+0.26).
6 Conclusion
In this paper we revisited the assumption underlying recent retrieval-augmented forecasting on frozen
Time Series Foundation Models: that learned fusion is required for a frozen backbone to benefit from
retrieved context. We introduced Align-RAG, a training-free method that aligns retrieved past–future
windows to the query in amplitude and phase before they enter the backbone’s context. Under
the standard seven-dataset TS-RAG benchmark protocol with a frozen Chronos-Bolt, Align-RAG
matches or exceeds the trained TS-RAG mixer on every dataset in MSE, and the same configuration
improves zero-shot performance on four additional frozen TSFMs without per-backbone tuning.
A retriever ×alignment ablation locates the gain in demonstration alignment rather than retrieval
ranking, and a behavioural probe shows that under our alignment the model’s prediction shift tracks
the closed-form ridge prediction shift on the same demonstrations, with a future-shuffle causal control
consistent with regression on the past–future pairs.
These findings carry empirical and mechanistic implications. Empirically, aligned demonstrations
carry the bulk of the improvement in retrieval-augmented forecasting on a frozen TSFM, and learned
fusion appears largely to compensate for the absence of such alignment. Mechanistically, frozen
pretrained TSFMs admit a behavioural regression structure on aligned past–future pairs, consistent
with predictions from the in-context learning literature on synthetic-regression-trained transformers.
Natural future directions are richer retrieval geometries, including multivariate and covariate-informed
settings, and replication of the retriever ×alignment ablation across additional backbones to test the
generality of the attribution.
Limitations.The mechanism evidence is correlational: ρGD= 0.45 aligned versus 0.30 unaligned
is directional evidence, not the equivalence ( ρGD=1) given by construction theorems. Our evaluation
uses the standard univariate TS-RAG benchmark for direct comparison to the state-of-the-art; multi-
variate covariates and distribution shifts remain open. Finally, the retriever ×alignment ablation is
Chronos-Bolt-only; per-backbone replication is left to future work.
9

Acknowledgements
Mohammad Asadi was supported by the Amazon AI PhD Fellowship during this work. We thank the
Amazon fellowship program for its support.
References
Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning
algorithm is in-context learning? investigations with linear models. InInternational Conference
on Learning Representations, 2023.
Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen,
Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper
Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon
Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time
series.Transactions on Machine Learning Research, 2024. arXiv:2403.07815.
Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado,
Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami,
Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erick-
son, Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, and
Michael Bohlke-Schneider. Chronos-2: From univariate to universal forecasting.arXiv preprint
arXiv:2510.15821, 2025.
Jaime Carbonell and Jade Goldstein. The use of MMR, diversity-based reranking for reordering
documents and producing summaries. InProceedings of the 21st Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval, pages 335–336, 1998.
Ben Cohen, Emaad Khwaja, Kan Wang, Charles Masson, Elise Ramé, Youssef Doubli, and Oth-
mane Abou-Amal. Toto: Time series optimized transformer for observability.arXiv preprint
arXiv:2407.07874, 2024.
Marco Cuturi and Mathieu Blondel. Soft-DTW: a differentiable loss function for time-series. In
International Conference on Machine Learning, 2017.
Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation
model for time-series forecasting. InInternational Conference on Machine Learning, 2024.
arXiv:2310.10688.
Matthew Faw, Rajat Sen, Yichen Zhou, and Abhimanyu Das. In-context fine-tuning for time-series
foundation models. InProceedings of the 42nd International Conference on Machine Learning,
volume 267 ofProceedings of Machine Learning Research, pages 16355–16374. PMLR, 2025.
arXiv:2410.24087.
Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn
in-context? a case study of simple function classes. InAdvances in Neural Information Processing
Systems, 2022.
Sungwon Han, Seungeon Lee, Meeyoung Cha, Sercan O. Arik, and Jinsung Yoon. Retrieval
augmented time series forecasting. InProceedings of the 42nd International Conference on
Machine Learning, volume 267 ofProceedings of Machine Learning Research, pages 21774–
21797. PMLR, 2025.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization
through memorization: Nearest neighbor language models. InInternational Conference on
Learning Representations, 2020.
C. H. Knapp and G. C. Carter. The generalized correlation method for estimation of time delay.
IEEE Transactions on Acoustics, Speech, and Signal Processing, 24(4):320–327, 1976. doi:
10.1109/TASSP.1976.1162830.
Hans R. Künsch. The jackknife and the bootstrap for general stationary observations.Annals of
Statistics, 17(3):1217–1241, 1989.
10

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems, 2020.
Jingwei Liu, Ling Yang, Hongyan Li, and Shenda Hong. Retrieval-augmented diffusion mod-
els for time series forecasting. InAdvances in Neural Information Processing Systems, 2024.
arXiv:2410.18712.
Regina Y . Liu and Kesar Singh. Moving blocks jackknife and bootstrap capture weak dependence.
In Raoul LePage and Lynne Billard, editors,Exploring the Limits of Bootstrap, pages 225–248.
Wiley, New York, 1992.
Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang, James Zhang, Kashif Rasul, Anderson Schneider,
Lintao Ma, Yuriy Nevmyvaka, and Dongjin Song. TS-RAG: Retrieval-augmented generation based
time series foundation models are stronger zero-shot forecaster. InAdvances in Neural Information
Processing Systems, 2025. arXiv:2503.07649.
John Paparrizos and Luis Gravano. k-Shape: Efficient and accurate clustering of time series. In
Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data, pages
1855–1870, 2015.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and
Yoav Shoham. In-context retrieval-augmented language models.Transactions of the Association
for Computational Linguistics, 2023. arXiv:2302.00083.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. REPLUG: Retrieval-augmented black-box language models. In
NAACL, 2024. arXiv:2301.12652.
Kutay Tire, Ege Onur Taga, Muhammed Emrullah Ildiz, and Samet Oymak. Retrieval augmented
time series forecasting. 2024.
Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordv-
intsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient
descent. InProceedings of the 40th International Conference on Machine Learning, volume 202
ofProceedings of Machine Learning Research, pages 35151–35174. PMLR, 2023.
Norbert Wiener.Extrapolation, Interpolation, and Smoothing of Stationary Time Series: With
Engineering Applications. MIT Press, 1949.
Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo.
Unified training of universal time series forecasting transformers. InInternational Conference on
Machine Learning, 2024. arXiv:2402.02592.
Silin Yang, Dong Wang, Haoqi Zheng, and Ruochun Jin. TimeRAG: Boosting LLM time series
forecasting via retrieval-augmented generation. InIEEE International Conference on Acoustics,
Speech and Signal Processing, pages 1–5, 2025. doi: 10.1109/ICASSP49660.2025.10889933.
arXiv:2412.16643.
Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau,
Diego Furtado Silva, Abdullah Mueen, and Eamonn Keogh. Matrix profile I: All pairs similarity
joins for time series: A unifying view that includes motifs, discords and shapelets. InICDM, 2016.
Huanyu Zhang, Chang Xu, Yifan Zhang, Zhang Zhang, Liang Wang, and Jiang Bian. TimeRAF:
Retrieval-augmented foundation model for zero-shot time series forecasting.IEEE Transactions on
Knowledge and Data Engineering, 37(9):5654–5665, 2025. doi: 10.1109/TKDE.2025.3579137.
Jiaping Zhao and Laurent Itti. shapeDTW: Shape dynamic time warping.Pattern Recognition, 2018.
11

A Implementation details
Retrieval index.Section 4.2 uses the released top-20 per (query, channel) index from
TS-RAG [Ning et al., 2025]. Section 4.4 uses an in-house per-dataset, per-channel, train-split-
onlyz-normalised Euclidean nearest-neighbor index, with a strict leakage audit that rejects any
returned neighbor whose S+H -step span enters the validation or test split. Distances are Euclidean
in the normalised-past space.
Hyperparameters. K=10 , λMMR=0.3 , β=0.15 , α=0.60 , M=5 ,
(Md, Sd, Ms, Ss)=(2,256,8,32) , CC search range |τ| ≤S/4 = 128 , affine fit on pionly.
We use the same values for every dataset and backbone reported. Sensitivity sweeps over α(Table 6)
and component leave-one-out (Table 5) are in the appendix.
Compute.A single H100 evaluates Align-RAG with Chronos-Bolt across the seven datasets in
under six wall-clock hours.
Code and reproducibility.The Align-RAG implementation is approximately 720lines of Python
across six files (alignment, retrieval interface, run-Bolt, run-Chronos-2, aggregation, metrics) and
contains no learned parameters. Code, the cross-backbone retrieval-index build script, and the exact
bootstrap-resample seeds are released alongside the paper. Section 4.2 reuses TS-RAG’s public
top-20 retrieval index without modification. The cross-backbone retrieval index of Section 4.4 uses a
stride- 1window scan with z-normalisation per channel, Euclidean distance on the normalised past,
and a leakage filter that removes any candidate whose S+H -step span overlaps the validation or test
split.
B Leave-one-out component ablation
Table 5 removes each component in turn from the unified implementation on two representative
datasets. Every component is non-trivial on at least one of the two datasets.
Table 5: Leave-one-out component ablation on Chronos-Bolt. MSE change vs. full Align-RAG on
two datasets.
Removed ETTh1 ETTm1
MSE∆MSE∆
nothing (full Align-RAG) 0.3452 – 0.2688 –
−CC phase alignment 0.3451+0.0%0.2701+0.5%
−MMR diversification 0.3484+0.9%0.2738+1.9%
−tiered layout (useK=2) 0.3459+0.2%0.2709+0.8%
−future blend (β=0) 0.3518+1.9%0.2762+2.8%
C Second-pass averaging: decomposition andαsweep
MSE/MAE asymmetry.The two predictions byuandbychave per-window error correlation r∈
[0.94,0.99] across the seven datasets, with 87–94% same-sign agreement. Averaging same-sign
errors of comparable magnitude reduces tails more than it reduces medians: on Electricity, the
per-step q=0.99 error quantile drops from |eu|=1.291 and|ec|=1.270 to1.265 at the average, while
theq=0.50 quantile moves by less than 10−3. This is why the second pass helps MSE more than
MAE.
Murphy–Theil decomposition of the Electricity penalty.Under steps (i)–(v) alone, Electricity
is the single dataset on which Align-RAG loses +3.1% MSE relative to TS-RAG . Decomposing
the per-window MSE of byuandbycasMSE = bias2+ scale2+ shape withbias2= (¯by−¯y)2,
scale2= (σ by−σy)2, andshape = 2σ byσy(1−ρ)gives:
12

unifiedby u consensusby c gap (unified−cons.)
bias20.03328 0.03181 +0.00147(39%of gap)
scale20.01833 0.01677 +0.00157(42%of gap)
shape 0.06363 0.06290 +0.00072(19%of gap)
MSE0.11524 0.11148 +0.00376
81% of the in-context penalty lives in level and scale, not shape. The shape correlation ρ=0.64 is
unchanged to 10−3between the two predictions. Averaging at α=0.60 closes the gap to −0.07%
MSE.
αsweep.Table 6 reports a fine sweep over byfinal=αbyu+ (1−α) byc. The MSE-win count is 7/7
for every α∈[0.50,0.62] , so the configuration is robust over a wide range of mixing weights. Within
that range α∈ {0.60,0.62} also achieves 6/7MAE wins; we pick α=0.60 once as the released
setting and hold it fixed.
Table 6: Fineαsweep. Win counts out of seven and averages relative to TS-RAG.
αMSE wins MSE avg MAE wins MAE avg elec. MSE elec. MAE exch. MSE exch. MAE
0.50 7/7−3.49% 5/7−0.82%−0.47% +0.44%−0.12% +0.25%
0.58 7/7−3.71% 5/7−0.92%−0.16% +0.67%−0.59% +0.00%
0.60 7/7−3.75%6/7−0.93%−0.07% +0.73%−0.69%−0.05%
0.62 6/7−3.78% 6/7−0.95% +0.03% +0.79%−0.77%−0.09%
0.70 6/7−3.84% 6/7−0.97% +0.47% +1.08%−1.00%−0.22%
D Mechanism: extended evidence
This appendix collects the per-dataset within-condition evidence behind the implicit-ridge mecha-
nism summarised in Section 5: the within-dataset signature/MSE correlation, the cross-architecture
replication on Chronos-2, and a concrete worked window.
Within-dataset GD-MSE correlation.Holding dataset fixed and correlating, across the five
conditions, ρGDwith the MSE improvement vs. zero-shot (Fig. 4): on 5/7datasets the within-
dataset Pearson correlation is large and positive (ETTh1 r= + 0.99 , ETTm2 +0.99 , Weather +0.94 ,
Electricity +0.98 , ETTh2 +0.48 ). The two negative cases are diagnostic. ETTm1 ( r=−0.61 ): the
highest- ρGDcell is aligned_fut_shuf but its ridge target is biased by the shuffle, so the model
faithfully implements that biased ridge and the MSE worsens. Exchange Rate ( r=−1.00 ): zero-
shot is already near the dataset’s noise floor (MSE 0.062 ), so any implicit-ridge update moves the
prediction off that floor. Both exceptions strengthen rather than weaken the mechanism: the signature
measureswhetherthe model is doing implicit ridge regression on the demos; the MSE outcome
additionally requires the demos to carry useful information beyond zero-shot.
Cross-architecture replication.Figure 5 replicates the implicit-GD signature on Chronos-2, which
has a fundamentally different ICL pathway (group attention over series rather than concatenation).
On the same seven datasets ( N=100 windows each), aligned demos again yield a higher signature
than unaligned on 6/7datasets (Weather is the single inversion). Combining both backbones,
ρaligned
GD >ρunaligned
GD on13/14(dataset, backbone) cells.
Concrete window.The ETTm1 window in Fig. 3 (main paper, test index 50998 ) illustrates the
mechanism on a single query. The ten raw retrievals are misaligned in both amplitude and phase,
and the resulting unaligned ICL prediction ( ρun
GD=+0.26 , MSE 1.94) is barely distinguishable from
zero-shot (MSE 2.06). After alignment, the demonstration cloud collapses onto the query’s regime,
the model’s prediction shift tracks the closed-form ridge shift on the same demos almost step-for-step
(ρaln
GD=+0.94), and MSE falls to0.70– a2.9×reduction over zero-shot on this window.
13

0.30 0.35 0.40 0.45
GD
02468MSE improvement vs zs (%)ETTh1  (r=+0.99)
0.20 0.25 0.30 0.35 0.40
GD
16
14
12
10
8
6
4
2
0MSE improvement vs zs (%)ETTh2  (r=+0.48)
0.45 0.50 0.55 0.60
GD
10
5
0510MSE improvement vs zs (%)ETTm1  (r=-0.61)
0.35 0.40 0.45 0.50 0.55
GD
20
15
10
5
051015MSE improvement vs zs (%)ETTm2  (r=+0.99)
0.30 0.35 0.40
GD
8
6
4
2
0MSE improvement vs zs (%)Weather  (r=+0.94)
0.2 0.3 0.4
GD
0246810MSE improvement vs zs (%)Exchange  (r=-1.00)
0.325 0.350 0.375 0.400 0.425 0.450
GD
20
15
10
5
0MSE improvement vs zs (%)Electricity  (r=+0.98)
unaligned
cc only
amp only
aligned
aln+fut-shufFigure 4: Within-dataset correlation between implicit-GD signature ρGDand MSE improvement vs.
zero-shot (%). One panel per dataset, five markers per panel (one per condition); Pearson rshown.
Five datasets (ETTh1, ETTh2, ETTm2, Weather, Electricity) show large positive r, as the mechanism
predicts. ETTm1 ( r=−0.61 ) is a diagnostic case (the aligned_fut_shuf cell has high ρGDbut a
biased ridge target, so the model’s faithful tracking yields worse MSE; see future-shuffle control).
Exchange ( r=−1.00 ) is at zero-shot’s noise floor (MSE 0.062 ), where any in-context shift moves
prediction off the floor; we discuss both anomalies in Sec. 5.
ETTh1 ETTh2 ETTm1 ETTm2WeatherExchange Electricity0.00.10.20.30.40.50.60.7cos(ymodel,yridgeyzs)
(a) Chronos-Bolt (encoder decoder T5)
unaligned
amp only (Wiener)
aligned (Wiener+CC)
ETTh1 ETTh2 ETTm1 ETTm2WeatherExchange Electricity0.00.10.20.30.40.50.60.7cos(ymodel,yridgeyzs)
(b) Chronos-2 (group-attention)
unaligned
amp only (Wiener)
aligned (Wiener+FFT)
Figure 5: Cross-architecture replication of the implicit-GD signature. (a) Chronos-Bolt (T5 encoder–
decoder, CC phase alignment; N=200 ): aligned >unaligned on 7/7, all bootstrap-significant.
(b) Chronos-2 (group attention, FFT phase alignment; N=100 ): aligned >unaligned on 6/7. The
amp_only condition is competitive with aligned on Chronos-2, matching the cross-backbone
Wiener-vs-clip ablation in Sec. 4.4: FFT phase transfer is most useful on the ETT family.
E Isolating the in-context contribution from blend and second pass
Align-RAG combines four in-context steps (diversification, amplitude alignment, phase alignment,
layout) with two post-processing additions: the future blend at β=0.15 and the second-pass average
atα=0.60 . Setting β=0 and/or α=0 isolates the contribution of the in-context steps alone. Table 7
reports the four corners of this2×2on Chronos-Bolt against TS-RAG.
Aligned demos fed through nothing but the in-context layout beat trained TS-RAG on 6/7datasets at
average −1.64% MSE; the only exception is Electricity. The βblend recovers Electricity ( +4.1%→
+1.0%); theαsecond pass closes the gap to−0.1%.
14

Table 7: Isolating the four in-context steps from the post-processing βblend and αconsensus, on
Chronos-Bolt. Each cell is per-dataset MSE change vs. TS-RAG ; bold indicates a gain. Wins/ 7
counts datasets with strictly negative∆MSE; avg∆%averages the seven per-dataset deltas.
Variant ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange Electricity wins avg∆%
Full Align-RAG (β=0.15, α=0.60)−3.4%−2.9%−8.0%−5.5%−5.6%−0.7%−0.1% 7/7−3.75%
−αonly (β=0.15, α=0)−3.3%−2.2%−8.0%−4.5%−6.1%−1.0% +2.7% 6/7−3.20%
−βonly (β=0, α=0.60)−1.0%−1.5%−4.2%−4.8%−2.6%−2.5% +1.0% 6/7−2.20%
−αand−β(pure aligned ICL)−1.0%−0.9%−4.9%−3.2%−3.3%−2.2% +4.1% 6/7−1.64%
F Wiener vs. clipped affine on cross-backbone Weather and Exchange
The plug-in scale ratio σq/σpiproduces Q99values of 1.51 on ETTh1 but 3.0on Weather and 6.7
on Exchange, with maxima exceeding106on the rare flat retrievals that any nearest-neighbor index
occasionally returns. Two regularisers stabilise this tail: hard-clipping ai∈[1/M, M] atM=5 , or
Wiener shrinkage. On Weather, where the tail is heaviest, Wiener strictly dominates clipping on every
backbone (Table 8); on Exchange the two regularisers are competitive, with Wiener slightly better
on three of five backbones and clipping slightly better on the other two. On the five datasets where
the tail is bounded, the two regularisers agree to four decimal places. We deploy Wiener shrinkage
globally because it never explodes and matches or beats clipping on the two tail-heavy datasets.
Table 8: Step-(ii) regularisers on the two datasets where the plug-in scale ratio is unbounded. MSE
change vs. each backbone’s zero-shot. Underline marks the better regulariser. The Wiener row
reproduces the deployed values from Table 3; the plug-in and clipped rows are from the same retrieval
setup with only step (ii) substituted.
Bolt Chronos-2 TimesFM Moirai Toto
Weather
Plug-in+8.0% +10.2%—+2.8% +11.7%
Clipped+1.3% +3.0%—+0.3% +4.1%
Wiener−6.26% −1.97% —−20.18% −0.24%
Exchange
Plug-in+3.0% +6.1% +4.8% +7.1% +8.0%
Clipped−2.9% −0.6%−1.4% +3.6% +0.7%
Wiener−1.20%−0.68% +2.03%−1.24% +0.25%
G Multi-horizon results
The TS-RAG protocol of Ning et al. [2025] fixes the forecast horizon at H=64 . To check that
Align-RAG’s gains are not specific to that one horizon, we evaluate at H∈ {64,96,192} on the five
small datasets, with the same frozen Chronos-Bolt backbone and the same alignment hyperparameters
as the main result (Section 4.2). The retrieval index here is the in-house z-normalised Euclidean
nearest-neighbour index used in the cross-backbone evaluation (Section 4.4) rather than the TS-RAG
released top- 20index used in Table 1; the two retrievers produce different absolute MSE values on
the same backbone, so the H=64 column of Table 9 is not numerically identical to the corresponding
cells of Table 1, even though the pipeline is otherwise the same.
The MSE gains hold at every (dataset, horizon) cell, ranging from −1.20% to−6.59% . ETTm1 has
the largest gains ( ≈ −6.5% across horizons); ETTh2 and Exchange have the smallest. The MAE
picture is similar but with smaller magnitudes (MAE is dominated by trend and shape rather than
scale, which is where in-context demonstrations help most), with two cells where MAE is essentially
flat (+0.04% on ETTm2 at H=64 and+0.47% on ETTh2 at H=64 ). The horizon ranking of the
relative gains is monotone or nearly so on every dataset, so the H=64 result reported in the main
paper is not a peculiarity of that specific horizon.
15

Table 9: Multi-horizon MSE on a frozen Chronos-Bolt. ∆% is the relative MSE change vs. zero-shot
at the same horizon; negative means Align-RAG wins.
DatasetHZero-shot Align-RAG∆%
ETTh164 0.36150.3461−4.26%
ETTh196 0.38280.3663−4.31%
ETTh1192 0.43640.4148−4.94%
ETTh264 0.25160.2462−2.14%
ETTh296 0.28750.2807−2.35%
ETTh2192 0.35390.3403−3.85%
ETTm164 0.31070.2904−6.53%
ETTm196 0.33080.3090−6.59%
ETTm1192 0.38360.3594−6.31%
ETTm264 0.14860.1441−2.98%
ETTm296 0.17670.1710−3.25%
ETTm2192 0.24760.2389−3.54%
Exchange64 0.06710.0652−2.90%
Exchange96 0.09920.0970−2.22%
Exchange192 0.19380.1914−1.20%
Table 10: Multi-horizon MAE on a frozen Chronos-Bolt. Each cell reports zero-shot / Align-RAG /
∆%at the indicated horizon.
DatasetH=64H=96H=192
ETTh10.3650/0.3635/−0.40% 0.3795/0.3772/−0.62% 0.4117/0.4070/−1.14%
ETTh20.2992/0.3006/+ 0.47% 0.3250/0.3257/+ 0.22% 0.3688/0.3672/−0.43%
ETTm10.3183/0.3137/−1.44% 0.3331/0.3275/−1.70% 0.3653/0.3584/−1.89%
ETTm20.2235/0.2236/+ 0.04% 0.2447/0.2439/−0.34% 0.2932/0.2911/−0.72%
Exchange0.1778/0.1762/−0.93% 0.2190/0.2163/−1.24% 0.3138/0.3098/−1.28%
H Comparison against RAFT
RAFT [Han et al., 2025] is a trained-adapter retrieval-augmented forecaster: a projection module
and prediction head are trained on top of the retrieved patterns, on a frozen backbone. We compare
Align-RAG against RAFT on the four ETT datasets at the two horizons ( H=96,192 ) reported by
RAFT, with both methods using a frozen Chronos-Bolt backbone.
Table 11: Align-RAG vs. RAFT [Han et al., 2025] on a frozen Chronos-Bolt at H∈ {96,192} .
Align-RAG attains the lower MSE on every cell. “ARvs. RAFT” is the relative MSE change of
Align-RAG against RAFT; “ARvs. ZS” is the relative change against zero-shot Chronos-Bolt at the
same horizon.
DatasetHRAFT Bolt-ZS Align-RAG ARvs. RAFT ARvs. ZS
ETTh196 0.3868 0.38280.3663−5.29%−4.31%
ETTh1192 0.4225 0.43640.4148−1.82%−4.94%
ETTh296 0.2960 0.28750.2807−5.17%−2.35%
ETTh2192 0.3840 0.35390.3403−11.37%−3.85%
ETTm196 0.3290 0.33080.3090−6.08%−6.59%
ETTm1192 0.3634 0.38360.3594−1.11%−6.31%
ETTm296 0.1773 0.17670.1710−3.59%−3.25%
ETTm2192 0.2425 0.24760.2389−1.50%−3.54%
Wins / avg8/8,−4.5% 8/8,−4.4%
Align-RAG attains the lower MSE on 8/8cells with an average reduction of −4.5% relative to RAFT.
The largest single gain is −11.37% on ETTh2 at H=192 ; the smallest is −1.11% on ETTm1 at
H=192 . Align-RAG also continues to improve on zero-shot Chronos-Bolt at every cell ( −2.35% to
−6.59% ), so the comparison cannot be reduced to “RAFT is a weak baseline” – RAFT is a non-trivial
16

gain over zero-shot on six of eight cells, and Align-RAG still beats it on every one without training
any parameters.
I Cross-dataset retrieval
A natural concern with retrieval-augmented forecasting is whether the retrieval pool needs to come
from the same dataset as the query. We test the cross-dataset (xds) regime by replacing each target’s
same-corpus retrieval index with one drawn from a held-out ETT dataset (a different ETT subset than
the target), keeping every other component of Align-RAG fixed.
Table 12: Cross-dataset retrieval on a frozen Chronos-Bolt at H=64 . “Same-corpus” draws retrievals
from the target’s own training split; “cross-dataset” draws them from a held-out ETT subset. “xds vs.
ZS” is the relative MSE change of cross-dataset Align-RAG vs. zero-shot; “xds vs. same-corpus” is
the relative change vs. same-corpus Align-RAG (positive means same-corpus wins).
Target Bolt-ZS AR(same-corpus) AR(cross-dataset) xds vs. ZS xds vs. same-corpus
ETTh10.3615 0.34340.3418−5.44%−0.46%
ETTh20.25160.23790.2487−1.17% +4.52%
ETTm10.31070.26710.3036−2.27% +13.65%
ETTm20.14860.13840.1427−3.98% +3.06%
Cross-dataset Align-RAG still beats zero-shot on all four ETT targets ( −1.17% to−5.44% MSE), so
the gain does not vanish when retrieval shifts to an out-of-distribution corpus. On ETTh1 the cross-
dataset variant is essentially indistinguishable from the same-corpus variant ( −0.46% ); on ETTh2
and ETTm2 the same-corpus variant is meaningfully better ( +3to+5percentage points); on ETTm1
the gap is largest ( +13.65% ). The takeaway is two-sided: alignment of retrieved demonstrations is
sufficient to extract a non-trivial gain even from cross-dataset retrieval, but the same-corpus regime is
the operating point we recommend when in-domain retrieval is feasible.
17