# Know Before You Fetch: Calibrated Retrieval-Budget Allocation for Retrieval-Augmented Generation

**Authors**: Zhe Dong, Fang Qin, Manish Shah, Yicheng Wang

**Published**: 2026-06-29 08:36:33

**PDF URL**: [https://arxiv.org/pdf/2606.29959v1](https://arxiv.org/pdf/2606.29959v1)

## Abstract
Retrieval-augmented generation (RAG) typically retrieves a fixed number of passages for every query. This is wasteful when the reader already knows the answer, and it can be harmful when irrelevant or partially relevant passages distract the reader. We formulate adaptive RAG as calibrated retrieval-budget allocation: given a query, decide whether to answer closed-book, retrieve a compact context (k=1), retrieve a full context (k=5), or abstain. The contribution is a probability interface rather than a new raw uncertainty signal. We calibrate sequence log-probability and prefix-logit uncertainty signals into probabilities of correctness, then use these probabilities for graded context selection, selective abstention, and explicit latency/token trade-offs. Across core QA experiments on TriviaQA, Natural Questions, and MS MARCO, with auxiliary PopQA motivation and Qwen/Llama family checks, diagnostic out-of-fold calibration improves probability quality dramatically: for sequence log-probability, ECE drops from 0.275 to 0.062 on TriviaQA, 0.643 to 0.009 on NQ, and 0.711 to 0.031 on MS MARCO. Graded retrieval improves full-context and passage-budget frontiers for both our signal and TARG-style prefix entropy/margin, while retrieval-call AUC remains essentially tied with binary gating because k=1 is still a retrieval call. Held-out train/validation/test threshold experiments report deployable operating points. At matched-accuracy frontier operating points, a measured cost model reveals that gating is not universally faster: it increases latency by about 27% on Qwen3-8B but saves about 8% on Qwen3-32B. These results support a nuanced view of adaptive RAG: calibrated confidence is best understood as a reusable interface for allocating retrieval budget under task and system constraints.

## Full Text


<!-- PDF content starts -->

Know Before You Fetch: Calibrated Retrieval-Budget Allocation for
Retrieval-Augmented Generation
Zhe Dong∗
University of Maine at Presque Isle
zhe.dong@maine.edu
dongzhe181@gmail.comFang Qin
Stanford University
fangq@stanford.edu
Manish Shah
Independent Researcher
shahmh@ieee.orgYicheng Wang
Independent Researcher
ethanwang63@163.com
Abstract
Retrieval-augmented generation (RAG) typically retrieves a fixed number of passages for
every query. This is wasteful when the reader already knows the answer, and it can be harmful
when irrelevant or partially relevant passages distract the reader. We formulate adaptive RAG
as calibrated retrieval-budget allocation: given a query, decide whether to answer closed-book,
retrieve a compact context ( k= 1), retrieve a full context ( k= 5), or abstain. The contribution
is a probability interface rather than a new raw uncertainty signal. We calibrate sequence
log-probability and prefix-logit uncertainty signals into probabilities of correctness, then use
these probabilities for graded context selection, selective abstention, and explicit latency/token
trade-offs. Across core QA experiments on TriviaQA, Natural Questions, and MS MARCO, with
auxiliary PopQA motivation and Qwen/Llama family checks, diagnostic out-of-fold calibration
improves probability quality dramatically: for sequence log-probability, ECE drops from 0.275
to 0.062 on TriviaQA, 0.643 to 0.009 on NQ, and 0.711 to 0.031 on MS MARCO. Graded
retrieval improves full-context and passage-budget frontiers for both our signal andTARG-style
prefix entropy/margin, while retrieval-call AUC remains essentially tied with binary gating
becausek= 1is still a retrieval call. Held-out train/validation/test threshold experiments
report deployable operating points. At matched-accuracy frontier operating points, a measured
cost model reveals that gating is not universally faster: it increases latency by about 27% on
Qwen3-8B but saves about 8% on Qwen3-32B. These results support a nuanced view of adaptive
RAG: calibrated confidence is best understood as a reusable interface for allocating retrieval
budget under task and system constraints.
Code and Data Availability
Code, scripts, prompts, per-query decision tables, label-correction files, and archived artifacts are
available at https://github.com/dongzhe1/know-before-you-fetch . The archived release is
available athttps://doi.org/10.5281/zenodo.20954595.
∗Corresponding author.
1arXiv:2606.29959v1  [cs.IR]  29 Jun 2026

1 Introduction
RAG systems couple a retriever with a generative reader, often by unconditionally retrieving a fixed
number of passages and prepending them to every prompt [Lewis et al., 2020, Karpukhin et al., 2020].
This pattern is simple, but it ignores two facts about modern language models. First, many queries
are already answerable from parametric memory. For these queries, retrieval consumes latency,
bandwidth, and context tokens without adding information. Second, retrieval is not monotonic:
additional passages can introduce distractors or entity confusions that make a correct closed-book
answer wrong. A high-quality RAG system should therefore decide not onlywhetherto retrieve,
but alsohow muchevidence to retrieve andwhether to answer at all.
Prior adaptive RAG work has explored self-reflective retrieval triggers, query-complexity routers,
active retrieval during generation, and training-free prefix-logit gates [Asai et al., 2024, Jeong et al.,
2024, Jiang et al., 2023, Wang et al., 2026]. We build on this line of work but shift the object of
study from a raw gate to a calibrated decision interface. A raw uncertainty score may rank queries
well, but it is hard to compare across datasets, readers, and actions. A calibrated probability of
correctness can be used to select thresholds on validation data, to trade accuracy against retrieval
and token costs, and to abstain when neither closed-book nor open-book answers are trustworthy.
The simplest version of the method is intentionally lightweight. For each query, we run the frozen
reader once without retrieval, decode a short greedy answer, and compute its length-normalized
sequence log-probability. A logistic calibrator maps this scalar to P(closed-book answer correct ).
We then allocate one of three context budgets: k= 0(closed-book answer), k= 1(compact
retrieval), or k= 5(full retrieval). For selective RAG, we add a fourth decision: abstain if the
confidence of the chosen answer is too low. We also apply the same calibration-and-allocation layer
toTARG-style prefix entropy and margin signals, treating them as alternative raw features.
The central empirical conclusions are as follows.
1.Calibration substantially improves probability quality while preserving ranking. This is crucial
because the paper’s claim is about probabilistic decisions, not merely AUROC.
2.Graded allocation improves full-context and passage-budget frontiers, but not retrieval-call
frontiers. This distinction prevents the misleading claim thatk= 1retrieval is free.
3.TARGprefix entropy is a strong raw ranking signal. Our framework is complementary: applying
graded calibrated allocation toTARGentropy improves its passage-budget and full-context
frontier.
4.Held-out validation thresholds approach the diagnostic operating points, with high skip rates on
TriviaQA and near-always retrieval on low-headroom NQ/MS MARCO.
5.A realistic two-stage cost model changes the efficiency story. A closed-book probe is an added
generation pass, so gating is faster only when the skip rate clears a measurable break-even
threshold.
2 Related Work
RAG and open-domain QA.Retrieval-augmented generation combines parametric generation
with non-parametric memory access [Lewis et al., 2020]. Dense passage retrieval established a strong
retriever for open-domain question answering [Karpukhin et al., 2020]. We evaluate core QA metrics
on TriviaQA [Joshi et al., 2017], Natural Questions [Kwiatkowski et al., 2019], and MS MARCO
[Nguyen et al., 2016]; PopQA [Mallen et al., 2023] is used as an auxiliary motivation analysis of
popularity and parametric memory.
2

Adaptive retrieval.Self-RAG fine-tunes a model to emit reflection tokens for retrieval and
critique [Asai et al., 2024]. Adaptive-RAG routes questions by predicted complexity [Jeong et al.,
2024]. FLARE retrieves during generation when token confidence drops [Jiang et al., 2023].TARG
makes a training-free retrieval decision from prefix-logit uncertainty in a short no-context draft
[Wang et al., 2026]. Our focus differs: we turn uncertainty signals into calibrated probabilities and
use them for multi-action budget allocation.
Calibration and selective prediction.Neural networks are often miscalibrated despite high
accuracy [Guo et al., 2017]. Selective prediction studies when to abstain in exchange for lower risk
at lower coverage [Geifman and El-Yaniv, 2017, 2019]. We adapt this view to RAG: retrieval is an
information-acquisition action between predicting and abstaining.
LLM self-knowledge.Language models partly know when they know [Kadavath et al., 2022],
and entity popularity affects the boundary between parametric and retrieved memory [Mallen et al.,
2023]. We use these signals operationally to decide when retrieval is useful.
3 Problem Setup and Method
3.1 Per-query outcome table
For each query qi, an evaluation table contains the closed-book answer a(0)
i, open-book answers
a(1)
ianda(5)
iusing top-1 and top-5 passages, correctness labels y(0)
i,y(1)
i,y(5)
i∈{0,1}, and one or
more uncertainty scores. We do not train the reader or retriever. Instead, the decision policy selects
which precomputed action would be used at deployment.
3.2 Calibration
Given a scalar signal si, such as length-normalized sequence log-probability, we fit a logistic calibrator
pi= Pr(y(0)
i= 1|s i) =σ(αs i+β).(1)
We use two calibration protocols. Unless otherwise stated, calibration tables and frontier plots are
diagnostic out-of-fold estimates on the evaluation pool: each query is scored by a calibrator not
trained on that query, and thresholds are swept to characterize attainable trade-offs. Deployment
experiments use train/validation/test splits: train fits the calibrator, validation selects thresholds,
and test evaluates once. ForTARGcomparisons, we apply the same calibration protocol to prefix
entropy and top-1/top-2 margin.
3.3 Policies and cost axes
A binary policy skips retrieval when pi≥τand otherwise retrieves k= 5. A graded policy uses two
thresholds:
ai=

0, p i≥τ0,
1, τ 1≤pi<τ0,
5, p i<τ1.(2)
3

For a graded policy, we report three x-axes:
xcall= Pr(a i>0),(3)
xfull= Pr(a i= 5),(4)
xpassage = (Pr(a i= 1) + 5 Pr(a i= 5))/5.(5)
This is the key accounting correction. Compact retrieval reduces passage budget and full-context
usage, but it is still a retrieval call.
3.4 Selective RAG
For abstention, we use the confidence of the answer the system actually returns. If the policy skips,
the confidence is the calibrated closed-book probability; if it retrieves, the confidence is a separate
logistic calibration of the length-normalized sequence log-probability of the selected k= 5open-book
answer against open-book correctness. Sorting by this proper chosen-answer confidence yields a
risk-coverage curve. AURC is lower when the system can remove its likely errors early.
3.5 Latency and token cost
Always-RAG costsc ret,5+cOB,5. A binary gate costs
Cbinary =cCB+p5(cret,5+cOB,5),(6)
so it is faster only when
Pr(k= 0)>cCB
cret,5+cOB,5.(7)
A graded gate costs
Cgraded =cCB+p1(cret,1+cOB,1) +p 5(cret,5+cOB,5).(8)
We separately report retrieval calls, passage tokens, and measured latency because these costs differ
across deployments.
4 Experimental Setup
Models.Main experiments use Qwen3-8B. Scaling experiments use Qwen3-1.7B and Qwen3-32B;
family checks use Qwen3.5-9B and Llama-3.1-8B [Qwen Team, 2025, 2026, Llama Team, AI @ Meta,
2024]. We use greedy decoding for answer generation.
Retrievers and corpora.TriviaQA uses BGE-large by default and BGE-small for robustness
[Xiao et al., 2024]; the exact checkpoints are BAAI/bge-large-en-v1.5 and BAAI/bge-small-en-v1.5.
NQ uses DPR. MS MARCO uses its passage pool. We include shared-corpus variants and a full
Wikipedia index of 3.5M passages for a harsher corpus-scale condition.
Baselines.We compare against random skipping, query length,TARGentropy/margin, a Self-
RAG trigger under a matched-reader decision table, and an Adaptive-RAG-inspired query-text
classifier. The latter two are not full pipeline reproductions; they isolate decision quality under a
fixed reader.
4

Table 1: Calibration quality for Qwen3-8B. Raw metrics use min-max normalized scores; calibrated
metrics use diagnostic out-of-fold logistic probabilities. Lower is better for ECE/Brier/NLL.
Dataset Signal AUROC Raw ECE Cal. ECE Raw Br. Cal. Br. Cal. NLL
TriviaQASeq-logprob .790 .275 .062 .268 .182 .540
TARG entropy .810 .268 .045 .259 .172 .519
TARG margin .719 .185 .039 .244 .210 .609
Fused valid .808 .081 .050 .179 .173 .520
NQSeq-logprob .740 .643 .009 .581 .162 .487
TARG entropy .749 .631 .017 .565 .160 .483
TARG margin .657 .092 .032 .184 .174 .528
Fused valid .744 .214 .021 .225 .161 .484
MS MARCOSeq-logprob .682 .711 .031 .627 .116 .387
TARG entropy .694 .709 .016 .622 .115 .384
TARG margin .583 .218 .009 .182 .121 .404
Fused valid .692 .315 .014 .235 .115 .385
Evaluation.Accuracy is answer correctness under the selected action. Frontier AUC integrates
accuracy over the relevant budget axis. Calibration is measured by AUROC, ECE, Brier score, and
NLL. Harm is the fraction of examples where the closed-book answer is correct but the open-book
answer at the reported kis incorrect. Saved is the fraction of harmful retrievals avoided by the
selected gate operating point. MS MARCO has noisier and more diverse answer semantics than
TriviaQA/NQ, so we treat it as a low-headroom boundary dataset rather than the strongest evidence
for the method. We report bootstrap intervals for frontier deltas. Correctness labels are audited;
corrections are released separately and do not change qualitative conclusions.
5 Main Results
5.1 Calibration quality
Table 1 shows that calibration is not cosmetic. Raw min-max normalized sequence log-probability
has poor ECE and Brier score, especially on NQ and MS MARCO, whereas diagnostic out-of-fold
logistic calibration produces well-behaved probabilities.TARGentropy is often a stronger ranking
signal than sequence log-probability, but it also benefits from calibration.
5.2 Graded budget allocation
Figure 1 and Table 2 separate the three cost axes. On TriviaQA-8B, sequence log-probability has
binary call-AUC 0.731. Grading raises full-context AUC to 0.773 and passage-budget AUC to 0.760,
but call-AUC remains 0.732. The same framework improvesTARGentropy: binary call-AUC 0.736
becomes full-context AUC 0.775 and passage-budget AUC 0.762. Results at 32B show the same
pattern with smaller deltas.
5.3 Scaling, family transfer, and difficulty regimes
Figure 2 shows the scaling trend. On TriviaQA, closed-book accuracy increases from 0.273 to 0.693
across Qwen3-1.7B, 8B, and 32B; the retrieval rate needed to match always-RAG falls from 99.5%
5

Figure1: Gradedallocationimprovesfull-contextandpassage-budgetfrontiers. Itdoesnotmaterially
improve retrieval-call AUC becausek= 1is still a retrieval call.
Table 2: Binary vs. graded frontier AUC on TriviaQA. Deltas are relative to each signal’s binary
call-AUC.
Model Signal Binary Full ctx Call Passage
8BSeq-logprob .731 .773 .732 .760
TARG entropy .736 .775 .737 .762
TARG margin .719 .761 .720 .746
32BSeq-logprob .793 .810 .793 .804
TARG entropy .800 .812 .801 .806
TARG margin .603 .707 .604 .671
to 46.0%. The operational skip rate therefore grows from almost zero to 54%.
Cross-family results are consistent. Qwen3.5-9B and Llama-3.1-8B have higher closed-book
accuracy than Qwen3-8B on TriviaQA and retain strong gate AUROC. This suggests that the
method is not tied to a single reader family, although thresholds should still be calibrated per model.
5.4 Held-out deployment thresholds
Table 3 reports deployable near-match thresholds. On TriviaQA, validation-selected thresholds
skip roughly one-third of queries while approaching always-RAG accuracy: sequence log-probability
comes within 1.5 percentage points at 65.2% retrieval. On NQ and MS MARCO, the same protocol
chooses near-always retrieval because closed-book accuracy is low and open-book retrieval provides
most of the utility. This is the desired behavior: the policy should not force skipping when the task
has little closed-book headroom.
5.5 Selective RAG and abstention
Figure 4 evaluates abstention. Proper chosen-answer confidence dominates naive closed-book
confidence on TriviaQA and NQ. At 80% coverage, TriviaQA accuracy improves from 80.8% to
86.0%, and NQ from 66.9% to 75.0%. MS MARCO is a useful negative case: proper confidence
6

Figure 2: Scaling trend on TriviaQA. Larger readers can skip more retrieval while maintaining
always-RAG accuracy.
Table 3: Held-out threshold deployment for near-match operating points. Thresholds are selected
on validation folds and evaluated once on test folds.
Dataset Signal Always Acc. Ret. rate
TriviaQA Seq-logprob .785 .770 .652
TriviaQA TARG entropy .785 .765 .623
NQ Seq-logprob .685 .682 .987
NQ TARG entropy .685 .678 .975
MS MARCO Seq-logprob .310 .303 .973
MS MARCO TARG entropy .310 .302 .973
beats random but not naive closed-book confidence, likely because open-book confidence has low
headroom and the dataset contains noisier answer semantics.
5.6 Measured cost regimes
A closed-book probe is a real generation pass. Table 4 applies the measured timings at oracle
frontier operating points that match always-RAG accuracy on the evaluation pool; Table 3 reports
deployable validation-selected thresholds. For Qwen3-8B, always-RAG costs 107.5 ms per query,
while binary gating costs 136.1 ms and graded gating costs 133.3 ms. For Qwen3-32B, always-RAG
costs 346.5 ms, while binary and graded gating cost 319.9 ms and 316.1 ms. Figure 5 visualizes the
break-even condition in Equation 7.
6 Additional Analyses
6.1 Scale-up to 2k examples
Table 5 verifies the main conclusion on larger samples. TriviaQA has the largest open-book headroom
and the strongest frontiers. NQ has lower closed-book accuracy but still large passage-budget gains
7

Figure 3: Cross-family and cross-dataset behavior. The gate is strongest when the reader has
meaningful closed-book headroom but retrieval still provides rescue opportunities.
Figure 4: Selective RAG risk-coverage curves. Proper chosen-answer confidence helps on TriviaQA
and NQ; MS MARCO is a low-headroom boundary case.
from grading. MS MARCO is low-headroom: retrieval helps less, label noise is more visible, and
saved-at-match is limited.
6.2 Retriever quality and corpus scale
Table 6 shows that absolute RAG accuracy depends strongly on retriever quality. The full-Wikipedia
condition is much harsher than the per-query pool, especially for NQ. We interpret this as a stress
test: confidence remains predictive of reader knowledge, but no gating method can compensate for
a very noisy retriever if the goal is maximum answer accuracy.
6.3 Decision-signal baselines
Figure 6 compares against Self-RAG trigger and Adaptive-RAG-inspired decision signals under a
fixed reader. This is a deliberately conservative comparison: it isolates the retrieve/skip decision
8

Table 4: Measured cost at oracle matched-accuracy operating points on TriviaQA. Pass. is the
normalized passage budget; deployable validation-selected operating points appear in Table 3.
Model Policy Skip Call Full Pass. ms/q
Qwen3-8B Binary .335 .665 .665 .665 136.1
Qwen3-8B Graded .325 .675 .540 .567 133.3
Qwen3-32B Binary .542 .458 .458 .458 319.9
Qwen3-32B Graded .528 .472 .355 .378 316.1
Table 5: Scale-up ton= 2000with Qwen3-8B. Graded AUC uses the passage-budget axis.
Dataset CB OB@1 OB@5 AUROC Bin. Grad.
TriviaQA .541 .778 .842 .810 .754 .792
NQ .238 .614 .660 .727 .491 .598
MS MARCO .148 .304 .309 .665 .234 .284
rather than mixing reader training, retrieval format, and answer generation. Confidence-based
decisions are stronger under this matched protocol, but this is not a full Self-RAG or Adaptive-RAG
pipeline reproduction.
6.4 Why not just retrieve more?
On TriviaQA, k= 10is worse than k= 5for both Qwen3-8B and Qwen3-32B. For 8B, accuracy
falls from 0.785 at k= 5to 0.730 at k= 10; for 32B, from 0.822 to 0.760. In both cases roughly
8.5% of examples correct at k= 5become wrong at k= 10. This supports the broader claim that
retrieval budget should be allocated, not maximized.
6.5 Popularity and parametric memory
Figure 7 reproduces the PopQA motivation: parametric recall is higher for popular entities, but
retrieval can still rescue tail entities. Confidence gating operationalizes this observation at the query
level rather than relying only on entity popularity.
7 Discussion
What calibration adds.If the objective is only to rank queries by whether closed-book answering
is safe, prefix entropy may be an excellent signal. Calibration adds a different capability: it lets the
system compare risks across actions and thresholds. This matters for validation-selected thresholds,
abstention, cost-sensitive deployment, and combining heterogeneous signals.
When the method helps.The method is strongest when three conditions hold: closed-book
accuracy is nontrivial, retrieval has enough headroom to rescue low-confidence queries, and retrieval
can also harm high-confidence queries. TriviaQA exhibits all three. NQ has strong retrieval
headroom but low closed-book accuracy, so the policy usually retrieves. MS MARCO has low
headroom and noisier semantics, so benefits are smaller and selective RAG has a negative case.
9

Figure5: Cost-regimeanalysis. Curvessubstitutemodel-specificmeasured cCBandvaryρ=cret/cOB
in Equation 7; gating is faster only above the resulting break-even skip threshold.
How to deploy.A practical deployment should first decide its dominant cost: retrieval calls,
context tokens, latency, or monetary token price. It should then calibrate on a small validation set,
select thresholds on the relevant budget axis, and verify the break-even condition in Equation 7.
For high-stakes domains, retrieval may be mandatory even for high-confidence queries.
8 Limitations
Most experiments are on short-answer QA, where correctness can be judged with aliases. Long-form
answers require evidence attribution and human evaluation. The measured latency numbers are
batch-1 and hardware-specific; production batching, asynchronous retrieval, and KV-cache reuse
may shift the break-even point. We do not reproduce full Self-RAG or Adaptive-RAG pipelines,
and our comparison to those methods should be read as decision-signal evidence only. Finally,
calibration can drift under domain shift, so deployment requires monitoring.
9 Ethical Considerations
Adaptive retrieval can reduce unnecessary computation and can encourage abstention when confi-
dence is low. The main risk is over-trusting calibrated confidence outside the domain on which it
was validated. Miscalibration could lead to skipped retrieval for questions that require up-to-date
evidence, or unequal abstention rates across user groups and topics. We recommend conservative
10

Table 6: Retriever and corpus robustness. Wiki uses a 3.5M-passage index. Harm is closed-book-
correct/open-book-wrong; Saved is the fraction of harmful retrievals avoided at the selected operating
point.
Dataset Retrieval OB@5 AUC Grad. Harm Saved
TriviaQA BGE-large pool .785 .731 .747 .102 .298
TriviaQA BGE-small pool .782 .731 .741 .105 .342
TriviaQA Shared corpus .768 .724 .738 .108 .330
TriviaQA Wiki index .595 .619 – .251 .912
NQ DPR pool .685 .509 .566 .115 .008
NQ Shared corpus .653 .488 .545 .122 .008
NQ Wiki index .312 .295 – .351 .318
MS MARCO Pool .310 .235 .263 .153 .000
Figure 6: Matched-reader decision-signal baselines. “Self-RAG trigger signal” denotes the trigger
evaluated on the same reader table, not a full Self-RAG pipeline.
thresholds in high-stakes settings, audits across subpopulations and domains, and policies that
mandate retrieval when evidence is required for safety, compliance, or transparency. The experiments
use public datasets and do not introduce new human-subject data.
10 Conclusion
We presented calibrated retrieval-budget allocation for RAG. The main lesson is not that a single
uncertainty signal wins everywhere, but that calibrated probabilities provide a reusable interface
for deciding among no retrieval, compact retrieval, full retrieval, and abstention. This interface
complements strong training-free signals such asTARG, clarifies the distinction between retrieval
calls and context budget, and exposes the cost regimes in which adaptive retrieval truly saves
latency.
11

Figure 7: Motivation from PopQA: parametric memory and retrieval value vary with popularity.
A Extended Method Details
A.1 Prompting and decoding
Closed-book prompts ask the reader to answer the question directly in a short phrase. Open-book
prompts prepend retrieved passages and ask for the final answer. Main generations use greedy
decoding with a short answer length cap. Sequence log-probability is length-normalized over
generated answer tokens to avoid mechanically favoring short outputs.
A.2 Answer evaluation and label audit
Correctness uses normalized string matching and alias sets when available. Manual audit identified
a small number of semantic-equivalence and yes/no errors, especially in MS MARCO. Corrected
tables are included as sensitivity artifacts. Corrections do not change qualitative conclusions; the
largest numerical change is MS MARCO AUROC, which increases by about two percentage points
after fixing obvious label errors.
12

Figure 8: Three-action view of skip, retrieve, and abstain. Accuracy is computed over answered
queries, so it should be read together with coverage.
B Additional Figures
C Calibration Ablations
D Token-Cost Comparison
At a retrieval rate of 35%, the average prompt-token estimate is 306 for our full-answer probe,
314 for a 32-tokenTARGprefix, 346 for a 64-token prefix, and 410 for a 128-token prefix. The
comparison is intentionally transparent:TARGcan be cheaper when a very short prefix suffices,
whereas our full closed-book answer can be reused directly when the policy skips retrieval.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection. InInternational Conference on Learning
Representations, 2024.
13

Figure 9: Difficulty and retrieval-harm analysis.
Table 7: Calibration method comparison for sequence log-probability.
Dataset Method AUROC ECE
TriviaQA Logistic .790 .062
TriviaQA Isotonic .768 .068
TriviaQA Temperature .789 .199
TriviaQA Raw norm. .793 .275
NQ Logistic .740 .009
NQ Isotonic .728 .027
NQ Temperature .742 .018
NQ Raw norm. .745 .643
MS MARCO Logistic .682 .031
MS MARCO Isotonic .667 .018
MS MARCO Temperature .684 .127
MS MARCO Raw norm. .685 .711
Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. InAdvances
in Neural Information Processing Systems, 2017.
Yonatan Geifman and Ran El-Yaniv. SelectiveNet: A deep neural network with an integrated reject
option. InProceedings of the 36th International Conference on Machine Learning, volume 97 of
Proceedings of Machine Learning Research, pages 2151–2159, 2019.
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural
networks. InProceedings of the 34th International Conference on Machine Learning, pages
1321–1330, 2017.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-RAG:
Learning to adapt retrieval-augmented large language models through question complexity.
InProceedings of the 2024 Conference of the North American Chapter of the Association for
14

Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages
7036–7050, 2024.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing, 2023.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehension. InProceedings of the 55th Annual Meeting
of the Association for Computational Linguistics, pages 1601–1611, 2017.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk,
Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Samuel R. Bowman, Stanislav
Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane
Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark,
Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models
(mostly) know what they know.arXiv preprint arXiv:2207.05221, 2022.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages
6769–6781, 2020.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research.Transactions of the
Association for Computational Linguistics, 7:452–466, 2019.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems, volume 33, pages 9459–9474, 2020.
Llama Team, AI @ Meta. The Llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi.
When not to trust language models: Investigating effectiveness of parametric and non-parametric
memories. InProceedings of the 61st Annual Meeting of the Association for Computational
Linguistics, pages 9802–9822, 2023.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and
Li Deng. MS MARCO: A human generated machine reading comprehension dataset. InProceedings
of the Workshop on Cognitive Computation: Integrating Neural and Symbolic Approaches at NIPS
2016, CEUR Workshop Proceedings, 2016.
Qwen Team. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.
ai/blog?id=qwen3.5.
15

Yufeng Wang, Lu Wei, and Haibin Ling. Retrieval as a decision: Training-free adaptive gating for
efficient RAG.Transactions on Machine Learning Research, 2026.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-Pack:
Packed resources for general chinese embeddings. InProceedings of the 47th International ACM
SIGIR Conference on Research and Development in Information Retrieval, pages 641–649, 2024.
16