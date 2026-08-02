# FinAbstain: Uncertainty-Calibrated Multimodal RAG for Selective Financial Forecasting

**Authors**: Dorothy Torres, Wei Cheng, Henan Huang

**Published**: 2026-07-27 07:05:22

**PDF URL**: [https://arxiv.org/pdf/2607.24875v1](https://arxiv.org/pdf/2607.24875v1)

## Abstract
Large language models (LLMs) can synthesize financial narratives but may express high confidence when evidence is sparse, stale, or contradictory. This failure is especially consequential in forecasting, where filings, news, prices, volume, and technical signals can disagree. We present FinAbstain, a research framework for uncertainty-calibrated multimodal retrieval-augmented generation (RAG) with selective prediction. A point-in-time retriever admits only information public at the forecast timestamp and supplies modality-specific evidence to fundamental, news, technical, risk, and verification agents. Their probabilistic assessments are aggregated with retrieval relevance, evidence contradiction, repeated-sample consistency, and historical calibration statistics. Temperature scaling, isotonic regression, conformal prediction, and a proposed hybrid uncertainty score are evaluated under a common chronological protocol. A controller predicts bullish, bearish, or neutral outcomes only when uncertainty is below a validated threshold; otherwise it abstains, requests evidence, reduces exposure, or routes the case to human review. The evaluation covers one- and five-day abnormal-return direction, twenty-day volatility intervals, and abstention decisions, using accuracy, calibration, risk--coverage, citation, trading, latency, and cost metrics. To make the design auditable before a full data collection is complete, we report explicitly labeled simulated results rather than empirical claims. These results illustrate the intended hypothesis: calibrated abstention may trade coverage for lower selective error and drawdown. The contribution is a time-safe architecture, a composite uncertainty formulation, and a reproducible evaluation blueprint for evidence-grounded selective financial forecasting.

## Full Text


<!-- PDF content starts -->

FinAbstain: Uncertainty-Calibrated Multimodal RAG
for Selective Financial Forecasting
Dorothy Torres
School of Science, Technology, Engineering and Mathematics
263 Academy Street, Room S204
Jersey City, NJ 07306Wei Cheng, Henan Huang
School of Electrical Engineering and Computer Science
The Pennsylvania State University
207 Electrical Engineering West
University Park, PA 16802, USA
Abstract—Large language models (LLMs) can synthesize
financial narratives but may express high confidence when
evidence is sparse, stale, or contradictory. This failure is especially
consequential in forecasting, where filings, news, prices, volume,
and technical signals can disagree. We present FinAbstain,
a research framework for uncertainty-calibrated multimodal
retrieval-augmented generation (RAG) with selective prediction.
A point-in-time retriever admits only information public at
the forecast timestamp and supplies modality-specific evidence
to fundamental, news, technical, risk, and verification agents.
Their probabilistic assessments are aggregated with retrieval
relevance, evidence contradiction, repeated-sample consistency,
and historical calibration statistics. Temperature scaling, isotonic
regression, conformal prediction, and a proposed hybrid uncer-
tainty score are evaluated under a common chronological protocol.
A controller predicts bullish, bearish, or neutral outcomes only
when uncertainty is below a validated threshold; otherwise it
abstains, requests evidence, reduces exposure, or routes the case to
human review. The evaluation covers one- and five-day abnormal-
return direction, twenty-day volatility intervals, and abstention
decisions, using accuracy, calibration, risk–coverage, citation,
trading, latency, and cost metrics. To make the design auditable
before a full data collection is complete, we report explicitly
labeled simulated results rather than empirical claims. These
results illustrate the intended hypothesis: calibrated abstention
may trade coverage for lower selective error and drawdown. The
contribution is a time-safe architecture, a composite uncertainty
formulation, and a reproducible evaluation blueprint for evidence-
grounded selective financial forecasting.
Index Terms—financial forecasting, retrieval-augmented gen-
eration, multimodal learning, uncertainty calibration, selective
prediction, abstention, LLM agents
I. INTRODUCTION
Financial forecasting is difficult not merely because markets
are noisy, but because evidence arrives asynchronously and
may point in different directions. An earnings filing can reveal
improving margins while a contemporaneous news shock
depresses sentiment; momentum can remain positive while
option-implied or realized volatility rises. The efficient-market
perspective further cautions that public information is rapidly
incorporated into prices [1]. Language models nevertheless
tend to produce a fluent answer even when their evidence is
incomplete. In a decision system, such overconfidence can
convert an epistemic limitation into an avoidable position.
Retrieval-augmented generation (RAG) grounds generation
in external documents [2], and domain models such as FinBERT
and BloombergGPT improve financial language representations[3]–[5]. Yet three gaps remain. First, ordinary RAG does not
guarantee that a document was public at the decision timestamp,
so a backtest may leak future information. Second, a single
synthesis can hide disagreement among fundamental, news, and
market signals. Third, confidence is rarely calibrated for the
operational option of refusing a forecast. Backtest overfitting
and data snooping amplify these concerns [6], [7].
FinAbstain addresses this gap by coupling point-in-time mul-
timodal retrieval, role-separated agents, post-hoc and conformal
calibration, and a selective controller. Abstention is treated as
a decision, not a missing output: the system can request more
evidence, reduce position size, or refer the case to a person.
This paper makes four contributions:
•a temporal RAG specification for filings, calls, news,
prices, volume, and technical indicators that enforces
availability at the forecast timestamp;
•a multi-agent evidence protocol that exposes disagreement,
contradiction, and citation support rather than collapsing
them into one narrative;
•a hybrid uncertainty score and formal controller con-
necting calibration, coverage, selective risk, and position
sizing; and
•a reproducible chronological evaluation and backtesting
blueprint, with clearly labeled simulated tables and figures
that are not empirical findings.
II. RELATEDWORK
A. Financial Language and Forecasting
Financial sentiment models adapt contextual encoders to
specialized vocabulary and disclosure style [3], [4]. Large-
scale financial LLMs broaden generative capability [5], while
FinQA demonstrates numerical reasoning over reports [8].
Textual forecasting complements a long literature on return
predictability and technical rules. Early evidence found apparent
value in technical analysis [9], whereas later work stresses
instability, costs, and data snooping [10]. Recent studies
further explore sensitivity-calibrated MACD design and cost–
performance trade-offs in LLM context management [11], [12].
These strands motivate multimodal evidence but do not supply
calibrated refusal.
arXiv:2607.24875v1  [cs.LG]  27 Jul 2026

B. Retrieval and Agentic Reasoning
RAG retrieves passages before generation [2]; dense passage
retrieval provides a standard semantic retrieval mechanism
[13]. Multimodal retrieval models such as CLIP show how
heterogeneous representations can share a contrastive space
[14], although financial time series require modality-specific
encoders and strict timestamps. Tool-using and reasoning
agents decompose complex tasks [15], and multi-agent debate
can reveal inconsistent reasoning [16]. In finance, however,
independence is useful only if agents cite common, time-valid
evidence and aggregation preserves their conflicts.
C. Calibration and Selective Prediction
Modern neural networks are often miscalibrated; temperature
scaling is a strong post-hoc baseline [17]. Isotonic regression is
flexible but may overfit small calibration sets [18]. Conformal
prediction supplies finite-sample marginal coverage under
exchangeability [19], [20], an assumption weakened by market
regime shifts. Selective classification formalizes the risk–
coverage trade-off [21], while selective generation extends
refusal to free-form outputs [22]. FinAbstain integrates these
ideas with evidence quality and trading consequences rather
than treating probability calibration as an isolated diagnostic.
III. PROBLEMFORMULATION
Let asset ibe evaluated at forecast time t. Its point-in-time
evidence is
Ei,t=[
m∈M{e(m)
j:τpub
j≤t, τing
j≤t},(1)
where modalities Minclude filings, calls, news, prices, volume,
and indicators; τpubandτingare public and system-ingestion
timestamps. Both constraints prevent future leakage.
For class set Y={bearish,neutral,bullish} , agent a∈
{1, . . . , A} returns probabilities pa,i,t∈∆2, a rationale, and
cited evidence. Pairwise agent disagreement is the mean Jensen–
Shannon divergence
Di,t=2
A(A−1)X
a<bJSD(p a,i,t∥pb,i,t).(2)
If the verifier labels a retrieved claim pair (j, k) with contra-
diction probability qjk, its relevance-weighted contradiction
score is
Ci,t=P
j<krjrkqjkP
j<krjrk+ϵ,(3)
where rj∈[0,1] is retrieval relevance and ϵ >0 stabilizes the
ratio.
The aggregated class probability is ¯pi,t. LetSbe one minus
agreement across repeated inference samples, R= 1−¯r
retrieval deficiency, Hnormalized predictive entropy, and G
recent calibration gap. The proposed uncertainty is
Ui,t=σ(w 0+wDDi,t+wCCi,t+wSSi,t
+wRRi,t+wHHi,t+wGGi,t),(4)with validation-fitted weights wand logistic function σ. The
controller predicts whenU i,t≤θand otherwise abstains:
ˆyi,t=(
arg max y¯pi,t(y), U i,t≤θ,
⊥, U i,t> θ.(5)
For acceptance indicator gθ(x) =⊮[U(x)≤θ] , coverage
and selective risk under lossℓare
Cov(θ) =1
nX
ngθ(xn),Risk(θ) =P
ngθ(xn)ℓ(ˆyn, yn)P
ngθ(xn) +ϵ.
(6)
ForBconfidence bins, expected calibration error is
ECE =BX
b=1|Ib|
n|acc(I b)−conf(I b)|.(7)
The threshold θis selected only on validation data to minimize
selective risk subject to a minimum coverage cmin, or to
minimize a cost-sensitive loss that includes abstention and
erroneous trades.
IV. PROPOSEDFINABSTAINFRAMEWORK
A. Temporal Multimodal Retriever
Figure 1 summarizes the system. SEC 10-K, 10-Q, and 8-K
filings, earnings-call transcripts, timestamped financial news,
adjusted prices, volume, MACD, RSI, moving averages, and
historical volatility are stored with event, publication, ingestion,
and revision timestamps. At t, an immutable snapshot excludes
documents published later and market bars unavailable before
the chosen close. Filing amendments are versioned; news is
keyed by first-public time; indicators are recomputed only from
bars ending at t. Text is retrieved by a hybrid dense/BM25
score, while time series are matched by regime and recency.
The top- kbudget is allocated across modalities so that prolific
news cannot crowd out filings or market evidence.
B. Role-Separated Financial Agents
The fundamental agent extracts changes in revenue, margins,
cash flow, leverage, guidance, and accounting risks. The news
agent estimates event polarity, novelty, and source diversity.
The technical agent receives only numerical features and
describes trend, momentum, liquidity, and volatility. The risk
agent constructs the strongest counterargument and enumerates
missing evidence. The verifier checks every material claim
against retrieved spans, detects cross-source contradictions, and
rejects citations that do not entail the claim. Agents analyze
independently before seeing peer outputs, reducing anchoring.
A second aggregation pass uses reliability weights estimated
on past validation windows; it cannot introduce uncited facts.
C. Calibration and Selective Control
We compare raw maximum probability, temperature scaling,
isotonic regression, split conformal prediction, and the hybrid
score in (4). Temperature and isotonic mappings are fitted
on rolling validation windows. Conformal class sets use
nonconformity 1−¯p y; sets with zero or multiple classes trigger
abstention unless a cost rule permits neutral. Because temporal

Filings, calls, news
prices, volume, indicatorsPoint-in-time
multimodal retrieverFundamental
News sentiment
Technical
Risk/counterargument
Evidence verifierEvidence and
probability aggregationUncertainty
calibrationSelective
controllerPredict / abstain
human review
Fig. 1. Overall FinAbstain architecture. Every evidence path is filtered by public availability at the decision timestamp.
dependence challenges classical exchangeability, coverage is
reported by regime and month rather than asserted as an
unconditional guarantee.
Accepted outputs include citations, probabilities, and uncer-
tainty factors. High uncertainty invokes one of four policies:
retrieve an additional modality, abstain, reduce target exposure
by1−U , or refer to an analyst. Human review is mandatory
when contradiction exceeds a validation-set safety threshold
even if aggregate entropy is low.
V. EXPERIMENTALMETHODOLOGY
A. Data, Tasks, and Splits
The proposed study uses constituents of the S&P 100
with survivorship-aware membership histories. SEC EDGAR
supplies filings [23]; market data are adjusted consistently for
corporate actions. Table I specifies the planned corpus. Counts
are simulated planning values pending licensed-news ingestion
and deduplication; they must be replaced by audited counts
before an empirical submission.
One- and five-day labels are the sign of market-model
abnormal return with a neutral band estimated from training
volatility. The twenty-day task predicts low, medium, or high
realized-volatility intervals. No temporal sequence is randomly
shuffled. Hyperparameters, retrieval k, calibration mappings,
label bands, and θare selected on validation data. The final
test is touched once. Seeds, prompts, model versions, evidence
snapshots, and token accounting are logged.
B. Baselines and Metrics
Baselines are (1) a gradient-boosted technical-indicator
classifier, (2) a FinBERT sentiment model, (3) a single-LLM
predictor, (4) standard multimodal RAG, (5) multi-agent RAG
without calibration, (6) a calibrated model without abstention,
and (7) full FinAbstain. Metrics are accuracy, macro F1,
AUROC, Brier score, ECE, selective accuracy, coverage, risk–
coverage area, abstention precision, citation correctness, Sharpe
ratio, maximum drawdown (MDD), transaction-cost-adjusted
return, tokens, and end-to-end latency.
Let accepted position zt∈[−1,1] , asset return rt+1, turnover
|zt−zt−1|, and proportional costκ. Net strategy return is
rnet
t+1=ztrt+1−κ|z t−zt−1|.(8)With mean ¯r, standard deviation sr, and annualization factor
K,
Sharpe =√
K¯r
sr,MDD = max
t
1−Wt
max u≤tWu
,
(9)
where Wt=Q
u≤t(1 +rnet
u). We report costs at 5, 10, and
20 basis points and use block bootstrap confidence intervals.
Statistical comparisons use paired block resampling across
identical asset–date cases.
VI. RESULTS ANDDISCUSSION
A. Illustrative Main Results
Table II and all plots in this section contain simulated
values constructed to test reporting, plotting, and consistency.
They are not outcomes of an executed backtest. Under this
scenario, FinAbstain has the best calibration and selective
accuracy while covering 72% of cases. The calibrated, non-
abstaining model isolates the effect of selection: its ECE
improves, but accuracy and drawdown remain below the
selective policy. The comparison is consistent with, but does
not prove, the hypothesis that refusal concentrates decisions
on better-supported cases.
Figure 2 shows the designed risk–coverage behavior. At equal
coverage, the FinAbstain curve remains below the alternatives
in the simulation; its advantage grows when the controller
accepts only the most supported cases. Operationally, very low
coverage is not automatically optimal: fixed costs, opportunity
costs, and selection-induced regime concentration must be
considered.
B. Calibration and Ablations
The reliability diagram in Fig. 3 compares raw and hybrid
confidence. The simulated hybrid curve is closer to the diagonal,
particularly in the 0.7–0.9 region where overconfident trades
would otherwise accumulate. Table III illustrates diagnostic
ablations. Removing temporal filtering appears deceptively
favorable in accuracy because the simulated variant is inten-
tionally contaminated; it is invalid and excluded from any fair
comparison. This row demonstrates why predictive scores alone
cannot certify a financial backtest.
Calibration must also be examined under drift. A mapping
fitted in a low-volatility period may fail during a crisis;
therefore the production design monitors rolling ECE, class-
conditional coverage, evidence age, and abstention frequency.
High abstention precision means rejected cases are indeed

TABLE I
DATASET AND TASK STATISTICS(ALL COUNTS ARE SIMULATED PLANNING VALUES,NOT OBSERVATIONS FROM A COMPLETED EXPERIMENT).
Sources Firms Period Documents News Market samples Tasks / chronological split
10-K/10-Q/8-K,
calls, licensed news,
OHLCV , MACD,
RSI, MA, volatility100 2015–2024 46,820 1,284,000 226,400 1-day and 5-day abnormal-
return direction; 20-day
volatility interval; abstain.
Train 2015–2020, validation
2021–2022, test 2023–2024.
TABLE II
MAIN EXPERIMENTAL RESULTS(SIMULATED ILLUSTRATIVE VALUES ONLY; ACC.ANDF1ARE FOR THE FIVE-DAY DIRECTION TASK;HIGHER IS BETTER
EXCEPTBRIER, ECE,ANDMDD).
Method Acc. Macro F1 Brier ECE Sel. Acc. Coverage Sharpe MDD
Technical classifier .542 .518 .636 .112 .542 1.00 .48 .184
Financial sentiment .551 .527 .621 .104 .551 1.00 .53 .176
Single LLM .566 .548 .598 .128 .566 1.00 .57 .171
Standard multimodal RAG .584 .565 .572 .091 .584 1.00 .68 .158
Multi-agent RAG, uncalibrated .601 .583 .548 .083 .601 1.00 .76 .151
Calibrated, no abstention .606 .590 .529 .037 .606 1.00 .81 .146
FinAbstain .632 .617 .487 .026 .688.721.08 .112
0.4 0.5 0.6 0.7 0.8 0.9 10.20.30.4
CoverageSelective errorSingle LLM
Standard RAG
Multi-agent RAG
FinAbstain
Fig. 2. Risk–coverage curves using simulated illustrative values.
TABLE III
ABLATION STUDY(SIMULATED ILLUSTRATIVE VALUES).
Variant Sel. Acc. ECE Coverage
Full FinAbstain .688 .026 .72
No disagreement .665 .034 .75
No contradiction .659 .038 .76
No calibration .648 .085 .73
No abstention .606 .037 1.00
No technical indicators .674 .029 .71
No temporal filter†.704 .031 .74
†Invalid, future-contaminated diagnostic; not a legitimate baseline.
error-prone, but a system that rejects an entire sector could still
be operationally biased. Citation correctness is measured by
blinded entailment review of a stratified sample, with source
validity and claim support scored separately.0 0.2 0.4 0.6 0.8 100.20.40.60.81
Mean confidenceEmpirical accuracyIdeal
Raw
Hybrid
Fig. 3. Reliability diagram using simulated illustrative values.
C. Limitations and Reproducibility Risks
The framework cannot eliminate irreducible market uncer-
tainty, correlated source errors, or strategic misinformation.
LLM rationales are not causal explanations. Conformal validity
may degrade under temporal dependence, and simulated results
cannot establish economic benefit. Licensed news availability,
timestamp precision, delisting returns, corporate-action adjust-
ment, and realistic market impact can materially change a
backtest. The proposed experiment therefore requires a frozen
data manifest, executable feature pipeline, prompt registry,
model hashes, and an independent audit before any performance
claim.
VII. CONCLUSION
FinAbstain specifies a selective financial forecasting system
that joins point-in-time multimodal retrieval, independent spe-

cialist agents, evidence verification, uncertainty calibration, and
explicit abstention. Its central design principle is that a useful
financial model must know when its evidence does not justify
a directional decision. The mathematical formulation connects
agent disagreement and evidence contradiction to calibrated
uncertainty, coverage, selective risk, and cost-adjusted trading
outcomes. The presented numbers are simulated artifacts
for testing the methodology and paper layout, not empirical
results. Future work will execute the preregistered chronological
study, replace all planning counts and simulated values, test
calibration under regime shifts, measure human-review utility,
and study portfolio-level selection with liquidity and market-
impact constraints.
REFERENCES
[1]E. F. Fama, “Efficient capital markets: A review of theory and empirical
work,”The Journal of Finance, vol. 25, no. 2, pp. 383–417, 1970.
[2]P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W. tau Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive nlp tasks,” in
Advances in Neural Information Processing Systems, vol. 33, 2020, pp.
9459–9474.
[3]D. Araci, “Finbert: Financial sentiment analysis with pre-trained language
models,”arXiv preprint arXiv:1908.10063, 2019.
[4]Y . Yang, M. C. S. UY , and A. Huang, “Finbert: A pretrained language
model for financial communications,”arXiv preprint arXiv:2006.08097,
2020.
[5]S. Wu, O. Irsoy, S. Lu, V . Dabravolski, M. Dredze, S. Gehrmann,
P. Kambadur, D. Rosenberg, and G. Mann, “Bloomberggpt: A large
language model for finance,”arXiv preprint arXiv:2303.17564, 2023.
[6]M. L. de Prado,Advances in Financial Machine Learning. Wiley, 2018.
[7]D. H. Bailey, J. M. Borwein, M. L. de Prado, and Q. J. Zhu, “The
probability of backtest overfitting,”Journal of Computational Finance,
vol. 20, no. 4, pp. 39–69, 2017.
[8]Z. Chen, W. Chen, C. Smiley, S. Shah, I. Borova, D. Langdon, R. Moussa,
M. Beane, T.-H. Huang, B. Routledge, and W. Y . Wang, “Finqa: A dataset
of numerical reasoning over financial data,” inProceedings of EMNLP,
2021, pp. 3697–3711.
[9]W. Brock, J. Lakonishok, and B. LeBaron, “Simple technical trading
rules and the stochastic properties of stock returns,”The Journal of
Finance, vol. 47, no. 5, pp. 1731–1764, 1992.
[10] R. Sullivan, A. Timmermann, and H. White, “Data-snooping, technical
trading rule performance, and the bootstrap,”The Journal of Finance,
vol. 54, no. 5, pp. 1647–1691, 1999.
[11] L. Lin, L. Lin, Z. Zhang, M. Zheng, and Y . Wang, “A volume-
price-adjusted MACD trading strategy with sensitivity calibration for
U.S. equity indices,”arXiv preprint arXiv:2604.26063, 2026. [Online].
Available: https://arxiv.org/abs/2604.26063
[12] B. Shen, L. Jin, H. Cai, L. Hu, and Y . Xin, “The efficiency frontier:
A unified framework for cost-performance optimization in LLM
context management,”arXiv preprint arXiv:2605.23071, 2026. [Online].
Available: https://arxiv.org/abs/2605.23071
[13] V . Karpukhin, B. O ˘guz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen,
and W. tau Yih, “Dense passage retrieval for open-domain question
answering,” inProceedings of EMNLP, 2020, pp. 6769–6781.
[14] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever,
“Learning transferable visual models from natural language supervision,”
inProceedings of ICML, 2021, pp. 8748–8763.
[15] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and
Y . Cao, “React: Synergizing reasoning and acting in language models,”
inInternational Conference on Learning Representations, 2023.
[16] Y . Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, “Improving
factuality and reasoning in language models through multiagent debate,”
arXiv preprint arXiv:2305.14325, 2023.
[17] C. Guo, G. Pleiss, Y . Sun, and K. Q. Weinberger, “On calibration of
modern neural networks,” inProceedings of ICML, 2017, pp. 1321–1330.[18] B. Zadrozny and C. Elkan, “Transforming classifier scores into accurate
multiclass probability estimates,” inProceedings of the Eighth ACM
SIGKDD International Conference on Knowledge Discovery and Data
Mining, 2002, pp. 694–699.
[19] V . V ovk, A. Gammerman, and G. Shafer,Algorithmic Learning in a
Random World. Springer, 2005.
[20] A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal
prediction and distribution-free uncertainty quantification,”Foundations
and Trends in Machine Learning, vol. 16, no. 4, pp. 494–591, 2023.
[21] Y . Geifman and R. El-Yaniv, “Selective classification for deep neural
networks,” inAdvances in Neural Information Processing Systems, vol. 30,
2017.
[22] J. Ren, Y . Zhao, T. Vu, P. J. Liu, and B. Lakshminarayanan, “Self-
evaluation improves selective generation in large language models,”arXiv
preprint arXiv:2312.09300, 2023.
[23] U.S. Securities and Exchange Commission, “Edgar application
programming interfaces,” https://www.sec.gov/search-filings/
edgar-application-programming-interfaces, 2024, accessed: 2026-
07-24.