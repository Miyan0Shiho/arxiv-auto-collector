# Agent Memory with Episodic Retrieval for Financial Decision-Making

**Authors**: Nuoyue Xu, Jiang Liu, Wenxuan Huang, Xiang Zhang, Juntai Cao, Jiaqi Wei

**Published**: 2026-09-23 20:32:54

**PDF URL**: [https://arxiv.org/pdf/2609.28771v1](https://arxiv.org/pdf/2609.28771v1)

## Abstract
Large language models (LLMs) have demonstrated strong capabilities in financial analysis and reasoning, inspiring recent advances in agent-based trading frameworks. While these systems show promise, prior approaches either emphasize long-horizon forecasting or operate as stateless analyzers, limiting their applicability to the demands of trading in complicated settings. To address these gaps, we introduce META (Memory Enhanced Trading Agent), the first RAG-like episodic-memory-augmented multi-agent framework for financial decision making. META integrates a family of specialized indicator agents (e.g., Trend, MACD, Stochastic, RSI, SMA, AVWAP, Heikin-Ashi) with a Decision Agent that fuses their reports, and a Memory module that retrieves and updates past trading episodes encoded as market state embeddings with outcomes and reflections. By recalling relevant experiences and adaptively reweighting signals under similar market regimes, META achieves improved directional accuracy and robustness under short-horizon evaluation. Our results demonstrate that episodic memory provides a powerful mechanism for regime-aware, interpretable, and low-latency decision-making in trading and decision making. The code of this project is released on GitHub.

## Full Text


<!-- PDF content starts -->

Agent Memory with Episodic Retrieval for Financial Decision-Making
Nuoyue Xu1,*, Jiang Liu2,*, Wenxuan Huang3,*, Xiang Zhang1, Juntai Cao1, Jiaqi Wei4
1University of British Columbia,2University of Science and Technology of China,
3Fudan University,4Zhejiang University
*Equal contribution.
Abstract
Large language models (LLMs) have demon-
strated strong capabilities in financial analy-
sis and reasoning, inspiring recent advances in
agent-based trading frameworks. While these
systems show promise, prior approaches either
emphasize long-horizon forecasting or oper-
ate as stateless analyzers, limiting their appli-
cability to the demands of trading in compli-
cated settings. To address these gaps, we in-
troduce META (Memory Enhanced Trading
Agent), the first RAG-like episodic-memory-
augmented multi-agent framework for finan-
cial decision making. META integrates a
family of specialized indicator agents (e.g.,
Trend, MACD, Stochastic, RSI, SMA, A VWAP,
Heikin-Ashi) with a Decision Agent that fuses
their reports, and a Memory module that re-
trieves and updates past trading episodes en-
coded as market state embeddings with out-
comes and reflections. By recalling relevant
experiences and adaptively reweighting signals
under similar market regimes, META achieves
improved directional accuracy and robustness
under short-horizon evaluation. Our results
demonstrate thatepisodic memoryprovides
a powerful mechanism for regime-aware, in-
terpretable, and low-latency decision-making
in trading and decision making. Code of this
project is released at GitHub.
1 Introduction
Building autonomous agents capable of reliable
decision-making in complex, non-stationary envi-
ronments remains a central challenge in machine
learning. High-frequency trading (HFT) exempli-
fies this difficulty: agents must process rapidly
evolving, high-dimensional data under strict la-
tency constraints, where profitable patterns can
invert within minutes (Yang et al., 2023; Zhang
et al., 2023a). Traditional quantitative systems rely
on handcrafted logic for low-latency execution but
collapse under regime shifts. Conversely, LargeLanguage Model (LLM) agents demonstrate ad-
vanced reasoning ability (Wang et al., 2024; Li
et al., 2023b), yet remainstateless—each decision
made in isolation, without learning from prior out-
comes. This absence of experiential continuity lim-
its adaptability in markets defined by recurrence
and volatility.
We argue that progress in this domain requires
agents endowed withexperiential memory, ca-
pable of recognizing and reusing structural analo-
gies between past and present market states (Yu
et al., 2023; Ji et al., 2023). While dynamic mem-
ory frameworks such as Mem0 and MemGPT
(Chhikara et al., 2025; Packer et al., 2023; Xu
et al., 2025) enable long-term reasoning in text-
based tasks, they fail to generalize to the stochastic,
high-dimensional structure of financial time series.
Unlike language, market data evolve as continuous
numerical manifolds, demanding memory systems
that operate over both symbolic abstractions and
quantitative dynamics. This motivates a new direc-
tion: re-engineering episodic memory for real-time
environments where state representations are not
linguistic but numerical and non-stationary.
To this end, we introduceMETA(Memory
Enhanced Trading Agent), a multi-agent cogni-
tive architecture that reframes trading as a “Per-
ceive–Recall–Synthesize” process. In thePercep-
tionstage, specialized Signal Agents extract in-
terpretable technical features from market streams.
In theRecallstage, a Memory Operator performs
case-based retrieval from prior trades to identify
historically analogous contexts. Finally, theSyn-
thesisstage fuses real-time analysis with retrieved
precedents to produce context-aware trading deci-
sions. This process is powered by adual-stream
state representationthat separates symbolic indi-
cator vectors (for reasoning and retrieval) from nor-
malized price–volume embeddings (for similarity
search), balancing interpretability and numerical
precision.
arXiv:2609.28771v1  [cs.AI]  23 Sep 2026

Figure 1:Architecture of the META memory system.The framework consists of three core components: Memory
System Establishment, Relevant Memory Retrieval, and Memory Database Management.
META is the first cognitive trading framework
to embed retrieval-augmented episodic memory di-
rectly into the high-frequency decision loop. Its
memory operator extends dynamic memory prin-
ciples (Chhikara et al., 2025; Li et al., 2023a; An
et al., 2025) to structured numerical domains, en-
abling adaptive reasoning grounded in market feed-
back. Empirically, META achieves substantial
gains in profitability and stability over stateless
baselines, showing that coupling cognition with
memory yields agents that not only react to mar-
kets—butlearnfrom them.
2 Related Work
LLMs for Financial Decision-Making.Large
Language Models (LLMs) are increasingly de-
ployed as autonomous agents for financial trading,
analysis, and alpha discovery (Yang et al., 2023;
Zhang et al., 2023a; Wu et al., 2023; Gong et al.,
2025; Xiong et al., 2026). LLM-based trading
agents generally follow three paradigms: leverag-
ing sentiment from news and social media (Lopez-
Lira and Tang, 2023; Kirtac and Germano, 2024;
Zhang et al., 2024a), using reinforcement learning
(RL) to refine policies through simulated trading(Ding et al., 2023; Koa et al., 2024), or employing
explicit reasoning frameworks (Ji et al., 2023; Li
et al., 2023b; Yu et al., 2023; Zhang et al., 2024b,
2023b). A complementary line of work formalizes
such inference-time reasoning as an explicit search
process over intermediate decision states (Wei et al.,
2026b), and documents the broader transition from
single-model prompting to autonomous, agentic
systems that plan and act over extended horizons
(Wei et al., 2025). Our work, META, advances the
reasoning-driven paradigm with a specialized mem-
ory framework for technical analysis. It uses an
“Indicator Vector” (e.g., MACD, RSI) to retrieve
analogous historical scenarios via cosine similar-
ity. A key contribution is ourMemory Update and
Reflectionloop, which refines memory based on
trade outcomes, allowing the agent to learn from
the performance of technical patterns. Unlike sys-
tems that generate new alpha factors (Xiong et al.,
2025; Wang et al., 2023b; Zhang et al., 2025a),
META focuses on adaptive decision-making with
established indicators.
Memory Mechanisms in Agents.Long-term
reasoning is a central challenge for LLM agents.
Early methods that appended full interaction histo-

Figure 2: The META cognitive workflow, illustrating the interplay of its three core operators. The process unfolds
in stages: (1)Perception ( Φ): A distributed ensemble of Signal Agents encodes the market state in parallel. (2)
Synthesis ( Π): Real-time reports are fused with retrieved historical precedents to formulate a decision. (3)Memory
(Ψ): The memory buffer is updated via a reflection-augmented consolidation cycle.
ries to prompts were inefficient and scaled poorly
(Hatalis et al., 2023; Liu et al., 2024; An et al.,
2025). This spurred the development of external
memory frameworks, including layered architec-
tures, dynamic knowledge networks, and cogni-
tively inspired systems (Packer et al., 2023; Xu
et al., 2025; Lee et al., 2024; Salama et al., 2025;
Liu et al., 2022; Wang et al., 2023a; Zhao et al.,
2025). However, persistent challenges include
memory modules that are decoupled from the
agent’s reasoning policy (Li et al., 2023a; Yoon
et al., 2024; Chhikara et al., 2025) and RL-based
systems that still rely on full-history accumulation
without selective retention (Jin et al., 2025; Zheng
et al., 2025). More recent retrieval-augmented sys-
tems further show that retrieval alone is insufficient:
retrieved evidence must be critiqued and optimized
at test time before it can reliably inform a deci-
sion (Wei et al., 2026c). META addresses these
limitations in quantitative trading by tightly inte-
grating a structured, indicator-based memory with
a reflection-driven decision cycle. By using co-
sine similarity retrieval and continuously updating
its knowledge based on trade performance, META
enhances adaptivity in non-stationary market envi-
ronments.3 Method
We proposeMETA(Memory-EnhancedTrading
Agent), a cognitive framework that decomposes
the trading process into three modular operators:
Perception ( Φ), Memory ( Ψ), and Synthesis ( Π).
Unlike conventional end-to-end black-box models,
META introduces an explicit and auditable reason-
ing workflow. Its compositional design enables
functional modularity, leverages non-parametric
memory for generalization, and supports inter-
pretable, retrieval-grounded decision making.
3.1 Problem Formulation
We cast trading as the problem of learning a
retrieval-augmented policy π, which maps a market
statezt∈ Sand a time-varying episodic memory
Mtto a structured decision dt. The objective is
to learn an optimal policy π⋆that maximizes the
expected cumulative return:
J(π) =E"TX
t=0γtot(dt)#
, π⋆= arg max
πJ(π),
where otdenotes the outcome from decision dt,
andγ∈(0,1]is a temporal discount factor.
At the heart of META lies the hypothesis that
optimal trading policies arefunctionally decom-

posable. We define πas the composition of two
interpretable modules:
dt= Π 
Φ(zt)|{z}
Perception,Ψ(z t,Mt)|{z}
Memory
.
This structured decomposition confers three main
benefits: (i)specialized processingof heteroge-
neous financial inputs; (ii)retrieval-augmented
generalizationthrough episodic memory; and (iii)
transparent inference chainsthat elucidate the
agent’s decision logic (Fig. 2).
3.2 Bifurcated State Representation
Financial markets exhibit both high-level struc-
tural regimes and fine-grained temporal fluctua-
tions. To model this duality, META employs a
bifurcated state representationthat jointly captures
symbolic abstractions and numerical microstruc-
ture. From a sliding window of observations
Xt={(O, H, L, C, V) t−i}W−1
i=0 , we construct:
zt=h
z(sym)
tz(num)
ti
,
where∥denotes concatenation.
Thesymbolic component z(sym)
t∈RMen-
codes several interpretable, domain-specific fea-
tures—such as momentum, volatility, and mean-
reversion indicators—using compact heuristic
transformations. This channel supports logical and
rule-based reasoning, providing the cognitive sub-
strate for high-level pattern recognition and sym-
bolic inference.
Thenumerical component z(num)
t∈R5Wpre-
serves the normalized, high-fidelity raw market
microstructure (open, high, low, close, volume). It
enables precise retrieval through vector similarity,
capturing minute market configurations often lost
in purely symbolic abstractions.
This bifurcated encoding allows the architecture
to reason at two complementary levels: the sym-
bolic stream offers interpretability and abstraction,
while the numerical stream ensures precision and
adaptability. By leveraging both, META bridges
symbolic and subsymbolic reasoning.
3.3 Perception Operator (Φ)
The perception operator Φimplements a distributed
ensemble ofSignal Agents {Am}M
m=1, where each
agent embodies a domain heuristic or learned detec-
tor (see Appendix B for the detailed mathematicalformulation of each indicator). Each Amprocesses
the market statez tto produce a compact report:
rm
t=Am(zt).
The collective output forms a perceptual set Rt=
{rm
t}M
m=1, in which each report rm
tencapsulates
both quantitative metrics (e.g., indicator strength)
and qualitative semantics (e.g.,“bullish diver-
gence, ” “breakout detected”).
This design provides two crucial benefits. First,
it modularizes perception—each agent can be re-
fined or replaced independently, supporting contin-
uous integration of new trading signals. Second,
it yields structured, interpretable percepts, allow-
ing higher modules to perform symbolic reasoning
without operating directly on raw data. The same
principle of decomposing a complex task across
role-specialized LLM agents, rather than a single
monolithic model, has proven effective in multi-
agent generation pipelines across domains (Zhang
et al., 2025c), and recent work further learns to op-
timize the workflow over such agents at execution
time (Xu et al., 2026).
3.4 Memory Operator (Ψ)
The memory operator Ψimplementscase-based
analogical reasoning. The episodic memory M=
{mi}N
i=1stores discrete trading experiences, where
each entry mi= (z i, ai, oi, ρi)logs the state, ac-
tion, outcome, and an optional textual reflection
summarizing the context and lesson learned.
Retrieval.Given the current state zt,Ψidentifies
historically similar contexts by computing cosine
similarity over the numerical embeddings:
sim(z t, zi) =(z(num)
t )⊤z(num)
i
∥z(num)
t∥∥z(num)
i∥.
A top- kselection of analogous cases forms the
retrieved neighborhood:
Nt={(m i, si)|si= sim(z t, zi)≥τ} Top-k.
These retrieved cases constitute the agent’s
“episodic evidence,” grounding its current reason-
ing in empirical precedent.
Consolidation.To preserve computational effi-
ciency and diversity, Ψemploys a dynamic curation
operator Op(m t,M) that manages the lifecycle of
memories. Low-utility or redundant entries are
pruned, while novel, high-impact experiences are

retained. This continual abstraction transforms M
into a structured, evolving repository of archety-
pal scenarios—a non-parametric analog of synaptic
consolidation in biological systems. Crucially, re-
trieved cases are not consumed verbatim: every
episode carries the outcome and reflection of the
original trade, so retrieved evidence is re-critiqued
and re-weighted at test time instead of being naively
appended to the prompt (Wei et al., 2026c).
3.5 Synthesis Operator (Π)
The synthesis operator Πemploys a LLM to in-
tegrate perceptual summaries Rtand retrieved
episodesN tinto a structured decision output:
dt≜(a t, κt,Et, ρt),
where atdenotes the action (e.g., LONG ),κtis the
confidence score, Et⊆R t∪ N tis the subset of
selected evidence, and ρtis a concise natural lan-
guage rationale justifyinga t.
We enforce anevidence-first, constrained de-
codingprocess: the LLM must first select Etfrom
the available context, and only then generate the
rationale ρt. This separation enforces grounded
reasoning and produces an auditable explanation
chain from input evidence to trading action. How
such a reasoning chain is prompted is not a neutral
design choice: the structure of the chain-of-thought
prompt interacts with the answer space and deter-
mines whether deliberation helps or hurts (Zhang
et al., 2025b). Likewise, deciding how much rea-
soning to expose as an explicit rationale is itself a
policy to be learned, since verbose deliberation in-
curs latency while silent decisions are unauditable
(Wei et al., 2026a). META adopts a fixed, domain-
motivated disclosure policy—select evidence first,
then emit a short justification—which keeps the
additional token budget small enough to remain
within HFT latency limits.
4 Experiments and Results
4.1 Experimental Setup
To evaluate the multi-agent framework of META,
we perform experiments on a asset consisting of
five representative financial assets constructed by
prior work Xiong et al. (2025): Crude Oil Fu-
tures (CL), S&P 500 E-mini Futures (ES), Nasdaq-
100 Futures (NQ), Invesco QQQ ETF (QQQ), and
Bitcoin (BTC). Each dataset consists of 4-hour
OHLCV (Open, High, Low, Close, V olume) se-
quences sampled over multi-year horizons, cap-turing diverse market regimes and volatility pat-
terns. Following the experimental protocol of
Xiong et al. (2025), each asset is partitioned into
non-overlapping windows of 100 candlesticks, with
the final three windows reserved for validation.
For the memory configuration, META employs
the top- kand similarity-threshold retrieval strat-
egy. Unless otherwise stated, kis set to 5 and the
similarity threshold τto 0.7. This configuration
balances efficiency and contextual relevance for
low-latency operation.
As baselines, we include the QUANTAGENT
framework (Xiong et al., 2025)—a multi-agent trad-
ing system without long-term memory—and a ran-
dom decision generator that selects between LONG
and SHORTuniformly at random while sharing the
same evaluation segments and reporting format.
Evaluation.We adopt the evaluation protocol
proposed by Xiong et al. (2025) to ensure com-
parability. The primary metric isdirectional
accuracy( α), which quantifies the correctness
of LONG/SHORTforecasts relative to subsequent
price movements. In addition, we report four com-
plementary return-based metrics from Xiong et al.
(2025) that jointly assess profitability, robustness,
and memory relevance: Average cumulative re-
turn correlation ( Rcc) Similarity-weighted return
alignment ( Rsim) Maximum realized gain ( Rmax)
Maximum drawdown ( Rmin) Detailed definitions,
computation procedures, and theoretical motivation
for these metrics are provided in Appendix D.
4.2 Main Results
Table 1 presents a comprehensive comparison be-
tween META, QUANTAGENT, and a random trad-
ing policy across five assets. The results demon-
strate that META consistently outperforms both
baselines in directional accuracy and return-based
metrics, confirming the effectiveness of its memory-
augmented decision process.
In terms of accuracy ( α), META achieves the
highest performance on all assets except Crude
Oil (CL), where it performs competitively with
QUANTAGENT. Notably, on S&P 500 E-mini (ES),
accuracy rises from 55.0% in QUANTAGENTto
64.0% with META, corresponding to a relative im-
provement of +55.0%. Similar gains are observed
on Nasdaq-100 (NQ) and Bitcoin (BTC), where
accuracy improves by +50.1% and +26.4%, respec-
tively. Even for more stable equity markets such
as QQQ, META maintains a marginal but consis-

Method Accuracyα(%↑) ∆α(%↑)R cc↑R max↑R min↑R sim↑
CL
+ Random 49.0 – -0.312 0.999 -1.384 -0.255
+ QuantAgent 57.7 +17.8% -0.195 1.181-1.202-0.133
+ META (Ours) 56.0 +6.1% -0.484 1.024 -1.187 -0.433
ES
+ Random 41.3 – 0.006 0.560 -0.539 0.006
+ QuantAgent 55.0 +33.2% 0.1790.613-0.485 0.179
+ META (Ours) 64.0 +55.0% 0.278 0.584 -0.431 0.250
NQ
+ Random 41.3 – -0.033 0.717 -0.736 -0.033
+ QuantAgent 53.3 +29.1% 0.0780.747-0.7050.078
+ META (Ours) 62.0 +50.1% 0.089 0.631 -0.667 -0.029
QQQ
+ Random 39.7 – -0.265 0.966 -1.038 -0.252
+ QuantAgent 59.7 +50.4% 0.189 1.052-0.952 0.193
+ META (Ours) 61.5 +54.9% 0.627 1.094 -1.102 -0.029
BTC
+ Random 44.3 – -0.259 1.115 -1.366 -0.277
+ QuantAgent 50.7 +14.5% 0.081 1.232 -1.2490.004
+ META (Ours) 56.0 +26.4% 0.120 1.296 -0.980 -0.040
Table 1: Performance comparison across assets. Accuracy ( α), relative improvement ( ∆α), correlation Rcc,
maximum and minimum returns ( Rmax,Rmin), and similarity metric ( Rsim). Red indicates better performance,
Green indicates worse, and black numbers denote baseline or neutral results.
tent edge over QUANTAGENT(61.5% vs. 59.7%).
Beyond directional accuracy, the return-based met-
rics further corroborate the effectiveness of META.
The model consistently achieves the strongest cu-
mulative return correlations ( Rcc) on ES ( 0.278 ),
QQQ ( 0.627 ), and BTC ( 0.120 ), indicating that
its predicted trade directions align more reliably
with realized price movements compared to prior
baselines. Moreover, META demonstrates notably
improved downside resilience, achieving the least
negative drawdowns ( Rmin) across all assets except
QQQ, where it remains competitive.
4.3 Text Embeddings vs. Indicator Vectors
We compare META’s performance under two re-
trieval modalities: text embeddings derived from
agent reports versus indicator vectors composed of
technical features and normalized OHLCV data.As
shown in Figure 4, indicator-vector retrieval con-
sistently outperforms text-embedding retrieval. On
ES futures, accuracy improves from 56.0% to
64.0%, and on QQQ from 50.0% to 60.0%. Even
for more stable markets such as CL and BTC, indi-
cator retrieval yields gains of 2–6 percentage points.
Overall, indicator-based retrieval achieves 2–10%
higher accuracy than text embeddings.
Beyond accuracy, the two modalities exhibit dis-Asset Similarity (Mean) Variance (Mean)
Text Indic. Text Indic.
BTC 0.9002 0.6428 0.0002 0.0031
CL 0.9089 0.6905 0.0002 0.0028
ES 0.9036 0.6853 0.0003 0.0030
NQ 0.9139 0.6826 0.0002 0.0033
QQQ 0.9095 0.7011 0.0003 0.0026
Table 2: Comparison of mean similarity and variance
between text embeddings and indicator vectors. Indica-
tor vectors show lower similarity but higher variance,
leading to more discriminative retrieval.
tinct retrieval geometries (Zhang et al., 2024c).
Text embeddings produce uniformly high simi-
larity scores (0.90–0.91) with minimal variance
(≈0.0002 –0.0003 ), implying that semantically
similar reports cluster together regardless of mar-
ket context—limiting discriminative power. In
contrast, indicator vectors yield lower mean
similarities (0.64–0.70) but higher variance ( ≈
0.0026 –0.0033 ), indicating sharper separation be-
tween relevant and irrelevant precedents. This
sharper separation improves retrieval selectivity,
which is crucial in high-frequency trading where
noisy matches quickly degrade performance. These
findings confirm META’s design choice to ground
memory retrieval in structured indicator vectors

(a) BTC Similarity
 (b) CL Similarity
 (c) ES Similarity
 (d) NQ Similarity
 (e) QQQ Similarity
 (f) BTC Variance
(g) CL Variance
 (h) ES Variance
 (i) NQ Variance
 (j) QQQ Variance
Figure 3: Similarity (top row per asset) and variance (bottom row) distributions for text embeddings vs. indicator
vectors across all assets.
Figure 4: Accuracy comparison of text-embedding vs.
indicator-vector retrieval.
rather than textual embeddings. The advantage
of compact, structured numeric features over
dense semantic embeddings also echoes results
in few-shot tabular learning, where LLM-derived
structured representations substantially strengthen
downstream predictors (Yang et al., 2026).
4.4 Memory Retrieval Guidance
We analyze how the top- kand similarity threshold
(τ) hyperparameters control retrieval quality and
efficiency, validating META’s choice of combining
both mechanisms for stable and adaptive recall.
Retrieved counts at fixed threshold.At a fixed
threshold (e.g., τ= 0.7 ), the number of retrieved
memories per query (Figure 5, top) increases
steadily as the memory base grows. Although in-
dividual counts fluctuate, the rolling average rises
over time, showing that thresholding functions as a
quality filter—restricting retrieval early when mem-
ory is sparse, and gradually broadening it as more
relevant experiences accumulate.
Average top- ksimilarity at fixed k.With k
fixed (e.g., k= 10 ), the average similarity of re-
trieved memories (Figure 5, middle) improves as
the memory expands, stabilizing at a high level.Asset Text Acc. Indic. Acc. Text Time Indic. Time
CL 50.0 52.0 3.11 0.00610
ES 56.064.0 3.18 0.00580
NQ 60.0 62.0 3.15 0.00595
QQQ 50.061.0 3.12 0.00620
BTC 50.0 56.0 3.17 0.00575
Table 3: Accuracy (%) and retrieval latency (s) com-
parison between text-embedding and indicator-vector
memory retrieval.
This demonstrates that top- kselection maintains
focus on the most relevant precedents while pre-
venting uncontrolled growth in reference size.
Joint effect of kandτ.Heatmaps of average
retrieval counts across k–τcombinations (Figure 5,
bottom) reveal their complementary dynamics. At
lowτ, increasing krapidly enlarges the retrieved
set, while very high τcollapses retrieval to near
zero. Balanced behavior occurs around moderate
settings ( τ= 0.6 –0.7, k= 5 –15), yielding both
sufficient coverage and high reliability.
4.5 Latency-Aware Retrieval Analysis
Latency is a critical constraint in high-frequency
trading, META therefore adopts two latency-aware
design choices relative to prior LLM-based trading
agents: (i) all Signal Agents operate in parallel,
avoiding stacking agentic reasoning process; and
(ii) episodic memory is retrieved using cosine simi-
larity in indicator-vector space rather than semantic
text embeddings. Table 3 summarizes the accuracy–
latency tradeoff across five assets. Indicator-vector
retrieval improves directional accuracy by 2–11
percentage points while reducing retrieval latency
from approximately 3.1 seconds to 0.006 seconds
per query. This corresponds to an average500 ×
speedupin memory access across all assets. The
results confirm that indicator vectors enable fast,
domain-aligned retrieval without sacrificing accu-

(a) BTC
 (b) CL
 (c) ES
 (d) NQ
 (e) QQQ
Figure 5: Top- kand threshold retrieval analysis. For each asset: # memories retrieved ( τ= 0.7 ), average top- k
similarities (k= 10), and heatmap of retrieval counts overkandτ.
(a) CL
 (b) ES
 (c) NQ
 (d) QQQ
 (e) BTC
Figure 6: Accuracy versus number of memories. Each curve shows the moving average of directional accuracy as
META’s memory base expands. The analysis begins at 300 memories, corresponding to the point where episodic
retrieval starts to meaningfully influence decision-making.
racy, allowing episodic memory to be integrated
directly into the trading loop.
4.6 Learning Dynamics with Memory
Accumulation
We examine how META’s performance evolves
as its memory base grows. Figure 6 shows the
moving-average directional accuracy. The analy-
sis starts at 300 memories, since the initial 0–300
trades rely primarily on perception and reasoning
modules rather than episodic recall; early accuracy
variations thus reflect exploratory behavior rather
than learned adaptation. Beyond this phase, all as-
sets exhibit a clear upward trend in accuracy, with a
transient volatility window around 300–350 trades
as retrieval stabilizes and the memory base diversi-
fies. As memory density increases further, accuracy
steadily improves and variance declines, confirm-
ing that broader episodic coverage enhances con-
textual reasoning and decision precision. Notably,
higher-volatility assets such as BTC and NQ dis-
play sharper accuracy gains, highlighting the value
of episodic retrieval in turbulent markets.
4.7 Robustness Under Varying Market
Conditions
We evaluate META’s stability across four represen-
tative market regimes—uptrend,downtrend,steady,
Figure 7: Performance of META under different market
conditions across assets.
andsudden change—to assess its adaptability un-
der dynamic trading environments. As shown in
Figure 7, performance remains highly consistent
across regimes, with accuracy variations confined
to within 4–6% across all assets.
Across the five assets, META achieves its great-
est stability in steady markets (average accuracy
above 0.53) by exploiting recurring structural pat-
terns through memory retrieval, while comparable
accuracy inuptrendanddowntrendphases indi-
cates effective generalization without directional
bias. The framework is also resilient during sudden
price shocks (e.g., 0.60 accuracy on QQQ), as its
retrieval-guided decisions draw on structurally sim-
ilar high-volatility episodes rather than short-term
fluctuations.

Asset Method Accuracyα(%↑)∆α(↑)
CL w/o Mem 52.5 –
w/ Mem 56.0 +3.5
ES w/o Mem 50.0 –
w/ Mem 64.0 +14.0
NQ w/o Mem 40.0 –
w/ Mem 62.0 +22.0
QQQ w/o Mem 55.0 –
w/ Mem 61.5 +6.5
BTC w/o Mem 48.0 –
w/ Mem 56.0 +8.0
Table 4: Ablation study of memory system.
4.8 Memory System Ablation
We perform an ablation study to isolate the impact
of the memory module by comparing META’s
performance with and without memory across five
assets (Table 4). Across all settings, incorporating
memory yields consistent and substantial gains in
directional accuracy.
For high-volatility assets such as ES and NQ, the
improvements are especially pronounced: accuracy
increases from 50.0% to 64.0% (+14.0 points) on
ES and from 40.0% to 62.0% (+22.0 points) on
NQ. These results highlight the value of episodic
retrieval in distinguishing subtle recurring patterns
that short-term indicators alone fail to capture.
Steadier assets also benefit, with CL improving
from 52.5% to 56.0% (+3.5 points) and QQQ from
55.0% to 60.0% (+5.0 points). Even BTC, despite
its noisy structure, shows an +8.0 point gain.
Overall, these consistent gains across volatile
and stable markets confirm that the memory mecha-
nism enhances predictive robustness and contextual
reasoning.
4.9 Indicator Contributions
To analyze the internal dynamics of the META
framework, we perform an ablation study assessing
the contribution of each technical indicator to over-
all trading performance. Each indicator—MACD,
A VWAP,Trend,RSI,Heiken Ashi,Stochastic,
andSMA—is removed individually, and the result-
ing change in directional accuracy quantifies its
marginal impact on decision quality.
As shown in Figure 8, removing any single indi-
cator reduces accuracy, confirming that all compo-
nents contribute meaningfully to META’s percep-
tion of market dynamics.TrendandMACDyield
the largest declines ( −2.42% and−2.02% ), reflect-
ing their central role in capturing mid- to long-
Figure 8: Impact of removing each technical indicator
on META’s directional accuracy. Larger decreases in-
dicate greater importance of that indicator to overall
system stability and predictive accuracy.
term momentum;Heiken AshiandStochasticpro-
duce moderate drops ( −1.82% and−1.74% ); and
A VWAPandSMAshow smaller effects ( −1.40%
and−1.23% ), suggesting partial redundancy with
other trend-based indicators.
5 Conclusion
We present META, a memory-augmented trad-
ing framework that integrates multi-agent market
perception with reflective reasoning. By combin-
ing specialized Indicator Agents, a decision fu-
sion mechanism, and a dynamic Memory Operator,
META enables adaptive and interpretable trading
decisions in a zero-shot setting. Extensive experi-
ments across diverse assets show that META con-
sistently outperforms baselines, validating the ef-
fectiveness of memory-driven contextual recall for
high-frequency decision making.
Limitations
While META demonstrates notable improvements
in short-horizon trading accuracy and adaptability,
several limitations remain. The current framework
relies on handcrafted technical indicators for per-
ceptual grounding, which may restrict scalability
to more diverse market structures or alternative
data modalities. Although the episodic memory
improves contextual reasoning, its similarity-based
retrieval remains static; future work could explore
adaptive or learned retrieval metrics that evolve
with market regimes. In addition, our evaluation
focuses on retrospective simulation under histor-
ical data rather than live market execution, and
thus omits slippage, transaction costs, and real-time
latency constraints. Addressing these limitations
would move META closer to a fully autonomous,
production-ready cognitive trading agent.

References
C. An, J. Zhang, M. Zhong, L. Li, S. Gong, Y . Luo,
J. Xu, and L. Kong. 2025. Why does the effective
context length of LLMs fall short? InProceedings
of the International Conference on Learning Repre-
sentations (ICLR).
P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav.
2025. Mem0: Building production-ready AI agents
with scalable long-term memory.arXiv preprint
arXiv:2504.19413.
Y . Ding, S. Jia, T. Ma, B. Mao, X. Zhou, L. Li, and
D. Han. 2023. Integrating stock features and global
information via large language models for enhanced
stock return prediction.
Mingyang Gong, Jing Fan, Guohui Lin, Bing Su, Zi-
han Su, and Xiang Zhang. 2025. Multiprocessor
scheduling with testing: improved online algorithms
and numerical experiments.Journal of Scheduling,
28(5):513–527.
K. Hatalis, D. Christou, J. Myers, S. Jones, K. Lambert,
A. Amos-Binks, Z. Dannenhauer, and D. Dannen-
hauer. 2023. Memory matters: The need to improve
long-term memory in LLM-agents. InProceedings
of the AAAI Symposium Series, volume 2, pages 277–
280.
Z. Ji, T. Yu, Y . Xu, N. Lee, E. Ishii, and P. Fung. 2023.
Towards mitigating hallucination in large language
models via self-reflection.
B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D. Wang,
H. Zamani, and J. Han. 2025. Search-R1: Train-
ing LLMs to reason and leverage search en-
gines with reinforcement learning.arXiv preprint
arXiv:2503.09516.
K. Kirtac and G. Germano. 2024. Sentiment trading
with large language models.Finance Research Let-
ters, 62:105227.
K. J. Koa, Y . Ma, R. Ng, and T.-S. Chua. 2024. Learning
to generate explainable stock predictions using self-
reflective large language models.
K.-H. Lee, X. Chen, H. Furuta, J. Canny, and I. Fischer.
2024. A human-inspired reading agent with gist
memory of very long contexts.
Y . Li, B. Dong, F. Guerin, and C. Lin. 2023a. Compress-
ing context to enhance inference efficiency of large
language models. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP), pages 6921–6935.
Y . Li, Y . Yu, H. Li, Z. Chen, and K. Khashanah. 2023b.
TradingGPT: Multi-agent system with layered mem-
ory and distinct characters for enhanced financial
trading performance.
N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilac-
qua, F. Petroni, and P. Liang. 2024. Lost in the
middle: How language models use long contexts.Transactions of the Association for Computational
Linguistics, 12:157–173.
Puyuan Liu, Xiang Zhang, and Lili Mou. 2022. A
character-level length-control algorithm for non-
autoregressive sentence summarization.Advances in
Neural Information Processing Systems, 35:29101–
29112.
A. Lopez-Lira and Y . Tang. 2023. Can ChatGPT fore-
cast stock price movements? return predictability and
large language models.
C. Packer, V . Fang, S. Patil, K. Lin, S. Wooders, and
J. Gonzalez. 2023. MemGPT: Towards LLMs as
operating systems.
R. Salama, J. Cai, M. Yuan, A. Currey, M. Sunkara,
Y . Zhang, and Y . Benajiba. 2025. MemInsight: Au-
tonomous memory augmentation for LLM agents.
arXiv preprint arXiv:2503.21760.
B. Wang, X. Liang, J. Yang, H. Huang, S. Wu, P. Wu,
L. Lu, Z. Ma, and Z. Li. 2023a. Enhancing large
language models with self-controlled memory frame-
work.arXiv preprint arXiv:2304.13343.
S. Wang, H. Yuan, L. M. Ni, and J. Guo. 2024.
QuantAgent: Seeking holy grail in trading by self-
improving large language model.arXiv preprint
arXiv:2402.03755.
S. Wang, H. Yuan, L. Zhou, L. M. Ni, H.-Y . Shum,
and J. Guo. 2023b. AlphaGPT: Human–AI interac-
tive alpha mining for quantitative investment.arXiv
preprint arXiv:2308.00016.
Jiaqi Wei, Xuehang Guo, Pengfei Yu, Xiang Zhang,
Wanli Ouyang, Siqi Sun, Qingyun Wang, and Chenyu
You. 2026a. When to think, when to speak: Learning
disclosure policies for llm reasoning.arXiv preprint
arXiv:2605.03314.
Jiaqi Wei, Yuejin Yang, Xiang Zhang, Yuhan Chen,
Xiang Zhuang, Zhangyang Gao, Dongzhan Zhou,
Guangshuai Wang, Zhiqiang Gao, Juntai Cao, et al.
2025. From ai for science to agentic science: A
survey on autonomous scientific discovery.arXiv
preprint arXiv:2508.14111.
Jiaqi Wei, Xiang Zhang, Yuejin Yang, Wenxuan Huang,
Juntai Cao, Sheng Xu, Xiang Zhuang, Zhangyang
Gao, Muhammad Abdul-Mageed, Laks VS Laksh-
manan, et al. 2026b. When llm meets tree search:
A systematic view of inference as search in large
language models.arXiv preprint arXiv:2608.30395.
Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie
Qiu, Noah Wei, Jinzhe Li, Wanli Ouyang, and Siqi
Sun. 2026c. Retrieval is not enough: Enhancing
rag through test-time critique and optimization.Ad-
vances in Neural Information Processing Systems,
38:21484–21520.

S. Wu, O. Irsoy, S. Lu, V . Dabravolski, M. Dredze,
S. Gehrmann, P. Kambadur, D. Rosenberg, and
G. Mann. 2023. BloombergGPT: A large language
model for finance.
F. Xiong, X. Zhang, A. Feng, S. Sun, and C. You.
2025. QuantAgent: Price-driven multi-agent
LLMs for high-frequency trading.arXiv preprint
arXiv:2509.06784.
Fei Xiong, Xiang Zhang, Aosong Feng, Siqi Sun,
and Chenyu You. 2026. Quantharness: Price-
driven multi-agent llms for high-frequency trading.
Preprint, arXiv:2509.09995.
C. Xu, K. Ke, Z. Liu, J. Wei, Z. Shao, W. Guo, and
C. Yu. 2026. EvoMAS: Learning execution-time
workflows for multi-agent systems.arXiv preprint
arXiv:2605.08769.
W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y . Zhang.
2025. A-Mem: Agentic memory for LLM agents.
arXiv preprint arXiv:2502.12110.
H. Yang, X.-Y . Liu, and C. D. Wang. 2023. FinGPT:
Open-source financial large language models.
Zhihan Yang, Jiaqi Wei, Xiang Zhang, Haoyu Dong, Yi-
wen Wang, Xiaoke Guo, Pengkun Zhang, Yiwei Xu,
and Chenyu You. 2026. Forestllm: Large language
models make random forest great on few-shot tabular
learning.arXiv preprint arXiv:2601.11311.
C. Yoon, T. Lee, H. Hwang, M. Jeong, and J. Kang.
2024. CompAct: Compressing retrieved documents
actively for question answering. InProceedings of
the 2024 Conference on Empirical Methods in Nat-
ural Language Processing (EMNLP), pages 21424–
21439.
Y . Yu, H. Li, Z. Chen, Y . Jiang, Y . Li, D. Zhang, R. Liu,
J. W. Suchow, and K. Khashanah. 2023. FinMem:
A performance-enhanced LLM trading agent with
layered memory and character design.
B. Zhang, H. Yang, and X.-Y . Liu. 2023a. Instruct-
FinGPT: Financial sentiment analysis by instruction
tuning of general-purpose large language models.
H. Zhang, F. Hua, C. Xu, H. Kong, R. Zuo, and J. Guo.
2024a. Unveiling the potential of sentiment: Can
large language models predict chinese stock price
movements?
W. Zhang, L. Zhao, H. Xia, S. Sun, J. Sun, M. Qin,
X. Li, Y . Zhao, Y . Zhao, X. Cai, L. Zheng, X. Wang,
and B. An. 2024b. A multimodal foundation agent
for financial trading: Tool-augmented, diversified,
and generalist.
Xiang Zhang, Juntai Cao, Jiaqi Wei, Yiwei Xu, and
Chenyu You. 2025a. Tokenization constraints in llms:
A study of symbolic and arithmetic reasoning limits.
arXiv preprint arXiv:2505.14178.Xiang Zhang, Juntai Cao, Chenyu You, and Dujian Ding.
2025b. Why prompt design matters and works: A
complexity analysis of prompt search space in LLMs.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 32525–32555, Vienna, Austria.
Association for Computational Linguistics.
Xiang Zhang, Senyu Li, Bradley Hauer, Ning Shi, and
Grzegorz Kondrak. 2023b. Don’t trust chatgpt when
your question is not in english: a study of multi-
lingual abilities and types of llms.arXiv preprint
arXiv:2305.16339.
Xiang Zhang, Senyu Li, Ning Shi, Bradley Hauer, Zijun
Wu, Grzegorz Kondrak, Muhammad Abdul-Mageed,
and Laks VS Lakshmanan. 2024c. Cross-modal con-
sistency in multimodal large language models.arXiv
preprint arXiv:2411.09273.
Z. Zhang, X. Zhang, J. Wei, Y . Xu, and C. You. 2025c.
Postergen: Aesthetic-aware paper-to-poster genera-
tion via multi-agent LLMs.arXiv e-prints, pages
arXiv–2508.
Haokun Zhao, Xiang Zhang, Jiaqi Wei, Yiwei Xu, Yut-
ing He, Siqi Sun, and Chenyu You. 2025. Timeseries-
scientist: A general-purpose ai agent for time series
analysis.arXiv preprint arXiv:2510.01538.
Y . Zheng, D. Fu, X. Hu, X. Cai, L. Ye, P. Lu, and P. Liu.
2025. DeepResearcher: Scaling deep research via
reinforcement learning in real-world environments.
arXiv preprint arXiv:2504.03160.

A Prompts
A.1 MACD Agent
To instantiate theMACDAgent, we construct a
prompt that guides the agent to identify short-term
momentum shifts using theMoving Average Con-
vergence Divergence (MACD)indicator. Oper-
ating under high-frequency trading (HFT) con-
straints, the agent follows a two-step reasoning
process: first generating the MACD chart via tool
invocation, then analyzing key momentum patterns
such as crossovers, histogram variation, and diver-
gence before issuing a trading signal.
Prompt
You are a MACD (Moving Average Conver-
gence Divergence) analysis assistant operating
in a high-frequency trading context.
You must first call the tool
generate_macd_image using the pro-
videdkline_data.
Once the MACD chart is generated, analyze
the image for the following key signals:
1.Signal Line Crossovers: MACD line
crossing above/below the signal line
2.Zero Line Crossovers: MACD line cross-
ing above/below the zero line
3.Histogram Analysis: Changes in his-
togram direction and magnitude
4.Divergences: Price vs. MACD diver-
gences (bullish/bearish)
5.Momentum Strength: Distance from
zero line and slope steepness
Only after generating and analyzing the MACD
image should you make a prediction about mo-
mentum direction and trading signals.
Do not make any predictions before generating
and analyzing the image.
Prompt for MACDAgent in our multi-agent
LLM framework.The agent receives OHLC data
as input and first invokes generate_macd_image
to obtain a visual depiction of market momentum.
This prompt enforces a tool-first reasoning loop,
ensuring predictions are grounded on the generated
chart rather than raw prices.Prompt
This is a —time_frame— MACD chart gener-
ated from recent OHLC market data. The chart
includes:
MACD Line (blue)– difference between 12-
period and 26-period EMAs;
Signal Line (red)– 9-period EMA of MACD
line;
Histogram (bars)– difference between
MACD and Signal lines;
Zero Line– reference for bullish/bearish mo-
mentum.
Analyze the chart with respect to the following
aspects:
1.Signal Line Crossovers:Identify
whether the MACD line crosses above
(bullish) or below (bearish) the signal line
and its relation to the zero line.
2.Zero Line Crossovers:Determine over-
all momentum and recent transitions
across the zero baseline.
3.Histogram Dynamics:Evaluate bar ex-
pansion or contraction to gauge momen-
tum strength.
4.Divergence Patterns:Compare price ac-
tion with MACD behavior for bullish or
bearish divergences.
5.Momentum Strength:Assess slope di-
rection and distance from zero to estimate
trend intensity.
Provide:
•Primary Signal:Buy, Sell, or Hold
•Momentum Direction:Bullish, Bearish,
or Neutral
•Confidence Level:High, Medium, or
Low
•Supporting Evidence:Observed MACD
behavior
•Risk Considerations:Potential false or
conflicting signals
Support your prediction with reasoning and
observed evidence.

Graph-based prompt for MACDAgent in our
multi-agent LLM framework.Given a generated
MACD chart, the agent examines crossovers, his-
togram movements, and divergence to infer direc-
tional bias and confidence. This visual grounding
enables consistent momentum interpretation across
varying market conditions.
A.2 A VWAP Agent
To instantiate theA VWAPAgent, we construct a
prompt that guides the agent to analyze price be-
havior relative to theAnchored Volume Weighted
Average Price (A VWAP). Operating under high-
frequency trading (HFT) constraints, the agent com-
bines tool-based chart generation with structured
reasoning to detect support/resistance reactions,
mean-reversion signals, and breakout dynamics
around volume-weighted fair value.
Prompt
You are an Anchored VWAP (V olume
Weighted Average Price) analysis assistant
operating in a high-frequency trading context.
You must first call the tool
generate_anchored_vwap_image using
the providedkline_data.
Once the Anchored VWAP chart is generated,
analyze the image for the following key
patterns:
1.Support/Resistance Levels: How price
reacts when touching the A VWAP line.
2.Mean Reversion Signals: Extreme devia-
tions from A VWAP suggesting pullbacks.
3.Breakout Patterns: Decisive moves
above/below A VWAP with volume confir-
mation.
4.A VWAP Slope Analysis: Direction and
steepness indicating market bias.
5.Volume Confirmation: V olume spikes
during A VWAP interactions.
6.Distance Analysis: Magnitude of price
deviation from fair value anchor.
Only after generating and analyzing the An-
chored VWAP image should you make predic-
tions about price direction and trading oppor-
tunities.
Do not make any predictions before generating
and analyzing the image.Prompt for A VWAPAgent in our multi-agent
LLM framework.The agent first generates an
Anchored VWAP chart from OHLC data, enforcing
a visual-first reasoning flow. This ensures that the
model grounds its interpretation on price–volume
structure rather than raw numerical sequences.
Prompt
This is a —time_frame— Anchored VWAP
chart generated from recent OHLC market data.
The chart shows the volume-weighted average
price anchored to a significant market event or
time point.
Analyze the chart with respect to the following
aspects:
1.Support/Resistance Analysis:Deter-
mine whether A VWAP acts as support
(price above) or resistance (price below)
and assess reactions upon contact.
2.Mean Reversion Opportunities:Evalu-
ate deviations from A VWAP and identify
potential pullbacks toward fair value.
3.Breakout/Breakdown Signals:Detect
decisive moves above or below A VWAP,
especially with volume confirmation.
4.A VWAP Slope and Bias:Interpret the
slope’s direction (upward = bullish, down-
ward = bearish) and steepness to infer bias
strength.
5.Fair Value and Volume Context:As-
sess whether price trades at a premium
or discount to A VWAP and how volume
supports current price action.
Provide:
•Primary Trade Signal:Buy, Sell, or
Hold with rationale
•A VWAP Bias:Bullish, Bearish, or Neu-
tral
•Fair Value Assessment:Premium, Dis-
count, or Fair
•Entry Strategy:Suggested positioning
based on current setup
•Risk Management:Stop-loss and profit
target zones

•Volume Confirmation:How volume
aligns with price behavior
Support your prediction with reasoning and
observed signals.
Graph-based prompt for A VWAPAgent in our
multi-agent LLM framework.The agent inter-
prets A VWAP as a dynamic reference of institu-
tional fair value, analyzing slope, distance, and vol-
ume interplay to infer bias and entry timing. This
visual grounding allows robust short-term reason-
ing around market equilibrium and mean-reversion
behavior.
A.3 RSI Agent
To instantiate theRSIAgent, we construct a prompt
that enables the agent to evaluate momentum
strength and reversal potential using theRelative
Strength Index (RSI)indicator. Operating un-
der high-frequency trading (HFT) constraints, the
agent first generates an RSI chart via tool invo-
cation, then interprets overbought/oversold condi-
tions, divergences, and swing structures to infer
directional bias.
Prompt
You are an RSI (Relative Strength Index)
analysis assistant operating in a high-frequency
trading context.
You must first call the tool
generate_rsi_image using the provided
kline_data.
Once the RSI chart is generated, analyze the
image for the following key signals:
1.Overbought/Oversold Levels: RSI
crossing above 70 or below 30.
2.Centerline Crossovers: RSI crossing
above/below 50.
3.Failure Swings: RSI breaking prior
swing points after overbought/oversold
moves.
4.Divergences: Price–RSI mismatches
(bullish/bearish).
5.Hidden Divergences: Trend-
continuation signals during pullbacks.
6.RSI Trendlines: RSI breakout patterns
anticipating price breakouts.7.Swing Rejections: RSI rejection patterns
near extreme zones.
Only after generating and analyzing the RSI
image should you make a prediction about mo-
mentum direction and trading signals.
Do not make any predictions before generating
and analyzing the image.
Prompt for RSIAgent in our multi-agent
LLM framework.The agent first produces an
RSI chart from OHLC input, enforcing a tool-first
reasoning step. This ensures that its predictions are
visually grounded in oscillator behavior rather than
inferred directly from raw data.
Prompt
This is a —time_frame— RSI chart gener-
ated from recent OHLC market data. The plot
shows:
RSI Line(0–100 scale),Overbought Level
(70),Oversold Level(30), andCenter Line
(50).
Analyze the chart with respect to the following
aspects:
1.Overbought/Oversold Conditions:
Identify sustained or recent entries into
extreme zones and crossovers signaling
reversals.
2.Centerline Crossovers:Determine
whether RSI is above or below 50 and
assess momentum bias.
3.Failure Swings:Detect RSI reversals
breaking prior highs/lows after over-
bought or oversold readings.
4.Divergences:Compare price and RSI
trajectories for bullish or bearish diver-
gences, including hidden ones.
5.Trendlines and Patterns:Check for RSI
trendline or pattern breakouts that may
precede price shifts.
6.Swing Rejections:Evaluate bullish or
bearish rejection sequences within over-
bought/oversold regions.
Provide:
•Primary Signal:Buy, Sell, or Hold

•Momentum Direction:Bullish, Bearish,
or Neutral
•Confidence Level:High, Medium, or
Low
•Supporting Evidence:Observed RSI be-
havior
•Risk Considerations:Potential false or
conflicting signals
Support your prediction with reasoning and
observed patterns.
Graph-based prompt for RSIAgent in our
multi-agent LLM framework.Given the gen-
erated RSI chart, the agent assesses momentum
strength, reversals, and divergences to infer direc-
tional bias. This visually grounded reasoning im-
proves short-term momentum interpretation and
reversal detection consistency across different mar-
ket conditions.
A.4 Heiken Ashi Agent
To instantiate theHeikenAshiAgent, we construct
a prompt that guides the agent to identify di-
rectional trends and momentum transitions using
Heiken Ashi (HA)candlestick representations.
Operating under high-frequency trading (HFT) con-
ditions, the agent first generates a Heiken Ashi
chart and then interprets trend strength, reversals,
and momentum consistency to support short-term
directional reasoning.
Prompt
You are a Heiken Ashi candlestick analysis
assistant operating in a high-frequency trading
context.
You must first call the tool
generate_heiken_ashi_image using
the providedkline_data.
Once the Heiken Ashi chart is generated,
analyze the image for the following key
patterns:
1.Trend Direction: Consecutive green
(bullish) or red (bearish) candles.
2.Trend Strength: Candle body size, wick
formation, and consistency.
3.Reversal Signals: Color changes, dojiformations, or wicks indicating indeci-
sion.
4.Momentum Analysis: Body-size accel-
eration or deceleration.
5.Consolidation Patterns: Mixed colors
and small bodies showing indecision.
6.Entry/Exit Timing: Ideal trade points
derived from HA formations.
Only after generating and analyzing the Heiken
Ashi image should you make predictions about
trend direction and trading opportunities.
Do not make any predictions before generating
and analyzing the image.
Prompt for HeikenAshiAgent in our
multi-agent LLM framework.The
agent receives OHLC input and invokes
generate_heiken_ashi_image to visualize
smoothed candlestick dynamics. This tool-first ap-
proach ensures the reasoning process is grounded
in visually observable trend patterns rather than
raw market data.
Prompt
This is a —time_frame— Heiken Ashi chart
generated from recent OHLC data. The chart
smooths price action to reduce noise and high-
light trend direction.
Green candlesrepresent bullish momentum,
red candlesrepresent bearish momentum, and
wicks indicate indecision or volatility.
Analyze the chart according to the following
aspects:
1.Trend Direction & Consistency:Deter-
mine whether consecutive candles form
a clear bullish or bearish sequence and
detect color transitions signaling possible
reversals.
2.Trend Strength:Assess candle body size,
wick presence, and overall uniformity to
gauge momentum.
3.Reversal Indications:Identify doji-like
or spinning-top formations, color shifts
after long trends, or weakening candle
bodies.
4.Momentum Dynamics:Observe

whether candle bodies expand (accelera-
tion) or contract (deceleration).
5.Consolidation vs Trending:Detect
mixed-color candles and narrow ranges
indicating indecision.
6.Entry/Exit Signals:Evaluate pullback
entries or exhaustion exits based on
Heiken Ashi structure.
Provide:
•Trend Status:Strong Bullish, Weak
Bullish, Strong Bearish, Weak Bearish,
or Sideways
•Momentum Phase:Accelerating, Steady,
Decelerating, or Reversing
•Signal Quality:High, Medium, or Low
•Trading Action:Buy, Sell, Hold, or Wait
•Risk Assessment:Probability of false or
conflicting signals
Support your prediction with reasoning and
evidence observed from the chart.
Graph-based prompt for HeikenAshiAgent in
our multi-agent LLM framework.Given the gen-
erated Heiken Ashi chart, the agent identifies domi-
nant color sequences, momentum acceleration, and
potential reversal setups. This visual grounding
enhances consistency in trend identification and
trade-timing inference across high-frequency mar-
ket conditions.
A.5 SMA Agent
To instantiate theSMAAgent, we construct a
prompt that enables the agent to identify trend di-
rection and momentum shifts using theSimple
Moving Average (SMA)indicator. Operating un-
der high-frequency trading (HFT) constraints, the
agent first generates an SMA chart via tool invo-
cation, then interprets crossovers, slope dynamics,
and price alignment relative to multiple SMAs.
Prompt
You are a Simple Moving Average (SMA)
analysis assistant operating in a high-frequency
trading context.
You must first call the toolgenerate_sma_image using the provided
kline_data.
Once the SMA chart is generated, analyze the
image for the following key signals:
1.Trend Identification: Price position rela-
tive to key SMAs (above/below).
2.Moving Average Crossovers: Faster
SMA crossing above/below slower SMA.
3.Dynamic Support/Resistance: Price in-
teracting with SMA levels.
4.Multiple MA Ribbon: Alignment and
spacing between multiple SMAs.
5.Mean Reversion: Distance of price from
SMA suggesting potential reversal.
6.SMA Slope: Direction and steepness in-
dicating trend strength.
Only after generating and analyzing the SMA
image should you make predictions about trend
direction and trading signals.
Do not make any predictions before generating
and analyzing the image.
Prompt for SMAAgent in our multi-agent
LLM framework.The agent receives OHLC data
and generates an SMA chart as a visual reference
for trend detection. This approach ensures that
reasoning about market direction is grounded in
moving-average structure rather than isolated price
movements.
Prompt
This is a —time_frame— SMA chart generated
from recent OHLC market data. The chart in-
cludes multiple Simple Moving Averages (e.g.,
20-, 50-, and 200-period), each capturing a
different time horizon of trend behavior.
Analyze the chart with respect to the following
aspects:
1.Trend Identification:Assess whether
price is above or below key SMAs and
infer the overall directional bias.
2.Crossovers:Identify golden or death
crosses between short and long SMAs as
trend reversal signals.
3.Support/Resistance:Evaluate SMA lev-

els where price consistently bounces or
reverses.
4.MA Ribbon Alignment:Determine if
SMAs are neatly ordered (strong trend) or
entangled (ranging phase).
5.Mean Reversion:Identify overextended
price movements far from SMAs suggest-
ing pullbacks.
6.Slope and Momentum:Examine the
slope of each SMA to assess momentum
direction and strength.
Provide:
•Primary Trend Direction:Bullish, Bear-
ish, or Range-bound
•Trend Strength:Strong, Moderate, or
Weak
•Trading Bias:Buy on Dips, Sell on Ral-
lies, or Stay Flat
•Key SMA Levels:Notable support/resis-
tance zones
•Confirmation Needed:Other indicators
that would validate the trend
Support your prediction with structured reason-
ing and observed SMA configurations.
Graph-based prompt for SMAAgent in our
multi-agent LLM framework.Given the gen-
erated SMA chart, the agent interprets crossover
events, slope direction, and ribbon spacing to esti-
mate trend intensity and bias. This visual reasoning
approach improves short-term adaptability while
maintaining consistency in trend assessment.
A.6 Stochastic Agent
To instantiate theStochasticAgent, we construct
a prompt that enables the agent to analyze mo-
mentum reversals and exhaustion levels using the
Stochastic Oscillator. Operating under high-
frequency trading (HFT) conditions, the agent
first generates a Stochastic chart, then interprets
crossovers, divergences, and overbought/oversold
patterns to infer short-term momentum bias.Prompt
You are a Stochastic Oscillator analysis
assistant operating in a high-frequency trading
context.
You must first call the tool
generate_stochastic_image using the
providedkline_data.
Once the Stochastic chart is generated, analyze
the image for the following key signals:
1.Overbought/Oversold Conditions: Lev-
els above 80 (overbought) or below 20
(oversold).
2.Line Crossovers:
3.Divergences: Price–Stochastic mis-
matches (bullish/bearish).
4.Failure Swings: Internal divergence pat-
terns signaling reversals.
5.Centerline Crossovers: Crossings
above/below the 50 level.
6.Slope Analysis: Direction and steepness
of
Only after generating and analyzing the
Stochastic image should you make predictions
about momentum direction and trading signals.
Do not make any predictions before generating
and analyzing the image.
Prompt for StochasticAgent in our multi-
agent LLM framework.The agent begins by in-
voking a visualization tool to produce the Stochas-
tic chart, ensuring that subsequent reasoning about
reversals and momentum bias is grounded in oscil-
lator behavior rather than abstract data interpreta-
tion.
Prompt
This is a —time_frame— Stochastic Oscilla-
tor chart generated from recent OHLC data.
It plots the%K line(fast) and the%D line
(slow) along with overbought (80), oversold
(20), and center (50) reference levels.
Analyze the chart according to the following
aspects:
1.Overbought/Oversold Conditions:
Identify whether the oscillator is above
80 or below 20 and if it is exiting these

zones.
2.Line Crossovers:Determine where the
%K crosses the %D, especially within ex-
treme regions.
3.Divergences:Compare price and
Stochastic behavior to detect bullish or
bearish divergences.
4.Failure Swings:Identify internal
Stochastic divergences signaling momen-
tum exhaustion.
5.Centerline Position:Assess whether the
oscillator is above or below 50 to deter-
mine directional bias.
6.Slope and Momentum:Evaluate the
steepness and direction of %K and %D
for acceleration or deceleration.
Provide:
•Primary Signal:Buy, Sell, or Hold
•Momentum Direction:Bullish, Bearish,
or Neutral
•Confidence Level:High, Medium, or
Low
•Supporting Evidence:Key Stochastic
patterns observed
•Risk Considerations:Potential false or
conflicting signals
Support your prediction with structured reason-
ing and explicit pattern-based justification.
Graph-based prompt for StochasticAgent in
our multi-agent LLM framework.Once the
Stochastic chart is generated, the agent evaluates
oscillator dynamics to detect reversals, divergences,
and short-term exhaustion points. This process al-
lows visually anchored reasoning for momentum-
based entry and exit signals in fast-moving markets.
A.7 Trend Agent
To instantiate theTrendAgent, we construct a
prompt that directs the agent to identify short-term
directional bias based onsupport and resistance
trendlinesderived from recent K-line data. Operat-
ing under high-frequency trading (HFT) conditions,
the agent first generates a trendline chart and theninterprets slope, compression, and breakout behav-
ior to infer near-term market direction.
Prompt
You are a K-line trend pattern recognition
assistant operating in a high-frequency trading
context.
You must first call the tool
generate_trend_image using the pro-
videdkline_data.
Once the chart is generated, analyze the image
for support/resistance trendlines and notable
candlestick formations.
Only after generating and analyzing the
image should you make a prediction about
the short-term trend (upward, downward, or
sideways).
Do not make any predictions before generating
and analyzing the image.
Prompt for TrendAgent in our multi-agent
LLM framework.The agent receives recent
OHLC data and first visualizes it with automated
trendlines. This ensures that all reasoning about di-
rectional bias and breakout probability is grounded
in geometric structure and price interaction rather
than raw numerical features.
Prompt
This is a —time_frame— K-line chart with au-
tomatically generated trendlines: theblue line
represents support, and thered linerepresents
resistance, both derived from recent closing
prices.
Analyze the chart based on the following as-
pects:
1.Trendline Interaction:Observe whether
price is bouncing off, breaking through,
or consolidating between trendlines.
2.Slope Direction:Identify whether the
dominant slope is upward (bullish), down-
ward (bearish), or flat (neutral).
3.Compression/Expansion:Detect nar-
rowing or widening distances between
trendlines as signs of volatility contrac-
tion or breakout setup.
4.Candlestick Behavior:Assess wick di-
rection, candle clustering, and breakout
attempts along support/resistance zones.

5.Breakout Probability:Estimate the like-
lihood of a trendline breach based on
slope angle and recent price clustering.
Provide:
•Predicted Trend:Upward, Downward,
or Sideways
•Trend Strength:Strong, Moderate, or
Weak
•Key Evidence:Trendline behavior and
candlestick interactions
•Risk Considerations:False breakout po-
tential or low-volatility traps
Support your prediction with structured reason-
ing and clear trendline-based evidence.
Graph-based prompt for TrendAgent in our
multi-agent LLM framework.Once the chart
is generated, the agent analyzes trendline slope,
spacing, and price compression to infer directional
momentum. This visual grounding enables consis-
tent short-term trend interpretation across volatile
high-frequency market states.
A.8 Decision Agent
To instantiate theDecisionAgent, we construct two
prompts around the final action step: a pre-decision
synthesis that fuses all agent reports with retrieved
memories, and a post-decision reflection that audits
the outcome for learning and calibration. This pre-
serves the tool-first, evidence-grounded style used
across META while adapting it to a decision-and-
reflect workflow.
Prompt
You are an elite HFT quantitative analyst mak-
ing real-time trade decisions on{stock name}
using the{time frame}timeframe.
MISSION:Issue an immediate execution or-
der(LONG or SHORT only; HOLD is pro-
hibited)based on comprehensive analysis.
FORECAST HORIZON:Predict market di-
rection for the next 1–2 candlesticks ({time
frame} period).
HISTORICAL CONTEXT — Similar Mar-
ket Conditions:If similar memories are pro-
vided, summarize key lessons and outcomes;
otherwise state that no highly relevant memo-ries were found.
CURRENT MARKET ANALYSIS (pro-
vided in context):Indicator, Pattern, Trend,
MACD, Anchored VWAP, Heiken Ashi, SMA,
Stochastic, and RSI reports.
DECISION FRAMEWORK
•Primary Signals (High Weight):Mo-
mentum confirmations (e.g., MACD
cross, RSI extremes), completed break-
outs with volume, decisive trendline
breaks.
•Secondary Signals (Medium Weight):
Multi-indicator alignment, S/R interac-
tions, near-complete patterns.
•Risk Filters:Downweight conflicting sig-
nals unless one side dominates; reflect his-
torical outcomes; setrisk–rewardin[1.2,
1.8]based on volatility and conviction.
•Memory Integration:Reference compa-
rable past scenarios, outcomes, and confi-
dence adjustments.
REQUIRED OUTPUT FORMAT (JSON):
Return only valid JSON
{
"forecast_horizon":"Next {time_frame}
candlestick prediction",
"decision":"LONG or SHORT",
"confidence_level":"HIGH/MEDIUM/LOW",
"primary_drivers":["2-3 strongest
signals"],
"memory_insights":"Key lessons from
similar scenarios",
"justification":"Concise reasoning
combining current analysis with
history",
"risk_reward_ratio":"<float between
1.2 and 1.8>",
"stop_loss_rationale":"Brief risk
management rationale"
}
Decision Hierarchy
1.Strong Alignment:All reports + positive
memory outcomes →High confidence
trade.
2.Moderate Alignment:2/3 reports align
+ supportive memory →Medium confi-
dence.

3.Weak Signals:Mixed reports →Choose
direction with strongest recent momen-
tum and best precedent.
Execute with precision. The market waits for
no one.
Prompt for DecisionAgent in our multi-agent
LLM framework.The agent fuses multi-indicator
evidence with memory context and outputs asingle
LONG/SHORT decision in strict JSON, including
confidence and risk–reward calibration.
Prompt
You are conducting a post-trade analysis as an
elite HFT quantitative analyst.
TRADE SUMMARY (given):Stock, Time-
frame, Decision, Confidence, Expected Risk–
Reward.
ORIGINAL ANALYSIS SUMMARY
(given):Concise record of drivers, memory
insights, justification, and risk management.
ACTUAL TRADE OUTCOME (given):
P&L, final status (profitable/loss/breakeven),
and realized market behavior.
REFLECTION FRAMEWORK
1.Decision Quality:Was the analysis
sound? Did primary drivers anticipate the
move? Did memory insights help?
2.Execution:Interpretation fidelity of indi-
cators/patterns; appropriateness of confi-
dence.
3.Learning:Under/overestimated condi-
tions; most/least reliable signals; how to
adapt next time.
4.Memory Integration:How does this out-
come update memory; any weight shifts
for indicators.
REQUIRED OUTPUT:Produce a concise
3–5 sentence reflection covering: key factors
behind the outcome; what worked/failed; con-
crete lessons; any adjustments to approach or
confidence calibration. Focus on actionable
improvements for similar future setups.
Post-decision reflection prompt for Decision-
Agent in our multi-agent LLM framework.The
reflection converts outcome feedback into memory-
aligned guidance, tightening future signal weight-ing and confidence calibration.
B Computation and Mathematical
Formulation of Indicators
This appendix provides the formal definitions and
computational procedures for the six technical in-
dicators integrated within the META (Memory-
Enhanced Trading Agent) architecture. Each in-
dicator serves as a distinct analytical perspective
on market dynamics—capturing momentum, mean
reversion, trend persistence, or value anchoring.
All indicators are computed from raw OHLCV
(Open–High–Low–Close–V olume) data and act as
the quantitative foundation for the multi-agent rea-
soning framework described in Appendix A.
B.1 Moving Average Convergence Divergence
(MACD)
Definition.The Moving Average Convergence Di-
vergence (MACD) is a momentum indicator de-
rived from the difference between two exponen-
tial moving averages (EMAs) of price, designed to
quantify acceleration and deceleration of market
trends. Formally, for price seriesP t:
MACD t=EMA fast,t−EMA slow,t,
where
EMA k,t=αkPt+(1−α k)EMA k,t−1, α k=2
k+ 1.
Common parameters are(fast,slow) = (12,26).
The signal line is defined as:
Signalt=EMA 9(MACD t),
and the histogram, which reflects short-term diver-
gence between trend speeds, is:
Histt=MACD t−Signalt.
Interpretation.The MACD measures momen-
tum by comparing short- and long-term EMAs. A
positive value implies upward acceleration; nega-
tive values imply downward momentum. Crossover
events between MACD and the signal line indicate
potential reversals, while histogram expansion sug-
gests strengthening momentum. Because MACD
captures the rate of change in trend direction, it is
frequently used in HFT environments for detecting
early inflection points.

B.2 Relative Strength Index (RSI)
Definition.The Relative Strength Index (RSI)
quantifies the internal strength of recent price move-
ments by comparing average gains to average
losses within a defined lookback period n(typically
n= 14 ). For a price difference ∆Pt=Pt−Pt−1:
AvgGaint=EMA n(max(∆P t,0))
AvgLosst=EMA n(max(−∆P t,0)).
The relative strength (RS) and RSI are then com-
puted as:
RSt=AvgGaint
AvgLosst,RSI t= 100−100
1 +RS t.
Interpretation.RSI values oscillate between 0
and 100, serving as a bounded momentum oscilla-
tor. Readings above 70 denote overbought condi-
tions (potential reversal risk), while readings below
30 suggest oversold conditions (potential recovery).
Crossings of the 50 midpoint line are interpreted as
transitions in trend dominance. In high-frequency
contexts, RSI slope and failure-swing formations
provide early warnings of momentum exhaustion
before price reversal becomes visible.
B.3 Anchored Volume Weighted Average
Price (A VWAP)
Definition.The Anchored V olume Weighted Av-
erage Price (A VWAP) extends the conventional
VWAP by fixing the accumulation anchor to a spe-
cific event time τ(e.g., major news or breakout). It
represents the cumulative volume-weighted mean
of traded prices since the anchor point:
A VWAP t=Pt
i=τPiViPt
i=τVi,
where Pidenotes the typical or closing price and
Vithe corresponding trading volume.
Interpretation.A VWAP serves as a dynamic
“fair value” benchmark representing the mean price
paid per traded volume since a specific anchor.
When price trades above A VWAP, the market is
said to be in a bullish premium phase; when below,
in a bearish discount phase. Because institutional
participants often reference anchored VWAP levels
for liquidity decisions, these lines act as adaptive
support or resistance zones in HFT systems.B.4 Simple Moving Average (SMA)
Definition.The Simple Moving Average (SMA) is
a non-weighted mean of the most recent nclosing
prices:
SMA t=1
nn−1X
i=0Pt−i.
For two SMAs with different windows n1< n 2,
the crossover between SMA n1andSMA n2pro-
vides directional signals.
Interpretation.SMA acts as a linear filter that
removes short-term noise to reveal the underlying
trend. Shorter-period SMAs respond quickly to
price fluctuations, while longer-period SMAs de-
fine macro-trend bias. The alignment of multiple
SMAs into a monotonic order (short > medium >
long) indicates trend strength, whereas frequent
crossovers reflect low directional conviction or
range-bound conditions.
B.5 Stochastic Oscillator
Definition.The Stochastic Oscillator measures the
relative position of the current close within a recent
price range of lengthn:
%Kt= 100×Pt−Ln
Hn−Ln,%D t=SMA 3(%K t),
where Hn= max(P t−n+1 , . . . , P t)andLn=
min(P t−n+1 , . . . , P t)denote the highest and low-
est prices in the past nperiods (commonly n= 14 ).
Interpretation.The Stochastic Oscillator quan-
tifies momentum relative to the recent range rather
than absolute changes. Readings above 80 signify
potential overbought conditions, while values be-
low 20 indicate oversold conditions. Crossovers
between %K and %D provide early entry and exit
signals. In HFT environments, the slope and curva-
ture of %K serve as short-term acceleration indica-
tors, and divergence between Stochastic and price
action often precedes directional reversal.
B.6 Heiken Ashi (HA)
Definition.Heiken Ashi candlesticks are a trans-
formation of traditional OHLC data that average
and smooth price information to reduce noise. The
recursive formulation is:
HA_Close t=1
4(Ot+H t+Lt+Ct),
HA_Opent=1
2(HA_Opent−1+HA_Close t−1),
HA_Hight= max(H t,HA_Opent,HA_Close t),
HA_Low t= min(L t,HA_Opent,HA_Close t).

Interpretation.Heiken Ashi averages smooth
intra-period volatility to better capture directional
persistence. Extended sequences of uniform-color
candles represent strong trends, while small-bodied
candles or alternating colors indicate weakening
momentum or consolidation. This transformation
is particularly useful in high-frequency applica-
tions, where standard candlestick noise can obscure
subtle but statistically persistent directional biases.
C Case Study
This case study illustrates a full trading cycle within
META, from signal perception to reflective mem-
ory update. Each trade begins with inputK-line
data analyzed in parallel by specializedIndicator
Agents(e.g., MACD, RSI, A VWAP, SMA, Stochas-
tic, Heiken Ashi), each generating a structured sig-
nal report. These outputs form anindicator vector
that encodes the current market state and is used to
retrieve historically similar episodes from the mem-
ory base through cosine similarity search. The re-
trieved memories, together with real-time analyses,
are then passed to theDecision Agentfor synthesis
into a trading action (LONG, SHORT, or HOLD)
with associated confidence. Following trade execu-
tion, theReflection Moduleevaluates the outcome
and updates the episodic memory—adding, refin-
ing, or pruning trade instances to ensure continual
learning and adaptive decision-making.
Trend Agent Output Response
1. Trendline Interaction:
•The price is interacting with both the support
(blue) and resistance (red) lines.
•Recent candles have bounced off the support line,
indicating it is holding for now.
•The resistance line has been tested multiple times,
with the price failing to break through.
2. Trendline Slope:
•Both the support and resistance lines are sloping
downward, indicating a bearish trend.
3. Recent K-line Behavior:
•The price is compressing between the support and
resistance lines, showing consolidation.
•There is a slight downward bias as the price is
closer to the support line.
Prediction: Downward Trend
Reasoning:
•The downward slope of both trendlines suggests a
bearish trend.•The price is closer to the support line, and repeated
tests of this line may lead to a breakdown.
•The inability to break above the resistance line
indicates strong selling pressure.
Signals:
•Watch for a break below the support line for con-
firmation of a downward move.
•If the price breaks above the resistance line, it
could indicate a reversal, but current signals favor
a downward trend.
MACD Agent Output Response
MACD Analysis:
1. Signal Line Crossovers:
•The MACD line (blue) is currently below the sig-
nal line (red), indicating a bearish crossover.
•This crossover occurs below the zero line, rein-
forcing bearish momentum.
2. Zero Line Crossovers:
•The MACD line is below zero, indicating negative
momentum.
•No recent crossovers of the zero line, suggesting
sustained bearish momentum.
3. Histogram Analysis:
•The histogram bars are contracting, indicating
weakening bearish momentum.
• No recent histogram zero-line crossovers, but the
contraction suggests a potential momentum shift.
4. Divergence Patterns:
•No clear bullish or bearish divergences observed
between price action and MACD behavior.
•Price lows align with MACD lows, confirming
current trend.
5. Momentum Strength:
•The MACD line is significantly below zero, indi-
cating strong bearish momentum.
•Both MACD and signal lines are sloping down-
ward, but the slope is decreasing, suggesting de-
celeration.
Conclusion:
•Primary Signal:Hold
•Momentum Direction:Bearish
•Confidence Level:Medium
•Key Supporting Evidence:Bearish crossover
below zero line, contracting histogram.
•Risk Considerations:Potential for false sig-
nals due to histogram contraction; watch for any
bullish crossover or divergence for a trend rever-
sal.

A VWAP Agent Output Response
1. Support/Resistance Analysis:
•Price Position:The price is currently below the
A VWAP line.
•Price Reaction:The price has been rejected at
the A VWAP line multiple times, indicating it is
acting as resistance.
•Dynamic Role:A VWAP is acting as dynamic
resistance in a downtrend.
2. Mean Reversion Opportunities:
•Deviation:The price has deviated significantly
below the A VWAP, suggesting a potential pullback
to fair value.
•Extreme Stretches:The current stretch below
A VWAP suggests a possible mean reversion op-
portunity.
•Typical Range:The price has shown a tendency
to revert to the A VWAP after deviations.
3. Breakout/Breakdown Signals:
•Decisive Breaks:There have been decisive breaks
below the A VWAP with strong volume.
•Reclaiming/Losing A VWAP:The price has lost
the A VWAP after being above it.
•Volume Behavior:V olume spikes coincide with
A VWAP tests and breakdowns.
4. A VWAP Slope and Bias:
•Slope:The A VWAP line is sloping downward,
indicating a bearish bias.
•Steepness:The slope is moderately steep, sug-
gesting a strong directional bias.
•Slope Dynamics:The slope is consistent, main-
taining a bearish trend.
5. Fair Value Assessment:
•Institutional Fair Value:The A VWAP level rep-
resents institutional fair value.
•Price Position:Current prices are trading at a
discount to volume-weighted fair value.
•Evolution:The fair value relationship has shifted
from premium to discount as the session pro-
gressed.
6. Volume-Price Relationship:
•Volume Increase:V olume increases when the
price approaches the A VWAP.
•Volume Spikes:Notable volume spikes occur
during A VWAP tests and breakdowns.
•Volume Support:V olume supports the current
bearish price action relative to A VWAP.
Comprehensive Assessment:•Primary Trade Signal: Sell
–Rationale:The price is below a downward-
sloping A VWAP, acting as resistance with
strong volume confirmation.
•A VWAP Bias: Bearish
–Based on:Downward slope and price posi-
tion below A VWAP.
•Fair Value Assessment: Discount
–Relative to A VWAP:Current prices are be-
low the volume-weighted fair value.
•Entry Strategy:
–Approach:Enter short positions on rallies
towards the A VWAP with confirmation of
rejection.
•Risk Management:
–Stop Loss:Above recent highs near the
A VWAP.
–Profit Target:At recent lows or further
below, considering trend continuation.
•Volume Confirmation:
–Support:V olume supports the bearish sig-
nal, with spikes during A VWAP interactions
confirming resistance.
This analysis suggests a continuation of the downtrend,
with AVWAP serving as a key resistance level.
Heiken Ashi Agent Output Response
Heiken Ashi Analysis
1. Trend Direction & Consistency:
•Consecutive Candles:There is a sequence of
green candles followed by a longer sequence of
red candles.
•Dominant Color:Red candles dominate, indicat-
ing a bearish trend.
•Recent Color Changes:A few green candles
appear at the end, suggesting a potential shift.
•Trend Cleanliness:The trend is relatively clean
with minimal alternation during the bearish phase.
2. Trend Strength Indicators:
•Candle Bodies:Initially large green bodies, fol-
lowed by large red bodies, indicating strong mo-
mentum in both directions.
•Wicks:Red candles have minimal upper wicks,
showing strong bearish momentum.
•Body Size:Decreasing body size towards the end
of the red sequence suggests deceleration.
3. Reversal Pattern Recognition:
•Doji-like Formations:Some small-bodied can-
dles with wicks appear, indicating indecision.

•Color Change:Green candles after a strong red
sequence suggest a potential reversal.
•Spinning Tops/Hammer-like Patterns:Present
towards the end of the red sequence.
4. Momentum Acceleration/Deceleration:
•Candle Size:Recent candles are smaller, indicat-
ing fading momentum.
•Gaps:No significant gaps observed.
•Wick Reduction:Progressive wick reduction in
the bearish direction.
5. Consolidation vs Trending Phases:
•Color Alternation:Recent alternation suggests a
potential consolidation phase.
•Small Bodies with Long Wicks:Indicate range-
bound action.
6. Entry/Exit Signal Quality:
•Signal Quality:Medium confidence due to recent
indecision and potential reversal signals.
•Optimal Entry Points:Consider buying if a
strong green candle confirms a reversal.
•Trend Exhaustion:Watch for further green can-
dles to confirm trend change.
7. Multi-Candle Pattern Analysis:
•Three Consecutive Same-Color Candles:
Present in both green and red sequences,
confirming trends.
•Two-Candle Reversal Patterns:Potential rever-
sal pattern forming with recent green candles.
•Extended Trend Sequences:Red sequence near-
ing exhaustion.
Comprehensive Assessment:
•Trend Status:Weak Bearish transitioning to Side-
ways
•Momentum Phase:Decelerating
•Signal Quality:Medium
•Trading Action:Wait for better setup
•Entry Strategy:Look for a strong green candle
to confirm reversal before entering a buy position.
•Exit Criteria:Exit if red candles reappear with
strong bodies and minimal wicks.
•Risk Assessment:Moderate probability of false
signals due to recent indecision and small-bodied
candles.SMA Agent Output Response
SMA Analysis:
1. Trend Identification:
•Price/SMA Relationship:The price is currently
below both the 20-period (blue) and 40-period
(red) SMAs, indicating a bearish trend.
•SMA Alignment:The 20-period SMA is below
the 40-period SMA, reinforcing the downtrend.
2. Moving Average Crossovers:
•Death Cross:The 20-period SMA has crossed
below the 40-period SMA, confirming a bearish
signal.
•Recent Crossovers:No recent golden cross ob-
served; the last significant crossover was bearish.
3. Dynamic Support/Resistance:
•Support:Price is not finding support at the SMAs;
it is below both.
•Resistance:The 20-period SMA is acting as re-
sistance during rallies.
•Respect of Levels:Price has consistently stayed
below the SMAs, respecting them as resistance.
4. Multiple MA Ribbon Analysis:
•SMA Stacking:The SMAs are neatly stacked
with the shorter below the longer, indicating a
strong bearish trend.
•Ribbon Behavior:The ribbon is fanning out, sug-
gesting a strengthening downtrend.
5. Mean Reversion Signals:
•Price Extension:Price is significantly extended
below the SMAs, indicating potential for a mean
reversion or pullback.
•Deviation:The price is stretched, suggesting a
possible short-term reversal.
6. Slope and Alignment:
•SMA Slopes:Both SMAs are sloping downwards,
indicating a strong bearish trend.
•Alignment:SMAs are aligned in the same down-
ward direction, confirming trend strength.
•Slope Steepness:The slopes are moderately steep,
indicating a strong trend.
Conclusion:
•Primary Trend Direction:Bearish
•Trend Strength:Strong
•Trading Bias:Sell on Rallies
•Key SMA Levels:Watch the 20-period SMA
(∼86000) as resistance.
•Confirmation Needed:Look for confirmation
from other indicators like RSI for oversold condi-
tions or MACD for momentum confirmation.
This analysis suggests a strong bearish trend with po-
tential short-term pullbacks. Traders should consider
selling on rallies, especially if price approaches the 20-
period SMA resistance.

Stochastic Oscillator Agent Output Re-
sponse
Stochastic Oscillator Analysis:
1. Overbought/Oversold Conditions:
•The Stochastic is currently below the 20 level,
indicating oversold conditions.
•It has been in the oversold zone for a short period.
•There is no clear exit from the oversold zone yet,
suggesting a potential continuation of the current
trend.
2. Line Crossovers:
•The %K line is crossing above the %D line, which
is a bullish signal.
•This crossover is occurring in the oversold zone,
which can indicate a potential reversal.
3. Divergence Patterns:
•There is no clear bullish divergence as the price
and Stochastic both show lower lows.
• No bearish divergence is present.
4. Failure Swings:
•No clear bullish failure swing is observed as the
Stochastic has not made a higher low in the over-
sold territory.
• No bearish failure swing is present.
5. Centerline Analysis:
•The Stochastic is below the 50 centerline, indicat-
ing bearish momentum.
•A crossover above 50 would suggest a shift to
bullish momentum, but this has not occurred yet.
6. Slope and Momentum:
•The %K line is starting to slope upwards, indicat-
ing a potential increase in bullish momentum.
•The %D line is also beginning to turn upwards,
supporting the potential for a bullish reversal.
•Both lines are making lower highs and lows, con-
sistent with the current downtrend.
Conclusion:
•Primary Signal:Hold
•Momentum Direction:Neutral to Bullish (poten-
tial reversal)
•Confidence Level:Medium
•Key Supporting Evidence:Bullish crossover in
oversold zone, upward slope of %K and %D lines.
•Risk Considerations:The Stochastic is still in
the oversold zone, and the overall trend is bearish.
There is a risk of false signals if the price does not
confirm the reversal.
This analysis suggests caution, as the bullish signals are
emerging but not yet confirmed by price action.RSI Agent Output Response
RSI Analysis:
1. Overbought/Oversold Conditions:
•The RSI is not currently in overbought (>70) or
oversold (<30) territory. It is around the 40 level.
•There is no recent crossover from these extreme
zones.
2. Centerline Crossovers:
•The RSI is below the centerline (50), indicating
bearish momentum.
•There have been no recent crossovers of the 50
level.
3. Failure Swings:
•No clear failure swings are observed as RSI hasn’t
reached extreme levels recently.
4. Divergence Patterns:
•Bullish Divergence:The price made a lower low,
while RSI made a higher low, suggesting potential
bullish reversal.
5. RSI Trendlines & Patterns:
•No visible trendlines or chart patterns like trian-
gles are apparent on the RSI.
6. Swing Rejection Patterns:
• No clear swing rejection patterns are observed.
7. RSI Range Analysis:
•RSI is ranging between 30 and 50, showing bear-
ish momentum but not extreme exhaustion.
• Recent support around 30 and resistance near 50.
Conclusion:
•Primary Signal:Hold
•Momentum Direction:Bearish
•Confidence Level:Medium
•Key Supporting Evidence:Bullish divergence
observed, but overall momentum remains bearish.
•Risk Considerations:Potential for false signals
due to lack of strong reversal patterns and RSI not
reaching extreme levels.
The bullish divergence suggests a possible reversal, but
the overall bearish momentum and lack of strong rever-
sal signals warrant caution.
After analyzing the current market state, META
encodes the aggregated indicator outputs into an
indicator vectorand retrieves the most similar past
trades from memory. These analogous cases serve
as experiential references, anchoring the decision

process in prior market behaviors.
Retrieved Memory 1 (Similarity: 0.89)
Date:2025-09-30T17:47:07.801365
Forecast Horizon:Next 4-hour candlestick prediction
Confidence Level:Medium
Primary Drivers:
•Descending Triangle pattern indicating potential
bearish breakout
•Price below VWAP and SMAs, confirming bearish
trend
Memory Insights:No highly relevant past trade memo-
ries found for current market conditions.
Justification:The analysis suggested a bearish senti-
ment with indicators pointing toward oversold conditions.
The descending triangle and price positioning below key
moving averages supported a continuation of the down-
ward trend.
Risk–Reward Ratio:1.5
Stop-Loss Rationale:Stop loss placed above recent
resistance to mitigate reversal risk in alignment with the
bearish setup.
Decision:SHORT
Outcome:387.31
Reflection:The trade’s success stemmed from correctly
identifying the descending triangle and confirming bear-
ish momentum via VWAP and SMA alignment. Ef-
fective risk management further supported profitability.
However, the medium confidence rating highlights an
opportunity to enhance robustness through expanded his-
torical referencing and refined indicator weighting in
future decisions.
Retrieved Memory 2 (Similarity: 0.83)
Date:2025-09-17T15:16:46.572820
Forecast Horizon:Next 4-hour candlestick prediction
Confidence Level:Medium
Primary Drivers:
•Price below both 20 and 40-period SMAs indicat-
ing a bearish trend
•MACD line significantly below signal line with
negative histogram indicating strong bearish mo-
mentum
•Price below downward-sloping A VWAP acting as
resistance
Memory Insights:Previous scenarios with similar bear-
ish setups have shown mixed outcomes, highlighting the
need for cautious confidence and dynamic risk manage-
ment.
Justification:Technical indicators including the MACD,
SMAs, and A VWAP suggested a continuation of the
bearish trend. Historical analysis supported this decision,
though with mixed results.
Risk–Reward Ratio:1.5
Stop-Loss Rationale:Stop loss placed above recent
highs near the A VWAP to manage reversal risk.
Decision:SHORT
Outcome:–76.69
Reflection:The loss arose from overreliance on tech-
nical bearish signals that failed to materialize. Futureimprovements include better weighting of historical in-
sight, adaptive stop-loss adjustment, and accounting for
broader market factors.
Retrieved Memory 3 (Similarity: 0.83)
Date:2025-09-18T16:40:44.859425
Forecast Horizon:Next 4-hour candlestick prediction
Confidence Level:Medium
Primary Drivers:
•Price below both 20 and 40-period SMAs indicat-
ing a bearish trend
•MACD line below signal line with negative his-
togram indicating bearish momentum
•Descending Triangle pattern suggesting bearish
breakout
Memory Insights:Similar setups with descending tri-
angles and bearish indicators have produced mixed out-
comes, calling for cautious confidence.
Justification:The bearish alignment of MACD, SMAs,
and chart pattern supported a short position. Historical
results validated this setup.
Risk–Reward Ratio:1.5
Stop-Loss Rationale:Stop loss placed above recent
highs near the A VWAP.
Decision:SHORT
Outcome:571.08
Reflection:Profit resulted from precise identification
of the descending triangle and indicator alignment. The
trade confirms the importance of weighted indicator cali-
bration and dynamic confidence scaling.
Retrieved Memory 4 (Similarity: 0.82)
Date:2025-09-16T17:21:41.742237
Forecast Horizon:Next 4-hour candlestick prediction
Confidence Level:Medium
Primary Drivers:
•MACD line below signal line with negative his-
togram indicating bearish momentum
•Price below both 20 and 40-period SMAs confirm-
ing downtrend
•Descending Triangle pattern suggesting potential
breakdown
Memory Insights:Prior successful shorts in similar
descending triangle setups with bearish indicators.
Justification:Strong bearish signals supported a short
trade. Historical parallels suggested profit potential.
Risk–Reward Ratio:1.5
Stop-Loss Rationale:Stop loss placed above recent
highs near the A VWAP.
Decision:SHORT
Outcome:–150.00
Reflection:The loss resulted from overestimating bear-
ish momentum. Future improvements include multi-
signal confirmation, contextual confidence scaling, and
indicator reweighting.

Retrieved Memory 5 (Similarity: 0.81)
Date:2025-09-20T00:33:33.390160
Forecast Horizon:Next 4-hour candlestick prediction
Confidence Level:Medium
Primary Drivers:
•RSI and Stochastic Oscillator in oversold territory
indicating potential reversal
• Bullish crossover in Stochastic Oscillator
•Price trading below VWAP suggesting potential
undervaluation
Memory Insights:Oversold conditions and bullish
crossovers often lead to short-term reversals in past cases.
Justification:Despite a bearish backdrop, RSI, Stochas-
tic, and VWAP indicated undervaluation warranting a
cautious long.
Risk–Reward Ratio:1.5
Stop-Loss Rationale:Stop loss placed below recent
lows to mitigate continuation risk.
Decision:LONG
Outcome:76.69
Reflection:Success stemmed from recognizing oversold
conditions and aligning multiple reversal signals. Future
enhancements include dynamic indicator weighting and
adaptive confidence tuning based on confluence strength.
The retrieved memories provide valuable con-
textual grounding for the current trade. By exam-
ining both profitable and unprofitable precedents,
META’s synthesis stage integrates lessons from
historical performance into the present decision-
making process. Profitable cases such as Memory 3
highlight the effectiveness of aligning descend-
ing triangle patterns with bearish indicator con-
vergence, reinforcing confidence in similar market
setups. Conversely, losses observed in Memory 2
and Memory 4 underscore the risks of overreliance
on technical signals without sufficient contextual
validation or adaptive stop-loss control. Through
reflective comparison, META dynamically adjusts
the weighting of key indicators—such as MACD,
SMA alignment, and A VWAP slope—while moder-
ating confidence when past patterns exhibit mixed
outcomes. This learning mechanism enables the
agent to refine its strategy beyond static rule-
following, improving robustness by encoding both
the strengths and weaknesses of prior trades into
future reasoning.
Decision Agent Output Response
Forecast Horizon:Next 4h candlestick prediction
Decision:SHORT
Confidence Level:Medium
Primary Drivers:
• Descending Triangle pattern indicating potential
bearish breakout•Price below both 20 and 40-period SMAs con-
firming downtrend
•MACD line below signal line with negative his-
togram indicating bearish momentum
Memory Insights:Previous similar setups with de-
scending triangle patterns and price below key moving
averages have resulted in mixed outcomes, highlight-
ing the need for cautious confidence and dynamic risk
management.
Justification:The current market analysis shows a
strong bearish trend with the price below key moving
averages and a descending triangle pattern suggesting a
potential breakdown. Historical memory supports the
decision to short in similar conditions, where the align-
ment of technical indicators led to both successful and
unsuccessful outcomes. The overall bearish sentiment
is reinforced by the price’s position below key moving
averages and the A VWAP.
Risk–Reward Ratio:1.5
Stop-Loss Rationale:Place stop loss above recent
highs near the A VWAP to manage risk in case of a
reversal.
Simulated Trade Outcome
Outcome:WIN
PnL:+615.30
Entry Price:82,548.81
Exit Price:81,933.51
Performance Summary:The trade achieved a prof-
itable outcome, with price movement aligning closely
to the anticipated bearish continuation. Execution ad-
hered to the risk–reward target of 1.5, validating both
directional accuracy and the stop-loss placement strat-
egy.
Trade Reflection
The key factors that led to the profitable outcome were
the accurate identification of the descending triangle pat-
tern and the confirmation of bearish momentum through
the MACD and moving averages. The original analysis
was sound, as the primary drivers correctly predicted
market movement, and historical memory insights pro-
vided a balanced perspective on risk management.
However, the medium confidence level suggests there
was room for greater conviction given the alignment
of technical indicators. For future scenarios, it would
be beneficial to emphasize pattern strength and multi-
indicator confluence, adjusting confidence upward in
similar conditions. Additionally, refining stop-loss
placement to account for volatility could further im-
prove risk management and trade robustness.
Following the trade’s completion and reflection
phase, META transitions into the final stage of its
cognitive workflow—theMemory Consolidation
Phase. In this stage, theMemory Operator( Ψ)
evaluates the outcome of the trade alongside re-
trieved precedents to determine how the memory
base should evolve. Based on the reflection, past
memories may be updated, retained, or removed,
while the newly completed trade is encoded as a

new episodic memory. This process ensures that
the agent’s experiential knowledge remains both
adaptive and performance-oriented, continually re-
fining its future decision quality.
Memory Agent Output Response
Post-Trade Memory Operations:
•Memory 1:NOOP — retained without modifi-
cation.
•Memory 2:NOOP — maintained as historical
reference.
•Memory 3:UPDATE — performance data ap-
pended with revised confidence weighting.
•Memory 4:DELETE — removed due to re-
peated underperformance in similar contexts.
•Memory 5:UPDATE — outcome consistency
improved; metadata refined for future retrieval
weighting.
•New Memory:ADD — stored as a new episodic
record representing the current trade outcome and
reflection.
Summary:The Memory Operator integrates experien-
tial feedback by updating profitable analogs, pruning
uninformative cases, and adding the newly completed
trade as a fresh memory entry. This targeted consolida-
tion ensures the memory base remains relevant, adap-
tive, and performance-aligned across future retrieval
cycles.
This case study illustrates META’s complete
cognitive workflow, demonstrating how the sys-
tem transforms perception-driven analysis into
experience-informed reasoning. By integrating in-
sights from multiple indicator agents, META con-
structs a structured representation of the market
state that captures both short-term and long-term
dynamics. The retrieval of analogous trade mem-
ories provides contextual grounding, enabling the
decision agent to synthesize new judgments with
awareness of past outcomes. Finally, the reflection
and memory consolidation stages close the cogni-
tive loop, allowing the agent to refine its internal
knowledge base through targeted updates and se-
lective forgetting.
Through this process, META exhibits several
key advantages: (1) it enhances predictive stability
by grounding real-time reasoning in historical evi-
dence; (2) it maintains interpretability through ex-
plicit agent-level outputs; and (3) it achieves contin-
ual learning without retraining, instead relying on
reflection-driven memory management. Together,
these mechanisms position META as a step toward
genuinely adaptive, self-improving trading intelli-gence capable of evolving alongside dynamic mar-
ket conditions.
D Metrics Design
To ensure comparability with prior work, we adopt
the evaluation metrics introduced in QUANTAGENT
(Xiong et al., 2026), which quantify both predictive
accuracy and profitability under risk-constrained
execution. All metrics are computed within a fixed
three-candlestick horizon, corresponding to the
short-term forecasting window defined by the trad-
ing agent.
Directional Accuracy ( α).Directional accuracy
measures the model’s consistency in predicting the
correct price movement over the next three can-
dlesticks. For a LONG decision, each candle that
closes above the current close counts as a correct
prediction; for a SHORT decision, each candle
that closes below the current close counts. The
accuracy is given by:
α=C
T,
where Cdenotes the number of correctly predicted
candles and T= 3 the evaluation horizon. This
formulation aligns with the Mean Directional Ac-
curacy metric used in financial forecasting.
Rate-of-Return (RoR) Metrics.To evaluate
trade profitability and risk exposure, we measure
multiple forms of rate of return between the entry
and exit prices, incorporating realistic stop-loss and
take-profit execution. The base stop-loss thresh-
oldρis fixed at 0.0005 (0.05%), while the reward
threshold is determined by the model-generated
risk–reward ratio r, yielding a maximum allowed
gainR=r·ρand maximum loss−r·ρ:
R=r·ρ, ρ= 0.0005.
Each trade is exited at the first intra-candle price
that hits either threshold.
•Cumulative Return ( Rcc):The realized re-
turn over the forecast horizon, reflecting the
net gain or loss achieved at the earliest hit of
either the take-profit or stop-loss limit:
Rcc=

Pexit−P entry
Pentry,if LONG,
Pentry−P exit
Pentry,if SHORT.
This simulates a risk-managed intra-candle
execution with bounded return outcomes.

•Maximum Return ( Rmax):The best-case
rate of return achievable over the same win-
dow, assuming an optimal intra-candle exit:
Rmax=

Phigh−P entry
Pentry,if LONG,
Pentry−P low
Pentry,if SHORT.
Rmaxrepresents the upper bound of potential
profit.
•Minimum Return ( Rmin):The most adverse
price movement within the same horizon:
Rmin=

Plow−P entry
Pentry,if LONG,
Pentry−P high
Pentry,if SHORT.
Rmincaptures the worst-case drawdown under
identical trade direction.
•Similarity-weighted Return Alignment
(Rsim):This metric assesses the consistency
between similarity scores of retrieved memo-
ries and realized profitability. For each trade
i, with similarity weights wijoverkretrieved
episodes and their respective realized returns
R(j)
cc, we define:
Rsim=1
NNX
i=1Pk
j=1wij·R(j)
cc
Pk
j=1wij,
where Nis the number of evaluated trades.
Higher Rsimvalues indicate that the retrieval
mechanism successfully identifies memories
whose return patterns align with the current
market context.
Together, these metrics provide a comprehensive
view of agent performance: αmeasures directional
accuracy; Rccreflects realized profitability; Rmax
andRminestablish upper and lower risk bounds;
andRsimquantifies the semantic consistency be-
tween retrieved experiences and market outcomes.
This unified metric suite captures predictive pre-
cision, profitability, and memory relevance under
realistic HFT execution conditions, fully aligned
with the QuantAgent evaluation protocol.
EUse of Large Language Models (LLMs)
Large Language Models (LLMs) were employed
solely as supportive instruments to improve theclarity, coherence, and grammatical precision of
the manuscript. Specifically, GPT-5 was utilized
to refine portions of the text—primarily within the
Introduction and Methodology sections—to ensure
readability and stylistic consistency. The authors
retain full responsibility for all scientific and tech-
nical contributions, including the formulation of
research questions, experimental design, implemen-
tation of the META framework, and validation of
empirical findings. The use of LLMs was limited
strictly to editorial assistance and did not influence
the conceptual, analytical, or experimental aspects
of this work.