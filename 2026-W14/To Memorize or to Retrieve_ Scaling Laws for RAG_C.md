# To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining

**Authors**: Karan Singh, Michael Yu, Varun Gangal, Zhuofu Tao, Sachin Kumar, Emmy Liu, Steven Y. Feng

**Published**: 2026-04-01 10:26:03

**PDF URL**: [https://arxiv.org/pdf/2604.00715v1](https://arxiv.org/pdf/2604.00715v1)

## Abstract
Retrieval-augmented generation (RAG) improves language model (LM) performance by providing relevant context at test time for knowledge-intensive situations. However, the relationship between parametric knowledge acquired during pretraining and non-parametric knowledge accessed via retrieval remains poorly understood, especially under fixed data budgets. In this work, we systematically study the trade-off between pretraining corpus size and retrieval store size across a wide range of model and data scales. We train OLMo-2-based LMs ranging from 30M to 3B parameters on up to 100B tokens of DCLM data, while varying both pretraining data scale (1-150x the number of parameters) and retrieval store size (1-20x), and evaluate performance across a diverse suite of benchmarks spanning reasoning, scientific QA, and open-domain QA. We find that retrieval consistently improves performance over parametric-only baselines across model scales and introduce a three-dimensional scaling framework that models performance as a function of model size, pretraining tokens, and retrieval corpus size. This scaling manifold enables us to estimate optimal allocations of a fixed data budget between pretraining and retrieval, revealing that the marginal utility of retrieval depends strongly on model scale, task type, and the degree of pretraining saturation. Our results provide a quantitative foundation for understanding when and how retrieval should complement pretraining, offering practical guidance for allocating data resources in the design of scalable language modeling systems.

## Full Text


<!-- PDF content starts -->

Preprint. Under review.
To Memorize or to Retrieve:
Scaling Laws for RAG-Considerate Pretraining
Karan Singh1,†, Michael Yu2,†, Varun Gangal3,†,
Zhuofu Tao2,†,Sachin Kumar4,†,Emmy Liu5,†,Steven Y. Feng1,†
1Stanford University2Independent Researcher3Patronus AI
4The Ohio State University5Carnegie Mellon University †DegenAI Labs
Abstract
Retrieval-augmented generation (RAG) improves language model (LM) per-
formance by providing relevant context at test time for knowledge-intensive
situations. However, the relationship between parametric knowledge ac-
quired during pretraining and non-parametric knowledge accessed via
retrieval remains poorly understood, especially under fixed data budgets.
In this work, we systematically study the trade-off between pretraining
corpus size and retrieval store size across a wide range of model and data
scales. We train OLMo-2-based LMs ranging from 30M to 3B parameters on
up to 100B tokens of DCLM data, while varying both pretraining data scale
(1–150 ×the number of parameters) and retrieval store size (1–20 ×), and
evaluate performance across a diverse suite of benchmarks spanning reason-
ing, scientific QA, and open-domain QA. We find that retrieval consistently
improves performance over parametric-only baselines across model scales
and introduce a three-dimensional scaling framework that models perfor-
mance as a function of model size, pretraining tokens, and retrieval corpus
size. This scaling manifold enables us to estimate optimal allocations of
a fixed data budget between pretraining and retrieval, revealing that the
marginal utility of retrieval depends strongly on model scale, task type,
and the degree of pretraining saturation. Our results provide a quantitative
foundation for understanding when and how retrieval should complement
pretraining, offering practical guidance for allocating data resources in the
design of scalable language modeling systems.
1 Introduction
Scaling laws (Hestness et al., 2017; Kaplan et al., 2020) have established how language model
(LM) performance improves with parameters and training tokens, but they treat the training
corpus as monolithic. In standard pretraining, all available data is consumed parametrically,
implicitly assuming that knowledge should be compressed into model weights. Retrieval-
augmented generation (RAG) introduces a new degree of freedom: a portion of the corpus
can instead be held out as an external datastore and accessed at inference time. These two
uses of data are fundamentally different, with distinct computational costs, inductive biases,
and failure modes. For example, parametric learning may lead to an inaccurate internal
world model and hallucination tendencies (Liu et al., 2026a), while RAG may lead to errors
from retrieving irrelevant or misleading documents (Lewis et al., 2021). There are also ties
to cognition: people typically internalize abstract reasoning skills while relying on external
memory (e.g., books, search engines, notes) for factual recall (Wegner, 1987; Risko & Gilbert,
2016; Sparrow et al., 2011; Norman, 1988; Clark & Chalmers, 1998). While we do not directly
study these mechanisms, it motivates viewing parametric and non-parametric knowledge
as complementary resources, raising the question of how to allocate data between them.
Code and data:https://github.com/DegenAI-Labs/RAG-scaling-laws
Correspondence to karanps@stanford.edu, vgtomahawk@gmail.com, and syfeng@stanford.edu.
1arXiv:2604.00715v1  [cs.CL]  1 Apr 2026

Preprint. Under review.
Figure 1:Trade-off between pretraining and retrieval under a fixed data budget.Left:We
train OLMo-2 models ranging from 30M to 3B parameters on DCLM data while constructing
retrieval stores from held-out portions of the same corpus.Center:We conceptualize this as
an optimization problem over a 2D allocation space of pretraining and retrieval tokens. For
a fixed data budget, feasible configurations lie along a constraint frontier, and performance
varies smoothly; our goal is to identify the optimal allocation along this frontier.Right:
Retrieval allocation trade-off at fixed pretraining scale. As the % of data used for retrieval
increases, performance changes non-monotonically, with scale dependence: smaller models
benefit most, while larger models exhibit diminishing returns and over-allocation sensitivity.
We therefore ask: given a fixed corpus of Ntokens, what is the optimal allocation between
pretraining data and retrieval store? This is a resource allocation problem with no established
answer. While several prior works incorporate retrieval, none systematically vary how much
data is allocated to weights versus retrieval during pretraining. To our knowledge, this is
the first study to treat them as competing recipients of the same data budget, enabling a true
scaling-law analysis of knowledge placement as the model learns fundamental capabilities.
Pretraining builds up parametric knowledge but incurs substantial training cost, while
retrieval is effectively free during training but depends on retrieval quality and introduces
inference-time overhead. Understanding how to optimally trade off these two mechanisms
is essential for designing efficient and scalable LM systems. We study this empirically across
model scales ranging from 30M to 3B parameters, systematically varying both the amount of
pretraining data and the size of the retrieval datastore constructed from the same underlying
corpus. We evaluate across a diverse set of benchmarks spanning multiple domains and
knowledge types. In summary, we make the following contributions:
•We show that the relationship between pretraining and retrieval is structured but non-
trivial, with retrieval yielding scale- and regime-dependent, non-monotonic effects.
•To characterize this interplay, we model performance as a function of both pretraining
tokens and retrieval tokens, revealing an approximate scaling law over this two-
dimensional allocation space and enabling quantification of their substitutability.
•We identify a scale-dependent crossover point beyond which retrieval becomes an
efficient substitute for pretraining.
Taken together, our findings establish a unified scaling perspective on parametric and non-
parametric knowledge, and provide practical guidance for RAG-aware training. Rather
than treating pretraining and retrieval as separate design choices, we show they can be
jointly optimized under a fixed data budget, enabling efficient use of large-scale corpora.
2 Related Works
2.1 Scaling Laws for Pretraining
Many works study how LM performance scales with model size, dataset size, and compute.
Kaplan et al. (2020) established predictable power-law relationships between these factors.
2

Preprint. Under review.
Chinchilla later showed that compute-optimal training requires jointly scaling model and
data size (Hoffmann et al., 2022). Gadre et al. (2024) show that scaling laws remain predictive
in overtrained regimes and relate pretraining loss to downstream task performance, while
other work incorporates data mixture (Ye et al., 2025; Shukor et al., 2025) and domain-specific
continual pretraining (Que et al., 2024). These works suggest that pretraining efficiency
depends not only on parameters and tokens, but on the data shown and in what proportions.
Related work studies scaling when inference cost matters. Sardana et al. (2024) show that
smaller models trained longer can be preferable when inference demand is high. Bian et al.
(2025) incorporate architecture-aware latency into scaling analysis, showing that parameter
count alone is an incomplete proxy for deployment efficiency. This motivates treating
pretraining, model size, data quality, and serving cost as coupled optimization problems.
2.2 Retrieval-Augmented Language Models
Retrieval-augmented LMs address a key limitation of purely parametric LMs: knowledge
is stored implicitly in weights, making updates expensive and provenance difficult to
trace. REALM was among the first to integrate retrieval directly into pretraining by jointly
learning a dense retriever with a masked-LM objective (Guu et al., 2020). RAG popularized
retrieval-augmented generation for knowledge-intensive tasks by conditioning a generator
on retrieved Wikipedia passages (Lewis et al., 2021), while kNN-LM showed that nearest-
neighbor lookup can improve perplexity and domain adaptation (Khandelwal et al., 2020).
Subsequent work scaled this to larger corpora and general-purpose LMs. RETRO demon-
strated that large external datastores can match much larger parametric models (Borgeaud
et al., 2022), and Atlas showed strong few-shot performance with retrieval-augmented
models that can be updated independently of the generator (Izacard et al., 2022). Recent
surveys frame these systems as retrieval-augmented LMs and emphasize trade-offs among
retriever quality, memory freshness, grounding, and system complexity (Hu & Lu, 2025).
While much of the RAG literature focuses on improving downstream factuality at test time,
systems such as REALM, RETRO, and Atlas suggest that retrieval can alter the pretraining
trade-off itself by offloading some knowledge from parameters into external memory.
2.3 Small Language Models: Pretraining & Evaluation
Recent work on small language models (SLMs) has emphasized that strong performance
under tight parameter budgets depends heavily on architecture, data quality, and training
duration. TinyLlama showed that a 1.1B model trained on ∼1T tokens can substantially
outperform earlier open models of similar size (Zhang et al., 2024). SmolLM2 showed that a
1.7B model overtrained on a careful mixture of web, math, code, and instruction data can
outperform several recent baselines (Allal et al., 2025). On the extremely small data scale, the
BabyLM challenge (Warstadt et al., 2023; Hu et al., 2024a) investigates SLM training using
fixed budgets of 10M and 100M tokens. This has led to studies about models’ inductive
biases (Kallini et al., 2024) and systematic asymmetries (Hu et al., 2025), among others.
Evaluation methodology is especially important here since SLMs are more sensitive. HELM
argued for a multi-metric, scenario-based view of LM evaluation (Liang et al., 2023), and
DataComp-LM paired standardized pretraining recipes with broad downstream evaluation
(Li et al., 2025). A recurring lesson is that higher-quality or more targeted data can partially
substitute for scale: Gunasekar et al. (2023); Penedo et al. (2024); Allal et al. (2025) show
that specialized data can substantially strengthen compact models. Some works train SLMs
with concept, knowledge, and visual information augmentation (somewhat analogous to
retrieval) as additional guidance to improve generative commonsense reasoning (Lin et al.,
2020; Feng et al., 2021; 2023; 2022). This all demonstrates that retrieval is especially relevant
for SLMs, where external information may help compensate for limited parametric capacity.
2.4 Data-Efficient Pretraining
Other works study how to extract more performance from a fixed training budget. Lee
et al. (2022) showed that common pretraining corpora contain substantial duplication,
3

Preprint. Under review.
and that deduplication can reduce memorization while improving performance. Penedo
et al. (2024) showed that education-focused subsets of data can substantially improve
reasoning and knowledge-heavy evaluations. Other works focus on selecting or mixing
data more intelligently. Xie et al. (2023) showed that domain reweighting with a small
proxy model can substantially improve pretraining efficiency, and Ye et al. (2025) enabled
mixture optimization from small-scale experiments. Related work also explores optimal
data mixing for two-stage pretraining (Feng et al., 2024a; Liu et al., 2026b), and the optimal
mixing of code and target-domain data during pretraining (Ma et al., 2023; Baek et al., 2026).
In terms of data ordering, Feng et al. (2024b) study curriculum design with increasing age
of child-directed speech data, and Singh et al. (2026) pair data curriculum with progressive
model scaling. Overall, this literature suggests that data efficiency depends not only on
what data is used, but also on when and how it is presented during training.
Summary and Motivation.Prior work has extensively studied scaling laws for pretraining
and, separately, the benefits of retrieval-augmented LMs, but these directions have largely
been explored in isolation. We bridge this gap by studying how pretraining and retrieval
interact under fixed compute and model-size constraints, with a particular focus on how to
allocate data between parametric learning and external memory across different scales.
3 Methods
3.1 Experimental Setup
For our experiments, we use the OLMo-2 series (OLMo et al., 2024) of LMs due to its strong
empirical performance, alignment with open research practices, and modern architectural
design. We define our own OLMo-2 model sizes and pretrained them across various scales:
30M, 136M, 233M, 728M, 1B, and 3B parameters (hyperparameter details in Appendix A.1).
We use 100B tokens of DCLM data as our pretraining corpus (Li et al., 2025). We train all
models using AdamW with a 3 ×10−4peak learning rate ( lr),β1=0.9,β2=0.95, and 0.1
weight decay. We adopt a warmup-stable-decay (WSD) schedule (Hu et al., 2024b) with 10%
linear warmup (capped at 2k steps), a stable phase, and 10% linear decay to a minimumlr
of 6e−5. Models are evaluated every 2k steps and at the end of training.
3.2 Index Construction
We now describe how we build our embedding store, i.e., a collection of vector represen-
tations over which retrieval is performed. We construct retrieval indices across multiple
scales (1B–20B tokens) via FAISS (Douze et al., 2025) from a held-out slice of DCLM by first
computing per-chunk token counts over the embedding store, and then selecting chunks
via a seeded random permutation. For each target budget (e.g., 30M, 60M, etc.), we take
the shortest prefix of that permutation whose cumulative token count meets or slightly
exceeds the target, then materialize the corresponding chunk texts/metadata and build a
FAISS index over the selected embeddings. Because all budgets are prefixes of the same
permutation (for fixed source data, filtering config, and seed), smaller-budget indices are
strict subsets of larger-budget indices (e.g., 30 M⊂ 60M), enabling controlled scaling com-
parisons where corpus size is the primary varying factor. For our index construction, we
chose Qwen3-Embedding-8B from amongst 4 candidate choices on the basis of recall, and
IVPFQ as the indexing algorithm. More index construction details are in Appendix A.2.
3.3 Evaluation Protocol
We evaluate all models using a retrieval-augmented variant of EleutherAI’slm-evaluation-
harness(Gao et al., 2024), theRAG-Evaluation-Harnessframework (Shao et al., 2024), across
multiple benchmarks spanning reasoning, scientific QA, and open-domain QA: AI2-ARC
(Easy and Challenge) (Clark et al., 2018), HellaSwag (Zellers et al., 2019), OpenBookQA (Mi-
haylov et al., 2018), SciQ (scientific QA) (Welbl et al., 2017), Natural Questions (Kwiatkowski
et al., 2019), StrategyQA (Geva et al., 2021), SimpleQA (Wei et al., 2024), PIQA (Bisk et al.,
2020), and CommonsenseQA (Talmor et al., 2019).
4

Preprint. Under review.
RAG evaluation setup.We retrieve the top- kpassages ( k= 5, chosen via a small pilot
sweep as a trade-off between retrieval quality and context budget) from a fixed FAISS
index. Retrieved passages are concatenated as context, followed by the original question
and answer choices (if applicable). The retriever is frozen and shared across all evaluations
to isolate the effect of retrieval scale and query formulation.
Metrics.We evaluate two metrics: accuracy ( acc) and perplexity. Acc is computed by
selecting the answer choice with the highest total log-likelihood and comparing it to the
ground truth, yielding a binary per-example score averaged over the dataset. While this
is the most common metric, it is insufficient for scaling analysis. As models improve, acc
often exhibits thresholded or step-like behavior: small improvements in likelihood may not
change the predicted label, leading to flat regions followed by sudden jumps. This obscures
the underlying scaling trends and makes it difficult to fit smooth functional relationships.
To address this, we use perplexity ( PPL ) as our primary metric. PPL provides a continuous,
length-normalized measure of model performance. We compute the average log-likelihood
per token of the gold answer continuation and report exp(−mean log-likelihood) across
examples. For RAG, this corresponds to how well the model predicts the correct answer
conditioned on both the retrieved context and the task prompt. As noted by Tay et al. (2021),
the‘transfer gap’between pretraining objective and downstream task suggests that PPL is
a more granular indicator of model ability than discrete success metrics, e.g., acc, which
have an emergent nature (Krajewski et al., 2025). Unlike acc,PPL captures incremental
improvements in model confidence and yields smooth trends across model and data scales,
making it suited for fitting scaling laws and analyzing pretraining-retrieval trade-offs.
4 Experimental Results
4.1 Parametric Scaling Baselines
We begin by establishing parametric scaling baselines in the absence of retrieval (retrieval
index sizeR=0), varying model size (N) and pretraining data (D). This serves as a sanity
check that our experimental setup reproduces the standard scaling-law behavior observed
in prior work such as Hoffmann et al. (2022). Following Hoffmann et al. (2022), we model
loss for parametric models as a function of model size and data using a power-law form:
L(N,D) =AN
109−α
+BD
109−β
+L 0 (1)
where (A,α)capture scaling with model size, (B,β)capture scaling with data, and L0is an
irreducible loss floor. Here, (A,α)govern the model-size contribution, (B,β)govern the
data contribution, and L0is the asymptotic loss floor. Intuitively, larger αimplies stronger
sensitivity to model scaling, while larger βimplies stronger sensitivity to data scaling.
Across benchmarks, as seen in Figure 2 and Table 1, we observe smooth and predictable
improvements as either NorDincreases, with diminishing returns in both directions as
expected from power-law scaling. The fitted model achieves low average relative error
ARE= 1/nn
∑
i=1|(Lpred
i− Lobs
i)/Lobs
i| ×100%, and scaling exponents broadly align with
previously reported values in scaling literature (Hoffmann et al., 2022). Overall, these
baseline fits validate that our setup reliably reproduces canonical scaling-law behavior.
4.2 Scaling Laws for Retrieval
To model retrieval-augmented scaling, we extend the 2D parametric law with an additional
retrieval axis using a logarithmic gain term:
L(N,D,R) =AN
109−α
+BD
109−β
−Clog
1+ηR
109
+L 0 (2)
where Nis model size (parameters), Dis pretraining tokens, and Ris retrieval/index tokens.
Here, (A,α)and (B,β)govern parametric scaling with model size and data, while (C,η)
5

Preprint. Under review.
Benchmark CV ARE (%) LOMO ARE (%)α βL 0ReasoningPIQA 42.55 129.04 0.3786 1.3033 1.0920
CommonsenseQA 6.87 10.76 0.1558 0.0869 2.1619
HellaSwag 2.18 4.94 0.3923 0.4713 1.3714
StrategyQA 12.28 14.93 2.0000 0.4647 7.6735Scientific/Open QASciQ 13.07 24.03 0.5267 0.2606 0.9522
OpenBookQA 4.78 9.41 0.3688 0.2120 1.6579
AI2C-ARC Easy 7.10 13.67 0.3566 0.2195 1.0299
AI2C-ARC Challenge 3.93 6.09 0.2811 0.2817 1.1133
Natural Questions 7.19 17.39 0.3595 0.3523 1.2593
SimpleQA 11.46 21.09 0.2285 0.9580 1.7196
Table 1: Power-law fit quality and scaling exponents for parametric baselines ( R= 0). We
report cross-validation average relative error (CV ARE) [interpolation error under random
splits] and leave-one-model-size-out ARE (LOMO ARE) [extrapolation error to unseen
model size]. Lower ARE indicate better fit quality. Exponents αandβ, when combined with
L0, summarize how loss scales with model size and pretraining data in the baseline regime.
ARE should be interpreted relative to the inherent noise and discreteness of benchmarks,
with smoother, likelihood-based tasks yielding low errors, and reasoning-heavy tasks (e.g.,
PIQA) showing higher variance and ARE.
Figure 2: Parametric scaling baselines without RAG ( R= 0).Left:Empirical measurements
across model sizes and data budgets, overlaid with iso-loss contours from the power-law
model. Each point corresponds to a trained model configuration, colored by observed
perplexity. The blue line denotes the compute-efficient frontier and the vertical dashed lines,
discrete training budgets.Right:Iso-compute slices of the scaling surface, showing predicted
loss as a function of model size (N). Empirical observations are overlaid for reference.
govern retrieval gain and saturation. Because retrieval enters as a subtractive gain term,
larger Cincreases the maximum retrieval benefit, while larger ηincreases how quickly gains
are realized asRgrows.L 0is the asymptotic loss floor.
Empirically, the log-form retrieval law provides strong fits on most benchmarks — results
using a power retrieval law are in Appendix A.4. In Table 2, CV ARE (Cross-Validation
Average Relative Error) is low for many tasks, while LOMO (Leave-One-Model-Out) errors
are generally higher, indicating that interpolation is easier than extrapolation to held-out
model scales. Reasoning-heavy tasks remain less stable (especially PIQA and StrategyQA),
with larger held-out errors (additional measures reported in Appendix A.3). The fitted
retrieval-rate parameter ηshows two broad regimes. For some tasks, ηis moderate ( ≈10−3
to≈2), indicating gradual retrieval gains. For others, ηreaches the optimization ceiling
(near 10 in our current constrained fit), suggesting rapid saturation over the observed
retrieval range, or limited identifiability of retrieval dynamics from available points. These
6

Preprint. Under review.
Benchmark CV ARE (%) LOMO (%)α β ηL 0ReasoningPIQA 40.10 39.82 0.3786 1.3033 0.0001 1.0920
CommonsenseQA 5.74 7.42 0.1558 0.0869 0.3432 2.1619
HellaSwag 2.11 2.43 0.3923 0.4713 0.0007 1.3714
StrategyQA 15.80 15.26 2.0000 0.4647 10.0000 7.6735Scientific/Open QASciQ 9.96 12.39 0.5267 0.2606 0.9008 0.9522
OpenBookQA 4.29 5.10 0.3688 0.2120 1.8620 1.6579
AI2-ARC Easy 4.95 6.68 0.3566 0.2195 10.0000 1.0299
AI2-ARC Challenge 3.44 4.19 0.2811 0.2817 10.0000 1.1133
Natural Questions 5.67 7.10 0.3595 0.3523 10.0000 1.2593
SimpleQA 11.73 13.24 0.2285 0.9580 9.9999 1.7196
Table 2: Cross-validated fit quality and scaling exponents for 3D power-law fits incor-
porating a retrieval axis. We report cross-validation average relative error (CV ARE),
leave-one-model-out average relative error (LOMO), and the fitted exponents governing
model size (α), pretraining data (β), and retrieval (η), and the irreducible loss floorL 0.
results are stable across multiple training seeds (see Appendix A.7). Overall, retrieval
improves performance with diminishing returns, and both the magnitude and saturation
rate of those gains are strongly task-dependent.
4.3 Pretraining–Retrieval Trade-off Curves
We now investigate the trade-off between pretraining data ( D) and retrieval ( R), with the
goal of understanding how retrieval can substitute for pretraining in reducing loss. Figure 3
summarizes this trade-off across model scales. We analyze two complementary perspectives:
(i) thesubstitutabilityof retrieval for pretraining, and (ii) themarginal benefitof retrieval.
Substitutability of retrieval.For each model and pretraining scale, we fit scaling laws and
compute the amount of retrieval required to match the performance of a baseline model
trained without retrieval. We express retrieval in units of equivalent pretraining tokens
as follows. For a configuration (N,D,Ropt)with measured loss L∗
RAG, we project this loss
onto theN,Dscaling curve (Eq. 2) to find the equivalent pretraining budget, giving us the
projection DRAG
eff. From this, we compute thesubstitutability σwhich represents the number
of pretraining tokens saved per retrieval token:
DRAG
eff=L∗
RAG− L 0−A·N−α
B−1/β
⇒σ=DRAG
eff−D
Ropt(3)
We observe a clear crossover behavior in Figure 3 (left). In low-data regimes, retrieval
cannot effectively replace pretraining. However, it becomes increasingly effective beyond
a threshold of ∼D/N= 4.14 Pretraining Tokens Per Parameter across all model scales
(estimated using the line-of-best-fit), with each retrieval token replacing multiple pretraining
ones. In this regime, the gains grow ∼log-linearly, indicating that retrieval serves as an
efficient alternative to additional pretraining. Importantly, this reflectsrelative efficiency
rather than absolute improvement: even when retrieval substitutes efficiently for pretraining,
the total achievable gain may be small if the baseline model is already near saturation.
Marginal benefit of retrievalis defined as the reduction in loss per unit of retrieval data
κ=∆L/(R/ 109), where ∆L=L R=0− L∗
RAG (higher is better). Figure 3 (right) shows this
quantity for models trained near their optimal pretraining ratio. We find that smaller models
(e.g., 30M) benefit most from retrieval, achieving large improvements per unit of retrieved
data. As model size increases, the marginal benefit decreases, with gains diminishing
substantially at larger scales and largely saturating by 3B parameters. This suggests that
while retrieval may remain an efficient substitute for pretraining at larger scales (left), the
absolute improvement it provides diminishes as models become increasingly saturated.
Summary.There clearly exists a scale-dependent trade-off between pretraining and retrieval.
Retrieval is most valuable (a strong substitute for pretraining) in undertrained and smaller-
7

Preprint. Under review.
Figure 3:Trade-off between pretraining and retrieval under a fixed data budget.Left:We
quantify the substitutability between retrieval and pretraining via the number of pretraining
tokens saved per retrieval token, computed by fitting scaling laws and determining, for each
pretraining scale, the amount of retrieval required to match baseline performance without
retrieval. The dotted line represents a linear line-of-best-fit across all model scales.Right:
We measure the marginal benefit of retrieval as perplexity improvement per billion retrieval
tokens (higher is better) for models trained near their optimal pretraining ratio.
model regimes. As model size and pretraining increase, its marginal utility decreases,
indicating a transition from retrieval-dominated to pretraining-dominated regimes.
4.4 RAG Improvements
While our primary focus is on the allocation trade-off between pretraining and retrieval, this
raises a complementary question: how much of the observed benefit depends on retrieval
quality? Qualitatively, retrieved contexts on factoid QA often capture the correct topic
but do not consistently contain directly answer-bearing evidence (e.g., specific entities or
dates), suggesting that retrieval precision remains a limiting factor (see Appendix A.8 for
further qualitative discussion). To probe this further, we evaluate a simple strategy for
improving retrieval: varying retriever query formulation. We compare, with a fixed corpus
index across methods, (i) question-only queries, (ii) queries augmented with answer choices
(when applicable), and (iii) queries that include the gold answer (an oracle ablation).1
Figure 4 shows RAG improvement results on SimpleQA, with additional benchmarks in
Appendix A.9. Retrieval yields modest gains on knowledge-heavy tasks (SimpleQA, Com-
monsenseQA), particularly when queries better align with the answer, with improvements
increasing at larger model scales. In contrast, reasoning-heavy tasks (GSM8K (Cobbe et al.,
2021), LAMBADA (Paperno et al., 2016)2), show minimal change. Across tasks, improved
retrieval yields incremental gains but does not alter the scaling trends observed earlier. This
reinforces prior takeaways that retrieval is not a uniform substitute for pretraining, and that
its effectiveness depends on both model scale and task type.
5 Discussion
Our results suggest that retrieval and pretraining should not be viewed as independent
design choices, but as two competing mechanisms for allocating a fixed data budget. Pre-
training stores knowledge parametrically in model weights, while retrieval stores it non-
parametrically in an external index. By studying both jointly, we find that their interaction is
structured and can be captured by simple scaling laws over model size, pretraining data, and
retrieval store size. Retrieval appears most useful in regimes where parametric knowledge is
1This is closer to an approximate upper bound by more closely approximating an optimal scenario
where one would have a (near)perfectretriever to maximize the potential benefits of RAG.
2We try RAG improvements on these two additional benchmarks (math and word prediction) that
performed poorly with retrieval on our initial pilot studies.
8

Preprint. Under review.
Figure 4:Effect of retrieval query formulation on performance.Comparison of standard
generation on SimpleQA without retrieval (Baseline) to RAG under two query formulations:
(i)RAG (Query), which retrieves top- kpassages using only the question, and (ii)RAG
(Query + Gold), which includes the gold answer in the query too (an oracle-style ablation).
SimpleQA is not multiple-choice (no answer choices), so we do not reportRAG (Query +
Choices)here. All methods use a shared corpus index constructed from 20% of the data,
retrieving the top-5 passages per query.Left:OLMo-2 136M.Right:OLMo-2 1B.
still limited. In smaller or less-saturated models, it can substitute for additional pretraining
and yield substantial reductions in loss. This benefit is not uniform: it depends on both
model scale and task type, and exhibits diminishing returns as models become larger or
more heavily pretrained. This suggests that retrieval is not merely an additive improvement
on pretraining, but a scale-dependent alternative for where knowledge is stored.
Our findings also clarify the role of retrieval quality. Improvements from better query formu-
lation and oracle-style retrieval indicate that some of the observed trade-off is bottlenecked
by the retriever rather than the language model alone. However, even stronger retrieval
does not uniformly eliminate the need for parametric capacity, especially on reasoning-
heavy tasks where the limiting factor appears to be computation over knowledge rather
than access to it. In this sense, retrieval is most naturally interpreted as a complement
to parametric learning rather than a universal substitute for it. More broadly, this work
suggests a shift in how pretraining corpora should be conceptualized. Rather than assuming
that all available data should be compressed into weights, future language model design
may benefit from explicitly partitioning corpora into data intended for internalization versus
external access. This perspective aligns naturally with practical system design, where model
capacity, training cost, memory footprint, and inference latency are all coupled.
Limitations.There are several factors in our present study that could be expanded. First, the
retrieval setup is intentionally simple and fixed: we use a single retriever, a fixed chunking
strategy, and a fixed top- kprotocol. Although this isolates the effect of retrieval scale,
it likely understates the gains achievable with stronger retrieval pipelines, as we briefly
investigate. Second, our evaluation focuses primarily on perplexity as the most stable metric
for scaling analysis; while appropriate for fitting smooth laws, this does not fully capture all
downstream behaviors of interest. Third, although we study a broad range of model sizes,
our conclusions are still limited to the scales, architectures, and corpora explored here.
Future Work.A natural extension is to study how stronger retrieval systems, e.g., reranking,
learned filtering, adaptive chunking, or LLM-based relevance estimation, shift the optimal
pretraining-retrieval allocation. Another direction is to develop a more principled way to
unify scaling behavior across benchmarks. Here, we fit benchmark-specific scaling laws,
but a broader goal is to identify shared latent structure that explains why some tasks benefit
more from retrieval than others. This could involve characterizing benchmarks by their
degree of knowledge dependence, retrieval sensitivity, or reasoning burden, and using these
properties to build more general scaling laws over model size, pretraining, and external
memory. Finally, inspired by human cognition, future research could explore the purposeful
allocation of abstract reasoning to pretraining vs. long-tail factual knowledge to retrieval.
9

Preprint. Under review.
Acknowledgments
We gratefully acknowledge the National Science Foundation ACCESS Program and Modal
for providing compute resources that enabled this work. EL was supported by the National
Sciences and Engineering Research Council of Canada (NSERC), [funding reference number
578085], as well as the SoftBank-ARM Fellowship.
References
Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Mart ´ın Bl ´azquez, Guilherme
Penedo, Lewis Tunstall, Andr ´es Marafioti, Hynek Kydl ´ıˇcek, Agust ´ın Piqueres Lajar ´ın,
Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Cl ´ementine Four-
rier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin
Raffel, Leandro von Werra, and Thomas Wolf. SmolLM2: When Smol Goes Big – Data-
Centric Training of a Small Language Model. 2 2025.
Christina Baek, Ricardo Pio Monti, David Schwab, Amro Abbas, Rishabh Adiga, Cody
Blakeney, Maximilian B ¨other, Paul Burstein, Aldo Gael Carranza, Alvin Deng, Parth
Doshi, Vineeth Dorna, Alex Fang, Tony Jiang, Siddharth Joshi, Brett W. Larsen, Jason Chan
Lee, Katherine L. Mentzer, Luke Merrick, Haakon Mongstad, Fan Pan, Anshuman Suri,
Darren Teh, Jason Telanoff, Jack Urbanek, Zhengping Wang, Josh Wills, Haoli Yin, Aditi
Raghunathan, J. Zico Kolter, Bogdan Gaza, Ari Morcos, Matthew Leavitt, and Pratyush
Maini. The Finetuner’s Fallacy: When to Pretrain with Your Finetuning Data. 3 2026.
Song Bian, Minghao Yan, and Shivaram Venkataraman. Scaling Inference-Efficient Lan-
guage Models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix
Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.),Proceedings of the 42nd
International Conference on Machine Learning, volume 267 ofProceedings of Machine Learn-
ing Research, pp. 4303–4323. PMLR, 3 2025. URL https://proceedings.mlr.press/v267/
bian25b.html.
Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning
about Physical Commonsense in Natural Language.Proceedings of the AAAI Conference on
Artificial Intelligence, 34(05):7432–7439, 4 2020. ISSN 2374-3468. doi: 10.1609/aaai.v34i05.
6239.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie
Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark,
Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron
Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini,
Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich
Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of
tokens. 2 2022.
Andy Clark and David Chalmers. The extended mind.Analysis, 58(1):7–19, 1998.
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,
and Oyvind Tafjord. Think you have Solved Question Answering? Try ARC, the AI2
Reasoning Challenge. 3 2018.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz
Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher
Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL
https://arxiv.org/abs/2110.14168.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-
Emmanuel Mazar ´e, Maria Lomeli, Lucas Hosseini, and Herv ´e J´egou. The faiss library.
IEEE Transactions on Big Data, 2025.
10

Preprint. Under review.
Steven Feng, Shrimai Prabhumoye, Kezhi Kong, Dan Su, Mostofa Patwary, Mohammad
Shoeybi, and Bryan Catanzaro. Maximize Your Data’s Potential: Enhancing LLM Accuracy
with Two-Phase Pretraining. 12 2024a.
Steven Y. Feng, Jessica Huynh, Chaitanya Prasad Narisetty, Eduard Hovy, and Varun
Gangal. SAPPHIRE: Approaches for enhanced concept-to-text generation. In Anya
Belz, Angela Fan, Ehud Reiter, and Yaji Sripada (eds.),Proceedings of the 14th International
Conference on Natural Language Generation, pp. 212–225, Aberdeen, Scotland, UK, August
2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.inlg-1.21. URL
https://aclanthology.org/2021.inlg-1.21/.
Steven Y. Feng, Kevin Lu, Zhuofu Tao, Malihe Alikhani, Teruko Mitamura, Eduard
Hovy, and Varun Gangal. Retrieve, caption, generate: Visual grounding for enhanc-
ing commonsense in text generation models.Proceedings of the AAAI Conference on Ar-
tificial Intelligence, 36(10):10618–10626, Jun. 2022. doi: 10.1609/aaai.v36i10.21306. URL
https://ojs.aaai.org/index.php/AAAI/article/view/21306.
Steven Y. Feng, Vivek Khetan, Bogdan Sacaleanu, Anatole Gershman, and Eduard Hovy.
CHARD: Clinical health-aware reasoning across dimensions for text generation models.
In Andreas Vlachos and Isabelle Augenstein (eds.),Proceedings of the 17th Conference of the
European Chapter of the Association for Computational Linguistics, pp. 313–327, Dubrovnik,
Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.
eacl-main.24. URLhttps://aclanthology.org/2023.eacl-main.24/.
Steven Y. Feng, Noah D. Goodman, and Michael C. Frank. Is Child-Directed Speech Effective
Training Data for Language Models? 10 2024b.
Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell
Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, Rui Xin, Marianna
Nezhurina, Igor Vasiljevic, Jenia Jitsev, Luca Soldaini, Alexandros G. Dimakis, Gabriel
Ilharco, Pang Wei Koh, Shuran Song, Thomas Kollar, Yair Carmon, Achal Dave, Reinhard
Heckel, Niklas Muennighoff, and Ludwig Schmidt. Language models scale reliably with
over-training and on downstream tasks. 6 2024.
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles
Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell,
Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf,
Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and
Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/
records/12608602.
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did
Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning
Strategies. 1 2021.
Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio C ´esar Teodoro Mendes, Allie Del Giorno,
Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi,
Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, S ´ebastien Bubeck, Ronen Eldan,
Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks Are All You Need. 10 2023.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM:
Retrieval-Augmented Language Model Pre-Training. 2 2020.
Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan
Kianinejad, Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep Learning Scaling
is Predictable, Empirically. 12 2017.
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai,
Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark,
Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc,
Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals,
and Laurent Sifre. Training Compute-Optimal Large Language Models. 3 2022.
11

Preprint. Under review.
Jennifer Hu, Alvin Wei Ming Tan, Steven Y. Feng, and Michael C. Frank. Language
production is harder than comprehension for children and language models. InPro-
ceedings of the Annual Meeting of the Cognitive Science Society, volume 47, 2025. URL
https://escholarship.org/uc/item/5rz8b9jg.
Michael Y Hu, Aaron Mueller, Candace Ross, Adina Williams, Tal Linzen, Chengxu Zhuang,
Ryan Cotterell, Leshem Choshen, Alex Warstadt, and Ethan Gotlieb Wilcox. Findings of
the second babylm challenge: Sample-efficient pretraining on developmentally plausible
corpora. InThe 2nd BabyLM Challenge at the 28th Conference on Computational Natural
Language Learning, pp. 1–21, 2024a.
Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei
Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small
language models with scalable training strategies.arXiv preprint arXiv:2404.06395, 2024b.
Yucheng Hu and Yuxing Lu. RAG and RAU: A Survey on Retrieval-Augmented Language
Model in Natural Language Processing. 6 2025.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick,
Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot
Learning with Retrieval Augmented Language Models. 11 2022.
Julie Kallini, Isabel Papadimitriou, Richard Futrell, Kyle Mahowald, and Christopher Potts.
Mission: Impossible language models.arXiv preprint arXiv:2401.06416, 2024.
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon
Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling Laws for Neural
Language Models. 1 2020.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Gener-
alization through Memorization: Nearest Neighbor Language Models. 2 2020.
Jakub Krajewski, Amitis Shidani, Dan Busbridge, Sam Wiseman, and Jason Ramapuram.
Revisiting the scaling properties of downstream metrics in large language model training.
arXiv preprint arXiv:2512.08894, 2025.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh,
Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina
Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszko-
reit, Quoc Le, and Slav Petrov. Natural Questions: A Benchmark for Question Answering
Research.Transactions of the Association for Computational Linguistics, 7:453–466, 11 2019.
ISSN 2307-387X. doi: 10.1162/tacl{\ }a{\}00276.
Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris
Callison-Burch, and Nicholas Carlini. Deduplicating Training Data Makes Language
Models Better. 3 2022.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
4 2021.
Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal,
Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff,
Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman,
Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh,
Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt,
Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu
Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay
Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby,
Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini,
Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal
Dave, Ludwig Schmidt, and Vaishaal Shankar. DataComp-LM: In search of the next
generation of training sets for language models. 4 2025.
12

Preprint. Under review.
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Ya-
sunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman,
Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning,
Christopher R ´e, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus,
Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam,
Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha,
Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael
Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang,
Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta
Koreeda. Holistic Evaluation of Language Models. 10 2023.
Bill Yuchen Lin, Wangchunshu Zhou, Ming Shen, Pei Zhou, Chandra Bhagavatula, Yejin
Choi, and Xiang Ren. CommonGen: A constrained text generation challenge for gen-
erative commonsense reasoning. In Trevor Cohn, Yulan He, and Yang Liu (eds.),Find-
ings of the Association for Computational Linguistics: EMNLP 2020, pp. 1823–1840, Online,
November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.
findings-emnlp.165. URLhttps://aclanthology.org/2020.findings-emnlp.165/.
Emmy Liu, Varun Gangal, Chelsea Zou, Michael Yu, Xiaoqi Huang, Alex Chang, Zhuofu Tao,
Karan Singh, Sachin Kumar, and Steven Y. Feng. A Unified Definition of Hallucination:
It’s The World Model, Stupid! 2 2026a.
Emmy Liu, Graham Neubig, and Chenyan Xiong. Midtraining bridges pretraining and
posttraining distributions, 2026b. URLhttps://arxiv.org/abs/2510.14865.
Yingwei Ma, Yue Liu, Yue Yu, Yuanliang Zhang, Yu Jiang, Changjian Wang, and Shanshan
Li. At Which Training Stage Does Code Data Help LLMs Reasoning? 9 2023.
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a Suit of Armor
Conduct Electricity? A New Dataset for Open Book Question Answering. 9 2018.
Donald A Norman.The Psychology of Everyday Things. Basic Books, New York, NY, 1988.
ISBN 978-0465067091.
Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita
Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 olmo 2 furious.arXiv preprint
arXiv:2501.00656, 2024.
OpenAI. tiktoken, 2022. URL https://github.com/openai/tiktoken . Fast BPE tokenizer
for use with OpenAI’s models.
Denis Paperno, Germ ´an Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella
Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fern ´andez. The
LAMBADA dataset: Word prediction requiring a broad discourse context. In Katrin
Erk and Noah A. Smith (eds.),Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, Berlin, Germany,
August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1144.
URLhttps://aclanthology.org/P16-1144/.
Guilherme Penedo, Hynek Kydl ´ıˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell,
Colin Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb Datasets: Decanting
the Web for the Finest Text Data at Scale. 10 2024.
Haoran Que, Jiaheng Liu, Ge Zhang, Chenchen Zhang, Xingwei Qu, Yinghao Ma, Feiyu
Duan, Zhiqi Bai, Jiakai Wang, Yuanxing Zhang, Xu Tan, Jie Fu, Wenbo Su, Jiamang Wang,
Lin Qu, and Bo Zheng. D-CPT Law: Domain-specific Continual Pre-Training Scaling Law
for Large Language Models. 6 2024.
Evan F. Risko and Sam J. Gilbert. Cognitive offloading.Trends in Cognitive Sciences, 20(9):
676–688, 2016. doi: 10.1016/j.tics.2016.07.002.
13

Preprint. Under review.
Nikhil Sardana, Jacob Portes, Sasha Doubov, and Jonathan Frankle. Beyond Chinchilla-
Optimal: Accounting for Inference in Language Model Scaling Laws. In Ruslan Salakhut-
dinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and
Felix Berkenkamp (eds.),Proceedings of the 41st International Conference on Machine Learning,
volume 235 ofProceedings of Machine Learning Research, pp. 43445–43460. PMLR, 3 2024.
URLhttps://proceedings.mlr.press/v235/sardana24a.html.
Rulin Shao, Jacqueline He, Akari Asai, Weijia Shi, Tim Dettmers, Sewon Min, Luke Zettle-
moyer, and Pang Wei Koh. Scaling Retrieval-Based Language Models with a Trillion-Token
Datastore. 7 2024.
Mustafa Shukor, Louis Bethune, Dan Busbridge, David Grangier, Enrico Fini, Alaaeldin
El-Nouby, and Pierre Ablin. Scaling Laws for Optimal Data Mixtures. 10 2025.
Karanpartap Singh, Neil Band, and Ehsan Adeli. Curriculum-Guided Layer Scaling for
Language Model Pretraining. 2 2026.
Betsy Sparrow, Jenny Liu, and Daniel M. Wegner. Google effects on memory: Cognitive
consequences of having information at our fingertips.Science, 333(6043):776–778, 2011.
doi: 10.1126/science.1207745.
Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A
Question Answering Challenge Targeting Commonsense Knowledge. 3 2019.
Yi Tay, Mostafa Dehghani, Jinfeng Rao, William Fedus, Samira Abnar, Hyung Won Chung,
Sharan Narang, Dani Yogatama, Ashish Vaswani, and Donald Metzler. Scale efficiently:
Insights from pre-training and fine-tuning transformers.arXiv preprint arXiv:2109.10686,
2021.
Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro,
Rafael Mosquera, Bhargavi Paranjabe, Adina Williams, Tal Linzen, et al. Findings of the
babylm challenge: Sample-efficient pretraining on developmentally plausible corpora. In
Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language
Learning, 2023.
Daniel M. Wegner. Transactive memory: A contemporary analysis of the group mind. In
B. Mullen and G. R. Goethals (eds.),Theories of group behavior, pp. 185–208. Springer, New
York, NY, 1987. doi: 10.1007/978-1-4612-4634-3 9.
Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia
Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large
language models. 11 2024.
Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing Multiple Choice Science
Questions. InProceedings of the 3rd Workshop on Noisy User-generated Text, pp. 94–106,
Stroudsburg, PA, USA, 2017. Association for Computational Linguistics. doi: 10.18653/
v1/W17-4413.
Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang,
Quoc V . Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing Data Mixtures Speeds
Up Language Model Pretraining. 11 2023.
Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, and Xipeng Qiu. Data
Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.
3 2025.
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can
a Machine Really Finish Your Sentence? InProceedings of the 57th Annual Meeting of
the Association for Computational Linguistics, pp. 4791–4800, Stroudsburg, PA, USA, 2019.
Association for Computational Linguistics. doi: 10.18653/v1/P19-1472.
Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. TinyLlama: An Open-Source
Small Language Model. 6 2024.
14

Preprint. Under review.
A Appendix
A.1 Pretraining setup
All pretraining runs use NVIDIA H100 GPUs with 8 devices per job. We train with FSDP
data-parallelism, in mixed precision, and without model parallelism. We use varied per-
device micro-batch sizes depending on the model scale, and gradient accumulation to
achieve an effective global batch size of 256 across runs. Training uses context length (block
size) 4096 across all models. Additional hyperparameter details for each model scale is
provided in Table 3.
Model Size Layers Hidden Dim Number of Heads Query Groups Intermediate Size Block Size
30M 8 256 4 5 512 4096
136M 8 512 8 8 2048 4096
233M 16 640 10 10 2560 4096
728M 18 1280 10 10 5120 4096
1B 24 1408 11 11 5632 4096
3B 26 2560 20 20 10240 4096
Table 3: Pretraining hyperparameters across model sizes
A.2 Index Construction
A.2.1 Retrieval Corpus
We construct a retrieval corpus from a held-out split of the DCLM dataset, chunked into
overlapping token windows and embedded using a pretrained embedding model. All
embeddings are L2-normalized and indexed using FAISS with product quantization. We
vary the retrieval corpus size across multiple scales and construct separate indices for each
setting. Each split corresponds to an increasing retrieval corpus size derived from disjoint
subsets of the DCLM corpus. Index construction-relevant details are in Tables 4, 5, and 6.
A.2.2 Embedding Choice, Chunking & Tokenization
We explored several embedding models including BAAI/bge-base-en-v1.5,
google/embeddinggemma-300m, and Qwen3-Embedding-8B. We chose Qwen3-
Embedding-8B as it showed strongest semantic recall and was consistently at the top of
public RAG benchmarks. We used IVFPQ for our indexing algorithm, a chunk length of 900
tokens, and a stride length of 256 tokens. We built our chunks using TikToken cl100k-base
(OpenAI, 2022), then decoded back to text.
Component Configuration
Embedding model Qwen3-Embedding-8B
Embedding dimension 4096
Pooling Last non-padded token
Normalization L2-normalized
Tokenizer cl100k base
Chunk length 900 tokens
Stride 256 tokens (∼28% overlap)
Table 4: Embedding and chunking configuration for retrieval corpus construction.
A.3 Additional Cross-Validation Fit Quality Numbers: LODO andR2
As an auxiliary addendum to Table 2 and to further establish the predictive reliability of the
3D scaling law fits, we also show the Leave-One-Dataset-Out (LODO) ARE, as well as the
R2values for both LODO and LOMO in Table 7. We note the following broad observations
based on these additional numbers.
15

Preprint. Under review.
Component Configuration
Index type IVFPQ (inner product)
Sub-quantizers 128
Bits per code 8
Search parameter nprobe = 64
Training set size max(nlist×50, 105)
Batch size (add) 500K vectors
Table 5: FAISS index configuration.
Corpus Split nlist Shards
5% 32,768 32
10% 32,768 64
20% 65,536 128
30% 65,536 128
Table 6: Per-split FAISS index configuration.
Across the majority of benchmarks, LOMO and LODO errors are of comparable magnitude
(typically within 1–2% of each other). This suggests that the 3D power-law scaling surface
generalizes effectively across both unseen model architectures and unseen task distributions
within the same family.
A.4 3D Power Fits
As an alternative to the logarithmic retrieval formulation used in the main text, we also
consider a power-law parameterization for retrieval-augmented scaling:
L(N,D,R) =AN
109−α
+BD
109−β
+C
1+R
109−γ
+L 0. (4)
Here, Ndenotes model size (parameters), Dthe number of pretraining tokens, and Rthe
size of the retrieval corpus. The parameters (A,α)and (B,β)govern scaling with model
size and pretraining data as in the baseline setting, while (C,γ)capture the magnitude and
rate of retrieval gains. Specifically, largerCcorresponds to a larger potential improvement
from retrieval, while larger γimplies faster saturation as Rincreases. L0represents the
irreducible loss floor.
Table 8 reports fit quality and estimated exponents under this formulation. Consistent
with the logarithmic model, we observe low cross-validation error (CV ARE) across many
benchmarks, indicating that the joint (N,D,R)scaling surface is well-approximated within
the observed regime. Leave-one-model-out (LOMO) errors are generally higher, reflecting
the increased difficulty of extrapolating across model scales.
The fitted retrieval exponent γexhibits substantially more variability than the corresponding
logarithmic parameter ηin the main text (Table 2). While several tasks show moderate
values of γ(e.g.,∼0.3–1), indicating gradual improvements with increasing retrieval,
others reach very large values (e.g., γ≈ 10, the upper-bound we set), corresponding
to effectively immediate saturation over the observed retrieval range. In contrast, the
logarithmic formulation yields more stable and interpretable retrieval-rate parameters, with
ηtypically falling into a narrower range and more clearly capturing gradual gain regimes.
Despite these differences in parameter behavior, both formulations achieve similar fit quality
and recover consistent qualitative trends: retrieval provides diminishing returns, and both
the magnitude and rate of these returns vary significantly across tasks. However, the greater
stability and interpretability of the logarithmic parameterization motivate its use in the main
analysis, while the power-law results serve as a complementary validation of the robustness
of the observed scaling behavior.
16

Preprint. Under review.
Benchmark LOMO ARE (%) LODO ARE (%) LOMOR2LODOR2
ReasoningCommonsenseQA 7.42 5.77 0.5748 0.6692
HellaSwag 2.43 2.95 0.9847 0.9696
Scientific/Open QASciQ 12.39 10.42 0.8230 0.8670
OpenBookQA 5.10 4.85 0.9262 0.9240
AI2-ARC Easy 6.68 5.30 0.9121 0.9383
AI2-ARC Challenge 4.19 4.07 0.9612 0.9569
Table 7: Refined scaling performance metrics for 3D power-law fits. We report the Leave-
One-Model-Out (LOMO) and Leave-One-Dataset-Out (LODO) average relative errors (ARE)
alongside their respective coefficients of determination (R2).
Benchmark CV ARE (%) LOMO (%)α β γL 0ReasoningPIQA 42.88 46.24 0.3786 1.3033 0.4640 1.0920
CommonsenseQA 6.30 8.35 0.1558 0.0869 0.9119 2.1619
HellaSwag 2.10 2.41 0.3923 0.4713 0.4648 1.3714
StrategyQA 20.29 20.24 2.0000 0.4647 9.9999 7.6735Scientific/Open QASciQ 12.73 18.12 0.5267 0.2606 4.6703 0.9522
OpenBookQA 4.02 4.74 0.3688 0.2120 0.3050 1.6579
AI2-ARC Easy 6.16 8.91 0.3566 0.2195 6.3804 1.0299
AI2-ARC Challenge 3.70 4.57 0.2811 0.2817 10.0000 1.1133
Natural Questions 6.39 8.08 0.3595 0.3523 6.4292 1.2593
SimpleQA 12.20 13.64 0.2285 0.9580 10.0000 1.7196
Table 8: Cross-validated fit quality and scaling exponents for 3D power-law fits incorporat-
ing a retrieval axis. We report cross-validation average relative error (CV ARE), leave-one-
model-out average relative error (LOMO), and the fitted exponents governing model size
(α), pretraining data ( β), and retrieval ( γ), along with the irreducible loss floor L0. Compared
to the logarithmic formulation in the main text, the power-law parameterization exhibits a
wider spread in retrieval exponents, with some tasks showing gradual scaling ( γ≈ 0.3–1)
and others exhibiting rapid saturation ( γ≫ 1). Fit quality remains comparable across most
benchmarks, though extrapolation error (LOMO) is generally higher, particularly on noisier
reasoning tasks.
A.5 Some Further Notes On Retrieval Efficiency Metrics (σandκ)
To complement the scaling-law analysis in §4.3, we introduce two derived metrics that
quantify the efficiency of retrieval relative to pretraining:replacement cost( σ) andmarginal
benefit( κ). These metrics provide an interpretable view of the trade-off between parametric
and non-parametric knowledge. We explain them further here.
Multiplicative nature of σ.Replacement cost σis a ratio that measures how many pre-
training tokens are replaced per retrieval token. As such, it is inherently multiplicative: a
change fromσ=1 toσ=10 represents a tenfold increase in efficiency.
Consistent with this interpretation, σspans multiple orders of magnitude across tasks (e.g.,
from<1 to>103), and follows approximately log-linear trends with respect to pretraining
scale. Therefore, we aggregateσusing the geometric mean:
σGM=exp 
1
nn
∑
i=1lnσ i!
(5)
This preserves multiplicative structure, reduces sensitivity to extreme values, and aligns
with the power-law scaling behavior underlying our analysis.
17

Preprint. Under review.
Additive nature of κ.In contrast, the marginal benefit κmeasures an absolute reduction in
loss per unit of retrieval data. This is an additive quantity: improvements combine linearly
and may be positive, zero, or negative depending on the task.
Because κcan take non-positive values and does not exhibit multiplicative structure, geo-
metric aggregation is not appropriate. Instead, we summarizeκusing the median:
κmed =median({κ 1, . . . ,κ n})(6)
which provides a robust estimate of the typical improvement across tasks.
Pretraining Regimes.We report results across three regimes defined by the token-to-
parameter ratio:
• 1×:D/N≈1 (undertrained)
• 10×:D/N≈10 (near-optimal)
• 100×:D/N≈100 (overtrained)
A.5.1 Benchmark-Level Results
Benchmark Meanσ σ@1×σ@10×σ@100×
AI2C-ARC Challenge 4.73 0.06 9.14 11.20
AI2C-ARC Easy 22.54 0.38 27.83 64.40
CommonsenseQA 677.51 677.93 1216.95 1951.00
NQ-Open 5.28 0.23 2.72 13.63
OpenBookQA 6.65 0.32 6.05 -10.49
SciQ 24.21 0.20 11.69 78.62
StrategyQA 95.59 7.89 0.95 105.40
Table 9: Replacement costσacross benchmarks and training regimes.
Benchmark Meanκ κ@1×κ@10×κ@100×
AI2C-ARC Challenge 0.4778 0.1582 2.4505 0.4216
AI2C-ARC Easy 0.3701 0.1048 0.4348 0.5736
CommonsenseQA 0.5738 0.2715 1.4521 1.0438
NQ-Open 0.4421 0.4118 0.5427 0.3192
OpenBookQA 2.2450 0.2014 3.8579 0.4100
SciQ 0.9830 0.3722 0.2576 0.4918
StrategyQA 3.5006 0.0420 0.3923 2.1839
Table 10: Marginal benefitκacross benchmarks and training regimes.
Observations:
•Retrieval substitutability increases with saturation: σgrows with D, indicating that
retrieval becomes more effective once pretraining enters diminishing returns.
•Diminishing marginal returns: κdecreases with model size, consistent with larger
models internalizing more knowledge parametrically.
•Strong task dependence:knowledge-intensive tasks (e.g., CommonsenseQA) exhibit
very high σ, while reasoning-heavy tasks (e.g., HellaSwag, PIQA) show weak or
negative gains.
These results provide a quantitative interpretation of the trade-off curves in Figure 3,
reinforcing the view that retrieval acts as a scale-dependent substitute for pretraining.
A.6 Calibration Plots
Calibration plots (Figure 5) allow us to visualize how our predicted loss (from the 3D scaling
law fit) compares to the actual loss values.
18

Preprint. Under review.
(a) ARC Challenge
 (b) ARC Easy
(c) HellaSwag
 (d) Science Questions (SciQ)
(e) CommonsenseQA
 (f) OpenBookQA
Figure 5:Calibration plots for 3D scaling law fits across benchmarks.We show the
alignment between predicted and observed PPL forN,D, and Racross six benchmarks. The
tight grouping around the diagonal indicates that the usual Hoffman power-law formulation
with a log term for retrieval effectively captures the retrieval-augmented scaling behavior.
A.7 Stability Analysis
To assess the robustness of our scaling-law fits, we evaluate stability across multiple random
seeds and model initializations. We consider three random seeds each for three model
families (30M, 136M, 233M), yielding 27 total runs. For each config, we fit scaling laws
independently and compute cross-validation average relative error (CV ARE) and leave-one-
model-out ARE (LOMO ARE). Figure 6 reports the mean and std of these metrics across runs
for each benchmark. Overall, we observe low variance in both CV ARE and LOMO ARE
across most tasks, indicating that the fitted scaling relationships are stable w.r.t. initialization
and data ordering. Reasoning-heavy tasks such as PIQA and StrategyQA exhibit higher
variance and larger abs errors, suggesting that their scaling behavior is noisier and less well
captured by simple parametric forms. In contrast, more knowledge-driven benchmarks
(e.g., ARC, OpenBookQA) show consistently low variance and strong fit quality across runs.
19

Preprint. Under review.
Figure 6:Stability of scaling-law fits across random seeds.We report mean ±standard
deviation of cross-validation ARE (CV ARE) and leave-one-model-out ARE (LOMO ARE)
across 27 separate fits (every possible combination of 3 model families×3 seeds each).
A.8 Qualitative Analysis of Retrieval Behavior
To better understand when retrieval helps, we manually inspected retrieved contexts for
a couple of datasets where RAG helps (SimpleQA, NQ-Open) and one where it does not
(GSM8K). The key contrast is that retrieval is often useful astopical groundingfor SimpleQA
and NQ-Open, but is much less useful for GSM8K, where problems are typically self-
contained and external text is often distractive.
For SimpleQA, query-only contexts often match the right domain (e.g., history, philoso-
phy, astronomy, biology) and sometimes surface source-adjacent material that can support
answer extraction. For example, legal-history questions aboutA Survey of Londonretrieve
medieval court/tower records; philosophy questions about Hegel retrieve prose that dis-
cusses his views on Romantic art and its main art forms; and technical biomedical questions
retrieve mutation/gene-focused research text rather than generic web chatter.
For NQ-Open, RAG helps mostly when retrieval injects a single concrete anchor
(year/number/entity). For example,“when was the first Australian prime minister elected”
changes from baseline1977to correct1901with RAG, and“what age do you need to be to
buy a BB gun”shifts from5to18 years oldafter retrieval includes age-threshold text (“over
the age of 18. . . over 21 for handguns”). Likewise,“when was the last time anyone was on the
moon”moves from“200 years ago”to1972once Apollo timeline snippets appear. Sometimes
the same mechanism still misses the target: retrieval changes the model’s answer, but the
new answer remains incorrect. Overall, the benefit is best characterized as occasional fact
anchoring from salient cues, rather than consistently reliable evidence grounding.
For GSM8K, query-only retrieval is usually not needed and often noisy. Retrieved passages
are commonly worksheets, forum posts, product listings, dictionary pages, or malformed
index fragments (e.g.,“files ...”), which rarely contribute to the arithmetic decomposition
required by the question. Occasionally, retrieval provides a useful conversion fact (e.g.,
gallon-to-pints), but most examples are either redundant with what is already stated in the
prompt or off-task. This qualitative pattern aligns with the quantitative result that RAG
yields little benefit on GSM8K in this setup, and with existing conclusions that RAG helps
more heavily with long-tail factual knowledge than things like mathematics.
20

Preprint. Under review.
Figure 7:Effect of retrieval query formulation across GSM8K and CommonsenseQA.
We compare standard generation (Baseline) to retrieval-augmented setups under different
query constructions: question-only (RAG Query), query augmented with the gold answer
(RAG Query + Gold / Answer), and additionally for CommonsenseQA: query with answer
choices (RAG Query + Choices) and both choices and gold answer (RAG Query + Choices +
Answer). Both panels show OLMo-2 1B as a function of pretraining tokens per parameter.
Figure 8:Effect of retrieval query formulation on LAMBADA.Similar to Figure 7 above.
Left:OLMo-2 136M.Right:OLMo-2 1B.
A.9 RAG Improvements
We provide results analyzing the effect of retrieval query formulation across additional
benchmarks. We compare standard generation without retrieval (Baseline) to retrieval-
augmented setups using different query constructions. Specifically, we consider: (i)RAG
(Query), which retrieves using only the task question; (ii)RAG (Query + Gold / Answer), which
augments the query with the gold answer (oracle-style ablation); and for multiple-choice
settings, (iii)RAG (Query + Choices)when multiple-choice references are available and (iv)
RAG (Query + Choices + Answer). All experiments use a shared retrieval setup with a fixed
FAISS index constructed from a held-out corpus with size equivalent to 20% of the max
pretraining tokens, and top- k=5 retrieved passages prepended to the prompt. We sweep
pretraining tokens per parameter while keeping the retrieval configuration fixed. Figures 7
and 8 show results on GSM8K, CommonsenseQA, and LAMBADA across model scales.
B LLM Usage Disclosure
We used LLMs (e.g., GPT-5) to assist with parts of the coding process and limited aspects
of paper preparation, including LaTeX table formatting and minor editing for clarity and
grammar. All outputs were carefully reviewed to ensure accuracy and appropriateness.
21