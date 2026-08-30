# The RAT: A Unified Bayesian Model for RAG Evaluation

**Authors**: Pius von Däniken, Felix Matthias Saaro, Mark Cieliebak, Jan Deriu

**Published**: 2026-08-25 15:54:30

**PDF URL**: [https://arxiv.org/pdf/2608.24753v1](https://arxiv.org/pdf/2608.24753v1)

## Abstract
Evaluating Retrieval-Augmented Generation (RAG) systems requires assessing not only end-to-end correctness but also how individual components interact and how errors propagate through the pipeline. We introduce a Bayesian evaluation framework that jointly models retrieval success, abstention behavior, and answer correctness, factorized according to the pipeline's information flow. The model distinguishes task success. Whether the user received a correct answer (from generator success) and whether the generator behaved appropriately given the retrieval outcome. We apply the framework to 27 RAG configurations across three datasets, three retrievers, and three generators, and show that the conditional decomposition reveals substantial behavioral differences between systems that appear equivalent under marginal metrics. We further analyze the annotation allocation problem, demonstrating that retrieval-success annotations are more informative than task-success annotations for estimating policy adherence, and provide an information-theoretic explanation for this asymmetry. Finally, we extend the model to incorporate LLM-as-a-judge annotations as calibrated noisy observations, enabling practitioners to combine limited human judgments with cheaper automated assessments within a unified probabilistic model.

## Full Text


<!-- PDF content starts -->

The RAT: A Unified Bayesian Model for RAG Evaluation
Pius von Däniken*Felix Matthias Saaro*
Mark Cieliebak Jan Milan Deriu
Centre for Artificial Intelligence
ZHAW School of Engineering
{vode,saaf,ciel,deri}@zhaw.ch
Abstract
Evaluating Retrieval-Augmented Generation
(RAG) systems requires assessing not only
end-to-end correctness but also how individual
components interact and how errors propagate
through the pipeline. We introduce a Bayesian
evaluation framework that jointly models re-
trieval success, abstention behavior, and an-
swer correctness, factorized according to the
pipeline’s information flow. The model dis-
tinguishestask success. whether the user re-
ceived a correct answer, fromgenerator suc-
cess, whether the generator behaved appropri-
ately given the retrieval outcome. We apply the
framework to 27 RAG configurations across
three datasets, three retrievers, and three gen-
erators, and show that the conditional decom-
position reveals substantial behavioral differ-
ences between systems that appear equivalent
under marginal metrics. We further analyze
the annotation allocation problem, demonstrat-
ing that retrieval-success annotations are more
informative than task-success annotations for
estimating policy adherence, and provide an
information-theoretic explanation for this asym-
metry. Finally, we extend the model to in-
corporate LLM-as-a-judge annotations as cali-
brated noisy observations, enabling practition-
ers to combine limited human judgments with
cheaper automated assessments within a uni-
fied probabilistic model.
1 The Introduction
Evaluating Retrieval-Augmented Generation
(RAG) systems is challenging and often tedious.
It is not sufficient to evaluate only the system’s
end-to-end behavior; each component must be
evaluated both in isolation and within the pipeline.
Since RAG systems are built in a pipelined
approach, errors tend to propagate through the
pipeline. If retrieval fails, the generator has
no basis for producing a correct answer and
*Equal contribution
Figure 1: The dependency structure of the evaluation
variables: generator success depends on retrieval suc-
cess (which governs whether the generator should ab-
stain or answer), abstention behavior, and task success.
should abstain from answering the question.
Most benchmarks developed for RAG evaluation
consider each component in isolation (Es et al.,
2024; Saad-Falcon et al., 2024; Rau et al., 2024),
or evaluate generators using adversarial retrieval
results (Wang et al., 2024). However, a holistic
view of the evaluation that accounts for the
dependencies between components is still missing.
A key shortcoming of end-to-end evaluation is
that it conflates two distinct notions of success.
Task successmeasures whether the user received
a correct answer, whereasgenerator successmea-
sures whether the generator behaved appropriately
given the retrieval outcome, i.e., answering when
retrieval succeeded and abstaining when it failed.
These two quantities can diverge substantially: a
generator that never abstains may achieve reason-
able task success while systematically violating
the desired policy, and two systems with identical
task success can exhibit very different behaviors
under retrieval failure. To make such distinctions
explicit, we need a model that captures the depen-
dencies between retrieval, abstention, and answer
correctness.
arXiv:2608.24753v1  [cs.CL]  25 Aug 2026

We propose a Bayesian evaluation framework
that models these dependencies explicitly. Figure 1
illustrates the dependency structure, which shows
how these variables interact: retrieval success gov-
erns both the abstention decision and the condi-
tions under which task success is meaningful, and
generator success is a deterministic function of all
three. Casting this as a Bayesian model allows us to
propagate uncertainty through the full dependency
structure and to handle partially observed data by
marginalizing over unobserved variables, which is
a practical advantage when some annotations are
expensive while others are cheap. The framework
is further extended by incorporating LLM-as-a-
judge annotations alongside human gold-standard
labels, using a calibration model that treats auto-
mated judgments as noisy observations of the un-
derlying ground-truth variables (von Däniken et al.,
2022, 2024). This allows practitioners to combine
small amounts of expensive human annotation with
larger volumes of automated judgments in a princi-
pled way. We make the following contributions1:
•We introduce a Bayesian evaluation model for
RAG systems that factorizes the joint distribution
over retrieval success, abstention, and task suc-
cess according to the pipeline’s information flow,
and derives generator success as a deterministic
variable over these random variables (Section 3).
•We apply the model to 27 RAG configurations
(3 retrievers ×3 generators ×3 datasets) and
show that the conditional decomposition reveals
behavioral differences that marginal metrics con-
ceal, in particular, systems with near-identical
task success can differ sharply in policy adher-
ence.
•We analyze the annotation allocation problem:
given a fixed budget, we derive and empirically
validate which combination of retrieval and task-
success annotations minimizes estimation error,
and provide an information-theoretic explanation
for the observed asymmetry.
•We extend the model to incorporate calibrated
LLM-as-a-judge annotations as noisy observa-
tions, enabling practitioners to combine human
and automated judgments within the same proba-
bilistic framework.
1Our code can be found at: https://github.com/
vodezhaw/rat.2 The Related Work
Retrieval-Augmented Generation (RAG) is the pro-
cess of incorporating the output of an information
retrieval (IR) system into the context of a large
language model (LLM), so that the LLM’s an-
swer is informed by relevant documents or pas-
sages (Lewis et al., 2020; Borgeaud et al., 2022).
The process is composed of two components: re-
trieval and generation. The user’s question is trans-
formed into a query for an IR engine, which re-
trieves a set of relevant documents or passages,
which are then given to the LLM to generate an
answer to the user’s query. This setting, while
straightforward to describe, is highly challenging
to evaluate. In general, the two components are
evaluated separately. For an extensive overview of
RAG evaluation, we refer the reader to (Yu et al.,
2025).
Retrieval Evaluation.There are two main settings:
either there is a gold standard available, that is, for
a set of queries, all relevant documents, or there is
no gold standard available. In the case of an exist-
ing gold standard, which is highly costly to create,
one applies the standard scores used in the IR liter-
ature (Tang and Yang, 2024), such as Mean Recip-
rocal Rank, or Mean Average Precision (Schütze
et al., 2008). In cases where no gold standard is
available, two approaches are commonly used: ei-
ther silver-standard generation or an unreferenced
LLM-as-a-judge (Zheng et al., 2023). The gen-
eration of silver standards consists of an inverted
process by providing an LLM with a document
or passage, and letting the LLM generate ques-
tions (Es et al., 2024). Then the provided document
serves as the relevant document (while disregarding
other potentially relevant documents). For LLM-
as-a-judge, one provides the LLM with the result
of the retrieval and the user question, and asks to
rate whether the question can be answered with the
provided retrieval (Saad-Falcon et al., 2024).
Generator Evaluation.The evaluation of the gen-
erator often entangles two concepts: the end-to-end
success, and the evaluation of the generator un-
der different retrieval settings. For instance, Wang
et al. (2024) evaluates the behavior of the generator
under different adversarial retrieval settings. Chen
et al. (2024) also highlights the importance of evalu-
ating the impact of the retrieval on the generation in
isolation. They also state the need to handle cases
where the retrieval does not find relevant sources in
the generator by allowing abstention from answer-

ing, and, in turn, evaluating the LLM’s behavior in
these cases.
Pipeline Evaluation.Rau et al. (2024) introduces a
retrieval benchmarking library that focuses on eval-
uating pipeline components. ARES (Saad-Falcon
et al., 2024) introduces a LLM-as-a-judge-based
evaluation framework to evaluate each component
of the pipeline separately using various dimensions.
RAGAs (Es et al., 2024) introduces a reference-free
framework for evaluating multiple dimensions of
RAG pipelines without relying on ground-truth hu-
man annotations, covering context relevance, faith-
fulness, and answer relevance. Both RAGAs and
ARES follow the RAG Triad framing popularized
byTruLens(TruLens, 2024). CRUD-RAG (Lyu
et al., 2025) is a benchmark oriented towards real-
world scenarios rather than the QA tasks popular
in academic settings. They include evaluation of
multiple components, such as chunk size selection
and reranking. While these frameworks evaluate
multiple dimensions, they treat each dimension in-
dependently and do not model the statistical depen-
dencies between retrieval and generation outcomes.
Our Workbuilds on these efforts and integrates
them into a single probabilistic framework that
jointly models the dependencies between retrieval,
abstention, and answer correctness. This unified
view enables distinguishing policy adherence from
end-task success, propagating uncertainty across
the pipeline, reasoning about annotation allocation,
and combining human and automated judgments
via calibration.
3 The Model
Evaluating a RAG system only by whether its fi-
nal output matches a reference answer does not
tell the full story. End-task correctness quantifies
the system’s overall output, but it does not distin-
guish among the generator’s underlying behaviors.
In particular, the common intuition that better re-
trieval should lead to better answers is only valid
if the generator can use the retrieved information
appropriately. Likewise, abstention can be desir-
able when retrieval fails, even though it may reduce
overall answer rates and, in turn, affect aggregate
performance metrics.
The Variables.To make these distinctions explicit,
we model RAG behavior using four binary vari-
ables. Let Rdenote retrieval success, where R= 1
means that the retrieved context contains all in-
formation required to answer the question. Let Adenote abstention, where A= 1 means that the
generator declines to answer by producing an ex-
plicit "I don’t know." response. Let Tdenote task
success, where T= 1 means that the final answer
is correct. Finally, let Gdenote generator success,
or policy adherence, where G= 1 if the generator
behaves as desired. We use binary variables as a
deliberate simplification.
The Conditionals.Studying only the marginals
of these variables is insufficient to characterize the
system’s behavior. For example, a system with
slightly higher task success may still be less desir-
able if it achieves that gain by answering questions
without appropriate support, rather than abstain-
ing appropriately. To understand such trade-offs,
we study conditional probabilities that separate re-
trieval quality, abstention behavior, and answer cor-
rectness.
We therefore propose a factorization of the joint
distribution that follows the flow of information
through the RAG pipeline:
P(R, A, T) =P(R)P(A|R)P(T|A, R).
In our data (see Section 4), task success is only
possible for non-abstained responses, so effectively
T= 0 whenever A= 1 . This yields conditional
probabilities with direct behavioral interpretations:
P(R) captures retrieval quality, P(A|R) cap-
tures how abstention depends on retrieval success
or failure, and P(T|A= 0, R) captures answer
correctness conditional on both the retrieval state
and the decision to answer.
The Generator Success.Finally, we define gener-
ator success Gdeterministically from (R, A, T) to
encode the desired policy: G= 
(R= 0)∧(A=
1)
∨ 
(R= 1)∧(A= 0)∧(T= 1)
.That is,
the generator is successful if it abstains when re-
trieval fails, or if it answers correctly when retrieval
succeeds. This construction makes it possible to
distinguish correct behavior from correct output
and to analyze how different RAG configurations
trade off retrieval, abstention, and answer quality.
The Unified Model.We cast this model in a
Bayesian framework. This provides two practi-
cal advantages for our setting. First, it propagates
uncertainty throughout the full model, allowing
us to quantify uncertainty not only for the primi-
tive conditional probabilities but also for derived
quantities such as policy adherence. Second, it
naturally accommodates partially observed data by
marginalizing over unobserved variables, which

is particularly useful when some annotations are
expensive while others are cheap.
Under the factorization above, the model is pa-
rameterized by five probabilities: θR=P(R= 1) ,
θA−=P(A= 1|R= 0) ,θA+=P(A= 1|
R= 1) ,θT−=P(T= 1|R= 0, A= 0) ,
andθT+=P(T= 1|R= 1, A= 0) .
These correspond directly to interpretable aspects
of RAG behavior: retrieval success, abstention
on retrieval failure, abstention despite success-
ful retrieval, unsupported-answer success, and
supported-answer success.
The Automated Judge Extension.To add the
LLM-as-a-judge to the model, we extend the factor-
ization by an additional term to represent the binary
RJandTJvariables: P(R, A, T, R J, TJ) =
P(R)P(A|R)P(T|R, A)P(R J, TJ|R, A, T) .
This adds 18 degrees of freedom to the model,
corresponding to the 6 feasible (R, A, T) states
and a 4-category conditional distribution for
each state. The resulting judgment model is
more complex than simpler factorizations such
asP(R J|R)P(T J|T), but avoids imposing
conditional independence assumptions that were
not supported by preliminary analyses.
We place independent uniform priors on all five
basic model parameters and a Dirichlet prior on the
conditional probability of automated judge obser-
vations. The resulting posterior can then be queried
for both the primitive parameters and any determin-
istic function of them, including model-implied
marginals and the derived policy-adherence quan-
tity. We implement the model in Stan (Stan Devel-
opment Team, 2026) and perform posterior infer-
ence using Hamiltonian Monte Carlo with the No-
U-Turn Sampler (NUTS) (Hoffman and Gelman,
2014). For each experiment, we run five chains
with 2,000 warmup iterations and collect 10,000
posterior samples per chain.
4 The Retrievers, The Generators, and
The Tasks
The Data.This work employs the KILT bench-
mark (Petroni et al., 2021), a unified library for
knowledge intensive languag tasks comprising 11
datasets across 5 task categories, including fact-
checking and open-domain question answering.
All datasets are grounded in a shared, pre-processed
Wikipedia snapshot, thereby ensuring consistent
evaluation conditions and facilitating interoperabil-
ity across tasks with minimal preprocessing over-head.
•FEVER (FEV)(Thorne et al., 2018) is a fact-
checking dataset consisting of approximately
126,000 human-annotated claims, each as-
signed one of three labels:Supported,Refuted,
orNotEnoughInfo. The majority of queries
have 1 relevant paragraph.
•HotpotQA (HQA)(Yang et al., 2018) is an
open-domain question answering dataset com-
prising approximately 113,000 questions that
necessitate multi-hop reasoning across multi-
ple Wikipedia articles. The question-answer
pairs were collected using crowd workers who
were shown pairs of Wikipedia paragraphs
and asked to write multi-hop questions along
with their answer. All queries have 2 relevant
paragraphs.
•Natural Questions (NQ)(Kwiatkowski et al.,
2019) consists of approximately 92,000 nat-
urally occurring queries submitted to the
Google search engine, each paired with a long
and short answer annotation curated by crowd
workers. In this work, only the short answer
annotations are used. All queries have exactly
1 relevant paragraph.
For each task, a subset of 10,000 queries was ran-
domly selected. A single shared knowledge base
of 1,720,160 paragraphs was subsequently con-
structed from all documents relevant to the selected
queries across all tasks2.
The Retrieval Strategies.Three strategies were
evaluated:
•Sparse (S), a term-based retrieval approach em-
ploying the BM25 ranking function based on
lexical matching.
•Dense (D), semantic retrieval approach leverag-
ing multi-task text embeddings (Sturua et al.,
2024), indexed using a Hierarchical Navigable
Small World (HNSW) graph via the FAISS li-
brary (Douze et al., 2024) for efficient approxi-
mate nearest-neighbour search.
•Hybrid (H), a combination of sparse and dense
retrieval, with result fusion performed using Re-
ciprocal Rank Fusion (RRF) (Cormack et al.,
2009).
2This knowledge base represents a more controlled re-
trieval environment than a fully open-domain corpus. Con-
sequently, the absolute retrieval-success rates (see Section 5)
may be interpreted as optimistic relative to fully open-domain
deployment.

DS Gen. Ret.P(R=1)P(A=1)P(T=1)P(G=1)
FEVApt
D 0.6030.135 0.753 0.608
Gem 0.180 0.784 0.736
Qwn 0.199 0.7510.744
Apt
H0.6550.0890.801 0.614
Gem 0.0960.8640.701
Qwn 0.112 0.838 0.703
Apt
S 0.4560.124 0.7590.504
Gem 0.221 0.744 0.644
Qwn0.2460.7080.657
HQAApt
D 0.1510.038 0.2600.110
Gem 0.3110.2530.391
Qwn0.4890.2600.574
Apt
H 0.2070.0320.304 0.138
Gem 0.218 0.315 0.337
Qwn 0.3750.3290.503
Apt
S0.2180.033 0.318 0.152
Gem 0.226 0.318 0.360
Qwn 0.391 0.326 0.535
NQApt
D 0.3470.028 0.239 0.164
Gem 0.256 0.242 0.414
Qwn 0.355 0.243 0.496
Apt
H0.3480.0240.2670.163
Gem 0.198 0.267 0.359
Qwn 0.307 0.266 0.451
Apt
S 0.2440.039 0.2330.136
Gem 0.3000.2200.412
Qwn0.4340.2150.531
Table 1: Marginal probabilities across all 27 RAG con-
figurations. P(R=1) is shared across generators within
each dataset–retriever pair. Highest per-dataset values
inbold, lowest initalics.
The Generators.Three large language models
were employed as generators:
•Apertus 8B(Hernández-Cano et al., 2025),
a fully open-source, decoder-only transformer
model with 8 billion parameters, developed with
an emphasis on full transparency of both model
weights and training data.
•Gemma3 12B(Team et al., 2025), a lightweight
open multimodal model developed by Google,
comprising 12 billion parameters and trained on
12 trillion tokens spanning more than 140 lan-
guages.
•Qwen3.5 9B(Qwen Team, 2026), a multimodal
model developed by Alibaba Cloud with 9 bil-
lion parameters, supporting multilingual infer-
ence across 201 languages. Although the model
includes a dedicated reasoning mode, this func-
tionality was not employed in the present work5 The Experiments
5.1 The Marginals
Table 1 reports the marginal probabilities across all
27 configurations. We count retrieval as successful
only whenallrelevant documents are present in the
retrieved context3. Hybrid retrieval achieves the
highest retrieval success onFEVandNQ, whereas
sparse retrieval performs best onHQA. Retrieval
success is substantially higher onFEVthan on the
other two datasets because each sample has exactly
one relevant document, making the retrieval task
considerably easier. Turning to the generator-level
metrics,Apertusexhibits consistently low absten-
tion rates across all datasets, with a maximum of
0.135 onFEVwith dense retrieval. In contrast,
Gemma3andQwen3.5abstain substantially more
often, reaching rates as high as 0.489 forQwen3.5
onHQAwith dense retrieval.
Task success is highest onFEV, where each ques-
tion has only two answer options, and lower on
HQAandNQ, which require an exact match to
an open-form reference answer. Across datasets,
the three generation models achieve broadly sim-
ilar task success rates. Policy adherence, how-
ever, differs much more strongly across models:
Qwen3.5achieves the highest policy adherence on
all datasets, whereasApertusconsistently lags be-
hind.
5.2 The Conditionals
We fit the model introduced in Section 3 separately
to the data from each configuration. The model-
implied marginals closely match the correspond-
ing count-based marginals up to Monte Carlo er-
ror. Table 2 therefore focuses on the conditional
probabilities, which provide a more interpretable
decomposition of system behavior.
A first pattern is thatGemma3andQwen3.5dis-
tinguish between retrieval success and failure much
more clearly thanApertus. Across datasets, both
models assign substantially higher abstention prob-
abilities when retrieval fails than when retrieval
succeeds. For example, onHQAwith hybrid re-
trieval,Apertushas P(A=1|R=0) = 0.039
andP(A=1|R=1) = 0.005 , whereasGemma3
reaches 0.270 and0.021 , andQwen3.5 0.463 and
0.036 . Thus, under the same retrieval conditions,
the models exhibit markedly different abstention
policies. At the same time, abstention conditional
3This binary definition is a modeling simplification (see
Section 3). We discuss partial retrieval in Appendix C.

DS Gen. Ret.θA− θA+ θT− θT+
FEVApt
D0.2450.0630.8100.903
Gem 0.422 0.021 0.941 0.962
Qwn0.4620.027 0.894 0.954
Apt
H0.1580.053 0.8310.901
Gem 0.242 0.019 0.940 0.961
Qwn 0.274 0.027 0.913 0.955
Apt
S0.197 0.038 0.828 0.906
Gem 0.3910.0180.942 0.963
Qwn 0.428 0.028 0.916 0.956
HQAApt
D0.044 0.0100.229 0.492
Gem 0.361 0.026 0.313 0.570
Qwn0.568 0.044 0.4580.638
Apt
H0.039 0.0050.258 0.519
Gem 0.270 0.021 0.331 0.608
Qwn 0.463 0.036 0.456 0.678
Apt
S0.041 0.006 0.264 0.552
Gem 0.286 0.011 0.326 0.632
Qwn 0.493 0.028 0.4460.704
NQApt
D0.039 0.0070.161 0.402
Gem 0.382 0.021 0.192 0.485
Qwn 0.513 0.060 0.256 0.494
Apt
H0.034 0.0060.179 0.408
Gem 0.290 0.024 0.210 0.499
Qwn 0.439 0.061 0.277 0.503
Apt
S0.050 0.009 0.187 0.407
Gem 0.387 0.032 0.216 0.505
Qwn0.550 0.077 0.293 0.513
Table 2: Model-implied conditional probabilities θA−,
θA+,θT−, and θT+across all 27 RAG configurations,
reported as posterior mean estimates. We show the
highest value for each dataset inboldand the lowest in
italics.
on retrieval success remains low across all models,
indicating that unnecessary abstention is uncom-
mon.
A second pattern is that retrieval success con-
sistently improves answer success among non-
abstained responses, but the magnitude of this im-
provement depends strongly on the dataset. On
FEV, unsupported task success remains high even
when retrieval fails, reflecting the relative ease of
the task and the fact that each question has only
two answer options. By contrast, onHQAand
NQ, where retrieval success requires recovering
all necessary documents and answers must exactly
match an open-form reference, the gap between
P(T=1|R=0, A=0) andP(T=1|R=1, A=0)
is much larger.
The conditional decomposition also clarifies why
similar marginal task success can hide substan-
tial behavioral differences. OnNQwith dense
retrieval, the three generators achieve nearly iden-tical task success rates (Apertus: 0.239 ,Gemma3:
0.242 ,Qwen3.5: 0.243 ), yet their policy adherence
differs sharply (Apertus: 0.164 ,Gemma3: 0.414 ,
Qwen3.5: 0.496 ). In other words, comparable end-
task accuracy does not imply comparable behavior
under retrieval failure. The conditional probabili-
ties reveal that these differences are largely driven
by the abstention policy rather than by answer ac-
curacy alone.
Finally, improved retrieval does not automati-
cally translate into proportional gains in either task
success or policy adherence. Better retrieval in-
creases the opportunity to answer correctly, but the
realized benefit depends on whether the generator
both recognizes retrieval failure and uses retrieved
support effectively when it is available.
5.3 The Sample Allocation Problem
Here, we investigate a setting closer to what one
might encounter in real-world applications, where
annotation scarcity is common. Thus, we assume
access to 100 annotations (Basesamples) for re-
trieval and task success, and that abstention is al-
ways observed (via simple string matching). Then,
we assume that we are given an additional budget
for more annotations (Add.samples). The question
is where to allocate the additional samples: to mea-
suring retrieval success, task success, or a mix of
the two. We apply the model to 100 Base samples
with full annotations for retrieval and task success,
assuming that abstention is always observed via
simple string matching. Given an additional anno-
tation budget of Add.∈60,100,200,500 samples,
we investigate five allocation strategies: (1) all ad-
ditional samples receive full annotations (i.e., both
task-success and retrieval-success annotations), (2)
half receive full annotations and half only retrieval-
success annotations, (3) half receive full and half
only task-success annotations, (4) all additional
samples receive only retrieval-success annotations,
and (5) all additional samples receive only task-
success annotations. To ensure robust estimates,
we subsample 500 times from the 10,000 avail-
able samples and report the Mean Absolute Error
(MAE) relative to the full-data point estimate, as
well as the 95% credible interval width. We run
experiments on the HQA dataset for Apertus and
Qwen using the Hybrid retriever.
The Observation.Figure 2 reveals a consistent
asymmetry across both models: retrieval-focused
strategies reduce estimation error more effectively
for policy adherence (G), while task-focused strate-

0 60100 200 500
Additional budget0.0100.0150.0200.0250.0300.0350.0400.045MAE
Qwen  Policy adherence P(G=1)
0 60100 200 500
Additional budget0.0100.0150.0200.0250.0300.0350.0400.045MAE
Qwen  Task success P(T=1)
0 60100 200 500
Additional budget0.0100.0150.0200.0250.0300.0350.0400.045MAE
Apertus  Policy adherence P(G=1)
0 60100 200 500
Additional budget0.0100.0150.0200.0250.0300.0350.0400.045MAE
Apertus  Task success P(T=1)
All joint (baseline) Half-joint R Half-joint T All R partial All T partialFigure 2: MAE for policy adherence P(G=1) and task success P(T=1) under five annotation allocation strategies,
for Qwen (left) and Apertus (right). All configurations start with 100 fully annotated base samples. Abstention is
always observed. Results on HotpotQA with Hybrid retrieval, averaged over 500 subsamples.
gies are more effective for task success (T). For
P(G=1) , the all-R strategy matches or approaches
the all-joint baseline for Qwen, whereas all-T
shows markedly slower improvement and plateaus
at a higher MAE. This pattern holds for Apertus,
though the gap between all-R and all-joint widens
at larger budgets. For P(T=1) , the pattern re-
verses: all-T closely tracks the all-joint baseline
for both models, while all-R provides negligible
improvement; its MAE and CI width remain nearly
flat regardless of budget, particularly for Aper-
tus. The half-joint strategies consistently fall be-
tween the corresponding extremes, with half-joint-
R closer to all-joint for P(G=1) and half-joint-T
closer to all-joint for P(T=1) . Notably, the all-
joint strategy never underperforms the best partial
strategy by a large margin, making it a robust de-
fault when the estimation target is not known in
advance.
The Explanation.To understand this asymmetry,
we analyze the information gain of each strategy
for estimating P(G=1) (the case for P(T=1) is
straightforward, since task-success annotations di-
rectly observe T). Since the abstention Ais always
observed, we measure the conditional information
gainI(G;· |A) of additionally observing R,T, or
both. The key insight follows from the definition of
G: of the four (R, A) cells, three resolve Gdeter-
ministically, only (R=1, A=0) requires knowing
T(see Table 3a). Thus, observing Rresolves G
in three of four cases, while observing Tresolves
Gonly in one of three non-zero (A, T) cells. This
structural asymmetry explains why retrieval anno-
tations are consistently more informative for G
than task annotations. Table 3b quantifies this us-
ing the conditional probabilities from Table 2 (full
derivations in Appendix F). For both models, the
All-R strategy captures the majority of the infor-mation provided by the All-Joint strategy ( 65.6%
for Qwen, 58.0% for Apertus), whereas All-T cap-
tures far less ( 29.3% for Qwen, 40.8% for Aper-
tus). The difference between models reflects their
abstention behavior: Qwen’s strong retrieval, ab-
stention dependency ( P(A=1|R=0) = 0.463 vs.
P(A=1|R=1) = 0.036 ) means that Robserva-
tions yield highly variable Glabels, whereas Aper-
tus rarely abstains regardless of retrieval ( P(A=1|
R=0) = 0.039 ), reducing the discriminative value
ofR. The information-theoretic ranking correctly
predicts the empirical ordering of partial strate-
gies. The only discrepancy is that for Qwen, All-R
slightly outperforms All-Joint at large budgets, de-
spite lower per-sample information gain. This is
because the 100 base joint samples already con-
strain θT+=P(T=1|R=1, A=0) in the single
ambiguous cell, after which additional joint annota-
tions provide diminishing returns (see Appendix F
for details).
5.4 The Automated Judge
Since human annotations are highly time and cost-
intensive, it has become common practice to use
LLM-as-a-judge to automate parts of the evalua-
tion. We investigate the impact of using automated
judgments on the evaluation pipeline. The difficulty
stems from the need to calibrate automated judg-
ments to match human judgments (von Däniken
et al., 2022), which introduces uncertainty that de-
pends on the automated judge’s performance.
The Judge.We use gpt-4o-mini (OpenAI et al.,
2024) as our judge for both the task and retrieval
success rates following the evaluation framework
proposed by (Saad-Falcon et al., 2024) for context
and answer relevance. We compare the retrieval
success rates according to the judge P(R J=
1), P(T J= 1) to those according to humans

R A GQwen Apertus
0 110.367 0.031
0 000.426 0.762
1 100.007 0.001
1 0T0.200 0.206
(a) Joint probabilities P(R, A) with the resulting value of G.
In three of four cells, Gis determined by (R, A) alone; only
(R=1, A=0)requiresT.
Strategy Qwen Apertus
All-Joint 0.526 0.490
Half-Joint-R 0.436 0.387
All-R 0.345 0.284
Half-Joint-T 0.340 0.345
All-T 0.154 0.200
(b) Information gain I(G;· |A) per sample for each an-
notation strategy. Higher values indicate more informative
observations for estimatingP(G=1).
Table 3: Information-theoretic analysis of annotation
strategies on HotpotQA with Hybrid retrieval.
DS Ret. TPR FPRP(R)P(R J)
HQA D 0.77 0.17 0.15 0.26
HQA H 0.77 0.19 0.21 0.31
HQA S 0.77 0.18 0.22 0.31
(a) Retrieval judge.
DS Gen. TPR FPRP(T)P(T J)
HQA Qwn 0.94 0.32 0.21 0.52
HQA Apt 0.88 0.43 0.21 0.57
HQA Gem 0.93 0.38 0.21 0.55
(b) Task judge.
Table 4: LLM-as-a-judge calibration on HotpotQA.
TPR and FPR denote the judge’s true and false posi-
tive rates, e.g., TPR=P(R J=1|R=1) andFPR=
P(RJ=1|R=0) for retrieval. The judge exhibits high
sensitivity but a substantial false-positive rate, leading
to an overestimation of success rates.
P(R= 1), P(T= 1) . The judge’s performance is
measured in terms of true-and-false positive rates
(TPR= (R J= 1|R= 1) ,FPR=P(R J=
1|R= 0) ). An analogous procedure is applied
to task success, comparing TJwithTto derive
the TPR and FPR. Tables 4a and 4b report cal-
ibration results for the retrieval and task judges
on HotpotQA, showing that while both judges
achieve high sensitivity, they exhibit substantial
FPR, leading to overestimated success rates P(R J)
andP(T J).
The Judge’s Impact.We investigate whether sup-
plementing the 200 human-annotated base sam-
ples with automated judge annotations improvesP(G=1) P(T=1)
nadd MAE CI W. MAE CI W.
0 0.0174 0.0941 0.0232 0.1246
500 0.0166 0.0838 0.0243 0.1176
5000 0.0170 0.0802 0.0230 0.1150
Table 5: Mean absolute error and 95% credible interval
width for P(G=1) and P(T=1) based on 200 fully anno-
tated observations and additional observations where
we observeR Jinstead of R andT Jinstead of T.
estimation quality. We add nadd∈ {0,500,5000}
judge-annotated samples to the base set, treating
RJandTJas noisy observations of RandTwith
the calibrated TPR and FPR from Table 4. Ta-
ble 5 reports results for Apertus on HotpotQA with
Hybrid retrieval. Adding automated judgments
yields only marginal improvements: the CI width
forP(G=1) decreases from 0.094 to0.080 even
with 5,000 additional samples, while MAE remains
essentially unchanged. The pattern for P(T=1) is
similar. This is consistent with the judge’s high
false-positive rates (Table 4): when the FPR is sub-
stantial, each automated annotation carries limited
information, and large volumes of noisy labels can-
not substitute for even modest amounts of human
annotation. The result highlights that the value of
LLM-as-a-judge annotations depends critically on
calibration quality. While our empirical findings
are specific to the judges and calibration setup con-
sidered here, they illustrate that a poorly calibrated
judge may add annotation volume without mean-
ingfully reducing uncertainty. This is consistent
with the findings of prior literature (von Däniken
et al., 2022).
6 The Conclusion
We presented a Bayesian evaluation framework for
RAG systems that factorizes the joint distribution
over retrieval success, abstention, and task suc-
cess according to the pipeline’s information flow.
The key distinction is between task success, i.e.,
whether the user received a correct answer, and gen-
erator success, i.e., whether the generator behaved
appropriately given the retrieval outcome. Apply-
ing the model to 27 configurations, we showed
that systems with near-identical task success can
differ sharply in policy adherence, a distinction
that marginal metrics conceal but the conditional
decomposition makes explicit.
On the practical side, we analyzed the anno-
tation allocation problem and demonstrated that

retrieval-success annotations are more informative
than task-success annotations for estimating pol-
icy adherence. This asymmetry has a structural
explanation: observing retrieval resolves generator
success in three of four joint cells, whereas ob-
serving task success resolves only one. We further
extended the model to incorporate LLM-as-a-judge
annotations as calibrated noisy observations. Our
results show that when the judge exhibits high false-
positive rates, even thousands of automated annota-
tions yield only marginal gains over a small set of
human judgments, underscoring the importance of
calibration quality.
More broadly, the framework illustrates how
casting evaluation as probabilistic inference en-
ables reasoning about uncertainty, partial observ-
ability, and annotation efficiency within a unified
model. Natural extensions include continuous qual-
ity scales and more complex pipelines incorpo-
rating reranking, query reformulation, or iterative
retrieval. Future work could also consider a hi-
erarchical model that shares statistical strength
across related datasets, retrievers, and generators,
rather than treating each RAG configuration inde-
pendently.
Limitations
Binary Model.The core limitation of this work
is the simplified assumption that all judgments are
provided on a binary scale (fail vs. success). Espe-
cially in Information Retrieval, the use of contin-
uous metrics such as MAP and MRR is common
and yields a more fine-grained model.
Deterministic Policy Definition.Generator suc-
cess is defined by a single fixed policy (abstain iff
retrieval fails, answer correctly otherwise). In prac-
tice, reasonable policies may differ; for instance,
a system might legitimately attempt a partial an-
swer when retrieval is incomplete, or abstention
thresholds may be application-dependent. So far,
the framework does not accommodate soft or alter-
native policy definitions.
Limited Task and Model Diversity.Due to bud-
get limitations, the experiments cover three datasets
(one fact-checking, two QA) and three relatively
small open-weight generators (8–12B parameters).
Simple RAG Pipeline.The framework models
a minimal retrieve-then-generate pipeline. Current
production RAG systems often include additionalcomponents such as query reformulation, rerank-
ing, chunk filtering, or multi-turn retrieval. Each
additional component introduces its own failure
modes and dependencies that the current model
does not capture. Extending the framework to
deeper pipelines would require additional variables
and a more complex dependency structure.
Single-Turn Evaluation.The framework evalu-
ates each query in isolation as a single-turn inter-
action. It does not account for multi-turn conversa-
tional RAG settings, where retrieval and generation
decisions depend on dialogue history, and where
errors in earlier turns can compound across the
conversation.
Abstention Detection.Abstention is detected
through exact matching against the prescribed ab-
stention response. This is appropriate in our con-
strained generation setting, where answers are re-
stricted to a single entity, number, or fixed label, but
would not capture hedged or indirect abstentions in
free-form generation. Such settings could instead
incorporate a calibrated abstention classifier as a
noisy observation model.
Acknowledgments
This work was supported by the Swiss National Sci-
ence Foundation (SNF) within the project "Unified
Model for Evaluation of Text Generation Systems
(UniVal)" [200020_219819].
References
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, Diego
De Las Casas, Aurelia Guy, Jacob Menick, Roman
Ring, Tom Hennigan, Saffron Huang, Loren Mag-
giore, Chris Jones, Albin Cassirer, and 9 others. 2022.
Improving language models by retrieving from tril-
lions of tokens. InProceedings of the 39th Interna-
tional Conference on Machine Learning, volume 162
ofProceedings of Machine Learning Research, pages
2206–2240. PMLR.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation.Proceedings of
the AAAI Conference on Artificial Intelligence,
38(16):17754–17762.
Gordon V . Cormack, Charles L A Clarke, and Stefan
Buettcher. 2009. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. In
Proceedings of the 32nd International ACM SIGIR

Conference on Research and Development in Infor-
mation Retrieval, SIGIR ’09, page 758–759, New
York, NY , USA. Association for Computing Machin-
ery.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré,
Maria Lomeli, Lucas Hosseini, and Hervé Jégou.
2024. The faiss library.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158, St. Julians,
Malta. Association for Computational Linguistics.
Alejandro Hernández-Cano, Alexander Hägele,
Allen Hao Huang, Angelika Romanou, Antoni-
Joan Solergibert, Barna Pasztor, Bettina Mess-
mer, Dhia Garbaya, Eduard Frank ˇDurech, Ido
Hakimi, Juan García Giraldo, Mete Ismayilzada,
Negar Foroutan, Skander Moalla, Tiancheng
Chen, Vinko Sabol ˇcec, Yixuan Xu, Michael
Aerni, Badr AlKhamissi, and 82 others. 2025.
Apertus: Democratizing Open and Compli-
ant LLMs for Global Language Environments.
https://arxiv.org/abs/2509.14233.
Matthew D. Hoffman and Andrew Gelman. 2014. The
no-u-turn sampler: Adaptively setting path lengths
in hamiltonian monte carlo.Journal of Machine
Learning Research, 15(47):1593–1623.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong,
Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu,
Tong Xu, and Enhong Chen. 2025. CRUD-RAG:
A comprehensive chinese benchmark for retrieval-
augmented generation of large language models.
ACM Trans. Inf. Syst., 43(2).
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb,Alex Beutel, Alex Borzunov, Alex Carney, Alex
Chow, Alex Kirillov, and 401 others. 2024. GPT-
4o system card.Preprint, arXiv:2410.21276.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
Vassilis Plachouras, Tim Rocktäschel, and Sebastian
Riedel. 2021. KILT: a benchmark for knowledge
intensive language tasks. InProceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, pages 2523–2544, Online.
Association for Computational Linguistics.
Qwen Team. 2026. Qwen3.5: Towards native multi-
modal agents.
David Rau, Hervé Déjean, Nadezhda Chirkova, Thibault
Formal, Shuai Wang, Stéphane Clinchant, and Vas-
silina Nikoulina. 2024. BERGEN: A benchmarking
library for retrieval-augmented generation. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 7640–7663, Miami, Florida,
USA. Association for Computational Linguistics.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 338–354,
Mexico City, Mexico. Association for Computational
Linguistics.
Hinrich Schütze, Christopher D Manning, and Prab-
hakar Raghavan. 2008.Introduction to Information
Retrieval, volume 39. Cambridge University Press
Cambridge.
Stan Development Team. 2026. Stan Reference Manual
2.38. https://mc-stan.org.
Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram,
Michael Günther, Bo Wang, Markus Krimmel, Feng
Wang, Georgios Mastrapas, Andreas Koukounas, An-
dreas Koukounas, Nan Wang, and Han Xiao. 2024.
jina-embeddings-v3: Multilingual embeddings with
task LoRA.Preprint, arXiv:2409.10173.
Yixuan Tang and Yi Yang. 2024. Multihop-RAG:
Benchmarking retrieval-augmented generation for
multi-hop queries.arXiv preprint arXiv:2401.15391.
Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya
Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin,
Tatiana Matejovicova, Alexandre Ramé, Morgane
Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey
Cideron, Jean bastien Grill, Sabela Ramos, Edouard
Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev,
and 197 others. 2025. Gemma 3 technical report.
Preprint, arXiv:2503.19786.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.

FEVER: a large-scale dataset for fact extraction and
VERification. InNAACL-HLT.
TruLens. 2024. The RAG triad. https:
//www.trulens.org/getting_started/core_
concepts/rag_triad/. Accessed: 2026-07-30.
Pius von Däniken, Jan Deriu, Don Tuggener, and Mark
Cieliebak. 2022. On the effectiveness of automated
metrics for text generation systems. InFindings
of the Association for Computational Linguistics:
EMNLP 2022, pages 1503–1522, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Pius von Däniken, Jan Milan Deriu, Alvaro Rodrigo,
and Mark Cieliebak. 2024. Improving quantifica-
tion with minimal in-domain annotations: Beyond
classify and count. InProceedings of the Interna-
tional AAAI Conference on Web and Social Media,
volume 18, pages 1585–1598.
Shuai Wang, Ekaterina Khramtsova, Shengyao Zhuang,
and Guido Zuccon. 2024. Feb4rag: Evaluating fed-
erated search in the context of retrieval augmented
generation. InProceedings of the 47th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 763–773.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. InConference on Empirical Methods in Natural
Language Processing (EMNLP).
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu,
and Zhaofeng Liu. 2025. Evaluation of retrieval-
augmented generation: A survey. InBig Data, pages
102–120, Singapore. Springer Nature Singapore.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph E Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. In
Advances in Neural Information Processing Systems,
volume 36, pages 46595–46623. Curran Associates,
Inc.
A Experimental Setup
A.1 Retrieval
Table 6 summarizes the retrieval configuration. All
retrievers use top-K= 5.
A.2 Generation
A.3 Evaluation
No normalization, article stripping, or case adjust-
ment is applied to generated answers prior to match-
ing against the ground truth. Abstention is detected
via exact string matching: for FEVER, the labelSparse (BM25)
Tokenizer Whitespace
Stemmer English
Stopwords English
Min. DF 1
Lowercase True
Ampersand norm. True
Special char norm. True
Acronym norm. True
Punctuation removal True
Dense
Embedding modeljinaai/jina-embeddings-v3
Embedding tasktext-matching
Retrieval taskretrieval.query
HNSW dim. 1024
HNSWM32
HNSW metric Inner product
Hybrid
RRFk1.0
Table 6: Retrieval hyperparameters.
Generator
Temperature 0.7
Max tokens 512
Max concurrent requests 15
Thinking None
LLM-as-a-Judge
Modelgpt-4o-mini-2024-07-18
Max tokens 1
Temperature 1
Logprobs True
Top logprobs 20
Table 7: Generation and judge hyperparameters.
NOT_ENOUGH_INFO is matched; for HotpotQA and
NQ, the outputI DO NOT KNOWis matched.
B Generation Prompts
B.1 System Prompts
Depending on the task, a different system prompt
is used.
B.1.1 Fact Checking
Used for the Fever task.
You are a master fact checker. From
given passages and a claim you can say
whether the claim is: SUPPORTS, RE-
FUTES.
If the passages do not provide a clear an-
swer you need to say: "NOT ENOUGH
INFO".

Only answer with one of the given
labels: SUPPORTS, REFUTES, NOT
ENOUGH INFO.
B.1.2 QA
Used for both the HotpotQA and Natural Question
task.
You are a precise question-answering as-
sistant. You will be given a question
and a list of retrieved paragraphs con-
taining the answer. Your response must
be ONLY the answer itself. A single
word, name, entity, or number. No expla-
nation, no punctuation, no full sentences,
no preamble. Only use the retrieved para-
graphs to answer the question! If the an-
swer is not in the retrieved paragraphs
you must say: I DO NOT KNOW
Examples:
Q: What nationality is Friedrich Merz?
A: German
Q: How many siblings did Marie Curie
have?
A: Four
B.1.3 User Prompts
For all tasks the same user prompt is used. It con-
sists of a list of paragraphs and the original query.
{% if retrieved|length > 0 %}
<retrieved>
{% for p in retrieved %}
<paragraph
document_id="{{ p.document_id }}"
index="{{ p.index }}"
>
{{ p.text }}
</paragraph>
{% endfor %}
</retrieved>
{% endif %}
<question>
{{ input }}
</question>
C Partial Retrieval Success
In Section 3, we define retrieval success Ras bi-
nary and measure downstream abstention and taskDataset Retriever Partial retrievals,n(%)
FEVERDense 419 (4.19%)
Hybrid 453 (4.53%)
Sparse 300 (3.00%)
HotpotQADense 4,724 (47.24%)
Hybrid 5,647 (56.47%)
Sparse 5,233 (52.33%)
NQDense 0 (0.00%)
Hybrid 0 (0.00%)
Sparse 0 (0.00%)
Table 8: Number of partial retrievals for each dataset
and retriever.
behavior conditional on this binary outcome. In
practice, however, a generation model may be able
to produce a correct answer from only partial con-
text4. Here, we add an experiment studying this
setting. We define a ternary retrieval outcome
R∈fail,partial,success , where failure means that
no relevant documents were retrieved, partial re-
trieval means that some but not all relevant docu-
ments were retrieved, and success means that all
relevant documents were retrieved.
Table 8 shows the number and proportion of par-
tial retrievals for each dataset and retriever. ForNQ,
the rate of partial retrieval is 0%, since each query
in this dataset has exactly one relevant document.
Similarly, the majority ofFEVqueries have ex-
actly one relevant document, while a minority have
more than one, leading to a maximum of 453 par-
tial retrievals with theHybridretriever. ForHQA,
on the other hand, every query has two relevant
documents by construction, sinceHQAis designed
for multi-hop reasoning. Consequently, its partial
retrieval rate reaches up to 56.47
We calculate the conditional decomposition for
this ternary setup in Table 9. We define the condi-
tional probability parameters analogously to Sec-
tion 3: θA,r=P(A= 1|R=r) andθT,r=
P(T= 1|A= 0, R=r) , where r∈f, p, s cor-
responds to failure, partial retrieval, and success,
respectively. The results are overall consistent with
the discussion in Section 5. In particular, the ab-
stention rate decreases monotonically as retrieval
quality increases, while the task success rate in-
creases.
4This is in part due to relevance annotations not distin-
guishing which documents are individually sufficient or jointly
necessary.

DS Gen. Ret.θ A,f θA,p θA,s θT,f θT,p θT,s
HQAApt
D0.0640.0270.0090.1200.3130.492
Gem 0.592 0.177 0.025 0.143 0.379 0.570
Qwn0.812 0.373 0.044 0.272 0.5020.638
Apt
H0.0580.0310.0050.1270.3100.519
Gem 0.518 0.169 0.020 0.158 0.372 0.608
Qwn 0.723 0.358 0.036 0.242 0.493 0.679
Apt
S0.068 0.0280.0050.146 0.321 0.553
Gem 0.525 0.168 0.011 0.155 0.374 0.633
Qwn 0.744 0.369 0.027 0.208 0.4940.705
Table 9: Count-based conditional probabilities for HotpotQA treating retrieval outcome as ternary. We show the
highest value inboldand the lowest initalics.
D LLM-as-a-Judge Prompts
The following prompts are used for the LLM-as-a-
judge results.
D.1 System Prompts
System prompts differ between the two judges.
D.1.1 Retrieval Judge
Given the following question and doc-
uments, you must analyze the provided
documents and determine whether they
are sufficient for answering the question.
In your evaluation, you should consider
the content of the documents and how
they relate to the provided question. Out-
put your final verdict by strictly follow-
ing this format: "Yes" if the documents
are sufficient and "No" if the documents
provided are not sufficient. Do not pro-
vide any additional explanation for your
decision.
D.1.2 Task Judge
Given the following question, docu-
ments, and answer, you must analyze the
provided answer and documents before
determining whether the answer is rele-
vant for the provided question. In your
evaluation, you should consider whether
the answer addresses all aspects of the
question and provides only correct in-
formation from the documents for an-
swering the question. Output your final
verdict by strictly following this format:
"Yes" if the answer is relevant for the
given question and "No" if the answer is
not relevant for the given question. Do
not provide any additional explanation
for your decision.D.2 User Prompts
Both the retrieval and task judge use the same user
prompt template. However the retrieval judge does
not see the generated answer.
<context>
{% for document in context %}
<document>
{{ document }}
</document>
{% endfor %}
</context>
<question>{{ query }}</question>
{% if response|length > 0 %}
<answer>{{ response }}</answer>
{% endif %}
E Extended Sample Allocation Plot
Figure 3 reports both MAE and 95% credible in-
terval width for policy adherence P(G=1) and
task success P(T=1) across all five annotation
strategies, complementing the MAE-only summary
in the main text (Figure 2). The credible inter-
val width closely tracks the MAE ordering: for
P(G=1) , retrieval-focused strategies yield nar-
rower intervals than task-focused ones at every bud-
get level, while for P(T=1) the pattern reverses.
For instance, at budget 500, the All-R strategy
achieves a CI width of 0.105 for Qwen’s P(G=1)
compared to 0.132 for All-T, whereas for P(T=1)
the All-T strategy reaches 0.075 versus 0.150 for
All-R. Coverage remains close to the nominal 95%
level across all strategies and budget levels, indicat-
ing that the posterior intervals are well-calibrated
regardless of the allocation strategy. The all-joint
strategy consistently yields among the narrowest
intervals for both quantities, confirming its robust-

R A GQwen Apertus
0 110.367 0.031
0 000.426 0.762
1 100.007 0.001
1 0T0.200 0.206
Table 10: Joint probabilities P(R, A) for each cell of
the(R, A) contingency table, with the resulting value of
G. In three of four cells, Gis fully determined by (R, A)
alone. Only the cell (R=1, A=0) leaves Gunresolved,
where G=T . Values shown for Qwen and Apertus on
HotpotQA with Hybrid retrieval.
ness as a default when the estimation target is un-
known in advance.
F Information Gain Calculations
We now analyze why certain annotation strategies
are more effective than others for estimating policy
adherence P(G=1) . Since abstention is always
observed via string matching, we condition on A
throughout and ask: how much does additionally
observing R,T, or both reduce our uncertainty
aboutG?
Joint Probability Structure.Recall that Gis
defined as:
G= (R=0∧A=1)∨(R=1∧A=0∧T=1).
Table 10 shows the four (R, A) cells and whether
Gis determined. In three cells, Gfollows directly
from RandA—these correspond to cases where
either retrieval or abstention went wrong. Only
when both retrieval and abstention are correct, i.e.,
(R=1, A=0) , does Gdepend on T. This asymme-
try is the structural basis for the annotation effi-
ciency differences we observe.
Base Uncertainty.The uncertainty about G
given onlyAis:
H(G|A) =X
aP(A=a)·H(G|A=a).(1)
When A=1 (the generator abstained), Gdepends
only on whether the abstention was justified, i.e.,
whetherR=0. SinceRis unobserved:
H(G|A=1) =hP(A=1, R=0)
P(A=1)
,(2)
where h(·) denotes the binary entropy function.
For Qwen, P(R=0|A=1) = 0.367/0.374 =
0.981 , yielding h(0.981) = 0.134 ; for Apertus,
0.031/0.032 = 0.969 , yielding h(0.969) = 0.196 .When A=0 (the generator answered), G=1 re-
quires both R=1 andT=1 . With neither observed:
H(G|A=0) =h 
P(G=1|A=0)
(3)
where
P(G=1|A=0) =P(R=1)P(A=0|R=1)θ T+
P(A=0)
(4)
Rememberθ T+=P(T= 1|R= 1, A= 0).
For Qwen, P(G=1|A=0) = 0.216 , giv-
ingh(0.216) = 0.760 ; for Apertus, P(G=1|
A=0) = 0.110, givingh(0.110) = 0.500.
Combining both cases:
H(G|A) =P(A=1)·h A=1 +P(A=0)·h A=0,
(5)
yielding 0.526 for Qwen and 0.490 for Apertus.
Despite very different abstention behaviors, both
models exhibit similar base uncertainty—though
from different sources: for Qwen, a balanced split
between G=1 andG=0 in the A=0 cell; for Aper-
tus, the sheer mass of the A=0 cell (96.8% of sam-
ples) compensates for its lower per-sample entropy.
All-Joint Strategy.Observing both RandTre-
solves Gcompletely, since Gis a deterministic
function of(R, A, T):
I(G;R, T|A) =H(G|A)−H(G|R, A, T)|{z }
= 0
=H(G|A).(6)
This represents the maximum achievable informa-
tion gain per sample.
All-R Strategy.Observing onlyRyields:
I(G;R|A) =H(G|A)−H(G|R, A).(7)
From Table 10, three of four (R, A) cells resolve G
deterministically. Only (R=1, A=0) leaves Gun-
resolved, where G=T . The residual uncertainty
is therefore:
H(G|R, A) =P(R=1, A=0)·h(θ T+).(8)
For Qwen: 0.200×h(0.678) = 0.181 ; for Apertus:
0.206×h(0.519) = 0.206 . The resulting gains are
I(G;R|A) = 0.345 (Qwen) and 0.284 (Aper-
tus). Observing Reliminates the majority of the
uncertainty aboutG.

0.0100.0150.0200.0250.0300.0350.0400.045MAE
Qwen  MAE
0.060.080.100.120.140.160.180.20CI width
Qwen  CI width
0.0100.0150.0200.0250.0300.0350.0400.045MAE
Apertus  MAE
0.060.080.100.120.140.160.180.20CI width
Policy adherence P(G=1)Apertus  CI width
0 60100 200 500
Additional budget0.0100.0150.0200.0250.0300.0350.0400.045MAE
0 60100 200 500
Additional budget0.060.080.100.120.140.160.180.20CI width
0 60100 200 500
Additional budget0.0100.0150.0200.0250.0300.0350.0400.045MAE
0 60100 200 500
Additional budget0.060.080.100.120.140.160.180.20CI width
Task success P(T=1)
All joint (baseline) Half-joint R Half-joint T All R partial All T partialFigure 3: Estimation quality (MAE and 95% credible interval width) for policy adherence P(G= 1) (top) and task
success P(T= 1) (bottom) under five annotation allocation strategies, for Qwen (left) and Apertus (right). All
configurations start with 100 fully annotated base samples and increase the additional budget. Abstention is always
observed. Results on HotpotQA with Hybrid retrieval, averaged over 500 subsamples.
All-T Strategy.Observing onlyTyields:
I(G;T|A) =H(G|A)−H(G|A, T).(9)
Of the three non-zero (A, T) cells, only
(A=0, T=0) resolves Gdeterministically ( G=0 ,
since both branches of the definition fail). The
remaining two cells requireR:
•(A=1, T=0) : the generator abstained, but Gde-
pends on whether the abstention was justified,
i.e.,H=h(P(R=0|A=1)).
•(A=0, T=1) : the generator answered correctly,
butGdepends on whether the answer was sup-
ported, i.e.,H=h(P(R=1|A=0, T=1)).
The residual uncertainty is:
H(G|A, T) =P(A=1)·h(P(R=0|A=1))
+P(A=0, T=1)
·h(P(R=1|A=0, T=1)),
(10)
where P(R=1|A=0, T=1) is obtained via Bayes’
rule:
P(R=1)P(A=0|R=1)θ T+
P(A=0, T=1).(11)
For Qwen: P(R=1|A=0, T=1) = 0.411 , yield-
ingH(G|A, T) = 0.372 andI(G;T|A) =
0.154 . For Apertus: P(R=1|A=0, T=1) =
0.352 , yielding H(G|A, T) = 0.290 and
I(G;T|A) = 0.200.Half-Joint-R Strategy.Here, half the additional
budget is allocated to joint samples and the other
half to retrieval-only samples. Each joint sample
contributes I(G;R, T|A) =H(G|A) , while
each R-partial sample contributes I(G;R|A) .
The expected per-sample gain is the average:
1
2H(G|A) +1
2I(G;R|A),(12)
yielding 0.436 for Qwen and 0.387 for Apertus.
Since the R-partial component already captures
most of the uncertainty (three of four cells re-
solved), the loss relative to the all-joint strategy
is modest.
Half-Joint-T Strategy.Analogously, half the
budget goes to joint samples and half to task-only
samples. The expected per-sample gain is:
1
2H(G|A) +1
2I(G;T|A),(13)
yielding 0.340 for Qwen and 0.345 for Apertus.
Despite receiving the same number of joint sam-
ples as Half-Joint-R, this strategy is less effective
because the T-partial component contributes less in-
formation about G—it leaves two cells unresolved
rather than one.
Summary.Table 11 summarizes the information
gains. The structural asymmetry is clear: observ-
ingRleaves one ambiguous cell for G, while

observing Tleaves two. This directly explains
why retrieval-focused strategies outperform task-
focused strategies for estimating policy adherence.
The mixed strategies interpolate linearly between
these extremes.
Discussion.The information-theoretic ranking in
Table 11 correctly predicts the relative ordering of
partial strategies: All-R consistently outperforms
All-T for estimating P(G=1) , and the half-joint
strategies fall between the corresponding extremes.
However, the predicted ranking does not perfectly
match the empirical results at all budget levels. For
Apertus, the empirical MAE ordering at budget
500 follows the theoretical ranking closely: All-
Joint≈Half-Joint-R <Half-Joint-T <All-R <
All-T. For Qwen, however, All-R and Half-Joint-R
slightly outperform All-Joint despite having lower
per-sample information gain.
This discrepancy arises because the information-
theoretic analysis treats each sample independently,
whereas the Bayesian model shares information
across cells through its parameterization. The
only cell left unresolved by (R, A) observations
is(R=1, A=0) , where G=T . The 100 base joint
samples provide direct observations of Tin this
cell, allowing the model to estimate θT+. Once this
parameter is sufficiently well-estimated, additional
joint samples yield diminishing returns—they an-
notate Tin cells where Gis already resolved by
(R, A) alone. R-partial samples, by contrast, con-
tribute exclusively to the three resolved cells, where
each observation provides a clean binary label for
G.
For Apertus, θT+= 0.519 with near-maximal
entropy ( h(0.519) = 0.999 ), making it intrinsi-
cally harder to estimate. The 100 base samples
are insufficient to pin it down, and additional joint
samples that directly observe Tin the ambiguous
cell remain valuable. For Qwen, θT+= 0.678
with lower entropy ( h(0.678) = 0.904 ), allowing
the base samples to estimate it more effectively
and reducing the marginal value of additional joint
annotations.
In summary, the information-theoretic analysis
reliably predictswhichpartial observation is more
informative for a given target, while the question
of whether partial annotations can fullysubstitute
for joint annotations depends on how well the base
samples constrain the parameters in the ambiguous
cells.Strategy Qwen Apertus
All-Joint 0.526 0.490
Half-Joint-R 0.436 0.387
All-R 0.345 0.284
Half-Joint-T 0.340 0.345
All-T 0.154 0.200
Table 11: Information gain I(G;· |A) per sample for
each annotation strategy, computed from the conditional
probabilities in Table 2. Higher values indicate more
informative observations for estimating P(G=1) . The
ranking is consistent with the empirical MAE ordering
in Figure 2.
G AI Assistants
AI assistants (Claude, Anthropic) were used for
code development, data analysis, and iterative draft-
ing of manuscript text. All outputs were reviewed
and validated by the authors.