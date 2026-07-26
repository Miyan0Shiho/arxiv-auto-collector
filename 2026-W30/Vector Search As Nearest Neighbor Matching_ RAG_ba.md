# Vector Search As Nearest Neighbor Matching: RAG-based Policy Learning in Causal Inference

**Authors**: Masahiro Kato, Taka Kato

**Published**: 2026-07-20 17:57:20

**PDF URL**: [https://arxiv.org/pdf/2607.18225v1](https://arxiv.org/pdf/2607.18225v1)

## Abstract
We propose one-step and two-step methods for policy learning with retrieval-augmented generation (RAG). We formulate RAG-based action selection under the potential outcome framework. In the two-step method, vector search retrieves action-specific neighboring evidence in an embedding space, the generator estimates conditional expected outcomes or their contrasts, and a plug-in rule selects an action. This formulation connects action-specific vector search with nearest-neighbor matching in causal inference. We decompose the regret of the two-step method into candidate-generation regret and within-candidate choice regret, and we bound the latter using prediction-error guarantees for nearest-neighbor estimators and transformers. We evaluate the one-step method directly as a policy because its intermediate computation is unobserved.

## Full Text


<!-- PDF content starts -->

Vector Search As Nearest Neighbor Matching:
RAG-based Policy Learning in Causal Inference
Masahiro Kato∗and Taka Kato†
∗The University of Tokyo.
†NP-hard Inc.
July 21, 2026
Abstract
We propose one-step and two-step methods for policy learning with retrieval-
augmented generation (RAG). We formulate RAG-based action selection under the
potential outcome framework. In the two-step method, vector search retrieves action-
specific neighboring evidence in an embedding space, the generator estimates conditional
expected outcomes or their contrasts, and a plug-in rule selects an action. This formu-
lation connects action-specific vector search with nearest-neighbor matching in causal
inference. We decompose the regret of the two-step method into candidate-generation
regret and within-candidate choice regret, and we bound the latter using prediction-error
guarantees for nearest-neighbor estimators and transformers. We evaluate the one-step
method directly as a policy because its intermediate computation is unobserved.
Keywords:causal inference, policy learning, retrieval-augmented generation, vector search,
nearest-neighbor matching, plug-in classifier, conditional average treatment effect, minimax
rate, in-context learning.
1 Introduction
Retrieval-augmented generation (RAG) has become a common architecture for action selection
in systems built on language models. In a RAG algorithm, a query is embedded in a vector
space, and documents or other information are retrieved according to distances in that space.
Conditional on the retrieved evidence, a generative model produces an answer, which may
include a recommendation or a tool choice.
Despite its widespread use, the decision-theoretic guarantees of RAG have not been fully
studied. If a query is treated as a covariate and the action recommended in the generated
∗Email:mkato-csecon@g.ecc.u-tokyo.ac.jp
†Email:taka@np-hard.co.jp
1
arXiv:2607.18225v1  [econ.EM]  20 Jul 2026

answer is treated as the decision, RAG-based decision-making can be viewed as a policy
learning problem. Action selection and policy learning have long been studied in causal
inference and reinforcement learning.
In causal inference, counterfactual decision-making has been studied as a policy learning
problem (Athey & Wager, 2021). The goal is to recommend the action with the highest
conditional expected outcome, so estimation of the conditional expected outcome is central.
Two main approaches are the plug-in approach and the counterfactual risk minimization
(CRM) approach (Swaminathan & Joachims, 2015); the latter is also called empirical wel-
fare maximization (Kitagawa & Tetenov, 2018). The plug-in approach first estimates the
conditional expected outcome for each action and then chooses the action with the largest
estimate. By contrast, the CRM approach estimates a value functional and learns a policy
by maximizing the estimated value.
1.1 Contents and Contributions
We formulate RAG-based action selection under the potential outcome framework, also known
as the Neyman–Rubin causal model. The explicit two-step procedure is a plug-in policy rule
because it estimates action-specific conditional expected outcomes before choosing an action.
We evaluate the one-step output as a policy without imposing the same interpretation on its
unobserved internal computation.
We first consider the case in which Kactions are given. Let Xdenote covariates
that represent the query or decision context supplied to the RAG system, let A∈ K :=
{0,1, . . . , K−1} denote the action, and let Y(a) denote the potential outcome under action
a. Our goal is to choose the action with the highest conditional expected outcome, and we
write the context-specific optimal action as
a∗(x)∈arg max
a∈KE[Y(a)|X=x].
Because vector search retrieves documents or examples that are similar or relevant to the
query, we interpret the RAG system as producing an estimate of a∗(x) from the context x.
When the retrieval database contains covariates, actions, and outcomes, vector search supplies
nearest neighbors used to estimate the conditional expected outcomes that determine a∗(x).
Similarity is therefore used to select matched evidence for each action, rather than to choose
the action that appears most often among neighboring cases. This operation corresponds to
nearest-neighbor matching in causal inference.
We next extend this formulation to settings in which the available action set is not
specified in advance. Although the preceding formulation assumes that Kactions are given,
the set of available actions is often not known in advance in applications of language models.
For example, in workplace applications, a user may submit a query and ask what to do
next without providing a candidate action set. To reflect this setting, we propose two forms
of RAG-based policy learning. The one-step method receives a query and directly returns
a recommended action, while the actions considered internally remain unobserved. The
two-step method proceeds as follows:
•We provide a query and ask the RAG system to generate a finite set of candidate
actions.
2

•For each candidate action, the RAG system retrieves matched evidence and evaluates
or ranks the action according to its estimated conditional expected outcome.
•We select the action with the largest estimated conditional expected outcome or the
highest rank.
The two-step method separates action generation from action choice. This separation allows
us to decompose its regret into the loss from the generated candidate set and the loss from
expected-outcome estimation or ranking within that set.
This study makes three contributions:
•We formulate RAG-based action selection as policy learning under the potential outcome
framework and give an explicit plug-in interpretation for the two-step procedure.
•We interpret action-specific vector search over observed cases as nearest-neighbor
matching and distinguish evidence retrieval from action choice.
•We decompose regret into candidate-set regret and within-candidate regret, and we
bound the latter using prediction-error guarantees.
1.2 Related Work
This study is related to three streams of literature: policy learning, RAG, and nonparametric
analysis of neural networks and transformers.
Policy learning.We study policy learning in causal inference (Swaminathan & Joachims,
2015; Kitagawa & Tetenov, 2018; Athey & Wager, 2021). As in supervised learning (Audibert
& Tsybakov, 2007), policy learning can be approached through plug-in estimation or counter-
factual risk minimization. The latter is also referred to as empirical welfare maximization
(EWM) (Kitagawa & Tetenov, 2018), and a theoretical comparison of the two approaches
is given by Kitagawa & Tetenov (2018). Although that comparison gives favorable results
for EWM relative to the plug-in approach under its conditions, their relative performance
depends on the underlying data-generating process, as discussed by Audibert & Tsybakov
(2007). Because the structure of RAG is more directly compatible with plug-in estimation,
we formulate RAG-based policy learning using the plug-in approach.
In particular, we interpret action-specific retrieval over observed cases as a nearest-
neighbor matching procedure. Nearest-neighbor match counts have density-ratio limits that
are proportional to inverse propensity scores, and nearest-neighbor matching can also be
related to Riesz regression (Lin et al., 2023; Kato, 2025b). The analyses of Kitagawa &
Tetenov (2018) and Athey & Wager (2021) use inverse probability weighting (IPW) or
augmented IPW estimators. The representation results in Lin et al. (2023) and Kato (2025b)
therefore connect RAG-based policy learning to those analyses.
RAG.Our study is also related to work on RAG. Guu et al. (2020) jointly trains a
latent document retriever and a language model. Lewis et al. (2020) combines a dense
passage index with a sequence-to-sequence generator and marginalizes over retrieved passages.
3

Dense passage retrieval maps queries and passages into a shared embedding space and
ranks passages by inner-product similarity (Karpukhin et al., 2020). This retrieval step is
computationally related to nearest-neighbor methods. In particular, the k-nearest-neighbor
language model constructs a nonparametric next-token distribution from nearby contextual
representations and interpolates it with the distribution produced by a parametric language
model (Khandelwal et al., 2020). Later studies develop different ways to incorporate retrieved
information into generation. Fusion-in-Decoder encodes retrieved passages separately and
combines their representations in the decoder (Izacard & Grave, 2021). REPLUG augments
a frozen black-box language model and trains the retriever using feedback from the language
model (Shi et al., 2024). Self-RAG learns when to retrieve and uses self-reflection signals
to assess whether the retrieved passages support the generated response (Asai et al., 2024).
These methods select external evidence according to proximity in a representation space,
but they do not interpret the retrieved items as matched observations under the potential
outcome framework.
Retrieval has also been incorporated into sequential decision-making. Goyal et al. (2022)
augments reinforcement learning agents with direct access to datasets of past experience.
Humphreys et al. (2022) uses approximate nearest-neighbor search over a large collection
of expert states to support offline reinforcement learning. REGENT conditions a generalist
policy on demonstrations retrieved from related tasks (Sridhar et al., 2025), whereas STRAP
retrieves relevant sub-trajectories before learning a policy for a target task (Memmel et al.,
2024). These studies place retrieval within reinforcement learning or imitation learning rather
than plug-in policy learning under the potential outcome framework. In a different direction,
CausalRAG incorporates causal graphs into retrieval and generation for knowledge-intensive
tasks (Wang et al., 2025). Its use of causal information concerns relations among concepts in
the retrieved documents rather than counterfactual outcomes under alternative actions.
Unlike these approaches, we formulate RAG-based action selection under the potential
outcome framework. For the two-step method, vector search supplies local evidence for
outcome regression, the generator estimates action-specific conditional expected outcomes or
their contrasts, and the algorithm selects the action with the largest estimate. The one-step
method is evaluated at the level of its returned policy because its internal criterion is not
observed. The two-step formulation also separates regret due to candidate generation from
regret due to expected-outcome estimation or ranking within the generated set.
Nonparametric analysis of neural networks and transformers.The statistical theory
of neural networks provides a basis for viewing transformers as nonparametric estimators.
Nonparametric regression with deep ReLU networks has been studied under compositional
assumptions and over Besov-type function classes (Schmidt-Hieber, 2020; Suzuki, 2019; Suzuki
& Nitanda, 2021). For transformers, Yun et al. (2020) establishes universal approximation
results for sequence-to-sequence functions, while Takakura & Suzuki (2023) and Havrilla &
Liao (2024) examine approximation and estimation for high-dimensional sequence inputs and
data with low-dimensional structure.
A related literature studies in-context learning as a statistical learning procedure. Existing
work examines the function classes that transformers can learn from in-context examples and
relates their predictions to regression algorithms, gradient descent, and algorithm selection
4

(Garg et al., 2022; Aky¨ urek et al., 2023; Von Oswald et al., 2023; Bai et al., 2023). More
recent studies analyze in-context learning for nonparametric regression and adaptation to
low-dimensional target functions (Kim et al., 2024; Oko et al., 2024; Ching et al., 2026).
We use prediction-error guarantees from this literature as inputs to the regret analysis for
transformer-based expected-outcome estimation.
2 Setup
LetX∈ X ⊆Rddenote covariates, which represent a query or decision context. The
covariates can combine the query with pre-action individual attributes, such as age, gender,
and occupation, when those attributes are available and relevant to the decision problem.
LetA∈ A denote an action, where the action space Acan be finite or uncountably infinite.
LetY∈ Y ⊆R denote a scalar outcome. We denote the conditional expected outcome of Y
givenA=a∈ AandX=x∈ Xbyf 0(a, x) :=E[Y|A=a, X=x].
2.1 Policy and Policy Value
We focus on deterministic policies. A policy is a measurable function
π:X → A.
Let Π be a class of such policies. Forπ∈Π, define its value by
V(π) :=E[f 0(π(X), X)].
We denote a learned or otherwise data-dependent policy by ba:X → A , so that ba(x) is
the action selected at covariate valuex. Its value is
V(ba) :=E[f 0(ba(X), X)].
Whenbadepends on training data, the retrieval database, or auxiliary algorithmic randomness,
the expectation also averages over this randomness.
For a chosen actionba(·), we define its value by
V(ba) :=E[f 0(ba(X), X)].
2.2 Optimal Policy
We assume that the maximum of f0(a, x) over a∈ A is attained and that a measurable
maximizer can be selected. For eachx∈ X, let
a∗(x)∈arg max
a∈Af0(a, x).
The measurable selectora∗:X → Ais the optimal deterministic policy. Its value is
V(a∗) =E
max
a∈Af0(a, X)
.
5

2.3 Regret
The regret ofbarelative to the optimal action rulea∗is
R(ba) :=V(a∗)−V(ba) =E[f 0(a∗(X), X)−f 0(ba(X), X)].
3 RAG-PL
In this section, we describe our proposed method, called RAG-based policy learning (RAG-
PL). We consider two forms of RAG-PL. The one-step method returns an action directly from
the covariates. The two-step method first generates candidate actions and then selects one
action from those candidates. The two-step method has two stages. The first stage constructs
the candidate set. The second stage evaluates the candidates using retrieved evidence and
selects one of them.
3.1 RAG
We consider three input-output forms of a RAG system. The first returns a set of candidate
actions given the covariates. he second returns an estimated conditional expected outcome for
each action or ranks the candidate actions by that quantity. The third returns a recommended
action directly from the covariates.
Action-set RAG.Let F(A):={C⊆ A: 1≤ |C|<∞} denote the collection of all non-
empty finite subsets of A. We represent an action-set RAG as a set-valued function g:X →
F(A). For each x∈ X , the set g(x)⊆ A contains the candidate actions returned by the RAG
system.
Vector search as nearest-neighbor matching.Let φ:X →Rdφbe the embedding
map used for vector search, and write H=φ(X). Let D={Dj}Ndb
j=1be a retrieval database.
When Ais discrete and the database contains observations of covariates, actions, and
outcomes, we write Dj= (Xj, Aj, Yj) and, for an action a, define the action-specific index set
Ia:={j:A j=a}. Given xanda, letNk,a(x) be the indices of the kobservations indexed by
Iawhose embeddings are closest to φ(x), and let Rk(x, a):={Dj:j∈ N k,a(x)} denote the
retrieved evidence. We assume that |Ia| ≥k for every action evaluated through action-specific
retrieval. This operation matches the current context with similar cases within each action,
as in nearest-neighbor matching. For a general document corpus, Rk(x, a) denotes evidence
retrieved by a query that includes the candidate action. In either case, similarity determines
which evidence is used to evaluate an action. It is not itself the criterion for choosing the
final action, and we do not select an action by the frequency with which it appears among
neighboring cases. For continuous action spaces, exact action-specific retrieval based on
Ia={j:A j=a} is generally unavailable. In that case, retrieval must use an action-aware
distance or kernel on A × X , or another procedure that borrows information across nearby
actions. The nearest-neighbor analysis in Section 5 is restricted to binary actions.
6

Expected-outcome RAG or ranking RAG.We represent an expected-outcome RAG
by a function bf:A × X →R . Given an action a∈ A and covariates x∈ X , it returns
an estimate bf(a, x) of the conditional expected outcome f0(a, x). When the generator and
retrieved evidence are written explicitly, we use
bf(a, x) =G θ 
x, a,R k(x, a)
.
A ranking RAG instead receives xand a finite candidate set C={a1, . . . , a m} ∈F (A).
It retrieves evidence for the candidate actions and returns an ordered tuple r(x, C) =
(a(1), . . . , a (m)), which is a permutation of the actions in Cfrom the highest to the lowest
estimated conditional expected outcome.
Remark(Fixed action set).When the action set is given and finite, we write A=K:=
{0,1, . . . , K−1} . These actions are also called treatments or arms. In the potential-outcome
notation, each unit has potential outcomes Y(0), Y(1), . . . , Y (K−1). We use the special case
K= 2when discussing the relation between the selected action and the conditional average
treatment effect.
Policy RAG.A policy RAG receives covariates and directly returns a recommended action.
We represent it by a functionπ RAG:X → A.
3.2 One-Step RAG-PL
The one-step RAG-PL method applies a policy RAG directly. Given x∈ X , the RAG system
is instructed to recommend an action baone(x) with the highest conditional expected outcome,
and we write baone(x) =πRAG(x). The policy implemented by the RAG system is usually not
observed separately from its output. In particular, the one-step method does not expose an
intermediate candidate set, expected outcomes for the candidate actions, or a separate ranking
step. We therefore evaluate the returned policy without imposing a particular interpretation
on its internal computation.
3.3 Two-Step RAG-PL
Action-set generation.Givenx∈ X, the action-set RAG returns a finite candidate set
g(x) =
a1(x), . . . , a M(x)(x)	
⊆ A,1≤M(x)<∞.
Thus, even when Ais uncountably infinite, the subsequent evaluation or ranking is performed
only over the finite set g(x). The candidate set can contain actions found in matched cases
and actions generated by the language model. Its purpose is to include actions with high
conditional expected outcomes, rather than to reproduce the empirical distribution of actions
in the retrieved evidence.
Ranking or expected-outcome estimation.For every a∈g (x), the RAG system re-
trieves Rk(x, a). When an expected-outcome RAG is used, it evaluates bf(a, x) =Gθ(x, a,R k(x, a)).
When a ranking RAG is used, it returns the ordered tuple r(x, g(x)) = ( a(1)(x), . . . , a (M(x)) (x)),
wherea (1)(x) is the candidate with the highest estimated conditional expected outcome.
7

Action choice.Under expected-outcome estimation, we choose
batwo(x)∈arg max
a∈g(x)bf(a, x),
where a fixed rule is used to break ties. Under ranking, we choose
batwo(x):=a (1)(x).
Both methods define a policy from XtoA. The two-step method makes action generation,
retrieval of similar cases, and action choice explicit, which allows their contributions to regret
to be studied separately.
4 Regret Analysis
We analyze the regret of the one-step and two-step methods under the setup above. We write
R(ba) for the regret of a selected action ba(·). When the candidate set, the expected-outcome
estimate, or the selected action is random, the expectation defining R(ba) and all expectations
below also average over this randomness. We first separate the effect of the generated
candidate set from the effect of selecting an action within that set. We then connect the
second term to nonparametric prediction error. Section 5 studies the case in which vector
search is represented explicitly by nearest-neighbor matching.
4.1 Regret Decomposition for Two-Step RAG-PL
For eachx∈ X, let
a∗
g(x)∈arg max
a∈g(x)f0(a, x),
where the fixed tie-breaking rule is used when needed, and we assume that the resulting
selector is measurable. The maximum is attained becauseg(x) is finite. We define
Rgen(g):=E
f0(a∗(X), X)−f 0(a∗
g(X), X)
and
Rchoice(batwo;g):=E
f0(a∗
g(X), X)−f 0(batwo(X), X)
.
The first term is the regret caused by restricting the choice to g(X). The second is the
regret caused by expected-outcome estimation or ranking within g(X). We call Rgen(g) the
candidate-set regret andR choice(batwo;g) the within-candidate regret.
Adding and subtractingf 0(a∗
g(X), X) gives
R(ba two) =R gen(g) +R choice(batwo;g).(1)
This identity does not require independence between candidate generation and action choice.
8

4.2 Bounding the Candidate-Set Regret
Forε≥0, define the set ofε-optimal actions by
A∗
ε(x):={a∈ A:f 0(a∗(x), x)−f 0(a, x)≤ε}.
Exact inclusion of a∗(x) is not required. It is enough that the generated set contain an action
whose conditional expected outcome is close to the optimum.
Proposition 4.1(Bounds for the generated candidate set).The following statements hold.
1. Ifg(x)∩ A∗
ε(x)̸=∅holds forP X-almost everyx, then we have
Rgen(g)≤ε.
2. Suppose that0≤f 0(a∗(x), x)−f 0(a, x)≤B ffor every(a, x)∈ A × X. If
Pr (g(X)∩ A∗
ε(X) =∅)≤δ
holds, then we have
Rgen(g)≤ε+B fδ.
3.LetdAbe a metric on A, and define dA(a, C):=min b∈CdA(a, b)for every non-empty
finite setC. Suppose that, for someL A>0andβ A>0,
|f0(a, x)−f 0(b, x)| ≤L AdA(a, b)βA
holds for everya, b∈ Aandx∈ X. Then, we have
Rgen(g)≤L AE
dA(a∗(X), g(X))βA
.
The third statement controls the value of the best action in g(x) through its distance
from a∗(x). It does not concern the estimation error of bf. The smoothness condition on f0
converts distance in the action space into a difference in conditional expected outcomes.
4.3 Expected-Outcome Estimation and Ranking
Suppose that the two-step method chooses
batwo(x)∈arg max
a∈g(x)bf(a, x).
For eachx∈ X, define
δf(x;g) := max
a∈g(x)|bf(a, x)−f 0(a, x)|.
Letv∗
g(x):=max a∈g(x) f0(a, x) and define the smallest positive gap within the candidate set
by
∆g(x):= min
a∈g(x):f 0(a,x)<v∗g(x)
v∗
g(x)−f 0(a, x)	
,
where ∆ g(x) = +∞if every action ing(x) attainsv∗
g(x).
9

Theorem 4.2(Regret from expected-outcome estimation).For everyx∈ X, it holds that
f0(a∗
g(x), x)−f 0(batwo(x), x)≤2δ f(x;g).
Consequently,
R(ba two)≤R gen(g) + 2E[δ f(X;g)].
Suppose in addition that δf(X;g)≤rfalmost surely and that, for some CM>0,κ≥0, and
t0>0,
Pr (0<∆ g(X)≤t)≤C Mtκ,0< t≤t 0.
If2r f≤t0, then
Rchoice(batwo;g)≤C M(2rf)1+κ.
The theorem is stated for the realized set g(X) and therefore does not require candidate
generation to be independent of bf. A high-probability error bound can be used in the same
way. If 0 ≤f 0(a∗
g(X), X)−f0(a, X)≤B fholds almost surely for every a∈g (X), the margin
condition holds, and
Pr (δ f(X;g)> r f)≤η f,
then
Rchoice(batwo;g)≤C M(2rf)1+κ+B fηf.(2)
For example, suppose that|g(X)| ≤Mand, conditional onXandg(X),
Pr
|bf(a, X)−f 0(a, X)|> b n+uX, g(X)
≤c1exp(−c 2anu2)
for everya∈g(X). A union bound gives
Pr (δ f(X;g)> b n+u)≤c 1Mexp(−c 2anu2).
Thus, the number of candidates enters the simultaneous error through a logarithmic term
whenuis chosen to make the right-hand side small.
For a ranking RAG, leta (1)(x) be the first action inr(x, g(x)). Then,
R(a (1)) =R gen(g) +E
f0(a∗
g(X), X)−f 0(a(1)(X), X)
.
When the ranking orders the candidates according to bf(a, x), Theorem 4.2 applies. If the RAG
system returns only an ordering, its analysis requires a direct bound on the true conditional
expected-outcome difference between the best candidate and the first-ranked candidate.
4.4 Relation to Policy-Value Maximization
Let
Sg:={s:X → A:sis measurable ands(x)∈g(x)}
and define the value induced by bfas
bVf(s):=Eh
bf(s(X), X)i
.
10

If a measurable selectorbs fsatisfies
bsf(x)∈arg max
a∈g(x)bf(a, x),
then, for everys∈ S g, it holds that bf(bsf(x), x)≥ bf(s(x), x). Hence,
bsf∈arg max
s∈SgbVf(s).
The same pointwise argument applies to an empirical average over target contexts. Thus,
when the policy class contains every measurable selector and its value is constructed from bf,
pointwise maximization and maximization over the policy class are the same optimization
problem. They can differ when the policy class is restricted or when the value estimator uses
observed outcomes through IPW or AIPW rather than being constructed from bfalone.
4.5 Substitution of Nonparametric Rates
The preceding results allow a prediction rate to be inserted into the regret bound without
changing the RAG-PL algorithm. Suppose that Rgen(g)≤rg,nand that the expected-outcome
error on the generated set is bounded by rf,n. Under the finite-candidate margin condition in
Theorem 4.2, we have
R(ba two)≤r g,n+C M(2rf,n)1+κ.
For example, if a nonparametric estimator has a uniform error boundr f,n≍n−α/(2α+d), the
second term is of order n−α(1+κ)/(2α+d). This substitution is valid only when the available
prediction result controls the error required by the regret theorem. An integrated mean-
squared error cannot be used as a uniform error without an additional argument.
The same distinction matters for transformers. The analysis of Kim et al. (2024) gives a
mean-squared prediction bound for a transformer composed of a learned neural representation
and a linear-attention layer. In the Besov setting, a representative bound has the form
qICL≲N−2α/d
rep +NreplogN rep
nctx+N2
replogN rep
Tpre,
where Nrepis the representation dimension, nctxis the number of in-context examples, and
Tpreis the number of pretraining tasks. If this bound holds for both binary expected-outcome
estimates, Theorem 4.3 below gives
R(ba)≲(q ICL,0+qICL,1)(1+κ)/(2+κ).
A sharper exponent 1 + κfor the root prediction rate requires a uniform error bound or
a pointwise exponential-deviation bound. It does not follow from an MSE result alone.
The results of Oko et al. (2024) and Ching et al. (2026) further describe adaptation to
low-dimensional target functions and minimax MSE rates, and they enter the regret analysis
according to the same distinction between error types.
The general regret decomposition itself has no sample size. For a specified retrieval
database, its size and the number of matched observations have a direct statistical meaning.
11

Transformer theory uses different quantities for the representation dimension, in-context
examples, and pretraining tasks. For an already trained RAG system whose training data
are not observed, the total number of pretraining tokens is not by itself a sample size for
conditional expected-outcome estimation. In that case, the regret bound should be stated in
terms of a prediction error that is either assumed or evaluated separately.
4.6 Relation Between One-Step and Two-Step RAG-PL
The regret of the one-step output is
R(ba one) =E[f 0(a∗(X), X)−f 0(baone(X), X)].
For comparison, its observed output can be represented by the singleton set gone(x) =
{baone(x)}. This representation does not assert that the RAG system uses a singleton set
internally.
Suppose that baone(x)∈g(x) for PX-almost every x. Because a∗
g(x) is at least as good as
every action ing(x), we have
Rgen(g)≤R(ba one).
Combining this inequality with (1) gives
R(ba two)≤R(ba one) +R choice(batwo;g).
If the two-step method selects the best action in g(x) according to f0, its regret is no larger
than the regret of the one-step output. With an estimated expected outcome, Theorem 4.2
instead gives
R(ba two)≤R(ba one) + 2E[δ f(X;g)].
Thus, a larger candidate set can contain a better action than the one-step output, but the
gain can be lost when expected-outcome estimation or ranking is inaccurate.
4.7 Fixed Action Sets and Binary Actions
When the action set is fixed and finite, we can set g(x) =A=Kfor every x. Then,
Rgen(g) = 0, and Theorem 4.2 gives
R(ba two)≤2E
max
a∈K|bf(a, X)−f 0(a, X)|
.
For the binary action setK={0,1}, define
τ0(x):=f 0(1, x)−f 0(0, x),bτ(x) :=bf(1, x)− bf(0, x).
Using the tie-breaking rule that selects action 1 at equality, we have a∗(x) = 1[τ0(x)≥0]
andba(x) =1[bτ(x)≥0]. For any selected actionba(x)∈ {0,1},
R(ba) =E[|τ 0(X)|1[ba(X)̸=a∗(X)]].(3)
Whenf 0(a, x) =E[Y(a)|X=x],τ 0(x) is the conditional average treatment effect.
12

We use the margin condition
Pr (0<|τ 0(X)| ≤t)≤C Mtκ,0< t≤t 0.(4)
It controls the probability of contexts for which the two actions have nearly equal conditional
expected outcomes (Audibert & Tsybakov, 2007).
Theorem 4.3(Binary regret bounds).Suppose that(4)holds.
1. If∥bτ−τ 0∥∞≤rτholds almost surely andr τ≤t0, then
R(ba)≤C Mr1+κ
τ.
In particular, if max a∈{0,1} |bf(a, x)−f0(a, x)| ≤r fforPX-almost every x, then R(ba)≤
CM(2rf)1+κwhenever2r f≤t0.
2.Suppose that |τ0(X)| ≤B τholds almost surely and that there exist bn≥0,an>0, and
δn≥0, together with constantsc 1, c2>0, such that
Pr (|bτ(X)−τ 0(X)| ≥b n+u|X=x)≤c 1exp(−c 2anu2) +δ n
holds for PX-almost every x∈ X and every u >0. If bn≤t0/4and a−1/2
n≤t0/4, then
R(ba)≤C(b n+a−1/2
n)1+κ+B τδn.
3. If
E
(bτ(X)−τ 0(X))2
≤qn
andq1/(2+κ)
n ≤t0hold, then
R(ba)≤Cq(1+κ)/(2+κ)
n .
The constants in the second and third statements depend only on the constants displayed in
their assumptions and the margin condition.
5Nearest-Neighbor Matching and Expected-Outcome
Regret
Vector search connects RAG-PL with causal inference by selecting observations whose
covariates are close to the current context. This section gives an explicit nearest-neighbor
regression model in which the expected-outcome estimate in Section 3 is formed from matched
observations and its error can be substituted into the regret bounds above.
13

5.1 Nearest-Neighbor Matching
For eacha∈ {0,1}, let
Da={(H a,j, Ya,j)}Na
j=1
be a database for action a, where Ha,1, . . . , H a,Naare independently and identically distributed
according toQ aonRdφand
Ya,j=m a(Ha,j) +ξ a,j.
We assume that the query Xis independent of the action-specific databases. For a query x,
write h=φ(x) and let Nk,a(h) be the indices of the knearest embeddings in Da. Define the
matched expected-outcome estimate by
efk(a, x) :=1
kX
j∈Nk,a(h)Ya,j.
The following result gives a standard bias and variance calculation for nearest-neighbor
regression in the notation used by RAG-PL (Jiang, 2019). Its use as an action-specific
matching estimator follows the setup of Abadie & Imbens (2006).
Theorem 5.1(Estimation error of the expected outcome from nearest-neighbor matching).
Fixa∈ {0,1} . Suppose that the support of Ha,jhas finite diameter DHand that the following
conditions hold.
1. The functionm aisβ-H¨ older for someβ∈(0,1]andL a>0:
|ma(h)−m a(h′)| ≤L a∥h−h′∥β.
2. For somec a>0andr 0>0, every query embeddinghand every0< r≤r 0satisfy
Qa(B(h, r))≥c ardφ,
whereB(h, r) :=
h′∈Rdφ:∥h′−h∥ ≤r	
.
3.Conditional on the embeddings, the errors ξa,jare independent, mean zero, and σ2
a-sub-
Gaussian for someσ a>0.
Let
ra,k:=2k
caNa1/dφ
, b φ,a:= sup
x∈X|ma(φ(x))−f 0(a, x)|,
and suppose thatr a,k≤r0. Then, for everyx∈ Xandu >0, we have
Pr
|efk(a, x)−f 0(a, x)| ≥b φ,a+L arβ
a,k+u
≤2 exp
−ku2
2σ2
a
+ exp(−k/4).
Moreover, we have
Eh
(efk(a, X)−f 0(a, X))2i
≤C 
b2
φ,a+k
Na2β/d φ
+1
k+ exp(−k/4)!
.
14

The constantCdepends only on the constants in the assumptions.
The term bφ,ais zero when the embedding retains all information needed for the conditional
expected outcome, so f0(a, x) =ma(φ(x)). Otherwise, it records the difference between
conditioning on Xand conditioning on its embedding. The local-mass condition requires
enough observations under action anear each query. It complements the positivity condition
used for causal identification.
The displayed estimator is a benchmark for the expected-outcome RAG. Suppose that
the RAG output satisfies
max
a∈{0,1}sup
x∈X|bf(a, x)− efk(a, x)| ≤η n.
Suppose that the margin condition (4)holds. For the deviation-based bound below, also
suppose that |τ0(X)| ≤B τholds almost surely. Let Nmin=min{N 0, N1}andbφ=bφ,0+bφ,1.
Combining Theorem 5.1 with Theorem 4.3 gives
R(ba)≤C 
bφ+k
Nminβ/dφ
+k−1/2+ 2η n!1+κ
+ 2B τexp(−k/4),(5)
provided that the term in parentheses is sufficiently small relative to t0. The MSE statement
gives the alternative bound
R(ba)≤C 
b2
φ+k
Nmin2β/d φ
+1
k+η2
n!(1+κ)/(2+κ)
.(6)
Choosing k≍N2β/(2β+d φ)
min balances the nearest-neighbor bias and variance. In (5), this gives
the root prediction rate N−β/(2β+d φ)
min before the margin exponent is applied. In (6), the term
exp(−k/4) is absorbed into 1/k.
For a general document corpus without observed outcomes, Theorem 5.1 does not apply
directly. In that case, the theorem describes the matching calculation that the RAG output
is meant to approximate, and a regret rate requires an assumption or an evaluation of the
difference represented byη n.
5.2 Matching Weights and Propensity Scores
Similarity is used to select comparable evidence, not to choose the action that appears most
often in nearby cases. Let NH
k(h) denote the indices of the knearest observations to hin the
pooled embedding database. If
beH
k(a|h) =1
kX
j∈NH
k(h)1[A j=a],
thenbeH
k(a|h) estimates Pr(A=a|H =h) under standard nearest-neighbor conditions. At
h=φ(x), this quantity equals Pr(A=a|X =x) only under an additional condition such
asA⊥X|H . Maximizing this quantity selects the most common past action near x, not
15

the action with the largest conditional expected outcome. The joint distribution of ( X, A)
can remain unchanged while the conditional outcomes are changed so that either action is
optimal. Thus, observations of contexts and actions alone do not determine the optimal
policy.
Nearest-neighbor matching also has a weighting representation. Let Xtar
1, . . . , Xtar
mbe
target contexts and define
efk(a, Xtar
i) =1
kNaX
j=11
j∈ N k,a(φ(Xtar
i))
Ya,j.
Define
Ka,j:=mX
i=11
j∈ N k,a(φ(Xtar
i))
.
Then, changing the order of summation gives
1
mmX
i=1efk(a, Xtar
i) =NaX
j=1Ka,j
mkYa,j.
The match count therefore acts as a weight when local predictions are averaged over target
contexts. For the propensity-score interpretation, suppose that the action-specific databases
are sampled from the same observational population, so Qa=PH|A=a , and that the target
embeddings followP H. Bayes’ rule then gives
pH(h)
pH|A=a (h)=Pr(A=a)
Pr(A=a|H=h).
The density ratio is therefore proportional to an inverse propensity score defined conditional
on the embedding. The density ratio is proportional to an inverse propensity score. Lin
et al. (2023) studies the density-ratio limit of nearest-neighbor match counts, and Kato
(2025b) relates nearest-neighbor matching to least-squares density-ratio estimation and Riesz
regression. These results connect matching with IPW and AIPW representations when
outcomes are available. They do not make the propensity score an expected outcome.
6 Discussion
6.1 Identification
The regret analysis is written in terms of f0(a, x) =E[Y|A=a, X=x] . A causal interpreta-
tion requires consistency, conditional exchangeability, and positivity. Under these conditions,
it holds that
E[Y|A=a, X=x] =E[Y(a)|X=x].
The information contained in Xis therefore part of the identification argument. It is natural
to write X= (Q, W ), where Qis the query and Wcontains variables observed before
the action, such as age, gender, occupation, and prior decisions. If a variable affects both
16

past action selection and the outcome, omitting it can prevent the conditional comparison
from having a causal interpretation. Including more variables does not establish conditional
exchangeability by itself, and post-action variables should not be used as controls.
Vector search is usually performed on an embedding H=φ(X). Matching on His
sufficient only when the representation retains the information required for adjustment or
for the conditional expected outcomes. The GenAI-Powered Inference framework combines
structured covariates with low-dimensional features extracted from unstructured data (Imai
& Nakamura, 2025). This suggests keeping personal attributes explicit while representing the
query text through an embedding. Embedding-powered BISG provides an example in which
embeddings are used to estimate a missing personal attribute probabilistically (Dasanaike
& Imai, 2026). Such an estimate is a probabilistic proxy rather than an observed covariate,
so its estimation error and possible distribution shift remain relevant. The variables used
for adjustment can also be richer than the variables allowed in the final policy. In that case,
expected outcomes are first adjusted using the larger information set and then averaged over
variables excluded from the policy.
Instrumental variables provide another identification strategy when conditional exchange-
ability fails in the data used to estimate expected outcomes. They enter through the
source-data model, not by treating the RAG output as an instrument. Existing studies
consider optimal treatment regimes under additional IV assumptions (Cui & Tchetgen, 2021;
Qiu et al., 2021) and decision rules under partial identification (Pu & Zhang, 2021). These
problems have different target values. Applying RAG-PL to them would require the expected
outcome supplied to the RAG system to be defined for the corresponding target.
6.2 Expected Outcome and Policy-Value Formulations
When bf(a, x) is available and the policy class contains all measurable selectors, maximizing
bf(a, x) at each context is equivalent to maximizing Eh
bf(s(X), X)i
over the policy class. The
same equivalence holds for an empirical average over target contexts. A restricted policy
class can introduce an approximation loss, but the criterion is still constructed from the same
expected-outcome estimate. The relation between EWM and least-squares CATE estimation
for a reparameterized binary policy class is studied by Kato (2025a).
A different analysis is needed when the policy value is constructed from observed outcomes
rather than from bfalone. IPW uses weighted outcomes, and AIPW combines an expected-
outcome estimate with an outcome-residual correction (Kitagawa & Tetenov, 2018; Athey
& Wager, 2021). In the main setting of this study, the analyst starts from a trained RAG
system and may not observe unit-level outcomes and assignment probabilities. The fact
that outcome-related text may have appeared during pretraining does not provide the data
required to construct IPW or AIPW. If a retrieval database contains unit-level observations of
(X, A, Y ), IPW and AIPW can instead be constructed directly from those data. For one-step
RAG-PL, the internal criterion is not observed, so we evaluate the returned policy without
assigning it to one of these formulations.
17

6.3 Margin Conditions
The margin condition is separate from identification. Identification determines whether the
contrast has a causal interpretation, while the margin condition controls how often estimation
error changes the preferred action. In the binary case, it bounds the probability that |τ0(X)|
is close to zero. When few contexts lie near this boundary, the regret decreases faster than
the expected-outcome error.
The margin condition depends on the information used to define the conditional expected
outcomes. Omitting a relevant attribute can average positive and negative effects into a
contrast near zero, while adding an attribute can also reveal finer effects that lie near zero.
Replacing Xby an embedding can change the margin for the same reason. For two-step
RAG-PL, the margin within g(x) is also distinct from the quality of the generated set. A
large gap among poor candidates does not removeR gen(g).
6.4 Action Generation in Large Action Spaces
When Ais finite and given, Rgen(g) = 0. For a large or uncountable action space, the
generated set determines which actions can be compared. Exact inclusion of a∗(x) is stronger
than necessary; Proposition 4.1 uses an ε-optimal action, a coverage probability, or distance
in the action space. These conditions describe the candidate set through the best conditional
expected outcome it contains.
Increasing the number of candidates can improve the chance of including a good action,
but more expected outcomes must then be estimated or ranked. Equation (1)keeps these
effects separate. A useful candidate-set size depends on both the quality of generation and
the accuracy of the second step.
7 Simulation Studies
We evaluate whether action-specific retrieval of cases with similar pre-action information
improves policy choice. Each data-generating process first produces a structured pre-action
state together with the conditional expected outcome under every available action. We
convert the structured states into complete English queries and case reports before executing
the notebooks. The RAG procedures observe only the resulting texts, while the numerical
state is retained to calculate the optimal action and regret. Because each numerical level
is mapped deterministically to a verbal category, the text retains the full discrete state
information. Appendix G gives the full specifications.
Data-generating processes.DGP 1 and DGP 2 use two actions. Action 0 is temporary
workload reduction and action 1 is weekly individual coaching. Let X= (X1, . . . , X 6) describe
recent performance; workload; task complexity; experience; schedule flexibility; and team
support. Each component is independently and uniformly distributed on {−2,−1,0,1,2} .
DGP 1 has a linear conditional effect and observational action assignment. DGP 2 has
randomized action assignment and a smooth nonlinear conditional effect whose population
mean is zero. The preferred action therefore varies with the current state even though neither
18

constant action is favored on average. DGP 3 contains 24 interventions formed from four
intervention types with three intensity levels and two durations. Its conditional expected
outcome rewards agreement between the employee’s stated needs and the intervention profile.
Methods and implementation.The main results use GPT-5.4 mini with temperature
zero. Retrieval vectors are obtained from text-embedding-3-small . The embedding input
contains only a fixed-order description of the pre-action information. It excludes the historical
action and the observed outcome. It also excludes the decision question. Cosine distance is
used after normalization.
In DGP 1 and DGP 2, One-step RAG-PL and Two-step RAG-PL receive the same
current query and the same six nearest reports under each action. The twelve reports are
presented in alternating action order. The prompt also gives the exact mean of the six
displayed scores under each action. Providing these means allows the model to use exact
local sample averages, so the experiment does not test whether it can calculate those averages
from the reports. One-step RAG-PL returns an action directly. Two-step RAG-PL returns
the two action-specific conditional expected outcomes, and the experiment code selects the
larger estimate. RAG without covariates receives neither the current profile nor historical
pre-action profiles. It uses six randomly sampled action-and-outcome reports under each
action and returns one population-level decision for the retrieval database. That decision
is applied to all ten test queries in the repetition. RAG without covariates is therefore a
population-level baseline that differs from RAG-PL in both the current-state information
supplied and the retrieval rule. In DGP 3, both RAG-PL procedures receive the same twelve
nearest reports and the complete catalog of 24 actions. One-step RAG-PL returns one action
directly. Two-step RAG-PL generates five distinct candidate actions. It retrieves five reports
observed under each candidate and estimates the five candidate-specific conditional expected
outcomes. It then selects the candidate with the largest estimate.
Evaluation.Each DGP is repeated 20 times. Each repetition contains ten test queries.
DGP 1 and DGP 2 use 500 historical reports, while DGP 3 uses 600. For each repetition,
we first average regret and policy value over its ten queries. Table 1 reports the mean
of these repetition-level quantities. The regret standard error is calculated across the 20
repetitions. The optimal-action probability counts any action attaining the largest true
conditional expected outcome as optimal. Figures 1–3 show the repetition-level mean regret.
Results.In DGP 1, One-step RAG-PL and Two-step RAG-PL both reduce mean regret
from 4 .5450 under RAG without covariates to 3 .3700. The paired mean-regret reduction is
1.1750 for each procedure, with standard errors of 0 .3562 for One-step RAG-PL and 0 .3635
for Two-step RAG-PL. The optimal-action probability rises from 0 .4100 under RAG without
covariates to 0.5500 under One-step RAG-PL and 0.5550 under Two-step RAG-PL.
DGP 2 isolates personalization because historical actions are randomized. One-step
RAG-PL has mean regret 1 .5676 and Two-step RAG-PL has mean regret 1 .6071, compared
with 2 .3638 for RAG without covariates. The paired reductions relative to RAG without
covariates are 0 .7963 with standard error 0 .2257 and 0 .7567 with standard error 0 .2473,
19

Table 1: Main simulation results with GPT-5.4 mini
Method Regret Standard error Policy value Optimal-action probability
DGP 1
One-step RAG-PL3.37000.420066.69250.5500
Two-step RAG-PL3.37000.426666.6925 0.5550
RAG without covariates 4.5450 0.3943 65.5175 0.4100
DGP 2
One-step RAG-PL1.56760.156465.7886 0.6400
Two-step RAG-PL 1.6071 0.1746 65.74910.6400
RAG without covariates 2.3638 0.2192 64.9924 0.5100
DGP 3
One-step RAG-PL 1.4985 0.1134 65.1192 0.0950
Two-step RAG-PL1.21540.083765.4023 0.1400
respectively. Both RAG-PL procedures select an optimal action with probability 0 .6400,
compared with 0.5100 under RAG without covariates.
In DGP 3, Two-step RAG-PL lowers mean regret from 1 .4985 to 1 .2154. The paired
difference, Two-step RAG-PL minus One-step RAG-PL, is −0.2831 with standard error 0 .1431.
The regret decomposition for Two-step RAG-PL gives candidate-set regret 0 .4822 and within-
candidate regret 0 .7332. Thus, both candidate generation and selection among the generated
candidates contribute to the remaining regret. The generated set contains an exactly optimal
action with probability 0 .4450 and an action within 0 .5 of the optimum with probability
0.6700. Additional estimation and robustness results are reported in Appendix G.9.
8 Conclusion
We proposed one-step and two-step RAG-based policy learning methods. We interpreted the
vector-search component of RAG as nearest-neighbor matching and related this matching
procedure to propensity-score weighting, density-ratio estimation, and Riesz regression. This
interpretation connects RAG-based decision-making to the existing literature on policy
learning. We then derived regret upper bounds for the proposed methods by combining
policy-learning analysis with nonparametric prediction-error bounds for nearest-neighbor
estimators and transformers. In the simulation studies, both RAG-PL methods had lower
mean regret than RAG without covariates in the binary settings, and the two-step method
had lower mean regret than the one-step method in the 24-action setting.
References
Alberto Abadie and Guido W. Imbens. Large sample properties of matching estimators for
average treatment effects.Econometrica, 74(1):235–267, 2006. 14
Ekin Aky¨ urek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What
20

One-step RAG PL T wo-step RAG PLRAG without covariates
Method012345678RegretFigure 1: Repetition-level mean regret in DGP 1. The diamond denotes the mean across
repetitions.
learning algorithm is in-context learning? investigations with linear models. InInternational
Conference on Learning Representations (ICLR), 2023. 5
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learn-
ing to retrieve, generate, and critique through self-reflection. InInternational Conference
on Learning Representations (ICLR), 2024. 4
Susan Athey and Stefan Wager. Policy learning with observational data.Econometrica, 89
(1):133–161, 2021. 2, 3, 17
Jean-Yves Audibert and Alexandre B. Tsybakov. Fast learning rates for plug-in classifiers.
The Annals of Statistics, 35(2):608–633, 2007. 3, 13
Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians:
Provable in-context learning with in-context algorithm selection. InWorkshop on Efficient
Systems for Foundation Models at ICML2023, 2023. 5
Michelle Ching, Ioana Popescu, Nico Smith, Tianyi Ma, William G. Underwood, and Richard J.
Samworth. Efficient and minimax optimal in-context nonparametric regression with
transformers. InInternational Conference on Machine Learning (ICML), 2026. 5, 11, 35
Yifan Cui and Eric Tchetgen Tchetgen. A semiparametric instrumental variable approach
to optimal treatment regimes under endogeneity.Journal of the American Statistical
Association, 116(533):162–173, 2021. 17
Noah Dasanaike and Kosuke Imai. Using embedding models to improve probabilistic race
prediction, 2026. arXiv: 2604.22555. 17, 32
21

One-step RAG PL T wo-step RAG PLRAG without covariates
Method0.51.01.52.02.53.03.54.04.5RegretFigure 2: Repetition-level mean regret in DGP 2. The diamond denotes the mean across
repetitions.
Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers
learn in-context? a case study of simple function classes. InInternational Conference on
Neural Information Processing Systems (NeurIPS), 2022. 5
Anirudh Goyal, Abram Friesen, Andrea Banino, Theophane Weber, Nan Rosemary Ke,
Adri` a Puigdom` enech Badia, Arthur Guez, Mehdi Mirza, Peter C Humphreys, Ksenia
Konyushova, Michal Valko, Simon Osindero, Timothy Lillicrap, Nicolas Heess, and Charles
Blundell. Retrieval-augmented reinforcement learning. InInternational Conference on
Machine Learning (ICML), pp. 7740–7765, 2022. 4
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm:
retrieval-augmented language model pre-training. InInternational Conference on Machine
Learning (ICML), 2020. 3
Alexander Havrilla and Wenjing Liao. Understanding scaling laws with statistical and
approximation theory for transformer neural networks on intrinsically low-dimensional
data. InAnnual Conference on Neural Information Processing Systems (NeurIPS), 2024.
4, 35
Peter C. Humphreys, Arthur Guez, Olivier Tieleman, Laurent Sifre, Th´ eophane Weber,
and Timothy Lillicrap. Large-scale retrieval for reinforcement learning. InInternational
Conference on Neural Information Processing Systems (NeurIPS), 2022. 4
Kosuke Imai and Kentaro Nakamura. Genai-powered inference, 2025. arXiv: 2507.03897. 17
22

One-step RAG PL T wo-step RAG PL
Method0.51.01.52.02.5RegretFigure 3: Repetition-level mean regret in DGP 3. The diamond denotes the mean across
repetitions.
Kosuke Imai and Kentaro Nakamura. Causal inference with generative artificial intelligence:
Application to texts as treatments, 2026. arXiv: 2410.00903. 32
Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for
open domain question answering. InProceedings of the 16th Conference of the European
Chapter of the Association for Computational Linguistics: Main Volume, 2021. 4
Heinrich Jiang. Non-asymptotic uniform rates of consistency for k-nn regression.Proceedings
of the AAAI Conference on Artificial Intelligence, 33(01):3999–4006, Jul. 2019. 14
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InConference on Empirical Methods in Natural Language Processing (EMNLP), 2020. 4
Masahiro Kato. Bridging the gap between empirical welfare maximization and conditional
average treatment effect estimation in policy learning, 2025a. arXiv: 2510.26723. 17, 34
Masahiro Kato. Nearest neighbor matching as least squares density ratio estimation and
riesz regression, 2025b. arXiv: 2510.24433. 3, 16, 33
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Gen-
eralization through memorization: Nearest neighbor language models. InInternational
Conference on Learning Representations (ICLR), 2020. 4
23

Juno Kim, Tai Nakamaki, and Taiji Suzuki. Transformers are minimax optimal nonparametric
in-context learners. InAnnual Conference on Neural Information Processing Systems
(NeurIPS), 2024. 5, 11, 34
Toru Kitagawa and Aleksey Tetenov. Who should be treated? empirical welfare maximization
methods for treatment choice.Econometrica, 86(2):591–616, 2018. 2, 3, 17
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K¨ uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In
International Conference on Neural Information Processing Systems (NeurIPS), 2020. 3
Zhexiao Lin, Peng Ding, and Fang Han. Estimation based on nearest neighbor matching:
from density ratio to average treatment effect.Econometrica, 91(6):2187–2217, 2023. 3, 16,
33
Marius Memmel, Jacob Berg, Bingqing Chen, Abhishek Gupta, and Jonathan Francis.
STRAP: Robot sub-trajectory retrieval for augmented policy learning. InCoRL 2024
Workshop on Mastering Robot Manipulation in a World of Abundant Data, 2024. 4
Kazusato Oko, Yujin Song, Taiji Suzuki, and Denny Wu. Pretrained transformer efficiently
learns low-dimensional target functions in-context. InAnnual Conference on Neural
Information Processing Systems (NeurIPS), 2024. 5, 11, 34
Hongming Pu and Bo Zhang. Estimating optimal treatment rules with an instrumental
variable: A partial identification learning approach.Journal of the Royal Statistical Society
Series B: Statistical Methodology, 83(2):318–345, 2021. 17
Hongxiang Qiu, Marco Carone, Ekaterina Sadikova, Maria Petukhova, Ronald C. Kessler, and
Alex Luedtke. Optimal individualized decision rules using instrumental variable methods.
Journal of the American Statistical Association, 116(533):174–191, 2021. 17
Paul R. Rosenbaum and Donald B. Rubin. The central role of the propensity score in
observational studies for causal effects.Biometrika, 70(1):41–55, 1983. 31
Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with ReLU
activation function.Annals of Statistics, 48(4):1875–1897, 2020. 4
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. REPLUG: Retrieval-augmented black-box language models.
InProceedings of the 2024 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
2024. 4
Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, and Insup Lee. REGENT: A
retrieval-augmented generalist agent that can act in-context in new environments. In
International Conference on Learning Representations (ICLR), 2025. 4
24

Taiji Suzuki. Adaptivity of deep reLU network for learning in besov and mixed smooth besov
spaces: optimal rate and curse of dimensionality. InInternational Conference on Learning
Representations (ICLR), 2019. 4
Taiji Suzuki and Atsushi Nitanda. Deep learning is adaptive to intrinsic dimensionality
of model smoothness in anisotropic besov space. InInternational Conference on Neural
Information Processing Systems (NeurIPS), 2021. 4
Adith Swaminathan and Thorsten Joachims. Counterfactual risk minimization: learning
from logged bandit feedback. InInternational Conference on Machine Learning (ICML),
2015. 2, 3
Shokichi Takakura and Taiji Suzuki. Approximation and estimation ability of transformers for
sequence-to-sequence functions with infinite dimensional input. InInternational Conference
on Machine Learning (ICML), 2023. 4, 35
Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo˜ ao Sacramento, Alexander
Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by
gradient descent. InInternational Conference on Machine Learning (ICML). JMLR.org,
2023. 5
Nengbo Wang, Xiaotian Han, Jagdip Singh, Jing Ma, and Vipin Chaudhary. CausalRAG: In-
tegrating causal graphs into retrieval-augmented generation. InFindings of the Association
for Computational Linguistics: ACL 2025, 2025. 4
Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv
Kumar. Are transformers universal approximators of sequence-to-sequence functions? In
International Conference on Learning Representations (ICLR), 2020. 4
25

A Proofs for the Regret Analysis
A.1 Regret decomposition and candidate-set bounds
For everyx∈ X, adding and subtractingf 0(a∗
g(x), x) gives
f0(a∗(x), x)−f 0(batwo(x), x)
=f 0(a∗(x), x)−f 0(a∗
g(x), x) +f 0(a∗
g(x), x)−f 0(batwo(x), x).
Taking expectations proves (1).
Proof of Proposition 4.1. For the first statement, suppose that g(x)∩ A∗
ε(x)̸=∅. Then,
there is an actiona∈g(x) such that
f0(a∗(x), x)−f 0(a, x)≤ε.
Sincea∗
g(x) maximizesf 0(a, x) overg(x), it holds thatf 0(a∗
g(x), x)≥f 0(a, x). Therefore,
f0(a∗(x), x)−f 0(a∗
g(x), x)≤ε.
Taking expectations proves the first statement.
For the second statement, define
Eε:={g(X)∩ A∗
ε(X)̸=∅}.
The first statement gives a pointwise bound of εonEε, while the bounded-difference assump-
tion gives a bound ofB fonEc
ε. Hence,
Rgen(g)≤εPr(E ε) +B fPr(Ec
ε)
≤ε+B fδ.
For the third statement, fix x. Because g(x) is finite, there is an action eag(x)∈g(x) such
that
dA(a∗(x),ea g(x)) =d A(a∗(x), g(x)).
Sincea∗
g(x) maximizesf 0overg(x),
f0(a∗(x), x)−f 0(a∗
g(x), x)≤f 0(a∗(x), x)−f 0(eag(x), x)
≤LAdA(a∗(x), g(x))βA.
Taking expectations completes the proof.
A.2 Expected-outcome estimation
Proof of Theorem 4.2.Fixx∈ X. The definition ofba two(x) gives
bf(a∗
g(x), x)− bf(batwo(x), x)≤0.
26

Therefore,
f0(a∗
g(x), x)−f 0(batwo(x), x)
=f 0(a∗
g(x), x)− bf(a∗
g(x), x)
+bf(a∗
g(x), x)− bf(batwo(x), x)
+bf(batwo(x), x)−f 0(batwo(x), x)
≤2δ f(x;g).
Taking expectations and using (1) proves the first two statements.
Suppose now thatδ f(X;g)≤r f. If the selected action is not optimal ing(X), then
0<∆ g(X)≤f 0(a∗
g(X), X)−f 0(batwo(X), X)≤2r f.
It follows that
Rchoice(batwo;g)≤2r fPr (0<∆ g(X)≤2r f)
≤C M(2rf)1+κ.
To derive (2), letEf={δf(X;g)≤r f}. On Ef, the preceding margin argument applies.
OnEc
f, the value difference is at mostB f. Hence,
Rchoice(batwo;g)≤C M(2rf)1+κ+B fPr(Ec
f)
≤C M(2rf)1+κ+B fηf.
The finite-candidate tail bound follows from
Pr
max
a∈g(X)|bf(a, X)−f 0(a, X)|> b n+uX, g(X)
≤X
a∈g(X)Pr
|bf(a, X)−f 0(a, X)|> b n+uX, g(X)
≤c1Mexp(−c 2anu2).
A.3 Relation between the one-step and two-step methods
Suppose thatba one(x)∈g(x). Then,
f0(a∗
g(x), x)≥f 0(baone(x), x),
so
f0(a∗(x), x)−f 0(a∗
g(x), x)≤f 0(a∗(x), x)−f 0(baone(x), x).
Taking expectations givesR gen(g)≤R(ba one). The remaining inequalities in Section 4 follow
from (1) and Theorem 4.2.
27

A.4 Proof of Theorem 4.3
The identity (3)follows pointwise. If ba(x) =a∗(x), the loss is zero. Otherwise, the selected
action has the lower conditional expected outcome, and the difference is|τ 0(x)|.
For the first statement, suppose that ba(x)̸=a∗(x) and τ0(x)̸= 0. Then, τ0(x)bτ(x)≤0,
which implies
0<|τ 0(x)| ≤ |bτ(x)−τ 0(x)| ≤r τ.
Using (3) and the margin condition,
R(ba)≤E[|τ 0(X)|1[0<|τ 0(X)| ≤r τ]]
≤rτPr (0<|τ 0(X)| ≤r τ)
≤C Mr1+κ
τ.
If each expected-outcome error is at mostr f, then
|bτ(x)−τ 0(x)| ≤ | bf(1, x)−f 0(1, x)|+| bf(0, x)−f 0(0, x)| ≤2r f.
For the second statement, put T=|τ0(X)|. On a sign error with T >0, it holds that
|bτ(X)−τ 0(X)| ≥T. The contribution from 0< T≤2b nis at most
2bnPr (0< T≤2b n)≤C M(2bn)1+κ.
OnT >2b n, the assumed deviation inequality gives, conditional onX,
Pr (ba(X)̸=a∗(X)|X)≤c 1exp(−c 2anT2/4) +δ n.
Lets=a−1/2
nandc=c2/4. The region 0 < T≤s contributes at most CMs1+κ. For j≥0,
define
Sj:=
2js < T≤2j+1s	
.
Whenever 2j+1s≤t 0,
E
Texp(−ca nT2)1[S j]
≤2j+1sexp(−c4j) Pr 
0< T≤2j+1s
≤C Ms1+κ2(j+1)(1+κ)exp(−c4j).
The series over jis finite. On T > t 0, boundedness gives a term of order exp(−ca nt2
0), which
is bounded by a constant multiple of s1+κunder the stated condition. The contribution of δn
is at mostB τδn. Therefore,
R(ba)≤C 1b1+κ
n+C 2a−(1+κ)/2
n +B τδn,
and the displayed result follows.
For the third statement, let E(X) =bτ(X)−τ0(X) and choose t∈(0, t0]. By the sign-error
containment,
R(ba)≤E[|τ 0(X)|1[0<|τ 0(X)| ≤t]]
+E[|τ 0(X)|1[ba(X)̸=a∗(X),|τ 0(X)|> t]].
28

The first term is at most CMt1+κ. On the event in the second term, |E(X)| ≥ |τ 0(X)|> t, so
|τ0(X)| ≤E(X)2
t.
Consequently,
R(ba)≤C Mt1+κ+qn
t.
Takingt=q1/(2+κ)
n proves the result.
B Proofs for Nearest-Neighbor Matching
B.1 Nearest-neighbor radius
Fix an actionaand a query embeddingh. Let
Ra,k(h):= max
j∈Nk,a(h)∥Ha,j−h∥.
The event Ra,k(h)> ra,koccurs only if fewer than kdatabase points fall in B(h, ra,k). The
number of points in that ball is binomial with mean at least
NaQa(B(h, r a,k))≥N acardφ
a,k= 2k.
The multiplicative Chernoff inequality therefore gives
Pr (R a,k(h)> r a,k)≤exp(−k/4).(7)
Since the support has diameterD H,
E
Ra,k(h)2β
≤r2β
a,k+D2β
Hexp(−k/4).
B.2 Proof of Theorem 5.1
On the eventR a,k(h)≤r a,k, decompose
efk(a, x)−f 0(a, x) =1
kX
j∈Nk,a(h){ma(Ha,j)−m a(h)}
+1
kX
j∈Nk,a(h)ξa,j
+m a(h)−f 0(a, x).
The absolute values of the first and third terms are bounded by Larβ
a,kandbφ,a, respectively.
Conditional on the embeddings, the middle term is mean zero and sub-Gaussian with variance
proxyσ2
a/k. Hence,
Pr
1
kX
j∈Nk,a(h)ξa,j≥u{Ha,j}Na
j=1
≤2 exp
−ku2
2σ2
a
.
29

Combining this inequality with (7) proves the pointwise deviation bound.
For the MSE bound, condition on the embeddings. The conditional variance is at most
σ2
a/k. The squared conditional bias is bounded by a constant multiple of
b2
φ,a+L2
aRa,k(h)2β.
Taking expectations and using the radius moment bound gives
Eh
(efk(a, x)−f 0(a, x))2i
≤C 
b2
φ,a+k
Na2β/d φ
+1
k+ exp(−k/4)!
.
Integrating overXproves the stated MSE result.
B.3 Derivation of the regret bounds
Let
eτk(x) =efk(1, x)− efk(0, x).
A union bound and Theorem 5.1 give, for everyxandu >0,
Pr
|eτk(x)−τ 0(x)| ≥b φ+L 0rβ
0,k+L 1rβ
1,k+u
≤4 exp
−ku2
8σ2
max
+ 2 exp(−k/4),
whereσ max= max(σ 0, σ1). Ifbfdiffers from efkby at mostη nfor each action, then
|bτ(x)−eτ k(x)| ≤2η n.
The second statement of Theorem 4.3 applies with a deterministic term of order
bφ+k
Nminβ/dφ
+ 2η n,
a concentration scalek−1/2, and a failure probability 2 exp(−k/4). This proves (5).
For the MSE bound, the inequality (u+v)2≤2u2+ 2v2gives
E
(bτ(X)−τ 0(X))2
≤C 
b2
φ+k
Nmin2β/d φ
+1
k+η2
n+ exp(−k/4)!
.
The third statement of Theorem 4.3 proves (6). Balancing ( k/N min)β/dφandk−1/2gives
k≍N2β/(2β+d φ)
min .
30

CCausal Identification with Covariates and Embed-
dings
This section records the conditions under which the observational conditional mean can be
interpreted as a potential-outcome conditional mean. We use a discrete action space because
that is the setting needed for the matching discussion.
Proposition C.1(Identification with observed covariates).Suppose that consistency holds,
soY=Y(A). Suppose also that
{Y(a) :a∈ A} ⊥A|X
and thatPr(A=a|X=x)>0on the covariate support of interest. Then,
f0(a, x) =E[Y(a)|X=x].
Proof.By consistency,Y=Y(a) on the eventA=a. Conditional exchangeability gives
f0(a, x) =E[Y|A=a, X=x]
=E[Y(a)|A=a, X=x]
=E[Y(a)|X=x].
Positivity ensures that the conditional mean given A=ais defined on the target support.
LetH=φ(X). The standard balancing-score argument gives the following result
(Rosenbaum & Rubin, 1983).
Proposition C.2(Identification with a balancing representation).Suppose that the conditions
of Proposition C.1 hold and that A⊥X|H . Then, Y(a)⊥A|H for every action a. If
Pr(A=a|H=h)>0, then
E[Y(a)|H=h] =E[Y|A=a, H=h].
Proof. For every bounded measurable function u, conditional exchangeability given Ximplies
E[u(Y(a))|A, X, H] =E[u(Y(a))|X].
Taking the conditional expectation given (A, H) and usingA⊥X|Hgives
E[u(Y(a))|A, H] =E[E[u(Y(a))|X]|A, H]
=E[E[u(Y(a))|X]|H].
The last expression does not depend on A, soY(a)⊥A|H . Consistency and positivity then
give the conditional-mean identity.
A balancing representation identifies an average conditional on H. Recovering the more
detailed target E[Y(a)|X=x] from matching on Hrequires the conditional expected
outcome to depend onXthroughH.
31

Proposition C.3(Outcome information retained by the embedding).Suppose that there is
a functionm asuch that
E[Y(a)|X] =m a(φ(X))
almost surely. Under the conditions of Proposition C.1, it holds that
f0(a, x) =m a(φ(x))
forP X-almost everyx.
Proof. Proposition C.1 gives f0(a, X) =E[Y(a)|X] . Substituting the stated condition
proves the result.
The two representation conditions have different roles. Balancing concerns adjustment
for action assignment. The last proposition concerns the heterogeneity needed to retain the
same conditional expected outcomes as the original covariates. The construction of Imai &
Nakamura (2026) combines structured covariates with features extracted from unstructured
data. The estimates studied by Dasanaike & Imai (2026) can supply a probabilistic measure-
ment of an unavailable attribute, but they do not imply either condition without further
assumptions.
The variables used for adjustment need not all appear in the final policy. Suppose
X= (Q, W ) and the policy is restricted to depend on Q. After the conditional expected
outcomes are adjusted using (Q, W), the value relevant to aQ-only policy is based on
fQ
0(a, q) :=E[f 0(a, Q, W)|Q=q].
The optimal policy within this restricted information set selects an action that maximizes
fQ
0(a, q).
D Matching Weights and Outcome Information
Contexts and historical actions alone do not determine the optimal action. To see this, fix any
joint distribution of ( X, A). One possible conditional-outcome model sets f0(1, x) = 1 and
f0(0, x) = 0 for every x, while another sets f0(1, x) = 0 and f0(0, x) = 1. Both models have
the same distribution of ( X, A), but their optimal actions are opposite. Outcome information
or an external model of expected outcomes is therefore needed.
For a neighborhoodN k(x), the local action frequency
bek(a|x) =1
kX
j∈Nk(x)1[A j=a]
targets the propensity score e0(a|x) =Pr(A=a|X =x) under standard nearest-neighbor
conditions. If this estimator is consistent and the propensity has a unique maximizer, selecting
the largest local frequency converges to the most likely historical action. Its limiting regret is
E[f 0(a∗(X), X)−f 0(ahist(X), X)], a hist(x)∈arg max
a∈Ae0(a|x),
32

which need not be zero.
The matching-weight identity in Section 5 follows directly from changing the order of
summation. If
Ka,j=mX
i=11
j∈ N k,a(φ(Xtar
i))
,
then
1
mmX
i=1efk(a, Xtar
i) =1
mkmX
i=1NaX
j=11
j∈ N k,a(Xtar
i)
Ya,j
=NaX
j=1Ka,j
mkYa,j.
Bayes’ rule gives pX(x)/pa(x) =Pr(A=a)/Pr(A=a|X =x). This is the population
relation behind the connection among match counts, density-ratio estimation, and inverse
propensity weighting (Lin et al., 2023; Kato, 2025b).
E Policy-Value Estimation
E.1 Value induced by an expected-outcome estimate
Letbs f(x)∈arg maxa∈g(x)bf(a, x) be measurable. For anys∈ S g,
bf(bsf(x), x)≥ bf(s(x), x)
for every x. Taking expectations gives bVf(bsf)≥bVf(s). Applying the same inequality at each
target context proves the empirical version. This is the equivalence used in Section 4.
E.2 Binary EWM and least-squares CATE estimation
Letµa(x) =E[Y(a)|X=x] andτ(x) =µ1(x)−µ 0(x). For a deterministic binary policy π,
defineg π(x) = 2π(x)−1. Then,
V(π) =E[µ 0(X) +π(X)τ(X)]
=E[µ 0(X)] +1
2E[τ(X)] +1
2E[g π(X)τ(X)].
Proposition E.1(EWM and least-squares CATE estimation).LetΠbe a class of determin-
istic binary policies and let GΠ={2π−1 :π∈Π} . Maximizing V(π)overΠis equivalent,
underg= 2π−1, to minimizing
E
(τ(X)−g(X))2
overGΠ. The same algebra applies to empirical criteria when τ(Xi)is replaced by a common
pseudo-outcome.
33

Proof.Everyg∈ G Πsatisfiesg(X)2= 1. Therefore,
E
(τ(X)−g(X))2
=E
τ(X)2
+ 1−2E[τ(X)g(X)].
The first two terms do not depend on g. Minimizing the squared loss is equivalent to
maximizing E[τ(X)g(X)] , which is equivalent to maximizing V(π). The empirical statement
follows in the same way. This is the equivalence studied by Kato (2025a).
E.3 IPW and AIPW policy values
Suppose that binary actions satisfy conditional exchangeability and positivity, and write
e(x) = Pr(A= 1|X=x). For a deterministic policyπ, the IPW value estimator is
bVIPW(π) =1
nnX
i=1AiYiπ(X i)
be(X i)+(1−A i)Yi(1−π(X i))
1−be(X i)
.
With the true propensity score, iterated expectations give Eh
bVIPW(π)i
=V(π). Letbµa(x)
be expected-outcome estimates. The AIPW value estimator is
bVAIPW(π) =1
nnX
i=1 
bµπ(Xi)(Xi)
+1[A i=π(X i)]
be(A i|Xi)(Yi−bµ Ai(Xi))!
,
wherebe(1|x) =be(x) andbe(0|x) = 1−be(x). The second term uses the observed outcome
to correct the expected-outcome estimate. This is the statistical difference discussed in
Section 6.
FNonparametric Rates for Transformer Expected-Outcome
Estimates
The regret theorems take a prediction-error bound as an input. This section records the
substitution for the transformer results cited in the main text and keeps the representation
dimension, in-context sample size, and number of pretraining tasks separate.
The analysis of Kim et al. (2024) considers a transformer composed of a deep neural
network with Nrep-dimensional output and one linear-attention layer. In the Besov setting,
one of its main risk bounds has the form
qICL≲N−2α/d
rep +NreplogN rep
nctx+N2
replogN rep
Tpre.
The three terms are the approximation error, the in-context generalization error, and the
pretraining generalization error. When Tpreis sufficiently large and Nrep≍nd/(2α+d)
ctx , the
first two terms give the minimax MSE raten−2α/(2α+d)
ctx up to logarithmic factors. Oko et al.
34

(2024) studies adaptation to low-dimensional target functions, while Ching et al. (2026) gives
a transformer construction attaining the H¨ older minimax MSE rate and states a separate
requirement on the number of pretraining sequences. Related approximation and estimation
results are given by Takakura & Suzuki (2023) and Havrilla & Liao (2024).
Suppose that the two binary expected-outcome estimates satisfy
Eh
(bf(a, X)−f 0(a, X))2i
≤qa,n, a∈ {0,1}.
Then,
E
(bτ(X)−τ 0(X))2
≤2q 0,n+ 2q 1,n.
The MSE statement of Theorem 4.3 therefore gives
R(ba)≤C(q 0,n+q1,n)(1+κ)/(2+κ).
Substituting the displayed in-context risk yields the corresponding regret bound. If the
prediction result is instead a pointwise exponential-deviation inequality with root scale rn,
the second statement of Theorem 4.3 gives a bound of order r1+κ
n. The MSE result alone
does not imply that sharper bound.
The sampling model must also match the RAG application. The in-context examples in
these analyses are sampled under the task model used for pretraining, while RAG retrieves
examples because they are close to the query. A direct application therefore requires the
theoretical sampling condition to cover this selected context. Section 5 instead analyzes
nearest-neighbor retrieval directly. For a general trained generator, the prediction error relative
to the required conditional expected outcome must be evaluated or assumed separately.
G Details of the Simulation Studies
G.1 Prepared Natural-Language Queries and Case Reports
All natural-language documents are completed before notebook execution. Each DGP uses
100 manually written source templates. Within every repetition, DGP 1 and DGP 2 use each
source pattern five times in the historical corpus, while DGP 3 uses each source pattern six
times. Ten distinct source patterns are used for the ten test queries in a repetition.
Each historical record contains the covariates together with the observed action and
outcome. Each test record contains covariates and a decision question. The outcome is
written as a final performance score to one decimal place. The retrieval profile is stored
separately from the prose document. It lists the attributes in a fixed order but excludes the
action and outcome as well as the decision question. The numerical state is not included in
the text supplied to the language model.
For DGP 1 and DGP 2, the levels in {−2,−1,0,1,2} are expressed as categories from
very low to very high. Experience is also expressed in years. For DGP 3, the levels in
{0.1,0.3,0.5,0.7,0.9} are expressed as categories from very low to very high. The same
completed corpus and test queries are used by every method within a repetition.
35

G.2 DGP 1: Linear Conditional Effects with Observational As-
signment
LetX= (X1, . . . , X 6), where the components are independently and uniformly distributed on
{−2,−1,0,1,2} . They represent recent performance; workload; task complexity; experience;
schedule flexibility; and team support. The two actions are
0 : temporary workload reduction 1 : weekly individual coaching.
We define
µ0(x) = 65 + 4x 1−3x 2−2x 3+ 2x 4+ 1.5x 6,
τ0(x) = 5x 1−4x 2+ 3x 5,
µ1(x) =µ 0(x) +τ 0(x).
The historical probability of action 1 is
e0(x) = clip[0.10,0.90] 
logit−1(0.5x 2−0.4x 4+ 0.3x 6−0.2x 1)
.
We drawA|X=x∼Bernoulli(e 0(x)) and generate
Y=µ A(X) +ϵ, ϵ∼ N(0,32).
All variables entering the propensity and the two conditional expected outcomes are present
in the text.
G.3 DGP 2: Smooth Nonlinear Personalization with Randomized
Assignment
DGP 2 uses the same state support and the same two actions. We define
µ0(x) = 65 + 3 sinπx1
4
−2x 2−1.2x 3+ 1.4x 4+ 1.2x 6
+ 0.5x 2x6−0.4x2
4
and
b(x) = 1.25x 1−1.50x 2+x 3−x 4+ 0.75x 5−0.75x 6
+ 0.80 sinπ(x 1−x 2)
4
+ 0.60 sinπ(x 3−x 4)
4
,
τ0(x) = 8 tanhb(x)
4
,
µ1(x) =µ 0(x) +τ 0(x).
We drawA∼Bernoulli(1/2) independently ofXand generate
Y=µ A(X) +ϵ, ϵ∼ N(0,2.52).
The distribution of Xis symmetric and b(−x) =−b(x). It follows that E[τ0(X)] = 0. Both
actions are optimal on substantial parts of the state space, while any policy that receives no
current state is restricted to a population-level choice. The conditional effect is smooth in all
six variables used to construct the retrieval profile.
36

G.4 DGP 3: Large Finite Action Space
Let
G={0.1,0.3,0.5,0.7,0.9}.
The context is X= (N, I, D, B ), where N= (N1, N2, N3, N4)∈ G4describes the need for
technical skill development; workload relief; managerial guidance; and peer support. The
variables I,D, and BinGdescribe suitable intervention intensity; suitable duration; and
baseline performance. All seven variables are drawn independently and uniformly fromG.
An action is a triple a= (t, ℓ, r ). The intervention type tis individual coaching; temporary
workload reduction; technical training; or peer support. The intensity level is ℓ∈ {1,2,3}
and the duration isr∈ {2,4}weeks. The four type profiles are
vcoaching = (0.25,0.10,1.00,0.20),
vworkload = (0.05,1.00,0.20,0.05),
vtraining = (1.00,0.05,0.35,0.15),
vpeer= (0.20,0.15,0.45,1.00).
The complete action space contains 24 actions. We set
c(a) = 0.35ℓ+ 0.15r
and define
f0(a, x) = 62 + 3B+9
2.2N⊤vt−4
I−ℓ
32
−3
D−r
42
−c(a).
Historical actions are generated from the misspecified score
s(a, x) = 2N⊤vt−0.7c(a).
The assignment probability is
Pr(A=a|X=x) = 0.20exp(s(a, x))P
a′∈Aexp(s(a′, x))+0.80
24.
The assignment rule omits suitable intensity and duration as well as baseline performance,
even though all three enterf 0. The realized outcome is
Y=f 0(A, X) +ϵ, ϵ∼ N(0,2.52).
Each action has a stable identifier from A01through A24and a natural-language description.
G.5 Embedding and Retrieval
The main experiments use the OpenAI embedding model text-embedding-3-small . Every
vector is normalized and cosine distance is used. The embedding input is the fixed-order
retrieval profile rather than the full prose report. This profile contains all attributes but
37

excludes the historical action and observed outcome as well as the decision question. The
generator receives the full prose query and the full retrieved case reports.
For DGP 1 and DGP 2, the experiment retrieves the six nearest reports observed under
action 0 and the six nearest reports observed under action 1. One-step RAG-PL and Two-step
RAG-PL receive exactly the same query and reports. Reports are displayed by retrieval rank.
The action shown first alternates across ranks, and the initial action is randomized by query.
The action descriptions and the two score summaries follow the same randomized order. The
prompt states that this order has no priority.
For DGP 3, the initial retrieval returns the twelve nearest reports without restricting
the historical action. Both procedures receive these reports and the complete action catalog.
Two-step RAG-PL then retrieves five reports within each generated candidate action.
G.6 Implementation of the RAG Procedures
The main experiments use the model identifier gpt-5.4-mini . The temperature is set to 0,
and the reasoning effort is set to none. For DGPs 1 and 2, the request configuration allows
at most 16,384 input tokens and 256 output tokens. For DGP 3, the corresponding limits are
32,768 input tokens and 1,200 output tokens. Each test query is processed independently,
and no conversation history is carried across test queries. All model outputs are validated
against predefined structured-output schemas.
In DGPs 1 and 2, every method receives six displayed reports under each action and the
exact arithmetic mean of the six displayed outcomes under each action. These summaries are
deterministic functions of the displayed reports and therefore add no observations. Providing
the summaries to every method ensures that differences among the methods are not driven
merely by the model’s ability to calculate sample averages from the displayed outcomes.
One-step RAG-PL returns only a selected action. Two-step RAG-PL returns one estimated
conditional expected outcome under each action. The experiment code selects action 1 when
its estimated conditional expected outcome is greater than or equal to that under action 0.
Thus, action 1 is selected under an exact tie.
RAG without covariates receives neither a current query profile nor historical pre-action
profiles. Its historical reports contain only the action and observed outcome. Within each
repetition, six reports are sampled uniformly without replacement under each action. The
model receives the exact arithmetic mean of the six displayed outcomes under each action
and returns two population-level expected-outcome estimates. The experiment code applies
the same tie-breaking rule as above. Because the input to this baseline does not vary across
the test queries within a repetition, the baseline is queried once for each retrieval database,
and its selected action is applied to every test query in that repetition.
In DGP 3, both RAG-PL procedures receive the same current query, the same twelve
initially retrieved reports, and the complete catalog of 24 actions. One-step RAG-PL returns
one valid action identifier from the catalog. Two-step RAG-PL first returns five distinct valid
action identifiers. The procedure then retrieves five reports observed under each candidate
action. A subsequent candidate-evaluation request supplies the five candidate actions and
their matched reports and asks for one conditional expected-outcome estimate for each
candidate. The experiment code selects the candidate with the largest estimate, using a fixed
rule to break ties.
38

During candidate evaluation, valid estimates contained in an incomplete response are
retained, and only the missing candidate estimates are requested in subsequent attempts.
Candidate generation and candidate evaluation each permit at most eight attempts in total.
All experiments reported below obtained complete valid outputs within these limits.
The DGP 3 comparison is not matched in either the number of model requests or
the amount of retrieved evidence. In particular, Two-step RAG-PL uses an additional
candidate-generation stage, candidate-specific retrieval, and a candidate-evaluation stage.
We therefore interpret the DGP 3 results as a comparison of the implemented end-to-end
pipelines, rather than as an isolated comparison of one-step and two-step action selection
under equal computational budgets.
G.7 Evaluation
Each DGP contains 20 independent repetitions and ten test queries per repetition. DGP 1
and DGP 2 contain 500 historical reports per repetition. DGP 3 contains 600. All methods
in a repetition use the same completed corpus and queries. All 20 repetitions were completed
for every method and were included in the main summaries.
For each test query, we record the selected and optimal actions. We also record their true
values and the resulting regret. An action is counted as optimal whenever its true conditional
expected outcome equals the maximum, so ties are handled by value rather than by agreement
with one tie-breaking label. Query-level quantities are averaged within a repetition. Means
and standard errors are then calculated across repetitions.
For methods that return expected outcomes, we also calculate expected-outcome mean-
squared error. In the binary DGPs, we calculate the squared error of the estimated contrast.
For DGP 3, we calculate candidate-specific expected-outcome mean-squared error and centered
mean-squared error. We also calculate pairwise ranking accuracy.
For DGP 3, let g(x) be the five generated candidates and let a∗
g(x) be the best action in
this set. We record
f0(a∗(x), x)−f 0(a∗
g(x), x)
and
f0(a∗
g(x), x)−f 0(batwo(x), x).
Their sum equals the query-level total regret. We also record exact candidate coverage.
Approximate coverage is recorded at value-gap thresholds of 0.1, 0.25, and 0.5.
G.8 Language-Model Robustness
After all three GPT-5.4 mini experiments are completed, DGP 1 is repeated with Qwen 3.5 4B
and Gemma 3 4B. The local models use the same completed documents and OpenAI retrieval
vectors as the main DGP 1 experiment. They also use the same matched reports and method
definitions. They are used only as RAG generators. Results are reported separately by model
because the purpose is to check the direction of the method comparison rather than rank
language models.
39

Table 2: Expected-outcome estimation in the binary DGPs
Method Expected-outcome MSE Contrast MSE
DGP 1
Two-step RAG-PL 111.4406 161.4745
RAG without covariates 229.4167 251.6467
DGP 2
Two-step RAG-PL 28.5777 31.1131
RAG without covariates 51.9164 41.3173
Table 3: Regret decomposition and candidate evaluation in DGP 3
Quantity Mean Standard error
Total regret 1.2154 0.0837
Candidate-set regret 0.4822 0.0387
Within-candidate regret 0.7332 0.0616
Exact candidate coverage 0.4450 0.0294
Candidate coverage within 0.1 0.4500 0.0286
Candidate coverage within 0.25 0.5200 0.0258
Candidate coverage within 0.5 0.6700 0.0309
Candidate expected-outcome MSE 3.0358 0.1412
Centered candidate outcome MSE 1.9420 0.1096
Pairwise ranking accuracy 0.6200 0.0153
Best-candidate selection probability 0.3450 0.0359
G.9 Additional Results
Table 2 reports the expected-outcome errors for procedures that return numerical estimates in
the binary DGPs. Two-step RAG-PL has lower expected-outcome and contrast mean-squared
error than RAG without covariates in both DGPs.
Table 3 separates the regret of Two-step RAG-PL in DGP 3. Candidate generation
accounts for mean regret 0 .4822, while selection within the generated set accounts for 0 .7332.
Pairwise ranking accuracy within the candidate set is 0 .6200, and the best candidate is
selected with probability 0.3450.
Table 4 gives the DGP 1 results for the two local generators. For both models, One-step
RAG-PL and Two-step RAG-PL have lower mean regret than RAG without covariates. The
relative ordering of the one-step and two-step procedures varies with the generator, so these
results are used as a robustness check rather than a model ranking.
40

Table 4: DGP 1 robustness results with local language models
Method Regret Regret SE Policy value Optimal-action probability
Qwen 3.5 4B
One-step RAG-PL 3.4600 0.4353 66.6025 0.5450
Two-step RAG-PL 3.6050 0.4418 66.4575 0.5250
RAG without covariates 4.5450 0.3943 65.5175 0.4100
Gemma 3 4B
One-step RAG-PL 3.1450 0.4055 66.9175 0.5700
Two-step RAG-PL 3.5200 0.4070 66.5425 0.5450
RAG without covariates 4.5450 0.3943 65.5175 0.4100
41