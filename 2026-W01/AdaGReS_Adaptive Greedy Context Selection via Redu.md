# AdaGReS:Adaptive Greedy Context Selection via Redundancy-Aware Scoring for Token-Budgeted RAG

**Authors**: Chao Peng, Bin Wang, Zhilei Long, Jinfang Sheng

**Published**: 2025-12-31 18:48:07

**PDF URL**: [https://arxiv.org/pdf/2512.25052v1](https://arxiv.org/pdf/2512.25052v1)

## Abstract
Retrieval-augmented generation (RAG) is highly sensitive to the quality of selected context, yet standard top-k retrieval often returns redundant or near-duplicate chunks that waste token budget and degrade downstream generation. We present AdaGReS, a redundancy-aware context selection framework for token-budgeted RAG that optimizes a set-level objective combining query-chunk relevance and intra-set redundancy penalties. AdaGReS performs greedy selection under a token-budget constraint using marginal gains derived from the objective, and introduces a closed-form, instance-adaptive calibration of the relevance-redundancy trade-off parameter to eliminate manual tuning and adapt to candidate-pool statistics and budget limits. We further provide a theoretical analysis showing that the proposed objective exhibits epsilon-approximate submodularity under practical embedding similarity conditions, yielding near-optimality guarantees for greedy selection. Experiments on open-domain question answering (Natural Questions) and a high-redundancy biomedical (drug) corpus demonstrate consistent improvements in redundancy control and context quality, translating to better end-to-end answer quality and robustness across settings.

## Full Text


<!-- PDF content starts -->

AdaGReS:Adaptive Greedy Context Selection via Redundancy-Aware
Scoring for Token-Budgeted RAG
Chao Peng Bin Wang Zhilei Long Jinfang Sheng
Central South University
Yizhi Intelligent (YZInt)
chao.peng@yzint.cn
Abstract
Retrieval-augmented generation (RAG) is
highly sensitive to the quality of selected
context, yet standard top-k retrieval often
returns redundant or near-duplicate chunks
that waste token budget and degrade down-
stream generation. We present AdaGReS, a
redundancy-aware context selection framework
for token-budgeted RAG that optimizes a set-
level objective combining query–chunk rele-
vance and intra-set redundancy penalties. Ada-
GReS performs greedy selection under a token-
budget constraint using marginal gains derived
from the objective, and introduces a closed-
form, instance-adaptive calibration of the rele-
vance–redundancy trade-off parameter to elim-
inate manual tuning and to adapt to candidate-
pool statistics and budget limits. We further
provide a theoretical analysis showing that the
proposed objective exhibits ε-approximate sub-
modularity under practical embedding similar-
ity conditions, yielding near-optimality guaran-
tees for greedy selection. Experiments on open-
domain question answering (Natural Questions)
and a high-redundancy biomedical (drug) cor-
pus demonstrate consistent improvements in
redundancy control and context quality, trans-
lating to better end-to-end answer quality and
robustness across settings.
1 Introduction
Retrieval-augmented generation (RAG), first intro-
duced by Lewis et al. (2020)(Lewis et al., 2020),
has rapidly developed into a mainstream technique
for enabling large language models (LLMs) to
incorporate external knowledge and enhance per-
formance on knowledge-intensive tasks. By inte-
grating external documents or knowledge chunks
with large models, RAG allows systems to dynami-
cally access up-to-date, domain-specific informa-
tion without frequent retraining, thereby improving
access to up-to-date and domain-specific informa-
tion without frequent retraining. Dense passageretrievers such as DPR proposed by Karpukhin et
al. (2020)(Karpukhin et al., 2020), the ColBERT
model by Khattab and Zaharia (2020)(Khattab and
Zaharia, 2020), as well as later architectures like
REALM(Guu et al., 2020) and FiD(Lewis et al.,
2020), have further improved the retrieval, encod-
ing, and fusion mechanisms of RAG in practi-
cal applications. Today, RAG is widely applied
in open-domain question answering(Kwiatkowski
et al., 2019), scientific literature retrieval(Lála
et al., 2023), healthcare, enterprise knowledge man-
agement, and other scenarios, becoming a key
paradigm for enabling LLMs to efficiently leverage
external knowledge.
Despite the significant advancements brought by
RAG, particularly in enhancing knowledge time-
liness, factual consistency, and task adaptability
for LLMs, overall performance is still highly de-
pendent on the quality of context chunks returned
by the retrieval module. A persistent challenge
is how to ensure that the retrieved results are not
only highly relevant to the user’s query but also
exhibit sufficient diversity in content. Numerous
empirical studies have found that systems tend to
return overlapping or near-duplicate chunks un-
der top-k retrieval, especially when documents are
chunked densely or the corpus is highly redun-
dant(Pradeep, Thakur, Sharifymoghaddam, Zhang,
Nguyen, Campos, Craswell, and Lin, 2025)(Tang
et al., 2025). Such redundancy not only wastes
valuable context window (token budget) but can
also obscure key information, limiting the model’s
capacity for deep reasoning, comparative analysis,
or multi-perspective synthesis, ultimately under-
mining factual accuracy and logical coherence.
For instance, in multi-hop question answering
and multi-evidence reasoning tasks, if the retriever
mainly returns paraphrased but essentially identical
chunks, the model will struggle to acquire complete
causal chains or diverse perspectives. This type of
pseudo-relevance phenomenon has been shown to
1arXiv:2512.25052v1  [cs.CL]  31 Dec 2025

be an important contributor to hallucinations in
RAG systems: when lacking sufficient heteroge-
neous evidence, the model may rely on internal
priors and produce superficially coherent but ex-
ternally unsupported and erroneous content(Zhang
and Zhang, 2025).
To address fragment redundancy and hallucina-
tion, Maximal Marginal Relevance (MMR) and its
variants have been widely adopted in existing RAG
systems as well as in emerging frameworks such as
GraphRAG(Peng et al., 2024) and FreshLLM(Vu
et al., 2024). By balancing relevance and diver-
sity in the set of retrieved candidates, MMR can
reduce redundancy and improve coverage. While
effective in practice, these approaches still suffer
from notable limitations: (1) their weight param-
eters are highly dependent on manual tuning and
cannot dynamically adapt to the structure of differ-
ent candidate pools or token budgets; (2) they only
support local greedy selection, making it difficult to
achieve set-level global optimality and potentially
missing the best combination of chunks.
To systematically solve the issues of context re-
dundancy, limited diversity, and cumbersome pa-
rameter tuning in RAG, this paper proposes and
implements a novel context scoring and selection
mechanism based on redundancy-awareness and
fully adaptive weighting. Specifically, we design
a set-level scoring function that not only measures
the relevance between each candidate chunk and
the query, but also explicitly penalizes redundancy
among the selected fragments. The entire scoring
process is mathematically modeled as the weighted
difference between a relevance term and a redun-
dancy term, with a tunable parameter βcontrolling
the trade-off between them. Building on this, we
further propose a dynamic and adaptive βadjust-
ment strategy: by analyzing the average length,
mean relevance, and redundancy distribution of the
candidate pool, we derive a closed-form solution
for the redundancy weight that adapts to differ-
ent queries and budget constraints. This strategy
provides a principled closed-form estimate of β,
eliminating the need for manual parameter tuning
or external heuristics. We also provide engineering
implementations for instance-level β, as well as
interfaces for fine-tuning on small validation sets
and domain-specific customization, improving the
method’s robustness and usability in real-world,
variable scenarios.
To validate the theoretical foundation and prac-
tical effectiveness of the proposed approach, weconduct a rigorous theoretical analysis of the
redundancy-aware adaptive selection framework.
By proving the ε-approximate submodularity of
the objective function, we establish approxima-
tion guarantees for greedy selection under ε-
approximate submodularity. Our analysis further
reveals how the adaptive βmechanism dynami-
cally suppresses excessive redundancy, enhances
coverage, and prevents performance degradation,
especially in complex data distributions or under
tight budget constraints. Experiments demonstrate
that the proposed method significantly outperforms
traditional baselines on key metrics such as answer
quality, coverage, and redundancy control, both in
open-domain and biomedical knowledge retrieval
tasks.
The main contributions of this work are as fol-
lows:
(1)We propose a redundancy-aware, fully adap-
tive context scoring and selection framework that
systematically addresses key challenges such as
context redundancy and limited diversity in RAG
scenarios;
(2)We design a set-level relevance-redundancy
joint scoring function and derive a closed-form
adaptive solution to the βparameter, enabling dy-
namic, instance-specific and budget-specific trade-
off without manual tuning;
(3)We provide a theoretical analysis and proof
ofε-approximate submodularity for our objective,
offering theoretical guarantees for the near-global
optimality of the greedy selection algorithm.
2 Related Work
2.1 Retrieval-Augmented Generation and
Context Selection
Modern RAG systems typically employ dense or
hybrid retrievers, such as DPR(Karpukhin et al.,
2020), ColBERT(Khattab and Zaharia, 2020), or
bi-encoder models, for initial passage retrieval, fol-
lowed by subsequent ranking or selection modules
to assemble the final context. This architecture
achieves state-of-the-art results on benchmarks like
Natural Questions and MS MARCO (Kwiatkowski
et al., 2019; Nguyen et al., 2016), but practical de-
ployment in real-world domains reveals challenges
in retrieval accuracy, context selection quality, and
robustness to distributional shifts (Pradeep et al.,
2025; Reuter et al., 2025; Amugongo et al., 2025).
Recent studies have focused on deeper aspects
of context selection and retrieval effectiveness. For
2

example, Xu et al. (2025) developed a token-
level framework showing that simply expanding
retrieved context can mislead LLMs and reduce
answer quality. Other works have explored in-
tegrating structured knowledge: KG²RAG(Tang
et al., 2025)and similar knowledge-graph-based re-
trieval systems(Zhang and Zhang, 2025)improve
factual grounding, but also raise new questions
about chunk granularity and overlap. Advances
such as HeteRAG (Yang et al., 2025) and modular
retriever-generator architectures(Peng et al., 2024;
Vu et al., 2024)reflect a move toward decoupling
retrieval and generation representations.
Nevertheless, a persistent challenge in RAG re-
mains the redundancy and overlap among selected
context chunks. Most traditional retrievers prior-
itize chunk–query relevance, and even diversity-
aware rerankers such as Maximal Marginal Rel-
evance (MMR) are typically applied with fixed
tradeoff weights and myopic greedy decisions; as
a result, the selected context can still contain re-
peated or overlapping information at the set level.
This wastes token budget and can degrade language
model performance, e.g., by amplifying halluci-
nations or reducing factual precision. Although
redundancy-aware and diversity-promoting meth-
ods have recently appeared—such as attention-
guided pruning (AttentionRAG, Fang et al., 2025)
and dynamic chunk selection—many of these so-
lutions still rely on heuristic or static parameters
and require manual tuning; moreover, only a sub-
set explicitly optimizes a set-level objective under
strict token constraints, which can limit robustness
and scalability in practice(Nguyen et al., 2016).
Additionally, many methods fail to address token
constraints and semantic variability in industrial-
scale deployments(Reuter et al., 2025; Amugongo
et al., 2025; Gao et al., 2023).
To address these deficiencies, we propose a
redundancy-aware scoring framework that unifies
query relevance and intra-set redundancy into a
principled, set-level objective. Our approach em-
ploys a greedy selection algorithm, with theoretical
guarantees based on approximate submodularity,
to construct high-coverage, low-redundancy con-
text sets. Critically, we introduce a closed-form,
fully adaptive calibration of the redundancy trade-
off parameter, allowing the system to automatically
adjust according to candidate pool properties and
token budget—removing manual tuning and en-
suring robustness across domains and scales. Ex-
periments demonstrate that our approach achievesbetter coverage, less redundancy, and improved
answer quality versus prior baselines. This frame-
work bridges the gap between real-world RAG con-
straints and optimal context selection, providing an
effective and theoretically grounded solution for
modern RAG systems.
2.2 Retrieval-Augmented Generation and
Context Selection
Balancing relevance and diversity has long been a
central objective in retrieval-based systems. The
classical Maximal Marginal Relevance (MMR)
framework addresses this by selecting items that
maximize similarity to the query while minimiz-
ing similarity to previously selected elements,
thereby reducing redundancy(Carbonell and Gold-
stein, 1998). This foundational idea inspired sev-
eral extensions across information retrieval and text
summarization. For instance, diversity-promoting
retrieval methods strengthen the anti-redundancy
component to cover multiple semantic clusters,
but their performance heavily depends on man-
ually tuned relevance–diversity coefficients(Yao
et al., 2017). Determinantal Point Processes (DPPs)
model subset diversity through determinant-based
selection, providing strong theoretical properties
but suffering from high computational cost as can-
didate pools scale(Cho et al., 2019). Submodular
optimization approaches generalize MMR to set-
level selection using predefined utility functions,
yet often rely on fixed or validation-tuned parame-
ters, making them less responsive to the redundancy
structure of new candidate pools(Lin and Bilmes,
2010). Embedding clustering or centroid-based
selection enhances semantic coverage but may sac-
rifice fine-grained relevance and overlook subtle
but crucial information(Mohd et al., 2020).
Across these MMR -related approaches, several
limitations consistently appear: (1) they rely on
fixed or manually tuned tradeoff parameters, (2)
they optimize selection locally rather than glob-
ally, and (3) they do not adapt to the characteris-
tics of the candidate pool, such as varying redun-
dancy levels or semantic density. Most importantly
for retrieval -augmented generation, these methods
are not designed to account for strict token con-
straints in RAG database retrieval, where select-
ing too many redundant chunks directly wastes the
available token budget and degrades downstream
generation quality.
Our method Adaptive Greedy Context Selec-
tion via Redundancy-Aware Scoring addresses
3

these limitations by introducing a fully adaptive
redundancy-aware scoring function that calibrates
the relevance–redundancy tradeoff according to
candidate -pool statistics and token budget, and in-
tegrates this improved scoring mechanism directly
into the RAG context selection process, enabling
efficient, redundancy -controlled retrieval under re-
alistic token constraints.
2.3 Selection Algorithms and Theoretical
Guarantees
Selection algorithms grounded in submodular opti-
mization have become foundational for data subset
selection, document summarization, and retrieval-
augmented generation. The concept of submodu-
larity—first formalized in combinatorial optimiza-
tion literature (Nemhauser et al., 1978)(Nemhauser
et al., 1978)—describes set functions exhibiting
the diminishing returns property: the marginal gain
of adding an item to a smaller set is greater than
adding it to a larger set. This property is critical
because it enables efficient greedy algorithms to
obtain strong theoretical guarantees. In particular,
the seminal result proved that a simple greedy al-
gorithm achieves at least a ( 1−1
e)-approximation
for maximizing any monotone submodular func-
tion subject to a cardinality constraint. In token-
budgeted RAG, the constraint is cost/budget-based
rather than purely cardinality-based, but the sub-
modularity framework remains valuable for moti-
vating efficient greedy-style approximations under
such constraints.
In context selection and related problems, sub-
modular set functions have been widely adopted to
model both relevance and diversity. For example,
Lin and Bilmes (2011)(Lin and Bilmes, 2011)lever-
aged submodular functions for extractive document
summarization, demonstrating empirical and theo-
retical benefits in content coverage and redundancy
reduction. More recently, Wei et al. (2015)(Wei
et al., 2015)and Mirzasoleiman et al. (2016)(Mirza-
soleiman et al., 2015)extended these ideas to large-
scale data subset selection in machine learning
pipelines, relying on submodularity to enable scal-
able and theoretically sound selection algorithms
even as candidate pool size grows.
Our method, Adaptive Greedy Context Selec-
tion via Redundancy-Aware Scoring, inherits these
theoretical properties: the redundancy-aware scor-
ing function we propose exhibits approximate sub-
modularity under realistic embedding distributions,
allowing a greedy selection procedure to provideprovable near-optimality for token-budgeted selec-
tion tasks in RAG. Detailed theoretical analysis and
formal guarantees for our approach are presented
in Section 4.
3 Method
3.1 Redundancy-Aware Scoring Function
Retrieval-augmented generation (RAG) pipelines
often suffer from redundant or highly overlapping
context selections, especially under strict token
budgets. Conventional similarity-based retrievers
maximize the relevance between the query and se-
lected chunks but largely ignore information redun-
dancy within the selected set, resulting in inefficient
budget usage and repetitive evidence.
To address this, we introduce a redundancy-
aware scoring function that jointly considers the
relevance of each candidate chunk to the query
and penalizes intra-set redundancy. Given a query
embedding q∈Rdand a set of candidate chunk
embeddings V={c 1, . . . , c N},where q and all ci
are L2-normalized to unit norm, we aim to select
a subset C⊂ V that maximizes the total query-
aligned evidence mass (i.e.,P
c∈Csim(q, c) ) while
minimizing duplication.
Formally, our scoring function for any candidate
subsetCis defined as:
F(q, C) =αS qC(q, C)−βS CC(C)(1)
where SqC(q, C) =P
c∈Csim(q, c) measures
the total relevance between the query and the se-
lected chunks. SCC(C) =P
i<j,c i,cj∈Csim(c i, cj)
measures the total redundancy (pairwise similar-
ity) among the selected chunks.Note that SCC(C)
scales with the set size, so the appropriate mag-
nitude of βdepends on the expected number of
selected chunks under the token budget; we ad-
dress this via an instance-adaptive closed-form
βin Section 3.3.Here, sim(a, b) denotes a non-
negative cosine similarity, sim(a, b) =a⊤bun-
der unit norm. The hyperparameters α >0 and
β≥0 control the tradeoff between relevance
and redundancy.When β= 0 , the function re-
duces to standard relevance maximization, equiv-
alent to vanilla similarity-based selection.As βin-
creases, the method penalizes redundant chunks
more strongly, promoting diversity within the se-
lected context.
Our approach generalizes the widely used max-
imal marginal relevance paradigm by making the
4

relevance–redundancy tradeoff explicit in a set-
level objective. While MMR is commonly imple-
mented as a sequential greedy reranking rule with a
fixed tradeoff weight, our formulation defines sub-
set quality directly via F(q, C) , which provides a
principled target for token-budgeted selection and
motivates instance-adaptive calibration ofβ.
In summary, the proposed scoring function estab-
lishes a unified objective that maximizes coverage
and information diversity, forming the basis for
subsequent selection algorithms.
3.2 Greedy Context Selection with Token
Budget
Maximizing the redundancy-aware objective
F(q, C) over all possible chunk subsets is a classi-
cal NP-hard combinatorial problem; exact search
is infeasible for realistic candidate pool sizes. To
efficiently approximate the optimal solution, we
adopt a greedy selection algorithm that incremen-
tally builds the context set by always choosing the
candidate chunk with the highest marginal gain at
each step, subject to the token budget constraint.
Formally, let Ccurdenote the current set of se-
lected chunks. For any candidate x /∈C cur, we
define the marginal gain as:
∆F(x|C cur) =F(q, C cur∪ {x})−F(q, C cur)
=αsim(q, x)−βP
c∈C cursim(x, c)
(2)
At each iteration, we select
x∗= arg max
x∈V\C cur∆F(x|C cur)(3)
and add x∗toCcur, provided that adding x∗does
not exceed the overall token budget Tmax. The
process stops when (1) no remaining chunk yields a
positive marginal gain, or (2) the token limit would
be exceeded by any further addition.
The complete greedy selection procedure is sum-
marized below:
3.2.1 Limitation of Greedy Selection: Local
Optima
It is important to recognize that greedy selection,
by its very nature, only optimizes the marginal
gain at each step. This locally optimal decision
process does not guarantee a globally optimal solu-
tion—especially in cases where initial non-optimal
choices could lead to better overall combinations.
In other words, the algorithm may become trapped
in local optima and miss out on chunk sets that
would yield higher objective scores if considered
jointly.Algorithm 1: Greedy Context Selection with
Token Budget
Input:query embedding q; candidate set
V={c 1, . . . , c N}; token budget Tmax; scor-
ing weightsα, β; token length functionℓ(·)
Output:selected subsetC∗
1. InitializeC← ∅, total_tokens←0
2.while total_tokens< T maxand∃x∈V\C
with∆F(x|C)>0do
Selectx∗←arg max x∈V\C ∆F(x|C)
iftotal_tokens+ℓ(x∗)≤T maxthen
C←C∪ {x∗}
total_tokens←total_tokens+ℓ(x∗)
elsecontinue
3.returnC
3.2.2 Theoretical Justification: Approximate
Submodularity
Despite this limitation, our design leverages impor-
tant theoretical properties of the scoring function.
As we will detail in Section 4, the redundancy-
aware objective F(q, C) exhibits ϵ-approximate
submodularity under typical embedding distribu-
tions, where chunk-to-chunk similarities are gener-
ally small. This property enables the greedy algo-
rithm to achieve solutions that are provably close to
the global optimum, with a bounded approximation
gap. Our later analysis provides formal guarantees
for the quality of greedy selection in this context.
In summary, greedy selection offers a practical
and highly efficient approach to redundancy-aware
context selection under token constraints. Its effec-
tiveness is theoretically grounded by the approxi-
mate submodular nature of our scoring function, as
will be established in the following sections.
3.3 Adaptiveβfor Dynamic Token
Constraints
The tradeoff parameter βin the redundancy-aware
scoring function plays a critical role in balancing
query relevance and intra-set redundancy. How-
ever, its value is sensitive to both the candidate
pool properties and the available token budget. A
fixedβmay lead to excessive redundancy or overly
aggressive pruning, resulting in suboptimal context
selection. Therefore, we propose a principled ap-
proach to adapt βper instance, using simple statis-
tics of the candidate pool and the current token
constraint.
5

3.3.1 Motivation and Problem Statement
When the token budget Tmax is tight, the num-
ber of selectable chunks is limited, and a larger
βis typically needed to prevent redundancy from
consuming the scarce budget.When the budget is
ample, or when the candidate pool is inherently
diverse, a smaller βoften suffices to prioritize rele-
vance.Our goal is to automatically calibrate βsuch
that, at the expected context set size dictated by the
token budget, the marginal gain of adding a new
chunk just reaches zero—balancing the contribu-
tions of relevance and redundancy at the decision
boundary.
3.3.2 Theoretical Derivation
Let the average token length of the top candidate
chunks be ¯L.Under token budget Tmax, the ex-
pected number of selectable chunks is approxi-
mated as
¯k≈Tmax
¯L(4)
To encourage the greedy process to stop around ¯k,
we set the expected marginal gain at the boundary
to be approximately zero:
∆F(x| ¯C) =αE x[sim(q, x)]−βX
c∈¯CE[sim(x, c)]≈0
(5)
where expectations are taken over the top-N can-
didate pool, and ¯Cis a typical set of ¯k−1 selected
chunks. We approximate the redundancy increment
by assuming that, for a typical boundary candidate
xxx, the average similarity to each previously se-
lected chunk is close to the average chunk–chunk
similarity within the top-N pool. This yields:
X
c∈¯CE[sim(x, c)]≈( ¯k−1)·E x̸=y∼νtop[sim(x, y)].
(6)
Solving forβgives the adaptive calibration:
β∗=αEx∼Vtop[sim(q, x)]
¯k−1
2·Ex̸=y∼Vtop[sim(x, y)](7)
where ε >0 is a small constant used
for numerical stability (e.g., when the pool
redundancy estimate is near zero). Vtop
is the top-N candidate chunks most similar
to the query, Ex∼Vtop[sim(q, x)] is the average
query–chunk similarity, Ex̸=y∼Vtop[sim(x, y)] is
the average pairwise similarity within the top-N
candidate pool, which serves as a proxy for typical
redundancy.This closed-form solution provides a fully adap-
tive way to tune βfor any given query and candi-
date set.
3.3.3 Why Use Expectation?
The identity of the final selected chunk in the
greedy process is not known a priori, and its spe-
cific relevance/redundancy cannot be determined
before selection completes. Using empirical av-
erages over the candidate pool provides a stable,
low-variance estimate of “typical” relevance and
redundancy near the selection boundary, enabling
robust automatic calibration across a wide range of
queries and corpus redundancy profiles.
3.3.4 Practical Implementation
To implement adaptiveβin practice:
1. Compute the average token length ¯Lof the
top-N candidate chunks.
2. Estimate the expected set size ¯k≈T max/¯L
under the token budget.
3. Compute the average query–chunk similarity
Ex∼Vtop[sim(q, x)].
4. Estimate the average pairwise redundancy
withinVtop
Ex̸=y∼Vtop[sim(x, y)]
Since an exact computation is O(N2), we either
use a modest Nor estimate it by sampling random
pairs fromVtop.
5. Plug the statistics into the formula to compute
β∗, and applyβ∗in greedy selection.
In practice, these computations add only
lightweight overhead relative to embedding re-
trieval and scoring, and can be performed on a
per-query basis.
Empirical Bias and Future Extensions
While β∗is robust in most cases, highly skewed
or noisy candidate pools may benefit from addi-
tional calibration. We therefore optionally intro-
duce an empirical scaling/bias:
β=λ·β∗+β0
where λ(scaling) and β0(bias) can be tuned on a
small validation set or set to default values ( λ= 1 ,
β0= 0) for fully automatic use. In engineering
practice, βcan also be clipped to a reasonable range
to avoid extreme pruning or extreme redundancy
in outlier cases.
6

Moreover, these parameters could be learned via
reinforcement learning or meta-learning to opti-
mize downstream task performance or user prefer-
ences. We leave such learning-based calibration as
future work.
In summary, adaptive βis not merely an empiri-
cal trick. By calibrating βusing observed average
redundancy and the budget-implied set size, we
align the selection behavior with the redundancy
profile of the candidate pool. This calibration helps
keep the objective well-behaved for greedy opti-
mization across datasets with different redundancy
characteristics, thereby bridging practical robust-
ness and theoretical soundness.
4 Theoretical Analysis
4.1 Modularity and Supermodularity of
Objective Components
Our redundancy-aware objective function is com-
posed of two main components—a relevance term
that encourages alignment between the selected
context and the query, and a redundancy term that
penalizes overlap within the selected set. To ana-
lyze the theoretical properties of our objective, we
first separately examine the modularity and (super-
)modularity of each term.
4.1.1 Relevance Term: Modularity
The relevance component is given by:
SqC(q, C) =X
c∈Csim(q, c)(8)
where sim(q, c) denotes cosine similarity. This
function is modular, meaning that the contribution
of each chunk cto the overall score is independent
of the other selected chunks. Adding a chunk xto
any subsetAyields a constant marginal gain:
SqC(q, A∪ {x})−S qC(q, A) = sim(q, x)(9)
This is independent of A, confirming strict mod-
ularity. Modular functions are a special case of
submodular functions. Moreover, SqCis monotone
non-decreasing when s(q, c)≥0 for all candidates,
which holds under our non-negative similarity defi-
nition.
4.1.2 Redundancy Term: Supermodularity
The redundancy penalty is defined as:
SCC(C) =X
i<j, c i,cj∈Csim(c i, cj)(10)This measures the total pairwise similarity
among selected chunks. The marginal increase
when adding a chunkxto a setAis:
SCC(A∪ {x})−S CC(A) =X
c∈Asim(x, c)(11)
Now consider two setsA⊆B, withx /∈B:
SCC(B∪ {x})−S CC(B) =X
c∈Bsim(x, c)(12)
SinceBcontains all elements ofA, we have:
SCC(B∪{x})−S CC(B)≥S CC(A∪{x})−S CC(A)
(13)
because the sum in Bincludes all terms from A,
plus additional terms from B\A ,Since s(˙,˙)≥by
construction, the marginal penalty increases with
set size, establishing supermodularity This demon-
strates that the redundancy term is supermodular,
the marginal penalty for adding a chunk increases
with the size of the set.
4.1.3 Overall Structure: Modular Minus
Supermodular
Our full objective is
F(q, C) =αS qC(q, C)−βS CC(C)(14)
which is a modular term minus a supermodular
term. This decomposition motivates a closer analy-
sis of submodularity and the resulting greedy guar-
antees under token-budget constraints (see §4.2).
While the relevance term always yields constant,
context-independent gains, the redundancy term
introduces a context-dependent penalty that grows
superlinearly with the size and density of the set.
4.2 Failure of Strict Submodularity
While our redundancy-aware objective F(q, C) el-
egantly balances query relevance and intra-set re-
dundancy, its mathematical structure as a modular
minus supermodular function means that, in gen-
eral, it does not satisfy strict submodularity. This
section provides a detailed analysis of why this is
the case, and lays the groundwork for our subse-
quent notion of approximate submodularity.
7

4.2.1 Definition and Intuition of
Submodularity
Recall that a set function f: 2V→R is submodu-
lar if, for anyA⊆B⊆Vandx /∈B,
f(A∪ {x})−f(A)≥f(B∪ {x})−f(B)(15)
This property, known as diminishing marginal
returns, is crucial for guaranteeing the near-
optimality of greedy algorithms.
4.2.2 Analysis of Our Objective
Given
F(q, C) =αS qC(q, C)−βS CC(C)(16)
the marginal gain of adding chunkxto setCis
∆F(x|C) =αsim(q, x)−βX
c∈Csim(x, c)(17)
Consider two sets A⊆B⊆V andx /∈B . The
difference in marginal gains is
∆F(x|A)−∆F(x|B) =βX
c∈B\Asim(x, c)
(18)
Ifβ >0 and similarities are non-negative, then
∆F(x|A)−∆F(x|B)≥0 , i.e., marginal gains
decrease as the set grows, which is consistent with
diminishing returns. However, with cosine simi-
larity, some pairwise similarities may be negative,
and the sumP
c∈B\A sim(x, c) can become nega-
tive, yielding ∆F(x|A)<∆F(x|B) and violating
strict submodularity.
4.2.3 Counterexample
For strict submodularity, we would require
∆F(x|A)≥∆F(x|B)(19)
But as shown above,
∆F(x|A)−∆F(x|B) =βX
c∈B\Asim(x, c)
(20)
If there exists any c∈B\A such that
sim(x, c)>0 andβ >0 , the right-hand side
is negative, so the submodularity condition is vio-
lated.
Figure 1: It illustrates the difference in marginal gain be-
haviors between strictly submodular objectives and our
modular-minus-supermodular objective: strictly sub-
modular objectives see marginal gains that consistently
decrease as the set grows, while our modular-minus-
supermodular objective may exhibit flat or even in-
creasing marginal gains when high redundancy or non-
overlapping candidates are present.
4.2.4 Visualization of Marginal Gain
Interpretation
- The relevance term alone is modular and thus
trivially submodular.
- The redundancy penalty grows with the set,
causing marginal gain to decrease less than ex-
pected—or even to increase—when adding ele-
ments to larger sets.
- As a result, the overall function may exhibit
supermodular effects, especially when βor chunk
similarities are large.
This limitation suggests that classical greedy
guarantees for monotone submodular maximiza-
tion under a cardinality constraint do not directly
transfer to our setting. In addition, our selection
is constrained by a token budget (knapsack-style)
constraint, which further changes the conditions re-
quired for standard 1−1
e-type results. In particular,
greedy selection can fail to find the true global opti-
mum, especially in worst-case settings with highly
redundant candidate pools or poorly chosenβ.
4.2.5 Practical Implications
Despite the lack of strict submodularity, in practical
scenarios—such as typical embedding distributions
where intra-chunk similarities are small—the objec-
tive may approximately satisfy diminishing returns,
as we analyze in the next section. This justifies the
strong empirical performance of greedy selection
observed in our experiments.
8

4.3ε-Approximate Submodularity and
Greedy Guarantee
Given that our objective couples relevance with
pairwise redundancy, its marginal gain depends
on the already-selected set. In particular, when
the similarity measure is allowed to take signed
values, strict diminishing returns may not hold
in the worst case. We therefore quantify how far
our objective can deviate from submodularity via
ε-approximate submodularity, and show that this
deviation is bounded under mild similarity con-
straints—providing a principled justification for
greedy selection in typical low-redundancy RAG
candidate pools. we introduce and formalize the
notion of ϵ-approximate submodularity for our ob-
jective, show that—under typical conditions—this
violation is tightly bounded, and explain how our
adaptive βstrategy works hand-in-hand with theory
to ensure practical near-optimality.
4.3.1 Definition:ε-Approximate
Submodularity
A set function fis said to be ϵ-approximately sub-
modular if for anyA⊆B⊆Vandx /∈B,
f(A∪{x})−f(A)≥f(B∪{x})−f(B)−ϵ(21)
where ϵ≥0 quantifies the maximal deviation
from strict submodularity. When ϵis small, the
function behaves nearly submodular, and greedy
algorithms retain strong approximation guarantees.
4.3.2 Boundingεfor Our Objective
Recall from the previous section that
∆F(x|A)−∆F(x|B) =βX
c∈B\Asim(x, c)
(22)
To bound this, suppose the pairwise cosine simi-
larity between any two distinct candidate chunks is
upper bounded byδ >0:
sim(x, c)≤δ∀x̸=c(23)
Suppose |B\A| ≤k (where kis the maximal
set size imposed by the token budget). Then,
∆F(x|A)−∆F(x|B)≤βkδ(24)
Therefore, our objective is ϵ-approximately sub-
modular with
ϵ=βkδ(25)4.3.3 Theoretical Role of Adaptiveβ
It is important to note that the value of ϵ—and thus
the strength of our greedy approximation—depends
directly on the choice of β. In scenarios where the
candidate set exhibits high redundancy (i.e., large δ
or large k), a fixed and overly large βcould result
in a loose bound and poorer greedy performance.
Our adaptive βstrategy (Section 3.3) is thus not
merely an engineering choice, but is fundamentally
grounded in theory: by estimating the average intra-
candidate redundancy and expected set size for
each instance, adaptive βdynamically controls ϵ
in the ϵ-approximate submodularity bound. As
redundancy or set size increases, βis automatically
reduced, constraining ϵand preserving the near-
optimality of the greedy solution. This coupling
between parameter design and theoretical analysis
is a key novelty of our approach.
4.3.4 Greedy Guarantee under
ε-Approximate Submodularity
Prior work (e.g., Feige et al., 2011[28]; Horel &
Singer, 2016[29]) shows that for ϵ-approximately
submodular functions, the standard greedy algo-
rithm still provides strong guarantees:
f(S greedy)≥
1−1
e
OPT−kϵ
e(26)
where OPT is the global optimum and kis
the maximal set size. Thus, the solution quality
degrades only by an additive term proportional
toϵ—which, with adaptive β, remains tightly
bounded in practice.
4.3.5 Practical Interpretation
- In practice, chunk-to-chunk similarities are usu-
ally low due to semantic diversity in retrieved can-
didates, soδis typically≪1.
- The penalty ϵis therefore small, and greedy
selection closely tracks the theoretical optimum.
- If candidate redundancy increases or βis set
too high, the additive gap grows, which can impact
solution quality—but our adaptive βmechanism
is specifically designed to prevent this, by auto-
matically calibrating βbased on candidate pool
statistics (see Section 3.3).
4.3.6 Summary and Theoretical Takeaway
Our redundancy-aware objective exhibits ϵ-
approximate submodularity with
ϵ=βkδ(27)
9

- Adaptive βis essential for dynamically control-
lingϵand preserving greedy near-optimality, even
under shifting candidate pool properties.
- Greedy selection remains nearly optimal in
practical RAG scenarios.
- This analysis provides theoretical justification
for both the strong empirical performance of our
method and the use of greedy selection as a practi-
cal solution to redundancy-aware context selection.
5 Experiments
5.1 Experimental Setup
We conduct experiments on two distinct datasets
to evaluate the generality and effectiveness of our
method:
Domain-specific (Drug) Dataset:Our proprietary
dataset consists of drug-related documents from
the pharmaceutical domain, characterized by a high
level of content redundancy. Open-domain (Nat-
ural Questions) Dataset:We also evaluate our ap-
proach on the widely-used Natural Questions (NQ)
dataset, following the standard practice of using
the entire English Wikipedia as the retrieval corpus,
with articles split into fixed-length chunks.
Retrieval and Selection: For both datasets, we
use the Conan-embedding-v1 model to generate
dense embeddings for all chunks. Candidate pas-
sages are ranked and selected using either our
redundancy-aware greedy selection (AdaGReS) or
the similarity-only (top-k) baseline. For each redun-
dancy penalty parameter β, we first run AdaGReS
to determine the number of selected chunks k for
each query, and then apply the similarity-only base-
line with the same k, ensuring a fair and controlled
comparison.
Evaluation Metrics: Our primary evaluation met-
ric is Intersection-over-Union (IOU), which mea-
sures the overlap between selected passages and
the gold standard reference set. IOU in this context
reflects both informativeness and precision: higher
IOU indicates that the selected passages closely
match the relevant ground-truth segments, whereas
selecting excessive or irrelevant content will de-
crease IOU due to the expansion of the union set.
In addition, we conduct a qualitative human eval-
uation: for a set of representative queries, we com-
pare answers generated by GLM-4.5-air using con-
texts retrieved by AdaGReS and the similarity-only
baseline, respectively. This provides direct evi-
dence of the end-to-end impact of different context
selection strategies on QA performance.Reproducibility: All code, experiment scripts,
and the domain-specific dataset will be released at
[https://github.com/orderer0001/AdaGReS] and
[https://huggingface.co/datasets/Sofun2009/yzint-
drug-data] to ensure full transparency and
reproducibility.
5.2 Results and Analysis
5.2.1 Results on the Open Domain (NQ)
Dataset
In open-domain question answering tasks (e.g.,
Natural Questions), answer information is of-
ten scattered across different semantic segments,
which imposes higher requirements on the cover-
age breadth and anti-interference ability of context
selection. Although the overall redundancy of can-
didate segments in this scenario is lower than that
in specific domains, traditional retrieval methods
still tend to encounter the issue of "high similarity
but low information increment". Specifically, multi-
ple high-scoring segments essentially only provide
the same facts and fail to supplement effective new
information for the question answering task.
In such scenarios, the AdaGReS model still
demonstrates stable advantages, with its core lying
in the built-in dynamic βmechanism. This mecha-
nism can adjust the intensity of redundancy penalty
in real time based on the semantic distribution of
the candidate pool: when there is less redundant
information in the candidate pool, it moderately
relaxes the diversity constraints to retain informa-
tion relevance; when local semantics form dense
clusters (i.e., similar segments appear in concen-
tration), it automatically enhances deduplication
capability to avoid repeatedly selecting similar seg-
ments. This dynamically balanced design enables
the model to significantly reduce the redundancy
rate while maintaining a high recall rate, and also
effectively improves its support capability for multi-
hop reasoning and multi-entity related questions.
From a quantitative perspective, Figure 2 (Visu-
alization of Intersection over Union (IOU) Scores
between the Dynamic βMethod and the Base-
line Method in Open-Domain Question Answering
Tasks) presents the performance comparison be-
tween the two methods. Experimental results show
that the IOU score of the dynamic βmethod is con-
sistently higher than that of the baseline method
across all experimental scenarios; meanwhile, the
IOU score of the dynamic βmethod remains stably
at 0.15 and above throughout the entire testing pro-
10

Figure 2: Visualization of IOU (Intersection over Union) scores between the dynamic βmethod and the baseline
method.
Figure 3: Visualization of Intersection over Union (IOU) Scores between the Dynamic βMethod and Baseline
Methods under Different Redundancy Thresholds
cess. The aforementioned results not only reflect
the continuous performance advantage of the dy-
namic βmethod over the baseline method, but also
verify the stability of its IOU score, jointly sup-
porting the advantages of the AdaGReS model in
the accuracy and effectiveness of context selection
from both intuitive and quantitative perspectives.
5.2.2 Quantitative Results on Domain-specific
Dataset
In specific domain knowledge retrieval tasks, con-
sidering that the employed embedding encoder has
not been fine-tuned on the target domain dataset,
its ability to distinguish subtle semantic differences
within the domain is limited, which easily leads
to highly semantically overlapping context frag-
ments returned by retrieval. To alleviate the prob-
lem of information redundancy caused by this, we
explicitly introduce a fixed penalty coefficient for
redundant items themselves in the context scoring
function of AdaGReS. That is, when calculating
the comprehensive score of each candidate context,
its redundancy metric is multiplied by a preset fixed
weight less than 1, thereby directly weakening the
contribution of redundant components to the final
ranking. It should be emphasized that this penalty
acts on the interior of redundant items rather than
performing posterior weighting on the entire con-
text score, thus enabling more refined regulation ofthe balance between redundancy and relevance.
As shown in Figure 3, we compared the In-
tersection over Union (IOU) scores of the Ada-
GReS method (with the introduced fixed redun-
dancy penalty coefficient) and the baseline meth-
ods under different redundancy penalties (0.05, 0.2,
0.3). The experimental results show that even with-
out domain fine-tuning, this strategy can bring con-
sistent and stable performance improvements under
all penalty settings. Although the overall gain is
limited, its robustness verifies the effectiveness of
this mechanism in high-redundancy professional
scenarios.
The reasons for this limited improvement mainly
come from two aspects: First, vertical domain
knowledge itself has a high degree of semantic
concentration. Contexts from different sources of-
ten revolve around the same core concepts, terms,
or facts, resulting in natural content overlap and
a relatively limited space for effective redundancy
removal. Second, general pre-trained embedding
models, in the absence of domain adaptation, strug-
gle to accurately distinguish expressions that are
semantically similar but informationally comple-
mentary (such as different descriptions of the same
concept), leading to noise in the redundancy met-
rics themselves, which in turn limits the optimiza-
tion upper limit that the fixed penalty mechanism
can achieve.
11

Method AdaGReS Baseline
Input 1 “The MAF sensor detects the mass of air
flowing into the engine and sends this data
to the ECU. . . ”“The MAF sensor detects the mass of air
flowing into the engine and sends this data
to the ECU. . . ”
Input 2 “A mass airflow sensor (MAF) is used to
determine the mass flow rate of air enter-
ing a fuel-injected internal combustion en-
gine. . . ”“A mass airflow sensor (MAF) is used to
determine the mass flow rate of air enter-
ing a fuel-injected internal combustion en-
gine. . . ”
Input 3 “As a key detection component of the en-
gine’s air intake system. . . ”“The MAF sensor determines the mass of
air flowing into the engine’s air intake sys-
tem. The theory of operation of. . . ”
Response The mass air flow (MAF) sensor is a key
component in the engine’s air intake sys-
tem. It measures the mass flow rate of air
entering a fuel-injected internal combus-
tion engine and sends this data to the en-
gine control unit (ECU). The ECU uses this
information to accurately calculate the ap-
propriate amount of fuel to inject, ensuring
an optimal air-fuel mixture and efficient
combustion. Additionally, the MAF sensor
enables the ECU to adapt to environmental
changes—such as variations in altitude and
temperature—helping maintain consistent
engine performance. A faulty MAF sen-
sor can result in poor fuel economy, rough
idling, or drivability issues.The mass air flow (MAF) sensor measures
the mass of air entering the engine and
sends this information to the engine con-
trol unit (ECU). The ECU uses this data to
calculate the appropriate amount of fuel to
inject for optimal combustion.
Table 1: Comparison of generated text between AdaGReS and the baseline model on MAF sensor description tasks.
To sum up, under the realistic constraint of not
performing domain fine-tuning, by applying a fixed
penalty coefficient to the redundant items them-
selves, AdaGReS can suppress the interference of
semantic repetition on retrieval quality in a concise
and interpretable manner. While maintaining com-
putational efficiency, it effectively improves the
diversity and information coverage of the results,
providing a lightweight and robust redundancy con-
trol strategy for professional domain knowledge
retrieval.
5.2.3 Qualitative/Human Evaluation
As shown in Table 1, AdaGReS focuses closely on
the core requirement of “explaining the function
of the mass air flow meter”. All returned segments
are highly concentrated on functional descriptions
without redundant repetition, and each output di-
rectly responds to the query, thus ensuring the rel-
evance and conciseness of the results. In contrast,
the baseline exhibits obvious redundancy issues in
its retrieval results. It repeatedly outputs overlap-ping content related to the functions of the mass air
flow meter, without performing effective screening
and deduplication. These duplicated segments do
not add new information to the answer of the query,
but only lead to redundancy and bulkiness of the
results.
The experimental results verify the design goals
of AdaGReS: through the mechanisms of redun-
dancy awareness, global optimization, and adap-
tive weighting, this method achieves synergistic
improvement in context relevance, breadth of in-
formation coverage, and redundancy control ca-
pability. This optimization not only improves the
utilization efficiency of token resources, but also
enhances the knowledge integration and factual
reasoning capabilities of generative models in com-
plex retrieval-augmented scenarios.
5.2.4 Ablation and Efficiency
To further verify the role of each component,
we conducted ablation experiments, replacing the
adaptiveβwith a fixedβvalue.
12

Figure 4: Comparison forβ= 0.55, 0.65, 0.7 (greedy vs. simple)
Figures 4 show the IOU comparison between
AdaGReS (greedy) and the similarity-only (simple)
method under different redundancy penalty param-
eterβsettings. The dashed line represents Ada-
GReS, and the solid line represents the similarity-
only method; the legend on the right lists the aver-
age IOU of each method under the corresponding
βvalue.
Specifically, for each βsetting, we first use Ada-
GReS to dynamically determine the number of se-
lected blocks k, and then apply the similarity-only
baseline method with the same k value to ensure
fair comparison. In all figures, the dashed curve
represents AdaGReS (greedy selection), and the
solid curve represents the baseline method. In all
test configurations, AdaGReS achieved the highest
average IOU, fully verifying the robustness and
general advantages of this method.
The experimental results show that in the phar-
maceutical domain dataset, even when using a fixed
βvalue, the performance of AdaGReS is still bet-
ter than the baseline method, although its IOU de-
creases slightly. This indicates that explicitly penal-
izing redundancy in the context selection process
can effectively filter out more informative and di-
verse paragraphs, making them more consistent
with the reference answers, further confirming theeffectiveness and applicability of this mechanism.
We further evaluated the performance difference
between AdaGReS and the similarity-only (top-k)
selection method on the Natural Questions (NQ)
dataset. NQ contains a large number of real user
queries, and the retrieval corpus covers the full En-
glish Wikipedia, posing severe challenges to the
information coverage and generalization capabili-
ties of context selection algorithms in large-scale,
open-domain scenarios.
Consistent with the domain-specific experiments,
we used the Conan-embedding-v1 model to gener-
ate block-level embeddings, and for each query, we
used AdaGReS and the baseline method to filter k
paragraphs, where k is dynamically determined by
AdaGReS under each βparameter. The evaluation
still uses Intersection over Union (IOU) as the main
indicator.
Figures 5 show that AdaGReS also significantly
outperforms the baseline method on the NQ task,
especially in complex queries where answer in-
formation is scattered and multiple semantic per-
spectives need to be integrated, the IOU improve-
ment is more prominent. This indicates that Ada-
GReS is not only applicable to highly redundant
industry knowledge bases but also has excellent
robustness and information integration capabilities
13

Figure 5: Comparison forβ= 0.55, 0.65, 0.7 (greedy vs. simple)
in open-domain, large-scale corpus environments.
For some ultra-long or extremely difficult-to-cover
questions, the average IOU improvement of Ada-
GReS can reach 8–15 percentage points, further
verifying its generality and practical value.
5.3 Summary
Overall, the experimental results fully validate the
wide applicability and stable benefits of our pro-
posed AdaGReS method in specific domains with
highly redundant data (such as the pharmaceuti-
cal field) and complex retrieval environments in
open domains (such as natural question datasets).
Specifically, in terms of quantitative metrics: under
different settings of the redundancy penalty param-
eterβ, the IOU score of AdaGReS is consistently
higher than that of the similarity-based baseline
model. For complex open-domain queries, the IOU
achieves a significant improvement. Even without
domain fine-tuning, it can still achieve certain per-
formance improvements in high-redundancy spe-cific domain tasks. In terms of end-to-end genera-
tion effects: AdaGReS effectively avoids redundant
repetitions in the retrieval context, enabling the
generation model to produce more comprehensive,
concise, and information-rich responses compared
to the baseline model. In terms of human subjec-
tive evaluation: the retrieval results of AdaGReS
are more focused on core query needs, without re-
dundant overlapping content, and are significantly
superior to the baseline model in terms of relevance
and conciseness. In terms of computational effi-
ciency and robustness: AdaGReS achieves refined
redundancy control through an adaptive or fixed β
mechanism while maintaining high computational
efficiency, and its advantages remain stable under
various experimental configurations.
6 conclusion
In this work, we propose Adaptive Greedy Con-
text Selection via Redundancy-Aware Scoring for
Token-Budgeted RAG, a principled framework that
14

enables efficient and globally optimized context
selection for retrieval-augmented generation under
strict token constraints. Our approach features a
redundancy-aware set-level scoring function, com-
bined with an adaptive mechanism that dynami-
cally calibrates the relevance–redundancy tradeoff
based on candidate pool statistics and token budget.
This design effectively mitigates the long-standing
challenges of context redundancy and information
overlap, maximizing the informativeness and diver-
sity of selected evidence without requiring manual
parameter tuning. Empirical results and theoreti-
cal analysis demonstrate that our method consis-
tently improves token efficiency and answer quality
across various domains and scenarios.
In scenarios involving extremely long contexts
with highly non-uniform redundancy distributions,
our method may still present some minor limita-
tions. When only a very small subset of candidate
chunks is highly redundant while the rest are highly
diverse, the greedy selection strategy, despite its
theoretical guarantees, may occasionally fall short
of achieving an ideal balance between redundancy
suppression and coverage in practice. Nevertheless,
such extreme distributions are uncommon in typical
retrieval tasks, so this limitation does not affect the
general applicability of the method. In the future,
this aspect could be further improved through more
refined diversity modeling or multi-pass selection
strategies.
References
Lameck Mbangula Amugongo, Pietro Mascheroni,
Steven Brooks, Stefan Doering, and Jan Seidel. 2025.
Retrieval augmented generation for large language
models in healthcare: A systematic review.PLOS
Digital Health, 4(6):e0000877.
Jaime Carbonell and Jade Goldstein. 1998. The use of
mmr, diversity-based reranking for reordering doc-
uments and producing summaries. InProceedings
of the 21st annual international ACM SIGIR confer-
ence on Research and development in information
retrieval, pages 335–336.
Sangwoo Cho, Logan Lebanoff, Hassan Foroosh, and
Fei Liu. 2019. Improving the similarity mea-
sure of determinantal point processes for extrac-
tive multi-document summarization.arXiv preprint
arXiv:1906.00072.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval, pages 39–
48.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Jakub Lála, Odhran O’Donoghue, Aleksandar Shtedrit-
ski, Sam Cox, Samuel G Rodriques, and Andrew D
White. 2023. Paperqa: Retrieval-augmented gener-
ative agent for scientific research.arXiv preprint
arXiv:2312.07559.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Hui Lin and Jeff Bilmes. 2010. Multi-document sum-
marization via budgeted maximization of submod-
ular functions. InHuman Language Technologies:
The 2010 Annual Conference of the North Ameri-
can Chapter of the Association for Computational
Linguistics, pages 912–920.
Hui Lin and Jeff Bilmes. 2011. A class of submodular
functions for document summarization. InProceed-
ings of the 49th annual meeting of the association
for computational linguistics: human language tech-
nologies, pages 510–520.
Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru,
Amin Karbasi, Jan V ondrák, and Andreas Krause.
2015. Lazier than lazy greedy. InProceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 29.
Mudasir Mohd, Rafiya Jan, and Muzaffar Shah. 2020.
Text document summarization using word embed-
ding.Expert Systems with Applications, 143:112958.
George L Nemhauser, Laurence A Wolsey, and Mar-
shall L Fisher. 1978. An analysis of approximations
for maximizing submodular set functions—i.Mathe-
matical programming, 14(1):265–294.
15

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng.
2016. Ms marco: A human-generated machine read-
ing comprehension dataset.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Sil-
iang Tang. 2024. Graph retrieval-augmented gener-
ation: A survey.ACM Transactions on Information
Systems.
Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghad-
dam, Eric Zhang, Ryan Nguyen, Daniel Campos,
Nick Craswell, and Jimmy Lin. 2025. Ragnarök: A
reusable rag framework and baselines for trec 2024
retrieval-augmented generation track. InEuropean
Conference on Information Retrieval, pages 132–148.
Springer.
Markus Reuter, Tobias Lingenberg, Ruta Liepina,
Francesca Lagioia, Marco Lippi, Giovanni Sartor,
Andrea Passerini, and Burcu Sayin. 2025. To-
wards reliable retrieval in rag systems for large legal
datasets. InProceedings of the Natural Legal Lan-
guage Processing Workshop 2025, pages 17–30.
Yixuan Tang, Yuanyuan Shi, Yiqun Sun, and Anthony
Kum Hoe Tung. 2025. Uncovering the bigger picture:
Comprehensive event understanding via diverse news
retrieval. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 33927–33945.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry
Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny
Zhou, Quoc Le, and 1 others. 2024. Freshllms: Re-
freshing large language models with search engine
augmentation. InFindings of the Association for
Computational Linguistics: ACL 2024, pages 13697–
13720.
Kai Wei, Rishabh Iyer, and Jeff Bilmes. 2015. Submod-
ularity in data subset selection and active learning. In
International conference on machine learning, pages
1954–1963. PMLR.
Jin-ge Yao, Xiaojun Wan, and Jianguo Xiao. 2017. Re-
cent advances in document summarization.Knowl-
edge and Information Systems, 53(2):297–336.
Wan Zhang and Jing Zhang. 2025. Hallucination mitiga-
tion for retrieval-augmented large language models:
a review.Mathematics, 13(5):856.
16