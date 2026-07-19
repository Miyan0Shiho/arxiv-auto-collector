# QUBO-Optimized Evidence Selection for Retrieval-Augmented Question Answering with Unconventional Solvers

**Authors**: Rahul Singh, Madhav Vadlamani

**Published**: 2026-07-14 04:29:07

**PDF URL**: [https://arxiv.org/pdf/2607.12334v1](https://arxiv.org/pdf/2607.12334v1)

## Abstract
Retrieval-augmented question answering depends on selecting evidence passages that jointly support answer generation. However, many RAG pipelines rely on top-\(k\) ranking, where passages are selected mainly by individual relevance scores, even though multi-hop questions often require complementary evidence satisfying multiple information requirements. Recent LLM-based selectors address this by treating retrieval as set selection, but using an LLM for this intermediate stage can be costly and difficult to scale. In this work, we formulate evidence selection as a Quadratic Unconstrained Binary Optimization (QUBO) problem. Given a question, candidate passages, and decomposed information requirements, our method constructs an energy function that balances relevance, requirement coverage, support strength, redundancy, complementarity, and compactness. Low-energy solutions correspond to compact evidence subsets that cover the needed requirements while avoiding unnecessary or repetitive context. The selected passages are then passed to a downstream language model for answer generation, separating combinatorial evidence selection from semantic answer generation. We evaluate the proposed QUBO selector on HotpotQA and compare it with LLM-based set selectors and non-LLM baselines including BM25, relevance top-\(k\), maximal marginal relevance, hybrid lexical--semantic ranking, greedy coverage, and random selection. The QUBO selector achieves competitive exact-match and token-F1 performance relative to LLM-based selectors while providing a solver-compatible formulation for structured evidence selection. These results suggest that multi-hop evidence selection can be cast as discrete optimization, opening a path toward RAG pipelines where LLMs are reserved for semantic processing and answer generation, while context selection is handled by Ising/QUBO-compatible solvers.

## Full Text


<!-- PDF content starts -->

QUBO-Optimized Evidence Selection for Retrieval-Augmented Question
Answering with Unconventional Solvers
Rahul Singh
University of California Santa Barbara
Santa Barbara, California
rps@ucsb.eduMadhav Vadlamani*
Georgia Institute of Technology
Atlanta, Gerogia
mvadlamani6@gatech.edu
Abstract
Retrieval-augmented question answering de-
pends on selecting an appropriate set of evi-
dence passages before answer generation. How-
ever, many retrieval pipelines rely on top- k
ranking, where passages are selected primar-
ily by individual relevance scores, even though
multi-hop questions often require passages that
jointly satisfy multiple information require-
ments. Recent LLM-based methods address
this limitation by treating retrieval as a set-
selection problem, but using an LLM for this
intermediate selection stage can be expensive
and difficult to scale. In this work, we formu-
late evidence selection as a Quadratic Uncon-
strained Binary Optimization (QUBO) prob-
lem. Given a question, candidate passages,
and decomposed information requirements, our
method constructs an energy function that bal-
ances passage relevance, requirement coverage,
support strength, redundancy, complementarity,
and compactness. Low-energy configurations
correspond to evidence subsets that cover the
necessary requirements while avoiding unnec-
essary or repetitive context. The selected pas-
sages are then passed to a downstream language
model for answer generation, separating the
combinatorial evidence-selection step from the
semantic answer-generation step. We evaluate
the proposed QUBO selector on HotpotQA, a
multi-hop question-answering benchmark, and
compare it with LLM-based set selection as
well as non-LLM baselines including BM25,
relevance top- k, maximal marginal relevance,
hybrid lexical-semantic ranking, greedy cover-
age, and random selection. The QUBO selector
achieves competitive exact-match and token-
F1 performance relative to LLM-based set se-
lectors, while providing a solver-compatible
formulation for structured evidence selection.
These results suggest that multi-hop evidence
selection can be cast as a discrete optimiza-
tion problem, opening a path toward retrieval
pipelines in which LLMs are reserved primar-
*Equal Contributionily for answer generation while the selection
stage is executed by unconventional Ising ma-
chines, quantum annealers, quantum-inspired
optimizers, or gate-model quantum algorithms
for QUBO solving.
1 Introduction
Large language models (LLMs) have demonstrated
strong capabilities in natural language understand-
ing and generation, but their responses remain con-
strained by both the information encoded in their
parameters and the quality of the context provided
at inference time. Retrieval-Augmented Genera-
tion (RAG) addresses this limitation by coupling a
parametric language model with an external non-
parametric knowledge source, allowing generation
to be conditioned on retrieved evidence rather than
relying solely on memorized knowledge (Lewis
et al., 2020). This paradigm is especially useful for
knowledge-intensive tasks requiring factual ground-
ing, provenance, domain adaptation, and access to
dynamically updated information (Gao et al., 2024;
Barnett et al., 2024). However, RAG performance
is highly sensitive to the selected context: retrieval
errors, missing evidence, redundant passages, and
irrelevant distractors can degrade generation, espe-
cially in multi-hop question answering where the
answer may require combining complementary evi-
dence across passages (Yang et al., 2018; Yan et al.,
2024). Simply increasing the number of retrieved
passages is also insufficient, since long-context lan-
guage models may fail to use relevant evidence
depending on its position and salience (Liu et al.,
2024).
Most RAG systems therefore follow a retrieve-
then-rerank pipeline, where a sparse, dense, or
hybrid retriever produces candidate passages, a
reranker orders them, and the top- kpassages are
passed to the generator. While cross-encoder and
LLM-based listwise rerankers improve ranking
quality (Sun et al., 2023; Liu et al., 2025), they
arXiv:2607.12334v1  [cs.CL]  14 Jul 2026

typically optimize passage ordering rather than the
final evidence subset directly. This distinction mat-
ters because set-level properties such as require-
ment coverage, redundancy, complementarity, and
joint answer support are not always explicitly op-
timized (Lee et al., 2025). Diversity-aware meth-
ods such as maximal marginal relevance partially
address redundancy by balancing relevance and
novelty (Carbonell and Goldstein, 1998), but they
do not directly model query-specific information
requirements or optimize requirement-level cov-
erage within a unified subset-selection objective.
At the same time, large-scale retrieval and rerank-
ing can introduce substantial deployment cost, la-
tency, memory-capacity, and data-movement chal-
lenges (Hsu et al., 2026). These observations moti-
vate a set-selection view of RAG context construc-
tion (Lee et al., 2025; Jiang et al., 2026), where
the goal is not merely to select the highest-ranked
individual passages, but to construct a compact
evidence subset that collectively satisfies the infor-
mational requirements of the query.
This set-selection perspective also creates an
opportunity for specialized optimization hard-
ware. Unlike LLM-based reranking or prompt-
ing, which requires repeated transformer infer-
ence, a QUBO/Ising formulation exposes the con-
text selection decision as an explicit binary en-
ergy minimization problem. Such problems can be
mapped to quantum annealers and gate-based quan-
tum optimization algorithms, and can also be exe-
cuted by quantum-inspired hardware such as digital
annealers, coherent Ising machines, and CMOS-
compatible Ising machines.These platforms are
designed to search the energy landscape for low
energy configurations using physical dynamics
or hardware-specific optimization rules. Prior
work has reported problem dependent speedups for
digital annealers on fully connected QUBO/spin-
glass instances, scalable coherent-Ising-machine
behavior on large graph-optimization problems,
and improved execution-time or energy character-
istics for CMOS-compatible Ising designs (Ara-
mon et al., 2019; Haribara et al., 2016; Zhang
et al., 2022). While we do not assume a univer-
sal quantum speedup, this hardware direction is
attractive for large-scale RAG pipelines because
the candidate pool and pairwise passage interac-
tions can grow substantially with corpus size, mak-
ing the final context-selection step a natural tar-
get for low-latency and potentially energy-efficient
QUBO/Ising acceleration.In this work, we formulate requirement-aware
passage selection as a Quadratic Unconstrained Bi-
nary Optimization (QUBO) problem. QUBO pro-
vides a standard framework for binary optimization
with linear utility terms and quadratic interaction
terms (Glover et al., 2022), making it well suited
to RAG context construction: each candidate pas-
sage is represented by a binary variable, while the
objective rewards relevance, requirement coverage,
support strength, and complementarity, and penal-
izes redundancy and unnecessary context length.
Low-energy solutions therefore correspond to com-
pact evidence subsets selected for their collective
utility rather than only their individual relevance
scores.
The proposed formulation adds a structured opti-
mization layer between retrieval and answer gener-
ation. It does not replace the retriever or the genera-
tor; instead, it replaces the intermediate context-
selection step with an explicit set-optimization
problem over an already retrieved candidate pool.
The selected passages are then passed to a down-
stream language model for answer generation, sep-
arating combinatorial evidence selection from se-
mantic answer synthesis. To the best of our knowl-
edge, this is the first work to formulate requirement-
aware RAG context selection as a QUBO prob-
lem, enabling the selection stage to be mapped to
Ising/QUBO-compatible solvers such as simulated
annealers, quantum annealers, digital annealers, co-
herent Ising machines, CMOS/SRAM-based Ising
machines, and gate-model quantum optimization
algorithms after suitable mapping (Ceselli and Pre-
moli, 2023; ¸ Seker et al., 2022; Wei et al., 2026;
Vadlamani et al., 2026; Perlin et al., 2026; He et al.,
2025). This solver-compatible representation is par-
ticularly relevant as candidate pools and retrieval
corpora grow: specialized QUBO and Ising ac-
celerators can target the combinatorial selection
step directly, offering a path toward lower-latency
and potentially more energy-efficient context se-
lection than repeated LLM-based reranking or set-
selection calls.
The main contributions of this paper are summa-
rized as follows:
•We formulate RAG context construction as
a requirement-aware binary subset-selection
problem, where the selected evidence is op-
timized at the set level rather than obtained
through independent top-kranking.
•We derive a QUBO objective for evidence se-

lection that jointly encodes passage relevance,
requirement coverage, support strength, redun-
dancy reduction, complementarity, compact-
ness, and selection-budget constraints within
a single solver-compatible energy function.
•We present a modular RAG pipeline that sep-
arates semantic preprocessing, combinatorial
context selection, and answer generation: se-
mantic scores define the QUBO instance, the
solver selects the context subset, and a down-
stream language model generates the final an-
swer.
•We evaluate the proposed approach on multi-
hop question answering using both down-
stream answer-generation metrics and set-
level evidence-selection metrics, comparing
against SetR-style LLM set selectors and non-
LLM baselines including BM25, relevance
top-k, maximal marginal relevance, hybrid
lexical–semantic ranking, greedy coverage,
and random selection.
2 Background
The proposed method modifies the context-
selection stage of Retrieval-Augmented Gen-
eration (RAG). Rather than reintroducing the
full RAG pipeline, this section summarizes the
context-selection problem and then introduces
Quadratic Unconstrained Binary Optimization
(QUBO), which provides the mathematical basis
for the proposed formulation.
2.1 RAG Context Selection
RAG augments a parametric language model with
retrieved non-parametric evidence, allowing an-
swer generation to be conditioned on external con-
text rather than solely on memorized model param-
eters (Lewis et al., 2020). In a standard pipeline, an
input query is used to retrieve or rerank a candidate
pool of passages, and the final context is commonly
formed by selecting the top- Kpassages according
to a relevance or reranking score. If Cqdenotes the
candidate pool for query q, this selection can be
written as
Stop-K
q = arg max
S⊆C q,|S|=KX
pi∈Sρ(q, p i),(1)
where ρ(q, p i)is the score assigned to passage pi.
This additive objective selects passages accordingto their individual scores, but it does not explicitly
optimize interactions among selected passages.
This limitation is important in multi-hop and
requirement-sensitive question answering, where
the answer may require combining complemen-
tary evidence distributed across multiple pas-
sages (Yang et al., 2018). A passage with high
individual relevance may repeat evidence already
present in another passage, while a moderately
ranked passage may be necessary because it sup-
plies a missing bridge fact or completes an evi-
dence chain. Recent work has therefore argued
for set-level context construction, where passages
are selected according to how well they collec-
tively satisfy the information needs of the query
rather than only by their independent relevance
scores (Lee et al., 2025; Jiang et al., 2026). This
motivates treating RAG context construction as a
binary subset-selection problem.
2.2 Quadratic Unconstrained Binary
Optimization
Quadratic Unconstrained Binary Optimization
(QUBO) is a standard formulation for discrete opti-
mization over binary variables (Glover et al., 2022).
Given a binary vector
x= [x 1, x2, . . . , x n]⊤, x i∈ {0,1},(2)
a QUBO problem is commonly written as
min
x∈{0,1}nE(x) =x⊤Qx,(3)
where Q∈Rn×nis the QUBO coefficient ma-
trix. The diagonal entries Qiiencode linear costs
or rewards because x2
i=xifor binary variables,
while the off-diagonal entries encode pairwise in-
teractions between variables. Equivalently, using
an upper-triangular convention,
E(x) =nX
i=1Qiixi+X
1≤i<j≤nQijxixj.(4)
QUBO objectives are closely related to Ising
model Hamiltonians (Lucas, 2014) through the
affine transformation
si= 2x i−1, s i∈ {−1,+1}.(5)
This relationship makes QUBO a solver-
compatible representation for simulated annealing
and for Ising/QUBO-oriented optimization back-
ends, including quantum annealers, digital anneal-
ers, coherent Ising machines, CMOS/SRAM-based

Ising machines, and gate-model quantum optimiza-
tion algorithms after suitable mapping (Ceselli
and Premoli, 2023; ¸ Seker et al., 2022; Wei
et al., 2026; Vadlamani et al., 2026; Perlin et al.,
2026; He et al., 2025). In this work, the QUBO
representation is used to encode requirement-aware
passage selection, and the construction of the
corresponding matrixQis detailed in Sec. 3.
3 Methods
This section presents the proposed framework for
set-based context selection in RAG. The goal is to
select a compact subset of passages that provides
sufficient evidence for answer generation. In con-
trast to retrieve-then-rerank pipelines that typically
pass the top- kranked passages to the generator,
the proposed framework treats context construction
as a binary subset-selection problem. This allows
the final context to be optimized with respect to
passage relevance, requirement coverage, support
strength, complementarity, redundancy, and com-
pactness.
3.1 Pipeline Overview
For an input query, the proposed pipeline proceeds
in five stages. First, the query is decomposed into
information requirements representing the evidence
conditions needed to answer it. Second, a candi-
date passage pool is obtained from a retriever, a
dataset-provided context source, or a hybrid lexical–
semantic retrieval mechanism. Third, a scoring
stage assigns coefficients describing passage rele-
vance, requirement support, pairwise redundancy,
and pairwise complementarity. Fourth, these co-
efficients parameterize a QUBO objective whose
solution identifies the selected passage subset. Fi-
nally, the selected passages are concatenated with
the original query and passed to a language model
for answer generation. This design inserts a modu-
lar context-optimization layer between retrieval and
generation: the retriever produces a broad candi-
date pool, the QUBO solver performs the final set-
selection decision, and the downstream language
model generates the answer from the optimized
context. Fig. 1 summarizes this workflow.
3.2 QUBO Formulation for RAG Context
Selection
We formulate the final context-selection stage as
a binary optimization problem. Given an input
query q, the objective is to select a compact pas-
sage subset that provides useful and non-redundantevidence for answer generation under a limited con-
text budget.
LetP={p 1, . . . , p N}denote the candidate pas-
sages for the query, and let R={r 1, . . . , r M}de-
note the generated information requirements. Each
requirement rjis assigned a nonnegative impor-
tance weight wj, and each passage piis assigned
a relevance score si. We introduce a binary selec-
tion variable xi∈ {0,1} , where xi= 1indicates
that passage piis selected for the final context and
xi= 0otherwise.
The scoring stage also assigns a requirement-
support strength Sji∈ {0,1,2} , where Sji= 0
indicates that passage pidoes not support require-
ment rj,Sji= 1indicates partial, bridge, or con-
textual support, and Sji= 2indicates direct sup-
port. To distinguish partial evidence from direct
evidence, we define
AP
ji=1[S ji≥1], AD
ji=1[S ji= 2].(6)
Here, AP
jiindicates that passage piprovides at least
partial support for requirement rj, while AD
jiindi-
cates that the passage provides direct support.
For each requirement, we introduce two binary
coverage variables
yP
j∈ {0,1}, yD
j∈ {0,1}.(7)
The variable yP
j= 1indicates that requirement rj
is partially covered by at least one selected pas-
sage, while yD
j= 1indicates that requirement rj
is directly covered by at least one selected passage.
These variables are constrained by
yP
j≤NX
i=1AP
jixi, yD
j≤NX
i=1AD
jixi,(8)
We also impose the consistency constraint yD
j≤
yP
j, so that direct coverage cannot be activated
unless partial coverage is also active. This di-
rect/partial separation prevents multiple weak or
bridge passages from being treated as equivalent to
direct evidence.
To discourage selecting passages that do not sup-
port any generated requirement, we define
ui=(
1, S ji= 0∀j∈ {1, . . . , M},
0,otherwise.(9)
Thus, ui= 1identifies passages with no verified
requirement support.

?Query
Input Query✓Requirements
Weighted Information Requirements
≣Passages
Retrieved / Candidate DatasetΣScoring
Relevance, Support, Redundancy,
Complementarity Set-based Cost Function Selected context passed to LLMQQUBO AAnswer
Optional QUBO Solver Backends
0
1
1
010
Ising Machineq1
q2H
HZZ
ZZZZq0 H Rx
Rx
Rx
Gate-based QCIterationE
Classical SA Quantum AnnealerFigure 1: Overview of the proposed pipeline. Given an input query, the pipeline first derives query-specific
information requirements and constructs a candidate passage pool. A scoring stage defines numerical coefficients
such as passage relevance si, requirement support Sji, redundancy ˜dik, and complementarity bik. These coefficients
define a QUBO objective whose low-energy solution determines the selected passage subset. The resulting binary
solution identifies the optimized context, which is then passed to a language model for answer generation.
The redundancy penalty is modulated by
requirement-support overlap. Let Si={r j∈
R:S ji>0} denote the set of requirements sup-
ported by passage pi. Given a base redundancy
coefficient dikbetween passages piandpk, we
define the support-aware redundancy coefficient
˜dik=dikρik, where
ρik=(
J(Si,Sk),S i∩ Sk̸=∅,
ρdiff,S i∩ Sk=∅,(10)
with0≤ρ diff<1, and
J(Si,Sk) =|Si∩ Sk|
|Si∪ Sk|.(11)
Passages supporting overlapping requirements
therefore receive stronger redundancy penalties,
while passages supporting different requirements
are penalized less strongly.
We also define a pairwise complementarity co-
efficient bik≥0, which rewards selecting pairs of
passages that support different or complementary
requirements. This term encourages the selected
set to cover multiple information needs instead of
repeatedly selecting passages that provide similar
evidence.
For compactness, letP
i,P
j, andP
i<kdenote
sums over i= 1, . . . , N ,j= 1, . . . , M , and 1≤i < k≤N , respectively. The constrained selection
problem can be written as
max
x,yP,yDX
i
αsi+ηX
jwjSji−κu i−µ
xi
+X
jwj 
γPyP
j+γDyD
j
+X
i<k
δbik−β˜dik
xixk,
s.t.yP
j≤X
iAP
jixi, yD
j≤X
iAD
jixi,
yD
j≤yP
j,X
ixi≤K.
(12)
where the first three constraints hold for all j=
1, . . . , M ,xi∈ {0,1} for all i= 1, . . . , N , and
yP
j, yD
j∈ {0,1}for allj= 1, . . . , M.
The first line of the objective collects passage-
level terms: relevance reward, support-strength
reward, unsupported-passage penalty, and size
penalty. The second line rewards partial and
direct requirement coverage. The third line
contains pairwise set-level interactions, reward-
ing complementary passage pairs and penaliz-
ing redundant passage pairs. The coefficients
α, η, γ P, γD, δ, β, κ, µ≥0 control the relative

strength of these rewards and penalties. For
minimization-based QUBO solvers, the equivalent
constrained minimization problem is obtained by
replacing the objective in (12) with its additive in-
verse, while leaving the constraint set unchanged.
Given the binary variable domains defined above,
the remaining feasibility conditions in (12) are all
inequality constraints. We convert these inequali-
ties to equalities using nonnegative slack variables:
X
ixi+qK−K= 0,
yP
j−X
iAP
jixi+vP
j= 0,
yD
j−X
iAD
jixi+vD
j= 0,
yD
j−yP
j+m j= 0.(13)
The last three equalities hold for all j= 1, . . . , M .
The slack variables qK,vP
j,vD
j, and mjare non-
negative integers; in the implemented QUBO, each
is represented using a binary expansion. The corre-
sponding quadratic penalty is
P=λ K X
ixi+qK−K!2
+λPX
j 
yP
j−X
iAP
jixi+vP
j!2
+λDX
j 
yD
j−X
iAD
jixi+vD
j!2
+λHX
j 
yD
j−yP
j+m j2.(14)
LetF(x,yP,yD)denote the maximization ob-
jective in (12). The final unconstrained QUBO
objective is then
min
x,yP,yD,z−F(x,yP,yD) +P,(15)
where zdenotes the binary variables used to encode
the slack variables, and λK, λP, λD, λH>0 are
penalty weights.
The coefficients used in the objective, includ-
ing relevance scores, requirement weights, support
labels, redundancy coefficients, complementarity
coefficients, and unsupported-passage indicators,
are fixed before optimization. After the slack vari-
ables in (13) are encoded using binary variables,
each residual in (14) is linear in the full binaryvariable vector. Therefore, the squared penalties
introduce only linear and pairwise quadratic terms,
using z2=z for binary variables. Consequently,
the unconstrained objective in (15) can be written
in standard QUBO matrix form as
min
v∈{0,1}nQv⊤Qv,(16)
where vconcatenates the passage-selection vari-
ables, coverage variables, and slack-encoding vari-
ables; Qis the resulting QUBO coefficient matrix;
andn Qis the total number of binary variables.
3.3 Answer Generation
Letv⋆denote the binary solution returned by the
QUBO solver. The selected passage subset is re-
covered from the passage-selection variables as
P⋆={p i∈ P:x⋆
i= 1}.(17)
The selected passages are then assembled into the
final context and concatenated with the original
query to form the answer-generation prompt.
4 Results
We evaluate the proposed QUBO selector from
three perspectives. First, we perform a
development-set sweep over QUBO configurations
to select a fixed operating point. Second, we com-
pare answer-generation performance against SetR-
style LLM set selectors and non-LLM selection
baselines. Third, we analyze the selected evidence
sets directly through ablation experiments. Un-
less otherwise stated, we use an integer-quantized
QUBO representation in which real-valued scor-
ing coefficients are scaled and rounded to integer
values before solving. This quantization controls
coefficient precision and dynamic range, which
is important for finite-precision QUBO and Ising
solvers, including hardware-oriented implementa-
tions with limited coefficient resolution.
4.1 Experimental Setup
We evaluate the proposed selector on the HotpotQA
distractor setting (Yang et al., 2018), where each
example contains a multi-hop question, a reference
answer, and context passages containing both sup-
porting and distractor evidence. To make evidence
selection explicit, each titled context is flattened
into sentence-level candidate passages, and all com-
pared methods operate on the same candidate pool
P={p i}N
i=1. All LLM-related components use

Cfg. Changed setting EM F1 Cov. #Sel.
C0baseline 0.6000 0.7151 0.9883 4.39
C1weak constraints0.6200 0.74850.9883 4.61
C2strong constraints 0.6000 0.7218 0.9833 4.40
C3K= 30.6000 0.7051 0.9700 3.05
C4K= 40.6000 0.7151 0.9700 3.70
C5K= 60.6100 0.7318 0.9883 5.01
C6low cov. pen. 0.6000 0.7151 0.9883 4.37
C7high cov. pen. 0.5900 0.7185 0.9783 4.38
C8very high cov. pen. 0.6100 0.7418 0.9883 4.51
C9low budget pen. 0.6100 0.7465 0.9883 4.78
C10high budget pen. 0.5800 0.7154 0.9733 4.54
C11low size pen. 0.6000 0.7218 0.9883 4.37
C12high size pen. 0.6000 0.7151 0.9883 4.40
Table 1: Development-set sweep over QUBO configura-
tions using simulated annealing. Cov. denotes average
requirement coverage and #Sel. denotes the average
number of selected passages.
the same local model, Ollama 3.6:31b , includ-
ing information-requirement generation, passage-
support labeling, LLM-based selector baselines,
and final answer generation. This setup keeps the
candidate pool and generator fixed across methods,
so differences in performance primarily reflect the
context-selection strategy. In the proposed pipeline,
the LLM provides semantic measurements used to
construct the QUBO, while the final passage sub-
set is selected by the QUBO solver. This modular
design separates semantic preprocessing, combi-
natorial context selection, and answer generation,
allowing each component to be replaced indepen-
dently.
4.2 Development-Set Configuration Sweep
We first evaluate a development-set sweep over
QUBO configurations. All configurations use the
same base objective weights α= 10 ,β= 10 ,
γP= 4,γD= 10 ,δ= 8 , support reward 10,
unsupported-passage penalty 20, and ρdiff= 0.25 .
The sweep varies constraint penalties, passage bud-
get, and compactness penalty. Table 1 summarizes
the resulting answer-level and set-level behavior.
The QUBO selector is relatively stable across
nearby settings: most configurations maintain high
requirement coverage while varying in compact-
ness and downstream answer quality. The best
answer-level performance is obtained by C1, which
weakens the budget and coverage penalties relative
to the baseline while preserving high coverage. Re-
ducing the budget to K= 3 produces more com-
pact contexts but lowers F1, while increasing the
budget to K= 6 selects more passages without
improving over C1. Very large budget penalties
also reduce performance, suggesting that overly
rigid constraint enforcement can interfere with the
selector’s ability to balance coverage, support, andMethod EM F1 Cov. #Sel.
QUBO selector 0.6500 0.78660.98933.862
SetR 0.6540 0.7930 0.9847 2.320
SetRO0.6580 0.80260.98452.306
Table 2: Comparison with SetR-style LLM set selectors
on 500 HotpotQA examples. All methods use the same
candidate pool and answer-generation model.
compactness.
4.3 Comparison with Prior LLM-Based Set
Selection
We next compare the proposed QUBO selector
against SetR-style LLM set-selection baselines in-
spired by SetR (Lee et al., 2025). Since the official
SetR model is not publicly available to us, we im-
plement prompt-based SetR-style selectors using
the same local qwen3.6:35b model used elsewhere
in the pipeline.SetRdenotes the SetR-aligned
baseline in which the model is given the precom-
puted information requirements and selects at most
Kpassages from the same candidate pool. And,
SetROdenotes the original-prompt-style baseline
in which the model infers information requirements
from the question and candidate passages before
selecting the final evidence set. Thus, SetR tests
LLM-based selection with the same requirement
structure used by the QUBO formulation, while
SetRO tests a closer SetR-style prompting setup
where requirements are inferred inside the LLM
prompt.
Table 2 shows that the QUBO selector is compet-
itive with direct LLM-based set selection. SetR and
SetRO obtain slightly higher answer-level scores,
but the margin is small: the QUBO selector is
within 0.004 EM and 0.0064 F1 of SetR, and
within 0.008 EM and 0.0160 F1 of SetRO. At the
same time, the QUBO selector achieves the highest
requirement coverage, indicating that it tends to
select contexts that more explicitly satisfy the gen-
erated information requirements. This supports the
intended role of the QUBO layer: it replaces direct
LLM set selection with explicit subset optimization
while preserving competitive downstream answer
quality.
4.4 Comparison with Non-LLM Selectors
We also compare against non-LLM selection base-
lines: relevance top- k, maximal marginal relevance
(MMR), hybrid lexical–semantic ranking, BM25,
greedy coverage, and random selection (Robert-
son and Zaragoza, 2009; Carbonell and Gold-

Method EM F1
Relevance top-k0.6500 0.7718
MMR-Relevance0.65000.7651
QUBO selector0.65000.7585
Greedy Coverage Compact 0.6100 0.7296
Hybrid Rel.+BM25 top-k0.5900 0.7320
MMR-BM25 0.4100 0.5061
BM25 top-k0.4000 0.4941
Random 0.0700 0.0786
Table 3: Answer-level comparison with non-LLM selec-
tion baselines on 100 HotpotQA examples.
stein, 1998; Karpukhin et al., 2020; Ma et al.,
2022). These methods provide reference points
for relevance-only selection, diversity-aware selec-
tion, lexical retrieval, coverage-driven selection,
and chance-level selection.
Table 3 shows that the QUBO selector matches
the best exact-match score among non-LLM base-
lines and remains close to relevance top- kand
MMR-Relevance in F1. The QUBO selector sub-
stantially outperforms BM25-based and random
selection methods, confirming that the selected
contexts are not simply the result of lexical over-
lap or chance. Although relevance top- kobtains
the highest F1 in this subset, it does not explic-
itly optimize requirement coverage, redundancy,
complementarity, or compactness. The QUBO for-
mulation therefore provides a structured alternative
to relevance-only context construction while main-
taining competitive answer-level performance.
4.5 Ablation Study
We ablate the main components of the QUBO ob-
jective to examine how different terms affect both
downstream answer generation and the structure of
the selected evidence set. The answer-level abla-
tion measures whether the selected context enables
the generator to produce the correct answer, while
the set-level ablation isolates the behavior of the
selector itself.
4.5.1 Answer-Level Ablation
Table 4 reports exact match and token-level F1 for
the full QUBO objective and its ablated variants.
The results show that relevance and requirement
coverage are the strongest contributors to answer-
level performance on this subset. The relevance
and coverage only variant obtains the highest F1,
while removing support strength produces the low-
est F1 among the tested variants. The full QUBO
objective is slightly lower in answer F1 than some
ablations, indicating that optimizing for richer set-Variant EM F1
Rel.+Coverage only0.6600 0.7751
No Complementarity0.66000.7685
No Redundancy0.66000.7685
No Size Penalty0.66000.7685
No Unsupported Penalty0.66000.7685
Full QUBO 0.6500 0.7585
No Support Strength 0.6500 0.7518
Table 4: Answer-level ablation of QUBO objective com-
ponents on 100 HotpotQA examples.
Variant WC CS MU Rel. #Sel.
Full QUBO 0.9852 0.96930.50567.4156 3.5667
No Redundancy 0.9852 0.9693 0.5039 7.3856 3.6000
No Complementarity 0.9852 0.96930.5056 7.4222 3.5333
No Support Strength 0.9852 0.96930.50567.41393.5333
No Size Penalty 0.9852 0.96930.50567.4156 3.5667
No Unsupported Penalty 0.9852 0.9693 0.4417 6.9017 3.8333
Rel.+Coverage only 0.9852 0.9693 0.4194 6.5778 3.9667
Table 5: Set-level ablation results. WC denotes
weighted coverage, CS denotes capped support, MU
denotes marginal utility fraction, Rel. denotes average
relevance, and #Sel. denotes the average number of
selected passages.
level structure does not always translate directly
into the best short-answer score. This indicates
that the objective terms that improve evidence-set
structure do not always yield proportional gains in
exact-match or token-F1, which are indirect mea-
sures of selector quality.
4.5.2 Set-Level Ablation
To isolate the behavior of the selector itself, Table 5
evaluates the selected evidence sets directly using
weighted coverage, capped support, marginal util-
ity, average relevance, and selected-context size.
The set-level results show that all variants main-
tain the same weighted coverage and capped sup-
port, indicating that the requirement-coverage con-
straints are robust across these ablations. The dif-
ferences instead appear in compactness, marginal
utility, and average relevance. Removing the
unsupported-passage penalty or using only rele-
vance and coverage increases the number of se-
lected passages while reducing marginal utility and
average relevance, suggesting that these variants
include more low-utility context even when they
preserve coverage. In contrast, the full QUBO
maintains high coverage while selecting a more
compact and higher-utility evidence set. Together,
the answer-level and set-level ablations show that
simple relevance-and-coverage objectives can be
strong for answer generation, while the full QUBO
better controls the structure and compactness of the
selected evidence set.

5 Conclusion
We presented a QUBO-based formulation for
requirement-aware context selection in retrieval-
augmented question answering. Instead of se-
lecting passages solely by independent relevance
scores or asking an LLM to directly choose the final
context, the proposed method casts evidence selec-
tion as a binary subset-optimization problem. Can-
didate passages are represented by binary variables,
and the objective combines passage relevance, re-
quirement coverage, support strength, redundancy,
complementarity, compactness, and budget con-
straints within a single solver-compatible formu-
lation. The selected passages are then passed to a
downstream language model for answer generation,
separating combinatorial evidence selection from
semantic answer synthesis.
Experiments on HotpotQA show that the QUBO
selector achieves competitive answer-level perfor-
mance relative to SetR-style LLM selectors and
non-LLM baselines while providing a transparent
optimization objective for the intermediate context-
selection step. The QUBO selector is within a small
margin of SetR-style LLM baselines in exact match
and token-level F1, and it achieves strong require-
ment coverage across experiments. Ablation results
further show that relevance and requirement cover-
age are the dominant contributors to downstream
answer accuracy, while additional terms such as
unsupported-passage penalties, redundancy, com-
plementarity, and compactness shape the quality
and structure of the selected evidence set.
These results suggest that multi-hop RAG con-
text construction can be treated as a structured dis-
crete optimization problem rather than only as a
ranking or prompting problem. The formulation is
modular: semantic preprocessing, QUBO solving,
and answer generation can be replaced indepen-
dently. This opens a path toward retrieval pipelines
in which LLMs are reserved primarily for seman-
tic measurement and answer generation, while the
final combinatorial selection step is handled by
classical, quantum-inspired, or hardware-oriented
QUBO solvers. Future work includes evaluating
larger and more diverse QA benchmarks, reduc-
ing reliance on LLM-based semantic preprocess-
ing, learning the QUBO coefficients from data, and
studying the behavior of the formulation on spe-
cialized Ising/QUBO hardware.6 Acknowledgment
The authors gratefully acknowledge Prof. Xifeng
Yan for his thoughtful discussions, valuable feed-
back, and insightful comments, which helped im-
prove the presentation and positioning of this work.
References
Maliheh Aramon, Gili Rosenberg, Elisabetta Valiante,
Toshiyuki Miyazawa, Hirotaka Tamura, and Hel-
mut G. Katzgraber. 2019. Physics-inspired optimiza-
tion for quadratic unconstrained problems using a
digital annealer.Frontiers in Physics, V olume 7 -
2019.
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu,
Zach Brannelly, and Mohamed Abdelrazek. 2024.
Seven failure points when engineering a retrieval
augmented generation system. InProceedings of
the IEEE/ACM 3rd International Conference on AI
Engineering - Software Engineering for AI, CAIN
’24, page 194–199, New York, NY , USA. Association
for Computing Machinery.
Jaime Carbonell and Jade Goldstein. 1998. The use of
mmr, diversity-based reranking for reordering doc-
uments and producing summaries. InProceedings
of the 21st Annual International ACM SIGIR Confer-
ence on Research and Development in Information
Retrieval, SIGIR ’98, page 335–336, New York, NY ,
USA. Association for Computing Machinery.
Alberto Ceselli and Marco Premoli. 2023. On good
encodings for quantum annealer and digital optimiza-
tion solvers.Scientific Reports, 13(1):5628.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey.Preprint,
arXiv:2312.10997.
Fred Glover, Gary Kochenberger, Rick Hennig, and
Yu Du. 2022. Quantum bridge analytics i: a tutorial
on formulating and using qubo models.Annals of
Operations Research, 314(1):141–183.
Yoshitaka Haribara, Shoko Utsunomiya, and Yoshi-
hisa Yamamoto. 2016.A Coherent Ising Machine
for MAX-CUT Problems: Performance Evaluation
against Semidefinite Programming and Simulated An-
nealing, pages 251–262. Springer Japan, Tokyo.
Zichang He, David Amaro, Ruslan Shaydulin, and
Marco Pistoia. 2025. Performance of quantum ap-
proximate optimization with quantum error detection.
Communications Physics, 8(1):217.
Po-Kai Hsu, Weihong Xu, Qunyou Liu, Tajana Rosing,
and Shimeng Yu. 2026. Haven: High-bandwidth
flash augmented vector engine for large-scale approx-
imate nearest-neighbor search acceleration.Preprint,
arXiv:2603.01175.

Yi Jiang, Sendong Zhao, Jianbo Li, Bairui Hu, Yan-
rui Du, Haochun Wang, and Bing Qin. 2026. Op-
tiset: Unified optimizing set selection and rank-
ing for retrieval-augmented generation.Preprint,
arXiv:2601.05027.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Dahyun Lee, Yongrae Jo, Haeju Park, and Moontae
Lee. 2025. Shifting from ranking to set selection
for retrieval augmented generation. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 17606–17619, Vienna, Austria. Association
for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Qi Liu, Bo Wang, Nan Wang, and Jiaxin Mao. 2025.
Leveraging passage embeddings for efficient listwise
reranking with large language models. InProceed-
ings of the ACM on Web Conference 2025, WWW
’25, page 4274–4283, New York, NY , USA. Associa-
tion for Computing Machinery.
Andrew Lucas. 2014. Ising formulations of many np
problems.Frontiers in Physics, V olume 2 - 2014.
Xueguang Ma, Kai Sun, Ronak Pradeep, Minghan Li,
and Jimmy Lin. 2022. Another look at dpr: Repro-
duction of training and replication of retrieval. In
Advances in Information Retrieval: 44th European
Conference on IR Research, ECIR 2022, Stavanger,
Norway, April 10–14, 2022, Proceedings, Part I, page
613–626, Berlin, Heidelberg. Springer-Verlag.
Michael A. Perlin, Zichang He, Anthony Alexiades
Armenakas, Pablo Andres-Martinez, Tianyi Hao,
Dylan Herman, Yuwei Jin, Karl Mayer, Chris
Self, David Amaro, Ciaran Ryan-Anderson, and
Ruslan Shaydulin. 2026. Fault-tolerant execution
of error-corrected quantum algorithms.Preprint,
arXiv:2603.04584.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Found. Trends Inf. Retr., 3(4):333–389.Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023. Is ChatGPT good at search?
investigating large language models as re-ranking
agents. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 14918–14937, Singapore. Association for
Computational Linguistics.
Madhav Vadlamani, Rahul Singh, Yuyao Kong, Zheng
Zhang, and Shimeng Yu. 2026. Scalable digital
compute-in-memory ising machines for robustness
verification of binary neural networks.Preprint,
arXiv:2603.05677.
Hai Wei, Chengjun Ai, Putuo Guo, Bingjie Jia, Lixin
Yuan, Hanquan Song, Shaobo Chen, Chongyu Cao,
Jie Wu, Chao Ju, Yin Ma, Jintao Fan, Minglie Hu,
Chuan Wang, and Kai Wen. 2026. A versatile co-
herent ising computing platform.Light: Science &
Applications, 15(1):74.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
Preprint, arXiv:2401.15884.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Yiqiao Zhang, Richard Afoakwa, Uday Kumar Reddy
Vengalam, Michael Huang, and Zeljko Ignjatovic.
2022. A cmos compatible bistable resistively-
coupled ising machine-brim. In2022 IEEE Interna-
tional Symposium on Circuits and Systems (ISCAS),
pages 1665–1669.
Oylum ¸ Seker, Neda Tanoumand, and Merve Bodur.
2022. Digital annealer for quadratic unconstrained bi-
nary optimization: A comparative performance anal-
ysis.Applied Soft Computing, 127:109367.