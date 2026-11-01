# p-DCIM: A 28nm Probabilistic SRAM DCIM based Ising Machine for Near-Memory Context Selection in LLM-RAG Pipeline

**Authors**: Madhav Vadlamani, Rahul Singh, Yuyao Kong, Haomei Liu, Zheng Zhang, Shimeng Yu

**Published**: 2026-10-27 11:10:26

**PDF URL**: [https://arxiv.org/pdf/2610.37033v1](https://arxiv.org/pdf/2610.37033v1)

## Abstract
We present a fabricated SRAM DCIM Ising machine that performs set-level context selection for large language model (LLM) with retrieval-augmented generation (RAG). In RAG, the evidence-selection stage is increasingly delegated to repeated language-model reranking, which adds substantial energy to the inference pipeline. We reformulate evidence selection as a quadratic unconstrained binary optimization (QUBO) problem and solve it on a dual-mode digital compute-in-memory (DCIM) macro: the same multiply-and-accumulate datapath computes the embedding-similarity scores that populate the QUBO coefficients and evaluates the per-spin Hamiltonian differences that drive the anneal. The annealing temperature is set by a digital plane-count control along with the SRAM pseudo-read random bit flips, giving graded cooling entirely from on-chip digital configuration. The macro is taped-out in a 28 nm process [0.16 mm2, 0.41 mW] and is proposed to be integrated as part of the logic base-die in hybrid high-bandwidth memory (HBM) and high-bandwidth flash (HBF) stacks in a GPU package, where only the selected passage identifiers return to the host. Replacing the language-model selection pass with the near-memory Ising solver yields a macro normalized selection-energy estimate more than six orders of magnitude lower than the LLM-based baseline and reduces the evaluated total energy per query by 67%, while maintaining comparable token-level F1 on the evaluated examples.

## Full Text


<!-- PDF content starts -->

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 1
p-DCIM: A 28nm Probabilistic SRAM DCIM based Ising Machine
for Near-Memory Context Selection in LLM-RAG Pipeline
Madhav Vadlamani*, 1, Rahul Singh*, 2, Yuyao Kong1, Haomei Liu1, Zheng Zhang2, Shimeng Yu†,1
1School of Electrical and Computer Engineering, Georgia Institute of Technology, Atlanta, GA, 30332
2Department of Electrical and Computer Engineering, University of California, Santa Barbara, CA, 93106
*Equal contribution.†Corresponding author: Shimeng Yu (shimeng.yu@ece.gatech.edu).
Abstract—We present a fabricated SRAM DCIM Ising ma-
chine that performs set-level context selection for large language
model (LLM) with retrieval-augmented generation (RAG). In
RAG, the evidence-selection stage is increasingly delegated to re-
peated language-model reranking, which adds substantial energy
to the inference pipeline. We reformulate evidence selection as a
quadratic unconstrained binary optimization (QUBO) problem
and solve it on a dual-mode digital compute-in-memory (DCIM)
macro: the same multiply-and-accumulate datapath computes the
embedding-similarity scores that populate the QUBO coefficients
and evaluates the per-spin Hamiltonian differences that drive
the anneal. The annealing temperature is set by a digital plane-
count control along with the SRAM pseudo-read random bit flips,
giving graded cooling entirely from on-chip digital configuration.
The macro is taped-out in a 28 nm process [0.16mm2,0.41mW]
and is proposed to be integrated as part of the logic base-die
in hybrid high-bandwidth memory (HBM) and high-bandwidth
flash (HBF) stacks in a GPU package, where only the selected
passage identifiers return to the host. Replacing the language-
model selection pass with the near-memory Ising solver yields a
macro-normalized selection-energy estimate more than six orders
of magnitude lower than the LLM-based baseline and reduces
the evaluated total energy per query by67%, while maintaining
comparable token-level F1 on the evaluated examples.
Index Terms—retrieval-augmented generation, near-memory,
ising machine, digital compute-in-memory, large language mod-
els, SRAM
I. INTRODUCTION
LARGE language models (LLMs) have demonstrated
strong capabilities in natural-language understanding and
generation, but their responses remain limited by the informa-
tion encoded in their parameters and by the context supplied
during inference. Retrieval-augmented generation (RAG) ad-
dresses this limitation by grounding generation in information
retrieved from an external knowledge source, enabling access
to dynamically updated or domain-specific content without
modifying the underlying model [1], [2], [3]. A typical RAG
pipeline first retrieves a candidate set of passages from a vector
database, selects or reranks a smaller subset, and then supplies
the resulting context to the LLM for answer generation.
As model sizes, knowledge bases, and retrieved candidate
Manuscript created October, 2020; This work was developed by the IEEE
Publication Technology Department. This work is distributed under the L ATEX
Project Public License (LPPL) ( http://www.latex-project.org/ ) version 1.3. A
copy of the LPPL, version 1.3, is included in the base L ATEX documentation of
all distributions of L ATEX released 2003/12/01 or later. The opinions expressed
here are entirely that of the author. No warranty is expressed or implied. User
assumes all risk.pools continue to increase, however, the intermediate context-
selection stage becomes an increasingly important determinant
of both answer quality and system cost.
Conventional retrieval pipelines commonly select the top-K
passages according to independent relevance scores. Although
computationally simple, this approach does not explicitly
account for interactions among the selected passages. In multi-
hop and evidence-intensive question answering, the desired
context is inherently a set-level object: multiple passages
may provide redundant information, while individually lower-
ranked passages may contain complementary evidence re-
quired to satisfy different information needs of the query [4],
[5]. Recent approaches therefore perform set-wise context
selection using an LLM, allowing the selector to reason jointly
over candidate passages and query requirements [5]. Such
methods can improve evidence construction, but they introduce
another transformer-inference pass into the RAG pipeline. For
large models, this additional stage requires repeated movement
of model weights between high-bandwidth memory (HBM)
and the GPU and consequently incurs substantial latency and
energy even though its output is only a small subset of passage
identifiers.
These characteristics motivate separating semantic process-
ing from combinatorial context selection. Recent work has
formulated requirement-aware RAG evidence selection as a
quadratic unconstrained binary optimization (QUBO) prob-
lem, in which binary variables indicate selected passages and
the objective jointly captures passage relevance, requirement
coverage, support strength, redundancy, complementarity, and
context-size constraints [6]. Unlike independent top-Krank-
ing, the resulting objective explicitly represents pairwise inter-
actions among candidate passages and therefore optimizes the
utility of the selected set as a whole. Once written in QUBO
form, the selection problem can be executed using a broad
class of Ising machines. This creates an opportunity to replace
a computationally expensive LLM-based selection pass with
an Ising machine dedicated to binary energy minimization.
Ising machines address combinatorial optimization prob-
lems by evolving toward low-energy spin configurations,
and prior Ising and p-bit solvers have reported substan-
tial energy and execution-time benefits on classical bench-
marks such as the traveling-salesman, MAX-CUT, and sat-
isfiability problems [7], [8], [9], [10]. SRAM-based digi-
tal compute-in-memory (DCIM) implementations provide a
CMOS-compatible realization of this paradigm by storing
arXiv:2610.37033v1  [cs.AR]  27 Oct 2026

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 2
the coupling coefficients directly in memory and evaluat-
ing the weighted sums required for spin updates in situ,
thereby reducing repeated movement of the QUBO matrix
between memory and arithmetic units [11], [12]. These ar-
chitectures have been demonstrated primarily on conventional
combinatorial-optimization workloads such as the traveling-
salesman problem.
In our recent work, we extended the SRAM DCIM Ising
architecture from problem-specific combinatorial optimization
toward a more general QUBO-solving framework, enabling
arbitrary QUBO coefficient matrices to be mapped onto the in-
memory annealing datapath and demonstrating the architecture
on binary neural network robustness verification [13]. In that
application, adversarial counterexample search was formulated
as a QUBO and executed using the DCIM Ising machine, es-
tablishing the applicability of the architecture to structured AI-
derived optimization problems beyond classical benchmarks.
The annealing stochasticity is generated directly from voltage-
controlled SRAM pseudo-read errors, eliminating the need for
dedicated noise-generation peripherals such as linear-feedback
shift registers (LFSRs) or other external random-number-
generation circuitry. The present work advances this direction
further by targeting a recurring optimization stage within an
LLM inference pipeline and by incorporating not only QUBO
solution, but also hardware-efficient construction of the QUBO
itself. In contrast to our previous BNN-verification study,
which relied on simulation and PPA projections, the present
work is supported by a new fabricated 28-nm SRAM-DCIM
chip, with measured stochastic characteristics, power, and on-
chip QUBO-solving operation.
RAG context selection introduces a different hardware
challenge. Constructing the selection objective itself requires
query-to-passage and passage-to-passage similarity evaluations
over the retrieved candidate embeddings. ForNcandidates,
the pairwise terms can requireO(N2)embedding interactions
before annealing begins. If these embeddings are first retrieved
from storage and transferred to the GPU merely to construct
the QUBO, much of the data-movement cost that an Ising
accelerator is intended to avoid is incurred before optimization
starts. This issue becomes particularly relevant in emerging
near-storage RAG architectures such as HA VEN [14], where
vector retrieval is performed directly within a High-Bandwidth
Flash (HBF) stack integrated in the GPU package. Such
architectures provide a natural location for context-selection
acceleration: the retrieved candidate embeddings are already
available near the storage array, allowing the optimization
objective to be constructed and solved before the candidate
data leave the memory subsystem.
In this work, we present p-DCIM, a fabricated 28-nm
probabilistic SRAM digital compute-in-memory Ising machine
for near-memory context selection in RAG. The proposed
architecture reuses the same SRAM DCIM array and signed
accumulation datapath for two operating modes. In the de-
terministic mode, candidate embeddings are streamed through
the array to evaluate the dot products used to construct the
QUBO coefficients. In the probabilistic mode, the resulting
coefficients are stored in the same array and the datapath
evaluates the local Hamiltonian differences required for se-
Fig. 1. Proposed near-memory RAG context-selection flow. Near-storage
retrieval on the HBF stack produces candidate passages, which are scored
and jointly selected by the proposed p-DCIM accelerator through QUBO
construction and optimization. Only the selected subset is returned to the
GPU for LLM answer generation.
quential Ising updates. Thus, QUBO construction and QUBO
optimization share the same storage and arithmetic resources,
amortizing access to the candidate embeddings rather than
treating coefficient generation and annealing as two indepen-
dent computational stages.
The probabilistic operation exploits measured pseudo-read
errors of the SRAM cells rather than requiring dedicated
stochastic devices or external random-number generation. Re-
ducing the SRAM supply voltageV DDM induces probabilistic
disturbances in the stored coefficient bits, and the measured
0→1and1→0error characteristics provide the stochastic
perturbation used during annealing. Because the measured
voltage dependence contains a narrow transition region that
would be difficult to regulate as an analog temperature control
in a heterogeneous memory package, we operate the array
at a fixed reduced supply and instead control the effec-
tive perturbation strength digitally. Successive magnitude bit-
planes are protected from pseudo-read disturbance as anneal-
ing progresses, while the sign bit remains protected throughout
operation. The resulting plane-count schedule provides a dis-
crete cooling mechanism whose stochastic characteristics are
directly derived from measurements of the fabricated SRAM
macro.
At the system level, we further investigate integration of
the proposed accelerator on the custom logic base die of an
HBF stack in an HBM/HBF-equipped GPU package. Can-
didate retrieval is performed near storage, after which the

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 3
candidate embeddings are consumed locally by p-DCIM for
QUBO construction and context selection. Only the query-side
information required to construct the objective is supplied by
the GPU, and only the selected passage identifiers are returned
for final answer generation. Consequently, theO(N2)pairwise
candidate processing used to construct the set-level objective
remains within the memory subsystem rather than travers-
ing the package interconnect. The fabricated macro occupies
0.16 mm2, operates at500 MHz, and consumes approximately
0.41 mWin probabilistic operation. Small QUBO instances
are evaluated directly on silicon, while larger HotpotQA
instances are evaluated using a measurement-parametrized
behavioral model derived from the measured pseudo-read
statistics and datapath precision. At the evaluated operating
point, context selection requires approximately2.2 msand
0.9µJ, while maintaining answer quality competitive with
LLM-based set-selection baselines.
The main contributions of this work are summarized as
follows:
•We present a fabricated 28-nm dual-mode SRAM-DCIM
accelerator that uses a shared memory array and signed
accumulation datapath to both construct and solve QUBO
objectives for RAG context selection.
•We employ a hardware-controlled annealing mechanism
based on measured SRAM pseudo-read stochasticity, in
which a digitally programmable magnitude-plane expo-
sure schedule provides graded cooling while preserving
the coefficient sign.
•We map requirement-aware, set-level RAG evidence
selection onto the fabricated DCIM architecture and
demonstrate the complete execution flow from candidate-
embedding scoring and QUBO construction through in-
memory Hamiltonian evaluation and passage selection.
•We propose a near-memory integration architecture with
an HBF-based retrieval system, keeping candidate em-
beddings and their pairwise interactions near storage and
returning only the selected passage identifiers to the GPU.
•We characterize the fabricated chip and use a
measurement-parametrized model to evaluate larger RAG
workloads, demonstrating competitive HotpotQA answer
quality while reducing the selection-stage energy by more
than six orders of magnitude relative to an LLM-based
selector and reducing the evaluated end-to-end query
energy by67%.
The remainder of this paper is organized as follows. Section
II reviews set-level context selection in RAG and the QUBO
formulation used in this work. Section III describes the fabri-
cated p-DCIM architecture, including deterministic coefficient
construction, probabilistic Hamiltonian evaluation, and the
proposed digital annealing schedule. Section IV presents the
near-memory system architecture and integration with HBF-
based retrieval. Section V reports the measured chip char-
acteristics and evaluates the proposed context-selection flow
on RAG workloads. Finally, Section VI concludes the paper
and discusses opportunities for scalable near-memory Ising
acceleration.II. QUBO FORMULATION
Top-kretrieval ranks passages independently and may there-
fore fail to select a subset that collectively satisfies the query
requirements. Following [6], we instead optimize over a subset
of the top-Ncandidates returned by HBF-based near-storage
retrieval [14], replacing neural-network-based selection during
inference. The Ising hardware considered in this work operates
directly in the binary domain and therefore requires the context
selection problem to be expressed as a Quadratic Uncon-
strained Binary Optimization (QUBO) problem. A QUBO is
given by
HQUBO =X
iQiiqi+X
i<jQijqiqj,(1)
whereq i∈ {0,1}are binary decision variables andQcontains
the corresponding linear and quadratic coefficients. QUBO
is equivalent to the Ising formulation through the variable
transformations i= 2q i−1, wheres i∈ {−1,+1}denotes
the corresponding spin variable.
To formulate candidate selection for Retrieval-Augmented
Generation (RAG) as a binary optimization problem, each
candidate passagep iis associated with a binary variable
xi∈ {0,1}, wherex i= 1indicates selection. Each passage
is also assigned a query-relevance scorer i. The original
query is decomposed by an LLM into a set of requirements,
or subqueries, indexed byj. The parameterw jdenotes the
importance of requirementjwith respect to the original query,
whilea ji∈ {0,1}indicates whether candidate passagep i
is relevant to requirementj. A binary variablec j∈ {0,1}
indicates whether requirementjis covered by the selected
passages. Beyond individual relevance, candidate passages
may also interact with one another. Two passages may provide
overlapping evidence for the same requirements, making their
joint selection redundant, whereas another pair may support
different but jointly useful requirements. Since the number
of selected passages is limited by the context budget, the
selection process should avoid redundant information while
favoring complementary evidence. Accordingly, we define
dik∈[0,1]as the pairwise redundancy between passages
piandp k, accounting for overlap in their information and
supported requirements, andb ik∈[0,1]as their pairwise
complementarity, capturing the additional utility of selecting
passages that support different but jointly useful requirements.
Coverage is enforced by
cj≤X
iajixi,(2)
such thatc jcan be set to one only if at least one selected
passage is relevant to requirementj. The context budget is
X
ixi≤K,(3)
whereKdenotes the maximum number of passages included
in the context. The resulting constrained binary optimization

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 4
problem is
max
x,cX
irixi+X
jwjcj−λX
ixi
−X
i<kdikxixk+X
i<kbikxixk,
s.t. c j≤X
iajixi,∀j,
X
ixi≤K,(4)
The objective balances query relevance and requirement cov-
erage against context size and redundancy, while rewarding
complementary evidence. The parameterλcontrols the cost
of including additional passages within the budgetK. The
scoresd ikandb ikare normalized to the same range to avoid
additional weighting parameters.
The relevance, coverage, and selection-cost terms in (4)
contribute linear coefficients to the QUBO objective, while
the redundancy and complementarity terms provide pairwise
quadratic coefficients. To remove the remaining constraints,
the coverage and context-budget inequalities are converted
into equalities using binary-encoded slack variables, and their
residuals are added as squared penalties. After negating the
original maximization objective and collecting the objective
and penalty terms, the context-selection problem is expressed
in the standard QUBO form
min
q∈{0,1}nq⊤Qq,(5)
whereqcontains the passage-selection, coverage, and aux-
iliary slack variables, whileQcontains the resulting linear
and pairwise coefficients. The resulting QUBO is solved by
the DCIM datapath described in Section III, and the passages
corresponding to the optimized selection variables are passed
unchanged to the answer generator.
III. FABRICATED P-DCIM ARCHITECTURE AND
DUAL-MODEOPERATION
The proposed p-DCIM accelerator is designed to execute
the two computational phases required by the context-selection
flow of Section II: first, construction of the QUBO coeffi-
cients from the retrieved candidate embeddings, and second,
iterative solution of the resulting QUBO through stochastic
annealing. Rather than implementing these phases using sepa-
rate similarity-computation and optimization engines, p-DCIM
reuses a single SRAM-based DCIM array and its associated
signed accumulation datapath across both modes. The de-
terministic mode computes the embedding-derived quantities
used to populate the QUBO matrix, whereas the probabilistic
mode stores the resulting quantized coefficients and evaluates
the local energy differences required for sequential QUBO
updates. This dual-mode reuse is the central architectural
feature of the proposed design.
A. Fabricated SRAM-DCIM Macro
Fig. 2 shows the fabricated p-DCIM macro and its mea-
surement setup. The chip was fabricated in a 28-nm CMOSprocess and contains a64×64logical DCIM array supporting
signed 8-bit coefficients. The eight coefficient bit planes are
distributed across 512 physical wordlines, resulting in a total
SRAM capacity of 32 Kb. The macro occupies0.16 mm2
and operates at a clock frequency of 500 MHz. The fabricated
chip is characterized through an FPGA/AXI-based measure-
ment platform, which provides coefficient loading, spin-state
control, mode configuration, and readout of the resulting states.
At the cell level, the compute-enabled memory element is
a 12T SRAM cell consisting of a conventional 6T storage
latch, a 4T NOR compute gate, and a 2T transmission-gate
cell multiplexer, as illustrated in Fig. 2(d). The SRAM latch
stores one coefficient bit, while the compute gate conditionally
enables its contribution according to the applied input. The
cell multiplexer selects the active compute path and permits
the bit-sliced coefficient planes to be accessed independently.
Across a selected row or column, the cell-level outputs are
reduced by a signed digital adder tree to form the weighted
accumulation required by both operating modes.
The coefficient precision is represented through one pro-
tected sign plane and multiple magnitude planes. Conceptually,
a signed coefficient can be written as
eQij=sijB−2X
b=02bm(b)
ij, s ij∈ {−1,+1},(6)
wherem(b)
ij∈ {0,1}denotes the stored magnitude bit at
planebandB= 8in the fabricated macro. The bit-
sliced organization allows the same physical memory to retain
the optimization coefficients and participate directly in their
arithmetic evaluation.
As shown in Fig. 2(e), the SRAM array dominates the
measured macro area, while the remaining area is occupied
by the main controller, read/write circuitry, wordline drivers,
multiplexing logic, and the signed adder tree. The measured
active power is similar in the two operating modes:0.406 mW
in deterministic mode and0.413 mWin probabilistic mode,
as shown in Fig. 2(f). The small difference between the two
modes follows from their use of the same physical array and
accumulation datapath; probabilistic operation additionally in-
vokes the pseudo-read disturbance mechanism described in
Section III-D.
The use of SRAM is particularly suitable for the proposed
execution flow. QUBO construction produces a new coefficient
matrix for each query, requiring the array to be reprogrammed
before annealing. During probabilistic operation, the nominal
coefficients are also periodically restored after noise injection.
Consequently, the architecture performs frequent writes rather
than treating the weight matrix as a stationary parameter set.
SRAM provides low write latency and effectively unrestricted
write endurance for this operating pattern, while simultane-
ously supporting the controlled read-disturb mechanism used
to generate annealing stochasticity.
B. Deterministic Mode: QUBO Construction
In deterministic mode, the p-DCIM datapath is used to
construct the QUBO coefficients required for context selection.
As illustrated in Fig. 3, candidate embeddings are streamed

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 5
Fig. 2. Fabricated 28-nm p-DCIM SRAM Ising accelerator. (a) FPGA-based chip measurement setup. (b) Die micrograph of the fabricated macro. (c) SRAM-
DCIM array organization for storing signed, bit-sliced coefficients and performing in-memory accumulation. (d) 12T compute cell consisting of a 6T SRAM
latch, 4T NOR compute gate, and 2T transmission-gate cell multiplexer. (e) Measured area breakdown of the0.16 mm2macro. (f) Measured active-power
breakdown in deterministic and probabilistic operating modes, with total powers of0.406 mWand0.413 mW, respectively.
through the SRAM-DCIM array with the corresponding query
or requirement embedding, and the signed adder tree evaluates
the dot products used to construct the QUBO coefficients. In
this mode, the complete signed MAC output is read from the
datapath rather than only its sign.
Lete q,ei, ande ujdenote the normalized embeddings of the
query, candidate passagep i, and requirementj, respectively.
The signed MAC datapath evaluates the dot products required
to construct the relevance and requirement-support terms as
ri=e⊤
qei, w j= max 
0,e⊤
qeuj
,
aji=Ih
e⊤
ujei≥τai
,(7)whereτ ais the requirement-support threshold. Pairwise pas-
sage similarity is obtained asσ ik=e⊤
iekand combined with
requirement overlap to form the redundancy coefficient
dik= max(0, σ ik)P
jajiajkP
jaji+P
jajk−P
jajiajk.(8)
The complementarity coefficientb ikis computed from
the requirement-support information following the definition
in [6], rewarding passage pairs that contribute evidence to-
ward different but complementary requirements. Thus, the
same MAC datapath provides the similarity quantities used
to construct both the linear and pairwise QUBO coefficients.

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 6
Fig. 3. Deterministic QUBO-construction mode of p-DCIM. Candidate passages are tokenized and encoded into embeddings, which are streamed through the
DCIM MAC datapath together with the corresponding query- or requirement-dependent embedding. The signed accumulation computes the query–candidate,
requirement–candidate, and candidate–candidate similarity quantities used to form the data-dependent linear and pairwise QUBO coefficients. Deterministic
constraint- and penalty-derived terms are combined with these coefficients in the host control path before quantization and write-back into the SRAM array.
If the embedding dimension exceeds the physical array width, partial MAC results are accumulated across successive passes.
Fig. 4. Measured evolution of the stored QUBO coefficients during probabilistic operation for a representative 8-variable instance. Pseudo-read operations at
reducedV DDM probabilistically perturb the exposed magnitude bits, providing the stochasticity required for annealing. The nominal coefficients are periodically
restored from the on-chip buffer, as illustrated by the refresh at iteration 100, while the optimization state is retained. As the digital plane-count schedule
progresses, fewer magnitude planes are exposed to pseudo-read disturbance and the injected coefficient noise decreases toward the end of the anneal.
Fig. 5. Measured SRAM pseudo-read stochasticity and digital annealing control. (a) Measured pseudo-read error probability versusV DDM for cells storing
“0” and “1”, showing the asymmetric disturbance probabilities and the approximately60 mVtransition from the high-error regime to near-deterministic
operation. The dashed line marks the selected0.6 Voperating point. (b) Digital cooling schedule in which the sign plane remains protected and the number
of magnitude planes exposed to pseudo-read disturbance is progressively reduced fromk= 7tok= 1. (c) Measured RMS coefficient perturbation at
VDDM = 0.6 Vversus exposed-plane count, demonstrating graded reduction of the effective annealing noise as additional magnitude planes are protected.
The remaining constraint-derived coefficients associated
with the slack-variable penalties in Section II are generated
digitally from the known constraint parameters and combined
with the embedding-derived coefficients before quantization
and write-back. Thus, p-DCIM performs the data-dependent
similarity computations that dominate QUBO construction,
while the deterministic algebraic penalty terms are incorpo-
rated through the accompanying host control logic.
If the embedding dimension exceeds the physical array
width, the dot product is evaluated over multiple MAC passes
and the partial results are accumulated. The resulting QUBO
coefficients are buffered, quantized, and written back into thesame SRAM array for probabilistic operation.
Deterministic operation is performed atV DDM = 0.9 V,
where the SRAM operates reliably and pseudo-read-induced
disturbances are suppressed. After QUBO construction, the
supply voltage is reduced for probabilistic operation, reusing
the same SRAM array and signed accumulation datapath for
annealing. This eliminates the need for a separate QUBO-
construction engine.
C. Probabilistic Mode: In-Memory QUBO Annealing
After the QUBO coefficients are constructed and written
into the SRAM array, p-DCIM transitions to probabilistic

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 7
mode to perform the annealing search. The same DCIM
array and signed adder tree used for deterministic coefficient
construction are now reused to evaluate the local energy
change associated with updating each binary variable. For a
QUBO objective
E(x) =xTQx, x i∈ {0,1},(9)
the update decision for variablex iis determined by the
corresponding local energy difference∆E i. The current binary
state is applied to the array, the stored QUBO coefficients
participate in the in-memory accumulation, and the sign of
the resulting∆E idetermines whether the visited variable is
flipped. Unlike deterministic mode, where the complete signed
MAC result is used to form a QUBO coefficient, probabilistic
mode requires only the sign of the local energy difference to
drive the state update.
For general QUBOs with nonzero diagonal coefficients, we
employ the pinned-one embedding developed in our prior
work [13], which maps each diagonal term to a coupling with
an auxiliary variable clamped to one, allowing the same DCIM
MAC datapath to evaluate the single-variable energy change
without a separate diagonal-add path.
Stochasticity is introduced directly into the stored QUBO
coefficients through SRAM pseudo-read disturbances at re-
ducedV DDM , rather than through a separate random-number-
generation circuit. As illustrated in Fig. 4, the programmed
coefficient matrix evolves during annealing as pseudo-read
events probabilistically perturb the exposed magnitude bits.
The nominal coefficients are periodically restored from an
on-chip buffer to prevent uncontrolled accumulation of read-
induced errors, while the binary optimization state is retained
across the refresh. The example in Fig. 4 shows this behavior
for a representative 8-variable instance: the coefficients be-
come progressively perturbed during the early iterations, are
restored at the programmed refresh point, and subsequently
experience decreasing perturbation as the digital plane-count
schedule cools the anneal.
Variables are updated sequentially, with each accepted up-
date committed before the next local energy difference is eval-
uated. The pseudo-read perturbations therefore modify the ef-
fective local fields encountered during the scan and provide the
stochastic exploration required for annealing without dedicated
peripherals such as linear-feedback shift registers (LFSRs).
As the number of magnitude planes exposed to pseudo-read
noise is progressively reduced, the coefficient perturbations
decrease and the system approaches increasingly deterministic
operation. The measured pseudo-read characteristics and the
corresponding digital cooling mechanism are described in
detail in Section III-D.
Thus, the transition from deterministic to probabilistic op-
eration requires no separate optimization datapath: the co-
efficients constructed using the full signed MAC output are
written back into the same array, after which the same array
and adder tree are reused for repeated local-energy evaluation
during annealing.D. SRAM Pseudo-Read Stochasticity and Digital Cooling
The stochasticity required for annealing is generated from
the intrinsic pseudo-read disturbance of the SRAM cells at
reduced supply voltage. Fig. 5(a) shows the measured pseudo-
read error probability as a function ofV DDM for cells initially
storing “0” and “1”. The disturbance is asymmetric between
the two stored states; at the selectedV DDM = 0.6 Voperat-
ing point, the measured error probabilities are approximately
63%and38%, respectively. AsV DDM is increased, the error
probability falls rapidly toward zero over a narrow transition
region of approximately60 mV.
Although this voltage dependence could in principle be used
directly as an annealing schedule, the narrow transition makes
fine-grained noise control throughV DDM difficult, particularly
in the proposed HBM/HBF package environment. We there-
fore operate the SRAM at a fixedV DDM = 0.6 V, where the
pseudo-read disturbance is relatively stable, and control the
effective annealing strength digitally through the number of
coefficient magnitude planes exposed to pseudo-read noise.
Fig. 5(b) illustrates the resulting digital cooling schedule.
The sign plane is protected throughout annealing so that
pseudo-read errors perturb only the coefficient magnitude and
do not reverse the polarity of the corresponding interaction.
The number of exposed magnitude planes is then progressively
reduced as annealing proceeds. In the example schedule, the
exposed-plane count is stepped throughk= 7,5,3,1,
with each stage held for one quarter of the total annealing
duration. Early iterations therefore experience stronger coeffi-
cient perturbations and greater stochastic exploration, whereas
later iterations expose fewer magnitude planes and approach
increasingly deterministic operation.
Because the coefficient representation is binary weighted,
disturbances in the more significant planes contribute more
strongly to the numerical coefficient error. Fig. 5(c) shows the
measured RMS coefficient perturbation atV DDM = 0.6 Vas
a function of the number of exposed magnitude planes. The
perturbation decreases monotonically as planes are protected,
providing a graded effective-temperature schedule without
requiring fine analog regulation of the SRAM supply. In
particular, removing higher-significance planes produces a
substantial reduction in the coefficient disturbance because
adjacent bit planes differ by a factor of two in numerical
weight.
The proposed scheme therefore separates the source of
stochasticity from its annealing control: device-level pseudo-
read errors provide the randomness, while simple digital plane-
enable signals set its effective strength. This avoids both a
finely controlled analogV DDM sweep and dedicated random-
noise-generation circuitry, while retaining a programmable
cooling trajectory for the in-memory annealing process.
E. End-to-End Dual-Mode Control Flow
Fig. 6(b) summarizes the complete p-DCIM execution flow.
The accelerator first operates deterministically atV DDM = 0.9
V to compute, quantize, and program the QUBO coefficients.
The supply is then reduced to0.6V for probabilistic oper-
ation, where pseudo-read disturbance, in-memory local-field

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 8
Fig. 6. Dual-mode operation and control flow of the proposed p-DCIM
accelerator. (a) Shared SRAM-DCIM array and signed adder tree used in both
operating modes. In deterministic mode, the full signed MAC output is used
to construct and populate the QUBO coefficients, whereas in probabilistic
mode only the sign of the local energy difference is required to determine
the spin update. (b) End-to-end control flow. The accelerator first operates
in deterministic mode atV DDM = 0.9 Vto load embeddings, compute
embedding dot products, and write the resulting QUBO coefficients into the
SRAM array. The supply is then reduced to0.6 Vfor probabilistic operation,
where pseudo-read disturbance injects stochasticity into the stored coefficients,
followed by in-memory Hamiltonian evaluation and sequential spin updates.
The nominal coefficients are periodically refreshed during annealing until the
prescribed iteration count is reached.evaluation, sequential spin updates, and periodic coefficient
refresh are repeated according to the digital cooling schedule
until the prescribed iteration count is reached. The final
passage-selection state is then returned for downstream answer
generation. Thus, embedding scoring, QUBO solution, and
annealing stochasticity are supported by the same SRAM-
DCIM substrate.
IV. NEAR-MEMORYRAG SYSTEMARCHITECTURE
The circuit architecture described in Section III enables
QUBO construction and optimization to be performed using
the same SRAM-DCIM datapath. To exploit this capability
at the system level, the accelerator should be placed close
to the retrieved candidate embeddings rather than treated as
a discrete coprocessor attached to the GPU. In this work,
we therefore consider a near-memory integration in which
the fabricated p-DCIM architecture is placed on the custom
logic base die of a High-Bandwidth Flash (HBF) stack within
an HBM/HBF-equipped GPU package. Fig. 7 contrasts this
proposed organization with a conventional RAG execution
flow.
A. Data Movement in Conventional RAG Systems
In a conventional RAG system, the vector database resides
outside the GPU package, for example in host memory or SSD
storage. For each query, candidate passages or their embed-
dings are retrieved from the external database and transferred
across the host interface to the GPU, as illustrated in Fig. 7(a).
The GPU then performs the subsequent context-selection or
reranking operation before invoking the same LLM again for
answer generation.
This organization introduces two sources of data movement.
First, the retrieved candidate representations must cross the
package boundary before the selection decision can be made.
Second, when context selection is implemented using an LLM-
based set selector, the model weights must be streamed from
HBM to the GPU for an additional transformer-inference pass.
The result of this relatively expensive computation is neverthe-
less only a small set of passage identifiers or selected passages
that are subsequently supplied to the answer-generation stage.
The cost becomes more pronounced for set-level context
selection. Unlike independent relevance ranking, the QUBO
formulation considered in this work includes pairwise interac-
tions among candidate passages. For a candidate pool of size
N, constructing these terms can requireO(N2)candidate-
pair similarity evaluations. Performing this processing after
moving the retrieved candidates to the GPU would therefore
incur substantial data movement before the Ising optimization
itself begins. Consequently, merely attaching an Ising solver to
a conventional GPU pipeline does not eliminate the communi-
cation associated with constructing the optimization instance.
B. Proposed HBF-Integrated p-DCIM Architecture
The proposed system places p-DCIM adjacent to the vector
database rather than treating it as a discrete accelerator at-
tached to the GPU. As illustrated in Fig. 7(b), we consider

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 9
Fig. 7. Conventional and proposed hardware organizations for RAG context selection. (a) Conventional RAG execution, where candidate embeddings retrieved
from an off-package vector database are transferred to the GPU and context selection is performed using GPU computation and HBM-resident model weights.
(b) Proposed near-memory organization in which the vector database and near-storage retriever reside in an HBF stack and the p-DCIM accelerator is placed
on the custom HBF base die. Retrieved candidate embeddings remain within the HBF subsystem while p-DCIM constructs and solves the context-selection
QUBO; only the selected passage identifiers are returned to the GPU for final LLM answer generation. The HBM stack continues to store the LLM weights
and is not accessed by p-DCIM during context selection.
an HBF-based near-storage retrieval architecture following
HA VEN [14], in which the vector database is stored in the
HBF flash dies and candidate retrieval is performed using
logic integrated on the custom HBF base die. The p-DCIM
accelerator is placed on the same base die and consumes
the retrieved candidate embeddings locally. Meanwhile, the
HBM stack continues to store the LLM weights, and the GPU
performs the semantic operations that require full transformer
inference, including query-dependent requirement generation
and final answer generation.
This placement is important because the context-selection
QUBO contains both query-to-candidate and candidate-to-
candidate terms. For a candidate pool of sizeN, the pair-
wise component can requireO(N2)similarity evaluations.
If the retrieved candidate embeddings were first transferred
to the GPU, this pairwise processing would incur substantial
package-level data movement before optimization even begins.
In the proposed architecture, the retrieved embeddings remain
within the HBF subsystem and are consumed directly by p-
DCIM. The deterministic mode described in Section III-B
computes the similarity quantities required to form the QUBO
coefficients, after which the coefficients are quantized and
written back into the same SRAM-DCIM array.
The resulting QUBO is then solved locally using the prob-
abilistic mode of Section III-C. Because QUBO construction
and optimization reuse the same physical array and adder-tree
datapath, neither the candidate embeddings nor the complete
QUBO matrix need to be transferred to a separate accelerator
during context selection. Only the final passage-selection result
is returned to the GPU for downstream generation.
The end-to-end query flow therefore proceeds as follows.
First, the GPU generates the query-dependent information
required for retrieval and context selection and sends it to
the HBF subsystem. Second, the near-storage retrieval engine
searches the HBF-resident vector database and produces the
candidate pool locally. Third, p-DCIM constructs the QUBOfrom the retrieved candidate embeddings and solves it using
the dual-mode execution described in Section III. Fourth, the
selected passage identifiers are returned to the GPU. These
identifiers are subsequently used to fetch only the selected
Kpassage contents required for generation; the remaining
candidate passages and embeddings remain within the HBF
subsystem. Finally, the GPU assembles the selected context
and performs answer generation using the LLM weights stored
in HBM.
This organization keeps the high-dimensional candidate
representations and their pairwise processing close to stor-
age while reserving the GPU for full LLM inference. The
O(N2)candidate-interaction workload remains on the HBF
base die, and only a compact selection result crosses back to
the GPU after optimization. Fig. 7(b) represents a proposed
package-level integration of the fabricated p-DCIM macro;
the HBM/HBF-integrated system itself is not fabricated in
this work. The resulting latency and energy implications are
evaluated in Section V.
V. EXPERIMENTALRESULTS ANDSYSTEMEVALUATION
We evaluate the proposed system at both the circuit and
application levels. Small QUBO instances that fit within the
fabricated64×64p-DCIM macro are executed directly
on chip, while larger RAG context-selection instances are
evaluated using a measurement-parametrized behavioral model
derived from the measured SRAM pseudo-read statistics and
the finite-precision datapath of the fabricated chip. At the
application level, we evaluate whether the selected contexts
preserve downstream answer quality on HotpotQA, and at the
system level we compare the latency and energy of p-DCIM
context selection against an LLM-based set-selection baseline
executed on a GPU.

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 10
Fig. 8. Representative end-to-end HotpotQA context-selection example. The query and information requirements are generated on the GPU, and a pre-generated
candidate pool is supplied to the p-DCIM selection stage. The fabricated chip performs the context-selection QUBO on-chip, retaining passages that jointly
provide the required evidence (green) while rejecting an irrelevant candidate (red). The selected context is returned to the GPU, where the LLM generates
the final answer.
Fig. 9. Application and system-level evaluation of the proposed p-DCIM context-selection flow. (a) QUBO energy versus annealing iteration for a representative
HotpotQA instance, showing the mean and±2σvariation across repeated runs of the measurement-parametrized model derived from measured chip
characteristics. (b) Downstream answer quality compared with SetR and SetRO [5] using Exact Match, Gold-Contains, and token-level F1 metrics. (c)
Energy per query by RAG pipeline stage. Requirement generation and answer generation remain on the GPU in both systems, while the LLM-based selection
pass is replaced by p-DCIM, reducing selection energy to approximately0.9µJand the evaluated total query energy by67%.
A. Experimental Methodology and End-to-End Demonstration
The fabricated p-DCIM macro is characterized using the
FPGA/AXI measurement setup described in Section III.
QUBO instances that fit within the physical array are executed
directly on chip, while larger RAG context-selection instances
are evaluated using a measurement-parametrized behavioral
model incorporating the measured SRAM pseudo-read statis-
tics, signed 8-bit coefficient quantization, sequential spin up-
dates, periodic coefficient refresh, and the digital plane-count
cooling schedule of Section III-D. This enables application-
scale evaluation while retaining the measured stochastic and
finite-precision characteristics of the fabricated hardware.
For the RAG evaluation, the query, information require-
ments, and candidate passages are generated before the p-
DCIM selection stage; requirement generation and final an-
swer generation remain on the GPU. Fig. 8 illustrates a
representative HotpotQA example. The fabricated chip selects
complementary passages required to answer the query while
rejecting an irrelevant candidate, after which the selected
context is returned to the GPU for answer generation. This
example demonstrates the intended role of p-DCIM as the
intermediate combinatorial context selector in an otherwise
conventional RAG pipeline.B. QUBO Solution and RAG Answer Quality
The fabricated macro executes the complete annealing
loop for QUBO instances that fit within the physical ar-
ray; across all ten measured instances, the minimum-energy
state reached by p-DCIM matched the brute-force optimum.
Larger HotpotQA-derived QUBOs are evaluated using the
measurement-parametrized model. As shown in Fig. 9(a), the
mean QUBO energy decreases during annealing and stabilizes
as the digital plane-count schedule reduces the injected coef-
ficient perturbation.
The resulting passage selections are evaluated through
downstream answer generation. Fig. 9(b) compares the pro-
posed method with SetR and SetRO [5]. On the evalu-
ated HotpotQA examples, p-DCIM achieves an Exact Match
score of0.70, Gold-Contains of0.90, and token-level F1 of
0.8524, compared with0.70,0.80, and0.7857, respectively,
for both LLM-based selectors. These results demonstrate that
the QUBO-based selector can replace the LLM set-selection
pass while preserving competitive downstream answer quality.
The present evaluation is intended as a hardware-system
demonstration; a larger-scale evaluation of the underlying
QUBO selection formulation is reported separately in [6].

JOURNAL OF L ATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 11
C. Selection Latency and Energy
For the evaluated workload, p-DCIM completes context
selection in approximately2.2 ms. Using the measured
probabilistic-mode power of0.413 mW, this corresponds to
approximately0.9µJper selection. The resulting0.9µJ
should therefore be interpreted as a macro-normalized estimate
based on the measured power of the fabricated array, rather
than a full-array power projection. In comparison, the SetR
selection stage requires an additional LLM inference pass on
an NVIDIA RTX A6000 and consumes approximately358 J
per query. As shown in Fig. 9(c), replacing this LLM-based
selection pass with p-DCIM reduces the selection-stage energy
by more than six orders of magnitude.
Requirement generation and final answer generation remain
unchanged and are executed on the GPU in both systems.
Under the evaluated workload, replacing only the selection
stage reduces the total query energy from approximately
536 Jto175 J, corresponding to a67%reduction. The
p-DCIM power and stochastic characteristics are measured
from the fabricated chip, while HotpotQA-scale selection and
the proposed HBF-integrated system are evaluated using the
measurement-parametrized model described above.
VI. CONCLUSION
This work presented p-DCIM, a fabricated 28-nm proba-
bilistic SRAM digital compute-in-memory Ising accelerator
for near-memory context selection in RAG. The proposed
dual-mode architecture reuses the same SRAM array and
signed accumulation datapath to construct and solve the
context-selection QUBO: deterministic operation evaluates
embedding-derived similarity terms, while probabilistic op-
eration performs in-memory local-energy evaluation for an-
nealing. SRAM pseudo-read disturbances provide the required
stochasticity without dedicated random-noise-generation pe-
ripherals, while a digital magnitude-plane schedule controls
the effective annealing strength. Measurements of the fab-
ricated macro demonstrate the pseudo-read characteristics,
dual-mode operation, and on-chip solution of small QUBO
instances.
At the system level, we proposed placing p-DCIM on the
logic base die of an HBF-based near-storage retrieval system,
allowing retrieved candidate embeddings to remain near stor-
age during QUBO construction and selection. HotpotQA-scale
instances are evaluated using a measurement-parametrized
model derived from the measured chip characteristics. The
proposed flow preserves competitive downstream answer qual-
ity while reducing the evaluated selection-stage energy from an
LLM-based GPU selection pass to approximately0.9µJand
reducing total query energy by67%. These results demonstrate
how a general QUBO-capable DCIM substrate can extend
Ising acceleration from optimization alone to application-
dependent construction and solution of practical AI workloads.
ACKNOWLEDGMENTS
AI Chatbot tools were used to assist with grammar and fig-
ure polishing; all scientific content and analysis were created
solely by the authors. This work was supported in part by NSF
Grant 2311295.REFERENCES
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel,
and D. Kiela, “Retrieval-augmented generation for knowledge-
intensive nlp tasks,” inAdvances in Neural Information Processing
Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and
H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 9459–
9474. [Online]. Available: https://proceedings.neurips.cc/paper files/
paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[2] S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, and M. Abdelrazek,
“Seven failure points when engineering a retrieval augmented generation
system,” inProceedings of the IEEE/ACM 3rd International Conference
on AI Engineering - Software Engineering for AI, ser. CAIN ’24.
New York, NY , USA: Association for Computing Machinery, 2024, p.
194–199. [Online]. Available: https://doi.org/10.1145/3644815.3644945
[3] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, Q. Guo,
M. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,”ArXiv, vol. abs/2312.10997, 2023. [Online].
Available: https://api.semanticscholar.org/CorpusID:266359151
[4] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen, R. Salakhutdinov,
and C. D. Manning, “HotpotQA: A dataset for diverse, explainable
multi-hop question answering,” inProceedings of the 2018 Conference
on Empirical Methods in Natural Language Processing, E. Riloff,
D. Chiang, J. Hockenmaier, and J. Tsujii, Eds. Brussels, Belgium:
Association for Computational Linguistics, Oct.-Nov. 2018, pp.
2369–2380. [Online]. Available: https://aclanthology.org/D18-1259/
[5] D. Lee, Y . Jo, H. Park, and M. Lee, “Shifting from ranking to set
selection for retrieval augmented generation,” inProceedings of the
63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and
M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational
Linguistics, Jul. 2025, pp. 17 606–17 619. [Online]. Available: https:
//aclanthology.org/2025.acl-long.861/
[6] R. Singh and M. Vadlamani, “Qubo-optimized evidence selection for
retrieval-augmented question answering with unconventional solvers,”
2026. [Online]. Available: https://arxiv.org/abs/2607.12334
[7] N. A. Aadit, A. Grimaldi, M. Carpentieri, L. Theogarajan, J. M. Martinis,
G. Finocchio, and K. Y . Camsari, “Massively parallel probabilistic
computing with sparse ising machines,”Nature Electronics, vol. 5, no. 7,
pp. 460–468, 2022.
[8] K. Yamamoto, K. Ando, N. Mertig, T. Takemoto, M. Yamaoka, H. Ter-
amoto, A. Sakai, S. Takamaeda-Yamazaki, and M. Motomura, “7.3
statica: A 512-spin 0.25m-weight full-digital annealing processor with
a near-memory all-spin-updates-at-once architecture for combinatorial
optimization with complete spin-spin interactions,” in2020 IEEE Inter-
national Solid-State Circuits Conference - (ISSCC), 2020, pp. 138–140.
[9] M. Aramon, G. Rosenberg, E. Valiante, T. Miyazawa, H. Tamura,
and H. G. Katzgraber, “Physics-inspired optimization for quadratic
unconstrained problems using a digital annealer,”Frontiers in
Physics, vol. V olume 7 - 2019, 2019. [Online]. Available: https:
//www.frontiersin.org/journals/physics/articles/10.3389/fphy.2019.00048
[10] N. A. Aadit, A. Grimaldi, M. Carpentieri, L. Theogarajan, G. Finocchio,
and K. Y . Camsari, “Computing with invertible logic: Combinatorial op-
timization with probabilistic bits,” in2021 IEEE International Electron
Devices Meeting (IEDM), 2021, pp. 40.3.1–40.3.4.
[11] A. Lu, J. Lee, Y .-C. Luo, H. Li, I. Young, and S. Yu, “Digital cim
with noisy sram bit: A compact clustered annealer for large-scale
combinatorial optimization,” inProceedings of the 61st ACM/IEEE
Design Automation Conference, ser. DAC ’24. New York, NY , USA:
Association for Computing Machinery, 2024. [Online]. Available:
https://doi.org/10.1145/3649329.3655673
[12] Y . Kong, A. Lu, H. Liu, V . Garg, R. Wahabzada, H. Li, I. Young,
and S. Yu, “A compact digital compute-in-memory Ising annealer with
probabilistic SRAM bit in 28nm for travelling salesman problem,”npj
Unconv. Comput., 2026.
[13] M. Vadlamani, R. Singh, Y . Kong, Z. Zhang, and S. Yu,
“Scalable digital compute-in-memory ising machines for robustness
verification of binary neural networks,” 2026. [Online]. Available:
https://arxiv.org/abs/2603.05677
[14] P.-K. Hsu, W. Xu, Q. Liu, T. Rosing, and S. Yu, “Haven: High-
bandwidth flash augmented vector engine for large-scale approximate
nearest-neighbor search acceleration,” 2026. [Online]. Available:
https://arxiv.org/abs/2603.01175