# xTBloom: GPU Batching for High-Throughput GFN-xTB Calculations

**Authors**: Zitian Lu, Shengrui Xu, Tong Zhu, Qiangqiang Gu, Jinzhe Zeng

**Published**: 2026-11-18 17:45:11

**PDF URL**: [https://arxiv.org/pdf/2611.24807v1](https://arxiv.org/pdf/2611.24807v1)

## Abstract
Conformational searches and multiwindow free-energy sampling require large numbers of GFN-xTB energy and force evaluations across molecular conformations and sampling windows. Processing each system separately on a GPU makes it difficult to amortize memory management, host-device data transfer, and kernel-launch overhead. Here, we present xTBloom, a GPU batching library for GFN-xTB energy and force evaluations. xTBloom sorts evaluation queues by atomic orbital (AO) count before forming batches, accommodates size differences through a native ragged layout, maintains independent self-consistent charge (SCC) and failure states for each system, and selects execution paths according to data residency. We focus the quantitative evaluation on GFN2-xTB through cross-program comparisons on QM9 and OMol25, CPU-CUDA comparisons, finite-difference force checks, and controlled failure tests. On a near-size OMol25 workload, GPU execution achieved a 2.9-fold speedup over xTBloom's single-core CPU backend. A companion LAMMPS plugin integrates cross-window batching with periodic quantum mechanical/molecular mechanical (QM/MM) simulation. In a short synchronized test with 48 windows on one GPU, the complete coupled workflow achieved an aggregate speedup of approximately 14 over sequential execution of the same windows. These results demonstrate the throughput benefits of batched GFN-xTB evaluation for repeated electronic-structure calculations and multiwindow sampling.

## Full Text


<!-- PDF content starts -->

xTBloom: GPU Batching for
High-Throughput GFN-xTB Calculations
Zitian Lu,†,‡Shengrui Xu,†,¶,‡Tong Zhu,∗,§,¶Qiangqiang Gu,∗,†,‡,∥and Jinzhe
Zeng∗,†,⊥,‡,∥
†School of Artificial Intelligence and Data Science, University of Science and Technology of
China, Hefei 230026, P.R. China
‡Suzhou Institute for Advanced Research, University of Science and Technology of China,
Suzhou 215123, P.R. China
¶SII-ECNU Joint Autonomous Chemistry Laboratory, Shanghai Innovation Institute,
Shanghai 200003, P.R. China
§Shanghai Engineering Research Center of Molecular Therapeutics and New Drug
Development, School of Chemistry and Molecular Engineering, East China Normal
University, Shanghai 200062, P.R. China
∥Suzhou Big Data & AI Research and Engineering Center, Suzhou 215123, P.R. China
⊥State Key Laboratory of Precision and Intelligent Chemistry, University of Science and
Technology of China, Hefei 230026, P.R. China
E-mail: tzhu@lps.ecnu.edu.cn; guqq@ustc.edu.cn; jinzhe.zeng@ustc.edu.cn
Abstract
Conformational searches and multiwindow free-energy sampling require large num-
bers of GFN-xTB energy and force evaluations across molecular conformations and
sampling windows. Processing each system separately on a GPU makes it difficult
to amortize memory management, host–device data transfer, and kernel-launch over-
head. Here, we present xTBloom, a GPU batching library for GFN-xTB energy and
force evaluations. xTBloom sorts evaluation queues by atomic orbital (AO) count be-
fore forming batches, accommodates size differences through a native ragged layout,
maintains independent self-consistent charge (SCC) and failure states for each system,
and selects execution paths according to data residency. We focus the quantitative
evaluation on GFN2-xTB through cross-program comparisons on QM9 and OMol25,
CPU–CUDA comparisons, finite-difference force checks, and controlled failure tests.
On a near-size OMol25 workload, GPU execution achieved a 2.9-fold speedup over
xTBloom’s single-core CPU backend. A companion LAMMPS plugin integrates cross-
window batching with periodic quantum mechanical/molecular mechanical (QM/MM)
simulation. In a short synchronized test with 48 windows on one GPU, the complete
coupled workflow achieved an aggregate speedup of approximately 14 over sequential
execution of the same windows. These results demonstrate the throughput benefits of
batched GFN-xTB evaluation for repeated electronic-structure calculations and multi-
window sampling.
1
arXiv:2611.24807v1  [physics.chem-ph]  18 Nov 2026

1 Introduction
GFN1-xTB1and GFN2-xTB2,3are semiempirical tight-binding quantum-chemical methods
applicable to a broad range of elements, molecular structures, and noncovalent interactions.
Their relatively low computational cost has supported widespread use in conformational
searches, geometry optimization, reaction exploration, and multiscale simulation3,4. These
applications repeatedly evaluate energies and forces for many molecules, conformations, or
sampled frames. Conformational search tools such as CREST use GFN2-xTB to evaluate
large numbers of conformations and explore chemical space5–7. QMugs8uses GFN2-xTB to
optimize drug-like molecular conformations and records the resulting energies, structures,
and electronic properties. Studies of path sampling, geometry optimization, and reaction
screening likewise use members of the GFN-xTB family for repeated energy and force eval-
uations9–12. In quantum mechanical/molecular mechanical (QM/MM) simulation combined
with machine-learning correction potentials, GFN-xTB also provides a low-level baseline for
dynamics13,14and higher-level energy corrections15. These multimolecule, multiconforma-
tion, and multiwindow tasks create a demand for batched energy and force evaluation.
We target batches of similarly sized molecules, multiple conformations of one molecule,
and independent sampling windows. These tasks provide many independent calculations
that can be aggregated, while retaining differences in system size and self-consistent charge
(SCC) convergence within a batch. Efficient execution therefore requires control of data-
organization and scheduling costs. Per-system calls limit the amortization of memory man-
agement, data transfer, and GPU kernel launches, and padding all inputs to the largest
dimension in a batch adds redundant storage. Successful results also remain useful when a
small number of systems fail to converge. We therefore organize batches by size to aggre-
gate parallel work while accommodating within-batch differences and preserving per-system
results.
GPU implementations of semiempirical methods16–19and other electronic-structure ap-
proaches20–24offer several routes to accelerated solution. Steinbach and Bannwarth demon-
strated asynchronous execution of multiple GFN2-xTB SCF calculations on one GPU and
applied it to conformational sampling16. Integral optimizations for xTB reduce redundant
work by reusing recurrence intermediates and screening shell pairs25. The embeddable inter-
faces of xTB and tblite operate on individual systems3,4, with the caller organizing multisys-
tem workloads. dxtb supports CUDA and batched SCC with padded tensors for mixed-size
inputs26. TBMaLT also uses padded batches and removes converged systems during SCC27.
GPU acceleration of DFTB for large systems addresses Hamiltonian diagonalization and
density-matrix construction28, and parallel capabilities have also been developed systemat-
ically in DFTB+29. Here we examine queue organization and natural-size storage in the
target small-matrix regime and connect batched evaluation to cross-window execution in
periodic QM/MM.
WedevelopedxTBloomasaGPUbatchinglibraryforGFN-xTBenergyandforceevalua-
tion, withGFN2-xTBasthemainquantitativetarget. xTBloomsortsworkbyatomicorbital
(AO) count, accommodates within-batch size differences through native ragged inputs, and
maintains each system’s SCC state and results independently. Its CPU and CUDA back-
ends share a C application binary interface (ABI). Python interfaces for ASE30and dpdata31
connect batched electronic-structure evaluation to molecular simulation and data-generation
2

workflows.
Toassessintegrationandaccelerationinmolecularsimulation, weuseacompanionplugin
for LAMMPS32to perform cross-window QM/MM umbrella sampling on a single GPU
and incorporate a Deep Potential–Range Correction (DPRc) model. Numerical consistency
tests and GPU benchmarks evaluate xTBloom itself. Free-energy calculations with the
uncorrected QM/MM model and the DPRc-corrected model demonstrate its application,
while matched timings against sequential window execution quantify the throughput benefit
of cross-window batching.
2 Theoretical background
2.1 GFN-xTB energies and self-consistent electronic structure
GFN1-xTB1and GFN2-xTB2combine parameterized electronic, repulsive, dispersion, and
related energy contributions. GFN2-xTB additionally includes multipolar electrostatics and
density-dependent dispersion3. For modelmand nuclear coordinatesR, we denote the
finite-electronic-temperature energy scale byτand the dimensionless electronic entropy by
eSe.
m∈ {GFN1-xTB,GFN2-xTB}, τ=k BTe,eSe=S e/kB.(1)
The variational objective of the self-consistent electronic problem can then be written as
Fm(P,x;R, τ) =E m(P,x;R)−τ eSe.(2)
Here,Pis the total electronic density matrix, obtained by summing the density matri-
ces of the spin channels. The vectorxcollectively denotes the SCC variables determined
from those spin-channel density matrices. For GFN2-xTB, these variables include shell
charges and atomic multipoles, together with the corresponding spin layout when the spin-
polarization extension is enabled. At finite electronic temperature, the xTBloom public
interface reports the electronic Helmholtz free energy.
The schematic energy decomposition for GFN2-xTB and its optional spin-polarization
extension separates geometric and electronic-state dependencies1–3,33
EGFN2 = Tr[PH0(R)] +E iso(q;R) +E aniso(q,µ,Θ;R)
+E(2)
D4(q;R) +E rep(R) +E(3)
D4(R) +E spin.(3)
Here,qdenotes shell and atomic charges, andµandΘdenote atomic dipole and
quadrupole moments. The isotropic and anisotropic terms each contain their corresponding
electrostaticandexchange-correlationcontributions. ThespGFN2-xTBextensionintroduces
the spin-dependent energyE spin33. The original GFN2-xTB model hasE spin= 0. At fixed
geometry, the overlap matrix, zeroth-order Hamiltonian, and geometric interaction factors
can be constructed in advance. The repulsive term and the charge-independent D4 three-
body term are also constant. Charges, multipoles, and their associated potentials change
with the electronic state, and the D4 two-body term participates in the charge-dependent
3

self-consistent calculation. Geometry-dependent data can therefore be reused across SCC
iterations, while density-dependent energies and potentials are updated at each iteration.
A GFN-xTB evaluation first prepares the structure and parameters and constructs
geometry-dependent matrices and interaction factors. It then assembles the Hamiltonian
from the current SCC state and enters the self-consistent iteration. For each spin channel
σ, it solves the generalized eigenproblem
Hσ(x;R)Cσ=S(R)Cσεσ,(4)
Here,Cis the orbital-coefficient matrix andεis the diagonal matrix of orbital energies.
The orbital coefficients and the diagonal occupation matrixfdetermine the density matrix
Pσ=Cσfσ(Cσ)T,P=X
σPσ.(5)
Each spin channel satisfies orbital normalization in the nonorthogonal basis and an
electron-number constraint,
(Cσ)TSCσ=I,Tr(PσS) =X
ifσ
i=N σ.(6)
The model’s reference valence-electron counts and the molecular charge determine the
total number of valence electrons. The spin constraint further specifies the electron number
in each spin channel,N σ. Restricted calculations share spatial orbitals while determining the
occupationsandelectronic-entropycontributionsofthetwophysicalspinchannelsseparately.
Population analysis in a nonorthogonal atomic orbital basis incorporates the overlap matrix.
The resulting shell charges and atomic multipoles determine the corresponding potentials
for the next Hamiltonian construction.
Self-consistency requires agreement between the input SCC state and the output state
obtained from the updated density. A mixing algorithm uses their residual and its history
to generate the next input and suppress oscillations. In the common-orbital GFN2-xTB
calculations considered here, convergence requires both the root-mean-square (RMS) resid-
ual of the full SCC state and the absolute change in electronic free energy to be strictly
smaller than their respective tolerances. The residual contains shell-charge, atomic-dipole,
and quadrupole components. SI S1.2 defines their units and normalization. Each iteration
evaluates the free energy from the updated density and its associated electronic variables.
The final energy and forces are evaluated after the convergence conditions are met.
The cost of the eigensolver depends primarily on the AO dimension, while short-range
interaction costs depend on atom count and spatial distribution. The number of SCC it-
erations determines how often electronic-structure operations are repeated. Differences in
orbital spectra, initial electronic states, and charge response can produce different iteration
counts even at the same AO dimension. Organizing work by atom count alone therefore
gives an incomplete account of these costs.
4

2.2 Energies and analytic forces in QM/MM coupling
At finite electronic temperature, analytic forces are derivatives of the Helmholtz free en-
ergy, including its electronic-entropy contribution. Coordinate derivatives also involve the
overlap, Hamiltonian, and other geometry-dependent terms. Energy definitions, occupation
conventions, and force definitions must therefore be consistent in geometry optimization and
dynamics.
Analytic forces connect electronic-structure evaluation to geometry optimization and
molecular dynamics. In QM/MM models34,35, the electronic state of the quantum region
responds to the classical environment, which also contributes through classical interactions
and electrostatic embedding. Interoperable free-energy software frameworks that combine
QM/MM and machine-learning potentials provide related workflow context36.
The total QM/MM energy includes the retained classical bonded terms, MM–MM and
QM–MMLennard–Jonesinteractions, MM–MMelectrostatics, andtheGFN2-xTBQM/MM
electronic free energy. Biased sampling adds the corresponding bias potential. Near-field
MM point charges enter Hamiltonian construction during SCC. The response of the periodic
environment to QM atomic charges and its energy contribution are written as
ϕper=b+Aq, E per=qTb+1
2qTAq.(7)
Here,qis the vector of net QM atomic charges,bis the potential shift independent of
the current QM charges, andAis a symmetric response matrix. Consequently,
ϕper=∂Eper
∂q(8)
Periodic electrostatics includes complementary Ewald terms37,38, self-interaction correc-
tions, and subtraction of interactions already included explicitly39,40. It also requires the
coordinate derivatives of the external operators.
Explicit MM point charges enter the electronic-structure problem through screened shell-
monopole interactions. For fixed geometry and point-charge inputs, this external potential
can be precomputed and added to the shell potentials in each SCC iteration. Changes in QM
charges still modify the self-consistent energy and electronic state. The periodic response
includes feedback from the current QM charges and is updated with the SCC state. Both
explicit point charges and the periodic response therefore participate in the embedded self-
consistent solution.
For a given geometry and set of external inputs, SCC convergence yields the embedded-
system free energy, including the model energy terms, explicit point charges, and periodic
response. Denoting this free energy by Femb
m(R;b,A, τ), xTBloom returns the QM atomic
force
FxTBloom
I =−∂Femb
m
∂RI
b,A,τ.(9)
This derivative includes the geometric dependencies represented within the library and
holdsb,A, and the electronic-temperature energy scaleτfixed. xTBloom evaluates the
5

analytic derivatives of explicit point-charge interactions with respect to both QM-atom and
point-charge coordinates and returns point-charge forces when requested. For a periodic
response that depends on coordinates or the cell, the coupling driver supplies the explicit
derivatives ofbandAand combines them with classical-interaction and bias forces to obtain
the complete QM/MM force.
Deep Potential–Range Correction (DPRc)41learns the energy and force differences be-
tween higher-level and lower-level QM/MM descriptions. Related range-corrected∆MLP
work examines the transferability of MACE graph neural networks in QM/MM42. The dif-
ference potential and its negative gradient are added to the lower-level QM/MM baseline43.
LetU highandU lowdenote the higher-level and lower-level effective QM/MM potential
energies. The difference potential and corrected energy are then
∆U θ(R)≃U high(R)−U low(R),
Ucorr(R) =U low(R) + ∆U θ(R).(10)
Here,θdenotes the model parameters, andRcontains the coordinates of the QM and
MM atoms.
The difference labels and production simulations use the same lower-level energy defini-
tion. Atfiniteelectronictemperature, theelectroniccontributionisthefreeenergyconsistent
with the analytic forces.
The corrected energy and forces retain the lower-level QM/MM baseline, so advancing
the dynamics continues to require repeated GFN-xTB electronic-structure solutions.
3 Methods and implementation
xTBloom solves individual molecules independently and executes batches on the GPU. Sys-
tems share computational resources while retaining their own electronic states, convergence
histories, andoutputs. ThecompleteexecutionschemeorganizesbatchesthroughAOsorting
and adjusts work according to the active SCC state. Successive evaluations reuse topology
plans, workspaces, and compatible electronic states to reduce repeated preparation and data
transfer.
3.1 BatchorganizationanddatalayoutforbatchedGPUevaluation
Even size-grouped molecular batches can differ in atom count, orbital dimension, and in-
teraction count. These operations favor different data organizations. Atom-based layouts
provide only an indirect measure of eigenproblem size, while uniform padding to the largest
system in a batch introduces redundant storage. xTBloom separates the external molecular
representation from the execution layout of each computational stage. Callers provide inputs
at each system’s natural size, and the execution layer organizes GPU tasks by the actual
computational dimensions.
The public interface represents ragged batches using flat atom arrays, offsets, and system-
level charge and spin information. Preparation establishes mappings among atoms, shells,
orbitals, and matrix storage. Original system indices link internal tasks to output locations.
6

Internal matrices may be packed or padded to meet solver requirements, with system identity
and result order preserved throughout these layout transformations.
Before partitioning a queue into batches, xTBloom sorts the entire queue by AO count
to concentrate systems of the same dimension within a call. Matrix solves within each
batch are grouped by exact AO dimension, and the solver path is selected from the matrix
size and group population. System mappings restore results to the original input order.
Short-range geometric processing selects dense enumeration or spatial-grid atom-pair lists
according to atom count. Each short-range interaction uses its own cutoff definition. Long-
range electrostatics is exempt from a uniform short-range cutoff.
3.2 Batched SCC execution for nonuniform convergence
Systems within one size group may still require different numbers of SCC iterations.
xTBloom maintains a separate electronic state, mixing history, iteration count, and termi-
nation state for each system and advances it according to its own convergence conditions.
After each update, unconverged systems without numerical failures form the active set for
the next iteration. Converged systems retain their final electronic states. Systems with
numerical failures enter the corresponding failure state. Electronic variables are frozen for
both classes of terminated systems.
The execution layer handles the shrinking active set through compaction or masking.
For eligible closed-shell batches with small AO dimensions, it compacts active members and
performs subsequent eigensolves and orbital back-transformation on those members. Other
eligible batches retain their allocated computational capacity and mask state updates for
terminated members. Compaction reduces some redundant computation, while masking
avoids repeated batch reorganization. Supplementary scans show that the benefit of active-
capacity scheduling depends jointly on AO dimension, solver-bucket capacity, and active
fraction. Reorganization and dispatch costs dominate most measurements with larger AO
dimensions and bucket capacities, and substantial component-level gains occur mainly for
very small active sets (SI S5.3). Supported GPU execution paths perform convergence checks
andschedulesubsequentiterationsonthedevice,reducingper-iterationhostinteraction. The
SI specifies the implementation conditions.
SCC nonconvergence and eigensolver failures attributable to an individual system are re-
turnedasper-systemstatuses. Whenthecallreturnssuccessfully, allrequestedfloating-point
results for a failed member are filled with NaNs, while healthy members retain their success-
ful results. Input-validation, resource-allocation, and device or stream runtime failures are
handled as call-level errors. SI S2.2 defines output-commit semantics.
3.3 Data reuse and residency for successive evaluations
Geometry optimization and molecular dynamics repeatedly evaluate one topology, and each
geometry requires multiple SCC iterations. xTBloom organizes data reuse at the topology,
geometry, and electronic-state levels. Each class of data is reused over its applicable com-
putational stages to reduce repeated preparation and memory-management costs. Figure 1c
summarizes these reuse levels.
7

A fixed-topology plan stores elements, system boundaries, charges, spin information, and
computational policy and owns reusable workspace. The plan and storage remain reusable
as coordinates change, with geometry-dependent numerical quantities rebuilt for the new
geometry. Within the SCC iterations of one geometry, the overlap matrix and its Cholesky
factor remain constant and can be reused to transform each generalized eigenproblem.
Electronic-state contributions to the Hamiltonian are updated at each iteration. Topology
and workspace reuse thus spans successive geometries. Overlap-matrix and Cholesky-factor
reuse applies within a fixed geometry.
Electronic-state warm starts exploit continuity between successive geometries. For the
most recent fully converged batch in the same context, the runtime can reuse its electronic
state as the SCC initial guess for a new geometry when the policy remains compatible. The
runtime rejects warm starts when the topology or policy is incompatible. Initial-state reuse
controls iteration initialization, while workspace reuse controls storage management. Each
evaluation assesses convergence for its current geometry.
On GPU execution paths, transfers follow the actual locations of the input and output
buffers, supporting host, device, and mixed memory modes. Computational workspace re-
mains on the device, and results are delivered to their specified destinations. GPU-resident
data can therefore pass directly to subsequent computations. The same solver interface
supports host-driven independent batches, repeated calls, and device-side workflows.
8

aBatch layout and task organization
Molecular inputs
0
1
2
3Flat atoms + offsets
03 913 20
System identity
0 1 2 3Short-range geometry
Dense pairs / spatial pair lists
Matrix-size-aware solver paths
d1 d2d3
0 2 1 3Per-system output
SCC → E, F, state
0 E, F
1 E, F
2 NaN
3 E, F
bIndependent SCC histories cData lifetime and reuse
1 2 3 4 5 6 7 8
0
1
2
3
SCC iteration
Active compaction
After iteration 41 3Masked updates
Same capacity0 1 2 3R(t) R(t + 1) R(t + 2)
Across
geometries
Within
one geometry
Each SCC
iterationTopology plan + workspace
S, L S, L S, L
Compatible warm startFigure 1. GPU batch organization and data reuse. (a) Flat atom arrays and offsets preserve
system identity; solver paths depend on matrix size and group population. (b) Each system
has an independent SCC history and termination state. Active members are processed by
compaction or masking; lanes and iteration counts are schematic. (c) Topology plans and
workspaces are reused across compatible geometries, whereas the overlap matrixSand its
Cholesky factorLare reused within a fixed geometry. Dashed arrows indicate electronic-
state warm starts after full convergence of the previous geometry under a compatible policy.
9

3.4 LAMMPS QM/MM plugin and cross-window batching on one
GPU
We developed a QM/MM plugin in the companion LAMMPS project to apply cross-window
GPU batching to molecular dynamics. The plugin uses synchronized multipartition execu-
tion (Figure 2) to batch QM/MM coupling evaluations across windows at the same inte-
gration step, allowing independent windows to share one GPU. Batching covers GFN2-xTB
electronic-structure evaluation, classical nonbonded interactions, periodic electrostatics us-
ingtheparticle–particleparticle–mesh(PPPM)methodwithfastFouriertransforms(FFTs),
and DeePMD/DPRc inference when enabled. Each window retains its own coordinates, in-
tegrator, and electronic-state history.
Each LAMMPS partition advances one window trajectory and handles classical inter-
actions, periodic boundaries, the thermostat, and bias potentials. Partition root processes
coordinate through intranode MPI shared memory to collect QM coordinates, MM point
charges, and periodic-response inputs at synchronized integration steps. A coordinator calls
the xTBloom batch executor and distributes energies, forces, charges, and SCC statuses us-
ing fixed window indices. This mapping preserves the electronic-state history of each window
across successive geometries. After all windows complete their current-step evaluations suc-
cessfully, the dynamics driver assembles the total forces and advances integration. A solver
failure in any window stops the current synchronized step.
The plugin evaluates periodic electrostatics in two stages. Before SCC, it constructs the
periodic environmental potentialband charge-response matrixAfrom the current geometry
and MM charges. During SCC, the periodic feedback is updated from the current QM
charges throughb+Aq. After convergence, the final QM charges complete the QM mesh
evaluation. The two stages reuse field and spectral data. The LAMMPS coupling layer
supplies external-operator derivatives and combines them with explicit point-charge reaction
forces and classical forces. Periodic electrostatics, electronic structure, and the classical force
field therefore use a consistent charge state within each integration step.
In DPRc-corrected simulations, each window evaluates the lower-level GFN2-xTB
QM/MM baseline using the QM coordinates, MM environment, and constraint state
from the same time step. DPA4C44represents the local environmental dependence, and
DeePMD-kit45,46evaluates the difference potential and its coordinate derivatives. The
correction omits MM–MM interaction terms. The dynamics workflow adds the difference
potential and its negative gradient to the lower-level QM/MM energy and forces, together
with the bias potential and its forces. Section 2.2 defines the corresponding energies.
10

aIndependent LAMMPS windows
Window 1
Coordinates
Integrator
SCC historyWindow k
Coordinates
Integrator
SCC historyWindow 48
Coordinates
Integrator
SCC history…
MPI shar ed-memory coordinator
Collect / scatter by window index
bCross-window batch evaluation
Classical nonbonded
batch
LJ + real-space electrostaticsPeriodic pr e-SCC · batch
PPPM / FFT
MM mesh → b, AxTBloom · GPU batch
GFN2-xTB SCC → E, q, F
QM + point-char ge forcesDeePMD-kit · batch
DPA4C-DPRc · optional
ΔE, ΔF
Periodic post-SCC · batch
QM mesh +
operator derivativesb, A
Reuse fields
q
Total for ces + synchr onized-step check
Classical · periodic · QM · bias · optional DPRc
Advance all windows Stop step · r eset history
Other workflow xTBloom GPU batch Optional DPRc Field reuse / failureFigure 2. Cross-window QM/MM batching on one GPU. (a) Independent LAMMPS win-
dows retain their own coordinates and SCC histories; an MPI shared-memory coordinator
collects and distributes data by window index. (b) Classical, periodic, GFN2-xTB and op-
tional DPRc computations are batched across windows, followed by force assembly and a
synchronized-step check. Periodic preprocessing suppliesbandAto xTBloom, and the
converged QM chargesqenter postprocessing. Gray marks the windows and coordination,
classical and periodic calculations, and force assembly; teal marks xTBloom GPU batching,
and ochre marks optional DPRc. Dashed arrows indicate field reuse or, at the cross, stopping
the step and resetting electronic-state history after a failure.
11

4 Computational details
We used four groups of experiments to evaluate numerical agreement, full-evaluation per-
formance, execution design and application integration. Cross-program comparisons, finite
differences and controlled mixed batches tested numerical behavior and failure handling. Full
evaluations compared the GPU implementation with CPU and external-program baselines.
AO-sorting and active-capacity experiments examined the effects of batch composition and
SCC scheduling in Section 5.2.2. Cross-window QM/MM umbrella sampling tested integra-
tion into molecular simulation, with matched sequential-window and batched timings used
to measure application throughput.
4.1 Numerical comparisons and per-system failure isolation
The numerical tests used the standard conformance suite, QM9 and OMol25. QM947pro-
vided a baseline of small, neutral CHONF molecules. OMol2548covered a wider range of
elements, charges, spins and nonequilibrium geometries. Its numerical stress subset con-
tributed difficult configurations to the numerical statistics. A separate operational stress set
was used to assess per-system completion in Section 5.3. Table 1 summarizes the samples
and evaluation tasks.
ThexTBprogramandtbliteprovidedtheGFN2-xTBnumericalreferences. Wealsocom-
pared the energies, forces and execution statuses returned by the xTBloom CPU and CUDA
backends. The optional spin-polarization extension was disabled in every numerical com-
parison in this section. Open-shell inputs retained their supplied unpaired-electron counts
under a restricted, common-orbital treatment. Results were aligned by system identifier.
Program-specific model and numerical settings are given in SI S1.2, and the performance
protocol is described in Section 4.2.
For each system, we recorded input support, execution status and numerical-test out-
comes. SCC nonconvergence, GPU memory exhaustion, timeouts, reference-calculation fail-
ures and implementation errors were recorded separately. Analytic forces were tested by
finite differences. Controlled mixed batches tested the return of per-system results when one
member failed to converge (SI S2.2).
Each of the 20,000 numerical-test systems in Table 1 was evaluated with xTBloom CPU,
xTBCPU,tbliteCPUandxTBloomCUDA.FortheCPU-programcomparisons, theenergy-
difference threshold was5×10−7Eh. The force metric was the maximum absolute difference
over all Cartesian components of each system, with a threshold of5×10−6Eh/bohr. The
corresponding CPU–CUDA thresholds were1×10−6Eh and1×10−6Eh/bohr. CPU-
program error distributions and joint pass rates used the common-success set of all three
CPU programs. CPU–CUDA statistics used the intersection of successful results from the
two backends.
Operationalcoveragewastestedseparatelyon2,000difficultOMol25configurationsspan-
ning open-shell, charged, metal-containing, large, environmental and reactive inputs (Table
1). Eachengineevaluatedtheseinputsindividually. Theexecutionenvironmentandresource
allocation are described in Section 4.2.
12

Table 1. Datasets and numerical-validation tasks.
Dataset Systems
or configu-
rationsCoverage Evaluation metrics
QM9 numerical set 10,000 Small neutral molecules Three-program CPU differences,
CPU–CUDA differences and execu-
tion status
OMol25 main numeri-
cal set8,000 Multiple elements, charges, spins and
nonequilibrium geometriesAs above
OMol25 numerical
stress subset2,000 Difficult electronic states and geome-
triesAs above
OMol25 performance
set2,048 Similar sizes, neutral singlets and
metal-free light main-group elementsFull-queue time, throughput and
output differences
OMol25 operational
stress set2,000 Open-shell, charged, metal-
containing, large, environmental
and reactive inputsPer-system success rate and failure
categories
4.2 Performance workloads and timing protocol
The CPU backend uses the same model, parameters and result definitions as the CUDA
backend and serves as a numerical reference and single-core performance baseline. The
primary performance comparison evaluates the full implementation. Sorting and batch-size
comparisons are described in Section 4.3.
The performance workload comprised a fixed queue of 2,048 similarly sized OMol25
configurations. The inputs were neutral, singlet, metal-free light main-group systems with
100–119 GFN2 AOs. xTBloom sorted the queue by AO count and evaluated CUDA batches
ofB= 256. The CPU baselines used single-core, sequential per-system calls to xTBloom
and xTB. We tested dxtb atB= 1, 64, 128 and 256. Its fastest configuration that completed
the full queue,B= 64, was used for the primary comparison. The remaining configurations
and complete results are reported in the SI.
All configurations evaluated double-precision GFN2-xTB energies and forces at an elec-
tronic temperature of 300 K, with an SCC limit of 500 iterations and a FRESH initial
state. Each CPU configuration used one physical core, and each GPU configuration used
one RTX 5090. xTBloom was built in Release mode. Software versions, build options,
reference-program settings and sample composition are given in the SI.
The timing protocol specified one warmup traversal and three measured traversals of
the full input queue for each configuration. We report the median primary evaluation time
and its minimum–maximum range. Primary time includes per-batch adapter and descriptor
preparation, required transfers, SCC and force calculation, synchronization and complete
publication of host results. xTBloom preparation includes context creation and reuse han-
dling. Input reading, AO sorting, Executor construction, destruction of the final retained
context and compressed result storage fall outside the primary timing interval. Full-input
throughput is the number of configurations divided by the primary time for the complete
queue. Execution status, finite outputs and CPU–CUDA agreement were checked alongside
13

timing. Configuration-specific timing scopes are defined in SI S4.2.
The same-topology workload used an ideal geometry and locally perturbed configura-
tions. ForB= 512, each batch contained the ideal geometry and 511 configurations with
independent Gaussian perturbations (0.020 bohr standard deviation per Cartesian coordi-
nate). xTBloom CPU and CUDA were compared with sequential xTB/tblite calls under
a one-physical-core host constraint. The primary configuration wasB= 512with a cold
electronic state. TheB= 128auto-warm configuration examined consecutive evaluations.
Configuration generation, hardware, core affinity and initialization protocols are given in SI
S6.
4.3 Ablation and configuration-sensitivity protocols
The AO-sorting ablation compared the full implementation with original-order batching on
the same fixed queue atB= 256. Both configurations retained grouping by exact AO
dimension within each batch. All other computational and timing settings followed Section
4.2. Per-system results were aligned by their original identifiers to compare energies, forces,
execution statuses and SCC iteration counts. A separate original-order scan atB= 1, 64
and 256 measured batch-size sensitivity.
The primary AO-matrix layout comparison used the same inputs and software settings
to test natural-size storage against storage padded to the largest AO dimension in each
batch. Batch membership, actual-dimension eigensolves, precision and SCC settings were
held fixed. Full energy-and-force evaluation times and independently sampled GPU memory
were measured. The complete protocol is given in SI S5.4.
The SCC-scheduling ablation compared an active-capacity dispatch chain with a fixed-
capacity CUDA Graph using the same batch inputs, initial electronic states and SCC-loop
execution counts. The AO and SCC-scheduling experiments used their respective workloads
and settings, detailed in SI S5.1 and Table S5a. The scheduling results are reported in
Section 5.2.2. An additional single-iteration SCC component comparison held the initial
state and termination mask fixed while varying AO count, solver-bucket capacity and active
fraction. Its protocol and results are given in SI S5.3.
4.4 QM/MM and DPRc free-energy simulations
Thefree-energyapplicationusedtheethylenephosphatemodelreactionwithamethoxyleav-
ing groupstudied byGiese et al49. The periodic systemcontained a16-atom QMregion com-
prising two complete molecules and 2,974 TIP4P-Ew water molecules50. The QM region had
a total charge of−2eand used closed-shell GFN2-xTB. We completed separate sampling cal-
culations with uncorrected GFN2-xTB QM/MM and GFN2-xTB QM/MM+DPA4C-DPRc,
retaining the same classical environment and constraints.
The reaction coordinate wasξ=r(P,O5′)−r(P,O2′), where O2′is the attacking oxygen
and O5′is the leaving oxygen49. We placed 48 umbrella windows at 0.1 Å intervals from
−3.1to 1.6 Å51,52. Each window was equilibrated for 100 ps and then sampled in three 250
ps production trials with distinct random-number streams, giving 36 ns per Hamiltonian.
LAMMPS32propagated the dynamics with a 1 fs time step and a Langevin thermostat53
14

at 298 K. Colvars54applied the biases, and SHAKE55constrained the water geometry. A
coordinator allowed all 48 windows to share one GPU for batched QM/MM evaluation.
The DPRc dataset contained 6,560 periodic configurations labeled with the energy and
force differences between PBE0/6-31G* QM/MM56,57and GFN2-xTB QM/MM. The high-
level calculations used AmberTools with QUICK58,59. A DPA4C correction model was used
for the DPRc production trajectories. Label construction, training, window-level data splits
and held-out errors are described in SI S8.
The weighted histogram analysis method (WHAM) combined the window samples to
obtain the potential of mean force (PMF) for each Hamiltonian60. The resulting profiles
were compared with the digitized gwTP-PBE0 reference of Giese et al49. Statistical analysis
and the comparison protocol are described in SI S9.
Application performance was measured in short synchronized runs on one RTX PRO
6000, comparing cross-window batches with sequential execution of the same windows. The
short-run timings used the DPRc model available before retraining. Production sampling
used the model retrained on 6,560 configurations. Training and data splits are described in
SI S8, and the full timing protocol is given in SI S7.1.
5 Results and discussion
5.1 Numerical agreement and analytic-force tests
Among the systems successfully evaluated by all three CPU programs, xTBloom and tblite
jointly passed the energy and force criteria for 99.985% of configurations. xTBloom’s GFN2
parameters were generated from tblite data, parts of its numerical implementation used
tblite as a reference, and the relevant numerical settings were aligned for this comparison.
The xTBloom–xTB and tblite–xTB differences were similar in magnitude (Table 2a). Both
comparisons involving xTB had lower joint pass rates for restricted open-shell inputs than
for closed-shell inputs (Table S1f). Full error distributions, statistics stratified by closed-shell
and restricted open-shell inputs, outlier configurations and detailed settings are reported in
SI S1.
Table 2a. The 95th percentiles of absolute differences over 19,897 systems successfully eval-
uated by all three CPU programs.
Program pair Energy difference / Eh Maximum force-component dif-
ference / (Eh/bohr)
xTBloom–tblite 2.26×10-101.48×10-8
xTBloom–xTB 2.94×10-56.39×10-6
tblite–xTB 2.94×10-56.39×10-6
The CPU and CUDA backends jointly passed the energy and force criteria for 99.995% of
their common-success systems. Table 2b summarizes the numerical differences and execution
statuses.
15

Table 2b. Numerical agreement and execution statuses of xTBloom CPU and CUDA.
Metric Result
Common-success systems 19,937
Both energy and force criteria
satisfied19,936 (99.995%)
SCC nonconvergence in both
backends37
CPU success, CUDA failure 13
CUDA success, CPU failure 13
Maximum energy difference /
Eh2.40×10-2
Maximum force-component
difference / (Eh/bohr)2.90×10-1
The CPU and CUDA results for the outlier configuration corresponded to two different
self-consistent SCC solutions. Exchanging the converged terminal states as initial guesses
allowed both backends to reproduce each solution branch. Energy and force differences
within each branch satisfied the original comparison thresholds.
Both backends completed the finite-difference tests for all 32 QM9 systems and 29/32
OMol25 systems. Every examined force component of the completed systems satisfied the
test threshold. The other 3 OMol25 systems terminated because of invalid input or SCC
nonconvergence. The selected systems, errors and failure statuses are listed in the SI.
5.2 GPU performance
We first evaluated the full implementation on a queue of distinct molecules and on batches
of conformers with one topology. Ablations then examined the contributions and operating
regimes of the execution strategies.
5.2.1 Performance of the full implementation
Onthenear-sizeOMol25queueofdistinctmolecules, xTBloomCUDAatB= 256completed
all 2,048 configurations in each of the three measured traversals and achieved 71.13 config-
urations s-1(Figure 3a). Primary evaluation time decreased from 83.802 s on its single-core
CPU backend to 28.792 s on CUDA, corresponding to a 2.91-fold throughput increase. This
comparison establishes a full-evaluation throughput benefit for the tested queue of similarly
sized neutral singlets. AO sorting and batch-size effects are examined in Section 5.2.2.
Under the stated numerical settings, primary times for single-core xTB and dxtb atB=
64were 2.87 and 11.19 times the xTBloom CUDA time, respectively. The fastest complete
dxtb configuration wasB= 64, and itsB= 256runs exhausted GPU memory. Table S4b
lists every configuration. Table S4c reports cross-program energy and force differences and
sensitivity to a stricter xTB reference.
16

xTBloom
CUDA, B=256xTBloom
CPU, B=1xTB
CPU, B=1dxtb
CUDA, B=640100200300Full-queue time (s)
28.79283.802 82.493322.294(a) OMol25 queue, 2,048 configurations
14
2632
6262
122122
242
Atoms (top) / AOs (bottom)100101102Full-batch time (s)
66.555
52.808
2.926(b) Same topology, B=512 (cold)
xTB CPU
xTBloom CPU
xTBloom CUDA
Host: 1 physical CPU core (EPYC 9354)  |  GPU: RTX 5090  |  median and range of 3 measurementsFigure 3. Energy-and-force evaluation on two workloads. (a) Primary evaluation time
for the full near-size OMol25 queue of 2,048 configurations; batch sizes are shown in the
panel. Timing includes per-batch preparation, transfers, calculation, synchronization and
host-result publication; AO sorting was completed beforehand. (b) Full-batch time for 512
same-topology geometries at each molecular size, using a cold electronic state. Contexts
and descriptors were prepared before timing; all energies and forces were available on the
host when timing ended. xTB CPU evaluated the geometries sequentially. The atom-count
and time axes in (b) are logarithmic. Bars and points show medians, and whiskers show
minimum–maximum ranges over three full-queue traversals in (a) and three runs in (b).
Each host process used one EPYC 9354 physical core; GPU runs used one RTX 5090. See
SI S4.2 and S6 for timing protocols.
17

Figure 3b gives the single-core comparison for locally perturbed conformers with a com-
mon topology. AtB= 512with cold electronic states, median xTBloom CUDA batch times
were 0.876 and 2.926 s for the 62- and 122-atom systems. The corresponding ratios of se-
quential xTB time to CUDA time were 22.55 and 22.75. xTBloom CPU batch times were
12.942 and 52.808 s. Every path was restricted to one physical CPU core. The full size and
batch-size curves are shown in Figure S4.
5.2.2 Ablation experiments
We used controlled comparisons of AO sorting, primary AO-matrix storage and SCC active-
capacity scheduling to examine how execution choices and batch size affect evaluation cost.
AO sorting used the near-size queue from Section 5.2.1. SCC scheduling used controlled
small-AO batches, followed by scans over AO count, solver-bucket capacity and active frac-
tion to characterize its operating regime.
AO sorting of the near-size queueImproving batch composition through AO sorting
reduced evaluation cost for the tested queue. Using the fullB= 256configuration as the
reference, we disabled AO sorting on the same near-size queue and tested the batch-size
sensitivity of original-order batching (Figure 4 and Table S4b).
For fixed inputs andB= 256, primary time increased from 28.792 s with the full im-
plementation to 38.094 s with AO sorting disabled. AO sorting therefore reduced primary
evaluation time by 24.4% relative to original-order batching on this queue.
With AO sorting disabled, original-order batches of 1, 64 and 256 achieved 6.97, 27.63
and 53.76 configurations s-1, respectively. Throughput atB= 1was below the single-core
CPU baseline. Batches of 64 and 256 exceeded that baseline, showing the effect of per-batch
workload on measured GPU throughput.
Across these configurations, the maximum differences from the first measured single-core
CPU traversal were9.52×10−13Eh in energy and4.18×10−13Eh/bohr in force components.
Per-configuration statuses, convergence flags and SCC iteration counts matched.
A supplementary diagnostic used 256 wide-size OMol25 configurations that both CPU
and GPU evaluations completed successfully. With FRESH electronic states and host inputs
and outputs, median end-to-end time was 8.964 s for sequential single-core CPU evaluation
and 29.525 s for original-order GPU batching atB= 256. This batch contained 100 exact-
AO solver groups and had a maximum SCC count of 95 iterations.
At the same nominal batch size, grouping by exact AO count can produce many groups
containing only a few systems. The batched solver interface submits matrices of a common
dimension, so small groups provide limited parallel work compared with large same-topology
groups. The long tail of SCC iteration counts can also extend batch completion time. The
general masked path freezes the electronic-state updates of terminated members while re-
taining solver-group capacity, and the batch continues until its last active member finishes
SCC. Full diagnostics and supplementary sorting comparisons are provided in the SI.
18

0 20 40 60 80 100
Throughput (config. s−1
)AO sorted  B=256
AO sort off  B=256
AO sort off  B=64
AO sort off  B=171.13  (2.91× CPU)
53.76
27.63
6.97xTBloom CUDA xTBloom CPU, 1 physical coreFigure 4. Throughput in the AO-sorting ablation and batch-size scan. The first bar is
the full AO-sortedB= 256xTBloom CUDA implementation in Figure 3a; the others use
original-order batches without AO sorting. Bars and labels show medians over three full-
queue traversals; whiskers show minimum–maximum ranges. The dashed line is xTBloom
CPU throughput on one physical CPU core. Inputs and timing follow Section 4.2; complete
results are in Table S4b.
Storage and performance effects of the native ragged layoutWe separated the ef-
fectsofbatchorganizationandmatrixstoragebycomparingnatural-sizestoragewithstorage
padded to the maximum AO count in each batch under the same input and software settings.
Both paths retained batch membership, numerical kernels, actual-dimension eigensolves and
exact-AO groups. Only the storage and indexing of the primary AO matrices changed. The
comparison covered the near-size queue and both AO-sorted and strongly mixed batches of
the same wide-size queue. Full energy-and-force evaluation time and GPU memory were
measured independently, as detailed in SI S5.4.
Table 3 summarizes the three paired comparisons. For strongly mixed batches, the
NVML-sampled peak decreased from 11,572 to 3,802 MiB, and natural-size storage was
faster in all three full-evaluation traversals.
Table 3. Changes in storage and evaluation time for native ragged storage relative to storage
padded to the largest AO count in each batch.
Queue and batching (count, AO
range)Primary AO-matrix
capacity reduction
/ %Sampled GPU-
memory peak re-
duction / %Full E+F time re-
duction / %
Near-size, AO-sorted (2,048, 100–119) 2.16 1.03 0.19
Wide-size, AO-sorted (1,317, 8–180) 34.36 30.88 0.38
Wide-size, strongly mixed (1,317, 8–
180)75.90 67.14 4.02
Reductionsusepaddedstorageasthebaselineandarecalculatedas(paddedminusnative
ragged)/padded. Evaluation times are medians of three full energy-and-force traversals.
Absolute values and individual traversals are reported in Table S5e.
19

For the tested wide-size queue, native ragged storage reduced redundant matrix capacity
and GPU-memory demand.
Scheduling the SCC active setReducing the capacity of subsequent active solves can
reduce redundant work. The resulting performance benefit depends on whether the saved
computation offsets reorganization and dispatch overhead. To test execution scheduling
above the per-system SCC states, we compared a device dispatch chain that submits solves
at the active capacity with a CUDA Graph that retains the full capacity on controlled small-
AO workloads. The paths used the same batch inputs and initial states. Both retained per-
system convergence decisions, mixing histories, termination statuses and result mappings.
The test used mixed batches of H 2, He, LiH and CH 2atB= 1, 8, 32 and 128. Fully,
half- and quarter-active initial states were constructed by setting selected systems’ initial
iteration counters to the limit. The paths executed the same number of SCC loops and
matched in final energies, densities, populations, iteration counts and statuses. Initially
fully active configurations were also compared with a sequential CPU reference. Figure 5
shows the relative latencies.
For partially active batches, the dispatch chain reduced subsequent solve costs associated
with terminated slots. AtB= 32, the fixed-capacity/dispatch time ratios were 1.29 and 1.53
forthehalf-andquarter-activeconfigurations. Withinthiscontrolledworkloadandatafixed
batch size, lower initial active fractions produced a larger relative benefit from reducing the
submitted solver capacity.
The initially fully active paths had similar latencies. Fixed-capacity/dispatch time ratios
of 0.95 atB= 128and 0.96 atB= 1indicate slightly slower active-capacity dispatch. These
configurations offer less redundant solve work to remove, making reorganization and dispatch
costs more prominent. The extended scan showed that the benefit depends jointly on AO
count, solver-bucket capacity and active fraction. For AO dimensions of 122 and 180,B=
256and 1–2 active members, single-iteration component latency decreased by approximately
16.2%–27.0%. Reorganization and dispatch costs dominated most other tested points (SI
S5.3).
20

0.25 0.50 1.00
Initial active fraction0.91.01.11.21.31.41.5Time ratio (tfixed/tactive)
1.48
1.22
1.021.53
1.29
1.011.38
1.18
0.950.96
B = 8 B = 32 B = 128 B = 1Figure 5. Relative SCC-loop latency of the active-capacity dispatch chain and fixed-capacity
CUDA Graph at different initial active fractions. The ordinate is mean Graph time divided
by mean dispatch time; values above 1 favor dispatch.Bdenotes batch size; only the initially
fully active case was measured atB= 1. Each configuration was measured 50 times.
Timing covers the SCC loop through batch termination and excludes input preparation,
result downloads and force calculation. The dashed line marks equal time. Details are given
in SI S5.1 and Table S5a.
21

5.3 Per-systemfailureisolationandoperationalcoverageofdifficult
inputs
Successful results remained independently available when another batch member failed. A
healthy system and an SCC-nonconvergent system were evaluated together on the CPU host
path and the CUDA mixed-memory path. The failed member returned a per-system failure
status, with its energy, force and charge outputs filled with NaNs. Differences between
separate and joint evaluations of the healthy member satisfied the predefined tolerances.
Settings and results are given in SI S2.2.
We further evaluated operational coverage on the independent set of 2,000 difficult
OMol25 configurations using per-system runs. xTBloom CUDA completed 1,981 systems,
a success rate of 99.05%. The other 19 terminated through SCC nonconvergence, and the
GPU-memory-exhaustion count was zero. Figure 6b shows failure categories for the com-
parison programs. Full status counts are reported in Table S3b.
0.0 0.1 0.2 0.3 0.4
Failed inputs (%)0.25%
0.25%
0.30%
0.26%
0 1 2 3 4 5
Failed inputs (%)0.95%
3.85%
0.80%
0.60%(a) Numerical-comparison cohort n = 20,000 Failure counts
(b) Independent stress queue n = 2,000 Failure countsFailure rate Total SCC OOM Force
xTBloom CPU 50 50 — 0
xTBloom CUDA 50 50 0 0
xTB CPU 60 59 — 1
tblite CPU 52 52 — 0
Failure rate Total SCC OOM Invalid
xTBloom CUDA 19 19 0 0
dxtb CUDA 77 71 6 0
xTB CPU 16 16 — 0
tblite CPU 12 12 — 0
SCC: nonconvergence     OOM: GPU memory exhaustion     Force: non-finite required force
Invalid: invalid input     —: not applicable (CPU OOM)
Figure 6. Failure rates and counts in two input sets. (a) The numerical-comparison cohort
(n= 20,000); (b) an independent set of 2,000 difficult OMol25 configurations. Points show
the percentage of failed inputs, and columns give total failures and recorded categories.
SCCdenotesself-consistentchargenonconvergence; OOMdenotesGPUmemoryexhaustion.
Force in (a) denotes a non-finite required force, and Invalid in (b) denotes invalid input. A
dash marks GPU OOM as not applicable to CPU runs. The horizontal axes use different
ranges. Dataset composition is listed in Table 1; category counts are in SI Tables S1e(d) and
S3b.
These tests characterize the completion coverage and failure composition of each engine
on difficult inputs. xTBloom failures in this stress set were associated with the SCC conver-
gence tail. dxtb also encountered GPU memory exhaustion, indicating that electronic-state
behavior and execution resources both affect the observed failure categories.
22

5.4 QM/MM application results and cross-window acceleration
The LAMMPS integration used single-GPU cross-window batches to complete both the
uncorrected GFN2-xTB QM/MM and DPA4C-DPRc umbrella-sampling calculations. Each
calculation comprised 48 windows, with three 250 ps production trials per window and 36
ns of aggregate production sampling.
Single-window and 48-window evaluations of the same configuration agreed in energy and
forces, with a maximum force-component difference of approximately4.4×10−11kcal mol-1
Å-1. The DPRc production model passed 5 ps microcanonical (NVE) checks at three anchor
windows both before and after equilibration. Together with the two completed sampling
calculations, these checks show that batch electronic-structure evaluation can be integrated
with periodic embedding, a learned correction potential and dynamical propagation.
Figure 7a shows the central free-energy profiles. After each curve was independently
aligned toA(0) = 0, the root-mean-square errors (RMSEs) relative to the digitized gwTP-
PBE0 reference49were 6.61 kcal/mol for bare QM/MM and 0.866 kcal/mol for the DPRc-
corrected profile over 233 shared grid points. The two principal DPRc peak-to-reactant
free-energy differences were 9.21 and 16.45 kcal/mol as descriptive central values, differing
from the reference centers by +0.17 and−0.33kcal/mol. A local difference of approximately
3.1 kcal/mol remained nearξ≈1.53Å. Statistical diagnostics are given in SI S9.
23

−3 −2 −1 0 1
Reaction coordinate ξ (Å)−5051015A(ξ)−A(0) (kcal mol−1
)(a) Free-energy profiles
GFN2-xTB QM/MM QM/MM + DPRc gwTP-PBE0 reference
1 8 16 24 32 40 48
Number of umbrella-sampling windows050100150200Aggregate MD steps s−1
 GPU−1(b) Short-run throughput
QM/MM
QM/MM + DPRc
1 8 16 24 32 40 48
Number of umbrella-sampling windows0246810121416Matched-singleton speedup14.73×
13.65×(c) Short-run batching gain
QM/MM
QM/MM + DPRcFigure 7. QM/MM free-energy profiles and cross-window performance. (a) Central profiles
for bare GFN2-xTB QM/MM, the retrained DPRc model and the digitized gwTP-PBE0
reference, each aligned toA(0) = 0. (b,c) Single-GPU short-run timings as a function of
the number of umbrella-sampling windows. (b) Aggregate throughput; lines and shading
show the median and minimum–maximum range over five consecutive segments of the same
trajectory. (c) Speedup in total elapsed time relative to sequential execution of the same
windows. The comparison grid, protocol differences and statistical scope are defined in SI
S9.3; timing scopes are given in SI S7.1.
Application performance was measured in the single-GPU short synchronized runs (Fig-
ure 7b,c). At 48 windows, QM/MM and QM/MM+DPRc achieved aggregate throughputs
of 173.35 and 148.06 MD steps s-1GPU-1. The corresponding speedups over sequential ex-
ecution of the same windows were 14.73 and 13.65, reflecting the full coupled LAMMPS
batching scheme. The three production trials recorded 131.08–133.23 MD steps s-1GPU-1,
calculatedfromthesumsofproduction-segmentwallclockswithineachtrial. Thetwotiming
24

scopes are defined in SI S7.1.
6 Conclusions
xTBloom organizes independent GFN-xTB energy-and-force evaluations into GPU batches
forrepeatedelectronic-structureworkloads. Thegrowingdemandfrommolecularsimulation,
conformational sampling and scientific data generation makes sustained execution of these
independent tasks an important throughput problem. Batch electronic-structure evaluation
provides a route for connecting these calculations to larger molecular-simulation workloads.
The implementation combines AO-sorted task organization with per-system SCC histo-
ries and failure states, compatible data reuse and common simulation interfaces. Under the
tested restricted common-orbital settings and common-success definitions, xTBloom showed
close agreement with tblite and between its CPU and CUDA backends. Comparisons involv-
ing xTB contained threshold exceedances. On the representative near-size OMol25 workload,
full GPU evaluation achieved approximately 2.9 times its single-core CPU throughput. The
execution studies show that the benefit depends on batch composition, convergence behavior
and active fraction through the combined costs of task organization, SCC scheduling and
reuse.
Integration with LAMMPS supported periodic QM/MM dynamics and completed um-
brella sampling with both GFN2-xTB and DPA4C-DPRc correction. The free-energy calcu-
lations demonstrate application of the common batch interface to the baseline and corrected
potentials. In matched short synchronized tests with 48 windows on one GPU, cross-window
batching increased aggregate throughput by approximately an order of magnitude over se-
quential execution of the same windows. This integration extends the measured benefit to a
simulation workflow requiring repeated electronic-structure evaluations.
Overall, this work demonstrates the potential of GPU batching to improve throughput
for GFN-xTB energy-and-force evaluation and provides a computational basis for future
larger-scale QM/MM simulations incorporating learned correction potentials.
Code Availability
Source code, build instructions and interface documentation are available from the xTBloom
repository athttps://github.com/jinzhezenggroup/xtbloom. The cross-window coordi-
nator and periodic QM/MM coupling are available from the LAMMPS-DPRc repository at
https://github.com/jinzhezenggroup/lammps-dprc.
Acknowledgement
J.Z. was supported by the Advanced Materials - National Science and Technology Ma-
jor Project (No. 2025ZD0618702) and the National Natural Science Foundation of China
(Grant No. 22603140). T.Z. was supported by the National Natural Science Foundation of
China (Grant Nos. 92461313 and 22361132538), the AI for Science Program of the Shang-
hai Municipal Commission of Economy and Informatization (Grant No. 2025-GZL-RGZN-
25

BTBX-01004), and the Open Research Fund of Suzhou Laboratory (No. SZLAB-1508-2024-
TS017). Q.G. was supported by the National Natural Science Foundation of China (Grant
No. 12504285), the Advanced Materials - National Science and Technology Major Project
(No. 2025ZD0618401), and the Natural Science Foundation of Jiangsu Province (Grant
No.BK20250472). J.Z.andQ.G.arebothsupportedbytheFundamentalResearchFundsfor
the Central Universities. The AI-driven experiments, simulations, and model training were
performed on the robotic AI-Scientist platform of the Chinese Academy of Sciences. The nu-
merical calculations were performed on the supercomputing system in the Supercomputing
Center of the University of Science and Technology of China.
Supporting Information Available
Benchmark data and numerical settings; GPU and single-core CPU timing protocols; batch
scheduling, failure handling, and memory usage; periodic QM/MM umbrella sampling and
free-energy analysis; and data sources.
References
(1) Grimme, S.; Bannwarth, C.; Shushkov, P. A Robust and Accurate Tight-Binding Quan-
tum Chemical Method for Structures, Vibrational Frequencies, and Noncovalent Inter-
actions of Large Molecular Systems Parametrized for All spd-Block Elements (Z =
1-86).J. Chem. Theory Comput.2017,13, 1989–2009.
(2) Bannwarth, C.; Ehlert, S.; Grimme, S. GFN2-xTB-An Accurate and Broadly
Parametrized Self-Consistent Tight- Binding Quantum Chemical Method with Multi-
pole Electrostatics and Density-Dependent Dispersion Contributions.J. Chem. Theory
Comput.2019,15, 1652–1671.
(3) Bannwarth, C.; Caldeweyher, E.; Ehlert, S.; Hansen, A.; Pracht, P.; Seibert, J.;
Spicher, S.; Grimme, S. Extended tight-binding quantum chemistry methods.Wires
Comput. Mol Sci2021,11, e1493.
(4) Katbashev, A.; Stahn, M.; Rose, T.; Alizadeh, V.; Friede, M.; Plett, C.; Steinbach, P.;
Ehlert, S. Overview on Building Blocks and Applications of Efficient and Robust Ex-
tended Tight Binding.J. Phys. Chem., A2025,129, 2667–2682.
(5) Grimme, S. Exploration of Chemical Compound, Conformer, and Reaction Space with
Meta-Dynamics Simulations Based on Tight-Binding Quantum Chemical Calculations.
J. Chem. Theory Comput.2019,15, 2847–2862.
(6) Pracht, P.; Bohle, F.; Grimme, S. Automated exploration of the low-energy chemical
space with fast quantum chemical methods.Phys. Chem. Chem. Phys.2020,22, 7169–
7192.
(7) Pracht, P.; Grimme, S.; Bannwarth, C.; Bohle, F.; Ehlert, S.; Feldmann, G.; Gorges, J.;
Müller, M.; Neudecker, T.; Plett, C.; Spicher, S.; Steinbach, P.; Wesołowski, P. A.;
26

Zeller, F. CREST-A program for the exploration of low-energy molecular chemical
space.J. Chem. Phys.2024,160, 114110.
(8) Isert, C.; Atz, K.; Jiménez-Luna, J.; Schneider, G. QMugs, quantum mechanical prop-
erties of drug-like molecules.Sci. Data2022,9, 273.
(9) Li,X.-T.; Mi,S.; Xu,Y.; Li,B.-W.; Zhu,T.; Zhang,J.Z.H.DiscoveryofNewSynthetic
Routes of Amino Acids in Prebiotic Chemistry.Jacs Au2024,4, 4757–4768.
(10) Yang, M.; Zhang, D.; Wang, X.; Li, B.; Zhang, L.; E, W.; Zhu, T.; Wang, H. Ab Initio
Accuracy Neural Network Potential for Drug-Like Molecules.Res. (Washington D.C.)
2025,8, 0837.
(11) Li,B.; Xiao,J.; Gao,Y.; Zhang,J.Z.H.; Zhu,T.TransitionStateSearchingAccelerated
by Neural Network Potential.J. Chem. Inf. Model.2025,65, 2297–2303.
(12) Li, D.; Li, X.; Yun, J.; Liu, Q.; Gao, S.; Li, H.; Zhu, T.; Gou, R. Automated reaction
path discovery for coal pyrolysis: scalable dataset construction and mechanistic insights
for clean energy applications.Clean Energy2026,10, 157–168.
(13) Giese, T. J.; Zeng, J.; Lerew, L.; McCarthy, E.; Tao, Y.; Ekesan, Ş.; York, D. M.
Software Infrastructure for Next-Generation QM/MM-∆MLP Force Fields.J. Phys.
Chem., B2024,128, 6257–6271.
(14) Pan, X.; Yang, J.; Van, R.; Epifanovsky, E.; Ho, J.; Huang, J.; Pu, J.; Mei, Y.; Nam, K.;
Shao, Y. Machine-Learning-Assisted Free Energy Simulation of Solution-Phase and
Enzyme Reactions.J. Chem. Theory Comput.2021,17, 5745–5758.
(15) Li, B.; Mi, S.; Xiao, J.; Zhang, D.; Zhang, S.; Zhang, J.; Wang, H.; Zhu, T. General
reactive machine learning potentials for CHON elements.ChemRxiv2025, ChemRxiv,
Version 2.
(16) Steinbach, P.; Bannwarth, C. Acceleration of Semiempirical Electronic Structure The-
ory Calculations on Consumer-Grade GPUs Using Mixed-Precision Density Matrix Pu-
rification.J. Chem. Theory Comput.2025,21, 7335–7351.
(17) Wu, X.; Koslowski, A.; Thiel, W. Semiempirical Quantum Chemical Calculations Ac-
celerated on a Hybrid Multicore CPU-GPU Computing Platform.J. Chem. Theory
Comput.2012,8, 2272–2281.
(18) Zhou,G.; Nebgen,B.; Lubbers,N.; Malone,W.; Niklasson,A.M.N.; Tretiak,S.Graph-
ics Processing Unit-Accelerated Semiempirical Born Oppenheimer Molecular Dynamics
Using PyTorch.J. Chem. Theory Comput.2020,16, 4951–4962.
(19) Athavale,V.; Fedik,N.; Colglazier,W.; Niklasson,A.M.N.; Kulichenko,M.; Tretiak,S.
PYSEQM2.0: AcceleratedSemiempiricalExcited-StateCalculationsonGraphicalPro-
cessing Units.J. Chem. Theory Comput.2025,21, 9498–9510.
27

(20) Ufimtsev, I. S.; Martínez, T. J. Quantum Chemistry on Graphical Processing Units.
1. Strategies for Two-Electron Integral Evaluation.J. Chem. Theory Comput.2008,4,
222–231.
(21) Seritan, S.; Bannwarth, C.; Fales, B. S.; Hohenstein, E. G.; Isborn, C. M.; Kokkila-
Schumacher, S. I. L.; Li, X.; Liu, F.; Luehr, N.; Snyder, J. W.; Song, C.; Titov, A. V.;
Ufimtsev, I. S.; Wang, L.; Martínez, T. J. TeraChem : A graphical processing unit
-accelerated electronic structure package for large-scale ab initio molecular dynamics.
Wires Comput. Mol Sci2021,11, e1494.
(22) Barca, G. M. J.; Alkan, M.; Galvez-Vallejo, J. L.; Poole, D. L.; Rendell, A. P.; Gor-
don, M. S. Faster Self-Consistent Field (SCF) Calculations on GPU Clusters.J. Chem.
Theory Comput.2021,17, 7486–7503.
(23) Andrade, X.; Pemmaraju, C. D.; Kartsev, A.; Xiao, J.; Lindenberg, A.; Rajpurohit, S.;
Tan, L. Z.; Ogitsu, T.; Correa, A. A. Inq, a Modern GPU-Accelerated Computational
Framework for (Time- Dependent) Density Functional Theory.J. Chem. Theory Com-
put.2021,17, 7447–7467.
(24) Ju, F.; Wei, X.; Huang, L.; Jenkins, A. J.; Xia, L.; Zhang, J.; Zhu, J.; Yang, H.;
Shao, B.; Dai, P.; Williams-Young, D. B.; Mayya, A.; Hooshmand, Z.; Efimovskaya, A.;
Baker, N. A.; Troyer, M.; Liu, H. Acceleration without Disruption: DFT Software as a
Service.J. Chem. Theory Comput.2024,20, 10838–10851.
(25) Sevi, Y. Speed up overlap, dipole and quadrupole integrals and their gradients. grimme-
lab/xtb, GitHub pull request #1465, 2026;https://github.com/grimme-lab/xtb/p
ull/1465, Accessed 2026-10-18.
(26) Friede, M.; Hölzer, C.; Ehlert, S.; Grimme, S. dxtb-An efficient and fully differentiable
framework for extended tight-binding.J. Chem. Phys.2024,161, 062501.
(27) McSloy, A.; Fan, G.; Sun, W.; Hölzer, C.; Friede, M.; Ehlert, S.; Schütte, N.-E.;
Grimme, S.; Frauenheim, T.; Aradi, B. TBMaLT, a flexible toolkit for combining tight-
binding and machine learning.J. Chem. Phys.2023,158, 034801.
(28) Vuong,V.-Q.; Cevallos,C.; Hourahine,B.; Aradi,B.; Jakowski,J.; Irle,S.; Camacho,C.
Accelerating the density-functional tight-binding method using graphical processing
units.J. Chem. Phys.2023,158, 084802.
(29) Hourahine, B.; Berdakin, M.; Bich, J. A.; Bonafé, F. P.; Camacho, C.; Cui, Q.; De-
shaye, M. Y.; Díaz Mirón, G.; Ehlert, S.; Elstner, M.; Frauenheim, T.; Goldman, N.;
González León, R. A.; van der Heide, T.; Irle, S.; Kowalczyk, T.; Kubař, T.; Lee, I. S.;
Lien-Medrano, C. R.; Maryewski, A.; Melson, T.; Min, S. K.; Niehaus, T.; Niklas-
son, A. M. N.; Pecchia, A.; Reuter, K.; Sánchez, C. G.; Scheurer, C.; Sentef, M. A.;
Stishenko, P. V.; Vuong, V. Q.; Aradi, B. Recent Developments in DFTB+, a Software
Package for Efficient Atomistic Quantum Mechanical Simulations.J. Phys. Chem., A
2025,129, 5373–5390.
28

(30) Hjorth Larsen, A.; Jørgen Mortensen, J.; Blomqvist, J.; Castelli, I. E.; Christensen, R.;
Dułak, M.; Friis, J.; Groves, M. N.; Hammer, B.; Hargus, C.; Hermes, E. D.;
Jennings, P. C.; Bjerre Jensen, P.; Kermode, J.; Kitchin, J. R.; Leonhard Kolsb-
jerg, E.; Kubal, J.; Kaasbjerg, K.; Lysgaard, S.; Bergmann Maronsson, J.; Max-
son, T.; Olsen, T.; Pastewka, L.; Peterson, A.; Rostgaard, C.; Schiøtz, J.; Schütt, O.;
Strange, M.; Thygesen, K. S.; Vegge, T.; Vilhelmsen, L.; Walter, M.; Zeng, Z.; Ja-
cobsen, K. W. The atomic simulation environment-a Python library for working with
atoms.J. Phys., Condens. Matter: Inst. Phys. J.2017,29, 273002.
(31) Zeng, J.; Peng, X.; Zhuang, Y.-B.; Wang, H.; Yuan, F.; Zhang, D.; Liu, R.; Wang, Y.;
Tuo, P.; Zhang, Y.; Chen, Y.; Li, Y.; Nguyen, C. T.; Huang, J.; Peng, A.; Rynik, M.;
Xu, W.-H.; Zhang, Z.; Zhou, X.-Y.; Chen, T.; Fan, J.; Jiang, W.; Li, B.; Li, D.;
Li, H.; Liang, W.; Liao, R.; Liu, L.; Luo, C.; Ward, L.; Wan, K.; Wang, J.; Xiang, P.;
Zhang, C.; Zhang, J.; Zhou, R.; Zhu, J.-X.; Zhang, L.; Wang, H. dpdata: A Scalable
Python Toolkit for Atomistic Machine Learning Data Sets.J. Chem. Inf. Model.2025,
65, 11497–11504.
(32) Thompson, A. P.; Aktulga, H. M.; Berger, R.; Bolintineanu, D. S.; Brown, W. M.;
Crozier, P. S.; in ’t Veld, P. J.; Kohlmeyer, A.; Moore, S. G.; Nguyen, T. D.; Shan, R.;
Stevens,M.J.; Tranchida,J.; Trott,C.; Plimpton,S.J.LAMMPS-aflexiblesimulation
tool for particle-based materials modeling at the atomic, meso, and continuum scales.
Comput. Phys. Commun.2022,271, 108171.
(33) Neugebauer, H.; Bädorf, B.; Ehlert, S.; Hansen, A.; Grimme, S. High-throughput
screening of spin states for transition metal complexes with spin-polarized extended
tight-binding methods.J. Comput. Chem.2023,44, 2120–2129.
(34) Lin, H.; Truhlar, D. G. QM/MM: what have we learned, where are we, and where do
we go from here?Theor Chem Acc2007,117, 185–199.
(35) Senn, H. M.; Thiel, W. QM/MM methods for biomolecular systems.Angew. Chem.
(International, Engl.)2009,48, 1198–1229.
(36) Tao, Y.; Giese, T. J.; Ekesan, Ş.; Zeng, J.; Aradi, B.; Hourahine, B.; Aktulga, H. M.;
Götz, A. W.; Merz Jr, K. M.; York, D. M. Amber free energy tools: Interoperable
software for free energy simulations using generalized quantum mechanical/molecular
mechanical and machine learning potentials.J. Chem. Phys.2024,160, 224104.
(37) Darden, T.; York, D.; Pedersen, L. Particle mesh Ewald: An N log(N) method for
Ewald sums in large systems.J. Chem. Phys.1993,98, 10089–10092.
(38) Essmann, U.; Perera, L.; Berkowitz, M. L.; Darden, T.; Lee, H.; Pedersen, L. G. A
smooth particle mesh Ewald method.J. Chem. Phys.1995,103, 8577–8593.
(39) Nam, K.; Gao, J.; York, D. M. An efficient linear-scaling Ewald method for long-range
electrostatic interactions in combined QM/MM calculations.J. Chem. Theory Comput.
2005,1, 2–13.
29

(40) Giese, T. J.; York, D. M. Ambient-Potential Composite Ewald Method for ab Initio
QuantumMechanical/MolecularMechanical MolecularDynamicsSimulation.J. Chem.
Theory Comput.2016,12, 2611–2632.
(41) Zeng, J.; Giese, T. J.; Ekesan, Ş.; York, D. M. Development of Range-Corrected Deep
Learning Potentials for Fast, Accurate Quantum Mechanical/Molecular Mechanical
Simulations of Chemical Reactions in Solution.J. Chem. Theory Comput.2021,17,
6993–7009.
(42) Giese, T. J.; Zeng, J.; York, D. M. Transferability of MACE Graph Neural Network
for Range Corrected∆-Machine Learning Potential QM/MM Applications.J. Phys.
Chem., B2025,129, 5477–5490.
(43) Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A. Big Data Meets Quan-
tum Chemistry Approximations: The∆-Machine Learning Approach.J. Chem. Theory
Comput.2015,11, 2087–2096.
(44) Li, T.; Xue, J.; Zhang, L.; Zhang, D.; Wang, H. Universal Machine-learning Molecular
Dynamics at the Speed of Empirical Potentials.arXiv2026, 2608.19041.
(45) Wang, H.; Zhang, L.; Han, J.; E, W. DeePMD-kit: A deep learning package for many-
body potential energy representation and molecular dynamics.Comput. Phys. Com-
mun.2018,228, 178–184.
(46) Zeng, J.; Zhang, D.; Peng, A.; Zhang, X.; He, S.; Wang, Y.; Liu, X.; Bi, H.; Li, Y.;
Cai, C.; Zhang, C.; Du, Y.; Zhu, J.-X.; Mo, P.; Huang, Z.; Zeng, Q.; Shi, S.; Qin, X.;
Yu, Z.; Luo, C.; Ding, Y.; Liu, Y.-P.; Shi, R.; Wang, Z.; Bore, S. L.; Chang, J.;
Deng, Z.; Ding, Z.; Han, S.; Jiang, W.; Ke, G.; Liu, Z.; Lu, D.; Muraoka, K.; Oliaei, H.;
Singh, A. K.; Que, H.; Xu, W.; Xu, Z.; Zhuang, Y.-B.; Dai, J.; Giese, T. J.; Jia, W.;
Xu, B.; York, D. M.; Zhang, L.; Wang, H. DeePMD-kit v3: A Multiple-Backend Frame-
work for Machine Learning Potentials.J. Chem. Theory Comput.2025,21, 4375–4385.
(47) Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A. Quantum chemistry
structures and properties of 134 kilo molecules.Sci. Data2014,1, 140022.
(48) Levine, D. S.; Shuaibi, M.; Spotte-Smith, E. W. C.; Taylor, M. G.; Hasyim, M. R.;
Michel, K.; Batatia, I.; Csányi, G.; Dzamba, M.; Eastman, P.; Frey, N. C.; Fu, X.;
Gharakhanyan, V.; Krishnapriyan, A. S.; Rackers, J. A.; Raja, S.; Rizvi, A.;
Rosen, A. S.; Ulissi, Z.; Vargas, S.; Zitnick, C. L.; Blau, S. M.; Wood, B. M. The Open
Molecules 2025 (OMol25) Dataset, Evaluations, and Models.arXiv2025, 2505.08762.
(49) Giese, T. J.; Zeng, J.; York, D. M. Multireference Generalization of the Weighted
Thermodynamic Perturbation Method.J. Phys. Chem. A2022,126, 8519–8533.
(50) Horn, H. W.; Swope, W. C.; Pitera, J. W.; Madura, J. D.; Dick, T. J.; Hura, G. L.;
Head-Gordon, T. Development of an improved four-site water model for biomolecular
simulations: TIP4P-Ew.J. Chem. Phys.2004,120, 9665–9678.
30

(51) Torrie, G.; Valleau, J. Nonphysical sampling distributions in Monte Carlo free-energy
estimation: Umbrella sampling.J. Comput. Phys.1977,23, 187–199.
(52) Kästner, J. Umbrella sampling.Wires Comput. Mol Sci2011,1, 932–942.
(53) Schneider, T.; Stoll, E. Molecular-dynamics study of a three-dimensional one-
component model for distortive phase transitions.Phys. Rev. B1978,17, 1302–1322.
(54) Fiorin,G.; Klein,M.L.; Hénin,J.Usingcollectivevariablestodrivemoleculardynamics
simulations.Mol. Phys.2013,111, 3345–3362.
(55) Ryckaert, J.-P.; Ciccotti, G.; Berendsen, H. J. Numerical integration of the cartesian
equations of motion of a system with constraints: molecular dynamics of n-alkanes.J.
Comput. Phys.1977,23, 327–341.
(56) Adamo, C.; Barone, V. Toward reliable density functional methods without adjustable
parameters: The PBE0 model.J. Chem. Phys.1999,110, 6158–6170.
(57) Hariharan, P. C.; Pople, J. A. The influence of polarization functions on molecular
orbital hydrogenation energies.Theor., Chim, Acta1973,28, 213–222.
(58) Case, D. A.; Aktulga, H. M.; Belfon, K.; Cerutti, D. S.; Cisneros, G. A.; Cruzeiro, V.
W. D.; Forouzesh, N.; Giese, T. J.; Götz, A. W.; Gohlke, H.; Izadi, S.; Kasavajhala, K.;
Kaymak, M. C.; King, E.; Kurtzman, T.; Lee, T.-S.; Li, P.; Liu, J.; Luchko, T.; Luo, R.;
Manathunga, M.; Machado, M. R.; Nguyen, H. M.; O’Hearn, K. A.; Onufriev, A. V.;
Pan, F.; Pantano, S.; Qi, R.; Rahnamoun, A.; Risheh, A.; Schott-Verdugo, S.; Sha-
jan, A.; Swails, J.; Wang, J.; Wei, H.; Wu, X.; Wu, Y.; Zhang, S.; Zhao, S.; Zhu, Q.;
Cheatham 3rd, T. E.; Roe, D. R.; Roitberg, A.; Simmerling, C.; York, D. M.; Na-
gan, M. C.; Merz Jr, K. M. AmberTools.J. Chem. Inf. Model.2023,63, 6183–6191.
(59) Cruzeiro, V. W. D.; Manathunga, M.; Merz Jr, K. M.; Götz, A. W. Open-Source
Multi-GPU-Accelerated QM/MM Simulations with AMBER and QUICK.J. Chem.
Inf. Model.2021,61, 2109–2115.
(60) Kumar, S.; Rosenberg, J. M.; Bouzida, D.; Swendsen, R. H.; Kollman, P. A. THE
weighted histogram analysis method for free-energy calculations on biomolecules. I.
The method.J Comput. Chem1992,13, 1011–1021.
(61) Chodera, J. D. A Simple Method for Automated Equilibration Detection in Molecular
Simulations.J. Chem. Theory Comput.2016,12, 1799–1805.
(62) Flyvbjerg, H.; Petersen, H. G. Error estimates on averages of correlated data.J. Chem.
Phys.1989,91, 461–466.
(63) Geyer, C. J. Practical Markov Chain Monte Carlo.Stat., Sci,1992,7, 473–483.
31

TOC Graphic
Molecular inputs GPU batch execution Per-system outputs
A A Charge · spin Free energy · Forces
B B Charge · spin Free energy · Forces
C C Charge · spin NaNGPU
xTBloom
GFN-xTB
Atoms + offsets
32

Supporting Information:
xTBloom: GPU Batching for
High-Throughput GFN-xTB Calculations
Zitian Lu,†,‡Shengrui Xu,†,¶,‡Tong Zhu,∗,§,¶Qiangqiang Gu,∗,†,‡,∥and Jinzhe
Zeng∗,†,⊥,‡,∥
†School of Artificial Intelligence and Data Science, University of Science and Technology of
China, Hefei 230026, P.R. China
‡Suzhou Institute for Advanced Research, University of Science and Technology of China,
Suzhou 215123, P.R. China
¶SII-ECNU Joint Autonomous Chemistry Laboratory, Shanghai Innovation Institute,
Shanghai 200003, P.R. China
§Shanghai Engineering Research Center of Molecular Therapeutics and New Drug
Development, School of Chemistry and Molecular Engineering, East China Normal
University, Shanghai 200062, P.R. China
∥Suzhou Big Data & AI Research and Engineering Center, Suzhou 215123, P.R. China
⊥State Key Laboratory of Precision and Intelligent Chemistry, University of Science and
Technology of China, Hefei 230026, P.R. China
E-mail: tzhu@lps.ecnu.edu.cn; guqq@ustc.edu.cn; jinzhe.zeng@ustc.edu.cn
ThisSupportingInformationdescribesthesamplecomposition, numericaltests, performance
configurations, execution implementation, QM/MM simulations, and statistical methods
used in the main text. References are numbered independently in this document. Summary
values are reported with units and appropriate significant figures. Raw timing measurements
retain their full precision, and all statistics and pass criteria are evaluated using unrounded
data.
S1. Datasets and numerical comparisons
S-1

S1.1 Sample composition
The numerical comparison used QM9 and OMol25 dataS1,S2. We selected 10,000 QM9 con-
figurations, 8,000 configurations from the main OMol25 set and 2,000 configurations from
an OMol25 numerical stress subset, giving 20,000 unique inputs. The numerical stress sub-
set is included in the aggregate statistics in this section. The execution stress test in S3 is
documented separately. Table S1a summarizes the sample composition.
This section reports sample composition and unweighted statistics.
Table S1a. Sample composition for the numerical comparison.
Data group Configu-
rationsClosed-
shellOpen-
shellAtoms, median/
maximumAOs, median/
maximum
QM9 main set 10,000 10,000 0 18 / 29 45 / 56
OMol25 main set 8,000 6,439 1,561 41 / 350 114 / 884
OMol25 numerical
stress subset2,000 1,428 572 61 / 329 173.500 / 957
Total 20,000 17,867 2,133 19 / 350 46 / 957
An input was classified as open-shell when its specified number of unpaired electrons was
greater than zero. The statistics for the selected samples are unweighted.
Each input was evaluated with xTBloom CPU, native xTB CPUS3, tblite CPUS4and
xTBloom CUDA, producing 80,000 execution records. Failed runs are retained in the status
statistics for all inputs. Numerical differences are calculated on the corresponding common-
success sets.
S1.2 Calculation settings and difference statistics
The comparison used the nonperiodic molecular GFN2-xTB modelS5. Spin-polarization
extensions were disabled in every program. xTBloom CPU/CUDA used restricted mode
withanexplicitsingle-channelrepresentation, andtblitewasusedwithoutaspin-polarization
container. Open-shellinputsretainedtheirspecifiedcharge, multiplicityandnonzeronumber
of unpaired electrons and were treated with restricted open-shell common orbitals. The
compared quantities were the electronic free energy at finite electronic temperature and the
analytic forces.
Table S1b. Settings for the numerical comparison of 20,000 configurations.
Item Setting
Model and program versions GFN2-xTB, native xTB 6.7.1, tblite 0.7.0(C API 700).
Spin polarization Disabled in all programs in this section. No spin-polarization container
was attached to tblite. xTBloom CPU/CUDA used restricted mode with
an explicit single-channel representation.
Continued on next page
S-2

Item Setting
Common-orbital occupations Nα=(N e+u)/2 and Nβ=(N e−u)/2. Two sets of Fermi–Dirac occupations
are determined on the same spatial orbital spectrum and summed to
construct the total density. The input specifies the number of unpaired
electrons u. Each program retains its own numerical occupation solver.
GFN2 parameter source xTBloom GFN2 parameters were generated from the structured export of
tblite 0.7.0. Native xTB used the parameters built into version 6.7.1.
D4 parameter source The xTBloom D4 fitting and damping coefficients are included in the
tblite export described above. The D4 reference polarizability/C6 tables
were generated with dftd4. Program-specific cutoffs are given below.
Initial electronic guess Each system was cold-started independently, with no wavefunction reuse
between systems. xTBloom FRESH used SAD, and tblite used its default
SAD guess. Native xTB 6.7.1 retained its default Goedecker/EEQ guess
and its SAD fallback on failure. Each program used its own initial-guess
settings.
Electronic temperature 300 K, kBT = 9.500×10-4Eh
Maximum SCC iterations 500
xTBloom free-energy-
change/state-residual RMS
tolerances10-10Eh/ 10-8
xTB and tblite accuracy 10-4
xTBloom execution B = 32, host input/output, FRESH initial state
Reference-program execution Per-system calls, one CPU thread, single-threaded BLAS
Cross-program comparison
thresholds|∆E|≤5×10-7Eh, max|∆F|≤5×10-6Ehbohr-1
CPU–CUDA comparison
thresholds|∆E|≤10-6Eh, max|∆F|≤10-6Ehbohr-1
D4 cutoffs Hard molecular cutoffs in xTBloom/tblite, with tblite switch width = 0
bohr
Numerical libraries and inter-
preterLP64 OpenBLAS 0.3.30 and Python 3.12.3. tblite used its separately
bundled numerical library.
GPU and toolchain NVIDIA RTX 5090, driver 580.95.05, CUDA 12.9.1, nvcc 12.9.86, sm120,
GCC 13.3
The D4 reference data were generated with dftd4S6and used in the corresponding pa-
rameter tables.
The molecular D4 coordination-number, two-body and three-body cutoff radii in
xTBloom and tblite were 30, 50 and 25 bohr, respectively. The coordination-number and
three-body cutoff radii in native xTB were both 40 bohr. The two-body path in its SCC
dispersion-matrix routinebuild_wdispmatused a radius of 72 bohr. Each program retained
its own SCC residual definition.
Forthecommon-orbital, explicitsingle-channelGFN2-xTBpathusedhere, theSCCstate
vector contains all shell charges, followed by 3 dipole components and 6 stored quadrupole
S-3

components per atom. Charges, dipoles and quadrupoles use e, e a 0and e a 02, respectively.
The state vector stores the numerical values obtained by dividing each quantity by its atomic
unit. With Nsh shells and Nat atoms, its dimension is n=Nsh+9Nat.
r(k)= x(k)
out−x(k)
in, R(k)
RMS=vuut1
nnX
j=1(r(k)
j)2.(S1)
The input state constructs the Hamiltonian for the current iteration. The output state is
obtained from that iteration’s orbital occupations and density-matrix update. The residual
is calculated before mixing, with equal weight for every stored component. No additional
field scaling or off-diagonal multiplicity weighting is applied. The public API fieldcharge_t
olerancespecifies the RMS threshold for this complete state vector, andenergy_tolerance
specifies the threshold for the change in the complete electronic free energy. The stopping
conditions are
R(k)
RMS< εR,|F(k)− F(k−1)|< ε F.(S2)
Convergencerequiresbothconditionstohold. Thepreviousfree-energyvalueisinitialized
to zero for the first iteration and subsequently taken from the preceding iteration. These
definitions agree with Table S1b. CPU and CUDA use the same final stopping conditions.
The energy difference is the absolute difference in total energy. For each configuration,
the force difference is first reduced to the largest absolute difference over all 3N Cartesian
components. P50, P95, P99 and the maximum are then calculated across configurations.
Quantiles use linear interpolation. A joint pass requires both the energy and force thresholds
to be met. All three CPU program pairs use the same set of 19,897 configurations for which
all three programs succeeded. The CPU–CUDA comparison uses 19,937 configurations for
which both backends succeeded.
Table S1c. Complete error statistics for the three CPU program pairs, with N = 19,897 in
every row.
Program pair Quantity P50 P95 P99 Maximum
differenceConfigu-
rations
exceed-
ing
thresh-
old
xTBloom–xTB Energy / E h1.779×10-82.941×10-57.871×10-56.359 3684
xTBloom–xTB Force / E h
bohr-15.128×10-96.390×10-61.288×10-52.213×10-11441
xTBloom–tblite Energy / E h4.619×10-142.264×10-103.920×10-99.142×10-33
xTBloom–tblite Force / E h
bohr-13.244×10-91.482×10-83.267×10-82.914×10-23
xTB–tblite Energy / E h1.778×10-82.941×10-57.869×10-56.359 3682
xTB–tblite Force / E h
bohr-13.717×10-96.389×10-61.282×10-52.213×10-11440
S-4

10−1110−910−710−510−3
P95 |ΔE| (Eh)xTBloom–tblite
xTBloom–xTB
tblite–xTB2.264e-10
2.941e-05
2.941e-05Thresholda  Energy difference P95
10−1010−810−610−410−2
P95 max |ΔF| (Eh bohr−1)xTBloom–tblite
xTBloom–xTB
tblite–xTB1.482e-08
6.390e-06
6.389e-06Thresholdb  Force-component difference P95Figure S1. P95 numerical differences between CPU programs. (a) Absolute energy difference
per configuration. (b) Maximum absolute force-component difference per configuration. All
three comparisons use the same 19,897 common-success configurations. The horizontal axes
are logarithmic. Dashed lines mark the comparison thresholds of 5×10-7Ehand 5×10-6
Ehbohr-1, respectively. Table S1c gives all quantiles, maximum differences and exceedance
counts.
Table S1d. CPU–CUDA numerical differences and backend execution status. (a) Errors on
the common-success set, N = 19,937.
Quantity P50 P95 P99 Maximum dif-
ferenceConfigu-
rations
exceeding
threshold
Energy / E h 3.908×10-141.620×10-123.752×10-122.395×10-21
Force / E hbohr-13.018×10-154.808×10-135.465×10-112.899×10-11
Table S1d (continued). (b) Backend status combinations for all 20,000 inputs.
CPU status CUDA status Configurations
Success Success 19,937
SCC not converged SCC not converged 37
SCC not converged Success 13
Success SCC not converged 13
S-5

S1.3 Joint passes and outlier configurations
Table S1e. Joint comparison results and outlier configurations. (a) Joint energy-and-force
pass counts.
Comparison Common-success
configurationsJoint-pass configu-
rationsJoint pass rate
xTBloom CPU–tblite CPU 19,897 19,894 99.985%
xTBloom CPU–CUDA 19,937 19,936 99.995%
All three pairs of the three CPU programs 19,897 16,118 81.007%
Pairwise classification of the three CPU programs gave 16,212 configurations for which all
three pairs met the energy threshold and 18,455 for which all three met the force threshold.
For energy, xTBloom, xTB and tblite were the sole outlier in 1, 3,680 and 1 configurations,
respectively. Another 1 configuration had three mutually distinct results, and 2 showed
nontransitive threshold relations. The corresponding force counts were 1, 1,438, 1, 1 and
1. A sole outlier means that only the pair formed by the other two programs passed. A
nontransitive threshold relation means that exactly two program pairs passed.
Table S1e (continued). (b) The three configurations exceeding the xTBloom–tblite compar-
ison thresholds and their three-program differences.
Configuration ID Program pair |∆E| / E h max|∆F| / E hbohr-1
CO_6464156197254019571837842xTBloom–xTB 1.424×10-41.430×10-3
CO_6464156197254019571837842xTBloom–tblite 1.423×10-41.430×10-3
CO_6464156197254019571837842xTB–tblite 1.257×10-87.855×10-9
CO_1728990913863245799454246xTBloom–xTB 9.189×10-34.770×10-3
CO_1728990913863245799454246xTBloom–tblite 9.142×10-34.770×10-3
CO_1728990913863245799454246xTB–tblite 4.772×10-56.538×10-6
CO_6758694963201820714974302xTBloom–xTB 1.046×10-82.810×10-8
CO_6758694963201820714974302xTBloom–tblite 2.802×10-32.914×10-2
CO_6758694963201820714974302xTB–tblite 2.802×10-32.914×10-2
CO_6464156197254019571837842is a 20-atom configuration from the main set with
charge +3, 4 unpaired electrons and Pd. It was classified as an xTBloom-only outlier.
CO_1728990913863245799454246is a 116-atom, closed-shell configuration from the main
set with charge−2 and Ca, Si and Te. All three programs gave mutually distinct results.
CO_6758694963201820714974302is a 4-atom, closed-shell configuration from the numerical
stress subset with charge +3 and Al, O and H. It was classified as a tblite-only outlier.
S-6

Table S1e (continued). (c) The single configuration that succeeded on both CPU and CUDA
and exceeded the comparison thresholds.
Configuration ID CPU energy / E hCUDA energy /
Eh|∆E| / E hmax|∆F| / E h
bohr-1
CO_11122377363235061240527
26-64.895 -64.919 2.395×10-22.899×10-1
This configuration belongs to the main OMol25 set and contains 23 atoms with charge
+8 and a closed-shell specification. CPU and CUDA required 463 and 484 SCC iterations,
respectively. tblite did not converge within 500 iterations, so this configuration contributes
only to the common-success backend comparison.
For configurationCO_1112237736323506124052726, a diagnostic build of the original
numerical-comparison version was used to inject the shell charges, atomic charges, atomic
dipoles and quadrupoles from the converged CPU and CUDA cold-start final states into
the original B=32 batch. Each initial-guess type was supplied to both backends, with the
mixing history reinitialized. The diagnostic used host input/output, a 300 K electronic
temperature, a 500-iteration SCC limit, a free-energy-change tolerance of 10-10Ehand an
SCC state-residual RMS tolerance of 10-8. Both backends converged to the corresponding
branch from each initial-guess type. Within each branch, the energy and maximum force-
component differences met the CPU–CUDA thresholds in Table S1b.
TableS1e(continued). Final-stateinitial-guessexchangefortheconfigurationthatsucceeded
on both CPU and CUDA and exceeded the thresholds.
Shared initial-
guess sourceCPU energy /
EhCUDA energy /
Eh|∆E| / E hmax|∆F| / (E h
bohr-1)SCC itera-
tions, CPU/
CUDA
CPU cold-start
final state-64.89522334858208 -64.89522334858238 2.984×10-131.254×10-136/6
CUDA cold-start
final state-64.91917719009682 -64.91917719009554 1.279×10-123.557×10-137/7
Table S1e (continued). (d) Complete execution status by program.
Program QM9 main set OMol25 main set OMol25 numerical
stress subsetTotal
xTBloom CPU 10000/0/0 7961/39/0 1989/11/0 19950/50/0
xTB CPU 10000/0/0 7954/45/1 1986/14/0 19940/59/1
tblite CPU 10000/0/0 7961/39/0 1987/13/0 19948/52/0
xTBloom CUDA 10000/0/0 7962/38/0 1988/12/0 19950/50/0
Each cell in Table S1e(d) lists the numbers of successful runs, SCC nonconvergence cases
and nonfinite required outputs, in that order. The 80,000 execution records contain 212
S-7

failuresinvolving111distinctconfigurations. Ofthesefailures,211wereSCCnonconvergence
cases, and 1 was a nonfinite force output from native xTB for configurationCO_580259221
7901689389998917.
Table S1f. Numerical consistency stratified by closed-shell and restricted open-shell inputs.
Category Program pair Joint pass/
common successPass rate P95 |∆E| / E hP95 max|∆F| /
(Ehbohr-1)
Closed-shell xTBloom–xTB 14,740/17,782 82.893% 2.950×10-56.500×10-6
Closed-shell xTBloom–tblite 17,780/17,782 99.989% 1.341×10-111.198×10-8
Closed-shell xTB–tblite 14,740/17,782 82.893% 2.956×10-56.503×10-6
Restricted
open-shellxTBloom–xTB 1,379/2,115 65.201% 2.797×10-55.681×10-6
Restricted
open-shellxTBloom–tblite 2,114/2,115 99.953% 3.152×10-92.937×10-8
Restricted
open-shellxTB–tblite 1,381/2,115 65.296% 2.755×10-55.673×10-6
Closed-shell xTBloom
CPU–CUDA17,818/17,819 99.994% 1.478×10-122.391×10-13
Restricted
open-shellxTBloom
CPU–CUDA2,118/2,118 100.000% 2.046×10-122.573×10-12
Closed-shell and restricted open-shell inputs are defined by u=0 and u>0 unpaired elec-
trons,respectively. ThethreeCPUprogrampairsusethethree-programcommon-successset,
with closed-shell/restricted open-shell denominators of 17,782/2,115 and energy and force
thresholds of 5×10-7Ehand 5×10-6Ehbohr-1, respectively. CPU–CUDA uses the backend
common-success set, with corresponding denominators of 17,819/2,118 and a threshold of
10-6for both quantities in their respective atomic units. Force differences are reduced to
the maximum absolute Cartesian-component difference within each system. P95 is then cal-
culated by linear quantiles over the common-success systems in each group. A joint pass
requires both energy and force to meet their thresholds. Unsuccessful systems remain in
the status statistics in Table S1e. This table summarizes the per-system results for the
three-program and CPU–CUDA common-success sets.
S2. Analytic forces and failure isolation
S2.1 Finite-difference method and results
The finite-difference test selected 32 configurations from each of QM9 and OMol25 and fixed
8 Cartesian coordinates per configuration. Second-order central finite-difference forces were
calculated with h = 5×10-4, 10-3and 2×10-3bohr. Every displaced point was evaluated
independently from a FRESH initial state, with an electronic temperature of 300 K, a max-
imum of 500 SCC iterations, a free-energy-change tolerance of 10-8Ehand an SCC state
RMS tolerance of 10-6. The absolute analytic-versus-finite-difference force threshold was
10-5Ehbohr-1. This test checks each program’s energy derivatives against its own analytic
S-8

forces. For OMol25, 29/32 configurations completed, giving 696 checks. Table S2a reports
the maximum error over the three step sizes.
FFD
i(h) =−E(R+he i)−E(R−he i)
2h, ε i(h) =Fanalytic
i −FFD
i(h).(S3)
Table S2a. Finite-difference checks of analytic forces.
Dataset Backend Selected
configu-
rationsCompleted
configu-
rationsFinite-
difference
checksConfigu-
rations with
execution
failureNumerical
threshold
exceed-
ancesMaximum error / E h
bohr-1
QM9 CPU 32 32 768 0 0 1.374×10-6
QM9 CUDA 32 32 768 0 0 1.374×10-6
OMol25 CPU 32 29 696 3 0 2.347×10-6
OMol25 CUDA 32 29 696 3 0 2.347×10-6
Each completed configuration contributes 8 coordinates at 3 step sizes, giving 24 finite-
difference checks. The three incomplete OMol25 configurations terminated during evaluation
of the undisplaced structure. Table S2b lists the causes. CPU and CUDA failed on the same
configurations with the same failure categories.
Table S2b. OMol25 configurations that did not complete the finite-difference test.
Configuration ID Returned sta-
tusCause
CO_1057867831392699108473564Invalid input Incompatible parity of total and unpaired
electron counts
CO_2681531878022357052243628SCC not con-
vergedReached the 500-iteration SCC limit
CO_8298694747446508682846328SCC not con-
vergedReached the 500-iteration SCC limit
The finite-difference calculations ran on an NVIDIA RTX PRO 6000 Blackwell Work-
station Edition with driver 595.91.07 and CUDA toolkit 12.9.1. CPU and BLAS execution
were single-threaded.
S2.2 Controlled mixed batch
The controlled mixed batch contained the healthy configurationCO_6325012143019994842
243708with 13 atoms and the SCC-nonconverged configurationCO_9343753771713800272
817479with 48 atoms. Each configuration was first evaluated separately. They were then
submitted together as a ragged batch in healthy-then-failed order, requesting energy, forces
and atomic charges. CPU used host descriptors, and CUDA used mixed descriptors. Energy,
S-9

forces and charge use E h, Ehbohr-1and e, respectively. For the healthy configuration, the
absolute tolerance between separate and batched results was 10-10for each quantity.
Table S2c. Batched results for the healthy and failed configurations.
Execution
pathHealthy/
failed sta-
tus codesHealthy/
failed SCC
iterationsHealthy
energy dif-
ference /
EhHealthy maxi-
mum force dif-
ference / E h
bohr-1Healthy max-
imum charge
difference / eFailed E/
F/charge
CPU host 0 / 7 17 / 500 0 0 0 All NaN
CUDA mixed 0 / 7 17 / 500 0 3.123×10-170 All NaN
Per-system status codes 0, 7 and 8 denote success, SCC nonconvergence and eigensolver
failure, respectively. The call return code and per-system statuses were checked separately.
The two-member experiment ran on an NVIDIA RTX PRO 6000 Blackwell Workstation
Edition with driver 595.91.07 and CUDA toolkit 12.9.1. Both CPU and CUDA mixed mode
were tested with separate per-system calls and ragged batching. Table S2c reports member
statuses, convergence iteration counts and the output state of the failed system.
SCC nonconvergence and eigensolver failures attributable to an individual system are
reported through member statuses within a successful call. Input-validation, resource-
allocation and device/stream runtime errors are call-level failures. A call-level error detected
before caller-output commit leaves all result buffers and result flags unchanged. A catas-
trophic CUDA error after output commit begins returns an internal error and may leave
partially written outputs.
This experiment verifies member-level isolation of SCC nonconvergence within a success-
ful batched call and NaN filling of the failed member’s E/F/charge slices.
FRESH calls reinitialize the electronic state. Strict WARM is available only when the
preceding batch converged completely and the electronic-state identity is compatible.
S2.3 Tested configurations and coordinate indices
We selected 32 configurations from each of QM9 and OMol25 with 8 fixed Cartesian test
coordinates per configuration. CPU and CUDA used identical selections. Table S2d lists
all 64 configuration identifiers and coordinate indices. The indices use zero-based flattened
Cartesian numbering, with 3i, 3i+1 and 3i+2 corresponding to the x, y and z components of
zero-based atom i. The list includes configurations that were attempted but did not complete
the test. Tables S2a–S2b give execution statuses and error statistics.
Table S2d. Configurations and eight coordinate indices used for finite-difference checks.
Dataset Configuration ID Tested coordinate indices
QM9qm9_00745861, 16, 22, 5, 48, 56, 34, 45
QM9qm9_1333304, 0, 27, 12, 25, 3, 6, 20
Continued on next page
S-10

Dataset Configuration ID Tested coordinate indices
QM9qm9_04059730, 12, 46, 45, 37, 43, 23, 11
QM9qm9_11607251, 34, 11, 19, 33, 28, 41, 12
QM9qm9_08996443, 9, 3, 2, 52, 17, 12, 45
QM9qm9_04691553, 15, 6, 22, 3, 38, 5, 24
QM9qm9_04523726, 11, 13, 14, 30, 50, 46, 7
QM9qm9_05282237, 18, 47, 10, 16, 5, 0, 9
QM9qm9_05092040, 35, 27, 0, 42, 21, 17, 44
QM9qm9_1195815, 47, 42, 49, 34, 11, 15, 16
QM9qm9_01560918, 20, 13, 32, 36, 14, 23, 31
QM9qm9_07858238, 6, 28, 16, 34, 12, 23, 18
QM9qm9_0480918, 17, 43, 15, 27, 30, 34, 20
QM9qm9_07987921, 9, 54, 28, 10, 46, 56, 13
QM9qm9_01407111, 38, 16, 23, 39, 49, 17, 36
QM9qm9_02548831, 34, 41, 2, 12, 4, 29, 16
QM9qm9_11280737, 68, 32, 26, 59, 56, 65, 31
QM9qm9_03058610, 3, 40, 35, 44, 20, 6, 37
QM9qm9_08647020, 36, 24, 28, 10, 31, 12, 26
QM9qm9_08649041, 54, 13, 23, 22, 49, 44, 35
QM9qm9_02308625, 8, 43, 28, 4, 2, 32, 29
QM9qm9_07125921, 16, 26, 52, 0, 36, 47, 32
QM9qm9_05254139, 25, 7, 13, 20, 12, 16, 1
QM9qm9_04415731, 16, 7, 37, 33, 10, 23, 47
QM9qm9_0704673, 38, 46, 29, 9, 16, 1, 40
QM9qm9_07834754, 31, 1, 22, 35, 2, 10, 24
QM9qm9_06430636, 34, 52, 50, 38, 22, 29, 60
QM9qm9_1276631, 13, 25, 24, 0, 33, 4, 12
QM9qm9_04365613, 12, 35, 2, 26, 20, 16, 8
QM9qm9_06717743, 41, 0, 15, 30, 6, 4, 44
QM9qm9_12821616, 44, 46, 17, 24, 15, 54, 11
QM9qm9_08980142, 2, 12, 4, 30, 20, 13, 53
OMol25CO_113643124454311406442589621, 20, 9, 29, 5, 13, 28, 14
OMol25CO_6189934124259438229170486182, 168, 119, 89, 81, 160, 113, 140
OMol25CO_105786783139269910847356439, 12, 6, 21, 7, 27, 10, 16
OMol25CO_10506620510560063578590027, 28, 9, 15, 6, 31, 4, 13
OMol25CO_2681531878022357052243628103, 270, 2, 69, 100, 345, 286, 74
OMol25CO_884803980678865074330713156, 34, 60, 15, 48, 64, 17, 59
OMol25CO_3071188985923719149160631241, 164, 71, 45, 124, 294, 178, 87
OMol25CO_8298694747446508682846328193, 31, 245, 47, 184, 23, 120, 166
OMol25CO_6454588750467394776830157238, 21, 147, 175, 109, 66, 131, 228
OMol25CO_53771210899629089691486659, 5, 63, 50, 45, 4, 8, 11
Continued on next page
S-11

Dataset Configuration ID Tested coordinate indices
OMol25CO_681714843667615864216648931, 25, 121, 54, 16, 101, 110, 82
OMol25CO_1882940930435698542755778143, 46, 61, 152, 288, 24, 359, 181
OMol25CO_1100507435607172621851273171, 8, 164, 58, 169, 153, 170, 136
OMol25CO_7791841176610489324341915101, 6, 127, 252, 21, 303, 102, 220
OMol25CO_134479620074738482595139165, 22, 137, 10, 35, 109, 71, 136
OMol25CO_5095567109539540080386136201, 80, 167, 48, 209, 210, 183, 161
OMol25CO_48252483851919836973516886, 22, 12, 2, 8, 10, 18, 7
OMol25CO_156410265201884908205301352, 182, 19, 188, 140, 194, 227, 42
OMol25CO_171519321484240805137152043, 79, 123, 132, 121, 73, 125, 107
OMol25CO_6944664227163535839419051315, 106, 204, 77, 111, 235, 107, 229
OMol25CO_1781181311303345976285781159, 228, 129, 229, 73, 126, 101, 70
OMol25CO_527166041041273165414809637, 35, 29, 21, 30, 31, 40, 8
OMol25CO_155098808062544308547874425, 28, 24, 37, 33, 8, 30, 6
OMol25CO_124965729259804877371836156, 41, 77, 1, 95, 67, 3, 102
OMol25CO_115003665052404869151298833, 116, 140, 32, 120, 90, 138, 95
OMol25CO_116320940871874178879147616, 60, 52, 29, 51, 2, 12, 19
OMol25CO_131814712445616889261411860, 71, 102, 64, 96, 88, 17, 160
OMol25CO_652750393952843910036121136, 18, 25, 23, 3, 9, 43, 20
OMol25CO_7289135314389910106135448184, 331, 134, 344, 146, 277, 358, 38
OMol25CO_109102193409456414482870831, 23, 17, 33, 35, 21, 6, 38
OMol25CO_1261546634852081578532461151, 57, 224, 76, 292, 150, 216, 50
OMol25CO_109892434874200080382472023, 21, 41, 15, 25, 48, 44, 1
S2.4 Multiple-step-size diagnostics for three QM9 force components
Among the 71 QM9 force cases classified as native-xTB-only outliers in the three-program
CPU comparison, we selected the three configurations with the largest xTB–tblite force
differences. These wereqm9_066596,qm9_020345andqm9_108978, with force components
at flattened indices 25, 1 and 10, respectively. CPU cold-start calculations used step sizes
of 10-3, 5×10-4and 10-4bohr under two settings, comprising the original numerical settings
and a request for stricter SCC convergence. They produced 144 energy-point records and 54
derivative checks. Native xTB passed 0/18 checks in this diagnostic panel.
The second panel set the xTBloom free-energy-change and SCC state RMS tolerances to
10-12Ehand 10-10, respectively, and requested accuracy = 10-6from both reference programs.
The public xTB interface clamps this request to 10-4, giving the same effective xTB accuracy
in both panels. All energy points were evaluated successfully. Repeated evaluations of the
central point gave zero energy and force differences. The diagnostic error threshold remained
10-5Ehbohr-1.
S-12

Table S2e. Derivative diagnostics for three selected QM9 force components.
Program Passes/checks Maximum finite-difference error / E hbohr-1
xTBloom CPU 18 / 18 2.209×10-7
tblite CPU 18 / 18 2.211×10-7
native xTB CPU 0 / 18 3.537×10-2
S3. Difficult configurations and memory diagnostics
S3.1 Independent execution stress test
The execution stress test used 2,000 OMol25 configurationsS2. Table S3a summarizes the
categories, counted according to a predefined primary-category assignment rule.
Table S3a. Category composition of the execution stress set.
Stress category Configurations
Open-shell 600
Charged 600
Metal/heavy elements 300
Large systems 200
Environment 150
Reactivity/distortion 150
Total 2,000
The test evaluated GFN2-xTB energies and forces at an electronic temperature of 300 K,
with a maximum of 500 SCC iterations and reference-program accuracy = 10-4. xTBloom
and dxtb each used one RTX 5090 with B = 1. Native xTB and tblite used per-system calls
on one CPU thread. The CPU was an AMD EPYC 9354, and the interpreter was Python
3.10.19.
Table S3b. Complete execution status for the stress set.
Program Success SCC not con-
vergedOOM Invalid input Success rate
xTBloom CUDA 1,981 19 0 0 99.050%
dxtb CUDA 1,923 71 6 0 96.150%
native xTB CPU 1,984 16 — 0 99.200%
tblite CPU 1,988 12 — 0 99.400%
The two GPU programs shared 1,913 successful configurations, the two CPU programs
shared 1,979, and all four programs shared 1,905.
S-13

S3.2 Size-stratified execution tests on H200
ThesupplementaryH200testdividedthevariable-sizequeueintothreeAO-dimensionranges
and also evaluated the complete mixed queue in its original order. xTBloom used B=1, 64
and 256, and dxtb used B=1. Each execution configuration completed three measured
traversals. Throughput was calculated as the successful-system count divided by traversal
time, including the cost of failures. This test provides a size-stratified H200 diagnostic,
reported separately from the main RTX 5090 performance results.
S-14

Figure S2. Size-stratified throughput for the heterogeneous H200 queue. (a–d) AO counts
of 1–38, 39–69 and 70–180, followed by the complete mixed queue in its original order,
containing 739, 634, 675 and 2,048 configurations, respectively. Solid lines show xTBloom
throughput across batch sizes, and dashed lines show dxtb B=1. The vertical-axis ranges
differ between panels. Values are medians over three measured traversals.
S-15

S3.3 dxtb GPU memory for the near-size queue
The dxtb memory diagnostic used the same 2,048 near-size configurations and calculation
settings as S4. The atom-padding ratio is the sum of padded atom slots across batches
divided by the total number of real atoms. Memory is reported in GiB as the largest
allocated-memory record from the PyTorch allocator over the three measured traversals.
Table S3c. dxtb atom padding, memory allocation and queue completion.
Configuration Atom-padding
ratioMaximum recorded allo-
cated memory / GiBSuccessful configu-
rations per traversal
dxtb GPU batch 1 1.000 0.137 2048/2048
dxtb GPU batch 64 1.261 10.176 2048/2048
dxtb GPU batch 128 1.300 20.246 2048/2048
dxtb GPU batch 256, original order 1.322 30.154 0/2048
dxtb GPU batch 256, AO-sorted 1.266 30.309 256/2048
B = 1, 64 and 128 completed the full queue. At B = 256, the original-order and AO-
sorted configurations encountered OOM after publishing 0 and 256 successful results per
traversal, respectively. Their memory values report allocation before failure and exclude the
unsuccessful allocation request.
1 64 128 256 256
AO-sorted
dxtb batch size0102030Peak allocated (GiB)OOM OOMAllocated GPU memory
Complete traversal Recorded before OOM
Figure S3. Allocated dxtb GPU memory for the near-size queue. Solid bars denote config-
urations that completed the queue. Orange hatched bars denote the maximum allocation
recorded before OOM. Table S3c gives batch sizes, atom-padding ratios and successful-
configuration counts.
S-16

S4. Main near-size performance benchmark
S4.1 Fixed queue and queue-wide AO sorting
The main near-size performance benchmark used 2,048geom_orca6configurations from
the OMol25 train 4M catalog. All were marked eligible, had charge 0 and multiplicity 1,
contained no metals, belonged to thelight_main_groupelement category and had 100–119
GFN2 AOs.
This section gives the selection criteria and summary statistics for the 2,048 configura-
tions.
The fixed queue contained 27–58 atoms per configuration, with a mean of 41.974 and
85,963 atoms in total. The mean AO count was 110.644, the sample standard deviation
was 5.637 and the coefficient of variation was 5.095%. The default policy sorts the entire
queue by AO count before forming batches. In this benchmark, the runner performed the
same queue-wide AO sort and submitted B = 256 batches. AO-sorting ablations retained
the input order and measured B = 1, 64 and 256. Outputs were restored to their original
correspondence by configuration ID.
S4.2 Hardware, evaluation settings and timing
Thebenchmarksranonadual-socketAMDEPYC9354host. EachcomputeprocessorGPU-
driving host process waspinnedto one physical core, with OMP, BLAS and relevant PyTorch
thread counts set to 1. GPU configurations used one NVIDIA RTX 5090 with approximately
31.800 GiB of reported memory. xTBloom and external programs were measured separately
on different physical GPUs of the same model and different pinned CPU cores.
Table S4a. Calculation controls and measurement settings for the main performance bench-
mark.
Item Setting
Common settings GFN2-xTB, binary64, energy and forces, electronic temperature 300 K,
maximum 500 SCC iterations, FRESH initial state
xTBloom Public CPU/CUDA API, host input/output, SCC free-energy-change tol-
erance 10-8Eh, state-residual RMS tolerance 10-6, native analytic forces
native xTB libxtb 6.7.1, public C API, accuracy = 1.0, a new Calculator/Results per
batch, forces obtained by negating gradients
dxtb dxtb 0.4.0, PyTorch 2.10.0+cu128, float64,x_atol= 10-4,x_atol_max=
10-5,f_atol= 10-4,force_convergence= True,fermi_etemp= 300 K,
singlepoint with autograd forces
Build and runtime environment Release, CUDA toolkit 12.9.1, sm120, LP64 OpenBLAS 0.3.30, Python
3.10.19
Repetition protocol One full-queue warmup and three measured traversals per configuration,
with 2,048 structures per traversal
Continued on next page
S-17

Item Setting
Primary timing Sum of per-batch preparation, synchronous computation and complete
host-output publication
Traversal wall clock Also includes per-batch resource release, Python GC and progress han-
dling
Native xTB accuracy controls both SCC stopping criteria and integral screening. At ac-
curacy = 1.0, the energy-change threshold is approximately 10-6Eh, and the shell-charge and
multipole residual threshold is approximately 10-4. The tested dxtb path determines conver-
gence from the L2 norm and maximum component of the unmixed Fock fixed-point residual,
controlled byx_atolandx_atol_max, respectively.f_atolis a retained configuration field.
The dxtb method is described inS7.
One xTBloom context was reused between batches within each traversal, while every
compute call remained FRESH. Preparation included the batch adapter, descriptors, input
processing, required uploads, native-context creation, temporary-context destruction and
reuse switching. Computation included SCC, force evaluation and completion synchroniza-
tion. Publication included transfer of results to the host, removal of padding and complete
output publication. Primary timing is the sum of these three stages. Input loading, AO
sorting, Executor construction, initial module or library loading, final destruction of the re-
tained context and compressed result storage were outside both primary timing and traversal
wall clock. Per-batch Calculator or model creation in native xTB and dxtb was included in
preparation. Post-batch dxtb garbage collection was included in traversal wall clock, with
primary timing ending before that cleanup. Throughput was calculated as 2,048 divided by
the median primary time over three measurements.
Each of the 11 configurations underwent one warmup and three measured traversals,
giving 44 traversals and 90,112 target-configuration records in total. The measured runs
comprised 33 traversals and 67,584 target-configuration records. Inputs were fixed for every
configuration. Failed outputs had their energy and complete force slice filled with NaN.
S4.3 All configurations and timing results
Table S4b. Measured results for all 11 performance configurations.
Configuration Primary time, median
[minimum–maximum] /
sMedian wall
clock / sThroughput /
configurations
s-1Successes/
attempts
xTBloom CUDA default pol-
icy, B=256 (AO-sorted)28.792 [28.730–28.793] 28.859 71.130 6144/6144
xTBloom CUDA ablation,
B=256 (AO sorting disabled)38.094 [38.034–38.171] 38.160 53.760 6144/6144
xTBloom CUDA, B=64 (AO
sorting disabled)74.133 [73.667–74.271] 74.204 27.630 6144/6144
Continued on next page
S-18

Configuration Primary time, median
[minimum–maximum] /
sMedian wall
clock / sThroughput /
configurations
s-1Successes/
attempts
xTB CPU, single core 82.493 [82.374–85.488] 82.756 24.830 6144/6144
xTBloom CPU, single core 83.802 [83.776–83.891] 84.006 24.440 6144/6144
xTBloom CUDA, B=1 (AO
sorting disabled)293.822 [293.637–294.759] 294.067 6.970 6144/6144
dxtb GPU batch 64 322.294 [321.827–322.703] 327.654 6.350 6144/6144
dxtb GPU batch 128 403.220 [394.534–405.795] 406.697 5.080 6144/6144
dxtb GPU batch 1 813.846 [810.560–826.274] 1198.143 2.520 6144/6144
dxtb GPU batch 256, original
orderOOM 141.580 — 0/6144
dxtb GPU batch 256, AO-
sortedOOM 188.357 — 768/6144
Successes/attempts use the 6,144 target records from three measured traversals as the
denominator. For the two OOM configurations, the wall-clock column records time spent on
the failed attempts, and full-queue throughput is left blank. The other nine configurations
completed all 2,048 structures in every measured traversal.
With the same fixed inputs and software implementation, the B = 256 primary times
were 28.792 s with default AO sorting and 38.094 s with sorting disabled. The default policy
reduced evaluation time by 24.4% relative to this ablation. The single-core xTBloom CPU
primary time was 83.802 s, giving a throughput ratio of 2.91 for the default AO-sorted GPU
configuration.
Acrossall20traversalsofthefivexTBloomconfigurations, themaximumdifferencesfrom
the first measured CPU output were 9.521×10-13Ehin energy and 4.179×10-13Ehbohr-1
in any force component. Per-configuration statuses, convergence flags and SCC iteration
counts were identical.
S4.4 Sensitivity to external-reference settings
We used the same cross-program absolute thresholds as S1, comprising 5×10-7Ehfor energy
and 5×10-6Ehbohr-1for forces. Relative to the first measured native xTB output at ac-
curacy = 1.0, each traversal of the five xTBloom configurations jointly passed 1,382/2,048
configurations. The complete dxtb configurations passed 0/2,048. The accuracy=10-4ref-
erence was used for sensitivity diagnostics. Throughputs in Table S4b correspond to the
specified program settings and do not constitute an equal-accuracy comparison.
We additionally evaluated the same inputs once on the CPU with the same native xTB li-
brary at accuracy = 10-4. The maximum iteration count, electronic temperature and FRESH
settings were unchanged. These results provide the numerical comparison in Table S4c. Ta-
ble S4b retains the original performance measurements at accuracy = 1.0. Joint-pass counts
use all 2,048 target configurations as the denominator, with successful-pair counts reported
separately.
S-19

Table S4c. Numerical comparison against the native xTB reference at accuracy = 10-4.
Configuration Successful
pairsEnergy
exceed-
ancesForce
exceed-
ancesJoint
passesMaximum
|∆E| / E hMaximum |∆F| /
Ehbohr-1
Five xTBloom con-
figurations2048/2048 384 0 1664/2048 2.116×10-54.725×10-6
xTB CPU accuracy
1.02048/2048 0 369 1679/2048 2.089×10-82.240×10-5
dxtb batch
1/64/1282048/2048 2048 1 0/2048 3.043×10-61.387×10-5
dxtb batch 256,
AO-sorted256/2048 256 0 0/2048 2.638×10-63.928×10-6
dxtb batch 256,
original order0/2048 — — 0/2048 No paired output No paired output
Against the stricter reference, all xTBloom force components met the comparison thresh-
old, while 384 configurations exceeded the energy threshold. Comparing the two native xTB
accuracy settings gave 369 configurations exceeding the force threshold. The complete dxtb
configurationshad2,048energyexceedancesand1forceexceedance, withamaximumenergy
difference of 3.043×10-6Eh. Pass counts were calculated per configuration using joint energy
and force thresholds. S4.2 defines the stopping criteria and residuals for each program. The
timing ratios in Table S4b correspond to these numerical settings.
S5. AO sorting, SCC scheduling and supplementary
variable-size experiments
S5.1 Protocols for the two ablation studies
The AO-sorting comparison used the 2,048 near-size OMol25 configurations from the main
performance benchmark. Inputs, neutral-singlet constraints, software implementation, elec-
tronic temperature, SCC tolerances and timing boundaries were fixed. The complete default
AO-sorted scheme at B=256 served as the reference. Sorting alone was disabled for the
paired ablation. Additional measurements with sorting disabled used B=1, 64 and 256.
The benchmark runner sorted the queue before the calls. Each configuration completed one
full-queue warmup and three measured traversals. All outputs were aligned by original con-
figuration identity to compare energies, complete force blocks, statuses, convergence flags
and SCC iteration counts.
The SCC-scheduling comparison used small-AO, neutral, restricted-spin batches formed
by cycling through H 2, He, LiH and CH 2. The maximum SCC iteration count was 8, the
mixing-history length was 3 and the electronic-temperature parameter was 0. D4, explicit
point charges and periodic embedding were disabled in this controlled SCC loop. An active-
capacity scheduling chain and a fixed-capacity device-tail Graph were constructed from the
same input and initial state. Selected systems had their initial iteration counters set to
the limit, causing them to exit at the first active-state check and producing initial active
S-20

fractions of 1, 1/2 and 1/4. Membership was selected using the system index modulo 1, 2 or
4. At B=1, only the single fully active configuration was retained.
The same initial state was restored before every measurement. Each path had three
warmups and 50 measured runs. CUDA events on the same stream measured the complete
SCC loop until all members terminated. Final-state download occurred after the timed
interval. Paired checks compared the number of numerical updates, free energy, eigenval-
ues, occupations, density, weighted density, populations, multipoles, iteration counters and
statuses. Initially fully active configurations were also checked against sequential CPU cal-
culations.
Table S5a. Controlled-comparison protocols for AO sorting and active SCC scheduling.
Item AO sorting and batch-size
scanActive SCC scheduling
Workload 2,048 OMol25 configurations,
AO100–119Batches cycling through
H2/He/LiH/CH 2
Batch size 1, 64, 256, with the sorting pair at
B=2561, 8, 32, 128
Comparison variable AO-sorting switch or original-
order batch sizeFixed-capacity Graph and active-
capacity scheduling chain
Initial electronic state FRESH Identical initial state and preset
iteration counts
Initial active fraction Natural SCC evolution 1, 1/2, 1/4
Warmup/measured runs Full queue, 1/3 Each path, 3/50
Timing statistic Median and minimum–maximum Mean
Timing boundary Preparation, synchronous evalua-
tion and output publication, with
sorting outside timingCUDA events enclose the com-
plete SCC loop. Preparation,
download and force evaluation
are outside the interval.
Resources One EPYC9354 physical core and
one RTX5090EPYC7K62 and one RTX5090
Correctness checks Per-configuration energy, forces,
status and iterationsComplete final states and
numerical-update counts for both
paths, with an additional CPU
reference for fully active batches
S5.2 Variable-size batches and context reuse
The variable-size experiment used the final 256 configurations of the mixed OMol25 queue
in its original order. All of these configurations completed evaluation in the CPU and
GPU configurations used here. Calculations used host input/output, a 300 K electronic
temperature, up to 500 SCC iterations and FRESH initial states. The strict configuration
set the SCC state-residual RMS tolerance to 10-6and the free-energy-change tolerance to 10-8
S-21

Eh. Each execution configuration completed one full warmup and three measured runs. End-
to-end timing included input preparation, synchronous evaluation and output publication.
Table S5b. Supplementary timing for the final healthy batch of 256 variable-size configura-
tions.
Execution configuration Tolerance setting Median end-to-end time
/ sCompleted per traver-
sal
CPU, per-system Strict 8.964 256/256
GPU, original order,
B=256Strict 29.525 256/256
GPU, original order, B=64 Strict 30.027 256/256
GPU B=64, sorted by prior
iteration countStrict 24.944 256/256
GPU B=64, sorted by AO
countStrict 26.067 256/256
GPU, original order,
B=256Loose 22.762 256/256
The loose configuration used state-residual RMS and free-energy-change tolerances of
10-4and 10-6Eh, respectively. Sorting by prior iteration count used iteration counts from
preceding per-system GPU evaluations as a retrospective grouping comparison. Both AO
sorting and iteration-count sorting were performed outside timing. Relative to original-order
B=64, sortingbyprioriterationcountandbyAOcountreducedsynchronousevaluationtime
by approximately 16.9% and 13.2%, respectively, using unrounded values.
The strict original-order B=256 batch contained 100 exact-AO solver groups and required
up to 95 SCC iterations. The general masked path retained each group’s solver capacity and
continued until the final active member terminated. Equal-dimension work was distributed
amongmanysmallgroups, whileSCCconvergencealsohadalongtail. Bothfeaturesaffected
execution organization for this batch.
A separate comparison fixed the same healthy final batch and four original-order B=64
batches, changing only context ownership. With one shared context across the four batches,
median end-to-end and synchronous evaluation times were 30.052 and 29.961 s, respectively.
With a separate retained context per batch, they were 29.013 and 28.928 s. The two config-
urations produced identical energies, a maximum force-component difference of 1.390×10-16
Eh/bohr and identical SCC iteration counts.
S5.3 AO, capacity and active-fraction scans for active-capacity
scheduling
To examine the operating range of active-capacity scheduling, we compared the existing
device scheduling chain with a fixed-capacity device-tail Graph on an RTX 5090. Let n
denote the AO count, B the original capacity of a same-AO-dimension solver bucket, and A
the number of currently active systems. The active fraction is A/B. Synthetic component
S-22

probes were used for 40, 41 and 180 AOs. C 10H22and C20H42supplied the 62- and 122-AO
probes, respectively.
Both paths restored the same FRESH checkpoint and termination mask. Preset iteration
counters controlled active membership, and at most one SCC iteration was executed. CUDA
events measured that iteration’s graph-execution latency, including each path’s organization,
dispatch and solver work. Initialization, final-state readback and comparison were performed
separately. ThebuildusedGCC13.3.0, CUDA12.9.86, Releasemodeandsm120, withdriver
580.95.05.
The single-iteration masked-dispatch comparison measures one SCC graph execution
from the same checkpoint.
The initial scan contained 81 coordinates. The five AO probes used A=128, 96, 64,
32, 16, 8 and 0 at B=128. The 41-, 122- and 180-AO probes also used A=B, B/4 and
max(1,B/16) at B=8, 32, 64 and 256. All five probes additionally tested A=1 and 0 at B=1.
A supplementary scan then added 16 points for 122/180 AOs and B=128/256 at A=8, 4, 2
and 1. The 8 points with B=256 were independently rechecked. Together, the stages covered
95 unique coordinates and 105 stage-specific observations.
One warmup pair was excluded at each point. The initial and supplementary scans
used four timing pairs per point, and the independent recheck used six. The paths were
interleaved in AB/BA order with a fixed random seed of 515. Statistics were calculated
separately for each stage. Relative cost was represented by R, the median of the paired
scheduling-chain/fixed-capacity latency ratios. The prespecified candidate-benefit margin
required R≤0.900 and every timing-pair ratio below 1. The initial scan used one 5090, while
the supplementary scan and independent recheck used another GPU of the same model.
Table S5c. Single-iteration SCC component latency at 122 AOs and B=128. Scheduling-
chain and fixed-capacity columns report their respective median latencies. R is the median
paired latency ratio, with R>1 indicating a slower scheduling chain. A=0 is the zero-active
control.
Active systems
AActive fraction /
%Scheduling chain /
msFixed capacity / ms R
128 100 126.787 14.016 9.045
96 75 97.872 13.164 7.434
64 50 68.734 11.984 5.735
32 25 39.537 11.013 3.589
16 12.500 24.933 10.356 2.408
8 6.250 17.842 10.276 1.736
0 0 0.012 0.012 1.000
All 71 nonzero-active coordinates in the initial scan had R greater than 1.020. There
were also 10 zero-active controls.
S-23

Table S5d. Paired latency ratio R at low active counts. Four pairs were used in the supple-
mentary scan and six in the independent recheck. Column headings give the active count
A.
AO / B Timing pairs A=8 A=4 A=2 A=1
122 / 128 4 1.728 1.403 1.218 1.161
122 / 256 4 1.147 0.948 0.836 0.805
122 / 256 6 1.149 0.946 0.838 0.805
180 / 128 4 1.721 1.357 1.149 1.011
180 / 256 4 1.164 0.933 0.812 0.730
180 / 256 6 1.164 0.932 0.813 0.730
For 122/180 AOs and B=128, the supplementary scan did not meet the candidate-
benefit margin even at A=1. At B=256 and A=4, the recheck showed latency reduc-
tions of approximately 5.400%–6.800%. The prespecified margin was reached at A=1 or
2, with median recheck ratios of 0.730–0.838. These points corresponded to active fractions
of 0.391%–0.781% and component-latency reductions of approximately 16.200%–27.000%.
The benefit region was concentrated at very small active sets. This supports the current
restriction of active-capacity scheduling to validated small-AO layouts, with fixed-capacity
execution retained for other layouts.
S5.4 Natural-size and padded AO matrix layouts
To isolate the effect of the principal AO matrix layout, we compared A, the ragged natural-
size n×n layout, with B, the padded-natural layout stored as m×m with a uniform stride,
where m is the largest AO count in the batch. Both used the same experimental inputs
and software implementation. Padding covered overlap and Hamiltonian matrices, dipole
and quadrupole integral planes, orbital coefficients, density and energy-weighted density,
and associated workspaces and analytic-force routines that consume these matrices. Re-
quired packing and retrieval were included in call cost. Both variants retained physical loops
and divide-and-conquer eigensolution at the true dimension n and used the same exact-AO
grouping.
The near-size queue reused the 2,048 AO100–119 configurations. The variable-size queue
retained all 1,317 configurations with AO8–180 that satisfied the input criteria of neu-
trality, singlet multiplicity, no unpaired electrons, light main-group elements, no metal or
heavy-element flags and AO≤180. Selection did not use convergence or timing results. In
the sorted condition, inputs were ordered by (AO, configuration identifier) and divided
into B=256 batches. The near-size queue formed 8 batches, and the variable-size queue
formed 5×256+37. The strongly mixed condition reused the same variable-size set. After
AO/identifier sorting, it was split into six contiguous quantile groups of sizes 220, 220, 220,
219, 219 and 219. Round-robin interleaving across the groups was followed by batching as
5×256+37 and sorting only within each batch. Each configuration was used once. The
sums of exact-AO group counts over batches were 27, 164 and 192 for the near-size, sorted
variable-size and strongly mixed conditions, respectively. These are static group counts, with
S-24

dynamic GPU occupancy remaining a separate quantity.
All A/B configurations used double-precision GFN2-xTB, host input/output, strict
FRESH initial states, a 300 K electronic temperature and up to 500 SCC iterations. SCC
state-residual RMS and free-energy-change tolerances were 10-6and 10-8Eh, respectively.
CPU and BLAS execution were limited to one thread on a single physical core, with one
RTX 5090. The build used CUDA 12.9.86 and Release mode. Each mode completed one full
warmup and three measured traversals. The strongly mixed condition used measured orders
BA, AB and BA. Primary time included per-batch preparation, synchronous complete
energy-and-force evaluation and host-result publication. File reading, sorting, initial context
creation and result checks were excluded. Memory was measured in separate processes,
with one traversal per configuration and NVML sampling of the target process every 20 ms,
independently of the measured timing runs.
Table S5e. Principal AO matrix layouts within the same software implementation. A uses
natural-size storage, and B uses padded storage with true-dimension solves. All reductions
use B as the denominator. Times are medians and minimum–maximum ranges over three
measured traversals. Sampled peaks come from separate memory traversals. Capacity is the
queue-wide sum of storage-element counts for one principal AO matrix per batch, a quantity
distinct from peak GPU memory and FLOPs.
Queue (configu-
rations, AO)Median E+F time [min-
imum, maximum] / sA / B sam-
pled peak /
MiBPrincipal-
matrix
capacity
reduction
/ %Sampled-
peak re-
duction /
%Time
reduction
/ %A, B suc-
cesses/
attempts
per
traversal
Near-size, AO-
sorted (2,048,
100–119)A 29.339 [29.249, 30.197], B
29.395 [29.360, 29.574]7288 / 7364 2.16 1.03 0.19 2048/2048
Variable-size, AO-
sorted (1,317,
8–180)A 140.847 [140.843,
140.870], B 141.389
[141.371, 141.672]5324 / 7702 34.36 30.88 0.38 1314/1317
Variable-size,
strongly mixed
(1,317, 8–180)A 82.827 [82.737, 83.006], B
86.296 [86.251, 86.543]3802 / 11572 75.90 67.14 4.02 1314/1317
For the near-size, AO-sorted queue (2,048, 100–119), the three A times were 30.197343,
29.338962 and 29.249334 s, and the B times were 29.360499, 29.394766 and 29.573740 s.
Total element counts for one principal matrix were 25,136,660/25,691,392 for A/B.
For the variable-size, AO-sorted queue (1,317, 8–180), the three A times were 140.842652,
140.846581 and 140.869701 s, and the B times were 141.371168, 141.388672 and 141.671975
s. Total element counts for one principal matrix were 7,177,368/10,934,480 for A/B.
For the variable-size, strongly mixed queue (1,317, 8–180), the three A times were
83.005766, 82.737267 and 82.827109 s, and the B times were 86.543091, 86.296201 and
86.250849 s. Total element counts for one principal matrix were 7,177,368/29,775,568 for
A/B.
A was faster than B in all three strongly mixed runs. The median time differences were
S-25

below 1% for both AO-sorted queues. Sorted and strongly mixed conditions used the same
RTX5090modelanddriver580.95.05. A/Bcomparisonswithineachconditionwerematched
on the same physical GPU.
Paired thresholds required maximum absolute differences of at most 10-10Eh for
successful-system energies, 10-10Eh/bohr for forces and 10-10e for precheck charges. System
identifiers, atom offsets, statuses, convergence flags and SCC iteration counts also had
to match. A/B prechecks and measured-output verification passed. Measured traversals
requested energy and forces, with charges checked in the precheck. For the complete strongly
mixed queue, precheck energy and charge differences were zero, and the maximum force
difference was 2.22×10-16Eh/bohr. Across batching arrangements, statuses, convergence
flags and iteration counts were unchanged. Maximum energy and force differences were
5.26×10-13Eh and 4.45×10-13Eh/bohr, respectively.
Every near-size traversal succeeded on 2,048/2,048 configurations. Both variable-size
batching arrangements succeeded on 1,314/1,317. The same three configurations,CO_72702
61446849249421858501,CO_9343753771713800272817479andCO_9847448562207090849
131680, remained SCC-nonconverged after 500 iterations, with all requested floating-point
output slices filled with NaN. Failed systems were retained in timing and in the attempted-
throughput denominator.
A separate precheck of C, padded-unified, also expanded the solver dimension to m
after padding storage. In the 37-configuration variable-size diagnostic, its maximum charge
difference was 1.2638×10-10e, exceeding the prespecified threshold. C therefore did not enter
measured timing or memory comparisons.
Two complete batches of 256 systems each passed CUDA memory checking. CUDA
synchronization checking stopped after producing 512 diagnostics for the natural-size layout.
Synchronization checks remain incomplete for the two padded-storage variants using natural
or expanded solver dimensions.
Timings in this section are reported separately from the main performance timings in S4.
S6. Repeated evaluation performance for conformers with
a common topology
S6.1 Conformer generation and comparison protocol
One alkane composition and topology were used at each molecular size. The first conformer
in each batch retained the ideal geometry. Independent Gaussian perturbations with a
standard deviation of 0.020 bohr were applied to every Cartesian coordinate of the remaining
conformers. Therandomseedwastheatomcount×1000+B.Allconformerswereneutraland
used a restricted spin treatment. GFN2-xTB energies and forces were evaluated in double
precision. Because the seed included B, the conformer lists could differ between batch sizes.
Tests ran on an AMD EPYC 9354 host with an NVIDIA RTX 5090. The host process
of each engine was pinned to one physical CPU core. The xTBloom CPU, OMP, and BLAS
thread counts were all 1. The PyTorch intra-op and inter-op thread counts for dxtb were
also 1. The xTB and tblite adapters called their public single-system interfaces sequentially.
Contexts and descriptors were prepared before timing, and the complete batch of energies
S-26

and forces was available on the host when timing ended. Each configuration received one
warmup and three measured runs. A single-threaded tblite calculation provided the numer-
ical reference for each panel.
Table S6a. Initialization and convergence settings for the common-topology conformer
benchmark.
Item Setting
Cold-electronic-state panels B=1 and B=512. xTBloom performs FRESH initialization inside the call.
xTB/tblite recreate the calculation object outside the measured interval. The
dxtb call includes reset.
Auto-warm panel B=128. After an untimed cold seed, xTBloom uses WARM, xTB/tblite reuse
persistent objects, and dxtb uses reset/cold.
xTBloom SCC State-residual RMS tolerance of 10-4, free-energy-change tolerance of 10-6Eh,
and a maximum of 500 iterations
xTB and tblite accuracy=1.0 and a maximum of 500 iterations
dxtbx_atol=10-4,x_atol_max=10-5,f_atol=10-4,force_convergenceenabled,
and a maximum of 500 iterations
Output comparison Against the tblite reference in the same panel, maximum energy
difference≤2×10-3Eh and maximum force-component difference≤2×10-3
Eh/bohr
Repeatability check Applies only to cold/reset calls, with tolerances of 10-10Eh for energy and 10-8
Eh/bohr for force components
Reported statistics Median and minimum–maximum range of three measured runs
Table S6b. Molecular compositions and AO dimensions in the common-topology conformer
benchmark.
Atom count Composition GFN2-xTB AO count
14 C 4H10 26
32 C 10H22 62
62 C 20H42 122
122 C 40H82 242
242 C 80H162 482
362 C 120H242 722
S-27

S6.2 Complete size-scaling curves and numerical checks
14 32 62 122 242 362
Atoms per system1101001k10k100k1MTime per batch (ms)
a   Batch = 1
14 32 62 122
Atoms per system1101001k10k100k1MTime per batch (ms)
b   Batch = 128
14 32 62 122
Atoms per system1101001k10k100k1MTime per batch (ms)
c   Batch = 512
EPYC 9354 · one CPU core per run · RTX 5090
Median of three measurements, whiskers show min–maxxtbloom CPU
xtbloom CUDAxTB
tblitedxtb CPU
dxtb CUDA
Figure S4. Energy-and-force evaluation latency for locally perturbed conformers with a common topology
under a single-physical-CPU-core constraint. (a) B=1 with a cold electronic state for systems containing 14,
32, 62, 122, 242, and 362 atoms. (b) B=128 with auto-warm initialization. (c) B=512 with a cold electronic
state. The latter two panels use 14, 32, 62, and 122 atoms. The GPU is an RTX 5090. Each execution path
uses one physical CPU core, with OMP, BLAS, and PyTorch thread counts of 1. The vertical axis gives the
latency to evaluate the complete batch. Error bars show the minimum–maximum range of three measured
runs. Gaps or crosses indicate unavailable configurations or GPU out-of-memory failures.
S-28

Across the common-topology tests, the maximum energy and force-component differences
of xTBloom CPU and CUDA from the same-panel tblite reference are 4.571×10-7Eh and
2.932×10-5Eh/bohr. All timed configurations returned finite energies and forces. Results
from xTBloom, native xTB, and dxtb passed the same-panel tblite reference checks. GPU
out-of-memory failures prevented completion for dxtb-cuda at B=512 with 62 atoms, dxtb-
cuda at B=512 with 122 atoms, and dxtb-cuda at B=128 with 122 atoms.
The xTBloom measurements use a Release build of version ab3a3bc6132c, CUDA toolkit
12.9.1, and target architecture sm120. The remaining versions are native xTB 6.7.1, tblite
0.7.0, LP64 OpenBLAS 0.3.30, dxtb 0.4.0, PyTorch 2.10.0+cu128, and tad-libcint 0.3.0.
S7. QM/MM system, biases, and dynamics parameters
We used the ethylene phosphate model reaction with a methoxy leaving group fromS8. The
periodic system contains 8,938 real atoms. Its 16-atom QM region comprises two complete
molecules with composition P 1O5C3H7. The environment contains 2,974 TIP4P-Ew water
moleculesS9. Including the virtual charge sites of the water model gives 11,912 electrostatic
sites. The QM region has a total charge of−2e and uses closed-shell GFN2-xTB without link
atoms. The bare QM/MM and DPRc-corrected sampling use the same classical environment
and geometric restraints.
The reaction coordinate isξ=r(P,O5′)−r(P,O2′)=d(1,12)−d(5,1), where 1=P, 5=O2′is
the attacking oxygen, and 12=O5′is the leaving oxygen. The attack angle is∠(5,1,12).
LAMMPSS10propagates the dynamics with a Langevin thermostatS11, and SHAKES12con-
strains the water geometry. ColvarsS13applies the harmonic biases for umbrella sam-
plingS14,S15. At each step, a coordinator collects inputs from 48 windows for cross-window
QM/MM batching on one shared GPU. The batched work covers GFN2-xTB evaluation,
classical nonbonded interactions, periodic electrostatics using the particle–particle particle–
mesh (PPPM) method with fast Fourier transforms (FFTs), and DeePMD/DPRc inference
when enabled. Each window retains its own coordinates, integrator, and SCC history. Peri-
odic electrostatic preprocessing, postprocessing, and force assembly follow Section 3.4 of the
main text.
Table S7. System and dynamics parameters for QM/MM umbrella sampling.
Parameter Setting
QM region and environment 16-atom P 1O5C3H7, charge−2e, closed shell, and 2,974 TIP4P-Ew water
molecules
Window centers−3.100 to 1.600 Å in increments of 0.100 Å, giving 48 windows
Harmonic force constant for
the reaction coordinate200 kcal mol-1Å-2
Attack-angle center and har-
monic force constant180◦/0.050 kcal mol-1degree-2
Equilibration per window 100 ps
Continued on next page
S-29

Parameter Setting
Production per window Three independent random-number streams, each with 250 ps
Total production sampling per
simulation set36 ns
Time step 1 fs
Thermostat Langevin thermostat at 298 K with damping time 200 fs
Water constraints SHAKE with tolerance 10-6and a maximum of 200 iterations
Lennard–Jones cutoff 9 Å
Real-space electrostatic cutoff 9 Å
Periodic electrostatics Fourth-order TIP4P/PPPM, a 50×50×50 grid, and target accuracy 10-6
Ewald splitting parameter 3.488×10-1Å-1
GFN2-xTB SCC control accuracy=0.001 and a maximum of 250 iterations
Electronic temperature 300 K, with kBT = 9.500×10-4Ehfor electronic occupation broadening
Periodic QM response parame-
terskmax=(8,8,8), ksqmax=100
Recording interval for the reac-
tion coordinate and angle25 fs
Production GPU One NVIDIA RTX PRO6000 Blackwell
The 300 K value in the table sets the numerical broadening of electronic occupations,
with kBT = 9.500×10-4Eh. The MD thermostat and WHAM analysis both use 298 K.
S7.1 Single-GPU short-run cross-window timing protocol
Item Setting
GPU and CPU configuration One NVIDIA RTX PRO 6000 Blackwell 96 GB and an AMD EPYC 9J14
with 48 physical cores allocated
Windows and parallel configu-
rationEvery integer B=1–48, with one MPI process and one thread per window.
The remaining allocated cores are idle at small B.
Warmup and timing segments 25 warmup steps followed by five consecutive 100-step segments from the
same trajectory
Synchronized segment duration Maximum LAMMPS loop time across windows within each segment
Timing boundary Includes force evaluation, synchronization, neighbor calculations, and
integration. Loading, initialization, and final file output are outside the
measured interval.
DPRc model used for timing DPRc model from before retraining
LettB
w,rdenotethelooptimeofwindowwintherth100-stepsegmentwhenBwindowsare
batched, and lett1
w,rdenote the corresponding duration when that window runs sequentially.
Segment-level aggregate throughput and paired speedup are defined as
S-30

RB,r=100B
max w∈BtB
w,r,S B=P
w∈BP5
r=1t1
w,rP5
r=1max w∈BtB
w,r.(S4)
Throughput is reported as the median of the five consecutive segments. Shaded regions
show the minimum–maximum range. The numerator of the speedup uses the total sequential
execution time for the same group of windows.
The GFN2-xTB settings in Table S7, accuracy=0.001 and a maximum of 250 iterations,
apply to the QM/MM production runs.
The PBE0 SCF, geometry-pairing, and label-assembly checks passed for all 6,560 frames.
Themaximumhigh-level/low-levelgeometry-pairingresidualwas5.69×10-14Å,andthemax-
imum classical-baseline energy residual was 5.83×10-11kcal mol-1. Each of the four models
wastrainedfor100,000steps. All144window–trialproductiontrajectorieswiththeretrained
DPRc model reached their target step counts. The 10 paired checks, consistency between
single-window and 48-window evaluation, and short NVE stability tests of this production
model were recorded as passes. The maximum force-component difference between single-
window and 48-window evaluation was 4.4×10-11kcal mol-1Å-1. The production model
passed 5 ps NVE checks in three anchor windows both before and after equilibration.
In one 5 ps NVE check in the short-run timing environment, the net energy drift of one
window exceeded the original threshold. The retrained production model passed the 5 ps
NVE checks in three anchor windows before and after equilibration.
Table S7a. Advancement rates during DPRc production sampling. Each trial uses one
RTX PRO 6000 and contains 48 windows with 250,000 steps per window. The aggregate
advancement rate is the total window step count divided by the sum of segment wall times
for that trial. The mean rate per window is this aggregate rate divided by 48.
Production
recordTotal steps
across win-
dowsSegment
countSum of segment
wall times / sAggregate MD
steps s-1GPU-1Mean MD steps s-1
per window
trial 0 12000000 50 90849.709 132.09 2.752
trial 1 12000000 50 91545.798 131.08 2.731
trial 2 12000000 50 90069.025 133.23 2.776
These advancement rates use the sum of wall times for 50 production segments, with
intersegment scheduling waits outside the timing boundary. The short-run tests use syn-
chronized LAMMPS loop time.
The short-run timing tests use a 4 Å neighbor-list skin and check displacements every 10
steps. Both xTBloom evaluation and DeePMD inference use batch size B.
S-31

Table S7b. Complete batch-size scan for single-GPU short-run cross-window timing. Aggre-
gate throughput is in MD steps s-1GPU-1. Entries give the median [minimum, maximum]
of five consecutive segments. Paired speedup is calculated using the equation in this section.
The DPRc timing uses the model from before retraining.
B QM/MM aggregate through-
putQM/MM
paired speedupQM/MM+DPRc aggregate
throughputQM/
MM+DPRc
paired speedup
1 12.183 [12.048, 12.292] 1.000 10.571 [10.299, 10.608] 1.000
2 23.997 [23.862, 24.244] 1.970 20.381 [13.027, 20.911] 1.876
3 34.038 [33.819, 34.387] 2.806 29.150 [18.424, 29.277] 2.719
4 43.305 [43.058, 43.730] 3.560 37.010 [23.873, 37.614] 3.401
5 52.157 [51.654, 52.841] 4.281 45.021 [28.863, 45.647] 4.160
6 60.621 [60.386, 61.358] 4.988 51.607 [34.113, 52.579] 4.892
7 67.210 [66.844, 68.953] 5.539 55.851 [37.788, 57.623] 5.269
8 75.093 [74.649, 76.243] 6.174 64.389 [43.502, 66.371] 5.982
9 82.998 [82.179, 84.256] 6.807 67.678 [46.490, 68.517] 6.238
10 89.565 [88.635, 90.961] 7.356 71.998 [50.580, 73.071] 6.665
11 95.064 [93.782, 96.277] 7.798 75.641 [54.537, 78.168] 7.077
12 101.079 [67.608, 102.102] 7.653 80.968 [46.825, 82.381] 7.146
13 108.391 [72.455, 108.905] 8.169 85.837 [49.890, 87.218] 7.532
14 114.017 [77.002, 114.433] 8.620 89.271 [53.072, 92.443] 7.907
15 117.464 [81.147, 117.807] 8.915 93.275 [56.227, 96.900] 8.333
16 122.991 [84.973, 123.261] 9.422 98.032 [59.394, 101.621] 8.804
17 125.771 [89.522, 127.907] 9.173 101.591 [62.363, 104.711] 9.185
18 132.579 [92.795, 133.479] 9.661 105.602 [65.587, 109.200] 9.564
19 138.529 [97.403, 139.150] 10.070 108.702 [68.269, 112.963] 9.853
20 142.453 [100.732, 142.959] 10.388 112.464 [59.519, 115.541] 9.689
21 147.242 [104.767, 147.708] 10.717 117.458 [62.242, 119.834] 10.090
22 150.704 [105.505, 150.949] 10.951 118.731 [64.045, 122.079] 10.248
23 154.822 [109.734, 155.109] 11.283 123.184 [65.765, 126.706] 10.642
24 159.284 [113.501, 159.497] 11.624 124.698 [67.768, 129.198] 10.458
25 153.281 [111.950, 154.230] 11.318 126.875 [69.699, 130.856] 10.649
26 153.167 [114.756, 156.379] 11.469 130.584 [71.986, 134.115] 10.949
27 157.104 [117.556, 159.974] 11.751 132.966 [74.098, 136.531] 11.171
28 161.153 [120.972, 163.582] 12.053 136.103 [76.413, 139.694] 11.445
29 162.848 [123.833, 165.343] 12.229 139.142 [78.173, 142.844] 11.676
30 165.992 [123.638, 168.680] 12.413 141.670 [80.105, 145.492] 11.948
31 170.119 [126.285, 172.534] 12.688 142.956 [81.966, 145.893] 12.086
32 172.806 [128.745, 175.273] 12.901 146.179 [84.405, 149.272] 12.375
33 174.873 [131.162, 177.361] 13.084 149.865 [86.345, 152.841] 12.643
34 177.227 [133.527, 179.790] 13.281 126.579 [87.833, 153.037] 12.316
Continued on next page
S-32

B QM/MM aggregate through-
putQM/MM
paired speedupQM/MM+DPRc aggregate
throughputQM/
MM+DPRc
paired speedup
35 180.864 [136.632, 183.275] 13.566 129.665 [90.005, 156.887] 12.603
36 182.625 [118.141, 185.017] 13.108 130.673 [91.488, 157.268] 12.738
37 184.786 [120.227, 187.185] 13.272 133.156 [93.925, 161.398] 13.026
38 185.876 [121.878, 188.015] 13.373 134.166 [95.750, 162.328] 13.217
39 189.775 [124.553, 192.246] 13.650 137.510 [97.661, 166.173] 13.476
40 154.702 [125.988, 194.068] 13.313 138.839 [99.013, 167.842] 13.653
41 157.778 [128.780, 197.451] 13.572 139.896 [100.736, 169.479] 13.836
42 159.953 [130.904, 199.580] 13.743 142.032 [102.184, 170.467] 13.962
43 159.975 [127.845, 197.931] 13.761 142.889 [93.438, 174.348] 13.798
44 163.146 [130.696, 202.752] 14.050 144.289 [94.715, 175.541] 13.954
45 168.375 [132.997, 204.549] 14.290 143.800 [96.588, 176.887] 13.670
46 170.021 [134.647, 205.135] 14.458 143.914 [97.246, 175.225] 13.297
47 170.046 [135.396, 205.298] 14.473 146.851 [99.135, 178.868] 13.542
48 173.348 [137.977, 208.696] 14.725 148.062 [100.436, 180.092] 13.649
S8. DPRc labels, window-level splits, and model diagnos-
tics
DPRc learns the differences between high-level and low-level QM/MM energies and forces for
complete periodic configurationsS16,S17. The high-level calculations use PBE0/6-31G*S18,S19
in AmberTools with QUICKS20,S21, and the low-level calculations use the xTBloom baseline
matched to production simulations. Differences are evaluated for each complete periodic
configuration before constructing local inputs. The QM region and complete water molecules
within 6 Å are then unwrapped using the minimum-image convention. DPA4C represents
the local environment, and DeePMD-kit evaluates the correction potential and its force
correctionS22–S25.
The dataset contains 6,560 periodic configurations and is partitioned by complete um-
brella windows. The negative-coordinate subset contains 1,560 frames, with window centers
from−3.100 to−1.700 Å and 104 frames per window. The windows centered at−2.700
and−2.400 Å are assigned to validation and testing, respectively. Four models with random
seeds 20260829, 20260830, 20260831, and 20260901 are trained from scratch for 100,000 steps
each. Production trajectories use the final model with seed 20260829, selected in advance.
Labels use binary64, and the descriptor and fitting network use FP32.
S-33

Table S8. DPRc label and model-training configuration.
Item Setting
Number of periodic labels 6,560
High-level method PBE0/6-31G* QM/MM, AmberTools/QUICK
Low-level method xTBloom GFN2-xTB QM/MM matched to production simulations
Order of differencing and local-
izationEvaluate differences for the complete periodic configuration, then retain
the QM region and complete water molecules within 6 Å
Actual reaction-coordinate
range of the 15-window
negative-coordinate subset[−3.073,−1.541] Å for 1,560 frames
Partitioning unit Complete umbrella windows
Training/validation/test frame
counts5,519/520/521
Number of independent models 4
Training steps per model 100,000
Production model Final primary model with seed 20260829, selected in advance
Held-out committee diagnostic 521 held-out test configurations, comprising 104 frames from the
negative-coordinate window subset and 417 other frames
The four-model force-vector RMS spread is computed for each QM atom and summa-
rized over 8,336 QM atoms in 521 held-out configurations. Its mean, P95, and maximum
are 0.026063, 0.046128, and 0.116216 eV/Å, respectively. This statistic describes model dis-
agreement on the held-out set. Label errors of the primary production model are reported
in Table S8a.
Training uses a DPA4C descriptor with a Bessel basis, channels=32, lmax=2,n_ra
dial=16,radial_modes=4, and rcut=6 Å. The SiLU fitting network has dimensions
192×192×192andresnet_dt=false. ElementtypesareorderedasP,O,C,H,OW,andHW.
OWandHWdenotewateroxygenandwaterhydrogen. MM–MMpairsareexcludedfromthe
descriptor and model pair calculations, and the isolated-atom energies of water oxygen and
hydrogen are fixed at zero. The descriptor and fitting network use FP32, withuse_amp=false
andenable_compile=false. The optimizer is HybridMuon, withweight_decay=0.001 and
gradient_max_norm=5. The learning rate follows a WSD schedule from 0.001 to 10-6,
withwarmup_ratio=0.003,warmup_start_factor=0.2,decay_type=inverse_linear,
anddecay_phase_ratio=0.2. The settings aretraining_data.batch_size=1,vali
dation_data.batch_size=1, andvalidation_data.numb_btch=16. The loss is MAE,
both energy and force prefactors are 20, the virial prefactor is 0,f_use_norm=true, and
force_loss_use_model_mask=true. Table S8a reports energy, QM-force, and environment-
force errors for the original held-out domain of 417 frames and the additional negative-
coordinate held-out window of 104 frames.
Energy MAE is the mean absolute error in the total correction energy over held-out
configurations. Force MAE is the mean absolute error in Cartesian force components of
the relevant atoms. Committee spread summarizes the distribution of per-QM-atom force-
S-34

vector disagreement. It is reported alongside the primary model’s label error as a separate
evaluation quantity.
Table S8a. Held-out errors of the primary production model with seed 20260829. Energy
errors refer to the total correction energy per configuration and are given in eV. Force errors
refer to Cartesian components of the indicated atoms and are given in eV/Å.
Held-out domain Configu-
ration
countEnergy MAE
/ eVEnergy
RMSE /
eVQM force-
component
MAE /
(eV/Å)Environment
force-
component
MAE /
(eV/Å)
Original held-out domain 417 0.079242 0.100816 0.029978 0.024046
Additional negative-coordinate
held-out window centered at
−2.4 Å104 0.081962 0.104369 0.033248 0.023784
The two domains contain 521 held-out configurations in total.
The 6,560 frames are partitioned by complete umbrella windows into training, validation,
and test sets containing 5,519, 520, and 521 frames, respectively. The negative-coordinate
window subset contains 1,560 frames from earlier bare GFN2-xTB QM/MM sampling, in-
cluding 104 frames in the held-out test set. Training excludes frames from production tra-
jectories generated by the retrained DPRc model.
All 6,560 configurations are labeled with PBE0/6-31G* QM/MM minus GFN2-xTB
QM/MM energy and force differences. Model MAE is evaluated on the 521-frame held-
out set.
S9. WHAM free-energy analysis and statistical diagnos-
tics
S9.1 Statistical treatment of production trajectories
WHAM combines histogram counts from all windows and production trialsS26. The binning
range is [−3.600,2.100] Å, the bin width is 0.020 Å, the analysis temperature is 298 K, and
the dimensionless convergence tolerance is 10-10. WHAM removes the reaction-coordinate
umbrella bias while retaining the common attack-angle restraint.
For each trial–window series, statistical inefficiencies are estimated separately for the
reaction coordinate and attack angle. The larger value characterizes the temporal correlation
of that seriesS27–S29. Let Ntw be the number of samples in trial t and window w. The pooled
effective sample size is
gtw= max(g ξ,tw, gα,tw), N eff,w=X
tNtw
gtw.(S5)
S-35

Block resampling is performed independently for each trial–window series, with a block
length of ceil(g) samples. Resampled counts are pooled and WHAM is reconstructed in each
of 500 rounds. Every resampled curve is shifted to its global observed minimum. The 2.500%
and 97.500% quantiles of finite PMF values in each bin define pointwise 95% intervals. PMF
and barrier central values are reported in kcal mol-1. These pointwise 95% intervals are
descriptive. Their interpretation is restricted to individual PMF bins and excludes joint
confidence claims for peak–valley or barrier differences.
Each window is equilibrated for 100 ps, and production statistics begin after this pre-
scribed segment. No additional burn-in is removed from the production segments, and au-
tomatic equilibration detection is not used to replace the predefined equilibration segment.
Raw and retained frame counts, statistical inefficiency, effective sample size, failures, and
bootstrap coverage are recorded for each trial–window series using the definitions in this
section.
S9.2 Diagnostic definitions in the analysis implementation
The analysis implementation defines adjacent-window overlap, resampling coverage, and
between-trialconsistency. Adjacentwindowsusethesamebins. Theirnormalizedhistograms
give the Bhattacharyya coefficient
Ow,w+1 =X
b√pw,bpw+1,b,X
bpw,b= 1.(S6)
The implementation estimates statistical inefficiency using Geyer’s initial monotone pos-
itive sequence. Non-overlapping circular blocks wrap around at the end of the series. Re-
sampling draws complete blocks and any required short prefix to retain the sample count
of each trial–window series. The original coverage diagnostic considers every observed bin
in the full analysis range and requires a finite PMF in at least 0.900 of the 500 resampling
rounds for each bin. Coverage of the 235 points within the window-center range is reported
separately in Section S9.3. For between-trial consistency, curves are first aligned at the low-
est pooled-PMF bin supported by every trial. Comparisons then use the core region where
the pooled PMF is at most 8 kcal/mol.
Table S9a. WHAM and block-resampling parameters.
Parameter Value or definition
Analysis temperature 298 K
Binning range and bin width [−3.600,2.100] Å/0.020 Å
WHAM convergence tolerance Dimensionless tolerance of 10-10
Effective sample size Sum of N/max(gξ,gα) over trials
Block length and number of
resampling roundsceil(max(gξ,gα))/500
Interval and zero reference Pointwise 95% intervals, with each curve shifted to its global observed
minimum
S-36

Table S9b. Analysis parameters and diagnostic definitions.
Parameter Value or definition
Maximum WHAM iterations 100,000
Statistical-inefficiency estima-
torGeyer initial monotone positive sequence
Resampling random seed 982451653
Minimum and target adjacent-
window overlap0.050/0.200
Minimum pooled effective sam-
ple size per window50
Per-bin resampling coverage Every observed bin in the original analysis range should have a finite-
PMF resampling fraction of at least 0.900. The minimum fraction over
the original observed bins is 0.602. In a separate check, all 235 points
within the window-center range have a fraction of 1.000. Section S9.3
specifies that range.
Between-trial consistency Each trial supports the region with pooled PMF≤8 kcal/mol, and the
maximum pairwise difference is≤2 kcal/mol
S9.3 Literature comparison and DPRc statistical results
The reference comprises 705 digitized points from the red gwTP-PBE0 curve in Fig. 3a
of Giese, Zeng, and York (2022). Each simulated curve and the reference curve is shifted
by its own A(0), obtained by linear interpolation. MAE and RMSE are evaluated with
equal weights at 233 shared grid centers from−3.070 to 1.570 Å with spacing 0.020 Å. The
horizontal coordinate is retained without fitting a shift. The reference-curve source, pixel
calibration, pointwise aligned values, and residuals follow the calculation definitions in this
section.
The red reference curve is a PBE0/6-31G* free-energy estimate obtained by gwTP
reweighting of sampling from four MNDO/d QM/MM+DPRc reference potentialsS8. The
original high-level QM/MM calculations used SANDER/HFDF with ambient-potential com-
posite Ewald, 8 Å cutoffs for both real-space electrostatics and Lennard–Jones interactions,
and an LJ tail correction. Labels in the present study use AmberTools/QUICK PBE0/6-
31G* with a 9 Å cutoff. The resulting MAE and RMSE therefore include differences in
Hamiltonian, computational protocol, and sampling. Both sampling sets in this work retain
the 180◦attack-angle restraint.
The comparison uses the 705 digitized points, pixel calibration, and pointwise residuals
according to the definitions in this section.
For the retrained DPRc model, the minimum adjacent-window overlap coefficient is 0.567
and the minimum pooled effective sample size per window is 105.013. Both meet the original
thresholds. The minimum finite-bootstrap fraction over observed bins in the original analysis
range is 0.602, below 0.900, so the original coverage and between-trial support criteria are
incompletelysatisfied. Between-trialsupportismissingfrom+1.870to+1.910Å.Withinthe
S-37

window-center range from−3.100 to 1.600 Å, all 235 analysis grid points (ξ i=−3.09+0.02i,
i=0,...,234) have finite results in 500/500 bootstrap rounds.
Free-energy differences among the principal peaks, reactant, and intermediate are de-
scriptive central values. The digitization covers the literature central curve and leaves its
error bars unquantified. The original PMF and its pointwise 95% intervals use the global
observed minimum of each curve as the zero reference. Alignment at A(0) is used to compare
central curves. The RMSE values of the bare QM/MM and DPRc curves are 6.61 and 0.866
kcal mol-1, respectively. The free-energy differences of the two principal DPRc peaks relative
to the reactant are 9.21 and 16.45 kcal mol-1. Their differences from the reference central
values are +0.17 and−0.33 kcal mol-1, respectively.
S10. Data sources and licenses
Use, citation, and redistribution of the original QM9 and OMol25 data follow their respective
upstream licenses. Those terms continue to govern the upstream data when used in the
summaries, figures, and analyses generated in this study.
The xTBloom repository (https://github.com/jinzhezenggroup/xtbloom) provides
source code, build instructions, interfaces, and benchmark protocols. The LAMMPS-DPRc
repository (https://github.com/jinzhezenggroup/lammps-dprc) provides the periodic
QM/MM plugin and coordinator.
References
(S1) Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A. Quantum chemistry
structures and properties of 134 kilo molecules.Sci. Data2014,1, 140022.
(S2) Levine, D. S.; Shuaibi, M.; Spotte-Smith, E. W. C.; Taylor, M. G.; Hasyim, M. R.;
Michel, K.; Batatia, I.; Csányi, G.; Dzamba, M.; Eastman, P.; Frey, N. C.; Fu, X.;
Gharakhanyan, V.; Krishnapriyan, A. S.; Rackers, J. A.; Raja, S.; Rizvi, A.;
Rosen, A. S.; Ulissi, Z.; Vargas, S.; Zitnick, C. L.; Blau, S. M.; Wood, B. M. The Open
Molecules 2025 (OMol25) Dataset, Evaluations, and Models.arXiv2025, 2505.08762.
(S3) Bannwarth, C.; Caldeweyher, E.; Ehlert, S.; Hansen, A.; Pracht, P.; Seibert, J.;
Spicher, S.; Grimme, S. Extended tight-binding quantum chemistry methods.Wires
Comput. Mol Sci2021,11, e1493.
(S4) Katbashev, A.; Stahn, M.; Rose, T.; Alizadeh, V.; Friede, M.; Plett, C.; Steinbach, P.;
Ehlert, S. Overview on Building Blocks and Applications of Efficient and Robust
Extended Tight Binding.J. Phys. Chem., A2025,129, 2667–2682.
(S5) Bannwarth, C.; Ehlert, S.; Grimme, S. GFN2-xTB-An Accurate and Broadly
Parametrized Self-Consistent Tight- Binding Quantum Chemical Method with Multi-
poleElectrostaticsandDensity-DependentDispersionContributions.J. Chem. Theory
Comput.2019,15, 1652–1671.
S-38

(S6) Caldeweyher, E.; Ehlert, S.; Hansen, A.; Neugebauer, H.; Spicher, S.; Bannwarth, C.;
Grimme, S. A generally applicable atomic-charge dependent London dispersion cor-
rection.J. Chem. Phys.2019,150, 154122.
(S7) Friede, M.; Hölzer, C.; Ehlert, S.; Grimme, S. dxtb-An efficient and fully differentiable
framework for extended tight-binding.J. Chem. Phys.2024,161, 062501.
(S8) Giese, T. J.; Zeng, J.; York, D. M. Multireference Generalization of the Weighted
Thermodynamic Perturbation Method.J. Phys. Chem. A2022,126, 8519–8533.
(S9) Horn, H. W.; Swope, W. C.; Pitera, J. W.; Madura, J. D.; Dick, T. J.; Hura, G. L.;
Head-Gordon, T. Development of an improved four-site water model for biomolecular
simulations: TIP4P-Ew.J. Chem. Phys.2004,120, 9665–9678.
(S10) Thompson, A. P.; Aktulga, H. M.; Berger, R.; Bolintineanu, D. S.; Brown, W. M.;
Crozier, P. S.; in ’t Veld, P. J.; Kohlmeyer, A.; Moore, S. G.; Nguyen, T. D.; Shan, R.;
Stevens, M. J.; Tranchida, J.; Trott, C.; Plimpton, S. J. LAMMPS - a flexible simu-
lation tool for particle-based materials modeling at the atomic, meso, and continuum
scales.Comput. Phys. Commun.2022,271, 108171.
(S11) Schneider, T.; Stoll, E. Molecular-dynamics study of a three-dimensional one-
component model for distortive phase transitions.Phys. Rev. B1978,17, 1302–1322.
(S12) Ryckaert, J.-P.; Ciccotti, G.; Berendsen, H. J. Numerical integration of the cartesian
equations of motion of a system with constraints: molecular dynamics of n-alkanes.
J. Comput. Phys.1977,23, 327–341.
(S13) Fiorin, G.; Klein, M. L.; Hénin, J. Using collective variables to drive molecular dy-
namics simulations.Mol. Phys.2013,111, 3345–3362.
(S14) Torrie, G.; Valleau, J. Nonphysical sampling distributions in Monte Carlo free-energy
estimation: Umbrella sampling.J. Comput. Phys.1977,23, 187–199.
(S15) Kästner, J. Umbrella sampling.Wires Comput. Mol Sci2011,1, 932–942.
(S16) Zeng, J.; Giese, T. J.; Ekesan, Ş.; York, D. M. Development of Range-Corrected Deep
Learning Potentials for Fast, Accurate Quantum Mechanical/Molecular Mechanical
Simulations of Chemical Reactions in Solution.J. Chem. Theory Comput.2021,17,
6993–7009.
(S17) Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A. Big Data Meets
Quantum Chemistry Approximations: The∆-Machine Learning Approach.J. Chem.
Theory Comput.2015,11, 2087–2096.
(S18) Adamo, C.; Barone, V. Toward reliable density functional methods without adjustable
parameters: The PBE0 model.J. Chem. Phys.1999,110, 6158–6170.
(S19) Hariharan, P. C.; Pople, J. A. The influence of polarization functions on molecular
orbital hydrogenation energies.Theor., Chim, Acta1973,28, 213–222.
S-39

(S20) Case, D. A.; Aktulga, H. M.; Belfon, K.; Cerutti, D. S.; Cisneros, G. A.; Cruzeiro, V.
W.D.; Forouzesh,N.; Giese,T.J.; Götz,A.W.; Gohlke,H.; Izadi,S.; Kasavajhala,K.;
Kaymak, M. C.; King, E.; Kurtzman, T.; Lee, T.-S.; Li, P.; Liu, J.; Luchko, T.;
Luo, R.; Manathunga, M.; Machado, M. R.; Nguyen, H. M.; O’Hearn, K. A.;
Onufriev, A. V.; Pan, F.; Pantano, S.; Qi, R.; Rahnamoun, A.; Risheh, A.; Schott-
Verdugo, S.; Shajan, A.; Swails, J.; Wang, J.; Wei, H.; Wu, X.; Wu, Y.; Zhang, S.;
Zhao, S.; Zhu, Q.; Cheatham 3rd, T. E.; Roe, D. R.; Roitberg, A.; Simmerling, C.;
York, D. M.; Nagan, M. C.; Merz Jr, K. M. AmberTools.J. Chem. Inf. Model.2023,
63, 6183–6191.
(S21) Cruzeiro, V. W. D.; Manathunga, M.; Merz Jr, K. M.; Götz, A. W. Open-Source
Multi-GPU-Accelerated QM/MM Simulations with AMBER and QUICK.J. Chem.
Inf. Model.2021,61, 2109–2115.
(S22) Zhang, L.; Han, J.; Wang, H.; Car, R.; E, W. Deep potential molecular dynamics: a
scalable model with the accuracy of quantum mechanics.Phys. Rev. Lett.2018,120,
143001.
(S23) Wang, H.; Zhang, L.; Han, J.; E, W. DeePMD-kit: A deep learning package for
many-body potential energy representation and molecular dynamics.Comput. Phys.
Commun.2018,228, 178–184.
(S24) Li, T.; Xue, J.; Zhang, L.; Zhang, D.; Wang, H. Universal Machine-learning Molecular
Dynamics at the Speed of Empirical Potentials.arXiv2026, 2608.19041.
(S25) Zeng, J.; Zhang, D.; Peng, A.; Zhang, X.; He, S.; Wang, Y.; Liu, X.; Bi, H.; Li, Y.;
Cai, C.; Zhang, C.; Du, Y.; Zhu, J.-X.; Mo, P.; Huang, Z.; Zeng, Q.; Shi, S.; Qin, X.;
Yu, Z.; Luo, C.; Ding, Y.; Liu, Y.-P.; Shi, R.; Wang, Z.; Bore, S. L.; Chang, J.;
Deng,Z.; Ding,Z.; Han,S.; Jiang,W.; Ke,G.; Liu,Z.; Lu,D.; Muraoka,K.; Oliaei,H.;
Singh, A. K.; Que, H.; Xu, W.; Xu, Z.; Zhuang, Y.-B.; Dai, J.; Giese, T. J.; Jia, W.;
Xu, B.; York, D. M.; Zhang, L.; Wang, H. DeePMD-kit v3: A Multiple-Backend
Framework for Machine Learning Potentials.J. Chem. Theory Comput.2025,21,
4375–4385.
(S26) Kumar, S.; Rosenberg, J. M.; Bouzida, D.; Swendsen, R. H.; Kollman, P. A. THE
weighted histogram analysis method for free-energy calculations on biomolecules. I.
The method.J Comput. Chem1992,13, 1011–1021.
(S27) Chodera, J. D. A Simple Method for Automated Equilibration Detection in Molecular
Simulations.J. Chem. Theory Comput.2016,12, 1799–1805.
(S28) Flyvbjerg,H.; Petersen,H.G.Errorestimatesonaveragesofcorrelateddata.J. Chem.
Phys.1989,91, 461–466.
(S29) Geyer, C. J. Practical Markov Chain Monte Carlo.Stat., Sci,1992,7, 473–483.
(S30) Darden, T.; York, D.; Pedersen, L. Particle mesh Ewald: An N log(N) method for
Ewald sums in large systems.J. Chem. Phys.1993,98, 10089–10092.
S-40

(S31) Essmann, U.; Perera, L.; Berkowitz, M. L.; Darden, T.; Lee, H.; Pedersen, L. G. A
smooth particle mesh Ewald method.J. Chem. Phys.1995,103, 8577–8593.
(S32) Nam, K.; Gao, J.; York, D. M. An efficient linear-scaling Ewald method for long-
range electrostatic interactions in combined QM/MM calculations.J. Chem. Theory
Comput.2005,1, 2–13.
(S33) Giese, T. J.; York, D. M. Ambient-Potential Composite Ewald Method for ab Ini-
tio Quantum Mechanical/Molecular Mechanical Molecular Dynamics Simulation.J.
Chem. Theory Comput.2016,12, 2611–2632.
S-41