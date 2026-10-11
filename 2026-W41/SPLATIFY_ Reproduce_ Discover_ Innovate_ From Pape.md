# SPLATIFY: Reproduce, Discover, Innovate! From Papers and Ideas to Trainable 3DGS Code

**Authors**: Seemandhar Jain, Keshav Gupta, Manmohan Chandraker

**Published**: 2026-10-06 21:06:33

**PDF URL**: [https://arxiv.org/pdf/2610.09116v1](https://arxiv.org/pdf/2610.09116v1)

## Abstract
The rapid growth of 3D Gaussian Splatting (3DGS) research demands significant effort to reimplement papers before building on them. We introduce SPLATIFY, a multi-agent framework that converts 3DGS papers into trainable gsplat-based implementations, where generic paper-to-code methods and frontier models fail. SPLATIFY achieves this through five innovations: (1) A context-free grammar for gsplat over a modular method template with extension points for losses, densification, rendering, and optimization, constraining synthesis so generated code satisfies gsplat's architectural invariants by construction. (2) Architectural elements for faithful reproduction: fork-aware citation recovery retrieving component-level code at function-level granularity, Graph-of-Thought synthesis in topological dependency order, RAG-guided in-context example selection from over 20 verified implementations, and visual feedback combining PSNR-guided regeneration, Gaussian-level structural checks, and VLM-driven patching. (3) Knowledge-driven compositional improvement that autonomously finds weaknesses and composes complementary regularizers, losses, and densification strategies to improve upon original results. (4) Interdisciplinary method discovery where agents retrieve physical priors from outside the 3DGS literature and compose them with rendering knowledge to produce methods for previously unaddressed scene types. (5) SPLATIFY-Bench, an evaluation framework across 30 diverse 3DGS papers. On papers without public code, SPLATIFY matches expert implementations while reducing development time from weeks to minutes, and through compositional discovery further improves PSNR by up to 2.4 dB. We additionally demonstrate novel methods for volumetric nebula rendering and other scientific domains, synthesized entirely by SPLATIFY.

## Full Text


<!-- PDF content starts -->

SPLATIFY: Reproduce, Discover, Innovate!
From Papers and Ideas to Trainable 3DGS Code
Seemandhar Jain Keshav Gupta Manmohan Chandraker
University of California, San Diego
{sejain, keg019, mkchandraker}@ucsd.edu
Time 
Saved: 
~99% 
(Weeks 
? 
Minutes)
Baselines 
Ours
+2dB 
PSNR 
over 
Vanilla 
3DGS
Cross 
Domain 
Innovation 
- 
Method 
Invention
For 
scene 
types 
that 
no 
3DGS 
paper 
has 
tackled
Radiative 
Transfer 
Physics
(Outside 
3DGS 
lit.)
No 
Prior 
3DGS 
method 
exists
SPLATIFY
Agent 
generates 
completely 
new 
method
NebulaGS
-
Per 
Gaussian 
Emission 
Parameter
-
Spectral 
Frequncy 
Loss 
via 
2D 
FFT
-
Scale 
isotropy 
penalty
+
Text 
Citations
Repo 
Diffs
Function 
Level 
Components
Fork 
Aware 
Citation 
Recovery
Critique 
Agent
Graph 
of 
Thought 
Code 
Gereration
def 
loss():
...
def 
densify():
 
...
def 
render():
...
gsplat 
CFG 
& 
Modular 
Template
SPLATIFY 
Faithful 
Reproduction 
from 
Paper
Knowledge 
Base 
K
Knowledge 
Discovery 
& 
Compositional 
Improvement
Autonomously 
diagnoses 
& 
Compose 
Complementary 
Techniques
Reproduced 
Papers
Reproduced 
Papers
PSNR 
Improvement
+2.4 
dB
AbsGS 
regularizer 
+ 
TrickGS 
pruning
Figure 1.SPLATIFY: Reproduce, Discover, Innovate.Given a 3DGS paper, SPLATIFYproduces trainable gsplat code in minutes existing
approaches cannot (Level 1), autonomously composes complementary techniques for up to +2.4 dB PSNR improvement for existing
literature (Level 2), and invents entirely new methods by retrieving physical priors from outside the 3DGS literature (Level 3).
Abstract
The rapid growth of 3D Gaussian Splatting (3DGS) research
demands significant effort to reimplement papers before
building on them. We introduceSPLATIFY, a multi-agent
framework that converts 3DGS papers into trainable gsplat-
based implementations, where generic paper-to-code meth-
ods and frontier models fail.SPLATIFYachieves this through
five innovations: (1) Acontext-free grammar for gsplat
over a modular method template with extension points for
losses, densification, rendering, and optimization, constrain-
ing synthesis so generated code satisfies gsplat’s architec-
tural invariants by construction. (2)Architectural elements
for faithful reproduction: fork-aware citation recovery re-
trieving component-level code at function-level granularity,
Graph-of-Thought synthesis in topological dependency or-
der, RAG-guided in-context example selection from over
20 verified implementations, and visual feedback combin-
ing PSNR-guided regeneration, Gaussian-level structural
checks, and VLM-driven patching. (3)Knowledge-driven
compositional improvementthat autonomously finds weak-nesses and composes complementary regularizers, losses,
and densification strategies to improve upon original results.
(4)Interdisciplinary method discoverywhere agents retrieve
physical priors from outside the 3DGS literature and com-
pose them with rendering knowledge to produce methods for
previously unaddressed scene types. (5)SPLATIFY-BENCH,
an evaluation framework across 30 diverse 3DGS papers.
On papers without public code,SPLATIFYmatches expert im-
plementations while reducing development time from weeks
to minutes, and through compositional discovery further im-
proves PSNR by up to 2.4 dB. We additionally demonstrate
novel methods for volumetric nebula rendering and other
scientific domains, synthesized entirely bySPLATIFY. Code,
data, and all implementations will be publicly released.
1. Introduction
3D Gaussian Splatting has triggered a fast-growing sub-
fieldof neural rendering, yet the community publishes
knowledge far faster than it can absorb. Since Kerbl et
al. [28] introduced 3DGS in 2023, follow-up papers have
grown to over 1,500 in under three years [ 7,18], with track-
1
arXiv:2610.09116v1  [cs.CV]  6 Oct 2026

ing repositories recording roughly three new arXiv submis-
sions per day [ 30]. For a researcher who wants tousea
recently published method, however, the path from paper to
working code remains prohibitively slow.
Existing infrastructure cannot keep pace with this
growth.While over 40% of tracked 3DGS papers release
no code [ 2], others publish isolated forks with incompat-
ible assumptions about attribute layouts, density control,
rendering conventions [ 30], and CUDA kernels. Building
on a recent method typically demands weeks of manual
reimplementation and debugging. Generic pipelines like Pa-
per2Code [ 46] and AutoP2C [ 33] produce code that do not
compile against any rasterization backend, and frontier mod-
els generate Python that collapses at training time for lack
of domain knowledge about Gaussian attribute coupling or
adaptive density control. No existing paper-to-code system,
in any domain, handles 3DGS’s fork-dominated landscape,
provides Gaussian-level structural verification, or moves be-
yond reproduction to compositional improvement.
Even if reproduction were perfectly solved, the deeper
bottleneck would persist: the knowledge across 2k+ pa-
pers is neither discoverable nor composable.There is no
efficient way to determine whether combining a depth dis-
tortion loss from 2DGS [ 23] with stochastic splitting from
3DGS-MCMC [ 29] and an opacity regularizer from Ab-
sGS [ 61] would outperform any of them individually. With
hundreds of published techniques, the number of valid com-
positions is enormous, and manual exploration covers a neg-
ligible fraction. Entire application domains where Gaussians
are a natural geometric fit, from optically thin nebulae to
fluorescence microscopy, remain untouched because the en-
gineering barrier is too high. The field needs infrastructure
fordiscovering,composing, andextendingthe knowledge its
papers contain.
We introduce SPLATIFY(Fig. 1), a multi-agent frame-
work operating at three levels: faithful reproduction of
3DGS papers as trainable gsplat [ 60] plugins, compositional
improvement that autonomously combines complementary
techniques, and cross-domain method invention for unad-
dressed scene types. Its design rests on five contributions:
1.A context-free grammar for gsplat pluginsover a mod-
ular template with nine extension points covering the full
3DGS modification space, ensuring generated code satis-
fies architectural invariants by construction (Sec. 3.2).
2.Faithful reproductionextending the multi-stage pipeline
of [25] with fork-aware citation recovery at function-level
granularity, Gaussian-specific smoke tests, and PSNR
training-curve monitoring (Sec. 3.3).
3.Compositional improvementthrough autonomous diag-
nosis of weaknesses in reproduced methods and compo-
sition of complementary regularizers, losses, and densifi-
cation strategies from the knowledge base, yielding up to
+2.4 dB PSNR gain with no human guidance (Sec. 3.4).4.Cross-domain method invention, where agents retrieve
physical priors from outside the 3DGS literature and syn-
thesize physics-informed modifications for unaddressed
domains such as volumetric nebula rendering and thin-
film bubble iridescence (Sec. 3.4).
5.SPLATIFY-Bench, a 30-paper benchmark stratified by
code availability and technical complexity (Sec. 4.1).
On papers without public code, SPLATIFYmatches expert
implementations within ±0.5 dB PSNR and ±0.02 SSIM
while existing baselines fail to produce trainable code. Cross-
domain invention produces working methods for scene types
no prior paper has addressed, entirely without human guid-
ance. SPLATIFY, SPLATIFY-BENCH, all implementations,
and the code will be publicly released.
2. Related Work
The 3DGS Reproducibility Problem.3D Gaussian Splat-
ting [ 28] represents scenes as anisotropic Gaussians ras-
terized through a differentiable tile-based renderer. The
field has grown to over a thousand papers [ 7,18] spanning
anti-aliasing [ 62], geometry [ 23], densification [ 29], and
compression [ 16,31]. We target gsplat [ 60] integrated with
Nerfstudio [ 50] as the compilation target. A fundamental
barrier is the fragmented implementation landscape: most
papers fork the original INRIA codebase into one-off repos-
itories that resist comparison or reuse, demanding deeper
domain awareness than any existing system provides.
Automated Research Reproduction.Paper2Code [ 46] and
AutoP2C [ 33] use staged parsing but produce non-trainable
outputs for vision methods; AutoReproduce [ 68] and Re-
Pro [ 69] improve fidelity through citation lineage extraction
yet remain domain-agnostic. PaperBench [ 48] quantifies the
gap: the strongest agent scores 21% against 41.4% for hu-
man researchers on ICML papers. Multi-agent frameworks
including MetaGPT [ 22], ChatDev [ 44], AutoGen [ 54], and
repository-level agents [ 53,56] advance role-specialized col-
laboration but lack scientific rendering-specific reasoning.
The closest prior work is NERFIFY[ 25], which introduces
grammar-constrained generation, compositional citation re-
covery, and visual feedback for NeRF-to-Nerfstudio syn-
thesis. However, NERFIFYis limited to NeRFs, and 3DGS
methods alter tightly coupled components across fragmented
repository forks, unlike NeRF systems that modify relatively
isolated neural modules. NERFIFYcannot handle this fork-
dominated ecosystem and lacks Gaussian-level verification.
SPLATIFY adds fork-aware recovery via parent-repository
diffs and Gaussian structural checks with PSNR curve mon-
itoring, and goes beyond reproduction: it composes tech-
niques to surpass the reproduced result and retrieves physical
priors from outside the field for unaddressed scene types.
Structured Reasoning and Iterative Code Refinement.
Constraining LLM outputs with formal structure improves
code correctness, as shown by pushdown-automaton decod-
2

Figure 2.Simplified SPLATIFY pipeline.A high-level view of
reproduction, refinement, and interdisciplinary method discovery.
Fig. 3 shows the detailed architecture.
ing [ 14], grammar prompting [ 51], and CFG-based visual
program synthesis [ 65]. Iterative refinement through execu-
tion feedback is equally standard: Graph of Thoughts [ 5]
and Tree of Thoughts [ 59] organize reasoning as search-
able structures, while Self-Debugging [ 8], Reflexion [ 47],
and Self-Refine [ 35] enable models to critique and repair
their own outputs. SPLATIFYcombines both: a gsplat CFG
guarantees architectural validity, while PSNR training-curve
analysis and VLM-guided patching provide closed-loop feed-
back to converge on expert-quality implementations.
3. Proposed Approach
3.1. Problem Statement
We consider three objectives of increasing scope:faithful
reproduction,compositional improvement, andinterdisci-
plinary method discovery.
Faithful Reproduction.Given a 3DGS paper P, we
synthesize an executable repository C= (F, G) , where
F={f 1, . . . , f n}is a set of source files and G=
BUILDREPODAG(F) encodes their import and dataflow
dependencies. A parsing pipeline extracts structured in-
formation E(P) =⟨T, I, Q, B⟩ (text, figures, equations,
bibliography) and passes it to our multi-agent system A: 
E(P);R
7→ C. The resource tuple R= (K,W,X) com-
prises a CFG-grounded gsplat knowledge base K(Sec. 3.2),
web resources Wfor cited papers and fork-level diffs, and a
library of verified implementations Xserving as in-context
examples. Reproduction succeeds when Ccompiles against
gsplat, trains to convergence, and achieves rendering quality
within ϵ= 0.5 dB PSNR of reported metrics while correctly
implementing every novel component inP.
Compositional Improvement.Many 3DGS methods ad-
dress a single weakness while remaining vulnerable to others.
Given a faithful reproduction C0and a library of verified tech-
niques Ktech={t 1, .., t k} ⊂ K , the agent uses visual feed-
back to find weaknesses, queries Kfor complementary reme-
dies, and validates that C+=COMPOSE(C 0, S⊆ K tech)
remains a valid grammar derivation and improves quality on
held-out views. The agent receives no human guidance; it
must independently diagnose, retrieve, and verify.
Interdisciplinary Method Discovery.The furthest-
reaching objective targets scene types no published 3DGSpaper has addressed, such as volumetric astronomical phe-
nomena or fluorescence microscopy. Given a target do-
mainDand a multi-view training set, the agent retrieves
physical priors Wphys⊂ W from literature outside 3DGS,
identifies which splatting assumptions are violated, and
synthesizes physics-informed modifications within gsplat:
CD=A 
D,W phys;K,G
. Success is measured by im-
provement over vanilla 3DGS, cross-view consistency, and
physical plausibility. This moves beyond translating existing
knowledge toward generating new knowledge by bridging
disciplinary boundaries.
3.2. CFG-Constrained gsplat Method Template
Unconstrained LLM generation can produce code that looks
plausible but violates framework invariants in ways that sur-
face only at training time: dead optimizer entries, missing
density-control bookkeeping, losses wired into the wrong
module. Our solution is to formalize gsplat-based Nerfstu-
dio plugins as a CFG G= (Σ, V, S, P) and provide agents
with a concrete method template instantiating this grammar
as runnable code. This makes synthesis a constrained task
where agents decidewhichextension points a paper’s con-
tributions map to andwhatcode to place there, while the
structural skeleton is fixed by construction.
Template design.By analysis of over 40 3DGS pa-
pers spanning anti-aliasing [ 62], geometry [ 23], densifica-
tion [ 29], depth supervision, and feature distillation, we
identifynine extension pointscovering the full modi-
fication space (listed in the supplementary). The tem-
plate’s core class GSTemplateModel inherits tile-based
gsplat rasterization, adaptive density control, SH sched-
ules, camera optimization, and the standard optimizer
set. Agents subclass it and override only relevant hooks:
compute custom losses() for a new regularizer,
orpopulate modules() plussplit gaussians()
for a new per-Gaussian attribute.
The grammar.Terminals Σare framework classes, non-
terminals Vare abstract components, S=⟨PLUGIN⟩ is the
start symbol, and productions Pencode valid compositions.
The grammar serves two roles: it is included in every agent’s
system prompt as a structural specification, and it defines the
verification contracts used during GoT synthesis (Sec. 3.3),
where the master agent checks after each generated file that
the repository remains a valid derivation of G. An abbrevi-
ated version is shown in Fig. 3 Stage 1; the full grammar is
in the supplementary.
3.3. Proposed Multi-Stage Framework
Fig. 2 provides a simplified view of the overall SPLATIFY
pipeline, while Fig. 3 shows the complete system architec-
ture. We design a multi-stage framework similar to [ 25],
but with several extensions and 3DGS-specific components,
which are described below.
3

Code 
1
Paper 
1
Code 
2
Paper 
2
Code 
N
Paper 
N
RAG 
Knowledge 
Base 
(K)
Compositional 
Dependency 
Graph 
Generation
RETRIEVAL 
COMPONENT
Searches 
and 
fetches 
most 
relevant 
papers 
from 
the 
knowledge 
base
Knowledge 
Indexing
Retrieved 
Context
We 
finally 
get 
a 
dependency 
graph 
where 
each 
edge 
gives 
a 
code 
snippet 
or 
information 
corresponding 
to 
the 
fork 
diff 
or 
text 
parsed.
Implement 
the 
3DGS 
followup 
paper, 
GOF
1.
Component 
Identification
Rendering
Regularization
Densification
2. 
Extract 
Components 
Via 
Fork 
Diff
3DGS
2DGS
Scaffold-GS
Mip-Splatting
Repo
 
Forks 
a 
known 
Base
 
Repo?
Compute 
Diff 
between 
Base
 
and 
Repo
Repo
 
Exists
Repo
 
Does 
NOT 
Exist
Download 
and 
parse 
PDF 
for 
component 
extraction 
(equations/constraints)
Base
 
Repos
YES
NO
2. 
Extract 
Components 
Via 
Text 
Parsing
Create 
a 
semantic 
label 
corresponding 
to 
the 
Diff 
and 
store 
the 
code 
at 
function 
level 
granularity
Iterative 
Multi-Hop 
Retrieval 
Identification
Graph 
Of 
Thought 
Code 
Generation
Critique 
Agent
Metric 
Branch
Geometry
 
Branch
Semantics
 
Branch
PSNR 
Training 
Curve 
Monitoring
Stalled 
Plateau
Late 
Stage 
Collapse
Normal 
Training
Per 
iteration 
PSNR 
Monitoring 
on 
held 
out 
validation 
view
VLM 
Feedback
Cross 
View 
Artifact 
Consensus
(Flags 
Floaters)
Local 
Window 
PSNR 
Maps
The 
master 
agent 
orchestrates 
code 
generation 
through 
GOT, 
exploiting 
the 
constrained 
structure 
of 
the 
method 
template
Knowledge 
Discovery 
& 
Invention
Densification 
Strategies 
- 
D
Composable 
Loss 
Terms 
L
Anti 
Aliasing 
A
Appearance 
Models 
-     
M
Training 
Schedule 
-   
T
Search 
Space 
S
Propose 
Variants 
(mutate 
one 
axis 
of 
S)
Explorer 
Agents
8 
Code 
Variants
2k 
Iterations
Multi- 
Fidelity 
Cascade
V1
V2
V3
V4
V5
V6
V7
V8
Q1
Q2
Q3
Q4
Q5
Q6
Q7
Q8
7k 
Iterations
V3
V5
V7
V8
Q3
Q5
Q7
Q8
V5
V8
Q5
Q8
30k 
Iterations
For 
each 
variant 
(V), 
compute 
Quality 
(Q)
Q 
= 
(0.4·PSNR+0.3·(1?LPIPS)+0.3·SSIM)
Stage 
2
Stage 
3 
and 
4
Stage 
1
Abbreviated 
CFG 
G
 
for 
gsplat-based 
Nerfstudio 
plugins. 
Paper     
+     
Code
Knowledge 
Discovery
New 
Scene
Invention
I 
want 
to 
improve 
upon 
my 
already 
implemented 
paper
I 
want 
to 
make 
a 
new 
3DGS 
method 
so 
that 
it 
runs 
on 
my  
novel 
scene
Scenario 
1
Scenario 
2
Knowledge 
outside 
3DGS
Trick-GS
3DGS
RAIN-GS
Taming 
3DGS
E2E 
R-D 
3DGS
EAGLES
Spec-Gaussian
LightGaussian
SOGS
Pruning
Frequency 
based 
scoring
Masking
Gaussian 
blurring
Resolution 
scheduling
Progressive 
Gaussian 
scale
optimized 
fused 
SSIM
densification 
and 
rasterizationFigure 3.SPLATIFYoverview.Reproduction(left): a parsing agent structures papers into a CFG-grounded knowledge base; fork-aware
dependency resolution recovers components via citation graphs and repository diffs; GoT synthesis generates code in topological order with
PSNR-guided and VLM-driven refinement.Knowledge discovery and invention(right): the populated knowledge base enables compositional
improvement, where agents compose techniques to surpass original results, and interdisciplinary method creation, where physical priors
from outside 3DGS literature are retrieved to address novel scene types.
3.3.1. Stage 1: RAG-Guided In-Context Learning and
CFG
SPLATIFYcenters on the knowledge base Kand a library of
verified implementations X={(P i,Ci)}|X|
i=1, seeded with
20 manually validated papers. The agent first converts P
to structured markdown via MinerU [ 40], stripping non-
technical sections while preserving equations, pseudocode,
diagrams, and citation relationships. For each new target, a
retrieval agent identifies the relevant extension points and
selects the most similar subset of Xby extension-point over-
lap and technique type, so a paper proposing a regularization
loss retrieves other regularization implementations. Any
paper SPLATIFYsuccessfully implements is automatically
added back toX, progressively broadening coverage.
3.3.2. Stage 2: Fork-Aware Compositional Dependency
Resolution
3DGS papers are deeply compositional: a single method
routinely assembles rendering, regularization, densification,
and post-processing components from many prior works.
Unlike NeRF methods, which ship as standalone packages,
roughly 70% of 3DGS papers release code as forks of the IN-
RIA repository [ 28]. A system that retrieves only text-level
references misses these code-level dependencies entirely.
Fork-aware recovery.The agent builds a dependency graph
via iterative multi-hop retrieval, identifying borrowed com-
ponents, recursively resolving transitive dependencies, and
terminating when all interface contracts are satisfied. The
key departure from prior work is at the extraction step:
for each cited paper, the agent checks whether its repos-itory forks a known base (3DGS, Mip-Splatting, 2DGS,
Scaffold-GS), computes a structured diff against the par-
ent, and maps each changed function to a semantic la-
bel (e.g., loss/depth distortion ), yielding exact
function-level contributions rather than noisy text-level ci-
tations. When no repository exists, the agent falls back to
PDF-level extraction. Transitive fork chains are resolved by
chaining diffs to recover the full modification stack.
Running example.Figure 3 Stage 1 shows Trick-GS [ 1],
which assembles 8 techniques from 7 papers: volume mask-
ing from Self-Organizing Gaussians [ 37], significance prun-
ing from LightGaussian [ 16], SH band masking from E2E-
RD [ 52], progressive training from EAGLES [ 19] and Spec-
Gaussian [ 58], progressive scales from RAIN-GS [ 27], and
accelerated rasterization from Taming 3DGS [ 36]. Several
of these fork INRIA 3DGS. Our agent diffs each fork against
its parent (Fig. 3), tags changed functions with semantic
labels, and chains results across transitive edges, expanding
the graph to 12 papers. Such depth is common: Gaussian-
Pro [10] spans 8 dependencies, while Scaffold-GS [34] and
PGSR [6] each span 7.
3.3.3. Stage 3: Grammar-Guided Repository Generation
The master agent orchestrates a Graph-of-Thought (GoT)
code generation approach exploiting the constrained struc-
ture of our method template. Since every plugin subclasses
GSTemplateModel and inherits the full splatfacto train-
ing loop, rasterization, density control, and optimizer wiring,
synthesis reduces to identifying extension points to which
a paper’s contributions map and filling in only the relevant
4

hooks. The agent classifies contributions into one or more
of the nine extension points defined by G, verifies the se-
lected combination before writing code, then generates files
in topological order with DAG construction, interface freez-
ing, implementation, and smoke testing. Most repositories
pass within two iterations.
3.3.4. Stage 4: Visual-Driven Feedback
After the system produces trainable code, we perform smoke
training for 3k iterations and dispatch rendered images to
a critique agent operating on four branches (Fig. 3): amet-
ric branchcomputing local-window PSNR and SSIM er-
ror maps, ageometry branchflagging floaters and ghosting
via cross-view artifact consensus, asemantics branchus-
ing a VLM to produce structured diagnoses with candidate
patches, and atraining-curve branchunique to SPLATIFY.
The last branch exploits a key difference from NeRF: be-
cause 3DGS trains in minutes, the agent can monitor the full
PSNR trajectory. 3DGS has a structurally distinct densifi-
cation phase terminating around 15k iterations, after which
errors in pruning or loss schedules manifest as failures invis-
ible during short smoke runs. The agent flags two modes:
alate-stage collapse(PSNR drops sharply post-15k) and a
stalled plateau(PSNR improves by less than 0.1dB over any
5k-iteration window after warm-up). In both cases the agent
localizes the fault to a specific extension point and issues a
targeted patch. The loop continues until no further feedback
is produced, the iteration budget is exhausted, or the paper’s
reported PSNR is matched.
3.4. Knowledge Discovery via Evolutionary Tech-
nique Composition
Beyond faithful reproduction, SPLATIFYoperates in two
modes (Fig. 3, right):compositional discovery, which im-
proves a synthesized implementation by composing com-
plementary techniques from the 3DGS literature, andcross-
domain invention, which retrieves physical priors from out-
side the field to address scene types no prior paper has tack-
led.
Technique search space.The gsplat extension points define
a structured configuration S= (D,L,A,M,T) : mutu-
ally exclusive densification strategies D(ADC, AbsGrad,
MCMC), composable loss terms L= 2{ℓ1,...,ℓ n}(depth, nor-
mal, flatness, distortion), anti-aliasing options A, appearance
models M, and continuous training-schedule parameters T.
Losses compose additively; densification strategies are mu-
tually exclusive as they govern the same split-and-prune
logic. For cross-domain invention, Lis augmented with
physics-informed losses retrieved from outside the 3DGS
literature.
Evolutionary search.Inspired by FunSearch [ 45], we main-
tainK=8 islands each managed by an explorer agent. Each
round proceeds in three steps (Fig. 3): (1) each explorerproposes a variant by mutating one axis of Sor compos-
ing an additional loss or strategy from Kwhen a diagnosed
weakness suggests it; (2) the Stage 3 pipeline converts each
proposal into runnable gsplat code, evaluated via a multi-
fidelity cascade ( 2k→7k→30k iterations, eliminating the
bottom half at each stage) to reduce per-round compute;
(3) agents are ranked by Pareto sorting over normalized
0.4·PSNR+0.3·(1−LPIPS)+0.3·SSIM on subset of train-
ing views. The loop terminates when no agent improves
for three consecutive rounds or the round budget is ex-
hausted. For discovery, the search space is restricted to
techniques already in K; for invention, the agents addition-
ally query Wphys to retrieve domain-specific priors and pro-
pose physics-informed losses not present in any existing
3DGS paper. We demonstrate both modes in Sec. 4.5.1:
compositional discovery yields up to +2.4 dB PSNR over
the reproduction-only baseline, and cross-domain invention
produces new methods for volumetric nebula rendering and
thin-film bubble iridescence.
4. Experiments
We evaluate SPLATIFYon SPLATIFY-BENCH, a new bench-
mark for 3DGS paper-to-code synthesis, and present re-
sults on reproduction quality (Sec. 4.3), novelty preservation
(Sec. 4.4), knowledge discovery (Sec. 4.5), and ablations.
4.1.SPLATIFY-BENCH
Existing paper-to-code benchmarks [ 46,48] evaluate gen-
eral ML code generation and lack criteria specific to 3DGS,
where implementations must coordinate Python-level train-
ing logic with a CUDA rasterization backend, manage heuris-
tic adaptive density control, and navigate a fragmented
ecosystem of forked repositories. SPLATIFY-BENCHis the
first benchmark targeting this domain, comprising30 papers
across three sets.
Set 1: Never-Implemented Papers (10).Papers with no
publicly available code, paired with expert reference im-
plementations we commissioned. Since no implementation
exists online, LLMs cannot have encountered corresponding
code during pretraining, ensuring uncontaminated evalua-
tion. [11, 15, 17, 24, 39, 41, 43, 49, 55, 57]
Set 2: Papers with Public Code (10).Papers with re-
leased repositories spanning densification strategies, regu-
larization losses, and appearance modeling, all portable to
gsplat, allowing direct comparison against original author
code.[9, 12, 13, 20, 21, 28, 32, 38, 42, 67]
Set 3: Novelty Coverage Papers (10).Papers chosen for the
number and clarity of their distinct technical contributions,
each with multiple verifiable novelty items (loss formula-
tions, architectural modules, training schedules, Gaussian
attribute modifications), used exclusively for the novelty
analysis in Sec. 4.4. [1, 23, 26, 29, 31, 34, 58, 63, 64, 66]
5

Table 1.SPLATIFYvs. reported results and human implementa-
tions on Set 1.All other baselines failed to produce trainable code
for any paper. Full results in the supplementary.
Paper Scene Reported Expert Impl. SPLATIFY(Ours)
PSNR↑SSIM↑LPIPS↓#GS (106) PSNR↑SSIM↑LPIPS↓#GS (106) PSNR↑SSIM↑LPIPS↓#GS (106)
DWTGS [39] mipnerf360 19.75 0.590 0.343 – 17.92 0.508 0.471 1.21 18.68 0.510 0.374 1.641
Dec. Densification [24] mipnerf360 27.79 0.826 0.214 1.469 26.85 0.792 0.286 1.44 27.17 0.820 0.261 0.650
BOGausS [43] mipnerf360 27.54 0.804 0.251 0.32 27.25 0.801 0.276 0.81 27.34 0.809 0.263 0.440
Opti3DGS [17] mipnerf360 27.09 0.809 0.237 1.327 27.51 0.816 0.221 6.18 27.71 0.841 0.209 1.406
CLoD-GS [11] Truck 25.29 0.880 0.148 1.46 24.83 0.854 0.227 0.40 25.48 0.873 0.128 0.384
4.2. Baselines, Metrics, and Experimental Setup
Baselines.We compare against five systems: generic
paper-to-code pipelines Paper2Code [ 46] and AutoP2C [ 33];
frontier LLMs such as GPT-5.2 ET, Gemini 3.1 Pro, and
DeepSeek 3.2; and expert implementations by researchers
with 3DGS experience (typically 1–2 weeks per paper).
SPLATIFYuses the full pipeline with GPT-5.2 as the base
LLM for agents. All baselines were evaluated under the same
setup: identical paper PDF and supplementary, prompts, and
compute budget.
Metrics.We report PSNR (dB), SSIM, LPIPS (lower is
better), and Gaussian count (#GS, millions) as a memory
proxy. SPLATIFYcompletes in 10–30 minutes per paper.
API cost analysis is in the supplementary.
Datasets and protocol.Our primary benchmark is Mip-
NeRF 360 [ 4] (five outdoor, four indoor unbounded scenes).
We follow the evaluation conventions of [ 3]. Every 8th
image is held out for testing. We report on all nine scenes
including the often-omittedflowersandtreehill, and restrict
Tanks and Temples totrainandtruck. All methods train
for 30k iterations on identical data and test viewpoints on
NVIDIA A6000 GPUs (48 GB), averaged on multiple seeds.
Per-scene breakdowns are in the supplementary.
4.3. Comparison to Expert Implementations
4.3.1. Set 1: Never-Implemented Papers
We first evaluate SPLATIFYon papers for which no public
implementation exists. This setting provides the strongest
test of paper-driven synthesis capability: since no code for
these methods has been released, the LLMs powering both
our agents and the baselines cannot have encountered cor-
responding implementations during pretraining, ruling out
data contamination. Expert developers required one to two
weeks per paper to produce reference implementations, while
SPLATIFYgenerates comparable code in minutes.
Quantitative and Qualitative Results.Table 1 shows
results for representative Set 1 papers, each evaluated un-
der its own reported protocol (most average across all nine
Mip-NeRF 360 scenes; some report per-scene). SPLATIFY
achieves visual quality within ±0.5 dB PSNR and ±0.02
SSIM of expert implementations on average, with qualitative
comparisons shown in Fig. 4. All baselines fail to produce
trainable code (Table 2): generic systems cannot recover
GT
Expert
SPLATIFY
GT
Expert
SPLATIFY
BOGausS
CLoD-GS
DWT-GS
Dec. 
Densification
MSAA
VOD-GSFigure 4.Qualitative comparison of SPLATIFYagainst baselines
on Set 1 papers.Rendered novel views from SPLATIFYand expert
human implementation. All baselines fail to produce trainable code;
SPLATIFYclosely matches expert quality.
Table 2.Executability comparison.All baselines fail to produce
trainable code. ∼denotes occasional success that never reaches
training stability or paper-level results.
Metric Paper2Code AutoP2C GPT-5.2 Gemini 3.1 Pro DeepSeek 3.2 NERFIFY SPLATIFY (Ours)
Imports Resolve✓×✓ ✓ ✓ ✓ ✓
Compiles / Trainable × ×∼ ∼×✓ ✓
Training Stability × × × × × ×✓
Gaussian Invariants × × × × × ×✓
Converges to Paper Results × × × × ×∼✓
Table 3.Set 2: SPLATIFYvs. original author repositories.Met-
rics from original papers compared against SPLATIFY’s automated
gsplat implementations trained under identical conditions. Full
results in the supplementary.
Paper Scene Original Repository SPLATIFY(Ours)
PSNR↑SSIM↑LPIPS↓#GS (106) PSNR↑SSIM↑LPIPS↓#GS (106)
3DGS [28] Bicycle 25.246 0.771 0.205 – 26.162 0.765 0.359 0.964
LP3DGS [67] Bicycle 24.906 0.737 0.264 2.51 24.639 0.740 0.189 1.310
GNS [13] Truck 26.520 0.899 0.119 0.60 25.970 0.891 0.095 0.490
Micro-Splatting [32] Kitchen 34.100 0.960 0.011 0.67 33.145 0.935 0.099 1.620
CompGS [38] Counter 28.467 0.895 0.222 0.385 28.283 0.894 0.153 0.665
implicit dependencies from citations or structure multi-file
gsplat plugins correctly.
4.3.2. Set 2: Papers with Public Code
Table 3 compares SPLATIFYagainst original author reposito-
ries under identical training conditions. SPLATIFYachieves
comparable performance with average deviations under
0.3 dB PSNR, while translating each method into a stan-
dardized gsplat plugin within the Nerfstudio ecosystem for
fair cross-method comparison. Although LLMs may have
encountered these codebases during pretraining, baselines
still fail to produce trainable code even with this potential
advantage.
4.4. Novelty Coverage Analysis
A system that compiles and trains but silently drops a
novel loss term has failed at the task even if aggregate
metrics appear reasonable. We evaluate novelty preser-
vation on 10 papers from Set 3 using four metrics fol-
lowing [ 25]:C(fraction correctly implemented), I(par-
tial/incorrect), M(missing), and hyperparameter fidelity
W=|{n∈ N:|θ n−ˆθn|<0.1|θ n|}|/|N| . We addition-
ally compute a weighted semantic score (P
iwisi)/P
iwi,
where wi∈[0,1] reflects component importance and si∈
6

{0,0.2, . . . ,1.0} rates completeness, recognizing function-
ally equivalent implementations while penalizing semantic
errors. All scoring is performed by Claude Opus 4.6 as an
LLM judge, provided with the source paper, generated codes,
and a structured rubric; human experts with 3DGS develop-
ment experience independently scored all components blind
to system identity, agreeing with the judge at κ=0.87 on the
C/I/M label and Spearman ρ=0.92 on the semantic score,
reading each implementation function-by-function against
its paper and flagging the exact baseline deviations (e.g.,
Euler angles for quaternions, Sigmoid vsTanh opacity)
independent of PSNR; they also verified every component
added in compositional discovery and cross-domain inven-
tion. SPLATIFYattains near perfect scores with semantic
scores above 0.94 across all papers (Table 4).
4.5. Knowledge Discovery: Beyond Reproduction
We now evaluate SPLATIFY’s ability to go beyond faithful
reproduction in two ways: improving already-reproduced
methods by composing complementary techniques from the
knowledge base, and synthesizing entirely new methods for
scene types that no published paper has addressed.
4.5.1. Improving Existing Methods
For papers in Set 1 and Set 2, we run a second pass in
which the agents query the knowledge base for techniques
that address deficiencies identified during visual feedback.
The agents receive no instructions about which techniques
to apply; they must independently diagnose weaknesses,
retrieve remedies, and validate that the composition improves
quality. For comparison, we also ran the same loop with the
explorer agents replaced by uniform random sampling over
S. Random sampling used 8 times as many tokens and
reached only a quarter of the PSNR gain.
Table 5 report results. Compositional discovery yields an
improvement of up to +2.4 dB PSNR over the reproduction-
only baseline. The goal is not to invent new primitives but to
identify combinations of existing techniques that have not yet
been explored; since the number of valid compositions grows
quickly with published innovations, manual exploration of
this space is infeasible.
4.5.2. Cross-Domain Method Creation
All prior paper-to-code systems are limited to reproducing
existing literature. We evaluate a qualitatively different capa-
bility: given a scene type that no published 3DGS paper has
addressed, SPLATIFYretrieves physical priors from outside
the 3DGS community and composes them with rendering
knowledge to synthesize a working method from scratch. We
test on three domains where Gaussian primitives are a natu-
ral geometric fit, but standard 3DGS fails due to mismatched
physical assumptions. Both scenarios are human-verified at
the code level. Figure 5 reports results.
GT
3DGS
NebulaGS 
(Ours)
GT
3DGS
BubbleGS 
(Ours)
PSNR: 
31.19
SSIM: 
0.862
LPIPS: 
0.290
FPS: 
26.6
PSNR: 
32.52
SSIM: 
0.914
LPIPS: 
0.241
FPS: 
27.0
PSNR: 
27.81
SSIM: 
0.869
LPIPS: 
0.244
FPS: 
69.1
PSNR: 
26.97
SSIM: 
0.858
LPIPS: 
0.264
FPS: 
99.1Figure 5.Cross-domain comparison.Ground truth, 3DGS, and
agent-generated method for Nebula and Bubble. NebulaGS resolves
opaque shell artifacts; BubbleGS captures iridescent color variation.
Additional scenes in the supplementary.
Scenario 1: Volumetric Nebula Rendering (NebulaGS).
We evaluate on multi-view imagery of a planetary nebula
(157 training, 23 test views). Nebulae are optically thin
emission media whose intensity is a purely additive in-
tegral I(ν) =RL
0jν(s) ds , with no absorption or occlu-
sion. This directly conflicts with standard alpha compositing
C=P
iTiαiciwhere Ti=Q
j<i(1−α j), which drives
opacities toward {0,1} and produces opaque shell artifacts.
Nebular emission also follows Kolmogorov turbulence with
energy spectrum E(k)∝k−5/3, placing significant detail at
high spatial frequencies that pixel-space losses under-weight.
The agents autonomously retrieved this radiative transfer
and turbulence theory and identified six modifications: (i) an
opacity ceiling loss ( α <0.95 ) suppressing opaque shells;
(ii) a per-Gaussian emission parameter decoupling bright-
ness from opacity; (iii) a spectral frequency loss via 2D FFT
log-amplitude matching; (iv) a scale isotropy penalty encour-
aging spherical Gaussians; (v) multi-scale L1 supervision
to capture the Kolmogorov cascade; and (vi) SH degree set
to 0. All modifications are training-time losses requiring no
rasterizer changes. NebulaGS achieves 32.52 dB PSNR and
0.914 SSIM versus 31.19 dB and 0.862 for vanilla 3DGS
(+1.33 dB, +0.052 SSIM), with per-view standard deviation
dropping from 6.29 to 2.62 and rendering speed unaffected.
Scenario 2: Iridescent Thin-Film Rendering (BubbleGS).
Soap bubbles and oil slicks are dominated by thin-film in-
terference. For a film of thickness dand refractive index
nf≈1.33 , the optical path difference is δ= 2n fdcosθ t
where θtfollows Snell’s law nairsinθ i=n fsinθ t. Per-
wavelength reflectance follows the Airy function:
R(λ) =2R0(1−cosϕ)
1 +R2
0−2R 0cosϕ, ϕ=2πδ
λ, (1)
where R0= nf−1
nf+12≈0.02 . The agents evaluate Eq. (1)at
λR=650 nm,λG=550 nm,λB=450 nm and L2-normalize
the result to obtain a physics-predicted unit chrominance
ˆcphys. Standard 3DGS fails because SH cannot represent the
rapid color variation from Eq. (1), opacity binarization cre-
7

Table 4.Novelty coverage on Set 3. C: correct, I: incorrect/partial, M: missing, W: hyperparameter fidelity, Score: weighted semantic
score (0–1). Full results in the supplementary.
Paper2Code Gemini 3.1 Pro DeepSeek 3.2 GPT-5.2 Thinking SPLATIFY
PaperC↑I↓M↓W↑Score↑C↑I↓M↓W↑Score↑C↑I↓M↓W↑Score↑C↑I↓M↓W↑Score↑C↑I↓M↓W↑Score↑
TrickGS [1] 0.35 0.30 0.35 0.40 0.41 0.50 0.20 0.30 0.50 0.51 0.65 0.15 0.20 0.60 0.64 0.60 0.15 0.25 0.65 0.62 0.90 0.05 0.05 0.92 0.94
2DGS [23] 0.85 0.08 0.07 0.50 0.55 0.55 0.15 0.30 0.40 0.40 0.80 0.10 0.10 0.55 0.58 0.75 0.10 0.15 0.60 0.61 1.00 0.00 0.00 0.95 0.95
Scaffold-GS [34] 0.35 0.20 0.45 0.60 0.34 0.45 0.10 0.45 0.45 0.44 0.40 0.35 0.25 0.40 0.42 0.60 0.25 0.15 0.55 0.59 1.00 0.00 0.00 1.00 1.00
GaussianShader [26] 0.45 0.30 0.25 0.40 0.40 0.55 0.28 0.17 0.45 0.50 0.60 0.22 0.18 0.48 0.58 0.72 0.15 0.13 0.62 0.73 1.00 0.00 0.00 0.95 1.00
Pixel-GS [66] 0.25 0.45 0.30 0.60 0.40 0.40 0.10 0.50 0.80 0.46 0.55 0.15 0.30 0.80 0.67 0.65 0.10 0.25 0.85 0.72 1.00 0.00 0.00 1.00 1.00
Table 5.Compositional discovery on Set 1, Set 2 papers.“Before”
is the reproduction-only baseline; “After” is the best validated
configuration found by the discovery pipeline. All numbers on
same scene as Tables 1 and 3.
Before After
Paper PSNR ↑SSIM↑LPIPS ↓#GS PSNR ↑SSIM↑LPIPS ↓#GS Things Added
Opti3DGS [17] 27.71 .841 .209 1.40 28.05 .853 .125 4.95 Adaptive densif.
threshold + opacity
entropy reg.
MSAA 3DGS [55] 25.35 .758 .369 2.73 26.19 .791 .293 2.96 Bilateral grid + losses
GNS [13] 25.97 .891 .095 0.49 26.15 .902 .097 0.61 AA rasterization +
denser init +
post-densif. losses
DWTGS [39] 18.68 .510 .374 1.64 21.09 .725 .281 2.43 Patch structure loss +
opacity/normal/depth
reg.
CompGS [38] 28.28 .894 .153 0.67 28.50 .899 .146 0.63 EMA centroid
updates + grad-based
codebook refine
ates opaque shells where peak reflectance is only 4R0≈8% ,
no geometric prior encourages flat Gaussian arrangements.
The agents composed six modifications: (i) a per-
Gaussian film thickness parameter acting as soft segmen-
tation; (ii) a thin-film chrominance loss penalizing cosine
distance between ˆcphys and rendered chrominance ˆcrender =
c/∥c∥ 2; (iii) an opacity ceiling at α=0.45 for film Gaus-
sians; (iv) a shell flatness loss encouraging disk-like Gaus-
sians; (v) total variation on the thickness map enforcing
smooth drainage gradients; and (vi) L1 sparsity on thickness.
All losses are gated on the learned thickness and require no
rasterizer changes. BubbleGS achieves 27.81 dB PSNR and
0.869 SSIM versus 26.97 dB and 0.858 for vanilla 3DGS,
with LPIPS improving from 0.264 to 0.244 and per-view
standard deviation dropping from 8.23 to 4.85. Rendering
speed decreases from 99.1 to 69.1 FPS due to the second
thickness rasterization pass, though both remain well above
real-time. Both scenarios are human-verified at the code
level. The supplementary video shows SPLATIFYrunning
end to end on a live paper, from PDF to trainable gsplat
plugin, followed by a compositional discovery run.
4.6. Ablation Study
Table 6 shows component ablations averaged over 10 Set 1
papers. CFG constraints have the largest impact on trainabil-
ity (70% without, vs. 100% with), as agents misplace den-
sification logic or wire optimizers incorrectly. Fork-aware
recovery is equally critical: without it, 30% of novelties goTable 6.Component ablation.Impact of each component on
synthesis quality, averaged over 10 papers from SPLATIFYBENCH.
Configuration Score Train. C
(%)
SPLATIFY(Full) 0.97 100 1.00
Knowledge Sources:
w/o CFG Constraints (Stage 1) 0.91 70 0.85
w/o In-context Examples (Stage 1) 0.73 100 1.00
w/o Fork-aware Recovery (Stage 2) 0.64 100 0.70
w/ Simple Citation Recovery 0.79 100 0.90
w/o Both (ICE + Recovery) 0.52 70 0.65
Validation & Feedback:
w/o Smoke Tests (Stage 3) 0.68 50 1.00
w/o Gaussian Struct. Checks (Stage 3) 0.89 80 0.95
w/o VLM Feedback (Stage 4) 0.94 100 1.00
w/o PSNR Curve Monitor (Stage 4) 0.94 100 0.95
Planning Strategy:
One-Shot w/o GoT (GPT 5.2 w/ other novelties) (Stage 3) 0.49 60 0.95
unimplemented. Smoke tests account for the largest single
trainability drop (50%), with Gaussian structural checks re-
sponsible for roughly half. PSNR curve monitoring catches
late-stage failures invisible during short runs. One-shot gen-
eration without GoT collapses to score 0.49 despite high
novelty fidelity, as agents produce monolithic files breaking
dependency ordering.
5. Conclusion
We presented SPLATIFY, a multi-agent framework that con-
verts 3DGS papers into trainable gsplat implementations
through grammar-constrained synthesis, fork-aware citation
recovery, and closed-loop visual feedback. On SPLATIFY-
BENCH, the framework matches expert implementations
within ±0.5 dB PSNR while all baselines fail to produce
trainable code, reducing development time from weeks to
minutes. Compositional discovery yields up to +2.4 dB
PSNR gain, and cross-domain invention produces working
methods for scene types no prior paper has addressed, en-
tirely without human guidance.
Generalizations, limitations, and reproducibility.The
architectural elements of SPLATIFYare not specific to Gaus-
sian Splatting; adapting to a new domain such as 3D re-
construction pipelines or robotics simulation stacks requires
defining a CFG over that framework’s extension points and
curating a seed set of verified paper-code pairs. Code, data,
and all generated implementations will be publicly released.
8

References
[1]Anil Armagan, Albert Sa `a-Garriga, Bruno Manganelli, Ma-
teusz Nowak, and Mehmet Kerim Yucel. Trick-gs: A balanced
bag of tricks for efficient gaussian splatting, 2025. 4, 5, 8
[2]Awesome3DGS. 3D-Gaussian-Splatting-Papers: A curated
list of 3D Gaussian Splatting papers. https://github.
com / Awesome3DGS / 3D - Gaussian - Splatting -
Papers, 2026. Accessed: 2026-06-27. 2
[3]Milena T. Bagdasarian, Paul Knoll, Yi-Hsin Li, Florian
Barthel, Anna Hilsmann, Peter Eisert, and Wieland Mor-
genstern. 3DGS.zip: A survey on 3D Gaussian Splatting
compression methods.Computer Graphics Forum, 44, 2025.
6
[4]Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded
anti-aliased neural radiance fields. InIEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
5470–5479, 2022. 6
[5]Maciej Besta, Nils Blach, Ales Kubicek, Robert Gersten-
berger, Lukas Gianinazzi, Joanna Gajda, et al. Graph of
thoughts: Solving elaborate problems with large language
models. InProceedings of the AAAI Conference on Artificial
Intelligence, pages 17682–17690, 2024. 3
[6]Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang. Pgsr: Planar-based gaussian splatting
for efficient and high-fidelity surface reconstruction.IEEE
Transactions on Visualization and Computer Graphics, 31(9):
6100–6111, 2024. 4
[7]Guikun Chen and Wenguan Wang. A survey on 3D Gaussian
Splatting.arXiv preprint arXiv:2401.03890, 2024. 1, 2
[8]Xinyun Chen, Maxwell Lin, Nathanael Sch ¨arli, and Denny
Zhou. Teaching large language models to self-debug. InIn-
ternational Conference on Learning Representations (ICLR),
2024. 3
[9]Youyu Chen, Junjun Jiang, Kui Jiang, Xiao Tang, Zhihao Li,
Xianming Liu, and Yinyu Nie. Dashgaussian: Optimizing
3d gaussian splatting in 200 seconds. InProceedings of the
Computer Vision and Pattern Recognition Conference, pages
11146–11155, 2025. 5
[10] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin,
Yuexin Ma, Wenping Wang, and Xuejin Chen. GaussianPro:
3d gaussian splatting with progressive propagation. InIn-
ternational Conference on Machine Learning (ICML), 2024.
4
[11] Zhigang Cheng, Mingchao Sun, Yu Liu, Zengye Ge, Luyang
Tang, Mu Xu, Yangyan Li, and Peng Pan. CLoD-GS: Con-
tinuous level-of-detail via 3d gaussian splatting, 2025. 5,
6
[12] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee. Depth-
regularized optimization for 3d gaussian splatting in few-shot
images. InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 811–820,
2024. 5
[13] Xiaobin Deng, Qiuli Yu, Changyu Diao, Min Li, and Duan-
qing Xu. Gradient-driven natural selection for compact 3d
gaussian splatting, 2025. 5, 6, 8[14] Yihong Dong, Ge Li, and Zhi Jin. CODEP: Grammatical
seq2seq model for general-purpose code generation. InPro-
ceedings of the ACM SIGSOFT International Symposium on
Software Testing and Analysis (ISSTA), pages 456–468, 2023.
3
[15] Abdelrhman Elrawy and Emad A. Mohammed. Opacity-
gradient driven density control for compact and efficient few-
shot 3d gaussian splatting, 2025. 5
[16] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia
Xu, and Zhangyang Wang. LightGaussian: Unbounded 3D
Gaussian compression with 15x reduction and 200+ FPS.
arXiv preprint arXiv:2311.17245, 2023. 2, 4
[17] Umar Farooq, Jean-Yves Guillemaut, Graham Thomas,
Adrian Hilton, and Marco V olino. Optimized 3d gaussian
splatting using coarse-to-fine image frequency modulation. In
European Conference on Visual Media Production (CVMP).
ACM, 2025. 5, 6, 8
[18] Ben Fei, Jingkang Xu, Rui Zhang, Qiang Zhou, Weidong
Yang, and Ying He. 3D Gaussian Splatting as new era: A
survey.IEEE Transactions on Visualization and Computer
Graphics, 2024. 1, 2
[19] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. EA-
GLES: Efficient accelerated 3D Gaussians with lightweight
encodings. InEuropean Conference on Computer Vision
(ECCV), 2024. 4
[20] Glenn Grubert, Florian Barthel, Anna Hilsmann, and Peter
Eisert. Improving adaptive density control for 3d gaussian
splatting.arXiv preprint arXiv:2503.14274, 2025. 5
[21] Liang Han, Junsheng Zhou, Yu-Shen Liu, and Zhizhong Han.
Binocular-guided 3d gaussian splatting with view consistency
for sparse view synthesis. InAdvances in Neural Information
Processing Systems (NeurIPS), 2024. 5
[22] Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng,
Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang,
Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu
Ran, Lingfeng Xiao, Chenglin Wu, and J ¨urgen Schmidhuber.
MetaGPT: Meta programming for a multi-agent collabora-
tive framework. InInternational Conference on Learning
Representations (ICLR), 2024. 2
[23] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2D Gaussian Splatting for geometrically ac-
curate radiance fields. InACM SIGGRAPH 2024 Conference
Papers. ACM, 2024. 2, 3, 5, 8
[24] Binxiao Huang, Zhengwu Liu, and Ngai Wong. Decompos-
ing densification in gaussian splatting for faster 3d scene
reconstruction.arXiv preprint arXiv:2507.20239, 2025. 5, 6
[25] Seemandhar Jain, Keshav Gupta, Kunal Gupta, and Man-
mohan Chandraker. Nerfify: A multi-agent framework for
turning nerf papers into code, 2026. 2, 3, 6
[26] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaox-
iao Long, Wenping Wang, and Yuexin Ma. GaussianShader:
3D Gaussian Splatting with shading functions for reflective
surfaces. InIEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2024. 5, 8
[27] Jaewoo Jung, Jisang Han, Honggyu An, Jiwon Kang,
Seonghoon Park, and Seungryong Kim. RAIN-GS: Relaxing
accurate initialization constraint for 3D Gaussian Splatting.
arXiv preprint arXiv:2403.09413, 2024. 4
9

[28] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk ¨uhler, and
George Drettakis. 3D Gaussian Splatting for real-time radi-
ance field rendering.ACM Transactions on Graphics, 42(4):
1–14, 2023. 1, 2, 4, 5, 6
[29] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Jeff Tseng, Hossam Isack, Abhishek Kar, Andrea
Tagliasacchi, and Kwang Moo Yi. 3D Gaussian Splatting as
Markov Chain Monte Carlo. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS), 2024. 2, 3, 5
[30] Jonas Kulh ´anek and Torsten Sattler. NerfBaselines: Con-
sistent and reproducible evaluation of novel view synthesis
methods.arXiv preprint arXiv:2406.17345, 2024. 2
[31] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and
Eunbyung Park. Compact 3D Gaussian representation for
radiance field. InIEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 2024. 2, 5
[32] Jee Won Lee, Hansol Lim, Sooyeun Yang, and
Jongseong Brad Choi. Micro-splatting: Multistage
isotropy-informed covariance regularization optimization for
high-fidelity 3d gaussian splatting, 2025. 5, 6
[33] Zijie Lin, Yiqing Shen, Qilin Cai, He Sun, Jinrui Zhou, and
Mingjun Xiao. AutoP2C: An LLM-based agent framework
for code repository generation from multimodal content in
academic papers.arXiv preprint arXiv:2504.20115, 2025. 2,
6
[34] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang,
Dahua Lin, and Bo Dai. Scaffold-GS: Structured 3D Gaus-
sians for view-adaptive rendering. InIEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2024.
Highlight. 4, 5, 8
[35] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hal-
linan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri,
Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iter-
ative refinement with self-feedback. InAdvances in Neural
Information Processing Systems (NeurIPS), 2023. 3
[36] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl,
Markus Steinberger, Francisco Vicente Carrasco, and Fer-
nando De La Torre. Taming 3dgs: High-quality radiance
fields with limited resources. InSIGGRAPH Asia 2024 Con-
ference Papers, pages 1–11, 2024. 4
[37] Wieland Morgenstern, Florian Barthel, Anna Hilsmann, and
Peter Eisert. Compact 3d scene representation via self-
organizing gaussian grids. InEuropean conference on com-
puter vision, pages 18–34. Springer, 2024. 4
[38] KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Ab-
basi Koohpayegani, and Hamed Pirsiavash. Compgs: Smaller
and faster gaussian splatting with vector quantization. In
European Conference on Computer Vision, pages 330–349.
Springer, 2024. 5, 6, 8
[39] Hung Nguyen, Runfa Li, An Le, and Truong Nguyen. Dwtgs:
Rethinking frequency regularization for sparse-view 3d gaus-
sian splatting, 2025. 5, 6, 8
[40] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke
Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qin-
tong Zhang, Zhenjiang Jin, Guang Liang, Rui Zhang, Wen-
zheng Zhang, Yuan Qu, Zhifei Ren, Yuefeng Sun, Yuanhong
Zheng, Dongsheng Ma, Zirui Tang, Boyu Niu, Ziyang Miao,Hejun Dong, Siyi Qian, Junyuan Zhang, Jingzhou Chen, Fang-
dong Wang, Xiaomeng Zhao, Liqun Wei, Wei Li, Shasha
Wang, Ruiliang Xu, Yuanyuan Cao, Lu Chen, Qianqian Wu,
Huaiyu Gu, Lindong Lu, Keming Wang, Dechen Lin, Guanlin
Shen, Xuanhe Zhou, Linfeng Zhang, Yuhang Zang, Xiaoyi
Dong, Jiaqi Wang, Bo Zhang, Lei Bai, Pei Chu, Weijia Li,
Jiang Wu, Lijun Wu, Zhenxiang Li, Guangyu Wang, Zhongy-
ing Tu, Chao Xu, Kai Chen, Yu Qiao, Bowen Zhou, Dahua
Lin, Wentao Zhang, and Conghui He. Mineru2.5: A de-
coupled vision-language model for efficient high-resolution
document parsing, 2025. 4
[41] Mateusz Nowak, Wojciech Jarosz, and Peter Chin. V oD-
3DGS: View-opacity-dependent 3d gaussian splatting, 2025.
5
[42] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian:
Structural regularization for sparse-view gaussian splatting.
InProceedings of the computer vision and pattern recognition
conference, pages 21600–21609, 2025. 5
[43] St´ephane Pateux, Matthieu Gendrin, Luce Morin, Th ´eo
Ladune, and Xiaoran Jiang. BOGausS: Better optimized
gaussian splatting, 2025. 5, 6
[44] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang,
Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong,
Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. Chat-
Dev: Communicative agents for software development. In
Annual Meeting of the Association for Computational Lin-
guistics (ACL), 2024. 2
[45] Bernardino Romera-Paredes, Mohammadamin Barekatain,
Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien
Dupont, Francisco J. R. Ruiz, Jordan Ellenberg, Pengming
Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi.
Mathematical discoveries from program search with large
language models.Nature, 2023. 5
[46] Minju Seo, Jinheon Baek, Seongyun Lee, and Sung Ju Hwang.
Paper2code: Automating code generation from scientific pa-
pers in machine learning.arXiv preprint arXiv:2504.17192,
2025. 2, 5, 6
[47] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik
Narasimhan, and Shunyu Yao. Reflexion: Language agents
with verbal reinforcement learning. InAdvances in Neural
Information Processing Systems (NeurIPS), 2023. 3
[48] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung,
Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays,
Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke,
Amelia Glaese, and Tejal Patwardhan. Paperbench: Evalu-
ating AI’s ability to replicate AI research. InInternational
Conference on Machine Learning (ICML), 2025. 2, 5
[49] Ashkan Taghipour, Vahid Naghshin, Benjamin Southwell,
Farid Boussaid, Hamid Laga, and Mohammed Bennamoun.
Svr-gs: Spatially variant regularization for probabilistic
masks in 3d gaussian splatting, 2025. 5
[50] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent
Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake
Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, and
Angjoo Kanazawa. Nerfstudio: A modular framework for
neural radiance field development. InACM SIGGRAPH Con-
ference Proceedings, 2023. 2
10

[51] Bailin Wang, Zi Wang, Xuezhi Wang, Yuan Cao, Rif A.
Saurous, and Yoon Kim. Grammar prompting for domain-
specific language generation with large language models.
InAdvances in Neural Information Processing Systems
(NeurIPS), 2023. 3
[52] Henan Wang, Hanxin Zhu, Tianyu He, Runsen Feng, Jiajun
Deng, Jiang Bian, and Zhibo Chen. End-to-end rate-distortion
optimized 3d gaussian representation. InEuropean Confer-
ence on Computer Vision, pages 76–92. Springer, 2024. 4
[53] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yun-
zhu Li, Hao Peng, and Heng Ji. Executable code actions elicit
better LLM agents. InInternational Conference on Machine
Learning (ICML), 2024. 2
[54] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin
Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang,
Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug
Burger, and Chi Wang. AutoGen: Enabling next-gen LLM
applications via multi-agent conversation. InConference on
Language Modeling (COLM), 2024. 2
[55] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee.
Multi-scale 3d gaussian splatting for anti-aliased rendering.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pages 20923–20931,
2024. 5, 8
[56] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian
Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press.
SWE-agent: Agent-computer interfaces enable automated
software engineering. InAdvances in Neural Information
Processing Systems (NeurIPS), pages 50528–50652, 2024. 2
[57] Sooyeun Yang, Cheyul Im, Jee Won Lee, and Jongseong Brad
Choi. Tidi-gs: Floater suppression in 3d gaussian splat-
ting for enhanced indoor scene fidelity.arXiv preprint
arXiv:2601.09291, 2026. 5
[58] Ziyi Yang, Xinyu Gao, Yang-Tian Sun, Yi-Hua Huang, Xi-
aoyang Lyu, Wen Zhou, Shaohui Jiao, Xiaojuan Qi, and
Xiaogang Jin. Spec-gaussian: Anisotropic view-dependent
appearance for 3d gaussian splatting.Advances in Neural
Information Processing Systems, 37:61192–61216, 2024. 4, 5
[59] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L.
Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of
thoughts: Deliberate problem solving with large language
models. InAdvances in Neural Information Processing Sys-
tems (NeurIPS), 2023. 3
[60] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent
Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu,
Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-
source library for Gaussian Splatting.Journal of Machine
Learning Research, 26:1–15, 2025. 2
[61] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou.
Absgs: Recovering fine details for 3D Gaussian Splatting. In
ACM International Conference on Multimedia (ACM MM),
2024. 2
[62] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-Splatting: Alias-free 3D Gaussian Splat-
ting. InIEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), 2024. Best Student Paper. 2, 3
[63] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction inunbounded scenes.ACM Transactions on Graphics, 43(6):
1–13, 2024. 5
[64] Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and
Eric Xing. FreGS: 3D Gaussian Splatting with progressive
frequency regularization. InIEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2024. 5
[65] Yunzhi Zhang, Zizhang Li, Matt Zhou, Shangzhe Wu, and
Jiajun Wu. The scene language: Representing scenes
with programs, words, and embeddings.arXiv preprint
arXiv:2410.16770, 2024. 3
[66] Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and Heng-
shuang Zhao. Pixel-GS: Density control with pixel-aware
gradient for 3D Gaussian Splatting. InEuropean Conference
on Computer Vision (ECCV), 2024. 5, 8
[67] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang,
Cheng Peng, Rama Chellappa, and Deliang Fan. Lp-3dgs:
Learning to prune 3d gaussian splatting.Advances in Neural
Information Processing Systems, 37:122434–122457, 2024.
5, 6
[68] Xuanle Zhao, Zilin Sang, Yuxuan Li, Qi Shi, Weilun
Zhao, Shuo Wang, Duzhen Zhang, Xu Han, Zhiyuan Liu,
and Maosong Sun. Autoreproduce: Automatic AI exper-
iment reproduction with paper lineage.arXiv preprint
arXiv:2505.20662, 2025. 2
[69] Mingyang Zhou, Quanming Yao, Lun Du, Lanning Wei, and
Da Zheng. Reflective paper-to-code reproduction enabled by
fine-grained verification.arXiv preprint arXiv:2508.16671,
2025. 2
11