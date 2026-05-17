# RFAmpDesigner: A Self-Evolving Multi-Agent LLM Framework for Automated Radio Frequency Amplifier Design

**Authors**: Hang Lu, Guochang Li, Qianyu Chen, Huiyan Gao, Shaogang Wang, Xuanyu He, Yiwei Liu, Gaopeng Chen, Nayu Li, Xiaokang Qi, Chunyi Song, Zhiwei Xu

**Published**: 2026-05-11 07:11:03

**PDF URL**: [https://arxiv.org/pdf/2605.10093v1](https://arxiv.org/pdf/2605.10093v1)

## Abstract
Automating radio frequency (RF) amplifier design remains challenging because existing methods suffer from the curse of dimensionality, weak use of domain knowledge, and poor transferability, leading to low data efficiency. Meanwhile, although large language models (LLMs) have shown promise in many scientific domains, applying them directly to RF sizing is nontrivial due to the numerical nature of circuit optimization and the reliance on domain-specific design flows. To address this, this paper proposes RFAmpDesigner, a multi-agent framework that automates RF amplifier sizing. It introduces a resource-allocation middleware that reframes high-dimensional parameter tuning as a low-dimensional resource distribution problem, making it easier to inject sizing knowledge into general-purpose LLMs. The framework also follows standard design practice, enabling LLMs to distinguish between high- and low-cost actions and search in parallel. To realize a self-evolving optimization process, the framework employs retrieval-augmented generation (RAG) to reuse past knowledge and experience from memory base. As a proof of concept, we apply RFAmpDesigner to low noise amplifiers of varying complexity. The experimental results show that it can automatically synthesize designs with fractional bandwidths ranging from 10\% to 80\% and center frequencies from 10 GHz to 50 GHz. To the best of our knowledge, this work develops the first LLM-driven approach for RF amplifier sizing that operates on design concepts instead of treating netlists as text, offering a novel solution to mitigate data scarcity in RF design.

## Full Text


<!-- PDF content starts -->

1
RFAmpDesigner: A Self-Evolving Multi-Agent LLM Framework for
Automated Radio Frequency Amplifier Design
Hang Lu, Guochang Li, Qianyu Chen, Huiyan Gao, Shaogang Wang,Graduate Student Member, IEEE
Xuanyu He, Yiwei Liu, Gaopeng Chen, Nayu Li, Xiaokang Qi, Chunyi SongMember, IEEEand
Zhiwei Xu,Senior Member, IEEE
Abstract—Automating radio frequency (RF) amplifier design
remains challenging because existing methods suffer from the
curse of dimensionality, weak use of domain knowledge, and
poor transferability, leading to low data efficiency. Meanwhile,
although large language models (LLMs) have shown promise in
many scientific domains, applying them directly to RF sizing is
nontrivial due to the numerical nature of circuit optimization and
the reliance on domain-specific design flows. To address this, this
paper proposes RFAmpDesigner, a multi-agent framework that
automates RF amplifier sizing. It introduces a resource-allocation
middleware that reframes high-dimensional parameter tuning
as a low-dimensional resource distribution problem, making it
easier to inject sizing knowledge into general-purpose LLMs.
The framework also follows standard design practice, enabling
LLMs to distinguish between high- and low-cost actions and
search in parallel. To realize a self-evolving optimization process,
the framework employs retrieval-augmented generation (RAG)
to reuse past knowledge and experience from memory base.
As a proof of concept, we apply RFAmpDesigner to low noise
amplifiers of varying complexity. The experimental results show
that it can automatically synthesize designs with fractional
bandwidths ranging from 10% to 80% and center frequencies
from 10 GHz to 50 GHz. To the best of our knowledge, this work
develops the first LLM-driven approach for RF amplifier sizing
that operates on design concepts instead of treating netlists as
text, offering a novel solution to mitigate data scarcity in RF
design.
Index Terms—Agentic workflow, design automation, large
language models, radio frequency integrated circuits
I. INTRODUCTION
THE chip industry relentlessly pursues both peak perfor-
mance and maximal efficiency. As a result, the demand
This paragraph of the first footnote will contain the date on which you
submitted your paper for review. This work was supported by the Donghai
Laboratory (Grant no. DH-2023QD0004), by the National Natural Science
Foundation of China, Key Project No.62434008, by the National Key Research
and Development Program of China under Grant No. 2023YFB4403304,
supported by the Fundamental Research Funds for the Central Universities
226-2025-00003. (Corresponding authors: Chunyi Song; Zhiwei Xu.)
Hang Lu, Qianyu Chen, Huiyan Gao, Shaogang Wang, Xuanyu He,
Yiwei Liu, Gaopeng Chen, Xiaokang Qi and Zhiwei Xu are with the
State Key Laboratory of Ocean Sensing (& Institute of Fundamental and
Transdisciplinary Research, the Institute of Marine Electronics and Intelligent
Systems, Ocean College,), Zhejiang University, Zhoushan, 316021, China,
and also with the Engineering Research Center of Oceanic Sensing Technol-
ogyand Equipment, Ministry of Education, Zhoushan 316021, China (email:
xuzw@zju.edu.cn).
Nayu Li and Chunyi Song are with the Donghai Laboratory, Zhoushan
316021, China, and also with the State Key Laboratory of Ocean Sensing
(& Institute of Fundamental and Transdisciplinary Research, the Institute
of Marine Electronics and Intelligent Systems, Ocean College,), Zhejiang
University, Zhoushan, 316021, China (e-mail: linayu@zju.edu.cn).
Guochang Li is with College of Computer Science, Zhejiang University,
Hangzhou, China (e-mail: gcli@zju.edu.cn).for electronic design automation (EDA) has never ceased,
either to broaden the design space in pursuit of peak per-
formance or to enhance iteration efficiency and minimize
time-to-market. At present, EDA tools for digital and analog
circuit design have reached a relatively mature stage. However,
automation for radio frequency (RF) circuit design remains in
its early development phase [1].
This gap arises from two principal challenges: parameter
sizing and passive component modeling. First, the design of
RF circuits, particularly wideband implementations, requires
achieving multi-objective trade-offs while maintaining the
desired operating bandwidth. Compared to digital and low-
frequency analog circuits, this inherent complexity results
in a sparse distribution of feasible solutions. Second, the
high operating frequencies cause the physical dimensions
of passive components to approach the signal wavelength,
leading to pronounced distributed effects and parasitic loss.
Simulation models in standard process design kit (PDK) often
fail to maintain accuracy under such conditions, compelling
designers to rely on time-consuming full-wave electromagnetic
(EM) simulations. Taken together, these challenges necessitate
that EDA algorithms navigate a sparsely populated solution
space, requiring extensive exploration through iterative and
computationally intensive electromagnetic and circuit simula-
tions to identify feasible designs. This significantly increases
the complexity and computational burden associated with de-
veloping robust EDA methodologies for RF circuit design. The
subsequent discussion elaborates on prior efforts addressing (i)
active parameter sizing and (ii) passive-component physical
design.
To address the first active parameter sizing challenge,
existing approaches can be categorized into two categories:
optimization-based methods [2]–[5] and learning-based meth-
ods [6]–[9]. Commercial tools such as Keysight ADS [10]
integrate several gradient-based and stochastic search algo-
rithms (e.g. simulated annealing, particle swarm) to facilitate
the circuit design process. Population-based methods, such
as nondominated sorting genetic algorithm (NSGA-II), have
been employed to optimize each sub-block individually and
assemble pareto-optimal fronts (POFs) in a bottom-up manner
[2], [3]. Bayesian optimization [5] treats circuit sizing as an
expensive black-box function and employs Gaussian Processes
(GP) to construct surrogate models to reduce computation cost.
On the learning side, neural networks have been trained to
automate physical parameter sizing of transformers [7] and
distributed circuits [8]. Recent reinforcement learning (RL)
approaches employ graph embeddings to represent circuitsarXiv:2605.10093v1  [cs.AR]  11 May 2026

2
Gradient 
Based
Population 
BasedSurrogate Model
Based
Actor -Critic Agent
Language Model Tools
POFOptimization RL LLM
Isolated Data Points
 Design Trajectory Design Trajectory
High -dimensional search
Norefinemen t
Start from scratch
No human involved
High -dimensional search
Iterative refinemen t
No more costly training
No human involved
Low-dimensional sub -
problems by using tools
Iterative refinemen t
Adapt over time
Copilot with human
Fig. 1. Conceptual overview of widely adopted methodologies in RF circuits
design and this work.
and learn policies that refine designs iteratively [9]. Despite
these advances, both categories typically rely on extensive
simulations or dedicated datasets, which limit their scalability
and adaptability in practical RF design.
As for the second challenge of physical design of passive
microwave component, various methods have been proposed
since the 1960s [11]. Among these, techniques that have
stood the test of time include space mapping [12] and neural
networks [13]. Space mapping is an optimization method-
ology that aims to achieve a satisfactory solution with a
minimal number of computationally expensive high-fidelity
model evaluations. It iteratively refines low-fidelity surrogate
models with simulations through parameter mapping. Re-
cently, various variants of neural network, such as feedforward
neural networks (FNN) [14], recurrent neural networks (RNN)
[15], and convolutional neural networks (CNN) [16], have
been used to construct surrogate models for critical passive
components, including on-chip inductors [17], input/output
matching networks in power amplifiers [18], [19], filters [20]
and antennas [21]. Overall, surrogate-based approaches have
proven effective in replacing EM solvers for modeling indi-
vidual passive components. Nevertheless, the mapping from
physical dimensions to electrical performance is inherently
non-injective, which prevents direct physical realization from
performance specifications. As a result, optimization- or RL-
based pipelines remain indispensable for completing the phys-
ical design.
The limited efficiency of existing methods call for ap-
proaches that can leverage domain knowledge, perform com-
plex reasoning, and transfer across tasks. Large language
models (LLMs) naturally possess these capabilities and have
already transformed a range of scientific domains [22]. To fur-
ther substantiate the necessity of applying LLMs to RF design
automation, a systematic comparison among optimization-
based, RL-based, and LLM-driven methods is provided in
Fig. 1. Their advantages and disadvantages are summarized
at the bottom along the four dimensions shown below.
Dimension 1: Search Dimensionality. Conventional op-
timization and RL methods directly search the full high-
dimensional parameter space, where the curse of dimension-ality impairs algorithmic efficiency and reward acquisition.
In contrast, LLMs leverage tools to exploit human sizing
expertise and reformulate the task as a hierarchy of lower-
dimensional subproblems.
Dimension 2: Candidate Refinement. Optimization meth-
ods typically generate either isolated solutions or Pareto fronts,
with no mechanism for refining prior candidate solution us-
ing domain expertise. Although RL-based methods generate
sequential trajectories, these are rarely interpretable to hu-
man designers. Compared with them, LLM-driven approaches
leverage domain expertise to augment each decision with
explicit reasoning, thereby enhancing both interpretability and
design professionalism.
Dimension 3: Knowledge Transferability. Optimization
approaches solve each task from scratch, with no mechanism
to retain knowledge. RL partially alleviates this by embed-
ding knowledge in value or policy networks, yet changes in
design-space dimensionality still demand extensive retraining.
Whereas, LLM-driven methods employ memory modules to
recall prior solutions and learns from human-provided circuit-
level reasoning examples how to reuse them for new designs,
thereby offering a degree of generalization even under varia-
tions in the solution space dimensionality.
Dimension 4: Human-in-the-loop. Optimization and RL
methods run in closed loops without human intervention, leav-
ing domain knowledge underutilized and often incurring long
runtime even near feasible designs. On the other hand, LLM-
based approaches naturally support user interaction, enabling
the direct incorporation of human expertise.
In summary, the four-dimension comparison highlights that
LLM-based methods offer clear advantages over optimization
and RL approaches in terms of dimension reduction, knowl-
edge utilization, and transferability. Consequently, recent stud-
ies have explored their use in EDA tasks, which can be broadly
categorized into two main directions: topology synthesis and
parameter sizing. Specifically, topology synthesis works utilize
the generative capabilities of LLMs to produce circuit netlists,
which typically require subsequent optimization to strictly
meet user specifications [23]–[29]. In contrast, parameter
sizing approaches leverage the reasoning capabilities of LLMs
to assist optimization algorithms by refining search spaces
[30]–[33]. In addition, other studies have investigated LLMs as
copilots for tasks such as layout design [34]. However, current
approaches have two common limitations for our setting. First,
RF design is seldom addressed, where frequency shaping
increases the effective design dimensionality and rewards are
often sparse, making optimization more difficult. Second,
direct manipulation of netlists by the LLM is required, which
can hinder the practical deployment of smaller models.
When addressing the highly specialized task of RF amplifier
parameter sizing, relying solely on the internal knowledge of
LLMs and generic multi-step reasoning proves insufficient.
Three challenges underlie this limitation: (i) the parameter
space is high-dimensional and strongly coupled, a complex-
ity that even human experts struggle to fully interpret and
write accurate prompts to guide LLMs; (ii) circuit design
is inherently an optimization problem, where sparse domain
knowledge is in numerical format which cannot be effec-

3
Active Sizing
[Stage 1]
Load&Source
Pull [Stage 2]
Search Tool
Band Planning
[Stage 3]
Run Simulation
[Stage 4]
Refine ToolTool Middleware[Sec. II -B]
Design results
(netlist&
simulation)Design workflow
Agent decide 
the current 
and gain 
distribution of 
each stages
text
Active 
sizing 
and 
biasingActive 
parameters
Impedance 
matching Active 
parameters
&passive 
parameters 
(partial)
Active 
parameters
&passive 
parameters
Full chain 
evaluationNetlist and 
simulation 
results
Meet 
goal?Y
N
Details of 
simulationBand 
Planning
Design goal: 
Load R and X,
Power Budget,
fL~fH , Gain,
NF, IP1
Pcell and Netlist 
CreationTopology &Knowledge Preparation[Sec .II-A]
Python+Ocean to run 
test bench
Power -Performance 
Database 
 Candidate QueueRFAmpManager
Memory Base[KB and EB]
RAGStore
Pick and Store 
Candidates
Candidates 
𝑿∈𝑹𝟑𝟎
Two-Tier Multi -Agent Framework[Sec. II -C]
RFAmpSeacher
ReasoningAction
Observation
RFAmpRefiner
Reasoning
Refine 
summary
Action
Observation…Multiple reasoning paths
Mode 1 Search Refine
MemoryRefine
Search …Refine
Memory Search… RefineMode 2
Fig. 2. (up) The proposed design workflow of LLM. (down) Architecture of RFAmpDesigner: The workflow starts with the Topology and knowledge
preparation (yellow block, Sec. II-A), which provides the initial design context. It then proceeds to the Resource-allocation multi-fidelity tool middleware
(green block, Sec. II-B), which manages tool calls for agents. Finally, the Two-Tier multi-agent framework (blue block, Sec. II-C) iteratively uses tools and
simulation feedbacks to guide the sizing process.
tively encoded into model parameters; and (iii) in expert-
intensive domains, well-established design pipelines already
exist, making it impractical for LLMs to start from scratch
and autonomously orchestrate efficient search paths.
To address these challenges, this paper makes the following
core contributions:
1) This paper proposes RFAmpDesigner, a self-evolving
multi-agent framework that represents the first LLM-
driven approach that performs specification-driven RF
amplifier parameter sizing via a resource-allocation tool
middleware without operating directly on netlists.
2) RFAmpDesigner introduces, for the first time, a novel
tool middleware that abstracts circuit optimization as
resource allocation to reduce dimension, embedding
domain-specific sizing knowledge into general LLMs,
enabling LLMs of different sizes to be applied directly
in RF design automation.
3) RFAmpDesigner consists of three agents, collaborating
in a two-tier workflow aligned with established circuit
design pipelines, which enables parallel execution and
efficient searching and refining.
4) RFAmpDesigner employs retrieval-augmented genera-
tion (RAG) to reuse past knowledge and experience,
enhancing the efficiency of data utilization during self-
evolving optimization.
The rest of this paper is organized as follows. Section II
demonstrates the framework of RFAmpDesigner. Section III
presents the experimental results. Section IV concludes the
paper.Candidate X= [X 1,X2,1,X2,2,X2,3,X3]∈R30
X1= [W 1,W2,W3,VGS,1,VGS,2,VGS,3]
X2,i= [k 1,i, L1,i, L2,i, R1,i, R2,i, C1,i, C2,i]
X3= [L par,Lg,Ls]
VB2VB3 Lg
LsLparVB1VG1
VDDRFIN
RLCL
Fig. 3. Topology used in experiments for proof of concept.
II. FRAMEWORK OFRFAMPDESIGNER
As shown in the upper part of Fig. 2, we propose a design
workflow that enables an LLM to tune circuit parameters in
a human-like manner, i.e., by iteratively making concept-level
decisions rather than directly manipulating high-dimensional
netlist variables. The LLM outputs textual design intents,
which are progressively compiled by our tool middleware
(highlighted in the green block) into an executable netlist.
The generated netlist is then simulated, and the resulting
detailed performance report is fed back to the LLM to support
reflection and subsequent refinement. Concretely, the overall
framework comprises three core modules:
Topology and knowledge preparation. Once topology and
process technology are specified, the corresponding scripts,
prompts, and expert knowledge are fixed, providing the foun-

4
TABLE I
DESIGNTOOLSOVERVIEW.
Name Format Documentation
Active sizing
& biasingInput
<power_ratio_list>Inputs a list of power ratios of each stages, e.g. [0.4, 0.3, 0.3] means that the first, second and third
stages use 40 percents, 30 percents and 30 percents of the given power constraints respectively.
Output
<active_params_dict_list>Outputs all possible combinations of active parameters in a list. Each item is a dictionary containing
the width, bias voltage, transconductance, dc current, and input/output impedance of each stages.
Impedance
matchingInput
<active_params_dict>
[NF_headroom]
[Gain_require]Takes a list of active parameters and performs source and load matching for critical performance
stages to satisfy user-specified performance requirements. Optionally accepts NF headroom and
Gain require to constrain noise figure and gain.
Output
<passive_params_cpstages>
<sim_result_cpstages>Optimizes according to user-specified requirements, gain constraints, and headroom. Returns the
optimized passive parameters of the critical performance stages together with the corresponding
simulation results.
Band
planningInput
<active_params_dict>
<gain_list>
[Gain_req_list]Uses a low-fidelity computation method based on the positions of interleaved peaks to achieve the re-
quired gain and ripple within the specified operating bandwidth. Optionally accepts Gain require list
to constrain gain distribution of each stages.
Output
<passive_params_dict>
<cal_gain_dict>After completing optimization, it outputs the passive parameter dictionary of the remaining stages
together with the low-fidelity theoretical gain dictionary.
Fullchain
evaluationInput
<wholechain_active_dict>
<wholechain_passive_dict>Receives a circuit dictionary, invokes the simulator to perform full-chain evaluation.
Output
<sim_results_dict>After completing simulation, it returns the true simulation results for both small-signal and large-
signal analyses.
*Required arguments are enclosed in<>and optional arguments are in []
dation for subsequent tool and agent construction.
Resource-allocation multi-fidelity tool middleware. Act-
ing as an abstraction layer between the LLM and commercial
EDA tools, the middleware reformulates RF amplifier design
as a resource-allocation problem—distributing current across
active devices and gain across passive networks via two design
tools:Search ToolandRefine Tool. This formulation reduces
problem dimensionality enables models of varying scales to
function effectively within this vertical domain.
Multi-agent framework. This part introduces two oper-
ational modes of RFAmpDesigner:Autonomous Searchand
Self-Evolving Retrieve-and-Refine. Three agents collaborate in
alignment with human design pipelines to balance exhaustive
searching with lightweight refinement, thereby meeting user
queries, summarizing and reusing past experience.
A. Topology and Knowledge Preparation
To verify the framework, a classic LNA topology is em-
ployed as the design target, as shown in Fig. 3. The active com-
ponents are first considered. In the design of multi-stage RF
amplifiers, a primary challenge stems from the circuit’s non-
ideal unilateral characteristics, which cause coupling effects
whereby modifications in one stage inevitably influence the
behavior of others. This interdependence substantially compli-
cates the design process. To alleviate such effects, differential
common-source amplifiers with neutralization capacitors have
become a classic choice in microwave and millimeter-wave ap-
plications. For receiver front-ends, the noise figure (NF) is the
dominant performance metric. To balance noise performance
with input matching, source-degeneration inductance is com-
monly employed to provide local negative feedback. The first
stage adopts a single-ended cascode topology, which obviates
the need for a broadband balun in the input network, thereby
avoiding noise degradation and reducing power consumption.The discussion now turns to the passive components. For
a given allowable ripple, wider bandwidth requires higher-
order passive networks, which typically incurs greater losses
at microwave and millimeter-wave frequencies. Among the
fourth-order coupled-resonator networks commonly used to
interconnect two gain stages, magnetically coupled resonators
(MCRs) have gained particular traction due to their compact
footprint and inherent support for both dc biasing and ac
coupling [35].
To construct design tools, an expert knowledge database
about active device is constructed based on the TSMC N65
PDK, shown in the yellow block of Fig. 2 at the bottom
right. Cadence parameterized cells are developed for differ-
ential common-source stages with neutralization capacitors
and single-ended cascode stages, enabling batch layout gen-
eration. Calibre DRC/LVS/PEX are then employed to ex-
tract layout parasitics and generate post-layout netlists. To
unify device characterization with respect to power consump-
tion, automated OCEAN and shell scripts are developed for
batch circuit simulations, yielding lookup tables that record
power consumption across device sizes and bias voltages. The
same methodology is further applied to capture input/output
impedances and equivalent transconductance when driving a
50 Ωload for later use. The final dataset encompasses cascode
devices with dimensions ranging from45µmto180µmin
9µmincrements, differential common-source devices from
45µmto117µmin9µmsteps and bias voltages from300 mV
mV to500 mVin25 mVincrements.
B. Resource-allocation Multi-fidelity Tool Middleware
Consider a practical case of designing the three-stage LNA
shown in Fig. 3, where the design space spans 30 parame-
ters. To minimize noise, a human designer would typically

5
TABLE II
VARIABLE RANGES FORSTAGE2ANDSTAGE3OPTIMIZATION
Stage 2 Stage 3
Variables Ranges Variables Ranges
k[0.1,0.7] k[0.1,0.8]
L1(pH) [100,1000] ω0(GHz) [ω L, ωH]
L2(pH) [100,1000] Q0 [0.1,8]
R1(Ω) [100,1000] C(fF) [C 2,200]
R2(Ω) [100,1000]
C1(fF) [0,50]
C2(fF) [0,50]
Ls(pH) [0,150]
V
11I11Y
k
2C 2R Cs RsIs
BV
Fig. 4. The lumped model of MCR.
perform load-pull and source-pull analyses on the first stage
to determine the optimal impedance conditions. If linearity
constraints are also imposed, gain must be redistributed across
stages through the joint adjustment of both active and passive
components. This example illustrates a central difficulty in
RF design: human sizing expertise is concept-driven, often
involving simultaneous adjustments of coupled parameters. Yet
parameter–performance relationships are inherently numerical,
and direct manipulation of such values is a weakness of LLMs,
making them difficult to express as effective prompts.
To address these difficulties, a novel tool middleware is
developed grounded in circuit design principles, shown in the
green block of Fig. 2 at the middle right. The middleware de-
couples the problem and reformulates it as a Six-dimensional
resource allocation problem. This abstraction allows domain
knowledge and design heuristics to be encoded as deterministic
rules and illustrative examples, which can then be leveraged by
LLMs. To ensure transparency and reproducibility, documen-
tation for the complete toolset has been developed. As shown
in Table I, it provides a systematic overview of each tool’s
functionality together with its input–output specifications. In
the following sections, a stage-by-stage explanation of how the
middleware transforms this abstract formulation into concrete
circuit parameters is presented.
1) Stage 1:Active Device Sizing and Bias Selection
The first stage focuses on determining transistor widths and
bias voltages. The tool takes as input a current allocation list
specifying the fraction of the total current budget assigned
to each active stage. Based on this budget and the allocation
ratios provided by the LLM agent, the allowable current
for each stage is computed. The corresponding device size-
voltage pairs are then retrieved from the aforementioned expert
knowledge database. For a given current, multiple realizations
may exist, such as a large transistor with low bias or a
small transistor with high bias. By enumerating all possible
combinations across stages, the tool generates the feasible
active device configurations corresponding to the specified
current allocation.In RF amplifiers, different performance metrics are primar-
ily determined by different stages. Take LNA for example, the
noise figure is predominantly determined by the first stage.
Other metrics, such as linearity, tend to exhibit a loosely
proportional relationship with transistor size and bias. To
reduce computational complexity, the tool categorizes active
stages into critical and non-critical groups. For each group,
only two extreme configurations—the largest and the smallest
device realizations—are retained. Consequently, each current
allocation produces four representative candidate active device
configurations.
2) Stage 2:Impedance Matching for Critical Perfor-
mance Stages
The second stage focuses on impedance matching for
critical performance stages. In LNA, the tool employs a
particle swarm optimization algorithm to minimize the noise
figure and input return loss (S 11) by designing the in-
put matching network and the first-stage balun. The opti-
mization parameters include seven balun-related variables,
namelyk, L 1, L2, R1, R2, C1, C2, together with the source-
degeneration inductorL s. The ranges of these variables are
summarized in Table II.
NFviolation = max 
0,NF sim−(NF user−0.2)
S11,violation = max 
0,S 11,sim + 20
cost= 1000×NF violation + 1000×S 11,violation (1)
Due to the inherent difficulty of analytically modeling input
impedance and noise figure, the tool adopts a high-fidelity
simulation-driven approach. For each candidate design, it first
invokes an OCEAN script to simulate the first two stages of
the amplifier without the input matching network. This step
extracts the input impedance of the first-stage transistor. The
tool then applies a low-fidelity, theory-based method derived
from smith chart principles to compute the L-type matching
network values (L parandL g) that achieve perfect matching
at the center frequency. Subsequently, another OCEAN sim-
ulation that includes the input matching network is executed
to evaluate theNF simandS 11,sim at the center frequency.
The cost function is calculated according to the user’s noise
requirements (NF user) as defined in (1). Within a limited
number of iterations, the algorithm yields a feasible solution
of passive component values that satisfy the performance
constraints for the critical stage.
3) Stage 3:Frequency Bandwidth PlanningThe third
stage focuses on planning the frequency response of the entire
signal chain. Ideally, one would desire a filter with perfect
passband transmission and complete stopband rejection. How-
ever, fourier theory dictates that such a frequency response is
unattainable in practice, and chebyshev filters are commonly
adopted as practical approximations. The staggered MCRs are
employed to implement the filter. The lumped model of the
MCR, shown in Fig. 4, has been extensively studied in prior
work [36], [37]. The source to drive the MCR is modeled as
an ideal current source in parallel with an RC parallel network
(RS||CS), while the load is represented byR 2||C2. Recall that
the transconductance values in our database were characterized

6
0.0010.0020.0030.0040.00
20.00 25.00 30.00 35.00 40.00Gain (dB)
Frequency (GHz)3.3dB2.5dBCalculation
Sim_wo_cali
Sim_w_cali
(a)
58.38501.69
0 200 400 600CPU Time (s)88% reduction
Multi -fidelity based
Simulation based
(b)
Fig. 5. (a) MCR frequency response: calculation vs. simulation (with/without
calibration); (b) Time Consumption: full simulation vs. multi-fidelity.
under a50Ωload. When the load is replaced by an MCR, the
voltage response of each gain stage is calculated according to
(3).
gainviolation = max 
0,|gainuser−min(gaincal)| −3
rippleviolation = max 
0,max(gaincal)−min(gaincal)−3
cost= 1000×gainviolation + 1000×rippleviolation
(2)
Building on this model, the tool adopts a simulated anneal-
ing algorithm to optimize the fullchain frequency response.
The optimization variables include four parameters for each
MCR stage, namelyk, ω 0, Q0, Cwith their ranges summa-
rized in Table II. For each candidate design, the fullchain
voltage frequency response (gaincal) is obtained by combining
the simulated voltage response of the critical performance
components from stage 2 with the theoretically calculated
responses of the MCR stages (gaincal). The cost function is
then evaluated based on the gain requirement (gainuser) and
a ripple constraint of 3dB, as defined in (2). After efficient
search process, a simple binary search maps MCR parameters
to the corresponding physical design.
In our calculations, the tool adopts the symmetric case of
the MCR to derive equation (3) and represents the source andload impedances as single-point values. This simplification
introduces an approximate 3% deviation between theoretical
and simulated results for a single MCR. In cascaded config-
urations, the broadband response arises from the interleaving
of several MCRs, while the sharper stopband edges further
amplify this deviation at higher frequencies. As illustrated
in Fig. 5a, the theoretically calculated cascaded gain (black
curve) diverges significantly from the simulated result (red
curve), with the gain at higher frequencies dropping beyond
the 3-dB constraint of ripple.
To resolve this issue, a multi-fidelity approach is employed.
In this strategy, a limited number of simulations are incorpo-
rated into the theoretical calculations, and the residual between
simulation and theory is iteratively fed back into subsequent
computations until the final simulated result satisfies the design
constraints. As shown by the blue curve in Fig. 5a, this cali-
bration process successfully reduces the gain ripple to within
3 dB. Compared with a purely simulation-based approach, the
proposed multi-fidelity method achieves an 88% reduction in
runtime, as demonstrated in Fig. 5b.
4) Stage 4:Full-Chain Performance Evaluation and
Refinement
By the end of stage 3, a promising candidate that satisfies
the power, band and gain requirements has been identified.
The next stage involves high-fidelity simulation to evaluate
whether this candidate meets the specifications for noise figure
and linearity. The tool performs fullchain small signal, S-
parameter, and harmonic balance simulations using precon-
figured OCEAN scripts. Based on the outcomes, the LLM
agent selects appropriate refinement tools depending on which
performance metric fails to meet the target.
The agent applies a staged backtracking policy. If noise
falls short, it re-runs stages 2–4 and increases the noise
figure headroom (initially 0.2) in proportion to the observed
degradation. If linearity is the bottleneck, it either (i) re-
plans the passband in stage 3 with tighter gain-distribution
constraints or (ii) re-runs stage 2-4 with global gain constraints
across all MCRs to find a better candidate. If specifications are
still unmet, the agent escalates to stage 1 to reallocate bias
current and regenerate candidate designs.
C. Multi-Agent Framework
This section presents the design rationale, applicable sce-
narios, and technical details of two operating modes of our
Y11(s) =(1−k2)Q2s4+ 2(1−k2)Qω 0s3+ (1−k2+ 2Q2)ω2
0s2+ 2Qω3
0s+Q2ω4
0
(1−k2)RSω0Qs3+ (1−k2)RSω2
0s2+R Sω3
0Q
RL=1
ℜ 
Y11, C L=ℑ 
Y11
s
Z21(s) =kQω3
0√RSR2s
(1−k2)Q2s4+ 2(1−k2)Qω 0s3+ (1−k2+ 2Q2)ω2
0s2+ 2Qω3
0s+Q2ω4
0
I11(s) =g m50R SCSs+R S+ 50
RSRS(RLCLs+ 1)
RSRL(CS+C L)s+R S+R L
V(s) =I11(s)Z 21(s)
2.(3)

7
Step1: Retrieve candidates from 
Knowledge Base.
Step2: Retrieve insights from 
Experience Base
Step3: Use < refine_candidate > 
to refine the most promising one
Step4: Use < search_candidate > 
to search new candidates
Step5: update KB and EB and 
output a final reportPlanRFAmpManager
Searcher 
Refiner <think> …This task is nfhard or linearity hard. From 
the insights from EB, when I want a gain1 of 12dB, 
the mag nfheadroom should be 0.15* NF_req …
</think> <tools> <search> power_ratio_list =[0.4,0.3,0.
3], gain1=12dB, NF_headroom =0.75dB </tools>
<observation> The result candidate has a nf1 of 3dB 
a IP1dB of -7dBm and it compresses… </obs>
<think> …The candidate has a nf1 of 3dB… The 
NF_req is 5dB. From the insights from EB, the gain2 
should no less than 6dB… </think> 
<tools> <refine>gain2=6dB, gain3=2dB </tools>
<observation> <failed>The result has a nfof 4.6dB, 
a IP1 of -22dBm, ac gain of 21dB, ac ripple of 1dB… 
</obs>…<answer> The refinement is failed, the 
reason is the gain1 is large … </answer>KBTemplate Format
power | gain | power_ratio_list | gain 
distribution | NF1 | NF | IP1_1 | IP1_2 
| IP1_3 | IP1 | params_dict
EBTemplate Format
x dB NF1 and y dB gain1 leads to z 
dB NF. The headroom shoud be a%
All stages are compressing, x, y ,z of 
IP1_n and a, b, c of gain_n leads to 
IP1 of m dBm.User Query 
Iwant todesign a3-stage RFAmp
working from 27GHz to33GHz whose
load is1787 .0ohm and 62.0fF.Iwant to
use 30.0mA toachieve 25.0dBgain,
3.5dBNoise Figure and-15.0dBm IP1dB.
Fig. 6. Proposed multi-agent framework and template formats of each part.
agent system:Autonomous Search ModeandSelf-Evolving
Retrieve-and-Refine Mode. In common RF design practice,
the early phase for a new technology node or topology
emphasizes coarse-to-fine exploration to delineate the feasible
region and establish upper bounds under given constraints;
once sufficient knowledge has accumulated, prior design points
are exploited to make tighter trade-offs via data-driven retrieval
and local refinement. This workflow is widely adopted in
academic practice [38], [39]. Inspired by this, RFAmpDesigner
instantiates two complementary modes: the autonomous search
mode operationalizes the exploratory phase, whereas the self-
evolving retrieve-and-refine mode operationalizes the exploita-
tive phase by retrieving relevant exemplars and iteratively
refining them under stricter specifications.
1) Mode 1:Autonomous Search ModeRFAmpDesigner
is a multi-agent collaboration framework for RF amplifier
design via user-driven goal decomposition and tool invocation.
The framework adopts a two-tier architecture with a central
primary agent and two stateless sub-agents. The primary agent
interacts with the user, maintains conversational context, and
manages system state. Acting as an orchestrator, it decomposes
the design task into specialized subtasks and delegates them
to the appropriate sub-agents. The sub-agents are specialized
modules with their own toolchains and a standardized I/O
schema. They are invoked in a stateless manner—each call
is independent and retains no conversational memory—which
enables parallel execution and fault isolation. Within each
invocation, every agent executes a ReAct-based [40] multi-
turn loop that interleaves reasoning with tool calls under the
standardized schema. Each agent supports autonomous early
termination of its current task. Formats of conversations are
demonstrated in Fig. 6. Details of these agents are presented
next.
a) RFAmpManager:Given a user query specifying the
number of stages, operating band, load impedance at the center
frequency, power budget, target small-signal gain, and the
noise figure and linearity at the center frequency, RFAmpMan-
ager decomposes the task into two subtasks: (i) searching for
promising candidates for the critical stages and (ii) continu-
ously refining a selected candidate. The agent’s state spaceS
is a priority queue of candidate critical modules; the action
spaceA={SEARCHCANDIDATES,REFINECANDIDATE}.The design rationale is twofold. First, in cascaded RF am-
plifiers, the input and output stages are most critical: the
first stage dominates the overall noise figure, whereas the
last stage governs large-signal linearity; both therefore require
high-fidelity evaluation, which is computationally expensive.
Second, bandwidth planning can be executed efficiently using
low-fidelity, model-guided optimization. Decoupling the costly
evaluation of critical components from the faster refinement
stage improves parallelism and end-to-end efficiency. Based
on user requirements, RFAmpManager ranks candidates along
three axes—noise figure, linearity, and large-signal behav-
ior—and schedules refinement. It then updates the candidate
priorities using the sub-agent’s summary reports to decide the
next action.
b) RFAmpSearcher:The sub-agent takes as input a
search instruction and the current candidate queueSand out-
puts a set of newly identified candidates∆S. For a three-stage
LNA, the “critical stage” comprises the first-stage transistor
with its input/output matching networks together with the
second-stage transistor loading the first stage. RFAmpSearcher
executes a tool-integrated, ReAct-style step-by-step procedure
with self-reflection, conditioned on user requirements and the
current search state. It first proposes a power-budget split for
the first two stages; for each split, it enumerates two extreme
device-sizing configurations at the PDK bounds (maximal and
minimal feasible options). For each active-device size, two
variants of critical-stage synthesis are solved: one with an
explicit gain constraint and one unconstrained. The resulting
candidates are returned to RFAmpManager for evaluation and
downstream refinement.
c) RFAmpRefiner:The sub-agent receives a promising
candidate from RFAmpManager, performs continuous refine-
ment, and returns both a finalized design (if any) and a
structured summary report. For a three-stage LNA, the remain-
ing decision variables include the third-stage power budget
and the gain allocation between the second and third stages.
The agent likewise executes a tool-integrated reasoning loop,
iteratively updating these variables based on circuit-simulation
feedback. Refinement proceeds until any of the following
stopping criteria is triggered: (i) all user-specified constraints
are satisfied within prescribed tolerances; (ii) simulation or
iteration budgets are exhausted. If the candidate is deemed

8
TABLE III
THELNA DESIGNBENCHMARK
Specsfc
(GHz)fBWPower
(mA)Gain
(dB)NF
(dB)IP1dB
(dBm)
S1 10 20% 30 25 ≤5 ≥-25
S2 30 20% 30 25 ≤5 ≥-25
S3 50 20% 30 20 ≤5 ≥-20
S4 30 10% 30 25 ≤5 ≥-25
S5 30 60% 30 25 ≤5 ≥-25
S6 30 80% 30 20 ≤5 ≥-20
S7 30 20% 30 25≤2.5 ≥-25
S8 30 20% 15 25 ≤5 ≥-25
S9 30 20% 30 25 ≤5 ≥-15
S10 30 20% 30 25≤3.5 ≥-15
infeasible under the given constraints, RFAmpRefiner termi-
nates early and returns a failure report. The summary report
records a success flag, attained performance metrics, any
violated constraints with magnitudes, and brief failure reasons,
enabling RFAmpManager to update priorities and schedule
subsequent actions.
2) Mode 2:Self-Evolving Retrieve-and-Refine Mode
In the search mode, the system maintains persistent memory
that records factual design knowledge accumulated during
exploration. Concretely, RFAmpSearcher and RFAmpRefiner
persist candidate configurations together with full-chain circuit
parameters and their measured performance metrics as struc-
tured records (JSON-like) in aKnowledge Base(KB). The KB
serves as the foundation for the retrieve-and-refine mode. In
addition, RFAmpRefiner stores derived indicators—including
the system noise figure, first-stage noise figure, per-stage input
1dB compression point (IP1dB), and the per-stage gain distri-
bution—into anExperience Base(EB) using predefined tem-
plates. These EB records provide actionable guidance when
high-fidelity evaluation is unavailable or cost-prohibitive.
In the retrieve-and-refine mode, the same three agents
collaborate as in search mode, augmented with RAG. Upon
receiving a new user query, RFAmpManager queries the KB
to retrieve top-kdesign points most similar to the target
specifications. The retrieved points are preloaded into the
candidate queue as priors for refinement. If refinement of
all preloaded candidates fail to satisfy the constraints within
budget, RFAmpSearcher is re-invoked to expand the candidate
set. Meanwhile, the EB is used for screening and pruning:
it (i) provides initialization hints (e.g., recommended power
splits and per-stage gain budgets) and (ii) provides insights
that estimate constraint headroom for each candidate. Agents
can reorder the queue and prune candidates in reasoning,
concentrating high-fidelity simulation on promising options.
By combining KB-based retrieval with EB-based screening,
the system improves data efficiency and end-to-end runtime
while preserving accuracy.
III. EXPERIMENTS AND PERFORMANCE ANALYSIS
This section evaluates RFAmpDesigner on wideband LNA
design. Comparisons with established baselines validate the
effectiveness of this work. Ablation studies isolate the contri-
butions of the tool middleware and each module of the frame-
work. Backbone-swap experiments show that the workflow is
-30.00-20.00-10.000.0010.0020.0030.00
5 15 25 35 45 55S Parameter (dB)
Frequency (GHz)Spec1 Spec2 Spec3
S21
S11(a)
1.002.003.004.005.006.00
5 15 25 35 45 55S Parameter (dB)
Frequency (GHz)NF
NFminSpec1Spec2
Spec7Spec3 (b)
-30.00-20.00-10.000.0010.0020.0030.00
5 15 25 35 45 55S Parameter (dB)
Frequency (GHz)Spec4
Spec5
Spec6S21
S1118 -42 GHz21 -39GHz
27 -33GHz
(c)
0.005.0010.0015.0020.0025.00
-40 -30 -20 -10 0Gain (dB)
Input_Power (dBm)Spec1
Spec9 (d)
Fig. 7. Design results of Spec 1-9: a)S 11andS 22of Spec 1-3; b)NF,
NFmin of Spec 1-3 and 7; c)S 11andS 22of Spec 4-6; d) Gain vs. Input
power of Spec 1, 9.
robust across different LLM backbones. Latency and cost are
analyzed for industrial deployment. A final section discusses
the generality and migration of the proposed abstraction.
A. Experiment Settings
1) Implementation details:All three agents are imple-
mented in Python and, for controlled comparison, execute
reasoning sequentially. The tool framework invokes Cadence
OCEAN scripts via python–shell interfaces for circuit simula-
tions and employs particle swarm optimization and simulated
annealing fromscikit-optto solve sub-optimization tasks.
All circuit simulations are performed using Cadence Spectre.
Experiments run on a Linux workstation with two AMD
EPYC 7763 (64 cores each).
2) Benchmark:A custom benchmark of ten tasks with
varying difficulty is constructed shown in Table III. Tasks 1–3
fix the fractional bandwidth at 20% and vary the center
frequency from 10 to 50 GHz. Tasks 4–6 fix the center
frequency at 30 GHz and vary the fractional bandwidth from
10% to 80%. Task 7 targets a lower noise figure; Task 8
imposes a tighter power budget. Task 9 requires output 1dB
compression point (OP1dB) of9dBm, and Task 10 combines
lowNFwith high OP1dB. Prior studies on LNAs with
comparable structures [41], [42] report typical OP1dB around
0 dBm, making Tasks 9–10 particularly challenging. Unless
otherwise specified, the amplifier has three stages and the load
is modeled as1787 Ω∥62 fF.
3) Baselines and metrics:For tasks 1–8 (low difficulty),
RFAmpDesigner are evaluated against several categories of
baselines: (i) vanilla Bayesian Optimization (BO) [43] and

9
TABLE IV
EXPERIMENT RESULTS OF THE FIRST8TASKS
Spec GA-vanilla BO-vanilla GA-subtool BO-subtool AnalogCoder ADO-LLM Our tool
pass
@1Avg
Time(s)pass
@1Avg
Time(s)pass
@1Avg
Time(s)pass
@1Avg
Time(s)pass
@1Avg
Time(s)pass
@1Avg
Time(s)pass
@1Avg
Time(s)
S1 2/5 1684 2/5 2114 5/5 771 5/5 659 4/5 72 5/5 517 5/5 655
S2 0/5 - 1/5 1615 3/5 928 3/5 1722 2/5 253 5/5 1406 5/5 620
S3 0/5 - 0/5 - 0/5 - 0/5 - 0/5 - 0/5 - 3/5 2307
S4 3/5 2006 4/5 1962 4/5 1133 4/5 2007 2/5 104 4/5 363 5/5 682
S5 0/5 - 0/5 - 0/5 - 0/5 - 0/5 - 0/5 - 4/5 904
S6 0/5 - 0/5 - 0/5 - 0/5 - 0/5 - 0/5 - 2/5 765
S7 0/5 - 0/5 - 1/5 725 0/5 - 1/5 502 3/5 1274 3/5 1401
S8 0/5 - 0/5 - 0/5 - 0/5 - 0/5 - 2/5 1922 5/5 760
*- indicates that the baseline failed to complete the task within the time limit.
01000200030004000
Spec1 Spec2 Spec4Time (s)GA
BO
Ours
ADO
+94%
-70%+304%
-85%+80%
-91%+39%
-69%
+13.8%
-10.4%+154%
-85%
+4.8%
-6.8%+ 195%
-78%+168%
-90%+68%
-46%
+5.6%
-11.7%+107%
-70%
Fig. 8. Mean completion time with max-min variation range (%) across tasks.
Genetic Algorithms (GA) [44] performing flat-search in the
original 18-dimensional space; (ii) tool-enhanced BO and GA,
which utilize our Stage2 sub-tools to operate in a reduced
16-dimensional space; (iii)AnalogCoder[26], a prompt-based
LLM design method; and (iv)ADO-LLM[30], an LLM-based
optimizer. This comparison is designed to highlight how the
proposed tool middleware accelerates search efficiency by
reducing design dimensionality and densifying sparse rewards.
For tasks 9–10 (high difficulty), BO, GA and ADO are aug-
mented with the proposed design tools (replacing the agentic
reasoning) to isolate the contribution of the tool middleware
from that of the multi-agent framework. Reinforcement learn-
ing methods are excluded from our baseline comparisons:
while effective in other analog design settings [45], [46], the
extreme reward sparsity and data scarcity in wideband RF
design lead to prohibitive training costs and high inference
latency (e.g., frequency bandwidth planning alone requires
around 5 million training steps, and inference still takes at least
50 steps). Three primary metrics are reported: (i) average wall-
clock time to a feasible solution, (ii) pass@1 success rate over
five independent runs (distinct random seeds), and (iii) token
consumption for all LLM-based methods to evaluate compu-
tational overhead. All methods are constrained by identical
simulation budgets and stopping criteria.
4) Agent backbones:To evaluate the robustness and scala-
bility of RFAmpDesigner across different LLM architectures,
we employ a diverse set of backbones ranging from fron-
tier closed-source models to specialized open-source variants.First, four state-of-the-art frontier models are evaluated, in-
cluding GPT-5.1, Claude-Sonnet-4.5, Gemini-2.5-Flash, and
DeepSeek-V3.2 to benchmark the performance and cost-
effectiveness of our framework. Second, to investigate the
impact of model scale and specialized fine-tuning, two models
from the Qwen3 family [47] are adopted: Qwen3-Coder-480B-
A35B-Instruct (mid-scale of activated 35B; code-oriented and
fine-tuned for tool use) and Qwen3-30B-A3B-Instruct (small-
scale of activated 3B; no dedicated tool-use fine-tuning). The
latter setting probes the model’s ability to understand and
operationalize complex RF-circuit concepts under concurrent
constraints on parameter count and tool-use capability.
B. LNA Design Examples
To demonstrate the capability of the proposed RFAmpDe-
signer for wideband RF amplifier design, the results of the first
eight tasks are summarized in Fig. 7. Fig. 7a and 7b present
the S-parameters,NF, andNF minfor the first three tasks,
confirming strong frequency scalability from 10 to 50GHz.
Fig. 7b further reports theNFandNF minof Spec 7 under
a stringent low-NFrequirement. Fig.7c illustrates theS 11
andS 21responses of Specs 4–6, showing that the framework
accommodates user-specified bandwidths from10%to80%,
thereby demonstrating both practicality and customizability.
Finally, Fig. 7d depicts the IP1dB curve of Spec 1. To the
best of our knowledge, this represents the first demonstration
of agile and customizable frequency- and bandwidth-aware
design for multi-stage wideband RF amplifiers.
C. Performance and Ablation Study
1) Effectiveness of the tool middleware:Table IV sum-
marizes the first eight tasks. Since the cost is defined by
task completion rather than a single FoM (thus admitting
multiple optima), the evaluation is simplified by fixing the
power distribution to a balanced ratio of [0.4, 0.3, 0.3], with
a one-hour timeout. The tasks are analyzed in three clusters:
Specs 1–3, 4–6, and 7–8. As center frequency rises, bandwidth
widens, and power/noise constraints tighten, tasks difficulty
increase and feasible regions become sparser. Accordingly,
all methods show lower success rates and longer runtimes.
BO and GA degrade most rapidly, as they neither encode RF
priors (e.g., high-frequency scaling ofL/C, transformer-based
passband flattening, or first-stage gain constraints for NF) nor

10
TABLE V
EXPERIMENT RESULTS OF THE LAST2TASKS
Methods S9 S10
pass
@1Avg Time
(s)Avg Prompt
TokensAvg Completion
Tokenspass
@1Avg Time
(s)Avg Prompt
TokensAvg Completion
Tokens
GA-wtool 3/5 9993 - - 1/5 11576 - -
BO-wtool 2/5 7625 - - 3/5 10700 - -
AnalogCoder(GPT-4o) 0/5 - 167468 11371 0/5 - 19338 11721
ADO-LLM(GPT-4o) 0/5 - 226495 19773 1/5 14884 162433 10688
ADO-LLM-wtool(GPT-4o) 3/5 5786 1093 590 2/5 6706 1099 633
Ours-Gemini-2.5-Flash 5/5 4521 36215 5005 3/5 2014 18840 2609
Ours-DeepSeek-V3.2 5/5 6114 62388 5652 5/5 6788 71939 6576
Ours-GPT-5.1 5/5 5611 62069 4963 3/5 13064 126893 9106
Ours-Claude-Sonnet-4.5 5/5 4785 25759 3986 5/5 6577 40136 5398
Ours-Claude-Sonnet-4.5
(Single Agent)5/5 6024 161150 6816 5/5 8308 55785 3658
Ours-Claude-Sonnet-4.5
(without Refiner)4/5 8681 8200 2719 2/5 16015 15135 3672
Ours-Claude-Sonnet-4.5
(without ReAct)5/5 10411 92489 3344 3/5 13816 96341 3364
Ours-Claude-Sonnet-4.5
(with EB)5/5 6100 8643 2399 5/5 6469 81884 7828
Ours-Qwen3-480B-A35B 4/5 5624 47623 2254 3/5 8302 29488 2344
Ours-Qwen3-30B-A3B 2/5 3926 65454 2953 1/5 2670 7454 1805
exploit stage-aware decomposition, and they are reliable only
on the easiest cases (Specs 1, 2, and 4).
Across methods, the progression from vanilla BO/GA
search (18-D) to sub-tool-equipped BO/GA search (16-D), and
finally to our full tool middleware shows a consistent trend:
reducing dimensionality and densifying rewards improves suc-
cess rates while shortening completion time. This indicates that
the proposed middleware effectively mitigates reward sparsity,
which is particularly severe under the wideband “frequency-
shaping” requirements.
Among LLM-based baselines,AnalogCoderis efficient on
narrow-band tasks (Specs 1, 2, and 4), achieving nearly
3×shorter completion time than GA/BO andADO-LLM,
suggesting that prompt-based reasoning can provide strong
initial points. However, its reduced success rate on harder tasks
implies that prompts alone are insufficient for navigating com-
plex high-performance design landscapes. In contrast,ADO-
LLMbenefits substantially from sub-tools and consistently
outperforms vanilla BO/GA in success rate, supporting that
LLMs can also steer optimization via high-level reasoning
when paired with appropriate tooling.
Finally, Fig. 8 compares mean completion time and variance
for Specs 1, 2, and 4. Our framework exhibits the best runtime
stability, with deviations capped at 25% (relative to BO, GA,
andADO-LLM), while maintaining the highest success rates
and the lowest average completion time on most tasks. Overall,
these results highlight how reward sparsity fundamentally
limits conventional optimizers in RF design, and demonstrate
that the proposed tool middleware improves both robustness
and efficiency across difficulty regimes.
2) Validity and ablation of the multi-agent framework:Ta-
ble V reports the results on the more challenging Specs 9–10,
where the timeout is extended to five hours. Under this setting,
flat-search baselines (GA and BO) failed to find any feasible
designs. We therefore replace them with tool-equipped variants
to isolate the effect of the multi-agent framework. With the
0246810
Multi-Agent Single-Agent No Refiner No ReAct Multi-Agent
with EBTimes
SettingsSpec9_search
Spec10_search
Spec9_refine
Spec10_refine0246810TimesSpec9_search Spec10_search
Spec9_refine Spec10_refineFig. 9. Mean number of search and refine per seed of different models
(up); Mean number of search and refine per seed of different configuration of
RFAmpDesigner (down).
middleware, the tasks become solvable, but the success rate
remains below 60%, and the average completion time is no
less than 2.1 hours, suggesting that tooling alone is insufficient
without expert reasoning for directional search and pruning.
Moreover, the direct LLM-based design approach (ADO-LLM)
performs poorly. For example, GPT-4o without tools achieves
0% success on Spec 9 despite consuming over 226k prompt
tokens. Even with tools (ADO-LLM), single-step prompting
is still inferior to our framework in both success rate and
convergence speed. These comparisons indicate that the chal-
lenge of RF circuit design lies not only in parameter tuning,
but high-level reasoning required to navigate sparse feasible
regions, which our framework addresses through structured
agentic decomposition.
RFAmpDesigner remains reliable across frontier LLMs. On

11
110100100010000
0123456Simulation Steps
Tasks IDOurs
GA
BO
110100100010000
1 2 3 4 5 6Simulation Steps
Task IDOurs
GA
BO
Fig. 10. Cumulative simulation cost with error bar over 5 seeds (left); Average
per-task simulation cost over 5 seeds (right).
Spec 9, all four leading models reach 100% success rate with
comparable runtimes, though GPT-5.1 and DeepSeek-V3.2
incur higher token overhead. On Spec 10, Gemini-2.5-Flash
and GPT-5.1 drop to 60% success, whereas Claude-Sonnet-
4.5 and DeepSeek-V3.2 maintain 100%. Notably, DeepSeek-
V3.2 consumes significantly more prompt tokens than Claude-
Sonnet-4.5. Fig. 9 further compares numbers of search and
refine operations. The search iterations are broadly similar
across LLM-based methods and increase with task difficulty.
In contrast, all LLM-based methods require markedly fewer
refine iterations than the baselines, suggesting that LLMs can
leverage limited refinement feedback to prune unpromising
candidates. Overall, Claude-Sonnet-4.5 offers the best balance
among success rate, runtime, and token cost.
The robustness of RFAmpDesigner to backbone size is also
evaluated using two open-source models: Qwen3-30B-A3B-
Instruct-2507 and Qwen3-Coder-480B-A35B-Instruct-2025-
07-22. Larger models achieve higher success rates, while
runtime shows no consistent trend and is largely driven by
the quality of the critical-stage candidates generated during
search. Although the activated 3B Qwen3-30B-A3B-Instruct
model yields the fewest refine iterations, trajectory inspection
shows frequent formatting failures and misconceptions about
complex circuit concepts. In contrast, Qwen3-Coder-480B-
A35B-Instruct performs comparably to Claude-Sonnet-4.5.
These results suggest that stronger reasoning and tool-use ca-
pabilities are crucial for stable agent–environment interaction.
Small models struggle with reasoning on complex concepts,
and models without tool-use fine-tuning behave inconsistently.
Therefore, a 30B tool-finetuned model is a practical starting
point for subsequent supervised fine-tuning and reinforcement
learning.
Finally, to substantiate the necessity of the Man-
ager–Searcher–Refiner structure, ablation studies are con-
ducted using Claude-Sonnet-4.5 as the backbone.
Ablation 1: Multi-Agent vs Single-Agent. The Single-
Agent variant achieves a comparable success rate, but con-
sumes far more prompt tokens. For example, its prompt token
consumption on Spec 9 (161k) is more than 6 times higher than
our proposed framework (25k). This indicates that multi-agent
0102030
10 100 1000 10000FoM
Simulation StepsOurs
GA
BO
5.5x Sample EfficiencyFig. 11. FoM Evolution vs. Simulation Cost.
structure substantially reduces reasoning burden and improves
token efficiency by agentic decomposition.
Ablation 2: Effect of the Refiner. Removing the Refiner
loop causes a pronounced degradation, especially on Spec 10
where success drops from 100% to 40%. Although token
usage decreases (as the Refiner is the main token contributor),
search becomes less effective. Fig. 9 shows the refinement
iterations nearly double, approaching the GA baseline. This
suggests that, without expert-guided refinement and pruning,
the workflow largely collapses into unguided stochastic search.
Ablation 3: Reasoning Mode. The ”without ReAct” variant
exhibits a significant increase in average time and a lower suc-
cess rate on Spec 10. This validates that the ”Thought-Action”
reasoning chain is crucial for the agents to interpret simulation
feedback and adjust optimization strategies dynamically.
Ablation 4: Role of the Experience Base (EB). While
the inclusion of EB yields only minor changes in overall
success rate and wall-clock time on these tasks, it consistently
improves internal efficiency. In particular, EB reduces the
number of refinement iterations by at least 40%, even when
compared to the Claude-Sonnet-4.5 backbone, which already
exhibits the lowest iteration count among all models. Retrieved
historical design traces provide an effective pruning signal,
enabling faster convergence with fewer redundant trials.
3) Evaluation of self-evolution mechanism:To evaluate
the efficacy of the self-evolution mechanism, we construct a
sequential design challenge based on the narrow-band setting
of Spec 4. Specifically, we define a series of tasks with
progressively increasing gain requirements (from 20 to 45dB),
while using the same Figure of Merit (FoM) definition as in
[48].
As shown in the left of Fig. 10, the proposed agent
avoids the exponential cost growth observed in GA and BO
(clearly visible on the logarithmic scale), and instead exhibits
a near-linear in cumulative cost. The underlying mechanism
is revealed by the ”sawtooth” pattern in the right of Fig. 10.
Unlike the baselines, which degrade monotonically due to
the cold-start issue across tasks, our agent alternates between
exploration (peaks when encountering new constraints) and
exploitation (valleys enabled by memory reuse). Notably, the

12
TABLE VI
COMPARISON OF EFFECTIVE LATENCY AND MONETARY COST ACROSS
DIFFERENT BACKBONES
Model S9 S10
Eff
Latency(s)Eff Cost
(USD)Eff
Latency(s)Eff Cost
(USD)
Gemini-2.5-Flash 4521 0.023 3356 0.020
DeepSeek-V3.2 6114 0.018 6788 0.020
GPT-5.1 5611 0.127 21773 0.416
Claude-Sonnet-4.5 4785 0.137 6577 0.201
Qwen3-480B-A35B 7030 0.061 13837 0.056
Qwen3-30B-A3B 9815 0.021 13352 0.008
ADO-LLM - - 74420 2.56
*- means the method fails to deliver feasible solutions
*Effective latency and cost are calculated by dividing the pass@1 rate.
agent solves harder tasks (e.g., Task 4) significantly faster
than the preceding exploration phases, indicating that the
agent is effectively transferring evolved knowledge rather than
restarting from scratch.
Importantly, this acceleration does not come at the expense
of design quality. Fig. 11 shows the evolution of FoM across
the tasks. The proposed agent consistently converges to FoM
values comparable to or exceeding those of the baselines
with lower variance. As annotated in Fig. 11, the faster
convergence corresponds to a 5.5×improvement in sample
efficiency. Overall, these results support that the self-evolution
mechanism improves both efficiency and solution quality by
enabling more effective reuse of accumulated experience when
navigating the Pareto front.
4) Analysis of computational cost:To assess deployment
feasibility, we analyze the trade-off between computational
overhead and performance across different backbone models.
Table VI reports the effective latency and monetary cost
for each setting. Here, effective latency includes both LLM
inference time and the runtime of external simulation tools,
and thus reflects the end-to-end wall-clock time of a complete
design run.
The results indicate that the framework is cost-effective
in practice. Even with LLM inference, the average monetary
cost is below $0.5 per successful design. This low cost is
primarily attributed to the improved sample efficiency. The
LLM’s ability to think before execution drastically prunes the
search space, avoiding thousands of redundant simulations
that traditional baselines would otherwise require. Overall,
the framework provides measurable gains in robustness and
convergence with only modest additional inference overhead,
making it economically viable for iterative RF design work-
flows.
D. Discussion about Generality and Migration
The proposed method is build on a stage-wise interface:
the LLM proposes resource budgets (e.g., gain/current/noise/-
linearity margins) and the optimizer tunes circuit parameters
within the search space defined by each stage tool. This
design makes the framework less tied to a specific circuit
instance. (i) Multi-stage circuits: adding stages mainly increase
the number of budgets. The same decision loop is reused.
(ii) Power amplifiers: migration mainly requires changing theperformance targets (e.g.,P out, efficiency) and implementing
PA-specific stage tools with the output stage typically becom-
ing the critical stage. Preceding stages reduce to driver-like
design (cf. Spec 9,10). (iii) Topology changes: the workflow
is unchanged, but tool implementations must be updated to
match new schematic templates. We include this discussion
to clarify the engineering workload. A full PA or single-end
amplifier benchmark is left for future work due to space.
IV. CONCLUSION
This paper presents RFAmpDesigner, an end-to-end LLM-
based framework for automated RF amplifier parameter sizing,
demonstrated on low-noise amplifiers from specifications to
schematics. The key idea is a multi-fidelity tool middleware
that turns sizing into a resource-allocation problem, making
domain knowledge easier to apply for LLMs and improving
optimization in high-dimensional, sparse-feedback settings.
On top of this, a two-tier, three-agent workflow separates
heavy simulation tasks from lightweight reasoning to enable
efficient search and refinement. By integrating a knowledge
base and an experience base via retrieval, the framework
supports a self-improving loop and better sample efficiency.
While this work is a proof of concept, the same abstraction is
applicable to other amplifier types and can be extended toward
more diverse topologies and more complete design flows.
ACKNOWLEDGMENTS
The authors acknowledge the use of ChatGPT solely to
refine the sentence structure and grammar of the manuscript.
The authors reviewed and edited the content as needed and
take full responsibility for the content of the publication.
REFERENCES
[1] I. Guven, M. Parlak, D. Lederer, and C. D. Vleeschouwer, “Ai-driven
integrated circuit design: A survey of techniques, challenges, and op-
portunities,”IEEE Access, pp. 1–1, 2025.
[2] S. Yin, R. Wang, J. Zhang, X. Liu, and Y . Wang, “Automatic design for
w-band front-end system via bottom-up sizing and layout generation,”
IEEE Transactions on Computer-Aided Design of Integrated Circuits
and Systems, vol. 43, no. 3, pp. 705–715, 2024.
[3] F. Passos, M. Chanca, E. Roca, R. Castro-L ´opez, and F. V . Fern ´andez,
“Synthesis of mm-wave wideband receivers in 28-nm cmos technology
for automotive radar applications,”IEEE Transactions on Computer-
Aided Design of Integrated Circuits and Systems, vol. 39, no. 12, pp.
4375–4384, 2020.
[4] B. Liu, D. Zhao, P. Reynaert, and G. G. E. Gielen, “Gaspad: A general
and efficient mm-wave integrated circuit synthesis method based on
surrogate model assisted evolutionary algorithm,”IEEE Transactions on
Computer-Aided Design of Integrated Circuits and Systems, vol. 33,
no. 2, pp. 169–182, 2014.
[5] W. Lyu, F. Yang, C. Yan, D. Zhou, and X. Zeng, “Multi-objective
bayesian optimization for analog/rf circuit synthesis,” in2018 55th
ACM/ESDA/IEEE Design Automation Conference (DAC), 2018, pp. 1–6.
[6] M. Pirrone, E. Dall’Anese, and T. W. Barton, “Data-driven optimization
strategies for tunable rf systems,”IEEE Transactions on Microwave
Theory and Techniques, vol. 72, no. 3, pp. 1919–1931, 2024.
[7] D. Lee, G. Shin, S. Lee, K. Kim, T.-H. Oh, and H.-J. Song, “Neural-
network-based automated synthesis of transformer matching circuits
for rf amplifier design,”IEEE Transactions on Microwave Theory and
Techniques, vol. 70, no. 11, pp. 4726–4739, 2022.
[8] G. Zhang, H. He, and D. Katabi, “Circuit-GNN: Graph neural
networks for distributed circuit design,” inProceedings of the 36th
International Conference on Machine Learning, ser. Proceedings of
Machine Learning Research, K. Chaudhuri and R. Salakhutdinov, Eds.,
vol. 97. PMLR, 09–15 Jun 2019, pp. 7364–7373. [Online]. Available:
https://proceedings.mlr.press/v97/zhang19e.html

13
[9] W. Cao, M. Benosman, X. Zhang, and R. Ma, “Domain knowledge-
infused deep learning for automated analog/radio-frequency circuit
parameter optimization,” inProceedings of the 59th ACM/IEEE
Design Automation Conference, ser. DAC ’22. New York, NY , USA:
Association for Computing Machinery, 2022, p. 1015–1020. [Online].
Available: https://doi.org/10.1145/3489517.3530501
[10] K. Technologies,Advanced Design System (ADS), Keysight Tech-
nologies, 2025, version 2025, Available: https://www.keysight.com/find/
eesof-ads.
[11] J. E. Rayas-S ´anchez, S. Koziel, and J. W. Bandler, “Advanced rf and
microwave design optimization: A journey and a vision of future trends,”
IEEE Journal of Microwaves, vol. 1, no. 1, pp. 481–493, 2021.
[12] J. Bandler, R. Biernacki, S. H. Chen, P. Grobelny, and R. Hemmers,
“Space mapping technique for electromagnetic optimization,”IEEE
Transactions on Microwave Theory and Techniques, vol. 42, no. 12,
pp. 2536–2544, 1994.
[13] H. Kabir, Y . Wang, M. Yu, and Q.-J. Zhang, “Neural network inverse
modeling and applications to microwave filter design,”IEEE Transac-
tions on Microwave Theory and Techniques, vol. 56, no. 4, pp. 867–879,
2008.
[14] S. Er, E. Liu, M. Chen, Y . Li, Y . Liu, T. Zhao, and H. Wang, “Deep
learning assisted end-to-end synthesis of mm-wave passive networks
with 3d em structures: A study on a transformer-based matching
network,” in2021 IEEE MTT-S International Microwave Symposium
(IMS), 2021, pp. 66–69.
[15] D. Munzer, S. Er, M. Chen, Y . Li, N. S. Mannem, T. Zhao, and H. Wang,
“Residual network based direct synthesis of em structures: A study on
one-to-one transformers,” in2020 IEEE Radio Frequency Integrated
Circuits Symposium (RFIC), 2020, pp. 143–146.
[16] H. Chae, H. Yu, S. Li, and D. Z. Pan, “Pulserf: Physics augmented
ml modeling and synthesis for high-frequency rfic design,” in
Proceedings of the 43rd IEEE/ACM International Conference on
Computer-Aided Design, ser. ICCAD ’24. New York, NY , USA:
Association for Computing Machinery, 2025. [Online]. Available:
https://doi.org/10.1145/3676536.3676768
[17] J. Wei, W. Chen, Y . Gong, Q. Wu, G. Lu, W. Gao, L. Wang, M. Li, and
H. Wang, “Highly efficient automatic synthesis of a millimeter-wave on-
chip deformable spiral inductor using a hybrid knowledge-guided and
data-driven technique,”IEEE Transactions on Computer-Aided Design
of Integrated Circuits and Systems, vol. 42, no. 12, pp. 4413–4422, 2023.
[18] E. A. Karahan, Z. Liu, and K. Sengupta, “Deep-learning-based inverse-
designed millimeter-wave passives and power amplifiers,”IEEE Journal
of Solid-State Circuits, vol. 58, no. 11, pp. 3074–3088, 2023.
[19] J. Zhou, E. A. Karahan, S. Ghozzy, Z. Liu, H. Jalili, and K. Sengupta,
“25.3 ai-enabled design space discovery and end-to-end synthesis for
rfics with reinforcement learning and inverse methods demonstrating
mm-wave/sub-thz pas between 30 and 120ghz,” in2025 IEEE Interna-
tional Solid-State Circuits Conference (ISSCC), vol. 68, 2025, pp. 1–3.
[20] J. Jin, C. Zhang, F. Feng, W. Na, J. Ma, and Q.-J. Zhang, “Deep
neural network technique for high-dimensional microwave modeling
and applications to parameter extraction of microwave filters,”IEEE
Transactions on Microwave Theory and Techniques, vol. 67, no. 10, pp.
4140–4155, 2019.
[21] P. Naseri and S. V . Hum, “A generative machine learning-based approach
for inverse design of multilayer metasurfaces,”IEEE Transactions on
Antennas and Propagation, vol. 69, no. 9, pp. 5725–5739, 2021.
[22] K. Ding, J. Yu, J. Huang, Y . Yang, Q. Zhang, and H. Chen, “Scitoolagent:
a knowledge-graph-driven scientific agent for multitool integration,”
Nature Computational Science, pp. 1–11, 2025.
[23] C.-C. Chang, Y . Shen, S. Fan, J. Li, S. Zhang, N. Cao, Y . Chen,
and X. Zhang, “Lamagic: language-model-based topology generation
for analog integrated circuits,” inProceedings of the 41st International
Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.
[24] Z. Chen, J. Huang, Y . Liu, F. Yang, L. Shang, D. Zhou, and X. Zeng,
“Artisan: Automated operational amplifier design via domain-specific
large language model,” inProceedings of the 61st ACM/IEEE Design
Automation Conference, ser. DAC ’24. New York, NY , USA:
Association for Computing Machinery, 2024. [Online]. Available:
https://doi.org/10.1145/3649329.3655903
[25] J. Shen, Z. Chen, J. Zhuang, J. Huang, F. Yang, L. Shang, Z. Bi,
C. Yan, D. Zhou, and X. Zeng, “Atelier: An automated analog circuit
design framework via multiple large language model-based agents,”
IEEE Transactions on Computer-Aided Design of Integrated Circuits
and Systems, pp. 1–1, 2025.
[26] Y . Lai, S. Lee, G. Chen, S. Poddar, M. Hu, D. Z. Pan,
and P. Luo, “Analogcoder: analog circuit design via training-
free code generation,” inProceedings of the Thirty-Ninth AAAIConference on Artificial Intelligence and Thirty-Seventh Conference
on Innovative Applications of Artificial Intelligence and Fifteenth
Symposium on Educational Advances in Artificial Intelligence, ser.
AAAI’25/IAAI’25/EAAI’25. AAAI Press, 2025. [Online]. Available:
https://doi.org/10.1609/aaai.v39i1.32016
[27] H. Zhang, S. Sun, Y . Lin, R. Wang, and J. Bian, “Analogxpert:
Automating analog topology synthesis by incorporating circuit design
expertise into large language models,” in2025 International Symposium
of Electronics Design Automation (ISEDA), 2025, pp. 772–777.
[28] J. Gao, W. Cao, J. Yang, and X. Zhang, “Analoggenie: A generative
engine for automatic discovery of analog circuit topologies,” inThe
Thirteenth International Conference on Learning Representations, 2025.
[Online]. Available: https://openreview.net/forum?id=jCPak79Kev
[29] P.-H. Chen, Y .-S. Lin, W.-C. Lee, T.-Y . Leu, P.-H. Hsu, A. Dissanayake,
S. Oh, and C.-S. Chiu, “Menter: A fully-automated multi-agent
workflow for end-to-end rf/analog circuits netlist design,” 2025.
[Online]. Available: https://arxiv.org/abs/2505.22990
[30] Y . Yin, Y . Wang, B. Xu, and P. Li,ADO-LLM: Analog Design Bayesian
Optimization with In-Context Learning of Large Language Models.
New York, NY , USA: Association for Computing Machinery, 2025.
[Online]. Available: https://doi.org/10.1145/3676536.3676816
[31] D. V . Kochar, H. Wang, A. Chandrakasan, and X. Zhang, “Ledro:
Llm-enhanced design space reduction and optimization for analog
circuits,” 2025. [Online]. Available: https://arxiv.org/abs/2411.12930
[32] C. Liu, W. Chen, H. Xu, Y . Du, J. Yang, and L. Du, “A large
language model-based multi-agent framework for analog circuits’
sizing relationships extraction,” 2025. [Online]. Available: https:
//arxiv.org/abs/2506.18424
[33] M. Ahmadzadeh, K. Chen, and G. Gielen, “Anaflow: Agentic
llm-based workflow for reasoning-driven explainable and sample-
efficient analog circuit sizing,” 2025. [Online]. Available: https:
//arxiv.org/abs/2511.03697
[34] B. Liu, H. Zhang, X. Gao, Z. Kong, X. Tang, Y . Lin, R. Wang, and
R. Huang, “Layoutcopilot: An llm-powered multi-agent collaborative
framework for interactive analog layout design,” 2025. [Online].
Available: https://arxiv.org/abs/2406.18873
[35] M. Vigilante and P. Reynaert, “On the design of wideband transformer-
based fourth order matching networks forE-band receivers in 28-nm
cmos,”IEEE Journal of Solid-State Circuits, vol. 52, no. 8, pp. 2071–
2082, 2017.
[36] H. Jia, C. C. Prawoto, B. Chi, Z. Wang, and C. P. Yue, “A full ka-band
power amplifier with 32.9% pae and 15.3-dbm power in 65-nm cmos,”
IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 65,
no. 9, pp. 2657–2668, 2018.
[37] R. Singh, S. Mondal, and J. Paramesh, “A millimeter-wave receiver using
a wideband low-noise amplifier with one-port coupled resonator loads,”
IEEE Transactions on Microwave Theory and Techniques, vol. 68, no. 9,
pp. 3794–3803, 2020.
[38] H. Wang,LNA Survey, ETH Zurich, 2025, available: https://ideas.ethz.
ch/Surveys/lna-survey.html.
[39] H. Wang,PA Survey, ETH Zurich, 2025, available: https://ideas.ethz.ch/
Surveys/pa-survey.html.
[40] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“React: Synergizing reasoning and acting in language models,”arXiv
preprint arXiv:2210.03629, 2022.
[41] J.-H. Kim, J.-T. Son, J.-T. Lim, H.-W. Choi, and C.-Y . Kim, “Ultralow
noise figure and broadband cmos lna with three-winding transformer
and large transistor,”IEEE Transactions on Microwave Theory and
Techniques, vol. 72, no. 5, pp. 2734–2744, 2024.
[42] H. Chen, H. Zhu, L. Wu, W. Che, and Q. Xue, “A wideband cmos
lna using transformer-based input matching and pole-tuning technique,”
IEEE Transactions on Microwave Theory and Techniques, vol. 69, no. 7,
pp. 3335–3347, 2021.
[43] F. Nogueira, “Bayesian Optimization: Open source constrained
global optimization tool for Python,” 2014–. [Online]. Available:
https://github.com/bayesian-optimization/BayesianOptimization
[44] G. Guo, “scikit-opt,” GitHub repository, 2017, [Online]. Available: https:
//github.com/guofei9987/scikit-opt.
[45] K. Settaluri, A. Haj-Ali, Q. Huang, K. Hakhamaneshi, and B. Nikolic,
“Autockt: Deep reinforcement learning of analog circuit designs,” in
2020 Design, Automation & Test in Europe Conference & Exhibition
(DATE), 2020, pp. 490–495.
[46] M. Choi, Y . Choi, K. Lee, and S. Kang, “Reinforcement learning-based
analog circuit optimizer using gm/id for sizing,” in2023 60th ACM/IEEE
Design Automation Conference (DAC), 2023, pp. 1–6.

14
[47] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao,
C. Huang, C. Lvet al., “Qwen3 technical report,”arXiv preprint
arXiv:2505.09388, 2025.
[48] B. Lin, A. Ruffino, N. Villaggi, T.-Y . Huang, C. Chu, and H. Wang, “V-
/e-band joint-feeding gate/source broadband lna with three-line coupler
for simultaneous noise/power matching,”IEEE Journal of Solid-State
Circuits, pp. 1–12, 2025.
Hang Lu(Graduate Student Member, IEEE) re-
ceived the B.S. degree from Zhejiang University,
Hangzhou, China, in 2021. He is currently pursu-
ing the Ph.D. degree with the institute of marine
electronics and intelligent systems, Ocean College
Zhejiang University, Zhoushan, 316021 China. His
research interests include Agent-based analog, RF
and millimeter-wave integrated circuits design au-
tomation.
Guochang Li(Graduate Student Member, IEEE)
received the B.S. degrees from Zhejiang University,
China, in 2021. He is currently studying for the Ph.D
degree in Zhejiang University. His research interests
include Automated program repair and Agent-based
Software Engineering.
Qianyu Chen(Member, IEEE) is currently pursuing
the B.S. degree in Zhejiang University, Hangzhou,
China. Her research interests include multi-agent
systems and agentic workflow frameworks for RF
circuit optimization and design automation.
Huiyan Gao(Graduate Student Member, IEEE)
received the B.S. degree in 2018 and Ph.D de-
gree in 2024 from Zhejiang University, Hangzhou,
China. His research interests include analog, RF
and millimeter-wave integrated circuits in silicon
technologies.
Shaogang Wang(Graduate Student Member, IEEE)
received the B.S. degree from Northwestern Poly-
technical University, Xi’an, China, in 2018. He
is currently pursuing the Ph.D. degree with the
institute of marine electronics and intelligent sys-
tems, Ocean College Zhejiang University, Zhoushan,
316021 China. His current research interests include
RF and millimeter-wave integrated circuits for wire-
less communications and phased-array systems.
Xuanyu He(Graduate Student Member, IEEE) re-
ceived the B.E. degree from the College of Elec-
tronic and Information Engineering, Beijng Jiaotong
University, Beijing, China, in 2023. He is currently
pursuing the M.E. degree at the Ocean College, Insti-
tute of Marine Electronics and Intelligent Systems,
Zhejiang University, Zhoushan, China. His research
interests include radio frequency integrated circuit
(RFIC) design and RFIC EDA.
Yiwei Liu(Graduate Student Member, IEEE) re-
ceived the B.S. degrees from Zhejiang University,
Hangzhou, China, in 2023. He is currently studying
for the Ph.D degree in Zhejiang University. His re-
search interests include analog, RF, and millimeter-
wave integrated circuits in silicon technologies. Re-
cently he is doing researches on deep learning As-
sisted RF Circuit Design.
Gaopeng Chen(Member, IEEE) received the B.S.
from the University of Science and Technology
of China, and Ph.D. degree from the Institute of
Microelectronics of the Chinese Academy of Sci-
ences, in microelectronics. He has held industry
positions with RDA Micro-electronics Inc. and Etra
Semiconductors, where he managed the R&D teams
to develop microwave and millimeter wave ICs for
cellular phones and base stations. Currently, he is
working on the next generation integrated sensing
and communication circuits and systems.
Nayu Li(Member, IEEE) received the B.S. degree
in information engineering from College of Infor-
mation Science and Electronic Engineering, Zhe-
jiang University, Hangzhou, China, in 2017, and the
Ph.D. degree in ocean technology and engineering
from Ocean College, Zhejiang University, Zhoushan,
China, in 2022. He is currently a research fellow
with the Donghai Laboratory, Zhoushan, China, and
also with the Institute of Marine Electronics and In-
telligent Systems, Ocean College, Zhejiang Univer-
sity, Zhoushan, China. His research interests include
analog, RF, and millimeter-wave integrated circuits in silicon technologies. Dr.
Li was a recipient of the 2024 IEEE MTT-S International Wireless Symposium
(IWS) FLASH Competition Second Place Winner and was a co-recipient
of the 2023 IEEE MTT-S International Microwave Symposium (IMS) Best
Student Paper Award (Third Place), and the 2024 IEEE Radio Frequency
Integrated Circuits (RFIC) Symposium Best Student Paper Award Finalist.
Xiaokang Qi(Member, IEEE) received the B.S. de-
gree from the School of Electronics and Information,
Northwestern Polytechnical University, Xian, China,
in 2013, and the Ph.D. degree from the College
of Information Science and Electronics Engineering,
Zhejiang University, Hangzhou, China, in 2018. He
became a Researcher with the Ocean College, Zhe-
jiang University, in 2020. His current research inter-
ests include mm-wave communication and sensing
systems, RF theory, and navigation algorithms.
Chunyi Song(Member, IEEE) received the Ph.D.
degree in electronic and communication engineering
from Waseda University, Tokyo, Japan. He was a
Research Associate with Waseda University, from
2007 to 2009. He then joined the National Institute
of Information and Communications Technology
(NICT), Japan, as a Researcher, from 2009 to 2013,
and as a Senior Researcher, in 2014. Since 2014,
he has been with Zhejiang University, China, as an
Associate Professor, where he is also serving as the
Vice Director of The Engineering Research Center
of Oceanic Sensing Technology and Equipment, Ministry of Education.
He was elected to the Thousand Talents Program of Zhejiang Province, in
2016, and a Core Member of Leading Innovative Team of Zhejiang, in 2018.
Zhiwei Xu(Senior Member, IEEE) received the
B.S.and M.S. degrees from Fudan University, Shang-
hai, China, in 1997 and 2000, respectively, all in
electrical engineering, and the Ph.D. degree in elec-
trical engineering from the University of California
at Los Angeles, Los Angeles, CA, USA, in 2003. He
has held industry positions with G-Plus Inc., SST
Communications, Conexant Systems, NXP Semi-
conductors, and HRL Laboratories, where he led
the development for wireless LAN and SoC solution
for proprietary wireless multimedia systems, CMOS
cellular transceiver, Multimedia over Cable (MoCA) system and TV tuners,
various aspects of millimeter- and sub-millimeter-wave integrated circuits and
systems, software defined radios, high-speed ADC, and ultralow power analog
VLSI. He is currently a Professor with Zhejiang University, Hangzhou, China,
where he is researching on integrated circuits and systems for Internet-of-
Things and communication applications.