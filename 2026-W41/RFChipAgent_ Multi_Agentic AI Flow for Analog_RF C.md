# RFChipAgent: Multi-Agentic AI Flow for Analog/RF Chip Design

**Authors**: Awani Khodkumbhe, Yunfei Feng, Raj Rangarajan, Kevin Wang, Kamal Sahota

**Published**: 2026-10-07 20:02:33

**PDF URL**: [https://arxiv.org/pdf/2610.10858v1](https://arxiv.org/pdf/2610.10858v1)

## Abstract
Analog/RF circuits remain the critical interface between digital computation and the physical world, and emerging standards from Wi-Fi 7 to 6G place stringent demands on them, yet analog/RF design remains one of the most labor-intensive steps in chip development. We present RFChipAgent, a first-of-its-kind multi-agent flow of large language model (LLM) agents for end-to-end analog/RF circuit design automation, in which AI agents collaboratively orchestrate the complete design flow under human supervision. RFChipAgent is built around four technical pillars. First, a multimodal retrieval-augmented generation (RAG) subsystem with private per-document FAISS indexing extracts design knowledge from existing engineering documentation. Second, a topology agent drives topology selection, and a schematic and testbench agent automates circuit and testbench assembly. Third, a closed-loop hybrid circuit-sizing engine combines Tree-structured Parzen Estimator (TPE) and CMA-ES optimization, evaluating every candidate in a simulator-in-the-loop framework. Fourth, a trust-scored simulation database accumulates verified performance data and builds an adaptive optimization model that informs subsequent trials. We validate RFChipAgent on a family of GF22FDSOI 60 GHz wideband mm-wave low-noise amplifier (LNA) topologies, demonstrating automated topology generation, specification-driven design-space exploration, and simulator-guided optimization. Experimental results show substantial reductions in design effort while maintaining signoff-quality verification. This work establishes a foundation for LLM-driven multi-agent electronic design automation (EDA) for analog/RF circuits.

## Full Text


<!-- PDF content starts -->

PREPRINT 1
RFChipAgent: Multi-Agentic AI Flow for Analog/RF Chip Design
Awani Khodkumbhe
Qualcomm Technologies, Inc.,
UC BerkeleyYunfei Feng
Qualcomm Technologies, Inc.Raj Rangarajan
Qualcomm Technologies, Inc.
Kevin Wang
Qualcomm Technologies, Inc.Kamal Sahota
Qualcomm Technologies, Inc.
Abstract
The wireless connectivity standards — from Wi-Fi 7 to the emerg-
ing 6G frontier — place stringent performance demands on Analog
and RF circuits, which remain the critical interface between digi-
tal computation and the physical world. Yet, the Analog/RF chip
design process remains one of the most labor-intensive and time-
consuming. This work introduces a first-of-its-kind multi-agentic
AI flow RFChipAgent for end-to-end design automation of Ana-
log/RF circuits, in which autonomous AI agents collaboratively
orchestrate the complete design flow under human supervision.
RFChipAgent is built around four technical pillars. First, a mul-
timodal retrieval-augmented generation (RAG) subsystem with
private per-document FAISS indexing enables extraction of design
knowledge from existing engineering documentation. Second, a
topology agent drives topology selection, and a schematic and test-
bench agent automates circuit and testbench assembly. Third, a
closed-loop hybrid sizing engine combines Tree-structured Parzen
Estimator (TPE) and Covariance Matrix Adaptation Evolution Strat-
egy (CMA-ES), evaluating every candidate in a simulator-in-the-
loop framework. Fourth, a trust-scored simulation database accumu-
lates verified performance data across trials and builds an adaptive
optimization model that progressively informs subsequent trials.
Together, these pillars replace the traditional trial-and-error de-
sign methodology with a systematic, self-improving agentic work-
flow. We validate RFChipAgent on a family of GF22FDSOI 60 GHz
wideband Low Noise Amplifier (LNA) topologies, demonstrating
automated topology generation, specification-driven design-space
exploration, and simulator-guided optimization. Experimental re-
sults show substantial reductions in design effort while maintaining
signoff-quality verification. This work establishes a foundation for
next-generation multi-agentic AI flows for analog/RF electronic
design automation.
Keywords
Large language models, LLM agents, Agentic AI for circuit design,
AI-based chip design flow automation, Electronic design automa-
tion, Analog/RF circuits, Circuit sizing, Retrieval-augmented gener-
ation, Optimization, RFIC, mm-wave LNA1 Introduction
The design of analog/RF integrated circuits remains a lengthy and
laborious process because topology selection, device sizing, biasing,
and passive-network design are tightly coupled. For example, in
a 60 GHz Low Noise Amplifier (LNA), a change in transformer
resonance or transistor finger count can substantially alter cir-
cuit performance. Recent AI-driven chip design approaches have
shown promising directions toward automation, including topology
generation [ 1], multi-agent design frameworks [ 2], reinforcement
learning for optimization [ 3], and LLM-based design reasoning [ 4].
However, an end-to-end multi-agentic analog/RF design AI flow
remains unrealized due to the lack of a trustworthy framework that
can seamlessly integrate design knowledge, topology synthesis,
simulation-driven optimization, and continual learning.
As shown in Fig. 1, the proposed framework is orchestrated
by a superagent that coordinates topology selection, automated
schematic and testbench generation, simulator-in-the-loop opti-
mization, and trust-scored knowledge accumulation. A retrieval-
augmented generation (RAG) subsystem supports the topology
agent by providing domain-specific design knowledge extracted
from existing documentation. A key requirement for a trustwor-
thy AI design flow is ensuring that invalid schematic assumptions,
erroneous design modifications, and non-improving optimization
actions are detected and filtered before they can propagate through
subsequent design iterations.
The contributions are as follows:
•To the best of our knowledge, RFChipAgent is the first multi-
agentic AI flow for end-to-end analog/RF circuit design au-
tomation, spanning knowledge ingestion, topology selection,
schematic and testbench generation, simulator-in-the-loop
optimization, and trust-scored knowledge accumulation.
•A multimodal RAG subsystem with per-document FAISS
indexing enables reliable retrieval of design knowledge while
mitigating cross-document hallucinations.
•A physics-aware topology agent that follows causally or-
dered design decisions, including stage selection, device se-
lection, and matching-network synthesis, coupled with an
automated schematic and testbench generation agent.
•Physical-validity gates, including DC operating-point and
stability checks, that are executed prior to objective evalua-
tion, preventing invalid designs from propagating through
the optimization loop.
•A trust-scored simulation database that stores only verified
design evaluations and uses accumulated performance data
to progressively improve subsequent optimization iterations.
This work has been submitted to the ACM/IEEE for possible publication. Copyright may be
transferred without notice, after which this version may no longer be accessible.
arXiv:2610.10858v1  [cs.AR]  7 Oct 2026

PREPRINT 2
Orchestrator
Superagent
Knowledge
Ingestion
AgentTopology
AgentSchematic &
Testbench
AgentSizing
OptimizerDatabase
PopulationTopology library
locallocal
global
Optimizing
Model
1 2 3 4 5
verified
trialsguided
proposals orchestration artifact hand-off learning loop
Figure 1: Illustration of the proposed RFChipAgent flow.
2 Preliminaries
2.1 Generative Analog/RF Design
Recent work such as EVA [ 1] treats analog topology discovery as
a generative task, using a transformer to predict device-pin con-
nections and fine-tuning with Proximal Policy Optimization (PPO)
or Direct Preference Optimization (DPO) for specification-driven
topology generation. Multi-agent generative synthesis frameworks
further separate topology generation from inverse design [ 2]. While
these approaches demonstrate the promise of generative AI for
topology exploration, they do not provide a complete design flow
spanning knowledge ingestion, schematic generation, simulation-
driven optimization, and trusted knowledge accumulation.
2.2 Optimization and Reasoning
RoSE [ 3] combines Bayesian optimization and reinforcement learn-
ing to improve the efficiency of analog parameter optimization
under PVT variation. Reasoning-based LLM approaches for hard-
ware generation have demonstrated that structured revision and
compiler feedback can improve RTL correctness and reduce gen-
eration errors [ 4]. These works illustrate promising directions for
introducing learning and reasoning capabilities into chip design
workflows.
3 Proposed RFChipAgent Flow
In the proposed RFChipAgent flow, a superagent coordinates a
knowledge-ingestion agent, a topology agent, a schematic and test-
bench generation agent, a sizing optimizer, and a trust-scored data-
base. Accumulated simulation data guides optimization, while an
LLM-in-the-loop provides recommendations for subsequent design
campaigns, enabling a self-improving design flow.3.1 Knowledge Ingestion Agent
The use of LLMs for design knowledge retrieval introduces the risk
of hallucination, where a model may fabricate performance metrics,
topology descriptions, or design insights that are not grounded in
source documents. Such errors can silently propagate into down-
stream topology-selection decisions. Prior work such as MuaLLM
[6] mitigates this challenge by decoupling retrieval from inference
through a hybrid RAG agent and ReAct workflow. However, these
approaches typically rely on a single vector database spanning all
ingested documents. While effective for broad retrieval, a unified
index can introduce a more subtle provenance challenge in which
performance data retrieved from one document may be incorrectly
associated with a topology described in another.
To address this issue, the proposed RAG framework adopts a
provenance-preserving retrieval strategy based on per-document
indexing. As shown in Fig. 2, each PDF is independently parsed,
chunked using a recursive text splitter, embedded using a private
embedding model, and stored in its own FAISS index. A temporary
combined search space is constructed only at query time. Because
every retrieved chunk retains its source-document identity through-
out retrieval and reasoning, topology descriptions, performance
metrics, and design insights remain explicitly traceable to their
original references. This design significantly reduces the risk of
cross-document attribution errors that can arise in conventional
merged-index RAG systems.
For the LNA case study, the retrieved design knowledge directly
influenced the topology agent’s architectural decisions. Specifically,
the agent selected a transformer-based matching network for op-
eration across the 60 GHz band, leveraging evidence from prior
designs that demonstrated wideband impedance matching at mm-
wave frequencies. In contrast, a conventional LC matching network

PREPRINT 3
PDF
Paper 1
Chunks
Embedding
Private
FAISS 1PDF
Paper 2
Chunks
Embedding
Private
FAISS 2PDF
Paper N
Chunks
Embedding
Private
FAISS N
LLM
reasoningVision Vision Vision
Performance Comparison Table
Figure 2: Proposed RAG framework for knowledge ingestion.
would provide a narrower operating bandwidth and would be less
suitable for the target specifications. The extracted knowledge base
comprises ten LNA design references spanning a range of architec-
tures. From these references, the most influential design priors were:
(1) transformer-based matching is a widely adopted solution for
wideband 60 GHz operation; (2) prior GF22FDSOI designs targeting
the 60 GHz bands demonstrate the suitability of the technology
node for mm-wave LNAs near the target specification range; and (3)
when gain is the performance-limiting metric, the output stage of-
ten dominates overall gain and therefore warrants additional design
emphasis. These retrieved priors provided an evidence-based foun-
dation for topology selection and reduced reliance on unsupported
design assumptions.
3.2 Topology Agent
The topology agent converts the target LNA specifications into a
schematic design strategy using the retrieved design knowledge
and the structured RF design rules. Although validated on LNAs in
this work, the agent architecture is extensible to other analog/RF
circuit classes through block-specific rules. Candidate topologies are
filtered using physics-based validity checks prior to optimization.
For the LNA case study, the following checks are applied.
Stage Count: For the LNA example, the agent uses PDK pri-
ors and power consumption budget to estimate whether a 2-stage
design can meet the gain targets before adding a third stage. For
the 15 mW power budget, a 2-stage design is preferred; 3-stage
escalation is allowed only after the 2-stage loop exhausts its budget
and the critic skill reports an unresolvable gain blocker.
Active Selection: In the LNA example, the first stage is selected
primarily for noise. The agent encodes the characterization of theGF22FDSOI device and treats the cascode as an isolation/gain mech-
anism. It also captures the lessons from the existing database.
Passive Selection: For the LNA case, the input network is as-
signed for NF and 𝑆11; interstage and output transformers are as-
signed for𝑆21and edge-band flatness. It also captures the lessons
from the existing database.
Prior Consistency: Retrieved priors as well as database, if any,
are used to initialize ranges. For example, in the LNA case study
the knowledge-ingestion layer identifies transformer-based fourth-
order networks as successful in wideband mm-wave LNAs, while
the optimizer is still free to tune inductance and coupling.
These checks are applied in a sequential order: stage count is
fixed first from the power budget, single-ended versus differential
operation is decided next using range of power budget, each active
device is then selected in turn (first stage for noise, later stages for
gain), and only once every active device is final, the agent chooses
the passive matching network.
3.3 Schematic and Testbench Generation Agent
The RFChipAgent flow separates cell validation, schematic genera-
tion, and testbench assembly into independently auditable stages,
as shown in Fig. 3.
Cell Validation: Before schematic generation, a validation skill
maintains inventories of active cells and passive matching-network
cells available within the design library. Active cells include neu-
tralized differential pairs, cascodes, and common-source stages, etc.
while passive cells include transformers, T-networks, 𝜋-networks,
and L-match structures, etc. All active cells expose a standardized
10-pin interface and all passive cells expose a standardized 7-pin
interface, allowing cells to be interchangeably instantiated within a
common module. Each candidate cell is verified for schematic exis-
tence, netlist availability, and schematic-validation status. Cells that
fail any verification step are flagged and automatically rebuilt until
a verified implementation is available, preventing invalid library
elements from propagating into downstream design stages.
Schematic Generation: Once the topology agent produces a
sequence of active and passive modules, the schematic-generation
agent automatically instantiates and connects the selected cells to
form the target LNA architecture. A topology-consistency check
is subsequently performed to verify connectivity and architectural
correctness. Detected violations trigger automatic reconstruction
attempts before the design is admitted to the optimization flow.
Testbench Assembly: The testbench-generation agent con-
structs a simulation-ready environment using a reference test-
bench template containing simulator directives, port definitions,
bias sources, and analysis configurations. The generated circuit
netlist is automatically inserted into this template to create a com-
plete verification setup. Missing or invalid upstream artifacts result
in explicit preflight failures, ensuring that incomplete designs are
detected before simulation and optimization.
3.4 Sizing Optimizer
The generated topology contains FETs that must be biased and sized,
as well as matching networks whose component values must be
optimized to satisfy the design targets. A multi-stage optimization
flow is therefore employed.

PREPRINT 4
Schematic &
Testbench Agent
generates
Schematic Component Registry
Module 1 Module 2 Module 3 Module 4 Module 5
In Out
Vbk k
Netlist
Schematic netlist + reference testbench wrapper netlist
Figure 3: Schematic and testbench agent generating the de-
sign.
Sigmoid-Normalized Multi-Objective Function: Combining
noise figure (NF), gain ( 𝑆21), input return loss ( 𝑆11), gain ripple, and
DC power consumption ( 𝑃𝐷𝐶) into a single objective function is
challenging for two reasons. First, these metrics exist on different
physical scales (dB versus mW). Second, some specifications are
maximization objectives ( 𝑆21) while others are minimization ob-
jectives (NF, 𝑆11, and𝑃𝐷𝐶). Directly averaging raw values would
distort the optimization process because numerically larger quanti-
ties could dominate the objective despite having similar margins
relative to their specifications. Likewise, commonly used figures
of merit (FoMs) do not fully capture circuit quality; for example,
a passive wire could achieve a deceptively high FoM compared to
an actual amplifier. Following the sigmoid-plus-harmonic-mean
formulation of [ 5], each metric is first mapped independently onto
a common quality scale between 0 and 1 using a sigmoid function,
𝜎(𝑧,𝑘)=1
1+𝑒−𝑧/𝑘(1)
where𝑧represents the margin relative to the specification and
𝑘controls the transition steepness. The metric quality is defined
according to the specification type:
𝑄max(𝑥)=𝜎 𝑥−𝑥 spec,𝑘, 𝑄 min(𝑥)=𝜎 𝑥spec−𝑥,𝑘.(2)
This formulation ensures that𝑄→1when the design satisfies its
specification and 𝑄→ 0when it violates the specification, regard-
less of whether the metric is a maximize-type or minimize-type
objective. Accordingly, 𝑄𝑆21uses𝑄max, while𝑄𝑁𝐹,𝑄𝑆11, and𝑄𝑃𝐷𝐶
use𝑄min. The flatness metric 𝑄𝑓𝑙𝑎𝑡additionally penalizes in-band
gain ripple once 𝑆21approaches its target value, ensuring ripple
optimization is only emphasized when meaningful. The steepness
parameter𝑘is selected individually for each specification according
to its design criticality. Smaller values of 𝑘produce a sharper tran-
sition centered at the specification boundary, creating a stronger
local gradient for hard constraints. Larger values spread the gra-
dient across a broader margin range, allowing the optimizer to
differentiate varying levels of violation without immediate satu-
ration. In the LNA validation example, 𝑘𝑁𝐹=0.4is used because
NF permits little margin, whereas 𝑘𝑆21=𝑘𝑆11=2.0is chosen topreserve useful optimization signal over several decibels of gain
or return-loss margin. The five quality metrics are then combined
using a harmonic mean,
𝑄total=5
1
𝑄𝑁𝐹+1
𝑄𝑆21+1
𝑄𝑆11+1
𝑄𝑓𝑙𝑎𝑡+1
𝑄𝑃𝐷𝐶,(3)
rather than an arithmetic mean. Because the harmonic mean is
dominated by its smallest term, it prevents designs from achieving
a high overall score by excelling in only a subset of metrics while
failing another.
Exploration and Exploitation Algorithms: The circuit de-
sign space may contain up to 21 optimization variables (15 for the
LNA in Section 4) spanning transistor dimensions, bias conditions,
and matching-network parameters. At this dimensionality, a single
optimization strategy is inefficient. Therefore, a two-stage optimiza-
tion approach is adopted. Tree-structured Parzen Estimator (TPE)
optimization is first used for global exploration because it efficiently
handles mixed continuous and discrete variables, including integer-
valued transistor finger counts. Once a promising region has been
identified, Covariance Matrix Adaptation Evolution Strategy (CMA-
ES) is applied for local exploitation and fine-grained refinement
of the continuous matching-network parameters around the best
solutions discovered by TPE.
Per-Trial Physical Validity Gates: An optimizer operating
solely on a scalar objective cannot distinguish between a physically
valid low-performing design and a physically invalid design that
violates fundamental circuit requirements. As a result, optimiza-
tion resources may be wasted exploring infeasible regions of the
design space. To address this issue, two physical-validity checks are
performed after each simulation trial and before objective-function
evaluation. First, a DC operating-point check verifies the region of
operation of every transistor. Any trial in which at least one device
falls outside saturation or subthreshold operation is immediately
discarded. Second, a stability check computes the Edwards–Sinsky
𝜇-factor from the extracted S-parameters over the frequency range
from 1 Hz to 90 GHz. Trials are rejected if 𝜇min<1or if the global
𝑆11exceeds 0 dB at any frequency within this range, indicating
potential instability. These validity gates prevent physically infeasi-
ble solutions from consuming optimization budget and ensure that
only realizable circuit candidates contribute to the search process.
3.5 Trust-Score Gated Database
RFChipAgent flow maintains a persisted database of every trial’s
parameter vector and outcome, partitioned into agoodset Gand
abadsetBby graded cost and trust status. Two non-parametric
density estimates are fit over the free-variable space, 𝑎(𝑥) over
Gand𝑏(𝑥) overB, and a candidate point is preferred when the
likelihood ratio 𝑎(𝑥)/𝑏(𝑥) is large: this identifies regions of the
design space that have historically produced good outcomes relative
to bad ones, rather than regions that are simply under-sampled.
This model can be audited and reused across campaigns. How-
ever, the model is only as reliable as the labels feeding it. Therefore,
a trial is only admitted to GorBif it clears the trust-score threshold.
Every run is assigned a trust score with 70% weight on execution
integrity and 30% on data provenance:
𝑇(𝑑)=0.7𝑇 𝑒𝑥𝑒𝑐(𝑑)+0.3𝑇 𝑝𝑟𝑜𝑣(𝑑).(4)

PREPRINT 5
Execution integrity is computed as a weighted sum of verification
indicators,
𝑇𝑒𝑥𝑒𝑐=∑︁
𝑖𝑤𝑖𝐼𝑖,(5)
where𝐼𝑖=1only when the corresponding check is satisfied. The
checks are PSF availability, nonzero 𝑃𝐷𝐶, full-band stability ex-
traction, and optional corner/EM coverage. Provenance separately
scores whether each reported number comes from verified Spectre
data, PDK-derived priors, estimates, or missing data.
Between campaigns, the LLM computes deterministic diagnos-
tics and what-if analysis over the completed trial database—search
stagnation since the last cost improvement, which free variables sit
pinned at a search-space boundary in the top trials, and whether
the best trial’s margin on any unmet specification sits on the steep
or flat region of that specification’s sigmoid penalty. For boundary-
pinned variables, the LLM helps in distinguishing a physical edge
(propose widening the bound) from a nonphysical edge, such as
a capacitance being pushed toward a negative value (propose re-
moving that component from the topology instead, a judgment a
numeric optimizer cannot make on its own).
4 Experiments and Evaluations
4.1 Experimental Setup
The RFChipAgent flow is validated end-to-end on a GF22FDSOI two-
stage LNA targeting 60 GHz wideband operation, chosen because
it exercises every stage of the loop—topology selection, schematic/
testbench assembly, closed-loop sizing optimizer, and trust-gated
database—under the challenging specifications characteristic of
RFICs for emerging wireless standards. The specification is 𝑆21min
≥25dB, NF max≤4.5dB,𝑆11max≤ −10dB, in-band ripple ≤
3.5dB, and𝑃𝐷𝐶≤15mW. The topology agent selects a neutralized-
differential-pair second stage cascaded with a common-source de-
generated first stage, matched with an input auto-transformer
and two transformer-based interstage/output networks, consistent
with the transformer-preference prior extracted by the knowledge-
ingestion agent. The sizing optimizer runs its TPE global phase
followed by CMA-ES local polish with simulator-in-the-loop for
every trial; every completed trial is passed through the two per-trial
validity gates to ensure correct DC operating point and stability.
The trust-score is computed before the data is written to the trust-
gated database, so every number reported below is drawn only
from trials that cleared both the physical-validity gates and the
trust threshold, not from raw simulator output.
4.2 Performance Results
Table 1 reports the five highest-ranked trials retained by the trust-
gated database, alongside the specification for direct comparison.
Trial #174 meets every specification simultaneously. Fig. 4 shows
the top five candidates’ 𝑆21,𝑆11, and NF traces across the full 60 GHz
band. Table 2 lists every free variable at the Trial #174 design point.
4.3 Efficiency and Computational Cost
Table 3 reports the aggregate token usage and wall-clock time
for RFChipAgent, attributed to superagent orchestrating every
step in Fig. 1—knowledge ingestion, topology selection, schematic/
testbench assembly, and optimizer—end to end. The campaign com-
pleted in approximately 2 hours of wall-clock time at an estimated
2224262830𝑆21(dB)(a)
specification ≥25dB
−35−30−25−20−15−10𝑆11(dB)(b)
specification ≤ −10dB
58 60 62 64 66 68 70
Frequency (GHz)3.63.84.04.24.4NF (dB)(c)
specification ≤4.5dBTrial #43 Trial #70 Trial #86 Trial #104
Trial #174 (meets all specifications)
1Figure 4: RFChipAgent flow output: performance of top 5
candidates.
cost of $156.20. Compared to manual labor-intensive circuit design,
this is a substantial improvement.
5 Conclusion
We presented a multi-agentic AI flow for end-to-end design automa-
tion of Analog/RF circuits. The flow decides a topology, generates
schematic and assembles testbench as well as optimizes the circuit.
It verifies every trial, and records results in a database. A private
FAISS-based RAG subsystem supports the knowledge ingestion. Us-
ing a GF22FDSOI 60 GHz wideband LNA as validation example, the
flow is able to meet target performance provided to it. RFChipAgent
achieves this result in wall-clock time of around 2 hrs with compu-
tation cost of $156.20—a considerable improvement over manual
design. RFChipAgent provides a practical path toward AI-native
analog/RFIC design automation.
Acknowledgments
The authors thank Qualcomm Technologies, Inc. WRD WAN RFIC
for discussions and Hasnain Lakdawala, Lincoln Leung, and Jeremy
Dunworth for useful feedback.

PREPRINT 6
Table 1: RFChipAgent flow top-5 proposed design candidates performance for GF22FDSOI 60 GHz wideband LNA. Specifications:
S21≥25 dB, NF≤4.5 dB, S 11≤−10 dB, Ripple≤3.5 dB, P DC≤15 mW
Performance metrics
Trial S 21 NF S 11 Ripple P DC Specification pass?
(dB) (dB) (dB) (dB) (mW)
Specification≥25≤4.5≤−10≤3.5≤15—
#43 21.7 4.2−7.9 2.6 13.3 No
#70 22.0 4.2−10.1 2.6 13.4 No
#86 24.1 4.3−11.8 1.4 13.2 No
#104 23.4 4.3−10.8 3.7 11.8 No
#174 27.1 4.2−16.9 3.5 14.3 Yes
Table 2: RFChipAgent flow proposed design candidate
variables for the GF22FDSOI 60 GHz wideband LNA (best
trial #174).
Description Unit Value
Active device sizing
Stage-1 FET finger count — 28
Stage-2 FET finger count — 16
Stage-1 gate inductance pH 174
Stage-1 noise-figure inductor pH 53.9
Stage-1 source degeneration inductor pH 50.1
Stage-1 gate-source coupling — 0.094
Bias currents
Stage-1 bias current𝜇A 866
Stage-2 bias current𝜇A 295
Interstage transformer
Coupling factor — 0.345
Primary inductor pH 285
Secondary inductor pH 105
Input auto-transformer
Shunt inductor pH 211
Output transformer
Coupling factor — 0.570
Primary inductor pH 220
Secondary inductor pH 169
References
[1]J. Gao, W. Fu, X. Guo, W. Cao, and X. Zhang. 2025. EVA: An Efficient and Versatile
Generative Engine for Targeted Discovery of Novel Analog Circuits. ACM/IEEE
Design Automation Conference (DAC).
[2]S. Wang, Q. Li, H. He, J. Gao, Z. Wang, Y. Sun, X. Zhang, T. Chi, and W. Cao. 2025.
Invited Paper: Multi-Agent Generative Synthesis for Analog/RF Circuit: from Scal-
able Topology Generation to Efficient Inverse Design. IEEE/ACM International
Conference on Computer-Aided Design (ICCAD).
[3]J. Gao, W. Cao, and X. Zhang. 2023. RoSE: Robust Analog Circuit Parameter Op-
timization with Sampling-Efficient Reinforcement Learning. ACM/IEEE Design
Automation Conference (DAC).Table 3: RFChipAgent flow estimated cost and computation
time
Metric Value
Input tokens 10,756
Cache write tokens 24,543,985
Cache read tokens 152,703,166
Output tokens 1,196,007
Est. cost $156.20
Wall-clock time∼2 hr
[4]W. Fu, S. Li, K. Yang, X. Zhang, Y. Jin, and X. Guo. 2025. Building Reasoning LLMs
for Hardware Design Generation via Function-Aligned Differentiated Revision.
IEEE/ACM International Conference on Computer-Aided Design (ICCAD).
[5] V. Chenna and H. Hashemi. 2026. A Nonintuitively Frequency-Staggered Wide-
band mm-Wave Low-Noise Amplifier. IEEE International Solid-State Circuits
Conference (ISSCC).
[6]P. Abbineni, S. Aldowaish, C. Liechty, S. Noorzad, A. Ghazizadeh Ghalati, and M.
Fayazi. 2026. MuaLLM: A Multimodal Large Language Model Agent for Circuit
Design Assistance with Hybrid Contextual Retrieval-Augmented Generation.
Asia South Pacific Design Automation Conference (ASP-DAC).