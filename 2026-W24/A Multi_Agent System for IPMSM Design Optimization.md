# A Multi-Agent System for IPMSM Design Optimization via an FEA-AI Hybrid Approach

**Authors**: Jinseong Han, Sunwoong Yang, Namwoo Kang

**Published**: 2026-06-08 05:07:46

**PDF URL**: [https://arxiv.org/pdf/2606.09037v1](https://arxiv.org/pdf/2606.09037v1)

## Abstract
Interior permanent magnet synchronous motor (IPMSM) design requires balancing conflicting objectives and multi-physics constraints, while modern optimization workflows face three bottlenecks: manual problem setup, high finite element analysis (FEA) cost, and unreliable surrogate-based search in sparse or out-of-distribution regions. To address these limitations, we propose an end-to-end automated IPMSM design optimization framework that integrates retrieval-augmented generation (RAG) for structured problem definition with an uncertainty-aware FEA-AI hybrid optimization pipeline. A Design agent, connected to a motor textbook through RAG, provides domain-knowledge-based options and engineering tips, and compiles an optimization card and a design-of-experiments plan for AI-model training. A Training agent automates electromagnetic FEA, records geometry-validation and solver-failure logs, analyzes failed geometries using ANOVA-based data analysis and LLM reasoning, and invokes a Design Sampling agent to redefine the design space and generate additional samples. An Optimization agent performs GA-based search with uncertainty-driven switching: low-uncertainty candidates are evaluated by AI-surrogate inference, whereas high-uncertainty and reliability-critical Pareto-front or top-K candidates are corrected by high-fidelity FEA and reused for iterative retraining. The framework converts manual, experience-dependent configuration into a reproducible workflow that balances computational cost and prediction reliability. Experimental results under a matched high-fidelity FEA budget show that the proposed hybrid approach achieves better objective performance while maintaining low and further reducible predictive uncertainty, outperforming FEA-only search, which is limited by early budget exhaustion, and AI-only search, which converges to a low-confidence optimum.

## Full Text


<!-- PDF content starts -->

A MULTI-AGENTSYSTEM FORIPMSM DESIGNOPTIMIZATION
VIA ANFEA-AI HYBRIDAPPROACH
Jinseong Han
Cho Chun Shik Graduate School of Mobility, KAIST
Daejeon, Republic of Korea
nethan1012@kaist.ac.kr
Sunwoong Yang
Department of Mechanical Engineering, Hanyang University
Ansan, Republic of Korea
sunwoongy@hanyang.ac.kr
Namwoo Kang
Cho Chun Shik Graduate School of Mobility, KAIST
Narnia Labs, Daejeon, Republic of Korea
nwkang@kaist.ac.kr
June 9, 2026
ABSTRACT
Interior permanent magnet synchronous motor (IPMSM) design requires balancing conflicting
objectives and multi-physics constraints, and modern optimization workflows face three coupled
bottlenecks. The first is the manual burden of defining design variables, objectives, constraints, and
analysis workflows, where the practical difficulty often lies in problem definition rather than the
optimizer itself. The second is the massive computational cost of finite element analysis (FEA)-
driven optimization, because genetic algorithm (GA)-based search repeatedly requires high-fidelity
simulations for many candidate geometries across generations. The third is the low reliability of
AI-surrogate-model-based GA optimization: although surrogate inference greatly reduces evaluation
time, predictive uncertainty in sparse or out-of-distribution regions can guide the search toward
low-confidence optima. To address these limitations, we propose an end-to-end automated IPMSM
design optimization framework that integrates retrieval-augmented generation (RAG) for structured
problem definition with an uncertainty-aware FEA–AI hybrid optimization pipeline. A Design
agent, connected to a motor textbook through RAG, provides domain-knowledge-based options
and engineering tips for problem definition and compiles an optimization card and a DOE plan for
AI-model training. A Training agent automates electromagnetic FEA, records geometry-validation
and solver-failure logs, analyzes fail geometries through ANOV A-based data analysis, and invokes a
Design Sampling agent through LLM reasoning to redefine the design space and generate additional
sampling points. An Optimization agent then performs GA-based search with uncertainty-driven
switching: low-uncertainty candidates are evaluated by AI-surrogate inference to reduce unnecessary
FEA calls, whereas high-uncertainty and reliability-critical Pareto-front or top-K candidates are
corrected by high-fidelity FEA and reused for iterative retraining. The framework thereby converts
manual, experience-dependent configuration into a reproducible workflow that balances computational
cost and prediction reliability. Experimental results under a matched high-fidelity FEA budget show
that the proposed hybrid approach achieves better objective performance while maintaining low and
further reducible predictive uncertainty, outperforming FEA-only search, which is constrained by
early budget exhaustion, and AI-only search, which converges to a low-confidence optimum.arXiv:2606.09037v1  [cs.AI]  8 Jun 2026

APREPRINT- JUNE9, 2026
KeywordsMulti-agent, FEA-AI hybrid model, Design optimization, Uncertainty quantification, Interior permanent
magnet synchronous motor (IPMSM)
1 Introduction
The demand for high power density and high efficiency in electric vehicles and industrial drives has increased the
importance of optimal design for interior permanent magnet synchronous motors (IPMSMs)[ 1,2]. However, practical
IPMSM optimization remains difficult because engineers must handle conflicting objectives and coupled multi-physics
constraints at the same time. Typical objectives include maximizing average torque and minimizing losses, torque
ripple, and cogging torque, while typical constraints include demagnetization margin, rotor-bridge stress/thickness,
voltage/current limits, and manufacturing minimum thickness[3, 4, 5].
Although optimization algorithms are mature, the major industrial bottleneck often appears in problem definition and
workflow operation. Engineers must repeatedly define design variables, objective/constraint formulations, analysis
conditions, and automation parameters before optimization can be run[ 6]. Junior engineers with limited motor-design
domain knowledge struggle with this high-dimensional setup process, and senior engineers still spend substantial
time rebuilding analysis/optimization settings whenever design requirements change. Recent studies on LLM-based
autonomous agents for design-engineering tasks indicate that general-purpose agents can improve interaction efficiency
but still require careful domain grounding to avoid generic or inconsistent technical guidance[ 7,8]. In many projects,
setup and operation tasks dominate total optimization lead time[9].
A second bottleneck is computational cost in high-fidelity FEA optimization. In population-based evolutionary
optimization, each individual in every generation typically requires a separate FEA run for fitness evaluation, so the
total simulation demand scales with population size ×number of generations and quickly becomes massive. Under
fixed computational budgets, reducing population size or generations narrows exploration and limits attainable solution
quality[10, 11].
To reduce this burden, many studies adopt AI-surrogate-model-based optimization (often coupled with GA) to replace
a large portion of expensive performance evaluations with low-cost predictions[ 12]. However, this introduces a
third bottleneck: reliability. Surrogate accuracy is not uniform across the design space and often degrades in out-
of-distribution (OOD) regions[ 13,14]. If prediction errors accumulate through iterative candidate selection and
model-guided updates, the optimization process can converge toward biased solutions.
FEA–AI hybrid optimization addresses this trade-off by using both FEA and AI during performance evaluation:
AI-surrogate inference handles most candidates at low cost, while FEA is selectively used for additional verification in
regions with high surrogate prediction error and for high-performing shapes such as Pareto-front candidates. In this way,
it reduces the heavy computational burden of pure FEA while preserving decision quality for critical candidates. Yet in
many existing hybrid workflows, when and where FEA is invoked is still determined by engineer-defined heuristics,
which makes consistent reproduction difficult across different problems[ 15,16]. To address this limitation, we design a
multi-agent-based FEA–AI hybrid model in which agents switch between surrogate inference and FEA using predefined
verification criteria, including checks in high predicted-error regions and validation of Pareto-front candidates, within
the optimization loop.
A further challenge arises during design sampling and FEA data generation itself: under uncertain or intentionally
wide variable ranges, many DOE-sampled geometries are infeasible (component intersections, non-manufacturable
or non-meshable shapes) and fail at geometry validation or FEA meshing, shrinking the analysis-feasible training
set[6]. Conventional pipelines typically cope only by discarding the failed samples or by imposing conservative bounds
that limit design-space exploration[ 17]. We instead close this feasibility gap with an autonomous, LLM-reasoning-
driven resampling loop that interprets geometry-validation and solver-failure logs together with an ANOV A-based
attribution of the failure-driving variables to autonomously redefine the design space and recover the requested number
of analysis-feasible samples—a use of LLM reasoning for sampling-to-FEA feasibility recovery not exploited in prior
motor-optimization frameworks.
To address these bottlenecks and the sampling-to-FEA feasibility challenge, we propose an end-to-end automated
IPMSM optimization framework that combines (i) RAG-guided problem definition and workflow automation and
(ii) an uncertainty-aware FEA–AI hybrid multi-agent pipeline. The RAG and automation components target the first
bottleneck by reducing problem-definition ambiguity and repetitive setup overhead. Unlike non-RAG prompting that
relies only on parametric memory and is more vulnerable to hallucinated or weakly grounded technical statements[ 18],
our approach injects motor-textbook/reference knowledge into the design conversation through retrieval-grounded
generation[ 19,20]. The multi-agent pipeline then automates the full process from optimization-card generation to DOE,
FEA data production, UQ-surrogate training, and evolutionary hybrid search. Most importantly, uncertainty-threshold-
2

APREPRINT- JUNE9, 2026
Figure 1: Overall framework of the proposed agent-based FEA–AI hybrid model for IPMSM design optimization.
The pipeline consists of a Design agent for problem definition and DOE generation, a Training agent for FEA data
generation and uncertainty-aware surrogate training, and an Optimization agent for evolutionary hybrid search with
uncertainty-threshold switching between surrogate and FEA evaluations.
based surrogate/FEA switching targets the second and third bottlenecks simultaneously by lowering high-fidelity FEA
demand while improving surrogate reliability, thereby strengthening hybrid optimization beyond ad hoc switching
heuristics.
Our main contributions are as follows:
•RAG-grounded problem-definition support for IPMSM optimization.We use retrieval-based domain
grounding to guide objective/target/variable/constraint definition in a structured chat workflow[19].
•Automated data generation for UQ surrogate training via LLM-reasoning resampling.We automate
the full data-generation pipeline (DOE sampling, FEA execution, and deep-ensemble training) and add an
autonomous, log-informed resampling loop that uses LLM reasoning to recover analysis-feasible samples from
failure-prone design spaces, securing a valid training dataset for the UQ surrogate despite frequent geometry
failures[17, 6, 14].
•Uncertainty-aware switching for reinforced hybrid GA operation.We introduce uncertainty-threshold
switching between surrogate and FEA evaluations, reducing high FEA computation demand while mitigating
low surrogate reliability to strengthen hybrid optimization under limited computational budgets[10, 15].
•Motor-targeted multi-agent optimization pipeline.We provide an integrated agent workflow specialized for
IPMSM optimization, covering problem definition, training-data generation, and final design search in one
system.
The remainder of this paper is organized as follows. Section 2 presents the proposed methodology, including the
Design, Training, and Optimization agents. Section 3 provides a RAG Study that compares Design-agent outputs with
and without motor-domain retrieval grounding. Section 4 demonstrates the operation of the log-informed resampling
loop on a representative problem-definition scenario. Section 5 presents the experimental setup, a switching-threshold
sensitivity analysis with a round-wise iteration study at the selected threshold, and comparative results under matched
conditions for FEA-only, AI-only, and hybrid strategies. Section 6 concludes the paper.
3

APREPRINT- JUNE9, 2026
2 Methodology
To address the challenges identified in Section 1, we propose an end-to-end agent-based FEA–AI hybrid framework for
IPMSM design optimization. As shown in Fig. 1, the proposed methodology has three stages: Step 1 (Design agent)
formalizes optimization requirements and generates DOE samples, Step 2 (Training agent) performs electromagnetic
FEA and trains a UQ-capable surrogate, and Step 3 (Optimization agent) runs uncertainty-aware hybrid evolutionary
search. In each stage, the top-level agent acts as an orchestrator that decomposes the stage objective into specialized
sub-agent tasks and coordinates the corresponding sub-agents—for example, problem definition and design sampling
sub-agents in Step 1, FEA/resampling/training sub-agents in Step 2, and inner-loop/outer-loop optimization sub-agents in
Step 3. Rather than operating as isolated modules, the agents assigned to the three steps exchange standardized artifacts
and can invoke one another when problem redefinition, resampling, surrogate update, or optimization refinement
is required, allowing the workflow to proceed through agent-to-agent interaction even without intermediate user
intervention; thus, the workflow functions as a fully integrated framework.
Figure 2: Role-assignment and chain-of-thought (CoT) prompt template used for the Design agent. The prompt specifies
the agent role, required stepwise reasoning sequence, and structured output format for optimization setup guidance.
2.1 Agent setup
The multi-agent system adopts GPT-OSS 20B as a lightweight open-source local LLM backbone and runs it through an
Ollama-based local serving environment. Compared with larger 60B-scale models or cloud-hosted proprietary services
(e.g., Gemini-, Claude-, or GPT-5-class APIs), this 20B-parameter model is easier to deploy in local engineering
environments with lower compute overhead while retaining sufficient instruction-following and reasoning capability
for structured agent tasks. This design is practical for confidentiality-sensitive motor-design projects because prompts,
intermediate artifacts, and optimization datasets can remain on-premises, while preserving a clear expansion path for
future domain adaptation through continued pretraining or fine-tuning on proprietary motor-design corpora.
Building on this local deployment setup, we implement agent orchestration with LangChain and assign each agent a
prompt-defined role using RAG+CoT templates with explicit input requirements, reasoning order, and structured output
constraints (Fig. 2). The Design agent uses retrieval-grounded prompting for objective/variable recommendation quality,
the Training agent uses stepwise prompts to combine ANOV A-based fail-geometry analysis with geometry-validation
logs and autonomously trigger Design Sampling agent calls, and the Optimization agent uses rule-constrained CoT
prompts for uncertainty-threshold switching during GA execution. User responses are converted into standardized
4

APREPRINT- JUNE9, 2026
vectors (e.g., objective priority, variable bounds, operating conditions, and uncertainty tolerance) and propagated to
downstream stages, reducing hallucination-prone guidance while improving end-to-end operational consistency.
Retrieval pipeline.The Design agent is grounded in a permanent-magnet motor reference [ 21]. The source document
is parsed page-by-page and filtered to remove front/back matter, table-of-contents, and bibliography/boilerplate text;
the remaining text is segmented into overlapping passages of about 800characters with a 120-character overlap using
recursive character splitting. Each passage is embedded with a local nomic-embed-text model ( 768-dim) and indexed
in a FAISS vector store; in parallel, a lexical BM25 index [ 22] is built over the same passages. At query time, the Design
agent retrieves through ahybrid retriever: the dense (embedding) and BM25 rankings are combined by reciprocal rank
fusion [ 23], and the top k=16 passages are injected into the agent’s chain-of-thought prompt as a dedicated grounding
block, so that objective/variable candidates and engineering tips are conditioned on retrieved motor-domain text rather
than on parametric memory alone [ 19]. Combining lexical and dense matching improves recall of both conceptual
passages and specific numerical values reported in the reference, as analyzed in Section 3.2. Generation uses the same
on-premises GPT-OSS 20B backbone, so the full retrieval-and-reasoning pipeline remains local.
Figure 3: Example interaction in the Design problem definition agent. The agent provides objective-function candidate
lists and engineering tips, and user responses are used to finalize the optimization problem definition.
2.2 Step 1: Agent-guided optimization problem definition (Design agent)
The Design agent orchestrates two sub-agents: a design problem definition agent and a design sampling agent. The
objective of this stage is to transform vague natural-language requests into a structured optimization card and a
DOE-ready sampling plan with minimal manual tuning.
Problem definition agent
The user first provides optimization requirements in natural language. A RAG-connected agent retrieves motor
textbook/reference knowledge and guides the user through a sequential decision flow: objective function →design
target→design variables →design space[ 19]. As illustrated in Fig. 3, the agent provides candidate lists and engineering
tips through a chat interface, and user responses iteratively refine the setup. The interaction outputs are encoded into a
structured setup vector (objective priorities, selected variables, variable bounds, and operating conditions) and passed to
downstream agents. The guidance includes:
• objective candidates and trade-off criteria,
• target-component candidates (rotor/PM/stator/coil) and relevance to selected objectives,
• variable candidates and expected influence on each objective,
• default constraints/bounds and risk notes.
Based on this interactive process (Fig. 3), the output is a structuredoptimization cardthat stores objectives, constraints,
design targets, design variables, nominal operating points, and metadata for downstream automation.
5

APREPRINT- JUNE9, 2026
Design sampling agent
Once the interaction with the Design agent is completed and the optimization problem is defined, the motor geometry is
parameterized and linked to DOE generation, as shown in Fig. 4. In Fig. 4(a), the parameterized IPMSM is described
using 12 geometric parameters: x1(gap between teeth tips), x2(teeth-tip height), x3(stator outer radius), x4(coil
height), x5(coil width), x6(PM angle), x7(PM length), x8(slot pitch), x9(void location), x10(void radius), x11
(rotor outer radius), and x12(rotor inner radius). This 12-parameter set is the geometry parameterization prepared for
the IPMSM case study in this work, not a fixed limitation of the framework. When users define additional geometry
parameters for a different motor template or a more detailed design study, the sampling agent can append their names,
bounds, and feasibility constraints to the optimization card, and the DOE generation/validation steps operate on the
expanded design vector. These parameterized variables directly affect internal magnetic reluctance, magnetization-
direction-dependent flux paths, and equivalent magnetic-circuit characteristics, and therefore strongly influence IPMSM
electromagnetic performance[11, 4]. Fig. 4(b) shows sampling points generated by DOE in this parameter space[17].
Given the optimization card, the sampling agent defines variable-wise min/max bounds through user interaction. When
users can specify reliable design ranges, the entered min/max values are used directly for DOE generation. When users
are uncertain about appropriate ranges, the agent initializes provisional default bounds from the parameterized geometry
template and domain-rule database, and these bounds can later be autonomously redefined through the log-informed
resampling feedback loop described in Step 2. Sample counts are then determined according to variable dimension and
computational budget. The user-selected design variables are perturbed to generate multiple design candidates through
DOE, while the remaining non-selected parameters are fixed at their default values before forwarding the DOE set to
the Training agent.
Figure 4: Design-sampling results after Design-agent interaction. (a) Parameterized IPMSM geometry used for variable
definition. (b) DOE-generated sampling points in the selected design-variable space.
2.3 Step 2: Dataset generation and surrogate training with uncertainty (Training agent)
The Training agent orchestrates infeasible-case handling/resampling, electromagnetic analysis, preprocessing of
electromagnetic-analysis outputs into a trainable dataset, and surrogate learning. This stage converts the design intent
from Step 1 into a validated training dataset and a UQ-capable predictive model.
Resampling process
The resampling process is implemented as an autonomous agentic feedback loop, not as a one-time failure filter or a
purely user-driven correction stage. Before each FEA batch, the Training agent performs rule-based geometry validation
on the DOE candidates generated from the current design-space definition. This validator checks both input consistency
and deterministic geometric feasibility, including incompatible variable-bound combinations, radial ordering, air-gap
clearance, minimum bridge/thickness, PM/void/stator/rotor intersections, and parameter ranges that make the geometry
non-manufacturable or non-meshable. Candidates that violate these rules are labeled as geometry-fail samples, and
6

APREPRINT- JUNE9, 2026
their sample ID, design-variable vector, violated rule, failed component relationship, and violation margin are stored in
a geometry-validation log file. This log preserves detailed evidence about why each sampled geometry failed before
expensive FEA execution.
Figure 5: Examples of failed geometries from DOE sampling. Cases (a)–(c) show different invalid shapes caused by
component intersections, while case (d) shows a geometry protruding outside the rotor boundary. These failures make
electromagnetic FEA infeasible.
For candidates that pass the rule-based validation, electromagnetic FEA is then executed by the FEA agent described in
the next subsection. Some geometries can still become unsimulatable because of solver-level meshing failure or invalid
geometric configurations among rotor/stator/void/PM parts, as illustrated in Fig. 5. These solver-level failures are also
recorded together with the validation-failure logs, including solver status, mesh-error messages, and the corresponding
design-variable values. This issue is often amplified when the initial design space is wide or when aggressive sample
counts are used in Step 1. In such cases, many failed samples reduce the number of valid data points, so the target
sampling count for AI-model training cannot be fully secured.
To resolve this issue, the framework includes a resampling feedback loop after the initial validation and FEA run. Failed
and valid cases are separated automatically, and ANOV A-based probabilistic analysis quantifies how each variable and
variable interaction contributes to the observed failure probability. Unlike a simple statistical filter, the Training agent
combines this ANOV A result with the recorded geometry-validation and solver-failure log files. The LLM then reasons
over both sources of evidence to identify fail-prone variable ranges, high-risk variable interactions, repeated violation
types, and safe regions that should be preserved for additional sampling.
Based on this reasoning result, the Training agent autonomously invokes the Design Sampling agent, which redefines
the design space while maintaining the original optimization requirements and deterministic design constraints. The
Design Sampling agent narrows or shifts failure-prone variable bounds, removes invalid variable-combination regions,
and determines the number of additional sampling points required to recover the target valid-sample count. It then
generates additional DOE candidates in the revised feasible design space and returns them to the Training agent for
another validation–FEA cycle.
These findings and actions are delivered as a PDF analysis report (Fig. 6) so the user can inspect the failure mechanisms,
ANOV A results, referenced log evidence, and automatically revised sampling plan. As shown in Fig. 7, the loop
proceeds through agent-to-agent interaction: validation logs and ANOV A results are passed to the LLM reasoning
module, the Design Sampling agent is called to update the design space and produce additional points, and the newly
validated samples are appended to the training dataset. This loop continues until the target number of analysis-feasible
7

APREPRINT- JUNE9, 2026
Figure 6: Example PDF analysis report provided by the agent during the resampling process. The report summarizes
fail-geometry patterns, geometry-validation logs, and variable-wise failure tendencies identified from ANOV A-based
probabilistic analysis.
sampling points is secured or a predefined resampling limit is reached, after which the valid resampled points are
forwarded to the surrogate-training stage of the Training agent.
FEA agent
The agent receives both the DOE sampling points generated in Step 1 and the additional sampling points obtained
through the resampling process, and performs high-fidelity electromagnetic analysis in Ansys Maxwell 2D. As illustrated
in Fig. 9, the Training agent first interacts with the user to determine analysis control points (e.g., current, speed, and
operating condition). The default control-point setting is initialized at the rated operating point, but users can redefine
these values according to target requirements before batch execution. Based on the finalized control points and the
Step 1 sampling set, the agent runs electromagnetic FEA for each sample and extracts core electromagnetic responses,
including flux linkage and iron loss.
The electromagnetic analysis is governed by Maxwell’s equations in their time-domain and harmonic forms, as solved
by Ansys Maxwell 2D [ 6]. In the proposed framework, Maxwell 2D project generation, parameter updates, solver
execution, and result extraction are performed through a code-driven automation layer based on the PyAEDT library,
enabling the Training agent to execute repeated DOE analyses without manual GUI operation. The key field quantities
extracted are the magnetic flux densityBand the associated loss integrals, which are post-processed under sinusoidal
steady-state excitation F(t) =F mcos(ωt+θ) . Average electromagnetic torque is computed from the Maxwell stress
tensor:
Tavg=ragLstk
µ0I
ΓBrBθdΓ(1)
where ragis the air-gap radius, Lstkis the stack length, and Br,Bθare the radial and tangential flux densities on
integration contourΓ. Core (iron) loss is computed by the modified Bertotti model:
Pcore=khfBα
max+kc(fB max)2+ke(fB max)1.5(2)
where kh,kc,keare the hysteresis, classical-eddy-current, and excess-loss coefficients, respectively, and αis the
Steinmetz exponent fitted to the lamination material data.
The field solutions are post-processed to compute the objective values defined in the optimization card (e.g., torque-
related metrics and loss/efficiency indicators)[ 6]. The final output is a consistent simulation dataset containing design
variables, operating conditions, solver status, and objective values, which is directly used for surrogate training.
8

APREPRINT- JUNE9, 2026
Figure 7: Autonomous resampling workflow after fail-geometry analysis. The Training agent combines ANOV A-based
failure analysis with geometry-validation log files, uses LLM reasoning to invoke the Design Sampling agent, and
automatically redefines the design space to generate additional sampling points.
Training agent
The surrogate training agent uses the valid electromagnetic dataset collected from the previous FEA stage and converts
it into a trainable form for objective-function prediction. Specifically, after Step 1 defines the optimization targets, the
agent aligns each sample’s geometry vector with the corresponding FEA-derived labels (e.g., torque- or loss-related
objective values selected by the user). Invalid or failed samples are excluded through the resampling logic, and the
remaining dataset is preprocessed, cleaned, and normalized so that the learning process is numerically stable and
consistent across different variable scales.
As shown in Fig. 8, panel (a) presents the default deep-ensemble surrogate architecture composed of five parallel MLP
regressors (MLP ×5). Each MLP uses a 12–64–64–2 structure with ReLU activations in the hidden layers, while panel
(b) presents an example prediction result with uncertainty when the objective function is core loss[ 14]. Each network
is trained on the processed electromagnetic dataset, and ensemble statistics are used to obtain both prediction and
uncertainty. The input to the surrogate is a 12-dimensional parameterized IPMSM design vector, and the two output
nodes estimate the predicted objective-function value and predictive variance for the corresponding geometry. In this
work, the surrogate outputs are represented as (µ(x), σ2(x)), where µ(x) denotes the ensemble mean prediction and
σ2(x)denotes predictive variance.
Following the uncertainty interpretation of deep ensembles and Bayesian deep learning, this predictive variance is
decomposed into aleatoric and epistemic components[ 14,24]. For M= 5 ensemble members, let µm(x)andσ2
m(x)
denote the mean and variance predicted by member m. The ensemble mean and total predictive variance are computed
as
µ(x) =1
MMX
m=1µm(x), σ2(x) =1
MMX
m=1σ2
m(x)
|{z }
aleatoric+1
MMX
m=1(µm(x)−µ(x))2
| {z }
epistemic.(3)
9

APREPRINT- JUNE9, 2026
The aleatoric term represents data- or label-related uncertainty that remains even near observed samples, whereas the
epistemic term represents model uncertainty caused by limited training data and therefore tends to increase in sparse or
OOD regions. This decomposition provides the basis for using σ2(x)and the derived CV metric as reliability indicators:
candidates with high uncertainty are treated as low-confidence predictions and are selectively corrected by FEA in the
hybrid optimization loop.
To train heteroscedastic uncertainty stably, each MLP is optimized using the β-NLL loss rather than a standard Gaussian
NLL. As reported in the referenced β-NLL study, standard NLL-based heteroscedastic training can suffer from unstable
mean–variance coupling because the mean-gradient term is inversely scaled by the predicted variance; consequently,
high-variance samples receive a smaller effective learning signal and can be underlearned. The β-NLL formulation
mitigates this issue by multiplying the NLL term by a stop-gradient variance-dependent factor, thereby increasing the
relative weight of high-variance samples and preventing their learning contribution from being overly attenuated[ 25].
The loss for a sample(x, y)is defined as
Lβ-NLL =1
NNX
i=1
sg 
ˆσ2(xi)β×"
1
2log ˆσ2(xi) +(yi−ˆµ(x i))2
2ˆσ2(xi)+1
2log(2π)#
, (4)
where ˆµ(xi)andˆσ2(xi)are the network outputs for sample i,Nis the batch size, βcontrols the variance-based
reweighting strength, and sg(·) denotes stop-gradient. Following the referenced β-NLL study, βis set to 0.5 in this
work[ 25]. Early stopping based on validation β-NLL is also applied to prevent overfitting and unstable uncertainty
estimates.
Surrogate generalization is evaluated on a held-out test set (20% random split with seed 7, stratified by feasibility) after
training. For the core-loss prediction task used in the optimization experiment, the deep-ensemble model achieves
R2= 0.973 andRMSE = 0.84 W (mean absolute error MAE = 0.61 W ). These metrics confirm sufficient accuracy
for GA-based fitness ranking, and the predictive variance σ2(x)reliably identifies out-of-distribution design candidates
that require FEA verification.
Figure 8: Deep-ensemble surrogate in the Surrogate training agent. (a) Architecture of the deep-ensemble model
composed of five MLPs trained on preprocessed and normalized electromagnetic FEA data. Each MLP follows a
12–64–64–2 structure. (b) Example prediction graph showing the model output and UQ when the objective function is
core loss. Given a 12-dimensional IPMSM design-variable vector, the ensemble predicts user-selected objective-function
values and provides uncertainty estimates.
Early stopping is enabled during surrogate training, and layer/node settings can be adjusted through user-agent
interaction when the optimization problem is reconfigured. This configuration enables the framework to provide
not only fast objective prediction for unseen samples but also confidence-aware guidance for downstream hybrid
optimization, where uncertain regions can be selectively re-evaluated by high-fidelity FEA.
10

APREPRINT- JUNE9, 2026
Figure 9: Electromagnetic analysis stage in Step 2. The Training agent interacts with the user to set analysis control
points, then runs electromagnetic FEA for the Step 1 DOE sampling points. Default control points are initialized at the
rated operating point and can be redefined by the user.
2.4 Step 3: Uncertainty-aware hybrid optimization (Optimization agent)
Using the optimization card from Step 1 and the trained deep-ensemble surrogate from Step 2, the Optimization agent
conducts evolutionary design search through two coordinated sub-agents: an inner loop agent and an outer loop agent.
The optimizer is selected according to the objective type: a standard GA is used for single-objective optimization,
whereas NSGA-II is used for multi-objective optimization. The inner loop agent performs fast generation-level search
using only the AI surrogate for routine fitness evaluation, whereas the outer loop agent monitors optimization-level
uncertainty and invokes the FEA–AI hybrid evaluator when reliability-critical samples must be checked. This separation
allows the framework to retain the exploration speed of surrogate-assisted evolutionary search while improving
confidence in the initial population, Pareto-front candidates, and final recommended designs.
Inner loop agent
The inner loop agent manages the generation-wise evolutionary search. Starting from a validated population, each
generation applies evolutionary operators—selection, crossover, and mutation—to produce offspring design-variable
sets. Because these operators can generate infeasible motor geometries, the resulting candidates are first passed through
geometry validation before fitness evaluation. Candidates that violate deterministic feasibility rules, such as component
intersections, invalid radial ordering, air-gap violations, or minimum-thickness constraints, are filtered out as failed
geometries. To preserve the intended population size, the agent generates additional design-variable sets and repeats
validation until the missing feasible candidates are replenished.
After geometry validation, the inner loop agent evaluates fitness using only the AI surrogate model. No FEA is called
inside this routine inner loop; instead, the surrogate provides fast objective estimates for all feasible offspring. The
evaluated offspring population is then merged with the parent population, and the union of parents and offspring is
ranked for survivor selection. For multi-objective optimization, this ranking produces a Pareto-front candidate set; for
single-objective optimization, it produces the top- Kcandidate designs with the best predicted objective values. These
candidates become the output of the inner loop and are passed to the outer loop agent for uncertainty-aware reliability
checking and possible reuse in the next optimization round.
Outer loop agent
The outer loop agent controls the optimization process across multiple optimization rounds. At the beginning of
each round, an initial population is generated by Latin hypercube sampling (LHS)[ 17] and then filtered by the same
geometry-validation rules used in the inner loop, so only feasible motor geometries enter the optimization process.
From the second round onward, the Pareto-front candidates from a multi-objective search or the top- Kcandidates from
a single-objective search are inserted into the next round’s initial population together with newly sampled feasible
candidates. This reuse mechanism preserves promising designs while allowing the next optimization round to explore
additional regions of the design space.
Unlike the inner loop, the outer loop agent evaluates reliability-critical samples using the FEA–AI hybrid model shown
in Fig. 11. The hybrid evaluator is applied to the initial population and to the Pareto-front or best-candidate set returned
by the inner loop. For each candidate x, the surrogate predicts both the objective value µ(x) and predictive variance
σ2(x). Following the relative-uncertainty interpretation of the coefficient of variation (CV)[ 26], the variance output is
11

APREPRINT- JUNE9, 2026
Figure 10: Inner/outer loop execution strategy in the Optimization agent. The inner loop performs routine evolutionary
operations with geometry validation and AI-surrogate-only fitness evaluation, while the outer loop evaluates reliability-
critical populations and Pareto-front (top- K) candidates using the uncertainty-aware FEA–AI hybrid model, iterating
until the Pareto-front (top-K) mean uncertainty falls below the switching threshold.
converted to its corresponding square-root variance scale only for CV computation, and the uncertainty signal used by
the outer loop is defined as
CV(x) =p
σ2(x)
|µ(x)|×100 [%],(5)
where CV(x) denotes the CV-based predictive uncertainty in percent. The agent then compares this uncertainty with
the hybrid-switching thresholdCV Th,hybrid and switches the evaluator according to
•CV(x)≤CV Th,hybrid : use AI-surrogate prediction for fitness evaluation,
•CV(x)>CV Th,hybrid : run high-fidelity FEA, use the FEA result as the label, and append the sample to the
training dataset.
The hybrid-switching threshold CVTh,hybrid is not fixed in advance; it can be modified through user–agent interaction
according to the desired trade-off between FEA cost and surrogate reliability. The experimental setup in Section 5
specifies the threshold used in this study. Whenever FEA is invoked, the newly labeled samples are reused for
online surrogate updating, which reduces uncertainty in high-risk regions and improves the reliability of subsequent
optimization rounds[10, 14, 15].
The outer loop also evaluates a round-level termination criterion based on the reliability of the candidate optimal designs
rather than of the entire round population. To this end, it computes the mean CV over the Pareto-front candidates for a
multi-objective search, or over the top- Kcandidates for a single-objective search, because these candidates are the
ones that directly determine the final engineering decision. This Pareto-front (top- K) mean uncertainty is denoted as
CVPareto , and the agent compares it with the same hybrid-switching thresholdCV Th,hybrid according to
•CV Pareto >CV Th,hybrid : proceed to the next optimization round by resampling additional feasible candidates,
inserting the previously obtained Pareto-front or top- Kcandidates into the new initial population, and rerunning
the inner loop,
•CV Pareto≤CV Th,hybrid : stop the iterative optimization process and obtain the final optimal design from
the current Pareto-front candidates for multi-objective optimization or from the current top-Kcandidates for
single-objective optimization.
The same termination rule is applied when the available FEA budget is exhausted, in which case the best available
Pareto-front or top-Kcandidates are used for final design selection.
During this iterative outer-loop process, the FEA-labeled samples accumulated by the FEA–AI hybrid evaluator are
merged with the original training dataset generated by the Training agent. The AI surrogate is then fine-tuned for
30 epochs using the updated dataset before the next optimization round begins. Because each outer-loop round adds
informative high-uncertainty FEA labels and updates the surrogate accordingly, the predictive uncertainty decreases
12

APREPRINT- JUNE9, 2026
progressively whenever a new optimization round is executed. Consequently, the final optimization is performed in a
more reliable surrogate environment, enabling the framework to obtain an optimal design with both high performance
and high confidence. Through this structure, the FEA–AI hybrid evaluator is concentrated on the initial population and
the final candidate front, improving confidence in both the full optimization loop and the selected optimal designs while
avoiding unnecessary FEA calls during routine generation-level search.
Figure 11: Uncertainty-aware FEA–AI hybrid evaluator used by the outer loop agent. The evaluator compares predictive
uncertainty against a threshold and switches between AI-surrogate inference and high-fidelity FEA for reliability-critical
candidates such as the initial population and Pareto-front or top-Kdesigns.
3 RAG Study
To verify whether domain-grounded retrieval improves early-stage problem definition, we conducted a controlled
comparison on the Design agent output direction. The same user intent, interaction budget, and target motor context
were provided to two settings: (i) a general-purpose local LLM (GPT-OSS 20B) without domain retrieval and (ii) a
motor-textbook-connected RAG Design agent. The evaluation focuses on practical problem-definition quality rather
than linguistic fluency.
3.1 Comparing output direction in the Design agent
The comparison is designed to answer a core question: does a motor-domain RAG pipeline improve optimization
setup quality over a general model? Two evaluation axes are used. First,optimization-card completenessis checked
by whether essential fields are consistently and explicitly generated, including objective/constraint definitions, design-
variable lists and bounds, operating conditions, fail-handling logic, and DOE settings. Second,list/tip qualityis assessed
by engineering usefulness, physical consistency, and actionability from an IPMSM design perspective.
To make this comparison concrete, we additionally conduct a question-level response analysis using one fixed user
prompt:“In an IPMSM, why is the q-axis inductance larger than the d-axis inductance ( Lq> L d), and what is
reluctance torque?”This question requires correct electromagnetic reasoning rather than generic motor knowledge, so
it cleanly separates retrieval-grounded answers from parametric ones. Figure 12 compares the generated answers from
two settings under the same local GPT-OSS 20B backbone: (a) without RAG and (b) with motor-textbook RAG.
Baseline: GPT-OSS 20B (no RAG)
The general model typically produces fluent and plausible responses, but the technical grounding is often weak for
IPMSM-specific reasoning. As observed in Fig. 12(a), the no-RAG answer reaches the correct conclusion ( Lq> Ld)
13

APREPRINT- JUNE9, 2026
through a physically inverted argument: it states that ahigherreluctance along the q-axis yields alarger q-axis
inductance, which contradicts the inverse relationship between magnetic reluctance and inductance, and it attributes the
smaller d-axis inductance to a magnet-flux “short circuit” rather than to the high reluctance of the magnet path. Such
fluent but mechanistically incorrect explanations are difficult for a non-expert user to detect, and when propagated into
objective/variable rationale they require substantial expert correction before the optimization card can be trusted.
Figure 12: Response comparison for an IPMSM saliency question in the RAG study. The user asks why the q-axis
inductance exceeds the d-axis inductance ( Lq> Ld) and what reluctance torque is. (a) General GPT-OSS 20B without
RAG returns a fluent but mechanistically incorrect explanation (flawed reluctance–inductance reasoning). (b) GPT-OSS
20B connected to a motor textbook via RAG correctly attributes Lq> Ldto the low-reluctance q-axis flux path through
the rotor steel versus the high-reluctance d-axis path through the magnets and air gap, and defines reluctance torque
accordingly; the retrieval source includes [21].
Proposed: RAG design agent
When textbook-based retrieval is enabled, the Design agent generates candidate lists and engineering tips with stronger
domain traceability. In Fig. 12(b), the RAG-connected response gives the physically correct mechanism: the q-axis
flux flows through the low-reluctance rotor steel between and around the magnets, whereas the d-axis flux must cross
the high-reluctance magnets and air gap, so that Lq> Ld; it then defines reluctance torque as the component arising
from this d/qpermeance (inductance) difference. Because the explanation is grounded in magnetic-circuit reasoning
rather than surface association, it is directly reusable for optimization setup—for example, in deciding how rotor/PM
geometry variables should be co-interpreted with the saliency-driven torque component instead of being treated as
isolated scalar inputs. This grounding, supported by textbook retrieval [ 21], improves both the completeness and the
practical usefulness of the optimization card while reducing hallucination-prone guidance for non-expert users.
3.2 Quantitative ablation on textbook-grounded questions
To complement the qualitative analysis above, we evaluate two local open backbones (GPT-OSS 20B and Llama-3 8B)
with and without textbook retrieval, under identical decoding ( T=0 ), on a fixed set of 90domain-specialized questions
derived from the motor reference [ 21]: (i)30numericalproblems adapted from worked examples, whose ground truth
is recomputed from the textbook formulas (e.g., winding factor, Carter’s coefficient, back-EMF, cogging-torque index);
(ii)30book-specificlookups, i.e., specific numerical values that appear only inside particular worked examples and
cannot be inferred from parametric knowledge; and (iii) 30conceptualquestions on the textbook’s defined coefficients
and design rules (e.g., saliency/reluctance torque, Carter’s coefficient, the cogging-torque index, the effective air gap).
Numerical and book-specific answers are judged correct within a per-item tolerance of the ground truth ( ±1–6%);
conceptual answers are scored against fixed gold key points by an LLM judge (GPT-OSS 20B), applied identically
to both settings so that any judging bias cancels. Retrieval quality is measured by Hit@ k(a passage containing the
14

APREPRINT- JUNE9, 2026
answer appears among the top k=16 retrieved chunks). The RAG condition uses the Design agent’s hybrid retriever
(dense + BM25 with reciprocal rank fusion,k=16), shared by both backbones.
Table 1 summarizes the results. Textbook retrieval improves both backbones on this domain-specialized set, with
the largest gains on knowledge the closed-book model cannot possess. For GPT-OSS 20B, RAG raises numerical
accuracy from 43% to77%, book-specific accuracy from 3%to67%, and conceptual accuracy from 47% to80%. For
the smaller Llama-3 8B, RAG raises book-specific accuracy from 0%to43% and conceptual accuracy from 23% to
60%; its numerical accuracy improves only marginally (from 13% to20%) because the multi-step arithmetic exceeds
the model’s capability even when the correct method is retrieved. A relevant passage is retrieved for 86% of items
(shared by both models). The book-specific gains—from near zero without retrieval—show that RAG supplies the
worked-example knowledge the base models cannot possess, while the conceptual and numerical gains additionally
depend on the backbone’s reasoning capability.
Table 1: RAG ablation on 90domain-specialized textbook questions for two local backbones (identical decoding, T=0 ).
RAG denotes the Design agent’s textbook retriever (dense + BM25, top- 16); Hit@16 (shared by both backbones) is the
retrieval quality. “acc.” = accuracy;↑higher is better.
Backbone Setting Numerical Book-specific Conceptual Hit@16
GPT-OSS 20B No-RAG43% 3% 47%—
GPT-OSS 20B RAG77%67%80% 86%
Llama-3 8B No-RAG13% 0% 23%—
Llama-3 8B RAG20%43%60% 86%
Figure 13: Per-backbone answer accuracy by question type, without retrieval (No-RAG) and with textbook retrieval
(RAG, dense+BM25 top-16), on the 90-question domain-specialized set. Retrieval lifts both GPT-OSS 20B and Llama-3
8B, most strongly on book-specific and conceptual questions (shared retrieval Hit@16= 86%).
Retrieval analysis and limitations.Breaking Hit@ kdown by question type isolates where retrieval helps. Conceptual
and numerical queries are retrieved reliably (Hit 100% and83%), which explains most of the accuracy gain. Book-
specific value lookups remain the hardest to retrieve (Hit 73%): the answer-bearing passage is one of many topically
similar worked-example chunks, and a dense-only ranking can place it outside the retrieved set because its embedding
is degraded by surrounding equation text extracted from the PDF. Weighting the lexical BM25 ranking in the reciprocal-
rank fusion lifts such passages back into the top k=16 , so that the retrieval-grounded GPT-OSS 20B answers 67% of
the book-specific lookups versus 3%for the closed-book model. The residual failures occur when the numerical answer
is split from its identifying context across chunk boundaries or when equation extraction corrupts the value, which
motivates equation- and table-aware indexing as future work. Based on these results, the hybrid-retrieval RAG Design
agent is adopted as the default problem-definition interface in the proposed framework.
15

APREPRINT- JUNE9, 2026
4 Resampling Process
Section 3 fixed the problem-definition interface, and the autonomous resampling loop was introduced as part of the
Training agent in Step 2 (Section 2). This section demonstrates the operation of that log-informed resampling loop on a
concrete user-defined scenario and reports how it recovers the user-requested number of analysis-feasible sampling
points from a low-yield initial design space.
Figure 14: Operation of the log-informed resampling loop for the demonstration scenario (objective: iron loss; all
twelve variables x1–x12; target: 100valid samples). (a) ANOV A effect size η2of each design variable at the initial
(wide) design space; the failures concentrate on x6, x7, x9, x10, x12, which control the inner magnet, the flux-barrier
void, and the rotor core. (b) Geometry success ratio (orange) and cumulative valid-sample count (blue bars) versus
resampling stage; the autonomous loop raises geometry feasibility from 28% to84% and secures exactly the requested
100valid samples in three iterations.
4.1 Demonstration scenario and initial sampling
Through interaction with the Design agent and the Design Sampling agent, the user specifies a single-objective problem
that minimizes iron loss, selects all twelve geometry parameters x1–x12defined in Fig. 4(a) as design variables, and
requests 100analysis-feasible sampling points for surrogate training. Because the user does not provide reliable
per-variable ranges, the Design Sampling agent initializes provisional default bounds from the parameterized geometry
template and generates an initial Latin-hypercube DOE of100candidates over this intentionally wide design space.
4.2 Fail-geometry analysis at the initial design space
The Training agent applies the rule-based geometry validation of Step 2 to the initial DOE. Because the default bounds
are wide, only 28% of the 100candidates ( 28samples) are analysis-feasible, so the requested valid count cannot be met
from the initial sampling alone. Each failed candidate is logged with its violated rule, the failed component relationship,
and the violation margin, and ANOV A-based probabilistic analysis quantifies how strongly each variable is associated
with the observed failures. Figure 14(a) reports the resulting variable-wise effect size η2. The failures concentrate on five
variables— x6(PM angle), x7(PM length), x9(void location), x10(void radius), and x12(rotor inner radius)—whereas
the remaining variables show negligible association; x10contributes mainly through its interaction with the void location
x9, which lowers its marginal effect size. These variables correspond exactly to the geometry objects that the validation
log records as invalid or mutually intersecting—the inner magnet ( PM_I1 ), the rotor flux-barrier void ( void1 ), and
the rotor core ( Rotor )—so the statistical signal is consistent with the recorded geometric failure mechanisms such as
void1∩Rotor andPM_I1∩Rotor . A small, variable-independent fraction of the failures reflects solver-level meshing
issues that cannot be removed by design-space refinement.
16

APREPRINT- JUNE9, 2026
4.3 LLM-driven design-space refinement
The Training agent forwards the geometry-validation log and the ANOV A result to the LLM reasoning module, which
is additionally provided with an explicit object-to-variable map so that each failing geometry object is translated into the
variables that control it ( void1→ {x 9, x10},Rotor→ {x 11, x12},PM_I1→ {x 6, x7}). Reasoning over this combined
evidence, the agent autonomously invokes the Design Sampling agent to redefine the design space: it narrows the
bounds of the fail-prone variables away from the high-risk regions while preserving the low-influence variables to
maintain sampling diversity, and it determines the number of additional candidates required to recover the target valid
count. Figure 15 traces the bounds of the five fail-prone variables across iterations and shows that their active ranges are
progressively tightened toward the feasible region—for example, the upper bound of the void location x9is reduced
from 65to about 55and that of the PM angle x6from 45to about 33—whereas the original envelope is retained for the
variables without failure evidence.
4.4 Convergence to the target valid-sample count
The redefined design space is returned to a new validation cycle, and the loop repeats until the requested number of
analysis-feasible points is secured. Figure 14(b) shows the resulting convergence. The geometry success ratio rises
monotonically from 28% at the initial design space to 60%,71%, and 84% over three successive refinements, while
the cumulative valid-sample count increases accordingly ( 28→52→84→100 ). The loop terminates automatically
at the third iteration, as soon as the requested 100valid points are secured, and only the requested number of valid
samples is retained, so the dataset is neither under- nor over-sampled. The complete demonstration, including every
LLM reasoning call, runs in about 420s on the local GPT-OSS 20B backbone, after which the 100analysis-feasible
geometries are forwarded to the electromagnetic-analysis and surrogate-training stages of the Training agent.
Figure 15: LLM-driven design-space refinement across resampling iterations for the five fail-prone variables. At
each iteration the Design Sampling agent narrows the active [min,max] range (blue band) toward the feasible region
while keeping it inside the original envelope (gray), guided by the ANOV A result and the object-to-variable map;
low-influence variables are left unchanged to preserve sampling diversity.
5 Experimental Setup and Results
With the problem-definition interface fixed in Section 3 and the 100analysis-feasible samples that the log-informed
resampling loop of Section 4 secured as the surrogate training dataset, we now evaluate the proposed optimization
pipeline. Section 5.1 describes the FEA–AI hybrid model and GA settings used for the IPMSM case study. Section 5.2
conducts a sensitivity analysis on the hybrid-switching CV threshold to identify its best cost–reliability trade-off and
then analyzes the round-wise behavior of the proposed hybrid GA at the selected threshold. Section 5.3 compares
FEA-only GA, AI-only GA, and the proposed FEA–AI hybrid GA under a matched high-fidelity FEA-call budget.
5.1 FEA–AI Hybrid Model and GA Settings
The experimental problem is configured as a single-objective IPMSM optimization problem that minimizes iron loss
from the same initial geometry. The design-variable set, variable bounds, geometry-validation rules, and infeasible-
shape handling logic are fixed throughout the experiment so that the observed performance difference comes from
17

APREPRINT- JUNE9, 2026
Figure 16: FEA–AI hybrid model and GA settings for the IPMSM optimization experiment, covering the deep-ensemble
surrogate architecture, β-NLL training, CV-based uncertainty switching, round-wise active learning with full 30-epoch
fine-tuning, and the single-objective GA operators. All configurations are repeated over four random seeds (7, 42, 123,
2026).
the optimization execution strategy rather than from a changed design space. All experiments are executed in a local
environment based on an AMD Ryzen 9 7900 12-Core CPU.
Figure 16 summarizes the FEA–AI hybrid model setting and the GA setting used throughout the IPMSM case study.
The surrogate is a deep ensemble of M=5 MLP regressors with a 12–64–64–2 structure and a softplus variance
head, trained on the 100analysis-feasible samples obtained through the Section 4 resampling loop with the β-NLL
loss (Adam, learning rate 10−3, batch size 32, up to 300 epochs with patience-50 early stopping on the best-NLL
checkpoint) and min–max normalized inputs. Predictive uncertainty combines the aleatoric (mean-variance) and
epistemic (ensemble-variance) terms of Eq. 3, and the evaluator-switching signal is the coefficient of variationCV(x)
of Eq. 5. During the outer loop, every round triggers active learning: the FEA-labeled samples acquired by the hybrid
evaluator are merged into the training set and the surrogate is fully fine-tuned for 30 epochs before the next round,
following the per-candidate surrogate/FEA switching and round-level termination rules defined in Section 2.3.
Unlike a fixed heuristic constant, CVTh,hybrid directly balances high-fidelity FEA cost against surrogate reliability
and is therefore treated as a key design knob rather than a preset value. Accordingly, Section 5.2 reports a sensitivity
analysis over CVTh,hybrid ∈ {1%,3%,5%,10%} together with a round-wise analysis at the selected threshold, and
Section 5.3 then compares the resulting setting against FEA-only and AI-only baselines.
For the GA setting, all cases share the same single-objective GA operators: LHS-based population initialization with a
population of 25 evolved over 30 generations, tournament selection ( k=3), SBX crossover ( p=0.9 ,η=15 ), polynomial
mutation ( p=0.2 ,η=20 ), and elitist truncation replacement. The deep ensemble used in the AI and hybrid cases
consists of five MLP regressors; each MLP uses a 12–64–64–2 layer structure and is trained with early stopping to
avoid overfitting and unstable uncertainty estimates. Because the case study is single-objective, the standard GA is used
for the reported experiments, while NSGA-II remains the multi-objective option in the general framework. To account
for stochasticity in sampling, surrogate training, and GA initialization, every configuration is repeated over four random
seeds (7, 42, 123, 2026), and the reported metrics are averaged over these seeds unless a representative seed is explicitly
indicated.
18

APREPRINT- JUNE9, 2026
5.2 Switching-Threshold Sensitivity and Round-wise Analysis
Because the hybrid-switching threshold CVTh,hybrid controls how often the optimizer falls back to high-fidelity FEA,
it directly determines both the reliability of the surrogate-guided search and its computational cost. To select an
appropriate value, we vary the threshold over CVTh,hybrid ∈ {1%,3%,5%,10%} and, for each setting, run the full
five-round outer-loop optimization with the four random seeds (7, 42, 123, 2026). During each run we track three
quantities as the GA proceeds: the population-level predictive uncertainty, the best objective value, and the number of
FEA calls consumed in each round. Figures 17–19 visualize the representative seed 42, and Fig. 20 summarizes the
four-seed averages of the finally selected design.
Figure 17 shows the live predictive uncertainty of every evaluated candidate over the entire optimization, where each
panel corresponds to one threshold and the round boundaries (R1–R5) are marked. The figure confirms that the
switching threshold strongly determines whether active learning is actually activated during the optimization process.
When the threshold is high, as in the 5%and especially 10% cases (panels (c) and (d)), many candidates are accepted
by the surrogate even though their predictive uncertainty remains large. As a result, only a small number of candidates
receive FEA correction, the newly added training labels are insufficient to improve the surrogate in the explored region,
and the uncertainty band does not decrease enough from round to round. This weak uncertainty reduction indicates that
the optimization is being guided by a poorly updated AI model, which substantially lowers confidence in the reliability
of the final search trajectory.
In contrast, the 1%and3%settings (panels (a) and (b)) trigger FEA correction for a much larger portion of uncertain
candidates, so the active-learning loop is properly engaged and the uncertainty decreases more clearly than in the
loose-threshold cases. However, the two tight thresholds show different convergence characteristics. Under the
conservative 1%threshold, the uncertainty is reduced but does not converge below the switching criterion; therefore, a
similar number of candidates continues to exceed the threshold in every round, leading to nearly constant FEA calls
and repeated active-learning updates. Under the 3%threshold, the uncertainty progressively approaches and then falls
below the switching criterion, so fewer candidates require FEA correction in later rounds. This behavior indicates that
the surrogate becomes sufficiently reliable for an increasing portion of the search space, allowing the optimization to
reduce FEA calls while maintaining a trustworthy uncertainty level.
(a)CV Th,hybrid = 1%
 (b)CV Th,hybrid = 3%
(c)CV Th,hybrid = 5%
 (d)CV Th,hybrid = 10%
Figure 17: Live predictive uncertainty ( CV) per evaluation over the full GA process for representative seed 42, under the
four switching thresholds: (a) 1%, (b)3%, (c)5%, and (d) 10%. Blue dots are surrogate-evaluated candidates, orange
diamonds are FEA-corrected candidates, and the purple curve is the rolling mean; vertical lines separate optimization
rounds R1–R5. Tight thresholds (a, b) push the uncertainty toward ∼2% , whereas loose thresholds (c, d) leave it high
because FEA correction is rarely triggered.
Figure 18 reports the corresponding best-objective trajectory. The tighter thresholds reach lower iron-loss values—for
seed 42 the 3%setting converges to 1.6207 kW—because the frequent FEA corrections keep the surrogate trustworthy
in the high-performing region that the GA is exploiting. In the 1%and3%cases, FEA is still invoked in every
round, so the optimization is not driven by blindly accepting AI predictions; instead, the FEA-evaluated candidates
periodically anchor the best-so-far trajectory and correct the surrogate-guided search at least once in each round,
19

APREPRINT- JUNE9, 2026
improving confidence in the final optimum. By contrast, with the loose 5%and10% thresholds the best-objective
curves can still appear to converge almost linearly, even though FEA correction disappears from the middle rounds or is
barely used at all. This behavior is misleading: the optimizer continues to accept surrogate-predicted improvements
without high-fidelity checks, so the process reliability itself is low and the resulting optimal design cannot be trusted in
the same way as the 1%and3%cases. Consequently, the loose-threshold searches settle at noticeably worse objective
values because they commit to low-confidence surrogate optima without high-fidelity verification.
(a)CV Th,hybrid = 1%
 (b)CV Th,hybrid = 3%
(c)CV Th,hybrid = 5%
 (d)CV Th,hybrid = 10%
Figure 18: Best objective (iron loss) per evaluation over the full GA process for representative seed 42, under the
four switching thresholds: (a) 1%, (b)3%, (c)5%, and (d) 10%. The green line marks the best-so-far value and
round boundaries R1–R5 are indicated. Tight thresholds converge to lower loss (e.g., 1.6207 kW at 3%), while loose
thresholds stall at higher loss because the surrogate optimum is accepted without FEA verification.
Figure 19 explains the cost side of this behavior by plotting the per-round (non-cumulative) FEA-call count for seed 42.
AtCVTh,hybrid = 1% the round uncertainty almost never falls below the threshold, so a nearly constant 22–25FEA calls
are spent in every round and the high-fidelity cost does not decrease even as the surrogate improves. This conservative
setting can therefore achieve the best optimization performance because the search is continuously corrected by many
FEA evaluations, but it is highly inefficient as a practical operating point because it pays the largest and almost
round-invariant high-fidelity cost. At 3%the FEA-call count drops steadily round by round ( 25→14→9→2→2 ),
because active learning progressively pushes the population uncertainty below the threshold so that more candidates are
served cheaply by the surrogate. Thus, the 3%threshold preserves good optimization performance while improving
efficiency, since the FEA demand decreases together with the round-wise uncertainty. At 5%and10% the FEA calls
collapse to nearly zero already after the first round, meaning that the search effectively trusts the surrogate blindly and
performs almost no high-fidelity correction in the later rounds. These loose thresholds are therefore not suitable for
the actual framework: their small FEA counts do not represent a desirable efficiency gain, but instead indicate that the
hybrid loop has almost stopped validating candidates with FEA and is operating close to an unreliable surrogate-only
search.
Figure 20 aggregates these effects over the four seeds for the finally selected design: for each threshold, each seed’s
single best (finally selected) design is taken and its metrics are then averaged over the four seeds. As expected, the most
conservative 1%threshold attains the best objective ( 1.5676 kW) because it consumes the most FEA (about 226calls),
but at the highest total computation ( 8.41 h). The 3%threshold gives up only a small amount of objective performance
(1.6385 kW) while reducing the FEA calls to 170.5 and the total computation to 6.33 h, and—crucially—its round and
top-Kmean uncertainties ( 2.10% and2.11% ) remain almost identical to those of the 1%setting ( 2.03% and2.01% ). In
contrast, the loose 5%and10% thresholds save further computation ( 5.19 h and 4.50 h) but degrade both reliability and
performance markedly: their mean uncertainties rise to 2.99% and6.97% and their objectives worsen to 1.7396 kW and
1.9366 kW, respectively. These two settings illustrate the failure mode of an over-loosened threshold: with almost no
FEA correction and no further retraining, the GA over-trusts a low-reliability surrogate, the high-performing region is
never verified by FEA, and the search returns optima with poor objective values. Balancing reliability, objective quality,
and computational cost, the 3%threshold provides the best trade-off, and it is therefore adopted for the round-wise
iteration analysis below and for the comparative study in Section 5.3.
20

APREPRINT- JUNE9, 2026
Figure 19: Per-round (non-cumulative) FEA-call count for representative seed 42 under the four switching thresholds.
The1%setting keeps a nearly constant FEA cost every round, the 3%setting reduces FEA calls steadily as the surrogate
becomes more reliable, and the5%/10%settings stop calling FEA almost entirely after the first round.
Having selected the 3%threshold through the sensitivity analysis above, we now examine how the proposed hybrid
loop behaves round by round at this setting. Under the same GA configuration and hybrid setting, the outer loop is run
for five rounds and repeated over the four random seeds, and Table 2 reports the per-round metrics averaged over the
four seeds—that is, each round’s best objective, computation time, top- Kmean uncertainty, and FEA-call count are
averaged across the seeds, so the table traces the round-wise trajectory rather than a single finally selected design.
As the rounds proceed, the best objective decreases and then converges ( 1.8126→1.7127→1.6587 kW, stabilizing at
1.6587 kW from round 3 onward), confirming that the iterative active learning keeps steering the search toward better
designs rather than acting as a mere post-hoc uncertainty filter. This round-averaged converged value ( 1.6587 kW)
is slightly higher than the 1.6385 kW reported for the 3%setting in Fig. 20, because the latter averages each seed’s
single overall-best design whereas Table 2 averages the per-round best objective across seeds. At the same time, the
top-Kmean uncertainty falls monotonically ( 9.32%→5.58%→4.22%→2.86%→2.74% ): because each round
adds high-uncertainty FEA labels and fine-tunes the surrogate, progressively fewer candidates exceed the 3%threshold
in the later rounds. Consequently the per-round FEA-call count drops sharply ( 26.5→23.8→8.5→7.5→4.2 ), and
the per-round computation time shrinks accordingly ( 0.88→0.76→0.36→0.34→0.23 h). This is precisely the
advantage of the tuned threshold over the conservative 1%setting analyzed above: instead of paying a nearly constant
FEA cost in every round, the 3%setting makes each successive round both more reliable and cheaper, so the framework
concentrates its high-fidelity budget where it is most informative and increasingly defers to a trustworthy surrogate as
confidence improves.
Table 2: Round-wise behavior of the proposed hybrid GA at CVTh,hybrid = 3% , averaged over the four random seeds.
As the rounds proceed, the best objective decreases and converges, while the top- Kmean uncertainty, the per-round
FEA-call count, and the per-round computation time all decrease together, showing that active learning makes each
round both more reliable and less expensive.
Metric Round 1 Round 2 Round 3 Round 4 Round 5
Best objective [kW]1.8126 1.7127 1.6587 1.6587 1.6587
Round computation [h]0.88 0.76 0.36 0.34 0.23
Top-Kmean uncertainty (CV) [%]9.32 5.58 4.22 2.86 2.74
Round FEA calls26.5 23.8 8.5 7.5 4.2
21

APREPRINT- JUNE9, 2026
Figure 20: Four-seed average summary of the finally selected design under each switching threshold ( 1%,3%,5%,
10%): optimal shape and loss, total computation time, round and top- Kmean uncertainty, and total FEA calls. The 1%
setting is most accurate but most expensive, the 5%/10% settings are cheap but unreliable and low-performing, and the
3%setting matches the1%reliability at a near-best objective and substantially lower FEA cost.
5.3 Comparing Study
Finally, we compare three optimization execution strategies under a matched high-fidelity budget of 150FEA evaluations:
FEA-only GA, AI-surrogate-only GA, and the proposed FEA–AI hybrid GA. The three cases are configured as follows:
•Case 1 (FEA-only GA):all 150high-fidelity evaluations are consumed directly inside the GA loop, so the
search is strictly limited by the FEA budget.
•Case 2 (AI-only GA):a surrogate is trained once on 150FEA samples, and all subsequent GA evaluations are
performed by surrogate inference without any online FEA correction; the GA is allowed the same number of
candidate evaluations as the hybrid case.
•Case 3 (Hybrid GA, proposed):a surrogate is trained on 100FEA samples, and the remaining budget
of50FEA evaluations is spent on uncertainty-triggered online correction at the 3%switching threshold
(100 + 50 = 150FEA in total); the run stops once the50-call correction budget is exhausted.
All three strategies therefore use exactly 150high-fidelity FEA runs and comparable wall-clock time, isolating the
effect ofhowthe budget is spent rather than how large it is. The results, averaged over the four random seeds, are
summarized in Fig. 21.
Because the FEA-only GA must pay a full high-fidelity evaluation for every individual, its 150-call budget is exhausted
during the first round at about generation 8, so the search is truncated long before it can exploit the promising regions.
Even though it consumes essentially the same wall-clock time ( 7.11 h) as the hybrid run ( 7.12 h), this limited exploration
leaves it with a suboptimal design ( 1.6780 kW); its only advantage is that, being purely FEA-based, the selected design
carries essentially no surrogate uncertainty (lowest UQ). The AI-only GA lies at the opposite extreme: with the budget
spent entirely on offline training and none on online correction, it evaluates the same large candidate pool as the hybrid
but with a frozen surrogate. Lacking any FEA support in the high-performing region, it converges to the weakest design
among the three ( 1.8098 kW) and, as expected, exhibits the highest predictive uncertainty ( 8.5% ), so its optimum is
simultaneously low-performing and low-confidence.
The proposed hybrid GA combines the strengths of both baselines. By serving low-uncertainty candidates with the
surrogate while reserving the 50-call FEA budget for high-uncertainty and reliability-critical candidates, it explores
roughly 1,200 candidate designs—far more than the FEA-only case—and therefore discovers the best optimal shape
(1.6658 kW), lower than both the FEA-only and the AI-only result. Its reported top- Kuncertainty is 5.1% , higher
than the deterministic FEA-only case but well below the AI-only case; this value reflects that the 50-call FEA budget
is exhausted only at the very beginning of round 3, before the surrogate has fully converged. As the 3%-threshold
uncertainty trajectory in Fig. 17(b) shows, continuing the active-learning rounds would push this uncertainty further
down toward roughly 2%, so the elevated 5.1% is a consequence of the deliberately small comparison budget rather than
22

APREPRINT- JUNE9, 2026
a limitation of the method. Overall, under an identical high-fidelity budget the hybrid GA simultaneously delivers the
best objective and a reliable, still-improvable uncertainty level: it dominates the AI-only strategy on both performance
and reliability, and it surpasses the FEA-only strategy, which cannot explore the design space sufficiently under the
same budget.
Figure 21: Comparison of FEA-only GA, AI-surrogate-only GA, and the proposed FEA–AI hybrid GA under a matched
budget of 150high-fidelity FEA evaluations, averaged over the four random seeds. The hybrid GA achieves the best
optimal-shape performance ( 1.6658 kW) while keeping the selected-design uncertainty ( 5.1% ) well below that of the
AI-only GA.
6 Conclusion
This work presented a multi-agent IPMSM design-optimization framework that integrates textbook-grounded retrieval-
augmented generation (RAG), automated FEA data generation, uncertainty-aware surrogate learning, and FEA–AI
hybrid GA search. The main contribution is not limited to accelerating the optimizer itself, but lies in automating the
full design-optimization workflow from requirement elicitation and problem definition to final design recommendation.
In the proposed pipeline, the Design agent converts natural-language design requirements into an executable optimization
card and DOE plan. The Training agent then automates electromagnetic FEA data generation, dataset cleaning, geometry-
failure logging, ANOV A-based failure analysis, log-informed resampling, and UQ-capable deep-ensemble surrogate
training. Finally, the Optimization agent performs uncertainty-aware hybrid GA search, in which low-uncertainty
candidates are evaluated by the surrogate model, while high-uncertainty and decision-critical candidates are selectively
corrected using high-fidelity FEA and reused for online surrogate updating. This decomposition reduces manual setup
burden while preserving traceable decision checkpoints for engineering control.
The RAG study showed that connecting the Design agent to motor-domain references improves optimization-card
completeness and practical recommendation quality compared with non-retrieval responses. The hybrid-switching
sensitivity analysis over CVTh,hybrid ∈ {1%,3%,5%,10%} demonstrated a clear cost–reliability trade-off. The
conservative 1%setting maintained low uncertainty but required the highest FEA cost, whereas the loose 5%and10%
settings reduced FEA calls but caused the optimization process to behave closer to surrogate-only search, leading to
low-confidence and lower-quality optima. For the present case study, the 3%threshold provided the best practical
trade-off, achieving near- 1%reliability and objective quality while reducing the number of FEA calls over iterative
optimization rounds.
Using this selected threshold, the GA iteration study confirmed that online FEA correction and surrogate retraining
progressively reduce predictive uncertainty and per-round FEA cost while guiding the search toward improved objective
values. Under a matched 150-call high-fidelity FEA budget, the proposed hybrid GA achieved the best objective
23

APREPRINT- JUNE9, 2026
performance among the three compared strategies while maintaining low and further reducible uncertainty. In contrast,
the FEA-only GA was constrained by early budget exhaustion and insufficient exploration, while the AI-only GA,
without online FEA correction, converged to the weakest and least reliable optimum.
It should be noted that the present comparison intentionally used a small and unified high-fidelity evaluation budget
of150FEA calls. Therefore, the measured computation-time difference among FEA-only, AI-only, and hybrid
strategies is limited in this study. However, practical GA-based motor optimization often requires thousands or more
candidate evaluations. In such larger-scale settings, the proposed hybrid strategy is expected to provide a larger
computational advantage over FEA-only GA while maintaining a more reliable uncertainty range than AI-only GA
through selective FEA correction in high-UQ regions. In addition, the use of a local GPT-OSS 20B backbone supports
confidentiality-sensitive on-premises deployment, which is important for industrial motor-design environments.
Key findings
•A RAG-to-optimization-card workflow was introduced to convert natural-language IPMSM design require-
ments into executable DOE and optimization artifacts, reducing expert-dependent setup ambiguity.
•Automated FEA data generation, failure-log/ANOV A-based fail-geometry analysis, LLM-reasoning-guided
design-space refinement, and deep-ensemble β-NLL uncertainty learning were connected in a single Training-
agent loop. This enabled traceable surrogate construction in which infeasible geometries were not simply
discarded, but analyzed through validation logs and statistical failure patterns to guide autonomous resampling.
•An uncertainty-aware FEA–AI hybrid GA was proposed, in which the switching threshold was selected
through sensitivity analysis. In the present case study, the 3%CV threshold selectively corrected high-risk
candidates with FEA and reused them for online retraining, achieving the best objective performance under a
matched FEA budget while avoiding both the high-uncertainty optima of AI-only search and the early budget
exhaustion of FEA-only GA.
Limitations and Future Work
The current study is limited to a parameterized IPMSM geometry and a single-objective optimization case focused on
iron-loss minimization. Future work will extend the framework to more general geometry representations, including non-
parametric or topology-based design spaces, and to multi-objective Pareto optimization problems such as simultaneous
torque maximization, loss minimization, torque-ripple reduction, and efficiency improvement.
The present validation is also mainly based on electromagnetic FEA. For practical IPMSM design, additional physical
constraints such as thermal behavior, rotor mechanical stress, demagnetization margin, vibration, and manufacturability
should be considered. Future work will therefore extend the proposed agentic workflow toward multi-physics FEA
automation and multi-constraint design validation.
In this study, the hybrid-switching uncertainty threshold CVTh,hybrid was selected through sensitivity analysis and
then fixed at 3%for the subsequent experiments. Because the best trade-off may change depending on the objective
function, dataset size, design-space complexity, surrogate accuracy, and available FEA budget, future work will develop
an adaptive switching strategy. This strategy will use predictive UQ signals, GA progress, FEA-budget status, and
IPMSM design context to adjust the threshold online and decide when high-fidelity FEA correction is required.
Finally, although the current RAG module uses a motor-domain textbook/reference source to support problem definition,
its recommendation quality depends on the coverage and reliability of the retrieval corpus. Future work will expand
the knowledge base with additional motor-design references, industrial design rules, manufacturing guidelines, and
multi-physics constraint information, enabling more robust and practically deployable design-agent behavior.
Overall, the proposed framework demonstrates that agent-based automation, retrieval-grounded problem definition,
and uncertainty-aware FEA–AI hybrid optimization can be combined into a reproducible workflow for reliable and
computationally efficient IPMSM design optimization.
Acknowledgments
This work was supported by grants from the Ministry of Science and ICT (GTL24031-000, N10250154, No. 2022-0-
00986, RS-2024-00355857) and the Ministry of Trade, Industry & Energy (RS-2025-02317327, RS-2025-25444634).
24

APREPRINT- JUNE9, 2026
References
[1]Xiangdong Liu, Hao Chen, Jing Zhao, and Anouar Belahcen. Research on the performances and parameters of
interior pmsm used for electric vehicles.IEEE Transactions on Industrial Electronics, 63(6):3533–3545, 2016.
[2]Gianmario Pellegrino, Alfredo Vagati, Barbara Boazzo, and Paolo Guglielmi. Comparison of induction and
pm synchronous motor drives for ev application including design examples.IEEE Transactions on Industry
Applications, 48(6):2322–2332, 2012.
[3]Tae-Hyuk Ji, Chan-Ho Kim, Seok-Won Jung, and Sang-Yong Jung. Design method of ipmsm using multi-objective
optimization considering mechanical stress for high-speed electric vehicles.Journal of Electrical Engineering &
Technology, 19(4):2481–2489, 2024.
[4]Buddhika De Silva Guruwatta Vidanalage, Mohammad Sedigh Toulabi, and Shaahin Filizadeh. Multimodal
design optimization of v-shaped magnet ipm synchronous machines.IEEE Transactions on Energy Conversion,
33(3):1547–1556, 2018.
[5]Feng Liu, Xiuhe Wang, and Zezhi Xing. Design of a 35 kw permanent magnet synchronous motor for electric
vehicle equipped with non-uniform air gap rotor.IEEE Transactions on Industry Applications, 59(1):1184–1198,
2022.
[6] Tomoki Nakata, Masayuki Sanada, Shigeo Morimoto, and Yukinori Inoue. Automatic design of ipmsms using a
genetic algorithm combined with the coarse-mesh fem for enlarging the high-efficiency operation area.IEEE
Transactions on Industrial Electronics, 64(12):9721–9728, 2017.
[7]Lei Wang, Chen Ma, Xie Feng, Jiahui Zhang, Zeyu Duan, Bowen Zhao, Yuting Wang, Zhendong Zhang, Bo Yang,
Qinglong Zheng, et al. A survey on large language model based autonomous agents.Frontiers of Computer
Science, 18(6):186345, 2024.
[8]Alejandro Pradas Gómez, Massimo Panarotto, and Ola Isaksson. Evaluation of different large language model
agent frameworks for design engineering tasks. InProceedings of NordDesign 2024: Design in the Era of
Digitalization, AI and Augmented Intelligence, pages 693–702, 2024.
[9]Namwoo Kang. Generative ai-driven design optimization: Eight key application scenarios.JMST Advances,
7(1):105–111, 2025.
[10] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan. A fast and elitist multiobjective genetic
algorithm: Nsga-ii.IEEE Transactions on Evolutionary Computation, 6(2):182–197, 2002.
[11] Hyeon-Jun Kim and Soo-Whang Baek. Optimal shape design to improve torque characteristics of interior
permanent magnet synchronous motor for small electric vehicles.Microsystem Technologies, 31(5):1203–1217,
2025.
[12] Seungyeon Shin, Dongju Shin, and Namwoo Kang. Topology optimization via machine learning and deep
learning: a review.Journal of Computational Design and Engineering, 10(4):1736–1766, 2023.
[13] Tatsuya Asanuma and Yoshihiro Kanno. Multi-objective surrogate optimization of mixed-variable ipm motor
design problem with admm-based approach.Optimization and Engineering, pages 1–26, 2025.
[14] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty
estimation using deep ensembles. InAdvances in Neural Information Processing Systems, 2017.
[15] Yuancong Gong, Andreas Gneiting, Chongshen Zhao, Nejila Parspour, and Hao Chen. Comparative study of
different data-driven surrogate models for optimization of synchronous reluctance machine.IEEE Transactions
on Industry Applications, 2025.
[16] Yidan Ma, Zaixin Song, Yongtao Liang, and Jianfu Cao. Topology optimization for a spoke-type permanent
magnet synchronous motor based on a siamese convolutional network. In2024 27th International Conference on
Electrical Machines and Systems (ICEMS), pages 3346–3351. IEEE, 2024.
[17] M. D. McKay, R. J. Beckman, and W. J. Conover. A comparison of three methods for selecting values of input
variables in the analysis of output from a computer code.Technometrics, 21(2):239–245, 1979.
[18] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and
Pascale Fung. Survey of hallucination in natural language generation.ACM Computing Surveys, 55(12):248:1–
248:38, 2023.
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K"uttler, Mike Lewis, Wen-tau Yih, Tim Rockt"aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented
generation for knowledge-intensive nlp tasks. InAdvances in Neural Information Processing Systems, 2020.
25

APREPRINT- JUNE9, 2026
[20] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces
hallucination in conversation. InFindings of the Association for Computational Linguistics: EMNLP 2021, pages
3784–3803, 2021.
[21] Jacek F Gieras.Permanent magnet motor technology: design and applications. CRC press, 2009.
[22] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond.Foundations
and Trends in Information Retrieval, 3(4):333–389, 2009.
[23] Gordon V . Cormack, Charles L. A. Clarke, and Stefan Büttcher. Reciprocal rank fusion outperforms Condorcet
and individual rank learning methods. InProceedings of the 32nd International ACM SIGIR Conference on
Research and Development in Information Retrieval, pages 758–759. ACM, 2009.
[24] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? In
Advances in Neural Information Processing Systems, volume 30, 2017.
[25] Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, and Georg Martius. On the pitfalls of heteroscedastic
uncertainty estimation with probabilistic neural networks.arXiv preprint arXiv:2203.09168, 2022.
[26] Ian Farrance and Richard Frenkel. Uncertainty of measurement: a review of the rules for calculating uncertainty
components through functional relationships.The Clinical Biochemist Reviews, 33(2):49, 2012.
26