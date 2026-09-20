# SpecLens: LLM-Based Verilog Generation with Specification-Derived Constraints via Behavioral Divergence

**Authors**: Wen Bing, Bing Li

**Published**: 2026-09-15 06:59:00

**PDF URL**: [https://arxiv.org/pdf/2609.16729v1](https://arxiv.org/pdf/2609.16729v1)

## Abstract
Large language models (LLMs) have recently shown promise in Verilog generation, but producing functionally correct RTL directly from natural-language specifications remains a highly challenging task. Existing approaches improve LLM-based Verilog generation mainly with retrieval-augmented generation (RAG), self-planning, or few-shot prompting. However, these methods focus primarily on external or generic forms of enhancement rather than strengthening the specification with task-specific constraints. In this work, we propose SpecLens, an automated framework for LLM-based Verilog generation that derives specification-driven constraints by analyzing behavioral divergence among multiple candidate implementations, using the original specification as the only external semantic source during generation. On the VerilogEval v2.0 spec-to-RTL benchmark, SpecLens achieves a functional pass@1 ratio of 86.2\% with o3-mini-medium and 89.4\% with o3-mini-high. This corresponds to a 3.6 percentage-point gain over the SOTA prompting method with o3-mini-medium and a 3.8 percentage-point gain over the SOTA behavioral divergence method with o3-mini-high. In addition, on RTLLM v1.1 and v2.0, analysis shows that SpecLens is more specification-faithful and less prone to benchmark-aligned priors. SpecLens achieves 100\% syntactic correctness on VerilogEval v2.0, 86.2\% on RTLLM v1.1, and 88\% on RTLLM v2.0, even without using costly compile-repair loops to revise generated code iteratively. The code is open source and available at https://anonymous.4open.science/r/SpecLens-4632/readme.md.

## Full Text


<!-- PDF content starts -->

SpecLens: LLM-Based Verilog Generation with
Specification-Derived Constraints via Behavioral Divergence
Wen Bing
TU Ilmenau
Ilmenau, Germany
wen.bing@tu-ilmenau.deBing Li
TU Ilmenau
Ilmenau, Germany
bing.li@tu-ilmenau.de
Abstract
Large language models (LLMs) have recently shown promise in
Verilog generation, but producing functionally correct RTL directly
from natural-language specifications remains a highly challenging
task. Existing approaches improve LLM-based Verilog generation
mainly with retrieval-augmented generation (RAG), self-planning,
or few-shot prompting. However, these methods focus primarily
on external or generic forms of enhancement rather than strength-
ening the specification with task-specific constraints. In this work,
we propose SpecLens, an automated framework for LLM-based
Verilog generation that derives specification-driven constraints by
analyzing behavioral divergence among multiple candidate imple-
mentations, using the original specification as the only external
semantic source during generation. On the VerilogEval v2.0 spec-
to-RTL benchmark, SpecLens achieves a functional pass@1 ratio
of 86.2% with o3-mini-medium and 89.4% with o3-mini-high. This
corresponds to a 3.6 percentage-point gain over the SOTA prompt-
ing method with o3-mini-medium and a 3.8 percentage-point gain
over the SOTA behavioral divergence method with o3-mini-high.
In addition, on RTLLM v1.1 and v2.0, analysis shows that SpecLens
is more specification-faithful and less prone to benchmark-aligned
priors. SpecLens achieves 100% syntactic correctness on VerilogEval
v2.0, 86.2% on RTLLM v1.1, and 88% on RTLLM v2.0, even without
using costly compile-repair loops to revise generated code itera-
tively. The code is open source and available at https://anonymous.
4open.science/r/SpecLens-4632/readme.md.
CCS Concepts
•Hardware→Hardware description languages and compila-
tion.
Keywords
LLM Verilog generation, specification enhancement, behavioral
divergence, entropy filtering, self-consistency
1 Introduction
Large language models (LLMs) have stimulated growing interest in
their application to electronic design automation (EDA), particu-
larly for generating register-transfer-level (RTL) designs in Verilog
from natural-language specifications. Early work explores Verilog
generation based on GPT [ 1], while subsequent studies investigate
reinforcement learning with hardware-aware rewards [ 2], HLS-C
repair [ 3], and automated testbench generation [ 4,5]. Despite this
progress, functional correctness remains limited due to the scarcity
of high-quality Verilog training data and model hallucinations [ 7].
A straightforward solution to improve correctness is to rely on
testbench feedback or fine-tuning. Multi-agent approaches [ 8,9]have reported strong results, but testbenches are often unavailable
in practice when RTL generation starts from natural-language spec-
ifications. Fine-tuning methods [ 10–12] have also shown strong
performance on specific datasets, but there is insufficient high qual-
ity Verilog data to fine-tune a general Verilog model for broad use.
This motivates an alternative direction: improving Verilog genera-
tion by enhancing the specification.
Existing specification-enhancement approaches also have impor-
tant limitations. Interactive workflows [ 13,14] can obtain useful
information, but require substantial human effort and hardware
expertise. RAG methods [ 15–17] attempt to strengthen specifica-
tion understanding or repair Verilog syntax using external libraries.
However, their effectiveness depends heavily on retrieval quality
and may degrade when the retrieved examples are not sufficiently
important for the current task or are misleading. In-context learn-
ing methods such as few-shot prompting provide examples that
help the model become more familiar with tasks in this domain [ 18].
Self-planning asks the LLM to derive an implementation plan from
the specification, thereby introducing additional intermediate infor-
mation [ 19]. However, these methods do not dynamically provide
the task-specific information that the model actually needs for
each problem, especially for parts of the specification that are unfa-
miliar or difficult for the LLM. Therefore, the key challenge is to
determine when and where a specification is ambiguous from the
model’s perspective.
A possible approach to identify such ambiguity is through self-
consistency-based methods, in which behavioral divergence signals
potential specification ambiguity. The intuition is that if a specifi-
cation is sufficiently clear to the LLM, independently generated im-
plementations should have the same behavior even across multiple
sampling runs. In the software domain, prior work [ 14] shows that
output inconsistency among multiple LLM-generated programs can
expose specification ambiguity, but resolving such ambiguity still
requires further human interaction. SpecFix [ 20] uses golden exam-
ples provided with the specification to identify where behavioral
disagreement arises. However, hardware benchmarks typically do
not provide such golden examples. In the RTL domain, methods
based on self-consistency such as VRank [ 21] and VFocus [ 22] have
also been explored, but they mainly use behavioral clustering as a
back-end mechanism for candidate ranking or selection. As a result,
they do not provide targeted semantic supplementation for those
parts of the specification that the model fails to understand well.
We address this gap by using behavioral divergence not merely
as a ranking signal but as a source of specification-derived dis-
ambiguation constraints. If a specification is sufficiently explicit,
independently generated Verilog candidates should converge to
the same observable behavior under the same stimuli. Based on
arXiv:2609.16729v1  [cs.AR]  15 Sep 2026

ASPDAC ’27, January 25–28, 2027, Tokyo, Japan Wen Bing and Bing Li
Speciﬁcation
LLMLLM
ConstraintClustering Based on
Verilog TracesVerilog Candidates
Cluster 2 Cluster 1 Cluster MSimulatorStructured Input Stimulus
Verilog Traces from
Different ClustersLLMCore of
SpecLens
Entropy-based
Validation Validated ConstraintExisting Behavioral-divergence Backbone
Figure 1: Core of SpecLens.
this idea, we propose SpecLens, an automated constraint derivation
framework for LLM-based Verilog generation. SpecLens extracts
explicit behavioral requirements from the given specification, con-
structs structured scenarios and stimuli, analyzes Verilog output
behavior, and derives disambiguation constraints. To improve re-
liability, it further applies entropy-guided filtering, retaining only
those constraints that reduce behavioral uncertainty. Our key con-
tributions are as follows:
•We propose SpecLens, an automated framework for LLM-based
Verilog generation that derives disambiguation constraints di-
rectly from the original specification by using behavioral diver-
gence among candidate implementations to identify ambiguities
from the model’s perspective, without relying on external re-
trieval or golden testbenches during generation.
•On VerilogEval v2.0 spec-to-RTL, SpecLens achieves 100% syn-
tactic pass@1 with o3-mini-medium without using any syntax
repair loop during generation, and also reaches the best func-
tional pass@1 among all compared baselines, achieving 86.2%
with o3-mini-medium and 89.4% with o3-mini-high.
•The derived constraints are reusable across models and can serve
as additional guidance for smaller models. With these constraints,
GPT-4o-mini improves functional pass@1 by 6.84% compared to
using the original specification alone.
•Experiments on RTLLM show that SpecLens is more sensitive to
explicit local changes to the specification and therefore produces
more specification-faithful implementations, while other direct
generation methods tend to reproduce previously learned canon-
ical implementations, which can introduce critical discrepancies
in the results.
2 Concept Overview
In this section, we briefly introduce the fundamental changes made
by SpecLens over existing behavioral-divergence-based workflows
and clarify several concepts used throughout the rest of the paper.
As illustrated in Figure 1, existing behavioral-divergence backbones
mainly apply input stimuli to multiple LLM-generated Verilog candi-
dates and then select the final Verilog from the resulting behavioral
clusters. However, they do not systematically construct or exploit
Surviving  Requirement–
Scenario–Stimulus  FamiliesBehavioral
RequirementsSpeciﬁcation
Iteration
Verilog Candidates 
 and the Last Stimulus 
No Surviving Family
Final V erilogStage II: Entropy-Guided Constraint Derivation  and IterationStage I: Structured Scenario–Stimulus Construction & Preﬁltering
Stage III: Final V erilog Selection and InitializationAccumulated
Constraints of the
Given SpeciﬁcationRequirement
Extractor
Fallback V erilog CandidateRequirement
AggregatorFigure 2: Overall workflow of SpecLens.
the information provided by behavioral clustering.
The core of SpecLens lies in three aspects: how LLMs gener-
ate structured input stimuli, how LLMs derive constraints from
behavioral clustering, and how the framework validates whether
the derived constraints effectively drive subsequent Verilog genera-
tion toward convergence. In all figures in this paper, solid arrows
indicate the main execution flow, whereas dashed arrows denote
contextual data dependencies. In this work, a Verilog trace refers to
the sampled Verilog output sequence produced by a Verilog imple-
mentation in the simulator under a given input stimulus. Behavioral
divergence refers to the case where different Verilog candidates pro-
duce different Verilog traces under the same stimulus. Candidates
with identical Verilog traces are grouped into the same cluster. The
process of applying a stimulus to Verilog candidates and partition-
ing them into different clusters is referred to as clustering.
3 Methodology
3.1 Framework Overview
In this section, we present the overall workflow of SpecLens, as
shown in Figure 2. Starting from the original specification, the
framework first extracts behavioral requirements describing ex-
pected design behavior rather than interface definitions. A Python-
based Requirement Aggregator concatenates all extracted behav-
ioral requirements into an additional composite requirement, which
is appended to the requirement list and processed after the indi-
vidual requirements. The stimulus generated from this composite
requirement serves as a global behavioral evaluation stimulus for
final candidate selection in Stage III, since its generation context
jointly incorporates all extracted behavioral requirements. Stage I
then converts each requirement into a requirement-targeted sce-
nario and stimulus and performs prefiltering to determine whether
further constraint derivation is needed. If the stimulus for every
behavioral requirement produces only a single cluster, no behav-
ioral divergence is detected under the current candidate pool and
generated stimuli. In this case, no surviving requirement family is
produced, and the framework follows the fallback branch. One Ver-
ilog candidate from Stage I is selected as the fallback candidate and
passed directly to Stage III. Otherwise, each requirement–scenario–

SpecLens: LLM-Based Verilog Generation with Specification-Derived Constraints via Behavioral Divergence ASPDAC ’27, January 25–28, 2027, Tokyo, Japan
ﬁxed initial V erilog
candidate pool
CECReﬁned Stimuli
Preﬁlter OutputSpeciﬁcation
Behavioral 
RequirementsInterface Info
Template
Input stimuliFixed Initial Candidate
Generator
Scenario
GeneratorUp to Q
Iterations
Entropy InfoScenariosScenarios Input stimuli
Process: Stimulus Generation and Entropy-Guided ReﬁnementStimulus
GeneratorDriver
GeneratorEntropy-Guided
Stimulus Reﬁnement 
Figure 3: Stage I: structured scenario–stimulus construction and prefiltering.
stimulus combination that produces multiple behavioral clusters
is retained as a surviving requirement–scenario–stimulus (RSS)
family and forwarded to Stage II for constraint derivation. After
Stage II, the Verilog candidates generated under the accumulated
constraints, together with the last stimulus generated from the com-
posite requirement, are passed to Stage III. Stage III then performs
final candidate selection and output initialization to produce the
final Verilog implementation.
3.2 Stage I: Structured Scenario–Stimulus
Construction and Prefiltering
This section introduces how Stage I constructs structured stimuli
and performs prefiltering, as illustrated in Figure 3. Here, structured
means that the generated scenarios and stimuli are written in an
explicit, machine-usable form rather than as free-form text.
Starting from the original specification, LLMs first generate a
fixed initial pool of 𝑁Verilog candidates, which is used for prefilter-
ing and for extracting shared interface information. This interface
information is necessary because the LLM must know the input
and output port names in order to generate a correct simulation
driver template.
In parallel, SpecLens converts each behavioral requirement into
a requirement-targeted scenario through the LLM-based Scenario
Generator. A scenario supplements the requirement with the miss-
ing testing context implied by the specification. In many cases, the
behavior described by a requirement begins in an intermediate
state, while the stimulus must also specify how that state is reached.
The scenario therefore provides a more complete description of the
behavior to be tested. Each scenario is then processed by the LLM-
based Stimulus Generator, which produces a structured stimulus
template using the interface information. A Python-based Driver
Generator then automatically converts this template into an exe-
cutable Verilog driver file ( driver.v ) that applies inputs, handles
clock and reset behavior, samples outputs, and logs traces.
For the current requirement family, the generated driver.v is
executed against the fixed initial Verilog candidate pool, and the
resulting Verilog traces are grouped into the same cluster if they
are identical. Candidates that fail to produce valid traces, typically
because of syntax or compilation errors, are automatically filtered
out. The clustering result is then used to guide LLM-based stimulus
refinement. For the same set of Verilog candidates, a stimulus that
separates behaviors into more clusters provides more information
about potential specification ambiguity. To quantify this effect,
SpecLens computes behavioral entropy over the clusters inducedby the stimulus. Let
G(𝑥)={𝐺 1,𝐺2,...,𝐺𝐾}(1)
denote the𝐾clusters induced by stimulus 𝑥, ordered in descending
order by cluster size. Let 𝑛𝑘denote the number of candidates in
cluster𝐺𝑘, and let
𝑝𝑘=𝑛𝑘
𝑀(𝑥)(2)
where𝑀(𝑥)=Í𝐾
𝑘=1𝑛𝑘denotes the number of Verilog candidates
that produce valid traces under stimulus 𝑥. The behavioral entropy
is then defined as
𝐻(G)=−𝐾∑︁
𝑘=1𝑝𝑘log2𝑝𝑘 (3)
Higher entropy indicates greater behavioral dispersion across
clusters. Given the specification, a scenario, and its current stim-
ulus, the LLM proposes a refined stimulus, and both the original
and refined stimuli are evaluated on the same fixed initial Verilog
candidates through the Python-based Clustering and Entropy Com-
parison (CEC). If entropy increases, refinement stops early, since the
refined stimulus has already exposed greater behavioral dispersion.
Otherwise, refinement continues for up to𝑄rounds.
Stage I has two possible prefiltering outcomes. If all requirements
produce only a single behavioral cluster, one Verilog candidate from
the fixed initial pool is randomly selected and passed directly to
Stage III. Otherwise, any RSS combination that produces multiple
clusters is kept as a surviving RSS family. These surviving families
are then forwarded to Stage II for constraint derivation.
3.3 Stage II: Entropy-Guided Constraint
Derivation and Iteration
Stage II describes how constraints are iteratively derived from each
RSS family and how entropy is used to validate them, as detailed in
Figure 4. Each surviving RSS family from Stage I enters an iterative
constraint-derivation loop. This loop contains three steps.
Step 1: Constraint Derivation: This step derives disambigua-
tion constraints from the input RSS family. At the beginning, the
LLM-based Baseline Verilog Generator generates a batch of 𝑁Ver-
ilog candidates from the original specification, the behavioral re-
quirement in the current RSS family, and the accumulated con-
straints for the same specification, if any. The CEC module then
applies either the current stimulus from the RSS family or a newly
refined stimulus fromStep 3to these candidates in order to cluster
their resulting traces and compare different behaviors. If multiple
clusters remain, the corresponding Verilog traces are forwarded for
subsequent constraint derivation. If the candidates converge to a

ASPDAC ’27, January 25–28, 2027, Tokyo, Japan Wen Bing and Bing Li
Constraint
Verilog Candidates
 Lower Entropy
Same Requirement
CEC: Clustering and Entropy ComparisonSpeciﬁcation
AC
Verilog Candidates
and the Last Stimulus RequirementsVerilog
Traces
Select One
Surviving FamilySurviving RSS
Families
ConstraintInteraction
Process: Stimulus Generation and
Entropy-Guided ReﬁnementStep 1: Constraint Derivation
CECIterate until Entropy = 0
If all requirements
are checked CEC
Verilog Candidates
Speciﬁcation
Requirements Or until P  Iterations 
Verilog Candidates
Interface InfoStep 2: Constraint 
Convergence V alidation
Step 3: Independent Scenario–Stimulus Regeneration and Re-evaluation
AC: Accumulated Constraints RSS: Requirement–Scenario–StimulusVerilog
Traces
Extracted
InfoNew Constraint-Applied
Verilog Generator
Baseline  Verilog
GeneratorWinner Selector Constraint Deriver
Figure 4: Stage II: entropy-guided constraint derivation and iteration.
single cluster and the entropy becomes zero, the current RSS family
is considered behaviorally converged under the current stimulus
and the loop proceeds to the next RSS family.
If the entropy is nonzero, the top 𝑀clusters are selected, and
their Verilog traces are used by the LLMs for winner selection and
constraint derivation. Both steps are performed without access to
the underlying Verilog code, so the framework focuses on differ-
ences in Verilog traces when judging whether a behavior satisfies
the specification. This design avoids constraints tied to internal reg-
isters or implementation details. Instead, the resulting constraints
supplement the specification with externally observable behavior.
The LLM-based Winner Selector uses only the original specifi-
cation, the current requirement family, and the 𝑀Verilog traces
to determine which trace is most consistent with the specification.
The selected trace is then used as an anchor for subsequent con-
straint derivation, allowing the LLM-based Constraint Deriver to
focus on the key behavioral difference among the𝑀traces. Using
the specification, the current requirement family, the anchor trace,
and the𝑀Verilog traces, it then derives a constraint to help later
generated Verilog candidates avoid similar incorrect behavior.
For example, in the VerilogEval task ece241_2014_q5a , one com-
mon failure mode is a one-cycle-delayed response in which the out-
put does not immediately reflect the first high serial input bit after
reset. This error is identified by comparing Verilog traces under
the same stimulus rather than inspecting internal code structure.
Based on the observed mismatch, SpecLens derives an externally
observable constraint.
Example 3.1 (Derived constraint).For ece241_2014_q5a , when
areset is deasserted and the module is processing serial input, on
the positive clock edge where the first occurrence of x=1is applied
after an initial sequence of 0s, the observed output zmust be 1in
the same post-edge observation.
Step 2: Constraint Convergence Validation: A derived con-
straint fromStep 1is accepted only if it reduces behavioral entropy.
To validate the constraint, the LLM-based New Constraint-Applied
Verilog Generator takes it as additional semantic information and
generates a new batch of Verilog candidates under the same specifi-
cation and requirement context. These newly generated candidates
are then evaluated by CEC under the same stimulus as before. If
the entropy decreases, the constraint is accepted and added to the
Final V erilog with Initial
Output V alueCluster-Based
Candidate SelectionVerilog Candidates from Stage II
and Last Stimulus
Verilog Candidates
from Stage ITwo V erilog
Candidates 
Final V erilog Verilog Selector Output InitializationSingle-Cluster CandidateFigure 5: Stage III: final Verilog selection and initialization.
accumulated constraint set. Otherwise, it is rejected.
Step 3: Independent Scenario–Stimulus Regeneration and
Re-evaluation: AfterStep 2, the current requirement is sent back
to the stimulus-generation and entropy-guided refinement process
shown in Figure 3, which independently regenerates a new scenario
and refined stimulus. This regeneration improves the diversity
and reliability of subsequent verification. IfStep 2accepts a new
constraint, the LLM-based Baseline Verilog Generator inStep 3
produces a new batch of Verilog candidates under the updated
accumulated constraint set, and these candidates are evaluated
by CEC inStep 1using the newly generated stimulus. If no new
constraint is accepted, the Verilog candidates remain unchanged,
and only the new stimulus is used inStep 1to produce new Verilog
traces.
The iterative loop for each RSS family continues until its entropy
reaches zero or the maximum number of iterations 𝑃is reached.
The framework then proceeds to the next surviving RSS family
and repeats the same procedure. After all surviving RSS families
have been processed, Stage II outputs the current Verilog candi-
dates together with the last stimulus generated from the composite
requirement processed at the end of the requirement list.
3.4 Stage III: Final Verilog Selection and
Initialization
This section presents how the final Verilog is produced in Stage III,
which has two input paths, as shown in Figure 5. If no requirement
family survives Stage I, one Verilog candidate from the fixed initial
pool is passed directly to the initialization step, where the LLM
assigns output initial values only according to the original specifica-

SpecLens: LLM-Based Verilog Generation with Specification-Derived Constraints via Behavioral Divergence ASPDAC ’27, January 25–28, 2027, Tokyo, Japan
Table 1: Comparison on VerilogEval v2.0 spec-to-RTL.
Method Model / SettingSyntactic Correctness Functional Correctness
pass@1 pass@5 pass@1 pass@5
Self-planning [19] o3-mini-medium 94.9 99.7 78.0 88.2
Two-shot prompting [22] o3-mini-medium 98.5 99.9 82.6 90.0
SpecLens (ours) o3-mini-medium 100.0 100.0 86.2 92.6
DeepV [15] GPT-5 Chat + RAG 99.4 100.0 76.9 81.3
VeriSeek [2] Reinforcement Learning 85.1 98.3 61.6 76.9
SiliconMind-V1 [12] Fine-tuning – – 76.5 84.7
VeriGRAG [17] Fine-tuning + RAG 99.4 100.0 62.2 69.5
tion. Otherwise, Stage III receives the Verilog candidates generated
in Stage II together with the last stimulus. A Python-based cluster-
selection module clusters and ranks the candidates according to
their traces under this stimulus. If only one cluster remains, a can-
didate proceeds directly to initialization. If multiple clusters remain,
one candidate is selected from each of the top two clusters, and the
two candidates are sent to the LLM-based Verilog Selector, which
selects the candidate that is more consistent with the specification.
The selected Verilog is then initialized by an LLM according to
the specification. This post-processing step is restricted to adding
specification-supported output-initialization assignments and is
intended to handle benchmark tasks with explicit default-output
requirements.
4 Experimental Results
4.1 Setup
Experiments were conducted on the original benchmark releases,
including VerilogEval v2.0 spec-to-RTL [ 23], RTLLM v1.1 [ 19], and
RTLLM v2.0 [ 24]. For syntax checking and functional verifica-
tion, we used Icarus Verilog [ 25] and Verilator [ 26]. Functional
correctness was measured by running the final Verilog generated
in Stage III against the benchmark testbench.
The maximum number of stimulus refinement rounds was set to
𝑄= 3in both Stage I and Stage II. We set the maximum number of
iterations per requirement in Stage II to 𝑃=3. The number of fixed
initial Verilog candidates generated in Stage I was set to 𝑁= 8,
and the number of Verilog candidates generated in each Stage II
round for entropy evaluation was also set to 𝑁= 8. The primary
evaluation metric is pass@k. Let 𝑛denote the number of sampled
candidates for a problem and 𝑐denote the number of functionally
correct candidates among them. The pass@k score is computed as
pass@𝑘=E problems"
1− 𝑛−𝑐
𝑘
 𝑛
𝑘#
(4)
This metric estimates the probability that at least one correct solu-
tion appears among𝑘sampled candidates. We set𝑛=20.
In Step 1 of Stage II, we set CEC to select Verilog traces only from
the top𝑀= 2clusters. In a post hoc analysis of SpecLens failure
cases, we examine the risk of restricting comparison to the top two
clusters. Under our main setting with o3-mini-medium, the proba-
bility that a round produces more than two clusters and that neither
of the top two clusters contains a functionally correct Verilog im-
plementation is only 2.73%. For o3-mini-high, the corresponding
probability averages 15.8%. This suggests that stronger models more
often produce correct implementations as low-frequency outliers.Developing more robust cluster-selection mechanisms for such
settings is left for future work.
Table 1 compares SpecLens with representative prompting, rein-
forcement learning, fine-tuning, and retrieval-based methods on
VerilogEval v2.0 spec-to-RTL. The two-shot in-context learning
baseline uses two examples from [ 22]: a combinational XOR gate
and a sequential 8-bit registered incrementer. The self-planning
prompt is taken from [ 19]. We also ran these two baselines with
o3-mini-medium. The comparison also includes retrieval-based
methods, such as DeepV [ 15]. For comparison with fine-tuning-
based methods, we include SiliconMind-V1 [ 12] and VeriGRAG [ 17].
SiliconMind-V1 is a recent fine-tuned Verilog model, while Veri-
GRAG combines fine-tuning with graph-embedding-based retrieval.
It is worth noting that DeepV and VeriSeek report results on the
original VerilogEval benchmark, whereas our main evaluation uses
VerilogEval v2.0. We therefore include these results only as cross-
version references, rather than as directly comparable baselines.
4.2 Results and Comparison
As shown in Table 1, SpecLens achieves the best overall perfor-
mance, reaching 100.0% for both syntactic pass@1 and pass@5, as
well as 86.2% and 92.6% for functional pass@1 and pass@5, respec-
tively. It outperforms both self-planning and two-shot prompting.
A similar advantage is observed over retrieval-augmented meth-
ods such as DeepV. These results suggest that directly identify-
ing specification-specific uncertainty is more effective than simply
adding externally provided information.
Furthermore, we compare against VRank [ 21] and VFocus [ 22].
Because the original VRank paper does not report results for o3-
mini, we use the o3-mini-high VRank result reported in the follow-
up VFocus study [ 22]. For a fair comparison, SpecLens, VRank,
and VFocus are all compared under the same o3-mini-high setting.
Under this setting, VRank achieves a functional pass@1 of 77.4%,
and VFocus improves it to 85.6%. SpecLens further reaches 89.4%.
These results suggest that behavioral divergence is useful not only
as a back-end mechanism for candidate selection, but also as a way
to identify where the LLM is uncertain about the specification and
to provide targeted constraints for those ambiguous parts.
Figure 6 reports the main results on RTLLM v1.1 and RTLLM
v2.0. The line plots show syntactic pass@1, and the bar plots show
functional pass@1. Three baselines are included: direct prompting,
two-shot prompting [ 22], and self-planning [ 19], all under the same
base model setting, o3-mini-medium.
Under the originally released testbenches, SpecLens achieves
higher syntactic pass@1 than the baselines, but functional pass@1

ASPDAC ’27, January 25–28, 2027, Tokyo, Japan Wen Bing and Bing Li
Forced-error tasks, Func. pass@1 Original testbench, Func. pass@1
Revised mismatch testbenches, Func. pass@1 Syntactic pass@1 Revised mismatch testbenches, Func. pass@1 Syntactic pass@1
DirectT wo-shotSelf-planningSpecLens (ours)60708090Correctness (%)
RTLLM v1.1
58.0 57.6 57.061.666.1
64.2 64.0
61.669.7 69.6
67.481.085.0 85.2
82.586.2
DirectT wo-shotSelf-planningSpecLens (ours)60708090Correctness (%)
RTLLM v2.0
59.2 59.7
57.162.466.065.3
63.362.467.1 66.9
64.373.786.1 86.3
83.088.0
Figure 6: Comparison of syntactic and functional pass@1 on
RTLLM v1.1 and RTLLM v2.0 under three settings.
is lower on both datasets, as shown by the green bars. A manual
audit reveals specification–testbench mismatches in many tasks for
which the baselines are counted as correct but SpecLens is counted
as incorrect. A representative example is alu. In the released spec-
ification, the final sentence states that the LUI behavior should
use the upper 16 bits of input a. However, the baselines still of-
ten generate Verilog that matches the released testbench rather
than this local specification statement. To further verify this effect,
we intentionally replaced the local statement with the following
version: “For the LUI operation, the upper 8 bits of input aare
concatenated with 8 zeros to form the result.” Even after this ex-
plicit local change, the three baselines still frequently produced the
original benchmark-aligned implementation. This suggests that the
baselines are strongly influenced by pretraining and tend to pre-
serve previously learned RTL patterns, such as standard LUI rules,
instead of following local specification changes. As a result, an im-
plementation may pass the released testbench without necessarily
conforming to the current specification.
Specification–testbench mismatches are identified in six tasks
in RTLLM v1.1: asyn_fifo, multi_pipe_8bit, parallel2serial, alu, traf-
fic_light, and width_8to16. RTLLM v2.0 contains the same six cases
plus one additional case, fixed_point_substractor. Based on these
identified mismatch cases, we further report two additional evalua-
tion settings by changing only how these tasks are counted, while
leaving all other tasks unchanged. In both settings, the evaluated
Verilog candidates are exactly the same as those already produced
and tested under the originally released testbenches, and no new
Verilog is generated. Therefore, syntactic pass@1 is unchanged.
SpecLens has the highest syntactic pass@1, at 86.2% on RTLLM
v1.1 and 88.0% on RTLLM v2.0. In the forced-error mismatch tasks
setting, all seven tasks are counted as functionally incorrect forall four methods; this result is shown by the orange bars. Notably,
SpecLens has the same functional pass@1 under the forced-error
setting and under the original testbench setting, because none of
these seven tasks passes the originally released testbench under
SpecLens. In the revised-testbench setting, the original specification
and testbench for each of these seven tasks are jointly provided to
the LLM to generate a revised testbench, which is then manually
reviewed to ensure better consistency with the specification; this
result is shown by the purple bars. The goal of this revision is to re-
move clear mismatches between the specification and the released
testbench. For example, in asyn_fifo , where the specification gives
conflicting definitions of DEPTH , the revised testbench checks only
the external FIFO behaviors shared by both interpretations, thereby
avoiding dependence on either ambiguous interpretation. Under
the revised testbenches, SpecLens reaches 81.0% functional pass@1
on RTLLM v1.1 and 73.7% on RTLLM v2.0.
These results suggest that direct prompting more readily pre-
serves standard RTL templates and architectural priors acquired
during pretraining. Few-shot prompting and self-planning can en-
rich the input context, but they do not force the model to prioritize
a modified local statement over stronger benchmark-aligned priors.
By contrast, SpecLens is designed to surface and reinforce such lo-
cal behavioral statements, making its outputs more likely to depart
from pretrained answers when the specification changes.
The derived disambiguation constraints are transferable across
models. On VerilogEval v2.0, using them as additional guidance
improves GPT-4o-mini’s functional pass@1 ratio by 6.84% relative
to using the original specification alone, reaching pass@ 1=54.7%.
4.3 Ablation Study
Effect of Prefiltering. Stage I prefiltering prevents unnecessary re-
quirements from entering downstream constraint derivation. Under
our main setting on VerilogEval v2.0, it reduces token consumption
by 58.7% to 28.7M tokens and runtime by 72.7% to 20,527 seconds,
with nearly unchanged functional correctness.
Effect of Entropy-guided Constraint Validation. On VerilogEval
v2.0, removing entropy filtering reduces functional pass@1 from
86.2% to 84.0%. Without it, newly generated constraints are applied
directly even when they are redundant or partially misleading.
Importance of Explicit Requirement Extraction. When the require-
ment extractor and aggregator are removed and scenarios are gen-
erated directly from the specification, functional pass@1 on Verilo-
gEval v2.0 decreases from 86.2% to 82.9%. Scenario coverage also
decreases, from 891 generated scenarios with requirement extrac-
tion to 781 without it.
Effect of Stimulus Refinement. On VerilogEval v2.0, the average
stimulus length in Stage I increases from 3.85 to 4.32 steps. In
Stage II, the average stimulus length increases from 7.34 to 8.71
steps, a gain of about 19%. The task-level probability that stimulus
refinement produces an entropy-improving stimulus is 3.84%.
5 Conclusion
This paper presents SpecLens, an automated framework for LLM-
based Verilog generation that derives disambiguation constraints
from behavioral divergence among candidate implementations.
Without relying on large training corpora or external Verilog li-
braries, SpecLens is more general and scalable than fine-tuning

SpecLens: LLM-Based Verilog Generation with Specification-Derived Constraints via Behavioral Divergence ASPDAC ’27, January 25–28, 2027, Tokyo, Japan
and retrieval-based methods. Experiments show that it improves
functional correctness on VerilogEval v2.0 and produces more
specification-faithful Verilog on RTLLM under flawed or locally
modified specifications.
References
[1]Hammond Pearce, Benjamin Tan, and Ramesh Karri. DAVE: Deriving Automati-
cally Verilog from English.ACM/IEEE Workshop on Machine Learning for CAD
(MLCAD), 2020. doi: https://doi.org/10.1145/3380446.3430634.
[2]N. Wang et al. Large Language Model for Verilog Generation with Code-Structure-
Guided Reinforcement Learning.IEEE International Conference on LLM-Aided
Design (ICLAD), 2025. doi: https://doi.org/10.1109/ICLAD65226.2025.00025.
[3]K. Xu et al. Automated C/C++ Program Repair for High-Level Synthesis via Large
Language Models.ACM/IEEE International Symposium on Machine Learning for
CAD (MLCAD), 2024. doi: https://doi.org/10.1145/3670474.3685953.
[4]R. Qiu et al. AutoBench: Automatic Testbench Generation and Evaluation Using
LLMs for HDL Design.ACM/IEEE International Symposium on Machine Learning
for CAD (MLCAD), 2024. doi: https://doi.org/10.1145/3670474.3685956.
[5]R. Qiu et al. CorrectBench: Automatic Testbench Generation with Functional
Self-Correction Using LLMs for HDL Design.Design, Automation & Test in Europe
Conference (DATE), 2025. doi: https://doi.org/10.23919/DATE64628.2025.10992873.
[6]R. Qiu et al. ConfiBench: Automatic Testbench Generation with Confidence-
Based Scenario Mask and Testbench Ensemble Using LLMs for HDL Design.ACM
Transactions on Design Automation of Electronic Systems (TODAES)31(6), 2026.
doi: https://doi.org/10.1145/3773087.
[7]L. Huang et al. A Survey on Hallucination in Large Language Models: Principles,
Taxonomy, Challenges, and Open Questions.ACM Transactions on Information
Systems (TOIS)43(2), 2025. doi: https://doi.org/10.1145/3703155.
[8]Y. Zhao et al. MAGE: A Multi-Agent Engine for Automated RTL Code Generation.
ACM/IEEE Design Automation Conference (DAC), 2025. doi: https://doi.org/10.
1109/DAC63849.2025.11133191.
[9]Chia-Tung Ho, Haoxing Ren, and Brucek Khailany. VerilogCoder: Autonomous
Verilog Coding Agents with Graph-Based Planning and Abstract Syntax Tree
(AST)-Based Waveform Tracing Tool.AAAI Conference on Artificial Intelligence
(AAAI), 2025. doi: https://doi.org/10.1609/aaai.v39i1.32007.
[10] Y. Liu et al. DeepRTL2: A Versatile Model for RTL-Related Tasks.Association
for Computational Linguistics (ACL), 2025. doi: https://doi.org/10.18653/v1/2025.
findings-acl.336.
[11] Y. Zhao et al. CodeV: Empowering LLMs with HDL Generation through Multi-
Level Summarization.IEEE Transactions on Computer-Aided Design of Integrated
Circuits and Systems (TCAD), 45(4), 2026. doi: https://doi.org/10.1109/TCAD.2025.
3604320.
[12] M.-C. Chen et al. SiliconMind-V1: Multi-Agent Distillation and Debug-Reasoning
Workflows for Verilog Code Generation. https://arxiv.org/abs/2603.08719.
[13] J. Blocklove et al. Chip-Chat: Challenges and Opportunities in Conversational
Hardware Design.ACM/IEEE Workshop on Machine Learning for CAD (MLCAD),
2023. doi: https://doi.org/10.1109/MLCAD58807.2023.10299874.
[14] F. Mu et al. ClarifyGPT: A Framework for Enhancing LLM-Based Code Generation
via Requirements Clarification.Proceedings of the ACM on Software Engineering
(PACMSE), 2024. doi: https://doi.org/10.1145/3660810.
[15] Z. Ibnat et al. DeepV: A Model-Agnostic Retrieval-Augmented Framework for
Verilog Code Generation with a High-Quality Knowledge Base. https://arxiv.org/
abs/2510.05327.
[16] Yunda Tsai, Mingjie Liu, and Haoxing Ren. RTLFixer: Automatically Fixing
RTL Syntax Errors with Large Language Model.ACM/IEEE Design Automation
Conference (DAC), 2024. doi: https://doi.org/10.1145/3649329.3657353.
[17] Jiayu Zhao and Song Chen. VeriGRAG: Enhancing LLM-Based Verilog Code
Generation with Structure-Aware Soft Prompts. https://arxiv.org/abs/2510.15914.
[18] Y. Fu et al. GPT4AIGChip: Towards Next-Generation AI Accelerator Design
Automation via Large Language Models.IEEE/ACM International Conference on
Computer-Aided Design (ICCAD), 2023. doi: https://doi.org/10.1109/ICCAD57390.
2023.10323953.
[19] Y. Lu et al. RTLLM: An Open-Source Benchmark for Design RTL Generation with
Large Language Model.Asia and South Pacific Design Automation Conference
(ASP-DAC), 2024. doi: https://doi.org/10.1109/ASP-DAC58780.2024.10473904.
[20] H. Jia et al. Automated Repair of Ambiguous Problem Descriptions for LLM-Based
Code Generation.IEEE/ACM International Conference on Automated Software
Engineering (ASE), 2025. doi: https://doi.org/10.1109/ASE63991.2025.00038.
[21] Z. Zhao et al. VRank: Enhancing Verilog Code Generation from Large Language
Models via Self-Consistency.International Symposium on Quality Electronic
Design (ISQED), 2025. doi: https://doi.org/10.1109/ISQED65160.2025.11014398.
[22] Z. Zhao et al. VFocus: Better Verilog Generation from Large Language Model via
Focused Reasoning.IEEE International System-on-Chip Conference (SOCC), 2025.
doi: https://doi.org/10.1109/SOCC66126.2025.11235465.
[23] N. Pinckney et al. Revisiting VerilogEval: A Year of Improvements in Large-Language Models for Hardware Code Generation.ACM Transactions on Design
Automation of Electronic Systems (TODAES)30(6), 2025. doi: https://doi.org/10.
1145/3718088.
[24] S. Liu et al. OpenLLM-RTL: Open Dataset and Benchmark for LLM-Aided Design
RTL Generation.IEEE/ACM International Conference on Computer-Aided Design
(ICCAD), 2025. doi: https://doi.org/10.1145/3676536.3697118.
[25] Icarus Verilog Contributors. Icarus Verilog. https://github.com/steveicarus/
iverilog. Accessed July 8, 2026.
[26] Wilson Snyder and Verilator Contributors. Verilator. https://verilator.org. Ac-
cessed July 8, 2026.
[27] M. Chen et al. Evaluating Large Language Models Trained on Code. https:
//arxiv.org/abs/2107.03374.