# ProtoMedAgent: Multimodal Clinical Interpretability via Privacy-Aware Agentic Workflows

**Authors**: Alvaro Lopez Pellicer, Plamen Angelov, Marwan Bukhari, Yi Li, Eduardo Soares, Jemma Kerns

**Published**: 2026-05-13 20:57:37

**PDF URL**: [https://arxiv.org/pdf/2605.14113v1](https://arxiv.org/pdf/2605.14113v1)

## Abstract
While interpretable prototype networks offer compelling case-based reasoning for clinical diagnostics, their raw continuous outputs lack the semantic structure required for medical documentation. Bridging this gap via standard Retrieval-Augmented Generation (RAG) routinely triggers ``retrieval sycophancy,'' where Large Language Models (LLMs) hallucinate post-hoc rationalizations to align with visual predictions. We introduce ProtoMedAgent, a framework that formalizes multimodal clinical reporting as an iterative, zero-gradient test-time optimization problem over a strict neuro-symbolic bottleneck. Operating on a frozen prototype backbone, we distill latent visual and tabular features into a discrete semantic memory. Online generation is strictly constrained by exact set-theoretic differentials and a reflective Scribe-Critic loop, mathematically precluding unsupported narrative claims. To safely bound data disclosure, we introduce a semantic privacy gate governed by $k$-anonymity and $\ell$-diversity. Evaluated on a 4,160-patient clinical cohort, ProtoMedAgent achieves 91.2\% Comparison Set Faithfulness where it fundamentally outperforms standard RAG (46.2\%). ProtoMedAgent additionally leverages a binding $\ell$-diversity phase transition to systematically reduce artifact-level membership inference risks by an absolute 9.8\%.

## Full Text


<!-- PDF content starts -->

ProtoMedAgent: Multimodal Clinical Interpretability via Privacy-Aware Agentic
Workflows
Alvaro Lopez Pellicer1,†, Plamen Angelov1, Marwan Bukhari2, Yi Li1, Eduardo Soares3, Jemma Kerns2
1School of Computing and Communications, Lancaster University
2Lancaster Medical School, Lancaster University3PUC-Rio, Puc-Behring Institute for AI
Abstract
While interpretable prototype networks offer compelling
case-based reasoning for clinical diagnostics, their raw
continuous outputs lack the semantic structure required
for medical documentation. Bridging this gap via stan-
dard Retrieval-Augmented Generation (RAG) routinely trig-
gers “retrieval sycophancy, ” where Large Language Models
(LLMs) hallucinate post-hoc rationalizations to align with
visual predictions. We introduce ProtoMedAgent, a frame-
work that formalizes multimodal clinical reporting as an
iterative, zero-gradient test-time optimization problem over
a strict neuro-symbolic bottleneck. Operating on a frozen
prototype backbone, we distill latent visual and tabular fea-
tures into a discrete semantic memory. Online generation is
strictly constrained by exact set-theoretic differentials and
a reflective Scribe-Critic loop, mathematically precluding
unsupported narrative claims. To safely bound data dis-
closure, we introduce a semantic privacy gate governed by
k-anonymity and ℓ-diversity. Evaluated on a 4,160-patient
clinical cohort, ProtoMedAgent achieves 91.2% Compar-
ison Set Faithfulness where it fundamentally outperforms
standard RAG (46.2%). ProtoMedAgent additionally lever-
ages a binding ℓ-diversity phase transition to systematically
reduce artifact-level membership inference risks by an abso-
lute 9.8%.
1. Introduction
In clinical practice, high-stakes diagnostic assessment is
inherently multimodal. For instance, in bone health screen-
ing, DEXA imaging provides dense structural and spatial
cues, while tabular patient variables (e.g., age, BMI, FRAX
risk factors) contextualize the overarching disease sever-
ity [2,31]. While modern deep networks readily optimize
over these heterogeneous modalities to achieve high predic-
tive accuracy, they leave a critical gap in clinician-facing
†Corresponding author. Email: a.lopezpellicer@lancaster.ac.uk
Figure 1.Overview of an Agentic Clinical Report Generation
frameworkA multimodal patient case, comprising a lumbar DEXA
scan and a patient record, is first processed by a frozen prototype
backbone that retrieves raw visual exemplars and tabular statistics.
Reconciling these outputs through unconstrained prototype seman-
tics via a generic LLM yields an ungrounded, hallucination-prone
clinical report. In contrast, by enforcing constrained prototype
semantics, ProtoMedAgent utilizes a suite of dedicated agents (Vi-
sion, Tabular, Memory, and Verification) to safely translate retrieved
evidence into a fully grounded and verifiable clinical report.
documentation: they fundamentally fail to generate trans-
parent, auditable, and privacy-compliant rationales. This
creates a mismatch between model performance and clin-
ical usability, where accurate predictions lack trustworthy
justification. This need is increasingly regulatory as well as
technical: emerging EU rules reward clinical AI designs that
are auditable, human-overseeable, robust, and compatible
with controlled data access and interoperable data infrastruc-
tures [10, 11].
Interpretable-by-design prototype networks have emergedarXiv:2605.14113v1  [cs.CV]  13 May 2026

as a compelling alternative, offering case-based reasoning
by exposing training exemplars with the logic of “this looks
like that” [ 4,26,27]. Yet, the raw evidence surface of proto-
type retrieval—comprising scalar similarity scores and con-
tinuous feature maps—is semantically disconnected from
clinical reporting workflows. The standard intervention of
coupling a Large Language Model (LLM) via Retrieval-
Augmented Generation (RAG) to interpret these latent proto-
types routinely collapses. When tasked with reconciling un-
constrained visual features and clinical priors, LLMs exhibit
severe multimodal hallucination and retrieval sycophancy,
fabricating similarities to post-rationalize the network’s pre-
diction [34, 36].
Inspired by recent advances in reflective agentic systems,
we introduce ProtoMedAgent (Figure 1), a framework that
bridges the gap between continuous visual reasoning and
discrete clinical documentation. We hypothesize that faith-
ful multimodal reporting requires decoupling thediagnos-
tic forward passfrom theevidence documentation. Rather
than treating reporting as an unconstrained generative pass,
we cast it as aniterative LLM-in-the-loop test-time opti-
mization problemoperating over a strict neuro-symbolic
bottleneck.
Our framework acts upon a frozen prototype backbone,
ProtoMedX [ 31]. Offline, we project the continuous pro-
totype space into a privacy-gated semantic memory, distill-
ing visual and tabular distributions into discrete assertions
bounded by strict k-anonymity and ℓ-diversity. Online, we
introduce a reflective refinement loop: exact set-theoretic dif-
ferentials are computed between the query and the retrieved
memory, establishing an absolute admissible state space. A
Scribe-Critic loop then iteratively optimizes the generated
report, utilizing the LLM as both a proposal distribution
and a constraint evaluator until the artifact achieves perfect
neuro-symbolic alignment.
This work investigates the following research questions:
•RQ1 (Neuro-Symbolic Grounding):How can latent mul-
timodal prototype retrieval be converted into a discrete
semantic bottleneck that promotes evidence-grounded gen-
eration and reduces the space for unsupported LLM out-
puts?
•RQ2 (Test-Time Optimization):Can an iterative reflec-
tive critic at test time improve reporting faithfulness with-
out requiring gradient updates or altering the predictive
behavior of the frozen vision backbone?
Contributions.
1.Agentic LLM-in-the-Loop Test-Time Optimization:
We formalize clinical reporting as a zero-gradient test-
time optimization process, leveraging an offline privacy-
gated memory and an online reflective reasoning loop to
safely decode a frozen prototype classifier.
2.Neuro-Symbolic Verification:We replace heuristicLLM comparisons with deterministic set-theoretic dif-
ferentials. This is enforced via an iterative Scribe-Critic
optimization loop that guarantees citation accuracy and
typed-value consistency, fundamentally mitigating re-
trieval sycophancy.
3.Provable Privacy-Utility Bounds:We introduce a
novel semantic release gate utilizing k-anonymity and
ℓ-diversity for continuous prototype memory, demonstrat-
ing mathematically bounded disclosure risk and improved
empirical faithfulness on a 4,160-patient clinical cohort.
2. Related Work
Interpretable-by-design prototype models natively explain
their predictions by retrieving latent exemplars from the
training distribution, operationalizing the intuitive logic of
“this looks like that” [ 1,4,26,27]. In high-stakes medi-
cal domains, such inherently transparent architectures are
strongly preferred over post-hoc explainers [ 35]. While re-
cent approaches like ProtoMedX [ 31] advance this paradigm
via input-dependent multimodal gating, the native evidence
surface of these models—comprising scalar similarities and
continuous feature activations—exhibits a profound seman-
tic gap when mapped to discrete clinical reporting workflows.
Recent works have explored the use of LLMs for clinical
report generation and multimodal reasoning over medical
images and structured data [ 25], including radiology report
generation [ 6] and clinical decision support systems [ 40].
However, these approaches typically rely on end-to-end gen-
erative pipelines without explicit or verifiable grounding to
underlying evidence, limiting their reliability in high-stakes
settings [3, 16, 40].
To bridge this representational gap, recent advancements
have increasingly scaled test-time compute by configur-
ing Large Language Models (LLMs) as compound, reflec-
tive agents. Frameworks such as AlphaCodium [ 32] and
TimeXL [ 17] demonstrate that interleaving generation with
iterative self-critique [ 37,45] yields significant performance
gains for complex, multi-step reasoning without requir-
ing gradient updates. However, simply grounding these
agentic loops via standard Retrieval-Augmented Generation
(RAG) [ 20] introduces critical failure modes in clinical set-
tings. Tasking an LLM to heuristicallycompareretrieved
items reliably induces retrieval sycophancy and multimodal
hallucination [ 24,34,36]. Because language models tend to
prioritize internal semantic priors over contradicting external
evidence, even cited RAG outputs frequently suffer from
spurious post-rationalization [28, 42].
Modern constrained decoding engines, including Out-
lines [ 43], XGrammar [ 7], and JSONFormer [ 33], attempt
to reign in this behavior by enforcing strict syntactic schema
adherence [ 12,39]. Yet, while these tools guarantee struc-
tural validity, they cannot mathematically guarantee seman-
tic faithfulness. Resolving this tension requires replacing un-

constrained heuristic comparison with deterministic, neuro-
symbolic bottlenecks that formally bound the generative
hypothesis space.
Beyond the challenge of hallucination and other vulner-
abilities [ 30], translating retrieved training exemplars into
natural language exposes compound systems to severe pri-
vacy vulnerabilities. Surfacing raw exemplars during re-
trieval creates direct vectors for membership and attribute
inference attacks [ 15,38]. While Differential Privacy (DP)
offers robust theoretical bounds via noise injection [ 13,29],
it often catastrophically degrades the exact clinical fidelity
required in dense medical imagery. Alternatively, classical
k-anonymity [ 41] and its semantic extensions, such as ℓ-
diversity [ 23] and t-closeness [ 22], formalize disclosure risk
without injecting continuous noise. By conceptualizing these
classical privacy mechanisms as a non-differentiable Markov
bottleneck prior to LLM context aggregation, it becomes pos-
sible to mathematically secure generated text artifacts against
exact linkage and attribute inference, thereby satisfying strin-
gent medical data regulations [ 8,9] without modifying the
underlying training dynamics. Beyond privacy law, this de-
sign is also well aligned with emerging EU digital regulation.
The EU AI Act foregrounds risk management, technical doc-
umentation, logging, deployer-facing transparency, human
oversight, and lifecycle accuracy/robustness for high-risk
AI systems, while the EU Data Act reinforces the value of
controlled data access, portability, and interoperable data-
processing infrastructures. Together, these trends further
motivate privacy-aware, modular, and auditable semantic
abstractions for clinical AI deployment [10, 11].
3. ProtoMedAgent
ProtoMedAgent conceptualizes clinical reporting as an
evidence-grounded translation problem under explicit test-
time constraints, as illustrated in Figure 2. By preserv-
ing a frozen predictive backbone and introducing a neuro-
symbolic bottleneck, language generation is restricted to
operate as an iterative, deterministic interpreter rather than
an unconstrained confabulator. Concretely, we decouple
prediction from reporting by constraining the generative pro-
cess to operate over a verified, discrete semantic state derived
from prototype evidence.
3.1. Problem Setting
Let a patient query be x= (I,t) , comprising a high-
dimensional visual input Iand structured tabular features
t. We utilize a frozen prototype network, ProtoMedX [ 31],
which maps xto a predicted class ˆy, an auxiliary severity
scalar ˆz(x), a modality attention gate α(x)∈[0,1] , and a
retrieved prototype neighborhood:
Nk(x) ={(j, w j, sj)}k
j=1where jis the prototype index, wjthe voting weight, and sj
the continuous retrieval similarity.
Our goal is to generate a natural language report Rthat
maximizes clinical fluency while strictly adhering to the evi-
dence encoded in Nk(x). To achieve this without gradient
updates to the generative model, we project xandNk(x)
into a shared, discrete semantic space of structured clinical
assertions, computing exact morphological and clinical dif-
ferentials that define the admissible semantic state prior to
LLM-in-the-loop generation.
3.2. Offline Memory Distillation & Privacy
Directly exposing raw training images ( Ij) to an LLM in-
vites sycophantic hallucinations and introduces unaccept-
able privacy attack vectors. We distill training prototypes
pj= (I j,tj, yj)into a discrete memory bank ofProtoCards,
Cj.
Perceptual and Tabular Abstraction.A radiology Vision-
Language Model (VLM), gvlm, serves as a Perception Agent.
To extract robust features, gvlmevaluates Ijunder Rcom-
plementary prompting views. We apply embedding-based
clustering and contradiction filtering to extract a conserva-
tive set of visual assertions Aj. Concurrently, a deterministic
Tabular Agent projects continuous patient variables to quan-
tized bins ˜Tj=Q(t j). A schema-constrained generator
gcardformats the tuple (Aj,˜Tj)into a structured ProtoCard
Cj.
Semantic Privacy Gate.We operationalize a semantic
privacy gate acting as a non-differentiable regularizer over
the memory bank. We enforce a pragmatic k-anonymity
andℓ-diversity rule over a release signature σ(j), derived
from quantized clinical fields and semantic buckets of the
consensus assertions. The visible export surface is:
Ψ(j) =

Cj,ifcnt(σ(j))≥k
andldiv(σ(j))≥ℓ,
[REDACTED],otherwise.(1)
Redacted prototypes contribute to the diagnostic forward
pass but are completely obfuscated from the LLM’s test-
time reasoning context.
3.3. Online Reflective Test-Time Optimization
During inference, the query xis similarly abstracted into a
CaseCard containing visual assertions A(x) and quantized
data˜T(x).
Deterministic Differentials and Visual Deferral.To pre-
vent the LLM from hallucinating correlations, a Supervi-
sor agent evaluates the alignment between A(x) and the
retrieved prototype signatures. If discordance exceeds a

Figure 2. Detailed architecture of ProtoMedAgent. (a)Offline Memory Distillation & Privacyabstracts raw images and tabular records
into a discrete semantic memory bank governed by a k-anonymity and ℓ-diversity privacy gate. (b) The intermediateSemantic Memory
Banksecurely stores verifiable ProtoCards. (c)Online Reflective Test-Time Optimizationmaps a new patient query into a CaseCard,
computes exact set-theoretic differentials ( ∆j), and applies a zero-gradient Scribe-Critic optimization loop to decode the frozen ProtoMedX
backbone into a verifiable report.
threshold τ, the framework triggersvisual deferral, sup-
pressing image-based similarities to avoid forcing the model
to rationalize contradictions.
For visible prototypes ¯Cj= Ψ(j) , we bypass heuristic
LLM-driven reasoning by computing an exact set-theoretic
differential:
∆j= 
A(x)∩A j, A(x)\A j, Aj\A(x),∆tab
j
where ∆tab
jrepresents the deterministic mismatch in clinical
factors.
Zero-Gradient LLM-in-the-Loop Search.We frame re-
port generation as an iterative, zero-gradient search over the
discrete state space of natural language reports R. The Critic
agent gcriticparameterizes an energy function Ecritic:R →
R≥0defined by a multi-objective constraint:
Ecritic(Rt) =λ 1Lschema +λ 2Lcite+λ 3Ltype+λ 4LNLI
where the loss terms penalize JSON invalidity, citation hal-
lucination against the grounded state E(x) , typed-value in-
consistency, and logical entailment failures. The Scribe
agent gscribe acts as a conditional proposal distribution
Pscribe(Rt+1|E(x),∇E critic(Rt)), where the gradient∇E criticis instantiated as structured natural language reflec-
tions. This process is formalized in Algorithm 1.
Algorithm 1LLM-in-the-Loop Reflective Optimization
Require: Grounded state E(x) , max iterations T, energy
thresholdϵ= 0
1:R 0∼P scribe(· |E(x))
2:fort= 0toT−1do
3:Evaluate energye t=E critic(Rt)
4:ife t≤ϵthen
5:returnR t ▷Convergence to faithful state
6:end if
7:Generate reflective critiquec t← ∇E critic(Rt)
8:Sample new reportR t+1∼P scribe(· |E(x), c t)
9:end for
10:returnDEFER ▷Failed to satisfy neuro-symbolic
constraints
3.4. Theoretical Guarantees of the Framework
By restructuring multimodal reporting as an energy-based
optimization problem over a neuro-symbolic bottleneck, Pro-
toMedAgent provides two distinct operational guarantees
regarding generative faithfulness and privacy exposure. For-

mal proofs for both theorems are provided in Supplementary
Section B.
Theorem 1 (Bounded Generative Faithfulness):Let R
be the hypothesis space of all possible natural language re-
ports, and let the deterministic critic parameterize a barrier
energy function Ecritic:R → {0,∞} indicating an admis-
sible constraint set. If the optimization converges such that
Ecritic(R∗) = 0 , the set of atomic semantic claims within
the accepted report R∗is strictly bounded by the determinis-
tic differential ∆j, mathematically precluding ungrounded
generative confabulation.
This theorem establishes that while the underlying Vision-
Language Model (VLM) dictates the absolute visual truth
of the system, the generative LLM is restricted from hallu-
cinating structural or tabular comparisons outside the pre-
computed intersection (A(x)∩A j)and relative complements
(A(x)\A j, Aj\A(x)) . The generative hypothesis space is
thereby safely bounded to strict interpretation of the sym-
bolic state rather than open-ended invention.
Theorem 2 (Information-Theoretic Privacy Bound):
Assuming the export function Ψ(j) enforces k-anonymity
and strict ℓ-diversity, and an adversary Apossesses a prior
that matches the empirical distribution of the equivalence
class (i.e., no auxiliary background knowledge), the maxi-
mum posterior probability of Asuccessfully executing an
exact record linkage attack from the generated report R∗
is upper-bounded by 1/k, and the probability of targeted
attribute inference is upper-bounded by1/ℓ.
Because the semantic privacy gate acts as a non-
differentiable Markov bottleneck prior to test-time aggrega-
tion, the generative model operates exclusively on a sanitized
context space. By the conditional independence established
by the Markov property, the LLM cannot reconstruct case-
specific signatures destroyed by Ψ(j) , neutralizing artifact-
level inversion attacks regardless of the LLM’s learned se-
mantic priors.
4. Experiments
4.1. Task and dataset
We study three-class bone-health classification (NORMAL,
OSTEOPENIA, OSTEOPOROSIS) from lumbar-spine DEXA
scans and structured clinical risk factors. We follow the
dataset and protocol of ProtoMedX [ 31]: 4,160 de-identified
NHS patients with 11 FRAX®-aligned variables, using the
published preprocessing pipeline and an 80/20 train–test
split with an internal 10% validation split.
4.2. Frozen encoder & memory
All experiments use one frozen ProtoMedX checkpoint. For
every case, the report writer receives the same backbone
prediction ˆy, auxiliary T-score ˆz(x), modality gate α(x) , and
retrieved neighborhood Nk(x). The backbone is never fine-tuned inside the reporting experiments, so any differences in
report quality arise from the evidence-to-language interface
rather than from re-training the predictor.
4.3. Backbone Agent choices and baselines
For the Perception Agent, we use openly available medical
VLMs (RadFM [ 44] and LLaV A-Med [ 21]) to extract atomic
image assertions under multiple prompts, followed by the
consensus step described in Sec. 3.2. For ProtoCard writing
and report generation, we use medical instruction models
such as BioMistral [ 19] and Meditron [ 5] under structured
decoding [7, 43].
We compare against two reporting baselines that operate
on the same frozen ProtoMedX outputs.Standard RAG
conditions an LLM on the retrieved evidence and directly
generates a report.LLM (LoRA)uses a domain-adapted
schema-filling model [ 14], but still relies on the model to
infer case–prototype agreement implicitly.ProtoMedAgent
adds privacy-gated ProtoCards, deterministic comparison
objects, supervisor-controlled visual deferral, and a final
Scribe–Critic verification loop.
4.4. Evaluation protocol
ProtoMedAgent is accuracy-neutral, so we evaluate the re-
porting layer rather than predictive accuracy. Our primary
metric isComparison Set Faithfulness(CSF): for each
query–prototype pair, we derive deterministic reference sets
of shared and differing evidence items from the comparison
object ∆j, extract the items verbalized in the final report,
and compute CSF precision, recall, F1, and weighted ac-
curacy. These metrics test whether a report preserves the
case–prototype comparison actually available to the system.
We also evaluate theprivacy–utilitybehavior of the re-
leased evidence surface. First, we sweep the release gate
over different (k, ℓ) values and report similarity-weighted ev-
idence utility, visible-card rate, redaction rate, and top-1 link-
age success rate (Link.) on the ProtoCard surface. Second, at
the selected operating point, we evaluate three artifact-level
disclosure attackson the released outputs: Membership-
Inference Accuracy (MIA), which tests whether an attacker
can determine whether the underlying case belongs to the
protected source corpus [ 38]; Attribute-Inference Accuracy
(AIA), which tests whether sensitive source attributes can be
inferred from the released artifact, potentially with auxiliary
information [ 15]; and top-1 Linkage success rate (Link.),
which tests whether sparse released cues suffice to re-identify
the correct source record [ 18]. These attacks are directly
relevant because ProtoMedAgent’s privacy gate and non-
differentiable Markov bottleneck are designed to suppress
residual membership leakage, sensitive-attribute disclosure,
and exact re-identification risk.
Formal definitions for each of our evaluation metrics are
provided in Supplementary material section A.

5. Results
Because ProtoMedAgent operates strictly as a test-time re-
porting layer over a frozen ProtoMedX backbone, it is de-
signed to be accuracy-neutral with respect to the underlying
predictive task. Our evaluation therefore isolates the efficacy
of the agentic translation layer. Specifically, we investigate
whether formulating report generation as an iterative op-
timization over a discrete neuro-symbolic bottleneck can
enforce strict comparison faithfulness and bound privacy
exposure, without sacrificing clinical fluency. Accordingly,
all reported improvements reflect gains in reporting fidelity,
safety, and privacy rather than predictive performance.
5.1. Safety and Comparison Faithfulness
To evaluate the fidelity of the generated reports against the
retrieved visual evidence (answering RQ2), we introduce
the Comparison Set Faithfulness (CSF) metric. CSF iso-
lates the explicit matches and differences asserted in the
generated text and evaluates them against the deterministic,
set-theoretic differentials ( ∆j) computed between the query
and the retrieved ProtoCards, providing a direct measure of
evidence-grounded comparison fidelity.
As reported in Table 1, ProtoMedAgent achieves state-
of-the-art performance for grounded clinical reporting. By
decoupling the diagnostic forward pass from evidence docu-
mentation, our framework restricts the Scribe agent to nar-
rating only those evidence-supported partitions backed by
exact citations. Furthermore, the iterative Critic-repair loop
promotes item-level consistency: any hallucinated correla-
tion that cannot be traced to the admissible semantic state is
dynamically corrected or removed. This behavior is consis-
tent with Theorem 1, which bounds the generative space to
evidence-supported differentials.
This mechanism directly addresses the vulnerability of
standard generative approaches. Baselines such as Standard
RAG [ 20] and domain-adapted LoRA [ 14] rely on implicit,
unconstrained reasoning over retrieved contexts. Without
explicit neuro-symbolic constraints, these models are prone
to retrieval sycophancy, inventing post-hoc rationalizations
to align with the backbone’s prediction. This failure mode is
empirically confirmed by their degradation in CSF-Precision
(46.2% and 63.1%, respectively) compared to the bounded
output of ProtoMedAgent (91.2%).
5.2. Privacy-Utility Dynamics & Disclosure Risk
We evaluate the semantic privacy gate (answering RQ1)
through a dual-lens approach: (i) arelease-surface analy-
sismapping the (k, ℓ) configuration space to characterize
the privacy-utility frontier, and (ii) rigorousartifact-level
attack probesmeasuring residual disclosure risk on the final
generated text.
A key empirical finding illustrated in Fig. 3(a) is the
phase transition governed by ℓ-diversity. Under ℓ=1 , scal-Table 1. Safety and Faithfulness Study on the test cohort. By
strictly bounding the generative hypothesis space to pre-computed
deterministic differentials, ProtoMedAgent fundamentally miti-
gates multimodal hallucination and unsupported claims.
Method CSF-P↑CSF-R↑CSF-F1↑CSF-WA↑
Standard RAG 46.2 41.5 43.7 58.6
LLM (LoRA) 63.1 58.9 60.9 70.2
Ours 91.2 78.0 84.1 92.7
0.0 0.1 0.2 0.3 0.4
Top-1 linkage on released ProtoCards  
0.00.20.40.60.81.0Evidence utility  
k=3k=9
(k,)=(5,2)
=3: zero release
=2 decisive
(a)  Release-surface frontier
=1
=2  (all k)
=3
Evidence
utility
Visible-card
rate
Redaction
rate
Linkage
exposure0.673
0.593
0.407
0.167(b)  Operating point (k,)=(5,2)
Figure 3.Privacy-utility frontier of the exportable prototype ev-
idence surface.Sweeping semantic k-anonymity ( k∈{3,5,7,9} )
andℓ-diversity ( ℓ∈{1,2,3} ).(a)Under ℓ=1 (orange), increasing
kprovides a smooth trade-off between utility and exposure. Im-
posing ℓ= 2 (green) triggers a phase transition, collapsing all k
values to a single bounded regime, indicating diversity binds be-
forek-anonymity.(b)The selected operating point (k, ℓ)=(5,2)
retains robust evidence utility (0.673) while suppressing linkage
exposure to 0.167.
ingk-anonymity yields a smooth, proportional decay in both
evidence utility and linkage exposure. However, enforcing
ℓ= 2 causes all tested kvalues to collapse into an iden-
tical, highly constrained operating regime (utility = 0.673,
linkage = 0.167). This indicates that the diversity of the un-
derlying clinical attributes becomes the dominant constraint
prior to cohort size ( k-anonymity), effectively saturating the
admissible equivalence classes.
This behavior reveals an important structural property of
the semantic release surface: privacy constraints do not com-
bine linearly. Instead, ℓ-diversity introduces a combinatorial
constraint over attribute distributions, which sharply reduces
the feasible support of the exportable memory. As a result,
once diversity is enforced, increasing kyields diminishing
returns, explaining the observed collapse across all kvalues.
Pushing the constraint further to ℓ=3 uniformly eliminates
valid equivalence classes, reducing the visible surface to
zero.
At the selected operating point of (k, ℓ)=(5,2) , the gate
retains 32 of 54 candidate ProtoCards (59.3% visibility) and

Near-threshold normal study with heterogeneous lumbar preservation
Query:89F; lumbar DXA + tabular record.ProtoMedX output:NORMAL,T=−0.2, Confidence 89.4%; p1=normal proto 1[N], p2/p3 = N/N.
ProtoMedAgent report:Lumbar bone density remains within normal limits. Preservation is not uniform across the lumbar column: the lower lumbar
vertebrae appear denser than the upper lumbar vertebrae, while the upper lumbar spine sits much closer to the normal/osteopenia boundary. Mild lower-lumbar
degenerative change may be contributing to that apparent distal preservation, and incomplete coverage with indistinct vertebral labels limits exact vertebral
assignment. Alcohol use is the only notable modifiable risk factor. Compared with preserved-density normal references, the study remains closer to a
heterogeneous normal pattern than to the smoother early-loss osteopenic prototypes, so the final impression stays cautiously normal rather than low bone mass.ProtoCard N1
Prototype:normal proto 1[N].
Typical pattern: the lower lumbar spine remains distinctly denser
than the upper lumbar vertebrae and the whole column reads
comfortably normal across a broad preserved range. Prior fragility
fracture can coexist with this prototype, but unlike the boundary
normal or osteopenic references the preserved structure remains
convincing even when mild sclerosis or overlay exaggerates the
distal vertebrae.
Boundary osteopenia distinguished from preserved density
Query:88F; lumbar DXA + tabular record.ProtoMedX output:OSTEOPENIA, T=−0.8 , Confidence 69.6%; p1= osteopenia proto 0[Op],
p2/p3 = Op/N.
ProtoMedAgent report:Low bone mass consistent with osteopenia is favored despite a threshold-adjacent estimate. The scan retains a lower-lumbar-
greater-than-upper gradient, but preservation is discontinuous rather than smooth: the upper lumbar spine shows attenuation, and the column as a whole
no longer reads as uniformly preserved. Probable lower-lumbar degenerative change, focal radiopaque overlay, and incomplete coverage all argue against
overcalling the apparently denser lower vertebrae. Previous fragility fracture makes this boundary presentation clinically significant. Compared with the stronger
preserved-density normal references, the study falls short of a truly cohesive normal pattern and is better explained by an early-loss osteopenic prototype.ProtoCard Op0
Prototype:osteopenia proto 0[Op].
Typical pattern: lower lumbar reserve remains visible, but preser-
vation is uneven across the column and the upper lumbar spine
shows reduced density relative to the lower vertebrae. Rotation,
limited coverage, overlay, and degenerative change are common,
which is why this prototype anchors careful early-loss osteopenic
wording.
Established osteoporosis with diffuse lumbar demineralization
Query:78F; lumbar DXA + tabular record.ProtoMedX output:OSTEOPOROSIS, T=−3.7 , Confidence 96.4%; p1= osteoporosis proto 2[Os],
p2/p3 = Os/Op.
ProtoMedAgent report:Bone density is in the osteoporotic range with established skeletal fragility. The lumbar column is diffusely demineralized, and the
limited lower-lumbar density reserve is more plausibly explained by degenerative accentuation, focal overlay, and incomplete coverage than by genuinely
preserved bone. Previous fragility fracture reinforces a high fracture-risk state. Compared with the threshold-sensitive osteopenic references, the study aligns
with a cohesive low-density osteoporotic prototype, supporting a decisive osteoporosis interpretation.ProtoCard Os2
Prototype:osteoporosis proto 2[Os].
Typical study: deeper low density that sits farther from the os-
teopenic boundary than the moderate severe references. The
defining feature is depth of mineral loss rather than a dense ac-
companying risk profile, so the image itself carries most of the
diagnostic weight.
Figure 4.Final ProtoMedAgent qualitative outputs.Each row pairs a final ProtoMedAgent report at left with the displayed same-class
ProtoCard reference at right. The header line shows the query summary and compact ProtoMedX output; p1is the ProtoCard shown
beside the report, while p2/p3 summarize the remaining retrieved neighborhood. Here N = NORMAL, Op = OSTEOPENIA, and Os =
OSTEOPOROSIS. The three examples show a cautious near-threshold normal interpretation, a boundary osteopenic case clarified by an
early-loss osteopenic reference, and a decisively osteoporotic study anchored by deep low-density prototype evidence. Supplementary
Sec. S4 expands the report gallery to routine, restricted-evidence, and harder boundary cases, and Supplementary Sec. S5 provides the full
prototype atlas.
Table 2. Artifact-level disclosure risk on exported evidence. Met-
rics include Membership-Inference Accuracy (MIA), Attribute-
Inference Accuracy (AIA), and top-1 Linkage success rate (Link.).
ProtoMedAgent systematically lowers empirical attack vulnerabili-
ties by sanitizing the generative context window.
Method MIA↓AIA↓Link.↓
Standard RAG 51.2 34.2 39.3
LLM (LoRA) 49.7 33.0 38.1
Ours 41.4 25.8 34.5
deliberately enforces a 40.7% redaction rate. Importantly,
this redaction is not random but structured: removed proto-
types correspond to semantically rare or uniquely identifying
clinical configurations. This confirms our theoretical claim
that privacy-aware prototype reporting should be formulated
as an explicit operating-point optimization over a semantic
release surface, rather than treating raw exemplar exposure
as a binary heuristic.
Table 2 measures the residual disclosure risk remain-
ing in the final text artifacts. Because ProtoMedAgent
enforces generation through a non-differentiable Markov
bottleneck, the LLM is effectively prevented from recon-
structing case-specific signatures. This constraint reduces
the effective attack surface by restricting generation to ag-
gregated, equivalence-class-level information rather thaninstance-level features.
This theoretical protection translates into robust empirical
defense, yielding the lowest risk profiles across all metrics,
notably reducing MIA by an absolute 9.8% and AIA by
8.4% relative to Standard RAG. Importantly, these reduc-
tions are achieved without modifying the underlying gen-
erative model, indicating that controlling the information
interface—rather than the model parameters—is sufficient
to significantly mitigate disclosure risk.
5.3. Qualitative Results
While quantitative metrics confirm strict neuro-symbolic
adherence, Figure 4 demonstrates that this bottleneck does
not materially degrade clinical fluency. The examples illus-
trate how the LLM maps discrete ProtoCard references into
cohesive, highly contextual natural language.
Crucially, the qualitative variations across these reports
are clinically substantive rather than stochastic artifacts of
the language model. For instance, in the threshold-adjacent
osteopenic study, the narrative does not simply echo a
generic class label. Instead, it grounds its early mineral
loss interpretation by explicitly contrasting the scan against
stronger preserved-density normal prototypes, carefully ac-
knowledging the residual lower-lumbar reserve. Conversely,
the osteoporotic report synthesizes diffuse demineralization,
historical fragility fractures, and alignment with a severe low-
density reference prototype to anchor a coherent structural

degradation narrative.
Beyond descriptive fidelity, these examples highlight
three important behavioral properties of ProtoMedAgent.
First, the generated reports exhibitevidence-calibrated rea-
soning: claims are modulated by the strength and consistency
of retrieved prototypes, with boundary cases producing more
nuanced, comparative language rather than overconfident
categorical statements. Second, the framework demonstrates
counterfactual sensitivity: small shifts in the retrieved pro-
totype neighborhood lead to controlled and interpretable
changes in the generated narrative, indicating that the LLM
is responding to structured evidence rather than latent priors.
Third, the system enforcesfaithful abstraction: while the
wording remains fluent and expressive, all semantic content
can be traced back to admissible set-theoretic differentials,
preventing unsupported clinical assertions.
Importantly, we also observe that when evidence is par-
tially redacted by the privacy gate, the model adapts by pro-
ducing more conservative and less specific language, rather
than hallucinating missing details. This behavior suggests
that the neuro-symbolic bottleneck induces a form of implicit
uncertainty awareness in the generative process.
By displaying the cited ProtoCards alongside the gener-
ated text, ProtoMedAgent renders the LLM’s wording fully
auditable in concrete visual and tabular terms. Notably, all
narrative variations remain grounded in the admissible se-
mantic state, demonstrating that expressivity is preserved
without sacrificing faithfulness. Supplementary section C ex-
pands this qualitative analysis to include complex boundary
cases and restricted-evidence scenarios, while the complete
prototype atlas is provided in Supplementary material sec-
tion D.
6. Discussion and Conclusion
ProtoMedAgent formalizesmultimodal prototype retrieval
as a constrained, test-time optimization problem, effectively
bridging the semantic gap between continuous representa-
tions and discrete clinical documentation. By executing
a strict Scribe ↔Critic loop over a privacy-gated semantic
memory and a frozen ProtoMedX backbone, we reposition
the LLM as a verifiable interpretability layer. This approach
achievesstate-of-the-art Comparison Set Faithfulness
(91.2% CSF-Precision), proving that generative faithfulness
can be structurally enforced without costly model parameter
modifications. Furthermore, framingprivacy-aware report-
ingas an explicit operating-point optimization reveals a
binding ℓ-diversity phase transition. This non-differentiable
Markov bottleneck severely restricts case-specific informa-
tion flow, delivering robust empirical defense. Including a
9.8% absolute reduction in Membership Inference at-
tackswithout altering the underlying weights.
Despite these systemic guarantees, the framework re-
mains bounded by its upstream Perception Agent, which isvulnerable to correlated visual blind spots. Additionally, our
heuristic Discordance Sentinel introduces an inherent ten-
sion between generative expressivity and conservative visual
deferral. From a privacy perspective, semantic k-anonymity
andℓ-diversity lack the worst-case theoretical guarantees of
Differential Privacy, leaving residual exposure to adversaries
with out-of-schema auxiliary knowledge. Crucially, while
our neuro-symbolic bottleneck mathematically precludes
hallucinated structural claims, it strictly bounds thespace of
admissible explanationswithout guaranteeing the absolute
truthfulnessof the underlying backbone’s perception.
Nevertheless, as a non-diagnostic, assistive auditing tool
that supports explicit human deferral and continuous “patch-
ability” of prototype memories, ProtoMedAgent aligns di-
rectly with the stringent technical transparency and human
oversight mandates of theEU AI Act and EU Data Act.
To fully research this clinical utility, a critical next step
isconducting controlled user studiesto rigorously val-
idate our qualitative findings and assess the framework’s
true impact on decision-making workflows. Methodologi-
cally, future research should prioritize integrating calibrated
uncertainty into the perception layer, formalizing citation
semantics via schema-versioned references (e.g., JSONPath),
and exploring hybrid DP-symbolic architectures to extend
formal privacy bounds to continuous prototype systems.
Acknowledgment
This study is approved by the North West- Preston Research
Ethics Committee (REC ref 21/NW/0309). This work is sup-
ported by ELSA – European Lighthouse on Secure and Safe
AI funded by the European Union under grant agreement No.
101070617.
References
[1]Plamen Angelov and Eduardo Soares. Towards explainable
deep neural networks (xdnn).Neural Networks, 130:185–194,
2020. 2
[2]OFM Riaz Rahman Aranya and Kevin Desai. Trace: Tem-
poral radiology with anatomical change explanation for
grounded x-ray report generation, 2026. 1
[3]Yaara Artsi, Eyal Klang, Jeremy D Collins, Benjamin S
Glicksberg, Girish N Nadkarni, Panagiotis Korfiatis, and Vera
Sorin. Large language models in radiology reporting-a sys-
tematic review of performance, limitations, and clinical im-
plications.Intelligence-Based Medicine, page 100287, 2025.
2
[4]Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett,
Jonathan Su, and Cynthia Rudin. This looks like that: Deep
learning for interpretable image recognition. InAdvances in
Neural Information Processing Systems (NeurIPS), 2019. 2
[5]Zeming Chen, Alejandro Hern ´andez Cano, Angelika Ro-
manou, Antoine Bonnet, Kyle Matoba, Francesco Salvi,
Matteo Pagliardini, Simin Fan, Andreas K ¨opf, Amirkeivan
Mohtashami, Alexandre Sallinen, Alireza Sakhaeirad, Vini-

tra Swamy, Igor Krawczuk, Deniz Bayazit, Axel Marmet,
Syrielle Montariol, Mary-Anne Hartley, Martin Jaggi, and
Antoine Bosselut. Meditron-70b: Scaling medical pretraining
for large language models, 2023. 5
[6]Nicolas Deperrois, Hidetoshi Matsuo, Samuel Ruip ´erez-
Campillo, Moritz Vandenhirtz, Sonia Laguna, Alain Ryser,
Koji Fujimoto, Mizuho Nishio, Thomas M Sutter, Ju-
lia E V ogt, et al. Radvlm: a multitask conversational
vision-language model for radiology.arXiv preprint
arXiv:2502.03333, 2025. 2
[7]Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi
Xu, Yilong Zhao, and Tianqi Chen. Xgrammar: Flexible
and efficient structured generation engine for large language
models. arXiv:2411.15100, 2024. 2, 5
[8]Electronic Code of Federal Regulations. 45 cfr §164.514:
Other requirements relating to uses and disclosures of pro-
tected health information (de-identification), 2025. Accessed:
2026-02-18. 3
[9]European Parliament and Council of the European Union.
Regulation (eu) 2016/679 (general data protection regulation),
2016. 3
[10] European Parliament and Council of the European Union.
Regulation (eu) 2023/2854 of the european parliament and of
the council of 13 december 2023 on harmonised rules on fair
access to and use of data (data act). Official Journal of the
European Union, OJ L 2023/2854, 22 December 2023, 2023.
Accessed: 2026-03-22. 1, 3
[11] European Parliament and Council of the European Union.
Regulation (eu) 2024/1689 of the european parliament and
of the council of 13 june 2024 laying down harmonised rules
on artificial intelligence (artificial intelligence act). Official
Journal of the European Union, OJ L 2024/1689, 12 July
2024, 2024. Accessed: 2026-03-22. 1, 3
[12] Saibo Geng, Hudson Cooper, Michał Moskal, Samuel Jenkins,
Julian Berman, Nathan Ranchin, Robert West, Eric Horvitz,
and Harsha Nori. Generating structured outputs from lan-
guage models: Benchmark and studies. arXiv:2501.10868,
2025. 2
[13] M. Emre Gursoy, Asim Inan, M. Emin Nergiz, and Yucel
Saygin. Differentially private nearest neighbor classification.
Data Mining and Knowledge Discovery, 31(5):1544–1575,
2017. 3
[14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
LoRA: Low-rank adaptation of large language models. In
International Conference on Learning Representations, 2022.
5, 6
[15] Bargav Jayaraman and David Evans. Are attribute inference
attacks just imputation? InProceedings of The ACM Con-
ference on Computer and Communications Security (CCS),
2022. 3, 5
[16] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su,
Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and
Pascale Fung. Survey of hallucination in natural language
generation.ACM computing surveys, 55(12):1–38, 2023. 2
[17] Yushan Jiang, Wenchao Yu, Geon Lee, Dongjin Song, Kijung
Shin, Wei Cheng, Yanchi Liu, and Haifeng Chen. Timexl:Explainable multi-modal time series prediction with llm-in-
the-loop.arXiv preprint arXiv:2503.01013, 2025. 2
[18] Myeongseob Ko, Jihyun Jeong, Sumiran Singh Thakur,
Gyuhak Kim, and Ruoxi Jia. From weak cues to real iden-
tities: Evaluating inference-driven de-anonymization in llm
agents.arXiv preprint arXiv:2603.18382, 2026. 5
[19] Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-
Antoine Gourraud, Mickael Rouvier, and Richard Dufour.
Biomistral: A collection of open-source pretrained large lan-
guage models for medical domains, 2024. 5
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike
Lewis, Wen tau Yih, Tim Rockt ¨aschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Information
Processing Systems (NeurIPS), 2020. 2, 6
[21] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama,
Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon,
and Jianfeng Gao. Llava-med: Training a large language-and-
vision assistant for biomedicine in one day.arXiv preprint
arXiv:2306.00890, 2023. 5
[22] Ninghui Li, Tiancheng Li, and Suresh Venkatasubramanian.
t-closeness: Privacy beyond k-anonymity and l-diversity. In
Proceedings of the 23rd International Conference on Data
Engineering (ICDE), pages 106–115, 2007. 3
[23] Ashwin Machanavajjhala, Johannes Gehrke, Daniel Kifer,
and Muthuramakrishnan Venkitasubramaniam. l-diversity:
Privacy beyond k-anonymity.ACM Transactions on Knowl-
edge Discovery from Data, 1(1):3:1–3:52, 2007. 3
[24] Lars Malmqvist. Sycophancy in large language models:
Causes and mitigations, 2024. 2
[25] Yoojin Nam, Dong Yeong Kim, Sunggu Kyung, Jinyoung
Seo, Jeong Min Song, Jimin Kwon, Jihyun Kim, Wooyoung
Jo, Hyungbin Park, Jimin Sung, et al. Multimodal large
language models in medical imaging: current state and future
directions.Korean Journal of Radiology, 26(10):900, 2025. 2
[26] Meike Nauta, Ron van Bree, and Christin Seifert. Neural pro-
totype trees for interpretable fine-grained image recognition.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2021. 2
[27] Meike Nauta, Ron van Bree, and Christin Seifert. Pip-net:
Patch-based intuitive prototypes for interpretable image clas-
sification. InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2023. 2
[28] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
Ragtruth: A hallucination corpus for developing trustworthy
retrieval-augmented language models. InProceedings of the
62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10862–10878,
2024. 2
[29] Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghu-
nathan, Kunal Talwar, and ´Ulfar Erlingsson. Scalable private
learning with pate, 2018. 3
[30] Alvaro Lopez Pellicer, Plamen Angelov, and Neeraj Suri. Se-
curing (vision-based) autonomous systems: taxonomy, chal-
lenges, and defense mechanisms against adversarial threats.
Artificial Intelligence Review, 58(12):373, 2025. 3

[31] Alvaro Lopez Pellicer, Andre Mariucci, Plamen Angelov,
Marwan Bukhari, and Jemma G. Kerns. Protomedx: Towards
explainable multi-modal prototype learning for bone health
classification. InProceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV) Workshops, pages
7357–7366, 2025. 1, 2, 3, 5
[32] Tal Ridnik, Dedy Kredo, and Itamar Friedman. Code gener-
ation with alphacodium: From prompt engineering to flow
engineering.arXiv preprint arXiv:2401.08500, 2024. 2
[33] Isaac Rogers. Jsonformer: Generate structured json from
language models. GitHub repository, 2023. 2
[34] Aswin Rrv, Nemika Tyagi, Md Nayem Uddin, Neeraj Varsh-
ney, and Chitta Baral. Chaos with keywords: Exposing large
language models sycophancy to misleading keywords and
evaluating defense strategies. InFindings of the Associa-
tion for Computational Linguistics: ACL 2024, pages 12717–
12733. Association for Computational Linguistics, 2024. 2
[35] Cynthia Rudin. Stop explaining black box machine learning
models for high stakes decisions and use interpretable models
instead.Nature Machine Intelligence, 1:206–215, 2019. 2
[36] Mrinank Sharma, Meg Tong, Tomasz Korbak, David Du-
venaud, Amanda Askell, Samuel R. Bowman, Newton
Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. John-
ston, Shauna Kravec, Timothy Maxwell, Sam McCandlish,
Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan,
Miranda Zhang, and Ethan Perez. Towards understanding
sycophancy in language models. arXiv:2310.13548, 2023. 2
[37] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik
Narasimhan, and Shunyu Yao. Reflexion: Language
agents with verbal reinforcement learning.arXiv preprint
arXiv:2303.11366, 2023. 2
[38] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly
Shmatikov. Membership inference attacks against machine
learning models. In2017 IEEE Symposium on Security and
Privacy (SP), pages 3–18. IEEE, 2017. 3, 5
[39] Connor Shorten, Charles Pierse, Thomas Benjamin Smith,
Erika Cardenas, Akanksha Sharma, John Trengrove, and Bob
van Luijt. Structuredrag: Json response formatting with large
language models. arXiv:2408.11061, 2024. 2
[40] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi,
Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani,
Heather Cole-Lewis, Stephen Pfohl, et al. Large language
models encode clinical knowledge.Nature, 620(7972):172–
180, 2023. 2
[41] Latanya Sweeney. k-anonymity: A model for protecting
privacy.International Journal on Uncertainty, Fuzziness and
Knowledge-based Systems, 10(5):557–570, 2002. 3
[42] Jonas Wallat, Maria Heuss, Maarten de Rijke, and Avishek
Anand. Correctness is not faithfulness in retrieval augmented
generation attributions. InProceedings of the 2025 Interna-
tional ACM SIGIR Conference on Innovative Concepts and
Theories in Information Retrieval (ICTIR ’25), 2025. 2
[43] Brandon T. Willard and R ´emi Louf. Efficient guided genera-
tion for large language models. arXiv:2307.09702, 2023. 2,
5
[44] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and
Weidi Xie. Towards generalist foundation model for radiologyby leveraging web-scale 2d&3d medical data.arXiv preprint
arXiv:2308.02463, 2023. 5
[45] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran,
Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing
reasoning and acting in language models.arXiv preprint
arXiv:2210.03629, 2022. 2