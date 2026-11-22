# Some Memories Must Fade: Persistence and Decay of User Memories in Model Parameters

**Authors**: Sanidhya Vijayvargiya, Rahul Lokesh

**Published**: 2026-11-18 22:08:51

**PDF URL**: [https://arxiv.org/pdf/2611.25235v1](https://arxiv.org/pdf/2611.25235v1)

## Abstract
Parametric memory allows Large Language Models to internalize knowledge directly into their weights, offering a potentially powerful alternative to retrieval-based memory (RAG). However, existing approaches to parametric memory are difficult to deploy as they largely mirror the Continual Learning (CL) objective of retaining information indefinitely. This is neither sustainable nor sufficient for a memory system, which must govern knowledge across its lifecycle by retaining stable facts, reinforcing recurring observations, updating changed beliefs, and allowing transient information to fade. We propose a system for parametric memory management consisting of a memory substrate (handling information storage and forgetting) and a learned policy (what to remember, how deeply, and for how long). Our substrate is a fixed-capacity LoRA adapter that consists of k-hot rank combinations which store facts without catastrophic interference and maintain an independently controlled temporal decay, enabling outdated information to fade while long-term facts persist. On this substrate, we train a lightweight controller that governs each incoming user event to ensure the memory system reflects the user's true, updated belief state. To evaluate how memory systems manage user memory, we introduce a controlled dataset of long-horizon user trajectories with varying persistence of facts based on changing belief states and user-behavioral shifts, and probe queries that test recall while information is valid and abstention after it expires. Our policy learns to effectively manage user memory over its entire lifecycle, balancing recall and abstention to achieve 86% of oracle performance, unlike CL and RAG baselines that primarily default to one of the two behaviors. By providing the first system to manage memory directly in model weights, this work establishes a foundation for long-horizon parametric memory.

## Full Text


<!-- PDF content starts -->

SOMEMEMORIESMUSTFADE: PERSISTENCE AND
DECAY OFUSERMEMORIES INMODELPARAMETERS
Sanidhya VijayvargiyaandRahul Lokesh
Samsung Research America
ABSTRACT
Parametric memory allows Large Language Models (LLMs) to internalize knowl-
edge directly into their weights, offering a potentially powerful alternative to
retrieval-based memory (RAG). However, existing approaches to parametric
memory are difficult to deploy as they largely mirror the Continual Learning (CL)
objective of retaining information indefinitely. This is neither sustainable nor suf-
ficient for a memory system, which must govern knowledge across its lifecycle
by retaining stable facts, reinforcing recurring observations, updating changed be-
liefs, and allowing transient information to fade. We propose a system for para-
metric memory management consisting of a memory substrate (handling informa-
tion storage and forgetting) and a learned policy (what to remember, how deeply,
and for how long). Our substrate is a fixed-capacity LoRA adapter that stores
memories in sparsek-hot combinations of adapter ranks to limit interference.
This allows each rank and its stored information to maintain an independently
controlled temporal decay, enabling outdated information to fade while long-term
facts persist. On this substrate, we train a lightweight 2M-parameter controller
that governs each incoming user event to ensure the memory system reflects the
user’s true, updated belief state. To evaluate how memory systems manage user
memory, we introduce a controlled dataset of long-horizon user trajectories with
varying persistence of facts based on changing belief states and user-behavioral
shifts, with probe queries that test recall while information is valid and absten-
tion after it expires. Our policy learns to effectively manage user memory over
its entire lifecycle, balancing recall and abstention to achieve 86% of oracle per-
formance, unlike CL and RAG baselines that primarily default to one of the two
behaviors. By providing the first system to manage memory directly in model
weights, this work establishes a foundation for long-horizon parametric memory.
1 INTRODUCTION
Large Language Models (LLMs) increasingly operate over long horizons as agents and personalized
assistants, necessitating memory systems to match evolving user preferences (Ding et al., 2026;
Liu et al., 2025). Currently, retrieval-based memory is widely adopted for its simplicity—storing
information externally and injecting it into the context when relevant (Lewis et al., 2021; Kang
et al., 2025). Parametric memory instead encodes information directly into the model’s weights,
allowing LLMs to leverage memory implicitly without cluttering the prompt context (Wang et al.,
2025b; Behrouz et al., 2024; Su et al., 2025). However, this approach remains difficult to deploy in
practice due to its rigidity in adapting to the evolving nature of user memory. Current approaches
largely inherit the objective of continual learning (CL), focusing on how to acquire new knowledge
without forgetting (McCloskey & Cohen, 1989; Kirkpatrick et al., 2017). We demonstrate that
this objective is insufficient for long-horizon assistants. Memory management instead asks which
knowledge should be retained, reinforced, overwritten, or forgotten, taking downstream use into
account. For instance,I had a bagel for breakfastshould not remain valid even the next day, while
I work as a teachershould likely be relevant information for years. External stores expose lifecycle
operations directly through addition, update, and deletion, but model parameters do not (Yan et al.,
2026; Yu et al., 2026). Consequently, in our evaluation, no CL method is able to represent outdated
information even though they store values well (0.82 retrieval with PCGrad (Yu et al., 2020)).
We hypothesize that usable parametric memory requires separating the storage dynamics (how infor-
mation is stored, and isolated in the substrate) from its policy (what to store, how deeply, and for how
1
arXiv:2611.25235v1  [cs.LG]  18 Nov 2026

long). For the substrate dynamics, we design a fixed-size LoRA substrate (Hu et al., 2021) where
each fact is routed to ak-hot combination of adapter ranks (Zou et al., 2025). This bounds gradient
interference between unrelated facts, while an independent temporal decay rate on each rank allows
stale information to fade (Ebbinghaus, 1885) without explicit unlearning. For the policy, we train
a lightweight 2M-parameter controller that decides, per incoming event, whether to write, at what
gradient depth, and with what lifespan. Stable information receives deep durable writes, repeated
routines see accumulating writes, while transient events see short or no lifespans.
Evaluating these lifecycle decisions requires ground truth that existing memory benchmarks do not
provide at the event level. We therefore construct909controlled, long-horizon user trajectories with
lifecycle annotations capturing validity, persistence, reinforcement, and contradiction, along with
programmatically injected queries that probe per-event management decisions. On this testbed,
our learned management system reaches0.56class-balanced accuracy, compared with0.42for the
strongest CL baseline (PCGrad) and0.50for a compute-matched retrieval-based memory manager,
recovering 86% of oracle performance. We study two questions: (1) Can a fixed-capacity para-
metric substrate support reliable information persistence and decay with varying per-fact dynamics?
(2) Given such a substrate, can a policy infer from online observations how each fact should be
managed? Our contributions are:
•Task and controlled benchmark(§2): We formalize parametric memory management around
four requirements—valid retention, selective expiration, reinforcement and contradiction, and
user-specific dynamics—and construct a dataset of long-horizon user trajectories with event-level
lifecycle supervision and diagnostic probes.
•Memory substrate(§3): We propose a fixed-capacity LoRA substrate combining sparsek-hot
rank allocation with per-rank temporal decay, providing controllable mechanisms for selective
persistence, reinforcement, overwrite, and calibrated expiration.
•Learned management(§4): We define a(binary gate,depth,lifespan)action space over the sub-
strate and train a 2M-parameter controller that recovers 86-92% of oracle performance, demon-
strating that effective lifecycle decisions can be inferred from the available online signals.
2 TASK: PARAMETRICMEMORYMANAGEMENT
2.1 MEMORYMANAGEMENTREQUIREMENTS
We study parametric memory management for lifelong user-facing agents, where rather than pre-
serving all written information, the system must manage each fact’s lifecycle as user knowledge
evolves. We identify four requirements:
1.Valid retention:Recall stored information while it remains valid.
2.Selective expiration:Once information is no longer valid, abstain.
3.Reinforcement and contradiction:Repeated observations should strengthen an existing belief,
while contradictory observations should weaken or overwrite it.
4.User-specific dynamics:Persistence and decay must be inferred from individual history, as the
same type of information can be long-term for one user but transient for another.
Success requires controlling when and how individual memories persist, accumulate, change, or ex-
pire. Existing long-term memory benchmarks evaluate several related capabilities, including recall
and belief revision, but none evaluate these lifecycle requirements at the event level (Table 1).
2.2 CONTROLLEDLONG-HORIZONTRAJECTORIES
Evaluating these requirements requires ground truth for each fact at different stages in its lifecy-
cle. Real long-term user histories generally do not provide this supervision. Privacy constraints
limit collection, management events such as contradictions and temporary shifts are sparse, and the
true validity of individual memories is generally unknown even when long interaction histories are
available (Uddin et al., 2026).
We therefore generate a benchmark of 909 controlled, multi-year user trajectories (spanning
700–1,500 days) with dense event-level supervision from Gemini 3.7 Flash. Each trajectory is con-
structed via a five-stage pipeline: (1) sampling a behavioral motif (e.g., frequent routine changes)
2

Belief∆Expiry Reinforcement User-dynamics Lifecycle GT
ForgetBench (Gu et al., 2026)G #✗ ✗ ✗ ✗
LoCoMo (Maharana et al., 2024)G #✗ ✗ ✗ ✗
PersonaMem (Jiang et al., 2025)✓ ✗ ✗G # G #
MemoryAgentBench (Hu et al., 2026)✓ ✗ ✗ ✗ ✗
LongMemEval (Wu et al., 2025)✓ ✗ ✗ ✗G #
Memora (Uddin et al., 2026)✓ ✗G # G #✓
Ours✓ ✓ ✓ ✓ ✓
Table 1: Coverage of memory-management properties (✓full,G #partial,✗absent). Ours evaluates
belief revision, time-based expiry, reinforcement, and user-conditioned lifespans while providing
dense lifecycle supervision. Further details are in Appendix C.4.
and a persona, (2) generating a multi-year backstory, (3) converting the backstory into a chronologi-
cal event stream, (4) masking entities and injecting ground-truth labels for lifespan and stability, and
(5) validating chronological consistency.
100101102103104
Base Decay Days (Log Scale)0.00.20.40.60.81.0Stability Score
Class
CONTESTED
STABLE
ONESHOT
ROUTINE
TRANSIENT
Figure 1: Distribution of the five memory classes
by ground-truth stability score and validity hori-
zon.To prevent the model from relying on pretrained
world knowledge, we replace concrete entities
with typed placeholders inside fixed templates
(e.g., replacing ”apple” withFRUIT 4in the
templateMy favorite fruit is [MASK]). Both
writes and queries use this templated form as
input in our main experiments and the model
must output the valid value that should be in
place of [MASK] or abstain if expired. By
stripping away real-world priors, we force the
controller to learn generalizable patterns that
are not entity-specific or data generation arti-
facts (not memorizing typical lifespans for enti-
ties). The event-level annotations support both
evaluation and training. They provide dense su-
pervision over memory validity, lifespans, re-
inforcement, and contradiction, enabling super-
vised or imitation learning, dense reward con-
struction, and end-to-end reinforcement learn-
ing for memory management (Appendix C).1
2.3 MEMORYDYNAMICS
The benchmark requirements are informed by memory behaviors emphasized across prior work
on long-term memory and personalization (Maharana et al., 2024; Wu et al., 2025; Jiang et al.,
2025; Uddin et al., 2026; McClelland et al., 1995; Park & Bak, 2024). Following generation, the
facts are grouped into five management classes based on their annotated validity horizon, stability,
reinforcement history, and contradictions:
1.STABLE (26%):Long-lived (90–3650 days), typically introduced once (e.g., profession). Re-
quires durable retention over long horizons.
2.ROUTINE (42%):Short-lived (2–21 days) observations that repeatedly recur with the same
value. Requires evidence to accumulate rather than treating each observation independently.
3.CONTESTED (20%):Information whose value is revised over time (e.g., current project).
Requires updating the belief state to the latest value and superseding previous state.
4.TRANSIENT (7%):Very short-lived (1–10 days), unrepeated information (e.g., today’s lunch).
Requires expiration and eventual abstention even without a subsequent contradiction.
5.ONESHOT (4%):Unrepeated information with an intermediate validity horizon (15–90 days).
Requires temporary retention without either immediate loss or indefinite lifespans.
1We also provide naturalistic, non-templated versions of all inputs for evaluating retrieval-based methods.
§4.7 runs the full learned system on these natural-language (NL) inputs.
3

1Incoming
event
e.g.
“my
gym
is
FitLab”
Event embedding
…
et ∈ ℝd2Router
Select a sparse k-hot set of ranks
Router
r(et )
hash
(PMM-PR)
or
learned
(PMM-LR)Selected ranks
{r2, r5}
(k = 2)3LoRA
memory
substrateΔW = A × B
A
(frozen)
orthonorm al
init
A ∈ ℝd×R×B
(per-rank lifetime τ), R = 32 ranks
r1τ2
r2 r3 r4τ5
r5 r6 r7 r8…decay between events: Bᵢ ← Bᵢ · exp(−Δt / τᵢ)4Controller
Decide write behavior
Inputs
Event semantics et
User history ht
Memory state st
Controller
c(et, ht, st)
Gate
g ∈ [0, 1]Depth
d (# SGD steps)Lifetime
τ (days)
write
/
update
the
selected
ranksFigure 2:Overview of the parametric memory management architecture.An incoming user
memory is mapped by the router to its storage in the LoRA adapter, consisting of ak-hot set of
ranks in the up-projection (B) matrix. Based on the available online features including features from
these ranks, the controller decides whether and how the information should be memorized.
2.4 QUERYEVALUATION
We evaluate management using queries placed at controlled points in each fact’s lifecycle. Are-
trieval query (VAL)occurs while the fact is valid and is scored on recovering the current value. An
abstention query (ABS)occurs after expiry and is scored on answeringUnknown. For CONTESTED
facts, queries target the currently valid value. Repeated ROUTINE events test whether consistent
observations accumulate into stronger retention. From Figure 1, we can see that some classes have
distinctive ground-truth patterns (STABLE vs TRANSIENT) but others can be overlapping (ROU-
TINE vs CONTESTED vs TRANSIENT). A valid management policy must therefore identify the
user-level signals that segregate overlapping classes. Per-fact query budgets are reweighted to en-
sure that important but less frequent behaviors, such as expiration, receive sufficient evaluation mass
despite write imbalances (Appendix C). The resulting query distribution is therefore designed as a
controlled stress test of memory-management capabilities rather than an estimate of natural query
frequency. Each class isolates a distinct challenge: long-horizon retention (STABLE), reinforcement
accumulation (ROUTINE), belief revision (CONTESTED), principled expiration (TRANSIENT),
and intermediate-horizon retention (ONESHOT). We report class-balanced (CB) accuracy so that
performance is not dominated by the most frequent lifecycle class.
3 MEMORYSUBSTRATE
Prior work shows that LoRA adapters can serve as compact parametric stores for information (Xu
et al., 2026; Back et al., 2026). Standard LoRA, however, is not well suited to lifelong memory
management as consecutive writes update the same shared parameter space, causing interference
between unrelated memories, while information written once has no mechanism to expire selec-
tively (Yang et al., 2024). We therefore design a fixed-capacity substrate on a Qwen 3 1.7B backbone
with three properties required for parametric memory management in the attached LoRA adapter:
isolated writes,controllable lifespans, andcalibrated expiration. Together, these provide the
memory dynamics over which the management policy in §4 operates (Figure 2).
3.1 SPARSECOMBINATORIALMEMORYLOCATIONS
We treat theRrank columns of the adapter’sBmatrix as an addressable memory bank. Each
write is routed to ak-hot subset of columns while all others remain frozen and retrieval loads the
same subset (Zou et al., 2025). TheAmatrix is orthonormally initialized and frozen throughout, as
freezingAyields a more predictable, approximately proportional relationship between write depth
and memorization strength (Zhang et al., 2026). WithR= 32andk= 2as our main configuration,
this yields 32
2
= 496addressable locations from32shared basis columns. Largerkraises recall
at the expense of abstention (Appendix A.1). This value-abstain trade-off dictates the limits of the
management policies analyzed in §4.
This routing provides two core properties. First,interference is boundedas writes to disjoint masks
share no gradient path, and contention scales with mask overlap|m 1∩m 2|only. Second,repeated
writes accumulateas five writes at depthd= 20(20 gradient descent steps at fixed learning
rate) achieve comparable retention to a singled= 100write (Appendix A.2), allowing policies
to reinforce facts via incremental updates. A key component of this architecture is the router that
assigns related facts and queries to the same k-hot ranks for reliable updates and retrieval. We
4

evaluate two routers. Enabled by our templated setting, theperfect routerdeterministically hashes
templates to masks, upper-bounding routing. Thelearned routerassigns slots using cosine similarity
on the fact’s whitened embedding (Su et al., 2021) with existing slots, allocating new masks when
similarity is low (Appendix A). Routing is a separable, independently tuned module. By evaluating
both perfect and learned variants, we decouple the impact of storage from management.
3.2 CONTROLLABLETEMPORALLIFESPANS
To support selective expiration, each rank columnicarries an independent decay constantτ i. Be-
tween events separated by∆tdays,B[:, i]←B[:, i]·exp(−∆t/τ i). Each write overwritesτ ifor
its columns, letting the policy assign different lifespan horizons to different information. We dis-
cretizeτinto five temporal scales using empirical decay timelines, from short-lived to effectively
permanent (values in §4.1). Per-location decay is structurally necessary because stable and transient
information coexist in the same fixed-capacity adapter. A single global rate either expires stable facts
prematurely or preserves transient ones indefinitely (validated in §4.5, where a globalτreduces ab-
stention on all volatile classes). Independent decay makes forgetting a controllable operation where
information fades and its capacity is reclaimed without an explicit unlearning step. However, when
two facts are routed to overlapping columns, the older fact’s decay rate is overwritten by the newer
fact’s decay rate. We analyze the impact of this shared-column limitation in §4.6.
3.3 CALIBRATEDRETRIEVAL ANDABSTENTION
A memory substrate must also represent absence. A weak or expired trace should produce abstention
rather than a hallucinated value. Base models tend to fabricate values from weak parametric traces,
and prior work treats this behavior as an intrinsic threshold of parametric memorization (Xu et al.,
2026). We show that abstention from weak traces is instead a resolvable, calibration problem.
A separate rank-32retrieval adapteroperates jointly with the memory adapter during inference,
translating memory traces into expected outputs. Instead of memorizing facts itself, the retrieval
adapter is trained to read the underlying substrate’s state using depth-conditioned labels. If a routed
slot is empty, it is trained to outputUnknown. If a fact was written to the memory adapter at depth
d, the retrieval adapter’s target isUnknownwith label probabilityp unk(d)and the correct value
otherwise (e.g.p unk(20) = 0.9andp unk(150) = 0).
The shape of the monotonically decreasing curvep unk(d)(Appendix A.3) governs behavior under
weak writes. A curve that drops too soon may retrieve hallucinated values from weak traces, while
one that drops too late may collapse all queries to abstention. This probabilistic mapping yields a
graded storage mechanism, rather than a binary retrievable threshold, giving the controller meaning-
ful control over retention strength viad. Crucially, because the retrieval adapter uses the state of the
substrate to generate answers rather than the input, retrieval is separated from memory management.
The same substrate can be paired with alternative adapters, such as one trained for free-form natural
language instead of our templates, without altering the underlying mechanics. (§4.7, Appendix D).
3.4 WHYCONTINUALLEARNINGISINSUFFICIENT
Method CB↑V AL↑ABS↑
Naive LoRA 0.23 0.51 0.00
EWC 0.22 0.50 0.00
PCGrad 0.420.820.00
Ours (PMM-PR)0.600.680.59
Table 2: CL protects stored information but
cannot represent selective expiration. Per-class
breakdowns are in the Appendix A.4.Do these properties emerge from applying stan-
dard continual learning (CL) to a shared LoRA
store? We test EWC (Kirkpatrick et al., 2017)
and PCGrad with sequential fixed-depth LoRA
writes, selectingd= 15(optimal selection from
d= 10,15,20,40,60). Both reduce interfer-
ence by protecting previously written information
but provide no mechanism for selective expira-
tion. Consequently, every CL baseline achieves
0.00ABS despite PCGrad reaching0.82V AL
(Table 2). Adding uniform exponential decay
changes accuracy by<0.01, since dense shared
parameters cannot assign different lifespans to different memories. Our substrate resolves this struc-
turally as specific ranks decay to physically represent forgetting, which trigger the retrieval adapter
to abstain. Sparse locations, independent decay, and calibrated retrieval provide this control.
5

0Train
the
retrieval
adapter once, oﬄine
Layer
28
Layer
1frozen
LLM∇
Retrieval
adapterMemory
adapter
synthetic
queryTARGET
value,
or
“Unknown”
Synthetic
writes
random facts, random depth d1Write
a
fact per user, as facts arrive
Layer
28
Layer
1frozen
LLM
Retrieval
adapterMemory
adapter
gym
=
FitLabTARGET
FitLab
Memory
manager
picks slots m, depth d, lifetime τ2Read per query
Layer
28
Layer
1frozen
LLM
Retrieval
adapterMemory
adapter
my
gym?ANSWER
FitLab
/
“Unknown”
Memory
manager
picks slots m; nothing is written
∇ trained / updated by this step (dash ed: gradient) frozen slots of this fact other slots (faded by time) hidden at read Per user, only the memory adapter and τ are stored (11 MB).Figure 3: End-to-end workflow of parametric memory management (PMM)
4 LEARNEDMEMORYMANAGEMENT
The substrate defines the storage of parametric memory. Here, we investigate (1) what actions are
required to manage this substrate, (2) whether the information available online is sufficient to infer
effective management decisions, and (3) what trade-offs remain even under effective management
due to the substrate’s finite capacity.
4.1 MEMORYMANAGEMENTACTIONS
For each incoming event, the controller emits an action(g, d, τ):
•Gate(g∈0,1): whether to write the event at all. Declining leaves the substrate unchanged.
•Depth(d∈0,20,30,40,60,80,110,150SGD steps): how strongly to write. Higherdyields
stronger retention but a larger gradient footprint on shared mask columns.
•Lifespan(τ∈25,100,400,1600,20000): the decay timescale assigned to the routed columns,
controlling how long the resulting trace persists.
The three actions are predicted by independent discretized heads, allowing each to be conditioned
and bias-initialized separately (§4.4). Discretization also constrains the policy search space and sim-
plifies optimization. We find two design choices to be important. First, the action space distinguishes
rejectinginformation (g= 0) frombriefly retainingit through a shallow write with short lifespan.
Different volatile events may require either behavior, motivating independent gate and lifespan con-
trols. Second, routing is separated from the management policy. The router in Section 3.1 determines
which ranks an event is written in, while the controller determines how/whether it is stored through
gate, depth, and lifespan. We evaluate both perfect and learned routing to separate allocation errors
from management errors.
4.2 STATE ANDPOLICYARCHITECTURE
The 2M-parameter controller (<0.4%of the base model) conditions on three sources of online
information: (1) theincoming eventrepresented by its semantic embedding, (2)trajectory history
represented by a GRU state (Cho et al., 2014), and (3)substrate state, consisting of per-rank ages
(time since last write for each rank), per-rank norms (to capture stored information), and token-
level prediction signals from a forward pass that indicate whether the incoming event reinforces or
contradicts the existing memory. A shared neural trunk feeds independent decision heads for gate,
depth, and lifespan (Appendix B.4). Persistence cannot be determined from event content alone.
Because identical facts exhibit different stabilities across users, relying solely on semantics leads to
a prediction ceiling of onlyR2= 0.36(Appendix B.13). The controller addresses this ambiguity by
grounding its decisions in the user’s trajectory history. Finally, to stabilize early training, the action
heads are initialized with empirical biases derived from an expert policy.
4.3 PIPELINEWORKFLOW
Components.The system is composed of a frozen Qwen3-1.7B backbone, two additive LoRA
adapters (Memory and Retrieval), and two management components (a router and a controller).
Both adapters are attached toq projandv projin every layer, so a layer’s output ish=Wx+
BreadAreadx+B memAmemx. The base model is never updated in the process (Figure 3).
6

Retrieval-adapter training (before deployment).We simulate writes of random facts at random
depths and train the retrieval adapter to output either the stored value orUnknown, according to the
depth-conditioned label probabilityp unk(d)(§3.3). It is then frozen for all subsequent steps.
Write (per incoming fact).The router maps the fact to an existing rank mask, or allocates a new
one. The mask is specific to a fact family and shared across all layers (we usek= 2columns per
family). The controller reads the fact embedding, the memory-adapter state, and the user history
(through its GRU), and outputs whether to write the fact, a depthd∈[20,150], and a lifetime
τ∈[25,20000]days. If written to the Memory adapter, the write isdSGD steps on the selected
columns ofB memonly, and those columns take decay rateτ.A memis orthonormal and frozen
throughout. The retrieval adapter is not active during writes.
Decay (between events).Each column decays asB i←B iexp(−∆t/τ i), making a fact expire.
Read (per query).The router selects the query’s mask, and ranks outside it are zeroed. The Memory
and Retrieval adapters are both active, and the Retrieval adapter decides from the memory state
whether to emit the stored value orUnknown. The controller is not used.
4.4 LEARNING FROM AMETADATAORACLE
We train the controller by imitation from a manually configuredcausal metadata oracle. The oracle
observes the same online history as the controller together with ground-truth lifecycle metadata
(stability, time to decay, and reinforcement status) and computes a target write depth based on a
deficit. The deficit is the estimated depth required for a fact to remain retrievable over its target
horizon, minus any depth already accumulated through prior writes, capped at 150. The oracle has
no access to future events, but receives lifecycle metadata that the learned controller must infer from
its observable state features. Crucially, this supervision adapts to the student’s actual substrate state
under DAgger (Ross et al., 2011). When the student deviates from the oracle, the target depth is
recomputed from what the student has actually written rather than assuming ideal previous actions.
Replacing this state-aware supervision with fixed target depths derived solely from ground-truth
metadata reduces class-balanced accuracy from0.56to0.49(Appendix B.7).
Training.The training pipeline has three stages: (A) stability-head MSE training, (B) behavior
cloning against the oracle via DAgger, and (C) optional PPO refinement (Schulman et al., 2017) (Ap-
pendix B.15). Stage B reaches near-oracle performance and is our primary configuration. We com-
pare our parametric memory management (PMM) controllers trained with perfect routing (PMM-
PR) and learned routing (PMM-LR) to separate the learned policy impact from allocation quality.
4.5 RESULTS
We evaluate on15held-out user trajectories (∼600events per user) with sequential LoRA writes
and queries. Results are reported per lifecycle class and per query type (V AL, ABS), aggregated
using class-balanced (CB) accuracy. Table 3 answers three questions. First, can conventional para-
metric memorization solve the lifecycle problem? No, CL baselines and unmanaged-write on our
substrate retrieve values well but fail to expire outdated information. Second, is lifecycle manage-
ment learnable? Yes, PMM-LR recovers 86% of the metadata-oracle ceiling. Third, what limits
performance? The fully learned PMM-LR reaches0.56CB accuracy, while perfect routing raises
performance to0.60against a metadata-oracle ceiling of0.65. This hierarchy separates distinct
sources of error. The PMM-LR-to-PMM-PR gap reflects semantic allocation, the PMM-PR-to-
oracle gap primarily reflects lifecycle inference, and the oracle ceiling reflects limitations of the
substrate itself, including shared-column interference between distinct memories andτassignment
(analyzed below). Improving routing can close the first gap, richer online management signals can
close the second, and changes to allocation or capacity are required to raise the substrate ceiling.
Both controllers outperform all CL and unmanaged-write baselines, as well as the retrieval-based
memory managers tested at compute parity. PMM-LR reaches0.56versus0.50for compute-
matched RAG-MM and also exceeds RAG-MM with a substantially larger 122B management model
(0.53). A trained manager in RAG-MM matches performance, though it often struggles to synthe-
size correct answers even with improved management. Retrieval surpasses our controllers when its
management pass is replaced with ground-truth lifecycle information (0.64at1.7B and0.82at122B
§4.6), highlighting the importance of the management signal independent of the underlying memory
substrate. Once information is written, queries require no retrieval over an external history and no
context growth with trajectory length.
7

System CB STABLE ROUTINE CONTESTED TRANSIENT ONESHOT
Metadata oracle0.65±.03 0.56±.05 0.72±.06 0.51±.07 0.77±.05 0.68±.05
PCGrad (best CL,d= 15)0.42±.02 0.60±.05 0.66±.05 0.41±.05 0.13±.05 0.31±.05
RAG (1.7B)0.50±.02 0.30±.05 0.42±.05 0.55±.05 0.75±.05 0.49±.06
RAG-MM (1.7B)0.50±.02 0.41±.05 0.45±.05 0.48±.06 0.68±.05 0.47±.05
RAG-MM (122B)0.53±.02 0.47±.04 0.69±.05 0.47±.05 0.33±.06 0.67±.06
RAG-MM (1.7B, trained)0.55±.03 0.38±.06 0.48±.05 0.54±.07 0.77±.05 0.57±.05
Unmanagedd= 20 0.45±.02 0.56±.05 0.63±.05 0.43±.06 0.26±.05 0.38±.05
Unmanagedd= 80 0.50±.02 0.65±.05 0.69±.05 0.51±.06 0.11±.05 0.54±.05
PMM-LR0.56±.020.40±.05 0.76±.05 0.56±.05 0.44±.06 0.65±.07
PMM-PR0.60±.02 0.39±.05 0.75±.05 0.57±.06 0.59±.06 0.68±.06
Metadata oracle (NL)0.67±.02 0.59±.05 0.78±.05 0.57±.05 0.52±.06 0.87±.05
PMM-PR (NL)0.60±.03 0.45±.05 0.78±.05 0.58±.05 0.47±.06 0.73±.07
PMM-LR (NL)0.54±.03 0.47±.05 0.79±.05 0.55±.05 0.34±.05 0.55±.08
No per-rankτ(PMM-PR)0.48±.02 0.57±.05 0.70±.05 0.44±.05 0.24±.06 0.47±.06
Table 3: Evaluation results onn= 15held-out users,4584queries. RAG-MM refers to RAG with a
write-time memory-management architecture (Appendix B.3). Naive/EWC continual-learning vari-
ants (≤0.23CB), oracle-metadata RAG upper bounds, and no-τunmanaged baselines are reported
in Appendix B.6. NL refers to natural-language interfacing of the same trajectories.95%percentile
bootstrap CIs clustered by user (users resampled, 2,000 resamples).
4.6 PER-CLASSTRADE-OFFS ANDFAILUREMODES
0.0 0.2 0.4 0.6 0.8 1.0
VAL (value retrieval accuracy)0.00.20.40.60.8ABS (abstention accuracy)RAG (1.7B)
RAG-MM (1.7B)RAG+oracle (1.7B)
Unmanaged d=20
PCGrad
Unmanaged d=80RAG+oracle (122B)
Metadata oraclePMM-PR (ours)
PMM-LR (ours)RAG-MM (122B)
Compute-parity
Augmented (compute only)Metadata-privileged
Ours
Figure 4: Pareto positions in value-retrieval (V AL) ver-
sus abstention (ABS) accuracy. Four regimes emerge:
(1) continual learning and unmanaged writes at high
V AL and near-zero ABS, (2) compute-parity retrieval
at high ABS and low V AL, (3) augmented retrieval us-
ing greater compute or privileged lifecycle information
reaching the top-right, and (4) our controllers occupy-
ing a balanced regime that no compute-parity baseline
reaches. Breakdown in Table 12.Memorization without management
fails at abstention.Fixed-depth writes
demonstrate a clear trend. Atd= 20,
V AL ranges from0.60-0.90across classes
while ABS remains only0.01-0.22.
Increasing depth tod= 80raises V AL
to0.85-1.00while driving ABS to
≤0.01(per-class V AL/ABS splits in
Appendix B.1). Addingτwithout event-
specific management does not improve
this trade-off (Appendix B.2).
Retrieval-based memory also requires
explicit management.Compute-
matched simple RAG fails due to
frequently defaulting toUnknownunder
ambiguous context (ABS0.75–0.87)
while rarely committing to values (V AL
0.16–0.30on most classes). RAG-MM’s
write-time management pass—resolving
reinforcement, contradiction, and expiry
as events arrive (Appendix B.3)—shifts
this Pareto position. Replacing the learned
management pass with ground-truth
lifecycle information raises RAG-MM
to0.64at1.7B and0.82at122B (Ap-
pendix B.6), providing an upper bound
when management is ideal and not inferred (Figure 4).
Management matters where events require different lifespans.Our controllers occupy a bal-
anced regime that other compute-parity baseline do not reach. For TRANSIENT facts, no unman-
aged fixed-depth policy exceeds0.26accuracy because writing them strongly enough for immediate
retrieval causes them to remain accessible after expiry. PMM-LR and PMM-PR reach0.44and0.59,
respectively, by assigning short lifespans to volatile information. Conversely, on STABLE facts, un-
managedd= 80reaches0.65while PMM reaches0.39–0.40, pairing moderate V AL (0.33–0.34)
with strong ABS (0.78–0.81).
8

Controller Layout CB STABLE V AL STABLE ABS MB/user
Oracle 32 shared .65 .53 .74 11
Oracle 64 shared .70 .73 .58 22
Oracle 32 + 32 dedicated .71 .82 .68 22
PMM-PR 32 shared .60 .33 .81 11
PMM-PR 32 + 32 dedicated .63 .70 .67 22
PMM-LR 32 shared .56 .34 .78 11
PMM-LR 32 + 32, trained with routing .59 .75 .27 22
Table 4: Dedicated columns for long-lived facts improve their retention. MB refers to size of adapter.
The STABLE gap is due toτoverwrites on shared columns.Failures in retrieval of STABLE
facts occur primarily at longer query horizons and under greater foreign-write interference, rather
than from policy failures like declined writes or insufficient initial depth (Appendix B.14). Because
each write overwritesτon the columns it touches, a STABLE fact can inherit the shorter timescale
of a later volatile write sharing the same column over its long horizon. Preserving the maximum
τrecovers∼13points of STABLE V AL but collapses ABS on volatile classes, reducing CB accu-
racy to0.45. This exposes a lifespan-compression trade-off in the shared substrate. Reinforcement
partially mitigates the effect as STABLE facts with3+writes achieve0.68retrieval success as com-
pared to0.51for single-write facts. High STABLE retrieval performance can be achieved by routing
them to non-overlapping ranks, at the cost of compression (Table 4).
Per-rankτenables joint optimization.Removing per-rank lifespan control and forcing a global
τdrops PMM-PR from0.60to0.48. Without the ability to selectively expire information, the
controller maintains retrieval only by sacrificing abstention. V AL remains≥0.79across volatile
classes, while TRANSIENT ABS falls to0.09, ROUTINE ABS to0.06, and CONTESTED ABS to
0.04. This establishes independent lifespan control as a necessary mechanism for supporting both
retention and expiration in this substrate.
Variant CB acc Notable effect
PMM-PR (full)0.60baseline
No history0.58ONESHOT V AL−24
No substrate0.58ROUTINE ABS−9
No priors0.52overall−8pt
No semantic0.47overall−13pt
Table 5: Ablations on controller input features.Ablations.Table 5 reports input-feature ab-
lations for PMM-PR. Removing semantic fea-
tures or configured priors substantially re-
duces aggregate accuracy (0.47and0.52, re-
spectively), confirming that both provide non-
redundant signals during training. History and
substrate state are partially redundant because
both reflect previous observations (through
trajectory or through substrate state impact).
However, class-specific failures highlight the
distinct mechanistic role of each feature. Without trajectory history, ONESHOT V AL drops by
24 points because these first-time events lack a pre-existing trace to provide a redundant signal.
Conversely, without substrate state, ROUTINE ABS drops by 9 points. Blind to how strongly a re-
curring fact is already embedded, the controller over-relies on historical frequency and over-writes
the trace, preventing it from properly expiring. These results show that effective lifecycle man-
agement depends on event semantics, the user’s trajectory, and the current state of memory. Full
per-class results are provided in Appendix B.16.
4.7 NATURALLANGUAGETRANSFER
The experiments above use templated writes and queries to decouple management from memoriza-
tion. Here we run the full learned system on the naturalistic version of the same trajectories, with the
same architecture and training recipe. Natural-language writes behave differently (recall plateaus af-
ter about60SGD steps), so we re-measured write strength and decay once, and refit the controller
targets and the retrieval adapter on these measurements (Appendix D). As seen in Table 3, every
controller lands within0.02CB of its templated counterpart.
9

5 RELATEDWORK
5.1 PARAMETRICMEMORY ANDLIFELONGKNOWLEDGEEDITING
A growing body of work stores information in model parameters or latent states rather than external
text (Sun et al., 2025; Ba et al., 2016). Recent systems divide into latent-state memory, maintaining
pools of hidden-state memory tokens (Wang et al., 2024; 2025a), and parametric memory, writing
into weights through context distillation (Wang et al., 2025b), dedicated memory modules (Berges
et al., 2024), or test-time weight updates (Behrouz et al., 2024). In parallel, lifelong knowledge edit-
ing develops efficient, localized modification of model knowledge (Meng et al., 2023; Hartvigsen
et al., 2023; Yu et al., 2023). Both lines study whether information can be written, retrieved, or
updated while preserving unrelated knowledge. We instead study memory lifecycle management
which explores how strongly information should be stored, how long it should persist, and when
new observations should reinforce or replace existing beliefs.
5.2 SPARSE ANDMODULARPARAMETERUPDATES
Sparse and modular weight updates reduce interference during updates. O-LoRA (Wang et al., 2023)
uses near-orthogonal task subspaces, N-LoRA (Yang et al., 2024), and FlyLoRA (Zou et al., 2025)
allocate sparse adapter ranks to increase capacity and limit overlap. Our substrate similarly uses
sparse combinations of adapter ranks, but as reusable memory locations rather than task-specific
modules. Further, parametric overlap is not uniformly undesirable as unrelated writes should stay
isolated, while overlap between related information supports reinforcement and overwrite. We fur-
ther provide a controlled temporal decay for each rank, making persistence an explicit property of
the memory rather than a byproduct of the memorization.
5.3 MEMORYMANAGEMENT ANDCONTINUALLEARNING
To our knowledge, no prior work attempts to govern the parametric memory lifecycle—retention,
expiration, reinforcement, and overwrite—as a learned management problem. Retrieval-based mem-
ory management systems organize external stores through explicit hierarchies and paging (Packer
et al., 2024; Kang et al., 2025), learned memory operations (Yu et al., 2026; Yan et al., 2026), or
importance- and lifespan-based retention (Park & Bak, 2024), demonstrating that what is retained or
forgotten can be explicitly managed. These managers, however, operate over explicit entries that can
be directly added, updated, or deleted. Our controller instead governs the formation and lifespan of
information encoded in parameters—deciding whether, how strongly, and for how long to write—so
retention and forgetting emerge from the dynamics of the substrate itself.
Continual learning (Zeng et al., 2019; Yu et al., 2020; Saha et al., 2021) addresses the complementary
problem of acquiring new knowledge while minimizing interference with the old. Our experiments
(§3.4) show memorization alone is insufficient for long-horizon memory, where some information
must persist while other information must become inaccessible.
6 CONCLUSION
We present parametric memorization and parametric memory management as distinct problems,
often conflated by prior work. Long-horizon memory requires deciding what to retain, reinforce,
overwrite, or let expire. Existing parametric memory mechanisms do not explicitly expose or control
these lifecycle decisions. Separating memory into its substrate dynamics—a fixed-capacity substrate
with sparse, addressable locations and controllable temporal decay—and its policy—a lightweight
controller governing the substrate through a defined action space—enables the system to balance
retrieval of valid information against abstention on information that should no longer be accessible.
Our controlled environment supports two central findings: (1) a fixed-capacity parametric substrate
can support varying memory lifecycles, and (2) the decisions required to manage those lifecycles
can be inferred from online observations.
Limitations.Moving to a new input format type requires re-measuring write strength and decay
(§4.7, Appendix D). We also assume memory identification, determining what information consti-
tutes a memory event, is given, and our controlled trajectories may not cover all real-world user
dynamics. Dedicated columns for long-lived facts increase per-user memory (§4.6).
Future work.The learned-routing, controller, and oracle gaps motivate better semantic address-
ing, richer write-time signals, and improved capacity allocation. RL over long-term memory utility
10

may further improve the controller, and different types of information like workflows can be stored
in the substrate.
Overall, our results suggest that reliable parametric memory requires progress beyond how models
memorize. It requires controlling how information evolves after it has been written. By showing
that a 2M-parameter policy operating on online signals recovers 86-92% of a metadata-privileged
oracle, we establish parametric memory management as a distinct and learnable problem and provide
a foundation for lifecycle-aware memory in model parameters.
REFERENCES
Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and
extraction, 2024. URLhttps://arxiv.org/abs/2309.14316.
Jimmy Ba, Geoffrey Hinton, V olodymyr Mnih, Joel Z. Leibo, and Catalin Ionescu. Using fast
weights to attend to the recent past, 2016. URLhttps://arxiv.org/abs/1610.06258.
Seungju Back, Dongwoo Lee, Naun Kang, Taehee Lee, S. K. Hong, Youngjune Gwon, and Sungjin
Ahn. Understanding lora as knowledge memory: An empirical analysis, 2026. URLhttps:
//arxiv.org/abs/2603.01097.
Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time, 2024.
URLhttps://arxiv.org/abs/2501.00663.
Vincent-Pierre Berges, Barlas O ˘guz, Daniel Haziza, Wen tau Yih, Luke Zettlemoyer, and Gargi
Ghosh. Memory layers at scale, 2024. URLhttps://arxiv.org/abs/2412.09764.
Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Hol-
ger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder
for statistical machine translation, 2014. URLhttps://arxiv.org/abs/1406.1078.
Tianyu Ding, Aditya Nannapaneni, Bingfan Liu, and Ling Zhang. Always-onagents:a survey of
persistent memory, state, and governance in llmagents, 2026. URLhttps://arxiv.org/
abs/2606.30306.
Hermann Ebbinghaus.Memory: A Contribution to Experimental Psychology. Duncker & Humblot,
Leipzig, Germany, 1885. English translation published 1913 by Teachers College, Columbia
University.
Ruxi Gu, Zhenliang Zhang, and Wei Wang. Forgetbench: Benchmarking forgetting dynamics of
long-term parametric memory in language models, 2026. URLhttps://arxiv.org/abs/
2607.26455.
Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, and Marzyeh Ghassemi.
Aging with grace: Lifelong model editing with discrete key-value adaptors, 2023. URLhttps:
//arxiv.org/abs/2211.11031.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URLhttps:
//arxiv.org/abs/2106.09685.
Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in llm agents via incremental
multi-turn interactions, 2026. URLhttps://arxiv.org/abs/2507.05257.
Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar,
Camillo J. Taylor, and Dan Roth. Know me, respond to me: Benchmarking llms for dynamic
user profiling and personalized responses at scale, 2025. URLhttps://arxiv.org/abs/
2504.14225.
Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory os of ai agent, 2025. URLhttps:
//arxiv.org/abs/2506.06326.
11

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, An-
drei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis
Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic
forgetting in neural networks.Proceedings of the National Academy of Sciences, 114(13):
3521–3526, March 2017. ISSN 1091-6490. doi: 10.1073/pnas.1611835114. URLhttp:
//dx.doi.org/10.1073/pnas.1611835114.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen tau Yih, Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URLhttps:
//arxiv.org/abs/2005.11401.
Jiahong Liu, Zexuan Qiu, Zhongyang Li, Quanyu Dai, Wenhao Yu, Jieming Zhu, Minda Hu,
Menglin Yang, Tat-Seng Chua, and Irwin King. A survey of personalized large language models:
Progress and future directions, 2025. URLhttps://arxiv.org/abs/2502.11528.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei
Fang. Evaluating very long-term conversational memory of llm agents, 2024. URLhttps:
//arxiv.org/abs/2402.17753.
James L McClelland, Bruce L McNaughton, and Randall C O’Reilly. Why there are complementary
learning systems in the hippocampus and neocortex: Insights from the successes and failures of
connectionist models of learning and memory.Psychological Review, 102(3):419–457, 1995.
Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The
sequential learning problem. volume 24 ofPsychology of Learning and Motivation, pp. 109–165.
Academic Press, 1989. doi: https://doi.org/10.1016/S0079-7421(08)60536-8. URLhttps:
//www.sciencedirect.com/science/article/pii/S0079742108605368.
Nick Mecklenburg, Yiyou Lin, Xiaoxiao Li, Daniel Holstein, Leonardo Nunes, Sara Malvar, Bruno
Silva, Ranveer Chandra, Vijay Aski, Pavan Kumar Reddy Yannam, Tolga Aktas, and Todd
Hendry. Injecting new knowledge into large language models via supervised fine-tuning, 2024.
URLhttps://arxiv.org/abs/2404.00213.
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual
associations in gpt, 2023. URLhttps://arxiv.org/abs/2202.05262.
Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha. Fine-tuning or retrieval? com-
paring knowledge injection in LLMs. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen
(eds.),Proceedings of the 2024 Conference on Empirical Methods in Natural Language Pro-
cessing, pp. 237–250, Miami, Florida, USA, November 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.emnlp-main.15. URLhttps://aclanthology.org/
2024.emnlp-main.15/.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E.
Gonzalez. Memgpt: Towards llms as operating systems, 2024. URLhttps://arxiv.org/
abs/2310.08560.
Sangjun Park and JinYeong Bak. Memoria: Resolving fateful forgetting problem through human-
inspired memory architecture, 2024. URLhttps://arxiv.org/abs/2310.03052.
Stephane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation learning and
structured prediction to no-regret online learning, 2011. URLhttps://arxiv.org/abs/
1011.0686.
Gobinda Saha, Isha Garg, and Kaushik Roy. Gradient projection memory for continual learning,
2021. URLhttps://arxiv.org/abs/2103.09762.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms, 2017. URLhttps://arxiv.org/abs/1707.06347.
Jianlin Su, Jiarun Cao, Weijie Liu, and Yangyiwen Ou. Whitening sentence representations for
better semantics and faster retrieval, 2021. URLhttps://arxiv.org/abs/2103.15316.
12

Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye,
Yujia Zhou, and Yiqun Liu. Parametric retrieval augmented generation, 2025. URLhttps:
//arxiv.org/abs/2501.15915.
Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei
Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to
(learn at test time): Rnns with expressive hidden states, 2025. URLhttps://arxiv.org/
abs/2407.04620.
Md Nayem Uddin, Kumar Shubham, Eduardo Blanco, Chitta Baral, and Gengyu Wang. From recall
to forgetting: Benchmarking long-term memory for personalized agents, 2026. URLhttps:
//arxiv.org/abs/2604.20006.
Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and
Xuanjing Huang. Orthogonal subspace learning for language model continual learning, 2023.
URLhttps://arxiv.org/abs/2310.14152.
Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng
Li, Xian Li, Bing Yin, Jingbo Shang, and Julian McAuley. Memoryllm: Towards self-updatable
large language models, 2024. URLhttps://arxiv.org/abs/2402.04624.
Yu Wang, Dmitry Krotov, Yuanzhe Hu, Yifan Gao, Wangchunshu Zhou, Julian McAuley, Dan Gut-
freund, Rogerio Feris, and Zexue He. M+: Extending memoryllm with scalable long-term mem-
ory, 2025a. URLhttps://arxiv.org/abs/2502.00592.
Yu Wang, Xinshuang Liu, Xiusi Chen, Sean O’Brien, Junda Wu, and Julian McAuley. Self-
updatable large language models by integrating context into model parameters, 2025b. URL
https://arxiv.org/abs/2410.00487.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval:
Benchmarking chat assistants on long-term interactive memory, 2025. URLhttps://arxiv.
org/abs/2410.10813.
Ziwen Xu, Haiwen Hong, Linsong Yu, Benglei Cui, Longtao Huang, Hui Xue, and Ningyu Zhang.
How lora remembers? a parametric memory law for llm finetuning, 2026. URLhttps://
arxiv.org/abs/2605.30260.
Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma,
Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Sch ¨utze, V olker Tresp, and Yunpu Ma. Memory-
r1: Enhancing large language model agents to manage and utilize memories via reinforcement
learning, 2026. URLhttps://arxiv.org/abs/2508.19828.
Shuo Yang, Kun-Peng Ning, Yu-Yang Liu, Jia-Yu Yao, Yong-Hong Tian, Yi-Bing Song, and
Li Yuan. Is parameter collision hindering continual learning in llms?, 2024. URLhttps:
//arxiv.org/abs/2410.10179.
Lang Yu, Qin Chen, Jie Zhou, and Liang He. Melo: Enhancing model editing with neuron-indexed
dynamic lora, 2023. URLhttps://arxiv.org/abs/2312.11795.
Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn.
Gradient surgery for multi-task learning, 2020. URLhttps://arxiv.org/abs/2001.
06782.
Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, and Libing Wu. Agentic
memory: Learning unified long-term and short-term memory management for large language
model agents, 2026. URLhttps://arxiv.org/abs/2601.01885.
Guanxiong Zeng, Yang Chen, Bo Cui, and Shan Yu. Continual learning of context-dependent pro-
cessing in neural networks.Nature Machine Intelligence, 1(8):364–372, August 2019. ISSN
2522-5839. doi: 10.1038/s42256-019-0080-x. URLhttp://dx.doi.org/10.1038/
s42256-019-0080-x.
13

Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, and Bo Li. Lora-fa: Efficient and ef-
fective low rank representation fine-tuning, 2026. URLhttps://arxiv.org/abs/2308.
03303.
Heming Zou, Yunliang Zang, Wutong Xu, Yao Zhu, and Xiangyang Ji. Flylora: Boosting task
decoupling and parameter efficiency via implicit rank-wise mixture-of-experts, 2025. URL
https://arxiv.org/abs/2510.08396.
A SUBSTRATEDETAILS
This appendix expands the substrate measurements, routing mechanics, retrieval adapter training,
and continual-learning comparison summarized in Section 3.
A.1 RANK ANDkSELECTIONDATA
Table 6 shows the full metadata oracle selection acrossRandkwith per-class V AL/ABS breakdown.
All configurations use real substrate with per-rankτand each measurement is over5held-out users
(1452queries).
STABLE ROUTINE CONTESTED TRANSIENT ONESHOT
Config V AL ABS V AL ABS V AL ABS V AL ABS V AL ABS CB
FixedR= 32, varyingk
k= 10.58 0.70 0.33 0.83 0.42 0.53 0.26 0.89 0.62 0.77 0.61
k= 20.64 0.70 0.76 0.74 0.85 0.27 0.46 0.80 0.97 0.460.69
k= 40.57 0.75 0.97 0.58 1.00 0.15 0.92 0.41 0.97 0.62 0.66
k= 80.49 0.75 0.95 0.40 1.00 0.09 0.97 0.21 0.92 0.48 0.57
k= 160.36 0.75 0.90 0.21 0.98 0.06 0.97 0.09 0.75 0.33 0.48
Fixedk= 2, varyingR
R= 160.52 0.82 0.81 0.67 0.91 0.23 0.67 0.63 0.85 0.62 0.65
R= 320.64 0.70 0.76 0.74 0.85 0.27 0.46 0.80 0.97 0.46 0.69
R= 640.73 0.58 0.67 0.75 0.73 0.27 0.41 0.91 0.96 0.590.70
R= 1280.82 0.67 0.61 0.79 0.68 0.23 0.13 0.93 1.00 0.57 0.70
Table 6: Full metadata oracle selection across substrate rankRand mask sizek, with V AL/ABS
split per class. The value-abstain trade-off is visible in both dimensions. Increasingkgenerally
shifts volatile and revision-heavy classes toward higher V AL at the cost of ABS, while increasingR
redistributes between classes without changing aggregate substantially aboveR= 32.
The per-class V AL/ABS decomposition surfaces two mechanisms:
Increasingkshifts the value-abstention trade-off.CONTESTED V AL climbs from0.85atk=
2to1.00atk= 4and above, but CONTESTED ABS falls from0.27to0.06. The same pattern
is observed for TRANSIENT (V AL0.46→0.97, ABS0.80→0.09). Largerkmeans each write
touches more columns, so facts are more likely to persist under any query (raising V AL) but also
less likely to expire correctly (lowering ABS). Atk= 16, the substrate is essentially always writing
to overlapping columns and per-factτhas been almost entirely overwritten.
IncreasingRprimarily improves STABLE.STABLE V AL climbs from0.52atR= 16to
0.82atR= 128, a30-point improvement. This is exactly the mechanism we expect as more
rank columns means each STABLE fact is less likely to share columns with subsequently-written
short-τfacts. However, ROUTINE and CONTESTED V AL both drop asRgrows because their
routing dispersion increases (the router allocates similar facts to more distinct columns). The trade-
off produces near-flat aggregate accuracy aboveR= 32, at which pointR= 32is preferred for its
2–4×smaller inner-loop compute per write.
Dynamic router mechanics.The dynamic router projects each fact’s routing embedding through
a learned whitening transform (fitted to dimension256), then compares it by cosine similarity against
the accumulated signature of every currently allocated mask. A mask’s signature is maintained as an
L2-normalized exponential moving average (α= 0.5) of the embeddings previously routed there. If
14

the highest similarity exceeds a threshold of0.85, the write is routed to that mask and its signature
is updated, otherwise a new mask is allocated, with up to R
k
locations total.
A.2 SUBSTRATEDYNAMICSMEASUREMENTS
Retrieval after first write, full curve.Table 7 extends Figure/Table 2 (main text) to all tested
depths.
Depthd10 15 20 30 40 60 80 110 150
Perfect router 0.00 0.00 0.00 0.05 0.13 0.61 0.78 0.89 0.97
Learnt router 0.00 0.00 0.01 0.15 0.49 0.84 0.90 0.94 0.98
Table 7: Retrieval after first writeP(value)per depth.
Accumulation under repeated writes.Table 8 supports the constructive-accumulation claim of
§3.1. Five writes atd= 20(P≥0.93) substantially exceed a single write at the same depth (P≈0
atd= 20per the table above), and approach the retention of a single deep write in thed= 80–110
range.
n= 1n= 3n= 5n= 8
d= 20(perfect) 0.00 0.29 0.93 0.97
d= 20(learnt) 0.01 0.45 0.95 0.98
Table 8: Cumulative retentionP(value)afternwrites at depthd= 20with minimal intervening
foreign traffic.
Contradict-primed overwrite.Table 9 characterizes overwrite behavior (§3.4): a contradicting
value written atd≥40dominates retrieval under both routing configurations.
Challenger depth 20 40 80 110
P(v 2)perfect router 0.55 1.00 1.00 1.00
P(v 2)learnt router 0.92 1.00 1.00 1.00
Table 9: Contradict overwrite. Initial fact is written atd= 40with valuev 1. The overwrite is a
subsequent write with valuev 2at varying depths.P(v 2)is the retrieval probability of the overwrite.
Priming and decay validation.To validate that per-rank exponential decay behaves as designed,
we run a simple test. A fact is written at depthdwithτ= 1000, decayed for∆tdays, and retrieved.
Measured retention matches predictedP(cold, d)·exp(−∆t/τ)within measurement noise ford∈
{40,80,110}and∆t∈ {10,100,300}days.
Slot norm regression details.The slot norm regression takes as input a five-dimensional stan-
dardized feature vector: self steps, foreign steps,log(1 +self steps),log(1 +foreign steps), and
time since last write. Cross-validatedR2is0.87perfect routing and0.75on the learned router.
The dynamic router’s lower predictability reflects that its slot signatures drift with fact allocation,
adding noise the regression cannot capture. This regression is what allows offline controller training
to substitute predicted slot norm for a measured one (§4.2), at a15×speedup.
Forward pass test.Before each write, we computeP(value|current slot state)−P(value|
clean slot)This provides a substrate-derived signal of whether the incoming value is already rep-
resented in the routed slot. The difference is positive for reinforcement events and near zero for
first-write events. AUC against the ground-truth reinforcement labels is0.99on both routers.
15

Cost and latency.Table 10 compares PMM with RAG-MM, measured on5users. Reads take a
constant94ms regardless of how many facts are stored, versus360ms for RAG-MM, whose cost
grows with the retrieved context. Writes are more expensive and depend on depth (1.8s atd= 20
to13.1s atd= 150, versus2.4s), but they can run asynchronously, off the query path. Over a full
user trajectory, PMM costs702s against617s (+14%), and is cheaper for2of the5users.
PMM RAG-MM
Read (per query) 94 ms 360 ms
Write (per fact) 1.8 s (d=20) – 13.1 s (d=150) 2.4 s
Full user trajectory 702 s 617 s
Table 10: End-to-end timing (5users).
A.3 RETRIEVALADAPTERTRAINING
We train a rank-32retrieval adapter jointly loaded with the memory adapter during both writes and
reads. The training data is a mixed recipe: some facts get shallow writes (d∈ {10,15,20}), some
medium (d∈ {30,40}), some deep (d∈ {60,80}), some very deep (d∈ {100,130}), and some
slots are left empty. For each written fact, we sample the target asUnknownwith probabilityp unk(d)
or the correct value otherwise.
Table 11 shows the final noise curve alongside the implied target retrieval probability. The constraint
punk(d)≤0.55prevents the retrieval from collapsing to always-abstain at any depth.
Depthd10 20 30 40 60 80 110+
punk(d)0.55 0.38 0.30 0.22 0.12 0.05 0.00
ImpliedP(value)0.10 0.20 0.32 0.45 0.70 0.85 0.95+
Table 11: Retrieval adapter noise curve. Each depth targets a specificP(value)during SFT.
A.4 CONTINUALLEARNINGBASELINEDETAILS
We evaluate two continual-learning methods against our substrate, all applied to LoRA writes at
fixed depthd= 15(chosen fromd∈ {15,30,60}for each method).d= 15maximized each
method’s aggregate accuracy. All CL baselines use the same base model and LoRA (rank32on
qproj,vproj) but with fixed depth and no per-rankτ.
Naive fine-tuning.Direct backprop on the LoRA parameters at each write event, no regularization.
Serves as the floor baseline for LoRA-based approaches.
EWC (Elastic Weight Consolidation).Adds a Fisher-weighted penalty term to write losses:
Ltotal=L write+λP
iFi(θi−θ∗
i)2whereF iis the diagonal Fisher information at parameteri
estimated from the calibration data,θ∗
iis the pre-write parameter, andλ= 1000(from (Kirkpatrick
et al., 2017)). Fisher is updated with exponential moving average,α= 0.9.
PCGrad (Projected Conflicting Gradients).Maintains a running memory of thek= 5most
recent write gradients. Each new write gradient is projected to be orthogonal to any stored gradient
with which it has negative cosine similarity, following (Yu et al., 2020). This is designed to prevent
new writes from destroying prior ones.
Retrieval adapter compatibility.CL baselines cannot use our retrieval adapter directly because
the resoute was trained assuming thek-hot mask retrieval. For fair comparison, we retrained a
compatible retrieval adapter for each CL baseline that reads from the full LoRA weight rather than
a masked subset. This ensures accuracy differences reflect substrate design, not reading calibration
mismatches.
16

B LEARNEDMEMORYMANAGEMENTDETAILS
This appendix expands the controller architecture, expert policy, training details, per-class break-
downs, retrieval baseline architecture, and negative results summarized in Section 4.
B.1 FULLVAL/ABS BREAKDOWN
Table 12 shows the per-pattern accuracy split by target type (V AL versus ABS) for the systems in
Table 3 and Appendix B.6.
STABLE ROUTINE CONTESTED TRANSIENT ONESHOT
System V AL ABS V AL ABS V AL ABS V AL ABS V AL ABS
Metadata oracle 0.53 0.74 0.75 0.65 0.84 0.24 0.52 0.81 0.93 0.46
Naived= 150.17 0.00 0.72 0.00 0.63 0.00 0.76 0.00 0.26 0.00
EWCd= 150.19 0.00 0.73 0.00 0.64 0.00 0.76 0.01 0.19 0.00
PCGradd= 150.70 0.00 0.96 0.00 0.89 0.00 0.91 0.00 0.64 0.00
Unmanagedd= 200.64 0.09 0.90 0.01 0.87 0.06 0.77 0.22 0.55 0.27
Unmanagedd= 400.81 0.00 0.99 0.00 0.99 0.01 0.96 0.02 0.92 0.03
Unmanagedd= 800.85 0.01 0.99 0.00 0.98 0.00 1.00 0.00 0.97 0.01
Unmanagedd= 1100.83 0.00 1.00 0.00 1.00 0.00 1.00 0.00 0.97 0.00
RAG (1.7B) 0.22 0.76 0.27 0.75 0.16 0.87 0.26 0.84 0.16 0.79
RAG-MM (1.7B) 0.39 0.53 0.45 0.46 0.38 0.56 0.36 0.74 0.30 0.63
RAG-MM (122B) 0.44 0.65 0.81 0.42 0.75 0.24 0.94 0.22 0.55 0.78
RAG-MM (1.7B, trained) 0.31 0.83 0.38 0.69 0.30 0.75 0.40 0.83 0.29 0.82
RAG + oracle metadata (1.7B) 0.39 0.78 0.55 0.65 0.50 0.75 0.60 0.90 0.41 1.00
RAG + oracle metadata (122B) 0.86 0.77 0.84 0.54 1.00 0.53 1.00 0.75 1.00 0.97
PMM-PR (ours)0.33 0.81 0.80 0.64 0.81 0.34 0.66 0.58 0.82 0.58
PMM-LR (ours)0.34 0.78 0.88 0.47 0.93 0.21 0.83 0.35 0.71 0.60
No per-rankτ(PMM-PR) 0.55 0.68 0.96 0.06 0.87 0.04 0.83 0.09 0.79 0.22
Metadata oracle (NL) 0.55 0.85 0.86 0.58 0.95 0.22 0.90 0.46 0.93 0.81
PMM-PR (NL) 0.36 0.97 0.93 0.40 0.94 0.24 0.92 0.40 0.62 0.86
PMM-LR (NL) 0.40 0.92 0.94 0.40 0.96 0.17 0.97 0.24 0.45 0.65
Table 12: Per-pattern accuracy split into V AL and ABS query types.n= 15held-out users.
B.2 UNMANAGEDBASELINESWITHOUTτ
For completeness, we ran the unmanaged baseline depth selections both with per-rankτ(main table)
and withoutτ(this table).
System CB STABLE ROUTINE CONTESTED TRANSIENT ONESHOT
d= 20 0.45 0.54 0.62 0.41 0.30 0.40
d= 40 0.49 0.70 0.68 0.46 0.15 0.45
d= 80 0.50 0.73 0.68 0.45 0.15 0.47
d= 110 0.49 0.71 0.69 0.46 0.15 0.47
Table 13: Unmanaged baselines withoutτare nearly identical to their with-τcounterparts.
The near-identical aggregate numbers (all within±0.01CI overlap) demonstrate an important point
thatτalone is not what enables our controller’s performance. The controller’s benefit comes from
combining per-factτpolicy with per-fact depth and gate decisions.
B.3 RETRIEVALBASELINEDETAILS
All retrieval baselines share three components: an external text store containing the naturalistic
realization of each event, a BGE-small embedder (33M parameters) for indexing and query-time
retrieval, and a generator. Because the data is an event stream, no text chunking is required and
facts are embedded directly as individual, timestamped strings (e.g.,[Clock: 0045] I cook
dinner). At query time, the system retrieves the topk= 5most similar entries. The generator
17

prompt explicitly instructs the model to read the retrieved timestamped entries and outputUnknown
if the memory does not clearly support an answer or appears outdated.
The variants differ only in how the store is maintained:
•RAG:the standard baseline. Events are appended to the store as they arrive and no management
is applied. Retrieval at query time returns the top-kmost similar entries.
•RAG-MM:the store is actively managed at write time by an LLM management pass. As each
event arrives, the pass resolves its relationship to existing entries—merging reinforcements of the
same fact, superseding contradicted values, expiring entries past their inferred validity, and dedu-
plicating. It receives the new fact and retrieved candidates, and outputs a JSON decision to either
ADD, UPDATE (soft-deleting the old entry), REINFORCE (merging and extending lifespan), or
IGNORE. It also assigns an estimated decay in days (e.g., Transient:7.0, Routine:30.0, Stable:
3650.0).
•RAG + oracle metadata:identical to RAG-MM except the LLM management pass is replaced by
the ground-truth lifecycle policy, using each event’s generation-time metadata (validity horizon,
contradiction and reinforcement status) to maintain the store perfectly. This removes lifecycle-
management error while retaining the same retrieval and generation pipeline, providing an upper
bound on retrieval-side memory management under privileged lifecycle information.
Generators.Compute-matched rows (labeled1.7B) use Qwen3-1.7B, the same base model as our
substrate. Large-generator rows (labeled122B) use Qwen 3.5122B A10B.
B.4 CONTROLLERARCHITECTUREDETAILS
The learned management system is a 2M-parameter neural network consisting of two primary mod-
ules: a State Representation Engine (which maintains trajectory context and extracts semantic fea-
tures) and the Action Policy (which predicts lifecycle management decisions). All internal repre-
sentations use a hidden dimension ofd= 256.
1. State Representation EngineThe representation engine processes the raw outputs of the base
language model into dense, controller-ready features. Letx t∈R2048be the last-token embedding
of the incoming event template from the base model.
1.Semantic Projection:The raw embedding is normalized via a fitted input statistics
layer and projected to a dense representation to decouple it from lexical noise:z t=
MLP sem(xt)∈R256.
2.Trajectory GRU:To capture user-specific behavioral context, a Gated Recurrent Unit
(GRU) maintains a hidden stateh t∈R256. At each event, the GRU consumes the dense
embeddingz tand the token-level forward pass result (probability of fact before writing)
phit∈[0,1](indicating if the model already knows the fact):
ht=GRUCell(x in= [zt⊕p hit], ht−1)
2. Auxiliary State HeadsBefore computing actions, the network predicts the fact’s structural
properties using one auxiliary head attached to the representation engine. Because this property
depend only on semantics (not the substrate), its input is detached from the computational graph to
prevent auxiliary gradients from corrupting the shared representation trunk.
•Stability Head:A scalar prediction of the fact’s true stability score, optimized via Mean
Squared Error (MSE):ˆs t=σ(Linear(z t⊕ht))∈(0,1).
3. Action Policy and Shared TrunkThe main controller determines the action tuple
(gate,depth, τ)by observing both the semantic trajectory and the state of the underlying LoRA
substrate. The input to the shared policy trunk is the concatenation of six feature families:
spolicy=h
zt⊕ht⊕A⊕ ||W slot||2⊕ˆst⊕F rank⊕p hiti
whereA∈RRare the rank ages,||W slot||2is the predicted or measuredL 2norm of the active LoRA
mask, andF rankare the substrate interference features (foreign and self steps).
18

4. Decision Heads and PriorsThe policy states policy is passed through independent linear heads
to produce the final action logits. The model utilizes empirical priors to stabilize training and guide
the depth decision:
•Gate Head (g∈ {0,1}):Outputs a single logitl gate. To prevent the gate from collapsing
to0early in training, the logit is initialized with an empirical biasb gatederived from the
training set’s marginal write rate:P(write) =σ(l gate+b gate).
•Persistence Head (τ∈ {25,100,400,1600,20000}):Outputs a 5-dimensional logit vec-
torlτ, shifted by empirical class frequency biases to account for the heavy skew toward
shorter decay horizons.
•Depth Head (d∈ D >0):Outputs a 7-dimensional logit vectorl depthover the available non-
zero write depths{20,30,40,60,80,110,150}. The depth logits are directly augmented
by the auxiliary head, acting as a learned prior.
lfinal
depth=l depth+b empirical +α 1(ˆstVstab)
whereV stabis a vector pushing high-stability predictions toward deeper bins.
B.5 TRAININGDYNAMICS ANDLOSSFORMULATION
Training a shared representation trunk alongside multiple action heads presents significant opti-
mization challenges. To stabilize Behavioral Cloning (Phase 1) and ensure smooth convergence, we
employ three specific training techniques.
1. Ordinal Soft Targets for Depth ClassificationStandard cross-entropy loss heavily penalizes
any deviation from the target class. However, write depthdis an ordinal variable where adjacent
bins represent similar retention strengths. To reflect this, the magnitude head is trained via Kullback-
Leibler (KL) divergence against ordinal soft targets. If the expert deficit calculation targets depthd∗,
the target distribution smears probability mass across adjacent available bins according to proximity,
rather than placing a one-hot1.0on the nearest bin. This ensures that near-miss predictions (e.g.,
predictingd= 80when the target is110) are penalized less than catastrophic under-writes.
2. DAgger Expert AnnealingDuring Phase 1, we use DAgger (Ross et al., 2011) to supervise the
policy over the substrate. The probabilityβof executing the expert’s target action versus the student
policy’s sampled action follows a linear decay schedule. For the first5%of training (warmup),
β= 0.95, ensuring the substrate remains sound while the trunk learns. Over the remainder of
training,βdecays linearly to0.30, forcing the student to increasingly manage the consequences of
its own imperfect writes.
Total Phase 1 Loss FormulationThe complete multi-task objective during Behavioral Cloning
is:
Ltotal=w ctrl
λgLBCE(gate) +λ dLKL(depth) +λ τLCE(τ)
+λsLMSE(stability)
wherew ctrlis the warmup schedule scaler, andλare static hyperparameter weights balancing the
relative entropy of the action spaces.
B.6 FULLRESULTS: ADDITIONALBASELINES
Table 14 gives the remaining systems omitted from Table 3 for space.
B.7 EXPERTPOLICYDETAILS
The deficit-based expert (capped deficit) computes, for each write event, a scalar deficit and
returns an action tuple.
The foreign thresholdF∗is calibrated per machine as the75th percentile ofFover training-set write
events, producing roughly25%of writes triggering non-trivial foreign discount.
19

System CB STABLE ROUTINE CONTESTED TRANSIENT ONESHOT
Naive0.23 0.14±.05 0.50±.05 0.29±.05 0.11±.04 0.12±.04
EWC0.22 0.16±.04 0.50±.05 0.29±.05 0.12±.05 0.09±.04
Oracle RAG (1.7B)0.64 0.45±.05 0.58±.05 0.63±.06 0.86±.06 0.71±.05
Oracle RAG (122B)0.82 0.84±.04 0.75±.05 0.75±.05 0.79±.05 0.99±.04
Table 14: Additional baselines omitted from Table 3 for space: weaker continual-learning variants
and retrieval given ground-truth lifecycle metadata in place of a learned or LLM-managed store.
Algorithm 1Deficit-based expert policy
1:Inputs:stability scores,is reinforcementflagr, base decayb, self-steps on maskS,
foreign-steps on maskF, depth bins{d i}, foreign thresholdF∗, toleranceϵ, per-event cap
C= 150.
2:Compute target retention depthD∗=D∗(s, b, r)based on substrate curves.
3:Compute effective accumulated depthD acc=S−γFwith foreign discountγ= min(1, F/F∗).
4:Compute deficit∆ =D∗−D acc.
5:if∆≤ϵthen
6:return(g= 0, d= 0, τ= 0){decline}
7:end if
8:Set write depthd∗= min(∆, C).
9:Snapd∗to nearest available bin.
10:Setτ∗= arg min b|logτ b−log(24.5·b)|.
11:return(g= 1, d=d∗, τ=τ∗).
The target depth functionD∗(s, b, r)interpolates the substrate retrieval curve to determine what
accumulated depth is needed for a fact of stabilitysand base decaybto remain retrievable through
its expected query horizon. Reinforcement events use a slightly higherD∗than first-write events to
preserve accumulation opportunity for future queries.
B.8 STATEFEATURES
The controller conditions on six feature families summarized in Section 4.2. Full details:
Dense semantic embedding.A projection of the fact template’s last-token representation from the
base model, passed through an input normalization and a small MLP (sem proj) that decorrelates
raw LM features from lexical surface form. Dimension256.
GRU hidden state.A recurrent summary of the trajectory so far, updated per event. Takes the cur-
rent fact’s dense embedding, event type indicator, and time delta since the previous event. Provides
temporal context that a bag-of-features representation would lose. Dimension256.
Rank ages.For each of theRrank columns, the time in days since that column was last writ-
ten. Long-inactive columns are candidates for aggressive writes because their existing content has
already decayed; recently touched columns carry current content that will be interfered with. Di-
mensionR.
Slot norm.TheL 2norm of the mask-restricted LoRA weights, either measured directly on the
substrate (during online training and evaluation) or predicted from ledger state via the fitted regres-
sion from Appendix A.2 (during offline training, for a15×speedup). Dimension1.
Predicted stability.A scalar prediction from a dedicated head trained via MSE against the ground-
truth stability score. Enables the controller to condition depth and gate decisions on predicted fact
durability without needing symbolic access to metadata at inference.
20

Mask-level substrate features and forward-pass signal.For the specific mask that would re-
ceive this write, mean rank age within the mask, mean foreign-step count within the mask, and
the forward pass test hit (Appendix A.2). The forward pass provides a substrate-derived signal of
whether the incoming value is already represented in the current slot, complementing the rank-age
and interference features.AUC against ground-truth reinforcement labels reaches 0.99. Dimension
Nrank feats + 1.
B.9 TAUHEAD
Theτhead is a five-way classifier over{25,100,400,1600,20000}days. For a fact with base decay
b, the target lifespan class is
τ∗(b) = arg min
τj∈T|logτ j−log(24.5b)|,
whereT={25,100,400,1600,20000}days.
Class weighting.Theτhead uses inverse-frequency class weights since the target distribution is
more skewed. Bin0(τ= 25) receives roughly46%of events, bin1(τ= 100) receives10%, bin
3(τ= 1600) receives2%. Inverse-frequency weighting maintains recall on rare bins at the cost of
noisier training signal.
B.10 EMPIRICALBIASINITIALIZATION
Each action head’s output bias is initialized tolog(f)wherefis the marginal frequency of each
action in the expert policy on a calibration set of150users. Table 15 shows the initialized values.
Head Initialized bias values
Store0.99(empirical write rate72%)
Depth[−0.67,−4.47,−3.77,−2.01,−2.23,−2.03,−2.52]over[20,30,40,60,80,110,150]
τ[−0.77,−2.26,−1.35,−3.81,−1.88]over[25,100,400,1600,20000]
Table 15: Empirical bias initialization from expert marginals on150-user calibration set.
The depth bias is heavily negative on rare bins (e.g.,d= 30at−4.47reflects that only1%of
expert writes use this depth). The controller learns to adjust these biases per fact context and the
initialization ensures early training does not have to first recover the marginal distribution.
B.11 LIFECYCLE-CLASSDEPTHPRIOR
The prior matrixW∈R5×7maps predicted lifecycle-class probabilities to depth-preference adjust-
ments.
Patternd= 20d= 30d= 40d= 60d= 80d= 110d= 150
STABLE−0.46−0.46−0.46−0.42−0.19 +0.99 +0.99
ROUTINE+1.15 +0.99 +0.60−0.30−0.74−0.85−0.85
CONTESTED+1.03 +1.03 +0.74−0.18−0.75−0.93−0.93
TRANSIENT+1.15 +0.99 +0.60−0.30−0.74−0.85−0.85
ONESHOT−0.60−0.31 +0.11 +0.98 +0.98−0.31−0.85
Table 16: Pattern-conditional depth prior matrixW. Each row is mean-centered so the prior does not
shift aggregate write rate. The prior is combined with the pattern head’s softmax asα pattern·ppattern·W
and added to the depth head’s raw logits.
A parallel stability-conditional depth prior encourages the depth head toward deep bins when pre-
dicted stability is high, using a seven-dimensional vector[−1,−1,−0.5,0,0.5,1.5,1.5]over the
depth bins, scaled byα staband the predicted stability scalar.
21

B.12 PER-ROUTERHYPERPARAMETERS
Building on the bias initialization (Appendix B.10) and depth priors (Appendix B.11) above, this
section gives the remaining per-model hyperparameters. The prior scales differ between routers due
to different substrate retrieval curves.
Hyperparameter PMM-PR PMM-LR
αpattern 1.0 0.5
αstab 2.5 0.5
wτbase weight1.0 1.0
wboost
τ 4.0 4.0
τboost duration (steps)200 200
τclass weighting inverse inverse
Depth class weighting sqrt sqrt
DAggerβschedule0.95→0.50 0.95→0.50
Controller warmup (steps)100 100
Trunk learning rate3×10−43×10−4
Controller learning rate9×10−49×10−4
BC epochs2 2
Batch size (users per step)2 2
Table 17: Hyperparameters for the two model configurations. PMM-LR uses smaller prior scales
because dynamic routing’s permissive shallow-bin dynamics would otherwise push the controller to
over-invest in deep writes.
B.13 STABILITYHEADPRETRAINING
The stability head is pretrained via MSE regression against ground-truth stability scores over
template-stability pairs from the training user pool. On920unseen templates (held-out from34
users not seen during training), Stage A pretraining reachesR2= 0.22at epoch10, peaking slightly
before plateauing.
The template-only prediction ceiling (best any model could achieve from the template embedding
alone) is approximately0.36. The gap from 0.22 to 0.36 is due to residual variance the head cannot
extract from the embedding alone. Meanwhile, the gap from 0.36 to 1.00 represents the limit of this
feature as identical fact templates have different true stabilities across different users depending on
their personal patterns. This limit flows downstream to theτhead, which reads from the stability
prediction.
B.14 STABLE ANALYSISDETAILS
Per-query diagnostics on STABLE V AL outcomes were collected on56STABLE V AL queries
across a5-user analysis sample using the PMM-LR controller.
Success (n= 28) Failure (n= 28)
Prior writes (mean)1.39 1.11
Last write depth (mean)109.6 104.3
Accumulated depth (mean)142.1 113.9
Days since last write (mean)18.5 536.7
Self steps at query124.1 57.2
Foreign steps at query1127.8 1205.4
Foreign-to-self ratio (median)8.6 23.2
Table 18: Per-query diagnostics on STABLE V AL. Successes and failures similar on writing behav-
ior (depth, prior writes) but differ dramatically on query horizon and foreign-to-self ratio at query
time.
The distinguishing features between success and failure are query horizon and foreign-to-self ratio
at query time. Successful STABLE queries fire on average18.5days after last write; failed queries
22

Semantic Generation Physics & Enforcement
Phase 1: LLM Gen.
Motif Sam-
pling & MaskingRaw Masked Ledgers LLM Pruning
Storage
Decay eff=
D0×1.8streakTrajectory Expansion
Constructive
AccumulationQuery Injector
Placement via
Decay Curve
Final DatasetFresh:t < D eff
=⇒V ALUE TargetStale:t > D eff
=⇒ABSTAIN Target
Figure 5: Overall dataset generation pipeline. Phase 1 grounds the data via LLM generation, while
Phase 2 strictly enforces temporal dynamics, constructive accumulation, and delay-calibrated query
injection.
fire536.7days after. Failed queries suffered from a2.7×higher foreign-to-self gradient ratio. This
was not caused by encountering more absolute foreign traffic (means 1205 vs. 1128), but by the
stable fact receiving far fewer reinforcing writes to defend its own trace against that traffic (57 vs.
124).
Max-τablation.An alternativeτupdate rule that preserves the longestτany fact has assigned
to a column (rather than overwriting) protects STABLE facts on shared columns from subsequent
shorter-τwrites. This variant achieves class-balanced accuracy0.45(5-user eval), with STABLE
V AL improving by∼13points but ABS accuracy collapsing across TRANSIENT, ROUTINE,
CONTESTED, and ONESHOT (each below0.05). TRANSIENT facts written to shared columns
accidentally inherit the longτof older STABLE facts. Consequently, these volatile facts never
decay, causing the system to provide outdated answers instead of correctly abstaining during stale
queries. This shows that the trade-off cannot be resolved by this simple max-τupdate rule and
reflects the coupling introduced by shared-column compression.
Reinforcement effect.STABLE facts written multiple times over their trajectory re-assert longτ
on their columns and refresh self-investment against accumulated foreign traffic. STABLE facts with
3+writes hit at0.68versus0.51for single-write STABLE. The controller does not independently
trigger reinforcement and it is driven by the upstream event stream identifying repeated observations
of the same fact.
B.15 PHASE2 PPO DETAILS
Phase 2 refines the phase 1 controller via PPO. Reward is only outcome-based for correct V AL and
ABS queries. The phase 1 checkpoint serves as the reference policy for KL regularization.
KL divergence to reference is penalized with weight0.10on the store head and0.02on depth. Batch
size is4users per PPO iteration, with3PPO epochs and minibatch size32. Value coefficient0.5,
auxiliary loss coefficient0.3, entropy coefficient0.015on depth and0.005on the store head.
In our experiments, phase 2 provided marginal improvement over phase1(roughly±1point class-
balanced), primarily by shifting the policy’s V AL-ABS balance without meaningfully improving
23

aggregate. We attribute this to phase 1 already being close to the metadata oracle (∼86%), leaving
little room for refinement via reward signal. We treat phase 2 as an optional stage.
B.16 INPUTFEATUREABLATIONS
To understand which information is strictly necessary for parametric memory management, we per-
formed a set of input feature ablations. Because the controller uses a shared representation trunk that
feeds multiple heads, simply removing an architectural head could confound the results. Instead, we
ablate by zeroing out the specific input sources entirely before they reach the controller, directly
testing the information bottleneck. We evaluate four ablations against the PMM-PR baseline:
1.No Semantic:Zeroes out the dense text embedding.
2.No Priors:Removes the Stage A stability warm-start and the mathematical depth-bias.
3.No History:Zeroes out the GRU trajectory hidden state.
4.No Substrate:Zeroes out substrate readings (probe hit,slot norm,foreign steps).
STABLE ROUTINE CONTESTED TRANSIENT ONESHOT
System CB V AL ABS V AL ABS V AL ABS V AL ABS V AL ABS
PMM-PR0.600.33 0.81 0.80 0.64 0.81 0.34 0.66 0.58 0.82 0.58
No History 0.58 0.33 0.91 0.67 0.68 0.67 0.32 0.74 0.68 0.58 0.76
No Substrate 0.58 0.34 0.95 0.77 0.55 0.76 0.28 0.70 0.55 0.65 0.72
No Priors 0.52 0.42 0.80 0.66 0.57 0.73 0.20 0.78 0.41 0.69 0.44
No Semantic 0.47 0.28 0.88 0.70 0.33 0.63 0.28 0.61 0.35 0.49 0.60
Table 19: Input feature ablations. Losing semantic features or priors causes substantial failure.
History and Substrate act as a dual-redundancy system, but isolating them reveals specific, severe
capability collapses (e.g., losing history crashes ONESHOT V AL by 24 points).
The per-class V AL/ABS splits (Table 19) reveal exactly how the controller uses its inputs, surfacing
three core findings:
1. Semantic embeddings and priors provide substantial management signal.Without seman-
tic embeddings, the model cannot map an incoming fact to a behavioral pattern. Without this infor-
mation, the policy falls back toward less differentiated write behavior, reducing its ability to separate
persistent and volatile events. Removing semantic features lowers CB to0.47, while removing the
substrate-derived priors lowers it to0.52.
2. History and substrate state provide complementary information.TheNo History(CB 0.58)
andNo Substrate(CB 0.58) ablations survive in aggregate because the GRU and the LoRA state
provide overlapping signals for reinforcement. If the GRU is removed, the substrate’sprobe hit
still tells the controller a fact is recurring. If the substrate is removed, the GRU’s temporal memory
knows it has seen the embedding multiple times. However, the class-wise splits show they are not
perfectly redundant:
•History disambiguates lifecycle from trajectory context:Without the GRU, the model is blind
to the timeline. When a ONESHOT fact arrives, the model cannot confidently assign it moderate
depth because it lacks the behavioral context to separate it from the first write of a ROUTINE fact.
Consequently, it hedges its bets, and ONESHOT V AL crashes by24points (0.82→0.58).
•Substrate state exposes the consequences of prior writes:Without the substrate, the model
cannot see collision metrics (foreign steps,slot norm). Blind to overlap, it overwrites
frequently. This leaves stale traces persisting when they should have decayed, dropping absten-
tion (ABS) accuracy on volatile, highly-reinforced classes (ROUTINE ABS drops0.64→0.55;
CONTESTED ABS drops0.34→0.28).
3. Higher abstention under ablation reflects more conservative writing.STABLE and
ONESHOT ABS actually improve under these ablations (climbing to 0.91-0.95), but this is not a
genuine gain as the controller compensates for information loss by defaulting to shallower writes,
which decay faster and inflate ABS at the cost of V AL.
24

C DATASET ANDTRAJECTORYGENERATIONDETAILS
This appendix provides extended statistics, motif distributions, and qualitative examples for the
controlled long-horizon user trajectories introduced in Section 2.
C.1 DATASETSTATISTICS
The dataset consists of909user trajectories spanning multi-year horizons. Table 20 summarizes the
aggregate dataset statistics and the class balance across the five memory classes. Figure 6 visualizes
how identical fact properties map to different classes, while Figure 7 illustrates the query injection
density across a fact’s decay threshold.
Metric Value
Total Users909
Query Horizon Avg5,434days (Min:309, Max:25,282)
Total Writes170,121
Total Queries144,164
Query Target Split69.3%V ALUE|30.7%ABSTAIN
Classes % of Writes % of Queries
STABLE27.2% 46.7%
ROUTINE41.0% 24.3%
CONTESTED20.5% 11.4%
TRANSIENT7.7% 8.9%
ONESHOT3.7% 8.6%
Table 20: At-a-glance dataset statistics (top) and class balance distributions (bottom). Queries are
intentionally reweighted to ensure critical behaviors like expiration (TRANSIENT) are adequately
represented despite write imbalances. Query horizon is separate from write events horizon of 700-
1500 days.
1 2 3 4 5 6 7 8 9 10+
Number of Writes per Fact010002000300040005000Count of FactsClass
ROUTINE
CONTESTED
Figure 6: Two facts can have very similar properties such as stability and number of writes, yet
belong to very different classes based on user history and semantic meaning.
C.2 MOTIFDISTRIBUTION ANDCROSS-USERVARIANCE
To ensure heterogeneous, user-level memory dynamics, user personas were generated using a variety
of underlying temporal motifs governing their behavior (Table 21). Because these motifs dictate
behavior at the user level, the same fact type (e.g., a hobby or a job) can be stable for one user but
volatile for another. Figure 8 illustrates the resulting cross-user variance in routine predictability.
25

0.0 0.5 1.0 1.5 2.0 2.5 3.0
Normalized Query Time = (tquery−trecent_write)/decayeff01234DensityVALUE (Fresh Query)
ABSTAIN (Stale Query)
Decay ThresholdFigure 7: Fresh versus stale query injection density. Retrieval (V AL) and abstention (ABS) queries
are placed relative to each fact’s validity horizon.
Temporal Motif Users
NORMAL DRIFT: Slow, mundane, almost imperceptible changes347
THE MICRO-V ARIATION: Highly stable daily routine75
THE SEASONAL SHIFT: 2-to-3 month block behavior67
THE PHASED TRANSITION: Gradual shift over 2 months59
THE V ACATION/TRIP: Stable routine changes for 2 weeks51
THE INSTANT CORRECTION: Realizes mistake minutes/hours later45
THE CASCADE EVENT: Single major event triggers 3-4 routines41
THE FAILED HABIT: Tries for 5 days, then drops completely36
THE CONTEXT-DEPENDENT TOGGLE: Toggles based on day of week33
THE RELAPSE: Abandoned habit returns heavily for 2 weeks30
DORMANT RESURGENCE: Ignored for 6 months, then relevant26
THE RAPID PROMOTER: Switches roles every 3-4 months24
THE VOLATILE ROUTINE: Constantly breaks habit and gets back22
THE NOMAD: Moves and changes routine every 6 months20
THE CHAMELEON: Adopts completely new hobbies every season18
THE BINGE: Intense, short-term burst never happening again15
Table 21: Distribution of the underlying temporal motifs used to govern user memory dynamics
during dataset generation.
C.3 QUALITATIVEEXAMPLE
Table 22 provides a qualitative snippet of a user trajectory demonstrating how writes, updates, and
queries are interleaved. Entities are masked to decouple semantic world knowledge from parametric
retention.
C.4 BENCHMARKCOMPARISONCRITERIA
Table 1 compares prior long-term memory benchmarks along the following properties:
Belief revision.Whether later observations can supersede an earlier value and evaluation requires
recovering the currently valid state.
Time-based expiry.Whether previously valid information can become invalid solely because its
validity horizon has elapsed, without an explicit replacement, correction, or deletion event.
Reinforcement.Whether repeated consistent observations affect subsequent memory behavior,
such as strengthening retention, rather than being treated as independent events.
26

0.0 0.2 0.4 0.6 0.8 1.0
User Routine Fraction0102030405060Number of UsersFigure 8: Cross-user variance in routine prevalence. The variation across users illustrates why
lifespans cannot be determined solely from fact type and must also depend on individual trajectory
context.
Day Event Pattern Ledger Entry / Query
Day 0 INTRODUCE STABLE I work as a [JOB 8]
Day 2 INTRODUCE ROUTINE I ordered my morning coffee from [LOC CAFE 4]
Day 5 INTRODUCE TRANSIENT I grabbed a quick lunch with Sarah at [LOC BISTRO 9]
Day 12 ABS QUERY TRANSIENT I grabbed a quick lunch with Sarah at [MASK]
Day 14 INTRODUCE CONTESTED My current priority project is [PROJ 2]
Day 15 UPDATE ROUTINE I ordered my morning coffee from [LOC CAFE 4]
Day 18 V AL QUERY ROUTINE I ordered my morning coffee from [MASK]
Day 22 CONTRADICT CONTESTED My current priority project is [PROJ 7]
Day 25 V AL QUERY CONTESTED My current priority project is [MASK]
Day 30 INTRODUCE ONESHOT I booked a 2-week vacation to [LOC 1]
Day 45 V AL QUERY ONESHOT I booked a 2-week vacation to [MASK]
Day 400 V AL QUERY STABLE I work as a [MASK]
Table 22: A narrative snippet of a user trajectory demonstrating all five memory lifecycles. The
example illustrates the distinct target behaviors: TRANSIENT information is queried after its valid-
ity horizon, CONTESTED information is queried after a value revision, and STABLE information
must remain retrievable across long intervening histories. The non-query inputs also follow the same
[MASK]-Output format to memorize as the query inputs but are presented in a readable form here
with outputs values inserted in their respective [MASK] positions.
User-conditioned lifespan.Whether the appropriate lifespan of a memory depends on the indi-
vidual user’s observed history rather than only on the semantic type of the information.
Lifecycle supervision.Whether the benchmark provides event-level ground truth describing how
a memory’s validity or state evolves over time.
We mark a property asfullwhen it is explicitly constructed and evaluated,partialwhen related
behavior is present but not directly isolated, andabsentotherwise.
C.5 DATASETSCHEMA ANDCOMPREHENSIVETRAJECTORYEXAMPLE
Because the full dataset consists of multi-year trajectories with hundreds of events per user, we
provide a detailed example here to illustrate the exact inputs, metadata, and query formats utilized
by both our parametric controllers and the retrieval-based baselines.
27

Each user in the dataset is represented as a JSON object containing afacts metadatadictio-
nary (the ground-truth for each fact) and aneventsarray (the chronological stream of writes and
queries).
1. Ground-Truth Metadata DictionaryThefacts metadatadictionary maps each semantic
fact type (e.g.,favorite coffee shop) to its ground-truth lifecycle properties which represents a user’s
belief states. This metadata is strictly hidden from the controller at inference time but is used by the
DAgger oracle during Phase 1 training to compute target depths.
"facts_metadata": {
"user_profession": {
"write_pattern": "STABLE",
"stability_score": 0.92,
"base_decay_days": 3650.0
},
"morning_coffee_spot": {
"write_pattern": "ROUTINE",
"stability_score": 0.45,
"base_decay_days": 14.0
},
"current_project": {
"write_pattern": "CONTESTED",
"stability_score": 0.65,
"base_decay_days": 90.0
},
"todays_lunch": {
"write_pattern": "TRANSIENT",
"stability_score": 0.12,
"base_decay_days": 2.0
}
}
2. Chronological Event StreamTheeventsarray represents the timeline. Each event contains
a timestamp, the event type (INTRODUCE,UPDATE,CONTRADICT, orQUERY), and the event.
Below is an illustrative sequence showing how themorning coffee spot(a ROUTINE fact)
andtodays lunch(a TRANSIENT fact) are processed over a 20-day window.
Event 1: First observation of a routine (Day 0)
{
"clock_days": 0.0,
"event_type": "INTRODUCE",
"fact_type": "morning_coffee_spot",
"value": "CAFE_4",
"is_reinforcement": false,
"template_input": "I ordered my morning coffee from [MASK]",
"natural_language": "[Clock: 0000] I ordered my morning coffee
from CAFE_4."
}
Event 2: A transient, unrepeated event (Day 2)
{
"clock_days": 2.5,
"event_type": "INTRODUCE",
"fact_type": "todays_lunch",
"value": "RESTAURANT_9",
"is_reinforcement": false,
28

"template_input": "I grabbed a quick lunch with Sarah at [MASK]",
"natural_language": "[Clock: 0002] I grabbed a quick lunch with
Sarah at RESTAURANT_9."
}
Event 3: Reinforcement of the routine (Day 5)
{
"clock_days": 5.0,
"event_type": "UPDATE",
"fact_type": "morning_coffee_spot",
"value": "CAFE_4",
"is_reinforcement": true,
"template_input": "I ordered my morning coffee from [MASK]",
"natural_language": "[Clock: 0005] I ordered my morning coffee
from CAFE_4."
}
Event 4: Stale Abstention Query for the Transient fact (Day 15)
Becausetodays lunchhas a base decay of 2.0 days, and 12.5 days have passed since Event 2,
the ground-truth target strictly demands abstention.
{
"clock_days": 15.0,
"event_type": "QUERY",
"target_type": "ABSTAIN",
"fact_type": "todays_lunch",
"target_value": "Unknown",
"template_input": "I grabbed a quick lunch with Sarah at [MASK]",
"natural_language": "Where did you grab lunch with Sarah?"
}
Event 5: Value Retrieval Query for the Routine fact (Day 18)
Because the routine was reinforced at Day 5, the controller should have accumulated depth, keeping
the memory valid for this query 13 days later.
{
"clock_days": 18.0,
"event_type": "QUERY",
"target_type": "VALUE",
"fact_type": "morning_coffee_spot",
"target_value": "CAFE_4",
"template_input": "I ordered my morning coffee from [MASK]",
"natural_language": "Question: Where do you usually
order your morning coffee?"
}
D NATURAL-LANGUAGEMEMORYMANAGEMENTEND TOEND
The main paper isolates memory management with a templated interface. Here we run thecom-
pletesystem with natural-language (NL) writes and queries: the same substrate, controller archi-
tecture, action space and training recipe (§4). Only quantities that depend on the input format are
re-measured. On the same held-out users, NL performance is within0.02class-balanced (CB) ac-
curacy of templated performance for the oracle, PMM-PR and PMM-LR (Table 3).
D.1 SETUP
Each write is expressed as10question–answer pairs about the fact (e.g.,Where did I take my dog?
→the park). This means that natural language interfacing requires an additional question-answer
29

generation module to ensure correct writes. Model editing approaches may further help with the
write step and do away with the question generation module. A depth-dwrite performsdSGD
steps on the routed columns, split evenly across the pairs, with the loss on answer tokens. Queries
are free-form questions that do not repeat the write wording. The substrate mechanics are format-
independent, but write strength and its survival under decay are not. Table 23 measures both for
NL writes. Two properties differ from the templated setting: recall plateaus early (reliable from
d≈60), and decay acts as a cliff rather than a gradual fade (recall is0.88–0.96atf= 0.9, but at
most0.19atf= 0.7and zero atf= 0.5).
decay factorf= exp(−∆t/τ)at query time
depthd1.0 0.9 0.7 0.5
200.23 0.12 0.00 0.00
400.83 0.56 0.06 0.00
600.96 0.88 0.08 0.00
1101.00 0.94 0.17 0.00
1501.00 0.96 0.19 0.00
Table 23: Recall of a single NL fact vs. write depth and remaining decay factor (question–answer
writes, matched retrieval adapter,48facts).
D.2 RELATION TOPRIORWORK
Extracting knowledge written into weights with free-form questions depends on how it was written.
Allen-Zhu & Li (2024) show that extractability requires diverse formulations of each fact during
training, and Mecklenburg et al. (2024) and Ovadia et al. (2024) find that fine-tuned question an-
swering improves with paraphrase augmentation. Our lifecycle management (when to write, how
deeply, for how long) transfers to natural language without change to the substrate, or controller
training recipe.
30