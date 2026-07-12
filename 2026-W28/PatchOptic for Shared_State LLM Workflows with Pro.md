# PatchOptic for Shared-State LLM Workflows with Projected Views and Verified Structured Updates

**Authors**: Zhaoyu Bai, Jiaqi Cai

**Published**: 2026-07-06 16:13:56

**PDF URL**: [https://arxiv.org/pdf/2607.05483v1](https://arxiv.org/pdf/2607.05483v1)

## Abstract
Agentic workflows often operate over shared, structured state. Because LLM context windows are limited, each model invocation is typically shown only the state fragment needed for the current workflow step, a pattern commonly known as progressive disclosure. Modern systems construct such model-facing views using grep-like keyword search, retrieval-augmented generation (RAG), abstract-syntax-tree (AST) queries, and task-specific agent skills. These methods make the read side manageable, but they do not define when a locally proposed rewrite is valid after it is applied back to the full state. The missing piece is a contract between local updates and global validity. We introduce PatchOptic, an optic-inspired interface for shared-state LLM workflows. Optics are compositional bidirectional accessors that describe how views of structured data are read and updated. PatchOptic borrows this view/update intuition and realizes it through projected reads and verified structured patches. Each workflow step declares a projected read view, an authorized write region, and a patch-source region. Beyond runtime enforcement, the same declaration yields a path-level footprint that supports delegation, sub-workflow composition, and static certificates for reordering independent steps within the same phase. We evaluate this design with PatchBench, a benchmark with 46 cases across domains. The results show that projected reads reduce reported leakage and token cost while preserving accepted-output quality under the strong actor. Runtime verification blocks declared workflow-contract violations before commit, and patch-read enforcement rejects compromised patch artifacts that use hidden sources.

## Full Text


<!-- PDF content starts -->

PatchOptic for Shared-State LLM Workflows with Projected Views
and Verified Structured Updates
Zhaoyu Bai*1and Jiaqi Cai∗2
1Department of Particle Physics and Astrophysics, Weizmann Institute of Science, Rehovot,
Israel, 7610001
2Department of Physics, Massachusetts Institute of Technology, Cambridge, MA, USA, 02139
2026-06-30
Abstract
Agentic workflows often operate over shared, structured state [1 –3]. Because LLM context
windows are limited, each model invocation is typically shown only the state fragment needed
for the current workflow step, a pattern commonly known as progressive disclosure. Modern
systems construct such model-facing views using grep-like keyword search, retrieval-augmented
generation (RAG), abstract-syntax-tree (AST) queries, and task-specific agent skills. These
methods make the read side manageable, but they do not define when a locally proposed rewrite
is valid after it is applied back to the full state. The missing piece is a contract between local
updates and global validity. We introduce PatchOptic, an optic-inspired interface for shared-state
LLM workflows. Optics are compositional bidirectional accessors that describe how views of
structured data are read and updated. PatchOptic borrows this view/update intuition and
realizes it through projected reads and verified structured patches. Each workflow step declares a
projected read view, an authorized write region, and a patch-source region. The actor sees only
the projected view, while the verifier checks each proposed patch against the full state before
commit. Beyond runtime enforcement, the same declaration yields a path-level footprint that
supports delegation, sub-workflow composition, and static certificates for reordering independent
steps within the same phase. We evaluate this design with PatchBench, a benchmark with
46 cases across domains. The results show that projected reads reduce reported leakage and
token cost while preserving accepted-output quality under the strong actor. Runtime verification
blocks declared workflow-contract violations before commit, and patch-read enforcement rejects
compromised patch artifacts that use hidden sources. Together, these results pave the way for
safely using LLM workflow agents in large shared-state systems, where local actions can be
composed, delegated, and verified through explicit workflow footprints.
1 Introduction
Progressive disclosure is now common in agentic workflows. Because LLM context windows are
limited, systems typically expose to each model invocation only the fragment of the full state needed
for the current step. For example, RAG retrieves relevant corpus passages, agent skills provide
task-specific tools and instructions, and syntax-aware AST queries expose relevant code structure.
These techniques make the read side manageable [4 –9]. The write side is less settled. In a multi-step
∗Corresponding authors’ emails:jiaqic@mit.edu;zhaoyu.bai@weizmann.ac.il
1
arXiv:2607.05483v1  [cs.LG]  6 Jul 2026

workflow over shared structured state [1 –3], an agent often proposes an update based only on the
fragment it was shown. With only that fragment, the agent may miss how the proposed write
relates to the rest of the state, including dependencies, schema, phase, references, and permissions.
Judging global validity from this local view places a heavy cognitive burden on the agent. Recent
agent benchmarks evaluate this multi-step stateful regime [7,10 –13]. Prompt-injection work studies
a related class of failures in which untrusted context affects agent behavior [14–16].
Existing controls cover only parts of this problem. Schema validators can check output structure
and some local value constraints, but they do not by themselves establish that the values are
semantically correct or derived from authorized state [17]. Constrained decoding can restrict the
syntax of the generated string, but it does not verify the provenance or authorization of the state
values used to construct that string [18,19]. Used alone, these mechanisms remain local to the
generated artifact or to a single transition. We still need a way to judge a local write against the
whole state and to scale that judgment across a multi-step, multi-agent workflow. Closing both
gaps requires a contract that connects local rewrites to global validity and composes across steps.
We propose PatchOptic, an optic-inspired step interface for shared-state workflows. Optics
are compositional bidirectional accessors that describe how views of structured data are read and
updated [20,21]. PatchOptic borrows this view/update intuition, then realizes it through projected
views and patch verification. For each workflow step, the actor receives a local view, while its
proposed write is checked against the whole state before commit.
At runtime, PatchOptic gives the model only the projected view for its step. The model
returns a patch, and the verifier checks that patch against the whole state before commit. The
same step contract determines both sides of the interface. It contains an optic-inspired authority
triple together with phase constraints, schemas, applicability checks, and invariant predicates. The
authority triple defines the projected-read view, write scope, and patch-source paths (Figure 1). For
static reasoning, each step also has afootprintthat records projected-read, write, and patch-source
paths. Footprints support restriction for delegation and union for peer composition. They also give
a static commutation check when two steps have disjoint writes and neither reads what the other
writes.
Contributions.
(i) A runtime mechanism that couples projected views with verified JSON Patch writes over
shared state, with structured traces that record projected views, proposed patches, verifier decisions,
and rejection diagnostics.
(ii) A footprint algebra for declared projected-read, write, and patch-source regions. It supports
delegated restriction, peer composition by union, data-dependent regions, and same-phase reordering
certificates.
(iii) PatchBench, a 46-case benchmark spanning six domains, and an evaluation protocol that
separates live actor runs from no-model compromised-artifact tests. Projected-read settings raise
accepted-output semantic pass from about 0.61 to 0.78–0.80 under GPT-5-mini. PatchOptic
combines this read-side benefit with runtime verification, reducing leak runs to 0.1 per round and
rejecting all hidden-source patch artifacts in the containment test.
The rest of the paper is organized as follows. Section 2 defines the shared-state workflow setting,
the hidden sources that must be protected, and the workflow risks considered in this paper. Section 3
presents the PatchOptic design, including the runtime path that gives the actor a projected view
and checks a JSON Patch before commit, and the footprint algebra that supports delegation,
workflow composition, and safe reordering of independent same-phase steps. Section 4 describes
PatchBench and the evaluation protocol, including the separation between live actor runs and tests
using compromised artifacts without a model. Section 5 reports the empirical results, focusing on
output quality, reported leakage, runtime enforcement, and token cost. Section 6 discusses how the
2

results relate to the paper’s runtime, algebraic, and benchmark contributions, situates the work
relative to prior systems, and outlines limitations and future work. Section 7 concludes.
2 Shared-State Workflow Setting and Failure Model
In a multi-step workflow over shared structured state, two broad failure families arise. First, the
state may contain private values that are irrelevant to a given step. If those values enter the step’s
view, a minor prompt perturbation or compromised instruction can cause the actor to disclose them.
We call this family secret leakage.
Second, an untrusted or unreliable actor may return a patch that is not legal for the workflow.
The patch may be malformed, target the wrong phase, write outside the step scope, violate schema,
or construct an update from a source path that the step was not allowed to use. We call this family
commit-legality failures. A separate class of semantic output errors remains possible when the actor
invents a contract-valid but wrong value.
We focus on failures caused by an untrusted LLM invocation in a shared-state workflow. The
step prompt, actor, or returned patch may be faulty, compromised, or adversarial. It may disclose
hidden information, propose writes outside the intended task, act in the wrong workflow phase, or
construct an update from state that should not have influenced the step. We assume the workflow
policy itself is trusted, and discuss policy mistakes as a limitation in Section 6.
Thisframingfollowstheleast-privilegeandmediated-accesstraditioninprotectionsystems[22,23].
In our setting, the protected resource is shared workflow state. Table 1 summarizes the failures
considered in this work.
Failure family Covered? Scope
Direct secret exposure Partly Projection removes fields outside the declared
read view; transformed or policy-authorized dis-
closures may remain.
Commit-legality violations Yes Phase, write-scope, patch-source, applicability,
schema, and invariant violations are rejected be-
fore commit.
Semantic output errors No The actor may still fabricate an allowed-looking
but wrong value.
Policy-authorized disclosure No Ifthepolicyauthorizesthedisclosure, theruntime
enforces that policy.
Misuse of policy-visible context No The system does not prevent the actor from rely-
ing on information that the policy intentionally
exposes.
Table 1: Failure-model summary.
3 PatchOptic Design
3.1 Step Interface and Runtime Path
PatchOptic addresses these failure modes with a single declared step interface that is used both
at prompt time and at commit time. At runtime, projection restricts the information visible to
an actor, while verification checks proposed mutations before commit. The declared regions also
3

support static checks for delegation, composition, and same-phase reordering before the workflow
runs.
Formally, the step contract contains an optic-inspired authority triple
Is= (O s,Ws,Ps),
along with phase constraints, schemas, applicability checks, invariants, and optional preconditions
or postconditions.
HereOsis a projection over the shared state, Wsis the write region, and Psis the patch-source
region. The projection maps the full state xto a step-local view Os(x).Wsdefines the destinations
that the patch may modify, and Psdefines the current-state paths that patch operations may read
during application. When the projection is path-realizable, we write Rsfor the set of leaf paths
exposed by Os. The static footprint used below is then( Rs,Ws,Ps). The optic terminology follows
lens and view-update work [20,21]. In PatchOptic, this authority triple serves as part of the workflow
contract. Projection and verification therefore operate from one step declaration without claiming a
full categorical optic implementation.
Figure 1 summarizes the resulting runtime path. For a step s, the actor receives the projected
viewOs(x), performs the local task, and returns a mutation proposal p. The interface is independent
of any particular patch representation. In the current prototype, proposals are represented as
JSON Patch [24] over JSON Pointer paths [25], which makes proposed state changes explicit before
commit.
Shared state
xProjected view
Os(x)LLM
actorPatch proposal
p
Verifier
scope schema phase invariantsverifyp
currentx
commit:x←p(x) reject:xunchangedall gates pass any gate fails
mutate
Figure 1: Operational workflow and inspection point. The runtime holds the full shared state
x; the model only sees the projected view Os(x), with hidden fields absent. Each step returns a
JSON-Patch proposal pthat passes through a single verifier with four gates: write/patch-source
scope, schema, phase, and invariants. State mutates only when every gate passes; otherwise the
proposal is rejected and recorded in the audit log.
A mutation proposal must pass through the verifier before any state change occurs. Verification
interprets the declared write and patch-source regions( Ws,Ps), then checks whether the proposed
state transition is admissible. To do so, the verifier evaluates both the proposed destinations and
any source paths referenced by patch operators. For move, the frompath is also a write effect,
because JSON Patch removes that value before adding it at the target. The verifier also checks
phase constraints, write scope, patch applicability, schema validity, and workflow invariants. A
proposal that fails any check leaves the shared state unchanged. Table 2 summarizes the checks.
The implementation also supports optional preconditions and postconditions, which are checked
4

Check Question answered before mutation
Phase ordering Is this step allowed to act in the current workflow phase?
Write scope Do target paths stay inside the step’s authorized write region, and for
move, does thefrompath also stay inside that region?
Patch-source scope Docopy,move, and testsource paths stay inside allowed source
regions?
Patch applicability Is the patch structurally valid and applicable to the current state?
Schema validity Does the resulting state satisfy the case schema and output contract?
Invariants Do the workflow’s declared invariants still hold on the resulting state?
Table 2: Verifier checks executed before any mutation commits. The four gates in Figure 1 (scope,
schema, phase, invariants) group these six rows: “scope” covers both write scope and patch-source
scope, “schema” covers applicability plus output shape, and “phase” and “invariants” are one-to-one
with the rows here.
before scope validation and after invariant validation, respectively. They are not used as separate
benchmark dimensions in the results below.
Beyond enforcement, PatchOptic also emits audit traces. The runtime records projected views,
mutation proposals, verifier outcomes, and lineage edges. These traces support debugging and audit,
but they do not affect the runtime accept/reject decision.
3.2 Footprints for Static Checks
The runtime path above explains how one proposed update is checked before it commits. That
same step declaration is also used before any actor runs. Its declared read, write, and patch-source
regions give a footprint for the step. PatchOptic compares these footprints to check delegation,
sub-workflow composition, local actor replacement, and same-phase reordering before the workflow
runs.
We now define the footprint used for these static checks. We separate the step policy from the
prompt input used by each setting. Static checks compare declared regions of a structured state.
WriteLfor the locations used by the state representation, and write ⪯for containment between
locations. We use stable names for locations, so containment is a partial order: if two locations
contain each other, they are the same location.
For a state x, letΠ(x)⊆Lbe the leaf paths that currently exist in x. A declared region is
a finite subset of L. When a declared region Sis used to materialize a view on x, it exposes the
current leaves
S[x] ={ℓ∈Π(x)|∃s∈S. s⪯ℓ}.
In the JSON prototype, Lis the JSON Pointer region universe,Π( x)is the set of existing leaf paths,
and⪯is prefix containment.
The step policy for actorAis the footprint(R A,WA,PA), where
RA,WA,PA⊆L, P A⊆R A.
The regionRAis the read region authorized by the step. The region WAcontains the locations whose
write effects the verifier may accept. The patch-source region PAcontains current-state locations
that an update may read during application. In the prototype, these source reads correspond to the
frompaths of copyand move, and to the pathread by test. Amovesource must also lie in WA,
because JSON Patch removes the source value as part of the operation. In the prototype, PA=RA.
Thuscopy,move, andtestmay read only from the step’s authorized read region.
5

The prompt input is separate. Let VA⊆Π(x)be the leaf paths shown to the actor in a given
setting. Projected-read settings use VA=RA[x]. Full-input settings, including Verify Only, use
VA= Π(x). This larger prompt input does not change the verifier policy. Verify Only may show the
full state to the actor, but patch-source checks are still evaluated againstP A.
Byfootprint algebra, we mean two operations on footprints,restrictionandunion, together
with acommutation conditionfor reordering within one workflow phase. Restriction maps an actor
footprint to a smaller footprint:
(RA,WA,PA)∝⇕⊣√∫⊔≀→(R′
A,W′
A,P′
A),
where
R′
A⊆R A, W′
A⊆W A, P′
A⊆PA, P′
A⊆R′
A.
Union combines two peer footprints component-wise:
(RA,WA,PA)∪(R B,WB,PB) = (R A∪RB, WA∪W B, PA∪PB).
For two declared regionsSandT, writeS⊥ ⪯Twhen neither region contains a location from the
other:
S⊥⪯T⇐⇒ ∀s∈S,t∈T.¬(s⪯t∨t⪯s).
The commutation condition certifies two actors in the same workflow phase as independent when
WA⊥⪯WB, R A⊥⪯WB, R B⊥⪯WA.
The first condition prevents conflicting writes, including parent-child conflicts such as writing /user
while another actor writes /user/name . The other two conditions ensure that neither actor reads
a value written by the other actor. Since PA⊆R AandPB⊆R B, the same check also covers
patch-source paths. For operations that can change array shape, such as add,remove, ormoveon
an array element, the write-effect set used by the certificate also includes the containing array path.
The patch-source region PAis the component checked by the patch-source-scope rule in Table 2 and
exercised directly by the no-model artifact test.
These definitions draw on two lines of prior work. The projection x|RAis the view side of the
optic. It follows the lens and view-update lineage, where a view exposes part of a source state and
an update propagates changes back to that source [20,21]. In the current prototype, the update side
is realized as a checked JSON Patch rather than an unconstrained writeback. As an access-control
mechanism,( RA,WA,PA)is closer to role-based access control (RBAC), attribute-based access
control (ABAC), and capability-style protection [26 –28]. Together, the declared regions make
path-level authority explicit before the actor reads state or commits a mutation.
The footprint algebra above turns each step declaration into a static object that can be compared
before the actors run. Restriction gives the boundary used for delegation and local replacement.
The commutation condition gives the pairwise check used for same-phase reordering. Union is used
again when a certified sub-workflow is packaged for composition reuse, where reads and patch-source
paths supplied by earlier actors become internal handoffs rather than external inputs. The next
subsection fixes the actor behavior assumed by these checks. We then use that model to state the
commutation certificate and the reusable composite interface.
3.3 Static Actor Model
With this static footprint in place, we model the behavior attached to one step. In this model,
an actor is a deterministic function from its prompt input to a proposed patch. This is the actor
6

behavior used by the static checks; live runs may still be stochastic and are evaluated empirically. For
π∈Π(x), writex[π]for the value stored at that path. Given a state x, a step policy( RA,WA,PA),
and a setting-specific prompt inputV A, actorAreceivesx| VAand proposes a patch
A:x| VA∝⇕⊣√∫⊔≀−→p A.
The verifier does not use VAas the authority boundary. It accepts pAonly when its update targets
lie inWAand every current-state path read by the patch lies in PA. For move, the source path
must satisfy both conditions, since the operation reads that path and removes it. In the current
prototype, source reads arise from the copy,move, and testoperations of JSON Patch. For
projected-read executions, including PatchOptic, VA=RA[x]. For full-input ablations such as Verify
Only,VA= Π(x). Thus the static model separates the local actor logic from the step policy that
bounds patch sources and writes.
Restriction gives the delegation rule. If actor Bis delegated from actor A, thenB’s footprint
must be a component-wise restriction ofA’s footprint:
RB⊆R A, W B⊆W A, P B⊆PA, P B⊆R B.
The actor layer uses this relation to certify B’s narrowed footprint before materializing B’s view.
ActorBthen receives only x|RB, and the verifier checks B’s patch against WBandPB. Fields
outsideR Bare not represented inB’s prompt, and writes outsideW Bfail before commit.
Localactorreplacementusesthesameboundary. Twoactorswiththesamefootprint( RA,WA,PA)
may use different prompts or model implementations. For projected-read execution, the runtime
still gives them the same RAprompt input, and the verifier still accepts only the same target and
source paths. The replacement can change values insideW A. It cannot expand the projected-read
surface, cite new patch-source paths, or write outside WAwithout changing the declared footprint.
Such a change is an interface change and is rejected by the scope checks unless the declaration is
updated.
3.4 Commutation Certificate
Same-phase reordering uses the static actor model from the previous subsection. After phase
constraints have selected actors that may run in one phase, the certificate compares the committed
state produced by the two possible orders. We first attach a state transition to each actor. For
actorA, letu Abe its update producer. On statex, the proposed update is
pA(x) =u A(x|VA).
Letcommit (p,x)denote the runtime commit rule. It returns apply (p,x)whenpis applicable and
accepted by the verifier, and returnsxotherwise. The state transition induced by actorAis
TA(x) = commit(p A(x),x).
The certificate uses the following phase assumptions. Each actor is deterministic in its prompt
input and runs once. Actors have no externally visible side effects except their declared writes, and
the schedule has no retries, rollbacks, or exception-dependent branches. Footprints are fixed for
the phase. Conflict checks are prefix-closed, and verifier predicates use declared support disjoint
from the other actor’s writes. These predicate supports cover applicability, schemas, preconditions,
postconditions, and invariants.
7

For two actors AandB, the certificate checks disjoint writes and no read-after-write dependence
across the pair:
WA⊥⪯WB, R A⊥⪯WB, R B⊥⪯WA.
Here the first condition prevents conflicting writes. The other two conditions ensure that neither
actor reads a value written by the other actor. The patch-source regions matter because update
operations may read state through source paths. For example, under a JSON Patch realization,
copy,move, and testread from existing paths. A movesource is also a write effect, so it must
be covered by the write region as well. Because PA⊆R AandPB⊆R B, the read-dependence
conditions also keep patch sources away from the other actor’s writes. The certificate also needs
structural conflicts to be represented in the footprint. A prefix-closed conflict relation treats parent
and child paths as conflicting. A leaf-granular realization must instead expand every structural
write to the leaves whose values or applicability it can affect. Parent-container rewrites, array-index
shifts, and other shape-changing updates are not certified unless this effect is represented in the
declared footprint. If A’s source paths could reach into WB, the two schedules could give Adifferent
source values.
The prototype implements this as a conservative sufficient test over realized leaf footprints. It
refuses pairs with data-dependent prism footprints, opaque pre/post conditions, or invariants whose
declared support touches both write regions. Pairs outside these assumptions are kept in the given
order or sent to the differential check discussed below.
For a workflow phase with Nactors and fixed footprints, certification costs/parenleftbigN
2/parenrightbigpairwise conflict
checks and no model execution. Under the assumptions above, if all pairs pass, all N!orderings
produce the same final state in the static actor model. Any two orderings differ by a finite sequence
of adjacent transpositions. Each ordering terminates after exactly Nsteps because each actor runs
once. For each adjacent pair of certified actorsiandi+ 1, the pairwise certificate gives
Ti+1(Ti(x)) =T i(Ti+1(x))
on every reachable state x, since the conflict check and declared verifier supports are stable when
footprints are fixed for the phase. Applying these equalities along the adjacent-swap sequence gives
equal final states for any two certified orderings. The certificate therefore replaces an enumeration
of schedules with the/parenleftbigN
2/parenrightbigpairwise checks above. When the fixed-footprint assumptions are not
available, the workflow keeps the declared order, or PatchOptic compares the two deterministic
executions on fixed emitted artifacts.
3.5 Composition and Incremental Reuse
On top of the reordering result, the same footprint algebra supports composition reuse. A certified
sequence should expose only the state it still needs from the larger workflow. Reads and patch-source
paths supplied by earlier actors in the sequence become internal handoffs, while the remaining reads,
remaining patch-source paths, and all writes form the composite boundary. Write A;Bfor the
composite that runs Aand thenB. For a fixed source state x, let∆ A(x)⊆W Abe the leaf paths
whose values are created, replaced, removed, or otherwise modified by the accepted run ofAonx.
Then the external footprint ofA;Bis
Rext
A;B=R A∪/parenleftbigRB\∆A(x)/parenrightbig,
WA;B=W A∪W B,
and, for patch-source paths,
Pext
A;B=PA∪/parenleftbigPB\∆A(x)/parenrightbig.
8

A later projected read or patch source drops out of the external boundary only when the later
dependency is satisfied by a prior write inside the composite. If B’s projected-read region contains
more paths than Awrote, the unwritten leaves remain external. This is the handoff internalization
used by composition. Thus the external footprint of a composite can be state-dependent whenever
the set of actually written paths is state-dependent.
For a sub-workflow P= (A1;...;Ak), let∆ <i(x)be the leaf paths written by the prefix
A1;...;A i−1on the source state used for certification. The reusable interface ofPis
σ(P) =/parenleftiggk/uniondisplay
i=1/parenleftbigRAi\∆<i(x)/parenrightbig,k/uniondisplay
i=1WAi,k/uniondisplay
i=1/parenleftbigPAi\∆<i(x)/parenrightbig/parenrightigg
.
The first component contains only reads not supplied by earlier actors in P. The second component
is the composite write boundary. The third component applies the same boundary calculation to
patch-source paths. Internal handoffs are checked when Pis certified, including both the path
region and the schema expected by the consuming actor.
OncePis certified, the larger workflow can use σ(P)as the boundary of a single packaged actor.
It checks new connections against σ(P)rather than reopening the internal actor graph. A refactor
that changes only the grouping of already certified actors can reuse the cached interface, while a
refactor that changes an external projected-read path, patch-source path, write path, or handoff
shape changesσ(P)and must be checked again.
3.6 Data-Dependent Footprints
The reordering and reuse rules above use fixed realized paths. This is literal for lenses and for
any optic whose realized paths stay fixed during the phase. Traversals and prisms use the same
view/update interface from the optic lineage [20,21], but their concrete leaf paths are obtained
by evaluating the optic on the current document. A traversal expands one declaration over the
current members of a container, so its realized paths depend on the container shape. A prism filters
candidate members by a predicate, so its realized paths also depend on the selected values. When
those realized paths can change during the phase, the footprint is better written as a state-indexed
function
FA(x) = (R A(x),W A(x),P A(x)).
The pairwise commutation test compares the regions the actors will actually use:
WA(x)⊥⪯WB(x), R A(x)⊥⪯WB(x), R B(x)⊥⪯WA(x).
For a state-indexed footprint, checking those prefix conflicts at xis not enough if one actor can
change the other actor’s realized paths. If Bwrites a field used by A’s prism predicate, or changes
the container shape used by a traversal, then A’s footprint after Bruns isFA(B(x)), not necessarily
FA(x). Thus
FA(x)̸=F A(B(x))
can hold, and the prefix-conflict check at the initial state may refer to the wrong path sets.
The current commute implementation therefore refuses predicate-dependent footprints and relies
on shape-stability assumptions for traversal regions. It does not try to prove that each actor leaves
the other actor’s selector inputs unchanged. When reordering still matters, commute_differential
runsA;BandB;Aon the current document, using deterministic actors or frozen proposals, and
accepts only if the final documents are identical. The conclusion is tied to that document, because
a different document can realize different path sets.
9

Taken together, these footprint operations provide capabilities that are not captured by path-
prefix ACLs or finite-state workflow policies. Path-prefix ACLs typically authorize individual target
paths, and FSM-style policies authorize phase transitions. The footprint layer keeps those checks,
but it also records the projected read surface, patch-source authority, reusable composite interfaces,
handoff boundaries, and sufficient static certificates for same-phase reordering. It also extends
the same interface to state-indexed regions such as traversals and prisms with the caveats above.
When the static certificate does not apply, the differential check can still compare the two concrete
executions on the current document.
4 Benchmark and Methodology
The design above defines PatchOptic as a projected-read and verified-write interface for shared-state
workflows. We now introduce PatchBench, a benchmark that evaluates this design on realistic,
domain-inspired tasks. PatchBench is designed to measure whether projection and verification
improve artifact quality, reduce leakage, and reject invalid commits, and to separate their effects
through ablations.
4.1 Benchmark Construction
Following the problem setting and failure model in Section 2, a PatchBench case consists of a shared
structured state, a task instruction, a workflow contract, and an expected runtime outcome. The
contract specifies the step read policy, the allowed write region, the patch-source policy, and the
runtime checks that apply to the step or workflow. Each setting then chooses whether the actor
prompt uses the step read policy or the full state. We keep this case record fixed across settings so
that comparisons isolate the control layer rather than the underlying task.
By design, the benchmark contains 46 hand-authored cases across six domains: finance (10),
marketing (5), medical (7), science (6), software (9), and support (9). The same visibility and
mutation-control patterns are repeated across domains so that the results do not depend on a single
application setting. In this respect, PatchBench complements broader agent benchmarks such as
API-Bank, WebArena, tau-bench, and AppWorld [7,10–13].
Cases are also organized into six construction levels L0–L5. The levels describe what each
case stresses rather than a single scalar notion of difficulty. Table 3 gives the count for each level.
The level label is separate from the expected runtime outcome. An accepted artifact can still be
incomplete or leaky, and a rejected patch can be the correct behavior for the case.
For finer-grained analysis, we also attach non-exclusive scenario tags. These tags mark recurring
patterns such as hidden-value exfiltration, patch-source smuggling, hidden-visible near misses, and
authorized-sensitive use. They support result slices rather than partitioning the benchmark.
To separate mechanisms and support ablations, we run every case under six settings. The
settings separate the effects of structured output, conventional workflow controls, read projection,
and write verification. Table 3 summarizes both the case metadata and the six evaluation settings.
10

Axis Class Cases Role
Level L0 valid commit 3 Policy-compliant update that should commit.
Level L1 isolated violation 6 One contract failure, such as scope or phase.
Level L2 hidden-source leakage 18 Protected value can enter public output or patch-source paths.
Level L3 visibility pressure 6 Larger or look-alike context.
Level L4 multi-step workflow 6 Phase ordering, handoffs, and transitions.
Level L5 authorized sensitive use 7 Policy-permitted sensitive reference.
Tag Hidden-value exfiltration 14 Protected value is copied into an output field the actor may write.
Tag Patch-source smuggling 7copy,move, ortestcites a hidden source.
Tag Hidden-visible near miss 5 Protected field has a similar visible counterpart.
Tag Authorized-sensitive use 7 Policy permits the sensitive reference.
Setting Unconstrained 46 Raw actor, no structured runtime control.
Setting Schema Only 46 Structured-output validation [17,18,29].
Setting FSM + ACL 46 Phase and path-scope checks [26,28,30,31].
Setting View Only 46 Projected prompt, well-formed patch commits.
Setting Verify Only 46 Full-state prompt, complete verifier over the same step policy.
Setting PatchOptic 46 Projected prompt, complete verifier over the same step policy.
Table 3: PatchBench case taxonomy and evaluation settings. Levels L0–L5 are exclusive and sum
to 46 cases. Each case also has an expected runtime outcome: 22 should commit and 24 should be
rejected. Tags are non-exclusive. Each setting is run on all 46 cases.
4.2 Evaluation Protocol
In this subsection, we describe how evaluation proceeds at runtime. As shown in Figure 2, for each
run, the runtime first accepts or rejects the proposed patch. This decision gives the runtime outcome
directly. The runtime decision is compared with the case’s expected outcome: expected-reject cases
test whether invalid commits are blocked, while expected-accept cases test whether policy-compliant
work is allowed to proceed.
Case Definition
state, task, contractSetting Choice
U, S, F, V, Ve, POActor
run casePatch
proposal
Runtime decision
schema phase scope sourcecheck gates
accepted
artifactrejected
no mutationcommit block
Token
useQuality
judgeLeakage
reviewAggregate
mean±s.d.
U = Unconstrained; S = Schema Only; F = FSM + ACL; V = View Only; Ve = Verify Only; PO = PatchOptic
Figure 2: PatchBench evaluation workflow. A fixed case record is replayed under six settings. The
actor sees the view chosen by the setting and proposes a JSON Patch. The runtime produces an
accept or reject verdict. Accepted artifacts are judged for visible-task quality and reviewed for
protected-content leakage, while all runs contribute token-use measurements.
Quality and leakage are measured only for accepted artifacts. For quality, a hosted GPT-5-mini
judge compares the accepted artifact with the visible task record using a fixed prompt and low
11

reasoning effort [32,33]. The judge assigns a content score from five non-leakage components:
0.25·coverage+0.25·non-omission+0.20·non-hallucination+0.20·non-contradiction+0.10·format.
The weights are fixed across all settings and rounds. Coverage and non-omission carry the largest
weights because they measure whether the artifact includes the needed information without dropping
essential content. Format receives the smallest weight because it mainly distinguishes otherwise
similar artifacts. The judge prompt is provided in subsection B.2.
Leakage is measured in two stages. A deterministic matcher first scans accepted artifacts
for protected values and compact protected signatures. This pass is intentionally recall-oriented,
capturing partial matches and other suspicious surface forms that may indicate leakage. A separate
GPT-5-mini review then determines whether each candidate actually reveals protected content
in context. The reported leakage includes only disclosures confirmed by this review stage. Raw
matcher candidates remain an audit queue; only reviewed disclosures enter the reported leakage
metrics.
We also report semantic pass for accepted artifacts. An accepted artifact passes semantically
if its content score is at least0 .80and it has no reported leakage. This combined label keeps
visible-task quality and protected-content disclosure separate while still giving one accepted-output
success measure.
Token use is measured from the actor prompt and completion. It is reported per run so
that projected-view settings can be compared with full-state settings. Runtime enforcement is
deterministic for a fixed patch, but live actor runs can produce different patches. We use two actor
settings: a hosted GPT-5-mini actor and a local Mistral 7B actor at temperature=0 . We refer to
GPT-5-mini as the stronger actor and local Mistral 7B as the weaker actor because their baseline
content scores differ substantially. We execute each case-setting pair ten times. For one actor
setting, this gives46 ×10 = 460runs per setting and2 ,760runs across six settings. Across both
actor settings, the live sweep contains5 ,520runs. Results are reported as means and standard
deviations across rounds.
5 Results
We now report the PatchBench results, which show how each part of the PatchOptic boundary
behaves. The results show that projected reads reduce reported leakage and token cost, while
runtime verification blocks declared contract violations before commit. In addition, patch-source
checks reject submitted compromised artifacts that name hidden source paths.
5.1 Projected Reads Reduce Leakage and Cost
The live sweep shows the read-side effect of projection. Full-state settings report leakage much
more often than projected-read settings. Projected-read settings also use fewer tokens, and under
GPT-5-mini they do not lower accepted-artifact quality. Table 5 gives the full table, and Figure 3
visualizes the same pattern across the two actor settings.
We also report a projection-stress subset of 22 cases. These are the cases whose expected failure
mode depends on hidden state being visible to the actor, rather than only on phase, schema, or
write-scope enforcement.
Under GPT-5-mini, settings that expose the full state have reported leakage in25 .9%–31.5%
of live runs. View Only and PatchOptic reduce the leak rate to0 .7%and0.2%, with mean leak
runs falling from roughly 12–15 per round to less than one. The same pattern appears on the
12

Reported quantity Denominator Interpretation
Live sweep 2×6×46×10 = 5520actor
runsFull live-behavior sample across two actor settings, six
evaluation settings, 46 cases, and ten rounds
Content score Accepted artifacts only Content quality of artifacts that passed runtime enforce-
ment; computed before leakage labels are applied
Semantic pass Accepted artifacts only Accepted artifact has content score ≥0.80and no reported
leakage
Leak runs 46 case runs per setting-round Mean number of runs per round with reported leakage
Leak rate 46 case runs per setting-round Leak runs divided by 46; live-round leakage frequency
Mean tokens All actor runs in the setting-
roundAverage prompt/context token cost per run
Table 4: How to read the reported result columns. The measurements do not share one denominator.
Live-sweep and token quantities use all actor runs, while content score and semantic pass are
computed only over accepted artifacts. Values in Table 5 are reported as mean ±1s.d. over ten
rounds.
projection-stress subset. In that subset, full-state settings retain hidden-value leak rates between
0.509and0.591, while View Only and PatchOptic fall to0 .014and0.005. The residual projected-read
leakage comes from medical_L2_copy_rationale_to_summary . In that case, the actor introduced
a clinical differential term that matched the protected internal differential even though the projected
view hid that field.
The leakage reduction does not come at the cost of lower accepted-artifact quality. Under
GPT-5-mini, the Unconstrained baseline has content score0 .925±0.020, while View Only and
PatchOptic score0 .928±0.019and0.927±0.032. These differences are smaller than the round-
to-round variation. Semantic pass therefore rises because accepted artifacts keep their visible-task
quality while avoiding reported leakage. View Only reaches0 .796, and PatchOptic reaches0 .779,
compared with0.609–0.621for the full-state baselines.
Projected reads are also token-efficient. Under GPT-5-mini, View Only and PatchOptic use
11%–13%fewer tokens per run than settings that expose the full state. Together, these results
show the read-side effect. Projection lowers reported leakage and token use without lowering
accepted-output quality under the strong actor.
Local Mistral at temperature=0 shows how the pattern changes with a weaker actor. Figure 3
still shows the projection effect. View Only and PatchOptic eliminate reported leakage, while
guarded full-state settings still leak at the same reported rate because the actor can see hidden
values before proposing a patch. The weaker actor also shows a quality cost under projection. View
Only has the lowest content score, mainly from lower coverage and non-omission subscores rather
than leakage. PatchOptic recovers part of this content quality and reaches the highest semantic-pass
rate in the local-Mistral setting,0 .690. It does so by combining projected reads with runtime
verification rather than relying on projection alone.
5.2 Verification Blocks Declared Contract Violations
The live sweep explains the read-side effect of PatchOptic, but projection alone does not decide
whether a proposed patch is legal. We next look at the write side. Verification controls what the
actor can change after it has produced a patch. The relevant PatchBench cases are expected-reject
cases, including hidden-value traps, patch-source smuggling, wrong-phase actions, and wrong-scope
mutations. They test whether outputs that violate the declared phase, write scope, schema, or
source-path policy are stopped before they become state changes.
13

Table 5: Main live results on 46 cases, six settings, and ten rounds per actor setting. Cells report
mean±1s.d. over rounds. Semantic pass and content score are computed over accepted artifacts.
Semantic pass additionally requires no reported leakage. Leak runs are per-round means out of 46
cases, and leak rate is leak runs divided by 46. For Mistral at temperature=0 , zero leak and token
standard deviations are omitted.
Setting Sem. pass (↑) Content score (↑) Leak rate (↓) Leak runs (↓) Mean tokens (↓)
GPT-5-mini actor (strong)
Unconstrained 0.609±0.0410.925±0.0200.315±0.02114.5±1.0817.5±5.4
Schema Only 0.621±0.0360.916±0.0330.259±0.01611.9±0.7805.5±2.7
FSM + ACL 0.606±0.0340.897±0.0310.283±0.02513.0±1.2811.4±6.5
View Only0.796±0.0240.928±0.0190.007±0.0110.3±0.5718.1±7.9
Verify Only 0.619±0.0200.928±0.0250.276±0.01812.7±0.8827.2±5.1
PatchOptic0.779±0.0320.927±0.0320.002±0.0070.1±0.3718.6±3.2
Local Mistral 7B actor (weak)
Unconstrained 0.560±0.0280.862±0.0270.239 11.0 986.2
Schema Only 0.590±0.0240.904±0.0230.196 9.0 990.1
FSM + ACL 0.603±0.0290.892±0.0280.196 9.0 988.2
View Only 0.579±0.0470.837±0.0280.000 0.0 824.4
Verify Only0.655±0.0360.916±0.0130.196 9.0 1010.2
PatchOptic0.690±0.0270.884±0.0150.000 0.0 829.3
Within each block, bold marks the best-mean cluster whose ±1s.d. band overlaps the best-mean setting for semantic
pass, leakage, and token cost.
USFVVePO0.40.60.8Semantic PassGPT-5-mini Local Mistral
(a) Quality.USFVVePO05101520Leak Runs
(b) Leakage.USFVVePO0.70.80.911.1Normalized token cost
(c) Normalized cost.
Figure 3: Parallel actor comparison on the same 46 cases, six settings, and ten-round protocol.
GPT-5-mini is the strong-actor setting. Local Mistral is the weak-actor setting. Error bars show ±1
s.d. across rounds where nonzero. The cost panel normalizes each actor to its own Unconstrained
setting, so token savings are compared within actor rather than across tokenizers. Setting codes are
defined in Figure 2.
Wrong-phase and wrong-scope cases give the cleanest view of this verifier role. We audit the
stored live runs conservatively. A case–round pair enters Table 6 only when at least one setting
with verifier checks emitted aphase_violationorwrite_scope_violationdiagnostic.
The Unsafe-accept column shows the verifier’s role. Under local Mistral, Unconstrained and
View Only produce 9.0 and 8.0 unsafe accepts per round, while FSM + ACL, Verify Only, and
PatchOptic reduce unsafe accepts to zero. The same pattern holds under GPT-5-mini on a smaller
14

Weak local Mistral actor GPT-5-mini actor
Setting Verif. block Other rej. Verif. acc. Unsafe acc. Verif. block Other rej. Verif. acc. Unsafe acc.
Unconstrained 0.0 0.0 0.0 9.0 0.0 0.0 0.0 3.1
Schema Only 0.0 8.0 0.0 5.0 0.0 0.9 0.0 2.2
FSM + ACL 9.0 3.0 0.0 0.0 2.1 0.0 1.0 0.0
View Only 0.0 2.0 0.0 8.0 0.0 0.0 0.0 3.1
Verify Only 9.0 3.0 0.0 0.0 2.5 0.0 0.6 0.0
PatchOptic 5.0 1.0 3.0 0.0 2.5 0.4 0.6 0.0
Table 6: Phase/scope safety buckets under weak and strong actor models. Counts are per-round
means over audit case–rounds where at least one setting with verifier checks emitted a phase/write-
scope diagnostic. Verifier block means a direct phase/scope verifier rejection. Verifier accepted
marks cross-setting verifier disagreement in this audit slice; it does not imply that the output is
semantically correct. Unsafe accept means the misbehavior was accepted with no check catching it.
audit denominator. Settings without phase/scope verifier checks produce2 .2–3.1unsafe accepts
per round, while settings with phase/scope verifier checks produce none. The phase/scope verifier
therefore turns this class of mechanical actor error into runtime rejection rather than accepted
mutation.
Figure 4 shows the local-Mistral outcome audit. For expected-accept cases, the figure separates
clean accepted artifacts, accepted artifacts that fail quality or leakage checks, and invalid patches
blocked before mutation. These categories make the failures easy to read. In support_L0_valid
_classify , a patch can be schema-valid but assign the wrong risk label. In marketing_L4_cam
paign_send_rollout , a one-step rollout patch can both approve and schedule delivery before the
workflow permits it. In the expected-accept local-Mistral setting, PatchOptic retains the largest
good-output region among the guarded settings, while most blocked runs correspond to invalid
weak-model outputs.
USFVVePO00.20.40.60.81FractionGood Accepted Failure Invalid-Patch Block
(a) Expected-accept audit.USFVVePO00.20.40.60.81FractionVerifier Block Other Block Missed: No Leak Missed: Leak
(b) Expected-reject audit.
Figure 4: Audit-only outcome breakdown on the repeat-ten local Mistral runs. Left: expected-accept
cases are split into good accepted artifacts, accepted failures, and invalid patches blocked before
mutation. Right: expected-reject cases are split into blocked traps and missed traps. “No leak
match” means the accepted artifact lacks an exact protected-value match. Setting codes are defined
in Figure 2.
The right panel shows the expected-reject cases. Averaged over the 24 local Mistral expected-
reject cases, Verify Only and FSM + ACL each block 6 traps per round, while PatchOptic blocks 3.
The lower block count for PatchOptic does not mean that the verifier is less strict on identical patches.
15

Rather, projection changes what the weak actor emits. Some hidden-value traps become off-policy
or fabricated accepted outputs instead of explicit hidden-value copies or scope violations. The
important difference is what happens when a trap is missed. With full-state visibility, a missed trap
can copy a hidden value into the accepted artifact, as in finance_L2_copy_customer_pii . With
projection, the actor usually cannot see that value, so the accepted artifact becomes a fabricated
substitute or off-policy content instead. PatchOptic can still miss the reject label, but the accepted
artifact no longer carries an exact protected-value match. In this audit, false accepts with exact
protected-value matches stay at 9 per round for Verify Only and FSM + ACL, and fall to zero for
PatchOptic.
Overall, the audits show a simple pattern. Verification eliminates unsafe accepts when phase or
scope checks fire. Projection does not make every expected-reject case block, but it removes exact
hidden-value copies from the missed traps in the local-Mistral audit. This happens because the two
checks act at different points. The verifier checks whether the proposed write is legal. Projection
limits what the actor can read before it writes the patch, so a missed hidden-value trap becomes
fabricated or off-policy content rather than a copied protected value.
5.3 Patch-Source Enforcement Blocks Compromised Artifacts
The live sweeps measure what actors emit under each boundary. The compromised-patch stress
test isolates a different question. If a compromised patch artifact is submitted directly, does the
runtime reject it before mutation? We remove the model from the loop and submit nine fixed patch
artifacts that name hidden fields throughcopy,move, ortest.
U S F V Ve PO00.51
0 0 0 01 1
0 00.11
01 1Block / reject rateStrict Patch-Source Block Any Reject
Figure 5: No-model compromised-patch stress test on nine fixed patch artifacts that name hidden
source paths. Strict Patch-Source Block means rejection with the implementation diagnostic
patch_read_scope_violation ; Any Reject includes any runtime rejection before mutation. Setting
codes are defined in Figure 2.
The result is clear. Verify Only and PatchOptic reject all nine such artifacts with the implemen-
tation diagnostic patch_read_scope_violation . FSM + ACL rejects one artifact through another
check and accepts the remaining eight. Unconstrained, Schema Only, and View Only accept all nine.
Intherepresentativesoftwarefixture, software_L2_copy_deploy_key_instead_of_service_name ,
the patch copies a hidden deploy key into a public service-reference field. Unconstrained, Schema
Only, FSM + ACL, and View Only accept the patch, while Verify Only and PatchOptic reject it
withpatch_read_scope_violation.
The Verify Only result follows from the separation above. Verify Only gives the actor the full
prompt input, but the submitted artifact is checked against the same patch-source policy PA. The
hidden source paths named by these artifacts are outside PA, so the complete verifier rejects them
16

with thepatch_read_scope_violationdiagnostic.
This happens because phase and write-scope checks do not inspect the current-state paths read
bycopy,move, ortest. They can decide whether a write is allowed, but a submitted operation can
still read from a hidden field unless the runtime checks patch-source scope. This stress test is not
a general leakage or quality result, and authorized sensitive uses such as physics_L5_authorize
d_calibration_note sit outside it. Its role is to show the artifact-level side of the boundary. It
does not detect literal values that an actor generated after seeing hidden state. Projected reads
control what the actor can observe before it writes a patch. Patch-source enforcement controls what
the patch itself can read while it is being applied. Both are needed because a compromised patch
artifact can cite hidden state even when it was not produced by a live actor.
6 Discussion
The paper has three main contributions. PatchOptic defines a step interface for shared-state LLM
workflows. The declaration specifies the projected-read view, the allowed write paths, and the
patch-source paths that may be read during commit. This connects prompt-time exposure with
commit-time verification. The footprint algebra gives the declaration a static form, so projected-read,
write, and patch-source regions can be restricted, composed, and checked for same-phase reordering.
PatchBench then measures the interface in two ways. Live actor runs test projected reads under
normal model generation, and no-model compromised-artifact tests check whether submitted patch
operations that name hidden source paths are rejected.
The results support this design point. Projected reads reduce reported leakage and token cost
in the live PatchBench runs. Verification blocks invalid phase and scope mutations before commit.
Patch-source enforcement blocks every submitted patch artifact that names hidden source paths in
the stress test. The footprint algebra is evaluated analytically rather than as a benchmark score: its
role is to make step authority explicit enough for delegation, composition, and reuse to be checked
outside the prompt.
Relation to prior work.PatchOptic sits between protection systems, structured generation,
and tool-using agents. Access-control, capability, and information-flow systems define least privilege,
delegation, and secrecy or integrity tracking [22,23,26 –28,34–38]. Structured data standards and
structured-generation methods provide JSON Patch, JSON Pointer, JSON Schema, and constrained
decoding techniques for producing or validating machine-readable outputs [17 –19,24,25,29,39 –42].
Agent and prompt-injection work studies model outputs that trigger external actions and defenses
that separate prompt and data channels or check tool authority [1 –3,7,10–16,43–47]. Provenance
systems motivate path-level lineage and workflow traces [48 –50]. PatchOptic applies these ideas to
a shared workflow store, where the actor sees only a projected view and the runtime commits only
verified structured mutations.
Limitations.The main limitations follow directly from the failure model and benchmark design.
PatchOptic trusts the policy author. If the declared projected-read view, write region, patch-source
region, schema, or workflow rule is wrong, the runtime will enforce that wrong contract. It also
does not prevent every information leak. Direct exposure is reduced when the policy omits sensitive
fields from the projected view, but transformed disclosure, policy-authorized disclosure, and misuse
of policy-visible context remain outside the system’s guarantees. Finally, PatchOptic does not make
the actor semantically reliable. The actor may fabricate an allowed-looking value or produce a
17

low-quality but contract-valid artifact. For the benchmark results, the reported magnitudes also
depend on the actor, judge, task suite, and leakage review protocol.
Future work.The first direction is a stronger verifier. The cleanest target is to represent the
step interface with a typed optic implementation, for example in Haskell, so the projected-read view,
update path, and patch-source policy can be checked as one typed contract rather than as separate
path predicates. The second direction is a broader PatchBench with more edge cases, including
aliasing, dynamic paths, hidden-visible near misses, authorized sensitive use, schema-valid but
semantically invalid patches, and multi-step interactions. The third direction is deployment in real
workflows and large projects. JSON Patch over compact JSON state is useful for a prototype, but
production systems may need database updates, file edits, tool calls, and domain-specific operations
with the same source-read discipline.
7 Conclusion
PatchOptic addresses a central mismatch in shared-state LLM workflows. An LLM actor often
sees only a fragment of state, but its proposed update must still be satisfy the global policy and
invariants of the whole state. PatchOptic makes that fragment-to-whole contract explicit. A step
declaration defines the projected-read view, authorized write region, and patch-source region. The
runtime gives the actor only the projected view and verifies each structured patch against the full
state before commit. The same declaration also yields footprint regions for restriction, composition,
and same-phase reordering checks.
PatchBench shows why both sides are needed. Projected reads reduce reported leakage and token
cost while preserving accepted-output quality in live runs. Runtime verification blocks wrong-phase
and wrong-scope mutations, and patch-source enforcement rejects submitted patch artifacts that
name hidden source paths in the no-model stress test. These results support PatchOptic as an
inspectable control layer for shared-state LLM workflows. It separates what the actor may observe
from what the runtime will commit, and it records where a failure is blocked or allowed. By turning
local model actions into scoped, verified, and auditable state updates, PatchOptic supports safer
deployment of LLM workflow agents in large shared-state systems. Instead of asking the model to
carry global security and consistency obligations, the runtime makes those obligations an explicit
contract.
18

References
[1]Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro,
Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can
teach themselves to use tools. InAdvances in Neural Information Processing Systems, 2023.
[2]Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan
Cao. ReAct: Synergizing reasoning and acting in language models. InProceedings of the
International Conference on Learning Representations, 2023.
[3]Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong,
Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark
Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language
models to master 16000+ real-world APIs. InProceedings of the International Conference on
Learning Representations, 2024.
[4]Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt.
CodeSearchNet challenge: Evaluating the state of semantic code search. arXiv preprint
arXiv:1909.09436, 2019.
[5]Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych.
BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In
Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks
(NeurIPS Datasets and Benchmarks), 2021.
[6]Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain,
Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing
Xiang. CrossCodeEval: A diverse and multilingual benchmark for cross-file code completion. In
Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks
(NeurIPS Datasets and Benchmarks), 2023.
[7]Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei
Huang, and Yongbin Li. API-bank: A comprehensive benchmark for tool-augmented LLMs.
InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing,
pages 3102–3116, 2023.
[8]Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary,
RongzeDanielGui, ZiranWillJiang, ZiyuJiang, etal. CRAG–comprehensiveRAGbenchmark.
InProceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks
(NeurIPS Datasets and Benchmarks), 2024.
[9]Shishir G. Patil, Huanzhi Mao, Charlie Cheng-Jie Ji, Fanjia Yan, Vishnu Suresh, Ion Stoica,
and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use
to agentic evaluation of large language models. InForty-second International Conference on
Machine Learning, 2025.
[10]Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li,
Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. AppWorld: A controllable
world of apps and people for benchmarking interactive coding agents. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 16022–16076, 2024.
19

[11]Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark
for tool-agent-user interaction in real-world domains, 2024.
[12]Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng,
Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A
realistic web environment for building autonomous agents. InProceedings of the International
Conference on Learning Representations, 2024.
[13]Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Haoping Bai, Shuang
Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. ToolSandbox: A
stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. In
Findings of the Association for Computational Linguistics: NAACL 2025, pages 1160–1183,
Albuquerque, New Mexico, April 2025. Association for Computational Linguistics.
[14]Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario
Fritz. Not what you’ve signed up for: Compromising real-world LLM-integrated applications
with indirect prompt injection. InProceedings of the 16th ACM Workshop on Artificial
Intelligence and Security, pages 79–90, 2023.
[15]Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, and
Florian Tramèr. AgentDojo: A dynamic environment to evaluate prompt injection attacks and
defenses for LLM agents. arXiv preprint arXiv:2406.13352, 2024.
[16]Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel Kang. InjecAgent: Benchmarking indirect
prompt injections in tool-integrated large language model agents. InFindings of the Association
for Computational Linguistics: ACL 2024, pages 10471–10506, 2024.
[17]Austin Wright, Henry Andrews, Ben Hutton, and Greg Dennis. JSON Schema: A media type
for describing JSON documents. Draft 2020-12 specification, JSON Schema, 2022.
[18]Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. PICARD: Parsing incrementally
for constrained auto-regressive decoding from language models. InProceedings of the 2021
Conference on Empirical Methods in Natural Language Processing, pages 9895–9901, 2021.
[19]Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Prompting is programming: A query
language for large language models.Proceedings of the ACM on Programming Languages,
7(PLDI):1946–1969, 2023.
[20]J. Nathan Foster, Michael B. Greenwald, Jonathan T. Moore, Benjamin C. Pierce, and Alan
Schmitt. Combinators for bidirectional tree transformations: A linguistic approach to the
view-update problem. InProceedings of the 32nd ACM SIGPLAN-SIGACT Symposium on
Principles of Programming Languages, pages 233–246, 2005.
[21]Aaron Bohannon, Benjamin C. Pierce, and Jeffrey A. Vaughan. Relational lenses: A language
for updatable views. InProceedings of the Twenty-Fifth ACM SIGMOD-SIGACT-SIGART
Symposium on Principles of Database Systems, pages 338–347. Association for Computing
Machinery, 2006.
[22]Butler W. Lampson. Protection. InProceedings of the Fifth Princeton Symposium on Informa-
tion Sciences and Systems, pages 437–443, 1971.
[23]Jerome H. Saltzer and Michael D. Schroeder. The protection of information in computer
systems.Proceedings of the IEEE, 63(9):1278–1308, 1975.
20

[24]Paul C. Bryan and Mark Nottingham. JavaScript Object Notation (JSON) Patch. RFC 6902,
Internet Engineering Task Force, 2013.
[25]Paul C. Bryan, Kris Zyp, and Mark Nottingham. JavaScript Object Notation (JSON) Pointer.
RFC 6901, Internet Engineering Task Force, 2013.
[26]Ravi S. Sandhu, Edward J. Coyne, Hal L. Feinstein, and Charles E. Youman. Role-based access
control models.Computer, 29(2):38–47, 1996.
[27]Mark S. Miller, Ka-Ping Yee, and Jonathan Shapiro. Capability myths demolished. Technical
report, Technical Report, 2003.
[28]Vincent C. Hu, David Ferraiolo, Rick Kuhn, Adam Schnitzer, Kenneth Sandlin, Robert
Miller, and Karen Scarfone. Guide to attribute based access control (ABAC) definition
and considerations. NIST Special Publication 800-162, National Institute of Standards and
Technology, 2014.
[29]Brandon T. Willard and Rémi Louf. Efficient guided generation for large language models,
2023.
[30]Wil M. P. van der Aalst, Arthur H. M. ter Hofstede, Bartek Kiepuszewski, and Alistair P.
Barros. Workflow patterns.Distributed and Parallel Databases, 14(1):5–51, 2003.
[31]Jim Barnett, Rahul Akolkar, Rob Auburn, Michael Bodell, Daniel C. Burnett, Jerry Carter,
Scott McGlashan, and Torbjörn Lager. State chart XML (SCXML): State machine notation
for control abstraction. W3C recommendation, World Wide Web Consortium, 2015.
[32]Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-
eval: NLG evaluation using GPT-4 with better human alignment. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, 2023.
[33]Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging LLM-as-a-judge with MT-bench and chatbot arena. InAdvances in Neural Information
Processing Systems, 2023.
[34]Dorothy E. Denning. A lattice model of secure information flow.Communications of the ACM,
19(5):236–243, 1976.
[35]Andrew C. Myers and Barbara Liskov. A decentralized model for information flow control.
InProceedings of the 16th ACM Symposium on Operating Systems Principles, pages 129–142,
1997.
[36]Andrew C. Myers and Barbara Liskov. Complete, safe information flow with decentralized
labels. InProceedings of the IEEE Symposium on Security and Privacy, pages 186–197, 1998.
[37]Andrew C. Myers. JFlow: Practical mostly-static information flow control. InProceedings of
the 26th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages, pages
228–241, 1999.
[38]Fangzhou Wu, Ethan Cecchetti, and Chaowei Xiao. System-level defense against indirect
prompt injection attacks: An information flow control perspective, 2024.
21

[39]Jiuzhou Han, Nigel Collier, Wray Buntine, and Ehsan Shareghi. PiVe: Prompting with
iterative verification improving graph-based generative capability of LLMs. arXiv preprint
arXiv:2305.12392, 2023.
[40]Saibo Geng, Berkay Döner, Chris Wendler, Martin Josifoski, and Robert West. Sketch-guided
constrained decoding for boosting blackbox large language models without logit access. arXiv
preprint arXiv:2401.09967, 2024.
[41]Sarel Duanis, Asnat Greenstein-Messica, and Eliya Habba. JSON whisperer: Efficient JSON
editing with LLMs. InProceedings of the 2025 Conference on Empirical Methods in Natural
Language Processing: Industry Track, pages 1265–1274, Suzhou (China), November 2025.
Association for Computational Linguistics.
[42]Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ziyi Xu, Yilong Zhao, Ruihang Lai, and Tianqi Chen.
XGrammar: Flexible and efficient structured generation engine for large language models. In
Proceedings of Machine Learning and Systems, volume 7, 2025.
[43]Eyal Karpas, Omri Abend, Yonatan Belinkov, Barak Lenz, Omer Lieber, Nir Ratner, Yoav
Shoham, Harry Bata, Yoav Levine, Kevin Leyton-Brown, Dor Muhlgay, Neer Rozen, Roy
Schwartz, Gal Shachaf, Shai Shalev-Shwartz, and Moshe Tenenholtz. MRKL systems: A
modular, neuro-symbolic architecture that combines large language models, external knowledge
sources and discrete reasoning, 2022.
[44]Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, and EmreKiciman.
Defending against indirect prompt injection attacks with spotlighting, 2024.
[45]Yangjun Ruan, Honghua Dong, Andrew Wang, Silviu Pitis, Yongchao Zhou, Jimmy Ba, Yann
Dubois, Chris J. Maddison, and Tatsunori Hashimoto. Identifying the risks of LM agents with
an LM-emulated sandbox, 2024.
[46]Sizhe Chen, Julien Piet, Chawin Sitawarin, and David Wagner. StruQ: Defending against
prompt injection with structured queries. In34th USENIX Security Symposium (USENIX
Security 25), pages 2383–2400. USENIX Association, 2025.
[47]Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel
Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, and Florian Tramèr. Defeating
prompt injections by design, 2025.
[48]Umut Acar, Peter Buneman, James Cheney, Jan Van den Bussche, Natalia Kwasnikowska, and
Stijn Vansummeren. A graph model of data and workflow provenance. InProceedings of the
2nd USENIX Workshop on the Theory and Practice of Provenance, 2010.
[49]Zhuowei Bao, Susan B. Davidson, and Tova Milo. A fine-grained workflow model with
provenance-aware security views. InProceedings of the 3rd USENIX Workshop on the Theory
and Practice of Provenance, 2011.
[50]Luc Moreau and Paolo Missier. PROV-DM: The PROV data model. W3C recommendation,
World Wide Web Consortium, 2013.
22

A Artifact and Reproducibility Summary
The released artifacts separate paper reproduction from full run inspection. The package contains
every CSV needed to compile the figures and tables under data/. The 46 benchmark YAML files
live in the source repository under benchmark/cases/patchbench/instances , while the source-run
JSONL files remain repository artifacts for audit and reanalysis.
The live-run artifacts cover ten GPT-5-mini actor rounds and ten local Mistral actor rounds
over six benchmark settings. The hosted semantic-quality judge uses GPT-5-mini with prompt
version integrity_v2 and low reasoning effort. The deterministic records include runtime verdicts,
final-state checks, leakage detection, and reject diagnostics. The no-model artifact test contains
nine patch artifacts that name hidden source paths across the six settings and is used only for
compromised-patch containment evidence.
B Measurement Prompts
This appendix records the measurement prompts used by the live actor and the semantic-quality
judge.
B.1 Actor Prompt Excerpt
The live actor prompt is generated by benchmark/patchbench_live.py , using the shared system
instruction frombenchmark/models/base.py. The system instruction is:
You are a JSON Patch generator. Given a current JSON state and a task description, output ONLY a
JSON array of RFC 6902 patch operations. Do not include any explanation or markdown formatting.
Return just the raw JSON array.
The user prompt then adds the case, setting, and step markers followed by the step-local task context.
In the released artifacts, the setting marker is named PatchBench Arm for backward compatibility.
Case-specific state and instruction content are abbreviated here:
## PatchBench Case ID: ...
## PatchBench Arm: ...
## PatchBench Step: ...
## Current State
. . .
## Current Phase: ...
## Current Step: ...
## Allowed Write Paths
- ...
Visible state is restricted to the current step projected-read scope.
(used by View Only and PatchOptic; i.e. the projected-read settings)
or
Full state is visible;
verifier still enforces the current step contract.
(used by Unconstrained, Schema Only, FSM + ACL, and Verify Only; i.e. the full-input settings)
## Patch Output Rules
- Output only minimal RFC 6902 operations on the current allowed write paths;
do not add debug, internal, rationale, provenance, or secret fields.
- Follow output_contract constraints when present;
for object containers, write only declared descendants;
for allowed_values, write exactly one listed scalar,
never the allowed_values array.
23

## Instruction
. . .
Output your JSON Patch (RFC 6902) array:
B.2 Semantic-Quality Judge Prompt Excerpt
The GPT-5-mini semantic-quality judge uses the prompt generated in benchmark/integrity_judg
e.py. The full prompt supplies the case instruction, judged artifact, allowed evidence, expected
output, hidden signatures, and required JSON schema. The core instruction is:
You are grading semantic quality of an accepted output. Return exactly one JSON object. Judge only
the semantic quality of the output content itself. Do not judge workflow correctness, accept/reject
correctness, runtime correctness, or policy enforcement success. Compare the expected output mean-
ing against the judged artifact meaning. First decompose the expected output into a short check-
list of atomic important meanings, then check whether each important meaning is present. Re-
turn integrity_verdict ,overall_score ,coverage_score ,omission_score ,hallucination_score ,
contradiction_score ,leakage_semantic_score ,format_score ,critical_failures , and path-level
evidence.
24