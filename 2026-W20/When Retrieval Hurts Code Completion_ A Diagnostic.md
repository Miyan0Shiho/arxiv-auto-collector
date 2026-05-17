# When Retrieval Hurts Code Completion: A Diagnostic Study of Stale Repository Context

**Authors**: Haojun Weng, Qianqian Yang, Hao Fu, Haobin Pan, Xinwei Lv

**Published**: 2026-05-14 07:18:30

**PDF URL**: [https://arxiv.org/pdf/2605.14478v1](https://arxiv.org/pdf/2605.14478v1)

## Abstract
Context: Retrieval-augmented code generation relies on cross-file repository context, but retrieved snippets may come from obsolete project states.
  Objectives: We study whether temporally stale repository snippets act as harmless noise or actively induce current-state-incompatible code.
  Methods: We conduct a controlled diagnostic study on a curated 17-sample set of production-helper signature changes from five Python repositories. For each sample, we compare current-only, stale-only, no-retrieval, and mixed current/stale retrieval conditions under prompts that hide commit freshness and expected current signatures.
  Results: Under neutralized prompts, stale-only retrieval induces stale helper references on 15/17 Qwen2.5-Coder-7B-Instruct samples and 13/17 gpt-4.1-mini samples, corresponding to 88.2 and 76.5 percentage-point increases over current-only retrieval. No retrieval produces zero stale references but only 1/17 passing completions. The two models share 75.0% Jaccard overlap among stale-triggering samples, and mixed conditions show that adding valid current evidence largely rescues stale-only failures.
  Conclusion: Temporal validity of retrieved repository context is a distinct diagnostic variable for Code RAG robustness: stale context can actively bias models toward obsolete repository state rather than merely removing useful evidence.

## Full Text


<!-- PDF content starts -->

When Retrieval Hurts Code Completion: A Diagnostic
Study of Stale Repository Context
Haojun Wenga,1,∗, Qianqian Yangb,1, Hao Fua, Haobin Pana, Xinwei Lvb
aIndependent Researcher, California, USA
bIndependent Researcher, Beijing, China
Abstract
Context:Retrieval-augmented code generation relies on cross-file repository
context, but retrieved snippets may come from obsolete project states.Objec-
tives:We study whether temporally stale repository snippets act as harmless
noise or actively induce current-state-incompatible code.Methods:We con-
duct a controlled diagnostic study on a curated 17-sample set of production-
helper signature changes from five Python repositories. For each sample,
we compare current-only, stale-only, no-retrieval, and mixed current/stale
retrieval conditions under prompts that hide commit freshness and expected
current signatures.Results:Under neutralized prompts, stale-only re-
trieval induces stale helper references on 15/17 Qwen2.5-Coder-7B-Instruct
samples and 13/17 gpt-4.1-mini samples, corresponding to 88.2 and 76.5
percentage-point increases over current-only retrieval. No retrieval produces
zero stale references but only 1/17 passing completions. The two models share
75.0% Jaccard overlap among stale-triggering samples, and mixed conditions
show that adding valid current evidence largely rescues stale-only failures.
Conclusion:Temporal validity of retrieved repository context is a distinct
diagnostic variable for Code RAG robustness: stale context can actively bias
models toward obsolete repository state rather than merely removing useful
evidence.
Keywords:Code RAG, Retrieval-augmented generation, Repository-level
∗Corresponding author. ORCID: 0009-0006-7306-4816.
Email addresses:whjwitness1019@gmail.com(Haojun Weng),546987836@qq.com
(Qianqian Yang),koi.helloai@gmail.com(Hao Fu),bayron.p27@gmail.com(Haobin
Pan),18700910021@163.com(Xinwei Lv)
1These authors contributed equally as co-first authors.arXiv:2605.14478v1  [cs.SE]  14 May 2026

code completion, Temporal validity, Diagnostic study, Software engineering
1. Introduction
Modern code assistants increasingly rely on repository context. A model
asked to complete a function rarely has enough information in the current
file alone: it may need helper signatures, local conventions, wrappers, or
cross-file dependencies. Retrieval-augmented code generation addresses this
by supplying snippets from the surrounding repository or from external code
knowledge bases. This can turn an otherwise underspecified completion task
into one the model can solve. Yet the same mechanism also creates a new
failure channel. If the retrieved snippet is plausible but temporally stale, the
model may ground on a repository state that no longer exists.
This risk is realistic in active repositories. Helper functions are renamed,
arguments are added or removed, wrapper signatures drift, and project-
internal utilities change across commits. A retriever, cache, index, or external
code search system may return a snippet from an older state of the repository.
Such a snippet is not obviously irrelevant: it comes from the same project,
often uses the same helper name, and may look locally compatible with the
current task. The question is whether code models treat this stale evidence
as harmless noise, or whether it can actively steer them toward current-state-
incompatible code.
Prior work has studied several adjacent problems. API-evolution and
version-aware code-generation studies show that models can lag behind chang-
ing software ecosystems [ 1,2]. Repository-level code RAG systems and
benchmarks study how to retrieve cross-file context for project-level comple-
tion [3,4,5,6]. Context-filtering and mitigation methods show that retrieved
chunks can be helpful, neutral, or harmful [ 7,8,9]. Hallucination and API-
misuse work studies incorrect code and invalid API use [ 10,11]. To our
knowledge, the closest related work in recent Code RAG, selective retrieval,
software-evolution, and benchmark generation does not systematically control
whether retrieved repository context comes from an old commit or the current
repository state while measuring stale-reference outcomes (see Supplementary
Material S1 for the related-work audit).
We present a controlled diagnostic study of stale repository context in
retrieval-augmented code completion. The study manipulates the temporal
validity of retrieved cross-file context while holding the local task and oracle
2

fixed. For each sample, the model sees a local wrapper task and one of five
retrieval conditions: current context only, stale context only, no retrieval,
current-then-stale mixed retrieval, or stale-then-current mixed retrieval. The
prompt deliberately avoids explicit target-state anchors: it does not tell the
model which snippet is current, does not reveal commit freshness, and does
not name the expected current helper signature. This design follows earlier
pilot evidence that strong models can self-rescue when the task itself explicitly
names the intended API or library state.
Our dataset contains 17 curated production-helper signature changes
mined from five Python repositories: click,flask,httpx,requests , and
rich. Each sample pairs a stale helper signature from a parent commit with
thecurrenthelpersignaturefromthechildcommit, andusesstaticcall-pattern
oracles to classify model outputs as current-state matches, stale references,
or fail-no-match outputs. We evaluate Qwen2.5-Coder-7B-Instruct as an
open-source code-specialist model and gpt-4.1-mini as a proprietary general-
purpose model.
The results show a strong stale-context effect. Current-only retrieval
eliminates stale references in both models, while stale-only retrieval frequently
induces calls that match obsolete helper signatures. The no-retrieval baseline
is also informative: without repository context, models mostly fail to match
either oracle rather than producing stale-state calls. Thus stale retrieval is
not merely equivalent to missing context. It redirects the error pattern from
generic inability to complete the wrapper toward historical repository-state
grounding. A sensitivity run on the pre-quality-gate sample set confirms that
the main effect is not an artifact of Qwen-based filtering.
We also find that valid current evidence presence is more important than
retrievalrankorderinthecurrentsetup. Whenbothcurrentandstalesnippets
are retrieved, stale references drop sharply relative to stale-only retrieval.
Both models show 0.0 percentage-point aggregate rank-order deltas under
the neutralized evaluation. The safe conclusion is that presence or absence of
valid current evidence is the dominant mechanism in this diagnostic setting.
This paper makes four contributions:
1.We introduce a controlled diagnostic setup for repository-level Code
RAG that isolates temporal validity of retrieved cross-file context.
2.We provide cross-model evidence on 17 production-helper samples from
five Python repositories, showing large stale-reference effects in both
Qwen2.5-Coder-7B-Instructandgpt-4.1-mini.
3

3. We identify a mechanism-level pattern: stale retrieval actively induces
stale-state references, no retrieval produces a different fail-no-match
pattern, and valid current evidence largely rescues stale-only failures.
4.We release a prompt-rendering and leakage-audit artifact, including
opaque context identifiers, forbidden-token checks over model-visible
fields, prompt dumps, and excluded preliminary runs for audit trans-
parency.
We position this as a diagnostic study, not a mitigation method. The goal
is to characterize a failure mode that future retrievers, index freshness checks,
and context filters can address.
The rest of the paper describes the mining and evaluation protocol, reports
the controlled-condition results, and discusses limitations including sample
size, signature-change scope, one-parent-commit drift, and static regex oracles.
2. Related Work
2.1. API Evolution and Version-Aware Code Generation
Recent work shows that code generation models can lag behind evolving
software ecosystems. A study of Python API evolution documents that
model outputs often reflect outdated APIs and that additional structured API
information can improve generation under changing libraries [ 1]. RustEvo2
extends the version-aware evaluation perspective to the Rust ecosystem [ 2].
EVOR similarly treats retrieval as an evolving process: instead of relying on
a fixed knowledge source, it updates retrieval queries and knowledge bases to
support code generation in settings where external knowledge changes over
time [12]. These works establish software evolution as an important source of
code-model failure, but their main variables are API/library evolution and
knowledge-base adaptation. Our study asks a narrower retrieval-level question:
when repository context is retrieved for code completion, what happens if
that context comes from an old repository state rather than the current one?
This is orthogonal to EVOR’s focus: EVOR updates the retrieval pipeline
and knowledge base, while we hold the retrieval source family fixed and vary
the historical snapshot selected as context.
2.2. Repository-Level Code RAG
Repository-level code completion has motivated a family of methods and
benchmarks that retrieve cross-file context beyond the current editing buffer.
4

RepoCoder introduced iterative retrieval and generation for repository-level
completion, using generated code to refine subsequent retrieval queries [ 3].
RepoBench and CrossCodeEval evaluate repository or cross-file completion
settings where models must use project-level context rather than isolated
function bodies [ 5,6]. More recent systems such as CodeRAG improve
repository-level retrieval through query construction, multi-path retrieval, and
reranking over structured code knowledge [ 4]. CodePlan studies repository-
level coding through planning over code edits and build/type-check feedback,
making it a close neighbor for temporal code-change settings [ 13]. SWE-
Bench++ scales repository-level software-engineering benchmarks by turning
live pull requests into executable tasks with test-time repository oracles [ 14].
Its before/after state construction is adjacent to our temporal framing, but
the manipulated variable is different: SWE-Bench++ varies the benchmark
task and repository-state oracle, whereas our study holds the local task and
oracle fixed while varying the commit of the retrieved snippet.
This line of work demonstrates that repository context can be useful and
that retrieval design matters. These systems generally evaluate retrieval
quality, completion quality, planning, or benchmark performance under a
given repository snapshot.
2.3. Context Filtering and Mitigation
A second line of work studies when retrieved context should be used, fil-
tered, or constrained. Impact-driven context filtering labels retrieved cross-file
chunks as positive, neutral, or negative for completion and shows that harm-
ful context can degrade code generation [ 7]. Repoformer selectively decides
whether to retrieve for repository-level completion [ 9], while RepoShapley
uses Shapley-style attribution to model the contribution of retrieved chunks
and filter context accordingly [ 8]. MARIN addresses API hallucination from
a different mitigation angle: it uses hierarchical dependency awareness to
reduce invalid API use in project-specific code generation [15].
These works are important because they make clear that retrieval is not
uniformly helpful. They also form a crowded area that our paper deliberately
does not claim as new: we do not propose a new context filter, reranker,
dependency-aware decoder, or mitigation method.
2.4. Hallucination and Robustness in Code Generation
Code hallucination and API misuse have been studied from benchmark,
taxonomy, and mitigation perspectives. CodeHalu investigates code halluci-
5

nations through execution-based verification [ 10]. Package hallucination work
studies nonexistent package suggestions and their supply-chain risks, which is
adjacent to but distinct from real repository-state mismatch [ 16]. API misuse
work categorizes and mitigates incorrect API use within generated code,
including misuse of existing APIs [ 11]. These studies provide the broader
robustness motivation for our stale-reference metric: generated code can be
wrong not only because a model lacks knowledge, but because it grounds on
an incompatible external signal.
2.5. Positioning of This Study
Across recent work, we found substantial coverage of API evolution,
repository-level retrieval, context filtering, and code hallucination. We there-
fore avoid broad claims such as "stale context is unstudied" or "harmful
retrieval is unexplored." To our knowledge, the narrower gap is retrieval-
temporal: the closest related studies reviewed here do not treat old-vs-
current retrieved repository context as a controlled variable and measure
stale-reference behavior under otherwise matched code-completion tasks (see
Supplementary Material S1 for the related-work audit).
We fill this gap with a controlled diagnostic study rather than a new
retrieval method. The retrieved snippets in our study are real code from
the same project lineage rather than nonexistent packages or arbitrary incor-
rect API suggestions. The manipulated variable is retrieval-time temporal
validity: current-context-only, stale-context-only, no-retrieval, and mixed
current/stale retrieval. This framing complements prior work: API evolution
studies motivate the temporal risk, repository-level RAG work motivates the
retrieval setting, context-filtering work motivates why retrieval can hurt, and
hallucination work motivates the stale-reference outcome. The contribution
is to isolate temporal validity of retrieved repository context as the diagnostic
variable.
3. Method
3.1. Study Type
We conduct a controlled diagnostic study of repository-level code RAG.
The study isolates the temporal validity of retrieved cross-file context as the
manipulated variable:
6

Role Retrieval condition
Controlcurrent_context_only
Treatmentstale_context_only
Baselineno_retrieval
Robustnessmixed_current_top1_stale_top2
Robustnessmixed_stale_top1_current_top2
The study is diagnostic rather than a mitigation method. We do not train
a model, propose a new retriever, or sell context filtering as the contribution.
The purpose is to test whether plausible but temporally stale repository
context can induce code that is incompatible with the current repository
state.
3.2. Research Questions
RQ1: Does stale repository context induce current-state-incompatible
helper calls compared with current repository context and no retrieval?
RQ2: When both current and stale evidence are present, does rank order
change the stale-reference rate?
RQ3: Does adding valid current evidence to a prompt that contains stale
evidence reduce the stale-reference rate?
RQ4: Are stale-triggering samples shared across an open-source code-
specialist model and a proprietary general-purpose model?
3.3. No-Target-Anchor Prompting Policy
Earlier pilot gates in our study showed that strong models can self-rescue
when the task explicitly names the intended API, library, or repository state.
Therefore this study uses no-target-anchor prompting.
•The prompt does not reveal whether retrieved snippets are current or
stale.
•The task does not name the expected current helper signature.
•Commit hashes and freshness labels are oracle-only metadata.
•The model-visible evidence consists of a local call-site wrapper and the
retrieved repository snippet.
7

This policy is enforced at the dataset-construction level: candidate samples
that require revealing the current state to be meaningful are excluded by the
mining and curation pipeline rather than weakening the prompting policy.
The runtime harness also performs a lightweight static check for obvious
freshness or target-reference leakage, but the main enforcement is sample
construction and review.
3.4. Sample Mining
Weminecandidatetemporal-validityexamplesfromrealPythonrepository
history. The final neutralized dataset uses five Python repositories: click,
flask,httpx,requests, andrich(see supplementary mining report).
The mining pipeline scans recent commit history for Python function
signature changes. Each candidate stores the parent commit as stale context
and the child commit as current context. Candidates are filtered to remove:
•test functions and files under test directories
•zero-delta signatures with no static-oracle power
•duplicate helper families across nearby commits
•fragile dunder or high-arity cases when they fail the quality gate
Offline mining produced 31 candidates. Curation reduced these to 19
samples, and the Qwen current-context quality gate retained 17 samples.
Because Qwen is both the quality-gate model and one of the evaluated
models, we also keep the pre-gate 19-sample set as a sensitivity check. This
separates the conditioned-on-model-capable estimate from the less filtered
estimate.
The final neutralized evaluation set contains only production/helper sam-
ples:
Repositoryclick flask httpx requests richTotal
Samples 3 3 3 3 5 17
The signature-delta distribution is 13 samples with a one-argument change,
two samples with a two-argument change, and two samples with a three-
argument change.
All retained samples are signature_change examples with one-parent-
commit drift. Rename/remove changes and larger temporal gaps are left for
follow-up work or a future expansion.
8

3.5. Retrieval Conditions
For each sample, the harness constructs five retrieval conditions. The
current_context_only condition retrieves the current helper signature from
the child commit, while stale_context_only retrieves the stale helper sig-
nature from the parent commit. The no_retrieval baseline supplies no
cross-file repository snippet and tests whether stale context induces wrong-
state references rather than merely causing generic model failure. The two
mixed conditions include both snippets: mixed_current_top1_stale_top2
placesthecurrentsnippetfirst, and mixed_stale_top1_current_top2 places
the stale snippet first.
The primary comparison is stale-only versus current-only retrieval. The
no_retrieval condition is a baseline. The mixed conditions are robustness
checks for rank-order sensitivity and current-evidence rescue, not the main
decision gate.
This is an oracle-controlled retrieval setting rather than a deployed-
retriever evaluation. The prompt intentionally asks the model to use retrieved
repository evidence because the diagnostic question is whether temporally in-
valid evidence can induce wrong-state grounding when it is treated as project
evidence. We therefore interpret the stale-only condition as a controlled
upper-bound diagnostic of context-grounded failure, not as an estimate of
stale-index prevalence in deployed systems.
3.6. Models
The main evaluation uses two models:
Model Role
QwenOpen-source code-specialist model
gpt-4.1-miniProprietary general-purpose model
HereQwenabbreviatesQwen2.5-Coder-7B-Instruct.
DeepSeek-Coder-6.7B was explored during pilot work, but it is excluded
from the main analysis because its OpenAI-compatible serving path produced
tokenizer artifacts and low current-context pass rates under this harness. This
exclusion is documented in the supplementary run logs.
9

3.7. Inference Configuration
All model calls use the same OpenAI-compatible chat-completions
client from the experiment harness. The request payload explicitly sets
temperature=0 ; the client implementation is included in the supplementary
artifact.
The harness does not explicitly set top_p,seed, ormax_tokens ; these
remain provider or vLLM endpoint defaults. Qwen runs use the local vLLM
OpenAI-compatible endpoint, while gpt-4.1-mini runs use the OpenAI API
endpoint. Request delays affect rate limiting only and are not part of the
decoding configuration.
3.8. Metrics
The primary metric is stale-reference rate:
stale-reference rate=#{outputs matching the stale helper call pattern}
n.
The primary effect is:
∆primary =SRR stale-only −SRR current-only .
For the no-retrieval baseline, we report:
∆no-retrieval =SRR stale-only −SRR no-retrieval .
This separates stale-reference errors from generic inability to complete the
wrapper without repository context.
We also report pass rate:
pass rate=#{outputs matching current patterns and not stale patterns}
n.
This matters because a model can avoid stale references while still failing
to produce the current-state call. For example, gpt-4.1-mini has a 0/17
current-context stale-reference rate in the final neutralized evaluation, but its
current-context pass rate is 15/17 rather than 17/17.
We separately report fail-no-match rate:
fail-no-match rate=#{outputs matching neither oracle}
n.
10

This captures cases where the model output does not satisfy either static
oracle. It is reported separately from stale-reference rate because it can
indicate oracle strictness or model-format drift rather than a stale-context
failure.
For mixed conditions, we report rank-order stale-reference delta:
∆rank=SRR stale-top1 −SRR current-top1 .
Finally, we compute cross-model sample overlap among stale-triggering
samples:
J=|SQwen∩SGPT|
|SQwen∪SGPT|.
3.9. Oracles
The final neutralized evaluation uses static call-pattern oracles. For each
sample, the builder generates:
•current call regex patterns from the child-commit signature
•stale call regex patterns from the parent-commit signature
•expected current solution sketch
•forbidden stale reference sketch
The oracle is intentionally conservative. Outputs that do not match either
the current or stale pattern are counted as missing current signal, not as stale
references.
3.10. Prompt-Leakage Audit and Neutralized Final Run
Before the final evaluation, we performed a prompt-leakage audit over
every model-visible prompt field: local context, retrieved-context headers,
context identifiers, task text, and optional metadata fields. The audit found
that preliminary rendered prompts used condition-specific context identifiers
that could encode freshness cues. Because this violated our no-target-anchor
policy, we excluded the preliminary run from the main analysis.
We then constructed a neutralized evaluation set in which all model-
visible context identifiers were replaced with opaque per-sample identifiers
(e.g.,ctx-sig-001-a andctx-sig-001-b ), assigned by a deterministic hash
11

so that no positional convention across samples can be learned. In the retained
17-sample set, the aidentifier holds the current snippet for 9 samples and the
bidentifier holds it for the other 8. The complete five-condition protocol was
rerun for both models. All results reported below come from this neutralized
run. The excluded preliminary run and the audit history are provided in
Appendix Appendix B for transparency and are not used as evidence for the
main claims.
3.11. Validity-Threat Summary
The design choices above make the treatment controlled but narrow.
Section 6 discusses sample size, signature-change scope, one-parent drift,
oracle-controlled retrieval, static oracles, Qwen quality-gate entanglement,
backend reproducibility, and the prompt-leakage audit disclosure. We keep
the detailed threat discussion there to avoid mixing design definition with
validity analysis.
4. Results
4.1. RQ1: Stale Context Strongly Increases Stale References
On the 17-sample neutralized evaluation set, both main models show large
stale-reference deltas:
Model Current condition Stale condition∆ primary
QwenSRR 0/17; pass 17/17 SRR 15/17; pass 0/17 88.2 pp
gpt-4.1-miniSRR 0/17; pass 15/17 SRR 13/17; pass 1/17 76.5 pp
HereQwenabbreviatesQwen2.5-Coder-7B-Instruct.
Interpretation:
When retrieved repository context is the model-visible source of
repo-state evidence, stale helper signatures can induce current-
state-incompatible calls in both Qwen2.5-Coder-7B-Instruct and
gpt-4.1-mini.
We report stale-reference delta rather than only pass-rate delta because
some outputs fail to match either the current or stale static pattern. The
gpt-4.1-mini current-context condition has two such fail-no-match cases
(sig-013,sig-026), so it must not be described as 17/17 current-pass.
12

4.1.1. Failure Mode Taxonomy
To make the aggregate stale-reference rate inspectable, we manually cate-
gorized the stale-triggering outputs from the two main stale_context_only
runs. This analysis covers 28 stale-positive output events: 15 from
Qwen2.5-Coder-7B-Instruct and 13 from gpt-4.1-mini . These events
correspond to 16 unique samples because the two models agree on 12
stale-triggering samples, Qwen contributes three model-specific stale triggers,
andgpt-4.1-minicontributes one.
Table 1: Failure-mode taxonomy for 28 stale-positive output events across the two main
stale_context_only runs. The examples show representative sample IDs and the stale
call shape.
Failure mode Count Typical shape Example
New argument omit-
ted22/28 The current helper signature
requires one or more addi-
tional arguments, but the
model calls the stale shorter
signature.sig-027: stale two-argument
chop_cells call instead of the
current three-argument call.
Obsolete argument
retained5/28 The stale helper signature con-
tains an argument no longer
used by the current helper,
and the model keeps it.sig-001: stale three-
argument _nullpager call
instead of the current two-
argument call.
Stale helper recon-
struction1/28 The model reconstructs a
helper body or wrapper
around the stale signature
rather than simply making a
stale call.sig-003: Qwen defines a local
stale-shape _tempfilepager
helper before using it.
The dominant error is therefore not an arbitrary syntax failure. In most
stale-positive outputs, the model produces a plausible call whose argument list
matches the retrieved historical helper rather than the current repository state.
This strengthens the interpretation that stale context acts as a directional
misleading signal: it changes the shape of the generated call toward a concrete
historical signature.
The taxonomy also explains why pass rate and stale-reference rate should
remain separate. Some outputs match neither current nor stale patterns,
especially in high-arity wrapper samples. Those fail-no-match cases are not
counted in this taxonomy because they do not provide evidence that the
model followed stale repository state.
4.1.2. No-Retrieval Baseline
Both models show the same no-retrieval baseline:
13

Model No-retrieval SRR No-retrieval pass Fail-no-match
Qwen0/17 (0.0%) 1/17 (5.9%) 16/17 (94.1%)
gpt-4.1-mini0/17 (0.0%) 1/17 (5.9%) 16/17 (94.1%)
Compared with no retrieval, stale-only retrieval does not merely cause
generic failure. It specifically shifts many outputs toward stale helper refer-
ences:
∆stale vs. no-retrieval = 88.2%−0.0% = 88.2percentage points (Qwen)
∆stale vs. no-retrieval = 76.5%−0.0% = 76.5percentage points (gpt-4.1-mini).
The matching no-retrieval result matters because it separates two failure
types. Without retrieval, both models mostly produce outputs that match
neither oracle. With stale retrieval, the dominant failure becomes stale-state
helper references. Thus stale context is actively misleading rather than merely
uninformative.
The only no-retrieval pass in both models is sig-023 (requests
Session.get ), a common API likely recoverable from parametric model
knowledge.
4.1.3. Sensitivity: Qwen Without Quality-Gate Filtering
The main Qwen table uses 17 samples retained by a Qwen current-context
quality gate. To quantify this entanglement, we also ran Qwen on the 19
curated samples before that quality gate:
Set Current pass Current SRR Stale SRR
Qwen n=19 sensitivity 18/19 (94.7%) 0/19 (0.0%) 15/19 (78.9%)
The corresponding primary delta is 78.9 percentage points.
The gate-conditioned n=17 estimate is therefore 9.3 percentage points
higher than the ungated n=19 estimate. Both estimates remain far above our
pilot continuation threshold of 20 percentage points, so the main effect does
not depend on hiding the quality-gate entanglement. The n=19 sensitivity run
was executed after the original quality-gate decision. In that rerun, sig-012
passed current-context retrieval, while sig-024 still failed. We retain the
original n=17 main set to avoid post-hoc redefinition of the quality gate and
report n=19 only as sensitivity evidence.
14

4.2. RQ2: Rank Order Does Not Consistently Amplify the Failure
Qwen mixed-condition results:
Mixed order Stale-reference rate Stale-reference samples
Current first 4/17 (23.5%)sig-001,sig-002,sig-008,sig-010
Stale first 4/17 (23.5%)sig-001,sig-007,sig-010,sig-013
The Qwen rank-order delta is 0.0 percentage points. The two orders have
the same aggregate rate, with two shared stale-triggering samples and two
order-specific samples on each side.
gpt-4.1-minimixed-condition results:
Mixed order Stale-reference rate
Current first 5/17 (29.4%)
Stale first 5/17 (29.4%)
Forgpt-4.1-mini , the rank-order delta is 0.0 percentage points; the same
five samples trigger under both mixed orders.
Interpretation:
We do not find evidence that placing stale context at rank 1 am-
plifies the failure in the neutralized evaluation. Both models
have 0.0 percentage-point aggregate rank-order deltas. Qwen’s
equality comes from compensating order-specific samples, while
gpt-4.1-mini ’s equality comes from identical sample-level trig-
gers across both orders.
This result is consistent with our earlier pilot observations on version-skew
and cross-library distractors. We therefore treat it as evidence under the
current task framing rather than as a universal claim that rank order never
matters.
4.3. RQ3: Current Evidence Presence Rescues Most Stale-Only Failures
Adding valid current evidence reduces stale references from 88.2% (Qwen)
and 76.5% ( gpt-4.1-mini ) understale_context_only to much lower rates
under mixed conditions.
For Qwen:
15

presence-absence rescueQwen= 88.2%−23.5% = 64.7percentage points.
Forgpt-4.1-mini , both mixed orders produce the same stale-reference
rate and remain well below the stale-only condition:
rescue gpt-4.1-mini = 76.5%−29.4%
= 47.1percentage points.
This suggests that the dominant failure mechanism is not rank order itself,
but whether the valid current evidence is retrieved at all.
4.4. RQ4: Qwen and GPT-Mini Have Substantial Sample-Level Agreement
Qwen produces 15 stale-triggering samples and gpt-4.1-mini produces
13 understale_context_only. Their overlap is:
Quantity Value
Qwen stale-triggering samples 15
gpt-4.1-ministale-triggering samples 13
Intersection 12
Union 16
Jaccard overlap 75.0%
Wilson 95% interval for Jaccard [50.5%, 89.8%]
Qwen coverage bygpt-4.1-mini80.0%
gpt-4.1-minicoverage by Qwen 92.3%
The overlap breakdown is 12 samples triggered by both models, three
Qwen-only samples ( sig-016,sig-023,sig-026), onegpt-4.1-mini -only
sample (sig-028), and one neither-trigger sample (sig-013).
Per-repository agreement:
16

Table 2: Sample-level agreement by repository under stale_context_only . The table
reports whether stale-triggering samples are shared by both models or model-specific.
Repo n Both trigger Qwen-only GPT-only Neither
click 3 3 0 0 0
flask 3 3 0 0 0
requests 3 2 1 0 0
rich 5 3 1 1 0
httpx 3 1 1 0 1
Total 17 12 3 1 1
Interpretation:
On the neutralized dataset, the stale-context failure mode is not
merely an aggregate artifact: Qwen and gpt-4.1-mini agree on
most stale-triggering samples.
The confidence interval is wide because the union contains only 16 positive
samples. We therefore use Jaccard overlap as supporting evidence of sample-
level agreement, not as a precise estimate of cross-model stability.
However, we should not attribute the improvement over earlier
Qwen/DeepSeek pilot overlap solely to curation, because both the dataset
and model pair changed.
Note: sample sig-013, anhttpxtransport helper, has no stale-reference
classification under stale-only retrieval for either model. We treat this as a
possible oracle-strictness case rather than strong evidence of model robustness:
the output may use a keyword-shifted or receiver-prefixed call form that the
current static regex does not capture. A future iteration should review this
sample with relaxed oracle patterns before re-classifying it.
4.5. Excluded Samples
The Qwen current-context quality gate started from 19 curated samples
and retained 17:
sig-012 (httpx Client._init_proxy_transport ) andsig-024 (rich
render_scope ) were excluded because Qwen failed the original current-
context quality gate on these high-arity wrapper tasks.
These are not counted in the main table. They should be reported as a
limitation of the current task-wrapper format rather than as model failures
under stale retrieval.
17

The n=19 sensitivity run includes these two rows and still produces a
78.9 percentage-point Qwen primary delta. We treat this as a sensitivity
check for the quality-gate entanglement, not as an alternative main table. In
the neutralized sensitivity rerun, sig-012 passed current-context retrieval
whilesig-024 remained fail-no-match. We keep the original gate decision for
protocol consistency rather than reclassifying the main set after inspecting a
later rerun.
4.6. Statistical Note
For Qwen’s 15/17 stale-reference rate, the Wilson 95% interval is approxi-
mately 65.7% to 96.7%; for gpt-4.1-mini ’s 13/17 rate, it is approximately
52.7% to 90.4%. These Wilson intervals describe stale-only rates rather than
the paired treatment effect, and they remain too wide for tight population-
level effect-size estimation. For the cross-model Jaccard overlap, 12/16 gives
an approximate Wilson interval of 50.5% to 89.8%.
Paired McNemar exact two-sided tests on the primary contrast
(stale_context_only versuscurrent_context_only ) yield p= 6.10×10−5
forQwen2.5-Coder-7B-Instruct andp= 2.44×10−4forgpt-4.1-mini ,
paired within sample. For rank order, the exact paired tests yield p= 1.0for
both models. In gpt-4.1-mini , the two mixed orders produce no discordant
stale-reference samples, so the test should be read as no observed sample-level
rank-order difference in this run rather than as proof that rank can never
matter. Rescue contrasts between stale-only and mixed conditions are also
significant for both models (p≤7.81×10−3).
The paired tests strongly support the existence of the stale-context effect
in this curated diagnostic set. However, the sample is still too small for precise
population-level effect-size estimation or detailed subgroup analysis.
4.7. Takeaway
The evidence supports a controlled diagnostic study with three main
results:
1.Stale-only context induces large stale-reference deltas in both Qwen and
gpt-4.1-mini.
2.Adding valid current evidence largely rescues the failure; rank order
does not consistently amplify it.
3.The two models agree on most stale-triggering samples in the curated
set, though sample size and axis coverage remain explicit limitations.
18

5. Discussion
5.1. Stale Retrieval Is Not the Same as Missing Retrieval
Theno-retrievalbaselinechangeshowthemainresultshouldbeinterpreted.
Without retrieved repository context, both Qwen and gpt-4.1-mini rarely
produce the expected current helper call: each model passes only 1 of 17
samples and otherwise produces outputs that match neither the current nor
stale oracle. However, neither model emits stale references under no retrieval.
In contrast, stale-only retrieval induces stale references on 15 of 17 Qwen
samples and 13 of 17gpt-4.1-minisamples.
This distinction matters. A stale snippet is not merely absent information
or generic retrieval noise. It provides a concrete but outdated state hypothesis:
the model can use it to produce syntactically plausible code that is anchored
to an obsolete helper signature. In our setting, stale retrieval changes the
error pattern from fail-no-match to wrong-state grounding. This is the core
diagnostic value of the experiment.
The condition contrast also argues against a simpler explanation: that
the models merely copy whatever retrieved snippet they see. Current-only
retrieval also supplies a concrete helper snippet, but the stale-reference rate
remains 0.0% under that condition. The failure appears when the retrieved
helper is temporally invalid, not whenever retrieval text is present. This
suggests that the model is grounding on the repository state implied by
the retrieved snippet, rather than performing content-agnostic copying from
retrieval.
5.2. Valid Evidence Presence Dominates Rank Order
The mixed conditions show that current evidence can rescue most stale-
only failures. For Qwen, adding current context reduces stale-reference rate
from 88.2% under stale-only retrieval to 23.5% under mixed retrieval. For
gpt-4.1-mini , both mixed orders produce 29.4% stale-reference rates, far
belowstale-onlyretrieval. Thissuggeststhatthedominantquestioniswhether
valid current evidence is retrieved at all.
Rank order appears secondary in the current sample. Both models have
0.0 percentage-point aggregate rank-order deltas in the neutralized evaluation.
The sample-level behavior is still informative: Qwen’s equal rates come from
two shared mixed-condition stale triggers and compensating order-specific
triggers, whereas gpt-4.1-mini ’s equal rates come from the same five samples
triggering under both orders. A larger study could test whether different
19

models have weak rank-order preferences once both current and stale evidence
are present, but our evidence points to current-evidence recall as the stronger
factor.
5.3. Why No-Target-Anchor Prompting Matters
The result also explains why two earlier pilot ideas failed. In the version-
skew and cross-library pilots, the task often explicitly named the target
API, library, or intended state. Under those prompts, strong models treated
retrieved distractors as advisory and followed the explicit target anchor. In
the current stale-repo setup, the prompt removes that anchor: the retrieved
snippets are the model-visible source of repository-state evidence.
This suggests a methodological implication for future Code RAG robust-
ness studies. If the research question is whether retrieval can induce a wrong
state, the task prompt should not independently reveal the correct state.
Otherwise the experiment may measure the model’s ability to ignore retrieval
under strong instructional anchoring rather than the model’s reliance on
retrieved evidence.
5.4. Implications for Code RAG Systems
The study does not propose a freshness filter or retriever. Still, the results
suggest where such systems should be careful. Repository-level retrieval
pipelines should treat temporal validity as a first-class metadata property,
not merely as a side effect of relevance. A retrieved snippet can be highly
topical, from the same project, and close to the target helper name while still
being harmful because it reflects an older repository state.
This is especially relevant for systems that cache indexes, retrieve from
code search snapshots, use external documentation mirrors, or combine local
context with historical examples. If the current state is not retrieved, the
model may not simply abstain or fail generically; it may confidently bind
to the stale state. The no-retrieval comparison suggests that stale evidence
can be worse than no evidence along the specific dimension of wrong-state
references.
Concrete engineering checks follow naturally from this result. Retrieval
systems can attach commit-hash or index-build-time freshness headers to
retrieved snippets, enforce index-invalidation windows when the repository
changes, or run a current-evidence recall probe before generation. These
checks would need their own evaluation; we list them as implications rather
than as claims of a mitigation contribution.
20

5.5. What This Paper Does Not Claim
We do not claim that all harmful retrieval is unexplored, that context
filtering is new, or that rank order never matters. Prior work already studies
harmful retrieved chunks, selective retrieval, dependency-aware mitigation,
andrepository-levelcodecompletion. Ourclaimisnarrower: toourknowledge,
the closest related work reviewed here does not control old-vs-current retrieved
repository context as the treatment variable and report stale-reference behav-
ior under matched completion tasks.
We also do not claim a deployed mitigation. The current contribution is a
diagnostic protocol and a set of empirical observations. A natural next step is
to test whether simple freshness metadata, commit-aware index invalidation,
or current-evidence recall checks reduce stale-reference failures. Those systems
should be evaluated as future mitigation work rather than folded into the
present diagnostic study.
6. Limitations
6.1. Sample Size
The main evaluation uses 17 retained production-helper samples. The
stale-only effect is large enough to motivate a controlled diagnostic study,
but the sample is not large enough for tight effect-size estimation or detailed
subgroup analysis. The Wilson intervals for the 15/17 Qwen and 13/17
gpt-4.1-mini stale-reference rates remain wide, and the 75.0% cross-model
Jaccard overlap also has a wide interval. We therefore treat the estimates as
diagnostic evidence rather than final population-level rates.
The n=19 Qwen sensitivity run partially addresses selection concerns:
before Qwen current-context quality gating, the stale-reference delta is 78.9
percentage points rather than the gate-conditioned 88.2 percentage points.
The effect survives the ungated curated set, but this does not replace a larger
evaluation. A future expansion should move toward 40–60 samples if the goal
is tighter confidence intervals or per-repository analysis.
6.2. Prompt-Leakage Audit Disclosure
A preliminary version of the evaluation used model-visible context identi-
fiers that encoded condition-specific terms. This violated our no-target-anchor
policy and was caught during a pre-submission prompt-leakage audit. We
therefore neutralized all model-visible context identifiers and reran the full
21

evaluation. All main results in this paper use the neutralized run. This
experience suggests that Code RAG diagnostic evaluations should audit not
just task text and retrieved snippets, but also seemingly administrative fields
such as context identifiers and metadata headers.
6.3. Axis Coverage
All retained samples are signature-change examples with one-parent-
commit drift. This keeps the first study controlled, but it leaves out other
important temporal change types: renamed helpers, removed helpers, moved
modules, refactors that preserve signatures but change semantics, and cross-
release drift. The current study therefore should not be generalized to all
forms of stale repository context. It shows that stale retrieval can induce
wrong-state calls for signature drift; future work should test whether the same
mechanism holds across broader change types and longer temporal gaps.
6.4. Repository and Language Scope
ThedatasetusesfivePythonrepositories: click,flask,httpx,requests ,
andrich. These projects provide realistic production-helper changes, but
they do not cover other languages, typed build systems, monorepo structures,
generated code, or compiled dependency boundaries. The result is most
directly about Python repository-helper completion under static call-pattern
oracles. Applying the protocol to Java, Rust, TypeScript, or multi-language
repositories may reveal different failure rates and different oracle requirements.
The repository selection is also biased toward mature public libraries with
clean histories and reviewable helper functions. Private monorepos, generated
code, rapidly changing product repositories, or codebases with weaker tests
may exhibit different retrieval-freshness risks. The current sample should
therefore be read as evidence that the failure mode exists in realistic open-
source libraries, not as a population estimate for all repositories.
6.5. Task Framing
The task is a local wrapper completion around a single helper call. This
keeps the treatment controlled and makes stale-reference behavior inspectable,
but it may increase reliance on the retrieved helper snippet compared with
a richer IDE session containing more surrounding files, tests, or developer
intent. A larger follow-up should test whether the effect persists for broader
edit tasks and multi-function completions.
22

The no-target-anchor prompting policy is also load-bearing. It is appro-
priate for testing whether retrieved context can determine the model-visible
repository state, but it should not be generalized to prompts that explicitly
name the intended current API or commit state. Earlier pilots suggest that
such anchors can let strong models ignore distractors, so both prompt families
are important but answer different questions.
6.6. Static Oracles
The current evaluation uses static regex-style call-pattern oracles rather
than full executable tests. This makes the experiment cheap and repeatable,
but it can miss semantically equivalent variants. Sample sig-013 is the
clearest case: both models avoid a stale-reference classification under stale-
only retrieval, but the output may use a keyword-shifted, receiver-prefixed,
or otherwise valid form that the static oracle does not capture. We therefore
report fail-no-match separately from stale-reference rate and avoid treating
every fail-no-match as model robustness.
Executabletestswouldstrengthentheevidence. Theywouldalsointroduce
engineering costs: dependency reconstruction, historical environment setup,
and per-commit test harnesses. The present study prioritizes a controlled
diagnostic signal; a follow-up should convert the strongest samples into
executable tests.
6.7. Quality-Gate Entanglement
The 17-sample main set is filtered by Qwen current-context success. This
makes Qwen’s 17/17 current pass rate a construction property of the retained
sample set rather than an independent finding. We address this in two ways.
First, we report gpt-4.1-mini current-context pass separately; it was not
used for the quality gate and passes 15/17 samples. Second, we run Qwen on
the 19 curated samples before quality-gate filtering, where the stale-reference
delta remains 78.9 percentage points. This quantifies the Qwen-gate shift
at about 9.3 percentage points, but it does not eliminate the selection issue
entirely.
6.8. Model Coverage
The main evidence uses two models: Qwen2.5-Coder-7B-Instruct and
gpt-4.1-mini . This is enough to show that the effect is not limited to a
single model, but it is not a broad model-family study. DeepSeek-Coder-6.7B
was explored during pilot work, but its OpenAI-compatible serving path
23

produced tokenizer artifacts and low current-context pass rates under this
harness. We exclude it from the main analysis rather than forcing noisy
results into the paper.
The mixed-condition results also suggest that aggregate rank-order in-
variance can arise from different sample-level patterns. Qwen has the same
aggregate mixed-condition stale-reference rate under both orders but different
order-specific samples, while gpt-4.1-mini triggers on the same five samples
under both orders. Larger samples and additional models are needed before
making claims about rank-order sensitivity across model families.
6.9. Generation Variability
Most runs were stable at the aggregate level, but at least one edge sample
changed classification acrossrepeated Qwen current-context runs. Theharness
explicitly sends temperature=0 in the OpenAI-compatible request payload, so
this is not intended sampling diversity. However, the result rows do not store
the full request body or the serving backend’s effective sampling state. We
therefore treat the edge-case variation as backend or output-format variability
and note that future runs should log backend versions, provider defaults,
effective sampling parameters, and seed support explicitly.
6.10. Dataset Independence
Samples are mined from upstream repository history rather than derived
from nearby published datasets. This avoids directly reusing benchmark
artifacts from nearby work, but it also means our dataset is smaller and
more labor-intensive to construct. We make the sample construction process
transparent enough that future work can extend it without relying on our
exact sample set.
7. Reproducibility Statement
The experimental harness, sample files, oracle patterns, and raw outputs
are stored under experiments/stale_repo_rag_pilot/ . The final neutral-
ized evaluation set is data/curated_v3_neutralized.jsonl (17 samples),
with the pre-quality-gate sensitivity set in data/curated_candidates_v3_n
eutralized.jsonl (19 samples). Sample construction rules are documented
in the mining and curation artifacts under experiments/stale_repo_rag
_pilot/runs/ . The neutralization script, prompt validator, prompt dumps,
24

and mapping file are released with the artifact so that readers can verify that
model-visible context identifiers do not encode freshness labels.
All model calls use the OpenAI-compatible chat-completions client in
stale_repo_rag_pilot/clients.py . The request payload explicitly sets
temperature=0 ;top_p,seed, andmax_tokens are left to provider or vLLM
endpoint defaults. Qwen2.5-Coder-7B-Instruct is served through a local
vLLM OpenAI-compatible endpoint. gpt-4.1-mini is served through the
OpenAI API with the same prompt and oracle pipeline.
The raw JSONL result rows preserve generated outputs, oracle decisions,
model names, conditions, and rendered-prompt-derived classifications. They
do not reconstruct provider-internal sampling state beyond metadata exposed
by the OpenAI-compatible endpoints. We therefore report the client-side
request configuration from the harness and distinguish it from server-side
effective sampling state, which may depend on provider or vLLM endpoint
defaults.
The oracle implementation is in stale_repo_rag_pilot/oracle.py . It
stripsMarkdowncodefences, normalizesbyte-leveltokenizerartifactsobserved
in pilot serving, and evaluates generated code with static current and stale
call patterns stored in each JSONL sample. Raw per-condition logs for the
final neutralized evaluation are preserved in:
•runs/qwen-v3-primary/andruns/openai-v3-primary/
•runs/qwen-v3-sensitivity-n19/ for the ungated Qwen sensitivity
check
•runs/v3_smoke_prompt_dump/for rendered prompt leakage checks
The preliminary run predating the prompt-leakage audit is retained only as an
excluded audit-history artifact and is not used to reproduce the main tables.
Citation metadata was batch-checked in the citation-verification artifact.
A self-contained artifact accompanies this submission as supplementary
material. It contains: the final neutralized 17-sample dataset and the 19-
sample ungated sensitivity set; the per-sample opaque context-identifier
mapping file; V3 result logs for both models; the experimental harness source
code; the neutralization and prompt-audit scripts; and five representative
rendered prompts illustrating the model-visible inputs. An automated leakage-
audit summary over all 85 rendered prompts is also included. The exact
vLLM server version and model snapshot hashes used for the released runs
will be recorded in the camera-ready artifact alongside a permanent DOI.
25

8. Conclusion
This paper presents a controlled diagnostic study of stale repository
context in retrieval-augmented code completion. Across 17 production-helper
signature changes from five Python repositories, stale-only retrieved context
induces stale-state helper references on 15 of 17 Qwen samples and 13 of 17
gpt-4.1-mini samples, while current-only context eliminates stale references.
The no-retrieval baseline shows a different failure mode: without repository
context, both models mostly fail to match either oracle rather than producing
stale references. This distinction suggests that stale retrieval is actively
misleading, not merely unhelpful.
The results also clarify the mechanism. When valid current evidence is
present alongside stale evidence, stale-reference rates drop sharply. Both
models show 0.0 percentage-point aggregate rank-order deltas across mixed
conditions. The dominant factor in this study is therefore whether valid
current evidence is retrieved at all, not the relative ordering of stale and
current snippets once both are available.
These findings point to a concrete future expansion path. Larger studies
should expand to 40-60 samples across rename, remove, move, and semantic-
refactor axes; add executable oracles for stronger validation; and test addi-
tional model families and languages. System builders should treat temporal
validity as a first-class retrieval property: highly relevant repository context
can still be harmful if it represents an obsolete project state.
Appendix A. Prompt Templates
experiments/stale_repo_rag_pilot/stale_repo_rag_pilot/prompt
s.py.
Appendix A.1. System Prompt
You are a coding assistant. Return only code, no explanation.
Use the local file context and retrieved repository context
as the available project evidence.
Appendix A.2. User Prompt Template
Local file context:
{model_visible_local_context}
Task:
26

{task_prompt}
Retrieved repository context:
{retrieved_context_blocks}
Return only a minimal {language} code snippet that completes the task.
The model-visible local context is filtered before prompt construction.
Commit hashes, freshness labels, target-state metadata, expected current
references, and forbidden stale references are oracle-only and are not included
in the prompt. Retrieved context blocks expose an internal context id and the
snippet text. File paths can be exposed by an optional –context-metadata
pathflag, but the final neutralized runs use hidden context metadata.
Appendix A.3. Worked Example:stale-repo-static-sig-001
The following is the verbatim stale_context_only prompt shape for a
Click helper sample from the neutralized run. The context id is model-visible
but opaque; the mapping from this id to freshness state is oracle-only and
not present in the prompt.
Local file context:
{
"file_path": "generated/call_site.py",
"prefix": "def invoke_target(stream, color, generator):\n "
}
Task:
Complete invoke_target(stream, color, generator)
by calling the repository helper shown in the retrieved context.
Retrieved repository context:
[CONTEXT id=ctx-sig-001-a]
def _nullpager(
stream: t.TextIO, generator: cabc.Iterable[str], color: bool | None
) -> None:
"""Simply print unformatted text. This is the ultimate fallback."""
for text in generator:
if not color:
text = strip_ansi(text)
stream.write(text)
Return only a minimal python code snippet that completes the task.
27

For the corresponding current_context_only condition, the re-
trieved context contains [CONTEXT id=ctx-sig-001-b] with the current
_nullpager(stream, color) signature instead. For the mixed_* conditions,
both snippets are included and only their order changes.
Appendix A.4. No-Target-Anchor Guard
Thedatasetconstructionpolicyforbidsmodel-visiblehintssuchas"current
HEAD", "stale context", "old commit", "target commit", and explicit current
or stale helper references in the local task. The runtime harness performs a
lightweight static check for these leakage terms before issuing model calls.
Appendix B. Excluded Preliminary Run
The preliminary evaluation predating the prompt-leakage audit is reported
here only to document the audit history. It isexcludedfrom the main evidence
because its model-visible context identifiers contained condition-specific terms.
Table B.3: Aggregate comparison between the excluded preliminary run and the neutralized
final run. The preliminary run is not used as evidence for any claim in this paper.
Metric Excluded preliminary run Final neutralized run
Qwen stale-only SRR 14/17 15/17
GPT-mini stale-only SRR 14/17 13/17
Aggregate stale-only events 28/34 28/34
Qwen rank-order∆0.0 pp 0.0 pp
GPT-mini rank-order∆−17.6pp 0.0 pp
Cross-model Jaccard 75.0% 75.0%
The preliminary run is retained in the artifact only so that the audit trail
is interpretable. All main-paper tables and claims use the final neutralized
run.
References
[1]A. N. Ashik, S. Wang, T.-H. Chen, M. Asaduzzaman, Y. Tian, When
LLMs Lag Behind: Knowledge Conflicts from Evolving APIs in Code
Generation (2026).arXiv:2604.09515.
URLhttps://arxiv.org/abs/2604.09515
28

[2]L. Liang, J. Gong, M. Liu, C. Wang, G. Ou, Y. Wang, X. Peng, Z. Zheng,
RustEvo2: An Evolving Benchmark for API Evolution in LLM-based
Rust Code Generation (2025).arXiv:2503.16922.
URLhttps://arxiv.org/abs/2503.16922
[3]F. Zhang, B. Chen, Y. Zhang, J. Keung, J. Liu, D. Zan, Y. Mao, J.-G.
Lou, W. Chen, RepoCoder: Repository-Level Code Completion Through
Iterative Retrieval and Generation (2023).arXiv:2303.12570.
URLhttps://arxiv.org/abs/2303.12570
[4]S. Zhang, Y. Ding, S. Lian, S. Song, H. Li, CodeRAG: Finding Relevant
and Necessary Knowledge for Retrieval-Augmented Repository-Level
Code Completion (2025).arXiv:2509.16112.
URLhttps://arxiv.org/abs/2509.16112
[5]T. Liu, C. Xu, J. McAuley, RepoBench: Benchmarking Repository-Level
Code Auto-Completion Systems (2023).arXiv:2306.03091.
URLhttps://arxiv.org/abs/2306.03091
[6]Y. Ding, Z. Wang, W. U. Ahmad, H. Ding, M. Tan, N. Jain, M. K.
Ramanathan, R. Nallapati, P. Bhatia, D. Roth, B. Xiang, CrossCodeEval:
A Diverse and Multilingual Benchmark for Cross-File Code Completion
(2023).arXiv:2310.11248.
URLhttps://arxiv.org/abs/2310.11248
[7]Y. Li, S. Liu, K. Chen, T. Zhang, Y. Liu, Impact-driven Context Filtering
For Cross-file Code Completion (2025).arXiv:2508.05970.
URLhttps://arxiv.org/abs/2508.05970
[8]Y. Huo, K. Zeng, S. Zhang, Y. Lu, C. Yang, Y. Guo, X. Tang, Re-
poShapley: Shapley-Enhanced Context Filtering for Repository-Level
Code Completion (2026).arXiv:2601.03378.
URLhttps://arxiv.org/abs/2601.03378
[9]D.Wu, W.U.Ahmad, D.Zhang, M.K.Ramanathan, X.Ma, Repoformer:
Selective Retrieval for Repository-Level Code Completion (2024). arXiv:
2403.10059.
URLhttps://arxiv.org/abs/2403.10059
[10]Y. Tian, W. Yan, Q. Yang, X. Zhao, Q. Chen, W. Wang, Z. Luo, L. Ma,
D. Song, CodeHalu: Investigating Code Hallucinations in LLMs via
29

Execution-based Verification, in: Proceedings of the AAAI Conference
on Artificial Intelligence, Vol. 39, 2025, pp. 25300–25308. arXiv:2405.0
0253,doi:10.1609/aaai.v39i24.34717.
URLhttps://arxiv.org/abs/2405.00253
[11]T. Y. Zhuo, J. He, J. Sun, Z. Xing, D. Lo, J. Grundy, X. Du, Iden-
tifying and Mitigating API Misuse in Large Language Models, IEEE
Transactions on Software Engineering (2026). arXiv:2503.22821 ,
doi:10.1109/TSE.2026.3651566.
URLhttps://arxiv.org/abs/2503.22821
[12]H. Su, S. Jiang, Y. Lai, H. Wu, B. Shi, C. Liu, Q. Liu, T. Yu, EVOR:
Evolving Retrieval for Code Generation (2024).arXiv:2402.12317.
URLhttps://arxiv.org/abs/2402.12317
[13]R. Bairi, A. Sonwane, A. Kanade, V. D. C, A. Iyer, S. Parthasarathy,
S.Rajamani, B.Ashok, S.Shet, CodePlan: Repository-levelCodingusing
LLMs and Planning, Proceedings of the ACM on Software Engineering
1 (FSE) (2024) 675–698.arXiv:2309.12499,doi:10.1145/3643757.
URLhttps://arxiv.org/abs/2309.12499
[14]L. Wang, L. Ramalho, A. Celestino, P. A. Pham, Y. Liu, U. K. Sinha,
A. Portillo, O. Osunwa, G. Maduekwe, SWE-Bench++: A Framework
for the Scalable Generation of Software Engineering Benchmarks from
Open-Source Repositories (2025).arXiv:2512.17419.
URLhttps://arxiv.org/abs/2512.17419
[15]Y. Chen, M. Chen, C. Gao, Z. Jiang, Z. Li, Y. Ma, Towards Mitigat-
ing API Hallucination in Code Generated by LLMs with Hierarchical
Dependency Aware, in: Proceedings of the 33rd ACM International Con-
ference on the Foundations of Software Engineering Companion, 2025,
pp. 468–479.arXiv:2505.05057.
URLhttps://arxiv.org/abs/2505.05057
[16]J. Spracklen, R. Wijewickrama, A. H. M. N. Sakib, A. Maiti,
B. Viswanath, M. Jadliwala, We Have a Package for You! A Com-
prehensive Analysis of Package Hallucinations by Code Generating LLMs
(2024).arXiv:2406.10279.
URLhttps://arxiv.org/abs/2406.10279
30