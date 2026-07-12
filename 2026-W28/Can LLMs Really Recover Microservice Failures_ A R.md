# Can LLMs Really Recover Microservice Failures? A Recovery-Aware Evaluation of Diagnosis-to-Action Reasoning

**Authors**: Jiaxing Qi, Zhongzhi Luan, Hongyu Zhang, Shaohan Huang, Carol Fung, Yongxin Tong, Hailong Yang, Depei Qian

**Published**: 2026-07-06 03:05:17

**PDF URL**: [https://arxiv.org/pdf/2607.04623v1](https://arxiv.org/pdf/2607.04623v1)

## Abstract
Large language models (LLMs) are increasingly used to interpret operational evidence and assist incident response in cloud-native microservice systems. However, recovery-oriented use cases require more than identifying a root cause. After observing symptoms and diagnosing a fault, an operator or agent must translate the diagnosis into a concrete recovery action, apply it to an admissible target, and verify that service health has been restored. Existing RCA and log-analysis evaluations are well-suited to diagnosis, but they do not characterize this subsequent action decision. This paper presents R2Act, a recovery-action evaluation framework for post-diagnosis incident response. R2Act defines an incident schema, quality gate, action-space representation, recovery-validity metrics, offline evaluator, and live-replay protocol. We instantiate the framework as a benchmark dataset of 302 quality-audited Kubernetes incidents from \system. Each incident provides synchronized multi-modal observations, root-cause labels, an incident-specific action space, and annotated valid and invalid recovery plans. We evaluate heuristic, supervised, RCA-oriented, deep log, and LLM-based methods. The strongest RAG-based LLMs reach 91.4\%--99.7\% root-cause service accuracy, yet their recovery validity remains only 36.8\%--60.3\%. Even when both the root-cause service and fault type are correct, recovery-oriented methods still choose invalid actions for 39.5\%--62.0\% of correctly diagnosed incidents. Overall, this work reveals that many recovery failures arise not from missing diagnostic knowledge, but from the difficulty of translating diagnostic evidence into valid recovery actions and admissible targets. This work provides a reproducible, simplified starting point for research and evaluation.

## Full Text


<!-- PDF content starts -->

Can LLMs Really Recover Microservice Failures?
A Recovery-Aware Evaluation of
Diagnosis-to-Action Reasoning
Jiaxing Qi∗, Zhongzhi Luan∗, Hongyu Zhang†, Shaohan Huang‡
Carol Fung§, Yongxin Tong∗, Hailong Yang∗, and Depei Qian∗
∗Beihang University
jiaxingqi@buaa.edu.cn, luan.zhongzhi@buaa.edu.cn, yxtong@buaa.edu.cn, hailongyang@buaa.edu.cn, depeiq@buaa.edu.cn
†Chongqing University
hyzhang@cqu.edu.cn
‡Microsoft Research Asia
shaohanh@microsoft.com
§Concordia University
carol.fung@concordia.ca
Abstract—Large language models (LLMs) are increasingly
used to interpret operational evidence and assist incident re-
sponse in cloud-native microservice systems. However, recovery-
oriented use cases require more than identifying a root cause.
After observing symptoms and diagnosing a fault, an operator
or agent must translate the diagnosis into a concrete recovery
action, apply it to an admissible target, and verify that service
health has been restored. Existing RCA and log-analysis evalu-
ations are well-suited to diagnosis, but they do not characterize
this subsequent action decision. This paper presents R2Act, a
recovery-action evaluation framework for post-diagnosis incident
response. R2Act defines an incident schema, quality gate, action-
space representation, recovery-validity metrics, offline evaluator,
and live-replay protocol. We instantiate the framework as a
benchmark dataset of 302 quality-audited Kubernetes incidents
from Online Boutique. Each incident provides synchronized
multi-modal observations, root-cause labels, an incident-specific
action space, and annotated valid and invalid recovery plans. We
evaluate heuristic, supervised, RCA-oriented, deep log, and LLM-
based methods. The strongest RAG-based LLMs reach 91.4%–
99.7% root-cause service accuracy, yet their recovery validity
remains only 36.8%–60.3%. Even when both the root-cause
service and fault type are correct, recovery-oriented methods still
choose invalid actions for 39.5%–62.0% of correctly diagnosed
incidents. In validity-gated live replay, 146 of 302 Qwen-RAG
predictions are both offline-valid and recovered in live execution.
Overall, this work reveals that many recovery failures arise not
from missing diagnostic knowledge, but from the difficulty of
translating diagnostic evidence into valid recovery actions and
admissible targets. This work provides a reproducible, simplified
starting point for research and evaluation.
Index Terms—microservices, fault diagnosis, recovery plan-
ning, recovery-aware evaluation, LLMs
I. INTRODUCTION
AI-assisted operations are advancing rapidly along several
axes. LLM-based operational systems can summarize logs,
retrieve similar incidents, localize likely root causes, and
draft mitigation steps [1], [2]. At the same time, cloud-native
microservice systems have become harder to recover because
R2ActService
Restored
Scale Out
Fix 
Dependency
Restart
Service
Recovery Need
Rollback
Conﬁg
Diagnosis-to-Action GapRCA  Process
Root Cause
Service
Anomaly
Detection
Causal
GraphRoot Cause
RankingMonitoring
System
Fig. 1. R2Act adds a recovery-action evaluation layer after diagnosis,
checking whether methods can select a valid operation and target under
incident-specific constraints.
independent deployment, elastic scaling, and dense service
dependencies create more ways for failures to propagate [3],
[4], [5]. A user-visible incident may leave evidence scattered
across logs, Kubernetes events, metrics, and resource state.
These trends change incident response from a diagnosis-only
task into an action-oriented decision problem:an automated
method must not only explain what failed, but also decide
which recovery action should be taken and where that action
should be applied.
Yet there have been relatively few efforts in this recovery
process. Real microservice recovery begins with partial, noisy,
and multi-modal operational evidence, while the required
outcome is a concrete decision that may modify a live deploy-
ment (see Figure 1 for a high-level schematic of the missing
process). Reaching that decision requires a method to assess
the available evidence, infer the affected service and fault
mechanism, and choose an operation whose target is valid
for the current incident. The central question is therefore not
only whether a method can identify a root cause, but whether
it can turn that diagnosis into a valid recovery action under
incident-specific constraints.
Existing evaluations mostly isolate earlier stages or adjacent
forms of this workflow. Log-analysis benchmarks measurearXiv:2607.04623v1  [cs.SE]  6 Jul 2026

parsing, anomaly detection, log understanding, or diagno-
sis [6], [7], [8]; RCA methods and benchmarks evaluate which
component, service, or fault type caused the incident [9],
[10]; and LLM-based operational reasoning systems study
retrieval, prompting, diagnostic data collection, or mitigation
recommendation [11], [12], [13], [14], [15]. Remediation
and self-healing work moves closer to recovery execution
from diagnosis reports or predefined policies [16], [1], [17].
These settings rarely combine synchronized incident evidence,
typed recovery plans, incident-specific validity labels, live
replay, and reusable evaluation artifacts. As a result, the post-
diagnosis action decision remains under-specified.
This gap matters because diagnostic correctness does not
imply recovery-action validity. A method may correctly iden-
tify that a service is affected by a DNS fault while still propos-
ing a service restart instead of repairing the dependency-
resolution path. An HTTP routing fault may require rolling
back a traffic rule rather than restarting the diagnosed service.
A memory-pressure incident may require changing a resource
field rather than scaling an unrelated component. In such cases,
the method is not simply missing a root-cause label; it fails to
carry the implications of the diagnosis into a valid operation,
target, and optional parameter choice.
To fill this gap, we introduce R2Act, a benchmark con-
struction and evaluation framework for diagnosis-to-action
reasoning in microservice failure recovery. R2Act defines an
incident schema, a quality gate, an incident-specific action-
space representation, recovery-validity metrics, an offline eval-
uator, and a live-replay protocol. We instantiate the framework
as a benchmark dataset of 302 quality-audited Kubernetes
incidents collected from Online Boutique. Each incident pack-
ages synchronized logs, events, metrics, Kubernetes state,
root-cause labels, incident-specific recovery action spaces, and
annotated valid recovery plans with invalid alternatives. R2Act
is the evaluation framework, while the 302 incidents are the
current benchmark dataset instance used in our study.
Figure 2 previews the main phenomenon measured by
R2Act: the difference between diagnostic correctness and
recovery-action validity when methods must select operation-
target plans under incident-specific constraints. In our evalu-
ations, heuristic baselines, supervised models, RCA-oriented
techniques, deep log models, and LLM-based agents all show
this diagnosis-to-action gap. The strongest RAG-based LLMs
reach 91.4%–99.7% root-cause service accuracy, yet their
recovery validity remains only 36.8%–60.3%. Even when both
the root-cause service and fault type are correctly identi-
fied, recovery-oriented methods still choose invalid plans for
39.5%–62.0% of correctly diagnosed incidents. The errors
concentrate in DNS, HTTP, and memory faults, where valid
recovery depends on dependency, configuration, or resource
semantics beyond the diagnosed service. In validity-gated live
replay, 146 of 302 Qwen-RAG predictions are both offline-
valid and recovered in live execution, showing that action-level
validity bridges diagnostic evaluation and executable recovery.
This paper makes the following contributions:
•We formulate diagnosis-to-action recovery evaluation as a
0 20 40 60 80 100
RCA service accuracy (%)020406080100Recovery validity (%)Diagnosis-to-Action Gap
KeywordRuleNB
LogPromptLogRAGOpenRCA
DirectLMQwen-3.5Kimi-k2.6
MiniMax-M2.5DeepSeek-V4-proGLM-5.1
OpenRCA
CoTLMFig. 2. The plot compares diagnostic accuracy and recovery-action validity on
the same incident population. Their divergence shows that recovery-oriented
evaluation requires action-level metrics beyond diagnostic correctness.
distinct problem for microservice incident response. The
task asks whether a method can select a valid operation
and admissible target under incident-specific constraints.
•We present R2Act, a benchmark construction and eval-
uation framework with synchronized incident evidence,
typed recovery action spaces, validity metrics, an offline
evaluator, and a live-replay protocol. We instantiate it as a
302-incident Kubernetes benchmark covering six service
roles and eight fault categories.
•Extensive evaluation demonstrates that the diagnosis-to-
action gap is a persistent failure mode in microservice
recovery. Even after correct RCA, methods frequently
select invalid recovery plans, with errors dominated by
wrong operations and invalid plan structures. We further
validate offline recovery decisions through live replay,
which checks whether predicted actions actually restore
service health.
II. RELATEDWORK ANDMOTIVATION
A. Related Work
Microservice RCA methods and evaluations.Microser-
vice RCA has been studied with metric-based, causal-
inference-based, log-based, and learning-based methods.
These methods use causal graphs, random-walk ranking, hy-
pothesis testing, metric anomaly analysis, or learned represen-
tations to localize faulty services or metrics. Recent studies and
toolkits, including causal-inference RCA evaluations, PyRCA,
and RCAEval, make these methods easier to compare on
microservice telemetry [20], [21], [10]. Their evaluation target
is root-cause localization, which is the appropriate objective
for diagnosis-stage RCA. R2Act adds a downstream action-
level evaluation for recovery-oriented use cases.
Log analysis and LLM-based operational reasoning.
Log-analysis benchmarks evaluate parsing, anomaly detec-
tion, incident classification, and log-based diagnosis [6], [7],
[8]. Recent LLM-based systems extend this direction with
prompting, retrieval, and agent workflows for operational logs.
LogSage and LogSieve study CI/CD failure analysis and

TABLE I
QUALITATIVE COMPARISON WITH REPRESENTATIVE EVALUATION SETTINGS. THE FEATURE COLUMNS FOCUS ON RECOVERY-ORIENTED
INCIDENT-RESPONSE EVALUATION:SYNCHRONIZED INCIDENT EVIDENCE,ROOT-CAUSE LABELS,TYPED RECOVERY PLANS,INCIDENT-SPECIFIC
VALIDITY LABELS,LIVE RECOVERY REPLAY,AND REUSABLE EVALUATION ARTIFACTS.
Existing Works Primary TaskMulti-modal
Incident
EvidenceRoot-cause
LabelsTyped
Recovery
PlansIncident
specific
Validity
LabelsLive
Recovery
ReplayReusable
Evaluation
Artifact
Microservice RCA benchmarks [10] Root-cause localization from telemetry✓ ✓ ✗ ✗ ✗ ✓
Log analysis and LLM reasoning [8],
[11], [13]Log understanding, anomaly detection, or
root-cause analysis✗ ✓ ✗ ✗ ✗ ✗
Self-healing and remediation
methods [16], [1], [17]Mitigation or remediation generation from
diagnosis or policies✗ ✓ ✓ ✗ ✓ ✗
Execution-oriented SE benchmarks [18],
[19]Executable validation of generated software
artifacts✗ ✗ ✗ ✗ ✗ ✓
R2Act (Ours) Diagnosis-to-action recovery evaluation✓ ✓ ✓ ✓ ✓ ✓
log reduction, while LogPrompt, LogRAG, OpenRCA, and
RCAFlow explore LLM-based log analysis or RCA [11], [12],
[22], [23], [13], [24]. These studies show the value of LLMs
for operational reasoning, but they mainly evaluate diagnosis,
log understanding, or task-specific remediation support rather
than incident-specific recovery-action validity.
Self-healing and autonomic recovery methods.Recent
work has started to connect cloud incident diagnosis with
remediation [25]. LLM-based systems such as mitigation-
step recommendation, RCACopilot, and RCAgent support
root-cause analysis, diagnostic data collection, or mitigation
recommendation in production cloud settings [26], [15], [14].
Other work moves closer to recovery execution: Sarda et al. [1]
generate Ansible playbooks for microservice auto-remediation,
GenKubeSec [27] studies Kubernetes misconfiguration reme-
diation, and GALR [28] combines graph-based RCA with
LLM-assisted recovery planning. MicroRemed [17] is the
closest benchmark to R2Act because it evaluates LLMs on
executable microservice remediation from diagnosis reports.
R2Act is complementary: it starts from synchronized multi-
modal incident observations and evaluates whether methods
can select valid recovery operations and admissible targets
under incident-specific action spaces.
Prior remediation work often starts from diagnosis re-
ports, mitigation suggestions, or predefined adaptation poli-
cies. R2Act starts from synchronized incident observations
and asks whether methods can both diagnose the failure and
select a valid operation and target for that specific incident.
This setting connects diagnosis and recovery planning while
keeping the post-diagnosis recovery-action decision explicit.
Execution-oriented software engineering benchmarks.
Execution-oriented benchmarks evaluate generated outputs
through explicit checks rather than only textual similarity
or classification accuracy. SWE-bench checks whether code
changes resolve real GitHub issues, and SecBench provides
executable security tests for server-side JavaScript [18], [19].
These benchmarks show that software engineering evaluation
becomes more informative when outputs are validated against
task-specific execution criteria.B. Motivation
R2Act follows an execution-oriented benchmarking princi-
ple. A recovery plan should be evaluated not only by whether
it identifies the relevant service, but also by whether its
operation and target are valid for the specific incident. Table I
compares existing work along the dimensions required for
recovery-oriented microservice evaluation. Prior benchmarks
and systems have advanced diagnosis, log understanding,
remediation recommendations, and executable software tasks.
R2Act complements these efforts by introducing the action-
level evaluation layer needed when incident-response methods
are used to support or automate recovery. This comparison
reveals two missing layers that remain under-specified in
existing evaluations.
Missing Layer 1: Post-diagnosis action-decision evalu-
ation.Microservice recovery depends on heterogeneous evi-
dence, including logs, events, metrics, and service state. Ex-
isting log or RCA benchmarks reasonably evaluate diagnosis-
stage tasks such as log parsing [8], anomaly detection [29],
log understanding [12], and root-cause localization [10], [24].
Recovery-oriented incident response introduces an additional
downstream target: selecting a recovery operation and binding
it to an admissible target under the current incident constraints.
Missing Layer 2: Incident-specific recovery constraints
and admissible targets.RCA provides important evidence
for recovery, but service and fault-type labels do not fully
specify the recovery decision. A method can identify the
correct service and fault type while still selecting an invalid
operation or target. For example, an HTTP routing fault
may require rolling back a traffic rule rather than restarting
the diagnosed service. Such errors occur after diagnosis and
determine whether the system actually recovers and whether
an operator can safely approve the proposed change.
Given such landscape, R2Act connects multi-modal incident
evidence, root-cause labels, incident-specific action spaces,
valid and invalid recovery plans, and live replay evidence in
one evaluation setting.
III. OVERVIEW
R2Act is designed as an evaluation framework for recovery-
oriented incident response. It preserves incident-time evidence,

Microservice 
system• Services: 6 Online Boutique roles
• Faults: CPU, Memory, Pod,
  Network, HTTP, DNS
• Intensity: moderate, severe
• Repetitions: audited campaigns
Fault campaign
Chaos Mesh
apiVersion:
chaos-mesh.org/v1alpha1
kind: StressChaos
spec:
  stressors:
    cpu:
      workers: 4
      load: 100Three-Phase Collection
5min
baselineInjection
activeRecovery
observation
Output: 
Logs+Metrics+Events
Quality Gate
Manual Audit
• completeness check
• label validationLLM-based 
Validation
• consistency verrification
• semantic validation
{
  metadata: {...},
  rca_labels: {...},
  action_space: [...],
  valid_plans: [...],
  invalid_plans: [...]
}R2Act
DatasetEvaluation
     RQ1: RCA-only 
Readiness
RCA vs Recovery
RQ2: Modality 
Effects
Logs/Events/Metrics
RQ3:Residual
Action-decision
residual
RQ4: Failure Sources 
Wrong operation + invalid structure;
DNS/HTTP/MemoryRQ5: Live replay check 
Per-health-gate --- execute plan ---
post-recovery success
Quality Assurance & Benchmark MaterializationFault Injection & Multi-modal CollectionFig. 3. R2Act operationalizes an action-level evaluation layer for microservice incident response. It builds audited incidents from fault campaigns and
synchronized multi-modal observations, materializes incident-specific action spaces and recovery plans, and evaluates post-diagnosis operation-target decisions
through offline validity reports and live replay evidence.
exposes the operational task, and makes the evaluation rule
explicit. R2Act extends diagnosis-centered evaluation with
an action-level recovery metric, where a prediction is valid
only if its operation, target, and optional fields satisfy the
incident-specific action space. Figure 3 presents the framework
overview. R2Act defines fault campaigns over service roles,
fault mechanisms, intensities, and repetitions, executes them in
a live Kubernetes testbed, collects synchronized observations
around each injected fault, and converts quality-audited cases
into benchmark records and evaluation reports.
A. Benchmark Design
R2Act uses Online Boutique [30], deployed on a Ku-
bernetes cluster [31]. The main benchmark covers six
services that represent different roles in a microser-
vice application: the entry-pointfrontend, the orches-
tration servicecheckoutservice, the recommenda-
tion and computation servicerecommendationservice,
the state-relatedcartservice, the catalog read ser-
viceproductcatalogservice, and the transaction-path
paymentservice. This coverage avoids concentrating the
benchmark on a small set of structurally similar services.
In addition, R2Act covers eight fault categories, organized
into three broader mechanism groups. (1) Kubernetes lifecycle
and resource faults include pod failure, CPU saturation, and
memory pressure. (2) Network and application-layer faults
include network delay, HTTP error, and HTTP abort. (3)
Dependency-resolution faults include DNS fault and DNS
randomization. This grouping is used only to explain the fault
mechanisms; the evaluation still treats the eight fault categories
as distinct labels. Most generated cases use moderate and
severe intensity levels. Fault campaigns that did not pass
the quality audit, including unstable I/O-pressure cases andincomplete network-delay runs, are retained in the audit trail
but excluded from the main results.
Formally, each incident is represented as a record:
xi= (O i, Mi, Yi, Ai),(1)
whereO idenotes operational observations,M irecords cam-
paign metadata,Y icontains RCA and recovery labels, and
Aidefines the allowed recovery-action space. The observation
component is multi-modal:
Oi={Olog
i, Oevent
i, Ometric
i , Ostate
i, Ochaos
i}.(2)
These terms correspond to service logs, Kubernetes events,
Prometheus metrics, pod/deployment state, and Chaos Mesh
status. This notation makes explicit that log-only, log-event,
and full multi-modal settings are different views of the same
incident record rather than different datasets.
B. Construction and Quality Audit
For each planned incident, the collector first checks that
all deployments are ready, that the frontend health check
succeeds, that no leftover Chaos Mesh object remains, and that
Prometheus is reachable. It then records a pre-fault snapshot,
applies the corresponding Chaos Mesh fault object, waits
for the injection window, records a during-fault snapshot,
removes the fault object, verifies recovery, and records a
post-fault snapshot. Each run receives a stable incident iden-
tifier encoding the system, target service, fault mechanism,
intensity, and repetition id. The raw observation directory is
therefore traceable back to the campaign manifest and the
generated benchmark JSON. Each incident is collected in
three phases: pre-fault, during-fault, and post-fault. For each
phase, R2Act records service logs from Kubernetes pods,
Kubernetes events and pod/deployment states, Prometheus
metrics [32] with resource and service-level indicators, and

Chaos Mesh status [33] with the injected target and fault state.
The synchronized view supports log-only methods as well as
methods that use events and metrics.
Quality control is applied before any incident enters the
main dataset. An incident is admitted only when all three
observation phases exist, logs are present, metrics and Chaos
Mesh status are present, Kubernetes state artifacts are present,
the annotation JSON is valid, and the benchmark JSON can
be built. We model this decision as a quality gate:
qi=Y
m∈M⊮[Om
i̸=∅]·⊮[Y ivalid]·⊮[A ivalid],(3)
whereMis the set of required modalities. Only incidents with
qi= 1enter the main benchmark. The audit assigns every
incident to one of four groups: quality-audited main incidents,
warning incidents, failed incidents, or legacy pilot incidents.
Warning and failed incidents are preserved for transparency,
but they are excluded from the main evaluation tables. This
policy avoids mixing incomplete or unstable observations into
the primary benchmark while retaining enough metadata to
diagnose collection failures.
C. Benchmark Records and Recovery Semantics
After quality auditing, R2Act represents each incident
as a standardized benchmark instance containing observa-
tions, metadata, labels, action-space constraints, valid recovery
plans, and invalid recovery plans. Each incident is annotated
with the root-cause service, root-cause fault type, fault inten-
sity, fault mechanism, and recovery labels. Recovery plans
in R2Act are represented as typed operation-target decisions.
Each plan contains an operation type, a primary target service,
and optional fields required by the operation, such as a
dependency endpoint, configuration scope, route rule, or re-
source limit. The incident-specific action space enumerates the
operations and targets that are admissible under the injected
fault and the current deployment state. The valid recovery
plan for each incident is constructed from the fault injection
specification, the affected service-dependency or Kubernetes
object, and the expected post-recovery system state.
Table II lists the concrete recovery actions represented in
the current benchmark dataset. For example, a DNS fault
injected on the dependency path of cartservice is labeled
with a dependency-resolution action rather than a generic
service restart, because restarting the diagnosed service does
not repair the injected DNS rule. HTTP route faults are la-
beled with configuration-level recovery actions, and memory-
pressure faults are labeled with resource-limit actions when the
deployment constraint requires it. Invalid plans are constructed
to cover common recovery failure modes, including wrong
operation, wrong target, invalid dependency or configuration
scope, invalid plan structure, and no-op actions. They are not
random negative samples; they test whether a method dis-
tinguishes service localization from valid recovery semantics.
The evaluation distinguishes target-service correctness, action-
type correctness, exact plan match, recovery validity, and no-
op or irrelevant actions. For a predicted planˆp i, recoveryTABLE II
RECOVERY ACTIONS REPRESENTED INR2ACT.
Action Typical faults Required target fields
Restart service Pod or delay faults Target service
Scale out CPU saturation Target service and replicas
Roll back deployment Deployment rollback cases Target service
Increase memory limit Memory pressure Target service and resource limit
Roll back configuration HTTP error or abort Route or configuration scope
Repair DNS/dependency DNS fault or randomization Service dependency endpoint
No action Invalid or no-op prediction None
Incident
id="onlineboutique_cartservice_dns_fault_severe_r02",
service="cartservice", fault="dns_fault",
intensity="severe"
Observation
mechanism="chaos-mesh-dns-error",
parameters="action=error, scope=service-domain,
duration=120s";
phases=[pre, during, post]; modalities=[logs, events];
logs: 12 services, 43,847 entries, 12 pods; events:
kubernetes state
Data Structure
{metadata:{batch, fault_intensity, load_profile, ...},
fault:{fault_type, target_service, fault_yaml},
input:{logs: [{phase, service, pod, level, message},
...], events: [...]}}
Labels
rca=("cartservice", "dns_fault");
valid action="fix_dns_resolution(cartservice)"; invalid
action="restart_service(frontend)"
Fig. 4. Example standardized benchmark instance. The compact exhibit shows
key fields of the severe DNS error, while the released artifact retains the
complete 43,847 log entries and structured annotations.
validity is defined over the incident-specific action space, valid
plan setP+
i, and invalid plan setP−
i:
valid(ˆp i) =⊮[ˆp i∈P+
i∧ˆpi/∈P−
i∧ˆpi∈Ai].(4)
Here,P+
iis defined as a set rather than a single textual
label. This set-valued allows one incident to admit multiple
semantically equivalent recovery plans, and the evaluator ac-
cepts a prediction if it matches any member ofP+
i. Additional
equivalent plans can be incorporated when they satisfy the
same recovery-state constraint and are validated under the
same action schema.
All baselines and LLM-based methods are evaluated on
the same benchmark records, which separates the cost of
data collection from the evaluation of new methods. Once an
incident has passed the quality audit and been materialized as
a record, it can be reused without re-running fault injection.
New recovery methods can therefore be compared under
identical evidence and labels, while new campaigns can be
appended through the same manifest-driven fault specification
and quality-audit pipeline.
IV. DATASETCHARACTERIZATION
A. Dataset Scope and Quality Audit
Table III summarizes the main dataset and the audit trail
retained during construction. The released benchmark contains

Root-cause service
302
incidents
Cart 16.6% Checkout 17.2%
Frontend 14.6% Payment 18.2%
Catalog 17.9% Recommend 15.6%Fault type
302
incidents
CPU 11.9% Memory 11.3%
Pod 11.9% Network 4.3%
HTTP-E 15.9% HTTP-A 12.9%
DNS-F 15.9% DNS-R 15.9%
Fig. 5. Proportional composition of the 302 quality-audited incidents by root-
cause service and fault type. The donut charts show category proportions in
the main benchmark. HTTP-E, HTTP-A, DNS-F, and DNS-R denote HTTP
error, HTTP abort, DNS fault, and DNS random, respectively.
TABLE III
DATASET SUMMARY AND QUALITY AUDIT.
Statistic Value Meaning
Main incidents 302 Used in all reported results
Root-cause services 6 Distinct service roles
Fault categories 8 Distinct RCA-type labels
Fault mechanisms 8 Distinct injection mechanisms
Service-fault cells 44 Non-empty service–fault pairs
Cell size 3–8 Incidents per service–fault cell
Moderate incidents 129 Medium fault intensity
Severe incidents 173 High fault intensity
Log records 12.6M Total main-set log volume
Logs per incident 41,736 Average observation scale
Metrics-present samples 302 Prometheus data available
Chaos-status samples 302 Injection status available
Candidate operations 7 Shared recovery action space
Gold operation types 5 Operations used by valid plans
Invalid examples 604 Two invalid plans per incident
Non-main audit-trail incidents 71 Warning, failed, and pilot runs
302 quality-audited incidents. Each main incident includes
logs, Kubernetes state, metrics, Chaos Mesh status, annotation
JSON, and a standardized benchmark instance. The quality
gate excludes incomplete, unstable, or pilot-stage incidents
from the primary evaluation. The 71 non-main audit-trail
incidents consist of warning, failed, and legacy pilot runs. They
are preserved for transparency but are not used in any reported
evaluation result. This separation is important for reproducibil-
ity: the main benchmark defines the evaluation population,
while the audit trail documents why certain collection attempts
were not admitted.
Figure 4 shows a compact example of one standardized
benchmark instance. The figure reports only the key fields
needed to understand the released schema. The complete
artifact keeps the full observation files, including the raw log
window and structured annotation files. This example clarifies
that R2Act is not a collection of isolated log snippets. Each
record connects a concrete fault to multi-modal observations,
RCA labels, and recovery semantics.
B. Service and Fault-Type Composition
Figure 5 shows the proportional composition of the main
dataset. The service composition is close to balanced: eachof the six root-cause services contributes between 44 and
55 incidents. These services cover different system roles,
including the entry point, orchestration service, recommen-
dation/computation service, state-related service, catalog-read
service, and transaction-path service. The fault composition
covers eight categories across lifecycle, resource, network,
application-layer, and dependency-resolution mechanisms. The
smaller network-delay slice reflects the quality audit rather
than a design omission, because unstable network-delay cases
were excluded from the main benchmark. Together, Table III
and Figure 5 show that the dataset is not a repetition of a single
service or a single failure mechanism. The quality gate keeps
incomplete incidents out of the primary evaluation, while the
retained composition still covers different service roles and
fault mechanisms.
C. Recovery Labels, Observation Scale, and Cell Coverage
Figure 6 gives three additional views of the dataset.Fig-
ure 6(a): the recovery-label separates the shared action-space
from the gold recovery labels. The action-space contains
seven candidate operation types, while the 302 gold valid
recovery plans use five of them; the remaining operations
remain available candidates but are not gold actions for the
main fault incidents.Figure 6(b): each incident contains a
substantial observation window rather than a short synthetic
message. The 302 incidents contain 12.6M log records, with
an average of 41,736.47 records per incident.Figure 6(c): The
service-fault heatmap characterizes the diversity of the dataset
beyond the marginal service and fault distributions. It shows
that the benchmark covers 44 distinct service-fault cells across
six services and eight fault categories, with 3 to 8 incidents in
each cell. This diversity is important for evaluating methods
across different failure modes and for interpreting per-fault
performance differences.
Summary:R2Act provides 302 quality-audited incidents
over six service roles and eight fault categories. Each
main incident contains multi-modal observations, struc-
tured RCA labels, recovery labels, and a shared action-
space, enabling us to compare diagnosis and recovery
behavior over the same incident population.
V. EVALUATIONDESIGN
The experiments evaluate methods through the recovery-
action evaluation layer provided by R2Act; they do not eval-
uate R2Act as a recovery method. Our goal is to examine
whether recovery-oriented incident response can be reliably
assessed beyond RCA labels. We therefore organize the eval-
uation in a progressive way. We first compare diagnostic
correctness with recovery-action validity on the same incidents
to test whether they measure different capabilities. We then
study whether adding richer incident evidence can close this
gap. Next, we condition on correctly diagnosed incidents to
isolate the remaining post-diagnosis action-decision errors. We
further analyze where these errors occur across fault types and
error categories. Finally, we check whether offline recovery

0 20 40 60 80 100 120
Gold plansDependency
restart
Config
rollback
Service
restart
Scale-out
Memory-limit
increaseOperation type96
87
61
36
22
(a) Recovery Labels36 38 40 42 44 46
Log records / incident (K)
(b) Log ScaleQ1
40.4Median
41.8Q3
42.6
Mean
41.7Min 36.3 Max 45.4
CPUDNS-F DNS-RHTTP-A HTTP-E Memory NetworkPodCart
Checkout
Frontend
Payment
Catalog
Reco.Root-cause service5 8 8 8 8 5 3 5
5 8 8 8 8 5 5 5
5 8 8 0 8 5 5 5
8 8 8 7 8 8 0 8
8 8 8 8 8 6 0 8
5 8 8 8 8 5 0 5
(c) Service-Fault Cells
 (d) Fault SummaryValid 302 (33%) Invalid 604 (67%)906
labels
0 2 4 6
Action countGold-usedAction-space
57
02468Fig. 6. Characterization of the benchmark dataset. (a) Distribution of gold recovery plans across operation types. (b) Log-scale distribution of root-cause
services and fault types. (c) Heatmap of service-fault cells, where color intensity indicates the number of incidents in each cell. (d) Donut chart showing the
distribution of fault types.
validity is consistent with live replay behavior. We investigate
the following research questions:
•RQ1: How different are diagnostic correctness and
recovery-action validity across methods?
•RQ2: How does multi-modal incident evidence affect
diagnosis and recovery-action validity?
•RQ3: What post-diagnosis action-decision errors remain
after correct diagnosis?
•RQ4: Which fault types and error categories make
recovery-action selection difficult?
•RQ5: How well does offline action validity align with
live replay outcomes?
Baselines.(1)Recovery-oriented methodsinclude log-based
LLM methods, LogPrompt [22] and LogRAG [23]; OpenRCA
variants [13]; and five LLM backbones evaluated under zero-
shot, few-shot, and retrieval-augmented protocols. (2)RCA-
only methodsinclude Keyword, Rule, supervised Naive Bayes,
PyRCA methods [21] (Epsilon Diagnosis, RCD, and Random
Walk), RCAEval methods [10] (BARO and CIRCA), and deep
log diagnosis models LogFormer [29] and OneLog [34]. No-
training methods are evaluated once on all incidents; learning-
based methods are evaluated under a 5-fold cross-validation.
RCA-to-Action Mapper.For methods that output only
RCA results, we use a RCA-to-action mapper as an evaluation
bridge. The mapper converts a predicted root-cause serviceˆs i
and fault type ˆtiinto a structured recovery planˆp i=g(ˆs i,ˆti)
within the incident-specific action space. It follows a fixed
fault-type policy: pod failure and network delay trigger service
restart, CPU saturation triggers scale-out, memory pressure
triggers memory-limit increase, HTTP faults trigger config-
uration rollback, and DNS faults trigger dependency restart.
Required route, configuration, or dependency fields are filled
from admissible targets in the incident action space.
The mapper is not intended to represent an optimal recovery
policy. It is used to separate two questions: how far coarse
service/type diagnosis can be carried by a transparent rule,
and where incident-specific action semantics are still required.
We apply the mapper only to RCA-only methods, includ-
ing keyword matching, rule-based variants, supervised Naive
Bayes, PyRCA, RCAEval, LogFormer, and OneLog. Their
action scores therefore measure the downstream recoverability
of RCA outputs under this fixed policy, rather than the
methods’ own recovery-generation ability. Recovery-orientedmethods, such as LogPrompt, LogRAG, OpenRCA, and LLM
backbones, are evaluated using their generated recovery plans
directly. Gold RCA + Mapper serves as a diagnostic control
for the limits of coarse service/type-to-action mapping, not as
a recovery upper bound.
Evaluation Metrics.For diagnosis,RCA-Service accuracy
measures the fraction of incidents whose predicted root-cause
service matches the label, andRCA-Type accuracymeasures
the fraction whose predicted fault type matches the label. For
recovery planning,Action Hitchecks whether the predicted
operation matches one valid operation,Target Hitchecks
whether the predicted target service matches a valid target,
Exact Matchrequires both operation and target to match
the gold plan, andRecovery Validitymeasures whether the
predicted plan belongs to the incident-specific valid action
set.No-op Rateis the fraction of predictions that return no
action or an irrelevant action. LetCdenote the event that both
the root-cause service and fault type are correct, and letV
denote the event that the predicted recovery plan is valid. For
post-diagnosis recovery analysis, we report conditional action
validityV C= Pr(V | C)and the invalid-action rate after
correct diagnosisE C= Pr(¬V | C) = 1−V C.ECmeasures
recovery-decision errors that remain after service and fault-
type diagnosis are correct; lower values indicate stronger post-
diagnosis recovery-action selection.
VI. RESULTS
A. RQ1: Diagnosis vs. Recovery-Action Validity
Experimental Setup.We first compare diagnostic cor-
rectness and recovery-action validity on the same incident
population. Table IV reports representative methods across two
output interfaces. For recovery-oriented methods, the action
metrics are computed from their own generated recovery
plans. For RCA-only controls, the action metrics are computed
through the fixed mapper described above and should be read
as downstream recoverability under a transparent policy, not
as the original methods’ recovery-generation ability.
Results and Analysis.Table IV shows that diagnostic
metrics and recovery-action metrics measure different outputs.
The strongest RAG-based LLMs achieve high RCA-Service
accuracy, but their recovery metrics remain much lower. For
example, Qwen-RAG reaches 0.990 RCA-Service and 0.957
RCA-Type, yet only 0.258 Exact Match and 0.483 Recovery

TABLE IV
DIAGNOSIS-TO-ACTION EVALUATION FOR REPRESENTATIVE METHODS.
BOLD MARKS STRONG DIAGNOSIS,WHILE RED HIGHLIGHTS LOW
RECOVERY-ACTION VALIDITY.
Method RCA-S↑RCA-T↑Action↑Target↑Exact↑Valid↑No-op↓
Recovery-oriented methods
LogPrompt 0.603 0.162 0.228 0.513 0.000 0.109 0.470
LogRAG 0.616 0.142 0.228 0.546 0.000 0.123 0.447
OpenRCA-D 0.563 0.175 0.179 0.464 0.089 0.142 0.510
Qwen-RAG0.990 0.9570.719 0.709 0.258 0.483 0.119
Kimi-RAG0.990 1.0000.566 0.712 0.245 0.517 0.116
MiniMax-RAG0.983 0.9370.434 0.705 0.195 0.368 0.222
DeepSeek-RAG0.914 0.9340.838 0.666 0.222 0.530 0.103
GLM-ZS0.997 0.9930.321 0.712 0.073 0.315 0.288
GLM-RAG0.997 0.9970.722 0.709 0.242 0.603 0.043
RCA-only + fixed mapper
Keyword 0.116 0.063 0.149 0.103 0.026 0.089 0.775
Rule 0.338 0.228 0.298 0.278 0.139 0.222 0.642
NB0.679 0.5360.639 0.444 0.318 0.467 0.281
Epsilon 0.149 0.053 0.166 0.096 0.013 0.013 0.752
RCD 0.149 0.053 0.166 0.096 0.013 0.013 0.752
Random Walk 0.149 0.053 0.166 0.096 0.013 0.013 0.752
BARO 0.136 0.122 0.291 0.000 0.026 0.281 0.709
CIRCA 0.149 0.053 0.166 0.096 0.013 0.013 0.752
LogFormer 0.381 0.116 0.186 0.106 0.086 0.182 0.755
OneLog0.5330.086 0.169 0.166 0.109 0.152 0.732
Validity. Our results show that high-quality localization does
not guarantee a valid operation-target decision. The action-
level metrics also identify where recovery-action selection
breaks. Some methods can often name a plausible target while
failing to choose a valid operation or plan structure, as seen in
LogPrompt, LogRAG, and OpenRCA-D. Other methods have
low No-op rates but still limited Recovery Validity, showing
that the problem is not merely refusal or empty output. Thus,
Recovery Validity is not a redundant metric after RCA; it
captures whether the method can translate evidence into an
action that satisfies the incident-specific action space.
Summary:Diagnostic correctness and recovery-action va-
lidity diverge substantially on the same incident popula-
tion. This divergence motivates action-level metrics for
recovery-oriented methods, including operation hit, target
hit, exact match, and recovery validity.
B. RQ2: Modality Effects on Diagnosis and Recovery
Experimental Setup.To investigate how incident evidence
affects diagnosis and recovery-action validity, we conduct a
modality ablation study on methods that support the same
three-view evaluation protocol. For each method, we evaluate
three evidence settings: logs only, logs with Kubernetes events,
and full evidence (logs, events, metrics, and runtime state).
This allows us to assess how additional operational evidence
affects diagnosis accuracy and recovery validity. Table V
reports diagnosis accuracy, recovery validity, and no-op rate
under each setting.
Results and Analysis.Richer evidence generally improves
recovery, but recovery-action validity remains an additional
action-level metric. The effect is particularly evident for
weaker methods. For example, NB improves from 6.6% Re-
covery Validity with logs only to 46.0% under full evidence,
indicating that events, metrics, and runtime state provide
substantial information beyond raw logs. Similar trends canalso be observed for several LLM methods, although the
magnitude of improvement varies across models. Neverthe-
less, additional evidence alone does not determine action
validity. For instance, with full evidence, LogPrompt achieves
over 55% RCA-Service accuracy, while its Recovery Validity
remains close to 12%, indicating that a correct diagnosis
does not necessarily translate into executable recovery actions.
Even among stronger RAG-constrained backbones, models can
achieve near-perfect RCA-Service accuracy yet recover only
about half of the incidents correctly.
Summary:Additional evidence improves both diagnosis
and recovery in several methods, but recovery-action valid-
ity remains an additional action-level metric. The remain-
ing errors indicate that action selection requires operation
semantics and incident-specific constraints beyond richer
observation alone.
C. RQ3: Remaining Recovery Errors After Correct Diagnosis
Experimental Setup.To examine whether recovery failures
mainly stem from diagnosis errors, we condition our analysis
on correct RCA cases. We reconstruct the correct-RCA event
Cfrom service and fault-type labels, and measurePr(C),V C,
andR C. This analysis tests whether methods still fail to choose
valid recovery operations and targets after correctly diagnosing
the incident.
Results and Analysis.Table VI shows substantial recovery
errors even after correct RCA. The invalid-action rate gives
direct evidence: when both the root-cause service and fault
type are correctly identified, zero-shot and few-shot LLMs
still leaveE C= 0.591–0.753, and RAG-based LLMs leave
EC= 0.395–0.620. These results isolate recovery failures
from service- and fault-type misclassification. They show that
identifying what failed provides only part of the information
needed for recovery; the method must still select a valid
operation and target.
RCA-only methods with Mapper serve as a diagnostic
control. Their scores are determined by a fixed mapping policy,
so they measure how far service and fault-type labels can
be carried by a coarse service/type-to-action rule. Gold RCA
+ Mapper reachesV C= 0.599, showing that service/type
labels alone do not encode all fields required by an incident-
specific operation-target plan. Valid recovery depends on finer-
grained operational semantics, including configuration scopes,
resource constraints, and operation preconditions.
Summary:Correct service and fault-type diagnosis still
leaves many invalid recovery decisions. The invalid-action
rate quantifies recovery-planning errors that are not ex-
plained by coarse RCA mistakes.
D. RQ4: Error Sources in Recovery-Action Selection
Experimental Setup.To identify what makes recovery-
action selection difficult, we analyze recovery failures from
two complementary views. The first view categorizes invalid

TABLE V
MULTI-MODAL OBSERVATION ABLATION. EACH ENTRY REPORTS5-FOLD MEAN±STANDARD DEVIATION IN PERCENTAGE. SHADED CELLS MARK THE
BEST INPUT SETTING FOR EACH METHOD AND METRIC. ZERO-SHOT AND FEW-SHOTLLMBACKBONES ARE OMITTED.
MethodLogs Logs+Events Logs+Events+Metrics
RCA-S RCA-Type Recovery No-op RCA-S RCA-Type Recovery No-op RCA-S RCA-Type Recovery No-op
Rule 3.3±1.2 10.3±0.8 12.9±1.3 72.5±5.2 34.1±5.316.6±2.7 23.5±2.665.5±3.2 33.8±5.5 22.9±3.622.2±2.4 64.2±4.4
NB 17.9±3.0 16.2±5.0 6.6±4.3 61.9±8.3 64.9±4.7 42.0±8.0 32.4±6.2 30.4±9.5 68.6±3.3 52.7±6.0 46.0±5.2 28.8±6.2
LogRAG 12.6±2.6 11.9±1.5 2.6±0.9 65.5±7.1 59.6±6.6 13.9±3.710.3±4.6 46.0±2.959.3±7.8 13.9±3.3 11.9±4.0 46.0±4.2
LogPrompt 15.6±1.0 11.6±2.1 2.3±2.2 68.9±3.6 59.6±2.5 13.3±1.7 8.3±3.8 47.0±5.1 61.6±3.0 16.3±3.7 12.3±2.6 46.0±4.2
OpenRCA-D 12.6±2.8 10.9±1.5 4.3±1.9 75.5±2.9 52.7±3.7 13.9±1.9 11.3±4.3 55.6±1.6 55.6±2.7 15.2±1.5 12.2±3.0 52.6±3.0
Qwen-RAG 93.1±3.1 93.4±1.7 32.1±4.8 24.8±2.8 99.3±0.9 96.7±1.243.4±4.7 17.6±5.0 99.0±0.9 96.4±1.4 48.0±3.2 11.9±3.0
Kimi-RAG 97.4±1.5 100.0±0.054.6±3.7 9.9±2.6 99.3±0.9 100.0±0.0 56.0±2.6 8.3±2.499.0±1.5 100.0±0.051.0±5.3 12.6±4.9
MiniMax-RAG 71.8±42.5 69.8±41.3 23.9±14.3 43.5±33.1 98.3±1.2 96.7±2.632.8±4.2 25.5±2.4 97.7±1.9 92.4±2.5 36.1±2.8 22.9±2.9
DeepSeek-RAG 85.7±7.3 97.3±1.954.5±11.5 8.3±5.1 94.7±2.196.7±1.7 54.3±1.6 9.3±3.5 94.1±3.6 96.7±2.3 54.7±5.4 7.0±2.1
GLM-RAG 99.7±0.7 100.0±0.0 61.9±3.2 1.0±0.9 100.0±0.0 100.0±0.060.6±2.7 3.3±2.0 100.0±0.0 100.0±0.0 61.9±3.42.0±0.7
TABLE VI
RECOVERY ERRORS AFTER CORRECTRCA.CDENOTES CORRECTRCA,
VC= Pr(V | C),ANDE C= 1−V CIS THE INVALID-ACTION RATE
AMONG CORRECTLY DIAGNOSED INCIDENTS.
MethodPr(C)V CECMethodPr(C)V CEC
Keyword 0.000 – – MiniMax-RAG 0.924 0.380 0.620
Rule-L 0.010 1.000 0.000 DeepSeek-ZS 0.805 0.337 0.663
Rule-LE 0.126 1.000 0.000 DeepSeek-FS 0.728 0.409 0.591
Rule-LEM 0.212 0.984 0.016 DeepSeek-RAG 0.897 0.565 0.435
LogPrompt 0.149 0.511 0.489 GLM-ZS 0.990 0.318 0.682
LogRAG 0.132 0.675 0.325 GLM-FS 0.944 0.354 0.646
OpenRCA-D 0.149 0.778 0.222 GLM-RAG0.9970.605 0.395
OpenRCA-C 0.000 – – NB 0.434 0.832 0.168
Qwen-ZS 0.891 0.335 0.665 Epsilon 0.000 – –
Qwen-FS 0.758 0.354 0.646 Bayesian 0.000 – –
Qwen-RAG 0.947 0.500 0.500 RCD 0.000 – –
Kimi-ZS 0.993 0.320 0.680 RandomWalk 0.000 – –
Kimi-FS 0.907 0.380 0.620 BARO 0.000 – –
Kimi-RAG 0.990 0.522 0.478 CIRCA 0.000 – –
MiniMax-ZS 0.897 0.247 0.753 LogFormer 0.056 1.000 0.000
MiniMax-FS 0.825 0.285 0.715 OneLog 0.060 1.000 0.000
Diagnostic controlGold RCA + Mapper 1.000 0.599 0.401
recovery predictions into operation, target, structure, and no-
op errors. The second view groups predictions by fault type
and compares RCA-Service accuracy with Recovery Validity.
This analysis focuses on the five RAG-based LLM backbones
because they achieve the strongest diagnostic performance, yet
still exhibit a clear recovery-action gap.
Results and Analysis.Table VII identifies wrong operation
selection as the dominant source of invalid plans, accounting
for 67.9% of all invalid predictions. Invalid plan structure
contributes another 29.7%, while no-op or empty plans and
wrong target/dependency errors account for only 1.1% and
1.3%, respectively. The main difficulty therefore lies in choos-
ing an incident-appropriate operation and expressing it in a
valid recovery-plan structure, rather than merely deciding to
act or locating the affected service.
Figure 7 shows that this difficulty varies substantially across
fault categories. Pod failure, CPU saturation, and network
delay are recovered more reliably because their valid actions
align with common operational routines such as restart orTABLE VII
RECOVERY ERROR TAXONOMY
Error Type Count Ratio Interpretation
Wrong operation 513 67.9% The method selects an op-
eration that is not valid for
the incident.
Invalid plan structure 224 29.7% The operation is plausible,
but the plan does not sat-
isfy the recovery-plan se-
mantics.
Wrong target/dependency 10 1.3% The operation is correct,
but the target service or de-
pendency is invalid.
No-op or empty plan 8 1.1% The method produces no
executable recovery action.
Qwen KimiMini
MaxDeep
Seek GLM
CPU saturation
DNS fault
DNS random
HTTP abort
HTTP error
Memory pressure
Network delay
Pod failure1.00 1.00 1.00 1.00 1.00
0.98 1.00 1.00 0.88 1.00
1.00 1.00 1.00 0.92 1.00
1.00 0.97 0.97 0.82 1.00
0.96 0.96 0.94 0.85 0.98
1.00 1.00 1.00 0.97 1.00
1.00 1.00 1.00 1.00 1.00
1.00 1.00 0.97 0.97 1.00
(a) RCA-Service
Qwen KimiMini
MaxDeep
Seek GLM
CPU saturation
DNS fault
DNS random
HTTP abort
HTTP error
Memory pressure
Network delay
Pod failure1.00 1.00 1.00 1.00 1.00
0.00 0.00 0.00 0.00 0.00
0.00 0.00 0.00 0.00 0.00
0.59 0.82 0.33 0.85 1.00
0.56 0.42 0.15 0.60 0.75
0.32 0.56 0.29 0.50 0.65
1.00 1.00 0.92 0.92 1.00
1.00 1.00 0.92 0.92 1.00
(b) Recovery Validity
Fig. 7. Per-fault difficulty across RAG-based LLM backbones. Panel (a)
shows RCA-Service accuracy, while panel (b) shows recovery validity.
scale-out. DNS faults exhibit a stronger diagnosis-to-action
gap: RCA-Service accuracy remains high across LLMs, rang-
ing from 0.88 to 1.00, while recovery validity is 0.00 for
both DNS categories. HTTP faults show similar semantic
difficulty with larger model-dependent variation, where RCA-

Service remains high but recovery validity ranges from 0.15
to 1.00 across models and subtypes. Memory pressure further
exposes a resource-management failure mode, with strong
service localization but recovery validity limited to 0.29–0.65.
Summary:Future recovery-oriented methods should com-
plement service localization with explicit modeling of op-
eration semantics, dependency scopes, configuration con-
straints, and resource preconditions.
E. RQ5: Live Replay Check
Experimental Setup.To complement offline recovery va-
lidity with live execution behavior, we replay model-generated
recovery plans on a live Kubernetes cluster. We use Qwen-
RAG as the representative method and replay the full 302-
incident benchmark. For each case, we inject the original
Chaos Mesh fault into the cluster, execute the predicted
recovery action, and check post-recovery system health within
a 75-second window. To keep the online check aligned with
the offline metric, Table VIII reports a validity-gated replay
outcome: a case is counted as recovered only when the
prediction satisfies the offline recovery-validity specification
and the replay restores service health.
Results and Analysis.Table VIII shows that validity-gated
live replay is consistent with offline Recovery Validity. Among
the 302 replayed incidents, 146 Qwen-RAG predictions are
both offline-valid and successful in live execution, giving a
replay-consistent recovery rate of 48.3%. Since replay re-
injects the original Chaos Mesh fault and executes the pre-
dicted operation in the cluster, this result confirms that offline-
valid plans are executable and can restore service health under
the replay protocol. The per-fault results follow the offline
error pattern. CPU saturation, network delay, and pod failure
reach 100.0% replay RV because their repairs correspond
to direct routines such as scale-out or restart. In contrast,
DNS faults remain at 0.0% because the predicted plans often
omit the required dependency target. HTTP and memory-
pressure cases recover only partially, reflecting the need for
configuration- or resource-level recovery semantics. Overall,
live replay supports the main finding that the bottleneck is
producing a complete operation-target plan, rather than merely
executing an already valid action.
Summary:Validity-gated live replay confirms the offline
recovery-validity signal, indicating that invalid recovery
mainly comes from incomplete operation-target planning
rather than execution failure.
Supplementary Material.More details and results are
available in the supplementary material, including the full
dataset, annotation schema, recovery-plan semantics, method
implementation details, and additional analyses.
VII. DISCUSSION
Recovery Requires Additional Action-Level Metrics.
R2Act does not argue that RCA benchmarks should evaluate
recovery, nor that RCA methods are expected to solve recoveryTABLE VIII
VALIDITY-GATED LIVE REPLAY OUTCOMES FORQWEN-RAGOVER THE
FULL302-INCIDENT BENCHMARK.
Fault type Plan RV-valid Rec. Replay RV
CPU saturation 36 36 36 100.0%
DNS fault 48 0 0 0.0%
DNS random 48 0 0 0.0%
HTTP abort 39 23 23 59.0%
HTTP error 48 27 27 56.2%
Memory pressure 34 11 11 32.4%
Network delay 13 13 13 100.0%
Pod failure 36 36 36 100.0%
Total 302 146 146 48.3%
Plan: planned attempts;RV-valid: predictions satisfying the offline
recovery-validity specification;Rec.: offline-valid predictions that restore
service health in replay;Replay RV: validity-gated replay recovery rate.
by themselves. Diagnosis and recovery can use the same
incidents and evidence, but they answer different questions:
RCA metrics ask what failed, while recovery-action metrics
ask whether the selected operation and target can repair the
incident. RQ1 and RQ3 show that this distinction matters
because recovery-action validity diverges from diagnostic cor-
rectness even after correct RCA.
Recovery Depends on Action Semantics.The per-fault
results show why diagnosis alone is insufficient. Lifecycle
and some network cases align with direct routines such as
restart or scale-out, but DNS, HTTP, and memory cases
require reasoning about dependencies, configuration scopes, or
resource constraints. These cases show that recovery actions
should be modeled as typed operational decisions, not as free-
form recommendations attached to RCA outputs.
More Evidence Helps but Is Insufficient.RQ2 shows that
events, metrics, and runtime state improve recovery validity
for several methods. Yet Recovery Validity remains far below
RCA-Service even with full evidence, indicating that many
failures come from mapping evidence to a valid operation and
target rather than from missing observations alone.
Offline Validity Needs Execution Checks.The online
replay experiment connects offline recovery validity with
execution over the full 302-incident benchmark: 146 Qwen-
RAG predictions are both offline-valid and recovered in live
execution. This does not collapse the diagnosis-to-action gap.
Offline RV remains the controlling metric because it checks
whether the predicted plan contains the required operation,
target, and incident-specific fields before replay success is
counted. The two checks are therefore complementary: offline
labels encode operational constraints, and replay verifies re-
stored health under those constraints.
Scope and Limitations.The current benchmark instance
is built on Online Boutique and covers six service roles and
eight fault categories. As a controlled testbed, it does not yet
capture multiple production systems, larger service graphs, or
organization-specific recovery policies. To support extension,
R2Act separates the benchmark schema, quality gates, and

evaluator from the particular system used in the current
release. New systems, fault campaigns, and recovery actions
can therefore be appended through the same pipeline. The
current release prioritizes auditable recovery under controlled
fault injection, while the schema and evaluator also support
expanding with additional operator-validated equivalent plans.
VIII. CONCLUSION
This paper introduces R2Act, a recovery-action evaluation
framework for microservice failure recovery. We instantiate it
as a 302-incident benchmark dataset that connects multi-modal
incident evidence, RCA labels, incident-specific action spaces,
recovery-plan annotations, and selected live replay evidence.
RCA-centered metrics remain appropriate for diagnosis, but
they do not determine whether a method can choose an
admissible operation-target plan. The observed failures are
driven mainly by operation-selection and plan-structure errors
in faults that require dependency, configuration, or resource
semantics. These findings support judging future recovery-
oriented methods not only by diagnostic correctness, but also
by whether their actions are valid, executable, and able to
restore post-recovery health. Full live replay further shows that
offline action validity needs execution checks because cluster
readiness, executor coverage, and health-check granularity
can affect whether an executed plan restores service health.
Together, these results position recovery validity as a first-class
objective for recovery-oriented incident-response evaluation.
REFERENCES
[1] K. Sarda, Z. Namrud, M. Litoiu, L. Shwartz, and I. Watts, “Lever-
aging large language models for the auto-remediation of microservice
applications: An experimental study,” inCompanion Proceedings of the
32nd ACM International Conference on the Foundations of Software
Engineering, 2024, pp. 163–174.
[2] Z. Zhong, R. Fu, M. Ma, S. Zhang, Y . Sun, C. Bansal, and D. Pei,
“Llm-enhanced failure localization in microservices: Integrating multi-
modal data and expert interpretation,”IEEE Transactions on Services
Computing, pp. 1–14, 2026.
[3] X. Zhou, X. Peng, T. Xie, J. Sun, C. Xu, C. Ji, and W. Zhao, “Bench-
marking microservice systems for software engineering research,” in
Proceedings of the 40th International Conference on Software Engi-
neering: Companion Proceeedings, 2018, pp. 323–324.
[4] D. Liu, C. He, X. Peng, F. Lin, C. Zhang, S. Gong, Z. Li, J. Ou,
and Z. Wu, “Microhecl: High-efficient root cause localization in large-
scale microservice systems,” in2021 IEEE/ACM 43rd International
Conference on Software Engineering: Software Engineering in Practice
(ICSE-SEIP). IEEE, 2021, pp. 338–347.
[5] Y . Sun, Z. Lin, B. Shi, S. Zhang, S. Ma, P. Jin, Z. Zhong, L. Pan,
Y . Guo, and D. Pei, “Interpretable failure localization for microservice
systems based on graph autoencoder,”ACM Transactions on Software
Engineering and Methodology, vol. 34, no. 2, pp. 1–28, 2025.
[6] J. Zhu, S. He, P. He, J. Liu, and M. R. Lyu, “Loghub: A large collection
of system log datasets for ai-driven log analytics,” in2023 IEEE 34th
International Symposium on Software Reliability Engineering (ISSRE).
IEEE, 2023, pp. 355–366.
[7] Z. Jiang, J. Liu, J. Huang, Y . Li, Y . Huo, J. Gu, Z. Chen, J. Zhu, and
M. R. Lyu, “A large-scale evaluation for log parsing techniques: How
far are we?” inProceedings of the 33rd ACM SIGSOFT International
Symposium on Software Testing and Analysis, 2024, pp. 223–234.
[8] T. Cui, S. Ma, Z. Chen, T. Xiao, C. Zhao, S. Tao, Y . Liu, S. Zhang,
D. Lin, C. Liuet al., “Logeval: A comprehensive benchmark suite for
llms in log analysis,”Empirical Software Engineering, vol. 30, no. 6, p.
173, 2025.[9] Y . Wang, Z. Zhu, Q. Fu, Y . Ma, and P. He, “Mrca: Metric-level
root cause analysis for microservices via multi-modal data,” in
Proceedings of the 39th IEEE/ACM International Conference on
Automated Software Engineering, ser. ASE ’24. New York, NY , USA:
Association for Computing Machinery, 2024, p. 1057–1068. [Online].
Available: https://doi.org/10.1145/3691620.3695485
[10] L. Pham, H. Zhang, H. Ha, F. Salim, and X. Zhang, “Rcaeval: a bench-
mark for root cause analysis of microservice systems with telemetry
data,” inCompanion Proceedings of the ACM on Web Conference 2025,
2025, pp. 777–780.
[11] W. Xu, J. Luo, T. Huang, K. Sui, J. Geng, Q. Ma, I. Akasaka,
X. Shi, J. Tang, and P. Cai, “Logsage: An llm-based framework for
ci/cd failure detection and remediation with industrial validation,” in
2025 40th IEEE/ACM International Conference on Automated Software
Engineering (ASE). IEEE, 2025, pp. 3742–3753.
[12] M. E. Barnes, T. A. Ghaleb, and S. Hassan, “Logsieve: Task-aware
ci log reduction for sustainable llm-based analysis,”arXiv preprint
arXiv:2601.20148, 2026.
[13] J. Xu, Q. Zhang, Z. Zhong, S. He, C. Zhang, Q. Lin, D. Pei, P. He,
D. Zhang, and Q. Zhang, “Openrca: Can large language models locate
the root cause of software failures?” inThe thirteenth international
conference on learning representations, 2025.
[14] Z. Wang, Z. Liu, Y . Zhang, A. Zhong, J. Wang, F. Yin, L. Fan,
L. Wu, and Q. Wen, “Rcagent: Cloud root cause analysis by autonomous
agents with tool-augmented large language models,”arXiv preprint
arXiv:2310.16340, 2023.
[15] Y . Chen, H. Xie, M. Ma, Y . Kang, X. Gao, L. Shi, Y . Cao, X. Gao,
H. Fan, M. Wenet al., “Automatic root cause analysis via large language
models for cloud incidents,” inProceedings of the Nineteenth European
Conference on Computer Systems, 2024, pp. 674–688.
[16] A. Bucchiarone, C. Guidi, I. Lanese, N. Bencomo, and J. Spillner,
“A mape-k approach to autonomic microservices,” in2022 IEEE 19th
International Conference on Software Architecture Companion (ICSA-
C). IEEE, 2022, pp. 100–103.
[17] L. Zhang, Y . Zhai, T. Jia, C. Duan, M. He, L. Pan, Z. Liu, B. Ding, and
Y . Li, “Microremed: Benchmarking llms in microservices remediation,”
arXiv preprint arXiv:2511.01166, 2025.
[18] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and
K. Narasimhan, “Swe-bench: Can language models resolve real-world
github issues?” inInternational Conference on Learning Representa-
tions, vol. 2024, 2024, pp. 54 107–54 157.
[19] M. H. M. Bhuiyan, A. S. Parthasarathy, N. Vasilakis, M. Pradel,
and C.-A. Staicu, “Secbench. js: An executable security benchmark
suite for server-side javascript,” in2023 IEEE/ACM 45th International
Conference on Software Engineering (ICSE). IEEE, 2023, pp. 1059–
1070.
[20] L. Pham, H. Ha, and H. Zhang, “Root cause analysis for microservice
system based on causal inference: How far are we?” inProceedings of
the 39th IEEE/ACM International Conference on Automated Software
Engineering, 2024, pp. 706–715.
[21] C. Liu, W. Yang, H. Mittal, M. Singh, D. Sahoo, and S. C. Hoi,
“Pyrca: A library for metric-based root cause analysis,”arXiv preprint
arXiv:2306.11417, 2023.
[22] Y . Liu, S. Tao, W. Meng, F. Yao, X. Zhao, and H. Yang, “Logprompt:
Prompt engineering towards zero-shot and interpretable log analysis,” in
Proceedings of the 2024 IEEE/ACM 46th international conference on
software engineering: Companion proceedings, 2024, pp. 364–365.
[23] W. Zhang, Q. Zhang, E. Yu, Y . Ren, Y . Meng, M. Qiu, and J. Wang,
“Leveraging rag-enhanced large language model for semi-supervised log
anomaly detection,” in2024 IEEE 35th International Symposium on
Software Reliability Engineering (ISSRE). IEEE, 2024, pp. 168–179.
[24] Y . Gao, Z. Cai, and B. Yang, “Rcaflow: A workflow-informed hierarchi-
cal planning multi-agent system for root cause analysis,” inProceedings
of the AAAI Conference on Artificial Intelligence, vol. 40, no. 1, 2026,
pp. 300–308.
[25] R. Chen, Y . Pu, J. Xin, J. Wang, X. Liao, K. Zhang, and
W. Wu, “Grace: A strategic llm-enhanced graph reinforcement learning
framework for adaptive fault recovery in microservice systems,” in
Service-Oriented Computing: 23rd International Conference, ICSOC
2025, Shenzhen, China, December 1–4, 2025, Proceedings, Part
I. Berlin, Heidelberg: Springer-Verlag, 2025, p. 155–170. [Online].
Available: https://doi.org/10.1007/978-981-95-5012-8 12
[26] T. Ahmed, S. Ghosh, C. Bansal, T. Zimmermann, X. Zhang, and
S. Rajmohan, “Recommending root-cause and mitigation steps for cloud

incidents using large language models,” inProceedings of the 45th
IEEE/ACM International Conference on Software Engineering, 2023,
pp. 1737–1749.
[27] E. Malul, Y . Meidan, D. Mimran, Y . Elovici, and A. Shabtai,
“Genkubesec: Llm-based kubernetes misconfiguration detection, local-
ization, reasoning, and remediation,”arXiv preprint arXiv:2405.19954,
2024.
[28] W. Zhang, Z. Yang, F. Peng, L. Zhang, Y . Chen, and R. Chen,
“Galr: Graph-based root cause localization and llm-assisted recovery
for microservice systems,”Electronics, vol. 15, no. 1, p. 243, 2026.
[29] H. Guo, J. Yang, J. Liu, J. Bai, B. Wang, Z. Li, T. Zheng, B. Zhang,
J. Peng, and Q. Tian, “Logformer: A pre-train and tuning pipeline for log
anomaly detection,” inProceedings of the AAAI conference on artificial
intelligence, vol. 38, no. 1, 2024, pp. 135–143.
[30] Google Cloud, “Online boutique,” https://github.com/
GoogleCloudPlatform/microservices-demo, 2026, accessed 2026-
05-17.
[31] The Kubernetes Authors, “Kubernetes documentation,”
https://kubernetes.io/docs/, 2026, accessed 2026-05-17.
[32] Prometheus Authors, “Prometheus monitoring system,”
https://prometheus.io/docs/, 2026, accessed 2026-05-17.
[33] Chaos Mesh Authors, “Chaos mesh documentation,” https://chaos-mesh.
org/docs/, 2026, accessed 2026-05-17.
[34] S. Hashemi and M. M ¨antyl ¨a, “Onelog: towards end-to-end software log
anomaly detection,”Automated Software Engineering, vol. 31, no. 2,
p. 37, 2024.