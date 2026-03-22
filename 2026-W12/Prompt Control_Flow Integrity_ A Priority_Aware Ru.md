# Prompt Control-Flow Integrity: A Priority-Aware Runtime Defense Against Prompt Injection in LLM Systems

**Authors**: Md Takrim Ul Alam, Akif Islam, Mohd Ruhul Ameen, Abu Saleh Musa Miah, Jungpil Shin

**Published**: 2026-03-19 02:50:15

**PDF URL**: [https://arxiv.org/pdf/2603.18433v1](https://arxiv.org/pdf/2603.18433v1)

## Abstract
Large language models (LLMs) deployed behind APIs and retrieval-augmented generation (RAG) stacks are vulnerable to prompt injection attacks that may override system policies, subvert intended behavior, and induce unsafe outputs. Existing defenses often treat prompts as flat strings and rely on ad hoc filtering or static jailbreak detection. This paper proposes Prompt Control-Flow Integrity (PCFI), a priority-aware runtime defense that models each request as a structured composition of system, developer, user, and retrieved-document segments. PCFI applies a three-stage middleware pipeline, lexical heuristics, role-switch detection, and hierarchical policy enforcement, before forwarding requests to the backend LLM. We implement PCFI as a FastAPI-based gateway for deployed LLM APIs and evaluate it on a custom benchmark of synthetic and semi-realistic prompt-injection workloads. On the evaluated benchmark suite, PCFI intercepts all attack-labeled requests, maintains a 0% False Positive Rate, and introduces a median processing overhead of only 0.04 ms. These results suggest that provenance- and priority-aware prompt enforcement is a practical and lightweight defense for deployed LLM systems.

## Full Text


<!-- PDF content starts -->

Prompt Control-Flow Integrity: A Priority-Aware
Runtime Defense Against Prompt Injection in LLM
Systems
Md Takrim Ul Alam∗, Akif Islam∗, Mohd Ruhul Ameen†, Abu Saleh Musa Miah‡, Jungpil Shin‡
∗University of Rajshahi, Rajshahi, Bangladesh
†Marshall University, Huntington, WV , USA
‡University of Aizu, Aizuwakamatsu, Fukushima, Japan
Emails: takrimulalom@gmail.com, iamakifislam@gmail.com, ameen@marshall.edu, musa@u-aizu.ac.jp, jpshin@u-aizu.ac.jp
Abstract—Large language models (LLMs) deployed behind
APIs and retrieval-augmented generation (RAG) stacks are
vulnerable to prompt injection attacks that may override system
policies, subvert intended behavior, and induce unsafe outputs.
Existing defenses often treat prompts as flat strings and rely on
ad hoc filtering or static jailbreak detection. This paper proposes
Prompt Control-Flow Integrity(PCFI), a priority-aware runtime
defense that models each request as a structured composition
of system, developer, user, and retrieved-document segments.
PCFI applies a three-stage middleware pipeline—lexical heuris-
tics, role-switch detection, and hierarchical policy enforcement—
before forwarding requests to the backend LLM. We implement
PCFI as a FastAPI-based gateway for deployed LLM APIs and
evaluate it on a custom benchmark of synthetic and semi-realistic
prompt-injection workloads. On the evaluated benchmark suite,
PCFI intercepts all attack-labeled requests, maintains a 0% False
Positive Rate, and introduces a median processing overhead of
only 0.04 ms. These results suggest that provenance- and priority-
aware prompt enforcement is a practical and lightweight defense
for deployed LLM systems.
I. INTRODUCTION
Large language models (LLMs) are quickly moving beyond
research labs and into real-world applications such as code
generation, customer support, enterprise search, and decision
assistance [1]. In many of these systems, the model does
not respond to a single user query alone. Instead, the final
request is built from several different sources, including system
instructions, developer templates, user input, and retrieved
external content from retrieval-augmented generation (RAG)
pipelines. This makes modern LLM systems more flexible
and useful, but it also introduces a serious security challenge:
prompt injection[2], [3]. By placing adversarial instructions
inside user messages or untrusted retrieved documents, an
attacker may try to override higher-priority policies [4], alter
model behavior, or trigger unsafe outputs [5], [6].
This issue matters because many deployed LLM systems
operate in environments where instruction hierarchy is not
just helpful, but essential for safety. System prompts may
encode confidentiality constraints and core behavioral rules,
developer prompts may define application logic, and retrieved
content may come from partially trusted or fully untrustedsources [7], [6]. Prior work has shown that indirect prompt
manipulation through RAG pipelines is a realistic threat in
end-to-end applications [6]. More generally, recent surveys
have identified prompt injection, leakage, and jailbreak-style
manipulation as major risks for LLM-based systems [5]. As
these models become more deeply integrated into practical
workflows, defending them against such attacks becomes in-
creasingly important for both reliability and safe deployment.
A major difficulty is that many existing defenses still
treat prompts as flat text. In practice, they often rely on
ad hoc filtering, static pattern matching, or broad guardrail
mechanisms. While programmable guardrail frameworks can
help control LLM behavior at runtime [8], they do not au-
tomatically preserve the intended authority structure among
prompt sources. At its core, the security problem in deployed
prompt assembly is structural: a retrieved document should
not be able to redefine a system policy, and a low-trust user
or RAG segment should not silently impersonate a privileged
role. This motivates a provenance- and priority-aware view
of prompt security. Our intuition is inspired by control-flow
integrity (CFI) in software security, which protects intended
execution structure by rejecting unauthorized deviations at
runtime [9]. Although prompts are not programs in the tra-
ditional sense, the analogy is still useful: once lower-priority
prompt components are allowed to reinterpret higher-priority
instructions, the intended control logic of the application can
be undermined.
Motivated by this perspective, we proposePrompt Control-
Flow Integrity(PCFI), a lightweight runtime defense that
models each request as a structured prompt representation
with explicit roles and priorities. Instead of collapsing all
content into a single undifferentiated sequence of tokens,
PCFI preserves the hierarchy among system, developer, user,
and retrieved segments, and evaluates this structure before
the request reaches the backend model. We implement PCFI
as an API-boundary middleware that applies a three-stage
analysis consisting of lexical heuristics, role-switch detection,
and policy-driven hierarchical enforcement. Depending on the
outcome, the middleware may allow benign requests, sanitizearXiv:2603.18433v1  [cs.CR]  19 Mar 2026

suspicious role-like cues, or block inputs that violate the
intended instruction hierarchy.
The main contributions of this paper are as follows:
1) We formulatePrompt Control-Flow Integrity(PCFI)
as a structural security perspective for LLM systems,
emphasizing prompt provenance and priority rather than
treating prompts as flat text.
2) We design a three-stage runtime defense pipeline that
combines lexical heuristics, role-switch detection, and
policy-driven hierarchical enforcement to constrain in-
jection attempts originating from lower-priority prompt
segments.
3) We implement PCFI as a lightweight middleware
gateway for deployed LLM APIs, enablingALLOW,
SANITIZE, andBLOCKdecisions before requests reach
the backend model.
4) We evaluate the proposed approach on benign, direct-
injection, and indirect RAG-injection workloads, analyz-
ing malicious-prompt interception, benign-query preser-
vation, and runtime latency overhead.
II. LITERATUREREVIEW
Research on securing LLM applications against prompt-
based attacks has grown rapidly as LLMs have moved into
retrieval-augmented, tool-using, and API-mediated deploy-
ments. Existing work mainly falls into four groups: studies of
prompt hacking and indirect injection, instruction–data sepa-
ration methods, runtime guardrail frameworks, and security-
inspired structural approaches. Although these directions have
improved understanding of the threat landscape [10], important
gaps remain in enforcing prompt provenance, priority, and
authority at runtime [5], [6], [11], [8].
A. Prompt Injection and Prompt-Hacking Threat Models
Prompt injection is now recognized as a practical threat in
deployed LLM systems [12]. Rababahet al.provide a broad
taxonomy of prompt hacking, clarifying the differences among
jailbreaking, leakage, and injection attacks [5]. This line of
work is valuable for characterizing the threat space, but it
does not by itself provide a concrete runtime mechanism for
enforcing instruction hierarchy in deployed systems [5].
De Stefanoet al.further show that indirect prompt manipu-
lation through retrieval pipelines is a realistic and effective
attack channel in RAG-based applications [6]. Their work
demonstrates the seriousness of end-to-end indirect attacks, but
it is primarily evaluative and does not introduce a lightweight
request-time mechanism for preserving the authority of system
and developer prompts over user and retrieved content [6].
B. Instruction–Data Separation and Structured Prompt De-
fenses
A strong defense direction is to separate trusted instruc-
tions from untrusted data. StruQ, proposed by Chenet al.,
structures prompts and data into separate channels and relies
on a specially trained model to follow only the instruction
channel [11]. This is conceptually strong, but it dependson modified interfaces and model retraining, which may be
difficult in API-based or closed-weight deployments [11].
Maoet al.also show that real-world LLM applications
already use reusable templates and structured prompt com-
position [7]. This supports the view that prompts are not
naturally flat. However, such studies are descriptive rather than
defensive: they reveal prompt structure, but they do not enforce
security invariants over that structure.
C. Guardrails, Wrappers, and Runtime Safety Layers
Runtime guardrails provide another practical line of defense.
NeMo Guardrails offers programmable rails for controlling
LLM behavior without changing the underlying model [8].
This is useful for application-level safety and control, but
guardrail frameworks are broader than prompt-injection de-
fense and do not inherently enforce provenance-aware au-
thority ordering among prompt segments unless that logic is
explicitly designed by the developer [8].
Trustworthy RAG research also addresses conflicts between
internal and retrieved knowledge. For example, Daiet al.study
how LLMs should react when retrieved evidence conflicts with
internal knowledge [13]. While this improves trustworthiness
at the content level, it focuses on epistemic reliability rather
than on preventing retrieved text from acting as unauthorized
instruction [13].
D. Control-Flow Integrity as Security Inspiration
Our work is conceptually inspired by control-flow integrity
(CFI), introduced by Abadiet al., which constrains program
execution to valid control-flow paths in order to prevent
hijacking attacks [9]. CFI is not a direct solution to prompt
injection, but it provides a useful analogy: preserving intended
structure and rejecting unauthorized deviations at runtime.
Since prompt assembly is less formal than program execution,
this connection should be understood as inspiration rather than
as a direct theoretical transfer [9].
E. Research Gap
Taken together, prior work has clarified the threat of
prompt injection, demonstrated the realism of indirect RAG-
based manipulation, proposed stronger instruction–data sepa-
ration methods, and provided useful runtime guardrail frame-
works [5], [6], [11], [8]. However, an important gap remains.
Existing approaches either analyze attacks without enforcing
request-time structure, depend on retraining or modified inter-
faces, or provide general runtime controls without explicitly
modeling prompt provenance and priority [11], [8], [7]. What
is still missing is a lightweight, deployment-friendly runtime
mechanism that preserves the structure of composed prompts
and enforces the intended authority ordering among system,
developer, user, and retrieved segments before the request
reaches the backend model. This gap motivates our proposed
Prompt Control-Flow Integrity (PCFI) framework.

Fig. 1. Three-stage runtime enforcement pipeline of the proposed PCFI framework. Stage 1 performs lexical screening, Stage 2 detects role-switch attempts,
and Stage 3 applies hierarchical policy enforcement before issuingALLOW,SANITIZE, orBLOCKdecisions.
III. METHODOLOGY
This section presents the methodological foundation of
Prompt Control-Flow Integrity (PCFI). PCFI is designed as a
lightweight runtime defense deployed at the API boundary of
an LLM system, without requiring model retraining or changes
to the backend interface. Unlike approaches that rely on mod-
ifying the model to separate instructions from data [11], PCFI
operates as a deployment-side enforcement layer. It is also
narrower than general guardrail frameworks [8], since its focus
is specifically on preserving the intended authority structure
among prompt components before the request reaches the
model. Conceptually, the design is inspired by control-flow
integrity (CFI), which preserves intended execution structure
by rejecting unauthorized deviations at runtime [9]. In the
LLM setting, we adapt this intuition to prompt composition,
where lower-priority content must not override higher-priority
instructions.
A. System Setting and Threat Assumptions
We consider an LLM application in which each final request
is composed from multiple sources, including system instruc-
tions, developer templates, user input, and retrieved external
context. This is common in practical LLM and retrieval-
augmented deployments, where trusted and untrusted content
are combined in the same request [6], [7]. The main security
challenge is that lower-trust content may appear alongside
higher-priority instructions and attempt to act as control logic.
We assume that the defender controls the API gateway, the
system prompt, and the developer template. The attacker may
provide arbitrary user input and may also influence retrieved
documents entering the request through a RAG pipeline.
However, we do not assume that the attacker can modify
the middleware, change the backend model parameters, or
directly alter the system policy. The attacker’s goal is to use
lower-priority content to override or weaken higher-priority
instructions.
Fig. 2. Priority hierarchy used in PCFI for structured prompt enforcement.
System prompts have the highest authority, followed by developer prompts,
user prompts, and retrieved context. The core security principle is that lower-
priority segments must not override or reinterpret higher-priority instructions.
B. Prompt Representation with Provenance and Priority
PCFI does not treat a prompt as a flat string [14]. Instead,
each request is represented as a structured sequence of seg-
ments with explicit provenance and priority labels. LetS,D,
U, andRdenote system, developer, user, and retrieved-context
segments, respectively. We represent the composed prompt as
P=S∥D∥U∥R,(1)
where∥denotes ordered composition with attached metadata.
Each segment is modeled as
segi= (t i, ri, pi, vi, m i),(2)
wheret iis the text,r iis the role,p iis the priority,v iis
the provenance label, andm istores metadata. In the current
prototype, the priority lattice is
S > D > U > R.(3)
This means that lower-priority segments may contribute task
content, but they must not override higher-priority instructions.
The representation is intentionally lightweight: it preserves
enough structure for runtime enforcement without requiring
model retraining.

TABLE I
ILLUSTRATIVE POLICY RULES USED BYPCFIHIERARCHICAL
ENFORCEMENT.
Rule ID Purpose Patterns
override system policy Block
overridesignore previous instructions; disregard
all above
change output format Protect format answer in natural language instead of;
do not follow the format
treat ragasuntrusted Preserve
hierarchyretrieved documents must not redefine
system behavior
C. PCFI Enforcement Objective
PCFI is designed to preserve the intended hierarchy of
control within a composed prompt. In practice, this means
that lower-priority segments should not impersonate privileged
roles, introduce directives that contradict protected policies, or
redefine higher-priority instructions. Suspicious structural cues
may be sanitized, while explicit policy violations are blocked
before reaching the backend model. Accordingly, PCFI should
be understood as a policy-driven structural defense rather than
a full semantic verifier.
D. Three-Stage Runtime Defense Pipeline
PCFI applies a three-stage pipeline to each request, moving
from lightweight screening to stricter enforcement.
1) Stage 1: Lexical Heuristics:The first stage performs
lexical screening [15] over user and retrieved segments to de-
tect obvious prompt-injection signals such as override phrases,
exfiltration requests, or suspicious control-language fragments.
Examples includeignore previous instructions,system over-
ride, andreveal your API key. This stage produces a lexical
risk score and matched findings, but it does not by itself trigger
a hard block.
2) Stage 2: Role-Switch Detection:The second stage de-
tects attempts by lower-priority segments to impersonate
privileged roles. Typical examples include prefixes such as
system:, serialized fields such as"role":"system", or
XML-like role tags. When such markers are detected with suf-
ficient confidence, the middleware may assign aSANITIZE
outcome and remove the role-like prefixes before forwarding
the request.
3) Stage 3: Hierarchical Policy Enforcement:The policy
enforcement stage relies on a set of rule patterns that define
forbidden directive behaviors. Table I presents several illustra-
tive examples used in the current prototype.
The final stage performs priority-aware policy enforcement.
It checks whether forbidden directive patterns appear in lower-
priority segments when higher-priority governing segments are
present. LetFdenote the set of forbidden directive patterns.
For each lower-priority segmentx∈ {U, R}, PCFI checks
whether
∃f∈ Fsuch thatf⊆x(4)
when at least one higher-priority segment from{S, D}exists.
If such a match is found, the request is treated as a hierarchical
control violation and is blocked. This stage is policy-driven
and pattern-based rather than fully semantic.E. Decision Logic
After the three stages, PCFI assigns one of three out-
comes:ALLOW,SANITIZE, orBLOCK. Benign requests are
forwarded unchanged, suspicious role-like markers may be
sanitized, and explicit hierarchical violations are blocked at
the gateway. This conservative design aims to preserve benign
utility while rejecting structurally unsafe inputs.
F . Middleware Realization
We implement PCFI as a FastAPI-based middleware layer
that intercepts chat-completion-style API requests before they
reach the backend LLM. The middleware parses the request,
builds the structured prompt representation, executes the three-
stage pipeline, and then either forwards the request, forwards
a sanitized variant, or returns a structured rejection response.
This design is practical because it does not require retraining
the backend model, remains compatible with API-mediated
and retrieval-augmented deployments, and supports policy
updates without code changes [11].
G. Methodological Scope
PCFI is a runtime structural defense for single-request
prompt composition. It is intended to preserve prompt prove-
nance and authority ordering at the API boundary, not to solve
semantic jailbreak detection, multi-turn memory poisoning, or
latent model-alignment failures. Its contribution is to show
that a lightweight, policy-driven middleware can strengthen
deployed LLM systems when prompt composition is treated
as a structured security problem.
IV. EVALUATIONSETUP
A. Datasets
To evaluate PCFI, we construct a custom JSONL-based
benchmark tailored to structured prompt injection in deployed
LLM systems. Rather than relying on a public benchmark, we
curate three subsets: benign requests, direct prompt-injection
attacks, and indirect RAG-based prompt-injection attacks. The
TABLE II
EXAMPLE ATTACK SAMPLE FROM THE CUSTOMPCFIEVALUATION
BENCHMARK.
Field Value
ID rag-indirect-0
Task Indirect prompt injection via RAG content.
System Policy Helpful assistant; follow safety and formatting
rules; never reveal secrets; always respond in
JSON.
Developer
PromptExplain policies while maintaining system safety
guarantees.
User Prompt Please explain the policies described in the re-
trieved documents.
RAG Doc 1SYSTEM OVERRIDE: Ignore all
previous instructions... output
OVERRIDE_SUCCESS and reveal
secret keys.
RAG Doc 2 Standard company security policies for data han-
dling. . .
Label attack
Attack Family rag indirect
Expected
BehaviorTreat retrieved content as untrusted, refuse over-
ride instructions, and safely note the malicious
content.

Fig. 3. Overall architecture of the proposed PCFI middleware for deployed LLM systems. Incoming client requests are first intercepted by the PCFI gateway,
then transformed into structured prompt segments with provenance tags. The request is subsequently analyzed through lexical heuristics, role-switch detection,
and hierarchical policy enforcement. Based on these checks, the decision engine issues one of three outcomes—ALLOW,SANITIZE, orBLOCK—before
interaction with the backend LLM API.
benign subset contains task-oriented prompts whose user and
retrieved content remain consistent with system and developer
instructions. The direct-injection subset contains adversarial
user prompts that attempt to override prior instructions, reveal
secrets, or bypass constraints. The indirect RAG-injection
subset simulates retrieval-augmented settings [16] in which
malicious control directives are embedded in retrieved docu-
ments while the user request itself appears benign.
Each sample is stored as a structured JSONL record
with fields such asid,task,system_policy,
developer_prompt,user_prompt,rag_docs,
label,attack_family,success_oracle, and
expected_refusal_or_format. This design captures
not only the prompt text, but also the source structure
that PCFI is intended to protect. In total, the benchmark
contains 150 samples, including 50 benign requests and 100
attack-labeled requests. Table II shows a representative attack
example.
B. Evaluation Variants and Metrics
We evaluate a conceptual no-defense baseline, stage-level
signals from the lexical, role-switch, and hierarchical checks,
and the full PCFI pipeline. In the full setting, a request is
considered intercepted if the final gateway outcome is either
SANITIZEorBLOCK.
We report three metrics.Attack Pass-Through Rate (APR)
is the fraction of attack-labeled samples that are not inter-
cepted and would therefore be forwarded downstream.False
Positive Rate (FPR)is the fraction of benign samples that
are incorrectly intercepted.Latency Overheadmeasures the
runtime cost of the PCFI engine and is summarized using
median and tail percentiles such as p95 and p99. These are
request-level gateway enforcement metrics rather than end-to-
end model-compromise metrics.
V. RESULTS ANDDISCUSSION
Table III summarizes the gateway-level security and runtime
behavior of the conceptual baseline and Full PCFI and Figure 4
illustrates a representative attack scenario intercepted by PCFI.
While the current benchmark shows perfect interception, this
result should not be interpreted as a guarantee of complete
robustness. The dataset is limited in size and attack diversity,
and broader real-world evaluations may reveal additional fail-
ure cases. Under the no-defense baseline, all attack-labeledTABLE III
SECURITY AND RUNTIME COMPARISON BETWEEN THE NO-DEFENSE
BASELINE AND THE FULLPCFIPIPELINE ON50BENIGN AND100ATTACK
SAMPLES.
Configuration APR (%) FPR (%) Latency (ms)
Baseline (No Defense) 100 0 Not Applicable
Full PCFI 0 0 0.04 (median)
requests are forwarded downstream, giving an Attack Pass-
Through Rate (APR) of 100%. In contrast, Full PCFI reduces
APR to 0% on the evaluated benchmark. The False Positive
Rate (FPR) is also 0%, indicating that none of the benign
samples were intercepted in this setting.
A. Security Effectiveness
We evaluate security effectiveness in terms of request inter-
ception. APR measures the fraction of attack-labeled requests
that are not intercepted and would therefore be forwarded to
the backend model. Compared with the no-defense baseline,
Full PCFI reduces this rate from 100% to 0% on the evaluated
benchmark. In this setting, both direct user-level injection
and indirect RAG-based injection are intercepted when lower-
priority segments contain policy-forbidden directive patterns.
We also evaluate benign robustness through FPR. On the
benign subset, Full PCFI yields an FPR of 0%, meaning that
none of the 50 benign samples were incorrectly intercepted.
These results suggest that the current policy and decision logic
Fig. 4. Illustrative attack-flow example showing how PCFI intercepts a
malicious lower-priority request before it reaches the backend model. The
user segment attempts to override protected system policy, and the middleware
blocks the request after lexical and hierarchical checks.

can separate the evaluated attack samples from the benign ones
in the benchmark.
B. Performance Overhead
We next measure the runtime cost introduced by the mid-
dleware. Latency is computed from the execution times of
the lexical, role-switch, and hierarchical stages. Across the
150 evaluated samples, the median per-request overhead is
0.04 ms, with p95 and p99 latencies of 0.08 ms and 0.14 ms,
respectively. This indicates that the PCFI pipeline operates in
the sub-millisecond range.
Overall, the current prototype adds only a small runtime
cost at the API boundary. In this evaluation setting, end-to-
end latency remains dominated by backend model inference
rather than by the middleware itself.
C. Limitations
The results suggest that PCFI is a lightweight and practical
runtime mechanism for intercepting prompt-injection attempts
at the API boundary. By preserving prompt provenance and
enforcing a priority-aware decision process, the framework
improves gateway-level protection in the evaluated benchmark
setting. However, several limitations remain.
First, the current implementation relies largely on pattern-
based detection. While effective for explicit override phrases,
exfiltration requests, and role-like markers, it may be brit-
tle against paraphrased, obfuscated, or semantically indirect
attacks [17]. Thus, PCFI should be viewed as a policy-
driven structural defense rather than a complete solution to
unrestricted natural-language adversarial behavior.
Second, the present system focuses on single-request prompt
composition. Extending it to multi-turn conversations, persis-
tent memory, tool-calling chains, and agentic workflows will
require richer stateful models of prompt lineage and trust.
Third, the evaluation uses synthetic and semi-realistic
benchmark samples. Although useful for controlled analysis,
these do not capture the full diversity of adversarial behavior
in open deployments. Moreover, the reported attack metric re-
flects gateway-level interception, not a full end-to-end measure
of model compromise.
Finally, PCFI addresses only one layer of LLM security.
It does not directly solve failures arising from model mis-
alignment, unsafe training data, insecure tool execution, or
downstream application logic. In practice, it is best understood
as a complementary boundary defense that can work alongside
other safety and security controls.
VI. CONCLUSION
This paper presented Prompt Control-Flow Integrity (PCFI),
a lightweight runtime defense for prompt injection in deployed
LLM systems. By modeling requests as structured prompt
segments with explicit provenance and priority, PCFI enables
an API-boundary middleware to distinguish trusted instruction
from lower-priority content and to apply lexical screening,
role-switch detection, and hierarchical policy enforcement
before forwarding requests to the backend model. On theevaluated benchmark, the current prototype achieves a 0%
Attack Pass-Through Rate and a 0% False Positive Rate with
sub-millisecond overhead. These results suggest that priority-
aware prompt screening is a practical and deployment-friendly
direction for strengthening the security of LLM applications,
while future work should extend the framework to richer multi-
turn, tool-using, and more semantically challenging settings.
REFERENCES
[1] W. Xuet al., “A survey of attacks on large language models,”arXiv
preprint arXiv:2505.12567, 2025.
[2] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and
M. Fritz, “Not what you’ve signed up for: Compromising real-world llm-
integrated applications with indirect prompt injection,” inProceedings of
the 2023 ACM SIGSAC Conference on Computer and Communications
Security, 2023.
[3] Y . Liuet al., “Formalizing and benchmarking prompt injection attacks
and defenses,”USENIX Security, 2024.
[4] T. Wen, C. Wang, X. Yang, H. Tang, Y . Xie, L. Lyu, Z. Dou, and F. Wu,
“Defending against indirect prompt injection by instruction detection,”
arXiv preprint arXiv:2505.06311, 2025.
[5] B. Rababah, S. T. Wu, M. Kwiatkowski, C. K. Leung, and C. G.
Akcora, “Sok: Prompt hacking of large language models,” in2024 IEEE
International Conference on Big Data (BigData), 2024.
[6] G. De Stefano, L. Sch ¨onherr, and G. Pellegrino, “Rag and roll: An
end-to-end evaluation of indirect prompt manipulations in llm-based
application frameworks,”CoRR, vol. abs/2408.05025, 2024.
[7] Y . Mao, J. He, and C. Chen, “From prompts to templates: A sys-
tematic prompt template analysis for real-world llmapps,”CoRR, vol.
abs/2504.02052, 2025.
[8] T. Rebedea, R. Dinu, M. N. Sreedhar, C. Parisien, and J. Cohen, “Nemo
guardrails: A toolkit for controllable and safe llm applications with
programmable rails,” inProceedings of the 2023 Conference on Empir-
ical Methods in Natural Language Processing: System Demonstrations.
Singapore: Association for Computational Linguistics, 2023, pp. 431–
445.
[9] M. Abadi, M. Budiu, ´U. Erlingsson, and J. Ligatti, “Control-flow
integrity,” inProceedings of the 12th ACM Conference on Computer
and Communications Security. ACM, 2005, pp. 340–353.
[10] S. Gulyamov, “Prompt injection attacks in large language models,”
Information, 2026.
[11] S. Chen, J. Piet, C. Sitawarin, and D. Wagner, “Struq: Defending against
prompt injection with structured queries,”CoRR, vol. abs/2402.06363,
2024.
[12] Y . Liuet al., “Prompt injection attack against llm-integrated applica-
tions,”arXiv preprint arXiv:2306.05499, 2023.
[13] X. Dai, H. Hu, Y . Hua, J. Li, Y . Chen, R. Jin, N. Hu, and G. Qi,
“After retrieval, before generation: Enhancing the trustworthiness of
large language models in retrieval-augmented generation,”CoRR, vol.
abs/2505.17118, 2025.
[14] X. Suo, “Signed-prompt: A new approach to prevent prompt in-
jection attacks against llm-integrated applications,”arXiv preprint
arXiv:2401.07612, 2024.
[15] M. Alamsabi, “Embedding-based detection of indirect prompt injection
attacks,”Algorithms, 2026.
[16] Anonymous, “Hidden in plain text: A benchmark for social-web indirect
prompt injection in rag,”arXiv preprint arXiv:2601.10923, 2026.
[17] X. Liuet al., “Automatic and universal prompt injection attacks against
large language models,”arXiv preprint arXiv:2403.04957, 2024.