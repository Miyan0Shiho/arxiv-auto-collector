# Pramana: A Protocol-Layer Treatment of Claim Verification in Autonomous Agent Networks

**Authors**: Ravi Kiran Kadaboina

**Published**: 2026-05-19 17:00:33

**PDF URL**: [https://arxiv.org/pdf/2605.20312v1](https://arxiv.org/pdf/2605.20312v1)

## Abstract
Autonomous agents deployed in regulated domains must produce a verification artifact per consequential output: a record an auditor can re-execute offline, capturing what was claimed, against what source, by whom, when, and how. Production verification today splits into two unstandardized halves. Probabilistic verdict patterns (self-consistency voting, reviewer LLM ensembles) produce judgments, not artifacts. Artifact-producing patterns (RAG, tool-augmented traces, generator-verifier loops) produce vendor-specific records no external auditor can reconstruct without bespoke integration.
  Pramana defines the missing wire format. Every consequential agent output is wrapped in a typed ClaimAttestation with one of four variants (measurement, inference, analogy, citation), each paired with a verify() operation against the recorded source. verify() is deterministic for MeasurementClaim and CitationClaim. For InferenceClaim and AnalogyClaim, determinism is conditional on the oracle (audit-replayable when LLM-backed). The four-way typology derives from classical Indian epistemology (pramana, valid means of knowledge).
  The lifecycle is specified in TLA+ and exhaustively verified under TLC across three symmetry-reduced models: 38,563 distinct reachable states, zero invariant violations. The Python reference implementation passes 84 tests. An A2A and MCP wire-extension manifest layers three deployment-grade invariants: reachability, SLA bound, and offline re-verifiability.
  An exploratory pilot (n=100, 2,275 reviewer calls) probes LLM-as-judge in code generation. The strongest observation is a 40-percentage-point raw FPR delta across corpora, consistent with reference-solution quality contributing significantly. The pilot does not validate Pramana on its own; the structural argument and formal verification do that.

## Full Text


<!-- PDF content starts -->

Pramāṇa: A Protocol-Layer T reatment of Claim V erification in
Autonomous Agent Networks
Ravi Kiran Kadaboina
Independent Researcher
M.S., Computer Engineering, University of New Mexico, 2011
Abstract
Autonomous agents deployed in regulated domains must produce a verification artifact per conse-
quential output. The artifact is a record an auditor can re-execute offline, capturing what was
claimed, against what source, by whom, when, and how. Production verification today splits
into two unstandardized halves. Probabilistic verdict patterns (self-consistency voting, confidence-
scored outputs, reviewer LLM (large language model) ensembles) produce judgments about model
outputs, not artifacts. Artifact-producing patterns (retrieval-augmented generation with citations,
tool-augmented traces, generator-verifier loops, and recent multi-agent research systems such as
Google’s AI co-scientist [42]) produce vendor-specific records that an external auditor cannot re-
construct without bespoke integration. Pramana defines the missing wire format: a typed claim
attestation with a deterministic verify() contract, layered as an A2A and MCP extension.
Pramana wraps every consequential agent output in a typed ClaimAttestation with one of four
variants (measurement, inference, analogy , citation), each paired with a verify() operation against
the recorded source. F or MeasurementClaim and CitationClaim ,verify() is a deterministic
function of (claim, source) and no probabilistic judge participates in the verification step. F or
InferenceClaim and AnalogyClaim the determinism is conditional on the oracle: deterministic
when the oracle is a theorem prover, structural matcher, or fixed-metric similarity function; audit-
replayable when the oracle is an LLM. The four-way typology is derived from classical Indian
epistemology (pramana, “the means of valid knowledge”) and partitions agent claims by their
ground.
The Pramana lifecycle is specified in TLA+ (T emporal Logic of Actions) and exhaustively veri-
fied under TLC (TLA+’s model checker) across three symmetry-reduced models, totaling 38,563
distinct reachable states with zero invariant violations. The Python reference implementation
passes 84 unit and property-based tests. A wire-extension manifest publishes the format for A2A
(Agent2Agent) [1] and MCP (Model Context Protocol) [2] agents, with three deployment-grade
invariants (reachability , service-level-agreement (SLA) bound, and offline re-verifiability) layered
on top.
W e report an exploratory pilot (n=100 problems, approximately 1.3M tokens across 2,275 reviewer
calls) that probes the LLM-as-judge alternative in code generation. The strongest observation is a
roughly 40-percentage-point raw FPR (false-positive rate) delta between corpora under the same
ensemble, consistent with reference-solution quality contributing significantly; differential prompt-
overfit across corpora during the selection phase is an alternative driver that the single-rater design
cannot rule out. The pilot does not validate Pramana on its own; the structural argument in
Sections 1 through 4 and the formal verification in Section 6 do that.
1arXiv:2605.20312v1  [cs.CR]  19 May 2026

1. Introduction
A high-stakes agent deployment needs more than an output that is probably correct. It needs
a record of what was claimed, against what source, by whom, when, in a form an external au-
ditor can re-execute offline. Production verification today splits into two halves. Probabilistic
verdict patterns (self-consistency voting, confidence scores, reviewer LLM ensembles) produce judg-
ments about model outputs, not artifacts; aggregating judgments does not change their category .
Artifact-producing patterns (retrieval-augmented generation with citations, tool-augmented traces,
generator-verifier loops as in F unSearch [30] and AlphaEvolve [31], and recent multi-agent research
systems such as Google’s AI co-scientist [42]) produce vendor-specific records but no shared wire
format. An auditor inspecting a multi-vendor agent network cannot reconstruct verification across
patterns without bespoke per-vendor integration.
The verification artifact a regulator wants has a structurally different shape from a verdict.
An artifact records the claim, the source it was checked against, the procedure used to
check it, the verifier identity , the timestamp, and the outcome. Pramana defines the
typed wire format that standardizes this artifact. It is published as an A2A and MCP
extension manifest at https://ravikiran438.github.io/pramana-attestation/v1 via the
capabilities.extensions mechanism those host protocols define, alongside sibling extensions
(Anumati, Y athartha, Phala, Sauvidya, Pratyahara) addressing other wire-level concerns.
The rest of this paper is organized as follows. Section 1.1 describes what the protocol layer is
currently missing. Section 1.2 surveys the regulatory pressure motivating this work. Section 1.3
lists the contributions. Section 2 defines the Pramana primitive. Section 3 describes composition
with the existing stack. Section 4 maps the regulatory landscape onto specific primitives. Section
5 reports the supplementary empirical pilot. Section 6 reports the formal correctness properties.
Section 7 surveys related work. Section 8 lists limitations and future work.
1.1 What the protocol layer is missing
A2A [1] and MCP [2] standardize agent communication at the syntactic level (messages, tasks,
agent cards, tool calls, tool results), but neither defines a typed vocabulary for the epistemic
ground of an agent output. A tool result in MCP is a content blob; an A2A agent message is a
text payload. Neither carries a typed attestation of source URI, measurement record, or inference
chain. Pramana adds the missing vocabulary , with each variant naming the epistemic ground that
warrants the claim plus the metadata required to verify it independently .
ClaimAttestation is parameterized by claim type. F or MeasurementClaim and CitationClaim
the verification step is structurally deterministic: verify() is a function of (claim, source) and
no probabilistic judge participates. F or InferenceClaim and AnalogyClaim the verification step’s
determinism is conditional on the oracle plugged into the verify() slot, deterministic when the ora-
cle is a theorem prover, structural matcher, or fixed-metric similarity function, and audit-replayable
rather than bit-deterministic when the oracle is an LLM. Section 2.5 specifies the partition.
1.2 Regulatory pressure
Regulators across jurisdictions converge on artifact-requiring compliance bars. CFPB Circular
2023-03, F ederal Reserve SR 11-7, NYDFS Insurance Circular Letter No. 7, HHS OCR’s Section
1557 Final Rule, the EU AI Act, NYC Local Law 144, and the Colorado and California state AI
laws all require per-decision documentary records that an external auditor can re-execute. The
2

regulatory frameworks specify the audit substrate (a re-verifiable record per consequential output)
rather than a particular verification pattern. Section 4 maps each requirement to the Pramana
primitive that produces the substrate.
1.3 Contributions
1. A typed wire-format primitive: a ClaimAttestation with four variants (measurement, in-
ference, analogy , citation) derived from the classical pramana taxonomy of valid knowledge,
each paired with a verify() operation against the recorded source. The four variants parti-
tion agent claims by epistemic ground; verification determinism by claim type is specified in
Section 2.5.
2. A TLA+ specification of the lifecycle, exhaustively verified under TLC across three symmetry-
reduced models (38,563 distinct reachable states, zero invariant violations), plus a Python
reference implementation that passes 84 unit and property-based tests.
3. An A2A and MCP wire-extension manifest ( claim-attestation/v1 ) with three deployment-
grade invariants: reachability , SLA-bound, and offline re-verifiability .
4. An exploratory pilot (n=100, approximately 1.3M tokens) that probes the LLM-as-judge
alternative in code generation and characterizes the measurement regress in reviewer-ensemble
evaluation.
2. The Pramana Primitive
A ClaimAttestation is a structured assertion an agent makes about something it has observed,
inferred, compared, or cited. Each attestation declares the epistemic ground that warrants the
claim plus enough metadata for any party holding the source to verify the claim independently .
2.1 The four claim types
The taxonomy follows the four ways a claim can be grounded in observable reality .
Type Ground V erification operation
MeasurementClaim Direct observation or structured
recordSource-record fetch and
field-match
InferenceClaim Logical inference from prior
claimsInference-chain replay
AnalogyClaim Similarity to a known reference
caseSimilarity recompute against
reference
CitationClaim Attribution to an authoritative
sourceSource fetch plus faithful-citation
check
Each subtype is a Pydantic v2 model in a discriminated union over a claim_type field. Pydantic v2
is the data-modeling library used by both the A2A [1] and MCP [2] reference SDKs, so a Pramana
extension deserializes through the same code path the host agent already runs for native A2A or
MCP messages. All four share a common envelope ( claim_id ,claim_type ,claim ,attester_id ,
attested_at ) and contribute type-specific fields: a MeasurementClaim adds the measured value,
unit, method, source URI, and uncertainty; a CitationClaim adds the source URI, the verbatim
excerpt, and a retrieval timestamp. F ull per-type schemas are in the companion repository .
3

2.2 V erification semantics
Each variant’s verify() is a function from (claim, source) to one of three outcomes: VERIFIED ,
REJECTED , or UNVERIFIABLE . The function is deterministic when its injected dependency is deter-
ministic, such that re-running with the same inputs produces a bit-identical VerificationOutcome
(up to timestamp and verifier id). Source fetchers and hash matchers are naturally deterministic;
LLM-backed inference oracles and similarity computers are not, and Section 2.5 spells out the
audit-replayability fallback. The function is also injectable, with source-fetch, measurement-fetch,
similarity-computation, and inference- oracle dependencies as constructor arguments. This permits
production-grade caching, SSRF (server-side request forgery) protection, allow-listing, and offline
auditing without changing the verification logic.
The AnalogyClaim case has one design subtlety worth surfacing. When similarity_score
is recorded, verify() checks the recomputed score against it within tolerance. When
similarity_score is absent, verify() returns VERIFIED as long as the similarity computer
produced any score, such that the deterministic check reduces to “the similarity computation
succeeded. ” Emitters that want stricter verification should record a similarity_score explicitly .
Regulated deployments SHOULD require similarity_score for any AnalogyClaim and SHOULD
reject VERIFIED outcomes lacking it. A VERIFIED outcome without similarity_score is
documentary (the similarity computation succeeded) rather than substantive (the similarity
exceeded a threshold), and an auditor reviewing such an outcome cannot distinguish a successful
match from a near-miss from a wire-format inspection alone. The v1 wire format admits both
modes intentionally to support deployments without natural thresholds; a future revision could
introduce a distinct VERIFIED_UNTHRESHOLDED status to make the distinction visible at the audit
layer.
2.3 Lifecycle invariants
Five named safety invariants hold across all reachable states.
1. P-1 (Single Emission) : each claim has at most one emission audit entry .
2. P-2 (V erification Determinism) : each claim’s verification state is single-valued and
reaches a terminal state at most once.
3. P-3 (Audit Completeness) : every terminal verification, suppression, and disclosure is
recorded in the audit trail.
4. P-4 (Disclosure Coupling) : a claim shown to a principal has an emitted , a verified ,
and a disclosed entry in the audit trail.
5. SuppressionDisclosureDisjoint : no claim is both suppressed and shown to a principal.
F ull TLC verification is in Section 6.
2.4 What Pramana verifies and what it does not
Pramana is not a hallucination detector, a confidence score, or a replacement for unit tests or
human review. It names the verification record that a regulator can demand and the function that
produces it. Whether the underlying agent is well-behaved is out of scope. Pramana ensures that
if the agent makes a claim, the claim is accompanied by a record any auditor can re-execute.
The artifact-versus-judgment distinction is load-bearing for the regulated-domain use case.
Pramana does not propose to deterministically verify the underlying judgment in subjective-
ground-truth domains. Whether a diagnosis is correct, whether a loan applicant is creditworthy ,
4

whether an underwriting decision is fair: these are not deterministic questions and Pramana does
not pretend otherwise. What Pramana deterministically verifies is the artifact of the decision:
whether the MeasurementClaim ’s reported lab value matches the structured record, whether the
InferenceClaim ’s named decision rule was applied as documented, whether the CitationClaim ’s
reference to a guideline resolves to the cited text, whether the AnalogyClaim ’s stated similarity
to a prior case recomputes within tolerance. The regulator does not need to verify the diagnosis;
they need to verify the documentary record of how the diagnosis was constructed.
2.5 Threat model
Pramana assumes the source registry is honest within the deployment’s trust boundary , the crypto-
graphic primitives backing source_digest and artifact_signature are not broken, and verifiers
run uncompromised code. Under these assumptions the protocol defends against silent claim mod-
ification (CA-3 makes tampering observable), against unilateral retraction (P-1 admits at most
one emission per claim), and against unbounded verification latency (CA-2 forces a terminal state
within the declared SLA window). Pramana does not defend against an adversary who fabri-
cates source content at a URI it controls; that attack is observable under Pramana but not
prevented. In deployments where source authority matters, Pramana composes with separate
source-attestation primitives (cryptographic signatures, content-addressed storage, federated reg-
istries) via the source_uri integration point. Concretely , the source_digest field (Section 3.2)
carries a cryptographic hash of the source bytes at retrieval time; CitationClaim verifiers MUST
reject claims whose freshly-fetched content does not hash to the recorded digest. This binding is
mandatory for non- unverifiable outcomes (CA-3) and is the protocol-level integration surface
for external source- attestation infrastructure. The protocol does not specify particular attestation
mechanisms, but constrains their composition through the digest invariant.
The determinism guarantee holds unconditionally for MeasurementClaim and CitationClaim ,
where the injected dependency is a source fetcher or hash matcher. F or InferenceClaim and
AnalogyClaim the guarantee holds when the oracle is itself deterministic (theorem prover, struc-
tural matcher, fixed-metric similarity function). When the oracle is a commercial LLM API call,
the guarantee fails. Modern hosted LLMs do not produce bit-identical outputs even at tempera-
ture 0 and a fixed seed, due to floating-point non-determinism in batched GPU inference, sparse
mixture-of-experts routing variability , and silent backend updates that are not version-pinned by
the provider. F or these deployments Pramana’s contribution narrows to audit-replayability: the
audit trail records the oracle’s inputs, its output, the model identifier, and the timestamp, so an au-
ditor can re-execute and observe both the original verdict and any re-execution verdict that differs.
Strict offline re-verifiability (CA-3) requires constraining oracles to deterministic procedures.
Audit-replayability is categorically different from raw logging. The trail captures structured inputs
(premises as a typed list, conclusion as a separate field, inference_method as a named procedure,
source_uri as a resolvable reference), not the LLM prompt blob. An auditor querying the trail can
recover which decision rule was claimed to apply , what premises were cited, and which sources were
consulted, independent of whether the oracle’s verdict on re-execution is bit-identical. Re-execution
with the same oracle and inputs surfaces the direction and magnitude of any divergence, which is
itself queryable evidence. Deployments that need full verdict reproducibility plug in a deterministic
oracle in the same dependency slot, without changing the wire format or the audit-trail shape.
Composition with the existing agent-protocol stack, the wire-level extensions to A2A and MCP ,
and the regulatory landscape this primitive is designed to satisfy are addressed in the next three
5

sections.
3. Composition with the existing stack
3.1 Composition with sibling protocols
Pramana is one of six wire-level extensions in the agent-protocol-stack programme, each published
as an A2A and MCP extension via the capabilities.extensions mechanism the host proto-
cols define. Anumati / ACAP [10] specifies consent and adherence, where ACAP records the
consent for an action and Pramana records the verification of each claim that fed the action.
Y athartha [11] specifies the capability surface, such that a capability self-attestation can be a
CitationClaim (weak grounding, with the agent itself as source) and a third-party probed ca-
pability is a MeasurementClaim (strong grounding). Phala [12] (welfare feedback), Pratyahara /
NER VE [13] (sensory integrity), and Sauvidya / P ACE [14] (accessibility envelope) compose adja-
cently . Each protocol follows the same convention and Pramana takes the role of artifact verification
in the stack.
3.2 Wire-level extension to A2A
An agent declares Pramana support by including an entry in its A2A AgentCard
capabilities.extensions array whose uri matches the Pramana extension URI declared in §1.
The entry deserializes to a PramanaServiceRef declaring the verify endpoint, audit endpoint,
supported ClaimType values, source-fetcher schemes, signature algorithm, and SLA window.
The receiving agent uses this declaration to decide what claim types it can verify against and
to route incoming verification requests. The claim-attestation/v1 wire extension layers on
top, adding a per-claim envelope with verify_endpoint_hint ,verification_artifact_id ,
artifact_signature ,signature_alg ,source_digest , and sla_window_ms . The extension
enforces three deployment-grade invariants beyond the core lifecycle invariants of Section 2.3.
CA-1 (reachability) requires every emitted attestation to carry a verify_endpoint_hint so any
receiver can re-verify . CA-2 (SLA bound) requires verification to reach a terminal state within the
declared window or be auto-marked unverifiable . CA-3 (offline re-verifiability) requires every
non- unverifiable outcome to carry a source_digest and an artifact_signature so an offline
auditor can re-verify .
3.3 Wire-level extension to MCP
F or MCP , ClaimAttestation records attach as tool-result metadata under a pramana key in the
structuredContent field. The content payload is unchanged, and the attestation rides alongside
as structured metadata that an MCP-aware client can verify before surfacing the content to the
principal. T ools that do not understand Pramana ignore the annotation, and tools that do can
chain attestations across tool calls without re-parsing the content payload. The MCP wire encoding
follows the same discriminated-union pattern as the A2A wire format. F ull schemas are in the
companion repository .
The regulatory frameworks that require this artifact are surveyed next, with the primitive-to-
requirement mapping in Section 4.
6

4. Regulatory landscape
Regulators across jurisdictions converge on a common structure: per consequential decision, an
auditor must be able to recover what was claimed, against what source, by whom, when, and
how. None of the frameworks below specify a particular mechanism. They specify the verification
record. Pramana provides one mechanism that produces it. T able A summarizes the alignment;
the framework details follow.
F ramework Required record Pramana primitive
CFPB Circular 2023-03 / Reg B
[3]Specific source-linked reason for
adverse credit actionVerificationOutcome.evidence
per claim
SR 11-7 / OCC Bulletin 2011-12
[4]Model documentation suﬀicient
for examiner re-executionClaimAttestation plus
VerificationOutcome per model
output
NYDFS Circular Letter 7 (2024)
[5]Demonstrable rational
relationship for AI underwriting
variablesInferenceClaim.verify()
against named premises
Colorado DOI Reg 10-1-1 [6] Governance framework and
per-decision audit log for insurer
AIClaimAttestation audit chain
HIP AA §164.312(b) Examination of activity in ePHI
systemsVerificationOutcome per claim,
append-only
Section 1557 / 45 C.F.R. §92.210
[7]Evidence of “reasonable efforts”
to mitigate AI-tool discriminationRe-executable
VerificationOutcome chain
EU AI Act Articles 14, 50 [8] Effective human oversight and
decision disclosureInspectable ClaimAttestation
per decision component
GDPR Recital 71 [9] Right to explanation Per-claim source URI plus
excerpt plus re-verification
procedure
T able A. Regulatory record requirements and the Pramana primitive that produces each.
The mapping in T able A assumes claim types are paired with deterministic oracles; for
InferenceClaim and AnalogyClaim with LLM-backed oracles, Pramana provides the
audit-replayable documentary artifact rather than deterministic re-verification (see §2.5).
Pramana provides a documentary artifact each framework requires but does not by itself estab-
lish substantive compliance. NYDFS’s “rational, statistically significant” standard requires that
the model’s underwriting logic in fact be rational and statistically significant; §1557’s “reasonable
efforts” standard requires that the deployer’s efforts in fact be reasonable. Pramana makes these
properties auditable by recording the inputs, sources, and outputs of each decision; whether the
recorded properties hold under audit is a substantive evaluation outside the protocol’s scope and
inside the regulator’s. The protocol is necessary documentary infrastructure; substantive compli-
ance also requires that the model and the deployer’s process hold up under that audit. Whether
audit-replayability (as distinct from deterministic re-verification) satisfies the substantive eviden-
tiary standards of any cited regulation is an open legal question without settled precedent; this
paper maps structural alignment, not substantive suﬀiciency , and notes the open question rather
than claiming to resolve it.
7

4.1 Consumer lending (CFPB)
CFPB Circular 2023-03 [3] addresses creditors using complex algorithms including AI for credit de-
cisions. The Equal Credit Opportunity Act and Regulation B require lenders to provide specific rea-
sons for adverse action, such that “the model rejected you” does not satisfy the requirement and the
CFPB sample-form checkboxes cannot substitute for the actual reason. A Pramana-instrumented
lender produces, per credit decision, a ClaimAttestation chain in which each adverse-action factor
is grounded in a specific source claim with a re-executable verify() against the underlying data.
With this artifact attached to the notice, the consumer and a CFPB examiner can recover the basis
for the decision by re-running the verification chain.
4.2 Banking model risk (SR 11-7, OCC 2011-12)
F ederal Reserve SR 11-7 and OCC Bulletin 2011-12 [4] is the supervisory guidance on model risk
management for banks, predating large language models by a decade and applying to AI-driven
models by default. SR 11-7 requires documentation “suﬀiciently detailed so that parties unfamiliar
with a model can understand how the model operates, its limitations, and its key assumptions. ”
The SR 11-7 examination surface is what a ClaimAttestation chain provides, since the chain
records, per model output, what was claimed, against what source, by which model, when, and how
verification was performed. A bank using a Pramana-instrumented agent for credit underwriting,
fraud scoring, or trading-strategy generation produces the SR 11-7 audit trail as a side effect of
normal operation.
4.3 Insurance underwriting (NYDFS, Colorado DOI)
NYDFS Insurance Circular Letter No. 7 [5] requires insurers to demonstrate that AI-driven un-
derwriting variables have “a clear, empirical, statistically significant, rational, and not unfairly
discriminatory relationship” with the insured risk. The “demonstrate” requirement maps directly
to the InferenceClaim primitive, whose verify() operation replays the inference against the
named premises and returns a deterministic outcome rather than a model verdict. An NYDFS
examiner reviewing an algorithmic underwriting decision can re-execute the inference chain and
verify each step. Colorado Division of Insurance Regulation 10-1-1 [6] requires governance and
risk-management framework documentation for insurers using external consumer data, algorithms,
and predictive models. The ClaimAttestation audit chain provides the documentary artifact this
framework requires as the artifact of normal operation.
4.4 Healthcare (HIP AA, Section 1557, FDA SaMD)
HIP AA §164.312(b) audit controls require “examination of activity” in ePHI systems, a rule general
enough that it applies to algorithmic decision aids by default. HHS OCR’s Section 1557 Final Rule
(45 C.F.R. §92.210) [7] regulates patient care decision support tools and requires covered entities
to make “reasonable efforts” to identify and mitigate discrimination from such tools. The FDA
Software-as-a-Medical-Device framework requires “verifiable performance” of AI-driven clinical soft-
ware. Per-claim VerificationOutcome records provide the documentary artifact required by the
§164.312(b) examination requirement, the §92.210 reasonable-efforts standard, and the SaMD ver-
ifiability bar.
8

4.5 Employment screening (NYC Local Law 144, EEOC, ADA/ADEA/Title VII)
AI-driven hiring tools are the most-deployed application of automated decision-making in US labor
markets. F uller et al. [35] document that applicant-tracking systems screen out roughly 27 million
qualified US workers – veterans, people with disabilities, caregivers returning after employment
gaps, formerly incarcerated, and immigrants with non-traditional credentials – by treating status
proxies as disqualifiers. Per-candidate evidence of what the system claimed and against what source
is what an affected applicant or an auditor needs to contest a decision.
NYC Local Law 144 [36] requires any employer using an Automated Employment Decision T ool to
commission an independent bias audit annually , publish the audit summary , and notify candidates
ten business days in advance. The implementing rule defines the audit as selection-rate disparity
testing by protected category . Conducting that audit on a black-box system requires reconstructing
per-candidate scoring from logs after the fact; conducting it on a Pramana-instrumented system
means the per-candidate ClaimAttestation chain is the audit substrate by construction.
EEOC v. iT utorGroup [37] established that pre-existing federal employment discrimination statutes
(ADEA, Title VII) reach AI hiring tools without new legislation. Mobley v. W orkday [38] extended
liability to the AI vendor itself as an “agent” of its employer customers, conditionally certifying a
nationwide ADEA collective action. In discovery , plaintiffs may demand the vendor’s claim-level
reasoning trace; a vendor without one is in a weaker position than one holding the documentary
artifact this protocol standardizes. Whether that artifact substantively prevents discrimination is
for the regulator and the court to evaluate, per the necessary-versus-suﬀicient distinction at the
top of this section.
4.6 European Union and US state laws
EU AI Act Articles 14 (human oversight) and 50 (transparency) [8] take effect August 2, 2026,
for high-risk systems. As of writing, the EU AI Oﬀice has not published technical guidance on
consent or verification mechanisms for autonomous agent transactions, and the compliance deadline
binds before guidance is final. Pramana’s per-claim ClaimAttestation is the artifact an oversight
reviewer can inspect (Article 14), and the discoverable extension URI plus per-claim verification
log provides the disclosure surface Article 50 requires. GDPR Recital 71 [9] establishes the right
to explanation; an artifactless system cannot satisfy it. Three US states have enacted broader
automated-decision laws with audit-trail obligations: the Colorado AI Act, California’s ADMT
(Automated Decision-Making T echnology) regulations, and T exas TRAIGA (T exas Responsible AI
Governance Act). Each statute’s obligation maps directly to a ClaimAttestation chain. NYC
Local Law 144’s employment-specific mandate is covered in §4.5.
4.7 W orked example: NYDFS-relevant underwriting decision
Consider an autonomous underwriting agent processing an application against NYDFS Insurance
Circular Letter No. 7 [5] and its “rational, statistically significant relationship” requirement. The
agent emits three Pramana-attested claims that together constitute the decision artifact.
Source claim. A MeasurementClaim records the applicant’s structured features (debt-to-income
ratio, FICO score, claims history over three years) with measurement_source pointing at the appli-
cant’s structured-record URI. verify() fetches the source record and confirms field-match against
the declared measured_value . The oracle is the structured-record store, which is deterministic.
Decision claim. An InferenceClaim records the underwriting verdict ( claim = "applicant
9

qualifies for tier 2 pricing" ), the premises that warrant it ( premises = ["DTI < 0.40",
"FICO >= 700", "no claims in 3 years"] ), and the named rule ( inference_method =
"tier_2_eligibility_rule_v1.2" ). verify(oracle) replays the rule against the premises.
When the oracle is a deterministic rule engine, the outcome is bit-identically reproducible.
When the oracle is an LLM, the verify() inputs and outputs are audit-replayable per the scope
clarification at the top of this section.
Policy claim. A CitationClaim records the underwriting policy reference ( source_uri =
"policy://underwriting/v1.2#section-3" , source_excerpt = "Tier 2 pricing applies
when DTI < 0.40 AND FICO >= 700..." ). verify() re-fetches the policy and checks excerpt
presence; source_digest carries the SHA-256 of the policy text at retrieval time so post-decision
tampering is detectable (Section 2.5).
The claim-attestation/v1 wire envelope wraps each attestation with verify_endpoint_hint
(CA-1), sla_window_ms (CA-2), and source_digest plus artifact_signature (CA-3). An
NYDFS examiner reviewing this decision queries the audit endpoint, retrieves the three
attestations and their VerificationOutcome records, and re-executes verify() against the
recorded sources. The audit trail records each step in append-only form.
This is the documentary record NYDFS Circular Letter 7 requires for any algorithmic underwriting
decision. Whether tier_2_eligibility_rule_v1.2 is in fact “rational and statistically signifi-
cant” is the substantive evaluation the regulator performs, not what Pramana establishes (per the
necessary-versus-suﬀicient distinction at the top of this section).
The three attestations and their VerificationOutcome records, emitted end-to-end by the reference
implementation using the field names and verify() semantics described above, are committed at
samples/nydfs_underwriting_walkthrough.json in the companion repository; the script that
produces them is at samples/build_nydfs_walkthrough.py . All three claims verify against in-
memory mocks that stand in for the structured-record store, the named rule engine, and the policy
document store; in production those slot in as the same dependency-injection points the unit tests
exercise.
Section 5 reports an exploratory empirical pilot that characterizes the alternative pattern Pramana
is designed to replace.
5. Supplementary empirical pilot
The structural argument in Sections 1 through 4 motivates Pramana as the wire format for re-
verifiable claim artifacts. This section reports an exploratory pilot characterizing how well the
dominant alternative (an LLM scoring another LLM’s output) performs on code generation, the
most-studied LLM-as-judge surface. The pilot uses n=100 problems totaling approximately 1.3M
tokens across 2,275 reviewer calls (roughly 918K input plus 373K output tokens, computed from the
JSONL logs in the companion repository). It is not a controlled comparison: the three ensemble
configurations vary multiple factors at once (ensemble size, capability tier, family composition,
prompt-tuning), and the prompt-selection set overlaps deterministically with the main-run problem
set. Findings are descriptive observations, not mechanism claims. The pilot does not validate
Pramana on its own; the structural argument in Sections 1 through 4 and the formal verification
in Section 6 do that.
10

5.1 Design
MBPP (Mostly Basic Python Problems) [15] and HumanEval [16] canonical solutions are mu-
tated by AST-level operators (operator-swap, off-by-one, edge-case-drop, method-confusion, type-
confusion) to produce quiet buggy code that is parseable, runnable, and fails the supplied test suite.
Three reviewer-ensemble configurations evaluate each (clean, buggy) pair under three aggregation
rules (unanimous, majority , any): same-model (three independent Haiku 4.5 calls at temperature
0), same-family (Haiku 4.5 + Sonnet 4.6 + Opus 4.7, all Anthropic), and cross-family (Haiku
4.5 + GPT-4o-mini + Llama-3.3-70B). Prompt-variant selection methodology is in the companion
repository . The same-model configuration doubles as an implicit single-model zero-shot baseline:
at temperature 0 the three Haiku calls produce identical verdicts, so its detection and FPR are also
the single-Haiku numbers. The ensemble configurations are evaluated against this baseline.
5.2 Strongest observation: the dataset-curation effect
Under the same ensemble configuration (same-family majority), the same prompt, and the same
model versions, raw FPR on HumanEval differs from MBPP by roughly 40 percentage points (T able
1, Figure 1). The delta is consistent with reference-solution quality contributing significantly:
MBPP canonical solutions include latent defects that reviewers correctly flag but that the test
suite does not exercise. Differential prompt-overfit across the two corpora during the selection
phase is an alternative candidate driver that this design cannot rule out, and the single-rater
adjudication cannot establish that any one factor dominates. Isolating drivers would require held-
out replication on disjoint problems and multi-rater adjudication. Reported reviewer-ensemble
FPR depends nontrivially on the reference-solution distribution, so single-corpus FPR numbers
are misleading without disclosure of the test-suite quality . Per-case adjudication of clean-flagged-
buggy cases is in the companion repository; the adjudication shows that what looks like a high
false-positive rate on MBPP is partly reviewers correctly catching latent defects the tests miss.
T able 1. Per-corpus detection and FPR, majority rule, with 95% bootstrap CIs (10,000 resamples;
n=40 per HumanEval cell, n=60 per MBPP cell, per buggy/clean label). Numbers are subject to
the optimistic-bias caveat above.
Ensemble Corpus Detection FPR
same-model HumanEval 97.5% [92.5, 100.0] 27.5% [15.0, 42.5]
same-model MBPP 98.3% [95.0, 100.0] 66.7% [55.0, 78.3]
same-family HumanEval 100.0% [100.0, 100.0] 5.0% [0.0, 12.5]
same-family MBPP 96.7% [91.7, 100.0] 41.7% [30.0, 53.3]
cross-family HumanEval 97.5% [92.5, 100.0] 27.5% [15.0, 42.5]
cross-family MBPP 98.3% [95.0, 100.0] 60.0% [48.3, 71.7]
11

Figure 1: Figure 1: Detection and FPR by ensemble structure, faceted by source corpus, majority
rule. The same ensemble running the same prompt against the same model versions produces
sharply different FPR on MBPP vs HumanEval; reference-solution quality and per-corpus prompt-
overfit are both candidate drivers that this design cannot disentangle.
5.3 Configuration-shift observations
Detection and FPR per ensemble cell are in T able 2 (bootstrap 95% CIs over 10,000 resamples).
T able 2’s raw FPR estimates are subject to optimistic bias from prompt-selection overlap with
the evaluation set (Section 8). The paired McNemar comparisons in T able 3 are reported as
descriptive observations of configuration differences. The prompt-pilot optimized adjusted FPR
for a single Haiku reviewer, so the leakage is concentrated in Haiku-weighted configurations rather
than symmetric across them; the paired contrasts are therefore not strictly bias-immune, and the p-
values measure paired disagreement counts under asymmetric leakage rather than treatment-effect
significance.
T able 2. Detection and FPR per ensemble cell. n_buggy = 100, n_clean = 100.
Ensemble Rule Detection FPR
same-model all rules 98.0% [95.0, 100.0] 51.0% [41.0, 61.0]
same-family unanimous 88.0% [81.0, 94.0] 10.0% [5.0, 16.0]
same-family majority 98.0% [95.0, 100.0] 27.0% [19.0, 36.0]
same-family any 99.0% [97.0, 100.0] 54.0% [44.0, 64.0]
cross-family unanimous 83.0% [75.0, 90.0] 25.0% [17.0, 34.0]
cross-family majority 98.0% [95.0, 100.0] 47.0% [37.0, 57.0]
cross-family any 100.0% [100.0, 100.0] 79.0% [71.0, 87.0]
12

Figure 2: Figure 2: Detection vs FPR by ensemble structure and aggregation rule, with 95%
bootstrap CIs.
Under the majority rule, McNemar’s exact paired comparison identifies two configuration-shift
effects at p<0.0001 (T able 3, Figure 3). The 3-Haiku-at-temperature-0 configuration produces
three identical verdicts (collapsing to a single effective vote), and shifting to a 3-distinct-model
same-family configuration flips 24 clean cases from flagged-buggy to flagged-clean with zero shifts in
the reverse direction. Shifting from the same-family triple to the cross-provider triple flips 22 cases
in the opposite direction. The data does not isolate the treatment effect: ensemble size, capability
tier, family composition, and prompt-tuning vary together. W e report the McNemar shifts as
descriptive observations rather than diversity or scaling claims. The p-values measure paired-
disagreement counts on configurations that vary multiple factors at once; readers should interpret
them as evidence that the configurations differ, not as evidence for any specific causal mechanism.
The same-model configuration is documented for what it tests (production-reproducibility-mode
T=0 ensembling) and does not speak to self-consistency , which requires T>0 to sample diverse
reasoning paths.
T wo descriptive comparisons in the pilot hold most of their variables constant. (a) The corpus
split in T able 1 holds the ensemble constant (same-family majority) and varies the evaluation set,
isolating reference-solution-quality and other inter-corpus differences as candidate drivers. (b) The
implicit single-Haiku baseline (the same-model row) against the family-scaled ensemble {Haiku,
Sonnet, Opus} varies ensemble size and capability tier together but holds family and prompt con-
stant. Comparisons that vary family composition independently of capability tier are not isolated
by this design.
13

T able 3. McNemar’s paired comparisons, majority rule. Significant cells (p < 0.05) in bold. The
p-values reflect the direction of paired disagreement, not isolated treatment effects: each compar-
ison varies ensemble size, capability tier, family composition, and prompt-tuning simultaneously .
Readers should interpret the cells as evidence that the configurations differ in their labeling, not
as evidence for any specific causal mechanism.
Slice Comparison b c p (exact)
buggy (detection) all three pairwise <=1 <=1 1.000
clean (FPR) same-model vs
same-family24 0 <0.0001
clean (FPR) same-family vs
cross-family2 22 <0.0001
clean (FPR) same-model vs
cross-family5 1 0.219
Figure 3: Figure 3: McNemar discordant-cell counts on the clean slice for the two significant pair-
wise comparisons. Each comparison varies multiple factors at once (ensemble size, capability tier,
family composition); the shifts are reported as configuration differences rather than attributable to
a single treatment effect.
The raw program output from metrics.py against the run’s JSONL log reproduces the cells above:
=== Detection & FPR (point [95% bootstrap CI]) ===
same-family unanimous 88.0% [81.0, 94.0] 10.0% [ 5.0, 16.0]
same-family majority 98.0% [95.0, 100.0] 27.0% [19.0, 36.0]
cross-family majority 98.0% [95.0, 100.0] 47.0% [37.0, 57.0]
14

=== McNemar's paired comparisons (rule=majority, clean slice) ===
same-model vs same-family 24 0 22.04 0.0000
same-family vs cross-family 2 22 15.04 0.0000
5.4 Why the alternative fails structurally
The regress that §7.1 names structurally is observable in the pilot. External ground truth from test
suites is partial since unit tests miss bugs. Improving the ground truth either reduces to writing
better tests (the original problem the reviewer was deployed to solve) or requires a stronger judge
with the same problem one level up. Three classes of partial fix exist (importing external ground
truth, applying calibration regularization, removing the LLM from the judgment step entirely); the
first two push the problem one level up while still requiring the labelled data they were meant to
substitute for. Pramana takes the third route for MeasurementClaim and CitationClaim in full,
and partially for InferenceClaim and AnalogyClaim when the oracle is itself deterministic (see
Section 2.5). The structural escape is the contribution; the empirical pilot documents that the
alternative (probabilistic reviewer ensembles) faces the loop.
5.5 Adversarial robustness check
Five programmatic adversarial transformations were applied to 30 sampled buggy mutations; the
same-family majority detection per transformation is in T able 4 and Figure 4.
T able 4. Adversarial detection per transformation. Mean benign baseline is 98.0%.
T ransformation Detection Δ vs benign
misleading_docstring 96.7% −1.3 pp
defensive_assertion 93.3% −4.7 pp
rationalizing_comment 96.7% −1.3 pp
noisy_helpers 96.7% −1.3 pp
plausible_rename 96.7% −1.3 pp
15

Figure 4: Figure 4: Detection rate per adversarial transformation, same-family majority . The
benign baseline (98.0%) is shown as the dashed reference line. The largest drop is 4.7 pp (defen-
sive_assertion); the mean drop is 2.0 pp.
5.6 Scope and methodological constraints
The pilot uses code generation, which has objective binary ground truth via unit tests; the regu-
lated domains the protocol targets have ground truth that is contested, probabilistic, or unavailable
at decision time. The pilot cannot directly simulate those conditions. The adjudication of clean-
flagged-buggy cases is single-rater, a methodological ceiling on the adjusted-FPR numbers; the
mathematical analyses of specific failure modes (Vieta’s formulas for reciprocal roots, triangle-in-
semicircle area, range arithmetic, cyclic-encoding identity) are externally verifiable and invariant
to rater bias. The adversarial check at n=30 per transformation surfaces detection drops of 1.3 to
4.7 percentage points; this is exploratory , with effect sizes within sampling noise at this sample size.
A pilot in a regulated domain is the natural next study . Such a pilot would implement a Pramana-
instrumented agent against a documented decision-rule set (mock underwriting against NYDFS-
relevant variables, mock diagnostic against §92.210- relevant EHR (electronic health record) fields),
generate ClaimAttestation chains for representative decisions, and submit those artifacts to com-
pliance review against the specific regulatory framework’s bar. The current paper establishes the
protocol design (Section 2), formal lifecycle correctness (Section 6), the reference implementation,
and the structural argument that the artifact-shape regulators require is what Pramana produces.
Empirical regulatory-compliance validation is the next paper in this programme.
The lifecycle properties that make this primitive correct under concurrency , suppression, and dis-
closure are formalized next.
16

6. F ormal properties
6.1 Specification structure
The TLA+ specification at specification/Pramana.tla models four lifecycle actions ( EmitClaim ,
VerifyClaim ,SuppressClaim ,ShowToPrincipal ) and tracks emission, verification state, suppres-
sion, principal view, and an append-only audit trail. State variables capture per-claim type, source,
attester, and verification outcome. The audit trail is a sequence of records, each tagged with the
action that produced it. The five named safety invariants from Section 2.3 (P-1 through P-4, plus
SuppressionDisclosureDisjoint) are TLC INVARIANTS .
6.2 Models and results
Three model configurations exhaustively check the invariants.
Model Distinct states Result
Lifecycle (P-1 through P-3) 2,345 0 violations
Disclosure (P-1 through P-4, plus
disjointness)4,601 0 violations
Concurrency (multi-actor) 31,617 0 violations
The total across the three models is 38,563 distinct reachable states with zero invariant vi-
olations . The three-model split is the standard TLA+ small-model pattern, in which focused min-
imal models converge in seconds rather than one large model that does not. All three use SYMMETRY
ModelSymmetry over the agent, claim, source, and verifier sets, with a BoundedTrail constraint
capping Len(auditTrail) at 6. The verification is reproducible via specification/run_tlc.sh .
The raw TLC output captures the per-model state counts:
[Lifecycle] 2349 states generated, 2345 distinct states found, 0 states left on
queue.
[Disclosure] 7409 states generated, 4601 distinct states found, 0 states left on
queue.
[Concurrency] 47689 states generated, 31617 distinct states found, 0 states left
on queue.
A supplementary partial-exhaustive run (3 claims, 2 agents, 2 verifiers, 1 principal) was explored to
166M+ distinct states with zero violations before BFS (breadth-first search) exhausted compute.
6.3 Wire-extension invariants
The claim-attestation/v1 wire extension adds three deployment-grade invariants beyond the
core lifecycle invariants: CA-1 (reachability), CA-2 (SLA-bound), and CA-3 (offline re-verifiability).
CA-1 and CA-3 are separately TLC-verified at 21 distinct states with zero violations. CA-2 is a
runtime invariant enforced by validators in the reference implementation.
6.4 Reference implementation
The Python reference implementation passes 84 tests (59 core and 25 extension): 14 primitive-type
tests, 34 across the four verify() implementations covering every outcome branch, 6 TLC-config
17

tripwires preventing accidental drift between the spec and the runnable model, and 5 Hypothe-
sis property-based tests fuzzing wire-format invariants (JSON round-trip, discriminator dispatch,
claim-type invariance, never-raises, populated outcome.method ), plus 25 claim-attestation exten-
sion tests. The suite runs in under one second. Combined with TLC verification, this gives
end-to-end correctness coverage from the specification to the implementation.
Related work spans the empirical critique of LLM-as-judge, provenance-grounded alternatives (in-
cluding W3C V erifiable Credentials and PROV), agent-loop systems, and regulatory scholarship.
7. Related work
7.1 Empirical critique of LLM-as-judge and reviewer-ensemble patterns
Reviewer-ensemble evaluation inherits a measurement regress. Measuring the accuracy of an LLM
judge requires ground truth; the ground truth from external test suites is partial because unit
tests miss bugs; improving the ground truth either reduces to the original problem the reviewer
was deployed to solve or requires a stronger judge that faces the same problem one level up. The
regress is structural, not a tuning issue: any pipeline that uses an LLM to judge another LLM’s
output inherits it. The empirical work surveyed below documents specific failure modes that are
downstream of this structural problem.
A growing body of empirical work shows that LLM judges and reviewer ensembles do not per-
form the verification function they are deployed for. Kim et al. [17] observe that conditional
error correlation across 350+ models on two leaderboards is approximately 60% (agreement given
both wrong), with correlation higher within a family and persisting across providers when training
corpora overlap, and explicitly flag implications for the LLM-as-judge paradigm and multi-agent
systems. Li et al. [18] document preference leakage : when a generator and a judge share lineage,
the judge favors the generator’s outputs by margins larger than the well-known egocentric bias.
Zheng et al. [19], Panickssery et al. [20], and W ang et al. [21] critique LLM-as-judge for offline
benchmark scoring. Stureborg et al. [22] show that LLM evaluators are systematically inconsistent
across runs. Maloyan et al. [23] investigate prompt-injection vulnerabilities in judge architectures.
Nasr et al. [24] bypass twelve recent defense systems at greater than 90% success rate. Cemri et
al. [25] (MAST) taxonomize 14 multi-agent failure modes and identify a verification-deficit category
without proposing a wire-format remedy . None of these works names the production pattern of
generator-plus-reviewer-ensemble-as-safety-gate as an anti-pattern, characterizes it at the protocol
layer, and proposes a constructive wire-format alternative. The empirical work demonstrates the
problem; Pramana proposes the primitive.
7.2 Provenance-grounded alternatives
Citation-Grounded Code Comprehension [26] introduced mechanical citation verification and ar-
gued for architectural prevention superior to post-hoc detection. Cited but Not V erified [27] docu-
ments the gap between citations produced and citations that resolve to claimed content. SelfCheck-
GPT [28] approaches the same problem through within-model sampling, which is closer in spirit
to reviewer-ensembles than to Pramana and is complementary rather than competing. The survey
[29] confirms that no protocol in active use defines a typed attestation surface for agent claims.
W3C V erifiable Credentials. The V erifiable Credentials Data Model [39] standardizes typed
identity and attribute attestations with cryptographic verifiability . VC’s design centre is the
18

identity-attestation use case (an issuer asserts a property of a subject); its data model does not
type agent claims by epistemic ground, does not specify a verify() operation against a recorded
source, and does not address the LLM-oracle determinism partition. Pramana adopts VC’s archi-
tectural pattern of typed wire attestations and extends it with the four-way epistemic taxonomy ,
the verify() contract, and the wire-extension invariants specific to autonomous-agent claim veri-
fication.
General-purpose provenance models. The W3C PROV family , comprising the PROV Data
Model PROV-DM [33] and the OWL (W eb Ontology Language) ontology PROV-O [34], is the
established standard for domain-agnostic provenance, modelling entities, activities, and agents
along with their causal relations. PROV is deliberately a meta-model: it describes how to record
that an artifact was derived from another but leaves the substrate-specific verification semantics to
the consuming domain. Pramana composes with PROV at a different layer rather than competing
with it. A Pramana ClaimAttestation , viewed through a PROV lens, is a typed prov:Entity
derived from a prov:Activity (the agent’s reasoning step) acting on a source prov:Entity , and a
successful verify() outcome is a prov:wasDerivedFrom edge that a third party can independently
reconstruct. What Pramana adds on top of PROV is (i) a typed taxonomy of four claim types with
type-specific verification semantics rather than a single generic derivation edge, (ii) wire-format
invariants (CA-1, CA-2, CA-3) that make those derivations checkable end-to-end across agents,
and (iii) a verify() contract whose output is deterministic when paired with a deterministic oracle
and offline re-runnable rather than a curator-authored provenance record. A Pramana-conformant
agent could emit PROV-O serialisations of its audit trail; that mapping is straightforward and is
left as future work.
7.3 Agent-loop and self-improvement systems
A separate line of recent work pairs LLMs with deterministic verifiers in iterative loops. F unSearch
[30] and AlphaEvolve [31] both use a candidate-generator plus deterministic-evaluator architecture
to make new mathematical and algorithmic discoveries. The generator proposes candidates and the
evaluator verifies each candidate against a fitness function or test suite that does not invoke another
LLM, such that the loop’s correctness rests on the evaluator and not on the generator. Google’s
AI co-scientist [42] extends the pattern to multi-agent biomedical research, pairing specialized
generation and evaluation agents in tournament-style refinement loops that produce structured
hypothesis artifacts at every step. None of these systems define a shared wire format across vendors
or across agents, which is the gap Pramana fills. Pramana is the wire-format analogue of this
architecture for agent outputs: every claim is paired with a verify() against the recorded source,
and no probabilistic judge participates in the verification step.
7.4 Regulatory and policy work
Regulatory and policy work on algorithmic auditing has matured separately from the wire-format
literature. The frameworks cited in Section 4 reflect a regulator-side convergence on per-decision
documentary records as the audit substrate. Dell’Acqua et al. [32] characterize the “jagged fron-
tier” of AI capability that motivates per-claim-type verification rather than uniform model-level
treatment. Mökander et al. [40] survey ethics-based auditing methodology and distinguish process-
based from artifact-based audit substrates; Pramana’s per-decision record sits in the latter category .
Kroll et al. [41] develop the “accountable algorithms” framing from the legal-scholarship side, ar-
guing that verifiable algorithmic records are a necessary input to substantive due-process review
while emphasizing that structural documentation alone does not adjudicate substantive suﬀiciency .
19

Whether structural documentation in the Pramana sense suﬀices for substantive evidentiary stan-
dards (and the related question of vendor versus deployer liability raised in Mobley v. W orkday
[38]) remains an open research area outside this paper’s scope.
8. Limitations and future work
• Pilot scope and domain mismatch. Section 5 uses code generation, which has objective
binary ground truth via unit tests; the regulated domains the protocol targets have ground
truth that is contested, probabilistic, or unavailable at decision time. The pilot cannot directly
simulate those conditions. Per-domain validation in regulated domains is open work and the
natural next study .
• Sample size and confounding. The pilot uses n=100 problems and configuration-shift
comparisons confound multiple variables at once (ensemble size, capability tier, family com-
position, prompt-tuning). W e report McNemar shifts as descriptive observations rather than
mechanism claims. A larger study with held-out problem sets, a T>0 self-consistency control,
and factorial separation would substantiate quantitative claims.
• Prompt-pilot and main-run overlap. Prompt-variant selection drew from the same first-
N positions of each dataset that the main run later evaluates. The selection optimized adjusted
FPR on a single Haiku reviewer while the main run measures 7-slot ensemble FPR, so the
leakage’s effect on ensemble-level estimates is bounded by the imperfect coupling between
single-reviewer and ensemble-reviewer FPR. The leakage is concentrated in Haiku-weighted
ensemble configurations (same-model uses three Haiku calls; same-family and cross-family
include one Haiku call each), so the paired McNemar comparisons across configurations op-
erate under asymmetric leakage rather than symmetric bias; the §5.3 reporting reflects this.
Held-out replication is open work.
• Single-rater adjudication. Adjudication of clean-flagged-buggy cases is single-rater, a
methodological ceiling on the adjusted-FPR numbers. The mathematical analyses of specific
failure modes are externally verifiable and invariant to rater bias; the percentage splits are
first-rater estimates. Inter-rater reliability measures such as Cohen’s kappa are not reported,
as adjudication used a single rater; a multi-rater replication with reported kappa is the natural
next-study extension.
• Adversarial source fabrication. A CitationClaim whose source resolves to a fake source
the adversary controls is observable under Pramana but not prevented. Source-attestation
primitives compose with Pramana via the source_uri integration point and are out of scope
for v1.
• AnalogyClaim verification semantics. The v1 wire format treats AnalogyClaim VERI-
FIED outcomes uniformly whether or not a similarity_score threshold was applied (§2.2).
Regulated deployments should configure their emitters to require similarity_score ; a fu-
ture revision could introduce a distinct VERIFIED_UNTHRESHOLDED status so the distinction is
visible at the wire-format inspection layer rather than only via deployer configuration.
• TLA+ model size. Exhaustive verification uses minimal symmetry-reduced models (2
claims, BoundedT rail = 6). A supplementary 3-claim, 2-agent, 2-verifier run reached 166M+
states with zero violations before BFS exhausted compute. F ull inductive proof for arbitrary
deployment scales is deferred to a TLAPS (TLA+ Proof System) treatment.
• Capability attestation and multi-agent composition. Third-party capability attesta-
tion is absorbed by Y athartha [11] as a sibling primitive. Multi-agent claim composition (an
20

InferenceClaim depending on MeasurementClaim s from two other agents) is a v2 extension.
The TLA+ spec models single-agent lifecycles.
References
1. Agent2Agent (A2A) Protocol Specification. a2a-protocol.org.
2. Model Context Protocol (MCP) Specification. modelcontextprotocol.io.
3. Consumer Financial Protection Bureau. (2023, September 19). Adverse Action Notification
Requirements and the Proper Use of the CFPB’s Sample F orms Provided in Regulation B.
Circular 2023-03.
4. Board of Governors of the F ederal Reserve System & Oﬀice of the Comptroller of the Currency .
(2011, April 4). Supervisory Guidance on Model Risk Management. SR 11-7 / OCC Bulletin
2011-12.
5. New Y ork State Department of Financial Services. (2024, July 11). Use of Artificial Intel-
ligence Systems and External Consumer Data and Information Sources in Insurance Under-
writing and Pricing. Insurance Circular Letter No. 7 (2024).
6. Colorado Division of Insurance. (2023, September 21; effective November 14, 2023; expanded
October 15, 2025). Governance and Risk Management F ramework Requirements for Insurers’
Use of External Consumer Data and Information Sources, Algorithms, and Predictive Models.
Regulation 10-1-1.
7. U.S. Department of Health and Human Services, Oﬀice for Civil Rights, & Centers for Medi-
care & Medicaid Services. (2024, May 6). Nondiscrimination in Health Programs and Activ-
ities , Final Rule. 89 F ed. Reg. 28822 (codified at 45 C.F.R. pt. 92; §92.210 governs patient
care decision support tools).
8. EU Artificial Intel ligence Act (Regulation (EU) 2024/1689), Articles 14, 50. Effective August
2, 2026.
9. General Data Protection Regulation (GDPR), Recital 71.
10. Kadaboina, R. K. (2026). Anumati: Proof of Adherence as a F ormal Consent Model for
Autonomous Agent Protocols. arXiv:2604.16524.
11. Kadaboina, R. K. (2026). Y athartha: A Protocol-Layer T reatment of Jagged Intel ligence in
Autonomous Agent Networks. Zenodo DOI 10.5281/zenodo.19659633.
12. Kadaboina, R. K. (2026). Phala: Principal-Declared W elfare F eedback for Autonomous Agent
Networks. Zenodo DOI 10.5281/zenodo.19625612.
13. Kadaboina, R. K. (2026). Pratyahara: A Neural Tissue Defense Model for Detecting
Compromised Agents in Multi-Agent Networks. Specification name: NER VE. Zenodo DOI
10.5281/zenodo.19628589.
14. Kadaboina, R. K. (2026). Sauvidya: An Accessibility Protocol for Agent-to-Principal Interac-
tion in Autonomous Agent Networks. Specification name: P ACE. Zenodo DOI 10.5281/zen-
odo.19633139.
15. Austin, J., et al. (2021). Program Synthesis with Large Language Models. arXiv:2108.07732.
(MBPP dataset.)
16. Chen, M., et al. (2021). Evaluating Large Language Models T rained on Code.
arXiv:2107.03374. (HumanEval dataset.)
17. Kim, E., Garg, A., Peng, K., & Garg, N. (2025). Correlated Errors in Large Language Models.
arXiv:2506.07962.
18. Li, D., et al. (2025). Preference Leakage: A Contamination Problem in LLM-as-a-Judge.
21

arXiv:2502.01534.
19. Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
arXiv:2306.05685.
20. Panickssery , A., Bowman, S. R., & F eng, S. (2024). LLM Evaluators Recognize and F avor
Their Own Generations. arXiv:2404.13076.
21. W ang, P ., et al. (2023). Large Language Models Are Not F air Evaluators. arXiv:2305.17926.
22. Stureborg, R., Alikaniotis, D., & Suhara, Y. (2024). Large Language Models Are Inconsistent
and Biased Evaluators. arXiv:2405.01724.
23. Maloyan, N., Ashinov, B., & Namiot, D. (2025). Investigating the V ulnerability of LLM-as-
a-Judge Architectures to Prompt-Injection Attacks. arXiv:2505.13348.
24. Nasr, M., et al. (2025). The Attacker Moves Second: Stronger Adaptive Attacks Bypass
Defenses Against LLM Jailbreaks and Prompt Injections. arXiv:2510.09023.
25. Cemri, M., et al. (2025). Why Do Multi-Agent LLM Systems F ail? (MAST).
arXiv:2503.13657.
26. Arafat, J. (2025). Citation-Grounded Code Comprehension. arXiv:2512.12117.
27. Onweller, H., et al. (2026). Cited but Not V erified: Parsing and Evaluating Source Attribution
in LLM Deep Research Agents. arXiv:2605.06635.
28. Manakul, P ., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-Resource Black-Box
Hal lucination Detection for Generative Large Language Models. arXiv:2303.08896.
29. Y ang, Y., et al. (2025). A Survey of AI Agent Protocols. arXiv:2504.16736.
30. Romera-Paredes, B., et al. (2024). F unSearch: Mathematical Discoveries from Program Search
with Large Language Models. Nature 625, 468-475.
31. Novikov, A., et al. (2025). AlphaEvolve: A Coding Agent for Scientific and Algorithmic
Discovery. arXiv:2506.13131.
32. Dell’Acqua, F., McF owland III, E., Mollick, E., Lifshitz-Assaf, H., Kellogg, K. C., Rajendran,
S., Krayer, L., Candelon, F., & Lakhani, K. R. (2023). Navigating the Jagged T echnological
F rontier: Field Experimental Evidence of the Effects of Artificial Intel ligence on Knowledge
W orker Productivity and Quality. Harvard Business School W orking Paper No. 24-013. F orth-
coming in Organization Science.
33. Moreau, L., & Missier, P . (eds.). (2013, April 30). PROV-DM: The PROV Data Model. W3C
Recommendation. https://www.w3.org/TR/prov-dm/
34. Lebo, T., Sahoo, S., & McGuinness, D. (eds.). (2013, April 30). PROV-O: The PROV
Ontology. W3C Recommendation. https://www.w3.org/TR/prov-o/
35. F uller, J., Raman, M., Sage-Gavin, E., & Hines, K. (2021, September). Hidden W orkers:
Untapped T alent. Harvard Business School Project on Managing the F uture of W ork, in
collaboration with Accenture.
36. New Y ork City Department of Consumer and W orker Protection. (2023). Automated Em-
ployment Decision T ools: Final Rule. 6 RCNY § 5-300 et seq. Implementing Local Law 144
of 2021; enforcement effective July 5, 2023.
37. EEOC v. iT utorGroup, Inc., No. 1:22-cv-02565 (E.D.N.Y.). Consent decree filed August 9,
2023; settlement $365,000.
38. Mobley v. W orkday , Inc., No. 3:23-cv-00770 (N.D. Cal.). Order on motion to dismiss, July 12,
2024 (vendor liability as “agent” under Title VII, ADEA, ADA); collective action conditionally
certified May 16, 2025 (ADEA claims).
39. W3C. (2025). V erifiable Credentials Data Model v2.0. W3C Recommendation. https://ww
w.w3.org/TR/vc-data-model-2.0/
40. Mökander, J., Morley , J., T addeo, M., & Floridi, L. (2021, July). Ethics-Based Auditing of
Automated Decision-Making Systems: Nature, Scope, and Limitations. Science and Engineer-
22

ing Ethics, 27(4). DOI 10.1007/s11948-021-00319-4.
41. Kroll, J. A., Huey , J., Barocas, S., F elten, E. W., Reidenberg, J. R., Robinson, D. G., & Y u,
H. (2017). Accountable Algorithms. University of Pennsylvania Law Review , 165(3), 633-705.
42. Gottweis, J., et al. (2025). T owards an AI co-scientist. arXiv:2502.18864.
Appendix: artifacts and replication
The Pramana reference implementation, TLA+ specifications, A2A and MCP discovery mani-
fests, claim-attestation wire extension, empirical pilot scripts and raw API responses, the PRE-
REGISTRA TION.md artifact, and per-case adjudication files are at https://github.com/rav
ikiran438/pramana- attestation under the Apache 2.0 license. The TLC reproducer script
(specification/run_tlc.sh ) regenerates the state counts in Section 6.2; full per-type schemas,
the wire envelope specification, and the empirical pilot reproducer scripts are in the same repository .
23