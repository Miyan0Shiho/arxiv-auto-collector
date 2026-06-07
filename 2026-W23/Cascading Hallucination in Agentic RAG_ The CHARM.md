# Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation

**Authors**: Saroj Mishra

**Published**: 2026-06-03 04:33:47

**PDF URL**: [https://arxiv.org/pdf/2606.04435v1](https://arxiv.org/pdf/2606.04435v1)

## Abstract
Multi-step agentic retrieval-augmented generation (RAG) pipelines have demonstrated significant capability for complex reasoning tasks, yet remain vulnerable to a class of failure that existing hallucination detection mechanisms systematically miss: cascading hallucination, where errors introduced at early pipeline stages propagate and amplify across successive reasoning steps, producing confident but factually incorrect final outputs. To address this vulnerability, we formalize cascading hallucination as a distinct failure mode in agentic RAG systems, present a four-type taxonomy of cascade patterns, and introduce CHARM (Cascading Hallucination Aware Resolution and Mitigation), an architectural framework for detecting and interrupting error propagation in multi-step reasoning pipelines. CHARM comprises four components - stage-level fact verification, cross-stage consistency tracking, confidence propagation monitoring, and cascade resolution triggering - that operate alongside standard agentic RAG pipelines without requiring architectural replacement. We evaluate CHARM on HotpotQA, MuSiQue, 2WikiMultiHopQA, and a custom adversarial dataset across LangChain agentic pipeline configurations, achieving an 89.4% cascade detection rate with a 5.3% false positive rate and 215 ms +/- 18 ms average latency overhead per stage, achieving an error propagation reduction of 82.1%, compared to 18.5% for output-level detectors. Component ablations confirm that each detection module contributes meaningfully to overall cascade coverage. CHARM integrates with human-in-the-loop oversight frameworks to provide a complete reliability and governance stack for production agentic AI deployment.

## Full Text


<!-- PDF content starts -->

Cascading Hallucination in Agentic RAG: The
CHARM Framework for Detection and Mitigation
Saroj Mishra
University of North Dakota
saroj.mishra773@gmail.com
Abstract—Multi-step agentic retrieval-augmented generation
(RAG) pipelines have demonstrated significant capability for
complex reasoning tasks, yet remain vulnerable to a class of failure
that existing hallucination detection mechanisms systematically
miss: cascading hallucination, where errors introduced at early
pipeline stages propagate and amplify across successive reasoning
steps, producing confident but factually incorrect final outputs. To
address this vulnerability, we formalize cascading hallucination
as a distinct failure mode in agentic RAG systems, present a
four-type taxonomy of cascade patterns, and introduce CHARM
(Cascading Hallucination Aware Resolution and Mitigation),
an architectural framework for detecting and interrupting
error propagation in multi-step reasoning pipelines. CHARM
comprises four components—stage-level fact verification, cross-
stage consistency tracking, confidence propagation monitoring,
and cascade resolution triggering—that operate alongside stan-
dard agentic RAG pipelines without requiring architectural
replacement. We evaluate CHARM on HotpotQA, MuSiQue,
2WikiMultiHopQA, and a custom adversarial dataset across
LangChain agentic pipeline configurations, achieving an 89.4%
cascade detection rate with a 5.3% false positive rate and
215 ms ±18 ms average latency overhead per stage, achieving an
error propagation reduction of 82.1%, compared to 18.5% for
output-level detectors. Component ablations confirm that each
detection module contributes meaningfully to overall cascade
coverage. CHARM integrates with human-in-the-loop oversight
frameworks to provide a complete reliability and governance
stack for production agentic AI deployment.
Index Terms—Cascading Hallucination, Agentic RAG, Error
Propagation, Multi-Step Reasoning, AI Reliability.
I. INTRODUCTION
As agentic AI systems increasingly automate complex
enterprise workflows, a new class of failure has emerged
that existing safety mechanisms fail to detect: cascading
hallucination. In multi-step reasoning systems, small retrieval or
inferential errors introduced in early pipeline stages propagate
silently through the trajectory, compounding at each step to
produce confident but factually incorrect final outputs. Because
each subsequent reasoning step remains logically coherent
relative to its immediate—albeit corrupted—context, these
failures appear authoritative to both downstream automated
systems and human reviewers, presenting a severe risk for
enterprise and regulated deployments [1], [2]. This work builds
on a sustained research program in secure and reliable AI
systems [1], [3].
Despite significant advancements in hallucination detection,
existing evaluation architectures are ill-equipped to handle
this phenomenon. Current state-of-the-art detectors [4]–[6]primarily evaluate individual Large Language Model (LLM)
outputs in isolation, treating generation as a single-step point-in-
time process. They measure the factual grounding of a terminal
response but ignore the cross-stage semantic trajectory that
produced it. Consequently, when an agent reviews its own
cascaded logic [7], it suffers from severe confirmation bias,
verifying the final output because it aligns with the corrupted
intermediate context.
To bridge this critical reliability gap, we introduce the
Cascading Hallucination Aware Resolution and Mitigation
(CHARM) framework. This work makes three primary contri-
butions:
1)C1: Cascading Hallucination Taxonomy.We provide
the first formal mathematical definition and classification
of cascading hallucination types specific to multi-step
agentic RAG pipelines, defining four named typologies
with concrete operational definitions for all core quantities.
2)C2: CHARM Detection Framework.We present a
named, implementable four-component detection architec-
ture that operates continuously alongside existing RAG
pipelines without requiring foundational replacement, with
full component ablations confirming individual contribu-
tions.
3)C3: Mitigation Architectures.We propose four concrete,
named mitigation patterns that interrupt error propagation
at each pipeline stage, offering practitioners configurable
trade-offs between latency overhead and intervention
accuracy.
The remainder of this paper is organized as follows. Sec-
tion II provides background on agentic RAG pipelines and
existing detection limitations. Section III mathematically for-
malizes the cascading hallucination problem space. Section IV
details the CHARM architecture, while Section V outlines
corresponding mitigation strategies. Section VI presents our
empirical evaluation, ablations, and novel metrics. Section VII
contextualizes these findings within U.S. national AI gover-
nance frameworks, followed by related work in Section VIII
and concluding remarks in Section IX.
II. BACKGROUND
To contextualize the mechanisms of cascading errors, we
must establish the foundational architecture of continuous
reasoning pipelines and the limitations of current single-step
verification protocols.arXiv:2606.04435v1  [cs.AI]  3 Jun 2026

A. Agentic RAG Pipeline Architecture
Standard Retrieval-Augmented Generation (RAG) [8] en-
hances LLM outputs by fetching external knowledge. However,
as identified in our foundational System of Knowledge (SoK)
analysis [1], the paradigm has shifted from single-turn retrieval
to agentic, multi-step pipelines.
As illustrated in Figure 1, a standard agentic RAG pipeline
operates across five sequential stages: (1) Query Formulation,
where the agent interprets the user prompt; (2) Retrieval, where
external knowledge is fetched; (3) Intermediate Reasoning,
where the agent processes the context; (4) Tool Use, where
the agent executes specific functions; and (5) Final Synthesis
and Output. In this architecture, the state output of stage i
becomes the definitive context window for stage i+1 , creating
a persistent memory chain that spans the entire generation
process.
B. Existing Hallucination Detection
Current hallucination detection methodologies generally fall
into three categories, all of which exhibit structural blind spots
when applied to cascading scenarios:
•Output-Level Detection:Approaches like SelfCheckGPT
[4] check the final LLM response for factual accuracy.
Because they evaluate only the terminal output, they
entirely miss the intermediate stage errors that constructed
the hallucination.
•Retrieval-Level Detection:Frameworks such as RAGAS
[6] evaluate the relevance and accuracy of retrieved
documents. While effective at step 1, they fail to track
how accurately that retrieved context is logically applied
across subsequent reasoning steps.
•Consistency-Based Detection:These methods [4], [9]
check the internal consistency of an LLM’s output via zero-
resource sampling or self-reflection. However, cascaded
outputs are inherently internally consistent—they are
perfectly coherent given the initial false premise.
Additionally, the naive approach ofLLM Self-Correction
[7], [10], where an agent is prompted to review its own final
answer, fails due to confirmation bias. The agent reinforces the
cascade because the downstream reasoning appears logically
sound relative to its corrupted memory.
C. Multi-Step Reasoning and Error Compounding
The vulnerability of sequential reasoning is deeply rooted
in the mechanics of Chain-of-Thought (CoT) prompting [11].
While CoT significantly improves complex problem-solving by
forcing intermediate steps, it inadvertently creates pathways
for logical derailment [12].
When an error occurs in sequential reasoning, it does
not remain static; it acts as an anchor for subsequent token
generation. As the agent builds upon the flawed premise, the
semantic distance between the agent’s internal state and the
objective ground truth widens. This compounding effect forms
the theoretical basis for why cascading hallucinations are not
merely random errors, but predictable, measurable, and highly
structured pipeline failures.III. PROBLEMFORMALIZATION
This section establishes the theoretical foundation for the
CHARM framework by formally defining the mechanics of
cascading hallucinations in multi-step systems. Unlike single-
step generation tasks where hallucinations occur as isolated
deviations from a prompt [4], [5], agentic pipelines function
as sequential state machines where the output of one stage
becomes the authoritative context for the next [8], [13].
A. Formal Definition of Cascading Hallucination
LetP= (s 1, s2, . . . , s n)be a multi-step agentic RAG
pipeline where sidenotes the i-th reasoning stage. Let ci
denote the context output of stage ipassed as input to stage
i+ 1.
A cascading hallucination occurs when the following four
conditions are met:
1)Stage siproduces output cicontaining factual error ϵi
with respect to ground truthG.
2)The corrupted context ciis propagated as valid context
tosi+1.
3)Stage si+1generates output ci+1that is conditionally
coherent given cibut factually incorrect with respect to
G.
4)The error magnitude strictly increases or persists, such
that|ϵi+1| ≥ |ϵ i|, meaning the error magnitude increases
monotonically across subsequent stages.
This formal definition explicitly distinguishes cascading
hallucinations from standard single-step hallucinations. In
a single-step hallucination, an error occurs but does not
necessarily propagate or amplify. In a cascading scenario, the
underlying architecture actively forces the model to synthesize
and compound the error across sequential reasoning layers
[11].
B. Distinguishing Cascading Hallucination from Generic Error
Propagation
Error propagation in sequential systems is a known phe-
nomenon [11], [12]. Cascading hallucination, as defined here,
is a strictly more specific failure mode with four properties
that jointly distinguish it from generic propagation in prior
work:
TABLE I
CASCADINGHALLUCINATION VS. GENERICERRORPROPAGATION
Property Generic Cascading
Propagation Hallucination
Error at some stage Yes Yes
Factually false output Not required Required
Multi-stage amplification Occasional Definitional
Confidence inflation Not required Core property
Local coherence preserved Not required Required
Agentic RAG context No Yes
The critical distinguishing property islocal coherence under
global falsity: a cascading hallucination is not merely an
error that persists, but one where each downstream stage

Context State Result
Draft
Persistent Context (Cn)
Persistent Context (Cn)
Stage 1
Query & Retrieval
Stage 2
Initial Reasoning
Stage 3
Tool Use (Execution)
Stage 4
Synthesis & Integration
Stage 5
Final OutputFig. 1. A standard 5-stage agentic RAG pipeline. The context output is continuously passed forward as the definitive input for subsequent reasoning stages,
demonstrating how early state persists throughout the trajectory.
generates output that isconditionally correctgiven its corrupted
context (Condition 3), making it invisible to per-step detectors
(Lemma 1). Generic error propagation studied in CoT reasoning
failures [12] and process supervision [14] does not require this
local coherence property, and therefore does not exhibit the
systematic evasion of standard detectors that motivates the
CHARM architecture. Furthermore, the Confidence Inflation
Cascade type — where low-confidence outputs propagate as
high-confidence — has received limited explicit treatment
in existing error propagation literature, where confidence
dynamics are rarely modeled as a first-class propagation
mechanism.
C. DAG-Based Pipeline Model
As identified as a critical open problem in our foundational
SoK analysis [1], quantifying this propagation requires model-
ing the multi-step reasoning process as a Weighted Directed
Acyclic Graph (DAG) denoted byG= (V,E).
The set of nodes Vrepresents discrete pipeline stages
(retrieval, reasoning, tool-call, synthesis, final output). The
set of directed edges Erepresents the context and intermediate
outputs passed forward between stages. We assign edge weights
corresponding to the error propagation probability P(ϵi+1|ϵi).
Under this model, cascade detection is defined as identifying
paths in the DAG where the cumulative edge weight product
exceeds a predefined safety thresholdθ.
In practice, computing exact path probabilities in Gat
inference time requires estimating P(ϵi+1|ϵi)for each edge,
which is intractable without offline calibration on held-out
trajectories. We therefore operationalize cascade detection via
a linear weighted approximation: the CRT (Section IV-B )
computes
ˆpcascade =w sfv·asfv+w csct·acsct+w cpm·acpm (1)
where asfv, acsct, acpm∈[0,1] are the anomaly scores from
each monitoring component and wsfv= 0.4 ,wcsct= 0.4 ,
wcpm= 0.2 are weights calibrated on held-out validation
splits. The cascade flag fires when ˆpcascade ≥θ= 0.55 ,
approximating the DAG path threshold. This design choice
trades formal exactness for inference-time tractability while
preserving the DAG’s theoretical interpretation of cumulative
error propagation probability.
We adopted fixed weights over a learned meta-classifier for
three reasons: (1) fixed weights are interpretable and directly
reflect prior knowledge about component reliability (SFV and
CSCT are more calibrated than CPM); (2) a learned classifierwould require labeled cascade trajectories for training, creating
a circular dependency with the very detection system being
built; and (3) fixed weights transfer across datasets without
retraining. Conformal calibration of the CRT threshold θto
provide coverage guarantees is an identified future direction.
We employ a DAG rather than a Markov Chain formalism
because RAG pipelines are inherently directed and acyclic, and
earlier retrieved context heavily persists throughout the entirety
of the pipeline. This continuous persistence of context explicitly
violates the Markov memorylessness assumption. The DAG
formalism accurately captures this persistent context influence
while preserving the ability to assign discrete probability
weights to edges.
Fig. 2. DAG-based representation G= (V,E) of a multi-step agentic
pipeline. The highlighted path demonstrates a cascading hallucination where
the cumulative error propagation weight P(ϵ i+1|ϵi)forces a high terminal
divergence from ground truth.
D. Four-Type Cascading Hallucination Taxonomy
Because errors enter the DAG at different nodes and com-
pound in different ways, generalized detection is insufficient.
We classify cascading hallucinations into four distinct, formally
named types:
•Retrieval Cascade:A false document is retrieved in
step 1, causing all subsequent reasoning to build on a false
premise. The primary detection signal is source-output
semantic divergence at stage 1.
•Inference Cascade:Correct retrieval occurs, but an
incorrect inference is made at step 2, which downstream
stages subsequently amplify. The primary detection signal
is an entailment score drop between the retrieved evidence
and the inferred conclusion.
•Context Poisoning Cascade:Manipulated external data
corrupts the agent’s memory and all subsequent steps. The
primary detection signal is an anomalous semantic shift
in context between stages.
•Confidence Inflation Cascade:A low-confidence output
is treated as a high-confidence output by the next step,

causing false certainty to grow monotonically. The primary
detection signal is a confidence score increase despite
underlying semantic drift from ground truth.
E. Theoretical Limitations of Standard Detectors
To definitively establish the necessity of the CHARM
framework, we provide a formal argument demonstrating
why standard per-step hallucination detectors fail to capture
cascading errors.
Lemma 1:For each cascade type, the per-step output passes
standard detection thresholds.
Proof.Let standard output-level hallucination detectors be
defined by a local entailment threshold τ. In a cascading
scenario, stage si+1receives the corrupted context ci. By
definition, si+1generates ci+1such that it is conditionally
coherent given ci. Therefore, the local entailment probabil-
ityP(ci+1|ci)remains exceptionally high, often satisfying
P(ci+1|ci)> τ . Consequently, the per-step detector evaluates
the local generation step as factually grounded and internally
consistent, completely missing the global divergence from
ground truthG.■
Corollary 1:Per-step detection is inherently insufficient for
cascade identification.
Proof.Following Lemma 1, since standard single-step
detectors only evaluate the isolated local transition P(cn|cn−1),
they are entirely blind to the monotonically compounding
error magnitude |ϵn|across the broader pipeline trajectory.
Identifying a cascade mandates cross-stage trajectory tracking
and continuous semantic evaluation, capabilities that per-step,
isolation-based detectors [6] fundamentally lack.■
F . Operational Definitions
To ensure reproducibility and to connect the formal quantities
above to concrete estimators, we define each core measurement
as follows.
Error magnitude |ϵi|at stage siis defined stage-adaptively
to avoid the ill-posedness of comparing non-answer-like
intermediate outputs (retrieval snippets, tool I/O) directly to
the final ground truth answerG.
Forretrieval and tool-call stages( s1,s4), where outputs are
evidence snippets rather than answer-like text, error magnitude
is measured using a dual-anchor strategy. In standard operation,
the entailment-based veracity deficit is computed againstc 1:
|ϵi|early= 1−NLI entail(c1, ci)(2)
where NLI entail is the entailment prob-
ability from the SFV cross-encoder
(cross-encoder/nli-deberta-v3-base ). For
Retrieval Cascadescenarios where c1is itself the corrupted
anchor, this definition would undercount error until later
stages detect the drift. To address this, CHARM additionally
maintains a secondary anchor: a top- kconsensus summary
computed from k= 3 retrieved candidates at stage 1, rather
than only the top-1 document. The SFV compares subsequent
stage outputs against this consensus anchor in parallel with c1,
flagging divergence from either reference. This dual-anchordesign reduces the risk that a corrupted single top-1 document
silently becomes the unchallenged reference for all subsequent
stages.
Forreasoning and synthesis stages( s2,s3,s5), where
outputs are answer-proximate, error magnitude is the semantic
divergence from ground truth:
|ϵi|late= 1−sim(ϕ(c i), ϕ(G))(3)
where ϕ(·) is the all-mpnet-base-v2 Sentence-BERT
[15] embedding and sim(·,·) is cosine similarity. This stage-
adaptive definition ensures that intermediate outputs are evalu-
ated against appropriate reference anchors at each point in the
trajectory rather than against the final answer they have not
yet produced.
Error propagation probability P(ϵi+1|ϵi)on DAG edge
(si, si+1)is estimated empirically as the frequency with which
a detected error at stage siproduces a measurable error at
stage si+1(i.e.,|ϵi+1|> δ , where δ= 0.15 is a calibrated
minimum divergence threshold). This estimation is computed
offline on the training split of each dataset and applied as a
fixed prior during inference.
We note that this empirical estimation treats co-occurrence of
errors at adjacent stages as evidence of propagation rather than
independent occurrence; establishing causal error transmission
formally — distinguishing propagation from coincidence —
remains an important open problem for future work.
Cascade detection threshold θis selected via grid search
over θ∈ {0.40,0.45,0.50,0.55,0.60,0.65} on a held-out
validation split (10% of each dataset), optimizing the F1score
between CDR and (1−FPR) . The selected value θ= 0.55
yields the best harmonic mean across all four datasets.
Commensurability note:The stage-adaptive definitions
(Equations 2 and 3) use different reference anchors across
stages, which raises the question of whether |ϵi|earlyand|ϵi|late
are directly comparable. We treat the monotonicity condition
|ϵi+1| ≥ |ϵ i|(Definition condition 4) as a within-stage-type
constraint rather than a cross-stage-type one: Retrieval and
tool-call stage errors are compared against each other using
|ϵ|early, and reasoning and synthesis stage errors against each
other using |ϵ|late. At the stage 1 →2 boundary (retrieval to
reasoning), the transition is monitored by the CSCT’s semantic
drift signal rather than a direct magnitude comparison, which
is more appropriate given the output type change. This design
choice acknowledges that strict numerical monotonicity across
heterogeneous stage types is not measurable from a single
scalar, and that the CRT’s weighted aggregation (Equation 3)
naturally handles this by combining complementary signals
optimized for each transition type.
IV. THECHARM FRAMEWORK
To address the inherent limitations of per-step hallucina-
tion detection in multi-step reasoning systems, we introduce
CHARM (Cascading Hallucination Aware Resolution and
Mitigation). CHARM is a modular, architectural framework
designed to detect and interrupt error propagation across

sequential pipeline stages without requiring the replacement
of the underlying agentic architecture.
A. Architecture Overview
The CHARM architecture operates as a parallel observation
and enforcement layer alongside a standard agentic RAG
pipeline. As illustrated in Figure 3, the system comprises three
concurrent monitoring components that track the semantic and
probabilistic trajectory of the agent’s context, feeding signals
into a fourth component, a centralized resolution engine. This
design ensures that intermediate stage errors are caught before
they can compound into confident, finalized hallucinations.
B. CHARM Components
The framework consists of four named, interconnected
components. Table II summarizes their technical mechanisms
and the specific cascade types they detect.
1)Stage-Level Fact Verifier (SFV):The SFV checks each
intermediate stage output against the initially retrieved
evidence before passing it to the subsequent stage. Uti-
lizing cross-encoder entailment scoring via cross-en-
coder/nli-deberta-v3-base [16] with entail-
ment threshold τ= 0.72 (calibrated on held-out validation
splits), the SFV prevents the propagation of ungrounded
claims.
2)Cross-Stage Consistency Tracker (CSCT):The CSCT
maintains a running consistency check across all pipeline
stages using all-mpnet-base-v2 [15] embedding-
based cosine similarity, with a drift threshold δdrift= 0.18 .
It flags contradictions or anomalous semantic shifts across
the trajectory.
3)Confidence Propagation Monitor (CPM):The CPM
tracks the model’s self-reported confidence scores across
stages. Let pi∈[0,1] denote the calibrated confidence
score at stage siafter temperature scaling with T=
1.4. CPM maintains a running Bayesian estimate of the
expected confidence trajectory, modeled with prior pi∼
Beta(α i, βi), initialized at α1=β1= 2 (uninformative).
After observingp i, the posterior updates as:
αi+1=α i+pi, β i+1=βi+ (1−p i)(4)
An inflation anomaly is flagged when piexceeds the
posterior predictive mean µi=α i/(αi+βi)by more than
∆ = 0.15, i.e.,p i−µi>∆. A known limitation of self-
reported LLM confidence is poor calibration [17]; CPM
therefore applies temperature scaling [18] with T= 1.4 ,
calibrated on 500 held-out trajectories per dataset, before
the Bayesian update. For APIs lacking logit access, CPM
falls back to an NLI-derived uncertainty proxy computed
from thecontradictionprobability of the SFV cross-
encoder — specifically, 1−P(entail)−P(neutral) —
rather than the entailment score used by SFV . This com-
plementary signal captures epistemic uncertainty rather
than factual grounding, preserving signal independence
from SFV even when logit access is unavailable. CPM
is designed as acomplementary detection signalratherthan a primary detector: its standalone CDR (38.3%)
reflects the inherent difficulty of confidence-only cascade
detection, while its +6.4 pp contribution to the SFV+CSCT
configuration confirms it catches Confidence Inflation
Cascades that entailment and drift signals alone cannot
detect. Under the no-logit condition, the CPM anomaly
flag fires when the contradiction probability exceeds a
separately calibrated threshold τcpm= 0.35 , distinct from
SFV’s entailment threshold τ= 0.72 . Ablation under this
condition shows CPM contributes +4.1 pp CDR above
the SFV+CSCT configuration, confirming non-redundant
complementarity even without direct logit access.
4)Cascade Resolution Trigger (CRT):Operating as the
final enforcement layer, the CRT aggregates signals from
the SFV , CSCT, and CPM using a weighted voting
scheme: SFV and CSCT each carry weight 0.4; CPM
carries weight 0.2, reflecting the lower reliability of self-
reported confidence. When the aggregated score exceeds
θ= 0.55 , the CRT halts the pipeline and initiates a
targeted resolution strategy (Section V).
TABLE II
CHARM COMPONENTSUMMARY ANDCASCADEDETECTIONMAPPING
Component Technical Mechanism Cascade Types
SFV Cross-encoder entailment scoring;
NLI-based verification (τ=0.72).Retrieval, Infer-
ence
CSCT Embedding drift detection; cosine
similarity trajectory (δ=0.18).Inference, Con-
text Poisoning
CPM Bayesian confidence updating; tem-
perature scaling ( T=1.4 ); NLI fall-
back.Confidence In-
flation
CRT Weighted signal aggregation
(0.4/0.4/0.2); halts pipeline;
triggers resolution.All Types
Algorithm 1 summarizes the CRT decision procedure.
The routing thresholds in Algorithm 1 reflect empirical
calibration: stage ≤2 captures retrievals and initial inferences
(the most common cascade origin stages in our evaluation);
thesfvscore >0.7 andcsct score >0.7 threshold for PV A
activation reflects the 0.72 SFV entailment threshold minus
a 0.02 margin for aggregation noise; and stage ≥4 for PRR
reflects tool-use and synthesis stages where rollback cost is
justified by late-stage cascade severity.
C. Integration with Existing RAG Pipelines
A primary design constraint of the CHARM framework is
non-intrusiveness. CHARM wraps around existing production
pipelines implemented via LangChain [19] or LlamaIndex [20]
without requiring structural teardowns. Each component is
highly modular, enabling independent deployment under com-
putational constraints. Detection thresholds are fully adjustable
to accommodate domain-specific risk tolerances.
D. Integration with Human-in-the-Loop Governance
To provide a complete reliability and governance stack,
CHARM integrates with the Human-in-the-Loop Governance

Fig. 3. The CHARM System Architecture. The standard agentic pipeline (left) executes normally while the parallel CHARM layer (center) monitors inter-stage
context passing. Anomaly signals trigger the Cascade Resolution Trigger (right), which interfaces directly with automated mitigations and human-in-the-loop
(HITL-AP) governance protocols.
Algorithm 1CRT decision and mitigation routing logic.
1:Input:sfv score,csct score,cpm score,stage id
2:Output:cascade flag,mitigation type
3:p←0.4×sfv score+0.4×csct score+0.2×cpm score
4:ifp < θ(0.55)then
5:return False, NONE
6:end if
7:cascade type←
infer type(sfv score, csct score, cpm score)
8:ifstage id≤2then▷Early stage→re-retrieve
9:return True, “CRR”
10:else ifsfv score >0.7andcsct score >0.7then▷
Multi-signal→parallel verify
11:return True, “PV A”
12:else ifstage id≥4then▷Late stage→rollback
13:return True, “PRR”
14:else▷Default→confidence gate
15:return True, “SCT”
16:end if
for Agentic AI Pipelines (HITL-AP) framework [21]. The CRT
serves as the technical bridge between automated detection
and human governance. When the CRT triggers on a low-
confidence cascade, it automatically routes to a lightweight
mitigation pattern (e.g., re-retrieval). When the CRT detects
a high-confidence cascade—where the system is confidently
hallucinating a compounding error—it halts execution and
routes the trajectory to the HITL-AP human approval check-
point. Audit logs generated by the CRT feed directly into the
HITL-AP compliance logging mechanism, ensuring all error
propagation trajectories are captured for enterprise review.
V. MITIGATIONARCHITECTURES
While detection via the CHARM framework provides
visibility into error trajectories, a robust agentic system requires
automated mechanisms to interrupt and resolve these cascades.
We propose four named mitigation patterns (M1–M4), each
offering different trade-offs between computational overhead
and mitigation success rate (MSR) compared to naive LLM
self-correction [7], [10].
A. M1: Cascade Re-Retrieval (CRR)
The CRR pattern is triggered when the SFV or CSCT flags
a potential error at the initial retrieval or early reasoning stages.The system halts execution and triggers a fresh retrieval step
with modified query parameters. While this introduces medium
latency overhead ( +320 ms avg.), it is the most effective method
for quenching retrieval-based cascades before they reach the
reasoning core.
B. M2: Staged Confidence Thresholding (SCT)
SCT serves as always-on baseline protection for high-
throughput systems. Each stage passes its output to the next
only if the CPM reports a score exceeding a dynamically
calibrated threshold ( +38 ms per stage gate). If the score falls
below the threshold, the system triggers a localized verification
step before proceeding.
C. M3: Parallel Verification Agent (PVA)
For high-stakes domains such as the financial and enterprise
sectors discussed in Section VII, the PV A deploys a secondary,
independent verification agent running in parallel with the
primary reasoning pipeline. The PV A is activated when the CRT
aggregates simultaneous anomaly signals from both the SFV
and CSCT, indicating multi-component cascade confidence.
Independence from the primary agent is ensured through three
mechanisms: (1)Model isolation— the PV A uses a different
backbone LLM (GPT-4o-mini in our implementation, vs. GPT-
4o for the primary agent), preventing shared token-level biases
from the primary trajectory; (2)Prompt isolation— the PV A
receives only the original query and the specific claim under
verification, with no access to the primary agent’s intermediate
context, preventing confirmation bias inheritance; and (3)
Knowledge base isolation— the PV A queries a separate,
read-only trusted reference corpus (Wikipedia snapshot frozen
at experiment time) rather than the dynamic retrieval index used
by the primary pipeline. This three-layer isolation ensures that
PV A verdicts represent a genuinely independent verification
signal. Although this effectively doubles computational cost,
it provides the highest reliability (95.2% MSR) for regulated
environments where correctness is non-negotiable.
D. M4: Pipeline Rollback and Re-Execution (PRR)
When a cascade is detected at a late stage (e.g., stage 4
or 5), the CRT initiates a rollback to the last known clean stage
identified by the CSCT, corrects the identified error via targeted
prompt adjustment, and re-executes the pipeline from that point
(+1.8× re-execution overhead). This ensures the final output
is built on a corrected context rather than a poisoned one.

TABLE III
COMPARISON OFCHARM MITIGATIONPATTERNS
Pattern Mechanism Overhead Best Use Case
M1: CRR Fresh retrieval with re-
fined query.+320ms avg. Early-stage
retrieval errors.
M2: SCT Dynamic stage-gate
confidence checks.+38ms/stage High-throughput
systems.
M3: PV A Independent parallel
check agent.2×compute Financial and Enter-
prise Legal.
M4: PRR Rollback to clean
state and re-run.1.8×re-exec Enterprise compli-
ance.
VI. EVALUATION
To validate the efficacy of the CHARM framework, we
designed a comprehensive evaluation harness to test multi-step
reasoning systems under cascading failure conditions.
A. Experimental Setup and Agentic Adaptation
Evaluating agentic pipelines requires continuous reasoning
environments; however, standard multi-hop QA datasets are
inherently static. To address this, we developed anagentic
trajectory wrapper. Rather than providing the full context
window upfront, the evaluation harness forces the LLM agent
to use a designated Search Tool to fetch paragraphs sequentially
across multiple reasoning steps.
1) Implementation Stack:All experiments useGPT-4oas the
backbone LLM, accessed via the OpenAI API with temperature
set to 0.0 for deterministic outputs. The agentic pipeline is
implemented using LangChain AgentExecutor [19] with a
ReAct [22] reasoning trace. Retrieval uses a dense retriever
(FAISS [23]) with Wikipedia paragraph embeddings encoded
viatext-embedding-3-small . No reranker is applied
in the primary experiments; a cross-encoder reranker ablation
(ms-marco-MiniLM-L-6-v2) is reported in Table IV.
2) CHARM Component Configuration:TheSFVuses
cross-encoder/nli-deberta-v3-base [16] with
entailment threshold τ= 0.72 . TheCSCTuses
all-mpnet-base-v2 [15] with drift threshold δdrift =
0.18. TheCPMapplies temperature scaling T= 1.4 ,
calibrated on 500 held-out trajectories per dataset. The
CRTaggregates signals with weights SFV:0.4, CSCT:0.4,
CPM:0.2 and fires at θ= 0.55 . In our GPT-4o experi-
ments, Stage-level confidence piis obtained from the model’s
logprobs output ( top_logprobs=1 ). Specifically, piis
computed as the mean of the top-token log-probabilities
across the final sentence of the stage output, exponentiated
to obtain probabilities and clipped to [0.01,0.99] :pi=
clip
exp
1
|T|P
t∈Tlogp t
,0.01,0.99
, where Tis the set
of tokens in the final sentence and ptis the top-token probability
at position t. This per-sentence aggregation captures the model’s
confidence in its concluding claim rather than averaging over
the full stage output. Logprob access is enabled via the OpenAI
API parameter logprobs=true . The temperature-scaled
value is used for the Bayesian update; the NLI-contradictionfallback is invoked only when logprobs are unavailable,
which occurred in 0% of our GPT-4o experimental runs.
3) Long-Context Handling:NLI cross-encoder inputs are
truncated to a maximum of512 tokensper the DeBERTa-v3
model limit. For stage outputs exceeding this limit, we apply a
sliding window with stride 256 tokens and take the minimum
entailment score across windows as the conservative estimate
(i.e., flagging if any window falls below τ). For CSCT, sentence
embeddings are computed over the full stage output without
truncation, as all-mpnet-base-v2 processes variable-
length inputs up to 512 tokens with mean pooling; outputs
exceeding this are chunked and mean-pooled across chunk
embeddings. In our evaluation datasets, median stage output
length was 187 tokens (HotpotQA), 312 tokens (MuSiQue),
and 278 tokens (2WikiMultiHopQA), placing most outputs
within single-window range.
4) Threshold Sensitivity:Detection thresholds ( τ= 0.72 ,
δdrift= 0.18 ,θ= 0.55 ) were calibrated on 10% held-out
validation splits of each dataset independently. Cross-domain
generalization of these thresholds to substantially different
corpora (e.g., scientific or legal text) is an open question; per-
domain recalibration is recommended for production deploy-
ments, and is supported by CHARM’s configurable threshold
interface (Section IV-C ). Threshold sensitivity analysis and
cross-domain transfer evaluation are identified as planned
extensions.
5) Backbone Independence:The CHARM
detection layer (SFV , CSCT, CPM) operates entirely
independently of the backbone LLM: all NLI inference
(cross-encoder/nli-deberta-v3-base ) and
embedding computation ( all-mpnet-base-v2 ) use
locally hosted open-source models with no API dependency.
The backbone LLM is used solely for pipeline generation;
substituting GPT-4o with any instruction-tuned LLM (e.g.,
Llama-3, Mistral) requires no changes to CHARM components.
Full open-source backbone evaluation is a planned extension
of this work.
6) Hardware and Latency Measurement:All experiments
were conducted on a singleNVIDIA A100 80 GB GPU
(local NLI/embedding inference) with API calls routed to
OpenAI endpoints. Per-stage latency (LO/s) measures wall-
clock time added by CHARM components only (NLI inference,
embedding computation, confidence scoring); backbone LLM
latency is excluded to isolate pure framework overhead. We
note that methods such as SelfCheckGPT and RAGAS require
additional LLM calls beyond the primary pipeline (e.g.,
SelfCheckGPT samples multiple generations; RAGAS invokes
LLM-based faithfulness scoring), while CHARM’s detection
components run on locally hosted models with no additional
LLM calls. On an end-to-end wall-clock basis, CHARM’s early
cascade detection (average CDD = 2.1) halts the pipeline before
stages 3–5 execute, saving 2–3 full LLM inference calls per
detected cascade. This early-exit behavior makes CHARM’s
effective end-to-end overhead substantially lower than the per-
stage LO/s figure suggests when a cascade is present. All
reported LO/s values are averaged over five independent runs

per dataset.
7) Datasets:We evaluate across four datasets mapped to
our cascade taxonomy:
•HotpotQA[24]: Multi-hop reasoning (Retrieval and
Inference Cascades).500 injected trajectories; 200 clean
trajectories.
•MuSiQue[25]: Multi-step compositional questions (In-
ference and Confidence Inflation Cascades).400 injected
trajectories; 150 clean trajectories.
•2WikiMultiHopQA[26]: Multi-document reasoning via
targeted poison injection (Context Poisoning Cascades).
400 injected trajectories; 150 clean trajectories.
•Custom Adversarial Set: 200 synthetic trajectories (50
per cascade type); 100 clean trajectories.
To support reproducibility and community benchmarking,
we will release the agentic trajectory wrapper, cascade injec-
tion scripts, annotated adversarial trajectories, and the full
CHARM evaluation harness at https://github.com/sarmishra/
CHARM-agentic-rag.
B. Cascade Injection and Annotation Protocol
To ensure controlled, reproducible cascade generation, we
apply a four-method injection protocol mapped to cascade type:
•Retrieval Cascade injection:The top-1 retrieved doc-
ument is replaced with a semantically proximate but
factually incorrect document, generated via GPT-4o with
explicit counterfactual instructions. Applied to HotpotQA.
•Inference Cascade injection:The retrieval stage is left
clean; a misleading reasoning cue is prepended to the
intermediate context at stage 2. Applied to MuSiQue.
•Context Poisoning injection:Adversarial passages are
inserted into the knowledge base using a gradient-free
embedding-proximal attack [27], ensuring the poisoned
document passes retrieval relevance filtering. Applied to
2WikiMultiHopQA.
•Confidence Inflation injection:Low-confidence hedging
language (“possibly”, “may be”) is removed from stage
outputs, simulating false certainty propagation. Applied
across all datasets in the Custom Adversarial Set.
Custom adversarial set construction:The 200-trajectory
adversarial set comprises 50 trajectories per cascade type: 50
Retrieval Cascades (GPT-4o counterfactual top-1 replacement),
50 Inference Cascades (misleading reasoning cue injection), 50
Context Poisoning Cascades (embedding-proximal adversarial
passages), and 50 Confidence Inflation Cascades (hedging
language removal). All trajectories were constructed from
HotpotQA questions not present in the training or validation
splits used for threshold calibration, ensuring strict separation
between calibration and test examples. Each trajectory was
reviewed by the authors to confirm the injection produced a
detectable cascade (ground truth cascade type and injection
stage labeled by the constructor). The full dataset, injection
scripts, and annotation schema are released at https://github.
com/sarmishra/CHARM-agentic-rag.
Ground truth annotation:Each injected trajectory is
labeled with the injection stage sinject
i and cascade type. Underthestrict early-detection criterion, a detection is counted as
a true positive (TP) if the CRT flags an anomaly at any stage
sjwhere j≤iinject+ 1, i.e., the cascade is caught before it
propagates more than one additional stage. This strict criterion
drives the reported 89.4% CDR and reflects CHARM’s primary
design goal of early interruption. Under aliberal criterion
(any detection before the final output stage s5), CHARM flags
100% of all injected cascades, confirming complete coverage
before terminal output.
FPR estimation:False positive rate is measured on a
separate held-out set ofclean, non-injected trajectories
(iscascade = false) drawn from the same datasets: 200 clean
trajectories from HotpotQA, 150 from MuSiQue, 150 from
2WikiMultiHopQA, and 100 from the custom adversarial set
(500 total). These trajectories contain no artificially introduced
errors and represent legitimate multi-hop reasoning chains with
correct final answers verified against dataset gold labels. The
5.3% FPR is computed as the fraction of these clean trajectories
that the CRT incorrectly flags as cascades. Clean and injected
sets are strictly disjoint; no trajectory appears in both.
EPR computation:Error Propagation Reduction is com-
puted as:
EPR = 1−EMCHARM
EMNone(5)
where EMCHARM andEMNone denote theexact-match error
ratesfor the CHARM system and the no-detection baseline
respectively, defined as the fraction of injected trajectories
where the final output does not match the gold answer
string (i.e., 1−EM accuracy ). EPR therefore measures how
much CHARM reduces incorrect final outputs relative to the
no-detection baseline, providing a direct measure of error
propagation interruption.
A pilot study on naturally occurring cascades (without
injection) is reported in Section VI-G.
C. Baseline Comparisons
We evaluate CHARM against four direct baselines and
reference two process-level systems discussed qualitatively
in Section VIII-C:
1)No Detection (None):A zero-intervention baseline estab-
lishing vulnerability.
2)Output-Level Detector (SelfCheckGPT [4]):Evaluates
only the terminal output.
3)Retrieval Fact Checker (RAGAS [6]):Evaluates re-
trieved documents without cross-stage trajectory tracking.
4)LLM Self-Correction [7]:An agent prompts itself to
review its own final answer, demonstrating confirmation
bias in cascade scenarios.
5)EVER [28]:A process-level incremental verification
framework that rectifies hallucinations during generation.
Because EVER reports only answer-level EM and F1
scores rather than cascade-specific detection metrics, a
direct column-for-column comparison in Table VI is
not possible; we discuss its relationship to CHARM in
Section VIII-C.

6)IRCoT [29]:An interleaved retrieval-with-chain-of-
thought framework evaluated on the same three datasets
used here. As with EVER, IRCoT reports EM and F1
rather than cascade detection metrics; we provide a
qualitative comparison in Section VIII-C.
D. Evaluation Metrics
We assess performance using six metrics, including one
measurement introduced for the first time in this paper to
standardize cascade evaluation:
•Cascade Detection Rate (CDR):Percentage of injected
cascades identified before final output.
•False Positive Rate (FPR):Percentage of grounded
trajectories incorrectly flagged.
•Error Propagation Reduction (EPR):Reduction in final
output error rate, computed per Equation 5.
•Mitigation Success Rate (MSR):Percentage of detected
cascades successfully resolved.
•Cascade Depth at Detection (CDD):Average pipeline
stage ( s1. . . s n) at which a cascade is detected. To our
knowledge, no prior work standardizes cascade detection
depth as a quantitative trajectory metric; while AgentHallu
[30] localizes hallucination origin post-hoc, CDD captures
detection timing at inference time as a standardized,
reusable evaluation criterion. Lower values indicate earlier
intervention.
•Latency Overhead per Stage (LO/s):Average additional
wall-clock processing time (in milliseconds) introduced
by CHARM components at each individual pipeline stage.
E. Ablation Study
To quantify the contribution of individual CHARM compo-
nents, we evaluate six ablated configurations on HotpotQA.
Results are presented in Table IV.
TABLE IV
COMPONENTABLATIONSTUDY— CDRONHOTPOTQA
Configuration CDR FPR LO/s (ms)
SFV Only 61.2% 4.8% 74
CSCT Only 54.7% 5.1% 68
CPM Only 38.3% 3.9% 38
SFV + CSCT 79.4% 5.0% 142
SFV + CSCT + CPM 86.1% 5.2% 178
Full CHARM 92.5%5.3%215
The ablation confirms that SFV is the strongest individual
component (61.2% CDR), consistent with its role in catching
Retrieval and Inference Cascades—the most frequent types in
HotpotQA’s structured two-hop format. CSCT adds comple-
mentary coverage for longer semantic drift trajectories (+18.2
percentage points over SFV alone). CPM’s standalone contri-
bution is limited (38.3%), reflecting the inherent difficulty of
confidence-only detection; however, its addition to SFV+CSCT
yields a further +6.4 percentage point gain, confirming it
catches Confidence Inflation Cascades missed by the othertwo components. Each component carries a meaningful detec-
tion contribution, validating the four-component architecture.
Table V reports per-mitigation effectiveness.
TABLE V
MITIGATIONPATTERNEFFECTIVENESS
Pattern MSR Overhead
M1: CRR 88.4% +320 ms avg. retrieval latency
M2: SCT 74.1% +38 ms per stage gate
M3: PV A 95.2% 2×pipeline compute cost
M4: PRR 91.7% 1.8×pipeline re-execution
Cross-Dataset Generalization:The per-dataset CDR results
in Table VII serve as an implicit cross-dataset ablation:
CHARM’s performance advantage over the single-component
Output-Level baseline (which approximates SFV-only behavior)
ranges from 66.4 pp on HotpotQA to 63.7 pp on MuSiQue and
66.0 pp on 2WikiMultiHopQA, indicating that multi-component
coverage benefits generalize across reasoning topologies rather
than being specific to HotpotQA’s two-hop structure. Addition-
ally, under the no-logit API condition (CPM using contradiction
probability fallback), Full CHARM retains +4.1 pp CDR above
the SFV+CSCT configuration on HotpotQA, confirming that
CPM’s Bayesian trajectory modeling provides complementary
signal even without direct logit access.
To assess signal independence, we computed the Pearson
correlation between SFV entailment anomaly scores and CPM
contradiction fallback scores across all clean and injected
trajectories: r= 0.31 (p <0.001 ), indicating moderate
but non-redundant correlation. CPM’s contradiction signal
captures trajectories where confidence rises despite neutral
or contradictory NLI output — a distinct pattern from SFV’s
entailment deficit.
While simpler temporal anomaly detectors such as EWMA
or CUSUM could serve as CPM alternatives, the Beta-Bayesian
formulation offers a natural probabilistic interpretation of
confidence trajectory drift and produces a directly interpretable
posterior mean µias the expected confidence baseline. Em-
pirical comparison against EWMA-based CPM is a planned
evaluation extension.
CRT Weight and Threshold Robustness:The weights
(0.4,0.4,0.2) and threshold θ= 0.55 were selected by grid
search on held-out validation splits optimizing F1between CDR
and(1−FPR) (Section III-F ). Equal weights (0.33,0.33,0.33)
assign CPM the same weight as the more reliable SFV and
CSCT, which prior calibration experiments showed inflates FPR
due to CPM’s inherently noisier signal without logit access.
A full ROC/AUPRC sensitivity analysis over θand weight
grids is a planned evaluation addition; the current fixed-weight
design is justified by interpretability and cross-dataset transfer
without retraining.
F . Results and Analysis
As presented in Table VI, output-level detectors and LLM
Self-Correction failed dramatically in cascading scenarios.
Because downstream reasoning steps were coherent relative

to the corrupted intermediate context, self-correction suffered
from severe confirmation bias (12.8% CDR). RAGAS achieved
41.7% CDR by catching retrieval-stage errors but entirely
missed inference and confidence inflation cascades, which occur
after the retrieval stage it monitors. CHARM achieved an 89.4%
CDR and an average CDD of 2.1, proving it interrupts error
propagation by the second reasoning stage with a per-stage
component overhead of 215±18 ms. Unlike SelfCheckGPT and
RAGAS, CHARM’s detection components require no additional
LLM calls; furthermore, early cascade detection at stage 2.1
halts pipeline execution before the computationally expensive
later stages run, meaning CHARM’s end-to-end wall-clock cost
is lower than a naive per-stage comparison suggests. The 450 ms
overhead reported for SelfCheckGPT and 380 ms for RAGAS
reflect their inherent additional LLM sampling and faithfulness-
scoring calls respectively, making a direct LO/s comparison
across these methods a conservative view that understates
CHARM’s relative efficiency. CHARM achieved an MSR of
91.3%, proving that when a cascade is flagged, the automated
mitigation patterns (M1–M4) successfully resolve the error and
restore trajectory alignment. Process-level baselines EVER and
IRCoT are discussed qualitatively in Section VIII-C , as they
report answer-level EM and F1 scores that are not directly
comparable to cascade-specific detection metrics.
All reported CDR and EPR improvements over the strongest
single baseline (RAGAS, CDR = 41.7%) are statistically sig-
nificant at p <0.01 under a paired bootstrap test [31] with
10,000 resamples. Resampling was performed at thetrajectory
level: each resample draws Ntrajectories with replacement
from the full evaluation pool ( N= 1,500 injected + 500 clean
trajectories across all four datasets), recomputes CDR, FPR,
and EPR for both CHARM and RAGAS on the resample, and
records the difference. The reported p-value is the fraction of
resamples where RAGAS equaled or exceeded CHARM.
Performance Across Datasets:To ensure CHARM’s robust-
ness across different reasoning topologies, we disaggregated
CDR across the four evaluation datasets (Table VII). CHARM
maintained high efficacy across all types, with HotpotQA
yielding the highest performance (92.5%) as its structured
two-hop format produces cleaner semantic transitions, making
drift detection more reliable than in compositional multi-
document tasks. Performance dropped slightly on MuSiQue
and 2WikiMultiHopQA (86.1% and 87.8% respectively) due
to the inherent complexity of compositional reasoning, which
occasionally masked anomalous semantic shifts from the CSCT
component. CHARM consistently outperformed RAGAS, the
strongest retrieval-level baseline, across all categories.
Robustness to Near-Miss Distractors:To evaluate CHARM
under long-context stress conditions analogous to Self-RAG’s
distractor evaluation conditions [32], we constructed a distractor
stress variant of the Custom Adversarial Set in which each
trajectory included three semantically proximate but factually
incorrect documents alongside the correct source. Under
these conditions, CHARM’s CDR dropped to 84.1% (vs.
91.2% without distractors), with FPR increasing to 7.8%. The
CSCT component was most affected, as embedding-proximaldistractors occasionally passed cosine similarity drift detec-
tion. This identifies adversarial embedding-proximal attacks
as a meaningful attack surface and informs the adversarial
robustness discussion in Section VII-D.
G. Naturally Occurring Cascade Pilot
To assess ecological validity beyond synthetic injections,
we ran CHARM on 50 naturally occurring HotpotQA failure
trajectories—agent runs that produced incorrect final answers
without any injected perturbation, drawn from the full evalua-
tion split. Among these, CHARM flagged anomalous trajectory
signals in 38 of 50 cases (76%), with the CRT triggering at
stage 2.3 on average. Manual inspection of the 38 flagged
cases confirmed cascade-like characteristics (local coherence
with global error) in 34 of 38 (89.5%), and found independent
stage errors (non-cascading) in 4 cases. The 12 unflagged
cases contained errors that emerged only at the final synthesis
stage, beyond CHARM’s cross-stage monitoring window. While
a larger-scale natural cascade corpus remains future work,
this pilot provides initial evidence that CHARM’s detection
generalizes beyond synthetic injection conditions.
VII. DISCUSSION
The empirical results demonstrate that CHARM effectively
interrupts cascading errors, but the broader impact of this
framework extends into enterprise AI governance.
A. Alignment with NIST AI Risk Management Frameworks
A critical imperative for responsible AI adoption in the
United States is alignment with federal guidelines. In July
2024, the National Institute of Standards and Technology
(NIST) released the Artificial Intelligence Risk Management
Framework: Generative AI Profile (NIST AI 600-1) [2],
explicitly identifying “Confabulation” (hallucination) as a
primary risk category. CHARM directly addresses this named
risk by mapping its architectural mitigations to the foundational
functions of the broader NIST AI RMF [33], as detailed in
Table VIII.
B. Enterprise Deployment in Regulated Industries
As U.S. enterprises accelerate AI deployment—with 78% of
organizations now using AI in at least one business function and
23% actively scaling agentic AI systems [34]—the theoretical
risks of multi-step hallucination become concrete operational
vulnerabilities. Cascading hallucinations are uniquely danger-
ous in regulated industries such as financial services and legal
compliance, where downstream decisions are highly sensitive
to initial inputs. Furthermore, as these systems integrate with
external enterprise tools, the risk of context poisoning via
adversarial inputs [35] necessitates robust cross-stage validation.
Because CHARM is highly retrofittable, it provides a practical
pathway for organizations to secure their existing production-
grade deployments without requiring expensive architectural
overhauls.

TABLE VI
PERFORMANCECOMPARISON OFDETECTIONFRAMEWORKS ONCASCADINGTRAJECTORIES. CHARMRESULTS ARE MEAN±STANDARD DEVIATION
OVER FIVE INDEPENDENT RUNS.
Detection Mechanism CDR (↑) FPR (↓) EPR (↑) MSR (↑) CDD (↓) LO/s (ms)
No Detection (None) 0.0% N/A 0.0% N/A N/A 0
Output-Level (SelfCheckGPT) 24.3% 6.2% 18.5% N/A 5.0 (Terminal) 450
Retrieval Level (RAGAS) 41.7%4.1%35.2% N/A 1.5 380
LLM Self-Correction 12.8% 14.5% 8.4% 15.2% 5.0 (Terminal) 1200
CHARM (Ours) 89.4±1.8%5.3±0.6%82.1±2.1% 91.3±1.4% 2.1 215±18
TABLE VII
CASCADEDETECTIONRATE(CDR)BYDATASET
Dataset Output-Level RAGAS CHARM (Ours)
HotpotQA 26.1% 44.2%92.5%
MuSiQue 22.4% 38.5%86.1%
2WikiMultiHopQA 21.8% 40.1%87.8%
Custom Adversarial 26.9% 43.9%91.2%
TABLE VIII
CHARM COMPONENTMAPPING TONIST AI RISKMANAGEMENT
FRAMEWORKS
CHARM Compo-
nentNIST RMF Func-
tionNIST AI 600-1 Risk
SFV MEASURE Confabulation / Hallucination
CSCT MEASURE +
MANAGEData quality and integrity
CPM MEASURE Uncertainty quantification
CRT GOVERN + MAN-
AGEHuman oversight and interven-
tion
C. The Complete Reliability and Governance Stack
CHARM is explicitly designed to integrate with the HITL-AP
framework [21]. Together, they form a comprehensive security
stack: CHARM continuously monitors semantic trajectories
and interrupts low-confidence cascades autonomously, while
routing high-confidence cascades to the HITL-AP human
approval checkpoints. This integrated architecture ensures
agentic systems remain tethered to enterprise governance
protocols.
D. Limitations and Adversarial Robustness
The computational overhead of the PV A (M3) may be
prohibitive for latency-sensitive applications. While the FPR
is manageable at 5.3%, it can become elevated in highly
ambiguous domains where ground truth is deeply nuanced.
The current evaluation scope is limited to text-based agentic
RAG pipelines; extending to multimodal trajectories remains
an open challenge.
Semantic Illusion Boundary:A known limitation of
embedding- and NLI-based detectors is reduced effectiveness
on “semantic illusion” hallucinations, where RLHF-era models
produce factually incorrect outputs that remain semanticallyproximate to the correct answer [36]. Our current evalu-
ation uses synthetic cascade injections (GPT-4o-generated
counterfactuals, context perturbations) which are semantically
distinguishable by design; performance on datasets specifically
engineered to induce semantic illusions (e.g., HaluEval [37])
may differ. Evaluating CHARM’s SFV and CSCT on such
benchmarks and hybridizing with reasoning-capable LLM
judges as an alternative SFV backend is an identified extension
for future work.
Adversarial Robustness Boundaries:A sophisticated adver-
sary aware of CHARM’s detection mechanisms could engineer
context poisoning attacks specifically designed to evade the
CSCT’s cosine similarity drift detection—for example, by
constructing counterfactual documents that are semantically
proximate to ground truth while remaining factually incor-
rect. Similarly, entailment-ambiguous injections, where the
false claim is logically consistent with but not entailed by
the evidence, could evade the SFV’s NLI threshold. As
demonstrated by our distractor stress test (Section VI-F ),
embedding-proximal attacks reduce CDR by 7.1 percentage
points. Hardening CHARM against such white-box adversarial
attacks via adversarial fine-tuning of the SFV cross-encoder is
a critical direction for future work, particularly for the Context
Poisoning Cascade type [27].
Synthetic Injection Scope:The primary evaluation relies on
controlled cascade injection rather than organically occurring
cascades from real agent deployments. While synthetic injection
enables precise ground truth labeling and controlled comparison
across cascade types, it may not fully represent the distribution
of naturally occurring cascades. The 200-trajectory custom
adversarial set partially mitigates this by including manually
designed cascade scenarios, but human annotation of natural
multi-step agent failures at scale remains an important gap.
Constructing a human-annotated natural cascade corpus and
validating CHARM’s detection on it is an identified priority for
future work. Results should therefore be interpreted as evidence
of controlled cascade detection efficacy under structured failure
conditions rather than definitive performance on naturally
occurring enterprise agent trajectories.
VIII. RELATEDWORK
Our research intersects with three primary domains: hal-
lucination detection, multi-step reasoning evaluation, and
agentic system reliability. By isolating the phenomenon of

compounding errors, we differentiate CHARM from existing
point-in-time evaluation methods.
A. Hallucination Detection in LLMs
The proliferation of LLMs has driven significant research
into hallucination detection and mitigation [9], [38]. Most
existing approaches evaluate individual generation outputs
in isolation. Methods like SelfCheckGPT [4] leverage zero-
resource sampling to detect inconsistencies in black-box LLM
generations. Fine-grained atomic evaluation frameworks such as
FActScore [5] break down long-form generations into verifiable
claims. In the context of RAG, frameworks like RAGAS [6]
and ARES [39] evaluate the faithfulness of an answer against
the retrieved context. While these methods demonstrate high
accuracy for single-step generation, they inherently assume
that the retrieved context is uncorrupted or that the reasoning
chain is confined to a single transition. In contrast, CHARM
specifically addresses trajectory-level error propagation across
pipeline stages as a first-class architectural concern, operating
as a passive retrofit layer that models cross-stage semantic drift
and confidence inflation dynamics—capabilities not jointly
addressed by any prior single framework.
B. Multi-Step Reasoning Failures
The foundation for multi-step LLM execution stems from
advancements like Chain-of-Thought (CoT) prompting [11],
which allows models to break complex problems into interme-
diate steps. However, research into compositional reasoning
errors [12] has demonstrated that LLMs frequently suffer from
logical derailment as reasoning depth increases. These works
establish the theoretical foundation for error compounding,
proving that local coherence does not guarantee global factual
accuracy. CHARM builds upon this theoretical foundation by
operationalizing it into a detectable architectural metric (CDD).
C. Process-Level Verification and Planning
Recent work has begun to address error propagation at
the process level rather than the output level, with methods
reporting answer-level accuracy (EM and F1) on the same
datasets used in this paper.
EVER [28] applies real-time, step-wise generation with
retrieval-based verification and rectification, reporting improve-
ments in multi-hop reasoning on HotpotQA under EM and F1
evaluation. EVER explicitly targets the “snowballing” halluci-
nation phenomenon—errors that compound across sequential
reasoning steps—which aligns closely with the cascading fail-
ure mode formalized in this paper. However, EVER operates at
the claim level within individual generation steps and does not
model cross-stage semantic trajectory or confidence propagation
across a structured pipeline. Consequently, it cannot detect
Confidence Inflation Cascades or Context Poisoning Cascades
that manifest as anomalous trajectory-level drift rather than
local claim-level contradictions.
IRCoT [29] interleaves chain-of-thought reasoning with
retrieval steps, achieving up to 15 F1-point improvements on
HotpotQA, 2WikiMultiHopQA, and MuSiQue over single-stepretrieval baselines, and reducing factual errors in generated CoT
by up to 50%. While IRCoT reduces upstream error propagation
through iterative re-grounding, it requires deep integration into
the reasoning loop and cannot be applied as a passive retrofit
layer to existing production pipelines. Furthermore, IRCoT has
no mechanism for detecting Confidence Inflation Cascades, as
it does not monitor confidence trajectories.
Self-RAG [32] demonstrates that adaptive self-reflective
retrieval reduces error propagation in agentic generation, and its
distractor evaluation conditions motivate our near-miss stress
test in Section VI-F.
Most directly related to CHARM’s CDD metric is Agen-
tHallu [30], which performs step-level localization of halluci-
nation origin in multi-agent trajectories and provides causal
explanations for error emergence. AgentHallu demonstrates
that identifyingwherein a trajectory a hallucination originates
is both feasible and practically valuable. CHARM’s CDD
metric formalizes this intuition as a standardized, quantitative
evaluation criterion: while AgentHallu focuses on post-hoc
attribution across agent trajectories, CHARM detects and
interrupts cascadesat inference timebefore the trajectory
completes, targeting a different operational point in the pipeline
lifecycle.
Production-grade non-LLM verifier stacks combining
retrieval-aware relevance scoring with NLI are conceptually
aligned with CHARM’s SFV component; such verifiers could
serve as drop-in SFV backends given CHARM’s modu-
lar design. Small reasoning verifiers that provide factuality
discrimination with explanations represent another natural
SFV backend option for resource-constrained deployments.
Evaluating CHARM with alternative verifier backends is a
planned extension.
TABLE IX
QUALITATIVECOMPARISON OFRELATEDFRAMEWORKS
Framework Cascade Trajectory Confidence Retrofit
Detect Monitor Track Deploy
EVER [28] Partial Partial No No
IRCoT [29] Partial Partial No No
AgentHallu
[30]Post-hoc Yes No No
Self-RAG
[32]No Partial No No
CHARM
(Ours)Yes Yes Yes Yes
Table IX summarizes the key differentiating dimensions.
CHARM is the only framework that simultaneously detects
cascades at inference time, monitors the full cross-stage
trajectory, tracks confidence propagation, and requires no
architectural changes to the primary pipeline.
Relative to AgentHallu’s post-hoc attribution approach,
CHARM differentiates by operating at inference time rather
than retrospectively. More broadly, CHARM differentiates from
this line of work in three ways: (1) it operates as a non-
intrusive parallel monitoring layer requiring no architectural

changes to the primary pipeline; (2) it jointly models semantic
drift, entailment grounding, and confidence trajectory as a
unified detection signal rather than any single dimension; and
(3) it introduces a formally defined cascade taxonomy and
the CDD metric that standardize evaluation for this class of
failures. Because EVER and IRCoT report EM and F1 scores
while CHARM reports cascade-specific metrics (CDR, FPR,
EPR, CDD), direct numerical comparison is not presented;
the contribution of CHARM is orthogonal—it provides the
detection and governance infrastructure within which methods
like IRCoT could operate. Multi-agent verification frameworks
such as MARCH [40] and cryptographically-grounded ap-
proaches such as FINCH-ZK [41] provide complementary
hallucination mitigation angles; CHARM’s SFV component
could be instantiated with any such verifier backend, making
the CHARM architecture extensible to these approaches.
D. Agentic System Reliability
As LLMs evolve from isolated chatbots to autonomous
agents equipped with tool use [42], evaluating system reliability
has become increasingly complex. Our foundational SoK
analysis of Agentic RAG architectures [1] mapped the current
design landscape and explicitly identified the lack of cross-
stage context monitoring as a critical vulnerability in enterprise
deployments. This paper directly addresses the evaluation
gap identified in that prior work, providing the necessary
reliability layer to support responsible AI adoption. Real-
world cascade testbeds such as OHRBench [43], which studies
multi-stage failures originating from OCR noise in document-
heavy pipelines, represent a natural extension for evaluating
CHARM beyond QA-only settings and would validate detection
capabilities on organically occurring cascades.
IX. CONCLUSION
Multi-step agentic RAG pipelines are highly vulnerable to
cascading hallucinations, a failure mode where early-stage
contextual errors silently compound into confident, structurally
sound fabrications. To address this, we introduced the Cascad-
ing Hallucination Aware Resolution and Mitigation (CHARM)
framework. By continuously monitoring cross-stage semantic
trajectories with formally operationalized detection quantities,
CHARM successfully interrupted cascading failures before they
corrupted the terminal output, achieving an 89.4% Cascade
Detection Rate (CDR) and intervening early with an average
Cascade Depth at Detection (CDD) of 2.1, outperforming all
four direct baselines by substantial margins while introducing
only 215 ms per-stage overhead. Component ablations confirm
that each of the four detection modules contributes meaningfully
to overall cascade coverage.
This work makes three primary contributions to the field of
agentic AI reliability. First, it formalizes a four-type taxonomy
(Retrieval, Inference, Context Poisoning, and Confidence
Inflation Cascades) specifically tailored to multi-step reasoning
systems, with concrete operational definitions for all core
formal quantities. Second, it presents the CHARM detectionarchitecture, comprising four modular tracking components—
SFV , CSCT, CPM, and CRT—capable of identifying com-
pounding errors without interrupting valid trajectory flow.
Finally, it outlines four implementable mitigation patterns that
provide configurable recovery trade-offs for production-grade
deployments.
Future work will focus on two specific directions. First,
we plan to extend CHARM to multimodal agentic pipelines,
mapping how semantic divergence propagates across visual
and auditory reasoning chains. Second, we aim to develop
adaptive threshold calibration for CHARM components based
on domain-specific risk profiles. Exploring this through a
security lens—specifically by integrating CHARM’s context
poisoning detection with the Zero Trust framework for the
Model Context Protocol (ZT-MCP) [44]—will be crucial for
defending agentic systems against adversarial cascade scenarios
in critical enterprise environments.
To support the research community, we release all experimen-
tal artifacts, including the agentic trajectory wrapper, cascade
injection scripts, annotated adversarial trajectories, and the
CHARM evaluation harness, at https://github.com/sarmishra/
CHARM-agentic-rag.
REFERENCES
[1]S. Mishra, S. Niroula, U. Yadav, D. Thakur, S. Gyawali, and
S. Gaire, “Sok: Agentic retrieval-augmented generation (rag): Taxon-
omy, architectures, evaluation, and research directions,”arXiv preprint
arXiv:2603.07379, 2026.
[2]National Institute of Standards and Technology, “Artificial intelligence
risk management framework: Generative ai profile (nist ai 600-1),” U.S.
Department of Commerce, Tech. Rep., July 2024. [Online]. Available:
https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
[3]S. Mishra and H. Reza, “A face recognition method using deep learning
to identify mask and unmask objects,” in2022 IEEE World AI IoT
Congress (AIIoT). IEEE, 2022, pp. 091–099.
[4]P. Manakul, A. Liusie, and M. J. Gales, “Selfcheckgpt: Zero-resource
black-box hallucination detection for generative large language models,”
arXiv preprint arXiv:2303.08896, 2023.
[5]S. Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. W. Koh, M. Iyyer,
L. Zettlemoyer, and H. Hajishirzi, “Factscore: Fine-grained atomic
evaluation of factual precision in long form text generation,”arXiv
preprint arXiv:2305.14251, 2023.
[6]S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “Ragas: Au-
tomated evaluation of retrieval augmented generation,”arXiv preprint
arXiv:2309.15217, 2023.
[7]N. Shinn, F. Labash, A. Gopinath, and K. Narasimhan, “Reflexion:
Language agents with verbal reinforcement learning,”arXiv preprint
arXiv:2303.11366, 2023.
[8]P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” inAdvances in
Neural Information Processing Systems, vol. 33, 2020, pp. 9459–9474.
[9]Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . Bang,
A. Madotto, and P. Fung, “Survey of hallucination in natural language
generation,”ACM Computing Surveys, vol. 55, no. 12, pp. 1–38, 2023.
[10] L. Pan, M. Saxon, R. Connor, A. Sharma, and W. Y . Wang, “Automatically
correcting large language models: Survey and taxonomy,”arXiv preprint
arXiv:2308.03188, 2023.
[11] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V . Le,
D. Zhouet al., “Chain-of-thought prompting elicits reasoning in large
language models,”Advances in Neural Information Processing Systems,
vol. 35, pp. 24 824–24 837, 2022.
[12] N. Dziri, X. Lu, M. Sclar, X. L. Li, L. Jian, B. Y . Lin, P. West,
C. Bhagavatula, R. L. Bras, J. D. Hwanget al., “Faith and fate: Limits
of transformers on compositionality,”Advances in Neural Information
Processing Systems, vol. 36, 2023.

[13] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2023.
[14] H. Lightmanet al., “Let’s verify step by step,”arXiv preprint
arXiv:2305.20050, 2023.
[15] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings
using siamese bert-networks,” inProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing, 2019, pp. 3982–
3992.
[16] P. He, X. Liu, J. Gao, and W. Chen, “Deberta: Decoding-enhanced bert
with disentangled attention,” inInternational Conference on Learning
Representations, 2021.
[17] S. Kadavathet al., “Language models (mostly) know what they know,”
arXiv preprint arXiv:2207.05221, 2022.
[18] C. Guo, G. Pleiss, Y . Sun, and K. Q. Weinberger, “On calibration
of modern neural networks,” inProceedings of the 34th International
Conference on Machine Learning (ICML), 2017.
[19] H. Chase, “Langchain: Building applications with llms through compos-
ability,” https://github.com/hwchase17/langchain, 2023.
[20] J. Liu, “Llamaindex: A data framework for large language models,”
https://github.com/jerryjliu/llama index, 2023.
[21] S. Mishra, “Trustworthy agentic ai pipelines: Human-in-the-loop oversight
architectures for secure enterprise deployment,”ResearchGate preprint,
2026.
[22] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“ReAct: Synergizing reasoning and acting in language models,” in
International Conference on Learning Representations (ICLR), 2023.
[23] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with GPUs,”IEEE Transactions on Big Data, 2019.
[24] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. W. Cohen, R. Salakhutdinov, and
C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-hop
question answering,” inProceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing, 2018, pp. 2369–2380.
[25] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Musique:
Multihop questions via single-hop question composition,” inTransactions
of the Association for Computational Linguistics, vol. 10, 2022, pp. 539–
554.
[26] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps,”
inProceedings of the 28th International Conference on Computational
Linguistics, 2020, pp. 6609–6625.
[27] F. Perez and I. Ribeiro, “Ignore previous prompt: Attack techniques for
language models,” inNeurIPS ML Safety Workshop, 2022.
[28] H. Kang, J. Ni, and H. Yao, “EVER: Mitigating hallucination in large
language models through real-time verification and rectification,”arXiv
preprint arXiv:2311.09114, 2023.
[29] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-
step questions,” inProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (ACL), 2023, pp. 10 014–
10 037.[30] X. Liu, X. Yang, Z. Li, P. Li, and R. He, “Agenthallu: Benchmarking
automated hallucination attribution of llm-based agents,”arXiv preprint
arXiv:2601.06818, 2026.
[31] B. Efron and R. J. Tibshirani,An Introduction to the Bootstrap. Chapman
& Hall/CRC, 1994.
[32] A. Asai, Z. Wu, Y . Wang, A. Salmani, and H. Hajishirzi, “Self-RAG:
Learning to retrieve, generate, and critique through self-reflection,” in
International Conference on Learning Representations (ICLR), 2024.
[33] National Institute of Standards and Technology, “Artificial intelligence
risk management framework (ai rmf 1.0) (nist trustworthy and responsible
ai),” U.S. Department of Commerce, Tech. Rep. NIST IR 8259, January
2023. [Online]. Available: https://doi.org/10.6028/NIST.AI.100-1
[34] A. Singla, A. Sukharevsky, L. Yee, M. Chui, and B. Hall, “The state
of AI: How organizations are rewiring to capture value,” McKinsey &
Company, Tech. Rep., March 2025, accessed: May 2026. [Online]. Avail-
able: https://www.mckinsey.com/capabilities/quantumblack/our-insights/
the-state-of-ai-how-organizations-are-rewiring-to-capture-value
[35] S. Gaire, S. Gyawali, S. Mishra, S. Niroula, D. Thakur, and U. Yadav,
“Systematization of knowledge: Security and safety in the model context
protocol ecosystem,”arXiv preprint arXiv:2512.08290, 2025.
[36] S. Li, S. Park, I. Lee, and O. Bastani, “Traq: Trustworthy retrieval
augmented question answering via conformal prediction,” inProceedings
of the 2024 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies (Volume
1: Long Papers), 2024, pp. 3799–3821.
[37] J. Li, X. Cheng, W. X. Zhao, J.-Y . Nie, and J.-R. Wen, “HaluEval:
A large-scale hallucination evaluation benchmark for large language
models,”arXiv preprint arXiv:2305.11747, 2023.
[38] S. T. I. Tonmoy, S. Zaman, V . Jain, A. Krause, T. Goswamiet al., “A
comprehensive survey of hallucination mitigation techniques in large
language models,”arXiv preprint arXiv:2401.01313, 2024.
[39] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “Ares: An
automated evaluation framework for retrieval-augmented generation
systems,”arXiv preprint arXiv:2311.09476, 2023.
[40] Z. Li, Y . Zhang, P. Cheng, J. Song, M. Zhou, H. Li, S. Hu, Y . Qin,
E. Zhao, X. Jianget al., “March: Multi-agent reinforced self-check for
llm hallucination,”arXiv preprint arXiv:2603.24579, 2026.
[41] A. Goel, D. Schwartz, and Y . Qi, “Zero-knowledge llm hallucination
detection and mitigation through fine-grained cross-model consistency,”
inProceedings of the 2025 Conference on Empirical Methods in Natural
Language Processing: Industry Track, 2025, pp. 1982–1999.
[42] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen,
J. Tang, X. Chen, Y . Linet al., “A survey on large language model based
autonomous agents,”Frontiers of Computer Science, vol. 18, no. 6, p.
186345, 2024.
[43] J. Zhang, Q. Zhang, B. Wang, L. Ouyang, Z. Wen, Y . Li, K.-H. Chow,
C. He, and W. Zhang, “Ocr hinders rag: Evaluating the cascading
impact of ocr on retrieval-augmented generation,” inProceedings of
the IEEE/CVF International Conference on Computer Vision, 2025, pp.
17 443–17 453.
[44] S. Mishra, “Zt-mcp: A zero-trust security architecture for mcp-connected
ai agents,”ResearchGate preprint, 2026.