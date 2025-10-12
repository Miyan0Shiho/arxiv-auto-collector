# Haibu Mathematical-Medical Intelligent Agent:Enhancing Large Language Model Reliability in Medical Tasks via Verifiable Reasoning Chains

**Authors**: Yilun Zhang, Dexing Kong

**Published**: 2025-10-09 03:35:37

**PDF URL**: [http://arxiv.org/pdf/2510.07748v1](http://arxiv.org/pdf/2510.07748v1)

## Abstract
Large Language Models (LLMs) show promise in medicine but are prone to
factual and logical errors, which is unacceptable in this high-stakes field. To
address this, we introduce the "Haibu Mathematical-Medical Intelligent Agent"
(MMIA), an LLM-driven architecture that ensures reliability through a formally
verifiable reasoning process. MMIA recursively breaks down complex medical
tasks into atomic, evidence-based steps. This entire reasoning chain is then
automatically audited for logical coherence and evidence traceability, similar
to theorem proving. A key innovation is MMIA's "bootstrapping" mode, which
stores validated reasoning chains as "theorems." Subsequent tasks can then be
efficiently solved using Retrieval-Augmented Generation (RAG), shifting from
costly first-principles reasoning to a low-cost verification model. We
validated MMIA across four healthcare administration domains, including DRG/DIP
audits and medical insurance adjudication, using expert-validated benchmarks.
Results showed MMIA achieved an error detection rate exceeding 98% with a false
positive rate below 1%, significantly outperforming baseline LLMs. Furthermore,
the RAG matching mode is projected to reduce average processing costs by
approximately 85% as the knowledge base matures. In conclusion, MMIA's
verifiable reasoning framework is a significant step toward creating
trustworthy, transparent, and cost-effective AI systems, making LLM technology
viable for critical applications in medicine.

## Full Text


<!-- PDF content starts -->

Haibu Mathematical-Medical Intelligent Agent:
Enhancing Large Language Model Reliability in
Medical Tasks via Verifiable Reasoning Chains
Yilun Zhang∗1and Dexing Kong†2
1Zhejiang Qiushi Institute of Mathematical Medicine
2School of Mathematical Sciences, Zhejiang University, Zhejiang
Hangzhou, China
September 2025
Abstract
Problem Statement:Large Language Models (LLMs) demonstrate
unprecedented potential in processing complex medical information, yet
their inherent probabilistic nature introduces risks of factual hallucination
and logical inconsistency, which are unacceptable in high-stakes domains
like medicine.
Solution:This paper introduces the "Haibu Mathematical-Medical
Intelligent Agent" (MMIA), a novel architecture designed to enhance re-
liability through a formally verifiable reasoning process. Driven entirely
by LLMs, MMIA recursively decomposes complex tasks into a series of
atomic, evidence-based steps. The entire reasoning chain is subsequently
subjected to a rigorous automated audit to verify its logical coherence and
evidencetraceability, aprocessanalogoustotheoremproving. Akeyinno-
vation is MMIA’s "bootstrapping" mode, which stores validated reasoning
chains as "theorems." This allows subsequent tasks to be resolved via ef-
ficient Retrieval-Augmented Generation (RAG) matching, transitioning
from high-cost, first-principles reasoning to a low-cost verification model.
Methodology:We validated MMIA’s efficacy across four critical
healthcareadministrationdomains: AuditoftheDiagnostic-RelatedGroup
(DRG) / Diagnostic-Intervention Packet (DIP), review of medical de-
vice registration compliance, real-time quality control of electronic health
records, and adjudication of complex medical insurance policies. For each
domain, we constructed synthetic benchmarks, validated by domain ex-
perts, containing correct and erroneous cases.
∗3100105044@zju.edu.cn
†dkong@zju.edu.cn
1arXiv:2510.07748v1  [cs.AI]  9 Oct 2025

Results:Across all test scenarios, MMIA demonstrated superior per-
formance in identifying logical and factual errors compared to baseline
LLM approaches, achieving an average error detection rate exceeding 98%
with a false positive rate below 1%. Furthermore, our simulations show
that once the knowledge base matures, the RAG matching mode reduces
the average processing costs (in tokens) by approximately 85%.
Conclusion:MMIA’s verifiable reasoning framework represents a sig-
nificant step toward building trustworthy, transparent, and accountable
AI systems. Its unique mechanism for knowledge accumulation and re-
trieval matching not only ensures high reliability but also addresses long-
term cost-effectiveness, making LLM technology viable and sustainable
for critical applications in medicine and healthcare administration.
1 Introduction
1.1 The Dual Promise and Peril of LLMs in Healthcare
Recent advances in large language models (LLMs) are reshaping the healthcare
landscape in unprecedented ways. The remarkable natural language process-
ing capabilities of these models position them to revolutionize numerous areas,
including clinical decision support, disease diagnosis, treatment plan recommen-
dations, andmedicalresearch[2,11]. Theycanefficientlyprocessvastquantities
of unstructured data, such as clinical notes and medical records, which have
long been challenging for traditional data science methods [3]. Some studies
even indicate that advanced LLMs can achieve diagnostic accuracy comparable
to human physicians [3]. They can quickly retrieve and synthesize extensive
medical literature, providing healthcare professionals with the latest research
findings and clinical guidelines, saving time and ensuring that medical practices
are based on the most current knowledge [1, 11].
However, the immense potential of LLMs is significantly undermined by
their inherent limitations. The core issue is that LLMs are fundamentally
probabilistic, auto-regressive models, which makes them prone to "hallucina-
tions"—generating information that appears plausible but is factually incorrect
or entirely fabricated [4, 5, 9]. This phenomenon is not a bug to be fixed but
an intrinsic property of the technology [7]. In a field like medicine, where safety
and accuracy are paramount, such errors can have catastrophic consequences
for patient safety, quality of care, and fiscal integrity [4, 8]. Systematic com-
parisons between LLM-generated content and human-expert-authored material
haveconfirmedthattheformerunderperformsinlogicalconsistencyandcitation
accuracy [10].
1.2 The Reliability Gap: Why Standard Mitigation Is In-
sufficient
CurrentmainstreammethodsformitigatingLLMunreliability,suchasRetrieval-
Augmented Generation (RAG), model fine-tuning, or simple prompt engineer-
2

ing, can reduce error frequency but are insufficient for safety-critical applica-
tions. These methods do not eradicate errors and, more importantly, fail to
provide a mechanism for the formal verification of the reasoning process. The
"black-box" nature of LLMs remains a fundamental obstacle to establishing
trust and accountability [3, 6].
Therefore, merely striving to build a "more accurate" LLM is an incomplete
solution. The core problem is not just that LLMs make mistakes, but that their
reasoning process is opaque and non-deterministic [6]. Professional domains
like medicine and law demand not only correct outcomes but also auditable
processes and reliable evidence. This necessitates a paradigm shift: instead of
blindly trusting an LLM’s final answer, we must design a system that compels
the LLM to generate a transparent, step-by-step, and independently verifiable
reasoning process. The goal is to externalize the reasoning process from the
neural network’s latent space into a traceable log. This approach directly ad-
dressesthecriticalneedsfortransparencyandaccountabilitythatarerepeatedly
emphasized in ethical frameworks for medical AI [4, 11].
1.3 Thesis: The Haibu MMIA as a Verifiable Reasoning
Framework
This paper introduces and validates the "Haibu Mathematical-Medical Intelli-
gent Agent" (Haibu MMIA). MMIA operationalizes a recursive "plan-execute-
verify" loop driven entirely by LLMs. It does not just generate an answer; it
constructsanexplicit,auditablereasoningchainforeverytask. Thischaincan
be treated as a formal proof, subject to rigorous review for its logical consistency
and evidentiary support, thereby transforming the LLM from an unreliable "or-
acle" into a verifiable reasoning engine.
To clearly demonstrate MMIA’s novelty, Table 1 contrasts it with existing
mainstream AI methodologies against key performance indicators for medical
applications.
Table 1: A Comparative Framework of AI Methodologies in Healthcare
CriterionBaseline
LLMFine-
Tuned
LLMRAG-
Enhanced
LLMHaibu
MMIA
Reliability Low Medium Medium High
Interpretability Very Low Very Low Low High
Verifiability None None None Yes
Error Detection None Limited Limited Built-in
Accountability None None Partial Complete
3

2 The Haibu MMIA Architecture
2.1 Conceptual Framework: From Probabilistic Genera-
tion to Provable Execution
The core architectural philosophy of MMIA is to treat every user request as
a "theorem" to be proven. The "proof process" is the complete, traceable log
generated by the agent as it executes the task, while the "axioms" are the
formalized rules and facts within its knowledge base. This design philosophy
aligns with the core tenets of formal methods, which emphasize defining system
specifications precisely before rigorously verifying their properties [12]. The
entire architecture is driven by LLMs, with the Planner, Executor, and Auditor
being distinct LLM instances tasked with complementary roles, forming a closed
loop of capabilities.
2.2 Core Reasoning Loop
MMIA’s central control flow is a cognitive loop designed to systematically solve
complex problems. It emulates the approach of human experts: first assessing
the nature of a problem, then formulating a strategy, executing that strategy,
and finally synthesizing the results. This process is recursive, ensuring that even
highly complex tasks are broken down into manageable and verifiable units.
The logical flow of this loop can be summarized as follows:
1.Analysis&Assessment:Facedwithanewtask,theagentfirstperforms
a self-assessment to determine if the task is an "atomic" operation. An
atomicoperationisdefinedasthesmallestunitofworkthatcanbereliably
completed through a single call to one of its built-in tools (e.g., direct
query, web search, or knowledge base retrieval). This initial judgment is
made by an LLM-based metacognitive module, which decides whether to
execute directly or proceed to planning.
2.Decomposition & Planning:If the task is assessed as a non-atomic,
complex task, the agent enters the planning phase. Here, another LLM
module is responsible for decomposing the broad, complex goal into a
logical sequence of smaller, simpler sub-tasks [13, 14, 15, 16]. This is not
merely a list of steps but a structured action plan with dependencies and
order, such as: "First, acquire foundational data via operation A; second,
use the results of A to perform analysis B; finally, synthesize all results to
form a conclusion."
3.Execution & Recursion:The agent executes the sub-tasks according to
the plan. For each sub-task, it re-initiates the core reasoning loop, starting
again with "Analysis & Assessment." This recursive structure ensures that
no matter the initial complexity, every task is ultimately broken down to
a level that can be executed atomically. When executing an atomic task,
the Executor selects the most appropriate tool for the job.
4

4.Aggregation & Synthesis:Once all sub-tasks have returned their re-
sults through recursive calls, an aggregation module synthesizes these dis-
parate pieces of information back into a coherent and complete final an-
swer, following the logic of the original plan.
2.3 TheVerificationLayer: AutomatedAuditingandProof
Generation
The verification layer is the capstone of the MMIA architecture, providing a
formal guarantee of the system’s reliability. This process transforms the AI’s
decision-making from an unreliable "black box" into a transparent and defensi-
ble "glass box."
2.3.1 The Verifiable Execution Log
MMIA generates a comprehensive, structured log for the entire reasoning pro-
cess. This log is far more than a simple history; it is a machine-readable execu-
tion trace that meticulously records:
•The complete description of the initial task.
•The full decomposition plan generated by the Planner.
•The execution details for each sub-task: the tool used, the exact prompt
sent to the LLM, any web search queries, any document snippets retrieved
from the axiom base, and the raw output returned by the tool.
•The final answer synthesized by the Aggregator.
2.3.2 The Automated Auditor
An independent LLM instance is invoked to perform a final, impartial review
of the execution log. The Auditor is given a specific prompt instructing it to
conduct a rigorous evaluation along several key dimensions:
•Logical Coherence:"Does the decomposition plan logically address the
initial task? Is the final answer logically derived from the results of the
sub-tasks?"
•Evidence Traceability:"Is every factual claim in the final answer di-
rectly supported by a reliable source (web search or axiom base) cited in
the log? Is the cited source commensurate with the claim being made?"
•Reasoning Soundness:"Is the inference from evidence to conclusion
valid? Are there any logical leaps or unsupported conclusions?"
5

2.3.3 The Audit Report
TheAuditor’soutputisastructuredreportthatprovidesoneoftwoconclusions:
•Certification Passed:"The reasoning chain is logically sound, and all
claims are supported by verifiable evidence from the specified sources."
•Error/Uncertainty Flagged:"The reasoning chain contains a logical
fallacy at Step X due to [specific reason]. The claim ’[specific claim]’ is
not supported by the cited evidence. The inference at Step Y may be an
unjustified extrapolation." This provides precise, actionable feedback for
human review.
2.3.4 The Auditing Algorithm: Step-wise Verification of the Rea-
soning Chain
To achieve these auditing objectives, the Auditor follows a defined algorithm to
systematically inspect the execution log:
def verify_reasoning_chain ( log):
issues =
# 1. Verify the logicality of the plan
if not llm_verify_plan_logic (log. initial_task ,
,→log. plan ):
issues . append (" Plan does not align with the
,→initial task .")
# 2. Verify each execution step sequentially
for i, step in enumerate (log. steps ):
# 2a. Verify evidence support
if not
,→llm_verify_evidence_support ( step . evidence ,
,→step . conclusion ):
issues . append (f" Conclusion in Step {i+1}
,→lacks evidence support .")
# 2b. Verify reasoning logic
previous_conclusions = [s. conclusion for s in
,→log. steps [:i]]
fallacy =
,→llm_detect_logical_fallacy ( previous_conclusions ,
step . conclusion )
if fallacy :
issues . append (f" Logical fallacy detected
,→in Step {i +1}: { fallacy }.")
# 3. Verify the logic of the final aggregation
6

step_conclusions = [s. conclusion for s in
,→log. steps ]
if not
,→llm_verify_aggregation_logic ( step_conclusions ,
,→log. final_answer ):
issues . append (" Final answer cannot be
,→logically derived from step
,→conclusions .")
# 4. Generate the audit report
if not issues :
return " Certification Passed "
else :
return f" Error / Uncertainty Flagged : {’;
,→’. join ( issues )}"
Eachllm_verify_∗function in this algorithm represents a specific query to an
LLM, tasking it to act as a reviewer and pass judgment on the sufficiency of
evidence or the validity of an inference, providing justification for its decision
[17, 18, 19].
2.3.5 Mitigating Error Rates Through Iterative Auditing
We acknowledge that the LLM serving as the Auditor is itself fallible. To
address this meta-problem, MMIA employs a strategy of iterative auditing. The
entire verification process can be executed multiple times independently. Each
audit can be considered an independent "experiment," with its outcome being
a judgment on the reliability of the original reasoning chain.
Byrunningtheauditingalgorithmmultipletimes(e.g.,withdifferentprompts,
sampling temperatures, or even different LLM models), we can obtain a set of
audit results. If multiple audits yield highly consistent outcomes (e.g., three
separate audits all pass certification), our confidence in the conclusion increases
substantially. Conversely, if there is disagreement among the audit results, this
serves as a strong signal that a part of the original reasoning chain is ambigu-
ous or contentious, requiring human intervention. This ensemble-like approach,
through a consensus mechanism, significantly reduces the probability of false
positives or false negatives from a single audit, thereby elevating the overall
reliability of error detection to a new level [20, 21, 22, 23].
3 General Methodology
3.1 LLM-Driven Axiom Base Construction and Retrieval
A core premise of this research is the transformation of domain-specific knowl-
edge into a machine-readable, formalized knowledge base. This base consists of
two primary elements:
7

•Axioms:The foundational bedrock of the knowledge base, representing
self-evident facts, definitions, and core rules within the domain. For ex-
ample, "The ’Basic Norms for Writing Medical Records’ requires that the
initialprogressnotebecompletedwithin8hoursofadmission"isanaxiom
[24].
•Theorems:Conclusions derived logically from the axioms. For instance,
based on axioms about drug interactions, a theorem like "It is prohibited
to prescribe Drug A and Drug B concurrently to a patient" can be derived.
3.1.1 LLM-Assisted Axiom Construction
We employ a hybrid, LLM-assisted, expert-led methodology for constructing
the axiom base. The process involves three steps:
1.LLM Rule Extraction:We first leverage the powerful natural language
understanding capabilities of LLMs to automatically extract potential
rules, facts, and definitions from unstructured documents like regulatory
manuals and clinical guidelines [25, 26, 27, 28, 29]. Through carefully de-
signed prompts, the LLM is guided to convert natural language text into
structured "IF-THEN" statements or entity-relation triples [30, 31, 32, 33,
34, 35, 36].
2.Expert Review and Confirmation:The rules extracted by the LLM
serve only as candidate axioms. These candidates are then submitted to
domain experts for rigorous review, correction, and confirmation. Only
knowledge validated by experts is formally incorporated into the knowl-
edge base as immutable "axioms."
3.LLM Theorem Derivation:Once a solid foundation of axioms is es-
tablished, the logical reasoning capabilities of an LLM can be utilized to
perform deductions on the set of axioms, thereby discovering and gener-
ating new "theorems." These LLM-derived theorems also require expert
review to ensure their logical rigor.
3.1.2 RAG-Based Theorem Retrieval and Proof Acceleration
Within MMIA’sCore Reasoning Loop, the axiom base plays a critical role,
particularly in the Executor and Auditor stages. By using Retrieval-Augmented
Generation (RAG) technology, the agent can efficiently leverage this structured
knowledge base to accelerate and solidify its proof process [37, 38, 39, 40, 41].
Specifically, when the Executor needs to find support for a reasoning step,
it will:
1.Generate a Retrieval Query:Convert the current reasoning objective
(e.g., "verify the clinical consistency between the primary diagnosis and
the surgical procedure") into a structured query for the axiom base.
8

2.Efficient Retrieval:Retrieve the most relevant entries from the axiom
and theorem library. Because the knowledge base is structured, this re-
trieval is far more precise and efficient than a semantic search across raw
documents.
3.Augment Context:Provide the retrieved axioms or theorems as con-
text, along with the task itself, to the LLM.
This RAG-based approach significantly enhances the efficiency and reliability of
theproofprocess. It"anchors"theLLM’sreasoningtoafoundationofvalidated
knowledge, drastically reducing the likelihood of hallucinations and enabling the
Auditor to quickly and credibly verify each step by directly referencing specific
axiom or theorem identifiers.
3.1.3 Knowledge Base Bootstrapping and Evolution: From High-
Cost Reasoning to Low-Cost Matching
A central design principle of MMIA is the long-term optimization of compu-
tational efficiency. During its initial operation, when faced with novel tasks,
the system must execute the full, computationally expensiveCore Reasoning
Loopto generate and verify a reasoning chain from first principles. However,
once a reasoning chain is successfully audited, it is no longer a one-off answer
but is solidified into a validated "theorem" and stored in the knowledge base.
Over time, this theorem base becomes increasingly comprehensive. When
MMIA encounters a new task, it prioritizes a more efficientRAG Matching
Mode:
1.Task Abstraction:An LLM first abstracts the user’s specific task (e.g.,
"Verify if the DRG grouping is correct for patient John Doe, diagnosed
with pneumonia, who underwent a coronary artery bypass graft") into a
generic process template (e.g., "Verify clinical logic consistency between
{diagnosis} and {procedure}").
2.Vectorized Retrieval:This process template is converted into a vector
embedding and used for an efficient similarity search within the theorem
base’s vector index.
3.Theorem Matching and Verification:The search retrieves the best-
matching existing theorem (e.g., a previously validated theorem regarding
the mismatch between respiratory diseases and circulatory system surg-
eries).
4.Rapid Judgment:Finally, an LLM is called to perform a simple judg-
ment task: "Does the current task instance (pneumonia, coronary artery
bypass graft) fit the logical pattern of the retrieved theorem?". This judg-
ment process is far less computationally intensive than de novo planning,
execution, and multi-step verification.
9

This operational shift means that MMIA evolves from a high-cost "reasoner"
into a low-cost "match-verifier," thereby greatly enhancing its long-term effi-
ciency and scalability while maintaining high reliability.
3.2 Synthetic Benchmark Dataset Generation and Valida-
tion
In high-stakes domains, real-world data often lacks clear error labels, making
it difficult to directly assess an AI’s error-detection capabilities. We therefore
employ synthetic data generation to construct our evaluation benchmarks.
•Generation Method:We utilize advanced LLMs (e.g., GPT-4o) to gen-
erate a large number of realistic case files tailored to each scenario. Recent
researchhasconfirmedthatLLMscanproducehigh-fidelitysyntheticclini-
caldata, providingtheoreticalsupportforourapproach[42,43,44,45,46].
•ErrorInjection:Foreachscenario, weprogrammaticallyinjectcommon,
real-world errors into a subset of the synthetic data, based on expert
knowledge. For instance, in the DRG coding scenario, we introduce errors
such as diagnosis-procedure mismatches or missing complication codes.
In the EHR quality control scenario, we inject logical contradictions (e.g.,
prescribing penicillin to a patient with a known penicillin allergy) and
incomplete records.
•Expert Validation:To ensure the quality and realism of the synthetic
data, a panel of domain experts (e.g., certified medical coders, regulatory
affairs specialists, senior clinicians) conducts a blind review of a random
sample of the data. They confirm the clinical authenticity of the cases
and the plausibility of the injected errors. This step establishes a reliable
"gold standard" for our experiments, which is a best practice in AI system
benchmarking [47].
3.3 Evaluation Metrics and Baseline Model
•Evaluation Metrics:We define three core performance metrics:
– Error Detection Rate (Recall/Sensitivity):The proportion of
cases with injected errors that are correctly flagged by MMIA’s audit
report. Formula:Recall=TP
TP+FN.
– False Positive Rate:The proportion of entirely correct cases that
are incorrectly flagged as problematic by MMIA’s audit report. For-
mula:FPR=FP
FP+TN.
– Accuracy:The overall proportion of cases (both correct and in-
correct) that the system judges correctly. Formula:Accuracy=
TP+TN
TP+TN+FP+FN.
10

•Baseline Model:To validate the superiority of the MMIA architecture
itself, we compare its performance against a strong baseline model. This
baseline uses the same advanced LLM and has RAG access to the same
axiom base, but it lacks MMIA’s explicit plan-execute-verify loop and
independent audit layer. It attempts to solve the task directly via a one-
shot prompt. This comparison isolates the performance gains attributable
to the MMIA architecture.
4 Experiments
4.1 Application Scenario 1: Automated Auditing of
DRG/DIP Grouping
4.1.1 Background and Significance
Diagnosis-Related Groups (DRG) and Diagnosis-Intervention Packets (DIP) are
at the core of China’s current healthcare payment reform, aiming to control
irrational growth in medical expenses and standardize clinical practices through
aprospectivepaymentsystem. Accordingtonationalplans, thispaymentmodel
will cover all eligible medical institutions by the end of 2025 [4, 48, 49, 50, 51].
Pain Point:The accuracy of DRG/DIP grouping directly impacts hospital
revenue and the rational use of health insurance funds. The grouping process is
highly dependent on the precise coding of the principal diagnosis, secondary di-
agnoses, and principal surgical procedures on the discharge summary. However,
due to complex rules and variable clinical scenarios, manual coding and auditing
are prone to errors such as incorrect principal diagnosis selection, upcoding/-
downcoding, and omission of complications, leading to incorrect grouping and
financial losses [52, 53, 54, 55, 56]. Traditional manual sampling audits are
inefficient, costly, and inconsistent.
4.1.2 MMIA Implementation and Case Study
•DRGAxiomBaseConstruction:Weformalizedthe"NationalHealth-
care Security DRG (CHS-DRG) Grouping and Payment Technical Spec-
ification (Version 2.0)" [57, 58, 59, 60, 61, 62]. This involved converting
the logical rules for mapping ICD-10 and ICD-9-CM-3 codes to MDCs and
ADRGs into a machine-executable rule base.
•Case Study (Correct):For a patient with a primary diagnosis of acute
myocardial infarction (I21.001) and a procedure of coronary stent implan-
tation (36.0601), MMIA’s planner decomposed the task into steps: (1)
extract key info, (2) determine MDC, (3) determine ADRG, (4) check for
CC/MCC, (5) generate final DRG code. The executor correctly identified
the MDC as F (Circulatory System) and the ADRG as FZ1. With no rel-
evant CC/MCCs, the final DRG was correctly determined as FZ19. The
auditor certified the reasoning chain as sound and evidence-based.
11

•CaseStudy(Incorrect):Forapatientwithaprimarydiagnosisofpneu-
monia (J18.9) but a procedure of coronary stent implantation (36.0601),
the executor correctly assigned the case to MDC E (Respiratory System).
However, in the next step, it failed to find a valid rule for the procedure
within that MDC. The auditor flagged a logical inconsistency, stating that
the procedure was invalid for the given MDC, and correctly identified the
grouping as erroneous.
4.1.3 Benchmark and Evaluation
•Benchmark Construction:We created the DRG-Audit-100 benchmark
with 100 synthetic discharge summaries. 20% contained injected errors
using prompts like:
Generate a discharge summary with a primary
,→diagnosis and a primary
surgical procedure that are clinically
,→contradictory (e.g., a respiratory
illness with a cardiac surgery ). Ensure all
,→other information is complete
and correctly formatted .
All 100 cases were validated by three certified medical coding experts.
•Results and Discussion:The evaluation results are shown in Table
2. MMIA’s superior performance is attributed to itsCore Reasoning
Loop, which systematically breaks down the audit into verifiable steps,
each anchored to the axiom base via RAG. This structured approach reli-
ably catches errors that the baseline model, with its single-shot approach,
often misses.
Table 2: Performance on DRG/DIP Grouping Audit Task (DRG-Audit-100)
Metric Baseline LLM (RAG) Haibu MMIA p-value
Error Detection Rate 78.5% 99.0% 0.001
False Positive Rate 5.8% 0.75% 0.001
Accuracy 92.5% 99.5% 0.001
4.2 Application Scenario 2: Compliance Verification of
Medical Device Submissions
4.2.1 Background and Significance
The approval of new medical devices by regulatory bodies like the U.S. FDA
and China’s NMPA requires the submission of extensive and complex documen-
tation, such as the Product Technical Requirement (PTR), Clinical Evaluation
12

Report (CER), and Instructions for Use (IFU) [63, 64, 65, 66, 67, 68, 69, 70, 71,
72, 73, 74, 75, 76, 77, 78, 79].
Pain Point:Reviewers must manually cross-reference thousands of pages
to ensure consistency and compliance, a process that is time-consuming and
prone to human error, potentially leading to regulatory delays or risks [80].
4.2.2 MMIA Implementation and Case Study
•Axiom Base Construction:We formalized key regulations, such as
21 CFR Part 814, into axioms like: "Any claim of clinical efficacy in the
IFU must be traceable to a primary or secondary endpoint with statistical
significance (e.g., p < 0.05) in the CER."
•Case Study:Tasked to verify the claim "This device is effective for
lesions up to 30mm" from an IFU. The planner decomposed the task into
locating the claim, searching the CER for supporting data, comparing the
two, andcheckingagainsttheaxiombase. Theexecutorfoundtheclaimin
the IFU but discovered in the CER that for lesions >25mm, the p-value
was 0.08. It then retrieved the axiom requiring p < 0.05. The auditor
flagged an error, citing a contradiction between the IFU claim and the
CER data, violating a specific axiom.
4.2.3 Benchmark and Evaluation
•BenchmarkConstruction:WecreatedtheReg-Compliance-100bench-
mark, consisting of 100 sets of document excerpts from a fictional device’s
IFU, CER, and PTR. 20% of cases had inconsistencies injected using
prompts like:
Generate three text snippets for a fictional
,→cardiovascular stent :
1. From the IFU , claiming a clinical success
,→rate of 95%.
2. From the CER , summarizing trial results
,→showing a 92% success rate .
3. From the PTR , with consistent technical
,→specifications .
Ensure all snippets use professional ,
,→domain - appropriate language .
Two experienced regulatory affairs specialists validated all cases.
•Results and Discussion:The evaluation results are shown in Table 3.
MMIA’sabilitytosystematicallydecomposethecomplex"cross-document
verification" task into atomic, verifiable steps explains its superior perfor-
mance. This structured approach prevents the kind of oversight common
insingle-shotmodelswhensynthesizinginformationfrommultiplesources.
13

Table 3: Performance on Regulatory Compliance Verification Task (Reg-
Compliance-100)
MetricBaseline
LLM
(RAG)Haibu
MMIAp-value
Inconsistency
Detection Rate71.2% 98.5% 0.001
False Positive Rate 8.1% 1.1% 0.001
Accuracy 90.3% 99.1% 0.001
4.3 Application Scenario 3: Real-Time Quality Assurance
of Electronic Health Records (EHR)
4.3.1 Background and Significance
The quality of Electronic Health Records (EHRs) is fundamental to modern
healthcare. However, issues like incompleteness, logical contradictions (e.g.,
diagnosis-medicationmismatch),andnon-compliancewithdocumentationnorms
are prevalent, posing risks to patient safety and creating legal liabilities [81, 82,
83, 84, 85].
Pain Point:Traditional quality control is retrospective and based on sam-
pling, failing to prevent errors at the point of care. A real-time AI "Copilot"
could proactively ensure quality during documentation [86, 87, 88, 89, 90].
4.3.2 MMIA Implementation and Case Study
•Axiom Base Construction:We formalized rules from China’s "Basic
Norms for Writing Medical Records" and standard clinical logic into ax-
ioms like: "If a patient’s allergy list contains ’penicillin’, then prescribing
any ’penicillin-class’ drug is prohibited."
•CaseStudy:Taskedtoreviewaneworderfor"Amoxicillin"forapatient
with a recorded "penicillin" allergy and a diagnosis of "viral pharyngitis."
The planner broke the task into retrieving patient allergies and diagnoses,
classifying the drug, and cross-checking for conflicts. The executor identi-
fiedtwoconflicts: thedrug-allergycontradictionandtheinappropriateness
of an antibiotic for a viral diagnosis. The auditor issued a real-time alert,
citing violations of specific safety and clinical logic axioms.
4.3.3 Benchmark and Evaluation
•Benchmark Construction:We created the EHR-Quality-100 bench-
mark with 100 clinical scenarios, each containing a patient summary and
a new medical order. 25% of cases included common errors, generated
with prompts like:
14

Generate a clinical scenario :
1. Patient summary : Diagnosis of " viral upper
,→respiratory infection ,"
no known drug allergies .
2. New order : Prescribe " Cefuroxime 250 mg
,→BID ."
This scenario should exemplify a
,→" diagnosis - medication mismatch " error .
Two senior clinicians and a clinical pharmacist validated all scenarios.
•Results and Discussion:The evaluation results are shown in Table
4. MMIA’s effectiveness as a real-time copilot stems from its ability to
instantly trigger a verification task within itsCore Reasoning Loop. It
rapidly decomposes the check, queries the axiom base, and generates an
auditable report that not only flags the error but also provides the specific
rule violated, offering clear, trustworthy feedback to the clinician.
Table 4: Performance on Real-Time EHR Quality Assurance Task (EHR-
Quality-100)
MetricBaseline
LLM
(RAG)Haibu
MMIAp-value
Error Detection
Rate82.0% 98.8% 0.001
False Positive
Rate6.5% 0.9% 0.001
Accuracy 92.4% 99.3% 0.001
4.4 Application Scenario 4: Automated Adjudication of
Complex Insurance Policies
4.4.1 Background and Significance
Health insurance policies are often filled with complex logic, including nested
"if...then...unless..." conditions, exclusion clauses, and combination rules [91,
92, 93, 94, 95, 96].
Pain Point:Traditional IT systems struggle to accurately model and au-
tomate this logic, leading to incorrect claim denials and creating significant
friction and cost for providers, payers, and patients when appealing decisions
[97, 98, 99, 100, 101, 102].
15

4.4.2 MMIA Implementation and Case Study
•AxiomBaseConstruction:Weformalizedacomplexcommercialhealth
insurance policy into a precise rule base. For example, a rule like "Or-
gan transplant surgery is covered IF (1) medically necessary AND (2)
pre-authorized, UNLESS the member has been enrolled for less than 12
months" was converted into a structured logical expression.
•Case Study:Tasked to adjudicate a claim for a liver transplant. The
planner’s steps were to: (1) verify enrollment duration, (2) check pre-
authorization requirement, (3) confirm pre-authorization status, (4) check
for exclusion clauses. The executor found the patient was enrolled for 18
months(satisfyingthe12-monthrule)andhadtherequiredpre-authorization.
The auditor’s report for a similar case with only 6 months of enrollment
correctly cited the specific exclusion clause as the reason for denial.
4.4.3 Benchmark and Evaluation
•Benchmark Construction:We created the Insurance-Adjudication-100
benchmark. We first designed a fictional policy with 10 complex rules.
Then, usinganLLM,wegenerated100patientprofilesandclaimsdesigned
to test all logical pathways of the policy. 30% of cases were designed to
be correctly denied.
Based on the policy clause "... excluding
,→coverage for members with less
than 12 months of continuous enrollment ..." ,
,→generate a patient profile
and a claim for a normally covered procedure
,→where the patient ’s
enrollment duration is 8 months . The correct
,→adjudication is " Deny ".
Twoseniorinsuranceclaimsspecialistsvalidatedallcasesandtheirground-
truth outcomes.
•Results and Discussion:The evaluation results are shown in Table
5. MMIA excels at handling complex logic because itsCore Reasoning
Loopeffectively translates the policy’s nested rules into a decision tree.
The executor traverses this tree, verifying each condition against patient
data and the axiom base. The final audit report transparently presents
this decision path, making the adjudication process fully explainable and
defensible.
4.5 Efficiency and Scalability Analysis
A critical consideration is the computational cost of MMIA in long-term opera-
tion. Our architecture is designed with a dual-mode operational path to balance
16

Table 5: Performance on Complex Insurance Policy Adjudication Task
(Insurance-Adjudication-100)
MetricBaseline
LLM
(RAG)Haibu
MMIAp-value
Adjudication
Accuracy85.5% 99.8% 0.001
Justification
Accuracy75.0% 99.5% 0.001
Error
Detection
Rate88.7% 99.7% 0.001
the high initial cost of reasoning with the need for efficient long-term perfor-
mance. To quantify this efficiency gain, we conducted a simulation experiment
processing 200 DRG audit tasks. The first 100 tasks were treated as the "Ini-
tial Phase," requiring de novo reasoning. The subsequent 100 tasks were the
"Mature Phase," where 80% of tasks were logically similar to one of the initial
tasks and could be resolved via the RAG matching mode.
Table6: ComputationalCostAnalysisofMMIAinDifferentOperationalPhases
Phase Task TypeAvg. To-
kens/TaskRelative
Time/Task
Initial
PhaseDe Novo
Reasoning∼3,500 100%
Mature
PhaseRAG
Matching
(80%)∼500 15%
De Novo
Reasoning
(20%)∼3,500 100%
Mature
Phase
(Avg.)- ∼1,100 32%
As shown in Table 6, in the mature phase, the average token consumption
per task dropped to approximately 1,100, only 31.4% of the initial cost, with a
corresponding reduction in processing time. This demonstrates that the MMIA
architecture is not only reliable but also economically scalable, significantly
reducing operational costs over time by learning and reusing validated reasoning
patterns.
17

5 Discussion
5.1 Main Findings and Generalizability
The core finding of this research is that encapsulating an LLM’s capabilities
within a formal, verifiable reasoning framework significantly enhances its relia-
bility for complex, high-stakes medical tasks. MMIA demonstrated consistently
high performance across four distinct yet equally rigorous domains—from clin-
ical coding and regulatory compliance to EHR quality control and policy adju-
dication. This indicates that our proposed "plan-execute-verify" architecture is
highly generalizable. Its strength lies in transforming the task-solving process
from a dependency on the LLM’s internal, unreliable "thought process" into an
external, strictly auditable sequence of operations.
5.2 Implications for Trustworthy Medical AI
MMIA provides a practical technological pathway for building trustworthy AI
systems that meet the stringent requirements of the medical field. By external-
izing the reasoning process into a transparent, auditable log, the architecture
directly addresses the "black-box" problem of LLMs [6]. The final audit report
provides a solid foundation for accountability, allowing human experts to un-
derstand, verify, and ultimately trust the AI’s conclusions, thereby satisfying
a core ethical requirement for medical AI [4, 11]. This marks a crucial shift
from merely "trusting" AI to providing humans with the tools to "verify" it.
Furthermore, we emphasize that the verification process itself is LLM-driven.
While a single LLM auditor may err, a consensus mechanism achieved through
multiple, independent iterative verifications can reduce the probability of an
error in the original reasoning chain going undetected to a negligible level, thus
greatly enhancing the system’s overall reliability [20].
5.3 Limitations and Future Directions
Despitetheencouragingresults, thisstudyhasseverallimitations. First, thefor-
malizationofdomainknowledgeintoanaxiombaserequiresasignificantupfront
investment of time and collaboration between domain experts and knowledge
engineers, even with LLM assistance [30, 31, 32, 33, 34, 35, 36]. Second, the
system’s overall performance is still partially dependent on the capabilities of
the internal LLM components (e.g., the Planner and Auditor). A less capable
LLM might generate suboptimal plans or conduct insufficiently thorough audits.
Finally, the scope of this study was limited to textual data and did not involve
multimodal information.
Futureresearchcouldexploreseveraldirections: (1)developingsemi-automated
methods to accelerate the extraction and construction of axiom bases from un-
structureddocuments; (2)investigatingtheuseofmultiple,heterogeneousLLMs
within the framework (e.g., one for planning, another for auditing) to reduce
single-model bias and systemic risk; and (3) extending the MMIA architecture
18

to handle multimodal data, such as integrating medical imaging with clinical
text for comprehensive reasoning and verification.
6 Conclusion
TheapplicationofLargeLanguageModelsinmedicineisseverelyconstrainedby
their inherent unreliability. This paper introduces the "Haibu Mathematical-
Medical Intelligent Agent" (Haibu MMIA), a novel, LLM-driven architecture
designed to address this core challenge. Our contribution is not an attempt
to create a flawless LLM, but rather to ensure the reliability of outcomes by
enforcing a formally verifiable reasoning process. Through successful validation
across four high-stakes healthcare administration scenarios, we have demon-
strated that the "plan, execute, and audit" methodology is a viable and pow-
erful strategy. More importantly, by converting validated reasoning chains into
reusable theorems, MMIA significantly improves computational efficiency over
time, addressing the critical issues of scalability and cost-effectiveness. It en-
ables the construction of AI systems that are safe, transparent, accountable,
and economically sustainable—paving the way to bridge the gap between the
immense potential of generative AI and the stringent reliability requirements of
the medical domain.
References
[1] Large Language Models in patient education and engagement: a scoping
review,https://www.frontiersin.org/journals/medicine/articles/
10.3389/fmed.2024.1477898/full
[2] Applications of Large Language Models in Clinical Settings,https://www.
mdpi.com/2077-0383/13/11/3041
[3] Large language models in healthcare,https://mednexus.org/doi/10.
1016/j.jointm.2024.12.001
[4] Limitations and Challenges of Large Language Models in Healthcare,
https://www.i-jmr.org/2025/1/e59823/
[5] Hallucinations in LLMs: Types, Causes, and Approaches for En-
hanced Reliability,https://www.researchgate.net/publication/
385085962_Hallucinations_in_LLMs_Types_Causes_and_Approaches_
for_Enhanced_Reliability
[6] Challenges and opportunities for large language models in primary health
care: a narrative review,https://pmc.ncbi.nlm.nih.gov/articles/
PMC11960148/
[7] Assessing the Utility of Large Language Models for Clinical Text Summa-
rization,https://pmc.ncbi.nlm.nih.gov/articles/PMC12075489/
19

[8] Large Language Models in Biomedicine,https://www.annualreviews.
org/content/journals/10.1146/annurev-biodatasci-103123-094851
[9] The Promises and Perils of Large Language Models in Healthcare,https:
//www.mdpi.com/2077-0383/14/17/6169
[10] A Systematic Comparison of Large Language Model-Generated and
Human-Authored Clinical Reviews,https://pmc.ncbi.nlm.nih.gov/
articles/PMC11923074/
[11] Medical Ethics of Large Language Models in Medicine,https:
//www.researchgate.net/publication/381496095_Medical_Ethics_
of_Large_Language_Models_in_Medicine
[12] Formal Methods,https://users.ece.cmu.edu/~koopman/des_s99/
formal_methods/
[13] AMulti-LayerTaskDecompositionArchitectureforRobotsBasedonLarge
Language Models,https://www.mdpi.com/1424-8220/24/5/1687
[14] ADaPT: As-Needed Decomposition and Planning for Complex Tasks,
https://aclanthology.org/2024.findings-naacl.264/
[15] ADaPT: As-Needed Decomposition and Planning for Complex Tasks,
https://allenai.github.io/adaptllm/
[16] Task Navigator: Decomposing Complex Tasks for Multimodal Large Lan-
guage Models,https://openaccess.thecvf.com/content/CVPR2024W/
MAR/papers/Ma_Task_Navigator_Decomposing_Complex_Tasks_for_
Multimodal_Large_Language_Models_CVPRW_2024_paper.pdf
[17] Boosting Logical Fallacy Detection in Large Language Models via Logical
Structure Tree,https://aclanthology.org/2024.emnlp-main.730/
[18] Large Language Models Are Better Logical Fallacy Reasoners with Coun-
terargument, Explanation, and Goal-Aware Prompt Formulation,https:
//arxiv.org/abs/2503.23363
[19] Logical fallacy detection with LLMs,https://www.glukhov.org/post/
2024/05/logical-fallacy-detection-with-llms/
[20] Self-Refine: Iterative Refinement with Self-Feedback,https:
//openreview.net/pdf?id=S37hOerQLB
[21] Self-Refine: Iterative Refinement with Self-Feedback,https:
//selfrefine.info/
[22] Is Deep Iterative Reasoning Necessary?https://arxiv.org/html/2502.
10858v1
20

[23] IMPROVE: Iterative Refinement for LLM-based ML Pipeline Optimiza-
tion,https://arxiv.org/html/2502.18530v1
[24] China Hospital Information Management Association White Pa-
per,https://www.chima.org.cn/Sites/Uploaded/File/2020/07/
216373092145293830997645198.pdf
[25] LLM-based Automated Food Safety Compliance Checking,https://
arxiv.org/html/2404.17522v1
[26] How to Extract Data from Documents Using LLMs,https://www.
documentpro.ai/blog/extract-data-from-documents-using-llms/
[27] AI and LLMs in Regulatory Affairs,https://intuitionlabs.ai/
articles/ai-llms-regulatory-affairs
[28] LLM-Powered Rules Engines,https://artificio.ai/blog/
llm-powered-rules
[29] How to Use LLM to Extract Important Information
from Legal Documents,https://medium.com/@addepto/
how-to-use-llm-to-extract-important-information-from-legal-documents-a40ac6f974d8
[30] From Unstructured Text to Interactive Knowledge Graphs
Using LLMs,https://robert-mcdermott.medium.com/
from-unstructured-text-to-interactive-knowledge-graphs-using-llms-dd02a1f71cd6
[31] A General Framework for Building Knowledge Bases with an Aid of LLMs,
https://arxiv.org/html/2411.08278v1
[32] Combining Large Language Models and Knowl-
edge Graphs,https://www.wisecube.ai/blog/
combining-large-language-models-and-knowledge-graphs/
[33] Knowledge Graphs and LLMs: The Ultimate Guide,https://www.
datacamp.com/blog/knowledge-graphs-and-llms
[34] Building Knowledge Graphs Using Large Lan-
guage Models,https://medium.com/@shuchawl/
building-knowledge-graphs-using-large-language-models-07da1935b21a
[35] How to Convert Unstructured Text to Knowledge
Graphs Using LLMs,https://neo4j.com/blog/developer/
unstructured-text-to-knowledge-graph/
[36] Better approaches for building knowledge graphs from documents,
https://www.reddit.com/r/LangChain/comments/1jsqlhw/better_
approaches_for_building_knowledge_graphs/
[37] A Survey on Retrieval-Augmented Generation for Large Language Models,
https://arxiv.org/html/2503.10677v2
21

[38] Retrieval-augmented generation,https://en.wikipedia.org/wiki/
Retrieval-augmented_generation
[39] What is RAG (retrieval augmented generation)?https://www.ibm.com/
think/topics/retrieval-augmented-generation
[40] What is Retrieval-Augmented Generation (RAG)?https://www.k2view.
com/what-is-retrieval-augmented-generation
[41] Retrieval-Augmented Generation (RAG): Bridging LLMs
with External Knowledge,https://www.walturn.
com/insights/retrieval-augmented-generation-(rag)
-bridging-llms-with-external-knowledge
[42] Synthetic data generation: a privacy-preserving approach to acceler-
ate rare disease research,https://pmc.ncbi.nlm.nih.gov/articles/
PMC11958975/
[43] Synthetic Data Generation Using Generative AI,https://healthpolicy.
duke.edu/sites/default/files/2025-06/Synthetic\%20Data\
%20Generation\%20Using\%20Generative\%20AI.pdf
[44] Large language models generating synthetic clinical datasets,
https://www.frontiersin.org/journals/artificial-intelligence/
articles/10.3389/frai.2025.1533508/full
[45] A Systematic Review of Synthetic Data Generation Techniques Using Gen-
erative AI,https://www.mdpi.com/2079-9292/13/17/3509
[46] Generating Synthetic Electronic Health Record Data Using Generative Ad-
versarial Networks,https://ai.jmir.org/2024/1/e52615/
[47] Benchmarking Clinical Decision Support Search,https://arxiv.org/
abs/1801.09322
[48] China’s NHSA issued the Regulations on DRG/DIP Healthcare Security
Payment Mechanism,https://cms-lawnow.com/en/ealerts/2025/09/
china-s-nhsa-issued-the-regulations-on-drg-dip-healthcare-security-payment-mechanism
[49] Diagnosis-related Groups (DRG) pricing and payment policy in
China: where are we?https://www.researchgate.net/publication/
347201639_Diagnosis-related_Groups_DRG_pricing_and_payment_
policy_in_China_where_are_we
[50] Facilitators and barriers to the implementation of DIP payment methodol-
ogy reform,https://pmc.ncbi.nlm.nih.gov/articles/PMC12162575/
[51] DRG/DIP Reforms on Hospital Medical Device Procurement and Usage,
https://chinameddevice.com/drg-dip-china-medical-device/
22

[52] Avoiding the Top 10 Medical Coding Errors,https://www.allzonems.
com/top-10-medical-coding-errors/
[53] Medical coding mistakes that could cost you,
https://www.ama-assn.org/practice-management/cpt/
medical-coding-mistakes-could-cost-you
[54] Understanding DRG and its Relevance to Med-
ical Coding,https://moldstud.com/articles/
p-understanding-drg-and-its-relevance-to-medical-coding
[55] Mastering DRG Coding: Your Key to Rev-
enue Integrity,https://bluebrix.health/articles/
mastering-drg-coding-your-key-to-revenue-integrity-and-clinical-excellence-in-us-healthcare
[56] Challenges and Adverse Outcomes of Implementing Reimbursement Mech-
anisms Based on the DRG,https://pmc.ncbi.nlm.nih.gov/articles/
PMC7574807/
[57]国家医保局发布新版DRG/DIP付费分组方案,https://www.gov.cn/
lianbo/bumen/202407/content_6964140.htm
[58] DRG/DIP2.0版本分组方案正式出炉,https://www.zgylbx.com/index.
php?m=content&c=index&a=show&catid=6&id=47969
[59]国家医保局发布DRG/DIP 2.0版分组方案,https://www.hit180.com/
68272.html
[60]按病组（DRG）和病种分值（DIP）付费2.0版分组方案新闻发布会实录,
https://www.nhsa.gov.cn/art/2024/7/23/art_14_13318.html
[61]《关于印发按病组（DRG）和病种分值（DIP）付费2.0版分组方案并深入
推进相关工作的通知》政策解读,https://www.nhsa.gov.cn/art/2024/
7/23/art_105_13316.html
[62]国家医疗保障局办公室关于印发按病组和病种分值付费2.0版分组方案
并深入推进相关工作的通知,https://www.gov.cn/zhengce/zhengceku/
202407/content_6964136.htm
[63] 21 CFR Part 820: Medical Device Quality System Regulations,https:
//www.greenlight.guru/blog/21-cfr-part-820
[64] Overview of Device Regulation,https://www.fda.gov/
medical-devices/device-advice-comprehensive-regulatory-assistance/
overview-device-regulation
[65] FDA 21 CFR: Requirements Overview,https://www.qualio.com/blog/
fda-21-cfr-requirements-overview
23

[66] Navigating the NMPA Medical Device Registration Process in China,
https://www.emergobyul.com/sites/default/files/2024-12/
China-NMPA-Registration-Whitepaper.pdf
[67] Navigating China’s NMPA Program for Medical De-
vice Manufacturers,https://www.intertek.com/blog/2024/
07-24-navigating-chinas-nmpa-program-for-medical-device-manufacturers/
[68] Part 814 - Premarket Approval of Medical Devices,https://www.ecfr.
gov/current/title-21/chapter-I/subchapter-H/part-814
[69] Full List of 181 NMPA Guidelines 2022,https://chinameddevice.com/
181-guidelines-nmpa-2022/
[70] Guideline for Medical Device Registration in
China,https://www.cirs-group.com/en/md/
cfda-nmpa-guideline-for-medical-device-registration-in-china
[71] Laws and Regulations,https://english.nmpa.gov.cn/
lawsandregulations.html
[72] Center for Medical Device Evaluation,https://english.nmpa.gov.cn/
2019-07/19/c_389172.htm
[73] Search for FDA Guidance Documents,https://www.fda.gov/
regulatory-information/search-fda-guidance-documents
[74] Guidance Documents (Medical Devices and Radiation-
Emitting Products),https://www.fda.gov/medical-devices/
device-advice-comprehensive-regulatory-assistance/
guidance-documents-medical-devices-and-radiation-emitting-products
[75] Recent Final Medical Device Guidance Docu-
ments,https://www.fda.gov/medical-devices/
guidance-documents-medical-devices-and-radiation-emitting-products/
recent-final-medical-device-guidance-documents
[76] Electronic Submission Template for Medi-
cal Device Q-Submissions,https://www.fda.gov/
regulatory-information/search-fda-guidance-documents/
electronic-submission-template-medical-device-q-submissions
[77] Electronic Submission Template for Medical De-
vice 510(k) Submissions,https://www.fda.gov/
regulatory-information/search-fda-guidance-documents/
electronic-submission-template-medical-device-510k-submissions
[78] Subchapter H - Medical Devices,https://www.ecfr.gov/current/
title-21/chapter-I/subchapter-H
24

[79] Devices Guidances,https://www.fda.gov/vaccines-blood-biologics/
general-biologics-guidances/devices-guidances
[80] Developing and Responding to Deficiencies in Accordance
with the Least Burdensome Provisions,https://www.fda.gov/
regulatory-information/search-fda-guidance-documents/
developing-and-responding-deficiencies-accordance-least-burdensome-provisions
[81] CME: Avoiding common documentation errors,https://www.tmlt.org/
resource/cme-avoiding-common-documentation-errors
[82] 5 Common EHR Mistakes Your Staff Makes,
https://info.nhanow.com/learning-leading-blog/
5-common-ehr-mistakes-your-staff-is-making-and-what-theyre-costing-you
[83] The 10 Most Common EHR Documentation Er-
rors,https://www.chirohealthusa.com/consultants/
the-10-most-common-ehr-documentation-errors/
[84] Nursing documentation: How to avoid the most common medi-
cal errors,https://www.wolterskluwer.com/en/expert-insights/
nursing-documentation-how-to-avoid-the-most-common-medical-documentation-errors
[85] Do’s and don’ts of defensive documentation in the EHR,
https://www.nso.com/Learning/Artifacts/Articles/
Dos-and-donts-of-defensive-documentation-in-the-EHR
[86] Application of Medical Record Quality Control System Based on Artificial
Intelligence,https://pubmed.ncbi.nlm.nih.gov/38162053/
[87] Application of Medical Record Quality Control System Based on Artifi-
cial Intelligence,https://ykxb.scu.edu.cn/en/article/doi/10.12182/
20231160206?viewType=HTML
[88] Improving healthcare quality by unifying the American electronic
medical report system,https://pmc.ncbi.nlm.nih.gov/articles/
PMC10942963/
[89] Electronic Medical Record Systems,https://digital.ahrq.gov/
electronic-medical-record-systems
[90] Standardization and Automatic Extraction of Quality
Measures in an Ambulatory Electronic Medical Record
(EMR),https://digital.ahrq.gov/ahrq-funded-projects/
standardization-and-automatic-extraction-quality-measures-ambulatory-electronic
[91] How You Define Complex Approval Rules by Nesting Conditions,
https://docs.oracle.com/en/cloud/saas/procurement/25c/oapro/
how-you-define-complex-approval-rules-by-nesting-conditions.
html
25

[92] Managing Multiple Insurance Coverage: Secondary Insur-
ance and Coordination of Benefits,https://docstation.co/
managing-multiple-insurance-coverage-secondary-insurance-and-coordination-of-benefits-2/
[93] Best practices for verification of complex health insur-
ance plans,https://www.outsourcestrategies.com/blog/
streamlining-verification-complex-insurance-plans/
[94] Coverage Examples: Single Plan Logic Code Comments,https://www.
cms.gov/cciio/resources/files/downloads/sbc-cover-ex-logic.
pdf
[95] Navigating the Maze: A Look at Health Insurance Complexities
and Consumer Protections,https://www.kff.org/private-insurance/
navigating-the-maze-a-look-at-health-insurance-complexities-and-consumer-protections/
[96] Insurance Scenarios,https://www.mercycare.org/patients/
billing-insurance/pricing-transparency/insurance-scenarios/
[97] How Automated Claims Adjudication Software is Ending Man-
ual Misery,https://www.agentech.com/resources/articles/
automated-claims-adjudication-software
[98] How AI is Transforming Insurance Claims Process-
ing,https://www.equisoft.com/insights/insurance/
how-ai-revolutionizing-insurance-claims-processing
[99] Claim Auto-Adjudication Engine Development,https://kodjin.com/
blog/auto-adjudication-of-claims-system-case/
[100] Zero Touch Claims – How P&C insurers can optimize
claims processing,https://aws.amazon.com/blogs/industries/
zero-touch-claims-how-pc-insurers-can-optimize-claims-processing-using-aws-ai-ml-services/
[101] How Insurance Companies Are Automat-
ing Claims,https://www.kognitos.com/blog/
how-insurance-companies-are-automating-claims-processing/
[102] FHIR-based claims auto-adjudication engine,https://edenlab.io/
case/fhir-based-claims-auto-adjudication-engine
26