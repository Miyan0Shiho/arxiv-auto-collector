# An Agentic RAG and Evaluation Framework for Assurance Case Generation: Industrial Use Case for the EU Cyber Resilience Act Compliance

**Authors**: Fariz Ikhwantri, Iker Lasa Ojanguren, Dusica Marijan, Maria I. Maslioukova, José Arias Marin, Pavlos Kosmides

**Published**: 2026-08-19 23:54:30

**PDF URL**: [https://arxiv.org/pdf/2608.19509v1](https://arxiv.org/pdf/2608.19509v1)

## Abstract
Complying with the EU Cyber Resilience Act (CRA) is a resource-intensive challenge for SMEs due to the complexity of cybersecurity conformity assessment. Yet, it is essential for demonstrating regulatory compliance and ensuring product security and resilience. To address this, we introduce an automated framework for generating Assurance Cases (ACs) using an agentic Retrieval-Augmented Generation grounded in a formal Claim-Argument-Evidence logic. By systematically mapping technical documentation requirements, the framework streamlines the generation of certification evidence. We validate our approach on a case study of Catalink's PATROLIoT wildfire monitoring system, where the agentic RAG generated 70 ACs with high grounding density (~4.4 artefacts per AC). The proposed Natural Language Inference (NLI) evaluator achieves 0.88 accuracy, which provides robust evidence grounding and traceability, while expert-validated plausibility (3.06) supports interpretable justifications. For practitioners, this work provides a scalable, interpretable approach for automating mandatory CRA conformity assessments, reducing manual effort while maintaining transparent decision support.

## Full Text


<!-- PDF content starts -->

An Agentic RAG and Evaluation Framework for
Assurance Case Generation: Industrial Use Case for
the EU Cyber Resilience Act Compliance
Fariz Ikhwantri§, Iker Lasa Ojanguren†, Dusica Marijan§, Maria I. Maslioukova‡, Jos ´e Arias Marin†, Pavlos Kosmides‡
§Simula Research Laboratory, Kristian August Gate 23, 0164, Oslo, Norway
Emails:{fariz, dusica}@simula.no
†TECNALIA, Astondo Bidea 700, Derio 48160, Biscay, Spain
Emails:{iker.lasa, jose.arias}@tecnalia.com
‡Catalink Limited, Kleomenous 2, 1061, Nicosia, Cyprus
Emails:{maria.maslioukova, pkosmidis}@catalink.eu
Abstract—Complying with the EU Cyber Resilience Act (CRA)
is a resource-intensive challenge for SMEs due to the complexity
of cybersecurity conformity assessment. Yet, it is essential for
demonstrating regulatory compliance and ensuring product se-
curity and resilience. To address this, we introduce an automated
framework for generating Assurance Cases (ACs) using an agen-
tic Retrieval-Augmented Generation grounded in a formal Claim-
Argument-Evidence logic. By systematically mapping technical
documentation requirements, the framework streamlines the gen-
eration of certification evidence. We validate our approach on a
case study of Catalink’s PATROLIoT wildfire monitoring system,
where the agentic RAG generated 70 ACs with high grounding
density (≈4.4 artefacts per AC). The proposed Natural Language
Inference (NLI) evaluator achieves 0.88 accuracy, which provides
traceability, while expert-validated plausibility (3.06) supports
interpretable justifications. For practitioners, this work provides
a scalable, interpretable approach for automating mandatory
CRA conformity assessments, where it automates document
retrieval and AC construction steps that are otherwise performed
manually while maintaining transparent decision support.
Index Terms—Assurance Case, Compliance, Cyber Resilience
Act, Agentic AI.
I. INTRODUCTION
Assurance cases (ACs) are a structured form of reasoning
used in safety- and security-critical requirements engineer-
ing, typically expressed as tree-structured argumentation (e.g.,
Claims–Arguments–Evidence) [1]. They improve confidence
in system requirements by providing evidence of traceability,
logical consistency, and completeness [2]. However, con-
structing high-quality AC is a labour-intensive and expertise-
driven process, creating a bottleneck in the development and
certification of complex systems for companies [3].
The introduction of the EU Cyber Resilience Act (CRA)
further increases the need for efficient assurance engineering.
The CRA establishes mandatory cybersecurity requirements
for products with digital elements placed on the EU market.
It requires manufacturers to demonstrate compliance through
comprehensive technical documentation and supporting ev-
idence. While these requirements improve the security andresilience of digital products, preparing conformity evidence
imposes a substantial burden on small and medium-sized
enterprises (SMEs), which often lack dedicated compliance ex-
pertise and resources. Reducing the effort required to develop
transparent and traceable assurance arguments is therefore an
important industrial challenge.
Prior study has utilised structured patterns and Large
Language Models (LLMs) to automate the synthesis and
evaluation of assurance arguments [4]–[7]. While these ap-
proaches demonstrate the potential for automation, they are
often restricted to the instantiation of predefined templates
or suffer from a limited scope that does not account for
the heterogeneous and fragmented technical documentation
required for industrial certification. Consequently, establishing
deep, verifiable traceability between generated claims and the
specific local evidence remains a significant challenge.
Furthermore, industrial adoption is hindered by significant
practical barriers. Existing frameworks often assume cloud-
based access to LLM services, posing confidentiality risks for
proprietary documentation during the certification process [4],
[8]. On the evaluation side, automated review such as ”LLM-
as-a-judge” frequently struggles with the semantic nuances of
technical reasoning [9] or depends on expensive, manually
annotated datasets [10] that are unavailable for new regulatory
domains like the EU CRA. These limitations highlight an
urgent need for an end-to-end framework capable of grounded
generation and interpretable verification under realistic indus-
trial constraints. Our work addresses these gaps by integrating
an agentic RAG approach with a domain-expert plausibility
study, focusing on the utility of NLI-based rationales in a
regulatory environment where the training data is not readily
available.
To address these challenges, we propose an end-to-
end framework for automated assurance case generation
based on Agentic Retrieval-Augmented Generation (Agentic
RAG). The framework constructs assurance cases using the
Claim–Argument–Evidence (CAE) structure while ground-
arXiv:2608.19509v1  [cs.SE]  19 Aug 2026

ing each generated argument in organisation-specific techni-
cal documentation retrieved from controlled local knowledge
sources. Rather than verifying the correctness of the under-
lying evidence itself, our approach focuses on verifying the
logical consistency of the generated assurance text compo-
nents. Specifically, it assesses whether generated claims are
semantically supported by their associated evidence and sub-
claims. This enables transparent and interpretable assessment
of generated assurance cases while remaining compatible with
industrial confidentiality requirements.
To evaluate the quality of the generated ACs, we integrate
an interpretable NLI-based evaluation [11]–[13] framework
that measures semantic support between generated claims,
subclaims, and evidence. In addition, we complement the
automatic evaluation with a domain expert plausibility study
to reflect the industrial assurance needs better.
We validate our framework in an industrial case study of
Catalink’s PATROLIoT wildfire monitoring system, where we
generate ACs covering key requirements aligned with the
EU CRA. In total, 70 assurance cases spanning seven CRA
requirements are automatically constructed and evaluated. The
framework achieves≈4.4 grounded artefacts per AC, while the
NLI evaluator reaches 0.88 accuracy and 0.85 F1 on unseen
auditor data.
II. RELATEDWORK
A. Assurance Case Generation
There is increased interest in automating assurance case
generation to mitigate the high cost and manual effort of certi-
fication. Early approaches utilised safety patterns and reusable
templates [14], [15], which guide argument construction but
require significant expert customisation. To increase auton-
omy, research has shifted toward LLMs using sophisticated
prompting strategies to synthesise Goal Structured Notation
(GSN) structures from high-level requirements [4], [5]. Re-
cent advancements also explore hybrid formal methods like
TrustAs [8] and the automation ofeliminative argumentation,
which strengthens arguments by identifying potential weak-
nesses or “defeaters.” Frameworks such asCoDefeater[7]
and AI-supported eliminative systems [6] leverage LLMs to
automatically generate candidate defeaters, helping engineers
iteratively address vulnerabilities in safety and security claims.
Despite the generative capabilities, a critical industrial
bottleneck persists: most approaches rely on internal LLM
knowledge rather than grounded product artefacts, yielding ar-
guments that are syntactically valid but semantically unsound
or factually misaligned [16]. Since industrial documentation
is typically scattered and heterogeneous, existing frameworks
often fail by assuming pre-structured templates. Navigating
unstructured repositories under SME constraints requires an
agentic approach for iterative, grounded search and refinement
to ensure local evidence substantiates every claim.
B. Assurance Case Evaluation
AC evaluation has evolved from qualitative structural cri-
teria [17] and graph-based analysis [10] toward semanticverification. Early efforts in version comparison [18] and
more recent multi-hop NLI approaches [13] focus on ver-
ifying whether evidence logically entails a claim, moving
beyond syntax to analyse reasoning integrity. Most recently,
the ”LLM-as-a-judge” paradigm has been proposed to provide
automated review against rubrics [9]. However, as Yu et al.
noted, LLM judges often provide vague assessments due to
a lack of access to the external technical context. Our work
bridges this gap by extending our previous multi-hop NLI
framework [13] with Agentic RAG generation, synthetic-to-
real evaluation, and an industrial CRA case study.
III. INDUSTRIALUSECASE: CYBERRESILIENCEACT IN
FIREDETECTIONSYSTEM
PATROLIoT is an Internet of Things (IoT) system designed
to support firefighters and first responders in monitoring rural
areas and detecting potential wildfire events. The system is
equipped with environmental sensors that provide situational
awareness, as well as a camera that offers visual feedback from
the monitored area. Captured images are processed locally
using machine learning models to detect the presence of fire
or smoke and generate alerts in a user dashboard to support
timely response actions.
As an IoT system transmitting data to a backend, PA-
TROLIoT falls under the CRA. Since it does not meet
the CRA’s Annex III/IV criteria for ’important’ or ’critical’
products, it requires self-assessment, shifting the full bur-
den of demonstrating compliance with essential requirements
onto the SME. In practice, self-assessment does not make
the process straightforward for an SME. Without dedicated
cybersecurity or regulatory compliance personnel, it is often
unclear which existing technical documentation qualifies as
relevant evidence, how it should be organised to support
each requirement, and how to trace it systematically back
to specific CRA articles. Relevant information is typically
scattered across design documents, test reports, and internal
records. This makes the preparation of assurance arguments
a slow, expertise-dependent task and a significant barrier to
timely compliance.
IV. METHODOLOGY
A. Agentic RAG: Grounded Assurance Case Generation
We approach assurance case generation as an RAG problem
combined with agentic AI. The agent constructs an assurance
case grounded in regulatory requirements with multiple itera-
tions. We use Agentic AI based on the ReAct method [19]
from LangChain1to support iterative retrieval and query
refinement over heterogeneous technical documentation. The
agent starts with i) a requirement from the CRA and ii) the
product context information: a JSON string of less than 1800
characters that covers a short product description, component
type, deployment context, firmware OS, protocols, security
features and certification scope. Then, the agent iteratively
generates search queries informed by both the requirement
1https://github.com/langchain-ai/langchain

and product characteristics. As the agent retrieves additional
information, it dynamically refines its subsequent queries to
maximise relevance and coverage. When the agent determines
that sufficient supporting information has been collected, it
synthesises the findings into a hierarchical JSON artefact that
adheres to a Claim-Argument-Evidence tree schema. This
artefact reflects the decomposition of the initial requirement
and grounds each claim, argument, and evidence item in the
context of industrial product documentation.
Agent Instruction.The agent is steered by a comprehensive
system prompt that ties togetherthe CRA requirement, the
product context information, retrieval tools, and the assurance-
case output format. Session-specific fields (the standard, re-
quirement ID and text, and product context) are passed at
runtime. Recent conversation and retrieval activity can be
appended so that each turn reflects the current session, not
only the static instructions. The prompt is split into a few
sections:whatthe agent must produce (CAE-based JSON
with requirement traceability),howit should behave (evidence-
grounded, explicit about gaps, no persistence without user
confirmation) andwhich toolsto use for document search and
optional code sources.
Agentic Tools.Our agent architecture follows a hybrid design
that integrates both (a) document ingestion and (b) semantic
search tool functionalities.
(a) Document ingestionprocesses and organises heteroge-
neous artefacts, such as design documents, test reports, and
audit records, from industrial workflows. Documentation is
ingested viaDocling, segmented into 3,000-character chunks
with a 500-character overlap, and indexed in ChromaDB using
BGE-M3 embeddings.
(b) Semantic search toolsenable broad semantic explo-
ration when the evidence location is unknown and support
targeted, filtered retrieval based on exact content match and/or
filename patterns. An additional tool collects file names and
relevant metadata in advance, allowing the agent to contex-
tualise the documentation and leverage filename semantics.
These also help to inform and refine subsequent searches.
The agent executes semantic search via L2 distance, utilising:
(i) lexical constraints for keyword-anchored retrieval, and (ii)
a two-phase filename filter that identifies target documents
prior to querying to prevent post-retrieval truncation. These
tools ensure high recall across the heterogeneous logs and
specifications common in industrial workflows.
Output argumentation schema.The agentic RAG produces
a hierarchical JSON artefact following a CAE schema. The
rootClaimnode links to a CRA ID; sub-claims decompose
viaArgumentobjects (AND/OR/ARG). Terminal nodes list
Evidenceobjects, which capture descriptive content, epistemic
status (e.g., type, confidence scores), and provenance metadata
for traceability. The deterministic identifiers support cross-
referencing. To ensure structural integrity, the outputs are
validated and corrected using thejson repairlibrary2.
2github.com/mangiucugna/json repairB. Multi-hop Inference for Synthetic Data Utility Evaluation
Following the NLI-based assurance reasoning framework
proposed in our previous work [13], we introduce a practical
utility evaluation scenariodesigned for the industrial problem,
where real-world training data is unavailable. This method-
ology tests whether a model trained on synthetic data from
Agentic RAG is acceptable by an external assessment body.
For each CAE edge, the supporting child is the premise and its
parent the hypothesis (e.g., EvidenceE→ClaimC), forming
an Entailment pair. Not-Entailment pairs sample nodes without
direct support, e.g.,E→an unrelated claimC′at the same
hierarchical level.
We employ synthetic data generation for utility learning and
evaluation [20] to simulate the deployment of the framework in
an SME context. This approach(Synthetic-to-Real)consists
of two phases:
Phase 1: Synthetic Training (D syn).We utilise the Agentic
RAG framework to process internal technical documentation
for the PATROLIoT system, generating tree-based CAE struc-
tures grounded in local evidence. To construct the training
corpus, we traverse the hierarchical edges of the generated
CAE trees to produce NLI pairs. The textual content for
each premise and hypothesis is extracted directly from the
descriptionandrationalefields defined in our argumentation
schema (Section IV-A), effectively transforming the structured
argumentation logic into a dataset of natural language infer-
ence steps.
Phase 2: Utility Testing (D real).To validate the framework’s
practical utility, we leverage assessment records from an
independent auditor, representing an external perspective. This
dataset consists of real-world technical findings formatted as
structured tabular data, comprisingAuditor Comments,SME
Responses, andIdentified Gaps from Evidence. We transform
these records into a multi-hop natural language inference
task designed to measure the model’s ability to verify the
consistency of the industrial audit trail. This evaluation phase
determines how well the logic learned from synthetic training
generalises to the rigorous requirements of an independent
assessment.
Both of these phases,D synandD real, concern PATROLIoT.
Inference variants. Chainprovides the complete reasoning
path, [Premise]+[Intermediate Premises]→[Hypothesis],
whereasw/o Chainprovides only [Premise]→[Hypothesis],
requiring intermediate steps to be inferred implicitly.
C. Expert Plausibility Study
To evaluate the interpretability of our NLI evaluator, we
conducted an expert plausibility study. The goal was not to
validate the technical security of the product, but to assess
whether the NLI model’s internal reasoning, represented via
token-level heatmaps, aligns with the domain expert.
The PATROLIoT Product Owner (PO) performed an au-
dit of 60 NLI instances using anOffline Heatmap Eval-
uationinterface. For each sample, the label is provided
(Entailment/Not-Entailment) and an explanation (highlighted
tokens). The PO evaluated the plausibility of the explanation

consisting of four questions with a 5-point Likert scale, and
one multiple-choice question based on the following rubric:
Q1:(Justification) Do the highlighted tokens represent the
technical keywords that actually justify the relationship?
Q2:(Necessity) Would the prediction be less reliable if these
specific technical terms were ignored?
Q3:(Sufficiency) Are the highlights enough to explain the
logic without reading the entire document?
Q4:(Trust) Does the quality of the highlighting increase trust
in the NLI evaluator’s decision?
Q5:(Agreement) Do you think the model prediction is techni-
cally correct? (Yes / No / Unsure). This question measures
the alignment between the NLI model’s classification and
the Product Owner’s ground-truth judgment.
V. EXPERIMENTALSETTINGS
A. Research Questions
RQ1(Grounding and Traceability): How effectively does the
Agentic RAG framework navigate heterogeneous docu-
mentation to establish verifiable, grounded links between
generated assurance claims and technical evidence?
RQ2:(Inference Generalisation) How accurately does an
NLI model trained on synthetic, agentically-generated
documentation (D syn) generalise to unseen, structured
industrial assessment artefacts (D real)?
RQ3:(Explanation Plausibility): To what degree do the NLI
model’s rationales (token-level highlights) align with the
technical knowledge of a domain expert?
B. Models Configuration
To evaluate the contribution of agentic grounding to the
NLI evaluator’s performance, we implement a baseline training
scenario on different synthetic training data:
Vanilla-LLM (D vanilla ):We prompt a standard LLM to gen-
erate synthetic NLI pairs for CRA compliance without
access to the PATROLIoT technical documentation. The
model relies entirely on its internal parametric knowledge
to synthesise technical claims and evidence.
Agentic RAG (Proposed):The NLI evaluator is trained on
Dsyn, which is grounded in specific, retrieved technical
artefacts via our agentic framework.
For fair comparison, we usedQwen3-Coder-30B-A3B-
Instruct-FP83for both Vanilla-LLM and Agentic-RAG mod-
els.
For theNLI evaluation, we compared two architectures:
BERT4for supervised fine-tuning andLLaMA5for In-
Context Learning (ICL). The 1B Llama model was specifically
chosen to ensure that gradient-based explanation extraction,
such as Integrated Gradients, remains computationally feasi-
ble, allowing us to audit the model’s focus on technical key-
words. For theexpert plausibility study, we useQwen2.5-
32B-Instruct6as the judge LLM.
3https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
4https://huggingface.co/google-bert/bert-large-uncased
5https://huggingface.co/meta-llama/Llama-3.2-1B
6https://huggingface.co/Qwen/Qwen2.5-32B-InstructBoth the Agentic RAG pipeline and the NLI models were
executed locally, ensuring that proprietary PATROLIoT docu-
mentation was not transmitted to external model services.
C. Requirement Split for Dataset Configuration.
To evaluate the framework’s generalisation capability, we
implemented a requirement-wise data split across seven se-
lected CRA requirements:
Training Set (D synandD real):Four requirements from
the CRA were used to generate a synthetic corpus via the
Agentic RAG output as synthetic Training. We select the four
requirements from the cybersecurity requirements in Annex I
part II of the EU CRA. In total, from 40 assurance cases, we
convert into≈4.2k instance pairs for theChainsetting, and
≈2.7k instance pairs for thew/o Chainsetting.
Test Set (D real):The model was evaluated on the remaining
three unseen requirements from real-world tabular data from
an independent audit as theUtility Testing. This set comprises
two cybersecurity requirements (Annex I part I) and one
requirement about vulnerability management (Annex I part II).
In total, from 3 requirements from tabular data, we convert
them into 30 instance pairs for the Chain setting with at most
4 hops and 24 instance pairs for the w/o Chain setting.
This requirement-wise split evaluates generalisation to reg-
ulatory requirements unseen during training. It simulates a
realistic industrial scenario where an SME must apply learned
compliance logic to new or evolving regulatory articles without
access to pre-labelled training data for those specific mandates.
VI. RESULTS ANDDISCUSSION
A. Grounding and Traceability: Citation Coverage (RQ1)
We definecitation coverageas the ratio of substantiated
evidence nodes, those containing direct provenance links to
technical documentation, relative to the total number of evi-
dence nodes generated. This metric measures traceability, not
evidence correctness or overall assurance-case validity.
Across 70 generated ACs, 307 of 356 evidence nodes
contained artefact references, corresponding to 86.2% cita-
tion coverage and≈4.4 referenced artefacts per AC. The
discrepancy (13.8%, i.e., 49/356) where evidence nodes lacked
specific artefact links serves as a valuable diagnostic signal for
documentation gaps, proactively identifying requirements that
lack technical substantiation. These 307 references originated
from 42 unique source entries, only22 of which (52.3%,
i.e., 22/42) contained formal file extensions. This variability
underscores the agentic framework’s ability to maintain high
traceability while navigating the heterogeneous and inconsis-
tent metadata of industrial SME repositories.
B. Inference Generalisation (RQ2)
To evaluate the effectiveness of the NLI evaluator, we report
performance across two primary training scenarios. The first
scenario utilises real-world data for both training and testing to
establish a performance ceiling. The second scenario evaluates
the practical utility of our synthetic training strategy when
applied to unseen real-world assessment artefacts.

Baseline: Train (Real)→Test (Real).This represents the tra-
ditional supervised approach using the limited human-labelled
audit data available. As shown in Table I, this configuration
suffers from extreme data scarcity, leading to poor generali-
sation. In many cases, the model fails to learn the underlying
compliance logic, instead over-fitting to the specific linguistic
noise of the small sample.
TABLE I
NLI PERFORMANCE COMPARISON ACROSSTRAININGSOURCES AND
REASONINGSETTINGS(CHAIN VS.W/OCHAIN).
Model Training Source Setting Acc. F1
BERTReal (Baseline) Chain 0.53 0.23
Real (Baseline) w/o Chain 0.54 0.23
Vanilla LLM Chain 0.70 0.69
Agentic RAGChain 0.83 0.82
Agentic RAG w/o Chain 0.88 0.85
LLaMAReal (Baseline) Chain 0.37 0.31
Real (Baseline) w/o Chain 0.50 0.47
Vanilla LLM Chain 0.43 0.31
Agentic RAGChain 0.53 0.37
Agentic RAGw/o Chain 0.54 0.38
Utility Testing: Train (Synthetic)→Test (Real).This
scenario represents the core industrial challenge where real-
world training data is unavailable. We evaluate the utility of
synthetic data generated via aVanilla LLMversus ourAgentic
RAGframework.
To measure the practical utility of the NLI evaluator, we re-
port its performance on the unseen auditor dataset (D real) after
training on the synthetic SME-side documentation (D syn).
Classification Performance:The model achieved an accuracy
of88%and an F1-score of85%. These results indicate that
the ”internal” logic learned from the Agentic RAG-generated
samples translates effectively to the ”external” perspective of
independent audit tables. The results in Table I provide several
key insights.
Impact of Grounding:Models trained on Agentic RAG
data outperformed the ungrounded Vanilla-LLM baseline, in-
dicating a benefit from document grounding. This comparison
does not isolate agentic iteration from conventional RAG.
Value of Chain Reasoning:The NLI model with theChain
setting generally yielded lower performance than the model
in thew/o Chainsetting (e.g., 0.83 vs 0.88 for BERT).
This shows that the synthetic training data of hierarchical
Claim-Argument-Evidence trees can be linearised into a multi-
hop representation, which is also applicable to real-world
multi-hop tabular data. The models trained in direct settings
generalise more effectively to these tabular audit records,
whereas multi-hop logic may introduce cumulative error or
over-specialisation to hierarchical structures.
Fine-tuned BERT vs ICL LLaMA :Fine-tuned BERT
outperformed the 1B LLaMA ICL configuration by up to
35%. Since the models differ in both adaptation strategy and
capacity, these results do not isolate an encoder-versus-decoder
effect.TABLE II
MEAN PLAUSIBILITY SCORES(Q1–Q4, 1–5)ANDQ5CATEGORY
PROPORTIONS(%).
Source MetricBERT LLaMA
Chain w/o Chain Chain w/o Chain
Human Q1: Just. 3.06 3.25 2.17 1.83
Human Q2: Nec. 2.78 3.17 2.94 2.58
Human Q3: Suf. 2.94 3.33 1.78 1.75
Human Q4: Trust 2.78 2.50 1.78 1.58
HumanMean 2.89 3.06 2.17 1.94
Human Q5 Agree. (%) 50.0 41.7 16.7 25.0
LLM Judge Q1: Just. 3.11 3.17 3.00 3.25
LLM Judge Q2: Nec. 3.39 3.92 3.89 3.50
LLM Judge Q3: Suf. 3.00 3.00 2.78 3.08
LLM Judge Q4: Trust 3.28 3.42 3.00 3.25
LLM JudgeMean 3.19 3.38 3.17 3.27
LLM Judge Q5 Agree. (%) 11.1 25.0 38.9 33.3
C. Explanation Plausibility (RQ3)
Our expert plausibility study reveals a significant inter-
pretability gap between the BERT model with fine-tuning
and the LLaMA with the ICL approach, with human audi-
tors favouring BERT’s technical faithfulness. Human experts
awarded BERT a peak mean plausibility score of 3.06 and
a 50% technical agreement rate (Q5), whereas LLaMA’s
explanations were deemed insufficient and poorly justified,
scoring as low as 1.75 in sufficiency. Conversely, the LLM
Judge exhibited a notable ”plausibility bias,” consistently
overrating LLaMA’s justifications and failing to align with
human experts on technical correctness, which is shown by the
sharp discrepancy in BERT Chain agreement (11.1% LLM vs
50% Human). These results demonstrate that while ICL-based
explanations may appear linguistically plausible to automated
judges, supervised fine-tuning on agentically grounded data
can provide more technically aligned token-level justifications
in this setting.
D. Discussion
Implications for Practitioners.Our agentic RAG addresses
the labour-intensive self-assessment bottleneck for SMEs in
CRA by automating documentation retrieval across frag-
mented documents, then converting the CRA requirements into
assurance cases. In practice, our citation coverage metric could
be used to identify documentation gaps before formal audits
to reduce the risk of certification failure. Our NLI evaluation
tools could enable semi-automatic workflows with the Agentic
RAG, such as pruning the redundant or irrelevant nodes
generated by the LLMs. Our evaluation with domain experts
shows that the proposed approach provides traceability from
input tokens to generated outputs, supporting transparency
required for regulatory compliance in the European market.
Iterative Refinement and Security Level.The industrial
case study highlights that generating ACs is rarely a one-
shot process. Initial outputs often reveal a need to refine the
arguments or provide more granular information regarding the
product’s security level. Although our framework fulfils this
requirement through its agentic, iterative capabilities, achiev-
ing compliance with higher security levels requires updating

the evidence based on real-world practice or implementation
rather than further assurance case refinement.
E. Threats to Validity
Construct Validity.This study distinguishes between tech-
nical logical soundness and formal regulatory compliance.
While domain experts validated the semantic consistency of
the generated arguments, they are not legal authorities on
the CRA. Consequently, while the system produces logically
coherent evidence, it cannot guarantee legal audit success;
future work must involve legal practitioners to ensure formal
compliance-readiness.
Internal Validity.Subjective bias poses a primary threat
to internal validity since the domain expert evaluator also co-
authors this paper. This dual role creates a risk of confirmation
bias where the evaluator might inadvertently favour model out-
puts. To mitigate this, we cross-referenced expert ratings with
an independent LLM-as-a-Judge uninvolved in the generation
process. Reporting the agreement between the human expert
and the automated judge ensures a more objective assessment
of system performance.
External Validity.External validity is limited by the single-
product focus of this study. While providing deep insight
into an SME’s needs, the findings may not generalise across
all industrial sectors. We addressed this by focusing on the
logic of NLI ’plausibility’ (Q1-Q4) rather than an absolute
ground truth of compliance, which remains a matter of legal
interpretation.
VII. CONCLUSION
This paper introduced an Agentic RAG framework for auto-
mated Assurance Case generation and evaluation under the EU
CRA. Our results demonstrate that the framework effectively
addresses the industry problem, where real-world training
data is not always available at the beginning.RQ1:Agentic
RAG automates document retrieval and AC construction while
preserving provenance traceability across heterogeneous tech-
nical documentation.RQ2:Agentically grounded synthetic
data supports generalisation to unseen auditor requirements
when labelled real-world data are scarce.RQ3:Interpretable
NLI can support engineers in inspecting the technical basis
of compliance-related decisions. Overall, the framework pro-
vides traceable and interpretable support for CRA assurance
workflows, while formal compliance remains subject to expert
assessment.
Code and Data Availability:The NLI training code
and the evaluation interface are publicly accessible (available
via7[13]). Due to confidentiality with our industry partner,
the Agentic RAG implementation, the output and technical
datasets remain proprietary.
ACKNOWLEDGMENT
This work is funded by the European Commission under
grant agreement No. 101120606, CertifAI. We thank EZU for
the evaluation of the Catalink assessment data. This work is
7https://github.com/farizikhwantri/exclaimsupported by the eX3 Infrastructure, funded by the Research
Council of Norway under contract 270053.
REFERENCES
[1] J. Rushby, X. Xu, M. Rangarajan, and T. L. Weaver, “Understanding
and evaluating assurance cases,” Langley Research Center, National
Aeronautics and Space Administration, Tech. Rep., 2015.
[2] N. Mansourov and D. Campara, “Chapter 2 - confidence as a product,”
inSystem Assurance, ser. The MK/OMG Press, N. Mansourov and
D. Campara, Eds. Boston: Morgan Kaufmann, 2011, pp. 23–47.
[3] E. Kang, “Assurance cost reduction through architecture design,” in1st
Workshop on Formal Arguments for Cps CerTification, ser. FACCT ’25.
Association for Computing Machinery, 2025.
[4] M. Sivakumar, A. B. Belle, J. Shan, and K. K. Shahandashti, “Prompting
gpt–4 to support automatic safety case generation,”Expert Systems with
Applications, vol. 255, p. 124653, 2024.
[5] O. Odu, A. B. Belle, S. Wang, S. Kpodjedo, T. C. Lethbridge, and
H. Hemmati, “Automatic instantiation of assurance cases from patterns
using large language models,”Journal of Systems and Software, vol.
222, p. 112353, 2025.
[6] T. Viger, L. Murphy, S. Diemert, C. Menghi, J. Joyce, A. Di Sandro, and
M. Chechik, “Ai-supported eliminative argumentation: Practical experi-
ence generating defeaters to increase confidence in assurance cases,”
in2024 IEEE 35th International Symposium on Software Reliability
Engineering (ISSRE), 2024, pp. 284–294.
[7] U. Gohar, M. C. Hunter, R. R. Lutz, and M. B. Cohen, “Codefeater:
Using llms to find defeaters in assurance cases,” in39th IEEE/ACM
International Conference on Automated Software Engineering, ser. ASE
’24. Association for Computing Machinery, 2024, p. 2262–2267.
[8] Z. Chen, Y . Deng, and W. Du, “Trusta: Reasoning about assurance cases
with formal methods and large language models,”Science of Computer
Programming, vol. 244, p. 103288, 2025.
[9] G. Yu, M. Sivakumar, A. B. Belle, S. Ghari, S. Wang, and
T. C. Lethbridge, “Llms as judges: Toward the automatic review
of gsn-compliant assurance cases,” 2025. [Online]. Available: https:
//arxiv.org/abs/2511.02203
[10] F. Ikhwantri and D. Marijan, “Evaluating assurance cases as text-
attributed graphs for structure and provenance analysis,” 2026. [Online].
Available: https://arxiv.org/abs/2604.20577
[11] J. Stacey, Y . Belinkov, and M. Rei, “Supervising model attention
with human explanations for robust natural language inference,” in
Proceedings of the AAAI conference on artificial intelligence, vol. 36,
no. 10, 2022, pp. 11 349–11 357.
[12] J. Vladika, I. Hacajova, and F. Matthes, “Step-by-step fact verification
system for medical claims with explainable reasoning,” inConference
of the Nations of the Americas Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies, 2025, pp. 805–816.
[13] F. Ikhwantri and D. Marijan, “Explainable compliance detection with
multi-hop natural language inference on assurance case structure,”
2025. [Online]. Available: https://arxiv.org/abs/2506.08713
[14] D. Ne ˇsi´c, M. Nyberg, and B. Gallina, “Product-line assurance cases
from contract-based design,”Journal of Systems and Software, vol. 176,
p. 110922, 2021.
[15] Z. Porter, I. Habli, J. McDermid, and M. Kaas, “A principles-based
ethics assurance argument pattern for ai and autonomous systems,”AI
and Ethics, vol. 4, no. 2, p. 593–616, June 2023.
[16] Y . Khan and H. Khan, “Causal sycophancy: Systematic causal
admissibility failures in llm-generated explanations,” 04 2026. [Online].
Available: http://dx.doi.org/10.2139/ssrn.6504272
[17] T. Chowdhury, A. Wassyng, R. F. Paige, and M. Lawford, “Systematic
evaluation of (safety) assurance cases,” inInternational Conference on
Computer Safety, Reliability, and Security. Springer, 2020, pp. 18–33.
[18] N. KOBAYASHI, A. NAKAMOTO, M. KAWASE, and S. SHI-
RAKAWA, “Comparison of two quantitative evaluation methods for
assurance cases,”International Journal of Japan Association for Man-
agement Systems, vol. 8, no. 1, pp. 27–34, 2016.
[19] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“React: Synergizing reasoning and acting in language models,” in
International Conference on Learning Representations (ICLR), 2023.
[20] E. Ji, J. H. Ohn, H. Jo, M.-J. Park, H. J. Kim, C. M. Shin, and
S. Ahn, “Evaluating the utility of data integration with synthetic data
and statistical matching,”Scientific Reports, vol. 15, no. 1, 2025.