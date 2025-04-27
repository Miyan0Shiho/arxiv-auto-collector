# TVR: Automotive System Requirement Traceability Validation and Recovery Through Retrieval-Augmented Generation

**Authors**: Feifei Niu, Rongqi Pan, Lionel C. Briand, Hanyang Hu, Krishna Koravadi

**Published**: 2025-04-21 20:37:23

**PDF URL**: [http://arxiv.org/pdf/2504.15427v1](http://arxiv.org/pdf/2504.15427v1)

## Abstract
In automotive software development, as well as other domains, traceability
between stakeholder requirements and system requirements is crucial to ensure
consistency, correctness, and regulatory compliance. However, erroneous or
missing traceability relationships often arise due to improper propagation of
requirement changes or human errors in requirement mapping, leading to
inconsistencies and increased maintenance costs. Existing approaches do not
address traceability between stakeholder and system requirements, rely on
open-source data -- as opposed to automotive (or any industry) data -- and do
not address the validation of manual links established by engineers.
Additionally, automotive requirements often exhibit variations in the way they
are expressed, posing challenges for supervised models requiring training. The
recent advancements in large language models (LLMs) provide new opportunities
to address these challenges. In this paper, we introduce TVR, a requirement
Traceability Validation and Recovery approach primarily targeting automotive
systems, leveraging LLMs enhanced with retrieval-augmented generation (RAG).
TVR is designed to validate existing traceability links and recover missing
ones with high accuracy. We empirically evaluate TVR on automotive
requirements, achieving 98.87% accuracy in traceability validation and 85.50%
correctness in traceability recovery. Additionally, TVR demonstrates strong
robustness, achieving 97.13% in accuracy when handling unseen requirements
variations. The results highlight the practical effectiveness of RAG-based LLM
approaches in industrial settings, offering a promising solution for improving
requirements traceability in complex automotive systems.

## Full Text


<!-- PDF content starts -->

TVR: Automotive System Requirement T raceability V alidation
and R ecovery Through Retrieval-Augmented Generation
Feifei Niu
University of Ottawa
Ottawa, Canada
fniu2@uottawa.caRongqi Pan
University of Ottawa
Ottawa, Canada
rpan099@uottawa.caLionel C. Briand
University of Ottawa
Ottawa, Canada
Lero Centre, University of Limerick
Ireland
lbriand@uottawa.ca
Hanyang Hu
Wind River
Canada
phenom.hu@gmail.comKrishna Koravadi
Aptiv
USA
krishna.koravadi@aptiv.com
Abstract
In automotive software development, as well as other domains,
traceability between stakeholder requirements and system require-
ments is crucial to ensure consistency, correctness, and regula-
tory compliance. However, erroneous or missing traceability rela-
tionships often arise due to improper propagation of requirement
changes or human errors in requirement mapping, leading to incon-
sistencies and increased maintenance costs. Existing approaches do
not address traceability between stakeholder and system require-
ments, rely on open-source data—as opposed to automotive (or
any industry) data—and do not address the validation of manual
links established by engineers. Additionally, automotive require-
ments often exhibit variations in the way they are expressed, posing
challenges for supervised models requiring training. The recent
advancements in large language models (LLMs) provide new op-
portunities to address these challenges. In this paper, we introduce
TVR, a requirement Traceability Validation and Recovery approach
primarily targeting automotive systems, leveraging LLMs enhanced
with retrieval-augmented generation (RAG). TVR is designed to
validate existing traceability links and recover missing ones with
high accuracy. We empirically evaluate TVR on automotive re-
quirements, achieving 98.87% accuracy in traceability validation
and 85.50% correctness in traceability recovery. Additionally, TVR
demonstrates strong robustness, achieving 97.13% in accuracy when
handling unseen requirements variations. The results highlight the
practical effectiveness of RAG-based LLM approaches in industrial
settings, offering a promising solution for improving requirements
traceability in complex automotive systems.
1 Introduction
Automotive systems have become increasingly complex, compris-
ing a wide range of integrated components and features such as
powertrain systems, chassis systems, infotainment systems, ad-
vanced driver assistance systems (ADAS), and electric and au-
tonomous driving systems. These components collaborate through
electronic control units (ECUs), sensors, actuators, and software,working seamlessly to deliver performance, safety, and user experi-
ence1. Modern vehicles are thus highly sophisticated multi-system
platforms, often referred to as “computers on wheels.”
To ensure high-quality software development, the automotive
industry adheres to the Automotive Software Process Improvement
and Capability Determination (ASPICE) standard [ 15,29], which
is based on the ISO 3300x norm group [ 30–33]. A fundamental
aspect of ASPICE-compliant software development is requirements
engineering, where requirements are defined at different levels,
including stakeholder requirements and system requirements. Sys-
tem requirements are derived from stakeholder requirements, with
traceability links established between them to ensure consistency
and correctness, especially in the occurrence of frequent changes.
Maintaining accurate and high-quality traceability links between
stakeholder and system requirements is indeed crucial for multiple
reasons: 1) Enhancing system consistency; ensuring that require-
ments are correctly implemented minimizes functional errors and
inconsistencies [53, 65]. 2) Preventing error propagation – Detect-
ing incorrect requirement mappings early reduces maintenance
costs and avoids large-scale rework [ 61]. 3) Ensuring regulatory
compliance, thus complying with ASPICE standards to help improve
the safety, reliability, and auditability of automotive software [ 55].
However, significant challenges have been reported by industry
engineers regarding traceability between requirements, for exam-
ple in the context of Diagnostic Trouble Code (DTC) management,
which refer to standardized vehicle error codes indicating malfunc-
tions detected by the diagnostics system [ 44,54]. According to
system engineers, the primary causes of these challenges are re-
lated to resource and time constraints, and include: 1) Improper
propagation of requirement changes: When stakeholder require-
ments are modified (e.g., new requirements are added or existing
ones are updated), the corresponding system requirements may
not be updated accordingly, leading to outdated or invalid trace-
ability links. If a requirement is deleted or merged, the original
traceability link may persist, resulting in incorrect mappings that
could cause functional inconsistencies or safety risks. 2) Errors in
requirement mapping: Due to human error, system requirements
1https://en.wikipedia.org/wiki/Automotive_electronicsarXiv:2504.15427v1  [cs.SE]  21 Apr 2025

Anonymous, xxx, xxx Niu et al.
may be incorrectly mapped to unrelated stakeholder requirements,
or traceability links are missing.
The goal of this paper is to provide automated, effective means
to address the challenges above and thus identify invalid and miss-
ing traceability links between DTC requirements. Because require-
ments are not always expressed consistently within or across projects,
we rely on large language models (LLMs), hoping they are robust
to such variations.
We are motivated in our choice by the fact that LLMs have demon-
strated significant advancements in natural language processing
(NLP) [ 12,28,34,59] and software engineering tasks [ 12,28,34],
especially code generation [ 7,9,13,47,59,68], and code summa-
rization [ 1,16,48]. Further, LLMs have also shown to be effective
for traceability link recovery [10, 58].
To address traceability issues and meet the practical needs of our
industrial partner, we propose a Retrieval-Augmented Generation
(RAG) enhanced approach: Traceability Validation and Recovery
(TVR). We investigate and leverage many state-of-the-art (SOTA)
LLMs—with Claude Sonnet 3.5 [ 2] yielding the best results—to ver-
ify whether a traceability link is valid and recover missing links be-
tween requirements. We evaluated TVR on 2,312 DTC requirement
pairs from an automotive system, achieving an overall accuracy of
98.87%. To investigate its robustness against variations in the way
requirements are written (as detailed in Section 2), we evaluated
TVR on new requirement variations. The results show that TVR
still maintains 97.13% in accuracy, demonstrating strong robust-
ness. Additionally, to address the challenge of missing links, we
applied TVR to traceability link recovery, successfully identifying
502 missing traceability links with an accuracy of 85.50%.
Our contributions include:
•Proposing TVR, a RAG-enhanced LLM approach for validat-
ing traceability links between automotive requirements.
•Evaluating the robustness of TVR on unseen requirement
variations.
•Applying TVR to recover traceability links.
•Investigating 13 SOTA LLMs across five prompt engineer-
ing strategies, as well as the impact of different similarity
measures and the number of examples on RAG performance.
The remainder of this paper is structured as follows: Section 2
defines the industrial problems we address and provides the context.
Section 3 introduces our TVR approach. The study design is detailed
in Section 4, followed by an analysis of the experimental results in
Section 5, and a discussion in Section 6. Section 7 examines threats
to validity. Section 8 reviews the state of the art in requirements
traceability research and contrasts it with our contributions. Finally,
we conclude the paper in Section 9.
2 Problem Definition
This work aims to support traceability validation and recovery be-
tween stakeholder requirements and system requirements for DTCs
within automotive systems. This section provides an overview of
DTCs and requirements traceability, with a particular focus on
stakeholder and system requirements. Though our terminology
and sanitized examples come from the automotive domain, many
other critical domains, which typically need to comply with func-
tional safety standards, have similar challenges and concepts.2.1 Diagnostic Trouble Code Requirements
2.1.1 Diagnostic Trouble Code. Diagnostic trouble codes (DTCs)
(also known as fault codes) are standardized codes used in auto-
motive systems to identify and diagnose issues within a vehicle’s
electronic control units (ECUs) [ 44,54]. These codes are generated
when a malfunction is detected by the On-Board Diagnostics (OBD)
system, which monitors various components such as the engine,
transmission, and emission systems. Example conditions that trig-
ger a DTC are implausible or erroneous signal values or signals that
were not received [ 60]. DTCs typically consist of a letter (indicat-
ing the system, e.g., P for Powertrain) followed by four digits that
provide specific details about the fault. Technicians and diagnostic
tools use these codes to pinpoint problems efficiently, aiding in
vehicle repair and maintenance.
In modern automotive systems, DTC requirements are expressed
at various levels of abstraction: stakeholder requirements and sys-
tem requirements. These requirements define the conditions un-
der which faults are detected, recorded, and communicated within
ECUs, ensuring that the system operates effectively and meets both
technical and regulatory standards. In this study, we focus on the
requirements pertaining to the setting andclearing of DTCs, cor-
responding to the conditions for so-called mature and demature
DTCs, respectively, which are further described in Section 2.1.3.
2.1.2 DTC Stakeholder Requirements. Stakeholder requirements
are the high-level needs and expectations of all parties involved
or affected by the automotive system. As shown in Figure 1, DTC
stakeholder requirements include the following basic elements:
•Trigger Condition: Specify the conditions under which a
DTC is set or cleared.
•Input Message: Define the message that triggers the setting
or clearing of the DTC.
•Mitigation Action: Include specific actions for setting the
DTC to “Present” or “Not Present”.
•Validation Rules: Establish validation rules to ensure that
DTC setting and clearing operations comply with design
standards.
•Reference Documents: Provide relevant standards and docu-
ments supporting the setting and clearing of the DTC.
When put together, all these basic elements constitute a complete
DTC stakeholder requirement. In practice, a stakeholder require-
ment may encompass multiple independent requirements as it may
include both setting and clearing DTC requirements, which follow
the same template. In such cases, we can decompose it into atomic
requirements based on these fundamental elements, ensuring that
each atomic requirement is evaluated independently for traceability.
2.1.3 DTC System Requirements. System requirements define the
detailed specifications that the automotive system must meet to
satisfy stakeholder requirements. These requirements translate
stakeholder expectations into specific, actionable objectives for the
system.
In our context, the elements of a DTC system requirement in-
clude Name ,Number ,Description ,Priority ,Enable Condition ,Mature ,
Mature Time ,Demature ,Demature Time , and others. Among these,
the most critical elements are Mature andDemature .Mature spec-
ifies the conditions that must be met for a DTC to be set, while

TVR: Automotive System Requirement T raceability V alidation and R ecovery Through Retrieval-Augmented Generation Anonymous, xxx, xxx
/grStakeholder Requirements
VARIATION 1 IfTrigger_Condition = “RUN”, and the module Mdoes NOT receive the message MESSAGE_1 for a certain number
of message cycles defined in the “ Reference_Document ”, then the module Mshall:
Set the DTC to “Not Present” according to the appropriate validation rules contained in the “ Reference_Document ”.
VARIATION 2 IfTrigger_Condition = “RUN”, the internal signal MODULE_MODE != “SIGNAL_1_Fail ”, then the module Mmust:
Set the DTC to “Not Present” according to the appropriate validation rules contained in the “ Reference_Document ”.
VARIATION 3 IfTrigger_Condition = “RUN”, and the module Mdoes not detect a Plausibility Fault on a signal within the message
“MESSAGE_2 ”, then the module Mmust:
Set the DTC to “Not Present” according to the appropriate validation rules contained in the “ Reference_Document ”.
VARIATION 4 IfTrigger_Condition = “RUN”, the module Mdetermines there is a failure in SIGNAL_2 , then the module Mshall:
Set the INTERNAL_FLAG = “Faulted”.
Set the DTC to “Present” according to the appropriate validation rules contained in the “ Reference_Document ”.
Figure 1: Fictitious Variations of Stakeholder Requirements
System Requirement
i f ( ENABLE_COMPONENT i s e n a b l e d ) {
i f ( Missing_Msg_MESSAGE_1 | |
Missing_Msg_MESSAGE_2 | |
. . . ) {
LostComm_Module_M = TRUE ;
}
}
Figure 2: Example of Sanitized DTC System Requirement
Demature defines the conditions required for the DTC to be cleared.
Mature andDemature should be negations of each other.
Figure 2 presents an example of a Mature condition from a system
requirement. Both Mature andDemature processes start by verify-
ing that the relevant components are enabled. This ensures that the
system is actively monitoring the specified components and is pre-
pared to conduct further checks if the components are available and
operational. Next, the system verifies proper communication and
ensures that expected data has been received, confirming that the
system is functioning correctly. If a malfunction is detected, the sys-
tem sets the corresponding DTCs; otherwise, the DTCs are cleared.
2.1.4 DTC Types. There are different types of ECU failures, cor-
responding to various DTC types, and can be broadly categorized
into: internal failures and network failures [ 51,60]. Network failures
can be further divided into two types [51, 60]:
•Lost Communication: This occurs when an ECU fails to
receive an expected message from a source ECU. The absence
of an expected message is categorized as a Missing Message
failure mode.
•Implausible Data: This occurs when an ECU receives an
expected message but detects untrustworthy data that is
inconsistent, unrealistic, or beyond the expected domain.
Different types of DTCs follow distinct patterns. In this study,
we primarily focus on the two types of network failures due to dataavailability: “Lost Communication” and “Implausible Data”. Figure 1
illustrates, with sanitized examples, the variations we observed
across stakeholder requirements for network failures. Variation
1 represents a “Lost Communication” failure, where the expected
message (“ MESSAGE_1 ”) was missing for a certain number of cycles.
Variations 2, 3, and 4 correspond to different cases of “Implausi-
ble Data” faults, where the ECU receives untrustworthy data. In
addition, Variations 1, 2, and 3 describe conditions for “Mature”,
which involve setting DTCs, while Variation 4 describes “Demature”,
which involves clearing DTCs.
2.2 Requirements Traceability
Requirements traceability refers to the ability to track and document
the lifecycle of a requirement—both forward and backward—from
its origin through development, implementation, and usage [ 18]. In
the context of automotive systems, traceability helps ensure that
stakeholder needs are effectively aligned with the system design. It
plays a critical role in confirming that the system meets customer
expectations and regulatory standards. Moreover, traceability pro-
vides a clear audit trail for quality assurance, helping to detect
gaps, inconsistencies, or changes in requirements throughout the
development process.
As discussed in Section 2.1, DTC requirements come in different
types, each following distinct patterns. Moreover, both stakeholder
and system requirements exhibit variations in the way they are
expressed, with the possibility of encountering unseen variations
in the future. This presents a significant challenge: A general so-
lution capable of handling diverse and unseen variations is
essential . Traditional NLP approaches [ 6,56,61], based on super-
vised learning of observed variations, are challenged by unseen
variations. Moreover, a comparison of stakeholder requirements
(Figure 1) and system requirements (Figure 2) raises another chal-
lenge: Our traceability validation problem cannot be simply
achieved by calculating text similarity or finding word over-
laps. Since the system requirement involves checking whether
multiple messages or signals are missing, while a stakeholder re-
quirement only addresses the handling of a single message, a system

Anonymous, xxx, xxx Niu et al.
requirement may be traced to several stakeholder requirements,
each one covering a message or signal in the system requirement.
Further, as shown in Figure 1 and Figure 2, messages and signals
only differ slightly in terms of character strings.
To address these two challenges, this paper proposes TVR, a
method for verifying requirements traceability for automotive sys-
tems. TVR leverages the exceptional natural language understand-
ing capabilities of LLMs. Instead of relying on the general concept
of traceability links [ 20], we ask LLMs to verify whether our system
requirement covers the message or signal specified in the stake-
holder requirement. We guide the LLM to focus specifically on the
message or signal, rather than other parts of the requirement, to
avoid confusion.
3 TVR
TVR is a retrieval-augmented approach that leverages LLMs for
requirements traceability validation and recovery. Given that ob-
served variations are specific to the domain and context, we do
not retrieve relevant knowledge from an external knowledge base.
Instead, inspired by [ 40,67], we retrieve similar requirement pairs
to serve as demonstrations in the prompt. Given a language model
𝐺, a training set
𝐷train=⟨𝑠𝑡𝑎𝑘𝑒𝑅𝑒𝑞 𝑖,𝑠𝑦𝑠𝑅𝑒𝑞 𝑖,𝑙𝑎𝑏𝑒𝑙 𝑖⟩|𝑖=1,2,...,𝑁,
where𝑙𝑎𝑏𝑒𝑙 𝑖∈{𝑌𝑒𝑠,𝑁𝑜}
and a test case
𝑥test={𝑠𝑡𝑎𝑘𝑒𝑅𝑒𝑞 test,𝑠𝑦𝑠𝑅𝑒𝑞 test},
TVR retrieves similar examples from 𝐷train to assist𝐺in predicting
the label for 𝑥test.
The overall workflow of TVR, illustrated in Figure 3, consists of
two key components: a retriever and a generator . The retriever
identifies the most similar examples from 𝐷train based on their
similarity to 𝑥testand provides them as input to the generator.
The generator then utilizes these retrieved examples to assess the
validity of the traceability link between 𝑠𝑡𝑎𝑘𝑒 testand𝑠𝑦𝑠𝑅𝑒𝑞 test.
Figure 3: Overall Framework of TVR.
3.1 Retriever
The retriever identifies and retrieves the most similar examples from
𝐷train that closely match 𝑥test, serving as input to the generator.
The retriever operates in two stages. First, it generates embeddings
for each data point ( ⟨stakeReq,sysReq⟩) in𝐷train as well as for 𝑥test.
Then, using a similarity-based retrieval mechanism, it selects the𝑘×2most similar data points from 𝐷train, comprising 𝑘valid and
𝑘invalid examples.
Due to data privacy constraints, we utilize Amazon Titan2, the
only model provided by our industry partner to obtain embeddings
for pairs⟨stakeReq,sysReq⟩. We then leverage the FAISS library3
to compute cosine similarity and efficiently retrieve the Top- kmost
similar data points from both valid and invalid examples.
3.2 Generator
The generator plays a crucial role in evaluating the traceability
between a given pair ⟨𝑠𝑡𝑎𝑘𝑒𝑅𝑒𝑞 test,𝑠𝑦𝑠𝑅𝑒𝑞 test⟩. This process lever-
ages retrieval-augmented, in-context learning powered by LLMs,
e.g., Claude 3.5 Sonnet4, the best model as presented in Section 5.
It makes use of the top 𝑘×2similar examples obtained by the
retriever. These retrieved examples are incorporated into the gen-
erator’s prompt as contextual examples to guide its response gen-
eration. Using this augmented prompt, the generator analyzes the
relationship between the stakeholder requirement ( 𝑠𝑡𝑎𝑘𝑒𝑅𝑒𝑞 test)
and the system requirement ( 𝑠𝑦𝑠𝑅𝑒𝑞 test) and generates a response
indicating whether their traceability link is valid or not.
As shown in Figure 4, the input of the generator includes:
(1)Instruction: The specific task that the model needs to per-
form.
Rationale: We define our task as verifying whether a system
requirement covers a stakeholder requirement. While trace-
ability validation is a broad concept, we narrow its scope
in our context to focus specifically on whether the system
requirement covers the message or signal in the stakeholder
requirement. This refinement is necessary because a single
system requirement may correspond to multiple stakeholder
requirements in our case.
Our Prompt: See Figure 4, Lines 1–2.
(2)Context : External information or additional context that
can steer the model to better responses.
Rationale: In our context, we provide the 𝑘×2most similar
examples, including 𝑘valid and𝑘invalid ones, to help LLMs
learn to understand them.
Our Prompt: See Figure 4, Line 3.
(3)Input Data: the input or question we aim to answer.
Rationale: We use XML tags to clearly separate different
parts of the input data and ensure the prompt is well struc-
tured5.
Our Prompt: See Figure 4, Lines 4–7.
(4)Output Indicator: The type or format of the output.
Rationale: For experimental purposes, we only require the
model to output a predicted label, so the model only needs
to respond with “Yes” or “No.” Note that in practice, if an
explanation from LLMs is required, this sentence should be
adjusted to enable LLMs to generate a step-by-step reasoning
and analysis process.
2https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-
models.html
3https://github.com/facebookresearch/faiss
4https://aws.amazon.com/bedrock/claude/
5https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-
xml-tags

TVR: Automotive System Requirement T raceability V alidation and R ecovery Through Retrieval-Augmented Generation Anonymous, xxx, xxx
/eybardInput Prompt
Line 1: Please check if the message or signal from the stake-
holder requirement is correctly covered by the system
requirement.
Line 2: Please focus only on verifying the message orsig-
nalmentioned, without considering other parts of the re-
quirement.
Line 3: Example: <example>{ k×2 examples }</example>
Line 4: Now evaluate the following step by step and only
respond with either “Yes” or “No”:
Line 5: Stakeholder Requirement: <stakeholder> { stakeReq }
</stakeholder>
Line 6: System requirement: <system> { sysReq }</system>
Line 7: Response:
Figure 4: Input Prompt.
Our Prompt: “only respond with either ‘Yes’ or ‘No’.” (Line
4)
With the above prompt, the generator analyzes the input require-
ment pair according to the instructions and provided contextual
examples, and then responds with either “Yes” or “No” for the given
pair⟨𝑠𝑡𝑎𝑘𝑒𝑅𝑒𝑞 test,𝑠𝑦𝑠𝑅𝑒𝑞 test⟩.
3.3 Traceability Recovery
To recover potentially missing traceability links between any pair
of stakeholder requirement and system requirement in the dataset,
we first pair every stakeReq with every sysReq that lacks a trace link.
This results in a number of pairs approaching the Cartesian product
of the two sets (excluding existing traceability links), which would
be computationally expensive and time-consuming to analyze.
To address this challenge, we apply the following preprocess-
ing steps to reduce computational overhead:
Step 1: Variations Matching: As discussed in Section 2, in the stud-
ied dataset, stakeholder requirements can be grouped into
four distinct variations, each following a specific template,
while system requirements are classified into two categories
characterized by unique syntactic patterns. A traceability
link can only exist between requirements that belong to
the same type of DTC. For example, a stakeholder require-
ment for “Lost Communication” can only be linked to a sys-
tem requirement of the same DTC type. By excluding cross-
category mismatches, we substantially reduce the number
of pairs being considered.
Step 2: Condition Matching: Each atomic stakeholder requirement
corresponds exclusively to either a mature condition or a
demature condition in the system requirement. Based on
this distinction, we group stakeReq andsysReq accordingly
and only match those within the same condition type, thus
further eliminating irrelevant pairs.
Step 3: Message Overlap Matching: If a stakeholder requirement
and a system requirement do not share any message overlap,
there is no traceability between them. To check this, we first
tokenize the stakeholder and system requirements, remove
stop words (customized for our domain), and extract themessages. We then compare these messages and retain only
the pairs where at least one common message exists between
them. This step significantly reduces the number of pairs
requiring further validation.
Through these three steps, we effectively reduce the number
of candidate pairs while preserving all potential missing links. The
filtered pairs are then used to validate the following hypothesis ( H):
A valid traceability link, denoted as traceLink , exists
for⟨stakeReq,sysReq⟩.
We use TVR to validate this hypothesis. If His confirmed for
a pair, it implies that the traceLink is missing and should be added
to the dataset. Conversely, if His not confirmed, it indicates that
no traceability link exists between stakeReq andsysReq . Through
this process, we systematically recover missing traceability links in
the dataset.
4 Study Design
In this section, we present our research questions, the LLMs and
prompt engineering strategies evaluated, the dataset used for vali-
dating and recovering requirements traceability, and the evaluation
metrics employed to assess the performance of TVR.
4.1 Research Questions
RQ1: What is the performance of SOTA LLMs, with zero-shot
prompting, on requirements traceability validation?
This RQ aims to evaluate and compare the performance of
available SOTA LLMs, including Llama, Claude, Titan, and
Mistral, with zero-shot prompting.
RQ2: What is the best prompt strategy using the best LLM for
requirements traceability validation?
This RQ investigates and compares the performance of vari-
ous prompt engineering strategies applied to requirements
traceability validation, including zero-shot, few-shot, chain-
of-thought (CoT), self-consistency, and RAG.
RQ3: How robust is TVR for requirements traceability validation
in the presence of unseen requirements variations?
One of our main motivations in relying on LLMs is that we
expect them to be more robust to such variations. Due to
the numerous variations that are typically present in stake-
holder requirements, this RQ primarily evaluates the per-
formance of TVR across unseen variations in stakeholder
requirements.
RQ4: What is the performance of TVR on traceability link recov-
ery between stakeholder requirements and system require-
ments?
LLMs can not only be used for traceability validation but
also for missing links recovery by determining if there is
valid traceability between any two requirements that are not
linked. This research question leverages our TVR approach
to recover missing links and evaluates its accuracy.

Anonymous, xxx, xxx Niu et al.
4.2 Large Language Models and Prompts
Engineering
4.2.1 LLMs. In this study, due to our industry partner’s data pri-
vacy concerns, we were unable to use LLMs such as the GPT se-
ries [ 49,50]. Instead, we were provided with 13 SOTA LLMs avail-
able through the Amazon Bedrock service, which include Claude,
Llama, Mistral, and Titan. Extensive experiments were conducted
using these LLMs.
•Claude [ 25] is a family of LLMs developed by Anthropic.
These models offer several notable advantages: (1) a larger
context window, (2) strong performance across various tasks,
and (3) not storing any user input or output data.6The
evaluated models include Claude 3.5 Sonnet, Claude 3 Sonnet,
Claude 2, Claude Instant, and Claude 3 Haiku.
•Llama [ 27] is a family of autoregressive LLMs released by
Meta AI, which provides models ideal for natural language
tasks like Q&A and reading comprehension. The evaluated
models are Llama 3 8B and Llama 3 70B.
•Mistral AI [ 26] provides models with publicly available weights
supporting a variety of use cases from text summarization,
text classification, and text completion to code generation
and code completion. The evaluated models include Mistral
7B, Mixtral 8x7B, and Mistral Large 2402.
•Amazon Titan [ 24] foundation models are a family of foun-
dation models pre-trained by AWS on large datasets, making
them powerful, general-purpose models built to support a
variety of use cases. The evaluated models include Titan Text
Premier, Titan Text Express, and Titan Text Lite.
4.2.2 Prompt Engineering. We select the following SOTA prompt
engineering strategies for evaluating LLMs:
•Zero-shot Prompting [ 5]. Zero-shot prompting directly in-
structs the model to complete a task without providing
demonstrations or additional context.
•Chain-of-Thought Prompting (CoT) [ 64]. CoT enhances the
complex reasoning capabilities of LLMs by incorporating
intermediate reasoning steps.
•Few-shot Prompting [ 5]. Few-shot prompting improves model
performance by including a small number of task demon-
strations in the prompt. These demonstrations serve as con-
ditioning, guiding the model’s response for subsequent in-
puts [5, 66].
•Self-Consistency [ 63]. Self-consistency refines CoT prompt-
ing by sampling multiple diverse reasoning paths and select-
ing the most consistent answer. Self-consistency improves
the reliability of CoT prompting, especially for arithmetic
and commonsense reasoning, by leveraging agreement across
multiple reasoning trajectories [63].
•Retrieval-Augmented Generation (RAG) [ 39]. RAG enhances
LLMs’ capability of solving knowledge-intensive tasks by
integrating information retrieval with text generation. It
retrieves relevant external knowledge sources to provide
additional context, enabling models to generate more accu-
rate and up-to-date responses without requiring full retrain-
ing [14, 57].
6https://www.ibm.com/think/topics/claude-aiNote that in our experiments, each prompting strategy builds
upon the previous one. The CoT reasoning extends Zero-Shot, while
few-shot prompting enhances CoT by incorporating examples in
the prompt. The self-consistency strategy further refines few-Shot
prompting by applying a majority voting method. RAG improves
few-shot prompting by selecting more relevant examples.
4.3 Dataset
4.3.1 Data Collection. To evaluate the performance of LLMs in
autonomous requirements traceability validation and recovery, we
collected a dataset of DTC requirements from our industry partner.
We began by gathering high-quality DTC system requirements,
including the “Lost Communication” and “Implausible Data” DTC
system requirements, which are among the most common and
critical types of DTC requirements. Subsequently, we identified
the stakeholder requirements linked to these system requirements,
yielding a total of 1,320 stakeholder requirements.
4.3.2 Data Cleaning. After obtaining the system requirements and
stakeholder requirements, we performed data cleaning to ensure
the dataset’s high quality.
First, if a stakeholder requirement contained multiple atomic
requirements, we divided it (based on Section 2) so that each stake-
holder requirement contained only a single atomic requirement. We
then established the following two constraints for dataset cleaning:
(1) All stakeholder requirements must be complete.
(2)Every DTC system requirement must have been linked to at
least one stakeholder requirement.
To check the completeness of stakeholder requirements, we
defined regular expressions to validate their structure7. Specifi-
cally, each atomic requirement must include a trigger condition , an
input message , and a mitigation action . We filtered out both stake-
holder and system requirements that did not meet the above two
constraints. In practice, such requirements should be flagged as is-
sues requiring correction and must be addressed before performing
requirement traceability validation.
Finally, the dataset includes a total of 1,320 atomic stakeholder
requirements linked to 48 system requirements through 2,132 ex-
isting traceability links. The dataset contains 24 “Lost Communi-
cation” system requirements and 24 “Implausible Data” system
requirements.
4.3.3 Data Annotation. We manually annotated 2,132 traceability
links between stakeholder requirements and system requirements
pairs following the guidelines provided by system engineers em-
ployed by our industry partner. 1,913 traceability links were con-
firmed as valid, while 219 were identified as invalid, indicating that
no traceability link should exist between them.
4.4 Evaluation Metrics
4.4.1 Traceability Validation. To assess the performance of TVR
in autonomous requirements traceability validation, we employ
standard metrics, including accuracy, precision, recall, and F1 score.
𝑇𝑃is the number of traceability links that are correctly identified
as valid.𝐹𝑁is the number of traceability links that are incorrectly
7while we acknowledge that more advanced approaches for completeness checking
exist, this is not the primary focus of our study

TVR: Automotive System Requirement T raceability V alidation and R ecovery Through Retrieval-Augmented Generation Anonymous, xxx, xxx
identified as invalid. 𝐹𝑃is the number of traceability links that
are incorrectly identified as valid. 𝑇𝑁is the number of traceability
links that are correctly identified as invalid.
Accuracy is the proportion of correctly identified traceability
links (both valid and invalid) out of all traceability links. Precision is
the proportion of correctly identified valid links ( 𝑇𝑃) out of all links
identified as valid by the model ( 𝑇𝑃+𝐹𝑃). It measures how reliable
the model’s predictions of valid links are. Recall is the proportion
of correctly identified valid links ( 𝑇𝑃) out of all actual valid links
in the ground truth ( 𝑇𝑃+𝐹𝑁). It measures the model’s ability to
find all the valid links. F1 score is the harmonic mean of precision
and recall. It balances the trade-off between precision and recall,
providing a single measure of the model’s performance.
4.4.2 Traceability Recovery. To evaluate the performance of TVR
in autonomous requirements traceability recovery, we manually
verify their correctness (Formula 1). The evaluation metric is the
proportion of correct traceability links, calculated as the number of
links confirmed as correct through human verification divided by
the total number of links identified by the model.
𝐶𝑜𝑟𝑟𝑒𝑐𝑡𝑛𝑒𝑠𝑠 =𝑁𝑢𝑚𝑏𝑒𝑟𝑜𝑓𝐶𝑜𝑟𝑟𝑒𝑐𝑡𝐿𝑖𝑛𝑘𝑠𝑉𝑒𝑟𝑖𝑓𝑖𝑒𝑑𝑏𝑦𝐻𝑢𝑚𝑎𝑛𝑠
𝑇𝑜𝑡𝑎𝑙𝑁𝑢𝑚𝑏𝑒𝑟𝑜𝑓 𝐿𝑖𝑛𝑘𝑠𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑𝑏𝑦𝑡ℎ𝑒𝑀𝑜𝑑𝑒𝑙
(1)
5 Experiment Results
5.1 Performance of SOTA LLMs on
Requirements Traceability Validation (RQ1)
Approach. This RQ examines the performance of TVR using SOTA
LLMs with zero-shot prompting for requirements traceability valida-
tion, focusing on models available to our industry partner, including
Claude, Llama, Mistral, and Amazon Titan. Specifically, the zero-
shot prompting strategy involves providing the task description in
the prompt without giving the model any examples.
Results. The experimental results are summarized in Table 1. Claude
models generally outperform other model series, with Claude 2
(90.57%), Claude Instant (84.43%), and Claude 3.5 Sonnet (79.55%)
achieving the highest accuracy scores. In contrast, Llama 3 70B
(10.19%) and Mistral 8 ×7B (17.23%) show notably lower accuracy,
indicating weaker generalization capabilities for this task.
For valid pairs, most models exhibit high precision (above
90%), but recall varies significantly: Mistral Large 2402 achieves the
highest precision (100%) but has a recall of only 0.73%, leading to a
low F1 score of 1.44%. This suggests the model is highly conserva-
tive, prioritizing precision over recall, but missing many valid pairs.
Claude 2 achieves the best F1 score (95.04%), with well-balanced
precision (90.64%) and recall (99.90%), demonstrating strong recall
and overall predictive stability. Llama 3 70B and Mistral Large 2402
exhibit poor recall (only 0.95% and 0.73%, respectively), indicating
difficulty in capturing valid cases. Their F1 scores (1.88% and 1.44%)
further confirm their limitations. Within the Claude series, Claude 2
outperforms other models, achieving the highest F1 score (95.04%),
while Claude Instant follows with an F1 score of 91.51%.
However, detecting invalid pairs proves significantly more
challenging, as evidenced by the lower F1 scores across all models.
Claude 2 has the highest invalid precision (71.43%) but suffers fromextremely low recall (2.45%), resulting in an F1 score of only 4.74%,
indicating it misses most invalid pairs. Mistral Large 2402 achieves
the highest recall for invalid cases (100.00%) but at a cost of low
precision (9.63%), leading to an F1 score of 17.57%. This suggests it
identifies all invalid pairs but at the expense of many false positives.
Among weaker performers, Claude Instant has an invalid recall of
only 4.90% and an F1 score of 5.68%. Llama 3 70B achieves high
recall (96.53%) but with low precision (9.44%), leading to a low F1
score of 17.20%.
The reasons for the models’ poor performance include: (1)
Even when explicitly instructed to focus on determining whether
a message or signal from the stakeholder requirement is covered
by the linked system requirement, without domain knowledge,
LLMs still struggle to accurately understand and identify the key
relevant message or signal in the requirements but rather focus
on the wrong aspects, leading to incorrect predictions. (2) LLMs
tend to check for content consistency between requirements and
often misjudge the two requirement descriptions as inconsistent
due to differences in wording or structure. Since invalid pairs are
few in number, the small denominator in score calculations, such
as precision and recall, can lead to unreliable scores.
Answering RQ1: Results suggest unsatisfactory accuracy for all
models with zero-shot prompting for requirements traceability vali-
dation, especially for invalid links, thus underscoring the necessity
of employing more advanced prompt engineering strategies.
Table 1: Experimental Results of SOTA Models with Different
Prompting Strategies.
Prompt Model Acc(%)Valid Invalid
Pre(%) Recall(%) F(%) Pre(%) Recall(%) F(%)
Zero-shotClaude 3.5 Sonnet 79.55 93.42 83.25 88.04 21.98 44.61 29.45
Claude 3 Sonnet 32.08 96.51 25.83 40.75 11.51 91.18 20.44
Claude 2 90.57 90.64 99.90 95.04 71.43 2.45 4.74
Claude Instant 84.43 90.22 92.84 91.51 6.76 4.90 5.68
Claude 3 Haiku 49.72 91.00 49.27 63.93 10.11 53.92 17.03
Llama 3 8B 72.70 91.09 77.39 83.68 11.74 28.43 16.62
Llama 3 70B 10.19 72.00 0.95 1.88 9.44 96.53 17.20
Mistral 7B 57.09 91.96 57.45 70.72 12.08 53.77 19.72
Mixtral 8x7B 17.23 89.76 9.62 17.38 9.43 89.55 17.07
Mistral Large 2402 10.23 100.00 0.73 1.44 9.63 100.00 17.57
Titan Text Premier 20.68 92.17 13.43 23.45 9.83 89.22 17.71
Titan Text Express 49.81 90.17 49.95 64.29 9.30 48.53 15.62
Titan Text Lite 70.83 91.49 74.69 82.24 12.54 34.31 18.37
CoTClaude 3.5 Sonnet 76.31 98.70↑ 74.79 85.10 27.57↑ 90.69↑ 42.29↑
Claude 3 Sonnet 22.92 95.24 15.60 26.81 10.33 92.57↑ 18.59
Claude 2 66.47 94.39↑ 66.79 78.22 17.24 63.54↑ 27.12↑
Claude Instant 86.16↑ 90.36↑ 94.81↑ 92.53↑ 8.26↑ 4.41 5.75↑
Claude 3 Haiku 45.55↑ 94.97↑ 41.62 57.88 13.44↑ 80.42↑ 23.03↑
Llama 3 8B 86.25↑ 90.56 94.73↑ 92.59↑ 6.00 3.30 4.26
Llama 3 70B 35.32↑ 98.17↑ 28.94↑ 44.70↑ 12.51↑ 94.97 22.11↑
Mistral 7B 73.15↑ 92.35↑ 76.63↑ 83.76↑ 15.65↑ 40.59 22.59↑
Mixtral 8x7B 14.53 96.43↑ 5.80 10.94 9.82↑ 97.95↑ 17.84↑
Mistral Large 2402 10.27↑ 100.00 0.78↑ 1.54↑ 9.64↑ 100.00 17.58↑
Titan Text Premier 25.70↑ 92.57↑ 19.40↑ 32.08↑ 10.07↑ 85.29 18.01↑
Titan Text Express 54.41↑ 90.78↑ 55.19↑ 68.65↑ 10.00↑ 47.06 16.49↑
Titan Text Lite 69.93↑ 90.70↑ 74.38 81.73 10.34 27.94 15.10
Few-shotClaude 3.5 Sonnet 97.47↑ 98.70 98.50↑ 98.60↑ 86.06↑ 87.75 86.89↑
Claude 3 Sonnet 90.81↑ 94.00 95.95↑ 94.97↑ 52.44↑ 42.16 46.74↑
Claude 2 41.44 97.88↑ 36.01 52.66 13.29 92.65↑ 23.25
Claude Instant 90.95↑ 96.82↑ 93.05 94.90↑ 51.97↑ 71.08↑ 60.04↑
Claude 3 Haiku 92.73↑ 93.65 98.65↑ 96.08↑ 74.26↑ 36.76 49.18↑
Mistral 7B 90.01↑ 90.62 99.22↑ 94.73↑ 28.57↑ 2.94 5.33
Mixtral 8x7B 30.66↑ 99.13↑ 23.56↑ 38.07↑ 11.90↑ 98.03↑ 21.23↑
Mistral Large 2402 61.35↑ 98.94 57.88↑ 73.04↑ 19.12↑ 94.12 31.79↑
Self-Consistency Claude 3.5 Sonnet 97.61↑ 98.4 98.96↑ 98.68↑ 89.64↑ 84.8 87.15↑
TVR (RAG) Claude 3.5 Sonnet 98.87↑ 99.28↑ 99.48↑ 99.38↑ 95.00↑ 93.14↑ 94.06↑

Anonymous, xxx, xxx Niu et al.
5.2 Best Prompt Strategy for Requirements
Traceability Validation (RQ2)
Approach. This RQ explores the performance of different prompt
engineering strategies for the requirement traceability validation
task, specifically including CoT, few-shot, self-consistency, and
RAG.
For few-shot prompting, we consider all four distinct varia-
tions of stakeholder requirements presented in our context, each
corresponding to either mature or demature conditions of the linked
system requirements. Additionally, the traceability link between
each stakeholder requirement and system requirement pair can be
either valid or invalid. Given these factors, our dataset contains 16
possible combinations. To maximize the effectiveness of LLMs, we
randomly selected one example from each combination, resulting
in a total of 16 examples for the prompt. However, the available
Llama series models (Llama 3 8B and Llama 3 70B) and Amazon
Titan series models (Text Premier, Text Express, and Text Lite) have
a limited token capacity (8K), which is insufficient for a 16-shot
prompt. Consequently, we excluded these five models from our few-
shot prompt experiments, leaving eight models for requirements
traceability validation. For these models, we include all 16 examples
in the prompt.
Based on the results shown in Table 1, we select the best-
performing model (i.e., Claude 3.5 Sonnet) for few-shot prompting
and then evaluate self-consistency on this model. Specifically, we
run the model ten times and apply a majority voting method, se-
lecting the most frequent output among the ten results as the final
output of the model [63].
For RAG, we also use Claude 3.5 Sonnet and then use the
approach described in Section 3 to retrieve the 𝑘×2most sim-
ilar examples as part of the prompt. Due to the limited amount
of manually labeled ground truth data, we employ leave-one-out
cross-validation [ 21] in our experiment to make efficient use of
the data and ensure realistic estimates. In each iteration, one re-
quirements pair is used as test, while the remaining 𝑛−1pairs
serve as the retrieval database. We acknowledge that, in practice,
companies can obtain a larger volume of high-quality labeled data
to enhance the retriever’s database. For the retriever component,
we experimentally compare two similarity measures, namely co-
sine similarity and Euclidean distance, and evaluate the impact of
different values of k, ranging from 1 to 8.
Results. The experiment results of different models with various
prompt engineering strategies are shown in Table 1. We use “ ↑”
to highlight improved scores compared to the previous prompt
strategy in the table.
The experimental results indicate that incorporating CoT rea-
soning significantly enhances recall for both valid and invalid cases
across most models. Claude 3.5 Sonnet, Claude 2, and Claude 3
Haiku exhibit substantial improvements in invalid recall, leading
to higher F1 scores. Meanwhile, Llama 3 8B, Llama 3 70B, Mistral
7B, and Amazon Titan Text Premier show notable gains in valid F1
scores. However, for Claude 3 Sonnet, Llama 3 8B, and Amazon Ti-
tan Text Lite, invalid F1 scores decreased due to a drop in precision
when detecting invalid pairs.
Comparing the results of few-shot prompting with CoT prompt-
ing, we can observe that adding examples to the prompt effectivelyimproves the F1 score for most models in both categories (valid
and invalid pairs), except for Claude 2 and Mistral 7B. Apart from
Claude 2, all models show a significant improvement in precision
for the invalid category. This demonstrates that including examples
in the prompt helps models better distinguish the valid and invalid
requirement pairs, with significant improvements in the precision
for the invalid pairs. However, the drawback of adding too many
examples is that it increases the length of the prompt, which may
exceed the token limit of certain models.
For few-shot prompting, the best-performing model is Claude
3.5 Sonnet. It outperforms all other models in terms of average per-
formance across both categories. It achieves over 98% in precision,
recall, and F1 score for the valid category, and over 86% for invalid
category. However, applying the self-consistency method to Claude
3.5 Sonnet does not significantly improve its overall performance,
as the F1 score improvement (in percentage) in both categories
remains within 3%.
1 2 3 4 5 6 7 8
K86889092949698100F1 score (%)
Euclidean Valid
Euclidean Invalid
Cosine Valid
Cosine Invalid
Figure 5: F1 Score Comparison for Euclidean and Cosine
Similarity Across Different Values of K.
For RAG, we first compare the performance of TVR with dif-
ferent Ks (i.e., the number of examples provided in the prompt).
Figure 5 illustrates the F1 score curves for both categories as K
varies, using Euclidean and cosine similarity measures. The results
show that initially, as Kincreases, TVR’s performance improves in
both categories, with a particularly significant improvement in the
invalid category. When Kreaches 3, the F1 score peaks for both
categories, with the cosine similarity yielding a slightly higher F1
score than Euclidean distance. However, as Kcontinues to increase,
TVR’s performance slightly decreases. This may result from an
excessive number of examples, including dissimilar ones, that are
potentially confusing the model and thus lowering accuracy. There-
fore, the optimal configuration of RAG-based TVR is using cosine
similarity with K=3.
In conclusion, TVR using Claude 3.5 Sonnet with RAG achieved
the best performance in requirements traceability validation, with
the highest accuracy of 98.87% across all configurations. It also
demonstrated high precision, recall, and F1 scores, reaching over
99% for valid traceability links and over 93% for invalid ones, high-
lighting the effectiveness in retrieving similar examples and includ-
ing them in the prompt, thus helping the LLMs to better understand
the requirement pair to be verified.

TVR: Automotive System Requirement T raceability V alidation and R ecovery Through Retrieval-Augmented Generation Anonymous, xxx, xxx
Answering RQ2: Claude 3.5 Sonnet with RAG achieved the best
performance in requirements traceability validation when using
cosine similarity with K set to 3. Its high accuracy makes it a viable
solution for traceability validation in practice.
5.3 Robustness to Unseen Requirements
Variations (RQ3)
Approach. This RQ evaluates the robustness of TVR to unseen
variations of stakeholder requirements, a crucial aspect in practice.
To ensure a comprehensive evaluation, we employ cross-validation
by categorizing all requirement pairs based on their variations and
evaluating across all four categories in our dataset. Specifically, for
each requirement pair to be validated, our retriever component
first identifies its variation category and then retrieves the most
similar examples from the remaining three variation categories,
thus emulating situations where new variations are encountered.
Results. Table 2 presents the TVR robustness evaluation results.
Accuracy results suggest TVR’s strong generalization ability across
variation categories for valid pairs, with an F1 score of 98.43%.
However, TVR struggles with invalid pairs, exhibiting a lower recall
of 74.88%.
Acc(%)Valid Invalid
Pre(%) Recall(%) F(%) Pre(%) Recall(%) F(%)
All 97.13 97.40 99.48 98.43 93.83 74.88 83.29
V1 93.96 94.37 98.75 96.51 90.79 67.65 77.53
V2 98.78 98.81 99.89 99.35 98.28 83.82 90.48
V3 98.44 98.59 99.76 99.18 95.24 76.92 85.11
V4 92.59 95.00 95.00 95.00 85.71 85.71 85.71
Table 2: Robustness Evaluation Results.
Among variation categories, V4 shows the lowest accuracy
(92.59%), with F1 scores of 95.00% for valid pairs and 85.71% for
invalid pairs. V1 follows with an accuracy of 93.96%, showing a
significant drop in recall for invalid pairs (67.65%), suggesting fre-
quent misclassifications of invalid pairs as valid. In contrast, results
for V2 and V3 are similar, with high accuracy rates of 98.78% and
98.44%, respectively. For V2, TVR demonstrated strong adaptability,
excelling in both valid (F1 score: 99.35%) and invalid pairs (F1 score:
90.48%), whereas TVR achieved a lower recall for the invalid pairs
(76.92%) of V3.
These results may be attributed to two factors: (1) Variability
in category differences affects the quality of retrieved examples in
guiding TVR. Specifically, V1 belongs to the “Lost Communication”
type of DTCs, whereas the remaining three variations fall under the
“Implausible Data” category. This discrepancy likely contributed
to TVR’s poorer prediction performance during cross-validation
for V1. (2) Certain variations exhibit greater complexity, making
them more challenging for the LLM to understand and predict
accurately. Although V2, V3, and V4 all belong to the “Implausible
Data” category, our analysis revealed that V4 is more complex than
other variations, containing more messages and conditions to be
verified. Moreover, the template for V4 is not strictly fixed, leading
to slightly lower accuracy during cross-validation. Furthermore, V4
has only 27 cases, resulting in a small denominator in the evaluation
metrics, and thus lower reliability in results.Answering RQ3: TVR achieved an overall accuracy of 97.13% on
robustness evaluation, indicating its strong generalization ability
across unseen variation categories, with more challenges for certain
variation categories that are more distinct and invalid pairs.
5.4 Performance on Requirements Traceability
Recovery (RQ4)
Approach. As explained in Section 3.3, we employ a three-step
pre-filtering process to reduce the number of requirement pairs we
consider could have a missing link. For the remaining pairs, TVR
predicts whether there is a valid traceability link between them.
We then calculated the Correctness of pairs predicted as ‘Yes’.
Results. The three stages of pre-filtering yielded, in turn, 30,598,
14,494, and 1,919 requirement pairs. This demonstrates the effec-
tiveness of our three-step pre-filtering, which significantly reduces
infeasible pairs. Then 502 pairs out of 1,919 pairs are predicted to
have valid traceability links. After manually verifying all 502 pairs,
we obtained a Correctness of 85.50%, suggesting that most retrieved
links are correct, with a limited number of false positives.
We conducted an error analysis and found that all 73 pre-
diction errors stemmed from the same issue. In the stakeholder
requirement, TVR first verifies whether the trigger condition is set
to “RUN”, followed by checking the input message regarding setting
or clearing the DTC. Since a trigger condition is also a message, if
the corresponding system requirement includes the same message,
the model may incorrectly interpret that there is a traceability link,
whereas it should actually be checking for the input message. To
enhance retrieval accuracy, future approaches can further improve
by identifying and excluding the trigger condition while retaining
only the content related to the input message.
Answering RQ4: After effectively reducing the number of require-
ments pairs being considered with our three-step filtering approach,
TVR achieves an 85.50% correctness in recovering missing links, thus
making it applicable with a manageable number of false positives.
Based on an error analysis, we conclude that additional preprocess-
ing of messages and signals may further improve accuracy.
6 Discussion
6.1 Industry Feedback on TVR
Industry feedback on TVR underscores its significant advantages.
Engineers were impressed by its high accuracy and reliability, thus
recognizing its potential for ensuring traceability consistency in
practice. TVR’s high level of automation can significantly reduce
manual effort, leading to substantial time and cost savings.
Furthermore, we demonstrated TVR’s interpretable output to
system engineers by removing the instruction “ and only respond
with either ‘Yes’ or ‘No’ ” from our prompt. In its response, TVR first
identifies the key message and signal in the stakeholder require-
ment. Then it searches the relevant message or signal in the linked
system requirement and checks whether the system requirement
addresses the message or signal in the stakeholder requirement by
taking consistent action for it. For an invalid pair, TVR provides an
example explanation as follows:

Anonymous, xxx, xxx Niu et al.
“Step 1: Identify the key message/signal in the stakeholder requirement.
The key message is MESSAGE_1. Step 2: Search for this message
in the system requirement. After carefully examining the system
requirement, I don’t see any explicit mention of MESSAGE_1 or any
checks related to it. Step 3: Determine if the message is covered. Since
there is no explicit mention or check for MESSAGE_1, we can conclude
that this specific message is not covered. Step 4: Conclusion. The
response is: No. ”
This interpretable output enables system engineers to quickly
understand and validate traceability decisions, thereby enhancing
TVR’s applicability.
Looking ahead, industry professionals identified two key points
for improvement: Further minimizing false positives to further re-
fine precision and enhancing scalability to better adapt to large-
scale industrial datasets with diverse requirements. These insights
will inform future refinements, ensuring the approach remains both
effective and practical for real-world adoption.
6.2 Practical Application of TVR
TVR is designed to automate and enhance the accuracy of traceabil-
ity link validation and recovery for DTC requirements in automo-
tive systems. To apply TVR, practitioners only need to prepare three
key components: (1) The input pairs to be verified – stakeholder
requirements and system requirements. (2) The retrieval database –
a collection of requirement pairs with labels (valid or invalid) previ-
ously verified by system engineers. (3) The prompt specifying the
task description and instructions. Once these inputs are provided,
TVR generates a prediction result with an explanation as instructed
by the prompt, improving transparency and interpretability.
TVR is particularly well-suited for iterative software develop-
ment in industrial settings. As the retrieval database expands with
more human-confirmed data, the retriever can extract more rele-
vant examples, enhancing the LLM’s ability to make more accurate
and reliable assessments over time.
When applying TVR to different datasets (beyond DTC require-
ments) or various software artifacts, the prompt instructions must
be adjusted to ensure the task description aligns closely with the
specific problem. Additionally, selecting the appropriate number
of examples and the similarity measure requires empirical experi-
mentation with the specific data at hand to determine the optimal
configuration. A priori, the basic principles of TVR can also be
adapted to other domains and types of requirements.
In summary, TVR demonstrates excellent performance on in-
dustrial data, offering high accuracy, strong robustness, ease of use,
and broad applicability.
7 Threats to Validity
In this study, we identify the following threats to validity.
External Validity. TVR is specifically designed for DTC require-
ments, which are critical in the automotive domain, using a dataset
from an industry partner. Since TVR relies on a retrieval-based
approach to obtain prompt examples, the accuracy of TVR in prac-
tical applications may be affected by the quality of the retrieval
database. However, the general principles of TVR should be widely
applicable to many types of requirements in domains where trace-
ability between high-level stakeholder requirements and system
requirements is important.Internal Validity. The dataset was annotated by co-authors under
the guidance of system engineers. However, manual annotation is
inherently prone to errors, such as misinterpretation of require-
ments or inconsistencies in labeling. These annotation errors could
affect the evaluation of TVR’s performance. Another internal threat
arises from the model’s robustness evaluation, which is affected by
the degree of variation among different categories. To mitigate this
threat, we applied the concept of cross-validation to evaluate the
model’s robustness across all four categories.
8 Related Work
Traceability, defined as “the ability to describe and follow the life of
an artifact developed during the software lifecycle in both forward
and backward directions” [ 41], plays a crucial role in requirements
engineering and software engineering in general. To date, the most
extensively studied challenges are traceability link recovery and
traceability maintenance [20]. Researchers have proposed various
approaches to support software traceability between different arti-
facts, including requirements-to-code [ 11,22,37,52], document-to-
code [3, 35, 43], and document-to-model [8, 36].
Early automated traceability techniques relied on classical NLP
methods, primarily leveraging information retrieval techniques [ 20].
These methods establish potential traceability links by computing
textual similarity between artifacts using models such as the vec-
tor space model [ 3,42], latent semantic indexing (LSI) [ 43], latent
Dirichlet allocation (LDA) [ 4], and hybrid approaches [ 17,46]. How-
ever, these traditional models struggle to capture deep semantics.
Recent approaches have addressed this limitation by incorporating
word embeddings and deep neural networks [19, 22, 62, 70]. Addi-
tionally, some studies have explored active learning [ 45] and self-
attention mechanisms [ 69] for improving traceability link recovery.
With the advent of LLMs such as GPT [ 49,50], Llama [ 27],
and Claude [ 25], new opportunities have emerged for automating
traceability while addressing the limitations of previous methods.
Hey et al. [ 22] proposed the FTLR approach, which integrates word
embedding similarity and Word Mover’s Distance [ 38] to generate
requirements-to-code trace links. Hey et al. [ 23] further enhances
accuracy by incorporating a NoRBERT classifier to filter irrelevant
requirement segments. Rodriguez et al. [ 58] investigated the use
of Claude, to directly perform traceability link recovery through
natural language prompting. Their study demonstrated that gen-
erative models can not only recover trace links but also provide
explanations for identified links. Fuchss et al. [ 10] recently intro-
duced LiSSA, a RAG-enhanced LLM approach for traceability link
recovery across requirements-to-code, documentation-to-code, and
architecture documentation-to-models. LiSSA first retrieves rele-
vant elements based on similarity values between embeddings, then
generates LLM-based prompts to assess traceability links.
In contrast to prior studies, our research focuses on validating
traceability links as well as recovering missing links, addressing
specific industry needs in the automotive sector: 1) Industry practi-
tioners require a mechanism to verify the correctness of traceability
links between stakeholder requirements and system requirements
established by system engineers. 2) Both stakeholder and system
requirements exhibit variations in the way they are expressed, with
the possibility of encountering unseen variations in the future. 3)

TVR: Automotive System Requirement T raceability V alidation and R ecovery Through Retrieval-Augmented Generation Anonymous, xxx, xxx
Both stakeholder and system requirements follow loose templates,
with small differences in the way message and signal values are
expressed, thus rendering information retrieval-based approaches
based on term frequency and word overlapping ineffective. To ad-
dress these challenges, we frame the validation task as a binary
classification problem, where LLMs determine the validity of exist-
ing traceability links. LLMs are well-suited for this task due to their
ability to understand and capture the subtle differences in messages
and signals, in the presence of values among variations in require-
ments templates while simultaneously supporting explainability
for traceability link validation.
9 Conclusion
In this paper, we propose TVR, an approach leveraging RAG based
on LLMs to verify the validity of traceability between high-level
stakeholder requirements and system requirements in automotive
systems. TVR achieves 98.87% accuracy in detecting whether a trace-
ability link is valid. Furthermore, experimental results demonstrate
the robustness of TVR in effectively handling unseen variations in
requirements templates, retaining a 97.13% accuracy. Additionally,
TVR can identify missing links between requirements with 85.50%
correctness. These findings indicate that TVR is effective not only
for traceability validation but also for recovering missing links. TVR
can thus be applied to automotive systems, helping the industry to
save both time and cost. In the future, the basic principles of TVR
can be adapted to other types of requirements and domains where
such traceability is important.
Acknowledgments
This work was supported by a research grant from Aptiv, as well
as by the Discovery Grant and Canada Research Chair programs of
the Natural Sciences and Engineering Research Council of Canada
(NSERC) and Science Foundation Ireland grant 13/RC/2094-2.
References
[1]Toufique Ahmed, Kunal Suresh Pai, Premkumar Devanbu, and Earl Barr. 2024.
Automatic semantic augmentation of language model prompts (for code summa-
rization). In Proceedings of the IEEE/ACM 46th international conference on software
engineering . 1–13.
[2]Anthropic. 2024. Claude Sonnet 3.5. https://www.anthropic.com/news/claude-
3-5-sonnet
[3]Giuliano Antoniol, Gerardo Canfora, Gerardo Casazza, Andrea De Lucia, and Et-
tore Merlo. 2002. Recovering traceability links between code and documentation.
IEEE transactions on software engineering 28, 10 (2002), 970–983.
[4]Hazeline U Asuncion, Arthur U Asuncion, and Richard N Taylor. 2010. Software
traceability with topic modeling. In Proceedings of the 32nd ACM/IEEE interna-
tional conference on Software Engineering-Volume 1 . 95–104.
[5]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al .2020. Language models are few-shot learners. Advances in neural
information processing systems 33 (2020), 1877–1901.
[6]Sofia Charalampidou, Apostolos Ampatzoglou, Evangelos Karountzos, and Paris
Avgeriou. 2021. Empirical studies on software traceability: A mapping study.
Journal of Software: Evolution and Process 33, 2 (2021), e2294.
[7]Junkai Chen, Xing Hu, Zhenhao Li, Cuiyun Gao, Xin Xia, and David Lo. 2024.
Code search is all you need? improving code suggestions with code search. In
Proceedings of the IEEE/ACM 46th International Conference on Software Engineering .
1–13.
[8]Jane Cleland-Huang, Raffaella Settimi, Chuan Duan, and Xuchang Zou. 2005.
Utilizing supporting evidence to improve dynamic requirements traceability. In
13th IEEE international conference on Requirements Engineering (RE’05) . IEEE,
135–144.
[9]Xueying Du, Mingwei Liu, Kaixin Wang, Hanlin Wang, Junwei Liu, Yixuan Chen,
Jiayi Feng, Chaofeng Sha, Xin Peng, and Yiling Lou. 2024. Evaluating largelanguage models in class-level code generation. In Proceedings of the IEEE/ACM
46th International Conference on Software Engineering . 1–13.
[10] Dominik Fuchß, Tobias Hey, Jan Keim, Haoyu Liu, Niklas Ewald, Tobias Thirolf,
and Anne Koziolek. 2025. LiSSA: Toward Generic Traceability Link Recovery
through Retrieval-Augmented Generation. In Proceedings of the IEEE/ACM 47th
International Conference on Software Engineering. ICSE , Vol. 25.
[11] Hui Gao, Hongyu Kuang, Kexin Sun, Xiaoxing Ma, Alexander Egyed, Patrick
Mäder, Guoping Rong, Dong Shao, and He Zhang. 2022. Using consensual
biterms from text structures of requirements and code to improve IR-based
traceability recovery. In Proceedings of the 37th IEEE/ACM International Conference
on Automated Software Engineering . 1–1.
[12] Mingqi Gao, Xinyu Hu, Jie Ruan, Xiao Pu, and Xiaojun Wan. 2024. Llm-based
nlg evaluation: Current status and challenges. arXiv preprint arXiv:2402.01383
(2024).
[13] Shuzheng Gao, Xin-Cheng Wen, Cuiyun Gao, Wenxuan Wang, Hongyu Zhang,
and Michael R Lyu. 2023. What makes good in-context demonstrations for code
intelligence tasks with llms?. In 2023 38th IEEE/ACM International Conference on
Automated Software Engineering (ASE) . IEEE, 761–773.
[14] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey. arXiv preprint arXiv:2312.10997 2
(2023).
[15] Gabriel Alberto García-Mireles, M Ángeles Moraga, and Félix García. 2012. Devel-
opment of maturity models: a systematic literature review. In 16th International
Conference on Evaluation & Assessment in Software Engineering (EASE 2012) . IET,
279–283.
[16] Mingyang Geng, Shangwen Wang, Dezun Dong, Haotian Wang, Ge Li, Zhi
Jin, Xiaoguang Mao, and Xiangke Liao. 2024. Large language models are few-
shot summarizers: Multi-intent comment generation via in-context learning. In
Proceedings of the 46th IEEE/ACM International Conference on Software Engineering .
1–13.
[17] Malcom Gethers, Rocco Oliveto, Denys Poshyvanyk, and Andrea De Lucia. 2011.
On integrating orthogonal information retrieval methods to improve traceability
recovery. In 2011 27th IEEE International Conference on Software Maintenance
(ICSM) . IEEE, 133–142.
[18] Orlena CZ Gotel and CW Finkelstein. 1994. An analysis of the requirements trace-
ability problem. In Proceedings of IEEE international conference on requirements
engineering . IEEE, 94–101.
[19] Jin Guo, Jinghui Cheng, and Jane Cleland-Huang. 2017. Semantically enhanced
software traceability using deep learning techniques. In 2017 IEEE/ACM 39th
International Conference on Software Engineering (ICSE) . IEEE, 3–14.
[20] Jin LC Guo, Jan-Philipp Steghöfer, Andreas Vogelsang, and Jane Cleland-Huang.
2024. Natural Language Processing for Requirements Traceability. arXiv preprint
arXiv:2405.10845 (2024).
[21] Trevor Hastie, Robert Tibshirani, Jerome H Friedman, and Jerome H Friedman.
2009. The elements of statistical learning: data mining, inference, and prediction .
Vol. 2. Springer.
[22] Tobias Hey, Fei Chen, Sebastian Weigelt, and Walter F Tichy. 2021. Improving
traceability link recovery using fine-grained requirements-to-code relations. In
2021 IEEE International Conference on Software Maintenance and Evolution (ICSME) .
IEEE, 12–22.
[23] Tobias Hey, Jan Keim, and Sophie Corallo. 2024. Requirements classification for
traceability link recovery. In 2024 IEEE 32nd International Requirements Engineer-
ing Conference (RE) . IEEE, 155–167.
[24] https://aws.amazon.com/bedrock/amazon models/titan/. 2023. Amazon Titan.
https://aws.amazon.com/bedrock/amazon-models/titan/
[25] https://claude.ai/. 2023. Claude. https://claude.ai/
[26] https://mistral.ai/. 2023. Mistral. https://mistral.ai/
[27] https://www.llama.com/. 2023. Llama. https://www.llama.com/
[28] Yucheng Hu and Yuxing Lu. 2024. Rag and rau: A survey on retrieval-augmented
language model in natural language processing. arXiv preprint arXiv:2404.19543
(2024).
[29] Ali Idri and Laila Cheikhi. 2016. A survey of secondary studies in software process
improvement. In 2016 IEEE/ACS 13th International Conference of Computer Systems
and Applications (AICCSA) . IEEE, 1–8.
[30] ISO/IEC 33001. 2015-03. Information technology—Process assessment—Concepts
and terminology.
[31] ISO/IEC 33002. 2015-03. Information technology—Process assessment—Concepts
and terminology.
[32] ISO/IEC 33003. 2015-03. Information technology—Process assessment—Concepts
and terminology.
[33] ISO/IEC 33004. 2015-03. Information technology—Process assessment—Concepts
and terminology.
[34] Nikitas Karanikolas, Eirini Manga, Nikoletta Samaridi, Eleni Tousidou, and
Michael Vassilakopoulos. 2023. Large language models versus natural language
understanding and generation. In Proceedings of the 27th Pan-Hellenic Conference
on Progress in Computing and Informatics . 278–290.

Anonymous, xxx, xxx Niu et al.
[35] Jan Keim, Sophie Corallo, Dominik Fuchß, Tobias Hey, Tobias Telge, and Anne
Koziolek. 2024. Recovering Trace Links Between Software Documentation And
Code. In Proceedings of the IEEE/ACM 46th International Conference on Software
Engineering . 1–13.
[36] Jan Keim, Sophie Corallo, Dominik Fuchß, and Anne Koziolek. 2023. Detecting
inconsistencies in software architecture documentation using traceability link
recovery. In 2023 IEEE 20th International Conference on Software Architecture
(ICSA) . IEEE, 141–152.
[37] Hongyu Kuang, Patrick Mäder, Hao Hu, Achraf Ghabi, LiGuo Huang, Jian Lü, and
Alexander Egyed. 2015. Can method data dependencies support the assessment
of traceability between requirements and source code? Journal of Software:
Evolution and Process 27, 11 (2015), 838–866.
[38] Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. 2015. From word
embeddings to document distances. In International conference on machine learn-
ing. PMLR, 957–966.
[39] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems 33 (2020), 9459–9474.
[40] Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu, Yuan Ni, Guotong Xie,
Xiaoling Wang, and Xipeng Qiu. 2023. Unified Demonstration Retriever for
In-Context Learning. In Proceedings of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada,
July 9-14, 2023 , Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (Eds.).
Association for Computational Linguistics, 4644–4668. https://doi.org/10.18653/
V1/2023.ACL-LONG.256
[41] Andrea De Lucia, Fausto Fasano, Rocco Oliveto, and Genoveffa Tortora. 2007.
Recovering traceability links in software artifact management systems using
information retrieval methods. ACM Transactions on Software Engineering and
Methodology (TOSEM) 16, 4 (2007), 13–es.
[42] Anas Mahmoud. 2015. An information theoretic approach for extracting and
tracing non-functional requirements. In 2015 IEEE 23rd International Requirements
Engineering Conference (RE) . IEEE, 36–45.
[43] Andrian Marcus and Jonathan I Maletic. 2003. Recovering documentation-to-
source-code traceability links using latent semantic indexing. In 25th International
Conference on Software Engineering, 2003. Proceedings. IEEE, 125–135.
[44] Christoph Marscholik and Peter Subke. 2009. Road Vehicles: Diagnostic Commu-
nication: Technology and Applications . Laxmi Publications, Ltd.
[45] Chris Mills, Javier Escobar-Avila, Aditya Bhattacharya, Grigoriy Kondyukov,
Shayok Chakraborty, and Sonia Haiduc. 2019. Tracing with less data: active
learning for classification-based traceability link recovery. In 2019 IEEE Interna-
tional Conference on Software Maintenance and Evolution (ICSME) . IEEE, 103–113.
[46] Kevin Moran, David N Palacio, Carlos Bernal-Cárdenas, Daniel McCrystal, Denys
Poshyvanyk, Chris Shenefiel, and Jeff Johnson. 2020. Improving the effectiveness
of traceability link recovery using hierarchical bayesian networks. In Proceedings
of the ACM/IEEE 42nd International Conference on Software Engineering . 873–885.
[47] Fangwen Mu, Lin Shi, Song Wang, Zhuohao Yu, Binquan Zhang, ChenXue Wang,
Shichao Liu, and Qing Wang. 2024. Clarifygpt: A framework for enhancing
llm-based code generation via requirements clarification. Proceedings of the ACM
on Software Engineering 1, FSE (2024), 2332–2354.
[48] Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, and Brad
Myers. 2024. Using an llm to help with code understanding. In Proceedings of the
IEEE/ACM 46th International Conference on Software Engineering . 1–13.
[49] OpenAI. 2023. ChatGPT. https://openai.com/chatgpt
[50] OpenAI. 2023. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL] https://arxiv.
org/abs/2303.08774
[51] Dibyendu Palai. 2013. Vehicle level approach for optimization of on-board diag-
nostic strategies for fault management. (2013).
[52] Annibale Panichella, Collin McMillan, Evan Moritz, Davide Palmieri, Rocco
Oliveto, Denys Poshyvanyk, and Andrea De Lucia. 2013. When and how using
structural information to improve ir-based traceability recovery. In 2013 17th
European Conference on Software Maintenance and Reengineering . IEEE, 199–208.
[53] Shravan Pargaonkar. 2023. Synergizing requirements engineering and qual-
ity assurance: A comprehensive exploration in software quality engineering.
International Journal of Science and Research (IJSR) 12, 8 (2023), 2003–2007.
[54] Parivash Pirasteh, Slawomir Nowaczyk, Sepideh Pashami, Magnus Löwenadler,
Klas Thunberg, Henrik Ydreskog, and Peter Berck. 2019. Interactive feature
extraction for diagnostic trouble codes in predictive maintenance: A case study
from automotive domain. In Proceedings of the Workshop on Interactive Data
Mining . 1–10.
[55] Abdallah Qusef, Gabriele Bavota, Rocco Oliveto, Andrea De Lucia, and David
Binkley. 2011. Scotch: Slicing and coupling based test to code trace hunter. In
2011 18th Working Conference on Reverse Engineering . IEEE, 443–444.
[56] Mona Rahimi and Jane Cleland-Huang. 2018. Evolving software trace links
between requirements and source code. Empirical Software Engineering 23 (2018),
2198–2231.
[57] S Riedel, D Kiela, P Lewis, and A Piktus. 2020. Retrieval Augmented Generation:
Streamlining the creation of intelligent natural language processing models. MLApplications, Open Source (2020).
[58] Alberto D Rodriguez, Katherine R Dearstyne, and Jane Cleland-Huang. 2023.
Prompts matter: Insights and strategies for prompt engineering in automated
software traceability. In 2023 IEEE 31st International Requirements Engineering
Conference Workshops (REW) . IEEE, 455–464.
[59] Ze Tang, Jidong Ge, Shangqing Liu, Tingwei Zhu, Tongtong Xu, Liguo Huang,
and Bin Luo. 2023. Domain adaptive code completion via language models and
decoupled domain databases. In 2023 38th IEEE/ACM International Conference on
Automated Software Engineering (ASE) . IEEE, , 421–433.
[60] Andreas Theissler. 2017. Multi-class novelty detection in diagnostic trouble
codes from repair shops. In 2017 IEEE 15th International Conference on Industrial
Informatics (INDIN) . IEEE, 1043–1049.
[61] Hanny Tufail, Muhammad Faisal Masood, Babar Zeb, Farooque Azam, and
Muhammad Waseem Anwar. 2017. A systematic review of requirement traceabil-
ity techniques and tools. In 2017 2nd international conference on system reliability
and safety (ICSRS) . IEEE, 450–454.
[62] Wentao Wang, Nan Niu, Hui Liu, and Zhendong Niu. 2018. Enhancing automated
requirements traceability by resolving polysemy. In 2018 IEEE 26th International
Requirements Engineering Conference (RE) . IEEE, 40–51.
[63] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain
of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022).
[64] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning
in large language models. Advances in neural information processing systems 35
(2022), 24824–24837.
[65] Karl E Wiegers and Joy Beatty. 2013. Software requirements . Pearson Education.
[66] What Makes In-Context Learning Work. [n. d.]. Rethinking the Role of Demon-
strations: What Makes In-Context Learning Work? ([n. d.]).
[67] Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming
Huang, Xue Liu, Tei-Wei Kuo, Nan Guan, et al .2024. Retrieval-augmented gener-
ation for natural language processing: A survey. arXiv preprint arXiv:2407.13193
(2024).
[68] Huan Zhang, Wei Cheng, Yuhan Wu, and Wei Hu. 2024. A Pair Programming
Framework for Code Generation via Multi-Plan Exploration and Feedback-Driven
Refinement. In Proceedings of the 39th IEEE/ACM International Conference on
Automated Software Engineering . 1319–1331.
[69] Meng Zhang, Chuanqi Tao, Hongjing Guo, and Zhiqiu Huang. 2021. Recover-
ing semantic traceability between requirements and source code using feature
representation techniques. In 2021 IEEE 21st International Conference on Software
Quality, Reliability and Security (QRS) . IEEE, 873–882.
[70] Teng Zhao, Qinghua Cao, and Qing Sun. 2017. An improved approach to trace-
ability recovery based on word embeddings. In 2017 24th Asia-Pacific Software
Engineering Conference (APSEC) . IEEE, 81–89.