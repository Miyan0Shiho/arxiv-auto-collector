# LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters

**Authors**: Virgill van der Meer, Julien Rossi

**Published**: 2026-05-12 12:01:29

**PDF URL**: [https://arxiv.org/pdf/2605.12012v1](https://arxiv.org/pdf/2605.12012v1)

## Abstract
Public-sector legal departments in the Netherlands face acute staff shortages, increased case volumes, and increased pressure to meet regulatory compliance. This paper presents LegalCheck, a novel system that addresses these challenges by automating the drafting of objection response letters through a combination of Retrieval-Augmented Generation (RAG) and Context-Augmented Generation (CAG). Using a large language model (LLM) alongside curated legal knowledge bases, LegalCheck performs retrieval of relevant laws and precedents, and uses controlled prompting to incorporate both external knowledge and case-specific details into a coherent draft. An expert-in-the-loop review ensures that each generated letter is legally sound and contextually appropriate. In a real-world deployment within the Municipality of Amsterdam, LegalCheck produced near-final advice letters in minutes rather than hours, while maintaining high legal consistency and factual accuracy. The output is based on actual regulations and prior cases, providing explainable outputs that captured the vast majority of required legal reasoning (often 80\% to 100\% of essential content). Legal professionals found that the system reduced their workload and ensured a consistent application of legal standards, without replacing human judgment. These results demonstrate substantial efficiency gains, improved legal consistency, and positive user acceptance. More broadly, this work illustrates how responsible AI can be deployed in the legal domain by augmenting LLMs with domain knowledge and governance mechanisms.

## Full Text


<!-- PDF content starts -->

LegalCheck: Retrieval- and Context-Augmented Generation for
Drafting Municipal Legal Advice Letters
Virgill van der Meer
Municipality of Amsterdam
Amsterdam, Netherlands
v.van.der.meer@amsterdam.nlJulien Rossi
University of Amsterdam
Amsterdam, Netherlands
j.rossi@uva.nl
Abstract
Public-sector legal departments in the Netherlands face acute staff
shortages, increased case volumes, and increased pressure to meet
regulatory compliance. This paper presents LegalCheck, a novel
system that addresses these challenges by automating the drafting
of objection response letters through a combination of Retrieval-
Augmented Generation (RAG) and Context-Augmented Generation
(CAG). Using a large language model (LLM) alongside curated le-
gal knowledge bases, LegalCheck performs retrieval of relevant
laws and precedents, and uses controlled prompting to incorporate
both external knowledge and case-specific details into a coherent
draft. An expert-in-the-loop review ensures that each generated
letter is legally sound and contextually appropriate. In a real-world
deployment within the Municipality of Amsterdam, LegalCheck
produced near-final advice letters in minutes rather than hours,
while maintaining high legal consistency and factual accuracy. The
output is based on actual regulations and prior cases, providing
explainable outputs that captured the vast majority of required legal
reasoning (often 80% to 100% of essential content). Legal profession-
als found that the system reduced their workload and ensured a
consistent application of legal standards, without replacing human
judgment. These results demonstrate substantial efficiency gains,
improved legal consistency, and positive user acceptance. More
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
ICAIL 2026, Singapore
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN XXX-X-XXXX-XXXX-X/XXXX/XX
https://doi.org/XXXXXXX.XXXXXXXbroadly, this work illustrates how responsible AI can be deployed
in the legal domain by augmenting LLMs with domain knowledge
and governance mechanisms.
CCS Concepts
•Applied computing →Law;•Information systems →Lan-
guage models;•Computing methodologies →Information
extraction;Natural language generation.
Keywords
Legal Operations, Generative AI, Case Law Retrieval
ACM Reference Format:
Virgill van der Meer and Julien Rossi. 2026. LegalCheck: Retrieval- and
Context-Augmented Generation for Drafting Municipal Legal Advice Let-
ters. InProceedings of 21stInternational Conference on Artificial Intelli-
gence and Law (ICAIL 2026).ACM, New York, NY, USA, 11 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
Municipal legal departments face increasing caseloads and persis-
tent staff shortages, creating pressure to deliver more with fewer
resources [ 34] [24]. In the City of Amsterdam’s Legal Department,
the workload associated with handling formal citizen objections1
in administrative and environmental law has increased by 35% over
the last five years. This environment has driven a transformation to
AI-based and data-driven workflows [ 12]. A centralized digital case
library (Zaakbibliotheek) was established, consolidating 55,000 past
1In Dutch administrative law, the objection procedure (bezwaar) is a statutory internal
review mechanism. Before going to court, citizens can ask the administrative body to
reconsider its decision, with a right to be heard at low threshold. It plays a key role in
legal protection, functioning both as a learning mechanism for the administration and
as a filter that prevents unnecessary court cases.arXiv:2605.12012v1  [cs.AI]  12 May 2026

ICAIL 2026, June 08–12, 2026, Singapore van der Meer and Rossi
cases (objections and decisions) into an indexed knowledge base.
Building on this data foundation, the department began piloting AI
tools to alleviate routine tasks and improve efficiency. In particular,
several supportive applications were developed:Kern van de Zaak,
an AI-based summarization tool that extracts the core issues from
legal documents to enable faster research;Juris-Match, a seman-
tic case search engine for finding conceptually similar precedents;
Spraakzaak, a speech-to-text transcription and summarization sys-
tem for hearings; and theKOG-i-fier, a classifier that triages in-
coming objections by labeling straightforward (clearly unfounded)
cases using generative AI. Each tool addresses a specific pain point
– from legal research and case intake to transcript handling – illus-
trating how AI can unlock the value of legal data and automate the
clerical workload in the public sector.
The flagship of this AI-driven transformation isLegalCheck, a
system that drafts preliminary legal advice letters for citizen ob-
jection cases. Legal advice letters are complex, high-stakes doc-
uments: they must accurately cite laws and prior rulings, align
with case facts, and maintain a professional yet citizen-friendly
tone. Manual drafting of these letters is both time-intensive and
cognitively demanding, as it requires meticulous legal reasoning,
contributing to backlogs under heavy caseloads. The core idea of
LegalCheck follows a human-in-the-loop paradigm: the model pro-
duces a grounded draft, while a legal expert remains responsible for
reviewing, correcting, and approving the final letter. This approach
acknowledges government guidelines that speed gains from AI
must not come at the expense of due diligence and human judg-
ment [ 10] [13]. Developing “augmentation rather than automation,”
LegalCheck is positioned as an assistant to legal staff, with the aim
of improving productivity, substantive quality, and consistency in
decision-writing while maintaining accountability and trust [ 9] [7].
This research contributes to the growing literature on the topic
of RAG and expands it by presenting the first applied study (to
our knowledge) of using RAG for legal document drafting in a
real public-sector setting. This research bridges recent advances
in AI language models with the practical needs and constraints of
government legal practice. Unlike previous legal NLP research that
mostly addressed classification or outcome prediction [ 17], or recent
studies of AI assistance in law [ 8] [39], LegalCheck demonstrates
in vivohow generative AI can produce high-quality legal reasoning
[26] on scale when coupled with a domain-specific knowledge base
and human oversight [2] [32].
The system introduces a novel Context-Augmented Genera-
tion (CAG) pipeline layered on top of standard RAG. In addition
to retrieving relevant documents to ground the model’s output,
LegalCheck integrates case-specific input beyond the existing knowl-
edge base. Specifically, the jurist can upload key documents related
to the case – such as the citizen’s objection letter and the official
report of the enforcement officer –, specify the intended dictum
(uphold or reject), and optionally provide free-text “steering ad-
vice” that reflects expert knowledge or instructions not captured
elsewhere. After LegalCheck produces an initial draft, the human
reviewer can provide contextual feedback by highlighting omis-
sions or suggesting revisions. The AI then performs a refinement
pass, regenerating the letter to incorporate the expert’s input.
This multi-stage CAG approach uses the substantive content of
the new case, expert guidance and refinement, and the structure ofprevious similar cases to improve the precision and completeness of
the draft. We implement CAG through multiple LLM prompt stages.
In the initial prompt stage, the model is provided with the uploaded
case documentation along with any jurist annotations and generates
a draft that integrates all relevant information. In the refinement
stage, the model is prompted with additional annotations on this
generated draft and returns a revised version that incorporates the
requested corrections.
Our contributions are twofold: (1) an integrated RAG+CAG ar-
chitecture that produces quasi-final legal advice letters with proper
citations and a formal, accessible tone suitable for communication
with citizens and (2) an empirical evaluation of the impact of this
system on the efficiency of the drafting, output quality, and profes-
sional workflows in a municipal legal department.
We position this work in the growing academic discourse on “AI-
powered lawyering,” providing empirical evidence that retrieval-
based text generation can significantly improve the quality and
speed of legal work [ 26]. In addition, we address the organizational
and ethical context: the project is deployed in a government setting
subject to strict privacy, security, and transparency requirements.
Thus, the research contributes case-based insights on responsible
AI adoption in the legal domain, aligning with calls for trustworthy
AI in public services, including how to maintain legal precision, ex-
plainability, and human control when using large language models
(LLMs) for official decision documents [10].
The remainder of this paper is structured as follows. Section 2
reviews related work on legal NLP, retrieval-augmented generation,
and human–AI workflows in high-stakes legal settings. Section 3
describes the LegalCheck system architecture, including the RAG
pipeline and the multi-stage CAG pipeline that integrates case-
specific context before and after initial generation, and situates
it relative to existing AI techniques in law. Section 4 details our
methodology, including the datasets of waste fine objections and
the removal and towing of bicycles, motor vehicles, and boats the
semantic indexing and retrieval process, the prompt design, and our
evaluation setup (metrics and expert-in-the-loop validation). Sec-
tion 5 reports the results of both quantitative evaluations (output
quality, speed gains) and qualitative feedback from legal profession-
als. We also discuss the legal reasoning of the system, the role of
the human expert, and trust in the AI-assisted process. Section 6
concludes with implications for legal practice and future research
directions, including the reproducibility and generalizability of our
approach.
2 Literature Review
2.1 Legal NLP and Early AI in Law
Research on AI in the legal domain has traditionally focused on
analysis and prediction tasks rather than text generation [ 30]. Many
foundational legal NLP works applied machine learning to classify
documents or predict case outcomes [6].
For example, Aletras et al. (2016) used textual features to predict
decisions of the European Court of Human Rights [ 2], and Katz
et al. (2017) built a model to forecast U.S. Supreme Court verdicts
[17]. These approaches demonstrated that NLP can derive patterns
from legal text for tasks such as prediction of judgment, aligning
with broader efforts in legal analytics [ 6]. However, they did not

LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters ICAIL 2026, June 08–12, 2026, Singapore
attempt to generate new legal documents. Only recently, with the
advent of generative models, have researchers begun to explore AI
for the drafting of legal text [ 28]. Early explorations have either
fine-tuned LLMs on relatively small legal drafting corpora to gen-
erate prototype documents, or used retrieval-augmented models
for narrow legal question answering on case fragments rather than
for end-to-end drafting of official legal letters [20, 38].
In particular, Lin & Cheng (2024) showed that a fine-tuned
transformer could draft legal documents in a lab setting [ 20], and
Wiratunga et al. (2024) introduced a case-based retrieval approach
to improve the answer of the LLM to legal questions [ 38]. These
studies hint at the potential of generative AI in law but remain
confined to experimental or narrowly-scoped scenarios.
2.2 Retrieval-Augmentation and Human–AI
Workflow
A key challenge in legal text generation is ensuring accuracy and
trustworthiness. Therefore, recent techniques combine LLMs with
external knowledge. Retrieval-Augmented Generation (RAG) [ 19]
grounds an LLM’s output in relevant documents and has been ap-
plied to knowledge-intensive tasks to reduce errors. In the legal
context, grounding is crucial to prevent hallucinations (fabrica-
tions of laws or facts) by the model. By retrieving actual statutes or
precedents and feeding them into the prompt, AI’s suggestions can
be verified and legally sound [ 19] [21]. Our approach adopts this
strategy, drawing on a curated case library so that every generated
argument can be traced to real source material. Equally important
is the role of human expertise. Experts stress that in high-stakes do-
mains such as law, AI systems should augment rather than replace
human judgment [ 4] [10]. Following this principle, we implement a
human-in-the-loop review: the AI system generates a draft advice
letter, and a legal professional rigorously checks and edits it. This
design aligns with previous findings that combining human exper-
tise with AI can yield better results than either alone, provided that
the human supervises and corrects the AI output [ 4] [16], and with
recent work that conceptualizes meaningful human oversight as a
layered allocation of agency between AI “proposers” and human
“evaluators” [ 40]. Recent experimental studies underscore the need
for such oversight: GPT-3.5 could “make a case” in Dutch law com-
parable to a junior lawyer, but human review was still necessary
for nuanced correctness [ 39], legal aid attorneys benefited from
GPT-generated drafts for client letters, but only after verifying and
refining the content [8].
2.3 Novelty of Our Approach
In light of the literature above, LegalCheck is, to our knowledge, the
first system to deploy an LLM-based drafting assistant for official
municipal legal letters in a real public-sector workflow. Previous
works on AI-generated legal writing were proof-of-concept studies
(e.g. a fine-tuned model writing a generic legal document [ 20])
or simulations (comparing AI and human writing on a sample
task [ 39]), without integrating into everyday legal operations. By
contrast, our work integrates a GPT-4o-based pipeline with the
actual processes of a municipal legal department, handling live
citizen objection cases. This use case – drafting municipal objection
response letters – has not been addressed in the literature so far.We fill this gap by demonstrating a complete hybrid human–AI
workflow for decision drafting.
3 System Architecture and Technical Approach
3.1 High-Level
LegalCheck’s architecture follows a retrieval-augmented generation
paradigm (RAG), extended with a multi-stage context-Augmented
generation pipeline (CAG). Figure 1 illustrates the workflow. When
a new objection case is submitted, the system ingests the substantive
case documentation, including, at minimum, the citizen’s objection
letter and the enforcement officer’s report of the violation. In addi-
tion, the jurist is required to specify the intended dictum, that is,
whether the objection should be upheld or rejected. Optionally, the
jurist can add free-text steering advice that encodes case-specific
priorities or nuances that are not captured elsewhere. These inputs
form the core context for generation and also drive the retrieval
query.
From this combined representation, the system constructs a
textual query that summarizes the key facts and legal issues of
the case. The query is encoded using a pre-trained embedding
model (OpenAI’s text-embedding-ada-002, 1536-dimensional) to
enable semantic search. The encoded query is matched against
an internal knowledge base of previous cases and legal texts. In
our implementation, this knowledge base comprises curated sets of
historical legal advice letters from the domains included in the pilot,
namely waste-fine objections and towing of vehicles, bicycles, and
boats, together with the statutes and regulations most frequently
cited in those letters.
Each document in the knowledge base is pre-processed into
semantically meaningful chunks (typically by section) and stored
with its embedding. We perform this semantic indexing offline;
given the moderate size of the domain-specific datasets (ranging
from approximately 150 to 14,000 cases per domain), vectors can
be retrieved in-memory without a specialized vector database. The
retrieval step returns the "Top-K" most relevant text chunks from
previous cases, with "K" [50 - 200]. These sections typically include
previous “Explanation” sections from similar objection letters, as
well as snippets of legal provisions, ensuring that similar content
of the new case is available to ground the AI’s drafting.
In summary, LegalCheck’s architecture consists of a robust knowl-
edge retrieval layer feeding a GenAI, combined with a multi-stage
CAG pipeline and explicit expert-in-the-loop control. The system
combines the content of the new case, the collective memory of the
department (past cases), and real-time expert judgment. In doing so,
it produces advice letters that match or even exceed human-written
quality, while substantially reducing the drafting effort. Section
4 describes how we evaluated these claims through a pilot study
using both quantitative and qualitative metrics.
3.2 Initial RAG+CAG Generation
In the first generation stage, an LLM composes a draft advice letter
using both the retrieved texts and the new case context. We use
GPT-4o, deployed via the Azure OpenAI Service, as the primary
generative engine [ 23]. GPT-4o was chosen for its advanced capa-
bilities in contextual understanding, fluent formal writing, and an
extended context window (up to 128k tokens), which is beneficial

ICAIL 2026, June 08–12, 2026, Singapore van der Meer and Rossi
First Draft (Steps 1–3)Interactive Improvement (Steps 4–7)
1) Jurist input
– Objection letter
– Enforcement report
– Dictum:uphold or reject
– Optional:steering advice
Optional) De-identification
Remove personal identifiers
(names, addresses, IDs, license plate num-
bers, telephone numbers)
Knowledge Base
Prior cases
Legislation / policy
Guidelines
2) RAG: Retrieval
Semantic search + ranking
Top-𝑘relevant items
3) LLM draft generation (GPT-4o)
Prompt = jurist input + retrieved sources
Output: draft advice letter (v1)
Draft Advice Letter
(v1)4) Jurist review
Checks factual and legal correctness,
tone and structure
(expert-in-the-loop)
5) CAG: Feedback and refinement
Targeted feedback / annotations
in the interface
Refinement prompt builder
Draft v1 + jurist feedback
+ (optional) refreshed retrieval
6) LLM revised generation
Generates improved draft
(v2)
Revised Advice Letter
(v2)
7) Jurist finalization
Minor edits + approval
Optional) Re-identification
Re-insert personal identifiers
into the approved letterFinal Advice Letter
(issued)Optional: iterationsOptional:
retrieve
again
Figure 1: Overview of the LegalCheck pipeline, combining RAG with multi-stage CAG.
when many retrieved documents are included. The system con-
structs a prompt that includes (a) the officer’s observation report,
(b) the citizen’s objection letter, (c) any optional steering advice
from the jurist, (d) a list of retrieved reference chunks, and (e) a
predefined instruction template that guides the format and style of
the output. The prompt template explicitly instructs the model to
produce a structured legal advice letter that follows the municipal-
ity’s official style and uses the reference chunks provided to base its
reasoning. For example, it is instructed to write the section “Expla-
nation” in a formal, explanatory and empathetic tone that mirrors
how exemplary human advisors write, to incorporate relevant laws
and previous case details from the retrieved data, and to maintain
phrasing conventions (such as how dates, case numbers, and fine
amounts are referenced). By constraining the model in this way,
we seek to ensure that the generated letter is not a free-form cre-
ation but a grounded, patterned output that adheres to customized
legal writing norms. The model is run at low temperature (0.1) toreduce randomness, prioritizing factual consistency and stylistic
uniformity over creative variance.
The outcome of this stage is a draft legal advice letter that typ-
ically includes an introduction, a section detailing the objection,
a substantive “Explanation” section containing the legal assess-
ment (this is the core where the model’s use of retrieved precedents
comes in), and a conclusion stating the decision (uphold or reject
the objection). Because the model is conditioned on actual passages
from prior letters and legislation, it frequently produces explicit
references or citations (e.g., to a specific article of a regulation or to
the reasoning in similar earlier cases). This retrieval grounding sub-
stantially reduces hallucination and increases verifiability: AI key
assertions can be traced back to real source documents provided in
the prompt, rather than invented by the model. This design aligns
with previous work on RAG [ 19] [25] [22] and extends it to the
domain of legal decision documents.

LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters ICAIL 2026, June 08–12, 2026, Singapore
3.3 Multi-Stage CAG Refinement
The draft produced in the first generation stage is refined in an inter-
active CAG refinement stage. Through the interactive user interface,
the jurist reads the AI-generated letter and can provide additional
annotations, such as corrections, clarifications, or requests to add
or emphasize certain arguments. When the jurist triggers the re-
finement function, the system formulates a new prompt to the LLM
that includes the original AI draft, the underlying case documen-
tation, the jurist’s annotations, and retrieved similar past cases to
guide the refinement. The refine prompt instructs the model to re-
vise the draft accordingly – e.g., insert the missing legal argument,
adjust emphasis or tone, correct factual errors – while preserving
the correct parts of the draft. We use the same GPT model for this
refinement, with a custom prompt. We reuse the same model to
ensure consistency in legal reasoning and writing style between v1
and v2, and because the refinement step is essentially a constrained
rewrite of the same drafting task. In practice, a zero or one refine-
ment pass is usually sufficient; multiple iterations are possible but
rarely necessary.
This multi-stage CAG loop can be thought of as a form ofitera-
tive prompting, where the model’s own output plus a new context
(human feedback) serves as input for a second-generation pass. The
process is analogous to a lawyer writing a memo and another lawyer
marking it up with revision notes: LegalCheck then takes these
notes and produces a revised draft. By integrating CAG, LegalCheck
ensures that any case-specific nuances or expert intuitions that were
not captured via retrieval alone can be injected into the final text.
This proved valuable because even with high-quality retrieval, an
AI draft might miss subtle contextual elements that a human deems
important. The refinement step consistently produced improved
drafts that satisfied the experts’ requirements. LegalCheck always
keeps a human in the loop and never sends a letter directly to
citizens or decision-makers. The jurist remains the final author,
reviewing and approving the AI-assisted draft before it is issued.
During early deployment, a second reviewer also double-checked
AI-assisted letters as an additional safeguard.
3.4 System Implementation
The LegalCheck tool is implemented as a web application to inte-
grate into the department’s workflow. A Flask-based Python web
interface allows users to select a type of case and input the necessary
case details: the objection text, the officer’s report, any additional
documents (e.g., evidence attachments) and optional steering ad-
vice. Upon submission, the back-end loads the appropriate domain
configuration. The system is modular, with separate pipelines for
different case types. Each pipeline uses its own knowledge base
and its own set of domain-specific contextual prompts. For each
domain, the system configuration specifies which embedding index
to use, which Azure OpenAI deployments to call for generation,
refinement and embedding, and which document template to apply
when exporting the final letter. Extending LegalCheck to a new
legal domain (for example, building permit objections) therefore
requires preparing a dataset of past cases, embedding them, and
adding a corresponding configuration entry; the core logic remains
unchanged. The retrieval is performed by computing cosine similar-
ity between embeddings, and given the moderate dataset size, wedid not require a dedicated vector search library because a custom
Python implementation achieved millisecond-level latency even at
the full scale of 14k cases. We found this approach scalable, as no
significant slow-down occurred when moving from a few hundred
cases in prototyping to the full datasets. LLM calls (both initial draft
and refine) are made via the Azure AI Service, which allows the
deployment of GPT-4o in a secure cloud environment in compliance
with the city’s IT policies.
4 Methodology
4.1 Data Source and Knowledge Base
Our data originate from historical legal objection cases handled
by the municipality in the enforcement domains introduced in
Section 1. From the central case library, we extracted cases from
2022 to 2025 that resulted in a formal written advice letter from the
legal department. A typical case file contains the citizen’s objection
letter, the official report by an enforcement officer (describing the
alleged violation, for example dumping garbage at a specific time
and location) and the written advice of the legal department to
uphold or revoke the fine decision.
For retrieval, the advice letters served as the core corpus, since
they contain the legal reasoning patterns and references that LegalCheck
is designed to emulate. Across domains, we curated approximately
23,500 advice letters, with per-domain dataset sizes ranging from
about 150 to 14,000 cases. Each letter was segmented into logical
sections (e.g.,Case Details,Objection,Hearing,Explanation, andCon-
clusion). We applied section-based chunking so that the retrieval
component can, for example, return only the “Explanation” seg-
ment of a previous letter, which is typically the most informative
part to draft a new explanation. After chunking, we computed
embeddings for each segment using text-embedding-ada-002. The
resulting knowledge base is therefore a set of vector representa-
tions with pointers to their source text and metadata (such as case
identifier and section type).
To comply with privacy and data-protection requirements, all
previous case documents were anonymized or generalized before
being used in prompts. Personal identifiers such as citizen names
and addresses are replaced with placeholders, and any residual
sensitive details are carefully filtered out in accordance with GDPR-
compliant guidelines. All data used in the prompts are handled
carefully to minimize the inclusion of personal identifiers and other
sensitive information.
4.2 Retrieval and Generation Process
To generate a draft letter for a new case, the system first forms the
query embedding from the concatenated objection and the officer’s
report (and optionally a summary of any hearing transcript). Early
experiments showed that summarizing the objection before embed-
ding improved retrieval of similar cases, as it extracts the core issue.
We adopted this as a preprocessing step: a separate LLM call pro-
duces a brief summary of the objection, which we then embed (this
helps to abstract away verbose narratives while keeping key legal
points). We first retrieve 50–200 similar objection sections using
cosine similarity and then retrieve the corresponding explanation
sections from the same cases, capturing how the jurists responded
to those objections. This strategy directly yields material that can

ICAIL 2026, June 08–12, 2026, Singapore van der Meer and Rossi
inform the new response. These retrieved chunks are concatenated
into a structured prompt, after which the LLM, which has strong
multilingual capabilities and handles Dutch legal text generation
well, is then invoked to generate a first draft of the advice letter.
The prompt provides the case facts and the retrieved examples
and instructs the model to write the “Explanation” section as a
legally grounded analysis that mirrors the style and structure of the
examples, addresses each argument and cites relevant regulations or
past decisions. The prompt template and a concise system message
that frames the model as a legal expert were iteratively refined to
align the output with the department’s writing conventions and
to steer it toward drafts comparable in quality, tone, and structure
that are citizen-friendly to those of experienced jurists, with the
final responsibility left to the human reviewer.
4.3 Expert-in-the-Loop and CAG Refinement
After the AI generates the draft letter (typically within 20–25 sec-
onds, depending on the length of the context), the result is presented
to the legal advisor via the interface. At this point, the advisor
performs a qualitative review, verifying factual alignment, legal
accuracy, and the clarity, tone, structure, and overall organization
of the letter. In our evaluation phase, we log the time and nature of
edits the jurists made. We also built the refinement (CAG) feature
to help with this editing: the interface provides a text box for the
jurist to input any instruction or correction and then press “Refine
Draft”.
Figure 2: Screenshot of the LegalCheck user interface during
drafting.
The system takes the original AI draft, the jurist’s remarks, and
feeds them to GPT-4o in a special refine prompt (along with re-
injection of the top reference chunks if needed for context). We
found that this two-step generation significantly improved effi-
ciency: jurists could simply highlight issues and let the AI fix them,
rather than manually rewriting paragraphs. To evaluate CAG’s
effectiveness, we tracked when jurists used the refinement and
later asked their impressions. In all cases where it was used, the
refinement feature produced a satisfactory revised draft, which
the jurist then only needed to lightly proofread. It effectively han-
dled additions of missing arguments or rephrasing according to
the jurist’s guidance. The final reviewed (and refined, if applicable)
letter is then approved by the jurist. A DOCX or PDF version of
the letter can be downloaded, and a copy is logged together withmetadata of which sources were used and what changes were made.
This logging is critical for traceability and quality control, as every
AI-generated document is stored with references to the informa-
tion it relied on and how it was post-edited. This enables audits
or explanations of AI suggestions if needed, providing a form of
explainability through transparency: although the model’s internal
reasoning is opaque, its outputs are traceable to their sources [5].
4.4 Evaluation Design
We designed a mixed-method evaluation to assess (a) the quality
and precision of LegalCheck’s output, (b) the efficiency gains (time
saved), and (c) the user acceptance and trust in the system. The
evaluation was conducted over a six-month period during the im-
plementation of LegalCheck in the legal department’s workflow in
the pilot domains.
Quantitative Metrics:We used several quantitative measures to
compare AI-assisted drafting with traditional drafting:
•Similarity to Human-Written Letters:For a sample of his-
torical cases, we had LegalCheck generate a draft and compare
them to the original human-written letters for the same cases.
We measured an overlap ratio, defined as the percentage of
AI-generated sentences or tokens that were retained in the final
version after editing by jurists. On average, 81% of AI-generated
content was kept verbatim in the final letter, while jurists added
only about 8% new content. This suggests that the AI drafts
were largely near-final and required only limited editing. To
further quantify editing effort, we tracked the number of correc-
tions made and found that, in many cases, jurists mainly fixed
small errors (e.g., a reference detail) or added a few clarifying
sentences.
•Time Savings:We asked approximately 20 participating ju-
rists to log the time spent on each letter, broken down into
(1) time to gather inputs and invoke AI, (2) time for AI gener-
ation, and (3) time to review and edit the AI-generated draft.
We compared this with historical estimates of the time spent
conducting legal research and drafting the letters from scratch,
which averaged approximately 3 (waste), 6 (bicycles and ve-
hicles) or 16 (boats) hours per case depending on the domain.
The results showed an overall reduction of roughly 50–70% in
total time for straightforward cases. In absolute terms, work
that previously required 6 hours could be completed in 2–3
hours with the help of LegalCheck. This aligns with anecdotal
reports from jurists that having a solid draft “instantly” enabled
them to complete letters much more quickly, particularly for
recurring objection patterns. We further extrapolated the poten-
tial impact across the pilot domains: given 5,800 objections per
year, annual time savings are around 12k–17k hours of jurist
work, corresponding to roughly 10–14 full-time legal staff in
workload. At an average loaded cost of€100 per hour for a legal
officer, this equates to an estimated€1.2–1.7 million in notional
annual cost savings for the department. Although these figures
assume full deployment and that all saved time is effectively
realized as additional capacity, they underscore the substantial
business impact of the system. Instead of doing legal research
from scratch, jurists can reallocate time to more complex cases
or direct interactions with citizens.

LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters ICAIL 2026, June 08–12, 2026, Singapore
•Content Accuracy and Consistency:We evaluated whether
AI letters correctly cited relevant laws and applied appropriate
legal reasoning. In our sample of approximately 250 cases, the
AI draft omitted a context-specific detail in a small number of
instances, which the jurist then corrected during the refinement
step or in the final edit.
We measured the retention of key facts by checking whether
each AI draft contained the critical factual elements of the case
(such as date of offense, type of violation, and location). In >
99% of the drafts, all key facts were present. The consistency
of legal reasoning (e.g., following the standard structure of pro-
portionality analysis, citing relevant municipal ordinances) was
maintained across drafts, which helped to standardize letters.
A notable side effect was that AI-drafted letters tended to be
more comprehensive than human-written ones. In 68% of the
cases, AI letters were longer, with on average about 41% more
content, often in the form of additional justifications and re-
sponses to more arguments raised in the objection. In 32% of
cases, human letters were longer (by about 12%). This suggests
that the model, drawing on a broad range of previous cases,
sometimes introduced arguments that an individual jurist might
not have recalled. Jurists generally viewed this as positive, as
one noted,“the AI output sometimes raised a point I had not
initially considered, which made the letter stronger. ”
Qualitative Evaluation: We gathered feedback through surveys
and interviews with the legal officers who piloted the tool, as well
as additional stakeholders (other jurists, department managers,
and subject-matter experts in legal technology). The feedback was
positive with respect to time and quality. Jurists reported that
LegalCheck’s draft letters required only minimal editing and that
the tool“clearly speeds up the handling of cases”, especially for com-
mon objection scenarios. They noted that AI was adept at main-
taining formal tone and even injecting empathy appropriately for
example acknowledging the citizen’s perspective politely. Some
were initially skeptical about trusting an AI for legal writing, but
after reviewing the output, their trust increased. A jurist said“I
was looking for mistakes, but I mostly found correct references and
consistent reasoning after a few letters I became more confident in
it. ”Indeed, a couple of jurists who started as cautious evaluators
became enthusiastic adopters at the end of the pilot, actively inte-
grating the tool into their daily workflow. This echoes the general
observations that practitioners are more willing to trust AI as a
support tool once they see that it works reliably [36].
We explicitly solicited criticisms and concerns. The primary
concern was the need for oversight: jurists were clear that AI, no
matter how good it is, should not be used without human review
(a stance we fully endorse and built into the process). They felt
comfortable with the tool as“the first drafter”, but they asserted
that final judgments must remain with human jurists, especially for
borderline cases or ones involving discretion. This aligns with our
expert-in-the-loop design and with public sector AI guidelines that
stress human final decision-making for legal matters. Another point
raised was about explainability: if challenged, could the department
explain how the letter was generated? We addressed this by logging
the sources and the prompt for each case, effectively creating an
audit trail of the AI’s inputs and outputs. This way, if needed, anEvaluation aspect Metric (definition) Result
Content coverage Recall 93%
Content relevance Precision 91%
Overall quality F1-score 92%
Draft usability Retention ratio (share of AI
draft kept verbatim after jurist
editing)81%
Editing efficiency Time-saved ratio for drafting
stage92%
End-to-end efficiency Reduction in total case han-
dling time50–70%
Table 1: Evaluation results of LegalCheck compared to
human-written advice letters.
expert can reconstruct what information LegalCheck used and
why certain language or citations appear, providing some level
of explanation despite the LLM being a“black box.”Jurists also
suggested improvements such as integrating image analysis (some
cases include photo evidence, which is currently analyzed with
AI in early experiments) and expanding to other types of cases.
We have since started extending LegalCheck to the other domains
mentioned, with similarly promising early results.
We summarize the key evaluation outcomes in Table 1, which in-
cludes metrics such as content retention, time saved per document,
and user satisfaction ratings. In general, the evaluation confirmed
that LegalCheck meets its goals: it significantly accelerates the draft-
ing process while maintaining high legal quality, and it has been
well-received by the legal staff as a trustworthy assistant rather
than a replacement [ 14]. In the next section, we discuss in more
depth the implications of these findings for legal reasoning, the role
of the lawyer, and responsible AI use in law.
5 Discussion
5.1 Impact on Legal Reasoning and Professional
Roles
Effects on Legal Reasoning and Consistency:The deployment
of LegalCheck offers insight into how AI can augment legal rea-
soning in practice. By design, the system employs a case-based
reasoning approach: it draws on past cases to inform the solution
of a new case. This aligns with how lawyers traditionally work,
by finding precedents and analogies, but automates both the re-
trieval and the initial application of those precedents. An immediate
impact is increased consistency in legal advice letters. Since AI’s
output in each case is based on collective knowledge of previous
decisions, letters tend to reflect a more uniform application of rules
and arguments throughout the department. This can reduce inad-
vertent disparities between the way different staff write advice on
similar cases. It effectively standardizes best practices, as AI tends
to include the strongest and most citizen-oriented arguments and
citations present in the knowledge base. For legal professionals,
this means that their role shifts slightly from being sole content
generators to serving as content curators and reviewers [ 15]. They
spend less time on the assembly of legal arguments and more on
reviewing the AI’s draft, checking nuances, and making judgment
calls on edge cases. Importantly, the jurists remain the ultimate
decision-makers: the AI does not alter the outcome of a case (it
does not decide to uphold or overturn a fine on its own; it only

ICAIL 2026, June 08–12, 2026, Singapore van der Meer and Rossi
drafts the rationale for the decision that the human inputs as the
intended outcome).
Effects on Professional Roles, Skills and Learning:In our use
case, the legal outcome (uphold/waive the fine) was determined by
the human based on the facts; LegalCheck then assists in formulat-
ing the justification for that outcome. This is a crucial distinction
which maintains accountability with the human official. From the
interviews, lawyers felt empowered rather than threatened by the
tool, as it saved them from drudgery (“no more starting from a
blank page”) and from manually stitching together boilerplate text
blocks. This allowed them to focus on higher-level analysis and on
difficult cases that truly require their expertise.
This echoes themes in the legal AI literature that envision AI
systems handling a substantial portion of routine drafting and
research, allowing lawyers to focus on more complex high-stakes
tasks [ 31]. Jurists already use fixed text blocks and templates in their
daily work, but these boilerplates never fully capture the specific
constellation of facts and arguments in an individual case and must
then be manually adapted. LegalCheck goes beyond such template-
based support by selecting and rewriting relevant building blocks
in light of the full case context, producing a coherent, case-specific
explanation. We also observed that junior staff benefitted from AI
drafts as a learning aid; a junior jurist said it was like having a
“reference draft”from a seasoned colleague, which helped them
improve their own writing by example. Over time, as such tools
become common, the skill set for legal professionals may evolve,
and proficiency in working with AI (prompting it effectively and
critically evaluating its output) will become important, much like
proficiency with legal research databases is today.
Trust, Adoption and External Perception:Another significant
impact is on the efficiency of legal research. Tasks that previously
required combing through case files are now completed in sec-
onds by the retrieval module. In the pilot, jurists noted that they
discovered relevant past cases that they would not have found or
remembered themselves because the semantic search could surface
matches even when terminology differed. This improved the thor-
oughness of legal reasoning, as the advice letters could reference
more precedents. Indeed, LegalCheck integrates a legal research
engine and a drafting assistant into one seamless workflow.
Trust in AI-supported decision-making is paramount in the legal
domain. At the outset, we encountered healthy skepticism from
some jurists, who were concerned that AI might make subtle errors
or produce text that they could not fully trust [ 3]. Through the
expert-in-the-loop design and the consistent quality of the output,
this skepticism was largely overcome. Key trust-building features
were the inclusion of sources (which made jurists comfortable that
they could verify any statement) and the requirement that they
approve every letter (maintaining a sense of control). This confirms
research suggesting that users prefer AI as a support tool rather
than an autonomous agent in professional settings. Our approach
led jurists to treat AI almost like a colleague or first-draft writer:
useful but always subject to review. During the pilot, we saw a
progression: initial caution gave way to routine use, and by the end
some jurists remarked that they“would not want to go back”to the
old way of drafting. This level of acceptance is encouraging, but
was facilitated by transparent communication about the capabilitiesand limitations of AI, training sessions, and management support.
Department leaders emphasized that AI was introduced to support
lawyers rather than to evaluate them, and this cultural framework
helped reduce fears about automation. We recommend any similar
AI introduction follow a participatory approach: involve end-users
early, allow them to test and give feedback, and iterate the tool
accordingly.
One question often raised is how the use of AI in legal drafting
might be perceived by external parties (citizens or judges). In our
context, the final letters did not explicitly disclose that AI was
involved, since letters are always reviewed by a human. The content
was virtually indistinguishable from human-written letters. Thus,
there was no observable change in the way letters would appear in
subsequent legal proceedings. In the long run, should AI drafting
become commonplace, there may be discussions about whether to
disclose AI assistance for transparency.
5.2 Responsible AI and Compliance
Considerations
Privacy and Data Protection:Implementing LegalCheck in a
public sector context required careful attention to ethical and legal
compliance principles from the beginning. We address privacy by
ensuring that sensitive personal data are protected beyond what
is necessary [ 11]. Where feasible, data sent to the OpenAI API is
minimized and stripped of personal identifiers. In addition, under
the agreement of the municipality with the cloud provider, API data
is not used to train models and are processed under the configured
Azure OpenAI privacy and compliance settings, including encryp-
tion, access controls, and logging/retention controls. Additionally,
because the knowledge base consists of internal documents, we
hosted the embedding and retrieval components on a secure munic-
ipal server. For each request, only the specific retrieved passages
needed for answering are sent to Azure for LLM inference, rather
than the full knowledge base.
Transparency, Accountability and Risk Management:On the
ethical and transparency front, the traceability of AI decisions is a
key focus. As described, the system logs sources and outputs. This
not only aids internal audits, but also contributes to explainability:
even if we cannot explain the neural mechanisms of the GPT model,
we can explain what information it used and what instructions it
received, which is often sufficient in a legal setting to justify the
basis of an advice. To further enhance transparency, we structured
the AI’s output to reflect a logical legal argument, making it easier
for a human reviewer to follow and for an external observer to
understand the reasoning. In essence, the role of AI is disclosed
internally and all drafts are reviewed, providing accountability [ 1].
We also proactively aligned the project with the requirements
of the forthcoming EU AI Act, although the law is not yet in force
in 2025 [ 33]. We performed a LegalCheck risk classification and
determined that it falls into a likely “low-risk” category: an AI
system for administrative decision support (not fully autonomous
decision-making, not biometrics or surveillance, etc.) [ 29]. Low-risk
does not mean no risk, so we implemented voluntary safeguards:
documentation of the purpose and limitations, keeping jurists in
charge of decisions, and monitoring for errors or biases. The depart-
ment’s compliance team reviewed LegalCheck and approved the

LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters ICAIL 2026, June 08–12, 2026, Singapore
use of the OpenAI model through Azure, noting that it should use
minimal personal data and that legal validity checks must remain
in place. We also considered bias/fairness: since the system relies
on past decisions, there is a risk that it could perpetuate any biases
in those decisions. To mitigate this, we had legal experts review a
sample of AI-generated letters for signs of inconsistent treatment.
No issues were found in our pilot domains, but this is something
to monitor in other legal domains (e.g., if applied to social benefits
cases, one would watch for any bias against certain groups). Our
governance plan includes periodic audits of outputs if the system
is scaled up.
User Training and Organizational Measures:Finally, we em-
phasize user training and the organizational measures taken. All
jurists using LegalCheck attended training sessions on how the
system works, its intended use, and how to interpret its suggestions.
We created a user guide that explicitly instructs:“AI is a drafting
aid. You must read and correct the AI draft as you would review a
junior colleague draft. You are responsible for the final content. ”By
setting clear expectations, we ensured that the tool is used properly
and does not erode the sense of responsibility. We also set up a
feedback channel so that users could report any problematic AI
output to developers and data scientists. During the pilot, a few
minor bugs related to formatting or missing context were identi-
fied and promptly fixed, demonstrating the importance of having a
responsive support process.
In summary, LegalCheck’s implementation was accompanied by
a comprehensive and responsible AI framework: privacy-by-design,
human oversight, traceability, compliance with regulatory guide-
lines, and multidisciplinary governance oversight [ 35]. We believe
that this is a model for how to introduce AI into government work-
flows in a manner that maintains public trust [ 37]. This included
leadership support and strong cross-functional collaboration across
legal, data, ethics, and IT teams[18].
6 Conclusion
Academic and Practical Implications:We presented LegalCheck,
a retrieval-augmented generation system with a multi-stage context-
augmented generation pipeline to support municipal jurists in draft-
ing objection advice letters. In a pilot in the City of Amsterdam,
LegalCheck produced legally grounded drafts substantially faster
than manual drafting while keeping the jurist fully responsible for
the final content. [27].
For the AI-and-law research community, LegalCheck contributes
a case study of deploying a RAG-based system, enriched with multi-
stage CAG, in an environment that demands high accountabil-
ity. The results underscore the importance of domain-specific and
organization-specific data: instead of relying on a generic GPT de-
ployment, our model was “educated” in the municipality’s own
jurisprudence, which was crucial to its performance. Conceptually,
we contribute a multi-stage form of context-augmented generation
for legal text: the model conditions on rich case context at the ini-
tial generation stage and then uses expert feedback and patterns
from analogous past cases in subsequent refinement passes. This
opens avenues for further work on interactive, human-guided le-
gal text generation, in which LLMs operate as adaptive drafting
partners rather than one-shot generators. Our approach resonateswith current ideas of AI “reasoning models” coupled with law re-
trieval systems and provides a concrete implementation of such
architectures in a public-sector setting. In addition, our evaluation
design, which combines quantitative text-level metrics with quali-
tative insights on trust, workflow integration, and perceived value,
offers a template for assessing legal AI tools beyond standard NLP
benchmarks.
Limitations:Our study was conducted in a single municipal orga-
nization, in four specific enforcement domains, and with a relatively
small group of legal professionals. Generalization to other legal do-
mains (e.g., criminal law, civil litigation) therefore requires caution.
Different document types (briefs, judicial opinions, contracts) have
their own conventions and ethical stakes. However, the architec-
ture of LegalCheck is domain-agnostic; it can be reconfigured for
other use cases if a relevant case repository is available. Another
limitation is reliance on a third-party LLM. We mitigated these risks
by using Azure’s enterprise-grade offering and abstracting the LLM
layer in our code, making the system model-agnostic. Indeed, as
open-source LLMs improve, a future iteration might use an in-house
model to reduce dependency. We did not systematically examine
the longer-term effects on employment, professional identity, or
perceived fairness from the citizen’s perspective. Our focus was on
augmenting a team facing staff shortages and growing backlogs,
and the tool was perceived to reduce workload pressure rather than
threaten jobs. However, as these tools become more widespread,
an ongoing dialog with stakeholders, including potential citizens,
will be needed to navigate the balance between efficiency gains and
(only) the human touch in public services.
Future Work:We are extending LegalCheck to additional case
types (e.g. objections to permits for short-term vacation rentals and
building permits) to test scalability. Pilots on current case types
have shown similar results, although prompt configurations and
retrieval filters required adjustment to reflect domain-specific ter-
minology and decision patterns. A technical priority is to further
develop image analysis using the multimodal GPT-4o capabilities
of GPT-4o, using the vision-language model to produce structured
descriptions of such evidence that can be incorporated into the tex-
tual prompts for both the RAG and CAG stages of the pipeline. We
also intend to develop a more advanced user interface that supports
inline editing (that the jurist can edit the draft directly and the
system logs those edits to learn from them). On the research side,
we aim to formally evaluate the multi-stage CAG process itself, by
comparing one-pass RAG generation with RAG plus CAG refine-
ment in a blind study, asking jurists to rate letters on legal quality,
completeness, clarity, and citizen-friendliness without knowing
which condition produced them. We also intend to monitor longer-
term organizational effects, such as changes in throughput, backlog
levels, perceived workload, and the evolution of roles and skills
within the legal team. These sociotechnical aspects are essential
to understand the broader impact of AI-assisted writing on public
services and the transformation of legal work.
With regard to reproducibility and collaboration, the underlying
case data are confidential, but we are exploring the creation of an
anonymized synthetic dataset that mirrors the structure and logic
of the real objection letters without disclosing sensitive content.
This resource could serve as a reference corpus for the RAG and

ICAIL 2026, June 08–12, 2026, Singapore van der Meer and Rossi
CAG methods in the generation of legal texts. We have modularized
key components of the system, including the retrieval module, the
prompt-construction logic, evaluation scripts, and the configuration
layer for domain-specific pipelines, and we are currently exploring
how these artifacts can be shared for research purposes. By sharing
these artifacts, we hope to facilitate further work on legal AI systems
that are both technically rigorous and responsibly deployed.
Usage of Generative AI
Generative AI tools were used to refine the authors writing of this
paper and to generate figures or tables.
Acknowledgments
Generative AI tools were used to refine the authors writing of this
paper and to generate figures or tables.
References
[1]AI4Citizens. 2025. Ethical leaflet: Get transparency about the moral
implications of technology used. Interreg Europe – Good prac-
tices. https://www.interregeurope.eu/good-practices/ethical-leaflet-get-
transparency-about-moral-implications-of-technology-used
[2]Nikolaos Aletras, Dimitrios Tsarapatsanis, Daniel Preotiuc-Pietro, and Vasileios
Lampos. 2016. Predicting Judicial Decisions of the European Court of Human
Rights: A Natural Language Processing Perspective.PeerJ Computer Science2
(2016), e93. doi:10.7717/peerj-cs.93
[3]Saar Alon-Barkat and Madalina Busuioc. 2023. Human–AI Interactions in Public
Sector Decision Making: “Automation Bias” and “Selective Adherence” to Algo-
rithmic Advice.Journal of Public Administration Research and Theory33, 1 (Jan.
2023), 153–169. doi:10.1093/jopart/muac007
[4]Saleema Amershi, Daniel Weld, Mihaela Vorvoreanu, Adam Fourney, Besmira
Nushi, Penny Collisson, Jina Suh, Shamsi Iqbal, Paul N. Bennett, Kori Inkpen,
and Jaime Teevan. 2019. Guidelines for Human–AI Interaction. InProceedings of
the 2019 CHI Conference on Human Factors in Computing Systems (CHI ’19). ACM.
doi:10.1145/3290605.3300233 Article 3, pp. 1–13.
[5]Alejandro Barredo Arrieta, Natalia Díaz-Rodríguez, Javier Del Ser, Adrien
Bennetot, Siham Tabik, Alberto Barbado, Salvador García, Sergio Gil-López,
Daniel Molina, Richard Benjamins, Raja Chatila, and Francisco Herrera. 2020.
Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities
and challenges toward responsible AI.Information Fusion58 (2020), 82–115.
doi:10.1016/j.inffus.2019.12.012
[6]Kevin D. Ashley. 2017.Artificial Intelligence and Legal Analytics: New Tools for
Law Practice in the Digital Age. Cambridge University Press.
[7]Rishi Bommasani et al .2021.On the Opportunities and Risks of Foundation Models.
Technical Report. Stanford Institute for Human-Centered Artificial Intelligence.
https://arxiv.org/abs/2108.07258 arXiv:2108.07258.
[8]Colleen V. Chien and M Kim. 2024. Generative AI and Legal Aid: Results from a
Field Study and 100 Use Cases to Bridge the Access to Justice Gap. SSRN Working
Paper (UC Berkeley Public Law Research Paper; forthcoming in Loyola of Los
Angeles Law Review). https://ssrn.com/abstract=4733061
[9]Thomas H. Davenport and Rajeev Ronanki. 2018. Artificial Intelligence for the
Real World.Harvard Business Review96, 1 (2018), 108–116.
[10] European Commission: Directorate-General for Communications Networks, Con-
tent and Technology and High-Level Expert Group on Artificial Intelligence. 2019.
Ethics Guidelines for Trustworthy AI. doi:10.2759/346720
[11] European Union. 2016. Regulation (EU) 2016/679 (General Data Protection
Regulation). Official Journal of the European Union, L 119, 1–88. https://eur-
lex.europa.eu/eli/reg/2016/679/oj
[12] Samer Faraj, Stella Pachidi, and Karim Sayegh. 2018. Working and organizing
in the age of the learning algorithm.Information and Organization28, 1 (2018),
62–70. doi:10.1016/j.infoandorg.2018.02.005
[13] Gemeente Amsterdam. 2024. Amsterdam’s vision on AI (English version). https:
//www.amsterdam.nl/innovatie/amsterdamse-visie-ai/
[14] T. Haesevoets, B. Verschuere, and A. Roets. 2025. AI adoption in public ad-
ministration: Perspectives of public sector managers and public sector non-
managerial employees.Government Information Quarterly42, 2 (2025), 102029.
doi:10.1016/j.giq.2025.102029
[15] Yousra Hashem, Jonathan Bright, Shreya Chakraborty, Kait Onslow, James Fran-
cis, A. Poletav, and S. Esnaashari. 2025. Mapping the Potential: Generative AI and
Public Sector Work. Using time use data to identify opportunities for AI adoption
in Great Britain’s public sector. https://www.turing.ac.uk/sites/default/files/2025-
05/ons_tus_final_report.pdf[16] Kenneth Holstein, Jennifer Wortman Vaughan, Hal Daumé III, Miro Dudik, and
Hanna Wallach. 2019. Improving Fairness in Machine Learning Systems: What Do
Industry Practitioners Need?. InProceedings of the 2019 CHI Conference on Human
Factors in Computing Systems (CHI ’19). ACM. doi:10.1145/3290605.3300830 pp.
1–16.
[17] Daniel Martin Katz, Michael J. Bommarito, and Josh Blackman. 2017. A General
Approach for Predicting the Behavior of the Supreme Court of the United States.
PLoS ONE12, 4 (2017), e0174698. doi:10.1371/journal.pone.0174698
[18] John P. Kotter. 1996.Leading Change. Harvard Business School Press.
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems 33. 9459–9474.
[20] Chih-Hao Lin and Pei-Ju Cheng. 2024. Legal documents drafting with fine-
tuned pre-trained large language model. InProceedings of the 12th International
Conference on Software Engineering & Trends (SE 2024). Copenhagen, Denmark.
doi:10.48550/arXiv.2406.04202
[21] Varun Magesh, F. Surani, M. Dahl, Mirac Suzgun, Christopher D. Manning, and
Daniel E. Ho. 2024. Hallucination-free? Assessing the reliability of leading AI
legal research tools. arXiv preprint. doi:10.48550/arXiv.2405.20362
[22] Ikhtiyor Nematov, Tarik Kalai, Elizaveta Kuzmenko, Gabriele Fugagnoli, Dimitris
Sacharidis, Katja Hose, and Tomer Sagi. 2025. Source Attribution in Retrieval-
Augmented Generation.CoRRabs/2507.04480 (2025). https://arxiv.org/abs/2507.
04480 arXiv preprint.
[23] OpenAI. 2024. Hello GPT-4o. https://openai.com/nl-NL/index/hello-gpt-4o/
[24] PwC. 2023. Half of Dutch jobs might be significantly changed by gen-
erative AI. PwC Netherlands. https://www.pwc.nl/en/insights-and-
publications/themes/the-future-of-work/half-of-dutch-jobs-might-be-
significantly-changed-by-generative-ai.html
[25] Jirui Qi, Gabriele Sarti, Raquel Fernández, and Arianna Bisazza. 2024. Model
Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Gen-
eration. InProceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing. Association for Computational Linguistics, Miami, Florida,
USA. doi:10.18653/v1/2024.emnlp-main.347
[26] Daniel Schwarcz, Sam Manning, Patrick J. Barry, David R. Cleveland, J. J. Prescott,
and Beverly Rich. 2025.AI-Powered Lawyering: AI Reasoning Models, Retrieval
Augmented Generation, and the Future of Legal Practice. Technical Report. Min-
nesota Legal Studies Research Paper No. 25-16 (SSRN). Available at SSRN:
https://ssrn.com/abstract=5162111.
[27] R. SenGupta. 2025. Legal sector’s adaptability proves the key to success. Financial
Times. https://www.ft.com/content/6df512fe-c1a7-4ed1-be3d-d3dc5e5b2944
[28] Peizhang Shao, Linrui Xu, Jinxi Wang, Wei Zhou, and Xingyu Wu. 2025. When
Large Language Models Meet Law: Dual-Lens Taxonomy, Technical Advances,
and Ethical Governance. doi:10.48550/arXiv.2507.07748
[29] Benedict Sheehy and Yee-Fui Ng. 2024. The Challenges of AI-Decision-Making in
Government and Administrative Law: A Proposal for Regulatory Design.Indiana
Law Review57, 3 (June 2024), 665–699. doi:10.18060/28360
[30] Harry Surden. 2019. Artificial Intelligence and Law: An Overview.Georgia State
University Law Review35, 4 (2019), 1305–1337.
[31] Richard Susskind. 2019.Tomorrow’s Lawyers: An Introduction to Your Future(2nd
ed.). Oxford University Press.
[32] Thomson Reuters. 2025. Less drudge, more expertise: How AI is redefining
the future of legal professionals in Australia. The Guardian (Thomson
Reuters AI Futures). https://www.theguardian.com/thomson-reuters-ai-
futures/2025/jul/21/less-drudge-more-expertise-how-ai-is-redefining-the-
future-of-legal-professionals-in-australia
[33] European Union. 2024. Regulation (EU) 2024/1689 laying down harmonised rules
on artificial intelligence (AI Act).Official Journal of the European Union(2024).
[34] Vereniging van Nederlandse Gemeenten (VNG). 2024. Pilot big data & AI-
tools voor efficiëntere afhandeling bezwaarschriften. VNG website (Oct 29,
2024). URL: https://vng.nl/artikelen/pilot-big-data-ai-tools-voor-efficientere-
afhandeling-bezwaarschriften.
[35] Andrei F. Vatamanu and M. Tofan. 2025. Integrating artificial intelligence into
public administration: Challenges and vulnerabilities.Administrative Sciences15,
4 (2025), 149. doi:10.3390/admsci15040149
[36] Viswanath Venkatesh, Michael G. Morris, Gordon B. Davis, and Fred D. Davis.
2003. User Acceptance of Information Technology: Toward a Unified View.MIS
Quarterly27, 3 (2003), 425–478.
[37] S. Weerts. 2025. Generative AI in public administration in light of the regulatory
awakening in the US and EU.Cambridge Forum on AI: Law and Governance(2025),
e3. doi:10.1017/cfl.2024.10
[38] N. Wiratunga, R. Abeyratne, L. Jayawardena, K. Martin, S. Massie, I. Nkisi-Orji,
R. Weerasinghe, A. Liret, and B. Fleisch. 2024. CBR-RAG: Case-Based reasoning
for retrieval augmented generation in LLMs for legal question answering. arXiv
preprint. doi:10.48550/arXiv.2404.04302
[39] L. Wrzesniowska. 2023.Can AI make a case? AI vs. lawyer in the Dutch legal context.
Master’s thesis. University of Amsterdam. Master’s thesis; later appearing in

LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters ICAIL 2026, June 08–12, 2026, Singapore
The International Journal of Law, Ethics, and Technology.
[40] Liming Zhu, Qinghua Lu, Ding Ming, Sung Une Lee, and Chen Wang. 2025.
Designing Meaningful Human Oversight in AI. doi:10.2139/ssrn.5501939 SSRNworking paper.