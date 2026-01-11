# SoK: Privacy Risks and Mitigations in Retrieval-Augmented Generation Systems

**Authors**: Andreea-Elena Bodea, Stephen Meisenbacher, Alexandra Klymenko, Florian Matthes

**Published**: 2026-01-07 14:50:41

**PDF URL**: [https://arxiv.org/pdf/2601.03979v1](https://arxiv.org/pdf/2601.03979v1)

## Abstract
The continued promise of Large Language Models (LLMs), particularly in their natural language understanding and generation capabilities, has driven a rapidly increasing interest in identifying and developing LLM use cases. In an effort to complement the ingrained "knowledge" of LLMs, Retrieval-Augmented Generation (RAG) techniques have become widely popular. At its core, RAG involves the coupling of LLMs with domain-specific knowledge bases, whereby the generation of a response to a user question is augmented with contextual and up-to-date information. The proliferation of RAG has sparked concerns about data privacy, particularly with the inherent risks that arise when leveraging databases with potentially sensitive information. Numerous recent works have explored various aspects of privacy risks in RAG systems, from adversarial attacks to proposed mitigations. With the goal of surveying and unifying these works, we ask one simple question: What are the privacy risks in RAG, and how can they be measured and mitigated? To answer this question, we conduct a systematic literature review of RAG works addressing privacy, and we systematize our findings into a comprehensive set of privacy risks, mitigation techniques, and evaluation strategies. We supplement these findings with two primary artifacts: a Taxonomy of RAG Privacy Risks and a RAG Privacy Process Diagram. Our work contributes to the study of privacy in RAG not only by conducting the first systematization of risks and mitigations, but also by uncovering important considerations when mitigating privacy risks in RAG systems and assessing the current maturity of proposed mitigations.

## Full Text


<!-- PDF content starts -->

SoK: Privacy Risks and Mitigations in
Retrieval-Augmented Generation Systems
Andreea-Elena Bodea∗†, Stephen Meisenbacher∗‡, Alexandra Klymenko§, and Florian Matthes¶
Technical University of Munich
School of Computation, Information and Technology
Garching, Germany
Email:†andreea.bodea@tum.de,‡stephen.meisenbacher@tum.de,§alexandra.klymenko@tum.de,¶matthes@tum.de
Abstract—The continued promise of Large Language Models
(LLMs), particularly in their natural language understanding
and generation capabilities, has driven a rapidly increasing
interest in identifying and developing LLM use cases. In an effort
to complement the ingrained “knowledge” of LLMs, Retrieval-
Augmented Generation (RAG) techniques have become widely
popular. At its core, RAG involves the coupling of LLMs with
domain-specific knowledge bases, whereby the generation of a
response to a user question is augmented with contextual and
up-to-date information. The proliferation of RAG has sparked
concerns about data privacy, particularly with the inherent risks
that arise when leveraging databases with potentially sensitive
information. Numerous recent works have explored various
aspects of privacy risks in RAG systems, from adversarial attacks
to proposed mitigations. With the goal of surveying and unifying
these works, we ask one simple question:What are the privacy
risks in RAG, and how can they be measured and mitigated?To
answer this question, we conduct a systematic literature review of
RAG works addressing privacy, and we systematize our findings
into a comprehensive set of privacy risks, mitigation techniques,
and evaluation strategies. We supplement these findings with two
primary artifacts: a Taxonomy of RAG Privacy Risks and a RAG
Privacy Process Diagram. Our work contributes to the study of
privacy in RAG not only by conducting the first systematization
of risks and mitigations, but also by uncovering important
considerations when mitigating privacy risks in RAG systems
and assessing the current maturity of proposed mitigations.
Index Terms—privacy, RAG, natural language processing, risk
mitigation, systematic review.
I. INTRODUCTION
With the seemingly ubiquitous recent advances in the areas
of Artificial Intelligence and Natural Language Processing,
predominantly spearheaded by modern Large Language Mod-
els (LLMs), the number of innovative use cases leveraging
LLMs has grown at a likewise unfathomable rate [1]–[3].
LLMs have been pushed far beyond chatbots and translation
tools, with impressive heights being reached in reasoning
abilities, coding, multimodal generation, and agentic tasks.
With this plethora of promising use cases, one persistent
challenge with using LLMs is the inherent fact that these
models are static and must inevitably be restricted by some
“knowledge cutoff” [4], i.e., the most recent point in time at
which the data used to train a model was collected, as well as
other technical knowledge boundaries such as context size.
∗These authors contributed equally.As a direct answer to this important problem, the Retrieval-
Augmented Generation (RAG) paradigm [5] has starkly risen
in prevalence due to its simple yet effective method for
incorporating external knowledge into generation with LLMs.
By coupling such knowledge bases, which might contain
domain-specific information or otherwise previously unseen
data, the ability of LLMs to utilize information in context
can be effectively leveraged to create query-response systems
for answering user questions in an up-to-date and informed
manner [6], [7]. This plug-and-play method empowers users
with knowledge bases to unlock the information contained
within, and to make this knowledge accessible to others.
With this paradigm of interacting with LLMs, however, new
risks are introduced when coupling private information with
LLMs, and by extension, the RAG systems built around them.
Concerns of data privacy arise when considering the direct
interfacing of LLMs with potentially sensitive data contained
within the connected databases, particularly in light of known
LLM privacy issues [8]. Such risks, if exploited by malicious
users, may result in the exposure of private information or the
incorrect functioning of the RAG system, thus undermining
the demonstrated promise of RAG [6], [9], [10].
Many recent works have acknowledged the privacy risks
in RAG [11], [12]. While some works focus on exploring
potential privacy risks, others propose specific methods for
risk mitigation. Despite these existing works, however, there
remains a lack of systematization of privacy risks in RAG, and
moreover, of how these risks can be measured and mitigated.
In this, we see it as crucial for researchers and practitioners
alike to have a unified overview of privacy in RAG, and the
lack of systematization despite numerous works in the field
points to an important and timely research gap.
We strive to understand the scope of privacy risks in
RAG, uncovering the various ways in which recent works
have measured privacy risks, particularly in the evaluation of
proposed mitigation strategies. To gain such an overview, we
conduct a Systematic Literature Review (SLR) of 72 recent
papers at the intersection of RAG and privacy, studying the
investigated risks, mitigations, and evaluation strategies of
these works. We systematize this collection of 72 papers (Table
II) and assess current mitigations (Table III). The findings of
this survey lead to the creation of a Taxonomy of RAG Privacy
Risks (Figure 2), which enumerates risks and maps them toarXiv:2601.03979v1  [cs.CR]  7 Jan 2026

potential mitigations, and a RAG Privacy Process Diagram
(Figure 3), which illustrates a dynamic view of where along
the RAG pipeline risks materialize and can be mitigated.
Our literature survey teaches us that while many of the
privacy risks associated with RAG can be considered under the
umbrella ofinformation leakageandattacks, these take many
forms along the RAG pipeline. As such, the numerous types of
proposed mitigation strategies can each be mapped to specific
risks and, accordingly, to specific steps in the RAG process.
Current mitigation efforts, however, have received varying
degrees of attention, and we quantify both therelevanceand
maturityof privacy risk mitigations for RAG (Table III),
showing a disparity inproposedmitigations versus what may
be consideredmaturemitigations.
We contribute to the study of privacy in RAG as follows:
1) We conduct the first systematic study to survey known
RAG privacy risks, mitigations, and evaluation tech-
niques as proposed by the recent literature.
2) We offer two primary artifacts that systematize our main
findings from the literature review: a Taxonomy of RAG
Privacy Risks and a RAG Privacy Process Diagram.
3) We also provide a comprehensive mapping of proposed
mitigations to RAG privacy risks, and we quantify these
mitigations according to theirrelevanceandmaturity.
4) We publish a public repository of the surveyed papers,
grey literature sources, and complete literature analysis:
https://github.com/sebischair/SoK-RAG-Privacy.
II. FOUNDATIONS
RAG [5] is an advanced framework designed to enhance the
capabilities of LLMs by integrating external knowledge into
the generation process. These systems address some of the
inherent limitations of LLMs, such as hallucination, outdated
information, and limited domain specificity [13].
As displayed in Figure 1, the RAG process is characterized
by three distinct stages: indexing, retrieval, and generation.
During the indexing stage, raw data found in internal docu-
ments or external sources from the internet are cleaned, seg-
mented into chunks, and converted into vector representations
using embedding models. These vectors are then stored in
a database that has been optimized for conducting similarity
searches. In the retrieval stage, the system receives a raw text
user query, encodes it into a vector, and searches for the most
semantically similar top-krelevant text chunks in the vector
database. Finally, during the generation stage, the retrieved
chunks are inserted into the LLM together with a pre-defined
prompt. The LLM then produces a response by leveraging both
its pre-trained knowledge and the additional retrieved context.
The RAG paradigm has been well-received by both research
and practice, finding extensive use in a wide variety of domains
and scenarios [7], [14]. This includes, but is not limited
to, dialogue systems, translation, and summarization, as well
as coding and drug discovery. In addition to the “naive”
RAG pipeline [15], as depicted in Figure 1, there have been
numerous proposals for enhancements in various areas along
Fig. 1. A typical Retrieval-Augmented Generation system.
the pipeline [7], [15], including advanced retrieval techniques
and optimized hyperparameter selection.
As the RAG framework is not a single model but instead
a confluence of techniques, the study of RAG includes a
diverse variety of topics, which introduces complexities to
implementing real-world systems. Particularly with the re-
quirement to attach an external dataset for better response
contextualization, new challenges may arise, both technical
in nature (e.g., ensuring data quality) as well as increased
concerns of privacy (i.e., protecting sensitive data within the
provided database). Motivated by privacy aspects of RAG,
and the lack of systematization on what exactly this entails,
we seek to further the study of RAG with a comprehensive
overview of RAG privacy risks and mitigations.
First, however, we establish an operational definition ofpri-
vacy, as this becomes important to systematizing and analyzing
privacy risksin RAG systems.
Scope: We ground our study in two important notions:pri-
vateandconfidentialinformation, both of which play a role
in the context of RAG. Private and confidential information
differ in scope, context, and legal implications, though they
are often used interchangeably.Privateinformation refers to
the personal details of an individual or entity that are not
intended for public dissemination but are meant to remain
private. An example would be a person’s medical history.
Confidentialinformation also refers to sensitive data, but
that shared between parties under an explicit or implicit
agreement to be kept secret. An example would be propriety
company information on internal projects. It is important to
note that information can be private without being confi-
dential, and vice versa; nevertheless, in this work, we view
both privateandconfidential information under the umbrella
of “privacy”, or “data protection” (we do not distinguish
between the two). Accordingly, in the context of our survey,
we define privacy to meanthe safeguarding of private infor-
mation from improper disclosure and adversarial threats.
III. METHODOLOGY
In order to survey the current research on privacy in RAG,
we conduct an SLR following the framework established
by Kitchenham et al. [16]. The SLR enables the systematic
investigation of a body of work, and as the first step to guide
our investigation, we define one overarching research question:
RQ.What are the privacy risks in RAG systems, and what
techniques have been proposed to mitigate them?

Thus, we seek not only to discover risks and mitigations,
but also to explore a mapping between the two and to identify
strategies to measure the effectiveness of mitigations.
In line with the SLR methodology, we plan and conduct the
review according to the following steps:
1)Selecting Databases for Primary Sources: Both white
and grey literature [17] were utilized to ensure com-
prehensive coverage of this nascent topic. White liter-
ature sources included Google Scholar, ACM Digital
Library, and IEEE Xplore. Grey literature was collected
from Google Search and YouTube. This combination of
academic and non-academic sources ensured a balanced
perspective, capturing theoretical and practical insights.
2)Defining the Search Strings: We developed two search
strings to maximize relevant results while focusing the
scope of our search. In particular, both strings contain
two primary parts: terms referring to RAG and terms
related to privacy or adversarial attacks, following our
defined research scope. This resulted in the following:
S1: (“rag” OR “retrieval augmented” OR “augmented
generation”) AND (“private” OR ”privacy”)
S2: (“rag” OR “retrieval augmented” OR “augmented
generation”) AND (“attack”)
3)Conducting the Search: To maintain a relevant scope
for the eventual selected papers, we limited the search
results from Google Scholar to the first 150 results,
and Google Search and YouTube to the first 50 results
(five pages). Literature databases were limited to sources
published from 2020 onward, in order to include only
works after the formal introduction of RAG [5]. For
searching ACM DL and IEEE Xplore, the search strings
were applied to the title and abstract only. The final
search was performed in July 2025. Preprints were
included only if they had existed for more than a year.
4)Exclusion Criteria: Given the merged set of search re-
sults, exclusion criteria were applied via paper screening
to achieve the final literature set. Beyond cursory quality
checks and keeping only accessible sources, exclusion
criteria involved removing duplicate studies and filtering
out articles that do not explicitly address privacy-related
issues in RAG systems. This was especially important
for Google Scholar, since it does not allow for abstract
searching. Therefore, a pre-filtering was performed on
all retrieved results to remove sources that were clearly
out of scope. The threshold of 150 was chosen after a
pre-screen that revealed nearly all results thereafter to
be clearly not relevant. Examples of irrelevant papers
retrieved were papers implementing RAG systems and
mentioning that privacy should be a concern without
elaboration, i.e., where RAG systems are proposed for
specific use cases for which privacy would be very
important, but without any further justification or ex-
perimentation of privacy implications. Other examples
include using RAG systems for LLM unlearning or
to create attack graphs. Following these filtering steps,a final collection of145white literature sources was
retained, as depicted in Table I. In addition, 48 Google
Search results and 6 YouTube videos were deemed to be
relevant; the complete list of grey literature sources is
provided in the supplemental material in our repository.
5)Relevance Check: Following exclusion, a more in-depth
relevance check was conducted with the 145 selected
sources. First, the abstract of each source was read to
determine immediate relevance. If this was not clear, the
full text of the work was screened, focusing on important
aspects such as motivation, experiments, and discussion.
If privacy was only covered tangentially, the source was
filtered out. Specifically, the relevance check steps were:
a) The title of the paper was read. If relevance was
clear, it was included. Otherwise,
b) The abstract of the paper was read. If relevance
became clear, it was included. Otherwise,
c) The paper full-text was screened. If relevance
became clear, it was included. Otherwise, it was
deemed irrelevant.
After this process, the final set of relevant papers con-
sisted of72sources. We note that the entire process
of filtering and relevance checking was done manually
without the assistance of automatic tools.
6)Data Extraction: Data extraction was primarily carried
out by the lead researcher. Screening each primary
source was prefaced by a reading of the abstract for
familiarization with the work. Then, keywords such as
‘privacy’ and ‘attack’ were searched for, in order to find
relevant points in the work for understanding the au-
thors’ perspective on privacy in RAG. Key information
was extracted in a structured manner, including explic-
itly mentioned privacy risks, proposed mitigations, and
experimental setup and evaluation. The included grey
literature sources served to augment the findings from
the white literature, often providing more accessible
explanations of risks and attacks. These uncovered in-
sights and other themes were discussed weekly with the
complete research team over the course of the literature
review process. A link to the complete structured data
extraction results can be found in our public repository.
7)Synthesizing Results: Each paper was annotated for
privacy relevance(1-3, with 3 being the most relevant),
privacy focus (data leakage or adversarial manipulation,
introduced next), and primary purpose (privacy attacks
or mitigations). The findings from the literature review
were also analyzed for three major categories following
data extraction: (1) RAG privacy risks and attacks, (2)
mitigation strategies for these privacy risks, and (3)
evaluation datasets, tasks, and metrics for measuring
privacy in RAG systems. This synthesis provided a
comprehensive understanding of the current research
landscape and highlighted gaps for further investigation.
We systematize in Table II the structured data extracted from
the 72 papers. These insights underpin our findings, detailed

TABLE I
AN OVERVIEW OF THE COLLECTED PRIMARY SOURCES FROM THESLR,
BOTH BEFORE AND AFTER APPLYING FILTERING AND EXCLUSION
CRITERIA,AS WELL AS FOLLOWING THE FINAL RELEVANCE CHECK.
S1 S2
Before After Before After
Google Scholar 150 43 150 72
ACM DL 16 9 1 1
IEEE Xplore 61 20 10 0
Before exclusion 388
After exclusion 145
Final relevant set 72
next, which include two main artifacts, the Taxonomy of RAG
Privacy Risks and the RAG Privacy Process Diagram.
IV. RAG PRIVACYRISKS ANDMITIGATIONS
We present the main findings from our survey of 72 litera-
ture sources, which take the form of privacyrisks,mitigations,
andevaluation strategies. We illustrate that the privacy risks
in RAG can be categorized into two major categories,leakage
andadversarial manipulation. As much of the surveyed litera-
ture also focuses on mitigations, we provide a direct mapping
of mitigation strategies to risk points along the RAG pipeline.
Finally, we break down evaluation strategies for measuring
privacy protection into two primary aspects:datasetsand
metricsused for evaluation.
A. A Taxonomy of RAG Privacy Risks
The review of relevant literature addressing privacy in RAG
systems revealed two predominant ways in which privacy risks
can be perceived. The first refers to the idea ofattacks, or
malicious attempts to disrupt, disable, or misuse RAG systems.
We call theseAdversarial Manipulationrisks. We include
these in our presented taxonomy, but as these attack vectors
exists on the boundary between security and privacy risks, we
only briefly introduce them below. We refer the reader to the
cited works (Figure 2) for more details.
•Jailbreak Attacksuse specially designed prompts or se-
quences to bypass a RAG system’s safety filters, enabling
the generation of harmful, toxic, or restricted content.
These attacks exploit the generative model’s contextual
sensitivity to subvert built-in content moderation policies.
•Backdoor Attacksintroduce malicious triggers during
training or fine-tuning, which remain dormant until ac-
tivated by specific inputs. In RAG systems, these may
persist even in retrieved document chunks.
•Data Poisoning Attacksinvolve corrupting training or
retrieval datasets through the malicious injection of adver-
sarial examples, mislabeled data, or misleading content.
•Prompt Injection Attacksinvolve embedding adversar-
ial content within prompts to manipulate system behavior.
These attacks exploit the interpretative flexibility of gen-
erative models, potentially causing the system to execute
unintended instructions or disclose sensitive information.
•Membership Inference Attacksaim to determine
whether a specific data point was a part of a model’straining set or, in the context of RAG, if such data is
present in the knowledge base. The retrieval component
of RAG can exacerbate this risk by exposing responses
tied to unique data samples.
•In aData Extraction Attack, adversaries exploit model
outputs to reconstruct sensitive data. These attacks chal-
lenge the privacy of both the retrieval and generative
components, especially in systems lacking robust access
controls or output sanitization.
•Prompt Extractionentails the reconstruction of user
prompts from system behavior or responses. Such attacks
threaten to enable unauthorized parties to access or infer
other users’ inputs, which can contain private or confi-
dential information.
•LLM Extraction/Inversion Attackstarget the underly-
ing parameters (i.e., knowledge representation) of lan-
guage models. By systematically querying a RAG system,
adversaries may infer embedded facts or even reconstruct
portions of the training corpus.
Key finding: A survey of privacy in RAG intersects with the
study ofadversarial manipulation, which covers a variety of
adversarial attacks that, in some form, may affect the privacy
of the end user or compromise the confidentiality of data held
by private entities. In the remainder of this work, we view
adversarial manipulation in tandem withleakagerisks.
The second aspect of privacy in RAG is complex, and it
relates to the idea ofleakageresulting from RAG’s inner
workings. Leakage poses a privacy risk in the potential for
exposure of sensitive information originating from the data
stored within the RAG system, or conversely, data inputted
into the system via user prompts that may be leaked at a
later point. Beyond this, we noticed that while many works
cover this topic nominally, few works describe explicitly what
form of data leakage they are protecting against, as opposed
to defined attack vectors as introduced above.
Due to the unique and sequential nature of the RAG
pipeline, however, leakage can originate from many points,
and we find that this origination point is directly tied to
the point at which mitigation strategies would be applied.
For example, mitigatingdataset leakagewould imply that
protections are implemented immediately at the database level,
further implying that the repercussions of such protections are
acceptable for the functioning of the ensuing pipeline. How-
ever, if personal information is required to retrieve relevant
encoded chunks in the vector database, mitigation measures
might be applied after this stage, thereby mitigatingretrieved
chunk leakage. We therefore learn of the significant distinction
between types of data leakage, which becomes important for
defining risks and mapping appropriate mitigation strategies.
The complete Taxonomy of RAG Privacy Risks is pre-
sented in Figure 2. We introduce the five uncovered types
of leakage risks in RAG systems, four of which relate to
the potential risks from data being passed through the RAG
system internally, and one relating to the risks of user-provided

RAG Privacy RisksLeakageData Leakage [8], [10], [12], [18]–[54]Dataset Leakage
Vector Database Leakage
Retrieved Chunk Leakage
Answer Leakage
Prompt Leakage [8], [25], [41], [51], [55], [56]
Adversarial ManipulationData Poisoning Attacks [8]–[10], [18]–[20], [28]–[30], [37], [38], [41], [42], [50], [51], [57]–[78]
Backdoor Attacks [8], [10], [28], [51], [57], [79], [80]
Data Extraction Attacks [8], [10], [12], [27], [40], [81]
Jailbreak Attacks [8]–[10], [29], [51], [82]
LLM Extraction/Inversion Attacks [8], [41]
Prompt Extraction Attacks [8], [56]
Membership Inference Attacks [8], [10], [18], [35], [36], [39], [41], [50], [83]
Prompt Injection Attacks [8], [25], [31], [40], [41], [51], [69], [72], [73], [80], [82], [84], [85]
Fig. 2. The Taxonomy of RAG Privacy Risks. While we highlight the two-sided nature of privacy risks in RAG, leakage and adversarial manipulation, we
focus specifically ondata leakageand its mitigation. Adversarial manipulation, or attacks, primarily serve as a means to realize threats posed by leakage.
data via prompts. In this, we also introduce associated attack
types, thus showing the interconnection between Leakage and
Adversarial Manipulation. Finally, we include our findings
on suggestions for mitigations as reported in the literature;
a mapping of mitigations to risks is found in Section IV-B.
1) Dataset leakage:Dataset leakage is an issue particularly
when proprietary or sensitive information is stored using
unsafe storage solutions. Leakage can occur through external
exposure, internal access control failures, or both.
One risk is accidental exposure of proprietary data due
to insecure storage practices. If contributors store sensitive
documents in unprotected cloud storage, shared drives, or
even email attachments, unauthorized individuals may gain
access. Unlike structured SQL databases with well-defined
access controls, many traditional storage solutions rely on
manual access management, increasing the likelihood of mis-
configurations that lead to data breaches. Moreover, publicly
available datasets, such as those scraped from the internet,
may unknowingly contain private information, blurring the line
between open-source and confidential data.
Another concern is inadequate internal access controls,
which can lead to both intentional and accidental data ex-
posure. In many organizations, employees in both technical
and non-technical departments may have unrestricted access
to all stored documents. This level of access poses multiple
risks. First, employees might unintentionally modify metadata
or tags, making critical documents unavailable or incorrectly
prioritized during retrieval. Second, unrestricted editing rights
could lead to the unintentional inclusion of sensitive data,
potentially affecting downstream generated responses.
Non-malicious data leakage often occurs when actors inad-
vertently include PII or other sensitive information in docu-
ments without realizing these files will later be indexed into the
RAG system. A major failure point is the insufficient removal
or masking of personal data. If PII is not properly sanitized,
confidential details such as names, addresses, phone numbers,or legal case specifics may become part of the system’s
retrieval process. This can lead to unauthorized exposure when
an AI model retrieves and presents sensitive information in
response to user queries.
Mitigations.Addressing dataset leakage in RAG systems
requires a multi-layered approach. Organizations must imple-
ment robust access control policies [18], [22], [54], ensuring
that only authorized personnel can view or edit sensitive
documents (and their associated vector embeddings). This
can be supplemented with distributed data storage solutions
or specialized cloud architectures [5], [55]. Automated PII
detection, filtering, and redaction tools (i.e.,anonymization)
should be integrated into the data ingestion pipeline to prevent
accidental exposure [10], [18], [27]. Beyond PII handling,
rewriting or rephrasing techniques can be used to modify the
original documents while maintaining their semantic meaning
[28], [37], [38], [59], [63], [68]–[70]. Alternatively, synthetic
data could be used in lieu of the original data, assuming
this achieves acceptable performance [33]. Furthermore, mon-
itoring mechanisms can help identify and mitigate the risks
associated with data poisoning and backdoor injection.
Key finding: We learn of two primary aspects of dataset
leakage in RAG systems, leaking data to RAG end users via
PII or sensitive information, and exposing sensitive or confi-
dential information to unauthorized internal users. While the
majority of the literature proposing mitigations focuses on
the former, such as through redaction or anonymization, there
has been much less attention paid to the latter. Furthermore,
no works investigate the interconnectedness of database
leakage mitigation, for example, how proper anonymization
can serve as a supporting tool to database access controls.
2) Vector database leakage:The risk of vector database
leakage stems from the cases when proprietary or sensitive
data is stored in vector databases. Unlike traditional databases,
vector storage enables powerful semantic search but also intro-

duces new risks. One critical issue is embedding model mem-
orization, where the model retains patterns from its training
data. If the embedding model has been exposed to proprietary
documents, attackers can probe the system with crafted queries
to retrieve sensitive information. This risk increases when
embeddings are not properly sanitized, potentially allowing
unauthorized users to reconstruct proprietary data from the
model’s learned representations.
A direct risk arises when sensitive documents are stored
and then retrieved without proper controls. If a query closely
matches confidential content, the system may return private
information embedded in the vector store. For example, a
request about financial agreements between companies could
unintentionally reveal contract terms. Furthermore, attackers
can refine their queries to bypass simple safeguards.
An overlooked risk is misconfigured database access, where
weak authentication or improper permission settings expose
stored embeddings or document chunks to unauthorized users.
Exploits thus can extract embeddings, reconstruct sensitive
data, or query for proprietary documents.
Mitigations.To protect embeddings, Differential Privacy
(DP) techniques and synthetic data can help mitigate model
memorization [21], [33]. To prevent the improper access of
sensitive information, the information can be redacted before
indexing, access can be restricted based on user roles [18],
[22], [54], and query filtering can help to block the retrieval
of classified content. Other proposals include the injection of
redundant non-sensitive examples into the vector stores, as
well as simple duplication.
Key finding: An important aspect of RAG systems is the
transformation of texts into a vector database store. Privacy
issues here are mainly rooted in the leakage of information
stored in vector embeddings, despite the inherent belief that
embeddings may successfully obfuscate data. While we find
prior works focusing on access control and user roles for
vector databases, we find relatively few works that propose
mitigations at the embedding level, outside of exploratory
works on DP and synthetic data. Importantly, we also find
no mention of protecting themappedtext in the vector
databases, i.e., the original text chunks to which the vectors
correspond. This calls for important future research that aims
to balance semantic coherence and privacy protection in
embedding representations and stored texts of RAG data.
3) Retrieved chunk leakage:Retrieved chunk leakage oc-
curs when private information is exposed in system responses
due to the retrieval of sensitive or proprietary content. This
issue arises when the retrieval process pulls confidential infor-
mation from stored documents, incorporates such information
in the response generation, and presents a potentially leaky
answer to users. One major risk is internal manipulation
by actors with access to the retrieval pipeline. Technical
stakeholders, if malicious, could manipulate retrieval processes
to prioritize certain chunks, leading to biased or unauthorized
exposure of confidential data.
Mitigations.Mitigating retrieval-based leakage requires ro-
bust retrieval strategies [12], [18], [33], [52] and distancemetrics [10], [12], [33] to ensure that retrieved chunks are
both relevant and safe for disclosure. This could be achieved
by altering indexing mechanisms, modifying metadata, or
improving the ranking step that determines which chunks are
most relevant to a query. For example, Differential Privacy
techniques can be applied at the cross-attention stage in
reranking, adding controlled noise to reduce the likelihood of
retrieving highly sensitive content [86].
Key finding: Another unique aspect to RAG systems is the
retrieval stage, in which chunks of text data are retrieved,
often via similarity of user prompts to stored embedding rep-
resentations. Initial work has been performed looking mainly
at mitigations in the employedretrieval strategy. We find,
however, that further research is warranted, especially as the
retrieval stage serves both as the “last line of defense” before
LLM answer generation and also as a potential privacy-
utility “balance point”, where utility loss from early risk
mitigation on the database level (e.g., via anonymization)
can be avoided while still preventing unwanted leakage to
the LLM and in the output to the user.
4) Answer leakage:As the final stage in the RAG pipeline,
answer leakage can occur when private or sensitive informa-
tion is unintentionally revealed in the response generated by
the system. Even if access to the retrieved chunks is restricted,
the LLM may still incorporate confidential data into its output,
leading to unintentional exposure.
One primary concern is the content of the generated answer
itself. If an LLM receives sensitive chunks without proper
filtering, it may produce responses that disclose confidential
or private data, such as confidential discussions. An equally
important risk is the storage of generated responses, partic-
ularly in logging systems, conversation histories, or cached
outputs. If responses with sensitive data are stored, they can
be retrieved in later queries or accessed by unauthorized users,
further exacerbating privacy risks.
Mitigations.To mitigate answer leakage, organizations can
consider local deployment of RAG models to ensure full
control over data handling and prevent external exposure. Im-
plementing response safeguards [58] such as post-processing
filters, fact-checking mechanisms [67], and structured valida-
tion can help detect and redact sensitive information before
it is displayed. Additionally, enforcing source citation allows
transparency, ensuring that sensitive responses are traced back
to their origins, making it easier to flag and prevent private
data from being included in outputs.
Key finding: Investigating privacy risks at the answer gener-
ation stage in RAG systems is complex, in the way that data
leakage can be propagated from retrieved chunksas well as
from ingrained sensitive data in LLMs that may be triggered
by certain RAG inputs. Current proposed mitigations seem
to be heavily geared towards the risk that the LLMs pose,
where local LLM deployment and guardrails comprise the
vast majority of methods. This, however, leaves a relatively
wide gap in mitigating data leakedfrom the RAG system
itself, i.e., from the (vector) database and retrieved chunks.

5) Prompt leakage:Prompt leakage is another critical pri-
vacy concern in RAG systems, particularly when user prompts
contain sensitive information. This is especially pertinent when
a RAG system logs or stores prompts for future reference. If
queries contain private information, they may persist in conver-
sation history, be cached for optimization, or even be retained
in memory, potentially making them accessible in unintended
contexts. This becomes especially problematic when responses
based on those queries and retrieved chunks are also stored.
A user within the same session may unintentionally retrieve
sensitive information from previous exchanges, and another
user, whether intentionally or not, could later trigger references
to previously stored prompts or responses.
Mitigations.To enhance privacy, one can integrate mecha-
nisms such as anonymization, PII removal, and query filtering
to prevent sensitive data from persisting in stored queries [10],
[18], [27]. Paraphrasing, or rewriting prompts before process-
ing them can further reduce risks while maintaining query
intent [33], [37], [59]. Additionally, privacy-aware models
with augmented prompts can help models to recognize and
redact sensitive input dynamically. Finally, prompts can be
distributed using techniques such as Multi-Party Computation
[55], ensuring that no single server receives the entire prompt
with potentially sensitive information.
Key finding: While mitigating privacy risks from user inputs
to RAG systems may appear to be the most straightforward
of the uncovered leakage types, we observe in the literature
that the focus primarily lies on scrutinizing the content of the
prompt text itself. On the other hand, although the literature
points out that thecontextsurrounding the user prompt
is important, namely in determining which knowledge and
information a particular user is privy to, there is a scarcity
of mitigations that make these considerations. This also
becomes a crucial factor in ensuring that storage logs of
user prompts are privatized correctly.
B. Mapping Mitigation Strategies
In Table III, we summarize the RAG privacy risk mitigations
introduced above, and we map these strategies to the specific
stage in the RAG pipeline where they might be implemented.
Thus, each mitigation is directly associated with the type of
leakage it can prevent, which is important for researchers
and practitioners, not only for designing proper mitigation
strategies, but also for better understanding the implications
of placing privacy solutions at different points in the RAG
process. Although we also map mitigations to adversarial
attacks (Table III), we discuss in Section IV-D that mitigating
attacks and leakage can be viewed in tandem.
To perform such a mapping for each of the relevant papers
proposing mitigations, we extracted the stage in the RAG
pipeline where the risk occurs, i.e., where the mitigation is
applied, and the primary threat to which it is linked (e.g., data
leakage). It is important to note that we mapped a mitigation
to a stage only if the paper explicitly applies or discusses itTABLE II
ASYSTEMATIZATION OF THE72REVIEWED PAPERS. WE CLASSIFY EACH
OF THE PAPERS ALONG SEVERAL AXES:YEAR OF PUBLICATION,NUMBER
OF CITATIONS(AS OFSEPTEMBER24, 2025),PRIVACY RELEVANCE
(SCALE OF1-3,WITH3MEANING DIRECTLY RELEVANT),PRIVACY FOCUS
(LEAKAGE☂OR ADVERSARIAL MANIPULATION),PRIMARY PURPOSE
OF THE PAPER(ATTACK†,MITIGATIONὮ1,OR NEITHER“-”),WHETHER
THE PAPER INTRODUCES MITIGATIONS(IMPLEMENTS/,MENTIONS/cent,
OR NONE/reve),AND WHETHER EXPERIMENTS ARE RUN(YES/OR NO/reve),
CODE IS AVAILABLE(IF YES,LINKED TO/cde),AND DISCUSSIONS ON
PRACTICAL APPLICABILITY(LATENCY,OVERHEAD,ETC.)ARE INCLUDED
(YES/OR NO/reve).
Key Year Cit. Rel. Focus Purpose Mit.? Exp.? Code? Disc.?
[75] 2025 29 2 † / / /cde /reve
[35] 2025 13 3☂+ † / / /reve
[49] 2025 5 3☂Ὦ1 / /reve /cde /reve
[77] 2025 4 2 † /reve / /reve
[45] 2025 4 1☂ - / /reve /
[78] 2025 3 2 Ὦ1 / / /reve
[51] 2025 2 3 Ὦ1 /cent /reve /
[46] 2025 1 3☂Ὦ1 / / /reve
[73] 2025 1 3 † /cent / /cde /reve
[48] 2025 1 3☂Ὦ1 / / /
[42] 2025 1 1 - /cent /reve /reve
[41] 2025 0 3☂+Ὦ1 /cent /reve /reve
[50] 2025 0 3☂+Ὦ1 /cent /reve /reve
[74] 2025 0 2 † / / /reve
[43] 2025 0 2☂Ὦ1 / / /
[44] 2025 0 2☂Ὦ1 / / /cde /reve
[83] 2025 0 2 † /cent / /reve
[47] 2025 0 1☂+Ὦ1 / /reve /reve
[87] 2024 305 1☂ - /cent /reve /cde /reve
[70] 2024 214 3 † / / /cde /
[12] 2024 132 3☂ † / / /cde /reve
[59] 2024 127 1☂+ † / / /cde /reve
[52] 2024 99 1☂Ὦ1 / / /
[57] 2024 87 3 † / / /reve
[63] 2024 76 2☂+Ὦ1 / / /
[28] 2024 74 3☂+ † /cent / /reve
[79] 2024 64 3 † /cent / /cde /reve
[58] 2024 64 1☂+ † / / /cde /reve
[9] 2024 62 3☂+ † /reve / /reve
[65] 2024 56 2 Ὦ1 / / /cde /reve
[20] 2024 50 3☂ † / / /cde /reve
[40] 2024 48 2☂ † / / /cde /reve
[31] 2024 46 3☂+ † /cent / /cde /reve
[67] 2024 46 2☂+ † /cent / /cde /reve
[53] 2024 46 1☂ - /cent /reve /cde /
[36] 2024 41 2☂ † / / /reve
[85] 2024 38 2 † / / /cde /
[37] 2024 37 2☂+ † /cent / /reve
[33] 2024 29 3☂Ὦ1 / / /cde /reve
[86] 2024 29 1☂Ὦ1 / /reve /reve
[71] 2024 26 1 † / / /cde /reve
[88] 2024 25 1☂ - /cent /reve /
[38] 2024 23 2☂+ † / / /reve
[39] 2024 17 2☂+ † / / /reve
[62] 2024 17 1 † / / /reve
[25] 2024 16 3☂+ † / / /reve
[82] 2024 16 3 † /reve / /
[34] 2024 16 1☂Ὦ1 / /reve /cde /reve
[69] 2024 15 3 † / / /cde /
[18] 2024 15 3☂+ † /cent / /reve
[60] 2024 15 2☂+Ὦ1 / / /cde /
[61] 2024 15 2 † /cent / /cde /reve
[29] 2024 15 2☂+ † /reve / /reve
[81] 2024 11 3☂ † / / /cde /reve
[64] 2024 11 2 † / / /
[80] 2024 10 2 † /reve / /reve
[68] 2024 9 2 † / / /reve
[56] 2024 8 3☂Ὦ1 / / /
[30] 2024 6 3☂+ † / / /reve
[21] 2024 6 3☂Ὦ1 / / /reve
[84] 2024 5 2 † /reve / /reve
[24] 2024 5 1☂Ὦ1 / /reve /cde /reve
[22] 2024 5 1☂Ὦ1 / /reve /reve
[26] 2024 4 2☂Ὦ1 / / /cde /reve
[89] 2024 2 2☂Ὦ1 /reve /reve /cde /reve
[72] 2024 2 2 † /reve / /reve
[19] 2024 2 2☂ † / / /reve
[66] 2024 2 1☂+Ὦ1 / / /
[54] 2024 1 1☂ - /cent /reve /reve
[32] 2023 456 2☂ † /cent / /reve
[27] 2023 56 3☂Ὦ1 / / /cde /reve
[55] 2023 14 1☂Ὦ1 / / /cde /

TABLE III
AMAPPING OF PROPOSED MITIGATION STRATEGIES FROM THE
LITERATURE TO THE PRIVACY RISKS ASSOCIATED WITH DIFFERENT
STAGES OF THERAGPIPELINE,AS WELL AS WITH OUR SURVEY
ADVERSARIAL MANIPULATION THREATS.Rel.ANDMat.DENOTE THE
RELEVANCE AND MATURITY SCORES,WHICH ARE NORMALIZED TO
BETWEEN0AND1TO SHOW RELATIVE RANKING. FOR EASE OF
INTERPRETATION,WE ALSO COLOR-CODE THE SCORES INTO FOUR BINS:
[0, 0.25):○, [0.25, 0.50):○, [0.50, 0.75):○,AND[0.75, 1.00]:○.
Privacy Risk Proposed Mitigation Solutions Rel. Mat.
Leakage
DatasetAnonymization [10], [18], [27], [32], [41]–[43], [47], [50], [53] ○0.77 ○0.23
Synthetic Data [33] ○0.12 ○0.67
Text Rewriting / Rephrasing [33], [37], [59] ○0.19 ○0.22
Text Summarization [10], [12], [33] ○0.19 ○0.22
Differential Privacy [21], [42], [43], [46], [49] ○0.50 ○0.27
Distributed Data Storage / Hybrid Cloud Architectures [5], [55] ○0.15 ○0.00
Encryption [43], [50], [51] ○0.19 ○0.22
Vector DatabaseAccess Control [18], [22], [43], [44], [47], [50], [51], [54] ○0.65 ○0.17
Redundant Benign Knowledge Base / Duplication [25], [27], [62] ○0.42 ○0.22
Synthetic Data [33] ○0.12 ○0.67
Retrieved ChunksRetrieval Strategy (e.g., # of chunks, re-ranking) [12], [18], [33], [52] ○0.38 ○0.33
Distance Metric Strategy (e.g., thresholds) [10], [12], [33] ○0.19 ○0.22
Differential Privacy (in re-ranking) [86] ○0.12 ○0.00
AnswerLocal LLM Deployment [34], [43]–[45], [53], [54], [87], [88] ○0.58 ○0.13
Safeguards / Guardrails [45], [51], [53], [58] ○0.27 ○0.42
Fact-checking / Source Citation [67] ○0.10 ○0.00
PromptAnonymization (Remove/Mask/Filter PII) [10], [18], [27], [41], [50] ○0.31 ○0.13
Text Rewriting / Paraphrasing / Regrouping [33], [37], [40], [59] ○0.35 ○0.33
Prompt Augmentation / Guardrails [25] ○0.04 ○0.00
Multi-party Computation (MPC) [55] ○0.12 ○0.00
Adversarial Manipulation (Attacks)
Anonymization [10] ○0.00 ○0.00
BackdoorDistance Metric Strategy [10] ○0.00 ○0.00
Text Summarization [10] ○0.00 ○0.00
Knowledge Expansion [79] ○0.00 ○0.00
Detection of Anomaly Clusters [79] ○0.00 ○0.00
Data ExtractionAnonymization [10], [27] ○0.15 ○0.33
Distance Metric Strategy [10] ○0.00 ○0.00
Prompt Augmentation / Guardrails [40] ○0.12 ○0.67
Text Summarization [10] ○0.00 ○0.00
Fine Tuning [81] ○0.12 ○0.33
Data PoisoningAccess Control [28], [50], [51], [64] ○0.23 ○0.00
Anonymization [10], [18], [41], [42], [50] ○0.19 ○0.00
Distance Metric Strategy [10], [30], [42] ○0.19 ○0.11
Fact-checking [67] ○0.00 ○0.00
Guardrails [28], [51], [58] ○0.19 ○0.22
Retrieval Strategy [18], [37], [63], [69], [78] ○0.42 ○0.20
Rewriting / Rephrasing [28], [37], [38], [59], [63], [68]–[70], [78] ○1.00 ○0.37
Text Summarization [10], [75] ○0.15 ○0.17
Text Duplication / Knowledge Expansion [74]–[76] ○0.42 ○0.33
Text Filtering [19], [28], [38], [70], [74], [76] ○0.77 ○0.39
Perplexity [28], [57], [59], [69], [70], [74] ○0.85 ○0.38
Adversarial Training [18], [20], [41], [60], [65], [66] ○0.62 ○0.33
Grammar Checker [20] ○0.04 ○0.00
Jailbreak Guardrails / Alignment [25], [36], [51], [90] ○0.38 ○0.33
LLM Extraction Differential Privacy [8] ○0.00 ○0.00
Anonymization [18], [41] ○0.08 ○0.00
Distance Metric Strategy [10], [18] ○0.08 ○0.00
Membership Prompt Augmentation / Guardrails [35], [36], [39] ○0.42 ○0.33
Inference Retrieval Strategy (e.g. re-ranking) [18], [35], [39] ○0.35 ○0.33
Prompt Rewriting / Rephrasing [35], [39] ○0.27 ○0.33
Text Summarization [10] ○0.00 ○0.00
Differential Privacy [83] ○0.00 ○0.00
Prompt ExtractionAnonymization [10], [18], [50] ○0.12 ○0.00
Prompt Augmentation / Guardrails [56] ○0.12 ○1.00
Prompt Rewriting / Rephrasing [56] ○0.12 ○1.00
Prompt InjectionGuardrails [25], [41], [51] ○0.12 ○0.00
Text Filtering [31] ○0.12 ○0.00
Grammar Checker [31] ○0.12 ○0.00
at that stage, and thus we did not infer or extrapolate possible
applications from those stated in the works.
•HIGH: dedicated section or implementation in the paper.
•MID: only discussed somewhere in the paper.
•LOW: simply mentioned as a potential mitigation.
Similar guidelines were also established for thematurity
score, which considered the reproducibility (R), cross-domain
generalizability (G), and deployability (D) of the conducted
experiments in a given work (one score assigned for each axis):
•HIGH: code available, tests on more than one dataset,
discussion of deployment costs/overhead.
•LOW: no code available, 0 or 1 datasets used, no mention
of deployment considerations.
Finally, we aggregated the above categorizations into a rele-
vance and maturity score for each of the mitigation categories,
according to Equations 1 and 2.
relevance= (2×#HIGH) + (1×#MID) + (0.5×#LOW)(1)TABLE IV
AN OVERVIEW OF DATASETS USED IN THE EVALUATION SETUPS OF
WORKS ADDRESSING PRIVACY INRAG.
Type/Task Dataset Used In
Natural Questions [91] [20], [25], [28], [33], [35], [38], [39], [57], [60], [65], [66],
[68]–[70], [72]–[74], [77]–[79], [83]
MS-MARCO [92] [28], [35], [38], [57], [61], [68]–[70], [73], [74], [77]–[79]
HotpotQA [93] [28], [38], [64], [68], [70], [73], [74], [77]–[80]
TriviaQA [94] [20], [33], [39], [60], [65]
General WebQuestions [95] [33], [60], [65], [79]
Question PopQA [96] [60]
Answering StrategyQA [97] [59]
SQuAD [98] [20], [57]
Cosmos [99] [21]
CuratedTrec [100] [33]
RealtimeQA(-MC) [101] [63]
Quora [38]
TextBook [102] [30]
StatPearls [30]
HealthCareMagic [46], [103] [12], [18], [33], [35], [36], [48], [83]
(Bio)medical NFCorpus [104] [80]
Datasets MMLU-Med [87] [30], [81]
MedQAUS [87] [30]
MedMCQA [87] [30], [81]
PubMedQA [87], [105] [30], [81]
BioASQ-Y/N [87], [106] [30], [84]
Pile [107] [108]
FiQA [109] [38]
General Enron Emails [110] [10], [12], [27], [32], [33], [36], [48]
NLP WikiText [111] [27], [33]
Datasets WNUT 2017 [112] [21]
SST-2 [113] [79]
AG News [114] [79]
MovieLens [115] [73], [75]
Bias BBQ [116] [79]
and AdvBench-V3 [90] [79]
Factuality LLM Biographies [117] [63]
maturity=1
3X #HIGH i
Total # for Mitigation, i∈ {R, G, D}(2)
The normalized scores for both relevance and maturity are
presented in Table III, and are also binned for readability. We
note that since the scores are normalized, a score of 0 does
not mean the absence of relevance or maturity, but rather the
lowest as compared to other mitigations, and vice versa.
Worked Example.From the five papers that proposed
Differential Privacy as a mitigation for dataset leakage, only
three implemented their solutions and thus were classified
as HIGH relevance papers. The other two simply mentioned
Differential Privacy as a possible mitigation and had therefore
LOW relevance. The final Differential Privacy (dataset leak-
age) relevance score was computed as(2×3 HIGH papers) +
(1×0 MID papers) + (0.5×2 LOW papers) = 7, resulting
in a normalized value of0.5. The Differential Privacy (dataset
leakage) maturity score was computed as the average of the
number of papers with high reproducibility (1 paper out of
the total 5, so 20%), high cross-domain generalizability (only
3 papers out of the total 5, so 60%), and high deployability
which had a 0% score, because none of the 5 papers took
the cost or the latency of their method into consideration. The
final maturity score resulted in(0.2+0.6+0)
3≈0.27.
C. Evaluating Privacy in RAG
Important to studying RAG privacy risks and mitigations
is the evaluation strategies undertaken in experimental setups.
Recent works employ a wide variety of datasets and metrics
to measure both the utility (RAG performance) and privacy
(protection against risks) afforded by mitigation techniques.
Widely used general question-answering datasets, such as
MS-MARCO, HotpotQA, and TriviaQA, are also employed in
most research addressing privacy in RAG systems. However,
datasets for domain-specific tasks, particularly in (bio)medical,

TABLE V
EVALUATION METRICS USED IN WORKS ADDRESSING PRIVACY INRAG.
Metric Name Description Used In
Retrieval Metrics
Accuracy Metric for correctness of generated answers based on reference (e.g., top-k hit rate). [20], [21], [31], [32], [35], [57], [59], [62], [63], [66], [67], [77]–[79], [83], [84]
Precision / Recall Metric for proportion and coverage of relevant contexts among the top-k retrieved ones. [35], [44], [48], [66], [70], [73], [74], [79], [83]
F1-Score Harmonic mean of precision and recall. [13], [35], [40], [60], [68], [70], [74], [79], [83]
Generation Metrics
ROUGE-N/-L Metrics based on overlap of n-grams between generated and reference texts. [12], [21], [27], [33], [40], [46], [48], [56], [57], [71], [81]
BLEU-1/-4 Precision-based metric that compares n-gram overlaps between generated and reference texts. [33], [40], [46], [48], [71]
BERTScore Similarity metric that measures the cosine similarity of BERT embeddings to compare generated and reference texts. [40]
LLM-as-a-Judge A Large Language Model is used to evaluate the correctness, relevance, or quality of a generated response. [57], [63], [81], [84]
Answer Metrics
Rejection Rate Proportion of times the generator (model) refuses to answer. [10], [57], [58]
Benign Answers Proportion or count of answers that are safe, correct, and contain no policy violations or harmful content. [25]
Malicious Answers Proportion or count of answers that contain harmful, malicious, or disallowed content. [25]
Ambiguous Answers Proportion or count of answers that are unclear, vague, or could be interpreted in multiple ways. [25]
Inconclusive Answers Proportion or count of answers that do not provide a definitive statement. [25]
Attack Metrics
Attack Success Rate Percentage of attempts causing the system to reveal private content, or otherwise deviate from normal policy. [9], [20], [29], [38], [57]–[59], [62], [68], [70], [74], [77], [78], [80], [81], [84]
Retrieval Success Rate Percentage of queries for which the system successfully retrieves the target documents, whether poisoned or not. [30], [35], [38], [69], [76]
Retrieval Failure Rate Percentage of queries for which the system fails to retrieve the target documents, whether poisoned or not. [28]
Extraction Rate Percentage of successful attempts of extracting the targeted data. [18]
Targeted information Count of targeted information, such as poisoned documents or PII, that appear in the generated response. [27], [32], [33], [37], [76]
Other Metrics
Exact Match (Rate) Evaluates if a prediction precisely matches the correct answer. [13], [20], [60], [71], [79], [108]
Keyword Matching Rate Recall rate between the reference and response based on ROUGE-L. [79]
Mean Reciprocal Rank Average reciprocal rank of the first relevant item in a ranked list of results. [26], [61]
AUC ROC Metric for evaluating the trade-off between true and false positive rates across thresholds. [35], [36], [39], [70], [83]
financial, and bias-related contexts, are also prevalent. For ex-
ample, medical datasets such as TextBook, StatPearls, MedM-
CQA, MMLU-Med, and BioASQ are used to study privacy in
settings involving sensitive health-related data. These datasets
span a variety of domains and use cases, highlighting the broad
applicability and relevance of RAG privacy research.
Research conducting experiments on privacy in RAG sys-
tems utilizes a diverse set of evaluation metrics to empirically
measure privacy protection. These can be categorized into five
primary groups, providing a holistic view of how researchers
analyze RAG performance under privacy-related conditions:
•Retrieval metrics: assess the effectiveness of the retrieval
component in isolating relevant information, especially
in contexts where sensitive or adversarially injected data
may be present. These metrics are critical in determining
whether the system successfully retrieves harmful or
private data, which is often the first step in privacy-
adverse behaviors. Emphasis here is placed not only on
the presence of correct documents but also on the bal-
ance between over-retrieval (which may include sensitive
content) and under-retrieval (which could limit utility).
•Generation metrics: the focus shifts to the quality of the
outputs generated based on the retrieved contexts. These
methods are widely adapted from traditional natural lan-
guage generation evaluation techniques but take on new
relevance in privacy research, measuring to what extent
privacy risk mitigations affect generation quality.
•Answer metrics: these metrics evaluate the content pro-
duced by the RAG system. This includes whether answers
are benign, malicious, or ambiguous, or whether the
model opts to refrain from answering altogether. These
metrics are particularly useful for identifying indirect
privacy risks, such as vague or misleading responses that
may reflect underlying data exposure or misalignment
with system policy. Thus, answer metrics view evaluation
from a broader, ethical- and safety-focused lens.
•Attack metrics: measure the success of adversarial at-
tempts to tamper with a RAG system. They can reveal thesusceptibility of systems to prompt injection, poisoning,
or targeted extraction of private data. They often differ-
entiate between retrieval and generation failures, which
is critical in tracing the propagation of an attack.
•Other metrics: these encompass auxiliary evaluation
techniques, often borrowed from the machine learning
and information retrieval disciplines. This might include
precise matching and ranking-based metrics that quan-
tify system accuracy and decision confidence, providing
additional context to more targeted privacy evaluations.
A brief description of the diverse datasets and metrics for
privacy evaluation in RAG are provided in Tables IV and V.
D. A Dynamic View of Privacy in RAG
Though the SLR sheds light on a number of privacy risks
and proposed mitigations, the coverage of these findings
hitherto represents a “static” view of the RAG privacy risk
ecosystem. Specifically, the mapping presented in Table III is
useful for directly associating proposed mitigation strategies
and leakage types, yet there is no notion of how these
mitigations might impact the remainder of the RAG pipeline;
moreover, mitigating leakage should be better connected with
the mitigation of associated adversarial attack vectors.
As such, we create a more dynamic view of the RAG privacy
ecosystem, which we present in our second main artifact, the
RAG Privacy Process Diagram (Figure 3). This diagram is
divided intoRAG setupandRAG inference(i.e., runtime). In
this ecosystem, there exists a separation of concerns between
departments within an “organization”, or the entity responsible
for the RAG system provision. Moreover, we distinguish
betweenactivitiescarried out within this organization, and
those taking place outside it. These activities are presented
along with an indication of theactorresponsible for the task.
To bind the privacy risks (Figure 2) and mitigations (Table
III) to the complete RAG process, we indicate at which point
risksoriginate, and accordingly,wheremitigations serve to
protect against such risks. With this perspective, one can
visualize what further steps may be impacted by the realization

Fig. 3. The RAG Privacy Process Diagram.
of a privacy risk, or the introduction of a mitigation. This,
above all, serves to contextualize decisions made in privacy
risk mitigation plans, showing that such decisions cannot be
made in isolation from a particular threat actor or RAG process
activity. We note that the diagram models a “naive” (simple)
RAG pipeline; advanced RAG setups were out-scoped.
V. DISCUSSION
In this section, we reflect on the findings of our survey on
privacy in RAG, critically assess the current state of privacy
mitigations, and discuss implications for future research.
What is new with privacy in RAG?Our literature review
shows a clear increase in research attention paid to risks
associated with RAG systems, with just three relevant papers
in 2023 to 51 papers in 2024. This not only points to the
growing importance of the topic, but also to the diversity and
complexities within the research field of privacy in RAG.
A survey of the privacy risks in RAG systems reveals
three important factors that exacerbate the threat of known
adversarial attacks, as well as create complexities in the
elicitation of novel RAG privacy risks. Firstly, the usefulness
of RAG in bringing life to typically “static” LLMs is certainly
proven, yet it is a double-edged sword in the way that “live”
data presents an especially vulnerable point that LLMs alone
do not exhibit, i.e., exposed private data. As a second and
related point, the ultimate goal of RAG systems is to make
such data more accessible by allowing users, whether internal
or external to organizations, to interface with the data. This
naturally creates new risks of improper data disclosure withoutproper measures. Such measures are two-fold in the sense that
external safeguards must be in place, such as access control,
alongside system internal mitigations, such as anonymization.
This gives way to the third aspect of RAG systems, leading
to novelties in the study of privacy risks, which relates directly
to the “system” nature of RAG itself. As opposed to studying
privacy risks in LLMs, for example, RAG systems exhibit
many points of potential failure or compromise, and these
points take various forms, e.g., raw text data, vectorized doc-
ument chunks, or LLM-generated answers. This typical RAG
pipeline is incredibly dynamic, involving multiple stakeholders
and individual technologies, presenting not only a wide attack
surface, but also a more complex ecosystem for mitigations to
be implemented. Thus, whereas many of the potential privacy
threats to RAG systems may be similar to known risks from a
technical point of view, the landscape of RAG systems opens
the door to new avenues for attack realization and mitigation.
It’s all about data leakage, but leakage can mean
many things!Upon our initial reading of the selected primary
sources, we learned that many of the perceived privacy risks
relating to RAG systems revolve around the form ofleakage.
As displayed in our Taxonomy of RAG Privacy Risks (Figure
2), this is only one side of the story, and privacy risks are
often studied from the angle ofadversarial manipulation,
or designed attacks. This is exemplified by Table II, which
shows a near parity between papers focusing on the leakage
or adversarial aspects. Together,leakageandattacksform the
two-sided coin of RAG privacy risks, with the former focusing
onwhatis exploited, while the latter focuses on thehow.

The story runs deeper, however, and we learn that data
leakage cannot be viewed as a single phenomenon, but rather a
spectrum of possible privacy vulnerabilities. This largely roots
itself in the fact that data passes through many “checkpoints”
in the RAG pipeline, and at each of these, it may be re-
encoded, chunked, or harmonized with the help of LLMs.
These distinct stages make the investigation of RAG privacy
risks dynamic, and a useful way to reason about such risks is
to intertwine the RAG stages with points where leakage may
originate. Thus, we distinguish leakage that can be pinpointed
to thesystem(data leakage) or theuser(prompt leakage).
As a final piece to the story, we find that viewing privacy
risks and their mitigations in isolation (Table III) is useful but
not completely satisfactory. To contextualize such a mapping
further, we create a more dynamic view of the privacy risks
in RAG, in the form of the RAG Privacy Process Diagram
(Figure 3). In this, we propose two primary improvements to
the study of RAG privacy risks: (1) the perspective of risks
and mitigations as part of a larger ecosystem, where either
the fruition of a risk or the implementation of a mitigation
carries effects downstream, and (2) a more illustrative picture
of where mitigations can be implemented in the pipeline, and
for which risks (attacks) there may not currently be sufficient
protections. A prime example of the latter point surfaces at
the user interface of RAG systems, which represents the stage
at which many RAG privacy risks are realized, yet where
more proposals for novel mitigations can be made. Thus, we
hope that the process diagram becomes a living artifact, where
future updates may serve to track the progress made, as well
as the new risks arising, regarding privacy in RAG.
Mitigating leakage: viewing the RAG privacy process
in action.As previously introduced and in light of Figure
3, it becomes interesting to explore the impact of mitigation
techniques on the overall functioning of the RAG system.
The pipeline nature of RAG, in which data exists in many
forms (raw text, embedding vectors, text chunks), introduces
complexities for the implementation of mitigations, in that the
efficacy and trade-off resulting from the mitigation are directly
affected by the form of inputs as well as the nature of the
following states. As an example, performing anonymization
directly on the raw text data might be very effective for
removing sensitive information, but may degrade the quality
of the resulting embeddings, text chunk retrieval, and an-
swer generation. In contrast, implementing mitigations further
downstream may serve to preserve utility more effectively, but
at the cost of late or even post-hoc privatization.
What is uncertain from current research, however, is the
effectiveness of mitigationsin sequence. While it may seem
wise to implement protections at various points along the RAG
process, we found no evidence of such experimentation. Be-
yond feasibility, we envision that determining effective ensem-
bles of mitigations would require meticulous research. This
comes in addition to fundamental research on the strengths
and limitations of mitigations at different stages, e.g., database
anonymization versus generated answer rewriting.
As such, we hope that with the guidance of the RAG PrivacyProcess Diagram, future studies can focus on investigatingmit-
igations in context, giving credence to the merits of proposed
mitigations, while also uncovering their potential limitations.
The state of current mitigations.In order to investigate
deeper the main focus of the works we survey, we also
systematize the balance in the current literature between works
focusing on adversarial attacks affecting privacy, and those
that propose mitigations to combat these (Table II). We find
a relatively significant skew towards “attack papers” (56%),
whereas papers focusing specifically on privacy mitigations
comprise only 36% of the reviewed papers (with 8% in neither
category). While it is also the case that attack papers often test
mitigations, they rarely are the primary focus, suggesting the
need for an uptick in privacy mitigation research for RAG.
We also quantify therelevanceandmaturityof proposed
mitigations (Table III). These scores, while imperfect approx-
imations of the current state of privacy mitigations in RAG,
provide an overall sense of the relative attention a certain
mitigation strategy has received (relevance), as well as how
often they are practically tested (maturity). A mitigation with
a high relevance but low maturity may imply that many works
have proposed the mitigation, but fewer have tested it. On
the other hand, higher maturity than relevance would suggest
that, although the raw quantity of mentions may be less, such
a method may be more mature since it has been relatively
more often evaluated. With this, one can see that mitigations
for certain risks, such as dataset leakage and data poisoning,
are generally more mature than those of LLM extraction,
backdoor, or prompt injection attacks, for example. This
assessment, therefore, sheds light on the potentially under-
researched mitigation areas for RAG privacy risks.
As can be extrapolated from Tables II and III, the state of
current privacy risk mitigations in RAG leans on the immature
end, made evident by the generally lowrelevanceandmaturity
scores, as well as the relatively low amount of works that pub-
lish code to reproduce either attacks or mitigations (roughly
40%). This would suggest that reasoning about privacy risks
and mitigations in RAG, even in the research sphere, still exists
in the “ideation” phase, with growing numbers of actionable
artifacts being published. This is also made clear by the
low amount of papers discussing practical considerations of
mitigation adoption (24%), such as computational speed or
resources required to run such proposed mitigations.
With our systematization and mitigation assessments, we
strive for these scores to form a foundation for quantifying
the current state of mitigation strategies for RAG privacy,
and moreover, to provide a practical sense offeasibilityfor
researchers and practitioners going forward. In this light, we
synthesize our findings into a set of key focus areas for future
research on the mitigation of privacy risks in RAG, focusing
on the recurring challenges we observed from the literature:
•More practical considerations: as previously noted, we
found that only about a quarter of papers implementing
mitigations also provided details on computational costs
and other practical deployment considerations. We see

this is an important area to improve to increase the
practical applicability of privacy risk mitigations in RAG.
•Real-world testing: we found very few cases of exposing
privacy risks or testing mitigations on live RAG systems,
implying a potential disconnect between current research
and real-world privacy concerns in RAG. Furthermore,
our review of grey literature suggested that practical
privacy solutions currently do not match the intricacy of
those proposed in the literature, which likewise strength-
ens the need for practicality in RAG privacy research.
•All RAG is not the same: we found little evidence of
considering various RAG setups, leaving the question
unanswered how privacy is affected (for better or for
worse) by more complex RAG architectures, including
those implementing agentic systems.
•Towards RAG-specific privacy evaluation: particularly
in the use of datasets for RAG privacy evaluation, we
perceive that many of the leveraged datasets (Table IV)
are “reused” from other purposes, and there is a lack of
privacy-specific benchmark datasets for RAG use cases.
To an extent, this also applies to evaluation metrics (Table
V), which often relate to either pure RAG evaluation or
more security-specific measurements (e.g., rejection rate).
VI. RELATEDWORK
Surveying RAG.Previous works survey various aspects
of RAG systems, such as general applications [7], [118],
architectures and optimization strategies [15], and evaluation
strategies [11], [119]. Other works investigate more specific
aspects of RAG, such as trustworthiness [6], [10] or RAG
with multimodal data [14]. These works, however, make no
or very tangential connections to the topic of privacy in RAG.
In exploring privacy in RAG, Zeng et al. [12] consolidate
a number of adversarial attack types and propose defense
strategies to mitigate them. While this survey offers one of
the most detailed treatments of privacy in RAG systems in
the current literature, it is limited in its scope of literature
coverage, and it does not follow a formal methodology. In
contrast, our work is guided by an Systematic Literature
Review, analyzing how specific architectural and procedural
elements in RAG pipelines lead to privacy risks, and we
systematize a larger body of works on privacy aspects in RAG.
Privacy in LLMs.Beyond RAG systems, multiple other
works consider the privacy implications of LLMs. In their
survey, Wang et al. [8] also relate privacy risks in LLMs to
those in RAG, organizing security and privacy threats of LLMs
across five stages of their lifecycle: pre-training, fine-tuning,
deployment, LLM agents, and RAG. Their analysis of privacy
in RAG remains high-level, with RAG only discussed as part
of a broader LLM lifecycle. However, the view of privacy risks
as part of aprocessmotivated the broader contextualization of
RAG privacy risks in our work. In addition to other surveys
exploring privacy in LLMs [120], [121], recent works view
privacy risks from the standpoint of (generative) AI [122],
including in the context of the AI lifecycle [123], practical
perspectives [124], [125], and general user perceptions [126].VII. CONCLUSION
We systematize the extant literature investigating privacy
risks and proposed mitigations in RAG. We find that privacy
risks in RAG systems can be categorized into two primary
categories, leakage and adversarial manipulation, which can
be mapped to a variety of innovative mitigation techniques. In
the evaluation of privacy in RAG systems, a wide variety of
datasets and metrics have been utilized, pointing to a wealth
of potential evaluation strategies, but also to a general lack
of unification. To augment our Taxonomy of RAG Privacy
Risks, we contextualize risks and mitigations in a RAG Privacy
Process Diagram, which acknowledges the dynamic nature
of RAG and the confluence of risks, actors, and potential
mitigations within this pipeline. Together, our findings not only
illuminate the ecosystem of privacy risks in the context of
RAG, but also map the current progress of mitigation efforts,
providing a foundation for future studies at the intersection of
privacy, RAG, and responsible AI.
Limitations.We acknowledge a number of threats to valid-
ity, particularly concerning the conduction of the SLR. Firstly,
the literature exclusion and filtering process was carried out
solely by the primary researcher, introducing the possibility for
researcher bias and subjectivity. Likewise, the full literature
reading was carried out by this researcher. To mitigate this
bias, weekly meetings were held with the larger research team
over the course of the study, in order to validate the data
extracted, as well as to make decisions during the system-
atization process (e.g., how to group mitigations). We also
performed double coding of the structured literature analysis
(data extraction) by two additional researchers on the team. In
the literature review, we also considered more established non-
published / non-peer-reviewed preprints, which could poten-
tially have affected the validity of the final presented artifacts.
We also caution that our survey and systematization efforts,
including the resulting artifacts, were studied in the scope
of “simpler” (naive) RAG pipelines, as depicted in Figure
1. As such, we did not generally account for the intricacies,
and potential exacerbating factors to privacy risks, of more
complex or advanced RAG architectures. We leave this as
future work to build on our study results.
Outlook and Future Work.Our work systematizes a
continually expanding field, and we envision that our findings
may serve to ground future research, particularly from a clearer
definition ofwhatis being mitigated (Figure 2 and Table III)
andhowthis fits within the larger RAG pipeline (Figure 3).
From a practical perspective, we bring greater awareness to the
potential risks of hosting RAG systems, as well as a contex-
tualization of state-of-the-art mitigation techniques and their
current research attention and maturity. We see three important
points for future research: (1) validation of our proposed
artifacts, specifically the RAG Privacy Process Diagram, (2)
extensive experiments and feasibility studies involving privacy
mitigations at various points in the RAG pipeline, and (3)
studying user perceptions of RAG privacy risks, as well as
tolerance for trade-offs introduced by mitigation techniques.

REFERENCES
[1] Y . Liu, T. Han, S. Ma, J. Zhang, Y . Yang, J. Tian, H. He, A. Li, M. He,
Z. Liuet al., “Summary of ChatGPT-related research and perspective
towards the future of large language models,”Meta-radiology, vol. 1,
no. 2, p. 100017, 2023.
[2] A. J. Thirunavukarasu, D. S. J. Ting, K. Elangovan, L. Gutierrez, T. F.
Tan, and D. S. W. Ting, “Large language models in medicine,”Nature
medicine, vol. 29, no. 8, pp. 1930–1940, 2023.
[3] E. Kasneci, K. Sessler, S. K ¨uchemann, M. Bannert, D. Dementieva,
F. Fischer, U. Gasser, G. Groh, S. G ¨unnemann, E. H ¨ullermeier,
S. Krusche, G. Kutyniok, T. Michaeli, C. Nerdel, J. Pfeffer,
O. Poquet, M. Sailer, A. Schmidt, T. Seidel, M. Stadler, J. Weller,
J. Kuhn, and G. Kasneci, “ChatGPT for good? on opportunities and
challenges of large language models for education,”Learning and
Individual Differences, vol. 103, p. 102274, 2023. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S1041608023000195
[4] M. Li, Y . Zhao, W. Zhang, S. Li, W. Xie, S.-K. Ng, T.-S. Chua,
and Y . Deng, “Knowledge boundary of large language models: A
survey,” inProceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), W. Che,
J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria:
Association for Computational Linguistics, Jul. 2025, pp. 5131–5157.
[Online]. Available: https://aclanthology.org/2025.acl-long.256/
[5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive nlp
tasks,” inProceedings of the 34th International Conference on Neural
Information Processing Systems, ser. NIPS ’20. Red Hook, NY , USA:
Curran Associates Inc., 2020.
[6] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S.
Chua, and Q. Li, “A Survey on RAG Meeting LLMs: Towards
Retrieval-Augmented Large Language Models,” inProceedings of
the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining, ser. KDD ’24. New York, NY , USA: Association
for Computing Machinery, Aug. 2024, pp. 6491–6501. [Online].
Available: https://dl.acm.org/doi/10.1145/3637528.3671470
[7] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y . Geng, F. Fu, L. Yang, W. Zhang,
J. Jiang, and B. Cui, “Retrieval-augmented generation for ai-generated
content: A survey,”Data Science and Engineering, pp. 1–29, 2026.
[8] S. Wang, T. Zhu, B. Liu, M. Ding, D. Ye, W. Zhou, and P. Yu,
“Unique security and privacy threats of large language models: A
comprehensive survey,”ACM Comput. Surv., vol. 58, no. 4, Oct. 2025.
[Online]. Available: https://doi.org/10.1145/3764113
[9] G. Deng, Y . Liu, K. Wang, Y . Li, T. Zhang, and Y . Liu, “PANDORA:
Jailbreak gpts by retrieval augmented generation poisoning,”Workshop
on AI Systems with Confidential Computing (AISCC) 2024, 2024.
[10] Y . Zhou, Y . Liu, X. Li, J. Jin, H. Qian, Z. Liu, C. Li, Z. Dou, T.-Y .
Ho, and P. S. Yu, “Trustworthiness in retrieval-augmented generation
systems: A survey,” Sep. 2024, arXiv:2409.10102. [Online]. Available:
http://arxiv.org/abs/2409.10102
[11] X. Li, J. Jin, Y . Zhou, Y . Zhang, P. Zhang, Y . Zhu, and Z. Dou,
“From matching to generation: A survey on generative information
retrieval,”ACM Trans. Inf. Syst., Mar. 2025, just Accepted. [Online].
Available: https://doi.org/10.1145/3722552
[12] S. Zeng, J. Zhang, P. He, Y . Liu, Y . Xing, H. Xu, J. Ren,
Y . Chang, S. Wang, D. Yin, and J. Tang, “The good and the
bad: Exploring privacy issues in retrieval-augmented generation
(RAG),” inFindings of the Association for Computational Linguistics:
ACL 2024. Bangkok, Thailand: Association for Computational
Linguistics, Aug. 2024, pp. 4505–4524. [Online]. Available: https:
//aclanthology.org/2024.findings-acl.267/
[13] J. Li, Y . Yuan, and Z. Zhang, “Enhancing LLM factual accuracy with
RAG to counter hallucinations: A case study on domain-specific queries
in private knowledge-bases,”arXiv preprint arXiv:2403.10446, 2024.
[14] R. Zhao, H. Chen, W. Wang, F. Jiao, X. L. Do, C. Qin, B. Ding,
X. Guo, M. Li, X. Li, and S. Joty, “Retrieving multimodal information
for augmented generation: A survey,” inFindings of the Association
for Computational Linguistics: EMNLP 2023. Singapore: Association
for Computational Linguistics, Dec. 2023, pp. 4736–4756. [Online].
Available: https://aclanthology.org/2023.findings-emnlp.314/
[15] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai,
J. Sun, M. Wang, and H. Wang, “Retrieval-augmented generationfor large language models: A survey,” 2024. [Online]. Available:
https://arxiv.org/abs/2312.10997
[16] B. A. Kitchenham, D. Budgen, and P. Brereton,Evidence-Based
Software Engineering and Systematic Reviews. Chapman & Hall/CRC,
2015.
[17] V . Garousi, M. Felderer, and M. V . M ¨antyl ¨a, “Guidelines for including
grey literature and conducting multivocal literature reviews in software
engineering,”Information and software technology, vol. 106, pp. 101–
121, 2019.
[18] S. Cohen, R. Bitton, and B. Nassi, “Unleashing worms and extracting
data: Escalating the outcome of attacks against RAG-based inference
in scale and severity using jailbreaking,” Sep. 2024, arXiv:2409.08045.
[Online]. Available: http://arxiv.org/abs/2409.08045
[19] X. Xian, T. Wang, L. You, and Y . Qi, “Understanding data poisoning
attacks for RAG: Insights and algorithms,” Oct. 2024. [Online].
Available: https://openreview.net/forum?id=2aL6gcFX7q
[20] S. Cho, S. Jeong, J. Seo, T. Hwang, and J. C. Park, “Typos
that broke the RAG‘s back: Genetic attack on RAG pipeline
by simulating documents in the wild via low-level perturbations,”
inFindings of the Association for Computational Linguistics:
EMNLP 2024. Miami, Florida, USA: Association for Computational
Linguistics, Nov. 2024, pp. 2826–2844. [Online]. Available: https:
//aclanthology.org/2024.findings-emnlp.161/
[21] J. Yu, J. Zhou, Y . Ding, L. Zhang, Y . Guo, and H. Sato,
“Textual Differential Privacy for Context-Aware Reasoning with
Large Language Model,” in2024 IEEE 48th Annual Computers,
Software, and Applications Conference (COMPSAC), Jul. 2024, pp.
988–997, iSSN: 2836-3795. [Online]. Available: https://ieeexplore.
ieee.org/document/10633584
[22] V . Vizgirda, R. Zhao, and N. Goel, “SocialGenPod: Privacy-
Friendly Generative AI Social Web Applications with Decentralised
Personal Data Stores,” inCompanion Proceedings of the ACM Web
Conference 2024, ser. WWW ’24. New York, NY , USA: Association
for Computing Machinery, May 2024, pp. 1067–1070. [Online].
Available: https://dl.acm.org/doi/10.1145/3589335.3651251
[23] Y . Ng, D. Miyashita, Y . Hoshi, Y . Morioka, O. Torii, T. Kodama, and
J. Deguchi, “SimplyRetrieve: A Private and Lightweight Retrieval-
Centric Generative AI Tool,” Aug. 2023, arXiv:2308.03983. [Online].
Available: http://arxiv.org/abs/2308.03983
[24] C.-C. Chuang and K.-C. Chen, “Retrieval Augmented Generation on
Hybrid Cloud: A New Architecture for Knowledge Base Systems,”
in2024 16th IIAI International Congress on Advanced Applied
Informatics (IIAI-AAI), Jul. 2024, pp. 68–71, iSSN: 2472-0070.
[Online]. Available: https://ieeexplore.ieee.org/document/10707974
[25] G. D. Stefano, L. Sch ¨onherr, and G. Pellegrino, “Rag and roll: An
end-to-end evaluation of indirect prompt manipulations in LLM-based
application frameworks,” Aug. 2024, arXiv:2408.05025. [Online].
Available: http://arxiv.org/abs/2408.05025
[26] Q. Hu, H. Li, J. Bai, Z. Wang, and Y . Song, “Privacy-preserved neural
graph databases,” inProceedings of the 30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining, ser. KDD ’24. New York,
NY , USA: Association for Computing Machinery, 2024, p. 1108–1118.
[Online]. Available: https://doi.org/10.1145/3637528.3671678
[27] Y . Huang, S. Gupta, Z. Zhong, K. Li, and D. Chen, “Privacy
implications of retrieval-based language models,” inProceedings of
the 2023 Conference on Empirical Methods in Natural Language
Processing. Singapore: Association for Computational Linguistics,
Dec. 2023, pp. 14 887–14 902. [Online]. Available: https://aclanthology.
org/2023.emnlp-main.921/
[28] H. Chaudhari, G. Severi, J. Abascal, M. Jagielski, C. A. Choquette-
Choo, M. Nasr, C. Nita-Rotaru, and A. Oprea, “Phantom: General
Trigger Attacks on Retrieval Augmented Language Generation,” Oct.
2024, arXiv:2405.20485. [Online]. Available: http://arxiv.org/abs/2405.
20485
[29] Z. Wang, J. Liu, S. Zhang, and Y . Yang, “Poisoned LangChain:
Jailbreak LLMs by LangChain,” Jun. 2024, arXiv:2406.18122.
[Online]. Available: http://arxiv.org/abs/2406.18122
[30] X. Xian, G. Wang, X. Bi, J. Srinivasa, A. Kundu, C. Fleming,
M. Hong, and J. Ding, “On the Vulnerability of Applying Retrieval-
Augmented Generation within Knowledge-Intensive Application
Domains,” Sep. 2024, arXiv:2409.17275. [Online]. Available: http:
//arxiv.org/abs/2409.17275
[31] D. Pasquini, M. Strohmeier, and C. Troncoso, “Neural exec: Learning
(and learning from) execution triggers for prompt injection attacks,”

inProceedings of the 2024 Workshop on Artificial Intelligence
and Security, ser. AISec ’24. New York, NY , USA: Association
for Computing Machinery, 2024, p. 89–100. [Online]. Available:
https://doi.org/10.1145/3689932.3694764
[32] H. Li, D. Guo, W. Fan, M. Xu, J. Huang, F. Meng, and Y . Song, “Multi-
step jailbreaking privacy attacks on ChatGPT,” inFindings of the
Association for Computational Linguistics: EMNLP 2023. Singapore:
Association for Computational Linguistics, Dec. 2023, pp. 4138–4153.
[Online]. Available: https://aclanthology.org/2023.findings-emnlp.272/
[33] S. Zeng, J. Zhang, P. He, J. Ren, T. Zheng, H. Lu, H. Xu,
H. Liu, Y . Xing, and J. Tang, “Mitigating the privacy issues
in retrieval-augmented generation (RAG) via pure synthetic data,”
inProceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing, C. Christodoulopoulos, T. Chakraborty,
C. Rose, and V . Peng, Eds. Suzhou, China: Association for
Computational Linguistics, Nov. 2025, pp. 24 527–24 558. [Online].
Available: https://aclanthology.org/2025.emnlp-main.1247/
[34] Z. J. Wang and D. H. Chau, “MeMemo: On-device retrieval
augmentation for private and personalized text generation,” in
Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval, ser. SIGIR ’24.
New York, NY , USA: Association for Computing Machinery, 2024,
p. 2765–2770. [Online]. Available: https://doi.org/10.1145/3626772.
3657662
[35] M. Liu, S. Zhang, and C. Long, “Mask-based membership inference
attacks for retrieval-augmented generation,” inProceedings of the
ACM on Web Conference 2025, ser. WWW ’25. New York, NY ,
USA: Association for Computing Machinery, 2025, p. 2894–2907.
[Online]. Available: https://doi.org/10.1145/3696410.3714771
[36] M. Anderson, G. Amit, and A. Goldsteen, “Is my data in your retrieval
database? Membership inference attacks against retrieval augmented
generation,” inProceedings of the 11th International Conference on
Information Systems Security and Privacy - Volume 2: ICISSP, IN-
STICC. SciTePress, 2025, pp. 474–485.
[37] Q. Zhang, B. Zeng, C. Zhou, G. Go, H. Shi, and Y . Jiang,
“Human-Imperceptible Retrieval Poisoning Attacks in LLM-Powered
Applications,” inCompanion Proceedings of the 32nd ACM
International Conference on the Foundations of Software Engineering,
ser. FSE 2024. New York, NY , USA: Association for Computing
Machinery, Jul. 2024, pp. 502–506. [Online]. Available: https:
//dl.acm.org/doi/10.1145/3663529.3663786
[38] Z. Tan, C. Zhao, R. Moraffah, Y . Li, S. Wang, J. Li, T. Chen,
and H. Liu, “Glue pizza and eat rocks - exploiting vulnerabilities
in retrieval-augmented generative models,” inProceedings of the
2024 Conference on Empirical Methods in Natural Language
Processing. Miami, Florida, USA: Association for Computational
Linguistics, Nov. 2024, pp. 1610–1626. [Online]. Available: https:
//aclanthology.org/2024.emnlp-main.96/
[39] Y . Li, G. Liu, C. Wang, and Y . Yang, “Generating is believing:
Membership inference attacks against retrieval-augmented generation,”
inICASSP 2025 - 2025 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), 2025, pp. 1–5.
[40] Z. Qi, H. Zhang, E. P. Xing, S. M. Kakade, and H. Lakkaraju,
“Follow my instruction and spill the beans: Scalable data extraction
from retrieval-augmented generation systems,” inThe Thirteenth
International Conference on Learning Representations, 2025. [Online].
Available: https://openreview.net/forum?id=Y4aWwRh25b
[41] C. M. Ward and J. Harguess, “Adversarial threat vectors and risk
mitigation for retrieval-augmented generation systems,” inAssurance
and Security for AI-enabled Systems 2025, J. D. Harguess, N. D.
Bastian, and T. L. Pace, Eds., vol. 13476, International Society for
Optics and Photonics. SPIE, 2025, p. 134760A. [Online]. Available:
https://doi.org/10.1117/12.3055931
[42] A. Kulshreshtha, A. Choudhary, T. Taneja, and S. Verma, “Enhancing
healthcare accessibility: A RAG-based medical chatbot using
transformer models,” in2024 International Conference on IT
Innovation and Knowledge Discovery (ITIKD), Apr. 2025, p. 1–4.
[Online]. Available: https://ieeexplore.ieee.org/document/11005179
[43] X. Fang, L. Qiao, J. Shi, and H. An, “Guardian Angel: A secure
and efficient retrieval-augmented generation framework,” in2025
5th International Conference on Artificial Intelligence and Industrial
Technology Applications (AIITA), Mar. 2025, p. 1773–1777. [Online].
Available: https://ieeexplore.ieee.org/document/11047845[44] B. Chen, J. Tackman, M. Set ¨al¨a, T. Poranen, and Z. Zhang,
“Integrating access control with retrieval-augmented generation:
A proof of concept for managing sensitive patient profiles,” in
Proceedings of the 40th ACM/SIGAPP Symposium on Applied
Computing, ser. SAC ’25. New York, NY , USA: Association for
Computing Machinery, May 2025, p. 915–919. [Online]. Available:
https://dl.acm.org/doi/10.1145/3672608.3707848
[45] Z. Yu, S. Liu, P. Denny, A. Bergen, and M. Liut, “Integrating
small language models with retrieval-augmented generation in
computing education: Key takeaways, setup, and practical insights,”
inProceedings of the 56th ACM Technical Symposium on Computer
Science Education V . 1, ser. SIGCSETS 2025. New York, NY , USA:
Association for Computing Machinery, Feb. 2025, p. 1302–1308.
[Online]. Available: https://dl.acm.org/doi/10.1145/3641554.3701844
[46] L. He, P. Tang, Y . Zhang, P. Zhou, and S. Su, “Mitigating privacy
risks in retrieval-augmented generation via locally private entity per-
turbation,”Information Processing & Management, vol. 62, no. 4, p.
104150, Jul. 2025.
[47] W. Hussain, “Mitigating values debt in generative AI: Responsible
engineering with graph RAG,” in2025 IEEE/ACM International
Workshop on Responsible AI Engineering (RAIE), Apr. 2025, p. 9–12.
[Online]. Available: https://ieeexplore.ieee.org/document/11029428
[48] J. He, C. Liu, G. Hou, W. Jiang, and J. Li, “PRESS: Defending privacy
in retrieval-augmented generation via embedding space shifting,” in
ICASSP 2025 - 2025 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), Apr. 2025, p. 1–5. [Online].
Available: https://ieeexplore.ieee.org/document/10887843
[49] N. Grislain, “RAG with differential privacy,” in2025 IEEE Conference
on Artificial Intelligence (CAI), May 2025, p. 847–852. [Online].
Available: https://ieeexplore.ieee.org/document/11050672
[50] A. Mehta and A. Patel, “Secure framework for retrieval-augmented
generation: Challenges and solutions,”IJARCCE, no. 01, Jan. 2025.
[Online]. Available: https://ijarcce.com/wp-content/uploads/2025/01/
IJARCCE.2025.14114.pdf
[51] S. Nandagopal, “Securing retrieval-augmented generation pipelines: A
comprehensive framework,”Journal of Computer Science and Tech-
nology Studies, vol. 7, no. 11, p. 17–29, Jan. 2025.
[52] A. Golatkar, A. Achille, L. Zancato, Y .-X. Wang, A. Swaminathan,
and S. Soatto, “CPR: Retrieval augmented generation for copyright
protection,” inProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 12 374–12 384.
[53] S. Xue, C. Jiang, W. Shi, F. Cheng, K. Chen, H. Yang, Z. Zhang,
J. He, H. Zhang, G. Wei, W. Zhao, F. Zhou, D. Qi, H. Yi, S. Liu, and
F. Chen, “DB-GPT: Empowering Database Interactions with Private
Large Language Models,” Jan. 2024, arXiv:2312.17449. [Online].
Available: http://arxiv.org/abs/2312.17449
[54] J. Dou and X. Zhao, “Design and Application of Online Teaching
Resource Platform for College English Based on Retrieval-Augmented
Generation,” inProceedings of the 2nd International Conference on
Educational Knowledge and Informatization, ser. EKI ’24. New
York, NY , USA: Association for Computing Machinery, Oct. 2024, pp.
111–115. [Online]. Available: https://dl.acm.org/doi/10.1145/3691720.
3691739
[55] G. Zyskind, T. South, and A. Pentland, “Don‘t forget private
retrieval: distributed private similarity search for large language
models,” inProceedings of the Fifth Workshop on Privacy in
Natural Language Processing. Bangkok, Thailand: Association for
Computational Linguistics, Aug. 2024, pp. 7–19. [Online]. Available:
https://aclanthology.org/2024.privatenlp-1.2/
[56] D. Agarwal, A. Fabbri, B. Risher, P. Laban, S. Joty, and C.-S. Wu,
“Prompt leakage effect and mitigation strategies for multi-turn LLM
applications,” inProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing: Industry Track. Miami,
Florida, US: Association for Computational Linguistics, Nov. 2024,
pp. 1255–1275. [Online]. Available: https://aclanthology.org/2024.
emnlp-industry.94/
[57] J. Xue, M. Zheng, Y . Hu, F. Liu, X. Chen, and Q. Lou, “BadRAG:
Identifying Vulnerabilities in Retrieval Augmented Generation of
Large Language Models,” Jun. 2024, arXiv:2406.00083. [Online].
Available: http://arxiv.org/abs/2406.00083
[58] H. Zhang, J. Huang, K. Mei, Y . Yao, Z. Wang, C. Zhan, H. Wang,
and Y . Zhang, “Agent security bench (ASB): Formalizing and
benchmarking attacks and defenses in LLM-based agents,” inThe
Thirteenth International Conference on Learning Representations,

2025. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/
2025/file/5750f91d8fb9d5c02bd8ad2c3b44456b-Paper-Conference.pdf
[59] Z. Chen, Z. Xiang, C. Xiao, D. Song, and B. Li, “AgentPoison:
Red-teaming LLM agents via poisoning memory or knowledge
bases,” inAdvances in Neural Information Processing Systems,
vol. 37. Curran Associates, Inc., 2024, pp. 130 185–130 213.
[Online]. Available: https://proceedings.neurips.cc/paper files/paper/
2024/file/eb113910e9c3f6242541c1652e30dfd6-Paper-Conference.pdf
[60] J. Zhu, L. Yan, H. Shi, D. Yin, and L. Sha, “ATM: Adversarial tuning
multi-agent system makes a robust retrieval-augmented generator,”
inProceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing. Miami, Florida, USA: Association for
Computational Linguistics, Nov. 2024, pp. 10 902–10 919. [Online].
Available: https://aclanthology.org/2024.emnlp-main.610/
[61] Z. Chen, Y . Gong, J. Liu, M. Chen, H. Liu, Q. Cheng, F. Zhang,
W. Lu, and X. Liu, “Flippedrag: Black-box opinion manipulation
adversarial attacks to retrieval-augmented generation models,” in
Proceedings of the 2025 ACM SIGSAC Conference on Computer
and Communications Security, ser. CCS ’25. New York, NY , USA:
Association for Computing Machinery, 2025, p. 4109–4123. [Online].
Available: https://doi.org/10.1145/3719027.3765023
[62] R. Jiao, S. Xie, J. Yue, T. Sato, L. Wang, Y . Wang, Q. A. Chen, and
Q. Zhu, “Can we trust embodied agents? exploring backdoor attacks
against embodied LLM-based decision-making systems,” Oct. 2024,
arXiv:2405.20774. [Online]. Available: http://arxiv.org/abs/2405.20774
[63] C. Xiang, T. Wu, Z. Zhong, D. Wagner, D. Chen, and P. Mittal,
“Certifiably robust RAG against retrieval corruption,” May 2024,
arXiv:2405.15556. [Online]. Available: http://arxiv.org/abs/2405.15556
[64] A. RoyChowdhury, M. Luo, P. Sahu, S. Banerjee, and M. Tiwari,
“ConfusedPilot: Confused deputy risks in RAG-based LLMs,” Oct.
2024, arXiv:2408.04870. [Online]. Available: http://arxiv.org/abs/2408.
04870
[65] F. Fang, Y . Bai, S. Ni, M. Yang, X. Chen, and R. Xu,
“Enhancing noise robustness of retrieval-augmented language models
with adaptive adversarial training,” inProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). Bangkok, Thailand: Association for
Computational Linguistics, Aug. 2024, pp. 10 028–10 039. [Online].
Available: https://aclanthology.org/2024.acl-long.540/
[66] E. Altinisik, S. Messaoud, H. T. Sencar, H. Sajjad, and S. Chawla,
“Exploiting the layered intrinsic dimensionality of deep models for
practical adversarial training,” May 2024, arXiv:2405.17130. [Online].
Available: http://arxiv.org/abs/2405.17130
[67] T. Ju, Y . Wang, X. Ma, P. Cheng, H. Zhao, Y . Wang, L. Liu, J. Xie,
Z. Zhang, and G. Liu, “Flooding spread of manipulated knowledge in
LLM-based multi-agent communities,” Jul. 2024, arXiv:2407.07791.
[Online]. Available: http://arxiv.org/abs/2407.07791
[68] Y . Zhang, Q. Li, T. Du, X. Zhang, X. Zhao, Z. Feng, and J. Yin,
“HijackRAG: Hijacking Attacks against Retrieval-Augmented Large
Language Models,” Oct. 2024, arXiv:2410.22832. [Online]. Available:
http://arxiv.org/abs/2410.22832
[69] A. Shafran, R. Schuster, and V . Shmatikov, “Machine against the rag:
jamming retrieval-augmented generation with blocker documents,” in
Proceedings of the 34th USENIX Conference on Security Symposium,
ser. SEC ’25. USA: USENIX Association, 2025.
[70] W. Zou, R. Geng, B. Wang, and J. Jia, “Poisonedrag: knowledge
corruption attacks to retrieval-augmented generation of large language
models,” inProceedings of the 34th USENIX Conference on Security
Symposium, ser. SEC ’25. USA: USENIX Association, 2025.
[71] X. Jiang, Y . Fang, R. Qiu, H. Zhang, Y . Xu, H. Chen, W. Zhang,
R. Zhang, Y . Fang, X. Ma, X. Chu, J. Zhao, and Y . Wang, “TC–RAG:
Turing–complete RAG’s case study on medical LLM systems,”
inProceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), W. Che,
J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria:
Association for Computational Linguistics, Jul. 2025, pp. 11 400–
11 426. [Online]. Available: https://aclanthology.org/2025.acl-long.558/
[72] A. Kuppa, J. Nicholls, and N.-A. Le-Khac, “Manipulating
Prompts and Retrieval-Augmented Generation for LLM Service
Providers:,” inProceedings of the 21st International Conference
on Security and Cryptography. Dijon, France: SCITEPRESS
- Science and Technology Publications, 2024, pp. 777–785.
[Online]. Available: https://www.scitepress.org/DigitalLibrary/Link.
aspx?doi=10.5220/0012803100003767[73] F. Nazary, Y . Deldjoo, and T. Di Noia, “A resource for
studying textual poisoning attacks against embedding-based retrieval-
augmented generation in recommender systems,”GENNEXT@
SIGIR’25, 2025. [Online]. Available: https://sigirgennext.github.io/
GENNEXT-SIGIR-25/submissions/gennext sigir25 11.pdf
[74] Y . Mo, M. Tang, R. Lin, B. Zhou, and X. Li, “Broken bags: Disrupting
service through the contamination of large language models with
misinformation,”IEEE Access, vol. 13, p. 109607–109623, 2025.
[75] F. Nazary, Y . Deldjoo, and T. d. Noia, “Poison-RAG: Adversarial data
poisoning attacks on retrieval-augmented generation in recommender
systems,” inAdvances in Information Retrieval, C. Hauff, C. Macdon-
ald, D. Jannach, G. Kazai, F. M. Nardini, F. Pinelli, F. Silvestri, and
N. Tonellotto, Eds. Cham: Springer Nature Switzerland, 2025, p.
239–251.
[76] C. Zhang, X. Zhang, J. Lou, K. Wu, Z. Wang, and X. Chen,
“PoisonedEye: Knowledge poisoning attack on retrieval-augmented
generation based large vision-language models,” inForty-second
International Conference on Machine Learning, Jun. 2025. [Online].
Available: https://openreview.net/forum?id=6SIymOqJlc
[77] Y . Jiao, X. Wang, and K. Yang, “PR-Attack: Coordinated prompt-RAG
attacks on retrieval-augmented generation in large language models
via bilevel optimization,” inProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information
Retrieval, ser. SIGIR ’25. New York, NY , USA: Association for
Computing Machinery, Jul. 2025, p. 656–667. [Online]. Available:
https://dl.acm.org/doi/10.1145/3726302.3730058
[78] B. Zhang, H. Xin, M. Fang, Z. Liu, B. Yi, T. Li, and Z. Liu, “Traceback
of poisoning attacks to retrieval-augmented generation,” inProceedings
of the ACM on Web Conference 2025, ser. WWW ’25. New York, NY ,
USA: Association for Computing Machinery, Apr. 2025, p. 2085–2097.
[Online]. Available: https://dl.acm.org/doi/10.1145/3696410.3714756
[79] P. Cheng, Y . Ding, T. Ju, Z. Wu, W. Du, P. Yi, Z. Zhang, and G. Liu,
“TrojanRAG: Retrieval-augmented generation can be backdoor driver
in large language models,” Jul. 2024, arXiv:2405.13401. [Online].
Available: http://arxiv.org/abs/2405.13401
[80] C. Clop and Y . Teglia, “Backdoored retrievers for prompt injection
attacks on retrieval augmented generation of large language
models,” Oct. 2024, arXiv:2410.14479. [Online]. Available: http:
//arxiv.org/abs/2410.14479
[81] Y . Peng, J. Wang, H. Yu, and A. Houmansadr, “Data Extraction
Attacks in Retrieval-Augmented Generation via Backdoors,” Nov.
2024, arXiv:2411.01705 version: 1. [Online]. Available: http:
//arxiv.org/abs/2411.01705
[82] S. Pfrommer, Y . Bai, T. Gautam, and S. Sojoudi, “Ranking
manipulation for conversational search engines,” inProceedings of
the 2024 Conference on Empirical Methods in Natural Language
Processing. Miami, Florida, USA: Association for Computational
Linguistics, Nov. 2024, pp. 9523–9552. [Online]. Available: https:
//aclanthology.org/2024.emnlp-main.534/
[83] K. Feng, G. Zhang, H. Tian, H. Xu, Y . Zhang, T. Zhu, M. Ding, and
B. Liu, “RAGLeak: Membership inference attacks on RAG-based large
language models,” inAustralasian Conference on Information Security
and Privacy, W. Susilo and J. Pieprzyk, Eds. Singapore: Springer
Nature, 2025, p. 147–166.
[84] A. Bondarenko and A. Viehweger, “LLM robustness against
misinformation in biomedical question answering,” Oct. 2024,
arXiv:2410.21330. [Online]. Available: http://arxiv.org/abs/2410.21330
[85] Z. Hu, C. Wang, Y . Shu, H.-Y . Paik, and L. Zhu, “Prompt Perturbation
in Retrieval-Augmented Generation based Large Language Models,”
inProceedings of the 30th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, ser. KDD ’24. New York, NY , USA:
Association for Computing Machinery, Aug. 2024, pp. 1119–1130.
[Online]. Available: https://dl.acm.org/doi/10.1145/3637528.3671932
[86] Y . Liang, Z. Shi, Z. Song, and Y . Zhou, “Differential Privacy of Cross-
Attention with Provable Guarantee,” Oct. 2024, arXiv:2407.14717.
[Online]. Available: http://arxiv.org/abs/2407.14717
[87] G. Xiong, Q. Jin, Z. Lu, and A. Zhang, “Benchmarking retrieval-
augmented generation for medicine,” inFindings of the Association
for Computational Linguistics: ACL 2024. Bangkok, Thailand:
Association for Computational Linguistics, Aug. 2024, pp. 6233–6251.
[Online]. Available: https://aclanthology.org/2024.findings-acl.372/
[88] S. Liu, Z. Yu, F. Huang, Y . Bulbulia, A. Bergen, and M. Liut,
“Can Small Language Models With Retrieval-Augmented Generation
Replace Large Language Models When Learning Computer Science?”

inProceedings of the 2024 on Innovation and Technology in
Computer Science Education V. 1, ser. ITiCSE 2024. New York, NY ,
USA: Association for Computing Machinery, Jul. 2024, pp. 388–393.
[Online]. Available: https://dl.acm.org/doi/10.1145/3649217.3653554
[89] R. Shan, “Certifying Generative AI: Retrieval-Augmented Generation
Chatbots in High-Stakes Environments,”Computer, vol. 57, no. 9, pp.
35–44, Sep. 2024, conference Name: Computer. [Online]. Available:
https://ieeexplore.ieee.org/document/10660589
[90] W. Lu, Z. Zeng, J. Wang, Z. Lu, Z. Chen, H. Zhuang, and C. Chen,
“Eraser: Jailbreaking defense in large language models via unlearning
harmful knowledge,”arXiv preprint arXiv:2404.05880, 2024.
[91] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh,
C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova,
L. Jones, M. Kelcey, M.-W. Chang, A. M. Dai, J. Uszkoreit,
Q. Le, and S. Petrov, “Natural questions: A benchmark for
question answering research,”Transactions of the Association for
Computational Linguistics, vol. 7, pp. 452–466, 2019. [Online].
Available: https://aclanthology.org/Q19-1026/
[92] T. Nguyen, M. Rosenberg, X. Song, J. Gao, S. Tiwary,
R. Majumder, and L. Deng, “MS MARCO: A human generated
machine reading comprehension dataset,” November 2016. [Online].
Available: https://www.microsoft.com/en-us/research/publication/
ms-marco-human-generated-machine-reading-comprehension-dataset/
[93] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen, R. Salakhutdinov,
and C. D. Manning, “HotpotQA: A dataset for diverse, explainable
multi-hop question answering,” inProceedings of the 2018 Conference
on Empirical Methods in Natural Language Processing. Brussels,
Belgium: Association for Computational Linguistics, Oct.-Nov.
2018, pp. 2369–2380. [Online]. Available: https://aclanthology.org/
D18-1259/
[94] M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer, “TriviaQA:
A large scale distantly supervised challenge dataset for reading
comprehension,” inProceedings of the 55th Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers). Vancouver, Canada: Association for Computational
Linguistics, Jul. 2017, pp. 1601–1611. [Online]. Available: https:
//aclanthology.org/P17-1147/
[95] J. Berant, A. Chou, R. Frostig, and P. Liang, “Semantic parsing
on Freebase from question-answer pairs,” inProceedings of
the 2013 Conference on Empirical Methods in Natural Language
Processing. Seattle, Washington, USA: Association for Computational
Linguistics, Oct. 2013, pp. 1533–1544. [Online]. Available: https:
//aclanthology.org/D13-1160/
[96] A. Mallen, A. Asai, V . Zhong, R. Das, D. Khashabi, and H. Hajishirzi,
“When not to trust language models: Investigating effectiveness
of parametric and non-parametric memories,” inProceedings of
the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). Toronto, Canada: Association
for Computational Linguistics, Jul. 2023, pp. 9802–9822. [Online].
Available: https://aclanthology.org/2023.acl-long.546/
[97] M. Geva, D. Khashabi, E. Segal, T. Khot, D. Roth, and J. Berant,
“Did Aristotle use a laptop? A question answering benchmark with
implicit reasoning strategies,”Transactions of the Association for
Computational Linguistics, vol. 9, pp. 346–361, 2021. [Online].
Available: https://aclanthology.org/2021.tacl-1.21/
[98] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+
questions for machine comprehension of text,” inProceedings of
the 2016 Conference on Empirical Methods in Natural Language
Processing. Austin, Texas: Association for Computational Linguistics,
Nov. 2016, pp. 2383–2392. [Online]. Available: https://aclanthology.
org/D16-1264/
[99] L. Huang, R. Le Bras, C. Bhagavatula, and Y . Choi, “Cosmos
QA: Machine reading comprehension with contextual commonsense
reasoning,” inProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International
Joint Conference on Natural Language Processing (EMNLP-IJCNLP).
Hong Kong, China: Association for Computational Linguistics, Nov.
2019, pp. 2391–2401. [Online]. Available: https://aclanthology.org/
D19-1243/
[100] P. Baudi ˇs and J. ˇSediv ´y, “Modeling of the question answering task
in the yodaqa system,” inExperimental IR Meets Multilinguality,
Multimodality, and Interaction, J. Mothe, J. Savoy, J. Kamps, K. Pinel-
Sauvagnat, G. Jones, E. San Juan, L. Capellato, and N. Ferro, Eds.
Cham: Springer International Publishing, 2015, pp. 222–228.[101] J. Kasai, K. Sakaguchi, Y . Takahashi, R. Le Bras, A. Asai, X. V .
Yu, D. Radev, N. A. Smith, Y . Choi, and K. Inui, “Realtime QA:
what’s the answer right now?” inProceedings of the 37th International
Conference on Neural Information Processing Systems, ser. NIPS ’23.
Red Hook, NY , USA: Curran Associates Inc., 2023.
[102] D. Jin, E. Pan, N. Oufattole, W.-H. Weng, H. Fang, and
P. Szolovits, “What disease does this patient have? a large-scale
open domain question answering dataset from medical exams,”
Applied Sciences, vol. 11, no. 14, 2021. [Online]. Available:
https://www.mdpi.com/2076-3417/11/14/6421
[103] Y . Li, Z. Li, K. Zhang, R. Dan, S. Jiang, and Y . Zhang, “Chatdoctor:
A medical chat model fine-tuned on a large language model meta-ai
(llama) using medical domain knowledge,”Cureus, vol. 15, no. 6, 2023.
[104] V . Boteva, D. Gholipour, A. Sokolov, and S. Riezler, “A full-text
learning to rank dataset for medical information retrieval,” inAdvances
in Information Retrieval, N. Ferro, F. Crestani, M.-F. Moens, J. Mothe,
F. Silvestri, G. M. Di Nunzio, C. Hauff, and G. Silvello, Eds. Cham:
Springer International Publishing, 2016, pp. 716–722.
[105] Q. Jin, B. Dhingra, Z. Liu, W. Cohen, and X. Lu, “PubMedQA: A
dataset for biomedical research question answering,” inProceedings
of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP). Hong Kong, China:
Association for Computational Linguistics, Nov. 2019, pp. 2567–2577.
[Online]. Available: https://aclanthology.org/D19-1259/
[106] A. Krithara, A. Nentidis, K. Bougiatiotis, and G. Paliouras, “BioASQ-
QA: A manually curated corpus for Biomedical Question Answering,”
Scientific Data, vol. 10, no. 1, p. 170, Mar. 2023. [Online]. Available:
https://doi.org/10.1038/s41597-023-02068-4
[107] L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster,
J. Phang, H. He, A. Thite, N. Nabeshimaet al., “The pile: An
800gb dataset of diverse text for language modeling,”arXiv preprint
arXiv:2101.00027, 2020.
[108] S. Kim, S. Yun, H. Lee, M. Gubri, S. Yoon, and S. J. Oh, “ProPILE:
probing privacy leakage in large language models,” inProceedings of
the 37th International Conference on Neural Information Processing
Systems, ser. NIPS ’23. Red Hook, NY , USA: Curran Associates Inc.,
2023.
[109] M. Maia, S. Handschuh, A. Freitas, B. Davis, R. McDermott,
M. Zarrouk, and A. Balahur, “WWW’18 open challenge: Financial
opinion mining and question answering,” inCompanion Proceedings
of the The Web Conference 2018, ser. WWW ’18. Republic
and Canton of Geneva, CHE: International World Wide Web
Conferences Steering Committee, 2018, p. 1941–1942. [Online].
Available: https://doi.org/10.1145/3184558.3192301
[110] Carnegie Mellon University, “Enron email dataset,” https://www.cs.
cmu.edu/ ∼enron/, 2015.
[111] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer sentinel
mixture models,”arXiv preprint arXiv:1609.07843, 2016.
[112] L. Derczynski, E. Nichols, M. van Erp, and N. Limsopatham,
“Results of the WNUT2017 shared task on novel and emerging
entity recognition,” inProceedings of the 3rd Workshop on
Noisy User-generated Text. Copenhagen, Denmark: Association
for Computational Linguistics, Sep. 2017, pp. 140–147. [Online].
Available: https://aclanthology.org/W17-4418/
[113] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning,
A. Ng, and C. Potts, “Recursive deep models for semantic
compositionality over a sentiment treebank,” inProceedings of
the 2013 Conference on Empirical Methods in Natural Language
Processing. Seattle, Washington, USA: Association for Computational
Linguistics, Oct. 2013, pp. 1631–1642. [Online]. Available: https:
//aclanthology.org/D13-1170/
[114] X. Zhang, J. Zhao, and Y . LeCun, “Character-level convolutional net-
works for text classification,” inProceedings of the 29th International
Conference on Neural Information Processing Systems - Volume 1, ser.
NIPS’15. Cambridge, MA, USA: MIT Press, 2015, p. 649–657.
[115] F. M. Harper and J. A. Konstan, “The MovieLens datasets: History
and context,”ACM Trans. Interact. Intell. Syst., vol. 5, no. 4, Dec.
2015. [Online]. Available: https://doi.org/10.1145/2827872
[116] A. Parrish, A. Chen, N. Nangia, V . Padmakumar, J. Phang,
J. Thompson, P. M. Htut, and S. Bowman, “BBQ: A hand-built bias
benchmark for question answering,” inFindings of the Association for
Computational Linguistics: ACL 2022. Dublin, Ireland: Association

for Computational Linguistics, May 2022, pp. 2086–2105. [Online].
Available: https://aclanthology.org/2022.findings-acl.165/
[117] S. Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. Koh, M. Iyyer,
L. Zettlemoyer, and H. Hajishirzi, “FActScore: Fine-grained atomic
evaluation of factual precision in long form text generation,” in
Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing. Singapore: Association for Computational
Linguistics, Dec. 2023, pp. 12 076–12 100. [Online]. Available:
https://aclanthology.org/2023.emnlp-main.741/
[118] Y . Hu and Y . Lu, “RAG and RAU: A survey on retrieval-augmented
language model in natural language processing,” Apr. 2024,
arXiv:2404.19543. [Online]. Available: http://arxiv.org/abs/2404.19543
[119] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu, “Evaluation
of retrieval-augmented generation: A survey,” inBig Data, W. Zhu,
H. Xiong, X. Cheng, L. Cui, Z. Dou, J. Dong, S. Pang, L. Wang,
L. Kong, and Z. Chen, Eds. Singapore: Springer Nature Singapore,
2025, pp. 102–120.
[120] V . Smith, A. S. Shamsabadi, C. Ashurst, and A. Weller, “Identifying
and mitigating privacy risks stemming from language models: A
survey,”arXiv preprint arXiv:2310.01424, 2023.
[121] B. Yan, K. Li, M. Xu, Y . Dong, Y . Zhang, Z. Ren, and
X. Cheng, “On protecting the data privacy of large language
models (llms) and llm agents: A literature review,”High-Confidence
Computing, vol. 5, no. 2, p. 100300, 2025. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S2667295225000042
[122] A. Golda, K. Mekonen, A. Pandey, A. Singh, V . Hassija, V . Chamola,
and B. Sikdar, “Privacy and security concerns in generative ai: A
comprehensive survey,”IEEE Access, vol. 12, pp. 48 126–48 144, 2024.
[123] S. Shahriar, S. Allana, S. M. Hazratifard, and R. Dara, “A survey ofprivacy risks and mitigation strategies in the artificial intelligence life
cycle,”IEEE Access, vol. 11, pp. 61 829–61 854, 2023.
[124] H.-P. H. Lee, L. Gao, S. Yang, J. Forlizzi, and S. Das, “I
don’t know if we’re doing good. I don’t know if we’re doing
bad”: Investigating how practitioners scope, motivate, and conduct
privacy work when developing AI products,” in33rd USENIX
Security Symposium (USENIX Security 24). Philadelphia, PA:
USENIX Association, Aug. 2024, pp. 4873–4890. [Online]. Available:
https://www.usenix.org/conference/usenixsecurity24/presentation/lee
[125] A. Klymenko, S. Meisenbacher, P. G. Kelley, S. T. Peddinti, K. Thomas,
and F. Matthes, “We are not future-ready”: Understanding AI privacy
risks and existing mitigation strategies from the perspective of AI
developers in Europe,” inTwenty-First Symposium on Usable Privacy
and Security (SOUPS 2025), 2025, pp. 113–132. [Online]. Available:
https://www.usenix.org/conference/soups2025/presentation/klymenko
[126] P. G. Kelley, C. Cornejo, L. Hayes, E. S. Jin, A. Sedley, K. Thomas,
Y . Yang, and A. Woodruff, “There will be less privacy, of course”:
how and why people in 10 countries expect AI will affect privacy
in the future,” inProceedings of the Nineteenth USENIX Conference
on Usable Privacy and Security, ser. SOUPS ’23. USA: USENIX
Association, 2023.
APPENDIX
All supplemental materials, including a full list of literature
sources, our coded literature analysis, and complete table for
the mitigation scoring, can be found in our public GitHub
repository: https://github.com/sebischair/SoK-RAG-Privacy