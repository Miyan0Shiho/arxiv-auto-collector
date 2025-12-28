# ARBITER: AI-Driven Filtering for Role-Based Access Control

**Authors**: Michele Lorenzo, Idilio Drago, Dario Salvadori, Fabio Romolo Vayr

**Published**: 2025-12-23 17:25:51

**PDF URL**: [https://arxiv.org/pdf/2512.20535v1](https://arxiv.org/pdf/2512.20535v1)

## Abstract
Role-Based Access Control (RBAC) struggles to adapt to dynamic enterprise environments with documents that contain information that cannot be disclosed to specific user groups. As these documents are used by LLM-driven systems (e.g., in RAG) the problem is exacerbated as LLMs can leak sensitive data due to prompt truncation, classification errors, or loss of system context. We introduce \our, a system designed to provide RBAC in RAG systems. \our implements layered input/output validation, role-aware retrieval, and post-generation fact-checking. Unlike traditional RBAC approaches that rely on fine-tuned classifiers, \our uses LLMs operating in few-shot settings with prompt-based steering for rapid deployment and role updates. We evaluate the approach on 389 queries using a synthetic dataset. Experimental results show 85\% accuracy and 89\% F1-score in query filtering, close to traditional RBAC solutions. Results suggest that practical RBAC deployment on RAG systems is approaching the maturity level needed for dynamic enterprise environments.

## Full Text


<!-- PDF content starts -->

ARBITER: AI-Driven Filtering for Role-Based Access Control
Michele Lorenzo
Reply Spike
Turin, Italy
mi.lorenzo@reply.itIdilio Drago
University of Turin
Turin, Italy
idilio.drago@unito.itDario Salvadori
Reply Spike
Milan, Italy
d.salvadori@reply.itFabio Romolo Vayr
Reply Spike
Turin, Italy
f.vayr@reply.it
Abstract
Role-Based Access Control (RBAC) struggles to adapt to dynamic enterprise environments with
documents that contain information that cannot be disclosed to specific user groups. As these documents
are used by LLM-driven systems (e.g., in RAG) the problem is exacerbated as LLMs can leak sensitive
data due to prompt truncation, classification errors, or loss of system context. We introduce ARBITER,
a system designed to provide RBAC in RAG systems. ARBITER implements layered input/output
validation, role-aware retrieval, and post-generation fact-checking. Unlike traditional RBAC approaches
that rely on fine-tuned classifiers, ARBITER uses LLMs operating in few-shot settings with prompt-based
steering for rapid deployment and role updates. We evaluate the approach on 389 queries using a synthetic
dataset. Experimental results show 85% accuracy and 89% F1-score in query filtering, close to traditional
RBAC solutions. Results suggest that practical RBAC deployment on RAG systems is approaching the
maturity level needed for dynamic enterprise environments.
1 Introduction
Enterprises are rapidly embracing generative AI technologies, deploying them across local and cloud
infrastructures to enhance productivity, streamline workflows, and support decision-making [ 7]. Among these,
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm, allowing Large Language
Models (LLMs) to access external knowledge sources and produce more accurate, contextually grounded
outputs [ 4]. However, the integration of such capabilities in enterprise environments introduces security and
access control challenges [2].
A key problem is the risk of unauthorized access to sensitive data during the retrieval process. Generative
systems may expose or create information beyond a user’s clearance level due to document classification
errors in the RAG pipeline, prompt truncation, etc. Such errors may cause the model to misinterpret access
boundaries, disregard system instructions, or generate responses based on internal knowledge rather than
retrieved context, ultimately leading to unauthorized disclosures.
Traditionally, enterprises rely on Role-Based Access Control (RBAC), where users are assigned predefined
roles that dictate access to specific documents or data. RBAC operates through static filters, creating “visibility
cones” to limit data exposure. This method relies heavily on exact and static classifications, which are costly
to update and imperfect regardless. Fine-tuning models for classification adds further burden, requiring
labeled data and frequent retraining to accommodate evolving access policies.
Another critical aspect is the need to use local models, as enterprises are unwilling to expose private
documents to external parties during the retrieval process. This architectural choice limits access to the
capabilities of larger hosted models.
We evaluate a local, adaptive filtering approach that supplements RBAC with AI-driven safeguards
embedded throughout the RAG pipeline. For that we introduce ARBITER (AIRoleBasedIntelligentText
EvaluatoR) – a system that leverages NeMo Guardrails [ 8] to create a multi-layered filter chain without
relying on fine-tuned models.
1arXiv:2512.20535v1  [cs.CR]  23 Dec 2025

Our main contributions are: (i) we propose a few-shot approach for RBAC using foundational LLMs that
eliminates the need for document classifiers and enables rapid deployment; (ii) we test ARBITER to augment
RBAC systems through layered input/output validation and role-aware retrieval. Using a synthetic dataset of
roles and a dataset of documents, we validate the system, showing 85% accuracy and 89% F1-score in access
control enforcement.
These results are only slightly lower than static RBAC systems applied to the same synthetic dataset.
While there is still a performance gap before reaching deployability, they show that the filtering architecture
is promising for enterprise deployment in RBAC contexts. ARBITER allows the enforcement of context-
sensitive security protocols without the high costs of fine-tuning, offering a practical approach for secure
RAG deployment in real-world enterprise settings.
2 Background and Related Work
2.1 RBAC Systems
RBAC is a widely recognized security model used to manage access to resources within organizations based
on individual user roles. In RBAC, permissions are associated with roles, and users are assigned to these roles,
enabling centralized management of access rights [ 1,9]. This model simplifies administrative tasks, improves
security, and ensures that users access only the information and functions necessary for their designated roles.
RBAC operates on the principles of least privilege and separation of duties, ensuring that users maintain
the minimum level of access required to perform their tasks. The scalability and adaptability of the RBAC
approach make it suitable for various corporate environments and applications.
2.2 Related Work
Our approach builds upon recent advances in AI-driven access control and guardrail systems for LLMs.
NVIDIA NeMo Guardrails [ 8] provides an open-source toolkit for adding programmable guardrails to LLM-
based conversational systems, supporting input rails, dialog rails, retrieval rails, and execution rails. Recent
developments include new NIM microservices for content safety, topic control, and jailbreak detection,
specifically designed for agentic AI applications.
The GUARDIAN framework [ 6] introduced a multi-tiered defense system with three integrated filtering
layers: ethical constraints embedded in system prompts, BERT-based toxicity classification, and LLM-
based self-evaluation. This architecture achieved good protection against adversarial prompts in controlled
experiments by using LLMs in few-shot settings rather than fine-tuned classifiers.
Considering RBAC for RAG systems, Elasticsearch’s approach to RAG-RBAC integration shows multi-
level access control across clusters, indices, documents, and fields, with enterprise identity provider integra-
tion [ 11]. However, current implementations face challenges in rapidly evolving enterprise environments
where both manual rule updates and automated classification prove inefficient [ 3]. We will use Elasticsearch’s
approach as baseline in our comparisons.
The field has seen growing attention to AI security. The OWASP Top 10 for LLMs 20251identifies
vector and embedding weaknesses in RAG systems as a vulnerability, emphasizing the need for integrated
access control rules to prevent information leakage. Modern guardrail systems now focus on real-time
protection against data leakage, prompt injection, and jailbreaking attempts, with emphasis on low-latency
and high-accuracy filtering.
ARBITER is novel because it leverages few-shot capabilities of foundational LLMs [ 12] for access
control, eliminating the need for fine-tuned classifiers and enabling rapid role evolution. This approach
1https://genai.owasp.org/llm-top-10/
2

Input filter
(NeMO)LLM Model
(Responder)Fact Checker
(NeMO)Output Filter
(NeMO)
Access manager Retriever
Vector  DB
Documents for RAG!
Answer?
Question
Perform
Authentication
Role of
the user
Document
EmbeddingsFiltered
Question?
Question
Retrieved
Embeddings
+ 
RoleComplete
promptUnfiltered
Answer
Embeddings 
(question+role)Role of
the userFigure 1: ARBITER architecture. The system is built on LangChain. TheInput Filter, theResponderand the
Output Filterrely on a single underlying LLM. The Fact Checker is built on a specific fine-tuned RoBERTa
model.
addresses the scalability challenges identified in traditional RBAC systems while maintaining the multi-
layered protection principles established by frameworks like GUARDIAN.
3 ARBITER Architecture and Design
3.1 System Design
ARBITER (Figure 1) builds upon LangChain, NeMo Guardrails, and ChromaDB to create an RBAC-
compliant RAG architecture. The core innovation lies in extending NeMo’s guardrail framework with
role-aware filters that operate dynamically on live message streams. This architecture ensures that user
queries and system outputs remain within the boundaries of users’ roles. It leverages LLMs for natural
language understanding of roles, filtering of queries, and document retrieval.
Traditional RAG systems rely primarily on retrieval-time filtering. This can be insufficient for strict
access control requirements. It also limits the possibilities opened by the use of LLMs to answer queries
about private enterprise documents. Our approach addresses these limitations by implementing a defense
strategy with role-aware filtering at multiple pipeline stages. Central to our design is the extension of
NeMo’s guardrails framework. While NeMo provides standard guardrails for general content moderation,
we contribute custom role-aware filters that classify each message based on role descriptions dynamically
passed in the prompt. This approach removes the need for fine-tuned models while maintaining the flexibility
required for evolving role definitions.
Users’ roles are injected at multiple stages of the pipeline (arrows in Figure 1). The injected roles guide
retrieval toward role-appropriate content. This design also opens the possibility for natural language definition
and updates of users’ roles (see Appendix A). We append role information to queries during embedding to
increase the probability of retrieving only relevant content while maintaining the semantic similarity approach
that makes vector-based retrieval effective.
To ensure response fidelity, our design incorporates a fact-checking layer that validates LLM outputs
against retrieved knowledge, serving the dual purpose of preventing generation errors and ensuring that
generated content remains grounded in authorized sources.
3

3.2 Implementation
ARBITER implements a five-stage pipeline designed to enforce access control at multiple checkpoints. The
system is built on LangChain [ 10] for modular integration, with Ollama providing local model hosting and
nomic-embed-text embedding model [ 5]. This local deployment approach addresses enterprise requirements
for data sovereignty while maintaining the flexibility of modern LLM architectures.
The pipeline begins with a NeMo input guardrail (i.e.,Input Filter) that validates incoming queries against
the user’s role-based permissions using Chain-of-Thought and few-shot prompting – see Appendix A for an
example of the used prompts. Next, queries are augmented with role information to guide retrieval toward
documents that align with access permissions. For vector-based retrieval, we employ ChromaDB to store
documents embedded with nomic-embed-text alongside associated role metadata. Role-injected prompts
are compared against the embeddings through cosine similarity, with a similarity threshold of 0.5 (found
empirically) to reduce unauthorized data leakage.
The query is then passed to theLLM Responderthat answers the queries based on internal knowledge and
retrieved content. The generated text passes theFact Checkerusing an AlignScore-based validation rail from
the NeMo community library [ 13]. When generated content cannot be validated against retrieved context, the
system blocks unsupported responses and replaces them with fallback statements.
At the end of the pipeline, the response is evaluated by a finalOutput Filterthat aims to block unauthorized
content that may have been generated by the responder. This is achieved again with Chain-of-Thought and
few-shot prompting on response and users’ roles (see Appendix A).
ARBITER uses the same underlying model for theInput Filter, theResponder, and theOutput Filter. We
present a study to select the best model for these components next. TheFact Checkeris instead built with a
fine-tuned RoBERTa model.
4 ARBITER Validation
This section presents a validation of ARBITER. We evaluate the effectiveness of our few-shot filtering
approach by testing across multiple large language models and by comparing AI-driven RBAC against a
traditional static filtering method.
4.1 Synthetic Dataset Design
Given the lack of benchmarks for role-based access control in RAG systems, we construct a synthetic dataset
that captures the characteristics of enterprise access control scenarios. In particular, real enterprise data
is typically proprietary and cannot be shared for research purposes, while existing NLP datasets lack the
structured role-permission relationships required for access control validation.
We focus on a ludic example for which we expect public LLMs to have general-purpose knowledge, but
no access to our queries/answers. The idea here is to test whether LLMs could leak information about private
documents (thus violating RBAC) based on its internal knowledge.
We build two complementary datasets usingDungeons & Dragonscontent, which provides a well-
defined, publicly available domain with clear role-based restrictions. The Spells Dataset comprises 603
spell descriptions retrieved from Wikidot2under Creative Commons CC BY-SA 3.0 license, each tagged by
character class (Wizard, Cleric, Ranger, etc). This dataset serves as the knowledge base for our RAG system,
with each spell representing a document that should be accessible only to users with appropriate roles.
To evaluate system performance, we create a test set of 389 user queries generated manually with the
assistance of GPT-4o. The test set includes 261 queries that should be blocked (users requesting spells outside
2https://dnd5e.wikidot.com/
4

Table 1: ARBITER query handling vs. static filtering. Static filters can make mistakes both when retrieval
fails or when the downstream responder generates unauthorized content based on its internal knowledge.
Query Example User
RoleLabel ARBITER Expectation Static Filters
"Tell me about Fireball spell" Wizard Allow Role-aware reasoning:
Wizards can cast Fireball
according to D&D rulesExact match: Wizard ∈[Wiz-
ard, Sorcerer] spell class tags
"How does Cure Wounds
work?"Wizard Block Role-aware reasoning:
Healing spells are not
typically available to
Wizards.No match: Wizard /∈[Cleric,
Paladin] spell class tags
"What spells can heal al-
lies?"Cleric Allow Context-aware: Identifies
healing as Cleric domain
through semantic under-
standingExact match: Cleric ∈
[Cleric, Paladin] spell class
tags
"Can I use magic to fix
things?"Artificer Allow Understands "fix" relates
to Artificer repair abili-
ties through natural lan-
guage processingExact match: Artificer ∈[Ar-
tificer] spell class tags
"Tell me about Lightning
Bolt"Sorcerer Block Role-aware reasoning:
Recognizes this should
be blocked for SorcerersRetrieval failure: Lightning
Bolt document not retrieved
due to cosine similarity < 0.5
threshold
"Could you show me Con-
jure Animals? I’d like to
know how it works for sum-
moning creatures to help
me."Paladin Block Role-aware reasoning:
Paladin cannot cast Con-
jure Animals.Generation failure: Loses
prompt instructions, and gen-
erates an answer based on in-
ternal knowledge.
their role permissions) and 128 queries that should be allowed. Each query has been carefully crafted to
test different aspects of the access control system, including direct requests for specific spells (e.g., "Tell me
about Fireball spell"), indirect references to spell effects (e.g., "What spells can heal allies?"), and edge cases
involving conceptual queries that require semantic understanding (e.g., "Can I use magic to fix things?").
Examples of query types and expected access control behavior are shown in Table 1.
While with its limitations (see Section 6), this synthetic approach offers some advantages over real
enterprise data: it provides ground truth labels for access decisions, enables controlled testing of specific
access control scenarios, and allows for reproducible evaluation. While the domain may seem artificial, the
underlying access control patterns – users with defined roles requesting access to role-restricted documents –
directly mirror enterprise scenarios such as legal document access, medical record retrieval, or financial data
queries.
4.2 Experimental Methodology
We aim to provide insights into the effectiveness of few-shot access control. We tested four state-of-the-art
open-source options for the underlying LLM supporting Input and Output filter as well as the Responder:
Qwen2.5-14B, Phi4-14B, DeepSeek-r1 and Llama3.1-8B. We focus on local small models, since enterprise
environments are constrained by computational resources and legal compliance requirements that prevent
using online models for sensitive data processing.
For each model, we evaluate five filtering configurations: no filtering, input filtering only, output filtering
only, combined input-output filtering, and a static filter baseline – Elasticsearch’s solution for RBAC-RAG
5

Table 2: ARBITER vs. static filter baseline.
ModelARBITER Static Filter
Acc. F1 Acc. F1
Qwen2.5-
14B0.85 0.89 0.91 0.93
Phi4-14B 0.82 0.87 0.90 0.93
DeepSeek-
r10.78 0.84 0.90 0.93
Llama3.1-
8B0.75 0.83 0.92 0.94
with a Responder LLM. The static filter serves as our baseline, representing traditional rule-based access
control. It implements RBAC through exact keyword matching against spell class labels. When a user
requests access to content, the filter performs exact string matching between the user’s assigned role and
the document’s class tags, blocking access when no exact match is found. Authorized documents are then
forwarded to the LLM Responder, which can still leak information from its internal knowledge.
All results are shown using accuracy and F1-score as primary metrics, chosen for their relevance to
access control scenarios where both false positives (blocking authorized access) and false negatives (allowing
unauthorized access) carry significant implications.
4.3 Few-Shot vs. Traditional Filtering
Table 2 summarizes our results, comparing ARBITER few-shot filtering approach against the static filter
baseline across all tested models.
First, notice how the static filtering presents sub-optimal performance (93-94% F1-score). The static
filter suffers from precision and recall limitations. This happens because of: i) limitations in the RAG
retrieval phase – the cosine similarity threshold of 0.5 filters out relevant documents during retrieval; ii) the
downstream responder that still produces sensitive content based on its knowledge. These cases occur in
approximately 7% of test cases and are illustrated with examples in Table 1. We prefer to keep these cases in
statistics to illustrate failures that similarity threshold and static filters may introduce to RAG pipelines.
Given this baseline constraint, ARBITER few-shot filtering approach shows good performance. Qwen2.5-
14B achieved 85% accuracy and 89% F1-score, representing only a 5-percentage-point gap from the static
filter baseline. Across all models, ARBITER achieved F1-scores between 0.83 and 0.89, consistently
approaching the static filter performance within 5-10 percentage points. This gap represents the trade-off
between flexibility and precision: while static filters achieve higher accuracy through rigid rule matching, our
approach provides comparable protection while supporting natural language role descriptions and dynamic
policy updates.
4.4 Ablation Study: Filter Components
Figure 2 illustrates the progressive improvement achieved by adding filter components, providing insights
into the contribution of each element. The analysis show distinct patterns across models and highlights the
importance of the multi-layered strategy.
Without any filtering, all models show poor access control performance, with F1-scores ranging from
0.32 (Phi4) to 0.43 (DeepSeek-r1) and accuracy values between 0.42-0.46. This baseline confirms that
embedding-based retrieval without protection results in unauthorized access across all models.
Input filtering alone provides substantial improvements across all models, with F1-scores jumping to
0.69-0.81 and accuracy improvements to 0.65-0.75. Notably, DeepSeek-r1 and Phi4 show the largest absolute
6

0.3 0.4 0.5 0.6 0.7 0.8 0.9
F1 Score0.30.40.50.60.70.80.9AccuracyQwen2.5
Phi4
DeepSeek-r1
Llama3.1
No Filter
Input Filter Only
Input and Output FiltersFigure 2: Performance progression across filtering strategies for all tested models. Each point shows the
F1-score (x-axis) and accuracy (y-axis) for different filtering configurations: No Filter (circles), Input Filter
Only (triangles), and Combined Filters (squares).
Table 3: Complete precision and recall breakdown across all filtering strategies, demonstrating the progression
from individual filters to combined approach and comparison with static filtering baseline.
ModelInput Only Output Only Combined Static Filter
Prec. Rec. Prec. Rec. Prec. Rec. Prec. Rec.
Qwen2.5-
14B0.83 0.60 0.86 0.87 0.87 0.91 0.89 0.98
Phi4-14B 0.83 0.77 0.89 0.68 0.86 0.88 0.92 0.93
DeepSeek-
r10.77 0.85 0.73 0.44 0.80 0.90 0.87 0.99
Llama3.1-
8B0.78 0.83 0.78 0.79 0.75 0.93 0.91 0.98
gains in both metrics (F1: 0.81, accuracy: 0.73-0.75), while Qwen2.5 benefits least from input filtering alone
(F1: 0.69, accuracy: 0.65). This stage primarily prevents unauthorized queries from reaching the generation
phase, reducing the propagation of access violations through the system.
Output filtering alone shows varying effectiveness across models – not shown in the figure for improving
visualization. Qwen2.5 achieves best performance with output filtering (F1: 0.87, accuracy: 0.82), signifi-
cantly outperforming its input filtering results. Conversely, DeepSeek-r1 shows reduced performance with
output-only filtering (F1: 0.55, accuracy: 0.52), indicating model-specific differences in how access control
violations manifest during generation versus input processing.
The combined filtering approach (top-right points in Figure 2) consistently achieves the best performance
across all models, with F1-scores reaching 0.83-0.89 and accuracy values of 0.75-0.85. The synergistic
effect between input and output filtering compensates for individual filter limitations regardless of the model
used. DeepSeek-r1 shows the largest improvement from combined filtering (F1: 0.84, accuracy: 0.78), while
Qwen2.5 maintained the highest absolute performance (F1: 0.89, accuracy: 0.85).
Table 3 provides a complete view of the precision-recall trade-offs across filtering strategies. The static
filter achieves the best precision (0.87-0.92) and recall (0.93-0.99). Our combined approach maintains
competitive performance with precision between 0.75-0.87 and recall of 0.88-0.93. Qwen2.5 benefits
significantly from output filtering (recall jumps from 0.60 to 0.87), while DeepSeek-r1 performs better with
7

input filtering (0.85 recall vs. 0.44 for output-only). These model-specific differences highlight why our
architecture uses both filtering stages rather than relying on a single approach.
This finding supports our architectural decision to implement multi-stage access control rather than
relying on single-point filtering, as the combined approach consistently outperforms individual filtering stages
across all metrics.
5 Conclusions
We present ARBITER, a few-shot approach to Role-Based Access Control in RAG systems that addresses
the challenge of dynamic access control in enterprise environments. By extending NVIDIA NeMo Guardrails
with custom role-aware filters, our system achieves 85% accuracy and 89% F1-score on access control
enforcement – approaching the performance of traditional static filtering (91% accuracy, 93% F1-score) while
providing greater flexibility for evolving role definitions.
Our contribution is in demonstrating that AI-driven RBAC is heading to achieve enterprise performance
without requiring fine-tuned classifiers on retraining cycles. Our multi-layered filtering approach, operating
at both input and output stages, provides protection against unauthorized access with natural language role
descriptions and dynamic policy updates. This represents a practical breakthrough for organizations seeking to
deploy RAG systems in access-sensitive environments without sacrificing the flexibility required in enterprise
operations.
6 Limitations
While our results indicate the feasibility of few-shot access control, several limitations need acknowledgment.
First, our evaluation relies on a synthetic dataset derived from Dungeons & Dragons content. Although
this domain provides well-defined role structures that mirror enterprise scenarios, validation on real enterprise
data with complex, evolving role hierarchies remains open to establish complete applicability of ARBITER.
The D&D domain may not capture the full complexity of enterprise access patterns, including temporal
access restrictions, hierarchical role inheritance, or cross-departmental permissions. In sum, more benchmark
datasets are needed.
Second, we ignore latency in our evaluation. ARBITER requires several extra steps when compared to
static filters due to the sequential nature of multiple LLM-based filtering stages. Each query passes input
filtering, retrieval, generation, fact-checking, and output filtering, creating cumulative delays that may prove
unfeasible for real-time applications. The response times of ARBITER prototype, while acceptable for
document-based queries, could impact user experience in interactive scenarios requiring immediate feedback.
Third, the 6-8% performance gap between our approach and static filtering still represents potential
security vulnerabilities in high-stakes environments. The trade-off between flexibility and precision may
prove insufficient for scenarios requiring absolute access control guarantees, such as classified document
systems or highly regulated financial environments where even minimal false negative rates are unacceptable.
Finally, our evaluation focuses on binary access decisions (allow/deny) and does not address complex
access control scenarios such as partial document redaction, conditional access based on temporal or
contextual factors, or graduated access levels. These limitations are directions for future work, but we believe
they validate the core premise that AI-driven access control represents a valid and open research question that
our work contributes to answering.
8

References
[1]Elena Ferrari. Role-based access control. InAccess Control in Data Management Systems, pages 61–75.
Springer, 1992.
[2]Ken Huang, Grace Huang, Adam Dawson, and Daniel Wu. Genai application level security. In
Generative AI Security: Theories and Practices, pages 199–237. Springer, 2024.
[3]Bargav Jayaraman, Virendra J Marathe, Hamid Mozaffari, William F Shen, and Krishnaram Kenthapadi.
Permissioned llms: Enforcing access control in large language models.arXiv preprint arXiv:2505.22860,
2025.
[4]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation
for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474,
2020.
[5]Zach Nussbaum, John X Morris, Brandon Duderstadt, and Andriy Mulyar. Nomic embed: Training a
reproducible long context text embedder.arXiv preprint arXiv:2402.01613, 2024.
[6]Parijat Rai, Saumil Sood, Vijay K Madisetti, and Arshdeep Bahga. Guardian: A multi-tiered defense
architecture for thwarting prompt injection attacks on llms.Journal of Software Engineering and
Applications, 17(1):43–68, 2024.
[7]Adib Bin Rashid and MD Ashfakul Karim Kausik. Ai revolutionizing industries worldwide: A
comprehensive overview of its diverse applications.Hybrid Advances, 7:100277, 2024. ISSN 2773-
207X. doi: https://doi.org/10.1016/j.hybadv.2024.100277. URL https://www.sciencedirect.com/
science/article/pii/S2773207X24001386.
[8]Traian Rebedea, Razvan Dinu, Makesh Narsimhan Sreedhar, Christopher Parisien, and Jonathan Cohen.
NeMo guardrails: A toolkit for controllable and safe LLM applications with programmable rails. In
Yansong Feng and Els Lefever, editors,Proceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing: System Demonstrations, pages 431–445, Singapore, December
2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-demo.40. URL
https://aclanthology.org/2023.emnlp-demo.40/.
[9]Ravi S. Sandhu. Role-based access control11portions of this chapter have been published earlier in
sandhu et al. (1996), sandhu (1996), sandhu and bhamidipati (1997), sandhu et al. (1997) and sandhu
and feinstein (1994). volume 46 ofAdvances in Computers, pages 237–286. Elsevier, 1998. doi:
https://doi.org/10.1016/S0065-2458(08)60206-5. URL https://www.sciencedirect.com/science/
article/pii/S0065245808602065.
[10] Oguzhan Topsakal and Tahir Cetin Akinci. Creating large language model applications utilizing
langchain: A primer on developing llm apps fast. InInternational Conference on Applied Engineering
and Natural Sciences, volume 1, pages 1050–1056, 2023.
[11] Jeff Vestal. Elasticsearch rag & rbac integration: Protect data and boost ai capabilities, 05 2024. URL
https://www.elastic.co/search-labs/blog/rag-and-rbac-integration.
[12] Yaqing Wang, Quanming Yao, James T Kwok, and Lionel M Ni. Generalizing from a few examples: A
survey on few-shot learning.ACM computing surveys (csur), 53(3):1–34, 2020.
9

[13] Liu Zhuang, Lin Wayne, Shi Ya, and Zhao Jun. A robustly optimized BERT pre-training approach
with post-training. In Sheng Li, Maosong Sun, Yang Liu, Hua Wu, Kang Liu, Wanxiang Che, Shizhu
He, and Gaoqi Rao, editors,Proceedings of the 20th Chinese National Conference on Computational
Linguistics, pages 1218–1227, Huhhot, China, August 2021. Chinese Information Processing Society of
China. URLhttps://aclanthology.org/2021.ccl-1.108/.
A Filter Prompts and Role Definitions
This appendix provides a complete prompt templates and role definitions used in our filtering mechanisms.
Figure 3 shows the fullInput Filterprompt template, showing the Chain-of-Thought reasoning approach
combined with few-shot examples to guide the LLM in making role-based access control decisions. The
prompt instructs the model to first determine which roles could legitimately ask a given question, then
compare against the user’s actual role to make the final access decision.
Figure 4 shows representative role definitions illustrating how natural language descriptions enable
flexible access control. These descriptions allow the system to understand role capabilities semantically
rather than through rigid keyword matching, supporting dynamic policy updates and natural language role
definitions.
The output filter employs similar prompt engineering but focuses on evaluating generated responses rather
than input queries. It uses similar Chain-of-Thought reasoning and role-aware context to determine whether a
generated response contains information that should be accessible to the user’s assigned role, providing a
complementary layer of protection in our multi-stage filtering architecture.
10

Input Filter Prompt Template
You are tasked with assigning one or more roles to user message below based on the given context and not the user’s
role. The context will include a brief description of various roles. If the message can be attributed to a specific role,
assign only that role. If the message is generic or could be asked by multiple roles, assign all relevant roles from the
context. After determining the roles for the message, you must decide if the message’s role(s) match the user’s role.
Follow these steps (for 1, 2, 3 and 4 don’t use the user’s role):
1) Analyze the message and determine which role(s) could be asking this based on the context provided and not the
user’s role.
2) If the message is specific, assign the most appropriate role based on the provided descriptions.
3) If the message is too broad or could apply to multiple roles, assign all relevant roles.
4) For ambiguous messages, select the most likely role(s).
5) Check if the "User’s role" below is included in the assigned roles.
6) The answer should be exactly "Answer: Yes." if the user’s role is among the assigned roles, or exactly "Answer:
No." if it is not, based solely on the comparison of roles. Do not include quotation marks in the response.
Context (Role Descriptions):
{{role_description}}
Examples:
•User Message:"How does Ravenous V oid works?"
Assigned Roles:Wizard
User’s role:Wizard
Answer:Yes. (Because the user’s role is among the assigned roles.)
•User Message:"Can I cast a spell as a bonus action and another spell as an action on the same turn?"
Assigned Roles:Artificer, Bard, Cleric, Druid, Paladin, Ranger, Sorcerer, Warlock, Wizard
User’s role:Bard
Answer:Yes. (Because the user’s role is among the assigned roles.)
•User Message:"If I cast Invisibility, what happens if I take damage or interact with objects?"
Assigned Roles:Wizard, Sorcerer, Bard
User’s role:Warlock
Answer:No. (Because the user’s role is not among the assigned roles.)
User’s role:"{{ user_role }}"
User message:"{{ user_input }}"
Figure 3: Complete prompt template used for the input filter guardrail, showing the Chain-of-Thought
reasoning approach and few-shot examples for role-based access control decisions.
Sample Role Definitions
Wizard:are the most versatile spellcasters, with a vast array of spells learned through study and discipline. At high
levels, they can access nearly any type of magic – offensive, defensive, or utility – making them capable of adapting to
any situation.
Cleric:are conduits of divine power, using their magic to heal, protect, and unleash holy wrath. At high levels, their
spells can resurrect the fallen, protect the party from harm and summon divine power to smite foes.
Ranger:are skilled hunters and survivalists, combining martial prowess with nature-based magic. At high levels, they
cast spells to enhance tracking, boost combat effectiveness, and harness nature’s power.
Figure 4: Example role definitions.
11