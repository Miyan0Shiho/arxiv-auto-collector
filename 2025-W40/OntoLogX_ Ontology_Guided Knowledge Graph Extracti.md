# OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models

**Authors**: Luca Cotti, Idilio Drago, Anisa Rula, Devis Bianchini, Federico Cerutti

**Published**: 2025-10-01 19:46:15

**PDF URL**: [http://arxiv.org/pdf/2510.01409v1](http://arxiv.org/pdf/2510.01409v1)

## Abstract
System logs represent a valuable source of Cyber Threat Intelligence (CTI),
capturing attacker behaviors, exploited vulnerabilities, and traces of
malicious activity. Yet their utility is often limited by lack of structure,
semantic inconsistency, and fragmentation across devices and sessions.
Extracting actionable CTI from logs therefore requires approaches that can
reconcile noisy, heterogeneous data into coherent and interoperable
representations. We introduce OntoLogX, an autonomous Artificial Intelligence
(AI) agent that leverages Large Language Models (LLMs) to transform raw logs
into ontology-grounded Knowledge Graphs (KGs). OntoLogX integrates a
lightweight log ontology with Retrieval Augmented Generation (RAG) and
iterative correction steps, ensuring that generated KGs are syntactically and
semantically valid. Beyond event-level analysis, the system aggregates KGs into
sessions and employs a LLM to predict MITRE ATT&CK tactics, linking low-level
log evidence to higher-level adversarial objectives. We evaluate OntoLogX on
both logs from a public benchmark and a real-world honeypot dataset,
demonstrating robust KG generation across multiple KGs backends and accurate
mapping of adversarial activity to ATT&CK tactics. Results highlight the
benefits of retrieval and correction for precision and recall, the
effectiveness of code-oriented models in structured log analysis, and the value
of ontology-grounded representations for actionable CTI extraction.

## Full Text


<!-- PDF content starts -->

OntoLogX: Ontology-Guided Knowledge Graph Extraction from
Cybersecurity Logs with Large Language Models
Luca Cotti1*, Idilio Drago2, Anisa Rula1, Devis Bianchini1, and Federico Cerutti1,3,4
1Department of Information Engineering, University of Brescia, Italy
2Department of Computer Science, University of Turin, Italy
3School of Computer Science and Informatics, Cardiff University, Cardiff, United Kingdom
4Department of Electronics and Computer Science, University of Southampton, Southampton, United Kingdom
ABSTRACT
System logs represent a valuable source of Cyber Threat Intelligence (CTI), capturing attacker be-
haviors, exploited vulnerabilities, and traces of malicious activity. Yet their utility is often limited by
lack of structure, semantic inconsistency, and fragmentation across devices and sessions. Extracting
actionable CTI from logs therefore requires approaches that can reconcile noisy, heterogeneous data
into coherent and interoperable representations. We introduce OntoLogX, an autonomous Artifi-
cial Intelligence (AI) agent that leverages Large Language Models (LLMs) to transform raw logs
into ontology-grounded Knowledge Graphs (KGs). OntoLogX integrates a lightweight log ontology
with Retrieval Augmented Generation (RAG) and iterative correction steps, ensuring that generated
KGs are syntactically and semantically valid. Beyond event-level analysis, the system aggregates
KGs into sessions and employs a LLM to predict MITRE ATT&CK tactics, linking low-level log
evidence to higher-level adversarial objectives. We evaluate OntoLogX on both logs from a public
benchmark and a real-world honeypot dataset, demonstrating robust KG generation across multiple
KGs backends and accurate mapping of adversarial activity to ATT&CK tactics. Results highlight
the benefits of retrieval and correction for precision and recall, the effectiveness of code-oriented
models in structured log analysis, and the value of ontology-grounded representations for actionable
CTI extraction.
KeywordsCyber Threat Intelligence, Computer Security, Knowledge Graphs, Ontologies, Large Language Models,
Autonomous Agents
1 Introduction
The rapid evolution and increasing sophistication of cyber threats pose significant risks to individuals, organizations,
and governments, as adversaries continually adapt their tactics to exploit vulnerabilities and evade detection [1–3].
Traditional reactive defenses, such as signature- or rule-based systems, often fail to keep pace with this dynamic
landscape, motivating a shift toward proactive and anticipatory strategies [4, 5].
CTI, defined as the collection, processing, and analysis of information about threat actors’ motives, targets, and attack
behaviors, supports faster and better-informed decisions in cybersecurity operations. By enabling a transition from
reactive to proactive defense, CTI strengthens the ability of organizations, governments, and individuals to anticipate
and mitigate attacks [6, 7]. Among the various sources of CTI, system logs are especially valuable, and particularly
so those produced by honeypot and honeynet deployments. Unlike operational logs dominated by benign activity,
honeypots are designed to attract and record malicious interactions, resulting in data with a higher concentration of
adversarial behavior [8].
Nevertheless, processing logs remains a challenging task. They are typically unstructured, syntactically heteroge-
neous, and often ambiguous in meaning, complicating automated analysis. Moreover, the information required to
reconstruct attack scenarios is frequently fragmented across multiple logs, which may be distributed over different
devices. Traditional rule-based or heuristic techniques lack the adaptability to generalize across diverse and evolving
threat behaviors.arXiv:2510.01409v1  [cs.AI]  1 Oct 2025

To address these challenges, recent work has explored the construction of KGs as a means to structure and enrich
CTI [9–12]. KGs describe the concepts, entities, and events of the objective world, as well as the relations between
them, and expresses knowledge in a form closer to that of human cognition, in comparison with other organizational
representations [13]. Such representations facilitate semantic reasoning, integration with automated workflows, and
support for advanced analytical tasks.
At the same time, advances in LLMs have shown remarkable effectiveness in extracting structured information from
natural language [14–16], fueling widespread adoption across diverse domains [17]. Yet, applications of LLMs in CTI
remain limited, with existing approaches relying on heavily pre-processed inputs or substantial user interaction [9,18].
In this work, we presentOntoLogX, an autonomous AI agent designed to extract CTI transparently from raw logs.
OntoLogX leverages an LLM to construct detailed KGs that capture both the structural and contextual aspects of log
events, without requiring user intervention. The generated graphs conform to a domain-specific ontology tailored for
cybersecurity logs, enabling semantic querying, traceability, and automated reasoning. Furthermore, we demonstrate
the effectiveness of the generated KGs in CTI extraction by mapping them to the MITRE ATT&CK framework,
through a final LLM classification step
We evaluate OntoLogX by extracting structured intelligence from both public log datasets, thus ensuring reproducibil-
ity and comparability, and a novel honeypot dataset representing real-world adversarial activity. Extracted intelligence
is stored in an ontology-enriched graph database, supporting semantic exploration and downstream CTI applications.
We conclude by outlining future directions of this work. While our methodology is effective at extracting CTI from
logs, it is only but an initial step towards a full-scale analysis tool.
2 Background
2.1 Large Language Models
LLMs are transformer-based models trained on large text corpora to perform a broad range of natural language un-
derstanding and generation tasks [19–21]. By learning probabilistic representations of language, they can complete,
summarize, translate, and interpret text across multiple domains, and have shown strong performance in zero-shot and
few-shot scenarios [16]. Widely adopted examples include GPT [22], Llama [23], Qwen [24], and Mistral [25].
Despite their versatility, LLMs do not inherently ensure factual consistency, structural coherence, or domain-specific
accuracy. Outputs may inherit biases from training data or lack sufficient grounding in external knowledge, which
is especially problematic in specialized domains such as cybersecurity, where precise terminology and contextual
interpretation are critical.
A common strategy to address these limitations is RAG, which combines language generation with information re-
trieval to enhance factual grounding [15]. In this paradigm, a retriever identifies documents relevant to a query from
a knowledge base, and the retrieved content is provided to the model as additional context. This improves factual
grounding and domain relevance, particularly in areas where training data is insufficient or outdated [14,26]. In cyber-
security, RAG has the potential to enhance information extraction from CTI sources, where background knowledge is
often necessary to interpret incomplete or ambiguous entries.
2.2 AI Agents
AI agents are autonomous software entities engineered to perform goal-directed tasks within bounded digital environ-
ments [27, 28]. While the concept has gained renewed popularity with the rise of LLMs, definitions of autonomous
agents date back decades [29, 30]. An (autonomous) agent is typically understood as a system situated in an environ-
ment, capable of perceiving and acting on it over time in pursuit of defined objectives, and so as to effect what it senses
in the future [30].
AI agents are generally characterized by three properties [31]: (i)autonomy, the ability to operate with minimal human
intervention; (ii)task-specificity, a focus on narrow, well-defined goals; and (iii)reactivity and adaptability, allowing
them to respond to real-time inputs, learn from interactions, and adjust their behavior. These traits distinguish agents
from deterministic automation scripts, which follow rigid workflows, as well as from stand-alone LLMs, which mainly
act as reactive prompt followers.
2

2.3 Cybersecurity Ontologies
Ontologies are formal, explicit specifications of shared conceptualizations, encompassing classes, relationships, and
constraints within a domain [32]. They often serve as schemas guiding the construction of KGs, ensuring consistency
and semantic clarity.
The quality and compliance of ontology-based KGs can be assessed through Shapes Constraint Language
(SHACL) [33]. SHACL provides a declarative framework for defining and enforcing constraints on types, prop-
erty cardinalities, and relationship patterns. Such validation ensures that automatically generated knowledge remains
consistent, explainable, and auditable, even when derived from noisy or incomplete data [34,35]. This in turn enhances
the robustness of downstream CTI tasks.
In cybersecurity, ontology-based frameworks are increasingly adopted to organize and standardize threat-related
knowledge, enabling semantic interoperability, automated reasoning, and improved information integration across
heterogeneous sources [36]. Several well-known ontologies exist, each targeting different needs withing CTI analy-
sis. Unified Cyber Ontology (UCO) [36] provides comprehensive concepts and relationships for broad cybersecurity
knowledge integration. Structured Threat Information Expression (STIX) [37] defines a widely used standard for
threat intelligence exchange. CRATELO [38] focuses on cyber incident and forensic data. Malware Information
Sharing Platform (MISP) [39] facilitates structured sharing of malware and threat indicators. Particularly relevant to
this work is the SEPSES ontology [40], which provides a vocabulary for integrating already-parsed logs but does not
support information extraction from free-text messages or interaction with language models.
2.4 Related Work
A range of methods have been proposed for analyzing logs and extracting CTI.
Log parsing, defined as the process of dividing logs into static parts (static messages) and dynamic parts (vari-
ables) [41], is often used as a preliminary step. While efficient and capable of online processing, parsing alone
does not impose a standardized structure, limiting subsequent reasoning. SLOGERT [42] extends log parsing by con-
structing a KG in RDF based on a custom ontology, enabling continuous integration of parsed logs into an explorable,
queryable graph that integrates multiple log sources. KRYSTAL [43] builds on this approach by combining declara-
tive SPARQL queries with backward-forward chaining to detect attack patterns, outputting attack graphs aligned with
MITRE ATT&CK tactics and techniques. However, we argue that the variability of log formats across applications or
even across versions of the same system limits the effectiveness of rule-based parsing for CTI extraction.
More recent work has explored the use of language models. LogPr ´ecis [44] fine-tunes models on small sets of labeled
attacks to generate attack fingerprints aligned with MITRE ATT&CK, reducing large volumes of logs into more com-
pact and interpretable patterns. However, LogPr ´ecis lacks semantic grounding and requires pre-processed log sessions
rather than operating directly on raw logs. CyKG-RAG [45] combines rule-based and LLM-based approaches to con-
struct KGs from cybersecurity data, and integrates symbolic queries with vector similarity search for hybrid retrieval.
While effective for synthesizing responses to user queries, it still depends on rule-based steps for KG construction and
does not perform autonomous log analysis.
3 Methodology
In this section we detail the design of OntoLogX, explaining its components, the way they interact, and the design of
the underlying ontology used to structure the knowledge extracted from logs.
3.1 Overview
OntoLogX is an AI agent for online log analysis, designed to process events incrementally and sequentially, one by
one, in a setting that reflects realistic cybersecurity use cases requiring near real-time event analysis. Each log event is
analyzed with the support of an LLM, which produces an ontology-grounded KG representation. Optional context in-
formation (e.g., device, process, operating system, honeypot version) can be provided to enrich the generation process,
even if unstructured.
Figure 1 illustrates the overall workflow. When a log event arrives, the system first retrieves semantically related log
event KGs from the graph database. These serve as few-shot examples to help the LLM adapt its output to the ontology
and to previously seen patterns. The LLM then generates a candidate KG by combining the new log event, the optional
context, and the domain ontology. The candidate is validated against ontology constraints: if the output is malformed
or non-compliant, the model is prompted again within the same interaction to apply targeted corrections. This iterative
3

g r a p h :
  n o d e s :  l i s t [
    i d
    t y p e
    p r o p e r t i e s
  ]
  r e l a t i o n s h i p s :  l i s t [
    s o u r c e _ i d
    t a r g e t _ i d
    t y p e
  ]S t r u c t u r e d  o u t p u t  f o r m a t
L o g  o n t o l o g y
V e c t o r  i n d e xJ a n  2 3  0 8 : 0 6 : 1 9  m a i l _ s e r v e r  d o v e c o t :
  i m a p ( t r a c i . s t e v e n s o n ) :  L o g g e d  o u t
  i n = 7 0  o u t = 5 9 9L o g  e v e n td e v i c e :  m a i l _ s e r v e r
a p p l i c a t i o n :  d o v e c o t
. . .C o n t e x t  i n f o  ( o p t i o n a l )
E x a m p l e s  R e t r i e v a lR e l e v a n t  e x a m p l e sG e n e r a t i o nR e l e v a n t
e x a m p l e sC o r r e c t i o nC a n d i d a t e
g r a p hC o r r e c t i o n
p r o m p tG r a p h  d a t a b a s e
G e n e r a t e d  g r a p hT a c t i c s  p r e d i c t i o nG r a p h  s e s s i o nFigure 1: Methodology for generating a log event KG, starting from the raw log event and optional context information.
P a r a m e t e rU s e r
A p p l i c a t i o nS o u r c eN e t w o r k  A d d r e s sU R LF i l et i m e : I n s t a n t
S y s t e m  P r o c e s sT i m e s t a m pE v e n tw a s L o g g e d B yh a s P a r a m e t e r
S h e l l  C o m m a n dp r o v : E n t i t yp r o v : A g e n t
H a s h S t r i n gN e t w o r k  P r o t o c o lU s e r  C r e d e n t i a lh a s C r e d e n t i a lU s e r  N a m eU s e r  E m a i lU s e r  P a s s w o r dh a s P a r a m e t e rp r o v : w a s A t t r i b u t e d T o
Figure 2: Classes and object properties of the OntoLogX ontology. Data properties are omitted for conciseness.
Full arrows indicate eitherrdfs:subClassOforrdf:subPropertyOfobject properties. Colored boxes highlight
external classes
refinement continues until a valid representation is obtained. Once validated, the KG is stored independently in the
graph database, ensuring that it can be retrieved for future processing without requiring immediate integration with
other graphs. Finally, the generated KGs are grouped depending on the log session the originate from, and each session
is used to predict associated MITRE ATT&CK tactics labels through an LLM call.
It is worth noting that KGs are stored independently from each other, as the semantic connection of different KGs is
not the focus of this work.
The primary limitations of this approach are the significant computational resources required to run LLMs, both in
terms of execution time and cost. We acknowledge these constraints and argue that the execution times can be managed
through careful engineering of the resource orchestration, while future advancements in LLM technology are expected
to further mitigate these challenges.
3.2 Log Ontology
The KGs generated by OntoLogX are grounded in a custom log ontology that formalizes information extracted from
raw events. Beyond providing structure, the ontology guides the LLM during extraction by indicating which elements
4

2022-01-21 03:49:44 jhall/192.168.230.165:46011 VERIFY OK: CN=OpenVPN CA
{"eventid": "cowrie.client.version", "message": "Remote SSH version: SSH-2.0-Go", "sensor":
"cowrie-XXX", "src_ip": "192.168.1.1", "timestamp": "2025-08-10T22:21:57.894158Z", "version":
"SSH-2.0-Go"}
Table 1: Examples of an unstructured and a structured log.
J a n  2 3  0 8 : 0 6 : 1 9  m a i l _ s e r v e r  d o v e c o t :
  i m a p ( t r a c i . s t e v e n s o n ) :  L o g g e d  o u t
  i n = 7 0  o u t = 5 9 9L o g  e v e n td e v i c e :  m a i l _ s e r v e r
a p p l i c a t i o n :  d o v e c o t
. . .C o n t e x t  i n f o  ( o p t i o n a l )V e c t o r  s e a r c h
F u l l - t e x t  s e a r c hN o r m a l i z e ,  c o m b i n e  a n d  
r e o r d e r  b y  s c o r eG r a p h  d a t a b a s eM a x i m a l  M a r g i n a l  
R e l e v a n c e  r e r a n k i n g
R e t r i e v e d  g r a p h s
Figure 3: Hybrid retrieval process.
to identify in each log. This is particularly important because log entries, whether structured or unstructured, often
contain significant information but may encode them without consistency or explicit separation. Table 1) reports
examples for both types of logs, which capture information such as timestamp, user identity, network address, and
certificate details.
We developed a custom ontology because we argue that existing cybersecurity ontologies are not well suited for LLM-
based log processing. Minimal models, such as the one in SLOGERT’s, capture too few concepts to be useful for CTI
analysis. In contrast, large frameworks like UCO are overly complex for automated generation: their size increases
the likelihood of errors, and they assume pre-parsed or structured metadata, which is rarely available in raw logs. To
balance these issues, OntoLogX adopts a lightweight but expressive ontology tailored to log characteristics. Using a
predefined ontology also ensures formalization and consistency, while the methodology itself remains flexible: thanks
to the LLM-driven pipeline, the ontology can be swapped or extended depending on the user’s needs.
The resulting model, shown in Figure 2, is designed to capture the most common concepts in cybersecurity logs without
being rigid or monolithic. At its core is theEventclass, which represents a single log entry. Each event is linked to a
Source, describing the device or application that produced the log. These two classes are mapped to theEntityand
Agentclasses in theprov-oontology, aligning them with provenance standards. The information contained withing
the logs themselves are represented as subclasses ofParameter, including a dedicatedTimeStampparameter aligned
with the W3Ctimeontology. More complex structures are also supported: theApplicationparameter can reference
other parameters, enabling the modeling of application call arguments or chains of calls.UserCredentialmodels
various credentials that aUsercan have, with specialized classes available for username, email, and password.
To ensure quality and compliance, we developed a companion SHACL specification. Constraints enforce schema
validity by checking property cardinalities, type consistency, and the presence of required fields. This validation step
is essential in an LLM-based pipeline, where outputs may otherwise be incomplete or inconsistent. Together, the
ontology and its constraints guarantee that generated KGs remain semantically coherent, queryable, and interoperable
with broader CTI frameworks.
3.3 Examples Retrieval
OntoLogX incorporates an example retrieval step, which guides the LLM in constructing KGs and enables reuse of
knowledge from related logs. The objective is to identify semantically and textually similar log entries that can serve
as few-shot prompts, thereby improving both the structure and consistency of the generated graphs. Examples are
drawn from a dedicated store indexing previously generated KGs as well as manually crafted instances aligned with
the log ontology.
As illustrated in Figure 3, retrieval is performed through a hybrid strategy combining vector and full-text search. The
input log and its context are queried against both indices, allowing for semantic similarity matching and precise word-
based lookups. The vector index stores embeddings of the raw log event and context information used to generate
the graph. The full-text index instead contains the individual words of the log event and context. Using only one of
5

N o d ei d :    s t r i n g
t y p e :  N o d e T y p e
R e l a t i o n s h i pi d :         s t r i n g
t y p e :       R e l a t i o n s h i p T y p e
s o u r c e _ i d :  s t r i n g
t a r g e t _ i d :  s t r i n gA  r e l a t i o n s h i p
h a s  s o u r c e  a n d
d e s t i n a t i o n  n o d e sP r o p e r t yt y p e :   P r o p e r t y T y p e
v a l u e :  s t r i n gG r a p hn o d e s :          l i s t [ N o d e ]
r e l a t i o n s h i p s :  l i s t [ R e l a t i o n s h i p ]A  g r a p h  h a s
m a n y  n o d e sA  n o d e  h a s
m a n y  p r o p e r t i e s
A  g r a p h  h a s  m a n y
r e l a t i o n s h i p sR e l a t i o n s h i p T y p eN o d e T y p e
P r o p e r t y T y p eFigure 4: Format of structured output.NodeType,PropertyType, andRelationshipTyperespectively represent
the valid classes, data properties and object properties defined in the ontology.
these approaches would be insufficient: full-text search alone misses semantic nuances that help the model capture
hidden relationships, while vector search alone risks overlooking near-identical matches, which are often the most
useful as generation examples. The results from both searches are normalized onto a common score scale, merged,
and re-ranked to produce the final candidate set.
To further improve retrieval quality, OntoLogX employs Maximal Marginal Relevance (MMR) [46], a re-ranking strat-
egy that balancesrelevancewithdiversity. Rather than returning only the top-kmost similar items, MMR penalizes
redundancy by favoring candidates that are both close to the query and dissimilar from each other. Given a queryq,
candidate setD, and already selected itemsS⊂D, the next exampled∗is chosen by maximizing:
MMR(d) =λ·Sim(d, q)−(1−λ)·max
s∈SSim(d, s)
where Sim(d, q)measures the similarity between documentdand queryq, Sim(d, s)measures the similarity between
dand already selected documents, andλ∈[0,1]controls the trade-off between relevance and diversity.
3.4 Generation
In the generation step, OntoLogX employs a LLM to produce a KG from each log event and its associated context.
The model is also guided by the relevant KGs retrieved in the previous step, which serve as few-shot examples. This
combined input both anchors the model to the desired output structure and provides historical context from seman-
tically and textually related logs, improving the coherence and utility of the generated graphs. The prompt used to
instruct the LLM (Table 2) is designed to be model-agnostic, maximizing compatibility with a broad range of language
models. It comprises: (i) a clear role and task definition, (ii) detailed instructions grounded in the OntoLogX ontology,
and (iii) a set of constraints and clarifications based on common failure patterns observed in early experiments, such as
malformed URIs, incorrect predicate usage, and mismatched types. The expected ontology format is enforced through
the use of the structured output schema shown in Figure 4. This format defines the required fields and their expected
types, effectively serving as a strong constraint during generation and simplifying validation.
A key advantage of using a LLM is its ability to generate high-quality KGs without requiring domain-specific su-
pervised training. By leveraging knowledge gained during pretraining, the model can infer implicit information,
disambiguate vague or underspecified log entries, and normalize inconsistent terminology. These capabilities are par-
ticularly useful in cybersecurity logs, where entities and activities are often expressed in non-standard, abbreviated,
or noisy forms. The generalization afforded by pretrained LLMs thus enables the system to handle heterogeneous
sources and adapt to evolving logging formats.
3.5 Correction
A central challenge in using LLMs for ontology-grounded KG generation is the risk of producing outputs that are
incomplete, malformed, or semantically inconsistent with the intended schema. To mitigate this, OntoLogX introduces
a dedicatedcorrectionphase, where generated graphs are automatically validated and, if necessary, revised through
iterative feedback with the model.
6

The correction pipeline proceeds in three stages. First, the syntactic validity of the output is checked to ensure that
nodes and relationships are properly defined and the graph conforms to the required structured format. Second, on-
tology compliance is verified by enforcing the constraints specified in the SHACL rules. This step covers correct
class and property usage, consistent data typing, and satisfaction of required schema rules. Third, semantic validation
is performed to detect higher-level inconsistencies, such as: (i) the absence of anEventnode, (ii) the presence of
multipleEventnodes, (iii) relationships pointing to undefined entities, or (iv) duplicate node definitions.
When violations are identified, OntoLogX constructs a targeted correction prompt that highlights the errors and re-
quests specific revisions from the LLM. This feedback loop can iterate across multiple rounds, progressively refining
the output until a fully valid and ontology-compliant KG is obtained. If a valid graph cannot be produced within the
allowed attempts, the system defaults to outputting an empty graph, which negatively impacts evaluation but ensures
that invalid results do not contaminate the knowledge base.
Once a corrected KG passes all validation stages, it is persisted in the graph database along with the originating log
event and any contextual metadata, enabling future retrieval, and traceability should further analysis be required.
3.6 Tactics Prediction
The final stage of the OntoLogX pipeline focuses on predicting the MITRE ATT&CK tactics associated with the
input logs. This step provides higher-level semantic insights inferred across multiple, related events, and also helps in
contextualizing log activity within a widely adopted threat intelligence framework.
The prediction process begins once individual log KGs have been generated and validated. These graphs are first
grouped into sessions, capturing sets of temporally and contextually related events. An LLM is then prompted to
analyze the aggregated KGs and assign one or more MITRE ATT&CK tactics that best describe the observed behavior.
The prompt used for this task is reported in Table 4.
Using KGs rather than raw logs offers several advantages for this task. First, KGs provide a normalized, ontology-
compliant representation that abstracts away the heterogeneity of log formats, allowing the model to focus on entities,
relationships, and temporal structure rather than noisy or inconsistent syntax. Second, the explicit structuring of events
into semantic components facilitates reasoning about higher-level attack stages, making it easier to identify attacker
behaviors. In contrast, relying directly on raw logs would require the model to both parse and interpret the event
simultaneously, increasing the likelihood of errors and reducing generalizability across log sources.
4 Experiments
We organize the experimental evaluation of OntoLogX into two parts. First, we evaluate the quality of KG generation,
conducting an ablation study to assess the contribution of individual components of the pipeline (retrieval, correction,
structured output) and comparing the performance of different language models. Second, we examine the effectiveness
of tactics prediction, leveraging a cybersecurity-oriented LLM to map log sessions to MITRE ATT&CK tactics.
The implementation used in these experiments is designed to be general and model-agnostic, ensuring that the method-
ology does not depend on a specific LLM. For KG generation, structured output is enforced through function-calling
interfaces, which constrain the LLM to produce results in a predefined format. The correction phase is limited to a
maximum of three refinement prompts per log event; if no valid ontology-compliant graph is produced within this
limit, the output is considered an empty graph.
All validated KGs are stored in a Neo4j1graph database, extended with a vector index to support semantic and full-
text retrieval over node property embeddings. For embedding generation, we adopt thegte-multilingual-base
model [47]. To facilitate reproducibility and enable structured sharing of experimental results, the database is further
organized following the MLSchema ontology [48].
4.1 Knowledge Graph Generation
The first experiment evaluates the effectiveness of OntoLogX in generating ontology-compliant KGs from raw log
events. The study is designed as an ablation analysis, aimed at quantifying the contribution of each major component
of the pipeline and at assessing the impact of different LLMs. Five configurations are considered: (i) a baseline
without retrieval, structured output, or correction; (ii) retrieval only; (iii) structured output only; (iv) structured output
with corrections; and (v) the full OntoLogX pipeline. This comparison isolates the incremental benefits of RAG,
output structuring, and iterative correction mechanisms.
1https://neo4j.com/(on September 10, 2025).
7

4.1.1 Models
We evaluate OntoLogX using a set of eight LLMs, selected to span different architectures, parameter scales, and
licenses. The models considered along with their number of parameters are: Llama 3.3 (80B), Llama 3.1 (8B),
Claude Sonnet 4, Claude 3.5 Haiku, Mistral Large (123B), gpt-oss (20B and 120B), and Qwen3 Coder (32B). With
the exception of the Claude family, all models are distributed as open weights, making them deployable outside
of proprietary cloud environments. The Qwen3 Coder model represents a class of code-specialized LLMs, trained
primarily on source code and related artifacts. Such models are designed to excel at generating syntactically precise
and semantically consistent outputs, making them particularly well-suited to tasks involving structured representations
like knowledge graphs. gpt-oss, by contrast, belongs to a family of reasoning-oriented models that emphasize logical
inference and multi-step problem solving.
All models were accessed through AWS Bedrock2, with the exception of Qwen 3 Coder, which was executed locally
via vLLM [49] on a system equipped with four NVIDIA L4 GPUs. A temperature of 0.7 was applied uniformly across
runs to encourage creative reasoning, which we hypothesize aids the inference of implicit knowledge from log data.
To mitigate stochastic variability from this setting, each experiment was repeated ten times.
4.1.2 Dataset
ALGORITHM 1:Pseudocode for the dataset generation procedure.
1events←[] ;
2foreachlog fileinRussellMitchelldo
3fori←0to100do
4events.append(log file[i]);
5end
6end
7dataset← {};
8whilelength(dataset)<70do
9e←randomly sample fromevents;
10emb←compute embedding ofevent;
11is unique←True;
12foreache’indatasetdo
13emb’←compute embedding ofe’;
14ifcosine distance(emb, emb’)<0.7then
15is unique←False;
16break;
17end
18end
19ifis uniquethen
20dataset.add(event);
21end
22end
The evaluation dataset consists of log events sampled from the AIT-LDS corpus [50], a comprehensive benchmark
for log analysis. A total of 70 log entries were selected to ensure both syntactic and semantic diversity. To promote
heterogeneity, the first 100 events were extracted from each file in theRussellMitchelltestbed. From this pool, 70
entries were chosen using an embedding-based dissimilarity criterion. Specifically, embeddings were computed using
thenomic-embed-text-v1.5model, and cosine distances were calculated with respect to previously selected entries.
Candidates with a minimum distance below 0.7—indicating excessive similarity—were discarded in favor of more
diverse samples. Each selected log event was manually annotated with a gold-standard KG, ensuring a reliable basis
for evaluation. Finally, the dataset was randomly partitioned into three subsets: 10 examples reserved for few-shot
prompting, 10 for validation during prompt refinement, and 50 for testing.
4.1.3 Metrics
To assess the quality of KGs generated by OntoLogX, we combine ontological validation with semantic evaluation
using LLM-based scoring:
2https://aws.amazon.com/bedrock/(on September 10, 2025).
8

Baseline Retrieval
onlyStr.
output
onlyStr.
output and
corr.Starter
set
retrievalFull
retrieval
Experiment0.00.20.40.60.81.0ScoreMetric
G-Eval Score
F1 ScoreFigure 5: Comparison of G-Eval scores across different configurations using theQwen3 Coder 32Bmodel.
•Generation Success Ratio: proportion of log inputs for which a non-empty KG was successfully generated.
This metric captures the overall robustness of the pipeline.
•SHACL Violation Ratio: proportion of SHACL constraints violated across generated graphs. Lower values
indicate stronger adherence to the ontology’s formal rules.
•Precision: fraction of generated triples that are correct, i.e., present in the ground-truth KG, over the total
number of generated triples. High precision reflects accurate extractions with few spurious facts.
•Recall: fraction of ground-truth triples that were successfully generated, i.e., matched in the output. High
recall reflects comprehensive coverage of the relevant information in the logs.
•F1 Score: harmonic mean between precision and recall.
•Entity Linking Accuracy: percentage of correctly generated entities, defined as class instances along with
their associated properties.
•Relationship Linking Accuracy: percentage of correctly generated relationships between entities that are
themselves correct.
•G-Eval Score: a LLM-as-a-judge framework [51] that employs chain-of-thought reasoning and a form-filling
paradigm to evaluate natural language generation outputs. In our setting, it uses theLlama 3.3model to
assess the semantic fidelity of generated graphs. The evaluator is prompted (see Table 5) to produce natural
language summaries of both the raw log and the KG, and then to score their semantic overlap on a scale from
0 to 1. Additional information in the KG reduces the score only if deemed irrelevant. Due to the inherent
noisiness of logs, a higher score is not necessarily an indicator of quality: we expect that the ideal score is
between 0.7–0.8, which indicates high information retain without the noise.
4.1.4 Results
The results of the knowledge graph generation experiments are summarized in Table 6 and illustrated in Figure 6.
Overall, OntoLogX proves effective in producing ontology-compliant KGs, though the impact of individual compo-
nents varies significantly across models and configurations.
A first observation is that SHACL violation ratios are consistently low across all setups (Figure 6b), confirming the
effectiveness of the correction phase in enforcing structural validity. More complete configurations achieve an order
of magnitude fewer violations compared to minimal baselines, highlighting the benefit of layering constraints.
Execution times varied substantially across models due to differences in backends. Within the same model, however,
the baseline configuration was consistently the fastest, reflecting its shorter prompts and reduced context length.
When considering extraction quality, the full retrieval and starter-set retrieval variants achieve the highest precision,
recall, F1 scores, entity linking accuracy, and relationship linking accuracy (Figure 6c, Figure 6e, Figure 6f). Their
performances are nearly indistinguishable, which we attribute to the relatively small and diverse dataset: the selection
procedure ensured that the chosen logs were semantically different, reducing the advantage of a retrieval mechanism
that thrives on redundancy. Nevertheless, the full retrieval configuration particularly benefits when similar logs already
9

exist in the database, making it better suited for realistic deployment scenarios such as honeypots, where large volumes
of nearly identical events are common. By contrast, in small datasets such as this experiment, high-quality manually
annotated starter examples remain highly valuable. Retrieval-only also yields competitive results, particularly for
models with weaker structured output capabilities. By contrast, structured output alone performs poorly across most
models, often failing to call the output tool correctly or producing malformed graphs. The addition of corrections
mitigates this issue to some extent, but remains insufficient in isolation. This suggests that retrieval provides a stronger
inductive signal than syntactic scaffolding when used alone.
An interesting divergence emerges when comparing ontological metrics with semantic ones. As shown in Figure 5,
bare-bones methods such as structured output without retrieval achieve very high G-Eval scores, peaking at 0.912 for
theQwen3 Coder 32Bmodel, despite low F1 scores. This indicates that the LLM captures substantial information
from the raw logs, but often introduces noise that lowers precision while still producing ontology-compliant graphs.
Conversely, configurations with higher F1 scores stabilize around a G-Eval score of 0.8, suggesting a trade-off between
semantic coverage and ontological rigor.
Across all models, Claude Sonnet 4 achieves the strongest overall results, in the full retrieval and starter set retrieval
configuration. This confirms its robustness in log-to-KG translation. Notably,Qwen3 Coder 32Bemerges as an
impressive open-weights alternative, delivering competitive results despite its lighter footprint. These findings sug-
gest that code-oriented models are particularly well suited for OntoLogX, likely due to their stronger handling of
structured outputs and symbolic reasoning. By contrast, thegpt-ossreasoning models underperform across all met-
rics. Their weaker results suggest that the uniform prompting strategy used in this work is not well aligned with
their capabilities, and that they may require alternative prompting or training paradigms. This highlights a broader
limitation: OntoLogX, as currently designed, benefits most from general-purpose or code-specific models, while
reasoning-specialized architectures may demand tailored integration strategies.
4.2 MITRE ATT&CK Tactics Prediction
To evaluate the final step of the OntoLogX pipeline, we conducted experiments on the prediction of MITRE ATT&CK
tactics from log events. For these experiments, we adopted thefull retrievalconfiguration, which had previously
demonstrated the best trade-off between precision, recall, and semantic fidelity in the knowledge graph generation
task. This ensured that the model operated on high-quality, ontology-compliant KGs enriched with contextual exam-
ples. As the generation LLM backend, we selected Claude Sonnet 4, which achieved the strongest overall perfor-
mance in the generation experiments. For the tactics prediction using the generated graph sessions we instead use
Foundation-sec-8b, a cybersecurity-specific LLM, running onvLLMon the same machine as above.
4.2.1 Dataset
For the tactics prediction experiments, we relied on real-world data collected by the Politecnico di Torino through
the deployment of the Cowrie honeypot3. Cowrie is a widely used medium-interaction honeypot that emulates SSH
and Telnet services, thereby attracting attackers who attempt to exploit exposed credentials or misconfigured servers.
Once connected, adversaries can execute commands within the simulated environment, allowing the capture of both
interactive behavior and system-level logging. In our deployment, the honeypot was publicly exposed on the Internet,
enabling unsolicited traffic from scanning bots and opportunistic attackers to be recorded without active intervention.
The dataset used in our experiments was collected over a ten-day window, from August 4, 2025 to August 14, 2025.
During this period, the honeypot registered a high volume of automated activity, as evidenced by the prevalence of
repeated logs generated by discovery scripts and simple brute-force tools. Such redundancy is typical of large-scale
botnet activity and provides a realistic context for assessing the robustness of OntoLogX.
To enable tactics prediction, logs were grouped into sessions. Each session aggregates both the attacker’s direct
actions (e.g., executed commands) and accompanying meta-logs (e.g., connection attempts, authentication successes
or failures). On average, a session contained approximately ten logs, capturing a short but coherent sequence of
adversarial behavior. The dataset was partitioned into one session for training, two sessions for validation, and 100
sessions for testing. Finally, all test sessions were manually annotated with their corresponding MITRE ATT&CK
tactics, providing the ground truth against which model predictions were evaluated.
4.2.2 Metrics
For each MITRE ATT&CK tactic that appears either in the ground-truth or predicted set, we compute:
3https://www.cowrie.org/(on September 10, 2025).
10

(a) Generation success ratio, higher is better.
Claude
Sonnet 4Qwen3
Coder 32BLlama 3.3 Claude 3.5
HaikuMistral
LargeLlama 3.1 *gpt-oss
120B*gpt-oss
20B
Model0.00.20.40.60.81.0Generation Success Ratio
Baseline
Starter set retrieval
Full retrieval (b) SHACL violation ratio, lower is better.
0.70.80.91.0
Claude
Sonnet 4Qwen3
Coder 32BLlama 3.3 Claude 3.5
HaikuMistral
LargeLlama 3.1 *gpt-oss
120B*gpt-oss
20B
Model0.0000.0050.0100.0150.0200.0250.0300.0350.040SHACL Violation Ratio
(c) F1 score, higher is better.
Claude
Sonnet 4Qwen3
Coder 32BLlama 3.3 Claude 3.5
HaikuMistral
LargeLlama 3.1 *gpt-oss
120B*gpt-oss
20B
Model0.00.20.40.60.81.0F1 Score
Baseline
Starter set retrieval
Full retrieval (d) G-Eval Score.
Claude
Sonnet 4Qwen3
Coder 32BLlama 3.3 Claude 3.5
HaikuMistral
LargeLlama 3.1 *gpt-oss
120B*gpt-oss
20B
Model0.00.20.40.60.81.0G-Eval Score
(e) Entity linking accuracy, higher is better.
Claude
Sonnet 4Qwen3
Coder 32BLlama 3.3 Claude 3.5
HaikuMistral
LargeLlama 3.1 *gpt-oss
120B*gpt-oss
20B
Model0.00.20.40.60.81.0Entity Linking Accuracy
Baseline
Starter set retrieval
Full retrieval (f) Relationship linking accuracy, higher is better.
Claude
Sonnet 4Qwen3
Coder 32BLlama 3.3 Claude 3.5
HaikuMistral
LargeLlama 3.1 *gpt-oss
120B*gpt-oss
20B
Model0.00.20.40.60.81.0Relationship Linking Accuracy
Figure 6: Comparison of metrics between techniques. Reasoning models are highlighted with an asterisk before their
name.
11

0.00.20.40.60.81.0
ScoreInitial Access
Execution
Credential Access
Persistence
Discovery
Defense EvasionTacticTactics Results
Metric
Precision
Recall
F1 scoreFigure 7: Results of tactics evaluation over generated graphs.
•Precision: fraction of identified tactic labels that are correct, i.e. present in the ground truth tactics for that
particular session.
•Recall: fraction of ground-truth tactic labels that were correctly identified, i.e. matched in the output.
•F1 score: harmonic mean between precision and recall.
4.2.3 Results
The results for relevant MITRE ATT&CK tactics are reported in Figure 7. Overall, OntoLogX is effective in extracting
CTI from the generated graphs, with some tactics being successfully identified in the vast majority of cases.
5 Conclusions
In this work, we introduced OntoLogX, an ontology-guided AI agent that leverages large language models for the
extraction of CTI from raw system logs. By integrating a lightweight log ontology with retrieval-augmented generation
and iterative correction steps, OntoLogX produces syntactically and semantically valid knowledge graphs that capture
attacker behaviors, contextual information, and higher-level adversarial objectives.
Our evaluation on both benchmark datasets and a real-world honeypot deployment demonstrates that OntoLogX is
effective at generating ontology-compliant knowledge graphs, with retrieval and correction mechanisms substantially
improving precision and recall. Moreover, the use of code-oriented models proved particularly advantageous for
structured log analysis, highlighting the importance of model selection in CTI applications. The system further enables
the mapping of log sessions to MITRE ATT&CK tactics, bridging the gap between low-level evidence and high-level
threat modeling.
While the approach shows strong promise, several challenges remain. The reliance on computationally expensive
LLMs may limit scalability in high-throughput environments, and future work will explore optimization strategies and
incremental learning techniques to mitigate these costs. Furthermore, although our ontology provides a flexible and
expressive foundation, extending it to cover additional log sources and CTI standards represents an important next step
toward interoperability at scale.
Overall, OntoLogX contributes a novel methodology for transforming unstructured and heterogeneous logs into ac-
tionable intelligence. By combining ontology-driven structuring with the generative capabilities of LLMs, it advances
the state of the art in automated CTI extraction and opens new opportunities for proactive and explainable cyber
defense.
Acknowledgements
This work was partially supported by project SERICS (PE00000014) under the MUR National Recovery and Re-
silience Plan funded by the European Union – NextGenerationEU, specifically by the project NEACD: Neurosym-
bolic Enhanced Active Cyber Defence (CUP J33C22002810001). This project was also partially funded by the Italian
12

# Overview
You are a top-tier cybersecurity expert specialized in extracting structured information from
unstructured data to construct a knowledge graph according to a predefined "olx" ontology. You will
be provided with a log event, optionally accompanied by contextual information.
Your goal is to maximize information extraction from the event while maintaining absolute accuracy.
Leverage both the contextual information and your knowledge of computer systems and cybersecurity
to infer additional insights where possible. The objective is to achieve completeness in the
knowledge graph while remaining strictly ontology-compliant.
# Rules
You MUST adhere to the following constraints at all times:
- The graph must contain exactly one "Event" node.
- Use only the available types as defined in the ontology, without introducing new ones.
- Use the most specific type available for nodes and relationships, e.g. "UserPassword" instead of
"UserCredential".
- Respect the appropriate casing for all types.
- Use the appropriate node prefix for properties, e.g. "userUID" instead of "uid".
- Omit properties with empty values.
- Use the most specific type available for nodes and relationships.
- Respect the structural relationships to infer properties and relationships allowed by the
ontology for each node type.
- The graph must be connected: every node must be reachable from the "Event" node.
# Strict Compliance
Adhere to these rules strictly. Any deviation will result in termination.
Table 2: Prompt for log event KG generation, used in conjunction with structured output.
Ministry of University as part of the PRIN: PROGETTI DI RICERCA DI RILEV ANTE INTERESSE NAZIONALE
– Bando 2022, Prot. 2022EP2L7H. This paper was presented in part at the 1st International Workshop on eXplainable
AI, Knowledge Representation and Knowledge Graphs (XAI-KRKG), Bologna, Italy, October 2025.
Technical Appendix
5.1 Prompts
The various prompts used to invoke LLMs, either for generation or evaluation, are reported in Tables 2 to 5.
5.2 Full Experiment Results
The full results of the experiments conducted on the AIT-LDS dataset are reported in Table 6. The table includes
the run total time, which consists in the the number of seconds used to generate the graphs starting from the raw log
events, for the whole test dataset. These values are not directly comparable with each other, due to differences in the
backends used to invoke the LLMs.
References
[1] Y . Li and Q. Liu, “A comprehensive review study of cyber-attacks and cyber security; Emerging trends
and recent developments,”Energy Reports, vol. 7, pp. 8176–8186, Nov. 2021. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S2352484721007289
[2] K. Thakur, M. Qiu, K. Gai, and M. L. Ali, “An Investigation on Cyber Security Threats and Security Models,”
in2015 IEEE 2nd International Conference on Cyber Security and Cloud Computing, Nov. 2015, pp. 307–311.
[Online]. Available: https://ieeexplore.ieee.org/abstract/document/7371499
[3] N. M. Scala, A. C. Reilly, P. L. Goethals, and M. Cukier, “Risk and the Five Hard Problems
of Cybersecurity,”Risk Analysis, vol. 39, no. 10, pp. 2119–2126, 2019. [Online]. Available: https:
//onlinelibrary.wiley.com/doi/abs/10.1111/risa.13309
[4] E. M. Hutchins, M. J. Cloppert, and R. M. Amin, “Intelligence-driven computer network defense informed by
analysis of adversary campaigns and intrusion kill chains,”Leading Issues in Information Warfare & Security
Research, vol. 1, no. 1, p. 80, 2011.
13

# Overview
You are a top-tier cybersecurity expert specialized in extracting structured information from
unstructured data to construct a knowledge graph according to a predefined "olx" ontology. You will
be provided with a log event, optionally accompanied by contextual information.
Your goal is to maximize information extraction from the event while maintaining absolute accuracy.
Leverage both the contextual information and your knowledge of computer systems and cybersecurity
to infer additional insights where possible. The objective is to achieve completeness in the
knowledge graph while remaining strictly ontology-compliant.
# Rules
You MUST adhere to the following constraints at all times:
- The graph must contain exactly one "Event" node.
- Use only the available types as defined in the ontology, without introducing new ones.
- Use the most specific type available for nodes and relationships, e.g. "UserPassword" instead of
"UserCredential".
- Respect the appropriate casing for all types.
- Use the appropriate node prefix for properties, e.g. "userUID" instead of "uid".
- Omit properties with empty values.
- Use the most specific type available for nodes and relationships.
- Respect the structural relationships to infer properties and relationships allowed by the
ontology for each node type.
- The graph must be connected: every node must be reachable from the "Event" node.
- The output must contain only the JSON graph. No other text, comments, or explanations should be
included. The output must be valid JSON and parsable, without any escape characters or newlines.
The JSON must be formatted correctly, with all necessary commas and brackets in place.
# Output Format
The output graph must be in the following JSON format:
{{output_format}}
Each node type has a specific set of allowed properties. The allowed properties for each node type
are: {{properties_schema}}
Each relationship type has a predefined source and target node type. The allowed relationships,
formatted as (source type, relationship type, target type), are: {{triples}}
The following structural relationships exist among node types: {{structural_triples}}."
# Strict Compliance
Adhere to these rules strictly. Any deviation will result in termination.
Table 3: Baseline prompt for log event KG generation.
# Overview
You are a cybersecurity analyst AI. You are given as input a set of knowledge graphs representing
log events captured by a honeypot. Each knowledge graph encodes entities (e.g., processes, IP
addresses, files, commands) and their relationships, and all graphs belong to the same session of
activity, where some form of reconnaissance or attack may have taken place. It is possible that a
session is benevolous, i.e. no attack was conducted. Your task is to analyze the combined activity
across all these knowledge graphs and map them to MITRE ATT&CK tactics.
# Instructions
1. Carefully review the knowledge graphs to identify suspicious behaviors, attack patterns, or
reconnaissance steps.
2. Match observed behaviors to MITRE ATT&CK tactics (high-level adversary objectives, e.g.,
Execution, Persistence, Discovery).
3. If multiple tactics apply, include all plausible ones.
4. If no tactics are applicable, respond an empty list.
5. Do not invent tactics that are not defined in MITRE ATT&CK.
# Strict Compliance
Adhere to these rules strictly. Any deviation will result in termination.
Table 4: Prompt for MITRE ATT&CK tactics prediction.
14

1. Write a detailed description of the input log event in natural language. Include what occurred,
the involved entities, their roles, any parameters, timestamps, or contextual details conveyed in
the log."
2. Write a detailed description of the actual output knowledge graph in natural language. Include
what occurred, the involved entities, their roles, any parameters, timestamps, or contextual
details conveyed in the graph.
3. Assess whether the description of the actual output knowledge graph semantically captures the
same information as the log event's description. Check for:
- Coverage: Are all key elements from the log event present?
- Correctness: Are entities, actions, and relationships represented accurately?
- Relevance: Are any additional nodes or relationships relevant to the log event context?
It is acceptable if the graph contains more information than the log event, as long as the
information isn't contradicting.
Table 5: Prompt for G-Eval ”graph alignment” scoring.
[5] N. Papernot, P. McDaniel, A. Sinha, and M. P. Wellman, “SoK: Security and Privacy in Machine Learning,”
in2018 IEEE European Symposium on Security and Privacy (EuroS&P), Apr. 2018, pp. 399–414. [Online].
Available: https://ieeexplore.ieee.org/abstract/document/8406613
[6] W. Tounsi and H. Rais, “A survey on technical threat intelligence in the age of sophisticated
cyber attacks,”Computers & Security, vol. 72, pp. 212–233, Jan. 2018. [Online]. Available: https:
//www.sciencedirect.com/science/article/pii/S0167404817301839
[7] N. Sun, M. Ding, J. Jiang, W. Xu, X. Mo, Y . Tai, and J. Zhang, “Cyber Threat Intelligence Mining for Proactive
Cybersecurity Defense: A Survey and New Perspectives,”IEEE Communications Surveys & Tutorials, vol. 25,
no. 3, pp. 1748–1774, 2023. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/10117505
[8] M. Nawrocki, M. W ¨ahlisch, T. C. Schmidt, C. Keil, and J. Sch ¨onfelder, “A survey on honeypot software and data
analysis,”arXiv preprint arXiv:1608.06249, 2016.
[9] Y . Zhang, T. Du, Y . Ma, X. Wang, Y . Xie, G. Yang, Y . Lu, and E.-C. Chang, “AttacKG+: Boosting attack graph
construction with Large Language Models,”Computers & Security, vol. 150, p. 104220, Mar. 2025. [Online].
Available: https://linkinghub.elsevier.com/retrieve/pii/S0167404824005261
[10] Y .-T. Huang, R. Vaitheeshwari, M.-C. Chen, Y .-D. Lin, R.-H. Hwang, P.-C. Lin, Y .-C. Lai, E. H.-
K. Wu, C.-H. Chen, Z.-J. Liao, and C.-K. Chen, “MITREtrieval: Retrieving MITRE Techniques
From Unstructured Threat Reports by Fusion of Deep Learning and Ontology,”IEEE Transactions on
Network and Service Management, vol. 21, no. 4, pp. 4871–4887, Aug. 2024. [Online]. Available:
https://ieeexplore.ieee.org/abstract/document/10539631
[11] P. Falcarin and F. Dainese, “Building a Cybersecurity Knowledge Graph with CyberGraph,” inProceedings of the
2024 ACM/IEEE 4th International Workshop on Engineering and Cybersecurity of Critical Systems (EnCyCriS)
and 2024 IEEE/ACM Second International Workshop on Software Vulnerability, ser. EnCyCriS/SVM ’24.
New York, NY , USA: Association for Computing Machinery, Aug. 2024, pp. 29–36. [Online]. Available:
https://dl.acm.org/doi/10.1145/3643662.3643962
[12] J. Liu and J. Zhan, “Constructing Knowledge Graph from Cyber Threat Intelligence Using Large Language
Model,” in2023 IEEE International Conference on Big Data (BigData), Dec. 2023, pp. 516–521. [Online].
Available: https://ieeexplore.ieee.org/abstract/document/10386611
[13] X. Zhao, R. Jiang, Y . Han, A. Li, and Z. Peng, “A survey on cybersecurity knowledge graph
construction,”Computers & Security, vol. 136, p. 103524, Jan. 2024. [Online]. Available: https:
//www.sciencedirect.com/science/article/pii/S0167404823004340
[14] G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open domain question an-
swering,”arXiv preprint arXiv:2007.01282, 2020.
[15] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel,
and others, “Retrieval-augmented generation for knowledge-intensive nlp tasks,”Advances in neural information
processing systems, vol. 33, pp. 9459–9474, 2020.
15

ModelRun Total TimeGeneration
Success
RatioSHACL
Violation
RatioPrecision Recall F1 ScoreEntity
Linking
AccuracyRelationship
Linking
AccuracyG-Eval Score
Mean SD Mean SD Mean SD Mean SD Mean SD Mean SD Mean SD Mean SD Mean SD
Baseline
Claude 3.5 Haiku291.055 3.113 1.000 0.000 0.024 0.002 0.520 0.010 0.397 0.004 0.442 0.005 0.366 0.006 0.227 0.019 0.812 0.013
Claude Sonnet 4250.280 4.384 1.000 0.000 0.008 0.002 0.330 0.034 0.252 0.028 0.283 0.030 0.278 0.033 0.410 0.046 0.357 0.039
Llama 3.391.170 2.233 1.000 0.000 0.031 0.002 0.461 0.012 0.400 0.004 0.422 0.006 0.301 0.009 0.064 0.023 0.822 0.012
Llama 3.156.376 1.636 1.000 0.000 0.014 0.002 0.350 0.021 0.305 0.014 0.313 0.015 0.197 0.012 0.000 0.000 0.820 0.015
Mistral Large572.141 50.558 0.842 0.030 0.012 0.003 0.389 0.019 0.346 0.014 0.352 0.014 0.254 0.013 0.034 0.016 0.660 0.027
Qwen3 Coder 32B344.562 6.408 1.000 0.000 0.012 0.002 0.524 0.007 0.377 0.004 0.429 0.003 0.320 0.002 0.066 0.010 0.816 0.007
*gpt-oss 20B484.890 22.269 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
*gpt-oss 120B464.118 59.890 0.716 0.045 0.011 0.004 0.512 0.035 0.332 0.022 0.396 0.027 0.396 0.029 0.655 0.050 0.543 0.034
Retrieval only
Claude 3.5 Haiku373.647 21.104 1.000 0.000 0.012 0.005 0.705 0.047 0.691 0.075 0.687 0.061 0.607 0.060 0.602 0.158 0.837 0.042
Claude Sonnet 4314.732 30.250 1.000 0.000 0.013 0.002 0.798 0.077 0.739 0.104 0.758 0.091 0.695 0.078 0.104 0.054 0.737 0.063
Llama 3.3139.522 10.340 1.000 0.000 0.024 0.008 0.597 0.090 0.551 0.110 0.563 0.100 0.495 0.077 0.487 0.193 0.872 0.038
Llama 3.1189.516 30.883 1.000 0.000 0.023 0.012 0.553 0.059 0.707 0.034 0.599 0.054 0.583 0.080 0.611 0.275 0.765 0.013
Mistral Large835.787 41.128 1.000 0.000 0.000 0.000 0.001 0.004 0.002 0.005 0.001 0.004 0.001 0.005 0.001 0.003 0.002 0.006
Qwen3 Coder 32B411.240 27.165 1.000 0.000 0.007 0.003 0.722 0.070 0.679 0.083 0.687 0.079 0.594 0.093 0.548 0.169 0.835 0.031
*gpt-oss 20B321.400 12.215 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
*gpt-oss 120B325.110 14.547 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
Str. output only
Claude 3.5 Haiku355.270 5.404 0.978 0.020 0.003 0.001 0.522 0.011 0.513 0.010 0.510 0.010 0.285 0.011 0.011 0.012 0.820 0.019
Claude Sonnet 4294.311 8.359 1.000 0.000 0.012 0.001 0.667 0.006 0.562 0.006 0.602 0.005 0.548 0.008 0.692 0.015 0.784 0.007
Llama 3.387.174 2.742 1.743 0.055 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
Llama 3.170.938 8.571 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
Mistral Large717.396 52.363 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
Qwen3 Coder 32B324.953 17.321 1.000 0.000 0.007 0.002 0.538 0.014 0.421 0.010 0.460 0.009 0.327 0.012 0.078 0.032 0.912 0.025
*gpt-oss 20B465.402 33.846 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
*gpt-oss 120B465.704 57.807 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
Str. output and corr.
Claude 3.5 Haiku367.234 13.727 0.992 0.014 0.003 0.001 0.504 0.006 0.513 0.008 0.500 0.005 0.274 0.006 0.012 0.014 0.834 0.012
Claude Sonnet 4290.750 7.360 1.000 0.000 0.009 0.001 0.666 0.013 0.565 0.003 0.604 0.006 0.569 0.010 0.731 0.029 0.773 0.012
Llama 3.3185.488 7.648 0.970 0.029 0.003 0.001 0.512 0.026 0.350 0.011 0.402 0.015 0.251 0.014 0.010 0.011 0.731 0.015
Llama 3.1315.257 19.953 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
Mistral Large787.661 117.459 0.948 0.032 0.017 0.004 0.446 0.031 0.399 0.017 0.406 0.023 0.281 0.019 0.036 0.023 0.744 0.026
Qwen3 Coder 32B335.039 8.084 1.000 0.000 0.006 0.001 0.520 0.009 0.426 0.006 0.458 0.007 0.319 0.008 0.006 0.020 0.931 0.014
*gpt-oss 20B1337.544 92.934 0.080 0.033 0.002 0.001 0.051 0.021 0.016 0.008 0.024 0.011 0.018 0.011 0.006 0.010 0.025 0.008
*gpt-oss 120B1422.732 151.178 0.160 0.046 0.003 0.002 0.125 0.037 0.036 0.011 0.055 0.016 0.040 0.014 0.006 0.010 0.053 0.019
Starter set retrieval
Claude 3.5 Haiku360.905 20.233 0.994 0.010 0.004 0.001 0.676 0.081 0.705 0.067 0.681 0.074 0.522 0.105 0.394 0.232 0.810 0.015
Claude Sonnet 4263.309 13.360 1.000 0.000 0.008 0.001 0.809 0.040 0.762 0.072 0.775 0.057 0.727 0.0540.8440.046 0.761 0.015
Llama 3.3140.203 18.433 0.990 0.011 0.003 0.002 0.730 0.078 0.627 0.096 0.662 0.089 0.504 0.091 0.431 0.235 0.767 0.017
Llama 3.1301.552 40.101 0.470 0.187 0.004 0.002 0.361 0.147 0.368 0.156 0.358 0.148 0.312 0.135 0.357 0.151 0.360 0.137
Mistral Large773.065 92.966 0.966 0.019 0.010 0.003 0.671 0.073 0.658 0.081 0.651 0.076 0.548 0.083 0.499 0.169 0.743 0.009
Qwen3 Coder 32B348.197 9.987 1.000 0.000 0.005 0.001 0.728 0.071 0.682 0.077 0.694 0.074 0.585 0.082 0.549 0.172 0.821 0.042
*gpt-oss 20B757.344 149.026 0.608 0.157 0.006 0.002 0.500 0.134 0.494 0.140 0.487 0.134 0.451 0.132 0.487 0.154 0.443 0.126
*gpt-oss 120B991.744 299.646 0.740 0.177 0.007 0.002 0.622 0.155 0.599 0.183 0.596 0.171 0.543 0.181 0.564 0.204 0.531 0.154
Full retrieval
Claude 3.5 Haiku385.248 14.288 0.998 0.006 0.004 0.001 0.712 0.062 0.723 0.059 0.708 0.060 0.546 0.075 0.461 0.200 0.812 0.013
Claude Sonnet 4291.379 11.302 1.000 0.000 0.008 0.0010.8170.0330.7760.0570.7860.0450.7310.046 0.786 0.050 0.764 0.019
Llama 3.3129.136 5.681 0.986 0.019 0.003 0.001 0.764 0.042 0.568 0.059 0.630 0.050 0.528 0.060 0.560 0.171 0.714 0.034
Llama 3.1340.801 58.229 0.560 0.141 0.005 0.002 0.439 0.119 0.399 0.102 0.411 0.107 0.356 0.108 0.367 0.149 0.419 0.094
Mistral Large1182.990 200.353 0.904 0.040 0.008 0.003 0.661 0.061 0.646 0.066 0.638 0.063 0.534 0.073 0.490 0.140 0.690 0.044
Qwen3 Coder 32B415.110 29.709 1.000 0.000 0.004 0.002 0.758 0.067 0.702 0.059 0.717 0.063 0.598 0.070 0.539 0.192 0.787 0.035
*gpt-oss 20B577.353 120.219 0.826 0.110 0.009 0.003 0.639 0.099 0.664 0.116 0.641 0.106 0.579 0.085 0.563 0.095 0.630 0.098
*gpt-oss 120B769.929 248.346 0.878 0.151 0.007 0.003 0.734 0.129 0.731 0.150 0.721 0.143 0.666 0.132 0.706 0.147 0.652 0.133
Table 6: Results of ablation study of OntoLogX across metrics and LLMs. Reasoning models are highlighted with an
asterisk before their name. Best values for precision, recall, F1 score, entity linking accuracy, and relationship linking
accuracy are highlighted with bold text.
16

[16] A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta,
A. Garriga-Alonso, and others, “Beyond the imitation game: Quantifying and extrapolating the capabilities of
language models,”arXiv preprint arXiv:2206.04615, 2022.
[17] J. Zhang, H. Bu, H. Wen, Y . Liu, H. Fei, R. Xi, L. Li, Y . Yang, H. Zhu, and D. Meng, “When LLMs meet
cybersecurity: a systematic literature review,”Cybersecurity, vol. 8, no. 1, p. 55, Feb. 2025. [Online]. Available:
https://doi.org/10.1186/s42400-025-00361-w
[18] L. Payne and M. Xie, “Log File Anomaly Detection Using Knowledge Graph Completion,” inProceedings
of the 2024 8th International Conference on Deep Learning Technologies, ser. ICDLT ’24. New
York, NY , USA: Association for Computing Machinery, Nov. 2024, pp. 42–48. [Online]. Available:
https://dl.acm.org/doi/10.1145/3695719.3695726
[19] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin,
“Attention Is All You Need,” Aug. 2023. [Online]. Available: http://arxiv.org/abs/1706.03762
[20] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding,” May 2019. [Online]. Available: http://arxiv.org/abs/1810.04805
[21] T. B. Brown, B. Mann, N. Ryder, and others, “Language models are few-shot learners,”Advances in Neural
Information Processing Systems, vol. 33, pp. 1877–1901, 2020.
[22] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, “Improving language understanding by generative
pre-training,” 2018.
[23] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi `ere, N. Goyal, E. Hambro,
and F. Azhar, “Llama: Open and efficient foundation language models,”arXiv preprint arXiv:2302.13971, 2023.
[24] J. Bai, S. Bai, Y . Chu, Z. Cui, K. Dang, X. Deng, Y . Fan, W. Ge, Y . Han, F. Huang, B. Hui, L. Ji, M. Li, J. Lin,
R. Lin, D. Liu, G. Liu, C. Lu, K. Lu, J. Ma, R. Men, X. Ren, X. Ren, C. Tan, S. Tan, J. Tu, P. Wang, S. Wang,
W. Wang, S. Wu, B. Xu, J. Xu, A. Yang, H. Yang, J. Yang, S. Yang, Y . Yao, B. Yu, H. Yuan, Z. Yuan, J. Zhang,
X. Zhang, Y . Zhang, Z. Zhang, C. Zhou, J. Zhou, X. Zhou, and T. Zhu, “Qwen Technical Report,” Sep. 2023.
[Online]. Available: http://arxiv.org/abs/2309.16609
[25] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel,
G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and
W. E. Sayed, “Mistral 7B,” Oct. 2023. [Online]. Available: http://arxiv.org/abs/2310.06825
[26] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval augmented language model pre-training,” in
International conference on machine learning. PMLR, 2020, pp. 3929–3938.
[27] D. B. Acharya, K. Kuppan, and B. Divya, “Agentic AI: Autonomous Intelligence for Complex
Goals—A Comprehensive Survey,”IEEE Access, vol. 13, pp. 18 912–18 936, 2025. [Online]. Available:
https://ieeexplore.ieee.org/abstract/document/10849561
[28] F. Sado, C. K. Loo, W. S. Liew, M. Kerzel, and S. Wermter, “Explainable goal-driven agents and robots-a
comprehensive review,”ACM Computing Surveys, vol. 55, no. 10, pp. 1–41, 2023.
[29] C. Castelfranchi, “Modelling social action for AI agents,”Artificial Intelligence, vol. 103, no. 1, pp. 157–182,
Aug. 1998. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0004370298000563
[30] S. Franklin and A. Graesser, “Is It an agent, or just a program?: A taxonomy for autonomous agents,” inIntelli-
gent Agents III Agent Theories, Architectures, and Languages, J. P. M ¨uller, M. J. Wooldridge, and N. R. Jennings,
Eds. Berlin, Heidelberg: Springer, 1997, pp. 21–35.
[31] R. Sapkota, K. I. Roumeliotis, and M. Karkee, “AI Agents vs. Agentic AI: A Conceptual Taxonomy,
Applications and Challenges,” May 2025. [Online]. Available: http://arxiv.org/abs/2505.10468
[32] R. Studer, V . R. Benjamins, and D. Fensel, “Knowledge engineering: Principles and methods,” inData & knowl-
edge engineering. Elsevier, 1998, vol. 25, pp. 161–197.
[33] H. Knublauch and D. Kontokostas, “Shapes constraint language (SHACL),”W3C Recommendation, vol. 20,
2017. [Online]. Available: https://www.w3.org/TR/shacl/
[34] P. Pareti, G. Konstantinidis, T. J. Norman, and others, “Knowledge graph quality management with SHACL,”
Journal of Web Semantics, vol. 74, p. 100714, 2022.
[35] K. Rabbani, M. Lissandrini, and K. Hose, “SHACTOR: improving the quality of large-scale knowledge
graphs with validating shapes,” inCompanion of the 2023 international conference on management of data,
SIGMOD/PODS 2023, seattle, WA, USA, june 18-23, 2023, S. Das, I. Pandis, K. S. Candan, and S. Amer-Yahia,
Eds. ACM, 2023, pp. 151–154. [Online]. Available: https://doi.org/10.1145/3555041.3589723
17

[36] Z. Syed, A. Padia, T. Finin, L. Mathews, and A. Joshi, “UCO: a unified cybersecurity ontology,” inAAAI work-
shop: Artificial intelligence for cyber security, 2016, pp. 14–21.
[37] S. Barnum, “Standardizing cyber threat intelligence information with the structured threat information
eXpression (STIX),”The MITRE Corporation, 2012. [Online]. Available: http://stixproject.github.io/
[38] A. Oltramari, L. F. Cranor, R. J. Walls, and P. McDaniel, “Building an ontology of cyber security,” inProceedings
of the ninth conference on semantic technology for intelligence, defense, and security (STIDS). CEUR-WS.org,
2014, pp. 54–61.
[39] C. Wagner, A. Dulaunoy, G. Wagener, and A. Iklody, “MISP: The malware information sharing platform,” in
Proceedings of the 2016 ACM on workshop on information sharing and collaborative security. ACM, 2016,
pp. 49–56.
[40] E. Kiesling, A. Ekelhart, K. Kurniawan, and F. J. Ekaputra, “The SEPSES knowledge graph: An integrated
resource for cybersecurity,” inThe semantic web - ISWC 2019 - 18th international semantic web conference,
auckland, new zealand, october 26-30, 2019, proceedings, part II, ser. Lecture notes in computer science, vol.
11779. Springer, 2019, pp. 198–214. [Online]. Available: https://doi.org/10.1007/978-3-030-30796-7 13
[41] Z. Ma, D. J. Kim, and T.-H. Chen, “LibreLog: Accurate and Efficient Unsupervised Log Parsing Using
Open-Source Large Language Models,” Nov. 2024. [Online]. Available: http://arxiv.org/abs/2408.01585
[42] A. Ekelhart, F. J. Ekaputra, and E. Kiesling, “The SLOGERT Framework for Automated Log Knowledge Graph
Construction,” inThe Semantic Web, R. Verborgh, K. Hose, H. Paulheim, P.-A. Champin, M. Maleshkova,
O. Corcho, P. Ristoski, and M. Alam, Eds. Cham: Springer International Publishing, 2021, pp. 631–646.
[43] K. Kurniawan, A. Ekelhart, E. Kiesling, G. Quirchmayr, and A. M. Tjoa, “KRYSTAL: Knowledge graph-based
framework for tactical attack discovery in audit data,”Computers & Security, vol. 121, p. 102828, Oct. 2022.
[Online]. Available: https://www.sciencedirect.com/science/article/pii/S016740482200222X
[44] M. Boffa, I. Drago, M. Mellia, L. Vassio, D. Giordano, R. Valentim, and Z. B. Houidi, “LogPr ´ecis:
Unleashing language models for automated malicious log analysis: Pr ´ecis: A concise summary of essential
points, statements, or facts,”Computers & Security, vol. 141, p. 103805, Jun. 2024. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S0167404824001068
[45] K. Kurniawan, E. Kiesling, and A. Ekelhart, “CyKG-RAG: Towards knowledge-graph enhanced retrieval aug-
mented generation for cybersecurity,” 2024.
[46] J. Carbonell and J. Goldstein, “The use of MMR, diversity-based reranking for reordering documents and pro-
ducing summaries,” inProceedings of the 21st annual international ACM SIGIR conference on Research and
development in information retrieval, 1998, pp. 335–336, read Status: To read Read Status Date: 2025-06-
20T13:39:52.809Z.
[47] X. Zhang, Y . Zhang, D. Long, W. Xie, Z. Dai, J. Tang, H. Lin, B. Yang, P. Xie, F. Huang, M. Zhang,
W. Li, and M. Zhang, “mGTE: Generalized Long-Context Text Representation and Reranking Models for
Multilingual Text Retrieval,” Oct. 2024, arXiv:2407.19669 [cs] Read Status: To read Read Status Date:
2025-09-03T22:56:25.799Z. [Online]. Available: http://arxiv.org/abs/2407.19669
[48] G. C. Publio, D. Esteves, A. Ławrynowicz, P. Panov, L. Soldatova, T. Soru, J. Vanschoren, and H. Zafar,
“ML-Schema: Exposing the Semantics of Machine Learning with Schemas and Ontologies,” Jul. 2018,
arXiv:1807.05351 [cs] Read Status: Read Read Status Date: 2025-05-13T14:55:43.001Z. [Online]. Available:
http://arxiv.org/abs/1807.05351
[49] W. Kwon, Z. Li, S. Zhuang, Y . Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica,
“Efficient Memory Management for Large Language Model Serving with PagedAttention,” Sep. 2023,
arXiv:2309.06180 [cs] Read Status: To read Read Status Date: 2025-09-04T14:16:10.735Z. [Online].
Available: http://arxiv.org/abs/2309.06180
[50] M. Landauer, F. Skopik, M. Frank, W. Hotwagner, M. Wurzenberger, and A. Rauber, “AIT Log Data Set
V2.0,” Feb. 2022, read Status: Read Read Status Date: 2025-02-04T16:56:44.540Z. [Online]. Available:
https://zenodo.org/records/5789064
[51] Y . Liu, D. Iter, Y . Xu, S. Wang, R. Xu, and C. Zhu, “G-Eval: NLG Evaluation using GPT-4 with
Better Human Alignment,” May 2023, arXiv:2303.16634 [cs] Read Status: In progress Read Status Date:
2025-05-13T15:31:42.187Z. [Online]. Available: http://arxiv.org/abs/2303.16634
18

Luca Cotti graduated cum laude from the University of Brescia, Italy in 2024. He is currently a research fellow at the
Department of Information Engineering at the University of Brescia. His research interests include the use of machine
learning methodologies, particularly large language models, for the development of autonomous agents.
Idilio Drago received the Ph.D. degree in computer science from the University of Twente, The Netherlands. He is an
Associate Professor with the Computer Science Department, University of Turin, Italy. His research interests include
network security, machine learning, and Internet measurements. He is particularly interested on how big data and
machine learning can help to extract knowledge from network data and help secure the network and automate network
management tasks. He was awarded an Applied Networking Research Prize in 2013 by the IETF/IRTF for his work
on cloud storage traffic analysis.
Anisa Rula received the Ph.D. degree in computer science from the University of Milano-Bicocca, in 2014. She
has been an Associate Professor of computer science with the Department of Information Engineering, University of
Brescia, since December 2023. Her research interests include the intersection of semantic knowledge technologies
and data quality, with a particular focus on data integration. She is researching new solutions to data integration
with respect to the quality of data modeling and efficient solutions for large-scale data sources. Recently, she has
been working on data understanding for large and complex datasets, on knowledge extraction, and on semantic data
enrichment and refinement.
19

Devis Bianchini received the Ph.D. degree in information engineering from the University of Brescia, in 2006. He is
currently a Full Professor of computer science engineering and the Head of the Databases, Information Systems and
Web Research Group, Department of Information Engineering, University of Brescia. He is also the Chair of the Big
and Open Data Laboratory, University of Brescia. He is the author of papers published in international journals and
conference proceedings. He is the referee for international journals. He coordinated national and regional research
projects in the fields of smart cities and industry 4.0. His research interests include ontology-based resource discovery,
service-oriented architectures, big data management, and web information systems design.
Federico Cerutti is currently a Full Professor at the University of Brescia, Italy. He is also a Rita Levi-Montalcini
Fellowship Laureate and an Honorary Senior Lecturer at Cardiff University, U.K.; a Visiting Fellow at the University
of Southampton, U.K.; and the Chair of the University of Brescia’s Local Branch of the Italian Cybersecurity National
Laboratory. His research is in learning and reasoning with uncertain and sparse data for supporting (cyber-threat)
intelligence analysis.
20