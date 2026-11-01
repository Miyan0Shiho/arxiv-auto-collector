# OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning

**Authors**: Hui Yang, Jiaoyan Chen, Yongsheng Gao

**Published**: 2026-10-26 22:04:58

**PDF URL**: [https://arxiv.org/pdf/2610.36335v1](https://arxiv.org/pdf/2610.36335v1)

## Abstract
Large language models (LLMs) perform well across many real-world tasks but remain prone to hallucination, limiting their reliability in domains such as biomedicine and healthcare, where answers must be grounded in established knowledge. Retrieval-augmented generation (RAG) mitigates this problem by incorporating external information, but often overlooks logical relationships and relies heavily on the LLM's own reasoning, which can compromise answer faithfulness. We propose OA-RAG, an agent that integrates formal reasoning with LLM reasoning, drawing on both the logical structure and informal annotations of domain ontologies. For each answer, OA-RAG reports the ontology observations used as evidence and a model-assessed grounding level (entailed, supported, related, or unrelated) that characterises the strength of that evidence. Across benchmarks derived from three large-scale ontologies and six real-world medical question-answering tasks, OA-RAG achieves the best overall performance among the evaluated ontology-fine-tuned and popular RAG methods, with particularly strong results on reasoning-intensive tasks. The reported grounding levels also offer a useful indication of answer reliability, with stronger grounding generally associated with higher accuracy. Finally, our results demonstrate the value of ontologies as knowledge sources: OA-RAG is competitive with, and sometimes outperforms, domain-specific RAG systems that draw on sources such as textbooks.

## Full Text


<!-- PDF content starts -->

OA-RAG: Ontology-based Agentic RAG for Faithful LLM
Reasoning
Hui Yang
The University of Manchester
Manchester, United Kingdom
hui.yang-2@manchester.ac.ukJiaoyan Chen
The University of Manchester
Manchester, United Kingdom
jiaoyan.chen@manchester.ac.ukYongsheng Gao
SNOMED International
London, United Kingdom
yga@snomed.org
Abstract
Large language models (LLMs) perform well across many real-world
tasks but remain prone to hallucination, limiting their reliability
in domains such as biomedicine and healthcare, where answers
must be grounded in established knowledge. Retrieval-augmented
generation (RAG) mitigates this problem by incorporating external
information, but often overlooks logical relationships and relies
heavily on the LLM’s own reasoning, which can compromise an-
swer faithfulness. We propose OA-RAG, an agentic framework that
integrates formal reasoning with LLM reasoning, drawing on both
the logical structure and informal annotations of domain ontolo-
gies. For each answer, OA-RAG reports the ontology observations
used as evidence and a grounding level (entailed,supported,re-
lated, orunrelated) that characterises the evidence’s faithfulness.
Across benchmarks derived from three large-scale ontologies and
six real-world medical question-answering tasks, OA-RAG achieves
the best overall performance among the evaluated ontology-fine-
tuned and popular RAG methods, with particularly strong results
on reasoning-intensive tasks. The reported grounding levels also
offer a useful indication of answer reliability, with stronger ground-
ing generally associated with higher accuracy. Finally, our results
demonstrate the value of ontologies as knowledge sources: OA-RAG
is competitive with, and sometimes outperforms, domain-specific
RAG systems that draw on sources such as textbooks.
1 Introduction
Large language models (LLMs) have demonstrated strong capabil-
ities across a wide range of knowledge-intensive tasks. However,
their answers remain constrained by the knowledge encoded in
their parameters and may contain hallucinations, that is, plausible
but factually incorrect statements [ 19,20]. Their reliability is fur-
ther limited by the lack of transparent reasoning and by gaps in
domain- or organization-specific knowledge.
Retrieval-augmented generation (RAG) augments LLMs by
prompting with external evidence extracted from documents [ 9],
tables [ 52], and knowledge graphs [ 31], while tool-using agents
enable more adaptive, dynamic and autonomous access to such
sources [ 32,44,49,60]. However, existing methods mainly lever-
age textual and structured knowledge, leaving explicit logical
knowledge under-utilised. Consequently, logical inference still
relies heavily on the LLM’s internal reasoning, which can make its
conclusions less faithful to the given knowledge.
Meanwhile, many scientific and biomedical domains already
provide rich shared, explicit and formal knowledge through on-
tologies in Semantic Web standards like Description Logic under-
pinned OWL (Web Ontology Language) [ 18,46]. These ontologies
on the Web, such as the Gene Ontology (GO) [ 1], FoodOn [ 8], andINPUT / EVIDENCE
QUERY
(Natural Language)
WEB
ONTOLOGY
Formal Axioms
𝐴⊑𝐵
𝐴⊑∃𝑟.𝐵
Informal
Annotations
•Definitions
•Comments
•Synonyms(a)Formal Reasoning
Axioms
𝐴⊑𝐵
𝐴⊑∃𝑟.𝐵Exact
Answer
Formal Axioms Formal Reasoner
(b)Ontology-Grounded LLM Reasoning
Axioms
𝐴⊑𝐵
𝐴⊑∃𝑟.𝐵+ LLMOntology-
Grounded
Answer
Formal
AxiomsInformal
AnnotationsLLM
Reasoning
(c)Ontology-Assisted LLM Reasoning
+ LLM LLMOntology-
Assisted
AnswerOntology
EvidenceLLM Parametric
KnowledgeLLM
Reasoning
Figure 1: Three ontology-augmented LLM reasoning patterns,
ordered from greatest to least faithfulness to the ontology:
(a) the answer is can be inferred from the ontology’s formal
axioms; (b) an LLM can infer the answer based on the ontol-
ogy’s formal axioms and informal annotations; and (c) an
LLM generates the answer freely, with optional grounding
in relevant ontology content (classic RAG).
SNOMED CT [ 7], encode domain concepts and relations using
Description Logic axioms, allowing dedicated reasoners to derive
implicit relations soundly and efficiently [ 3]. Moreover, ontolo-
gies can contain rich textual annotations, including definitions and
comments, providing informal but relatively important and comple-
mentary domain knowledge. This combination of formal semantics
and textual annotations motivates our central question:
Can ontologies bridge formal reasoning and LLM reasoning to
produce more faithful answers to domain-specific questions?
According to the knowledge used, answering domain-specific ques-
tions utilisng LLM and ontology may include the following three
reasoning patterns, as illustrated in Figure 1:
•Formal reasoning: the answer is faithful to the ontology’s formal
axioms and can be derived by a symbolic ontology reasoner.
•Ontology-grounded LLM reasoning: the answer is faithful to the
ontology’s formal axioms and informal annotations (e.g., nat-
ural language labels and definitions), and requires an LLM for
inference based on these ontology’s information.
arXiv:2610.36335v1  [cs.AI]  26 Oct 2026

Yang et al.
•Ontology-assisted LLM reasoning: the answer is not faithful to the
ontology, and the LLM reasoning relies on the LLM’s parametric
knowledge as well.
Importantly, the appropriate reasoning pattern is not known in
advance and may vary across questions, requiring the system to
adaptively select among the three reasoning strategies.
To address this question, we propose OA-RAG, an agentic frame-
work that combines ontology reasoning with LLM reasoning in
three stages (§4): (1) pre-tool reasoning using the LLM’s paramet-
ric knowledge; (2) ReAct-based querying, where logical inference
results and textual annotations are accessed via SPARQL; and (3)
answer generation, which consolidates the evidence, and reports
the ontology evidence supporting the answer with a grounding
level (§4.1.3). Rather than treating the retrieved evidence as equally
informative, the four levels distinguish evidence that logically en-
tails the answer (entailed), supports the answer without strictly
entailing it (supported), is merely relevant (related) to the answer,
or is irrelevant (unrelated) to the answer. This reflects the answer’s
faithfulness to the ontology, where entailed, supported and related
correspond to formal reasoning, ontology-grounded LLM reasoning
and ontology-assisted LLM reasoning in Figure 1, respectively.
We developed an automatic procedure for constructing eval-
uation datasets for the first two patterns—formal reasoning and
ontology-grounded LLM reasoning—using three large-scale, real-
world ontologies: GO, FoodOn, and SNOMED CT. For the third
pattern, we evaluated OA-RAG on six real-world medical question-
answering tasks following the experimental setup of OntoTune [ 29].
Our main contributions are as follows:
•We propose trustful LLM reasoning based on ontology’s both
formal and informal knowledge, and develop an ontology-based
agentic RAG framework OA-RAG, which answers a domain-
specific question with evidence and their grounding levels.
•We formulate ontology-based faithful LLM reasoning in terms
of three patterns with ranging faithfulness, and construct a QA
benchmark with corresponding datasets for comprehensive eval-
uation.
•OA-RAG achieves strong performance across all three reasoning
patterns. On the first two patterns, it outperforms the strongest
baseline by 10–15 percentage points on average (44.3% vs. 28.9%
with Llama-3-8B-Instruct; 59.7% vs. 49.7% with Qwen3.5-9B). On
the third pattern, it remains competitive with specialized medical
QA methods while using10 ×fewer tokens than the strongest
baseline with Qwen3.5-9B (9.7k vs. 96.7k per question).
2 Related Work
LLMs with external knowledge.Retrieval-augmented genera-
tion (RAG) augments LLMs with external knowledge from diverse
sources, including unstructured text [ 9] and structured sources such
as tables [ 52] and knowledge graphs [ 31]. Domain-specific RAG
has also received much attention, particularly in medicine [ 48,54–
56,59]. More recently, growing interest has focused on RAG agents
that iteratively retrieve information throughout the reasoning pro-
cess [ 21,50,56]. For example, in knowledge-base question an-
swering [ 26], graph agents traverse knowledge graphs step by
step [ 32,49], often following the reasoning-and-acting paradigm
of tool-using LLMs [ 39,44,60]. However, these methods still relyprimarily on LLMs to reason over retrieved knowledge and may
therefore be less faithful and trustworthy than our approach, which
additionally incorporates formal reasoning.
LLMs as interfaces to formal reasoning.A prominent line
of work combines LLMs with formal reasoning through neuro-
symbolic methods, which typically translate natural language into
formal representations and delegate inference to symbolic engines.
Examples include executing structured queries over knowledge
bases [ 27,30] and translating problems into logical forms for ex-
ternal solvers [ 35,37]. Most existing methods, however, assume a
fully symbolic problem representation, where either all premises
are explicitly provided or the question is entirely expressible in the
vocabulary of a knowledge base. In contrast, we study a hybrid set-
ting in which the relevant premises span formal axioms, informal
annotations, and the LLM’s parametric knowledge, and a question
cannot necessarily be resolved by formal reasoning alone.
LLMs and ontologies.Language models have been studied
for probing factual and ontological knowledge [ 16,61], learning
ontology embeddings [ 17,57], reasoning over incomplete ontolo-
gies [ 58], and supporting ontology matching and learning [ 4,14,15].
Retrieval-based methods have also begun to exploit ontological
structure: OG-RAG [ 45] and ORT [ 28] use ontologies to structure
or constrain information retrieved from documents and knowledge
graphs, respectively. OntoTune [ 29] and TaxoLLaMA [ 33] instead
incorporate ontology knowledge into model parameters through
fine-tuning. However, prior work has largely treated ontologies as
sources of structured or textual knowledge, rather than as knowl-
edge bases with both formal and informal knowledge as well as
logical reasoning capabilities. To the best of our knowledge, none
of the existing works explicitly integrates ontology-based logical
reasoning with LLMs within an agentic framework.
3 Preliminaries
An OWL ontology [ 18] represents domain knowledge using con-
cepts (classes), relations (object properties), and logical axioms,
whose semantics can be characterised by Description Logics (DLs).
For the lightweight DL EL+, with concept names 𝐴∈N Cand role
names𝑟∈N R, concepts are built by
𝐶::=⊤|𝐴|𝐶⊓𝐷|∃𝑟.𝐶
and ontologies contain General Concept Inclusions (GCIs) contain
General Concept Inclusions (GCIs) 𝐶⊑𝐷 , equivalences 𝐶≡𝐷 , and
role inclusions 𝑟1◦···◦𝑟𝑛⊑𝑟[2]. We writeO|=𝛼 when the axiom
𝛼can be inferred from ontology O. For example, given Graduate⊑
Student andStudent⊑∃affiliatedWith.Department , we have
O|=Graduate⊑∃affiliatedWith.Department.
OWL ontologies may additionally containannotations, such as
rdfs:label andrdfs:comment , which provide lexical and descrip-
tive information, including alternative labels, definitions, and cross-
references. Under OWL direct semantics, annotations do not affect
DL entailments and are therefore ignored by DL reasoners.
Given an OWL ontology, one can construct an RDF graph consist-
ing of(subject,predicate,object) triples and execute SPARQL 1.1 [ 13]
queries over the graph to retrieve answers matching specified graph
patterns. In particular, an axiom of the form 𝐴⊑∃𝑟.𝐵 , whether

OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning
1. Pre-Tool Reasoning 2. ReAct with the Ontology Tool (Iterative Loop) 3. Answering
QueryLLM reasoning
Initial reasoning
(without the
ontology tool)
“Think step by step briefly to
reason about the question... ”Thought
Planning & Reasoning
(with the ontology tool)
“An ontology is available
through the tool below... ”Action
query(SPARQL)
DESCRIBE “<name>”
SELECT ?x { ... }ASK { ... }
SELECT ?o / ?r { ... }
Query fallback for an empty SELECT ?xAnswer or
limit reached?No
YesLLM answering
from the clean
evidence digest{}Output
Answer option B
Evidence E1 E2
Grounding level
entailed supported
related unrelatedsupported✓
Step 1“The tool phase is over. Using the question
and the observations below, . . . ”
Step 2“Your answer above is final; do not change
it. Now state what it rests on . . . ”Ontology
Service Concept Cards Name Linking Formal Axioms Ontology Reasoner
Figure 2: Overall framework of OA-RAG: (1) pre-tool reasoning; (2) a ReAct loop of SPARQL queries to the ontology service
(name linking, reasoner, query fallback); (3) answering from the evidence digest, with the cited evidence and a grounding level.
explicitly stated in the ontology or inferred by a DL reasoner, can be
materialised as a class-level triple (𝐴,𝑟,𝐵) . Ontology annotations
can likewise be represented as triples in the same graph. SPARQL
can then query both types of information using graph patterns with
variables: SELECT returns variable bindings, ASKtests whether a
matching pattern exists, and DESCRIBE returns an RDF description
of selected resources. These SPARQL query patterns form the basis
of OA-RAG.
4 OA-RAG
We now present our main architecture, OA-RAG, illustrated in
Figure 2. OA-RAG consists of two main components. The first is
the core agentic framework, which follows a standard ReAct [ 60]
architecture to answer a given question through iterative reasoning
and tool interaction (§4.1). The second is the ontology service,
which provides textual ontology representations and a backend that
interfaces LLM-generated queries with ontology reasoners (§4.2).
4.1 Main ReAct Architecture
As illustrated in Figure 2, OA-RAG adopts a reasoning-and-acting
architecture [ 60] comprising three stages: (1) pre-tool reasoning, (2)
iterative reasoning and acting with access to the ontology, and (3)
answer generation. The full prompts are provided in Appendix E.
4.1.1 Pre-tool Reasoning.Before entering the main ReAct loop,
OA-RAG prompts the LLM to perform an initial analysis of the
input context and question. Given a pre-tool reasoning prompt Ppre
and a question 𝑞, the modelMproduces an initial reasoning state
ℎ0=M(P pre,𝑞).
The purpose of this stage is to encourage the LLM to reason first
using only the information already available in the input context
and, when that information is sufficient, to form a preliminary
answer. In the subsequent stage, the ontology is introduced as an
additional source of structured knowledge. The model can then
use the ontology to retrieve information missing from the original
context and to verify, refine, or revise the preliminary answer when
one has been formed.4.1.2 ReAct Loop.Following the pre-tool reasoning stage, OA-
RAG enters the main ReAct loop, in which the model can interact
with an ontology tool to retrieve ontology information and invoke
the corresponding reasoner. The loop continues within the same
conversation. After producing the initial reasoning state ℎ0, the
model receives an agent prompt Pagent as a new user turn, which
introduces the available tool and specifies when it should be used.
At iteration 𝑡, the model generates a single action 𝑎𝑡conditioned
on the conversation history, including its previous actions and the
corresponding observations returned by the ontology service (§4.2).
Formally,
𝑎𝑡=M Ppre,𝑞,ℎ 0,Pagent,{(𝑎𝑗,𝑜𝑗)}𝑗<𝑡, 𝑜𝑡=S(𝑎𝑡),
whereSdenotes the ontology service. The resulting observation
𝑜𝑡is appended to the conversation and becomes part of the context
for the next iteration. This process repeats until the model produces
a final answer or reaches the per-item interaction budget.
Agent PromptP agent(condensed)
An ontology is available through the tool below. Continue from your answer
above. You can use tools of the form query(sparql) to execute SPARQL queries
over ontology concepts and relations, including DESCRIBE ,ASK, and SELECT
queries.
Use the ontology when the question concerns how its concepts are defined,
classified, or related. For questions that do not depend on the ontology, answer
using your own knowledge.
Action.In OA-RAG, actions are expressed as SPARQL queries [ 13].
We support three types of queries: DESCRIBE , which retrieves de-
tailed information about a concept; ASK, which checks whether
a subsumption is entailed by the ontology reasoner; and SELECT ,
which retrieves concepts or relations satisfying specified conditions.
The three action types are defined as follows:
(1)DESCRIBE "A" : Given a concept name 𝐴, this query returns
the concept card of 𝐴(§4.2.1): its names, definition, parents,
children, stated relations, and other annotations.
(2)ASK {"A" "r" "B"} : This query checks whether the subsump-
tionO|=𝐴⊑∃𝑟.𝐵 holds. By setting 𝑟to"is a" or"subclass
of", we can also check subsumptions of the form O|=𝐴⊑𝐵 .
The query returnsEntailedif the entailment is verified by the

Yang et al.
ontology reasoner andNot Entailedotherwise. Importantly,
under the open-world assumption,Not Entaileddoes not
imply that the corresponding subsumption is false, since the
ontology may be incomplete. We explicitly instruct the model
about this distinction in the prompt.
(3)SELECT ?x WHERE {<conditions>} : This query retrieves con-
cepts or relations that satisfy the specified conditions. For ex-
ample, a condition of the form {?x "𝑟" "𝐵"} returns concept
names𝐴such thatO|=𝐴⊑∃𝑟.𝐵 . Similarly, {"A" "r" ?x}
asks for concepts 𝐵such thatO|=𝐴⊑∃𝑟.𝐵 , while {"A" ?x
"B"}returns relations𝑟such thatO|=𝐴⊑∃𝑟.𝐵.
Query fallback.An exact SELECT ?x query may return no results,
particularly in ontology-grounded LLM reasoning, where relevant
information may appear only in informal annotations, which are
inaccessible to the exact reasoner. To handle such cases, we intro-
duce a query fallback. When a SELECT ?x query yields no matching
concept, we retrieve the top- 𝑘candidate concepts by comparing
the query against the concept cards (§4.2.1) using BM25 [ 43]. OA-
RAG can then examine these candidates, inspect their associated
information viaDESCRIBE, and continue the ReAct loop.
4.1.3 Answering.Finally, we introduce a dedicated answering step
that constructs a clean context from the observations gathered
during reasoning. It proceeds in two steps (Figure 2, stage 3).
Step 1: Answer.We discard observations produced by failed or
uninformative tool calls, such as execution errors and queries that
return empty results. If the preceding ReAct loop has already pro-
duced a candidate answer, we retain it as a draft, denoted as ˜𝑦.
Let𝐸denote the observations retained after filtering (theevidence
digest). The final answer is
ˆ𝑦=(
M Pfin,𝑞,𝐸, ˜𝑦if𝐸≠∅,
M Pown,𝑞otherwise.
We distinguish these two cases using separate prompts. When in-
formative observations remain after filtering ( 𝐸≠∅ ),Pfininstructs
the model to answer using the question, the retained observations,
and the draft. Otherwise, Pownasks the model to answer from its
parametric knowledge.
Step 2: Evidence and grounding level.When the observation set 𝐸
is non-empty, we continue the Step 1 conversation with one more
turn, the promptPatt, to identify which observations its answer
relies on and to assess the strength of their support:
(ˆ𝐶,ˆ𝑔)=M Step 1,P att,𝐸,
where ˆ𝐶⊆𝐸 denotes the set of cited evidence and ˆ𝑔denotes the
grounding level. We distinguish four grounding levels: (1)entailed,
when the answer can be logically derived from the facts stated in
the observations; (2)supported, when the answer can be reasonably
inferred from the observations, possibly in combination with the
LLM’s parametric knowledge, but does not follow as a strict logical
consequence; (3)related, when the observations are relevant to the
question but insufficient to determine the answer; and (4)unrelated,
when the observations are irrelevant to the question and the answer.
Accordingly, we expect formal reasoning (Figure 1) to produce
entailed answers; ontology-grounded LLM reasoning, entailed or
supported answers; and ontology-assisted LLM reasoning, mostlyrelated answers, although unrelated, supported, and entailed cases
may also occur. Although it does not provide a formal guarantee
of answer correctness, the reported grounding level reflects the
model’s assessment of the extent to which the available evidence
supports the answer and the answer is faithful to the ontology.
4.2 Ontology Service
4.2.1 Ontology Concept Cards.To make the ontology accessible
to the agent through standard text-based retrieval, we convert each
concept into aconcept card, as illustrated below. Each card preserves
the information in the ontology, with each item represented as a
key: valueline using the property’s original label. The fields appear
in a fixed order: the name, annotation definition, parents, children,
stated relations, comments, and remaining annotations. All values
are preserved as stated in the ontology, with three exceptions. The
parentsandchildrenfields list the concept’s inferred direct parents
and children, respectively1. Thestated relationsfield lists each pair
(𝑟,𝐵) for which the ontology contains an axiom of the form 𝐴⊑
∃𝑟.𝐵⊓𝐶or𝐴≡∃𝑟.𝐵⊓𝐶, represented as “r -> B”.
Concept card formitotic chromosome condensation(GO)
mitotic chromosome condensation
definition:The cell cycle process in which chromatin structure is compacted
prior to and during mitosis in eukaryotic cells. {database_cross_reference:
GOC:mah, . . . }
parents:chromosome condensation; mitotic cell cycle process
children:mitotic chromosome arm condensation; . . .
stated relations:part of -> mitotic cell cycle; . . .
database_cross_reference:Reactome:R-HSA-2299718 {rdfs:label: Condensation
of Prophase Chromosomes}; . . .
has_obo_namespace:biological_process
id:GO:0007076
4.2.2 Name Linking.For a SPARQL query generated by OA-RAG
to be executable by the ontology service, every concept and relation
name in the query must be linked to an ontology entity. We first
attempt exact matching; if this fails, we retrieve candidate entities
for OA-RAG to inspect and use for query repair. The procedure is
as follows:
•Exact match.We first normalise each name (e.g., by standardising
case, whitespace, and underscores) and match it against every
given alternative name. If every name maps uniquely to an on-
tology entity, the query is executed directly. If a name matches
multiple entities, it is marked as ambiguous, and the candidate
entities are returned for further disambiguation. If a name cannot
be matched to any ontology entity, we proceed to the fallback
step.
•Unknown-name fallback.For each concept or relation name that
cannot be matched exactly, we return the five closest ontology
names as candidate replacements. For concepts, we first retrieve
a pool of 200 candidate names using BM25 and then rerank them
by string similarity [ 41]. For relations, we match against the
relation vocabulary directly. Given these suggestions, OA-RAG
can revise the query using an appropriate ontology name or
reformulate the query if none of the candidates is suitable.
1𝐵is a direct parent of 𝐴ifO|=𝐴⊑𝐵 and there is no other concept 𝐵′such that
O|=𝐴⊑𝐵′⊑𝐵. Direct children are defined analogously.

OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning
5 Dataset Construction
We construct evaluation datasets for the two reasoning patterns
in which the ontology serves as the primary source of knowledge:
formal reasoning(§5.1) andontology-grounded LLM reasoning(§5.2).
For ontology-assisted LLM reasoning, we do not construct a new
dataset; instead, we reuse existing domain question answering
benchmarks following [29], with details deferred to Section 6.
5.1 Formal Reasoning
In the formal reasoning setting, we focus on the case where the an-
swer to a question can be computed directly by ontology reasoning.
Specifically, we consider two kinds of questions: (1) multiple-choice
questions (MCQ), where options are given and several of them may
be correct; and (2) open questions (OQ), which give no options, so
that the LLM has to find the answers in the ontology itself.
5.1.1 MCQ Construction.Our central idea is to construct ques-
tions directly from valid ontological subsumption statements by
masking one component of each statement and treating the masked
component as the prediction target. This process can be fully auto-
mated using an ontology reasoner. Specifically, given an entailed
axiom
O|=𝐴⊑∃𝑟.𝐵,
we construct a query by masking either a concept, namely 𝐴or
𝐵, or the relation 𝑟. The masked symbol determines the reference
answer, while additional concepts or relations are introduced as
candidate options.
First, we obtain the entailed axioms of the form O|=𝐴⊑∃𝑟.𝐵
by exhaustively applying the following decomposition rule:
𝐴⊑𝐶⊓∃𝑟.𝐵→𝐴⊑∃𝑟.𝐵, 𝐴⊑𝐶
to the asserted axioms in O. Here,𝐴and𝐵denote atomic concepts
and𝐶denotes an arbitrary, possibly complex, EL+-concept. Then,
we mask𝐴,𝑟, or𝐵to obtain questions. For each generated question,
we construct six candidate options as follows.
•When the masked answer is a concept, we randomly select 1–3
semantically valid concepts as correct options. If 𝐴is masked,
the correct options are sampled from 𝐴and its descendants; if 𝐵
is masked, they are sampled from 𝐵and its ancestors. To avoid
selecting concepts that are overly general or overly specific, we
restrict these candidates to those within a taxonomy distance of
at most two from 𝐴or𝐵, respectively. The remaining options are
sampled from sibling concepts of 𝐴or𝐵, with randomly sampled
concepts used when there are insufficient siblings.
•When the masked answer is a relation, the original relation
is used as the correct option, and five distractors are sampled
randomly from the other relations appearing in existential re-
strictions in the ontology.
Finally, to ensure the correctness of the constructed query–answer
pairs, we verify each candidate option with an ontology reasoner
and replace any incorrectly labeled option.
5.1.2 OQ Construction.We construct an OQ 𝑄by specifying 𝑛≥2
conditions and asking for concepts that satisfy all of them. We
restrict the search to the taxonomy branch rooted at a designatedExample 1: A generated multiple-choice item (GO, filler masked)
Source axiom.mesoderm formation⊑∃part of.mesoderm morphogenesis.
Question.Which concept completes this statement about “mesoderm formation”:
is part of [?]? Select the option or options that this ontology supports, and no option
that is merely plausible.
Options.
Amesoderm morphogenesisBtissue morphogenesis
Cmorphogenesis of an epitheliumDcartilage morphogenesis
Efibrous ring of heart morphogenesisFmuscle tissue morphogenesis
Gold.AandB: the asserted filler and its parenttissue morphogenesis
Example 2: A generated open question (GO,𝑛=2)
Concept.negative regulation of receptor signaling pathway via STAT
Question.Find one or more biological processes in this ontology that satisfy all
of these conditions: negatively regulates signaling; regulates cell surface receptor
signaling pathway via STAT.
Answer set.negative regulation of receptor signaling pathway via STAT;negative
regulation of receptor signaling pathway via JAK-STAT;negative regulation of
tyrosine phosphorylation of STAT protein.
general concept 𝐴0, that is, to concepts 𝐴such thatO|=𝐴⊑𝐴 0.2
An answer must additionally satisfy the relation restrictions ∃𝑟𝑖.𝐵𝑖,
for𝑖=1,...,𝑛. Thus,𝐴is an answer to𝑄if
O|=𝐴⊑𝐴 0⊓∃𝑟 1.𝐵1⊓···⊓∃𝑟 𝑛.𝐵𝑛.
We generate OQs by first selecting a concept 𝐴in the target
branch and then constructing query conditions that 𝐴is known
to satisfy. Specifically, for each concept 𝐴such thatO |=𝐴⊑
𝐴0, we first collect relation restrictions known to hold for 𝐴, and
then sample 𝑛mutually non-subsuming restrictions as the query
conditions. Since all selected conditions hold for 𝐴, the resulting
OQ is guaranteed to have at least one answer. Appendix C gives
the details.
5.2 Ontology-Grounded LLM Reasoning
Ontology-grounded LLM reasoning targets questions that can be
answered using information contained in the ontology but cannot
be solved by an ontology reasoner alone. In such cases, part of the
information required to answer the question is not represented as
formal axioms, but instead appears in other forms, such as anno-
tations of natural language comments and definitions. Answering
these questions therefore requires combining textual evidence with
the remaining logical structure of the ontology.
To construct a dataset for this setting, we reuse the same ques-
tions, options, gold answer sets, answer schemas, and sample identi-
fiers, while replacing the original ontology with a deterministically
revised version, as described below. The key idea is to move part
of the information encoded in logical axioms into the concepts’
natural-language definitions. As a result, the revised ontology still
contains sufficient information to answer the questions when both
its textual and logical content are considered, while an ontology
reasoner operating only on the remaining formal axioms can no
longer recover all required information.
The ontology is revised as follows. For every asserted axiom with
a named concept on the left-hand side, we remove its existential
2For example, 𝐴0isbiological processin GO,food productin FoodOn, andClinical
findingin SNOMED CT.

Yang et al.
Table 1: Overall accuracy (%). Average: mean across the three tests after within-test averaging.†: from OntoTune [ 29]. (262k):
full 262k-token context. Red / blue: best / second best.
MethodKnowledge
SourceTest 1
(formal)Test 2
(ontology-grounded)Test 3
(ontology-assisted) Average
ont docMCQ OQ MCQ OQ PubMedQA MedQA MedMCQA USMLE 1 USMLE 2 USMLE 3
Base LLMs
Llama-3-8B-Instruct – – 45.1 7.8 45.4 8.1 70.3†51.7†51.7†57.4†52.3†58.2†36.7
Qwen3.5-9B – – 56.4 8.3 56.7 8.6 72.2 78.3 68.1 86.2 82.6 89.3 48.2
Ontology Fine-Tuned LLMs (Backbone: Llama-3-8B-Instruct)
OntoTune-sft – – 44.4 11.2 44.7 11.5 72.0†51.5†56.7†57.4†54.1†60.7†38.2
OntoTune-dpo – – 46.4 6.5 46.8 6.7 65.5†53.3†57.2†58.5†51.4†59.0†36.9
OntoTune-sft+dpo – – 45.6 9.8 45.9 10.1 66.3†51.9†56.7†53.2†54.1†63.1†37.8
TaxoLLaMA∗– – – – – – 73.4†50.5†46.1†42.6†39.4†47.5†–
Prompting and RAG Methods (Backbone: Llama-3-8B-Instruct)
CoT – – 38.2 3.0 38.5 3.1 67.7 55.6 55.3 64.9 64.2 64.8 34.5
RAG ont(text)✓– 50.9 3.0 53.4 4.6 65.5 51.7 51.8 56.4 56.9 57.4 37.5
RAG ont(query)✓– 45.3 10.0 42.6 2.4 64.8 55.6 54.5 62.8 57.8 63.9 36.7
RAG doc(text) –✓43.0 2.8 43.3 2.9 61.6 54.0 55.5 56.4 55.0 55.7 34.1
MedRAG doc –✓40.0 5.7 40.3 5.8 77.0 55.1 55.7 64.9 55.0 59.0 35.7
ReAct ont ✓– 45.0 9.8 48.1 12.5 67.6 54.8 53.3 63.8 51.4 63.9 38.9
i-MedRAG doc –✓38.7 0.5 38.8 0.5 66.0 61.6 56.2 78.7 64.2 61.5 34.6
OA-RAG ✓ – 62.0 45.0 52.2 18.0 71.6 57.6 57.4 66.0 56.9 67.2 50.5
Prompting and RAG Methods (Backbone: Qwen3.5-9B)
CoT – – 61.1 12.7 61.4 13.0 76.4 82.4 69.0 89.4 87.2 90.2 52.2
RAG ont(text)✓– 68.7 18.2 69.1 20.1 72.6 84.4 70.4 91.5 82.6 89.3 56.6
RAG ont(query)✓– 68.2 26.2 56.9 5.8 75.5 84.8 71.2 91.5 88.1 86.9 53.8
RAG doc(text) –✓61.0 5.3 61.3 5.3 74.5 82.6 71.5 91.5 81.7 84.4 49.1
MedRAG doc –✓58.9 0.5 59.2 0.5 79.3 81.1 70.5 94.7 79.8 82.8 47.0
ReAct ont ✓– 69.2 31.5 65.3 32.9 75.4 84.7 72.1 91.5 88.1 86.9 60.9
i-MedRAG doc –✓40.1 0.5 40.4 0.5 71.2 69.8 66.4 73.4 65.1 54.1 35.8
i-MedRAG doc(262k) –✓40.9 1.0 41.2 1.0 77.2 88.6 72.9 94.7 91.7 86.1 42.4
OA-RAG ✓ – 79.9 62.0 73.7 23.2 77.0 86.3 71.7 94.7 91.7 87.7 68.1
restrictions from the logical layer. Specifically,
𝐴⊲ ⊳𝐷⊓.𝑚
𝑖=1∃𝑟𝑖.𝐶𝑖with⊲ ⊳∈{⊑,≡},↦−→𝐴⊑𝐷.
If no logical content remains, i.e., 𝐷=⊤ , the revised axiom is
removed entirely. Each removed restriction ∃𝑟𝑖.𝐶𝑖is then ver-
balised [ 16] and appended to the annotation definition of 𝐴(a
definition is added if 𝐴has none). We further verify that each
question from Section 5.1 is answerable from the revised axioms
combined with the informal annotations, but not from the axioms
alone, and exclude 23 questions that fail these criteria (detailed in
Appendix C).
6 Evaluation
6.1 Experimental Setup
Benchmarks.We apply the procedure of Section 5 to three widely
used ontologies that differ in scale, terminology, and modeling con-
ventions: GO [ 1], FoodOn [ 8], and SNOMED CT [ 7], and created 300
MCQs and 200 OQs per ontology (see Appendix C for details).Test 1(formal reasoning) asks these 1,500 questions against the source on-
tologies.Test 2(ontology-grounded LLM reasoning) ask 1,477 ques-
tions against the revised ontologies.Test 3(ontology-assisted LLM
reasoning) follows OntoTune [ 29]: 6,781 single-answer test ques-
tions from MedQA [ 22], USMLE Step 1/2/3 [ 12], MedMCQA [ 36],
and PubMedQA [ 23], with SNOMED CT as the underlying ontology.
Baselines.Besides the base LLMs, Llama-3-8B-Instruct [ 11] and
Qwen3.5-9B [ 40] (hereafter Llama and Qwen, respectively), we
evaluate the following two categories of baselines:
•Ontology fine-tuned LLMs.We include the three released Onto-
Tune variants (sft, dpo, and sft+dpo) and TaxoLLaMA∗, Onto-
Tune’s non-public re-implementation of TaxoLLaMA [ 33]; all
four are fine-tuned to integrate SNOMED CT knowledge into
Llama, and the Test 3 results of the Llama-based models (marked
with†) are quoted from OntoTune.
•Prompting and retrieval methods.We consider standard chain-
of-thought (CoT) prompting [ 53] and four retrieval-augmented
methods: RAG, ReAct [ 60], MedRAG [ 55], and i-MedRAG [ 56].
We consider two retrieval sources:(1) ont, the ontology concept
cards introduced in Section 4.2.1, and(2) doc, comprising the

OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning
(a) Grounding level percentage (%) (b) Answer accuracy (%)entailed supported related unrelated entailed supported related unrelated
0 25 50 75 100Llama Llama
Test 1
Test 2
Test 3Test 1
Test 2
Test 361.8 (m=1251) 24.8 (m=117) 7.7 (m=13) 0.0 (m=1)
45.3 (m=811) 29.6 (m=402) 5.4 (m=37) 50.0 (m=2)
61.3 (m=2843) 58.2 (m=1408) 45.5 (m=44) 52.6 (m=19)
Qwen Qwen
Test 1
Test 2
Test 3Test 1
Test 2
Test 383.0 (m=1257) 34.5 (m=116) 0.0 (m=19) 33.3 (m=3)
75.1 (m=924) 40.7 (m=209) 0.0 (m=133) 0.0 (m=23)
94.9 (m=59) 82.1 (m=2630) 54.3 (m=35) 66.4 (m=738)
Figure 3: Grounding levels reported by OA-RAG. (a) Percentage of all questions at each level; note that a part of answers have
no grounding levels. (b) Answer accuracy within each grounding level;𝑚is the number of corresponding questions.
medical textbooks [ 22] and StatPearls [ 47] used by MedRAG and
i-MedRAG. For RAG, we evaluate both docandontas retrieval
sources with textual retrieval, RAG doc(text) and RAG ont(text),
as well as a structured query variant, RAG ont(query), which
replaces textual retrieval with a single SPARQL query as in OA-
RAG. ReAct ontuses ont, while MedRAG docand i-MedRAG doc
usedoc. In particular, i-MedRAG can be viewed as a medically
specialized variant of ReAct. Implementation details are provided
in Appendices A and B.
For a fair comparison, we use an 8,192-token context window for
all methods, matching the maximum supported by Llama. We also
report i-MedRAG docon Qwen with a 262k-token window, as its
performance degrades markedly under the shorter context.
Metrics.We report accuracy (%) of the all tested questions. An
MCQ question is correct if the predicted set of options exactly
matches the gold set. An OQ is correct if the response is non-empty
and entirely correct.
6.2 Main Results
The overall results are shown in Table 1. For methods that do not use
the ontology, the Test 2 results are directly reused from Test 1. The
slight performance differences are due to the 23 excluded questions
described in Section 5.2.
OA-RAG achieves the best average accuracy with both backbones
(50.5% and 68.1%). OnTests 1 (formal reasoning)andTests 2
(ontology-grounded LLM reasoning), OA-RAG outperforms all
baselines in every setting except two: the Test 2 MCQs with Llama
(RAG ont(text), 53.4% vs. 52.2%) and the Test 2 OQs with Qwen
(ReAct ont, 32.9% vs. 23.2%). This is consistent with the fact that
these tasks are constructed directly from ontologies and require
reasoning over explicit ontological knowledge. Notably, two of
the ontology fine-tuning methods, OntoTune-sft and OntoTune-
sft+dpo, slightly improve over their base model on the OQs (e.g.,
from 7.8% to 11.2% on Test 1 for OntoTune-sft), indicating that
ontology knowledge can be partially internalised through fine-
tuning. In contrast, medical RAG methods such as MedRAG doc
and i-MedRAG docfall below the base model on the OQs with both
backbones, and on the MCQs too, except for MedRAG docwith
Qwen. These results suggest that existing medical RAG methods
may struggle when questions require explicit logical reasoning and
to be answered with high faithfulness to the ontology rather than
primarily retrieving relevant evidence.OnTest 3 (ontology-assisted LLM reasoning ), OA-RAG re-
mains competitive with the best-performing baseline, i-MedRAG doc.
However, with Qwen, i-MedRAG doc(262k) requires a substantially
larger retrieval and generation budget (about4 ×the LLM calls and
10×the tokens per question of OA-RAG: 22 vs. 5.8 calls and 96.7k
vs. 9.7k tokens, counting the input and generated tokens of all LLM
calls of a question). Most of the difference lies in the retrieved text
that the model reads, which is about66 ×larger for i-MedRAG doc,
and i-MedRAG docalso generates about5 ×as many tokens (Appen-
dix A). Moreover, when restricted to the same 8,192-token context
window as OA-RAG, its average Test 3 accuracy drops markedly
to 66.7%, compared with 84.9% for OA-RAG. We further find that
RAG ont(text) outperforms RAG doc(text) on Test 3 with both back-
bones, and outperforms MedRAG docwith Qwen. These findings
highlight the potential of ontologies as retrieval sources: compared
with medical textbooks, ontologies provide more concise and struc-
tured knowledge, which may also benefit real-world QA tasks in
related domains.
Beyond answer accuracy, OA-RAG reports thegrounding level
of each answer (§4.1.3).Figure 3ashows the distribution of ground-
ing levels in the final outputs. Because some answers lack grounding
annotations (e.g., when no evidence is available), the percentages
do not sum to 100%. Entailed and supported answers dominate,
accounting for 70.0% of all questions for Llama and 53.2% for Qwen,
whereas unrelated answers are rare, comprising only 0.2% for Llama
and 7.8% for Qwen.Figure 3bshows that, among answers labeled
entailed,supported, orrelated, stronger reported grounding is gen-
erally associated with higher accuracy. For example, with Qwen
on Test 3, 94.9% ofentailedand 82.1% ofsupportedanswers are
correct (together accounting for 39.7% of the questions), compared
with 72.1% of the remaining answers. The reported grounding level
therefore provides a useful signal of answer reliability beyond the
answer itself. Note thatunrelatedanswers are unsuitable for di-
rect comparison, as they tend to arise from questions that are less
closely aligned with the scope of the ontology. For example, some
may depend primarily on reading comprehension or calculation
rather than ontology-based reasoning, making their accuracy more
reflective of the LLM’s general capabilities.
6.3 Analysis
6.3.1 Evidence Retrieval and Selection.Unlike the grounding level,
thegold evidencecan be traced directly from the ontology for Tests 1
and 2, allowing us to evaluate exactly whether the required evidence

Yang et al.
Gold evidence retrieved: completely partly not at all
Retrieved evidence output:fully not fullyLlamaMCQ 64.1 29.6 31.7 13.3 55.0
OQ 36.8 8.0 55.2 99.0QwenMCQ 81.3 14.1 69.6 18.7 11.7
OQ 47.2 22.5 30.3 9.324.7 66.0
Test 1(formal reasoning) Test 2(ontology-grounded LLM reasoning)
Figure 4: Ratio of questions whose gold evidence is retrieved
and selected (%).
is retrieved and subsequently cited (Figure 4). The results suggest
that retrieval, rather than reasoning over retrieved evidence, is
the primary bottleneck. MCQs achieve a substantially higher rate
of fully retrieved gold evidence than OQs (61.7 vs. 23.6% over all
items), which is reasonable as the candidate answer options provide
additional cues that constrain retrieval. Moreover, once the relevant
evidence is retrieved, it is usually among the observations cited for
the answer (95.1–98.4% of the MCQs and at least 77.8% of the OQs).
Qwen also demonstrates a substantially stronger ability than Llama
to retrieve useful evidence. Nevertheless, both models perform
poorly on Test 2 OQs. This is expected because much of the relevant
information in Test 2 is encoded in definitions and is therefore less
directly accessible to the retrieval process. The absence of candidate
options in OQs further increases the difficulty of identifying the
relevant evidence.
6.3.2 Ablation Study.To analyse the contribution of each compo-
nent, we report an ablation study in Table 2. Across both backbones,
adding the ReAct loop and the final answering step consistently
improves over pre-tool reasoning alone, with the largest gains on
the first two test patterns (59.7 vs. 36.5% on Tests 1–2, with Qwen).
This highlights the importance of ontology-grounded reasoning
through SPARQL queries when the answer depends on structured
ontology knowledge. The finalanswering stepprovides a further
consistent gain (68.1 vs. 65.2% on average, with Qwen), showing
that synthesizing the intermediate reasoning and retrieved evidence
remains important even after tool use. Thequery fallbackexhibits
a more nuanced trade-off. While it improves overall performance
(68.1 vs. 67.7% on average, with Qwen), removing it can benefit
some settings, such as Test 1 MCQs, likely because doing so avoids
introducing less targeted retrieval when the structured ontology is
already sufficient.
6.3.3 Case Study.Figure 5 illustrates a common correction pat-
tern: the model’s pre-tool answer is plausible but based on a loose
association, whereas the ontology provides an explicit relation that
directly supports the correct option. In case (a), the model attributes
the liver injury to hemolysis after the transfusion, whereas the on-
tology describesHalothane hepatitisas a drug-induced hepatitis
whose causative agent is halothane, leading to A. In case (b), the
model associatesblind staggerswith molybdenum, while the on-
tology identifies it as an alternative name forSelenium poisoning
and directly links the condition to selenium, leading to C. These
cases show that the tool phase is most useful when the ontologyTable 2: Ablation of OA-RAG (accuracy %): each indented row
changes one component; ↓: drop of percentage points over
the original OA-RAG.
VariantTest 1 Test 2Test 3 Average
MCQ OQ MCQ OQ
Backbone: Llama-3-8B-Instruct
OA-RAG 62.0 45.0 52.2 18.0 62.8 50.5
pre-tool reasoning only 43.4 4.8 43.8 4.5 61.9 36.7↓13.8
w/o answering step 57.3 38.7 48.0 7.9 58.1 44.7↓5.8
w/o query fallback 62.1 42.7 52.6 9.3 62.8 48.7↓1.8
Backbone: Qwen3.5-9B
OA-RAG 79.9 62.0 73.7 23.2 84.9 68.1
pre-tool reasoning only 63.6 9.2 63.9 9.4 84.7 52.6↓15.5
w/o answering step 78.0 54.0 73.3 16.5 84.7 65.2↓2.9
w/o query fallback 80.4 61.7 75.8 17.8 85.1 67.7↓0.4
(a) The ontology names halothane as the causative agent MedQA
Q:Five days after undergoing an emergency appendectomy under general inhalational
anesthesia . . . she received a transfusion of 1 unit of packed red blood cells . . . Examination
shows jaundice . . . Eosinophils 13% . . . A biopsy of the liver shows massive centrilobular
necrosis. Which of the following is the most likely underlying cause . . . ?
(A) Adverse effect of anesthetic(B) Gram negative bacteria in the bloodstream
(C) Trauma to the bile duct (D) Acalculous inflammation of the gallbladder
(E) Excessive lysis of red blood cells
Pre-tool:“. . . The hemolysis releases free hemoglobin, which is toxic to the liver . . . ”⇒E✗
Tool phase(call 4 of 4):
4.DESCRIBE "Halothane hepatitis"
→Halothane hepatitis . . . parents: Drug-induced hepatitis; Poisoning
caused by halothane . . . stated relations: . . . ; Causative agent ->
Halothane; Finding site -> Liver structure
Answer:“. . . is diagnostic of Halothane hepatitis, which is an adverse effect of the anesthetic” ⇒
A✓
(b) The ontology maps blind staggers to selenium MedMCQA
Q:Poisoning by which of these elements causes blind staggers alkali disease in livestock?
(A) Magnesium (B) Manganese
(C) Selenium(D) Molybdenum
Pre-tool:“. . . Molybdenum: Toxicity (or high levels relative to sulfur/copper) is the direct cause
of the condition described as alkali disease leading to blind staggers”⇒D✗
Tool phase(calls 1 and 3 of 3):
1.DESCRIBE "blind staggers"
→Selenium poisoning . . . stated relations: Causative agent -> Selenium;
skos:altLabel: Alkali disease; Blind staggers
3.ASK { "blind staggers" "Causative agent" "Selenium" }
→every Selenium poisoning is related by Causative agent to some Selenium:
ENTAILED
Answer:“. . . The provided observations explicitly link ‘Selenium poisoning’ with the alternative
labels ‘Alkali disease’ and ‘Blind staggers’ and identify Selenium as the causative agent” ⇒C✓
Figure 5: Two Test 3 trajectories of OA-RAG (Qwen), with
intermediate reasoning omitted for brevity.
contains a direct, question-relevant relation that can override an
incorrect model association.
7 Conclusion
We presented OA-RAG, which, to our knowledge, is the first agen-
tic RAG framework that combines LLM reasoning with formal
OWL ontology reasoning. It can adaptively address three distinct

OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning
reasoning patterns: formal reasoning, ontology-grounded LLM rea-
soning, and ontology-assisted LLM reasoning. Experiments across
all three patterns with QA over three large-scale real-world ontolo-
gies demonstrate its strong and consistent performance. They also
show that ontologies provide an effective and efficient knowledge
resource, with competitive performance but substantially fewer
input and generated tokens in comparison with textual resources.
For future work, we plan to extend OA-RAG toward agentic web
that coordinates knowledge across multiple ontologies, supporting
more complex questions that require reasoning across domains.
References
[1] Michael Ashburner, Catherine A. Ball, Judith A. Blake, David Botstein, Heather
Butler, J. Michael Cherry, et al .2000. Gene Ontology: Tool for the Unification of
Biology.Nature Genetics25, 1 (2000), 25–29. doi:10.1038/75556
[2] Franz Baader, Ian Horrocks, Carsten Lutz, and Ulrike Sattler. 2017.An Introduction
to Description Logic. Cambridge University Press.
[3] Franz Baader, Ian Horrocks, Carsten Lutz, Uli Sattler, et al .2017.An introduction
to description logic. Vol. 24. Cambridge University Press Cambridge.
[4] Hamed Babaei Giglou, Jennifer D’Souza, and Sören Auer. 2023. LLMs4OL: Large
Language Models for Ontology Learning. InISWC. 408–427.
[5]James P. Balhoff and Christopher J. Mungall. 2024. Whelk: An OWL EL+RL
Reasoner Enabling New Use Cases.Transactions on Graph Data and Knowledge
2, 2 (2024), 7:1–7:17. doi:10.4230/TGDK.2.2.7
[6] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher. 2009. Reciprocal
Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In
SIGIR. 758–759. doi:10.1145/1571941.1572114
[7] Kevin Donnelly. 2006. SNOMED-CT: The Advanced Terminology and Coding
System for eHealth.Studies in Health Technology and Informatics121 (2006),
279–290.
[8] Damion M. Dooley, Emma J. Griffiths, Gurinder S. Gosal, Pier L. Buttigieg, Robert
Hoehndorf, Matthew C. Lange, Lynn M. Schriml, Fiona S. L. Brinkman, and
William W. L. Hsiao. 2018. FoodOn: A Harmonized Food Ontology to Increase
Global Food Traceability, Quality Control and Data Integration.npj Science of
Food2 (2018), 23. doi:10.1038/s41538-018-0032-6
[9] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Meng Wang, and Haofen Wang. 2023. Retrieval-Augmented Genera-
tion for Large Language Models: A Survey. arXiv:2312.10997.
[10] Birte Glimm, Ian Horrocks, Boris Motik, Giorgos Stoilos, and Zhe Wang. 2014.
HermiT: An OWL 2 Reasoner.Journal of Automated Reasoning53, 3 (2014),
245–269. doi:10.1007/s10817-014-9305-1
[11] Aaron Grattafiori, Abhimanyu Dubey, et al. 2024. The Llama 3 Herd of Models.
arXiv:2407.21783.
[12] Tianyu Han, Lisa C. Adams, Jens-Michalis Papaioannou, Paul Grundmann,
Tom Oberhauser, Alexei Figueroa, Alexander Löser, Daniel Truhn, and Keno K.
Bressem. 2023. MedAlpaca – An Open-Source Collection of Medical Conversa-
tional AI Models and Training Data. arXiv:2304.08247.
[13] Steve Harris and Andy Seaborne. 2013. SPARQL 1.1 Query Language. W3C
Recommendation, 21 March 2013. https://www.w3.org/TR/sparql11-query/.
[14] Yuan He, Jiaoyan Chen, Denvar Antonyrajah, and Ian Horrocks. 2022. BERTMap:
A BERT-Based Ontology Alignment System. InAAAI. 5684–5691.
[15] Yuan He, Jiaoyan Chen, Hang Dong, Ian Horrocks, Carlo Allocca, Taehun Kim,
and Brahmananda Sapkota. 2024. DeepOnto: A Python Package for Ontology
Engineering with Deep Learning.Semantic Web15, 5 (2024), 1991–2004.
[16] Yuan He, Jiaoyan Chen, Ernesto Jiménez-Ruiz, Hang Dong, and Ian Horrocks.
2023. Language Model Analysis for Ontology Subsumption Inference. InFindings
of ACL. 3439–3453. doi:10.18653/v1/2023.findings-acl.213
[17] Yuan He, Zhangdie Yuan, Jiaoyan Chen, and Ian Horrocks. 2024. Language
Models as Hierarchy Encoders. InNeurIPS, Vol. 37. 14690–14711.
[18] Ian Horrocks. 2008. Ontologies and the Semantic Web.Commun. ACM51, 12
(2008), 58–67. doi:10.1145/1409360.1409377
[19] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A Survey on Hallucination in Large Language Models: Principles,
Taxonomy, Challenges, and Open Questions.ACM Transactions on Information
Systems43, 2 (2025), 1–55. doi:10.1145/3703155
[20] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination
in Natural Language Generation.Comput. Surveys55, 12 (2023), 1–38.
[21] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. InEMNLP. 7969–7992.[22] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter
Szolovits. 2021. What Disease Does This Patient Have? A Large-Scale Open
Domain Question Answering Dataset from Medical Exams.Applied Sciences11,
14 (2021), 6421. doi:10.3390/app11146421
[23] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu.
2019. PubMedQA: A Dataset for Biomedical Research Question Answering. In
EMNLP-IJCNLP. 2567–2577. doi:10.18653/v1/D19-1259
[24] Qiao Jin, Won Kim, Qingyu Chen, Donald C. Comeau, Lana Yeganova, W. John
Wilbur, and Zhiyong Lu. 2023. MedCPT: Contrastive Pre-trained Transformers
with Large-scale PubMed Search Logs for Zero-shot Biomedical Information
Retrieval.Bioinformatics39, 11 (2023), btad651.
[25] Yevgeny Kazakov, Markus Krötzsch, and František Simančík. 2014. The Incredible
ELK: From Polynomial Procedures to Efficient Reasoning with EL Ontologies.
Journal of Automated Reasoning53, 1 (2014), 1–61. doi:10.1007/s10817-013-9296-3
[26] Yunshi Lan, Gaole He, Jinhao Jiang, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen.
2021. A Survey on Complex Knowledge Base Question Answering: Methods,
Challenges and Solutions. InIJCAI. 4483–4491. doi:10.24963/ijcai.2021/611
[27] Tianle Li, Xueguang Ma, Alex Zhuang, Yu Gu, Yu Su, and Wenhu Chen. 2023.
Few-shot In-context Learning on Knowledge Base Question Answering. InACL.
6966–6980. doi:10.18653/v1/2023.acl-long.385
[28] Runxuan Liu, Bei Luo, Jiaqi Li, Baoxin Wang, Ming Liu, Dayong Wu, Shijin
Wang, and Bing Qin. 2025. Ontology-Guided Reverse Thinking Makes Large
Language Models Stronger on Knowledge Graph Question Answering. InACL.
15269–15284.
[29] Zhiqiang Liu, Chengtao Gan, Junjie Wang, Yichi Zhang, Zhongpu Bo, Mengshu
Sun, Huajun Chen, and Wen Zhang. 2025. OntoTune: Ontology-Driven Self-
training for Aligning Large Language Models. InWWW. 119–133.
[30] Haoran Luo, Haihong E, Zichen Tang, Shiyao Peng, Yikai Guo, Wentai Zhang,
Chenghao Ma, Guanting Dong, Meina Song, Wei Lin, Yifan Zhu, and Anh Tuan
Luu. 2024. ChatKBQA: A Generate-then-Retrieve Framework for Knowledge
Base Question Answering with Fine-tuned Large Language Models. InFindings
of ACL. 2039–2056. doi:10.18653/v1/2024.findings-acl.122
[31] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning
on Graphs: Faithful and Interpretable Large Language Model Reasoning. InICLR.
[32] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation. In
ICLR.
[33] Viktor Moskvoretskii, Ekaterina Neminova, Alina Lobanova, Alexander
Panchenko, and Irina Nikishina. 2024. TaxoLLaMA: WordNet-based Model
for Solving Multiple Lexical Semantic Tasks. InACL. 2331–2350.
[34] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh
Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori
Hashimoto. 2025. s1: Simple Test-Time Scaling. InEMNLP. 20275–20321.
[35] Theo Olausson, Alex Gu, Ben Lipkin, Cedegao Zhang, Armando Solar-Lezama,
Joshua Tenenbaum, and Roger Levy. 2023. LINC: A Neurosymbolic Approach
for Logical Reasoning by Combining Language Models with First-Order Logic
Provers. InEMNLP. 5153–5176. doi:10.18653/v1/2023.emnlp-main.313
[36] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. 2022.
MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical Do-
main Question Answering. InCHIL. 248–260.
[37] Liangming Pan, Alon Albalak, Xinyi Wang, and William Wang. 2023. Logic-LM:
Empowering Large Language Models with Symbolic Solvers for Faithful Logical
Reasoning. InFindings of EMNLP. 3806–3824.
[38] Thomas Pellissier Tanon. 2023. Oxigraph: a graph database implementing the
SPARQL standard. doi:10.5281/zenodo.7408022 Software, https://oxigraph.org.
[39] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni
Zeng, Xuanhe Zhou, Yufei Huang, Chaojun Xiao, et al .2025. Tool Learning with
Foundation Models.Comput. Surveys57, 4 (2025), 1–40. doi:10.1145/3704435
[40] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. https://qwen.
ai/blog?id=qwen3.5.
[41] John W. Ratcliff and David E. Metzener. 1988. Pattern Matching: The Gestalt
Approach.Dr. Dobb’s Journal13, 7 (1988), 46.
[42] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. InEMNLP-IJCNLP. 3982–3992.
[43] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends in Information Retrieval3, 4
(2009), 333–389. doi:10.1561/1500000019
[44] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli,
Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language Models Can Teach Themselves to Use Tools. InNeurIPS.
[45] Kartik Sharma, Peeyush Kumar, and Yunqing Li. 2025. OG-RAG: Ontology-
Grounded Retrieval-Augmented Generation for Large Language Models. In
EMNLP. 32962–32981. doi:10.18653/v1/2025.emnlp-main.1674
[46] Barry Smith, Michael Ashburner, Cornelius Rosse, Jonathan Bard, William Bug,
Werner Ceusters, Louis J. Goldberg, Karen Eilbeck, Amelia Ireland, Christopher J.
Mungall, et al .2007. The OBO Foundry: Coordinated Evolution of Ontologies
to Support Biomedical Data Integration.Nature Biotechnology25, 11 (2007),

Yang et al.
1251–1255. doi:10.1038/nbt1346
[47] StatPearls Publishing. 2026. StatPearls. NCBI Bookshelf NBK430685, https:
//www.ncbi.nlm.nih.gov/books/NBK430685/. Archive of 27 September 2026.
[48] Xiaorui Su, Yibo Wang, Shanghua Gao, Xiaolong Liu, Valentina Giunchiglia,
Djork-Arné Clevert, and Marinka Zitnik. 2025. KGARevion: An AI Agent for
Knowledge-Intensive Biomedical QA. InICLR.
[49] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph:
Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.
InICLR.
[50] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InACL. 10014–10037.
[51] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou.
2020. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression
of Pre-Trained Transformers. InNeurIPS, Vol. 33. 5776–5788.
[52] Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent
Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee,
and Tomas Pfister. 2024. Chain-of-Table: Evolving Tables in the Reasoning Chain
for Table Understanding. InICLR.
[53] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei
Xia, Ed Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-Thought Prompting
Elicits Reasoning in Large Language Models. InNeurIPS.
[54] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina,
Yueming Jin, and Vicente Grau. 2025. Medical Graph RAG: Evidence-based
Medical Large Language Model via Graph Retrieval-Augmented Generation. In
ACL. 28443–28467.
[55] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. 2024. Benchmarking
Retrieval-Augmented Generation for Medicine. InFindings of ACL. 6233–6251.
[56] Guangzhi Xiong, Qiao Jin, Xiao Wang, Minjia Zhang, Zhiyong Lu, and Aidong
Zhang. 2024. Improving Retrieval-Augmented Generation in Medicine with
Iterative Follow-up Questions. InPSB. 199–214. doi:10.1142/9789819807024_0015
[57] Hui Yang, Jiaoyan Chen, Yuan He, Yongsheng Gao, and Ian Horrocks. 2025.
Language Models as Ontology Encoders. InISWC. 443–461.
[58] Hui Yang, Jiaoyan Chen, Yiping Song, Renate Schmidt, and Wen Zhang. 2026.
NeurOWL: An LLM-Based Neural-symbolic Framework for Incomplete OWL
Ontology Reasoning. arXiv:2607.15776.
[59] Rui Yang, Haoran Liu, Edison Marrese-Taylor, Qingcheng Zeng, Yuhe Ke, Wanxin
Li, Lechao Cheng, Qingyu Chen, James Caverlee, Yutaka Matsuo, and Irene
Li. 2024. KG-Rank: Enhancing Large Language Models for Medical QA with
Knowledge Graphs and Ranking Techniques. InBioNLP Workshop. 155–166.
[60] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. InICLR.
[61] Xiao Zhang, Huiyuan Lai, Qianru Meng, and Johan Bos. 2026. OntoURL: A
Benchmark for Evaluating Large Language Models on Symbolic Ontological
Understanding, Reasoning and Learning.Journal of Web Semantics91 (2026),
100886. doi:10.1016/j.websem.2026.100886
A Implementation Details
Agent and service.For each item, OA-RAG allows up to 10 tool
actions and 5,000 generated tokens across pre-tool reasoning, the
ReAct loop, and Step 1 of answering, plus 512 tokens for Step 2.
The ontology service decides ASKand SELECT queries over the
entailment closure of the ontology, precomputed with the OWL EL
reasoner Whelk [ 5] and stored in Oxigraph [ 38]; the datasets are
built and checked with ELK [25] and HermiT [10].
Cost.On Test 3 with Qwen, OA-RAG uses 9.7k tokens per ques-
tion over 5.8 LLM calls on average, including 7.9k input tokens
(1.0k from ontology observations) and 1.8k generated tokens. In
contrast, i-MedRAG doc(262k) uses 96.7k tokens over 22.0 calls, in-
cluding 87.8k input tokens (63.6k from retrieved documents) and
8.9k generated tokens.
Tool calls(Figure 6). Query choice varies systematically with
question type and model. On the OQs of Tests 1 and 2, Llama
relies almost exclusively on SELECT (97% of calls), whereas Qwen
primarily uses DESCRIBE (56%). On the MCQs, Llama divides its calls
between ASKandSELECT (40–45% each), while Qwen again favors
DESCRIBE (49–53%). On Test 3, Llama averages 4.9 calls per question,split evenly between ASKandSELECT (40 and 42%), whereas Qwen
averages 2.0 calls and usesDESCRIBEfor 84% of them.
B Retrieval and Document Baselines
The retrieval baselines differ primarily in their knowledge source:
RAG ont(text), ReAct ont, and the ablation search tool retrieve on-
tology concept cards, whereas RAG doc(text), MedRAG doc, and i-
MedRAG docretrieve medical documents.
Ontology retrieval.Each field of a concept card (§4.2.1)
forms a retrieval unit, grouped by type (names, definitions, etc.).
Units are retrieved using BM25 [ 43] and sentence embeddings
(all-MiniLM-L6-v2 [42,51]), with the resulting rankings com-
bined by reciprocal rank fusion [ 6]; each concept is scored according
to its highest-ranked unit. RAG ont(text) performs a single retrieval
using the question and answer options, then inserts the top-5
concept cards together with the top-5 verbalized axioms retrieved
using the same scoring procedure. ReAct ontuses the same underly-
ing retrieval mechanism as RAG ont(text) but each search returns
the top-5 concepts, each with only the field that matched rather
than its complete concept card, followed by the top-5 matching
verbalized axioms. Unlike RAG ont(text), ReAct ontgenerates search
queries dynamically using the LLM. The search tool used in our
ablation study follows the same approach.
Document baselines.Following i-MedRAG, we use Med-
CPT [ 24] to retrieve from medical textbooks [ 22] and StatPearls [ 47].
RAG doc(text) replaces ontology cards with the top-5 retrieved snip-
pets. Under their standard settings, MedRAG doc[55] answers from
32 snippets in a single call, while i-MedRAG doc[56] issues three
follow-up queries in each of four rounds. Both are run with the
authors’ toolkit using a 7,168-token document budget, with a forced
answer when the response reaches the context-window limit [ 34].
Results by source ontology(Table 3). OA-RAG achieves the
best performance in 19 of the 24 settings, one of them tied. In the
other five, it falls behind RAG ont(text) (with Llama on the Test 2 GO
and FoodOn MCQs, 53.7 vs. 55.7% and 46.8 vs. 51.8%) or ReAct ont
(on three Test 2 OQ settings); the largest gap is on the Test 2 GO OQs
with Qwen (22.0 vs. 42.0%), mainly because Qwen’s exact queries
rarely retrieve a correct concept there (for 15 of the 200 questions,
vs. 106 on Test 1), so it gives no answer to 62.5% of them.
DESCRIBE ASK SELECTLlama QwenTest 1 MCQ 3.6
Test 1 OQ 3.8
Test 2 MCQ 4.8
Test 2 OQ 4.9
Test 3 4.9
Test 1 MCQ 2.2
Test 1 OQ 4.5
Test 2 MCQ 3.3
Test 2 OQ 6.1
Test 3 2.0
0 2 4 6
Figure 6: Tool calls of OA-RAG per question by query form
(SELECT: all its variants). Test 3 pools its six sets.

OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning
Table 3: Test 1 and Test 2 accuracy (%) by source ontology. Red / blue: best / second best per model and test.
MethodTest 1 Test 2
GO FoodOn SNOMED CT GO FoodOn SNOMED CT
MCQ OQ MCQ OQ MCQ OQ MCQ OQ MCQ OQ MCQ OQ
Llama
Direct 50.7 5.0 33.7 12.0 51.0 6.5 50.7 5.0 33.8 12.4 51.9 6.9
CoT 47.7 1.0 32.0 6.5 35.0 1.5 47.7 1.0 32.1 6.7 35.6 1.6
RAG ont(text) 55.0 0.5 53.3 3.5 44.3 5.0 55.7 1.5 51.8 5.7 52.5 6.9
RAG ont(query) 52.7 13.0 41.0 4.0 42.3 13.0 48.3 1.5 39.8 3.1 39.7 2.6
RAG doc(text) 50.0 0.0 34.0 4.5 45.0 4.0 50.0 0.0 34.1 4.6 45.8 4.2
MedRAG doc 50.7 2.5 31.7 6.5 37.7 8.0 50.7 2.5 31.8 6.7 38.3 8.5
ReAct ont 50.0 10.0 44.7 8.0 40.3 11.5 54.3 13.0 43.8 13.4 46.1 11.1
i-MedRAG doc 46.7 0.0 31.7 1.0 37.7 0.5 46.7 0.0 31.8 1.0 38.0 0.5
OA-RAG 63.0 52.0 53.3 18.5 69.7 64.5 53.7 12.5 46.8 17.0 56.3 24.9
Qwen
Direct 56.7 4.5 51.7 14.0 61.0 6.5 56.7 4.5 51.8 14.4 61.7 6.9
CoT 65.3 13.0 57.0 12.0 61.0 13.0 65.3 13.0 57.2 12.4 61.7 13.8
RAG ont(text) 71.7 27.0 71.7 15.0 62.7 12.5 73.0 30.0 67.6 12.4 66.8 17.5
RAG ont(query) 70.3 31.0 61.0 11.0 73.3 36.5 60.0 4.0 53.8 8.2 56.9 5.3
RAG doc(text) 63.3 6.5 59.7 5.5 60.0 4.0 63.3 6.5 59.9 5.7 60.7 3.7
MedRAG doc 61.0 0.0 56.0 0.0 59.7 1.5 61.0 0.0 56.2 0.0 60.3 1.6
ReAct ont 72.7 39.0 71.3 32.0 63.7 23.5 66.7 42.0 61.5 30.9 67.8 25.4
i-MedRAG doc 47.3 0.5 29.7 0.0 43.3 1.0 47.3 0.5 29.8 0.0 44.1 1.1
i-MedRAG doc(262k) 51.0 1.0 27.3 0.0 44.3 2.0 51.0 1.0 27.4 0.0 45.1 2.1
OA-RAG 79.7 62.5 78.3 34.0 81.7 89.5 74.0 22.0 74.6 21.1 72.5 26.5
Llama Qwen at least one correct and no wrong
𝐵𝐴𝑟𝐵𝐴𝑟0255075100
Test 1 Test 2accuracy (%)MCQ:
masked slot
123 123
Test 1 Test 2MCQ:
#Ans
234 234
Test 1 Test 2OQ:
conditions𝑛
Figure 7: OA-RAG on Test 1 and Test 2 by item construction
(accuracy %; dashed: MCQs scored with at least one correct
and no wrong choice).
Table 4: Sizes of the ontologies (Test 1: source; Test 2: revised).
GO FoodOn SNOMED CT
Named concepts 51,986 30,995 374,546
Roles 9 69 126
Axioms (Test 1) 82,950 44,986 375,261
Axioms (Test 2) 65,810 36,881 375,261
C Benchmarks
Table 4summarizes the sizes of the three ontologies. We use the
GO release of 2026-07-26, the FoodOn release of 2022-08-12 (with
imports merged), and the SNOMED CT International Edition release
of 2025-07-01, converted to OWL using the SNOMED OWL Toolkit.
SNOMED CT is in the OWL 2 EL profile ( EL++). GO uses only
⊓and∃in its class expressions and lies in EL++except for one
inverse-object-property axiom relatingpart ofandhas part. FoodOn
additionally uses∀,¬,⊔, and cardinality restrictions, placing it in
SRIQ . The revised ontologies used in Test 2 preserve the signatureTable 5: Tests 1 (T1) and 2 (T2): MCQs by the number of correct
options (#Ans), and OQs by the number of conditions𝑛.
GO FoodOn SNOMED CT All
T1 T2 T1 T2 T1 T2 T1 T2
MCQ#Ans=1203 203 230 230 236 235 669 668
#Ans=258 58 43 43 39 39 140 140
#Ans=339 39 27 26 25 21 91 86
OQ𝑛=267 67 67 67 67 65 201 199
𝑛=367 67 67 65 67 61 201 193
𝑛=466 66 66 62 66 63 198 191
All questions 500 500 500 493 500 484 1,500 1,477
and use a subset of the same constructors.Table 5summarizes the
sizes of Tests 1 and 2. For each ontology, we construct 300 MCQs by
masking the filler, subject, or relation of 100 axioms each, together
with 200 OQs. Test 2 uses the same questions as Test 1, except for
23 questions that fail the checks described below.
OQ conditions.We construct the candidate OQ conditions as
follows. Let 𝑆(𝐴) contain the restrictions ∃𝑟.𝐵 that the MCQ de-
composition rule (§5.1) yields from the asserted axioms of 𝐴and its
ancestors, and let the filler be lifted by up to two taxonomy levels:
𝑆(𝐴)={∃𝑟.𝐵′:∃𝑟.𝐵∈𝑆(𝐴), 𝐵′∈{𝐵}∪Parent ≤2(𝐵)}.
A lifted restriction still holds for 𝐴, since𝐵⊑𝐵′. For each𝐴with
|𝑆(𝐴)|≥𝑛 , we sample 𝑛conditions from 𝑆(𝐴) and resample until
none subsumes another (O̸|=∃𝑟 𝑖.𝐵′
𝑖⊑∃𝑟𝑗.𝐵′
𝑗for all𝑖≠𝑗).
Test 2 checks.We apply two checks to ensure that Test 2 re-
mains in the intended pattern (§5.2). First, the revision may remove
information needed to preserve the original answer. In particular,
replacing an equivalence ( ⊲ ⊳=≡ ) with𝐴⊑𝐷 drops the reverse im-
plication𝐷⊓.
𝑖∃𝑟𝑖.𝐶𝑖⊑𝐴, which is not preserved by the informal
annotations. We therefore verify that each question retains the

Yang et al.
same gold answer when both the revised axioms and the informal
annotations are considered. Second, we require that the gold an-
swer cannot be derived from the revised ontology by an ontology
reasoner alone.
D Construction and Difficulty
Figure 7 shows how the performance of OA-RAG varies with item
construction, including the number of correct MCQ options, the
number of OQ conditions, and the masked slot.
Correct options.For Llama on Test 1, accuracy drops from
73.8% on single-answer MCQs to 32.1% and 20.9% with two and
three correct options, respectively, because all entailed options must
be identified. Despite this drop, OA-RAG remains well above all
baselines (at most 14.3%), as it can verify each option independently
with anASKquery.
Conditions.On Test 1, OQ accuracy is stable up to 𝑛=3con-
ditions (46.3% for Llama at both 𝑛= 2and𝑛= 3) and drops to
42.4% only at 𝑛= 4, since a single SELECT can evaluate the full
conjunction. On Test 2, however, accuracy is far lower throughout
and already falls at 𝑛=3(from 19.6% at 𝑛=2to 17.1%), as the
required conditions must be recovered from separate definition
sentences.
Masked slot.For both models, masking the relation is easiest,
while masking the filler is consistently hardest, because the filler’s
ancestors may also constitute correct answers.
E Prompts
OA-RAG uses four prompts, reproduced verbatim below: pre-tool
reasoning (Ppre), the ReAct loop (Pagent), final answering ( Pfinor
Pown, Step 1), and attribution ( Patt, Step 2).Color code.Black text is
fixed; this color marks ontology- or dataset-specific content; and red
marks the question block 𝑞, including the question, answer options,
PubMedQA abstract, and answer format.Persona.The persona is
prepended to the pre-tool and final-answer prompts. For Tests 1
and 2, SNOMED CT questions use the persona shown in Ppre, and
GO and FoodOn questions replace its first sentence with“You are
an expert in molecular biology”and“You are an expert in the science
of foods, their ingredients, origins and preparation”, respectively. For
Test 3, we use the dataset-specific OntoTune header.
Pre-tool reasoningP pre(SNOMED CT persona)
System:You are an expert physician in clinical medicine and healthcare termi-
nology, answering questions about how the entities of that field are defined,
organized and related. Answer the following question carefully. Think step by
step briefly to reason about the question. Use the information provided with the
question. Additional records are context; use them only when they bear on the
question. A draft is unverified. End your reply with the final answer as a single
JSON object matching the answer schema, on its own last line.
User:{question block𝑞}
Agent promptP agent(the user turn after the pre-tool reply)
An ontology is available through the tool below. Continue from your answer
above: where the ontology decides the question, verify that answer with the
tools; then give the final answer in the format below.
## Tool
query(sparql) : exact reasoning over the ontology’s axioms, asked as a short
SPARQL query over names: every concept and relation is written as its exact
name in double quotes (there are no identifiers or IRIs to use); the relation "isa"means classification (a kind of). Inheritance and sub-relations are applied for
you. The four query forms, each with a toy example:
•DESCRIBE "name" – the ontology’s record of one concept: definition, parents,
children, the relations it states, annotations. E.g.DESCRIBE "grand piano"
•ASK { "subject" "relation" "object" } – is every subject related by
relation to some object? Verdicts: ENTAILED, NOT ENTAILED, UNKNOWN
NAME. E.g.ASK { "grand piano" "part of" "orchestra" }
•SELECT ?x WHERE { ?x "relation" "object" . ... } – the concepts that
satisfy ALL the listed conditions; add VALUES ?x { "name" "name" ... }
to test only these candidates, one verdict each. E.g. SELECT ?x WHERE { ?x
"part of" "orchestra" . ?x "made of" "brass" }
•SELECT ?o WHERE { "subject" "relation" ?o } – what the subject is
related to by that relation; likewise SELECT ?r WHERE { "subject" ?r
"object" } lists the relations that hold between the two. E.g. SELECT ?o
WHERE { "grand piano" "part of" ?o }
## How to use it
Use the ontology when the question is about how its concepts are defined, classi-
fied or related to each other: query decides such a question exactly, so verify the
candidate answers with it before answering. Answer a question about a situa-
tion, a case, a measurement or general knowledge from your own knowledge.
Names must be spelled exactly as the ontology spells them; if a name comes back
UNKNOWN NAME, retry with one of the similar names it lists.
## What the verdicts mean
ENTAILED is certain: the ontology’s axioms derive it. NOT ENTAILED is not a
denial: the ontology is incomplete (open world), so a statement it does not derive
may still hold. Decide such a statement yourself from the records shown and
from what you know, and accept it when it is clearly true.
## Format
Thought: reason about the question and about what to do next – take as much
room as you need
Action:one tool call, written as a function call:query(<SPARQL query>)
Observation:the tool’s result, given to you – never write one yourself
. . . (this repeats as often as needed)
Thought:I now know the answer
Answer:<ONLY the JSON object in the answer schema the question states>
Write a Thought before every Action and before the Answer. A Thought may
also stand on its own; the reply after it must be an Action or an Answer. Begin.
Final answer:P fin(non-empty digest) andP own(empty digest)
Pfin,System:The tool phase of this session is over. Using the question and the
observations below, think step by step briefly, then end your reply with the final
answer JSON object that the question’s answer schema requests, on its own last
line. ENTAILED is certain: the ontology’s axioms derive it. NOT ENTAILED is
not a denial: the ontology is incomplete (open world), so a statement it does not
derive may still hold. Decide such a statement yourself from the records shown
and from what you know, and accept it when it is clearly true.
Pfin,User:{persona} {question block𝑞}
Observations so far:
{the digest𝐸}
Draft answer from the tool phase: {draft ˜𝑦}
Pown,System:Answer the question from your own knowledge. Think step by
step briefly, then end your reply with the final answer JSON on its own last line.
Pown,User:{persona} {question block𝑞}
AttributionP att(one more user turn after the reply ofP fin; §4.1.3)
Your answer above is final; do not change it. Now state what it rests on. The
observations you were given, numbered:
[E1] {observation𝑒 1}
[E2] {observation𝑒 2} . . .
Reply with only this JSON object:
{"evidence": [<the ids of the observations your answer relies on,
e.g. "E1", "E3"; empty if none>], "grounding": "<one word: entailed,
supported, related or unrelated>"}
Choose the grounding as the first of these that holds:
•entailed: your answer follows from the observations by strict logical reasoning:
for each option or name in your answer, the facts stated in the observations
derive it, with nothing added from your own knowledge. A fact counts whether
the tool states it formally (an ENTAILED verdict, a listing of the concepts
entailed to satisfy the question’s conditions, a stated relation, parent or child)
or a text field states it as explicitly as an axiom would, naming the concepts
and the relation (e.g. a definition sentence “Every X is related by R to some Y”).

OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning
Concept names, synonyms and keyword records marked “not verified” state
no fact.
•supported: your answer follows from the observations by sound reasoning that
is not such a derivation: it interprets text that does not state the fact explicitly
(a paraphrasing definition, a name or synonym match, a “not verified” keyword
record), or it combines the facts shown with your own knowledge.
•related: the observations give some valid information relevant to the question,
but not enough to arrive at your answer, or they point to a different answer.
•unrelated: the observations are essentially irrelevant to the question and to
your answer.
NOT ENTAILED or “0 concepts” only means that the axioms do not derive a
statement (open world); it does not deny it. So an empty answer (no option
selected, an empty list), which claims that nothing satisfies the question, is at
most related.
If the reply holds no such object with one of the four levels, one re-ask follows:
Reply with only the JSON object {"evidence": [...], "grounding": "..."} .
The first character of the reply must be ‘{’ and the last must be ‘}’.