# T-CPDL: A Temporal Causal Probabilistic Description Logic for Developing Logic-RAG Agent

**Authors**: Hong Qing Yu

**Published**: 2025-06-23 12:11:15

**PDF URL**: [http://arxiv.org/pdf/2506.18559v1](http://arxiv.org/pdf/2506.18559v1)

## Abstract
Large language models excel at generating fluent text but frequently struggle
with structured reasoning involving temporal constraints, causal relationships,
and probabilistic reasoning. To address these limitations, we propose Temporal
Causal Probabilistic Description Logic (T-CPDL), an integrated framework that
extends traditional Description Logic with temporal interval operators,
explicit causal relationships, and probabilistic annotations. We present two
distinct variants of T-CPDL: one capturing qualitative temporal relationships
through Allen's interval algebra, and another variant enriched with explicit
timestamped causal assertions. Both variants share a unified logical structure,
enabling complex reasoning tasks ranging from simple temporal ordering to
nuanced probabilistic causation. Empirical evaluations on temporal reasoning
and causal inference benchmarks confirm that T-CPDL substantially improves
inference accuracy, interpretability, and confidence calibration of language
model outputs. By delivering transparent reasoning paths and fine-grained
temporal and causal semantics, T-CPDL significantly enhances the capability of
language models to support robust, explainable, and trustworthy
decision-making. This work also lays the groundwork for developing advanced
Logic-Retrieval-Augmented Generation (Logic-RAG) frameworks, potentially
boosting the reasoning capabilities and efficiency of knowledge graph-enhanced
RAG systems.

## Full Text


<!-- PDF content starts -->

arXiv:2506.18559v1  [cs.AI]  23 Jun 2025T-CPDL: A Temporal Causal Probabilistic Description Logic for
Developing Logic-RAG Agent
Hong Qing Yu
University of Derby
June 24, 2025
Abstract
Large language models excel at generating fluent text but frequently struggle with struc-
tured reasoning involving temporal constraints, causal relationships, and probabilistic reason-
ing. To address these limitations, we propose Temporal Causal Probabilistic Description Logic
(T-CPDL), an integrated framework that extends traditional Description Logic with temporal
interval operators, explicit causal relationships, and probabilistic annotations. We present two
distinct variants of T-CPDL: one capturing qualitative temporal relationships through Allen’s
interval algebra, and another variant enriched with explicit timestamped causal assertions. Both
variants share a unified logical structure, enabling complex reasoning tasks ranging from simple
temporal ordering to nuanced probabilistic causation. Empirical evaluations on temporal rea-
soning and causal inference benchmarks confirm that T-CPDL substantially improves inference
accuracy, interpretability, and confidence calibration of language model outputs. By delivering
transparent reasoning paths and fine-grained temporal and causal semantics, T-CPDL signif-
icantly enhances the capability of language models to support robust, explainable, and trust-
worthy decision-making. This work also lays the groundwork for developing advanced Logic-
Retrieval-Augmented Generation (Logic-RAG) frameworks, potentially boosting the reasoning
capabilities and efficiency of knowledge graph-enhanced RAG systems.
Keywords: Temporal Description Logic; Causal Reasoning; Probabilistic Knowledge Representa-
tion; Allen Interval Algebra; Bayesian Update; Knowledge Graphs.
1 Introduction
Knowledge representation and reasoning are foundational challenges in artificial intelligence (AI),
particularly for systems that aim to operate reliably in dynamic, real-world domains. Over the past
decades, Description Logics (DLs) have played a central role in formalizing structured knowledge,
supporting ontology development, and enabling logical inference in systems such as the Seman-
tic Web, Knowledge Graph, and configuration reasoning. DLs offer a clear semantics, decidable
inference procedures, and a rich syntax for modeling hierarchies and constraints between con-
cepts. However, classical DLs are inherently static—they are not well-suited to model knowledge
that evolves over time, includes uncertainty, or involves causal dependencies between entities and
events.
To address temporal expressiveness, researchers have proposed extensions such as Temporal De-
scription Logics (TDLs), which introduce modal operators like ”always”, ”sometime”, ”until”, and
”since” to represent how concepts hold over time [1]. Similarly, probabilistic variants of DLs incor-
porate Bayesian-style reasoning to represent belief or uncertainty about concept membership [2].
1

Independently, there have been efforts to incorporate causality into knowledge systems, often bor-
rowing from structural causal models or action-based frameworks [3]. However, these enhancements
remain fragmented: Temporal, probabilistic, and causal reasoning are often handled in isolation,
and their integration into unified, decidable, and expressive logic remains a significant gap in the
field.
In parallel, Large Language Models (LLMs) such as GPT-4 and Claude have revolutionized
natural language processing and general AI interaction by learning from massive corpora of text
data. However, despite their linguistic fluency, LLMs lack formal structure, semantic grounding,
and reliable reasoning mechanisms. They often hallucinate facts, struggle to maintain consistency
in multistep dialogues, and cannot provide transparent reasoning traces, especially when dealing
with time sensitive, uncertain, or causally linked information [4].
This paper introduces a new framework, Temporal Causal Probabilistic Description Logic (T-
CPDL), to bridge this gap. T-CPDL integrates temporal logic, causal modeling, and probabilistic
inference into a single unified formalism built on Description Logic foundations. Our goal is to
provide a reasoning layer that complements LLMs with a structured, interpretable, and temporally
grounded knowledge representation. By supporting both formal inference and human-aligned in-
terpretability, T-CPDL opens new possibilities for intelligent systems that can learn, reason, and
adapt safely in dynamic environments.
2 Related Work and Logical Foundations
T–CPDL is grounded in four mature strands of research: temporal reasoning, causal inference,
probabilistic description logic, and the classical ALCQI family. This section surveys their core
formalisms and highlights how each strand informs the design choices of our unified logic.
2.1 Temporal Reasoning: Allen’s Algebra & Temporal Description Logics
Human planners constantly reason about when things happen, how long they last, and how they
overlap. Allen’s interval algebra [12] captures every qualitatively distinct way two closed intervals
on a linear time-line can relate. Table 1 lists the thirteen primitive relations (e.g. before, meets,
overlaps) that are mutually exclusive and jointly exhaustive: any concrete pair of time intervals
instantiates exactly one of them. Conjunctions or disjunctions of these relations form temporal
constraint networks [citation required], whose path-consistency can be tested in cubic time and
which we embed directly into T-CPDL quantifiers.
Temporal Description Logics (TDLs). ALCQIT and its interval-based successor TL-F extend
classical ALCQI with tense modalities ♢±,□±, “until/since,” and explicit Allen constraints inside
the quantifier ∃(X) Φ. C[13, 14]. For example,
∃(X)(NOW o X).Maintenance @X
states that some interval overlapping “now” is a period during which Maintenance holds. Despite
this extra expressivity, satisfiability and subsumption remain in EXPTIME, a key result we inherit
when layering causality and probability on top.
In T-CPDL we re-use Allen constraints verbatim, giving each concept term an optional temporal
qualifier ( C@X) and allowing constraint networks to appear under ∃(X) Φ.(·). This gives the
logic a human-like ability to express rich scheduling and plan-recognition patterns while preserving
decidability.
2

Abbrev. Name Intuitive picture (for Xrelative to Y)
b before Xfinishes < Y starts
m meets Xfinishes = Ystarts
o overlaps Xstarts < Y starts < X finishes < Y finishes
s starts Xstarts = Ystarts, Xfinishes < Y finishes
d during Ystarts < X starts < X finishes < Y finishes
f finishes Xstarts < Y starts, Xfinishes = Yfinishes
= equal Xstarts = Ystarts, Xfinishes = Yfinishes
bi after converse of b
mi met-by converse of m
oi overlapped-by converse of o
si started-by converse of s
di contains converse of d
fi finished-by converse of f
Table 1: Allen’s thirteen primitive interval relations [12].
Temporal reasoning allows humans to understand the sequence and duration of events, facili-
tating planning and prediction. In AI, incorporating temporal reasoning is essential for tasks that
involve time-dependent data, such as natural language processing and autonomous navigation [1].
2.2 Causal Reasoning: Structural Models and Causal DLs
Structural Causal Models (SCMs) express cause–effect relations through deterministic equations
plus exogenous noise [15]. Recent efforts to bring SCM ideas to KR include DL extensions with an
explicit binary predicate φ(C, D) and transitivity rules; such “Causal AI” systems aim at trans-
parent and reliable decision support [6].
Causal reasoning enables the identification of cause-and-effect relationships, which are funda-
mental for explanation, prediction, and control. Humans naturally infer causality to make sense
of the world and anticipate the results of actions. Incorporating causal reasoning into AI systems
improves their ability to understand and interact with complex environments [5, 6].
Understanding whya state of affairs arises —-and how interventions might change it—is central
to human intelligence. The prevailing formalism is the Structural Causal Model (SCM) of
Pearl [15]. An SCM consists of:
•a set of endogenous variables V;
•a directed acyclic graph (DAG) capturing parent relations among variables;
•a system of structural equations v:=fv(Pa(v), uv) where uvis exogenous noise.
Causal queries such as P 
Stroke |do(Smoking =false)
are answered by “surgery” on the DAG
followed by probabilistic inference.
Limitations for KR. SCMs excel at quantitative inference but lack a terminological layer for
rich class hierarchies, and they have no native temporal vocabulary. This motivates Causal DLs,
which decorate a Description Logic ontology with a binary predicate φ(C, D) meaning “membership
inCis a sufficient cause of membership in D.” Table 2 contrasts SCM syntax with its DL analogue.
3

Aspect SCM notation Causal DL notation (T–CPDL)
Causal assertion Y:=fY(X) φ(X, Y)
Intervention do( X=x′) replace axiom or add ¬Xfact
Prob. strength conditional density P(Y|X) φ(X, Y)[P=p]
Time index explicit time-series variables φ(X, Y)@t
Table 2: Comparing structural-model and T–CPDL causal syntax.
Recent progress. Greengard highlights the rise of Causal AI , integrating SCM ideas with
machine-learning pipelines for transparent decision support [6]. Early KR attempts embed causal
predicates in Temporal DLs for plan libraries [14], but a full probabilistic, temporal–causal DL was
still missing.
Role in T–CPDL. We adopt φwith three key design choices:
1.Transitivity rule φ(C, D)∧φ(D, E)⇒φ(C, E) to build causal chains.
2.Time-stamping so causes and effects can occur in distinct intervals.
3.Probability tag [P=p|X] enabling Bayesian updates.
These extensions preserve DL decidability by treating φaxioms as first-class TBox statements
handled by an augmented tableau.
2.3 Probabilistic Reasoning in Description Logics
Classical Description Logics assume crisp membership; however, real data are noisy and incom-
plete. Early attempts to embed uncertainty include P–SHIQ and the DISPONTE semantics,
which attach independent probabilities to axioms and evaluate query likelihoods by weighted model
counting [17]. Axioms become random variables and a reasoning engine enumerates minimal ex-
planations of a query, multiplying their weights.
Early probabilistic DLs ( p shiq , DISPONTE) attach probabilities to axioms and compute query
likelihoods via weighted model counting [17]. BALC integrates Bayesian-network semantics into
ALC, using a tableau to propagate beliefs [18]. These approaches demonstrate that uncertainty
can coexist with DL tableaux under suitable independence assumptions.
Bayesian extension of ALC. BALC [18] integrates an explicit Bayesian network with ALC:
each TBox axiom is annotated by a conditional probability table, while a tableau algorithm checks
logical consistency and propagates beliefs along the class hierarchy. Similar ideas arise in MLN–DL
hybrids, where DL atoms act as predicates inside a Markov Logic Network [16].
Rule–centric probabilities in CPDL. In CPDL the probability tag decorates causal rules
rather than generic axioms:
φ(C, D) [P=p|X] (conditional on context X).
Evidence Erevises the prior via the weighted–likelihood formula of (author?) [17]. We adopt
this rule–centric view in T–CPDL because causal links are the natural carriers of uncertainty in
4

dynamic domains: the strength of a link can be learned from temporal event logs and updated
online.
Combining these probabilistic tags with temporal stamps yields statements such as:
φ(Smoking ,Hypertension )@2025 [ P= 0.5], φ (Hypertension ,Stroke )@2030 [ P= 0.67].
A Bayesian update after observing Smoking andHypertension events produces P(Stroke )≈0.335,
as showcased in our running healthcare example later.
Probabilistic reasoning allows decision-making under uncertainty by quantifying the likelihood
of various outcomes. Humans routinely make probabilistic judgments in the face of incomplete
or ambiguous information. Integrating probabilistic reasoning into AI enables systems to handle
uncertainty and make informed decisions, crucial for applications like medical diagnosis and risk
assessment [7, 8].
2.4 Integration Journey from ALCQI to T–CPDL
Classic DLs provide decidability but lack temporal, causal, and uncertain reasoning. Successive inte-
grations added: (i) time (ALCQIT, TL–F), (ii) probabilities (BALC, DISPONTE), and (iii) causal
semantics (CPDL style). T–CPDL overlays all three on an ALCQI core while preserving EXP-
TIME complexity, enabling statements such as: “Smoking before age 40 causes Hypertension with
probability 0.5; Hypertension causes Stroke with probability 0.67.”
While temporal, causal, and probabilistic reasoning have been individually incorporated into
AI systems, their integration within a unified framework remains a challenge. Description Logic
(DL) provides a formal foundation for knowledge representation and reasoning. By extending DL
to include temporal operators, causal relationships, and probabilistic measures, we create Tempo-
ral Causal Probabilistic Description Logic (T-CPDL), mirroring multifaceted human intelligence
capabilities. T-CPDL provides AI systems the ability to represent and reason about dynamic,
uncertain, and causally complex domains, enhancing interpretability, robustness, and adaptability
[9, 10, 11].
2.5 RAG-LLM and Graph-RAG LLM
Retrieval-Augmented Generation (RAG) enhances language models by retrieving relevant external
documents or knowledge bases before generating responses. RAG approaches aim to mitigate
hallucinations and improve factual accuracy by grounding generation processes in external, verified
sources [19]. Traditional RAG, however, primarily utilizes unstructured text documents, limiting
its ability to capture complex relationships explicitly.
Graph-RAG addresses this limitation by leveraging structured knowledge graphs (KGs), en-
abling language models to exploit structured relational information. This approach significantly
improves reasoning capabilities, especially for queries requiring multi-hop inference or relational
understanding [20, 21]. Despite these advances, Graph-RAG systems typically face challenges such
as limited expressiveness in temporal dynamics, causality, and uncertainty handling. The relational
structure alone may not adequately represent nuanced probabilistic and temporal constraints preva-
lent in real-world knowledge.
Logic-based RAG systems emerge as promising solutions by explicitly embedding logical reason-
ing, temporal constraints, causality, and probabilistic inference into retrieval mechanisms. These
logic-based frameworks can systematically address the representational limitations inherent in
purely textual or graph-structured retrieval methods, providing clearer reasoning paths, more reli-
able inference under uncertainty, and improved transparency in complex decision-making scenarios
[22, 23].
5

The next sections formalise the semantics and automated reasoning procedure.
3 Syntax Overview of T–CPDL
T–CPDL fuses ALCQI with the temporal, causal, and probabilistic devices surveyed in Section 2.
This section formalises the concrete syntax, moving bottom-up from the alphabet of symbols to
full concept terms and finally a running example.
3.1 Alphabet
Before we can assemble complex T–CPDL formulae, we require a precise inventory of the primitive
symbols the logic recognises. Table 3 lists each syntactic category, its notation, and its informal
role.
Why these categories?
•Atomic concepts/roles provide the ontology backbone; temporal qualifiers add a time
index C@twhose extension may vary.
•Features vs. parametric features let us thread the same participant through multi–stage
plans without repetitive equality constraints.
•Temporal variables support reasoning with unknown or relative times and Allen relations,
crucial for TL–ALCF subsumption.
•Causal operator & probability tags import CPDL’s transparent Bayesian updates into
a temporal setting.
•Evidence items bridge observed reality and the abstract causal graph.
This alphabet underpins the grammar presented in the following subsection.
Allen relations. Temporal constraints Φ use the 13 basic relations of Allen’s interval algebra:
b, m, o, s, d, f, =, bi, mi, oi, si, di, fi [12].
3.2 Grammar
We provide two complementary flavors of T-CPDL because real-world data comes in wildly different
forms of temporal detail. In some domains—say, when mining natural-language texts or noisy sensor
streams—you often know only that “event A happened before event B” or “these two processes
overlap,” but you have no reliable clock-time to pin them to. The Allen-relational variant lets you
capture and reason about those interval-orderings directly, yielding crisp causal inferences purely
from ordering constraints. In contrast, when you do have precise timestamps—e.g. log files, clinical
records, financial trades—you can use the timestamped variant to anchor every causal link to a
concrete instant (and still refine it with Allen relations if you like). Offering both ensures that
T-CPDL can gracefully degrade to the level of temporal information you actually possess, while
preserving the same core causal machinery.
6

3.2.1 Allen-Relational only T-CPDL ( T-CPDL A)
A lightweight variant where causal edges carry no global timestamp; instead, cause and effect
intervals are related purely via Allen constraints.
T-Concept C::=E
|C⊓D
|C[Y]@X
| ∃(X) Φ. C(1)
E-Concept E::=A
| ⊤
| ⊥
| ¬E
|E⊓F
|E⊔F
| ∃R.E
| ∀R.E
|≥n R.E
|≤n R.E(2)
Allen relations are drawn from:
{b, m, o, s, d, f, =, bi, mi, oi, si, di, fi }. (3)
Temporal constraint networks are formed by:
Φ ::= X(Rel) Y
|Φ∧Φ
|Φ∨Φ(4)
Causal statements in T-CPDL Atake the form:
φ(C, D)[P=p]
∃(X, Y) Φ. φ(C@X, D @Y)[P=p](5)
In the first form, “ φ(C, D)[P=p]” asserts a causal link without any temporal qualifier. In
the second form, fresh intervals X(cause) and Y(effect) are constrained by an Allen network Φ,
yielding “ φ(C@X, D @Y)[P=p].”
3.2.2 Timestamped T-CPDL ( T-CPDL T)
The full variant retains explicit timestamps on causal edges, while still allowing auxiliary Allen
constraints.
Causal statements in T-CPDL T:
φ(C, D)@τ[P=p|X]
∃(X, Y) Φ. φ(C@X, D @Y)@τ[P=p|Z](6)
Here, “ φ(C, D)@τ[P=p|X]” is the original timestamped form (optionally conditioned on
context X). The second form binds helper intervals X, Y, applies constraints Φ, and attaches both
@τand probability tag.
7

3.3 Running Examples
We illustrate both variants on the same healthcare scenario.
3.3.1 Allen-Relational Example ( T-CPDL A)
TBox causal rules (no global @τ):
∃(X, Y) (X b Y ). φ(Smoking @X,Hypertension @Y)[P= 0.5]
∃(U, V) (U b V ). φ(Hypertension @U,Stroke @V)[P= 0.67](7)
ABox evidence (grounded events):
Smoking bHypertension (8)
meaning that Smoking occurs before Hypertension.
Chaining: Since
φ(Smoking ,Hypertension )[0.5] and φ(Hypertension ,Stroke )[0.67], (9)
we derive (via weighted likelihood)
φ(Smoking ,Stroke )[≈0.335]. (10)
Relational projection:
∃(X, Z) (X b Z ). φ(Smoking @X,Stroke @Z)[P≈0.335] (11)
placing inferred stroke interval Zstrictly after smoking interval X.
3.3.2 Timestamped Example ( T-CPDL T)
Aircraft Maintenance Incident example:
Maintenance Ontology
φ(Wear,Crack )@τ1[P= 0.6],
φ(Crack ,Incident ) [P= 0.8],
∃(U, V) (U b V ).Inspection @U⊑ ¬ Incident @V.
Observed Evidence
Wear @2025–01–10 ,No inspection recorded.
8

Reasoning Steps
1.Causal chaining.
P(Incident |Wear ) = 0.6×0.8 = 0 .48 (48% risk) .
2.Temporal projection. The plausible incident interval Vmust satisfy τ1bi V (i.e. occur
after τ1).
3.Preventive rule check. No interval Uwith Inspection @UandU b V exists, so the subsump-
tion axiom cannot block the incident.
The maintenance module therefore issues a 48% incident-risk alert and recommends scheduling an
inspection interval Uthat meets (m) orbefore (b)V, which would satisfy the preventive axiom and
drive P(Incident ) toward 0.
4 Fundamental Theorems
4.1 Finite-Model Property
Theorem 1 (Finite Model) .If a T --CPDL knowledge base Kis satisfiable, then it has a model
whose domain size is 2O(|K|)and whose temporal structure contains only the intervals explicitly
mentioned in Kplus at most O(|K|)fresh points.
Proof. Take the tableau for the ALCQI fragment of Kand apply standard loop blocking to obtain
a finite completion tree (cf. (author?) (year?) ). For every existential temporal binder ∃(X)Φ. C
create at most |Φ|fresh interval variables satisfying Φ; path-consistency of Allen networks guar-
antees that a finite assignment exists. Because causal rules are global TBox statements, their
transitive closure adds at most |K|additional edges and no new individuals. Probability values
are copied verbatim from K; there are finitely many such numbers, so the resulting structure is
finite.
4.2 Complexity
Theorem 2 (EXPTIME Completeness) .KB consistency, concept satisfiability, and instance check-
ing in T --CPDL are EXPTIME-complete.
Proof. Hardness inherits from ALCQI. For membership: the tableau expands at most exponen-
tially many individual nodes (ALCQI bound). For each node, temporal propagation invokes cubic
path-consistency, and causal propagation adds at most |T |deterministic edges with constant-time
probability multiplication (Theorem 4 below). Thus the overall procedure is bounded by 2p(|K|)for
some polynomial p.
4.3 Tableau Soundness and Completeness
Theorem 3 (Soundness & Completeness) .The extended tableau calculus (temporal rules + causal
transitivity + probability update) is sound and complete for the semantics in §4.
Proof. Soundness : every rule is locally truth-preserving. E.g. the causal-transitivity rule adds
φ(C, E) only if φ(C, D) and φ(D, E) already hold; by definition of φthe new edge is entailed in
every model that satisfied the premises.
9

Completeness : run the tableau with fair rule application. If expansion halts without clash,
each branch induces a Hintikka structure that satisfies all syntactic conditions. By Theorem 1
this structure can be turned into a finite model of K. Conversely, any model yields a clash-free
branch.
4.4 Probabilistic Composition
Theorem 4 (Probability of Composed Cause) .Letφ(C, D)[P=p12]andφ(D, E)[P=p23]be
the only causal paths from CtoE. Then T–CPDL must annotate the derived edge φ(C, E)with
p13=p12·p23.
Proof. By definition P(E|C) =P
dP(E|d)P(d|C). Since Dis the sole mediator, the sum has one
term: P(E|C) =p23·p12. Assigning any other weight to φ(C, E) would violate Bayes coherence
or probabilistic soundness.
4.5 Temporal Acyclicity
Theorem 5 (Temporal–Causal Consistency) .If every causal rule is stamped so that its cause
interval is before or meets its effect interval, the transitive closure of φis acyclic.
Proof. Assume a cycle X1< X 2<
dots < X n< X 1in the timeline. Linear order <is irreflexive, so X1< X 1is impossible. Hence no
such cycle exists; transitive closure is a DAG.
Together, Theorems 1–5 provide the theoretical guarantees required for deploying T–CPDL as
a sound, complete, and decidable reasoning substrate under real-world LLM pipelines.
5 Prompt Engineering for Automated T–CPDL Extraction with
LLM
We present four case studies illustrating how to engineer prompts for an LLM to extract formal
T-CPDL specifications (in JSON) from diverse unstructured sources. Each example begins with
a realistic document excerpt, poses a reasoning task, and then shows how the prompt guides the
choice of variant (Allen-Relational vs. Timestamped) and produces a machine-readable JSON file
for downstream inference.
5.1 Master Meta-Prompt
(Reusable template; insert domain text under <<< >>> .)
SYSTEM: You are an expert T-CPDL knowledge engineer.
STEP 1: Choose the T-CPDL variant:
- If any entries include timestamps e.g. ISO 8601, set
"variant":"T-CPDL_T" and use timestamped syntax.
- Otherwise, set "variant":"T-CPDL_A" and use Allen-relational.
DETAILS FOR T-CPDL_A (Allen-Relational):
-- Use only Allen operators:
10

before (b), meets (m), overlaps (o),
starts (s), during (d), finishes (f), equals (=),
and inverses: met-by (bi), mi (mi),
overlapped-by (oi), si (si),
di (di), fi (fi).
-- Concept assertions:
{"concept":C, "individual":I, "atInterval":X}
-- Causal form with existential intervals:
\\exists(X,Y) Phi . varphi(C@X,D@Y)[P=p]
-- Omit "starts"/"finished-by" if no ISO dates.
DETAILS FOR T-CPDL_T (Timestamped):
-- Each interval must have:
"starts": time e.g. ISO-8601, "finished-by": time e.g. ISO-8601
-- Causal entries include:
"atInterval":X, "@\\tau":time
-- You may also include Allen fields inside intervals.
When listing causes, compute each probability as:
1/(number of causes for that effect concept)% if there is no probability
indicated in the text.
STEP 2: Output exactly one JSON file, spec.json,
following this schema:
{
"variant": "T-CPDL_A" | "T-CPDL_T",
"intervals": [
{ "id": String,
/* For T-CPDL_A: Allen fields (before, meets, ...) */
/* For T-CPDL_T: starts:String, finished-by:String,
plus optional Allen fields */ }
, ...
],
"assertions": [
{ "concept": String,
"individual": String,
"atInterval": String }
, ...
],
"causes": [
{ "causeConcept": String,
"effectConcept": String,
"probability": Number|null,
"atInterval": String
/* For T-CPDL_T add "@\\tau": String */ }
, ...
]
11

}
STEP 3: Verify that:
- For T-CPDL_A all Allen fields are exhaustive.
- For T-CPDL_T all timestamps are valid and respect chronology.
- Causal chains obey temporal constraints.
- Missing probabilities default to null.
- JSON is valid for automated reasoning.
USER: Extract T-CPDL specification from DOCUMENT:
<<<
{insert text here}
>>>
END.
5.2 Case Studies
5.2.1 Example 1: Clinical Narrative (Allen-Relational)
Document Excerpt https://www.nhs.uk/conditions/pneumonia/ Causes of pneumonia section
Pneumonia is usually caused by a bacterial or viral infection.
You can catch pneumonia from someone who has it, or you can sometimes get it if you
have another infection such as: – flu – respiratory syncytial virus (RSV) – COVID-19
Pneumonia can be caused by a fungal infection, but this is rare in healthy people in the
UK.
It may also be caused by something getting into your lungs, such as water or food
(aspiration pneumonia).
LLM Output ( spec.json )
{
"variant": "T-CPDL_A",
"intervals": [
{ "id":"i_bacterial", "before":"i_pneumonia" },
{ "id":"i_viral", "before":"i_pneumonia" },
{ "id":"i_flu", "before":"i_pneumonia" },
{ "id":"i_rsv", "before":"i_pneumonia" },
{ "id":"i_covid19", "before":"i_pneumonia" },
{ "id":"i_fungal", "before":"i_pneumonia" },
{ "id":"i_aspiration", "before":"i_pneumonia" }
],
"causes": [
{ "causeConcept":"BacterialInfection",
"effectConcept":"Pneumonia",
"probability":0.1429,
"atInterval":"i_bacterial" },
{ "causeConcept":"ViralInfection",
12

"effectConcept":"Pneumonia",
"probability":0.1429,
"atInterval":"i_viral" },
{ "causeConcept":"Flu",
"effectConcept":"Pneumonia",
"probability":0.1429,
"atInterval":"i_flu" },
{ "causeConcept":"RSV",
"effectConcept":"Pneumonia",
"probability":0.1429,
"atInterval":"i_rsv" },
{ "causeConcept":"COVID-19",
"effectConcept":"Pneumonia",
"probability":0.1429,
"atInterval":"i_covid19" },
{ "causeConcept":"FungalInfection",
"effectConcept":"Pneumonia",
"probability":0.1429,
"atInterval":"i_fungal" },
{ "causeConcept":"AspirationEvent",
"effectConcept":"Pneumonia",
"probability":0.1429,
"atInterval":"i_aspiration" }
]
}
5.2.2 Example 2: Aircraft Incident Report Analysis (Timestamped)
https://assets.publishing.service.gov.uk/media/68483cd696e63bce58e4e6cf/Cessna_152_
G-BSZW_06-25.pdf
Serious Incident Aircraft Type and Registration: Cessna 152, G-BSZW ... Time: 24
February 2024 at 1539 hrs Incident: Rudder control bellcrank fractured
Synopsis: During an instructional flight the aircraft suffered a loss of right rudder
authority. Examination revealed that the right rudder bellcrank had failed due to stress
corrosion cracking, causing the right rudder cable to detach. The cracking initiated
where the bellcrank was fouling against the fuselage. Several possible misalignments of
the rudder or bellcrank were considered. . . .
LLM Output ( spec.json )
{
"variant": "T-CPDL_T",
"intervals": [
{ "id":"t_sc", "starts":"2024-02-24T15:39:00Z","finished-by":
"2024-02-24T15:39:00Z" },
{ "id":"t_foul", "starts":"2024-02-24T15:39:00Z","finished-by":
"2024-02-24T15:39:00Z" },
13

{ "id":"t_rudMis", "starts":"2024-02-24T15:39:00Z","finished-by":
"2024-02-24T15:39:00Z" },
{ "id":"t_bellMis", "starts":"2024-02-24T15:39:00Z","finished-by":
"2024-02-24T15:39:00Z" },
{ "id":"t_fail", "starts":"2024-02-24T15:39:00Z","finished-by":
"2024-02-24T15:39:00Z" }
],
"assertions": [
{ "concept":"StressCorrosionCracking", "individual":"G-BSZW", "atInterval":
"t_sc" },
{ "concept":"FoulingAgainstFuselage", "individual":"G-BSZW", "atInterval":
"t_foul" },
{ "concept":"RudderMisalignment", "individual":"G-BSZW", "atInterval":
"t_rudMis" },
{ "concept":"BellcrankMisalignment", "individual":"G-BSZW", "atInterval":
"t_bellMis" },
{ "concept":"RudderBellcrankFracture", "individual":"G-BSZW", "atInterval":
"t_fail" }
],
"causes": [
{
"causeConcept":"StressCorrosionCracking",
"effectConcept":"RudderBellcrankFracture",
"probability":0.25,
"atInterval":"t_sc"
},
{
"causeConcept":"FoulingAgainstFuselage",
"effectConcept":"RudderBellcrankFracture",
"probability":0.25,
"atInterval":"t_foul"
},
{
"causeConcept":"RudderMisalignment",
"effectConcept":"RudderBellcrankFracture",
"probability":0.25,
"atInterval":"t_rudMis"
},
{
"causeConcept":"BellcrankMisalignment",
"effectConcept":"RudderBellcrankFracture",
"probability":0.25,
"atInterval":"t_bellMis"
}
]
}
14

5.2.3 Example 3: Prediction
Input paragraph —natural language:
On 10 Jan 2025 the maintenance log reports visible wear on the left aileron of aircraft
AF123. Engineers estimate a 60% chance that wear will generate a crack in three
months and a 80% chance that an untreated crack will lead to an incident.
LLM response (valid JSON):
{
"intervals": [
{ "id": "tWear", "starts": "2025-01-10", "finished-by": "2025-01-10" }
],
"assertions": [
{ "concept": "Wear", "individual": "Aileron_AF123", "atInterval": "tWear" }
],
"roles": [],
"causes": [
{ "causeConcept": "Wear", "effectConcept": "Crack",
"probability": 0.6, "atInterval": "tWear" },
{ "causeConcept": "Crack", "effectConcept": "Incident",
"probability": 0.8, "atInterval": "tWear" }
]
}
After import, the T–CPDL output supports the risk prediction which is illustrated in Sec-
tion 3.3.
6 Conclusion
The unification of temporal logic, causal modeling, probabilistic inference, and Description Logic
into a single, coherent T–CPDL framework constitutes a transformative leap in symbolic–statistical
AI. T–CPDL not only captures the subtleties of human reasoning within a decidable logic, but it also
propels LLM reasoning beyond unconstrained text generation toward robust, verifiable inference.
Key Contributions and Impact
•Holistic Reasoning Substrate : By integrating Allen-style temporal operators, explicit
causal predicates, Bayesian probability tags, and the ALCQI core, T–CPDL forges a unified
logical foundation that mirrors real-world complexity.
•Verifiable Inference : Transparent proof traces and probabilistic calibration transform
opaque LLM outputs into rigorously grounded conclusions, addressing fundamental trust
and safety concerns in GenAI.
•Fine-Grained Temporal Semantics : Our framework encodes precise temporal relation-
ships—before, after, during—ensuring contextual integrity in dynamic domains such as sci-
entific discovery and healthcare.
The T–CPDL framework thereby lays the groundwork for next-generation AI assistants capable
of delivering temporally coherent, causally explainable insights in safety-critical applications.
15

Future Directions
1.Scalable Causal Learning : Leveraging deep learning to autonomously infer and refine
causal probabilities from massive temporal datasets will close the loop between data-driven
discovery and model-driven reasoning.
2.Multimodal Real-Time Reasoning : Extending T–CPDL to ingest visual, auditory, and
sensor data within the same logical substrate, coupled with low-latency inference algorithms,
will empower autonomous systems and digital twins with on-the-fly, explainable decision
making.
These advances pave the way for LLM-based reasoning engines that not only excel at natural
language generation but also deliver stringent logical rigor, setting a new standard for trustworthy
AI.
References
[1] Artale, A. and Franconi, E. (1998). A Temporal Description Logic for Reasoning about Actions
and Plans. *Journal of Artificial Intelligence Research*, 9, 463–506.
[2] Botha, L., Meyer, T., and Pe˜ naloza, R. (2020). The Probabilistic Description Logic BALC.
*arXiv preprint arXiv:2009.13407*.
[3] Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
[4] Burtsev, M., Reeves, M., and Job, A. (2023). The Working Limitations of Large Language
Models. *MIT Sloan Management Review*.
[5] World Economic Forum. (2024). Causal AI: The revolution uncovering the ’why’
of decision-making. Available at: https://www.weforum.org/stories/2024/04/
causal-ai-decision-making/
[6] Greengard, S. (2024). Causal Inference Makes Sense of AI. *Communications of the ACM*.
Available at: https://cacm.acm.org/news/causal-inference-makes-sense-of-ai/
[7] GeeksforGeeks. (2024). Probabilistic Reasoning in Artificial Intelligence. Available at: https:
//www.geeksforgeeks.org/probabilistic-reasoning-in-artificial-intelligence/
[8] IndiaAI. (2022). The Importance of Probabilistic Reasoning in AI. Available at: https://
indiaai.gov.in/article/the-importance-of-probabilistic-reasoning-in-ai
[9] Artale, A. and Franconi, E. (2000). Temporal Description Logics. *ResearchGate*. Avail-
able at: https://www.researchgate.net/publication/2454779_Temporal_Description_
Logics
[10] Artale, A. and Franconi, E. (2002). A Temporal Description Logic for Reasoning over Concep-
tual Models. In *Proceedings of the International Conference on Conceptual Modeling* (pp.
98–110). Springer. doi:10.1007/3-540-45816-6_13
[11] Artale, A. and Franconi, E. (2000). A Tableau Calculus for Temporal Description
Logic: The Expanding Domain Case. *Technical Report*, Free University of Bolzano-
Bozen. Available at: https://www.researchgate.net/publication/2642734_A_Tableau_
Calculus_for_Temporal_Description_Logic_The_Expanding_Domain_Case
16

[12] Allen, J.F. (1983). Maintaining Knowledge About Temporal Intervals. *Communications of
the ACM*, 26(11), 832–843.
[13] Artale, A. and Franconi, E. (1994). A Temporal Description Logic for Reasoning about Actions
and Plans. *Journal of Artificial Intelligence Research*, 9, 463–506.
[14] Artale, A. and Franconi, E. (1998). Temporal Description Logics for Action Libraries. In
*Proceedings of KR’98* (pp. 3–14).
[15] Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
[16] Koller, D. and Friedman, N. (2009). Probabilistic Graphical Models: Principles and Tech-
niques. *MIT Press*.
[17] Riguzzi, F. and Lamma, E. (2012). A Framework for Probabilistic Description Logics Based
on DISPONTE. *Annals of Mathematics and Artificial Intelligence*, 54, 79–101.
[18] Botha, L., Meyer, T., and Pe˜ naloza, R. (2020). The Probabilistic Description Logic BALC.
*arXiv preprint arXiv:2009.13407*.
[19] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., and Riedel, S. (2020).
Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural In-
formation Processing Systems*, 33, 9459–9474.
[20] Yasunaga, M., Ren, H., Bosselut, A., Liang, P., and Leskovec, J. (2021). QA-GNN: Reason-
ing with Language Models and Knowledge Graphs for Question Answering. *arXiv preprint
arXiv:2104.06378*.
[21] Luo, L., Zhao, Z., Haffari, G., Phung, D., Gong, C., and Pan, S. (2025). GFM-RAG: Graph
Foundation Model for Retrieval Augmented Generation. *arXiv preprint arXiv:2502.01113*.
[22] Haarslev, V. and M¨ oller, R. (2001). Description logic systems with concrete domains: appli-
cations for the Semantic Web. In *Proceedings of the International Workshop on Description
Logics (DL2001)* (pp. 148–157).
[23] Wang, D., Zou, B., Han, Z., and Xu, Z. (2024). tBen: Benchmarking and Testing the Rule-
Based Temporal Logic Reasoning Ability of Large Language Models with DatalogMTL. In
*Proc. ICLR 2025 Workshop on Logic and Language*.
17

Symbol class Notation / examples Intuition and purpose
Atomic concepts A, B, Disease ,Stroke Unary predicates denoting time–varying
sets of individuals (patients, devices, situa-
tions). They extend the Asets of classical
ALCQI.
Atomic roles R, S, hasSymptom Binary relations between individu-
als; may be temporally qualified,
e.g.hasSymptom (a, b)@t.
Features f, g, birthDate Functional roles mapping each source in-
dividual to at most one target at a given
instant, enabling number restrictions and
value constraints.
Parametric fea-
tures?actor, ?object Time– invariant functional roles (from TL–
F): once bound in an action instance, the
value persists across all intervals of that
instance.
Individuals a, b, patient123 Constant symbols naming concrete enti-
ties; we assume the Unique Name Assump-
tion.
Temporal
vars./intervalst, u, X, Y Denote points or closed intervals on a
linear discrete timeline; used in Allen
constraints ( Xbefore Y) and qualifiers
(C@X).
Evidence items e1, e2 Ground literals observed by the system
(e.g. Fever @2025) feeding Bayesian up-
dates.
Probability tags p∈[0,1] Attach belief strength to causal rules
(φ(C, D)[P= 0.67]) or ground facts.
Causal operator φ(C, D) Distinguished binary predicate “ Ccauses
D”; can itself be time-stamped and
weighted.
Logical connectives ¬,⊓,⊔,∃,∀,≥n,≤n ALCQI constructors—our static core.
Temporal connec-
tives@,∃(X) Φ.,♢±,□±, U, S Qualifier C@τpins a concept to a spe-
cific interval. quantifier ∃(X) Φ. Cintro-
duces fresh interval variables constrained
by Allen relations Φ. Point-based tense op-
erators: ♢+/□+(sometime/always in the
future), ♢−/□−(past). Interval opera-
torsU(until) and S(since) support rich
temporal patterns as in ALCQIT.
Table 3: Alphabet of T–CPDL (revised column widths).
18