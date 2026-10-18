# Do Agents Need a New Kind of Paper? SPRA: Structured Paper Representation for Agents

**Authors**: Yutong Leng, Haojiong Chen, Fukai Shang, Xiaoyang Wang, Mindong Liu, Runyuan Ma, Jiantao Qiu

**Published**: 2026-10-15 07:34:12

**PDF URL**: [https://arxiv.org/pdf/2610.21424v1](https://arxiv.org/pdf/2610.21424v1)

## Abstract
Scientific agents must connect research problems, methods, and findings dispersed through narrative articles. We introduce SPRA (Structured Paper Representation for Agents), a paper-level representation of persistent research objects and typed relations. Its empirical-AI profile organizes Problems, Contributions, Components, ExperimentSetups, Measures, and Findings for querying, traversal, and reuse. Selected paper-level citation relations record dependencies, component use, and comparisons with prior work; source locations support verification. We instantiate SPRA across more than one million AI papers, with 19.95 million research units and 24.07 million internal relations. Evaluations access SPRA through Schema-RAG with task evidence fixed before inference. Across seven subfields, SPRA improves source-conditioned scientific QA by 0.381/5 over passage retrieval while remaining close to Full Compact (0.078/5 lower on average). In directed cross-paper reasoning, qualification improves by 0.184/5 over passage retrieval; Top-5 synthesis shows mixed quality effects. Across all paired RQ2 tasks, including Top-5, input+output tokens decrease by 28.5% relative to passage retrieval and 95.3% relative to Full Compact. In idea generation, a 0.067-point lower Overall score than Full Compact accompanies 93.7% fewer input+output tokens. Together, these results support SPRA as a usable representation of scientific content and a foundation for research workflows that connect, inspect, and reuse paper-level research objects. The SPRA resource is accessible through the SciVerse scholarly-data platform.

## Full Text


<!-- PDF content starts -->

Do Agents Need a New Kind of Paper?
SPRA: Structured Paper Representation for Agents
Yutong Leng1,2Haojiong Chen1Fukai Shang1
Xiaoyang Wang1Mindong Liu1Runyuan Ma1Jiantao Qiu1,∗
1Shanghai Artificial Intelligence Laboratory
2Shanghai Jiao Tong University
*Corresponding author.
Website · Paper Schema API · Agent Skill · Extraction Code
Abstract
Scientific agents must connect research prob-
lems, methods, and findings dispersed through
narrative articles. We introduce SPRA (Struc-
tured Paper Representation for Agents), a
paper-level representation of persistent re-
search objects and typed relations. Its
empirical-AI profile organizes Problems, Con-
tributions, Components, ExperimentSetups,
Measures, and Findings for querying, traver-
sal, and reuse. Selected paper-level citation
relations record dependencies, component use,
and comparisons with prior work; source loca-
tions support verification. We instantiate SPRA
across more than one million AI papers, with
19.95 million research units and 24.07 million
internal relations. Evaluations access SPRA
through Schema-RAG with task evidence fixed
before inference.Across seven subfields, SPRA improves source-
conditioned scientific QA by 0.381/5 over pas-
sage retrieval while remaining close to Full
Compact ( 0.078/5 lower on average). In
directed cross-paper reasoning, qualification
improves by 0.184/5 over passage retrieval;
Top-5 synthesis shows mixed quality effects.
Across all paired RQ2 tasks, including Top-5,
input+output tokens decrease by 28.5% rela-
tive to passage retrieval and 95.3% relative to
Full Compact. In idea generation, a 0.067 -
point lower Overall score than Full Compact
accompanies 93.7% fewer input+output tokens.
Together, these results support SPRA as a us-
able representation of scientific content and a
foundation for research workflows that connect,
inspect, and reuse paper-level research objects.
The SPRA resource is accessible through the
SciVerse scholarly-data platform.
Document
parsing
Text
Figures
Tables
Formulas
LayoutMarkdown
§1 Introduction
§2 Method
§3 Experiments
§4 ResultsDocument JSON
paper_id string
title · authors text
year · venue scalarSections JSON
problem[] units
method[] units
evidence[] units
source locatorsRelations JSON
source → target units
type predicate
evidence locators
JSON-to-graph correspondence
Problem Section Method Section Evidence Section
Problem Contribution
ComponentExperiment
setupMeasure Findingmotivates
part_ofpart_of
part_ofabout
aboutsupports
resolvesNode types
Problem
Contribution /
Component
Experiment setup
Measure / Finding
Relations
Typed
Cross-unit
Figure 1: Research objects and typed relations in the empirical-AI profile. Problem, Method, and Evidence sections
encode the paper’s research argument; arrows show relation directions. Units and relations carry source bindings
for verification.
1
arXiv:2610.21424v1  [cs.AI]  15 Oct 2026

1 Introduction
AI paper submissions are growing rapidly (ICLR,
2026), and scientific agents increasingly participate
in literature discovery, comparison, and synthesis
(He et al., 2025; Asai et al., 2026). The same pa-
per can inform method comparisons, evaluation
queries, and research proposals. These operations
require its problems, contributions, evaluations,
and findings, together with their relationships. Con-
ventional articles express this structure through nar-
rative text, tables, and figures; agents must connect
findings to methods, conditions, and comparisons.
How should a paper expose these scientific rela-
tionships for agents to query, compare, and reuse?
Passage-based retrieval supplies relevant source
text (Lewis et al., 2020), while task-specific synthe-
sis assembles evidence for a question (Dasigi et al.,
2021; Lála et al., 2023; Skarlinski et al., 2024).
Research roles and dependencies may still need to
be reconstructed from these passages. Repeated
full-paper inputs incur token-based API charges
(Chen et al., 2023) and occupy context needed for
additional papers and task history. Larger con-
text windows also need not ensure reliable evi-
dence use: middle-position information can be
underused, with positional attention bias impli-
cated in this behavior (Liu et al., 2024; Hsieh et al.,
2024). These costs motivate evaluating retained
scientific utility jointly with token use. Compact,
explicit research structure makes scientific connec-
tions directly available while preserving capacity
for linked evidence.
We introduce SPRA, theStructured Paper Rep-
resentation for Agents. SPRA derives its research
roles and predicates from paper understanding,
method comparison, and evidence integration, rep-
resenting the connections among a contribution,
its evaluation, and its reported findings. Paper-
scoped identifiers let successive operations refer
to the same stored objects; internal relations con-
nect contributions to their evaluation and reported
outcomes, and a separate citation layer records
selected dependencies on prior work. Figure 1 il-
lustrates SPRA; Schema-RAG is one access mech-
anism used to evaluate this representation and re-
source.
Consider the speech-emotion machine-
unlearning paper of Ren et al. (2025). Its
privacy-cost Problem, forget-set-only Contribu-tion, unlearning-accuracy Measure, and baseline-
comparison Finding form a research argument.
An evaluation question selects the linked Mea-
sure and Finding; a method comparison selects
corresponding records from another paper. To
check the reported advantage, an agent can follow
these connections to the evaluation conditions and
reopen the associated source evidence.
The contributions are:
1.a paper-level representation that combines
persistent research objects, typed internal rela-
tions, citation dependencies, and source bind-
ings, with an extraction and export workflow
(Section 3);
2.a million-paper resource, programmatic ac-
cess, and aligned representation inputs and
frozen evaluation outputs (Table 2); and
3.paired evaluations of scientific QA, cross-
paper reasoning, and idea generation that
quantify answer quality and task-level token
use (Sections 4–5).
2 Related Work
Scholarly documents and provenance.GRO-
BID, S2ORC, and MinerU extract metadata, text,
and layout (Lopez, 2009; Lo et al., 2020; Wang
et al., 2024). JATS provides markup; PROV-O
models provenance; PROV-CONSTRAINTS de-
fines validity; RO-Crate packages research ob-
jects (National Information Standards Organiza-
tion, 2024; Lebo et al., 2013; Cheney et al., 2013;
Soiland-Reyes et al., 2022; Research Object Com-
munity, 2026).
Research roles and scientific extraction.Argu-
mentative Zoning and CoreSC label scientific text
by research function (Teufel et al., 2009; Liakata
et al., 2012). SciERC and DyGIE++ model enti-
ties, relations, and events (Luan et al., 2018; Wad-
den et al., 2019); SciREX, NLPContributionGraph,
SciER, and SciEvent represent document-level re-
lations, contribution triples, or scientific events
(Jain et al., 2020; D’Souza et al., 2021; Zhang
et al., 2024; Dong et al., 2025). SPRA assembles
research roles and extracted content into persistent
paper-owned objects with typed connections and
executable operations.
2

Relation Scientific role Source→Target
part_ofComponent or setup membership{Co,S} → {C,Co,S}
co_contributionJoint contributionsC→C
compares_toComparison of methods or measures{C,Co,S,M} → {C,Co,S,M}
evaluatesMeasure applied to a methodM→ {C,Co}
aboutSubject of a findingF→ {C,Co,S,M}
supportsProvides evidence links from research
objects to Findings{M,F,C,Co} →F
motivatesProblem motivating a method or setupP→ {C,Co,S}
resolvesFinding addressing a problemF→P
Table 1: Paper-internal relations and admissible endpoint signatures. P = Problem; C = Contribution; Co =
Component; S = ExperimentSetup; M = Measure; F = Finding. Braces denote allowed types. Construction stages
are given in Appendix A.3.
Structured scholarly knowledge.ORKG sup-
ports open scholarly aggregation through paper-
linked contributions, templates, comparisons,
APIs, and ASK retrieval (Jaradeh et al., 2019; Open
Research Knowledge Graph, n.d.; Oelen et al.,
2024). SPRA derives roles and predicates from
agent tasks and provides an extraction, modeling,
and export workflow for paper-owned research ob-
jects. These goals are complementary: property
alignment connects task-oriented objects to ORKG
aggregation. Appendix A.7 describes this engineer-
ing interface.
Structured content for scientific assessment.
SciCore combines full-manuscript review with a
judgment of an extracted science core, evaluating
stability under content-preserving rewrites along-
side discrimination across papers (Wang et al.,
2026). SciCore motivates an explicit view of sci-
entific content for automated assessment. SPRA
develops a persistent paper-level representation for
broader research operations: paper-owned units,
typed relations, and source bindings support query-
ing and comparison across tasks.
Agent access to scientific literature.PaperQA/-
PaperQA2, OpenScholar, and PaSa support lit-
erature retrieval and cited synthesis (Lála et al.,
2023; Skarlinski et al., 2024; Asai et al., 2026;
He et al., 2025); ResearchAgent iteratively devel-
ops research ideas (Baek et al., 2025). GraphRAG
builds entity communities for query-focused sum-
marization, and KG2RAG expands and organizes
retrieved chunks through graph connections (Edge
et al., 2024; Zhu et al., 2025). Intern-Atlas exposesmethod evolution for graph-guided ideation (Wu
et al., 2026). SPRA supplies a persistent paper-
level research representation that these workflows
can query and compose.
3 SPRA: Paper-Level Research
Representation
3.1 Representation and Profiles
The construction input is a source article parsed
into section-aware Markdown with source markers.
The output is a paper-scoped record of research
units, typed relations, citation dependencies, and
source bindings. Given a research question, an
agent selects and connects these stored objects to
assemble its evidence input.
SPRA stores research objects independently of
a particular query. Structured JSON encodes each
instance; a versioned profile specifies scientific
roles, relation signatures, cardinalities, and source-
linking rules.
The representation follows the paper’s scientific
argument: a Problem motivates a Contribution;
Components and ExperimentSetups describe what
is built and tested; Measures define evaluation tar-
gets; and Findings record reported observations.
These six roles organize the empirical-AI profile.
Normally one focused Problem expresses the cen-
tral research question, with a second permitted for
an independent question. Finding attributes record
applicable conditions and qualifications. Table 1
specifies the relations connecting these objects and
their admissible endpoints; citation roles express
external dependencies in a separate layer.
3

Statistic Scope Value
AI papers AI literature; 1965–2026 1,059,112
Core units Corpus; 18.84/paper 19,949,289
Typed relations Corpus; 22.73/paper 24,073,381
Evaluation sample 7 subfields; 2025; 1,407 each 9,849
Locator references Evaluation sample 409,082
Table 2: Corpus scale and evaluation scope.
The empirical-AI profile instantiates this con-
tract for method and evidence comparison. Profiles
make the representation extensible while retaining
explicit role and relation constraints.
For paper p, write Sp= (U p, Rp, Cp, Lp, Vp):
typed units, internal relations, citation dependen-
cies, source bindings, and source/profile revisions.
Cpjoins selected citation roles and verbatim sig-
nals to bibliographic metadata by citation key. Un-
der profile π, an internal edge (u, r, v)∈R pis
valid only if u, v∈U pand their types satisfy
r’s permitted signature. For example, motivates
maps a Problem to a Contribution, Component, or
ExperimentSetup. The supports predicate pro-
vides evidence links from research objects to a
Finding, enabling agents to retrieve associated con-
tent and inspect available source evidence.Schema
validitychecks the relation contract, andsource res-
olutionchecks whether a recorded address reaches
a source item.
Citation dependencies express how the focal
paper uses prior research. builds_on identifies
methodological dependencies, uses_component
records adopted components, and compares_with
records comparisons with cited work. Citation
keys join these roles and their verbatim signals
to bibliographic records, enabling dependency
queries and attribution checks. These paper-level
dependencies connect the focal research argument
to the prior work it uses.
Running example.In the unlearning case, the
Measure evaluates the forget-set-only Contribution
and the Finding records its comparison with a base-
line. Paper ownership assigns the proposed method
to the focal paper; citation roles identify external
work. An agent can retain these objects for com-
parison and reopen their source locators to check
the evaluation conditions. Appendix C gives the
records and source bindings.3.2 Corpus and Community Access
SciVerse is the scholarly-data platform that hosts
the SPRA collection and provides web and pro-
grammatic access to paper records. SPRA denotes
the research representation defined in this paper;
SciVerse provides its deployed resource and access
services.
The September 2026 SPRA snapshot contains
1,059,112 structured paper records spanning 1965–
2026, drawn from AI conference literature and
preprints, including arXiv. Each paper is pro-
cessed through the construction workflow below
into a stored SPRA instance, accessible through the
structured-paper services described in Section 3.3.
The evaluation sample comprises 9,996 candi-
date papers, of which 9,849 are source-aligned
and eligible. This domain-stratified sample covers
2025 publications, with 1,407 eligible papers in
each of seven AI subfields (Table 2). Structural
checks and task evaluations therefore assess de-
ployed representations from the same construction
workflow, with balanced coverage across the se-
lected fields.
The accompanying data archive includes the
complete 9,849-paper aligned RQ1 cohort with
all three input representations and frozen study
outputs; the software archive provides offline re-
analysis scripts. Project software is Apache-2.0;
benchmark data and documentation are CC BY
4.0, with source-derived content retaining its own
terms. Appendix A.2 details access and licensing;
Section 3.3 describes how SciVerse exposes SPRA
to researchers and agents.
3.3 Public Interfaces and Data Access
SciVerse exposes structured paper records un-
der the service namePaper Schema. The
/paper-schema prefix identifies this service in
API requests. SPRA specifies the representation
4

studied here, while Paper Schema names the de-
ployed access interface.
Researchers can explore SPRA through Sci-
Verse and access the same paper-level research
objects programmatically through the Paper
Schema API. For example, a JSON POST to
/paper-schema/search returns candidate paper
identifiers; /paper-schema/materials then re-
trieves research objects and relations for selected
papers. Appendix C.2 gives the HTTPS address,
token authentication, and cURL and Python calls.
The Paper Schema Agent Skill makes this access
sequence available to agents through task-oriented
tools for paper discovery, object retrieval, and evi-
dence assembly. SciVerse Agent Tools provides the
integration resources for using these tools in agent
workflows. The SPRA construction procedure and
its extraction implementation are described in Sec-
tion 3.4.
The online corpus and the frozen evaluation
materials serve complementary uses. Token-
authenticated API access supports discovery and
retrieval of the evolving paper collection. The
accompanying data.zip andsoftware.zip pre-
serve the study records and offline reanalysis work-
flow described in Appendix D. Corpus statistics
refer to the September 2026 snapshot; later service
responses may reflect updated records. API access
follows account permissions and quotas, while re-
distribution of source-derived content follows its
source terms.
3.4 Construction Implementation
The SPRA extraction implementation starts from
section-aware Markdown with paragraph markers
produced by MinerU (Figure 2). Five core LLM
calls establish a shared unit census, predict rela-
tions, and fill Problem, Method, and Evidence con-
tent. Two citation passes join roles and verbatim
signals to bibliographic metadata by citation key;
implementation and decoding settings appear in
Appendix A.3.
The assembly code resolves source markers, cap-
tures tables verbatim, reconciles identifiers, and
checks types, endpoints, provenance, and census
coverage. Optional numerical checks flag score
values absent from captured table cells and explic-
itly record unchecked cases. These construction
controls preserve dense evidence and make numer-ical transcription inspectable (Appendix A.3).
3.5 Operations on Research Objects
Typed lookup selects methods or findings; rela-
tion traversal connects them to measures and set-
tings; faceted filtering selects research roles or at-
tributes. The SciVerse Paper Schema API assem-
bles goal-specific objects, relations, and evidence
for paper overviews, literature surveys, benchmark
analysis, method comparison, and reproduction.
Related-paper discovery extends paper or object
queries into literature sets; source opening resolves
recorded locators for verification.
SciVerse presents research objects through read-
ing paths and topic graphs, with source addresses
for verification; the supplement includes an offline
demonstration. The shared registry (Figure 2) pre-
serves object identity across extraction passes and
later tasks.
4 Evaluating Use of the Representation
RQ1–RQ3 assess scientific task quality under fixed
research content and a common primary Reader.
RQ2 and RQ3 additionally measure input and out-
put token use for the evaluated workflows. Sci-
Verse supplies storage, retrieval, and maintenance;
Appendix D.5 defines the token-accounting scope.
4.1 Representation Quality
Structural validity.Structural checks assess
recorded source pointers and relation endpoints
in the sampled object registries. They measure
addressability and registry connectivity.
4.2 Functional Adequacy of Extracted
Content
RQ1 evaluates source-conditioned scientific QA
across problems and contributions, methods and
components, experiments and results, and limita-
tions and assumptions. G-U1 measures correct-
ness, coverage, and qualification against shared
Full Compact-generated references; G-U3 focuses
on Problem + Contribution. Structured questions
test research-unit lookup, internal-relation recov-
ery, and qualification filtering. Full Compact sup-
plies complete parsed article text (Appendix B.2);
model roles appear below.
5

Stage 1
Census +
sidecarsStage 2
RelationsStage 3
Section fillStages 4–5
Assemble +
validateFinal
SPRA artifact
Markdown
§1 Introduction
§2 Method
§3 Experiments
§4 Results
ReferencesMetadata
Title · authors
Year · venue
References
Citation roles
Bibliography
Units
Assign IDs
Record spans
Node registryShared registry
Planned links
part_of
co_contribution
compares_to
evaluates
Graph scaffoldSource binding
Problem
Question / gap
Method
Contribution
Component
Evidence
Setup
Measure
Finding
Linked unitsDeterministic
Merge
Units + relations
Derive resolves
Validate
Profile
Endpoints
Source links
Coverage
Validated JSONJSON
document {...}
sections {...}
relations [...]
notes {...}
Source locators
Internal graph
Units · relations · evidenceStrict separation
References stay externalCitation sidecar
References · roles · provenanceFigure 2: Five conceptual stages of SPRA construction. A shared unit census precedes relation planning and
role-specific filling; assembly joins source-linked records and stores external citations separately. Validation checks
profile conformance, relation endpoints, and source-link integrity.
4.3 Tasks and Research Questions
RQ1: source-conditioned scientific QA.The
evaluation frame contains 9,849 eligible papers
across seven AI subfields and seven question fami-
lies for semantic QA and structured operations.
RQ2: cross-paper reasoning.We evaluate
3,070 complete paired tasks. We score correctness,
grounding, and qualification on directed-relation
and Top-5 synthesis questions.
RQ3: downstream idea generation.We evalu-
ate 1,001 task variants across seven AI domains,
using shared paper selections to generate and as-
sess evidence-grounded proposals.
4.4 Controlled Representation Comparison
Conditions share questions, paper identities and
order, answer format, rubric, and Reader.Full
Compactsupplies complete parsed article text;
Text-RAG, verbatim task-relevant passages; and
Schema-RAG, SPRA objects and internal relations.
Complete multi-paper inputs can exceed context
capacity; retrieval keeps evidence compact. Taskevidence is fixed before inference. Both RQ2 re-
trieval conditions select one primary item per paper
for each of four shared facets. Text-RAG ranks pas-
sages with question-derived BM25F-like queries;
Schema-RAG ranks deterministically assembled
SPRA objects and typed relations.
4.5 Reader and Evaluation Protocol
Figure 3 summarizes model roles and criterion-
based evaluation (Liu et al., 2023; Zheng et al.,
2023; Kim et al., 2025). Llama 4 Scout is the pri-
mary Reader; two long-context RQ2 Full Com-
pact tasks use a GPT fallback (Appendix B).
For RQ1/RQ2, DeepSeek-v4-pro generates shared
reference answers from Full Compact sources.
MiniMax-M3 scores Reader outputs against these
references without condition labels. For RQ3,
DeepSeek-v4-pro assesses implementation, and
MiniMax-M2.5, Qwen3.5-27B, and GLM-5.1 in-
dependently review proposals.
Frozen outputs and scripts support offline replay
of RQ1–RQ3 statistics; Appendices A.4 and D.5
document structural checks and token-accounting
scope.
6

Study Condensed task Required evidence
RQ1 G-U3 Identify the problem and central contribution. Problem + Contribution
RQ1-qual. Bind qualifications to claims or methods. Finding + qualification
RQ2 Assess directed relations; synthesize papers. Typed relations + multi-paper context
RQ3 Generate a grounded benchmark-transfer
idea.Selected papers + qualifications
Table 3: Representative benchmark operations. Task formulations are condensed; the reproducibility package
contains the complete prompts.
Full Compact Text-RAG Schema-RAG
RQ1 / RQ2 Answer quality
Full CompactReference
DeepSeek
Reader
Llama 4Blind judge
MiniMaxAnswers
RQ3 Idea generation
Reader
Llama 4Audit
DeepSeekBlind review
MiniMax
Qwen
GLM
Figure 3: Reader and evaluation protocol for the three
representation conditions. Section 4.5 specifies model
roles and scoring.
4.6 Metrics and Statistical Analysis
The primary reported outcomes compare Schema-
RAG access with Text-RAG for RQ1 G-U1 and
RQ2 directed correctness, and with Full Compact
for RQ2 Top-5 correctness. G-U2, G-U3, ground-
ing, qualification, and token use provide supporting
analyses. RQ1 and RQ2 quality scores use a 0–5
scale. RQ2 qualification measures the conditions
attached to a specified answer. RQ3 reports Overall
quality and input+output tokens.
Paired bootstrap resampling preserves condition
pairing and study sampling structure to estimate
95% confidence intervals (Koehn, 2004); the three
RQ3 Overall contrasts receive Holm correction
(Holm, 1979). Appendix B specifies resampling
units, replicate counts, and tests.Measure Full
CompactText-
RAGSchema-
RAG
G-U12.4201.961 2.342
G-U21.3271.204 1.317
G-U3 3.201 2.1673.216
Table 4: RQ1 mean scores on the 0–5 scale. G-U1:
semantic answer quality judged against references gen-
erated from Full Compact; G-U2: structured operations;
G-U3: Problem + Contribution. Columns show Full
Compact, passage, and SPRA access; bold marks max-
ima.
5 Results
5.1 Structural and Semantic Assessment
In the sampled records, 203,628 of 214,239 core
units (95.0%) in the 9,849-paper frame have a
source pointer, and 106,439 of 109,591 relations
(97.1%) in the complete relation records for 4,350
papers join to both endpoints. These rates measure
addressability and connectivity; RQ1 evaluates
content through scientific QA and structured oper-
ations. Appendix A.4 gives the structural checks.
In a model-based assessment against pub-
lic NCG annotations, SPRA receives a mean
contribution-level semantic-alignment score of
84.49/100 from GPT-4.1. Information-unit align-
ment reaches 4.436/5, supporting correspondence
between the extracted research roles and annotated
contributions (Appendix A.5).
5.2 Extracted Structure for
Source-Conditioned QA
SPRA retains much of Full Compact’s source-
conditioned QA score while outperforming pas-
sage retrieval (Table 4; Figure 4a). G-U1
reaches 2.342/5 against Full Compact’s 2.420/5:
the paired difference is −0.078/5 (95% CI
7

Task Metric Full Compact Text-RAG Schema-RAG Difference / reduction
Answer quality (0–5)
Directed{Correctness 2.583 4.5434.569+0.026 [+0.004,+0.049]
Grounding 3.098 4.3234.368+0.045 [+0.016,+0.073]
Qualification 2.756 4.1044.288+0.184 [+0.154,+0.212]
Top-5{Correctness3.6992.958 3.114+0.156 [+0.114,+0.200]
Grounding3.4393.110 2.966−0.145 [−0.183,−0.105]
Qualification3.7252.858 2.533−0.325 [−0.366,−0.284]
Input+output (millions of tokens)
All tasks Total tokens 203.26 13.309.51−28.5% /−95.3%
Table 5: RQ2 scores (0–5) and input+output tokens across 3,070 paired tasks. Quality differences are Schema-RAG
minus Text-RAG [paired 95% CI]; token reductions are relative to Text-RAG / Full Compact, respectively. Bold
marks the highest quality score in each row and the lowest token total.
[−0.110,−0.046 ]). Relative to passage retrieval,
the gain is 0.381/5 (95% CI [+0.351,+0.411]). G-
U1 combines correctness, coverage, and qualifica-
tion against complete-article references.
For structured operations, SPRA exceeds Text-
RAG by 0.113/5 on G-U2. It also has the highest
observed G-U3 mean for identifying the problem
and contribution, 3.216/5 versus 3.201/5 for Full
Compact. These results support the use of SPRA
for answering source-conditioned questions and
identifying a paper’s problem and contribution.5.3 Cross-Paper Reasoning
Across 3,070 paired tasks (Appendix B.3), SPRA
uses 9.51M input+output tokens: 28.5% fewer than
Text-RAG (13.30M) and 95.3% fewer than Full
Compact (203.26M; Table 5).
Schema-RAG improves directed judgments
across all three measured criteria (Table 5). Rel-
ative to Text-RAG, qualification improves by
0.184/5 (95% CI [+0.154,+0.212]), correctness by
0.026/5, and grounding by 0.045/5. The largest
gain concerns qualification: retaining support, un-
certainty, and applicable conditions.
Top-5 synthesis shows a different result pat-
tern. SPRA improves correctness over Text-RAG
G-U1 quality G-U2 operations012345
2.420
1.3271.961
1.2042.342
1.317Quality score (0–5)Full Compact Text-RAG Schema-RAG
0 50 100 150 200Schema-RAGText-RAGFull Compact 203.26M
13.30M
9.51M95.3% fewer than Full Compact
28.5% fewer than Text-RAG
Input+output tokens (millions)(a) RQ1: single-paper quality (b) RQ2: cross-paper token use
Figure 4: Quality and token use: (a) RQ1 G-U1 semantic QA against Full Compact-generated references and G-U2
structured operations (0–5); (b) input+output tokens across all paired RQ2 tasks, including Top-5. Gray and blue
annotations show SPRA’s reductions relative to Full Compact and Text-RAG.
8

Metric SPRA seeds SPRA +
citations
Cross-paper
integration1.7792.032
Novelty 2.100 2.172
Evidence grounding 1.848 1.931
Qualification
handling2.750 2.744
Independent Overall 2.302 2.379
Table 6: Scientific quality with SPRA seeds and citation
expansion: 1,001 paired questions; quality scores use
0–5. Means average two reviewers; Overall is assessed
independently.
by 0.156/5, while grounding and qualification de-
crease by 0.145/5 and 0.325/5; Full Compact has
the highest means on all three criteria. These syn-
thesis results motivate further evaluation of input
selection that retains claim-specific qualifications
alongside core findings.
5.4 Idea Generation and Citation Expansion
The main RQ3 comparison evaluates evidence-
grounded proposals across 1,001 variants with
shared paper selections. The family-balanced
SPRA–Full Compact Overall contrast is −0.067
points (95% CI [ −0.102,−0.034 ]); the SPRA–
Text-RAG contrast is +0.011 points (95% CI
[−0.029,+0.058 ]). SPRA uses 93.7% fewer in-
put+output tokens than Full Compact and 17.7%
fewer than Text-RAG. Appendix B reports all three
paired contrasts and token totals.
The separate citation-expansion study tests ac-
cess to additional research through represented de-
pendencies. Using Qwen3.5-397B, both conditions
start from the same seed objects under a shared
10,000-token ceiling. Adding up to two resolved ci-
tation neighbors improves cross-paper integration
by+0.253 /5 (95% CI [+0.195,+0.308] ). Table 6
reports dimension-specific results. Appendix B.6
specifies neighbor selection, added evidence, and
paired analysis.
6 Discussion
SPRA makes the research argument an explicit,
reusable object of scientific work. Findings remain
connected to contributions, evaluations, and condi-
tions, while selected citation dependencies connectthe focal paper to prior work. These connections
support queries and relation traversal. Citation
expansion demonstrates their use in assembling
additional evidence under a shared context ceiling.
The evaluations connect representation quality
to scientific use across QA, cross-paper reasoning,
and idea generation. Token savings quantify work-
flow efficiency alongside the scientific utility of
explicit research structure.
7 Conclusion
SPRA addresses the title’s question by exposing
the research structure already present in scientific
papers as persistent, queryable objects and rela-
tions. The empirical-AI profile connects prob-
lems, contributions, methods, evaluations, and find-
ings, with source bindings for inspection and se-
lected citation dependencies for cross-paper access.
SciVerse makes this representation accessible at
million-paper scale through a website, the Paper
Schema API, and an Agent Skill; the extraction
implementation supports construction from parsed
articles.
The evaluations show that SPRA supports
source-conditioned QA and directed cross-paper
reasoning with substantially lower token use in the
evaluated retrieval workflows. RQ1 gains 0.381/5
over passage retrieval while scoring 0.078/5 be-
low Full Compact; RQ2 uses 28.5% fewer in-
put+output tokens than passage retrieval across
all paired tasks. Mixed Top-5 synthesis results and
the lower RQ3 Overall score relative to Full Com-
pact show that compact structure must be assessed
against each workflow’s information needs. SPRA
provides a concrete foundation for research sys-
tems that query, compare, and reuse paper-level
scientific content through a shared representation.
Limitations
Generalization and successive workflows.The
evaluations cover the empirical-AI profile and the
specified task workflows. Transfer to other do-
mains, successive-task reuse, and stability under
meaning-preserving rewrites remain future evalua-
tion directions. The comparisons assess the instan-
tiated representation and access workflows; they
do not isolate the contribution of individual role or
predicate labels.
9

Services and cost coverage.Updates to hosted
services can change behavior in new runs; the re-
sults characterize the recorded service invocations.
Token savings cover model inputs and outputs for
downstream tasks; storage, retrieval, and mainte-
nance are provided by SciVerse.
Ethical Considerations
SPRA preserves attribution and inspectable source
links. Automated extraction may omit or misstate
qualifications, which can change how a research
claim is understood. Corpus coverage is shaped by
the included venues, years, and source availabil-
ity, so missing or underrepresented work should
not be taken as evidence that a topic or finding is
absent from the literature. Users may also over-
rely on concise structured records and lose context
available in the full article. For important research
or other consequential uses, users should check
the relevant claims and qualifications against the
source article. Access and redistribution follow
source licenses; researchers retain responsibility
for interpretation and downstream validation.
Acknowledgements
OpenAI Codex assisted with editorial revision and
routine monitoring of experimental progress. The
authors reviewed the assisted work and take re-
sponsibility for the research decisions, analyses,
citations, interpretation, and presentation.
References
Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi,
Amanpreet Singh, Joseph Chee Chang, Kyle Lo,
Luca Soldaini, Sergey Feldman, Mike D’Arcy, David
Wadden, Matt Latzke, Jenna Sparks, Jena D. Hwang,
Varsha Kishore, Minyang Tian, Pan Ji, Shengyan
Liu, Hao Tong, and 9 others. 2026. Synthesizing sci-
entific literature with retrieval-augmented language
models.Nature, 650(8103):857–863.
Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan,
and Sung Ju Hwang. 2025. ResearchAgent: Itera-
tive research idea generation over scientific litera-
ture with large language models. InProceedings of
the 2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), pages 6709–6738, Albuquerque, New
Mexico. Association for Computational Linguistics.Sebastián Brzovic, Cristóbal Rojas, and Andrés
Abeliuk. 2025. Networks multiscale entropy analy-
sis.Preprint, arXiv:2510.11524v1.
Lingjiao Chen, Matei Zaharia, and James Zou. 2023.
FrugalGPT: How to use large language models while
reducing cost and improving performance.Preprint,
arXiv:2305.05176.
James Cheney, Paolo Missier, and Luc Moreau. 2013.
Constraints of the PROV data model. W3C recom-
mendation, World Wide Web Consortium.
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan,
Noah A. Smith, and Matt Gardner. 2021. A dataset
of information-seeking questions and answers an-
chored in research papers. InProceedings of the
2021 Conference of the North American Chapter
of the Association for Computational Linguistics:
Human Language Technologies, pages 4599–4610,
Online. Association for Computational Linguistics.
Bofu Dong, Pritesh Shah, Sumedh Sonawane, Tiyasha
Banerjee, Erin Brady, Xinya Du, and Ming Jiang.
2025. SciEvent: Benchmarking multi-domain sci-
entific event extraction. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 17225–17255, Suzhou,
China. Association for Computational Linguistics.
Jennifer D’Souza, Sören Auer, and Ted Pedersen. 2021.
SemEval-2021 task 11: NLPContributionGraph -
structuring scholarly NLP contributions for a re-
search knowledge graph. InProceedings of the
15th International Workshop on Semantic Evaluation
(SemEval-2021), pages 364–376, Online. Associa-
tion for Computational Linguistics.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summariza-
tion.Preprint, arXiv:2404.16130. Version 2, revised
February 2025.
Yichen He, Guanhua Huang, Peiyuan Feng, Yuan Lin,
Yuchen Zhang, Hang Li, and Weinan E. 2025. PaSa:
An LLM agent for comprehensive academic paper
search. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 11663–11679, Vienna,
Austria. Association for Computational Linguistics.
Sture Holm. 1979. A simple sequentially rejective mul-
tiple test procedure.Scandinavian Journal of Statis-
tics, 6(2):65–70.
Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li,
Zifeng Wang, Long Le, Abhishek Kumar, James
Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Kr-
ishna, and Tomas Pfister. 2024. Found in the middle:
Calibrating positional attention bias improves long
context utilization. InFindings of the Association for
Computational Linguistics: ACL 2024, pages 14982–
14995. Association for Computational Linguistics.
10

ICLR. 2026. ICLR 2026 fact sheet. International
Conference on Learning Representations. Accessed
2026-10-13.
Sarthak Jain, Madeleine van Zuylen, Hannaneh Ha-
jishirzi, and Iz Beltagy. 2020. SciREX: A challenge
dataset for document-level information extraction.
InProceedings of the 58th Annual Meeting of the
Association for Computational Linguistics, pages
7506–7516, Online. Association for Computational
Linguistics.
Mohamad Yaser Jaradeh, Allard Oelen, Kheir Ed-
dine Farfar, Manuel Prinz, Jennifer D’Souza, Gábor
Kismihók, Markus Stocker, and Sören Auer. 2019.
Open Research Knowledge Graph: Next generation
infrastructure for semantic scholarly knowledge. In
Proceedings of the 10th International Conference on
Knowledge Capture, pages 243–246. ACM. Open-
access manuscript.
Seungone Kim, Juyoung Suk, Ji Yong Cho, Shayne
Longpre, Chaeeun Kim, Dongkeun Yoon, Guijin Son,
Yejin Cho, Sheikh Shafayat, Jinheon Baek, Sue Hyun
Park, Hyeonbin Hwang, Jinkyung Jo, Hyowon Cho,
Haebin Shin, Seongyun Lee, Hanseok Oh, Noah Lee,
Namgyu Ho, and 13 others. 2025. The BiGGen
bench: A principled benchmark for fine-grained eval-
uation of language models with language models.
InProceedings of the 2025 Conference of the Na-
tions of the Americas Chapter of the Association
for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pages 5877–
5919, Albuquerque, New Mexico. Association for
Computational Linguistics.
Philipp Koehn. 2004. Statistical significance tests for
machine translation evaluation. InProceedings of the
2004 Conference on Empirical Methods in Natural
Language Processing, pages 388–395, Barcelona,
Spain. Association for Computational Linguistics.
Timothy Lebo, Satya Sahoo, and Deborah McGuinness.
2013. PROV-O: The PROV ontology. W3C recom-
mendation, World Wide Web Consortium.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Maria Liakata, Shyamasree Saha, Simon Dobnik, Colin
Batchelor, and Dietrich Rebholz-Schuhmann. 2012.
Automatic recognition of conceptualization zones in
scientific articles and two life science applications.
Bioinformatics, 28(7):991–1000.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How languagemodels use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using gpt-4 with better human align-
ment. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 2511–2522, Singapore. Association for
Computational Linguistics.
Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kin-
ney, and Daniel Weld. 2020. S2ORC: The semantic
scholar open research corpus. InProceedings of the
58th Annual Meeting of the Association for Compu-
tational Linguistics, pages 4969–4983, Online. As-
sociation for Computational Linguistics.
Patrice Lopez. 2009. GROBID: Combining automatic
bibliographic data recognition and term extraction
for scholarship publications. InResearch and Ad-
vanced Technology for Digital Libraries, pages 473–
474. Springer.
Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh
Hajishirzi. 2018. Multi-task identification of entities,
relations, and coreference for scientific knowledge
graph construction. InProceedings of the 2018 Con-
ference on Empirical Methods in Natural Language
Processing, pages 3219–3232, Brussels, Belgium.
Association for Computational Linguistics.
Jakub Lála, Odhran O’Donoghue, Aleksandar Shtedrit-
ski, Sam Cox, Samuel G. Rodriques, and Andrew D.
White. 2023. PaperQA: Retrieval-augmented gen-
erative agent for scientific research.Preprint,
arXiv:2312.07559.
Meta. 2025. Llama 4 Scout model card.
National Information Standards Organization. 2024.
ANSI/NISO Z39.96-2024: Journal Article Tag Suite
(JATS), Version 1.4.
Allard Oelen, Mohamad Yaser Jaradeh, and Sören
Auer. 2024. ORKG ASK: a neuro-symbolic schol-
arly search and exploration system.Preprint,
arXiv:2412.04977.
Open Research Knowledge Graph. n.d. Creating tem-
plates in the ORKG. Accessed 2026-10-11.
Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck,
Thanh Tam Nguyen, Sheng Li, and Tanja Schultz.
2025. Retain-free machine unlearning for speech
emotion recognition.Preprint, arXiv:2510.04251.
Research Object Community. 2026. RO-Crate metadata
specification 1.3. Recommendation, 22 June 2026.
Michael D. Skarlinski, Sam Cox, Jon M. Laurent,
James D. Braza, Michaela Hinks, Michael J. Ham-
merling, Manvitha Ponnapati, Samuel G. Rodriques,
and Andrew D. White. 2024. Language agents
achieve superhuman synthesis of scientific knowl-
edge.Preprint, arXiv:2409.13740.
11

Stian Soiland-Reyes, Peter Sefton, Mercè Crosas,
Leyla Jael Castro, Frederik Coppens, José M. Fernán-
dez, Daniel Garijo, Björn Grüning, Marco La Rosa,
Simone Leo, Eoghan Ó Carragáin, Marc Portier, Ana
Trisovic, RO-Crate Community, Paul Groth, and Ca-
role Goble. 2022. Packaging research artefacts with
RO-Crate.Data Science, 5(2):97–138.
Jiarui Song, Yunheng Shen, Chengbin Hou, Pengyu
Wang, Jinbao Wang, Ke Tang, and Hairong Lv.
2025. Fedaghn: Personalized federated learn-
ing with attentive graph hypernetworks.Preprint,
arXiv:2501.16379v2.
Simone Teufel, Advaith Siddharthan, and Colin Batche-
lor. 2009. Towards discipline-independent argumen-
tative zoning: Evidence from chemistry and compu-
tational linguistics. InProceedings of the 2009 Con-
ference on Empirical Methods in Natural Language
Processing, pages 1493–1502, Singapore. Associa-
tion for Computational Linguistics.
David Wadden, Ulme Wennberg, Yi Luan, and Han-
naneh Hajishirzi. 2019. Entity, relation, and event
extraction with contextualized span representations.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 5784–
5789, Hong Kong, China. Association for Computa-
tional Linguistics.
Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang,
Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan
Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui,
Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Con-
ghui He. 2024. MinerU: An open-source solution
for precise document content extraction.Preprint,
arXiv:2409.18839.
Chenguang Wang, Ming Li, Chengrui Fan, Jianpeng
Chen, Han Chen, Tianyi Zhou, and Dawei Zhou.
2026. A missing piece for trustworthy AI reviewers:
From benchmarking rhetorical robustness to SciCore
review.Preprint, arXiv:2609.39027.
Yujun Wu, Dongxu Zhang, Xinchen Li, Jinhang Xu,
Yiling Duan, Yumou Liu, Jiabao Pan, Qiyuan Zhu,
Xuanhe Zhou, Jingxuan Wei, Siyuan Li, Jintao
Chen, Conghui He, and Cheng Tan. 2026. Intern-
Atlas: A methodological evolution graph as re-
search infrastructure for ai scientists.Preprint,
arXiv:2604.28158v2.
Qi Zhang, Zhijia Chen, Huitong Pan, Cornelia Caragea,
Longin Jan Latecki, and Eduard Dragut. 2024.
SciER: An entity and relation extraction dataset for
datasets, methods, and tasks in scientific documents.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing, pages
13083–13100, Miami, Florida, USA. Association for
Computational Linguistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with mt-bench and chatbot arena.
InAdvances in Neural Information Processing Sys-
tems, volume 36, pages 46595–46623. Curran Asso-
ciates, Inc.
Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and
Wei Hu. 2025. Knowledge graph-guided retrieval
augmented generation. InProceedings of the 2025
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 8912–8924, Albuquerque, New Mexico.
Association for Computational Linguistics.
12

A Resource and Construction
A.1 Resource Scope and Sampling
The corpus contains 1,059,112 SPRA paper repre-
sentations spanning 1965–2026, produced through
the construction procedure in Appendix A.3 and ex-
posed through SciVerse. The source-aligned eval-
uation sample contains 9,849 eligible papers pub-
lished in 2025, with 1,407 per field across seven
AI subfields (Table 7). The cohort follows the
primary Reader’s documented August 2024 knowl-
edge cutoff (Meta, 2025). The sampled SPRA
instances retain their corpus research objects, re-
lations, and source bindings, linking evaluation to
the resource’s production workflow. The resulting
assessments characterize the eligible 2025 cohort
across these seven fields. RQ3 evaluates 1,001 task
variants across seven AI domains, as detailed in
Appendix B.
Corpus statistics describe the collection used
in this study, measured in September 2026. The
supplementary records specify the counting pop-
ulations and preserve the aggregate results and
evaluation data. The SciVerse Paper Schema API
supports continued paper- and object-level access;
Appendix D describes the accompanying evalua-
tion materials for offline inspection and numerical
reanalysis.
The corpus contains 19,949,289 core units
(18.84/paper) and 24,073,381 internal relations
(22.73/paper). Core units comprise Problems, Con-
tributions, Components, ExperimentSetups, Mea-
sures, and Findings; relation counts cover the eight
profile predicates and exclude citations. Machine-
readable statistics report per-type counts and the
year distribution.The sample contains 214,239 core units: 9,853
Problems, 11,395 Contributions, 36,522 Compo-
nents, 37,981 ExperimentSetups, 44,355 Measures,
and 74,133 Findings (mean 21.75/paper; median
21; range 4–213). Separate counts are 9,849 Docu-
ments, 177,003 References, and 8,646 Resources.
Its 409,082 locator references count repeated ad-
dresses separately.
Complete relation retrieval covers 4,350 papers
and 109,591 relations (mean 25.19/paper; median
24; range 4–295). These complete groups define
the structural-check population, rather than a sec-
ond balanced sample.
A.2 Sources, Acquisition, and Resource
Access
Sources and acquisition.The evaluation ros-
ter includes AAAI, ACL, EMNLP, ICLR, ICML,
NeurIPS, CVPR, ICCV , robotics proceedings,
and arXiv. Venue labels identify publica-
tion outlets; recorded arXiv, ACL Anthology,
AAAI, and DOI-linked URLs identify accessed
versions. SciVerse supplies metadata, parsed
sources, and SPRA instances; MinerU produces
section-aware Markdown. The packaged statis-
tics and source-linked examples are grouped
indata/data/supplementary/paper_results.
json and data/data/supplementary/frozen_
subset.json ; the SciVerse Paper Schema API
provides paper- and object-level corpus access.
Service and data access.The accompanying
data archive contains the complete 9,849-paper
aligned RQ1 cohort with all three input represen-
tations: complete text, retrieved passages, and
SPRA. Together with the profile, attributed exam-
ples, corpus statistics, frozen study outputs, and
AI subfield Eligible
Language, speech, and information access 1,407
Vision and multimodal perception 1,407
Learning, optimization, and general ML 1,407
Data, knowledge, and computational systems 1,407
Reasoning, agents, RL, and embodied AI 1,407
Trustworthy and human-centered AI 1,407
AI for science and domain applications 1,407
Total 9,849
Table 7: Eligible evaluation papers across seven AI sampling strata.
13

analysis scripts, this supports offline inspection
and numerical replay without service credentials
(Appendix D). SciVerse exposes the million-record
corpus through the token-authenticated SciVerse
Paper Schema API for paper discovery and access
to typed units, relations, citations, and source con-
text. The API supports discovery by paper meta-
data or research objects, followed by goal-specific
material selection for overview, survey, benchmark,
method, and reproduction tasks. Agents can re-
trieve typed records, follow relations and resolved
citation links, and reopen source context. Mate-
rial responses report available and returned counts,
including truncation, so callers can request addi-
tional records when needed. The Paper Schema
API documentation specifies request and response
formats; API Tokens are created through the Sci-
Verse account console. The Paper Schema Agent
Skill documentation describes the corresponding
tools and installation. Access is subject to account
permissions and quotas.
Licenses.Project software uses Apache-2.0;
project-authored benchmark records, labels,
prompts, and documentation use CC BY 4.0.
Third-party articles and service payloads retain
their source terms; redistribution follows the terms
of each source. The seven inspection instances
retain CC-BY attribution. Evaluation metadata
records 8,340 positive, 569 negative, and 940 un-
specified open-access flags, separately from redis-
tribution permissions in the source-version access
records.
A.3 Construction Procedure
MinerU supplies Markdown with paragraph mark-
ers and verbatim tables. Five core calls share a
paper-owned registry: an initial extraction stage
assigns unit identifiers and source markers; a struc-
tural pass predicts part_of ,co_contribution ,
compares_to , and evaluates ; three role-specific
passes enrich Problem, Method, and Evidence
units and predict relations such as motivates ,
about , and supports . The Evidence pass creates
Findings and binds their measures and conditions.
Asupports edge connects a Finding to associ-
ated research objects, giving agents direct access
to evidence-bearing content and, where recorded,
its source locations.Separate metadata, citation-role, and reference-
metadata calls form the bibliographic and external-
dependency layers. Deterministic assembly recon-
ciles identifiers, merges records, derives resolves ,
and validates fields, endpoints, ownership, and
unit coverage. Prompts constrain admissible types,
identifier reuse, and evidence; citation keys join ex-
ternal records without merging them into internal
claims.
The SPRA extraction implementation uses a
shared paper-level registry and role-specific extrac-
tion stages. Its model configuration and request
settings are summarized in Appendix D.3 and the
supplementary registry.
Validation.Required extraction stages must pro-
duce schema-valid records; unresolved validation
issues exclude a paper from task use. Assembly
normalizes identifiers and source markers and re-
moves duplicate, dangling, self-loop, unknown-
predicate, and invalid-signature edges. Metadata
or citation-extraction errors are retained with warn-
ings when core extraction remains usable.
Endpoint contract.For compares_to , both
source and target types belong to {Contribution,
Component, ExperimentSetup, Measure}. For
motivates , the source is a Problem and the target
is a Contribution, Component, or ExperimentSetup.
Every internal endpoint must belong to the same
paper’s unit registry. Table 1 lists the eight inter-
nal relation types and their admissible endpoint
signatures.
Source markers are normalized against parsed
blocks and headings. Tables are sliced verba-
tim. An optional numeric-presence checker flags
Measure values absent from captured table cells,
records unchecked cases, and leaves units and
scores unchanged. The construction rules require
preservation of values, units, comparison bases,
and conditions; numeric presence alone does not
verify these field associations.
A.4 Structural Validation
These checks quantify two prerequisites for in-
specting represented claims: source traceabil-
ity and relation endpoint connectivity. Among
214,239 sampled research units, 203,628 (95.0%)
have a nonempty provenance field. In the com-
plete relation records for 4,350 papers, 106,439 of
14

109,591 relations (97.1%) join to both endpoints;
102,024 of these joined relations have provenance.
These measures characterize the availability of
source pointers and connected relation endpoints
for agent access. The supports relation provides
evidence links between research objects and Find-
ings; agents can follow these links to retrieve as-
sociated content and inspect the source evidence
when available. RQ1 complements these checks
by evaluating the scientific use of extracted content
in source-conditioned answers.
A sampled object assessment provides comple-
mentary record-level evidence; the supplementary
materials document its protocol and records.
A.5 Contribution-Level Semantic Alignment
A paper representation should preserve the sci-
entific contributions expressed in the source ar-
ticle. We assess this property against the pub-
lic annotations of NCG/SemEval 2021 Task 11
(D’Souza et al., 2021), comparing SPRA objects
with annotated contribution sentences, concepts,
information-unit families, and contribution facts.
The projection implementation reads precom-
puted extraction records and maps their types and
content to NCG information-unit families: Prob-
lem objects supply research problems, Experi-
mentSetup objects supply experimental settings,
and Finding and Measure objects supply results.
For each gold contribution sentence, it selects the
extracted string with the highest token F1 as a can-
didate alignment. GPT-4.1 receives the gold anno-
tations and projected content, then judges semantic
correspondence rather than lexical overlap. Thus,
candidate matching organizes the evidence for a
gold-conditioned coverage assessment. The rubric
accepts paraphrase and canonicalization, and pe-
nalizes unsupported additions, incorrect attribution,
and omitted contribution facts. Five dimensions
receive scores from 0 to 5, alongside a separately
judged Overall score from 0 to 100. These paper-
level judgments measure the preservation of anno-
tated contribution structure.
SPRA achieves a mean Overall score of 84.49
and a median of 85. Information-unit alignment
scores higher (4.436/5) than contribution-fact cov-
erage (3.890/5; Table 8). This pattern suggests that
SPRA captures the organization of scientific con-
tributions more consistently than their fine-grainedDimension Mean
Contribution-sentence coverage 3.967
Key-concept coverage 4.031
Information-unit alignment 4.436
Contribution-fact coverage 3.890
Evidence traceability 4.998
Overall (0–100) 84.49
Table 8: NCG contribution-level semantic assessment.
Dimension scores use 0–5; Overall is a separately re-
turned model score on 0–100.
factual detail. The assessment complements RQ1
by examining correspondence with annotated sci-
entific content before its use in downstream an-
swers.
The supplement provides the projection and
judge implementations, rubric, aggregate scores,
domain summaries, and paper-level call metadata.
Complete run-specific judge inputs and individ-
ual judgments are unavailable; the retained code
documents the procedure, while the accompany-
ing NCG documentation identifies the additional
records needed to reproduce the reported aggregate
scores.
A.6 Structural Comparison of Scholarly
Representations
Table 9 compares documented capabilities of schol-
arly representations and infrastructure. JATS ad-
dresses article markup (National Information Stan-
dards Organization, 2024); PROV-O defines the
provenance ontology (Lebo et al., 2013), while
PROV-CONSTRAINTS specifies validity con-
straints (Cheney et al., 2013); RO-Crate addresses
research-object packaging (Research Object Com-
munity, 2026). SciREX and NLPContribution-
Graph (NCG) address scientific extraction, while
ORKG provides a maintained scholarly graph (Jain
et al., 2020; D’Souza et al., 2021; Jaradeh et al.,
2019; Open Research Knowledge Graph, n.d.; Oe-
len et al., 2024). The comparison concerns their
published scope, rather than measured extraction
accuracy.
Schema coverage and granularity.SciREX rep-
resents four entity types—Method, Task, Dataset,
and Metric—with mentions, coreference, salience,
and document-level n-ary result relations. NCG or-
15

CapabilityJATS
1.4 ORKG PROV-ORO-Crate
v1.3 SciREX NCGSPRA
Paper-associated research units△✓△ △✓ ✓ ✓
Typed scientific relations△✓△ △✓ ✓ ✓
Units linked to source content△ △ △ △✓ ✓ ✓
Persistent resource identifiers△✓ ✓ ✓△ △✓
Explicit schema/profile validation✓ ✓△✓△ △✓
Scientific lookup/traversal API×✓× × × △✓
Matched downstream task evaluation× △ × × × ×✓
Condition/qualification access△ △ △ △ △ △✓
Version/revision metadata✓ ✓ ✓ ✓△ △✓
Table 9: Feature coverage of adjacent scholarly representations. ✓denotes documented core support, △partial
support or support requiring a profile, integration, or extension, and ×no documented core support for the stated
capability. NCG = NLPContributionGraph.
ganizes contribution sentences, phrases, and triples
under 12 information units and records a from
sentence association. Both therefore already as-
sociate extracted content with a paper and its text.
SPRA’s six-role empirical-AI profile instead makes
the research argument explicit: Problems motivate
Contributions, Components and ExperimentSetups
describe implementation and evaluation context,
and Measures connect to Findings with conditions.
Its eight internal predicates have admissible end-
point signatures (Table 1). Type counts describe
granularity, not a coverage or quality ranking.
From scientific tasks to the profile.Task-
derived roles express a design mapping from
scientific operations to representation require-
ments. Paper-understanding questions call for
Problem and Contribution objects and their
motivates /resolves links. Method comparison
calls for Components and ExperimentSetups, mem-
bership, and comparison edges. Interpreting re-
ported evidence calls for Measures and Findings,
evaluates /about /supports links, and qualifica-
tion attributes. Here, supports is an extracted can-
didate evidence association for navigation; seman-
tic support is assessed by comparing a recorded
source locator, when present, with the Finding. Ver-
ification requires source locators; reuse requires
stable paper-scoped identifiers. This is an explicit
task-to-profile design rationale, rather than an au-
tomated schema-induction procedure.
Persistence, source binding, and operations.
ORKG already provides persistent, paper-linkedcontributions, templates with ranges and cardinal-
ities, comparisons, and APIs. Paper ownership
is a shared design property of ORKG and SPRA.
Source binding also differs in granularity: SciREX
retains mention spans, NCG links triples to contri-
bution sentences, and SPRA records addresses for
passages, tables, and figures. ORKG can represent
additional evidence and qualifiers through proper-
ties and templates. SPRA’s contribution is the inte-
gration of a paper-level research-argument profile,
constrained internal relations, source-opening oper-
ations, and a common object registry into a corpus-
scale extraction and access workflow. Typed
lookup, traversal, and filtering operate on these
same objects; RQ1–RQ3 assess their scientific use
under matched tasks. Matched downstream evalua-
tion in the table refers to representation-access stud-
ies, distinct from extraction benchmarks. Property
alignment connects this task-oriented representa-
tion to ORKG’s broader aggregation infrastructure.
A.7 ORKG Graph Sample and
Interoperability Scope
We examine interoperability using a sample of 200
papers from ORKG’s Artificial Intelligence field
and subfields. We select 200 papers from a 2,918-
paper frame by deterministic ordering before in-
specting graph properties. The sample includes
313 contribution roots and 6,578 statement occur-
rences, with contribution subgraphs traversed to
three hops.
The mapping analysis examines how ORKG
research-role classes and predicate labels corre-
16

Component Description
Sample frame 2,918 listed papers; 200 selected
deterministically
Graph contexts 313 contribution roots; three-hop
traversal
Scientific
statements6,578 statement occurrences
Mapping rules Explicit research-role classes and
predicate labels
Source identity Scientific resource identifiers and
paper-reference links
Table 10: ORKG graph sample and mapping specifica-
tion used to examine interoperability.
spond to the SPRA profile while preserving sci-
entific identities, reference-paper links, and rela-
tion directions. The graph sample, mapping rules,
and protocol are grouped in the ORKG section
ofdata/data/supplementary/paper_results.
json . These materials characterize the interop-
erability interface between the two representations.
B Evaluation Details
B.1 Protocol and Analysis Population
Complete paired analysis sets.Table 11 gives
the denominators behind the reported comparisons.
RQ1 excludes 411 items without complete three-
condition results: 383 stop at Reader generation, 21
at reference generation, and seven at primary judg-
ing. G-U3 is the Problem + Contribution subset of
G-U1; it is not an additional task population.
We compare representations using paired tasks
under a common primary Reader and evaluation
protocol. Evidence inputs are fixed before in-
ference, and the model configurations and study
records are provided in Appendix D.3.
All RQ1–RQ3 and citation-expansion quality
scores use the 0–5 scale. Judge dimensions
recorded on the 0–4 rubric are reported as s=
5r/4 . Paired differences and their intervals are ex-
pressed in score points. Rates and input+output
tokens use their stated units.
Llama 4 Scout is the primary Reader for RQ1–
RQ3; two long-context RQ2 Full Compact tasks
use the documented GPT fallback. DeepSeek-v4-
pro generates shared RQ1/RQ2 answer references
from Full Compact; MiniMax-M3, blind to condi-
tion, scores Reader responses against them. TheRQ1 evaluation frame contains 9,849 items across
seven AI subfields. RQ2 evaluates 3,070 complete
paired tasks; RQ3 evaluates 1,001 variants. Ta-
ble 12 lists the primary comparisons.
SPRA names the representation; Schema-RAG
names its access condition in this comparison. Task
evidence is fixed before Reader inference. For
RQ2, paper groups, source revisions, questions,
and the per-paper, per-facet selection slots are
shared across conditions; representation-specific
input packs are frozen before inference, and re-
trieval is independent of generated answers. Both
rankers use question-derived queries and BM25F-
like field weighting to select one primary evidence
item per paper for each of four scientific facets.
Text-RAG ranks source passages; Schema-RAG
ranks SPRA objects produced through schema-
constrained extraction and deterministic assembly,
together with typed internal relations. The shared
slots match selection opportunities while preserv-
ing each representation’s distinct evidence form
and resulting input length.
B.2 Controlled Representation Comparison
Full Compact input boundary.Full Compact
exposes the complete available parsed article text
in source order. This includes table contents, for-
mulas, and figure captions where retained in the
textual extraction. Figure pixels and page images
are not supplied as visual inputs; image links in
the text are not dereferenced. Thus, completeness
refers to the parsed textual representation, not to
every visual element of the original PDF. For RQ1,
this text supplies the source for shared reference
answers. Text-RAG exposes selected passages;
Schema-RAG exposes SPRA objects and typed in-
ternal relations. Llama 4 Scout answers from each
assigned condition, and MiniMax-M3 scores those
responses against the Full Compact-generated ref-
erences. Questions, paper identities, response con-
tracts, and scoring are fixed.
Full-text context handling.RQ2 Full Compact
concatenates complete section-aware article views
in shared paper order. Groups that fail the in-
put gate are excluded jointly rather than truncat-
ing the full-text condition into eligibility. The 32
input-gated groups are outside the 3,422-task input-
eligible population. RQ3 evidence digestion pro-
cesses at most one Full Compact paper per call and
17

Study or outcome Analysis frame Complete pairs Incomplete
RQ1, all question families 9,849 9,438 411
RQ1 G-U1, semantic QA 5,628 5,510 118
RQ1 G-U2, structured operations 4,221 3,928 293
RQ1 G-U3, Problem + Contribution 1,407 1,378 29
RQ2, directed 2,461 2,431 30
RQ2, Top-5 961 639 322
RQ3, proposal generation 1,001 1,001 0
Table 11: Analysis populations and complete paired observations. The frame is source-aligned for RQ1 and
input-eligible for RQ2; RQ3 includes all planned variants. G-U3 is included within G-U1.
Study Evaluation
scopePrimary comparison
RQ1 9,849 Schema-RAG vs. Text-RAG
(answer quality)
RQ2 3,070
complete
pairsCorrectness: Schema-RAG
vs. Text-RAG (directed);
Schema-RAG vs. Full
Compact (Top-5)
RQ3 1,001 Overall idea quality, three
conditions
Table 12: Evaluation scope and primary comparisons.
at most two papers per RAG call, with a 65,000-
token batch gate. These limits govern admissible
inputs; token accounting includes the complete re-
quests.
RQ1 metric construction.Native judge dimen-
sions are scored from 0 to 4 and displayed as
s= 5r/4 . G-U1 averages correctness, answer cov-
erage, and qualification over semantic questions;
G-U2 uses structured correctness on the three oper-
ation families; G-U3 applies the G-U1 scoring rule
to Problem + Contribution questions. These are
rubric-based model judgments against shared refer-
ences. They measure task performance, rather than
probabilities that individual extracted records are
correct. On the 0–5 reporting scale, the retained G-
U1 SPRA–Full Compact contrast is −0.078 (95%
CI [−0.110,−0.046 ]). The references can contain
source details absent from either selected represen-
tation, so the comparison assesses retained infor-
mation together with Reader performance. Token
totals are reported within their respective studies.B.3 RQ2 Token Accounting
Token accounting covers the same 3,070 scored
tasks in all three conditions: 2,431 directed two-
paper tasks and 639 Top-5 tasks. For each task and
condition, we sum one retained Reader input and
its corresponding output. Inputs include evidence,
instructions, task metadata, output-format instruc-
tions, and chat formatting. Reference generation,
judging, failed calls, and retries are outside this
retained-call measure.
The shared offline BERT WordPiece tok-
enizer ( bert-base-uncased ; frozen account-
ing identifier offline_bert_wordpiece_
conservative_gate_v1 ) measures both in-
put and output length. These standardized counts
describe model-visible text volume; provider
billing uses its own tokenization and input/output
rates. In Full Compact / Text-RAG / Schema-RAG
order, input totals are 199,460,508 / 10,678,861
/ 6,626,310 and output totals are 3,801,534 /
2,617,712 / 2,885,511. Their sums are 203,262,042
/ 13,296,573 / 9,511,821 tokens, reported as 203.26
/ 13.30 / 9.51 million.
Reductions use the unrounded aggregate totals,
100(1−T SPRA/Tbaseline ): 28.5% relative to Text-
RAG and 95.3% relative to Full Compact. Ta-
ble 13 separates the two task types and reports
per-task input+output distributions. Means divide
each group’s total by its task count; pooled means
weight tasks equally. The supplement provides the
individual counts and unrounded summaries.
B.4 RQ3 Idea-Generation Quality and Token
Use
RQ3 evaluates 1,001 idea-generation variants
across seven AI domains.
18

Tasks (n) Condition Mean Median Q1–Q3 P90
All (3,070) Full Compact 66,209 56,093 39,863–88,487 120,199
All (3,070) Text-RAG 4,331 3,550 3,204–4,157 7,591
All (3,070) Schema-RAG 3,098 2,557 2,438–2,773 5,335
Directed (2,431) Full Compact 54,032 48,610 36,786–64,701 85,691
Directed (2,431) Text-RAG 3,472 3,413 3,125–3,704 3,973
Directed (2,431) Schema-RAG 2,511 2,506 2,412–2,604 2,704
Top-5 (639) Full Compact 112,534 114,423 100,636–127,501 136,319
Top-5 (639) Text-RAG 7,600 7,535 7,155–7,968 8,463
Top-5 (639) Schema-RAG 5,333 5,323 5,156–5,508 5,687
Table 13: RQ2 input+output tokens per retained Reader call. Each task contributes one call per condition. Q1–Q3
denotes the interquartile interval; P90 is the 90th percentile. Quantiles use linear interpolation; displayed values are
rounded to the nearest token. Schema-RAG is SPRA-based access.
Access
conditionInput representation Overall (0–5)
Full Compact Complete text 2.965
Text-RAG Retrieved passages 2.886
Schema-RAG SPRA research objects
and relations2.898
Table 14: RQ3 conditions and independently judged
Overall means across 1,001 variants (0–5). Means
weight variants equally; the paired contrasts in Table 15
weight benchmark families equally.
Full Compact supplies complete text, Text-RAG
supplies retrieved passages, and Schema-RAG sup-
plies SPRA research objects and internal relations.
These conditions use the same paper selections and
order.
Tables 14 and 15 report the question-weighted
means and family-balanced paired tests. Overall is
independently judged rather than averaged from di-
mensions. Intervals use a 50,000-draw benchmark-
family cluster bootstrap; paired tests enumerate all
214family sign flips.
Input+output accounting.Table 16 sums input
and output tokens for digestion, generation, and
binding. Shared selection, reference generation, au-
diting, and review are outside this task-generation
measure.
Storage, retrieval, and maintenance are supplied
by SciVerse.
Evaluation scope.The studies use shared bench-
mark data and evaluators to define inference clus-
ters. Some tasks transfer benchmark task formsto public datasets rather than reproduce complete
source experiments.
B.5 Statistical Inference
RQ1 uses 10,000 domain–family stratified paired-
bootstrap replicates, preserving the paired-
comparison principle (Koehn, 2004). RQ2 uses
10,000 multi-membership paper-cluster percentile
replicates: unique papers are sampled multinomi-
ally, and each task receives the mean multiplicity
of its member papers. Both preserve within-task
condition pairing. RQ3 averages the three Over-
all reviews within each task–condition and equally
weights cluster means, with each cluster defined
by shared benchmark data and an evaluator. The
statistical analysis uses 50,000 benchmark-cluster
percentile-bootstrap draws, enumerates all paired
cluster sign flips for the exact test, and applies
Holm correction across the three Overall contrasts
(Holm, 1979).
The paired analyses use tasks with results under
all three conditions and the specified evaluators.
Appendix D describes numerical replay and sup-
porting materials.
B.6 Citation-Guided Research Integration
Across 1,001 variants from 14 benchmark families
and seven domains, both conditions read SPRA ob-
jects from the first two condition-neutral selected
papers. The expanded condition additionally opens
up to two resolved outgoing one-hop citation tar-
gets with their recorded roles. Seeds, prompt, and
10,000-token ceiling are shared; the contrast tests
relation-guided integration within this budget.
19

Contrast Difference 95% cluster-bootstrap interval ExactpHolmp
Schema-RAG–Full Compact−0.067 [−0.102,−0.034]0.0020 0.0059
Schema-RAG–Text-RAG+0.011 [−0.029,+0.058]0.6792 0.6792
Text-RAG–Full Compact−0.078 [−0.129,−0.031]0.0096 0.0193
Table 15: RQ3 paired Overall contrasts across 1,001 variants, expressed in score points. Intervals use the 50,000-
draw benchmark-family cluster bootstrap; exact pvalues enumerate all 14-family sign flips. Each of the 14
benchmark families has equal weight in the paired estimate; Holm correction covers the three contrasts.
Workflow Input+output (millions) Reduction
Full Compact 155.92 –
Text-RAG 11.87 92.4%
Schema-RAG 9.77 93.7%
Table 16: RQ3 input+output token accounting across the generation workflow. Reductions are relative to Full
Compact.
Selection and generation.Task-keyword over-
lap selects seed objects with research-role cover-
age and a shared per-paper cap; internal edges join
visible endpoints. Citation neighbors rank by key-
word overlap in target metadata and role evidence,
with paper-ID tie-breaking. The first two avail-
able records among the top ten are opened; ab-
sent neighbors leave seeds unchanged. Generation
uses qwen3.5-397b-a17b , temperature 0, disabled
thinking, and 2,800 output tokens. Proposals bind
hypotheses and plans to visible object and depen-
dency evidence.
The contrast evaluates the citation-guided ex-
pansion workflow as a whole: it changes access
to neighboring evidence. The study does not iso-
late the contribution of relation labels or compare
alternative neighbor-selection strategies.
Scientific quality. MiniMax-M2.5 andglm-5.1
independently score proposals against the same
union of source excerpts, with condition labels
and execution outcomes hidden. Reported means
average both reviewers after the individual-rating
transformation specified in Appendix B.
We report cross-paper integration directly and
summarize the four scientific dimensions with an
equal-weight descriptive composite:
Ssci= (N+G+I+Q)/4,(1)
where Nis novelty, Gevidence grounding, Icross-
paper integration, and Qqualification handling.
This descriptive composite and the independentlyjudged Overall score both use 0–5. Table 17 gives
the component scores and family-balanced con-
trasts.
Paired inference and evidence use.We average
paired differences within each of 14 benchmark
families, weight families equally, and bootstrap
families 10,000 times. The full cohort includes
the 28-question pilot and 973 remaining ques-
tions. The latter’s integration contrast is +0.254 /5
([+0.198,+0.310] ), positive in all families; the
composite is +0.100 ([+0.061,+0.141] ), positive
in 13. Opened neighbors appear in 932/1,001 ex-
panded proposals.
Reproducibility materials.The accompanying
evaluation package contains 4,004 numeric re-
viewer records, frozen citation edges, study
protocol, and full/remaining-cohort estimates
indata/data/supplementary/paper_results.
json . The reported composites and intervals
can be inspected alongside the underlying model-
reviewer scores. The analysis includes all complete
proposals and records reference conformance sep-
arately.
20

Metric Seed Expanded∆95% CI
Cross-paper integration 1.779 2.032+0.253 [+0.195,+0.308]
Novelty 2.100 2.172+0.072 [+0.037,+0.105]
Evidence grounding 1.848 1.931+0.083 [+0.029,+0.139]
Qualification 2.750 2.744−0.006 [−0.076,+0.055]
Table 17: Scientific synthesis with SPRA relation expansion across all 1,001 paired questions; absolute scores use
0–5 and paired differences use score points. ∆is the equal-weight mean over 14 benchmark families; brackets give
95% family-clustered bootstrap intervals. Seed and Expanded both use SPRA objects.
C Research Objects in Use
The three cases are drawn from the publicly avail-
able papers on speech-emotion unlearning (Ren
et al., 2025), personalized federated learning (Song
et al., 2025), and multiscale network entropy (Br-
zovic et al., 2025). They illustrate how SPRA con-
nects each paper’s problem and contribution to its
evaluation and findings, making the research argu-
ment easier to inspect and reuse.
Machine unlearning.The speech-emotion un-
learning study of Ren et al. (2025) asks how to
remove sensitive information using only the forget
set, without access to retained training examples.
SPRA links this Problem to the proposed forget-
set-only method, its unlearning-accuracy Measure,
and the reported comparison with a baseline. An
agent can retrieve the contribution and inspect the
conditions attached to its result.
Personalized federated learning.In personal-
ized federated learning, the FedAGHN case links
static collaboration to the proposed method and
reported overhead, scalability, and robustness find-
ings (Song et al., 2025). These connected records
let an agent inspect a method alongside its evalua-
tion.
Multiscale network entropy.The multiscale-
entropy case connects the limits of a single-scale
description to a proposed framework and its eval-
uation using R2, including a reported comparison
of0.92and0.68(Brzovic et al., 2025).
C.1 Application Interface
SciVerse demonstrates the SPRA interface through
paper discovery, typed lookup, filtering, relation
traversal, and source inspection. It presents re-
search units and their relationships around a ques-tion, with links to source passages and tables. An
offline walkthrough is included in the accompany-
ing materials; RQ1–RQ3 evaluate scientific task
performance.
C.2 From a Research Question to Reusable
Objects
Consider the speech-emotion unlearning example
in Appendix C. A researcher asks which proposed
method removes information using only the for-
get set and under what evaluation conditions it
succeeds. The access sequence is to discover the
paper, select its Problem and Contribution, follow
relations to the relevant Measure and Finding, and
inspect the recorded source context. Keeping the
paper and object identifiers with the selected evi-
dence allows subsequent comparisons to refer to
the same research objects.
REST access.Create a SciVerse API Token
through the SciVerse account console and provide
it as SCIVERSE_API_TOKEN in the local environ-
ment. Requests use a Bearer authorization header.
The Paper Schema API accepts JSON request bod-
ies for its POST operations. The examples use
https://api.sciverse.space as the API base
URL and /paper-schema as the service prefix.
With the token configured, a paper search can be
issued directly from a shell:
BASE=https://api.sciverse.space
TOKEN="${SCIVERSE_API_TOKEN:?Set token}"
AUTH="Authorization: Bearer $TOKEN"
curl --fail-with-body \
-X POST "$BASE/paper-schema/search" \
-H "$AUTH" \
-H "Content-Type: application/json" \
-d'{"query":"speech emotion unlearning",
"size":5}'
21

The response contains items , with schema_id
and paper metadata for each candidate, and total ,
which reports the match count and whether it is
exact or a lower bound. Select the intended paper
by its title and metadata. The following Python
helper sends authenticated JSON requests:
import json, os, urllib.request
base = "https://api.sciverse.space"
token = os.environ["SCIVERSE_API_TOKEN"]
headers = {
"Authorization": "Bearer " + token,
"Content-Type": "application/json"
}
def post(route, body):
request = urllib.request.Request(
base + "/paper-schema/" + route,
data=json.dumps(body).encode(),
headers=headers)
with urllib.request.urlopen(
request, timeout=60) as response:
return json.load(response)
Enter the selected schema_id from the search
response to retrieve method-oriented materials:
selected_id = input("schema_id: ").strip()
materials = post("materials", {
"schema_ids": [selected_id],
"goal": "method",
"include_relation_context": True,
"per_schema_entity_limit": 40,
"per_schema_relation_limit": 40,
"per_schema_attribute_limit": 40
})
Each returned item contains schema_paper ,
entities ,relations ,evidence , and terms .
Per-resource counts expose total ,returned , and
truncated . A material package selects content
for the requested goal; a complete object listing re-
quires pagination through the corresponding entity
or relation interfaces. The capability endpoint lists
the current service types; callers should use these
names when selecting resources.
For an object or relation with source mark-
ers, /paper-schema/resolve-provenance ac-
cepts its schema_id and returned marker_nums ,
together with a bounded context window. Preserve
these returned locators; do not derive them from
section numbers. The response provides source
segments and unresolved locators. Inspect warn-
ings when partial is true and retrieve additionalpages or narrower selections when truncated is
true. These examples specify the access procedure;
the paper’s quantitative findings come from the
frozen evaluations.
Capabilities and errors.An authenticated GET
to/paper-schema returns the current contract ver-
sion, supported resource types, and limits. HTTP
401 indicates an authentication problem; HTTP
403 indicates insufficient account permissions.
For HTTP 429, clients should wait according
toRetry-After or the response’s retry_after
value. The returned X-Request-ID supports re-
quest tracing.
Agent access.The Paper Schema Agent
Skill documentation provides installation and
configuration instructions. An agent can
use search_paper_schemas ,query_paper_
entities , query_paper_relations , and
resolve_paper_context for the same sequence,
orbuild_paper_materials for task-specific
selection. Both interfaces retain paper-scoped
identities. Related-paper discovery and resolved ci-
tation traversal support expansion to other papers;
names shared across papers do not by themselves
establish object identity.
D Reproducibility and Resource Use
D.1 Evaluation Materials and Reanalysis
The supplements provide code, tests, model set-
tings, source locators, corpus statistics, task-level
results, SPRA examples, and frozen inputs for of-
fline inspection and reanalysis under the terms in
Appendix A.2.
software.zip contains extraction code,
prompts, schemas, analysis and replay scripts,
and tests. data.zip contains study records and
fixtures plus the complete 9,849-paper aligned
RQ1 cohort, with complete-text, retrieved-passage,
and SPRA inputs for each item. The archive also
records the 9,996-paper candidate catalog and 147
exclusion records: 12 candidates lack a positive
Text-RAG candidate, and 135 are removed to
balance domain–task cells. Each eligible paper
has one saved three-representation input set; study-
specific complete-case counts remain as reported
in Appendix B. Matched inputs support offline
inspection; saved study outputs support numerical
reanalysis without API access. The archives create
22

sibling software/ and data/ directories and
include file checksums. The aligned inputs are
indata/data/main/ ; supporting records are in
data/data/supplementary/.
Offline reproduction.Extract both archives into
the same parent directory. From their shared parent
directory, run the packaged consistency check and
regression tests:
python3 software/software/verify_release.py
python3 -m unittest discover \
-s software/code
The complete analysis commands are docu-
mented in software/REPRODUCE.md . Numerical
reanalysis uses saved records without hosted mod-
els; fresh extraction or answer generation requires
source articles and hosted services.
D.2 Executable Implementation and Tests
The SPRA extraction implementation is included
with the evaluation code. Its prompts and JSON
schemas define the extraction stages; assembly
code validates identifiers, relation endpoints, and
source bindings. Appendix A.3 describes the con-
struction procedure.
The regression tests check selected reported val-
ues and their arithmetic against fixtures in the data
archive. software/REPRODUCE.md documents the
offline commands and the source articles and
hosted services required for fresh model runs.
D.3 Model Configurations and Request
Settings
Llama 4 Scout is the primary Reader for
RQ1–RQ3; DeepSeek-v4-pro generates source-
grounded RQ1/RQ2 references and audits the RQ3
implementation; MiniMax-M3 scores RQ1/RQ2
answers against shared references; and MiniMax-
M2.5, Qwen3.5-27B, and GLM-5.1 indepen-
dently review RQ3 proposals. The sup-
plementary data/data/supplementary/model_
settings.json records model aliases, role-level
output ceilings, and backend-specific request op-
tions. Primary evaluation calls use temperature
0; these per-call settings are distinct from Reader
input-context budgets.
Construction configuration.The reference ex-
tractor uses qwen3.5-35b-a3b , temperature 0, dis-
abled thinking, and output ceilings of 32,768 to-
kens for content and 24,576 for planning auxil-iaries. These settings describe the reference extrac-
tor; the corpus statistics summarize the deployed
collection rather than a single-model construction
run.
Evaluation request limits.The frozen registry
records temperature 0 and top- p1 for RQ1–RQ3.
RQ1 output ceilings are 24,000 tokens for the
Reader, 8,000 for references, and 2,400 for judging.
RQ2 uses 3,500/6,000 for directed/Top-5 answers,
12,000 for references, and 2,400 for judging. RQ3
uses 7,000 for digestion, 6,500 for candidate gener-
ation, 1,800 for binding, 2,200 for implementation
assessment, and 2,400 for review. These are config-
ured output limits; actual consumption is reported
separately. Hosted APIs manage model hardware,
so the recorded aliases and request settings identify
the evaluated services.
D.4 Task-Based Assessment of Extracted
Content
RQ1 assesses the functional adequacy of the ex-
tracted representation through source-conditioned
scientific QA and structured operations. DeepSeek-
v4-pro generates shared answer references from
Full Compact, the complete parsed-text input;
MiniMax-M3, blind to condition, scores Reader
responses against them. G-U1 measures correct-
ness, coverage, and qualification against these ref-
erences; G-U3 focuses on Problem + Contribution.
Source traceability and complete relation endpoints
support inspection and research-object operations.
The supplementary materials also document the
sampled object assessment.
D.5 Token-Accounting Scope
Storage, retrieval, and maintenance are supplied by
SciVerse. Token totals measure complete inputs
and outputs for the retained task calls. They quan-
tify the evaluated workflows’ text volume; they do
not represent the service bill or include storage and
retrieval costs. Appendix B specifies the included
stages and excluded calls for each study.
23