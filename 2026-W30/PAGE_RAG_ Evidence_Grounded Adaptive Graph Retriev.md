# PAGE-RAG: Evidence-Grounded Adaptive Graph Retrieval for Long-Document Question Answering

**Authors**: Xingyu Chen, Junxiu An, Jun Guo, Li Wang

**Published**: 2026-07-21 17:13:22

**PDF URL**: [https://arxiv.org/pdf/2607.19301v1](https://arxiv.org/pdf/2607.19301v1)

## Abstract
GraphRAG improves long-document question answering by introducing structured representations beyond conventional retrieval. However, automatically constructed graphs are inherently incomplete projections of source documents, and treating them as independent knowledge sources may lead to unreliable retrieval and generation. We propose PAGE-RAG, a projection-aware adaptive graph retrieval framework for reliable long-document question answering. PAGE-RAG views graph structures as semantic skeletons that organize and navigate document knowledge, rather than replacing the original knowledge source. Based on this perspective, PAGE-RAG introduces a task-adaptive retrieval routing strategy that dynamically selects appropriate retrieval behaviors according to query requirements. Furthermore, PAGE-RAG incorporates strict knowledge boundary control, ensuring that generated responses remain grounded within available evidence and abstaining from unsupported information beyond the accessible knowledge scope. Experiments demonstrate that PAGE-RAG achieves competitive answer quality while improving retrieval efficiency and knowledge reliability, highlighting the importance of projection-aware graph modeling, adaptive retrieval, and explicit knowledge boundary control for trustworthy GraphRAG systems. The source code is publicly available at https://github.com/CXY0112/PAGE-RAG.

## Full Text


<!-- PDF content starts -->

PAGE-RAG: Evidence-Grounded Adaptive Graph
Retrieval for Long-Document Question Answering
Xingyu Chena, Junxiu An*b, Jun Guoc,e, Li Wangd
aSchool of Software Engineering, Chengdu University of Information
Technology, Chengdu, 610225, China
bSchool of Statistics, Chengdu University of Information
Technology, Chengdu, 610103, China
cCollege of Applied Mathematics, Chengdu University of Information
Technology, Chengdu, 610225, China
dSchool of Artificial Intelligence, Beihang University, Beijing, 100191, China
eKey Laboratory of Mathematical Meteorology, Chengdu University of Information
Technology, Chengdu, 610225, China
Abstract
GraphRAG improves long-document question answering by introducing
structured representations beyond conventional retrieval. However, automat-
ically constructed graphs are inherently incomplete projections of source doc-
uments, and treating them as independent knowledge sources may lead to
unreliable retrieval and generation. We propose PAGE-RAG, a projection-
awareadaptivegraphretrievalframeworkforreliablelong-documentquestion
answering. PAGE-RAG views graph structures as semantic skeletons that or-
ganize and navigate document knowledge, rather than replacing the original
knowledge source. Based on this perspective, PAGE-RAG introduces a task-
adaptive retrieval routing strategy that dynamically selects appropriate re-
trieval behaviors according to query requirements. Furthermore, PAGE-RAG
incorporates strict knowledge boundary control, ensuring that generated re-
sponses remain grounded within available evidence and abstaining from un-
supported information beyond the accessible knowledge scope. Experiments
demonstrate that PAGE-RAG achieves competitive answer quality while im-
proving retrieval efficiency and knowledge reliability, highlighting the impor-
tance of projection-aware graph modeling, adaptive retrieval, and explicit
knowledge boundary control for trustworthy GraphRAG systems.The source
code is publicly available at https://github.com/CXY0112/PAGE-RAG.
Keywords:GraphRAG, retrieval-augmented generation, adaptive retrieval,
arXiv:2607.19301v1  [cs.IR]  21 Jul 2026

knowledge graphs, evidence-bounded generation
Introduction
Large language models (LLMs) and LLM-based agents have rapidly ex-
panded the range of tasks that can be handled with natural-language in-
terfaces, including question answering, planning, tool use, and interactive
knowledge work [1, 2, 3]. As these systems are applied to increasingly com-
plex domains, however, their dependence on finite context windows and para-
metric knowledge remains a central limitation [4]. Even with longer context
models, it is often impractical to place an entire book-length corpus, evolving
knowledge base, or domain archive into the prompt for every query [5, 6].
More importantly, domain-specific answers require not only broad language
competence but also faithful access to the relevant source evidence. When
the required evidence is absent, incomplete, or buried in long contexts, LLMs
may still produce fluent answers by relying on parametric associations, lead-
ing to hallucinations and unverifiable claims [7].
Retrieval-augmented generation (RAG) has become one of the main ap-
proaches for mitigating this problem [8]. By retrieving external evidence and
conditioning generation on the retrieved context, RAG shifts part of the
burden from model parameters to explicit source material. Sparse, dense,
and hybrid retrieval methods, often combined with reranking, have shown
strong effectiveness for grounding answers in relevant passages [9, 10]. For
long documents and domain corpora, RAG provides a practical way to access
information that cannot be fully carried in the model context. Nevertheless,
standard passage-based RAG primarily treats the corpus as a collection of
retrievable text chunks [11, 12]. It can recover local evidence well, but it
provides limited explicit support for relations that span distant passages,
multi-hop connections among entities and concepts, and global organization
over the document [13, 14].
GraphRAG extends RAG with graph-structured corpus representations,
including entities, relations, semantic paths, and communities [15]. This fam-
ily of methods is attractive because graphs can make cross-passage struc-
ture explicit and can support relational reasoning or global synthesis that
is difficult for passage retrieval alone[16]. Recent GraphRAG systems have
explored community summaries [17], graph-vector retrieval [18], path-based
reasoning [19], and other forms of structural augmentation. However, current
2

GraphRAG systems also expose several limitations. Some systems incur high
construction or query costs, especially when global reasoning requires broad
traversal or community-level map-reduce [20]. Some graph representations
are noisy or incomplete because they are automatically extracted from long,
heterogeneous texts [21]. Others may underperform strong passage-retrieval
baselines on local factual questions, suggesting that graph structure is not a
universal substitute for source passages [22, 23]. Finally, many systems rely
on prompting the generator to stay grounded, but do not make the evidence
boundary an explicit answer-or-refuse decision. As a result, when graph or
passage evidence is relevant but insufficient, the model may still synthesize
an unsupported answer [24].
We address these limitations from a projection-aware view of GraphRAG.
Our hypothesis is that a graph built from a long document should be treated
as a useful but lossy projection of the source text, rather than as a faith-
ful replacement for textual evidence. A graph can organize salient entities,
relations, events, claims, and communities, but it inevitably omits some
atomic facts, qualifications, and local textual details. Therefore, reliable long-
document GraphRAG should optimize not only answer quality and compu-
tational efficiency, but also whether the system respects the boundary of
available evidence. A system that answers more questions at lower cost is
not necessarily preferable if it achieves these gains by producing unsupported
answers when evidence is insufficient.
Based on this hypothesis, we propose PAGE-RAG, aProjection-aware
AdaptiveGraph-EnhancedRetrieval-AugmentedGeneration framework for
long-document question answering. PAGE-RAG maintains both a textual
passage index and a governed graph view of the corpus. At query time,
it adaptively routes questions to local or global retrieval modes, activates
graphpathsandrerankingonlywhenneeded,andkeepseachretrievalbranch
withinaboundedevidencebudget.Atgenerationtime,PAGE-RAGconnects
evidence sufficiency to the final answer action: the system answers when the
retrieved evidence is sufficient and refuses when it is not. This design aims
to improve the empirical trade-off among answer quality, query efficiency,
construction cost, and knowledge-boundary reliability, rather than universal
dominance over all GraphRAG operating points.
This paper makes the following contributions:
•Weformulatelong-documentGraphRAGasaprojection-awareretrieval
problem, arguing that graph structures provide a compressed semantic
3

skeleton for navigating textual evidence. This motivates an always-on
textual retrieval floor guided by governed graph abstractions.
•We introduce a query-adaptive retrieval routing strategy that matches
retrieval operators to query needs, selecting among textual evidence,
graphneighborhoods,semanticpaths,communitysummaries,andrerank-
ing. This task-aware routing improves retrieval relevance while bound-
ing graph computation and avoiding unnecessary global expansion.
•We introduce evidence-bounded generation as an explicit answer-or-
abstain mechanism, making boundary reliability a first-class optimiza-
tion objective alongside answer quality and efficiency. Across two book-
lengthdatasets,PAGE-RAGcorrectlyrefusesall24unanswerableques-
tions; removing the evidence constraint reduces correct refusal to 4/24
despite increasing apparent answer accuracy.
Related Work
LLMs store substantial factual and procedural knowledge in their pa-
rameters, but this knowledge is difficult to inspect, update, or constrain at
inference time [25, 26, 27]. Neuro-symbolic methods connect neural models
withexternalsymbolicstructuresorexecutorstomakereasoningmorecheck-
able and controllable [28, 29, 30, 31]. For example, PAL translates natural-
language problems into executable programs [32], while Logic-LM translates
reasoningproblemsintosymbolicformulationssolvedbydeterministicsolvers
[33]. RAG can be viewed as the knowledge-access counterpart of this broader
grounding paradigm: generation is conditioned on external evidence that can
be updated, attributed, and inspected.
EarlyRAGworkcombinedparametriclanguagemodelswithnon-parametric
retrieval memories for knowledge-intensive generation [34]. Subsequent work
established sparse, dense, hybrid, and reranked retrieval as the main com-
ponents of practical RAG systems. DPR demonstrated the value of dense
passage retrieval for open-domain QA [35], while BEIR highlighted the com-
plementary cross-domain strengths of lexical, dense, and reranking methods
[36]. Passage-based RAG preserves source evidence and supports relatively
efficient retrieval, but generally represents a corpus as a flat collection of
chunks, providing limited access to cross-passage relations, long-range de-
pendencies, and global thematic structure. RAPTOR addresses part of this
limitation by recursively clustering and summarizing chunks into a tree for
4

multi-level retrieval [37]. However, hierarchical summaries do not directly ex-
pose relation chains or graph topology among entities, events, and concepts.
GraphRAG extends structured retrieval with graph-based corpus repre-
sentations. Microsoft GraphRAG constructs entity graphs, detects communi-
ties, and generates community summaries for query-focused summarization
over private corpora [16]. Its key contribution is to support global sensemak-
ingquestionsthatcannotbeansweredreliablythroughlocalpassageretrieval
alone. Community-level map-reduce improves coverage for such questions,
butcanincursubstantialconstructionandquerycostswhenmanysummaries
must be read and synthesized. LightRAG targets these cost and update chal-
lenges by combining graph structures with vector representations in a dual-
level retrieval scheme [38]. It retrieves from low-level entity relations and
higher-level semantic structures, improving graph-vector retrieval efficiency
and incremental updateability. Nevertheless, automatically extracted graphs
remain lossy projections of the source text; treating them as replacements for
passages may omit local facts, qualifications, and fine-grained constraints.
Other work treats graphs as long-term memory or professional-domain
knowledge infrastructure. HippoRAG uses an LLM-built knowledge graph
and Personalized PageRank to support multi-hop retrieval and knowledge
integration [39], while KAG combines knowledge graphs, vector retrieval,
and logical-form-guided reasoning for professional-domain QA [40]. Together,
these systems demonstrate that graphs can serve as memory indexes, se-
mantic navigation layers, reasoning paths, and knowledge-organization struc-
tures. Their primary focus, however, is graph retrieval or graph reasoning,
leaving less attention to graph projection loss and to the choice of retrieval
operators for different query needs.
Overall, the GraphRAG line has progressed from flat passage retrieval to
hierarchical summaries, community representations, graph-vector retrieval,
and graph-based reasoning. Existing systems address global synthesis, rela-
tion modeling, multi-hop retrieval, or professional knowledge organization,
but do not yet jointly resolve three issues: graph structures cannot fully re-
place textual evidence, retrieval computation should adapt to query require-
ments, and evidence sufficiency should define an explicit answer boundary
rather than remain only a prompt-level instruction. These gaps motivate the
design of PAGE-RAG.
5

1 Projection-aware Hybrid Repository Construction2 Query-adaptive Retrieval Routing3Evidence-bounded Generation
•
•
•Grounding floor / citable source evidence
Evidence binding
b: V∪ Eg → 2T(D)User query q
Routing policy
Π(u,O) → ρ(O)Evidence budget
B = (Bt, Bn, Bp, Bb)
Activate structural retrieval only when needed         Evidence sufficiency check
      s(q,E) ∈ {0,1}
s(q,E)=1?Long-document
collection
D={d1,...,dn}
Document profiling
pi = ϕ(di)
form τi
themes θi
discourse mode μi
structure σi
boundary info κiT(D)={(tj, ej, mj)}Textual evidence repository T(D)
text chunks
tjevidence IDs
ejmetadata
mj
Graph semantic skeleton
G(D)=(V , Eg, C, S)
entities / concepts
semantic relations
communitiesHybrid
repository
R(D)=
(T(D),G(D),P ,b)
T(D)
G(D)
P profiles
b binding
Graph
compresses
and organizes;
text preserves
evidenceQuery profiling
u=ψ(q, R(D))
scope α:
( local / relational / global )
multi-hop βthematic synthesis γ
boundary sensitivity η
Bounded evidence set   E=RΠ,ρ(q,R(D);B)Composable retrieval operators O
Ot
Graph 
neighborhoodOg
Semantic
pathOp Oc
Community
summaryOb
Boundary
recall
Text snippets EtGraph neighborhoods  En Semantic paths  EpCommunity summaries  Ec
... ... ... ...Yes (1) No (0)
Answer a Refusal ⊥
Generate supported
answer with citationsInsuf ficient evidence
→ abstain
Source snippet
Graph path
Community summary
...Based on the provided
evidence, ...
[e21], [e31], [e47], ...Insuf ficient
reliable evidence
is available to
answer this
query .
I abstain.Figure1:PAGE-RAGframework.Thebuildstageconstructsahybridrepositorywithboth
textual evidence and a graph-based semantic skeleton. The retrieval stage routes queries
to textual evidence, graph neighborhoods, semantic paths, and community summaries
according to query type. The generation stage turns evidence sufficiency into an explicit
answer-or-refuse decision.
PAGE-RAG
PAGE-RAG follows the build–retrieve–generate pipeline. In the build
stage, it constructs a hybrid repository that preserves source passages while
deriving a governed graph view as a compressed semantic skeleton. In the
retrieval stage, it routes each query to retrieval operators that match the
query’s evidence needs, combining textual evidence, graph neighborhoods,
semantic paths, community summaries, and reranking under a bounded ev-
idence budget. In the generation stage, it turns evidence sufficiency into an
explicit answer-or-refuse decision. These three stages correspond to the three
challengesidentifiedabove:graphprojectionloss,querymismatchacrosstask
types, and unreliable knowledge boundaries.
Projection-Aware Hybrid Repository Construction
The build stage of PAGE-RAG does not construct only a vector store or
only a knowledge graph. Its goal is to construct a hybrid RAG repository
that contains both source evidence and a graph-based semantic skeleton.
Givenalong-documentcollectionD={d 1, . . . , d n},PAGE-RAGfirstprofiles
6

each document to determine how it should be segmented, extracted, and
organized:
pi= Φ(d i),
whereΦis the document-profiling function. The profilep icontains multiple
metadata dimensions, which we write as
pi= (τ i, δi, µi, σi, κi).
Here,τ idenotes the document form, such as fiction, research paper, report,
or mixed text;δ idenotes the domain or topical distribution;µ idenotes the
dominantdiscoursemode,suchasnarrative,conceptual,orargumentative;σ i
denotes internal structure, such as chapters, paragraphs, or topical sections;
andκ idenotes boundary information, such as chapter order, timeline, or
visible scope. The document profile is not answer evidence by itself. Instead,
it acts as a control signal for repository construction and later retrieval.
The document profile first affects the construction of the textual evidence
repositoryT(D). Narrative documents require the system to preserve chap-
ter order, event spans, and character mentions; conceptual or argumentative
documents require definitions, propositions, argument chains, and thematic
sections to remain recoverable; and global sensemaking corpora require cross-
document topics and summary structure. We represent the textual evidence
repository as
T(D) ={(x j, ej, mj)}N
j=1,
wherex jis a text chunk or evidence span,e jis a citable evidence identifier,
andm jis metadata inherited from or derived from document profiles, such
as chapter, discourse type, entity coverage, topic labels, and boundary scope.
The role ofT(D)is to preserve source evidence and provide the grounding
floor for all subsequent answers.
The document profile also controls the construction of the graph view
G(D). PAGE-RAG does not apply a single fixed schema to all documents.
Instead, it selects knowledge units and extraction behavior according to the
profile. For a spanx jfrom documentd i, we write this profile-conditioned
extraction as
zj=A µi(xj, pi),
whereA µiis an extractor controlled by the discourse modeµ i, andz jis a
candidate knowledge unit. For narrative text, the extracted units may em-
phasize characters, events, locations, and causal relations. For conceptual
7

text, they may emphasize concepts, definitions, attributes, and theoretical
relations. For argumentative text, they may emphasize claims, premises, ev-
idence, and conclusions. This profile-conditioned extraction allows the graph
to adapt to heterogeneous long documents rather than forcing all texts into
one entity-relation schema.
Candidate knowledge units are then refined into a graph-based semantic
skeleton:
G(D) = (V, E g, C, S),
whereVdenotes entity, concept, event, or claim nodes;E gdenotes semantic
relations or path edges;Cdenotes community structure; andSdenotes com-
munity summaries or high-level thematic summaries. We write the governed
graph construction process as
G(D) = Γ({z j}N
j=1, T(D), P),
whereP={p i}n
i=1is the set of document profiles, andΓdenotes the gov-
erned graph-construction function. In practice,Γincludes evidence bind-
ing, schema and domain-range validation, causal protection, confidence rout-
ing, and conservative entity alignment. This formulation emphasizes that the
graph is not an independent knowledge base generated freely from text; it
is a compressed semantic skeleton constructed under both profile constraints
and source-evidence constraints.
To prevent the graph from becoming a lossy substitute for source text,
PAGE-RAG maintains bindings between graph elements and textual evi-
dence:
b:V∪E g→2T(D).
The binding functionbmaps graph nodes and graph edges back to the ev-
idence spans that support them. Through this evidence binding, the graph
can guide navigation and evidence organization, while final answers can still
be traced back to source text or auditable graph-linked evidence. In other
words,G(D)provides semantic compression, relation organization, and re-
trieval navigation, whereasT(D)provides citable and verifiable factual sup-
port.
The resulting hybrid RAG repository is
R(D) = 
T(D), G(D), P, b
.
8

Here,T(D)is the textual evidence floor,G(D)is the graph-based semantic
skeleton,Pis the document-profile set that controls construction and re-
trieval, andbis the evidence binding between graph and text. This construc-
tion implements the projection-aware principle: graph structures compress,
navigate, and organize long documents without replacing source evidence,
while document profiles ensure that different types of texts enter the reposi-
tory through different extraction and organization strategies.
Query-Adaptive Retrieval Routing
After constructing the hybrid repositoryR(D) = (T(D), G(D), P, b),
PAGE-RAG treats retrieval as a query-adaptive routing problem rather than
a single similarity-ranking problem. Given a user queryq, the system first
estimates a query profile:
u= Ψ(q,R(D)),
whereΨis the query-profiling function andudescribes the evidence needs
and reasoning form of the query. We write this profile as
u= (α, β, γ, η),
whereαdenotes the query scope, such as local, relational, or global;βde-
notes whether the query requires multi-hop or path-like connections;γde-
notes whether the query requires thematic synthesis or community-level sum-
maries; andηdenotes boundary sensitivity, namely whether the query may
exceed the current corpus or visible scope. This query profile determines
which retrieval operators should be activated and how much budget should
be allocated to each evidence type.
PAGE-RAG defines a set of composable retrieval operators:
O={O t, On, Op, Oc, Or}.
Here,O tis the textual retrieval operator, which retrieves BM25 or dense
passages fromT(D);O nis the graph-neighborhood operator, which expands
a local subgraph around relevant entities, events, or concepts;O pis the
semantic-path operator, which connects multiple query-related nodes;O cis
the community-summary operator, which retrieves community summaries re-
lated to global themes; andO ris the reranking operator, which selects among
9

candidateevidenceitems.Insteadoftreatingretrievalasasingle-choiceroute,
PAGE-RAG selects a compositional retrieval plan
ρ(q) = Π(u,O),
whereΠis the routing policy andρ(q)⊆ Ois the set of retrieval oper-
ators activated for the current query. The plan is multi-select rather than
mutually exclusive: a complex query may combine textual retrieval, graph
neighborhoods, semantic paths, community summaries, and reranking in the
same evidence package. The core principle is that textual evidence remains
the grounding floor, while graph neighborhoods, paths, and community sum-
maries are activated when they match the query’s evidence needs.
Given a retrieval planρ(q), PAGE-RAG constructs a bounded evidence
set:
E=R ρ(q)(q,R(D);B).
The budgetBcan be distributed across evidence types:
B= (B t, Bn, Bp, Bc),
whereB tcontrols the passage or token budget for textual evidence,B ncon-
trols the graph-neighborhood expansion scope,B pcontrols the number or
length of semantic paths, andB ccontrols the number of community sum-
maries. This budgeted construction prevents the system from applying full
graph traversal or community-wide synthesis to every query, while still al-
lowing structural evidence to be used when the query requires it.
The core motivation of query-adaptive routing is task adaptation and
retrieval relevance. A fixed vector-only route may miss cross-passage rela-
tions; a fixed graph-only route may lose source details; and a fixed global
route may introduce excessive irrelevant context. PAGE-RAG explicitly links
query type, retrieval operators, and evidence budgets throughu,ρ(q), and
B, allowing the system to select appropriate evidence structures for local
factual, relational, multi-hop, and global synthesis questions. Because each
query expands only the passages, graph neighborhoods, semantic paths, or
community summaries that match its evidence needs, the system improves
retrieval focus while reducing unnecessary context expansion and token con-
sumption.
10

Evidence-Bounded Generation
The bounded evidence setEproduced by retrieval is not always sufficient
to support an answer. In long-document RAG, the risk is often not that re-
trieved evidence is entirely irrelevant, but that it is related yet insufficient:
it may mention related themes, characters, events, or concepts without con-
taining the key facts needed to answer the query. If the system is forced to
answer in this situation, the LLM may fill the gap with parametric knowl-
edge or semantic association, producing a fluent answer unsupported by the
current evidence. Long-document RAG therefore needs to make whether to
answer part of the generation decision.
RefusalhastworolesinPAGE-RAG.First,itmaintainsevidence-boundary
reliability:thesystemanswersonlywhenthecurrentevidencesetissufficient,
and otherwise refuses to package related but insufficient evidence as a cited
conclusion. This protects citation traceability and reduces hallucinations in-
duced by partially relevant evidence. Second, refusal provides a preliminary
mechanism for access-boundary management in private RAG settings. RAG
is often applied to professional domains or private corpora, where sensitive
information should be strictly controlled. Although this paper bases refusal
mainly on evidence sufficiency rather than a full permission system, it fol-
lows the principle that a system should answer according to the evidence
boundary available to the current user. This provides a foundation for future
private deployment, permission tiers, and sensitive-information control.
PAGE-RAG defines generation as an evidence-bounded answer-or-refuse
process. Given a queryqand bounded evidence setE, the system estimates
whether the evidence is sufficient:
s(q, E)∈ {0,1}.
Ifs(q, E) = 1, the generator outputs an answerasupported by citations; if
s(q, E) = 0, the system outputs refusal⊥:
y=a,ifs(q, E) = 1,
⊥,ifs(q, E) = 0.
Here,amust be supported by passages, graph-linked evidence, semantic
paths, or community summaries inE, while⊥indicates that the current
evidence is insufficient to answer. Abstention is therefore treated as a correct
system action under insufficient evidence, rather than as a generation failure.
11

This mechanism differs from prompt-only grounding. Asking the model
to answer from context is a soft instruction that still depends on model
compliance, whereas evidence-bounded generation turns evidence sufficiency
into an explicit output boundary. The final refusal boundary prevents related
but insufficient evidence from being completed into unsupported claims, and
provides an interface for future permission tiers and private knowledge gov-
ernance.
Evaluation
We evaluate whether PAGE-RAG provides competitive long-document
QAquality,improvesthebalanceamongquality,queryefficiency,andknowledge-
boundary reliability, and whether its key components contribute as intended.
Experimental Setup
Datasets.WeevaluateontwoEnglishbook-lengthdatasetsandonepub-
licglobalsensemakingbenchmark.Simulacra and Simulation[41]isaconcep-
tual, argumentative book dataset with 94 answerable and 12 unanswerable
questions; its abstract concepts and distributed arguments stress conceptual
relation recovery beyond entity lookup.One Hundred Years of Solitude[42]
is a narrative novel with 88 answerable and 12 unanswerable questions; as a
relationally complex novel, its character interactions and events form long-
range narrative dependencies. These two datasets test local evidence recov-
ery, cross-passage reasoning, thematic synthesis, and closed-corpus refusal.
UltraDomain-Mix [20] contains 125 public global sensemaking questions over
61 documents and 619K tokens, and is evaluated with reference-free pairwise
judgments. This public, heterogeneous benchmark tests generalization be-
yond single-book corpora to cross-domain global synthesis.
Systems.We compare PAGE-RAG with three baselines. BDR-RAG is
our controlled no-graph passage baseline, combining BM25, dense retrieval,
andrerankingoverthesamesourcespans,followingestablishedretrievalcom-
ponents in DPR and BEIR [35, 36]. It is intentionally strong and incurs no
LLM-based graph construction cost. LightRAG [38] is a lightweight graph-
vectorGraphRAGsystem.MicrosoftGraphRAG[16]isaheaviercommunity-
summary and map-reduce style GraphRAG system. We keep system-specific
behavior documented rather than modifying baselines to remove their defin-
ing retrieval mechanisms.
12

Configuration and metrics.All systems use the same source corpora,
the same answer model, the same embedding model, and comparable chunk-
ing or context settings when supported by the baseline. For the two books,
we report strict and lenient accuracy, correct refusal on unanswerable ques-
tions, median latency, query tokens per question, and one-time build to-
kens. For ablations, we use boundary-balanced score (BBS), the harmonic
mean of lenient accuracy and correct-refusal rate:BBS= 2·Acc lenient·
Ref/(Acc lenient +Ref). For UltraDomain-Mix, we report position-debiased
pairwise judgments on four dimensions–comprehensiveness, diversity, em-
powerment, and directness–together with query tokens and latency.
Main Results
100 150 200 250 300
Query efficiency (questions / 1M tokens)0.50.60.70.80.91.0Boundary Balanced Score (BBS)PAGE-RAG
MS GraphRAG
BDR-RAGLightRAG
1 2 3 4 5 6
Query speed (questions / minute)0.50.60.70.80.91.0
PAGE-RAG
MS GraphRAG
BDR-RAGLightRAG
101102
Query efficiency (questions / 1M tokens, log)0.10.20.30.40.50.6PAGE-RAG-anchored four-state qualityPAGE-RAG MS GraphRAG
BDR-RAG
LightRAG
1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
Query speed (questions / minute)0.10.20.30.40.50.6
PAGE-RAGMS GraphRAG
BDR-RAG
LightRAG
Figure 2: Empirical quality–efficiency frontier across book-length QA and UltraDomain-
Mix. Dashed segments show the baseline-only frontier and solid segments show the frontier
after adding PAGE-RAG. PAGE-RAG adds a balanced operating point rather than uni-
versally dominating every baseline.
Figure 2 visualizes the empirical frontier induced by the main results. For
the two book datasets, quality combines lenient accuracy and correct refusal,
so the plot reflects both answer correctness and boundary reliability. For
UltraDomain-Mix, quality is the PAGE-RAG-anchored four-state pairwise
index. The figure shows that PAGE-RAG is not merely a point with lower
cost;itoccupiesadifferentregionofthequality–efficiencyspace.Onthebook
datasets, it combines the strongest boundary-balanced quality with substan-
tially lower latency and query cost than LightRAG. On UltraDomain-Mix,
Microsoft GraphRAG remains the higher-quality extreme, but PAGE-RAG
reaches a nearby balanced region with far lower query tokens and latency.
Book-length QA.Table 1 shows the main results on the two book-
length datasets. PAGE-RAG is not universally best on strict accuracy, How-
ever, strict or lenient accuracy alone does not capture whether a system
answers beyond the available evidence. LightRAG leads strict accuracy on
13

both books, and BDR-RAG is very strong on the conceptual book. However,
PAGE-RAG is the only system that combines competitive lenient accuracy
with stable correct refusal on both datasets. OnSimulacra and Simulation,
PAGE-RAGreaches92.6%lenientaccuracyand12/12correctrefusal;onOne
Hundred Years of Solitude, it reachesthehighest lenient accuracy, 87.5%, and
again refuses all 12 unanswerable questions correctly. This pattern supports
the role of evidence-bounded generation: the system does not merely answer
more often, but preserves a closed-corpus boundary.
Table 1: Main results on the two book-length datasets. Acc. denotes accuracy, Ref. denotes
correct refusal, Lat. denotes median latency, QTok denotes query tokens per question, and
BTok denotes build LLM tokens.
Dataset System Strict Lenient Ref. Lat. QTok BTok
Simulacra PAGE-RAG 72.3 92.6 12/12 14.9s 4,213 842K
Simulacra BDR-RAG 77.7 93.6 5/12 15.9s 3,808 0
Simulacra LightRAG 80.9 85.1 12/12 47.3s 9,233 686K
Simulacra MS GraphRAG 52.1 67.0 11/12 11.8s 3,281 1.30M
Solitude PAGE-RAG 60.2 87.5 12/12 16.3s 7,052 1.85M
Solitude BDR-RAG 54.5 81.8 3/12 15.4s 6,215 0
Solitude LightRAG 64.8 81.8 9/12 48.4s 15,305 1.78M
Solitude MS GraphRAG 48.9 70.5 9/12 8.5s 3,142 4.41M
Table 2: UltraDomain-Mix four-dimensional pairwise results and efficiency. Each quality
cell reports PAGE-RAG wins / ties / baseline wins under position-debiased judgments.
Comparison Comp. Div. Emp. Direct. QTok / q Lat.
PAGE-RAG vs BDR-RAG 49/41/35 51/37/37 40/48/37 23/75/27 1,956 vs 2,351 13.4s vs 15.75s
PAGE-RAG vs MS GraphRAG 39/21/65 36/26/63 40/29/56 27/75/23 1,956 vs 136,509 13.4s vs 38.7s
PAGE-RAG vs LightRAG 101/22/2 102/20/3 103/18/4 76/45/4 1,956 vs 5,949 13.4s vs 33.1s
The two books also show why PAGE-RAG does not treat graph retrieval
as a universal replacement for passage retrieval. On the conceptual book, the
strong passage baseline is already highly competitive on answerable ques-
tions, suggesting that local textual evidence often suffices. On the narrative
book, PAGE-RAG obtains the best lenient accuracy, consistent with graph-
guidedevidenceorganizationbeingmoreusefulforcross-chapterevents,char-
acters, and relations. This supports the query-adaptive design: graph struc-
14

ture is most useful when the question requires structural organization, while
the textual floor remains critical for grounding.
UltraDomain-Mix.Table 2 reports four-dimensional pairwise results
on UltraDomain-Mix. PAGE-RAG is close to BDR-RAG across comprehen-
siveness and diversity, weaker on directness, and strongly ahead of LightRAG
across all four dimensions. Microsoft GraphRAG wins more often on the four
quality dimensions, but its cost profile is substantially different: Microsoft
GraphRAG uses 136,509 query tokens per question, while PAGE-RAG uses
1,956. This result positions Microsoft GraphRAG as a high-quality, high-cost
extreme for this setting, while PAGE-RAG adds a more balanced operating
point with lower latency and much lower query cost.
Takentogether,themainresultssupportPAGE-RAG’sintendedposition-
ing. It is not a universal quality-dominant system, and it does not replace
every GraphRAG operating point. Instead, it combines competitive answer
quality, reliable abstention on closed-corpus book questions, and low query-
time cost, thereby occupying a balanced empirical position on the quality–
efficiency–boundary Pareto frontier.
Ablation Study
Table 3 reports the core ablations on the two book-length datasets. The
results isolate two dominant mechanisms. First, the textual retrieval floor is
the basis of evidence supply: removing it reduces lenient accuracy from 90.1%
to 58.2%, even though the system still refuses all unanswerable questions
correctly. This supports the projection-aware claim that graph structures
can organize and navigate evidence, but should not replace source passages.
Second, the evidence constraint is the direct mechanism for the knowledge
boundary. Removing the evidence constraint creates an apparently better ac-
curacy point but a substantially worse system operating point. Lenient accu-
racy rises from 90.1% to 95.1%, while correct refusal collapses from 24/24 to
4/24,and22of24unanswerablequestionsreceivesubstantiveout-of-evidence
answers. This result shows that answer accuracy and boundary reliability are
not interchangeable: a system may move upward on an accuracy-only evalua-
tion while moving sharply backward on the true quality–efficiency–reliability
frontier.
Graph governance and structural guidance mainly contribute organiza-
tion, retrievability, and targeted gains. Raw Graph obtains lower lenient
accuracy than the full system, indicating that unguided automatic graph
extraction can introduce redundant entities, noisy relations, and fragmented
15

Table 3: Core ablations on the two book-length datasets. BBS denotes boundary-balanced
score; Ref. reports correct refusals on unanswerable questions.
Variant Lenient Acc. Ref. BBS
Full PAGE-RAG 90.1 24/24 94.8
No Floor 58.2 24/24 73.6
Raw Graph 86.3 24/24 92.6
No Evidence Constraint 95.1 4/24 28.4
No Structural Guidance 87.9 24/24 93.6
communities that weaken the graph as a retrieval skeleton. The structural ev-
idence is more direct on Solitude: Raw Graph contains 4,300 entities, 4,827
semantic edges, and 1,024 communities, while the full system compresses
them to 1,919 entities, 3,791 semantic edges, and 194 communities. Isolated
entities decrease from 763 to 175, and actual path use increases from 3 to
39 queries. On Simulacra, the full system also reduces entities from 1,904 to
1,555 and communities from 484 to 351, while increasing path use from 18 to
32 queries. These results show that graph governance is not merely an aux-
iliary cleanup step; it reduces entity duplication, community fragmentation,
andnoisystructuralcontext,makingthegraphmoresuitableasacompressed
semantic skeleton. A more compact graph lowers the search space and path-
enumeration difficulty, reduces the chance that irrelevant structure enters the
evidence context, and can therefore improve retrieval quality while helping
control token cost and query efficiency.
No Structural Guidance further shows that structural evidence is most
useful when activated selectively. Removing explicit paths and structural
hints lowers lenient accuracy from 90.1% to 87.9%; on routed path-using an-
swerable questions, the drop is larger, from 94.1% to 89.7%. This suggests
that explicit graph paths help most when the router identifies structural
evidence needs, rather than serving as a universal context expansion. Over-
all, the ablations support a mechanism account: textual retrieval supplies
grounded evidence, graph governance makes the semantic skeleton compact
and searchable, query-adaptive structural guidance provides targeted gains,
and evidence-bounded generation enforces the answer boundary.
16

Conclusion
This paper introduced PAGE-RAG to address the trade-off among an-
swer quality, efficiency, and knowledge-boundary reliability in long-document
GraphRAG. Its central principle is projection awareness: PAGE-RAG com-
bines an always-on textual retrieval floor with graph structures that serve as
compressedsemanticskeletonsfornavigatingandorganizingsourceevidence.
Throughgovernedgraphabstractions,query-adaptiverouting,boundedstruc-
tural retrieval, and evidence-bounded generation, PAGE-RAG adds a bal-
anced empirical operating point to the quality–efficiency frontier while en-
forcing a strict answer-or-refuse boundary when available evidence is insuffi-
cient.
Experiments on two books and UltraDomain-Mix show that PAGE-RAG
adds a more balanced empirical operating point rather than dominating ev-
ery system on every single quality metric. It maintains competitive answer
quality, reduces query-time token and latency cost, and preserves reliable
abstention on closed-corpus questions. Ablations further show that the tex-
tual floor supplies grounded evidence, graph governance improves structural
retrievability, and evidence constraints are central to boundary reliability.
Future work will explore finer-grained access control, cross-domain graph
construction, and more precise task-adaptive graph computation.
References
[1] M. Yue, A survey of large language model agents for question answering,
arXiv preprint arXiv:2503.19213 (2025).
[2] X. Chen, J. An, J. Guo, L. Wang, J. Guo, Kg -augmented executable
cot for mathematical coding, Neural Networks 202 (2026) 109006.
doi:https://doi.org/10.1016/j.neunet.2026.109006.
URL https://www.sciencedirect.com/science/article/pii/
S0893608026004673
[3] T. Masterman, S. Besen, M. Sawtell, A. Chao, The landscape of emerg-
ing ai agent architectures for reasoning, planning, and tool calling: A
survey, arXiv preprint arXiv:2404.11584 (2024).
[4] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni,
P. Liang, Lost in the middle: How language models use long contexts,
17

Transactions of the association for computational linguistics 12 (2024)
157–173.
[5] O. Press, N. A. Smith, M. Lewis, Train short, test long: Attention
with linear biases enables input length extrapolation, arXiv preprint
arXiv:2108.12409 (2021).
[6] S. Chen, S. Wong, L. Chen, Y. Tian, Extending context window
of large language models via positional interpolation, arXiv preprint
arXiv:2306.15595 (2023).
[7] R. Xu, Z. Qi, Z. Guo, C. Wang, H. Wang, Y. Zhang, W. Xu, Knowledge
conflicts for llms: A survey, in: Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing, 2024, pp. 8541–
8565.
[8] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,
H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al., Retrieval-
augmented generation for knowledge-intensive nlp tasks, Advances in
neural information processing systems 33 (2020) 9459–9474.
[9] R. Nogueira, K. Cho, Passage re-ranking with bert, arXiv preprint
arXiv:1901.04085 (2019).
[10] W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, Z. Ren,
Is chatgpt good at search? investigating large language models as re-
ranking agents, in: Proceedings of the 2023 conference on empirical
methods in natural language processing, 2023, pp. 14918–14937.
[11] Y.Gao,Y.Xiong,X.Gao,K.Jia,J.Pan,Y.Bi,Y.Dai,J.Sun,M.Wang,
H. Wang, Retrieval-augmented generation for large language models: A
survey, arXiv preprint arXiv:2312.10997 (2023).
[12] S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, M. Abdelrazek,
Seven failure points when engineering a retrieval augmented generation
system, in: Proceedings of the IEEE/ACM 3rd International Conference
on AI Engineering-Software Engineering for AI, 2024, pp. 194–199.
[13] X.He,Y.Tian,Y.Sun,N.V.Chawla,T.Laurent,Y.LeCun,X.Bresson,
B. Hooi, G-retriever: Retrieval-augmented generation for textual graph
18

understanding and question answering, Advances in Neural Information
Processing Systems 37 (2024) 132876–132907.
[14] Y. Yao, P. Wang, B. Tian, S. Cheng, Z. Li, S. Deng, H. Chen, N. Zhang,
Editing large language models: Problems, methods, and opportunities,
in: Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, 2023, pp. 10222–10240.
[15] H. Han, L. Ma, Y. Wang, H. Shomer, Y. Lei, Z. Qi, K. Guo, Z. Hua,
B. Long, H. Liu, et al., Rag vs. graphrag: A systematic evaluation and
key insights, arXiv preprint arXiv:2502.11371 (2025).
[16] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
J.Larson,Fromlocaltoglobal:AgraphRAGapproachtoquery-focused
summarization (2024). arXiv:2404.16130.
[17] V.A.Traag,L.Waltman,N.J.VanEck,Fromlouvaintoleiden:guaran-
teeing well-connected communities, Scientific reports 9 (1) (2019) 5233.
[18] J.Song,Z.Xu,M.He,J.Feng,B.Shen,Graphretrievalaugmentedlarge
language models for facial phenotype associated rare genetic disease,
NPJ digital medicine 8 (1) (2025) 543.
[19] Z. Zhu, X. Yuan, M. Galkin, L.-P. Xhonneux, M. Zhang, M. Gazeau,
J.Tang,A*net:Ascalablepath-basedreasoningapproachforknowledge
graphs, Advances in neural information processing systems 36 (2023)
59323–59336.
[20] W. Tao, X. Li, Y. Lan, W. Qian, Tagrag: Tag-guided hierarchical knowl-
edge graph retrieval-augmented generation, in: Findings of the Associa-
tion for Computational Linguistics: ACL 2026, 2026, pp. 6434–6456.
[21] B. Chen, Z. Guo, Z. Yang, Y. Chen, J. Chen, Z. Liu, C. Shi, C. Yang,
Pathrag: Pruning graph-based retrieval augmented generation with re-
lational paths, in: Proceedings of the AAAI conference on artificial in-
telligence, Vol. 40, 2026, pp. 30183–30191.
[22] B. Peng, Y. Zhu, Y. Liu, X. Bo, H. Shi, C. Hong, Y. Zhang, S. Tang,
Graph retrieval-augmented generation: A survey, ACM Transactions on
Information Systems 44 (2) (2025) 1–52.
19

[23] Q. Zhang, S. Chen, Y. Bei, Z. Yuan, H. Zhou, Z. Hong, H. Chen,
Y. Xiao, C. Zhou, J. Dong, et al., A survey of graph retrieval-
augmented generation for customized large language models, arXiv
preprint arXiv:2501.13958 (2025).
[24] K. S. Shin, The reasoning trap: An information-theoretic bound on
closed-system multi-step llm reasoning, arXiv preprint arXiv:2605.01704
(2026).
[25] F. Petroni, T. Rocktäschel, S. Riedel, P. Lewis, A. Bakhtin, Y. Wu,
A. Miller, Language models as knowledge bases?, in: Proceedings of the
2019 conference on empirical methods in natural language processing
andthe9thinternationaljointconferenceonnaturallanguageprocessing
(EMNLP-IJCNLP), 2019, pp. 2463–2473.
[26] K. Meng, D. Bau, A. Andonian, Y. Belinkov, Locating and editing fac-
tual associations in gpt, Advances in neural information processing sys-
tems 35 (2022) 17359–17372.
[27] Y. Bai, Z. Liu, H. Yan, Z. Xu, Y. Wan, C. Chen, Z. Wang, X. Yuan,
Y. Huang, G. Dou, et al., Inference-time control for trustworthy large
language models (2026).
[28] J. Mao, C. Gan, P. Kohli, J. B. Tenenbaum, J. Wu, The neuro-symbolic
concept learner: Interpreting scenes, words, and sentences from natural
supervision, arXiv preprint arXiv:1904.12584 (2019).
[29] P. Hitzler, A. Eberhart, M. Ebrahimi, M. K. Sarker, L. Zhou, Neuro-
symbolic approaches in artificial intelligence, National Science Review
9 (6) (2022) nwac035.
[30] W. Chen, X. Ma, X. Wang, W. W. Cohen, Program of thoughts prompt-
ing: Disentangling computation from reasoning for numerical reasoning
tasks, arXiv preprint arXiv:2211.12588 (2022).
[31] G. Aglionby, S. Teufel, Faithful knowledge graph explanations in com-
monsense question answering, in: Proceedings of the 2022 Conference on
Empirical Methods in Natural Language Processing, 2022, pp. 10811–
10817.
20

[32] L.Gao,A.Madaan,S.Zhou,U.Alon,P.Liu,Y.Yang,J.Callan,G.Neu-
big, Pal: Program-aided language models, in: International conference
on machine learning, PMLR, 2023, pp. 10764–10799.
[33] L. Pan, A. Albalak, X. Wang, W. Wang, Logic-lm: Empowering large
language models with symbolic solvers for faithful logical reasoning,
in: Findings of the Association for Computational Linguistics: EMNLP
2023, 2023, pp. 3806–3824.
[34] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,
H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, D. Kiela,
Retrieval-augmented generation for knowledge-intensive nlp tasks, in:
H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, H. Lin (Eds.),
Advances in Neural Information Processing Systems, Vol. 33, Curran
Associates, Inc., 2020, pp. 9459–9474.
URL https://proceedings.neurips.cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.pdf
[35] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen,
W.-t. Yih, Dense passage retrieval for open-domain question answering,
in: Proceedings of the 2020 conference on empirical methods in natural
language processing (EMNLP), 2020, pp. 6769–6781.
[36] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych, BEIR:
A heterogeneous benchmark for zero-shot evaluation of information re-
trieval models, in: Thirty-fifth Conference on Neural Information Pro-
cessing Systems Datasets and Benchmarks Track, 2021.
[37] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, C. Manning,
Raptor: Recursive abstractive processing for tree-organized retrieval, in:
International Conference on Learning Representations, Vol. 2024, 2024,
pp. 32628–32649.
[38] Z. Guo, L. Xia, Y. Yu, T. Ao, C. Huang, LightRAG: Simple
and fast retrieval-augmented generation, in: C. Christodoulopoulos,
T. Chakraborty, C. Rose, V. Peng (Eds.), Findings of the Associ-
ation for Computational Linguistics: EMNLP 2025, Association for
Computational Linguistics, Suzhou, China, 2025, pp. 10746–10761.
doi:10.18653/v1/2025.findings-emnlp.568.
URL https://aclanthology.org/2025.findings-emnlp.568/
21

[39] B. J. Gutiérrez, Y. Shu, Y. Gu, M. Yasunaga, Y. Su, Hipporag: Neuro-
biologically inspired long-term memory for large language models, Ad-
vances in neural information processing systems 37 (2024) 59532–59569.
[40] L. Liang, Z. Bo, Z. Gui, Z. Zhu, L. Zhong, P. Zhao, M. Sun, Z. Zhang,
J. Zhou, W. Chen, et al., Kag: Boosting llms in professional domains
via knowledge augmented generation, in: Companion Proceedings of the
ACM on Web Conference 2025, 2025, pp. 334–343.
[41] J. Baudrillard, Simulacra and simulation, University of Michigan press,
1994.
[42] G. G. Márquez, One hundred years of solitude, in: Medicine and litera-
ture, volume two, CRC Press, 2018, pp. 255–272.
22