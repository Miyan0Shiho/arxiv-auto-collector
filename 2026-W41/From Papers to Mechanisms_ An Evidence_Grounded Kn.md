# From Papers to Mechanisms: An Evidence-Grounded Knowledge Substrate for Scientific Language Models

**Authors**: Qiuhui Chen, Yibo Liu, Tao Dai, Jiafan Lu, Zhenglei Zhou, Weimin Zhong

**Published**: 2026-10-05 12:47:15

**PDF URL**: [https://arxiv.org/pdf/2610.06248v1](https://arxiv.org/pdf/2610.06248v1)

## Abstract
Scientific language models often access literature through untyped text chunks, which fragment the functional and evidential structure required for mechanism-rich questions. We introduce an evidence-grounded mechanism knowledge substrate that organizes scientific literature into provenance-linked evidence units, role-typed entities, and directed mechanism paths. We instantiate it as MS$^3$, a Material-Sensor-Signal-System schema for conductive-fiber flexible sensors, over 13,689 papers, 131,083 evidence items, and 26,648 mechanism objects. On in-domain and coverage-shift question-answering benchmarks, we compare closed-book generation, Web search, Raw-PDF RAG, and MS$^3$ retrieval across ten language models. MS$^3$ improves macro-averaged scientific correctness. It also improves citation entailment and answer completeness. These results support mechanism substrates as a reliable representation layer for scientific language models and motivate a source-repair workflow in which insufficient MS$^3$ evidence triggers targeted retrieval from its linked papers rather than assuming that a user has already supplied the correct PDFs.

## Full Text


<!-- PDF content starts -->

From Papers to Mechanisms: An Evidence-Grounded Knowledge Substrate for
Scientific Language Models
Qiuhui Chen1, Yibo Liu2, Tao Dai1,
Jiafan Lu1, Zhenglei Zhou3, Weimin Zhong1,*
1School of Information Science and Engineering, East China University of Science and Technology
2School of Computer Science, Shanghai Jiao Tong University3Tencent
Abstract
Scientificlanguagemodelsoftenaccessliteraturethroughun-
typedtextchunks,whichfragmentthefunctionalandeviden-
tialstructurerequiredformechanism-richquestions.Weintro-
duce an evidence-grounded mechanism knowledge substrate
that organizes scientific literature into provenance-linked ev-
idence units, role-typed entities, and directed mechanism
paths. We instantiate it as MS3, a Material–Sensor–Signal–
System schema for conductive-fiber flexible sensors, over
13,689 papers, 131,083 evidence items, and 26,648 mech-
anism objects. On in-domain and coverage-shift question-
answering benchmarks, we compare closed-book generation,
Web search, Raw-PDF RAG, and MS3retrieval across ten
language models. MS3improves macro-averaged scientific
correctness. It also improves citation entailment and answer
completeness.Theseresultssupportmechanismsubstratesas
a reliable representation layer for scientific language models
and motivate a source-repair workflow in which insufficient
MS3evidence triggers targeted retrieval from its linked pa-
persratherthanassumingthatauserhasalreadysuppliedthe
correct PDFs.
Introduction
Largelanguagemodelsareincreasinglyusedtosearch,sum-
marize,andreasonoverscientificliterature(Asaietal.2026;
Skarlinski et al. 2024). Because their internal knowledge is
incomplete and difficult to verify, retrieval-augmented gen-
eration has become a common interface between language
models and scientific corpora (Lewis et al. 2020; Lála et al.
2024). Most systems, however, represent papers as indepen-
dently indexed text chunks. This representation is effective
fordocumentlookupandlocalfactualquestions,butispoorly
matched to mechanism-oriented reasoning. Explaining how
a material enables a system-level function may require con-
necting its composition or fabrication, the resulting device,
its measurable response, and the downstream application,
withsupportingevidencedistributedacrosstext,figures,cap-
tions, and tables.
Structuredknowledgerepresentationsandagenticretrieval
improvehowscientificinformationisorganizedandaccessed
(Aueretal.2018;Jaradehetal.2019;Edgeetal.2024;Lála
et al. 2024; Skarlinski et al. 2024; Ding et al. 2026). Never-
theless,existingsystemsofteneitherabstractawaythesource
∗Corresponding author.
/gid00019/gid00032/gid00033/gid00032/gid00045/gid00032/gid00041/gid00030/gid00032
/gid00023/gid00028/gid00039/gid00036/gid00031/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039 /gid00020/gid00032/gid00041/gid00046/gid00042/gid00045 /gid00020/gid00052/gid00046/gid00047/gid00032/gid00040 /gid00020/gid00036/gid00034/gid00041/gid00028/gid00039/gid00007/gid00045/gid00028/gid00034/gid00040/gid00032/gid00041/gid00047/gid00032/gid00031/gid00001/gid00040/gid00048/gid00039/gid00047/gid00036/gid00040/gid00042/gid00031/gid00028/gid00039/gid00001/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032
/gid00020/gid00030/gid00035/gid00032/gid00040/gid00028/gid01162/gid00034/gid00048/gid00036/gid00031/gid00032/gid00031/gid00001/gid00028/gid00039/gid00036/gid00034/gid00041/gid00040/gid00032/gid00041/gid00047
/gid00028/gid00041/gid00031/gid00001/gid00043/gid00045/gid00042/gid00049/gid00032/gid00041/gid00028/gid00041/gid00030/gid00032/gid00001/gid00039/gid00036/gid00041/gid00038/gid00036/gid00041/gid00034
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid01162/gid00034/gid00045/gid00042/gid00048/gid00041/gid00031/gid00032/gid00031/gid00001/gid00040/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00001/gid00042/gid00029/gid00037/gid00032/gid00030/gid00047
/gid00017/gid00045/gid00042/gid00043/gid00032/gid00045/gid00047/gid00052/gid00006/gid00051/gid00043/gid00032/gid00045/gid00036/gid00040/gid00032/gid00041/gid00047
/gid00010/gid00041/gid00035/gid00028/gid00039/gid00028/gid00047/gid00036/gid00042/gid00041 /gid00006/gid00051/gid00035/gid00028/gid00039/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00020/gid00036/gid00047 /gid00020/gid00047/gid00028/gid00041/gid00031 /gid00024/gid00028/gid00039/gid00038 /gid00019/gid00048/gid00041 /gid00004/gid00039/gid00036/gid00040/gid00029/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00003/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00002
/gid00020/gid00040/gid00028/gid00045/gid00047/gid00001/gid00021/gid01162/gid00046/gid00035/gid00036/gid00045/gid00047
/gid01175/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00002/gid01748/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00003/gid01176
/gid00003/gid00036/gid00042/gid00009/gid00028/gid00045/gid00041/gid00032/gid00046/gid00046
/gid00019/gid00032/gid00033/gid00032/gid00045/gid00032/gid00041/gid00030/gid00032
/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00002 /gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00003/gid00017/gid00036/gid00032/gid00053/gid00042/gid00045/gid00032/gid00046/gid00036/gid00046/gid00047/gid00036/gid00049/gid00032/gid00001/gid00045/gid00032/gid00046/gid00043/gid00042/gid00041/gid00046/gid00032
/gid00612/gid00001/gid01175/gid01727/gid01176 /gid00612/gid00001/gid01175/gid01727/gid01176/gid00578/gid00019/gid01181/gid00019/gid01588/gid00001/gid01175/gid01727/gid01176
/gid01087/gid01141/gid01092/gid01090/gid01087/gid01141/gid01094/gid01095
/gid01087/gid01141/gid01090/gid01089
/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00002 /gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00003 /gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00002/gid01748/gid00003/gid00014/gid00002/gid00006/gid00001/gid01778
/gid00021/gid00032/gid00051/gid00047/gid00001/gid01170/gid00001/gid00033/gid00036/gid00034/gid00048/gid00045/gid00032/gid00046/gid00001/gid01170/gid00001/gid00047/gid00028/gid00029/gid00039/gid00032/gid00046
Figure1:Evidence-groundedconstructionofanMS3mech-
anism object.
evidencesupportingindividualrelationsorcontinuetooper-
ateoveruntypeddocumentchunks.Asaresult,thelanguage
modelmustreconstructfunctionalstructureatinferencetime
whilealsodeterminingwhichclaimsaresupportedbythere-
trieved evidence.
We introduce an evidence-grounded mechanism knowl-
edge substrate as an intermediate representation between
scientificdocumentsandlanguagemodels.Itorganizesliter-
ature into provenance-linked evidence units, role-typed en-
tities, directed relations, and complete or partial mechanism
paths. The framework is schema-configurable: each domain
defines its entity types, admissible relations, and structural
arXiv:2610.06248v1  [cs.AI]  5 Oct 2026

constraints,whileevidenceextraction,provenancepreserva-
tion, mechanism construction, retrieval, and evidence pack-
ing remain shared. Unsupported nodes and relations are left
absent rather than completed from model priors, and every
retainedelementislinkedtoitssupportingsourceevidence.
We instantiate the framework for conductive-fiber flexi-
blesensorsusingtheMaterial–Sensor–Signal–System(MS3)
schema,
Material→Sensor→Signal→System.
Material represents the constituent material or fabrication-
related implementation, Sensor the functional sensing el-
ement, Signal the measurable response, and System the
demonstrated or proposed downstream use. To distinguish
experimentalevidencefromenvisionedapplications,System
records the demonstrated system, proposed application, val-
idation setting, and deployment status separately. An MS3
object may contain a complete four-layer path or only the
subpath supported by the paper, with provenance linked to
sourcepassagesand,whenavailable,pages,sections,figures,
captions, tables, and quantitative evidence.
Using this framework, we construct an MS3substrate
from 13,689 papers, containing 131,083 source-grounded
evidence items and 26,648 complete or partial mechanism
objects.Atinferencetime,thesystemretrievescandidatepa-
persandselectsevidence-linkedmechanismsrelevanttothe
query. We evaluate MS3across 10 language models. MS3
substantially improves scientific correctness for nine of the
ten models in blinded cross-family evaluation.
Our main contributions are:
•Evidence-grounded mechanism representation.We
formulate a schema-configurable substrate that jointly
represents source evidence, functional roles, directed re-
lations, partial or complete mechanism paths, and prove-
nance.
•Large-scale MS3instantiation.We construct an
evidence-linked substrate for conductive-fiber flexible
sensors covering 13,689 papers and 26,648 mechanism
objects.
•Mechanism-aware retrieval interface.We develop de-
terministic lexical retrieval and provenance-aware top-
kevidence selection that expose structured, source-
verifiablemechanismcontextstoheterogeneouslanguage
models.
•Cross-model evaluation.We evaluate the substrate
across 10 language models using paired uncertainty esti-
mates and blinded cross-family judging.
Related Work
Scientific literature retrieval and synthesis.Retrieval-
augmented generation (RAG) augments language models
with non-parametric corpora and has become a standard
approach for knowledge-intensive generation (Lewis et al.
2020).Scientificvariantsincreasinglycombineretrievalwith
multi-stage language-model workflows. PaperQA retrieves
and evaluates passages from full-text papers before gener-
ating cited answers (Lála et al. 2024), while PaperQA2 ex-
tends this paradigm to literature search, synthesis, and con-
tradiction detection (Skarlinski et al. 2024). OpenScholarcouples large-scale passage retrieval with citation-backed,
multi-paper synthesis (Asai et al. 2026), and SciRAG in-
troduces adaptive retrieval, citation-aware reasoning, and
outline-guided generation (Ding et al. 2026). These sys-
temssubstantiallyimproveaccesstoscientificliterature,but
theirretrievalunitstypicallyremaindocuments,passages,or
citation-linked neighborhoods. Consequently, the language
model must still infer the functional roles of different evi-
dence pieces and determine whether they form a coherent
mechanism. Our work is complementary to advances in re-
trievalandagentorchestration:wefocusontherepresentation
layer and construct a persistent, provenance-linked mecha-
nismsubstratethatcanbequeriedbyheterogeneouslanguage
models.
Scientific information extraction and knowledge graphs.
Scientific information extraction seeks to convert papers
into entities and relations. SciERC jointly models enti-
ties, relations, and coreference in scientific abstracts (Luan
et al. 2018), and SciREX extends scientific extraction to
document-level salient entities andn-ary relations (Jain
et al. 2020). More recent schema-driven approaches such
as SPIRES use language models to populate user-defined
knowledgestructures(Caufieldetal.2024).Inparallel,scien-
tific knowledge graph efforts, including the Science Knowl-
edge Graph and the Open Research Knowledge Graph, rep-
resent scholarly entities, contributions, and comparisons in
machine-actionable form (Auer et al. 2018; Jaradeh et al.
2019). Domain-specific resources further demonstrate the
value of structured representations; for example, MatKG or-
ganizes materials, properties, synthesis methods, character-
ization methods, and applications extracted from materials-
science literature (Venugopal and Olivetti 2024). Graph-
based RAG methods such as GraphRAG and G-Retriever
retrieve graph communities or query-relevant subgraphs to
support summarization and question answering (Edge et al.
2024; He et al. 2024). Existing lines of work, however, do
not generally provide all of source-level evidence, domain-
specific functional roles, directed mechanism relations, ex-
plicit partial-path uncertainty, and a common interface for
downstreamlanguagemodels.Oursubstratecombinesthese
elements by treating evidence-grounded mechanism paths,
rather than isolated triples or document chunks, as the pri-
mary knowledge unit.
Evidence-grounded scientific question answering and
verification.Evidenceattributioniscentraltoreliablesci-
entific reasoning. QASPER pairs full-paper questions with
answers and supporting evidence spans (Dasigi et al. 2021).
SciFact formulates scientific claim verification as retrieving
evidence that supports or refutes a claim and identifying
the corresponding rationales (Wadden et al. 2020); SciFact-
Open extends this setting to large-corpus retrieval and ex-
poseschallengessuchasevidencespecificityandconflicting
findings(Waddenetal.2022).SciVerfurtherevaluatesclaim
verification over multimodal scientific evidence, including
text, tables, and figures (Wang et al. 2025). These bench-
marks provide important task-level supervision and evalua-
tion,butevidenceistypicallyattachedtoindividualquestions
orclaimsratherthanorganizedintoareusablescientificrep-

resentation. In contrast, our framework incorporates prove-
nance during substrate construction, links evidence to both
mechanismnodesanddirectedrelations,andretainsmissing
mechanism layers as explicit evidence gaps. This represen-
tationsupportsnotonlyquestionansweringandverification,
butalsomechanismreconstructionandcross-papercompari-
son.Thus,ourworkliesattheintersectionofscientificRAG,
knowledge-graphconstruction,andevidence-groundedveri-
fication,whilemakingtheorganizationofscientificevidence
itself the primary object of study.
Methodology
Problem Formulation
LetD={d i}N
i=1be a scientific corpus and letΣ δ=
(Tδ,Rδ,Cδ)denote the mechanism schema for domainδ.
Here,T δcontains functional entity types,R δcontains ad-
missible directed relations, andC δcontains structural con-
straints. The framework is schema-configurable: construc-
tion and retrieval are shared across domains, whereasΣ δ
suppliesdomain-specificsemantics.Forflexiblesensors,the
schema is instantiated asMaterial→Sensor→Signal→
System.
We transformDinto a substrate
Kδ= (D,E,V,R,P,A),(1)
whereEcontains source-grounded evidence units,Vand
Rare typed mechanism nodes and relations,Pcontains
complete or partial mechanism paths, andAmaps retained
nodes and relations to their supporting evidence. The sub-
strate is subsequently accessed through mechanism-aware
retrieval rather than model fine-tuning. Figure 1 illustrates
anevidence-groundedMS3object,andFigure3summarizes
the decomposed construction pipeline.
Constructing the Mechanism Knowledge Substrate
Evidence extraction and provenance grounding.Each
paper is parsed into ordered source blocks with page and
sectionmetadataandlinkstonearbyfigures,captions,tables,
andquantitativecontentwhenavailable.Consecutiveblocks
are grouped into context-preserving windows for extraction
while retaining their original locations. A task-specialized
extractor produces evidence unitse= (x e, ρe, πe), where
xeissource-derivedtext,ρ edenotesitsevidentialrole(e.g.,
experiment,property,validation,orreference),andπ estores
document-levelandlocalprovenance.Weacceptanevidence
unit only when its text, after deterministic normalization,
can be aligned to the originating source block. Unaligned
candidatesarerejected,andnearduplicatesaremergedwhile
preserving the union of their provenance links. This stage
produces compact, traceable evidence records rather than
free-form summaries.
Decomposed mechanism construction.We decompose
substrateconstructionintothreestructuredgenerationstages.
All three stages use independently trained QLoRA adapters
over Qwen2.5-72B-Instruct. The evidence extractor (E-
LoRA) maps a focused PDF chunk to a small set of local
evidenceitems.Theterminologymodel(T-LoRA)performsevidence-grounded MS3terminology normalization, recov-
ering Material, Sensor, Signal, and System terms from the
accepted evidence rather than linking them to an external
ontology.Themechanismmodel(M-LoRA)receivestheev-
idence, grounded MS3terms, and source-grounded figure
captions, and produces ordered mechanism objects with ev-
idence chains. Mechanism induction is constrained to the
ordered Material→Sensor→Signal→System schema. Com-
pletepathsandevidence-supportedpartialpathsareretained,
while unsupported layers are left absent.
A mechanism object is
p= (V p, Rp, Ap),(2)
whereV pcontains typed nodes,R pcontains directed rela-
tions allowed byΣ δ, andA plinks each retained node and
relationtoevidenceidentifiers.AcompleteMS3objectcon-
tains all four layers, while a partial object contains only the
supported subpath. Missing layers and relations are left ab-
sent rather than completed using model priors, making evi-
dence gaps explicit.
Validationandassembly.Weapplydeterministicgatesfor
node-type validity, evidence-reference consistency, source
alignment, provenance recovery, and duplicate removal. At
final inference time, an evidence excerpt is retained only
when its normalized text can be exactly aligned to a parsed
sourcechunk;otherwiseitisremoved.Invalidevidencerefer-
encesareremovedfrommechanismchains,andamechanism
is discarded when no valid supporting evidence remains.
Duplicate evidence and mechanism objects are removed
using normalized exact matching. Token-overlap rules are
used only to reconcile complementary partial mechanism
paths. The accepted records are assembled into linked evi-
dence and mechanism views that share paper-level evidence
identifiers.
Mechanism-Aware Access for Language Models
Query-conditionedretrieval.Theevaluatedsystemusesa
deterministiclexicalqueryparserratherthanalearnedquery
interpreter. It normalizes and tokenizes the query, removes
fixed stopwords and short tokens, and uses the remaining
terms to search separate SQLite FTS5 indices for mecha-
nismsandevidence.Quotedpapertitlesareresolvedbyexact
title matching when available.
Candidate papers are ranked by summing their
mechanism-indexandevidence-indexBM25retrievalscores
with equal weights:
S(p|q) =X
m∈R m(p,q)−BM25 m(m, q)
+X
e∈R e(p,q)−BM25 e(e, q).(3)
Within each candidate paper, mechanism objects are ranked
byquery-tokenoverlapwiththeirstatementsandMS3fields.
Evidence referenced by the selected mechanisms is priori-
tized, followed by other evidence ranked by query overlap.
This procedure expands mechanism-linked evidence identi-
fiers; it does not perform multi-hop traversal over database
graph edges.

/gid01088/gid01093/gid01142/gid01087/gid01091/gid01091/gid00001/gid01175/gid01093/gid01087/gid01141/gid01089/gid01727/gid01176 /gid01088/gid01087/gid01142/gid01090/gid01093/gid01091/gid00001/gid01175/gid01094/gid01092/gid01141/gid01094/gid01727/gid01176
/gid01088/gid01142/gid01096/gid01096/gid01094/gid00001/gid01175/gid01094/gid01141/gid01092/gid01727/gid01176 /gid01088/gid01142/gid01096/gid01092/gid01095/gid00001/gid01175/gid01088/gid01091/gid01141/gid01090/gid01727/gid01176
/gid01088/gid01142/gid01095/gid01091/gid01088/gid00001/gid01175/gid01093/gid01141/gid01096/gid01727/gid01176 /gid01088/gid01142/gid01094/gid01092/gid01089/gid00001/gid01175/gid01088/gid01089/gid01141/gid01095/gid01727/gid01176
/gid01088/gid01142/gid01093/gid01090/gid01088/gid00001/gid01175/gid01093/gid01141/gid01088/gid01727/gid01176 /gid01088/gid01142/gid01089/gid01096/gid01091/gid00001/gid01175/gid01096/gid01141/gid01092/gid01727/gid01176
/gid01088/gid01142/gid01092/gid01092/gid01087/gid00001/gid01175/gid01092/gid01141/gid01095/gid01727/gid01176 /gid01088/gid01142/gid01092/gid01087/gid01091/gid00001/gid01175/gid01088/gid01088/gid01141/gid01087/gid01727/gid01176
/gid01088/gid01142/gid01088/gid01093/gid01095/gid00001/gid01175/gid01091/gid01141/gid01091/gid01727/gid01176 /gid01095/gid01088/gid01092/gid00001/gid01175/gid01093/gid01141/gid01087/gid01727/gid01176
/gid01091/gid01092/gid01090/gid00001/gid01175/gid01088/gid01141/gid01094/gid01727/gid01176 /gid01089/gid01094/gid01092/gid00001/gid01175/gid01089/gid01141/gid01087/gid01727/gid01176
/gid01090/gid01096/gid01091/gid00001/gid01175/gid01088/gid01141/gid01092/gid01727/gid01176 /gid01089/gid01095/gid01088/gid00001/gid01175/gid01089/gid01141/gid01088/gid01727/gid01176a Corpus scale and temporal coverage
/gid01088/gid01090/gid01142/gid01093/gid01095/gid01096
/gid00017/gid00028/gid00043/gid00032/gid00045/gid00046
/gid01088/gid01090/gid01088/gid01142/gid01087/gid01095/gid01090
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00010/gid00047/gid00032/gid00040/gid00046
/gid01089/gid01093/gid01142/gid01093/gid01091/gid01095
/gid00014/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00001/gid00042/gid00029/gid00037/gid00032/gid00030/gid00047/gid00046
/gid01088/gid01093/gid01142/gid01087/gid01091/gid01091
/gid00004/gid00042/gid00040/gid00043/gid00039/gid00032/gid00047/gid00032/gid00001/gid00014/gid00020/gid01090/gid00001/gid00030/gid00035/gid00028/gid00036/gid00041/gid00046
/gid01089/gid01087/gid01088/gid01088 /gid01089/gid01087/gid01088/gid01090 /gid01089/gid01087/gid01088/gid01092 /gid01089/gid01087/gid01088/gid01094 /gid01089/gid01087/gid01088/gid01096 /gid01089/gid01087/gid01089/gid01088 /gid01089/gid01087/gid01089/gid01090 /gid01089/gid01087/gid01089/gid01092 /gid01089/gid01087/gid01089/gid01093/gid01089/gid01142/gid01088/gid01091/gid01089
/gid01754/gid01089/gid01087/gid01088/gid01087/gid00029/gid00001/gid00014/gid00020/gid01090/gid00001/gid00031/gid00042/gid00040/gid00028/gid00036/gid00041/gid00001/gid00030/gid00042/gid00049/gid00032/gid00045/gid00028/gid00034/gid00032
/gid00021/gid00032/gid00051/gid00047/gid00036/gid00039/gid00032/gid00001/gid00046/gid00048/gid00029/gid00046/gid00047/gid00045/gid00028/gid00047/gid00032/gid00046
/gid00016/gid00047/gid00035/gid00032/gid00045/gid00001/gid00043/gid00042/gid00039/gid00052/gid00040/gid00032/gid00045/gid00001/gid00040/gid00028/gid00047/gid00045/gid00036/gid00030/gid00032/gid00046
/gid00020/gid00036/gid00039/gid00036/gid00030/gid00042/gid00041/gid00032/gid00001/gid00032/gid00039/gid00028/gid00046/gid00047/gid00042/gid00040/gid00032/gid00045/gid00046
/gid00020/gid00036/gid00039/gid00049/gid00032/gid00045/gid01162/gid00029/gid00028/gid00046/gid00032/gid00031/gid00001/gid00030/gid00042/gid00041/gid00031/gid00048/gid00030/gid00047/gid00042/gid00045/gid00046
/gid00008/gid00045/gid00028/gid00043/gid00035/gid00032/gid00041/gid00032/gid01162/gid00029/gid00028/gid00046/gid00032/gid00031/gid00001/gid00040/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039/gid00046
/gid00004/gid00028/gid00045/gid00029/gid00042/gid00041/gid00001/gid00041/gid00028/gid00041/gid00042/gid00047/gid00048/gid00029/gid00032/gid00046
/gid00003/gid00028/gid00046/gid00032/gid01162/gid00040/gid00032/gid00047/gid00028/gid00039/gid00001/gid00030/gid00042/gid00041/gid00031/gid00048/gid00030/gid00047/gid00042/gid00045/gid00046
/gid00009/gid00052/gid00031/gid00045/gid00042/gid00034/gid00032/gid00039/gid00001/gid00040/gid00028/gid00047/gid00045/gid00036/gid00030/gid00032/gid00046
/gid00008/gid00042/gid00039/gid00031/gid01162/gid00029/gid00028/gid00046/gid00032/gid00031/gid00001/gid00030/gid00042/gid00041/gid00031/gid00048/gid00030/gid00047/gid00042/gid00045/gid00046
/gid00014/gid00025/gid00032/gid00041/gid00032/gid00046
/gid00020/gid00052/gid00046/gid00047/gid00032/gid00040
/gid00009/gid00032/gid00028/gid00039/gid00047/gid00035/gid00001/gid00040/gid00042/gid00041/gid00036/gid00047/gid00042/gid00045/gid00036/gid00041/gid00034
/gid00014/gid00042/gid00047/gid00036/gid00042/gid00041/gid00001/gid00040/gid00042/gid00041/gid00036/gid00047/gid00042/gid00045/gid00036/gid00041/gid00034
/gid00006/gid00041/gid00049/gid00036/gid00045/gid00042/gid00041/gid00040/gid00032/gid00041/gid00047/gid00028/gid00039/gid00001/gid00040/gid00042/gid00041/gid00036/gid00047/gid00042/gid00045/gid00036/gid00041/gid00034
/gid00009/gid00048/gid00040/gid00028/gid00041/gid01162/gid00040/gid00028/gid00030/gid00035/gid00036/gid00041/gid00032/gid00001/gid00036/gid00041/gid00047/gid00032/gid00045/gid00033/gid00028/gid00030/gid00032/gid00046
/gid00003/gid00036/gid00042/gid00040/gid00028/gid00045/gid00038/gid00032/gid00045/gid00001/gid00040/gid00042/gid00041/gid00036/gid00047/gid00042/gid00045/gid00036/gid00041/gid00034
/gid00019/gid00032/gid00035/gid00028/gid00029/gid00036/gid00039/gid00036/gid00047/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00017/gid00042/gid00036/gid00041/gid00047/gid01162/gid00042/gid00033/gid01162/gid00030/gid00028/gid00045/gid00032/gid00001/gid00031/gid00036/gid00028/gid00034/gid00041/gid00042/gid00046/gid00047/gid00036/gid00030/gid00046
/gid00020/gid00042/gid00517/gid00001/gid00045/gid00042/gid00029/gid00042/gid00047/gid00036/gid00030/gid00046
/gid00006/gid00039/gid00032/gid00030/gid00047/gid00045/gid00042/gid00041/gid00036/gid00030/gid00001/gid00046/gid00038/gid00036/gid00041
/gid00010/gid00041/gid00031/gid00048/gid00046/gid00047/gid00045/gid00036/gid00028/gid00039/gid00001/gid00040/gid00042/gid00041/gid00036/gid00047/gid00042/gid00045/gid00036/gid00041/gid00034/gid00017/gid00028/gid00043/gid00032/gid00045/gid00046/gid00001/gid00029/gid00052/gid00001/gid00043/gid00048/gid00029/gid00039/gid00036/gid00030/gid00028/gid00047/gid00036/gid00042/gid00041/gid00001/gid00052/gid00032/gid00028/gid00045
/gid00030/gid00001/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid01162/gid00045/gid00042/gid00039/gid00032/gid00001/gid00030/gid00042/gid00040/gid00043/gid00042/gid00046/gid00036/gid00047/gid00036/gid00042/gid00041
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00036/gid00047/gid00032/gid00040/gid00046/gid00001/gid00029/gid00052/gid00001/gid00043/gid00045/gid00036/gid00040/gid00028/gid00045/gid00052/gid00001/gid00047/gid00052/gid00043/gid00032
/gid01092/gid01092/gid01141/gid01090/gid01727
/gid01089/gid01090/gid01141/gid01095/gid01727 /gid01088/gid01091/gid01141/gid01090/gid01727
/gid00006/gid00051/gid00043/gid00032/gid00045/gid00036/gid00040/gid00032/gid00041/gid00047
/gid00041/gid01753/gid01094/gid01089/gid01142/gid01091/gid01096/gid01089/gid00017/gid00045/gid00042/gid00043/gid00032/gid00045/gid00047/gid00052
/gid00041/gid01753/gid01090/gid01088/gid01142/gid01088/gid01095/gid01094/gid00023/gid00028/gid00039/gid00036/gid00031/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00041/gid01753/gid01088/gid01095/gid01142/gid01094/gid01087/gid01090/gid00019/gid00032/gid00033/gid00032/gid00045/gid00032/gid00041/gid00030/gid00032
/gid00041/gid01753/gid01095/gid01142/gid01094/gid01087/gid01088
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid01162/gid00039/gid00036/gid00041/gid00038/gid00032/gid00031/gid00001/gid00040/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00046/gid00001/gid00029/gid00052/gid00001/gid00046/gid00048/gid00043/gid00043/gid00042/gid00045/gid00047/gid00036/gid00041/gid00034/gid00001/gid00045/gid00042/gid00039/gid00032
/gid01087/gid01727 /gid01092/gid01087/gid01727 /gid01088/gid01087/gid01087/gid01727
/gid00006/gid00051/gid00043/gid00032/gid00045/gid00036/gid00040/gid00032/gid00041/gid00047 /gid00041/gid01753/gid01089/gid01091/gid01142/gid01094/gid01092/gid01094
/gid00017/gid00045/gid00042/gid00043/gid00032/gid00045/gid00047/gid00052 /gid00041/gid01753/gid01095/gid01142/gid01093/gid01088/gid01095
/gid00023/gid00028/gid00039/gid00036/gid00031/gid00028/gid00047/gid00036/gid00042/gid00041 /gid00041/gid01753/gid01094/gid01142/gid01092/gid01092/gid01095
/gid00019/gid00032/gid00033/gid00032/gid00045/gid00032/gid00041/gid00030/gid00032 /gid00041/gid01753/gid01091/gid01142/gid01093/gid01091/gid01093/gid01093/gid01141/gid01093/gid01727/gid00031/gid00001/gid00014/gid00020/gid01090/gid00001/gid00046/gid00047/gid00045/gid00048/gid00030/gid00047/gid00048/gid00045/gid00028/gid00039/gid00001/gid00030/gid00035/gid00028/gid00036/gid00041/gid00001/gid00030/gid00042/gid00049/gid00032/gid00045/gid00028/gid00034/gid00032
/gid00032/gid00001/gid00014/gid00048/gid00039/gid00047/gid00036/gid00040/gid00042/gid00031/gid00028/gid00039/gid00001/gid00043/gid00045/gid00042/gid00049/gid00032/gid00041/gid00028/gid00041/gid00030/gid00032/gid00016/gid00029/gid00037/gid00032/gid00030/gid00047/gid00046 /gid00017/gid00028/gid00043/gid00032/gid00045/gid00046
/gid00007/gid00042/gid00048/gid00045/gid01162/gid00033/gid00036/gid00032/gid00039/gid00031/gid00001/gid00030/gid00028/gid00041/gid00031/gid00036/gid00031/gid00028/gid00047/gid00032
/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid01162/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039
/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039/gid01162/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045
/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid01162/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039/gid01162/gid00020/gid00052/gid00046/gid00047/gid00032/gid00040
/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039/gid01162/gid00020/gid00052/gid00046/gid00047/gid00032/gid00040
/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039/gid01162/gid00020/gid00052/gid00046/gid00047/gid00032/gid00040
/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039/gid01162/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid01162/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039
/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039/gid01162/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039/gid01162/gid00020/gid00052/gid00046/gid00047/gid00032/gid00040
/gid01088/gid01090/gid01088/gid01142/gid01087/gid01095/gid01090
/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00036/gid00047/gid00032/gid00040/gid00046/gid00021/gid00032/gid00051/gid00047/gid00001/gid00042/gid00041/gid00039/gid00052 /gid01089/gid01093/gid01141/gid01088/gid01727/gid00001/gid00041/gid01753/gid01090/gid01091/gid01142/gid01089/gid01090/gid01091
/gid00021/gid00032/gid00051/gid00047/gid00001/gid01748/gid00001/gid00033/gid00036/gid00034/gid00048/gid00045/gid00032 /gid01088/gid01088/gid01141/gid01092/gid01727/gid00001/gid00041/gid01753/gid01088/gid01092/gid01142/gid01087/gid01088/gid01094/gid00021/gid00032/gid00051/gid00047/gid00001/gid01748/gid00001/gid00044/gid00048/gid00028/gid00041/gid00047/gid00036/gid00047/gid00028/gid00047/gid00036/gid00049/gid00032 /gid01088/gid01095/gid01141/gid01094/gid01727/gid00001/gid00041/gid01753/gid01089/gid01091/gid01142/gid01092/gid01088/gid01092
/gid00002/gid00039/gid00039/gid00001/gid00047/gid00035/gid00045/gid00032/gid00032/gid00001/gid00040/gid00042/gid00031/gid00028/gid00039/gid00036/gid00047/gid00036/gid00032/gid00046 /gid01091/gid01090/gid01141/gid01094/gid01727/gid00001/gid00041/gid01753/gid01092/gid01094/gid01142/gid01090/gid01088/gid01094/gid00015/gid00048/gid00040/gid00029/gid00032/gid00045/gid00001/gid00042/gid00033/gid00001/gid00043/gid00028/gid00043/gid00032/gid00045/gid00046
/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039
/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045
/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039
/gid00020/gid00052/gid00046/gid00047/gid00032/gid00040
/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039/gid00019/gid00032/gid00046/gid00036/gid00046/gid00047/gid00036/gid00049/gid00032/gid00001/gid00045/gid00032/gid00028/gid00031/gid00042/gid00048/gid00047
/gid00004/gid00048/gid00045/gid00045/gid00032/gid00041/gid00047/gid00001/gid00042/gid00048/gid00047/gid00043/gid00048/gid00047
/gid00023/gid00042/gid00039/gid00047/gid00028/gid00034/gid00032/gid00001/gid00042/gid00048/gid00047/gid00043/gid00048/gid00047
/gid00016/gid00043/gid00047/gid00036/gid00030/gid00028/gid00039/gid00001/gid00036/gid00041/gid00047/gid00032/gid00041/gid00046/gid00036/gid00047/gid00052
/gid00004/gid00028/gid00043/gid00028/gid00030/gid00036/gid00047/gid00028/gid00041/gid00030/gid00032/gid00001/gid00030/gid00035/gid00028/gid00041/gid00034/gid00032
/gid00010/gid00040/gid00043/gid00032/gid00031/gid00028/gid00041/gid00030/gid00032/gid00001/gid00030/gid00035/gid00028/gid00041/gid00034/gid00032
/gid00024/gid00028/gid00049/gid00032/gid00039/gid00032/gid00041/gid00034/gid00047/gid00035/gid00001/gid00046/gid00035/gid00036/gid00517
/gid00007/gid00045/gid00032/gid00044/gid00048/gid00032/gid00041/gid00030/gid00052/gid00001/gid00046/gid00035/gid00036/gid00517
/gid00004/gid00035/gid00028/gid00045/gid00034/gid00032/gid00001/gid00042/gid00048/gid00047/gid00043/gid00048/gid00047
/gid00014/gid00048/gid00039/gid00047/gid00036/gid00030/gid00035/gid00028/gid00041/gid00041/gid00032/gid00039/gid00001/gid00045/gid00032/gid00028/gid00031/gid00042/gid00048/gid00047
/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00020/gid00047/gid00045/gid00028/gid00036/gid00041
/gid00017/gid00045/gid00032/gid00046/gid00046/gid00048/gid00045/gid00032
/gid00021/gid00032/gid00040/gid00043/gid00032/gid00045/gid00028/gid00047/gid00048/gid00045/gid00032
/gid00003/gid00036/gid00042/gid00040/gid00028/gid00045/gid00038/gid00032/gid00045/gid00046
/gid00004/gid00035/gid00032/gid00040/gid00036/gid00030/gid00028/gid00039/gid00001/gid00028/gid00041/gid00028/gid00039/gid00052/gid00047/gid00032/gid00046
/gid00002/gid00030/gid00030/gid00032/gid00039/gid00032/gid00045/gid00028/gid00047/gid00036/gid00042/gid00041/gid00001/gid00028/gid00041/gid00031/gid00001/gid00045/gid00042/gid00047/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00009/gid00048/gid00040/gid00036/gid00031/gid00036/gid00047/gid00052
/gid00021/gid00042/gid00048/gid00030/gid00035
/gid00008/gid00028/gid00046
/gid00013/gid00036/gid00034/gid00035/gid00047
/gid01089/gid01094/gid01141/gid01089/gid01727
/gid01089/gid01088/gid01141/gid01089/gid01727
/gid01088/gid01090/gid01141/gid01094/gid01727
/gid01088/gid01088/gid01141/gid01087/gid01727
/gid01096/gid01141/gid01094/gid01727
/gid01095/gid01141/gid01088/gid01727
/gid01093/gid01141/gid01096/gid01727
/gid01093/gid01141/gid01091/gid01727
/gid01093/gid01141/gid01091/gid01727
/gid01090/gid01141/gid01094/gid01727/gid00041/gid00001/gid01753/gid00001/gid01090/gid01142/gid01094/gid01089/gid01094
/gid00041/gid00001/gid01753/gid00001/gid01089/gid01142/gid01096/gid01087/gid01088
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01095/gid01094/gid01088
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01092/gid01087/gid01092
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01090/gid01089/gid01089
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01088/gid01087/gid01093
/gid00041/gid00001/gid01753/gid00001/gid01096/gid01091/gid01092
/gid00041/gid00001/gid01753/gid00001/gid01095/gid01095/gid01089
/gid00041/gid00001/gid01753/gid00001/gid01095/gid01094/gid01091
/gid00041/gid00001/gid01753/gid00001/gid01092/gid01087/gid01091
/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039
/gid01088/gid01087/gid01141/gid01095/gid01727
/gid01088/gid01087/gid01141/gid01091/gid01727
/gid01094/gid01141/gid01093/gid01727
/gid01091/gid01141/gid01094/gid01727
/gid01091/gid01141/gid01091/gid01727
/gid01090/gid01141/gid01094/gid01727
/gid01090/gid01141/gid01088/gid01727
/gid01089/gid01141/gid01094/gid01727
/gid01089/gid01141/gid01089/gid01727
/gid01088/gid01141/gid01092/gid01727/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01091/gid01094/gid01094
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01091/gid01090/gid01087
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01087/gid01091/gid01090
/gid00041/gid00001/gid01753/gid00001/gid01093/gid01091/gid01092
/gid00041/gid00001/gid01753/gid00001/gid01092/gid01096/gid01095
/gid00041/gid00001/gid01753/gid00001/gid01092/gid01087/gid01090
/gid00041/gid00001/gid01753/gid00001/gid01091/gid01089/gid01087
/gid00041/gid00001/gid01753/gid00001/gid01090/gid01094/gid01091
/gid00041/gid00001/gid01753/gid00001/gid01089/gid01096/gid01094
/gid00041/gid00001/gid01753/gid00001/gid01089/gid01087/gid01095
/gid01089/gid01088/gid01141/gid01092/gid01727
/gid01088/gid01090/gid01141/gid01095/gid01727
/gid01088/gid01089/gid01141/gid01093/gid01727
/gid01096/gid01141/gid01091/gid01727
/gid01093/gid01141/gid01092/gid01727
/gid01090/gid01141/gid01094/gid01727
/gid01089/gid01141/gid01090/gid01727
/gid01089/gid01141/gid01089/gid01727
/gid01087/gid01141/gid01095/gid01727
/gid01087/gid01141/gid01089/gid01727/gid00041/gid00001/gid01753/gid00001/gid01089/gid01142/gid01096/gid01091/gid01092
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01095/gid01096/gid01090
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01094/gid01090/gid01088
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01089/gid01096/gid01089
/gid00041/gid00001/gid01753/gid00001/gid01095/gid01095/gid01091
/gid00041/gid00001/gid01753/gid00001/gid01092/gid01087/gid01092
/gid00041/gid00001/gid01753/gid00001/gid01090/gid01088/gid01089
/gid00041/gid00001/gid01753/gid00001/gid01089/gid01096/gid01093
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01087/gid01095
/gid00041/gid00001/gid01753/gid00001/gid01089/gid01089
/gid01088/gid01091/gid01141/gid01090/gid01727
/gid01088/gid01089/gid01141/gid01088/gid01727
/gid01093/gid01141/gid01087/gid01727
/gid01091/gid01141/gid01090/gid01727
/gid01089/gid01141/gid01096/gid01727
/gid01089/gid01141/gid01092/gid01727
/gid01089/gid01141/gid01091/gid01727
/gid01089/gid01141/gid01089/gid01727
/gid01088/gid01141/gid01095/gid01727
/gid01088/gid01141/gid01091/gid01727/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01096/gid01093/gid01089
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01142/gid01093/gid01092/gid01091
/gid00041/gid00001/gid01753/gid00001/gid01095/gid01089/gid01087
/gid00041/gid00001/gid01753/gid00001/gid01092/gid01095/gid01094
/gid00041/gid00001/gid01753/gid00001/gid01090/gid01096/gid01091
/gid00041/gid00001/gid01753/gid00001/gid01090/gid01091/gid01095
/gid00041/gid00001/gid01753/gid00001/gid01090/gid01089/gid01090
/gid00041/gid00001/gid01753/gid00001/gid01089/gid01096/gid01094
/gid00041/gid00001/gid01753/gid00001/gid01089/gid01091/gid01094
/gid00041/gid00001/gid01753/gid00001/gid01088/gid01095/gid01092
/gid01096/gid01089/gid01141/gid01096/gid01727/gid00001
/gid01090/gid01089/gid01141/gid01090/gid01727/gid00001
/gid01089/gid01095/gid01141/gid01091/gid01727/gid00001
/gid01088/gid01094/gid01141/gid01091/gid01727/gid00001Figure2:OverviewoftheMS3knowledgesubstrate,includingcorpusscale,domaincoverage,evidencecomposition,structural
coverage, and multimodal provenance.
Provenance-aware evidence selection and generation.
For the in-domain evaluation, the system retains at most
four papers, five mechanism objects per paper, and twelve
evidenceitemsperpaper.Theselectedrecordsareserialized
directlyasmechanismcardscontainingtypedpaths,missing
layers, evidence excerpts, identifiers, and source locations.
The evaluated implementation uses fixed top-klimits rather
than an explicit input-token budget, minimum-cover solver,
or redundancy-aware packing objective.
TheresultingcontextC qissuppliedtotheanswermodel:
ˆy=f θ(q, C q).(4)
The prompt requires evidence identifiers for supported
claims and permits aninsufficient evidenceresponse when
the requested information is not present.
Evidence sufficiency and source repair.The target sys-
tem treats an MS3card as a verified index into source evi-
dence rather than as an immutable summary. A claim-level
verifierfirstcheckswhethertheretrievedcardssupportevery
requestedrole,relation,number,unit,validationsetting,and
deployment statement. If support is missing, conflicting, or
too coarse, the system follows the selected paper identifiers
back to their stored full text, retrieves claim-directed raw
passages, and re-audits the affected fields. If no relevant pa-
periscoveredbythesubstrate,scholarlyWebsearchisused
for source discovery before the same evidence gates are ap-
plied. Thus, raw-PDF retrieval is an internal repair operator
reached through retrieval and verification, not a user-facing
condition that is handed a gold paper. Repaired cards re-
mainrequest-localunlessaversionedupdatereceivesexpertapproval. The experiments below evaluate the single-pass
MS3retrieval component. The ongoing open-domain study
evaluatescoveragerouting,whilethefullsource-repairloop
requires a separate controlled evaluation.
Experiments
Researchquestions.WeaskwhetherMS3retrieval(i)im-
proves scientific correctness over a language model without
retrieval,(ii)improvesoverscholarlyWebevidenceavailable
toarealisticsearchagent,and(iii)yieldsconsistentbenefits
across model families rather than for a single generator.
Benchmark.Weusea300-questionin-domainbenchmark
frozenbeforeanswergeneration.Itcontains100mechanism-
path reconstruction questions, 60 quantitative evidence
questions, 60 cross-paper comparisons, and 80 evidence-
boundary questions that require separating a demonstrated
system from a proposal or unsupported deployment claim.
The questions cover 362 target papers and are disjoint from
theearlierdevelopmentbenchmarkatthepaperlevel.Several
questionsincludeapapertitle;thebenchmarkthereforemea-
sures grounded evidence use after a source can be resolved,
rather than fully open-domain literature discovery.
We additionally construct a 200-question out-of-domain
benchmark. The covered stratum includes 70 single-topic
corpus-discovery questions and 30 cross-paper synthesis
questions. The out-of-domain stratum contains 50 near-
domainand50far-domainquestions.Weremovepapertitles,
DOIs, paper and evidence IDs, and exact source-case iden-
tifiers from every question. Hidden source-paper IDs and

/gid00030/gid00001/gid00005/gid00032/gid00030/gid00042/gid00040/gid00043/gid00042/gid00046/gid00032/gid00031/gid00001/gid00032/gid00051/gid00047/gid00045/gid00028/gid00030/gid00047/gid00036/gid00042/gid00041/gid00001/gid00042/gid00033/gid00001/gid00047/gid00035/gid00032/gid00001/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid01162/gid00034/gid00045/gid00042/gid00048/gid00041/gid00031/gid00032/gid00031/gid00001/gid00014/gid00020/gid01577/gid00001/gid00046/gid00048/gid00029/gid00046/gid00047/gid00045/gid00028/gid00047/gid00032
/gid00006/gid01162/gid00013/gid00042/gid00019/gid00002
/gid00013/gid00013/gid00014
/gid00021/gid01162/gid00013/gid00042/gid00019/gid00002
/gid00013/gid00013/gid00014
/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039
/gid00014/gid00025/gid00032/gid00041/gid00032/gid01162/gid00030/gid00042/gid00028/gid00047/gid00032/gid00031/gid00001/gid00030/gid00042/gid00047/gid00047/gid00042/gid00041
/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045
/gid00043/gid00045/gid00032/gid00046/gid00046/gid00048/gid00045/gid00032/gid00001/gid00046/gid00032/gid00041/gid00046/gid00042/gid00045
/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039
/gid00045/gid00032/gid00039/gid00028/gid00047/gid00036/gid00049/gid00032/gid00001/gid00030/gid00048/gid00045/gid00045/gid00032/gid00041/gid00047/gid00001/gid00030/gid00035/gid00028/gid00041/gid00034/gid00032
/gid00020/gid00052/gid00046/gid00047/gid00032/gid00040
/gid00040/gid00042/gid00047/gid00036/gid00042/gid00041/gid00001/gid00031/gid00032/gid00047/gid00032/gid00030/gid00047/gid00036/gid00042/gid00041
/gid00014/gid01162/gid00013/gid00042/gid00019/gid00002
/gid00013/gid00013/gid00014
/gid00017/gid00005/gid00007/gid00013/gid00042/gid00041/gid00034/gid00001/gid00047/gid00032/gid00051/gid00047
/gid00030/gid00035/gid00048/gid00041/gid00038/gid00046
/gid00007/gid00036/gid00034/gid00048/gid00045/gid00032/gid00046/gid01142/gid00001
/gid00030/gid00028/gid00043/gid00047/gid00036/gid00042/gid00041/gid00046/gid00001
/gid00028/gid00041/gid00031/gid00001/gid00047/gid00028/gid00029/gid00039/gid00032/gid00046❄🔥❄ ❄🔥 🔥
/gid00006/gid00051/gid00043/gid00032/gid00045/gid00047/gid00046/gid00032/gid00032/gid00031/gid00001/gid00043/gid00028/gid00043/gid00032/gid00045/gid00046
/gid00013/gid00013/gid00014 ❄/gid01088/gid01087/gid01087/gid00001/gid00032/gid00051/gid00043/gid00032/gid00045/gid00047/gid00001/gid00028/gid00041/gid00041/gid00042/gid00047/gid00028/gid00047/gid00032/gid00031
/gid00043/gid00028/gid00043/gid00032/gid00045/gid00046
/gid01089/gid01087/gid01087/gid01087/gid01748/gid00001/gid00043/gid00028/gid00043/gid00032/gid00045/gid00046
/gid00031/gid00032/gid00032/gid00043/gid00046/gid00032/gid00032/gid00038/gid00001/gid00049/gid01091/gid00001/gid00043/gid00045/gid00042
/gid01141/gid01141/gid01141
/gid01141/gid01141/gid01141/gid01141/gid01141/gid01141
/gid01141/gid01141/gid01141/gid00021/gid00032/gid00028/gid00030/gid00035/gid00032/gid00045/gid01162/gid00028/gid00046/gid00046/gid00036/gid00046/gid00047/gid00032/gid00031
/gid00046/gid00036/gid00039/gid00049/gid00032/gid00045/gid00001/gid00039/gid00028/gid00029/gid00032/gid00039/gid00036/gid00041/gid00034
/gid00006/gid00051/gid00043/gid00032/gid00045/gid00047/gid00001/gid00046/gid00043/gid00042/gid00047/gid01162/gid00030/gid00035/gid00032/gid00030/gid00038
/gid00008/gid00045/gid00042/gid00048/gid00041/gid00031/gid00036/gid00041/gid00034/gid00001/gid00028/gid00041/gid00031
/gid00046/gid00030/gid00035/gid00032/gid00040/gid00028/gid00001/gid00049/gid00028/gid00039/gid00036/gid00031/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00023/gid00006/gid00019/gid00010/gid00007/gid00026
/gid00046/gid00030/gid00035/gid00032/gid00040/gid00028/gid00046/gid00042/gid00048/gid00045/gid00030/gid00032/gid00001/gid00039/gid00036/gid00041/gid00038/gid00046
/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00010/gid00005/gid00046
/gid00031/gid00032/gid00031/gid00048/gid00043/gid00039/gid00036/gid00030/gid00028/gid00047/gid00036/gid00042/gid00041/gid00014/gid00036/gid00051/gid00032/gid00031/gid00001/gid00006/gid00051/gid00043/gid00032/gid00045/gid00047/gid01748/gid00020/gid00036/gid00039/gid00049/gid00032/gid00045
/gid00020/gid00007/gid00021/gid00001/gid00030/gid00042/gid00045/gid00043/gid00048/gid00046
/gid00021/gid00042/gid00043/gid01090/gid00001/gid00045/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00032/gid00031
/gid00032/gid00051/gid00043/gid00032/gid00045/gid00047/gid00001
/gid00031/gid00032/gid00040/gid00042/gid00041/gid00046/gid00047/gid00045/gid00028/gid00047/gid00036/gid00042/gid00041/gid00046
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00046/gid00001/gid00001/gid00001/gid00001/gid00001/gid00539/gid00515/gid00036/gid00031/gid00515/gid00507/gid00001/gid00515/gid00006/gid00467/gid00515/gid00506
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00515/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00534/gid00047/gid00052/gid00043/gid00032/gid00515/gid00507/gid00001/gid00515/gid00032/gid00051/gid00043/gid00032/gid00045/gid00036/gid00040/gid00032/gid00041/gid00047/gid00515/gid00506
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00515/gid00028/gid00045/gid00047/gid00036/gid00030/gid00039/gid00032/gid00534/gid00032/gid00051/gid00030/gid00032/gid00045/gid00043/gid00047/gid00515/gid00507/gid00001/gid00537
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00515/gid00006/gid00039/gid00032/gid00030/gid00047/gid00045/gid00042/gid00046/gid00043/gid00048/gid00041/gid00001/gid00041/gid00028/gid00041/gid00042/gid00033/gid00036/gid00029/gid00032/gid00045/gid00046/gid00001/gid00035/gid00028/gid00049/gid00032/gid00001/gid00029/gid00032/gid00032/gid00041/gid00001
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00032/gid00051/gid00047/gid00032/gid00041/gid00046/gid00036/gid00049/gid00032/gid00039/gid00052/gid00001/gid00036/gid00041/gid00049/gid00032/gid00046/gid00047/gid00036/gid00034/gid00028/gid00047/gid00032/gid00031/gid00001/gid00036/gid00041/gid00001/gid00045/gid00032/gid00030/gid00032/gid00041/gid00047/gid00001/gid00052/gid00032/gid00028/gid00045/gid00046/gid00001
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00033/gid00042/gid00045/gid00001/gid00047/gid00035/gid00032/gid00001/gid00034/gid00032/gid00041/gid00032/gid00045/gid00028/gid00047/gid00036/gid00042/gid00041/gid00001/gid00042/gid00033/gid00001/gid00049/gid00028/gid00045/gid00036/gid00042/gid00048/gid00046/gid00001/gid00028/gid00031/gid00049/gid00028/gid00041/gid00030/gid00032/gid00031/gid00001
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00046/gid00032/gid00041/gid00046/gid00036/gid00041/gid00034/gid00001/gid00047/gid00032/gid00030/gid00035/gid00041/gid00036/gid00044/gid00048/gid00032/gid00001/gid00039/gid00036/gid00038/gid00032/gid00001/gid00506/gid00001/gid00505/gid00505/gid00505 /gid00515/gid00001/gid00538/gid00506/gid00001/gid00505/gid00505/gid00505/gid00540/gid00001/gid00001/gid00001/gid00001/gid00001/gid00539/gid00515/gid00036/gid00031/gid00515/gid00507/gid00001/gid00515/gid00006/gid00466/gid00515/gid00506
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00515/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00534/gid00047/gid00052/gid00043/gid00032/gid00515/gid00507/gid00001/gid00515/gid00032/gid00051/gid00043/gid00032/gid00045/gid00036/gid00040/gid00032/gid00041/gid00047/gid00515/gid00506
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00515/gid00028/gid00045/gid00047/gid00036/gid00030/gid00039/gid00032/gid00534/gid00032/gid00051/gid00030/gid00032/gid00045/gid00043/gid00047/gid00515/gid00507/gid00001/gid00537
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00515/gid00006/gid00039/gid00032/gid00030/gid00047/gid00045/gid00042/gid00046/gid00043/gid00048/gid00041/gid00001/gid00041/gid00028/gid00041/gid00042/gid00033/gid00036/gid00029/gid00032/gid00045/gid00046/gid00001/gid00035/gid00028/gid00049/gid00032/gid00001/gid00029/gid00032/gid00032/gid00041/gid00001
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00032/gid00051/gid00047/gid00032/gid00041/gid00046/gid00036/gid00049/gid00032/gid00039/gid00052/gid00001/gid00036/gid00041/gid00049/gid00032/gid00046/gid00047/gid00036/gid00034/gid00028/gid00047/gid00032/gid00031/gid00001/gid00036/gid00041/gid00001/gid00045/gid00032/gid00030/gid00032/gid00041/gid00047/gid00001/gid00052/gid00032/gid00028/gid00045/gid00046/gid00001
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00033/gid00042/gid00045/gid00001/gid00047/gid00035/gid00032/gid00001/gid00034/gid00032/gid00041/gid00032/gid00045/gid00028/gid00047/gid00036/gid00042/gid00041/gid00001/gid00042/gid00033/gid00001/gid00049/gid00028/gid00045/gid00036/gid00042/gid00048/gid00046/gid00001/gid00028/gid00031/gid00049/gid00028/gid00041/gid00030/gid00032/gid00031/gid00001
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00046/gid00032/gid00041/gid00046/gid00036/gid00041/gid00034/gid00001/gid00047/gid00032/gid00030/gid00035/gid00041/gid00036/gid00044/gid00048/gid00032/gid00001/gid00039/gid00036/gid00038/gid00032/gid00001/gid00506/gid00001/gid00505/gid00505/gid00505 /gid00515/gid00001/gid00538/gid00506/gid00001/gid00505/gid00505/gid00505/gid00540
/gid00014/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00001/gid00021/gid00032/gid00045/gid00040/gid00046
/gid00014/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00001/gid00042/gid00029/gid00037/gid00032/gid00030/gid00047/gid00046/gid00014/gid00016/gid00534/gid00465/gid00465/gid00466
/gid00008/gid00042/gid00039/gid00031/gid00001/gid00047/gid00032/gid00046/gid00047/gid00001/gid00046/gid00032/gid00047/gid00028/gid00001/gid00006/gid00051/gid00043/gid00032/gid00045/gid00047/gid01162/gid00034/gid00048/gid00036/gid00031/gid00032/gid00031/gid00001/gid00020/gid00007/gid00021/gid00001/gid00030/gid00042/gid00045/gid00043/gid00048/gid00046/gid00001/gid00030/gid00042/gid00041/gid00046/gid00047/gid00045/gid00048/gid00030/gid00047/gid00036/gid00042/gid00041
/gid00029/gid00001/gid00010/gid00041/gid00031/gid00032/gid00043/gid00032/gid00041/gid00031/gid00032/gid00041/gid00047/gid00001/gid00034/gid00042/gid00039/gid00031/gid01162/gid00047/gid00032/gid00046/gid00047/gid01162/gid00046/gid00032/gid00047/gid00001/gid00030/gid00042/gid00041/gid00046/gid00047/gid00045/gid00048/gid00030/gid00047/gid00036/gid00042/gid00041
/gid01141/gid01141/gid01141
/gid00006/gid00051/gid00043/gid00032/gid00045/gid00047/gid00001/gid01088
/gid00006/gid00051/gid00043/gid00032/gid00045/gid00047/gid00001/gid01089/gid00021/gid00035/gid00036/gid00045/gid00031/gid00001/gid00032/gid00051/gid00043/gid00032/gid00045/gid00047
/gid00028/gid00031/gid00037/gid00048/gid00031/gid00036/gid00030/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00028/gid00034/gid00045/gid00032/gid00032/gid01148
/gid00026/gid00032/gid00046/gid00015/gid00042
/gid00014/gid00020/gid01577/gid00001/gid00043/gid00028/gid00047/gid00035/gid01143/gid00001/gid00014/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039/gid00001/gid01777/gid00001/gid00020/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid01777/gid00001/gid00020/gid00036/gid00034/gid00041/gid00028/gid00039/gid00001/gid01777/gid00001/gid00020/gid00052/gid00046/gid00047/gid00032/gid00040
/gid00014/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00001/gid00046/gid00047/gid00028/gid00047/gid00032/gid00040/gid00032/gid00041/gid00047/gid01143/gid00001/gid00017/gid00045/gid00032/gid00046/gid00046/gid00048/gid00045/gid00032/gid01162/gid00036/gid00041/gid00031/gid00048/gid00030/gid00032/gid00031/gid00001/gid00030/gid00042/gid00041/gid00047/gid00028/gid00030/gid00047/gid01162/gid00028/gid00045/gid00032/gid00028/gid00001/gid00032/gid00051/gid00043/gid00028/gid00041/gid00046/gid00036/gid00042/gid00041/gid00001
/gid00031/gid00032/gid00030/gid00045/gid00032/gid00028/gid00046/gid00032/gid00046/gid00001/gid00045/gid00032/gid00046/gid00036/gid00046/gid00047/gid00028/gid00041/gid00030/gid00032/gid00001/gid00028/gid00041/gid00031/gid00001/gid00036/gid00041/gid00030/gid00045/gid00032/gid00028/gid00046/gid00032/gid00046/gid00001/gid00030/gid00048/gid00045/gid00045/gid00032/gid00041/gid00047/gid01141
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00030/gid00035/gid00028/gid00036/gid00041/gid01143/gid00001/gid00032/gid00051/gid00043/gid00032/gid00045/gid00036/gid00040/gid00032/gid00041/gid00047/gid00001/gid00006/gid01093/gid01142/gid00001/gid00043/gid00045/gid00042/gid00043/gid00032/gid00045/gid00047/gid00052/gid00001/gid00006/gid01090/gid01142/gid00001/gid00006/gid01096/gid01142/gid00001/gid00049/gid00028/gid00039/gid00036/gid00031/gid00028/gid00047/gid00036/gid00042/gid00041/gid00001/gid00006/gid01088/gid01088
/gid00023/gid00032/gid00045/gid00036/gid00033/gid00052/gid00017/gid00028/gid00043/gid00032/gid00045/gid00046Figure 3: Expert-guided supervision, paper-disjoint bench-
mark construction, and decomposed extraction of evidence-
grounded MS3mechanism objects.
answer outlines are retained only for post-retrieval auditing
and are never exposed to the retriever or answer generator.
Comparedsystems.Wecomparefourconditionsinthein-
domainevaluationandthreesingle-sourceorhybridretrieval
conditionsinthecoverage-shiftevaluation.Closed-bookDi-
rectreceives only the question and is instructed to provide a
best-effort answer from parametric knowledge without fab-
ricatingpaper-specificdetails.WebSearchreceivesevidence
frozen before answer generation from at most three queries
andfourscholarlysourcesdiscoveredthroughOpenAlexand
Crossref.VerifiedMS3receivestheauditedmechanismcards
andtheirsourceexcerpts.Forthein-domainevaluation,Raw-
PDF RAGreceives the complete text of the known target
paper and serves as an oracle-input comparison. It is not
evaluated in the coverage-shift setting, where no single tar-
get paper is assumed to be known in advance. Hybrid com-
bines frozen Web evidence with Verified-MS3mechanism
cards. The two sources are retrieved independently, dedu-
plicated while preserving provenance, and jointly provided
to the same generator. MS3supplies structured mechanism
paths and evidence boundaries, whereas Web retrieval con-
tributesevidencebeyondthecorpuscoverage.Thegenerator
mustcitethecorrespondingsourceandabstainwhenneither
source supports a claim.
Modelsandgeneration.Thecompletedevaluationcovers
Qwen2.5-14B, Qwen2.5-72B, Qwen3-30B-A3B, Mistral-Small-24B, DeepSeek-V4-Flash, DeepSeek-V4-Pro, Kimi-
K3,Kimi-K2.6,GLM-5.2,andGLM-5-Turbo.Openmodels
are run from frozen local checkpoints and the DeepSeek,
Kimi and GLM models through their versioned API end-
points. Each model receives the same question and method-
specificfrozencontext.Weusedeterministicdecodingwhere
supportedandacommonstructuredanswerschema.TheDi-
rect prompt permits the use of parametric knowledge and
does not force refusal merely because citations are unavail-
able; its citation field is empty by design.
Blind evaluation and statistics.For each question,
candidate identities, method names, model names, and
representation-specific citation identifiers are hidden from
thejudge.Thejudgereceivestheunionofverbatimevidence
neededtocheckallcandidatesandindependentlyscoressci-
entificcorrectness,citationentailment,andcompletenesson
a0–4scale.Absolutecalibrationcanthereforedifferbetween
judge groups; primary inference uses within-model paired
comparisons. Primary comparisons are conducted within
each generator using paired question-level scores. Citation
entailmentisnotapplicabletoClosed-bookDirectandisnot
used to penalize its scientific correctness.
Hardware and implementation.All local experiments
were conducted on a server equipped with three NVIDIA
A100 80GB GPUs. Training used all three GPUs, whereas
each local generator was evaluated on a single GPU. Our
implementation uses Python 3.11.13, PyTorch 2.5.1 with
CUDA 12.4, Transformers 4.57.6, PEFT 0.18.1, bitsand-
bytes 0.49.2, DeepSpeed 0.19.1, and FlashAttention 2.7.4.
We fix the training seed to 42 and freeze all benchmark
questions, retrieved contexts, and source documents before
answer generation.
MS3extraction models.The evidence extractor (E), ter-
minology grounder (T), and mechanism inducer (M) are
task-specific adapters of Qwen2.5-72B-Instruct. We use 4-
bit QLoRA with NF4 quantization and double quantiza-
tion. LoRA adapters are applied to theq_proj,k_proj,
v_proj, ando_projmodules with rank 16, scaling fac-
tor 16, and dropout 0.05. The maximum input length is
14,000 tokens, and sequence packing is disabled. Training
usesBF16,gradientcheckpointing,andDeepSpeedZeRO-3.
The per-device batch size is 1 with eight gradient accumu-
lation steps, giving an effective batch size of 24 over three
GPUs. We use a cosine learning-rate schedule, zero weight
decay, and a maximum gradient norm of 0.3.
Mainresults.Table1showsaconsistentretrievalbenefit.
MS3improvesscientificcorrectnessoverClosed-bookDirect
for all ten generators, with a macro-average gain of 54.4
points. Compared with frozen Web Search, MS3gains 26.9
points on average and is better for 9 of 10 generators. The
exception is Qwen3-30B-A3B, for which Web exceeds MS3
by 5.0 points; Thus, structured context is not equally usable
by every generator.
Table2showsthatthecorrectnessgainisaccompaniedby
stronger grounding rather than only longer answers. Com-
pared with Web Search, MS3improves citation entailment

/gid01088/gid01090/gid01142/gid01093/gid01095/gid01096/gid00001/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid01162/gid00039/gid00036/gid00041/gid00038/gid00032/gid00031
/gid00043/gid00028/gid00043/gid00032/gid00045/gid00001/gid00045/gid00032/gid00030/gid00042/gid00045/gid00031/gid00046
/gid01141/gid01141/gid01141/gid00016/gid00514/gid00039/gid00036/gid00041/gid00032/gid00001/gid00041/gid00042/gid00045/gid00040/gid00028/gid00039/gid00036/gid00053/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00028/gid00041/gid00031/gid00001/gid00036/gid00041/gid00031/gid00032/gid00051/gid00036/gid00041/gid00034/gid00020/gid00030/gid00036/gid00032/gid00041/gid00047/gid00036/gid00033/gid00036/gid00030/gid00001/gid00018/gid00048/gid00032/gid00046/gid00047/gid00036/gid00042/gid00041
/gid00007/gid00042/gid00045/gid00001/gid00034/gid00028/gid00046/gid01162/gid00043/gid00035/gid00028/gid00046/gid00032/gid00001/gid00020/gid01175/gid01749/gid01176/gid01162/gid00039/gid00036/gid00040/gid00042/gid00041/gid00032/gid00041/gid00032/gid00001/gid00046/gid00032/gid00041/gid00046/gid00036/gid00041/gid00034/gid00001/gid00036/gid00041/gid00001/gid00043/gid00045/gid00032/gid00031/gid00036/gid00030/gid00047/gid00036/gid00049/gid00032/gid00001
/gid00028/gid00034/gid00045/gid00036/gid00030/gid00048/gid00039/gid00047/gid00048/gid00045/gid00032/gid01142/gid00001/gid00050/gid00035/gid00028/gid00047/gid00001/gid00040/gid00028/gid00047/gid00032/gid00045/gid00036/gid00028/gid00039/gid01164/gid00046/gid00032/gid00041/gid00046/gid00042/gid00045/gid00001/gid00028/gid00045/gid00030/gid00035/gid00036/gid00047/gid00032/gid00030/gid00047/gid00048/gid00045/gid00032/gid00001/gid00035/gid00028/gid00046/gid00001
/gid00029/gid00032/gid00032/gid00041/gid00001/gid00031/gid00032/gid00040/gid00042/gid00041/gid00046/gid00047/gid00045/gid00028/gid00047/gid00032/gid00031/gid01142/gid00001/gid00035/gid00042/gid00050/gid00001/gid00036/gid00046/gid00001/gid00047/gid00035/gid00032/gid00001/gid00032/gid00039/gid00032/gid00030/gid00047/gid00045/gid00036/gid00030/gid00028/gid00039/gid00001/gid00046/gid00036/gid00034/gid00041/gid00028/gid00039/gid00001
/gid00034/gid00032/gid00041/gid00032/gid00045/gid00028/gid00047/gid00032/gid00031/gid01142/gid00001/gid00028/gid00041/gid00031/gid00001/gid00050/gid00035/gid00028/gid00047/gid00001/gid00043/gid00032/gid00045/gid00033/gid00042/gid00045/gid00040/gid00028/gid00041/gid00030/gid00032/gid00001/gid00028/gid00041/gid00031/gid00001/gid00032/gid00041/gid00049/gid00036/gid00045/gid00042/gid00041/gid01162
/gid00040/gid00032/gid00041/gid00047/gid00028/gid00039/gid00001/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00035/gid00028/gid00046/gid00001/gid00029/gid00032/gid00032/gid00041/gid00001/gid00045/gid00032/gid00043/gid00042/gid00045/gid00047/gid00032/gid00031/gid01148
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032
/gid00010/gid00041/gid00031/gid00032/gid00051/gid00016/gid00514/gid00039/gid00036/gid00041/gid00032
/gid00013/gid00013/gid00014/gid00018/gid00048/gid00032/gid00045/gid00052/gid00021/gid00035/gid00032/gid00001/gid00046/gid00047/gid00048/gid00031/gid00052/gid00001/gid00031/gid00032/gid00040/gid00042/gid00041/gid00046/gid00047/gid00045/gid00028/gid00047/gid00032/gid00046/gid00001/gid00014/gid00010/gid00017/gid00001/gid00041/gid00028/gid00041/gid00042/gid00043/gid00028/gid00045/gid00047/gid00036/gid00030/gid00039/gid00032/gid00046/gid00001/gid00033/gid00048/gid00041/gid00030/gid00047/gid00036/gid00042/gid00041/gid00028/gid00039/gid00036/gid00053/gid00032/gid00031/gid00001
/gid00042/gid00041/gid00047/gid00042/gid00001/gid00041/gid00028/gid00041/gid00042/gid00033/gid00036/gid00029/gid00032/gid00045/gid00046/gid00001/gid00033/gid00042/gid00045/gid00001/gid00034/gid00028/gid00046/gid01162/gid00043/gid00035/gid00028/gid00046/gid00032/gid00001/gid00039/gid00036/gid00040/gid00042/gid00041/gid00032/gid00041/gid00032/gid00001/gid00031/gid00032/gid00047/gid00032/gid00030/gid00047/gid00036/gid00042/gid00041/gid01141/gid00001
/gid00013/gid00036/gid00040/gid00042/gid00041/gid00032/gid00041/gid00032/gid00001/gid00029/gid00036/gid00041/gid00031/gid00036/gid00041/gid00034/gid00001/gid00047/gid00042/gid00001/gid00014/gid00010/gid00017/gid00001/gid00030/gid00028/gid00049/gid00036/gid00047/gid00036/gid00032/gid00046/gid00001/gid00030/gid00035/gid00028/gid00041/gid00034/gid00032/gid00046/gid00001/gid00047/gid00035/gid00032/gid00001/gid00032/gid00039/gid00032/gid00030/gid00047/gid00045/gid00036/gid00030/gid00028/gid00039/gid00001
/gid00043/gid00045/gid00042/gid00043/gid00032/gid00045/gid00047/gid00036/gid00032/gid00046/gid00001/gid00042/gid00033/gid00001/gid00047/gid00035/gid00032/gid00001/gid00030/gid00042/gid00041/gid00031/gid00048/gid00030/gid00047/gid00036/gid00049/gid00032/gid00001/gid00041/gid00028/gid00041/gid00042/gid00033/gid00036/gid00029/gid00032/gid00045/gid00001/gid00040/gid00028/gid00047/gid01142/gid00001/gid00050/gid00035/gid00036/gid00030/gid00035/gid00001/gid00036/gid00046/gid00001
/gid00031/gid00032/gid00046/gid00030/gid00045/gid00036/gid00029/gid00032/gid00031/gid00001/gid00028/gid00046/gid00001/gid00028/gid00001/gid00045/gid00032/gid00046/gid00036/gid00046/gid00047/gid00028/gid00041/gid00030/gid00032/gid00001/gid00030/gid00035/gid00028/gid00041/gid00034/gid00032/gid01141/gid00001/gid00021/gid00035/gid00032/gid00001/gid00050/gid00042/gid00045/gid00038/gid00001/gid00036/gid00046/gid00001/gid00043/gid00045/gid00032/gid00046/gid00032/gid00041/gid00047/gid00032/gid00031/gid00001
/gid00028/gid00046/gid00001/gid00028/gid00001/gid00039/gid00028/gid00029/gid00042/gid00045/gid00028/gid00047/gid00042/gid00045/gid00052/gid00001/gid00043/gid00045/gid00042/gid00042/gid00033/gid01162/gid00042/gid00033/gid01162/gid00030/gid00042/gid00041/gid00030/gid00032/gid00043/gid00047/gid00001/gid00033/gid00042/gid00045/gid00001/gid00028/gid00034/gid00045/gid00036/gid00030/gid00048/gid00039/gid00047/gid00048/gid00045/gid00028/gid00039/gid00001/gid00023/gid00016/gid00004/gid00001
/gid00040/gid00042/gid00041/gid00036/gid00047/gid00042/gid00045/gid00036/gid00041/gid00034/gid01142/gid00001/gid00050/gid00036/gid00047/gid00035/gid00042/gid00048/gid00047/gid00001/gid00033/gid00036/gid00032/gid00039/gid00031/gid00001/gid00049/gid00028/gid00039/gid00036/gid00031/gid00028/gid00047/gid00036/gid00042/gid00041/gid01141/gid00013/gid00013/gid00014/gid01162/gid00042/gid00041/gid00039/gid00052/gid00001/gid00019/gid00032/gid00046/gid00043/gid00042/gid00041/gid00046/gid00032
/gid00013/gid00013/gid00014/gid00018/gid00048/gid00032/gid00045/gid00052
/gid00019/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00032/gid00031/gid00001/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00036/gid00031/gid00032/gid00041/gid00047/gid00036/gid00033/gid00036/gid00032/gid00046/gid00001/gid00014/gid00010/gid00017/gid00001/gid00041/gid00028/gid00041/gid00042/gid00043/gid00028/gid00045/gid00047/gid00036/gid00030/gid00039/gid00032/gid00046
/gid00032/gid00040/gid00029/gid00032/gid00031/gid00031/gid00032/gid00031/gid00001/gid00036/gid00041/gid00001/gid00017/gid00023/gid00017/gid01181/gid00014/gid00024/gid00004/gid00015/gid00021/gid00001/gid00041/gid00028/gid00041/gid00042/gid00033/gid00036/gid00029/gid00032/gid00045/gid00046/gid00001/gid00042/gid00041/gid00001/gid00036/gid00041/gid00047/gid00032/gid00045/gid00031/gid00036/gid00034/gid00036/gid00047/gid00028/gid00047/gid00032/gid00031/gid00001
/gid00032/gid00039/gid00032/gid00030/gid00047/gid00045/gid00042/gid00031/gid00032/gid00046/gid00001/gid00028/gid00046/gid00001/gid00047/gid00035/gid00032/gid00001/gid00030/gid00035/gid00032/gid00040/gid00036/gid00045/gid00032/gid00046/gid00036/gid00046/gid00047/gid00036/gid00049/gid00032/gid00001/gid00028/gid00045/gid00030/gid00035/gid00036/gid00047/gid00032/gid00030/gid00047/gid00048/gid00045/gid00032/gid00001/gid01177/gid00017/gid01090/gid01091/gid01091/gid01091/gid01143
/gid00006/gid01088/gid01087/gid01178/gid01141/gid00001/gid00013/gid00036/gid00040/gid00042/gid00041/gid00032/gid00041/gid00032/gid00001/gid00032/gid00051/gid00043/gid00042/gid00046/gid00048/gid00045/gid00032/gid00001/gid00043/gid00045/gid00042/gid00031/gid00048/gid00030/gid00032/gid00046/gid00001/gid00028/gid00001/gid00041/gid00042/gid00045/gid00040/gid00028/gid00039/gid00036/gid00053/gid00032/gid00031/gid00001
/gid00030/gid00048/gid00045/gid00045/gid00032/gid00041/gid00047/gid00001/gid00045/gid00032/gid00046/gid00043/gid00042/gid00041/gid00046/gid00032/gid01142/gid00001/gid00578/gid00010/gid01181/gid00010/gid01588/gid00001/gid01177/gid00017/gid01090/gid01091/gid01091/gid01091/gid01143/gid00006/gid01092/gid01178/gid01141/gid00001/gid00021/gid00035/gid00032/gid00001/gid00045/gid00032/gid00043/gid00042/gid00045/gid00047/gid00032/gid00031/gid00001/gid00013/gid00016/gid00005/gid00001/gid00036/gid00046/gid00001
/gid01088/gid01096/gid01087/gid00001/gid00043/gid00043/gid00029/gid00001/gid00028/gid00041/gid00031/gid00001/gid00047/gid00035/gid00032/gid00001/gid00046/gid00032/gid00039/gid00032/gid00030/gid00047/gid00036/gid00049/gid00036/gid00047/gid00052/gid00001/gid00036/gid00041/gid00031/gid00032/gid00051/gid00001/gid00036/gid00046/gid00001/gid01094/gid01090/gid01727/gid00001/gid01177/gid00017/gid01090/gid01091/gid01091/gid01091/gid01143/gid00006/gid01088/gid01087/gid01178/gid01144/gid00001
/gid00047/gid01096/gid01087/gid00001/gid00036/gid00046/gid00001/gid01088/gid01095/gid01087/gid00001/gid00046/gid00001/gid00028/gid00047/gid00001/gid01090/gid01087/gid00001/gid00043/gid00043/gid00040/gid00001/gid00028/gid00041/gid00031/gid00001/gid01092/gid01092/gid01727/gid00001/gid00019/gid00009/gid00001/gid01177/gid00017/gid01090/gid01091/gid01091/gid01091/gid01143/gid00006/gid01092/gid01178/gid01141/gid00001/gid00019/gid00032/gid00031/gid00048/gid00030/gid00036/gid00041/gid00034/gid00001
/gid00019/gid00009/gid00001/gid00033/gid00045/gid00042/gid00040/gid00001/gid01095/gid01087/gid01727/gid00001/gid00047/gid00042/gid00001/gid01092/gid01092/gid01727/gid00001/gid00036/gid00041/gid00030/gid00045/gid00032/gid00028/gid00046/gid00032/gid00031/gid00001/gid00578/gid00010/gid01181/gid00010/gid01588/gid00001/gid00033/gid00045/gid00042/gid00040/gid00001/gid01088/gid01141/gid01087/gid00001/gid00047/gid00042/gid00001/gid01090/gid01141/gid01093/gid00001/gid00028/gid00041/gid00031/gid00001
/gid00046/gid00035/gid00042/gid00045/gid00047/gid00032/gid00041/gid00032/gid00031/gid00001/gid00047/gid01096/gid01087/gid00001/gid00033/gid00045/gid00042/gid00040/gid00001/gid01090/gid01089/gid01092/gid00001/gid00047/gid00042/gid00001/gid01088/gid01090/gid01092/gid00001/gid00046/gid00001/gid01177/gid00017/gid01090/gid01091/gid01091/gid01091/gid01143/gid00006/gid01096/gid01178/gid01141/gid00001/gid00007/gid00036/gid00032/gid00039/gid00031/gid00001
/gid00031/gid00032/gid00043/gid00039/gid00042/gid00052/gid00040/gid00032/gid00041/gid00047/gid00001/gid00036/gid00041/gid00001/gid00028/gid00034/gid00045/gid00036/gid00030/gid00048/gid00039/gid00047/gid00048/gid00045/gid00032/gid00001/gid00050/gid00028/gid00046/gid00001/gid00041/gid00042/gid00047/gid00001/gid00031/gid00032/gid00040/gid00042/gid00041/gid00046/gid00047/gid00045/gid00028/gid00047/gid00032/gid00031/gid00001/gid00036/gid00041/gid00001/gid00047/gid00035/gid00032/gid00001
/gid00045/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00032/gid00031/gid00001/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid01141/gid00014/gid00020/gid01619/gid01162/gid00034/gid00045/gid00042/gid00048/gid00041/gid00031/gid00032/gid00031/gid00001/gid00019/gid00032/gid00046/gid00043/gid00042/gid00041/gid00046/gid00032
/gid00014/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00001/gid00016/gid00029/gid00037/gid00032/gid00030/gid00047
/gid00010/gid00041/gid00031/gid00032/gid00051/gid00006/gid01092
/gid00014
/gid00020
/gid00020/gid00036/gid00034
/gid00020/gid00052/gid00046/gid01779/gid00001/gid00001/gid00001/gid00017/gid00028/gid00045/gid00028/gid00040/gid00032/gid00047/gid00045/gid00036/gid00030/gid00001/gid00038/gid00041/gid00042/gid00050/gid00039/gid00032/gid00031/gid00034/gid00032
/gid01779/gid00001/gid00001/gid00001/gid00015/gid00042/gid00001/gid00030/gid00042/gid00045/gid00043/gid00048/gid00046/gid00001/gid00030/gid00036/gid00047/gid00028/gid00047/gid00036/gid00042/gid00041
/gid00006/gid01096
/gid00006/gid01088/gid01087
/gid00006/gid01088/gid01088/gid00014/gid00016/gid01088
/gid00014/gid00016/gid01089
/gid00014/gid00016/gid01090
/gid00014/gid00016/gid01164/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032
/gid00017/gid00045/gid00042/gid00049/gid00032/gid00041/gid00028/gid00041/gid00030/gid00032/gid00001/gid00013/gid00036/gid00041/gid00038/gid00046/gid01779/gid00001/gid00001/gid00001/gid00019/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00032/gid00031/gid00001/gid00040/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00001/gid00030/gid00042/gid00041/gid00047/gid00032/gid00051/gid00047/gid00001/gid00001
/gid01779/gid00001/gid00001/gid00001/gid00004/gid00039/gid00028/gid00036/gid00040/gid01162/gid00039/gid00032/gid00049/gid00032/gid00039/gid00001/gid00032/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00030/gid00036/gid00047/gid00028/gid00047/gid00036/gid00042/gid00041/gid00001/gid00001
/gid01779/gid00001/gid00001/gid00006/gid00051/gid00043/gid00039/gid00036/gid00030/gid00036/gid00047/gid00001/gid00046/gid00048/gid00043/gid00043/gid00042/gid00045/gid00047/gid00001/gid00029/gid00042/gid00048/gid00041/gid00031/gid00028/gid00045/gid00052/gid00014/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid01162/gid00004/gid00032/gid00041/gid00047/gid00032/gid00045/gid00032/gid00031/gid00001/gid00019/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00028/gid00039/gid00001/gid00028/gid00041/gid00031/gid00001/gid00008/gid00045/gid00042/gid00048/gid00041/gid00031/gid00036/gid00041/gid00034
/gid01088/gid00005/gid00048/gid00028/gid00039/gid01162/gid00036/gid00041/gid00031/gid00032/gid00051/gid00001
/gid00004/gid00028/gid00041/gid00031/gid00036/gid00031/gid00028/gid00047/gid00032/gid00001/gid00019/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00028/gid00039/gid01089/gid00014/gid00016/gid01162/gid00030/gid00032/gid00041/gid00047/gid00032/gid00045/gid00032/gid00031
/gid00017/gid00045/gid00042/gid00049/gid00032/gid00041/gid00028/gid00041/gid00030/gid00032/gid00001/gid00006/gid00051/gid00043/gid00028/gid00041/gid00046/gid00036/gid00042/gid00041/gid01090/gid00017/gid00045/gid00042/gid00049/gid00032/gid00041/gid00028/gid00041/gid00030/gid00032/gid01162/gid00028/gid00050/gid00028/gid00045/gid00032
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00017/gid00028/gid00030/gid00038/gid00036/gid00041/gid00034
🔍
/gid00014/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040/gid00001/gid00010/gid00041/gid00031/gid00032/gid00051/gid00001
/gid01777/gid00001/gid00014/gid00016/gid01088/gid00001/gid00001/gid00014/gid00016/gid01089/gid00001/gid00001/gid00014/gid00016/gid01090 /gid01141/gid01141/gid01141
/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001/gid00010/gid00041/gid00031/gid00032/gid00051/gid00001/gid00001
/gid01777/gid00001/gid00006/gid01092/gid00001/gid00001/gid00001/gid00006/gid01096/gid00001/gid00001/gid00001/gid00006/gid01088/gid01087 /gid01141/gid01141/gid01141
/gid00006/gid01088/gid00001/gid00001/gid00006/gid01092/gid00001/gid00001/gid00006/gid01095/gid00001/gid00001/gid00006/gid01096/gid00001/gid00001/gid00006/gid01088/gid01087/gid00001/gid00001/gid00006/gid01088/gid01088
/gid00014/gid00016/gid01088/gid00001/gid01170/gid00001/gid00006/gid01092/gid00001/gid01170/gid00001/gid00006/gid01096/gid00001/gid01170/gid00001/gid00006/gid01088/gid01087/gid00006/gid00049/gid00036/gid00031/gid00032/gid00041/gid00030/gid00032/gid00001
/gid00045/gid00028/gid00041/gid00038/gid00036/gid00041/gid00034
/gid00014
/gid00020
/gid00020/gid00036/gid00034
/gid00020/gid00052/gid00046/gid00014/gid00016/gid01088
/gid00006/gid01088/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00006/gid01092/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00006/gid01096
/gid00006/gid01088/gid01087/gid00008/gid00045/gid00042/gid00048/gid00041/gid00031/gid00036/gid00041/gid00034
/gid00030/gid00042/gid00041/gid00047/gid00032/gid00051/gid00047Figure 4: Mechanism-centered MS3retrieval and grounding. A scientific query is answered either from parametric knowledge
alone or through dual-index retrieval, provenance expansion, and evidence-aware context packing over the MS3substrate,
enabling mechanism-grounded responses with claim-level citations and explicit support boundaries.
In-domain Coverage-shift
Generator Params. Inference Direct Web Raw-PDF Verified-MS3∆V−RWeb Verified-MS3Hybrid∆ H−B
Qwen2.5-14B 14B Local-4bit 37.5 58.380.8 80.80.0 51.9 52.154.2+2.1
Qwen2.5-72B 72B Local-4bit 34.2 60.0 81.786.7+5.0 52.2 54.657.8+3.2
Qwen3-30B-A3B 30B/3B active Local-4bit 23.353.346.7 48.3 +1.7 48.9 51.153.8+2.7
Mistral-Small-24B 24B Local-4bit 25.0 54.2 65.070.0+5.0 53.3 52.155.6+2.3
DeepSeek-V4-Flash Undisc. API 25.8 51.7 80.089.2+9.2 56.2 58.368.8+10.4
DeepSeek-V4-Pro Undisc. API 28.3 46.7 82.590.0+7.5 56.3 64.670.9+6.3
Kimi-K3 Undisc. API 26.7 56.7 82.583.3+0.849.241.7 41.7 -7.5
Kimi-K2.6 Undisc. API 26.7 56.7 80.090.8+10.8 61.3 64.671.3+6.7
GLM-5.2 Undisc. API 29.2 60.8 83.394.2+10.8 63.4 66.768.8+2.1
GLM-5-Turbo Undisc. API 24.2 57.5 84.291.7+7.5 43.5 50.058.3+8.3
Macro average – – 28.1 55.6 76.782.5+5.8 53.6 55.660.1+3.7
Table1:Scientificcorrectness(%).V,R,andHdenoteVerified-MS3,Raw-PDFRAG,andHybridretrieval,respectively.The
Raw-PDF condition is a source-resolved oracle supplied with the benchmark-matched paper, rather than a user-facing retrieval
baseline.∆ V−R =V−RmeasurestheimprovementofVerified-MS3overRaw-PDFRAG.Forthecoverage-shiftevaluation,
B= max(Web, V)denotes the strongest single-source retrieval condition, and∆ H−B =H−Bmeasures the additional
benefit of combining Web and Verified-MS3. Bold indicates the best result within each evaluation setting.

Scientific correctness Citation entailment Completeness
Generator Direct Web Raw-PDF Verified-MS3Web Raw-PDF Verified-MS3Direct Web Raw-PDF Verified-MS3
Qwen2.5-14B 37.5 58.380.8 80.858.387.585.0 20.8 42.5 69.273.3
Qwen2.5-72B 34.2 60.0 81.786.761.7 86.790.020.0 45.0 75.879.2
Qwen3-30B-A3B 23.353.346.7 48.3 49.250.049.2 24.2 48.3 45.849.2
Mistral-Small-24B 25.0 54.2 65.070.050.8 70.075.020.0 40.0 59.264.2
DeepSeek-V4-Flash 25.8 51.7 80.089.268.3 90.897.518.3 36.7 77.587.5
DeepSeek-V4-Pro 28.3 46.7 82.590.062.5 92.597.523.3 34.2 85.093.3
Kimi-K3 26.7 56.7 82.583.356.785.084.2 20.8 50.083.3 83.3
Kimi-K2.6 26.7 56.7 80.090.855.0 83.391.720.8 47.5 80.090.0
GLM-5.2 29.2 60.8 83.394.260.0 88.396.715.8 46.7 82.592.5
GLM-5-Turbo 24.2 57.5 84.291.756.7 89.294.214.2 43.3 75.885.8
Macro average 28.1 55.6 76.782.557.9 82.386.119.8 43.4 73.479.8
Table 2: Per-generator results on the in-domain evaluation (%). Citation entailment is not applicable to Closed-book Direct.
Bold indicates the best method within each metric group for a given generator.
/gid01087
/gid01089/gid01092
/gid01092/gid01087
/gid01094/gid01092
/gid01088/gid01087/gid01087
/gid00020/gid00030/gid00042/gid00045/gid00032/gid00001/gid01175/gid01727/gid01176
/gid00014/gid00020/gid01577/gid00001/gid00047/gid00032/gid00045/gid00040
/gid00014/gid00032/gid00030/gid00035/gid00028/gid00041/gid00036/gid00046/gid00040
/gid00046/gid00048/gid00043/gid00043/gid00042/gid00045/gid00047
/gid00020/gid00032/gid00040/gid00028/gid00041/gid00047/gid00036/gid00030
/gid00033/gid00048/gid00039/gid00039/gid00001/gid00043/gid00028/gid00047/gid00035
/gid00006/gid00514/gid00032/gid00030/gid00047/gid00036/gid00049/gid00032
/gid00044/gid00048/gid00028/gid00039/gid00036/gid00047/gid00052
88%
80%
65%
58%
80%
86%
74%
43%
59%
89%
84%
71%
72%
/gid00003/gid00028/gid00046/gid00032/gid00001/gid00031/gid00036/gid00045/gid00032/gid00030/gid00047
/gid00016/gid00041/gid00032/gid01162/gid00046/gid00035/gid00042/gid00047/gid00001/gid00013/gid00042/gid00019/gid00002
/gid00005/gid00032/gid00030/gid00042/gid00040/gid00043/gid00042/gid00046/gid00032/gid00031
/gid00006/gid01181/gid00021/gid01181/gid00014/gid00001/gid01748/gid00001/gid00049/gid00032/gid00045/gid00036/gid00033/gid00036/gid00032/gid00045
82%
67%
59%
Figure 5: Ablation of the MS3substrate construction
pipeline. Error bars denote 95% confidence intervals.
by 28.2 points and completeness by 36.4 points. This pat-
tern is consistent with the benefit of same-paper evidence
links, functional roles, and explicit evidence boundaries. It
does not establish the contribution of each structural com-
ponent individually; those effects require evidence-matched
ablations.
Inthecoverage-shiftevaluation,Hybridretrievalachieves
a macro-average scientific correctness of 60.1%, exceeding
the strongest single-source condition for each generator by
3.7 points on average. Hybrid improves over the better of
Web Search and Verified-MS3for nine of the ten gener-
ators, with particularly large gains for DeepSeek-V4-Flash
(+10.4 points), GLM-5-Turbo (+8.3 points), and Kimi-K2.6
(+6.7 points). The exception is Kimi-K3, for which Hybrid
underperforms the strongest single-source condition by 7.5
points. These results indicate that Web evidence and MS3
mechanism cards are generally complementary under cov-
erage shift, but that combining them does not automatically
improvegeneration.Thefailurecasesuggeststhatsomegen-
erators may be sensitive to conflicting evidence, increased
contextlength,ortheneedtointegrateinformationwithhet-
erogeneous structures and provenance.AblationStudiesWecomparedirectgeneration,one-shot
LoRA, decomposed extraction, and the complete E/T/M
pipeline with verification. As shown in Figure 5, the full
pipelineachievesthebestsemanticfull-pathaccuracy(74%)
and effective quality (72%), while maintaining strong MS3
term accuracy (88%) and mechanism support (84%). No-
tably, decomposed extraction without verification performs
poorlyonfull-pathreconstruction(43%),suggestingthatthe
verifier is important for correcting inconsistencies and as-
sembling reliable mechanism chains.
The results also show that decomposition alone is insuf-
ficient: although it preserves term extraction performance,
errors accumulate during full-path assembly. Adding verifi-
cationrecovers31pointsinsemanticfull-pathaccuracyand
22 points in effective quality, indicating that validation is
particularly important for relation-level consistency.
LimitationsThe current instantiation is restricted to
conductive-fiber flexible sensors, and schema transfer to
other scientific domains remains unevaluated. The bench-
mark primarily measures evidence use after candidate
sources can be identified, rather than unrestricted litera-
turediscovery. Inaddition,the presentexperimentsevaluate
single-pass MS3retrieval rather than the complete source-
repair workflow.
Conclusion
We presented an evidence-grounded mechanism knowledge
substratethatorganizesscientificliteratureintoprovenance-
linked, role-typed mechanism paths. Instantiated as MS3
for conductive-fiber flexible sensors, the substrate supports
grounded mechanism reconstruction, quantitative lookup,
cross-paper comparison, and evidence-boundary reasoning.
Onthein-domainbenchmark,VerifiedMS3improvesmacro-
averagedscientificcorrectnessby26.9percentagepointsover
Web search and by 5.8 points over a source-resolved Raw-
PDForacle.Undercoverageshift,Hybridretrievalimproves
over the stronger single-source condition by 3.7 points on
average. These findings demonstrate that knowledge repre-
sentation and evidence routing materially affect grounded
scientific question answering.

References
Asai, A.; He, J.; Shao, R.; Shi, W.; Singh, A.; Chang, J. C.;
Lo, K.; Soldaini, L.; Feldman, S.; D’Arcy, M.; Wadden, D.;
Latzke, M.; Sparks, J.; Hwang, J. D.; Kishore, V.; Tian, M.;
Ji, P.; Liu, S.; Tong, H.; Wu, B.; Xiong, Y.; Zettlemoyer, L.;
Neubig,G.;Weld,D.S.;Downey,D.;Yih,W.-t.;Koh,P.W.;
and Hajishirzi, H. 2026. Synthesizing Scientific Literature
with Retrieval-Augmented Language Models.Nature, 650:
857–863.
Auer, S.; Kovtun, V.; Prinz, M.; Kasprzik, A.; Stocker, M.;
and Vidal, M.-E. 2018. Towards a Knowledge Graph for
Science. InProceedingsofthe8thInternationalConference
onWebIntelligence,MiningandSemantics,WIMS’18,1–6.
NewYork,NY,USA:AssociationforComputingMachinery.
Caufield, J. H.; Hegde, H.; Emonet, V.; Harris, N. L.;
Joachimiak, M. P.; Matentzoglu, N.; Kim, H.; Moxon, S.
A. T.; Reese, J. T.; Haendel, M. A.; Robinson, P. N.; and
Mungall, C. J. 2024. Structured Prompt Interrogation and
Recursive Extraction of Semantics (SPIRES): A Method
forPopulatingKnowledgeBasesUsingZero-ShotLearning.
Bioinformatics, 40(3): btae104.
Dasigi, P.; Lo, K.; Beltagy, I.; Cohan, A.; Smith, N. A.;
and Gardner, M. 2021. A Dataset of Information-Seeking
Questions and Answers Anchored in Research Papers. In
Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics:
Human Language Technologies, 4599–4610. Online: Asso-
ciation for Computational Linguistics.
Ding, H.; Zhao, Y.; Hu, T.; Wang, Z.; Patwardhan, M.; and
Cohan, A. 2026. SciRAG: Adaptive, Citation-Aware, and
Outline-GuidedRetrievalandSynthesisforScientificLitera-
ture. InProceedingsofthe19thConferenceoftheEuropean
Chapter of the Association for Computational Linguistics
(Volume1:LongPapers),6440–6460.Rabat,Morocco:As-
sociation for Computational Linguistics.
Edge,D.;Trinh,H.;Cheng,N.;Bradley,J.;Chao,A.;Mody,
A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and Lar-
son, J. 2024. From Local to Global: A Graph RAG Ap-
proach to Query-Focused Summarization.arXiv preprint
arXiv:2404.16130.
He,X.;Tian,Y.;Sun,Y.;Chawla,N.V.;Laurent,T.;LeCun,
Y.; Bresson, X.; and Hooi, B. 2024. G-Retriever: Retrieval-
AugmentedGenerationforTextualGraphUnderstandingand
Question Answering. InAdvances in Neural Information
ProcessingSystems,volume37,132876–132907.CurranAs-
sociates, Inc.
Jain,S.;vanZuylen,M.;Hajishirzi,H.;andBeltagy,I.2020.
SciREX:AChallengeDatasetforDocument-LevelInforma-
tionExtraction.InProceedingsofthe58thAnnualMeetingof
the Association for Computational Linguistics, 7506–7516.
Online: Association for Computational Linguistics.
Jaradeh,M.Y.;Oelen,A.;Farfar,K.E.;Prinz,M.;D’Souza,
J.; Kismihók, G.; Stocker, M.; and Auer, S. 2019. Open
ResearchKnowledgeGraph:NextGenerationInfrastructure
for Semantic Scholarly Knowledge. InProceedings of the
10th International Conference on Knowledge Capture, K-CAP ’19, 243–246. New York, NY, USA: Association for
Computing Machinery.
Lála, J.; O’Donoghue, O.; Shtedritski, A.; Cox, S.; Ro-
driques, S. G.; and White, A. D. 2024. PaperQA: Retrieval-
AugmentedGenerativeAgentforScientificResearch. InThe
Twelfth International Conference on Learning Representa-
tions.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Küttler,H.;Lewis,M.;Yih,W.-t.;Rocktäschel,T.;
Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Gen-
erationforKnowledge-IntensiveNLPTasks. InAdvancesin
Neural Information Processing Systems, volume 33, 9459–
9474. Curran Associates, Inc.
Luan, Y.; He, L.; Ostendorf, M.; and Hajishirzi, H. 2018.
Multi-TaskIdentificationofEntities,Relations,andCorefer-
ence for Scientific Knowledge Graph Construction. InPro-
ceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, 3219–3232. Brussels, Bel-
gium: Association for Computational Linguistics.
Skarlinski, M. D.; Cox, S.; Laurent, J. M.; Braza, J. D.;
Hinks, M.; Hammerling, M. J.; Ponnapati, M.; Rodriques,
S.G.;andWhite,A.D.2024. LanguageAgentsAchieveSu-
perhumanSynthesisofScientificKnowledge.arXivpreprint
arXiv:2409.13740.
Venugopal, V.; and Olivetti, E. 2024. MatKG: An Au-
tonomously Generated Knowledge Graph in Material Sci-
ence.Scientific Data, 11: 217.
Wadden, D.; Lin, S.; Lo, K.; Wang, L. L.; van Zuylen, M.;
Cohan, A.; and Hajishirzi, H. 2020. Fact or Fiction: Verify-
ing Scientific Claims. InProceedings of the 2020 Confer-
enceonEmpiricalMethodsinNaturalLanguageProcessing
(EMNLP), 7534–7550. Online: Association for Computa-
tional Linguistics.
Wadden,D.;Lo,K.;Kuehl,B.;Cohan,A.;Beltagy,I.;Wang,
L.L.;andHajishirzi,H.2022.SciFact-Open:TowardsOpen-
Domain Scientific Claim Verification. InFindings of the
Association for Computational Linguistics: EMNLP 2022,
4719–4734. Abu Dhabi, United Arab Emirates: Association
for Computational Linguistics.
Wang, C.; Shen, Y.; Kuang, Z.; Cohan, A.; and Zhao, Y.
2025. SciVer: Evaluating Foundation Models for Multi-
modal Scientific Claim Verification. InProceedings of the
63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), 8562–8579. Vienna,
Austria: Association for Computational Linguistics.