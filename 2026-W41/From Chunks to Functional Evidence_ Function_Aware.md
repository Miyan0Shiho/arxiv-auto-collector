# From Chunks to Functional Evidence: Function-Aware Retrieval for EDA Documentation QA

**Authors**: Xiaotian Qiu, Kairui Liu, Shi Chenyi, Jinyuan Deng, Qi Sun, Cheng Zhuo

**Published**: 2026-10-07 03:16:06

**PDF URL**: [https://arxiv.org/pdf/2610.09361v1](https://arxiv.org/pdf/2610.09361v1)

## Abstract
Retrieval-Augmented Generation (RAG) is widely used to ground answers in documents. For complex technical documentation, however, the primary bottleneck is often not model reasoning but a mismatch between a query and the way knowledge is organized for retrieval. This mismatch is pronounced in Electronic Design Automation (EDA) documentation, where the information needed for an answer is scattered across heterogeneous yet tightly coupled artifacts. We therefore redesign the basic retrieval unit of RAG. Instead of operating on isolated chunks or binary relations, we collect typed artifacts into EDA functional units. Each unit is recorded as a hyperedge with links to its source chunks. We then train an encoder to align queries with functional units and combine unit retrieval with direct chunk retrieval. After mapping the selected units back to their sources, a unified reranker chooses the evidence given to the generator. On the newly constructed EDADocEval-QA dataset, our method improves ROUGE-L by 37.1% over Chunk RAG and 55.6% over the strongest graph baseline. On the public ORD-MMBench benchmark, it improves ROUGE-L by 30.0% over the strongest baseline. These results support function-aware evidence organization in the evaluated EDA documentation settings.

## Full Text


<!-- PDF content starts -->

From Chunks to Functional Evidence: Function-Aware Retrieval for EDA
Documentation QA
Xiaotian Qiu1,2, Kairui Liu1, Shi Chenyi1, Jinyuan Deng1, Qi Sun1, Cheng Zhuo1
1Zhejiang University
2Shanghai Innovation Institute
Abstract
Retrieval-Augmented Generation (RAG) is widely used to
ground answers in documents. For complex technical docu-
mentation,however,theprimarybottleneckisoftennotmodel
reasoningbutamismatchbetweenaqueryandthewayknowl-
edgeisorganizedforretrieval.Thismismatchispronouncedin
Electronic Design Automation (EDA) documentation, where
the information needed for an answer is scattered across het-
erogeneousyettightlycoupledartifacts.Wethereforeredesign
the basic retrieval unit of RAG. Instead of operating on iso-
lated chunks or binary relations, we collect typed artifacts
into EDA functional units. Each unit is recorded as a hyper-
edgewithlinkstoitssourcechunks.Wethentrainanencoder
to align queries with functional units and combine unit re-
trievalwithdirectchunkretrieval.Aftermappingtheselected
units back to their sources, a unified reranker chooses the
evidence given to the generator. On the newly constructed
EDADocEval-QA dataset, our method improves ROUGE-L
by 37.1% over Chunk RAG and 55.6% over the strongest
graph baseline. On the public ORD-MMBench benchmark,
it improves ROUGE-L by 30.0% over the strongest baseline.
These results support function-aware evidence organization
in the evaluated EDA documentation settings.
1 Introduction
ElectronicDesignAutomation(EDA)toolssupportthecom-
plete circuit design flow through tightly coupled imple-
mentation stages (Ajayi et al. 2019). Their documentation
exemplifies a class of technical sources with intrinsically
complex knowledge organization. Information is distributed
across code blocks, parameter tables, figures, procedures,
constraints, and descriptive text that may appear in different
sectionsbutjointlyspecifyonetoolfunction.Werefertothis
mixtureofheterogeneousevidenceformsasmulti-typedoc-
umentation. Answering a practical question often requires
composing several such artifacts rather than retrieving one
isolated passage, as illustrated in Figure 1.
RAG is a natural way to ground answers in these docu-
ments, but its primary bottleneck here is often not model
reasoning. It is the mismatch between a question and the
way evidence is organized for retrieval. Conventional dense
retrievers treat each chunk as an independent unit and as-
sumethatvectorsimilarityseparatesrelevantfromirrelevant
passages. In EDA documentation, topically similar chunks
may instead describe different commands, flow stages, or
Figure 1: An illustrative embedding diagnostic: top-ranked
EDA chunks have more tightly clustered similarity scores
than the general-domain example, motivating retrieval units
that preserve complementary functional roles.
operating conditions. Figure 1 illustrates how repeated do-
main terminology compresses the score differences among
top-rankedchunks.Theretrievableunitshouldthereforepre-
servewhichartifactsparticipateinthesamefunction,aswell
as where each artifact came from.
This challenge calls for rethinking how knowledge is
organized in RAG (Lewis et al. 2020). Chunk RAG re-
trieves independently encoded passages (Karpukhin et al.
2020). When dependent evidence appears in different sec-
tions,thispassage-levelorganizationdoesnotpreservetheir
functionalgrouping.GraphRAGorganizesentitiesandcom-
munitysummariesforglobalsensemaking(Edgeetal.2024),
while HippoRAG uses graph traversal for multi-hop re-
trieval (Gutiérrez et al. 2024). Neither necessarily retains
the source context of a higher-order EDA usage event. Hy-
pergraphs provide a convenient representation in which one
unit can connect several artifacts (Feng et al. 2019), but the
representation alone does not decide which artifacts belong
together.Genericnamerepetitionorlocalco-occurrencemay
stillconnectevidencethatistopicallysimilaryetfunctionally
incompatible.
Toaddresstheseissues,weredesignthebasicretrievalunit
around EDA functionality. Our framework organizes typed
artifacts into functional units, uses a hypergraph to retain
their higher-order membership and source provenance, and
arXiv:2610.09361v1  [cs.IR]  7 Oct 2026

retrievestheoriginalevidencethatsupportsaquestion.Such
groundeddocumentationQAalsoprovidesanevidencelayer
for EDA assistants that must select commands, set parame-
ters,andinterpretwarningsbeforeacting.Ourcontributions
are summarized as follows:
•We characterize the mismatch between conventional re-
trievalunitsandEDAdocumentation:artifactsneededfor
oneanswerarefunctionallycoupledbutcanbeseparated
by chunking or flattened by binary relations.
•WeintroduceEDAfunctionalunitsasretrievableknowl-
edgeunits.Occurrencesremainlocaltotheirsourcesbe-
foretheyaregroupedaroundafunctionalanchor.Hyper-
edges preserve their types and provenance and support
conservative connections across documents.
•Weproposequery–unitcontrastivealignmentthatdistin-
guishessemanticallysimilarunitsaccordingtotheartifact
types available to answer a query.
•We develop a function-aware hybrid retriever that com-
bines functional unit retrieval with direct chunk retrieval
and returns a unified evidence context to the generator.
We evaluate our method against representative baselines
on the public ORD-MMBench benchmark (Pu et al. 2025)
and on EDADocEval-QA, a newly constructed domain-
specificdataset.EDADocEval-QAcoversparameterlookup,
syntaxretrieval,proceduresynthesis,andconstraintanalysis.
Our method obtains 0.473 ROUGE-L and 93.6 LLM-Score
onthisdataset,whilethepublicbenchmarkprovidesacom-
plementaryevaluationofgenerationanddocumentretrieval.
2 Related Work
LLMs and RAG in EDA
LLMs are widely used in EDA, demonstrating immense po-
tential in tasks such as code generation (Liu et al. 2023a;
Thakur et al. 2024; Lu et al. 2024), EDA scripting (Wu
etal.2024),CPUdesignandverification(Wangetal.2024),
and conversational hardware design (Blocklove et al. 2023;
Changet al.2023).Leadingindustrial efforts,suchasChip-
NeMo (Liu et al. 2023b), have further validated the feasi-
bility of domain-adaptive LLMs. However, EDA documen-
tation presents significant challenges for general LLMs due
toproprietaryprotocols,complexparameterconstraints,and
specific terminologies. LLMs often generate hallucinations
oroutdatedinformationduetothelackofdomainknowledge.
RAG addresses this issue by retrieving factual information
from external databases (Guu et al. 2020). Recent studies
have optimized RAG for EDA by fine-tuning embedding
models on domain corpora (Pu et al. 2024). Despite these
improvements,mostsystemsstillorganizeevidenceasinde-
pendentchunksorgenericrelationsratherthanexplicitEDA
functional roles.
Evidence Organization in RAG
AkeycharacteristicofEDAdocumentsisthetightcoupling
ofmulti-typeentities.ExistingRAGparadigmsmakediffer-
ent trade-offs when handling this characteristic.Chunk-based RAGtreats documents as linear text se-
quences, utilizing sparse retrieval (e.g., BM25 (Robert-
son and Zaragoza 2009)) or dense retrieval (e.g.,
DPR (Karpukhin et al. 2020)) to fetch segments. This ap-
proachcanseparatecross-entitydependencies.Afunctional
definitionoftenappearsinonesection,whileitstimingcon-
straints appear in another. Chunk-based retrieval can there-
fore return isolated fragments. Consequently, the LLM may
receive an incomplete evidence chain (Xing et al. 2026).
Graph-based RAGextracts entities and relations to con-
struct knowledge graphs (Pan et al. 2024). GraphRAG uses
entity graphs and community summaries for global sense-
making (Edge et al. 2024), while HippoRAG uses knowl-
edge graphs and Personalized PageRank for multi-hop re-
trieval (Gutiérrez et al. 2024). In the EDA setting studied
here,however,entity-orrelation-levelretrievalmaystillsep-
arate a higher-order usage event in which code blocks, pa-
rameter tables, and procedural descriptions jointly support
an answer. Compressing detailed procedural descriptions or
code blocks into relational summaries can also omit source
context needed for generation.
Structured Evidence Organization
Recent retrieval methods increasingly organize evidence
abovethelevelofindependentchunks.SG-Filterbuildsahi-
erarchical summarized-semantic index to distinguish other-
wisesimilarpassages(Yeetal.2025).EvidenceUnitsgroups
visual assets with their contextual text through ontology-
grounded document organization (Han 2026), while Tech-
DocRAG preserves relations among clauses, tables, figures,
and procedural steps in technical documents (Lee and Choi
2026). Hypergraphs offer another representation: a single
hyperedge can connect multiple artifacts (Feng et al. 2019;
Yadati et al. 2019), and recent systems use this property to
organize higher-order evidence for multi-hop retrieval (Luo
etal.2025;ZhangandMa2026;Anetal.2026).Ourframe-
workusesthisrepresentationtoretainsource-localizedEDA
occurrences,butdefinesretrievalunitsaroundtoolfunctions
and learns to match queries with the artifact roles available
in each unit.
Reasoning-Augmented RAG
Reasoning-enhanced frameworks such as Chain-of-Thought
(CoT) (Wei et al. 2022) and Tree of Thoughts (ToT) (Yao
etal.2023a)improveLLMreasoningbydecomposingprob-
lems into intermediate steps. IRCoT (Trivedi et al. 2023)
interleaves retrieval with reasoning, while ReAct (Yao et al.
2023b) interleaves reasoning with external actions such as
search.Theirefficacystilldependsontheevidenceretrieved
ateachstep.Usinggenericvector-basedRAGorwebsearch
APIs proves ineffective for the specialized and proprietary
nature of EDA documentation. Our framework is comple-
mentary: it encodes typed documentation dependencies in
the retrieval index so that a reasoning method can begin
from retrieved units with intact source context.

Figure 2: Framework overview. (a) Typed, source-localized EDA artifacts are organized into functional units, each represented
as a hyperedge linked to its source chunks. (b) A function-aware hybrid retriever searches functional units and source chunks,
after which a unified reranker selects the generation evidence. (c) Query–unit contrastive alignment trains the function-aware
encoder.
3 Method
Problem Formulation and Overview
We formulate EDA documentation QA as an evidence re-
trievalproblem.GivenaqueryqanddocumentchunksC,the
system selects a generation contextC q⊆ Cthat covers the
facts and artifact types needed for the answer. Fixed chunks
oftenseparateevidencethatbelongstothesameEDAopera-
tion,whilebinaryrelationsdivideaneventinvolvingseveral
complementaryroles.Beforereceivingaquery,weorganize
the corpus into source chunks and EDA functional units. A
unit groups compatible occurrences around one canonical
functionwhileretainingthesourcelocationofeveryartifact.
We represent it as a typed hyperedge linked to its source
chunks.
Our framework has three components. First, EDA func-
tional unit organization groups artifacts that play comple-
mentaryrolesinthesameoperation.Second,query–unitcon-
trastivealignmentlearnswhichartifactcombinationsdistin-
guishonefunctionfromnearbyalternatives.Third,function-
awareevidenceretrievalsearchestheunitandchunkindexes
in parallel. Once selected for a query, the returned units and
chunks become candidate evidence. Selected units are ex-
pandedtotheirsourcechunks,andaunifiedrerankerchooses
the top-Kchunks asC q.
EDA Functional Unit Organization
Wefirstidentifytypedartifactsandorganizethemintofunc-
tionalunitsbeforeanyqueryisreceived.Theresultingindex
is encoded as a hypergraphH= (V,E hyper): a node rep-
resents a documentation artifact, and a hyperedge records
the artifacts and source chunks that constitute one unit. Thehypergraph is therefore the representation of the index; the
substantive construction decision is which artifacts are al-
lowed to form a functional unit.
Occurrences within a source.Construction begins from
documentstructureratherthanawhole-corpuslexicalgraph.
A deterministic Markdown parser records headings, block
boundaries, line spans, and image references. Blocks under
the same section are grouped locally, with code blocks, or-
deredprocedures,anddiagnosticmessagesservingasnatural
boundaries between distinct usage events. Very short neigh-
boringsegmentsaremergedonlywithinthesamedocument.
This produces occurrence candidates whose provenance is
known before semantic extraction.
Each candidate is then decomposed into typed artifacts.
OurschemaTcoverscode,inlinecommands,tables,figures,
procedures,descriptivetext,constraints,diagnostics,andfile
artifacts.Deterministicrulesidentifyexplicitstructuressuch
ascodeblocksandtables,whileanLLMidentifiessemantic
spans such as procedures and constraints. For every artifact
v,weretainitscontent,artifacttypeτ(v),sourcedocument,
source chunk, and extraction metadata.
Artifact type and functional role are related but not in-
terchangeable.τ(v)describes the evidence form, such as
tableorcodeblock. A functional role describes how
that artifact participates in a particular usage event, for ex-
ample as the operation, parameter specification, condition,
example, or diagnostic consequence. We infer this relation
from the surrounding source context and its functional an-
chor;itisnotasecondsupervisednodelabel.Thisdistinction
matters later because the training objective operates on ob-
servable artifact types, not on latent role names.

Formally,afunctionaloccurrencefromsourceregionris
or= (a r,Vr,Cr, pr),(1)
wherea risacommandorfileanchor,V rcontainsthetyped
artifacts in that local event,C rcontains its source chunks,
andp rstoresdocumentandlineprovenance.Thisdefinition
is stronger than local word co-occurrence: the grouped arti-
factsmustbelongtoonestructuralsourceregionandremain
traceable to it.
Functional units across sources.An occurrence within
one source already forms a useful unit. Some functions,
however, are described across manuals: one document may
define a command while another provides a parameter ta-
ble or worked example. We connect such occurrences only
through an admissible canonical anchor. LetO(a) ={o r:
norm(a r) =a}. A unit spanning documents is created only
whenO(a)contains at least two source occurrences from
different documents and their union contains at least two ar-
tifact types. The linker accepts stable command and file an-
chors,filtersgenericplaceholdersandlanguagetypealiases,
and caps the number of occurrences attached to one anchor.
Theserestrictionspreventfrequentbutirrelevantstringsfrom
becoming hubs across the corpus.
Eachretainedunitisrepresentedasahyperedgee∈ E hyper:
e= (V(e), S(e),chunks(e), P(e)),(2)
whereV(e)⊆ Vis the artifact set,S(e)is its retrieval sum-
mary,chunks(e)⊆ Cis the deduplicated set of original
chunks, andP(e)retains the contributing occurrences and
source references. We serializeS(e)by artifact type so that
the unit embedding retainsT(e) ={τ(v)|v∈ V(e)}. The
summaryisusedforunitretrievalonly;thegeneratorreceives
the original source chunks reached throughchunks(e).
Validation and updates.Structural checks reject empty
units,missingsourcereferences,invalidtypelabels,andlinks
withoutsufficienttypediversity.Theyestablishstructuralva-
lidity and provenance, not the semantic accuracy of LLM
extraction. Construction can resume from its cache, so vali-
dated outputs survive an interrupted call. When a document
is added or changed, the parser and extractor run only on
itschunks.Thelinkerthenresolvestheiranchorsagainstthe
retained index and recomputes the affected units; a deletion
follows the same provenance links in reverse. Consequently,
adding the 33rd document does not require extracting the
unchanged 32-document corpus again.
Concrete example.Consider thedefine_pdn_grid
command in OpenROAD. Its default-starts_with:
GROUND, syntax, and strap-orientation constraints occur in
different source regions. The canonical command anchor
connectstheparametertable,codeblock,andconstraintpara-
graph, while their source IDs remain intact. The resulting
functionalunitisrepresentedbyahyperedgewithtypecom-
position{table,codeblock,text_paragraph}and
backtracks to all three chunks.
Query–Unit Contrastive Alignment
Effective retrieval requires more than locating semantically
similartext.Itmustalsoidentifyfunctionalunitswhoseavail-able artifact types can support the query. Since EDA docu-
ments tend to cluster more tightly in the embedding space,
relyingsolelyonsemanticsimilarityoftenfailstodistinguish
among candidates, resulting in recalls that appear seman-
tically relevant but lack the required evidence forms. Our
contrastive objective aligns queries with units that contain
the required combination of types while preserving general
semantic similarity.
We use the structured summaryS(e)as the textual inter-
faceforencodingeachfunctionalunit.Forunitsthatinclude
visual or structured entities, we serialize them into the mul-
timodalinputformatsothattheunitembeddingreflectsboth
semantic content and artifact type. This shifts supervision
from local chunk matching to query–unit alignment.
Trainingqueriesareconstructedfromfunctionalunits,so
eachqueryqanditsreferenceunitsetP qsharethetargettype
combinationbyconstruction.Welearnthealignmentthrough
contrastive positive and negative pairs together with an ar-
tifact coverage ranking objective. For each query, we select
oneunitfromP qasthepositivesamplee+andsampleN(q)
from the remaining units. Hard negatives are semantically
plausible but lack a critical artifact type; random negatives
preserve general semantic discrimination.
Hard negatives capture a common failure mode in multi-
type documentation: candidates can be topically related yet
miss evidence needed for an answer. For instance, a purely
descriptive paragraph may be semantically relevant to a pa-
rameter query but lacks the concrete table row that specifies
defaults or ranges. By contrasting such candidates against
thepositiveunit,themodellearnstopreferusabletypecom-
binations within a narrow semantic neighborhood. We use
a mixed sampler: units with partial overlap provide difficult
typenegatives,whilesemanticallyhardorrandomunitskeep
the learned space from collapsing into a type classifier. The
query is always textual, whereas a candidate unit may com-
bine textual and visual content. This asymmetry teaches the
queryrepresentationtoselecttheevidenceformsavailablein
a unit without requiring a separate type router at inference.
We use pretrained Visualized-BGE-M3 as theBase Se-
mantic Encoder. Applying Low-Rank Adaptation (LoRA)
with the objective below produces theFunction-Aware En-
coder. Both encoders accommodate textual and visual enti-
ties in multimodal EDA documents. The semantic compo-
nent follows the InfoNCE formulation:
LNCE(q) =−logexp(s+/τ)
exp(s+/τ) +P
e−∈Nexp(s−/τ)(3)
wheres+= sim(q, e+),s−= sim(q, e−), andτis the
temperature. LetT qdenote the artifact types required by
trainingqueryqandT(e)thetypespresentincandidateunit
e. We defineκ q(e) =|T q∩ T(e)|/|T q|, withκ q(e) = 1
whenT qis empty. For the positive and sampled negative
candidates, lets i= sim(q, e i)/τandR q={(i, j) :i <
j,|κ q(ei)−κ q(ej)|>0.01}. For each(i, j)∈ R q, write
∆κ
ij=κq(ei)−κ q(ej)and∆s
ij=si−sj.Apairwisemargin
term then orders candidates with different artifact coverage:
Ltype=1
|Rq|X
(i,j)∈R qmax 
0, m−sgn(∆κ
ij)∆s
ij
.(4)

We setL type= 0whenR qis empty and use marginm=
0.1;thefinalobjectiveisL=L NCE+λL type.Atinference
time,themodeldoesnotexplicitlypredictatypeset.Instead,
the learned embedding space recalls functional units whose
artifact composition matches the query.
Function-Aware Evidence Retrieval
The function-aware hybrid retriever uses two complemen-
tary indices. The unit index captures the evidence forms as-
sociatedwithanEDAoperation.Thechunkindexpreserves
accesstolocalizeddetails.Searchingbothisimportant:func-
tional units provide broader coverage, whereas direct chunk
retrieval can recover a precise detail that is weakly repre-
sented in a unit summary.
Givenq, we retrieve the top-k pfunctional units using
the Function-Aware Encoder and expand each unit through
chunks(e). Expansion is limited to the source chunks that
belong to the selected units; it does not traverse unrelated
neighbors. In parallel, the Base Semantic Encoder retrieves
the top-k csource chunks. At this point, the retrieved units
andchunksbecomecandidateevidence.Wemergeanddedu-
plicate their source chunks to obtainC cand.
Theunitpathisnotanunrestrictedgraphwalk.Theexplicit
chunks(e)mappingproducesacandidatesetconditionedon
thefunctionretrievedforthequery.Thispreventsaunitwith
ahighscorefrompullinginunrelatedneighborsthatmerely
shareagenericentity.Thechunkpathplaysthecomplemen-
taryrole:itcanrecoveralocalcommandoption,tablerow,or
warning that is highly relevant even when its unit summary
is broad. Their union separates two decisions that a single
retriever would conflate: locating the relevant function and
locatingtheprecisesourcespanwithinornearthatfunction.
The unified evidence reranker scores every candidate in
both encoder spaces:
s(c) =αsim(q base, cbase) + (1−α) sim(q func, cfunc)(5)
whereα= 0.4andeachquery–chunksimilarityiscomputed
inoneinternallyalignedencoderspace.Weselectthetop-K
chunks as the generation contextC q.
The Function-Aware Encoder carries the type preference
learned in Section 3, while the Base Semantic Encoder pre-
serveslocalsemanticmatching.Theirfusiondeterminesthe
source chunks seen by the generator rather than using unit
retrieval only as an auxiliary signal. Because both paths ter-
minateatsourcechunks,generationisgroundedinthesame
evidenceformatusedbyconventionalRAG.Functionalunits
changehowcandidateevidenceisorganizedandselected,not
what is presented to the generator.
4 Experiments
Experimental Setup
Evaluation settings.We use two complementary EDA
documentation QA settings. (1)EDADocEval-QAcontains
100 QA pairs over 33 OpenROAD technical documents
(324K tokens), with 25 questions in each of four categories.
The four query types have distinct evidence compositions:
Parameter Lookup (PL), Syntax Retrieval (SR), ProcedureSynthesis (PS), and Debug & Constraint (DC). These cate-
gories require different combinations of structured and tex-
tualevidence.(2)ORD-MMBench(Puetal.2025)isapub-
lic multi-modal EDA benchmark with text and screenshot
queries over EDA documentation; we report overall gener-
ation quality and official document-retrieval performance.
ORD-MMBench provides an independent public evaluation
within the broader EDA documentation setting.
Baselines.We consider three families of baselines that
represent mainstream RAG paradigms.Chunk-level RAG
includes Chunk RAG (Lewis et al. 2020), which per-
forms dense retrieval over fixed-length document chunks.
Graph-basedRAGincludesGraphRAG(Edgeetal.2024),
which organizes evidence with an entity-relation graph and
community-based summarization, LightRAG (Guo et al.
2025), which targets lightweight graph maintenance, Hip-
poRAG 2 (Gutiérrez et al. 2025), which incorporates
memory-augmented retrieval, and HyperGraphRAG (Luo
et al. 2025), which extends graph retrieval to multi-ary
relations.Reasoning-augmented frameworksinclude Re-
Act(Yaoetal.2023b),whichinterleavesreasoningwithtool
use,andSearch-o1(Lietal.2025),whichaugmentsreason-
ing with agentic search. For ORD-MMBench, we use the
benchmark’s reported baselines: VisRAG (Yu et al. 2025),
EchoSight (Yan and Xie 2024), RAG-EDA (Pu et al. 2024),
and VISTA-RAG (Zhou et al. 2024), which span text-based
and multimodal retrieval.
Implementation.We use Qwen2.5-72B as the LLM
backbone for answer generation. The retrieval encoder is
Visualized-BGE-M3 (873M parameters) fine-tuned with
LoRA (rank=8,α=16, dropout=0.1) on a single NVIDIA
RTX4090GPU,yielding786Ktrainableparameters(0.09%
of total). Training data consists of 1,250 query–unit pairs
automatically generated from 250 training packs, with an
85/15 train/validation split. The contrastive objective com-
bines InfoNCE with a type-aware margin ranking loss
weighted byλ=0.3(temperatureτ=0.07, 7 negatives per
query using mixed hard-negative mining). Training runs for
10 epochs with AdamW (lr=10−4) and early stopping; the
best checkpoint is selected at epoch 8 (val loss = 0.735).
For functional-unit retrieval, we set top-M=10; for chunk
retrieval,top-N=5;thefinalcontextusestop-K=5afteruni-
fied reranking. Unless a row is explicitly labeled DeepSeek-
R1,EDADocEval-QAsystemsuseQwen2.5-72Bforanswer
generation; all systems use the same 33-document corpus
and QA/evaluation protocol, while retaining their method-
specificevidenceorganization.ForORD-MMBench,weuse
GPT-4oasthegenerationbackbonetoalignwiththeoriginal
benchmark configuration.
Metrics.WereportBLEU(Papinenietal.2002),ROUGE-
L (Lin 2004), and a 0–100 LLM-Score that sums five 0–20
ratingsforcorrectness,completeness,relevance,profession-
alism,andconcisenessagainstthereferenceanswer.Because
Qwen2.5-72B is both a generator and the LLM-Score judge
forseveralsystems,thismetriciscomplementaryratherthan
independentfactualityevidence;wethereforeinterpretitto-
gether with lexical and retrieval metrics.

Main Results
Table 1 reports results on EDADocEval-QA under a unified
RAGsetting. Wenext analyzecomponent contributionsand
construction cost before evaluating generation and retrieval
on the public ORD-MMBench benchmark.
As shown in Table 1, our method achieves ROUGE-L
0.473,BLEU0.189,andLLM-Score93.6onEDADocEval-
QA,rankingfirstonallthreemetrics.ComparedwithChunk
RAG using Qwen2.5-72B, it improves ROUGE-L by 0.128
(0.345→0.473),BLEUby0.137(0.052→0.189),andLLM-
Scoreby5.3points(88.3→93.6).Thesimultaneousincrease
inlexicalmetricsandLLM-Scoreisconsistentwithimproved
alignmenttoanswer-criticaldetails,althoughtheLLM-based
score is not independent factuality evidence.
Reasoning-augmented methods provide a strong base-
line: ReAct achieves ROUGE-L 0.374, outperforming most
retrieval-only baselines, yet still trails our method by 0.099
in ROUGE-L and 0.098 in BLEU. This pattern is consis-
tent with evidence organization remaining important when
support is fragmented across heterogeneous artifact types;
ourfunctionalunitsexplicitlytargetcompletenessacrossev-
idence types.
We also observe metric divergence where some methods
obtainhighLLM-Scorebutlowlexicalmetrics.Forexample,
LightRAGreachesLLM-Score92.8butonly0.219ROUGE-
L and 0.024 BLEU. The metrics therefore capture different
properties of the generated answers; without answer-level
analysis,theirdisagreementdoesnotidentifyasinglecause.
Our method improves lexical metrics together with LLM-
Score, which is consistent with retrieving both complemen-
tary evidence and answer details.
Ablation Study
We ablate functional unit organization, hybrid retrieval, and
query–unit alignment on EDADocEval-QA. As shown in
Table2,organizingthecorpusintofunctionalunitsprovides
the largest gain, improving ROUGE-L from 0.345 to 0.426
(an increase of 0.081, or 23%). This indicates that grouping
artifacts by their functional roles mitigates evidence frag-
mentation. This is consistent with EDA queries requiring
parameters, code snippets, and procedural descriptions that
participate in the same function but occur in different docu-
ment regions.
The function-aware hybrid retriever further raises
ROUGE-L to 0.461, an increase of 0.035 over functional
unit retrieval alone. This result indicates that source chunks
complementfunctionalunitsinthissetup.Query–unitalign-
ment adds 0.012, reaching 0.473. Since this final increment
is small and no confidence interval is available, we do not
interpret it as a statistically established standalone gain.
Construction and Retrieval Cost
We measured hypergraph construction on the full 33-
document corpus using Qwen3.6-27B. The pipeline pro-
cessed 404 chunks and extracted 3,650 typed nodes. Every
chunkretaineditssourcelocationandcachedmetadata,none
required fallback processing, and all structural consistencychecks passed. We did not manually label extraction accu-
racy, so this result verifies index completeness and internal
consistencyratherthansemanticcorrectness.Buildingthein-
dexrequired427,824LLMtokensand1,668.0seconds,with
peak RSS of 84.0 MiB. At query time, the function-aware
hybridretrieverreducedtheaveragegenerationcontextfrom
1,697 to 1,415 tokens, a reduction of 16.6%, while retrieval
latency increased from 20.69 to 27.67 ms.
Incremental updates and parser robustness.To test
whether the index can absorb new material without rebuild-
ingthefullcorpus,wefirstconstructeditfrom32documents
andthenaddedtheremainingdocument.Theupdaterreused
402 existing units and rebuilt two. All seven semantic out-
puts matched those produced by rebuilding the index from
all 33 documents. We also applied four formatting changes
to the source Markdown, including alternative code fences
and repeated blank lines. Every variant produced the same
semanticgraph,containing404hyperedgesand2,481canon-
icalnodes,coveredall33documents,andretainednoinvalid
units. These experiments test index maintenance and parser
stability, not the semantic precision or recall of extracted
entities.
ORD-MMBench Results
WeevaluateonthepublicORD-MMBenchbenchmarkusing
GPT-4oasthegenerationbackbonetoalignwiththeoriginal
configuration.
AsshowninTable3,ourmethodachievesthebestoverall
performance, with an LLM-Score of 79.6, BLEU of 0.240,
andROUGE-Lof0.412.Comparedwiththestrongestbase-
line VISTA-RAG, it improves overall ROUGE-L by 0.095
(0.317→0.412),BLEUby0.060(0.180→0.240),andLLM-
Score by 10.0 points (69.6→79.6).
Document retrieval reaches Recall@1/5 of 0.700/0.917
versus0.679/0.896forMM-GRADE,providinganindepen-
dent public retrieval comparison alongside the generation
results.
ORD-MMBench assigns one target document to each
query, so its official retrieval metric is document-level Re-
call@K. We compare BGE-M3 (Chen et al. 2024), RAG-
EDA (Pu et al. 2024), DEDR (Salemi, Pizzorno, and Za-
mani2023),BLIP(Lietal.2022),VisRAG(Yuetal.2025),
EchoSight (Yan and Xie 2024), Visualized-BGE-M3 (Zhou
et al. 2024), and MM-GRADE (Pu et al. 2025).
As detailed in Table 4, the retrieval advantage is con-
sistent but modest at every cutoff: the margin over MM-
GRADE ranges from 0.015 to 0.022. Text-oriented retriev-
ers remain substantially stronger than vision-only retrieval
on this benchmark. These results support evidence cover-
age as one contributor to the generation gains, rather than
attributing the full difference to the hypergraph alone.
Discussion
Table 1 compares complete systems under a shared corpus,
question set, and evaluation protocol; the DeepSeek-R1 row
is the only explicitly labeled generator change. Since re-
triever adaptation differs across baseline families, this table
measures end-to-end performance rather than the isolated

Method ROUGE-L↑BLEU↑LLM-Score↑
Chunk-level RAG
Chunk RAG (Qwen2.5-72B) (Lewis et al. 2020) 0.345 0.052 88.3
Chunk RAG (DeepSeek-R1) (Lewis et al. 2020) 0.327 0.097 87.5
Graph-based RAG
GraphRAG (Edge et al. 2024) 0.226 0.050 87.8
LightRAG (Guo et al. 2025) 0.219 0.024 92.8
HyperGraphRAG (Luo et al. 2025) 0.304 0.094 77.2
HippoRAG 2 (Gutiérrez et al. 2025) 0.301 0.168 88.0
Reasoning framework
ReAct (Yao et al. 2023b) 0.374 0.091 84.7
Search-o1 (Li et al. 2025) 0.283 0.044 89.9
Ours 0.473 0.189 93.6
Table 1: Main results on EDADocEval-QA. LLM-Score is a five-part quality score (0–100) evaluated by Qwen2.5-72B. Best
results are in bold.
Configuration ROUGE-L∆
Chunk RAG 0.345 –
+ Functional unit organization 0.426 +23%
+ Function-aware hybrid retrieval 0.461 +34%
+ Query–unit alignment0.473 +37%
Table 2: Cumulative ablation of the three method compo-
nents.
Method LLM-Score BLEU ROUGE-L
VisRAG 20.4 0.084 0.225
EchoSight 57.9 0.139 0.276
RAG-EDA 49.1 0.134 0.268
VISTA-RAG 69.6 0.180 0.317
Ours 79.6 0.240 0.412
Table 3: Overall ORD-MMBench generation results. Base-
line values are from MM-GRADE (Pu et al. 2025).
effect of hypergraph structure. The cumulative ablation of-
fers a more direct comparison within our framework, while
ORD-MMBench tests transfer to a public EDA benchmark.
These results do not establish generality beyond technical
EDA documentation.
The method also introduces a practical trade-off. Retriev-
ing functional units alongside source chunks reduces the
context sent to the generator but adds retrieval latency. Di-
rect chunk retrieval remains useful when one local passage
is sufficient or construction misses an occurrence. The re-
ported results measure answer quality, document retrieval,
construction consistency, and runtime cost, but not the se-
manticaccuracyofintermediateextraction.Eachtableentry
comes from one completed evaluation; we report neither
multi-seed averages nor statistical significance.Method R@1 R@2 R@3 R@4 R@5
BGE-M3 0.590 0.694 0.731 0.761 0.776
RAG-EDA 0.530 0.612 0.679 0.724 0.739
DEDR 0.112 0.187 0.231 0.284 0.299
BLIP 0.127 0.209 0.276 0.299 0.313
VisRAG 0.025 0.042 0.050 0.083 0.108
EchoSight 0.383 0.492 0.583 0.625 0.667
Visualized-BGE-M3 0.582 0.716 0.739 0.761 0.799
MM-GRADE 0.679 0.828 0.843 0.866 0.896
Ours 0.700 0.850 0.858 0.883 0.917
Table 4: Document retrieval on ORD-MMBench. Baseline
values are from MM-GRADE (Pu et al. 2025).
5 Conclusion
WepresentedaframeworkthatorganizestypedEDAartifacts
into functional units and uses query–unit alignment with
hybrid retrieval to select evidence from functional units and
sourcechunks.ItimprovesEDADocEval-QAROUGE-Lby
37%overChunkRAGandtransferstoORD-MMBench.The
studyremainslimitedtoonedomain-specificdatasetandone
public EDA benchmark; construction requires offline LLM
extraction,andhybridretrievalreducesgenerationcontextat
the cost of latency. Future work should evaluate extraction
accuracy and additional EDA tool ecosystems.
References
Ajayi,T.;Blaauw,D.;Chan,T.-B.;Cheng,C.-K.;Chhabria,
V. A.; Choo, D.-K.; Coltella, M.; Dobre, S.; Dreslinski, R.;
Fogaça,M.;etal.2019. OpenROAD:TowardaSelf-Driving,
Open-SourceDigitalLayoutImplementationToolChain. In
Proceedings of Government Microcircuit Applications and
Critical Technology Conference, 1105–1110.
An, H.-Y.; Zhang, Y.-J.; Liang, C.-W.; Zhang, T.-Y.; Ding,
J.; Wu, Y.-L.; Li, A.-B.; Su, W.-C.; Saifullah; and Wang,
M. 2026. HyCE-RAG: Hypergraph Chain-of-Evidence

Retrieval-AugmentedGenerationforExplainableMulti-hop
Question Answering.arXiv preprint arXiv:2607.22597.
Blocklove, J.; et al. 2023. Chip-Chat: Challenges and Op-
portunitiesinConversationalHardwareDesign. InMLCAD.
Chang, K.; Wang, Y.; Ren, H.; Wang, M.; Liang, S.; Han,
Y.; Li, H.; and Li, X. 2023. ChipGPT: How Far Are We
from Natural Language Hardware Design.arXiv preprint
arXiv:2305.14019.
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and
Liu, Z. 2024. M3-Embedding: Multi-Linguality, Multi-
Functionality, Multi-Granularity Text Embeddings Through
Self-Knowledge Distillation. InFindings of the Association
for Computational Linguistics: ACL 2024, 2318–2335.
Edge,D.;Trinh,H.;Cheng,N.;Bradley,J.;Chao,A.;Mody,
A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and Lar-
son, J. 2024. From Local to Global: A Graph RAG Ap-
proach to Query-Focused Summarization.arXiv preprint
arXiv:2404.16130.
Feng,Y.;etal.2019. Hypergraphneuralnetworks. InAAAI.
Guo, Z.; et al. 2025. LightRAG: Simple and Fast Retrieval-
Augmented Generation. InFindings of the Association for
Computational Linguistics: EMNLP 2025, 10746–10761.
Gutiérrez, B. J.; Shu, Y.; Gu, Y.; Yasunaga, M.; and Su, Y.
2024. HippoRAG: Neurobiologically Inspired Long-Term
MemoryforLargeLanguageModels. InAdvancesinNeural
Information Processing Systems, volume 37.
Gutiérrez, B. J.; Shu, Y.; Qi, W.; Zhou, S.; and Su, Y. 2025.
FromRAGtoMemory:Non-ParametricContinualLearning
for Large Language Models. InICML.
Guu, K.; et al. 2020. REALM: Retrieval-Augmented Lan-
guage Model Pre-Training. InICML.
Han, Y. 2026. Evidence Units: Ontology-Grounded Docu-
ment Organization for Parser-Independent Retrieval.arXiv
preprint arXiv:2604.00500.
Karpukhin, V.; et al. 2020. Dense Passage Retrieval for
Open-Domain Question Answering. InEMNLP.
Lee, S.; and Choi, M. 2026. TechDocRAG: Relation-
Preserving Retrieval-Augmented Generation (RAG) for
Technical Documents.AI, 7(5): 161.
Lewis, P.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. InNeurIPS.
Li, J.; Li, D.; Xiong, C.; and Hoi, S. 2022. BLIP: Boot-
strapping Language-Image Pre-training for Unified Vision-
Language Understanding and Generation. InProceedings
of the 39th International Conference on Machine Learning,
volume 162 ofProceedings of Machine Learning Research,
12888–12900. PMLR.
Li,X.;Dong,G.;Jin,J.;Zhang,Y.;Zhou,Y.;Zhu,Y.;Zhang,
P.; and Dou, Z. 2025. Search-o1: Agentic Search-Enhanced
LargeReasoningModels.InProceedingsofthe2025Confer-
enceonEmpiricalMethodsinNaturalLanguageProcessing,
5420–5438.
Lin, C.-Y. 2004. ROUGE: A Package for Automatic Evalu-
ation of Summaries. InText Summarization Branches Out,
74–81.Liu, M.; Pinckney, N.; Khailany, B.; and Ren, H. 2023a.
VerilogEval: Evaluating large language models for Verilog
code generation. InICCAD.
Liu, M.; et al. 2023b. ChipNeMo: Domain-Adapted LLMs
for Chip Design.arXiv preprint arXiv:2311.00176.
Lu, Y.; et al. 2024. RTLLM: An Open-Source Benchmark
forDesignRTLGenerationwithLargeLanguageModel. In
ASP-DAC.
Luo, H.; E, H.; Chen, G.; Zheng, Y.; Wu, X.; Guo, Y.; Lin,
Q.; Feng, Y.; Kuang, Z.; Song, M.; Zhu, Y.; and Luu, A. T.
2025. HyperGraphRAG: Retrieval-Augmented Generation
via Hypergraph-Structured Knowledge Representation. In
Advances in Neural Information Processing Systems, vol-
ume 38.
Pan, S.; et al. 2024. Unifying Large Language Models and
Knowledge Graphs: A Roadmap.IEEE Transactions on
Knowledge and Data Engineering.
Papineni, K.; Roukos, S.; Ward, T.; and Zhu, W.-J. 2002.
BLEU: A Method for Automatic Evaluation of Machine
Translation. InACL, 311–318.
Pu, Y.; He, Z.; Lin, S.; Qin, J.; Zhang, X.; Han, H.; Zheng,
H.; Jiang, Y.; Zhuo, C.; Sun, Q.; et al. 2025. MM-GRADE:
A Multi-Modal EDA Tool Documentation QA Framework
Leveraging Retrieval Augmented Generation. In2025
IEEE/ACM International Conference On Computer Aided
Design (ICCAD), 1–9. IEEE.
Pu, Y.; He, Z.; Qiu, T.; Wu, H.; and Yu, B. 2024. Cus-
tomized Retrieval Augmented Generation and Benchmark-
ing for EDA Tool Documentation QA. InProceedings of
the43rdIEEE/ACMInternationalConferenceonComputer-
Aided Design, 116:1–116:9.
Robertson, S.; and Zaragoza, H. 2009. The probabilistic
relevance framework: BM25 and beyond.Foundations and
Trends in Information Retrieval.
Salemi, A.; Pizzorno, J. A.; and Zamani, H. 2023. A
Symmetric Dual Encoding Dense Retrieval Framework for
Knowledge-Intensive Visual Question Answering. InPro-
ceedings of the 46th International ACM SIGIR Conference
onResearchandDevelopmentinInformationRetrieval,110–
120.
Thakur,S.;Ahmad,B.;Pearce,H.;Tan,B.;Dolan-Gavitt,B.;
Karri, R.; and Garg, S. 2024. VeriGen: A Large Language
Model for Verilog Code Generation.ACM Transactions on
DesignAutomationofElectronicSystems,29(3):46:1–46:31.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A.2023. InterleavingRetrievalwithChain-of-ThoughtRea-
soning for Knowledge-Intensive Multi-Step Questions. In
Proceedingsofthe61stAnnualMeetingoftheAssociationfor
ComputationalLinguistics(Volume1:LongPapers),10014–
10037.
Wang,X.;Wan,G.-W.;Wong,S.-Z.;Zhang,L.;Liu,T.;Tian,
Q.; and Ye, J. 2024. ChatCPU: An Agile CPU Design and
Verification Platform with LLM. InDAC, 212:1–212:6.
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.;
Xia, F.; Chi, E. H.; Le, Q. V.; and Zhou, D. 2022. Chain-
of-ThoughtPromptingElicits ReasoninginLargeLanguage
Models.NeurIPS.

Wu, H.; He, Z.; Zhang, X.; Yao, X.; Zheng, S.; Zheng, H.;
andYu,B.2024. ChatEDA:ALargeLanguageModelPow-
ered Autonomous Agent for EDA.IEEE Transactions on
Computer-Aided Design of Integrated Circuits and Systems,
43(10): 3184–3197.
Xing, C.; Wong, S.; Wan, X.; Lu, Y.; Zhang, M.; Ma, Z.;
Qi, L.; Li, Z.; Guan, N.; Jiang, Z.; Wang, X.; and Yang,
J. 2026. ChipMind: Retrieval-Augmented Reasoning for
Long-ContextCircuitDesignSpecifications.Proceedingsof
theAAAIConferenceonArtificialIntelligence,40(2):1337–
1345.
Yadati, N.; et al. 2019. HyperGCN: A New Method for
TrainingGraphConvolutionalNetworksonHypergraphs. In
NeurIPS.
Yan, Y.; and Xie, W. 2024. EchoSight: Advancing Visual-
Language Models with Wiki Knowledge. InFindings of the
Association for Computational Linguistics: EMNLP 2024,
1538–1551.
Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Griffiths, T. L.; Cao,
Y.;andNarasimhan,K.2023a. TreeofThoughts:Deliberate
Problem Solving with Large Language Models.NeurIPS.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,
K.; and Cao, Y. 2023b. ReAct: Synergizing Reasoning and
ActinginLanguageModels. InInternationalConferenceon
Learning Representations.
Ye, J.; Liu, J.; Zhang, H.; Sheng, M.; Yang, T.; Xu, J.; Li,
J.; Wang, Y.; and Dai, G. 2025. SG-Filter: Enhancing Sim-
ilar Text Retrieval via Hierarchical Summarized-Semantic
Index and Adaptive Filtering. InProceedings of the 34th
ACM International Conference on Information and Knowl-
edge Management, 3866–3876.
Yu, S.; Tang, C.; Xu, B.; Cui, J.; Ran, J.; Yan, Y.; Liu, Z.;
Wang, S.; Han, X.; Liu, Z.; and Sun, M. 2025. VisRAG:
Vision-based Retrieval-augmented Generation on Multi-
modality Documents. InThe Thirteenth International Con-
ference on Learning Representations.
Zhang, M.; and Ma, Y. 2026. HKVM-RAG: Key-Value-
SeparatedHypergraphEvidenceOrganizationforMulti-Hop
RAG.arXiv preprint arXiv:2606.07218.
Zhou, J.; Liu, Z.; Xiao, S.; Zhao, B.; and Xiong, Y. 2024.
VISTA: Visualized Text Embedding For Universal Multi-
ModalRetrieval.InProceedingsofthe62ndAnnualMeeting
oftheAssociationforComputationalLinguistics(Volume1:
Long Papers), 3185–3200.

A Indexing and Retrieval Procedure
Algorithm 1Functional-unit indexing and function-aware
retrieval
Require:documentsD, queryq, trained encodersf base, ffunc,
budgetsk p, kc, K;α= 0.4
Ensure:generation contextC q
Indexing
1:O ← ∅;C ←Chunks(D)
2:for allr∈Parse(d), d∈ Ddo
3:V r←ExtractTypedArtifacts(r);a r←
CanonicalAnchor(r,V r)
4: append(a r,Vr,Chunks(r),Prov(r))toO
5:end for
6:E ←MergeAdmissibleOccurrences(O)
7: storeS(e)andchunks(e)fore∈ E
Retrieval
8:Eq←TopItems kp(E;e7→sim(f func(q), f func(S(e))))
9:Cu←S
e∈Eqchunks(e)
10:C c←TopItems kc(C;c7→sim(f base(q), f base(c)))
11:C cand←Deduplicate(C u∪ Cc)
12:s(c)←αsim(f base(q), f base(c)) + (1−
α) sim(f func(q), f func(c))
13:returnC q←TopItems K(Ccand;c7→s(c))
B Incremental Indexing and Parser
Invariance
Incremental Updates
Starting from a 32-document index, we inserted
27Flute3.mdand compared the update with a fresh 33-
document build. Only the added document and anchors af-
fected by it were recomputed.
Measurement Result
Unchanged functional units reused 402
Functional units built from new document 2
Index outputs matching full build 7/7
Incremental index assembly 0.407 s
Full index assembly 1.396 s
Table A1: Incremental update from 32 to 33 documents.
The seven matched files are the hyperedge, node, prove-
nance,bipartite-edge,chunk-content,type-bag,andrejection
records; single-run times are shown.
Markdown Normalization Invariance
We independently indexed four equivalent corpus variants:
closing heading hashes, repeated blank lines, tilde code
fences, and a UTF-8 byte-order mark.
Measurement Result
Variants with identical semantic graph 4/4
Variants with byte-identical serialized output 3/4
Functional-unit hyperedges per variant 404
Source documents covered 33/33
Invalid retained units per variant 0
TableA2:IndexinvarianceunderequivalentMarkdownsyn-
tax. Semantic-graph equality is determined by hash.All four variants preserve the semantic graph. Repeated
blank lines change the serialized-output hash while preserv-
ing the semantic-graph hash.
C Qualitative Example
Query.How does the-sink_clustering_enablepa-
rameter affect the clock tree synthesis process in the
clock_tree_synthesiscommand?
Referencecontent.Pre-clusteringcreatesonesub-treelevelbefore
the H-tree; each cluster buffer becomes an H-tree endpoint.
Chunk RAG.“...enables sink clustering during the clock tree
synthesisprocess,whichgroupsclocksinksintoclusterstooptimize
the clock tree structure.”
Ours.“...enables pre-clustering of sinks to create one level of
sub-treebeforebuildingtheH-tree,whereeachclusterisdrivenby
a buffer that becomes an endpoint of the H-tree structure.”
One functional-unit-derived and four directly retrieved chunks re-
cover the pre-H-tree subtree and buffer-endpoint relation.