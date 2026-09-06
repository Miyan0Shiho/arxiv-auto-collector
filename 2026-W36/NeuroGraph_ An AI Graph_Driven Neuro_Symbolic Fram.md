# NeuroGraph: An AI Graph-Driven Neuro-Symbolic Framework for Explainable Threat Reasoning in Advanced Manufacturing

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Published**: 2026-09-01 02:45:44

**PDF URL**: [https://arxiv.org/pdf/2609.00604v1](https://arxiv.org/pdf/2609.00604v1)

## Abstract
The growing complexity of cyber-physical attack surfaces in advanced manufacturing has made cyber threat intelligence analysis increasingly difficult. Although large language models and retrieval-augmented generation have improved CTI workflows, text-based approaches remain vulnerable to hallucinations and provide limited support for structured reasoning over interconnected threats. Graph-based RAG reduces some of these limitations, but existing approaches often lack ontology-consistent multi-hop reasoning and transparent evidence tracing across heterogeneous cybersecurity data. This paper proposes a graph-grounded neuro-symbolic framework that integrates ontology-aware symbolic query generation, knowledge graph retrieval, and neural language generation to support accurate and explainable threat analysis across information technology and operational technology environments. The framework adopts a dual-large language model architecture: the first model translates natural-language questions into executable Cypher queries for symbolic graph retrieval, while the second generates answers strictly from the retrieved graph evidence. Experimental evaluation using publicly available cyber threat intelligence benchmarks shows consistent improvements over the published baseline in reasoning accuracy, while also reducing hallucinations, strengthening multi-hop reasoning, and improving robustness to adversarial perturbations. Runtime and explainability analyses further demonstrate that the framework maintains interactive inference performance and exposes graph-grounded reasoning artifacts that allow analysts to inspect and verify each stage of the analysis. Overall, the results highlight the potential of graph-grounded neuro-symbolic reasoning as a scalable, interpretable, and reliable approach to cyber threat intelligence for next-generation Industry 5.0 environments.

## Full Text


<!-- PDF content starts -->

International Journal of Information Security manuscript No.
(will be inserted by the editor)
NeuroGraph: An AI Graph-Driven Neuro-Symbolic Framework for
Explainable Threat Reasoning in Advanced Manufacturing
Padmeswari Nandiya1,b, Ahmad Mohsin1,a, Iqbal H. Sarker1,c, Ahmed Ibrahim1,d, Helge
Janicke1,e
1School of Science (Computing & Security Discipline), Edith Cowan University, Perth, WA 6027, Australia
Received: date / Accepted: date
Abstract Thegrowingcomplexityof cyber-physicalattack
surfaces in advanced manufacturing environments has made
reliablecyberthreatintelligenceanalysisincreasinglychal-
lenging.Whilelargelanguagemodelsandretrieval-augmented
generationhaveenhancedcyberthreatintelligencecapabili-
ties,text-basedapproachesremainpronetohallucinationsand
lack structured reasoning over interconnected threats. Graph-
based retrieval-augmented generation partially addresses this
limitationbutoftenfailstosupportontology-consistentmulti-
hop reasoning and transparent evidence tracing across het-
erogeneous cybersecurity knowledge. This paper proposes a
graph-grounded neuro-symbolic framework that integrates
ontology-awaresymbolicquerygeneration,knowledgegraph
retrieval,andneurallanguagegenerationtoenableaccurate
andexplainablethreatanalysisacrossinformationtechnology
and operational technology domains. The proposed frame-
work employs a dual-large language model architecture in
which the first model translates natural-language queries into
executable Cypher statements for symbolic graph retrieval,
while the second generates responses exclusively from the re-
trieved graph evidence. Experimental evaluation on publicly
available cyber threat intelligence benchmarks demonstrates
consistentimprovementsoverthepublishedbaselineinrea-
soning accuracy while reducing hallucinations, improving
multi-hop reasoning, and increasing robustness against ad-
versarial perturbations. Runtime and explainability analyses
furthershowthattheframeworkmaintainsinteractiveinfer-
ence performance while exposing graph-grounded reasoning
artifacts that enable analysts to inspect and verify each stage
ofthereasoningprocess.Theseresultsdemonstratethepoten-
tial of graph-grounded neuro-symbolic reasoning to provide
aCorresponding author: a.mohsin@ecu.edu.au
be-mail: p.nandiya@ecu.edu.au
ce-mail: m.sarker@ecu.edu.au
de-mail: ahmed.ibrahim@ecu.edu.au
ee-mail: h.janicke@ecu.edu.auscalable,interpretable,andreliablecyberthreatintelligence
for next-generation Industry 5.0 environments.
KeywordsExplainable Artificial Intelligence ·Large
LanguageModels·Cybersecurity·AdvancedManufacturing ·
Knowledge Graphs·Neuro-symbolic
1 Introduction
Industry5.0representstheemergingevolutionofadvanced
manufacturing,placingemphasisonHuman-centriccollab-
orationbetweendomainexpertsandintelligentsystems[ 1].
In this context, Industrial Control Systems (ICS) constitute a
foundational component, as they govern and automate pro-
cessesacrossmodernindustrialenvironments.Theincreasing
integration of intelligent and connected technologies within
ICS amplifies the importance of AI-driven assistants capable
ofsupportingcyberresilienceandinformeddecision-making
[2].
ICS are extensively deployed across critical infrastruc-
turesectors,includingenergy,manufacturing,andessential
utilities, where their secure and continuous operation is of
strategic importance [ 3]. However, the ongoing digitalization
and increasing connectivity of these systems have substan-
tiallyexpandedtheirattacksurface,resultinginheightened
exposure to cyber threats [ 4]. This growing vulnerability has
beendemonstratedbyseveralhigh-profilecyberincidents.No-
tably,theStuxnetattackin2010andtheTritonattackin2017
illustratedthecapabilityofadversariestomanipulateanddis-
ruptphysicalindustrialprocessesthroughcybermeans[ 5,6].
More recently, the 2021 breach of the Oldsmar water treat-
ment facility revealed that even small-scale municipal ICS
deployments remain susceptible to cyber compromise [ 7].
Collectively,theseincidentsdemonstratethatcyberattacks
targeting ICS extend beyond digital infrastructures to pro-
arXiv:2609.00604v1  [cs.CR]  1 Sep 2026

2
ducetangiblephysicalconsequences,emphasizingtheneed
forintelligentsecuritymechanismscapableofassistingan-
alystsinunderstandingandmitigatingcomplexmulti-stage
cyber-physical attack scenarios.
RecentadvancesinLargeLanguageModels(LLMs)have
significantlyimprovedcyberthreatintelligence(CTI)analysis
throughRetrieval-AugmentedGeneration(RAG),enabling
language models to access external cybersecurity knowledge
during inference [ 8–11]. While promising, conventional text-
based RAG systems remain susceptible to hallucinations and
often struggle to perform structured reasoning over intercon-
nected cyber-physical environments where attack paths span
information technology (IT), operational technology (OT),
and physical processes [ 12–14]. To address these limitations,
recent studies have proposed Graph Retrieval-Augmented
Generation(Graph-RAG),whichleveragesknowledgegraphs
toprovidestructuredandrelationalreasoningacrossassets,
vulnerabilities, weaknesses, and attack patterns [ 15]. Al-
though Graph-RAG represents a significant advancement
overtext-onlyretrieval,existingapproachesremainlimitedin
supporting the heterogeneous, multi-hop reasoning required
in Industry 5.0 cyber-physical environments.
Severalpracticalchallengescontinuetohinderthedeploy-
ment of Graph-RAG in industrial cybersecurity. First, cyber-
securityknowledgeisinherentlyheterogeneous,requiringthe
integrationofdiverseintelligencesourcessuchasCVE,CWE,
MITREATT&CK,CAPEC,assetinventories,andincident
reportsacrossbothITandOTdomains[ 11,16,17].Existing
cybersecurity ontologies frequently model only subsets of
thisecosystem,leavingimportantcross-domainrelationships
insufficientlyrepresented [ 18,19].Second,evenwhencom-
prehensiveknowledgegraphsareavailable,cybersecurityana-
lystsrarelyinteractdirectlywithgraphquerylanguages,while
existinggraph-basedretrievalsystemsprovidelimitedsupport
for transparent multi-hop reasoning over complex industrial
environments[ 20].Finally,althoughGraph-RAGenableslan-
guagemodelstoretrievestructuredgraphknowledge,many
existing systems still provide limited reasoning transparency,
making it difficult for analysts to inspect intermediate reason-
ing steps and verify how conclusions are derived [ 21,22].
These challenges highlight the need for a graph-grounded
reasoningframeworkcapableofcombiningstructuredsym-
bolicreasoningwiththeflexibilityofneurallanguagemodels
while maintaining explainability for cyber-physical threat
analysis. These challenges motivate the development of a
graph-grounded neuro-symbolic framework that combines
the transparency of symbolic reasoning with the flexibility
of neural language models to enable explainable and reliable
cyberthreatanalysisacrossinterconnectedITandOTenvi-
ronments. Accordingly, this paper proposesNeuroGraph, a
graph-groundedneuro-symbolicframeworkforthreatreason-
inginadvancedmanufacturingsystems,implementedthrough
theGRICS(Graph-Integrated Retrieval for Industry-CentricSecurity) architecture. GRICS combines a cyber-physical
knowledge graph with a dual-LLM architecture. The first
LLMtranslates analyst queriesintoontology-awareCypher
queriesforsymbolicknowledgeretrieval,whilethesecond
LLM generates natural-language responses using only the
retrieved graph evidence. By separating symbolic retrieval
from language generation, GRICS provides graph-grounded
reasoning that improves transparency, supports multi-hop
inferenceacrossheterogeneouscyber-physicalentities,and
reduces unsupported model generation.
GRICS is built upon the BRIDG-ICS ontology devel-
opedinthepreviouswork[ 23],whichmodelscyber-physical
assets, vulnerabilities, weaknesses, attack techniques, and
adversarial behaviours within a unified industrial knowledge
graph. Leveraging this ontology, GRICS enables explainable
graph-groundedmulti-hopreasoningacrossinterconnected
IT and OT environments. The methodological novelty of
GRICS lies not merely in combining knowledge graphs with
large language models, but in the way its neuro-symbolic
architecture couples neural and symbolic components dur-
ingretrievalandreasoning.UnlikeGraph-RAGapproaches
that primarily rely on embedding similarity, graph expan-
sion,clustering,orlearnedgraphrepresentationstoconstruct
context,GRICSadoptssymbolicCypherexecutionasthepri-
maryevidence-retrievalmechanism.Thelanguagemodelfirst
translatesan analystqueryinto anontology-constrainedexe-
cutableprogram,andonlyevidencereturnedthroughexplicit
graphexecutionisadmittedtodownstreamanswergeneration.
Embedding-based retrieval is activated only when the initial
symbolic query fails and serves solely to identify a graph
anchorfromwhichanontology-compliantCypherqueryis
regenerated. Consequently, neural retrieval assists symbolic
reasoning rather than replacing it, enabling deterministic
graphtraversal,ontology-consistentmulti-hopreasoning,and
explicit evidence traceability across interconnected IT and
OT cybersecurity entities.
The proposed framework assumes the availability of a
curated cybersecurity knowledge graph derived from the
BRIDG-ICS ontology and focuses on graph-grounded threat
reasoning rather than ontology construction or maintenance.
Consequently,thecurrentstudydoesnotaddressautomated
ontologyevolution,continuouscyberthreatintelligencein-
gestion,orreal-timeknowledgegraphsynchronization.These
assumptions, together with the scope and current limitations
oftheframework,arediscussedindetailinSectionIV.The
primary contributions of GRICS are summarised as follows:
nandiya2025bridgicsaigroundedknowledgegraphs
–Graph-Grounded Threat Reasoning for Smart Manu-
facturing.We propose a graph-grounded retrieval frame-
work that combines knowledge graph retrieval with LLM
reasoning to enable multi-hop inference across assets,
vulnerabilities, weaknesses, and attack techniques for
cyber-physical threat analysis.

3
–Symbolic-First Neuro-Symbolic Retrieval.We intro-
duce a symbolic-first retrieval architecture in which
ontology-constrainedCypherexecutionconstitutesthepri-
mary reasoning pathway, while embedding retrieval is re-
stricted to recovery of graph anchors following symbolic-
queryfailure.Unlikehybridretrievalschemesinwhich
neural and symbolic evidence are independently com-
bined,GRICSrequiresrecoveredanchorstobeconverted
back into executable ontology-compliant Cypher queries
before evidence is admitted to downstream reasoning.
–Explainable NeuroGraph Reasoning.We develop a
transparentreasoningworkflowthatexposesthegenerated
Cypher query, retrieved graph evidence, and grounded
natural-language response, enabling analysts to inspect
and verify each stage of the reasoning process.
The remainder of this paper is organized as follows.
Section2introducesthetechnicalpreliminariesunderlying
graph-grounded cyber threat reasoning. Section 3 reviews
the related research and identifies the existing research gaps.
Section4formulatestheresearchproblemandoutlinesthe
assumptions and scope of the proposed framework. Sec-
tion5presentstheproposedgraph-groundedneuro-symbolic
framework.Section6describestheexperimentalevaluation
and discusses explainability and robustness analyses. Finally,
Section 7 concludes the paper and outlines future research
directions.
2 Preliminaries
2.1 Cybersecurity Knowledge Graphs
In Industry 5.0 environments, Industrial Control Systems
(ICS) face increasingly complex cyber threats due to inter-
connectedandAI-enabledinfrastructures[ 1,24].Knowledge
graphs (KGs) provide a structured representation that inte-
grates diverse cybersecurity data across IT and OT domains,
enabling a unified and machine-interpretable view of sys-
tementitiesandtheirrelationships[ 16,23,25].Theserela-
tionships support multi-hop reasoning over interconnected
components, facilitating the analysis of attack dependen-
cies and system-level risks [ 26]. Formally, a cybersecurity
knowledgegraphisrepresentedbythegraphmodelshownin
Equation (1):
G=(V,E,R)(1)
2.2 Graph Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has the potential
toenhanceLLMsbyincorporatingexternalknowledgedur-
ing inference, thereby improving factual grounding [ 27–29].However, it relies on unstructured text, limiting its abil-
ity to capture relationships between cybersecurity entities.
Graph-based RAG (Graph-RAG) addresses this limitation
by retrievingstructured subgraphs 𝑅⊆𝐺from aknowledge
graph𝐺,enablingmulti-hopreasoningacrossinterconnected
entities[30–32].Graph-RAGenablesLLMstoreasonover
structured, multi-hop knowledge by integrating symbolic
graph traversal with embedding-based similarity search.
2.3 Neuro-Symbolic Reasoning
Symbolic–semanticreasoningmapsnaturallanguageinputsto
executable structured representations over knowledge graphs,
enablingprecise,explainableinference.Givenaninput 𝑞∈𝑋∗,
the goalis toderive 𝑐∈𝐶such thatexecuting iton graph 𝐺
yieldsrelevantresults, 𝑅=exec(𝑐,𝐺) .Thiscombinesseman-
tic interpretation with symbolic graph operations to support
multi-hop reasoning over interconnected cybersecurity en-
tities, providing traceable, context-aware threat analysis in
cyber–physical environments.
2.4 Cyber-Physical Threat Modelling
Cyber-physicalthreatmodellingrepresentssecurityeventsas
interconnectedrelationshipsamongdigitalassets,operational
components, vulnerabilities, weaknesses, attack patterns, ad-
versarial techniques, and physical processes [ 33]. In indus-
trial environments, security incidents rarely remain confined
to a single software component or network layer. Instead,
compromise can propagate across information technology
(IT) and operational technology (OT) domains, affecting
industrial assets,control processes,and potentially physical
operations [2].
A cyber-physical attack can therefore be viewed as a
sequenceof dependent security eventsrather than anisolated
vulnerability [ 11]. For example, exploitation of a software
vulnerabilitymayexposeanunderlyingweakness,enablea
particularattackpattern,correspondtoaknownadversarial
technique, and subsequently affect an operational asset or
process [ 34]. Representing these dependencies explicitly
supportstheanalysisofmulti-stageattackpaths.Suchanalysis
requires consideration of both semantic relationships among
cybersecurity concepts and operational relationships among
assets, systems, and networked components [23, 35].
These structured threat representations are particularly
important in Industry 5.0 environments, where increased
connectivity among intelligent systems, industrial assets, and
human operators creates complex interdependencies [ 36].
Cyber-physical threat modelling therefore provides a con-
ceptual foundation for analyzing multi-hop attack scenarios,
correlating heterogeneous cybersecurity information, and

4
supportingexplainablesecuritydecision-makingacrossinter-
connected IT and OT infrastructures.
3 Related Work
This section reviews the literature most relevant to graph-
groundedcyberthreatintelligenceandpositionstheproposed
GRICS framework within the current state of the art. The
reviewedstudieswereselectedtorepresenttheprincipalre-
searchdirectionsrelevanttothiswork,includingcybersecurity
knowledge graphs, retrieval-augmented generation, Graph-
RAG, neuro-symbolic reasoning, and explainable artificial
intelligence.Prioritywasgiventorecentstudies(2022–2025)
to reflect the rapid development of Graph-RAG and large
language models in cybersecurity, while seminal works were
retained where necessary to provide foundational context.
Representativestudieswerefurtherselectedbasedontheirrel-
evance to industrial control systems, cyber-physical systems,
andcyber threatintelligence. Basedonthese criteria,the lit-
erature is organized into four themes: (i) retrieval-augmented
generationandGraph-RAG,(ii)neuro-symbolicretrievaland
reasoning, (iii) threat reasoning in cybersecurity, and (iv)
ontologicalmodellingforcybersecurity.Thisstructureallows
GRICS’s research gaps to be systematically identified.
3.1 Retrieval-Augmented Generation for Cyber Threat
Intelligence
RAGforLLMLearning.Largelanguagemodels(LLMs)
excelatnatural-languageunderstandingandgenerationbut
rely on parametric knowledge, limiting factual reliability and
adaptability to evolving domains [ 37]. Retrieval-Augmented
Generation(RAG)addressestheseissuesbygroundingout-
putsinexternalknowledge,improvingfactualaccuracyand
timeliness[ 27,38].However,mostRAGframeworksretrieve
from unstructured text [ 39,40], which offers limited struc-
ture and is insufficient for multi-step reasoning over complex
relations [41].
Recent advances have sought to overcome these limita-
tionsbyintegratingstructuredgraph-basedknowledgeinto
retrieval-augmented reasoning [ 15]. Graph representations
explicitlyencodeentitiesandrelations,enablingrichercontex-
tual grounding and more effective multi-step inference. This
has led to the emergence of Graph-RAG frameworks, which
extendtraditionalRAGbyleveragingstructuredknowledge
toimproveretrievalqualityandreasoninginterpretability.Ex-
istingGraph-RAGframeworksimprovestructuredretrieval
butremainlimitedinsupportingdeterministicgraphtraversal,
ontology-aware reasoning, and transparent multi-hop infer-
ence over heterogeneous cyber-physical knowledge. These
limitationsmotivate thegraph-grounded neuro-symbolicde-
sign adopted by GRICS.Graph-RAGMethodsforCyberThreatIntelligence.
FollowingtheemergenceofGraph-RAGframeworks,existing
approaches differ in how graph structures are incorporated
into retrieval and reasoning processes. These systems can be
broadly characterized according to their underlying retrieval
strategies, which directly influence their ability to support
multi-hop reasoning, scalability, and interpretability.
Thefirst categoryconsists ofembedding-based retrieval
methods, which map entities and relations into continuous
vector spaces and perform semantic similarity search [ 42–
44].Theseapproachessupportefficientlarge-scaleretrieval
and are widely adopted in RAG pipelines. However, because
they rely on approximate matching, they do not explicitly
enforcestructuralconstraintsandmaythereforeretrievese-
manticallyrelevantbutcausallyinvalidconnections.Asec-
ond categoryaddresses this limitation throughsubgraph
andmulti-hopexpansion,wherereasoningisperformedby
iteratively exploring connected graph neighborhoods. Sys-
tems such as Think-on-Graph 2.0 [ 45] explicitly guide LLM
reasoning through graph traversal, while KGMP [ 46] and
DRKG[47]introducestructuredmulti-hopreasoningstrate-
gies to improve interpretability. Similarly, SimGRAG [ 30]
andKG2RAG[ 31]retrievesubgraphstocapturebroaderrela-
tionalcontext.Althoughtheseapproachesimprovemulti-step
reasoning, unconstrained expansion in densely connected
graphs may introduce irrelevant or redundant paths.
Thethird categoryfocuses onhierarchical and struc-
tured retrieval methods, which seek to improve scalability
and support reasoning over larger or longer-context knowl-
edge spaces. RAPTOR [ 48], for instance, organizes infor-
mationintotree-likestructuresforrecursiveretrieval,while
GraphRAG [ 49] employs clustering and summarisation to
improve knowledge organisation. Recent frameworks such as
GRAG [32] further extend graph-centric retrieval to improve
flexibility and retrieval effectiveness. Finally, thefourth cate-
gorycomprisesneuralgraphreasoningapproaches,which
usegraphneuralnetworksandhybridarchitecturestolearn
complex dependencies directly from graph structure [ 50–
52]. These methods provide strong representation-learning
capabilitiesandcancapturecomplexrelationalpatterns.How-
ever,theirreasoningprocessesareoftenencodedimplicitly
withinmodelparameters,whichcanlimittransparencyand
makeintermediatereasoningpathsmoredifficulttoinspect.
Despitetheseadvances,mostGraph-RAGsystemsstillpro-
vide limited support for explicit, constraint-driven reasoning.
As shown in Table 1, symbolic querying remains under-
used, restricting deterministic and interpretable multi-hop
inference.Thislimitationisparticularlyimportantincyberse-
curity,wheregeneral-purposeGraph-RAGsystemsoftenlack
cyber-physical semantics, attacker-behaviour models, and
explicit IT/OT integration required forIndustry 5.0threat
intelligence. Cybersecurity-oriented RAG systems such as
MoRSE[ 53]andProveRAG[ 54]primarilyrelyonunstruc-

5
tured text retrieval, while graph-based approaches such as
CyKG-RAG [ 55] and GraphRAG under Fire [ 56] improve
structuredretrievalbutremainlimitedinrepresentingcom-
plex, multi-stage cyber–physical dependencies.
3.2 Neuro-Symbolic Reasoning in Graph-RAG
Neuro-symbolic approaches aim to integrate data-driven
learning with structured reasoning over knowledge graphs,
combining the flexibility of neural models with the consis-
tency of symbolic representations [ 57,58]. In Graph-RAG
systems, this paradigm enables the use of learned embed-
dingsforretrievalwhileleveraginggraphstructurestoprovide
contextual grounding for reasoning [59].
Inpractice,manyexistingapproachesemphasizeneural
retrieval mechanisms, where embeddings or learned repre-
sentations guide the selection of relevant nodes and sub-
graphs[60].Whileeffectiveforscalabilityandgeneralization,
such approaches do not explicitly enforce logical constraints,
and reasoning is often guided by similarity rather than struc-
tural validity [ 61]. As a result, multi-hop inference may
include semantically relevant but structurally inconsistent re-
lationships, particularly in complex domains [ 62]. Symbolic
querying (e.g., Cypher, SPARQL) provides an alternative by
enabling deterministic and constraint-driven traversal over
knowledgegraphs[ 63–65].Byenforcingexplicitstructural
conditions, symbolic methods ensure that all inferred re-
lationships adhere to valid graph connections, supporting
precise and interpretable reasoning. Existing neuro-symbolic
Graph-RAGsystems,however,rarelyintegrateexplicitsym-
bolicquerygenerationandneuralretrievalwithinaunified
reasoning pipeline. As a result, deterministic graph traversal,
ontology-aware reasoning, and transparent evidence tracing
remainunderexplored,particularlyincyber-physicalthreat
intelligence.
Recent graph-grounded reasoning approaches have in-
creasinglymovedbeyondstaticretrievaltowardmulti-strategy,
agentic, and neuro-symbolic architectures. BYOKG-RAG
combines LLM-generated graph artifacts, including can-
didate entities, reasoning paths, and OpenCypher queries,
with specialized graph-retrieval tools and iterative refine-
ment over heterogeneous knowledge graphs [ 66]. SymAgent
adopts an agentic neuro-symbolic architecture in which an
Agent-Plannerextractssymbolicreasoningstructuresfrom
the knowledge graph and an Agent-Executor dynamically
invokesexternal andgraph-basedtools toaddress complex
reasoningtasksandincompletegraphknowledge[ 67].Inpar-
allel,TUNSRinvestigatesaunifiedneuro-symbolicreasoning
framework that combines neural representations with sym-
bolicfirst-orderlogicreasoningoverdynamicallyconstructed
reasoning graphs [ 68]. These approaches demonstrate im-
portant advances toward adaptive graph interaction, iterativetooluse,andtighterintegrationbetweenneuralandsymbolic
reasoning.
GRICSdiffersfromtheserecentapproachesinbothitsrea-
soningobjectiveandthefunctionalrolesassignedtoitsneural
andsymboliccomponents.BYOKG-RAGemploysmultiple
graph-retrievalstrategiestoimproveknowledge-graphques-
tionanswering,whereasGRICSconstrainsacceptedevidence
toontology-compliantCypherexecutionovertheBRIDG-ICS
cybersecurity ontology. Compared with agentic architectures
such as SymAgent, which rely on iterative planning and
dynamictoolinvocation,GRICSadoptsamoreconstrained
dual-LLMreasoningpipelineinwhichthefirstLLMgener-
ates executable symbolic queries and the second synthesises
responses only after graph evidence has been retrieved. Rela-
tive to broader neuro-symbolic frameworks such as TUNSR,
GRICS is specifically designed for explainable cybersecu-
rityreasoningacrossvulnerability,weakness,attack-pattern,
MITRE ATT&CK, and industrial-asset relationships.
A key distinction is that semantic retrieval in GRICS
cannot independently determine the final reasoning evidence.
Embedding-basedretrievalisusedtorecovercandidategraph
anchors when symbolic retrieval fails, but the recovered can-
didatemustsubsequentlypassthroughontology-compliant
Cyphergenerationandgraphexecutionbeforebeingsupplied
to the Answer-LLM. This establishes an explicit verifica-
tion boundary between semantic interpretation and factual
retrieval. Consequently, the principal advantage of GRICS
is not unrestricted reasoning flexibility, but the combina-
tion of neural adaptability with deterministic graph traversal,
ontology consistency, and end-to-end evidence traceability
required for cybersecurity decision support.
3.3 Threat Reasoning in Cybersecurity
Threat reasoning has become an increasingly important re-
searchdirectionforsupportingcyberthreatintelligence(CTI)
analysis,enablingsystemstoinferrelationshipsamongvul-
nerabilities, attack techniques, threat actors, and defensive
actions rather than performing isolated information retrieval.
Recentadvanceshaveleveragedlargelanguagemodelsand
knowledgegraphstoimprovestructuredreasoningovercy-
bersecurity knowledge.
Fieblingeret al.[ 69] combine knowledge graphs with
large language models to transform unstructured CTI reports
into actionable threat intelligence, facilitating contextual
analysis and information extraction. Wuet al.[ 70] further
extendthisdirectionbyintegratingknowledgegraphswith
large language models for cyber threat intelligence credi-
bility assessment, demonstrating how structured semantic
relationshipscanimprovethereliabilityofCTI.Morerecently,
CTI-Thinker [ 71] incorporates ATT&CK-aligned knowledge
graphs within a Graph-RAG framework to support attack
intent inference and CTI question answering.

6
Table 1: Comparison of representative RAG and Graph-RAG approaches highlighting gaps in reasoning, constraints, and
retrieval mechanisms.
Work (Year) Symbolic Rea-
soningMulti-hop Con-
sistencyConstraint-
based RetrievalEmbedding-
basedRetrievalRetrieval Type KG LLM Fine-
Tuning
GraphRAG (2025) [49]✗ ✗ ✗✓ Clustering / Sub-
graph✓✗
KG2RAG (2025) [31]✗ ✗ ✗✓ Graph-guided Re-
trieval✓✗
GRAG (2025) [32]✗ ✗ ✗✓Graph Retrieval✓✗
GNN-RAG (2024) [52]✗ ✗ ✗✓ Neural Graph Rea-
soning✓✗
MoRSE (2024) [53]✗ ✗ ✗✓Textual Retrieval✗ ✗
CyKG-RAG (2024) [55]✗△✗✓KG-based Retrieval✓✗
GRICS (Ours) ✓ ✓ ✓ ✓ Neural + Symbolic
(Cypher-based)✓ ✓
Thesestudiesdemonstratethatstructuredknowledgesub-
stantiallyimprovescybersecurityreasoningcomparedwith
language-model-onlyapproaches.However,existingmethods
primarilyrelyonGraphRAGorembedding-basedretrieval
over cybersecurity knowledge graphs. Consequently, the rea-
soningprocessremainslargelyimplicitwithinthelanguage
modelandprovideslimitedsupportfordeterministicgraph
traversal or explicit verification of intermediate reasoning
steps. In contrast, this work investigates graph-grounded
neuro-symbolic reasoning through ontology-aware Cypher
generation andsymbolic graphexecution, enabling transpar-
entmulti-hopreasoninginwhicheverygeneratedresponse
can be traced to explicit graph evidence.
3.4 Ontological Modelling in Cybersecurity
Ontological modelling has been central to cybersecurity
research, enabling structured and machine-interpretable rep-
resentations of threat intelligence, vulnerabilities, and at-
tackbehaviours.EarlyeffortssuchasMITREATT&CKand
CAPEContologies [ 72,73], along with frameworks like
STIX[74],CVE, andCWE, established standardised repre-
sentationstosupportknowledgesharingandanalysis.Recent
workextendsthesefoundationsintoCyberKnowledgeGraphs
(CKGs),integratingheterogeneousdatasourcesandcapturing
relationships among entities for contextual reasoning [ 75–
79]. However, these approaches often emphasize structural
completenessoveradaptivereasoning,limitingtheirability
to support dynamic, multi-hop analysis in cyber–physical
environments.
As highlighted in Table 1, current RAG and Graph-RAG
approaches do not provide integrated symbolic reasoning,
cyber–physical modelling, or explainable threat analysis,
whichrestrictstheirapplicabilityincomplex,safety-critical
settings. These limitations are addressed by the proposed
GRICS framework.4 Problem Statement
The increasing adoption of Industry 5.0 technologies has
createdhighlyinterconnectedcyber-physicalenvironments
spanninginformationtechnology(IT),operationaltechnol-
ogy(OT),industrialassets,vulnerabilities,andattackerbe-
haviours.Effectivecyberthreatintelligence(CTI)therefore
requiresreasoningacrossheterogeneousentitiesandmultiple
abstraction levels rather than relying on isolated document
retrieval or keyword matching.
Although Retrieval-Augmented Generation (RAG) im-
proves factual grounding through external knowledge, con-
ventional text-based retrieval lacks explicit representations
of relationships among cybersecurity entities. Recent Graph-
RAG approaches incorporate knowledge graphs, but many
still rely on embedding similarity, heuristic graph expansion,
orsummarisation,limitingdeterministictraversal,ontology
compliance,andtransparentmulti-hopreasoning.Cyberse-
curity analysts also typically express investigation goals in
natural language rather than graph query languages, creating
aneedforsystemsthatcaninterpretuserintentwhilepreserv-
ingontologyconstraintsandreasoningtraceability.GRICS
addresses these limitations through a unified neuro-symbolic
pipeline that combines ontology-aware Cypher generation,
symbolicgraphretrieval,embedding-assistedqueryrecovery,
and grounded natural-language response generation. This
work is developed under the following assumptions:
–Thecybersecurityknowledgegraphisconstructedfroma
trustedontologyandaccuratelyrepresentscyber-physical
entities and their relationships.
–Retrievedgraphinformationisconsideredauthoritative,
while the language model is responsible only for inter-
pretingandsummarizingretrievedevidenceratherthan
generating unsupported knowledge.
–Analystsinteractwiththeframeworkusingnatural-language
queriesratherthandirectlywritinggraphquerylanguages
such as Cypher.
The scope of this work is limited to graph-grounded
cyber threat reasoning over the BRIDG-ICS ontology. The

7
frameworkdoesnotaddressautomaticontologyconstruction,
knowledge graph population, or ontology evolution. Like-
wise, although explainability is achieved through transparent
graph-grounded reasoning artifacts, formal human-subject
evaluation of analyst trust and cognitive workload is beyond
the scope of this study and remains an important direction
for future work.
5 Proposed GRICS Framework
This section presents the proposed NeuroGraph-based neuro-
symbolicreasoningframework,referredtoasGraph-Integrated
Retrieval for Industry-Centric Security (GRICS), for ad-
vanced threat reasoning in Industry 5.0 cyber–physical en-
vironments. GRICS is implemented as a domain-specific
knowledge-graph-based retrieval-augmented generation (KG-
RAG) framework that enables multi-hop attack-path analy-
sis, threat-impact assessment, and mitigation-strategy gen-
eration by combining structured cyber–physical knowledge
with large language model reasoning. The framework inte-
gratesthreecorecomponents:(i)acyber–physicalknowledge
graph derived from the BRIDG-ICS ontology that unifies
information-technology and operational-technology assets,
(ii) a dual-LLM pipeline that separates ontology-aware sym-
bolicretrievalfromanswersynthesis,and(iii)athreat-centric
KG-RAG question-answering dataset designed to support
multi-hop reasoning and attack-scenario analysis.
ThedesignofGRICSisinformedbyrecentadvancesin
knowledge-graph-guidedRAGsystems,whichdemonstrate
that structured knowledge can improve retrieval precision,
interpretability, and reasoning robustness. GRICS extends
theseapproachesthroughneuro-symbolicgraphreasoningby
combininglargelanguagemodelinterpretationwithontology-
constrained Cypher generation and explicit knowledge-graph
traversal.Incontrasttogeneral-domainKG-RAGapproaches,
GRICS is tailored to Industrial Control Systems security,
withafocusoninterpretableattack-pathreasoningandcyber–
physicaldependency modelling.Anoverview oftheGRICS
architectureisshowninFigure1.Theimplementationdetails
and source code are publicly available1.
5.1 Data Collection
Industrialcyber-physicalenvironmentsarehighlyintercon-
nected and often exhibit cascading security risks, where
vulnerabilities in networkinfrastructure or security controls
canpropagateacrossmultiplefactorycomponents.Capturing
such interactions requires integrating heterogeneous cyberse-
curity information from both cyber and operational domains.
1https://github.com/ahmadspm/Resellient-Industry-5.0
--kG-Digital-Twins/tree/mainThedatasetsusedinthisstudyarecollectedfrommultiple
authoritativecybersecurityrepositories,includingCommon
VulnerabilitiesandExposures(CVE),CommonPlatformEnu-
meration(CPE),CommonWeaknessEnumeration(CWE),
CAPEC, MITRE ATT&CK, and industrial asset information
obtained from real-world Industrial Internet of Things (IIoT)
and Operational Technology (OT) testbeds. These hetero-
geneous sources are transformed into node and edge CSV
filesandrepresentedusingthelabelledpropertygraph(LPG)
model, in which cybersecurity entities are encoded as la-
belled nodes, relationships are represented as typed edges,
anddescriptiveattributesarestoredasnodeorrelationship
properties.TheresultingLPGisimplementedinNeo4j,which
providesthegraphdatabaseenvironmentusedforknowledge-
graph construction, storage, and subsequent Cypher-based
retrieval.
5.2 Knowledge Sources and Ontology
The imported cybersecurity datasets are organized accord-
ing to the BRIDG-ICS ontology [ 23], which provides the
semanticstructureforintegratingindustrialassets,software
products,vendors,vulnerabilities(CVE),weaknesses(CWE),
attack patterns (CAPEC), MITRE ATT&CK techniques, and
operationalzones.Eachentityisassignedasource-specific
uniqueidentifierand mappedtothecorrespondingontology
class, helping align records across heterogeneous cyberse-
curity sources and prevent duplicate nodes during graph
construction.
The ontology model uses a domain-driven design where
cybersecurity concepts are modelled as classes and linked by
typed relationships. Relations such ashas_Vulnerability,At-
tack(Asset),has_CWE,use_Technique, andlocated_In(Zone)
enableexplainablemulti-hopreasoningacrosscyber-physical
attack paths. Relationships are instantiated only when sup-
portedbysourcedatamappingsordefinedintheBRIDG-ICS
ontology,preventingunsupportedassociationsduringgraph
construction. In this study, a fixed snapshot of the BRIDG-
ICSknowledgegraphisusedforalltrainingandevaluationto
ensureaconsistentexperimentalsetup.Cybersecurityrecords
fromselectedsourcesarepreprocessedofflineintonormalised
nodes and relationships following the BRIDG-ICS schema
and then imported into the graph. When the graph is updated
outsideevaluation,embeddingsaregeneratedfornewormod-
ified nodes to keep the embedding-based fallback retrieval
mechanism consistent.
5.2.1 Knowledge Graph Construction
Theknowledge-graphconstructionpipelineconsistsoffour
stages:sourceextraction,entitynormalization,relationship
mapping,andgraphingestion.First,recordsfromtheselected
cybersecuritysourcesareconvertedintointermediatenode

8
Fig. 1: GRICS KG-RAG architecture and dual LLM reasoning pipeline.
and relationship tables. Second, entities such as CVE, CWE,
CAPEC, MITRE ATT&CK techniques, and CPE records are
normalisedusingcanonicalidentifierstoalignheterogeneous
sourcesandreduceduplication.Third,relationshipsamong
these entities, together with industrial assets and software
components, are established according to the BRIDG-ICS
schema.Finally,theresultingnodeandrelationshiptablesare
importedintothegraphdatabaseforgraph-basedretrievaland
reasoning. The complete BRIDG-ICS ontology schema, rela-
tionshipdefinitions,andassociatedimplementationresources
are available in the project repository.2
Based onthe integrateddataset, the resultingknowledge
graphisrepresentedas G=(V,E) ,whereVdenotestheset
of cyber-physical entities and Erepresents the semantic and
operational relationships among them. A principal reasoning
chain represented in the graph is:
Industry 5.0 Assets →CVE→CWE→CAPEC→
MITRE ATT&CK.
Thisstructureenablesmulti-hopreasoningforattack-path
analysis and threat attribution across interconnected cyber-
2https://github.com/ahmadspm/Industry-5.0--Intellige
nt-Threat-Analytics-KGs-and-LLMssecurityentities.Inadditiontothesesemanticrelationships,
thegraphmodelsoperationalinteractionsthroughtypededges
(𝑢,𝑣)∈E commusingtherelation COMMUNICATES_WITH .These
edgesareenrichedwithrisk-relatedattributes,includingpEx-
ploit,riskWeight,controlStrength,andcostAttack,supporting
quantitative assessment of vulnerability propagation.
5.2.2 Vector Embedding Database
Afterknowledge-graphconstruction,theintermediatenode
records stored in CSV format are transformed into dense
vector representations. Each node is embedded into a 384-
dimensional vector using the all-MiniLM-L6-v2 model,
whichprovidesabalancebetweencomputationalefficiency
and semantic representation quality. Each row in the node
CSV corresponds to a single graph entity and is converted
into a fixed-length embedding representation.
The resulting embeddings are stored in a vector database
to support semantic fallback retrieval. Similarity between
the embedded user query and graph-node representations
is computed using cosine similarity to identify semanti-
cally related candidate nodes. These candidates serve only
as graph anchors for query recovery and are subsequently

9
Fig. 2: Knowledge graph structure and neuro-symbolic re-
trieval workflow in GRICS.
supplied tothe Cypher-generationmodule to regenerate an
ontology-compliantquerybeforesymbolicgraphexecution
resumes. Given the 384-dimensional representation, each
embedding requires approximately 1.5 KB of raw storage
in 32-bit floating-point precision, excluding database and
indexing overhead.
5.3 KG-RAG and Reasoning Pipeline
The KG-RAG pipeline extends the constructed knowledge
graphintoaquery-drivenreasoningsystemforcyber–physical
threatanalysis.Givenauserquery,theframeworkretrieves
relevantgraph-structuredevidencefordownstreamreasoning.
UnlikeconventionalRetrieval-AugmentedGeneration(RAG)
approaches that rely on unstructured text retrieval, the pro-
posedmethodoperatesdirectlyontheknowledgegraph.This
enablesthesystemtofollowexplicitrelationalpathsbetween
entities, allowing dependencies across the cyber–physical
attacksurfacetobeexploredinastructuredandinterpretable
manner. The retrieved subgraph serves as grounded evidence
for downstream reasoning, ensuring that generated responses
are consistent with the underlying graph structure. This de-
signsupportsreliableinferenceovercomplexattackscenarios
while maintaining traceability between retrieved evidence
andgeneratedoutputs.Figure2illustratesthegraph-basedre-
trievalandreasoningprocess,demonstratinghowthepipeline
grounds reasoning in explicit cyber–physical relationships.
Thisdesignalignswithpriorworkshowingthatknowledge
graph integration improves relational coherence and retrieval
quality in RAG systems [10, 11, 80].5.3.1 Information Retrieval via Symbolic & Controlled
Fallback
Thesymbolicretrievalprocessbeginsbytranslatinganatural-
language cybersecurity query into an executable Cypher
query. To improve the reliability of query generation, prompt
engineeringanddomainadaptationareemployedtoconstrain
the generated queries to valid node labels, relationship types,
andpropertynamesdefinedintheBRIDG-ICSontology.The
generated Cypher query is subsequently executed over the
knowledge graph to retrieve graph-grounded evidence for
downstream reasoning.
A neuro-symbolic retrieval mechanism is adopted to
integrate ontology-guided symbolic reasoning with neural
language understanding. Given a user query 𝑞, a Cypher-
generatinglanguagemodelproducesanexecutablequery 𝑐
conditioned on the domain ontology. This symbolic query
is treated as theprimary retrieval pathway, ensuring that all
retrieved evidence is grounded in the structured semantics of
theknowledgegraph.Executingthisqueryovertheknowledge
graphGyields an evidence set 𝑅=EXEC(𝑐,G) , where𝑅
consistsofontology-groundednodes,edges,andmulti-hop
paths that provide interpretable and traceable evidence for
downstream reasoning.
TheCypher-generatingLLMreceivesastructuredprompt
consisting of three components: (i) a compact representation
of the BRIDG-ICS ontology schema, including node labels,
relationship types, and property definitions; (ii) explicit in-
structions requiring the model to generate only executable
Cypher statements using ontology-defined node labels, rela-
tionship types, and properties, without introducing entities
or relations outside the BRIDG-ICS schema; and (iii) the
analyst’snatural-languagequery.ThegeneratedCypherquery
issyntacticallyvalidatedbeforeexecutionovertheknowledge
graph.Thisontology-guidedpromptingstrategyconstrains
symbolic retrieval to valid graph structures while reducing
invalid schema references and unsupported query generation.
However,thegeneratedquery 𝑐maybeunusabledueto
syntacticerrors,executionfailures,orinvalidschemarefer-
ences.Toensurerobustness,acontrolledembedding-based
fallbackmechanismisactivatedonlywhenCypherexecution
fails. Let𝑓emb:V→R𝑑denote a node embedding function.
The query𝑞is embedded into the same space, and cosine
similarityiscomputedagainstallnodeembeddings.Themost
relevant anchor node is selected according to Equation (2):
ˆ𝑣=arg max
𝑣′∈Vcosine(𝑓 emb(𝑞),𝑓emb(𝑣′))(2)
which is then used to guide the regeneration of a valid
symbolic query.This design preserves interpretabilitywhile
improvingrobustness,aligningwithrecentneuro-symbolic
RAG approaches [ 10,11,23]. Consequently, the embedding-
based retrieval module does not replace symbolic graph

10
reasoning. Instead, it serves solely as a recovery mechanism
thatre-establishesgraph-groundedretrievalwhensymbolic
Cypher execution fails, ensuring that all downstream reason-
ing remains grounded in explicit knowledge graph traversal.
5.3.2 Hybrid Information Retrieval and Implementation
The proposed retrieval framework operationalizes the above
neuro-symbolic mechanism within a practical system set-
ting. While symbolic Cypher queries provide precise and
interpretable retrieval, real-world queries often vary in struc-
ture, length, and semantic focus. In particular, CVE nodes
containheterogeneousattributes(e.g.,identifier,riskscore,
description, Common Vulnerability Scoring System (CVSS)
metrics), and user queries may target any subset of these
fields,makingconsistentsymbolicquerygenerationchalleng-
ing. In such cases, the embedding-based fallback mechanism
improves retrieval robustness by capturing semantic intent.
This is particularly effective for long or descriptive queries
that lack explicit identifiers. For example, a query such as
“Whattypeofvulnerabilityallowsattackerstoinjectmalicious
scriptsintowebpages?”canbesemanticallymappedtonodes
associated with cross-site scripting (e.g., CVE-2021-XXX),
whichthenserveasanchorsforsymbolicqueryrefinement
and subsequent multi-hop graph traversal.
Once the most relevant anchor node is identified through
embeddingsimilarity,itisnotreturneddirectlyasthefinal
answer. Instead, the original user query together with the
recovered graph anchor are supplied back to the Cypher gen-
eration module, which regenerates an ontology-compliant
Cypher query grounded on the identified entity while pre-
serving the user’s original reasoning intent. The regenerated
Cypherqueryissubsequentlyexecutedovertheknowledge
graphusingthesamesymbolicqueryexecutionpipelineas
the primary reasoning process, enabling both one-hop and
multi-hoptraversalacrossconnectedcybersecurityentities.
Thistraversalretrievescomprehensivegraph-groundedevi-
dence,includingnodeattributes(e.g.,descriptions,severity
scores, and CVSS metrics) as well as connected entities such
asweaknesses,attacktechniques,andmitigations.Theresult-
ingsubgraphservesasstructuredevidencefordownstream
answergeneration,enablingexplainableanalysisofvulner-
abilities,attackpaths,vulnerabilitypropagation,ATT&CK
technique attribution, and mitigation recommendations. Con-
sequently, the embedding module functions solely as a re-
covery mechanism for identifying the initial graph anchor,
while all subsequent reasoning remains grounded in explicit
symbolic graph traversal.
CypherPromptEngineering.Toimproveretrievalrobust-
ness,theCypher-generatingLLMproducesuptothreecan-
didate queries for each analyst request. Duplicate and syn-
tactically invalid queries are discarded before execution over
the labelled property graph, where cybersecurity entitiesare represented as labelled nodes, relationships as typed
edges,andrelevantattributesasgraphproperties.Forlengthy
or semantically complex analyst queries, the input is de-
composed into multiple semantic segments using CONTAINS
filters, allowing relevant ontology concepts to be queried
independently while remaining within the context window of
the language model. Fine-tuning further improves the genera-
tionofontology-compliantCypherqueries,particularlyfor
multi-hopreasoninginvolvingheterogeneouscybersecurity
entities and relationships. Details of the fine-tuning proce-
durearepresentedinSection6.1.Theretrievalworkflowis
summarised in Algorithm 1, while the prompt segmentation
strategy is described in Algorithm 2.
Design Considerations.The hybridframework is informed
by threekey considerations: (i)token limitations,necessitat-
ing compact ontology representations; (ii)query robustness,
where embedding-based fallback compensates for failures
insymbolicquerygeneration;and(iii)accuracy–efficiency
trade-off,balancingprecisegraph-basedretrievalwithflexible
semantic matching.
5.3.3 Two-Stage Reasoning
The reasoning process follows a two-stage KG-RAG pipeline
that separatesgraph-based retrieval from answer genera-
tion.Inthefirststage,theneuro-symbolicretrievalmodule
extractsarelevantsubgraphfromtheknowledgegraph,pro-
ducinganevidenceset 𝑅groundedinexplicitcyber–physical
relationships.
In the second stage, the retrieved subgraph, including
graph paths, node attributes, and relationship information, is
provided as structured context to a second language model
that performs only response synthesis. Unlike the Cypher-
generating LLM, this model is not fine-tuned and receives
an explicit instructionto generate responses grounded exclu-
sively on the retrieved graph evidence without introducing
unsupported external knowledge. This separation between
symbolic retrieval and natural-language generation improves
reasoning transparency while reducing hallucination, consis-
tent with prior work on knowledge-grounded reasoning in
RAG systems [13, 81].
5.4 Human–AI Collaborative Decision
Thefinaloutcomeoftheproposedframeworkisaninteractive
human–AI system that enables users to query complex cyber-
physical security knowledge through natural language. Users
cansubmithigh-levelquestions(e.g.,relatedtovulnerabilities,
attackpaths,orsystemrisks),whichareprocessedthroughthe
hybridretrievalandreasoningpipeline.Thesystemintegrates
symbolicgraph-basedreasoningwithembedding-supported
retrieval to generate accurate and context-aware responses
grounded in the knowledge graph.

11
Algorithm 1KG-RAG Retrieval with Symbolic and Embed-
ding Fallback
Require: User query𝑞, ontologyO, knowledge graph G=(V,E) ,
Cypher LLMLLM cypher, encoder𝑓 enc, top-𝑘parameter𝑘
Ensure:Retrieved evidence set𝑅for downstream answer synthesis
1:// Phase 1: Symbolic retrieval
2:ˆ𝐶←LLM cypher(𝑞,O)⊲Generate candidate Cypher queries
3: Deduplicate ˆ𝐶
4:for allˆ𝑐∈ ˆ𝐶do
5:𝑅←EXEC(ˆ𝑐,G)
6:if𝑅≠∅then
7:return𝑅
8:end if
9:end for
10:// Phase 2: Embedding-based fallback
11:z 𝑞←𝑓enc(𝑞)
12:for all𝑣 𝑖∈Vdo
13:z 𝑖←𝑓enc(𝑣𝑖)
14:𝑠 𝑖←sim(z 𝑞,z𝑖)⊲e.g., cosine similarity
15:end for
16:V 𝑘←TopK({(𝑣 𝑖,𝑠𝑖)},𝑘)
17:// Re-anchor symbolic retrieval
18:for all𝑣∈V 𝑘do
19:ˆ𝑐′←LLM cypher(𝑞,𝑣,O)
20:𝑅′←EXEC(ˆ𝑐′,G)
21:if𝑅′≠∅then
22:return𝑅′
23:end if
24:end for
25:return∅
Algorithm 2Prompt Segmentation and Cypher Generation
Require: Userquery𝑞,ontologyO,maximumsegments 𝑆max,Cypher
LLMLLM cypher
Ensure:Set of Cypher queries ˆ𝐶
1:iflength(𝑞)≤token thresholdthen
2:S←{𝑞}
3:else
4: Segment𝑞into phrasesS={𝑠 1,...,𝑠 𝑆}with𝑆≤𝑆 max
5:end if
6: Construct prompt𝑃using:
7: compact ontology schema fromO
8: Cypher construction instructions
9: segmented phrases wrapped withCONTAINSfilters
10: ˆ𝐶←LLM cypher(𝑃)
11: Deduplicate ˆ𝐶and discard syntactically invalid queries
12:return ˆ𝐶
The retrieved results are presented in an interpretable
manner, allowing users to explore relationships between vul-
nerabilities,weaknesses,andattacktechniques.Thissupports
a range of decision-making tasks, including attack analy-
sis,vulnerabilityidentification,andmitigationplanning.By
combining structured graph reasoning with language model
capabilities, the framework facilitates explainable and effi-
cient Human–AI collaboration for cybersecurity analysis.6 Evaluation
This section assesses GRICS in terms of benchmark per-
formance,componentcontributions,adversarialrobustness,
runtime efficiency, and explainability. Four configurations
arecomparedusingidenticalbenchmarkinputsandinference
settings: (i) Base KG-RAG, (ii) KG-RAG with fine-tuning
(KG-RAG+FT),(iii)KG-RAGwithembeddingfallback(KG-
RAG+EF), and (iv) Full GRICS. The analysis combines
CTI-Benchmark tasks with multi-hop reasoning experiments
to examine retrieval accuracy, robustness, computational per-
formance, ontology consistency, and reasoning traceability.
6.1 Experimental Setup
The experimental setup comprises the construction of the
KG-RAGquestion-answeringdataset,thefine-tuningstrategy
for the Cypher-LLM, the training configuration, and the
evaluation protocol for threat-centric retrieval and reasoning
tasks.
6.1.1 KG-RAG QA Dataset
A supervised KG-RAG question-answering (QA) dataset
was constructed to evaluate the effectiveness of the pro-
posed framework in supporting threat-centric reasoning. The
dataset wasderived from 65real-world CVEs selectedfrom
theBRIDG-ICSknowledgegraph,eachassociatedwith com-
pleteCVE→CWE→CAPEC→ATT&CKmappings.From
this set, multiple query instances were generated, resulting in
atotalof450samplesalignedwiththeBRIDG-ICSontology,
enabling structured and ontology-aware retrieval and reason-
ing. The dataset captures diverse cybersecurity reasoning
tasks,including:(i)inter-noderelationships(e.g.,CVE-CWE,
CWE-CAPEC),(ii)entitydependencyreasoning,(iii)vulner-
abilityandattacktechniqueexplanations,(iv)multi-hopgraph
traversal and attack path discovery, (v) mitigation-oriented
queries, (vi) cross-domain IT–OT relationships, and (vii)
Vulnerability Propagation Risk (VPR) analysis.
Structuralreasoningisencouragedwhileavoidingover-
fitting to specific identifiers through controlled paraphrasing
and identifier variation. Each CVE instance is associated
withmultipleparaphrasedqueryforms(e.g.,“WhichCVEs
are related to CWE-200” and “Find CVEs associated with
CWE-200”),whileCVEidentifiersmaybemodifiedorsub-
stituted(e.g., CVE-2024-30051 )topromotegeneralisation.
Despite these variations, all question–answer pairs remain
explicitlygroundedintheBRIDG-ICSschema,ensuringcon-
sistencybetweennaturallanguagequeriesandgraph-based
representations.
6.1.2 Model Training and Fine-Tuning
TheCyphergenerationmoduleisimplementedusingapre-
trained Llama-3.1-8B Text2Cypher model and adapted to

12
Table 2: Summary of KG-RAG QA fine-tuning dataset.
Category Samples
CWE and CVE inter-node queries 50
Entity relations and traversal 65
Entity explanations by identifier 150
Entity explanations by name 50
Multi-hop graph traversal 100
Path dependency analysis 35
Total 450
the cybersecurity domain through parameter-efficient fine-
tuningusingtheUnslothframework,whichprovidesmemory-
efficient optimisation for large language models. A total of
450curatedtrainingsamplesconstructedfromtheBRIDG-
ICSknowledgegraphwereusedtofine-tunetheCypher-LLM
for Cypher generation and graph retrieval tasks. During infer-
ence,few-shotpromptingisemployedtoimprovesyntactic
correctness and semantic consistency by constraining gener-
atedCypherqueriestovalidnodelabels,relationshiptypes,
and property names defined in the BRIDG-ICS ontology.
Specifically, the ontology schema is explicitly provided as
part of the prompt context, allowing the model to generate
ontology-compliantCypher queries by adapting to the sup-
plied graph schema during inference rather than requiring
thecompleteontologystructuretobeencodedinthemodel
parameters.
TheCypher-LLMisfine-tunedtoenhancethetranslation
of natural-language cybersecurity queries into executable
Cypher statements and to align the model with domain-
specificontologystructuresandquerypatterns.Incontrast,
the Answer-LLM remains prompt-based to preserve flexibil-
ityandavoidover-specialisationtoafixedresponseformat.
Formally, the Cypher generation process is defined by Equa-
tion (3):
ˆ𝑐=F𝜃(𝑞,O)(3)
where𝑞denotes the input natural-language query, O
represents the BRIDG-ICS ontology schema, F𝜃denotes the
fine-tuned Cypher generation model parameterised by 𝜃, and
ˆ𝑐is the generated executable Cypher query.
This objective encourages the generation of syntactically
validandsemanticallyconsistentCypherquerieswhilereduc-
ing reliance on extensive prompt engineering. Fine-tuning
further improves robustness under incomplete graph struc-
turesbyenablingalternativegraphtraversalstrategieswhen
direct relationships are unavailable. Pathfinding and traversal
operations are implemented using graph data science tech-
niques,enablingefficientmulti-hopreasoningandattack-path
explorationwithinthecybersecurityknowledgegraph.The
model converged with a final training loss of0.0052, indi-cating stable optimisation and effective alignment between
natural-language instructions and symbolic query generation.
Thecompleteprompttemplates,fine-tuningimplemen-
tation,anddatagenerationpipelinearepubliclyavailablein
the project repository.3
Table 3: Training configuration for Cypher-LLM fine-tuning.
Parameter Value
per device train batch size 1
gradient accumulation steps 4
num train epochs 5
learning rate2×10−5
bf16 True
logging steps 40
save strategy epoch
remove unused columns False
gradient checkpointing False
During inference, ontology-guided few-shot prompting
is used to improve the syntactic validity and semantic con-
sistencyofgeneratedCypherqueries.Thepromptstructure,
validationprocess,andevidence-groundedresponsesynthesis
are described in Section 6.1.3.
6.1.3 Prompt Engineering and Inference Constraints
GRICS uses separate prompts for the Cypher-LLM and
Answer-LLM. The Cypher-LLM translates natural-language
requestsintoexecutableCypherqueries,whereastheAnswer-
LLM synthesises responses from retrieved graph evidence.
TheCypherpromptincludestaskinstructions,therelevant
BRIDG-ICS schema, few-shot examples, and the analyst
query, while restricting generated queries to the supplied
node labels, relationships, and properties and producing
up to three candidates per request. These candidates are
evaluatedsequentiallyforsyntax,ontologyconformity,and
executability over the LPG, with the first valid query that
retrieves relevant evidence being selected. When symbolic
retrieval fails, embedding retrieval identifies semantically
relatedgraphanchors,whicharesuppliedtotheCypher-LLM
forquery regeneration;theregeneratedquerymuststillpass
validation and symbolic execution.
The Answer-LLM receives the original query and the
retrieved graph evidence. Its prompt restricts response gener-
ation to the supplied entities, relationships, properties, and
graph paths, ensuring that the final output remains grounded
and traceable.
3https://github.com/ahmadspm/Resellient-Industry-5.0
--kG-Digital-Twins

13
6.1.4 Training Configuration
Fine-tuning experiments were conducted on a workstation
equipped with an NVIDIA RTX 6000 Ada Generation GPU
(48GB VRAM). The model was trained using the Unsloth
optimisation framework with Low-Rank Adaptation (LoRA)-
basedadapters.Thetrainingconfigurationincludesasequence
lengthof2048tokensandaneffectivebatchsizeof4achieved
via gradient accumulation. The key hyperparameters are
summarised in Table 3.
6.1.5 Benchmark Datasets and Evaluation Protocol
The proposed framework was evaluated using the publicly
available CTI-Benchmark (CTIBench) [ 82], which provides
standardised cybersecurity reasoning tasks for evaluating
large language models. Three benchmark tasks were selected
inthisstudy:CTI-RCM2024,CTI-RCM2021,andCTI-ATE.
The CTI-RCM benchmarks evaluated the classification of
CVEdescriptionsintotheircorrespondingCWEcategories
using two temporal splits, while CTI-ATE evaluated the
identification of MITRE ATT&CK techniques associated
with software or malware entities.
Since CTI-Benchmark is originally designed for prompt-
basedquestionanswering,thebenchmarkpromptswerere-
formulated into natural-language questions compatible with
the proposed KG-RAG framework. For example, a vulner-
abilityclassificationpromptwasreformulatedas:“Whatis
theCWEassociatedwiththeCVEdescribedas‘...’?”This
reformulationenablestheCypher-LLMtotranslatenatural-
language questions into executable Cypher queries while
preserving the original benchmark objectives. Benchmark
evaluation focuses on the Cypher generation and symbolic
graph retrieval stages of GRICS. Specifically, the bench-
mark measures whether the Cypher-LLM correctly maps the
natural-language query to the corresponding cybersecurity
entity through graph retrieval. This evaluation protocol is
appropriate becausethe benchmarkground truthconsists of
asinglestructuredentity(e.g.,aCWEidentifierorMITRE
ATT&CKtechnique),whichisdeterminedbeforeresponse
synthesis. The subsequent Answer-LLM is responsible only
fortransformingtheretrievedgraphevidenceintoa human-
readable explanation and does not alter the retrieved entity.
Consequently,benchmarkperformancereflectsthecorrect-
nessofthegraph-groundedretrievalandreasoningprocess
rather than the natural-language generation stage.
The Base KG-RAG configuration employs prompt-based
Cyphergenerationtogetherwithsymbolicretrievaloverthe
BRIDG-ICS knowledge graph. The KG-RAG+FT configura-
tionadditionallyenablesCypherfine-tuning,KG-RAG+EF
enablestheembedding-basedfallbackmechanism,andFull
GRICS combines both components while maintaining the
same symbolic retrieval pipeline.Performance is evaluated using Accuracy, Precision, Re-
call, and F1-score for the CTI-Benchmark tasks. Adversarial
robustnessisevaluatedusingAttackSuccessRate(ASR)and
Tokens per Query (TPQ), while explainability is assessed
usingHallucinationRate(HR),QueryViolationRate(QVR),
andSchema ConsistencyRate(SCR). Identicalprompts, in-
ferencesettings,andevaluationmetricsaremaintainedacross
allconfigurationstoensurefairandreproduciblecomparison.
6.2 Benchmark Performance Comparison
This section evaluates the performance of GRICS on three
tasks from the CTI-Benchmark: CTI-RCM 2024, CTI-RCM
2021,andCTI-ATE.TheCTI-RCMtasksevaluatetheability
of the framework to associate vulnerability descriptions with
theircorrespondingCommonWeaknessEnumeration(CWE)
categories, whereas CTI-ATE evaluates the identification of
MITRE ATT&CK techniques associated with software or
malware entities. Collectively, these tasks assess ontology-
aware entity retrieval, structured cybersecurity knowledge
reasoning, and graph-grounded threat analysis.
Table4presentstheperformanceoftheevaluatedGRICS
configurationsacrossthethreebenchmarktasks.FullGRICS
achieves the highest performance on all three datasets, ob-
taininganaccuracyof87.60%andanF1scoreof0.930on
CTI-RCM2024,anaccuracyof90.40%andanF1scoreof
0.949onCTI-RCM2021,andanaccuracyof77.15%withan
F1scoreof0.871onCTI-ATE.TheresultsshowthatGRICS
performsstronglyonthetwovulnerability-to-weaknessclassi-
ficationtasks,whilethelowerCTI-ATEperformancereflects
thegreaterrelationalcomplexityofATT&CKtechniqueiden-
tificationacrosssoftware,malware,attackpatterns,vulnera-
bilities,andtechniques.Nevertheless,FullGRICSremains
effectivefordeepergraphtraversalandheterogeneousreason-
ing.Precisionis100%acrossallconfigurations,indicating
that valid predictions retrieve the correct entity; therefore,
differences in accuracy, recall, and F1 score mainly reflect
each configuration’s ability to resolve queries and retrieve
sufficient graph-grounded evidence.
The benchmark results demonstrate that Full GRICS
performs effectively across vulnerability classification and
ATT&CK technique identification. The following ablation
study further examines the contributions of graph grounding
and the individual architectural components.
Statistical Significance.Pairwise statistical comparisons
were performed using McNemar’s test at a significance level
of𝛼=0.05foreachofthethreebenchmarktasks:CTI-RCM
2024,CTI-RCM2021,andCTI-ATE.Oneachbenchmark,
thepublished BaseLLM was comparedseparately withBase
KG-RAG, KG-RAG+FT, KG-RAG+EF, and Full GRICS.
TheresultsshowedthatBaseKG-RAGsignificantlyoutper-
formed the Base LLM across CTI-RCM 2024, CTI-RCM

14
Table4:Component-wiseevaluationofGRICSontheCTI-
Benchmark.
CTI-RCM 2024
Model Acc. Prec. Rec. F1
Base KG-RAG 62.80 100 62.80 0.771
KG-RAG + FT 67.20 100 67.20 0.800
KG-RAG + EF 85.70 100 85.70 0.923
Full GRICS87.6010087.60 0.930
CTI-RCM 2021
Base KG-RAG 66.50 100 66.50 0.798
KG-RAG + FT 78.60 100 78.60 0.880
KG-RAG + EF 88.20 100 88.20 0.937
Full GRICS90.4010090.40 0.949
CTI-ATE
Base KG-RAG 18.33 100 18.33 0.309
KG-RAG + FT 64.72 100 64.72 0.786
KG-RAG + EF 53.33 100 53.33 0.695
Full GRICS77.1510077.15 0.871
2021, and CTI-ATE ( 𝑝<0.05), demonstrating the benefit
of symbolic knowledge-graph retrieval. KG-RAG+FT also
achieved statistically significant improvements over the Base
LLM on all three benchmarks ( 𝑝<0.05), indicating the
contribution of domain-specific Cypher fine-tuning. Simi-
larly,KG-RAG+EFsignificantlyoutperformedtheBaseLLM
acrossthethreetasks( 𝑝<0.05),highlightingtheeffectiveness
of embedding-assisted query recovery. Full GRICS achieved
the strongest performance and significantly outperformed
the Base LLM on CTI-RCM 2024, CTI-RCM 2021, and
CTI-ATE (𝑝<0.05), demonstrating the combined benefit of
symbolic graph retrieval, Cypher fine-tuning, and CC BY:
Creative Commons Attribution CC BY-SA: Creative Com-
monsAttribution-ShareAlikeCCBY-NC-SA: Cembedding
fallback.
6.2.1 Ablation Study
Theablationstudyanalysesthecontributionofgraph-grounded
symbolicretrieval,Cypherfine-tuning,andembedding-based
fallback.SincegraphgroundinginGRICSisrealisedthrough
symbolic Cypher execution over the BRIDG-ICS knowl-
edge graph, it cannot be removed while preserving an exe-
cutableKG-RAGconfiguration.Therefore,thecontributionof
graphgroundingisassessedbycomparingthepublishedCTI-
BenchmarkLLMbaseline,whichoperateswithoutknowledge
graph grounding,with Base KG-RAG. It usesprompt-based
Cyphergenerationfollowedbysymbolicexecutionoverthe
BRIDG-ICS knowledge graph. The difference between the
published LLM baseline and Base KG-RAG therefore repre-
sents the contribution of graph-grounded symbolic retrieval.
TheremainingconfigurationsprogressivelyintroduceCypher
fine-tuning(KG-RAG+FT),embedding-basedfallback(KG-
RAG + EF), and their combination in Full GRICS.
Figure 3 illustrates the contribution of each component
across the three benchmark tasks. Compared with the pub-lishedLLMbaseline,BaseKG-RAGdemonstratesthebenefit
ofgraph-groundedretrieval,whileCypherfine-tuningfurther
improves query validity and semantic alignment, particularly
on the more relationally complex CTI-ATE task.
Theembedding-basedfallbackmechanismimprovesre-
trieval coverage when the initial Cypher query is invalid,
overly restrictive, or produces no relevant result. Importantly,
the fallback mechanism does not independently generate the
final answer or replace symbolic graph reasoning. Instead,
it identifiessemantically relatedgraph anchorsand supplies
them to the Cypher generation module, which regenerates
anontology-compliantquerybeforesymbolicexecutionre-
sumes.
Full GRICS achieves the strongest performance across
all three benchmark tasks, demonstrating that Cypher fine-
tuningandembedding-basedfallbackaddresscomplementary
limitations. Fine-tuning improves the quality of the initial
symbolicquery,whereasembedding-assistedrecCCBY:Cre-
ativeCommonsAttributionCCBY-SA:CreativeCommons
Attribution-ShareAlike CC BY-NC-SA: Covery improves
robustness when the initial retrieval attempt is unsuccess-
ful.TheseablationresultsshowthatGRICS’sperformance
gainsstemfromthreecomplementarycapabilities:explicit
knowledgegraphgrounding,task-specificCyphergeneration,
and embedding-assisted query recovery. Comparison with
thepublishedCTI-BenchmarkLLMbaselinequantifiesthe
impact of graph-grounded symbolic retrieval, while inter-
nalconfigurationcomparisonsisolatetheaddedbenefitsof
Cypher fine-tuning and the embedding-based fallback.
Explainability Analysis.The explainability of GRICS is ex-
amined from two complementary perspectives: semantic cov-
erageandretrievaltransparency.WhileFigure3demonstrates
the performance contribution of the individual architectural
components,Figures4and5providefurtherinsightintohow
thesecomponentsinfluencethereasoningbehaviourofthe
framework.Figure4presentsCWE-levelpredictioncoverage
for the CTI-RCM 2024 and CTI-RCM 2021 benchmarks.
TheBaseKG-RAGconfigurationexhibitslimitedcoverage
and inconsistent predictions across several CWE classes. In
contrast, KG-RAG + FT produces broader semantic cover-
age and more consistent predictions, particularly for less
frequent and semantically complex CWE categories. This
improvement indicates that fine-tuning strengthens the align-
ment between natural-language vulnerability descriptions,
generatedCypherqueries,andthestructuredcybersecurity
concepts represented in the BRIDG-ICS knowledge graph.
Retrieval-level explainabilityis further analysed through the
retrieval outcome distributions shown in Figure 5. The distri-
butions distinguish queries resolved through direct symbolic
retrieval, queries recovered through the embedding-based
fallback mechanism, and unresolved queries.

15
(a) CTI-RCM 2024
 (b) CTI-RCM 2021
 (c) CTI-ATE
Fig.3:Component-wiseablationofGRICSontheCTI-Benchmarkdatasets.ThepublishedCTI-BenchmarkLLMbaseline
represents prediction without knowledge graph grounding. Base KG-RAG employs prompt-based Cypher generation and
symbolicgraphretrieval,+FTenablesCypherfine-tuning,+EFenablestheembedding-basedfallbackmechanism,andFull
GRICS combines both enhancements.
(a)CTI-RCM2024BaseKG-RAG
 (b)CTI-RCM2024KG-RAG+FT
(c)CTI-RCM2021BaseKG-RAG
 (d)CTI-RCM2021KG-RAG+FT
Fig. 4: Heatmap comparison of predictive accuracy across
thetop15CWEclassesforCTI-RCM2024andCTI-RCM
2021.
The fine-tuned configurations resolve a larger proportion
of queries through direct symbolic reasoning and produce
fewer unresolved outcomes. This behaviour improves reason-
ingtraceabilitybecausemorepredictionscanbeassociated
with an explicit Cypher query and a corresponding graph
traversalpath.Theembedding-basedfallbackmechanismpro-
videsanadditionalrecoverypathwhentheinitialsymbolic
queryisunsuccessful,butthefinalresultremainsgrounded
insymbolicgraphexecutionbecausetheretrievedsemantic
anchorisusedtoregenerateanontology-compliantCypher
query.Theseresultsshowthatthearchitecturalenhancements
improve more than predictive performance. Cypher fine-
tuningincreasessemanticcoverageandtheshareofqueries
resolved by direct graph traversal, while embedding-assisted
recovery reduces unresolved cases without replacing sym-
bolicreasoning.GRICSthusoffersaninterpretablereasoning
pipelinewhereretrievedevidenceistracedtoexplicitentities
and relationships in the BRIDG-ICS knowledge graph.Semantic Stability of GE Embeddings.The semantic stabil-
ity of the graph embeddings used by the fallback mechanism
is evaluated when the initial symbolic retrieval attempt is
unsuccessful. Semantic stability refers to the consistency
ofsimilarity-basedcandidaterankingsandthepreservation
of meaningful neighbourhood structures in the embedding
space.Allgraph-nodeandqueryembeddingsweregenerated
using the CTI-RCM model. Three complementary analy-
ses are conducted: similarity-score decay across retrieval
ranks, similarity margins between the two highest-ranked
candidates,andatwo-dimensionalprincipalcomponentanal-
ysis(PCA)projection.Theseanalysesrespectivelyexamine
rank-basedrelevancedecay,candidateseparation,andlocal
neighbourhood structure.
Figure6ashowsacleardeclineincosinesimilarityacross
retrieval ranks, based on 20 queries from each of the CTI-
RCM 2021 and CTI-RCM 2024 datasets, indicating that
the embedding model consistently prioritises a small set
of semantically relevant graph entities. Figure 6b further
evaluatescandidateseparationusingthemargin Δ=𝑠 top1−
𝑠top2, where smaller margins for CTI-RCM 2021 indicate
greater ambiguity between the highest-ranked candidates,
while larger margins for CTI-RCM 2024 suggest stronger
separation and more reliable graph-anchor selection.
ThePCAprojectioninFigure6cprovidesaqualitativerep-
resentationoftheembeddingneighbourhoodforCTI-RCM
2024. The query embedding appears near its highest-ranked
graph nodes and forms a coherent local cluster with semanti-
callyrelevantentities.AlthoughPCAdoesnotpreserveexact
distancesfromtheoriginalhigh-dimensionalspace,theob-
servedclusteringprovidesqualitativeevidencethatrelevant
graph nodes occupy a similar semantic neighbourhood to the
query. The apparent proximity of some background nodes in
the PCA projection does not contradict the retrieval rankings
becausecandidateselectionisperformedusingcosinesimi-
larity in the original embedding space rather than Euclidean
distance in the two-dimensional projection.

16
(a) CTI-RCM 2024
 (b) CTI-RCM 2021
 (c) CTI-ATE
Fig. 5: Retrieval outcome distributions across the CTI-RCM 2024, CTI-RCM 2021, and CTI-ATE benchmarks, showing
symbolic-onlyretrievals,embedding-assistedfallbackretrievals,andunresolvedqueriesfortheevaluatedGRICSconfigurations.
(a) Similarity decay across retrieval rank.
 (b) Similarity margin in embedding-based fall-
back retrieval.
(c) PCA projection of GE embeddings.
Fig. 6: Semantic stability analysis of the graph embeddings used by the GRICS fallback retrieval mechanism.
Based on the observed similarity decay and candidate-
marginseparation,fallbackretrievalinGRICSisrestricted
to the three highest-ranked graph nodes. The results indicate
diminishing semantic relevance beyond these candidates.
Limiting retrieval to the top three nodes therefore reduces
the introduction of noisy graph anchors while preserving the
candidates most likely to support successful ontology-aware
Cypher regeneration.
6.3 Robustness Against Adversarial Attacks
Evaluating the robustness of KG-RAG systems against adver-
sarial attacks remains challenging due to the absence of stan-
dardisedbenchmarksandthediversityofknowledgegraph
structures, ontology designs, and reasoning tasks adopted
acrossexistingstudies.ToassesstherobustnessofGRICS,we
followtheevaluationmethodologyof[ 56]andadoptAttack
Success Rate(ASR) as the primary evaluation metric.For
untargetedattacks,AttackSuccessRate(ASR)iscomputed
using Equation (4):
ASR=Í
(𝑥,𝑦)∈𝑋 ⊮ˆ𝑦≠𝑦
|𝑋|.(4)
where𝑋denotes the set of evaluation queries, 𝑦is the
ground-truthanswer, ˆ𝑦isthemodelpredictionunderadver-
sarial perturbation, and ⊮(·)is the indicator function thatreturns 1 when the attack successfully changes the prediction
and0otherwise.AlowerASRindicatesgreaterrobustness,
as fewer adversarial attacks successfully alter the model’s
prediction.
Tokens per Query (TPQ) is computed according to Equa-
tion (5):
TPQ=#tokens in𝐷 poison
|𝑋|.(5)
where𝐷poisonrepresents the injected adversarial content
and|𝑋|is the number of evaluated queries. TPQ measures
the average number of adversarial tokens injected per query.
Larger TPQ values indicate that an attacker must inject more
adversarialcontenttoinfluencetheretrievalprocess,whereas
lower TPQ values indicate that successful attacks require
fewer injected tokens.
Following theexperimental methodology of [ 83], a total
of120representativemulti-hopreasoningquerieswereevalu-
atedovertheBRIDG-ICSontology.Toimprovecomparability
with PoisonRAG and GRAPPOISON, the robustness eval-
uationwasrestrictedtoATT&CK-orientedreasoningtasks
involving malware, ATT&CK techniques, and mitigation re-
lationships, reflecting the common evaluation scope adopted
inpriorgraph-poisoningstudies.Eachquerywassubjected
toadversarialperturbationsfollowingtheprompt-injection
strategy described in [56].

17
Theperturbationsweregroupedintolexicalandseman-
tic attack categories to assess whether different forms of
input manipulation affect GRICS differently. Lexical attacks
modifythesurfaceformofthequerythroughtoken-orphrase-
level changes intended to influence query interpretation. Ex-
amples include inserting misleading keywords, replacing
entity-relatedterms,ormodifyinglocalphrasingaroundan
ATT&CK technique or mitigation entity. Semantic attacks
insteadmodifythecontextualmeaningorinstructionstruc-
ture of the query while preserving the original cybersecurity
reasoningtarget.Forexample,asemanticperturbationmay
introduce conflicting instructions or misleading contextual
statementsintendedtoredirectthemodeltowardanincorrect
ATT&CKtechnique,mitigation,orgraphrelationshipdespite
theunderlyingqueryobjective.Thesetwocategoriesthere-
forecapturecomplementaryadversarialbehaviours:lexical
attackstargetsurface-levelinterpretation,whereassemantic
attacksattempttomanipulatethehigher-levelreasoningintent
of the generated query.
Category-specific robustness was measured using Attack
Success Rate (ASR). GRICS achieved an ASR of approx-
imately 68.3% under lexical adversarial perturbations and
74.9% under semantic adversarial perturbations, yielding an
aggregate ASR of 71.6% across the complete adversarial
evaluation set. The higher ASR observed under semantic per-
turbationsindicatesthatcontext-levelmanipulationpresentsa
greater challenge to the reasoning pipeline than surface-level
lexical modification. Tokens per Query (TPQ) was addition-
ally used to characterize the amount of adversarial content
required during the aggregate attack evaluation.
For broader context, Table 5 reports the robustness re-
sults of GRICS alongside representative PoisonRAG and
GRAPPOISON results reproduced from [ 56]. Because these
approaches were evaluated using different knowledge graphs,
graph-construction procedures, adversarial implementations,
and evaluationprotocols, thecomparison isintended topro-
vide qualitative context rather than a controlled head-to-head
assessment. Although the robustness evaluation of GRICS
was restricted to an ATT&CK-oriented reasoning scope to
better align with prior studies, the underlying BRIDG-ICS
knowledge graph differs from the knowledge graphs used by
PoisonRAG and GRAPPOISON in terms of ontology design,
graphconstruction,andrelationalstructure.Consequently,the
reportedASRandTPQvaluesshouldbeinterpretedaspro-
vidingqualitativeinsightintotherobustnesscharacteristics
of graph-grounded retrieval systems rather than establishing
directquantitativesuperiority.Acontrolledcomparisonusing
identical knowledge graphs, attack settings, and evaluation
protocols remains an important direction for future work.Table 5: Qualitative comparison of adversarial robustness.
ResultsforPoisonRAGandGRAPPOISONarereproduced
from [56] and were obtained under different knowledge
graphs, reasoning tasks, and attack settings.
Method ASR (%) TPQ
PoisonRAG 63.2 184.50
GRAPPOISON 96.9 103.80
GRICS 71.6 96.72
6.3.1 Robustness to Noisy Analyst Inputs
In addition to intentional adversarial manipulation, practical
cyber-threatintelligencesystemsmusttoleratenon-malicious
imperfections in analyst queries. Such inputs may contain
spelling errors, incomplete grammatical structures, or incon-
sistentformattingofcybersecurityidentifierswhilepreserving
the same underlying analytical intent. The noisy-input analy-
sisthereforeconsiderstwosettings:natural-languagenoise,
combiningtypographicalandgrammaticalperturbations,and
entity-formatnoise.Unliketheadversarialperturbationseval-
uated in the previous subsection, these modifications are not
intendedtoredirectthereasoningprocesstowardanincorrect
answer, but instead simulate realistic imperfections that may
occur during analyst interaction.
For typographical and grammatical noise, the CTI-RCM
2024 benchmark was selected because its vulnerability de-
scriptions contain substantial natural-languagecontext, mak-
ingitsuitableforevaluatingrobustnesswhenthelinguistic
formulationofaqueryisdegraded.Theperturbationsintro-
ducespellingerrors,character-levelmodifications,omitted
words,andgrammaticallyincompleteorirregularsentence
structureswhilepreservingtheoriginalcybersecurityreason-
ing target. For example, a correctly formulated vulnerability-
relatedquerymaybemodifiedthroughmisspelledsecurity
terminology or fragmented grammatical structure without
changing the expected CWE classification.
Table 6 reports the performance of the evaluated configu-
rations under these combined natural-language perturbations.
Base KG-RAG achieved an accuracy of 46.3%, and the ad-
dition of fine-tuning alone resulted in the same accuracy
of 46.3%. This behaviour can be explained by the fact that
fine-tuningimprovesthemodel’sabilitytogenerateontology-
compliant Cypher queries, but it does not correct incorrect
or corrupted information contained in the input itself. Con-
sequently, when typographical or grammatical perturbations
distort the entity or relation expressed in the query, the gener-
ated Cypher statement may still be structurally valid while
referring to an entity or pattern that does not exist in the
knowledge graph, resulting in an empty or incorrect retrieval.
KG-RAG+EF achieved an accuracy of 82.7%, while Full
GRICSachievedasimilaraccuracyof83.8%.Thesubstan-
tial improvement over the configurations without embedding

18
fallback indicates that semantic graph-anchor recovery is
the primary mechanism responsible for robustness under
noisy-input conditions.When typographical orgrammatical
perturbationspreventtheinitialsymbolicqueryfromreliably
identifying the intended graph entity, the embedding mecha-
nismcanrecoverasemanticallyrelatedanchor,afterwhich
an ontology-compliant Cypher query is regenerated and sym-
bolic retrieval resumes. The small performance difference
between KG-RAG+EF and Full GRICS is associated with
casesinvolvingomittedwordsorfragmentedphrasing,where
the fine-tuned Cypher-LLM may generate a more suitable
alternativeinterpretationorselectamoreinformativequery
phrase from the remaining context. In some instances, this
canhelpFullGRICSrecoveravalidsymbolicquerythatKG-
RAG+EFdoesnotproduce.However,becausethiseffectis
limitedandmaydependonthemodel’sgenerationbehaviour,
bothconfigurationsexhibitbroadlycomparablerobustness,
with embedding-assisted recovery remaining the dominant
mechanism under noisy-input conditions.
Table 6: Robustness to combined typographical and gram-
matical noise on the CTI-RCM 2024 benchmark.
Configuration Accuracy (%)
Base KG-RAG 46.3
KG-RAG + FT 46.3
KG-RAG + EF 82.7
Full GRICS 83.8
Meanwhile, entity-format noise was evaluated separately
using the one-hop question set employed in both the runtime
analysis(Section 6.4)andthe explainabilityand traceability
analysis(Section6.5).Thissetting examinescasesinwhich
a cybersecurity identifier or entity name is expressed using a
non-canonical format, including omitted separators, spacing
variations, capitalization differences,or minor textualerrors.
Forentity-formatperturbations,BaseKG-RAGandKG-
RAG+FT achieved 0% successful resolution, whereas KG-
RAG+EF and Full GRICS correctly resolved 100% of the
evaluated one- to three-hop queries when the intended identi-
fierremainedrecoverablethroughsemanticsimilarity.This
differencearisesbecausetheentityidentifierfrequentlyserves
astheprincipalmatchingtermintheCypher WHEREclause.
When the identifier is malformed or expressed in a non-
canonicalformat,BaseKG-RAGandKG-RAG+FTcannot
recoverthecorrespondinggraphentityandthereforefailto
returnrelevantresults.Incontrast,configurationsequipped
withembeddingfallbackcanidentifythesemanticallyclosest
graph entity, use it as an anchor, and regenerate an ontology-
compliant Cypher query before symbolic execution resumes.
The recovered embedding candidate is therefore used only
to reconstruct the query and is not accepted directly as fi-
nal evidence. For reasoning paths beyond three hops, theincreased relational complexity may lead to incomplete or
hallucinated query structures; this behaviour is examined
further in Section 6.5.
A different behaviour occurs when the numerical com-
ponent of an identifier is modified. Changing the numeric
portionofaCVE,CWE,CAPEC,orsimilaridentifiermayre-
fer to a different valid cybersecurity entity rather than merely
representing formatting noise. In this situation, GRICS does
not assume that the altered identifier corresponds to the
originallyintended entityandmaythereforereturn informa-
tion associated with the identifier actually supplied in the
query. This distinction prevents semantic fallback from incor-
rectly overriding potentially valid cybersecurity identifiers.
Thesefindingsconfirmthatembedding-assistedrecoveryis
themain mechanismsupportingGRICS robustnessto noisy
natural-language and non-canonical entity inputs.
6.3.2 Uncertainty-Aware Graph Reasoning
In practice, complete deterministic mappings are not always
availableacrossthevulnerability-to-MITREATT&CKrea-
soning chain. For example, not every CVE is associated with
a CWE, not every CWE is linked to a CAPEC attack pat-
tern, and not every CAPEC entry has a complete mapping
to a MITRE ATT&CK technique. As a result, a multi-hop
reasoning path may become incomplete even though other
potentially relevant associations are available.
The BRIDG-ICS ontology addresses this limitation by
explicitly preserving candidate associations through rela-
tionship types such as HAS_POSSIBLE_[NODE_NAME] [23].
Theserelationsrepresentpotentiallyrelevantmappingswhen
a definitive relationship is unavailable. In this way, incom-
plete mappings are handled within the ontology itself rather
than requiring the reasoning framework to infer unsupported
relationships.
GRICS incorporates these uncertainty-aware relation-
shipsdirectlyintoitsmulti-hopreasoningprocess.Whena
query traverses the CVE–CWE–CAPEC–MITRE ATT&CK
chain,thegenerated Cypherqueriescanretrieveboth defini-
tive relationships and corresponding HAS_POSSIBLE_* re-
lationships when they are represented in the graph. For
example, if a CVE does not have a definitive CWE mapping
butisconnectedthrough HAS_POSSIBLE_CWE ,theassociated
possible CWE entities can still be returned as part of the
graph-grounded evidence. Because GRICS generates mul-
tipleontology-compliantCyphercandidates,differentvalid
relationship patterns can be explored within the same analyst
request.
Theembedding-fallbackmechanismfurthersupportsthis
process when the initial query cannot reliably identify the
intended graph anchor. Once a relevant graph entity is re-
covered, GRICSregeneratesan ontology-compliantCypher
query and retrieves the available direct and possible relation-

19
ships associated with that entity. Importantly, the embedding
mechanismisusedonlytorecoverarelevantgraphanchor;
it does not create or infer a missing relationship. The final
evidenceremainsrestrictedtorelationshipsexplicitlyrepre-
sentedinBRIDG-ICS.Robustnesstoincompletemappings
is therefore supported through the framework’s ability to
preserve and expose ontology-defined possible relationships
during normal multi-hop retrieval. These possible associa-
tionsremainexplicitlyidentifiedbytheirrelationshiptypes
and are not presented as confirmed mappings.
The practical purpose of this mechanism is to support an-
alysts when a complete reasoning path is unavailable. Rather
than terminating the investigation at the first missing de-
terministic mapping or requiring the analyst to restart the
analysisfromtheoriginalvulnerability,GRICScanexpose
theavailablecandidaterelationshipsandrelatedentities.This
allows the analyst to continue investigating relevant weak-
nesses, attack patterns, and MITRE ATT&CK techniques
whileretainingvisibilityofwhichrelationshipsareconfirmed
and which remain possible.
6.4 Runtime Performance Analysis
Inadditiontoreasoningaccuracy,thepracticaldeployment
of GRICS depends on its computational efficiency for real-
worldcyber threatinvestigations.Runtimeperformance was
evaluated using 100 representative queries sampled from
the evaluation datasets, comprising 20 entity lookup queries,
20one-hopgraphtraversalqueries,40multi-hopreasoning
queries (covering both two-to-three-hop and four-to-five-hop
reasoning), and 20 CTI-Benchmark queries.
Theruntimeanalysisseparatelymeasuresthelatencyof
thethreeprincipalstagesoftheinferencepipeline:(i)Cypher
query generation, (ii) symbolic graph retrieval, and (iii)
natural-languageresponse synthesis.Timingmeasurements
wereperformedusingPython’s time.perf_counter() af-
ter all models had been loaded into GPU memory and initial-
ized. Consequently, the reported latency represents steady-
stateonlineinferenceandexcludesone-timemodelloading
and initialization overhead.
Table7summarizestheaveragelatencyforCyphergen-
eration and symbolic graph retrieval across representative
cybersecurityreasoningtasks.Entitylookupexhibitsthelow-
estlatencybecausequeriesareanchoredtoindexedentities,
enabling efficient Cypher generation and localized graph
traversal. As reasoning depth increases, both Cypher genera-
tionandsymbolicretrievalrequireadditionalprocessingtime
duetoincreasinglycomplexgraphtraversal.Attackpathanal-
ysis incurs the highest retrieval latency because substantially
larger graph neighbourhoods must be explored.
Inpractice,runtimeisdominatedbythetwoLLMinfer-
encestagesratherthangraphretrieval.DependingonqueryTable 7: Average latency across representative cybersecurity
reasoning tasks.
Query Category Cypher (s) Query Exec. (s)
Entity lookup 0.86 0.82
One-hop reasoning 1.39 0.67
Multi-hop (2–3 hops) 1.61 0.96
Multi-hop (4–5 hops) 1.94 1.42
CTI-Benchmark 2.32 1.87
Attack path analysis 2.58 6.42
LLM response synthesis 1.46 –
complexity, Cypher generation requires between 0.86 and
2.58seconds,whileresponsesynthesisrequiresapproximately
1.46secondsonaverage.Symbolicgraphretrievalcontributes
comparativelylittleoverheadformostquerytypes,andthe
embedding-basedfallbackmechanismintroduces onlymin-
imal additional latency because semantic similarity search
is performed over precomputed node embeddings and is
activated only when symbolic retrieval fails.
Although the current evaluation was conducted on the
BRIDG-ICSknowledgegraph,thearchitectureisdesigned
to support larger industrial knowledge graphs, although its
performanceatsubstantiallygreaterscaleremainstobeempir-
ically validated. Symbolic retrieval performs localized graph
traversalanchoredbyidentifiedentitiesratherthanexhaustive
graphexploration,makingqueryexecutiondependentprimar-
ily on reasoning depth and the retrieved subgraph rather than
thetotalgraphsize.Furthermore,nodeembeddingsaregener-
atedofflineandreusedduringinference,allowingembedding-
based retrieval to scale independently of LLM inference.
Futuredeploymentsonmillion-scaleknowledgegraphscould
further benefit from approximate nearest-neighbour indexing
and distributed graph databases.
Table 8 summarizes the steady-state runtime memory
utilisation after model initialization. The majority of GPU
memory is occupied by the fine-tuned Llama-3.1-8B model,
whereastheNeo4j knowledgegraphand precomputednode
embeddingsresideinhostmemory.Sincenodeembeddings
aregeneratedoffline,embedding-basedretrievalintroduces
only a modest memory overhead while enabling efficient
semantic retrieval.
GRICS introduces an additional LLM inference stage for
ontology-awareCyphergenerationbeforeresponsesynthesis.
Thisincreasesinferencecostandlatency,buttheadditional
computationenablestheanalystquerytobeconvertedinto
an executable symbolic graph query prior to answer gen-
eration. The resulting trade-off is therefore between lower
computationaloverheadandstrongerontologycompliance,
graph-grounded verification, and evidence traceability.

20
Table 8: Runtime memory utilisation of GRICS.
Component Runtime Memory
Llama-3.1-8B (4-bit)∼6.3 GB GPU
Embedding model (MiniLM-L6-v2)<100 MB
Neo4j knowledge graph 1.9 GB Host RAM
Node embeddings (precomputed)∼45 MB RAM
6.4.1 Computational Complexity and Scalability
Thecomputational costofGRICS canbeconsideredacross
four principal components: Cypher generation, symbolic
graph retrieval, embedding-assisted fallback retrieval, and
natural-language response generation. Let 𝑁=|V|denote
the number of nodes in the knowledge graph, 𝑑the embed-
dingdimensionality,and ℎthereasoning depthof thegraph
query. The two LLM-based stages, Cypher generation and
response synthesis, are primarily influenced by model size
andinput/outputsequencelengthandremainthedominant
contributorstoinferencelatency,asreflectedintheruntime
measurements reported above.
For symbolic retrieval, GRICS does not perform exhaus-
tive traversal of the complete knowledge graph for each
request. Generated Cypher queries are anchored to identi-
fied cybersecurity entities and retrieve only the relationships
requiredbytherequestedreasoningpath.Consequently,prac-
tical graph-query cost is influenced mainly by the size and
density of the local neighbourhood, relationship branching
factor, and reasoningdepth ℎ, ratherthan by directtraversal
ofall𝑁graphnodes.Astheknowledgegraphgrows,retrieval
latency is therefore expected to depend on whether the addi-
tional entities and relationships increase the neighbourhood
explored by a particular query. This behaviour is reflected
in the current runtime results, where deeper multi-hop and
attack-pathqueriesexhibitgreaterretrievallatencybecause
larger relational neighbourhoods are traversed.
Theembedding-assistedfallbackmechanismintroducesa
separate scalability consideration. Node embeddings are gen-
erated offline and reused during inference, avoiding repeated
embeddingcomputationduringnormalqueryprocessing.For
𝑁graphentitiesrepresentedby 𝑑-dimensionalembeddings,
direct similarity comparison has an approximate computa-
tional cost of 𝑂(𝑁𝑑). As the graph grows, more efficient
vector-retrievalmechanismsmaythereforeberequiredtopre-
ventsemanticfallbacklatencyfromincreasingproportionally
with the number of embedded entities.
Embedding regenerationalso introducesan offline com-
putational cost as the knowledge graph evolves. A complete
regenerationrequiresembeddingthetextualrepresentation
of each graph entity and therefore increases approximately
withthenumberofentitiesbeingprocessed.However,routine
graph updates do not necessarily require regeneration of the
completeembeddingcollection;newlyintroducedormodi-fied entities can be embedded separately while unchanged
representations are retained. Consequently, embedding main-
tenance primarily affects offline graph-update operations
rather than the latency of every analyst query.
Memoryrequirementssimilarlyincreasewithknowledge-
graph size. In the current implementation, the knowledge
graphoccupiesapproximately1.9GBofhostmemoryandthe
precomputed node embeddings approximately 45 MB, while
thedominant GPU-memoryrequirement isthe 4-bitLlama-
3.1-8Bmodelatapproximately6.3GB,asreportedinTable8.
For a fixed embedding dimensionality, embedding storage
grows approximately linearly with the number of embedded
entities.Knowledge-graphstoragealsoincreaseswithboth
the number of entities and relationships, with relationship
densitybecomingparticularlyrelevantformulti-hopretrieval.
These observations suggest that increasing graph scale
primarilyaffectsthegraphandvector-retrievallayers,whereas
the memory required by the LLM remains largely indepen-
dent of the number of graph entities. The current evaluation
provides the measured computational baseline for BRIDG-
ICS;substantiallylargerindustrialknowledgegraphswould
require further empirical evaluation to determine the prac-
ticallatencyandmemorybehaviourunderenterprise-scale
workloads.
6.5 Explainability and Reasoning Traceability
Explainability in GRICS is evaluated from both quantitative
and operational perspectives. The quantitative evaluation
measureswhethergeneratedCypherqueriesremaingrounded
intheBRIDG-ICSontology,complywithquery-generation
constraints, and preserve structural validity as relational
complexityincreases.Theoperationalevaluationexamines
whethertheresultingreasoningpathscanbeinspectedand
verified by security analysts.
GRICSsupportstraceabilitythroughexplicitmulti-hop
reasoning chains grounded in verifiable knowledge-graph
entitiesandrelationships.Ratherthanproducingunsupported
conclusions, the framework exposes the generated Cypher
queries, retrieved entities, and traversed relationships as-
sociated with each result. This design allows analysts to
inspectintermediate reasoning steps,verifythesupporting
evidence,andcontextualizemodeloutputswithinoperational
Industry 5.0 security workflows.
6.5.1 Quantitative Explainability Metrics
Quantitative explainability is evaluated using a structured
query corpus derived from the KG-RAG question-answering
datasetdescribedinSection6.1.1.Thecorpusenablessystem-
atic analysis of structural validity and reasoning behaviour
across multi-hop queries. In particular, the evaluation ex-
amines whether generated queries remain grounded in the

21
Table 9: Quantitative explainability metrics across reasoning
hop depth.
Baseline Fine-Tuned
Hop HR QVR SCR HR QVR SCR
1-Hop 0.35 0.20 0.82 0.15 0.10 0.93
2-Hop 0.45 0.28 0.76 0.22 0.15 0.84
3-Hop 0.58 0.40 0.53 0.30 0.22 0.67
4-Hop 0.78 0.60 0.25 0.45 0.35 0.55
5-Hop 1.00 0.80 0.00 0.70 0.50 0.35
ontologyandcomplywithquery-generationinstructionsas
relational complexity increases.
Threelog-derivedindicatorsareused:HallucinationRate
(HR), Query Violation Rate (QVR), and Schema Consis-
tency Rate (SCR). These metrics capture complementary
aspectsofmodelbehaviour,includinggroundingreliability,
instruction adherence, and ontology conformity. HR, defined
in Equation 6, measures the proportion of generated queries
containing fabricated entities, relationships, properties, or
identifiers that are unsupported by the BRIDG-ICS ontology:
HR=𝑁hall
𝑁gen,(6)
where𝑁halldenotesthenumberofhallucinatedqueries
and𝑁gendenotesthetotalnumberofgeneratedqueries.Query
ViolationRate(QVR),definedinEquation7,measuresthe
proportionofevaluationrunsinwhichthemodelviolatesthe
three-query generation constraint:
QVR=𝑁viol
𝑁runs,(7)
where𝑁violisthenumberofrunsthatviolatethequery-
generation constraint and 𝑁runsis the total number of eval-
uation runs. Schema Consistency Rate (SCR), defined in
Equation8,measurestheproportionofgeneratedqueriesthat
usevalidontologyentities,relationshiptypes,andproperties:
SCR=𝑁valid
𝑁gen,(8)
where𝑁validdenotes the number of ontology-aligned
queries.LowerHRandQVRvalues,togetherwithahigher
SCR value, indicate stronger grounding, improved structural
control, and greater explainability.
Table 9 summarizes explainability performance across
one- to five-hop reasoning. Both models exhibit compara-
tivelystablebehaviouronshallowqueries,whilestructural
degradation becomes increasingly evident beyond three hops.
As relationaldepth increases,HR andQVR increasewhile
SCR decreases, reflecting the growing difficulty of maintain-
ing valid graph structure, ontology alignment, and query-
generation constraints.Despite this degradation, the fine-tuned model consis-
tently outperforms the baseline at every reasoning depth.
For one-hop queries, fine-tuning reduces HR from 0.35 to
0.15 and increases SCR from 0.82 to 0.93. This indicates
that fine-tuning improves structural reliability even when the
requiredgraphtraversalisrelativelysimple.Theperformance
differencebecomesmorepronouncedasrelationalcomplexity
increases. At four hops, fine-tuning reduces HR from 0.78 to
0.45andQVRfrom0.60to0.35,whileincreasingSCRfrom
0.25 to 0.55. At five hops, the baseline exhibits complete
structural failure, with an HR of 1.00 and an SCR of 0.00.
In contrast, the fine-tuned model retains partial structural
validity,obtaininganHRof0.70andanSCRof0.35.Fine-
tuningmarkedlyimprovesontologyconformity,instruction
adherence, and grounding reliability as relational complexity
increases.Whileperformancedropsforbothmodelsatdeeper
hop levels, the fine-tuned model declines more slowly and
retains more structurally interpretable queries.
6.5.2 Reasoning Completeness Across Hop Depth
The effect of relational depth on reasoning completeness
is analyzed across increasing hop complexity within the
BRIDG-ICSontologyusingthesame450-queryevaluation
corpus. A query is considered successful when the model
generatesacorrectandexecutableCypherquerythatretrieves
the intended graph path.
In the baseline model, the three generated query alter-
natives frequently contain mismatched identifiers, invalid
relationships, or repetitive query patterns. These errors result
in incomplete or non-executable graph traversals. In con-
trast, the fine-tuned model produces more structurally coher-
ent alternatives using ontology-defined relationships such as
HAS_POSSIBLE_CWE andHAS_POSSIBLE_TECHNIQUE .Con-
sequently, alternative queries remain semantically grounded
evenwhenthefirstgeneratedqueryisunsuccessful.Asshown
inFigure7a,bothmodelsperformcomparativelywellonone-
to three-hop queries, where the required relational chains
remain relatively shallow. Structural difficulty becomes more
apparent at four hops, where the model may need to recon-
struct a complete relation chain such as
CVE→CWE→CAPEC→Technique.(9)
At this depth, the baseline exhibits increasing instability,
whereasthefine-tunedmodelmaintainsstrongerperformance
byusingdomain-adaptedrepresentationsandontology-aware
relationship patterns learned during fine-tuning. These pat-
terns support more coherent reconstruction of multi-layer
graph traversals, includingpaths that spanboth information
technology and operational technology entities. The diver-
genceismostpronouncedatfour-andfive-hopdepths,where
relationalcomplexityincreasessubstantially.Atfivehops,the

22
baseline frequentlyfails to reconstructcomplete multi-layer
paths. As illustrated in Figure 7b, the fine-tuned model re-
covers complete paths of four or more hops in approximately
62% of cases, compared with 22% for the baseline.
These findings indicate that shallow reasoning involving
one to three hops remains manageable for both models. How-
ever, fine-tuning substantially improvesstructural continuity
and reasoning completenessfor deeper four- and five-hop
queries.
6.5.3 Structural Stability and Retrieval behaviour
Structural robustness under multi-query generation is eval-
uated through query-constraint adherence, reasoning com-
pleteness,retrievalbehaviour,anderrorcharacteristics.These
complementary analyses are summarised in Figure 8.
Query Violation Rate provides an initial measure of
structural control, as shown in Figure 8a. The baseline fre-
quently exceeds the three-query constraint, particularly as
relational complexity increases. In contrast, the fine-tuned
model demonstrates stronger instruction adherence and more
consistentlygeneratesqueryalternativeswithinthepermitted
boundary. This result indicates improved regulation of query
generation under constrained reasoning conditions.
Reasoning completeness is further evaluated through
complete-path recovery for multi-hop queries in Figure 8b,
where the fine-tuned model achieves a substantially higher
recoveryratethanthebaseline,indicatingimprovedstructural
controlandreasoningcontinuity.AsshowninFigure8c,italso
resolvesmorequeriesthroughdirectsymbolicgraphtraversal,
whereasthebaselinereliesmoreheavilyonembedding-based
fallbackandproducesmoreunresolvedcases.Directsymbolic
traversal improves traceability by producing explicit and
verifiable reasoning paths through the BRIDG-ICS ontology.
Althoughfallbackretrievalsupportsrecoverywhentheinitial
Cypherqueryfails,itmayintroduceambiguitywhenmultiple
graph nodes have similar semantic relevance; however, the
retrieved anchors are used only to regenerate an ontology-
compliant Cypher query before symbolic graph execution
resumes.
Error-type composition provides further insight into
model behaviour, as shown in Figure 8d. Baseline failures
arepredominantlycompounderrorsthatcombineincorrect
relationshiptargetswithfabricatedormismatchednodeidenti-
fiers. The fine-tuned model produces fewer compound errors
and generates outputs that remain more structurally inter-
pretable, even when the retrieved path is incorrect. These
results indicate that fine-tuning improves query-generation
stability,adherencetostructuralconstraints,reasoningcon-
tinuity, and reliance on direct symbolic grounding. These
improvements strengthen explainability by ensuring that a
largerproportionofmodeloutputscanbetracedtoexplicit
ontology entities, relationships, and executable graph paths.Together, the quantitative metrics and behavioural analy-
ses show that fine-tuning improves both GRICS performance
and interpretability. Lower hallucination and query-violation
ratesindicatestrongergroundingandinstructionadherence,
whilehigherschemaconsistencyandcomplete-pathrecovery
reflect better ontology alignment and reasoning continuity.
Foranalysts,thesegainsyieldmoretransparent,verifiableout-
puts: generated Cypherqueriesrevealthe relationshipsused
in reasoning, retrieved graph paths provide explicit evidence,
anderrorcategoriesremainstructurallyinterpretable.GRICS
thusenablesexplainablecyber-threatreasoningbycombin-
ing measurable structural reliability with evidence-linked
reasoning traceability.
6.6 Use Cases
Thissectionpresentsrepresentativeusecasesthatdemonstrate
how GRICS performs multi-hop reasoning over heteroge-
neous cybersecurity knowledge graphs integrating MITRE
ATT&CK,NVDvulnerabilitydata,CAPECattackpatterns,
andindustrialassetinformation.Theselectedexamplesare
organized to reflect four core analytical capabilities of the
framework:identifying attackpaths acrossindustrial assets,
analysingvulnerability-specificevidence,attributingadversar-
ialtechniquesthroughstructuredthreat-intelligencerelations,
and deriving mitigations across interconnected semantic lay-
ers. In this way, the section shows how GRICS unifies asset-
level, vulnerability-level, threat-level, and mitigation-level
reasoning within a single graph-grounded framework for
Industry 5.0 cybersecurity analysis.
Attack Path Reasoning.We first consider a multi-hop at-
tack scenario within an industrial control system. Given
the query“Show path between MQTT_BROKER_1 and
SAFETY_PLC_2”, GRICS automatically generates an exe-
cutableCypherquerytoretrievevalidgraphpathsbetween
thespecifiedassets.Theresultingpathsrepresentpotential
communication or attack routes formed by interconnected
components,includingbrokers, runtimeservers,conveyors,
and PLCs. The path length reflects reasoning depth, cap-
turingdirectandindirectdependenciesacrossnetworkand
operational layers. Retrieved paths depend on the underlying
infrastructure;differencesintopology,deviceconfiguration,
segmentation policies, and Industry 5.0 architectures can
yielddifferentreasoningoutcomes.Thesepathsarethensum-
marised in natural language, and context-aware mitigation
recommendations are generated to support cyber–physical
risk assessment.
Vulnerability-Centric Analysis.Beyond asset connectivity,
GRICS supports vulnerability-focused reasoning through
structured graph retrieval. As shown in Figure 10, a query
forCVE-2025-9492retrieves associated attributes such as
CVSSseverityscores,exploitabilitymetrics,andlinkedCWE
classifications.Theseelementsaresynthesizedintoaconcise

23
(a)Multi-hopsuccessrateacrossone-tofive-hopqueriesoriginating
from CVE entities.
(b) Hallucination rate across four- and five-hop queries.
Fig. 7: Reasoning completeness and structural stability under increasing multi-hop complexity.
(a) Query violation rate.
 (b) Complete-path recovery for
queries requiring four or more
hops.
(c) Retrieval-mode distribution.
 (d) Error-type composition.
Fig.8:Structuralstability,retrievalbehaviour,anderrorcharacteristicsundermulti-querygenerationacrossincreasinghop
depth.
technical description of the vulnerability’s impact. To ensure
correctness, precision, and traceability, the retrieved informa-
tionisalignedwiththeofficialCVErecord4,demonstrating
consistency with authoritative vulnerability data sources.
Basedonthisvalidatedknowledge,thesystemderivesmitiga-
tion guidance, including patch deployment, input validation,
and secure configuration practices.
AdversarialTechniqueAttribution.GRICSfurtherenables
adversarial behaviouranalysis within the MITREATT&CK
domain. For the query“Which technique is used by group
Cleaver via malware TinyZBot?”, the framework performs
multi-hoptraversalacrossGroupMalware–Techniquerela-
tionships in the knowledge graph.
As illustrated in Figure 11, the proposed framework iden-
tifiestherelevantmalwareentityanditsassociatedMITRE
ATT&CK techniques through structured graph traversal. The
retrieved techniques are subsequently synthesized into a
human-readable explanation, describing their operational
intent and corresponding defensive considerations. The iden-
tified malware entity and its associated ATT&CK techniques
are consistent with the official MITRE ATT&CK knowledge
4http://cve.org/CVERecord?id=CVE-2025-9492base,demonstratingthecorrectnessofthegraph-grounded
reasoningprocess.Thisconsistencyisvalidatedagainstthe
official MITRE ATT&CK software entry for TinyZBot5.
Mitigation Derivation Across Abstraction Layers.GRICS
derives solution-oriented mitigation strategies through multi-
hop traversal across CVE, CWE, CAPEC, ATT&CK, and
mitigationentities.AsshowninFigure12,thequery“Find
mitigations for CVE-2025-9492”retrieves both ATT&CK-
levelandCWE-levelcountermeasures,includingbehaviour
prevention,operatingsystemhardening,userawarenesstrain-
ing, and secure design practices. All returned mitigations
are consistent with ground-truth relationships encoded in
the BRIDG-ICS knowledge graph. Fine-tuning further en-
ables the model to infer additional plausible mitigation paths
beyondexplicitlylinkededges,producingstructurallyvalid
recommendations through relational generalization. These
inferred strategies remain semantically aligned with estab-
lished countermeasures, indicating that fine-tuning supports
controlledpathexpansionwithoutintroducingunsupported
mitigation associations.
5https://attack.mitre.org/software/S0004/

24
Fig. 9: Example of multi-hop attack path reasoning between industrial assets.
Fig. 10: Example of vulnerability-centric analysis for a specific CVE.
Fig. 11: Adversarial technique attribution using multi-hop graph reasoning in GRICS. The framework traversesGroup–
Malware–Techniquerelationships to identify relevant ATT&CK techniques and generate structured explanations.
ThepresentedusecaseshighlightthecapabilityofGRICS
to seamlessly integrate attack path analysis, vulnerability
assessment,adversarialattribution,andmitigationplanning
withinaunifiedgraph-groundedreasoningframework.Throughinterpretablemulti-hopinferenceandstructuredknowledge
synthesis, the system minimizes manual correlation efforts
andstrengthensevidence-basedcybersecuritydecision-making
in complex Industry 5.0 industrial ecosystems.

25
Fig. 12: Example of automatically generated mitigation recommendations derived from multi-hop graph reasoning.
End-to-End Explainability Example
Figure 13 presents an end-to-end reasoning example illustrat-
ing the transparency of the proposed framework by exposing
each stage of the graph-grounded inference pipeline. The
exampledemonstrateshowanalystscaninspectthegenerated
Cypher query, retrieved graph evidence, intermediate graph
traversals, and the final grounded response, enabling every
reasoning step to be traced and verified.
UnlikeconventionalLLM-basedsystemsthatdirectlygen-
erateresponses, GRICSseparatesreasoning intotwostages.
The first LLM translates the analyst’s natural-language query
into anexecutable Cypher query,which is subsequentlyexe-
cutedovertheBRIDG-ICSknowledgegraph.Theretrieved
graph evidence, including the discovered attack paths and in-
termediate entities, is then provided to a second LLM, which
is not fine-tuned and is responsible solely for generating a
human-readable summary of the retrieved evidence.
Thisgraph-groundeddesignenableseverygeneratedre-
sponsetobetracedbacktoexplicitknowledgegraphevidence
rather than opaque neural reasoning. Analysts can verify the
generated Cypher query, inspect the returned graph enti-
ties and relationships, and confirm that the final response
is fully supported by the retrieved evidence. Consequently,
thereasoningprocessremainstransparent,explainable,and
auditable throughout the inference pipeline. Although this
example is not a formal human-subject study, it illustrates
theframework’sexplainabilityartifactsthatsupportanalyst
inspection,evidenceverification,andinformedcybersecurity
decisions.7 Discussion
7.1 Neuro-Symbolic Reasoning
TheexperimentalresultsdemonstratethatGRICSimproves
cyberthreatintelligencereasoningacrossbothCTI-RCMand
CTI-ATE tasks. These gains extend beyond predictive perfor-
mance,reflectingashifttowardstructuredandsemantically
grounded reasoning enabled by the integration of fine-tuning
andgraphgrounding.Unlikeconventionaltext-basedRAG
approaches,whichrelyonpatternmatchingoverunstructured
data,GRICSleveragesexplicitrelationshipswithintheknowl-
edge graph, preserving logical consistency across entities
such as CVEs, CWEs, and ATT&CK techniques. This is
particularly critical in cybersecurity, where incorrect associa-
tionsmaypropagatethroughdownstreamanalysis.Compared
to existing Graph-RAG approaches such as GraphRAG [49]
and KG2RAG [ 31], which primarily rely on embedding-
based or subgraph expansion strategies, GRICS introduces a
tighterintegrationofsymbolicqueryexecutionwithneural
retrieval. While these methods improve contextual relevance,
they may still produce semantically plausible but structurally
inconsistent relationships due to the absence of constraint-
driven reasoning. In contrast, the Cypher-based symbolic
retrieval in GRICS ensures that inferred relationships remain
consistentwiththeunderlyinggraphstructure,improvingrea-
soningfidelityandinterpretability.Furthermore,relativeto
cybersecurity-focused approaches such as CyKG-RAG [ 55],
GRICS extends structured retrieval through fine-tuned query
generationandhybridfallbackmechanisms,enablingmore
robust handling of ambiguous and compositional queries.
This structured reasoning also contributes to the ro-
bustness characteristics of GRICS. Under the ATT&CK-
oriented robustness evaluation, GRICS demonstrates re-

26
Fig. 13: Explainability case study showing the analyst query, generated Cypher query, retrieved graph evidence, and final
grounded response.
silienceagainstadversarialpromptinjectionwhilepreserving
graph-consistentretrieval.Althoughitdoesnotachievethe
lowestAttackSuccessRate(ASR),theresultsindicatethat
symbolic graph grounding provides a degree of protection
againstadversarialmanipulation.Nevertheless,theevaluation
also shows that graph grounding alone does not eliminate
adversarial vulnerabilities, highlighting the need for more
robust graph-aware defence mechanisms and standardised
robustness evaluation protocols.
7.2 Human–AI Collaborative Intelligence
ThefindingsindicatethatGRICSismostappropriatelypo-
sitioned as a decision-support framework rather than an
autonomous cybersecurity system. Its graph-grounded rea-
soning capabilities can reduce the effort required to correlate
vulnerabilities, weaknesses, attack patterns, adversarial tech-
niques,affectedassets,andmitigations.However,theresulting
recommendations should complement, rather than replace,
expert judgement.
This collaborative role is particularly important in Indus-
try 5.0 environments, where cybersecurity decisions may af-
fect safety-critical assets, production continuity, and physical
operations. By exposing generated Cypher queries, retrieved
entities, and graph traversal paths, GRICS allows analysts
to examine the evidence underlying each recommendation
and determine whether it is consistent with the operational
context. Human experts remain responsible for incorporating
information that may not be represented in the knowledge
graph,includingassetconfigurations,organisationalpriori-
ties,operationalconstraints,andacceptablerisklevels.Theinteraction between analysts and GRICS therefore combines
complementarycapabilities.Theframeworkprovidesscalable
knowledge correlation and structured multi-hop reasoning,
whileanalystscontributecontextualunderstanding,domain
expertise, and accountability. This division of responsibil-
ity supports more transparent and informed cybersecurity
decision-makingwhilelimitingtherisksassociatedwithfully
automated responses.
TheexplainabilityartifactsexposedbyGRICS,including
generatedCypherqueries,retrievedentities,andexplicitgraph
traversalpaths,supportanalystverificationandaccountable
decision-making without transferring final authority to the
automated system.
7.3Implications for Cybersecurity and Industry 5.0 Systems
The findingshaveimportant implicationsfor cybersecurity
analysis in Industry 5.0 environments as follows:
First, the ability to generate explainable multi-hop reasoning
pathsaddressesakeylimitationofexistingAI-drivensecurity
systems, which often lack transparency. By exposing inter-
mediate Cypher queries and graph traversal paths, GRICS
enables analysts to verify reasoning steps, improving trust,
auditability,anddecisionreliabilityinsafety-criticalcontexts.
Second, the hybrid retrieval mechanism demonstrates that
combining symbolic reasoning with embedding-based fall-
back provides a practical balance between precision and
robustness. The embedding space analysis confirms stable
semantic neighbourhoods, enabling reliable identification of
relevantanchornodeswhensymbolicqueryexecutionfails.
Thisintegration strengthens retrievalreliability by allowing

27
semanticrecoverywhileensuringthatthefinalevidencere-
mains grounded through ontology-compliant symbolic graph
execution.
Third,GRICSadvancesHuman–AIcollaborationbyenabling
analyststointeractwithcomplexcyber-physicalknowledge
through natural language while maintaining traceability to
underlying graph evidence. This capability is particularly rel-
evantforIndustry 5.0systems,wheretightlycoupledIT/OT
environments require interpretable and context-aware deci-
sion support.
Finally,theframeworkhighlightsthepotentialforintegration
with digital twin environments. By coupling graph-based
reasoningwithreal-timeorsimulatedsystemrepresentations,
GRICScouldsupportdynamicriskassessment,attackpath
simulation, and proactive defence planning, enabling contin-
uous monitoring and adaptive reasoning in cyber–physical
systems.
Scalability Considerations.The current evaluation is con-
ducted on the BRIDG-ICS knowledge graph and therefore
provides an initial computational baseline for GRICS. For
substantially larger enterprise knowledge graphs, scalabil-
ity would depend on graph size, local relationship density,
reasoningdepth,embedding-searchcost,andgraph-update
frequency.BecauseGRICSperformslocalizedCyphertraver-
sal rather than exhaustive graph exploration, retrieval latency
isexpectedtobeinfluencedmorebythesizeanddensityof
the queried neighbourhood than by total graph size alone,
althoughdeepermulti-hopqueriesmayincurhighertraversal
cost.
Node embeddings are generated offline and reused dur-
ing inference, so embedding regeneration is mainly required
whengraphentitiesareaddedorupdatedratherthanforevery
query. Memory requirements would also increase with the
number of graph entities, relationships, and stored embed-
dings, while the main GPU-memory requirement remains
associatedwiththeLLM.Performanceonsubstantiallylarger
industrial knowledge graphs has not yet been empirically
validated and may require further optimisation or alternative
architectural approaches for enterprise-scale deployment.
OntologyMaintenance andEvolution.Industrialcyberse-
curityknowledgeevolvescontinuouslyasnewvulnerabilities,
attack techniques, assets, and threat relationships emerge.
Although dynamic ontology evolution is not evaluated in
the current implementation, GRICS can accommodate future
BRIDG-ICS updates through periodic or continuous CTI in-
gestion.NewlycollectedCTIrecordscouldbenormalisedand
mappedtotheexistingontologybeforebeingsynchronised
with the Neo4j knowledge graph. New or modified graph
entities would also require corresponding embedding up-
datestomaintainconsistencybetweensymbolicandsemantic
retrieval.More substantial ontology changes, such as the intro-
duction of new entity classes or relationship types, would
additionallyrequiresynchronisationwiththeontologycon-
straintsusedforCyphergeneration.Maintainingalignment
among the ontology, knowledge graph, embeddings, and
query-generation instructions would therefore be an impor-
tant consideration for future deployment in continuously
evolving industrial environments.
7.4 Limitations and Open Research Questions
Limitations.Thestudyissubjecttothefollowinglimitations:
1.Reasoning and retrieval limitations.Reasoning qual-
ity is sensitive to graph sparsity, ontology complexity,
and increasing hop depth. Long or compositional queries
may produce incomplete, redundant, or invalid Cypher
statements, while the size of the BRIDG-ICS ontology
introduces token overhead that may restrict the available
context.Thecurrentevaluationalsoconsidersonlyone
embedding model; therefore, the effects of alternative
embedding approaches on retrieval quality and seman-
tic separation remain uncertain. In addition, after an
embedding-basedgraphanchorisidentified,successful
retrieval still depends on regenerating a valid ontology-
compliantCypherquery.Predefinedquerypatternsand
overlapping candidate queries may further reduce flexi-
bility and increase retrieval overhead.
2.Scalabilityanddeploymentlimitations.Althoughthe
current evaluation characterises runtime, memory utilisa-
tion,andcomputationalcomplexitywithintheBRIDG-
ICSenvironment,performanceoversubstantiallylarger
enterprise knowledge graphs has not yet been empiri-
cally validated. Larger graph structures may introduce
additionalretrievallatency,embedding-maintenancecost,
memory requirements, and operational overhead depend-
ing on graph density and update frequency. Enterprise-
scaledeploymentmaythereforerequireadditionalopti-
misationstrategiesoralternativearchitecturalapproaches
tomaintainpracticalretrievalandreasoningperformance
as the knowledge graph grows.
3.Dual-LLM deployment considerations.The dual-LLM
architecture introduces additional computational andde-
ployment overhead compared with single-model RAG
pipelinesbecauseseparateinferencestagesarerequired
forCyphergenerationandresponsesynthesis.Thiscanin-
creaseinferencecost,latency,memory requirements, and
system-integrationcomplexity,particularlywhenlarger
language models are employed. Although the current im-
plementationcanoperatewithlocallydeployedmodels,
deployments that rely on externally hosted LLM services
mayadditionallyintroduceAPIcosts,networklatency,ser-
vice availability dependencies, and data-governance con-

28
siderations.Thesetrade-offsrepresentpracticaldeploy-
mentlimitationsthatshouldbeconsideredwhenapplying
GRICSinresource-constrainedorsecurity-sensitivein-
dustrial environments.
4.Human-centred evaluation.Although GRICS provides
graph-grounded explanations and traceable reasoning ev-
idence,thecurrentstudydoesnotevaluatethesecapabili-
tieswithcybersecuritypractitioners.Futureworkcould
involve user studies with analysts performing representa-
tivethreat-investigationtaskstoassessexplanationuse-
fulness, trust, evidence verification, and decision-making
efficiency.Suchevaluationwouldprovideamoredirect
assessmentofthepracticalvalueofGRICSforhuman–AI
collaboration in cybersecurity operations.
5.Cross-framework robustness comparison.Although
the robustness evaluation was restricted to ATT&CK-
orientedreasoningtaskstoimprovecomparabilitywith
PoisonRAG and GRAPPOISON, differences remain in
the underlying knowledge graphs, ontology structures,
retrieval architectures, and adversarial settings. Conse-
quently, the reported ASR and TPQ results provide quali-
tative insight rather than a fully controlled comparison,
and direct quantitative superiority over these frameworks
cannot be established.
OpenResearchQuestionsThelimitationsobservedinthis
studyhighlightseveralunresolvedchallengesingraph-grounded
cyberthreatintelligence.Reasoningoverindustrialknowledge
graphscontainingmillionsofentitiesrequiresscalableindex-
ing,retrieval,andtraversalmechanismsthatpreserveexplain-
ability. Future work should therefore investigate approximate
nearest-neighbourindexing,distributedgraphprocessing,and
moreefficientgraphrepresentations.Cybersecurityontolo-
gies must also adapt to emerging threats, vulnerabilities, and
relationships with limited manual intervention, motivating
researchintoautomatedcyberthreatintelligenceingestion,in-
crementalgraphupdates,andontologyevolution.Inaddition,
controlled robustness evaluations using identicalknowledge
graphs, attack settings, and evaluation protocols are required
tosupportfaircomparisonwithotherGraph-RAGsystems.Fi-
nally, systematic studies involving cybersecurity analysts are
neededtoassessusability,interpretability,cognitiveworkload,
decisionquality,andanalysttrustinoperationalenvironments.
8 Conclusion
This paper presents GRICS, a Human–AI collaborative
knowledge-graph framework for explainable threat reasoning
in advanced manufacturing and Industry 5.0 environments.
The study addresses a key limitation of existing RAG and
Graph-RAG approaches for cybersecurity, which often strug-
gle to maintain structurally valid multi-hop reasoning acrosstightly coupled IT and OT systems. The proposed frame-
work addresses this challenge by unifying the BRIDG-ICS
ontology, Cypher-based symbolic retrieval, controlled em-
bedding fallback, and LLM answer synthesis in a single
neuro-symbolic pipeline. Experiments show that it improves
reasoningaccuracyandusability.OnCTI-RCMandCTI-ATE
benchmarks, it consistently outperformed baseline LLMs,
with the fine-tuned, graph-enhanced setup performing the
best. The framework also increases retrieval transparency by
resolvingmorequeriesviasymbolicgraphtraversal,reducing
relianceonapproximatefallback,andgeneratingtraceableevi-
dencechains.Robustnessanalysisshowsthatgraph-grounded
reasoningbetterpreservescoherentmulti-hopinferenceunder
adversarial and structurally complex queries. By exposing
intermediate Cypher queries, graph evidence, and reasoning
paths,theframeworksupportsinterpretableHuman–AIcol-
laboration, analyst trust, and informed decision-making in
safety-critical settings.
FutureworkwillinvestigateSymbolicIntelligence-driven
digital twins, augmented with AI and ontological reason-
ing,alongsideadaptiveretrievalstrategiesandreal-timecy-
ber–physical threat monitoring. Combining graph-grounded
reasoning with dynamic cyber-physical system representa-
tions offers a promising path for real-time risk assessment,
attack simulation, and adaptive defence planning. Such ad-
vanceswouldbridgestaticthreatintelligencewithevolving
system states, positioning the framework as a foundation
fornext-generation,resilient,andexplainablecybersecurity
assistants in complex industrial environments.
Thisworkdemonstratesthatgraph-groundedneuro-symbolic
reasoning provides an effective foundation for explainable
cyber threat intelligence in Industry 5.0 environments. By
combining symbolic graph retrieval with large language
model reasoning, GRICS enables transparent, context-aware,
and multi-hop threat analysis while maintaining traceabil-
ity to structured cybersecurity knowledge. We believe this
framework provides a promising foundation for future graph-
grounded cybersecurity assistants supporting advanced man-
ufacturing systems.
Declarations
DataAvailability.Thedatasetsusedinthisstudyarepublicly
accessible via the project’s GitHub repository:Resellient-
Industry-5.0–KGawareCybersecurityIntelligenceModelling.
Funding.This work is supported by the Edith Cowan Uni-
versity, Australia Early and Mid-Career Research (EMCR)
Grant.
Acknowledgements.We acknowledge the School of Science
(Computing and Security Discipline) for providing access to
the Industry 5.0 systems testbed and GPU resources used for
LLM-driven knowledge graph enrichment and fine-tuning.

29
Generative AI Use.The authors acknowledge the use of
OpenAI ChatGPT to assist with language refinement and
grammatical review.
Competing Interests.The authors declare no competing
interests.
CRediT authorship contribution statement
Writing – original draft: P.N., A.M.; Writing – review and
editing: A.M., P.N., A.I., I.H.S., H.J.; Project administration:
A.M.
References
1.S.Nahavandi,Sustainability11(16),4371(2019). DOI
10.3390/su11164371
2.H.Srinivasan,M.Karimi. Threat-basedsecuritycontrols
toprotectindustrialcontrolsystems(2025). URL https:
//arxiv.org/abs/2501.13268
3.D.Bhamare,M.Zolanvari,A.Erbad,R.Jain,K.Khan,
N. Meskin. Cybersecurity for industrial control systems:
Asurvey(2020). URL https://arxiv.org/abs/20
02.04124
4.M. Nankya, R. Chataut, R. Akl, Sensors23(21), 8840
(2023). DOI10.3390/s23218840. URL https://doi.
org/10.3390/s23218840
5.M.D. Firoozjaei, N. Mahmoudyar, Y. Baseri, A.A.
Ghorbani, International Journal of Critical Infrastruc-
ture Protection36, 100487 (2022). DOI https://do
i.org/10.1016/j.ijcip.2021.100487. URL https:
//www.sciencedirect.com/science/article/pi
i/S1874548221000718
6.A. Saini, K. Krishan, M.S. Gaur, Computers and Elec-
trical Engineering132, 110967 (2026). DOI https:
//doi.org/10.1016/j.compeleceng.2026.110967. URL
https://www.sciencedirect.com/science/arti
cle/pii/S0045790626000352
7.M.Homaei,M.Tarif,P.G.Rodríguez,A.Caro,M.Ávila,
Machine Learning with Applications23, 100824 (2026).
DOI 10.1016/j.mlwa.2025.100824. URL http://dx.d
oi.org/10.1016/j.mlwa.2025.100824
8.A. Borah, M.T. Alam, N. Rastogi. Adapting large
language models to emerging cybersecurity using re-
trieval augmented generation (2025). URL https:
//arxiv.org/abs/2510.27080
9.A. Gusarov, A. Volkova, V. Khrulkov, A. Kuznetsov,
E. Maslov, I. Oseledets. Multi-agent graphrag: A text-
to-cypher framework for labeled property graphs (2025).
URLhttps://arxiv.org/abs/2511.08274
10.M. Kim, J. Wang, K. Moore, D. Goel, D. Wang,
A. Mohsin, A. Ibrahim, R. Doss, S. Camtepe, H. Jan-
icke, inCompanion Proceedings of the ACM on WebConference2025(AssociationforComputingMachinery,
New York, NY, USA, 2025), WWW ’25, p. 2851–2854.
DOI 10.1145/3701716.3715171. URL https:
//doi.org/10.1145/3701716.3715171
11.A. Mohsin, H. Janicke, A. Ibrahim, M.I. Sarker,
S. Camtepe. A unified framework for human–ai col-
laboration in security operations centers with trusted
autonomy (2026). DOI 10.1145/3837073. URL
https://doi.org/10.1145/3837073
12.A.O.M.Saleh,G.Tur,Y.Saygin,inProceedingsofthe
7thInternationalConferenceonNaturalLanguageand
Speech Processing (ICNLSP 2024), ed. by M. Abbas,
A.A.Freihat(AssociationforComputationalLinguistics,
Trento, 2024), pp. 439–448. URL https://aclantho
logy.org/2024.icnlsp-1.45/
13.J. Fang, Z. Meng, C. Macdonald. Trace the evidence:
Constructing knowledge-grounded reasoning chains for
retrieval-augmented generation (2024). URL https:
//arxiv.org/abs/2406.11460
14.J.Chen,H.Lin,X.Han,L.Sun. Benchmarkinglargelan-
guagemodelsin retrieval-augmentedgeneration(2023).
URLhttps://arxiv.org/abs/2309.01431
15.Q.Zhang,S.Chen,Y.Bei,Z.Yuan,H.Zhou,Z.Hong,
H.Chen,Y.Xiao,C.Zhou,J.Dong,Y.Chang,X.Huang.
A survey of graph retrieval-augmented generation for
customized large language models (2025). URL https:
//arxiv.org/abs/2501.13958
16.L.F.Sikos,KnowledgeandInformationSystems65,3511
(2023). DOI 10.1007/s10115-023-01860-3. URL ht
tps://doi.org/10.1007/s10115-023-01860-3 .
Received:1November2021;Revised:26February2023;
Accepted: 11 March 2023; Published: 29 April 2023
17.H. Gao, H. Tong, B. Yong, G. Shen, Electronics15(3)
(2026). DOI 10.3390/electronics15030552. URL
https://www.mdpi.com/2079-9292/15/3/552
18.B. Lourenço, P. Adão, J.F. Ferreira, M.M. Marques,
C.Vaz. Structuringsecurity:Asurveyofcybersecurity
ontologies,semanticlogprocessing,andllmsapplication
(2025). URL https://arxiv.org/abs/2510.16610
19.F...zdemirS..nmez,C.Hankin,P.Malacaria,Computers
&Security123,102938(2022). DOI https://doi.org/10
.1016/j.cose.2022.102938. URL https://www.scie
ncedirect.com/science/article/pii/S0167404
822003303
20.P. Liu, X. Wang, Q. Fu, Y. Yang, Y.F. Li, Q. Zhang,
Knowledge-Based Systems250, 108870 (2022). DOI
https://doi.org/10.1016/j.knosys.2022.108870. URL
https://www.sciencedirect.com/science/arti
cle/pii/S0950705122004154
21.Z.Z.S.Moghaddam,Z.Dehghani,M.Rani,K.Aslansefat,
B.K. Mishra, R.R. Kureshi, D. Thakker. Explainable
knowledge graph retrieval-augmented generation (kg-
rag)withkg-smile(2025). URL https://arxiv.org/

30
abs/2509.03626
22.A. Hoenig, K. Roy, Y. Acquaah, S. Yi, IEEE AccessPP,
1 (2024). DOI 10.1109/ACCESS.2024.3395444
23.P.Nandiya,A.Mohsin,A.Ibrahim,I.H.Sarker,H.Jan-
icke,Cybersecurity9(1),167(2026). DOI10.1186/s424
00-026-00597-0. URL https://doi.org/10.1186/
s42400-026-00597-0
24.A.Rejeb,K.Rejeb,I.Zrelli,etal.,DiscoverSustainability
6,307(2025). DOI10.1007/s43621-025-01166-0. URL
https://doi.org/10.1007/s43621-025-01166
-0
25.K. Liu, F. Wang, Z. Ding, S. Liang, Z. Yu, Y. Zhou.
A review of knowledge graph application scenarios in
cybersecurity(2022). URL https://arxiv.org/ab
s/2204.04769
26.K. Kurniawan, E. Kiesling, D. Winkler, A. Ekelhart, pp.
153–170 (2025)
27.P.Lewis,E.Perez,A.Piktus,F.Petroni,V.Karpukhin,
N. Goyal, H. Küttler, M. Lewis, W. tau Yih, T. Rock-
täschel, S. Riedel, D. Kiela. Retrieval-augmented gen-
erationforknowledge-intensivenlptasks(2021). URL
https://arxiv.org/abs/2005.11401
28.Y. Hu, Y. Lu. Rag and rau: A survey on retrieval-
augmented language model in natural language process-
ing (2025). URL https://arxiv.org/abs/2404.1
9543
29.W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T.S.
Chua,Q.Li,inProceedingsofthe30thACMSIGKDD
ConferenceonKnowledgeDiscoveryandDataMining
(Association for Computing Machinery, New York, NY,
USA, 2024), KDD ’24, p. 6491–6501. DOI 10.1145/36
37528.3671470. URL https://doi.org/10.1145/
3637528.3671470
30.Y. Cai, Z. Guo, Y. Pei, W. Bian, W. Zheng. Simgrag:
Leveraging similar subgraphs for knowledge graphs
driven retrieval-augmented generation (2025). URL
https://arxiv.org/abs/2412.15272
31.X. Zhu, Y. Xie, Y. Liu, Y. Li, W. Hu. Knowledge graph-
guided retrieval augmented generation (2025). URL
https://arxiv.org/abs/2502.06864
32.Y. Hu, Z. Lei, Z. Zhang, B. Pan, C. Ling, L. Zhao. Grag:
Graph retrieval-augmented generation (2025). URL
https://arxiv.org/abs/2405.16506
33.M. Barrère, C. Hankin, D. O’Reilly, Computers & Secu-
rity132,103348(2023). DOIhttps://doi.org/10.1016/j.
cose.2023.103348. URL https://www.sciencedir
ect.com/science/article/pii/S0167404823002
584
34.M.H. Rahman, E.Y. Hamedani, Y.J. Son, M. Shafae,
Journalof Computingand InformationSciencein Engi-
neering24(7) (2024). DOI 10.1115/1.4063729. URL
http://dx.doi.org/10.1115/1.406372935.M. Barrère, C. Hankin, N. Nicolaou, D.G. Eliades,
T. Parisini, Journal of Information Security and Applica-
tions52,102471(2020). DOIhttps://doi.org/10.1016/j.ji
sa.2020.102471. URL https://www.sciencedirect.
com/science/article/pii/S2214212619311342
36.K.Liu,Y.Xie,S.Xie,L.Sun,JournalofProcessControl
132,103131(2023).DOIhttps://doi.org/10.1016/j.jproco
nt.2023.103131. URL https://www.sciencedirect.
com/science/article/pii/S0959152423002184
37.D. Zheng, M. Lapata, J.Z. Pan. How reliable are llms as
knowledge bases? re-thinking facutality and consistency
(2024). URL https://arxiv.org/abs/2407.13578
38.V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu,
S. Edunov, D. Chen, W. tau Yih. Dense passage re-
trievalforopen-domainquestionanswering(2020). URL
https://arxiv.org/abs/2004.04906
39.S. Wang, H. Yang, W. Liu, Scientific Reports15, 40425
(2025).DOI10.1038/s41598-025-21222-z.URL https:
//doi.org/10.1038/s41598-025-21222-z
40.Y.Zhang,S.Mao,T.Ge,X.Wang,A.deWynter,Y.Xia,
W.Wu,T.Song,M.Lan,F.Wei. Llmasamastermind:A
surveyofstrategicreasoningwithlargelanguagemodels
(2024). URL https://arxiv.org/abs/2404.01230
41.K.Guu,K.Lee,Z.Tung,P.Pasupat,M.W.Chang. Realm:
Retrieval-augmentedlanguagemodelpre-training(2020).
URLhttps://arxiv.org/abs/2002.08909
42.A. Grover, J. Leskovec, inProceedings of the 22nd
ACMSIGKDDInternationalConferenceonKnowledge
Discovery and Data Mining(2016), pp. 855–864
43. X.Xie, Z.Li,X.Wang, Z.Xi,N.Zhang. Lambdakg:A
library for pre-trained language model-based knowledge
graphembeddings(2023). URL https://arxiv.org/
abs/2210.00305
44.Z. Chen, X. Wang, Z. Li, W. Guo, D. He. Kg-bilm:
Knowledge graph embedding via bidirectional language
models (2025). URL https://arxiv.org/abs/2506
.03576
45.S. Ma, C. Xu, X. Jiang, M. Li, H. Qu, C. Yang,
J. Mao, J. Guo. Think-on-graph 2.0: Deep and faith-
ful large language model reasoning with knowledge-
guided retrieval augmented generation (2025). URL
https://arxiv.org/abs/2407.10805
46.Z. Yang, L. Weng, L. Zhang, R. Tong, J. Xie, Z. Zeng,
D. Chen, PLOS ONE20(10), e0333037 (2025). DOI
10.1371/journal.pone.0333037
47.Y.Chen,S.Sun,X.Hu,AppliedSciences15(12)(2025).
DOI 10.3390/app15126722. URL https://www.mdpi
.com/2076-3417/15/12/6722
48.P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie,
C.D.Manning. Raptor:Recursiveabstractiveprocessing
for tree-organized retrieval (2024). URL https://ar
xiv.org/abs/2401.18059

31
49.H. Han, Y. Wang, H. Shomer, K. Guo, J. Ding, Y. Lei,
M. Halappanavar, R.A. Rossi, S. Mukherjee, X. Tang,
Q. He, Z. Hua, B. Long, T. Zhao, N. Shah, A. Javari,
Y.Xia, J.Tang,arXivpreprintarXiv:2501.00309(2025).
DOI10.48550/arXiv.2501.00309. URL https://arxi
v.org/abs/2501.00309
50.Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, P.S. Yu,
IEEE Transactions on Neural Networks and Learning
Systems32(1), 4 (2020)
51.Y. Cao, Z. Gao, Z. Li, X. Xie, S.K. Zhou, J. Xu, Pro-
ceedingsoftheVLDBEndowment18(10),3269–3283
(2025). DOI 10.14778/3748191.3748194. URL
http://dx.doi.org/10.14778/3748191.3748194
52.C. Mavromatis, G. Karypis. Gnn-rag: Graph neural
retrieval for large language model reasoning (2024).
URLhttps://arxiv.org/abs/2405.20139
53.M. Simoni, A. Saracino, V. P., M. Conti. Morse:
Bridging the gap in cybersecurity expertise with re-
trieval augmented generation (2024). URL https:
//arxiv.org/abs/2407.15748
54.R. Fayyazi, S.H. Trueba, M. Zuzak, S.J. Yang. Proverag:
Provenance-drivenvulnerabilityanalysiswithautomated
retrieval-augmentedllms(2025). URL https://arxi
v.org/abs/2410.17406
55. K. Kurniawan, E. Kiesling, A. Ekelhart, (2024)
56.J. Liang, Y. Wang, C. Li, R. Zhu, T. Jiang, N. Gong,
T. Wang. Graphrag under fire (2025). URL https:
//arxiv.org/abs/2501.14050
57.L. Liu, Z. Wang, H. Tong. Neural-symbolic reasoning
over knowledge graphs:A survey from a query perspec-
tive (2024). URL https://arxiv.org/abs/2412.1
0390
58.L.N. DeLong, R.F. Mir, J.D. Fleuriot, IEEE Transac-
tionsonNeuralNetworksandLearningSystems36(5),
7822–7842 (2025). DOI 10.1109/tnnls.2024.3420218.
URL http://dx.doi.org/10.1109/TNNLS.2024.
3420218
59.J.Zhang,B.Chen,L.Zhang,X.Ke,H.Ding,AIOpen
2,14(2021). DOIhttps://doi.org/10.1016/j.aiopen.202
1.03.001. URL https://www.sciencedirect.com/
science/article/pii/S2666651021000061
60.S. Chen, H. Fang, Y. Cai, X. Huang, M. Sun, inPro-
ceedings of the 37th International Conference on Neural
Information Processing Systems(Curran Associates Inc.,
Red Hook, NY, USA, 2023), NIPS ’23
61.K.Cheng,N.K.Ahmed,R.A.Rossi,T.Willke,Y.Sun,
ACM Trans. Knowl. Discov. Data18(9) (2024). DOI
10.1145/3686806. URL https://doi.org/10.1145/
3686806
62.X. Liu, T. Mao, Y. Shi, Y. Ren, Neurocomputing585,
127571 (2024). DOI https://doi.org/10.1016/j.neucom.2
024.127571. URL https://www.sciencedirect.co
m/science/article/pii/S092523122400342463.R.Angles,M.Arenas,P.Barceló,A.Hogan,J.L.Reutter,
D. Vrgoč, ACM Computing Surveys50(5), 1 (2018)
64.S. Purkayastha, S. Dana, D. Garg, D. Khandelwal, G.P.S.
Bhargav.Knowledgegraphquestionansweringviasparql
silhouettegeneration(2021). URL https://arxiv.or
g/abs/2109.09475
65.Z. Zhao, X. Ge, Z. Shen. S2ctrans: Building a bridge
from sparql to cypher (2023). URL https://arxiv.
org/abs/2304.00531
66.C. Mavromatis, S. Adeshina, V.N. Ioannidis, Z. Han,
Q. Zhu, I. Robinson, B. Thompson, H. Rangwala,
G. Karypis. Byokg-rag: Multi-strategy graph retrieval
for knowledge graph question answering (2025). URL
https://arxiv.org/abs/2507.04127
67.B.Liu,J.Zhang,F.Lin,C.Yang,M.Peng,W.Yin. Syma-
gent:Aneural-symbolicself-learningagentframework
for complex reasoning over knowledge graphs (2025).
URLhttps://arxiv.org/abs/2502.03283
68.Q. Lin, F. Xu, H. Lu, K. He, R. Mao, J. Liu, E. Cambria,
M.Feng. Towardsunifiedneurosymbolicreasoningon
knowledge graphs (2025). URL https://arxiv.org/
abs/2507.03697
69.R. Fieblinger, M.T. Alam, N. Rastogi. Actionable cyber
threat intelligence using knowledge graphs and large
language models (2024). URL https://arxiv.org/
abs/2407.02528
70.Z. Wu, F. Tang, M. Zhao, Y. Li. Kgv: Integrating
large language models with knowledge graphs for cyber
threat intelligence credibility assessment (2025). URL
https://arxiv.org/abs/2408.08088
71.X. Yang, R. Zhong, Y. Chen, G. Peng, D. Yao, C. Chen,
C. Wang, D. Zhang, Y. Zhou, Z. Yang, Cybersecurity
9(1), 106 (2026). DOI 10.1186/s42400-025-00505-y.
URL https://doi.org/10.1186/s42400-025-0
0505-y
72.B. Strom, A. Applebaum, D. Miller, K. Nickels, A. Pen-
nington, C. Thomas, MITRE Corporation (2020). Avail-
able at:https://attack.mitre.org
73.S.Barnum,inProceedingsofthe12thIEEEInternational
ConferenceonInformationTechnology:NewGenerations
(ITNG)(2019), pp. 347–352. DOI 10.1109/ITNG.2019.
00123
74.S.Barnum,etal.,Structuredthreatinformationexpres-
sion (stix®) version 2.0. Tech. rep., OASIS Open (2017).
Available at: https://oasis-open.github.io/ct
i-documentation/
75.A. Syed,J. Undercoffer, K.Maly,inProceedings ofthe
14thInternationalConferenceonAvailability,Reliability
andSecurity(ARES)(2019),pp.1–8. DOI10.1145/33
39252.3340502
76.M. Kandefer, S. Pritchett, M. Chiaramonte, R. Lemos,
inProceedings of the IEEE International Conference
on Big Data (BigData)(2020), pp. 5699–5708. DOI

32
10.1109/BigData50022.2020.9378342
77.X.Li,Q.Wang,J.Li,Y.Liu,inProceedingsoftheIEEE
Conference on Communications and Network Security
(CNS)(2022),pp. 249–257. DOI10.1109/CNS56104.2
022.00045
78.L.Sridhar,V.Rao,ProcediaComputerScience206,103
(2022). DOI 10.1016/j.procs.2022.10.276
79.B. Li, Q. Yang, C. Deng, H. Pan, Informatics12(3)
(2025). URL https://www.mdpi.com/2227-9709/
12/3/100
80.B.Peng,Y.Zhu,Y.Liu,X.Bo,H.Shi,C.Hong,Y.Zhang,
S. Tang, ACM Trans. Inf. Syst.44(2) (2025). DOI
10.1145/3777378. URL https://doi.org/10.1145/
3777378
81.Y. Li, W. Zhang, Y. Yang, W.C. Huang, Y. Wu, J. Luo,
Y. Bei, H.P. Zou, X. Luo, Y. Zhao, C. Chan, Y. Chen,Z.Deng,Y.Li,H.T.Zheng,D.Li,R.Jiang,M.Zhang,
Y. Song, P.S. Yu. Towards agentic rag with deep reason-
ing: A survey of rag-reasoning systems in llms (2025).
URLhttps://arxiv.org/abs/2507.09477
82.M.T.Alam,D.Bhusal,L.Nguyen,N.Rastogi. Ctibench:
Abenchmarkforevaluatingllmsincyberthreatintelli-
gence (2024). URL https://arxiv.org/abs/2406
.07599
83.Z. Yang, P. Qi, S. Zhang, Y. Bengio, W.W. Cohen,
R. Salakhutdinov, C.D. Manning. Hotpotqa: A dataset
for diverse, explainable multi-hop question answering
(2018). URL https://arxiv.org/abs/1809.09600