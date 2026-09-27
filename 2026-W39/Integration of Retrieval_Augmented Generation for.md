# Integration of Retrieval-Augmented Generation for Knowledge Access in the ELBE Accelerator Control System

**Authors**: Najmeh Mirian

**Published**: 2026-09-23 08:58:23

**PDF URL**: [https://arxiv.org/pdf/2609.27579v1](https://arxiv.org/pdf/2609.27579v1)

## Abstract
The efficient operation of accelerator facilities increas- ingly relies on rapid access to heterogeneous operational knowledge, including logbooks, interlock reports, machine parameters, and historical archive data. At ELBE, we pro- posed a Retrieval-Augmented Generation (RAG) frame- work that integrates facility documentation and operational records into a unified AI-assisted support tool for operators. The system is expected to index electronic logbooks, ma- chine archive time-series data, and subsystem manuals using domain-adapted embeddings stored in a vector database. User queries will be expected to be processed through a large language model that retrieves the most relevant oper- ational context and generates structured, operator-oriented responses with traceable source references. This contribu- tion presents the system architecture, data integration strat- egy, and challenges toward real-time AI-assisted accelerator operation

## Full Text


<!-- PDF content starts -->

INTEGRATION OF RETRIEVAL-AUGMENTED GENERATION FOR
KNOWLEDGE ACCESS IN THE ELBE ACCELERATOR CONTROL
SYSTEM
N. Mirian∗
Helmholtz-Zentrum Dresden-Rossendorf HZDR, Dresden Germany
Abstract
The efficient operation of accelerator facilities increas-
ingly relies on rapid access to heterogeneous operational
knowledge, including logbooks,interlock reports, machine
parameters,andhistoricalarchivedata. AtELBE,wepro-
posed a Retrieval-Augmented Generation (RAG) frame-
work that integrates facility documentation and operational
records into a unified AI-assisted support tool for operators.
The system is expected to index electronic logbooks, ma-
chinearchivetime-seriesdata,andsubsystemmanualsusing
domain-adapted embeddings stored in a vector database.
User queries will be expected to be processed through a
largelanguagemodelthatretrievesthemostrelevantoper-
ationalcontextandgeneratesstructured,operator-oriented
responseswithtraceablesourcereferences. Thiscontribu-
tion presents the system architecture, dataintegration strat-
egy, and challenges toward real-time AI-assisted accelerator
operation.
INTRODUCTION
Modernacceleratorfacilitiesoperatewithincreasingtech-
nicalcomplexityandstringentavailabilityrequirements. Re-
liable beam delivery depends on the coordinated perfor-
mance of radio-frequency systems, magnets, cryogenics,
diagnostics, vacuum, and interlock subsystems, each pro-
ducing large volumes of operational data. Machine states
aredocumentedinelectroniclogbooks,archivedtime-series
databases, subsystem reports, and technical manuals. While
essential for troubleshooting and performance optimization,
thisinformationistypicallydistributedacrossheterogeneous
and weakly connected data sources.
AttheelectronlinearacceleratorELBE(ElectronLinac
for beams with high Brilliance and low Emittance) at the
Helmholtz-Zentrum Dresden-Rossendorf (HZDR) [1], oper-
atorsfrequentlyanalyzehistoricalmachinestatestodiagnose
beaminterruptions,interlocktrips,RFinstabilities,orperfor-
mance drifts. This process often requires manual inspection
oflogbookentries,correlationofarchivedmachineparam-
eters, andconsultationofsubsystemdocumentation. Asa
result,troubleshootingcanbetime-consumingandhighly
dependentonoperatorexperience,particularlywhensimilar
faultpatternsoccurredinthepastbutaredifficulttoretrieve
through conventional keyword-based searches.
Thegrowingvolumeofarchivedoperationaldatamakes
systematicknowledgereuseincreasinglychallenging. Tra-
ditional databasequeries efficientlyaccess numericaltime-
∗n.mirian@hzdr.deseriesdatabutdonotsupportsemanticsearchacrosstextual
documentation. Conversely, large language models alone
lackreliableaccesstofacility-specificoperationalrecords.
A method that combines semantic retrieval of structured
andunstructureddatawithcontrolledlanguagegeneration
is therefore desirable.
In this work, we propose a Retrieval-Augmented Genera-
tion (RAG) [2] framework for operational support at ELBE.
The system will integrate electronic logbooks, archived ma-
chine parameters, interlock reports, and subsystem manuals
intoaunifiedretrievalpipelinebasedondomain-adaptedem-
beddingsandavectordatabase. Userqueriesareintendedto
processbyalargelanguagemodelthatretrievesrelevanthis-
toricalcontextandwillgeneratestructuredresponseswith
source traceability. The goal is to reduce troubleshooting
time and improve access to accumulated operational knowl-
edge.
OPERATIONAL CONTROL SYSTEM AND
DATA ENVIRONMENT AT ELBE
The electron linear accelerator ELBE operates in
continuous-wave (CW) mode as a multi-user facility pro-
viding electron beams and secondary radiation for scientific
applications. Commissioned in 2001, the facility consists
of a superconducting RF linac [3], an injector system, mag-
netic beam transport lines, undulator sections, and multiple
experimental beamlines. Reliable operation requires contin-
uous monitoring and coordination of RF systems, magnet
powersupplies,cryogenicinfrastructure,vacuumsystems,
diagnostics, and machine protection components.
The ELBE control system follows a hierarchical indus-
trial automation architecture [4] based on the IEC 62241-
1 standard [5]. It integrates accelerator subsystems using
several industrial control technologies, with ongoing de-
velopments focusing on improved device integration and
data accessibility via OPC UA [6,7]. The architecture is
organizedintoseverallayers(seeFig.1). Theprocessand
facility level (Level 0) comprises the physical accelerator
infrastructure including electron sources, LINACs, beam-
lines,targets,andutilitysystems. Thefieldandcontrollevel
(Level1)containssensors,actuators,front-endelectronics,
PLCs, IOCs, and distributed I/O systems responsible for
machine control, diagnostics, and interlocks, including a
fast machine protection system based on CPLD hardware
logic. The process management level (Level 2) provides op-
erator interfaces and high-level applications such as WinCC
SCADA [6], LabVIEW diagnostic tools [8], and EPICS
GUIs [9] for beamline, LLRF, and timing control. At the
arXiv:2609.27579v1  [physics.acc-ph]  23 Sep 2026

Figure1: HierarchicalarchitectureoftheELBEcontrolsystembasedontheIEC62241-1automationmodel. Thefigure
was generated using ChatGPT [10]
.
top, the enterprise management level (Levels 3–4) supports
operational management functions including beam schedul-
ing, electronic logbooks, maintenance planning, and user
management. Operational data at ELBE can be broadly cat-
egorized into three classes:Structured time-series data,
consistingofarchivedmachineparametersstoredinhistor-
ical databases with subsystem-dependent sampling rates;
event-based records,includinginterlocktriggers,subsys-
temwarnings,andmachinestatetransitions;andunstruc-
tured textual documentation,primarilyelectroniclogbook
entriesdescribingmachineoperation,anomalies,tuningpro-
cedures, and recovery actions. Although these data sources
are individually accessible, they are not semantically linked.
Correlatingbeaminterruptionswitharchivedparametervari-
ationsandrelevantlogbookentries typicallyrequiresman-
ual cross-referencing acrossmultiple systems. This lack of
unified semantic access represents a significant operational
bottleneck, particularly when similar fault patterns occurred
previously but are difficult to identify. These heterogeneous
datasourcesformthebasisfortheRAGframeworkdescribed
in the following sections.
RAG IMPLEMENTATION AND DATA
INTEGRATION
The proposed framework aims to reduce troubleshoot-
ing time and improve access to accumulated operational
knowledge by enabling unifiednatural-language accessto
heterogeneous operational knowledge. Key requirements
includecompatibilitywithstructuredtime-seriesdataandun-
structured documentation, robustness to accelerator-specific
terminology and abbreviations, traceability of generated
responses to original sources, and low-latency interactionsuitableforoperatorworkflows. Thesystemisexpectedto
bedesignedasadecision-supporttoolanddoesnotperform
control actions.
Data Ingestion and Normalization
The ingestion pipeline is expected to integrate electronic
logbooks,interlockandmachineprotectionrecords,archived
process variables, and subsystem documentation. Textual
sources (logbooks and manuals) would be converted to a
commonformatandsegmentedintosemanticallycoherent
chunks. Each chunk would be enriched with metadata such
as time stamps, subsystem labels (RF, magnets, vacuum,
cryogenics, diagnostics, machine protection), component
identifiers, and source type. Event-based records are in-
tended to be stored as structured entries containing event
time, subsystem, and affected components, with links to rel-
evantdocumentationwhereavailable. Time-seriesarchive
datawouldberetrievedwithinuser-definedtimewindows.
Parameter names would be normalized and associated with
subsystemmetadata,andselectedarchivesegmentswould
be summarized (e.g. statistics or trends) to provide compact
contextual information for language model queries.
Embedding and Retrieval
Text chunks and event records are intended to be em-
beddedusingadomain-adaptedembeddingmodelandin-
dexed in a vector database together with metadata. This
wouldenablehybridretrievalcombiningsemanticsimilarity
searchwithmetadataconstraintssuchassubsystemfilters
or time ranges. When a query is submitted, the system
wouldperformquerynormalization(includingacronymand
component-name handling), retrieve the top- 𝑘candidate
contexts,andoptionallyre-rankthembasedonsubsystem

Data Sources
Logbook
Archive
Interlocks
ManualsIndexing Layer
Chunking
Metadata
Embeddings
Vector DBRetrieval Layer
Similarity Search
Metadata Filtering
Time AlignmentGeneration Layer
LLM
Structured Output
Source Traceability
Figure 2: Layered architecture of the RAG-based operational support framework. Heterogeneous accelerator data are
normalizedandembeddedintoavectordatabase. Userqueriestriggersemanticretrievalwithmetadataandtimeconstraints
before structured response generation.
relevance, time proximity, and source reliability. The re-
trieved contexts are intended to be assembled into a struc-
tured prompt for the language model.
Response Generation and Traceability
The large language model [11,12] is expected to operate
strictlyinaretrieval-augmentedmode,generatingresponses
conditioned on the retrieved contexts. Outputs would be
structuredforoperatoruseandmayincludeshortsummaries,
suspected causes, and suggested checks. Each response
would include explicit references to the retrieved logbook
entries, event records, archive segments, or documentation
used to generate the answer, ensuring transparency and sup-
porting validation in a safety-critical environment.
Time-Correlated Archive Context
To correlate events with machine parameter behavior, the
framework aims to support time-aligned retrieval. Event
time stamps would be used to retrieve archive variables
within configurable pre- and post-event windows. Retrieved
segments would be summarized using statistical descriptors
such as extrema, mean values, and trends, and inserted into
the language model context together with relevant logbook
and event records.
Current Limitations
Currentchallengesincludeinconsistentnamingconven-
tions across subsystems, variability in logbook documenta-
tion quality, and imperfect time synchronization between
datasources. Retrievalperformancealsodependsoncorpus
coverage and update frequency. Proposed work would focus
onimprovedsubsystemtagging,richermetadataintegration,
andevaluationofretrievalperformanceandlatencyunder
realistic operator queries.
SYSTEM ARCHITECTURE AND RAG
PIPELINE
TheRAGframeworkisproposedtobeimplementedas
amodulararchitectureseparatingdataingestion,semantic
indexing, retrieval, and response generation. Figure 2 illus-
tratestheoverallsystemarchitecture. Thedesignsupports
traceable responses, subsystem-aware filtering, and integra-
tion of structured and unstructured accelerator data. The
system consists of four main layers:Data Layer:Electronic logbooks, interlock and machine
protectionrecords,archivedprocessvariables,andsubsys-
tem documentation.
Indexing Layer:Text chunking, metadata enrichment,
embedding generation, and storage in a vector database.
Retrieval Layer:Hybridsemanticsearchcombiningsim-
ilarity retrieval with metadata filtering and time constraints.
Generation Layer:Context-conditioned response gen-
erationusingalargelanguagemodelwithenforcedsource
traceability.
Data Processing and Indexing
Textual sources (logbooks, manuals, technical reports)
are expected to be segmented into semantically coherent
chunks and enriched with metadata fields including subsys-
temlabel,componentidentifier,timestamp(whenavailable),
and source type. Event-based records such as interlock trig-
gerswouldbestoredasstructuredentriescontainingevent
time,subsystem,andaffectedcomponents. Archivedtime-
series process variables are planned to be accessed through
a dedicated interface. For semantic integration, selected
timewindowswouldbesummarizedintocompactstatistical
descriptors before being included in the language model
context when required. All textual and event records would
beembeddedusingadomain-adaptedembeddingmodeland
storedtogetherwithmetadatainavectordatabasesupporting
similarity search and structured filtering.
Query and Retrieval Pipeline
When an operator submits a query, the system aims to
execute the following workflow:
Input: User query Q
1. Normalize Q (acronyms, components, time hints)
2. Compute embedding v_Q
3. Retrieve top-k contexts from vector database
4. Apply metadata filtering (subsystem, time)
5. If event time detected:
retrieve corresponding archive window
compute statistical summary
6. Assemble prompt {Q, contexts, archive summary}
7. Generate response with LLM and source references
Output: Structured response with traceable sources
This hybrid retrieval approach is expected to ensure that
generated responses remain grounded in facility-specific
operational data.

REFERENCES
[1]Helmholtz-ZentrumDresden-Rossendorf(HZDR). ELBE–
center for high-power radiation sources.
[2]Patrick Lewis et al. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InAdvances in Neural In-
formation Processing Systems, 2020.
[3]J. Teichert et al. RF status of superconducting module de-
velopmentsuitableforCWoperation: ELBEcryostats.Nu-
clear Instruments and Methods in Physics Research Section
A, 557:239, 2006.
[4]M. Justus, R. Steinbrueck, and K. Zenker. Control system of
theELBEelectronacceleratorfacility. Helmholtz-Zentrum
Dresden-Rossendorf, ELBE User Facility.
[5]IEC 62264-1:2013, enterprise-control system integration –
part 1: Models and terminology, 2013.
[6] Siemens AG. SIMATIC automation systems.
[7] OPC Foundation. OPC unified architecture.
[8] National Instruments. National instruments.
[9]EPICSCollaboration. Experimentalphysicsandindustrial
control system.
[10] OpenAI. ChatGPT: Large language model assistant, 2026.
[11]Tom B. Brown et al. Language models are few-shot learners.
InAdvances in Neural Information Processing Systems,2020.
[12]Wayne Xin Zhao et al. A survey of large language models.
ACM Computing Surveys, 2023.