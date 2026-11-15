# Chain of Physics: Supporting Fault Diagnosis with Engineering-Inspired Large Language Model Reasoning

**Authors**: Wenqi Zhou, Leonardo Baldo, Kai Goebel, Manuel Arias Chao

**Published**: 2026-11-10 16:09:44

**PDF URL**: [https://arxiv.org/pdf/2611.14263v1](https://arxiv.org/pdf/2611.14263v1)

## Abstract
Fault diagnosis in complex systems requires linking observed changes in physical quantities to the failure mechanisms that produce them. Experienced engineers perform this task by combining sensor information with knowledge of system behavior and fault propagation. Large Language Models (LLMs) can support this process, but their diagnostic reasoning depends on the availability of asset-specific engineering knowledge and on how this knowledge is represented and used during inference. This work introduces Chain of Physics (CoP), an LLM-based fault-diagnosis framework built around a Constrained Knowledge Graph (CKG) generated automatically from asset-specific reference documentation. CoP addresses failure-sensor association from textual engineering knowledge. The CKG stores diagnostic chains connecting failure modes, changes in physical quantities, and monitoring sensors. During inference, semantically similar graph paths are retrieved for each question and used as both information and in-context examples for the reasoning process. Candidate failure modes are extracted from the question and analysed independently through dedicated LLM calls, which generate plausible failure-to-observation chains. A final decision module compares these chains with the available answer choices. CoP is evaluated on the FailureSensorIQ benchmark across four industrial asset classes and three LLMs. Compared with direct prompting and conventional Retrieval-Augmented Generation (RAG), CoP improves diagnostic accuracy across all evaluated models, with gains ranging from 7.8 to 19.6 percentage points and from 11.7 to 18.2 percentage points, respectively. The results show that structured engineering knowledge can improve LLM-based diagnostic reasoning when graph retrieval is combined with in-context examples of fault propagation.

## Full Text


<!-- PDF content starts -->

Chain of Physics: Supporting Fault Diagnosis with
Engineering-Inspired Large Language Model Reasoning
Wenqi Zhoua, Leonardo Baldoa,∗, Kai Goebelb,cand Manuel Arias Chaoa,d
aInstitute for Data Science, Zurich University of Applied Science, Winterthur, 8401, Switzerland
bFragum Global, Mountain View, CA 94040, USA
cLuleå University of Technology, Luleå, 971 87, Sweden
dFaculty of Aerospace Engineering, Delft University of Technology, HS 2926 Delft, The Netherlands
ARTICLE INFO
Keywords:
Fault diagnosis
Diagnostic reasoning
Large Language Models
Knowledge Graphs
Engineering knowledge
In-context learningABSTRACT
Fault diagnosis in complex systems requires linking observed changes in physical quantities
to the failure mechanisms that produce them. Experienced engineers perform this task by
combining sensor information with knowledge of system behavior and fault propagation. Large
Language Models (LLMs) can support this process, but their diagnostic reasoning depends
on the availability of asset-specific engineering knowledge and on how this knowledge is
represented and used during inference. This work introduces Chain of Physics (CoP), an
LLM-based fault-diagnosis framework built around a Constrained Knowledge Graph (CKG)
generated automatically from asset-specific reference documentation. CoP addresses failure–
sensor association from textual engineering knowledge. The CKG stores diagnostic chains
connecting failure modes, changes in physical quantities, and monitoring sensors. During
inference, semantically similar graph paths are retrieved for each question and used as both
information and in-context examples for the reasoning process. Candidate failure modes are
extracted from the question and analysed independently through dedicated LLM calls, which
generateplausiblefailure-to-observationchains.Afinaldecisionmodulecomparesthesechains
with the available answer choices. CoP is evaluated on the FailureSensorIQ benchmark across
fourindustrialassetclassesandthreeLLMs.Comparedwithdirectpromptingandconventional
Retrieval-AugmentedGeneration(RAG),CoPimprovesdiagnosticaccuracyacrossallevaluated
models,withgainsrangingfrom7.8to19.6percentagepointsandfrom11.7to18.2percentage
points,respectively.TheresultsshowthatstructuredengineeringknowledgecanimproveLLM-
based diagnostic reasoning when graph retrieval is combined with in-context examples of fault
propagation.
1. Introduction
Fault diagnosis in complex systems requires identifying the failure mechanisms responsible for an observed
deviation from nominal behavior. Experienced engineers and technicians perform this task by combining available
measurements with their knowledge of system behavior and reasoning about the physical quantities through which
failureeffectsbecomeobservable[1].Theserelationshipslinkobservedsensorbehaviortospecificfailuremechanisms,
enablingFaultDetectionandIsolation(FDI)andsupplyingdiagnosticinformationfordownstreamhealthassessment
andmaintenancedecisionsinPrognosticsandHealthManagement(PHM)frameworks[2].Inindustrialapplications,
the knowledge required for this process is generally distributed across technical manuals, reference documents,
troubleshooting procedures, maintenance records, and accumulated expert experience [3]. Its interpretation and
integrationthereforeremainlargelydependentonexperiencedpersonnel,whomustretrieveandreconcileinformation
scattered across heterogeneous technical documents and information systems [4, 5]. The problem considered in this
workfocusesspecificallyondiagnosisfromengineeringknowledge,notonthedirectclassificationofmeasuredsensor
signals. Given a textual description of a failure mode or sensor observation, the objective is to recover the relevant
failure–sensor association and the physical relationships that connect them. The task can therefore be viewed as
knowledge association from engineering text, requiring the relationships among failure mechanisms, their physical
effects, and the corresponding sensing modalities to be established from textual information.
The increasing interest in Large Language Models (LLMs) introduces the possibility of assisting engineers in
this knowledge-intensive task [6]. Through natural-language interaction, LLMs can process technical information,
zhou@zhaw.ch(W. Zhou);*Corresponding Author leonardo.baldo@zhaw.ch(L. Baldo);kai.goebel@fragumglobal.com
(K. Goebel);aria@zhaw.ch(M. Arias Chao)
:Preprint submitted to ElsevierPage 1 of 29
arXiv:2611.14263v1  [cs.LG]  10 Nov 2026

Graph construction once per asset
Diagnostic inference per questionReference
documentationConstrained
extraction
Complete chains onlyLLM
Clarify
Clariﬁcation of 
ambiguous questionsLLMConstrained
Knowledge Graph
Failure-to-observation paths
Diagnostic
questionRetrieve
CKG paths
As knowledge +
reasoning examplesFailure-speciﬁc
reasoning
New failure-to-observation
chainsLLM
Answer
selection
Chain of Physics
reasoningLLM
AnswerReused across
questions
question
Sensor
or failure modeQuestion & candidate failure modesFigure 1:Overview of CoP. Once per asset, constrained extraction turns reference documentation into a CKG whose paths
are complete failure-to-observation chains. For each diagnostic question, retrieved CKG paths provide both asset-specific
knowledge and reasoning examples. These guide failure-specific reasoning, which produces new chains for the candidate
failure modes. A separate answer-selection stage makes the decision based on those chains.
interpret diagnostic questions, and formulate possible explanations without requiring a dedicated model for each
asset [5]. Despite the strong push towards Artificial Intelligence (AI) adoption and the proliferation of AI initiatives
across industrial sectors, the integration of LLMs into fault diagnosis remains challenging. Conventional task-
specific models generally require representative labelled data and retraining when new assets or failure conditions
areintroduced,limitingtheirapplicabilityinindustrialsettings.LLMsoffergreaterflexibility,butoftenlacktheasset-
specific and practical engineering knowledge required to interpret abnormal conditions correctly. This limitation can
lead to hallucinated or physically inconsistent diagnostic explanations, particularly when the model must establish
relationships between failure mechanisms, their effects on system behavior, and the available sensor measurements
[7]. Their effective deployment therefore requires incorporating established engineering knowledge and guiding the
reasoning process according to the physical relationships governing the system.
One of the most widely adopted approaches to incorporate external engineering knowledge into LLM inference
is Retrieval-Augmented Generation (RAG) [8]. By retrieving relevant passages from technical documentation, RAG
can provide information beyond that acquired during pre-training without requiring task-specific model training.
Nevertheless, the availability of relevant information does not guarantee that the model can reconstruct the physical
sequence linking a fault to its measurable effects. However, in fault diagnosis, these relationships are central to the
reasoning process: a failure affects one or more physical quantities, its effects may propagate through the system, and
the resulting changes are detected by specific sensors. Representing this sequence explicitly may therefore provide
a more suitable basis for diagnostic reasoning than supplying unstructured technical information alone. This type of
explicit failure-to-observation reasoning has a long tradition in engineering diagnosis, including qualitative physics
andmodel-baseddiagnosis[9],SignedDirectedGraph(SDG)representations[10],andFailureModeandSymptoms
Analysis(FMSA)[11].Theseapproachessimilarlyrelyonstructuredrelationshipsbetweenfailuremechanisms,their
physicalmanifestations,andtheobservationsthroughwhichtheycanbedetected.Inrecentyears,KnowledgeGraphs
(KGs)haveemergedasapossiblemeansofrepresentingthistypeofdiagnosticstructurethroughexplicitentitiesand
relationships[12].Forfaultdiagnosis,aKGcanrepresenttheprogressionfromafailuremodetothephysicalchanges
and the measurements through which those changes become observable. Even more, the close integration of LLMs
and KGs is increasingly being investigated for industrial fault diagnosis [13]. In fact, LLMs provide natural language
processing and reasoning capabilities, while KGs organize domain knowledge into explicit entities and relationships.
Their integration allows diagnostic systems to retrieve and use structured engineering knowledge during inference,
supporting more informed and interpretable diagnostic reasoning. Still, the practical integration of LLMs and KGs
intoanindustrialdiagnosticframeworkpresentstwomaindifficulties[14].First,manuallyconstructingasset-specific
knowledge representations requires considerable engineering effort, particularly when multiple systems and failure
mechanismsareconsidered.Second,agraphcontainingrelevantentitiesandgenericrelationshipsdoesnotnecessarily
preserve the physical sequence required for diagnostic reasoning.
:Preprint submitted to Elsevier Page 2 of 29

Againstthisbackdrop,thisworkintroducesChainofPhysics(CoP)asanengineering-inspiredframeworkforLLM-
assistedfaultdiagnosisbasedontheCKG,aimedtotackletheseexacttwochallenges.Themethodologyfollowstheway
anexperiencedengineerapproachesadiagnosticproblem:plausiblefailuremechanismsareconsidered,theirphysical
consequencesaretracedthroughthesystem,andtheresultingeffectsareconnectedtotheavailablemeasurements.This
placesCoPwithinthebroaderlineofworkthattranslateshumanreasoningproceduresintoexplicitintermediatesteps
that can be reproduced algorithmically during LLM inference [15, 16]. CoP organizes both knowledge extraction and
diagnosticreasoningaroundthesamefailure-to-observationstructure.Figure1summarizesthislogic,withtheoffline
construction of reusable asset knowledge on the left and the question-specific reasoning process on the right. During
graphconstruction,asset-specificreferencedocumentationisprocessedautomaticallyandcandidatechainsareretained
only when the source material supports a failure mode, at least one change in a physical quantity, and a monitoring
sensor.Theseelementsarerepresentedthroughafixedsetofrelationsdescribingtheinitialfailureeffect,itspropagation
through the system, and its observation by a sensor. The resulting CKG is therefore a diagnostic representation of the
source knowledge in which each complete path corresponds to a physically structured failure-to-observation chain.
Since the graph is generated once for each asset, the extracted knowledge can be reused across subsequent diagnostic
questions. During inference, relevant CKG paths are retrieved for the question under analysis and used in two ways.
Theyprovideasset-specificinformationaboutfailuremechanismsandmeasurableeffects,andtheyaresuppliedtothe
LLMasin-contextexamplesofthereasoningstructuretofollow.Candidatefailuremodesextractedfromthequestion
arethenanalysedindependently,allowingthemodeltogeneratenewfailure-to-observationchainsthatarenotrestricted
topathsalreadycontainedinthegraph.Aseparatedecisionstagegivestheanswerbasedonthegeneratedchains.CoP
thereforecombinesgraph-groundedretrievalwithin-contextreasoningoveraconstrainedengineeringrepresentation,
while maintaining a clear separation between failure-specific reasoning and final answer selection.
TheproposedapproachisevaluatedusingFailureSensorIQ,anopen-sourcebenchmarkaddressingtherelationships
between industrial failure modes and monitoring sensors. The assessment considers two complementary tasks:
identifying relevant sensors for a given failure mode and identifying possible failure modes from sensor evidence.
Experiments are conducted using three LLMs on four asset classes: an aero gas turbine, an electric motor, a fan,
and a power transformer. CoP is compared with direct LLM prompting and conventional RAG based on the same
referencedocumentation.Anablationstudyfurtherexaminesthecontributionofquestionclarificationandtheseparate
diagnostic-reasoninganddecisionstages.TheresultsshowthatCoPimprovesdiagnosticaccuracyacrossallevaluated
models,supportingtheuseofexplicitlystructuredengineeringknowledgetoassistfaultdiagnosiswithouttask-specific
model training. In summary, the main contributions of this work are:
•An automatic procedure for constructing an asset-specific CKG from technical reference documentation. The
graph is restricted to complete diagnostic chains linking failure modes, changes in physical quantities, and
monitoring sensors through a fixed set of relations.
•AdiagnosticinferenceframeworkthatusesretrievedCKGpathsasbothengineeringknowledgeandin-context
reasoning exemplars. These paths guide the generation of failure-specific chains without requiring the target
failure mode to be explicitly represented in the retrieved graph paths.
•A staged reasoning procedure that separates question clarification, failure-mode extraction, graph retrieval,
failure-specific reasoning, and final answer selection, allowing the physical reasoning process to be performed
independently from the multiple-choice decision.
•AnevaluationontheFailureSensorIQbenchmarkacrossfourindustrialassetclassesandthreeLLMs,including
comparisons with direct prompting and conventional RAG and an ablation analysis of the main inference
components.
The remainder of the paper is organised as follows. Section 2 reviews related work on LLM-based fault
diagnosis, KGs, RAG, and prompt-based reasoning. Section 3 presents the proposed Chain of Physics framework,
including the offline construction of the CKG and the online diagnostic inference procedure. Section 4 introduces
the FailureSensorIQ benchmark and the experimental setting. Section 5 presents and discusses the results. Finally,
Section 6 summarises the main findings and outlines directions for future work.
:Preprint submitted to Elsevier Page 3 of 29

2. Related Works
The proposed framework lies at the intersection of three research directions: (i) the use of external knowledge
during LLM inference, (ii) the representation of domain knowledge through graph-based structures, and (iii) the use
of in-context examples to guide reasoning. These areas have developed largely in parallel, but their combination is
particularlyrelevantforfaultdiagnosis,wherebothasset-specificknowledgeandthestructureofthereasoningprocess
matter. The following paragraph briefly introduces representative developments from the broader Machine Learning
(ML)literaturethatarerelevanttothemethodologicalbuildingblocksconsideredinthiswork,includingRAG,graph-
basedknowledgeaccess,andIn-ContextLearning(ICL).Afterthisbroadermethodologicaloverview,thesecondpart
of the section reviews works more directly connected to PHM and fault diagnosis. The purpose of the next paragraph
is then not to provide a comprehensive review of current ML research, but to establish the technical background and
state of the art needed to position the proposed framework. For a broader overview of the interaction between LLMs
and structured knowledge, the interested reader is referred to Pan et al. [14], who provide a comprehensive review of
KG-enhanced LLMs, LLM-augmented KGs, and complete LLM–KG frameworks.
A central limitation of LLMs in knowledge-intensive applications is their dependence on information acquired
during pre-training. This has motivated the development of methods that provide models with external knowledge
at inference time. RAG, introduced by Lewis et al. [17], is one of the most established approaches in this direction.
Conventional RAG retrieves relevant passages from an external document collection and supplies them to the model
as additional context for question answering or text generation. Beyond unstructured text retrieval, subsequent work
has investigated how external knowledge can be represented and accessed in a more structured form. Jiang et al. [18]
proposed StructGPT, which enables LLMs to interact with structured data through dedicated interfaces. Graph-based
approaches then extended this idea by exploiting explicit entities, relations, and paths during retrieval and reasoning.
Wang et al. [19], for example, introduced Knowledge Graph Prompting, where graph construction and traversal are
usedtoretrievesupportingevidencefrommultipledocuments.Luoetal.[20]developedReasoningonGraphs(RoG),
in which relation paths are used to guide the retrieval of graph-grounded evidence. More recently, Ma et al. [21]
proposedThink-on-Graph2.0,whichcombinesstructuredgraphretrievalwithunstructuredtextualinformation.These
approaches show how graph structure can be used to organize external knowledge and make relational information
directly available during LLM inference. A related line of research concerns ICL, where examples included in the
input prompt guide the model in performing a task without modifying its parameters. The choice of these examples
can substantially affect model behavior, motivating retrieval-based approaches in which demonstrations are selected
dynamicallyforeachinput[22,23].Thisprinciplehasalsobeenappliedtograph-structuredknowledge.Weietal.[24],
forinstance,proposedKICGPT,wherestructuralknowledgeretrievedfromaKGisencodedintodemonstrationsthat
guidetheLLMduringknowledge-graphcompletion.Inthissetting,retrievaldeterminesboththeinformationsupplied
to the model and the examples used to shape its reasoning process.
In this sense, the use of LLMs in PHM has received increasing attention, with recent studies investigating their
use across different stages of the PHM process. In their paper, Li et al. [35] reviewed the development of large-
scalefoundationmodelsforPHM,outliningtheirpotentialapplications,methodologicalrequirements,andremaining
challenges.Chenetal.[36]reviewedlargemodelsformachinemonitoringandfaultdiagnostics,whileTsallisetal.[37]
provided a systematic review of large language and foundation models for machinery health monitoring. Razaq et al.
[5]conductedasystematicliteraturereviewof37studiescombiningLLMsandKGsforfaultdiagnosis,identifyingtwo
dominantintegrationparadigms(i.e.,automatedKGconstructionandKG-grounded(GraphRAG-style)reasoning)and
reportingthatcurrentworkconcentratesalmostexclusivelyonrotatingmachinery,steelmanufacturing,andComputer
NumericalControl(CNC)equipment,highlightinglimitedmultilingualcoverageandpersistentchallengesconcerning
the explainability of diagnostic reasoning. Other contributions have investigated broader frameworks intended to
supportmultiplePHMtasks.Chaetal.[38],forinstance,proposedanLLM-basedautonomousagentthatorchestrates
data preprocessing, model selection, training, and optimization for fault classification and Remaining Useful Life
(RUL)prediction.Taoetal.[39]introducedtheconceptofaPHMLargeModel(PHM-LM),discussingitsunderlying
paradigms and the technical challenges associated with its development and application.
Alongside these broader frameworks, a growing body of research addresses individual PHM functions, including
fault detection and isolation, prognostics, and maintenance decision support. Fault diagnosis is particularly relevant
forknowledge-basedLLMapplications,astheinterpretationofabnormalconditionsoftenrequirescombiningsensor
evidencewithinformationcontainedintechnicalmanuals,troubleshootingprocedures,andmaintenancerecords.Such
:Preprint submitted to Elsevier Page 4 of 29

Table 1
Representative studies grouped by methodological similarity and their relationship to CoP.
Group Representative studies Key characteristic Distinction from CoP
Reasoning
procedure without
persistent KGGong et al. [16] Structured prompting and
agent-based architectures, including
multi-agent debate and planning.No reusable, offline-built
engineering-knowledge representation.
KG-based symptom
/ solution retrievalLin and Miao [25]; Nie et
al. [26]; Liu et al. [27]Graph retrieval links symptoms and
faults to causes and maintenance
actions.Do not explicitly impose CoP’s
constraining schema.
Graph-grounded
retrieval and
reasoningGu et al. [28]; Xie et al.
[29]; Miao et al. [30]Retrieved causal paths or subgraphs
provide structured, query-specific
context to support subsequent LLM
reasoning.Evaluated in individual application
domains, without a common
cross-asset benchmark or not
reproducible.
Causal /
mechanism-explicit
KGGuo et al. [31] Weighted causal graph of
fault-propagation paths derived from
monitoring data.Constructed from operational
measurements instead of reference
technical documentation.
Automatic
knowledge
construction from
documentationZhou et al. [32]; Wang et
al. [33]; Marandi et al. [34]Automatic extraction of structured
knowledge from technical documents.
Marandi et al. also reuse the resulting
representation for graph traversal and
LLM-based reasoning.Use different ontologies or functional
representations, without CoP’s
predefined constraining schema.
Comparable
evaluation designMa et al. [7]
(FDRKG-LLM)Comparison of a KG-grounded LLM
with direct prompting and multiple
conventional RAG baselines.Single equipment class, with
open-ended generation metrics
instead of multi-asset multiple-choice
accuracy.
information is frequently expressed in natural language, making it accessible to LLM-based reasoning and question-
answering approaches.
BeforeconsideringrecentLLM-basedapproachestothisproblem,itisworthrecallingthatexplicitrepresentations
of physical behaviour have a long history in engineering fault diagnosis. Classical work on qualitative physics
represented system behaviour through qualitative changes in physical quantities and causal relationships, providing a
basisforreasoningabouthowphysicaleffectspropagatethroughengineeredsystems[9].Closelyrelatedmodel-based
diagnostic approaches infer possible faults by comparing manifested system behaviour with the behaviour predicted
fromexplicitmodelsofcomponentstructureandfunction.Inprocessdiagnosis,causalandSDGrepresentationshave
similarlybeenusedtodescribequalitativerelationshipsamongprocessvariablesandsupportthepropagationoffault
effects [10]. FMSA provides a further connection from the condition-monitoring domain, relating identified failure
modes to their symptoms and to the monitoring techniques suitable for their detection [11].
TheuseofLLMsforfaultdiagnosisanddiagnosticreasoningisthereforetheresearchdirectionmostcloselyrelated
tothepresentworkandrequiresadeeperlook.Themainaxisalongwhichdifferentapproachescanbedistinguishedis
thewayengineeringknowledgeismadeavailabletothemodelandsubsequentlyused:(i)somemethodsrelyprimarily
on knowledge acquired during pre-training, (ii) others retrieve relevant passages from technical documentation, (iii)
while a third group organizes domain information into structured representations such as KGs.
Approaches based mainly on pretrained model knowledge use prompting, reasoning strategies, or agent-based
decomposition without introducing an external persistent representation of asset-specific engineering knowledge
[16]. As already mentioned, their main limitation is that diagnostic performance depends on information already
encoded in the model parameters, which may be incomplete, too generic, or inconsistent with the the physics of
the specific asset under analysis. This becomes particularly critical when diagnosis requires detailed relationships
betweenfailuremechanisms,localchangesinphysicalquantities,andthesensorsthroughwhichtheseeffectsbecome
observable. A second group augments the LLM with unstructured technical information retrieved from manuals,
maintenancedocuments,ortroubleshootingmaterial.Retrievalcanproviderelevantasset-specificevidenceandreduce
the dependence on pre-training knowledge. However, the retrieved passages remain expressed in natural language
and the model must reconstruct the diagnostic relationships contained in them during inference. Relevant evidence
may therefore be available in the context without explicitly preserving the sequence through which a failure produces
:Preprint submitted to Elsevier Page 5 of 29

physical effects and eventually becomes observable at a sensor. Lukens et al. [8], for example, investigated the use
of technical manuals within a RAG framework for fault diagnosis, providing a direct example of document-based
knowledge augmentation in PHM. Finally, structured approaches address this limitation by representing engineering
knowledgeexplicitlythroughentitiesandrelations.Inthesemethods,knowledgeextractionandrepresentationbecome
part of the diagnostic framework, allowing information about components, failure mechanisms, symptoms, and
monitoringvariablestobereusedduringinference.However,theusefulnessoftheresultinggraphdependsstronglyon
the adopted representation. Generic entity–relation graphs can store relevant domain information without necessarily
preserving the physical progression required to reason from a failure mechanism to its measurable effects. A further
refinement of this third category is provided by constrained knowledge representations, in which the admissible
relations between entities are defined a priori according to the semantics of the target problem. In this case, the
graph structure itself restricts how knowledge can be represented and connected, reducing arbitrary relation types
and enforcing paths that follow a predefined engineering logic. For diagnostic applications, such constraints can be
used to preserve the progression from a failure mechanism, through changes in physical quantities, to the sensor that
observes their effects.
The next paragraph focuses on applications of this last paradigm, which is particularly relevant and promising
for fault-diagnosis reasoning. Among the applications combining LLMs and structured diagnostic knowledge, Liu
et al. [40] proposed a joint KG–LLM model for fault localization in aviation assembly. The method incorporates
KG embeddings through prefix-tuning and uses knowledge retrieval to support the identification of fault causes and
troubleshootingsolutions.Maetal.[7]introducedFDRKG-LLM,combiningafaultdiagnosticreasoningKGwithan
LLMtosupportequipmentdiagnosisandmaintenancedecisions.Theirevaluationcomparestheproposedframework
with direct prompting and conventional RAG configurations on a self-constructed diagnostic database, reporting
improvements in text-generation metrics. Liu et al. [27] and Nie et al. [26] investigated related approaches for CNC
equipment, combining a domain KG, retrieval, and engineering information to support interactive fault diagnosis.
Guoetal.[31]combinedconstrainedcausaldiscoverywithLLM-enhancedknowledge-graphcompletionforrotating
machinery,derivingaweightedcausaldirectedacyclicgraphthattracesfault-propagationpathsfrommonitoringdata.
A comparably large empirical gain was reported by Lin and Miao [25], who improved automotive fault-diagnosis
accuracyusingadual-modeknowledge-graphandLLMretrievalscheme.Thesestudiesestablishtheuseofstructured
engineering knowledge in LLM-assisted industrial diagnosis, with different approaches to knowledge integration and
model adaptation.
Since the integration of LLMs with structured engineering knowledge appears especially promising for scalable
and reproducible fault diagnosis, the next question is: how are diagnostic knowledge representations created? For
instance, Liao et al. [41] investigated LLM-assisted extraction of fine-grained information from robotic maintenance
texts for KG construction, and comparable automatic pipelines have been proposed for automotive [32] and aviation
[42] components. Wang et al. [33] proposed GraphRAG-KM, an automated framework that converts industrial
technicaldocumentationintoontologyandconceptualmodelsthroughdocumentparsing,graph-basedretrieval,LLM
extraction, and clustering. Marandi et al. [34] proposed a framework for nuclear power plant diagnostics in which
a hierarchical functional model is automatically constructed from technical documentation and encoded into a KG.
The resulting representation supports both structured graph traversal and LLM-based interpretive reasoning through
Graph-RAG.Thesecontributionsaddressthepreparationandreuseofengineeringknowledge,includingitsintegration
into diagnostic inference.
Otherapproaches investigatehow retrievedknowledge canbe usedto structurethe reasoningprocess itself.Gong
etal.[16]proposedamulti-agentframeworkthatgeneratesdiagnosticreasoningblueprintsandadaptsthesubsequent
decision process according to question complexity. Evaluated on FailureSensorIQ [43], the framework improves
Llama-3.1-8Baccuracyfrom36.5%to54.6%.Guetal.[28]combineddomainknowledgeanddata-driveninformation
to construct a unified causal graph for fault diagnosis in distributed computing systems, supplying retrieved paths as
in-context reasoning exemplars. Xie et al. [29] and Miao et al. [30] investigated related graph-grounded reasoning
approaches for sensor networks and train bogies, respectively. Gao et al. [44] proposed LLM-FDQA, a KG-enhanced
framework for industrial fault diagnosis that combines semantic extraction, two-stage entity linking, and multi-hop
causal reasoning. The method retrieves and validates symptom–cause–solution chains to support diagnostic question
answering. Evaluated on a proprietary CNC fault-diagnosis dataset, it reports an end-to-end diagnostic accuracy of
94.04%.Thesestudiesshowhowgraph-derivedevidencecanbeusedbothasretrievedknowledgeandasstructurefor
subsequent diagnostic reasoning.
:Preprint submitted to Elsevier Page 6 of 29

Reproducibility remains a practical limitation when these approaches are compared. Several studies rely on
proprietary datasets or provide insufficient implementation detail for a controlled reimplementation of the complete
reasoning pipeline. The paper from Gong et al. [16] is among the few works evaluated on the public FailureSensorIQ
benchmark whose reasoning architecture is described in sufficient detail to reproduce. This makes their framework
particularly suitable for the additional controlled comparison reported in Section 5.7.
The present work combines these research directions within a single diagnostic framework. From the perspective
ofclassicaldiagnosticmethodsdiscussedabove,theCKGfollowsthesamegeneralprincipleofrepresentingexplicitly
therelationshipsbetweenafailureanditsobservablemanifestations,whileadoptingastructuretailoredtoLLM-based
reasoning.Itspathscontainfailuremodes,directionalchangesinphysicalquantities,andmonitoringsensorsconnected
throughaconstrainedsequenceofrelations.Moreover,themethodologyisshapedbythewayanengineerapproachesa
diagnosticproblem,byfollowingthephysicalrelationshipsthatconnectafailuremechanismtothechangesitproduces
inthesystemandtothemeasurementsthroughwhichthosechangesbecomeobservable.CoPautomaticallyconstructs
anasset-specificCKGfromreferencedocumentation,retainingchainsthatconnectfailuremodes,changesinphysical
quantities,andmonitoringsensors.Duringinference,query-conditionedgraphretrievalrecoverscompleteCKGpaths
that provide both asset-specific engineering knowledge and in-context exemplars of failure-to-observation reasoning.
CandidatefailuremodesareextractedindependentlyfromthediagnosticquestionandanalysedthroughseparateLLM
callsusingtheretrievedchainsasreasoningexamples.Theresultingfailure-specificchainsarethenpassedtoaseparate
Judgemodulethatselectsthefinalanswer.CoPthereforecombinesgraph-groundedretrievalwithretrieval-conditioned
in-context reasoning and a distinct final decision stage.
3. Methodology
Evaluating diagnostic reasoning is not straightforward. In realistic applications, the available evidence, operating
conditions, and level of diagnostic detail can vary substantially, making direct comparison between reasoning
approachesdifficult.Acommonwayofreducingthisvariabilityistoformulatethetaskasquestionanswering,where
theinformationrequiredtoreachtheansweriscontrolledandthefinalresponsecanbeevaluatedagainstaknowntarget.
InKnowledgeGraphQuestionAnswering(KGQA),thissettingiscommonlydistinguishedbetweensimplequestions,
whoseanswercanbeobtainedfromasinglerelation,andmorecomplexmulti-hopquestions,whichrequirecombining
several entities and relations before reaching the target answer [45, 46]. This formulation provides a practical way of
evaluatingwhetheramodelcanrecoverisolatedassociationsorcomposeasequenceofintermediaterelationsanditis
oneofthestandardmethodologiestotestandcomparereasoningalgorithminMLalsothankstothevariouspublished
ML datasets.
Asimilarsettingiscommonlyusedfordiagnosticreasoning.Afailure-mode-to-sensorquestionmaybeanswered
throughadirectassociationwhentherelevantrelationisexplicit,oritmayrequirereconstructingasequenceinwhich
afailureaffectsoneormorephysicalquantitiesbeforeitseffectbecomesobservableatasensor.FailureSensorIQ[43]
is, to the best of the authors’ knowledge, one of the few and the most comprehensive publicly available benchmarks
designed specifically to evaluate this type of reasoning across industrial assets. The dataset formulates diagnosis as
multiple-choice question answering and tests both directions of the failure-mode–sensor relationship: identifying the
sensors relevant to a given failure mode and inferring plausible failure modes from abnormal sensor observations.
Accordingly, the task considered here differs from conventional fault classification based on measured sensor data.
FailureSensorIQ evaluates diagnostic association from textual information. Given a failure mode, the model must
identifythesensingmodalitymostrelevanttoitsdetection,or,intheinversedirection,determinewhichfailuremodes
are consistent with a stated sensor observation. An example of a Failure Mode to Sensor (FM2Sensor) question is
reported below, where the model receives a diagnostic stem and a fixed set of candidate sensor classes and must
identify the right (gold) response.
Example Diagnostic Question
Stem.In an aero gas turbine, when air inlet blockage occurs, which sensor from the choices is most critical in
detecting the occurrence of the failure event?
:Preprint submitted to Elsevier Page 7 of 29

AGas generator temperature
BPower turbine temperature
COil debris
DCompressor temperature Gold
EPressure / pressure ratio
The example also illustrates a practical characteristic of the benchmark: diagnostic questions are not always
expressed with fully standardised or technically precise wording. Since FailureSensorIQ is generated through a
template-based procedure, this should not be interpreted as reproducing real industrial conversations. However,
redundant formulations, broad sensor names, and ambiguous expressions are present in the benchmark and resemble
thelinguisticimprecisionthatcanoccurintechnicaldiagnosticrequests.Thismakesthedatasetsuitableforevaluating
a pipeline that must first establish a clear interpretation of the question before reasoning about the underlying failure.
CoP organizes diagnostic reasoning around the physical consequences expected from each candidate failure
mechanism. The procedure follows the way an engineer approaches diagnosis by tracing the changes that a possible
causewouldproduceinthesystemandidentifyingthemeasurementsthroughwhichthosechangesbecomeobservable.
For failure-mode-to-sensor questions, the reasoning proceeds from a specified failure to its observable effects. For
sensor-to-failure-mode questions, candidate causes are assessed through their expected effects and their relevance to
the stated observations. The method comprises an offline stage, performed once for each asset, in which reference
documentationisconvertedintotheCKG,andanonlinestage,executedforeachdiagnosticquestion,inwhichretrieved
chainssupportfailure-specificreasoningandsubsequentanswerselection.Figure2visualizesthemodularity:theupper
part constructs the reusable asset-specific CKG, while the lower part shows how that representation is used for each
diagnostic question.
AllLLMcallsinCoPusededicatedinstructiontemplatesdesignedfollowingestablishedprompt-designprinciples:
the instructions define the task of each module, the context made available to it, the permitted operations, and the
required output structure as indicated in [47, 48]. Complex diagnostic reasoning is decomposed into separate calls,
retrievedfailure-to-observationchainsaresuppliedasin-contextdemonstrationswhereappropriate[15],andstructured
outputsareusedtoconstraintheinformationexchangedbetweenstages.Thecompletepromptsforallcallsarereported
in Appendix A.
3.1. Offline step: CKG generation from Reference Knowledge
Theofflinestageisperformedonceforeachassetandconvertsthecorrespondingreferencedocumentationintothe
CKGusedduringonlineinference.AsillustratedinFigure2,theprocedurecomprisesdifferentoperations:document
parsingandsegmentation,extractionofgroundedfailure-to-observationchains,andaggregationoftheacceptedchains
into the asset-specific CKG.
3.1.1. Document parsing and segmentation
The reference documentation is first converted from PDF to Markdown using Docling [49]. The parsing pipeline
preserves the document structure and enables the extraction of tables and mathematical expressions, while Optical
Character Recognition (OCR) can be enabled when required by the source document. The resulting Markdown
representation is then divided into text units of approximately 2,000 characters, which are processed independently
inthefollowingstage.Thissizehasbeenselectedfollowingthecommonlyrecommendedstartingpointforfixed-size
chunking [50] and lies within the range shown to benefit retrieval on technical, context-dependent documents [51].
This preprocessing step provides a common textual representation of the source material while keeping the
subsequent LLM calls local to relatively short portions of the document. Each unit is treated independently, so the
generation of a diagnostic chain is based only on the information contained in that portion of the reference text.
3.1.2. Grounded diagnostic-chain extraction
Each text unit is passed to a dedicated LLM call using a structured output schema designed specifically for the
diagnosticrepresentationadoptedinCoP.TheextractionpromptisreportedinAppendixA.1.Foreachunit,themodel
is asked to extract one coherent monitoring scenario composed of a failure mode, an ordered sequence of changes
in physical quantities, and the sensor that observes the final quantity. Each intermediate step contains the physical
:Preprint submitted to Elsevier Page 8 of 29

ONLINE · PER QUESTION
1Question
FailureSensorIQ
multiple choice
ABCD E2Clarify
Resolve ambiguity by 
rewording; leave 
unchanged if 
unambiguous. 
Choice order is kept.
LLM3bExtract failure modes
What failure is being asked about?
Read directly from the clariﬁed 
question and choices
(No CKG used).
LLM3aRetrieve & expand
What reasoning examples exist in the
CKG?
Embed question + choices; top-k (k 
=2) entity seeds by cosine similarity; 
walk stored edges forward or 
backward to complete chains.4Reason about each failure mode
Using that form, what physical chain could this
failure produce?
For each extracted failure mode, infer 
plausible failure → quantity changes → 
sensor chains, following the reasoning 
pattern provided by the retrieved chains.
LLMRETRIEVED EXEMPLAR · FORM
results in leads to leads to monitored by
same structure
INFERRED · NEW CONTENT5Judge
Compare all
reasoning chains
with the choices;
select one option.
LLM
question + choicesAnswerConstrained Knowledge Graph (CKG)
3 ﬁxed relations: results in, leads to, is monitored by
failure mode physical quantity · location · behavior sensorDUAL ROLE
① Condensed engineering knowledge
② Reasoning exemplar bank
Reference
textbookDivision into
chunksGrounding gate
LLMLLM keeps a chunk only if
one scenario contains all of:
1a failure mode
21–5 changes in physical
quantities at locations
3a monitoring sensoraccepted
rejected
if there is no 
failure-to-sensor 
linkChain → triples
failure mode
results in
quantity @ location ↑/↓
leads to (0–4×)
quantity @ location ↑/↓
is monitored by
sensormerge(a) Build the reasoning structure
(b) Use the reasoning structureOFFLINE · ONCE PER ASSET
stored chains
stored CKG chain
inferred chain
LLM LLM callFigure 2:Complete CoP methodology. The framework consists of an offline stage, performed once for each asset, and an
onlinestageexecutedforeachdiagnosticquestion.Offline,theasset-specificreferencedocumentationisdividedintochunks
and processed through a grounding gate. The accepted chains are converted into triples using the fixed relationsresults in,
leads to, andis monitored by, and subsequently merged into the CKG. Online, the clarified question is used independently
to retrieve and expand relevant CKG paths and to extract the failure modes under consideration. The retrieved paths guide
the generation of new diagnostic chains for each failure mode, after which the Judge maps the resulting reasoning to the
available answer choices.
variable, its location on the asset when available, and the direction of change, restricted toincreasesordecreases.
Between one and five physical-variable changes are allowed in each chain to avoid excessively long chains.
The extraction prompt acts as a grounding gate. A chain is retained only when the text unit supports the complete
relationship between a failure mode, at least one physical-variable change, and a monitoring sensor within the same
scenario. General theoretical descriptions, historical information, design equations, or passages without an explicit
failure-to-monitoring relationship are therefore discarded. The model is also instructed to use only information
contained in the current text unit, without introducing additional physical propagation steps. When several diagnostic
scenarios are present, the dominant one is selected.
The structured LLM output is subsequently checked before graph generation. A candidate chain is rejected if the
failure mode or sensor is missing, if no physical-variable step is available, or if any step does not contain a physical
variable. Repeated node descriptions within the same chain are also rejected, since their direct conversion into the
graphrepresentationcouldintroduceanartificial cycle.Thissecondvalidationthereforecomplementstheconstraints
imposed through the extraction prompt. For an accepted result, the resulting sequence has the form
failure mode→physical change1→⋯→physical change𝑛→sensor,where𝑛∈ [1,5].
Throughoutthechain,eachphysical-changenodecombinesthevariable,itslocationwhenreportedinthesource,and
its direction of change.
3.1.3. Graph construction and aggregation
EachacceptedchainisconverteddirectlyintoasmallgraphusingthethreerelationsdefinedbytheCKGschema.
The transition from the failure mode to the first physical effect is assigned the relationresults in; intermediate
propagation steps are connected throughleads to; and the final physical effect is connected to the corresponding
sensor throughis monitored by. The graph structure is therefore generated directly from the ordered chain returned
by the extraction stage. A separate graph is produced for every accepted text unit. Once all units associated with an
:Preprint submitted to Elsevier Page 9 of 29

asset have been processed, the resulting graphs are aggregated by taking the union of their entities, edge types, and
relations using KGGen [52]. The outcome is a single asset-specific CKG containing the failure-to-observation chains
supported by the reference documentation.
This construction procedure imposes the same reasoning structure across all retained knowledge. Each complete
pathstartsfromafailuremechanism,followsthephysicalquantitiesthroughwhichitseffectspropagate,andterminates
at the sensor through which the final effect becomes observable. The intuitive structure of the CKG allows experts to
cross check and edit the information easily. The CKG can consequently be reused during online inference both as a
condensedrepresentationofasset-specificengineeringknowledgeandasarepositoryofcompletediagnosticreasoning
examples.
TheextractedchainsusedintheevaluationwerereviewedbyaSubject-MatterExpert(SME)andcorrectedwhere
requiredbeforeonlineinference.Theconstrainedrepresentationmakesthisreviewdirect:theinitiatingcondition,every
physical-variablechange,itsdirection,andthemonitoringsensorareexposedasanorderedpathwhoseindividualsteps
can be checked against engineering knowledge. The resulting knowledge bases remain compact enough for complete
expertinspection. Section5.1 andFigure 4showan excerptof theresulting CKGand illustratehowthe storedchains
can be followed and verified relation by relation.
3.2. Online step: Inference on a Specific Question
Theonlinestage,reportedinthelowerhalfofFigure2,isexecutedindependentlyforeachdiagnosticquestionusing
theCKGpreviouslyconstructedforthecorrespondingasset.Itspurposeistocombinetheinformationcontainedinthe
questionwiththereasoningpatternsstoredinthegraph,whilekeepingknowledgeretrieval,failure-modeidentification,
physical reasoning, and answer selection as separate operations.
3.2.1. Question clarification
The first operation is a clarification step performed by a dedicated LLM call using the prompt reported in
Appendix A.2. The rationale follows a common pattern in human diagnostic reasoning: when a problem statement
is unclear or ambiguous, an engineer first reformulates it to obtain a more precise interpretation before analysing
the underlying failure mechanisms. The model receives the original question together with all candidate answers
and determines whether the wording is sufficiently clear for the subsequent reasoning stages. If no ambiguity is
identified, the original question and options are retained unchanged. Otherwise, the model reformulates the question
and, where required, the answer options while preserving their number, order, meaning, asset, and diagnostic intent.
The clarification module is explicitly instructed not to select an answer or introduce failure modes or sensors that are
notimpliedbytheoriginalitem.Ifthestructuredoutputcannotbeparsed,orifthenumberofreturnedoptionsdiffers
from the original one, the pipeline falls back to the unmodified question.
The resulting question and options are used by all subsequent modules. The clarification stage therefore acts only
on the interpretation of the diagnostic problem and does not access the CKG. Figure 5 later shows a question that is
left unchanged, while Appendix B reports a complementary worked example in which ambiguity is detected and the
question is reformulated before inference.
3.2.2. CKG retrieval and failure-mode extraction
After clarification, the pipeline separates into two parallel operations.
The first retrieves relevant reasoning examples from the asset-specific CKG. A retrieval query is formed by
concatenating the clarified question and all answer options. This query is embedded and compared through cosine
similarity with cached embeddings of the failure-mode and sensor entities contained in the CKG. The top-𝑘highest-
scoringentitiesareretainedasretrievalseeds,where𝑘= 2intheexperimentsreportedinthiswork.Theselectedseeds
are then expanded along the graph to recover complete failure-to-sensor chains. This operation corresponds to graph
traversal, i.e., following the stored relations from a selected node to reconstruct a complete path. When the retrieved
seedisafailuremode,thealgorithmfollowstherelationsforwarduntilamonitoringsensorisreached.Whentheseed
is a sensor, the relations are followed backward until the corresponding failure mode is reached. The path expansion
follows the constrained graph structure and recovers complete chains of the general form
failure mode→physical effect→⋯→sensor,
using the relationsresults in,leads to, andis monitored by. Duplicate paths are removed before the retrieved chains
are supplied to the reasoning stage.
:Preprint submitted to Elsevier Page 10 of 29

In parallel, a second LLM call extracts the failure modes mentioned or implied by the clarified question and its
answer options using the prompt reported in Appendix A.3. This operation is independent of the CKG. The extracted
failure modes are not required to correspond to nodes already present in the graph. The result is a list of candidate
failure modes defining what must subsequently be analysed.
This separation assigns different roles to the two branches to define what to reason for and how to reason. The
question determines the failure modes under consideration, while the CKG provides asset-specific examples of how
failureeffectspropagatetowardmeasurablequantities.Thetargetfailuremodethereforedoesnotneedtobeexplicitly
represented in the retrieved graph paths.
3.2.3. Retrieval-conditioned in-context reasoning
The reasoning stage follows an ICL formulation. As already mentioned, in ICL, examples placed in the input
context guide the model in performing the target task without updating its parameters [22]. Retrieval-based variants
select these demonstrations dynamically according to the current input, allowing the context presented to the model
to change from one query to another [23]. Related graph-based approaches have used structured KG knowledge as
in-context demonstrations to guide LLM inference [24].
CoP adopts retrieval-conditioned few-shot prompting at the level of diagnostic reasoning. Complete CKG chains
are dynamically retrieved for each question and supplied as demonstrations of the failure-to-observation structure
to follow. The complete CKG chains retrieved in the previous step are inserted into the prompt as examples of the
expected failure-to-observation reasoning structure. They therefore provide both asset-specific engineering context
and demonstrations of the sequence "failure mode→physical quantity changes→sensor".
A separate LLM call is performed for every extracted failure mode using the prompt reported in Appendix A.4.
ThisseparationmirrorstheengineeringreasoningprocessadoptedthroughoutCoP.Eachcandidatefailureisexamined
throughitsexpectedphysicalconsequencesbeforethefinaldiagnosticcomparison.Themodelreceivestheassetname,
the target failure mode, the retrieved CKG paths, and the chain format associated with the graph. Its task is to trace
plausible changes in physical quantities and identify the sensors capable of observing them. The reasoning module
does not receive the multiple-choice question or its answer options directly; the assessment of the generated chains
against the diagnostic question is assigned to the subsequent Judge module. Using the retrieved chains as in-context
reasoningexemplars,togetherwithitsdomainknowledge,theLLMgeneratesoneormoreplausiblechainslinkingthe
target failure mode to intermediate physical effects and ultimately to sensors capable of observing those effects. The
generatedchainsarenotrequiredtocorrespondtopathsalreadystoredintheCKG.Thestructuredoutputofthisstage
is therefore a set of newly inferred diagnostic chains for each extracted failure mode.
3.2.4. Final judgement
Once all failure modes have been analysed, their inferred chains are passed to a separate LLM-based decision
module,hereafterreferredtoastheJudge,togetherwiththeclarifiedquestionandthecompletesetofansweroptions.
TheJudgepromptisreportedinAppendixA.5.TheJudgethenselectsasingleoptionidentifierbasedonthegenerated
diagnostic reasoning. The resulting inference procedure separates retrieval of stored graph knowledge, generation of
failure-specificdiagnosticchains,andmatchingofthosechainstotheavailableanswerchoices.Thisseparationassigns
failure-specific analysis and final answer selection to distinct stages. The reasoning module receives candidate failure
modes and retrieved exemplars, while the Judge receives the diagnostic question and answer choices to assess the
relevance of the generated consequences. The answer choices can therefore influence reasoning indirectly through
retrieval and failure-mode extraction.
4. Dataset
The proposed approach is evaluated on FailureSensorIQ [43], one of the most complete and one of the few
publicly available benchmarks specifically designed to assess the ability of LLMs to establish relationships between
failure modes and monitoring sensors across different industrial assets. Developed by IBM Research, the dataset
comprises8,296multiple-choicequestionscoveringtenassetclasses.Thequestionsaregeneratedthroughatemplate-
based procedure using condition-monitoring knowledge from International Organization for Standardization (ISO)
documentation, including ISO 17359. The benchmark addresses two complementary diagnostic tasks: FM2Sensor,
which identifies the sensors relevant to a given failure mode, and Sensor-to-Failure-Mode (Sensor2FailureMode),
whichidentifiespossiblefailuremodesassociatedwithasensorobservation.Bothtasksincludepositiveandnegative
:Preprint submitted to Elsevier Page 11 of 29

Table 2
Industrial assets considered in the evaluation, reference documentation used for CKG construction, and number of single-
correct FailureSensorIQ questions [43].
Asset Reference documentation Questions
Aero gas turbineGas Turbine Theory, 7th ed. [53] 336
Electric motorNoninvasive Methods for Condition Monitoring and Electrical Fault Diagnosis of Induction
Motors[54]234
FanImproving Fan System Performance: A Sourcebook for Industry[55] 200
Power transformerTransformers: Basics, Maintenance, and Diagnostics[56] 544
Total 1,314
formulations, requiring the model to identify either the most relevant or the irrelevant option among the available
answers.
FailureSensorIQ provides 2,667 Single-Correct Multiple-Choice Question Answering (SC-MCQA) items and
5,629 Multiple-Correct Multiple-Choice Question Answering (MC-MCQA) items. Consistently with the primary
evaluation setting adopted in the original benchmark, this study considers only the single-correct subset. The
assessment is further restricted to four asset classes for which suitable reference documentation was available to
construct the corresponding CKGs: aero gas turbine, electric motor, fan, and power transformer. The resulting
evaluation set comprises 1,314 questions. Table 2 reports the selected assets, the reference documents used for graph
construction, and the number of questions considered for each asset.
An additional consideration worth mentioning concerning the interpretation of accuracy on FailureSensorIQ is
raised in the original dataset study by Constantinides et al. [43]. The benchmark is generated from failure-mode–
sensor associations contained in ISO documentation and expressed through predefined question templates. These
engineering relationships can be many-to-many: a failure mechanism may affect several measurable quantities and
maythereforebeobservablethroughdifferentsensingmodalities.Thesingle-correctformulationneverthelessrequires
one designated answer for each item, and expressions such as "most relevant sensor" consequently impose a ranking
amongrelationshipsthatmayallbetechnicallyplausibletodifferentdegrees.Somesensordescriptionsarealsobroad
or technically underspecified, as illustrated by the "length measurement" example discussed in Appendix B.
This characteristic should be considered when interpreting the absolute accuracy values reported in the following
section.TheoriginalFailureSensorIQstudyevaluatedfivehumanparticipantsonabalancedsubsetof80questionsand
reported a mean expert accuracy of approximately 60%, with the best-performing expert reaching 66.19%. Inter-rater
agreement among the experts was moderate (𝜅= 0.462) [43]. These results do not establish a formal upper bound
for the benchmark, since the human evaluation covers only a subset of the questions and expertise differs across asset
classes. They do, however, indicate that disagreement and ambiguity are present even among human domain experts.
Accordingly, no matter the methodology, the gap to 100 per cent accuracy should not be interpreted solely as model
error, since a portion of the residual error may reflect ambiguity in the benchmark formulation and the non-unique
nature of some failure–sensor relationships.
5. Results
The evaluation was conducted on the 1,314 single-correct FailureSensorIQ questions retained for the four assets
considered in this work. The section evaluates the proposed solution from the construction of the knowledge
representation to the final diagnostic decision. Section 5.1 first examines the CKGs produced offline and their
inspectability. Section 5.2 then follows a worked reasoning example. Sections 5.3–5.5 assess diagnostic accuracy,
thecontributionoftheinferencestages,andconsistencyacrossassets.Section5.7removestheCKGentirelytoisolate
the contribution of the failure-to-observation reasoning structure. Together, these analyses assess the quality of the
constructed knowledge, the benchmark performance of CoP, the role of its individual stages, and the contribution of
its reasoning representation.
5.1. CKG construction
Table3summarisestheoutcomeoftheofflinegraph-constructionprocedure.Thenumberofacceptedchainsvaries
substantiallyacrossthefourreferencedocuments.From981textchunksextractedfromtheaerogasturbinetextbook,
:Preprint submitted to Elsevier Page 12 of 29

Table 3
Outcome of the offline CKG construction for the four evaluated assets.
Asset Chunks Chains Pass rate Entities Relations
Aero gas turbine 981 55 5.6% 209 164
Electric motor 38 14 36.8% 44 35
Fan 152 25 16.4% 99 79
Power transformer 333 135 40.5% 429 348
0 20 40 60 80 100
% of chainsAero gas turbine
Electric motor
Fan
Power transformerChain length (FM → … → sensor)
2 hops 3 hops 4 hops 5 hops0 100 200 300 400
# entitiesAero gas turbine
Electric motor
Fan
Power transformerEntity roles in merged CKG
failure mode intermediate sensor
Figure 3
Structure of the four CKGs. Left: distribution of chain lengths, measured as the number of relations between the failure
mode and the sensor. Right: number of failure-mode, intermediate physical-change, and sensor entities in each merged
graph.
55completediagnosticchainswereretained,correspondingtoapassrateof5.6%.Thefansourceproduced25chains
from 152 chunks (16.4%), while the electric-motor and power-transformer documents produced 14 from 38 chunks
(36.8%) and 135 from 333 chunks (40.5%), respectively.
Figure3describesthestructureoftheresultinggraphs.Mostchainscontaintwoorthreerelations,corresponding
to one or two intermediate changes in physical quantities between the failure mode and the sensor. Longer chains,
withuptofiverelations,aremainlyextractedfromthefanandaerogasturbinedocuments,whereastheelectric-motor
graphcontainsnochainlongerthanfourrelations.Forallassets,intermediatephysical-changeentitiesformthelargest
group of nodes.
The extraction rates in Table 3 reflect the selective purpose of the CKG. Reference documents contain extensive
theoretical,historical,anddesignmaterial,whiletheextractionprocedureretainspassagessupportingcompletefailure-
to-observation chains. The resulting graphs contain between 14 and 135 accepted chains per asset, providing a
manageable body of diagnostic knowledge for independent SME review. This compactness is a practical advantage
oftherepresentation:eachretainedexampleexposesthephysicalrelationshipsusedtoconnectaninitiatingcondition
to a measurable effect.
Figure 4 illustrates a section of the aero gas turbine CKG. The stored chains can contain different numbers of
intermediate physical changes, reflecting the amount of propagation supported by the source material. The fixed
relations distinguish the initial effect, subsequent propagation, and sensor observation. This structure makes every
chainreadablefromtheinitiatingfailuretothemonitoredquantityandallowsanengineertocheckitscorrectnessstep
by step. The figure therefore also illustrates the practical basis for the SME review described in Section 3.1.3.
During inference, retrieved chains serve as engineering knowledge and few-shot demonstrations of diagnostic
reasoning. Their role is to provide complete examples of how physical effects connect an initiating condition to
observable quantities. CoP therefore uses a selective set of reviewed diagnostic exemplars to guide the analysis of
candidate failure modes, including those absent from the retrieved paths.
5.2. Analysis of the reasoning process
The illustrative example in Figure 5 illustrates one case of how the CoP inference procedure operates in practice.
Asecondexamplereportingthecompleteend-to-endoutputofeveryonlinestage,includingretrievalseeds,expanded
CKGchains,inferredchains,andJudgereasoning,isprovidedinAppendixC.Thequestionconcernsamountingfault
inafanandaskswhichoffivecandidatemeasurementshasthehighestdiagnosticrelevance.Theclarificationmodule
:Preprint submitted to Elsevier Page 13 of 29

initiating condition physical-variable change monitoring sensorturbine overheatingturbine core temperature
increasesgas-generator speed
decreasesgas-generator speed
sensor
turbine inlet
temperature increases
ﬂame-out
during rapid climbfuel ﬂow rate
decreasescombustor temperature
at combustor chamber
decreasesturbine inlet
temperature 
decreases
build-up of deposits
on compressor bladingcompressor isentropic 
efﬁciency decreasesturbine operating 
temperature increasesturbine inlet
temperature sensor
TIT probe
excessive accessory
power demandfuel ﬂow at combustor
increasesturbine inlet	
temperature increasesannulus temperature 
(thermocouple reading) 
increasesthermocouple arraynozzle area reduction
turbine inlet temperature 
exceeds safe limitnozzle area
decreases
results in
leads to
results in
results in
results in
leads to
results in
is monitored by
 leads to
is monitored by
leads to
 leads to
is monitored by
results in
leads to
is monitored by
is monitored by
leads to
results in
is monitored by
is monitored byFigure 4:Excerpt from the aero gas turbine CKG. Orange nodes denote initiating conditions, blue nodes denote physical-
variable changes, and green nodes denote monitoring sensors. The relationsresults in,leads to, andis monitored
bydistinguish initial effects, physical propagation, and observation. The explicit chain structure supports step-by-step
inspection by domain experts.
identifiestheitemassufficientlyclearandleavesthequestionunchanged.Thefailure-modeextractionstageidentifies
mounting fault in fanas the condition to be analysed.
In parallel, the retrieval stage searches the fan CKG using the clarified question and answer options. The two
recoveredchainsconcerncontaminantbuilduponfanbladesandfoulingoftheinlet-guide-vanelinkage.Theirphysical
pathsterminateatapowermeterandapressuretransducer,respectively.Neitherretrievedchaincontainsthemounting
fault appearing in the question or directly points to vibration, which is the correct answer.
The reasoning stage then uses these stored chains as examples of the expected failure-to-observation structure
and applies the same reasoning pattern to the extracted mounting fault. Among the generated chains, the model
links mounting faults to loose mounting bolts, shaft misalignment, and transmitted imbalance forces. Two of these
chains terminate in structural or lateral vibration measurements, while another terminates in motor-current or power
measurement.TheJudgesubsequentlycomparesthecompletesetofinferredchainswiththeavailableanswerchoices
and selectsvibration, corresponding to the gold label.
ThisexampleisusefulbecausetheretrievedCKGpathsprovidelimitedtopicaloverlapwiththetargetfailuremode.
Their contribution lies in supplying an asset-specific representation of how a failure is connected to physical effects
and observable quantities. The new mounting-fault chains are generated during the reasoning stage and are therefore
distinctfromthepathsstoredintheCKG.Figure5consequentlyshowstheintendedinteractionbetweengraphretrieval
andLLMinference:thegraphprovidesstructureddiagnosticexamples,whilethemodelextendsthesamestructureto
the failure mode under analysis.
The trace should be interpreted as an explanatory example of the inference process, while the quantitative results
reported in the previous sections provide the evidence for the observed performance improvements. The clarification
and decision stages further improve overall accuracy. The results are based on one run for each model with zero
temperature.
5.3. Diagnostic accuracy
The overall accuracy obtained by the three approaches is reported in Table 4. CoP achieves the highest accuracy
for all three models. With Claude Opus 4.7, accuracy increases from 51.98% with direct prompting and 50.61% with
RAG to 68.80% with CoP. The same trend is observed for the smaller models: Llama 3.1 8B increases from 30.75%
:Preprint submitted to Elsevier Page 14 of 29

Clear question + choices
Retrieved
CKG
Chains
All inferred chains1Question
“When dealing with a mounting fault in a fan, which sensor among the choices
has the highest relevance in detecting this issue?”
Aoil debris Bvibration Clength measurement Dpower Espeed2Clarify
LLMQuestion deemed unambiguous, 
hence no changes. 3bExtract failure mode
LLM"mounting fault in fan"What failure mode must be 
analysed?Is the question ambiguous 
or unclear?
3aRetrieve & expand What reasoning examples exist in the CKG?
Top-2 seeds · expanded to 2 complete chains
CKG chain 1
Contaminant buildup on fan blades
results in
blade surface contamination level
increases
leads to
fan efﬁciency decreases
leads to
electrical power consumption at motor
increases
is monitored by
Power meterCKG chain 2
Fouling of mechanical linkage
controlling inlet-guide vanes
results in
contaminant accumulation on linkage
components increases
leads to
inlet-guide vane angle deviates /
decreases
leads to
fan static pressure at discharge
increases
is monitored by
Pressure transducer measuring fan
discharge pressure4Reason about the failure mode
Using the retrieved CKG chains as examples of the diagnostic reasoning pattern,
what physical chains could this failure mode produce?
One LLM call → 7 candidate chains (3 shown)LLM
Mounting fault in fan
results in
loosening of mounting bolts at
fan base increases
leads to
structural vibration amplitude at
fan housing increases
is monitored by
Accelerometer / vibration
sensor on fan housingresults in
misalignment between fan shaft
and motor shaft at coupling
increases
leads to
lateral vibration at bearing
housing increases
is monitored by
Vibration sensor on bearing
housingresults in
dynamic imbalance forces
transmitted to motor at motor
mount increases
leads to
motor current ﬂuctuation at
motor increases
is monitored by
Motor current sensor / power
meterClear
question
+ choices
Target failure mode
5Judge
Sees question, options and all
inferred chainsLLM –Multiple chains directly indicate vibration.
–Power is a weaker indirect indicator.
–Oil debris, length measurement, and speed have little direct relevance.CHAIN SUPPORT
B vibration
D power
A, C, EAnswer
B — vibrationFigure 5:Example of the CoP inference process for a fan mounting fault. The input question is first assessed by the
clarification module and is left unchanged because no ambiguity is detected. The target failure mode is extracted directly
from the clarified question, while the CKG retrieval stage independently recovers complete stored chains associated with
the most relevant graph entities. These paths provide examples of the expected failure–physical effect–sensor reasoning
structure. The reasoning stage then generates new plausible chains for the mounting fault, including paths leading to
structural and lateral vibration measurements. The Judge receives the question, answer choices, and all inferred chains,
and selects vibration as the final answer.
Table 4
Overall diagnostic accuracy on the 1,314 single-correct FailureSensorIQ questions.
Model Direct RAG CoP
Claude Opus 4.7 51.98 50.6168.80
Llama 3.1 8B 30.75 26.8638.51
Qwen 2.5 7B 30.52 35.2450.15
to38.51%,whileQwen2.57Bincreasesfrom30.52%to50.15%.Overall,thegainoverdirectpromptingrangesfrom
7.8 to 19.6 percentage points, while the gain over conventional RAG ranges from 11.7 to 18.2 percentage points. The
absolute values should be interpreted in light of the benchmark characteristics discussed in Section 4.
The comparison with RAG is particularly relevant because both approaches rely on the same reference docu-
mentation and the same initial chunking procedure. The difference therefore lies in the representation and use of the
retrievedinformation.RAGprovidesthemodelwithsemanticallysimilartextpassages,whileCoPretrievescomplete
failure-to-observationchainsfromtheCKGandusesthemasreasoningexamples.Theresultsindicatethatthestructure
imposed on the engineering knowledge has a substantial influence on the final diagnostic performance. This effect is
visibleinallthreemodelsandisespeciallypronouncedforQwen2.57B,forwhichCoPimprovesaccuracybyalmost
15 percentage points over RAG.
The RAG results also show that the availability of additional textual context does not translate consistently into
higher accuracy. This result is aligned with the conclusion in [43], where a RAG pipeline is implemented. Claude
Opus4.7andLlama3.18BbothobtainslightlylowerperformancewithRAGthanwithdirectprompting,whileQwen
:Preprint submitted to Elsevier Page 15 of 29

Table 5
Ablation study of the CoP inference procedure.
Model w/o clarify + Judge w/o Judge Full CoP
Claude Opus 4.7 66.36 67.8168.80
Llama 3.1 8B 35.16 36.6138.51
Qwen 2.5 7B 43.53 48.5550.15
Direct
w/o clarify + Judgew/o Judge Full CoP20304050607080Accuracy (%)
+14.4+1.4+1.0
68.8Claude Opus 4.7
Direct
w/o clarify + Judgew/o Judge Full CoP
+4.4+1.4+1.9
38.5Llama 3.1 8B
Direct
w/o clarify + Judgew/o Judge Full CoP
+13.0+5.0+1.6
50.2Qwen 2.5 7B
Figure 6
Accuracy gain contributed by each inference stage, starting from direct prompting. Each point adds one component to the
previous configuration. Labels report the gain in percentage points.
2.5 7B benefits from retrieved context. This variability suggests that the contribution of retrieval depends on how
effectively the model can reconstruct the relevant diagnostic relationship from the retrieved passages. CoP reduces
this burden by retrieving information already organised according to the failure–physical effect–sensor structure used
during reasoning.
5.4. Contribution of the inference stages
The ablation study in Table 5 examines how the additional inference stages affect the performance obtained from
theCKG.AllconfigurationsusethesameCKG,thesameembedding-basedseedselection(𝑘= 2),andthesamepath
expansion. The configurations differ in the LLM calls surrounding retrieval:
•CoPw/oclarify+Judge.Theoriginalquestionandoptionsareuseddirectlyforretrieval.Noclarificationcall,
failure-mode extraction call, failure-specific reasoning call, or separate Judge is used. One LLM call receives
theoriginalquestion,theansweroptions, theretrievedCKGchains,andthechainpattern,andjointlyproduces
the diagnostic reasoning and the selected option.
•CoP w/o Judge.The clarification call is added before retrieval. The clarified question and options are then
passed to the same single answer-generation call used in the previous configuration. Failure-mode extraction
and separate failure-specific reasoning are still absent, and reasoning and option selection remain within one
LLM call. The comparison with the previous configuration therefore isolates the effect of clarification.
•Full CoP.The clarification stage is retained, and the single answer-generation call is replaced by separate
failure-modeextraction,onereasoningcallforeachextractedfailuremode,andthefinalJudgecalldescribedin
Section3.2.Thereasoningcallsdonotreceivetheanswerchoicesdirectly.ThecomparisonwithCoPw/oJudge
thereforeevaluatesthecombinedeffectoffailure-modeextraction,separatefailure-specificreasoning,andfinal
answer selection through the Judge.
TheconfigurationcombiningCKGretrievalwithasingleanswer-generationcallalreadyreaches66.36%,35.16%,
and 43.53% for the three models, respectively. These values account for a large part of the improvement observed
with the complete method and support the role of the constrained graph representation itself. Clarification adds 1.45
percentage points for Claude Opus 4.7 and Llama 3.1 8B, and 5.02 points for Qwen 2.5 7B. The full staged inference
procedure adds a further 0.99, 1.90, and 1.60 percentage points, respectively. Figure 6 shows the same results as a
:Preprint submitted to Elsevier Page 16 of 29

Table 6
Diagnostic accuracy (%) by asset class, model, and inference configuration. CKG denotes graph retrieval with direct answer
generation; CoP denotes the complete inference procedure, including clarification, separate failure-specific reasoning, and
final judgement. The number of evaluated questions is reported once for each asset. Bold values indicate the highest
accuracy within each row.
Asset (𝑁) Model Direct RAG CKG CoP
Aero gas turbine (336) Claude Opus 4.7 43.45 44.05 59.8261.01
Llama 3.1 8B 29.17 30.06 34.5238.99
Qwen 2.5 7B 27.68 31.85 37.8049.40
Electric motor (234) Claude Opus 4.7 66.67 54.70 66.6775.21
Llama 3.1 8B 36.32 24.3640.60 40.60
Qwen 2.5 7B 29.91 35.47 44.4450.43
Fan (200) Claude Opus 4.7 51.50 51.00 63.0065.00
Llama 3.1 8B 37.50 27.00 43.5045.50
Qwen 2.5 7B 38.00 46.50 53.0060.00
Power transformer (544) Claude Opus 4.7 51.10 52.76 71.5172.24
Llama 3.1 8B 26.84 25.92 30.1534.74
Qwen 2.5 7B 29.78 33.09 43.0146.88
    Aero gas
  turbineElectric
motor
Fan
Power
transformer20406080Claude Opus 4.7
    Aero gas
  turbineElectric
motor
Fan
Power
transformer20406080Llama 3.1 8B
Direct RAG CoP    Aero gas
  turbineElectric
motor
Fan
Power
transformer20406080Qwen 2.5 7B
Figure 7
Diagnostic accuracy (%) per asset for direct prompting, RAG, and CoP with the three evaluated LLMs.
cumulativesequencestartingfromdirectprompting.Forallthreemodels,thelargestsingleincreaseisobtainedwhen
CKG retrieval is introduced, while clarification and the Judge provide smaller additional gains.
5.5. Accuracy across asset classes
Table 6 reports diagnostic accuracy separately for each asset class, including the intermediate inference configu-
rations. Full CoP outperforms direct prompting and conventional RAG in all twelve asset–model combinations. The
gainsrangefrom4.28to21.14percentagepointsoverdirectpromptingandfrom8.93to20.51percentagepointsover
RAG. The aggregate improvement is therefore observed across all four evaluated asset classes. Figure 7 reports the
samecomparisonforeachasset.WithClaudeOpus4.7andQwen2.57B,CoPobtainsthehighestaccuracyonallfour
assets.WithLlama3.18B,themarginsaresmaller.CoPismostclearlyaheadontheelectricmotorandthefan,while
the three approaches remain close on the aero gas turbine and the power transformer.
:Preprint submitted to Elsevier Page 17 of 29

FM→S
n=226S→FM
n=226FM→S
n=431S→FM
n=431020406080Accuracy (%)select relevant select irrelevantClaude Opus 4.7
FM→S
n=226S→FM
n=226FM→S
n=431S→FM
n=431select relevant select irrelevantLlama 3.1 8B
FM→S
n=226S→FM
n=226FM→S
n=431S→FM
n=431select relevant select irrelevantQwen 2.5 7BDirect RAG CoP w/o clarify + Judge CoP w/o Judge CoPFigure 8:Accuracy (%) by question type for direct prompting, RAG, and the three CoP configurations. (FM→S: failure
mode to sensor; S→FM: sensor to failure mode;𝑛: number of questions of each type.)
The intermediate stages show greater variation. For Llama 3.1 8B, clarification reduces accuracy from 34.52%
to 32.44% on the aero gas turbine and from 40.60% to 40.17% on the electric motor. Adding separate failure-specific
reasoningandfinaljudgementreducesfanaccuracyfrom46.50%to45.50%forLlama,andpower-transformeraccuracy
from72.98%to72.24%forClaudeandfrom47.98%to46.88%forQwen.ItleavesQwen’sfanaccuracyunchangedat
60.00%. These results qualify the aggregate ablation: the additional stages improve overall accuracy, but their benefit
depends on the asset and underlying model.
Theseresultsalsoindicatethatthecontributionoftheindividualstagesdependsontheunderlyingmodel.Question
clarification has a limited effect for Claude Opus 4.7 and Llama 3.1 8B, while its effect is more marked for Qwen 2.5
7B.TheJudgeprovidesasmallerincreaseinaggregateaccuracyacrossallthreemodels,althoughtheasset-levelresults
show exceptions.
5.6. Accuracy by question type
FailureSensorIQ contains four question types, obtained by combining the reasoning direction (from failure mode
to sensor, FM→S, or from sensor to failure mode, S→FM) with the polarity of the question (selecting the relevant or
the irrelevant option). Figure 8 reports the accuracy of all configurations for each type.
WithClaudeOpus4.7,thelargestimprovementsoccurforquestionsthatrequireselectingtherelevantoption,where
CoPincreasesaccuracybyapproximately20to30percentagepointsinbothdirections,andforS→FMquestionswith
negativepolarity.FM→Squestionswithnegativepolarityarealreadyansweredcomparativelywellbydirectprompting
and show a smaller gain. With Qwen 2.5 7B, CoP improves accuracy for all four question types. With Llama 3.1 8B,
CoP improves accuracy for the two positive-polarity types and for S→FM questions with negative polarity, but not
for FM→S questions with negative polarity, where direct prompting remains slightly better. S→FM questions with
negative polarity are the most difficult type for this model across all approaches. Overall, the improvement obtained
withCoPisnotlimitedtoasinglequestionformulation,althoughitsmagnitudedependsonboththequestiontypeand
the model.
5.7. Reasoning structure without the CKG
The previous experiments evaluate CoP with both the constrained failure-to-observation reasoning structure and
theasset-specificknowledgestoredintheCKG.Toexaminethecontributionofthereasoningstructureindependently
ofgraphretrieval,wealsoevaluateavariantwithoutCKGsupportusingthemulti-agentarchitectureproposedbyGong
etal.[16].ThisstudywasselectedbecauseitisoneofthefewapproachesevaluatedonFailureSensorIQforwhichthe
agent roles, routing logic, and overall decision procedure are described in sufficient detail to support an independent
reimplementation.
The architecture described by Gong et al. comprises a Guide Agent, a Discriminator Agent, a Simple Answerer
Agent, one Option Advocate Agent for each answer choice, and a Judge Agent. The Guide produces a reasoning
blueprint, while the Discriminator classifies each question asSIMPLEorCOMPLEX. SIMPLE questions are routed
:Preprint submitted to Elsevier Page 18 of 29

Table 7
Accuracy (%) on the 2,667 single-correct FailureSensorIQ questions without graph retrieval. Gong et al. denotes the
reimplemented multi-agent framework of [16] with all agents instantiated using the same model.
Model Method AGT CMP EG EM FAN IGT PT PMP RICE ST Overall
Llama 3.1 8BGong et al.41.9640.45 41.88 44.44 46.0052.5036.9551.9747.92 38.01 43.34
CoP w/o CKG 41.6745.91 45.30 46.58 50.5052.0838.79 51.97 48.81 39.77 45.14
Qwen 2.5 7BGong et al. 41.67 47.73 52.9953.8554.00 55.00 43.38 52.63 56.85 43.27 49.34
CoP w/o CKG45.54 51.82 58.5550.0057.50 57.50 50.74 53.95 62.20 46.78 53.28
Questions 336 220 234 234 200 240 544 152 336 171 2,667
directly tothe Simple Answerer. COMPLEXquestions are processedby the OptionAdvocates, after which theJudge
selects the final answer from the generated arguments.
Since the original study does not provide a public code repository or the complete prompts used for the
individual agents, the framework was reconstructed from the methodological description, Algorithm 1, and the
agent responsibilities reported in the paper. The resulting implementation follows the published agent structure and
SIMPLE/COMPLEX routing procedure.
Withinthisreimplementation,thecomparisonisdesignedtoisolatetheformofthereasoningexchangedbetween
theOptionAdvocatesandtheJudge.IntheGongconfiguration,eachadvocateproducesafree-textargumentsupporting
its assigned answer option. In the CoP w/o CKG configuration, the same advocate is required to express its argument
as a diagnostic chain of the form
failure mode→physical change1→⋯→sensor.
ThechainisenforcedthroughstructuredoutputandthenpassedtotheJudgeinplaceofthefree-textargument.The
remainingarchitectureandroutingprocedurearekeptidenticalwithinourreimplementation.NoCKG,graphretrieval,
orreferencedocumentationisusedineitherconfiguration.Thediagnosticchainsarethereforegeneratedentirelyfrom
the model’s internal knowledge.
Sincethisexperimentdoesnotrequireasset-specificgraphconstruction,itisconductedonalltenassetclassesand
on the complete set of 2,667 single-correct FailureSensorIQ questions. Both configurations are evaluated with Llama
3.18BandQwen2.57B.IntheoriginalGongetal.study,theGuide,Discriminator,andJudgeuseGPT-4o-miniwhile
the evaluated open-source model is used for the remaining roles. In our comparison, all agents are instantiated with
thesamemodel.Thisremovesthecontributionofastrongerorchestrationmodelandallowstheeffectofthereasoning
representation to be assessed under a homogeneous-model setting.
The results are reported in Table 7. For Llama 3.1 8B, replacing free-text advocacy with the constrained failure-
to-observation chain increases overall accuracy from 43.34% to 45.14%. For Qwen 2.5 7B, accuracy increases from
49.34% to 53.28%. Across the individual asset classes, CoP w/o CKG improves performance in 16 of the 20 asset–
model combinations, matches the Gong configuration in one case, and performs lower in three.
The improvement is observed without any contribution from graph-derived knowledge or retrieved engineering
documentation.Thedifferencebetweenthetwoconfigurationsliesintherepresentationoftheintermediatediagnostic
argument presented to the Judge. Constraining this argument to an explicit failure-to-observation chain therefore
improves the final decision even when the chain is generated entirely from the model’s internal knowledge.
This result is relevant for the interpretation of the complete CoP framework. The CKG contributes asset-specific
engineering knowledge and retrieved examples of diagnostic propagation, while the chain representation determines
howtheintermediatereasoningisexpressed.Theexperimentinthissubsectionshowsthatthelatterremainsbeneficial
when the CKG is removed. The gains obtained with the full framework can therefore be associated with two distinct
elements:theengineeringknowledgesuppliedbythegraphandtheexplicitphysicalstructureimposedonthereasoning
process.
:Preprint submitted to Elsevier Page 19 of 29

6. Conclusions and Future Work
LLMsofferapracticalmeansofsupportingknowledge-intensiveengineeringtasks,providedthattheinformation
andreasoningstructurerequiredbythetaskaremadeexplicit.ThisstudyintroducedCoPforindustrialfaultdiagnosis,
translating the way an engineer traces a failure through its physical consequences into a constrained failure-to-
observation representation. Asset-specific reference documentation is converted offline into a CKG, and complete
graph paths are retrieved during inference as engineering knowledge and reasoning examples for the generation of
failure-specific diagnostic chains. CoP addresses a complementary diagnostic problem to conventional data-driven
fault classification: reasoning over engineering knowledge to establish the relationships between failure mechanisms,
their physical effects, and the sensing modalities through which those effects can be observed.
On1,314single-correctFailureSensorIQquestions,CoPimprovedaccuracyacrossallthreeevaluatedLLMs,with
gainsof7.8–19.6percentagepointsoverdirectpromptingand11.7–18.2pointsoverconventionalRAG.Theablation
analysis showed that the CKG provides a substantial part of this improvement, while the additional inference stages
provide further gains. The comparison without graph retrieval provides a second result: imposing the same failure-
to-observation structure within the reproducible framework of Gong et al. improves overall accuracy for both tested
models. This supports the use of physically ordered diagnostic chains as a reasoning representation independently of
the asset-specific knowledge supplied by the CKG.
Future work will extend the evaluation to real industrial datasets combining sensor measurements, maintenance
records, and naturally occurring diagnostic requests. Further work should also test alternative reasoning architectures
andrichergraphconstraints,includingcomponenttopology,operating-statedependencies,andadmissiblepropagation
paths,anddevelopconfidenceorabstentionmechanismsforcasesinwhichtheavailableengineeringknowledgedoes
not support a sufficiently reliable diagnostic conclusion.
CRediT Authorship Contribution Statement
Wenqi Zhou:Conceptualization, Methodology, Software, Validation, Investigation, Visualization, Writing -
Original Draft, Writing - Review & EditingLeonardo Baldo:Methodology, Visualization, Writing - Original Draft,
Writing-Review&EditingKaiGoebel:Conceptualization,Methodology,Writing-Review&EditingManuelArias
Chao:Conceptualization, Methodology, Validation, Writing - Review & Editing, Supervision, Funding acquisition
Declaration of Competing Interest
Theauthorsdeclarethattheyhavenoknowncompetingfinancialinterestsorpersonalrelationshipsthatcouldhave
appeared to influence the work reported in this paper.
Code and Data Availability Statement
Datasharingisnotapplicabletothisarticleasnonewdatawerecreatedorgenerated.TheFailureSensorIQdataset
employed in this study is publicly available [43]. The code will be made available upon request.
Declaration of generative AI and AI-assisted technologies in the manuscript preparation
process
During the preparation of this work, the authors used ChatGPT and Claude to assist with language editing, image
editing, improving readability, and refining the presentation of the manuscript. After using these tools, the authors
reviewed and edited the content as needed and take full responsibility for the content of the published article.
References
[1] International Organization for Standardization. ISO 17359:2018 – condition monitoring and diagnostics of machines – general guidelines.
Technical Report ISO 17359:2018, International Organization for Standardization, 2018.
[2] AndrewK.S.Jardine,DamingLin,andDraganBanjevic. Areviewonmachinerydiagnosticsandprognosticsimplementingcondition-based
maintenance.Mechanical Systems and Signal Processing, 20(7):1483–1510, 2006.
[3] RuiKang,JunxiangZhu,StephenGreen,andIoannisBrilakis. Knowledgegraphsforoperationaldecision-makinginindustrialmaintenance:
A systematic review.Expert Systems with Applications, 299:130172, 2026.
:Preprint submitted to Elsevier Page 20 of 29

[4] Yanying Wang, Ying Cheng, Qinglin Qi, and Fei Tao. IDS-KG: An industrial dataspace-based knowledge graph construction approach for
smart maintenance.Journal of Industrial Information Integration, 38:100566, 2024.
[5] WanBarmawiMuhammadRazaq,HaoChen,MarcosR.Machado,andJoãoLuizRebeloMoreira.Howcantheintegrationofailargelanguage
models and knowledge graph enhance fault diagnosis? a systematic literature review.Applied Soft Computing, 194:114908, 2026.
[6] Huihui Han, Jian Wang, and Xiaowen Wang. Enhancing reliability of steel production lines through advanced knowledge graph-based fault
diagnosis model.IEEE Transactions on Reliability, 74(2):2880–2892, 2025.
[7] Yunfei Ma, Shuai Zheng, Zheng Yang, Hongcheng Pan, and Jun Hong. A knowledge-graph enhanced large language model-based fault
diagnostic reasoning and maintenance decision support pipeline towards industry 5.0.International Journal of Production Research,
64(12):5239–5260, 2026.
[8] Sarah Lukens, Matthew Bishof, Nadir Siddiqui, and Destiny West. An evaluation framework for fault diagnosis using technical manuals in
retrieval-augmented large language models. InAnnual Conference of the PHM Society, volume 17, 2025.
[9] Johan de Kleer and Brian C. Williams. Diagnosing multiple faults.Artificial Intelligence, 32(1):97–130, 1987.
[10] MarkA.KramerandJr.Palowitch,B.L.Arule-basedapproachtofaultdiagnosisusingthesigneddirectedgraph.AIChEJournal,33(7):1067–
1078, 1987.
[11] International Organization for Standardization. Condition monitoring and diagnostics of machines—data interpretation and diagnostics
techniques—part 1: General guidelines, 2012.
[12] NourhanIbrahim,SamarAboulela,AhmedIbrahim,andRashaKashef. Asurveyonaugmentingknowledgegraphs(KGs)withlargelanguage
models (LLMs): models, evaluation metrics, benchmarks, and challenges.Discover Artificial Intelligence, 4(1):76, 2024.
[13] Jorge Martinez-Gil, Georg Buchgeher, David Gabauer, Bernhard Freudenthaler, Dominik Filipiak, and Anna Fensel. Root cause analysis in
the industrial domain using knowledge graphs: A case study on power transformers.Procedia Computer Science, 200:944–953, 2022. 3rd
International Conference on Industry 4.0 and Smart Manufacturing.
[14] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A
roadmap.IEEE Transactions on Knowledge and Data Engineering, 36(7):3580–3599, 2024.
[15] JasonWei,XuezhiWang,DaleSchuurmans,MaartenBosma,BrianIchter,FeiXia,EdChi,QuocV.Le,andDennyZhou. Chain-of-thought
promptingelicitsreasoninginlargelanguagemodels. InAdvancesinNeuralInformationProcessingSystems,volume35,pages24824–24837.
Curran Associates, Inc., 2022.
[16] Wei Gong, Shuang Qiao, Chenhong Cao, Shilei Tan, Junliang Ye, Haoxiang Liu, Si Chen, and Xuesong Wang. Harnessing collective
intelligence of multi-agent LLM systems for sensor failure reasoning in smart manufacturing.Journal of Industrial Information Integration,
49:101012, 2026.
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau
Yih,TimRocktäschel,SebastianRiedel,andDouweKiela. Retrieval-augmentedgenerationforknowledge-intensiveNLPtasks. InAdvances
in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.
[18] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. StructGPT: A general framework for large language
modeltoreasonoverstructureddata.InHoudaBouamor,JuanPino,andKalikaBali,editors,Proceedingsofthe2023ConferenceonEmpirical
Methods in Natural Language Processing, pages 9237–9251, Singapore, December 2023. Association for Computational Linguistics.
[19] Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. Knowledge graph prompting for multi-document question
answering.Proceedings of the AAAI Conference on Artificial Intelligence, 38(17):19206–19214, 2024.
[20] Linhao Luo, Yuan-Fang Li, Reza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning.
InInternational Conference on Learning Representations, 2024.
[21] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. Think-on-graph 2.0: Deep and
faithful large language model reasoning with knowledge-guided retrieval augmented generation. InInternational Conference on Learning
Representations, 2025.
[22] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. InProceedings of the 2022
Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655–
2671, Seattle, United States, 2022. Association for Computational Linguistics.
[23] JiachangLiu,DinghanShen,YizheZhang,BillDolan,LawrenceCarin,andWeizhuChen. Whatmakesgoodin-contextexamplesforGPT-3?
InProceedingsofDeepLearningInsideOut(DeeLIO2022):The3rdWorkshoponKnowledgeExtractionandIntegrationforDeepLearning
Architectures, pages 100–114, Dublin, Ireland and Online, 2022. Association for Computational Linguistics.
[24] YanbinWei,QiushiHuang,YuZhang,andJamesT.Kwok. KICGPT:Largelanguagemodelwithknowledgeincontextforknowledgegraph
completion. InFindingsoftheAssociationforComputationalLinguistics:EMNLP2023,pages8667–8683,Singapore,2023.Associationfor
Computational Linguistics.
[25] WeikunLinandKehuaMiao. Anautomotivefaultdiagnosisframeworkbasedonknowledgegraphsandlargelanguagemodels.Electronics,
14(21), 2025.
[26] Qingwei Nie, Junsai Geng, Dunbing Tang, and Changchun Liu. Industrial knowledge-enhanced fault diagnosis method: Integrating llm
and knowledge graph for fault reasoning and maintenance recommendation in cnc machine tools.Computers & Industrial Engineering,
214:111879, 2026.
[27] Yuhan Liu, Yuan Zhou, Yufei Liu, Zhen Xu, and Yixin He. Intelligent fault diagnosis for CNC through the integration of large language
models and domain knowledge graphs.Engineering, 53:311–322, 2025.
[28] YuGu,JianZhang,andYugenDu. Causalgraph-enhancedlargelanguagemodelsforautomatedfaultdiagnosisandintelligentoperationand
maintenance in distributed computing systems.Electronics, 15(11):2359, 2026.
[29] XinXie,JunboWang,YuHan,andWenjuanLi. Knowledgegraph-basedin-contextlearningforadvancedfaultdiagnosisinsensornetworks.
Sensors, 24(24), 2024.
:Preprint submitted to Elsevier Page 21 of 29

[30] Runsheng Miao, Tao Wu, and Zhenyu Zhang. Graph rag-based fault diagnosis for train bogies using knowledge graphs and large language
model.Knowledge-Based Systems, 331:114855, 2026.
[31] Zhen Guo, Zhiping Liu, Edgar Estupiñan, and Diego Cabrera. Fault inference diagnosis based on causal network and large language model-
enhanced knowledge graph network.Chinese Journal of Mechanical Engineering, 39:100162, 2026.
[32] Bin Zhou, Xingwang Shen, Runze Miao, and Jinsong Bao. Llm-driven knowledge graph construction for automotive parts fault diagnosis.
Digital Engineering, page 100137, 2026.
[33] Duyun Wang, Peilin Han, Shmuel Tyszberowicz, Mingyue Zhang, and Bo Liu. GraphRAG-KM: An automated framework for transforming
industrial documents into ontology and conceptual models. InKnowledge Science, Engineering and Management, volume 15920 ofLecture
Notes in Computer Science, pages 108–120. Springer Nature Singapore, 2026. KSEM 2025, Proceedings, Part II.
[34] SamanMarandi,Yu-ShuHu,andMohammadModarres. Complexsystemdiagnosticsusingaknowledgegraph-informedandlargelanguage
model-enhanced framework.Applied Sciences, 15(17):9428, 2025.
[35] Yan-Fu Li, Huan Wang, and Muxia Sun. ChatGPT-like large-scale foundation models for prognostics and health management: A survey and
roadmaps.Reliability Engineering & System Safety, 243:109850, 2024.
[36] Xuefeng Chen, Yaguo Lei, Yanfu Li, Simon Parkinson, Xiang Li, Jinxin Liu, Fan Lu, Huan Wang, Zisheng Wang, Bin Yang, Shilong Ye,
and Zhibin Zhao. Large models for machine monitoring and fault diagnostics: Opportunities, challenges, and future direction.Journal of
Dynamics, Monitoring and Diagnostics, 4(2):76–90, 2025.
[37] ChristosTsallis,PanagiotisPapageorgas,RaduAdrianMunteanu,andSofieneDellagi. Largelanguageandfoundationmodelsformachinery
health monitoring: A systematic review.Applied Sciences, 16(5):2493, 2026.
[38] MinhyeokCha,Sang-ilYoon,SeongraeKim,DaeyoungKang,KeonwooNam,TeakyongLee,andJoon-YoungKim. Largelanguagemodel-
based autonomous agent for prognostics and health management.Machines, 13(9):831, 2025.
[39] LaifaTao,ShangyuLi,HaifeiLiu,QixuanHuang,LiangMa,GuoaoNing,Yi-LingChen,YunlongWu,BinLi,WeiweiZhang,ZhengduoZhao,
Wenchao Zhan, Wenyan Cao, Chao Wang, Hongmei Liu, Jian Ma, Mingliang Suo, Yujie Cheng, Yu Ding, Dengwei Song, and Lü Chen. An
outlineofprognosticsandhealthmanagementlargemodel:Concepts,paradigms,andchallenges.MechanicalSystemsandSignalProcessing,
232:112683, 2025.
[40] Peifeng Liu, Lu Qian, Xingwei Zhao, and Bo Tao. Joint knowledge graph and large language model for fault diagnosis and its application in
aviation assembly.IEEE Transactions on Industrial Informatics, 20(6):8160–8169, 2024.
[41] Xingming Liao, Chong Chen, Zhuowei Wang, Ying Liu, Tao Wang, and Lianglun Cheng. Large language model assisted fine-grained
knowledge graph construction for robotic fault diagnosis.Advanced Engineering Informatics, 65:103134, 2025.
[42] Lunyong Li, Auwal Haruna, Wanming Ying, Khandaker Noman, and Yongbo Li. Knowledge graph-driven fault diagnosis for aviation
equipment:Integratingimprovedjointextractionwithlargelanguagemodel.JournalofIndustrialInformationIntegration,50:101039,2026.
[43] ChristodoulosConstantinides,DhavalC.Patel,ShuxinLin,ClaudioGuerrero,SunilDagajiraoPatil,andJayantKalagnanam.FailureSensorIQ:
Amulti-choiceQAdatasetforunderstandingsensorrelationshipsandfailuremodes. InAdvancesinNeuralInformationProcessingSystems,
volume 38, 2025. Datasets and Benchmarks Track.
[44] Yun Gao, Yuhua Qin, Tao Wang, Yue Cui, and Pengxiang Wang. Knowledge graph enhanced large language model framework for causal
chain reasoning in industrial fault diagnosis.Expert Systems with Applications, 318:132027, 2026.
[45] YiqingSong,WenfaLi,GuirenDai,andXinnaShang. Advancementsincomplexknowledgegraphquestionanswering:Asurvey.Electronics,
12(21):4395, 2023.
[46] C. Peng, F. Xia, M. Naseriparsa, and F. Osborne. Knowledge graphs: Opportunities and challenges.Artificial Intelligence Review,
56:13071–13102, 2023.
[47] SanderSchulhoff,MichaelIlie,NishantBalepur,KonstantineKahadze,AmandaLiu,ChengleiSi,YinhengLi,AayushGupta,HyoJungHan,
Sevien Schulhoff, Pranav Sandeep Dulepet, Saurav Vidyadhara, Dayeon Ki, Sweta Agrawal, Chau Pham, Gerson Kroiz, Feileen Li, Hudson
Tao,AshaySrivastava,HevanderDaCosta,SaloniGupta,MeganL.Rogers,InnaGoncearenco,GiuseppeSarli,IgorGalynker,DenisPeskoff,
Marine Carpuat, Jules White, Shyamal Anadkat, Alexander Hoyle, and Philip Resnik. The prompt report: A systematic survey of prompt
engineering techniques, 2025.
[48] Sondos Mahmoud Bsharat, Aidar Myrzakhan, and Zhiqiang Shen. Principled instructions are all you need for questioning LLaMA-1/2,
GPT-3.5/4, 2024.
[49] Deep Search Team. Docling technical report. Technical report, 8 2024.
[50] Microsoft. Chunk large documents for RAG and vector search in Azure AI Search.https://learn.microsoft.com/en-us/azure/
search/vector-search-how-to-chunk-documents. Accessed: 2026-10-09.
[51] Sinchana Ramakanth Bhat, Max Rudat, Jannis Spiekermann, and Nicolas Flores-Herr. Rethinking chunk size for long-document retrieval: A
multi-dataset analysis.arXiv preprint arXiv:2505.21700, 2025.
[52] Belinda Mo, Kyssen Yu, Joshua Kazdan, Joan Cabezas, Proud Mpala, Lisa Yu, Chris Cundy, Charilaos Kanatsoulis, and Sanmi Koyejo.
KGGen: Extracting knowledge graphs from plain text with language models. InAdvances in Neural Information Processing Systems,
volume 38, 2025.
[53] H. I. H. Saravanamuttoo, G. F. C. Rogers, Henry Cohen, Paul Straznicky, and Andrew Nix.Gas Turbine Theory. Pearson, 7 edition, 2017.
[54] MuhammadAmanSheikh,NursyarizalMohdNor,TaibIbrahim,SheikhTahirBakhsh,MuhammadIrfan,andHanitaBintiDaud.Noninvasive
methodsforconditionmonitoringandelectricalfaultdiagnosisofinductionmotors. InMustafaDemetgulandMuhammetÜnal,editors,Fault
Diagnosis and Detection. IntechOpen, 2017.
[55] U.S. Department of Energy and Air Movement and Control Association International. Improving fan system performance: A sourcebook for
industry. Technical Report DOE/GO-102003-1294, U.S. Department of Energy, 2003.
[56] U.S. Department of the Interior, Bureau of Reclamation. Transformers: Basics, maintenance, and diagnostics. Technical report, Bureau of
Reclamation, Denver, Colorado, April 2005.
:Preprint submitted to Elsevier Page 22 of 29

A. Prompts
The online CoP pipeline uses four LLM calls: clarification, failure-mode extraction, failure-mode reasoning, and
finaljudging.Eachmoduleusesadedicatedsysteminstructionandastructureduserprompt.Thetemplatesarereported
below. Terms enclosed in angle brackets indicate values populated at inference time.
A.1. Offline CKG Construction
The offline CKG construction uses one LLM call for each text unit extracted from the reference documentation.
Themodelisaskedtoidentifyonecompletefailure-to-observationchaingroundedinthecurrenttextunit.Theoutput
is constrained to a structured schema containing one failure mode, one to five changes in physical quantities, and one
monitoring sensor.
Diagnostic-Chain Extraction System Prompt
You extract ONE failure-mode→sensor causal chain for asset:<Asset>.
Describe ONE coherent monitoring scenario on a<Asset>as a chain:
failure_mode→variable_1 changes→variable_2 changes→...→sensor
Fields:
∙failure_mode:namedfault/degradation/abnormalconditionofthe<Asset>(oranamedsubcomponent).
∙steps: ordered list of 1–5 physical-variable changes. Each step contains:
–variable: the physical quantity that moves;
–location: where on the<Asset>it is taken; empty if the CHUNK does not say;
–direction: “increases” or “decreases”.
∙steps[0]is what the failure mode drives directly; each later step is caused by the one before it; the LAST
step is what the sensor reads.
∙sensor: instrument that reads the LAST step’s variable.
IftheCHUNKdoesnotgroundthefailure_mode,atleastonestep,ANDthesensorfortheSAMEscenario—
orisonlygeneraltheory/history/equations/designwithnofailure-to-monitoringlink—leavefailure_mode
andsensorempty andsteps=[].
Rules:
∙Use ONLY facts grounded in the CHUNK. Do NOT add physics hops the CHUNK does not state, and do
NOT pad the chain to make it longer. A 1-step chain is correct when that is all the CHUNK supports.
∙Everystepmustbeaphysicalvariablethatgoesupordown—notafault,notaninstrument,notamaintenance
action.
∙Do not repeat the same variable+location+direction twice in the chain.
∙Prefer the dominant failure scenario when several appear.
∙Stay specific to<Asset>; do not import failure modes from other asset classes.
∙Keep each field a short phrase; no markdown.
Diagnostic-Chain Extraction User Prompt Template
CHUNK:
<Reference Text Unit>
The structured output contains the fieldsfailure_mode,steps, andsensor. Each element ofstepscontains
avariable, an optionallocation, and adirectionrestricted toincreasesordecreases. Empty outputs are
interpreted as rejected text units and do not contribute to the CKG.
:Preprint submitted to Elsevier Page 23 of 29

A.2. Clarification
The clarification module checks whether the original multiple-choice question or its options require rewording
before the diagnostic process starts. The answer choices are preserved in number, order, and meaning.
Clarification System Prompt
You clarify failure-mode / sensor multiple-choice questions before they are answered.
Rules:
∙First decideneeds_clarification.
∙If already clear: setneeds_clarification=falseand leaveclarified_questionempty and
clarified_optionsas[]. Do not rewrite anything.
∙If clarification is needed: setneeds_clarification=true, rewriteclarified_question, and return
clarified_optionswith the same number/order of options.
∙Preserve the original meaning, asset, failure event, and which sensor each option refers to.
∙Do not add, drop, merge, or reorder options when rewriting.
∙Do not pick an answer. Do not invent new failure modes or sensors not implied by the text.
∙Make vague wording concrete without changing substance.
Clarification User Prompt Template
Question:<Question>
Options:
<Option ID 1> <Option 1>
<Option ID 2> <Option 2>
...
There are exactly<N>options with ids<Option IDs>.
Ifneeds_clarification=true,returnclarified_optionswiththesamelengthandorder;iffalse,leave
clarified_questionempty andclarified_options=[].
The structured output containsneeds_clarification,clarified_question,clarified_options, and a
short note describing the ambiguity when present.
A.3. Failure-Mode Extraction
Failure modes are extracted from the clarified question and answer options. This step is independent of the CKG.
Failure-Mode Extraction System Prompt
You are a failure analysis expert. Extract all failure modes mentioned or implied in the given question and
answer options.
A failure mode is a way in which a component, system, or process can fail to perform its intended function.
Look for:
∙Direct mentions of failure modes, e.g. “bearing wear” or “shaft misalignment”.
∙Implied failures described by symptoms or effects.
∙Failure conditions mentioned in context.
Extract the failure mode names as they naturally appear in the text. Be precise and avoid over-interpretation.
:Preprint submitted to Elsevier Page 24 of 29

Failure-Mode Extraction User Prompt Template
Question:<Clarified Question>
Options:
<Option ID 1> <Option 1>
<Option ID 2> <Option 2>
...
Extract all failure modes mentioned or implied in the above text.
A failure mode is a way a component or system can fail.
Look for direct mentions, descriptions of failures, or failure conditions.
The output is a list of extracted failure modes. Each failure mode is subsequently analysed independently.
A.4. Failure-Mode Reasoning
Foreachextractedfailuremode,themodelreceivestheretrievedCKGchainsandthecorrespondingchainstructure.
The retrieved paths provide examples of failure-to-sensor reasoning and are used to guide the generation of plausible
diagnostic chains for the target failure mode.
Failure-Mode Reasoning System Prompt
You are a failure analysis expert. Given a specific failure mode and knowledge graph examples, identify what
diagnostic chains this failure mode could produce.
Use the knowledge graph examples as patterns to understand how failure modes typically lead to signatures
and then to sensors.
Do NOT try to answer any specific question. Analyze the failure mode itself and list the possible diagnostic
chains it could lead to.
Failure-Mode Reasoning User Prompt Template
Asset:<Asset>
Failure mode to analyze:<Failure Mode>
Knowledge graph examples showing diagnostic chain patterns for<Asset>:
<Retrieved CKG Chains>
Chain pattern:
<Failure Mode> – results in –> <Signature> – leads to –> <Signature> – is monitored
by –> <Sensor>
Basedontheseexamplesandyourdomainknowledgeabout<Asset>systems,analyzewhatdiagnosticchains
the failure mode<Failure Mode>could produce.
What signatures might this failure mode cause?
What sensors could detect those signatures?
Do not reference any specific question or try to pick an answer. Analyze the failure mode itself in the context
of<Asset>systems.
The structured output contains the list of possible diagnostic chains inferred for the target failure mode.
A.5. Judge
TheJudgereceivestheclarifiedquestion,theanswerchoices,thetaskpolarity,andthereasoninggeneratedforall
extracted failure modes. It compares the inferred chains with the available choices and returns one option.
:Preprint submitted to Elsevier Page 25 of 29

Judge System Prompt
You are the Judge Agent for failure-mode / sensor multiple-choice questions.
You will receive independent reasoning about failure modes extracted from the question/options, along with
the original question and options.
Use the failure mode reasonings to understand what diagnostic chains are possible, then match those to the
options to select the best answer.
Follow the task polarity exactly: for MOST relevant select the strongest match; for LEAST relevant select the
weakest or most contradicted match.
Judge User Prompt Template
Question:<Clarified Question>
Options:
<Option ID 1> <Option 1>
<Option ID 2> <Option 2>
...
<Task Polarity Instruction>
Independent failure mode analysis:
Failure Mode:<Failure Mode 1>
Possible chains:<Possible Chains 1>
...
Based on the failure mode reasonings above, determine which option best matches the diagnostic chains.
Select exactly one option from<Candidate Option IDs>.
For positive questions, the task instruction asks for the strongest and most direct diagnostic relationship. For
negative questions, it asks for the weakest, most indirect, or most contradicted relationship. The Judge returns a
comparative reasoning field and one answer identifier.
B. Worked Example of Question Clarification
The ablation study in Table 5 shows that the clarification stage increases accuracy for all three models. The trace
in Figure 5 illustrates the case in which the question is already clear and is passed through unchanged. This appendix
reports the complementary case, in which the clarification module identifies an ambiguity, rewrites the question, and
changes the final answer.
The example is taken from the fan subset of FailureSensorIQ and was obtained with Claude Opus 4.7.
B.1. Original question
Input Question (Fan, Sensor2FailureMode, negative)
Stem.Which failure event is irrelevant for fan if there is an abnormal reading from the sensor length
measurement?
:Preprint submitted to Elsevier Page 26 of 29

Adamaged oil seals
Bbearing wear
Cbearing damage
D rotor fouled Gold
Emisalignment
The question contains two sources of ambiguity. First, the sensor is described as alength measurement, which
does not specify the physical quantity being measured. Second, the negative formulation (irrelevant ... if there is
an abnormal reading) can be read either as “which failure is unimportant in general” or as “which failure cannot be
indicated by this sensor”. Only the second reading corresponds to the diagnostic intent of the item.
B.2. Clarification output
Clarification Output
needs_clarification:true
Reasoning.“Sensorlengthmeasurement”isvague;clarifiedasadisplacement/positionmeasurementsensor.
Also clarified that the question asks which failure mode is NOT relevant.
Clarified question.For a fan, if a displacement/position sensor (length measurement, e.g., axial or radial
displacement)showsanabnormalreading,whichofthefollowingfailureeventsisNOTrelevantto(i.e.,cannot
be indicated by) this sensor?
Clarified options.
ADamaged oil seals
BBearing wear
CBearing damage
DFouled rotor
EShaft misalignment
Therewriteactsonbothambiguities.Thesensorisgroundedasadisplacement/positionprobemeasuringaxialor
radialdisplacement,andthenegativepolarityisrestatedexplicitlyas“cannotbeindicatedbythissensor”.Theanswer
options are only lightly normalised: their number, order, and meaning are preserved, and no answer is suggested by
the rewrite.
C. Full Inference Trace
This appendix reports the complete output of every stage of the online CoP pipeline for a single FailureSensorIQ
item. The item belongs to the aero gas turbine subset and was processed by the full CoP configuration with Claude
Opus 4.7. The prompt templates used at each step are given in Appendix A.
1Question
Stem.For aero gas turbine, if a failure event air inlet blockage occurs, which sensor out of the choices is the
most relevant sensor regarding the occurrence of the failure event?
:Preprint submitted to Elsevier Page 27 of 29

Avibration
Boil leakage/ consumption
C compresor pressure/ pressure ratio Gold
Dfuel pressure/ fuel flow
Epower turbine temperature
2ClarifyIs the question ambiguous or unclear?
needs_clarification:false clarified_options:[]
The question is deemed unambiguous, so no changes are made. The original stem and options are passed
unchanged to steps 3a and 3b.
3aRetrieve & expandWhat reasoning examples exist in the CKG?
Retrieval seeds.The question and options are embedded and compared with the failure-mode and sensor
entities of the aero gas turbine CKG. The top-2 seeds are:
Rank Seed entity Cosine similarity
1 Turbine inlet temperature sensor 0.6941
2 turbine inlet temperature sensor 0.6923
Expanded CKG chains.Backward expansion from the two sensor seeds yields nine complete chains. Each
chainiswrittenasfailuremode→physicalchange(s)→sensor,wherethearrowscorrespond,inorder,tothe
CKG relationsresults in,leads to, andis monitored by:
1. Variable stator overheating→Stator temperature at variable stators increases→Turbine efficiency at power turbine decreases→
Turbine inlet temperature at turbine inlet increases→Turbine inlet temperature sensor
2. High turbine inlet temperature causes excess NOx→Turbine inlet temperature at turbine inlet increases→Turbine inlet
temperature sensor
3. build-up of deposits on compressor blading→compressor isentropic efficiency at compressor decreases→turbine operating
temperature at turbine increases→turbine inlet temperature sensor
4. excessive accessory power demand→fuel flow at combustor increases→turbine inlet temperature at turbine inlet increases→
turbine inlet temperature sensor
5. turbine inlet temperature exceeds safe limit→turbine inlet temperature at turbine inlet increases→turbine inlet temperature
sensor
6. nozzle area reduction→nozzle area at propelling nozzle decreases→turbine inlet temperature at turbine inlet increases→
turbine inlet temperature sensor
7. Turbine inlet temperature exceeds limit→turbine inlet temperature at turbine inlet increases→turbine inlet temperature
sensor
8. turbine overheating→turbine inlet temperature increases→turbine inlet temperature sensor
9. flame-out during rapid climb→fuel flow rate at fuel metering unit decreases→combustor temperature at combustor chamber
decreases→turbine inlet temperature at turbine inlet decreases→turbine inlet temperature sensor
:Preprint submitted to Elsevier Page 28 of 29

3bExtract failure modeWhat failure mode must be analysed?
Output.{"failure_modes": ["air inlet blockage"]}
One failure mode is extracted from the question and options. This step does not access the CKG.
4Reason about the failure modeWhat physical chains could this failure mode produce?
Input.Asset (aero gas turbine), target failure mode (air inlet blockage), the nine CKG chains from step 3a as
in-context examples, and the chain pattern. The question and options are not provided.
Output.One LLM call produces ten candidate chains, none of which is copied from step 3a:
1. air inlet blockage→inlet air mass flow at compressor inlet decreases→compressor pressure ratio at compressor decreases→
compressor discharge pressure sensor
2. air inlet blockage→inlet air pressure at compressor inlet decreases→inlet pressure sensor
3. air inlet blockage→inlet air mass flow at compressor inlet decreases→air-fuel ratio at combustor decreases→turbine inlet
temperature at turbine inlet increases→turbine inlet temperature sensor
4. airinletblockage→compressorinletpressureatcompressorinletdecreases→compressorsurgemarginatcompressordecreases
→compressor pressure sensor
5. air inlet blockage→inlet air mass flow at compressor inlet decreases→engine thrust at nozzle decreases→thrust sensor
6. air inlet blockage→pressure drop across inlet at air intake increases→differential pressure sensor
7. air inlet blockage→compressor rotational speed at compressor decreases→shaft speed sensor (N1/N2)
8. air inlet blockage→inlet air mass flow at compressor inlet decreases→exhaust gas temperature at turbine exhaust increases→
exhaust gas temperature sensor
9. air inlet blockage→compressor inlet temperature at compressor inlet changes→inlet temperature sensor
10. air inlet blockage→inlet air mass flow at compressor inlet decreases→compressor stall margin at compressor decreases→
vibration at compressor increases→vibration sensor
5JudgeSees question, options and all inferred chains
Input.Original question and options, the task polarity instruction (MOST relevant: select the option with the
strongest, most direct diagnostic relationship), and the ten chains from step 4.
Judgereasoning.“Airinletblockagemostdirectlyreducesinletairmassflowandpressure,whichimmediately
manifests as a drop in compressor pressure/pressure ratio. Multiple diagnostic chains point to compressor
pressureastheprimarymonitoredparameter.EGT/turbinetempchangesandvibrationaresecondaryeffects,
while oil and fuel are unrelated.”
Answer
C — compresor pressure/ pressure ratio Correct
The trace shows the same behaviour as the fan example in Figure 5. The retrieved CKG chains concern different
failuremechanismsandpointtoasensorthatisnotthecorrectanswer.Theyneverthelessprovidethefailure–physical
change–sensorstructurethatthereasoningstepappliestoairinletblockage.Thecorrectanswerfollowsfromthenewly
inferred chains.
:Preprint submitted to Elsevier Page 29 of 29