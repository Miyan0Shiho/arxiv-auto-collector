# CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs

**Authors**: Muhammad Roman, Karen Rafferty, Barry Devereux

**Published**: 2026-08-03 16:40:16

**PDF URL**: [https://arxiv.org/pdf/2608.02472v1](https://arxiv.org/pdf/2608.02472v1)

## Abstract
Trust is fundamental in modern regulatory ecosystems, and compliance checking plays a critical role in fostering that trust. Regulatory compliance verification is essential for businesses operating in highly controlled environments, as it ensures alignment with sector-specific guidelines across domains such as financial reporting, data privacy, and cybersecurity. Manual compliance testing, however, is often time-intensive and prone to inconsistencies, particularly when compliance depends indirectly on third-party services such as cloud providers, where vendors rely on external providers to meet regulatory standards. In this paper, we present CTRAG, a novel Retrieval-Augmented Generation (RAG) pipeline designed for automated compliance checking. CTRAG employs advanced strategies, including adaptive chunking, dynamic retrieval configurations, and in-context learning, to improve the precision and relevance of compliance assessments. By extracting control questions from regulatory texts and cross-referencing them with unstructured company documentation, CTRAG achieves highly accurate, document-informed compliance verification, even in cases of indirect compliance through third-party services. Empirical evaluations demonstrate significant improvements, with CTRAG achieving an F1-score of 78% and a recall of 85% in the final deployed configuration, ensuring minimal missed non-compliance cases while reducing manual reviewer effort in a real-world deployment. To validate CTRAG value, we developed and deployed a POC within a Big Four professional services firm, applying it to real-world cases and cross-checking results against manual compliance reports. These findings highlight CTRAG potential to streamline compliance workflows, mitigate risks, and enhance regulatory trust in complex, high-stakes environments.

## Full Text


<!-- PDF content starts -->

Research Article
CTRAG: An In-Context Retrieval-based Framework for Automated
Compliance Checking using LLMs
Muhammad Romana, b, Karen Raffertyband Barry Devereuxb
aBristol Research and Innovation Laboratory (BRIL), Toshiba Europe Ltd., Bristol, United Kingdom
bQueen’s University Belfast, United Kingdom
Abstract
Trust is fundamental in modern regulatory ecosystems, and compliance checking plays a critical role in fostering that trust. Regulatory
compliance verification is essential for businesses operating in highly controlled environments, as it ensures alignment with sector-
specific guidelines across domains such as financial reporting, data privacy, and cybersecurity. Manual compliance testing, however,
is often time-intensive and prone to inconsistencies, particularly when compliance depends indirectly on third-party services such
as cloud providers, where vendors rely on external providers to meet regulatory standards. In this paper, we present CTRAG, a novel
Retrieval-Augmented Generation (RAG) pipeline designed for automated compliance checking. CTRAG employs advanced strategies,
including adaptive chunking, dynamic retrieval configurations, and in-context learning, to improve the precision and relevance of
compliance assessments. By extracting control questions from regulatory texts and cross-referencing them with unstructured company
documentation, CTRAG achieves highly accurate, document-informed compliance verification, even in cases of indirect compliance
through third-party services. Empirical evaluations demonstrate significant improvements, with CTRAG achieving an F1-score of 78%
and a recall of 85% in the final deployed configuration, ensuring minimal missed non-compliance cases while reducing manual reviewer
effort in a real-world deployment. To validate CTRAG value, we developed and deployed a POC within a Big Four professional services
firm, applying it to real-world cases and cross-checking results against manual compliance reports. These findings highlight CTRAG
potential to streamline compliance workflows, mitigate risks, and enhance regulatory trust in complex, high-stakes environments.
Keywords:Automated Compliance Checking, Retrieval Augmented Generation (RAG), Open Domain Question Answering, In-context learning
1. INTRODUCTION
Intheincreasinglyregulatedcorporatelandscape,maintainingcom-
pliancewithsector-specificcontrolshasbecomeessentialyetchalleng-
ingfororganizations. Regulatorybodiesrequirecompaniestoadhere
to established controls, which are often complex and multifaceted,
spanningacrossvariousdomainssuchasfinancialreporting,datapri-
vacy,environmentalstandards,andcybersecurity. Compliancewith
theseregulatorycontrolsistypicallyvalidatedthroughextensivedoc-
umentaudits,requiringorganisationstopresentevidencethattheir
operations align with regulatory expectations. In such high-stakes
environments,effectiveandefficientcompliancetestingisessential
tominimiserisksandensureadherencetotheregulations.
Traditionalcompliancetestingofteninvolveslabour-intensiveman-
ualreviews,depictedinFigure1,whereauditorsanalysesubstantial
volumes of documentation provided by companies. This process,
whilethorough,istime-consumingandcanintroducevariabilityin
resultsduetodifferinginterpretationsofregulatorylanguage. Com-
pliancereviewsrequireauditorstolocaterelevantinformationacross
extensivecollectionsofcompanydocuments. Newmaterialsareadded
witheachcase,whichsubstantiallyincreasesboththecomplexityand
theriskofthereview. WithadvancementsinNaturalLanguagePro-
cessing(NLP)andtheemergenceofRetrieval-AugmentedGeneration
(RAG)[1]models,thereisapromisingopportunitytostreamlinecom-
pliancetestingbyleveragingmachinelearning(ML)toautomatethe
retrievalandevaluationofrelevantcomplianceevidence.
WeintroducedaRAG-basedpipelinetailoredspecificallyforregu-
latorycompliancetesting. Usingthisstructuredrepresentation,our
approachsystematicallyverifiesacompany’scompliancebyanalysing
and cross-referencing these controls with documents provided for
audit. Thepipelineincorporatesseverallayersofcustomisationtoop-
timiseresponseaccuracyandrelevance. RAGhelpsbothinretrieving
the right information from the documents, and in using that infor-
mationforregulatorycompliancetests. Webeginbyextractingand
processinginformationfromcompany-providedPDFdocuments,ap-
plyingavarietyofchunkingstrategiesthatmaintainlogicalcoherence
while minimizing extraneous information. We dynamically adjust
chunksizesbasedonthetypeofdocument,leveragingmethodsthatensurebothcontextualintegrityandretrievalaccuracy. Wealsoadjust
thekvalueretrievingthetop-krelevantdocumentsthatcanpotentially
beusedtolookforthecompliancecheck. Thishelpsprovideaccessto
therightinformationthatplaysavitalroleinidentifyingcompliance,
aswitnessedbyexperiments. Finally,weintegratein-contextlearning
(ICL)[2,3],enrichingthepromptwithcuratedexamplesrelevantto
thequeries. Thissignificantlyimprovedresponserelevanceandalign-
mentwithregulatoryexpectations. Thispaperevaluatestheefficacy
ofthisRAG-basedcompliancetestingpipelineanditscontributions
toreducingaudittimewhilemaintainingcomplianceaccuracy. Our
findingsaimtodemonstratehowautomatedcomplianceverification
canstreamlineregulatoryassessmentsandmitigaterisksassociated
withnon-complianceincomplexregulatoryenvironments.
Themaincontributionsofthispaperareasfollows:
1.ThepaperintroducesaRAGpipelinespecificallytailoredforreg-
ulatorycomplianceverification. Thispipelineintegratestheex-
tractionofcontrolsfromregulatorydocumentstoenableprecise,
document-informedchecksagainstclient-providedfilescontain-
ingunstructuredcontents.
2.Thestudybenchmarkschunking,andnumberofchunkstofind
the best fit by extensive experimentation that helps research
communitytounderstandtheimpactofchunksizevariationin
similarproblems. Wehavealsoinvestigatedtheeffectsofvarious
LLMmodelsonthefinalevaluation.
3.It employs several adaptive strategies to enhance retrieval ac-
curacyandresponsequality,includingrobustPDFinformation
extraction,tailoredchunkingtechniques,andcontextextraction
mechanisms.
4.The pipeline uses a hybrid retrieval system to balance global
regulatorycontextwithgranularfactualdetails,dynamicallyset-
tingretrievalK-valuestoensurecontextuallyrelevantdocuments
contributetoresponsegeneration.
5.The approach employs in-context learning to refine responses
byprovidingcarefullyselectedexamplesthatguidetheLLM’s
decision-makingprocess.
arXiv PreprintAugust 4, 2026 arXiv Preprint1–10
arXiv:2608.02472v1  [cs.CL]  3 Aug 2026

CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs arXiv Preprint
Figure 1.Flowofthirdpartycompliancecheckingperformedmanuallybyacompliancetesterandvalidated
2. RELATED WORK
Automatedcompliancechecking(ACC)hasbeenanactiveareaofre-
searchforseveraldecades,drivenbythegrowingcomplexityofregula-
tionsandtheincreasingcostofmanualcomplianceassessment. Early
work focused primarily on formal and rule-based approaches that
soughttorepresentregulations,contracts,andorganisationalpolicies
inmachine-interpretableformats,enablingsystematicverificationof
compliancerequirements. Inthebusinessprocessdomain,compli-
ancecheckinghasbeeninvestigatedasameansofassessingwhether
organisationalprocessesconformtocontractualobligations,legalre-
quirements,andinternalgovernancepolicies[4]. Subsequentresearch
expanded these foundations by examining compliance throughout
thelifecycleofbusinessprocesses,includingdesign-timeverification,
run-timemonitoring,andpost-executionauditing,whilehighlighting
persistentchallengesarisingfromcomplexregulations,evolvingre-
quirements,andthedifficultyofintegratingcompliancemechanisms
intooperationalsystems[5]. Collectively,thesestudiesestablished
manyoftheconceptualandmethodologicalfoundationsthatcontinue
tounderpincontemporarycomplianceautomationresearch.
Beyond business process management, automated compliance
checkinghasbeenextensivelyinvestigatedintheArchitecture,Engi-
neering,Construction,andOperations(AECO)sector,whereregula-
toryverificationistraditionallyperformedthroughlabour-intensive
manual reviews of design artefacts. The emergence of Building In-
formationModelling(BIM)andIndustryFoundationClasses(IFC)
enabled researchers to represent building information in machine-
readable formats, creating opportunities for automating regulatory
assessmentsagainstbuildingcodesandstandards[6]. Researchinthis
areahasfocusedontranslatingtextualregulationsintocomputable
rules,developingsemanticallyrichobjectmodels,andestablishing
interoperability mechanisms capable of supporting regulatory rea-
soningthroughoutthebuildinglifecycle. Morerecentstudieshave
arguedthateffectivecompliancecheckingrequiresabroaderdigital
ecosysteminwhichregulatoryrequirements,designinformation,and
verification mechanisms interact seamlessly across multiple stake-
holdersandsystems[7,8]. Despitesubstantialprogress,challenges
relatedtotheformalisationofregulatoryprovisions,interoperability
acrossheterogeneousdatasources,andtheinterpretationofcomplex
requirementscontinuetolimitthescalabilityofautomatedcompli-
ancesolutions.
Outsidetheconstructiondomain,complianceautomationhasbe-
come increasingly important in cybersecurity, privacy, and cloud-
basedserviceenvironments,whereorganisationsmustdemonstrate
adherence to a growing number of regulatory and industry frame-
works. Recentstudieshavehighlightedtheshiftfromperiodicaudit-
ingtowardscontinuouscomplianceincloud-basedandregulatedenvi-
ronments,wherecompliance-as-codeandpolicy-as-codeapproachesareusedtoautomateevidencecollection,policyvalidation,andtrace-
abilityacrossheterogeneoussystems[9]. Inparallel,researchershave
investigated the specific compliance challenges faced by Software-
as-a-Service(SaaS)providers,whereregulatoryobligationsrelating
to data protection, privacy, and information security must be satis-
fiedwithinhighlydynamicanddistributedenvironments[10]. These
studiesemphasisetheneedforscalableandautomatedapproaches
capableofreducingthecostofcompliancemonitoringwhilemaintain-
ingconsistency,traceability,andalignmentwithevolvingregulatory
requirementsacrossdiverseoperationalcontexts.
Toaddressthelimitationsofpurelyrule-basedcomplianceverifi-
cation,researchersincreasinglyexploredartificialintelligencetech-
niquesforautomatingregulatoryinterpretationandcomplianceas-
sessment. Early efforts employed expert systems and knowledge-
basedreasoningtoencodedomainexpertiseandprovideautomated
compliancerecommendations[11]. Morerecentworkhasrevisited
theroleofartificialintelligenceincompliancechecking,highlighting
boththeopportunitiesandchallengesofapplyingAItechniquestoreg-
ulatoryreasoningandcomplianceassessment[12,13]. Subsequently,
advances in natural language processing and machine learning en-
abledtheextractionofnormativeknowledgefromregulations,stan-
dards,andpolicydocuments,reducingthedependenceonmanually
craftedrulesets[14,15]. Ontology-basedapproachesfurtherenhanced
automatedreasoningbyprovidingformalsemanticrepresentations
ofregulatoryconcepts,relationships,andconstraints,supportingob-
jectmapping,ruleexecution,andknowledgeinteroperabilityacross
compliancedomains[11,16]. Morerecentworkhasintegratedthese
semanticfoundationswithmachinelearning,naturallanguagepro-
cessing,andAItechniquestoimproveregulatoryinterpretationand
automatedcomplianceassessment[17,18].
Despitetheseadvances,automatedcompliancecheckingcontinues
to face significant challenges arising from the inherently complex
andoftenambiguousnatureofregulatoryrequirements. Manycom-
pliance obligations are expressed in natural language and contain
implicit assumptions, contextual dependencies, and subjective ter-
minology that are difficult to formalise computationally. Zhang et
al. [19] demonstrate that a substantial proportion of regulatory re-
quirementscannotbeevaluatedthroughsimplerule-basedmethods
alone, owing to both intentional ambiguity introduced to preserve
flexibilityandunintentionalambiguitystemmingfromlinguisticand
domain-specificcomplexities. Morebroadly,recentstudieshighlight
thattranslatingcomplexregulatorytextsintomachine-interpretable
representationsremainsamajorobstacletolarge-scalecomplianceau-
tomation,particularlywhenevidencemustbegatheredfromdiverse
andunstructuredorganisationaldocuments[20]. Thesechallenges
havestimulatedgrowinginterestinadvancedAItechniquescapable
ofimprovingregulatoryunderstanding,informationextraction,and
2–10

arXiv Preprint CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs
evidence-basedcomplianceassessment.
Theemergenceoflargelanguagemodels(LLMs)hascreatednew
opportunitiesforaddressinglongstandingchallengesinautomated
compliance checking. Unlike traditional rule-based or ontology-
drivensystems,LLMscanleverageadvancedlanguageunderstand-
ingcapabilitiestointerpretregulatoryrequirements,analyseorgan-
isational documents, and perform reasoning over complex textual
evidence. RecentstudieshaveexploredtheuseofLLMsforregula-
toryinterpretation,policyanalysis,compliancemonitoring,andauto-
matedcompliancecheckingacrossmultipledomains[12,13,20,21].
However, concerns regarding hallucinations, evidence attribution,
and explainability remain significant barriers to their adoption in
high-stakes regulatory environments. To address these limitations,
Retrieval-AugmentedGeneration(RAG)combinesexternalinforma-
tionretrievalwithgenerativelanguagemodels,enablingresponsesto
begroundedinrelevantsourcedocumentsratherthanrelyingsolely
onparametricknowledge[1]. Recentresearchhasshownthatretrieval
quality, document chunking strategies, and knowledge integration
mechanismsarecriticaldeterminantsofRAGperformance,particu-
larlyforenterprisedocumentanalysisandotherknowledge-intensive
tasks[22,23]. ThesecharacteristicsmakeRAGparticularlywellsuited
to compliance verification scenarios, where decisions must be sup-
portedbyevidenceextractedfromlargecollectionsoforganisational
documentation.
RecentresearchhasfocusedonimprovingtheeffectivenessofRAG
systemsthroughenhancedretrievalstrategies,documentchunking
methods,andcontextintegrationmechanisms. Studieshaveshown
thatretrievalqualityisoftenaprimarydeterminantofdownstream
generationperformance,motivatingthedevelopmentofspecialised
approachessuchasMeta-Chunking,whichsegmentstextaccording
tologicalstructureratherthanfixedboundaries[24],andLongRAG,
whichemployshybridretrievalmechanismstobalancebroadcontex-
tualunderstandingwithpreciseevidenceextractioninlong-document
settings[25]. Domain-specificadaptationshavealsoemerged,such
as RAG4ITOps, which incorporates specialised retrieval and repre-
sentationtechniquestoimprovequestionansweringoverenterprise
knowledge sources [26]. Similar efforts have also been reported in
regulatoryinformationretrievalandanswergeneration,wherespe-
cialisedretrievalpipelineshavebeendevelopedtosupportevidence-
groundedquestionansweringoverregulatorycorpora[27]. Recent
studies have further demonstrated that retrieval optimisation and
reranking strategies can significantly improve regulatory question-
answering performance in retrieval-augmented systems operating
overregulatorycorpora[28]. Inparallel,researchonin-contextlearn-
inghasdemonstratedthatcarefullyselectedexamplescansubstan-
tiallyimprovethereasoningcapabilitiesofLLMswithoutrequiring
task-specific model fine-tuning [2,3,29]. Despite these advances,
relativelylimitedresearchhasinvestigatedhowretrievalconfigura-
tion,documentchunkingstrategies,andin-contextlearningcanbe
jointly optimised for evidence-based compliance verification over
unstructuredorganisationaldocuments. Thisgapmotivatesthedevel-
opmentofCTRAG,acompliance-orientedRAGframeworkdesigned
to support accurate and traceable compliance assessment through
document-groundedreasoning.
3. PROPOSED SOLUTION
Ourproposedapproachtoautomatedcompliancecheckingengagesa
RAGpipeline,designedtohandlelarge,unstructuredregulatorydoc-
umentsandcompany-providedfiles. Sincecomplianceverification
involves checking a company adherence to various regulatory con-
trols,weframeeachcontrolasaquestionwithinaquestion-answering
setup. This approach aligned with the RAG structure, where each
controlquestiontriggersasearchforthemostrelevantcontentwithin
thecompanydocumentation,aimingtoextractinformationthatac-
curately addresses the compliance requirement. The multi-staged
setup of RAG pipeline, as depicted in Figure 2, allows flexibility in
Figure 2.RAG-basedapproachforcompliancecheckingusingclients
documentsandregulatoryquestions
configuringeachcomponenttohandlethecomplexityandspecificity
ofcompliancetesting. Bycombiningretrievalandgeneration,RAG
enablesthesystemtolocatepreciseanswersindocumentsandpresent
themascoherentresponses,reducingtheneedformanualcompliance
audits. Thisapproachensuresconsistencyandaccuracyincompli-
anceassessmentswhilesignificantlyreducingthetimerequiredfor
document review by generating responses aligned with regulatory
expectations.
In our RAG-based compliance verification pipeline, document
chunking is an essential process for breaking down extensive reg-
ulatoryandcompanydocumentsintomanageable,contextuallyrele-
vantsegments. Toachievethis,weimplementedtwodistinctchunk-
ing strategies tailored to support precise retrieval while preserving
the necessary context for compliance queries. Our first approach
utilisesLangChain’sRecursiveCharacterTextSplitterwithadjustable
window sizes and overlapping contexts, supporting both sentence-
levelandparagraph-levelchunking. Sentence-levelchunkingoffers
fine-grainedcontrol,whichisbeneficialforretrievingspecificclauses
inresponsetodirectcompliancequeries. Incontrast,paragraph-level
chunkingcapturesbroadercontextualinformation,makingitideal
for complex queries that rely on layered responses. This flexibility
ensures that each chunk retains meaningful content without frag-
mentinglogicallyconnectedinformation. InadditiontoLangChain
3–10

CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs arXiv Preprint
Table 1.Anillustrativeexampleofasinglecontrolentryinthedataset,showingthecontrolquestion,theground-truthlabel,andthesupportingdocuments
andrationalerecordedduringtheoriginalmanualassessment.
Field Example content
ControlID C-087
Basicquestion Do your security incident management procedures ensure that personnel are assigned roles and
responsibilitiesforrespondingtoincidents?
Guidelines Review the incident response policy and supporting procedures. Confirm that named roles (e.g.,
incidentmanager,communicationslead)aredefined,thatresponsibilitiesaredocumented,andthat
thepolicyhasbeenreviewedwithinthepast12months.
Sourcedocuments InformationSecurityPolicyv3.2;IncidentResponseProcedure;AnnualSecurityReviewReport2023.
Ground-truthlabel Pass
Analystrationale(metadata
only)Rolesforincidentmanager,technicallead,andcommunicationsleadareclearlydefinedinSection4.2
oftheIncidentResponseProcedure. LastreviewedMarch2023.
textsegmentation,weappliedMeta-chunking[24]tosegmentdocu-
mentsbasedondeeperlogicalrelationshipsratherthanfixedstruc-
tures. Meta-chunkingidentifiesnaturalboundarieswithinthetextby
groupingsentenceswithsharedlogicalconnections,suchascausal
or transitional elements. This strategy enhances retrieval accuracy
bycapturingmorecoherentsegments,enablingresponsesthatalign
closelywiththeintentandcontextofregulatorydocuments,rather
thanrelyingsolelyonsurface-leveltextfeatures.
Byindependentlydeployingthesechunkingmethods,weensure
thatretrievaloperatesoncontextuallyalignedsegmentstailoredto
thediversestructuresofcompliancedocuments. Thisdualstrategy
allowsustoaccommodatebothgranularandbroadercontextrequire-
ments,providingafoundationforaccurateandrelevantresponsesin
complianceverificationtasks.
Toenhancecomplianceverification,weusedqueryaugmentation
to add relevant document context to each regulatory control ques-
tion. Byincorporatingthetop-kretrievedchunksintoeachquery,we
providedarichercontextthatimprovedthealignmentofresponses
withcompliancerequirements. Theapproachstructuresqueryinputs
to allow the retrieval model to draw specific, contextually relevant
informationfromcompanydocuments,facilitatingamoreaccurate
generationofcomplianceresponses.
4. DATASET
Thecustomdatasetusedinthisstudywascuratedtoevaluateregula-
torycompliancewithinacorporateenvironment. Itincludes45PDF
documents that provide comprehensive insights into the company
operations,coveringcriticalareassuchassecurityprotocols,human
resourcespolicies,employeetrainingandonboardingprocess,com-
panypolicies,legalagreements,andannualreports. Eachdocument
isconsideredapotentialsourceofinformationrelevanttoverifying
thecompanyadherencetospecifiedregulatorystandards. Thedataset
alsocomprises240’controls’,whichareessentiallyquestionsderived
fromregulatoryguidelines. Eachcontrolhastwoparts: abasicques-
tionandanaccompanyingsetofguidelines. Thebasicquestiontargets
specificcompliancerequirements,whiletheguidelinesprovidehu-
man data analysts with structured instructions on where to search
for relevant information and how to assess compliance within the
documents. Table1showsanillustrativesampleoftherecords.
For each control, the dataset also includes binary ground truth
responses,labelledas’Pass’or’Fail,’indicatingcomplianceornon-
compliance with the corresponding regulatory standard. These re-
sponseswereestablishedbyateamofexpertcomplianceanalysts,and
eachwasfurthervalidatedthroughqualitycontroltoensuretherelia-
bilityofthedatasetanswers. Additionally,eachcontrolintheoriginal
complianceassessmentincludedopen-endedcommentary,detailing
metadatasuchastheanalystresponsible,thedateofthecompliance
check,thespecificdocumentsconsulted,andtherationalebehindthe
compliancedecision. However,tomaintainfocusandensureclarityinevaluation,onlythebinary’Pass’or’Fail’responseswereusedas
thegroundtruthanswerinthisstudy,asassessingopen-endedjusti-
ficationswouldrequireextensiveinterpretiveanalysis. Thedetailed
responseexplanation,however,helpedusinimprovingtheresponse
qualityandguidingthemodeltogivetherightresponse.
RAGaugmentsparametricmemorywithnon-parametricretrieval
anddoesnotrequiremodeltraining. Therefore,thedatasetisexclu-
sively used for testing, with no portion allocated for training. This
enabledarigorousevaluationoftheRAGpipelineretrievalandgener-
ationcapabilities. Althoughsmallerinscale,thedatasetpreciseand
verifiedresponsesmakeitarobustbaselineforhigh-stakescompli-
ancecheckingwhereaccuracyisahighfactor. Thedistinctstructure
ofthisdatasetcreatesareliablefoundationforadvancingcompliance
verificationusingRAGmodels.
5. EXPERIMENTS
Ourexperimentsweredesignedtoevaluatetheimpactofchunking
strategies,retrievalconfigurations,andin-contextlearningtechniques
withinourRAG-basedcomplianceverificationpipeline. Theprimary
objectivewastooptimisethepipelineresponseaccuracy,relevance,
and alignment with regulatory standards. Compliance verification
wasframedasaquestion-answeringtask,witheachregulatorycontrol
posedasaquery. Throughsystematicexperimentation,weassessed
theinfluenceofcontentsegmentation,retrievalparameters,andre-
sponsegenerationtechniquesonthepipelineoverallperformance.
5.1. Chunking Strategies
Weexploredmultiplechunkingstrategiestooptimisethealignment
betweencontrolquestions,andretrievedcontext. Fixed-sizechunking
was tested across four primary configurations: 200, 800, 1600, and
3000characters. The200-characterconfigurationwasexcludedfrom
laterexperimentsduetoconsistentlylowperformance;theremaining
threeconfigurationsarereferredtoassmall(800),moderate(1600),
andlarge(3000)charsthroughoutthepaper. Thesmallconfiguration
provideshighergranularity,isolatingspecificclauseseffectivelyfor
clause-level compliance queries. However, its limited context can
fragmenttherelevantinformation,particularlyformulti-clausecom-
pliancerequirements. Incontrast,thelargeconfigurationpreserves
broadercontexts,benefitingopen-endedqueriesbutsometimesintro-
ducingirrelevantornoisyinformation,whichimpactsprecision.
Toaddressthelimitationsoffixed-sizechunking,weimplemented
Meta-chunking. Thisapproachsegmentsdocumentsbasedonlogi-
calcoherenceratherthanfixedlengths,ensuringcontextuallyrele-
vantgroupings. Thismethodeffectivelycapturedtopicshiftswithin
content-richdocuments,creatingcoherentsegmentsthatmaintained
logicalconnectionsacrosssentences. Meta-chunkingexcelledinalign-
ingcontentwithmulti-layeredcompliancequeriesbutincurredhigher
preprocessingtimesduetoitscomputationalcomplexity.
4–10

arXiv Preprint CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs
WeemployedOpenAItext-embedding-3-largeandtext-embedding-
ada-002 models1, both of which were effective in capturing
compliance-specificlanguage. EmbeddingswerestoredinFAISS[30],
ascalablevectorstore,ensuringefficientandrapidretrievalacrossthe
dataset. Persistentvectorstorage allowedus to minimiseingestion
timewhentestingdifferentgenerationstrategies.
5.2. Retrieval Configuration
The number of top-retrieved documents ( 𝐾) included in the query
contextwassystematicallyvariedtoevaluateitsimpactonresponse
quality. Higher 𝐾valuesenrichedthecontextualinformationavail-
abletothemodelbutincreasedtheriskofincorporatingirrelevant
data. Conversely, lower 𝐾values maintained precision by limiting
thecontextbutoccasionallyfailedtoretrievekeydetailsessentialfor
nuanced queries. Balancing 𝐾was critical for ensuring responses
alignedwithregulatoryexpectations,particularlyforindirectcompli-
ancescenarios.
5.3. Response Generation Techniques
Weexperimentedwithdifferentresponsegenerationmethodsusing
LangChainRetrievalQA. Two key configurations were tested:stuff
andmap_reducechains. The‘stuff‘chaincombinedretrievedcontent
intoasingleresponse,deliveringefficientresultsforstraightforward
compliancechecks. Incontrast,themap_reducechainsynthesised
responses from individual retrieved documents, enabling nuanced
answersforcomplex,multi-sourcequeries. Bothmethodswereeval-
uatedfortheirabilitytoproduceaccurateandcontextuallyaligned
responses.
6. RESULTS
Theexperimentsconductedforcompliancecheckingevaluatedthe
performance of various chunking strategies, retrievals, LLMs, and
in-contextlearningprompts,usingprecision,recall,andF1-scoreas
keyevaluationmetrics. Theseexperimentsweredesignedtoexplore
theinfluenceofchunkingconfigurationsandmodelarchitectureson
compliance-related tasks. The chunking strategies tested included
fixed-size chunking with character lengths of 200, 800, 1600, and
3000,aswellasmeta-chunking,whichgroupstextbasedonlogical
coherence rather than fixed size. As shown in Figure 3, the small-
est200-characterchunkssubstantiallyincreasedtheoveralltimere-
quiredtopreprocessthedataset,takingnearlyanorderofmagnitude
longerthantheFixed800configuration. Fixed-sizechunksofFixed800,
Fixed1600,andFixed3000charactersallcompletedinunder70seconds.
Meta-chunkingrequiredsignificantlymoreprocessingtimethanany
fixed-size strategy due to its complexity. Within each model, gen-
erationtimewaslargelyunaffectedbythechunkingstrategyorthe
numberofretrievedchunks;thedominantdriverofgenerationtime
wasthechoiceofLLMitself. Amongthetestedmodels,Gemini-Pro
wasconsiderablyslowerthantheothersacrosseverychunkingcon-
figuration, withgenerationtimesroughlytwicethoseofGPT-4oor
Gemini Flash. This consistent gap suggests Gemini-Pro performs
morecomputationallyintensiveevaluationregardlessofinputsize.
Weobservedaconsiderablylowhitrateofapproximately22%when
usingsmallerchunks,comparedtotheactualhitrateof80%. Thehit
raterepresentstheproportionofinstanceswhererelevantcontextwas
successfullyretrievedtoperformthecompliancecheck. Wecouldnot
directlycomparetheretrievedchunkswiththeground-truthevidence
documents,asitrequiredmanualcheckingandthereforewerather
calculatedthehitrateonthebasisofthegeneratedresponsewherea
No-Evidenceisconsideredtobemissinginformation.
To investigate the optimal chunking size, Tables 2–6 provide a
detailedanalysisofresponsequalityacrossvariousmodels,utilising
differentchunkingstrategiesandtop- 𝐾documentretrievalsettings.
1https://platform.openai.com/docs/guides/embeddingsTable 2.Class-wisePrecision,Recall,andF1-Scoreforcompliancechecking
usingGPT-4modelacrossvariouschunkingstrategiesandnumberofchunks
Chunking KCompliant Non-Comp. No-Evidence
P R F1 P R F1 P R F1
Fixed 2001 55 16 24 0 0 0 18 71 29
3 56 17 26 7 3 4 18 71 29
5 52 14 22 12 5 7 18 69 28
Fixed 8001 65 39 49 44 10 16 27 83 40
3 69 47 56 64 23 33 28 77 42
5 67 46 55 60 15 24 28 77 41
Fixed 16001 63 31 41 58 18 27 24 83 38
3 68 57 62 70 18 28 27 63 38
5 70 62 66 82 23 35 30 66 41
Fixed 30001 77 31 44 57 20 30 24 86 37
3 69 56 62 82 23 35 27 66 38
5 69 60 64 86 30 44 27 57 36
Meta- 1 61 26 36 67 10 17 21 80 34
chunking 3 65 41 50 89 20 33 26 80 40
5 70 53 60 90 23 36 26 69 38
Table 3.Class-wisePrecision,Recall,andF1-Scoreforcompliancechecking
usingGPT-3.5Turbomodelacrossvariouschunkingstrategiesandnumberof
chunks
Chunk KCompliant Non-Comp. No-Evidence
P R F1 P R F1 P R F1
Fixed 2001 55 16 24 0 0 0 19 80 31
3 50 16 24 0 0 0 18 74 30
5 54 18 27 10 3 4 18 71 29
Fixed 8001 65 39 49 17 3 4 26 83 39
3 68 46 55 38 8 13 27 77 40
5 67 46 55 43 8 13 27 77 40
Fixed 16001 61 45 52 50 3 5 27 77 40
3 65 68 66 67 5 9 31 60 41
5 62 68 65 100 5 10 27 49 35
Fixed 30001 65 46 54 50 5 9 25 71 37
3 63 60 61 100 5 10 27 60 38
5 65 65 65 100 8 14 30 63 41
Meta- 1 57 37 45 60 8 13 22 69 34
chunking 3 59 47 53 80 10 18 23 60 33
5 62 54 57 67 10 17 25 60 36
Eachtablefocusesonadistinctmodel, establishingastandardised
benchmarkforevaluatingchunkingvariations.
Tables 2–6 report class-wise precision, recall, and F1 scores for
the five LLMs across the four chunking strategies and three values
ofK.Eachtableisolatesonemodelsothatchunkingeffectscanbe
comparedwithoutconfoundingbymodelarchitecture. Acrossallfive
tables,twopatternsareconsistent.
First, the Fixed200configuration underperforms in every model.
WhileitoccasionallyyieldshighprecisionontheCompliantandNon-
Compliantclasses,aside-effectofthemodelidentifyingveryfewcases
andbeingcorrectonmostofthem,recallintheseclassesisuniformly
low, dragging F1 to around 20–30% across the board. Increasing K
from1to5doeslittletohelp,indicatingthattheproblemisthechunk
contentitselfratherthantheretrievaldepth.
Second,increasingthechunksizehasaclearlypositiveeffect,par-
ticularlyontheunder-representedNon-CompliantandNo-Evidence
classes. TheFixed1600andFixed3000configurationsproducebalanced
and consistently higher F1 scores across all three classes. Meta-
chunkingachievesgoodprecisiononCompliantandNon-Compliant
casesbutsuffersfromlowrecall,drivenbyaninflatedNo-Evidence
rate. Themethodtendstomarkcontrolquestionsaslackingevidence
evenwhensupportingpassagesarepresent.
Aseparateobservationconcernsmodelsthatperformpoorlyun-
der direct prompting. GPT-4o and Gemini Flash miss most Non-
Compliantcaseswithoutfurtherguidance: inTable5,theFixed200
configurationshows100%precisiononNon-Compliantbutonly ∼17%
5–10

CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs arXiv Preprint
Figure 3.Processingtimesacrossmodelsandchunkingstrategies,withgenerationtime(a–e)andchunkingtime(f).
Table 4.Class-wisePrecision,Recall,andF1-Scoreforcompliancechecking
usingGPT-4omodelacrossvariouschunkingstrategiesandnumberofchunks
Chunking KCompliant Non-Comp. No-Evidence
P R F1 P R F1 P R F1
Fixed 2001 56 17 26 0 0 0 19 80 31
3 57 19 28 13 3 4 19 77 31
5 60 19 29 14 3 4 20 80 32
Fixed 8001 66 39 49 17 3 4 26 83 39
3 70 48 57 44 10 16 27 77 40
5 67 47 55 43 8 13 27 77 40
Fixed 16001 66 44 53 40 5 9 27 83 41
3 66 66 66 33 3 5 31 63 41
5 67 72 69 33 3 5 35 63 45
Fixed 30001 66 43 52 50 3 5 25 80 38
3 66 66 66 50 3 5 30 63 40
5 68 73 70 25 3 5 34 60 43
Meta- 1 66 36 47 33 3 5 24 83 37
chunking 3 68 48 57 50 3 5 28 83 41
5 67 56 61 50 3 5 28 71 40
recall,becausethemodelcorrectlyclassifiesthefewcasesitdoesflag
but flags very few overall. This motivates the in-context learning
experimentsdescribednext.
Inourexperiments,wealsoexploredtheuseofin-contextlearning
toenhancethedecision-makingcapabilitiesofamodelandimprove
the overall response quality. Analysing error cases from the direct
prompting approach revealed consistent shortcomings in the abil-
ityofamodeltodeterminecontrolcomplianceaccuratelybasedon
theretrieveddocumentinformation. Althoughrelevantinformation
was often present, the model struggled to align its decisions with
human annotators, particularly in nuanced cases such as indirectTable 5.Class-wisePrecision,Recall,andF1-Scoreforcompliancechecking
usingGemini-Flashmodelacrossvariouschunkingstrategiesandnumberof
chunks
Chunking KCompliant Non-Comp. No-Evidence
P R F1 P R F1 P R F1
Fixed 2001 55 15 23 18 5 8 20 80 31
3 58 17 26 18 5 8 20 80 32
5 60 17 26 23 8 11 20 80 32
Fixed 8001 70 41 51 17 3 4 25 80 38
3 69 50 58 38 8 13 27 74 39
5 67 48 56 58 18 27 28 74 41
Fixed 16001 60 29 39 17 3 4 22 80 35
3 61 47 53 50 8 13 24 63 34
5 64 56 59 100 18 30 28 66 39
Fixed 30001 62 26 37 33 3 5 22 86 35
3 64 49 56 50 5 9 24 66 35
5 67 63 65 25 3 5 29 63 39
Meta- 1 53 23 32 75 8 14 20 77 32
chunking 3 53 27 36 80 10 18 24 83 37
5 52 31 39 80 10 18 23 74 35
compliance. One such scenario involved vendors leveraging third-
party services, such as cloud service providers, that complied with
specificcontrols. Humanannotatorsconsideredsuchcasescompliant
because the third-party compliance indirectly satisfied the control
requirements. However,ourCTRAGframeworkclassifiedthesecases
asnon-compliant,asthevendoritselfdidnotdirectlyadheretothe
controlrequirements. Byincorporatingin-contextlearning,wewere
abletoguidethemodelinadaptingtothesescenarios,enablingitto
mimichuman-likedecision-makingmoreeffectively.
In-contextlearningcapabilitieswerealsoemployedtoaddressar-
easwherethemodelconsistentlymadeerrors. Byprovidingdetailed
6–10

arXiv Preprint CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs
Figure 4.F1scoreperformanceafterin-contextlearningacrossfourLLMsandthreechunksizes.
Table 6.Class-wisePrecision,Recall,andF1-Scoreforcompliancechecking
usingGemini-Promodelacrossvariouschunkingstrategiesandnumberof
chunks
Chunking KCompliant Non-Comp. No-Evidence
P R F1 P R F1 P R F1
Fixed 2001 59 16 25 21 8 11 20 80 32
3 58 17 26 29 13 18 20 77 32
5 58 17 26 29 13 18 20 77 32
Fixed 8001 71 38 49 29 5 9 24 80 37
3 68 49 57 17 3 4 26 74 39
5 65 48 55 44 10 16 28 74 40
Fixed 16001 69 47 56 0 0 0 25 74 38
3 66 46 54 25 5 8 26 74 39
5 63 92 74 44 10 16 63 29 39
Fixed 30001 61 82 70 0 0 0 26 26 26
3 61 86 72 30 8 12 33 20 25
5 60 86 71 40 5 9 13 9 10
Meta- 1 63 88 73 40 5 9 19 14 16
chunking 3 64 90 75 75 15 25 29 20 24
5 63 83 72 56 23 32 33 23 27
contextualexamplesthatillustratedspecificdecision-makingrules,
theabilityofamodeltoclassifycasesimprovedsignificantly. Table
7andFigure4demonstratethisimprovement,highlightingtheim-
portanceofin-contextlearninginaligningthemodelresponseswith
human judgment. As Figure 4 (a) shows, every model exhibits the
sameclassranking;Compliantscoreshighest,No-Evidencelowest,
regardless of chunk size, suggesting that the difficulty hierarchy is
intrinsictothetaskratherthanaquirkofanysinglemodelorchunk-
ing strategy. Figure 4 (b) further shows that weighted F1 declines
monotonicallyaschunksizegrowsacrossallfourmodels,withsmall
chunksconsistentlyoutperforminglargerones.
TheconfigurationsreportedinTable7reflectadeliberatenarrowing
oftheexperimentalspace. Weomittedthe200-characterchunksize,
asitsperformancewasconsistentlylowacrossallmetricsinthedirect-
promptingexperiments,andincreasingtheretrievaldepthfailedto
improveitseffectiveness. Meta-chunkingwasalsoexcludedduetoits
prohibitivechunkingtime,whichmadeitunsuitableforthisparticular
usecase.
Fortheseexperiments,wefixedtheretrievaldepthat 𝐾= 4basedonthefindingsfromthedirect-promptingexperiments. Acrossthe
evaluated models, retrieval depths between 𝐾= 3and𝐾= 5con-
sistentlyprovidedthebestbalancebetweencontextualcompleteness
andretrievalnoise. Selecting 𝐾= 4allowedustocapturesufficient
supporting evidence for compliance assessment while limiting the
inclusionofirrelevantinformationthatcouldnegativelyaffectgen-
erationquality. Thissettingthereforerepresentsapracticaltrade-off
betweenretrievaleffectivenessandcomputationalefficiency. Using
thisconfiguration,Fixed800chunksyieldedthestrongestoverallre-
sults across the compliance classes, particularly when paired with
GPT-4. Similar trends were observed across the other models, sup-
portingourobjectiveofmaintainingsmallerchunksizestominimise
tokenconsumptionwhilepreservingretrievaleffectiveness.
AsignificantchallengewasthelowrecallfortheNon-Compliant
class,whichindicatestheriskofmissingsomecompliancefailcases,
a critical concern for compliance checking. Many of these missed
caseswereinsteadclassifiedundertheNo-Evidenceclass,reflecting
situationswherethemodelfailedtofindrelevantinformationtode-
termine compliance. In the final solution, however, No-Evidence
classifications were reclassified as Non-Compliant, as the absence
of relevant information typically implies that the vendor does not
comply with the rules and regulations and therefore this informa-
tionisnotavailableinthedocuments. Thisreclassificationapproach
significantlyimprovedrecallfortheNon-Compliantclassavoiding
expensivefine-tuningforthetask,asshowninTable8. Byaddressing
theNo-Evidencecasesinthismanner,wecapturedmanypreviously
missedcompliancefailcases,achievingamorerobustandrealistic
compliance-checkingsolution. ThefinalresultspresentedinTable8
demonstrateanacceptablelevelofcompliance-checkingperformance,
aligningwellwithourobjectivesandthetaskrequirements.
TheheatmapinFigure5highlightsthatsmallchunkswith 𝐾= 4
consistently deliver better performance across all models, empha-
sising that increasing the chunk size does not necessarily enhance
compliance-checking capabilities. This finding may be attributed
totwosignificantfactors. First,the"needle-in-a-haystack"effectbe-
comesmoreprominentwithlargerchunks. Whenthemodelispre-
sentedwithasubstantialamountofcontext,therelevantinformation
might get buried within a sea of less pertinent details. This over-
whelming context canmakeit challenging forthe model toextract
7–10

CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs arXiv Preprint
Table 7.ImprovingthecompliancecheckingcapabilitiesoftheLLMsusingin-contextlearning
Model Chunking Compliant Non-Compliant No-Evidence Weighted Average
P(%) R(%) F1(%) P(%) R(%) F1(%) P(%) R(%) F1(%) P(%) R(%) F1(%)
GPT-4 Small 88.30 76.85 82.18 70.97 55.00 61.97 37.93 62.86 47.31 74.88 69.40 71.09
Moderate 81.55 77.78 79.62 62.86 55.00 58.67 37.78 48.57 42.50 69.09 67.21 67.94
Large 78.13 69.44 73.53 54.76 57.50 56.10 37.78 48.57 42.50 65.30 62.84 63.78
GPT-4o Small 81.52 69.44 75.00 43.48 50.00 46.51 38.64 48.57 43.04 65.00 61.20 62.66
Moderate 77.32 69.44 73.17 39.62 52.50 45.16 42.42 40.00 41.18 62.41 60.11 60.93
Large 75.00 63.89 69.00 38.60 55.00 45.36 41.18 40.00 40.58 60.57 57.38 58.40
Gemini Small 76.29 68.52 72.20 42.86 52.50 47.19 41.67 42.86 42.25 62.36 60.11 61.00
Flash Moderate 73.27 68.52 70.81 36.84 52.50 43.30 40.00 28.57 33.33 58.94 57.37 57.63
Large 70.41 63.89 66.99 37.05 57.50 45.54 41.67 28.57 33.90 57.76 55.74 56.30
Gemini Small 77.78 77.78 77.78 61.29 47.50 53.52 38.64 48.57 43.04 66.69 65.57 65.83
Pro Moderate 73.91 78.70 76.23 55.56 50.00 52.63 37.50 34.29 35.82 62.94 63.93 63.35
Large 69.44 69.44 69.44 46.51 50.00 48.19 37.50 34.29 35.82 58.32 58.47 58.37
Table 8.PerformanceComparisonAcrossModelsforOverallMetrics
Model Chunking Precision (%) Recall (%) F1 (%)
GPT-4 Small 71.91 85.33 78.05
Moderate 70.00 74.67 72.26
Large 62.07 72.00 66.67
GPT-4o Small 63.74 77.33 69.88
Moderate 61.63 70.67 65.84
Large 57.14 69.33 62.65
Gemini Flash Small 60.47 69.33 64.60
Moderate 58.54 64.00 61.15
Large 54.12 61.33 57.50
Gemini Pro Small 68.00 68.00 68.00
Moderate 66.18 60.00 62.94
Large 56.00 56.00 56.00
andsynthesizethekeyinformationrequiredtoarriveataconclusive
decision, thereby reducing its effectiveness. Second, larger chunks
inherentlycontainmoreirrelevantornoisytext. Thisadditionalnoise
increasesthecomplexityoftheretrievaltask,astheretrievermight
struggle to focus on the most relevant portions of the text. Conse-
quently,thiscanleadtosituationswheretheretrievermissescritical
chunksthatcontainthenecessaryevidenceforcompliancechecking.
Thecombinationofincreasednoiseandthedifficultyofamodelin
narrowingdownessentialinformationcouldexplainthediminished
performanceobservedwithlargerchunks. Theseobservationsunder-
scoretheimportanceofselectinganoptimalchunksizethatbalances
contextualrichnesswithretrievabilityandprocessingefficiency,en-
suring the ability to focus on the most relevant information while
minimisingnoise.
7. CONCLUSION
Thispaperinvestigatedtheimpactofchunkingstrategies,retrieval
configurations,LLMarchitectures,andin-contextlearningonauto-
mated compliance checking. Across the evaluated configurations,
smallchunkswith𝐾= 4achievedthemostconsistentperformance,
whilelargerchunksgenerallysufferedfromreducedprecisionand
recallduetocontextdilution. Meta-chunking,whileintendedtopre-
servelogicalcoherence,incurredprohibitivepreprocessingcostsand
was not competitive with fixed-size chunking on this task. These
findings suggest that retrieval quality and evidence granularity are
moreimportantthanincreasingcontextsizeforcompliance-oriented
documentanalysis. In-contextlearningsubstantiallyimprovedhow
themodelshandledcasesthatdirectpromptingstruggledwith,par-
ticularly indirect compliance via third-party services. Gains were
mostpronouncedfortheunder-representedNon-CompliantandNo-
Evidenceclasses,wheretargetedexampleshelpedthemodeladoptthe
decisionrulesusedbyhumanannotators. Thereporteddeployment
inaBigFourprofessionalservicesfirmproducedanapproximately
60% reduction in manual reviewer effort relative to a firm existing
Figure 5.HeatmapofF1scoresacrossmodelsandchunkingstrategies,
highlightingperformancevariations
process. These findings suggest the approach is well suited to scal-
ingbeyondthesingle-organizationdeploymentevaluatedhere,and
toadjacentdocument-analysistaskssuchaspolicygapanalysisand
contract review. The study has several limitations. The evaluation
is based on 240 controls and 45 documents from a single organiza-
tion;broadertestingacrossregulatorydomainsisneededtoestablish
howwellthechunkingandICLfindingsgeneralize. Thedatasetis
proprietary and cannot be released. Inter-annotator agreement on
thebinaryPass/Faillabelswasnotmeasured,andthefinalpipeline
reliesonaconservativeNo-Evidence-to-Non-Compliantreclassifica-
tionthattradesprecisionforrecall. Futureworkwilladdressthese
limitations,runstatisticalsignificancetestsagainstthechunkingand
K configurations, and extend CTRAG with proper Meta-Chunking
variants,semanticchunkingandnon-RAGbaselines.
8. ACKNOWLEDGMENTS
ThisresearchissupportedbytheAdvancedResearchandEngineering
Centre (ARC) in Northern Ireland, funded by PwC and Invest NI.
Theviewsexpressedarethoseoftheauthorsanddonotnecessarily
representthoseofARCorthefundingorganisations.
TheauthorsappreciatetheuseoftheKelvin2HighPerformance
ComputingclusteratQueen’sUniversityBelfast,andthecloudser-
vicesandAPIsprovidedbyPwCforcomputationalwork.
REFERENCES
[1]P.Lewis,E.Perez,A.Piktus,F.Petroni,V.Karpukhin,N.Goyal,
H.Küttler,M.Lewis,W.-t.Yih,T.Rocktäscheletal.,“Retrieval-
8–10

arXiv Preprint CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs
augmentedgenerationforknowledge-intensivenlptasks,”Ad-
vancesinneuralinformationprocessingsystems,vol.33,pp.9459–
9474,2020.
[2]T.Brown,B.Mann,N.Ryder,M.Subbiah,J.D.Kaplan,P.Dhari-
wal,A.Neelakantan,P.Shyam,G.Sastry,A.Askelletal.,“Lan-
guagemodelsarefew-shotlearners,”Advancesinneuralinfor-
mationprocessingsystems,vol.33,pp.1877–1901,2020.
[3]Y.Zhou,J.Li,Y.Xiang,H.Yan,L.Gui,andY.He,“Themysteryof
in-contextlearning: Acomprehensivesurveyoninterpretation
andanalysis,”inProceedingsofthe2024ConferenceonEmpirical
MethodsinNaturalLanguageProcessing,2024,pp.14365–14378.
[4]M.e.Kharbili,A.K.A.d.Medeiros,S.Stein,andW.M.vander
Aalst,“Businessprocesscompliancechecking: Currentstateand
futurechallenges,”ModellierungbetrieblicherInformationssys-
teme(MobIS2008),pp.107–113,2008.
[5]M.Hashmi,G.Governatori,H.-P.Lam,andM.T.Wynn,“Are
wedonewithbusinessprocesscompliance? stateoftheartand
challengesahead,”KnowledgeandInformationSystems,vol.57,
no.1,pp.79–133,2018.
[6]N.Chen,X.Lin,H.Jiang,andY.An,“Automatedbuildingin-
formationmodelingcompliancecheckthroughalargelanguage
model combinedwith deeplearning and ontology,”Buildings,
vol.14,no.7,p.1983,2024.
[7]R. Amor and J. Dimyadi, “The promise of automated compli-
ancechecking,”Developmentsinthebuiltenvironment,vol.5,p.
100039,2021.
[8]T.Beach,J.Yeung,N.Nisbet,andY.Rezgui,“Digitalapproaches
toconstructioncompliancechecking: Validatingthesuitability
ofanecosystemapproachtocompliancechecking,”Advanced
EngineeringInformatics,vol.59,p.102288,2024.
[9]T.Yanagawa,V.Agarwal,Y.Watanabe,L.Degenaro,andA.Sailer,
“Asecureframeworkforcontinuouscomplianceacrossheteroge-
neouspolicyvalidationpoints,”in2024IEEE17thInternational
ConferenceonCloudComputing(CLOUD),2024,pp.176–182.
[10]M.Humayun,M.Niazi,M.F.Almufareh,N.Z.Jhanjhi,S.Mah-
mood, and M. Alshayeb, “Software-as-a-service security chal-
lengesandbestpractices: Amultivocalliteraturereview,”Ap-
pliedSciences,vol.12,no.8,p.3953,2022.
[11]T. H. Beach, Y. Rezgui, H. Li, and T. Kasim, “A rule-based se-
mantic approach for automated regulatory compliance in the
construction sector,”Expert systems with applications, vol. 42,
no.12,pp.5219–5231,2015.
[12]V.Jain,A.Balakrishnan,D.Beeram,M.Najana,andP.Chintale,
“Leveragingartificialintelligenceforenhancingregulatorycom-
plianceinthefinancialsector,”InternationalJournalofComputer
TrendsandTechnology,vol.72,no.5,pp.116–125,2024.
[13]A. Berger, L. Hillebrand, D. Leonhard, T. Deußer, T. B. F.
De Oliveira, T. Dilmaghani, M. Khaled, B. Kliem, R. Loitz,
C.Bauckhageetal.,“Towardsautomatedregulatorycompliance
verificationinfinancialauditingwithlargelanguagemodels,”
in2023 IEEE International Conference on Big Data (BigData).
IEEE,2023,pp.4626–4635.
[14]D. M. Salama and N. M. El-Gohary, “Semantic text classifica-
tionforsupportingautomatedcompliancecheckinginconstruc-
tion,”JournalofComputinginCivilEngineering,vol.30,no.1,p.
04014106,2016.[15]J. Zhang and N. M. El-Gohary, “Semantic nlp-based informa-
tionextractionfromconstructionregulatorydocumentsforau-
tomated compliance checking,”Journal of computing in civil
engineering,vol.30,no.2,p.04015014,2016.
[16]P.ZhouandN.El-Gohary,“Ontology-basedautomatedinforma-
tion extraction from building energy conservation codes,”Au-
tomationinConstruction,vol.74,pp.103–117,2017.
[17]R. Zhang and N. El-Gohary, “Building information modeling,
naturallanguageprocessing,andartificialintelligenceforauto-
matedcompliancechecking,”inResearchcompaniontobuilding
informationmodeling. EdwardElgarPublishing,2022,pp.248–
267.
[18]M. Alnuzha and T. Bloch, “The role of machine learning in
automatedcodechecking-asystematicliteraturereview,”Journal
of Information Technology in Construction, vol. 30, pp. 22–44,
2025.
[19]Z.Zhang,L.Ma,andN.Nisbet,“Unpackingambiguityinbuild-
ingrequirementstosupportautomatedcompliancechecking,”
JournalofManagementinEngineering,vol.39,no.5,p.04023033,
2023.
[20]J. Jain, N. Dhanasekaran, and M. Diab, “From complexity to
clarity: Ai/nlp’sroleinregulatorycompliance,”inFindingsof
theAssociationforComputationalLinguistics: ACL2025,2025,
pp.26629–26641.
[21]J.Dimyadi,R.Amor,andW.Solihin,“Leveraginglargelanguage
models for BIM-based automated compliance checking,”Au-
tomationinConstruction,vol.170,p.106707,2025.
[22]A.Brown,M.Roman,andB.Devereux,“Asystematicliterature
reviewofretrieval-augmentedgeneration: Techniques,metrics,
andchallenges,”BigDataandCognitiveComputing,vol.9,no.12,
p.320,2025.
[23]M. Cheng, Y. Luo, J. Ouyang, Q. Liu, H. Liu, L. Li, S. Yu,
B. Zhang, J. Cao, J. Ma, D. Wang, and E. Chen, “A survey
onknowledge-orientedretrieval-augmentedgeneration,”arXiv
preprintarXiv:2503.10677,2025.
[24]J.Zhao,Z.Ji,Y.Feng,P.Qi,S.Niu,B.Tang,F.Xiong,andZ.Li,
“Meta-chunking: Learningtextsegmentationandsemanticcom-
pletionvialogicalperception,”arXivpreprintarXiv:2410.12788,
2024.
[25]Q.Zhao,R.Wang,Y.Cen,D.Zha,S.Tan,Y.Dong,andJ.Tang,
“LongRAG:Adual-perspectiveretrieval-augmentedgeneration
paradigmforlong-contextquestionanswering,”inProceedingsof
the2024ConferenceonEmpiricalMethodsinNaturalLanguage
Processing. AssociationforComputationalLinguistics,2024,
pp.22600–22632.
[26]T. Zhang, Z. Jiang, S. Bai, T. Zhang, L. Lin, Y. Liu, and J. Ren,
“Rag4itops: Asupervisedfine-tunableandcomprehensiverag
frameworkforitoperationsandmaintenance,”inProceedingsof
the2024ConferenceonEmpiricalMethodsinNaturalLanguage
Processing: IndustryTrack,2024,pp.738–754.
[27]T.Gokhan,K.Wang,I.Gurevych,andT.Briscoe,“Rirag: Regula-
toryinformationretrievalandanswergeneration,”arXivpreprint
arXiv:2409.05677,2024.
[28]K.Umar,H.Doğan,O.Özcan,I.Karakaya,A.Karamanlıoğlu,
andB.Demirel,“Enhancingregulatorycompliancethroughau-
tomatedretrieval,reranking,andanswergeneration,”inProceed-
ingsofthe1stRegulatoryNLPWorkshop(RegNLP2025),2025,pp.
91–96.
9–10

CTRAG: An In-Context Retrieval-based Framework for Automated Compliance Checking using LLMs arXiv Preprint
[29]J. Huang, W. Ping, P. Xu, M. Shoeybi, K. C.-C. Chang, and
B. Catanzaro, “RAVEN: In-context learning with retrieval-
augmented encoder-decoder language models,”Transactions
onMachineLearningResearch,2024.
[30]J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity
searchwithGPUs,”IEEETransactionsonBigData,vol.7,no.3,
pp.535–547,2021.
APPENDIX
.1. Complete K-level breakdown of chunking and generation time
Table 9.Chunking and Generation Times Across Models and Chunking
Strategies
Chunking Chunking Generation Times Avg.
Strategy Time K=1 K=3 K=5
GPT-3.5 Turbo
Fixed 200 263 98 100 103 100
Fixed 800 65 101 102 103 102
Fixed 1600 42 108 110 111 110
Fixed 3000 27 109 110 113 111
Meta-chunking 1674 105 106 108 106
GPT-4
Fixed 200 263 107 109 111 110
Fixed 800 65 105 106 108 107
Fixed 1600 42 114 115 117 115
Fixed 3000 27 119 121 122 121
Meta-chunking 1674 117 118 120 119
GPT-4o
Fixed 200 263 92 93 95 94
Fixed 800 65 81 82 84 82
Fixed 1600 42 82 84 85 84
Fixed 3000 27 84 85 87 85
Meta-chunking 1674 80 81 83 82
Gemini Flash
Fixed 200 263 88 90 92 90
Fixed 800 65 75 76 78 76
Fixed 1600 42 76 77 79 77
Fixed 3000 27 80 81 83 82
Meta-chunking 1674 79 81 82 81
Gemini Pro
Fixed 200 263 198 200 203 200
Fixed 800 65 171 172 174 173
Fixed 1600 42 167 169 171 169
Fixed 3000 27 156 158 160 158
Meta-chunking 1674 191 193 196 194
10–10