# MRD-RAG: Enhancing Medical Diagnosis with Multi-Round Retrieval-Augmented Generation

**Authors**: Yixiang Chen, Penglei Sun, Xiang Li, Xiaowen Chu

**Published**: 2025-04-10 13:17:51

**PDF URL**: [http://arxiv.org/pdf/2504.07724v1](http://arxiv.org/pdf/2504.07724v1)

## Abstract
In recent years, accurately and quickly deploying medical large language
models (LLMs) has become a significant trend. Among these, retrieval-augmented
generation (RAG) has garnered significant attention due to its features of
rapid deployment and privacy protection. However, existing medical RAG
frameworks still have shortcomings. Most existing medical RAG frameworks are
designed for single-round question answering tasks and are not suitable for
multi-round diagnostic dialogue. On the other hand, existing medical
multi-round RAG frameworks do not consider the interconnections between
potential diseases to inquire precisely like a doctor. To address these issues,
we propose a Multi-Round Diagnostic RAG (MRD-RAG) framework that mimics the
doctor's diagnostic process. This RAG framework can analyze diagnosis
information of potential diseases and accurately conduct multi-round diagnosis
like a doctor. To evaluate the effectiveness of our proposed frameworks, we
conduct experiments on two modern medical datasets and two traditional Chinese
medicine datasets, with evaluations by GPT and human doctors on different
methods. The results indicate that our RAG framework can significantly enhance
the diagnostic performance of LLMs, highlighting the potential of our approach
in medical diagnosis. The code and data can be found in our project website
https://github.com/YixiangCh/MRD-RAG/tree/master.

## Full Text


<!-- PDF content starts -->

Highlights
MRD-RAG: Enhancing Medical Diagnosis with Multi-Round Retrieval-Augmented Gener-
ation
Yixiang Chen, Penglei Sun, Xiang Li, Xiaowen Chu
•LLMs should mimic doctors via multi-round dialogue for accurate diagnosis
•Tree structure knowledge base facilitates the retrieval of diagnosis information
•Pseudo medical history bridges diseases’ info and patients’ utterances semanticallyarXiv:2504.07724v1  [cs.CL]  10 Apr 2025

MRD-RAG: Enhancing Medical Diagnosis with Multi-Round
Retrieval-Augmented Generation
Yixiang Chena, Penglei Suna,∗, Xiang Liaand Xiaowen Chua,∗
aThrust of Data Science and Analytics, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, 511453, China
ARTICLE INFO
Keywords :
Large Language Model
Retrieval-Augmented Generation
Medical DiagnosisABSTRACT
In recent years, accurately and quickly deploying medical large language models (LLMs) has
become a significant trend. Among these, retrieval-augmented generation (RAG) has garnered
significant attention due to its features of rapid deployment and privacy protection. However,
existing medical RAG frameworks still have shortcomings. Most existing medical RAG frame-
worksaredesignedforsingle-roundquestionansweringtasksandarenotsuitableformulti-round
diagnostic dialogue. On the other hand, existing medical multi-round RAG frameworks do not
consider the interconnections between potential diseases to inquire precisely like a doctor. To
addresstheseissues,weproposea Multi-Round DiagnosticRAG( MRD-RAG )frameworkthat
mimicsthedoctor’sdiagnosticprocess.ThisRAGframeworkcananalyzediagnosisinformation
of potential diseases and accurately conduct multi-round diagnosis like a doctor. To evaluate
theeffectivenessofourproposedframeworks,weconductexperimentsontwomodernmedical
datasets and two traditional Chinese medicine datasets, with evaluations by GPT and human
doctors on different methods. The results indicate that our RAG framework can significantly
enhance the diagnostic performance of LLMs, highlighting the potential of our approach in
medical diagnosis. The code and data can be found in our project website1.
1. Introduction
In recent years, medical large language models (LLMs) have emerged as crucial tools in addressing the growing
demands on healthcare systems [17]. With increasing patient numbers and limited medical professionals leading to
longer wait times, LLMs offer effective solutions to enhance healthcare delivery. They assist in diagnosis, treatment
planning, and research, ultimately improving patient outcomes and efficiency in clinical settings. There are generally
two approaches to endowing LLMs with medical knowledge: adjusting model parameters or incorporating medical
knowledge into the prompt [2, 20]. Adjusting model parameters often requires substantial computational resources
[34,3].Retrieval-augmentedgeneration(RAG)isarepresentativeofthelattertechnique[7,40].Typically,RAGfirst
retrieves knowledge from medical knowledge bases based on the query and then incorporates this knowledge into the
prompttogeneratemoreaccurateresponses.Comparedtotheformertechnique,RAGcanintegratethelatestknowledge
morequicklyandeffectively,whichisextremelyimportantinaknowledge-intensiveQ&A(questionanswering)system
like healthcare. On the other hand, existing models can be quickly deployed to different hospitals; each hospital can
build its own knowledge base, and hospitals do not need to hand over their data for training, thus protecting patient
privacy [12].
Existing medical RAG frameworks are designed to accommodate various medical scenarios. According to the
number of dialogue rounds, these frameworks can be classified into single-round and multi-round dialogue RAG
frameworks[36].Mostcurrentresearchfocusesonsingle-rounddialoguescenarios.Insingle-rounddialogues,LLMs
are generally required to provide an answer based on a single question, such as the medical knowledge Q&A. For
mostmedicalquestions,LLMscanretrieverelevantknowledgeinasingleretrievaloperationusingdifferentstrategies
and then generate accurate responses [16, 28, 25, 11]. For complex medical questions requiring multi-hop reasoning,
multiple retrievals may be necessary to obtain the correct answer [29], but these still fall within the scope of single-
round dialogues. Single-round dialogue RAG frameworks underestimate the interactivity of the system and do not
1https://github.com/YixiangCh/MRD-RAG/tree/master
∗Corresponding Author
ychen416@connect.hkust-gz.edu.cn (Y. Chen); psun012@connect.hkust-gz.edu.cn (P. Sun);
xli906@connect.hkust-gz.edu.cn (X. Li); xwchu@hkust-gz.edu.cn (X. Chu)
ORCID(s):0009-0008-0543-2677 (Y. Chen); 0009-0005-2290-0944 (P. Sun); 0009-0002-2879-081X (X. Li); 0000-0001-9745-4372 (X.
Chu)
:Preprint submitted to Elsevier Page 1 of 17

(a) Single -round RAG (c)MRD -RAG (b) Multi -round RAG
LLM…
[N-1thRound]
Patient: I also have
heartburn symptoms…
:Symptoms of
COPD, treatment ofGERD...
Doctor: Have you had any
lung related exams?
(Imprecise inquiry)
[NthRound]
Patient: My lung
auscultation showed...
:Symptoms of
COPD , treatment of GERD...
(Lack of diagnosis
information for GERD)
Doctor: You might have
COPD ...
LLM…
[N-1thRound]
Patient: I also have heartburn
symptoms…
:The diagnostic
differences between COPD and
GERD are... (Considering the
diagnostic dif ferences)
Doctor: Have you had
symptoms like acid reflux ?
(Precise inquiry)
[NthRound]
Patient: I have long -term acid
reflux...
:Acid reflux is
typical clinical manifestations
of GERD...
Doctor: You might have
GERD...End
LLM[First Round]
Patient: I've had a
cough, sputum, and
wheezing for 5
years.
:
Symptoms of
chronic bronchitis,
symptoms ,
treatment of
COPD ... (Incorrect
retrieval)
Doctor: You might
have chronic
bronchitis...Patient with GERD Patient with GERD Patient with GERDFigure 1: Comparison between different RAG frameworks.
utilizecontextualinformationfromhistoricaldialoguestodynamicallyretrievemorepreciseknowledge.Somestudies
have attempted to extend RAG to multi-round medical dialogue scenarios. RagPULSE extracts keywords from the
dialogue history to formulate more precise queries [8]. MedDM propose the LLM-executable clinical guidance tree
(CGT),enablingLLMstoperformmulti-roundreasoningbasedontheCGT,therebygeneratingmorereliablemedical
responses [13]. Multi-round RAG significantly improves the interactivity and retrieval performance of the system.
Figure 1 (a) and (b) give an overall demonstration of single-round RAG and multi-round RAG, respectively.
However, existing multi-round dialogue RAG frameworks do not consider the interconnections between potential
diseasestoenhancetheabilityofLLMstoaskquestionsprecisely,whichisparticularlyimportantinmedicaldiagnostic
scenarios. For instance, in the real-world diagnostic scenario, doctors engage in multi-round dialogues with patients.
They analyze the differences of various potential diseases in their minds and then proactively and precisely inquire
about the patient’s medical information (e.g., asking whether the patient has certain symptoms or has undergone
specific examination), or make a diagnosis based on the existing medical information of the patient. To achieve such
a doctor-like multi-round dialogue RAGframework, there are still some challenges in medical diagnostic scenarios .
Firstly, most current medical knowledge bases do not simultaneously include detailed diagnostic information on both
modern medicine ( MM) and traditional Chinese medicine ( TCM) diseases. Secondly, the patient’s utterances are not
semantically aligned with the knowledge in the knowledge base, resulting in inaccurate retrieval.
To cope with the existing challenges, we propose Multi-Round Diagnostic RAG ( MRD-RAG ), as shown in
Figure 1 (c). First, to address the challenge of the existing knowledge base, we collect diseases from the medical
encyclopedia website1 2. We organize each disease into a Disease Information Tree(DI-Tree) and then construct
one knowledge base containing 746 MM diseases and another containing 130 TCM diseases. This facilitates the
search of detailed diagnostic information and excludes a large amount of diagnosis-unrelated information to assist
LLMinadaptingtothediagnostictask.Second,weconstructtheMRD-RAGpipeline.Specifically,itmainlyconsists
ofthreemodules:retriever,analyzer,anddoctor.Theretrievermoduleisresponsibleforretrievingmultiplecandidate
diseases from the above DI-Tree knowledge base. We propose pseudo medical history index to address the semantic
misalignment between patient’s utterances and disease information in the knowledge base. Mimicking the thinking
processofahumandoctor,theanalyzermodulesummarizestheinterconnectionsanddifferencesofretrievedcandidate
diseases and analyzes the patient’s connection with each disease, while the doctor module responds based on the
dialogue history and the analyzer’s thinking process to continue inquiring the patient or make a diagnosis. Since the
retrieved candidate diseases are very relevant to the patient’s condition, the inquiry is more targeted and precise.
1https://www.yixue.com/
2https://www.dayi.org.cn/
:Preprint submitted to Elsevier Page 2 of 17

To evaluate the effectiveness of our proposed framework, we conduct experiments with existing general LLMs as
wellasmedicalLLMsontwoMMandtwoTCMdatasets.TheevaluationresultsfromGPTindicatethatMRD-RAG
improves the diagnostic performance of LLMs without RAG and with single-round RAG by an average of 9.4%and
6%, respectively. We also ask human doctors to evaluate the diagnostic performance of our framework against LLMs
without RAG and LLMs with single-round RAG. Our framework achieves an average improvement of 21.75%and
18%, respectively, over these two methods.
Our main contributions can be summarized as follows:
1. We construct a MM and a TCM disease knowledge base, containing 746 MM diseases and 130 TCM diseases,
respectively. Each disease is organized into a tree structure, making it easier to search for diagnosis-related
information and remove unrelated information.
2. We propose the MRD-RAG framework that enables multi-round dialogue with patients for diagnosis. This
frameworkcouldanalyzetheinterconnectionsanddifferencesamongmultiplecandidatediseasessoastoinquire
precisely or make accurate diagnosis for the patient like a doctor.
3. The evaluation results from both GPT and human doctors consistently demonstrate that our framework can
effectively enhance the diagnostic performance of different LLMs.
2. Related Work
2.1. Retrieval-Augmented Generation
Due to its low cost and good interpretability, RAG has garnered significant attention in recent years [21, 18].
RAG can be categorized into single-round and multi-round dialogue RAG based on the number of dialogue rounds.
Most existing RAG methods are designed to solve single-round dialogue tasks and can be further categorized into
single-retrieval and multi-retrieval based on the frequency of retrieval. HyKGE [11] improves retrieval accuracy by
reformulatingqueriestosemanticallyalignuserquestionswiththeknowledgebase.ChatDoctor[16]fine-tunesLlama
[23] and integrates external knowledge to answer general medical questions. Single-retrieval RAG methods are more
efficient but lack flexibility when handling complex problems. Multi-retrieval RAG methods employ strategies to
determine if and when retrieval is needed. ToG [22] and StructGPT [10] repeatedly access structured data, with each
access direction determined by the LLM, solving multi-hop questions. IM-RAG [33] enhances the interpretability of
theretrievalprocessbyrepeatedlyqueryingtheLLMinamannersimilarto“innermonologue”andrefiningthequery.
Multi-round dialogue tasks pose new challenges to the RAG framework (e.g., query formulation and coreference
resolution).Thesechallengeshinderdirecttransferofsolutionsfromsingle-rounddialoguetasks.Inthisregard,some
research has been proposed. For instance, RagPULSE [8] utilizes LLMs to summarize keywords in dialogue to form
queries,enhancingthemodel’sabilitytohandlemedicationconsultationtasks.ConvRAG[36]focusesoncoreference
resolutionproblemsduringmulti-rounddialogues.However,theseworksonlyconsiderrespondingtothelastquestion
in a dialogue and do not enhance the LLM’s ability to proactively inquire during multi-round dialogues.
2.2. Medical LLM
Medical LLMs focus on enhancing models’ capabilities across various medical tasks, such as medical common-
sense Q&A, medication consultation, and clinical diagnosis. Many researchers tune model parameters on specific
medical datasets to equip models with medical knowledge, enabling doctor-like performance in clinical tasks.
DISC-MedLLM [3] and DoctorGLM [30] construct high-quality medical multi-round dialogue datasets and perform
supervised fine-tuning on LLMs. HuatuoGPT-II [5] combines continued pre-training and supervised fine-tuning
stages into a single process so as to solve the problem of inconsistent data distribution in medical LLM training.
Baichuan-M1 is trained from scratch, focusing on improving medical capabilities [24]. Furthermore, some studies
apply RAG to medical LLMs to enhance reliability. Self-BioRAG [9] introduces Self-RAG [1] into the biomedical
field to dynamically decide whether retrieval is needed. DrHouse [32] utilizes sensor data to improve the diagnostic
performance.MedDM[13]proposes“clinicalguidancetree”(CGT)andlettheLLMreasononit.Unliketheseworks,
our framework analyzes the interconnections and differences of retrieved potential diseases that patients may suffer,
thereby enhancing the performance of LLMs in proactive inquiry and diagnosis.
:Preprint submitted to Elsevier Page 3 of 17

<h1>Lung Cancer</h1>
<h2>Symptom</h2>
<h3>E arly Symptom </h3>
<p>The common symptoms of lung cancer in the early
stage are cough, low fever , chest pain, tightness and so
on.</p>
<h3>Late symptom</h3>
<p>Face, neck edema, hoarseness, etc .</p>
<h2>T reatment</h2>
<h3>Chemotherapy</h3>
<p>Ef fect of chemotherapy on cell...</p>
<h3>Surgical Treatment</h3>
<p>Sur gery is the main treatment...</p>
HTML Disease Information T reeDiagnosis
Information Subtr ee
Indexing
Lung Cancer
Diagnosis
Information
Pseudo Medical
HistoryLung
Cancer
Treatment Symptom
Early
Symptom
The common
symptoms of
lung cancer ...Late
Symptom
Face, neck
edema,
hoarseness,
etc.Chemoth -
erapy
Effect of
chemother -
apy on cell...Surgical
Treatment
Effect of
chemother -
apy on cell...
Figure 2: DI-Tree knowledge base construction process.
3. Preliminary
Inthiswork,wefocusontwokindsofmedicalscenarios:modernmedical(MM)diagnosisandtraditionalChinese
medicine (TCM) syndrome differentiation. Similar to modern medical diagnosis, TCM syndrome differentiation
involvesidentifyingpatients’syndromebasedontheirmedicalcharacteristics,whichwecollectivelyrefertoas disease
diagnosis for clarity in the following discussion.
Theworldhealthorganization(WHO)classifiesdiseasesbasedontheirmedicalcharacteristicsandorganizesthem
hierarchicallyaccordingtospecificrules[27].ThisworkdoesnotaimforanLLMtodeterminethemostgranularlevel
ofapatient’sdiseasesolelythroughdialogue.Forexample,apatientwith “malignantneoplasmoflowerlobe,bronchus
orlung”hasadiseasewhosehigher-levelcategoryis “malignantneoplasmofbronchusandlung” .Ifitispossibleto
identifyacloselyrelatedhigher-levelcategoryofthediseasethatthepatientmayhave,thisshouldalsobeconsidered
avaliddiagnosis.Therefore,forapatientwithdisease 𝑑𝑖,ourtaskistousetheLLMtodiagnosethedisease 𝑑𝑖or𝑑𝑖′
(where 𝑑𝑖′isacloselyrelatedhigher-levelcategoryoraliasof 𝑑𝑖).Thisassistspatientsinquicklyunderstandingtheir
health status to make informed decisions moving forward.
Forapatient, 𝐼𝑝𝑎𝑡𝑖𝑒𝑛𝑡representstheirinherentmedicalinformation.Indiagnosticscenarios,amedicalLLMshould
be capable of engaging in multi-round dialogues with the patient. Let 𝑋𝑛−1= {𝑥1
𝑝𝑎𝑡𝑖𝑒𝑛𝑡, 𝑥1
𝑑𝑜𝑐𝑡𝑜𝑟,…, 𝑥𝑛−1
𝑝𝑎𝑡𝑖𝑒𝑛𝑡, 𝑥𝑛−1
𝑑𝑜𝑐𝑡𝑜𝑟}
denote the dialogue history up to round 𝑛− 1. The patient will converse with the doctor based on their medical
informationanddialoguehistory,representedas 𝑥𝑛
𝑝𝑎𝑡𝑖𝑒𝑛𝑡=𝑃 𝑎𝑡𝑖𝑒𝑛𝑡 (𝑋𝑛−1, 𝐼𝑝𝑎𝑡𝑖𝑒𝑛𝑡 ).Andthedialoguehistoryisupdated
as̃𝑋𝑛= {𝑥1
𝑝𝑎𝑡𝑖𝑒𝑛𝑡, 𝑥1
𝑑𝑜𝑐𝑡𝑜𝑟,…, 𝑥𝑛−1
𝑝𝑎𝑡𝑖𝑒𝑛𝑡, 𝑥𝑛−1
𝑑𝑜𝑐𝑡𝑜𝑟, 𝑥𝑛
𝑝𝑎𝑡𝑖𝑒𝑛𝑡}. Due to the patient’s lack of medical knowledge and clear
awarenessoftheirowncondition, 𝑥𝑛
𝑝𝑎𝑡𝑖𝑒𝑛𝑡maybevagueandcouldnotincludeallthekeypointsof 𝐼𝑝𝑎𝑡𝑖𝑒𝑛𝑡.Thetaskof
themedicalLLMistorespondtothepatientbasedonthedialoguehistory: 𝑥𝑛
𝑑𝑜𝑐𝑡𝑜𝑟=𝐿𝐿𝑀 (̃𝑋𝑛).IntheRAGsettings,
the LLM can also utilize external knowledge 𝐾and respond: 𝑥𝑛
𝑑𝑜𝑐𝑡𝑜𝑟=𝐿𝐿𝑀 (̃𝑋𝑛, 𝐾). By incorporating external
knowledge, the LLM can give a more appropriate response.
4. Method
To equip MRD-RAG with diagnostic knowledge, we first build two knowledge bases centered on MM and TCM
diseases.Subsequently,MRD-RAGisabletoengageinmulti-roundinteractionsandproactivelyinquirelikeadoctor.
It first retrieves multiple potential candidate diseases based on the patient’s utterance, then analyzes the diagnosis
information of these diseases, and finally makes appropriate responses.
4.1. DI-Tree Knowledge Base
4.1.1. Data Collection
The diagnostic task requires identifying the disease the patient suffers from, so the LLM needs to understand the
characteristics of various diseases (such as common symptoms, pathogenesis, etc.). Existing public Chinese medical
knowledge bases mainly include well-structured medical knowledge graphs and unstructured medical encyclopedia
:Preprint submitted to Elsevier Page 4 of 17

Table 1
Comparison of medical knowledge bases.
Knowledge Base MM Disease TCM Disease Domain Details of Diagnosis
StrokeKG [35] ! % Specialist %
KGHC [15] ! % Specialist %
CGT [13] ! % Generalist !
CMeKG [4] ! % Generalist !
RD-MKG [14] ! % Generalist %
DKD-TCMKG [39] % ! Specialist !
WZQ-TCMKG [6] ! ! Generalist %
ETCM2.0 [38] ! ! Generalist %
DI-Tree (Ours) ! ! Generalist !
websites1 2. However, existing medical knowledge graphs, while clearly describing the relationships among entities
(e.g., diseases, complications, medications, etc.), usually lack detailed diagnosis-related information about individual
diseases. In addition, the process of building knowledge graphs usually involves multiple steps such as information
extractionfromunstructuredtextandknowledgefusion,whichmayintroduceissuessuchasincorrectentityrecognition
and improper disambiguation. Thus, existing medical KGs are unsuitable for direct use in diagnostic tasks. Existing
medical encyclopedia websites are mostly written manually by medical experts, ensuring high professionalism and
accuracy.Therefore,wecollect746MMdiseaseentriesand130TCMdiseaseentriesfromtwomedicalencyclopedia
websites1 2as our data sources.
4.1.2. Disease Information Tree
Due to the varying preferences of different editors on the medical encyclopedia website, there are significant
differences in the level of structuring and the length of text when describing information about different diseases.
Additionally, there is a lot of redundant information. Therefore, it is not feasible to directly provide all disease
information to the LLM, and these differences need to be pre-emptively masked. Based on heuristic methods, we
observe that disease encyclopedia pages are generally organized based on different characteristics of the disease in
theirHTMLpages,asshowninFigure2.Basedonthisobservation,weprocesseachdiseaseindividually,organizing
them into a tree structure, namely disease information tree ( DI-Tree). Unlike common medical knowledge graphs,
the tree structure of disease information has a clearer hierarchical structure, making it easier to locate the necessary
diagnosis-related information (i.e. sub-trees). In the DI-Tree, HTML headings at various levels are represented as
intermediate nodes, and specific content is represented as leaf nodes.
We compare our knowledge base with the existing medical knowledge base, as shown in Table 1. Most existing
knowledge bases do not simultaneously include both MM diseases and TCM diseases. Among these, the clinical
guidancetree(CGT)inMedDM[13]alsousestree-structuretorepresentitsknowledge.ButdifferentfromtheCGT,our
DI-Treeisnotadecisiontreefordirectreasoning,it’sdesignedtoextractdiagnosis-relatedinformation.Additionally,
sincewefocusondiagnostictasks,wepaymoreattentiontowhethertheknowledgebasecontainsdetaileddiagnostic
information for each disease (i.e., the “Details of Diagnosis” column). Detailed diagnostic information typically
includes symptoms, clinical manifestations, diagnosis, medical examinations, and other relevant information about
the disease. This aids in equipping LLMs with professional medical knowledge to enhance diagnostic performance.
Although WZQ-TCMKG [6] and ETCM2.0 [38] include both MM diseases and TCM diseases, they do not contain
details of diagnosis, making them unsuitable for our diagnostic tasks. Our DI-Tree knowledge base provides detailed
information for diagnosing both MM and TCM diseases.
4.1.3. Indexing
After obtaining the DI-Tree, dense vector index can be constructed for subsequent retrieval. Since we focus on
the diagnostic task, we only extract diagnosis-related information from the DI-Tree, while information unrelated to
diagnosis (such as treatment) will not be used for index construction. Specifically, we predefine a keyword search
list (e.g., “symptoms”, “diagnosis”, “medical examination”), and if a keyword is present in an intermediate node, the
subtree rooted at that node is retained ( diagnosis information subtree ). The information of the retained subtree will
bereorganizedasthe diagnosis information text𝑡𝑖
𝐷𝐼forconstructingthedensevectorindex.Foreachdisease 𝑑𝑖,we
useatextembeddingmodeltocomputeitsrepresentation: 𝑟𝑖
𝐷𝐼=𝐸𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 (𝑡𝑖
𝐷𝐼).However,whenpatientsdescribe
:Preprint submitted to Elsevier Page 5 of 17

𝑥𝑝𝑎𝑡𝑖𝑒𝑛𝑡1
𝑥𝑝𝑎𝑡𝑖𝑒𝑛𝑡2
𝑥𝑝𝑎𝑡𝑖𝑒𝑛𝑡𝑛. . .Retriever
QueryCandidate
Diseases
Analyzer
LLMAnalyzerDifferences between diseases , tests
that patients should under go...
Retrieve
DoctorAsk
questionsDiagnoseDialogue History
LLMDoctorPatientConcat allThe nthroundFigure 3: The pipeline of MRD-RAG.
their own information, their utterances may be unclear. On the other hand, 𝑡𝑖
𝐷𝐼is sometimes too specialized. So there
may be a significant semantic misalignment between the patient’s utterances and the corresponding 𝑡𝑖
𝐷𝐼. To mitigate
this misalignment,we also use theLLMto generate pseudo medical history text𝑡𝑖
𝑃 𝑀𝐻based on 𝑡𝑖
𝐷𝐼, describingthe
medical information of a patient with the disease. The prompt for generating 𝑡𝑖
𝑃 𝑀𝐻can be found in Supplementary
Materials. We then compute its representation 𝑟𝑖
𝑃 𝑀𝐻=𝐸𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 (𝑡𝑖
𝑃 𝑀𝐻). Compared to the diagnosis information
text𝑡𝑖
𝐷𝐼,𝑡𝑖
𝑃 𝑀𝐻is more semantically similar to the patient’s utterances, facilitating the retrieval. Therefore, we also
use𝑟𝑖
𝑃 𝑀𝐻to represent 𝑑𝑖. The indexes constructed from 𝑟𝑖
𝐷𝐼and𝑟𝑖
𝑃 𝑀𝐻are denoted as 𝐼𝑛𝑑𝑒𝑥𝐷𝐼and𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻,
respectively, and both are retained.
4.2. Multi-round Diagnostic RAG Framework
AsshowninFigure3,MRD-RAGcomprisesthreemainmodules: Retriever ,Analyzer ,and Doctor.Theretriever
module is responsible for retrieving multiple candidate diseases the patient may have. The analyzer module further
processes the retrieved candidate disease information, and the doctor module poses further questions or make a
diagnosis. Both analyzer and doctor modules are played by the LLM.
4.2.1. Retriever Module
The retriever accesses the DI-Tree knowledge base to retrieve information of candidate diseases. However, we
observe that the utterances of patients in certain rounds might not contribute additional diagnostic information
comparedtotheirpreviousdialoguehistory.Therefore,beforeproceedingwithformalretrieval,weleveragetheLLM
to assess whether retrieval is necessary. The prompt for the LLM is shown in Supplementary Materials. If retrieval is
needed, the process follows the standard retriever-analyzer pathway; otherwise, the doctor module directly generates
a response. This reduces the computational overhead of invoking the retriever and analyzer.
Because of the insufficient information provided by the patient at the beginning of the dialogue, the doctor may
provide inaccurate diagnostic inferences during the initial dialogue, and a query based on the doctor’s words would
matchmanyunrelateddiseases.Thepatient’sutterancesoftencontainmorerelevantinformation(e.g.,symptoms)and
aredirectlyusedtoretrievecandidatediseases.Hence,toavoidusinginaccuratediagnosticinferencesduringretrieval,
MRD-RAG uses only the patient’s utterances as the query. For the 𝑛𝑡ℎround of dialogue, all the patient’s previous
utterancesareconcatenatedtoformthequery: 𝑞=𝑐𝑎𝑡(𝑥1
𝑝𝑎𝑡𝑖𝑒𝑛𝑡,…, 𝑥𝑛
𝑝𝑎𝑡𝑖𝑒𝑛𝑡),where 𝑐𝑎𝑡denotesthestringconcatenation
function. Then, a text embedding model is used to calculate the representation of 𝑞:𝑟𝑞=𝐸𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 (𝑞). To obtain
thecandidatediseasesthepatientmayhave,retrievalcanbeperformedusing 𝐼𝑛𝑑𝑒𝑥𝐷𝐼or𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻asmentionedin
4.1.3.Specifically,bycalculatingthecosinesimilarityscorebetween 𝑟𝑞and𝑟𝑖
𝑃 𝑀𝐻or𝑟𝑖
𝐷𝐼,the𝑡𝑜𝑝-𝑘candidatediseases
𝑑𝑖with high scores are obtained. Similar to the indexing phase, MRD-RAG utilizes the DI-Tree of 𝑑𝑖to obtain the
diagnosis information texts 𝑡𝑖
𝐷𝐼as the output knowledge 𝐾of the retriever.
:Preprint submitted to Elsevier Page 6 of 17

Table 2
Staticstics for the test dataset: the number of cases, diseases and intersections with our DI-Tree knowledge base.
Dataset Case Disease Intersection
CMB-Clin 74 74 66
MM-Cases 609 609 609
TCM-SD-100 100 35 33
TCM-Cases 130 130 130
4.2.2. Analyzer Module
Thetextlengthofmultiplecandidatediseasescanbeexcessivelylong,makingitchallengingforthe 𝐿𝐿𝑀𝐷𝑜𝑐𝑡𝑜𝑟to
simultaneously comprehend the long text and dialogue to respond to the patient. Additionally, candidate diseases are
often similar, with certain interconnections and differences, and identifying these interconnections and differences is
crucialfordiagnosis.Inreal-worldscenarios,doctorstypicallycontemplatevariouspotentialdiseasesbeforereaching
a diagnosis conclusion. Similarly, before 𝐿𝐿𝑀𝐷𝑜𝑐𝑡𝑜𝑟responds to the patient, the analyzer module can first consider
andsummarizemultiplecandidatediseasesinrelationtothepatient’scondition,whichmimicsthethinkingprocessof
thedoctor.Thisincludesidentifyingdifferencesandinterconnectionsamongthediseases,consideringtherelationship
betweenthecurrentpatientandeachcandidatedisease,andthinkingabouthowtoinquireordiagnosethepatient,etc.
The thinking process of the analyzer module serves as more precise and critical external knowledge for the doctor
module: 𝐾𝑡ℎ𝑖𝑛𝑘𝑖𝑛𝑔 =𝐿𝐿𝑀𝐴𝑛𝑎𝑙𝑦𝑧𝑒𝑟 (̃𝑋𝑛, 𝐾). The prompt for the analyzer module can be found in Supplementary
Materials.
4.2.3. Doctor Module
Aftergettingthethinkingprocessfromtheanalyzerregardingcandidatediseases,thedoctorfocusesonproviding
a patient-friendly response: 𝑥𝑛
𝑑𝑜𝑐𝑡𝑜𝑟=𝐿𝐿𝑀𝐷𝑜𝑐𝑡𝑜𝑟 (̃𝑋𝑛, 𝐾𝑡ℎ𝑖𝑛𝑘𝑖𝑛𝑔 ). It will continue to inquire the patient or make a
diagnosisconclusion.Typically,therearetwosituations:ifsufficientmedicalinformationaboutthepatienthasalready
been obtained in the current dialogue, a diagnosis result can be provided. Otherwise, the doctor will pose questions
based on 𝐾𝑡ℎ𝑖𝑛𝑘𝑖𝑛𝑔to efficiently gather more medical information from the patient for subsequent decision-making.
The prompt for the doctor module can be found in Supplementary Materials.
5. Experiments
5.1. Experimental Setup
5.1.1. Evaluation Dataset
Weutilizefourdatasetsforevaluation,includingtwopubliclyavailabledatasetsandtwoconstructedbyus.CMB-
Clin [26] is one of the publicly available subsets of a Chinese medical benchmark, containing 74 complex modern
medicalclinicaldiagnosiscases.TCM-SD[19]isthefirstpublicbenchmarkforTCMsyndromedifferentiation,andit
normalizes all TCM syndrome names. From its test set, we randomly select 100 cases (TCM-SD-100) for evaluation.
For MM-Cases and TCM-Cases, we utilize GPT-4o-mini3in combination with collected disease information to
generatepatientcasesforcorrespondingdiseases,manuallyfilteringoutdiseasesthatevidentlydonotrequiredialogue
diagnosis(e.g.,shock,tonguebite).Thesecasesarethenevaluatedbydoctors,whoscreenoutcaseswherethedisease
doesnotmatchthepatient’smedicalinformation,resultingin609synthesizedMMpatientcasesand130synthesized
TCM patient cases. Each case in all datasets contains patient medical information and the name of the disease. Some
diseasesinthepublicdatasetsdonotexistinourknowledgebase.Thenumberofcases,diseases,andtheintersection
of diseases between the datasets and the knowledge bases are summarized in Table 2.
5.1.2. Implementation Details
During the multi-round dialogue, we consistently employ GPT-4o-mini to play the role of the patient and engage
in dialogue with different methods. As described in Section 3, in each round of dialogue, the patient converses based
ontheirmedicalinformation 𝐼𝑝𝑎𝑡𝑖𝑒𝑛𝑡andthedialoguehistory.Tosimulatethereal-worldpatientconsultationscenario,
we encourage the patient to avoid directly revealing their real disease name and to refrain from disclosing excessive
3https://platform.openai.com/docs/guides/text-generation
:Preprint submitted to Elsevier Page 7 of 17

information in a single utterance. As for the embedding model, we uniformly use text-embedding-3-small4from
OpenAI in the indexing and retrieval stages, which is a text embedding model for a general domain. We retrieve 5
diseasesfromtheDI-Treeknowledgebaseatatime( 𝑡𝑜𝑝-𝑘= 5).ConsideringthatMRD-RAGdoesnotrequiretraining,
weusethreegeneralLLMsandonemedicalLLMasthebasemodels:Qwen2-7B-Instruct,Qwen2-72B-Instruct-AWQ
[31],GPT-4o-miniandBaichuan-M1-14B-Instruct[24].Foranalyzeranddoctormodulesofthesameexperiment,they
sharethesameLLM.Additionally,themaximumnumberofdialogueroundsissetto3[3].Inthefollowingdiscussion,
MRD-RAG with 𝐼𝑛𝑑𝑒𝑥𝐷𝐼and𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻will be denoted as MRD-RAG-DI and MRD-RAG-PMH, respectively.
5.1.3. Baseline
We select the following four groups of methods as baselines:
•General LLMs: Thisgroupincludestwoopen-sourcegeneralChineseLLMs:Qwen2-7B-InstructandQwen2-
72B-Instruct-AWQ, as well as two popular closed-source LLMs: GPT-4o-mini and GPT-4o.
•Medical Domain LLMs: We choose DISC-MedLLM [3], HuatuoGPT-II-7B [5] and Baichuan-M1-14B-
Instruct [24] for comparison. Both of them have been trained on different specialized medical datasets.
•Single-Round RAG: To validate the effectiveness of MRD-RAG, we also compare it with single-round RAG.
Specifically,theLLMonlyutilizetheretrievedknowledgeduringthefirstroundofdialogue.Forfaircomparison,
wealsoselectQwen2-7B-Instruct,Qwen2-72B-Instruct-AWQ,GPT-4o-miniandBaichuan-M1-14B-Instructas
base models.
•RagPULSE: RagPULSE[8]utilizesLLMstosummarizekeywordsindialoguetoformqueriesforretrieval.It
also falls under the category of multi-round RAG, so we include it in the baseline.
5.1.4. Metric
Allmethodsdialoguewiththepatient.Subsequently,weestablishdifferentevaluationmetricstocomparedifferent
methods in the dialogues.
•GPT Evaluation Metric: Some studies indicate that powerful LLMs can achieve high alignment with human
doctors’judgments[3,34,37].WeemployGPT-4o-minitosimulatehumandoctorsforevaluation.Specifically,
we provide the patient information 𝐼𝑝𝑎𝑡𝑖𝑒𝑛𝑡and the corresponding dialogues of different methods in the prompt,
allowingGPT-4o-minitoscorethedifferentmethods(1-5points).Tomitigatetheimpactofthevaryingpositions
of dialogues in the prompt on the scores of each method, we randomize the positions of dialogues for each
method.Wefocusonevaluatingthe“DiagnosisAccuracy”oftheLLMs.“DiagnosisAccuracy”assesseswhether
the model can accurately diagnose the patient’s disease. The prompt is shown in Supplementary Materials.
•Human Doctor Evaluation Metric: ConsideringthatGPT-4o-minimaylackexpertiseinthemedicaldomain,
as referenced in [37, 5], we invite human doctors (including practitioners of modern medicine and TCM)
to perform pairwise comparisons of the quality of multi-round dialogues generated by different methods. In
each evaluation case, we provide doctors with the patient’s medical information and the multi-round dialogues
generatedbytwodifferentmethods(e.g.ourmethodandbaselinemethod)forthatpatient.Theirtaskistoselect
the model that generates the better response.
•Text Generation Evaluation Metric: To obtain more objective evaluation results, we also utilize BLEU,
ROUGE and METEOR as evaluation metrics. These 3 text generation evaluation metrics require ground truth
reference texts for comparison. In our experimental setup, the patient would mention their medical information
𝐼𝑝𝑎𝑡𝑖𝑒𝑛𝑡as little as possible. A good doctor should be able to deduce the patient’s medical information through
inquiries or inferences. Therefore, we select the dialogue as the candidate text and 𝐼𝑝𝑎𝑡𝑖𝑒𝑛𝑡as the reference text
to compute these scores. A better doctor could get a higher score.
5.2. Experimental Results
5.2.1. Analysis of GPT Evaluation Metric Result
The results of GPT’s evaluation for each method are shown in Table 3. Most of the general LLM performs well,
withQwen2-72B-Instruct-AWQoutperformingallmethodswithoutRAG.TheexistingmedicalLLMsunderperform,
4https://platform.openai.com/docs/guides/embeddings
:Preprint submitted to Elsevier Page 8 of 17

Table 3
GPT scores for each method and model (1-5 points).
Method Model CMB MM-Cases TCM-SD-100 TCM-Cases Average
General LLMQwen2-7B-Instruct 3.12 2.85 3.20 3.32 3.12
Qwen2-72B-Instruct-AWQ 3.24 2.93 3.26 3.69 3.28
GPT-4o-mini 2.91 2.71 3.07 3.24 2.98
GPT-4o 2.99 2.85 3.39 3.42 3.16
Medical LLMDISC-MedLLM 2.74 2.49 2.22 2.33 2.45
HuatuoGPT-II-7B 2.82 2.69 2.35 2.86 2.68
Baichuan-M1-14B 2.85 2.73 2.75 3.13 2.86
Single-Round RAGQwen2-7B-Instruct 3.15 2.96 3.03 3.90 3.26
Qwen2-72B-Instruct-AWQ 3.31 3.27 3.31 4.21 3.52
GPT-4o-mini 3.12 2.92 3.20 3.81 3.26
Baichuan-M1-14B 2.70 2.68 2.49 3.59 2.87
RagPULSE PULSE-20B 2.77 2.65 2.32 2.55 2.57
MRD-RAG-DIQwen2-7B-Instruct 3.23 3.23 3.14 3.98 3.39
Qwen2-72B-Instruct-AWQ 3.26 3.35 3.19 3.99 3.45
GPT-4o-mini 3.14 3.24 3.19 4.12 3.42
Baichuan-M1-14B 3.18 3.22 2.99 3.75 3.28
MRD-RAG-PMHQwen2-7B-Instruct 3.59 3.83 3.26 4.36 3.76
Qwen2-72B-Instruct-AWQ 3.62 3.77 3.09 4.43 3.73
GPT-4o-mini 3.53 3.75 3.23 4.42 3.73
Baichuan-M1-14B 3.26 3.82 2.82 4.02 3.48
/uni00000013 /uni00000015/uni00000013 /uni00000017/uni00000013 /uni00000019/uni00000013 /uni0000001b/uni00000013 /uni00000014/uni00000013/uni00000013/uni00000036/uni0000004c/uni00000051/uni0000004a/uni0000004f/uni00000048/uni00000010/uni00000035/uni00000052/uni00000058/uni00000051/uni00000047/uni00000003/uni00000035/uni00000024/uni0000002a/uni0000003a/uni0000004c/uni00000057/uni0000004b/uni00000052/uni00000058/uni00000057/uni00000003/uni00000035/uni00000024/uni0000002a
/uni00000018/uni00000014/uni00000011/uni00000018/uni00000018/uni0000001c/uni00000011/uni00000013
/uni00000014/uni0000001b/uni00000011/uni00000013/uni0000001c/uni00000011/uni00000018
/uni00000016/uni00000013/uni00000011/uni00000018/uni00000016/uni00000014/uni00000011/uni00000018/uni00000030/uni00000030/uni00000003/uni00000027/uni00000044/uni00000057/uni00000044/uni00000032/uni00000058/uni00000055/uni00000056/uni00000003/uni0000003a/uni0000004c/uni00000051 /uni00000037/uni0000004c/uni00000048 /uni00000032/uni00000058/uni00000055/uni00000056/uni00000003/uni0000002f/uni00000052/uni00000056/uni00000056
/uni00000013 /uni00000015/uni00000013 /uni00000017/uni00000013 /uni00000019/uni00000013 /uni0000001b/uni00000013 /uni00000014/uni00000013/uni00000013/uni00000036/uni0000004c/uni00000051/uni0000004a/uni0000004f/uni00000048/uni00000010/uni00000035/uni00000052/uni00000058/uni00000051/uni00000047/uni00000003/uni00000035/uni00000024/uni0000002a/uni0000003a/uni0000004c/uni00000057/uni0000004b/uni00000052/uni00000058/uni00000057/uni00000003/uni00000035/uni00000024/uni0000002a
/uni00000016/uni00000015/uni00000011/uni00000013/uni00000018/uni00000014/uni00000011/uni00000013
/uni00000018/uni00000014/uni00000011/uni00000013/uni00000014/uni00000017/uni00000011/uni00000013
/uni00000014/uni0000001a/uni00000011/uni00000013/uni00000016/uni00000018/uni00000011/uni00000013/uni00000037/uni00000026/uni00000030/uni00000003/uni00000027/uni00000044/uni00000057/uni00000044
Figure 4: Comparison and evaluation results of different methods by human doctors.
mainlyduetothefactthatmostoftheirfine-tuneddatasetscomefromcommonsensemedicaltasksandlackdiagnostic
reasoning tasks. In addition, scores of single-round RAG methods get improved relative to methods without RAG in
general. However, for RagPULSE, since it’s specifically designed for multi-round medication consultation tasks, it
doesn’tperformwell.ComparingMRD-RAGwiththecorrespondingbasemodelwithoutRAGandwithsingle-round
RAG, its score improves by an average of 0.47 ( ↑9.4%) and 0.30 ( ↑6%), respectively, which shows that DI-Tree
knowledge bases can provide assistance in the diagnostic task. When comparing MRD-RAG-PMH and MRD-RAG-
DI, it is found that GPT-4o-mini prefers the former more. This is due to the fact that 𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻can improve the
retrieval accuracy effectively compared to 𝐼𝑛𝑑𝑒𝑥𝐷𝐼, allowing LLM to receive more accurate information about the
candidate diseases, thus indirectly improving the diagnostic performance of LLM.
5.2.2. Analysis of Human Doctor Evaluation Metric Result
To reduce costs, we randomly select 100 dialogues from MM data (CMB and MM-Cases) and another 100 from
TCM data (TCM-SD-100 and TCM-Cases) for evaluation by two MM and two TCM doctors. Specifically, doctors
need to compare MRD-RAG-PMH with the general LLM without RAG and the single-round RAG method (each
method use Qwen2-7B-Instruct as the base model). The results are presented in Figure 4. Regardless of whether the
data is from MM or TCM, doctors tend to believe that our method provide more accurate diagnosis compared to the
baselinemethods.Forexample,ontheMMdata,ourmethodperformssimilarlytotheLLMwithoutRAGin9.5cases
on average, while our method outperforms in 59 other cases on average, and the LLM without RAG performs better
:Preprint submitted to Elsevier Page 9 of 17

Table 4
Text generation evaluation metrics for CMB.
Method Model BLEU-1 BLEU-2 ROUGE-1 ROUGE-2 METEOR
General LLMQwen2-7B-Instruct 18.24 8.98 17.53 5.10 15.98
Qwen2-72B-Instruct-AWQ 18.02 8.65 17.86 5.04 15.95
GPT-4o-mini 20.13 10.13 17.49 5.35 15.13
GPT-4o 19.82 9.74 17.94 5.03 15.75
Medical LLMDISC-MedLLM 18.41 8.95 16.31 5.23 14.00
HuatuoGPT-II-7B 18.71 9.00 17.05 4.71 15.02
Baichuan-M1-14B 14.33 7.47 18.56 6.60 11.52
Single-Round RAGQwen2-7B-Instruct 17.71 8.53 18.43 4.82 18.23
Qwen2-72B-Instruct-AWQ 17.21 8.59 19.39 5.56 18.71
GPT-4o-mini 20.78 10.50 19.50 5.70 16.83
Baichuan-M1-14B 12.60 6.56 18.13 6.13 10.82
RagPULSE PULSE-20B 17.46 8.65 19.06 6.70 12.89
MRD-RAG-DIQwen2-7B-Instruct 18.38 8.86 19.39 5.11 18.38
Qwen2-72B-Instruct-AWQ 18.02 8.78 18.44 5.12 16.81
GPT-4o-mini 20.80 10.10 21.35 5.76 19.48
Baichuan-M1-14B 22.05 11.26 21.56 6.98 15.68
MRD-RAG-PMHQwen2-7B-Instruct 18.14 8.84 19.65 5.20 18.20
Qwen2-72B-Instruct-AWQ 17.73 8.58 18.56 5.08 16.47
GPT-4o-mini 21.13 10.48 21.96 6.06 20.08
Baichuan-M1-14B 21.13 10.95 21.04 7.18 15.13
Table 5
Text generation evaluation metrics for MM-Cases.
Method Model BLEU-1 BLEU-2 ROUGE-1 ROUGE-2 METEOR
General LLMQwen2-7B-Instruct 26.38 14.51 27.73 8.85 22.09
Qwen2-72B-Instruct-AWQ 24.90 13.81 28.20 8.90 22.20
GPT-4o-mini 28.63 16.07 27.67 8.79 21.63
GPT-4o 27.26 15.30 28.22 8.56 22.79
Medical LLMDISC-MedLLM 27.64 14.34 27.45 8.65 19.56
HuatuoGPT-II-7B 25.77 13.81 26.04 7.69 20.29
Baichuan-M1-14B 20.93 11.44 26.72 9.39 15.16
Single-Round RAGQwen2-7B-Instruct 22.75 13.06 28.62 8.76 25.74
Qwen2-72B-Instruct-AWQ 22.24 12.96 29.56 9.32 24.97
GPT-4o-mini 28.28 16.14 29.91 9.23 23.61
Baichuan-M1-14B 18.42 10.20 26.52 9.01 14.61
RagPULSE PULSE-20B 25.06 13.41 28.84 10.04 17.41
MRD-RAG-DIQwen2-7B-Instruct 24.43 13.93 31.25 9.32 26.10
Qwen2-72B-Instruct-AWQ 23.94 13.28 29.94 8.59 23.88
GPT-4o-mini 27.26 16.15 32.27 9.88 27.52
Baichuan-M1-14B 30.16 16.90 32.02 10.93 20.84
MRD-RAG-PMHQwen2-7B-Instruct 25.04 14.40 31.94 9.65 26.60
Qwen2-72B-Instruct-AWQ 24.62 13.77 31.04 9.05 24.48
GPT-4o-mini 27.75 16.65 33.48 10.54 28.31
Baichuan-M1-14B 30.59 17.43 33.09 11.46 21.99
on an average of only 31.5 cases. Overall, in 100 cases, the average win ratio of our method compared to the LLM
without RAG method is 55:33.25 ( ↑21.75%), and compared to the single-round RAG method, the average win ratio
is 41.75:23.75 ( ↑18%). The evaluation results from human doctors are consistent with the previous GPT evaluation
conducted by GPT-4o-mini.
5.2.3. Analysis of Text Generation Evaluation Metrics Result
BLEU, ROUGE, and METEOR scores are shown from Table 4 to Table 7. The evaluation results are not always
consistentwiththoseofothermetrics.ThesemetricscannotadequatelyconsidertheexpertiseofLLMsinthemedical
:Preprint submitted to Elsevier Page 10 of 17

Table 6
Text generation evaluation metrics for TCM-SD-100.
Method Model BLEU-1 BLEU-2 ROUGE-1 ROUGE-2 METEOR
General LLMQwen2-7B-Instruct 14.40 6.21 13.45 4.03 16.84
Qwen2-72B-Instruct-AWQ 15.11 6.55 13.63 4.24 16.83
GPT-4o-mini 15.59 6.72 13.78 4.03 16.66
GPT-4o 16.41 7.29 13.97 4.55 16.92
Medical LLMDISC-MedLLM 17.59 7.81 14.71 5.03 16.41
HuatuoGPT-II-7B 14.57 6.45 14.39 4.22 14.89
Baichuan-M1-14B 17.60 8.70 16.13 5.99 12.20
Single-Round RAGQwen2-7B-Instruct 12.74 5.68 14.09 3.91 17.01
Qwen2-72B-Instruct-AWQ 14.84 6.83 14.76 4.69 17.62
GPT-4o-mini 15.93 6.82 14.10 3.99 16.68
Baichuan-M1-14B 14.57 7.65 16.53 6.45 11.50
RagPULSE PULSE-20B 17.88 9.40 18.47 8.03 13.38
MRD-RAG-DIQwen2-7B-Instruct 14.43 6.70 15.12 4.75 17.79
Qwen2-72B-Instruct-AWQ 13.34 6.07 14.64 4.30 17.33
GPT-4o-mini 13.90 6.26 14.27 4.03 17.09
Baichuan-M1-14B 19.93 10.14 17.77 7.37 15.59
MRD-RAG-PMHQwen2-7B-Instruct 14.06 6.48 14.69 4.65 17.62
Qwen2-72B-Instruct-AWQ 13.51 6.32 14.47 4.53 17.60
GPT-4o-mini 14.26 6.64 14.75 4.39 17.51
Baichuan-M1-14B 19.76 10.08 17.31 7.15 15.77
Table 7
Text generation evaluation metrics for TCM-Cases.
Method Model BLEU-1 BLEU-2 ROUGE-1 ROUGE-2 METEOR
General LLMQwen2-7B-Instruct 26.94 15.99 28.56 11.25 23.77
Qwen2-72B-Instruct-AWQ 24.52 14.90 27.49 10.74 24.20
GPT-4o-mini 24.73 15.19 26.18 10.01 25.88
GPT-4o 27.58 16.85 27.41 10.76 24.81
Medical LLMDISC-MedLLM 28.77 16.37 28.56 10.98 21.12
HuatuoGPT-II-7B 25.03 14.12 25.91 9.07 22.88
Baichuan-M1-14B 23.99 14.54 29.27 12.87 17.25
Single-Round RAGQwen2-7B-Instruct 24.19 14.75 29.66 10.77 28.32
Qwen2-72B-Instruct-AWQ 25.22 15.55 29.42 11.32 27.29
GPT-4o-mini 25.44 15.30 28.01 9.97 27.18
Baichuan-M1-14B 22.92 14.10 29.51 12.99 17.80
RagPULSE PULSE-20B 19.03 11.66 28.34 13.06 15.80
MRD-RAG-DIQwen2-7B-Instruct 25.45 15.47 30.52 11.39 27.95
Qwen2-72B-Instruct-AWQ 24.98 15.32 30.75 11.25 26.91
GPT-4o-mini 25.00 15.31 28.21 10.27 28.45
Baichuan-M1-14B 30.99 18.95 33.95 14.52 21.41
MRD-RAG-PMHQwen2-7B-Instruct 26.14 15.93 31.34 11.65 28.13
Qwen2-72B-Instruct-AWQ 25.58 15.67 31.21 11.44 27.31
GPT-4o-mini 25.07 15.44 28.51 10.51 28.34
Baichuan-M1-14B 30.77 18.88 33.70 14.44 21.31
dialogue. Unlike evaluations by GPT or human doctors, these metrics treat 𝑑𝑖and𝑑𝑖′as different diseases because
thesemetricsonlyconsiderthesimilarityofstrings.TheLLMwhofailtogivetheexactnameofthediagnoseddisease
inadialoguemayreceivealowerscore.ForRAGmethods,iftheyretrievethewrongcandidatediseases,willmakethe
contentofthedialoguefurtherdeviatefromtherealinformationofthepatient,resultinginadecreaseinthesescores.
Therefore, the LLM with RAG method suffers performance drops on certain datasets (e.g. TCM-SD-100). However,
in general, the MRD-RAG method demonstrates better scores. Taking the results on the CMB dataset as an example,
MRD-RAGachievesaverageBLEU-1scoreimprovementsof1.99and2.60comparedtothesameLLMwithoutRAG
and with single-round RAG method, respectively. These results objectively reflect the effectiveness of MRD-RAG.
:Preprint submitted to Elsevier Page 11 of 17

Table 8
Retrieval performance of 𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻and𝐼𝑛𝑑𝑒𝑥𝐷𝐼.
Index Type MRR Hits@1 Hits@3 Hits@10
CMB𝐼𝑛𝑑𝑒𝑥𝐷𝐼0.084 0.027 0.068 0.203
𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻0.254 0.135 0.338 0.432
MM-Cases𝐼𝑛𝑑𝑒𝑥𝐷𝐼0.169 0.094 0.177 0.312
𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻0.440 0.322 0.481 0.686
TCM-SD-100𝐼𝑛𝑑𝑒𝑥𝐷𝐼0.111 0.040 0.120 0.250
𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻0.090 0.020 0.060 0.240
TCM-Cases𝐼𝑛𝑑𝑒𝑥𝐷𝐼0.472 0.308 0.569 0.792
𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻0.694 0.585 0.754 0.908
/uni00000014 /uni00000016 /uni00000018 /uni0000001a /uni0000001c
/uni00000057/uni00000052/uni00000053/uni00000010/uni0000004e/uni00000016/uni00000011/uni00000018/uni00000018/uni00000016/uni00000011/uni00000019/uni00000013/uni00000016/uni00000011/uni00000019/uni00000018/uni00000016/uni00000011/uni0000001a/uni00000013/uni00000016/uni00000011/uni0000001a/uni00000018/uni0000002a/uni00000033/uni00000037/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
Figure 5: Hyper-parameter study with 𝑡𝑜𝑝-𝑘, from 1 to 9.
5.2.4. Analysis of Retrieval Performance
To verify the effectiveness of 𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻, we use the first utterance of the patient in the dialogue as a query to
retrieve the DI-Tree knowledge base and compare it with 𝐼𝑛𝑑𝑒𝑥𝐷𝐼. The name of the disease suffered by the patient
is considered a hit if it matches the name of the disease retrieved from the DI-Tree knowledge base. We use mean
reciprocal rank (MRR) and Hits@n as an evaluation metric for retrieval performance. As shown in Table 8, retrieval
performance of 𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻is substantially improved compared to 𝐼𝑛𝑑𝑒𝑥𝐷𝐼on most of the datasets. For example,
on the MM-Cases dataset, 𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻achieves a 0.27 higher MRR score compared to 𝐼𝑛𝑑𝑒𝑥𝐷𝐼, indicating that
𝐼𝑛𝑑𝑒𝑥𝑃 𝑀𝐻can more accurately retrieve patient diseases. This also indirectly improves the diagnostic performance
of MRD-RAG on the MM-Cases dataset, aligning with the corresponding results in Table 3. It suggests that pseudo
medical history is closer to the semantics of the patient, consistent with the previous analysis.
5.2.5. Analysis of Top-k in Diagnostic Performance
In this section, we focus on analyzing the number of candidate diseases retrieved, specifically the impact of the
hyper-parameter 𝑡𝑜𝑝-𝑘onthediagnosticperformanceofMRD-RAG.Specifically,weanalyze 𝑡𝑜𝑝-𝑘valuesfromthelist
[1, 3, 5, 7, 9] to assess sensitivity. We randomly select 200 cases from all datasets to test the changes in MRD-RAG’s
GPTscorewithdifferent 𝑡𝑜𝑝-𝑘values.AsshowninFigure5,theGPTscoreinitiallyincreasesandthendecreaseswith
𝑡𝑜𝑝-𝑘.When 𝑡𝑜𝑝-𝑘= 1,theLLM’sdiagnosticperformanceissignificantlyworsethanatother 𝑡𝑜𝑝-𝑘valuesbecausethe
singleretrieveddiseaseislikelynottheactualdiseasethepatienthas,whichcouldmisleadtheLLM’sdiagnosis.When
𝑡𝑜𝑝-𝑘= 5,MRD-RAGachievesthebestdiagnosticperformance.However,if 𝑡𝑜𝑝-𝑘continuestoincrease,MRD-RAG’s
diagnosticperformancegetsworse.Larger 𝑡𝑜𝑝-𝑘valuesresultinretrievingtoomanyirrelevantdiseases,whichaffects
MRD-RAG’s ability to analyze differences between diseases, thereby impacting diagnostic performance.
:Preprint submitted to Elsevier Page 12 of 17

[First Round]
Patient: For the past six months, I have frequently experienced
symptoms of hypoglycemia , including palpitations, irritability , hunger ,
excessive sweating, and trembling limbs . 
Doctor:  Have you previously under gone blood sugar , insulin
measurement , or any other related tests? These tests are crucial for
diagnosing insulinoma or other causes of hypoglycemia. Also, please
tell me your gender, age, and whether you have any other
accompanying symptoms, such as vision changes or consciousness
disturbances. How are your dietary habits and exercise routine ? Are
you currently taking any medications?  
[Second Round]  
Patient:  I am a 45-year -old male, and recently my fasting blood sugar
has often been below  2.5 mmol/L , with elevated insulin levels . My diet
is regular , and I am not taking any medications.  
Doctor:  Based on your description and test results, your symptoms and
changes in blood sugar and insulin levels suggest the possibility of an
insulinoma . Insulinoma is a rare endocrine tumor that causes excessive
secretion of insulin by the pancreatic B cells, leading to hypoglycemia
symptoms. We recommend further tests, including Whipple's triad,
blood sugar , insulin measurement, and insulinoma stimulation tests, to
confirm whether it is an insulinoma. At the same time, we will also pay
attention to your other symptoms, such as vision changes and
consciousness disturbances, to rule out other possible diseases. While
awaiting further test results, we advise you to avoid prolonged fasting,
maintain a regular diet, and avoid strenuous exercise to prevent
hypoglycemic episodes. If you experience any discomfort or worsening
of symptoms, please seek medical attention promptly . [Kthinking in the Second Round]
### Summary and Distinction of Diseases  
1. Insulin oma: Symptoms: Paroxysm al hypoglycemia or coma,
neuropsychiatric symptoms such as palpitations, irritability , hunger ,
sweating, and tremors of the limbs. Clinical Manifestations: Occurs in
the morni ng, when fasting, after exertion, or during emotional stress.
Symptoms appear when blood glucose is below 2.8 mmol/L. Diagnosis:
Whipple's triad, blood glucose and insulin measurements, insulinoma
stimulation test , etc.
2. Functional β-cell tumor of the pancreas: Symptoms: Hypoglycemia
syndrome, possibly accompanied by gradual weight gain, and decline in
memory and reaction speed. Clinical Manifestations: Similar to
insulinoma but may lack typical symptoms; later stages may include
abdominal pain, weight loss, jaundice, etc. Diagnosis: Blood glucose,
insulin, and C-pept ide measurements, glucose tolerance test, fasting
test, etc.
......
### Analysis of Patient and Disease Correlation
Insulinoma: The symptoms described by the patient align with the
typical clinical manifestations of insulinoma , such as hypoglycemic
symptoms and blood glucose levels below the normal range. It is
recommended to conduct blood glucose and insulin measurements for
further diagnosis.
Functional β-cell tumor of the pancreas: The symptoms described by
the patient are simil ar to the clinical manifestations of a functional β-
cell tumor of the pancreas, such as hypoglycemic symptoms and
possible weight gain. Relevant tests are recommended to rule out or
diagnose the condition.
......[Kthinking in the First Round]  
### Summary and Distinction of Diseases  
1. Insulinoma: Symptoms: Paroxysmal hypoglycemia or coma,
neuropsychiatric symptoms such as palpitations, irritability , hunger ,
sweating, and trembling of the limbs. Clinical Manifestations:
Symptoms occur in the early morning when fasting, blood glucose
levels drop below 2.8 mmol/L, and symptoms disappear after oral or
intravenous glucose  intake. Diagnosis: Confirm ed through auxiliary
examinations such as blood glucose measurement, insulin
measurement, and sulfonylurea stimulation test.
2. Destruction of Islet Cells (Insulin-Dependent Diabetes
Mellitus): Symptoms: Frequen t urination, thirst, hunger, blurred vision,
fatigue, weight loss, and ketoacidosis. Clinical Manifestations: Elevated
blood glucose levels, presence of glucosuria, long-term effects on the
heart, eyes, kidneys, and nerves. Diagnosis: Comprehensive assessment
based on symptoms, blood glucose levels, and urine glucose tests.
......
### Questions to Ask:
What is your gender and age? Have you under gone blood glucose,
insulin measurements , or other related tests? Do your hypoglycemic
symptoms occur at specific times or under certain conditions? Do you
have any accompanying symptoms, such as visual changes or
consciousness disturbances? What are your dietary  and exercise
habits? Are you currently taking any medications?
......[Patient Information]
Disease Name: Insulinoma.
Patient Information: Patient Name: XXX Gender: Male Age: 45 years
Chief Complaint: Recurrent episode s of hypoglycemia. Current
Medical History: The patient reports experiencing recurrent episodes of
hypoglycemia over the past six months, primarily manifesting as
palpitations, irritability , hunger , sweating, and tremors in the limbs
when fasting in the morning. During each episode, blood glucose levels
drop below  2.8 mmol/L, and symptoms can be rapidly relieved by oral
or intravenous glucose administration. The patient states that the
frequency of these symptoms is irregular , occurring sometimes daily ,
sometimes only once every few weeks or months. Recently , due to
worsening symptoms, the patient sought medical attention at our
hospital. Past Medical History: The patient has no history of chronic
diseases such as hypertension or diabetes, nor any history of surgeries
or trauma. Physical Examination: No significant abnormalities were
found upon examination. The patient is conscious, with a normal heart
rate and no tremors in the limbs. Auxiliary Examinations: 1. Fasting
Blood Glucose Measurement: Fasting blood glucose levels are below
2.5 mmol/L. 2. Insulin Measurement: Fasting peripheral venous blood
insulin levels are significantly elevated, and insulin levels remain high
during hypoglycem ic episodes. 3. Sulfonylurea Stimulation Test:
Significant hypogl ycemia occurs after intravenous injection of
sulfonylurea, with a prolonged duration. 4. Selective Celiac
Arteriography: Suggests the possible presence of an insulinoma,
requiring further localization tests. Diagnosis: Insulinoma.Figure 6: A diagnostic case of MRD-RAG on the MM case. Red indicates keywords from the patient’s utterance that
contribute to the diagnosis. Green indicates keywords from retrieved knowledge. Blue indicates diagnosis results.
5.3. Case Study
5.3.1. MM Case
Figure6illustratesaconversationbetweenaninsulinomapatientandourMRD-RAGframework.Inthefirstround
ofdialogue,thepatientinitiallyrevealssymptomssuchashypoglycemia.Subsequently,theretrievermoduleretrieves
:Preprint submitted to Elsevier Page 13 of 17

[First Round]  
Patient: I've been experiencing cough, chest tightness,
shortness of breath, aversion to cold, and cold limbs  recently . 
Doctor:  Based on your description, you may be experiencing
symptoms related to Cold-Phlegm Obstructing the Lung, W ind-
Cold Invading the Lung, or W ind-Cold External Attack. T o
more accurately assess your condition, I need some additional
information. Could you please tell me if you are male or
female? What is your approximate age? Additionally , do you
have symptoms like fever , nasal congestion, or sore throat ?
What are the characteristics of your tongue and pulse ? For
example, the color and thickness of your tongue coating, and
the speed, strength of your pulse , etc. Also, what are your daily
dietary and lifestyle habits like? This information will help me
understand your condition more comprehensively and provide
more suitable advice.
[Second Round]  
Patient:  I am male, 45 years old. Recently , I haven't had fever ,
nasal cong estion, or sore throat. My tongue coating is white,
and my pulse feels  relatively tight. I have little appe tite for
food, and my lifestyle is quite busy . 
Doctor:  Based on your description and the information
provided, your symptoms indeed align with Cold Phlegm
Obstructing the Lung Syndrome . Considering your white greasy
tongue coating and wiry, tight pulse , this suggests the presence
of cold and dampness in the body , with significant phlegm and
dampness. Additionally , your bland taste and poor appetite,
along with a busy lifestyle, may indicate spleen deficiency ,
leading to poor water metabolism and accumulation of
dampness into phlegm.  [Kthinking in the Second Round]
### Summary and Analysis  
1. Wind-Cold Invading the Exterior Syndrome. Symptoms include aversion to cold
and fever , absence of sweating, headac he and body aches, nasal congestion with
clear nasal dischar ge, cough, expectoration of clear , thin sputum, heavy and turbid
voice, thin white tongue coating, and a floating and tight pulse......
2. Cold Phlegm Obstructing the Lung Syndrome. Symptoms include cough, copious
sputum that is white  and thick or clear and thin, easy expectoration, chest tightness,
wheezing, phlegm sounds in the throat, aversion to cold, cold limbs, pale tongue
with a white greasy or slippery coating, and a wiry or slippery pulse. Clinical
manifestations are cough, copious sputum, chest tightness, wheezing, aversion to
cold, cold limbs, white greasy or slippery tongue coating, and a wiry or slippery
pulse. Analysis: The symptoms described by the patient are consistent with common
symptoms of Cold Phlegm Obstructing the Lung Syndrome, such as cough, copious
sputum, chest tightness, wheezing, aversion to cold, cold limbs, white greasy or
slippery tongue coating, and a wiry or slippery pulse, suggesting a possible
diagnosis of this syndrome.
......
### Thought and Analysis Process:
1. Initial Judgment: Based on the symptoms described by the patient and the
characteristics of the tongue and pulse, it is initially judged that the patient may
have Wind-Cold Invading the Exterior Syndrome, Wind-Cold Invading the Lung
Syndrome, Cold Phlegm Obstructing the Lung Syndrome, Heart Yang Deficiency
Syndrome, or Wind-Cold Invading the Exterior Syndrome. 2. Exclusion and
Confirmation: The absence of symptoms such as fever , nasal congestion, and sore
throat excludes the possibility of Wind-Cold Invading the Exterior Syndrome and
Wind-Cold Invading  the Exterior Syndrome. The white greasy tongue coating and
wiry tight pulse  are consistent with the tongue and pulse characteristics of Cold
Phlegm Obstructing the Lung Syndrome, but the patient did not describe obvious
symptoms such as copious sputum, chest tightness, wheezing, or phlegm sounds in
the throat, so further confirmation is needed to determine whether it is Cold Phlegm
Obstructing the Lung Syndrome. 3. Comprehensive Consideration: Considering the
patient's age, gender , lifestyle habits (bland taste, poor appetite), combined with the
symptoms and tongue and pulse chara cteristics, it is comprehensively considered
that the patient is more likely to have Cold Phlegm Obstructing the Lung Syndrome.
......[Kthinking in the First Round]  
### Summary of Symptoms, Causes, Clinical Manifestations, and Information from
the Four Diagnostic Methods for Each Syndrome  
1. Cold Phlegm Obstructing the Lung Syndrome: Symptoms include cough,
wheezing, chest tightness, and expectoration of copious white sputum. Clinical
manifestations are aversion to cold, fear of cold, cold limbs, pale tongue with a
white, slippery coating, and a wiry and tight pulse. Four diagnostic methods
information: Facial complexion appea rs dull and bluish on inspection, phlegm
sounds are heard on auscultation, cough, wheezing, and chest tightness are reported
on inquiry , and a wiry and tight pulse is palpated.
2. Wind-Cold Invading the Lung Syndrome: Symptoms include cough, chest
tightness, wheezing, and expectoration of white sputum......
......
### Inquiry:
Patients should be asked about their gender , age, and symptoms such as fever , nasal
congestion, and sore throat , as well as specific characteristics of their tongue and
pulse  to further  clarify the diagnosis. Diagnosis direction: Based on the symptoms
described by the patient, there is a greater likelihood of Cold Phlegm Obstructing
the Lung Syndrom e, Wind-Cold Invading the Lung Syndrome, or Wind-C old
Invading the Exterior Syndrome. Detai led inquiry and observation using the four
diagnostic methods are needed to determine the specific syndrome type and further
understand the patient's overall health condition, including dietary and lifestyle
habits, to develop a personalized treatment plan.
......[Patient Information]  
Syndrome Name: Cold Phlegm Obstructing the Lung
Syndr ome
Patient Information: Medical Record. Name: XXX. Gender:
Male. Age: 45 years. Occupation: Driver . Chief Complaint: The
patient seeks medical help due to symptoms such as cough,
chest tightness, and shortness of breath. Present Illness: The
patient recently developed a cough, accompanied by copious,
white, and thick phlegm, chest tightness, and shortness of
breath, sometimes with wheezing sounds in the throat. The
patient feels aversio n to cold and has cold limbs. Additionally ,
the patient experiences a bland taste in the mouth, poor appetite,
pale tongu e with a greasy white coating, and a wiry, tight pulse.
The symptoms have persisted for several days, affecting normal
work and life. Past Medical History: The patient has a history of
chronic bronchitis, with recurrent episodes of cough and
shortness of breath, but has not sough t serious treatment. No
history of surgeries  or drug allergies. Physical Examination:
General Condition: The patient is consc ious, in fair spirits, with
a dull and bluish complexion. Phys ical Examination: No
abnormalities found in heart and lung auscultation, wheezing
sounds heard in the lungs, no obvious tenderness in the chest,
no masses in the abdomen, liver and spleen not palpable or
tender , no edema in the limbs. Preliminary Diagnosis: Based on
the patient's chief complaint, present  illness, and physical
examination results,  the preliminary diagnosis is Cold Phlegm
Obstructing the Lung Syndrome.Figure 7: A diagnostic case of MRD-RAG on the TCM case. Red indicates keywords from the patient’s utterance that
contribute to the diagnosis. Green indicates keywords from retrieved knowledge. Blue indicates diagnosis results.
:Preprint submitted to Elsevier Page 14 of 17

several candidate diseases including insulinoma and destruction of islet cells. These candidate diseases can cause the
symptoms described by the patient. The analyzer module further summarizes the interconnections and differences of
these candidate diseases, as well as key diagnostic information and questions that need to be asked, such as whether
the patient has undergone blood glucose measurement, insulin measurement, etc. The doctor module then responds
based on the dialogue and the output from the analyzer module. For example, the blood glucose measurement result
that the doctor module asks is the key to distinguish insulinoma and destruction of islet cells.
In the second round of the dialogue, after obtaining the patient’s fasting blood sugar and elevated insulin levels,
related candidate diseases are retrieved again. Based on the retrieved disease diagnostic information, the analyzer
module concludes that low fasting blood sugar and high elevated insulin levels are typical symptoms of insulinoma.
Therefore,theanalyzermoduleaccuratelyinfersthatthepatientislikelytohaveinsulinoma.Additionally,theanalyzer
summarizes that the key to diagnosing insulinoma lies in whipple’s triad. Then the doctor module also recommends
further diagnostic tests for the patient based on this thinking process. This thought process mimics the diagnostic
process of modern medical doctors.
5.3.2. TCM Case
Figure7showsaconversationbetweenapatientwith“coldphlegmobstructingthelungsyndrome”andourMRD-
RAG framework. In the first round of dialogue, the patient mentions symptoms such as cough and chest tightness.
Subsequently,theretrieverretrievesmultipleTCMsyndromeswiththesesymptoms,suchas“coldphlegmobstructing
the lung syndrome” and “wind-cold invading the lung syndrome”. These candidate diseases may cause symptoms
such as cough, chest tightness and shortness of breath. For further diagnosis, the analyzer module summarizes the
interconnections and differences of these TCM syndromes, indicating that questions still need to be asked, such as
whether the patient has fever, nasal congestion, characteristics of their tongue, etc. Then the doctor module precisely
inquire the patient according to the above analysis.
Inthesecondroundofdialogue,thepatientrevealsmoreinformationlikethecolorofhistonguecoating.Thiskey
informationhelptodifferentiatethesimilarTCMsyndromes.Sonowtheanalyzermodulecanaccuratelyidentifythat
thepatienthas“coldphlegmobstructingthelungsyndrome”combinedwiththeretrievedcandidatediseases.Finally,
thedoctormodulemakesanaccuratediagnosis.ThisprocessisconsistentwiththeTCMdoctor’sprocessinidentifying
different syndromes.
6. Conclusion
In this paper, we build a DI-Tree knowledge base for both modern medical and traditional Chinese medicine
diseases, and propose a multi-round diagnostic RAG framework called MRD-RAG. In terms of data processing, we
organize each disease into a tree structure to facilitate searching for diagnosis-related information using a predefined
search list. Additionally, we improve retrieval performance by generating pseudo medical history for each disease.
Mimicking the thinking process of a human doctor, the analyzer module in MRD-RAG generates a thinking process
based on the retrieved candidate diseases’ diagnosis information and the dialogue history. Subsequently, the doctor
module inquires the patient precisely to obtain more information or make a diagnosis. We conduct experiments on
two modern medical and two traditional Chinese medicine datasets to verify the effectiveness of MRD-RAG. Further
analysis demonstrates the superiority of MRD-RAG in multi-round diagnostic scenario.
CRediT authorship contribution statement
Yixiang Chen: Investigation, Methodology, Data curation, Validation, Writing – original draft, Writing – review
andediting. Penglei Sun: Investigation,Methodology,Writing–originaldraft,Writing–reviewandediting. Xiang Li:
Conceptualization,Writing–reviewandediting. Xiaowen Chu: Conceptualization,Resources,Supervision,Writing
– review and editing.
Data Availability
Our data can be found in our website https://github.com/YixiangCh/MRD-RAG/tree/master .
:Preprint submitted to Elsevier Page 15 of 17

References
[1] Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H., 2023. Self-rag: Learning to retrieve, generate, and critique through self-reflection. arXiv
preprint arXiv:2310.11511 .
[2] Balaguer,A.,Benara,V.,Cunha,R.L.d.F.,Hendry,T.,Holstein,D.,Marsman,J.,Mecklenburg,N.,Malvar,S.,Nunes,L.O.,Padilha,R.,etal.,
2024. Rag vs fine-tuning: Pipelines, tradeoffs, and a case study on agriculture. arXiv preprint arXiv:2401.08406 .
[3] Bao, Z., Chen, W., Xiao, S., Ren, K., Wu, J., Zhong, C., Peng, J., Huang, X., Wei, Z., 2023. Disc-medllm: Bridging general large language
models and real-world medical consultation. arXiv preprint arXiv:2308.14346 .
[4] Byambasuren, O., Yang, Y., Sui, Z., Dai, D., Chang, B., Li, S., Zan, H., 2019. Preliminary study on the construction of chinese medical
knowledge graph. Journal of Chinese Information Processing 33, 1–9.
[5] Chen,J.,Wang,X.,Gao,A.,Jiang,F.,Chen,S.,Zhang,H.,Song,D.,Xie,W.,Kong,C.,Li,J.,Wan,X.,Li,H.,Wang,B.,2023. Huatuogpt-ii,
one-stage training for medical adaption of llms. arXiv:2311.09774 .
[6] Duan,Y.,Zhou,Q.,Li,Y.,Qin,C.,Wang,Z.,Kan,H.,Hu,J.,2025. Researchonatraditionalchinesemedicinecase-basedquestion-answering
system integrating large language models and knowledge graphs. Frontiers in Medicine 11, 1512329.
[7] Fan, W., Ding, Y., Ning, L., Wang, S., Li, H., Yin, D., Chua, T.S., Li, Q., 2024. A survey on rag meeting llms: Towards retrieval-augmented
large language models, in: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 6491–6501.
[8] Huang, Z., Xue, K., Fan, Y., Mu, L., Liu, R., Ruan, T., Zhang, S., Zhang, X., 2024. Tool calling: Enhancing medication consultation via
retrieval-augmented large language models. arXiv preprint arXiv:2404.17897 .
[9] Jeong, M., Sohn, J., Sung, M., Kang, J., 2024. Improving medical reasoning through retrieval and self-reflection with retrieval-augmented
large language models. Bioinformatics 40, i119–i129.
[10] Jiang,J., Zhou, K.,Dong, Z.,Ye, K.,Zhao, W.X.,Wen, J.R.,2023a. Structgpt: Ageneral frameworkfor largelanguage modelto reasonover
structured data. arXiv preprint arXiv:2305.09645 .
[11] Jiang,X.,Zhang,R.,Xu,Y.,Qiu,R.,Fang,Y.,Wang,Z.,Tang,J.,Ding,H.,Chu,X.,Zhao,J.,etal.,2023b. Thinkandretrieval:Ahypothesis
knowledge graph enhanced medical large language models. arXiv preprint arXiv:2312.15883 .
[12] Jung,J.,Jeong,H.,Huh,E.N.,2025. Federatedlearningandragintegration:Ascalableapproachformedicallargelanguagemodels,in:2025
International Conference on Artificial Intelligence in Information and Communication (ICAIIC), IEEE. pp. 0968–0973.
[13] Li,B.,Meng,T.,Shi,X.,Zhai,J.,Ruan,T.,2023a. Meddm:Llm-executableclinicalguidancetreeforclinicaldecision-making. arXivpreprint
arXiv:2312.02441 .
[14] Li,L.,Wang,P.,Yan,J.,Wang,Y.,Li,S.,Jiang,J.,Sun,Z.,Tang,B.,Chang,T.H.,Wang,S.,etal.,2020a. Real-worlddatamedicalknowledge
graph: construction and applications. Artificial intelligence in medicine 103, 101817.
[15] Li, N., Yang, Z., Luo, L., Wang, L., Zhang, Y., Lin, H., Wang, J., 2020b. Kghc: a knowledge graph for hepatocellular carcinoma. BMC
Medical Informatics and Decision Making 20, 1–11.
[16] Li,Y.,Li,Z.,Zhang,K.,Dan,R.,Jiang,S.,Zhang,Y.,2023b. Chatdoctor:Amedicalchatmodelfine-tunedonalargelanguagemodelmeta-ai
(llama) using medical domain knowledge. Cureus 15.
[17] Liu, L., Yang, X., Lei, J., Liu, X., Shen, Y., Zhang, Z., Wei, P., Gu, J., Chu, Z., Qin, Z., et al., 2024. A survey on medical large language
models: Technology, application, trustworthiness, and future directions. arXiv preprint arXiv:2406.03712 .
[18] Liu,X.,Sun,P.,Chen,S.,Zhang,L.,Dong,P.,You,H.,Zhang,Y.,Yan,C.,Chu,X.,Zhang,T.y.,2025. Perovskite-llm:Knowledge-enhanced
large language models for perovskite solar cell research. arXiv preprint arXiv:2502.12669 .
[19] Mucheng,R.,Heyan,H.,Yuxiang,Z.,Qianwen,C.,Yuan,B.,Yang,G.,2022. Tcm-sd:abenchmarkforprobingsyndromedifferentiationvia
natural language processing, in: Proceedings of the 21st Chinese National Conference on Computational Linguistics, pp. 908–920.
[20] Ovadia, O., Brief, M., Mishaeli, M., Elisha, O., 2023. Fine-tuning or retrieval? comparing knowledge injection in llms. arXiv preprint
arXiv:2312.05934 .
[21] Song, Y., Sun, P., Liu, H., Li, Z., Song, W., Xiao, Y., Zhou, X., 2024. Scene-driven multimodal knowledge graph construction for embodied
ai. IEEE Transactions on Knowledge and Data Engineering .
[22] Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Shum, H.Y., Guo, J., 2023. Think-on-graph: Deep and responsible reasoning of large
language model with knowledge graph. arXiv preprint arXiv:2307.07697 .
[23] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al., 2023.
Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 .
[24] Wang, B., Zhao, H., Zhou, H., Song, L., Xu, M., Cheng, W., Zeng, X., Zhang, Y., Huo, Y., Wang, Z., et al., 2025. Baichuan-m1: Pushing the
medical capability of large language models. arXiv preprint arXiv:2502.12671 .
[25] Wang, J., Yang, Z., Yao, Z., Yu, H., 2024. Jmlr: Joint medical llm and retrieval training for enhancing reasoning and professional question
answering capability. arXiv preprint arXiv:2402.17887 .
[26] Wang, X., Chen, G.H., Song, D., Zhang, Z., Chen, Z., Xiao, Q., Jiang, F., Li, J., Wan, X., Wang, B., et al., 2023. Cmb: A comprehensive
medical benchmark in chinese. arXiv preprint arXiv:2308.08833 .
[27] WHO, 1983. Icd-10 wiki page. https://en.wikipedia.org/wiki/ICD-10 .
[28] Xiong,G.,Jin,Q.,Lu,Z.,Zhang,A.,2024a. Benchmarkingretrieval-augmentedgenerationformedicine,in:Ku,L.W.,Martins,A.,Srikumar,
V. (Eds.), Findings of the Association for Computational Linguistics: ACL 2024, Association for Computational Linguistics, Bangkok,
Thailand. pp. 6233–6251. URL: https://aclanthology.org/2024.findings-acl.372 , doi: 10.18653/v1/2024.findings-acl.372 .
[29] Xiong, G., Jin, Q., Wang, X., Zhang, M., Lu, Z., Zhang, A., 2024b. Improving retrieval-augmented generation in medicine with iterative
follow-up questions, in: Biocomputing 2025: Proceedings of the Pacific Symposium, World Scientific. pp. 199–214.
[30] Xiong, H., Wang, S., Zhu, Y., Zhao, Z., Liu, Y., Huang, L., Wang, Q., Shen, D., 2023. Doctorglm: Fine-tuning your chinese doctor is not a
herculean task. arXiv preprint arXiv:2304.01097 .
[31] Yang,A.,Yang,B.,Hui,B.,Zheng,B.,Yu,B.,Zhou,C.,Li,C.,Li,C.,Liu,D.,Huang,F.,Dong,G.,Wei,H.,Lin,H.,Tang,J.,Wang,J.,Yang,
J., Tu, J., Zhang, J., Ma, J., Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K., Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P.,
:Preprint submitted to Elsevier Page 16 of 17

Wang,P.,Peng,R.,Men,R.,Gao,R.,Lin,R.,Wang,S.,Bai,S.,Tan,S.,Zhu,T.,Li,T.,Liu,T.,Ge,W.,Deng,X.,Zhou,X.,Ren,X.,Zhang,
X.,Wei,X.,Ren,X.,Fan,Y.,Yao,Y.,Zhang,Y.,Wan,Y.,Chu,Y.,Liu,Y.,Cui,Z.,Zhang,Z.,Fan,Z.,2024a. Qwen2technicalreport. arXiv
preprint arXiv:2407.10671 .
[32] Yang, B., Jiang, S., Xu, L., Liu, K., Li, H., Xing, G., Chen, H., Jiang, X., Yan, Z., 2024b. Drhouse: An llm-empowered diagnostic reasoning
systemthroughharnessingoutcomesfromsensordataandexpertknowledge. ProceedingsoftheACMonInteractive,Mobile,Wearableand
Ubiquitous Technologies 8, 1–29.
[33] Yang, D., Rao, J., Chen, K., Guo, X., Zhang, Y., Yang, J., Zhang, Y., 2024c. Im-rag: Multi-round retrieval-augmented generation through
learning inner monologues, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information
Retrieval, pp. 730–740.
[34] Yang,S.,Zhao,H.,Zhu,S.,Zhou,G.,Xu,H.,Jia,Y.,Zan,H.,2024d. Zhongjing:Enhancingthechinesemedicalcapabilitiesoflargelanguage
model through expert feedback and real-world multi-turn dialogue, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp.
19368–19376.
[35] Yang, X., Wu, C., Nenadic, G., Wang, W., Lu, K., 2021. Mining a stroke knowledge graph from literature. BMC bioinformatics 22, 1–19.
[36] Ye,L.,Lei,Z.,Yin,J.,Chen,Q.,Zhou,J.,He,L.,2024. Boostingconversationalquestionansweringwithfine-grainedretrieval-augmentation
andself-check,in:Proceedingsofthe47thInternationalACMSIGIRConferenceonResearchandDevelopmentinInformationRetrieval,pp.
2301–2305.
[37] Zhang,H.,Chen,J.,Jiang,F.,Yu,F.,Chen,Z.,Li,J.,Chen,G.,Wu,X.,Zhang,Z.,Xiao,Q.,etal.,2023a. Huatuogpt,towardstaminglanguage
model to be a doctor. arXiv preprint arXiv:2305.15075 .
[38] Zhang,Y.,Li,X.,Shi,Y.,Chen,T.,Xu,Z.,Wang,P.,Yu,M.,Chen,W.,Li,B.,Jing,Z.,etal.,2023b. Etcmv2.0:anupdatewithcomprehensive
resource and rich annotations for traditional chinese medicine. Acta Pharmaceutica Sinica B 13, 2559–2571.
[39] Zhao, X., Wang, Y., Li, P., Xu, J., Sun, Y., Qiu, M., Pang, G., Wen, T., 2023. The construction of a tcm knowledge graph and application
of potential knowledge discovery in diabetic kidney disease by integrating diagnosis and treatment guidelines and real-world clinical data.
Frontiers in Pharmacology 14, 1147677.
[40] Zhu,X.,Li,Z.,Wang,X.,Jiang,X.,Sun,P.,Wang,X.,Xiao,Y.,Yuan,N.J.,2022. Multi-modalknowledgegraphconstructionandapplication:
A survey. IEEE Transactions on Knowledge and Data Engineering 36, 715–735.
:Preprint submitted to Elsevier Page 17 of 17