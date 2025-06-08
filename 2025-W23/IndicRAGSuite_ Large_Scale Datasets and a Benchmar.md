# IndicRAGSuite: Large-Scale Datasets and a Benchmark for Indian Language RAG Systems

**Authors**: Pasunuti Prasanjith, Prathmesh B More, Anoop Kunchukuttan, Raj Dabre

**Published**: 2025-06-02 12:55:51

**PDF URL**: [http://arxiv.org/pdf/2506.01615v2](http://arxiv.org/pdf/2506.01615v2)

## Abstract
Retrieval-Augmented Generation (RAG) systems enable language models to access
relevant information and generate accurate, well-grounded, and contextually
informed responses. However, for Indian languages, the development of
high-quality RAG systems is hindered by the lack of two critical resources: (1)
evaluation benchmarks for retrieval and generation tasks, and (2) large-scale
training datasets for multilingual retrieval. Most existing benchmarks and
datasets are centered around English or high-resource languages, making it
difficult to extend RAG capabilities to the diverse linguistic landscape of
India. To address the lack of evaluation benchmarks, we create IndicMSMarco, a
multilingual benchmark for evaluating retrieval quality and response generation
in 13 Indian languages, created via manual translation of 1000 diverse queries
from MS MARCO-dev set. To address the need for training data, we build a
large-scale dataset of (question, answer, relevant passage) tuples derived from
the Wikipedias of 19 Indian languages using state-of-the-art LLMs.
Additionally, we include translated versions of the original MS MARCO dataset
to further enrich the training data and ensure alignment with real-world
information-seeking tasks. Resources are available here:
https://huggingface.co/collections/ai4bharat/indicragsuite-683e7273cb2337208c8c0fcb

## Full Text


<!-- PDF content starts -->

IndicRAGSuite: Large­Scale Datasets and a Benchmark
forIndianLanguage RAG Systems
PasunutiPrasanjith1Prathmesh B More1;2
AnoopKunchukuttan1;2;3RajDabre1;2
1Nilekani Centre atAI4Bharat,
2Indian Institute ofTechnologyMadras, India
3Microsoft, India
Abstract
Retrieval­Augmented Generation (RAG) sys­
tems enable language models to access rele­
vant information and generate accurate, well­
grounded,andcontextuallyinformedresponses.
However, for Indian languages, the develop­
ment of high­quality RAG systems is hindered
by the lack of two critical resources: (1) evalu­
ationbenchmarksforretrievalandgeneration
tasks, and (2) large­scale training datasets for
multilingual retrieval. Most existing bench­
marksanddatasetsarecenteredaroundEnglish
or high­resource languages, making it difficult
to extend RAG capabilities to the diverse lin­
guisticlandscapeofIndia. Toaddress thelack
of evaluation benchmarks, we create IndicMS­
Marco,amultilingualbenchmarkforevaluating
retrieval quality and response generation in 13
Indian languages, created via manual transla­
tionof1000diversequeriesfromMSMARCO­
dev set. To address the need for training data,
webuildalarge­scaledatasetof(question,an­
swer, relevant passage) tuples derived from
the Wikipedias of 19 Indian languages using
state­of­the­artLLMs. Additionally,weinclude
translatedversionsoftheoriginalMSMARCO
dataset to further enrich the training data and
ensure alignment with real­world information­
seeking tasks. Resources are available here.
1 Introduction
Denseretrievalmodelshavesignificantlyadvanced
the field of information retrieval (IR), surpassing
traditionalmethodssuchasBM25inad­hocsearch
and question­answering tasks. These models lever­
age dense vector representations to capture seman­
ticrelationships,enablingefficientretrievalofrele­
vantdocumentsthroughapproximatenearestneigh­
borsearch. Suchcapabilitiesarepivotalforappli­
cationsincludingwebsearch,semanticsimilarity
tasks,andRetrieval­AugmentedGeneration(RAG)
systems, where dense retrievers allow language
models to access external knowledge efficiently.Dataset #Langs Source Size
NQ 1 Wiki 307K
TriviaQA 1 Web Docs 650K
SQuADv1.1 1 Wiki 100K
MSMARCO 1 Web Docs 8.8M
TREC­DL 1 Web Docs 367K
MKQA 26 Wiki 260K
TyDiQA 11 Wiki 204K
BEIR 1 Diverse Varies
IndicRAGSuite 19Wiki + MSMARCO 26M
Table 1: Statistics of existing retrieval model training
datasets.
However, the success of dense retrieval mod­
els critically depends on the quality and scale of
available training data and evaluation benchmarks
(Karpukhinetal. ,2020). Whilelarge­scaledatasets
suchasMSMARCO( Nguyenetal. ,2016),Natu­
ralQuestions ( Kwiatkowskiet al. ,2019),SQuAD
(Rajpurkar et al. ,2016), TriviaQA ( Joshi et al. ,
2017), and HotpotQA ( Yang et al. ,2018) have
propelled significant progress in English, the de­
velopment of robust dense retrieval systems for
under­resourced languages—particularly Indian
languages—remainsseverelyconstrained( Xiong
et al.,2021). Despite the demonstrated sample effi­
ciencyofdenseretrievalmodels( Quetal.,2021),
the scarcity of large­scale supervised datasets for
Indian languages continues to be a major bottle­
neck(Bonifacioetal. ,2021). Forinstance,English
datasets often contain millions of question­answer
pairs, whereas datasets for Indian languages are
limited to mere thousands ( Zhang et al. ,2021), fur­
therchallengedbylimiteddigitalpresence,script
diversity, and dialectal variations ( Jose and Bhat­
tacharyya ,2021).
Multilingual benchmarks such as MIRACL
(Zhangetal. ,2023),MKQA( Longpreetal. ,2021),
NeuCLIR ( Lawrie et al. ,2023), MLQA ( Lewis
et al.,2020), and XQuAD ( Artetxe et al. ,2020)
havecontributedsignificantlytoadvancingcross­
lingual retrieval. However, they predominantly
focus on high­resource languages, with limited
1

representation for Indian languages ( Ruder et al. ,
2021). Specialized domain­centric datasets such as
BioASQ( Nentidisetal. ,2023),FiQA(Angelidis
etal.,2020),andSciFact( Loetal.,2020)alsolack
substantial Indian language coverage ( Chakraborty
and Bhattacharyya ,2022). As a result, there re­
mains a critical gap: without adequate benchmarks
ortrainingdata,itisdifficulttobuild,evaluate,and
systematicallyimproveretrievalsystemsforIndia’s
rich and diverse linguistic landscape ( Joshi et al. ,
2020).
Toaddress this gap, we focus on creating essen­
tial infrastructure for Indian language retrieval:
Key Contributions
•Multilingual Benchmark for13 Indina lan­
guages: We manually translate a subset of
the MS MARCO dataset into 13 Indian lan­
guages,creatingamultilingualbenchmark(In­
dicMSMarco)forretrievalandresponsegen­
erationevaluation. Thisaddressestheabsence
of standardized evaluation datasets for Indian
languages and enables fair, systematic com­
parisons.
•Scalable Synthetic Dataset: We construct
a large­scale dataset comprising around 14
million (question, answer, relevant passage)
triplets across 19 Indian languages. This
datasetisgeneratedbyleveragingWikipedia’s
multilingual content and largelanguage mod­
els to create diverse and reasoning­rich ex­
amples. In addition, we translate the MS
MARCO train and dev sets into 14 Indian
languages, enabling supervised training of
dense retrievers in a multilingual setup. To­
gether, these datasets substantially expand the
resources available for training retrieval mod­
els for Indian languages.
2 RelatedWork
2.1 Multilingual Benchmarks forEvaluation
Recent multilingual retrieval benchmarks offer
valuable insights but remain inadequate for In­
dianlanguages. XOR­Retrieve( Asaietal. ,2021)
includes only Bengali and Telugu and focuses
on English­centric retrieval, limiting its monolin­
gual utility. MIRACL ( Zhang et al. ,2023) cov­
ersjustthreeIndianlanguagesandisrestrictedto
Wikipedia, which lacks regional depth. XTREME­
UP (Ruder et al. ,2023), though aimed at low­resource settings, suffers from noisy task inclu­
sionandstruggleswithscriptdiversity. Common
shortcomings across these efforts include limited
language coverage, inconsistent evaluation, and
translationartifacts—hinderingthedevelopmentof
robust retrieval systems for India’s linguistically
diverse population.
Toovercome these limitations, we introduce
IndicMSMARCO , a multilingual retrieval bench­
mark tailored for Indian languages, adapting the
high­quality MS MARCO framework ( Nguyen
etal.,2016)toregionalcontexts. Ourbenchmark
comprises1,000diversequeriesandpassagesfrom
MS MARCO, spanning topics like science, his­
tory, and technology, with balanced complexity
and length. Queries and passages are first trans­
lated into 13 Indian languages using LLaMA3.3
70B(Research,2024),andthen manuallyverified
andpost­editedbyexpertannotatorstoensurehigh­
quality translation. This post­editing process en­
sures linguistic and semantic fidelity, addressing
accuracy, fluency, and consistency, with particular
care for named entities and cultural nuances. In­
dicMSMARCOsupportsmonolingualretrieval,ad­
dresses script diversity, and provides standardized
evaluation metrics—filling a critical gap in bench­
marking retrieval systems for Indian languages.
2.2MultilingualRetrieversandTrainingData
Requirements
Severalmultilingualretrievalmodelshaveemerged
withvaryingarchitecturesandcapabilities. Early
baselineslikemBERT( Piresetal. ,2019)andXLM­
R(Conneauetal. ,2020)focusedoncross­lingual
understanding via masked language modeling and
have since been adapted for retrieval. mT5 ( Xue
et al.,2021) introduced a text­to­text paradigm and
has been used in both dual­encoder and genera­
tive retrieval settings. Dense retrievers such as
mDPR (Asai et al. ,2021) and mContriever ( Izac­
ard et al.,2022) leveraged parallel data and con­
trastive learning, respectively, while mE5 ( Wang
etal.,2022)usedmultitasklearningacross100+lan­
guagestodirectlyoptimizeretrievalperformance.
ProprietarysystemslikeOpenAI’stext­embedding­
ada­002 ( Neelakantan et al. ,2022) andVoyageAI
(VoyageAI ,2023)showstrongmultilingualperfor­
mance,thoughtheirtrainingremainsopaque. More
recently, jina­embeddings ( Günther et al. ,2024)
target long­context retrieval but still trail closed­
source models. Despite these advances, perfor­
manceremainsinconsistent across languagefami­
2

IndicMSMarco Benchmark Creation Workflow
Query Sampling
1000 diverse queries
from MS MARCO dev
set1
Translation
LLaMA 3.3 70B
model for 13
Indic languages2
Verification
Human validation of
semantic accuracy3
Evaluation
Benchmark tested
with multilingual
retrieval models4
Figure 1: Benchmark creation workflow for IndicMSMarco: from query selection to human­verified multilingual
evaluation.
lies,underscoringtheneedforinclusiveandtask­
specific multilingual training data.
Theavailability ofhigh­qualitytraining datare­
mains a key bottleneck for multilingual retrieval
systems. Existing resources typically rely on paral­
lelcorpora(e.g.,WikipediatranslationsinmDPR
(Asaietal. ,2021)),web­minedtextpairs(e.g.,mC4
inmContriever( Izacardetal. ,2022)),andlimited
human­annotated datasets like mMARCO ( Bonifa­
cioetal.,2021). Thesedatasets,however,areheav­
ily skewed toward English—with MS MARCO
offering 8.8M queries ( Nguyen et al. ,2016), while
Indian languages have access to only a fraction
of that volume. A recent effort to address these
gaps is the INDIC­MARCO dataset ( Haq et al. ,
2023),whichtranslatesMSMARCOinto11Indian
languages using NLLB­1.3B­Distilled via CTrans­
late2. However, its sentence­level translation strat­
egy fragments context, potentially reducing seman­
tic fidelity.
Toaddressdatalimitations, weconstruct alarge­
scale multilingual training dataset usingWikipedia
dumps from 19 Indian languages and generate
question­answer­reasoning triplets via the Llama
3.3 70B model. Unlike prior approaches that
split passages before translation, we retain full­
paragraphstructureandemployIndicTrans3­beta
(AI4Bharat )toensuresemanticcoherence. Wealso
translatetheMSMARCOtraininganddevsetsinto
14 Indian languages.
3 IndicMSMARCOBenchmark
To advance retrieval models for Indian lan­
guages,weintroduce IndicMSMARCO ,amulti­
lingualretrievalbenchmark. MSMARCO( Nguyen
et al.,2016) is a large­scale dataset designed for
question answering, passage ranking, and doc­
ument retrieval tasks. It comprises real­world
queries from Bing search logs, with relevant pas­
sages annotated by human assessors. While MSMARCOhasservedasacornerstoneforretrievalre­
search in English, the absence of comparable high­
quality benchmarks for Indian languages has hin­
deredthedevelopmentofrobustretrievalsystems
in these languages.
To address this gap, we adapt MS MARCO by
creating a multilingual variant specifically tailored
for Indian languages. Our benchmark consists of
1,000 carefully selected queries and their corre­
sponding passages from the MS MARCO develop­
ment set. The selection process prioritizes:
•Topic Diversity: Ensuring a wide range of
subject areas, including science, history, poli­
tics, health, and technology.
•Query Complexity Variation: Incorporat­
ingsimplefactualqueries,descriptivequeries,
and complex entity­based queries.
•Balanced Representation: Ensuring a mix
of short, medium, and long­form queries to
evaluate retrieval models comprehensively.
WeconstructtheIndicMSMARCObenchmark
in two phases: (1) automatic translation of queries
andpassagesusingtheLlama3.370Bmodel,and
(2)humanverification, correction,andannotation
toensure linguisticand semanticfidelity. Anillus­
trative example of a Hindi query–answer–passage
tripletfromIndicMSMARCOisshowninFigure 2.
3.1 AutomatedTranslationwith Llama 3.3
70B
To generate high­quality multilingual versions of
MSMARCOqueriesandpassages,weleveragethe
Llama 3.3 70B model, a state­of­the­art generative
languagemodelwithstrongmultilingualcapabili­
ties. Thetranslationpipelinefollowsastructured
approach:
•QueryTranslation: Each query from the se­
lected MS MARCO subset is translated into
3

Examples of query–answer–passagetriplet in Hindi (hi) from IndicMSMarco.
Query: कौनसारक्तप्रकारसबसेअ �धकबारहोताहै
Answer: ओपॉ�ज�टव
Passage: रक्तप्रकारऔरजनसंख्या।ओपॉ �ज�टवसबसेआमरक्तप्रकारहै।सभीजातीयसमूहोंमेंइनरक्तप्रकारों
कासमान�मश्रणनहींहोताहै।उदाहरणके �लए,�हस्पै�नकलोगोंमेंओरक्तप्रकारकीसंख्याअपेक्षाकृतअ �धकहोतीहै,जब�कए�शयाई
लोगोंमेंबीरक्तप्रकारकीसंख्याअपेक्षाकृतअ �धकहोतीहै।यू.एस.जनसंख्यामें �व�भन्नरक्तप्रकारोंका �मश्रणइसप्रकारहै:
Figure 2: Benchmark Example in Hindi
13majorIndianlanguages. Llama3.370Ben­
surestheretentionofqueryintentwhileadapt­
ing language­specific structures.
•PassageTranslation: Thecorrespondingpas­
sagesaretranslatedusingcontext­awaregen­
eration, ensuring coherence and fidelity to
the original English passage. The model is
promptedtopreservenamedentities,numeri­
caldata,anddomain­specificterminologyto
maintain retrieval relevance.
The automated translation process enables rapid
expansion of the benchmark to multiple Indian lan­
guages. However, machine translations often intro­
duce errors related to syntax, semantic drift, and
ambiguity. Toensurequality,weconductarigorous
human verification and annotation phase.
3.2 HumanVerificationandAnnotation
Aftertranslatingqueriesandpassagesintomultiple
languages through LLaMA 3.3 70B, we employ
astructuredhuman annotationprocessto validate,
correct,andrefinetranslations. Thisphaseinvolves
expertlinguists,nativespeakers,andbilingualan­
notators across differentIndian languages.
The verification process follows three key steps:
•LinguisticAccuracyCheck: Annotatorsre­
view translations for grammatical correctness,
fluency,andreadability. Thisstepensuresthat
thetranslatedqueriesandpassagesadhereto
the natural syntax and style of each language.
•Semantic Consistency Evaluation: Each
query and passage pair is cross­checked
againsttheoriginalEnglishversiontoverify
that the meaning remains intact. Annotators
flagandcorrectanyinstancesofsemanticdrift,
mistranslations, or ambiguous phrasing.
•Entity and Domain­SpecificValidation: To
maintainretrievalrelevance,expertsvalidatetechnicalterms,namedentities(e.g.,locations,
personnames,numericalvalues),andcontext­
sensitiveinformation. Necessarycorrections
are made to preserve factual and contextual
accuracy.
Inadditiontovalidation, annotatorsactivelycor­
rect translation errors to ensure precision and natu­
ralnessineverylanguage. Thismeticulousverifi­
cationandcorrectionprocessensuresthatIndicMS­
MARCO serves as a high­quality, reliable bench­
mark for evaluating retrieval models in Indian lan­
guages. Byincorporatingbothautomatedtransla­
tionandhumanrefinement,wecreateadatasetthat
is not only scalable but also linguistically robust.
3.3 Significance of IndicMSMARCO
The IndicMSMARCO benchmark is a crucial re­
source for the development of dense retrieval mod­
els tailored to Indian languages. It enables:
•StandardizedEvaluation: Providingacom­
mon ground for comparing retrieval perfor­
mance across multiple Indian languages.
•EnhancedMultilingualRetrievalResearch:
Facilitating the training and fine­tuning of
retrieval models for underrepresented lan­
guages.
•Real­WorldApplicability: Addressing prac­
ticalchallengesinmultilingualsearchsystems,
digital libraries, and knowledge retrieval ap­
plications in India.
By constructing IndicMSMARCO, we take a
significantsteptowardbridgingthelinguisticgapin
informationretrievalandfosteringequitableaccess
to advanced retrieval technologies across diverse
Indian languages.
4

Language Multilingual e5­small Multilingual e5­base Multilingual e5­largeLLM2VEC
LLaMA3.1 8B InstructBGE­M3
Assamese 0.30 0.40 0.45 0.42 0.46
Bengali 0.39 0.46 0.48 0.44 0.49
Gujarati 0.34 0.43 0.48 0.42 0.48
Hindi 0.44 0.49 0.52 0.49 0.52
Kannada 0.38 0.44 0.47 0.40 0.47
Malayalam 0.38 0.45 0.49 0.43 0.49
Marathi 0.36 0.45 0.49 0.45 0.49
Nepali 0.39 0.45 0.49 0.45 0.49
Odia 0.31 0.39 0.45 0.34 0.45
Punjabi 0.32 0.42 0.48 0.42 0.48
Tamil 0.38 0.45 0.49 0.40 0.49
Telugu 0.39 0.45 0.50 0.42 0.50
Urdu 0.35 0.45 0.49 0.44 0.48
Table 2: MRR scores on IndicMSMarco Benchmark for 13 Indian languages using various dense retrieval models.
Highest scores per language are in bold.
3.4 Experiments and Results
We evaluate the performance of various dense
retriever models on IndicMSMarco Benchmark
across 13 major Indian languages using Mean
Reciprocal Rank (MRR) as the evaluation met­
ric. The models compared include LLM2VEC
(LLaMA3.18BInstruct) ,BGE­M3 ,andtheMul­
tilingual E5 family—e5­small,e5­base, ande5­
large. These models span multilingual, instruction­
tuned,andretrieval­centricarchitectures,offering
insights into their strengths and limitations in mul­
tilingual Indian language settings.
AsshowninTable 2,BGE­M3 achievesthebest
ornear­bestMRRinthemajorityoflanguages,lead­
ingin8outof13languages. Notably,itscores 0.49
in Malayalam and Tamil, and 0.50in Telugu, in­
dicating strong generalization across linguistically
diverse Indian scripts.
Multilingual e5­large also performs consis­
tently well, obtaining the highest score in 4 lan­
guages, including Hindi (0.52) ,Gujarati (0.48) ,
andUrdu (0.49) . The steady improvement from
e5­small to e5­large demonstrates the benefits of
scalingformultilingualretrievaleffectiveness. The
smallere5modelsstilldeliverrespectableperfor­
mance, particularly in medium­resource languages.
LLM2VEC , based on the LLaMA 3.1 8B ar­
chitecture and fine­tuned for retrieval tasks, shows
competitiveresultsacrossseverallanguages. For
example,itachieves 0.49inHindiandMarathi,and
0.45inNepali. Whileitdoesnotdominateacross
alllanguages,itsresultsshowthatinstruction­tuned
LLMs are viable alternatives for dense retrieval in
multilingual contexts.
Languages such as Hindiconsistently receivehighMRRscoresacrossallmodels,likelydueto
betterrepresentationintrainingcorpora. Incontrast,
Assamese andOdiascore lower overall, reflect­
ing the challenges of retrieval in lower­resource
languages.
Overall,thebenchmarkresultsinTable 2high­
light the strength of modern dense retrievers like
BGE­M3 andMultilingual e5­large , particularly
in multilingual and low­resource settings. These
findingsestablishastrongbaselineforIndianlan­
guage retrieval and point toward key directions for
futurework onmultilingual and underrepresented
language support.
4 RAGTrainingDataset Construction
The effectiveness of any information retrieval (IR)
system is largely dependent on the quantity and
quality of its training data. Advancing IR re­
search in Indian languages has historically been
hinderedbythescarcityoflarge­scale,high­quality
datasets. Toaddressthisgap,weconstructedtwo
complementary training datasets: (1) aWikipedia­
generated multilingual dataset of question­answer­
reasoning triplets, and (2) a high­quality translated
version of the MS MARCO dataset adapted for
Indian languages.
4.1 Wikipedia­Based
Question­Answer­ReasoningDataset
4.1.1 Dataset Design andObjectives
Our primary objective was to construct a linguisti­
callydiversedatasetfortrainingretrievermodels
in Indian languages. TheWikipedia­based data set
was designed to meet the following criteria.
5

WikiExtractor LLMWikipedia
Dump Set of
paragraphs”Question ”:"को�वड-19के सामान्यलक्षणोंमेंक्याशा �मलहैं?",
”Answer”:"बुखार,खांसी,थकान,साँसलेनेमेंक �ठनाई,औरगंधकानुकसानऔर
आयु-स्वाद।"
”Reasoning ”:"को�वड-19के लक्षणप �रवत�नीयहैं,ले�कनसामान्यलक्षणोंमें
बुखार,खांसी,थकान,साँसलेनेमेंक �ठनाई,औरगंधकानुकसानऔरआयु-स्वाद
शा�मलहैं।यहजानकारीको �वड-19के लक्षणोंके बारेमें �वस्तारसेबतातीहै।"को�वड-19के लक्षणप �रवत�नशीलहोतेहैं,जोहल्के
लक्षणोंसेलेकरगंभीरबीमारीतकहोसकतेहैं।
सामान्यलक्षणोंमेंबुखार,खांसी,थकान,सांसलेनेमें
क�ठनाई,औरगंधकानुकसान(औरस्वादका
नुकसान)शा�मलहैं।एकहीसंक्रमणवालेलोगोंमें
अलग-अलगलक्षणहोसकतेहैं,औरउनके लक्षण
समयके साथबदलसकतेहैं।उदाहरणके �लए,एक
व्य�क्तकोतेजबुखार,खांसीऔरथकानहोसकती
है,जब�कदूसरेव्य�क्तकोबीमारीकीशुरुआतमें
कमबुखारहोसकताहैऔरएकसप्ताह
बादसांसलेनेमेंक �ठनाईहोसकतीहै।हल्के लक्षण,
�बनापरेशानी,�नकटऔरगलेकीलगन
(ईएनटीसंबंधी),स्वादकानुकसानऔरगंधका
नुकसानहोनाको �वड-19की�वशेषताके
साथ95%तकपायागयाहै।”Question ”:"को�वड-19संक्रमणवालेलोगोंमेंलक्षणोंकी �व�वधताक्या"
"होसकतीहै?",
”Answer”:"अलग-अलगलक्षणऔरसमयके साथबदलनेवालेलक्षण।" ,
”Reasoning ”:"एकहीसंक्रमणवालेलोगोंमेंअलग-अलगलक्षणहोसकतेहैं,और
उनके लक्षणसमयके साथबदलसकतेहैं।यहजानकारीको �वड-19के लक्षणोंकी
�व�वधताकोदशा�तीहै।"
”Question ”:"कोविड-19मेंगंधकीहानिकीविशिष्टताक्याहै?",
”Answer”:"95%।",
”Reasoning ”:"�बनापूव�कान,नाकऔरगलेके लोगोंमें(ईएनटी �वकार,स्वादका
नुकसानगंधकीहा �नको�वड-19�व�शष्टताके साथ95%है।यहजानकारीको �वड-19
मेंगंधकीहा�नकी�व�शष्टताकोदशा�तीहै।"”wiki_id”: ”1238767”,
”Title”:"को�वड-19के लक्षण"
Figure3: Thedataprocessingpipeline—fromrawWikipediadumpstoparagraphextractionandanLLM—generating
Hindi Q&Apairs with explanatory reasoning
•Scale:Millions of question­answer­reasoning
tripletsperlanguage tosupportrobustmodel
training.
•Diversity: Coverage across a wide range of
topicsandlinguisticnuancesreflectingIndia’s
cultural and regional diversity.
•Quality: Contextually accurate, grammati­
cally correct, and semantically meaningful
triplets.
•Multilingual Coverage: Broadapplicability
across 19 major Indian languages.
4.1.2 Source: WikipediaDumps
Toconstructthedataset,weusedWikipediadumps,
compressedarchivesthatcontain full­textarticles
in various languages. Wikipedia serves as an ideal
source due to the following:
•MultilingualAvailability: Coverageacross
all 19 targetedIndian languages.
•TopicDiversity: Wide­rangingsubjectmatter,
includingscience,history,culture,andcurrent
events.
•OpenAccess: Unrestricted usage, allowing
the creation of large­scaledata sets.
4.1.3 Data Extraction and Preprocessing
We processed raw Wikipedia dumps using
WikiExtractor , cleaning the extracted content
through:
•Removalofmetadata,HTMLtags,formatting,
and hyperlinks.•Segmentationofarticlesintoparagraph­level
chunks to ground question­answer pairs in lo­
calized contexts.
Paragraph­level segmentation was crucial to en­
surethatthegeneratedquestionsandanswersmain­
tained a tight contextual relevance.
4.1.4 TripletGeneration UsingLLaMA3.3
70B
To generate high­quality question­answer­
reasoning triplets for Indian languages, we curated
a pipeline that transforms raw Wikipedia content
intostructuredQAdata. AsillustratedinFigure ??,
the process begins with extracting paragraphs
fromWikipedia dumps using the WikiExtractor
tool. Each extracted paragraph is associated with
metadata such as the article title and a unique wiki
ID.
WethenusetheLLaMA3.370Bmodeltogen­
erate structured triplets for each paragraph. Specif­
ically,themodelproducesthreedistinctquestion­
answerpairs,eachaccompaniedbyadetailedrea­
soningsegment. Thisreasoningcomponentiscru­
cial—itensuresthatthe answerisgroundedinthe
paragraph and that the model interprets content be­
yond superficial keyword matching. Moreover, the
triplets are crafted to coverdiversequestiontypes
(e.g., “what,” “why,” “how,” “when”) and differ­
ent parts of the paragraph, thereby reducing bias
toward the initial lines.
Key aspects of this step include:
•Comprehensiveness: Questions are gener­
ated to span the full semantic content of the
paragraph,promotingdiverseinformationcov­
erage.
6

•Reasoning­driven generation: The addition
of explanatory reasoning promotes deeper un­
derstanding and better supports answer valid­
ity.
•Multilingual robustness: The LLaMA 3.3
70B model was prompted to adhere to the
grammaticalandsyntacticstructuresofeach
targetIndian language.
Figure3demonstratesan example inHindi. The
paragraph describes COVID­19 symptoms, from
which the model generates semantically varied
questions. Each question is paired with an appro­
priate answer and a reasoning span that justifies
theanswerchoiceusingexplicitcontextfromthe
paragraph.
4.1.5 Scale and Multilingual Coverage
The final dataset comprises approximately 14 mil­
lion question­answer­reasoning triplets across 19
Indian languages. This large­scale dataset is de­
signed to support robust training and evaluation of
multilingual information retrieval (IR) models in
linguistically diverse and low­resource settings.
To ensure the quality and utility of the dataset,
we incorporated a filtering step as part of our data
curationpipeline. Duringthisstage,paragraphsthat
were either too short (lacking sufficient context)
or excessively long (risking coherence issues or
hallucination by the LLM) were excluded. This
filtering was applied prior to triplet generation to
maximizeconsistencyandrelevanceintheresulting
data.
Table3providesdetailedstatisticsforeachlan­
guage, including the number of paragraphs and
triplets both before and after filtering.
4.2 TranslatedMS MARCO Dataset
While theWikipedia­based dataset offers wide top­
icaldiversityandsupportsparagraph­groundedrea­
soning, it lacks the structured, real­world query
characteristics critical for training effective re­
trieval models. To address this, we constructed
a translated version of the MS MARCO dataset
specifically tailored for Indian languages. Our
translation pipeline begins by selecting queries
andcorrespondingpassagesfromtheoriginalMS
MARCO training and development sets. These
weretranslatedinto14IndianlanguagesusingIn­
dicTrans3­beta, a state­of­the­art translation model
fine­tuned for Indian language translation tasks.
UnlikeprioreffortssuchasIndicIRSuite( Haqetal. ,Table3: WikipediaGeneratedTrainingDataStatistics
for Each Language
Language BeforeFiltering AfterFiltering
Assamese 333,705 217,018
Bengali 3,320,042 2,060,963
English 6,384,632 4,109,199
Gujarati 354,824 245,063
Hindi 2,220,115 1,182,023
Kannada 1,114,088 670,236
Kashmiri 29,487 1,138
Maithili 92,722 38,028
Malayalam 1,371,674 901,402
Manipuri 46,458 31,389
Marathi 200,000 96,820
Nepali 402,100 222,597
Odia 268,239 175,743
Punjabi 689,306 393,769
Santali 189,066 97,963
Sindhi 250,836 118,869
Tamil 946,544 507,664
Telugu 3,276,885 1,824,025
Urdu 199,999 27,575
Total 21,740,681 13,927,586
2023),whichtranslatedsentence­levelfragments
after splitting passages using tools like Moses Sen­
tenceSplitter,ourmethod preservesfull­paragraph
structure throughout translation. This approach
maintains better contextual coherence, semantic
alignment, and domain fidelity.
While Indic­MARCO employed the int­8 quan­
tized version of the NLLB­1.3B Distilled model
primarilyfortranslationefficiency,weprioritized
translation quality and linguistic richness, select­
ing IndicTrans3­beta( AI4Bharat ) for its superior
BLEUscoresandfluencyinIndianlanguages. Spe­
cial attention was paid to preserving the original
search intent in queries and minimizing distortions
caused by automatic translation.
This high­fidelity, paragraph­level translated
MSMARCOdatasetenablesmorerealistic,task­
specific training of dense retrievers for Indian
languages. It complements our Wikipedia­based
datasetbyaddingreal­world,query­drivenexam­
ples,thusfacilitatingrobustretrievalperformance
across both open­domain and structured query sce­
narios. Throughdeeperlinguisticintegrity,broader
languagecoverage,andstrongeralignmentwiththe
retrievaltask,ourapproachprovidesasubstantially
improvedtrainingresourcecomparedtoprevious
multilingual adaptations of MS MARCO.
4.3 FutureWork
Withtheconstructionofhigh­qualitymultilingual
datasets—comprising aWikipedia­basedquestion­
7

Table4: TranslatedMSMarcoTrainingDataStatistics
by Language
Language Code #TrainDataset #ValDataset
Assamese asm 778,638 97,941
Bengali ben 778,638 97,941
Gujarati guj 778,638 97,941
Hindi hin 778,638 97,941
Kannada kan 778,638 97,941
Malayalam mal 778,638 97,941
Marathi mar 765,873 97,941
Nepali nep 754,154 97,941
Odia ori 782,282 97,941
Punjabi pan 778,638 97,941
Sanskrit san 778,638 97,941
Tamil tam 778,638 97,941
Telugu tel 778,638 97,941
Urdu urd 770,089 97,941
Total 10,848,130 1,371,174
answer­reasoning corpus and translated version of
MS MARCO—the next phase of our work will
focus on training and evaluating dense retriever
models using these resources. This includes fine­
tuning already existing retrieval architectures to
understand the individual and combined impact of
syntheticdataandreal­worldquery­passagepairs.
We aim to benchmark performance across 13 In­
dianlanguages,withspecialemphasisongainsin
low­resource language settings. Additionally, fu­
ture directions include integrating domain­specific
corpora like legal or medical texts, and incorpo­
rating human­in­the­loop refinement, ultimately
moving toward the development of robust, open­
domain multilingual IR systems tailored for Indian
language users.
5 Conclusion
Wepresent IndicMSMARCO ,ahuman­verified
multilingualbenchmarkforinformationretrievalin
13Indianlanguages. ByadaptingtheMSMARCO
development set using Llama 3.3 70B and ex­
pertlinguisticcorrection,IndicMSMARCOmain­
tains semantic accuracy and fluency across diverse
queries and topics. It enables standardized eval­
uation of retrieval models in low­resource Indian
language settings.
Tosupportmodeltraining,weintroduceadual­
sourcecorpusthatcombinescontextuallytranslated
MS MARCO data with a large­scale Wikipedia­
based dataset. This hybrid strategy captures both
real­world search relevance and broad domain
knowledge, enhancing model generalization across
diverse IR scenarios in Indian languages.References
AI4Bharat. Indictrans3­beta: Multilingual translation
for 22 indic languages. https://huggingface.
co/spaces/ai4bharat/IndicTrans3-beta .
Stefanos Angelidis, Thanasis Mavropoulos, and Van­
gelis Karkaletsis. 2020. Fiqa: Financial opinion
mining and question answering. arXiv preprint
arXiv:2004.12403 .
Mikel Artetxe, Sebastian Ruder, and Dani Yogatama.
2020. Cross­lingual question answering as a starting
point for zero­shot semantic parsing. In Proceedings
ofACL.
AkariAsai, Kyungjae Lee, Xing Li, and Eunsol Choi.
2021. Multilingualpassageretrievalforopen­domain
question answering. In Proceedings ofACL­IJCNLP .
Luiz Bonifacio, Israel Campiotti, Rodrigo Nogueira,
andRobertoLotufo.2021. mmarco: Amultilingual
version of ms marco passage ranking dataset. In
Proceedings of EACL .
TanmoyChakrabortyandPushpakBhattacharyya.2022.
Indian language information retrieval: Challenges
and opportunities. In Proceedings of FIRE .
AlexisConneau,KartikayKhandelwal,NamanGoyal,
Vishrav Chaudhary, and 1 others. 2020. Unsuper­
vised cross­lingual representation learning at scale.
arXiv preprint arXiv:1911.02116 .
MichaelGünther,JonathanAbb,LucaCostabello,and
1 others. 2024. Jina embeddings: Open­source mod­
elsforlong­contextrepresentations. arXivpreprint
arXiv:2401.17201 .
Saiful Haq, Ashutosh Sharma, and Pushpak Bhat­
tacharyya. 2023. Indicirsuite: Multilingual dataset
and neural information models for indian languages .
Preprint, arXiv:2312.09508.
Gautier Izacard, Patrick Lewis, Lucas Hosseini, and
1 others. 2022. Few­shot dense retrieval with con­
trastive learning. arXiv preprint arXiv:2212.03551 .
BlessonJoseand PushpakBhattacharyya. 2021. Asur­
vey of multilingual information retrieval for indian
languages. ACM Computing Surveys .
Mandar Joshi, Eunsol Choi, Daniel SWeld, and Luke
Zettlemoyer.2017. Triviaqa: Alargescaledistantly
supervised challenge dataset for reading comprehen­
sion. InProceedings ofACL .
Prasenjit Joshi, Parthasarathi Majumder, and Mandar
Mitra. 2020. The state and future of ir for indian
languages. ACM Transactions on Asian Language
Information Processing .
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, LedellWu, Sergey Edunov, Danqi Chen, and
Wen­tau Yih. 2020. Dense passage retrieval for
open­domain question answering. In Proceedings
of EMNLP .
8

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red­
field,and1others.2019. Naturalquestions: Abench­
mark for question answering research. In Transac­
tions of theACL .
Dawn Lawrie, James Mayfield, and Paul McNamee.
2023. Neuclir: A benchmark for neural chinese­
language information retrieval. In Proceedings of
TREC.
Patrick Lewis, Barlas Oğuz, Ruty Rinott, Sebastian
Riedel, and Holger Schwenk. 2020. Mlqa: Eval­
uating cross­lingual extractive question answering .
Preprint, arXiv:1910.07475.
Kyle Lo, Lucy LuWang, Mark Neumann, and 1 others.
2020. Scifact: Adataset for scientific claim verifica­
tion. InProceedings of EMNLP .
Shayne Longpre, Yi Lu, and Joachim Daiber. 2021.
Mkqa: Amultilingualknowledgequestionanswering
benchmark. arXiv preprint arXiv:2107.13613 .
Arvind Neelakantan, Tao Xu, Raul Puri, and 1 others.
2022. Text and code embeddings by contrastive pre­
training.OpenAI TechnicalReport .
Anastasios Nentidis, Georgios Katsimpras, Anasta­
sia Krithara, Salvador Lima López, Eulália Farré­
Maduell, Luis Gasco, Martin Krallinger, and Geor­
giosPaliouras.2023. Overviewof BioASQ2023: The
EleventhBioASQChallengeon Large­ScaleBiomedi­
calSemanticIndexingand QuestionAnswering ,page
227–250. Springer Nature Switzerland.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng.
2016. Ms marco: A human generated machine
reading comprehension dataset. arXiv preprint
arXiv:1611.09268 .
Telmo Pires, Eva Schlinger, and Dan Garrette. 2019.
How multilingual is multilingual BERT? In Proceed­
ings ofACL .
YingqiQu,YuchenDing,JingLiu,and1others.2021.
Rocketqa: Anoptimizedtrainingapproachtodense
passageretrievalforopen­domainquestionanswer­
ing. InProceedings of NAACL­HLT .
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100,000+ questions for
machine comprehension of text. In Proceedings of
EMNLP.
MetaAI Research. 2024. Llama 3.3 technical report .
MetaAI Research Report.
SebastianRuder,JonathanClark,andAlexanderGutkin.
2023. Xtreme­r: Towards more challenging and
multilingual multimodal learning. In Proceedings
of EMNLP .
SebastianRuder,NoahConstant,JanBotha,and1oth­
ers. 2021. Xtreme­r: Towards more challenging
and nuanced multilingual evaluation. arXiv preprint
arXiv:2104.07462 .Voyage AI. 2023. Voyage­lite­01­instruct: Efficient
multilingual embeddings. TechnicalReport.
LiangWang,NanYang,Xiaolong Huang,and 1others.
2022. E5: Towards text embeddings that transfer
better across languages and tasks. arXiv preprint
arXiv:2212.03563 .
LeeXiong,ChenyanXiong,YeLi,and1others.2021.
Pretrained transformers for text ranking: Bert and
beyond. In Proceedings of NAACL­HLT .
LintingXue,NoahConstant,AdamRoberts,MihirKale,
RamiAl­Rfou,AdityaSiddhant,AdityaBarua,and
Colin Raffel. 2021. mt5: Amassively multilingual
pre­trainedtext­to­texttransformer. Proceedingsof
NAACL.
Zhilin Yang, Peng Qi, Saizheng Zhang, and 1 others.
2018. Hotpotqa: Adataset for diverse, explainable
multi­hop question answering. In Proceedings of
EMNLP.
Xinyu Zhang, Nandan Thakur, Barlas Oguz, Sachin
Gupta, andWen­tauYih. 2023. Miracl: Amultilin­
gualretrievalbenchmark. In ProceedingsofNeurIPS .
XinyuZhang,NandanThakur,BarlasOguz,and1others.
2021. Mr. tydi: Amulti­lingual benchmark for dense
retrieval. In Proceedings ofACL­IJCNLP .
9

AppendixA: PromptTemplateforQuestion­Answer­ReasoningGeneration fromWikipediaArticles
SystemPrompt:
YouareapreciseandhelpfulQuestion­AnswerGeneratorthatcreatesfactualquestionswithverifiableanswersfrom
providedcontent in <target_language>.
TaskPrompt:
You will first be given an example of how the desired output will look like. Then you will be given the content based on
which you have to generate up to three challenging, logically coherent questions that strictly meet the following criteria:
1.Standalone&AdditionalContext­Independent: Thequestionsshouldbeunderstandablewithoutadditionalcontext
and must not contain any references to “the paragraph” or “the article” outside of the content provided.
2.UnambiguousAnswer: Each question should have a single, clear,and factual answer.
3.Grounded in Context & Conceptual Format: Each question must be conceptually rooted in the provided article’s
content and follow this format:
­ Start with a clear question word (e.g., What,How,Where,When).
­ Integrate key information from the article smoothly,using logical connectors (e.g., “in relation to”, “compared to”,
“as a result of”, “which also”, “in addition to”).
­ If no valid questions can be generated from the content, do not generate any questions.
Foreach question:
­ Provide the answer in parentheses after the question. The answer can be either one word or a phrase.
­ Clearly explain the reasoning process, using an excerpt from the article as a reference.
­Do not use mixedlanguagefor numbering; always use theformat“Question 1”, “Question 2”, etc. Avoidnon­English
numbering even for non­English datasets.
­ Except for numbering headers, the questions, answers, and reasonings should be in the same language as the article,
which is <target_language>.
Example:
Question 1: [Sample question]
Reasoning: [Explanation referencing article content]
Content: [Title]: [ArticleText]
Figure4: Systemand taskprompt usedfor generatinghigh­quality,language­specificquestion­answerpairs from
article content.
10