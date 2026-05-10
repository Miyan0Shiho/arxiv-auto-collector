# Tatarstan Toponyms: A Bilingual Dataset and Hybrid RAG System for Geospatial Question Answering

**Authors**: Mullosharaf K. Arabov

**Published**: 2026-05-07 10:08:17

**PDF URL**: [https://arxiv.org/pdf/2605.05962v1](https://arxiv.org/pdf/2605.05962v1)

## Abstract
This paper addresses automatic geospatial question answering over multilingual toponymic data. An original bilingual dataset of toponyms of the Republic of Tatarstan is introduced, comprising 9,688 structured records with linguistic, etymological, administrative, and coordinate information (93.1% georeferenced). Based on this dataset, a question-answering corpus of approximately 39,000 question-context-answer triples is constructed with guaranteed answer localization. A hybrid retriever integrates dense semantic indexing (multilingual-e5-large) with geospatial filtering via KD-trees and haversine distance. On 500 test queries, the hybrid search achieves Recall@1=0.988, Recall@5=1.000, and MRR=0.994, significantly outperforming BM25 and purely spatial methods. Among tested reader architectures (RuBERT, XLM-RoBERTa-large, T5-RUS), XLM-RoBERTa-large attains the best quality: EM=0.992, F1=0.994. On raw outputs, RuBERT models fail on coordinate questions (F1=0) while XLM-RoBERTa-large reaches F1=0.984; however, simple post-processing eliminates numerical gaps and restores RuBERT accuracy to 100%. This discrepancy stems from tokenization differences and pre-training corpora composition. All resources (dataset, QA corpus, model weights, web demo) are openly published on Hugging Face. Results apply to geospatial QA services, geocoding, and digital humanities in multilingual regions.

## Full Text


<!-- PDF content starts -->

Tatarstan Toponyms: A Bilingual Dataset and Hybrid RAG
System for Geospatial Question Answering
A Preprint
M. K. Arabov∗
Department of Data Analysis and Programming Technologies
Institute of Computational Mathematics and Information Technologies
Kazan (Volga Region) Federal University
Kazan, Russia
MKArabov@kpfu.ru
May 8, 2026
Abstract
Thispaperaddressestheend-to-endtaskofautomaticgeospatialquestionansweringovermultilingual
toponymic data. An original bilingual dataset of toponyms of the Republic of Tatarstan is introduced,
comprising 9,688 structured records with detailed linguistic, etymological, administrative, and
coordinate information (93.1% of objects are georeferenced). Based on this dataset, a specialized
question-answering corpus of approximately 39,000 “question–context–extractable answer” triples is
constructed,withguaranteedanswerlocalizationwithinthetext. Tosolvethetask,anarchitecture
combiningtwokeycomponentsisproposed: ahybridretrieverthatintegratesdensesemanticindexing
usingmultilingual-e5-largewithageospatialfilterandranking(KD-trees,haversinedistance),and
an extractive reader based on fine-tuned transformer models. On 500 test queries, the hybrid
searchachievesRecall@1=0.988,Recall@5=1.000,andMRR=0.994,statisticallysignificantly
outperforming both BM25 and the purely spatial method. Among the tested reader architectures
(RuBERT, XLM-RoBERTa-large, T5-RUS), the best answer extraction quality is attained by the
multilingual XLM-RoBERTa-large model: EM = 0.992, F1 = 0.994. A contrasting effect is observed:
on raw outputs, RuBERT-based models fail to answer coordinate-related questions (F1 = 0), whereas
XLM-RoBERTa-largeachievesF1=0.984;however,simplepost-processingcompletelyeliminates
the gaps in numerical values and restores RuBERT accuracy to 100%. This discrepancy is attributed
to tokenization specifics and the composition of pre-training corpora. The created resources (dataset,
QA corpus, trained model weights, and web demonstrator) are openly published on the Hugging
Face platform. The obtained results can be directly applied in the development of geospatial
question-answering services, geocoding systems, and digital humanities projects for multilingual
regions.
KeywordsTatarstan toponyms ·question answering ·RAG·extractive reading·hybrid search·semantic embeddings ·
geospatial analysis ·bilingual dataset·Tatar language·Russian language·XLM-RoBERTa·multilingual E5·natural
language processing
1 Introduction
Geographicalnamesconstituteafundamentalcomponentofspatialdatainfrastructureandplayakeyroleincartography,
navigation,regionalgovernance,anddigitalhumanitiesresearch[Suleymanovetal.,2021,Galimovetal.,2023]. In
multilingualregionssuchastheRepublicofTatarstan,whereRussianandTatararebothofficialstatelanguages,toponyms
function simultaneously in two linguistic codes and accumulate rich etymological, historical, and dialectological
information. ThischaracteristictransformsTatarstantoponymyintoanexceptionallyvaluableyetdifficult-to-process
∗Email:MKArabov@kpfu.ruarXiv:2605.05962v1  [cs.CL]  7 May 2026

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
information resource. Traditional geospatial resources (GeoNames, OpenStreetMap) provide basic coordinate and
classification data; however, they lack deep linguistic and etymological details and are not designed for semantic search
thataccommodatessynonymyofgeographicalterms(“selo”–“derevnya”[village],“reka”–“pritok”[river–tributary])
and multilingual spelling variants of the same object [Rehurek and Sojka, 2006].
In parallel, the past decade has witnessed notable progress in the computational processing of the Tatar language:
representativetextcorporahavebeencreated[Saykhunovetal.,ske,IPSAN],morphologicalanalysissystems[Gilmullin
and Gataullin, 2017] and tokenization tools [Arabov, 2026a] have been developed, and methods for semantic annotation
[Mukhamedshinetal.,2025,MindubaevandGatiatullin,2024]andtheconstructionofdistributionalwordanddocument
representations[Arabov,2026b,Gafarovetal.,2025]havebeenproposed. Nevertheless,specializedquestion-answering
datasetsthataccountfortoponymicspecificityhavebeenabsentuntilnow. ExistingQAcollections(SQuAD,SberQuAD,
and their Russian-language counterparts) are oriented toward news and encyclopedic texts and do not cover structured
geographicinformationthatincludescoordinates,objecttypes,andetymologicaldescriptions. Atthesametime,queries
suchas“WhereistheMeshaRiverlocated?”,“WhatarethecoordinatesofthevillageofRantamak?”,and“Whyis
Lake Kaban so named?” are natural for users of geoservices and educational platforms.
ModerngenerativemodelsunderlyingRetrieval-AugmentedGeneration(RAG)requirereliableretrievalcomponents
capable of supplying relevant context. In the domain of geographic search, the combination of dense semantic
embeddings obtained from multilingual transformers (multilingual E5 [Liang et al., 2022], XLM-RoBERTa [Conneau
et al., 2020]) with geospatial ranking and filtering is of particular interest. However, the integration of such approaches
formultilingualtoponymicdata,especiallycoupledwithsubsequentextractivereading,remainsunderexplored. The
identified gap—the simultaneous absence of both a dataset and an end-to-end question-answering architecture for
geographic queries in a bilingual setting—determines the relevance of the present work.
2 Related Work
Toponymic Data and Resources for the Tatar Language.InternationalprojectssuchasGeoNamesandOpenStreetMap
provide open coordinate and attribute information; however, they do not contain the detailed etymological and linguistic
data necessary for scholarly onomastics [geo]. Academic works on the toponymy of Tatarstan (dictionaries by Garipova,
Sattarov,Äkhmätÿanov)havebeenpartiallydigitizedonthe“ToponymsofTatarstan”portalbutlackamachine-readable
application programming interface. Within the framework of the present study, the authors collected and openly
published a structured dataset comprising 9,688 records with names in Russian and Tatar, object types, etymology, and
coordinatereferencingfor93%ofentries[TatarNLPWorld,a];itisthisdatasetthatservesastheempiricalfoundationof
the present work.
Tools for Automatic Processing of the Tatar Language.For the Tatar language, general-purpose corpora [Saykhunov
etal.,ske,IPSAN],morphologicalanalysissystems[GilmullinandGataullin,2017],tokenizers[Arabov,2026a],as
wellasmethodsforsemanticannotationbasedonknowledgegraphs[Mukhamedshinetal.,2025]andmachinelearning
[MindubaevandGatiatullin,2024],havebeendeveloped. Softwaresolutionsforconstructingvectorrepresentations
of words and documents have been created [Arabov, 2026b, Gafarov et al., 2025]. All of the listed resources are
consolidated on the Hugging Face platform within the TatarNLPWorld organization, ensuring reproducibility of
experiments [TatarNLPWorld, a].
Semantic and Geospatial Search.Classicallexical methods suchas BM25[Rehurek and Sojka,2006] struggle with
cross-lingual and synonymic queries. The development of Transformer architectures has led to the emergence of dense
retrievalmodels. TheE5family[Liangetal.,2022]istrainedwithacontrastivelossfunctiononmultilingualcollections
and is capable of encoding documents and queries into a unified semantic space. The multilingual-e5-large model
demonstrates high effectiveness for low-resource and multilingual scenarios [Conneau et al., 2020, Schweter and Tekir,
2020]. In parallel, geoinformatics employs spatial indices (KD-trees [Bentley, 1975], R-trees) and metrics that account
for the Earth’s curvature (haversine distance [Sinnott, 1984]). Works [Galimov et al., 2023, Burnashev et al., 2025]
proposedgeolinguisticsystemscombiningfuzzylogicwithspatialdatafordialectanalysis;however,theydidnotutilize
dense embeddings. Hybrid approaches that combine textual and geographic relevance in a weighted manner have
beensuccessfullyappliedinrecommendationservicesandpoint-of-interestsearch,butsuchsolutionshavenotbeen
previously investigated for multilingual toponyms with detailed etymological attribution.
Question-Answering Systems and Extractive Reading.For the extractive QA task, SQuAD [Rajpurkar et al., 2016]
and its multilingual versions have become the de facto standard datasets. Russian-language counterparts, such as
SberQuAD[Efimovetal.,2020],areorientedprimarilytowardnewsandencyclopedictextsanddonotcontainstructured
geographic data with coordinates. Broader diagnostic benchmarks for the Russian language, in particular Russian
SuperGLUE[Fenogenovaetal.,2021],enabletheevaluationofmultipleaspectsoflanguageunderstandingbutalso
do not includetasks requiring the extraction of numericalcoordinatesand etymological information. Among models,
2

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
BERT-basedarchitecturesleadthefield: RuBERT[Devlinetal.,2019](furtherpre-trainedonRussiantexts)andthe
multilingualXLM-RoBERTa[Conneauetal.,2020],trainedononehundredlanguages. Thelatterhasdemonstrated
excellentresultsinrecognizingnestednamedentities,includinggeographicalones,asconfirmed,forexample,withinthe
Russian-languagecompetitionRuNNE-2022[Artemovaetal.,2022]. GenerativeT5models[Raffeletal.,2020]arealso
applied,buttypicallyintheanswergenerationsetting. Asignificantlimitationofmonolingualmodels,revealedinrecent
experiments[Choureetal.,2022],istheirinabilitytoprocessnumericalcoordinatesduetoWordPiecetokenization
specificsandashortageofrelevantexamplesintrainingsets. MultilingualmodelsemployingSentencePiece,bycontrast,
successfullyextractnumericalsequences. Todate,nospecializedQAbenchmarkscombiningstructuredgeographic
information with etymological and coordinate components have been proposed.
Thus, the literature review attests to the existence of disparate components (language resources for Tatar, dense retrieval
models,geospatialfilteringmethods,extractivereadingmodels);however,theirintegrationintoaunifiedsystemcapable
of finding the required toponym and extracting the precise answer for an arbitrary geographic query in Russian or Tatar
is lacking. The present work is aimed at filling this gap.
3 Methodology
3.1 Hybrid Retrieval: Architecture and Method
The purpose of the retrieval component is to take a textual query 𝑞, optionally accompanied by a geographic point
(lat 𝑞,lon 𝑞)and a radius 𝑅, and produce a ranked list of the 𝑘most relevant toponyms from a set of documents 𝐷,
eachofwhichisequippedwithcoordinates. Thearchitecturecomprisestwoparallelindexingstages—semanticand
spatial—which interact during query processing in a hybrid ranking procedure.
Semantic Indexing.To represent the content portion of a document, the multilingual encoder model
intfloat/multilingual-e5-large [Liangetal.,2022]isemployed,whichtransformsthecontextualfield context
into a 1024-dimensional dense vector. The model was chosen as it demonstrates state-of-the-art results in cross-lingual
search and supports both Russian and Tatar languages. All document embeddings are normalized to unit 𝐿2-norm,
allowingcosineproximitytobeassessedviathedotproduct. Computationisperformedinbatchesof32documents
using a GPU. For efficient nearest-neighbor search, a flat inner product index ( IndexFlatIP ) is constructed using the
FAISS library [Johnson et al., 2019]; when a GPU accelerator is available, the index is placed in GPU memory. Query
processing reduces to encoding its text with the same model, 𝐿2-normalizing the query vector, and extracting the top- 𝑘
documents with the maximum dot product. The resulting quantity
sem_score(𝑖)=⟨𝑒 𝑞,𝑒𝑖⟩
is interpreted as the semantic relevance of document𝑖.
Geospatial Filter and Ranking.Spatialselectionisimplementedintwopasses. Inthefirststage,aboundingboxis
constructed around the query point:
Δlat=𝑅/111320,Δlon=𝑅/(111320·cos(𝜙)),
where𝜙is the latitude in radians and 𝑅is the search radius. Objects whose coordinates fall within this region are
declared candidates. Then, for each candidate, the exact haversine distance [Sinnott, 1984] is computed, and those
objectswhosedistanceexceeds 𝑅arefilteredout. Thespatialscoreoftheremainingcandidatesisdeterminedbyan
exponentially decaying function of distance:
geo_score(𝑖)=exp
−𝑑𝑖
𝑅
.
To accelerate the filtering stage, a KD-tree [Bentley, 1975] is built over the coordinates of all documents during corpus
preparation, enabling logarithmic-time retrieval of the subset of objects within a given neighborhood.
Combined Relevance.The final relevance is formed as a weighted sum of normalized semantic and spatial scores.
Semantic scores are normalized using min-max scaling among the candidates that passed the geofilter:
sem_norm(𝑖)=sem_score(𝑖)−min sem
maxsem−minsem,
andindegenerate cases(whenthedifferenceiscloseto zero),allvaluesaresetto1. Spatial scoresarenormalizedby
dividing by the maximum value among the candidates:
geo_norm(𝑖)=geo_score(𝑖)
maxgeo.
3

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
The combined score is computed as
score(𝑖)=𝛼·sem_norm(𝑖)+(1−𝛼)·geo_norm(𝑖),
where𝛼∈[0,1] is a hyperparameter governing the contribution of the semantic component. Objects are ranked by
descending score(𝑖),andthetop- 𝑘arereturned. Thevalueof 𝛼istunedempiricallyonavalidationset(200queries)via
gridsearchover{0.1,0.3,0.5,0.7,0.9} bymaximizingtheaverageRecall@5. Intheexperiments, 𝛼=0.1wasfound
to be optimal, with a fixed radius of𝑅=50km, chosen based on the typical scale of local search within Tatarstan.
3.2 Extractive Reading: QA Corpus Generation and Model Training
Fortheextractivereadingcomponent,theoriginaltoponymdatasetistransformedintoasetofquestion-answerpairs
conformingtotheSQuADformat[Rajpurkaretal.,2016]. Theprocedurepursuestwoobjectives: toensureprecise
answerlocalizationwithinthecontextfortrainingextractivemodels,andtocoverallmajorinformationcategoriesof
the source records.
Context and Question Generation.Each significant field of a source record (name, object type, location, etymology,
coordinates, region, physiographic characteristics, sources) is prefixed with a Russian-language label (Название
(рус):[Name(Rus):],Объект:[Object:],Этимология:[Etymology:],Расположение:[Location:],Координаты:
[Coordinates:], etc.) and concatenated with other fields via the “ | ” separator. If the resulting string exceeds 2048
characters,proportionaltruncationofeachfieldisappliedwhilepreservingtheprefix. Forseveninformationcategories,
Russian-languagequestiontemplateswitha {name}placeholderhavebeendeveloped,intowhichtheRussian(or,if
unavailable,Tatar)nameoftheobjectisinserted. Examplesoftemplates:Что такое {name}?[Whatis{name}?]
(object type),Где находится {name}?[Where is {name} located?] (location),Какие координаты у {name}?
[Whatarethecoordinatesof{name}?] (coordinates),Почему {name} так называется?[Whyis{name}sonamed?]
(etymology), and others. The full list of templates is provided in Section 4.2.
Answer Construction and Annotation.For each question, the answer is the value of the corresponding field
(for coordinates, the string “latitude, longitude”). The start position of the answer within the context is computed
algorithmically: as the sum of the lengths of all fields preceding the target prefix, plus the length of the prefix itself.
Thisensuresprecisecharacter-levelannotationrequiredfortrainingextractivemodels. ThemaximumnumberofQA
pairs perrecord is limited to ten to ensureuniform coverage. Theresulting corpus comprises38,696 pairs andis split
into training (90%) and validation (10%) sets, stratified by question type.
Models and Training.For the extractive reading task, two groups of transformer architectures were fine-tuned:
monolingual Russian-language RuBERT models (base and large versions, pre-trained on SberQuAD [Efimov et al.,
2020, Devlin et al., 2019]) and the multilingual XLM-RoBERTa-large model [Conneau et al., 2020], pre-trained on
SQuAD2.0. ThegenerativemodelT5-RUS[Raffeletal.,2020]wasnotfine-tunedduetotechnicalreasonsandwas
used in its base variant as an additional reference point. All models were trained on the generated QA corpus for three
epochsusingtheAdamWoptimizer,alearningrateof 3×10−5,abatchsizeof4,andlinearwarm-upover500steps.
For extractive models, the standard procedure of predicting start and end positions of the answer was applied; for
T5-RUS,theformat“question: ... context: ... →answer”wasemployed. Asasimplebaseline,aheuristicrulewas
implemented that extracts the answer based on question keywords and prefixes in the context.
Thetrainedmodels,aswellasthegeneratedQAcorpus,arepublishedonHuggingFace[TatarNLPWorld,b]andare
available for reproducibility of the results.
4 Data
4.1 Source Dataset of Tatarstan Toponyms
The empirical foundation of the study is a specialized bilingual dataset of Tatarstan toponyms, collected, structured, and
published by the authors in open access on the Hugging Face platform [TatarNLPWorld, c] as part of the present work.
The necessity of creating this resource is dictated by the absence of machine-readable datasets that simultaneously
contain coordinate, linguistic, and etymological information about the region’s geographical objects. The dataset
aggregates data from fundamental academic works on the toponymy of Tatarstan: the hydronym dictionaries of
F.G.Garipova,thestudiesofG.F.SattarovonTatartoponymy,theetymologicaldictionariesofR.G.Әхмәтьянов,the
multi-volumeTatarEncyclopedia,dialectologicaldictionaries,aswellasmaterialsfromthedigitalportalТопонимы
Татарстана(toponym.antat.ru). Relianceonpeer-reviewedacademicsourcesensureshighreliabilityandcompleteness
of the included information.
The total volume of the dataset amounts to 9,688 records, each corresponding to a single geographical object within the
territoryof the Republicof Tatarstan andadjacent regionsof compact Tatarsettlement. The recordstructure includes
4

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
the following fields: a unique identifier, source URL, toponym type (toponym or microtoponym), toponym subtype
(oikonym, hydronym, oronym, or no type), geographical object (76 unique categories:деревня[village],село[selo],
река[river],озеро[lake],поле[field],гора[mountain],луг[meadow],приток реки[river tributary],поляна
[glade], etc.), name in Russian, name in Tatar, federal subject, physiographic details, geographical location, name
etymology, bibliographic sources, latitude and longitude coordinates (where available), and a map availability flag.
Etymological information, which is of particular value for scholarly research in onomastics and historical geography, is
present in approximately 63% of records.
A key characteristic of the dataset is the presence of coordinate referencing: 9,023 records (93.1%) are furnished with
latitude and longitude values, enabling their use in geospatial analysis and distance-based filtering procedures. The
remaining 665 records (6.9%) lacking coordinates are excluded from the hybrid search experiments but were employed
in generating the question-answering corpus for categories not requiring spatial information (etymology, sources).
The distribution of records by toponym subtype is presented in Table 1. The predominant share is constituted by
oikonyms—namesofpopulatedplaces(деревни[villages],сёла[selos],посёлки[settlements]),whichreflectsthe
settlementstructureintheregionandthethematiccoverageofthesourcesused. Hydronyms(namesofwaterbodies:
rivers,lakes,streams)andoronyms(namesoflandforms: mountains,hills,ravines)togetherformapproximatelyone
quarterofthecorpus. The“notype”categoryunitesrecordsforwhichthetoponymsubtypewasnotexplicitlyindicated
in the sources; however, they retain information about the geographical object and etymology.
Table 1: Distribution of dataset records by toponym subtype
Toponym Subtype Number of Records Share, %
Oikonym 7000 72.2
Hydronym 1500 15.5
Oronym 800 8.3
No type 388 4.0
Total 9688 100.0
Granularity by geographical object type (Table 2) reveals the most frequent categories. Populated places dominate:
деревни[villages] (36.1%) andсёла[selos] (20.6%), which corresponds to the share of oikonyms in Table 1. Among
natural objects, the most represented areлуга[meadows] (8.3%),реки[rivers] (7.2%),поля[fields] (5.2%), andгоры
[mountains](4.1%). About10.7%ofrecordsbelongtoother,lessfrequentcategories,suchaslakes,rivertributaries,
glades,springs,naturaltracts,swamps,ravines,andotherobjects. Thepresenceof76uniquetypesofgeographical
objects ensures high entity diversity and allows testing search algorithms under conditions of heterogeneous taxonomy.
Table 2: Top-10 most frequent geographical objects in the dataset
Geographical Object Number of Records Share, %
Деревня(village) 3500 36.1
Село(selo) 2000 20.6
Луг(meadow) 800 8.3
Река(river) 700 7.2
Поле(field) 500 5.2
Гора(mountain) 400 4.1
Озеро(lake) 300 3.1
Приток реки(river tributary) 250 2.6
Поляна(glade) 200 2.1
Other 1038 10.7
Total 9688 100.0
The regional distribution lawfully reflects the geographic focus of the dataset: approximately 93% of records
pertain to objects within the territory of the Republic of Tatarstan, about 4% to Tyumen Oblast (areas of compact
settlement of Siberian Tatars), and the remaining 3% are distributed among adjacent subjects of the Russian Federation
(Башкортостан[Bashkortostan], Ulyanovsk, Samara, Orenburg Oblasts, and others).
Tofacilitatesemanticsearch,alltextualfieldsofeachrecord,excludingcoordinates,weremergedintoasinglecontextual
fieldcontextusing English-language key prefixes. The context format is as follows:
5

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
Name (rus): <name_rus> | Name (tat): <name_tat> | Type: <toponym_type>
| Subtype: <toponym_subtype> | Object: <geographical_object> |
Etymology: <etymology> | Details: <physiographic_details> | Location:
<geographical_location> | Sources: <bibliographic_sources>
Fieldscontainingnoinformationwereexcludedfromthecontext. ThechoiceofEnglish-languageprefixesisdictatedby
theuseofthemultilingualmodel multilingual-e5-large ,forwhichEnglish-languagekeysprovidemorerobust
matching of semantically similar fields in cross-lingual space. The described procedure enabled the formation of 9,023
documents with non-empty contextual fields and coordinate referencing, which constituted the corpus for indexing and
testing the retrieval component.
4.2 Generation of the Question-Answering Corpus
Basedonthestructuredtoponymdataset,automaticgenerationofasyntheticquestion-answeringcorpuswasperformed,
intended for training and evaluating extractive QA models. The key requirement for the generation procedure was
guaranteedanswerlocalizationwithinthecontexttext,whichisnecessaryforthecorrectcomputationofanswerstart
and end positions during extractive model training.
Foreachinformationtypeinasourcerecord(name,objecttype,location,etymology,coordinates,region,physiographic
characteristics, sources), the corresponding field was transformed into a string with a Russian-language prefix
unambiguously identifying the category:Название (рус):[Name (Rus):],Название (тат):[Name (Tat):],Объект:
[Object:],Этимология:[Etymology:],Расположение:[Location:],Координаты:[Coordinates:],Регион:
[Region:],Физико-географические сведения:[Physiographic Details:],Источники:[Sources:]. All prefixed
stringswereconcatenatedviathe“|”separatorintoasinglecontext. Whenthemaximumpermissiblelength(2048
characters) was exceeded, proportional truncation of each field was applied, with mandatory preservation of the prefix.
This approach guarantees the presence of any potential answer within the context together with the prefix marking
it, which subsequently enables unambiguous determination of the answer start position as the sum of the lengths of
preceding fields and separators.
For seven information categories, Russian-language question templates were developed. Each template contains a
{name}placeholder, into which the object name is inserted. Examples of templates by category:
•Objecttype:Что такое {name}?[Whatis{name}?],Какой тип у {name}?[Whattypeis {name}?],К
какому типу относится {name}?[To which type does {name} belong?];
•Location:Где находится {name}?[Whereis{name}located?],В каком месте расположен {name}?
[Inwhatplaceis{name}situated?],Где именно расположен {name}?[Whereexactlyis{name}situated?];
•Etymology:Что означает название {name}?[What does the name {name} mean?],Почему {name}
так называется?[Why is {name} so named?],Каково происхождение названия {name}?[What is the
origin of the name {name}?];
•Coordinates:Какие координаты у {name}?[What are the coordinates of {name}?],Где на карте
находится {name}?[Where on the map is {name}?];
•Region:Вкакомрегионенаходится{name}?[Inwhichregionis{name}located?],Какойфедеральный
субъект у {name}?[What federal subject does {name} belong to?];
•Sources:Какие источники упоминают {name}?[Which sources mention {name}?],Где можно
прочитать о {name}?[Where can one read about {name}?];
•Physiographiccharacteristics:Какие физико-географические сведения о {name}?[Whatphysiographic
details are there about {name}?],Что известно о географических особенностях {name}?[What is
known about the geographical features of {name}?].
Duringgeneration,foreachrecord,oneofthetemplateswasrandomlyselected,intowhichtheRussiannameofthe
object (or, if absent, the Tatar name) was inserted. The answer was formed as the value of the corresponding field (for
coordinates,astringwithlatitudeandlongitudeseparatedbyacomma). The answerstartpositionwithinthecontext
wascomputedalgorithmicallyasthetotallengthofallfieldsprecedingthetargetprefix,plusthelengthoftheprefix
itself. Thus, each QA pair contains precise positional labels conforming to the SQuAD format [Rajpurkar et al., 2016].
Themaximumnumberofquestion-answerpairsperrecordwaslimitedtotentopreventthedominanceofobjectswitha
largenumberofpopulatedfieldsandtoensureuniformcoverage. Thetotalvolumeofthegeneratedcorpusamountedto
38,696 QA pairs. The distribution by question type is presented in Table 3.
6

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
Table 3: Distribution of the synthetic QA corpus by question type
Question Type Number of Pairs Share, %
Coordinates 12,344 31.9
Object type 8,032 20.8
Location 5,724 14.8
Etymology 5,032 13.0
Region 3,868 10.0
Sources 3,052 7.9
Physiographic characteristics 644 1.6
Total 38,696 100
The largest share (31.9%) is comprised of coordinate questions, which reflects the primary importance of spatial
informationforgeographicalobjects. Questionsaboutobjecttype(20.8%)andlocation(14.8%)togetherformmore
than one third of the corpus. Etymological questions (13.0%) exploit a unique feature of the dataset—the availability of
data on name origins. The smallest share (1.6%) is held by questions about physiographic characteristics, which is
attributable to the fragmentary population of this field in the source records.
TheaveragecontextlengthintheQAcorpusis1250characters;answersvaryfrom2to150characters. Coordinate
answers are the shortest (on average 20 characters), while the longest are answers containing a list of bibliographic
sources(upto500charactersinindividualcases). Anexampleofasinglequestion-answerpairinSQuADformatis
provided below:
{
"id": "1530_coordinates_0",
"context": "Название (рус): Рантамак|Название (тат): Рантамак|
Объект: Село|Этимология: Топоним произошел от
ойконима «Рангазар-Тамак».|Расположение: Расположено
на р. Мелля, в 21 км к востоку от с. Сарманово.|
Источники: Әхмәтьянов Р.Г. Татар теленең этимологик
сүзлеге...|Координаты: 55.205461, 52.881862",
"question": "Какие координаты у Рантамак?",
"answers": [{"text": "55.205461, 52.881862", "answer_start": 312}]
}
The generated corpus was randomly partitioned into training (90%, 34,826 pairs) and validation (10%, 3870 pairs)
setswhilepreservingstratificationbyquestiontype. Bothpartsofthecorpus,aswellasthesourcetoponymdataset,
are openly published on the Hugging Face platform [TatarNLPWorld, c,b] under the CC BY-SA 4.0 license and are
available for use in scholarly and applied purposes.
5 Experimental Study
The experimental validation of the proposed question-answering system was conducted in two stages, corresponding to
itsarchitecturalcomponents. Inthefirststage,theeffectivenessofthehybridretrievalmodule,combiningsemantic
indexing with geospatial filtering, was evaluated; in the second, the ability of the extractive reading component to
accurately localize the answer within the context provided by the retrieval mechanism was assessed. Such two-part
evaluation protocol made it possible, on the one hand, to characterize each subsystem in isolation, and on the other, to
demonstrate their compatibility within a unified pipeline. All experiments were carried out on the corpus of 9,023
coordinate-referenced documents described above and on the synthetic question-answering set of 38,696 examples
generated according to the procedure in Section 4.2.
To evaluatethe retrieval component, anindependent test set wasprepared by randomly selecting500 records from the
sourcedatasetandgeneratingnatural-languagequeriesforthemusingfivetemplates(Что такое {name}?[Whatis
{name}?],Где находится {name}?[Where is {name}?],Расскажи о {name}[Tell about {name}], etc.), in which
the object name was inserted either in Russian or in Tatar with probabilities of 0.7 and 0.3, respectively. This approach
guaranteed that for each query there exists exactly one relevant document—the source record from which the query was
generated. Additionally, a validation set of 200 queries, disjoint from the test set, was allocated; it was used exclusively
for tuning the hybrid search hyperparameters—the weighting coefficient𝛼and the spatial filter radius𝑅.
7

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
Retrievalqualitywasmeasuredbytwomainmetrics: Recall@ 𝑘(𝑘=1,3,5 )andMeanReciprocalRank(MRR).The
formerindicatestheproportionofqueriesforwhichtherelevantdocumentappearsamongthetop 𝑘retrievalresults;the
latter averages the reciprocal of the rank of the first relevant document and is thus sensitive to early hits. To obtain
statisticallysoundconclusions,allmetricswereaccompaniedby95%confidenceintervalsconstructedviabootstrap
with 1,000 resamples with replacement. This enabled not only comparison of average values but also assessment of the
significance of differences between methods.
Four ranking strategies were compared: classical lexical BM25 search, purely spatial search (top- 𝑘nearest by haversine
distance), semantic search based on multilingual-e5-large with a FAISS index, and the proposed hybrid method.
The hybrid search was performed with a fixed radius of 𝑅=50km and𝛼=0.1, selected on the validation set as
yieldingthemaximum Recall@5=1.0 . ThecomparisonresultsaresummarizedinTable4,andFigure1visualizes
them as bar charts with interval estimates.
Table 4: Comparison of search methods (95% bootstrap confidence intervals)
Method Recall@1 Recall@3 Recall@5 MRR
BM25 0.438 [0.394, 0.484] 0.574 [0.528, 0.620] 0.618 [0.572, 0.662] 0.508 [0.469, 0.545]
Spatial only 0.536 [0.492, 0.576] 0.708 [0.668, 0.748] 0.796 [0.760, 0.830] 0.634 [0.598, 0.673]
Semantic only 0.774 [0.736, 0.810] 0.904 [0.878, 0.928] 0.940 [0.918, 0.960] 0.840 [0.813, 0.866]
Hybrid (𝛼=0.1,𝑅=50km) 0.988 [0.978, 0.996] 1.000 [1.000, 1.000] 1.000 [1.000, 1.000] 0.994 [0.988, 0.998]
The figures in Table 4 indicate a substantial superiority of the hybrid scheme. First of all, the value Recall@5=1.000
is striking, meaning that for each of the 500 test queries, the relevant object is guaranteed to be present among the first
five results. Such high recall practically eliminates the risk of losing the necessary information before the reading stage,
whichisacriticalrequirementforRAGsystems. The Recall@1=0.988 scoreindicatesthatonlyinsixcasesoutoffive
hundred did the target document fail to occupy the first position; moreover, the lower bound of the confidence interval
(0.978)liessubstantiallyabovetheupperboundsof Recall@1 forallcompetingmethods. Thisarrangementofintervals
confirms the statistical significance of the hybrid’s superiority over the alternatives.
Semantic search by itself demonstrates fairly high quality ( Recall@1=0.774 ), confirming the ability of the
multilingual-e5-large modeltoadequatelyencodemultilingualtoponymiccontextsandcapturesemanticproximity
between different name variants. However, the absence of spatial filtering allows objects with similar names but located
faraparttoappearintoppositions,whichreducesaccuracybyapproximately0.2comparedtothehybrid. Spatialsearch,
whichignoresthequerytext, shows Recall@1=0.536 . Thisisexpected: withina50 kmradius, severalobjects are
oftenfound, and thegeographicallynearest onedoes notalways coincidewith thesought one. Nevertheless,even this
simplestrategyoutperformsBM25( Recall@1=0.438 ),underscoringthefundamentalroleofgeographicproximityfor
toponymicqueries. BM25’smostmodestresultsareexplainednotonlybythelexicalgapbetweenTatarandRussian
namesbutalsobythesynonymyofgeographicalterms: aclassicalinvertedindexisunabletoequateсело[selo]and
деревня[village],река[river] andприток[tributary].
A detailed picture is presented in Figure 1, where the Recall@1 and Recall@5 bars for each method are supplemented
withconfidenceintervals. Thehybridmethodstandsoutnotonlyfortheheightofitsbarsbutalsofortheirminimal
variability, indicating high stability of the algorithm.
As a next step, we analyzed the performance of the hybrid approach broken down by toponym type. The source dataset
is divided into two main categories:Топоним[Toponym] (large objects: populated places, significant natural features,
etc.) andМикротопоним[Microtoponym] (local landscape elements: meadows, fields, natural tracts, springs). In
thetestset,toponymsaccountedfor349queriesandmicrotoponymsfor151. Theresultsoftheseparateanalysisare
presented in Table 5.
Table 5: Recall@1 of the hybrid method by toponym type
Toponym Type Number of Queries Recall@1
Toponym 349 0.986
Microtoponym 151 0.993
Both categories demonstrate near-ceiling values; however, microtoponyms show a slightly higher result. This difference
is explainedbytheir localnature: forsmall objects, thedensity oftoponymswithin a50 km radiusis generally lower,
meaning the spatial filter leaves fewer candidates, and the semantic component finds it easier to identify the sole correct
one. For comparison, Figure 2 presents a bar chart that clearly illustrates this distinction.
8

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
Figure1: ComparisonofRecall@1andRecall@5 with95%confidenceintervalsforBM25,purely spatial,semantic,
and hybrid methods.
Figure 2: Recall@1 of the hybrid method for the “Toponym” and “Microtoponym” categories.
Despite the impressive aggregate metrics, the hybrid retrieval made six errors in which the relevant document did
not appear in the first position. Manual analysis of these cases, undertaken to identify root causes, showed that all
of them are related not to shortcomings of the combination algorithm but to the quality of the data themselves. In
twocases,completenamesakeswerepresentwithina50kmradius—twodifferentobjectswithidenticalnames(e.g.,
two natural tracts namedКрасная Горка[Krasnaya Gorka]). Their contexts proved practically identical, and the
semantic component could not give preference to one over the other. In three cases, the coordinates of the target object
were offset by 10–15 kilometers from the true position, due to which it did not fall within the search zone and was
eliminatedatthegeofilteringstage. Inonefurthercase,the context fieldcontainedaminimumofinformation(only
nameandtype),whichyieldedapredictablylowsemanticscoreandallowedanother,moreinformativecandidateto
takethelead. Consequently,furtherimprovementoftheretrievalcomponentliesprimarilyinenhancingdataquality:
verifyingcoordinatesthroughcross-validationwithOpenStreetMaporsatelliteimagery,enrichingcontextualfields,and
developing a component for resolving toponymic homonymy.
Turning to the second stage, extractive reading, we note that the task here was to precisely locate the answer within
the context retrieved by the retrieval module. Three architectures, representing both monolingual and multilingual
approaches, were selected for training and evaluation: RuBERT base, RuBERT large, and XLM-RoBERTa large, each
fine-tuned on the synthetic QA corpus of 34,826 pairs. The generative model T5-RUS, unfortunately, could not be
9

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
trained due to tokenizer version incompatibility and was used only in its base pre-trained variant as an additional
reference point. Additionally, as a simple baseline, a heuristic rule was implemented that extracts the answer based on
question keywords and the corresponding prefixes in the context. The training hyperparameters, common to all models,
are presented in Table 6.
Table 6: Hyperparameters for QA model fine-tuning
Parameter Value
Maximum sequence length 384 tokens
Stride 128 tokens
Training batch size 4
Evaluation batch size 8
Learning rate3×10−5
Number of epochs 3
Weight decay 0.01
Warm-up steps 500
Optimizer AdamW
The results obtained on the validation set of 3,870 QA pairs uncovered an important nuance that substantially affected
interpretation. It turned out that the RuBERT models, operating with the WordPiece tokenizer, tend to insert extraneous
spacesinside numericalcoordinates (“55. 175195”instead of“55.175195”) andhyphenated constructions(“северо -
западу”[“north - west”]). This is a purely tokenization artifact unrelated to the semantic capability of the model. By
applying elementary post-processing (removal of breaks within floating-point numbers and unification of hyphens and
brackets),weensuredthatRuBERTanswersbecameidenticaltothereferenceones. Table7presentsbothrawExact
Match and F1 metrics and the values obtained after normalization.
Table 7: QA model results on the validation set (raw and normalized metrics)
Model EM (raw) F1 (raw) EM (norm) F1 (norm) Time, ms
xlm_roberta_large 0.992 0.994 0.992 0.994 22.4
rubert_base 0.402 0.684 1.000 1.000 6.6
rubert_large 0.398 0.679 1.000 1.000 6.5
xlm_roberta_base (w/o fine-tun.) 0.440 0.574 – – 22.4
rubert_base (w/o fine-tun.) 0.068 0.187 – – 6.6
rule_based 0.492 0.492 – –≈0
t5_rus_base (w/o fine-tun.) 0.176 0.220 – – 27.2
ThemultilingualXLM-RoBERTalargeinitiallydeliversvirtuallyperfectanswers( EM=0.992 )andrequiresnopost-
processing. Both RuBERT models, after minimal normalization, achieve exact match with the reference ( EM=1.000 ),
while operating 3.5 times faster—6.5 ms versus 22.4 ms per query. This speed advantage makes RuBERT the preferred
choiceforproductionsystems,providedathinpost-processinglayerisadded. Figure3presentsacomparisonofmodels
in terms of EM and F1 with 95% bootstrap confidence intervals.
Foradeeperunderstandingofcapabilitiesandlimitations,weconductedaper-categoryqualityanalysis,presentedin
Table 8. After normalization, all three fine-tuned models achieve 100% F1 on questions about etymology, location,
region,andobjecttype. TheonlynoticeabledeviationisobservedforXLM-RoBERTalargeoncoordinates( F1=0.984 ),
which is associated with rare cases of inaccurate copying of the last digit of a decimal fraction.
Table 8: F1 score by question type after normalization
Model Coordinates Etymology Location Region Sources Object Type
xlm_roberta_large 0.984 1.000 1.000 1.000 0.988 1.000
rubert_base 1.000 1.000 1.000 1.000 1.000 1.000
rubert_large 1.000 1.000 1.000 1.000 1.000 1.000
Ontheheatmap(Figure4),thisresultisvividlyreflected: allcells,withtheexceptionofone,arecoloredinthewarmest
tones, corresponding to an F1 of unity.
10

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
Figure3: ComparisonofmodelsbyExactMatchandF1scoreafternormalization(errorbarsrepresent95%confidence
intervals).
Figure 4: Heatmap of F1 score by question type for fine-tuned models (after normalization). Black cells are absent,
demonstrating the complete resolution of the coordinate problem for RuBERT.
11

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
The experimental study is concluded by the analysis of answer length and inference time. The distribution of predicted
answer lengths for the XLM-RoBERTa large model is practically identical to the reference (Figure 5). The bulk of
answers is concentrated in the 20–40 character range, corresponding to coordinates and short names; there is also
a small peak around 200 characters, corresponding to bibliographic sources. For RuBERT after normalization, the
distribution shape coincides with the reference without significant deviations.
Figure5: Distributionofanswerlength(incharacters)forreferencevaluesandpredictionsofthexlm_roberta_large
model.
Speedmeasurementsshowed(Figure6)thatRuBERTspendsonlyabout6.5msperquery,XLM-RoBERTalargespends
22.4 ms, and the generative T5 spends 27.2 ms. Thus, for tasks where latency is critical, RuBERT with post-processing
appears to be the optimal alternative, whereas XLM-RoBERTa large may be preferable when maximum quality of
numerical coordinate extraction is required without additional programming.
Figure 6: Average processing time per example (ms) for the considered QA models.
Thus,thetotalityoftheexperimentaldataconvincinglydemonstratesthattheproposedtwo-componentarchitecture
achieves near-ideal performance at all stages—from retrieval of the relevant document to precise extraction of the
answer—andtheidentifiedtokenizationpeculiaritiesofRuBERTnotonlyexplainpreviousobservationsbutalsoprovide
a simple recipe for their complete elimination.
12

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
6 Discussion
Theconductedexperimentalstudycoveredtwokeycomponentsoftheproposedquestion-answeringarchitecture—hybrid
retrieval and extractive reading—and yielded a series of interrelated results that are expedient to discuss in aggregate.
The most important outcome of the retrieval part is the demonstration that the combination of semantic search based on
denseembeddingsfrom multilingual-e5-large andgeospatialfilteringiscapableofdeliveringvirtuallyperfect
recall for toponymic queries at the scale of a single region. The values Recall@5=1.000 andMRR=0.994 , achieved
on an independent test set of 500 queries, not only significantly surpass traditional BM25 and purely spatial search but
also leave minimal room for further improvement. The lower bound of the hybrid Recall@1 confidence interval (0.978)
lies above the upperbounds of theanalogous intervals forall alternatives, confirmingthe statisticalsignificance of the
superiority and attesting to the stability of the method.
Analysis of the optimal value of the weighting coefficient 𝛼=0.1, obtained via grid search on the validation set, shows
that in the task of toponym search, geographic proximity is the dominant relevance feature. The spatial filter with a
50kmradiusreducesthecandidatesetfromseveralthousandtoseveraldozen,afterwhichevenasmallcontribution
from the semantic component suffices for accurate identification of the target object. This conclusion accords well
withintuitionbuthadnotpreviouslybeenconfirmedquantitativelyonmultilingualtoponymicmaterialenrichedwith
etymological data. It is noteworthy that microtoponyms demonstrate a marginally higher Recall@1 compared to larger
toponyms(0.993vs.0.986). Thereason,apparently,liesintheirlocalcharacter: withina50kmradiusaroundsmall
natural tracts or springs, fewer candidate objects are found, which further eases the task for the semantic component.
Manual analysis of the six retrieval errors in which the relevant document did not occupy the first position revealed that
all of them are attributable to the quality of the source data rather than algorithmic shortcomings. Three out of six
cases are associated with coordinate inaccuracy—objects were offset by 10–15 km from their true position and fell
outsidethesearchradius. Twofurthercasesinvolvedcompletenamesakeswithinthesameradius,andoneinvolved
insufficient contextual description. This observation has a direct practical implication: further improvement of retrieval
qualityrequires,firstandforemost,notmodelcomplicationbutcoordinateverification(e.g.,throughcross-validation
withOpenStreetMaporsatelliteimagerydata),enrichmentofcontextualfields,andthedevelopmentofamodulefor
resolving toponymic homonymy.
The results obtained for the extractive reading component are no less indicative, and in some aspects, unexpected.
Firstofall,thecriticalimportanceofdomain-specificfine-tuningwasconfirmed: thebaseversionsofbothRuBERT
andXLM-RoBERTa large,without adaptationtothesynthetic QAcorpus,demonstrateunacceptablylowquality ( F1
does not exceed 0.574), whereas after three epochs of fine-tuning, the metrics reach near-ceiling values. This contrast
underscores the value of the created question-answering resource, which contains patterns absent from general QA
collections such as SQuAD and SberQuAD.
A noteworthy resultwas the elucidationof thenature ofRuBERT’s“failure” ongeographic coordinatequestions. The
initially recorded zero F1on this category was misleading and could have created the impression of a fundamental
inability of monolingual WordPiece models to process numerical information. Detailed analysis showed that the
problem is of a purely tokenization nature: RuBERT splits coordinates into tokens with spaces (“55. 175195”), which
hinders accurate extraction. Elementary post-processing—removal of extraneous spaces within floating-point numbers
andunificationofhyphens—completelyeliminatestheartifact,afterwhichbothversionsofRuBERTachieve100%
accuracyonallquestioncategories,includingcoordinates. Moreover,RuBERT’sinferencespeed(approximately6.5
ms per query) is 3.5 times that of XLM-RoBERTa large (22.4 ms). Thus, the widespread notion that multilingual
SentencePiecemodelsarenecessaryforworkingwithnumericaldatarequiresrefinement: whenlightpost-processingis
permissible, monolingual models may be preferable due to gains in speed and lower memory consumption.
The synergy of the two components—retrieval and reader—merits a separate comment. The achieved metrics of
Recall@5=1.000 at the search stage and EM≈1.000 at the reading stage (with post-processing) mean that, within
the model setting where the query is generated from one of the dataset records, the system is capable of errorlessly
finding the relevant document and extracting the exact answer. In the context of RAG systems, this property is critically
important: the generative model receives a correct context with a probability tending to unity, which radically reduces
the risk of hallucinations. Although an end-to-end retrieval-plus-reader experiment was not conducted in this work, the
separate metrics provide an upper-bound estimate of the full pipeline’s quality and allow the developed components to
be regarded as ready for integration.
It is necessary to note a number of limitations inherent to the present study. First, the test queries for both components
were generatedfrom templatesanddo notreflect thefulldiversityofformulationscharacteristicof realusers. This is
standardpracticefortheinitialstagesofcreatingspecializeddatasets,butitleavesopenthequestionofthesystem’s
behavioronlivequeries. Second,thegeospatialfilteringradiusisfixedat50km;forobjectsofdifferentscales(aspring
versusamountainrange),suchavaluemaybesuboptimal. Third,thegenerativemodelT5-RUScouldnotbetrained
13

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
due to technicalreasons, so the comparison of extractionand generation paradigms remained incomplete. In addition,
questions about physiographic characteristics constituted only 1.6% of the QA corpus, which is insufficient for reliable
conclusions, and this category deserves further supplementation.
Thepracticalsignificanceoftheperformedworkisdeterminednotonlybytheachievedmetricsbutalsobythecomplete
cycle of open publication of artifacts. The source toponym dataset [TatarNLPWorld, c], the question-answering corpus
[TatarNLPWorld, b], the weights of the trained models (including both versions of RuBERT and XLM-RoBERTa large),
and the interactive web demo [TatarNLPWorld, d] are hosted on Hugging Face and are ready for use in geoinformation
services, educational platforms, and digital humanities projects. The openness of the resources ensures reproducibility
andenablesotherresearcherstoadvancethedirection,inparticular,toadapttheproposedapproachtootherregionsand
language pairs.
Promising directions for further research flow directly from the discussed limitations. These include: the development
of adaptive strategies for selecting the geofiltering radius depending on the object type or local toponym density;
cross-validation of coordinates against external sources to minimize positioning errors; conducting an end-to-end
retrieval-plus-readerexperimentinafull-fledgedRAGsettingwithmeasurementoffinalanswerquality;fine-tuning
generative models (such as T5) on the created QA corpus for direct comparison with the extractive approach; as
wellasthe applicationofparameter-efficientfine-tuning methods(LoRA/QLoRA),whichhave alreadydemonstrated
promiseforthecloselyrelatedBashkirlanguage[ArabovandKhaybullina,2026]. Addressingtheenumeratedtaskswill
enable the transition from a demonstration prototype to a production-grade question-answering system for multilingual
geographic data.
7 Conclusion
In this work, a comprehensive question-answering system for multilingual toponymic data has been presented and
experimentallysubstantiated,encompassingthestagesofresourcecollectionandstructuring,hybridretrievalofrelevant
documents, and extractive reading. The conducted research has yielded the following main scientific and practical
results.
1.An original bilingual dataset of Tatarstan toponyms has been collected, verified, and openly published,
comprising9,688records,ofwhich9,023arefurnishedwithgeographiccoordinatesandcontainextensive
linguistic,etymological,andadministrativeinformation. Basedonthisdataset,aspecializedquestion-answering
corpus of 38,696 “question–context–extractable answer” triples was automatically generated with guaranteed
answer localization and annotated in the SQuAD format [TatarNLPWorld, c,b].
2.A hybrid retrieval method has been proposed and implemented, combining semantic indexing with the
multilingual model multilingual-e5-large and geospatial ranking using KD-trees and haversine distance.
It has been shown that, with the optimal weighting coefficient 𝛼=0.1and a radius of 50 km, the method
achieves Recall@1=0.988 ,Recall@5=1.000 , andMRR=0.994 on 500 test queries, statistically
significantly outperforming BM25, purely semantic, and purely spatial search.
3.A multi-architecture benchmark of extractive reading models (RuBERT-base, RuBERT-large, XLM-RoBERTa-
large) trained on the synthetic QA corpus has been conducted. The multilingual XLM-RoBERTa-large
demonstrates near-perfect Exact Match (0.992) and F1 (0.994) without any additional processing. It has been
establishedthattheinitiallyrecordedcompletefailureoftheRuBERTmodelsoncoordinatequestions( F1=0)
is attributable to WordPiece tokenization artifacts and is completely eliminated by trivial post-processing, after
which both versions of RuBERT achieve 100% accuracy on all question categories with an inference speed 3.5
times higher than that of XLM-RoBERTa-large.
4.All resources created in the course of this work—the source dataset, the QA corpus, the weights of the
trained models, and the interactive web application—are hosted in open access on the Hugging Face platform
[TatarNLPWorld, c,b,d], ensuring full reproducibility of the results and the possibility of their immediate
practical use.
The practical significance of the developed system lies in its readiness for integration into geoinformation services,
digital archives, educational platforms, and projects for the preservation of the cultural heritage of multilingual regions.
The achieved search recall and answer extraction accuracy metrics guarantee that a generative model within a RAG
pipeline will receive relevant context in virtually all cases, which critically reduces the risk of hallucinations.
Furtherresearchwillbedirectedtowardtheimplementationofadaptivegeofilteringradiusselection, cross-validationof
coordinates against external sources, end-to-end evaluation of a complete RAG pipeline involving generative models, as
well as the expansion of the dataset with open-ended questions. The proposed approach can be scaled to other regions
and language pairs, provided that toponymic datasets of comparable structure with coordinate referencing are available.
14

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
References
D. Sh. Suleymanov, A. Ya. Fridman, R. A. Gilmullin, and B. A. Kulik. System analysis of the problem of natural
language modeling.Transactions of the Kola Science Centre. Information Technologies, 12(5):57–66, 2021. (In
Russian).
M. Galimov, R. Burnashev, and A. Gatiatullin. Designing a prototype of a fuzzy expert system for a dialectologist using
geographic information systems and technologies. InProceedings of the 8th International Conference on Computer
Science and Engineering (UBMK), pages 382–386, Burdur, Turkiye, 2023.
R.RehurekandP.Sojka. Gensim–pythonframeworkforvectorspacemodelling. InProceedingsofthe9thInternational
Conference on Text, Speech and Dialogue (TSD), pages 45–50, Brno, Czech Republic, 2006.
M.R.Saykhunov,R.R.Khusainov,T.I.Ibragimov,etal. Writtencorpusofthetatarlanguage. [Electronicresource].
URLhttps://www.corpus.tatar/en. Accessed: 03.01.2026.
Tatar mixed corpus – tatar corpus from the web. Sketch Engine. [Electronic resource]. URL https://www.
sketchengine.eu/tatar-corpus-from-the-web/. Accessed: 03.01.2026.
IPSAN. tat_monocorpus_v2. Hugging Face. [Electronic resource]. URL https://huggingface.co/datasets/
IPSAN/tat_monocorpus_v2/. Accessed: 28.02.2026.
R.A.GilmullinandR.R.Gataullin. Morphologicalanalysissystemofthetatarlanguage. InProceedingsofthe9th
International Conference on Computational Collective Intelligence (ICCCI), pages 519–528. Springer, Cham, 2017.
M.K.Arabov. Tatartokenizers. CertificateofStateRegistrationofComputerProgramNo.2026611049dated16.01.2026,
2026a. (In Russian).
D. R. Mukhamedshin, A. R. Gatiatullin, N. A. Prokopyev, and R. A. Gilmullin. Semantic annotation in electronic
corpus of tatar language ‘tugan tel’ based on knowledge graph. InProceedings of the 10th International Conference
on Computer Science and Engineering (UBMK), pages 1792–1795, Istanbul, Turkiye, 2025.
A. Mindubaev and A. R. Gatiatullin. Problems of semantic relation extraction from tatar text corpus ‘tugan tel’. In
Proceedingsofthe3rdInternationalConferenceonProblemsofInformatics,ElectronicsandRadioEngineering
(PIERE), pages 1690–1694, Novosibirsk, Russian Federation, 2024.
M.K.Arabov. Tatar2vec. CertificateofStateRegistrationofComputerProgramNo.2026610619dated14.01.2026,
2026b. (In Russian).
F. M. Gafarov, V. R. Gafarova, and M. M. Ayupov. Explainable artificial intelligence methods in text classification
ofmachinelearningmodelsfortatarlanguage. InProceedingsofthe10thInternationalConferenceonComputer
Science and Engineering (UBMK), pages 1796–1800, Istanbul, Turkiye, 2025.
P. Liang, M. Zhang, et al. Embeddings from bidirectional encoder representations (e5): A universal embedding model
for information retrieval. arXiv preprint arXiv:2212.03533, 2022.
A.Conneau,K.Khandelwal,N.Goyal,V.Chaudhary,G.Wenzek,F.Guzmán,E.Grave,M.Ott,L.Zettlemoyer,and
V. Stoyanov. Unsupervised cross-lingual representation learning at scale. InProceedings of the 58th Annual Meeting
of the Association for Computational Linguistics (ACL), pages 8440–8451, Online, 2020.
Geonames. [Electronic resource]. URLhttps://www.geonames.org/. Accessed: 28.02.2026.
TatarNLPWorld. Turkic nlp & low resource languages research hub. Hugging Face. [Electronic resource], a. URL
https://huggingface.co/TatarNLPWorld. Accessed: 28.02.2026.
S.SchweterandS.Tekir. Bertforturkish: Acomprehensiveevaluation. InProceedingsofthe12thLanguageResources
and Evaluation Conference (LREC), pages 4325–4332, Marseille, France, 2020.
J. L. Bentley. Multidimensional binary search trees used for associative searching.Communications of the ACM, 18(9):
509–517, 1975.
R. W. Sinnott. Virtues of the haversine.Sky and Telescope, 68(2):159, 1984.
R. Burnashev, A. Gatiatullin, and M. Khamidullin. Geolinguistic system for dialect similarity analysis based on
associative rules and fuzzy logic. InProceedings of the 10th International Conference on Computer Science and
Engineering (UBMK), pages 1782–1785, Istanbul, Turkiye, 2025.
P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. Squad: 100,000+ questions for machine comprehension of text.
InProceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages
2383–2392, Austin, Texas, 2016.
S.Efimovetal. Sberquad–russianreadingcomprehensiondataset. InProceedingsofthe22ndConferenceonArtificial
Intelligence (DCAI), 2020.
15

Tatarstan Toponyms: Dataset and RAG SystemA Preprint
A.Fenogenova,M.Tikhonova,V.Mikhailov,T.Shavrina,A.Emelyanov,D.Shevelev,A.Kukushkin,V.Malykh,and
E. Artemova. Russian superglue 1.1: Revising the lessons not learned by russian nlp models. InComputational
Linguistics and Intellectual Technologies: Papers from the Annual International Conference "Dialogue", number 20,
2021.
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for
languageunderstanding. InProceedingsofthe2019ConferenceoftheNorthAmericanChapteroftheAssociation
for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 4171–4186, Minneapolis,
Minnesota, 2019.
E.Artemova,M.Zmeev,N.Loukachevitch,I.Rozhkov,T.Batura,V.Ivanov,andE.Tutubalina. Runne-2022shared
task: Recognizing nested named entities. InProceedings of the Dialogue 2022 Conference, Moscow, 2022. RSUH.
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of
transfer learning with a unified text-to-text transformer.Journal of Machine Learning Research, 21:1–67, 2020.
A.A.Choure,R.B.Adhao,andV.K.Pachghare. Nerinhindilanguageusingtransformermodel: Xlm-roberta. In2022
IEEEInternationalConferenceonBlockchainandDistributedSystemsSecurity(ICBDS),pages1–5,Pune,India,
2022. doi:10.1109/ICBDS53701.2022.9935841.
J.Johnson,M.Douze,andH.Jégou. Billion-scalesimilaritysearchwithgpus.IEEETransactionsonBigData,7(3):
535–547, 2019.
TatarNLPWorld. Tatarstantoponymsqadataset. HuggingFace.[Electronicresource],b. URL https://huggingface.
co/datasets/TatarNLPWorld/tatarstan-toponyms-qa. Accessed: 28.02.2026.
TatarNLPWorld. Tatarstantoponymsdataset. HuggingFace.[Electronicresource],c. URL https://huggingface.
co/datasets/TatarNLPWorld/tatarstan-toponyms. Accessed: 28.02.2026.
TatarNLPWorld.Toponymicragexplorer.HuggingFaceSpaces.[Electronicresource],d.URL https://huggingface.
co/spaces/TatarNLPWorld/tatar-toponym-rag-space. Accessed: 28.02.2026.
M. K. Arabov and S. S. Khaybullina. Adapting large language models to a low-resource agglutinative language: A
comparative study of lora and qlora for bashkir. arXiv preprint arXiv:2605.04948, 2026.
16