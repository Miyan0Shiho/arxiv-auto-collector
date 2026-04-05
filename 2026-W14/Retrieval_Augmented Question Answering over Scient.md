# Retrieval-Augmented Question Answering over Scientific Literature for the Electron-Ion Collider

**Authors**: Tina. J. Jat, T. Ghosh, Karthik Suresh

**Published**: 2026-04-02 16:52:48

**PDF URL**: [https://arxiv.org/pdf/2604.02259v1](https://arxiv.org/pdf/2604.02259v1)

## Abstract
To harness the power of Language Models in answering domain specific specialized technical questions, Retrieval Augmented Generation (RAG) is been used widely. In this work, we have developed a Q\&A application inspired by the Retrieval Augmented Generation (RAG), which is comprised of an in-house database indexed on the arXiv articles related to the Electron-Ion Collider (EIC) experiment - one of the largest international scientific collaboration and incorporated an open-source LLaMA model for answer generation. This is an extension to it's proceeding application built on proprietary model and Cloud-hosted external knowledge-base for the EIC experiment. This locally-deployed RAG-system offers a cost-effective, resource-constraint alternative solution to build a RAG-assisted Q\&A application on answering domain-specific queries in the field of experimental nuclear physics. This set-up facilitates data-privacy, avoids sending any pre-publication scientific data and information to public domain. Future improvement will expand the knowledge base to encompass heterogeneous EIC-related publications and reports and upgrade the application pipeline orchestration to the LangGraph framework.

## Full Text


<!-- PDF content starts -->

Prepared for submission to JINST
Retrieval-Augmented Question Answering over
Scientific Literature for the Electron-Ion Collider
Tina. J. Jat1, T. Ghosh1∗, Karthik Suresh2
1Ramaiah University of Applied Sciences, India
2College of William & Mary, USA
E-mail:tapasi03@gmail.com
Abstract: To harness the power of Language Models in answering domain specific specialized
technicalquestions,RetrievalAugmentedGeneration(RAG)isbeenusedwidely. Inthiswork,we
havedevelopedaQ&AapplicationinspiredbytheRetrievalAugmentedGeneration(RAG),which
iscomprisedofanin-housedatabaseindexedonthearXivarticlesrelatedtotheElectron-IonCol-
lider(EIC)experiment-oneofthelargestinternationalscientificcollaborationandincorporatedan
open-source LLaMA model for answer generation. This is an extension to it’s proceeding applica-
tionbuiltonproprietarymodelandCloud-hostedexternalknowledge-basefortheEICexperiment.
This locally-deployed RAG-system offers a cost-effective, resource-constraint alternative solution
to build a RAG-assisted Q&A application on answering domain-specific queries in the field of ex-
perimentalnuclearphysics. Thisset-upfacilitatesdata-privacy,avoidssendinganypre-publication
scientific data and information to public domain. Future improvement will expand the knowledge
basetoencompassheterogeneousEIC-relatedpublicationsandreportsandupgradetheapplication
pipeline orchestration to the LangGraph framework.
Keywords:Electron-IonCollider(EIC),Retrieval-AugmentedGeneration(RAG),LargeLanguage
model Meta AI (LLaMA), Retrieval-Augmented Generation Assessment(RAGAS)arXiv:2604.02259v1  [hep-ex]  2 Apr 2026

Contents
1 Introduction 1
2 Previous Work 2
3 Methodology and Application Architecture 2
4 Results Analysis and Discussion 4
4.1 Benchmark Dataset 4
4.2 Evaluation and result analysis 5
5 Conclusion 8
1 Introduction
In the era of Artificial Intelligence(AI), Natural Language Processing(NLP) and Language Mod-
els(LM)arethedomainsthatencodetheessenceofthemeaningoflanguagebasedonitssurrounding
or context to facilitate AI-driven reasoning and decision making tasks. Any advancement in this
domainhascreatedanenormousimplicationstothehumanlivesacrossdomainsincludingscience,
technology,commerceandsoon. Asaconsequence,anybiasorconfidentsoundingfactuallyincor-
rect generated outcome imputed in those applications will create a domino effects across domains
amongthevariousstakeholders. Thepresentstate-of-theartLanguageModelseitherproprietaryor
open-source which are trained on exorbitantly large amount of text available in the internet exhibit
this problem of generating fluent, confident but factually incorrect responses popularly known as
hallucination [1, 2]. In 2021, Meta research team has introduced the concept of Retrieval Aug-
mented Generation aka RAG [3] to mitigate hallucination and make the LMs grounded. In our
research work we have implemented this methodology to build a Question-Answering application
to generate reliable, scientifically accurate answer for the Electron-Ion Collider(EIC) [4] experi-
ment. EIC being a multi-continental, multi-institutional projects usually produce large amount of
scholarly articles in the form of research publication, conference/workshop presentation, working
group meeting presentations, technical design report etc. A large scale international collaboration
with more than 190 participating institutes across the world provides an excellent opportunity to
deploy such an AI-based application to facilitate smooth, time-efficient on-boarding to the newly
joined collaborators as well as the seasoned researchers to extract vital scientific and key research
objectives/milestones of the experiment along with detailed description of user-specific questions
from various scientific details of the theory, simulation, detectors, hardware specification catering
the needs of the user.
With this ambition the AI4EIC [5] built the foundational work on RAG Q&A system with
proprietarymodelsandCloud-basedstorageforexternalknowledge-base. Inthispresentwork,we
– 1 –

have further extended this application using cost-effective open-source Language Models.Another
majoradvancementistheuseofin-housedatabasetoindextheEIC-relatedarticlesextractedfrom
arXivrepository. ThegeneratedanswersaretracedbacktotheoriginalarXivarticlesfromwhichthe
context is being retrieved. Emphasis is given on the authenticity of this citation mechanism which
isfacilitatedthroughLangSmith[6],anobservabilityplatformforLLMapplications,allowingany
interested reader to dive deeper into the subject through those cited references.
This paper is organized in the following way: a brief review of the previous work is presented
in Section 2, the methodology adopted is detailed in Section 3, results and evaluation performance
are reported in Section 4 and finally conclusions along with a brief road-map for future work are
provided in Section 5.
2 Previous Work
ThepresentdevelopmentontheRAG-basedQ&AapplicationfortheEICprojecthasbroadenedthe
foundationalwork[5]. Inthesaidapplication,abenchmarkQ&Adatasetwithgoldstandardisbeing
createdusingtheOpenAIGPT4.0model. Thedatasetcontainsspecificsetsof51questions,andfor
eachquestionthereisassociatedsub-questionsreferredas"claims",answersagainsteachclaimand
a comprehensive response. These questions are referenced from EIC-related articles from arXiv
repository published after 2021. More details about the strategy and the quality of this dataset can
befoundin[5]. ThepipelinealsofacilitatesgenerationofnewQ&Aannotationsbyuserthrougha
web-basedchatbotinterface. Inthiswork,theauthorsusedOpenAI’stext-embedding-ada-002
model toembed context andPinecone [7] tostore the embeddingsinto a knowledgebase. OpenAI
GPT4.5 model generated the answer. The entire RAG pipeline is orchestrated by LangChain
framework and traced by the LangSmith, including the citations of resources.
3 Methodology and Application Architecture
TheQ&AapplicationimplementsaRAG-pipelineconsistingoffivemajorsteps: contextingestion
and pre-processing, chunking, embedding generation, retrieval via similarity search and LLM-
conditionedanswergenerationencodedthroughprompttemplate. Theschematicofthearchitecture
is demonstrated in Figure 1.
The knowledge base of this RAG-inspired Q&A application is constructed with 178 EIC-
related research articles from the arXiv preprint repository. These scholarly articles span research
domains across phenomenology, software development, detector design, accelerator physics etc.
To enhance inference and retrieval fidelity, several metadata associated with each article such as
arXiv ID, authors, year of publication etc, are concatenated with chunked text, then embedded and
stored in the vector database. This additional information helps the model differentiate between
semanticallysimilarchunksfromdifferentsources,improvingcontextualaccuracyandtraceability.
This approach allows the system to refer to the correct citations rather than giving isolated or
ambiguous text snippets.
The selected arXiv articles are divided into fixed-sized smaller text segments, known as
"chunks" [8].RecursiveCharacterTextSplitterfrom LangChain [9] splits each document
into "chunks" of lengths 120 and 180 characters with a 20 character overlap between consecutive
– 2 –

chunks. This overlap preserves semantic continuity across chunks and mitigate any artificial
fragmentation during inference. We followed the chunking philosophy of the earlier work [5],
howeveradditionallycross-validatedwithchunksizese.g. 120and180characters. Determiningthe
optimalchunksizeisanimportantfactorinelevatingtheperformanceofaRAG-basedapplication,
since smaller chunks may increase precision but risk missing essential information, while larger
chunks capture more content but may compromise relevance [10].
Subsequently,eachofthesechunksareembeddedintoa1024dimensionaldensevectorrepre-
sentationusingthemxbai-embed-largemodelprovidedbyMixedbreadAI[11].mxbai-embed-large
is a transformer-based model that facilitates local deployability, has a strong performance record
in the Massive Text Embedding Benchmark(MTEB) [12] and no API-dependency. The external
knowledgebaseissavedintoapersistencestoragesystem. Severaldatabaseoptionswereexplored,
including FAISS [13], Pinecone [7], LanceDB [14], and ChromaDB [15], to efficiently store the
embedded chunks. While FAISS offers high-performance similarity search; Pinecone provides a
managed, cloud-based service with scalability, ChromaDB was chosen due to its combination of
practical advantages of data privacy through local deployment and seamless integration into the
LangChainorchestrationframework,simplifyingtheconstructionandimplementationoftheRAG
pipeline.
The retrieval stage extract the most semantically aligned documents related to the user’s
query. Theusers’queryisencodedintoa1024-dimensionalvectorusingthemxbai-embed-large
embedding model, the same dimension as the indexed documents stored in the vector database.
The retrieval stage seeks to retrieve documents that best match the intent of the user’s query. The
effectivenessoftheretrieveddocumentsheavilydependsonboththequalityoftheembeddingsand
thestrategyadoptedtoencodethesemanticsimilarity,astheydirectlyinfluencethemostpertinent
documents. Although many similarity metrics have been proposed in prior research, this study
adopted two strategies: cosine similarity and Maximum Marginal Relevance(MMR) [16].
CosineSimilaritymeasurestheangularalignmentbetweentwonon-zeroembeddingvectorsby
thecosineoftheanglebetweenthem[17]. Theanglebetweenthemcapturessemanticsimilarityand
themostsimilarretrievedcontextsattainsthetopscores. However,itisinvarianttothemagnitudeof
the embedding vectors which may encode information such as importance or confidence, enabling
toproduceredundantresults[18]. MMRextendsthisbybalancingrelevanceanddiversity: afteran
initial cosine-based retrieval, it iteratively selects documents that are both relevant and minimally
similar to previous chosen ones [16], which reduces the redundancy and improves the diversity of
the retrieved content [19].
Top20retrievedchunksareconcatenatedintoapromptandpassedthrougheitherLLaMA3.2
or LLaMA 3.3 model, which are deployed on-premises without any API-dependency. The prompt
providesguardrailstorestrictanswergenerationgroundedontheprovidedcontextonlyandrefrain
fromanyerroneousresponses. AcitationtracingmechanismisorchestratedthroughLangSmith[6],
whichtracesthegeneratedanswerstospecificarXivarticlesfromwhichtherelevantandsupporting
contextisretrieved,therebygroundingtheanswers. LangSmithfacilitatestherecordingandtracing
of each step of the inference pipeline: the user query, retrieved content along with the associated
metadata, the prompt which propagated these information to the LMs and the generated answer.
This allows the Q&A system system into a transparent research assistance.
FinallythegeneratedanswersareevaluatedbytheRAGassessmentmetricsandthedetailsare
– 3 –

elaborated in 4.
Figure 1. The schematic of the Q&A System Design for the EIC. The pipeline consists of article ingestion,
documentchunking,vectorembedding,ChromaDBindexing,retrieval,formationofpromptingtopropagate
theinformationandresponsegeneration. First,aqueryisencodedintoadense-vectorrepresentationduring
thedataingestionphase;subsequentlysimilaritysearchisinitiatedbetweentheembeddedqueryandlocally
deployedvectorizeddatabasetoretrievethemostrelevantcontexts. Theseretrievedchunksaresubsequently
mergedwiththequerythroughacarefullydesignedprompttemplateandpassedtoalanguagemodel,which
finally generates a contextually grounded response.
4 Results Analysis and Discussion
ToinferoftheperformanceoftheRAG-system,thegeneratedanswersareevaluatedbycomparing
withanexpertcurateddataset,acollectionoftheEIC-relatedquestionsandtheiroptimalsolutions.
4.1 Benchmark Dataset
TheAI4EIC2023_DATASETSisthehighqualitybenchmarkdatasetthatcontainsthegroundtruth
answers of a set of 51 questions [5]. To evaluate the output generated from the RAG-application,
thisgoldstandarddatasetiscurated. Thesegroundtruthquestion-answersaregeneratedusingGPT-
4.0 model, contextualized from the EIC-related publications from the arXiv pre-print repository
across domains e.g., high energy phenomenology(hep.ph), nuclear experiments(nucl.ex) etc. Each
question in this dataset is mapped to a pre-defined number of sub-parts called "claims", individual
answer against each claim and a comprehensive answer of the entire question. The AI-generated
QA-pairs are meticulously vetted by human experts to create a gold-truth for validating the RAG-
generated responses. There is also provision to generate question-answers pairs by an annotator
– 4 –

directlyfromuser-chosenarXivarticlesfollowingthesimilarQ&Astructureof"claims",supporting
responses etc. from the web-based annotation interface.
4.2 Evaluation and result analysis
The performance of the application is evaluated with the RAGAS framework [20] encoded into
a set of 6 evaluation metrics; Context Entity Recall, Context Precision, Context Recall, Answer
Relevancy,AnswerCorrectnessandFaithfulness. Thefirstthreemetricsaretovalidatetheground-
edness and factuality of the generated answers with respect to the retrieved context, where as the
later three metrics encode the factual accuracy and semantic similarity of the generated response.
We have explored the evaluation scores across four different configurations: chunk sizes 120 and
180andsimilaritymetricscosine&MMRforcontextretrieval. Thedistributionsscoresforallthese
combinationsareshowninFigure4andFigure5. Theperformanceoftheend-to-endRAG-pipeline
is evaluated by measuring latency of the retrieval and answer generation steps independently. The
retrieval systems’ performance is encapsulated via the retrieval latency: the time elapsed between
thequerysubmissionandthereturnofthemostrelevant20documentsretrievedfromtheknowledge
base.
/uni00000014/uni00000015/uni00000013 /uni00000014/uni0000001b/uni00000013
/uni00000026/uni0000004b/uni00000058/uni00000051/uni0000004e/uni00000003/uni00000036/uni0000004c/uni0000005d/uni00000048/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000018/uni00000013/uni00000011/uni00000014/uni00000013/uni00000013/uni00000013/uni00000011/uni00000014/uni00000015/uni00000018/uni00000013/uni00000011/uni00000014/uni00000018/uni00000013/uni00000013/uni00000011/uni00000014/uni0000001a/uni00000018/uni00000013/uni00000011/uni00000015/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000015/uni00000018/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000056/uni00000048/uni00000046/uni00000052/uni00000051/uni00000047/uni00000056/uni0000000c
/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni00000045/uni0000005c/uni00000003/uni00000026/uni0000004b/uni00000058/uni00000051/uni0000004e/uni00000003/uni00000036/uni0000004c/uni0000005d/uni00000048
/uni00000026/uni00000052/uni00000056/uni0000004c/uni00000051/uni00000048 /uni00000030/uni00000030/uni00000035
/uni00000036/uni0000004c/uni00000050/uni0000004c/uni0000004f/uni00000044/uni00000055/uni0000004c/uni00000057/uni0000005c/uni00000003/uni00000030/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000046/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000018/uni00000013/uni00000011/uni00000014/uni00000013/uni00000013/uni00000013/uni00000011/uni00000014/uni00000015/uni00000018/uni00000013/uni00000011/uni00000014/uni00000018/uni00000013/uni00000013/uni00000011/uni00000014/uni0000001a/uni00000018/uni00000013/uni00000011/uni00000015/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000015/uni00000018/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000056/uni00000048/uni00000046/uni00000052/uni00000051/uni00000047/uni00000056/uni0000000c
/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni00000045/uni0000005c/uni00000003/uni00000036/uni0000004c/uni00000050/uni0000004c/uni0000004f/uni00000044/uni00000055/uni0000004c/uni00000057/uni0000005c/uni00000003/uni00000030/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000046
Figure 2. Left : Retrieval latency of the RAG-system for chunk size 120 and 180 characters with overlap
of 20 characters between two consecutive chunks. Right: Retrieval latency measured for two two different
similarity retrieval mechanism: Cosine similarity and MMR.
Figure 2 shows the retrieval latency distributions for chunk sizes - 120 and 180, and for
similarity metrics - Cosine and MMR. It is represented as box plots to highlight both the central
tendencyandthevariability. Themedianlatencyforchunksizeof120charactersis0.11s,whereas
for180itis0.11-0.12s. Itisalsoobservedthatthechoicesofsimilaritymetricsandthechunksize
yield similar latency period highlighting no significant advantage of one over the other. However,
the larger chunk introduces slightly wider variability.
On the contrary, the choice of LLMs has a significant impact on the inference latency: the
timetakenfortokengenerationbytheLMsafterpromptsubmission. Themostprominentvariation
is observed in the Figure 3, where LLaMA 3.2 outperforms LLaMA 3.3 model performance by
an order of magnitude. The LLaMA 3.3 model utilizes more compute and exhibits substantially
higher and more varying latency. LLaMA 3.2 depicts stability with median latency of 10-20 sec,
a narrow inter-quartile range and moderate outliers of 50-60 sec. Whereas LLaMA 3.3 yields a
drastic shift in median latency. It shows wider variability accompanied by extreme outliers. This
– 5 –

/uni0000002f/uni0000002f/uni00000044/uni00000030/uni00000024/uni00000003/uni00000016/uni00000011/uni00000015 /uni0000002f/uni0000002f/uni00000044/uni00000030/uni00000024/uni00000003/uni00000016/uni00000011/uni00000016
/uni0000002f/uni00000044/uni00000051/uni0000004a/uni00000058/uni00000044/uni0000004a/uni00000048/uni00000003/uni00000030/uni00000052/uni00000047/uni00000048/uni0000004f/uni00000013/uni00000014/uni00000013/uni00000013/uni00000015/uni00000013/uni00000013/uni00000016/uni00000013/uni00000013/uni00000017/uni00000013/uni00000013/uni00000018/uni00000013/uni00000013/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000056/uni00000048/uni00000046/uni00000052/uni00000051/uni00000047/uni00000056/uni0000000c
/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni00000026/uni00000052/uni00000050/uni00000053/uni00000044/uni00000055/uni0000004c/uni00000056/uni00000052/uni00000051/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002f/uni00000044/uni00000051/uni0000004a/uni00000058/uni00000044/uni0000004a/uni00000048/uni00000003/uni00000030/uni00000052/uni00000047/uni00000048/uni0000004f/uni00000056Figure 3. LatencydistributionduringanswergenerationoftheRAG-applicationfortwodifferentLanguage
Models LLaMA 3.2 and LLaMA 3.3 respectively.
highercomputationoverheadofthelargermodelisnotsuitableforaQ&Achatbotapplicationand
hence we incorporated the LLaMA 3.2 model for further study.
The RAG pipeline is evaluated with the RAGAS metrics to access both retrieval and answer
generation steps.
The performance of the retrieval system is captured through three metrics namely: Context
Entity Recall, Context Precision and Context Recall. Context Entity recall [24] examines the
proportion of the entities from the ground truth that are been recalled in the retrieved context.
Whereas Context Recall [25] captures the percentage of the number of claims in the ground truth
that have been extracted in the retrieved context. The Context Precision [26] reveals the retrieved
chunks that are relevant to the user query, hence the proportion expresses the precision of the
retrieved chunks. Among these three metrics Context Recall shows robust performance across
combinations with scores clustered around 1.0 highlighting that the significant proportions of
claims of the ground truth are extracted in the retrieved context. The performance is improved for
chunk size 180 where frequencies are increased near to score 1.0. This improvement is aligned
with the hypothesis that the larger chunk preserve the coherence and semantics of the text. The
Context Precision exhibits moderate bimodal distribution with scores spread across both low end
within[0.1,0.3]and high values above 0.8. Context Precision distribution is broadly invariant
among similarity choices and the chunk lengths. The Context Entity Recall metric distribution is
wide, spread across low to moderate scores for all the four combinations. This under-performance
highlightsthelimitationoftheretrievalmechanismtoextractscientificnamedentitiesasthedense
embedding models are optimized for general semantic not for any specific scientific terminology.
As mentioned earlier the quality of the generated answer is encoded through three RAGAS
evaluationmetrics. AnswerRelevancemeasuressemanticalignmentbetweenthegeneratedanswer
and the query by calculating the mean of the cosine similarity between the original question and a
set of reverse-engineered questions derived from the generated answer [21]. Similarly, the Answer
Correctness [22] is the weighted average of the semantic similarity and factual consistency of the
answer. ThefactualconsistencyisencodedthroughtheF1-scorewhilethecosineanglebetweenthe
ground truth answer and the generated answer captures the semantic relevance. The Faithfulness
metric measures the number of "claims" in the generated answer which are also supported by the
claims in the retrieved context [23], which aids in identifying and quantifying the hallucination.
As shown in Fig 4, Faithfulness score exhibits wider variability for 120 chunk, whereas for 180
– 6 –

/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013/uni00000014/uni00000015/uni00000018/uni00000014/uni00000018/uni00000013/uni00000029/uni00000044/uni0000004c/uni00000057/uni0000004b/uni00000049/uni00000058/uni0000004f/uni00000051/uni00000048/uni00000056/uni00000056
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni00000051/uni00000048/uni00000056/uni00000056
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000035/uni00000048/uni0000004f/uni00000048/uni00000059/uni00000044/uni00000051/uni00000046/uni0000005c
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000028/uni00000051/uni00000057/uni0000004c/uni00000057/uni0000005c/uni00000003/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000015/uni00000013/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000033/uni00000055/uni00000048/uni00000046/uni0000004c/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000026/uni0000004b/uni00000058/uni00000051/uni0000004e/uni00000003/uni00000036/uni0000004c/uni0000005d/uni00000048/uni00000003/uni00000014/uni00000015/uni00000013/uni00000003 /uni00000003/uni00000026/uni00000052/uni00000056/uni0000004c/uni00000051/uni00000048/uni00000003/uni00000036/uni0000004c/uni00000050/uni0000004c/uni0000004f/uni00000044/uni00000055/uni0000004c/uni00000057/uni0000005c
/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000029/uni00000055/uni00000048/uni00000054/uni00000058/uni00000048/uni00000051/uni00000046/uni0000005c
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013/uni00000014/uni00000015/uni00000018/uni00000014/uni00000018/uni00000013/uni00000029/uni00000044/uni0000004c/uni00000057/uni0000004b/uni00000049/uni00000058/uni0000004f/uni00000051/uni00000048/uni00000056/uni00000056
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni00000051/uni00000048/uni00000056/uni00000056
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000035/uni00000048/uni0000004f/uni00000048/uni00000059/uni00000044/uni00000051/uni00000046/uni0000005c
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000028/uni00000051/uni00000057/uni0000004c/uni00000057/uni0000005c/uni00000003/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000015/uni00000013/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000033/uni00000055/uni00000048/uni00000046/uni0000004c/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000026/uni0000004b/uni00000058/uni00000051/uni0000004e/uni00000003/uni00000036/uni0000004c/uni0000005d/uni00000048/uni00000003/uni00000014/uni00000015/uni00000013/uni00000003 /uni00000003/uni00000030/uni00000030/uni00000035/uni00000003/uni00000036/uni0000004c/uni00000050/uni0000004c/uni0000004f/uni00000044/uni00000055/uni0000004c/uni00000057/uni0000005c
/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000029/uni00000055/uni00000048/uni00000054/uni00000058/uni00000048/uni00000051/uni00000046/uni0000005cFigure 4. The performance of the Q&A is represented by the six RAGAS evaluation metrics for retrieval
and answer generation. Upper panel: The evaluation metrics for chunk size of 120 characters for Cosine
similarity; Lower panel: Same metrics as measured for MMR retrieval technique and 120 chunk size.
chunkthedistributionisstronglyright-skewedandachievesmorethan90%scoresforlargenumber
of instances. This demonstrates that larger chunk size renders richer contextual representation
and more factual responses, as the proportions of claims in retrieved context are usually higher
than the smaller chunk size irrespective of the choice of retrieval strategy. A similar trend is also
reflected in the Answer Relevancy score. For 180 chunk size, the distribution is concentrated
largely above0.9score where as for 120 chuck size it is bimodal and wide ranged. The Answer
Correctness encapsulates both the factual similarity and the groundedness of the generated answer
withrespectivetothegroundtruth. Thescoresarepooracrossallthecombinations,whichislikely
due to the EIC-experiment specific complex factual details and also the lightweight LLaMA 3.2
language model.
– 7 –

/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013/uni00000014/uni00000015/uni00000018/uni00000014/uni00000018/uni00000013/uni00000029/uni00000044/uni0000004c/uni00000057/uni0000004b/uni00000049/uni00000058/uni0000004f/uni00000051/uni00000048/uni00000056/uni00000056
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni00000051/uni00000048/uni00000056/uni00000056
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000035/uni00000048/uni0000004f/uni00000048/uni00000059/uni00000044/uni00000051/uni00000046/uni0000005c
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000028/uni00000051/uni00000057/uni0000004c/uni00000057/uni0000005c/uni00000003/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000015/uni00000013/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000033/uni00000055/uni00000048/uni00000046/uni0000004c/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000026/uni0000004b/uni00000058/uni00000051/uni0000004e/uni00000003/uni00000036/uni0000004c/uni0000005d/uni00000048/uni00000003/uni00000014/uni0000001b/uni00000013/uni00000003 /uni00000003/uni00000026/uni00000052/uni00000056/uni0000004c/uni00000051/uni00000048/uni00000003/uni00000036/uni0000004c/uni00000050/uni0000004c/uni0000004f/uni00000044/uni00000055/uni0000004c/uni00000057/uni0000005c
/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000029/uni00000055/uni00000048/uni00000054/uni00000058/uni00000048/uni00000051/uni00000046/uni0000005c
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013/uni00000014/uni00000015/uni00000018/uni00000014/uni00000018/uni00000013/uni00000029/uni00000044/uni0000004c/uni00000057/uni0000004b/uni00000049/uni00000058/uni0000004f/uni00000051/uni00000048/uni00000056/uni00000056
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni00000051/uni00000048/uni00000056/uni00000056
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055/uni00000003/uni00000035/uni00000048/uni0000004f/uni00000048/uni00000059/uni00000044/uni00000051/uni00000046/uni0000005c
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000028/uni00000051/uni00000057/uni0000004c/uni00000057/uni0000005c/uni00000003/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000015/uni00000013/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000026/uni00000052/uni00000051/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000033/uni00000055/uni00000048/uni00000046/uni0000004c/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000026/uni0000004b/uni00000058/uni00000051/uni0000004e/uni00000003/uni00000036/uni0000004c/uni0000005d/uni00000048/uni00000003/uni00000014/uni0000001b/uni00000013/uni00000003 /uni00000003/uni00000030/uni00000030/uni00000035/uni00000003/uni00000036/uni0000004c/uni00000050/uni0000004c/uni0000004f/uni00000044/uni00000055/uni0000004c/uni00000057/uni0000005c
/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000029/uni00000055/uni00000048/uni00000054/uni00000058/uni00000048/uni00000051/uni00000046/uni0000005cFigure 5. The generation performance as characterized by the six metrics of RAGAS framework for chunk
size of 180 characters with Cosine similarity(upper panel) and MMR retrieval(lower panel) respectively.
5 Conclusion
Thisworkpresentsproof-of-conceptdesign,implementationandquantitativeevaluationofaRAG-
basedQuestion-answeringsystemtailoredtopublicationsrelatedtotheEIC-experiment. Thesystem
isdeployableon-premisesandbuiltonopen-sourcecomponentswhichprovideacost-effectiveand
secure alternative to Cloud-based storage and proprietary model. The initiative is aligned towards
data primacy and operational independence of large international scientific collaboration. The
pipeline is constructed with an external knowledge base consisting of PDF articles published in
arXiv,open-sourcecomponents: themxbai-embed-largemodelfortextembedding,ChromaDB
for persistent storage, LLaMA models for answer generation. The pipeline is orchestrated through
LangChain, while citation tracing is enabled by LangSmith. LLaMA 3.3 results an order-of
magnitudeincreaseininferencelatencyalongwithwidervariabilityandextremeoutliersforsome
queries. This larger model may lead to better reasoning ability, however this could not be verified
– 8 –

in this work owing to the compute constraint. The result also highlights that the model scaling has
greater impact on the latency than the design choices such as chunk size and similarity metrics.
The RAGAS evaluation established that chunk-size 180 offers optimal configuration; however, in
our study, MMR mechanism does not demonstrate any added advantage over cosine similarity.
Future work will focus on incorporating additional resources, such as PowerPoint presentations,
wiki, white paper, reports, etc. into the knowledge base. An imminent major planned upgrade of
the pipeline is to migrate to LangGraph [27] orchestration framework.
Acknowledgments
T.GhoshacknowledgestheresearchsupportreceivedfromRamaiahUniversityofAppliedSciences
though out the preparation of this work.
References
[1] J. Maynez, S. Narayan, B. Bohnet and R. McDonald,On faithfulness and factuality in abstractive
summarization, inProc. 58th Annual Meeting of the Association for Computational Linguistics
(ACL), 2020, pp. 1906–1919, arXiv:2005.00661.
[2] Z. Ji, N. Lee, R. Frieske et al.,Survey of hallucination in natural language generation,ACM Comput.
Surv.55(2023) 1, arXiv:2202.03629.
[3] P. Lewis et al.,Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,
arXiv:2005.11401.
[4] R. Abdul Khalek et al.,Science requirements and detector concepts for the electron-ion collider,
Nucl. Phys. A1026(2022) 122447, arXiv:2103.05419.
[5] K. Suresh, N. Kackar, L. Schleck and C. Fanelli,Towards a RAG-based Summarization Agent for the
Electron-Ion Collider, arXiv:2403.15729.
[6] LangChain AI,LangSmith SDK: Platform for LLM application monitoring and evaluation, GitHub
repository (2026).
[7] Pinecone Systems Inc.,Pinecone vector database, online service (2026).
[8] A. J. Yepes, Y. You, J. Milczek, S. Laverde, and R. Li,Financial Report Chunking for Effective
Retrieval-Augmented Generation, arXiv:2402.05131.
[9] LangChain,RecursiveCharacterTextSplitter documentation, online documentation (2026).
[10] Z. Jiang, X. Ma, and W. Chen,LongRAG: Enhancing Retrieval-Augmented Generation with
Long-Context LLMs, arXiv:2406.15319.
[11] S. Lee, A. Shakir, D. Koenig and J. Lipp,Open Source Strikes Bread — New Fluffy Embedding
Model, Mixedbread Blog (March 8, 2024).
[12] N. Muennighoff, N. Tazi, L. Magne and N. Reimers,MTEB: Massive Text Embedding Benchmark,
arXiv:2210.07316.
[13] M. Douze et al.,The Faiss library, arXiv:2401.08281.
[14] LanceDB,LanceDB: Developer-friendly OSS embedded retrieval library for multimodal AI, GitHub
repository (2026).
– 9 –

[15] Chroma Developers,Chroma vector database — fast, scalable search for AI applications,
https://www.trychroma.com/.
[16] J. Goldstein and J. G. Carbonell,Summarization: (1) using MMR for diversity-based reranking and
(2) evaluating summaries,TIPSTER Text Program Phase III Workshop(1998) 181.
doi:10.3115/1119089.1119120.
[17] H. Steck, C. Ekanadham and N. Kallus,Is cosine-similarity of embeddings really about similarity?,
Companion Proc. ACM Web Conf.2024(2024) 887, doi:10.1145/3589335.3651526.
[18] K. Zhou, K. Ethayarajh, D. Card and D. Jurafsky,Problems with cosine as a measure of embedding
similarity for high frequency words, arXiv:2205.05092.
[19] J. D. Hwang, J. Kwon, H. Kamigaito and M. Okumura,Considering length diversity in
retrieval-augmented summarization, arXiv:2503.09249.
[20] S. Es, J. James, L. Espinosa-Anke and S. Schockaert,Ragas: Automated Evaluation of Retrieval
Augmented Generation, arXiv:2309.15217.
[21] Ragas,Answer Relevance, Ragas Documentation (v0.1.21),
https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer_relevance.html.
[22] Ragas,Answer Correctness, Ragas Documentation (v0.1.21),
https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer_correctness.html.
[23] Ragas,Faithfulness, Ragas Documentation (v0.1.21),
https://docs.ragas.io/en/v0.1.21/concepts/metrics/faithfulness.html.
[24] Ragas,Context Entities Recall, Ragas Documentation (v0.1.21),https:
//docs.ragas.io/en/v0.1.21/concepts/metrics/context_entities_recall.html.
[25] Ragas,Context Recall, Ragas Documentation (v0.1.21),
https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_recall.html.
[26] Ragas,Context Precision, Ragas Documentation (v0.1.21),
https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_precision.html.
[27] LangGraph,LangGraph: Agent Orchestration Framework for Reliable AI Agents, [online]. Available:
https://www.langchain.com/langgraph, 2026.
– 10 –