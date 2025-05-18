# Leveraging Graph Retrieval-Augmented Generation to Support Learners' Understanding of Knowledge Concepts in MOOCs

**Authors**: Mohamed Abdelmagied, Mohamed Amine Chatti, Shoeb Joarder, Qurat Ul Ain, Rawaa Alatrash

**Published**: 2025-05-15 08:24:47

**PDF URL**: [http://arxiv.org/pdf/2505.10074v1](http://arxiv.org/pdf/2505.10074v1)

## Abstract
Massive Open Online Courses (MOOCs) lack direct interaction between learners
and instructors, making it challenging for learners to understand new knowledge
concepts. Recently, learners have increasingly used Large Language Models
(LLMs) to support them in acquiring new knowledge. However, LLMs are prone to
hallucinations which limits their reliability. Retrieval-Augmented Generation
(RAG) addresses this issue by retrieving relevant documents before generating a
response. However, the application of RAG across different MOOCs is limited by
unstructured learning material. Furthermore, current RAG systems do not
actively guide learners toward their learning needs. To address these
challenges, we propose a Graph RAG pipeline that leverages Educational
Knowledge Graphs (EduKGs) and Personal Knowledge Graphs (PKGs) to guide
learners to understand knowledge concepts in the MOOC platform CourseMapper.
Specifically, we implement (1) a PKG-based Question Generation method to
recommend personalized questions for learners in context, and (2) an
EduKG-based Question Answering method that leverages the relationships between
knowledge concepts in the EduKG to answer learner selected questions. To
evaluate both methods, we conducted a study with 3 expert instructors on 3
different MOOCs in the MOOC platform CourseMapper. The results of the
evaluation show the potential of Graph RAG to empower learners to understand
new knowledge concepts in a personalized learning experience.

## Full Text


<!-- PDF content starts -->

Leveraging Graph Retrieval-Augmented
Generation to Support Learners’ Understanding
of Knowledge Concepts in MOOCs
Mohamed Abdelmagied, Mohamed Amine Chatti, Shoeb Joarder, Qurat Ul
Ain, and Rawaa Alatrash
Social Computing Group, Faculty of Computer Science, University of
Duisburg-Essen, Germany
https://www.uni-due.de/soco/
Abstract. Massive Open Online Courses (MOOCs) lack direct interac-
tion between learners and instructors, making it challenging for learn-
ers to understand new knowledge concepts. Recently, learners have in-
creasingly used Large Language Models (LLMs) to support them in
acquiring new knowledge. However, LLMs are prone to hallucinations
which limits their reliability. Retrieval-Augmented Generation (RAG)
addresses this issue by retrieving relevant documents before generating
a response. However, the application of RAG across different MOOCs
is limited by unstructured learning material. Furthermore, current RAG
systems do not actively guide learners toward their learning needs. To
address these challenges, we propose a Graph RAG pipeline that lever-
ages Educational Knowledge Graphs (EduKGs) and Personal Knowledge
Graphs (PKGs) to guide learners to understand knowledge concepts in
the MOOC platform CourseMapper. Specifically, we implement (1) a
PKG-based Question Generation method to recommend personalized
questions for learners in context, and (2) an EduKG-based Question
Answering method that leverages the relationships between knowledge
concepts in the EduKG to answer learner selected questions. To evalu-
ate both methods, we conducted a study with 3 expert instructors on
3 different MOOCs in the MOOC platform CourseMapper. The results
of the evaluation show the potential of Graph RAG to empower learn-
ers to understand new knowledge concepts in a personalized learning
experience.
Keywords: EducationalKnowledgeGraphs ·PersonalKnowledgeGraphs
·Graph Retrieval-Augmented Generation ·Large Language Models
1 Introduction
Massive Open Online Course (MOOC) platforms have emerged as a key digital
technology driving the change in the educational landscape in the last decade,
as they promote self-regulated and lifelong learning [7]. However, MOOC plat-
forms present learners with new challenges due to lack of interaction with theirarXiv:2505.10074v1  [cs.AI]  15 May 2025

2 M. Abdelmagied et al.
instructors. Therefore, learners might find it difficult to acquire new knowledge
thathelpsthemachievetheirgoals[8].Recently,LargeLanguageModels(LLMs)
have shown remarkable performances in numerous Natural Language Processing
(NLP) tasks such as Question Answering [11]. Their ability to answer general
knowledge questions led to extensive use by learners to answer their questions in
education.However,thereisconcernabouttheunsystematicuseofthesemodels,
as it has several practical issues, such as hallucinations [13]. Hallucinations lead
LLMs to generate context-unaware or even false content, which can have nega-
tiveeffectsonlearners.Therefore,severaleducationalsystemshaveemergedthat
try to mitigate these issues using methodologies such as Retrieval-Augmented
Generation (RAG). RAG aims to reduce hallucinations by retrieving relevant
documents from a knowledge base and providing them to the LLM before gen-
erating a response [5,10,9].
However, the complex structure and relationships between different concepts
in knowledge bases can represent a challenge for RAG systems. Graph RAG
[12] can address this challenge by its ability to capture complex relationships
between different concepts and establishing links between different documents
in Knowledge Graphs (KGs) to retrieve more relevant and personalized informa-
tion. Graph RAG can be divided into three main steps, which are graph-based in-
dexing,graph-guided retrieval andgraph-enhanced generation [12]. Graph-based
indexing stores KG data in a manner that allows efficient traversal of graph
information, which enhances retrieval quality. Graph-guided retrieval leverages
structuralinformationinaKGtoretrievemoreaccurateresults. Graph-enhanced
generation uses the information provided from the KG to generate a response
with better reasoning.
AnotherlimitationofRAGthatcausesitsapplicationtobelimitedtospecific
courses is that it requires the creation of knowledge bases from course materi-
als that are specifically curated for retrieval. However, this is not always the
case in MOOCs, where instructors often upload unstructured materials such
as lecture slides that may lack the necessary structure for effective knowledge
retrieval. Moreover, this constraint limits learners from accessing additional re-
sources that could provide deeper insights beyond the content presented in the
MOOC. Educational Knowledge Graphs (EduKGs) can be an effective tool to
extract structured knowledge concepts from learning material and linking to ex-
ternal learning resources that can be used as a supporting data source in RAG
[1].
RAG systems in education have an additional limitation in that they require
learners to pro-actively formulate questions according to their learning needs
[9,10,5]. However, they do not actively guide learners in identifying the specific
knowledge or questions required to achieve their learning goals and do not direct
them in learning the essential knowledge concepts necessary for their educational
progress. This can cause learners to diverge from their learning goals or ask
off-topic questions that would cause the system to hallucinate. Therefore, it
is essential for such systems to guide learners by indicating what knowledge
concepts they need to learn and providing personalized recommendations of

Leveraging Graph RAG to Support Learners 3
questions that would help them understand the knowledge concepts. Asking the
right questions requires an effective modeling of the learners’ needs. Recently,
Personal Knowledge Graphs (PKGs) have been proposed to model learners in a
MOOC context, based on the knowledge concepts that they did not understand
[4].ThesePKGscanhelplearnersidentifyknowledgeconceptstheyneedtolearn
toachievetheirgoals.Moreover,PKGscanbeleveragedtogeneratepersonalized
questions that can help learners understand new knowledge concepts.
In this paper, we leverage EduKGs and PKGs to implement a Graph RAG
pipeline that can guide learners in understanding new knowledge concepts in
the MOOC platform CourseMapper [3]. Specifically, we propose (1) a PKG-
based Question Generation method to recommend personalized questions for
each learner according to the knowledge concepts that they do not understand,
and (2) an EduKG-based Question Answering method to answer user-selected
questions by leveraging the relationships between knowledge concepts in the
EduKG. Furthermore, we evaluated the two proposed methods in our Graph
RAG pipeline with three expert instructors on three different MOOCs to assess
the linguistic and task-oriented qualities of the generated questions and answers.
Our study demonstrates the potential of our proposed Graph RAG pipeline to
empower learners to control the Question Generation and Answering processes
in MOOCs, according to their learning needs and goals. In particular, leveraging
PKGs to guide learners to ask questions that help them understand the knowl-
edge concepts was perceived as effective to help learners ask the right questions
in context.
2 EduKG and PKG Construction in CourseMapper
In general, a KG is a graph in which the nodes are knowledge entities and
the edges are relationships between them. An EduKG can represent any type
of entities in educational systems. These entities can be knowledge concepts,
instructors, courses, or even universities [2]. In CourseMapper, instructors can
upload Learning Materials (LM) to the MOOC platform. For every LM, an
EduKG is automatically constructed, where the EduKG includes further entities
such as Slides (S), Main Concepts (MC), Related Concepts (RC) and Learners
(L), as shown in Figure 1. Each LM contains a set of S. Then, each S consists of
a set of MCs extracted using a keyphrase extraction algorithm. These MCs are
taggedwithWikipediaarticlesthatdiscusstheconcept.TheEduKGisexpanded
by extracting other Wikipedia articles referenced in the MC and adding them
as RCs. Furthermore, learners can update their knowledge states by identifying
MCs as "Did Not Understand" (DNU). By allowing learners to update their
states, learners can personalize the EduKG to produce their PKG. Both the
EduKGandPKGinformationarestoredinaNeo4jgraphdatabase.TheEduKG
and PKG give learners a structured overview of what concepts should be learned
to achieve their learning goals.

4 M. Abdelmagied et al.
Fig.1: An overview of an example EduKG in CourseMapper: Each Learning
Material (LM) contains Slides (S), Each Slide consists of Main Concepts (MCs)
which also correspond to Wikipedia Articles. Each MC is related to Related
Concepts (RCs) which are further concepts extracted from the MC article on
Wikipedia.
3 Methodology
In this section, we present our approach to achieve personalized Question Gen-
eration and Question Answering using Graph RAG to support learners’ under-
standing of new knowledge concepts in CourseMapper. Firstly, we present the
abstract pipeline by referring to a user scenario. Then, we dive into the technical
aspects of the pipeline by discussing our workflows for implementing PKG-based
Question Generation and EduKG-based Question Answering.
3.1 User Scenario
Farah is a university student who is enrolled as a learner in the MOOC ’Learn-
ing Analytics’ delivered through CourseMapper. The instructor of the MOOC
uploaded a new LM titled ’Introduction to Machine Learning’. While navigating
the slides in the LM, she recognizes that she is unable to understand ’Slide 4’,
which mentions the definition of Machine Learning (Figure 2, a). Therefore, she
views her PKG for that slide. She notices several MCs for ’Slide 4’ (Figure 2, b).
She realizes that she still does not fully understand the concept of ’Artificial
Intelligence’, which is a broader field than Machine Learning. She selects the
concept and clicks on ’MARK AS NOT UNDERSTOOD’. Afterward, she sees
a dialog open with some suggested questions that can help her understand the
concept of ’Artificial Intelligence’ as shown in Figure 2 c. Therefore, she selects

Leveraging Graph RAG to Support Learners 5
the third question ’What are some applications of artificial intelligence’. Finally,
she receives answers 2 d, which are supported by evidence from a Wikipedia ar-
ticle in the EduKG. Now, Farah understands more about ’Artificial Intelligence’
and is motivated to explore more about this concept by selecting other generated
questions.
(a) Select ’Did Not Understand’
 (b) Mark ’Artificial Intelligence’ as DNU
(c) Question Generation for DNU Concept
 (d) Question Answering and Citation
Fig.2:AuserscenarioofthePKG-basedQuestionGenerationandEduKG-based
Question Answering in CourseMapper
3.2 PKG-based Question Generation
To generate questions that address individual learner’s needs in context, we
leverage their PKG to provide the LLM with a model of the learner. To ensure
a PKG-based Question Generation that generates questions that are relevant
to the current learner’s context, we carefully designed a prompt such that the
LLM generates questions about the DNU concepts that are only based on the
text provided from the current slide or the MCs contained in the slide. When
a learner marks a concept as DNU, the system triggers a graph-guided retrieval
function that retrieves information from the learner’s PKG stored in the Neo4j
database, including the DNU concept, the slide text (slide_text) that contains
the DNU concept, and other MCs that the slide contains (slide_concepts), as
shown in Figure 3 ( A). This knowledge is provided in a zero-shot Question
Generation prompt template for a GPT 3.5-turbo LLM model (Figure 3, P1).
To overcome challenges associated with Question Generation using LLMs, the
prompt template is designed to follow some rules (qg_rules), such as not to
repeat questions that are semantically similar. To ensure that learners inter-
act with the questions based on their importance to the slide, we developed
graph-based re-ranking of the questions by computing their embeddings using a
sentence-transformer model and ranking them based on the similarity of their
embeddings to the embedding of the slide text (Figure 3, B).

6 M. Abdelmagied et al.
Educational Knowledge Graph
The learner did not understand a slide
Generate Main Concepts (MC) from the slide
Mark an MC as „Did Not Understand“ (DNU)Learner Model ConstructionLMLearning Material                        SSlide 1MCMachine LearningRCDeep LearningMCArtificial IntelligenceHAS_READCONTAINS
CONSISTS_OFRELATED_TORELATED_TO
Wikipedia abstract/articleWikipedia abstract/articleWikipedia abstract/articleMC: Main ConceptRC: Related ConceptDNU: Did Not UnderstandLLearner      DNU
Fetch DNU concept, the slide that has ‘consist_of’ relationship with the DNU concept and other MCs of the slide.Question 1Question mRe-rank QuestionsAQuestion re-ranking based on the semantic similarityto the slide of the DNU concept
Show learner the re-ranked questionsPKG-basedQuestion Generation + Graph-basedRe-ranking ofQuestions
The learner selects a question
Graph-based indexing of Wikipedia Paragraphs (WPs) for MCs on each slide and loading Wikipedia Articles (WAs) of RCs
Wikipedia Paragraph--------------------------------------------------------
WA--------------------
Given the following question :{question} and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return {no_output_str}. Remember, *DO NOT* edit the extracted parts of the context.> Question: {{question}}> Context:>>>{{context}}>>>Extracted relevant parts:"""Top-k retrieved Wikipedia paragraphs (WPs)  of MCsGraph Vector StoreGraph Retrieval AugmentedGeneration (Graph RAG)EduKG-basedQuestion Answering+ Citation+ HighlightingGraph RAG for Question Generation and Answering
Question Answeringand Citation
Answer highlighting in the sourceThe learner clicks a citation
WP 1--------------------
WP 2--------------------
WP k--------------------
Wikipedia Paragraph--------------------------------------------------------
Wikipedia Paragraph--------------------------------------------------------
WP 1--------------------
Wikipedia Paragraph--------------------------------------------------------
Wikipedia Paragraph--------------------------------------------------------
WP 2--------------------SSlide nCONTAINSCONSISTS_OF
LMLearning Material                        SSlide 1MCMachine LearningRCDeep LearningMCArtificial IntelligenceHAS_READCONTAINS
CONSISTS_OFRELATED_TORELATED_TO
Wikipedia abstract/articleWikipedia abstract/articleWikipedia abstract/articleLLearnerDNUSSlide nCONTAINSCONSISTS_OFEmbedding
Embeddings
Large Language Model (LLM)Question Generation based on a DNU Concept
Graph StoreQuestions about DNUConcept
Large Language Model (LLM)
GivenONLY the following slide text:{slide_text}The learner does not understand the following concept:{DNU}Generate a set of questionsbased on information in the slide that can help the learner understand the concept.If the slide text is also insuffiecientto generate relevant questions, should also use the following concepts: {slide_concepts}You should also follow these rules:{qg_rules}### QUESTIONS:Question Generation prompt template
Extractive QA prompt template
B
CDFQuestion
GGiven the following question:{question}Choose from the following concepts, the most relevant concept to answer the question:{RCs}EduKGretrieval prompt templateRCs
Large Language Model (LLM)
Wikipedia Paragraph--------------------------------------------------------
WA--------------------EWikipedia Article (WA)of RCP1A
CDB
F
EP2P3G
Fig.3: An overview of the pipeline for implementing PKG-based Question Gen-
eration and EduKG-based Question Answering with the steps: ( A)graph-guided
retrieval for Question Generation, ( P1) Question Generation prompt template,
(B)graph-based re-ranking , (C)Graph-based indexing , (D) and ( E)graph-guided
retrieval for Question Answering, ( P2) Extractive Question Answering prompt
template, ( P3) EduKG retrieval prompt template, ( F) Answers with citations,
(G) Highlighted answers
3.3 EduKG-based Question Answering
To answer a learner’s selected question, we perform graph-guided retrieval using
the EduKG. To this end, the Wikipedia articles of the MCs in the current slide
are chunked into a set of paragraphs and each Wikipedia Paragraph (WP) is
indexed as a node in the graph vector store set up in Neo4j. The graph-based
indexing process takes place by calculating the vector embedding of each para-
graph using a sentence-transformers language model. When the learner selects a
question, the question is also transformed to a vector embedding and the most
similar WPs to the questions are retrieved based on the cosine similarity, as
shown in Figure 3 ( C). After retrieving the most relevant Wikipedia contexts to
the question (Figure 3, D), they are injected into the Extracive QA prompt tem-
plate shown in Figure 3 ( P2). This prompt should cause the LLM’s answer to be
strictly based on extracted Wikipedia contexts, thus reducing hallucinations. In
case that the LLM does not generate answers (i.e., the MC WPs do not contain
an answer to the selected question), we leverage the EduKG to get the RCs to
the MCs in the slide. For example, the answer to a question such as ’What is
parameter tuning in Machine Learning?’ might not be in the WPs of the MC
’Machine Learning’, however, it can be found in an RC of ’Machine Learning’,
such as ’Hyperparameter Optimization’ because it is a more specialized topic

Leveraging Graph RAG to Support Learners 7
that addresses this question. To achieve this, we load the full Wikipedia Articles
(WA) of the RCs but without embedding. We avoid indexing the WAs of RCs
in the graph vector store as each MC might have hundreds of RCs which would
cause the graph-based indexing to be slower. To retrieve answers from WAs of
RCs, we employ an LLM retriever that is prompted using the EduKG retrieval
prompt template (Figure 3, P3) to perform graph-guided retrieval by traversing
the EduKG and reasoning which RCs might contain an answer to the question
(Figure 3, E). The WA of the retrieved RC is then provided back to the Extrac-
tive QA prompt template P2to extract answers from them. After extracting
the answers from the given contexts, the answers are provided to the learners
along with citations of the resources (Figure 3, F). By clicking on the answer, the
learner is redirected to the source with the answer highlighted as illustrated in
Figure 3 ( G). This can ensure that learners explore beyond the answers provided
to them by the LLM.
4 Evaluation
In this section, we present the results of the human evaluation for the PKG-
based Question Generation and EduKG-based Question Answering pipelines, in
terms of linguistic and task-oriented dimensions.
4.1 PKG-based Question Generation Evaluation
We asked 3 instructors who deliver MOOCs through CourseMapper to evaluate
the PKG-based Question Generation pipeline according to linguistic and task-
oriented criteria. The evaluators are a Professor and two Teaching Assistants at
the local university. The evaluation was carried out over three different MOOCs
according to the instructor’s area of expertise. The topics of the MOOCs were
Learning Analytics (LA), Human-Computer Interaction (HCI) and Web Tech-
nologies (WT). Each instructor was asked to interact with the pipeline, in which
they would select an LM from their MOOC. Then, they were asked to find slides
that learners would find challenging. Then, they were asked to select any MC
as DNU. For every DNU concept, they were asked to evaluate all recommended
questions. The process was repeated for several DNU concepts in every MOOC.
For LA and WT, 6 DNU concepts and 30 Question-Answer pairs were evalu-
ated for each. For HCI, 8 DNU concepts and 40 Question-Answer pairs were
evaluated.
We follow an evaluation framework similar to Fu et al. [6]. The framework
provides seven dimensions for evaluating Question Generation models. The di-
mensions are divided into linguistic and task-oriented dimensions. Linguistic
dimensions are Fluency (Flu.), Clarity (Clar.), and Conciseness (Conc.). The
task-oriented dimensions are Relevance (Rel.), Consistency (Cons.), Answerabil-
ity (Ans.), and Answer Consistency (AnsC.). Relevance is the most prevalent
metric in evaluating Question Generation [6]. In the context of our pipeline,

8 M. Abdelmagied et al.
Relevance plays an important role since the main purpose of PKG-based Ques-
tion Generation is to personalize the learning experience for every learner. This
can be validated by measuring the relevance of the question to the chosen slide
(Rel slide) and DNU concept ( Rel dnuconcept ). The other task-oriented dimensions
are all particularly defined to address challenges in Question Generation for
reading comprehension tasks, which is not the focus of this work; therefore, we
refrain from evaluating questions according to them. The five dimensions that
we used, namely Flu., Clar., Conc., Rel slide, and Rel dnuconcept are evaluated on
a scale 1 to 3, the higher being better. Table 1 shows the results of the eval-
uation of the PKG-based Question Generation for the three different MOOCs.
The evaluation shows very promising results with regards to both linguistic as-
pects and relevance. Our results show that the pipeline does not face linguistic
challenges. In terms of relevance, the pipeline scores in general high, which is
very promising as it shows that it can effectively support learners to ask the right
questions in context, according to their needs. Furthermore, the instructors were
very impressed with the level of detail in the questions and the level of relevance
provided with regard to every slide and every DNU concept, and they showed
an interest in deploying the feature into their MOOCs in the future.
Table 1: Comparison of PKG-based Question Generation on linguistic and rele-
vance evaluation dimensions across different MOOCs
MOOCs Flu.Clar. Conc. Rel slideRel dnuconcept Avg.
Learning Analytics 2.9672.8672.9673.000 3.000 2.961
Human-Computer Interaction 3.0002.8753.0002.725 2.55 2.83
Web Technologies 2.9692.7502.9432.875 2.496 2.806
Weighted Avg. 2.9812.8352.9732.853 2.668 2.862
4.2 EduKG-based Question Answering Evaluation
We asked the instructors to review the accuracy of the responses they receive on
each question they select. While, there are other metrics to evaluate Question
Answering, we choose accuracy in order to align with the evaluations of other
RAG systems used in education [9,10], which define accuracy as either correct
or incorrect. Table 2 presents the results of the perceived accuracy of EduKG-
basedQuestionAnsweringforthethreedifferentMOOCs.Theseresultsshowthe
challenging aspect of EduKG-based Question Answering. According to the in-
structors, most answers were considered too abstract or not directly to the point
which leads them to being identified as incorrect. By reviewing the data, we find
several possible reasons for this. The EduKG is constructed of Wikipedia arti-
cles and the LLM is prompted to extract the most relevant information from the

Leveraging Graph RAG to Support Learners 9
retrieved Wikipedia articles as it is with no further reasoning. While this might
lead to higher levels of trust, as the retrieved text is exactly as in Wikipedia,
it leads the LLM to generate responses that are abstract. Another reason could
be the lack of disambiguation capabilities of the retrieval process. For exam-
ple, in the MOOC HCI, one of the evaluated DNU concepts was ’Emergency
Exit’ in reference to a concept in User Experience (UX). However, the retriever
was incapable to distinguish it from the ’Emergency Exit’ for buildings. In gen-
eral, the instructors found the pipeline to be highly interactive. According to
their comments, they believe that with further modifications to the EduKG-
based Question Answering, the pipeline can be an essential tool for learners and
instructors to have a more structured and personalized learning experience in
MOOCs.
Table 2: Accuracy of responses of the EduKG-based Question Answering in
MOOCs
MOOCs Accuracy (%) Number of Correct Answers / Total Number of Answers
Learning Analytics 56.67 17/30
Human-Computer Interaction 45.00 18/40
Web Technologies 33.33 10/30
Weighted Average 45.00 45/100
5 Conclusion and Future Work
In this paper, we presented an innovative approach that leverages Educational
Knowledge Graphs (EduKGs) and Personal Knowledge Graphs (PKGs) to im-
plement a Graph RAG pipeline that can guide learners in understanding new
knowledge concepts in the MOOC platform CourseMapper. The evaluation re-
sults show the potential of in-context Question Generation and Answering using
Graph RAG in MOOCs. In particular, PKG-based Question Generation was
perceived as effective to guide learners to learn new concepts in context. How-
ever, EduKG-based Question Answering still requires further enhancements to
improve its accuracy and reliability. In this aspect, the first modification could
be constructing the EduKG using further external learning resources that might
hold more details to knowledge concepts than Wikipedia articles such as the
learning material of other MOOCs, scientific literature or even recommended
videos about the concepts. The second modification could be to review further
methods that would allow the LLMs to use the provided resources as evidence
but still give it room to perform reasoning and provide further explanations. For
example, the chain-of-thought method can allow the LLM to leverage an EduKG
as a structured data source to perform advanced reasoning that is still relevant
to the content of the MOOC and supported by evidence from the EduKG.

10 M. Abdelmagied et al.
Disclosure of Interests. The authors have no competing interests to declare
that are relevant to the content of this article.
References
1. Abu-Rasheed, H., Jumbo, C., Amin, R.A., Weber, C., Wiese, V., Obermaisser,
R., Fathi, M.: Llm-assisted knowledge graph completion for curriculum and do-
main modelling in personalized higher education recommendations. arXiv preprint
arXiv:2501.12300 (2025)
2. Ain, Q.U., Chatti, M.A., Bakar, K.G.C., Joarder, S., Alatrash, R.: Automatic
construction of educational knowledge graphs: a word embedding-based approach.
Information 14(10), 526 (2023)
3. Ain, Q.U., Chatti, M.A., Joarder, S., Nassif, I., Wobiwo Teda, B.S., Guesmi,
M., Alatrash, R.: Learning channels to support interaction and collaboration in
coursemapper. In: Proceedings of the 14th International Conference on Education
Technology and Computers. pp. 252–260 (2022)
4. Ain, Q.U., Chatti, M.A., Meteng Kamdem, P.A., Alatrash, R., Joarder, S., Siep-
mann, C.: Learner modeling and recommendation of learning resources using per-
sonalknowledgegraphs.In:Proceedingsofthe14thLearningAnalyticsandKnowl-
edge Conference. pp. 273–283 (2024)
5. Dan, Y., Lei, Z., Gu, Y., Li, Y., Yin, J., Lin, J., Ye, L., Tie, Z., Zhou, Y., Wang, Y.,
et al.: EduChat: A large-scale language model-based chatbot system for intelligent
education. arXiv preprint arXiv:2308.02773 (2023)
6. Fu, W., Wei, B., Hu, J., Cai, Z., Liu, J.: Qgeval: Benchmarking multi-dimensional
evaluation for question generation. arXiv preprint arXiv:2406.05707 (2024)
7. Haleem, A., Javaid, M., Qadri, M.A., Suman, R.: Understanding the role of digital
technologies in education: A review. Sustainable Operations and Computers 3,
275–285 (2022)
8. Henderikx, M., Kreijns, K., Xu, K.M., Kalz, M.: Making barriers to learning in
moocs visible. a factor analytical approach. Open Praxis 13(2), 143–159 (2021)
9. Liu,C.,Hoang,L.,Stolman,A.,Wu,B.:HiTA:ARAG-basededucationalplatform
that centers educators in the instructional loop. In: International Conference on
Artificial Intelligence in Education. pp. 405–412. Springer (2024)
10. Liu, R., Zenke, C., Liu, C., Holmes, A., Thornton, P., Malan, D.J.: Teaching CS50
with AI: Leveraging generative artificial intelligence in computer science education.
In: Proceedings of the 55th ACM Technical Symposium on Computer Science Ed-
ucation V. 1. pp. 750–756 (2024)
11. Naveed, H., Khan, A.U., Qiu, S., Saqib, M., Anwar, S., Usman, M., Akhtar, N.,
Barnes, N., Mian, A.: A comprehensive overview of large language models. arXiv
preprint arXiv:2307.06435 (2023)
12. Peng, B., Zhu, Y., Liu, Y., Bo, X., Shi, H., Hong, C., Zhang, Y., Tang, S.: Graph
retrieval-augmented generation: A survey (2024), https://arxiv.org/abs/2408.
08921
13. Wang, S., Xu, T., Li, H., Zhang, C., Liang, J., Tang, J., Yu, P.S., Wen, Q.:
Large language models for education: A survey and outlook. arXiv preprint
arXiv:2403.18105 (2024)