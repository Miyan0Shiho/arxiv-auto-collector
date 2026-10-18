# Multi-sourced AI Tutor for Non-Native English Speaker Students

**Authors**: Yiyin Shen, Xinyi Ai, Weibing Wang, Rogers Jeffrey Leo John, Meenakshi Syamkumar

**Published**: 2026-10-15 16:26:00

**PDF URL**: [https://arxiv.org/pdf/2610.22163v1](https://arxiv.org/pdf/2610.22163v1)

## Abstract
Large language models (LLMs) are increasingly being adopted as teaching tools in computing education, particularly for developing AI tutors capable of answering student questions. With the rise in enrollment of international students in computing programs, researchers have begun to explore the difficulties that non-native English speaker (NNES) students face in computing courses due to language barriers. To aid such students, we developed a general-purpose AI tutor system that can process student questions in foreign languages (currently focused on Mandarin) and provide bilingual responses. Our system also incorporates the functionality of generating a glossary of technical terms and their explanations to enhance comprehension. Our approach utilizes a specialized Retrieval-Augmented Generation (RAG) pipeline with a novel question extraction mechanism, enabling us to effectively utilize a diverse array of course materials for answering student queries. We evaluated our AI tutor using 91 Piazza posts, demonstrating that it could answer 83.52% of the posts correctly and in a helpful manner. The tutor also provided high-quality translations for 76.92% of the questions and generated technical term glossaries for 64.47% of them. Our findings underscore the potential of multi-source RAG pipeline in enhancing AI tutor systems, offering support for non-native speakers that are challenging to provide with traditional supporting staff of a computing course.

## Full Text


<!-- PDF content starts -->

Multi-sourced AI Tutor for Non-Native English Speaker Students
Yiyin Shen
University of Wisconsin–Madison
Madison, WI, USA
yshen82@wisc.eduXinyi Ai
Independent Researcher
Mountain View, CA, USA
cathyaxy@gmail.comWeibing Wang
Massachusetts Institute of Technology
Cambridge, MA, USA
weibingw@mit.edu
Rogers Jeffrey Leo John
Independent Researcher
Madison, WI, USA
rogersjeffreyl@gmail.comMeenakshi Syamkumar
University of Wisconsin–Madison
Madison, WI, USA
ms@cs.wisc.edu
Abstract
Largelanguagemodels(LLMs)areincreasinglybeingadoptedasteaching
toolsincomputingeducation,particularlyfordevelopingAItutorscapable
ofansweringstudentquestions. Withtheriseinenrollmentofinternational
students in computing programs, researchers have begun to explore the
difficultiesthatnon-nativeEnglishspeaker(NNES)studentsfaceincomputing
coursesduetolanguagebarriers. Toaidsuchstudents,wedevelopedageneral-
purposeAItutorsystemthatcanprocessstudentquestionsinforeignlanguages
(currentlyfocusedonMandarin)andprovidebilingualresponses. Oursystem
alsoincorporatesthefunctionalityofgeneratingaglossaryoftechnicalterms
andtheirexplanationstoenhancecomprehension. Ourapproachutilizesa
specialized Retrieval-Augmented Generation (RAG) pipeline with a novel
question extraction mechanism, enabling us to effectively utilize a diverse
array of course materials for answering student queries. We evaluated our
AI tutor using 91 Piazza posts, demonstrating that it could answer 83.52%
of the posts correctly and in a helpful manner. The tutor also provided
high-quality translations for 76.92% of the questions and generated technical
term glossaries for 64.47% of them. Our findings underscore the potential of
multi-sourceRAGpipelineinenhancingAItutorsystems,offeringsupportfor
non-nativespeakersthatarechallengingtoprovidewithtraditionalsupporting
staff of a computing course.
Keywords:large language model, non-native English speaker students,
computing education
1 Introduction
Large Language Models (LLMs) have profoundly transformed com-
puting education, showcasing remarkable capabilities in solving
computing assignments [ 18,20]. LLMs are also being used for
facilitating instructional activities, such as generating assignment
questions[ 19],providingautomatedgrading[ 22],andnotably,de-
veloping AI tutor systems that address student queries. Furthermore,
as classrooms become increasingly multicultural, the need for inclu-
sivity ineducational materials andtechnologiesgrows. Non-native
English speaker (NNES) students, in particular, face language barri-
ersandalackofconfidence,whichoftencauseshesitationtoseek
help from instructors or TAs. Our AI tutor system seeks to assist
NNES students’ learning by accepting questions in the students’
native language (specifically Mandarin), delivering responses bilin-
gually,and generatinga technicaltermglossary toensure equitable
access to educational materials.
MostAItutors[ 3,10,11]utilizeRetrieval-AugmentedGeneration
(RAG)[9],whichenhancesquestion-answeringbydynamicallyre-
trievingandsynthesizingrelevantinformationfromadatarepository,
ensuringaccuracyandcontextualrelevance. OurRAG-basedGPT-4
tutordiffersfromsystemsthatrelyonasinglesource,suchaslecture
captionsortextbooks. Instead,itintegratesacomprehensiverange
of course materials, including lecture notes, project descriptions,
quizzes,pastexams,andthecoursewebsite,enablingholisticdata
retrieval.Two specialized mechanisms have been developed to enhance
our AI tutor. Firstly, our system incorporates a translation stage
that converts Mandarin questions into English before applying RAG,
enablingfullleverageofRAG’scapabilities. Secondly,oursystem
employsaquestionextractionmechanismthataddressesinstances
whereastudent’squeryreferencesonlythequestionnumberinsteadof
providingthequestiondetails,bypreciselyidentifyingandretrieving
details of the referred question.
We evaluated our system using 91 Piazza posts, demonstrating
that it can accurately and helpfully answer student questions and
generate technical term glossary in Mandarin. Our work focuses on
answering the following research questions:
•RQ1. What is the quality of the translation stage and to what
extent does it improve RAG’s performance in processing non-
English questions?
•RQ2. Whatis theaccuracy ofthequestion extractionstageand
to what extent does it enhance the system’s ability to manage
diverse course materials and student questions?
•RQ3. What is the relevance and usefulness of RAG’s results
concerning the questions?
•RQ4. What is the quality of the AI tutor’s responses in terms
ofcorrectness,helpfulness,solutionrevelation,translation,and
technical term glossary generation?
2 Related Work
2.1 AI Tutors
Since the debut of Large Language Models (LLMs), researchers
have consistently explored their potential applications in computing
education. Beyond aiding student learning by providing thorough
code explanations [ 12], step-wise worked examples [ 5], and prompt
problems[ 1],aswellasfacilitatingvariousinstructionalactivities,
LLMs have proven effective as the basis for educational question-
answering (QA) systems, or in other words, AI tutors.
Hicke et al. [ 3] investigated the effectiveness of various optimiza-
tiontechniquesinLLMsforansweringquestionscollectedfromaCS1
course Piazza. They enhanced Llama-2 using Retrieval-Augmented
Generation(RAG)ontextbooksandlabmanuals,supervisedfinetun-
ing withselected Piazza QA pairs, andreinforcement learning with
human feedback; and applied RAG to GPT-4. They show that these
techniques resulted in substantial improvements: Llama-2’s answer
accuracy (factual correctness & groundedness) increased from 0.30
to0.58,andGPT-4’sfrom0.40to0.73,whileusefulness(practical
valuestohumaninstructors)increasedfrom0.18to0.50forLlama-2
and from 0.28 to 0.65 for GPT-4.
Liu and M’hiri [ 10] developed a virtual teaching assistant (TA)
using GPT-3.5 and RAG on assignment documents. The virtual TA
arXiv:2610.22163v1  [cs.AI]  15 Oct 2026

first performs embedding searches to compare students’ questions
with assignment documents, preventing the revelation of answers
to the current assignments to students who directly ask for them.
Subsequently,theTAgeneratesresponsestailoredtoeachquestion
categoryandconductsqualitycheckstoensureresponseaccuracy
and avoid solution leak.
Furthermore,Liuetal. [ 11]andKazemitabaaretal. [ 6]developed
their own QA chatbots and deployed them in real course settings.
The two chatbots have distinct features: the prior utilized GPT-4
with RAG on lecture captioning and structured prompts to generate
responses, while the latter used the few-shot learning framework to
preventcoderevelation. Bothchatbotsmaintainedahighcorrectness
rateinresolvingstudentinquiresthroughoutthesemester. Moreover,
studentsexpressedsatisfactionwiththesetoolsinthefeedbacksurvey,
stating they appreciated having such a "24/7 personal tutor" and the
opportunity to "work alongside AI."
Most of the AI tutor systems above use RAG based on a single
source format, such as textbooks, assignment documents, or lecture
captions. OurAItutoraimstoincorporatealldocumentsinacourse,
including lecture materials and readings, lab and project description
documents, quizzes and past exams, as well as the course website,
by developing appropriate data pipelines and prompt techniques for
each type of course material.
2.2 Non-Native English Speakers in Computing Educa-
tion
The enrollment of international students in computing programs has
increased by 14% to 298,523 during the 2022-23 academic year.
Notably,23%ofthempursuedmathematicsandcomputerscience,
and19%studiedengineering[ 4]. Thesestudentsoftenfacelanguage
barriersandconfidenceissues,whichnegativelyimpacttheirlearning
experience in computing courses. In response, researchers are
exploring tailored supportive strategies for this demographic.
Polishchuk[ 15]surveyed124first-yearnon-nativeEnglishspeaker
(NNES) engineering and agriculture students, identifying several
key communication barriers, including vocabulary and grammar
difficultiesandchallengeswithaccentsandslang. Notably,85.5%
ofstudentsreportedtranslatingutterancestotheirnativelanguage,
whichhinderedtheirfluency,while61.25%wereafraidofmaking
mistakes, and 47.5% felt nervous speaking in front of the class.
Polishchuk recommended enhancing collaborative work and com-
munication practice outside the classroom.
Guo [2] noted how confidence issues affect NNES’s communica-
tion withinstructorsand TAs. Via a surveyof 840 respondents from
86 countries on a programming education website, this study found
studentsreluctanttoparticipateindiscussionsoraskquestionsdueto
fear of errors and being misunderstood. They preferred instructional
materialsinsimplifiedEnglish,supplementedwithvisualsandinline
dictionaries. Guosuggestedpeermentoringandlanguagesupport
workshops to improve participation and confidence.
Soosai Raj et al. [ 16,17] conducted a four-week experiment
with second-year engineering students in India, comparing those
taught only using English to those receiving bilingual instruction in
English and Tamil. The study found that bilingual teaching reduced
cognitive load, improved comprehension of programming concepts,
andboostedstudent engagement andconfidence. Lauand Yuen [ 8]
noted that K-12 students in Hong Kong understood programming
better with bilingual instruction, highlighting the value of native
language incorporation for NNES students.
To address the challenges faced by NNES students, we aim to
develop an AI tutor that accepts questions in the student’s native
languageand providesbilingual answers tofacilitate understandingof complex computing concepts. This approach will also help build
confidence among students who may hesitate to approach their
instructorsandTAs. SinceourauthorsaremainlynativeMandarin
speakers, we implemented our prototypical system for Mandarin.
3 Method
3.1 Students’ Questions
We collected 620 questions from a semester’s Piazza posts for an
intermediate-levelDataScience(DS)programmingcourseof400
students1. We excluded 238 posts that included images for two
reasons: (1)theyaredifficulttotranscribe,eitherbecausetheyare
entire desktop screenshots with only a small portion containing
relevantinformationortheyinvolvecomplexplotsandgraphs;(2)
theGPT-4APIvisionservicewasprohibitivelyexpensivepriortothe
release of GPT-4o. The analysis focused on the remaining 382 text-
basedposts,whichwecleanedbyremovingduplicates,identifiable
information, unnecessary HTML formatting, and enclosing code
snippets and error messages in backticks.
For Piazza posts, students must tag their posts with instructor-
definedfolders: exams(exam1,exam2,final),quizzes,programming
projects (P1, P2, ..., P7), labs, logistics, and others. One author used
these student-tagged folders and the post content to sort the posts
into four categories: 90 logistics, 54 conceptual, 41 environment
setup,and197programmingposts. Logisticquestionsdealtwithsyl-
labus specifics and demandsfor clarifications on acceptable coding
practices and plotting requirements. Conceptual questions typically
pertained to quiz and exam content. Environment setup questions
covered the setup and maintenance of virtual machines and Jupyter-
Lab, package installations, and git-related issues. Programming
questions primarily revolved around project development challenges,
addressing issues such as implementation of specific functionalities,
debugging, and optimization for code efficiency.
We then generated the Mandarin dataset by translating the Piazza
posts into Mandarin using the Google Translate API2. To ensure
precision,alltranslationsweremanuallyreviewedandeditedbya
nativeMandarin-speakingauthor. Sections previously marked with
backticks were excluded from the translation.
3.2 AI Tutor Data Pipeline
AflowchartofourpipelineisshowninFigure1. Itincludes4stages:
translation, question extraction, RAG, and response generation3.
3.2.1 Translation
Our pipeline processes two types of inputs: the original English
Piazza questions and their human-verified Mandarin translation
mentioned in Section 3.1. When a question is received in Mandarin,
the model translates it into English using the Microsoft Azure AI
Translate API4. We opted for translation over a Mandarin-dedicated
LLMtoleveragethecapabilitiesofGPT-4,whichremainsstate-of-
the-art, especially in programming context.
3.2.2 Question Extraction (QE)
A significant portion (about 35%) of the questions collected from
Piazza relate to specific projects, quizzes, and prior exam questions,
often referencing only the question number (e.g., "P2 Q10" and
"FA22 Exam 2 Q10") in the subject line, without including the
question’s content in the post. Question extraction (QE) stage of
1IRB details will be included upon publication.
2The translation component is modular and can be replaced with other APIs.
3Coursematerialsandsourcecodeforthepipelinewillbereleaseduponpublication.
4WechosetobuildoursystemusingAzureAITranslateAPIforcosteffectiveness.
This component is also modular and can be replaced with other APIs.

Student’s question
Question in Mandarin?
Translation to English
Question Extraction
RAG
Response Generationyes
no
Figure 1:Flowchart of the RAG-based pipeline
the pipeline is equipped to handle such questions. Unlike typical
computerscienceprojectsthatrequirestudentstoimplementwell-
known data structures or algorithms, DS projects involve sequential
(numbered) analysis tasks on given datasets, making it important for
themodeltoreceivetheoriginalquestion’sdetails. Furthermore,this
stepisirreplaceablebyRAGasbothembedding-basedandkeyword-
based retrieval methods face challenges in extracting the exact
questionfromthecoursematerialsonlyusingthequestionnumber.
Embedding-based methods struggle due to the short and numeric
natureofthesereferences, whilekeyword-basedsystemsencounter
difficulties due to variability of question number abbreviations (e.g.,
"Project 2 Question 10" and "Fall 2022 Midterm 2 Question 10").
Toaddresstheseissues,weusedtheLlama38Bmodeltogenerate
standardized question numbers from student posts. This model is
hosted using Ollama on a machine with a Nvidia GeForce RTX
4090. This link, https://bit.ly/3Lv3Eiw , provides the few-
shot prompt we used for the QE task. The prompt is structured as
follows: thetaskdescription,theinputformataccompaniedbyabrief
description of the course materials, the expected output format, and
several examples of input and output. The generated standardized
question number enables us to accurately fetch the question content
from a question bank using a rule-based algorithm. The question
bank is created as text files by parsing past exams and solutions
(PDFs) andCanvasquizzes usingfitz andCanvas API,respectively.
QE is a crucial step that ensures the subsequent RAG process
retrievesthemostrelevantcontentrelatedtothespecificquestions
in the student’s posts, thereby improving the accuracy and relevancy
of the generated responses.
3.2.3 RAG
We structured our RAG usingLangChain[ 7] with a comprehensive
vector store of all course materials. This includes lecture notes
(Python scripts and Jupyter Notebooks), PowerPoint slides, work-
sheets (PDFs), complementary readings (Python scripts and Jupyter
Notebooks), project and lab repositories (Python scripts, Jupyter
Notebooks, and Markdown), and the course website pages (HTML).
Various parsers and splitters were employed to manage these di-
verse data types. For parsing, we converted Jupyter Notebooks to
Python scriptsusing nbconvert, andPowerPoint slidesand PDFs toMarkdownusingpptx2md andfitz. MarkdownandHTML content
are initially split using corresponding Langchain Markdown and
HTML Header Text splitters, and further split byRecursiveChar-
acterTextSplitterin Python for excessively long chunks. Jupyter
Notebooks,whicharealreadyformattedusingMarkdownheaders,
are first split using a modified version of LangChainMarkdown-
HeaderTextSplitterforJupyterNotebooksandthenfurthersplitby
theRecursiveCharacterTextSplitter. We also included the source
nameinthesplitchunkstoallowreferencetocoursematerialsinthe
generated response. The vector store is then created using FAISS
[13] with OpenAI’s text-embedding-ada-002 embedding.
In similarity search phase, we input both the student post and any
extracted questions. We exclude past exams and Canvas quizzes
from our search to avoid skewing results towards similar exam or
quizquestions,whichcouldpreventtheretrievaloflecturematerials.
We retrieve the top 10 most relevant results.
3.2.4 Response Generation
Responses are generated using the Microsoft Azure OpenAI GPT-4
API. This link, https://bit.ly/3SeXvLg , provides the system
and user prompts for the generation task. In the system prompt, we
direct the model to function as an AI tutor for non-native English
speakerstudents,stipulatingthatitshouldnotdirectlyrevealsolutions
but rather provide hints, explanations, or suggestions. We instruct
the model to avoid writing code snippets. We also ask the model to
cite RAG results accordingly using the source path. The response
is required to be in both English and Mandarin. Any technical
vocabularyorerrormessagesthatmightchallengenon-nativeEnglish
speakerstudentsisaskedtobecompiledintoaglossaryattheendof
theresponsetoaidunderstanding. Intheuserprompt,weincludethe
originalstudentpost,anyextractedquestionnumbersandcontent,
and the results from the RAG.
4 Results
4.1 Examples
We will present the results from all four stages of our pipeline for
one conceptual and one programming question.
Thislink, https://bit.ly/3VbHpT8 ,illustratestheoutcomes
foraconceptualpostinMandarin,whichaskswhether"theDocument
ObjectModel(DOM)isatreeoradirectedacyclicgraph(DAG)".
Thetranslationaccuratelycapturesthestudent’squery. Thequestion
extractorsuccessfullyidentifiestherelevantquestionsfromSpring
22Exam2Question2andQuiz5Question2. TheRAGretrievals
includedefinitionsoftreesandDAGsfromlecturenotesandreadings.
Theresponseeffectivelyresolvesthestudent’sconfusionbyoutlining
the definitions shown in the RAG retrievals and clarifying that a tree
is a type of DAG.
This link, https://bit.ly/3Ra5Hf2 , displays the results for a
programming post asking for tips on P5 Q7. The question extractor
accurately identifies the targeted question. The first helpful RAG
retrieval provides information about the dataframe, while the second
introduces the necessary function from the EDGAR utils module.
The response correctly offers general guidance for the question
without revealing the solution.
Forbothquestions,theMandarintranslationandtechnicalterm
glossary were of high quality.
4.2 Quantitative Results
AllfourstagesofourpipelinewereevaluatedbytwonativeMandarin-
speaking authors. In instances of evaluative discrepancies, the third
author, who is also a native Mandarin speaker and the course
instructor, made the decisive evaluation.

logistics conceptual environment setup programming
English 97.78% 90.74% 100% 97.46%
Mandarin overlap 98.89% 72.22% 100% 97.97%
Translated overlap 95.56% 74.07% 100% 97.46%
Table 1:Question extraction accuracy for English posts and of percentages overlaps between QE results for English and Mandarin/translated posts
logistics conceptual environment setup programming
English relevancy 6.1 7.7 7.3 6.6
Mandarin overlap 7.0 5.7 7.5 6.8
Translated overlap 8.5 7.0 8.9 8.2
Table 2:Number of relevant RAG results for English posts and average numbers of overlaps between RAG results for English and Mandarin/translated posts out
of 10 candidates
Since we cannot feasibly evaluate all four stages for English,
Mandarin, and translated (posts after the translation stage) posts, we
focused on evaluatingthe question extraction (QE) and RAG stages
forEnglishposts,whichwewilldemonstratecanserveastheground
truth for evaluating the Mandarin and translated posts. This was
achieved by computing the percentage overlap between the English
and Mandarin/translated results. We will further demonstrate that
the Mandarin posts failed at the RAG stage. Hence, we evaluated
the response generation step only with translated posts.
4.2.1 Translation
The translation stage showed high quality in translating Mandarin
posts to English, answering the first part of RQ1. The Microsoft
Azure AI Translation API failed to translate only 2 of the 382
Mandarin posts; these remained untranslated. The translations of
all other posts retained their original meanings and all essential
information.
4.2.2 Question Extraction
Thequestionnumberextractorcanaccuratelyfetchquestionnumbers
for all categories of questions, answering the first part of RQ2. The
extractorwasdeemedcorrectifitoutputsthecorrectquestionnumber
when the original post (folders, subject, and content) contained
sufficientinformationtoidentifyaspecificquestion. Ifnospecific
questioncouldbedetermined,noquestionnumberwasoutput. As
indicated in the first row of Table 1, the extractor achieved accuracy
ratesof97.78%,90.74%,100%,and97.46%forlogistics,conceptual,
environment setup, and programming questions, respectively.
The extractor also performs comparably well on Mandarin and
translated posts compared to the original English posts. The second
andthirdrowsofTable1showthepercentagesofoverlapbetweenthe
QEresultsforEnglishandMandarin/translatedposts. Thepercent-
agesofoverlapsbetweenEnglishandMandarinpostsare98.89%,
72.22%,100%,and97.97%forthefourcategories,respectively. The
percentages of overlaps between English and translated posts are
95.56%,74.07%,100%,and97.46%. Wesuspectthehighsimilarity
betweenQEresultsfortheMandarinandtranslatedpostsisdueto
the fact that question numbers often appear in abbreviations like "P2
Q10", which remain unchanged in both Mandarin and translated
posts. Mostfailedcasesoccurredintheconceptualcategorywhen
theoriginalpostsdidnotcontainedaquestionnumber,yettheQE
resultsfortheMandarinandtranslatedpostsstillattemptedtofind
one.
4.2.3 RAG
To assess the RAG and Response Generation stages, we evaluated
the top 15% most viewed question posts and randomly selected 10%fromtheremainingpostsfromeachcategory. Thistestsetinclude21
logistics, 14 conceptual, 9 environment setup, and 47 programming
questions,totaling91posts. Thisapproachensuredcoverageofboth
thequestionsofhighestinteresttostudentsandlesscommonqueries
that might present unique challenges.
We computed two metrics from humanevaluation toassess the
RAG’sperformance: relevanceandhelpfulness. Relevancemeasured
whethertheretrievedchunkwaspertinenttothequestionpost,while
helpfulnessdeterminedifthechunkprovidedananswerorsignificant
contextual information to the question.
RAGcannotreplaceQE.Firstly,itcannotdifferentiatebetween
poststhatdoanddonotcontainquestionnumbers. Secondly,itdoes
not perform as well as our question number extractor. Among the
91evaluatedposts,6logistics,6conceptual,and24programming
posts contain a question number. When RAG searched all course
materials includingquizzes andexams, itonly correctlyretrieved 3
(50%), 4 (66.67%), and 20 (83.33%) of the questions within the top
10 searches, respectively. Furthermore, of the 20 correctly retrieved
programming questions, only 14 (70%) are the top 10 RAG results.
This confirms that QE is crucial in our pipeline to accommodate the
specialcaseofstudentqueriesthatreferquestionsbynumbersbut
not content, answering the second part of RQ2. Lastly, including
quizzes and exams in RAG searches tends to overwhelm the results
for conceptual questions with similar quiz and exam questions,
averaging 7.5 out of 10, which obscures the retrieval of relevant
lecturematerials. Therefore,weexcludequizzesandexamsinour
RAG search, reserving it only at the QE stage.
With QE, the RAG results are highly relevant and helpful, which
answers RQ3. Among the 10 RAG retrievals, the average number of
thembeingrelevantare6.1,7.7,7.3,and6.6forlogistics,conceptual,
environmentsetup,andprogrammingquestions,respectively(first
rowofTable2). Additionally,80.95%,71.43%,88.89%,and89.36%
ofthefourcategoriesofquestionpostscontainedatleastoneRAG
retrieval that was helpful to students’ questions.
RAG retrievals also performed well with translated posts but less
so with Mandarin posts. As shown in the third and fourth rows of
Table 2, out of the 10 retrieved documents for all 382 posts, the
averagenumbersofoverlapsbetweenRAGresultsforEnglishand
Mandarin/translated posts are 7.0, 5.7, 7.5, 6.8 and 8.5, 7.0, 8.9, 8.2
forthefourcategories,respectively. Thisindicatesthattranslation
before the RAG stage helps pick up relevant information, addressing
secondpartofRQ1byshowingthattranslationisnecessarytoour
AI tutor system where input and output are all in another language.

logistics conceptual environment setup programming inter-rater reliability
Correctness 80.95% 100% 100% 97.83% 94%
Helpfulness 61.90% 93.33% 88.89% 89.13% 85%
Solution presents in response 0% 0% 0% 13.04% 83%
Translation quality 71.43% 86.67% 77.78% 76.09% 67%
Generates technical term glossary 95.24% 86.67% 77.78% 86.96% 87%
Table 3:Evaluation scores for generated responses
4.2.4 Response Generation
To answer RQ4, we evaluate the generated responses using the
following task-specific metrics:
•Correctness: Whetherthegeneratedoutputisfactuallycorrect
anddoesnotcontainanymisinformation. Thismetrichelpsin
down voting LLM-generated hallucinations.
•Helpfulness: Whether the response effectively addresses the
student’squestion. Ananswercanbefactuallycorrect,butnot
necessarily situated to the student’s question.
•Solution Presents in Response: Whether the response directly
providessolutions,particularlycodesnippets,forprojects. Our
system provides students with hints and directions, not solutions,
which we try to achieve by tuning our prompts.
•Translation Quality: Whetherthetranslationofthegenerated
output from English to Mandarin is free of errors that could
impede understanding. Incomplete translations and missing
translations are also considered as errors.
•Generates Technical Term Glossary: Whetherthegenerated
outputincludesaglossaryofthetechnicaltermsrelatedtothe
questionandtheresponse. ThepromptsenttotheLLMincludes
instructions for generating such a glossary.
Table3presentstheevaluationscoresforthegeneratedresponses,
whereeachmetricisbinary, scoredas0or1. Thelastcolumnofthe
tablepresentstheinter-raterreliabilityforthefivegenerationmetrics.
These reliability scores measured the initial round of evaluations
conducted by two authors and excluding the final adjustments made
bythethirdauthorincasesofconflict. WeutilizedGwet’sAC1to
computethereliabilitybecauseitismorestablethanCohen’sKappa,
particularly when a specific binary rating predominates [ 21]. The
Gwet’sAC1scoresare94%,85%,83%,67%,and87%,reflecting
the consistent evaluation.
In terms of correctness, our pipeline demonstrated a high level of
accuracy,achievingperfectscoresforbothconceptualandenviron-
mentsetupquestions(100%each),andahighscoreforprogramming
questions (97.83%). However, the correctness score for logistics
questions was lower at 80.95%. This reduction often occurred in
situations where students inquired about specific requirements, such
as the necessary formatting for a plot or the use of a particular
function. In the absence of clear guidance from the RAG results,
the model made assumptions rather than acknowledging the lack of
available information.
The helpfulness of the responses, as shown in the second row of
Table 3, varied slightly across categories but generally remained ro-
bust. Responsesadequatelyaddressedstudentquestionsin61.90%of
logistics posts, 93.33% of conceptual posts, 88.89% of environment
setup posts, and 89.13% of programming posts.
The responses generally avoided disclosing direct answers. As
indicated in the third row, none of the logistics, conceptual, andenvironment setup questions included direct solutions. However,
programming questions experienced a modestly higher rate of direct
solution revelation at 13.04%, primarily because these responses
sometimes included parts of code snippets.
Translationqualityiscrucialforthecomprehensionofnon-native
Englishspeakerstudents. ThepipelineprovidedMandarinresponses
without significant errors in most instances, achieving success rates
of71.43%,86.67%,77.78% ,and76.09%forlogistics,conceptual,
environmentsetup,andprogrammingquestions,respectively. Fail-
ures typically occurred when responses were not fully translatedor
lacked Mandarin entirely. When translations were complete, they
maintained high quality consistently.
The inclusion rate of a technical term glossary is reflected in the
final row of Table 3. The results demonstrate that 95.24%, 86.67%,
77.78%, and86.96% ofthequeries inthe fourcategories containa
technical term glossary. When the glossary is provided, the terms
generatedareessentialtothequestions,uncommonlyencountered
in everyday English, and their explanations are accurate.
5 Discussion and Future Work
5.1 Our Contributions
Ourmulti-sourceRAG-basedAItutorsystem,whichutilizesdiverse
course materials and a question extraction mechanism, distinguishes
itself from single-sourced models prevalent in prior work. It can
generate precise, contextually informed responses, achieving an
average helpfulness score of 83.52% across the evaluated Piazza
questions. Additionally, 82.89% of these responses benefit from the
use of question extraction or RAG in generating helpful content.
Over 76.92% of the responses provide quality responses in Man-
darin, and 87.91% of them contain an effective technical term
glossary at the end. Non-native English speaker (NNES) students
may hesitatetoapproachinstructors/TAs dueto languagechallenges
and confidence issues. Our system has the potential to mitigate
suchchallengesbyofferinganon-judgmental,accessibleinterface,
fostering engagement and inclusivity in computing education.
5.2 Limitations and Future Work
For future work, we aim to deploy our AI tutor system to NNES
students in a real classroom setting after making the following
enhancements. We will work on improving accuracy of question
extraction, particularly for translated posts, to ensure all target
questions are retrieved. Additionally, we will implement safeguards
topreventthesystemfrominadvertentlyrevealinganswers,especially
in programming questions where unintended code disclosure can
occur. Tocreateamoreengaginglearningexperience,theAItutor
willbedesignedtointeractinaconversationalmannerratherthan
providing just a single response to posts. After deployment, we plan
to assess the questioning behavior and needs of NNES students and
tailor our system based on the feedback received.
The release of GPT-4o [ 14], notable for its advancements in
vision and non-English language processing, has reduced the cost of

vision services. This development enables us to expand our system
to accept questions accompanied by images. We will also assess
whether our system’s performance is enhanced with GPT-4o and
consider adapting our system for additional languages to test its
validity and extend its applicability.
Giventhetime-consumingandresource-intensivenatureofhuman
evaluation across our system’s multiple stages, we will explore
automatedmethodsofevaluationtoensurethateachcomponentof
our pipeline functions efficiently and effectively.
Moreover,weaimtoensurethatourAItutorevolvesalongsidethe
coursebycontinuouslyupdatingtheRAGvectorstorethroughoutthe
semester to incorporate new course materials. Additionally, we will
integrate mechanisms to detect when students ask about currently
active quiz or exam questions.
Lastly, we intend to develop an interface that allows other instruc-
tors to upload their course materials and customize their own AI
tutor, further broadening the reach and utility of our pipeline.
6 Conclusion
In this work, we developed an AI tutor system that processes student
questions in Mandarin and delivers bilingual responses that are
correct,helpful,andwell-translated,andcompletewithatechnical
termsglossary,withoutdirectlyrevealingtheanswers. Oursystem
utilizes Retrieval-Augmented Generation (RAG) with a special
question extraction mechanism to answer queries from a diverse
array of course materials, including Jupyter Notebooks, Python
scripts, PowerPoint slides, and PDFs. Another main contribution
of our AI tutor system is the design of the prompts that instruct
the GPT-4 model to apporpriately use resources from question
extractionandRAGtoprovidetheappropriatelevelofguidancein
responses. Our evaluation results for our AI tutor on data science
questionsunderscorethepotentialofAItutorstoenhanceeducational
accessibility for non-native English speaking students in computing
courses. Future efforts will focus on refining the system through
deployment in real classroom settings and expanding language
support to further improve student learning experiences.References
[1]P. Denny, J. Leinonen, J. Prather, A. Luxton-Reilly,
T. Amarouche, B.A. Becker, and B.N. Reeves. Prompt prob-
lems: A new programming exercise for the generative ai era.
InProceedings of the 55th ACM Technical Symposium on Com-
puter Science Education V. 1, SIGCSE 2024, page 296–302,
NewYork,NY,USA,2024.AssociationforComputingMachin-
ery. ISBN9798400704239. doi: 10.1145/3626252.3630909.
URLhttps://doi.org/10.1145/3626252.3630909.
[2]P. Guo. Non-native english speakers learning computer pro-
gramming: Barriers,desires,anddesignopportunities. pages
1–14, 04 2018. doi: 10.1145/3173574.3173970.
[3]Y. Hicke, A. Agarwal, Q. Ma, and P. Denny. Ai-ta: Towards
an intelligent question-answer teaching assistant using open-
source llms, 2023.
[4]Institute of International Education. Open doors 2023 re-
port on international educational exchange, 2023. https:
//opendoorsdata.org/annual-release/.
[5]B. Jury, A. Lorusso, J. Leinonen, P. Denny, and A. Luxton-
Reilly. Evaluating llm-generated worked examples in an
introductory programming course. InProceedings of the
26th Australasian Computing Education Conference, ACE
’24, page 77–86, New York, NY, USA, 2024. Association
for Computing Machinery. ISBN 9798400716195. doi:
10.1145/3636243.3636252. URL https://doi.org/10.
1145/3636243.3636252.
[6]M. Kazemitabaar, R. Ye, X. Wang, A. Z. Henley, P. Denny,
M. Craig, and T. Grossman. Codeaid: Evaluating a classroom
deployment of an llm-based programming assistant that bal-
ancesstudentandeducatorneeds. InProceedings of the CHI
Conference on Human Factors in Computing Systems,CHI’24,
NewYork,NY,USA,2024.AssociationforComputingMachin-
ery. ISBN9798400703300. doi: 10.1145/3613904.3642773.
URLhttps://doi.org/10.1145/3613904.3642773.
[7]LangChain.Applicationsthatcanreason.poweredbylangchain.
URLhttps://www.langchain.com/.
[8]W. W. Lau and A. H. Yuen. The impact of the medium of
instruction: The case of teaching and learning of computer
programming.Education and Information Technologies, 16:
183–201, 2011.
[9]P.Lewis,E.Perez,A.Piktus,F.Petroni,V.Karpukhin,N.Goyal,
H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel,
and D. Kiela. Retrieval-augmented generation for knowledge-
intensivenlptasks. InH.Larochelle,M.Ranzato,R.Hadsell,
M.Balcan,andH.Lin,editors,Advances in Neural Information
Processing Systems, volume 33, pages 9459–9474. Curran
Associates, Inc., 2020. URL https://proceedings.
neurips.cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.pdf.
[10]M. Liu and F. M’Hiri. Beyond traditional teaching: Large
language models as simulated teaching assistants in com-
puter science. InProceedings of the 55th ACM Technical
Symposium on Computer Science Education V. 1, SIGCSE
2024, page 743–749, New York, NY, USA, 2024. Asso-
ciation for Computing Machinery. ISBN 9798400704239.

doi: 10.1145/3626252.3630789. URL https://doi.org/
10.1145/3626252.3630789.
[11]R. Liu, C. Zenke, C. Liu, A. Holmes, P. Thornton, and D. J.
Malan. Teaching cs50 with ai: Leveraging generative artifi-
cial intelligence in computer science education. InProceed-
ings of the 55th ACM Technical Symposium on Computer
Science Education V. 1, SIGCSE 2024, page 750–756, New
York, NY, USA, 2024. Association for Computing Machinery.
ISBN 9798400704239. doi: 10.1145/3626252.3630938. URL
https://doi.org/10.1145/3626252.3630938.
[12]S. MacNeil, A. Tran, A. Hellas, J. Kim, S. Sarsa, P. Denny,
S. Bernstein, and J. Leinonen. Experiences from using code
explanations generated by large language models in a web
softwaredevelopmente-book. InProceedings of the 54th ACM
Technical Symposium on Computer Science Education V. 1,
SIGCSE 2023, page 931–937, New York, NY, USA, 2023.
Association for Computing Machinery. ISBN 9781450394314.
doi: 10.1145/3545945.3569785. URL https://doi.org/
10.1145/3545945.3569785.
[13] Meta. Faiss. URLhttps://ai.meta.com/tools/faiss/.
[14]OpenAI. Models. URL https://platform.openai.com/
docs/models.
[15]O. S. Polishchuk. Communication barriers faced by english
language learners at a university level: factors and solutions.
2017.
[16]A. G. S. Raj, K. Ketsuriyonk, J. M. Patel, and R. Halverson.
What do students feel about learning programming using both
english and their native language? In2017 International
Conference on Learning and Teaching in Computing and
Engineering (LaTICE), pages 1–8. IEEE, 2017.
[17]A. G. S. Raj, H. Zhang, V. Abhyankar, S. Mukerjee, E. Zhang,
J. Williams, R. Halverson, and J. M. Patel. Impact of bilingualcs education on student learning and engagement in a data
structures course. InProceedings of the 19th Koli Calling
International Conference on Computing Education Research,
pages 1–10, 2019.
[18]M. Richards, K. Waugh, M. Slaymaker, M. Petre,
J.Woodthorpe,andD.Gooch. Boborbot: Exploringchatgpt’s
answerstouniversitycomputerscienceassessment.ACM Trans.
Comput. Educ., 24(1), jan 2024. doi: 10.1145/3633287. URL
https://doi.org/10.1145/3633287.
[19]S. Sarsa, P. Denny, A. Hellas, and J. Leinonen. Automatic
generation of programming exercises and code explanations
using largelanguagemodels. InProceedings of the 2022 ACM
Conference on International Computing Education Research -
Volume 1, ICER ’22, page 27–43, New York, NY, USA, 2022.
Association for Computing Machinery. ISBN 9781450391948.
doi: 10.1145/3501385.3543957. URL https://doi.org/
10.1145/3501385.3543957.
[20]Y.Shen,X.Ai,A.G.SoosaiRaj,R.J.LeoJohn,andM.Syamku-
mar. Implications of chatgpt for data science education. In
Proceedings of the 55th ACM Technical Symposium on Com-
puter Science Education V. 1, SIGCSE 2024, page 1230–1236,
NewYork,NY,USA,2024.AssociationforComputingMachin-
ery. ISBN9798400704239. doi: 10.1145/3626252.3630874.
URLhttps://doi.org/10.1145/3626252.3630874.
[21]N. Wongpakaran, T. Wongpakaran, D. Wedding, and K. L.
Gwet. A comparison of cohen’s kappa and gwet’s ac1 when
calculatinginter-raterreliabilitycoefficients: astudyconducted
with personality disorder samples.BMC medical research
methodology, 13:1–7, 2013.
[22]W. Xie, J. Niu, C. J. Xue, and N. Guan. Grade like a human:
Rethinking automated assessment with large language models,
2024. URLhttps://arxiv.org/abs/2405.19694.