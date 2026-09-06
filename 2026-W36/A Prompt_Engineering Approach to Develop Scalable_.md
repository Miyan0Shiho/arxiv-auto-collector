# A Prompt-Engineering Approach to Develop Scalable, Flexible, and Real-Time Hybrid Micro-Level Personalization in a General Purpose AI Teaching Assistant

**Authors**: Saptarshi Basu, Sandeep Kakar, Ashok Goel

**Published**: 2026-09-03 06:01:47

**PDF URL**: [https://arxiv.org/pdf/2609.03402v1](https://arxiv.org/pdf/2609.03402v1)

## Abstract
Artificial intelligence (AI) teaching assistants powered by large language models (LLMs) offer scalable educational support but often provide limited personalization. This study presents a prompt-engineering-based framework for personalizing general-purpose LLM/RAG-based AI teaching assistants such as Jill Watson across academic disciplines and courses. The framework adapts responses using six learner-specific dimensions: self-assessment, abstraction preference, verbosity preference, perceptual orientation, information processing style, and level of understanding, yielding 96 distinct learner profiles. Student queries are additionally analyzed using Bloom's Taxonomy to estimate cognitive complexity at the interaction level. Learner attributes and cognitive assessments are encoded in structured prompts that condition the LLM without requiring model retraining. The framework is evaluated through experiments using NLP metrics and a human study with five participants. Results show perceived differences in response style and structure across personalization conditions, with statistical analyses identifying learner attributes associated with measurable response changes. These findings provide preliminary evidence that prompt-based personalization can support adaptive behavior in LLM-powered educational agents.

## Full Text


<!-- PDF content starts -->

A Prompt-Engineering Approach to Develop Scalable, Flexible, and Real-Time
Hybrid Micro-Level Personalization in a General Purpose AI Teaching Assistant.
Saptarshi Basu, Sandeep Kakar, Ashok Goel
Georgia Institute of Technology, Atlanta GA 30332, USA
sbasu7@gatech.edu, skakar6@gatech.edu, ashok.goel@cc.gatech.edu
Abstract
Artificial intelligence (AI) teaching assistants powered by
largelanguagemodels(LLMs)offerscalableeducationalsup-
port but often provide limited personalization. This study
presents a prompt-engineering-based framework for person-
alizing general-purpose LLM/RAG based AI teaching as-
sistant such as Jill Watson across academic disciplines and
courses. The framework adapts responses using six learner-
specific dimensions: self-assessment, abstraction preference,
verbositypreference,perceptualorientation,informationpro-
cessing style, and level of understanding, yielding 96 distinct
learnerprofiles.Studentqueriesareadditionallyanalyzedus-
ing Bloom’s Taxonomy to estimate cognitive complexity at
the interaction level. Learner attributes and cognitive assess-
ments are encoded in structured prompts that condition the
LLM without requiring model retraining. The framework is
evaluated through experiments using NLP metrics and a hu-
man study with five participants. Results show perceived dif-
ferences in response style and structure across personaliza-
tionconditions,withstatisticalanalysesidentifyinglearnerat-
tributes associated with measurable response changes. These
findingsprovidepreliminaryevidencethatprompt-basedper-
sonalization can support adaptive behavior in LLM-powered
educational agents.
Code, Survey Links, and Dataset— https:
//github.gatech.edu/sbasu7/IAAI27_JW_Personalization
Introduction
The aspiration to provide learners with educational expe-
riences tailored to their individual needs is decades old.
Bloom’s2Sigmafindingdemonstratedthatone-to-onetutor-
ing can produce learning gains approximately two standard
deviations above conventional classroom instruction, estab-
lishing personalization as a measurable educational objec-
tive(Bloom1984).Subsequentresearchhassoughtscalable
approaches that approximate the benefits of individualized
instruction through adaptive and intelligent learning sys-
tems (Shute and Zapata-Rivera 2012; Aleven et al. 2017;
Bernacki, Greene, and Lobczowski 2021).
The emergence of large language models (LLMs) has
substantially expanded the potential for personalized learn-
ing. LLM-based teaching assistants can generate fluent,
Copyright©2027, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.contextually relevant responses at scale and, when com-
bined with retrieval-augmented generation (RAG), provide
course-grounded instructional support across diverse disci-
plines(Tanejaetal.2024;Kakaretal.2024;MaitiandGoel
2024). However, their flexibility raises important questions
regardingwhichlearnercharacteristicsshoulddriveperson-
alization, how personalization should be implemented, and
whethersuchadaptationsproducemeaningfullydifferentin-
structional interactions.
This paper addresses these questions through a prompt-
engineering-based personalization framework for the Jill
Watsonvirtualteachingassistant(GoelandPolepeddi2018;
Taneja et al. 2024; Kakar et al. 2024). The framework oper-
atesatthelevelofindividualstudentquestionsandcombines
learner preferences with question-level cognitive demand.
Specifically,responsesarepersonalizedusingsixlearnerdi-
mensions: metacognitive self-assessment, abstraction pref-
erence, verbosity preference, perceptual orientation, infor-
mation processing style, and level of understanding. Cogni-
tive demand is estimated using Bloom’s Taxonomy, while
learnerpreferencesarebasedontheFelder-Silvermanlearn-
ing model (Bloom et al. 1956; Felder and Silverman 1988).
Their combination yields 96 distinct learner profiles.
The framework is implemented entirely through struc-
turedpromptengineeringoveranexistingRAG-basedLLM
tutor, enabling real-time personalization without model re-
training or domain-specific authoring. We evaluate the ap-
proachusing2,910generatedresponsesspanning30student
questions and 97 prompt configurations through NLP-based
analyses, followed by a human evaluation with five partic-
ipants. Results provide preliminary evidence that prompt-
based personalization produces measurable and perceptible
differences in response characteristics, supporting its po-
tential for adaptive behavior in LLM-powered educational
agents.
Literature Review
Bernacki et al. (Bernacki, Greene, and Lobczowski 2021)
proposed a broad personalization framework that character-
izeadaptivelearningthroughfourlenses:bywhom,towhat,
how,andforwhatpurpose.PlassandPawar(PlassandPawar
2020) further distinguish adaptivity (system-driven) from
adaptability (learner-driven), as well as macro- and micro-
level adaptation. The present work focuses on micro-level
arXiv:2609.03402v1  [cs.AI]  3 Sep 2026

cognitiveadaptationthatcombinessystem-drivenclassifica-
tionusingBloom’sTaxonomywithlearner-drivenpreference
selection.
Earlier adaptive learning systems typically follow a
diagnose-prescribe cycle involving learner modeling, ac-
tionselection,andmodelupdating(ShuteandZapata-Rivera
2012).Learnermodelscommonlyrepresentpriorknowledge
and skill mastery, with adaptation primarily implemented
throughcontentselectionorsequencing(Xieetal.2019).In
contrast, this work shifts personalization toward response-
form adaptation: answers remain grounded in a shared re-
trievedknowledgebasewhiletheirabstraction,structure,ver-
bosity, and cognitive framing are modified through prompt
conditioning.
Recent LLM-based tutoring systems have expanded per-
sonalizationthroughconversationalinteractionandretrieval-
augmented generation (RAG), including extensions of Jill
Watson (Taneja et al. 2024; Kakar et al. 2024; Maiti and
Goel 2024). Systems such as LPITutor (Liu et al. 2025b),
PATS (Li et al. 2025b), GPTutor (Chen et al. 2024), and
AgentTutor (Li et al. 2025a) adapt difficulty, personality, in-
structionalcontent,orteachingworkflows.Otherapproaches
integrate LLMs with cognitive diagnosis models to improve
learner modeling (Dong, Chen, and Wu 2025; Liu et al.
2025a; Wei et al. 2025; Zhang et al. 2025). Persona and
preference-aware systems, including Park et al. (Park et al.
2024) and CloChat (Ha et al. 2024), demonstrate the value
of incorporating learner preferences into prompts.
Compared with these approaches, to enable dynamic re-
sponse adaptation at the individual interaction level, the
proposed framework combines six learner dimensions with
question-level cognitive analysis using Bloom’s Taxonomy.
It allows learners to actively specify response character-
istics while automatically adapting cognitive depth. Thus,
rather than primarily adapting content or learning path-
ways, the LLM/RAG based AI teaching assistant (Jill Wat-
son) personalizes how shared instructional content is pre-
sented through prompt engineering. This hybrid integration
of learner-driven adaptability and system-driven cognitive
assessment represents an underexplored direction in LLM-
based educational personalization.
Personalization Framework Design
The proposed personalization framework consists of three
components: learner characteristics, learner preferences,
and a prompt-engineering mechanism that conditions the
underlying large language model (LLM). Following the
profile-conditioned approach of Park et al. (Park et al.
2024), learner characteristics are represented through self-
assessed metacognitive knowledge and the cognitive com-
plexity of individual questions determined using Bloom’s
Taxonomy (Bloom et al. 1956). This enables question-level
personalization based on both learner understanding and
query complexity.
LearnerpreferencesarederivedfromtheFelder-Silverman
learning model (Felder and Silverman 1988) and are treated
as user-selected preferences rather than fixed psychometric
classifications. Five dimensions are incorporated: abstrac-
tion, verbosity, perception, processing, and understanding.Factors Level I Level II Level
III
Metacognition
(Self Assess-
ment)Beginner Intermediate Expert
Abstraction In-Depth
(Techni-
cal)Big Picture
(High-Level)–
Verbosity Verbose Concise –
Perception Sensory Intuitive –
Processing Active Reflective –
Understanding Sequential Global –
Table 1: Learning Preferences and Levels
These dimensions control the granularity, length, commu-
nication orientation, engagement style, and organizational
structure of generated responses, respectively. The corre-
sponding categories are summarized in Table 1.
Thecombinationoflearnercharacteristicsandpreferences
produces96distinctlearnerprofiles.Inaddition,eachstudent
question is automatically classified according to Bloom’s
Taxonomy, enabling dynamic adaptation at the individual
interaction level. Learner preferences are explicitly selected
by students, whereas cognitive demand is inferred by the
system.Thiscreatesahybridframeworkcombininglearner-
driven adaptability with system-driven adaptivity.
Alllearnerattributesareencodedinanengineeredprompt
thatconditionsresponsegeneration.Thepromptisintegrated
with Jill Watson’s retrieval-augmented generation (RAG)
pipeline and course-specific knowledge base, allowing per-
sonalizationtomodifytheformandpresentationofresponses
while preserving grounding in retrieved instructional con-
tent. An example prompt is shown below:
Ihaveabeginnerlevelofknowledgeinthistopic.The
Bloom’sTaxonomycategoryofmyquestionisSynthe-
sis. Please provide atechnicalandconciseresponse,
using asensorycommunication style.I process infor-
mation in areflectiveway and prefer to understand
concepts in aglobalmanner.
Personalize the response based on my understanding
and preferences listed in this prompt. The query is as
follows:whatapproachshouldItaketobestsolvethe
Sheep and Wolves problem?
The modular design allows learner preferences to be up-
datedthroughtheJillWatsoninterfaceandincorporatedinto
prompts at runtime, enabling scalable personalization with-
outmodifyingorretrainingtheunderlyingLLM (Kakaretal.
2024).
Research Questions
The objective of this study is to determine whether the pro-
posed personalization framework produces distinct and per-
ceptibleresponsecharacteristics.Accordingly,weinvestigate
the following research questions (RQs):

1. RQ1: To what extent are learner profiles associated
with differences in the linguistic characteristics of LLM-
generated responses?
2. RQ2: Which learner dimensions are most strongly asso-
ciated with variations in response characteristics?
3. RQ3:Aretheobservedresponsecharacteristicsconsistent
with the intended effects of the corresponding learner
dimensions?
RQ1 examines whether different learner profiles produce
systematicallydifferentresponses.RQ2evaluatestherelative
contribution of individual learner dimensions to response
characteristics, including semantic similarity, complexity,
verbosity, abstraction, and processing style. RQ3 assesses
whetherobserveddifferencesalignwiththeintendedeffects
of each dimension; for example, whether higher verbosity
produces longer responses and higher abstraction produces
more technically complex explanations.
To address these questions, we combine automated NLP
analyses with human evaluation. Descriptive and inferential
statisticalmethods,includingmixed-effectsmodels,areused
to quantify differences and associations across learner pro-
files and dimensions.
Experimental Design and NLP Evaluation
Automated NLP analyses were conducted to determine
whetherpersonalizationdimensionsproducemeasurabledif-
ferences in LLM-generated responses.
Thirty real-world student questions were selected from
the Spring 2023 CS 7637 Knowledge-Based AI (KBAI)
course at the Georgia Institute of Technology, covering all
six Bloom’s Taxonomy categories. Following Maiti and
Goel (Maiti and Goel 2025), questions were classified us-
ing a fine-tuned BERT-based classifier trained on combined
labeled datasets (Gani and Sangodiah 2023; Yahya 2011).
Theclassifierusedbert-base-uncasedandachieved0.92test
accuracy, with F1 scores of 0.88 - 0.94 across categories.
The six learner dimensions and their levels (Table 1) pro-
duced96uniquelearnerprofiles.Foreachofthe30questions,
97 prompts were generated: 96 personalized configurations
and one non-personalized baseline, resulting in 2,910 re-
sponses.Theprompttemplateandmodelconfigurationwere
heldconstant,withonlylearner-profileattributesvaried.Re-
sponses were generated using GPT-4.1 with temperature set
to 0 to minimize stochastic variation.
Responses were evaluated using four NLP di-
mensions: lexical similarity, semantic similarity, lin-
guistic complexity, and verbosity. Semantic similarity
was measured using 384-dimensional embeddings from
all-MiniLM-L6-v2(Wang et al. 2020; Sentence-
Transformers Community on Hugging Face 2024), fol-
lowed by pairwise cosine similarity. Lexical overlap
was measured using ROUGE, linguistic complexity using
grade-level scores fromtextstat, and verbosity using
lexicon_count.
Descriptiveanalysesandordinaryleastsquares(OLS)re-
gression were used to examine associations between learner
dimensions and response characteristics. Together with thesubsequent human evaluation, these analyses address the
three research questions.
Human Evaluation Study Design
Ahumanevaluationstudyinvolvingfiveevaluatorscomple-
mented the automated NLP analysis by assessing response
characteristics that are difficult to capture automatically, in-
cluding perceived abstraction, depth of understanding, and
information-processing style. The evaluators were recruited
from current and former students of the KBAI course at the
Georgia Institute of Technology based on a pre-recruitment
survey. The survey collected information about the evalua-
tors’ subject-matter understanding, self-assessed expertise,
and learning preferences.
One representative question was selected from each
Bloom’s Taxonomy category. The evaluation examined
three personalization dimensions: self-assessment, abstrac-
tion, and processing style across 13 student profiles, includ-
ing a non-personalized baseline. Evaluators were recruited
through a screening survey capturing educational back-
ground,perceivedcompetency,andlearningpreferences,and
all data were anonymized.
UsingQualtrics,eachevaluatorassessedresponsesforall
13profilesacrossthesixquestions,yielding390evaluations.
Responses were rated on four dimensions: overall quality (0
-10scale),perceivedcomplexity(novicetoexpert),abstrac-
tion level (non-technical, neutral, technical), and processing
style (reflective, neutral, active).
For inferential analysis, the baseline was excluded, leav-
ing12personalizedprofiles.Mixed-effectsmodelswereused
with personalization dimensions as fixed effects and evalu-
ator identity as a random effect to account for inter-rater
variability.
Giventhesmallsampleoffiveevaluators,resultsareinter-
preted as exploratory rather than population-level evidence.
The factorial design supports estimation of main effects but
not interaction effects because of limited statistical power.
Largerandmorediversesamplesareneededtoestablishgen-
eralization and examine interactions among personalization
dimensions.
Results and Discussion
Inthissection,wepresentandbrieflydiscusstheresultsfrom
the NLP experiments and human evaluation study.
NLP Experiments Results
To qualitatively illustrate response differentiation, two re-
sponses generated for the question“What approach should
I take to best solve the Sheep and Wolves problem?”are
compared.Thefirstcorrespondstoabeginnerprofileprefer-
ring concise, non-technical, sensory, active, and sequential
explanations.Thesecondcorrespondstoanadvancedprofile
preferringconcise,technical,intuitive,reflective,andglobal
explanations. Snippets from the responses are reproduced
below as representative quotes.
Beginner, sensory, active, sequential profile:“Let’s
break it down into simple steps. Visualize the sce-

Figure 1: Cosine similarity and ROUGE scores for 97 re-
sponses to a representative question.
nario,identifytherules,planyourmoves,testdifferent
strategies.”
Advanced, intuitive, reflective, global profile:“Define
the problem space and constraints, identify primitive
actions,mapstatetransitions,applyexplanation-based
learning, and evaluate the solution globally.”
The responses illustrate qualitative differences in instruc-
tional strategy: the beginner response emphasizes visualiza-
tion, sequential steps, and experiential exploration, whereas
the advanced response emphasizes formal decomposition,
state-space reasoning, and abstraction. This provides qual-
itative evidence of response differentiation under different
personalization configurations (RQ1).
To quantify response variation, all 97 responses for
each question were encoded using SentenceTransformer
all-MiniLM-L6-v2(Wang et al. 2020; Sentence-
Transformers Community on Hugging Face 2024). Pairwise
cosine similarity and ROUGE scores were computed. Fig-
ure 1 shows high semantic similarity but substantially lower
lexical similarity, indicating that responses remain semanti-
callygroundedwhilevaryinginsurface-levelexpressionand
structure.
Figure 2 shows systematic variation in response length
across verbosity preferences, providing evidence that the
Figure 2: Response word count across verbosity preference
categories.
Figure3:OLScoefficientestimatesforresponsecomplexity.
corresponding personalization dimension influences output
length (RQ3).
Complexity was measured using thetextstatgrade-level
scoreandanalyzedusingOLSregressionwithlearner-profile
attributes and Bloom’s Taxonomy categories as predictors.
As shown in Figure 3, higher self-assessment, verbosity, re-
flective processing, and technical abstraction are associated
with greater response complexity. Evaluation and Analysis
questionsalsotendtoproducemorecomplexresponsesthan
other Bloom levels. These results indicate systematic asso-
ciations between personalization dimensions and response
characteristics (RQ2-RQ3).
Overall, the NLP analyses demonstrate measurable vari-
ation in response expression, length, and complexity across
personalization conditions.
Human Evaluation Study Results
Evaluatorsratedthe13responsesforeachquestiononoverall
accuracyandrelevance.Figure4showsvariationbothacross
participants and across responses within participants, indi-
cating that prompt-level personalization produced percepti-
bledifferencesdespiteasharedRAGpipelineandknowledge
base.

Figure 4: Question-specific evaluation scores based on perceived accuracy and relevance.
Figure5:Fixed-effectestimatesforoverallaccuracyandrel-
evance.
Figure6:Perceivedresponsecomplexitybyabstractionpref-
erence.
Alinearmixed-effectsmodelwithpersonalizationfactors
and Bloom’s level as fixed effects and evaluator identity as
a random effect showed that abstraction preference was as-
sociated with perceived response quality. Bloom’s level was
also associated with ratings, with more complex questions
generally receiving lower scores (Fig. 5).
Evaluators also rated perceived response complexity on
a five-level ordinal scale. Technical abstraction preferences
wereassociatedwithhigherperceivedcomplexity(Fig.6).A
Bayesianordinalmixed-effectsmodelconfirmedabstraction
preferenceasasignificantpredictorofperceivedcomplexity,
while self-assessment was not significant (Fig. 7).
Perceivedabstractionratingsgenerallyalignedwiththein-
tendedabstractionpreferences.ThecorrespondingBayesian
ordinalmixed-effectsmodelidentifiedbothabstractionpref-
Figure7:Fixed-effectestimatesforperceivedresponsecom-
plexity.
Figure 8: Fixed-effect estimates for perceived abstraction
level.
erence and Bloom’s level as significant predictors (Fig. 8),
providing evidence that the intended abstraction differences
were perceptible to evaluators.
Similarly, processing preferences were reflected in evalu-
atorratingsofresponseprocessingstyle.Theordinalmixed-
effects model identified both processing preference and
Bloom’slevelassignificantpredictorsofperceivedprocess-
ing style (Fig. 9).
Overall, the human evaluation provides evidence that
prompt-based personalization produces perceptible differ-
ences in response quality, complexity, abstraction, and pro-
cessing style. Abstraction and processing preferences were
particularlyconsistentwiththeirintendedeffects,whileself-

Figure 9: Fixed-effect estimates for perceived processing
style.
assessment showed weaker effects. These findings address
RQ1-RQ3,althoughthesmallevaluatorsamplelimitsgener-
alization.Largerstudiesareneededtoassessrobustnessand
interactions among personalization dimensions.
Path to Deployment
Theproposedpersonalizationmoduleisdesignedforintegra-
tionintotheexistingJillWatsonarchitecturethathasalready
been deployed across multiple institutions, without modify-
ing its core infrastructure (Taneja et al. 2024; Kakar et al.
2024; Maiti and Goel 2024). Development and integration
are targeted for completion by Spring 2027, followed by a
pilotdeploymentinselectedGeorgiaInstituteofTechnology
courses in Summer 2027. The pilot will evaluate real-world
performance and user feedback, informing subsequent re-
finement and broader deployment across additional courses
and institutions in Fall 2027 - Spring 2028.
Conclusions
This paper presents a personalization framework for RAG
and LLM based AI teaching assistants that enables flex-
ible, scalable, modular, and real-time customization. The
proposedframeworkemphasizesahybridapproachbetween
adaptabilityandadaptivity,enablingmicro-levelcustomiza-
tion at the interaction level. Key contributions include:
1. An engineered prompt that incorporates student cogni-
tive ability, question complexity (Bloom’s Taxonomy),
and learning preferences, generating 96 unique response
configurationsforquestion-level(micro)personalization.
2. Real-time adaptation of prompts based on learner-
selectedpreferences,supportedbysystem-levelcognitive
assessment using Bloom’s Taxonomy and a fine-tuned
BERT-based classifier at each interaction.
3. Amodulardesignthatenablesflexibleintegrationofaddi-
tional features, parameters, and prompt structures within
the LLM/RAG (Jill Watson) architecture.
4. Scalabilitytosupportdiverselearnermodels,knowledge
bases, question banks, courses, and institutional settings
without requiring domain-specific adaptation.
5. This study addressed three research questions related to
whether personalization leads to measurable responsevariation,whichlearnerfactorsdriveresponsedifferenti-
ation,andwhetherlearnerpreferencesalignwithintended
responsecharacteristics.NLP-basedexperimentsandhu-
manevaluationstudiesshowedsystematicresponsevari-
ation across conditions and identified key student profile
parameters associated with changes in LLM-generated
responses.
Thecurrentworkprimarilyfocusesontheproposedframe-
work’s ability to personalize general-purpose LLM/RAG
based AI teaching assistant’s responses to individual stu-
dent questions. Future work includes deployment of a UI-
integrated personalized AI teaching assistant for large-scale
classroomevaluation,A/Btesting,andassessmentofimpacts
on learning outcomes and student engagement.
AcknowledgmentsThis research has been supported by
NSF Grants 2112532 and 2247790 to the National AI Insti-
tuteforAdultLearningandOnlineEducationheadquartered
at Georgia Institute of Technology, Atlanta.
References
Aleven,V.;McLaughlin,E.A.;Glenn,R.A.;andKoedinger,
K. R. 2017. Instruction Based on Adaptive Learning Tech-
nologies. InMayer,R.E.;andAlexander,P.A.,eds.,Hand-
book of Research on Learning and Instruction, 522–560.
New York, NY: Routledge, 2nd edition.
Bernacki,M.L.;Greene,M.J.;andLobczowski,N.G.2021.
A Systematic Review of Research on Personalized Learn-
ing: Personalized by Whom, to What, How, and for What
Purpose(s)?Educational Psychology Review, 33(4): 1675–
1715.
Bloom, B. S. 1984. The 2 Sigma Problem: The Search for
Methods of Group Instruction as Effective as One-to-One
Tutoring.Educational Researcher, 13(6): 4–16.
Bloom,B.S.;Engelhart,M.D.;Furst,E.J.;Hill,W.H.;and
Krathwohl, D. R. 1956.Taxonomy of Educational Objec-
tives:TheClassificationofEducationalGoals.HandbookI:
Cognitive Domain. New York: Longman.
Chen, E.; Huang, R.; Chen, H.-S.; Tseng, Y.-H.; and Li,
L.-Y. 2024. GPTutor: Great Personalized Tutor with Large
LanguageModelsforPersonalizedLearningContentGener-
ation. InCompanion Proceedings of the ACM Web Confer-
ence.
Dong, Z.; Chen, J.; and Wu, F. 2025. Knowledge is Power:
HarnessingLargeLanguageModelsforEnhancedCognitive
Diagnosis.arXiv preprint arXiv:2502.05556.
Felder, R. M.; and Silverman, L. K. 1988. Learning and
TeachingStylesinEngineeringEducation.EngineeringEd-
ucation, 78(7): 674–681.
Gani, M. O.; and Sangodiah, A. 2023. Exam Question
Datasets. Figshare.
Goel, A. K.; and Polepeddi, L. 2018. Jill Watson: A Virtual
Teaching Assistant for Online Education. InLearning En-
gineering for Online Education: Theoretical Contexts and
Design-Based Examples. Routledge.

Ha,J.;Jeon,H.;Han,D.;Seo,J.;andOh,C.2024. CloChat:
Understanding how people customize, interact, and experi-
ence personas in large language models. InProceedings of
the 2024 CHI Conference on Human Factors in Computing
Systems, 1–24. ACM.
Kakar, S.; Maiti, P.; Taneja, K.; Nandula, A.; Nguyen, G.;
Zhao, A.; Nandan, V.; and Goel, A. K. 2024. Jill Watson:
Scaling and Deploying an AI Conversational Agent in On-
line Classrooms. InInternational Conference on Intelligent
Tutoring Systems, 78–90. Springer Nature Switzerland.
Li, C.; et al. 2025a. AgentTutor: Empowering Personalized
LearningwithMulti-TurnInteractiveTeachinginIntelligent
EducationSystems. InProceedingsoftheAAAIConference
on Artificial Intelligence.
Li, M.; et al. 2025b. PATS: Personality-Aware Teach-
ing Strategies with Large Language Model Tutors.arXiv
preprint.
Liu, Y.; et al. 2025a. LMCD: Language Models are
Zero-Shot Cognitive Diagnosis Learners.arXiv preprint
arXiv:2505.21239.
Liu, Z.; Agrawal, P.; Singhal, S.; Madaan, V.; Kumar, M.;
and Verma, P. K. 2025b. LPITutor: An LLM-Based Person-
alized Intelligent Tutoring System Using RAG and Prompt
Engineering.PeerJ Computer Science, 11: e2991.
Maiti, P.; and Goel, A. 2025. Can an AI Partner Empower
Learners to Ask Critical Questions? InProceedings of the
30thInternationalConferenceonIntelligentUserInterfaces
(IUI), 314–324.
Maiti, P.; and Goel, A. K. 2024. How Do Students Interact
with an LLM-Powered Virtual Teaching Assistant in Differ-
entEducationalSettings?arXivpreprintarXiv:2407.17429.
Park,M.;Kim,S.;Lee,S.;Kwon,S.;andKim,K.2024. Em-
powering Personalized Learning Through a Conversation-
BasedTutoringSystemwithStudentModeling. InExtended
AbstractsoftheCHIConferenceonHumanFactorsinCom-
puting Systems, 1–10.
Plass, J. L.; and Pawar, S. 2020. Toward a Taxonomy of
AdaptivityforLearning.JournalofResearchonTechnology
in Education, 52(3): 275–300.
Sentence-Transformers Community on Hugging Face.
2024. all-MiniLM-L6-v2 SentenceTransformer Model.
https://huggingface.co/sentence-transformers/all-MiniLM-
L6-v2. Accessed: 2026-06-26.
Shute, V. J.; and Zapata-Rivera, D. 2012. Adaptive Educa-
tional Systems. In Durlach, P. J.; and Lesgold, A. M., eds.,
Adaptive Technologies for Training and Education, 7–27.
Cambridge, MA: Cambridge University Press.
Taneja,K.;Maiti,P.;Kakar,S.;Guruprasad,P.;Rao,S.;and
Goel,A.K.2024. JillWatson:AVirtualTeachingAssistant
PoweredbyChatGPT. InArtificialIntelligenceinEducation
(AIED 2024). Springer.
Wang,W.;etal.2020. MiniLM:DeepSelf-AttentionDistil-
lation for Task-Agnostic Compression of Pre-Trained Trans-
formers.arXiv preprint arXiv:2002.10957.
Wei,G.;etal.2025. LLM4CD:LeveragingLargeLanguage
Models for Open-World Knowledge Augmented Cognitive
Diagnosis.arXiv preprint arXiv:2505.13492.Xie, H.; Chu, H.-C.; Hwang, G.-J.; and Wang, C.-C. 2019.
Trends and Development in Technology-Enhanced Adap-
tive/PersonalizedLearning:ASystematicReviewofJournal
Publications from 2007 to 2017.Computers & Education,
140: 103599.
Yahya, A. 2011. Bloom’s Taxonomy Cognitive Levels Data
Set. Data set available at ResearchGate. Dataset.
Zhang, Y.; et al. 2025. LLM-CDM: A Large Language
Model Enhanced Cognitive Diagnosis for Intelligent Edu-
cation.IEEE Transactions on Learning Technologies.