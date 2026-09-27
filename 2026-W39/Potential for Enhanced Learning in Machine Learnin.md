# Potential for Enhanced Learning in Machine Learning Classes by Using Wiki LLM Indexing

**Authors**: Brian Wright

**Published**: 2026-09-21 18:53:37

**PDF URL**: [https://arxiv.org/pdf/2609.25303v1](https://arxiv.org/pdf/2609.25303v1)

## Abstract
Large language models are increasingly deployed as course-specific tutors, but their usefulness depends on grounding in vetted instructional materials that are often revised mid-semester. Our prior work built a multimodal retrieval-augmented generation (RAG) system over an authentic machine learning course corpus (Foundations of Machine Learning) and found that retrieval improved contextual grounding, but that fixed retrieval strategies were suboptimal. That motivates a different question: whether how a corpus is structured at ingest time matters more than how much is retrieved at query time. We present a controlled head-to-head comparison of two knowledge representations over an identical classroom corpus: (A) vector RAG, replicating the best-performing configuration from our prior study, and (B) an LLM-compiled wiki (Karpathy framework), in which the corpus is synthesized at ingest into linked concept pages with explicit cross-references and citations back to source materials. We evaluate 59 questions spanning single-fact recall, cross-unit concept linking, synthesis and explanation, and currency after a syllabus revision, scored by an LLM judge against a human-authored rubric. Both representations answered single-fact questions about equally well (9.33 vs. 9.96 of 10), but diverged sharply on questions requiring links across course units. The compiled wiki remained accurate and grounded (9.93; 100% grounded in cited sources), while retrieval scored lower and was markedly less grounded (8.14; 64%). The wiki's citations let students and instructors trace any claim back to the lecture that introduced it, adding a layer of dynamic retrieval that machine learning courses require. While further testing is needed, instructors using AI to support learning in ML courses should consider wiki-based structure for its potential to support foundational elements of best practice.

## Full Text


<!-- PDF content starts -->

Potential for Enhanced Learning in Machine Learning Classes by Using Wiki
LLM Indexing
Brian Wright
School of Data Science, University of Virginia
brianwright@virginia.edu
Abstract
Large language models are increasingly deployed as course
specifictutorstoenhancestudentlearning,buttheirusefulness
depends on being grounded in vetted instructional materials
thatareoftenupdatedduringthesemester.Ourpriorworkbuilt
a multimodal retrieval augmented generation (RAG) system
over an authentic machine learning course corpus (Founda-
tionsofMachineLearning)andfoundthatretrievalimproved
contextual grounding, but that fixed retrieval strategies were
suboptimal.Thatresultmotivatesadifferentquestiononhow
a corpus is structured at ingest time matters more than how
muchisretrievedatquerytimeandhowthisfacilitateslearn-
ing. We present a controlled head to head comparison of two
knowledgerepresentationsoveranidenticalclassroomcorpus:
(A)vectorRAG,replicatingthebestperformingconfiguration
fromourpriorstudy,and(B)anLLMcompiledwiki,(Karpa-
thy Framework) in which the corpus is synthesized at ingest
into linked concept pages with explicit cross references and
citations back to source materials. We evaluate 59 questions
spanning single fact recall, cross unit concept linking, syn-
thesisandexplanation,andcurrencyafterasyllabusrevision,
scored by LLM judge with a human authored rubric. Both
representations answered single fact questions about equally
well (9.33 vs. 9.96 out of 10), but they diverged sharply on
questions that required linking material across course units.
Thecompiledwikistayedaccurateandgrounded(9.93,100%
grounded in cited source material) while retrieval’s answers
both scored lower and were markedly less grounded (8.14,
64% grounded). The compiled representation’s citations let a
student and instructor trace any claim back to the lecture that
introduced it and adds a authentic layer of dynamic informa-
tion retrieval that is necessary in machine learning courses.
For pedagogical practice, the results indicate a signal for ad-
vantages of the Wiki based structure. While further testing
is needed, instructors teaching ML courses when using AI
methodstoenhancestudentlearningshouldconsiderthewiki
structure as it has great potential for supporting foundational
elements of best practices in learning.
1 Introduction
1.1 Motivation
Large language models are rapidly being adopted as course
specific tutoring tools. Left ungrounded, however, a general
Copyright©2027, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.purposeLLManswersfromparametricknowledgethatmay
conflict with how a particular course defines, sequences, or
scopes its content; effective educational deployment to en-
hance learning outcomes therefore requires grounding re-
sponsesinvettedcoursematerialssothatanswersalignwith
the local curriculum and instructional level (Li et al. 2025;
Jain, Cui, and Chen 2025). This is especially true of cut-
tingedgecoursesthatfocusesonmachinelearningandmore
broadly AI, because the field changes so quickly is almost
impossibletodevelopacanonicaltextthateveryonecanuse.
Retrieval augmented generation (RAG) is the standard
mechanism for such grounding (Lewis et al. 2020). In our
prior work, we built a multimodal RAG system over au-
thentic materials from a machine learning course (DS3001)
including:lectureslides,lectureaudiotranscripts,andcourse
readingsandfoundthatretrievalsubstantiallyimprovedcon-
textual grounding (Wright et al. 2026). But the same study
surfaced a more troubling result:fixed retrieval strategies
are suboptimal. Performance depended sharply on question
specificity and on how the context window was composed.
Swapping retrieved text for top ranked images achieved per-
fectcontextrecallonspecificquestionsunderafivetext,five
imageconfiguration,yetthesamestrategydegradedfaithful-
ness and factual correctness on generic questions. No single
retrieval configuration served all question types well.
This finding reframes the design problem. Vector RAG
makes its structural commitments at query time: it retrieves
isolated, similarity ranked chunks and asks the generator to
assemble them into a coherent answer. An alternative is to
makethosecommitmentsatingesttimecompilingthecorpus
intoastructured,linkedwikiofconceptpages,withexplicit
cross references and citations back to the raw source mate-
rials, which the model then navigates when answering. This
“LLMcompiledwiki”approachtradesquerytimechunkre-
trievalforcuratedsynthesis,explicitconceptlinks,andbuilt
inprovenance.Ifhowcontextisstructuredandselectedmat-
tersmorethanhowmuchisretrieved,thentherepresentation
itself,nottheretrievalpolicy,maybethemoreconsequential
design choice.
1.2 Effective Teaching Practices with Chatbots
Groundingachatbotincoursematerialsisnecessarybutnot
sufficient for it to teach well. A growing body of classroom
deployment research argues that how a tutor is designed to
arXiv:2609.25303v1  [cs.AI]  21 Sep 2026

interact matters as much as what it can retrieve. Scaffolded,
Socratic designs that pose guiding questions rather than re-
solving a problem outright have been shown, in a deployed
undergraduate CS tutoring system, to shift students from
vague help seeking toward more deliberate, strategic use of
thetool(SunilandThakkar2025).Teacherpresenceinhowa
chatbotbasedactivityisdesignedsimilarlyshapeshowmuch
students engage with it (Li, Wu, and Chiu 2025). These ef-
fects,however,donottransfercleanlyfromcontrolledstudies
to real classrooms. An analysis of thousands of real student
tutorconversationsfoundapersistentmismatchbetweenthe
scaffolding behavior benchmarks assume and how students
actuallysteerrealinteractionstowardtheirowngoals(Neagu
et al. 2026) an argument for evaluating on authentic class-
room questions over authentic course material, as we do
here, rather than on synthetic benchmark sets. Practitioner
derived guidance converges on a related point, generative
AI should be taught and used in ways that promote discern-
ment and critical thinking rather than substituting for them
(Wall, Bedford, and Redmond 2025) and a broader review
of classroom deployments finds outcomes depend heavily
on how the tool is embedded in instruction rather than on
model capability alone (Wang, Zainuddin, and Leng 2025).
RecentfieldstudiesofinstructorsbuildingcoursespecificAI
tutorsatinstitutionalscalereportthatturningthesecommit-
ments into practice is difficult, and that instructors’ biggest
challenges center on getting a tutor to reflect a course’s own
structureandsequencingratherthanjustansweringquestions
correctly in isolation (Ko et al. 2026).
That last challenge is where knowledge representation
stops being a retrieval engineering detail and becomes a
pedagogical one. A tutor that scaffolds understanding, cites
its sources, and helps a student see how today’s material
builds on an earlier lecture needs access to a course’s con-
nectivestructure,itsdefinitions,itssequencing,anditscross
references not just to the passage nearest a query. This is
exactly the structure an LLM compiled wiki makes explicit
at ingest time, while vector RAG reconstructs it, if at all,
only implicitly through whatever happens to be retrieved.
Thecomparisoninthispaperisdesignedtotestwhetherthat
difference in representation actually shows up on the ques-
tion type teaching practices often care about, not isolated
recall, but questions that require connecting ideas across a
semester’s worth of material that given the field is almost in
constant flux.
1.3 The Gap
Head to head comparisons of compiled wikis against vector
RAG are beginning to appear. Cochran (2026) preregistered
suchacomparisononasmallmultidomainresearchcorpus,
findingthatvectorRAGwononsinglefactlookupwhilethe
wikiwononcrosspapersynthesis,withnoarchitecturedom-
inating every endpoint. But no such comparison exists in an
educationalsettingwithauthenticclassroomdata,wherethe
corpus is multimodal, the question distribution is pedagogi-
callyshaped,andalignmentwithvettedmaterialsmattersas
much as raw correctness (Jain, Cui, and Chen 2025). Mean-
while,thegraphandstructureaugmentedretrievalliterature
suggests that structure helps multi hop and relational ques-tionsbutcanhurtsimplefactlookup(Pengetal.2024;Zhou
et al. 2025; AboulEla et al. 2025) is a trade off that remains
untested on course corpora.
1.4 Research Questions
We ask three questions.RQ1:Does an LLM compiled wiki
improve response quality over vector RAG on classroom
questionanswering,holdingthegeneratorLLM,corpus,and
question set constant?RQ2:How do gains vary by question
type, single fact lookup vs. multi hop concept linking vs.
synthesis and lead to extending the generic/specific split of
our prior study into a finer taxonomy?RQ3:Do the results
support one approach versus the other as it relates to using
AI for learning in a machine learning class?
1.5 Contributions
This paper makes three contributions: (1) the first wiki vs
RAGcomparisononanauthenticcoursecorpus,usingatext
onlysubsetofDS3001;(2)afullyspecifiedevaluationproto-
colandafivepartquestiontaxonomyandmultiendpointmet-
ric suite extending our prior 30 question generic/specific set
(Section3)and(3)directionalempiricalevidencethatstruc-
ture at ingest most benefits cross topic synthesis questions,
including a groundedness without retrieval failure finding
thatcomplicatestheusualretrievalcentricaccountofRAG’s
shortcomings. All of which helps to inform best teaching
practices for instructors leading machine learning courses
that are working to include generative AI approaches for
learning.
2 Background
2.1 Learning Outcomes from AI Chatbots
The case for classroom chatbots ultimately rests on whether
they improve learning, not just whether they answer accu-
rately. The experimental evidence here is larger and more
consistent than the accuracy literature alone: a meta anal-
ysis of 24 randomized studies found AI chatbots produced
a large effect on students’ learning outcomes, with stronger
effectsinhighereducationthaninK12settingsand,notably,
strongereffectsforshorterinterventionsthansustainedones
(Wu and Yu 2024). A more recent meta analysis specific
to ChatGPT, pooling 35 experimental studies, found a mod-
erate to large effect on both cognitive and non cognitive
outcomes, with the instructional mode a chatbot is embed-
ded in the course design not merely its presence among the
significant moderators (Wu et al. 2026). Causal evidence
points the same direction, a randomized controlled trial in
an authentic physics classroom found AI tutored students
outperformed those in active learning in class sessions on
the same material (Kestin et al. 2025). Retrieval augmented
chatbots specifically, not just chatbots in general, show this
pattern too. A RAG grounded course chatbot deployed in a
materials science course was found to measurably enhance
students’ learning, not merely their perception of it (Thway
et al. 2025).
Twothingsfollowfromthisliteratureforthepresentstudy.
First, the implementation of these approaches center on en-
hancing learning outcomes through design choices and how
2

the tool is embedded rather than raw model capability, re-
inforcing the teaching practices argument of Section 1.2.
Knowledgerepresentationisonesuchdesignchoice,andits
effect on learning outcomes can be testable rather than as-
sumed.Second,groundingachatbotincoursematerialisnot
anendinitselfbutameanstoprotectthelearningoutcomes
sinceanungroundedorfabricatingchatbotrisksreproducing
the curriculum misalignment that motivated our prior RAG
work (Wright et al. 2026). The rest of this section reviews
the retrieval mechanisms available for that grounding.
2.2 Retrieval Augmented Generation
RAGconditionsageneratorondocumentsretrievedfroman
external corpus, originally formulated for knowledge inten-
sive NLP tasks (Lewis et al. 2020). It is now widely framed
as a practical reliability layer: grounding outputs in external
evidence, reducing hallucination, and enabling knowledge
updates without retraining (Peng et al. 2024; Neha, Bhati,
and Shukla 2025; Li, Yuan, and Zhang 2024). Domain re-
sultscanbestrikingforexample;aclinicalGPT4RAGsys-
temreached96.4%accuracywithnoobservedhallucinations
(Keetal.2025),andtheMEGARAGpipelineoutperformed
bothLLMonlyandstandardRAGbaselinesinpublichealth
question answering (Xu et al. 2025), yet the gains are con-
tingent. Systematic reviews and unified framework analyses
consistentlyfindthatRAG’sbenefitdependsonretrievalrel-
evance, task type, and evaluation design (Zhou et al. 2025;
Brown, Roman, and Devereux 2025). These contingencies
are precisely what our prior study observed at the level of
retrievalconfiguration,andtheymotivatetreatingtheknowl-
edge representation as an experimental variable in its own
right. For a classroom deployment the stakes of that contin-
gencyaretheoutcomegainsreviewedinSection2.1:aRAG
systemthatscoreswellonitsownevaluationmetriccanstill
fail to translate into better learning if its retrieval strategy is
mismatched to how students actually ask questions.
2.3 Structured and Graph Based Retrieval
Graph enhanced RAG augments or replaces chunk retrieval
with explicit entity and relation structure, supporting multi
hopreasoningandimprovingexplainability(Pengetal.2024;
Zhuetal.2025b).Theempiricalevidence,however,ismixed
in an instructive way. KG2RAG improves answer and re-
trieval F1 on HotpotQA over hybrid RAG baselines (Zhu
etal.2025a);knowledgegraphextendedRAGimprovesmulti
hopMetaQAaccuracyatthecostofaslightsinglehopdegra-
dation (Linders and Tomczak 2025); and graph RAG has
been found tounderperformnaïve RAG on SQuAD V2 and
TriviaQA even while handling complex relational queries
better(AboulElaetal.2025).Therecurringpatternstructure
helpscomposition,hurtslookupandtopicconnectionwhich
negatively effects learning. Consequently, this is the central
trade off our design tests on course content from a machine
learning course.
LLM compiled wikis are a distinct point in this design
space:ratherthanextractingagraphovertheoriginalchunks,
the corpus is rewritten at ingest into human readable, linked
concept pages with citations to source material. In the only
controlledcomparisontodate,Cochran(2026)foundvectorRAGbetteratsinglefactlookupandthewikibetteratcross
paper synthesis, with no single architecture winning every
dimension.
2.4 RAG in Education
Educationspecificworkaddsaconstraintthatgenericbench-
marksmiss:retrievalshouldaligntovettedcoursematerials
and the local curriculum, not merely to generally correct in-
formation(Lietal.2025;Jain,Cui,andChen2025).There-
lationshipbetweengroundingandqualityisalsononmono-
tonic in math tutoring, humans preferred RAG grounded
responsesunlessgrounding became so rigid that helpful-
ness suffered (Levonian et al. 2023). On the structured side,
an educational knowledge graph plus agentic RAG system
achieved 91.4% retrieval accuracy with improved learner
satisfaction (Gao et al. 2025), and hybrid RAG with in con-
textlearningoutperformedbaselinesforeducationalquestion
generation (Maity, Deroy, and Sarkar 2024). Together these
resultssuggeststructurehaspedagogicalvalue,butnostudy
has isolated the representation itself on an authentic course
corpus.
2.5 Prior Work: Multimodal RAG on DS3001
This study extends our prior evaluation of multimodal re-
trieval over DS3001 course data (Wright et al. 2026). That
workshowedthatmultimodalretrievalimprovedcontextre-
call, and that selectively replacing retrieved text with top
ranked images helped specific questions reaching perfect
context recall at five text plus five image chunks while hurt-
ingfaithfulnessandfactualcorrectnessongenericquestions.
The takeaway that motivates the present study is thathow
context is structured and selected matters more than how
much is retrieved. If context composition dominates context
volume,thenaturalnextexperimentcomparesaningesttime
structuredrepresentationagainstquerytimechunkretrieval.
2.6 Evaluating RAG Systems
Lexical overlap metrics such as BLEU and ROUGE are un-
stable under LLM output stochasticity, motivating LLM as
judgeandembeddingbasedevaluation(Lyuetal.2024).We
retain the RAGAS framework (ExplodingGradients 2025)
used in our prior paper for continuity, and follow the multi
endpointreportingofCochran(2026)scoringaccuracy,syn-
thesisquality,citationalignment,andcostseparatelytoavoid
winnertakeallconclusionsthatsingleaggregatescoresinvite
(Lyu et al. 2024).
3 Study Design
We ran a controlled head to head comparison: same corpus,
same base LLM, same prompts, same question set; only the
knowledge representation differs. The design is modeled on
the preregistered comparison of Cochran (2026), combined
withtheclassroomfocusedevaluationapproachofJain,Cui,
and Chen (2025), including its knowledge shift testing.
The corpus comprises the DS3001/DS 3021 course ma-
terials used in our prior study: lecture slides (text and im-
ages),lectureaudiotranscripts,andassignedMLpapersand
textbook excerpts. We reuse the prior extraction pipeline
3

unchanged, so that differences between arms cannot be at-
tributed to preprocessing.
ArmA:VectorRAG(control).Areplicationofourprior
pipeline:textchunkedat1,500tokenswith100tokenoverlap,
embeddedwithallmpnetbasev2,indexedinPinecone.The
bestperformingconfigurationfromthepriorstudyservesas
the baseline, making Arm A a strong rather than straw man
control.
Arm B: LLM compiled wiki.The same corpus is com-
piledatingestintoalinkedwiki(araw/+wiki/structure):
concept pages synthesized by an LLM, cross links between
related pages, and citations from every page back to the raw
sourcematerials.AtquerytimetheLLMnavigatesandreads
relevantpagesratherthanreceivingsimilarityrankedchunks.
A hybrid arm (wiki navigation plus vector retrieval) is
deferred to future work; this study is a clean two arm com-
parison. Held constant across arms: generator LLM, system
prompt, matching context, question set, and decoding pa-
rameters. Matching the context budget matters because the
wikiconditionotherwiserisksconflatingrepresentationwith
contextvolumeanLLMhandedanentirecompiledcorpusis
tested on reading comprehension, not on the representation.
We extend the prior 30 question generic/specific set into
a five part taxonomy: (a) single fact lookup, (b) multi hop
concept linking, (c) thematic explanation and synthesis, (d)
contradictionandambiguityhandling(Houetal.2024),and
(e) curriculum update questions probing sensitivity to re-
visedmaterials.Questionswerehumangeneratedwithrefer-
enceanswersandwordlimits,asbefore,andbalancedacross
course topics.
Wegeneratedaninitialpoolof145questionsoverthefull
corpusandfilteredto59thattestcoursecontentspecifically.
Ofthe59retained,45areanswerablefromasinglewikipage
(topic-pagequestions) and 14 required connecting material
across pages (cross-pagequestions). Each question is fur-
ther labeled by the reasoning it demands: factual (n= 18),
single-fact lookup a slide or transcript already states; con-
ceptual (n= 18), explanation of a concept in its own terms;
synthesis (n= 18), integrating material across a topic or
across pages (n= 5); sensitivity to a recently revised or up-
dated point in the curriculum. The 45 topic-page questions
are balanced evenly across three pages—ml-bias,knn,
anddecision-trees(15 each) and the remaining 14
draw on relationships spanning all three.
For continuity with the prior paper we retain the RAGAS
core metrics: context recall, faithfulness, and factual cor-
rectness (ExplodingGradients 2025). We add four endpoint
families: claim citation alignment (does each claim trace to
a cited source?); answer synthesis quality via blinded LLM
judgescoringwithahumanrubric;robustnessunderknowl-
edge shift, following Jain, Cui, and Chen (2025) and Hou
et al. (2024); and cost, measured as ingest compute, query
tokens, and latency.
Both indexes are built from a frozen corpus snapshot. All
questions are run through each arm; metric stability is esti-
matedbybootstrappedsampling(20,000rounds),separating
questionsamplingerrorfromanswererandjudgestochastic-
ity. Synthesis and preference endpoints use blinded judging
with arm labels hidden.Wegeneratedthreehypotheses.H1:vectorRAGperforms
at least as well as the wiki on single fact lookup.H2:the
wiki outperforms vector RAG on multi hop and synthesis
questions.H3:acrossendpointstheoutcomeisasetoftrade
offs rather than a single winner, consistent with Cochran
(2026).
The corpus actually evaluated is the subset of DS 3021
currentlycompiledintothewiki:12pages,7ofthemcourse
conceptpages(KNearestNeighbors,DecisionTrees,Model
Evaluation Metrics, Decision Tree Regression, K Means
Clustering,EnsembleMethods/RandomForest,andMLBias
& Fairness) and 5 navigational/meta pages (index, getting
started, status, and two course level overview pages). For
Arm A, this corpus chunks into 21 vectors at the configura-
tion above (1,500 tokens, 100 token overlap, all mpnet base
v2, topk= 5) small enough that retrieval failure is possible
but not guaranteed, unlike an earlier 9 vector version of this
corpus (3 topics only) where topk= 5covered more than
half the index on every query and could not meaningfully
fail.
Both arms use the same answerer (claude opus 5)
and the same judge (gpt 5 mini, a different provider
from the answerer so the model is not grading its own out-
put),holdingthegeneratorconstantacrossarmsasspecified
above.InplaceoftheRAGAStriad,thejudgeherereturnsa
single 1 to 10 correctness score plus a binary groundedness
flag(whethereveryclaimintheanswertracestothematerial
the answerer actually saw) a lightweight proxy for RAGAS
factualcorrectnessandfaithfulnessrespectively,notasubsti-
tuteforthem.Thejudgegradesagainstthefullcorpusinboth
arms,soonlytheanswerer’scontextwiki(ArmB)versustop
kretrieved chunks (Arm A) differs between conditions.
Alongside a 1–10 holistic quality score, the judge assigns
each answer a binarygroundedlabel. Whether every claim
in the answer is attributable to the context the model was
given (the retrieved chunks for Arm A, the wiki pages nav-
igated for Arm B) rather than drawn from the model’s own
parametric knowledge. Grounding and quality are concep-
tually distinct such that an answer can be fluent and largely
correct while still relying on information outside what was
actually retrieved, the failure mode we are most concerned
with in a classroom setting, where a wrong-but-checkable
answer is preferable to a confident, plausible one a student
cannot verify against course materials. The judge’s sensi-
tivity to this distinction was validated on four answers of
known quality (excellent, partial, unsupported, fabricated)
before scoring began: it discriminated correctly across all
four,andthefabricatedanswerwastheonlyonescoredboth
low (1/10) and ungrounded, confirming the label responds
togenuinefabricationratherthancovaryingwiththeholistic
score. In the wiki arm specifically, the flag fired grounded
on 98% of answers (n= 59), this reflects the arm having
comparativelylittleoccasiontofabricateanditbecomesdis-
criminatingoncecomparedagainstArmA,wheretherateis
81%.
4 Results
Table 1 shows the wiki ahead of vector RAG on both mea-
sures at this scale, with the larger gap in groundedness (17
4

Arm A (RAG) Arm B (Wiki)
Avg. score (1 10) 9.05 [8.49, 9.54]9.95[9.88, 10.00]
Grounded rate 81% [71%, 90%]98%[95%, 100%]
Table 1: 95% bootstrap CIs (paired question resampling,
B=20,000),59questions,botharmsgraded59/59withzero
errors.
Arm A (RAG) Arm B (Wiki)
Score Grounded Score Grounded
Topic page (n=45) 9.33 87% 9.96 98%
Cross page (n=14) 8.14 64% 9.93 100%
Table 2: Results split by question class. The wiki’s advan-
tage over RAG is roughly three times larger on cross page
questions than on topic page questions, in both score and
groundedness.
points, 95% CI on the difference [7, 27] points) than in raw
score (0.90 points, 95% CI [0.42, 1.44]) both CIs exclude
zero consistent with the qualitative pattern that RAG can
produce a fluent, plausible sounding answer that leans on
knowledge outside what it actually retrieved.
Table2isthemoreinformativeresult:splittingbyquestion
class reproduces, directionally, the pattern Cochran (2026)
report on their research paper corpus RAG is closer to com-
petitiveonsingletopicquestionsandfallsfurtherbehindon
synthesis.Contextanswering(ArmB)isnearlyindifferentto
questionclass(9.96vs.9.93score,98%vs.100%grounded);
vectorRAGisnot(9.33vs.8.14score,an0.62pointgapon
topicpage[95%CIonthegap:0.16,1.18]wideningto1.79
oncrosspage[95%CI:0.57,3.14];87%vs.64%grounded).
The score gap is statistically distinguishable from zero in
both classes, but the two classes’ groundedness gaps are not
equally well established: the cross page grounded rate gap
(36points,95%CI[14,64])clearlyexcludeszeroatn= 14,
while the topic page grounded rate gap (11 points, 95% CI
[0, 22]) just touches zero atn= 45, the one comparison in
this study that does not reach significance at the 95% level.
4.1 Where Vector RAG Lost Ground
Retrieval itself was largely successful: 43 of 45 topic page
questions(96%)retrievedachunkfromthequestion’sactual
source page in the top 5. The two misses both pulled in
shortnavigationalpages(index,gettingstarted,status,course
overview) instead of the target concept page, on questions
aboutMLBiasinterventionsandaboutwhichkmeansslide
deckiscanonicalbothscoredlow(3and4)andweremarked
ungrounded, as expected when the answerer never saw the
relevant material.
More striking is that retrieval failure explains only 2 of
the11ungroundedArmAanswers.Theother9retrievedthe
correct page (or, for cross page questions, relevant chunks
frommorethanonepage)andwerestillmarkedungrounded.
The answerer added unsupported detail even with the right
excerpts in context. This suggests chunked context invitesfabrication in a way full document context does not, inde-
pendent of whether retrieval itself succeeded.
4.2 Relation to the Hypotheses
H1(RAG competitive with the wiki on single-fact lookup)
is not well supported once the bootstrap CIs are consid-
ered. Arm A’s topic-page score (9.33) is numerically close
toArmB’s(9.96),butthe95%CIonthatgapexcludeszero
(Section 4.1), so the two arms are statistically distinguish-
ableevenonsingle-factlookup.TheonepartofH1thedata
doesnotruleoutisgroundednessspecifically:thetopic-page
grounded-rate gap (87% vs. 98%) is the only comparison in
this study whose CI touches zero, so RAG’s shortfall there
is directionally consistent with a gap but not statistically es-
tablished atn= 45.
H2(wiki outperforms RAG on multi-hop and synthe-
sis questions) is supported, and more decisively than H1
is disconfirmed: the wiki’s score advantage on cross-page
questions is roughly three times its advantage on topic-page
questions,andunlikethetopic-pagegroundednessgap,every
cross-page comparison’s CI excludes zero, though this rests
on justn= 14cross-page questions, the smallest cell in the
study.
H3(a set of trade-offs rather than a single architecture
dominating) is not supported at this corpus scale: the wiki
leads on both question classes tested, and the score gap is
statistically distinguishable from zero in both. However, we
treat this as provisional rather than a general claim, since
thereislikelyainformationadvantageforthewikiarmgiven
the nature of how it is constructed. RAG’s could have a
relative advantage, one a larger corpus, where full-context
conditioningcouldbebalanced.However,thiscouldsupport
the nature of how classes are built and taught, typically on
a week by week scale. Suggesting, potentially, that the wiki
structure is more in line with normative teaching practices
and can better support student learning as a result.
5 Discussion
5.1 RQ1/RQ2: Structure at Ingest vs. Retrieval at
Query
At the scale of this evaluation, the LLM compiled wiki an-
swers questions better than vector RAG on both question
classes, and the advantage is not uniform: it is roughly three
timeslargeroncrosspagesynthesisquestionsthanonsingle
topic lookup, in both correctness score and groundedness
(Section 4.1). Bootstrap CIs over the 59 questions (Sec-
tion 4.1) show this pattern is not just noise from a small
question set every gap except one (the topic page ground-
edness gap) is statistically distinguishable from zero though
those CIs speak only to question sampling error, not to an-
swererorjudgestochasticity,whichasingleruncannotsep-
arate out. RQ1 therefore has a directional answer an LLM
compiled wiki improves response quality over vector RAG
under a matched generator, prompts, and question set and
RQ2’s answer is that the improvement concentrates in ex-
actlythequestiontypetheliteraturewouldpredict:questions
thatrequirelinkingmaterialacrossthecorpusratherthanre-
trievingasinglerelevantpassage(Cochran2026;Pengetal.
5

2024; AboulEla et al. 2025). We still stressdirectionalover
definitive, this is a single run on a 21 vector index.
5.2 Structure Also Reduces Fabrication, Not Just
Retrieval Misses
ThemostsurprisingresultisinSection4.1:only2ofvector
RAG’s 11 ungrounded answers were retrieval misses. The
other 9 retrieved the right material and were still marked
ungrounded.Theanswereraddeddetailthattheretrievedex-
cerpts did not support. The graph and structure augmented
retrieval literature typically frames structure as a fix forre-
trieval(findingtherightchunk)ratherthangeneration(using
only what was found) (Peng et al. 2024; Zhu et al. 2025b).
Our results suggest structure at ingest does both: the wiki’s
cross referenced concept pages appear to constrain the gen-
erator’s elaboration as well as its access to relevant content.
ThisreframestheRAGvswikiquestionevenahypothetical
vector RAG system with perfect retrieval might not close
the groundedness gap, because part of the gap looks like an
artifactofchunked,decontextualizedexcerptsinvitingagen-
erator to fill in surrounding context on its own, not purely a
retrieval quality problem.
5.3 RQ3: Impacts on Learning
These results bear on a question broader than which archi-
tecturescoreshigher:whathappenstostudentlearningwhen
the tutor a course deploys differs in how well a student can
checkitswork.Thegroundnessgapfoundinthestudyspeaks
tothisconcern.ArmA’srawscoresarecloseenoughtoArm
B’s that a student skimming for correctness might not no-
tice a difference. The gap that matters pedagogically is in
whether a claim can be traced back to the lecture, slide, or
reading that introduced it. An answer that is fluent, plau-
sible, and ungrounded is the more dangerous failure mode
in a classroom than one that is simply wrong, because a
wrong answer a student can evaluate against the source ma-
terial trains. Which could even be a positive outcome but a
confidently ungrounded answer short-circuits that check. If
students come to treat an AI tutor’s fluency as a proxy for
correctness.
This reframes what "grounded" needs to mean for class-
room deployment specifically. A representation that pre-
serves where the content originated, lets a student follow
any claim back to its source. This helps keep the student
positioned, at least in part, as the one doing the checking,
ratherthanoutsourcingthatjudgmenttothetool.Itispossi-
ble this argues for treating citation-traceability as a deploy-
mentrequirementforclassroomAItoolsinmachinelearning
courses, not an optional feature to compare among several
credibleoptions.Inanon-trivialwaythismethodcouldhelp
develop trust between the emerging dynamic of Teacher-
Student-LLM that has become a practical default in most if
not all machine learning classrooms.
A second implication follows from the ceiling effects re-
portedinSection4:aninstrumentsaturatingat9+outof10
cannot distinguish a tutor that helps a student build durable
understandingfromonethatproducesanswersarubrichap-
pens to reward. Instructors adopting these tools should notread a high LLM-judge score as evidence of pedagogical
qualitywithoutseparatelyverifyingthatthereasoningpatha
studentfollows,notjusttheterminalanswer,survivescontact
with the tutor. Future classroom deployments would benefit
from evaluation designs that probe whether the representa-
tion’s structure is legible to students themselves, not only to
the grading model. While this is just an initial study the po-
tentialforpositiveoutcomesforin-classroomsettingsseems
quite high.
A final benefit of explicit cross-linking is what it does for
concepts that recur across the semester. Consider squared-
errorloss:astudentmeetsitfirstinweektwoorthreeasthe
objective linear regression minimizes, then meets it again
in week seven or eight as the variance-reduction criterion a
regression tree uses to choose a split. It’s the same quantity,
doing the same job, in a representation that looks quite dif-
ferent from the first. Retrieval over raw course materials has
noreasontoconnecttheseoccurrences;eachchunkisscored
onitsownlexicalorsemanticsimilaritytothequestion,and
the tree lecture rarely mentions "squared-error loss" by that
name. A compiled wiki page, by contrast, can carry an ex-
plicit link back to where the concept was first introduced,
surfacing the connection a student would otherwise have to
notice unaided. This matters because revisiting a concept at
a delay, in an unfamiliar context, is close to the textbook
definition of the conditions under which retrieval converts a
fragile, short-term encoding into a durable one. Distributed
rather than massed exposure produces more durable learn-
ing(Cepedaetal.2006),andtheactofretrievingaconcept,
rather than merely re-reading it, is itself what drives long-
term retention (Karpicke and Roediger 2008). Interleaving
problemsthatsharedeepstructureacrossdifferenttopicshas
beenshowntoproduceexactlythiskindoftransferinmath-
ematics instruction specifically (Rohrer and Taylor 2007),
whichisthesamepatternaloss-functioncitationtrailwould
expose across a regression-to-trees transition. As a practical
matter, this exact scenario actually occurred during testing.
5.4 Limitations and Threats to Validity
Severallimitationsarepresent.(1)generalizationisuntested
beyondasinglecoursecorpus(DS3001)anditstopicmix;(2)
the chunking/embedding configuration for Arm A is inher-
ited unchanged from our prior study (Section 2.5) to isolate
the representation variable, but this means a different RAG
configuration is a possible confound the reported gap could
narrow if Arm A were re tuned for this corpus rather than
reused from the prior one. (3) Scoring uses a single LLM
judge rating and a binary groundedness flag rather than the
full RAGAS context recall/faithfulness/factual correctness
triad. (4) Each arm was answered and graded only once per
question rather than repeated; we bootstrap the 59 questions
withreplacement(B=20,000)toreport95%CIsonquestion
sampling error, but this does not capture answerer or judge
stochasticity(thesamequestionreanswered,orthesamean-
swerregraded,mayscoredifferently),whichneedsrepeated
runs rather than resampling of a single run. (4) Lastly, cost
(ingest compute, query tokens, latency) was not measured
thoughgiventhesizeofthecorpus,thisislikelynotaissue.
We report these results as evidence the pipeline works end
6

to end and as a directional signal worth further exploration.
6 Conclusions
We compared two knowledge representations over an iden-
tical course corpus under a matched generator, prompt, and
questionset.Thecompiledwikiarmscoredhigherthanvec-
tor RAG overall (9.95 vs. 9.05 out of 10; 95% CI on the
difference [0.42, 1.44]) and was more often grounded in the
material the answerer actually saw (98% vs. 81%; CI on the
difference [7, 27] points). Both intervals exclude zero.
H1, that vector RAG would be at least competitive on
single-fact lookup, is not fully supported. The topic-page
score gap is numerically small (9.33 vs. 9.96) but its confi-
denceintervalexcludeszero.Thesinglecomparisonconsis-
tentwithH1istopic-pagegroundedness,wherethe11-point
gap has a CI of [0, 22] and is the only endpoint in the study
that does not reach significance at the 95% level. H2, that
the wiki would outperform on cross-page questions, is fully
supported: the score gap widens from 0.63 on topic-page
questionsto1.79oncross-pagequestions,andthegrounded-
rategapfrom11to36points,witheverycross-pageinterval
excludingzero.H3,thattheoutcomewouldbeasetoftrade-
offs rather than a single dominant architecture, is not sup-
ported at this corpus scale, the wiki leads on both question
classes tested.
WedonotreadthisasageneralresultasArmBwasgiven
a fixed set of indexed wiki sections, identical across all 59
questions and covering the material each question targets,
it therefore does not have a context-selection step that can
fail. Arm A retrieves five chunks from a 21-vector index
and misses on 2 of 45 topic-page questions. Read that way,
the finding is that vector RAG leaves roughly 0.9 points of
answer quality and 17 points of groundedness relative to
perfect context selection of the wiki, and that the shortfall
roughly triples on questions requiring material from more
than one page.
The most informative result is mechanistic rather than
comparative. Only 2 of vector RAG’s 11 ungrounded an-
swers were retrieval failures. The remaining 9 retrieved the
relevant material and were still marked ungrounded, the an-
swerer added details the retrieved excerpts did not support.
Structure-augmented retrieval is typically motivated as a fix
forfindingtherightcontent,howeverthissuggestsasubstan-
tial share of the groundedness gap originates in generation
ratherthanretrieval,andthatavectorRAGsystemwithper-
fect retrieval would not close it.
Whatthestudydoesestablishisthatthepipelinerunsend
to end, that the endpoints discriminate between conditions,
and that the gap between real retrieval and perfect context
selection over authentic course material is large enough to
be worth closing. This could have real implications for how
teachers design and use generative AI systems. Though fur-
thertestingisneeded,coursesthatfocusondeliveringcutting
edge dynamic material and build throughout the semester
would appear to benefit from considering the linked wiki
approachasitrelatestogenerativeAIassistantsintheclass-
room.6.1 Future Work
Further testing is needed to validate the approach in a class-
room setting with actual students. Moreover, cost implica-
tions and comparisons with a larger corpus would add to
robustness of the findings. A hybrid arm could also be de-
ployed that might take advantage of the strengths of both
approach, were a RAG database is prebuilt but wiki context
gets add throughout the semester. There’s likely a argument
for increasing the difficulty of the questions to see if the re-
sults continue to support one approach over the other or if a
ceiling of grounded responses can be reached. Also a graph
RAG arm would also help to creation direction on which
type of approach has the most potential to increase learning
in Machine Learning classrooms. Finally, a comparison to
a zero-shot model would also be informative but likely di-
rectsadifferentstudythatfocusesmoreonqualityofmodels
versus the method of retrieving course content.
7

References
AboulEla, S.; Zabihitari, P.; Ibrahim, N.; Afshar, M.; and
Kashef, R. F. 2025. Exploring RAG Solutions to Reduce
HallucinationsinLLMs.In2025IEEEInternationalSystems
Conference (SysCon), 1–8.
Brown, A.; Roman, M.; and Devereux, B. 2025. A Sys-
tematic Literature Review of Retrieval-Augmented Genera-
tion: Techniques, Metrics, and Challenges.arXiv preprint
arXiv:2508.06401.
Cepeda,N.J.;Pashler,H.;Vul,E.;Wixted,J.T.;andRohrer,
D. 2006. Distributed Practice in Verbal Recall Tasks: A
Review and Quantitative Synthesis.Psychological Bulletin,
132(3): 354–380.
Cochran,T.O.2026. VectorRAGvsLLM-CompiledWiki:
A Preregistered Comparison on a Small Multi-Domain Re-
search Corpus.arXiv preprint arXiv:2605.18490.
ExplodingGradients.2025.Ragas:Evaluationframeworkfor
LLM-generated responses. https://docs.ragas.io/en/latest/.
Accessed: 2025-07-08.
Gao, F.; Xu, S.-Y.; Hao, W.; and Lu, T. 2025. KA-
RAG:IntegratingKnowledgeGraphsandAgenticRetrieval-
Augmented Generation for an Intelligent Educational
Question-Answering Model.Applied Sciences, 15(23).
Hou, Y.; Pascale, A.; Carnerero-Cano, J.; Tchrakian, T.;
Marinescu,R.;Daly,E.;Padhi,I.;andSattigeri,P.2024. Wi-
kiContradict: A Benchmark for Evaluating LLMs on Real-
WorldKnowledgeConflictsfromWikipedia.arXivpreprint
arXiv:2406.13805.
Jain, A.; Cui, L.; and Chen, S. 2025. Aligning LLMs for
the Classroom with Knowledge-Based Retrieval: A Com-
parative RAG Study. In2025 IEEE International Confer-
enceonTeaching,Assessment,andLearningforEngineering
(TALE), 1–8.
Karpicke, J. D.; and Roediger, H. L. 2008. The Critical
Importance of Retrieval for Learning.Science, 319(5865):
966–968.
Ke, Y.; Jin, L.; Elangovan, K.; Abdullah, H.; Liu, N.; Sia,
A. T. H.; Soh, C. R.; Tung, J. Y. M.; Ong, J.; Kuo, C.-J.;
Wu, S.-C.; Kovacheva, V.; and Ting, D. 2025. Retrieval
Augmented Generation for 10 Large Language Models and
itsGeneralizabilityinAssessingMedicalFitness.npjDigital
Medicine, 8.
Kestin, G.; Miller, K.; Klales, A.; Milbourne, T.; and Ponti,
G.2025.AITutoringOutperformsIn-ClassActiveLearning:
An RCT Introducing a Novel Research-Based Design in an
Authentic Educational Setting.Scientific Reports, 15.
Ko, E. G.; Lee, H. H.; Singh, A.; Boddy, L.; Ford, K.; and
Huff, E. W. 2026. Toward Scalable and Responsible Inte-
grationofCourse-SpecificAITutors:InstructorExperiences
with a Campus-Wide Platform. InProceedings of the 2026
CHI Conference on Human Factors in Computing Systems.
Levonian, Z.; Li, C.; Zhu, W.; Gade, A.; Henkel, O.; Postle,
M.-E.; and Xing, W. 2023. Retrieval-Augmented Genera-
tion to Improve Math Question-Answering: Trade-offs Be-
tweenGroundednessandHumanPreference.arXivpreprint
arXiv:2310.03184.Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal, N.; Kulikov, I.; Ghazvininejad, M.; Zettlemoyer, L.;
and Kiela, D. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InAdvances in Neural In-
formation Processing Systems, volume 33, 9459–9474.
Li,J.;Yuan,Y.;andZhang,Z.2024.EnhancingLLMFactual
AccuracywithRAGtoCounterHallucinations:ACaseStudy
on Domain-Specific Queries in Private Knowledge-Bases.
arXiv preprint arXiv:2403.10446.
Li, Y.; Wu, Y.; and Chiu, T. K. F. 2025. How Teacher Pres-
ence Affects Student Engagement with a Generative Arti-
ficial Intelligence Chatbot in Learning Designed with First
PrinciplesofInstruction.JournalofResearchonTechnology
in Education, 58(5).
Li, Z.; Wang, Z.; Wang, W.; Hung, K.; Xie, H.; and Wang,
F.L.2025. Retrieval-AugmentedGenerationforEducational
Application: A Systematic Survey.Computers and Educa-
tion: Artificial Intelligence, 8: 100417.
Linders, J.; and Tomczak, J. M. 2025. Knowledge Graph-
ExtendedRetrievalAugmentedGenerationforQuestionAn-
swering.Applied Intelligence, 55.
Lyu, Y.; Li, Z.; Niu, S.; Xiong, F.; Tang, B.; Wang, W.;
Wu, H.; Liu, H.; Xu, T.; and Chen, E. 2024. CRUD-
RAG: A Comprehensive Chinese Benchmark for Retrieval-
Augmented Generation of Large Language Models.ACM
Transactions on Information Systems, 43: 1–32.
Maity, S.; Deroy, A.; and Sarkar, S. 2024. Leveraging In-
Context Learning and Retrieval-Augmented Generation for
AutomaticQuestionGenerationinEducationalDomains. In
Proceedings of the 16th Annual Meeting of the Forum for
Information Retrieval Evaluation.
Neagu, A.; Wong, J. T. H.; Messer, M.; Nelson, R.; and
Johnson,P.B.2026. RethinkingScaffoldinginLLMTutors:
TheInteractionalMismatchBetweenBenchmarksandReal-
World Deployments.arXiv preprint arXiv:2606.15766.
Neha, F.; Bhati, D.; and Shukla, D. K. 2025. Retrieval-
Augmented Generation (RAG) in Healthcare: A Compre-
hensive Review.AI, 6(9).
Peng,B.;Zhu,Y.;Liu,Y.;Bo,X.;Shi,H.;Hong,C.;Zhang,
Y.;andTang,S.2024. GraphRetrieval-AugmentedGenera-
tion: A Survey.ACM Transactions on Information Systems,
44: 1–52.
Rohrer, D.; and Taylor, K. 2007. The Shuffling of Mathe-
matics Problems Improves Learning.Instructional Science,
35(6): 481–498.
Sunil, K.; and Thakkar, A. 2025. SocraticAI: Transforming
LLMs into Guided CS Tutors Through Scaffolded Interac-
tion.arXiv preprint arXiv:2512.03501.
Thway, M.; Recatala-Gomez, J.; Lim, F. S.; Hippalgaonkar,
K.; and Ng, L. W. T. 2025. Harnessing GenAI for Higher
Education: A Study of a Retrieval Augmented Generation
Chatbot’sImpactonLearning.JournalofChemicalEduca-
tion, 102(9).
Wall, V.; Bedford, A.; and Redmond, P. 2025. Generative
ArtificialIntelligenceinEducation:InitialPrinciplesDevel-
oped from Practitioner Reflexive Research.The Journal of
Educational Research, 118(6).
8

Wang, X.; Zainuddin, Z.; and Leng, C. H. 2025. Generative
ArtificialIntelligenceinPedagogicalPractices:ASystematic
Review of Empirical Studies (2022–2024).Cogent Educa-
tion, 12(1).
Wright, B.; et al. 2026. Using Educational Data to Ex-
plore Multimodal (Audio, Visual, and Textual) LLM Re-
trieval Techniques.Working paper. Prior study; full venue
details to be added.
Wu, R.; and Yu, Z. 2024. Do AI Chatbots Improve Stu-
dents’LearningOutcomes?EvidencefromaMeta-Analysis.
British Journal of Educational Technology, 55(1).
Wu, X.; Zhu, P.; Zhang, J.; Yin, M.; and Wang, Y. 2026.
ChatGPT’sImpactonStudentLearningOutcomes:AMeta-
Analysisof35ExperimentalStudies.HumanitiesandSocial
Sciences Communications, 13(1).
Xu, S.; Yan, Z.; Dai, C.; and Wu, F. 2025. MEGA-RAG:
A Retrieval-Augmented Generation Framework with Multi-
Evidence Guided Answer Refinement for Mitigating Hallu-
cinations of LLMs in Public Health.Frontiers in Public
Health, 13.
Zhou,Y.;Su,Y.;Sun,Y.;Wang,S.;Wang,T.;He,R.;Zhang,
Y.; Liang, S.; Liu, X.; and Fang, Y. 2025. In-depth Analysis
ofGraph-basedRAGinaUnifiedFramework.arXivpreprint
arXiv:2503.04338.
Zhu, X.; Xie, Y.; Liu, Y.; Li, Y.; and Hu, W. 2025a. Knowl-
edge Graph-Guided Retrieval Augmented Generation. In
Proceedings of the 2025 Conference of the North American
Chapter of the Association for Computational Linguistics,
8912–8924.
Zhu, Z.; Huang, T.; Wang, K.; Ye, J.; Chen, X.; and Luo,
S. 2025b. Graph-Based Approaches and Functionalities in
Retrieval-AugmentedGeneration:AComprehensiveSurvey.
ACM Computing Surveys.
9