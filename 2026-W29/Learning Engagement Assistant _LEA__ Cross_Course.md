# Learning Engagement Assistant (LEA): Cross-Course Scalability and Classroom Evaluation of an Agentic AI Tutoring System

**Authors**: Teri Rumble, Javad Zarrin, P. George Lovell, Ruth Falconer

**Published**: 2026-07-15 01:25:22

**PDF URL**: [https://arxiv.org/pdf/2607.13370v1](https://arxiv.org/pdf/2607.13370v1)

## Abstract
This paper is an extension of a paper presented at the ICAART 2026 conference, which introduced LEA (Learning Engagement Assistant), an adaptive AI tutoring agent combining course-specific Retrieval-Augmented Generation (RAG) with structured Knowledge Component (KC) models across integrated Chat, Tutor, and Quiz modes. That prior work validated LEA on a single STEM course (CMP511) exclusively through simulation, using synthetic learner agents. This paper extends that work by reporting the first classroom deployment of LEA with real students (n = 8, CMP511) and the first empirical test of its cross-course scalability, deploying the system across three courses spanning two academic levels and two disciplinary domains. The study reveals a divergence from simulation predictions across modes, showing that synthetic evaluation alone cannot anticipate all aspects of real deployment. A RAGAS-based cross-course scalability evaluation (660 questions) finds Answer Relevancy and Context Precision broadly stable across courses (0.88-0.94 and 0.88-0.90 respectively), while Faithfulness declines with curriculum distance from the system's original course (0.69 to 0.50), a preliminary finding that may reflect generation logic tuned to the system's original subject rather than a scalability limitation. These findings suggest that while the orchestration layer requires no modification, full course-agnosticism of all downstream components requires further investigation.

## Full Text


<!-- PDF content starts -->

Learning Engagement Assistant (LEA):
Cross-Course Scalability and Classroom
Evaluation of an Agentic AI Tutoring System
Teri Rumble1, Javad Zarrin1*, P. George Lovell1, Ruth Falconer1
1Faculty of Design, Informatics and Business, Abertay University,
Dundee, United Kingdom.
*Corresponding author(s). E-mail(s): j.zarrin@abertay.ac.uk;
Contributing authors: teri.rumble@gmail.com; p.g.lovell@abertay.ac.uk;
r.falconer@abertay.ac.uk;
Abstract
This paper is an extension of a paper presented at the ICAART 2026 con-
ference, which introducedLEA(Learning Engagement Assistant), an adaptive
AI tutoring agent combining course-specific Retrieval-Augmented Generation
(RAG) with structured Knowledge Component (KC) models across integrated
Chat, Tutor, and Quiz modes. That prior work validatedLEAon a single STEM
course (CMP511) exclusively through simulation, using synthetic learner agents.
This paper extends that work by reporting the first classroom deployment of
LEAwith real students (n= 8, CMP511) and the first empirical test of its
cross-course scalability, deploying the system across three courses spanning two
academic levels and two disciplinary domains. The study reveals a divergence
from simulation predictions across modes, showing that synthetic evaluation
alone cannot anticipate all aspects of real deployment. A RAGAS-based cross-
course scalability evaluation (660 questions) finds Answer Relevancy and Context
Precision broadly stable across courses (0.88–0.94 and 0.88–0.90 respectively),
while Faithfulness declines with curriculum distance from the system’s original
course (0.69 to 0.50), a preliminary finding that may reflect generation logic
tuned to the system’s original subject rather than a scalability limitation. These
findings suggest that while the orchestration layer requires no modification, full
course-agnosticism of all downstream components requires further investigation.
Keywords:AI Agents, Intelligent Tutoring Systems, Personalised Learning,
Knowledge-based Systems, Retrieval-Augmented Generation, Knowledge Component
Modelling, Human-Subjects Evaluation, Cross-Course Scalability
1
arXiv:2607.13370v1  [cs.CY]  15 Jul 2026

1 INTRODUCTION
This article is an extended version of our conference paper presented at the 18th Inter-
national Conference on Agents and Artificial Intelligence (ICAART 2026) [1]. That
paper introduced theLearning Engagement Assistant(LEA) as a proof of concept val-
idated entirely through simulation on a single STEM course, and explicitly identified
real-world deployment and cross-course generalisation as necessary next steps. The
present article addresses both of these deferred questions and broadens its supporting
review of related work accordingly.
The landscape of AI-assisted education has shifted markedly since 2025. The prior
generation of Intelligent Tutoring Systems (ITS) focused on building systems capa-
ble of personalised learning by adapting to the needs of the individual learner [2].
In 2025, general-purpose large language models (LLMs) acquired learning-oriented
interfaces, and while vendor-published evaluations are beginning to appear [3], inde-
pendently replicated, classroom-grounded evidence of their pedagogical effectiveness
remains scarce. OpenAI’s ChatGPT Study Mode, Google’s Gemini Learn LM, and
Anthropic’s Claude for Education [4–6] have joined Khan Academy’s Khanmigo [7]
and Harvard’s CS50 AI Tutor [8] as widely deployed conversational tutoring tools.
Currently, there is a widening gap between deployment speed and evidence: these sys-
tems are entering classrooms faster than rigorous evidence of their effectiveness can
be established [2, 9].
The general-purpose LLM tools share a common architecture: a Socratic-style
prompting layer that guides learners toward answers rather than directly supplying
them, with no course-specific knowledge grounding beneath it. What none of these
tools share (including the more purpose-built systems such as Khanmigo and CS50) is
a unified architecture that combines a structured Knowledge Component (KC) model,
persistent mastery tracking, course-specific Retrieval-Augmented Generation (RAG)
grounding, and real-time estimation of Cognitive Load (CL), Zone of Proximal Devel-
opment (ZPD), and motivational state within a single system (Table 1). Mannekote
et al. argue that without structured learner modelling (tracking knowledge state,
motivation, and affect across interactions), LLM-based tutors cannot achieve the ped-
agogical depth required for genuine whole-learner support [9]. General-purpose LLM
tool course awareness is limited to whatever the student pastes into the conversation
window, and even where structured context is explicitly provided, LLM-generated
tutoring responses have been shown not to adapt to it reliably [10].
The evaluation landscape compounds this: Maurya et al. propose a unified taxon-
omy for assessing LLM-based tutors precisely because existing frameworks lack the
granularity to evaluate pedagogical behaviours beyond surface-level response qual-
ity [11]. Similarly, no standard methodology exists for evaluating the cross-course
generalisation of adaptive tutoring architectures [2], a gap the three-course deployment
reported here is designed to address. The result is a persistentdeployment–evidence
gap: systems are being placed in classrooms before the field has developed the frame-
works to evaluate whether they work, and before purpose-built adaptive ITS have been
empirically tested against the realities that those classrooms impose [9, 11]. A related
2

but distinct gap concerns scalability: the capacity of an adaptive tutoring architec-
ture to generalise across disciplinary domains without modification is widely claimed
in the ITS literature, but rarely empirically demonstrated [2, 12].
Our prior work,LEA, addressed the ITS architecture gap [1]. It is an agentic AI
tutoring system that delivers adaptive, personalised instruction through three inte-
grated modes (Chat, Tutor, and Quiz) grounded in a course-specific RAG knowledge
base and a hierarchical KC model. Its agent orchestrator continuously estimates the
learner’s CL, ZPD, and motivational state to dynamically adjust scaffolding type,
task difficulty, and feedback timing. Evaluated through simulation-based testing with
14,143 synthetic interactions across six learner profiles,LEAdemonstrated robust
retrieval accuracy, coherent multi-turn tutoring, and adaptive stability [1]. Simulation,
however, does not address the deployment-evidence gap entirely:does LEA work with
real students in a real classroom, anddoes it scale across disciplinary domains beyond
the graduate-level STEM course for which it was originally designed?
The central research questions driving this extended work are: (RQ1) How do
real students experienceLEAacross its three instructional modes, and where does
classroom behaviour diverge from simulation predictions? (RQ2) CanLEA’s course-
agnostic architecture be operationalised across courses of differing academic levels and
disciplinary domains without architectural modification, and if so, what aspects of the
system hold unchanged versus require course-specific adaptation?
To answer these questions, we report three new contributions that substantially
extend the conference paper. First, we describeLEA’s production deployment across
three university courses at Abertay University, Dundee: CMP511 (Machine Learn-
ing and Artificial Intelligence, MSc), CMP202 (Data Structures and Algorithms 2,
undergraduate, UG), and PSY555 (Human Psychology: In what ways are we all the
same?, MSc). This three-course deployment spans two degree levels and two disci-
plinary domains, providing the first empirical account ofLEA’s scalability claims.
The inclusion of PSY555, a non-computing social-science course with discursive, non-
procedural content, constitutes a cross-domain stress test. The central architectural
scalability claim is thatthe orchestration layer generalises; the knowledge layer adapts.
Second, we report the first pilot classroom study ofLEA, conducted with students
enrolled in CMP511 under ethics approval and pseudonymised data collection, using
a nine-section survey instrument to assess usability, mode-specific perceptions, learn-
ing outcomes, trust, and technical reliability. Third, we report a scalability evaluation
(660 questions) across all three courses using the Retrieval Augmented Generation
Assessment (RAGAS) framework to measure retrieval and generation quality.
These contributions advance the agenda called for by Mannekote et al. [9]: moving
beyond prompt-response interactions and synthetic evaluation toward systems that
are interpretable, pedagogically grounded, and validated against the complex ecology
of real learners. The remainder of this paper is structured as follows. Section 2 reviews
related work, expanding the conference paper’s background with literature on human-
subjects evaluation of AI tutors and cross-course ITS scalability. Section 3 summarises
the psychological and pedagogical foundations underlyingLEA’s design, including
cognitive load theory, Zone of Proximal Development, and motivational frameworks,
alongside the RAG and KC modelling principles that ground its content delivery.
3

Section 4 details the system architecture. Section 5 presents the multi-course deploy-
ment methodology. Section 6 summarises the simulation-based evaluation reported in
the conference paper [1] and reflects on its limitations, establishing the predictions
against which the classroom deployment is subsequently assessed. Section 7 reports
on the CMP511 pilot classroom study, including a direct comparison between simula-
tion predictions and observed student experience. Section 8 presents the cross-course
scalability evaluation. Section 9 discusses these findings in relation to RQ1 and RQ2,
and Section 10 concludes with a summary, limitations, and directions for future work.
2 RELATED WORK
This section surveys four bodies of prior work that situateLEAwithin the broader
landscape of adaptive tutoring research: the evolution of AI-driven tutoring sys-
tems, from early cognitive modelling through LLM-driven and agentic architectures
(Section 2.1); the use of RAG in educational systems (Section 2.2); the limited and
largely simulation-based evidence base for human-subjects evaluation of AI tutors
(Section 2.3); and cross-course ITS scalability, an architectural property that is
frequently claimed in the literature but rarely empirically demonstrated (Section 2.4).
2.1 From Rule-Based Systems to Agentic AI Tutors
Intelligent tutoring systems have evolved over three broad generations. Early cognitive
modelling systems likeMATHia(Carnegie Learning) andALEKS(McGraw-Hill)
are examples of learner mastery tracking through Bayesian Knowledge Tracing and
Knowledge Space Theory respectively, whileSquirrel AIextends this through pre-
dictive scaffolding and behavioural analytics [13–15]. All three achieve strong adaptive
assessment from static, pre-authored content libraries, but lack generative capabilities
or affective adaptation. A second generation of LLM-driven generative tutors, includ-
ingOATutor,LiveHint AI, andDuolingo AIoffer improved adaptability through
fine-tuned language models and GPT-4-based RAG, but have limited learner-state
modelling and no cognitive load estimation [16–18].
A third, more recent wave comprises agentic systems with structured course
grounding.Khanmigointegrates GPT-4 with Khan Academy’s curriculum via RAG,
providing strong Socratic scaffolding but no cognitive load monitoring [7];Jill Wat-
sondraws on instructor-supplied materials across seventeen courses but does not
provide scaffolding or estimate mastery [19]; andHarvard CS50’s AI Tutor
integrates course-specific RAG with careful scaffolding but lacks mastery estima-
tion or a motivational framework [8]. Beyond these dedicated systems, major LLM
providers released general-purpose learning interfaces (OpenAI’s ChatGPT Study
Mode, Google’s Gemini Learn LM, and Anthropic’s Claude for Education [4–6]) which
provide Socratic-style prompting with strong content generation but no KC model
capability, mastery estimation, persistent learner-state tracking, or course-specific
knowledge grounding. See Table 1 for comparison of these ITS platforms.
Across all three generations, leading platforms achieve strong mastery tracking
orgenerative content quality, but few integrate RAG-enhanced generation with cog-
nitive and motivational modelling in a unified, domain-agnostic framework [2].LEA
4

Table 1Comparative analysis of AI tutoring systems, extended to include general-purpose LLM
learning features released in 2024–2025 (lower block). Ratings:Strong(S),Moderate(M),Basic
(B),None(–). [4–8, 14–19]
System Domain KC Mastery Adaptive Course Cog. Motiv- Content
Model Estim. Scaffold RAG Load ation Gen.
Purpose-built adaptive ITS
MATHia K-12 Math S S S – M M B
ALEKS STEM S S S – – M –
Squirrel AI K–12 S S S – M M B
OATutor STEM S S S – – M S
LiveHint AI Math B B S – – M S
Jill Watson Higher Ed B – – S – M M
Khanmigo Multi-subject B M S S – S M
Harvard CS50 Comp Science S – S S – B S
Duolingo AI Languages B M S M – S S
General-purpose LLMs with learning overlays
ChatGPT General – – B – – – S
Study Mode
Gemini General – – M – M M S
Learn LM
Claude for General – – B – – – S
Education
LEA (this work)
LEA Higher Ed S S S S M M S
multi-course
Notes.KC Model = structured Knowledge Component hierarchy; Mastery Estim. = probabilistic or
threshold-based mastery tracking; Adaptive Scaffold = dynamic scaffolding adjusted to learner state; Course
RAG = course-specific RAG; Cog. Load = cognitive load monitoring; Motivation = motivation state moni-
toring; Content Gen. = LLM-based content generation.LEA’s ratings reflect the architecture and simulation
evaluation of the conference paper [1] and the deployments reported here. Ratings for the general-purpose
LLM block reflect product documentation [4–6] rather than independent empirical testing; their “adaptive”
scaffolding is prompt-level Socratic questioning with no persistent learner-state representation across turns.
addresses this gap through three design principles: (1) continuous learner-state aware-
ness via agents tracking cognitive load, proximal performance, and motivational state;
(2) adaptive orchestration of scaffolding, task difficulty, and motivational feedback;
and (3) pedagogically grounded content generation, anchored in a course-specific RAG
vector database and aligned with learning objectives through KC modelling.
2.2 RAG in Education
RAG integrates document retrieval with LLM content generation, grounding responses
in course-specific materials rather than relying on general model knowledge [20]. By
incorporating course knowledge into the generative process, RAG architectures reduce
hallucination and help ensure contextual accuracy and curricular alignment [21].
Recent educational deployments confirm the practical value of this approach: RAG-
augmented systems outperform conventional static tutors by grounding feedback in
course-specific content [22]; combined prompt engineering, RAG, and fine-tuning
improve pedagogical adherence over a baseline LLM in a robotics tutor [23]; and
knowledge-graph-enhanced RAG improves domain specificity for structured educa-
tional content [12]. A consistent finding across this literature is that retrieval quality
5

depends on the chunking strategy and embedding model used during corpus prepa-
ration, and that these parameters interact with domain characteristics: cross-course
benchmarking shows that identical chunker-embedding configurations yield systemat-
ically different retrieval quality across disciplinary domains [24], and generic chunking
heuristics have been shown to fragment the semantic structure of source code
specifically, degrading retrieval in ways that do not affect discursive prose [25].
The RAGAS [26] suite of metrics has emerged as a standard evaluation framework
for RAG pipeline quality. Its four metrics (Table 2), measure faithfulness and answer
relevancy reference-free, with context precision and recall computed against a reference
answer.LEAuses RAGAS as the primary metric for cross-course scalability evaluation
(Section 8), enabling comparison of RAG performance across the deployed courses.
Table 2The four RAGAS evaluation metrics [26] used in the cross-course scalability evaluation
(Section 8). All metrics are scored in [0,1], higher is better.
Metric What It
MeasuresHow It Is Calculated Reference
Answer
Required?
Faithfulness Factual consistency
of the generated
answer with the
retrieved contextThe answer is decomposed into individual
claims via LLM prompting; each claim is
verified against the retrieved context. Score
is the proportion of claims supported by the
context.No
Answer
RelevancyHow directly the
generated answer
addresses the
question askedAn LLM generates candidate questions from
the answer; the score is the mean cosine
similarity between the embeddings of these
generated questions and the original question.No
Context Precision Signal-to-noise
ratio of the
retrieved contextEach retrieved context chunk is judged for
relevance to the reference answer; the score
aggregates the proportion of relevant chunks,
weighted by rank position.Yesa
Context Recall Completeness of
the retrieved
contextThe reference answer is decomposed into
sentences; each is checked for attributability
to the retrieved context. Score is the
proportion of reference sentences the context
can support.Yesa
aIn the present evaluation, the system’s own generated answer is used as a stand-in reference in the
absence of human-authored ground truth; these two metrics are therefore interpreted as measures of internal
retrieval-generation consistency (Section 8).
2.3 Human-Subjects Evaluation of AI Tutors
Despite the proliferation of LLM-based tutoring systems, the empirical evidence base
for their classroom effectiveness remains thin. The majority of published evaluations
rely on simulation, expert annotation, or automated metrics rather than real learner
studies [2, 11]. Mannekote et al. [27] make this concern explicit, arguing that existing
evaluation approaches, including learner simulation, are insufficient to surface the
user-experience, domain-transfer, and trust-related dimensions that emerge only in
authentic deployment. Their analysis identifies synthetic-vs-real learner divergence as
a fundamental validation challenge and calls for controlled classroom studies.
6

Where human-subjects studies do exist, they tend to be small-scale, short-term, or
focused on a single interaction feature rather than whole-system pedagogical effective-
ness. Kahl et al. [23] include a small human evaluation component in their robotics
tutor study, assessing helpfulness and trustworthiness alongside automated metrics,
but acknowledge the limited power of the resulting comparisons. Deployed LLM-based
tutors have generated usage evidence at greater scale: Liu et al. [8] report classroom
telemetry from Harvard’s CS50 AI Tutor across a full course cohort, and Taneja et
al. [19] describe Jill Watson’s deployment across seventeen courses. However, neither is
framed as a controlled evaluation of mode-specific pedagogical effectiveness, trust, or
learning outcomes against a defined instrument. Maurya et al. [11] propose a structured
evaluation taxonomy for LLM-based AI tutors that covers pedagogy, engagement, and
safety dimensions, but note that existing systems have not been systematically eval-
uated against this taxonomy in classroom settings. The most consequential evidence
currently available is the Wu et al. meta-analysis [28], which synthesises 35 experimen-
tal studies (4,193 participants) and reports a moderate positive effect of ChatGPT on
learning outcomes (Hedges’g= 0.670). However, nearly all constituent studies use
ChatGPT as a supplementary resource rather than as a structured adaptive tutor,
and the meta-analysis explicitly flags instructional mode as a significant moderator,
suggesting that the effect size is meaningfully different when AI is used as a structured
tutoring system versus as a general-purpose assistant.
This gap between deployment speed and classroom evidence is the main motiva-
tion for the CMP511 pilot study reported in Section 7. It is, to our knowledge, the
first reported human-subjects study of a complete tri-modal adaptive tutoring system
spanning usability, mode-specific pedagogical perception, learning outcomes, and AI
trust in a higher education course.
2.4 Cross-Course ITS Scalability
Cross-course scalability (the capacity of an ITS to be deployed across disciplines with-
out architectural modification) is a widely stated design goal but is rarely empirically
demonstrated [2, 12]. Most ITS evaluations are conducted within a single domain,
and the knowledge representation choices made in that domain (e.g. Bayesian Knowl-
edge Tracing optimised for procedural mathematics, or rule-based ontologies designed
for physics) rarely transfer without substantive re-engineering [2]. Work on cross-
course knowledge tracing and transferable learner-state representations has examined
whether the same underlying model can serve different content domains [29], but
system-level deployment transfer, in which a complete adaptive tutoring architecture
is operationalised across disciplines without modification, is a distinct question that
has not been addressed with matching rigour. The proliferation of LLM-based systems
has altered this landscape: general-purpose language models aredomain-agnosticbut
remain course-blind without structured knowledge grounding.
Recent RAG-based architectures have advanced domain-agnostic pipeline designs,
but an evidentiary gap remains. Dong et al. [12] claim cross-disciplinary scalabil-
ity as an inherent property of modular knowledge-graph-augmented RAG pipelines,
while acknowledging that empirical deployment evidence across disciplines is still lack-
ing. Liu et al. [22] demonstrate successful RAG deployment in computer science and
7

engineering contexts but do not extend evaluation to social science courses. A 2026
systematic review of 86 tutoring-system studies found the same imbalance in litera-
ture as a whole, with only 9% of studies addressing social-science content against 52%
in STEM domains [2]. STEM and social science disciplines differ not only in content
vocabulary and density, but, as the present deployment demonstrates (Section 5), in
the structure of their knowledge components and the granularity of learning objec-
tives, the nature of assessment (procedural versus conceptual), and the chunking
characteristics of source material (code and mathematics versus discursive text).
LEA’s three-course deployment, detailed in Section 5, is positioned to address this
gap directly: rather than asserting cross-course scalability as an architectural property,
it tests the claim empirically across two academic levels and two disciplinary domains,
including a social science course with no code-based learning objectives.
3 BACKGROUND
The adaptive mechanisms described in Section 4 are grounded in four established
frameworks from cognitive science and educational psychology, summarised here.
Effective ITS design integrates cognitive and motivational principles of learning with
course-specific content grounded in course curricula [30, 31]. WithinLEA, psycho-
logical theories inform how the system interprets learner behaviours and delivers
scaffolding, while structured knowledge representations enable precise retrieval and
alignment with instructional goals.
3.1 Psychological Foundations of Learning
These four frameworks — Cognitive Load Theory, Zone of Proximal Development and
constructivist learning theory, engagement theory, and motivation and self-efficacy
theory — define the Integrated Psychological Framework that informsLEA’s learner-
state modelling and scaffolding decisions via the Agent Orchestrator (Section 4.1).
Cognitive Load Theory (CLT)holds that working memory is limited and that
instruction must balance intrinsic, extraneous, and germane load to support schema
acquisition in long-term memory [32, 33]. Adaptive tutors can infer cognitive effort
non-invasively from indicators such as task completion time and accuracy, adjusting
pacing and complexity to avoid overload [34].
Zone of Proximal Development (ZPD) and constructivist theoryhold
that learning is most effective within the band between what a learner can do inde-
pendently and what they can achieve with guidance [35, 36], and that active strategies
such as elaborative interrogation and problem-based learning support deeper cognitive
construction [37]. Adaptive systems operationalise this by tracking knowledge state
and fading scaffolding as competence increases [38].
Engagement Theory, grounded in Self-Determination Theory (SDT) and Sit-
uated Expectancy-Value Theory, holds that autonomy, competence, and relatedness
sustain participation [39], with interface design (mode choice, progress visualisation,
collaborative features) directly supporting these needs [40, 41].
Motivation and Self-Efficacy Theory, drawing on Bandura’s Social Cognitive
Theory and Zimmerman’s model of self-regulation [42, 43], influencesLEA’s use of
8

progressive scaffolding, vicarious modelling, and personalised feedback to build learner
confidence and persistence.
3.2 Retrieval-Augmented Generation (RAG) Library
TheLEARAG pipeline converts instructional resources, such as lecture notes, slides,
and assessments, into a searchable, vectorised knowledge space. The process begins
with document ingestion and validation, followed by semantic chunking that segments
content into coherent units optimised for LLM context windows. These chunks are
embedded using dense neural models to generate high-dimensional semantic vectors.
The resulting embeddings are stored in a scalable vector database to support real-
time similarity-based retrieval [20]. When a learner query is issued, relevant segments
are retrieved and reranked before being supplied to the LLM, grounding generated
explanations in validated course materials. This integration allowsLEAto provide con-
sistent, contextually relevant, and pedagogically aligned responses while maintaining
adaptability across multiple courses [21].
3.3 Knowledge Component (KC) Model
The KC Model defines the conceptual structure of a course by hierarchically mapping
weekly Learning Objectives (LO) to Granular Objectives (GO) [44].LEAdefines a
methodology for educators to specify high-level LOs, which are expanded into aligned
GO Knowledge Components using AI-assisted decomposition and metadata tagging.
The resulting KC schema provides an outline forLEA’s Tutor and Quiz modalities.
This hybrid human-AI pipeline combines educator intent with scalable automation,
providing a reproducible method for authoring course-specific tutoring environments
utilising pre-existing course materials. The RAG Library and KC Model constitute
the data foundation that inform and guideLEA’s adaptive learner-aware instruction.
4 SYSTEM ARCHITECTURE
LEAoperates through a distributed, agentic AI architecture designed to deliver
adaptive, student-specific instruction by integrating psychological, pedagogical, and
computational components into a unified system. The architecture described in
Figure 1 comprises three core subsystems:Agent Orchestrator,Mastery Tracker, and
Tri-Modal Content Generationthat collectively enable real-time adaptation while
remaining contextually grounded in instructional intent.
4.1 Agent Orchestrator
At the core ofLEAis an agentic AI environment where multiple specialised agents
collaborate to maintain real-time awareness of learner state and adapt instruction
accordingly. The Agent Orchestrator performs continuous assessment of CL, ZPD, and
motivational state using non-invasive behavioural and performance indicators [33, 45,
46]. These indices drive scaffolding decisions and dynamic task difficulty adjustments.
9

Fig. 1LEAAgentic Architecture showing the interaction between the User Interface, Agent Envi-
ronment, and Knowledge Representation layers. Reprinted from our previous conference paper. [1]
4.1.1 Cognitive Load and ZPD Assessment
Cognitive load is calculated as a weighted combination of task difficulty, recent accu-
racy, and fatigue indicators (Eq.(1)), enabling detection of overload or underload
conditions. The system interprets CL values on a 1–9 Paas rating scale [33], to deter-
mine scaffolding intensity and fading rate. The coefficients were determined via offline
Monte Carlo simulations (calibration remains open to refinement, Section 10).
CL= 0.5 + 1.0G+ 4.5(1−A) + 0.1Q+ 1.0T type (1)
where:
•G: normalized difficulty of the next GO
•A: mean accuracy across the last four responses
•Q: number of continuous learner interactions
•Ttype: next task type (1 = open-ended, 0 = all other)
The learner’s ZPD is continuously estimated as the mean accuracy across the last
four responses (Eq.(2)), targeting a performance band of 50–80%, a design choice
informed by desirable-difficulty research [45].
ZPD=1
ntX
i=t−n+1xi (2)
where:
•ZPD: zone of proximal development score
•n: window size (4 most recent responses)
•t: current time or interaction index
•xi: correctness of responsei(1 is correct)
10

These measures enableLEAto maintain challenge levels aligned with the learner’s
proximal capability. Figure 2 demonstrates how CL and ZPD are used to guide
scaffolding choices and to calibrate task difficulty.
Fig. 2Cognitive Load×ZPD Scaffolding Decision Matrix guiding adaptive scaffolding logic.
Reprinted from our previous conference paper. [1]
4.1.2 Motivation State Classification
Motivation state is inferred non-invasively from persistence, sentiment, and perfor-
mance trends (Eq.(3)), following work showing ITS can infer learner state from
behavioural data without self-report [46]. We define four states aligned with the
tri-modal interaction structure:cold start, when insufficient history prevents classifi-
cation [47];motivation drop, reflecting declines in ∆P, ∆S, ∆A;plateau, indicating
stable but non-improving state; andmaintained high motivation, indicating sustained
positive trends across all three signals. Each state triggers a corresponding instruc-
tional strategy grounded in Self-Determination Theory [39]: autonomy support during
cold start, competence celebration and affective intervention during motivation drop,
challenge escalation during plateau, and adaptive fading during maintained high moti-
vation. Motivational feedback is integrated contextually within the Tutor and Quiz
dialogues to sustain engagement and self-efficacy.
∆M= ∆P+ ∆S+ ∆A(3)
where:
•∆M: change in overall motivation level
•∆P: change in persistence (session continuation or interaction frequency)
•∆S: change in sentiment (derived from affective cues in learner responses)
•∆A: change in answer accuracy (reflecting changes in learner competence)
11

Positive ∆Mvalues indicate rising motivation and engagement, prompting adap-
tive fading or autonomy reinforcement, whereas negative shifts trigger supportive
scaffolding or affective interventions to restore learner momentum.
4.1.3 Scaffolding and Feedback Loop
The Agent Orchestrator integrates these cognitive and motivational indices to select
scaffolding type (conceptual, procedural, strategic, or meta-cognitive) and support
intensity [48]. Fading logic reduces scaffolding once stable performance or high motiva-
tion is detected. All orchestration events are logged, supporting transparent evaluation
and continuous optimisation of adaptive strategies.
4.2 Mastery Tracking
The mastery tracking system monitors learner progress at three hierarchical lev-
els, Course Week, Learning Objective (LO), and Granular Objective (GO), storing
dynamic mastery indicators alongside each learner profile. The system employs differ-
ing assessment pathways: for quiz interactions, it uses direct scoring with complexity-
adjusted mastery calculations. For tutor interactions,LEAuses an LLM for response
evaluation in an adapted Difficulty-aware Programming Knowledge Tracing (DPKT)
[49], supporting real time, multi-level assessment.
4.3 Tri-Modal Content Generation
LEAsupports three instructional modalities—Chat, Tutor, and Quiz—each grounded
in course-specific RAG content. The Chat mode facilitates open exploration of the
domain knowledge base and Q&A interactions. Tutor mode provides structured,
Socratic dialog guided by adaptive scaffolding and motivational feedback. Quiz mode
offers formative assessment through adaptive question generation and performance
evaluation. The Tutor and Quiz modes feed student state data to the Mastery Tracker
and Orchestrator, ensuring continuous adaptation across learning sessions.
5 MULTI-COURSE SCALABILITY:
DEPLOYMENT METHODOLOGY
5.1 Course-Agnostic Design Principle
LEA’s key architectural benefit is that scalability does not require redesigning the
adaptive engine for each new course. The system achieves this through a strict separa-
tion of concerns betweenorchestration logicandknowledge representation. The Agent
Orchestrator, Mastery Tracker, and scaffolding/motivation engines operate entirely on
abstract KC identifiers, cognitive state variables, and learner interaction records, none
of which encode course-specific content. Course-specific knowledge is instead confined
to two interchangeable artifacts: the RAG library (a ChromaDB vector store of course
materials) and the KC model (a hierarchical JSON representation of weekly Learn-
ing Objectives and Granular Objectives). Integrating a new course therefore consists
of populating these two artifacts rather than modifying the orchestration code base,
12

consistent with the course-agnostic scalability principle articulated in the conference
paper [1] and illustrated in Figure 1.
This design was tested through deployment across three courses spanning two
disciplines and two academic levels (Table 3); each required only course-specific RAG
and KC artifacts, with no changes to the orchestrator, mastery tracker, or scaffolding
engine. The single exception, a course-blind code-question generation routine within
the content-generation layer, is documented in Section 5.4.
5.2 Course Profiles
Table 3 summarises the three deployments discussed in this section.
Table 3Comparative course profile across the threeLEAdeployments.
Dimension CMP511 CMP202 PSY555
Course Machine Learning Data Structures Human Psyc.
Domain Computing Computing Social Science
Level MSc UG MSc
Weeks 11 10 13
Learning Objectives (LOs) 48 30 16
Granular Objectives (GOs) 161 90 79
Code-based GOsa15 (9.3%) 32 (35.6%) 0 (0%)
RAG document count 90 112 64
RAG chunk count 1,266 1,230 691
aCode-based GOs are those tagged with a procedural skill category (KC model).
CMP511: Machine Learning and Artificial Intelligence (MSc).CMP511
was the original deployment target and baseline against which CMP202 and PSY555
are compared. It is a postgraduate, technical STEM course covering supervised, unsu-
pervised, and reinforcement learning, combining conceptual material with Python-
based implementation exercises. CMP511’s KC model and RAG library were developed
iteratively over the course of the original system build and were the basis for the
simulation-based validation reported in Section 6 and in the conference paper [1].
CMP202: Data Structures and Algorithms 2 (UG).CMP202 is an under-
graduate course in the same disciplinary area as CMP511 (computing) but at a lower
academic level and with a distinct technical focus. Building on its prerequisite, it
introduces parallel programming on shared-memory and GPU architectures and the
design techniques underpinning parallel applications, illustrated through case studies
drawn from real-world workloads. This deployment was selected to testlevel scala-
bility: whether the architecture generalises to an undergraduate cohort. CMP202’s
code-based content required the system’s code-question generation logic to target C++
rather than the Python used in the CMP511 implementation, a distinction discussed
further in Section 5.4.
PSY555: Human Psychology (MSc).PSY555 is a postgraduate course in cog-
nitive human psychology, covering topics including neural architecture and perception,
13

memory models, attention, and decision-making. It contains no programming or code-
based content. PSY555 was selected as the cross-domain test case: a social-science
course with discursive learning objectives. PSY555’s KC model contains zero procedu-
ral Granular Objectives, making it a direct test of whether the KC model schema, and
the system’s downstream content-generation logic, can accommodate domains with no
code-based learning objectives.
5.3 Course Onboarding Pipeline
Adding a new course toLEAfollows a fixed three-stage pipeline, consistent across all
three courses deployed in this study.
Stage 1: RAG library creation.Course materials (lecture slides, lecture record-
ings, tutorial notes, and, where applicable, code files) are placed into a per-course
document repository following a standardised naming convention. An automated
pipeline performs file validation and change detection via MD5 signature compar-
ison, format-specific conversion to Markdown (PDFMiner for PDFs, Whisper for
video transcription, MarkItDown for Office documents), GPT-4-based document
analysis and summarisation, semantic chunking (500-token windows with 100-token
overlap, sentence-bounded via NLTK), and embedding generation using OpenAI’s
text-embedding-3-smallmodel. Embeddings are persisted to a dedicated Chro-
maDB collection, preventing cross-contamination between course knowledge bases.
Stage 2: KC model authoring.An instructor completes a structured Excel
template specifying, for each week, a description and a list of Learning Objectives
with references to supporting RAG materials. An automated extraction script decom-
poses each Learning Objective into Granular Objectives, assigns Bloom’s-taxonomy
cognitive level classifications, sets mastery thresholds, and identifies cross-week
dependencies, producing an Excel output for review.
Stage 3: Human-in-the-loop review.The instructor reviews and edits the
generated KC structure; accepting, editing, adding, or deleting Granular Objectives
as needed, before the reviewed spreadsheet is converted to the final, KC Model file.
This pipeline was applied identically across CMP511, CMP202, and PSY555,
with the only variation being the content and structure of the source materials and
instructor-authored learning objectives themselves.
5.4 A Course-Blind Failure Mode: Code-Question Generation
One component, however, was not course-agnostic in practice. Quiz mode’s logic hard-
coded Python question generation as a default (a choice originally made because
Python fluency is a core objective of CMP511) and relied on generic Granular Objec-
tive metadata with no awareness of which course was active. As a result, the system
generated Python code questions for PSY555, a course with no programming con-
tent. For CMP202, the same logic generated Python rather than the course’s actual
language (C++), since the underlying example code throughout the system had only
been written with CMP511 in mind. This was resolved by adding an explicit mapping
from course to programming language, so that PSY555 never generates code questions
14

and CMP202 generates C++ rather than Python questions. This was the only cross-
course failure requiring a code change during onboarding; all other adaptations were
confined to RAG library and KC model artefact generation.
6 SIMULATION-BASED EVALUATION
LEAwas previously assessed using simulation-based testing with synthetic learner
agents, consistent with established ITS evaluation practice [50]. The full simulation
methodology and results are reported in the conference paper [1]1; they are sum-
marised below to establish the predictions against which the classroom deployment is
currently assessed. Six learner profiles grounded in established learner characteristics
(prior knowledge, cognitive load tolerance, motivation, engagement style, and learn-
ing rate [29]) were constructed following the approach of profiling synthetic students
against the course’s Knowledge Components [51]. These profiles interfaced withLEA’s
full architecture across three studies: Chat (n= 1,980), Tutor (n= 5,563 turns), and
Quiz (n= 6,600 items), with each generative response scored against course content
using an LLM-as-a-Judge framework [52].
Chat Mode achieved 100% retrieval success andM= 0.880 generated answer
quality (95% CI [0.879, 0.882]), both exceeding target. Tutor Mode exceeded target
on both Multi-Turn Effectiveness (M= 0.861, 95% CI [0.859, 0.863]) and Adaptive
Feedback Appropriateness (M= 0.807, 95% CI [0.805, 0.809]). A two-way analysis of
variance (ANOVA) on theLearner State Matchmetric found that learner profile, not
instructional week, was the dominant source of variation (F(5,5495) = 26.70,p < .001,
η2
p= 0.018 for profile vs.η2
p= 0.002 for week), indicating high temporal stability
across the simulated term. Quiz Mode’s KC Alignment Score averagedM= 0.751
(95% CI [0.745, 0.757]), below target, with thematic analysis [53] of feedback logs
suggesting prompt phrasing favoured recall over higher-order reasoning. Aggregate
response accuracy (M= 0.398) varied strongly by learner profile (F(5,6585) = 107.68,
p < .001,η2
p= 0.974), confirming sensitivity to learner capability.
Reflection on Simulation Limitations.This design is well suited to questions
of internal consistency (i.e. whether the orchestrator responds differently to different
synthetic learner states), but is less well suited to three classes of question that only
the subsequent classroom deployment (Section 7) and cross-course analysis (Section 8)
could address:
•User experience.Synthetic agents respond to scaffolding and feedback accord-
ing to parameterised rules; they do not experience Tutor Mode’s Socratic loop as
rigid, or its motivational affirmations as performative, in the way that two CMP511
participants independently reported (Section 7). The simulation framework mea-
sures whether scaffolding wasappropriateby an external rubric, not whether itfelt
appropriate to the person receiving it.
•Real cognitive load.The orchestrator’s CL and motivation estimates were vali-
dated against synthetic learners whose internal states were generated by the same
family of formulae the orchestrator uses to infer them, a circularity no purely syn-
thetic evaluation can escape. The classroom pilot’s qualitative reports of scaffolding
1Source code and illustrative datasets are available at https://github.com/tep00018/LEA.
15

rigidity and affirmation fatigue (Section 7) are the first evidence of how these models
perform against learners whose states they did not generate.
•Domain transfer.All simulation-based evaluation reported above was conducted
exclusively against CMP511 content, prior toLEA’s deployment to CMP202 or
PSY555. The Faithfulness gradient identified in Section 8 (0.685 for CMP511 to
0.499 for PSY555) is a finding that simulation could not have reproduced, since the
evaluation was cross-week within CMP511 only and not cross-course.
These are limits on what synthetic-only evaluation on a single domain can establish,
not flaws in the simulation methodology. Section 7 and Section 8 test these boundaries
and report the divergences.
7 EXPLORATORY PILOT CLASSROOM STUDY
7.1 Study Design, Instrument, and Participants
The CMP511 pilot constitutes the first deployment ofLEAin a live classroom con-
text. Following simulation-based validation [1], the study aimed to characterise student
experience, perceived usability, and mode-specific pedagogical utility, and to provide
an empirical basis for comparing simulation predictions against real-world behaviour.
Ethical approval was granted by the Abertay University Research Ethics Committee
(ref. EMS11930, 9 March 2026) covering informed consent, voluntary participation,
right of withdrawal, GDPR (General Data Protection Regulation)-compliant data
handling, and pseudonymisation.
Participants were students enrolled in CMP511: Machine Learning and AI (MSc,
N= 41), 2025–26.LEAwas available to the full cohort as a voluntary study aid via
pseudonymised login. A nine-section survey (7-point Likert scale; Sections A–I cover-
ing background, usability, three mode experiences, learning outcomes, trust, technical
issues, and open design preferences) was distributed towards the end of term, with
several items reversed to detect inattentive responding. The full instrument, including
exact item wording and all reversed items, is provided as Supplementary Information
(Online Resource 1). Eight responses were received (20% response rate); the study is
treated throughout as anexploratory pilot. No inferential statistics are reported on
Likert data; quantitative items are summarised descriptively using frequency distribu-
tions and the proportion rating≥5/7. Open-text responses are analysed using Braun
and Clarke’s [53] reflexive thematic analysis.
All eight participants had at least occasional experience with general-purpose AI
tools, though four identifiedLEAas their first dedicated AI tutoring platform. Session
engagement was low overall: five participants reported one or two sessions across the
term, two reported three to five, and one six to ten —- a shallow engagement depth
that is acknowledged as a constraint throughout the findings.
7.2 Findings
Table 4 consolidates quantitative ratings across all survey sections (n= 8; 7-point
Likert). Full agreement (8/8,≥5/7) was achieved on all usability items (median 6),
confirming a coherent positive usability theme corroborated in open text: participants
16

Table 4Consolidated pilot study ratings across all survey sections (n= 8; 7-point Likert).
Section Item Median≥5/7
B: Usability
Easy to navigate and use 6 8/8 (100%)
Interface clear and well-organised 6 8/8 (100%)
Comfortable alongside study routine 6 8/8 (100%)
Command Centerakept focus on right content 6 7/8 (88%)
Would recommend to CMP511 peers 6 7/8 (88%)
Overall satisfaction 6 8/8 (100%)
C: Chat Mode
Accurate, course-relevant answers 6 8/8 (100%)
Responses easy to understand 6 8/8 (100%)
Helped explore topics I was uncertain about 6 8/8 (100%)
Did not trust information (rev.) 3.5 3/8 (38%)
Useful complement to lecture material 6 8/8 (100%)
D: Tutor Mode
Felt genuinely different from a standard chatbot 6 6/8 (75%)
Socratic style helped me think more deeply 6 7/8 (88%)
Adapted appropriately when I said I was unsure 4 3/8 (38%)
Helped improve understanding of difficult topics 6 6/8 (75%)
Motivational feedback kept me engaged 6 6/8 (75%)
E: Quiz Mode
Questions clearly worded and well-structured 6 8/8 (100%)
Questions well-aligned with CMP511 content 6 8/8 (100%)
Difficulty of questions felt appropriate 6 8/8 (100%)
Feedback helped me understand mistakes 6 8/8 (100%)
Quiz Mode useful for testing knowledge 6.5 8/8 (100%)
Progress tracking helped motivation 5.5 7/8 (88%)
F: Perceived Learning Outcomes
Helped me better understand CMP511 concepts 6 8/8 (100%)
Felt more confident after usingLEA5.5 8/8 (100%)
Helped identify gaps I might have missed 6 8/8 (100%)
Adaptive feedback relevant to my learning 6 7/8 (88%)
Overall engagement with CMP511 was higher 5.5 5/8 (63%)
Would have learned as much withoutLEA(rev.) 4 3/8 (38%)
G: Trust and AI Attitudes
TrustLEAto provide accurate information 6 8/8 (100%)
Comfortable data not personally identifiable 6 8/8 (100%)
Adaptation encouraged honest engagement 6 7/8 (88%)
AI tutoring tools belong in university education 6 6/8 (75%)
Concerned AI replaces human instruction (rev.) 4 3/8 (38%)
UsingLEAmade me more curious about AI in education 6 6/8 (75%)
Notes.Reversed items marked (rev.); for these,≥5/7 indicates agreement with the negative claim.
aThe Command Center isLEA’s landing interface for selecting course, week, and mode (Figure 1).
described the interface as “simple to use” and “clear,” and independently praised the
ability to move between modes within a session.
Chat Modewas the strongest performer: full agreement (8/8) on accuracy, com-
prehensibility, usefulness, and lecture complementarity. Two primary use patterns
emerged qualitatively:lecture condensation(extracting core concepts from a week’s
17

material) andconcept clarificationacross topics including neural networks, logistic
regression, and Random Forests. One participant explicitly comparedLEA’s Chat
responses favourably to ChatGPT and Gemini, attributing the advantage to course-
specific grounding, consistent with the design rationale for RAG-grounded ITS over
general-purpose alternatives [9].
Tutor Modeproduced the most divided ratings. The Socratic questioning
approach was endorsed by 7/8 participants as helpful for deeper thinking, but only 3/8
agreed thatLEAadapted appropriately when they expressed uncertainty (median 4,
Neutral) —– the most consequential divergence from simulation predictions in this
study (see Section 7.4). Two participants independently reportedaffirmation fatigue:
motivational affirmations felt “disconnected from the conversation flow,” with one
requesting their removal. A further qualitative theme concernedscaffolding rigid-
ity: one participant described copy-pastingLEA’s own text back as a response to
progress the dialogue, suggesting the Socratic loop did not gracefully handle genuine
comprehension failures.
Quiz Modereceived uniformly strong structural ratings (8/8 on all five content
and difficulty items; progress-tracking, a motivational item, at 7/8), in contrast with
the simulation’s below-target KC Alignment Score (M= 0.751, below the 0.80 target),
indicating that perceived alignment exceeded the LLM-as-Judge estimate. However, a
question-repetition bug was reported by 2/8 participants; one classified it as “severe”
and reported that it prevented meaningful use. This defect is one that simulation-based
evaluation is structurally unable to identify.
Learning outcomeswere uniformly positive on understanding (8/8), confidence
(8/8), and gap identification (8/8). Engagement uplift was more measured (5/8 agreed;
three selected Neutral). The reversed additionality item was endorsed by three respon-
dents; however, these were the same three respondents who endorsed the reversed trust
and additionality items while simultaneously affirming the corresponding positively-
worded items at equivalent levels —– an acquiescent response pattern of the kind
reversed items are included to detect —– and notably included the sample’s most
engaged participant rather than its least.
Trustwas unanimous: all eight participants trustedLEA’s accuracy and were
comfortable with data anonymisation. The apparent tension with Section C’s reversed
trust item (3/8 endorsing “did not trust the information”) is most plausibly read as
the acquiescent pattern noted above rather than substantive distrust, although with
n= 8 the two readings cannot be definitively separated. Six of eight agreed AI tutoring
tools belong in university education; only three expressed concern about AI replacing
human instruction. The dominant qualitative position was complementarity:LEA“is
good as a tool to use alongside, but not to replace” instructor-led teaching.
Five of eight participants reported no technical problems; one experienced the quiz
repetition bug, one could not accessLEAfrom the university network (attributable to
the deployment architecture), and one reported minor latency issues without impact
on engagement.
18

7.3 Qualitative Thematic Analysis
Following Braun and Clarke [53], five themes emerged from inductive analysis of open-
text responses across all nine sections:
•Course-specific grounding as differentiator: participants consistently pre-
ferredLEA’s course-anchored responses to general-purpose tools, validating RAG-
grounded ITS over unstructured AI [9].
•Structural clarity of the tri-modal design: fluid movement between Chat,
Tutor, and Quiz modes within a session was independently praised.
•Tutor Mode scaffolding perceived as rigid under genuine confusion: cor-
roborates, at the experiential level, the Learner State Match weakness identified for
low-knowledge profiles in simulation [1].
•Motivational affirmations perceived as performative by some: SDT-
grounded feedback [39] may require calibration for postgraduate learners preferring
direct instructional exchanges.
•LEAas complement, not replacement: participants distinguished individual-
level complementarity (welcome) from institutional-level cost substitution (a
concern), relevant to the Discussion’s ethical considerations.
7.4 Simulation-versus-Reality Comparison
The pilot findings above bear directly on how much weight simulation-based vali-
dation should carry going forward. Table 5 presents a mode-by-mode comparison of
simulation predictions against human-study observations. Three divergence categories
emerge: Chat Mode predictions matched student experience directly; Tutor Mode pre-
dictions were overoptimistic in the aggregate while consistent with the simulation’s
own more granular Learner State Match finding; Quiz Mode predictions were exceeded
by student ratings. A fourth pattern, the question-repetition bug, has no simulation
analogue. The analytical implications of these divergences are developed in Section 9.
7.5 Study Limitations
This study has several limitations. The sample (n= 8, 20% response rate) precludes
inferential analysis and limits generalisability; low session engagement (five of eight
participants reporting one or two sessions) means findings reflect initial impressions
rather than sustained use. The late-term deployment window may have depressed
engagement, and retrospective self-reports introduce standard method limitations. The
quiz repetition bug and network access constraint may also have introduced attrition
bias. The study should therefore be read as a first-deployment feasibility assessment:
it establishes thatLEAcan be deployed in a live classroom and generates initial
user-experience evidence, but does not constitute a controlled evaluation of learning
efficacy.
19

Table 5Simulation Predictions vs. Human Study Observations by Mode
Mode Simulation
MetricSimulation
ResultHuman Study
ObservationDivergence
Chat RAG Retrieval
Success Rate100% (exceeded
80% target)8/8 rated answers
as accurate and
course-relevantConsistent
Chat Generated
Answer Quality
(LLM-as-Judge)M= 0.880
(exceeded
target)8/8 rated responses
as easy to
understand;
perceived
favourably vs.
GPT/GeminiConsistent
Tutor Adaptive
Feedback
AppropriatenessM= 0.807 (met
80% target)Only 3/8 agreed
LEAadapted
appropriately;
reports of
scaffolding rigiditySimulation
overestimated
adaptive
responsiveness
Tutor Learner State
Match
(low-knowledge
profiles)Significantly
lower scores for
Novice Variable
and Struggling
Eager profilesParticipants
reported
inflexibility in
Socratic loop;
affirmation fatigue
reportedSimulation
finding
confirmed at
experiential
level
Quiz KC Alignment
ScoreM= 0.751
(below 80%
target)8/8 rated questions
as well-aligned to
CMP511 contentSimulation
underestimated
content
alignment
Quiz Technical
stabilityNot modelled Question repetition
bug; prevented
meaningful use for
2/8 participantsSimulation
could not
anticipate
defect
All
modesUser trust in
systemNot directly
modelled8/8 trusted
accuracy; data
privacy comfort 8/8Simulation had
no analogue
8 CROSS-COURSE SCALABILITY EVALUATION
As established in Section 6, the simulation-based KC Alignment Score (M= 0.751,
95% CI [0.745, 0.757]) is a single-domain estimate. To test whether retrieval-and-
generation quality more broadly holds across domains and academic levels, we
evaluated all three deployed courses using the RAGAS framework [26] against a course-
proportional question set generated directly from each course’s KC model. For each
Granular Objective, two questions were generated by prompting an LLM with the
objective’s skill descriptor and week context, yielding questions germane to the stated
curriculum independently of whether the RAG corpus contained supporting material.
20

Each question was answered usingLEA’s live, production Chat Mode retrieval and
generation pipeline, with no synthetic learner simulation layer.
Context Precision and Context Recall in the present evaluation use the system’s
own generated answer as a stand-in reference in the absence of human-authored ground
truth, and should be read as measures of internal retrieval-generation consistency
rather than alignment with an independent correctness standard (Table 2); Faithful-
ness and Answer Relevancy require no reference and are scored as RAGAS intends.
Section 10.1 discusses this constraint further.
Table 6 and Figure 3 report all four metrics for each course. Answer Relevancy and
Context Precision are comparatively stable across domains (0.875–0.935 and 0.877–
0.901 respectively), with only a modest decline for PSY555 on Answer Relevancy
(−0.06 relative to the computing courses), and less than a third of the Faithfulness
decline discussed below, indicating that the system consistently retrieves topically rel-
evant material and generates on-topic responses regardless of whether the underlying
content is technical STEM material or discursive psychology text. Faithfulness, by con-
trast, declines across the three courses, from 0.685 (CMP511) through 0.609 (CMP202)
to 0.499 (PSY555); as the pairwise tests below establish, this decline is carried by the
significant cross-domain drop at PSY555 rather than by the non-significant cross-level
step to CMP202. Since the Chat Mode generation prompt was originally authored and
tuned exclusively against CMP511 content, this gradient is consistent with the broader
pattern identified in Section 5.4: auxiliary content-generation logic downstream of the
architecturally course-agnostic orchestration layer can encode single-course assump-
tions that degrade content generation, rather than failing outright, as domain distance
increases.
Table 6RAGAS retrieval and generation quality across the three deployed courses. Cell entries
are the mean with a bootstrap 95% confidence interval (10,000 resamples) beneath. The final
column reports the Kruskal–Wallis omnibus test across courses (distribution-free; scores are
non-normal, Shapiro–Wilkp <10−6throughout). Context Precision and Context Recall use the
system’s own generated answer as a stand-in reference in the absence of human-authored ground
truth, and are read as measures of internal retrieval-generation consistency (Section 8).
Metric CMP511 CMP202 PSY555 Kruskal–Wallis
Faithfulness 0.685 0.609 0.499H= 25.01
[0.652, 0.716] [0.554, 0.663] [0.444, 0.555]p <10−5,ε2= 0.035
Answer Relevancy 0.933 0.935 0.875H= 6.15
[0.925, 0.938] [0.929, 0.941] [0.833, 0.912]p= 0.046,ε2= 0.006
Context Precision 0.901 0.881 0.877H= 1.11
[0.874, 0.926] [0.838, 0.918] [0.828, 0.921]p= 0.58 (n.s.)
Context Recall 0.609 0.404 0.556H= 32.00
[0.571, 0.645] [0.350, 0.458] [0.498, 0.614]p <10−6,ε2= 0.046
n(questions scored) 322a180 158
aOne CMP511 record returned a null Faithfulness score and was excluded from that metric only (n= 321).
21

Fig. 3RAGAS retrieval and generation quality metrics across the three deployed courses. Faithful-
ness declines with distance from CMP511’s original domain, while Answer Relevancy and Context
Precision remain stable across all three courses. Context Recall is notably lower for CMP202 than
for either CMP511 or PSY555.
Because RAGAS scores are bounded on [0,1] and strongly non-normal (Shapiro–
Wilkp <10−6for every course–metric combination), cross-course differences were
assessed with distribution-free methods: a Kruskal–Wallis omnibus test per met-
ric (Table 6), pairwise Mann–WhitneyUtests with Holm correction and Cliff’sδ
effect sizes, and bootstrap 95% confidence intervals on each mean (10,000 resam-
ples). The tests sharpen the descriptive pattern. Context Precision does not differ
significantly across the three courses, and Answer Relevancy differs only marginally
and with a negligible effect size, its sole significant pairwise contrast being CMP511
vs. PSY555 (p Holm = 0.026,δ=−0.15). Faithfulness differs significantly: PSY555
is significantly lower than both CMP511 (p Holm <10−6,δ= 0.29) and CMP202
(pHolm = 0.005,δ= 0.19), while the CMP511–CMP202 difference is not individu-
ally significant (p Holm = 0.31); the apparent monotonic decline in mean Faithfulness
is therefore driven principally by the cross-domain (PSY555) drop rather than the
cross-level (CMP202) one. Context Recall also differs significantly but with a distinct
pattern: CMP202 is significantly lower than both CMP511 (p Holm <10−7,δ= 0.31)
and PSY555 (p Holm <10−3,δ=−0.22), whereas CMP511 and PSY555 do not differ.
The recall deficit thus tracks CMP202 specifically rather than disciplinary distance,
consistent with the code-density explanation below.
A specific architectural prediction follows from the differing content structures
of the three courses: retrieval quality may vary systematically with content type,
since CMP511 and CMP202’s source materials are code-heavy and notation-dense,
while PSY555’s are discursive, qualitative text. The RAG pipeline’s chunking strategy
(Section 5.3, Stage 1) was designed and tuned against CMP511 content; its 500-token,
sentence-bounded chunking approach was not separately validated against code blocks
or against the longer, more elaborated sentence structures typical of psychology source
texts.
The RAGAS evaluation provides partial evidence for this prediction. Context
Recall is markedly lower for CMP202 (0.404) than for either CMP511 (0.609) or
PSY555 (0.556). Given that CMP202’s KC model has the highest proportion of pro-
cedural Granular Objectives of any of the three courses (32/90, or 35.6%, against
22

9.3% for CMP511 and 0% for PSY555; Table 3), and that its RAG corpus accord-
ingly contains a higher density of code notebooks and lab materials, this recall deficit
is consistent with the hypothesis that sentence-bounded chunking degrades retrieval
completeness specifically for code-heavy content, plausibly by splitting logically coher-
ent code blocks (functions, cells, or worked examples) across chunk boundaries in
ways that scatter relevant material rather than preserving it — a failure mode inde-
pendently documented for generic chunking of source code [25]. We treat this as a
plausible explanation rather than a confirmed mechanism; isolating chunking strategy
as the causal factor, as opposed to other differences between CMP202’s corpus and
the other two courses, would require a controlled ablation that was beyond the scope
of the present work.
From retrieval metric to curriculum-coverage signal.
A methodological observation emerged from constructing the evaluation set. Because
the questions were generated from each course’s KC model rather than sampled from
the RAG corpus, every question corresponds to a stated learning objective, whether
or not the corpus contains material that answers it. Faithfulness, the proportion of
generated claims grounded in retrieved context, is consequently depressed not only by
generation errors but also bycoverage gaps: an objective whose supporting content
is absent from the corpus yields a correct-but-ungrounded answer that scores low
through no fault of the generation pipeline. In the present evaluation this is a confound
on the Faithfulness gradient, which we therefore interpret jointly with the single-
course generation logic of Section 9.2. It also, however, suggests a constructive use of
the same machinery: the divergence between KC-derived and corpus-derived question
performance could localise where a course’s materials fail to cover its objectives. We
note this as a direction for future work (Section 10).
9 DISCUSSION
This section addresses the two research questions posed in Section 1, drawing on evi-
dence from the classroom pilot (Section 7), the simulation-based evaluation (Section 6),
and the cross-course scalability analysis (Section 8).
9.1 RQ1: How Do Real Students ExperienceLEA, and Where
Does Classroom Behaviour Diverge from Simulation?
The CMP511 pilot provides the first direct evidence on this question, and the
simulation-versus-reality comparison (Table 5) is the clearest single answer avail-
able. Three distinct patterns emerged, each with a different implication for how
simulation-based validation should be weighted against human evidence.
For Chat Mode, simulation was predictively accurate. The 100% retrieval success
rate andM= 0.880 answer-quality score reported in Section 6 translated directly into
student experience: all eight participants rated Chat responses as accurate, course-
relevant, and easy to understand, with several explicitly preferringLEA’s course-
grounded answers to general-purpose tools such as ChatGPT and Gemini. Where
23

simulation predicts a simple, single-turn retrieval-and-generation task performs well,
the human data confirms it.
For Tutor Mode, simulation waspartiallyoveroptimistic. The aggregate Adap-
tive Feedback score (M= 0.807) suggested the scaffolding engine was performing
above target, but this aggregate obscured a weakness the simulation itself had already
flagged at a more granular level: the Learner State Match metric was significantly
lower for the Novice Variable and Struggling Eager profiles (Section 6). Only three of
eight participants agreed thatLEAadapted appropriately when they expressed uncer-
tainty, and several described the Socratic loop as rigid under genuine confusion. The
lesson is not that simulation failed here, but that a single aggregate score can mask
a sub-population effect that only becomes visible once real, lower-knowledge learners
encounter the system.
For Quiz Mode, simulation was, if anything, conservative. The KC Alignment Score
of 0.751 fell below the simulation’s own 0.80 target and was treated as a design weak-
ness in the conference paper. Yet all eight pilot participants rated quiz questions as
well-aligned to course content. Whether this reflects a calibration gap in the LLM-
as-Judge rubric or a genuine difference between objective and perceived alignment
remains open, but the direction of the divergence — human perception exceeding the
simulated benchmark — is the opposite of the Tutor Mode finding, underlining that
simulation under- and over-estimates different aspects of the same system rather than
failing uniformly in one direction. Simulation also had no mechanism to detect the
quiz question-repetition bug that two participants described: an interface defect that
only manifests under human interaction.
These findings support treating learner simulation as a necessary but insufficient
validation step: it is informative aboutresponse qualityunder controlled conditions
but blind todeployment-levelfailure modes that only manifest under authentic use,
where learner behaviour, attention, and tolerance for interface friction vary in ways
synthetic agents do not model.
9.2 RQ2: CanLEA’s Architecture Be Operationalised Across
Courses Without Modification?
The answer is a qualified yes. Across all three deployments, the orchestrator, mastery
tracker, and scaffolding logic required zero code changes. However, auxiliary content-
generation logic downstream of these layers did require modification to adapt for
course-specific parameters (Section 5.4). Onboarding CMP202 and PSY555 required
only the population of two course-specific artefacts (the RAG library and the KC
model); the orchestration codebase was unchanged (Section 5.3).
The RAGAS-based evaluation (Section 8) supports this at a quantitative level for
the components closest to the orchestration layer: Answer Relevancy and Context Pre-
cision, both less directly shaped by the generation prompt than Faithfulness, remained
stable across all three courses regardless of domain or academic level (0.875–0.935 and
0.877–0.901 respectively).
Two findings complicate a simple yes. First, the course-blind code-question gener-
ation logic (Section 5.4) encoded assumptions specific to CMP511 and failed outright
24

when applied to PSY555 and CMP202, requiring an explicit fix before the architec-
ture’s course-agnostic claim held in practice. Second, the RAGAS Faithfulness gradient
(0.685 for CMP511 to 0.499 for PSY555) shows that even components that did not fail
outright can degrade as deployment moves further from the system’s original design
domain. A further complication concerns how domain distance itself should be mea-
sured. CMP202’s Context Recall (0.404) was markedly lower than either CMP511’s
(0.609) or PSY555’s (0.556), despite CMP202 sharing CMP511’s broad disciplinary
classification (computing) and PSY555 being, by that classification, the more distant
domain from CMP511. This is consistent with the chunking strategy underlyingLEA’s
RAG pipeline degrading specifically for code-dense material (Section 8) rather than
for non-STEM content, suggesting that content structure, not disciplinary class, may
be the more useful predictor of where cross-course scalability assumptions break down.
The orchestration architecture itself is course-agnostic by design and this was oper-
ationally confirmed; but implementation-wide course-agnosticism is not automatically
inherited by every component that touches course-specific data, and surfacing that gap
required both deploying a genuinely different course and applying systematic, metric-
based evaluation rather than functional testing alone. This section and Section 5
illustrate three distinct waysLEAcomponents responded to cross-course deployment:
components that arearchitecturally invariant(the orchestration formulae, mastery
hierarchy, and RAG pipeline structure), components that aredomain-adapted by
design(the KC model’s skill-category taxonomy, built to span procedural and non-
procedural objectives), and components wherelatent single-course assumptionswere
exposed only once a genuinely different domain was deployed against them (the
code-question generation logic).
10 CONCLUSION
This paper extendedLEA, a tri-modal agentic AI tutoring system, beyond the
simulation-only, single-course validation of the conference paper, addressing the two
deployment questions it had deferred. We reported the first classroom deployment
ofLEAwith real students (n= 8, CMP511) and the first empirical test of its
cross-course scalability claim, deploying the system across three courses spanning two
academic levels and two disciplinary domains—including PSY555, a non-STEM social-
science course with zero procedural learning objectives—and evaluating retrieval and
generation quality across all three using the RAGAS framework.
Together these evaluations show that synthetic simulation was predictively accu-
rate for simple interactions but neither sufficient to anticipate the complexities of
real human use nor to detect a production defect that surfaced only in deployment
(RQ1), and thatLEA’s core orchestration required no modification to support a non-
STEM, non-procedural course while its auxiliary, content-touching logic did not scale
as cleanly (RQ2), with content structure—specifically code-density—proving a better
predictor of scalability risk than STEM/non-STEM status.
These findings carry a practical implication: course-agnostic orchestration is
achievable and, on this evidence, robust, but any component that touches course-
specific content directly (worked-example generation, prompt templates, chunking
25

parameters) should be audited for hidden assumptions before a new domain is
onboarded, and simulation should be treated as a filter for refining functionality rather
than a substitute for deployment testing.
These findings advance the agenda articulated by Mannekote et al. [9]: moving
beyond prompt-response interaction and synthetic-only evaluation toward systems
validated against the complexities of real learners and real deployments.LEAhas tran-
sitioned, over the course of this work, from a simulation-validated proof-of-concept
to a classroom-deployed, multi-course platform with human evidence of usability
and trust, architectural evidence of cross-course scalability, and quantitative evidence
delineating precisely where that scalability claim holds without qualification and where
it requires refinement, and including a methodological by-product that re-purposes
retrieval metrics as a curriculum-coverage audit as a direction for future work.
10.1 Limitations
Four limitations bound the conclusions that can be drawn from this work. First, the
exploratory CMP511 study (n= 8, 20% response rate) precludes inferential analysis;
low session engagement means findings reflect initial impressions rather than sustained
use, and learning-outcome claims are accordingly conservative (Section 7). Second,
the cross-course RAGAS evaluation used the system’s own generated answers as a
stand-in reference for Context Precision and Context Recall in the absence of human-
authored ground truth; these two metrics should be read as measures of internal
retrieval consistency rather than alignment with an independent correctness standard,
though Faithfulness and Answer Relevancy are unaffected by this constraint. Third,
the hypothesised link between CMP202’s sentence-bounded chunking strategy and its
lower Context Recall was not isolated through controlled ablation, and other corpus-
level differences cannot be ruled out (Section 8). Finally, the three-course deployment,
while spanning two disciplinary domains and two academic levels, remains narrow in
absolute breadth.
10.2 Future Work
This work motivates six directions.
•Investigating the chunking-strategy hypothesis.Re-chunking CMP202’s cor-
pus with a code-aware strategy [25] and re-running the RAGAS evaluation would
constitute a controlled ablation isolating chunking strategy as the causal factor in
the observed Context Recall deficit, testing whether content structure rather than
disciplinary domain drives the degradation identified in Section 8.
•RAG corpus–curriculum coverage auditing.The cross-course evaluation gen-
erated its question set from each course’s KC model rather than from the RAG
corpus, so that every question mapped to a stated learning objective (Section 8).
A by-product of this design is that Faithfulness becomes sensitive tocorpus cov-
erage: objectives for which the uploaded materials contain no supporting content
yield correct-but-ungrounded answers. Turning this confound into an instrument,
a systematic comparison of RAGAS performance on KC-derived questions against
26

performance on questions synthesised directly from the indexed corpus (as in stan-
dard RAGAS test-set generation [26]) would localise, per learning objective, where
a course’s corpus fails to cover its stated curriculum. Such acorpus-adequacy audit
could help instructors identify and fill coverage gaps when authoring or refining a
course, before those gaps affect learners. To our knowledge, RAG evaluation metrics
have not previously been repurposed in this direction.
•Auditing remaining course-specific assumptions.The code-question gen-
eration logic and the Faithfulness gradient were two instances of a broader
pattern; a systematic audit of fallback content, default values, and example gen-
eration throughout the production codebase would identify whether similar latent
assumptions persist elsewhere, before a fourth or fifth course is onboarded.
•Human-authored reference answers.Collecting instructor-validated reference
answers for a sample of questions per course would allow Context Precision and
Context Recall to be recomputed against genuine ground truth.
•Longitudinal and higher-powered classroom evaluation.A larger, longer-
duration pilot, ideally spanning a full term from the start of teaching, would address
the sample size and engagement-depth limitations identified in Section 7.5, and
could be extended to CMP202 and PSY555 cohorts to directly test the mode-usage
and scaffolding-adaptation questions left open in Section 8.
•Pilot-informed refinement of the orchestrator equations.The conference
paper noted that bidirectional retrieval from historical learner interactions and per-
sonalised parameterisation of the cognitive-load model’sβcoefficients (currently
fixed via offline Monte Carlo simulation, (Eq.(1)) remain open issues. The pilot
(Section 7.2) highlights two specific targets: the Cognitive Load formulation (Eq.(1))
may need to weight recent struggle more heavily to address the scaffolding rigidity,
and the Motivation State formulation (Eq.(3)) may require personalised affirmative
feedback to mitigate affirmation-fatigue. Both require iterative refinement rather
than fixed population-level formulas.
References
[1] Rumble, T., Zarrin, J., Lovell, P.G., Falconer, R.: Learning Engagement Assis-
tant (LEA): a multi-agent AI framework for adaptive and personalized learning
with simulated student agents. In: 18th International Conference on Agents and
Artificial Intelligence, pp. 512–523 (2026). Science and Technology Publications
[2] Latif, E., Liu, V., Zhai, X.: A systematic review of intelligent and robot tutor-
ing systems: evolution, pedagogical design, and AI-driven classification. Smart
Learning Environments13(1), 1 (2026)
[3] Google DeepMind: Teaching with Gemini: Measuring the Impact of Guided
Learning on Student Mathematics Progress in Sierra Leone. https://storage.
googleapis.com/deepmind-media/LearnLM/learnLM sierraleone may26.pdf.
May 2026 technical report; Accessed June 2026 (2026)
[4] OpenAI: ChatGPT Study Mode: Socratic tutoring for all users. https://openai.
27

com/blog/chatgpt-study-mode. Launched July 2025; Accessed Apr 2026 (2025)
[5] Google DeepMind: LearnLM in Gemini. https://cloud.google.com/solutions/
learnlm/. Launched August 2025; Accessed Apr 2026 (2025)
[6] Anthropic: Claude for Education. https://www.anthropic.com/news/
introducing-claude-for-education. Launched April 2025; Accessed Apr 2026
(2025)
[7] Shetye, S.: An evaluation of Khanmigo, a generative AI tool, as a computer-
assisted language learning app. Studies in Applied Linguistics and TESOL24(1),
38–53 (2024)
[8] Liu, R., Zhao, J., Xu, B., Perez, C., Zhukovets, Y., Malan, D.J.: Improving AI
in CS50: Leveraging Human Feedback for Better Learning. In: Proceedings of
the 56th ACM Technical Symposium on Computer Science Education V. 1, pp.
715–721 (2025). ACM
[9] Mannekote, A., Davies, A., Pinto, J.D., Zhang, S., Olds, D., Schroeder, N.L.,
Lehman, B., Zapata-Rivera, D., Zhai, C.: Large language models for whole-
learner support: opportunities and challenges. Frontiers in Artificial Intelligence
7, 1460364 (2024)
[10] Borchers, C., Shou, T.: Can large language models match tutoring system
adaptivity? a benchmarking study. In: International Conference on Artificial
Intelligence in Education, Berlin, Heidelberg, pp. 407–420 (2025). Springer
[11] Maurya, K.K., Srivatsa, K.A., Petukhova, K., Kochmar, E.: Unifying AI tutor
evaluation: An evaluation taxonomy for pedagogical ability assessment of LLM-
powered AI tutors. In: Proceedings of the 2025 Conference of the Nations of the
Americas Chapter of the Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), pp. 1234–1251 (2025). Association
for Computational Linguistics
[12] Dong, C., Yuan, Y., Chen, K., Cheng, S., Wen, C.: How to Build an Adaptive AI
Tutor for Any Course Using Knowledge Graph-Enhanced Retrieval-Augmented
Generation (KG-RAG). In: 2025 14th International Conference on Educational
and Information Technology (ICEIT), pp. 152–157 (2025). IEEE
[13] Ritter, S., Anderson, J.R., Koedinger, K.R., Corbett, A.: Cognitive tutor: Applied
research in mathematics education. Psychonomic bulletin & review14(2), 249–
255 (2007)
[14] Cosyn, E., Uzun, H., Doble, C., Matayoshi, J.: A practical perspective on knowl-
edge space theory: ALEKS and its data. Journal of Mathematical Psychology
101, 102512 (2021)
28

[15] Li, H., Fan, X., Liang, J.: The Squirrel AI adaptive learning system accompany-
ing millions of children in their growth. In: Digital Transformation of Regional
Education in China, pp. 51–53. Springer, Heidelberg, Germany (2025)
[16] Pardos, Z.A., Tang, M., Anastasopoulos, I., Sheel, S.K., Zhang, E.: OATutor: An
open-source adaptive tutoring system and curated content library for learning
sciences research. In: Proceedings of the 2023 CHI Conference on Human Factors
in Computing Systems, pp. 1–17 (2023). ACM
[17] Carnegie Learning, Inc.: LiveHint AI - The First Generative AI Math Tutor
Trained to Think Like a Student. https://livehint.carnegielearning.com. Press
release via Business Wire, November 30, 2023 (2023)
[18] Duolingo, Inc.: “Duolingo Max” Shows the Future of AI Education. Press release,
14 March 2023. Accessed July 2025. Available: https://investors.duolingo.com/
news-releases/news-release-details/duolingo-max-shows-future-ai-education
(2023)
[19] Taneja, K., Maiti, P., Kakar, S., Guruprasad, P., Rao, S., Goel, A.K.: Jill Watson:
A virtual teaching assistant powered by ChatGPT. In: International Conference
on Artificial Intelligence in Education, pp. 324–337 (2024). Springer
[20] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler, H.,
Lewis, M., Yih, W.-t., Rockt¨ aschel, T., Riedel, S.: Retrieval-augmented generation
for knowledge-intensive NLP tasks. Advances in neural information processing
systems33, 9459–9474 (2020)
[21] Li, Z., Wang, Z., Wang, W., Hung, K., Xie, H., Wang, F.L.: Retrieval-augmented
generation for educational application: A systematic survey. Computers and
Education: Artificial Intelligence8, 100417 (2025)
[22] Liu, Z., Agrawal, P., Singhal, S., Madaan, V., Kumar, M., Verma, P.K.: LPITutor:
an LLM based personalized intelligent tutoring system using RAG and prompt
engineering. PeerJ Computer Science11, 2991 (2025)
[23] Kahl, S., L¨ offler, F., Maciol, M., Ridder, F., Schmitz, M., Spanagel, J., Wienkamp,
J., Burgahn, C., Schilling, M.: Evaluating the Impact of Advanced LLM Tech-
niques on AI Lecture Tutors for a Robotics Course. In: International Workshop
on AI in Education and Educational Research, pp. 149–160 (2024). Springer
[24] St¨ abler, M., Turnbull, S., M¨ uller, T., Langdon, C., Marx-Gom´ ez, J., K¨ oster, F.:
The impact of chunking strategies on domain-specific information retrieval in
RAG systems. In: 2025 IEEE International Conference on Omni-layer Intelligent
Systems (COINS), pp. 1–6 (2025). IEEE
[25] Zhang, Y., Zhao, X., Wang, Z.Z., Yang, C., Wei, J., Wu, T.: cAST: Enhancing
Code Retrieval-Augmented Generation with Structural Chunking via Abstract
29

Syntax Tree. In: Findings of the Association for Computational Linguistics:
EMNLP 2025, pp. 8106–8116. Association for Computational Linguistics, Suzhou,
China (2025)
[26] Es, S., James, J., Anke, L.E., Schockaert, S.: RAGAS: Automated evaluation
of retrieval augmented generation. In: Proceedings of the 18th Conference of
the European Chapter of the Association for Computational Linguistics: System
Demonstrations, pp. 150–158 (2024). Association for Computational Linguistics
[27] Mannekote, A., Davies, A., Kang, J., Boyer, K.E.: Can LLMs reliably simu-
late human learner actions? A simulation authoring framework for open-ended
learning environments. In: Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 39, pp. 29044–29052 (2025). AAAI Press
[28] Wu, X., Zhu, P., Zhang, J., Yin, M., Wang, Y.: ChatGPT’s impact on student
learning outcomes: a meta-analysis of 35 experimental studies. Humanities and
Social Sciences Communications13(684), 1–17 (2026)
[29] K¨ aser, T., Alexandron, G.: Simulated learners in educational technology: A sys-
tematic literature review and a Turing-like test. International Journal of Artificial
Intelligence in Education34(2), 545–585 (2024)
[30] VanLehn, K.: The relative effectiveness of human tutoring, intelligent tutoring
systems, and other tutoring systems. Educational psychologist46(4), 197–221
(2011)
[31] Villegas-Ch, W., Buenano-Fernandez, D., Navarro, A.M., Mera-Navarrete, A.:
Adaptive intelligent tutoring systems for STEM education: analysis of the
learning impact and effectiveness of personalized feedback. Smart Learning
Environments12(1), 1–31 (2025)
[32] Sweller, J., Van Merri¨ enboer, J.J., Paas, F.: Cognitive architecture and instruc-
tional design: 20 years later. Educational psychology review31(2), 261–292
(2019)
[33] Paas, F., Renkl, A., Sweller, J.: Cognitive load theory and instructional design:
Recent developments. Educational psychologist38(1), 1–4 (2003)
[34] Paas, F., Tuovinen, J.E., Tabbers, H., Van Gerven, P.W.: Cognitive Load Theory.
Routledge, Oxfordshire, UK (2016)
[35] Wood, D., Bruner, J.S., Ross, G.: The role of tutoring in problem solving. Journal
of child psychology and psychiatry17(2), 89–100 (1976)
[36] Vygotsky, L.S.: Mind in Society: The Development of Higher Psychological
Processes. Harvard University Press, Cambridge, MA (1978)
30

[37] Freeman, S., Eddy, S., McDonough, M., Smith, M., Okoroafor, N., Jordt, H.,
Wenderoth, M.P.: Active learning increases student performance in science, engi-
neering, and mathematics. Proceedings of the national academy of sciences
111(23), 8410–8415 (2014)
[38] Hariyanto, Kristianingsih, F.X.D., Maharani, R.: Artificial intelligence in adaptive
education: a systematic review of techniques for personalized learning. Discover
Education4(1), 458 (2025)
[39] Ryan, R.M., Deci, E.L.: Self-determination theory and the facilitation of intrinsic
motivation, social development, and well-being. American psychologist55(1), 68–
78 (2000)
[40] Peters, D., Calvo, R.A., Ryan, R.M.: Designing for motivation, engagement and
wellbeing in digital experience. Frontiers in psychology9, 1–15 (2018)
[41] Lee, J., Kim, D.: From awareness to empowerment: self-determination theory-
informed learning analytics dashboards to enhance student engagement in
asynchronous online courses. Journal of Computing in Higher Education37(3),
1078–1118 (2025)
[42] Bandura, A.: Self-efficacy: The exercise of control. Freeman, New York (1997)
[43] Zimmerman, B.J.: Becoming a self-regulated learner: An overview. Theory into
practice41(2), 64–70 (2002)
[44] Corbett, A.T., Anderson, J.R.: Knowledge tracing: Modeling the acquisition of
procedural knowledge. User modeling and user-adapted interaction4(4), 253–278
(1994)
[45] Bjork, E.L., Bjork, R.A.: Making things hard on yourself, but in a good way: Cre-
ating desirable difficulties to enhance learning. In: Gernsbacher, M.A., Pew, R.W.,
Hough, L.M., Pomerantz, J.R. (eds.) Psychology and the Real World: Essays
Illustrating Fundamental Contributions to Society, pp. 56–64. Worth Publishers,
New York, NY (2011)
[46] Kochmar, E., Vu, D.D., Belfer, R., Gupta, V., Serban, I.V., Pineau, J.: Automated
personalized feedback improves learning gains in an intelligent tutoring system.
In: International Conference on Artificial Intelligence in Education, pp. 140–146
(2020). Springer
[47] Pian, Y., Lu, Y., Huang, Y., Bittencourt, I.I.: A gamified solution to the cold-start
problem of intelligent tutoring system. In: International Conference on Artificial
Intelligence in Education, pp. 376–381 (2020). Springer
[48] Hannafin, M., Land, S., Oliver, K.: Open learning environments: Foundations,
methods, and models. In: Instructional-design Theories and Models, pp. 115–140.
31

Routledge, Oxfordshire, UK (2013)
[49] Yang, L., Sun, X., Li, H., Xu, R., Wei, X.: Difficulty aware programming
knowledge tracing via large language models. Scientific Reports15(1), 11475
(2025)
[50] Matsuda, N., Cohen, W.W., Sewall, J., Lacerda, G., Koedinger, K.R.: Evalu-
ating a simulated student using real students data for training and testing. In:
International Conference on User Modeling, pp. 107–116 (2007). Springer
[51] Lu, X., Wang, X.: Generative students: Using LLM-simulated student profiles
to support question item evaluation. In: Proceedings of the Eleventh ACM
Conference on Learning@Scale, pp. 16–27 (2024). ACM
[52] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z.,
Li, Z., Li, D., Xing, E., Zhang, H.: Judging LLM-as-a-Judge with MT-bench and
chatbot arena. Advances in neural information processing systems36, 46595–
46623 (2023)
[53] Braun, V., Clarke, V.: Using thematic analysis in psychology. Qualitative
Research in Psychology3(2), 77–101 (2006)
32